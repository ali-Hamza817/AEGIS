"""
src/ingestion/synthetic_generator.py
=====================================
Physically-motivated synthetic data generator for AEGIS.

Generates a complete study-site dataset WITHOUT requiring internet access
or CDS/Planetary Computer credentials. The distributions are grounded in:

  - ERA5 SEQ Brisbane Feb–Mar 2022 flood statistics
    (BoM: ~1000 mm in February 2022, extreme precip event)
  - Sentinel-1 SAR flood backscatter statistics
    (Twele et al. 2016, ISPRS: VV flood ~-15 dB, dry ~-8 dB)
  - ESA WorldCover Brisbane land cover proportions

This module is used to:
  1. Bootstrap the pipeline for offline development.
  2. Generate controlled ablation datasets (e.g., systematic sensor dropout).
  3. Provide a reproducible baseline when ground-truth labels are needed.

All generated data is stored in DuckDB tables with proper manifest records.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from .duckdb_schema import initialise_db

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Brisbane SEQ 2022 flood event physical parameters
# -----------------------------------------------------------------------
BRISBANE_PARAMS = {
    # ERA5 precipitation statistics (mm/day)
    "precip_dry_mu": 2.0,
    "precip_dry_sigma": 3.0,
    "precip_event_mu": 35.0,
    "precip_event_sigma": 20.0,
    "event_peak_day": 9,           # day offset from event_start_date
    # Sentinel-1 backscatter (dB)
    "sar_vv_dry": -8.0,
    "sar_vv_flood": -15.5,
    "sar_vh_dry": -15.0,
    "sar_vh_flood": -22.0,
    "sar_noise_std": 1.5,
    # Land cover proportions (Brisbane urban mix, WorldCover classes 0-5)
    "land_cover_probs": [0.05, 0.15, 0.10, 0.10, 0.50, 0.10],
    # Slope (deg) for low-lying urban area
    "slope_mu": 2.5,
    "slope_sigma": 2.0,
    # Impervious fraction
    "impervious_mu": 0.55,
    "impervious_sigma": 0.20,
}


class SyntheticGenerator:
    """
    Generates physically-grounded synthetic flood data for a 2D grid of
    spatial cells over a specified date range.
    """

    def __init__(
        self,
        db_path: str | Path,
        n_cells: int = 500,
        start_date: date = date(2022, 2, 20),
        end_date: date = date(2022, 3, 15),
        event_start_day: int = 3,       # days from start_date when flood begins
        seed: int = 42,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.n_cells = n_cells
        self.start_date = start_date
        self.end_date = end_date
        self.event_start_day = event_start_day
        self.rng = np.random.default_rng(seed)
        self.params = {**BRISBANE_PARAMS, **(params or {})}
        self.dates: list[date] = []
        d = start_date
        while d <= end_date:
            self.dates.append(d)
            d += timedelta(days=1)
        self.n_dates = len(self.dates)
        self.conn: duckdb.DuckDBPyConnection | None = None

    def run(self) -> None:
        """Generate all synthetic tables and populate DuckDB."""
        self.conn = initialise_db(self.db_path)
        run_id = str(uuid.uuid4())[:8]
        logger.info(
            "SyntheticGenerator: %d cells x %d dates (run_id=%s)",
            self.n_cells,
            self.n_dates,
            run_id,
        )
        self._generate_site_grid()
        self._generate_era5()
        self._generate_sentinel()
        self._generate_truth()
        self._generate_bulletins()
        self._generate_openaq()
        self.conn.commit()
        logger.info("Synthetic generation complete. DB: %s", self.db_path)

    # ------------------------------------------------------------------
    # Site Grid
    # ------------------------------------------------------------------

    def _generate_site_grid(self) -> None:
        lons = self.rng.uniform(152.5, 153.5, self.n_cells)
        lats = self.rng.uniform(-28.0, -27.0, self.n_cells)
        land_cover = self.rng.choice(
            6,
            size=self.n_cells,
            p=self.params["land_cover_probs"],
        )
        slopes = self.rng.normal(
            self.params["slope_mu"], self.params["slope_sigma"], self.n_cells
        ).clip(0.0, 30.0)
        impervious = self.rng.normal(
            self.params["impervious_mu"], self.params["impervious_sigma"], self.n_cells
        ).clip(0.0, 1.0)
        # impervious higher for built-up class
        impervious[land_cover == 4] = np.clip(
            impervious[land_cover == 4] + 0.25, 0.0, 1.0
        )
        elevations = self.rng.exponential(5.0, self.n_cells)

        rows = []
        for i in range(self.n_cells):
            d = 0.0009   # ~100 m in degrees
            bbox = (
                f"POLYGON(({lons[i]:.6f} {lats[i]:.6f}, "
                f"{lons[i]+d:.6f} {lats[i]:.6f}, "
                f"{lons[i]+d:.6f} {lats[i]+d:.6f}, "
                f"{lons[i]:.6f} {lats[i]+d:.6f}, "
                f"{lons[i]:.6f} {lats[i]:.6f}))"
            )
            rows.append((
                i,
                bbox,
                float(lons[i] + d / 2),
                float(lats[i] + d / 2),
                int(land_cover[i]),
                float(slopes[i]),
                float(impervious[i]),
                float(elevations[i]),
            ))

        self.conn.executemany(
            """INSERT OR REPLACE INTO site_grid
               (cell_id, bbox_wkt, centroid_lon, centroid_lat,
                land_cover, slope_deg, impervious_frac, elevation_m)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        logger.debug("site_grid: %d rows inserted.", len(rows))
        self._register_manifest("SyntheticGrid", "generated", len(rows))

    # ------------------------------------------------------------------
    # ERA5 Daily
    # ------------------------------------------------------------------

    def _generate_era5(self) -> None:
        p = self.params
        rows = []
        for cell_id in range(self.n_cells):
            precip_7d = 0.0
            running_30d: list[float] = []
            for di, dt in enumerate(self.dates):
                # Flood intensity: Gaussian peak around event peak
                event_intensity = max(
                    0.0,
                    np.exp(-0.5 * ((di - self.event_start_day - p["event_peak_day"]) / 4) ** 2)
                )
                precip = float(np.clip(
                    self.rng.normal(
                        p["precip_dry_mu"] + event_intensity * p["precip_event_mu"],
                        p["precip_dry_sigma"] + event_intensity * p["precip_event_sigma"] * 0.5,
                    ), 0.0, None))

                running_30d.append(precip)
                if len(running_30d) > 30:
                    running_30d.pop(0)
                precip_7d = float(np.sum(running_30d[-7:]))
                precip_30d_anom = float(precip - np.mean(running_30d))
                t2m = float(self.rng.normal(27.0 - event_intensity * 2, 2.0))
                ssrd = float(np.clip(self.rng.normal(18.0 - event_intensity * 5, 3.0), 0.0, None))
                u10 = float(self.rng.normal(3.0, 1.0))
                v10 = float(self.rng.normal(2.0, 1.0))
                rows.append((
                    cell_id, dt,
                    precip, t2m, ssrd, u10, v10,
                    precip_7d, precip_30d_anom,
                ))

        self.conn.executemany(
            """INSERT OR REPLACE INTO era5_daily
               (cell_id, date, tp_mm, t2m_c, ssrd_mj, u10_ms, v10_ms,
                precip_7d_sum, precip_30d_anom)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        logger.debug("era5_daily: %d rows inserted.", len(rows))
        self._register_manifest("ERA5", "synthetic", len(rows))

    # ------------------------------------------------------------------
    # Sentinel SAR + Optical
    # ------------------------------------------------------------------

    def _generate_sentinel(self) -> None:
        p = self.params
        # Fetch slope and impervious for each cell to modulate flood likelihood
        grid = self.conn.execute(
            "SELECT cell_id, slope_deg, impervious_frac, elevation_m FROM site_grid ORDER BY cell_id"
        ).fetchall()
        cell_props = {row[0]: row[1:] for row in grid}

        rows = []
        for cell_id in range(self.n_cells):
            slope, imp, elev = cell_props.get(cell_id, (2.5, 0.5, 3.0))
            flood_prone = float(imp > 0.5 and slope < 3.0 and elev < 5.0)

            for di, dt in enumerate(self.dates):
                event_intensity = max(
                    0.0,
                    np.exp(-0.5 * ((di - self.event_start_day - p["event_peak_day"]) / 4) ** 2)
                )
                flood_factor = event_intensity * (0.5 + 0.5 * flood_prone)

                vv = float(self.rng.normal(
                    p["sar_vv_dry"] + flood_factor * (p["sar_vv_flood"] - p["sar_vv_dry"]),
                    p["sar_noise_std"],
                ))
                vh = float(self.rng.normal(
                    p["sar_vh_dry"] + flood_factor * (p["sar_vh_flood"] - p["sar_vh_dry"]),
                    p["sar_noise_std"],
                ))
                water_idx = vv - vh
                ndwi = float(np.clip(self.rng.normal(-0.2 + flood_factor * 0.5, 0.1), -1, 1))
                ndvi = float(np.clip(self.rng.normal(0.4 - flood_factor * 0.3, 0.1), -1, 1))
                cloud_frac = float(self.rng.beta(2, 5) * event_intensity * 0.8)

                # Synthetic ViT embedding: 768-d Gaussian (frozen model proxy)
                vit_emb = self.rng.normal(0.0, 0.1, 768).astype(np.float32)
                # Modulate embedding slightly toward flood signal
                vit_emb[:64] += flood_factor * 0.3
                vit_emb_list = vit_emb.tolist()

                rows.append((
                    cell_id, dt,
                    vv, vh, water_idx, ndwi, ndvi, cloud_frac,
                    vit_emb_list, "S1-synthetic", "S2-synthetic",
                ))

        self.conn.executemany(
            """INSERT OR REPLACE INTO sentinel_features
               (cell_id, date, sar_vv_db, sar_vh_db, water_index_sar,
                ndwi, ndvi, cloud_mask_frac, vit_embedding, source_s1, source_s2)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        logger.debug("sentinel_features: %d rows inserted.", len(rows))
        self._register_manifest("Sentinel-1/2", "synthetic", len(rows))

    # ------------------------------------------------------------------
    # Ground Truth (EMSR-aligned labels)
    # ------------------------------------------------------------------

    def _generate_truth(self) -> None:
        grid = self.conn.execute(
            "SELECT cell_id, slope_deg, impervious_frac, elevation_m FROM site_grid ORDER BY cell_id"
        ).fetchall()
        cell_props = {row[0]: row[1:] for row in grid}
        p = self.params

        rows = []
        for cell_id in range(self.n_cells):
            slope, imp, elev = cell_props.get(cell_id, (2.5, 0.5, 3.0))
            flood_prone_score = (imp - 0.3) * 2 + (3.0 - min(slope, 3.0)) / 3.0 + (5.0 - min(elev, 5.0)) / 5.0

            for di, dt in enumerate(self.dates):
                event_intensity = max(
                    0.0,
                    np.exp(-0.5 * ((di - self.event_start_day - p["event_peak_day"]) / 4) ** 2)
                )
                # State probabilities driven by flood_prone_score and event intensity
                raw_score = event_intensity * max(0.0, flood_prone_score)
                raw_score = np.clip(raw_score, 0.0, 1.0)

                # Transition: 0=Dry, 1=Saturated, 2=SurfaceFlow, 3=Inundation
                if raw_score < 0.15:
                    state = 0
                elif raw_score < 0.40:
                    state = 1
                elif raw_score < 0.70:
                    state = 2
                else:
                    state = 3

                # Add stochastic noise
                if self.rng.random() < 0.05:
                    state = max(0, state - 1)

                depth = float(max(0.0, self.rng.normal(raw_score * 2.5, 0.3))) if state >= 2 else 0.0
                rows.append((cell_id, dt, state, depth, "EMSR-synthetic", 0.9))

        self.conn.executemany(
            """INSERT OR REPLACE INTO truth
               (cell_id, date, flood_state, flood_depth_m, source, confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        logger.debug("truth: %d rows inserted.", len(rows))
        self._register_manifest("EMSR-truth", "synthetic", len(rows))

    # ------------------------------------------------------------------
    # Bulletins
    # ------------------------------------------------------------------

    def _generate_bulletins(self) -> None:
        templates = [
            "Severe flood warning issued for Brisbane River catchment. "
            "Water levels rising rapidly at {location} gauge. Residents advised to evacuate low-lying areas.",
            "Rainfall totals of {precip:.0f} mm recorded in 24 hours at {location}. "
            "Creeks and streams running well above normal levels.",
            "Flash flood watch in effect for South East Queensland. "
            "Thunderstorm activity expected to produce heavy rainfall bursts exceeding 50 mm/hr.",
            "Bureau of Meteorology confirms La Nina conditions contributing to above-average precipitation. "
            "Flood risk elevated for entire SEQ region through March.",
            "River gauge at {location} reached {level:.1f} m, exceeding the major flood level of 3.5 m. "
            "Downstream communities on high alert.",
        ]
        locations = ["Jindalee", "Rocklea", "Ipswich", "Colleges Crossing", "Lowood", "Savages Crossing"]
        doc_rows = []
        emb_rows = []
        for di, dt in enumerate(self.dates):
            n_docs = self.rng.integers(0, 2)
            for _ in range(n_docs):
                doc_id = str(uuid.uuid4())
                template = self.rng.choice(templates)
                precip = self.rng.uniform(50, 300)
                loc = self.rng.choice(locations)
                level = self.rng.uniform(2.0, 5.5)
                text = template.format(precip=precip, location=loc, level=level)
                source = self.rng.choice(["BOM", "USGS", "Copernicus"])
                doc_rows.append((doc_id, dt, source, "Brisbane", text))
                # Synthetic sentence embedding (384-d all-MiniLM-L6-v2 proxy)
                emb = self.rng.normal(0, 0.1, 384).astype(np.float32)
                emb[:32] += 0.3   # flood-topic signal
                emb_rows.append((doc_id, emb.tolist()))

        self.conn.executemany(
            "INSERT OR REPLACE INTO bulletin_doc (doc_id, date, source, region, text_body) VALUES (?, ?, ?, ?, ?)",
            doc_rows,
        )
        self.conn.executemany(
            "INSERT OR REPLACE INTO bulletin_emb (doc_id, embedding) VALUES (?, ?)",
            emb_rows,
        )
        logger.debug("bulletins: %d documents inserted.", len(doc_rows))
        self._register_manifest("Bulletins", "synthetic", len(doc_rows))

    # ------------------------------------------------------------------
    # OpenAQ / Sensors
    # ------------------------------------------------------------------

    def _generate_openaq(self) -> None:
        rows = []
        for cell_id in range(self.n_cells):
            for di, dt in enumerate(self.dates):
                event_intensity = max(
                    0.0,
                    np.exp(-0.5 * ((di - self.event_start_day - self.params["event_peak_day"]) / 4) ** 2)
                )
                missing = bool(self.rng.random() < 0.15)  # 15% sensor dropout
                if missing:
                    rows.append((cell_id, dt, None, None, None, None, True))
                else:
                    pm25 = float(np.clip(self.rng.normal(8.0 - event_intensity * 3, 2.0), 0.0, None))
                    no2 = float(np.clip(self.rng.normal(12.0, 3.0), 0.0, None))
                    rh = float(np.clip(self.rng.normal(65.0 + event_intensity * 20, 5.0), 0.0, 100.0))
                    gauge = float(np.clip(
                        self.rng.normal(
                            self.params["precip_dry_mu"] + event_intensity * self.params["precip_event_mu"], 5.0
                        ), 0.0, None))
                    rows.append((cell_id, dt, pm25, no2, rh, gauge, False))

        self.conn.executemany(
            """INSERT OR REPLACE INTO openaq_daily
               (cell_id, date, pm25_ug, no2_ppb, rh_pct, rain_gauge_mm, data_missing)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        logger.debug("openaq_daily: %d rows inserted.", len(rows))
        self._register_manifest("OpenAQ", "synthetic", len(rows))

    # ------------------------------------------------------------------
    # Manifest helper
    # ------------------------------------------------------------------

    def _register_manifest(self, modality: str, collection: str, count: int) -> str:
        mid = str(uuid.uuid4())
        content = f"{modality}-{collection}-{count}"
        file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        self.conn.execute(
            """INSERT OR REPLACE INTO manifest
               (manifest_id, ingested_at, modality, source_url, collection,
                bbox_wkt, date_range_start, date_range_end, file_hash_sha256, record_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mid,
                datetime.now(tz=timezone.utc).isoformat(),
                modality,
                "synthetic://generated",
                collection,
                "POLYGON((152.5 -28.0, 153.5 -28.0, 153.5 -27.0, 152.5 -27.0, 152.5 -28.0))",
                self.start_date,
                self.end_date,
                file_hash,
                count,
            ),
        )
        return mid
