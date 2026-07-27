"""
src/ingestion/duckdb_schema.py
==============================
DuckDB database initialisation for AEGIS.

Creates the canonical flood.duckdb with all tables required for:
  - Spatial grid (site_grid)
  - ERA5 climate features (era5_daily)
  - Sentinel SAR + optical embeddings (sentinel_features)
  - Air quality / sensor data (openaq_daily)
  - Hydrological bulletin text and embeddings (bulletin_doc, bulletin_emb)
  - Subjective Logic opinion audit log (opinion_log)
  - Ground truth flood labels (truth)
  - Data provenance manifest (manifest)

All geometry columns are stored as WKT strings (DuckDB SPATIAL extension
is optional; operations on geometry use Python/GeoPandas instead).
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
-- ==============================================================
-- Site Spatial Grid
-- ==============================================================
CREATE TABLE IF NOT EXISTS site_grid (
    cell_id         UBIGINT PRIMARY KEY,
    bbox_wkt        VARCHAR NOT NULL,       -- WKT POLYGON of 100m cell
    centroid_lon    DOUBLE,
    centroid_lat    DOUBLE,
    land_cover      UTINYINT,               -- ESA WorldCover class
    slope_deg       REAL,
    impervious_frac REAL,                   -- fraction 0-1
    elevation_m     REAL
);

-- ==============================================================
-- ERA5 Daily Climate
-- ==============================================================
CREATE TABLE IF NOT EXISTS era5_daily (
    cell_id         UBIGINT  NOT NULL REFERENCES site_grid(cell_id),
    date            DATE     NOT NULL,
    tp_mm           REAL,                   -- total precipitation (mm)
    t2m_c           REAL,                   -- 2-m temperature (°C)
    ssrd_mj         REAL,                   -- surface solar radiation (MJ/m²)
    u10_ms          REAL,                   -- 10-m u-wind (m/s)
    v10_ms          REAL,                   -- 10-m v-wind (m/s)
    precip_7d_sum   REAL,                   -- 7-day rolling precipitation
    precip_30d_anom REAL,                   -- 30-day anomaly vs ERA5 clim.
    PRIMARY KEY (cell_id, date)
);

-- ==============================================================
-- Sentinel SAR + Optical Features
-- ==============================================================
CREATE TABLE IF NOT EXISTS sentinel_features (
    cell_id         UBIGINT  NOT NULL REFERENCES site_grid(cell_id),
    date            DATE     NOT NULL,
    sar_vv_db       REAL,                   -- Sentinel-1 VV backscatter (dB)
    sar_vh_db       REAL,                   -- Sentinel-1 VH backscatter (dB)
    water_index_sar REAL,                   -- (VV - VH) water index
    ndwi            REAL,                   -- S2 Normalised Difference Water Index
    ndvi            REAL,                   -- S2 Normalised Difference Vegetation Index
    cloud_mask_frac REAL,                   -- cloud fraction 0-1
    vit_embedding   FLOAT[768],             -- frozen ViT-Base/16 patch embedding
    source_s1       VARCHAR,                -- manifest_id for S1 tile
    source_s2       VARCHAR,                -- manifest_id for S2 tile
    PRIMARY KEY (cell_id, date)
);

-- ==============================================================
-- Air Quality / Sensor Data
-- ==============================================================
CREATE TABLE IF NOT EXISTS openaq_daily (
    cell_id         UBIGINT  NOT NULL REFERENCES site_grid(cell_id),
    date            DATE     NOT NULL,
    pm25_ug         REAL,
    no2_ppb         REAL,
    rh_pct          REAL,                   -- relative humidity proxy
    rain_gauge_mm   REAL,                   -- local gauge reading if available
    data_missing    BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (cell_id, date)
);

-- ==============================================================
-- Hydrological Bulletin Documents
-- ==============================================================
CREATE TABLE IF NOT EXISTS bulletin_doc (
    doc_id          VARCHAR  PRIMARY KEY,
    date            DATE     NOT NULL,
    source          VARCHAR  NOT NULL,      -- 'BOM' | 'USGS' | 'Copernicus'
    region          VARCHAR,
    text_body       VARCHAR  NOT NULL
);

CREATE TABLE IF NOT EXISTS bulletin_emb (
    doc_id          VARCHAR  PRIMARY KEY REFERENCES bulletin_doc(doc_id),
    embedding       FLOAT[384]              -- all-MiniLM-L6-v2 sentence embedding
);

-- ==============================================================
-- Ground Truth (EMSR / Global Flood Database)
-- ==============================================================
CREATE TABLE IF NOT EXISTS truth (
    cell_id         UBIGINT  NOT NULL REFERENCES site_grid(cell_id),
    date            DATE     NOT NULL,
    flood_state     UTINYINT NOT NULL,      -- 0=Dry 1=Saturated 2=SurfaceFlow 3=Inundation
    flood_depth_m   REAL,                   -- water depth estimate if available
    source          VARCHAR,                -- 'EMSR' | 'GFD' | 'manual'
    confidence      REAL DEFAULT 1.0,
    PRIMARY KEY (cell_id, date)
);

-- ==============================================================
-- Data Provenance Manifest
-- ==============================================================
CREATE TABLE IF NOT EXISTS manifest (
    manifest_id     VARCHAR  PRIMARY KEY,   -- UUID
    ingested_at     TIMESTAMPTZ NOT NULL,
    modality        VARCHAR  NOT NULL,      -- 'ERA5' | 'S1' | 'S2' | 'WorldCover' etc.
    source_url      VARCHAR,
    collection      VARCHAR,
    bbox_wkt        VARCHAR,
    date_range_start DATE,
    date_range_end  DATE,
    file_hash_sha256 VARCHAR,
    record_count    INTEGER
);

-- ==============================================================
-- Subjective Logic Opinion Audit Log
-- ==============================================================
CREATE TABLE IF NOT EXISTS opinion_log (
    log_id          VARCHAR  NOT NULL,      -- UUID for this log entry
    run_id          VARCHAR  NOT NULL,      -- experiment run identifier
    cell_id         UBIGINT  NOT NULL,
    date            DATE     NOT NULL,
    agent           VARCHAR  NOT NULL,
    stage           VARCHAR  NOT NULL,      -- 'emit' | 'partial_obs' | 'fused'
    fusion_op       VARCHAR,                -- 'WBF' | 'CCF' | NULL
    b_dry           DOUBLE,
    b_saturated     DOUBLE,
    b_surfaceflow   DOUBLE,
    b_inundation    DOUBLE,
    uncertainty_u   DOUBLE,
    credibility_gamma DOUBLE,
    modality_missing BOOLEAN DEFAULT FALSE,
    manifest_id     VARCHAR,                -- links to manifest
    model_ckpt      VARCHAR,
    logged_at       TIMESTAMPTZ
);
"""


def initialise_db(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """
    Connect to (or create) the AEGIS DuckDB database and apply schema.

    Args:
        db_path : Path to .duckdb file. Created if it does not exist.

    Returns:
        DuckDB connection (thread-local safe in read-write mode).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute(SCHEMA_SQL)
    conn.commit()
    logger.info("AEGIS database initialised at %s", db_path)
    return conn


def get_conn(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Return a read-write connection to an existing AEGIS database."""
    return duckdb.connect(str(db_path))
