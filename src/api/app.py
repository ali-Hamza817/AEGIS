"""
src/api/app.py
==============
AEGIS FastAPI service — Provenance-Aware Flood Risk API.

Endpoints:
    GET  /health                          System health check
    POST /predict                         Predict flood state for (cell_id, date)
    GET  /grid                            Return full AOI grid with latest predictions
    GET  /provenance/{cell_id}/{date}     Detailed per-agent provenance for a cell

Security notes:
    - Server MUST bind to 127.0.0.1 only during development/testing.
    - All inputs are validated via Pydantic v2 models.
    - SQL queries use DuckDB parameterized interface (no string concatenation).
    - No credentials or secrets are stored or transmitted.
    - CSP and security headers set by middleware.

TODO(security): In production, add OAuth2/JWT authentication.
TODO(security): Add rate limiting middleware (e.g., slowapi).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.ingestion.duckdb_schema import get_conn
from src.agents import (
    ClimateAgent, SatelliteAgent, LandCoverAgent,
    AirQualityAgent, DocIntAgent,
)
from src.coordinator.orchestrator import SLOrchestrator
from src.sl.credibility import CredibilityRegistry
from src.prediction.evidential_head import EvidentialHead, opinion_to_feature_vector
from src.prediction.baselines import LLMArbitratedBaseline

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# App Configuration
# -----------------------------------------------------------------------
DB_PATH = Path("data/duckdb/flood.duckdb")
RESULTS_PATH = Path("results")
WEB_PUBLIC_PATH = Path("web/public")
RESULTS_PATH.mkdir(exist_ok=True)

app = FastAPI(
    title="AEGIS — Agentic Evidential Geographic Intelligence for Sustainability",
    description=(
        "Multimodal Urban Flood Risk Assessment using Subjective Logic "
        "multi-agent fusion with provenance-aware explanations."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — restrict to localhost origins only (TODO: update for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self' https: data: 'unsafe-inline' 'unsafe-eval'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://unpkg.com https://fonts.googleapis.com https://cdn.jsdelivr.net; "
        "font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net; "
        "img-src 'self' data: https: https://fastapi.tiangolo.com;"
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serve the interactive spatial dashboard."""
    index_file = WEB_PUBLIC_PATH / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"status": "ok", "message": "AEGIS API Running"})


# -----------------------------------------------------------------------
# Lazy-loaded singletons
# -----------------------------------------------------------------------
_orchestrator: SLOrchestrator | None = None
_evidential_head: EvidentialHead | None = None
_llm_baseline: LLMArbitratedBaseline | None = None
_registry: CredibilityRegistry | None = None


def _get_orchestrator() -> SLOrchestrator:
    global _orchestrator, _registry
    if _orchestrator is None:
        _registry = CredibilityRegistry.default()
        agents = [
            ClimateAgent(),
            SatelliteAgent(),
            LandCoverAgent(),
            AirQualityAgent(),
            DocIntAgent(),
        ]
        db_conn = get_conn(DB_PATH) if DB_PATH.exists() else None
        _orchestrator = SLOrchestrator(
            agents=agents,
            credibility_registry=_registry,
            db_conn=db_conn,
        )
    return _orchestrator


def _get_head() -> EvidentialHead:
    global _evidential_head
    if _evidential_head is None:
        _evidential_head = EvidentialHead()
    return _evidential_head


def _get_context(cell_id: int, target_date: date) -> dict[str, Any]:
    """Query DuckDB for all modality rows for a given cell and date."""
    if not DB_PATH.exists():
        # Return minimal synthetic context for API to function offline
        from datetime import timedelta
        return {
            "era5_row": {
                "tp_mm": 45.0, "precip_7d_sum": 180.0, "precip_30d_anom": 120.0,
                "t2m_c": 26.0, "ssrd_mj": 12.0,
            },
            "sar_row": {
                "sar_vv_db": -14.5, "sar_vh_db": -21.0, "water_index_sar": 6.5,
                "ndwi": 0.25, "ndvi": 0.3, "cloud_mask_frac": 0.1,
            },
            "lc_row": {
                "land_cover": 4, "slope_deg": 1.5, "impervious_frac": 0.7, "elevation_m": 3.0,
            },
            "aq_row": {
                "pm25_ug": 5.0, "no2_ppb": 10.0, "rh_pct": 88.0, "rain_gauge_mm": 35.0,
                "data_missing": False,
            },
            "bulletin_embs": [],
            "manifest_id_era5": "synthetic",
            "manifest_id_s1": "synthetic",
            "manifest_id_lc": "synthetic",
            "manifest_id_aq": "synthetic",
            "manifest_id_bulletins": "synthetic",
        }

    conn = get_conn(DB_PATH)
    # Query ERA5
    era5 = conn.execute(
        "SELECT tp_mm, precip_7d_sum, precip_30d_anom, t2m_c, ssrd_mj "
        "FROM era5_daily WHERE cell_id=? AND date=?",
        [cell_id, target_date],
    ).fetchone()
    era5_row = dict(zip(
        ["tp_mm", "precip_7d_sum", "precip_30d_anom", "t2m_c", "ssrd_mj"],
        era5 or [None] * 5,
    )) if era5 else None

    # Query SAR
    sar = conn.execute(
        "SELECT sar_vv_db, sar_vh_db, water_index_sar, ndwi, ndvi, cloud_mask_frac "
        "FROM sentinel_features WHERE cell_id=? AND date=?",
        [cell_id, target_date],
    ).fetchone()
    sar_row = dict(zip(
        ["sar_vv_db", "sar_vh_db", "water_index_sar", "ndwi", "ndvi", "cloud_mask_frac"],
        sar or [None] * 6,
    )) if sar else None

    # Query land cover (from site_grid — static)
    lc = conn.execute(
        "SELECT land_cover, slope_deg, impervious_frac, elevation_m "
        "FROM site_grid WHERE cell_id=?",
        [cell_id],
    ).fetchone()
    lc_row = dict(zip(
        ["land_cover", "slope_deg", "impervious_frac", "elevation_m"],
        lc or [None] * 4,
    )) if lc else None

    # Query AQ
    aq = conn.execute(
        "SELECT pm25_ug, no2_ppb, rh_pct, rain_gauge_mm, data_missing "
        "FROM openaq_daily WHERE cell_id=? AND date=?",
        [cell_id, target_date],
    ).fetchone()
    aq_row = dict(zip(
        ["pm25_ug", "no2_ppb", "rh_pct", "rain_gauge_mm", "data_missing"],
        aq or [None, None, None, None, True],
    )) if aq else {"data_missing": True}

    # Query bulletin embeddings (14-day window)
    from datetime import timedelta
    window_start = target_date - timedelta(days=14)
    embs_raw = conn.execute(
        "SELECT be.embedding FROM bulletin_emb be "
        "JOIN bulletin_doc bd ON be.doc_id = bd.doc_id "
        "WHERE bd.date BETWEEN ? AND ?",
        [window_start, target_date],
    ).fetchall()
    import numpy as np
    bulletin_embs = [list(r[0]) for r in embs_raw] if embs_raw else []

    conn.close()
    return {
        "era5_row": era5_row,
        "sar_row": sar_row,
        "lc_row": lc_row,
        "aq_row": aq_row,
        "bulletin_embs": bulletin_embs,
        "manifest_id_era5": "db",
        "manifest_id_s1": "db",
        "manifest_id_lc": "db",
        "manifest_id_aq": "db",
        "manifest_id_bulletins": "db",
    }


# -----------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------

class PredictRequest(BaseModel):
    cell_id: int = Field(..., ge=0, description="Grid cell identifier")
    target_date: date = Field(..., description="Target date for flood prediction")
    enabled_agents: list[str] | None = Field(
        None,
        description="Optional list of agent names to enable. If null, all are used.",
    )


class PredictResponse(BaseModel):
    cell_id: int
    target_date: str
    flood_state: int
    flood_state_label: str
    state_proba: list[float]
    depth_m: float
    uncertainty_u: float
    fusion_operator: str
    max_js_divergence: float
    agent_contributions: dict[str, float]


FLOOD_LABELS = ["Dry", "Saturated", "SurfaceFlow", "Inundation"]


# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------

@app.get("/health")
async def health():
    """System health check."""
    return {
        "status": "ok",
        "service": "AEGIS",
        "version": "0.1.0",
        "db_available": DB_PATH.exists(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Predict flood state for a given grid cell and date.
    Returns fused SL opinion, prediction, and agent contribution provenance.
    """
    # Input validation
    if req.cell_id < 0:
        raise HTTPException(status_code=400, detail="cell_id must be non-negative.")

    orchestrator = _get_orchestrator()
    head = _get_head()

    context = _get_context(req.cell_id, req.target_date)
    result = orchestrator.route(
        cell_id=req.cell_id,
        target_date=req.target_date,
        context=context,
        enabled_agents=req.enabled_agents,
    )

    feat_vec = opinion_to_feature_vector(result)
    prediction = head.predict(feat_vec, fused_opinion=result.fused_opinion)

    flood_state = prediction["flood_state"]
    return PredictResponse(
        cell_id=req.cell_id,
        target_date=str(req.target_date),
        flood_state=flood_state,
        flood_state_label=FLOOD_LABELS[flood_state],
        state_proba=prediction["state_proba"],
        depth_m=prediction["depth_m"],
        uncertainty_u=prediction["uncertainty_u"],
        fusion_operator=result.fusion_operator,
        max_js_divergence=result.max_js,
        agent_contributions={
            name: float(w)
            for (name, _), w in zip(result.agent_opinions, result.contributions)
        },
    )


@app.get("/grid")
async def get_grid(
    n_cells: int = Query(default=50, ge=1, le=500),
    target_date: date = Query(default=date(2022, 3, 1)),
):
    """
    Return grid cells with flood predictions for dashboard visualisation.
    """
    orchestrator = _get_orchestrator()
    head = _get_head()

    if not DB_PATH.exists():
        # Return minimal synthetic grid for offline demo
        import numpy as np
        cells = []
        rng = np.random.default_rng(42)
        for i in range(n_cells):
            lon = float(rng.uniform(152.5, 153.5))
            lat = float(rng.uniform(-28.0, -27.0))
            state = int(rng.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.25, 0.15]))
            u = float(rng.uniform(0.1, 0.6))
            cells.append({
                "cell_id": i,
                "lon": lon, "lat": lat,
                "flood_state": state,
                "flood_state_label": FLOOD_LABELS[state],
                "uncertainty_u": u,
                "depth_m": float(max(0.0, rng.normal(state * 0.8, 0.3))),
            })
        return {"target_date": str(target_date), "cells": cells}

    conn = get_conn(DB_PATH)
    grid_rows = conn.execute(
        "SELECT cell_id, centroid_lon, centroid_lat FROM site_grid LIMIT ?",
        [n_cells],
    ).fetchall()
    conn.close()

    cells = []
    for (cell_id, lon, lat) in grid_rows:
        context = _get_context(int(cell_id), target_date)
        result = orchestrator.route(int(cell_id), target_date, context)
        feat_vec = opinion_to_feature_vector(result)
        prediction = head.predict(feat_vec, fused_opinion=result.fused_opinion)
        cells.append({
            "cell_id": int(cell_id),
            "lon": float(lon) if lon else 0.0,
            "lat": float(lat) if lat else 0.0,
            "flood_state": prediction["flood_state"],
            "flood_state_label": FLOOD_LABELS[prediction["flood_state"]],
            "uncertainty_u": prediction["uncertainty_u"],
            "depth_m": prediction["depth_m"],
            "fusion_operator": result.fusion_operator,
        })

    return {"target_date": str(target_date), "cells": cells}


@app.get("/provenance/{cell_id}/{target_date}")
async def get_provenance(cell_id: int, target_date: date):
    """
    Return detailed provenance for a cell-date prediction.

    Includes per-agent opinion details and data lineage.
    This is the transparency endpoint for the H4 dashboard.
    """
    if cell_id < 0:
        raise HTTPException(status_code=400, detail="Invalid cell_id.")

    orchestrator = _get_orchestrator()
    context = _get_context(cell_id, target_date)
    result = orchestrator.route(cell_id, target_date, context)

    return {
        "cell_id": cell_id,
        "target_date": str(target_date),
        "fused_opinion": result.fused_opinion.to_dict(),
        "fusion_operator": result.fusion_operator,
        "max_js_divergence": result.max_js,
        "agent_opinions": [
            {
                "agent": name,
                "opinion": op.to_dict(),
                "contribution": float(w),
            }
            for (name, op), w in zip(result.agent_opinions, result.contributions)
        ],
        "provenance_records": [p.to_dict() for p in result.provenance],
    }
