"""
tests/test_pipeline.py
=======================
Integration tests for the full AEGIS pipeline.

Tests:
  - DuckDB schema creation and data insertion.
  - SyntheticGenerator produces valid data.
  - Agent emit() with and without data.
  - Orchestrator routes correctly and returns valid results.
  - EvidentialHead predicts valid outputs.
  - LLMArbitratedBaseline returns valid structure.
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from src.ingestion.synthetic_generator import SyntheticGenerator
from src.ingestion.duckdb_schema import get_conn
from src.agents import ClimateAgent, SatelliteAgent, LandCoverAgent, AirQualityAgent, DocIntAgent
from src.coordinator.orchestrator import SLOrchestrator
from src.sl.credibility import CredibilityRegistry
from src.prediction.evidential_head import EvidentialHead, opinion_to_feature_vector
from src.prediction.baselines import LLMArbitratedBaseline


@pytest.fixture(scope="module")
def temp_db():
    """Create a small synthetic database for integration tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.duckdb"
        gen = SyntheticGenerator(
            db_path=db_path,
            n_cells=20,
            seed=0,
            start_date=date(2022, 2, 20),
            end_date=date(2022, 3, 1),
        )
        gen.run()
        yield db_path


@pytest.fixture(scope="module")
def orchestrator(temp_db):
    conn = get_conn(temp_db)
    agents = [ClimateAgent(), SatelliteAgent(), LandCoverAgent(), AirQualityAgent(), DocIntAgent()]
    reg = CredibilityRegistry.default()
    return SLOrchestrator(agents=agents, credibility_registry=reg, db_conn=conn)


class TestSyntheticGenerator:
    def test_tables_exist(self, temp_db):
        conn = get_conn(temp_db)
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        for tbl in ["site_grid", "era5_daily", "sentinel_features", "truth", "opinion_log"]:
            assert tbl in tables

    def test_row_counts(self, temp_db):
        conn = get_conn(temp_db)
        n_cells = conn.execute("SELECT COUNT(*) FROM site_grid").fetchone()[0]
        assert n_cells == 20
        n_era5 = conn.execute("SELECT COUNT(*) FROM era5_daily").fetchone()[0]
        assert n_era5 > 0
        n_truth = conn.execute("SELECT COUNT(*) FROM truth").fetchone()[0]
        assert n_truth > 0

    def test_truth_states_valid(self, temp_db):
        conn = get_conn(temp_db)
        states = [r[0] for r in conn.execute("SELECT DISTINCT flood_state FROM truth").fetchall()]
        for s in states:
            assert 0 <= s <= 3


class TestAgents:
    def _make_context(self):
        return {
            "era5_row": {"tp_mm": 40.0, "precip_7d_sum": 200.0, "precip_30d_anom": 100.0, "t2m_c": 25.0, "ssrd_mj": 12.0},
            "sar_row": {"sar_vv_db": -15.0, "sar_vh_db": -22.0, "water_index_sar": 7.0, "ndwi": 0.3, "ndvi": 0.2, "cloud_mask_frac": 0.1},
            "lc_row": {"land_cover": 4, "slope_deg": 1.5, "impervious_frac": 0.75, "elevation_m": 2.0},
            "aq_row": {"pm25_ug": 4.0, "no2_ppb": 10.0, "rh_pct": 90.0, "rain_gauge_mm": 40.0, "data_missing": False},
            "bulletin_embs": [np.random.randn(384).tolist()],
            "manifest_id_era5": "test", "manifest_id_s1": "test",
            "manifest_id_lc": "test", "manifest_id_aq": "test",
            "manifest_id_bulletins": "test",
        }

    def test_climate_agent_emit(self):
        ctx = self._make_context()
        agent = ClimateAgent()
        op, prov = agent.emit(0, date(2022, 3, 1), ctx)
        assert op.is_valid()
        assert abs(op.b.sum() + op.u - 1.0) < 1e-5
        assert not prov.modality_missing

    def test_satellite_agent_emit(self):
        ctx = self._make_context()
        agent = SatelliteAgent()
        op, prov = agent.emit(0, date(2022, 3, 1), ctx)
        assert op.is_valid()

    def test_landcover_agent_emit(self):
        ctx = self._make_context()
        agent = LandCoverAgent()
        op, prov = agent.emit(0, date(2022, 3, 1), ctx)
        assert op.is_valid()

    def test_missing_modality_returns_vacuous(self):
        # Missing ERA5 row -> partial-obs opinion with high uncertainty
        ctx = {"aq_row": {"data_missing": True}}
        agent = ClimateAgent()
        op, prov = agent.emit(0, date(2022, 3, 1), ctx)
        assert op.is_valid()
        assert prov.modality_missing
        assert op.u > 0.5

    def test_docint_no_bulletins(self):
        ctx = {"bulletin_embs": []}
        agent = DocIntAgent()
        op, prov = agent.emit(0, date(2022, 3, 1), ctx)
        assert op.is_valid()
        assert prov.modality_missing  # no bulletins -> missing


class TestOrchestrator:
    def test_route_returns_valid_opinion(self, orchestrator):
        ctx = {
            "era5_row": {"tp_mm": 40.0, "precip_7d_sum": 200.0, "precip_30d_anom": 100.0, "t2m_c": 25.0, "ssrd_mj": 12.0},
            "sar_row": {"sar_vv_db": -14.0, "sar_vh_db": -20.0, "water_index_sar": 6.0, "ndwi": 0.2, "ndvi": 0.3, "cloud_mask_frac": 0.0},
            "lc_row": {"land_cover": 4, "slope_deg": 2.0, "impervious_frac": 0.7, "elevation_m": 3.0},
            "aq_row": {"pm25_ug": 5.0, "no2_ppb": 12.0, "rh_pct": 88.0, "rain_gauge_mm": 35.0, "data_missing": False},
            "bulletin_embs": [],
            "manifest_id_era5": "t", "manifest_id_s1": "t", "manifest_id_lc": "t",
            "manifest_id_aq": "t", "manifest_id_bulletins": "t",
        }
        result = orchestrator.route(0, date(2022, 3, 1), ctx)
        assert result.fused_opinion.is_valid()
        assert result.fusion_operator in {"CCF", "WBF", "SINGLE", "NONE"}
        assert 0.0 <= result.max_js <= 1.0

    def test_partial_modality_missing(self, orchestrator):
        ctx = {"era5_row": None, "sar_row": None, "lc_row": None, "aq_row": {"data_missing": True}, "bulletin_embs": []}
        result = orchestrator.route(0, date(2022, 3, 1), ctx)
        assert result.fused_opinion.is_valid()
        # All modalities missing -> very high uncertainty
        assert result.fused_opinion.u > 0.5

    def test_enabled_agents_subset(self, orchestrator):
        ctx = {
            "era5_row": {"tp_mm": 30.0, "precip_7d_sum": 150.0, "precip_30d_anom": 80.0, "t2m_c": 26.0, "ssrd_mj": 15.0},
            "sar_row": None, "lc_row": None, "aq_row": {"data_missing": True}, "bulletin_embs": [],
        }
        result = orchestrator.route(0, date(2022, 3, 1), ctx, enabled_agents=["climate_agent"])
        assert result.fused_opinion.is_valid()
        assert len(result.agent_opinions) == 1


class TestEvidentialHead:
    def test_predict_valid_structure(self, orchestrator):
        ctx = {
            "era5_row": {"tp_mm": 40.0, "precip_7d_sum": 200.0, "precip_30d_anom": 100.0, "t2m_c": 25.0, "ssrd_mj": 12.0},
            "sar_row": {"sar_vv_db": -14.0, "sar_vh_db": -20.0, "water_index_sar": 6.0, "ndwi": 0.2, "ndvi": 0.3, "cloud_mask_frac": 0.0},
            "lc_row": {"land_cover": 4, "slope_deg": 2.0, "impervious_frac": 0.7, "elevation_m": 3.0},
            "aq_row": {"pm25_ug": 5.0, "no2_ppb": 12.0, "rh_pct": 88.0, "rain_gauge_mm": 35.0, "data_missing": False},
            "bulletin_embs": [],
        }
        result = orchestrator.route(0, date(2022, 3, 1), ctx)
        fv = opinion_to_feature_vector(result)
        head = EvidentialHead()
        pred = head.predict(fv, fused_opinion=result.fused_opinion)
        assert "flood_state" in pred
        assert 0 <= pred["flood_state"] <= 3
        assert abs(sum(pred["state_proba"]) - 1.0) < 1e-5
        assert pred["depth_m"] >= 0.0


class TestLLMBaseline:
    def test_predict_structure(self):
        bl3 = LLMArbitratedBaseline()
        agent_probas = {
            "climate_agent":    np.array([0.1, 0.2, 0.3, 0.4]),
            "satellite_agent":  np.array([0.05, 0.1, 0.35, 0.5]),
            "landcover_agent":  np.array([0.15, 0.25, 0.3, 0.3]),
        }
        result = bl3.predict(agent_probas)
        assert "flood_state" in result
        assert 0 <= result["flood_state"] <= 3
        assert 0.0 <= result["uncertainty_u"] <= 1.0
