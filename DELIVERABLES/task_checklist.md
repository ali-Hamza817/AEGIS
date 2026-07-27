# AEGIS Implementation Tasks

## Phase 1: Core Infrastructure
- [x] `pyproject.toml` — dependency pinning
- [x] `configs/study_site.yaml` — Brisbane AOI
- [x] `configs/experiments.yaml` — SL params, baselines, ablations

## Phase 2: Subjective Logic Engine
- [x] `src/sl/opinion.py` — Opinion class, Dirichlet bijection
- [x] `src/sl/fusion.py` — WBF, CCF for n ≥ 2 sources
- [x] `src/sl/partial_obs.py` — Partial-observable update (Kaplan 2015)
- [x] `src/sl/credibility.py` — Brier-score reputation (Wang & Singh 2007)

## Phase 3: Data Layer
- [x] `src/ingestion/duckdb_schema.py` — DuckDB schema + provenance tables
- [x] `src/ingestion/synthetic_generator.py` — Brisbane SEQ 2022 synthetic data

## Phase 4: Specialist Agents
- [x] `src/agents/base.py` — Abstract agent + ProvenanceRecord
- [x] `src/agents/climate_agent.py` — ERA5 precipitation
- [x] `src/agents/satellite_agent.py` — Sentinel-1 SAR + Sentinel-2
- [x] `src/agents/landcover_agent.py` — ESA WorldCover + DEM
- [x] `src/agents/airquality_agent.py` — OpenAQ + gauge
- [x] `src/agents/docint_agent.py` — Bulletin text embeddings

## Phase 5: Coordinator
- [x] `src/coordinator/orchestrator.py` — Deterministic SL routing

## Phase 6: Prediction & Baselines
- [x] `src/prediction/evidential_head.py` — LightGBM evidential head
- [x] `src/prediction/baselines.py` — BL1, BL2, BL3

## Phase 7: Evaluation
- [x] `src/eval/metrics.py` — F1, AUROC, ECE, RMSE, latency, monotonicity
- [x] `src/eval/ablations.py` — H1–H4 ablation runners
- [x] `src/eval/user_study.py` — H4 simulated user study

## Phase 8: API & Dashboard
- [x] `src/api/app.py` — FastAPI endpoints
- [x] `web/public/index.html` — Interactive Leaflet + Plotly dashboard

## Phase 9: Testing & Pipeline
- [x] `tests/test_sl.py` — 21 SL unit tests (all pass)
- [x] `tests/test_pipeline.py` — 13 integration tests (all pass)
- [x] `experiments/run_pipeline.py` — Full experiment pipeline execution

## Phase 10: Packaging
- [x] `docker/dockerfile` — Container definition
- [x] `README.md` — Project documentation
- [x] `walkthrough.md` — Final summary walkthrough
