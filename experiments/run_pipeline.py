"""
experiments/run_pipeline.py
============================
AEGIS Master Pipeline Runner.

Executes the full AEGIS experimental pipeline:

    Step 0: Setup — initialise database, generate synthetic data.
    Step 1: Build feature contexts from DuckDB for each (cell, date).
    Step 2: Run AEGIS-SL framework (all 5 agents + SL coordinator).
    Step 3: Run all 3 baselines on the same data.
    Step 4: Train evidential prediction heads (classifier + regressor).
    Step 5: Evaluate AEGIS vs baselines (H1-H3 metrics).
    Step 6: Run ablation studies (H1-H4).
    Step 7: Save results/metrics.json and results/opinions.parquet.
    Step 8: Print paper-ready summary table.

Usage:
    python experiments/run_pipeline.py --n-cells 200 --seed 42

Reproduces results in <6 hours on 6-core CPU, 16 GB RAM.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np

# --- Setup Python path ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.synthetic_generator import SyntheticGenerator
from src.ingestion.duckdb_schema import get_conn
from src.agents import (
    ClimateAgent, SatelliteAgent, LandCoverAgent,
    AirQualityAgent, DocIntAgent,
)
from src.coordinator.orchestrator import SLOrchestrator
from src.sl.credibility import CredibilityRegistry
from src.prediction.evidential_head import EvidentialHead, opinion_to_feature_vector
from src.prediction.baselines import (
    SingleModalityBaseline,
    MonolithicFusionBaseline,
    LLMArbitratedBaseline,
    ERA5_FEAT_NAMES,
)
from src.eval.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_ece,
    compute_uncertainty_monotonicity,
    summarise_results,
)
from src.eval.ablations import (
    run_h1_specialization_ablation,
    run_h2_fusion_ablation,
    run_h3_missing_modality_ablation,
    run_h4_provenance_ablation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger("AEGIS.pipeline")

FLOOD_LABELS = ["Dry", "Saturated", "SurfaceFlow", "Inundation"]


def parse_args():
    p = argparse.ArgumentParser(description="AEGIS Master Pipeline Runner")
    p.add_argument("--n-cells", type=int, default=200, help="Number of spatial cells")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--db-path", type=str, default="data/duckdb/flood.duckdb")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--skip-generation", action="store_true",
                   help="Skip data generation (use existing DB)")
    p.add_argument("--tau-low", type=float, default=0.1)
    p.add_argument("--tau-high", type=float, default=0.3)
    p.add_argument("--c-dirichlet", type=float, default=6.0)
    return p.parse_args()


def _get_context(conn, cell_id: int, target_date: date, n_days_bulletin: int = 14) -> dict:
    """Query all modality rows for a cell+date from DuckDB."""
    era5 = conn.execute(
        "SELECT tp_mm, precip_7d_sum, precip_30d_anom, t2m_c, ssrd_mj "
        "FROM era5_daily WHERE cell_id=? AND date=?",
        [cell_id, target_date],
    ).fetchone()
    era5_row = dict(zip(["tp_mm", "precip_7d_sum", "precip_30d_anom", "t2m_c", "ssrd_mj"],
                        era5 or [None] * 5)) if era5 else None

    sar = conn.execute(
        "SELECT sar_vv_db, sar_vh_db, water_index_sar, ndwi, ndvi, cloud_mask_frac "
        "FROM sentinel_features WHERE cell_id=? AND date=?",
        [cell_id, target_date],
    ).fetchone()
    sar_row = dict(zip(
        ["sar_vv_db", "sar_vh_db", "water_index_sar", "ndwi", "ndvi", "cloud_mask_frac"],
        sar or [None] * 6,
    )) if sar else None

    lc = conn.execute(
        "SELECT land_cover, slope_deg, impervious_frac, elevation_m FROM site_grid WHERE cell_id=?",
        [cell_id],
    ).fetchone()
    lc_row = dict(zip(["land_cover", "slope_deg", "impervious_frac", "elevation_m"],
                      lc or [None] * 4)) if lc else None

    aq = conn.execute(
        "SELECT pm25_ug, no2_ppb, rh_pct, rain_gauge_mm, data_missing "
        "FROM openaq_daily WHERE cell_id=? AND date=?",
        [cell_id, target_date],
    ).fetchone()
    aq_row = dict(zip(["pm25_ug", "no2_ppb", "rh_pct", "rain_gauge_mm", "data_missing"],
                      aq or [None, None, None, None, True])) if aq else {"data_missing": True}

    window_start = target_date - timedelta(days=n_days_bulletin)
    emb_rows = conn.execute(
        "SELECT be.embedding FROM bulletin_emb be "
        "JOIN bulletin_doc bd ON be.doc_id = bd.doc_id "
        "WHERE bd.date BETWEEN ? AND ?",
        [window_start, target_date],
    ).fetchall()
    bulletin_embs = [list(r[0]) for r in emb_rows] if emb_rows else []

    return {
        "era5_row": era5_row, "sar_row": sar_row, "lc_row": lc_row,
        "aq_row": aq_row, "bulletin_embs": bulletin_embs,
        "manifest_id_era5": "db", "manifest_id_s1": "db",
        "manifest_id_lc": "db", "manifest_id_aq": "db", "manifest_id_bulletins": "db",
    }


def _build_monolithic_feature_row(context: dict) -> np.ndarray:
    """Build the concatenated feature vector for Baseline 2."""
    era5 = context.get("era5_row") or {}
    sar = context.get("sar_row") or {}
    lc = context.get("lc_row") or {}
    aq = context.get("aq_row") or {}
    feats = (
        [float(era5.get(k) or 0.0) for k in ERA5_FEAT_NAMES] +
        [float(sar.get(k) or 0.0) for k in ["sar_vv_db", "sar_vh_db", "water_index_sar", "ndwi", "ndvi"]] +
        [float(lc.get(k) or 0.0) for k in ["land_cover", "slope_deg", "impervious_frac", "elevation_m"]] +
        [float(aq.get("rh_pct") or 50.0), float(aq.get("rain_gauge_mm") or 0.0)]
    )
    return np.array(feats, dtype=np.float64)


def main():
    args = parse_args()
    run_id = str(uuid.uuid4())[:8]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("AEGIS Pipeline Start (run_id=%s)", run_id)
    logger.info("=" * 60)

    # ---------------------------------------------------------------
    # STEP 0: Data Generation
    # ---------------------------------------------------------------
    db_path = Path(args.db_path)
    if not args.skip_generation or not db_path.exists():
        logger.info("Step 0: Generating synthetic Brisbane SEQ 2022 dataset...")
        gen = SyntheticGenerator(
            db_path=db_path,
            n_cells=args.n_cells,
            seed=args.seed,
        )
        gen.run()
    else:
        logger.info("Step 0: Skipped (using existing DB at %s).", db_path)

    conn = get_conn(db_path)

    # ---------------------------------------------------------------
    # STEP 1: Build (cell_id, date, context, truth) dataset
    # ---------------------------------------------------------------
    logger.info("Step 1: Building evaluation dataset...")
    all_rows = conn.execute(
        "SELECT cell_id, date, flood_state, flood_depth_m FROM truth ORDER BY cell_id, date"
    ).fetchall()

    dataset = []
    for (cell_id, dt, flood_state, depth) in all_rows:
        ctx = _get_context(conn, int(cell_id), dt)
        dataset.append((int(cell_id), dt, ctx, int(flood_state), float(depth or 0.0)))

    # Train / test split: first 70% of dates = train, rest = test
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    train_data = dataset[:n_train]
    test_data  = dataset[n_train:]
    logger.info("Dataset: %d total, %d train, %d test.", n_total, n_train, len(test_data))

    # ---------------------------------------------------------------
    # STEP 2: AEGIS-SL Framework
    # ---------------------------------------------------------------
    logger.info("Step 2: Running AEGIS-SL framework...")
    registry = CredibilityRegistry.default()
    agents = [
        ClimateAgent(), SatelliteAgent(), LandCoverAgent(),
        AirQualityAgent(), DocIntAgent(),
    ]
    orchestrator = SLOrchestrator(
        agents=agents,
        credibility_registry=registry,
        tau_low=args.tau_low,
        tau_high=args.tau_high,
        c_dirichlet=args.c_dirichlet,
        run_id=run_id,
        db_conn=conn,
    )

    aegis_train_results, aegis_test_results = [], []
    X_sl_train, X_sl_test = [], []

    for split_data, split_results, X_split in [
        (train_data, aegis_train_results, X_sl_train),
        (test_data, aegis_test_results, X_sl_test),
    ]:
        for cell_id, dt, ctx, _, _ in split_data:
            orch_result = orchestrator.route(cell_id, dt, ctx)
            fv = opinion_to_feature_vector(orch_result)
            pred_dict = orch_result.to_dict()
            pred_dict["fused_opinion_obj"] = orch_result.fused_opinion
            split_results.append(pred_dict)
            X_split.append(fv)

    logger.info("AEGIS: %d train + %d test predictions.", len(aegis_train_results), len(aegis_test_results))

    # ---------------------------------------------------------------
    # STEP 3: Baselines
    # ---------------------------------------------------------------
    logger.info("Step 3: Running baselines...")
    y_train = np.array([r[3] for r in train_data])
    y_test  = np.array([r[3] for r in test_data])
    depth_train = np.array([r[4] for r in train_data])
    depth_test  = np.array([r[4] for r in test_data])

    # Baseline 1: Single-modality (ERA5)
    X_era5_train = np.array([
        [float((c.get("era5_row") or {}).get(k) or 0.0) for k in ERA5_FEAT_NAMES]
        for (_, _, c, _, _) in train_data
    ])
    X_era5_test = np.array([
        [float((c.get("era5_row") or {}).get(k) or 0.0) for k in ERA5_FEAT_NAMES]
        for (_, _, c, _, _) in test_data
    ])
    bl1 = SingleModalityBaseline()
    try:
        bl1.fit(X_era5_train, y_train, save_path="results/models/bl1.pkl")
    except Exception as e:
        logger.warning("Baseline 1 fit failed: %s (using rule-based).", e)
    bl1_proba_test = bl1.predict_proba(X_era5_test)
    bl1_pred_test = np.argmax(bl1_proba_test, axis=1)
    bl1_results = [{"flood_state": int(p), "state_proba": pr.tolist()}
                   for p, pr in zip(bl1_pred_test, bl1_proba_test)]

    # Baseline 2: Monolithic Late-Fusion
    X_mono_train = np.array([_build_monolithic_feature_row(c) for (_, _, c, _, _) in train_data])
    X_mono_test  = np.array([_build_monolithic_feature_row(c) for (_, _, c, _, _) in test_data])
    bl2 = MonolithicFusionBaseline()
    try:
        bl2.fit(X_mono_train, y_train, save_path="results/models/bl2.pkl")
    except Exception as e:
        logger.warning("Baseline 2 fit failed: %s (using rule-based).", e)
    bl2_proba_test = bl2.predict_proba(X_mono_test)
    bl2_pred_test = np.argmax(bl2_proba_test, axis=1)
    bl2_results = [{"flood_state": int(p), "state_proba": pr.tolist()}
                   for p, pr in zip(bl2_pred_test, bl2_proba_test)]

    # Baseline 3: LLM-Arbitrated
    bl3 = LLMArbitratedBaseline()
    bl3_results = []
    for orch_result_dict, (_, _, ctx, _, _) in zip(aegis_test_results, test_data):
        # Re-route to get per-agent probas
        agent_probas = {}
        for ag in agents:
            try:
                op, _ = ag.emit(0, date(2022, 3, 1), ctx)
                agent_probas[ag.name] = op.projected_probability()
            except Exception:
                agent_probas[ag.name] = np.ones(4) / 4.0
        bl3_pred = bl3.predict(agent_probas)
        bl3_results.append(bl3_pred)

    # ---------------------------------------------------------------
    # STEP 4: Train Evidential Head
    # ---------------------------------------------------------------
    logger.info("Step 4: Training Evidential Head...")
    head = EvidentialHead()
    X_sl_train_arr = np.array(X_sl_train)
    X_sl_test_arr  = np.array(X_sl_test)
    try:
        head.fit_classifier(X_sl_train_arr, y_train, save_path="results/models/head_clf.pkl")
        head.fit_regressor(X_sl_train_arr, depth_train, save_path="results/models/head_reg.pkl")
    except Exception as e:
        logger.warning("Head training failed: %s (using rule-based fallback).", e)

    # AEGIS test predictions
    aegis_preds, aegis_probas, aegis_depths, aegis_u = [], [], [], []
    for fv, orch_dict in zip(X_sl_test_arr, aegis_test_results):
        fused_op = orch_dict.get("fused_opinion_obj")
        pred = head.predict(fv, fused_opinion=fused_op)
        aegis_preds.append(pred["flood_state"])
        aegis_probas.append(pred["state_proba"])
        aegis_depths.append(pred["depth_m"])
        aegis_u.append(pred["uncertainty_u"])

    # ---------------------------------------------------------------
    # STEP 5: Evaluation
    # ---------------------------------------------------------------
    logger.info("Step 5: Evaluating...")
    aegis_proba_arr = np.array(aegis_probas)
    aegis_pred_arr = np.array(aegis_preds)

    metrics = {
        "AEGIS_SL": {
            **compute_classification_metrics(y_test, aegis_pred_arr, aegis_proba_arr),
            **compute_regression_metrics(depth_test, np.array(aegis_depths)),
            "ece": compute_ece(y_test, aegis_proba_arr),
            "mean_uncertainty_u": float(np.mean(aegis_u)),
        },
        "Baseline1_ERA5_only": compute_classification_metrics(y_test, bl1_pred_test, bl1_proba_test),
        "Baseline2_MonolithicFusion": compute_classification_metrics(y_test, bl2_pred_test, bl2_proba_test),
        "Baseline3_LLM_Arbitrated": {
            **compute_classification_metrics(
                y_test,
                np.array([r["flood_state"] for r in bl3_results]),
                np.array([r["state_proba"] for r in bl3_results]),
            ),
            "mean_uncertainty_u": float(np.mean([r.get("uncertainty_u", 0.5) for r in bl3_results])),
        },
    }

    # ---------------------------------------------------------------
    # STEP 6: Ablations
    # ---------------------------------------------------------------
    logger.info("Step 6: Running ablation studies...")
    aegis_result_dicts_with_ops = []
    for orch_d in aegis_test_results[:50]:   # sample for ablation speed
        aegis_result_dicts_with_ops.append(orch_d)

    h1 = run_h1_specialization_ablation(
        [{"flood_state": p, "state_proba": pr} for p, pr in zip(aegis_preds, aegis_probas)],
        bl2_results,
        y_test,
    )
    h2 = run_h2_fusion_ablation(
        [{"flood_state": p, "state_proba": pr, "uncertainty_u": u}
         for p, pr, u in zip(aegis_preds, aegis_probas, aegis_u)],
        bl3_results,
        y_test,
    )

    # H3: Simulate missing modality ablation
    sample_inputs = [(r[0], r[1], r[2]) for r in test_data[:30]]
    h3 = run_h3_missing_modality_ablation(orchestrator, sample_inputs, max_drop=4)

    h4 = run_h4_provenance_ablation(aegis_test_results)

    ablation_results = {**h1, **h2, **h3, **h4}

    # ---------------------------------------------------------------
    # STEP 7: Save Results
    # ---------------------------------------------------------------
    full_results = {
        "run_id": run_id,
        "config": vars(args),
        "metrics": metrics,
        "ablations": ablation_results,
    }

    results_path = results_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    logger.info("Results saved to %s", results_path)

    # ---------------------------------------------------------------
    # STEP 8: Paper-ready summary
    # ---------------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("AEGIS RESULTS — PAPER TABLE")
    print("=" * 70)
    print(f"{'Method':<30} {'F1-Macro':>10} {'AUROC':>10} {'ECE':>8} {'Mean-u':>8}")
    print("-" * 70)
    for method, m in metrics.items():
        print(f"{method:<30} "
              f"{m.get('f1_macro', 0.0):>10.4f} "
              f"{m.get('auroc_macro', float('nan')):>10.4f} "
              f"{m.get('ece', float('nan')):>8.4f} "
              f"{m.get('mean_uncertainty_u', float('nan')):>8.4f}")
    print("=" * 70)
    print("\nH3 Monotone Uncertainty Check:")
    h3_data = ablation_results.get("H3_missing_modality_uncertainty", {})
    print(f"  Monotone: {h3_data.get('monotone', 'N/A')}")
    for n_miss, u_val in sorted(h3_data.get("mean_u_by_n_missing", {}).items()):
        print(f"  n_missing={n_miss}: mean_u={u_val:.4f}")
    print()
    conn.close()
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
