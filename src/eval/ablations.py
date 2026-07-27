"""
src/eval/ablations.py
======================
Ablation study runner for AEGIS.

Implements all four hypothesis ablations:

    H1: Specialization vs Monolithic Fusion
        Measure ΔF1: AEGIS-SL vs Baseline 2 (monolithic LightGBM).

    H2: Evidential Fusion vs LLM Arbitration
        Measure ΔAUROC: AEGIS-SL vs Baseline 3 (LLM-arbitrated).

    H3: Missing Modality Uncertainty Growth
        Systematically drop 1, 2, 3, 4 modalities and verify
        that mean uncertainty u grows monotonically (Kaplan 2015 prediction).

    H4: Provenance Explanation Quality
        Compare explanation richness: SL contribution weights vs
        baseline SHAP attribution (Likert simulation + accuracy retention).

All ablations operate on pre-computed OrchestratorResults for efficiency.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from itertools import combinations
from typing import Any

import numpy as np

from src.coordinator.orchestrator import SLOrchestrator
from src.eval.metrics import (
    compute_classification_metrics,
    compute_uncertainty_monotonicity,
)

logger = logging.getLogger(__name__)

AGENT_NAMES = [
    "climate_agent",
    "satellite_agent",
    "landcover_agent",
    "airquality_agent",
    "docint_agent",
]


def run_h1_specialization_ablation(
    aegis_results: list[dict[str, Any]],
    baseline2_results: list[dict[str, Any]],
    y_true: np.ndarray,
) -> dict[str, Any]:
    """
    H1: AEGIS-SL vs Monolithic Late-Fusion.

    Args:
        aegis_results     : List of dicts with 'flood_state' and 'state_proba'.
        baseline2_results : Same format from MonolithicFusionBaseline.
        y_true            : Ground truth labels.

    Returns:
        Delta-F1 (macro) and per-class comparison.
    """
    aegis_pred = np.array([r["flood_state"] for r in aegis_results])
    aegis_proba = np.array([r["state_proba"] for r in aegis_results])
    bl2_pred = np.array([r["flood_state"] for r in baseline2_results])
    bl2_proba = np.array([r["state_proba"] for r in baseline2_results])

    m_aegis = compute_classification_metrics(y_true, aegis_pred, aegis_proba)
    m_bl2 = compute_classification_metrics(y_true, bl2_pred, bl2_proba)

    delta_f1 = m_aegis["f1_macro"] - m_bl2["f1_macro"]
    delta_auroc = m_aegis.get("auroc_macro", 0.0) - m_bl2.get("auroc_macro", 0.0)

    return {
        "H1_specialization_vs_monolithic": {
            "aegis_f1_macro": m_aegis["f1_macro"],
            "baseline2_f1_macro": m_bl2["f1_macro"],
            "delta_f1_macro": delta_f1,
            "aegis_auroc": m_aegis.get("auroc_macro"),
            "baseline2_auroc": m_bl2.get("auroc_macro"),
            "delta_auroc": delta_auroc,
            "aegis_per_class_f1": m_aegis["f1_per_class"],
            "baseline2_per_class_f1": m_bl2["f1_per_class"],
        }
    }


def run_h2_fusion_ablation(
    aegis_results: list[dict[str, Any]],
    llm_results: list[dict[str, Any]],
    y_true: np.ndarray,
) -> dict[str, Any]:
    """
    H2: Evidential SL Fusion vs LLM-Arbitrated Agentic Baseline.
    """
    aegis_pred = np.array([r["flood_state"] for r in aegis_results])
    aegis_proba = np.array([r["state_proba"] for r in aegis_results])
    llm_pred = np.array([r["flood_state"] for r in llm_results])
    llm_proba = np.array([r["state_proba"] for r in llm_results])

    m_aegis = compute_classification_metrics(y_true, aegis_pred, aegis_proba)
    m_llm = compute_classification_metrics(y_true, llm_pred, llm_proba)

    delta_auroc = m_aegis.get("auroc_macro", 0.0) - m_llm.get("auroc_macro", 0.0)

    # Calibration comparison
    aegis_u = np.array([r.get("uncertainty_u", 0.5) for r in aegis_results])
    llm_u = np.array([r.get("uncertainty_u", 0.5) for r in llm_results])

    return {
        "H2_sl_fusion_vs_llm_arbitration": {
            "aegis_f1_macro": m_aegis["f1_macro"],
            "llm_f1_macro": m_llm["f1_macro"],
            "delta_f1": m_aegis["f1_macro"] - m_llm["f1_macro"],
            "aegis_auroc": m_aegis.get("auroc_macro"),
            "llm_auroc": m_llm.get("auroc_macro"),
            "delta_auroc": delta_auroc,
            "aegis_mean_u": float(aegis_u.mean()),
            "llm_mean_u_entropy_approx": float(llm_u.mean()),
            "calibration_note": (
                "SL uncertainty u is formally calibrated (Dirichlet bijection); "
                "LLM uncertainty is entropy-normalised (uncalibrated)."
            ),
        }
    }


def run_h3_missing_modality_ablation(
    orchestrator: SLOrchestrator,
    sample_inputs: list[tuple[int, Any, dict[str, Any]]],
    max_drop: int = 4,
) -> dict[str, Any]:
    """
    H3: Monotone uncertainty growth under missing modalities.

    For each number of dropped modalities n in {0, 1, 2, 3, 4}:
        Randomly select which agents to disable.
        Collect uncertainty u from coordinator result.
        Verify mean(u | n_missing=k) < mean(u | n_missing=k+1).

    Args:
        orchestrator  : Configured SLOrchestrator.
        sample_inputs : List of (cell_id, target_date, context) tuples.
        max_drop      : Maximum number of agents to drop.

    Returns:
        Monotonicity check + mean uncertainty per n_missing.
    """
    u_by_n_missing: dict[int, list[float]] = {k: [] for k in range(max_drop + 1)}

    for cell_id, target_date, context in sample_inputs:
        for n_drop in range(max_drop + 1):
            enabled = list(AGENT_NAMES)
            if n_drop > 0:
                # Deterministic selection: drop last n_drop agents
                enabled = enabled[: len(AGENT_NAMES) - n_drop]

            result = orchestrator.route(
                cell_id=cell_id,
                target_date=target_date,
                context=context,
                enabled_agents=enabled,
            )
            u = result.fused_opinion.u
            u_by_n_missing[n_drop].append(u)

    monotonicity = compute_uncertainty_monotonicity(u_by_n_missing)
    return {"H3_missing_modality_uncertainty": monotonicity}


def run_h4_provenance_ablation(
    orchestrator_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    H4: Provenance-tagged explanation quality assessment.

    Simulates the user study metrics:
    - Explanation completeness: fraction of results with per-agent contributions.
    - Average agent contribution entropy (lower = more decisive explanation).
    - Fraction of results where dominant agent contributes > 40%.

    Real user study (n=30 analysts) is conducted separately.
    """
    has_contributions = sum(
        1 for r in orchestrator_results if "agent_contributions" in r
    )
    completeness = float(has_contributions) / max(1, len(orchestrator_results))

    contribution_entropies = []
    dominant_above_40 = 0
    for r in orchestrator_results:
        contribs = r.get("agent_contributions", {})
        if contribs:
            w = np.array(list(contribs.values()), dtype=np.float64)
            w = w / w.sum()
            eps = 1e-12
            entropy = float(-np.sum(w * np.log(w + eps)))
            contribution_entropies.append(entropy)
            if w.max() > 0.4:
                dominant_above_40 += 1

    return {
        "H4_provenance_explanation": {
            "explanation_completeness": completeness,
            "mean_contribution_entropy": (
                float(np.mean(contribution_entropies)) if contribution_entropies else None
            ),
            "fraction_decisive_dominant_40pct": (
                dominant_above_40 / max(1, len(orchestrator_results))
            ),
            "note": (
                "Full user study (n=30, Likert scale) conducted separately. "
                "These are automated proxy metrics."
            ),
        }
    }
