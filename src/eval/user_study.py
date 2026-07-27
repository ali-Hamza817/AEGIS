"""
src/eval/user_study.py
=======================
User Study Simulation for H4: Provenance-Tagged Explanations vs SHAP.

Simulates the pre-registered user study (n=30 analysts):
    Panel A: SHAP-only attribution (feature importance bars)
    Panel B: SL provenance panel (agent contributions, credibility γ, source lineage)

Measures:
    - Likert trust rating (5-point scale) per panel
    - Time-to-decision (seconds)
    - Decision accuracy (% matching expert ground truth)
    - Accuracy retention (±1% vs SHAP-only)

This module generates simulated questionnaire data based on the
empirically-grounded hypothesis that provenance-tagged explanations
increase trust without harming decision accuracy (H4).

NOTE: Real user study (n=30 with IRB approval) must be conducted
separately. This simulation generates the expected data format
for analysis pipeline integration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def simulate_user_study(
    n_participants: int = 30,
    n_scenarios: int = 10,
    seed: int = 42,
) -> dict:
    """
    Simulate H4 user study data.

    Returns structured results compatible with statistical analysis
    (paired t-test, Wilcoxon signed-rank).
    """
    rng = np.random.default_rng(seed)

    participants = []
    for pid in range(n_participants):
        # Each participant evaluates n_scenarios under both panels
        shap_trust = []
        prov_trust = []
        shap_time = []
        prov_time = []
        shap_accuracy = []
        prov_accuracy = []

        for sid in range(n_scenarios):
            # SHAP panel: moderate trust, faster but less informed decisions
            st = int(rng.choice([2, 3, 3, 3, 4], p=[0.1, 0.35, 0.35, 0.1, 0.1]))
            shap_trust.append(st)
            shap_time.append(float(np.clip(rng.normal(25.0, 8.0), 5.0, None)))

            # Provenance panel: higher trust due to source traceability
            pt = int(rng.choice([3, 3, 4, 4, 5], p=[0.1, 0.2, 0.35, 0.25, 0.1]))
            prov_trust.append(pt)
            prov_time.append(float(np.clip(rng.normal(35.0, 10.0), 8.0, None)))

            # Accuracy: provenance ≈ same as SHAP (±1% hypothesis)
            scenario_difficulty = rng.uniform(0.5, 0.95)
            shap_accuracy.append(int(rng.random() < scenario_difficulty))
            prov_accuracy.append(int(rng.random() < scenario_difficulty * 1.01))

        participants.append({
            "participant_id": f"P{pid:03d}",
            "role": rng.choice(["hydrologist", "RS_analyst", "civil_protection"]),
            "shap": {
                "trust_likert": shap_trust,
                "time_s": shap_time,
                "accuracy": shap_accuracy,
            },
            "provenance": {
                "trust_likert": prov_trust,
                "time_s": prov_time,
                "accuracy": prov_accuracy,
            },
        })

    # Aggregate statistics
    all_shap_trust = [np.mean(p["shap"]["trust_likert"]) for p in participants]
    all_prov_trust = [np.mean(p["provenance"]["trust_likert"]) for p in participants]
    all_shap_acc = [np.mean(p["shap"]["accuracy"]) for p in participants]
    all_prov_acc = [np.mean(p["provenance"]["accuracy"]) for p in participants]
    all_shap_time = [np.mean(p["shap"]["time_s"]) for p in participants]
    all_prov_time = [np.mean(p["provenance"]["time_s"]) for p in participants]

    summary = {
        "n_participants": n_participants,
        "n_scenarios": n_scenarios,
        "shap_panel": {
            "mean_trust_likert": float(np.mean(all_shap_trust)),
            "std_trust": float(np.std(all_shap_trust)),
            "mean_accuracy": float(np.mean(all_shap_acc)),
            "mean_time_s": float(np.mean(all_shap_time)),
        },
        "provenance_panel": {
            "mean_trust_likert": float(np.mean(all_prov_trust)),
            "std_trust": float(np.std(all_prov_trust)),
            "mean_accuracy": float(np.mean(all_prov_acc)),
            "mean_time_s": float(np.mean(all_prov_time)),
        },
        "delta_trust": float(np.mean(all_prov_trust) - np.mean(all_shap_trust)),
        "delta_accuracy": float(np.mean(all_prov_acc) - np.mean(all_shap_acc)),
        "accuracy_retention_within_1pct": bool(
            abs(np.mean(all_prov_acc) - np.mean(all_shap_acc)) < 0.01
        ),
    }

    return {
        "summary": summary,
        "participants": participants,
    }


def run_user_study_and_save(
    output_path: str | Path = "results/user_study.json",
    **kwargs,
) -> dict:
    """Run the simulated user study and save results to JSON."""
    results = simulate_user_study(**kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("User study results saved to %s", path)
    logger.info(
        "H4 Summary: Δtrust=+%.2f Likert, accuracy retention=%s",
        results["summary"]["delta_trust"],
        results["summary"]["accuracy_retention_within_1pct"],
    )
    return results
