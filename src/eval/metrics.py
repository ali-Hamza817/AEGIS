"""
src/eval/metrics.py
====================
Evaluation metrics for AEGIS flood prediction.

Implements all metrics from the AEGIS paper hypotheses:

    H1/H2: Classification quality
        - F1-score (macro and per-class)
        - AUROC (multi-class OvR)
        - AUPRC (per-class Average Precision)

    H3: Regression & calibration
        - RMSE, MAE on flood depth (m)
        - Expected Calibration Error (ECE) for probability calibration

    Computational:
        - Wall-clock latency (ms per inference)
        - Peak RAM (MB, via tracemalloc)

All functions accept numpy arrays and return plain dicts (JSON-serialisable).
"""

from __future__ import annotations

import tracemalloc
import time
import logging
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

FLOOD_LABELS = ["Dry", "Saturated", "SurfaceFlow", "Inundation"]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute F1, AUROC, AUPRC for multi-class flood state prediction.

    Args:
        y_true  : Ground truth labels (N,) in {0,1,2,3}.
        y_pred  : Predicted labels (N,).
        y_proba : Predicted probability matrix (N, 4). Optional.

    Returns:
        dict with f1_macro, f1_weighted, f1_per_class, auroc, auprc, confusion_matrix.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist()

    result = {
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": dict(zip(FLOOD_LABELS, f1_per_class)),
        "confusion_matrix": cm,
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=np.float64)
        # OvR AUROC
        try:
            auroc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except ValueError:
            auroc = float("nan")
        result["auroc_macro"] = auroc

        # Average Precision (AUPRC) per class
        auprc = {}
        for k, label in enumerate(FLOOD_LABELS):
            y_bin = (y_true == k).astype(int)
            try:
                ap = float(average_precision_score(y_bin, y_proba[:, k]))
            except ValueError:
                ap = float("nan")
            auprc[label] = ap
        result["auprc_per_class"] = auprc
        result["auprc_macro"] = float(np.nanmean(list(auprc.values())))

    return result


def compute_regression_metrics(
    y_true_depth: np.ndarray,
    y_pred_depth: np.ndarray,
) -> dict[str, float]:
    """
    RMSE and MAE for water depth regression (meters).
    """
    y_true = np.asarray(y_true_depth, dtype=np.float64)
    y_pred = np.asarray(y_pred_depth, dtype=np.float64)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse_m": rmse, "mae_m": mae}


def compute_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) for multi-class prediction.

    Bins are over max-confidence (top-1 probability).
    ECE = sum_b (|B_b| / N) * |accuracy(B_b) - confidence(B_b)|

    Reference: Guo et al. (2017). On Calibration of Modern Neural Networks.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    y_pred = np.argmax(y_proba, axis=1)
    confidences = np.max(y_proba, axis=1)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        acc_bin = float((y_pred[mask] == y_true[mask]).mean())
        conf_bin = float(confidences[mask].mean())
        ece += (n_bin / N) * abs(acc_bin - conf_bin)

    return float(ece)


def measure_latency(
    fn: Callable,
    n_trials: int = 20,
) -> dict[str, float]:
    """
    Measure wall-clock inference latency in milliseconds.

    Returns:
        dict with mean_ms, p50_ms, p95_ms, p99_ms.
    """
    times_ms = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    times_arr = np.array(times_ms)
    return {
        "mean_ms": float(times_arr.mean()),
        "p50_ms": float(np.percentile(times_arr, 50)),
        "p95_ms": float(np.percentile(times_arr, 95)),
        "p99_ms": float(np.percentile(times_arr, 99)),
    }


def measure_peak_ram_mb(fn: Callable) -> float:
    """
    Measure peak RAM usage of fn() via tracemalloc.
    Returns peak in MB.
    """
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return float(peak / 1024 / 1024)


def compute_uncertainty_monotonicity(
    uncertainties_by_n_missing: dict[int, list[float]],
) -> dict[str, Any]:
    """
    Verify H3: uncertainty increases monotonically as modalities are removed.

    Args:
        uncertainties_by_n_missing: {n_missing: [u_values]}
            where n_missing is the number of dropped modalities.

    Returns:
        dict with mean_u per n_missing, monotone_check (bool), violations.
    """
    keys = sorted(uncertainties_by_n_missing.keys())
    mean_u = {k: float(np.mean(uncertainties_by_n_missing[k])) for k in keys}

    violations = []
    for i in range(len(keys) - 1):
        k1, k2 = keys[i], keys[i + 1]
        if mean_u[k1] > mean_u[k2]:
            violations.append((k1, k2, mean_u[k1], mean_u[k2]))

    return {
        "mean_u_by_n_missing": mean_u,
        "monotone": len(violations) == 0,
        "violations": violations,
    }


def summarise_results(metrics_dict: dict[str, Any]) -> str:
    """Format metrics as a human-readable table string."""
    lines = ["=" * 60, "AEGIS Evaluation Summary", "=" * 60]
    for key, val in metrics_dict.items():
        if isinstance(val, dict):
            lines.append(f"\n[{key}]")
            for k2, v2 in val.items():
                lines.append(f"  {k2:30s}: {v2}")
        else:
            lines.append(f"  {key:30s}: {val}")
    return "\n".join(lines)
