"""
src/agents/climate_agent.py
============================
Climate Agent — consumes ERA5 precipitation and temperature features.

Feature set:
    - tp_mm            : daily precipitation (mm)
    - precip_7d_sum    : 7-day rolling sum (mm) — primary flood driver
    - precip_30d_anom  : 30-day anomaly vs ERA5 climatology
    - t2m_c            : 2-m temperature
    - ssrd_mj          : surface solar radiation

Model: LightGBM multi-class classifier (4 flood states).
Trained on per-cell ERA5 features against EMSR ground-truth labels.

Uncertainty calibration:
    u = 1 - max(softmax proba)   capped at 0.9
    (converts LightGBM softmax confidence to SL uncertainty)
"""

from __future__ import annotations

import logging
import pickle
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseAgent, ModalityMissingError

logger = logging.getLogger(__name__)

ERA5_FEATURES = [
    "tp_mm", "precip_7d_sum", "precip_30d_anom", "t2m_c", "ssrd_mj"
]


class ClimateAgent(BaseAgent):

    def __init__(
        self,
        model_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="climate_agent",
            modality="ERA5",
            model_ckpt=str(model_path) if model_path else "rule-based",
            **kwargs,
        )
        self.model = None
        if model_path and Path(model_path).exists():
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self._is_fitted = True
            logger.info("ClimateAgent: loaded model from %s", model_path)

    def _compute_proba(
        self,
        cell_id: int,
        target_date: date,
        context: dict[str, Any],
    ) -> tuple[np.ndarray, float, str, list[str]]:
        """
        Extract ERA5 features from context and return flood-state probabilities.

        Context expected keys:
            era5_row : dict with ERA5_FEATURES keys (from DuckDB query)
            manifest_id: str
        """
        era5_row = context.get("era5_row")
        if era5_row is None:
            raise ModalityMissingError("ERA5 row not available in context.")

        feats = []
        missing_keys = []
        for feat in ERA5_FEATURES:
            val = era5_row.get(feat)
            if val is None:
                missing_keys.append(feat)
                feats.append(0.0)
            else:
                feats.append(float(val))

        if len(missing_keys) > 2:
            raise ModalityMissingError(f"Too many ERA5 fields missing: {missing_keys}")

        feat_arr = np.array(feats, dtype=np.float64).reshape(1, -1)
        manifest_id = context.get("manifest_id_era5", "unknown")

        if self.model is not None and self._is_fitted:
            proba = self.model.predict_proba(feat_arr)[0]
        else:
            # Rule-based fallback: parameterized by precip_7d_sum
            proba = self._rule_based(feats)

        uncertainty = float(1.0 - np.max(proba))
        uncertainty = min(uncertainty, 0.9)
        return proba, uncertainty, manifest_id, ERA5_FEATURES

    @staticmethod
    def _rule_based(feats: list[float]) -> np.ndarray:
        """
        Calibrated rule-based flood state estimation from ERA5 features.

        State thresholds derived from Brisbane flood literature:
            precip_7d_sum < 20 mm    -> Dry
            20-100 mm                -> Saturated
            100-250 mm               -> SurfaceFlow
            > 250 mm                 -> Inundation

        Raw logits are passed through softmax to produce calibrated probabilities.
        """
        feat_dict = dict(zip(ERA5_FEATURES, feats))
        p7d = max(0.0, feat_dict.get("precip_7d_sum", 0.0))
        anom = feat_dict.get("precip_30d_anom", 0.0)
        t2m = feat_dict.get("t2m_c", 25.0)

        # Logits: higher = more likely
        logit_dry       = max(0.0, 3.0 - p7d / 10.0)
        logit_saturated = max(0.0, 3.0 - abs(p7d - 50.0) / 30.0)
        logit_surface   = max(0.0, 3.0 - abs(p7d - 150.0) / 60.0)
        logit_inund     = max(0.0, (p7d - 200.0) / 50.0 + anom / 20.0)

        logits = np.array([logit_dry, logit_saturated, logit_surface, logit_inund])
        logits -= logits.max()
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum()
