"""
src/agents/airquality_agent.py
===============================
Air Quality / Sensor Agent.

Uses OpenAQ sensors as rainfall-correlated proxies:
    - PM2.5 drops during heavy rain (washout effect)
    - Relative humidity increases sharply before/during flood events
    - rain_gauge_mm (if available) is the primary direct signal

The agent gracefully handles sensor dropout (15-20% in OpenAQ data)
by triggering ModalityMissingError when data_missing=True.
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

AQ_FEATURES = ["pm25_ug", "no2_ppb", "rh_pct", "rain_gauge_mm"]


class AirQualityAgent(BaseAgent):

    def __init__(
        self,
        model_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="airquality_agent",
            modality="OpenAQ+Gauge",
            model_ckpt=str(model_path) if model_path else "rule-based",
            default_uncertainty=0.6,   # sensor data is noisiest modality
            **kwargs,
        )
        self.model = None
        if model_path and Path(model_path).exists():
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self._is_fitted = True

    def _compute_proba(
        self,
        cell_id: int,
        target_date: date,
        context: dict[str, Any],
    ) -> tuple[np.ndarray, float, str, list[str]]:
        aq_row = context.get("aq_row")
        if aq_row is None or aq_row.get("data_missing", True):
            raise ModalityMissingError("OpenAQ/gauge data missing for this cell/date.")

        feats = [
            float(aq_row.get("pm25_ug") or 0.0),
            float(aq_row.get("no2_ppb") or 0.0),
            float(aq_row.get("rh_pct") or 50.0),
            float(aq_row.get("rain_gauge_mm") or 0.0),
        ]
        feat_arr = np.array(feats, dtype=np.float64).reshape(1, -1)
        manifest_id = context.get("manifest_id_aq", "unknown")

        if self.model is not None and self._is_fitted:
            proba = self.model.predict_proba(feat_arr)[0]
        else:
            proba = self._rule_based(feats)

        # Sensors are noisy — floor uncertainty at 0.4
        uncertainty = max(float(1.0 - np.max(proba)), 0.4)
        return proba, uncertainty, manifest_id, AQ_FEATURES

    @staticmethod
    def _rule_based(feats: list[float]) -> np.ndarray:
        feat_dict = dict(zip(AQ_FEATURES, feats))
        rh = float(feat_dict.get("rh_pct", 50.0))
        gauge = float(feat_dict.get("rain_gauge_mm", 0.0))
        pm25 = float(feat_dict.get("pm25_ug", 10.0))

        # RH > 85% + gauge > 30 mm -> likely flood conditions
        rh_signal = max(0.0, (rh - 70.0) / 30.0)
        gauge_signal = max(0.0, (gauge - 10.0) / 40.0)
        pm_signal = max(0.0, (10.0 - pm25) / 10.0)  # low PM = washout = rain
        combined = (rh_signal + gauge_signal + pm_signal) / 3.0

        logit_dry       = max(0.0, 2.5 - combined * 4.0)
        logit_saturated = max(0.0, 2.0 - abs(combined - 0.3) * 5.0)
        logit_surface   = max(0.0, 2.0 - abs(combined - 0.65) * 4.0)
        logit_inund     = max(0.0, combined * 3.5 - 1.2)

        logits = np.array([logit_dry, logit_saturated, logit_surface, logit_inund])
        logits -= logits.max()
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum()
