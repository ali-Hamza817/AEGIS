"""
src/agents/satellite_agent.py
==============================
Satellite Agent — Sentinel-1 SAR + Sentinel-2 optical features.

Feature set:
    - sar_vv_db, sar_vh_db : SAR backscatter channels (dB)
    - water_index_sar      : VV − VH (flood water proxy)
    - ndwi                 : Normalised Difference Water Index (S2)
    - ndvi                 : Normalised Difference Vegetation Index
    - cloud_mask_frac      : cloud cover fraction

Model: LightGBM multi-class on SAR/optical features.

Uncertainty:
    - High cloud fraction -> increased uncertainty (missing optical evidence)
    - Uncertainty boosted by cloud_mask_frac: u += 0.3 * cloud_frac
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

SAR_FEATURES = [
    "sar_vv_db", "sar_vh_db", "water_index_sar", "ndwi", "ndvi", "cloud_mask_frac"
]

# Flood/dry backscatter thresholds from Twele et al. (2016)
VV_FLOOD_THRESHOLD = -13.0   # dB: below this strongly suggests open water
NDWI_FLOOD_THRESHOLD = 0.15  # above: water likely


class SatelliteAgent(BaseAgent):

    def __init__(
        self,
        model_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="satellite_agent",
            modality="Sentinel-1/2",
            model_ckpt=str(model_path) if model_path else "rule-based",
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
        sar_row = context.get("sar_row")
        if sar_row is None:
            raise ModalityMissingError("SAR/Sentinel row not available.")

        feats = []
        for feat in SAR_FEATURES:
            val = sar_row.get(feat)
            feats.append(float(val) if val is not None else 0.0)

        feat_arr = np.array(feats, dtype=np.float64).reshape(1, -1)
        manifest_id = context.get("manifest_id_s1", "unknown")

        if self.model is not None and self._is_fitted:
            proba = self.model.predict_proba(feat_arr)[0]
        else:
            proba = self._rule_based(feats)

        cloud_frac = float(sar_row.get("cloud_mask_frac", 0.0))
        uncertainty = float(1.0 - np.max(proba))
        uncertainty = min(uncertainty + 0.3 * cloud_frac, 0.95)
        return proba, uncertainty, manifest_id, SAR_FEATURES

    @staticmethod
    def _rule_based(feats: list[float]) -> np.ndarray:
        feat_dict = dict(zip(SAR_FEATURES, feats))
        vv = feat_dict.get("sar_vv_db", -8.0)
        ndwi = feat_dict.get("ndwi", -0.2)
        water_idx = feat_dict.get("water_index_sar", 7.0)

        # VV < VV_FLOOD_THRESHOLD strongly suggests flood water (Twele 2016)
        flood_signal_sar = max(0.0, (VV_FLOOD_THRESHOLD - vv) / 5.0)
        # NDWI > 0.15 confirms open water from optical
        flood_signal_opt = max(0.0, (ndwi - NDWI_FLOOD_THRESHOLD) / 0.5)
        combined = (flood_signal_sar + flood_signal_opt) / 2.0

        logit_dry       = max(0.0, 3.0 - combined * 5.0)
        logit_saturated = max(0.0, 2.0 - abs(combined - 0.3) * 5.0)
        logit_surface   = max(0.0, 2.0 - abs(combined - 0.6) * 4.0)
        logit_inund     = max(0.0, combined * 4.0 - 1.0)

        logits = np.array([logit_dry, logit_saturated, logit_surface, logit_inund])
        logits -= logits.max()
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum()
