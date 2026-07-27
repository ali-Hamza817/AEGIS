"""
src/agents/landcover_agent.py
==============================
Land Cover Agent — ESA WorldCover + Copernicus DEM topography.

Feature set:
    - land_cover      : ESA WorldCover class (0-5)
    - slope_deg       : terrain slope (degrees, from SRTM)
    - impervious_frac : impervious surface fraction (0-1)
    - elevation_m     : elevation above sea level

Rationale:
    - High impervious fraction → rain cannot infiltrate → surface runoff
    - Low slope + low elevation → water accumulates → inundation risk
    - Land cover class 4 (Built-up) → high impervious proxy

These are static features; the agent contributes a stable prior over the
flood-state frame that persists across time unless updated with precip.
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

LC_FEATURES = ["land_cover", "slope_deg", "impervious_frac", "elevation_m"]


class LandCoverAgent(BaseAgent):

    def __init__(
        self,
        model_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name="landcover_agent",
            modality="WorldCover+DEM",
            model_ckpt=str(model_path) if model_path else "rule-based",
            default_uncertainty=0.55,   # static features carry higher uncertainty
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
        lc_row = context.get("lc_row")
        if lc_row is None:
            raise ModalityMissingError("Land cover / DEM row not in context.")

        feats = [float(lc_row.get(f, 0.0)) for f in LC_FEATURES]
        feat_arr = np.array(feats, dtype=np.float64).reshape(1, -1)
        manifest_id = context.get("manifest_id_lc", "unknown")

        if self.model is not None and self._is_fitted:
            proba = self.model.predict_proba(feat_arr)[0]
        else:
            proba = self._rule_based(feats)

        # Static features → always higher base uncertainty
        uncertainty = max(float(1.0 - np.max(proba)), 0.35)
        return proba, uncertainty, manifest_id, LC_FEATURES

    @staticmethod
    def _rule_based(feats: list[float]) -> np.ndarray:
        feat_dict = dict(zip(LC_FEATURES, feats))
        imp = float(feat_dict.get("impervious_frac", 0.3))
        slope = float(feat_dict.get("slope_deg", 5.0))
        elev = float(feat_dict.get("elevation_m", 10.0))
        lc = int(feat_dict.get("land_cover", 0))

        # Topographic wetness proxy: high imp + low slope + low elev = high risk
        topo_score = imp * max(0.0, 1.0 - slope / 10.0) * max(0.0, 1.0 - elev / 20.0)
        # Built-up class bonus
        if lc == 4:
            topo_score = min(1.0, topo_score + 0.25)

        logit_dry       = max(0.0, 3.0 * (1.0 - topo_score))
        logit_saturated = max(0.0, 2.0 - abs(topo_score - 0.3) * 5.0)
        logit_surface   = max(0.0, 2.0 - abs(topo_score - 0.65) * 4.0)
        logit_inund     = max(0.0, topo_score * 3.0 - 1.0)

        logits = np.array([logit_dry, logit_saturated, logit_surface, logit_inund])
        logits -= logits.max()
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum()
