"""
src/prediction/evidential_head.py
==================================
Evidential prediction head for AEGIS.

Maps the fused SL opinion tensor (from coordinator output) to:
  1. flood_state     : argmax class prediction (0-3)
  2. state_proba     : probability distribution over flood states
  3. depth_m         : water depth regression (m), trained with LightGBM

Input feature vector construction from OrchestratorResult:
    [b_dry, b_saturated, b_surfaceflow, b_inundation,  # belief masses
     u,                                                  # uncertainty
     gamma_climate, gamma_sat, gamma_lc, gamma_aq, gamma_doc,  # credibilities
     max_js_divergence]                                  # conflict level

Total feature vector: 11 dimensions.

Training procedure:
    - Labels: EMSR flood_state (0-3) and flood_depth_m from truth table.
    - Classifier: LightGBM multi-class.
    - Regressor: LightGBM regression on log(depth_m + 0.01).
    - Train/val/test split: temporal (no data leakage across dates).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "b_dry", "b_saturated", "b_surfaceflow", "b_inundation",
    "uncertainty_u", "max_js",
    "gamma_climate", "gamma_sat", "gamma_lc", "gamma_aq", "gamma_doc",
]
N_FEATURES = len(FEATURE_NAMES)


def opinion_to_feature_vector(
    orchestrator_result: Any,
    credibility_dict: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Convert an OrchestratorResult to a fixed-length feature vector.

    Args:
        orchestrator_result : OrchestratorResult from SLOrchestrator.route()
        credibility_dict    : Optional override for per-agent gamma values.

    Returns:
        np.ndarray of shape (11,)
    """
    op = orchestrator_result.fused_opinion
    feats = list(op.b) + [op.u, orchestrator_result.max_js]

    # Per-agent credibilities (in canonical order)
    agent_order = ["climate_agent", "satellite_agent", "landcover_agent",
                   "airquality_agent", "docint_agent"]
    prov_map = {p.agent_name: p.credibility_gamma for p in orchestrator_result.provenance}
    if credibility_dict:
        prov_map.update(credibility_dict)

    for agent_name in agent_order:
        feats.append(float(prov_map.get(agent_name, 0.5)))

    return np.array(feats, dtype=np.float64)


class EvidentialHead:
    """
    LightGBM-backed evidential prediction head.

    Trained separately for:
        - Multi-class flood state classification (lgbm_clf)
        - Flood depth regression (lgbm_reg)

    Can operate in rule-based mode (no trained model) for immediate
    pipeline evaluation using the projected probability of the fused opinion.
    """

    def __init__(
        self,
        clf_path: str | Path | None = None,
        reg_path: str | Path | None = None,
    ) -> None:
        self.clf = None
        self.reg = None
        self._clf_fitted = False
        self._reg_fitted = False

        if clf_path and Path(clf_path).exists():
            with open(clf_path, "rb") as f:
                self.clf = pickle.load(f)
            self._clf_fitted = True
            logger.info("EvidentialHead: classifier loaded from %s", clf_path)

        if reg_path and Path(reg_path).exists():
            with open(reg_path, "rb") as f:
                self.reg = pickle.load(f)
            self._reg_fitted = True
            logger.info("EvidentialHead: regressor loaded from %s", reg_path)

    def predict(
        self,
        feature_vector: np.ndarray,
        fused_opinion: Any | None = None,
    ) -> dict[str, Any]:
        """
        Predict flood state and depth.

        Args:
            feature_vector : Shape (11,) or (1, 11).
            fused_opinion  : Optional Opinion object (used for projected proba fallback).

        Returns:
            dict with keys: flood_state, state_proba, depth_m, uncertainty_u
        """
        fv = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)

        if self._clf_fitted and self.clf is not None:
            state_proba = self.clf.predict_proba(fv)[0]
            flood_state = int(np.argmax(state_proba))
        else:
            # Fallback: use projected probability from the fused opinion
            if fused_opinion is not None:
                state_proba = fused_opinion.projected_probability()
            else:
                state_proba = fv[0, :4]
                state_proba = np.clip(state_proba, 0, None)
                state_proba /= state_proba.sum() + 1e-12
            flood_state = int(np.argmax(state_proba))

        if self._reg_fitted and self.reg is not None:
            log_depth = float(self.reg.predict(fv)[0])
            depth_m = max(0.0, np.expm1(log_depth))
        else:
            # Rule-based depth: proportional to inundation belief
            inund_b = float(fv[0, 3])   # b_inundation
            depth_m = max(0.0, inund_b * 3.5)

        uncertainty_u = float(fv[0, 4])

        return {
            "flood_state": flood_state,
            "state_proba": state_proba.tolist(),
            "depth_m": depth_m,
            "uncertainty_u": uncertainty_u,
        }

    def fit_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_path: str | Path | None = None,
    ) -> None:
        """Train LightGBM multi-class classifier."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise RuntimeError("lightgbm not installed. Run: pip install lightgbm")

        self.clf = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        self.clf.fit(X, y)
        self._clf_fitted = True
        logger.info("EvidentialHead classifier trained on %d samples.", len(y))

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(self.clf, f)

    def fit_regressor(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_path: str | Path | None = None,
    ) -> None:
        """Train LightGBM depth regressor on log(depth + 0.01)."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise RuntimeError("lightgbm not installed.")

        log_y = np.log1p(np.clip(y, 0.0, None))
        self.reg = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )
        self.reg.fit(X, log_y)
        self._reg_fitted = True
        logger.info("EvidentialHead regressor trained on %d samples.", len(y))

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(self.reg, f)
