"""
src/sl/credibility.py
=====================
Per-agent credibility (reputation) management for SL fusion.

Implements the trust/credibility discounting model from:
    Wang, Y., & Singh, M. P. (2007). Formal Trust Model for Multiagent
    Systems. Proc. IJCAI.

Each agent maintains a moving reputation score gamma in [gamma_min, gamma_max].
After each prediction event (when EMSR ground truth becomes available),
the reputation is updated using a Brier-score inspired feedback loop.

Brier Score for state prediction:
    BS = (1/K) * sum_k (p_k - y_k)^2

Reputation update rule (Wang & Singh 2007, Eq. 4):
    gamma_t+1 = (1 - lr) * gamma_t + lr * (1 - BS)

    where (1 - BS) is the normalised accuracy signal in [0, 1].

The reputation score enters Weighted Belief Fusion (WBF) as the credibility
weight gamma_i, so better-calibrated agents dominate the fused opinion.

Usage:
    registry = CredibilityRegistry.default()
    gamma = registry.get("climate_agent")
    registry.update("climate_agent", predicted_proba, truth_onehot)
    registry.save("results/credibility.json")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_AGENTS = [
    "climate_agent",
    "satellite_agent",
    "landcover_agent",
    "airquality_agent",
    "docint_agent",
]


class CredibilityRegistry:
    """
    Maintains per-agent reputation scores and update counts.

    Attributes:
        gamma      : dict[str, float]   current credibility in [gamma_min, gamma_max]
        update_count: dict[str, int]    number of Brier updates per agent
    """

    def __init__(
        self,
        initial: float = 1.0,
        gamma_min: float = 0.3,
        gamma_max: float = 1.0,
        lr: float = 0.1,
    ) -> None:
        self.initial = initial
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.lr = lr
        self.gamma: dict[str, float] = {}
        self.update_count: dict[str, int] = {}

    @classmethod
    def default(cls) -> "CredibilityRegistry":
        """Construct registry pre-populated with all AEGIS agents."""
        reg = cls()
        for name in DEFAULT_AGENTS:
            reg.gamma[name] = reg.initial
            reg.update_count[name] = 0
        return reg

    def get(self, agent_name: str) -> float:
        """Return current reputation score, initialising if unseen."""
        if agent_name not in self.gamma:
            self.gamma[agent_name] = self.initial
            self.update_count[agent_name] = 0
        return self.gamma[agent_name]

    def update(
        self,
        agent_name: str,
        predicted_proba: np.ndarray,
        truth_onehot: np.ndarray,
    ) -> float:
        """
        Update agent reputation using Brier Score feedback.

        Args:
            agent_name      : Agent identifier.
            predicted_proba : Predicted probability over states (K,).
            truth_onehot    : One-hot ground truth vector (K,).

        Returns:
            New credibility value.
        """
        p = np.asarray(predicted_proba, dtype=np.float64)
        y = np.asarray(truth_onehot, dtype=np.float64)
        p = p / p.sum()    # normalise
        y = y / y.sum()    # should already be binary but normalise defensively

        K = len(p)
        # Multi-class Brier Score (ranges 0 to 2, divide by 2 to normalise 0-1)
        brier = float(np.sum((p - y) ** 2)) / 2.0   # normalised Brier in [0, 1]
        accuracy_signal = 1.0 - brier                 # high is better

        gamma_old = self.get(agent_name)
        gamma_new = (1.0 - self.lr) * gamma_old + self.lr * accuracy_signal
        gamma_new = float(np.clip(gamma_new, self.gamma_min, self.gamma_max))

        self.gamma[agent_name] = gamma_new
        self.update_count[agent_name] = self.update_count.get(agent_name, 0) + 1

        logger.info(
            "CredibilityRegistry: %s  gamma %.3f -> %.3f  (BS=%.3f)",
            agent_name,
            gamma_old,
            gamma_new,
            brier,
        )
        return gamma_new

    def get_all_weights(self, agent_names: list[str]) -> np.ndarray:
        """Return gamma values as numpy array in the order of agent_names."""
        return np.array([self.get(n) for n in agent_names], dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "gamma": dict(self.gamma),
            "update_count": dict(self.update_count),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("CredibilityRegistry saved to %s", path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "CredibilityRegistry":
        with open(path) as f:
            data = json.load(f)
        reg = cls(**kwargs)
        reg.gamma = data.get("gamma", {})
        reg.update_count = data.get("update_count", {})
        return reg

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={v:.3f}" for k, v in self.gamma.items())
        return f"CredibilityRegistry({parts})"
