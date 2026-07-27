"""
src/sl/opinion.py
=================
Subjective Logic (SL) Opinion implementation.

An SL opinion over a frame of discernment Theta = {theta_1, ..., theta_K} is
the tuple omega = (b, d, u, a) where:

    b  : np.ndarray[K]  -- belief mass distribution  (b_i >= 0, sum(b) + u = 1)
    d  : np.ndarray[K]  -- disbelief mass             (d_i >= 0, b_i + d_i <= 1)
    u  : float          -- epistemic uncertainty mass  (u >= 0)
    a  : np.ndarray[K]  -- base-rate (prior) weights   (a_i > 0, sum(a) = 1)

Bijection to/from Dirichlet parameters (Josang 2016, Eq. 3.21):
    alpha_i = b_i * C + a_i * u * C,   where C = |Theta| (or a configurable scale)

Reference:
    Josang, A. (2016). Subjective Logic: A Formalism for Reasoning Under Uncertainty.
    Springer.

    Kaplan, S., et al. (2015). Subjective Logic Applied to ISMS.
    (partial-observable updates, Section 4)

    Heijden, R. et al. (2018). Distributed Subjective Logic.
    IEEE Transactions on Dependable and Secure Computing.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Default discernment frame for flood risk
FLOOD_FRAME = ("Dry", "Saturated", "SurfaceFlow", "Inundation")


class Opinion:
    """
    Represents a Subjective Logic trinomial opinion (b, d, u, a) over
    an ordered discrete frame Theta.

    Invariants enforced on construction:
        - All b_i, d_i, u >= 0
        - sum(b) + u = 1.0 (within tolerance)
        - 0 <= b_i + d_i <= 1 for all i
        - sum(a) = 1.0
        - len(b) == len(d) == len(a) == len(frame)
    """

    TOLERANCE: float = 1e-6

    def __init__(
        self,
        belief: np.ndarray | Sequence[float],
        disbelief: np.ndarray | Sequence[float],
        uncertainty: float,
        base_rate: np.ndarray | Sequence[float],
        frame: Sequence[str] = FLOOD_FRAME,
    ) -> None:
        b = np.asarray(belief, dtype=np.float64)
        d = np.asarray(disbelief, dtype=np.float64)
        u = float(uncertainty)
        a = np.asarray(base_rate, dtype=np.float64)

        K = len(frame)
        if not (b.shape == d.shape == a.shape == (K,)):
            raise ValueError(
                f"belief/disbelief/base_rate must each have length {K} "
                f"(|frame|), got b={b.shape}, d={d.shape}, a={a.shape}."
            )

        # Enforce non-negativity
        b = np.clip(b, 0.0, None)
        d = np.clip(d, 0.0, None)
        u = max(0.0, u)

        # Re-normalise to enforce sum(b) + u == 1
        mass_total = b.sum() + u
        if mass_total > self.TOLERANCE:
            scale = 1.0 / mass_total
            b = b * scale
            u = u * scale

        # Normalise base rate
        a_sum = a.sum()
        if a_sum < self.TOLERANCE:
            a = np.ones(K, dtype=np.float64) / K  # uniform fallback
        else:
            a = a / a_sum

        self.b: np.ndarray = b
        self.d: np.ndarray = d
        self.u: float = u
        self.a: np.ndarray = a
        self.frame: tuple[str, ...] = tuple(frame)
        self.K: int = K

    # ------------------------------------------------------------------
    # Dirichlet bijection
    # ------------------------------------------------------------------

    def to_dirichlet(self, C: float = 6.0) -> np.ndarray:
        """
        Map opinion to Dirichlet concentration parameters.

            alpha_k = b_k * C + a_k * u * C

        C is the evidence weight (default C=|Theta|*1.5, caller may override).
        """
        if C <= 0:
            raise ValueError(f"Dirichlet scale C must be > 0, got {C}")
        return self.b * C + self.a * (self.u * C)

    @classmethod
    def from_dirichlet(
        cls,
        alpha: np.ndarray | Sequence[float],
        frame: Sequence[str] = FLOOD_FRAME,
        base_rate: np.ndarray | Sequence[float] | None = None,
        C: float = 6.0,
    ) -> "Opinion":
        """
        Invert the Dirichlet bijection.

            u  = K * C / sum(alpha)
            b_k = (alpha_k / sum(alpha)) * (1 - u)
            d_k = (sum(alpha) - alpha_k - a_k * u * C) / sum(alpha)  [approx]
        """
        alpha = np.asarray(alpha, dtype=np.float64)
        K = len(frame)
        if base_rate is None:
            base_rate = np.ones(K) / K
        a = np.asarray(base_rate, dtype=np.float64)
        a = a / a.sum()

        S = alpha.sum()
        u = K * C / (S + K * C) if S > 0 else 1.0
        b = (alpha / S) * (1.0 - u) if S > 0 else a * 0.0
        # disbelief: remainder per state
        d = np.zeros(K)
        for k in range(K):
            d[k] = max(0.0, (S - alpha[k]) / S * (1.0 - u) - b.sum() + b[k])
        return cls(b, d, u, a, frame)

    # ------------------------------------------------------------------
    # Projected probability (expectation over Dirichlet)
    # ------------------------------------------------------------------

    def projected_probability(self) -> np.ndarray:
        """E[p_k] = b_k + a_k * u  (Josang 2016, Eq. 3.9)"""
        return self.b + self.a * self.u

    # ------------------------------------------------------------------
    # Diagnostics & serialisation
    # ------------------------------------------------------------------

    def is_valid(self) -> bool:
        """Check all SL invariants."""
        b, d, u, a = self.b, self.d, self.u, self.a
        try:
            assert (b >= 0).all()
            assert (d >= 0).all()
            assert u >= 0
            assert abs(b.sum() + u - 1.0) < self.TOLERANCE
            assert ((b + d) <= 1.0 + self.TOLERANCE).all()
            assert abs(a.sum() - 1.0) < self.TOLERANCE
        except AssertionError:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "frame": list(self.frame),
            "b": self.b.tolist(),
            "d": self.d.tolist(),
            "u": self.u,
            "a": self.a.tolist(),
            "projected_prob": self.projected_probability().tolist(),
        }

    def __repr__(self) -> str:
        pp = self.projected_probability()
        dominant = self.frame[int(np.argmax(pp))]
        return (
            f"Opinion(u={self.u:.3f}, dominant='{dominant}', "
            f"b={np.array2string(self.b, precision=3)})"
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def vacuous(cls, frame: Sequence[str] = FLOOD_FRAME) -> "Opinion":
        """Maximum-uncertainty vacuous opinion: b=0, u=1."""
        K = len(frame)
        return cls(
            belief=np.zeros(K),
            disbelief=np.zeros(K),
            uncertainty=1.0,
            base_rate=np.ones(K) / K,
            frame=frame,
        )

    @classmethod
    def dogmatic(
        cls, state_idx: int, frame: Sequence[str] = FLOOD_FRAME
    ) -> "Opinion":
        """Dogmatic (zero-uncertainty) belief in state_idx."""
        K = len(frame)
        b = np.zeros(K)
        b[state_idx] = 1.0
        return cls(
            belief=b,
            disbelief=1.0 - b,
            uncertainty=0.0,
            base_rate=np.ones(K) / K,
            frame=frame,
        )

    @classmethod
    def from_proba(
        cls,
        proba: np.ndarray | Sequence[float],
        uncertainty: float = 0.3,
        frame: Sequence[str] = FLOOD_FRAME,
    ) -> "Opinion":
        """
        Construct from softmax/probability vector.

        b_k = proba_k * (1 - u),  d_k = (1 - proba_k) * (1 - u)
        """
        p = np.asarray(proba, dtype=np.float64)
        p = p / p.sum()  # normalise
        u = float(np.clip(uncertainty, 0.0, 1.0))
        K = len(p)
        b = p * (1.0 - u)
        d = (1.0 - p) * (1.0 - u)
        a = p.copy()
        return cls(b, d, u, a, frame)
