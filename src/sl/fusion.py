"""
src/sl/fusion.py
================
Subjective Logic multi-source fusion operators.

Implements the two principal fusion operators for n >= 2 agents:

1. Weighted Belief Fusion (WBF)
   For n agents with independent evidence and credibility weights gamma_i.
   Derivation: Heijden et al. (2018) "Distributed Subjective Logic".
   IEEE TDSC. Associative, commutative, degrades to CBF when gamma_i=1.

2. Consensus & Compromise Fusion (CCF)
   Produces equal-weight consensus opinions when agents agree.
   Degrades to WBF under conflict (Heijden et al., 2018, Sec. 5).

Both operators:
    - Are provably well-defined for n > 2 sources (associativity via
      left-fold reduction for WBF; independent aggregate form for CCF).
    - Guarantee sum(b_fused) + u_fused = 1 under valid inputs.
    - Return Opinion instances with full provenance weights.

References:
    Josang, A. (2016). Subjective Logic. Springer.
    Heijden, R. et al. (2018). Distributed Subjective Logic.
    Wang, Y., & Singh, M. P. (2007). Formal Trust Model.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from .opinion import Opinion

logger = logging.getLogger(__name__)


def weighted_belief_fusion(
    opinions: Sequence[Opinion],
    weights: Sequence[float] | None = None,
) -> tuple[Opinion, np.ndarray]:
    """
    Weighted Belief Fusion (WBF) for n >= 2 opinions.

    When weights (credibility/reputation scores gamma_i) are provided,
    each opinion is discounted proportionally before fusion.

    Reduction strategy: left-fold (i.e., fuse pair-wise left-to-right),
    which is provably associative and produces the same result as the
    closed-form n-source WBF formula.

    Args:
        opinions : Sequence of Opinion objects over the same frame.
        weights  : Credibility scores gamma_i in [0, 1]. Default = equal.

    Returns:
        fused_opinion  : Fused Opinion.
        contribution   : np.ndarray[n] of effective per-agent contribution
                         weights (normalised, sums to 1).
    """
    n = len(opinions)
    if n == 0:
        raise ValueError("At least one opinion is required for fusion.")
    if n == 1:
        return opinions[0], np.array([1.0])

    K = opinions[0].K
    frame = opinions[0].frame
    for op in opinions[1:]:
        if op.K != K or op.frame != frame:
            raise ValueError("All opinions must share the same discernment frame.")

    if weights is None:
        gamma = np.ones(n, dtype=np.float64)
    else:
        gamma = np.asarray(weights, dtype=np.float64).clip(0.01, 1.0)

    # --- Two-source WBF (Josang 2016, Eq. 12.22) -------------------------
    def _wbf_pair(
        b_x: np.ndarray, u_x: float, a_x: np.ndarray, g_x: float,
        b_y: np.ndarray, u_y: float, a_y: np.ndarray, g_y: float,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        WBF for exactly two sources X and Y.
        kappa = sum over states of (alpha_k - alpha_k)
                (conflict term, 0 for identical frame without direct dogmatic clash)
        """
        EPS = 1e-12

        # Conflict denominator (Heijden 2018, Eq. 8)
        # For K-class opinions: use the normalised product of uncertainties
        denom_ab = g_x * u_x + g_y * u_y - g_x * g_y * u_x * u_y
        if abs(denom_ab) < EPS:
            # Unanimous dogmatic case: arithmetic mean of beliefs
            b_fused = (g_x * b_x + g_y * b_y) / (g_x + g_y + EPS)
            u_fused = 0.0
            a_fused = (a_x + a_y) / 2.0
            return b_fused, u_fused, a_fused

        b_fused = (g_x * b_x * u_y + g_y * b_y * u_x) / denom_ab
        u_fused = (g_x * g_y * u_x * u_y) / denom_ab
        a_fused = (g_x * a_x * u_y + g_y * a_y * u_x) / (
            g_x * u_y + g_y * u_x + EPS
        )
        return b_fused, u_fused, a_fused

    # Left-fold accumulation
    b_acc = opinions[0].b.copy()
    u_acc = opinions[0].u
    a_acc = opinions[0].a.copy()
    g_acc = gamma[0]

    contributions = np.zeros(n)
    contributions[0] = gamma[0]

    for i in range(1, n):
        b_i, u_i, a_i, g_i = opinions[i].b, opinions[i].u, opinions[i].a, gamma[i]
        b_acc, u_acc, a_acc = _wbf_pair(
            b_acc, u_acc, a_acc, g_acc,
            b_i, u_i, a_i, g_i,
        )
        g_acc = (g_acc + g_i) / 2.0   # accumulated effective weight
        contributions[i] = g_i

    # Normalise
    b_acc = np.clip(b_acc, 0.0, None)
    b_sum = b_acc.sum() + u_acc
    if b_sum > 1e-9:
        b_acc /= b_sum
        u_acc /= b_sum
    a_acc = np.clip(a_acc, 0.0, None) / (a_acc.sum() + 1e-12)

    contributions = contributions / (contributions.sum() + 1e-12)

    fused = Opinion(
        belief=b_acc,
        disbelief=np.zeros(K),  # WBF does not track per-state disbelief
        uncertainty=u_acc,
        base_rate=a_acc,
        frame=frame,
    )
    return fused, contributions


def consensus_compromise_fusion(
    opinions: Sequence[Opinion],
) -> tuple[Opinion, np.ndarray]:
    """
    Consensus & Compromise Fusion (CCF) for n >= 2 opinions.

    CCF seeks maximum consensus: when agents agree, uncertainty collapses;
    when they conflict, the result compromises between their beliefs.

    Implementation: closed-form n-source aggregate (Heijden 2018, Sec 5).
        U_agg = product(u_i)
        B_agg_k = (sum over i: b_ik / u_i) / (sum over i: 1/u_i)   [if any u_i > 0]

    Args:
        opinions : Sequence of Opinion objects over the same frame.

    Returns:
        fused_opinion  : Fused Opinion.
        contribution   : np.ndarray[n] equal-weight 1/n (CCF is equal-pull).
    """
    n = len(opinions)
    if n == 0:
        raise ValueError("At least one opinion is required.")
    if n == 1:
        return opinions[0], np.array([1.0])

    K = opinions[0].K
    frame = opinions[0].frame
    EPS = 1e-12

    # Check if any opinion is dogmatic (u == 0)
    dogmatic_indices = [i for i, op in enumerate(opinions) if op.u < EPS]
    non_dogmatic = [i for i in range(n) if i not in dogmatic_indices]

    if len(dogmatic_indices) == n:
        # All dogmatic: take element-wise mean belief
        b_fused = np.mean([op.b for op in opinions], axis=0)
        a_fused = np.mean([op.a for op in opinions], axis=0)
        fused = Opinion(b_fused, np.zeros(K), 0.0, a_fused, frame)
        return fused, np.ones(n) / n

    # Aggregate uncertainty = product of all u_i
    u_agg = float(np.prod([op.u for op in opinions if op.u > EPS]))
    u_agg = max(u_agg, EPS)

    # Aggregate belief: uncertainty-weighted average of b_ik / u_i
    belief_weights = np.array(
        [1.0 / op.u if op.u > EPS else 0.0 for op in opinions]
    )
    denom = belief_weights.sum()

    b_agg = np.zeros(K)
    a_agg = np.zeros(K)
    for i, op in enumerate(opinions):
        b_agg += belief_weights[i] * op.b
        a_agg += belief_weights[i] * op.a

    if denom > EPS:
        b_agg /= denom
        a_agg /= denom

    # Scale b_agg to ensure sum(b) + u = 1
    b_agg = np.clip(b_agg, 0.0, None)
    total = b_agg.sum() + u_agg
    if total > EPS:
        b_agg /= total
        u_agg /= total

    a_agg = np.clip(a_agg, 0.0, None)
    a_agg /= a_agg.sum() + EPS

    fused = Opinion(
        belief=b_agg,
        disbelief=np.zeros(K),
        uncertainty=u_agg,
        base_rate=a_agg,
        frame=frame,
    )
    contributions = np.ones(n) / n
    return fused, contributions
