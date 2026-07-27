"""
src/sl/partial_obs.py
=====================
Partial-Observable SL Update (Kaplan et al., 2015).

When a modality is unavailable for a given (cell_id, date), the agent
cannot form evidence-based belief masses. Instead of silently zero-filling
(which would inject spurious low-uncertainty certainty), we apply the
partial-observable projection rule:

    For state k:
        If evidence on k is available:
            alpha_k' = alpha_k              (keep evidential concentration)
        If evidence on k is NOT available:
            alpha_k' = a_k * u * C          (fall back to prior-scaled uncertainty)

This means:
    - Missing modality -> Dirichlet concentrations revert toward prior a_k
    - Uncertainty u' increases relative to the fully-observed opinion
    - The agent still contributes a valid opinion, but with reduced confidence

Reference:
    Kaplan, S., Celik, E., & Josang, A. (2015).
    Subjective Logic Applied to ISMS. Proc. IFIP SEC.
"""

from __future__ import annotations

import logging

import numpy as np

from .opinion import Opinion, FLOOD_FRAME

logger = logging.getLogger(__name__)


def partial_observable_update(
    prior: Opinion,
    observed_states: set[int] | None = None,
    C: float = 6.0,
) -> Opinion:
    """
    Apply partial-observable projection to an Opinion when some frame
    states lack direct evidence.

    Args:
        prior           : The agent's prior opinion (or last valid opinion).
        observed_states : Set of state indices for which evidence exists.
                          If None or empty -> all states missing (vacuous).
        C               : Dirichlet evidence weight (must match emission C).

    Returns:
        updated Opinion with increased uncertainty for missing states.
    """
    K = prior.K
    if observed_states is None:
        observed_states = set()

    # Current Dirichlet alphas
    alpha = prior.to_dirichlet(C=C)

    alpha_updated = np.zeros(K)
    for k in range(K):
        if k in observed_states:
            alpha_updated[k] = alpha[k]          # keep evidence
        else:
            # Fall back to prior-scaled pseudo-count
            alpha_updated[k] = prior.a[k] * prior.u * C

    # Ensure at least a minimal floor (avoid division by zero in fusion)
    alpha_updated = np.maximum(alpha_updated, prior.a * 1e-3)

    updated = Opinion.from_dirichlet(
        alpha=alpha_updated,
        frame=prior.frame,
        base_rate=prior.a,
        C=C,
    )
    logger.debug(
        "partial_obs_update: u %.3f -> %.3f (missing states: %s)",
        prior.u,
        updated.u,
        sorted(set(range(K)) - observed_states),
    )
    return updated


def vacuous_from_prior(
    base_rate: np.ndarray,
    frame: tuple[str, ...] = FLOOD_FRAME,
) -> Opinion:
    """
    Return the maximum-uncertainty vacuous opinion weighted by base_rate.
    Used as a safe fallback when no prior opinion exists either.
    """
    K = len(frame)
    return Opinion(
        belief=np.zeros(K),
        disbelief=np.zeros(K),
        uncertainty=1.0,
        base_rate=np.asarray(base_rate) / np.asarray(base_rate).sum(),
        frame=frame,
    )
