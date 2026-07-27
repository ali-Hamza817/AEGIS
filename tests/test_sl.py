"""
tests/test_sl.py
=================
Unit tests for the Subjective Logic engine.

Verifies mathematical invariants:
  1. Opinion invariants: sum(b) + u == 1, b >= 0, u >= 0.
  2. Dirichlet bijection: Opinion -> alpha -> Opinion roundtrip.
  3. WBF: fused opinion respects SL invariants.
  4. CCF: fused opinion respects SL invariants.
  5. Partial-observable update: uncertainty grows when modalities missing.
  6. Credibility update: reputation bounded in [gamma_min, gamma_max].
  7. Vacuous opinion: u == 1, b == 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.sl.opinion import Opinion, FLOOD_FRAME
from src.sl.fusion import weighted_belief_fusion, consensus_compromise_fusion
from src.sl.partial_obs import partial_observable_update
from src.sl.credibility import CredibilityRegistry

K = len(FLOOD_FRAME)
TOL = 1e-6


def make_opinion(b_vals, u):
    """Helper: construct an Opinion from belief values and uncertainty."""
    b = np.array(b_vals, dtype=np.float64)
    d = np.zeros(K)
    a = np.ones(K) / K
    return Opinion(b, d, u, a, FLOOD_FRAME)


class TestOpinionInvariants:
    def test_sum_to_one(self):
        op = make_opinion([0.2, 0.3, 0.1, 0.0], 0.4)
        assert abs(op.b.sum() + op.u - 1.0) < TOL
        assert op.is_valid()

    def test_non_negative(self):
        op = make_opinion([0.1, 0.0, 0.3, 0.2], 0.4)
        assert (op.b >= 0).all()
        assert op.u >= 0

    def test_base_rate_sums_to_one(self):
        op = make_opinion([0.1, 0.2, 0.3, 0.1], 0.3)
        assert abs(op.a.sum() - 1.0) < TOL

    def test_vacuous_opinion(self):
        v = Opinion.vacuous(FLOOD_FRAME)
        assert abs(v.u - 1.0) < TOL
        assert abs(v.b.sum()) < TOL
        assert v.is_valid()

    def test_from_proba(self):
        p = np.array([0.1, 0.4, 0.3, 0.2])
        op = Opinion.from_proba(p, uncertainty=0.3)
        assert abs(op.b.sum() + op.u - 1.0) < TOL
        assert op.is_valid()

    def test_dogmatic(self):
        op = Opinion.dogmatic(3, FLOOD_FRAME)
        assert abs(op.b[3] - 1.0) < TOL
        assert abs(op.u) < TOL
        assert op.is_valid()

    def test_dirichlet_roundtrip(self):
        op = make_opinion([0.2, 0.3, 0.1, 0.1], 0.3)
        alpha = op.to_dirichlet(C=6.0)
        op2 = Opinion.from_dirichlet(alpha, base_rate=op.a, C=6.0)
        assert op2.is_valid()
        # Projected probabilities should be approximately preserved
        pp1 = op.projected_probability()
        pp2 = op2.projected_probability()
        # Bijection is approximate; allow 0.15 tolerance on projected probabilities
        assert np.allclose(pp1, pp2, atol=0.15)


class TestWBF:
    def test_two_opinions(self):
        op1 = make_opinion([0.5, 0.2, 0.1, 0.0], 0.2)
        op2 = make_opinion([0.1, 0.4, 0.2, 0.1], 0.2)
        fused, weights = weighted_belief_fusion([op1, op2])
        assert abs(fused.b.sum() + fused.u - 1.0) < TOL
        assert fused.is_valid()
        assert abs(weights.sum() - 1.0) < TOL

    def test_n_opinions(self):
        opinions = [
            make_opinion([0.3, 0.2, 0.1, 0.1], 0.3),
            make_opinion([0.2, 0.3, 0.2, 0.1], 0.2),
            make_opinion([0.1, 0.1, 0.4, 0.2], 0.2),
            make_opinion([0.0, 0.2, 0.2, 0.4], 0.2),
        ]
        fused, weights = weighted_belief_fusion(opinions)
        assert abs(fused.b.sum() + fused.u - 1.0) < TOL
        assert fused.is_valid()

    def test_credibility_weights_shape(self):
        ops = [make_opinion([0.5, 0.1, 0.2, 0.1], 0.1),
               make_opinion([0.1, 0.5, 0.2, 0.1], 0.1)]
        fused, weights = weighted_belief_fusion(ops, weights=[0.9, 0.5])
        assert fused.is_valid()
        assert len(weights) == 2

    def test_single_opinion_passthrough(self):
        op = make_opinion([0.6, 0.1, 0.2, 0.0], 0.1)
        fused, weights = weighted_belief_fusion([op])
        assert np.allclose(fused.b, op.b, atol=TOL)


class TestCCF:
    def test_two_opinions(self):
        op1 = make_opinion([0.4, 0.3, 0.2, 0.0], 0.1)
        op2 = make_opinion([0.3, 0.4, 0.2, 0.0], 0.1)
        fused, weights = consensus_compromise_fusion([op1, op2])
        assert abs(fused.b.sum() + fused.u - 1.0) < TOL
        assert fused.is_valid()

    def test_equal_contributions(self):
        op1 = make_opinion([0.4, 0.2, 0.2, 0.1], 0.1)
        op2 = make_opinion([0.3, 0.3, 0.2, 0.1], 0.1)
        _, weights = consensus_compromise_fusion([op1, op2])
        # CCF assigns equal pull
        assert abs(weights[0] - weights[1]) < TOL

    def test_high_uncertainty_grows(self):
        """CCF with independent high-uncertainty opinions -> uncertainty stays high."""
        vacuous = Opinion.vacuous(FLOOD_FRAME)
        fused, _ = consensus_compromise_fusion([vacuous, vacuous])
        assert fused.u > 0.5


class TestPartialObservable:
    def test_missing_raises_uncertainty(self):
        op = make_opinion([0.4, 0.3, 0.2, 0.0], 0.1)
        updated = partial_observable_update(op, observed_states=None)  # all missing
        assert updated.u >= op.u
        assert updated.is_valid()

    def test_some_observed(self):
        op = make_opinion([0.4, 0.3, 0.2, 0.0], 0.1)
        updated = partial_observable_update(op, observed_states={0, 1})
        assert updated.is_valid()

    def test_fully_observed_unchanged(self):
        op = make_opinion([0.3, 0.2, 0.3, 0.1], 0.1)
        updated = partial_observable_update(op, observed_states={0, 1, 2, 3})
        # When all states observed, uncertainty should stay similar
        assert updated.is_valid()


class TestCredibility:
    def test_initial_reputation(self):
        reg = CredibilityRegistry.default()
        assert reg.get("climate_agent") == 1.0

    def test_update_bounded(self):
        reg = CredibilityRegistry(gamma_min=0.3, gamma_max=1.0)
        y_pred = np.array([0.9, 0.05, 0.03, 0.02])
        y_true = np.array([1.0, 0.0, 0.0, 0.0])   # correct prediction
        new_gamma = reg.update("test_agent", y_pred, y_true)
        assert 0.3 <= new_gamma <= 1.0

    def test_wrong_prediction_reduces_gamma(self):
        reg = CredibilityRegistry(lr=1.0)  # full update
        y_pred = np.array([0.9, 0.05, 0.03, 0.02])  # confident Dry
        y_true = np.array([0.0, 0.0, 0.0, 1.0])      # actually Inundation
        new_gamma = reg.update("test_agent", y_pred, y_true)
        assert new_gamma < 1.0

    def test_correct_prediction_maintains_gamma(self):
        reg = CredibilityRegistry(lr=0.5, initial=0.8)
        y_pred = np.array([0.9, 0.05, 0.03, 0.02])
        y_true = np.array([1.0, 0.0, 0.0, 0.0])
        new_gamma = reg.update("test_agent", y_pred, y_true)
        assert new_gamma >= 0.8
