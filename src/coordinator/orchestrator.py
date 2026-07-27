"""
src/coordinator/orchestrator.py
================================
AEGIS Deterministic SL Orchestrator — NO LLM calls.

The coordinator implements the evidence-routing policy described in Section 6
of the AEGIS specification:

    1. For each enabled agent: call agent.emit(cell_id, date, context).
    2. If ModalityMissingError: apply partial_observable_update (Kaplan 2015).
    3. Compute all-pairs Jensen-Shannon divergence over belief distributions.
    4. Route to CCF if max_JS < tau_LOW (high agreement).
    5. Route to WBF (credibility-weighted) if max_JS >= tau_LOW.
    6. Log everything to DuckDB opinion_log.

This is the only place in AEGIS where fusion decisions are made.
All routing is deterministic given the opinions and credibility scores.
No hallucination, no probabilistic sampling, no LLM.

References:
    Heijden et al. (2018) — WBF/CCF operator definitions and proofs.
    Kaplan et al. (2015) — partial-observable update.
    Wang & Singh (2007)  — credibility discounting.
    Nielsen & Parsons (2006) — conflict-resolution in multi-agent systems.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timezone
from typing import Any, Sequence

import numpy as np

from src.agents.base import BaseAgent, ProvenanceRecord, ModalityMissingError
from src.sl.opinion import Opinion, FLOOD_FRAME
from src.sl.fusion import weighted_belief_fusion, consensus_compromise_fusion
from src.sl.partial_obs import partial_observable_update, vacuous_from_prior
from src.sl.credibility import CredibilityRegistry

logger = logging.getLogger(__name__)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon divergence between two probability distributions.
    JS(p, q) = 0.5 * KL(p || M) + 0.5 * KL(q || M)  where M = (p+q)/2.
    Returns value in [0, log(2)] = [0, ~0.693].
    Normalised to [0, 1] by dividing by log(2).
    """
    eps = 1e-12
    p = np.clip(p, eps, None); p /= p.sum()
    q = np.clip(q, eps, None); q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m + eps))
    kl_qm = np.sum(q * np.log(q / m + eps))
    js = 0.5 * (kl_pm + kl_qm)
    return float(js / np.log(2.0 + eps))   # normalised to [0, 1]


def _max_pairwise_js(opinions: list[Opinion]) -> float:
    """Return the maximum pairwise JS divergence across all agent opinion pairs."""
    n = len(opinions)
    if n < 2:
        return 0.0
    projs = [op.projected_probability() for op in opinions]
    max_js = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            js = _js_divergence(projs[i], projs[j])
            max_js = max(max_js, js)
    return max_js


class OrchestratorResult:
    """Container for coordinator output."""
    def __init__(
        self,
        run_id: str,
        cell_id: int,
        target_date: date,
        fused_opinion: Opinion,
        fusion_operator: str,
        agent_opinions: list[tuple[str, Opinion]],
        contributions: np.ndarray,
        provenance: list[ProvenanceRecord],
        max_js: float,
    ) -> None:
        self.run_id = run_id
        self.cell_id = cell_id
        self.target_date = target_date
        self.fused_opinion = fused_opinion
        self.fusion_operator = fusion_operator
        self.agent_opinions = agent_opinions     # [(agent_name, opinion)]
        self.contributions = contributions       # normalised weights per agent
        self.provenance = provenance
        self.max_js = max_js

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "cell_id": self.cell_id,
            "target_date": str(self.target_date),
            "fused_opinion": self.fused_opinion.to_dict(),
            "fusion_operator": self.fusion_operator,
            "max_js_divergence": self.max_js,
            "agent_contributions": {
                name: float(w)
                for (name, _), w in zip(self.agent_opinions, self.contributions)
            },
            "provenance": [p.to_dict() for p in self.provenance],
        }


class SLOrchestrator:
    """
    Deterministic Subjective Logic coordinator.

    Args:
        agents              : List of specialist agents.
        credibility_registry: Per-agent credibility scores.
        tau_low             : JS threshold below which CCF is used (high agreement).
        tau_high            : JS threshold above which WBF is used (high conflict).
        c_dirichlet         : Dirichlet evidence weight C (default 6.0).
        run_id              : Experiment identifier.
        db_conn             : Optional DuckDB connection for opinion logging.
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        credibility_registry: CredibilityRegistry | None = None,
        tau_low: float = 0.1,
        tau_high: float = 0.3,
        c_dirichlet: float = 6.0,
        run_id: str | None = None,
        db_conn: Any | None = None,
    ) -> None:
        self.agents = agents
        self.registry = credibility_registry or CredibilityRegistry.default()
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.c_dirichlet = c_dirichlet
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.db_conn = db_conn

    def route(
        self,
        cell_id: int,
        target_date: date,
        context: dict[str, Any],
        enabled_agents: list[str] | None = None,
    ) -> OrchestratorResult:
        """
        Main entry point: collect opinions from all enabled agents,
        fuse them via WBF or CCF, log to DB, and return result.

        Args:
            cell_id       : Grid cell identifier.
            target_date   : Prediction target date.
            context       : Dict of per-modality data rows (from DuckDB query).
            enabled_agents: Optional subset of agent names to enable.
                            If None, all agents are enabled.
        """
        agent_pool = self.agents
        if enabled_agents is not None:
            agent_pool = [a for a in self.agents if a.name in enabled_agents]

        collected_opinions: list[Opinion] = []
        collected_names: list[str] = []
        collected_prov: list[ProvenanceRecord] = []
        credibility_weights: list[float] = []

        for agent in agent_pool:
            gamma = self.registry.get(agent.name)
            try:
                opinion, prov = agent.emit(cell_id, target_date, context, credibility_gamma=gamma)

                # If agent raised missing internally -> partial-obs already handled in base.py
                if prov.modality_missing:
                    # Apply formal partial-obs projection
                    prior_opinion = opinion   # base.py returned vacuous for missing
                    opinion = partial_observable_update(
                        prior=prior_opinion,
                        observed_states=None,   # all missing
                        C=self.c_dirichlet,
                    )

            except Exception as exc:
                logger.error(
                    "Agent '%s' failed unexpectedly (cell=%d, date=%s): %s",
                    agent.name, cell_id, target_date, exc,
                )
                prior = Opinion.vacuous(frame=FLOOD_FRAME)
                opinion = partial_observable_update(prior, observed_states=None, C=self.c_dirichlet)
                prov = ProvenanceRecord(
                    agent_name=agent.name,
                    modality=agent.modality,
                    manifest_id="error",
                    model_ckpt=agent.model_ckpt,
                    modality_missing=True,
                    source_ref="AGENT_ERROR",
                    features_used=[],
                    cell_id=cell_id,
                    target_date=target_date,
                    raw_proba=[0.25] * 4,
                    uncertainty_u=opinion.u,
                    credibility_gamma=gamma,
                )

            collected_opinions.append(opinion)
            collected_names.append(agent.name)
            collected_prov.append(prov)
            credibility_weights.append(gamma)

        # ---------------------------------------------------------------
        # Conflict detection
        # ---------------------------------------------------------------
        if len(collected_opinions) == 0:
            logger.warning("No agent opinions collected. Returning vacuous opinion.")
            fused = Opinion.vacuous(frame=FLOOD_FRAME)
            fusion_op = "NONE"
            contributions = np.array([])
            max_js = 0.0
        elif len(collected_opinions) == 1:
            fused = collected_opinions[0]
            fusion_op = "SINGLE"
            contributions = np.array([1.0])
            max_js = 0.0
        else:
            max_js = _max_pairwise_js(collected_opinions)
            gamma_arr = np.array(credibility_weights, dtype=np.float64)

            if max_js < self.tau_low:
                # High agreement → Consensus & Compromise Fusion
                fused, contributions = consensus_compromise_fusion(collected_opinions)
                fusion_op = "CCF"
                logger.debug(
                    "cell=%d date=%s: CCF (max_JS=%.3f < tau_low=%.3f)",
                    cell_id, target_date, max_js, self.tau_low,
                )
            else:
                # Conflict → Weighted Belief Fusion with credibility weights
                fused, contributions = weighted_belief_fusion(collected_opinions, gamma_arr)
                fusion_op = "WBF"
                logger.debug(
                    "cell=%d date=%s: WBF (max_JS=%.3f, gamma=%s)",
                    cell_id, target_date, max_js,
                    np.array2string(gamma_arr, precision=2),
                )

        result = OrchestratorResult(
            run_id=self.run_id,
            cell_id=cell_id,
            target_date=target_date,
            fused_opinion=fused,
            fusion_operator=fusion_op,
            agent_opinions=list(zip(collected_names, collected_opinions)),
            contributions=contributions,
            provenance=collected_prov,
            max_js=max_js,
        )

        # ---------------------------------------------------------------
        # Persist to opinion_log
        # ---------------------------------------------------------------
        if self.db_conn is not None:
            self._log_opinions(result)

        return result

    def _log_opinions(self, result: OrchestratorResult) -> None:
        """Write all agent opinions and fused opinion to opinion_log table."""
        now = datetime.now(tz=timezone.utc).isoformat()
        rows = []

        for (name, op), prov in zip(result.agent_opinions, result.provenance):
            rows.append((
                str(uuid.uuid4()),
                result.run_id,
                result.cell_id,
                result.target_date,
                name,
                "emit",
                None,                      # fusion_op not applicable for agent
                *op.b.tolist(),
                op.u,
                prov.credibility_gamma,
                prov.modality_missing,
                prov.manifest_id,
                prov.model_ckpt,
                now,
            ))

        # Fused opinion
        fused = result.fused_opinion
        rows.append((
            str(uuid.uuid4()),
            result.run_id,
            result.cell_id,
            result.target_date,
            "coordinator",
            "fused",
            result.fusion_operator,
            *fused.b.tolist(),
            fused.u,
            1.0,     # coordinator credibility = 1.0
            False,
            "fused",
            "SLOrchestrator",
            now,
        ))

        try:
            self.db_conn.executemany(
                """INSERT INTO opinion_log
                   (log_id, run_id, cell_id, date, agent, stage, fusion_op,
                    b_dry, b_saturated, b_surfaceflow, b_inundation,
                    uncertainty_u, credibility_gamma, modality_missing,
                    manifest_id, model_ckpt, logged_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            self.db_conn.commit()
        except Exception as exc:
            logger.warning("Failed to log opinions to DB: %s", exc)
