"""
src/agents/base.py
==================
Abstract base class for all AEGIS specialist agents.

Every agent must implement:
    emit(cell_id, date, context) -> (Opinion, ProvenanceRecord)

The ProvenanceRecord is the atomic unit of explanation — it carries:
  - Which data source was consumed (manifest_id)
  - Whether the modality was missing (triggers partial-observable update)
  - Which model checkpoint produced the opinion
  - All information needed for the explanation dashboard

Design contract:
  - Agents NEVER call each other.
  - Agents NEVER call an LLM.
  - Agents return a valid Opinion even when their modality is missing
    (via partial_observable_update with modality_missing=True flag).
"""

from __future__ import annotations

import abc
import dataclasses
import logging
from datetime import date
from typing import Any

import numpy as np

from src.sl.opinion import Opinion, FLOOD_FRAME

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ProvenanceRecord:
    """
    Audit record emitted alongside every Opinion.

    Fields:
        agent_name      : Unique agent identifier.
        modality        : Data type consumed ('ERA5', 'SAR', 'WorldCover', etc.).
        manifest_id     : FK into the manifest table (data lineage).
        model_ckpt      : Model checkpoint or rule-based tag.
        modality_missing: True if the modality was unavailable (opinion is partial-obs).
        source_ref      : Human-readable description of data source.
        features_used   : Names of feature columns used by this agent.
        cell_id         : Grid cell identifier.
        target_date     : Prediction target date.
        raw_proba       : Raw probability estimate before SL conversion.
        uncertainty_u   : The u component of the emitted Opinion.
        credibility_gamma: Agent's current credibility score.
    """
    agent_name: str
    modality: str
    manifest_id: str
    model_ckpt: str
    modality_missing: bool
    source_ref: str
    features_used: list[str]
    cell_id: int
    target_date: date
    raw_proba: list[float]
    uncertainty_u: float
    credibility_gamma: float = 1.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class ModalityMissingError(Exception):
    """Raised internally when a modality cannot be retrieved."""
    pass


class BaseAgent(abc.ABC):
    """
    Abstract specialist agent.

    Subclasses implement `_compute_proba` which returns:
        proba : np.ndarray[K]  -- multi-class probability over flood states
        uncertainty: float      -- epistemic uncertainty in [0, 1]
        manifest_id: str        -- provenance reference
        features_used: list[str]
    """

    FRAME = FLOOD_FRAME

    def __init__(
        self,
        name: str,
        modality: str,
        model_ckpt: str = "untrained",
        base_rate: np.ndarray | None = None,
        default_uncertainty: float = 0.4,
    ) -> None:
        self.name = name
        self.modality = modality
        self.model_ckpt = model_ckpt
        self.K = len(self.FRAME)
        self.base_rate = (
            base_rate if base_rate is not None
            else np.ones(self.K) / self.K
        )
        self.default_uncertainty = default_uncertainty
        self._is_fitted = False

    @abc.abstractmethod
    def _compute_proba(
        self,
        cell_id: int,
        target_date: date,
        context: dict[str, Any],
    ) -> tuple[np.ndarray, float, str, list[str]]:
        """
        Returns:
            proba         : np.ndarray[K] probability over flood states
            uncertainty   : float epistemic uncertainty in [0, 1]
            manifest_id   : str provenance key
            features_used : list[str] column names used
        Raises:
            ModalityMissingError if modality data unavailable.
        """
        ...

    def emit(
        self,
        cell_id: int,
        target_date: date,
        context: dict[str, Any],
        credibility_gamma: float = 1.0,
    ) -> tuple[Opinion, ProvenanceRecord]:
        """
        Compute and return an SL opinion + provenance record.

        Handles ModalityMissingError by returning a partial-observable
        vacuous opinion (uncertainty = 1, belief = 0).
        """
        missing = False
        manifest_id = "unknown"
        features_used: list[str] = []

        try:
            proba, uncertainty, manifest_id, features_used = self._compute_proba(
                cell_id, target_date, context
            )
            proba = np.asarray(proba, dtype=np.float64)
            proba = np.clip(proba, 1e-8, None)
            proba /= proba.sum()
            uncertainty = float(np.clip(uncertainty, 0.0, 0.99))
        except ModalityMissingError as exc:
            logger.warning(
                "Agent '%s' modality missing for cell=%d date=%s: %s",
                self.name, cell_id, target_date, exc,
            )
            proba = self.base_rate.copy()
            uncertainty = 1.0
            missing = True

        opinion = Opinion.from_proba(
            proba=proba,
            uncertainty=uncertainty,
            frame=self.FRAME,
        )

        prov = ProvenanceRecord(
            agent_name=self.name,
            modality=self.modality,
            manifest_id=manifest_id,
            model_ckpt=self.model_ckpt,
            modality_missing=missing,
            source_ref=f"{self.modality} via {self.model_ckpt}",
            features_used=features_used,
            cell_id=cell_id,
            target_date=target_date,
            raw_proba=proba.tolist(),
            uncertainty_u=opinion.u,
            credibility_gamma=credibility_gamma,
        )
        return opinion, prov
