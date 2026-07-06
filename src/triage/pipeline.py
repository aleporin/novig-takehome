"""predict(ticket) -> a full decision. Orchestrates the stages, nothing more.

Order: pre-screen -> classify -> gate -> soft policy -> emit. The gate and policy
decide draft/no-draft; draft *generation* lands in a later phase, so a ticket that
passes both gets a placeholder draft for now (the metrics that matter here are
category, urgency, and the draft/no-draft decision, not the draft text). Any
classifier failure fails closed: the ticket is emitted as a no-draft for a human.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from triage.config import Config
from triage.context.assembler import PromptAssembler
from triage.context.exemplars import select_exemplars
from triage.llm.types import LLMClient, LLMError
from triage.logging_setup import DECISION, RULE_FIRED, STAGE, ticket_context
from triage.schemas import Category, Classification, Prediction, RiskFlags, Ticket, Urgency
from triage.stages.classify import classify
from triage.stages.draft_policy import apply_soft_policy
from triage.stages.gate import GateDecision, evaluate_gate
from triage.stages.prescreen import prescreen

logger = logging.getLogger(__name__)

_DRAFT_PLACEHOLDER = "[draft generation lands in a later phase]"
_ERROR_REASON = "system error — routed to human review"
_TRUNCATED_CONFIDENCE_CAP = 0.6


@dataclass
class Enriched:
    """The prediction plus the internal signals, for analysis and logging."""

    ticket_id: str
    prediction: Prediction
    prescreen_flags: RiskFlags
    classification: Classification | None
    gate: GateDecision | None
    exemplar_ids: list[str]


def _confidence(classification: Classification, truncated: bool) -> float:
    if truncated:
        return min(classification.confidence, _TRUNCATED_CONFIDENCE_CAP)
    return classification.confidence


def _no_draft(ticket: Ticket, cls: Classification, reason: str, confidence: float) -> Prediction:
    return Prediction(
        ticket_id=ticket.ticket_id,
        category=cls.category,
        urgency=cls.urgency,
        should_draft=False,
        no_draft_reason=reason,
        confidence=confidence,
    )


def _error_prediction(ticket: Ticket) -> Prediction:
    return Prediction(
        ticket_id=ticket.ticket_id,
        category=Category.other,
        urgency=Urgency.escalate_immediately,
        should_draft=False,
        no_draft_reason=_ERROR_REASON,
        confidence=0.1,
    )


def predict(
    ticket: Ticket,
    *,
    client: LLMClient,
    assembler: PromptAssembler,
    exemplar_pool: list[Ticket],
    config: Config,
) -> Enriched:
    """Run one ticket end to end and return its decision plus internal signals."""
    with ticket_context(ticket.ticket_id):
        prescreen_flags = prescreen(ticket.subject, ticket.body)

        try:
            exemplars = select_exemplars(exemplar_pool, ticket.ticket_id)
            result = classify(ticket, exemplars, client=client, assembler=assembler, config=config)
        except LLMError:
            logger.exception("classifier failed; failing closed", extra={STAGE: "classify"})
            pred = _error_prediction(ticket)
            return Enriched(ticket.ticket_id, pred, prescreen_flags, None, None, [])

        cls = result.classification
        gate = evaluate_gate(cls.category, cls.flags, prescreen_flags)
        if not gate.should_draft:
            pred = _no_draft(ticket, cls, gate.no_draft_reason, confidence=1.0)
            _log_decision("no_draft", gate.rule)
            return Enriched(ticket.ticket_id, pred, prescreen_flags, cls, gate, result.exemplar_ids)

        policy = apply_soft_policy(cls.flags, binding_spec_mode=config.binding_spec_mode)
        confidence = _confidence(cls, result.truncated)
        if not policy.should_draft:
            pred = _no_draft(ticket, cls, policy.no_draft_reason, confidence=confidence)
            _log_decision("no_draft", policy.rule)
            return Enriched(ticket.ticket_id, pred, prescreen_flags, cls, gate, result.exemplar_ids)

        pred = Prediction(
            ticket_id=ticket.ticket_id,
            category=cls.category,
            urgency=cls.urgency,
            should_draft=True,
            draft_response=_DRAFT_PLACEHOLDER,
            confidence=confidence,
        )
        _log_decision("draft", None)
        return Enriched(ticket.ticket_id, pred, prescreen_flags, cls, gate, result.exemplar_ids)


def _log_decision(decision: str, rule: str | None) -> None:
    logger.info("decision", extra={STAGE: "pipeline", DECISION: decision, RULE_FIRED: rule})
