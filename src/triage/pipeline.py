"""predict(ticket) -> a full decision. Orchestrates the stages, nothing more.

Order: pre-screen -> classify (with T1->T2 routing) -> gate -> soft policy -> emit.
Escalation only ever strengthens the classification (adds flags, keeps a sensitive
category, raises urgency). Draft generation lands in a later phase, so a ticket
that passes both gate and policy gets a placeholder draft for now. Any classifier
failure fails closed: the ticket is emitted as a no-draft for a human.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from triage.config import Config
from triage.context.assembler import PromptAssembler
from triage.context.exemplars import select_exemplars
from triage.llm.types import LLMClient, LLMError
from triage.logging_setup import DECISION, RULE_FIRED, STAGE, ticket_context
from triage.routing.cost_tracker import CallRecord
from triage.routing.router import resolve, should_escalate
from triage.schemas import Category, Classification, Prediction, RiskFlags, Ticket, Urgency
from triage.stages.classify import classify
from triage.stages.draft_policy import apply_soft_policy
from triage.stages.gate import GateDecision, evaluate_gate
from triage.stages.output_guard import produce_draft
from triage.stages.prescreen import prescreen

logger = logging.getLogger(__name__)

_GUARDRAIL_REASON = "draft failed output guardrail"
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
    tier: str = "T1"
    escalated: bool = False
    escalation_reason: str | None = None
    calls: list[CallRecord] = field(default_factory=list)


@dataclass
class _Routed:
    classification: Classification
    tier: str
    escalated: bool
    reason: str | None
    calls: list[CallRecord]
    truncated: bool
    exemplar_ids: list[str]


def _classify_with_routing(
    ticket: Ticket,
    prescreen_flags: RiskFlags,
    exemplars: list[Ticket],
    *,
    client: LLMClient,
    assembler: PromptAssembler,
    config: Config,
) -> _Routed | None:
    """Classify with T1, escalating to T2 on any trigger. None if both models fail."""

    def run(model: str, tier: str):
        return classify(
            ticket,
            exemplars,
            client=client,
            assembler=assembler,
            config=config,
            model=model,
            tier=tier,
        )

    try:
        t1 = run(config.model_t1, "T1")
    except LLMError:
        # Trigger 3: T1 could not produce valid output; T2 is the stronger repair.
        try:
            t2 = run(config.model_t2, "T2")
        except LLMError:
            return None
        return _Routed(
            t2.classification, "T2", True, "t1_failure", [t2.call], t2.truncated, t2.exemplar_ids
        )

    decision = should_escalate(
        t1.classification, prescreen_flags, threshold=config.t1_confidence_threshold
    )
    if not decision.escalate:
        return _Routed(
            t1.classification, "T1", False, None, [t1.call], t1.truncated, t1.exemplar_ids
        )

    try:
        t2 = run(config.model_t2, "T2")
    except LLMError:
        logger.warning("T2 escalation failed; keeping T1", extra={STAGE: "route"})
        return _Routed(
            t1.classification, "T1", False, None, [t1.call], t1.truncated, t1.exemplar_ids
        )

    resolved = resolve(t1.classification, t2.classification)
    logger.warning("escalated T1->T2", extra={STAGE: "route", RULE_FIRED: decision.reason})
    return _Routed(
        resolved,
        "T2",
        True,
        decision.reason,
        [t1.call, t2.call],
        t1.truncated or t2.truncated,
        t1.exemplar_ids,
    )


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


def _error_enriched(ticket: Ticket, prescreen_flags: RiskFlags) -> Enriched:
    pred = Prediction(
        ticket_id=ticket.ticket_id,
        category=Category.other,
        urgency=Urgency.escalate_immediately,
        should_draft=False,
        no_draft_reason=_ERROR_REASON,
        confidence=0.1,
    )
    return Enriched(
        ticket.ticket_id, pred, prescreen_flags, None, None, [], "none", False, None, []
    )


def _enriched(ticket: Ticket, pred: Prediction, pf: RiskFlags, gate, routed: _Routed) -> Enriched:
    return Enriched(
        ticket.ticket_id,
        pred,
        pf,
        routed.classification,
        gate,
        routed.exemplar_ids,
        routed.tier,
        routed.escalated,
        routed.reason,
        routed.calls,
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
        exemplars = select_exemplars(exemplar_pool, ticket.ticket_id)
        routed = _classify_with_routing(
            ticket, prescreen_flags, exemplars, client=client, assembler=assembler, config=config
        )
        if routed is None:
            logger.exception("classifier failed; failing closed", extra={STAGE: "classify"})
            return _error_enriched(ticket, prescreen_flags)

        cls = routed.classification
        gate = evaluate_gate(cls.category, cls.flags, prescreen_flags)
        if not gate.should_draft:
            _log_decision("no_draft", gate.rule)
            return _enriched(
                ticket,
                _no_draft(ticket, cls, gate.no_draft_reason, 1.0),
                prescreen_flags,
                gate,
                routed,
            )

        policy = apply_soft_policy(
            cls.flags, prescreen_flags, binding_spec_mode=config.binding_spec_mode
        )
        confidence = _confidence(cls, routed.truncated)
        if not policy.should_draft:
            _log_decision("no_draft", policy.rule)
            pred = _no_draft(ticket, cls, policy.no_draft_reason, confidence)
            return _enriched(ticket, pred, prescreen_flags, gate, routed)

        draft, draft_calls = produce_draft(
            ticket, cls, client=client, assembler=assembler, config=config
        )
        routed.calls.extend(draft_calls)
        if draft is None:
            _log_decision("no_draft", "draft_failed_guardrail")
            pred = _no_draft(ticket, cls, _GUARDRAIL_REASON, confidence)
            return _enriched(ticket, pred, prescreen_flags, gate, routed)

        pred = Prediction(
            ticket_id=ticket.ticket_id,
            category=cls.category,
            urgency=cls.urgency,
            should_draft=True,
            draft_response=draft,
            confidence=confidence,
        )
        _log_decision("draft", None)
        return _enriched(ticket, pred, prescreen_flags, gate, routed)


def _log_decision(decision: str, rule: str | None) -> None:
    logger.info("decision", extra={STAGE: "pipeline", DECISION: decision, RULE_FIRED: rule})
