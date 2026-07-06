"""Tests for scoring and the layer-attribution diagnostic."""

from __future__ import annotations

from evals.pipeline_run import layer_attribution, scored_from
from triage.pipeline import Enriched
from triage.schemas import (
    Category,
    Classification,
    Label,
    Prediction,
    RiskFlags,
    Ticket,
    TicketMetadata,
    Urgency,
)


def _ticket(ticket_id, category, should_draft) -> Ticket:
    return Ticket(
        ticket_id=ticket_id,
        subject="s",
        body="b",
        metadata=TicketMetadata(user_tenure_days=1, submitted_at="2026-01-01T00:00:00Z"),
        label=Label(
            category=category,
            urgency=Urgency.low,
            should_draft=should_draft,
            gold_response_notes="n",
        ),
    )


def _enriched(ticket_id, *, prescreen, classifier_category, classifier_flags=None) -> Enriched:
    cls = Classification(
        category=classifier_category,
        urgency=Urgency.low,
        flags=RiskFlags(**(classifier_flags or {})),
        confidence=0.9,
    )
    pred = Prediction(
        ticket_id=ticket_id,
        category=classifier_category,
        urgency=Urgency.low,
        should_draft=False,
        no_draft_reason="x",
        confidence=1.0,
    )
    return Enriched(ticket_id, pred, RiskFlags(**prescreen), cls, None, [])


def test_layer_attribution_classifies_each_source() -> None:
    tickets = [
        _ticket("A", Category.account_compromise, False),
        _ticket("B", Category.account_compromise, False),
        _ticket("C", Category.deposits_withdrawals, False),
        _ticket("D", Category.trading_mechanics, True),  # gold draft -> excluded
    ]
    enriched = [
        _enriched(
            "A", prescreen={"active_fraud": True}, classifier_category=Category.account_compromise
        ),
        _enriched("B", prescreen={}, classifier_category=Category.account_compromise),
        _enriched(
            "C", prescreen={"legal_threat": True}, classifier_category=Category.deposits_withdrawals
        ),
        _enriched("D", prescreen={}, classifier_category=Category.trading_mechanics),
    ]
    rows = {r["ticket_id"]: r["layer"] for r in layer_attribution(enriched, tickets)}
    assert rows == {"A": "both", "B": "classifier_only", "C": "lexicon_only"}


def test_scored_from_filters_to_requested_ids() -> None:
    tickets = [
        _ticket("A", Category.account_compromise, False),
        _ticket("B", Category.deposits_withdrawals, True),
    ]
    enriched = [
        _enriched("A", prescreen={}, classifier_category=Category.account_compromise),
        _enriched("B", prescreen={}, classifier_category=Category.deposits_withdrawals),
    ]
    scored = scored_from(enriched, tickets, only_ids={"A"})
    assert [s.ticket_id for s in scored] == ["A"]
