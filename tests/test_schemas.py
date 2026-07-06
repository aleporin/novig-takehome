"""Tests for the ticket and prediction schemas.

Runs offline in well under a second. Fails if a schema field stops matching the
dataset or the draft/decline rules weaken.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from triage.config import Paths
from triage.schemas import Category, Prediction, Ticket, Urgency


def _load(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_all_train_tickets_parse_with_labels() -> None:
    for raw in _load(Paths().tickets_train):
        ticket = Ticket.model_validate(raw)
        assert ticket.label is not None
        assert ticket.label.category in set(Category)


def test_all_eval_tickets_parse_without_labels() -> None:
    tickets = [Ticket.model_validate(raw) for raw in _load(Paths().tickets_eval)]
    assert len(tickets) == 15
    assert all(t.label is None for t in tickets)


def test_unexpected_field_is_rejected_loudly(ticket) -> None:
    bad = ticket.model_dump()
    bad["surprise"] = "unexpected"
    with pytest.raises(ValidationError):
        Ticket.model_validate(bad)


def test_urgency_enum_uses_underscore() -> None:
    assert Urgency.escalate_immediately.value == "escalate_immediately"


def test_valid_draft_prediction_roundtrips() -> None:
    pred = Prediction(
        ticket_id="t_eval_001",
        category=Category.deposits_withdrawals,
        urgency=Urgency.high,
        should_draft=True,
        no_draft_reason=None,
        draft_response="Hi, thanks for reaching out...\n\nNovig Support",
        confidence=0.82,
    )
    line = pred.to_jsonl_line()
    assert "\n" not in line  # exactly one physical line
    restored = json.loads(line)
    # Compact: re-serializing with no separators spaces reproduces the line byte-for-byte.
    assert line == json.dumps(restored, separators=(",", ":"), ensure_ascii=False)
    assert set(restored) == {
        "ticket_id",
        "category",
        "urgency",
        "should_draft",
        "no_draft_reason",
        "draft_response",
        "confidence",
    }
    assert restored["no_draft_reason"] is None


def test_valid_decline_prediction() -> None:
    pred = Prediction(
        ticket_id="t_eval_003",
        category=Category.account_compromise,
        urgency=Urgency.escalate_immediately,
        should_draft=False,
        no_draft_reason="suspected account compromise",
        draft_response=None,
        confidence=1.0,
    )
    assert json.loads(pred.to_jsonl_line())["draft_response"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        # drafted but also carrying a decline reason
        {"should_draft": True, "no_draft_reason": "x", "draft_response": "d"},
        # drafted but no draft text
        {"should_draft": True, "no_draft_reason": None, "draft_response": None},
        # declined but still carrying a draft
        {"should_draft": False, "no_draft_reason": "x", "draft_response": "d"},
        # declined with an empty reason
        {"should_draft": False, "no_draft_reason": "  ", "draft_response": None},
        # declined with a null reason
        {"should_draft": False, "no_draft_reason": None, "draft_response": None},
    ],
)
def test_mutual_exclusion_violations_are_unconstructable(kwargs) -> None:
    with pytest.raises(ValidationError):
        Prediction(
            ticket_id="t",
            category=Category.other,
            urgency=Urgency.low,
            confidence=0.5,
            **kwargs,
        )


def test_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        Prediction(
            ticket_id="t",
            category=Category.other,
            urgency=Urgency.low,
            should_draft=False,
            no_draft_reason="x",
            confidence=1.5,
        )
