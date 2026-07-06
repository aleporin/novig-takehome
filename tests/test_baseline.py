"""Tests for the majority-class baseline predictor."""

from __future__ import annotations

import pytest

from evals.baseline import MajorityBaseline
from evals.dataset import load_tickets
from triage.config import Config
from triage.schemas import Category, Urgency


def test_from_labels_picks_majority_class() -> None:
    tickets = load_tickets(Config().paths.tickets_train)
    baseline = MajorityBaseline.from_labels(tickets)
    # Most common in train: deposits_withdrawals (6) and low (12).
    assert baseline.category == Category.deposits_withdrawals
    assert baseline.urgency == Urgency.low


def test_predict_declines_with_majority_labels(make_ticket) -> None:
    baseline = MajorityBaseline(category=Category.deposits_withdrawals, urgency=Urgency.low)
    pred = baseline.predict(make_ticket(ticket_id="t1"))
    assert pred.ticket_id == "t1"
    assert pred.should_draft is False
    assert pred.draft_response is None
    assert pred.category == Category.deposits_withdrawals
    assert pred.urgency == Urgency.low


def test_predict_is_deterministic(make_ticket) -> None:
    baseline = MajorityBaseline(category=Category.other, urgency=Urgency.low)
    ticket = make_ticket(ticket_id="t1")
    assert baseline.predict(ticket) == baseline.predict(ticket)


def test_from_labels_requires_labels(make_ticket) -> None:
    with pytest.raises(ValueError, match="unlabeled"):
        MajorityBaseline.from_labels([make_ticket(ticket_id="t1")])
