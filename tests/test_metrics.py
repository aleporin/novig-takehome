"""Metrics tests against a hand-computed fixture.

A wrong harness would give false confidence in every later phase, so the metric
math is checked against a tiny set whose answers are worked out by hand below.
"""

from __future__ import annotations

from evals.metrics import Scored, evaluate
from triage.schemas import Category, Label, Prediction, Urgency


def _pred(ticket_id, category, urgency, *, draft) -> Prediction:
    if draft:
        return Prediction(
            ticket_id=ticket_id,
            category=category,
            urgency=urgency,
            should_draft=True,
            draft_response="draft text\n\nNovig Support",
            confidence=0.8,
        )
    return Prediction(
        ticket_id=ticket_id,
        category=category,
        urgency=urgency,
        should_draft=False,
        no_draft_reason="declined",
        confidence=0.5,
    )


def _label(category, urgency, *, should_draft) -> Label:
    return Label(
        category=category,
        urgency=urgency,
        should_draft=should_draft,
        no_draft_reason=None if should_draft else "human review",
        gold_response_notes="notes",
    )


# Five hand-worked cases (predicted vs gold):
#  A cat hit,  urg hit,  drafts        -> cost 0
#  B cat miss, urg miss, drafts        -> gold account_compromise (HARD), cost 8, false-draft hard
#  C cat hit,  urg hit,  drafts        -> gold market_questions (SOFT), cost 0, false-draft soft
#  D cat hit,  urg hit,  declines      -> cost 0, false-decline (gold wanted a draft)
#  E cat hit,  urg miss, drafts        -> cost 2 (high mispredicted as medium)
FIXTURE = [
    Scored(
        "A",
        _pred("A", Category.deposits_withdrawals, Urgency.high, draft=True),
        _label(Category.deposits_withdrawals, Urgency.high, should_draft=True),
    ),
    Scored(
        "B",
        _pred("B", Category.deposits_withdrawals, Urgency.low, draft=True),
        _label(Category.account_compromise, Urgency.escalate_immediately, should_draft=False),
    ),
    Scored(
        "C",
        _pred("C", Category.market_questions, Urgency.medium, draft=True),
        _label(Category.market_questions, Urgency.medium, should_draft=False),
    ),
    Scored(
        "D",
        _pred("D", Category.trading_mechanics, Urgency.low, draft=False),
        _label(Category.trading_mechanics, Urgency.low, should_draft=True),
    ),
    Scored(
        "E",
        _pred("E", Category.kyc_verification, Urgency.medium, draft=True),
        _label(Category.kyc_verification, Urgency.high, should_draft=True),
    ),
]


def test_category_accuracy() -> None:
    # A, C, D, E correct; B wrong -> 4/5.
    assert evaluate(FIXTURE).category_accuracy.point == 0.8


def test_urgency_accuracy() -> None:
    # A, C, D correct; B, E wrong -> 3/5.
    assert evaluate(FIXTURE).urgency_accuracy.point == 0.6


def test_severity_weighted_error() -> None:
    # costs: A 0, B 8, C 0, D 0, E 2 -> mean 10/5 = 2.0.
    assert evaluate(FIXTURE).severity_weighted_error.point == 2.0


def test_urgency_confusion_matrix() -> None:
    matrix = evaluate(FIXTURE).urgency_confusion
    assert matrix["low"]["low"] == 1  # D
    assert matrix["medium"]["medium"] == 1  # C
    assert matrix["high"]["high"] == 1  # A
    assert matrix["high"]["medium"] == 1  # E
    assert matrix["escalate_immediately"]["low"] == 1  # B
    total = sum(count for row in matrix.values() for count in row.values())
    assert total == 5


def test_false_draft_split() -> None:
    report = evaluate(FIXTURE)
    assert (report.false_draft_hard.numerator, report.false_draft_hard.denominator) == (1, 1)
    assert (report.false_draft_soft.numerator, report.false_draft_soft.denominator) == (1, 1)


def test_false_decline() -> None:
    # gold should-draft: A, D, E; declined: D -> 1/3.
    report = evaluate(FIXTURE)
    assert (report.false_decline.numerator, report.false_decline.denominator) == (1, 3)
