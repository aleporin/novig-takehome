"""Metrics computed against the training labels.

All pure functions over (prediction, gold-label) pairs. The headline is the
false-draft rate on sensitive tickets, split into hard-rule and soft-rule, where
hard-rule tickets are those whose gold category is sensitive. That split is exact
for the provided data (the only non-sensitive-category no-draft is a soft
fact-dispute); once the classifier emits explicit minor/self-harm/fraud flags,
the split can use those too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from triage.schemas import SENSITIVE_CATEGORIES, Label, Prediction, Urgency

from .bootstrap import Estimate, bootstrap_ci

# Urgency from least to most urgent, for distance and confusion ordering.
URGENCY_ORDER: tuple[Urgency, ...] = (
    Urgency.low,
    Urgency.medium,
    Urgency.high,
    Urgency.escalate_immediately,
)

# Cost of predicting urgency P (column) when the truth is G (row).
# Under-calling (guessing lower than truth) costs more than over-calling, and
# missing escalate_immediately is worst. Correct is always 0.
SEVERITY_COST: dict[Urgency, dict[Urgency, float]] = {
    Urgency.low: {
        Urgency.low: 0,
        Urgency.medium: 1,
        Urgency.high: 2,
        Urgency.escalate_immediately: 3,
    },
    Urgency.medium: {
        Urgency.low: 2,
        Urgency.medium: 0,
        Urgency.high: 1,
        Urgency.escalate_immediately: 2,
    },
    Urgency.high: {
        Urgency.low: 4,
        Urgency.medium: 2,
        Urgency.high: 0,
        Urgency.escalate_immediately: 1,
    },
    Urgency.escalate_immediately: {
        Urgency.low: 8,
        Urgency.medium: 6,
        Urgency.high: 3,
        Urgency.escalate_immediately: 0,
    },
}


@dataclass(frozen=True)
class Scored:
    """One ticket's prediction paired with its gold label."""

    ticket_id: str
    predicted: Prediction
    gold: Label


@dataclass(frozen=True)
class RateStat:
    """A count-based rate: numerator over denominator, with an interval."""

    numerator: int
    denominator: int
    estimate: Estimate


@dataclass(frozen=True)
class MetricsReport:
    """Everything the harness computes for one labeled run."""

    n: int
    category_accuracy: Estimate
    urgency_accuracy: Estimate
    severity_weighted_error: Estimate
    urgency_confusion: dict[str, dict[str, int]]
    false_draft_hard: RateStat
    false_draft_soft: RateStat
    false_decline: RateStat
    flag_notes: list[str] = field(default_factory=list)


def _is_hard(gold: Label) -> bool:
    """A gold no-draft ticket is hard-rule if its category is sensitive."""
    return gold.category in SENSITIVE_CATEGORIES


def category_accuracy(scored: list[Scored], *, seed: int = 0) -> Estimate:
    hits = [1.0 if s.predicted.category == s.gold.category else 0.0 for s in scored]
    return bootstrap_ci(hits, seed=seed)


def urgency_accuracy(scored: list[Scored], *, seed: int = 0) -> Estimate:
    hits = [1.0 if s.predicted.urgency == s.gold.urgency else 0.0 for s in scored]
    return bootstrap_ci(hits, seed=seed)


def severity_weighted_error(scored: list[Scored], *, seed: int = 0) -> Estimate:
    costs = [float(SEVERITY_COST[s.gold.urgency][s.predicted.urgency]) for s in scored]
    return bootstrap_ci(costs, seed=seed)


def urgency_confusion(scored: list[Scored]) -> dict[str, dict[str, int]]:
    """4x4 count matrix, rows = gold urgency, columns = predicted urgency."""
    matrix = {g.value: {p.value: 0 for p in URGENCY_ORDER} for g in URGENCY_ORDER}
    for s in scored:
        matrix[s.gold.urgency.value][s.predicted.urgency.value] += 1
    return matrix


def _false_draft(scored: list[Scored], *, hard: bool, seed: int) -> RateStat:
    """Rate of drafting on gold no-draft tickets, within the hard or soft region."""
    region = [s for s in scored if not s.gold.should_draft and _is_hard(s.gold) == hard]
    indicators = [1.0 if s.predicted.should_draft else 0.0 for s in region]
    numerator = int(sum(indicators))
    return RateStat(numerator, len(region), bootstrap_ci(indicators, seed=seed))


def false_decline(scored: list[Scored], *, seed: int = 0) -> RateStat:
    """Rate of declining on gold should-draft tickets. The dual of false-draft."""
    region = [s for s in scored if s.gold.should_draft]
    indicators = [0.0 if s.predicted.should_draft else 1.0 for s in region]
    numerator = int(sum(indicators))
    return RateStat(numerator, len(region), bootstrap_ci(indicators, seed=seed))


def evaluate(scored: list[Scored], *, seed: int = 0) -> MetricsReport:
    """Compute the full metrics report for a set of scored predictions."""
    return MetricsReport(
        n=len(scored),
        category_accuracy=category_accuracy(scored, seed=seed),
        urgency_accuracy=urgency_accuracy(scored, seed=seed),
        severity_weighted_error=severity_weighted_error(scored, seed=seed),
        urgency_confusion=urgency_confusion(scored),
        false_draft_hard=_false_draft(scored, hard=True, seed=seed),
        false_draft_soft=_false_draft(scored, hard=False, seed=seed),
        false_decline=false_decline(scored, seed=seed),
    )
