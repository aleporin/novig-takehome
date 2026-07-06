"""Escalation triggers and conservative resolution for the T1->T2 cascade.

Escalation is safety-asymmetric: it can add a flag or raise urgency, but it can
never clear a flag or downgrade a sensitive category that T1 already reported.
Resolution unions the flags (T2 adds, never removes), keeps a sensitive category
from either tier, and takes the more urgent of the two. Category and urgency
disagreements on non-sensitive values defer to T2, the stronger model.
"""

from __future__ import annotations

from dataclasses import dataclass

from triage.schemas import SENSITIVE_CATEGORIES, Category, Classification, RiskFlags, Urgency

_FLAG_NAMES = tuple(RiskFlags.model_fields)

# Flags the lexical pre-screen can set; a disagreement on any of them escalates.
_PRESCREEN_FLAGS = (
    "self_harm_or_distress",
    "mentions_minor",
    "active_fraud",
    "unauthorized_access_reported",
    "legal_threat",
    "rg_signal",
    "disputes_novig_fact",
)

_URGENCY_RANK = {
    Urgency.low: 0,
    Urgency.medium: 1,
    Urgency.high: 2,
    Urgency.escalate_immediately: 3,
}
_CONFIDENCE_CAP_ON_DISAGREEMENT = 0.6


@dataclass(frozen=True)
class RoutingDecision:
    """Whether to escalate T1 to T2, and why."""

    escalate: bool
    reason: str | None = None


def _any_flag(flags: RiskFlags) -> bool:
    return any(getattr(flags, name) for name in _FLAG_NAMES)


def _prescreen_disagrees(t1_flags: RiskFlags, prescreen: RiskFlags) -> bool:
    return any(getattr(t1_flags, f) != getattr(prescreen, f) for f in _PRESCREEN_FLAGS)


def should_escalate(
    t1: Classification, prescreen: RiskFlags, *, threshold: float
) -> RoutingDecision:
    """Apply the escalation triggers in safety-priority order."""
    if t1.category in SENSITIVE_CATEGORIES:
        return RoutingDecision(True, "sensitive_category")
    if _any_flag(t1.flags):
        return RoutingDecision(True, "risk_flag")
    if _prescreen_disagrees(t1.flags, prescreen):
        return RoutingDecision(True, "prescreen_disagreement")
    if t1.confidence < threshold:
        return RoutingDecision(True, "low_confidence")
    return RoutingDecision(False, None)


def _resolve_category(t1_category: Category, t2_category: Category) -> Category:
    if t1_category in SENSITIVE_CATEGORIES:
        return t1_category  # T2 cannot downgrade a sensitive call to benign
    return t2_category  # T2 may escalate to sensitive, or refine a benign call


def _more_urgent(a: Urgency, b: Urgency) -> Urgency:
    return a if _URGENCY_RANK[a] >= _URGENCY_RANK[b] else b


def resolve(t1: Classification, t2: Classification) -> Classification:
    """Merge T1 and T2 into the conservative result."""
    merged = RiskFlags(
        **{name: getattr(t1.flags, name) or getattr(t2.flags, name) for name in _FLAG_NAMES}
    )
    agree = t1.category == t2.category and t1.urgency == t2.urgency and t1.flags == t2.flags
    confidence = (
        max(t1.confidence, t2.confidence)
        if agree
        else min(t1.confidence, t2.confidence, _CONFIDENCE_CAP_ON_DISAGREEMENT)
    )
    return Classification(
        category=_resolve_category(t1.category, t2.category),
        urgency=_more_urgent(t1.urgency, t2.urgency),
        flags=merged,
        confidence=confidence,
        reasoning=t2.reasoning,
    )
