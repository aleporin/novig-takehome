"""Router tests: escalation triggers and the conservative-resolution guarantee."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from triage.routing.router import resolve, should_escalate
from triage.schemas import SENSITIVE_CATEGORIES, Category, Classification, RiskFlags, Urgency

_FLAG_NAMES = list(RiskFlags.model_fields)


def _cls(category, urgency=Urgency.low, confidence=0.9, **flags) -> Classification:
    return Classification(
        category=category, urgency=urgency, flags=RiskFlags(**flags), confidence=confidence
    )


# --- escalation triggers ---


def test_sensitive_category_escalates() -> None:
    d = should_escalate(_cls(Category.account_compromise), RiskFlags(), threshold=0.75)
    assert d.escalate and d.reason == "sensitive_category"


def test_any_risk_flag_escalates() -> None:
    d = should_escalate(
        _cls(Category.trading_mechanics, active_fraud=True), RiskFlags(), threshold=0.75
    )
    assert d.escalate and d.reason == "risk_flag"


def test_prescreen_disagreement_escalates() -> None:
    d = should_escalate(
        _cls(Category.trading_mechanics), RiskFlags(self_harm_or_distress=True), threshold=0.75
    )
    assert d.escalate and d.reason == "prescreen_disagreement"


def test_low_confidence_escalates() -> None:
    d = should_escalate(
        _cls(Category.trading_mechanics, confidence=0.5), RiskFlags(), threshold=0.75
    )
    assert d.escalate and d.reason == "low_confidence"


def test_benign_high_confidence_does_not_escalate() -> None:
    d = should_escalate(
        _cls(Category.trading_mechanics, confidence=0.9), RiskFlags(), threshold=0.75
    )
    assert d.escalate is False


# --- conservative resolution ---


def test_resolution_unions_flags() -> None:
    resolved = resolve(
        _cls(Category.trading_mechanics, active_fraud=True),
        _cls(Category.trading_mechanics, legal_threat=True),
    )
    assert resolved.flags.active_fraud and resolved.flags.legal_threat


def test_t2_refines_benign_category_and_raises_urgency() -> None:
    resolved = resolve(
        _cls(Category.other, urgency=Urgency.high),
        _cls(Category.deposits_withdrawals, urgency=Urgency.low),
    )
    assert resolved.category == Category.deposits_withdrawals
    assert resolved.urgency == Urgency.high  # more urgent of the two


_flag_strategy = st.fixed_dictionaries({name: st.booleans() for name in _FLAG_NAMES})
_cls_strategy = st.builds(
    Classification,
    category=st.sampled_from(list(Category)),
    urgency=st.sampled_from(list(Urgency)),
    flags=st.builds(RiskFlags, **{name: st.booleans() for name in _FLAG_NAMES}),
    confidence=st.floats(0.0, 1.0),
)


@given(_cls_strategy, _cls_strategy)
def test_escalation_never_removes_a_flag(t1, t2) -> None:
    resolved = resolve(t1, t2)
    for name in _FLAG_NAMES:
        # A flag set by either tier must survive resolution.
        assert getattr(resolved.flags, name) >= getattr(t1.flags, name)
        assert getattr(resolved.flags, name) >= getattr(t2.flags, name)


@given(_cls_strategy, st.sampled_from(sorted(SENSITIVE_CATEGORIES)))
def test_t2_cannot_downgrade_a_sensitive_category(t2, sensitive_category) -> None:
    t1 = _cls(sensitive_category)
    assert resolve(t1, t2).category in SENSITIVE_CATEGORIES


@given(_cls_strategy)
def test_agreement_never_caps_below_disagreement_confidence(cls) -> None:
    # Identical tiers agree, so confidence is the max, never the disagreement cap.
    resolved = resolve(cls, cls)
    assert resolved.confidence == pytest.approx(cls.confidence)
