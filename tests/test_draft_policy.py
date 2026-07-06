"""Soft-rule policy tests: each judgment-call decline and the allow path."""

from __future__ import annotations

from triage.schemas import RiskFlags
from triage.stages.draft_policy import apply_soft_policy


def _flags(**names) -> RiskFlags:
    return RiskFlags(**names)


def test_no_soft_flags_allows_draft() -> None:
    assert apply_soft_policy(_flags()).should_draft is True


def test_fact_dispute_declines() -> None:
    decision = apply_soft_policy(_flags(disputes_novig_fact=True))
    assert decision.should_draft is False
    assert decision.rule == "soft_flag:disputes_novig_fact"


def test_jurisdiction_question_declines() -> None:
    decision = apply_soft_policy(_flags(jurisdictional_eligibility=True))
    assert decision.should_draft is False
    assert decision.rule == "soft_flag:jurisdictional_eligibility"


def test_binding_spec_declines_by_default() -> None:
    decision = apply_soft_policy(_flags(asks_binding_policy_or_spec=True))
    assert decision.should_draft is False
    assert decision.rule == "soft_flag:asks_binding_policy_or_spec"


def test_binding_spec_allowed_in_defer_mode() -> None:
    decision = apply_soft_policy(
        _flags(asks_binding_policy_or_spec=True), binding_spec_mode="defer_to_spec"
    )
    assert decision.should_draft is True


def test_fact_dispute_takes_priority_over_binding_spec() -> None:
    decision = apply_soft_policy(_flags(disputes_novig_fact=True, asks_binding_policy_or_spec=True))
    assert decision.rule == "soft_flag:disputes_novig_fact"
