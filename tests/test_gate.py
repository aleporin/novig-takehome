"""Gate tests: exhaustive enumeration plus universal safety properties.

The exhaustive test walks all 11 categories crossed with every subset of the
hard-rule flags. The property tests then assert laws that must hold for *any*
input the enumeration might not have imagined.
"""

from __future__ import annotations

from itertools import combinations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from triage.schemas import SENSITIVE_CATEGORIES, Category, RiskFlags
from triage.stages.gate import HARD_RULE_FLAGS, evaluate_gate

_FLAG_NAMES = list(RiskFlags.model_fields)


def _all_subsets(items):
    for size in range(len(items) + 1):
        yield from combinations(items, size)


def _risk(names) -> RiskFlags:
    return RiskFlags(**{name: True for name in names})


@pytest.mark.parametrize("category", list(Category))
@pytest.mark.parametrize("subset", list(_all_subsets(HARD_RULE_FLAGS)))
def test_every_category_and_flag_subset(category, subset) -> None:
    decision = evaluate_gate(category, _risk(subset), RiskFlags())
    if category in SENSITIVE_CATEGORIES:
        assert decision.should_draft is False
        assert decision.rule.startswith("sensitive_category")
    elif subset:
        assert decision.should_draft is False
        assert decision.rule.startswith("hard_flag")
        assert decision.no_draft_reason
    else:
        assert decision.should_draft is True
        assert decision.no_draft_reason is None


def test_prescreen_flag_alone_blocks_and_is_sourced() -> None:
    decision = evaluate_gate(Category.deposits_withdrawals, RiskFlags(), _risk(["active_fraud"]))
    assert decision.should_draft is False
    assert decision.flag_source == "prescreen"


def test_flag_in_both_sources_is_sourced_both() -> None:
    flags = _risk(["legal_threat"])
    decision = evaluate_gate(Category.other, flags, flags)
    assert decision.flag_source == "both"


def test_model_flag_alone_is_sourced_model() -> None:
    decision = evaluate_gate(Category.other, _risk(["rg_signal"]), RiskFlags())
    assert decision.flag_source == "model"


# --- Property-based safety laws ---

_risk_strategy = st.builds(RiskFlags, **{name: st.booleans() for name in _FLAG_NAMES})
_category_strategy = st.sampled_from(list(Category))


@given(_category_strategy, _risk_strategy, _risk_strategy)
def test_should_draft_iff_no_hard_condition(category, model, prescreen) -> None:
    blocked = category in SENSITIVE_CATEGORIES or any(
        getattr(model, f) or getattr(prescreen, f) for f in HARD_RULE_FLAGS
    )
    assert evaluate_gate(category, model, prescreen).should_draft is (not blocked)


@given(_category_strategy, _risk_strategy, _risk_strategy, st.sampled_from(HARD_RULE_FLAGS))
def test_adding_a_flag_never_unblocks(category, model, prescreen, extra) -> None:
    before = evaluate_gate(category, model, prescreen)
    after = evaluate_gate(category, model.model_copy(update={extra: True}), prescreen)
    if not before.should_draft:
        assert not after.should_draft


@given(_risk_strategy, _risk_strategy, st.sampled_from(sorted(SENSITIVE_CATEGORIES)))
def test_sensitive_category_always_blocks(model, prescreen, category) -> None:
    assert evaluate_gate(category, model, prescreen).should_draft is False
