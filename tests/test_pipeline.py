"""End-to-end pipeline tests with a scripted fake client (no network).

With routing, sensitive or flagged tickets escalate and consume a second (T2)
scripted response, so those tests queue two classifications.
"""

from __future__ import annotations

from tests.fakes import FakeLLMClient
from triage.config import Config
from triage.context.assembler import PromptAssembler
from triage.llm.types import LLMError
from triage.pipeline import predict
from triage.schemas import Category, Classification, RiskFlags, Urgency


def _cls(category, urgency=Urgency.low, confidence=0.9, **flags) -> Classification:
    return Classification(
        category=category, urgency=urgency, flags=RiskFlags(**flags), confidence=confidence
    )


def _run(ticket, *scripted):
    config = Config()
    return predict(
        ticket,
        client=FakeLLMClient(list(scripted)),
        assembler=PromptAssembler(config),
        exemplar_pool=[],
        config=config,
    )


def test_benign_high_confidence_stays_on_t1(make_ticket) -> None:
    item = _run(make_ticket(), _cls(Category.trading_mechanics, confidence=0.9))
    assert item.prediction.should_draft is True
    assert item.tier == "T1"
    assert item.escalated is False
    assert len(item.calls) == 1


def test_sensitive_category_escalates_and_blocks(make_ticket) -> None:
    item = _run(
        make_ticket(),
        _cls(Category.account_compromise),  # T1: triggers escalation
        _cls(Category.account_compromise),  # T2
    )
    assert item.escalated is True
    assert item.escalation_reason == "sensitive_category"
    assert item.prediction.should_draft is False
    assert item.prediction.confidence == 1.0


def test_prescreen_disagreement_escalates_and_blocks(make_ticket) -> None:
    ticket = make_ticket(body="I'm going to hurt myself because of gambling")
    item = _run(ticket, _cls(Category.trading_mechanics), _cls(Category.trading_mechanics))
    assert item.escalated is True
    assert item.escalation_reason == "prescreen_disagreement"
    assert item.prediction.should_draft is False
    assert item.gate.flag_source == "prescreen"


def test_soft_flag_escalates_then_declines(make_ticket) -> None:
    item = _run(
        make_ticket(),
        _cls(Category.market_questions, disputes_novig_fact=True),
        _cls(Category.market_questions, disputes_novig_fact=True),
    )
    assert item.escalated is True
    assert item.prediction.should_draft is False
    assert "fact" in item.prediction.no_draft_reason


def test_t1_failure_repairs_with_t2(make_ticket) -> None:
    item = _run(make_ticket(), LLMError("bad json"), _cls(Category.trading_mechanics))
    assert item.tier == "T2"
    assert item.escalation_reason == "t1_failure"
    assert item.prediction.should_draft is True


def test_both_models_fail_closes(make_ticket) -> None:
    item = _run(make_ticket(), LLMError("t1"), LLMError("t2"))
    assert item.prediction.should_draft is False
    assert item.prediction.no_draft_reason == "system error — routed to human review"
    assert item.classification is None
