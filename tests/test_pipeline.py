"""End-to-end pipeline tests with a scripted fake client (no network).

Each test injects one classification (or an error) and checks the whole
prescreen -> classify -> gate -> policy path lands on the right decision.
"""

from __future__ import annotations

from tests.fakes import FakeLLMClient
from triage.config import Config
from triage.context.assembler import PromptAssembler
from triage.llm.types import LLMError
from triage.pipeline import predict
from triage.schemas import Category, Classification, RiskFlags, Urgency


def _classification(category, urgency=Urgency.low, **flags) -> Classification:
    return Classification(
        category=category, urgency=urgency, flags=RiskFlags(**flags), confidence=0.9
    )


def _run(ticket, scripted):
    config = Config()
    return predict(
        ticket,
        client=FakeLLMClient([scripted]),
        assembler=PromptAssembler(config),
        exemplar_pool=[],
        config=config,
    )


def test_benign_ticket_drafts(make_ticket) -> None:
    item = _run(make_ticket(), _classification(Category.trading_mechanics))
    assert item.prediction.should_draft is True
    assert item.prediction.draft_response is not None
    assert item.prediction.confidence == 0.9


def test_sensitive_category_blocks_with_full_confidence(make_ticket) -> None:
    item = _run(make_ticket(), _classification(Category.account_compromise))
    assert item.prediction.should_draft is False
    assert item.prediction.confidence == 1.0
    assert "compromise" in item.prediction.no_draft_reason


def test_prescreen_blocks_even_when_classifier_says_benign(make_ticket) -> None:
    ticket = make_ticket(body="I'm going to hurt myself because of gambling")
    item = _run(ticket, _classification(Category.trading_mechanics))
    assert item.prediction.should_draft is False
    assert item.gate.flag_source == "prescreen"


def test_soft_flag_declines_after_passing_the_gate(make_ticket) -> None:
    item = _run(make_ticket(), _classification(Category.market_questions, disputes_novig_fact=True))
    assert item.gate.should_draft is True
    assert item.prediction.should_draft is False
    assert "fact" in item.prediction.no_draft_reason


def test_prescreen_backstops_a_missed_fact_dispute(make_ticket) -> None:
    # Classifier says benign, but the ticket text disputes a Novig grade.
    ticket = make_ticket(body="this market was graded wrong, it should have settled yes")
    item = _run(ticket, _classification(Category.market_questions))
    assert item.prediction.should_draft is False
    assert "fact" in item.prediction.no_draft_reason


def test_classifier_error_fails_closed(make_ticket) -> None:
    item = _run(make_ticket(), LLMError("boom"))
    assert item.prediction.should_draft is False
    assert item.prediction.no_draft_reason == "system error — routed to human review"
    assert item.prediction.urgency == Urgency.escalate_immediately
    assert item.classification is None
