"""Tests for the draft stage: generation and model routing."""

from __future__ import annotations

from tests.fakes import FakeLLMClient
from triage.config import Config
from triage.context.assembler import PromptAssembler
from triage.schemas import Category, Classification, RiskFlags, Urgency
from triage.stages.draft import draft_model, generate_draft


def _cls(category, urgency=Urgency.low) -> Classification:
    return Classification(category=category, urgency=urgency, flags=RiskFlags(), confidence=0.9)


def test_generate_draft_returns_text_and_call(make_ticket) -> None:
    config = Config()
    client = FakeLLMClient(["Here you go.\n\nNovig Support"])
    result = generate_draft(
        make_ticket(),
        Category.trading_mechanics,
        client=client,
        assembler=PromptAssembler(config),
        config=config,
        model=config.model_t1,
    )
    assert result.text == "Here you go.\n\nNovig Support"
    assert result.call.tier == "draft"


def test_money_in_motion_routes_to_stronger_model() -> None:
    config = Config()
    assert draft_model(_cls(Category.deposits_withdrawals), config) == config.model_t2


def test_high_urgency_routes_to_stronger_model() -> None:
    config = Config()
    assert draft_model(_cls(Category.trading_mechanics, Urgency.high), config) == config.model_t2


def test_simple_draft_routes_to_cheap_model() -> None:
    config = Config()
    assert draft_model(_cls(Category.trading_mechanics, Urgency.low), config) == config.model_t1
