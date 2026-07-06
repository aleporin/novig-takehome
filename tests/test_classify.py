"""Test the classify stage against a scripted fake client."""

from __future__ import annotations

from tests.fakes import FakeLLMClient
from triage.config import Config
from triage.context.assembler import PromptAssembler
from triage.schemas import Category, Classification, Urgency
from triage.stages.classify import classify


def test_classify_returns_parsed_result(make_ticket) -> None:
    config = Config()
    expected = Classification(
        category=Category.trading_mechanics, urgency=Urgency.low, confidence=0.8
    )
    client = FakeLLMClient([expected])
    result = classify(
        make_ticket(), [], client=client, assembler=PromptAssembler(config), config=config
    )
    assert result.classification is expected
    assert result.exemplar_ids == []
    assert result.truncated is False
    # the T1 model string was used for the call
    assert client.calls[0].model == config.model_t1
