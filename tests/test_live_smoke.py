"""Live smoke test: one real API call that checks auth and structured output.

Marked live, so it's skipped by default (the offline suite stays key-free). Run it
with make smoke. Skips if no key is set.
"""

from __future__ import annotations

import os

import pytest

from triage.config import Config, load_api_key
from triage.llm.client import AnthropicClient, _default_transport
from triage.schemas import Classification

pytestmark = pytest.mark.live


@pytest.fixture
def live_client() -> AnthropicClient:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        try:
            load_api_key()
        except RuntimeError:
            pytest.skip("ANTHROPIC_API_KEY not configured")
    cfg = Config()
    transport = _default_transport(os.environ["ANTHROPIC_API_KEY"], cfg.request_timeout_s)
    return AnthropicClient(transport, max_retries=cfg.max_retries)


def test_structured_classification_round_trips_live(live_client: AnthropicClient) -> None:
    from triage.llm.types import LLMRequest

    request = LLMRequest(
        model=Config().model_t1,
        system="You classify support tickets. Return the structured result.",
        prompt="Ticket: 'how do parlays work on novig?' Classify it.",
        response_schema=Classification,
        temperature=0.0,
        max_tokens=512,
    )
    response = live_client.complete(request)
    assert isinstance(response.parsed, Classification)
    assert response.usage.output_tokens > 0
