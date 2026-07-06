"""Tests for the LLM client: retries, structured parsing, and caching.

All offline: a scripted transport stands in for the network, so we test the retry
and parsing logic without an API key.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.fakes import FakeLLMClient
from triage.llm.cache import ResponseCache
from triage.llm.client import AnthropicClient
from triage.llm.types import LLMClient, LLMError, LLMRequest, TransientLLMError
from triage.schemas import Category, Classification, Urgency


def test_fake_client_satisfies_protocol() -> None:
    assert isinstance(FakeLLMClient(), LLMClient)


def _usage(i: int = 10, o: int = 5) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=i, output_tokens=o)


def _text_message(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], usage=_usage())


def _tool_message(payload: dict) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", name="emit", input=payload)
    return SimpleNamespace(content=[block], usage=_usage())


def _no_wait_client(transport, **kw) -> AnthropicClient:
    return AnthropicClient(transport, retry_wait_multiplier=0.0, retry_wait_max=0.0, **kw)


def test_plain_text_completion() -> None:
    client = _no_wait_client(lambda params: _text_message("hello world"))
    resp = client.complete(LLMRequest(model="m", system="s", prompt="p"))
    assert resp.text == "hello world"
    assert resp.parsed is None
    assert resp.usage.input_tokens == 10


def test_structured_completion_parses_into_schema() -> None:
    payload = {
        "category": "trading_mechanics",
        "urgency": "low",
        "confidence": 0.9,
        "reasoning": "clear",
        "flags": {},
    }
    client = _no_wait_client(lambda params: _tool_message(payload))
    req = LLMRequest(model="m", system="s", prompt="p", response_schema=Classification)
    resp = client.complete(req)
    assert isinstance(resp.parsed, Classification)
    assert resp.parsed.category == Category.trading_mechanics
    assert resp.parsed.urgency == Urgency.low


def test_transient_errors_are_retried_then_succeed() -> None:
    attempts = {"n": 0}

    def flaky(params):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TransientLLMError("rate limited")
        return _text_message("ok")

    client = _no_wait_client(flaky, max_retries=4)
    assert client.complete(LLMRequest(model="m", system="s", prompt="p")).text == "ok"
    assert attempts["n"] == 3


def test_retries_are_exhausted_and_reraised() -> None:
    def always_fail(params):
        raise TransientLLMError("still down")

    client = _no_wait_client(always_fail, max_retries=3)
    with pytest.raises(TransientLLMError):
        client.complete(LLMRequest(model="m", system="s", prompt="p"))


def test_structured_response_without_tool_use_is_terminal() -> None:
    client = _no_wait_client(lambda params: _text_message("no tool here"))
    req = LLMRequest(model="m", system="s", prompt="p", response_schema=Classification)
    with pytest.raises(LLMError):
        client.complete(req)


def test_invalid_structured_output_raises_llm_error() -> None:
    client = _no_wait_client(lambda params: _tool_message({"category": "not_a_category"}))
    req = LLMRequest(model="m", system="s", prompt="p", response_schema=Classification)
    with pytest.raises(LLMError):
        client.complete(req)


def test_invalid_structured_output_is_never_cached(tmp_path) -> None:
    cache = ResponseCache(tmp_path)
    client = _no_wait_client(
        lambda params: _tool_message({"category": "not_a_category"}), cache=cache
    )
    req = LLMRequest(model="m", system="s", prompt="p", response_schema=Classification)
    with pytest.raises(LLMError):
        client.complete(req)
    assert cache.get(req.cache_key()) is None  # the bad response did not poison the cache


def test_second_identical_call_is_served_from_cache(tmp_path) -> None:
    calls = {"n": 0}

    def counting(params):
        calls["n"] += 1
        return _text_message("cached-value")

    client = _no_wait_client(counting, cache=ResponseCache(tmp_path))
    req = LLMRequest(model="m", system="s", prompt="p")
    first = client.complete(req)
    second = client.complete(req)
    assert calls["n"] == 1
    assert first.cache_hit is False and second.cache_hit is True
    assert second.text == "cached-value"
