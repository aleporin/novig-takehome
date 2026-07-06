"""Cache tests: hit/miss determinism, key stability, atomic writes."""

from __future__ import annotations

from triage.llm.cache import ResponseCache
from triage.llm.types import LLMRequest
from triage.schemas import Classification


def test_set_then_get_roundtrips(tmp_path) -> None:
    cache = ResponseCache(tmp_path)
    cache.set("k1", {"text": "hello", "input_tokens": 1, "output_tokens": 2})
    assert cache.get("k1") == {"text": "hello", "input_tokens": 1, "output_tokens": 2}


def test_missing_key_returns_none(tmp_path) -> None:
    assert ResponseCache(tmp_path).get("absent") is None


def test_no_temp_file_left_after_set(tmp_path) -> None:
    ResponseCache(tmp_path).set("k", {"text": "x", "input_tokens": 0, "output_tokens": 0})
    assert not list(tmp_path.glob("*.tmp"))


def test_set_does_not_write_when_validator_raises(tmp_path) -> None:
    import pytest

    cache = ResponseCache(tmp_path)

    def reject(_payload):
        raise ValueError("malformed")

    with pytest.raises(ValueError):
        cache.set("k", {"text": "bad"}, validator=reject)
    assert cache.get("k") is None  # nothing was cached


def test_set_writes_when_validator_passes(tmp_path) -> None:
    cache = ResponseCache(tmp_path)
    cache.set("k", {"text": "ok"}, validator=lambda p: None)
    assert cache.get("k") == {"text": "ok"}


def test_cache_key_is_stable_and_content_addressed() -> None:
    a = LLMRequest(model="m", system="s", prompt="p")
    b = LLMRequest(model="m", system="s", prompt="p")
    assert a.cache_key() == b.cache_key()


def test_cache_key_changes_with_prompt() -> None:
    a = LLMRequest(model="m", system="s", prompt="p1")
    b = LLMRequest(model="m", system="s", prompt="p2")
    assert a.cache_key() != b.cache_key()


def test_cache_key_changes_with_schema() -> None:
    plain = LLMRequest(model="m", system="s", prompt="p")
    structured = LLMRequest(model="m", system="s", prompt="p", response_schema=Classification)
    assert plain.cache_key() != structured.cache_key()
