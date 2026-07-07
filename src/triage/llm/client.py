"""Anthropic-backed LLMClient: retries, structured output, and caching.

In normal use this calls the real Anthropic API. The network call is a swappable
"transport" function, so tests can pass a fake one and exercise the retry, parse,
and cache logic without a key. The default transport wraps the SDK and sorts
provider errors into retryable vs not.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from time import perf_counter
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .cache import ResponseCache
from .types import LLMError, LLMRequest, LLMResponse, TransientLLMError, Usage

logger = logging.getLogger(__name__)

Transport = Callable[[dict[str, Any]], Any]


def _default_transport(api_key: str | None, timeout_s: float) -> Transport:
    """Build a transport that calls the Anthropic SDK and maps its errors to ours.

    The SDK client is built lazily on the first live call, so a fully cached run
    needs no key; a cache miss with no key raises where the call is actually made.
    """
    import anthropic

    holder: dict[str, Any] = {}

    def _call(params: dict[str, Any]) -> Any:
        client = holder.get("client")
        if client is None:
            if not api_key:
                raise LLMError("ANTHROPIC_API_KEY is required for a live (cache-miss) call")
            client = holder["client"] = anthropic.Anthropic(api_key=api_key, timeout=timeout_s)
        try:
            return client.messages.create(**params)
        except (
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        ) as exc:
            raise TransientLLMError(str(exc)) from exc
        except anthropic.APIError as exc:
            raise LLMError(str(exc)) from exc

    return _call


class AnthropicClient:
    """Calls a model, with retries and optional caching. No routing logic here."""

    def __init__(
        self,
        transport: Transport,
        *,
        cache: ResponseCache | None = None,
        max_retries: int = 4,
        retry_wait_multiplier: float = 0.5,
        retry_wait_max: float = 8.0,
    ) -> None:
        self._transport = transport
        self._cache = cache
        self._max_retries = max_retries
        self._retry_wait_multiplier = retry_wait_multiplier
        self._retry_wait_max = retry_wait_max

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Run the request. Returns a cached response if we've seen it before."""
        key = request.cache_key()
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return self._response_from_payload(request, cached, cache_hit=True)

        started = perf_counter()
        message = self._call_with_retry(_build_params(request))
        latency_ms = (perf_counter() - started) * 1000.0

        text, usage = _extract(message, structured=request.response_schema is not None)
        payload = {
            "text": text,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }
        if self._cache is not None:
            # Validate before caching: a response that won't parse is never stored.
            self._cache.set(key, payload, validator=lambda p: _parse(request, p["text"]))
        response = self._response_from_payload(request, payload, cache_hit=False)
        response.latency_ms = latency_ms
        return response

    def _call_with_retry(self, params: dict[str, Any]) -> Any:
        retryer = retry(
            reraise=True,
            retry=retry_if_exception_type(TransientLLMError),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=self._retry_wait_multiplier, max=self._retry_wait_max),
        )
        return retryer(self._transport)(params)

    def _response_from_payload(
        self, request: LLMRequest, payload: dict, *, cache_hit: bool
    ) -> LLMResponse:
        parsed = _parse(request, payload["text"])
        return LLMResponse(
            text=payload["text"],
            usage=Usage(payload["input_tokens"], payload["output_tokens"]),
            model=request.model,
            latency_ms=0.0,
            cache_hit=cache_hit,
            parsed=parsed,
        )


_STRUCTURED_TOOL = "emit"


def _build_params(request: LLMRequest) -> dict[str, Any]:
    """Convert an LLMRequest into Anthropic Messages API arguments."""
    params: dict[str, Any] = {
        "model": request.model,
        "system": request.system,
        "max_tokens": request.max_tokens,
        "messages": [{"role": "user", "content": request.prompt}],
    }
    if request.temperature is not None:
        params["temperature"] = request.temperature
    if request.response_schema is not None:
        params["tools"] = [
            {
                "name": _STRUCTURED_TOOL,
                "description": "Return the structured result.",
                "input_schema": request.response_schema.model_json_schema(),
            }
        ]
        params["tool_choice"] = {"type": "tool", "name": _STRUCTURED_TOOL}
    return params


def _extract(message: Any, *, structured: bool) -> tuple[str, Usage]:
    """Get the text and token counts out of a model response."""
    usage = Usage(message.usage.input_tokens, message.usage.output_tokens)
    if structured:
        for block in message.content:
            if getattr(block, "type", None) == "tool_use":
                return json.dumps(block.input, sort_keys=True), usage
        raise LLMError("structured request returned no tool_use block")
    parts = [b.text for b in message.content if getattr(b, "type", None) == "text"]
    return "".join(parts), usage


def _unwrap_single_key(text: str) -> dict | None:
    """Some models wrap the object under one top-level key (e.g. {"Classification": {...}})."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and len(data) == 1:
        inner = next(iter(data.values()))
        if isinstance(inner, dict):
            return inner
    return None


def _parse(request: LLMRequest, text: str) -> Any | None:
    """Validate structured output against its schema. Returns None for plain text."""
    if request.response_schema is None:
        return None
    try:
        return request.response_schema.model_validate_json(text)
    except Exception as exc:
        unwrapped = _unwrap_single_key(text)
        if unwrapped is not None:
            try:
                return request.response_schema.model_validate(unwrapped)
            except Exception:
                pass
        # Any validation failure becomes our terminal error type.
        raise LLMError(f"structured output failed schema validation: {exc}") from exc
