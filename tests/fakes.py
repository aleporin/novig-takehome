"""A scripted LLMClient for offline tests.

Tests queue up the exact responses they want; this returns them in order, so they
can drive any path without a network call or API key.
"""

from __future__ import annotations

from pydantic import BaseModel

from triage.llm.types import LLMRequest, LLMResponse, Usage


class FakeLLMClient:
    """Returns pre-scripted responses in order; records every request it saw."""

    def __init__(self, responses: list[BaseModel | str] | None = None) -> None:
        self._queue: list[BaseModel | str] = list(responses or [])
        self.calls: list[LLMRequest] = []

    def queue(self, item: BaseModel | str) -> FakeLLMClient:
        """Add a response to the queue. Returns self so calls can chain."""
        self._queue.append(item)
        return self

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if not self._queue:
            raise AssertionError("FakeLLMClient exhausted: no scripted response queued")
        item = self._queue.pop(0)
        if isinstance(item, BaseModel):
            text, parsed = item.model_dump_json(), item
        else:
            text, parsed = str(item), None
        return LLMResponse(
            text=text,
            usage=Usage(input_tokens=0, output_tokens=0),
            model=request.model,
            latency_ms=0.0,
            cache_hit=False,
            parsed=parsed,
        )
