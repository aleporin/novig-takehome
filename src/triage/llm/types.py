"""The interface every LLM caller uses.

Requests and responses are plain objects, not the Anthropic SDK's types. That
keeps the fake client simple and lets the pipeline run in tests with no network.
Nothing outside this package imports the SDK directly; everything goes through
LLMClient.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class LLMError(Exception):
    """A failure not worth retrying: bad request, or output that won't parse."""


class TransientLLMError(Exception):
    """A failure worth retrying: timeout, rate limit, connection reset."""


@dataclass(frozen=True)
class LLMRequest:
    """One model call. Setting response_schema asks the model for structured output."""

    model: str
    system: str
    prompt: str
    response_schema: type[BaseModel] | None = None
    temperature: float = 0.0
    max_tokens: int = 1024

    def cache_key(self) -> str:
        """Hash of everything that affects the output.

        Same inputs give the same key; changing the model, prompt, schema, or
        sampling params changes it.
        """
        schema = self.response_schema.model_json_schema() if self.response_schema else None
        canonical = json.dumps(
            {
                "model": self.model,
                "system": self.system,
                "prompt": self.prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "schema": schema,
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Usage:
    """Token counts for one call."""

    input_tokens: int
    output_tokens: int


@dataclass
class LLMResponse:
    """Result of a model call. parsed is set only for structured requests."""

    text: str
    usage: Usage
    model: str
    latency_ms: float
    cache_hit: bool
    parsed: Any | None = None


@runtime_checkable
class LLMClient(Protocol):
    """What the pipeline calls. The real client and the fake both implement this."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Run the request and return the response. Raises LLMError if it can't."""
        ...
