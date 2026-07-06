"""Classification stage: one LLM call that returns a structured Classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from triage.config import Config
from triage.context.assembler import PromptAssembler
from triage.llm.types import LLMClient, LLMRequest
from triage.logging_setup import (
    CACHE_HIT,
    LATENCY_MS,
    MODEL,
    STAGE,
    TIER,
    TOKENS_IN,
    TOKENS_OUT,
)
from triage.schemas import Classification, Ticket

logger = logging.getLogger(__name__)


@dataclass
class ClassifyResult:
    """The classification plus the provenance and cost of the call."""

    classification: Classification
    exemplar_ids: list[str]
    truncated: bool
    latency_ms: float
    cache_hit: bool
    tokens_in: int
    tokens_out: int


def classify(
    ticket: Ticket,
    exemplars: list[Ticket],
    *,
    client: LLMClient,
    assembler: PromptAssembler,
    config: Config,
) -> ClassifyResult:
    """Run the T1 classifier on one ticket."""
    prompt = assembler.classification_prompt(ticket, exemplars)
    request = LLMRequest(
        model=config.model_t1,
        system=prompt.system,
        prompt=prompt.user,
        response_schema=Classification,
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
    )
    response = client.complete(request)
    classification: Classification = response.parsed

    logger.info(
        "classified",
        extra={
            STAGE: "classify",
            TIER: "T1",
            MODEL: config.model_t1,
            CACHE_HIT: response.cache_hit,
            LATENCY_MS: round(response.latency_ms, 1),
            TOKENS_IN: response.usage.input_tokens,
            TOKENS_OUT: response.usage.output_tokens,
        },
    )
    return ClassifyResult(
        classification=classification,
        exemplar_ids=prompt.exemplar_ids,
        truncated=prompt.truncated,
        latency_ms=response.latency_ms,
        cache_hit=response.cache_hit,
        tokens_in=response.usage.input_tokens,
        tokens_out=response.usage.output_tokens,
    )
