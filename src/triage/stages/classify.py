"""Classification stage: one LLM call that returns a structured Classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from triage.config import TEMPERATURE_DEPRECATED, Config
from triage.context.assembler import PromptAssembler
from triage.llm.types import LLMClient, LLMRequest
from triage.logging_setup import (
    CACHE_HIT,
    COST_USD,
    LATENCY_MS,
    MODEL,
    STAGE,
    TIER,
    TOKENS_IN,
    TOKENS_OUT,
)
from triage.routing.cost_tracker import CallRecord, call_cost
from triage.schemas import Classification, Ticket

logger = logging.getLogger(__name__)


@dataclass
class ClassifyResult:
    """The classification plus the call's provenance and cost."""

    classification: Classification
    exemplar_ids: list[str]
    truncated: bool
    call: CallRecord


def classify(
    ticket: Ticket,
    exemplars: list[Ticket],
    *,
    client: LLMClient,
    assembler: PromptAssembler,
    config: Config,
    model: str,
    tier: str,
) -> ClassifyResult:
    """Run one classification call with the given model, at the given tier."""
    prompt = assembler.classification_prompt(ticket, exemplars)
    temperature = None if model in TEMPERATURE_DEPRECATED else config.temperature
    request = LLMRequest(
        model=model,
        system=prompt.system,
        prompt=prompt.user,
        response_schema=Classification,
        temperature=temperature,
        max_tokens=config.max_output_tokens,
    )
    response = client.complete(request)
    cost = call_cost(model, response.usage.input_tokens, response.usage.output_tokens)

    logger.info(
        "classified",
        extra={
            STAGE: "classify",
            TIER: tier,
            MODEL: model,
            CACHE_HIT: response.cache_hit,
            LATENCY_MS: round(response.latency_ms, 1),
            TOKENS_IN: response.usage.input_tokens,
            TOKENS_OUT: response.usage.output_tokens,
            COST_USD: round(cost, 6),
        },
    )
    call = CallRecord(
        model=model,
        tier=tier,
        tokens_in=response.usage.input_tokens,
        tokens_out=response.usage.output_tokens,
        latency_ms=response.latency_ms,
        cache_hit=response.cache_hit,
        cost_usd=cost,
    )
    return ClassifyResult(response.parsed, prompt.exemplar_ids, prompt.truncated, call)
