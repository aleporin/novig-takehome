"""Draft generation: one LLM call that returns reply text (no structured schema).

Routed by complexity: money-in-motion or high-urgency tickets get the stronger
model, everything else the cheap one. Regeneration (after a guardrail rejection)
always uses the stronger model.
"""

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
    TOKENS_IN,
    TOKENS_OUT,
)
from triage.routing.cost_tracker import CallRecord, call_cost
from triage.schemas import Category, Classification, Ticket, Urgency

logger = logging.getLogger(__name__)

# Money-in-motion categories draft on the stronger model even at low urgency.
_MONEY_IN_MOTION = frozenset({Category.deposits_withdrawals})


@dataclass
class DraftResult:
    """A generated draft and its call accounting."""

    text: str
    call: CallRecord


def draft_model(classification: Classification, config: Config) -> str:
    """Pick the draft model: stronger for money-in-motion or high-urgency tickets."""
    if classification.category in _MONEY_IN_MOTION:
        return config.model_t2
    if classification.urgency in (Urgency.high, Urgency.escalate_immediately):
        return config.model_t2
    return config.model_t1


def generate_draft(
    ticket: Ticket,
    category: Category,
    *,
    client: LLMClient,
    assembler: PromptAssembler,
    config: Config,
    model: str,
    feedback: str | None = None,
) -> DraftResult:
    """Generate one draft reply with the given model."""
    prompt = assembler.drafting_prompt(ticket, category, feedback=feedback)
    temperature = None if model in TEMPERATURE_DEPRECATED else config.temperature
    request = LLMRequest(
        model=model,
        system=prompt.system,
        prompt=prompt.user,
        response_schema=None,
        temperature=temperature,
        max_tokens=config.max_output_tokens,
    )
    response = client.complete(request)
    cost = call_cost(model, response.usage.input_tokens, response.usage.output_tokens)
    logger.info(
        "drafted",
        extra={
            STAGE: "draft",
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
        tier="draft",
        tokens_in=response.usage.input_tokens,
        tokens_out=response.usage.output_tokens,
        latency_ms=response.latency_ms,
        cache_hit=response.cache_hit,
        cost_usd=cost,
    )
    return DraftResult(response.text.strip(), call)
