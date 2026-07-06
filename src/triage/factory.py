"""Build the live pipeline pieces from config. The one place that needs a key."""

from __future__ import annotations

from triage.config import Config, load_api_key
from triage.context.assembler import PromptAssembler
from triage.context.exemplars import load_exemplar_pool
from triage.llm.cache import ResponseCache
from triage.llm.client import AnthropicClient, _default_transport
from triage.schemas import Ticket


def build_client(config: Config) -> AnthropicClient:
    """Construct the real Anthropic-backed client with disk caching."""
    key = load_api_key()
    transport = _default_transport(key, config.request_timeout_s)
    cache = ResponseCache(config.paths.cache) if config.cache_enabled else None
    return AnthropicClient(transport, cache=cache, max_retries=config.max_retries)


def build_context(config: Config) -> tuple[PromptAssembler, list[Ticket]]:
    """Load the prompt assembler and the few-shot exemplar pool."""
    return PromptAssembler(config), load_exemplar_pool(config.paths.tickets_train)
