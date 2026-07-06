"""Builds the classification prompt from skills, taxonomy, exemplars, and ticket.

Composes in priority order: global skill -> task skill -> taxonomy -> few-shot
exemplars -> ticket. If the estimated size exceeds the token budget, exemplars are
dropped first (from the end) and the ticket body is truncated last. Any truncation
is recorded so it can be logged and reflected in confidence. Records the exemplar
ids actually used, so the eval can exclude prompted tickets from headline metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from triage.config import Config
from triage.logging_setup import STAGE
from triage.schemas import Ticket

from .exemplars import EXEMPLAR_FLAGS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssembledPrompt:
    """A ready-to-send prompt plus the provenance the eval needs."""

    system: str
    user: str
    exemplar_ids: list[str]
    truncated: bool


def _estimate_tokens(text: str) -> int:
    """Rough token count: about four characters per token."""
    return len(text) // 4


def _ticket_block(ticket: Ticket) -> str:
    return f"SUBJECT: {ticket.subject}\nBODY: {ticket.body}"


def _exemplar_block(ticket: Ticket) -> str:
    """One few-shot example: the ticket text and its target classification JSON."""
    label = ticket.label
    assert label is not None, "exemplars must be labeled train tickets"
    flags = {name: True for name in EXEMPLAR_FLAGS.get(ticket.ticket_id, ())}
    target = {
        "category": label.category.value,
        "urgency": label.urgency.value,
        "flags": flags,
        "confidence": 0.95,
    }
    return f"{_ticket_block(ticket)}\nCLASSIFICATION: {json.dumps(target)}"


class PromptAssembler:
    """Loads the prompt text once and assembles per-ticket classification prompts."""

    def __init__(self, config: Config) -> None:
        skills = config.paths.skills
        self._global = (skills / "GLOBAL.md").read_text(encoding="utf-8")
        self._task = (skills / "classify" / "SKILL.md").read_text(encoding="utf-8")
        self._taxonomy = config.paths.taxonomy.read_text(encoding="utf-8")
        self._budget = config.classify_token_budget

    def _system(self, exemplar_blocks: list[str]) -> str:
        parts = [self._global, self._task, "## Taxonomy\n" + self._taxonomy]
        if exemplar_blocks:
            parts.append("## Examples\n\n" + "\n\n".join(exemplar_blocks))
        return "\n\n".join(parts)

    def classification_prompt(self, ticket: Ticket, exemplars: list[Ticket]) -> AssembledPrompt:
        """Assemble a prompt, dropping exemplars then truncating the ticket to fit."""
        used = list(exemplars)
        blocks = [_exemplar_block(t) for t in used]
        user = _ticket_block(ticket)

        while used and _estimate_tokens(self._system(blocks) + user) > self._budget:
            used.pop()
            blocks.pop()

        truncated = False
        fixed_tokens = _estimate_tokens(self._system(blocks))
        if fixed_tokens + _estimate_tokens(user) > self._budget:
            budget_chars = max(0, (self._budget - fixed_tokens)) * 4
            user = user[:budget_chars]
            truncated = True
            logger.warning("ticket truncated to fit budget", extra={STAGE: "assemble"})

        return AssembledPrompt(
            system=self._system(blocks),
            user=user,
            exemplar_ids=[t.ticket_id for t in used],
            truncated=truncated,
        )
