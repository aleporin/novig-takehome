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
import re
from dataclasses import dataclass

from triage.config import Config
from triage.logging_setup import STAGE
from triage.schemas import Category, Ticket

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


# Taxonomy sections the classifier needs. The Output Schema and Notes on Drafting
# sections describe the prediction shape and drafting, not classification, so they
# are left out to avoid pointing the model at a different JSON shape.
_TAXONOMY_SECTIONS = ("Categories", "Urgency Levels", "When the System Must Not Draft")


def _taxonomy_excerpt(text: str) -> str:
    """Keep only the H2 sections a classifier needs, dropping the rest."""
    kept = []
    for chunk in re.split(r"^## ", text, flags=re.MULTILINE):
        title = chunk.splitlines()[0].strip() if chunk.strip() else ""
        if title in _TAXONOMY_SECTIONS:
            kept.append("## " + chunk.rstrip())
    return "\n\n".join(kept)


def _ticket_block(ticket: Ticket) -> str:
    return f"SUBJECT: {ticket.subject}\nBODY: {ticket.body}"


def _exemplar_block(ticket: Ticket) -> str:
    """One few-shot example: the ticket text and its target classification JSON.

    No confidence is shown: a constant would anchor the model and blind the
    confidence-based escalation trigger, so the model self-reports it per ticket.
    """
    label = ticket.label
    assert label is not None, "exemplars must be labeled train tickets"
    flags = {name: True for name in EXEMPLAR_FLAGS.get(ticket.ticket_id, ())}
    target = {
        "category": label.category.value,
        "urgency": label.urgency.value,
        "flags": flags,
    }
    return f"{_ticket_block(ticket)}\nCLASSIFICATION: {json.dumps(target)}"


class PromptAssembler:
    """Loads the prompt text once and assembles per-ticket classification prompts."""

    def __init__(self, config: Config) -> None:
        skills = config.paths.skills
        self._global = (skills / "GLOBAL.md").read_text(encoding="utf-8")
        self._task = (skills / "classify" / "SKILL.md").read_text(encoding="utf-8")
        self._taxonomy = _taxonomy_excerpt(config.paths.taxonomy.read_text(encoding="utf-8"))
        self._budget = config.classify_token_budget
        self._draft = (skills / "draft" / "SKILL.md").read_text(encoding="utf-8")
        self._per_category = skills / "draft" / "per_category"

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

    def _category_guidance(self, category: Category) -> str:
        path = self._per_category / f"{category.value}.md"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def drafting_prompt(
        self, ticket: Ticket, category: Category, *, feedback: str | None = None
    ) -> AssembledPrompt:
        """Assemble the drafting prompt: globals, draft rules, category guidance, ticket.

        ``feedback`` names a prior draft's guardrail violation for the one regeneration.
        """
        parts = [self._global, self._draft]
        guidance = self._category_guidance(category)
        if guidance:
            parts.append("## Category guidance\n" + guidance)

        user = _ticket_block(ticket)
        if feedback:
            user += f"\n\nYour previous draft was rejected: {feedback}\nWrite a corrected reply."
        return AssembledPrompt(
            system="\n\n".join(parts), user=user, exemplar_ids=[], truncated=False
        )
