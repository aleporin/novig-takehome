"""Tests for the prompt assembler: content, provenance, and budget handling."""

from __future__ import annotations

from triage.config import Config
from triage.context.assembler import PromptAssembler
from triage.context.exemplars import EXEMPLAR_IDS, load_exemplar_pool, select_exemplars


def test_prompt_has_skills_taxonomy_and_records_exemplars(make_ticket) -> None:
    config = Config()
    pool = load_exemplar_pool(config.paths.tickets_train)
    ticket = make_ticket(ticket_id="t_eval_999")
    prompt = PromptAssembler(config).classification_prompt(
        ticket, select_exemplars(pool, ticket.ticket_id)
    )
    assert "Novig Support" in prompt.system  # from GLOBAL.md
    assert "## Taxonomy" in prompt.system
    assert prompt.exemplar_ids == list(EXEMPLAR_IDS)
    assert prompt.truncated is False
    assert ticket.subject in prompt.user


def test_tiny_budget_drops_exemplars_and_truncates(make_ticket) -> None:
    config = Config(classify_token_budget=50)
    pool = load_exemplar_pool(config.paths.tickets_train)
    ticket = make_ticket(ticket_id="t1", body="x " * 2000)
    prompt = PromptAssembler(config).classification_prompt(ticket, select_exemplars(pool, "t1"))
    assert prompt.exemplar_ids == []
    assert prompt.truncated is True
