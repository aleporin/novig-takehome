"""Output guardrail on a generated draft: deterministic checks + a cheap LLM audit.

Deterministic scan catches the mechanical violations (guarantees, promised dates,
invented dollar amounts, missing signature, length). The LLM audit (cheap model)
catches ungrounded policy or disputed-fact claims. On any violation the draft is
regenerated once with the violation named; a second failure downgrades to no-draft.
Fail-closed: a draft never ships past a violation.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel

from triage.config import Config
from triage.context.assembler import PromptAssembler
from triage.llm.types import LLMClient, LLMError, LLMRequest
from triage.logging_setup import STAGE
from triage.routing.cost_tracker import CallRecord, call_cost
from triage.schemas import Classification, Ticket

from .draft import draft_model, generate_draft

logger = logging.getLogger(__name__)

_I = re.IGNORECASE
_BANNED = [
    (re.compile(r"\bguarantee", _I), "contains a guarantee"),
    (
        re.compile(r"\bby (mon|tues|wednes|thurs|fri|satur|sun)day\b", _I),
        "promises a day-of-week outcome",
    ),
    (re.compile(r"\bby (tomorrow|tonight|end of (the )?(day|week))\b", _I), "promises a deadline"),
    (
        re.compile(
            r"\bwill (land|arrive|be (credited|resolved|fixed|completed|processed|approved))\b", _I
        ),
        "promises an outcome",
    ),
]
_AMOUNT = re.compile(r"\$[\d,]+(?:\.\d+)?")
_STANDARD_AMOUNTS = {"$0", "$1"}  # standard card auth etc., not account-specific figures

_AUDIT_SYSTEM = (
    "You audit a draft support reply. Say whether it commits any of: promises an "
    "outcome the company cannot verify; states a policy, fee, or rule not grounded "
    "in the ticket; or confirms or denies a specific figure the user disputed. "
    "Return violates=true with a short reason, else violates=false."
)


class AuditVerdict(BaseModel):
    """The cheap LLM audit's yes/no verdict."""

    violates: bool
    reason: str = ""


def deterministic_violations(draft: str, ticket: Ticket, config: Config) -> list[str]:
    """Mechanical banned-commitment checks over the draft text."""
    violations = [msg for pattern, msg in _BANNED if pattern.search(draft)]
    if "Novig Support" not in draft:
        violations.append("missing 'Novig Support' signature")
    if len(draft) > config.draft_max_chars:
        violations.append(f"too long (> {config.draft_max_chars} chars)")
    if len(draft.strip()) < 20:
        violations.append("draft too short or empty")

    ticket_amounts = set(_AMOUNT.findall(f"{ticket.subject} {ticket.body}"))
    invented = set(_AMOUNT.findall(draft)) - ticket_amounts - _STANDARD_AMOUNTS
    if invented:
        violations.append(f"invented dollar amount(s) not in ticket: {sorted(invented)}")
    return violations


def audit_draft(
    draft: str, ticket: Ticket, *, client: LLMClient, config: Config
) -> tuple[AuditVerdict, CallRecord]:
    """Cheap LLM yes/no audit for ungrounded policy or disputed-fact claims."""
    request = LLMRequest(
        model=config.model_t1,
        system=_AUDIT_SYSTEM,
        prompt=f"TICKET:\n{ticket.subject}\n{ticket.body}\n\nDRAFT:\n{draft}",
        response_schema=AuditVerdict,
        temperature=config.temperature,
        max_tokens=256,
    )
    response = client.complete(request)
    cost = call_cost(config.model_t1, response.usage.input_tokens, response.usage.output_tokens)
    call = CallRecord(
        config.model_t1,
        "audit",
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.latency_ms,
        response.cache_hit,
        cost,
    )
    return response.parsed, call


def check_draft(
    draft: str, ticket: Ticket, *, client: LLMClient, config: Config
) -> tuple[list[str], CallRecord]:
    """Return all violations for a draft plus the audit call record."""
    violations = deterministic_violations(draft, ticket, config)
    verdict, call = audit_draft(draft, ticket, client=client, config=config)
    if verdict.violates:
        violations.append(f"audit: {verdict.reason}")
    return violations, call


def produce_draft(
    ticket: Ticket,
    classification: Classification,
    *,
    client: LLMClient,
    assembler: PromptAssembler,
    config: Config,
) -> tuple[str | None, list[CallRecord]]:
    """Generate, guard, regenerate once, then downgrade. None means downgrade to no-draft."""
    calls: list[CallRecord] = []
    try:
        first = generate_draft(
            ticket,
            classification.category,
            client=client,
            assembler=assembler,
            config=config,
            model=draft_model(classification, config),
        )
        calls.append(first.call)
        violations, audit_call = check_draft(first.text, ticket, client=client, config=config)
        calls.append(audit_call)
        if not violations:
            return first.text, calls

        logger.warning("draft failed guardrail; regenerating", extra={STAGE: "output_guard"})
        second = generate_draft(
            ticket,
            classification.category,
            client=client,
            assembler=assembler,
            config=config,
            model=config.model_t2,
            feedback="; ".join(violations),
        )
        calls.append(second.call)
        violations2, audit_call2 = check_draft(second.text, ticket, client=client, config=config)
        calls.append(audit_call2)
        if not violations2:
            return second.text, calls

        logger.warning(
            "draft failed guardrail after regen; downgrading", extra={STAGE: "output_guard"}
        )
        return None, calls
    except LLMError:
        logger.exception("draft generation failed; downgrading", extra={STAGE: "output_guard"})
        return None, calls
