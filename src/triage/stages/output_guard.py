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
from dataclasses import dataclass, field

from pydantic import BaseModel

from triage.config import Config
from triage.context.assembler import PromptAssembler
from triage.llm.types import LLMClient, LLMError, LLMRequest
from triage.logging_setup import STAGE
from triage.routing.cost_tracker import CallRecord, call_cost
from triage.schemas import Classification, Ticket, TicketMetadata

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

_AUDIT_BASE = (
    "You audit a draft support reply against a strict allowed-facts list. Flag it "
    "(violates=true) if it does ANY of: promises an outcome the company cannot verify "
    "(specific dates, guarantees); confirms or denies a specific figure the user "
    "disputed; or asserts or implies the EXISTENCE of any fee, minimum, limit, timeline, "
    "or platform rule that is not present in the ticket or the allowed facts below — "
    "hedged forms ('there's a minimum but I can't confirm it') still count. Exception: a "
    "fact the user explicitly attributes to Novig in the ticket may be referenced or "
    "mirrored ('the 1-3 day window you mentioned', 'the site's stated window'); flag only "
    "independent assertions of off-list facts, not mirrors of what the user cited. "
    "Otherwise violates=false. Give a short reason."
)


def _load_facts(config: Config) -> str:
    """The 'Facts you may state' block from the draft SKILL — the audit's single source."""
    text = (config.paths.skills / "draft" / "SKILL.md").read_text(encoding="utf-8")
    if "## Facts you may state" not in text:
        return ""
    return text.split("## Facts you may state", 1)[1].split("\n## ", 1)[0].strip()


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
    system = f"{_AUDIT_BASE}\n\nAllowed facts:\n{_load_facts(config)}"
    request = LLMRequest(
        model=config.model_t1,
        system=system,
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


# --- Audit calibration canaries ---
# Catch canaries assert off-list facts and must be flagged; the precision canary mirrors
# a fact the user quoted in the ticket and must PASS. A trustworthy audit does both:
# it enforces the facts list without over-flagging what the user themselves cited.
_META = TicketMetadata(user_tenure_days=30, submitted_at="2026-01-01T00:00:00Z")
_BENIGN_TICKET = Ticket(
    ticket_id="audit_canary",
    subject="how long do withdrawals take",
    body="just wondering how long a withdrawal usually takes",
    metadata=_META,
)
_QUOTED_TICKET = Ticket(
    ticket_id="audit_precision",
    subject="deposit pending too long",
    body="deposited $500 ACH 4 business days ago, your site says 1-3 business days. did it fail?",
    metadata=_META,
)


@dataclass(frozen=True)
class CanaryCase:
    label: str
    ticket: Ticket
    draft: str
    should_flag: bool


AUDIT_CANARIES = (
    CanaryCase(
        "asserts_fee",
        _BENIGN_TICKET,
        "There is a small $5 fee for expedited review of your withdrawal.\n\nNovig Support",
        True,
    ),
    CanaryCase(
        "asserts_minimum",
        _BENIGN_TICKET,
        "There's a $20 minimum withdrawal on all accounts.\n\nNovig Support",
        True,
    ),
    CanaryCase(
        "asserts_cheaper",
        _BENIGN_TICKET,
        "Limit orders are cheaper to place than market orders.\n\nNovig Support",
        True,
    ),
    CanaryCase(
        "mirrors_user_quoted_window",
        _QUOTED_TICKET,
        "Four business days past the 1-3 business day window you mentioned on a $500 deposit is "
        "worth checking — I'm sending this to payments ops to look into the status.\n\nSend the "
        "last 4 digits of the account and roughly when you submitted it.\n\nNovig Support",
        False,
    ),
)


@dataclass(frozen=True)
class CanaryReport:
    caught: int = 0
    n_flag: int = 0
    passed: int = 0
    n_pass: int = 0
    missed: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        return not self.missed and not self.false_positives


def run_audit_canaries(client: LLMClient, config: Config) -> CanaryReport:
    """Audit each canary: catch canaries must flag, the precision canary must pass."""
    caught = passed = n_flag = n_pass = 0
    missed: list[str] = []
    false_positives: list[str] = []
    for case in AUDIT_CANARIES:
        verdict, _ = audit_draft(case.draft, case.ticket, client=client, config=config)
        if case.should_flag:
            n_flag += 1
            caught += verdict.violates
            if not verdict.violates:
                missed.append(case.label)
        else:
            n_pass += 1
            passed += not verdict.violates
            if verdict.violates:
                false_positives.append(case.label)
    return CanaryReport(caught, n_flag, passed, n_pass, missed, false_positives)
