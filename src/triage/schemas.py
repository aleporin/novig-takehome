"""Data shapes for tickets (input) and predictions (output).

Every other module speaks in these types. The key one is Prediction: its
validator rejects contradictory results (drafted but also declined, or declined
with no reason), so a bad prediction can't be built. extra="forbid" means an
unexpected field raises instead of being silently dropped.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Category(StrEnum):
    """Ticket categories. Values match the dataset labels exactly."""

    account_access = "account_access"
    kyc_verification = "kyc_verification"
    deposits_withdrawals = "deposits_withdrawals"
    trading_mechanics = "trading_mechanics"
    market_questions = "market_questions"
    bug_report = "bug_report"
    tax_documents = "tax_documents"
    account_compromise = "account_compromise"
    problem_gambling = "problem_gambling"
    legal_regulatory = "legal_regulatory"
    other = "other"


class Urgency(StrEnum):
    """Urgency levels. Note the underscore in escalate_immediately."""

    low = "low"
    medium = "medium"
    high = "high"
    escalate_immediately = "escalate_immediately"


# Categories that always block drafting on their own. The gate imports this set.
SENSITIVE_CATEGORIES: frozenset[Category] = frozenset(
    {Category.account_compromise, Category.problem_gambling, Category.legal_regulatory}
)


class TicketMetadata(BaseModel):
    """Per-ticket metadata. Not part of the output."""

    model_config = ConfigDict(extra="forbid")

    user_tenure_days: int
    submitted_at: datetime


class Label(BaseModel):
    """The correct answer. Present on train tickets only."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    urgency: Urgency
    should_draft: bool
    no_draft_reason: str | None = None
    gold_response_notes: str


class Ticket(BaseModel):
    """A support ticket. Train tickets have a label; eval tickets don't.

    Parsing is strict: an unknown field raises instead of being dropped, so we
    never quietly discard a ticket we didn't fully understand.
    """

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    subject: str
    body: str
    metadata: TicketMetadata
    label: Label | None = None


class RiskFlags(BaseModel):
    """Risk signals the classifier sets, separate from category.

    Any flag can fire in any category (a deposits_withdrawals ticket can still
    mention a minor), so category alone is never the only safety check.
    """

    model_config = ConfigDict(extra="forbid")

    mentions_minor: bool = False
    self_harm_or_distress: bool = False
    active_fraud: bool = False
    unauthorized_access_reported: bool = False
    rg_signal: bool = False
    legal_threat: bool = False
    disputes_novig_fact: bool = False
    asks_binding_policy_or_spec: bool = False
    jurisdictional_eligibility: bool = False


class Classification(BaseModel):
    """What the classifier returns. Internal only, never part of the output."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    urgency: Urgency
    flags: RiskFlags = Field(default_factory=RiskFlags)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class Prediction(BaseModel):
    """One line of predictions.jsonl: exactly seven fields.

    The validator enforces the draft/decline rules below, so you can't build a
    contradictory result.
    """

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    category: Category
    urgency: Urgency
    should_draft: bool
    no_draft_reason: str | None = None
    draft_response: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _enforce_mutual_exclusion(self) -> Prediction:
        if self.should_draft:
            if self.no_draft_reason is not None:
                raise ValueError("should_draft=true requires no_draft_reason=null")
            if self.draft_response is None:
                raise ValueError("should_draft=true requires a non-null draft_response")
        else:
            if self.draft_response is not None:
                raise ValueError("should_draft=false requires draft_response=null")
            if not (self.no_draft_reason and self.no_draft_reason.strip()):
                raise ValueError("should_draft=false requires a non-empty no_draft_reason")
        return self

    def to_jsonl_line(self) -> str:
        """Return one compact JSON line. Field order matches the declared order."""
        return self.model_dump_json()
