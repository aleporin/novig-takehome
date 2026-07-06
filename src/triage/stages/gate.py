"""The safety gate: the deterministic hard-rule decision.

Pure function, no AI, no config that can soften it. It blocks drafting when the
category is sensitive, or when any hard-rule flag is set by either the classifier
or the pre-screen. The two flag sources are independent vetoes: a flag from either
one blocks, and neither can clear the other's flag. It names the exact rule that
fired so the decision is auditable.

The six veto flags include the three taxonomy hard-rule conditions (minor,
self-harm, active fraud) plus the flags that signal a sensitive situation the
classifier may have mislabeled into a benign category (unauthorized access, RG,
legal threat). Checking the flags directly is what stops a hard-rule condition
from slipping through under a benign category label.
"""

from __future__ import annotations

from dataclasses import dataclass

from triage.schemas import SENSITIVE_CATEGORIES, Category, RiskFlags

# Flags that block drafting on their own, in priority order for the reason string.
HARD_RULE_FLAGS: tuple[str, ...] = (
    "self_harm_or_distress",
    "mentions_minor",
    "active_fraud",
    "unauthorized_access_reported",
    "rg_signal",
    "legal_threat",
)

_CATEGORY_REASON = {
    Category.account_compromise: "suspected account compromise — routed to security review",
    Category.problem_gambling: "problem-gambling signals — routed to trained support",
    Category.legal_regulatory: "legal/regulatory matter — routed to compliance",
}

_FLAG_REASON = {
    "self_harm_or_distress": "self-harm or severe distress — routed to human support",
    "mentions_minor": "minor referenced — routed to human review",
    "active_fraud": "active fraud in progress — routed to security review",
    "unauthorized_access_reported": "unauthorized access reported — routed to security review",
    "rg_signal": "problem-gambling signals — routed to trained support",
    "legal_threat": "legal threat referenced — routed to compliance",
}


@dataclass(frozen=True)
class GateDecision:
    """The gate's verdict. should_draft is False whenever a hard rule fired."""

    should_draft: bool
    no_draft_reason: str | None = None
    rule: str | None = None
    flag_source: str | None = None


def evaluate_gate(
    category: Category, model_flags: RiskFlags, prescreen_flags: RiskFlags
) -> GateDecision:
    """Decide whether a hard rule blocks drafting this ticket."""
    if category in SENSITIVE_CATEGORIES:
        return GateDecision(
            should_draft=False,
            no_draft_reason=_CATEGORY_REASON[category],
            rule=f"sensitive_category:{category.value}",
            flag_source="category",
        )

    for name in HARD_RULE_FLAGS:
        in_model = getattr(model_flags, name)
        in_prescreen = getattr(prescreen_flags, name)
        if in_model or in_prescreen:
            source = "both" if in_model and in_prescreen else ("model" if in_model else "prescreen")
            return GateDecision(
                should_draft=False,
                no_draft_reason=_FLAG_REASON[name],
                rule=f"hard_flag:{name}",
                flag_source=source,
            )

    return GateDecision(should_draft=True)
