"""Soft-rule policy: the judgment-call declines for tickets that pass the gate.

These are not hard rules, but they usually warrant no-draft: disputing a
Novig-provided fact, a jurisdiction question whose answer changes over time, or a
request for a binding policy/spec answer. Binding-spec handling is configurable;
declining is the conservative default. Runs only after the hard gate allows a draft.

Like the gate, it reads two flag sources — the classifier and the pre-screen —
and a flag from either one declines. That gives the fact-dispute rule a lexical
backstop so a classifier miss on a real dispute does not silently draft.
"""

from __future__ import annotations

from dataclasses import dataclass

from triage.schemas import RiskFlags

# Soft flags checked in priority order, each mapped to its decline reason.
_SOFT_REASON = {
    "disputes_novig_fact": "disputes a Novig-provided fact — needs ops review first",
    "jurisdictional_eligibility": "jurisdictional eligibility question — answer changes over time",
}
_BINDING_SPEC_REASON = "binding policy/spec question — deferred to a human"


@dataclass(frozen=True)
class PolicyDecision:
    """The soft-rule verdict for a ticket that already passed the gate."""

    should_draft: bool
    no_draft_reason: str | None = None
    rule: str | None = None


def apply_soft_policy(
    model_flags: RiskFlags,
    prescreen_flags: RiskFlags,
    *,
    binding_spec_mode: str = "no_draft",
) -> PolicyDecision:
    """Decide whether a soft rule declines a ticket the gate allowed.

    A soft flag set by either the classifier or the pre-screen declines.
    """

    def fired(name: str) -> bool:
        return getattr(model_flags, name) or getattr(prescreen_flags, name)

    for name, reason in _SOFT_REASON.items():
        if fired(name):
            return PolicyDecision(
                should_draft=False, no_draft_reason=reason, rule=f"soft_flag:{name}"
            )

    if fired("asks_binding_policy_or_spec") and binding_spec_mode == "no_draft":
        return PolicyDecision(
            should_draft=False,
            no_draft_reason=_BINDING_SPEC_REASON,
            rule="soft_flag:asks_binding_policy_or_spec",
        )

    return PolicyDecision(should_draft=True)
