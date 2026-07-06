"""Run the pipeline over tickets and turn the results into metrics inputs.

Also computes the layer-attribution diagnostic: for each gold no-draft ticket,
which layer would have caught it -- the lexical pre-screen, the classifier, or
both. That table is the evidence for the two-independent-detectors claim.
"""

from __future__ import annotations

import json
from pathlib import Path

from triage.config import Config
from triage.context.assembler import PromptAssembler
from triage.llm.types import LLMClient
from triage.pipeline import Enriched, predict
from triage.routing.cost_tracker import CallRecord, CostReport
from triage.schemas import SENSITIVE_CATEGORIES, Ticket
from triage.stages.gate import HARD_RULE_FLAGS

from .metrics import Scored


def run_pipeline(
    tickets: list[Ticket],
    *,
    client: LLMClient,
    assembler: PromptAssembler,
    exemplar_pool: list[Ticket],
    config: Config,
) -> list[Enriched]:
    """Run every ticket through predict() and collect the enriched results."""
    return [
        predict(t, client=client, assembler=assembler, exemplar_pool=exemplar_pool, config=config)
        for t in tickets
    ]


def scored_from(
    enriched: list[Enriched], tickets: list[Ticket], *, only_ids: set[str] | None = None
) -> list[Scored]:
    """Pair predictions with gold labels for the labeled tickets we want to score."""
    by_id = {t.ticket_id: t for t in tickets}
    scored = []
    for item in enriched:
        ticket = by_id[item.ticket_id]
        if ticket.label is None:
            continue
        if only_ids is not None and item.ticket_id not in only_ids:
            continue
        scored.append(Scored(item.ticket_id, item.prediction, ticket.label))
    return scored


# Flags that cause a no-draft, hard or soft, so attribution reflects every reason a
# ticket is declined -- not just the hard rules.
_SOFT_FLAGS = ("disputes_novig_fact", "jurisdictional_eligibility", "asks_binding_policy_or_spec")
_NO_DRAFT_FLAGS = HARD_RULE_FLAGS + _SOFT_FLAGS


def _classifier_caught(item: Enriched) -> bool:
    if item.classification is None:
        return False
    if item.classification.category in SENSITIVE_CATEGORIES:
        return True
    return any(getattr(item.classification.flags, f) for f in _NO_DRAFT_FLAGS)


def layer_attribution(enriched: list[Enriched], tickets: list[Ticket]) -> list[dict]:
    """For each gold no-draft ticket, record which layer(s) flagged it."""
    by_id = {t.ticket_id: t for t in tickets}
    rows = []
    for item in enriched:
        label = by_id[item.ticket_id].label
        if label is None or label.should_draft:
            continue
        lexicon = any(getattr(item.prescreen_flags, f) for f in _NO_DRAFT_FLAGS)
        classifier = _classifier_caught(item)
        layer = (
            "both"
            if lexicon and classifier
            else "lexicon_only"
            if lexicon
            else "classifier_only"
            if classifier
            else "neither"
        )
        rows.append({"ticket_id": item.ticket_id, "category": label.category.value, "layer": layer})
    return rows


def unattributed_no_draft(rows: list[dict]) -> list[str]:
    """Gold no-draft tickets that no layer flagged. Must always be empty."""
    return [row["ticket_id"] for row in rows if row["layer"] == "neither"]


def collect_calls(enriched: list[Enriched]) -> list[CallRecord]:
    """Flatten every model call made across the run."""
    return [call for item in enriched for call in item.calls]


def escalation_summary(enriched: list[Enriched], total: int) -> dict:
    """Escalation count, rate, and a breakdown by trigger reason."""
    reasons: dict[str, int] = {}
    escalated = 0
    for item in enriched:
        if item.escalated:
            escalated += 1
            reasons[item.escalation_reason] = reasons.get(item.escalation_reason, 0) + 1
    return {
        "escalated": escalated,
        "total": total,
        "rate": escalated / total if total else 0.0,
        "reasons": reasons,
    }


def write_cost(run_dir: Path, cost: CostReport, escalation: dict) -> None:
    """Write the escalation and cost summary to cost.json."""
    payload = {
        "escalation": escalation,
        "cost": {
            "n_calls": cost.n_calls,
            "total_cost_usd": cost.total_cost_usd,
            "cost_by_tier": cost.cost_by_tier,
            "p50_latency_ms": cost.p50_latency_ms,
            "p95_latency_ms": cost.p95_latency_ms,
            "cache_hit_rate": cost.cache_hit_rate,
        },
    }
    (run_dir / "cost.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_enriched(run_dir: Path, enriched: list[Enriched]) -> None:
    """Write internal signals to enriched.jsonl (never into predictions.jsonl)."""
    lines = []
    for item in enriched:
        pred = item.prediction
        lines.append(
            json.dumps(
                {
                    "ticket_id": item.ticket_id,
                    "category": pred.category.value,
                    "urgency": pred.urgency.value,
                    "should_draft": pred.should_draft,
                    "no_draft_reason": pred.no_draft_reason,
                    "confidence": pred.confidence,
                    "prescreen_flags": _true_flags(item.prescreen_flags),
                    "classifier_flags": (
                        _true_flags(item.classification.flags) if item.classification else []
                    ),
                    "gate_rule": item.gate.rule if item.gate else None,
                    "exemplar_ids": item.exemplar_ids,
                }
            )
        )
    (run_dir / "enriched.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_diagnostic(run_dir: Path, rows: list[dict]) -> None:
    (run_dir / "layer_attribution.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _true_flags(flags) -> list[str]:
    return [name for name, value in flags.model_dump().items() if value]
