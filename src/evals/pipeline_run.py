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


def _classifier_caught(item: Enriched) -> bool:
    if item.classification is None:
        return False
    if item.classification.category in SENSITIVE_CATEGORIES:
        return True
    return any(getattr(item.classification.flags, f) for f in HARD_RULE_FLAGS)


def layer_attribution(enriched: list[Enriched], tickets: list[Ticket]) -> list[dict]:
    """For each gold no-draft ticket, record which layer(s) flagged it."""
    by_id = {t.ticket_id: t for t in tickets}
    rows = []
    for item in enriched:
        label = by_id[item.ticket_id].label
        if label is None or label.should_draft:
            continue
        lexicon = any(getattr(item.prescreen_flags, f) for f in HARD_RULE_FLAGS)
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
