"""Run cross-provider judges over the drafted train tickets.

Degrades gracefully: with no keys it prints 'skipped'. Each judge runs canary
calibration first; any judge that misses a seeded violation is demoted and its
scores are not quoted. When two judges are trustworthy, per-ticket and
per-criterion agreement is reported — that agreement rate is the eval signal,
not either judge's raw pass count.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from evals.batch import run_pipeline
from evals.dataset import load_tickets
from evals.judge import (
    _CRITERIA,
    JudgeVerdict,
    build_judges,
    load_rubric,
    run_canaries,
    score_draft,
)
from triage.config import Config
from triage.factory import build_client, build_context
from triage.logging_setup import report_logger, setup_logging
from triage.routing.cost_tracker import build_cost_report


def _score_all(judge, rubric, enriched, by_id) -> dict[str, JudgeVerdict]:
    """Score every drafted ticket with one judge. Keyed by ticket_id."""
    verdicts: dict[str, JudgeVerdict] = {}
    for item in enriched:
        pred = item.prediction
        if not pred.should_draft:
            continue
        gold = by_id[item.ticket_id].label.gold_response_notes
        verdicts[item.ticket_id] = score_draft(
            judge, rubric, by_id[item.ticket_id], pred.draft_response, gold
        )
    return verdicts


def _summarize_one(label: str, verdicts: dict[str, JudgeVerdict]) -> dict:
    scored = len(verdicts)
    passed = sum(1 for v in verdicts.values() if v.passed)
    per_criterion = {
        name: sum(int(getattr(v, name)) for v in verdicts.values()) for name in _CRITERIA
    }
    return {
        "label": label,
        "scored": scored,
        "passed": passed,
        "per_criterion": per_criterion,
    }


def _agreement(a: dict[str, JudgeVerdict], b: dict[str, JudgeVerdict]) -> dict:
    """Compare two judges over the tickets they both scored."""
    common = sorted(set(a) & set(b))
    overall_matches, criterion_matches, disagreements = 0, dict.fromkeys(_CRITERIA, 0), []
    for tid in common:
        va, vb = a[tid], b[tid]
        if va.passed == vb.passed:
            overall_matches += 1
        else:
            disagreements.append({"ticket_id": tid, "a_passed": va.passed, "b_passed": vb.passed})
        for name in _CRITERIA:
            if getattr(va, name) == getattr(vb, name):
                criterion_matches[name] += 1
    n = len(common)
    return {
        "n_common": n,
        "overall_agreement": overall_matches / n if n else 0.0,
        "overall_matches": overall_matches,
        "per_criterion_agreement": {
            name: (m / n if n else 0.0) for name, m in criterion_matches.items()
        },
        "disagreements": disagreements,
    }


def main() -> None:
    config = Config()
    setup_logging(console_level=logging.WARNING)
    log = report_logger()

    judges = build_judges(config)
    if not judges:
        log.info("judges: skipped (no OPENAI_API_KEY or GOOGLE_API_KEY)")
        return

    rubric = load_rubric(config)

    # Calibrate each judge on the seeded canaries; drop any that misses one.
    trusted: list[tuple[str, object]] = []
    for label, judge in judges:
        canary = run_canaries(judge, rubric)
        log.info("judge[%s] canaries: caught %d/%d", label, canary.caught, canary.total)
        if canary.trustworthy:
            trusted.append((label, judge))
        else:
            log.info("judge[%s] DEMOTED: missed %s", label, canary.misses)
    if not trusted:
        log.info("all judges demoted — no scores quoted")
        return

    client = build_client(config)
    assembler, pool = build_context(config)
    tickets = load_tickets(config.paths.tickets_train)
    by_id = {t.ticket_id: t for t in tickets}
    enriched = run_pipeline(
        tickets, client=client, assembler=assembler, exemplar_pool=pool, config=config
    )

    per_judge_verdicts: dict[str, dict[str, JudgeVerdict]] = {}
    per_judge_summary: list[dict] = []
    per_judge_cost: list[dict] = []
    for label, judge in trusted:
        verdicts = _score_all(judge, rubric, enriched, by_id)
        per_judge_verdicts[label] = verdicts
        summary = _summarize_one(label, verdicts)
        per_judge_summary.append(summary)
        log.info(
            "judge[%s]: %d/%d drafts pass all criteria", label, summary["passed"], summary["scored"]
        )
        for name in _CRITERIA:
            log.info("  %s: %d/%d", name, summary["per_criterion"][name], summary["scored"])
        cost = build_cost_report(getattr(judge, "calls", []))
        per_judge_cost.append(
            {
                "label": label,
                "n_calls": cost.n_calls,
                "total_cost_usd": cost.total_cost_usd,
                "cache_hit_rate": cost.cache_hit_rate,
            }
        )
        log.info(
            "judge[%s] cost: $%.4f notional, %d calls, cache hits %.0f%%",
            label,
            cost.total_cost_usd,
            cost.n_calls,
            100 * cost.cache_hit_rate,
        )

    agreement = None
    if len(trusted) >= 2:
        (label_a, _), (label_b, _) = trusted[0], trusted[1]
        agreement = _agreement(per_judge_verdicts[label_a], per_judge_verdicts[label_b])
        agreement["judges"] = [label_a, label_b]
        log.info(
            "agreement[%s vs %s]: %d/%d overall (%.0f%%)",
            label_a,
            label_b,
            agreement["overall_matches"],
            agreement["n_common"],
            100 * agreement["overall_agreement"],
        )
        for name, rate in agreement["per_criterion_agreement"].items():
            log.info("  %s: %.0f%%", name, 100 * rate)
        if agreement["disagreements"]:
            log.info("disagreements: %s", [d["ticket_id"] for d in agreement["disagreements"]])

    # Persist to a timestamped results folder so the numbers have a record on disk.
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.results / f"{stamp}_dual_judge"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "judges": [{"label": lbl, "model": j._model} for lbl, j in trusted],
        "per_judge": per_judge_summary,
        "per_judge_cost": per_judge_cost,
        "agreement": agreement,
        "per_ticket_verdicts": {
            label: {tid: v.model_dump() for tid, v in verdicts.items()}
            for label, verdicts in per_judge_verdicts.items()
        },
    }
    (run_dir / "judges.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    log.info("wrote %s", run_dir / "judges.json")


if __name__ == "__main__":
    main()
