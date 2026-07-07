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


def _majority(labels: list[str], per_judge: dict[str, dict[str, JudgeVerdict]]) -> dict:
    """Ticket-level majority vote across every judge that scored the ticket.

    A ticket 'passes' the panel if strictly more than half of the judges that
    scored it returned passed=True. Reports per-ticket votes plus the panel's
    pass rate, unanimity rate, and the tickets where any judge dissented.
    """
    common = set.intersection(*(set(per_judge[label]) for label in labels))
    per_ticket: list[dict] = []
    panel_pass = unanimous = split_count = 0
    for tid in sorted(common):
        votes = {label: per_judge[label][tid].passed for label in labels}
        yes = sum(1 for v in votes.values() if v)
        no = len(votes) - yes
        panel_verdict = yes > no
        if panel_verdict:
            panel_pass += 1
        if yes == 0 or no == 0:
            unanimous += 1
        else:
            split_count += 1
        per_ticket.append(
            {"ticket_id": tid, "votes": votes, "yes": yes, "no": no, "panel_passed": panel_verdict}
        )
    n = len(common)
    return {
        "judges": labels,
        "n_common": n,
        "panel_pass_rate": panel_pass / n if n else 0.0,
        "panel_passed": panel_pass,
        "unanimous": unanimous,
        "split": split_count,
        "per_ticket": per_ticket,
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

    # Calibrate each judge on the seeded canaries; drop any that misses one, and
    # drop any that fails to call at all (missing credits, transient outage) so a
    # single broken lab does not sink the panel.
    trusted: list[tuple[str, object]] = []
    for label, judge in judges:
        try:
            canary = run_canaries(judge, rubric)
        except Exception as exc:
            log.info("judge[%s] UNAVAILABLE: %s", label, exc)
            continue
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
    completed: list[tuple[str, object]] = []
    for label, judge in trusted:
        try:
            verdicts = _score_all(judge, rubric, enriched, by_id)
        except Exception as exc:
            log.info("judge[%s] SCORING FAILED mid-run: %s", label, exc)
            continue
        per_judge_verdicts[label] = verdicts
        completed.append((label, judge))
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

    # Pairwise agreement (every unordered pair) so the reader can see which two
    # labs disagree most, not just an overall panel number.
    pairwise: list[dict] = []
    labels = [label for label, _ in completed]
    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1 :]:
            pair = _agreement(per_judge_verdicts[label_a], per_judge_verdicts[label_b])
            pair["judges"] = [label_a, label_b]
            pairwise.append(pair)
            log.info(
                "agreement[%s vs %s]: %d/%d overall (%.0f%%)",
                label_a,
                label_b,
                pair["overall_matches"],
                pair["n_common"],
                100 * pair["overall_agreement"],
            )
            for name, rate in pair["per_criterion_agreement"].items():
                log.info("  %s: %.0f%%", name, 100 * rate)
            if pair["disagreements"]:
                log.info("  disagreements: %s", [d["ticket_id"] for d in pair["disagreements"]])

    # Majority vote across the panel — the single number a release gate would use.
    majority = None
    if len(completed) >= 3:
        majority = _majority(labels, per_judge_verdicts)
        log.info(
            "panel majority: %d/%d pass (%.0f%%); unanimous %d, split %d",
            majority["panel_passed"],
            majority["n_common"],
            100 * majority["panel_pass_rate"],
            majority["unanimous"],
            majority["split"],
        )
        split_ids = [t["ticket_id"] for t in majority["per_ticket"] if t["yes"] not in (0, 3)]
        if split_ids:
            log.info("panel splits (2-1): %s", split_ids)

    # Persist to a timestamped results folder so the numbers have a record on disk.
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    if len(completed) >= 3:
        tag = "triple_judge"
    elif len(completed) == 2:
        tag = "dual_judge"
    else:
        tag = "single_judge"
    run_dir = config.paths.results / f"{stamp}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "judges": [{"label": lbl, "model": j._model} for lbl, j in completed],
        "per_judge": per_judge_summary,
        "per_judge_cost": per_judge_cost,
        "pairwise_agreement": pairwise,
        "majority": majority,
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
