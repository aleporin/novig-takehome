"""Run the cross-provider judge over the drafted train tickets.

Degrades gracefully: with no OPENAI_API_KEY it prints 'skipped' and exits. Runs
canary calibration first; if the judge misses any seeded violation it is demoted
and no scores are quoted. Otherwise it scores each drafted ticket against the rubric.
"""

from __future__ import annotations

import logging

from evals.batch import run_pipeline
from evals.dataset import load_tickets
from evals.judge import _CRITERIA, build_judge, load_rubric, run_canaries, score_draft
from triage.config import Config
from triage.factory import build_client, build_context
from triage.logging_setup import report_logger, setup_logging
from triage.routing.cost_tracker import build_cost_report


def main() -> None:
    config = Config()
    setup_logging(console_level=logging.WARNING)
    log = report_logger()

    judge = build_judge(config)
    if judge is None:
        log.info("judge: skipped (no OPENAI_API_KEY)")
        return

    rubric = load_rubric(config)
    canary = run_canaries(judge, rubric)
    log.info("judge canaries: caught %d/%d", canary.caught, canary.total)
    if not canary.trustworthy:
        log.info("judge DEMOTED: missed %s — scores not quoted", canary.misses)
        return

    client = build_client(config)
    assembler, pool = build_context(config)
    tickets = load_tickets(config.paths.tickets_train)
    by_id = {t.ticket_id: t for t in tickets}
    enriched = run_pipeline(
        tickets, client=client, assembler=assembler, exemplar_pool=pool, config=config
    )

    scored = passed = 0
    per_criterion = dict.fromkeys(_CRITERIA, 0)
    for item in enriched:
        pred = item.prediction
        if not pred.should_draft:
            continue
        gold = by_id[item.ticket_id].label.gold_response_notes
        verdict = score_draft(judge, rubric, by_id[item.ticket_id], pred.draft_response, gold)
        scored += 1
        passed += verdict.passed
        for name in _CRITERIA:
            per_criterion[name] += getattr(verdict, name)

    log.info("judge: %d/%d drafts pass all criteria", passed, scored)
    for name in _CRITERIA:
        log.info("  %s: %d/%d", name, per_criterion[name], scored)

    judge_cost = build_cost_report(getattr(judge, "calls", []))
    log.info(
        "judge cost: $%.4f notional, %d calls, cache hits %.0f%%",
        judge_cost.total_cost_usd,
        judge_cost.n_calls,
        100 * judge_cost.cache_hit_rate,
    )


if __name__ == "__main__":
    main()
