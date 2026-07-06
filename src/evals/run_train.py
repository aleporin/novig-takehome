"""Run the real classifier pipeline over the training set and report metrics.

Headline metrics are on the validation pool (the 22 tickets never used as
exemplars). Also prints the layer-attribution diagnostic and writes the run to
results/. Makes real API calls; every call is disk-cached, so reruns are free.
"""

from __future__ import annotations

from triage.config import Config
from triage.context.exemplars import is_exemplar
from triage.factory import build_client, build_context
from triage.logging_setup import report_logger
from triage.routing.cost_tracker import build_cost_report

from .dataset import load_tickets
from .metrics import evaluate
from .pipeline_run import (
    collect_calls,
    escalation_summary,
    layer_attribution,
    run_pipeline,
    scored_from,
    unattributed_no_draft,
    write_cost,
    write_diagnostic,
    write_enriched,
)
from .report import render_summary, write_report


def main() -> None:
    config = Config()
    log = report_logger()
    client = build_client(config)
    assembler, pool = build_context(config)
    tickets = load_tickets(config.paths.tickets_train)

    enriched = run_pipeline(
        tickets, client=client, assembler=assembler, exemplar_pool=pool, config=config
    )

    validation_ids = {t.ticket_id for t in tickets if not is_exemplar(t.ticket_id)}
    scored = scored_from(enriched, tickets, only_ids=validation_ids)
    report = evaluate(scored)

    log.info(
        "Validation pool (n=%d, %d exemplars excluded):", len(scored), len(tickets) - len(scored)
    )
    log.info(render_summary(report))

    rows = layer_attribution(enriched, tickets)
    log.info("\nlayer attribution (gold no-draft tickets):")
    for row in rows:
        log.info("  %s [%s]: %s", row["ticket_id"], row["category"], row["layer"])

    escalation = escalation_summary(enriched, len(tickets))
    cost = build_cost_report(collect_calls(enriched))
    log.info(
        "\nescalation: %d/%d = %.0f%%  reasons=%s",
        escalation["escalated"],
        escalation["total"],
        100 * escalation["rate"],
        escalation["reasons"],
    )
    log.info(
        "cost: $%.4f total, %d calls; by tier %s; latency p50 %.0fms p95 %.0fms; cache hits %.0f%%",
        cost.total_cost_usd,
        cost.n_calls,
        {k: round(v, 4) for k, v in cost.cost_by_tier.items()},
        cost.p50_latency_ms,
        cost.p95_latency_ms,
        100 * cost.cache_hit_rate,
    )

    run_dir = write_report(
        report, config, run_name="cascade_train", predictor="T1->T2 cascade (Haiku/Sonnet)"
    )
    write_enriched(run_dir, enriched)
    write_diagnostic(run_dir, rows)
    write_cost(run_dir, cost, escalation)
    log.info("\nwrote %s", run_dir)

    missing = unattributed_no_draft(rows)
    if missing:
        raise SystemExit(f"regression: gold no-draft tickets flagged by neither layer: {missing}")


if __name__ == "__main__":
    main()
