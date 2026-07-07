"""Diagnostic: the T1 confidence distribution on the validation pool, misses marked.

Used to decide the escalation confidence threshold before it is frozen. Raw T1
only, no cascade. Cached from the classifier run, so it costs nothing to rerun.
"""

from __future__ import annotations

from evals.dataset import load_tickets
from triage.config import Config
from triage.context.exemplars import is_exemplar, select_exemplars
from triage.factory import build_client, build_context
from triage.logging_setup import report_logger
from triage.stages.classify import classify

_THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95)


def main() -> None:
    config = Config()
    log = report_logger()
    client = build_client(config)
    assembler, pool = build_context(config)
    tickets = [
        t
        for t in load_tickets(config.paths.tickets_train)
        if t.label is not None and not is_exemplar(t.ticket_id)
    ]

    rows = []
    for ticket in tickets:
        result = classify(
            ticket,
            select_exemplars(pool, ticket.ticket_id),
            client=client,
            assembler=assembler,
            config=config,
            model=config.model_t1,
            tier="T1",
        )
        cls = result.classification
        cat_ok = cls.category == ticket.label.category
        urg_ok = cls.urgency == ticket.label.urgency
        rows.append((cls.confidence, cat_ok, urg_ok, ticket.ticket_id))

    rows.sort()
    log.info("T1 confidence on validation pool (sorted). 'X' = miss.")
    log.info("%5s  %4s %4s  ticket", "conf", "cat", "urg")
    for conf, cat_ok, urg_ok, tid in rows:
        log.info("%5.2f  %4s %4s  %s", conf, "ok" if cat_ok else "X", "ok" if urg_ok else "X", tid)

    total_miss = sum(1 for _, c, u, _ in rows if not (c and u))
    log.info("\nthreshold sweep (confidence trigger alone):")
    for thr in _THRESHOLDS:
        escalated = [r for r in rows if r[0] < thr]
        misses_caught = sum(1 for _, c, u, _ in escalated if not (c and u))
        log.info(
            "  thr=%.2f: escalate %d/%d; of the %d misses, %d have conf < thr",
            thr,
            len(escalated),
            len(rows),
            total_miss,
            misses_caught,
        )


if __name__ == "__main__":
    main()
