"""Threshold sweep: run the T1->T2 cascade at candidate thresholds and compare.

The evidence for freezing the confidence threshold. For each threshold it reports
the escalation rate, the validation accuracy/severity after resolution, and — per
escalated ticket — whether T2 actually corrected a T1 miss. It also lists the T1
misses that no mandatory trigger reaches and that sit at high confidence: the
ceiling of confidence-based escalation. Cached, so it is cheap to rerun.
"""

from __future__ import annotations

from triage.config import Config
from triage.context.exemplars import is_exemplar, select_exemplars
from triage.factory import build_client, build_context
from triage.logging_setup import report_logger
from triage.routing.router import resolve, should_escalate
from triage.stages.classify import classify
from triage.stages.prescreen import prescreen

from .dataset import load_tickets
from .metrics import SEVERITY_COST

_THRESHOLDS = (0.80, 0.90)
_LOG = report_logger()


def _is_miss(cls, label) -> bool:
    return cls.category != label.category or cls.urgency != label.urgency


def main() -> None:
    config = Config()
    client = build_client(config)
    assembler, pool = build_context(config)
    tickets = load_tickets(config.paths.tickets_train)
    val = [t for t in tickets if t.label is not None and not is_exemplar(t.ticket_id)]

    def clf(ticket, model, tier):
        return classify(
            ticket,
            select_exemplars(pool, ticket.ticket_id),
            client=client,
            assembler=assembler,
            config=config,
            model=model,
            tier=tier,
        ).classification

    t1 = {t.ticket_id: clf(t, config.model_t1, "T1") for t in tickets}
    t2 = {t.ticket_id: clf(t, config.model_t2, "T2") for t in val}
    pf = {t.ticket_id: prescreen(t.subject, t.body) for t in tickets}

    for thr in _THRESHOLDS:
        _report_threshold(thr, tickets, val, t1, t2, pf)
    _report_ceiling(val, t1, pf)


def _final(ticket, thr, t1, t2, pf):
    decision = should_escalate(t1[ticket.ticket_id], pf[ticket.ticket_id], threshold=thr)
    if decision.escalate and ticket.ticket_id in t2:
        return resolve(t1[ticket.ticket_id], t2[ticket.ticket_id]), decision.reason
    return t1[ticket.ticket_id], None


def _report_threshold(thr, tickets, val, t1, t2, pf) -> None:
    esc_all = sum(
        should_escalate(t1[t.ticket_id], pf[t.ticket_id], threshold=thr).escalate for t in tickets
    )
    finals = {t.ticket_id: _final(t, thr, t1, t2, pf) for t in val}
    cat = sum(finals[t.ticket_id][0].category == t.label.category for t in val)
    urg = sum(finals[t.ticket_id][0].urgency == t.label.urgency for t in val)
    sev = sum(SEVERITY_COST[t.label.urgency][finals[t.ticket_id][0].urgency] for t in val) / len(
        val
    )

    _LOG.info("\n=== threshold %.2f ===", thr)
    _LOG.info("escalation: %d/%d all = %.0f%%", esc_all, len(tickets), 100 * esc_all / len(tickets))
    _LOG.info(
        "val: category %.0f%%  urgency %.0f%%  severity %.2f",
        100 * cat / len(val),
        100 * urg / len(val),
        sev,
    )
    _LOG.info("escalated val tickets (T1 -> resolved vs gold):")
    for t in val:
        final, reason = finals[t.ticket_id]
        if reason is None:
            continue
        a, g = t1[t.ticket_id], t.label
        status = (
            "FIXED"
            if _is_miss(a, g) and not _is_miss(final, g)
            else ("still-miss" if _is_miss(final, g) else "ok")
        )
        _LOG.info(
            "  %s [%s] T1 %s/%s -> %s/%s  gold %s/%s  %s",
            t.ticket_id,
            reason,
            a.category.value,
            a.urgency.value,
            final.category.value,
            final.urgency.value,
            g.category.value,
            g.urgency.value,
            status,
        )


def _report_ceiling(val, t1, pf) -> None:
    _LOG.info("\n=== T1 misses no mandatory trigger reaches (confidence ceiling) ===")
    for t in val:
        a, g = t1[t.ticket_id], t.label
        if not _is_miss(a, g):
            continue
        mandatory = should_escalate(a, pf[t.ticket_id], threshold=0.0).escalate
        if not mandatory:
            _LOG.info(
                "  %s conf=%.2f  %s/%s vs gold %s/%s  (only a conf threshold > %.2f reaches it)",
                t.ticket_id,
                a.confidence,
                a.category.value,
                a.urgency.value,
                g.category.value,
                g.urgency.value,
                a.confidence,
            )


if __name__ == "__main__":
    main()
