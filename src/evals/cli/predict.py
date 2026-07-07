"""`make predict TICKET=<id>`: run one ticket end to end and print the trace."""

from __future__ import annotations

import sys

from evals.dataset import find_ticket
from triage.config import Config
from triage.factory import build_client, build_context
from triage.logging_setup import report_logger
from triage.pipeline import Enriched, predict
from triage.schemas import RiskFlags, Ticket


def _true(flags: RiskFlags) -> list[str]:
    return [name for name, value in flags.model_dump().items() if value]


def _log_trace(ticket: Ticket, item: Enriched) -> None:
    log = report_logger()
    log.info("ticket %s: %s", ticket.ticket_id, ticket.subject)
    log.info("  prescreen flags: %s", _true(item.prescreen_flags) or "(none)")
    if item.classification is not None:
        cls = item.classification
        log.info(
            "  classified:      %s / %s (conf %s)",
            cls.category.value,
            cls.urgency.value,
            cls.confidence,
        )
        log.info("  classifier flags:%s", _true(cls.flags) or " (none)")
    if item.gate is not None:
        log.info(
            "  gate:            should_draft=%s rule=%s", item.gate.should_draft, item.gate.rule
        )
    pred = item.prediction
    log.info("  DECISION:        should_draft=%s confidence=%s", pred.should_draft, pred.confidence)
    if pred.no_draft_reason:
        log.info("  no_draft_reason: %s", pred.no_draft_reason)


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1]:
        raise SystemExit("usage: make predict TICKET=<id>")
    config = Config()
    ticket = find_ticket(config.paths.tickets_train, config.paths.tickets_eval, sys.argv[1])
    client = build_client(config)
    assembler, pool = build_context(config)
    item = predict(ticket, client=client, assembler=assembler, exemplar_pool=pool, config=config)
    _log_trace(ticket, item)


if __name__ == "__main__":
    main()
