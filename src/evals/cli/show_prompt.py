"""`make show-prompt TICKET=<id>`: print the assembled classify prompt, no API call."""

from __future__ import annotations

import sys

from evals.dataset import find_ticket
from triage.config import Config
from triage.context.assembler import PromptAssembler
from triage.context.exemplars import load_exemplar_pool, select_exemplars
from triage.logging_setup import report_logger


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1]:
        raise SystemExit("usage: make show-prompt TICKET=<id>")
    config = Config()
    ticket = find_ticket(config.paths.tickets_train, config.paths.tickets_eval, sys.argv[1])
    assembler = PromptAssembler(config)
    pool = load_exemplar_pool(config.paths.tickets_train)
    prompt = assembler.classification_prompt(ticket, select_exemplars(pool, ticket.ticket_id))

    log = report_logger()
    log.info("=== SYSTEM ===")
    log.info(prompt.system)
    log.info("\n=== USER ===")
    log.info(prompt.user)
    log.info("\nexemplars used: %s", prompt.exemplar_ids)
    log.info("truncated: %s", prompt.truncated)


if __name__ == "__main__":
    main()
