"""Run the classifier pipeline over the eval set, write predictions.jsonl, validate.

Drafts are still placeholders until the drafting phase, so the draft text is not
final here; the category, urgency, and draft/no-draft decisions are real.
"""

from __future__ import annotations

import logging

from triage.config import Config
from triage.factory import build_client, build_context
from triage.logging_setup import report_logger, setup_logging

from .dataset import load_tickets
from .pipeline_run import run_pipeline
from .validate import validate_file


def main() -> None:
    config = Config()
    setup_logging(console_level=logging.WARNING)
    client = build_client(config)
    assembler, pool = build_context(config)
    eval_tickets = load_tickets(config.paths.tickets_eval)

    enriched = run_pipeline(
        eval_tickets, client=client, assembler=assembler, exemplar_pool=pool, config=config
    )
    predictions = [item.prediction for item in enriched]

    out = config.paths.predictions
    out.write_text("\n".join(p.to_jsonl_line() for p in predictions) + "\n", encoding="utf-8")

    problems = validate_file(out, config.paths.tickets_eval)
    if problems:
        raise SystemExit("predictions.jsonl failed validation:\n" + "\n".join(problems))
    report_logger().info("wrote %s (%d predictions), validation OK", out, len(predictions))


if __name__ == "__main__":
    main()
