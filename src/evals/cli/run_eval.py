"""Run the classifier pipeline over the eval set, write predictions.jsonl, validate.

The eval set is unlabeled, so this also reports what we can without labels: the
predicted category/urgency distributions vs train, the draft and escalation rates,
and the confidence spread (an unlabeled drift signal).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from evals.batch import eval_signal, run_pipeline
from evals.dataset import load_tickets
from evals.validate import validate_file
from triage.config import Config
from triage.factory import build_client, build_context
from triage.logging_setup import report_logger, setup_logging


def main() -> None:
    config = Config()
    setup_logging(console_level=logging.WARNING)
    log = report_logger()
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
    log.info("wrote %s (%d predictions), validation OK", out, len(predictions))

    signal = eval_signal(load_tickets(config.paths.tickets_train), enriched)
    log.info("\n=== unlabeled-eval signal ===")
    log.info(
        "should-draft rate: train %.2f vs eval %.2f",
        signal["should_draft_rate"]["train"],
        signal["should_draft_rate"]["eval"],
    )
    log.info("eval escalation rate: %.2f", signal["eval_escalation_rate"])
    log.info(
        "eval confidence: min %.2f mean %.2f max %.2f",
        signal["eval_confidence"]["min"],
        signal["eval_confidence"]["mean"],
        signal["eval_confidence"]["max"],
    )
    log.info("category  train=%s", signal["category"]["train"])
    log.info("          eval =%s", signal["category"]["eval"])
    log.info("urgency   train=%s", signal["urgency"]["train"])
    log.info("          eval =%s", signal["urgency"]["eval"])

    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.results / f"{stamp}_eval_signal"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "signal.json").write_text(json.dumps(signal, indent=2), encoding="utf-8")
    log.info("\nwrote %s", run_dir)


if __name__ == "__main__":
    main()
