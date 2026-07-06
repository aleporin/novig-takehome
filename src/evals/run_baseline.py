"""Run the baseline over the training set, then print and save a metrics report."""

from __future__ import annotations

from triage.config import Config
from triage.logging_setup import report_logger

from .baseline import MajorityBaseline
from .dataset import load_tickets
from .metrics import Scored, evaluate
from .report import render_summary, write_report
from .validate import validate_predictions


def main() -> None:
    config = Config()
    tickets = load_tickets(config.paths.tickets_train)
    baseline = MajorityBaseline.from_labels(tickets)
    predictions = [baseline.predict(t) for t in tickets]

    problems = validate_predictions([p.model_dump(mode="json") for p in predictions], tickets)
    if problems:
        raise SystemExit("baseline predictions failed validation:\n" + "\n".join(problems))

    scored = [Scored(t.ticket_id, p, t.label) for t, p in zip(tickets, predictions) if t.label]
    report = evaluate(scored)
    log = report_logger()
    log.info(render_summary(report))

    predictor = (
        f"MajorityBaseline(category={baseline.category.value}, urgency={baseline.urgency.value})"
    )
    run_dir = write_report(report, config, run_name="baseline_train", predictor=predictor)
    log.info("\nwrote %s", run_dir)


if __name__ == "__main__":
    main()
