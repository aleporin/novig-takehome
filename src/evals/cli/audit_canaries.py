"""make audit-canaries: prove the output-guard audit catches off-list assertions."""

from __future__ import annotations

import logging

from triage.config import Config
from triage.factory import build_client
from triage.logging_setup import report_logger, setup_logging
from triage.stages.output_guard import run_audit_canaries


def main() -> None:
    config = Config()
    setup_logging(console_level=logging.WARNING)
    log = report_logger()
    client = build_client(config)
    report = run_audit_canaries(client, config)
    log.info(
        "audit canaries: %d/%d caught, %d/%d passed",
        report.caught,
        report.n_flag,
        report.passed,
        report.n_pass,
    )
    if report.missed:
        log.info("MISSED (off-list assertion not flagged): %s", report.missed)
    if report.false_positives:
        log.info("FALSE POSITIVE (over-flagged a user-quoted fact): %s", report.false_positives)


if __name__ == "__main__":
    main()
