"""Turn a MetricsReport into a printed summary and a saved run folder.

Each run writes a timestamped folder under results/ holding the metrics and a
config snapshot, so every reported number has a record on disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from triage.config import Config

from .bootstrap import Estimate
from .metrics import MetricsReport, RateStat


def _pct(estimate: Estimate) -> str:
    """Format a 0..1 estimate as 'point [low, high]' in percentages."""
    return f"{estimate.point:.0%} [{estimate.ci_low:.0%}, {estimate.ci_high:.0%}]"


def _cost(estimate: Estimate) -> str:
    """Format a raw-number estimate (not a percentage) as 'point [low, high]'."""
    return f"{estimate.point:.2f} [{estimate.ci_low:.2f}, {estimate.ci_high:.2f}]"


def _rate_line(label: str, rate: RateStat) -> str:
    return f"{label}: {rate.numerator}/{rate.denominator}  ({_pct(rate.estimate)})"


# Short labels so the confusion matrix lines up in a terminal.
_URGENCY_ABBREV = {
    "low": "low",
    "medium": "med",
    "high": "high",
    "escalate_immediately": "esc",
}


def render_summary(report: MetricsReport) -> str:
    """A human-readable metrics summary for stdout."""
    lines = [
        f"n = {report.n} labeled tickets",
        f"category accuracy:        {_pct(report.category_accuracy)}",
        f"urgency accuracy:         {_pct(report.urgency_accuracy)}",
        f"severity-weighted error:  {_cost(report.severity_weighted_error)}",
        _rate_line("false-draft (hard-rule)", report.false_draft_hard),
        _rate_line("false-draft (soft-rule)", report.false_draft_soft),
        _rate_line("false-decline", report.false_decline),
        "urgency confusion (rows=gold, cols=pred, esc=escalate_immediately):",
    ]
    cols = [_URGENCY_ABBREV[p] for p in report.urgency_confusion]
    lines.append("      " + "".join(f"{c:>6}" for c in cols))
    for gold, row in report.urgency_confusion.items():
        lines.append(
            f"{_URGENCY_ABBREV[gold]:>6}" + "".join(f"{count:>6}" for count in row.values())
        )
    return "\n".join(lines)


def _config_snapshot(config: Config, predictor: str) -> dict:
    """The settings worth recording with a run. Skill hashes land in a later phase."""
    return {
        "predictor": predictor,
        "model_t1": config.model_t1,
        "model_t2": config.model_t2,
        "model_judge": config.model_judge,
        "t1_confidence_threshold": config.t1_confidence_threshold,
        "escalation_rate_floor": config.escalation_rate_floor,
        "escalation_rate_ceiling": config.escalation_rate_ceiling,
        "temperature": config.temperature,
        "skills": "none yet",
    }


def write_report(report: MetricsReport, config: Config, *, run_name: str, predictor: str) -> Path:
    """Write metrics.json and config.json to a fresh results/<timestamp>_<name> folder."""
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.results / f"{stamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(_config_snapshot(config, predictor), indent=2), encoding="utf-8"
    )
    return run_dir
