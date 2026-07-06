"""Tests for the metrics summary and the results-folder writer."""

from __future__ import annotations

import json

from evals.metrics import Scored, evaluate
from evals.report import render_summary, write_report
from triage.config import Config, Paths
from triage.schemas import Category, Label, Prediction, Urgency


def _scored() -> list[Scored]:
    pred = Prediction(
        ticket_id="t1",
        category=Category.trading_mechanics,
        urgency=Urgency.low,
        should_draft=False,
        no_draft_reason="declined",
        confidence=0.5,
    )
    gold = Label(
        category=Category.trading_mechanics,
        urgency=Urgency.low,
        should_draft=True,
        gold_response_notes="notes",
    )
    return [Scored("t1", pred, gold)]


def test_summary_mentions_the_key_metrics() -> None:
    text = render_summary(evaluate(_scored()))
    assert "category accuracy" in text
    assert "false-draft (hard-rule)" in text
    assert "false-decline" in text


def test_write_report_creates_metrics_and_config(tmp_path) -> None:
    config = Config(paths=Paths(results=tmp_path))
    run_dir = write_report(evaluate(_scored()), config, run_name="test", predictor="baseline")

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "category_accuracy" in metrics
    assert metrics["n"] == 1

    snapshot = json.loads((run_dir / "config.json").read_text())
    assert snapshot["predictor"] == "baseline"
