"""Tests for per-call cost accounting and the run cost report."""

from __future__ import annotations

import pytest

from triage.config import HAIKU, SONNET
from triage.routing.cost_tracker import CallRecord, build_cost_report, call_cost


def test_call_cost_uses_the_price_table() -> None:
    # Haiku: $1/M in, $5/M out -> 1000 in + 500 out.
    cost = call_cost(HAIKU, 1000, 500)
    assert cost == pytest.approx(1000 / 1e6 * 1.0 + 500 / 1e6 * 5.0)


def _record(tier, model, latency, cache_hit=False, cost=0.001) -> CallRecord:
    return CallRecord(model, tier, 100, 50, latency, cache_hit, cost)


def test_cost_report_totals_and_percentiles() -> None:
    records = [
        _record("T1", HAIKU, 100.0, cost=0.001),
        _record("T1", HAIKU, 200.0, cost=0.001),
        _record("T2", SONNET, 900.0, cost=0.010),
    ]
    report = build_cost_report(records)
    assert report.n_calls == 3
    assert report.total_cost_usd == pytest.approx(0.012)
    assert report.cost_by_tier == {"T1": pytest.approx(0.002), "T2": pytest.approx(0.010)}
    assert report.p95_latency_ms == 900.0
    assert report.cache_hit_rate == 0.0


def test_cache_hit_rate() -> None:
    records = [_record("T1", HAIKU, 10.0, cache_hit=True), _record("T1", HAIKU, 10.0)]
    assert build_cost_report(records).cache_hit_rate == 0.5


def test_empty_report() -> None:
    report = build_cost_report([])
    assert report.n_calls == 0
    assert report.total_cost_usd == 0.0
    assert report.cache_hit_rate == 0.0
