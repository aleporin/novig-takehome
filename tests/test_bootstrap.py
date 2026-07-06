"""Tests for the bootstrap confidence-interval utility."""

from __future__ import annotations

from evals.bootstrap import bootstrap_ci


def test_point_estimate_is_the_mean() -> None:
    est = bootstrap_ci([1.0, 1.0, 0.0, 0.0], seed=1)
    assert est.point == 0.5


def test_interval_brackets_the_point() -> None:
    est = bootstrap_ci([1.0, 0.0, 1.0, 1.0, 0.0, 1.0], seed=1)
    assert est.ci_low <= est.point <= est.ci_high


def test_same_seed_is_reproducible() -> None:
    values = [1.0, 0.0, 1.0, 0.0, 1.0]
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)


def test_constant_values_have_zero_width_interval() -> None:
    est = bootstrap_ci([1.0, 1.0, 1.0], seed=3)
    assert est.point == est.ci_low == est.ci_high == 1.0


def test_empty_input_is_all_zero() -> None:
    est = bootstrap_ci([], seed=0)
    assert (est.point, est.ci_low, est.ci_high) == (0.0, 0.0, 0.0)
