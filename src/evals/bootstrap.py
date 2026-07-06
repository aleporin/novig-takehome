"""Bootstrap confidence intervals.

With ~30 labeled tickets a bare percentage is misleading. We resample the tickets
with replacement many times and report the 2.5th and 97.5th percentiles of the
statistic, giving a 95% interval. Seeded, so the interval is reproducible.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a confidence interval."""

    point: float
    ci_low: float
    ci_high: float


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Value at the given fraction (0..1) of a sorted list, by nearest rank."""
    if not sorted_values:
        return 0.0
    index = round(fraction * (len(sorted_values) - 1))
    return sorted_values[index]


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = mean,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Point estimate of ``statistic`` plus a (1-alpha) bootstrap interval.

    For a rate, pass per-ticket 0/1 values and the default mean statistic. Returns
    an all-zero estimate for an empty input so callers don't have to special-case it.
    """
    if not values:
        return Estimate(0.0, 0.0, 0.0)
    point = statistic(values)
    rng = random.Random(seed)
    n = len(values)
    resampled = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        resampled.append(statistic(sample))
    resampled.sort()
    return Estimate(
        point=point,
        ci_low=_percentile(resampled, alpha / 2),
        ci_high=_percentile(resampled, 1 - alpha / 2),
    )
