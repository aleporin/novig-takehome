"""Per-call cost and latency accounting.

Cost is notional token cost (tokens x price), reported regardless of caching, so
"cost per ticket" is stable; caching just makes reruns free. Consumes the same
token/latency numbers that get logged per call, so the log and the cost report
can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from triage.config import MODEL_PRICING


@dataclass(frozen=True)
class CallRecord:
    """One model call's accounting."""

    model: str
    tier: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cache_hit: bool
    cost_usd: float


def call_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Notional USD cost of a call from its token counts and the price table."""
    price = MODEL_PRICING[model]
    return tokens_in / 1e6 * price.input_per_mtok + tokens_out / 1e6 * price.output_per_mtok


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


@dataclass(frozen=True)
class CostReport:
    """Aggregate cost and latency across a run's calls."""

    n_calls: int
    total_cost_usd: float
    cost_by_tier: dict[str, float]
    p50_latency_ms: float
    p95_latency_ms: float
    cache_hit_rate: float
    extra: dict = field(default_factory=dict)


def build_cost_report(records: list[CallRecord]) -> CostReport:
    """Summarize a list of call records into totals and latency percentiles."""
    latencies = [r.latency_ms for r in records]
    by_tier: dict[str, float] = {}
    for r in records:
        by_tier[r.tier] = by_tier.get(r.tier, 0.0) + r.cost_usd
    hits = sum(1 for r in records if r.cache_hit)
    return CostReport(
        n_calls=len(records),
        total_cost_usd=sum(r.cost_usd for r in records),
        cost_by_tier=by_tier,
        p50_latency_ms=_percentile(latencies, 0.5),
        p95_latency_ms=_percentile(latencies, 0.95),
        cache_hit_rate=(hits / len(records)) if records else 0.0,
    )
