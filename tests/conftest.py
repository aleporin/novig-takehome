"""Shared fixtures and test setup.

Also does what a pytest config file would: puts src/ on the import path and
registers the live marker, before any test imports the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from collections.abc import Callable  # noqa: E402  (import after the path bootstrap above)

import pytest  # noqa: E402

from tests.fakes import FakeLLMClient  # noqa: E402
from triage.schemas import Category, Classification, RiskFlags, Ticket, Urgency  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    """Register markers here instead of in a config file."""
    config.addinivalue_line(
        "markers", "live: hits the real Anthropic API; excluded by default, run via `make smoke`."
    )


_BASE_TICKET = {
    "ticket_id": "t_train_001",
    "subject": "how do parlays work on novig?",
    "body": "new here, i don't get how parlays work on an exchange. thanks",
    "metadata": {"user_tenure_days": 3, "submitted_at": "2026-04-12T14:22:00Z"},
}


@pytest.fixture
def make_ticket() -> Callable[..., Ticket]:
    """Factory producing a valid eval-shaped ticket with optional overrides."""

    def _factory(**overrides: object) -> Ticket:
        data = {**_BASE_TICKET, **overrides}
        return Ticket.model_validate(data)

    return _factory


@pytest.fixture
def ticket(make_ticket: Callable[..., Ticket]) -> Ticket:
    return make_ticket()


@pytest.fixture
def classification() -> Classification:
    """A benign, high-confidence classification with no risk flags set."""
    return Classification(
        category=Category.trading_mechanics,
        urgency=Urgency.low,
        flags=RiskFlags(),
        confidence=0.9,
        reasoning="clear product question",
    )


@pytest.fixture
def fake_client() -> FakeLLMClient:
    return FakeLLMClient()
