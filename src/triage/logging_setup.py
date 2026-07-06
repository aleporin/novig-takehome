"""Logging setup and the shared log-field names.

Logs are the record of why the system made each decision. This is the one place
to configure logging and the one place the field-name constants are defined, so
the cost tracker, the reporter, and any log reader all use the same names.

Two outputs: a readable console handler and a JSON-lines file. The current ticket
id is attached to every line automatically, so stages don't pass it around.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from collections.abc import Iterator
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

# Log field names. Defined once here so everything uses the same strings.
TS = "ts"
LEVEL = "level"
LOGGER = "logger"
TICKET_ID = "ticket_id"
STAGE = "stage"
EVENT = "event"
TIER = "tier"
MODEL = "model"
DECISION = "decision"
RULE_FIRED = "rule_fired"
FLAG_SOURCE = "flag_source"
TOKENS_IN = "tokens_in"
TOKENS_OUT = "tokens_out"
COST_USD = "cost_usd"
LATENCY_MS = "latency_ms"
CACHE_HIT = "cache_hit"

# Extra fields the JSON formatter includes when present on a record.
_EVENT_FIELDS: tuple[str, ...] = (
    STAGE,
    EVENT,
    TIER,
    MODEL,
    DECISION,
    RULE_FIRED,
    FLAG_SOURCE,
    TOKENS_IN,
    TOKENS_OUT,
    COST_USD,
    LATENCY_MS,
    CACHE_HIT,
)

# Current ticket id. A filter copies it onto every log record.
_ticket_id_var: ContextVar[str | None] = ContextVar("ticket_id", default=None)


class TicketIdFilter(logging.Filter):
    """Copy the current ticket id onto each log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.ticket_id = _ticket_id_var.get()
        return True


class JsonLinesFormatter(logging.Formatter):
    """Format each record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            TS: datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            LEVEL: record.levelname,
            LOGGER: record.name,
            TICKET_ID: getattr(record, TICKET_ID, None),
            "message": record.getMessage(),
        }
        for field_name in _EVENT_FIELDS:
            if hasattr(record, field_name):
                payload[field_name] = getattr(record, field_name)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def report_logger() -> logging.Logger:
    """Logger for user-facing CLI output: plain lines to stdout, no prefixes.

    Used by the eval runners and inspection commands instead of print(). It has
    its own message-only handler and does not propagate, so report text stays
    clean and separate from the structured audit log.
    """
    logger = logging.getLogger("triage.report")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


@contextlib.contextmanager
def ticket_context(ticket_id: str) -> Iterator[None]:
    """Set the ticket id for a with-block. Log lines inside pick it up."""
    token = _ticket_id_var.set(ticket_id)
    try:
        yield
    finally:
        _ticket_id_var.reset(token)


def setup_logging(
    *,
    console_level: int = logging.INFO,
    log_dir: Path | None = None,
    file_level: int = logging.DEBUG,
    enable_file: bool = True,
) -> Path | None:
    """Configure logging. Returns the JSON log file path, or None if disabled.

    Safe to call more than once: it clears old handlers first. Safety-relevant
    events are logged at WARNING, so grepping the JSON file for WARNING shows
    every sensitive decision in a run.
    """
    root = logging.getLogger()
    root.setLevel(min(console_level, file_level))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    ticket_filter = TicketIdFilter()

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s [%(ticket_id)s] %(name)s: %(message)s")
    )
    console.addFilter(ticket_filter)
    root.addHandler(console)

    if not enable_file:
        return None

    log_dir = log_dir or (Path(__file__).resolve().parents[2] / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"run_{stamp}.jsonl"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(JsonLinesFormatter())
    file_handler.addFilter(ticket_filter)
    root.addHandler(file_handler)
    return log_path
