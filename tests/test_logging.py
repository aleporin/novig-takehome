"""Tests for JSON log formatting and automatic ticket-id tagging."""

from __future__ import annotations

import json
import logging

from triage.logging_setup import (
    STAGE,
    TICKET_ID,
    JsonLinesFormatter,
    TicketIdFilter,
    ticket_context,
)


def _record(msg: str, **extra) -> logging.LogRecord:
    record = logging.LogRecord("triage.test", logging.INFO, __file__, 1, msg, None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_emits_known_event_fields() -> None:
    record = _record("classified", **{STAGE: "classify"})
    TicketIdFilter().filter(record)
    payload = json.loads(JsonLinesFormatter().format(record))
    assert payload["message"] == "classified"
    assert payload[STAGE] == "classify"
    assert payload["level"] == "INFO"


def test_ticket_context_injects_id_into_records() -> None:
    formatter = JsonLinesFormatter()
    filt = TicketIdFilter()
    with ticket_context("t_eval_007"):
        record = _record("inside")
        filt.filter(record)
        assert json.loads(formatter.format(record))[TICKET_ID] == "t_eval_007"
    outside = _record("outside")
    filt.filter(outside)
    assert json.loads(formatter.format(outside))[TICKET_ID] is None
