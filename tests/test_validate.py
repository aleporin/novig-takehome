"""Tests for the predictions-file validator."""

from __future__ import annotations

from evals.validate import validate_predictions


def _good_row(ticket_id="t1", **overrides) -> dict:
    row = {
        "ticket_id": ticket_id,
        "category": "trading_mechanics",
        "urgency": "low",
        "should_draft": False,
        "no_draft_reason": "declined",
        "draft_response": None,
        "confidence": 0.5,
    }
    row.update(overrides)
    return row


def test_valid_file_has_no_errors(make_ticket) -> None:
    tickets = [make_ticket(ticket_id="t1")]
    assert validate_predictions([_good_row("t1")], tickets) == []


def test_line_count_mismatch(make_ticket) -> None:
    tickets = [make_ticket(ticket_id="t1"), make_ticket(ticket_id="t2")]
    errors = validate_predictions([_good_row("t1")], tickets)
    assert any("line count" in e for e in errors)


def test_order_mismatch_is_flagged(make_ticket) -> None:
    tickets = [make_ticket(ticket_id="t1")]
    errors = validate_predictions([_good_row("WRONG")], tickets)
    assert any("order" in e for e in errors)


def test_unexpected_field_is_flagged(make_ticket) -> None:
    tickets = [make_ticket(ticket_id="t1")]
    errors = validate_predictions([_good_row("t1", surprise=1)], tickets)
    assert any("unexpected fields" in e for e in errors)


def test_should_draft_as_string_is_flagged(make_ticket) -> None:
    tickets = [make_ticket(ticket_id="t1")]
    errors = validate_predictions([_good_row("t1", should_draft="false")], tickets)
    assert any("should_draft must be a JSON boolean" in e for e in errors)


def test_confidence_as_string_is_flagged(make_ticket) -> None:
    tickets = [make_ticket(ticket_id="t1")]
    errors = validate_predictions([_good_row("t1", confidence="0.5")], tickets)
    assert any("confidence must be a JSON number" in e for e in errors)


def test_mutual_exclusion_violation_is_flagged(make_ticket) -> None:
    tickets = [make_ticket(ticket_id="t1")]
    row = _good_row("t1", should_draft=True, no_draft_reason=None, draft_response=None)
    errors = validate_predictions([row], tickets)
    assert any("schema error" in e for e in errors)


def test_bad_enum_is_flagged(make_ticket) -> None:
    tickets = [make_ticket(ticket_id="t1")]
    errors = validate_predictions([_good_row("t1", category="not_a_category")], tickets)
    assert any("schema error" in e for e in errors)
