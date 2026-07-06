"""Validate a predictions file against the output contract.

Reusable for any predictions file and any tickets file, so it also checks
train-set runs during iteration, not just the final eval output. Returns a list
of problems (empty means valid) instead of raising, so a caller can print them all.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from triage.schemas import Prediction, Ticket

from .dataset import load_jsonl, load_tickets

_FIELDS = {
    "ticket_id",
    "category",
    "urgency",
    "should_draft",
    "no_draft_reason",
    "draft_response",
    "confidence",
}


def _raw_type_errors(row: dict) -> list[str]:
    """Check the JSON-level types the schema would otherwise quietly coerce."""
    errors = []
    extra = set(row) - _FIELDS
    missing = _FIELDS - set(row)
    if extra:
        errors.append(f"unexpected fields {sorted(extra)}")
    if missing:
        errors.append(f"missing fields {sorted(missing)}")
    if "should_draft" in row and not isinstance(row["should_draft"], bool):
        errors.append("should_draft must be a JSON boolean")
    if "confidence" in row:
        value = row["confidence"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append("confidence must be a JSON number")
    return errors


def validate_predictions(pred_rows: list[dict], tickets: list[Ticket]) -> list[str]:
    """Return every contract violation across the predictions and their tickets."""
    errors = []
    if len(pred_rows) != len(tickets):
        errors.append(f"line count {len(pred_rows)} != ticket count {len(tickets)}")

    for i, (row, ticket) in enumerate(zip(pred_rows, tickets)):
        prefix = f"line {i} ({ticket.ticket_id})"
        raw_errors = _raw_type_errors(row)
        errors.extend(f"{prefix}: {e}" for e in raw_errors)
        try:
            pred = Prediction.model_validate(row)
        except ValidationError as exc:
            errors.append(
                f"{prefix}: {exc.error_count()} schema error(s): {exc.errors()[0]['msg']}"
            )
            continue
        if pred.ticket_id != ticket.ticket_id:
            errors.append(f"{prefix}: ticket_id '{pred.ticket_id}' breaks input order")
    return errors


def validate_file(predictions_path: Path, tickets_path: Path) -> list[str]:
    """Load both files from disk and validate the predictions against the tickets."""
    return validate_predictions(load_jsonl(predictions_path), load_tickets(tickets_path))
