"""Loading tickets and JSONL files for the eval harness."""

from __future__ import annotations

import json
from pathlib import Path

from triage.schemas import Ticket


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_tickets(path: Path) -> list[Ticket]:
    """Read and validate tickets from a JSONL file, keeping file order."""
    return [Ticket.model_validate(row) for row in load_jsonl(path)]


def find_ticket(train_path: Path, eval_path: Path, ticket_id: str) -> Ticket:
    """Look up one ticket by id across the train and eval files."""
    for path in (train_path, eval_path):
        for ticket in load_tickets(path):
            if ticket.ticket_id == ticket_id:
                return ticket
    raise KeyError(f"ticket {ticket_id!r} not found in train or eval")
