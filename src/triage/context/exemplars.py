"""The few-shot exemplar pool and selection.

A fixed, balanced set of 8 training tickets used as few-shot examples: 4 draftable
and 4 no-draft, covering the three sensitive categories and a soft-rule fact
dispute. Because the pool is fixed, the eval can report headline metrics on the
other 22 tickets, which never appear in a prompt. When predicting a pool ticket
itself, it is left out of its own prompt.

Gold labels don't include risk flags, so the flags shown per exemplar are
annotated here to teach the target format.
"""

from __future__ import annotations

import json
from pathlib import Path

from triage.schemas import Ticket

EXEMPLAR_IDS: tuple[str, ...] = (
    "t_train_001",  # trading_mechanics, low, draft
    "t_train_002",  # deposits_withdrawals, high, draft
    "t_train_007",  # bug_report, medium, draft
    "t_train_008",  # kyc_verification, high, draft
    "t_train_003",  # account_compromise, no-draft
    "t_train_006",  # problem_gambling, no-draft
    "t_train_009",  # legal_regulatory, no-draft
    "t_train_029",  # market_questions, soft-rule fact dispute, no-draft
)

# Flags to display for each exemplar (gold labels don't carry flags).
EXEMPLAR_FLAGS: dict[str, tuple[str, ...]] = {
    "t_train_003": ("active_fraud", "unauthorized_access_reported"),
    "t_train_006": ("rg_signal",),
    "t_train_009": ("legal_threat",),
    "t_train_029": ("disputes_novig_fact",),
}


def load_exemplar_pool(train_path: Path) -> list[Ticket]:
    """Load the exemplar tickets from the training file, in pool order."""
    by_id = {}
    for line in train_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["ticket_id"] in EXEMPLAR_IDS:
                by_id[row["ticket_id"]] = Ticket.model_validate(row)
    return [by_id[tid] for tid in EXEMPLAR_IDS if tid in by_id]


def select_exemplars(pool: list[Ticket], exclude_id: str | None = None) -> list[Ticket]:
    """Return the pool, dropping the target ticket so it never sees itself."""
    return [t for t in pool if t.ticket_id != exclude_id]


def is_exemplar(ticket_id: str) -> bool:
    return ticket_id in EXEMPLAR_IDS
