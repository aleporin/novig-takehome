"""Tests for the few-shot exemplar pool and selection."""

from __future__ import annotations

from triage.config import Config
from triage.context.exemplars import (
    EXEMPLAR_IDS,
    is_exemplar,
    load_exemplar_pool,
    select_exemplars,
)


def test_pool_loads_all_exemplars_in_order() -> None:
    pool = load_exemplar_pool(Config().paths.tickets_train)
    assert [t.ticket_id for t in pool] == list(EXEMPLAR_IDS)
    assert all(t.label is not None for t in pool)


def test_select_excludes_the_target_ticket() -> None:
    pool = load_exemplar_pool(Config().paths.tickets_train)
    selected = select_exemplars(pool, "t_train_001")
    assert "t_train_001" not in {t.ticket_id for t in selected}
    assert len(selected) == len(pool) - 1


def test_is_exemplar() -> None:
    assert is_exemplar("t_train_001")
    assert not is_exemplar("t_train_030")
