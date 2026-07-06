"""Tests that config paths point at real files and the settings are sane."""

from __future__ import annotations

import pytest

from triage.config import MODEL_PRICING, Config, load_api_key


def test_data_paths_point_at_real_files() -> None:
    paths = Config().paths
    assert paths.tickets_train.exists()
    assert paths.tickets_eval.exists()
    assert paths.taxonomy.exists()


def test_escalation_budget_is_a_valid_interval() -> None:
    cfg = Config()
    assert 0.0 < cfg.escalation_rate_floor < cfg.escalation_rate_ceiling < 1.0


def test_routed_models_have_pricing() -> None:
    cfg = Config()
    for model in (cfg.model_t1, cfg.model_t2, cfg.model_judge):
        assert model in MODEL_PRICING


def test_missing_api_key_fails_clearly(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    empty_env = tmp_path / "secrets.env"
    empty_env.write_text("")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        load_api_key(empty_env)
