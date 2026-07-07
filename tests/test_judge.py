"""Tests for the cross-provider judge: verdicts, canaries, graceful degradation."""

from __future__ import annotations

from evals.judge import (
    _CRITERIA,
    JudgeVerdict,
    build_judge,
    build_judges,
    run_canaries,
    score_draft,
)
from triage.config import Config


def _verdict(*false_names) -> JudgeVerdict:
    kwargs = {name: name not in false_names for name in _CRITERIA}
    return JudgeVerdict(**kwargs)


class FakeJudge:
    """Returns queued verdicts in order and records what it was asked."""

    def __init__(self, verdicts) -> None:
        self._queue = list(verdicts)
        self.calls = []

    def score(self, system, user) -> JudgeVerdict:
        self.calls.append((system, user))
        return self._queue.pop(0)


def test_verdict_passed_requires_all_criteria() -> None:
    assert _verdict().passed is True
    assert _verdict("within_length").passed is False


def test_score_draft_forwards_rubric_and_context() -> None:
    judge = FakeJudge([_verdict()])
    from triage.schemas import Ticket, TicketMetadata

    ticket = Ticket(
        ticket_id="t1",
        subject="s",
        body="b",
        metadata=TicketMetadata(user_tenure_days=1, submitted_at="2026-01-01T00:00:00Z"),
    )
    score_draft(judge, "RUBRIC", ticket, "a draft", "gold notes")
    system, user = judge.calls[0]
    assert system == "RUBRIC"
    assert "a draft" in user and "gold notes" in user


def test_canaries_all_caught_is_trustworthy() -> None:
    # Each canary's expected-false criterion scores False -> caught.
    judge = FakeJudge(
        [
            _verdict("no_unverifiable_promise"),
            _verdict("invents_no_policy"),
            _verdict("acknowledges_specifics"),
        ]
    )
    report = run_canaries(judge, "RUBRIC")
    assert report.trustworthy is True
    assert report.hit_rate == 1.0


def test_canaries_missed_is_not_trustworthy() -> None:
    # An all-pass judge misses every planted violation.
    judge = FakeJudge([_verdict(), _verdict(), _verdict()])
    report = run_canaries(judge, "RUBRIC")
    assert report.trustworthy is False
    assert len(report.misses) == 3


def test_build_judge_is_none_without_key(monkeypatch) -> None:
    monkeypatch.setattr("evals.judge.load_openai_key", lambda: None)
    assert build_judge(Config()) is None


def test_build_judge_returns_judge_with_key(monkeypatch) -> None:
    monkeypatch.setattr("evals.judge.load_openai_key", lambda: "sk-test")
    assert build_judge(Config()) is not None


def test_build_judges_empty_without_keys(monkeypatch) -> None:
    monkeypatch.setattr("evals.judge.load_openai_key", lambda: None)
    monkeypatch.setattr("evals.judge.load_google_key", lambda: None)
    monkeypatch.setattr("evals.judge.load_xai_key", lambda: None)
    assert build_judges(Config()) == []


def test_build_judges_returns_all_three_when_all_keys_present(monkeypatch) -> None:
    monkeypatch.setattr("evals.judge.load_openai_key", lambda: "sk-test")
    monkeypatch.setattr("evals.judge.load_google_key", lambda: "gk-test")
    monkeypatch.setattr("evals.judge.load_xai_key", lambda: "xk-test")
    labels = [label for label, _ in build_judges(Config())]
    assert labels == ["openai", "gemini", "xai"]


def test_build_judges_drops_missing_labs(monkeypatch) -> None:
    monkeypatch.setattr("evals.judge.load_openai_key", lambda: None)
    monkeypatch.setattr("evals.judge.load_google_key", lambda: "gk-test")
    monkeypatch.setattr("evals.judge.load_xai_key", lambda: None)
    labels = [label for label, _ in build_judges(Config())]
    assert labels == ["gemini"]
