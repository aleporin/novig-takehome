"""Tests for the output guardrail: deterministic checks and regen/downgrade."""

from __future__ import annotations

from tests.fakes import FakeLLMClient
from triage.config import Config
from triage.context.assembler import PromptAssembler
from triage.schemas import Category, Classification, RiskFlags, Urgency
from triage.stages.output_guard import (
    AUDIT_CANARIES,
    AuditVerdict,
    deterministic_violations,
    produce_draft,
    run_audit_canaries,
)

_SIG = "\n\nNovig Support"
_CLEAN = AuditVerdict(violates=False)


def _cls() -> Classification:
    return Classification(
        category=Category.trading_mechanics, urgency=Urgency.low, flags=RiskFlags(), confidence=0.9
    )


def _det(draft, ticket):
    return deterministic_violations(draft, ticket, Config())


def test_clean_draft_has_no_deterministic_violations(make_ticket) -> None:
    assert _det("Thanks for reaching out, here is what happens next." + _SIG, make_ticket()) == []


def test_guarantee_and_date_promise_flagged(make_ticket) -> None:
    v = _det("We guarantee your funds will arrive by Friday." + _SIG, make_ticket())
    assert any("guarantee" in x for x in v)
    assert any("day-of-week" in x for x in v)


def test_missing_signature_flagged(make_ticket) -> None:
    assert any("signature" in x for x in _det("Thanks, we will look into this.", make_ticket()))


def test_invented_amount_flagged_but_standard_auth_allowed(make_ticket) -> None:
    ticket = make_ticket(body="my balance seems off")
    assert any("invented" in x for x in _det("Your balance is $4,300." + _SIG, ticket))
    assert _det("A $1 authorization is standard and reverses." + _SIG, ticket) == []


def _run_guard(ticket, *scripted):
    config = Config()
    return produce_draft(
        ticket,
        _cls(),
        client=FakeLLMClient(list(scripted)),
        assembler=PromptAssembler(config),
        config=config,
    )


def test_clean_draft_passes(make_ticket) -> None:
    draft, calls = _run_guard(make_ticket(), "All good here." + _SIG, _CLEAN)
    assert draft == "All good here." + _SIG
    assert len(calls) == 2  # draft + audit


def test_deterministic_violation_triggers_regen_then_passes(make_ticket) -> None:
    # First draft missing signature -> regen -> clean.
    draft, calls = _run_guard(
        make_ticket(), "No signature here.", _CLEAN, "Fixed reply." + _SIG, _CLEAN
    )
    assert draft == "Fixed reply." + _SIG
    assert len(calls) == 4  # draft, audit, regen, audit


def test_audit_violation_triggers_regen(make_ticket) -> None:
    draft, _ = _run_guard(
        make_ticket(),
        "Looks clean but promises too much." + _SIG,
        AuditVerdict(violates=True, reason="promises an outcome"),
        "Corrected." + _SIG,
        _CLEAN,
    )
    assert draft == "Corrected." + _SIG


def test_second_failure_downgrades_to_no_draft(make_ticket) -> None:
    draft, _ = _run_guard(make_ticket(), "no sig one", _CLEAN, "no sig two", _CLEAN)
    assert draft is None


def test_audit_canaries_catch_and_precision_pass() -> None:
    # Each catch canary flags, the precision canary passes.
    verdicts = [AuditVerdict(violates=c.should_flag) for c in AUDIT_CANARIES]
    report = run_audit_canaries(FakeLLMClient(verdicts), Config())
    assert report.caught == report.n_flag
    assert report.passed == report.n_pass
    assert report.trustworthy


def test_audit_canary_miss_and_false_positive_reported() -> None:
    # Invert every verdict: catch canaries missed, precision canary over-flagged.
    verdicts = [AuditVerdict(violates=not c.should_flag) for c in AUDIT_CANARIES]
    report = run_audit_canaries(FakeLLMClient(verdicts), Config())
    assert len(report.missed) == report.n_flag
    assert len(report.false_positives) == report.n_pass
    assert not report.trustworthy
