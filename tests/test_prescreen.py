"""Pre-screen tests: real sensitive phrasings must trip, look-alikes must not."""

from __future__ import annotations

import pytest

from triage.stages.prescreen import prescreen


def _fired(subject: str, body: str) -> set[str]:
    return {name for name, value in prescreen(subject, body).model_dump().items() if value}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("someone is in my account RIGHT NOW", "active_fraud"),
        ("a $500 withdrawal I did not make", "unauthorized_access_reported"),
        ("there is a $300 trade I did not authorize", "unauthorized_access_reported"),
        ("I need to stop. I've lost too much and can't keep doing this", "rg_signal"),
        ("I'd like to set a weekly deposit limit", "rg_signal"),
        ("shut it all down, I'm tired, take it all", "rg_signal"),
        ("I've already spoken with my attorney", "legal_threat"),
        ("filing a formal complaint with the CFTC", "legal_threat"),
        ("my son is really into this, he's 17", "mentions_minor"),
        ("my nephew is 19, can he use my account", "mentions_minor"),
        ("I'm going to hurt myself because of gambling", "self_harm_or_distress"),
        ("he is a minor and wants to sign up", "mentions_minor"),
        ("are you underage? he is a teenager", "mentions_minor"),
        ("your platform misgrading the Heat/Celtics market last Friday", "disputes_novig_fact"),
        ("this market was graded wrong, it should have settled yes", "disputes_novig_fact"),
        ("my payout amount looks wrong, is this a mistake?", "disputes_novig_fact"),
    ],
)
def test_sensitive_text_trips_the_right_flag(text, expected) -> None:
    assert expected in _fired("", text)


@pytest.mark.parametrize(
    "text",
    [
        # 018: a trivial discrepancy the user blames on themselves must not fire.
        "my balance is off from what I calculated by 30 cents, probably my math",
        "the total seems higher than I expected but might be me miscalculating",
    ],
)
def test_self_attributed_discrepancy_does_not_dispute(text) -> None:
    assert "disputes_novig_fact" not in _fired("", text)


@pytest.mark.parametrize(
    "subject, body",
    [
        (
            "balance wrong by like 30 cents",
            "this is super minor but my balance is off from what I calculated by "
            "about 30 cents. probably my math but figured I'd flag.",
        ),
        ("market question", "when does the attorney general election market settle?"),
        ("love the app", "my son loves using novig, great product"),
        ("withdrawal delay", "my withdrawal is 19 hours late, please help"),
        ("new user", "how do parlays work, i'm new here"),
        ("bug", "the app keeps shutting down randomly on launch"),
    ],
)
def test_innocent_lookalikes_do_not_trip(subject, body) -> None:
    assert _fired(subject, body) == set()


@pytest.mark.parametrize(
    "age, is_minor",
    [("17", True), ("20", True), ("21", False), ("25", False)],
)
def test_under_21_boundary(age, is_minor) -> None:
    fired = "mentions_minor" in _fired("", f"I'm {age} and want to sign up")
    assert fired is is_minor
