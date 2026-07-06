"""Lexical pre-screen: a model-independent tripwire that runs before any AI.

It scans the subject and body for obviously sensitive language and returns the
risk flags it finds. It only ever sets flags, never clears one, and it is tuned
to over-flag on ambiguous text: a needless escalation is cheaper than a missed
sensitive ticket. The gate treats these flags as independent vetoes, so even a
completely wrong classifier can't draft on a lexically obvious case.

Patterns are deliberately narrow where a broad word would misfire. "minor" is
never matched as a bare word (a bug report can be "super minor"); age is detected
from numbers in a person context or explicit under-age words instead.
"""

from __future__ import annotations

import re

from triage.schemas import RiskFlags

_FLAGS = re.IGNORECASE

# Explicit self-harm and severe hopelessness. Kept narrow to avoid catching
# ordinary frustration ("I don't want to wait") — note the specific objects.
_SELF_HARM = [
    r"\b(kill|hurt|harm)(ing)?\s+myself\b",
    r"\btake my own life\b",
    r"\bend (my life|it all|things)\b",
    r"\bsuicid",
    r"\bwant to die\b",
    r"\bbetter off dead\b",
    r"\bdon'?t want to (live|be here|exist)\b",
    r"\b(can'?t|cannot) go on\b",
    r"\bno reason to (live|go on)\b",
]

# Someone else is acting in the account, framed as happening now.
_ACTIVE_FRAUD = [
    r"\bright now\b",
    r"\bas we speak\b",
    r"\bsomeone('?s| is| is still)?\s+(in|inside|logged in|accessing|withdrawing)",
    r"\bthey'?re (in there|in my account|still in)\b",
    r"\bhappening now\b",
]

# Access or transactions the user says they did not do.
_UNAUTHORIZED = [
    r"\bdid ?n'?t (make|authorize|place|do|recognize)\b",
    r"\bdid not (make|authorize|place|do|recognize)\b",
    r"\bunauthoriz",
    r"\bsomeone (placed|made|opened|withdrew)\b",
    r"\bnot me\b",
    r"\bi was asleep\b",
]

# Legal or regulator threats. "attorney" excludes "attorney general" (a market
# name here); the regulator-office phrasing is caught only in a complaint context.
_LEGAL = [
    r"\battorney\b(?!\s+general)",
    r"\blawyer\b",
    r"\blaw firm\b",
    r"\blaw ?suit\b",
    r"\blitigation\b",
    r"\bsuing\b",
    r"\b(will|gonna|going to|planning to|i'?ll|we'?ll|intend to)\s+sue\b",
    r"\b(cftc|cfpb)\b",
    r"\bsubpoena\b",
    r"\bcease and desist\b",
    r"\blegal (action|proceedings?|counsel)\b",
    r"\b(file|filing|formal)\s+(a\s+)?(complaint|suit)\b",
    # regulator office, but only in a complaint context (not "attorney general market")
    r"\b(contact\w*|report\w*|notify\w*|fil\w+)\b[^.?!]{0,40}\b(attorney general|state ag)\b",
]

# Problem-gambling and loss-of-control language, plus self-exclusion requests.
_RG_SIGNAL = [
    r"\bself[-\s]?exclu",
    r"\bdeposit limit\b",
    r"\b(freeze|lock) me out\b",
    r"\bclose my account permanently\b",
    r"\bfreeze me out\b",
    r"\bshut (it all|it|this|everything|me) down\b",
    r"\b(need|want|have) to stop\b",
    r"\b(can'?t|cannot|don'?t know how to) stop\b",
    r"\bkeep losing\b",
    r"\blost too much\b",
    r"\blosing (control|too much)\b",
    r"\bgambling problem\b",
    r"\baddict",
    r"\bin too deep\b",
    r"\bcan'?t keep doing this\b",
    r"\btake a break from\b",
]

_COMPILED = {
    "self_harm_or_distress": [re.compile(p, _FLAGS) for p in _SELF_HARM],
    "active_fraud": [re.compile(p, _FLAGS) for p in _ACTIVE_FRAUD],
    "unauthorized_access_reported": [re.compile(p, _FLAGS) for p in _UNAUTHORIZED],
    "legal_threat": [re.compile(p, _FLAGS) for p in _LEGAL],
    "rg_signal": [re.compile(p, _FLAGS) for p in _RG_SIGNAL],
}

# Under-age detection. Words that always count, then numbers in a person context.
_MINOR_WORDS = re.compile(
    r"\b(under-?age|under\s?(the\s?age|18|21)|teenager|teen(?![a-z])|high[-\s]?school|juvenile)\b",
    _FLAGS,
)
# "a/the/is a minor" as a person, excluding the small-sense ("a minor bug").
_SMALL_SENSE = "bug|issue|glitch|problem|delay|detail|thing|change|discrepancy|error"
_MINOR_NOUN = re.compile(rf"\b(?:a|is a|the)\s+minor\b(?!\s+({_SMALL_SENSE}))", _FLAGS)
# "he's 17", "i'm 20", "my son is 17"
_AGE_PRONOUN = re.compile(
    r"\b(?:i'?m|i am|he'?s|he is|she'?s|she is|they'?re|"
    r"my (?:son|daughter|child|kid|nephew|niece|grandson|granddaughter) is)\s+(\d{1,2})\b",
    _FLAGS,
)
# "17 year old", "17-year-old", "17 yo"
_AGE_SUFFIX = re.compile(r"\b(\d{1,2})[-\s]*(?:years?[-\s]*old|y/?o|yrs?)\b", _FLAGS)


def _mentions_minor(text: str) -> bool:
    if _MINOR_WORDS.search(text) or _MINOR_NOUN.search(text):
        return True
    for pattern in (_AGE_PRONOUN, _AGE_SUFFIX):
        for match in pattern.finditer(text):
            if int(match.group(1)) < 21:
                return True
    return False


def prescreen(subject: str, body: str) -> RiskFlags:
    """Return the risk flags the lexical screen finds in a ticket's text."""
    text = f"{subject}\n{body}"
    found = {flag: any(rx.search(text) for rx in rxs) for flag, rxs in _COMPILED.items()}
    found["mentions_minor"] = _mentions_minor(text)
    return RiskFlags(**found)


if __name__ == "__main__":  # pragma: no cover - manual probing via `make check-safety`
    import sys

    flags = prescreen("", sys.argv[1] if len(sys.argv) > 1 else "")
    fired = [name for name, value in flags.model_dump().items() if value]
    print("flags fired:", ", ".join(fired) if fired else "(none)")
