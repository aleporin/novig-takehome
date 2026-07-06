"""Cross-provider (GPT-class) judge for draft quality. Eval-only.

The only non-Anthropic model in the system. Missing OPENAI_API_KEY degrades
gracefully: build_judge returns None and the reporter records 'skipped'. Same
discipline as the Anthropic client — retries, timeout, disk cache, faked in tests —
with the judge model pinned in config. Before any judge number is quoted, the judge
must catch every seeded canary violation, or it is demoted.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from triage.config import Config, load_openai_key
from triage.llm.cache import ResponseCache
from triage.llm.types import LLMError, TransientLLMError
from triage.routing.cost_tracker import CallRecord, call_cost
from triage.schemas import Ticket, TicketMetadata

logger = logging.getLogger(__name__)

_CRITERIA = (
    "acknowledges_specifics",
    "no_unverifiable_promise",
    "states_next_steps",
    "invents_no_policy",
    "consistent_with_gold",
    "within_length",
)


class JudgeVerdict(BaseModel):
    """Binary score per rubric criterion plus a one-line justification."""

    acknowledges_specifics: bool
    no_unverifiable_promise: bool
    states_next_steps: bool
    invents_no_policy: bool
    consistent_with_gold: bool
    within_length: bool
    justification: str = ""

    @property
    def passed(self) -> bool:
        return all(getattr(self, name) for name in _CRITERIA)


@runtime_checkable
class JudgeClient(Protocol):
    """The seam between the eval and the judge model (real or fake)."""

    def score(self, system: str, user: str) -> JudgeVerdict: ...


def _key(model: str, system: str, user: str) -> str:
    payload = json.dumps({"m": model, "s": system, "u": user}, sort_keys=True)
    return "judge_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OpenAIJudge:
    """OpenAI-backed judge with retries and disk caching."""

    def __init__(self, api_key, model, *, cache=None, timeout_s=60.0, max_retries=4) -> None:
        self._api_key = api_key
        self._model = model
        self._cache = cache
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._client = None
        self.calls: list[CallRecord] = []  # per-call usage for the judge cost report

    def _openai(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(api_key=self._api_key, timeout=self._timeout)
        return self._client

    def score(self, system: str, user: str) -> JudgeVerdict:
        key = _key(self._model, system, user)
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                self._record(
                    cached.get("tokens_in", 0), cached.get("tokens_out", 0), cache_hit=True
                )
                return JudgeVerdict.model_validate_json(cached["text"])
        verdict, tokens_in, tokens_out = self._call_with_retry(system, user)
        if self._cache is not None:
            self._cache.set(
                key,
                {
                    "text": verdict.model_dump_json(),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                },
                validator=lambda p: JudgeVerdict.model_validate_json(p["text"]),
            )
        self._record(tokens_in, tokens_out, cache_hit=False)
        return verdict

    def _record(self, tokens_in: int, tokens_out: int, *, cache_hit: bool) -> None:
        cost = call_cost(self._model, tokens_in, tokens_out)
        self.calls.append(
            CallRecord(self._model, "judge", tokens_in, tokens_out, 0.0, cache_hit, cost)
        )

    def _call_with_retry(self, system: str, user: str) -> tuple[JudgeVerdict, int, int]:
        retryer = retry(
            reraise=True,
            retry=retry_if_exception_type(TransientLLMError),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=0.5, max=8),
        )
        return retryer(self._invoke)(system, user)

    def _invoke(self, system: str, user: str) -> tuple[JudgeVerdict, int, int]:
        # Structured output: the pydantic schema forces flat booleans, not free JSON.
        import openai

        try:
            completion = self._openai().chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=JudgeVerdict,
            )
        except (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ) as exc:
            raise TransientLLMError(str(exc)) from exc
        except openai.OpenAIError as exc:
            raise LLMError(str(exc)) from exc
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise LLMError("judge returned no parsed output")
        usage = completion.usage
        return parsed, usage.prompt_tokens, usage.completion_tokens


def build_judge(config: Config) -> JudgeClient | None:
    """Return a judge, or None when OPENAI_API_KEY is absent (graceful skip)."""
    key = load_openai_key()
    if not key:
        return None
    cache = ResponseCache(config.paths.cache) if config.cache_enabled else None
    return OpenAIJudge(
        key,
        config.model_judge,
        cache=cache,
        timeout_s=config.request_timeout_s,
        max_retries=config.max_retries,
    )


def load_rubric(config: Config) -> str:
    return (config.paths.skills / "judge" / "SKILL.md").read_text(encoding="utf-8")


def _user_block(ticket: Ticket, draft: str, gold_notes: str) -> str:
    return (
        f"TICKET:\n{ticket.subject}\n{ticket.body}\n\nGOLD NOTES:\n{gold_notes}\n\nDRAFT:\n{draft}"
    )


def score_draft(
    judge: JudgeClient, rubric: str, ticket: Ticket, draft: str, gold_notes: str
) -> JudgeVerdict:
    return judge.score(rubric, _user_block(ticket, draft, gold_notes))


# --- Canary calibration ---

_CANARY_TICKET = Ticket(
    ticket_id="canary",
    subject="withdrawal stuck",
    body="my $500 ACH withdrawal has been pending for 3 days",
    metadata=TicketMetadata(user_tenure_days=100, submitted_at="2026-01-01T00:00:00Z"),
)
_CANARY_GOLD = (
    "Acknowledge the $500 amount and pending status; do not promise a date; say ops will review."
)


@dataclass(frozen=True)
class Canary:
    """A deliberately broken draft and the criterion that must catch it."""

    name: str
    draft: str
    expected_false: str


CANARIES = (
    Canary(
        "promised_date",
        "Your $500 withdrawal will land by Friday.\n\nNovig Support",
        "no_unverifiable_promise",
    ),
    Canary(
        "invented_fee",
        "On your $500 withdrawal there is a standard $25 fee.\n\nNovig Support",
        "invents_no_policy",
    ),
    Canary(
        "generic_no_specifics",
        "Thanks for reaching out. We will help you.\n\nNovig Support",
        "acknowledges_specifics",
    ),
)


@dataclass
class CanaryReport:
    """Result of running the judge over the seeded violations."""

    total: int
    caught: int
    misses: list[str] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.caught / self.total if self.total else 0.0

    @property
    def trustworthy(self) -> bool:
        return self.total > 0 and self.caught == self.total


def run_canaries(judge: JudgeClient, rubric: str) -> CanaryReport:
    """Score each seeded violation; a canary is caught if its criterion scores False."""
    caught, misses = 0, []
    for canary in CANARIES:
        verdict = score_draft(judge, rubric, _CANARY_TICKET, canary.draft, _CANARY_GOLD)
        if not getattr(verdict, canary.expected_false):
            caught += 1
        else:
            misses.append(canary.name)
    return CanaryReport(len(CANARIES), caught, misses)
