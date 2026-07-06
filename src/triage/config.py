"""Settings for the triage system, kept in one place.

Holds values only, no logic: model names, the confidence threshold that decides
when to use a stronger model, token and retry limits, and file paths. Change how
the system behaves by editing a value here. Safety rules are enforced in code, not
stored as settings, so they can't be turned off from this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repo root. This file is at src/triage/config.py, so go up three levels.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Pinned model versions so runs are reproducible.
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5"
OPUS = "claude-opus-4-8"  # reserved; not used yet

# Models that reject the temperature parameter (deprecated on newer models). We
# omit it for them; they are deterministic enough and caching makes runs reproducible.
TEMPERATURE_DEPRECATED = frozenset({SONNET, OPUS})


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens. Keep in sync with the Anthropic pricing page."""

    input_per_mtok: float
    output_per_mtok: float


# Check these against the pricing page before quoting cost numbers.
MODEL_PRICING: dict[str, Pricing] = {
    HAIKU: Pricing(1.00, 5.00),
    SONNET: Pricing(3.00, 15.00),
    OPUS: Pricing(5.00, 25.00),
}


@dataclass(frozen=True)
class Paths:
    """File and directory locations. logs/ and .cache/ are gitignored."""

    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    tickets_train: Path = PROJECT_ROOT / "data" / "tickets_train.jsonl"
    tickets_eval: Path = PROJECT_ROOT / "data" / "tickets_eval.jsonl"
    taxonomy: Path = PROJECT_ROOT / "data" / "taxonomy.md"
    skills: Path = PROJECT_ROOT / "skills"
    results: Path = PROJECT_ROOT / "results"
    logs: Path = PROJECT_ROOT / "logs"
    cache: Path = PROJECT_ROOT / ".cache"
    predictions: Path = PROJECT_ROOT / "predictions.jsonl"


@dataclass(frozen=True)
class Config:
    """Run settings. Build one and pass it down through the pipeline."""

    paths: Paths = field(default_factory=Paths)

    # Which model each tier uses.
    model_t1: str = HAIKU
    model_t2: str = SONNET
    model_judge: str = SONNET  # use a different model from the drafter when possible

    # T1 escalates to T2 below this confidence. Frozen at 0.90 from a measured
    # threshold sweep: it fixes the classifier's 'other' category traps (82%->95%).
    t1_confidence_threshold: float = 0.90
    escalation_rate_floor: float = 0.15
    escalation_rate_ceiling: float = 0.35

    # How to handle policy/spec questions. Decline by default.
    binding_spec_mode: str = "no_draft"  # or "defer_to_spec"

    # Prompt size limits per stage, in tokens.
    classify_token_budget: int = 3000
    draft_token_budget: int = 4000

    # LLM call behavior. Temperature 0 for repeatable output.
    temperature: float = 0.0
    request_timeout_s: float = 60.0
    max_retries: int = 4
    max_output_tokens: int = 1024

    # Cache model responses on disk so reruns are free and repeatable.
    cache_enabled: bool = True

    # Logging full prompts and responses is off by default: ticket text is personal data.
    log_payloads: bool = False


def load_api_key(env_path: Path | None = None) -> str:
    """Read ANTHROPIC_API_KEY from secrets.env or the environment.

    Not called at import time, so the offline tests never need a key. Raises if
    the key is missing.
    """
    from dotenv import load_dotenv

    load_dotenv(env_path or (PROJECT_ROOT / "secrets.env"))
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy secrets.env.example to secrets.env "
            "and add your key, or export it in the environment."
        )
    return key
