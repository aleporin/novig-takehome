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

# Pinned model snapshots. Both tiers are version-frozen: HAIKU uses the dated ID
# format, SONNET the dateless format (from the 4.6 generation on, a dateless id is
# the canonical pinned snapshot, not a floating alias).
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5"
OPUS = "claude-opus-4-8"  # reserved; not used yet

# Judges are cross-provider models — the only non-Anthropic models in the system,
# used in evals only. Two judges from different labs (OpenAI + Google) give an
# independent read on drafts written by an Anthropic model, and their agreement
# rate is itself an eval signal. Pinned like the others; swap here if needed.
OPENAI_JUDGE = "gpt-5"
GEMINI_JUDGE = "gemini-2.5-pro"
XAI_JUDGE = "grok-3-mini"

# Models that reject the temperature parameter (deprecated on newer models). We omit
# it for them, so their output is byte-reproducible via the disk cache, not via
# temperature-0. Applies to T2 (Sonnet); T1 (Haiku) still runs at temperature 0.
TEMPERATURE_DEPRECATED = frozenset({SONNET, OPUS})


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens. Keep in sync with the Anthropic pricing page."""

    input_per_mtok: float
    output_per_mtok: float


# Check these against the provider pricing pages before quoting cost numbers.
MODEL_PRICING: dict[str, Pricing] = {
    HAIKU: Pricing(1.00, 5.00),
    SONNET: Pricing(3.00, 15.00),
    OPUS: Pricing(5.00, 25.00),
    OPENAI_JUDGE: Pricing(1.25, 10.00),  # OpenAI judge (eval-only); approximate
    GEMINI_JUDGE: Pricing(1.25, 10.00),  # Google judge (eval-only); approximate
    XAI_JUDGE: Pricing(0.30, 0.50),  # xAI grok-3-mini (eval-only); cheap non-reasoning
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
    model_judge: str = OPENAI_JUDGE  # primary cross-provider judge; eval-only; optional key
    model_judge_secondary: str = GEMINI_JUDGE  # second-lab judge for agreement; optional key
    model_judge_tertiary: str = XAI_JUDGE  # third-lab judge to break ties by majority vote

    # Draft length bound (characters) enforced by the output guardrail.
    draft_max_chars: int = 1600

    # T1 escalates to T2 below this confidence. Frozen at 0.80 from a measured sweep:
    # after the 'other is a last resort' prompt fix T1 reaches 100% category on its
    # own, so 0.80 matches 0.90's accuracy at a lower escalation rate (33% vs 43%).
    t1_confidence_threshold: float = 0.80
    escalation_rate_floor: float = 0.15
    escalation_rate_ceiling: float = 0.35

    # How to handle policy/spec questions. Decline by default.
    binding_spec_mode: str = "no_draft"  # or "defer_to_spec"

    # Prompt size limits per stage, in tokens.
    classify_token_budget: int = 3000
    draft_token_budget: int = 4000

    # Temperature 0 where the model accepts it (T1/Haiku). Models in
    # TEMPERATURE_DEPRECATED (T2/Sonnet) omit it; those are reproducible via the cache.
    temperature: float = 0.0
    request_timeout_s: float = 60.0
    max_retries: int = 4
    max_output_tokens: int = 1024

    # Cache model responses on disk so reruns are free and repeatable.
    cache_enabled: bool = True

    # Logging full prompts and responses is off by default: ticket text is personal data.
    log_payloads: bool = False


def load_api_key(env_path: Path | None = None, *, required: bool = True) -> str | None:
    """Read ANTHROPIC_API_KEY from secrets.env or the environment.

    Not called at import time, so the offline tests never need a key. With
    required=True (default) raises if the key is missing; with required=False
    returns None, so a fully cached run can proceed with no key.
    """
    from dotenv import load_dotenv

    load_dotenv(env_path or (PROJECT_ROOT / "secrets.env"))
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key and required:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy secrets.env.example to secrets.env "
            "and add your key, or export it in the environment."
        )
    return key


def load_openai_key(env_path: Path | None = None) -> str | None:
    """Read OPENAI_API_KEY for the primary judge. Optional: returns None if absent.

    The judge is the only thing that needs it, and it degrades gracefully — a
    missing key makes judge metrics report 'skipped', never an error.
    """
    from dotenv import load_dotenv

    load_dotenv(env_path or (PROJECT_ROOT / "secrets.env"))
    return os.environ.get("OPENAI_API_KEY")


def load_google_key(env_path: Path | None = None) -> str | None:
    """Read GOOGLE_API_KEY for the secondary (Gemini) judge. Optional.

    Same graceful-degrade contract as load_openai_key: a missing key just drops
    the second judge from the agreement report.
    """
    from dotenv import load_dotenv

    load_dotenv(env_path or (PROJECT_ROOT / "secrets.env"))
    return os.environ.get("GOOGLE_API_KEY")


def load_xai_key(env_path: Path | None = None) -> str | None:
    """Read XAI_API_KEY for the tertiary (Grok) judge. Optional.

    Same graceful-degrade contract: a missing key just drops the third judge,
    and the majority vote falls back to pairwise agreement.
    """
    from dotenv import load_dotenv

    load_dotenv(env_path or (PROJECT_ROOT / "secrets.env"))
    return os.environ.get("XAI_API_KEY")
