# Support Triage — Writeup

## Approach
A linear pipeline of pure, testable stages:
`prescreen → classify → gate → soft policy → draft → output guard → emit`,
behind one interface: `predict(ticket) -> Prediction`. Two commitments shaped everything:

- **Safety is enforcement, not detection.** *Enforcement* is unconditional: no ticket
  with a sensitive category or a hard-rule flag is ever drafted — a pure-Python gate with
  no model in the path, proven across all 11 categories × every hard-flag combination plus
  property tests, at 100% branch coverage with no API key. *Detection* (which tickets carry
  a flag) is probabilistic, from **two independent detectors** — a lexical pre-screen and the
  LLM classifier — so a miss requires both to fail on the same ticket. I never claim
  "guaranteed safety"; I claim unconditional enforcement over probabilistic detection.
- **Evals first.** The harness was built before the classifier and iterated against
  (`results/` holds the committed run history: baseline → classifier → cascade → tuned).
  Runs are byte-reproducible via a validate-before-cache disk cache.

**Key decisions:** a Haiku→Sonnet cascade (escalation only *adds* protection, never downgrades —
property-tested); a cross-provider (OpenAI) judge, eval-only, demoted unless it catches every
seeded canary; **no RAG** (none provided — the drafter declines to invent facts).

## What the evals showed
Validation pool, n=22 (exemplars excluded); 95% bootstrap CIs:

| metric | value |
|---|---|
| category accuracy | 100% [100, 100] \* |
| urgency accuracy | 82% [64, 95] |
| severity-weighted urgency error | 0.18 |
| **false-draft, hard rules** | **0 / 5** (and 0 across all labeled data) |
| false-decline | 2 / 17 (12%), non-deterministic — see below |
| escalation rate | 33% (target 15–35%) |
| judge canary calibration | 3 / 3 caught → trustworthy |

Cost ≈ $0.50 (Anthropic, cached) + ~$0.22 (judge). \* 100% is **threshold-optimized on
validation**, not held-out — the eval set is unlabeled. Unlabeled eval signal: escalation drifts
33%→47%, draft-rate 70%→53%, confidence floor 0.75→0.60 — eval is more sensitive than train,
which the system reflects.

## Where it fails (with IDs)
- **Fail-closed declines — `t_train_004`, `t_train_019` (2/17).** The only false-declines are the
  guard downgrading a draft rather than assert an unverifiable product fact (settlement window, 1099
  rule). Non-deterministic (0–2 across runs — regen runs on temperature-free Sonnet); this pinned run
  is 2. The deterministic gate never false-declines. (`t_train_028`, an earlier example, is now fixed.)
- **Guardrail arc (closed).** The draft audit missed three invented-fact drafts (`t_train_022`,
  `t_eval_008`, the `t_eval_012`/`t_eval_002` class); fixed with a facts-list audit criterion, proven
  by canaries in both directions (3/3 violations caught, 1/1 clean user-quote passed). Residual: it
  over-flags routing language — deliberately kept, absorbed by the fail-closed regen path.
- **`t_train_011`, `t_train_016`** — high-confidence urgency over-calls no escalation threshold
  reaches. The cascade fixes category, not urgency.
- **Draft quality — one gap, two symptoms.** The judge flags 8 drafts as gold-inconsistent; ~5 are
  the system correctly deferring on a product fact no KB verifies (fee policy, 1099 threshold, a
  P&L-filter feature), the rest our no-specific-time-promise policy vs the gold's "within X hours"
  (2) plus one omission. Same gap as the declines above — the system defers (judge flags it) or
  fails closed (a decline) — so retrieval fixes both, hence next-week #1.

## Next week
1. **Real KB/RAG retrieval** — the biggest lever (most draft gaps are missing product facts).
2. Multi-judge panel (single-judge variance). 3. Threshold auto-tuning; ingestion adapters.

## Notes
- **Confidence** = probability a support lead would not overturn the decision (hard-rule no-drafts
  1.0; cross-tier disagreement caps it).
- **Reproducibility:** Haiku is temperature 0; Sonnet takes none, so its drafts reproduce only via
  the committed cache — `make eval` byte-reproduces `predictions.jsonl` on a fresh clone, no key.
  Both models pinned.
- **Flag annotations:** the exemplar risk flags are hand-annotated (the data has no gold flags), so
  flag-alignment metrics inherit annotator judgment.
- **Category accuracy understates safety:** `t_train_012` was miscategorized yet still correctly
  declined via its risk flag — decoupled flags-vs-category working.
- **Scope/time:** this is past the suggested few-hour budget — a deliberate over-investment in
  eval rigor and sensitive-case handling (the two behaviors weighted highest here) rather than
  breadth. The trade I'd defend; the hours I'd own.
