# Support Triage Writeup

This system triages support tickets for a regulated exchange: it classifies each ticket,
decides whether an AI draft is safe to produce, and refuses to draft, deterministically,
whenever a human must respond.

## Approach
A linear pipeline, `prescreen → classify → gate → soft policy → draft → output guard → emit`,
behind one interface: `predict(ticket) -> Prediction`. Two commitments shaped everything:

- **Safety is enforcement, not detection.** Enforcement is unconditional: no ticket with a
  sensitive category or hard-rule flag is ever drafted. The gate is pure Python with no model
  in the path, tested across all 11 categories × every hard-flag combination at 100% branch
  coverage, no API key needed. Detection (which tickets carry a flag) is probabilistic, from
  two independent detectors, a lexical pre-screen and the LLM classifier, so a miss requires
  both to fail on the same ticket. I claim unconditional enforcement over probabilistic
  detection, never "guaranteed safety."
- **Evals first.** The harness was built before the classifier and iterated against;
  `results/` holds the committed run history, and every run is byte-reproducible via a
  validate-before-cache disk cache.

**Key decisions:** a Haiku→Sonnet cascade (escalation only adds protection, never downgrades;
property-tested); two cross-provider judges from different labs (OpenAI GPT-5 + Google Gemini
2.5 Pro), eval-only, trusted only after each catches every seeded canary — the drafter is
Anthropic, so cross-lab judges give an independent read and their agreement rate is itself an
eval signal. A third judge (xAI grok-4) is plumbed in behind the same JudgeClient protocol —
build_judges gracefully drops it and falls back to the two-judge report when xAI access is
unavailable (as it was at submission); majority-vote is unlocked as soon as a third judge
completes. No RAG (none provided; the drafter declines to invent facts instead).

## What the evals showed
Validation pool, n=22 (the 8 few-shot exemplars are excluded from scoring); 95% bootstrap CIs:

| metric | what it measures | value |
|---|---|---|
| category accuracy | correct category, exact match | 100% [100, 100] \* |
| urgency accuracy | correct urgency level | 82% [64, 95] |
| severity-weighted error | urgency mistakes weighted by harm; under-calling costs most | 0.18 (baseline 2.8) |
| **false-draft, hard rules** | **drafted when a hard rule forbids it: the cardinal error** | **0 / 5** (0 on all labeled data) |
| false-decline | refused a ticket we should have drafted | 2 / 17, non-deterministic (see below) |
| escalation rate | tickets the cheap model hands to the strong one | 33% (budget 15–35%) |
| judge canaries | planted-flaw drafts each judge must catch before scores count | 3 / 3 each (OpenAI, Gemini) |
| judge agreement | two independent judges agree on overall pass/fail per draft | 16 / 19 (84%) |

Cost ≈ $0.50 (Anthropic, cached) + ~$0.22 (judge). \* 100% is **threshold-optimized on
validation**, not held-out; the eval set is unlabeled. On that unlabeled set the system's
behavior shifts the way it should for harder traffic: escalation rises 33%→47%, the share of
tickets it drafts falls 70%→53%, and the lowest confidence it reports drops 0.75→0.60. The
eval tickets are measurably more sensitive than training tickets, and the system gets more
cautious in response.

## Where it fails (with IDs)
- **Fail-closed declines: `t_train_004`, `t_train_019` (the 2/17).** Both are the output
  guard downgrading a draft rather than letting it assert an unverifiable product fact (a
  settlement window; a 1099 rule). The count is non-deterministic, 0–2 across runs, because
  rejected drafts regenerate on temperature-free Sonnet; this pinned run is 2. The
  deterministic gate never false-declines. (`t_train_028`, an earlier instance, is now fixed.)
- **Guardrail arc (closed).** The draft audit missed three invented-fact drafts
  (`t_train_022`, `t_eval_008`, the `t_eval_012`/`t_eval_002` class). Fixed with a facts-list
  audit criterion, proven by canaries in both directions (3/3 violations caught, 1/1 clean
  user-quote passed). Residual: it over-flags routing language, kept deliberately, and the
  fail-closed regen path absorbs the cost.
- **`t_train_011`, `t_train_016`:** high-confidence urgency over-calls that no escalation
  threshold reaches. The cascade fixes category, not urgency.
- **Draft quality: one gap, two symptoms.** The judge flags 8 drafts as inconsistent with the
  gold notes: ~5 are the system correctly deferring on product facts no knowledge base
  verifies, 2 are a deliberate policy clash (gold commits to "within X hours"; we never
  promise timeframes), 1 is a genuine omission. The same missing-KB gap drives the
  fail-closed declines above, so retrieval fixes both. Hence next-week #1.

## Next week
1. **Knowledge-base retrieval.** Index Novig's real policies (fees, minimums, ACH timing,
   settlement windows, KYC steps) and let the drafter cite them, attaching each retrieved
   snippet so the output guard can verify grounding. This converts today's deferrals and
   fail-closed declines into complete answers; it is the single biggest quality lever.
2. **Multi-judge panel.** Two judges from different labs are in now (OpenAI + Google, 84%
   agreement, disagreements concentrated on `consistent_with_gold` and
   `no_unverifiable_promise` — the subjective criteria). A third-lab judge (xAI grok-4)
   is already wired through the same `JudgeClient` protocol with pairwise-agreement and
   majority-vote code paths ready; enabling it is one API-access flip. Once three judges
   complete, gate releases on the majority verdict.
3. **Threshold auto-tuning.** The escalation threshold is currently chosen by a manual sweep;
   close the loop so it re-optimizes as labeled volume grows, with the sweep re-run as a CI
   step.
4. **Context management.** Today it is deliberately minimal: a fixed 8-shot exemplar set, a
   `chars/4` token estimate, lossy truncation on overflow, and one ticket at a time. Production
   needs a real tokenizer, per-ticket exemplar retrieval instead of a static set, thread history
   summarized rather than truncated, and provider-side prompt caching on the shared system
   prefix to cut cost and latency.
5. **Production plumbing.** Webhook/email ingestion in front of `predict()`, and drift alerts on
   the distribution signals above (escalation rate, draft rate, confidence floor).

## Definitions and caveats
- **Confidence** means the probability a support lead reviewing the decision would not
  overturn it. Hard-rule declines report 1.0 because the gate is certain even when the
  category is not; disagreement between the two cascade tiers caps it lower.
- **Reproducibility.** Haiku runs at temperature 0 and is fully deterministic. Sonnet accepts
  no temperature parameter, so its drafts vary run to run; the committed response cache pins
  them. On a fresh clone with no API key, `make eval` replays the cache and reproduces
  `predictions.jsonl` byte-for-byte; delete the cache and add a key to regenerate fresh.
- **Hand-annotated flags.** The dataset has no gold labels for risk flags, so the few-shot
  exemplars' flags are my annotations; any metric about flag agreement inherits that judgment.
- **Category accuracy understates safety.** `t_train_012` was miscategorized yet still
  correctly declined because its risk flag fired independently; flags and category are
  decoupled so a classification miss cannot become a safety miss.
- **Time.** This exceeds the suggested few-hour budget: a deliberate over-investment in eval
  rigor and sensitive-case handling, the two highest-weighted behaviors, rather than breadth.
  The trade I'd defend; the hours I'd own.
