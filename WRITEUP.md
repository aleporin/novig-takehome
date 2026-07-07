# Support Triage — Writeup

## Approach
A linear pipeline of pure, independently-testable stages:
`prescreen → classify (routed) → gate → soft policy → draft → output guard → emit`,
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

**Key decisions:** Haiku→Sonnet confidence cascade (escalation can only *add* flags / keep a
sensitive category, never downgrade — proven by property test); a cross-provider (OpenAI)
judge for draft quality, eval-only, key-optional, and demoted unless it catches every seeded
canary; **no RAG** (no knowledge base was provided — the drafter deliberately declines to
invent product facts rather than hallucinate them).

## What the evals showed
Validation pool, n=22 (exemplars excluded); 95% bootstrap CIs:

| metric | value |
|---|---|
| category accuracy | 100% [100, 100] \* |
| urgency accuracy | 82% [64, 95] |
| severity-weighted urgency error | 0.18 |
| **false-draft, hard rules** | **0 / 5** (and 0 across all labeled data) |
| false-decline | 1 / 17 (6%) |
| escalation rate | 33% (target 15–35%) |
| judge canary calibration | 3 / 3 caught → trustworthy |

Cost ≈ $0.52 (Anthropic, cached) + ~$0.20 (judge). \* 100% is **threshold-optimized on
validation**, not a held-out estimate — the eval set is unlabeled. Unlabeled signal there:
escalation drifts 33%→47%, draft-rate 70%→53%, confidence floor 0.75→0.60 — the eval set is
measurably more sensitive/urgent than train, which the system reflects.

## Where it fails (with IDs)
- **`t_train_028`** — the output guard rejected a valid draft twice and downgraded to no-draft.
  Safe, but it is the one false-decline: the cost of failing closed.
- **`t_train_022`** — the draft invented a "commission-on-fill" policy and the LLM audit missed
  it. A genuine guardrail false-negative.
- **`t_train_011`, `t_train_016`** — high-confidence urgency over-calls that no escalation
  threshold reaches. The cascade fixes category, not urgency.
- **Draft quality:** of 10 drafts the judge flagged as inconsistent with the gold notes,
  **6 are the system correctly refusing to invent facts it has no knowledge base for**
  (minimum deposit, fee policy, in-app paths) — only ~4 are genuine misses. True quality-miss
  is ~20%, and the dominant fix is retrieval, not prompting.

## Next week
1. **Real KB/RAG retrieval** — the single biggest lever; most draft gaps are missing product
   facts, not reasoning. 2. Multi-judge panel (single-judge variance is large). 3. Close the
   audit gap behind `t_train_022`; threshold auto-tuning; ingestion adapters (webhook/email).

## Notes
- **Confidence** = "probability a support lead reviewing this would not overturn the decision."
  Hard-rule no-drafts are 1.0; cross-tier disagreement caps it.
- **Determinism:** T1 (Haiku) runs at temperature 0; T2 (Sonnet) rejects the temperature param,
  so it is byte-reproducible via the cache, not via temp-0. Both models are version-pinned.
- **Scope/time:** this is past the suggested few-hour budget — a deliberate over-investment in
  eval rigor and sensitive-case handling (the two behaviors weighted highest here) rather than
  breadth. The trade I'd defend; the hours I'd own.
