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
| false-decline | 2 / 17 (12%), non-deterministic — see below |
| escalation rate | 33% (target 15–35%) |
| judge canary calibration | 3 / 3 caught → trustworthy |

Cost ≈ $0.50 (Anthropic, cached) + ~$0.22 (judge). \* 100% is **threshold-optimized on
validation**, not a held-out estimate — the eval set is unlabeled. Unlabeled signal there:
escalation drifts 33%→47%, draft-rate 70%→53%, confidence floor 0.75→0.60 — the eval set is
measurably more sensitive/urgent than train, which the system reflects.

## Where it fails (with IDs)
- **Fail-closed declines — `t_train_004`, `t_train_019` (2/17).** The only false-declines are the
  output guard downgrading a draft rather than letting it assert a product fact it can't verify (a
  settlement window, an in-app export path, a 1099 rule). The count is non-deterministic (0–2 across
  runs — the regen runs on temperature-free Sonnet); this pinned run is 2. The deterministic gate
  never false-declines. (`t_train_028`, an earlier example, is now fixed and drafts cleanly.)
- **Guardrail arc (closed).** Three false-negatives slipped past the LLM audit — `t_train_022`
  (invented "commission-on-fill" fee), `t_eval_008` ("price-time priority" stated as Novig's rule),
  and a later round on `t_eval_012`/`t_eval_002` (asserting a withdrawal minimum / an order-type fee
  difference). Fix: an audit criterion enforcing the drafter's facts-list — flag any fee, minimum,
  limit, timeline, or platform rule not present in the ticket or the list. Proven by canaries in **both
  directions**: 3/3 violations caught (including a seeded "fee for expedited review" draft) and 1/1
  clean draft mirroring a user-quoted fact passed. Residual characteristic: the audit over-flags
  routing/escalation language ("routed to security for identity verification") — kept deliberately (an
  exemption would be exploitable) and absorbed by the regenerate-once-then-fail-closed path.
- **`t_train_011`, `t_train_016`** — high-confidence urgency over-calls that no escalation
  threshold reaches. The cascade fixes category, not urgency.
- **Draft quality — one gap, two symptoms.** The judge flags 8 drafts as inconsistent with the gold
  notes; **~5 are the system correctly deferring on a product fact no knowledge base verifies** (fee
  policy, 1099 threshold, whether a P&L filter exists), and the other three are our deliberate
  no-specific-time-promise policy versus the gold's "commit within X hours" (2) plus one omission.
  This is the *same* gap as the fail-closed declines above: when the gold assumes a product fact we
  cannot verify, the system either defers (the judge flags it) or fails closed (a decline). Retrieval
  is the single fix for both — which is why it is next-week item #1.

## Next week
1. **Real KB/RAG retrieval** — the single biggest lever; most draft gaps are missing product
   facts, not reasoning. 2. Multi-judge panel (single-judge variance is large). 3. Grow the audit's
   static facts-list into real retrieval; threshold auto-tuning; ingestion adapters (webhook/email).

## Notes
- **Confidence** = "probability a support lead reviewing this would not overturn the decision."
  Hard-rule no-drafts are 1.0; cross-tier disagreement caps it.
- **Determinism / reproducibility:** T1 (Haiku) runs at temperature 0; T2 (Sonnet) takes no
  temperature, so Sonnet-drafted tickets are reproducible only via the response cache. That cache
  is committed, so `make eval` on a fresh clone byte-reproduces `predictions.jsonl` with no API key
  (the key is needed only on a cache miss). Both models are version-pinned.
- **Flag annotations:** the exemplar tickets' risk flags are hand-annotated (the data has no gold
  flags), so flag-alignment metrics inherit annotator judgment.
- **Category accuracy understates safety:** `t_train_012` was miscategorized yet still correctly
  declined via its risk flag — the decoupled flags-vs-category design working as intended.
- **Scope/time:** this is past the suggested few-hour budget — a deliberate over-investment in
  eval rigor and sensitive-case handling (the two behaviors weighted highest here) rather than
  breadth. The trade I'd defend; the hours I'd own.
