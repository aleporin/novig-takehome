# Support Triage Writeup

I built a system that classifies incoming support tickets, decides whether an AI-drafted
reply is safe to send, and hard-refuses to draft whenever a human needs to answer. Safety
is enforced deterministically; classification and drafting use LLMs.

## Approach

Every ticket flows through the same pipeline: **scan for danger words → classify →
decide (draft or defer) → write the draft → audit the draft → return the result.** It
sits behind one function: `predict(ticket) -> Prediction`.

Two commitments shaped everything:

- **Safety is enforced, not just detected.** If a ticket is flagged as sensitive
  (account compromise, self-harm, legal threat, problem gambling, etc.) the "draft or
  defer" step refuses to draft, period. That step is plain Python — no model in the
  loop — and is tested against every category × flag combination at 100% branch
  coverage. *Whether* a ticket gets flagged is probabilistic — a keyword scan plus
  the LLM classifier are two independent signals, so a miss requires both to fail —
  but once flagged, the refusal is guaranteed. I claim guaranteed *enforcement*, not
  guaranteed *detection*.
- **Evals were built first.** The harness existed before the classifier did, so every
  design change had to justify itself against the numbers. Every run is committed
  under `results/` and reproducible byte-for-byte from a disk cache.

**Key design decisions:**
- **A two-tier cascade** (Claude Haiku → Claude Sonnet). Haiku classifies cheaply; if
  it's unsure, or if the ticket looks sensitive, Sonnet takes over. Escalation only
  adds protection — Sonnet can never undo a Haiku safety flag (property-tested).
- **Three independent judges** grade the drafted replies: OpenAI GPT-5, Google Gemini
  2.5 Pro, and xAI Grok. Different labs than the drafter (Anthropic) means no lab is
  grading its own homework, so agreement — and disagreement — is real signal. Each
  judge must first catch three planted-flaw drafts ("canaries") before its verdicts
  count. Any judge that fails mid-run is dropped without sinking the panel.
- **No retrieval.** No knowledge base was provided; the drafter is prompted to defer
  on any fact it can't verify from the ticket itself.

## What the evals showed

Validation set of 22 tickets (30 labeled minus 8 held out as few-shot examples); 95%
bootstrap confidence intervals.

| Metric | What it measures | Value |
|---|---|---|
| Category accuracy | Predicted category matches the label | **100%** [100, 100] \* |
| Urgency accuracy | Predicted urgency level matches | 82% [64, 95] |
| Severity-weighted urgency error | Urgency mistakes, weighted by harm — under-calling costs most | 0.18 (majority-vote baseline: 2.8) |
| **False-draft on sensitive tickets** | **The cardinal error: drafted when a hard rule said don't** | **0 / 5** (0 across all labeled data) |
| False-decline | Refused to draft when we should have | 2 / 17, non-deterministic (see below) |
| Escalation rate | How often Haiku hands the ticket to Sonnet | 33% (target band 15–35%) |
| Judge canaries | Planted-flaw drafts each judge must catch first | 3 / 3 for all three judges |
| Judge pairwise agreement | How often two judges agree on overall pass/fail | OpenAI–Gemini 84%, OpenAI–xAI 79%, Gemini–xAI 74% |
| Judge panel majority verdict | 2-of-3 pass — the number a release gate would use | 5 / 19 pass; 13 unanimous, 6 split (2–1) |

**Cost:** ~$0.50 (Anthropic, cached) + $0.27 for all three judges combined (OpenAI
$0.23, Gemini $0.03, xAI $0.006).

**\* The "100%" caveat.** That's on the validation set the confidence threshold was
tuned against — not on the held-out eval set, whose labels I never see. On the eval
set I can't score category directly, but I can see the system behaving the way it
should for harder traffic: escalation rate rises 33% → 47%, share of tickets it
drafts falls 70% → 53%, and the lowest confidence it reports drops 0.75 → 0.60. The
eval tickets are measurably more sensitive, and the system gets more cautious.

## Where it fails (with ticket IDs)

- **Fail-closed refusals: `t_train_004`, `t_train_019` (the 2/17).** In both, the
  audit stage rejected a draft rather than let it assert something it couldn't
  verify (a specific settlement window; an IRS 1099 rule). The count is
  non-deterministic 0–2 across runs — rejected drafts regenerate on Sonnet, which
  has no temperature knob — and the disk cache pins this particular run at 2. The
  deterministic gate itself never false-declines.
- **Guardrail arc (resolved).** The draft audit initially missed three drafts that
  invented facts (`t_train_022`, `t_eval_008`, the `t_eval_012` / `t_eval_002`
  class). I added a facts-list criterion; canaries in both directions now pass (3/3
  violation drafts caught, 1/1 clean draft not falsely rejected). Residual
  over-flagging on routing language is deliberate — the fail-closed regen path
  absorbs it.
- **Urgency over-calls: `t_train_011`, `t_train_016`.** The classifier is
  confidently wrong on these, so no escalation threshold catches them. The cascade
  fixes category, not urgency.
- **Draft quality: one root cause, two symptoms.** The three judges flag 8–13 drafts
  as inconsistent with the gold notes (OpenAI 8, Gemini 9, xAI 13 — xAI is the
  strictest). Of the 8 OpenAI flags: ~5 are the system correctly deferring on
  product facts no knowledge base can verify, 2 are a deliberate policy difference
  (gold commits to "within X hours"; we never promise timeframes), 1 is a real
  omission. That missing-KB gap also causes the fail-closed refusals above, so
  retrieval fixes both — hence next-week #1.

## What I'd build next week

1. **Knowledge-base retrieval.** Index Novig's real policies (fees, minimums, ACH
   timing, settlement windows, KYC steps). Let the drafter cite them and let the
   audit stage verify citations are grounded. Converts today's deferrals and
   fail-closed refusals into real answers — biggest single quality lever.
2. **Extend the judge panel.** Three labs are running now; splits cluster on the
   subjective criteria (`consistent_with_gold`, `no_unverifiable_promise`) — exactly
   where a human reviewer would also hesitate, which is the right signal. Next:
   weight each judge by canary sensitivity, add a fourth for tie-breaking stability,
   and gate releases on the panel majority rather than any single judge.
3. **Threshold auto-tuning.** The escalation threshold is a manual sweep today;
   re-run it as a CI step so it re-optimizes as labeled traffic grows.
4. **Real context management.** Currently: fixed 8-shot examples, a rough
   character-count token estimate, one ticket at a time. Production needs a real
   tokenizer, per-ticket example retrieval, summarized (not truncated) thread
   history, and provider-side prompt caching on the shared system prefix.
5. **Production plumbing.** Webhook/email ingestion in front of `predict()`; drift
   alerts on the distribution signals above (escalation rate, draft rate,
   confidence floor).

## Definitions and caveats

- **Confidence** = the probability a support lead reviewing the decision would not
  overturn it. Hard-rule refusals report 1.0 because the *refusal* is certain even
  when the category isn't; disagreement between the two cascade tiers caps it lower.
- **Reproducibility.** Haiku runs at temperature 0 (fully deterministic). Sonnet
  doesn't accept a temperature parameter, so its drafts vary run to run; the
  committed cache pins them. On a fresh clone with no API key, `make eval` replays
  the cache and reproduces `predictions.jsonl` byte-for-byte.
- **Hand-annotated flags.** The dataset has no gold labels for risk flags, so the
  few-shot examples' flags are my annotations. Any metric about flag agreement
  inherits that judgment.
- **Category accuracy understates safety.** `t_train_012` was miscategorized yet
  still correctly refused — its risk flag fired independently. Flags and category
  are decoupled, so a classification miss cannot become a safety miss.
- **Time.** I exceeded the suggested 4–6 hour budget. Deliberate over-investment in
  eval rigor and sensitive-case handling — the two highest-weighted rubric criteria
  — rather than breadth. A trade I'd defend; the hours I'd own.
