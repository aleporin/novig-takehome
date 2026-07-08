# Support Triage Writeup

The system takes an incoming support ticket, sorts it (category and urgency), decides
whether an AI can safely draft the reply, writes the draft when it's safe, and hands the
ticket to a human when it isn't. Safety is handled by hard rules in Python, not by an
LLM. Classification and drafting use LLMs.

## How the system works

Every ticket goes through the same six steps:

1. **Scan for danger words** (self-harm, hacking, legal threats, problem gambling).
2. **Classify** the ticket (which category, how urgent).
3. **Decide**: draft an AI reply, or defer to a human.
4. **Write** the draft (only if step 3 said yes).
5. **Audit** the draft (does it invent facts, promise things, run too long?).
6. **Return** either the draft or the reason for deferring.

The whole thing sits behind one function call: `predict(ticket)`.

Two rules shaped every design choice:

**Safety is enforced, not guessed at.**
If a ticket looks sensitive (account compromise, self-harm, legal threat, problem
gambling), the system refuses to draft. That refusal is written in plain Python. No LLM
is in that decision. It is tested against every category and every flag combination at
100% branch coverage.

Deciding *whether* a ticket is sensitive is done by two independent checks: a keyword
scan and the LLM classifier. If both miss, the ticket slips through. If either catches
it, the refusal fires. The claim is "if we catch it, we always refuse." Not "we always
catch it."

**Evals were built first.**
The evaluation code existed before the classifier did. Every design change had to prove
itself against the numbers. Every eval run is committed under `results/`, and every run
can be reproduced exactly from a saved cache of past LLM responses.

## Key design decisions

**A two-tier cascade: Haiku, then Sonnet.**
Claude Haiku is cheap and fast, so it handles most tickets. If Haiku isn't confident, or
the ticket looks sensitive, Claude Sonnet takes over. Sonnet can only add protection. A
safety flag Haiku raised can never be undone by Sonnet. That property is verified with a
property-based test.

**Three independent judges score the drafts.**
The drafts are written by Claude (Anthropic). To grade the drafts, we use three LLMs
from three other labs: OpenAI GPT-5, Google Gemini 2.5 Pro, and xAI Grok. Using
different labs than the drafter is deliberate. No lab is grading its own work. Their
agreement, and disagreement, is real signal.

Each judge has to pass a warm-up first. We show it three drafts we know are bad
(planted problems: a fake fee, a promised date, a generic reply with no specifics). If a
judge misses any of them, its scores are thrown out. If a judge fails mid-run (rate
limit, outage), we drop it and finish with whatever judges are still working.

**No knowledge base.**
The takehome came without a Novig knowledge base. Instead of guessing at product facts,
the drafter is instructed to defer whenever it can't verify something from the ticket
text itself.

## What the evals showed

Ran on 22 validation tickets (30 labeled, 8 held back as few-shot examples). Ranges in
brackets are 95% bootstrap confidence intervals.

| Metric | What it measures | Value |
|---|---|---|
| Category accuracy | Did we pick the right category? | **100%** [100, 100] \* |
| Urgency accuracy | Did we pick the right urgency level? | 82% [64, 95] |
| Severity-weighted urgency error | Urgency mistakes weighted by cost. Under-calling hurts most. | 0.18 (baseline that always picks the most common answer: 2.8) |
| **False draft on sensitive tickets** | **The worst possible error: drafted when a hard rule said don't** | **0 / 5** (zero across every labeled ticket) |
| False decline | Refused to draft when we should have | 2 / 17. Varies run to run. See notes below. |
| Escalation rate | How often Haiku hands the ticket to Sonnet | 33% (target band 15% to 35%) |
| Judge warm-up (canaries) | Planted-flaw drafts each judge must catch before we trust their scores | 3 / 3 for all three judges |
| Judge pairwise agreement | How often two judges give the same overall pass/fail | OpenAI vs Gemini 84%. OpenAI vs xAI 79%. Gemini vs xAI 74%. |
| Judge panel majority | 2 of 3 judges say pass. The number a release gate would use. | 5 / 19 pass. 13 unanimous, 6 split (2 to 1). |

**Cost.** About $0.50 for the Anthropic calls (mostly served from cache), plus $0.27
across all three judges combined (OpenAI $0.23, Gemini $0.03, xAI $0.006).

**\* About the 100% category accuracy.**
That is on the validation set, and the confidence threshold was tuned against that same
set. The eval set is held out (labels are hidden from us), so we can't score category on
it directly. But we can see the system behaving the way it should on the harder eval
traffic: the escalation rate rises from 33% to 47%, the share of tickets the system
drafts falls from 70% to 53%, and the lowest confidence value it reports drops from 0.75
to 0.60. The eval tickets are measurably more sensitive than the training tickets, and
the system correctly gets more cautious.

## Where it fails (with ticket IDs)

**Fail-closed refusals: `t_train_004`, `t_train_019` (the 2/17).**
On these two tickets, the system was allowed to draft but decided at the last second not
to. The draft was about to make a specific claim we couldn't verify from the ticket (a
settlement window on one, an IRS 1099 rule on the other). Rather than let a wrong claim
reach a customer, the audit step killed the draft and sent the ticket to a human.

The exact count varies from run to run (sometimes 0, sometimes 1, sometimes 2). That is
because Sonnet doesn't accept a temperature setting, so its exact wording changes. The
safety gate itself (the deterministic Python one) never falsely refuses. Only the softer
draft audit occasionally over-refuses.

**Guardrail arc (now fixed).**
The draft audit used to have a hole. It let three drafts through that were making up
facts (`t_train_022`, `t_eval_008`, and the `t_eval_012` / `t_eval_002` class). I fixed
it by adding a "facts list" check: any specific claim in the draft has to come from the
ticket or the gold notes. I proved the fix works both ways. Three planted fake bad
drafts, all three caught. One clean draft that mentions the customer's own numbers, not
falsely rejected. The fix is slightly too strict on innocuous phrases like "we'll route
this to our payments team." I left that alone because being over-strict is safe here,
and the regeneration path handles the false alarms.

**Urgency over-calls: `t_train_011`, `t_train_016`.**
On these two, the classifier is confidently wrong about how urgent the ticket is. It
says "I'm very sure this is high urgency" when it's actually medium. Because the model
is confident, the "escalate to Sonnet if you're not sure" safety net doesn't fire. The
cascade fixes category mistakes, not urgency mistakes made with high confidence.

**Draft quality: one problem causes two symptoms.**
The three judges flagged some drafts as not matching the ideal answer (OpenAI 8,
Gemini 9, xAI 13). xAI is the strictest reader. Looking at the 8 that OpenAI flagged:

- About 5 are the system being correct. The ideal answer made up a specific fact
  (an exact fee, an exact date). We refused to make it up. The judge marks us as
  "different from the ideal," but we were right to be different.
- 2 are a policy difference on purpose. The ideal reply says "we'll respond within
  24 hours." We never promise timeframes on this kind of exchange. The judge marks
  us down, we accept it.
- 1 is a real omission we missed.

Both this and the fail-closed refusals above have the same root cause: there is no Novig
knowledge base to look things up in. So sometimes we correctly refuse to invent a fact,
and sometimes we defer when a knowledge base would have told us the answer. Fixing this
is the top item on the next-week list.

## What I'd build next week

1. **Add a knowledge base and retrieval.**
   Index Novig's real policies (fees, minimums, ACH timing, settlement windows, KYC
   steps). Let the drafter cite them. Let the audit step verify the citations are
   grounded. This single change fixes both the fail-closed refusals and the "not
   matching gold" flags above. Biggest single lever.
2. **Extend the judge panel.**
   Three labs are already running. The tickets where they split 2 to 1 cluster on the
   subjective judgments (matches the ideal reply, avoids unverified promises). That is
   the right signal: judges agree on the easy questions and disagree on the hard ones,
   which is exactly where a human reviewer would also hesitate. Next: weight each judge
   by how strict it was on the warm-up canaries, add a fourth judge for tie-breaking
   stability, and gate release on the panel majority rather than any single judge.
3. **Auto-tune the escalation threshold.**
   Today the threshold that decides when Haiku hands off to Sonnet is set by a manual
   sweep. Move it into CI so it re-tunes as more labeled tickets come in.
4. **Better prompt and context handling.**
   Right now we use a fixed set of 8 few-shot examples, estimate tokens by dividing
   character count by 4, and process one ticket at a time. Production would need a
   real tokenizer, per-ticket example retrieval (not a fixed set), thread history
   that's summarized instead of truncated, and provider-side caching on the shared
   system prefix.
5. **Production plumbing.**
   Add webhook or email ingestion in front of the `predict()` function, and set up
   drift alerts on the signals above (escalation rate, draft rate, lowest confidence).

## Notes and caveats

**Confidence** is the probability that a human support lead reviewing the decision would
agree with it. Hard-rule refusals report 1.0 because the refusal is certain even when
the exact category isn't. When the two cascade tiers disagree, confidence is capped
lower.

**Reproducibility.** Haiku runs at temperature 0 and is fully deterministic. Sonnet
doesn't accept a temperature parameter, so its drafts vary between runs. The committed
disk cache pins them so results replay exactly. On a fresh clone with no API key,
`make eval` reproduces `predictions.jsonl` byte for byte.

**Hand-annotated flags.** The dataset doesn't have gold labels for risk flags, so the
flags on the few-shot examples are my annotations. Any metric that involves flag
agreement inherits that judgment.

**Category accuracy understates safety.** On `t_train_012`, the classifier picked the
wrong category, but the ticket was still correctly refused because its risk flag fired
independently. Flags and categories are decoupled by design. A classification miss can
never become a safety miss.
