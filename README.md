# Support Triage System

Given a support ticket, this classifies it (category + urgency), decides whether to
auto-draft a reply or defer to a human, drafts the safe ones, and ships an eval harness
that measures all of it. Two commitments shape the design: **unconditional enforcement
over probabilistic detection** — deciding *whether* a ticket is sensitive is a model
judgment that can miss, but *once* flagged it is never drafted, by a pure-Python gate
with no model in the path; and **evals first** — the harness was built before the
classifier and iterated against, so the numbers here are measured, not asserted.

## Quickstart

```sh
git clone https://github.com/aleporin/novig-takehome.git && cd novig-takehome
make install       # pinned deps into your env
make eval          # run the pipeline over the eval set -> predictions.jsonl
make test          # 860 offline tests, no API key
```

`make eval` reproduces `predictions.jsonl` **byte-for-byte with no API keys** — every
model response is in the committed cache (`.cache/`), and the client only reaches for a
key on a cache miss. To regenerate against new tickets, add keys:
`cp secrets.env.example secrets.env` and set `ANTHROPIC_API_KEY` (`OPENAI_API_KEY` is
optional, for the eval judge only).

## Repo map

| Directory | What lives there |
|---|---|
| `src/triage/` | The pipeline: prescreen → classify (routed) → gate → policy → draft → guard → emit |
| `src/evals/` | Eval harness: metrics, bootstrap CIs, judge, runners (`evals/cli/`) |
| `skills/` | Every prompt, as versioned Markdown — never hardcoded in Python |
| `tests/` | 860 offline tests; safety paths run with no API key |
| `data/` | Provided tickets + taxonomy, untouched (Novig's original README is here) |
| `results/` | Committed record of every eval run — the iteration history |
| `.cache/` | Committed model responses — what makes `make eval` key-free |

One-file answers:
- **Why was this ticket declined?** → `stages/gate.py` (hard rules), `stages/draft_policy.py` (soft rules).
- **What did the model actually see?** → `skills/` + `make show-prompt TICKET=<id>`.
- **What did run N score?** → `results/<timestamp>_*/metrics.json`.

## Commands

| Command | What it does |
|---|---|
| `make eval` | Run the eval set → `predictions.jsonl` (key-free, from cache) |
| `make test` | 860 offline tests, no API key |
| `make coverage` | Enforce 100% branch coverage on the safety files |
| `make train-metrics` | Classifier metrics on the labeled train pool (accuracy, confusion, cost) |
| `make predict TICKET=<id>` | Trace one ticket end-to-end (prescreen → gate → decision) |
| `make show-prompt TICKET=<id>` | Print the assembled prompt, no API call |
| `make threshold-sweep` | The escalation-threshold tuning evidence |
| `make audit-canaries` | Prove the draft auditor catches invented facts without over-flagging |

## Writeup

**[WRITEUP.md](WRITEUP.md)** — approach and key decisions, eval results with confidence
intervals, and where the system fails and why.
