# Support Triage System

Drafts responses for routine support tickets while guaranteeing that sensitive
tickets (account compromise, problem gambling, legal, minors, self-harm, active
fraud) are never auto-drafted and are always routed to a human. Safety decisions
are deterministic and enforced in code; quality decisions are model-driven.

> Status: scaffolding. Implementation lands in subsequent branches.

## Layout

| Path | Purpose |
|---|---|
| `src/triage/` | Pipeline: ingest → prescreen → classify → gate → policy → draft → guard → emit |
| `src/evals/` | Eval harness: metrics, judge, run reporter |
| `skills/` | All prompt text, versioned as Markdown (never hardcoded in Python) |
| `tests/` | Offline test suite (safety-critical paths run with no API key) |
| `data/` | Provided datasets and taxonomy (untouched) |
| `results/` | Committed eval-run records |

## Setup

```sh
cp secrets.env.example secrets.env   # then add your ANTHROPIC_API_KEY
```

Where a decision is made always has a one-file answer: safety → `gate.py`,
model choice → `router.py`, prompt content → `skills/`, thresholds → `config.py`.
