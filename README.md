# Support Triage System

Classifies support tickets, decides whether to auto-draft a reply or defer to a
human, and drafts the safe ones. Sensitive tickets (account compromise, problem
gambling, legal, minors, self-harm, active fraud) are **never** auto-drafted —
enforcement is deterministic and in code; quality decisions are model-driven.

See **[WRITEUP.md](WRITEUP.md)** for approach, results, failure modes, and next steps.

## Quickstart

```sh
make install                         # pinned deps into your env

make test          # offline suite, no API key
make eval          # eval set -> predictions.jsonl (reproduces from the committed cache, no key)
make train-metrics # classifier metrics on the labeled train set
make coverage      # 100% branch coverage on the safety files
```

The response cache is committed, so `make eval` byte-reproduces `predictions.jsonl`
from a fresh clone with no API key. To run new tickets (cache misses), add a key:
`cp secrets.env.example secrets.env` and set `ANTHROPIC_API_KEY` (`OPENAI_API_KEY`
is optional, judge only).

## Layout

| Path | Purpose |
|---|---|
| `src/triage/` | Pipeline: prescreen → classify (routed) → gate → policy → draft → guard → emit |
| `src/evals/` | Eval harness: metrics, judge, batch runner; CLI entry points in `evals/cli/` |
| `skills/` | All prompt text, versioned as Markdown (never hardcoded in Python) |
| `tests/` | Offline test suite (safety-critical paths run with no API key) |
| `data/` | Provided datasets and taxonomy (untouched) |
| `results/` | Committed eval-run records (iteration history) |

Where a decision is made always has a one-file answer: safety → `stages/gate.py`,
model choice → `routing/router.py`, prompt content → `skills/`, settings → `config.py`.
