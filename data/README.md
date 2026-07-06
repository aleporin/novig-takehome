# Novig — AI Automation Take-Home: Support Triage System

Welcome. The full prompt is in `THM-002.pdf` (or `THM-002.tex`). This README covers what's in the package and how to get started.

## Files

| File | Description |
|---|---|
| `THM-002.pdf` | The full assessment prompt, constraints, and evaluation criteria. **Read this first.** |
| `taxonomy.md` | Category definitions, urgency levels, sensitive-case rules, and the exact output schema. |
| `tickets_train.jsonl` | 30 labeled tickets. Use however you like — few-shot examples, validation set, prompt iteration, etc. |
| `tickets_eval.jsonl` | 15 unlabeled tickets. Your system will be graded on its predictions for these. |

## What we expect in your repo

1. **Source code** — your triage system in Python or TypeScript.
2. **A one-command eval runner** — e.g., `make eval` or `python eval.py` that runs your system over `tickets_eval.jsonl` and writes `predictions.jsonl` in the schema defined in `taxonomy.md`.
3. **Reported metrics** — output from your eval harness on the training set (since you don't have eval labels). At minimum: category accuracy, urgency accuracy, urgency confusion matrix, and a false-draft-on-sensitive metric. How you compute these on the unlabeled eval is up to you (e.g., self-consistency across runs, drift from training set distribution, etc.) — surface whatever you can.
4. **`predictions.jsonl`** — your system's outputs on the eval set.
5. **`WRITEUP.md`** — max one page. Approach, eval results, failure modes, next steps.

## Setup

We're not prescribing a stack. Use what you're fast in.

You'll need an LLM API key. We'll reimburse up to $20 against a screenshot of your usage dashboard.

## Time

72 hours from receipt to submit your repo URL. We recommend 4–6 hours of actual work. If you're past six hours, ship what you have and tell us what you'd do next in the writeup.

## Questions

If something in the prompt is genuinely ambiguous and is blocking you, email the recruiter. We'd rather you ask than guess wrong on something we didn't intend to be a test of guessing.

Good luck.
