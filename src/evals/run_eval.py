"""Run the baseline over the eval set, write predictions.jsonl, and validate it.

The eval tickets are unlabeled, so there are no accuracy numbers here; this proves
the end-to-end path and that the output passes the contract validator.
"""

from __future__ import annotations

from triage.config import Config

from .baseline import MajorityBaseline
from .dataset import load_tickets
from .validate import validate_file


def main() -> None:
    config = Config()
    train = load_tickets(config.paths.tickets_train)
    eval_tickets = load_tickets(config.paths.tickets_eval)

    baseline = MajorityBaseline.from_labels(train)
    predictions = [baseline.predict(t) for t in eval_tickets]

    out = config.paths.predictions
    out.write_text("\n".join(p.to_jsonl_line() for p in predictions) + "\n", encoding="utf-8")

    problems = validate_file(out, config.paths.tickets_eval)
    if problems:
        raise SystemExit("predictions.jsonl failed validation:\n" + "\n".join(problems))
    print(f"wrote {out} ({len(predictions)} predictions), validation OK")


if __name__ == "__main__":
    main()
