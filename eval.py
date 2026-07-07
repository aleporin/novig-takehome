"""Single-command entry point: `python eval.py` -> predictions.jsonl.

The OS-agnostic way to run the pipeline over the eval set — no make, no PYTHONPATH
setup, no API key (it reproduces predictions.jsonl from the committed cache). Equivalent
to `make eval`. Add keys to secrets.env only to regenerate against new tickets.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from evals.cli.run_eval import main  # noqa: E402  (sys.path setup must precede this import)

if __name__ == "__main__":
    main()
