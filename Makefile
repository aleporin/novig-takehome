# Task entry points. `make help` lists them. No editable install: src/ goes on the
# path via the PYTHONPATH export below (and a bootstrap in tests/conftest.py).
PYTHON ?= python3
export PYTHONPATH := src

# Ruff config lives here as flags rather than a ruff.toml.
RUFF_SELECT := E,F,I,UP,SIM
RUFF_FLAGS := --line-length 100 --target-version py313

.PHONY: help install test smoke lint format eval train-metrics eval-train predict show-prompt

help:
	@echo "install       install pinned dependencies"
	@echo "test          run the offline suite (no API key)"
	@echo "smoke         run live API smoke tests (needs ANTHROPIC_API_KEY)"
	@echo "lint          ruff check + format --check"
	@echo "format        ruff format + autofix"
	@echo "eval          run the eval set -> predictions.jsonl   [lands with the pipeline]"
	@echo "train-metrics metrics report on labeled train data     [lands with the harness]"
	@echo "eval-train    validation-pool run + full metrics        [lands with the harness]"
	@echo "predict       one ticket end-to-end decision trace      [lands with the pipeline]"
	@echo "show-prompt   print an assembled prompt, no API call    [lands with the assembler]"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -m "not live"

smoke:
	$(PYTHON) -m pytest -m live

lint:
	$(PYTHON) -m ruff check --select $(RUFF_SELECT) $(RUFF_FLAGS) src tests
	$(PYTHON) -m ruff format --check --line-length 100 src tests

format:
	$(PYTHON) -m ruff format --line-length 100 src tests
	$(PYTHON) -m ruff check --fix --select $(RUFF_SELECT) $(RUFF_FLAGS) src tests

eval train-metrics eval-train predict show-prompt:
	@echo "'$@' is not wired yet — it arrives with a later branch."
	@exit 1
