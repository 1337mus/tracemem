PYTHON ?= python3
export PYTHONPATH := src

.PHONY: demo benchmark test lint typecheck check
.PHONY: audit-data

demo:
	$(PYTHON) -m tracemem demo

benchmark:
	$(PYTHON) -m tracemem benchmark

test:
	$(PYTHON) -m unittest discover -s tests -v

lint:
	.venv/bin/ruff check src tests experiments training
	.venv/bin/ruff format --check src tests experiments training

audit-data: ; $(PYTHON) -m experiments.audit_dataset

typecheck:
	.venv/bin/mypy

check: test lint typecheck
