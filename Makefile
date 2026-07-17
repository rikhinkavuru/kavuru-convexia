# kavuru-convexia — reliability audit harness for LLM drug-asset verdicts
PY  := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help setup test demo clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-8s %s\n", $$1, $$2}'

setup:  ## Build a clean venv and install the package + dev deps
	python3.13 -m venv .venv || python3 -m venv .venv
	$(PIP) install --upgrade pip wheel setuptools
	$(PIP) install -e ".[dev]"
	@echo "setup complete — set ANTHROPIC_API_KEY, then run: make demo"

test:  ## Run the pytest suite (no network; uses offline stub evaluators)
	KAVURU_OFFLINE=1 $(PY) -m pytest

demo:  ## Run the full audit on bundled assets -> figures, report, notebook in outputs/
	$(PY) -m kavuru_convexia.notebook_builder

clean:  ## Remove generated outputs (keeps the cached LLM responses)
	rm -rf outputs/figures outputs/reports outputs/*.ipynb
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
