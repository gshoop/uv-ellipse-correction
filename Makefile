# Makefile for common development tasks
#
# Tools run from ./venv when it exists (no need to activate it); otherwise from
# whatever python3 is on PATH (e.g. an activated venv). Override with
# `make VENV=/path/to/venv <target>`.

VENV ?= venv
ADC2KEV ?= $(HOME)/adc2kev-python
SYSTEM_PYTHON ?= python3.10

# Recursively expanded (=) so the lookup happens when a recipe runs: after
# `make venv` in the same invocation, later targets already use ./venv.
PYTHON = $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PIP = $(PYTHON) -m pip
# pip installs must go into a virtualenv, never the system/user site-packages.
REQUIRE_VENV = @$(PYTHON) -c "import sys; sys.exit(sys.prefix == sys.base_prefix)" || \
	{ echo "No virtualenv: run 'make venv' first or activate one"; exit 1; }

.PHONY: help venv install install-dev cython-check test test-fast test-realdata test-cov \
	lint format format-check type-check clean dev-check check

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

venv:  ## Create ./venv (python3.10) with the build tools (pip, setuptools, wheel, Cython)
	$(SYSTEM_PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip setuptools wheel Cython

install:  ## Install adc2kev (editable, from ADC2KEV=~/adc2kev-python) and uvcorr
	$(REQUIRE_VENV)
	$(PIP) install -e $(ADC2KEV)
	$(PIP) install -e .
	$(MAKE) cython-check

install-dev:  ## Install adc2kev (editable) and uvcorr with dev dependencies
	$(REQUIRE_VENV)
	$(PIP) install -e $(ADC2KEV)
	$(PIP) install -e ".[dev]"
	$(MAKE) cython-check

cython-check:  ## Check that adc2kev's Cython parser extension is available
	$(PYTHON) -c "import adc2kev.parser.packet_parser as p; \
	assert p.CYTHON_AVAILABLE and p.CYTHON_ARRAYS_AVAILABLE, \
	'adc2kev Cython parser missing: run python setup.py build_ext --inplace in $(ADC2KEV)'; \
	print('adc2kev Cython parser: OK')"

test:  ## Run tests (skip tests needing the real test data)
	$(PYTHON) -m pytest -m "not realdata"

test-fast:  ## Run tests (skip slow and real-data tests)
	$(PYTHON) -m pytest -m "not slow and not realdata"

test-realdata:  ## Run only the real-data tests (need the test .dat file or its cache)
	$(PYTHON) -m pytest -m realdata

test-cov:  ## Run tests with coverage (skip real-data tests)
	$(PYTHON) -m pytest -m "not realdata" --cov=uvcorr --cov-report=term-missing --cov-report=html --cov-report=xml

lint:  ## Run linter (ruff)
	$(PYTHON) -m ruff check src/ tests/ scripts/

format:  ## Format code with black
	$(PYTHON) -m black src/ tests/ scripts/

format-check:  ## Check code formatting
	$(PYTHON) -m black --check src/ tests/ scripts/

type-check:  ## Run type checker (mypy)
	$(PYTHON) -m mypy src/ scripts/

clean:  ## Clean build artifacts and tool caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info src/*.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage coverage.xml
	find . -path ./$(VENV) -prune -o -type d -name __pycache__ -exec rm -rf {} +
	find . -path ./$(VENV) -prune -o -type f -name "*.pyc" -exec rm -f {} +

# Development workflow
dev-check: format lint type-check test  ## Run all checks (format, lint, type-check, test)

check: format-check lint type-check test  ## Run all checks without modifying files (phase gate)
