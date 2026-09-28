# ============================================================
# Data-to-Decision Accelerator — Task Runner
# Run `make help` to list all targets.
# Windows tip: use Git Bash + mingw32-make, or just use
#              `python run_pipeline.py` (same pipeline, no make).
# ============================================================

VENV := .venv
ifeq ($(OS), Windows_NT)
    VENV_BIN := $(VENV)/Scripts
else
    VENV_BIN := $(VENV)/bin
endif
PY := $(VENV_BIN)/python

.DEFAULT_GOAL := help

# ---------- Setup ----------
.PHONY: setup
setup: ## Create virtualenv + install pinned dependencies
    python -m venv $(VENV)
    $(PY) -m pip install --upgrade pip
    $(PY) -m pip install -r requirements.txt
    $(PY) -m pip install -r requirements-dev.txt
    cp -n .env.example .env 2>/dev/null || true

# ---------- Pipeline stages ----------
.PHONY: generate
generate: ## Step 1: Generate synthetic raw dataset -> data/raw_sales.csv
    $(PY) -m src.data_generation

.PHONY: process
process: ## Step 2: Clean + engineer features -> data/processed_sales.parquet
    $(PY) -m src.data_processing

.PHONY: analyze
analyze: ## Step 3: KPIs, insights, executive memo
    $(PY) -m src.analysis

.PHONY: charts
charts: ## Step 4: Export decision-oriented charts -> insights/charts/
    $(PY) -m src.visualize

.PHONY: all
all: generate process analyze charts ## Full pipeline: data -> memo (one command)
    @echo ""
    @echo "Pipeline complete."
    @echo "  Memo:    insights/executive_memo.md"
    @echo "  Charts:  insights/charts/"
    @echo "  Next:    make run   (launch dashboard)"

# ---------- App ----------
.PHONY: run
run: ## Launch the Streamlit dashboard
    $(PY) -m streamlit run app.py

# ---------- Quality ----------
.PHONY: lint
lint: ## ruff (lint) + black --check (formatting)
    $(PY) -m ruff check src tests app.py run_pipeline.py
    $(PY) -m black --check src tests app.py run_pipeline.py

.PHONY: format
format: ## Auto-format with black + auto-fix ruff issues
    $(PY) -m black src tests app.py run_pipeline.py
    $(PY) -m ruff check --fix src tests app.py run_pipeline.py

.PHONY: test
test: ## Run pytest with coverage report
    $(PY) -m pytest tests/ -v --cov=src --cov-report=term-missing

# ---------- Housekeeping ----------
.PHONY: clean
clean: ## Remove generated artifacts (data, charts, logs) — keeps code
    rm -f data/raw_sales.csv data/processed_sales.parquet
    rm -rf insights/charts
    rm -rf logs
    rm -rf .pytest_cache .ruff_cache .coverage

.PHONY: clean-all
clean-all: clean ## Also remove the virtualenv
    rm -rf $(VENV)

.PHONY: help
help: ## Show this help
    @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
        awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
