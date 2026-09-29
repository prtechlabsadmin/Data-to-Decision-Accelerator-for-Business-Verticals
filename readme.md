# Data-to-Decision Accelerator — E-commerce Sales Performance & Action Dashboard

*From raw transactions to a boardroom-ready decision — in one command.*

---

## 1. Description

An end-to-end analytics pipeline and interactive dashboard that converts raw e-commerce sales data into executive-ready insights and concrete weekly commercial actions. Built specifically for aspiring data practitioners to demonstrate production-grade skills, and for e-commerce leadership drowning in numbers but starved of decisions.

It answers the three questions leadership actually asks:
* 💸 **Which products/categories are quietly bleeding margin due to high returns?**
* 🗺️ **Where are the regional or seasonal drop-offs?**
* 🎯 **What specific actions should the team take this week to reverse underperformance?**

---

## 2. Flow Chart / Architecture

### Pipeline Flow

<img width="877" height="456" alt="image" src="https://github.com/user-attachments/assets/21c83404-a228-4db0-8e3d-52569369491b" />

### Stage Details

| Stage | Implementation | Artifact |
| :--- | :--- | :--- |
| **Data source** | Seeded synthetic generator (deterministic, `seed=42`) | `data/raw_sales.csv` |
| **Ingestion** | CSV load + schema/type validation, realistic dirty rows injected | in-memory DataFrame |
| **Processing** | Cleaning, feature engineering (net revenue, margin, return rate), RFM-lite customer segmentation | `data/processed_sales.parquet` |
| **Model / engine** | No ML model — a rule-based insight engine ranks findings by business impact (margin at risk) | ranked insight list |
| **Evaluation** | `pytest` unit tests, data-quality assertions, gross-vs-net revenue reconciliation | test report |
| **Deployment** | Streamlit dashboard + one-page executive memo | `app.py`, `insights/executive_memo.md` |
| **Monitoring** | One-command pipeline re-run; freshness & distribution-drift checks on return rates | logs + refreshed dashboard |



## 3. Function List

### `src/data_generation.py`
* `generate_products(n_products)` — Build product catalogue with cost & price per SKU.
* `generate_customers(n_customers)` — Customer master with region assignment.
* `generate_orders(start, end, ...)` — Time-series orders with seasonality & return behaviour.
* `inject_data_issues(df)` — Inject realistic dirty rows (NaNs, dupes, negative qty).
* `save_raw_data(df, path)` — Write `data/raw_sales.csv`.

### `src/data_processing.py`
* `load_raw_data(path)` — Read CSV, parse dates, validate schema.
* `validate_schema(df)` — Raise `SchemaError` on missing/incorrect columns.
* `clean_data(df)` — Fix missing values, dupes, negatives, invalid dates (all logged).
* `engineer_features(df)` — Add `net_revenue`, `profit`, `margin`, `return_flag`, `return_rate`.
* `segment_customers(df)` — RFM-lite tags: Champion / Loyal / At-Risk / New.
* `save_processed(df, path)` — Write partitioned Parquet.

### `src/analysis.py`
* `compute_kpis(df)` — Net revenue, AOV, overall return rate, avg margin.
* `monthly_trend(df)` — Month-over-month revenue & margin trend.
* `category_performance(df)` — Revenue / margin / return rate per category.
* `regional_performance(df)` — Regional breakdown to spot drop-offs.
* `top_return_drivers(df, top_n)` — Rank SKUs by margin lost to returns.
* `detect_seasonal_drops(df)` — Flag months with anomalous declines.
* `rank_insights(...)` — Score & rank the top 3–5 actionable insights.
* `generate_executive_memo(insights, path)` — Write the one-page leadership memo.

### `src/visualize.py`
* `plot_revenue_trend(df)` — Net revenue over time (with return overlay).
* `plot_category_margin(df)` — Margin vs return rate by category.
* `plot_return_rates(df)` — Return-rate Pareto — the "margin bleed" chart.
* `plot_regional_performance(df)` — Regional revenue heatmap.
* `plot_customer_segments(df)` — Segment mix & value.
* `export_charts(df, out_dir)` — Save all charts as PNGs to `insights/charts/`.

### `app.py`
* `main()` — Streamlit entrypoint & page layout.
* `get_filters(df)` — Sidebar slicers (date range, region, category).
* `render_kpi_cards(kpis)` — KPI header row.
* `render_charts(df)` — Decision-oriented Plotly charts.
* `render_action_plan(insights)` — "What to do this week" recommendation panel.

---

## 4. Code Hygiene

* **Type hints:** Every public function fully typed (Python 3.10+ syntax):
  ```python
  def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
  ```
* **Unit tests:** `pytest` suite in `tests/` covering generation determinism, cleaning correctness, KPI math, and empty-input safety:
  ```bash
  pytest tests/ -v --cov=src --cov-report=term-missing
  ```
* **Linting:** `ruff` (style + lint) and `black` (formatting), wired into a Makefile target: `make lint`.
* **Config files:** All magic numbers live in `config.yaml` (thresholds like `high_return_threshold: 0.15`, seed, row counts, paths).
* **Environment variables:** `.env` + `python-dotenv` for `DATA_DIR`, `SEED`, `LOG_LEVEL`; `.env.example` committed, `.env` git-ignored.
* **Logging:** Standard `logging` module, per-stage INFO/WARN to console and `logs/pipeline.log`:
  ```python
  logger = logging.getLogger(__name__)
  logger.info("Cleaning %d rows | fixed %d duplicates", len(df), n_dupes)
  ```
* **Reproducible setup:** Fixed random seed (deterministic dataset), pinned `requirements.txt`, and a single command (`make all`) that regenerates the entire pipeline from zero. Anyone can clone → run → get identical results.

---

## 5. UI

Home page — KPI cards (net revenue, return rate, margin), filter sidebar
<img width="1820" height="801" alt="image" src="https://github.com/user-attachments/assets/dd13f1fa-e1a9-43b2-abf3-a406daa0831e" />

Input/output demo — raw CSV → processed parquet → margin-bleed chart 
<img width="1601" height="751" alt="image" src="https://github.com/user-attachments/assets/69cecd48-d4ab-4c7c-8fe1-fe0236d0a6d3" />


Evaluation dashboard — `pytest` coverage output + data-quality assertions 
<img width="1599" height="846" alt="image" src="https://github.com/user-attachments/assets/29fef99a-0ba4-4f62-8315-82d0cd437c07" />


Category Scorecard
<img width="1550" height="285" alt="image" src="https://github.com/user-attachments/assets/1fd7e3ad-ad8a-4487-ab0c-a47aa40618b5" />


---

## 6. Results

⚠️ *Numbers below are from a sample seeded run (`seed=42`, 50,000 orders, 12 months). Replace them with your own run's output for authenticity.*

| Metric | Baseline (raw data) | Final (after pipeline) |
| :--- | :--- | :--- |
| **Data quality** | 3.2% dirty rows (NaNs, dupes, negatives) | 0% — all fixed or safely logged |
| **Gross revenue** | $2.41M tracked & reconciled ✅ | — |
| **Net revenue (after returns)** | ❌ not visible | **$2.18M** |
| **Return rate visibility** | ❌ none | **9.4% overall; worst category 19.8%** |
| **Margin at risk (top 5 return-heavy SKUs)** | ❌ invisible | **$46.2K quantified & ranked** |
| **Ranked weekly actions for leadership** | 0 | **5 concrete recommendations + memo** |

* **Latency:** Full pipeline (generate → memo): ~55 s for 50K rows on a laptop; dashboard cold start ~1.8 s; filter interactions < 0.5 s.
* **Cost:** $0 locally; deploys to the free tier of Streamlit Cloud.
* **Failure cases handled:**
  * Missing / malformed dates → coerced; unparseable rows dropped & logged.
  * Negative quantities & prices → flagged and cleaned.
  * Duplicate order IDs → deduplicated (logged count).
  * Division-by-zero on return rate for low-volume categories → min-volume guard.
  * Empty dashboard filter selections → graceful "no data" state, no crash.
  * Missing/corrupt raw CSV → clear error message with regeneration hint.

---

## 7. How to Run

### Setup
* Python 3.10+, `git`, and (optionally) `make`

### Install
```bash
git clone https://github.com/<your-username>/data-to-decision-accelerator.git
cd data-to-decision-accelerator
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Run locally (one command → full pipeline)
```bash
python run_pipeline.py           # or: make all
# generates data → cleans → analyzes → charts → executive memo
streamlit run app.py             # dashboard at http://localhost:8501
```

### Run tests
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Lint
```bash
ruff check src tests app.py && black --check src tests app.py
```

### Deploy
* Easily deploys to the **Streamlit Cloud** free tier!
