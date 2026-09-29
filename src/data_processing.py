"""src/data_processing.py — Step 2: clean, verify, engineer, segment.

Pipeline position: consumes data/raw_sales.csv (from src.data_generation),
produces data/processed_sales.parquet for src.analysis / src.visualize / app.py.

Cleaning contract — the six issue types planted by Step 1 and how each is
handled (every fix counted, logged, and persisted to cleaning_report.json):

    planted issue            handling                            config knob
    ----------------------   ---------------------------------   -------------
    malformed order_date     strict ISO parse -> NaT -> dropped   (always)
    duplicate order_id       dedupe, keep first                  drop_duplicate_keys
    negative quantity        abs() + quantity_adjusted flag      fix_negative_quantities
    out-of-range unit_price  reset to NaN, then imputed          min/max_valid_price
    missing unit_price       impute (median / category_median /  missing_price_strategy
                             drop)
    missing customer_region  per-customer mode -> global mode    missing_region_strategy
                             fallback

STRUCTURAL missingness: return_reason is NaN exactly when returned is False
(a reason cannot exist for an order that never came back). That is not dirt.
This module preserves it and verifies it — see verify_cleaning.

Engineered features (money rounded to 2dp):
    return_flag, discounted_unit_price, gross_revenue, cogs, net_revenue,
    revenue_lost_to_returns, net_profit, margin_pct (NaN on returned rows:
    division-by-zero guard), sku_return_rate, order_year_month, month

Profit model assumption: returned units are written off — zero revenue AND
full COGS retained. Conservative, and it is precisely what makes the
"margin bleed" story visible downstream.

Artifacts:
    data/processed_sales.parquet   cleaned + enriched fact table (27 cols)
    data/cleaning_report.json      machine-readable cleaning audit

Module contract (see run_pipeline.py): main() -> Path (parquet artifact).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_config, get_logger, setup_logging

logger = get_logger(__name__)

__all__ = [
    "SchemaError",
    "DataQualityError",
    "load_raw_data",
    "validate_schema",
    "clean_data",
    "verify_cleaning",
    "engineer_features",
    "segment_customers",
    "save_processed",
    "main",
]


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class SchemaError(ValueError):
    """Raw data does not match the 13-column contract written by Step 1."""


class DataQualityError(RuntimeError):
    """Post-cleaning assertions failed — dirt survived the cleaning stage."""


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REQUIRED_COLUMNS: tuple[str, ...] = (
    "order_id", "order_date", "customer_id", "customer_region",
    "product_id", "product_name", "category", "quantity",
    "unit_price", "unit_cost", "discount_pct", "returned", "return_reason",
)

KEY_COLUMNS: tuple[str, ...] = ("order_id", "customer_id", "product_id")

#: Schema contract: Step 1 writes ISO dates. Strict format parsing is what
#: makes malformed-date handling deterministic — see the note in load_raw_data.
DATE_FORMAT = "%Y-%m-%d"

PRICE_STRATEGIES: tuple[str, ...] = ("median", "category_median", "drop")
REGION_STRATEGIES: tuple[str, ...] = ("mode", "drop")

UNKNOWN_RETURN_REASON = "Unknown"

TRUTHY: dict[Any, bool] = {
    True: True, False: False,  # note: 1/0 hash-collide with True/False — intended
    "True": True, "False": False, "true": True, "false": False,
    "TRUE": True, "FALSE": False, "1": True, "0": False,
}

MONEY_COLUMNS: tuple[str, ...] = (
    "discounted_unit_price", "gross_revenue", "cogs",
    "net_revenue", "revenue_lost_to_returns", "net_profit",
)

FINAL_COLUMN_ORDER: tuple[str, ...] = (
    # identity & time
    "order_id", "order_date", "order_year_month", "month",
    # who
    "customer_id", "customer_region", "customer_segment",
    # what
    "product_id", "product_name", "category",
    # transaction economics (+ audit flags)
    "quantity", "quantity_adjusted", "unit_price", "unit_price_imputed",
    "unit_cost", "discount_pct", "discounted_unit_price",
    # returns
    "returned", "return_flag", "return_reason",
    # money
    "gross_revenue", "net_revenue", "cogs", "revenue_lost_to_returns",
    "net_profit", "margin_pct", "sku_return_rate",
)


# --------------------------------------------------------------------------- #
# Ingestion & schema validation
# --------------------------------------------------------------------------- #

def validate_schema(df: pd.DataFrame) -> None:
    """Raise SchemaError if the raw frame breaks the 13-column contract."""
    if df.empty:
        raise SchemaError(
            "raw data contains 0 rows — regenerate it with: "
            "python run_pipeline.py --stages generate")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"raw data is missing required column(s): {', '.join(missing)} | "
            f"expected the 13-column schema written by src.data_generation "
            f"(was the CSV corrupted or truncated?)")
    extra = sorted(set(df.columns) - set(REQUIRED_COLUMNS))
    if extra:
        logger.warning("Ignoring unexpected extra column(s): %s", ", ".join(extra))


def _coerce_returned(df: pd.DataFrame) -> int:
    """Normalise the 'returned' column to bool; return count of unknown values."""
    col = df["returned"]
    if col.dtype == bool:
        return 0
    mapped = col.map(TRUTHY)
    unknown = int((mapped.isna() & col.notna()).sum())
    if unknown:
        logger.warning("%d unrecognized 'returned' value(s) coerced to False",
                       unknown)
    df["returned"] = mapped.fillna(False).astype(bool)
    return unknown


def load_raw_data(path: Path | str) -> pd.DataFrame:
    """Read the raw CSV, validate the schema, and coerce types.

    Dates are parsed STRICTLY against the ISO contract. Lenient parsing
    would silently accept '2024/31/12' as 2024-12-31 (dateutil resolves
    month=31 as day=31); strict ``format=`` turns every junk value into a
    deterministic NaT that clean_data then drops.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"raw data not found: {path} | generate it first: "
            f"python run_pipeline.py --stages generate")

    df = pd.read_csv(path)
    validate_schema(df)

    # Numeric coercion — anything unparseable becomes NaN for clean_data.
    for col in ("quantity", "unit_price", "unit_cost", "discount_pct"):
        before = int(df[col].isna().sum())
        df[col] = pd.to_numeric(df[col], errors="coerce")
        gained = int(df[col].isna().sum()) - before
        if gained:
            logger.warning("%d unparseable value(s) in '%s' coerced to NaN",
                           gained, col)

    df["order_date"] = pd.to_datetime(
        df["order_date"], format=DATE_FORMAT, errors="coerce")
    _coerce_returned(df)

    logger.info("Loaded raw data: %d rows x %d cols from %s (%.1f MB) | "
                "unparseable dates -> NaT: %d",
                len(df), df.shape[1], path, path.stat().st_size / 1e6,
                int(df["order_date"].isna().sum()))
    return df


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #

def _validate_cleaning_cfg(cfg: dict[str, Any]) -> None:
    required = ("drop_duplicate_keys", "fix_negative_quantities",
                "missing_price_strategy", "missing_region_strategy",
                "min_valid_price", "max_valid_price")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config.yaml 'cleaning' section is missing "
                         f"key(s): {', '.join(missing)}")
    if cfg["missing_price_strategy"] not in PRICE_STRATEGIES:
        raise ValueError(f"missing_price_strategy must be one of "
                         f"{PRICE_STRATEGIES}, got '{cfg['missing_price_strategy']}'")
    if cfg["missing_region_strategy"] not in REGION_STRATEGIES:
        raise ValueError(f"missing_region_strategy must be one of "
                         f"{REGION_STRATEGIES}, got '{cfg['missing_region_strategy']}'")
    if float(cfg["min_valid_price"]) <= 0:
        raise ValueError("min_valid_price must be positive")
    if float(cfg["max_valid_price"]) <= float(cfg["min_valid_price"]):
        raise ValueError("max_valid_price must exceed min_valid_price")


def clean_data(
    df: pd.DataFrame,
    cleaning_cfg: dict[str, Any] | None = None,
    *,
    report: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Fix the six planted issue types; never mutate the input frame.

    Pass ``report={}`` to capture per-issue counts (used by tests and the
    cleaning audit JSON). The optional ``cleaning_cfg`` exists so tests can
    inject settings without touching the lru_cached global config.
    """
    cfg = cleaning_cfg if cleaning_cfg is not None else get_config().get("cleaning", {})
    _validate_cleaning_cfg(cfg)
    out = df.copy()
    rep: dict[str, int] = report if report is not None else {}
    rep.setdefault("rows_in", len(out))

    # --- 1) unrecoverable rows: missing keys, NaN quantity / unit_cost -----
    bad_rows = (
        out[list(KEY_COLUMNS)].isna().any(axis=1)
        | out["quantity"].isna()
        | out["unit_cost"].isna()
    )
    rep["rows_dropped_unrecoverable"] = int(bad_rows.sum())
    if bad_rows.any():
        logger.warning("Dropping %d unrecoverable row(s) "
                       "(missing key / quantity / unit_cost)", int(bad_rows.sum()))
        out = out.loc[~bad_rows]

    # --- 2) malformed dates (already coerced to NaT in load_raw_data) ------
    bad_dates = out["order_date"].isna()
    rep["rows_dropped_bad_dates"] = int(bad_dates.sum())
    if bad_dates.any():
        logger.info("Dropping %d row(s) with unparseable order_date",
                    int(bad_dates.sum()))
        out = out.loc[~bad_dates]

    # --- 3) duplicate order_ids --------------------------------------------
    if cfg["drop_duplicate_keys"]:
        dupes = out.duplicated(subset="order_id", keep="first")
        rep["duplicates_removed"] = int(dupes.sum())
        if dupes.any():
            logger.info("Removing %d duplicate order_id row(s) (keep first)",
                        int(dupes.sum()))
            out = out.loc[~dupes]
    else:
        rep["duplicates_removed"] = 0

    # --- 4) negative quantities: abs() + audit flag, never silently drop ---
    if cfg["fix_negative_quantities"]:
        neg = out["quantity"] < 0
        out["quantity_adjusted"] = neg
        out.loc[neg, "quantity"] = out.loc[neg, "quantity"].abs()
    else:
        out["quantity_adjusted"] = out["quantity"] < 0  # left as-is, flagged
    rep["negative_quantities_fixed"] = int(out["quantity_adjusted"].sum())
    out["quantity"] = out["quantity"].astype("int64")

    # --- 5) prices: out-of-range -> NaN, then strategy ---------------------
    lo, hi = float(cfg["min_valid_price"]), float(cfg["max_valid_price"])
    out_of_range = out["unit_price"].notna() & ~out["unit_price"].between(lo, hi)
    out.loc[out_of_range, "unit_price"] = np.nan
    rep["prices_out_of_range_reset"] = int(out_of_range.sum())
    if out_of_range.any():
        logger.info("Reset %d out-of-range unit_price value(s) to NaN",
                    int(out_of_range.sum()))

    missing_price = out["unit_price"].isna()
    strategy = str(cfg["missing_price_strategy"])
    if strategy == "drop":
        rep["rows_dropped_missing_price"] = int(missing_price.sum())
        if missing_price.any():
            out = out.loc[~missing_price]
        out["unit_price_imputed"] = False
    elif strategy in ("median", "category_median"):
        if strategy == "category_median":
            cat_median = out.groupby("category")["unit_price"].transform("median")
            out["unit_price"] = out["unit_price"].fillna(cat_median)
        global_median = out["unit_price"].median()
        out["unit_price"] = out["unit_price"].fillna(global_median)
        out["unit_price_imputed"] = missing_price
        rep["prices_imputed"] = int(missing_price.sum())
        if missing_price.any():
            logger.info("Imputed %d missing unit_price value(s) (strategy=%s)",
                        int(missing_price.sum()), strategy)

    # Imputation artifact check: a lost true price can now sit below its cost.
    upside_down = out["unit_cost"] > out["unit_price"]
    if upside_down.any():
        logger.warning("%d row(s) now have unit_cost > unit_price (imputation "
                       "artifact) — margins on those rows are distorted; "
                       "see the unit_price_imputed flag", int(upside_down.sum()))

    # --- 6) missing regions: per-customer mode, then global mode ----------
    # Region is a customer attribute, not an order attribute — so we resolve
    # it from the customer's own other orders before falling back globally.
    missing_region = out["customer_region"].isna()
    if str(cfg["missing_region_strategy"]) == "drop":
        rep["rows_dropped_missing_region"] = int(missing_region.sum())
        if missing_region.any():
            out = out.loc[~missing_region]
        rep["regions_filled_by_customer_mode"] = 0
        rep["regions_filled_by_global_mode"] = 0
    else:
        cust_modes = (
            out.dropna(subset=["customer_region"])
               .groupby("customer_id")["customer_region"]
               .agg(lambda s: s.mode().iat[0])
        )
        by_customer = out["customer_id"].map(cust_modes)
        resolved = missing_region & by_customer.notna()
        out["customer_region"] = out["customer_region"].fillna(by_customer)
        fallback = missing_region & out["customer_region"].isna()
        global_mode = out["customer_region"].mode()
        if len(global_mode):
            out["customer_region"] = out["customer_region"].fillna(global_mode.iat[0])
        rep["regions_filled_by_customer_mode"] = int(resolved.sum())
        rep["regions_filled_by_global_mode"] = int(fallback.sum())
        if missing_region.any():
            logger.info("Filled %d missing region(s): %d from the customer's "
                        "own modal region, %d via global mode",
                        int(missing_region.sum()), int(resolved.sum()),
                        int(fallback.sum()))

    # --- 7) return_reason: preserve structural NaN, patch true holes -------
    orphan_reason = out["returned"] & out["return_reason"].isna()
    rep["return_reasons_filled_unknown"] = int(orphan_reason.sum())
    if orphan_reason.any():
        out.loc[orphan_reason, "return_reason"] = UNKNOWN_RETURN_REASON
        logger.info("Filled %d returned order(s) lacking a reason with '%s'",
                    int(orphan_reason.sum()), UNKNOWN_RETURN_REASON)
    stray = (~out["returned"]) & out["return_reason"].notna()
    if stray.any():
        logger.warning("%d row(s) carry a return_reason without a return — "
                       "left as-is (not part of the cleaning contract)",
                       int(stray.sum()))

    # --- 8) discounts: NaN -> 0, clip to [0, 100] --------------------------
    disc_missing = out["discount_pct"].isna()
    disc_bad = (out["discount_pct"] < 0) | (out["discount_pct"] > 100)
    out["discount_pct"] = out["discount_pct"].fillna(0.0).clip(0.0, 100.0)
    rep["discounts_filled_or_clipped"] = int(disc_missing.sum() + disc_bad.sum())

    # --- finalize -----------------------------------------------------------
    out = out.sort_values("order_date", kind="stable").reset_index(drop=True)
    rep["rows_out"] = len(out)
    logger.info("Cleaning complete: %d -> %d rows (%d dropped)",
                rep["rows_in"], len(out), rep["rows_in"] - len(out))
    return out


def verify_cleaning(
    df: pd.DataFrame, cleaning_cfg: dict[str, Any] | None = None
) -> None:
    """Post-cleaning assertions — the pipeline's built-in evaluation gate.

    Collects ALL violations and raises once with the full list (rather than
    failing on the first), so a bad run produces one actionable message.
    """
    cfg = cleaning_cfg if cleaning_cfg is not None else get_config().get("cleaning", {})
    problems: list[str] = []

    if df.empty:
        problems.append("no rows survive cleaning")
    else:
        if (n := int(df["order_date"].isna().sum())):
            problems.append(f"{n} row(s) still have an invalid order_date")
        if cfg.get("drop_duplicate_keys", True) and (n := int(df["order_id"].duplicated().sum())):
            problems.append(f"{n} duplicate order_id row(s) remain")
        if cfg.get("fix_negative_quantities", True) and (n := int((df["quantity"] < 0).sum())):
            problems.append(f"{n} negative quantity value(s) remain")
        if (n := int(df["unit_price"].isna().sum())):
            problems.append(f"{n} missing unit_price value(s) remain")
        out_of_range = df["unit_price"].notna() & ~df["unit_price"].between(
            float(cfg.get("min_valid_price", 0.01)),
            float(cfg.get("max_valid_price", 1e9)))
        if (n := int(out_of_range.sum())):
            problems.append(f"{n} unit_price value(s) outside valid range")
        if (n := int(df["customer_region"].isna().sum())):
            problems.append(f"{n} missing customer_region value(s) remain")
        if df["returned"].dtype != bool:
            problems.append(f"'returned' dtype is {df['returned'].dtype}, expected bool")
        if (n := int((df["returned"] & df["return_reason"].isna()).sum())):
            problems.append(f"{n} returned order(s) still lack a reason")

    if problems:
        raise DataQualityError(
            "data-quality verification FAILED — dirt survived cleaning:\n  - "
            + "\n  - ".join(problems))
    logger.info("Data-quality verification passed (9 checks, 0 violations)")


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add money, margin, return and time features (see module docstring)."""
    out = df.copy()

    out["return_flag"] = out["returned"].astype("int8")

    out["discounted_unit_price"] = out["unit_price"] * (1 - out["discount_pct"] / 100)
    out["gross_revenue"] = out["quantity"] * out["discounted_unit_price"]
    out["cogs"] = out["quantity"] * out["unit_cost"]
    out["net_revenue"] = out["gross_revenue"] * (1 - out["return_flag"])
    out["revenue_lost_to_returns"] = out["gross_revenue"] * out["return_flag"]
    # Write-off model: returned rows keep full COGS -> profit = -cogs there.
    out["net_profit"] = out["net_revenue"] - out["cogs"]
    # Division-by-zero guard: returned rows have net_revenue == 0 -> NaN.
    out["margin_pct"] = out["net_profit"] / out["net_revenue"].where(
        out["net_revenue"] != 0)

    out["sku_return_rate"] = out.groupby("product_id")["return_flag"].transform("mean")

    out["order_year_month"] = out["order_date"].dt.strftime("%Y-%m")
    out["month"] = out["order_date"].dt.month.astype("int8")

    for col in MONEY_COLUMNS:
        out[col] = out[col].round(2)
    out["margin_pct"] = out["margin_pct"].round(4)
    out["sku_return_rate"] = out["sku_return_rate"].round(4)

    logger.info("Engineered 11 feature columns (%d total)", out.shape[1])
    return out


# --------------------------------------------------------------------------- #
# Customer segmentation (RFM-lite)
# --------------------------------------------------------------------------- #

def _validate_segmentation_cfg(cfg: dict[str, Any]) -> None:
    required = ("champion_min_orders", "champion_recency_days",
                "loyal_min_orders", "at_risk_recency_days",
                "new_customer_window_days")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config.yaml 'segmentation' section is missing "
                         f"key(s): {', '.join(missing)}")
    for key in required:
        if int(cfg[key]) < 0:
            raise ValueError(f"segmentation.{key} must be non-negative")


def segment_customers(
    df: pd.DataFrame, seg_cfg: dict[str, Any] | None = None
) -> pd.DataFrame:
    """RFM-lite segmentation; merges customer_segment onto the order rows.

    Priority (first match wins): Champion > New > Loyal > At Risk, with
    'Occasional' as the catch-all. Snapshot date = latest order in the data.
    """
    if df.empty:
        raise ValueError("cannot segment an empty frame")
    cfg = seg_cfg if seg_cfg is not None else get_config().get("segmentation", {})
    _validate_segmentation_cfg(cfg)

    snapshot = df["order_date"].max()
    stats = df.groupby("customer_id", as_index=False).agg(
        first_order=("order_date", "min"),
        last_order=("order_date", "max"),
        n_orders=("order_id", "count"),
        monetary=("net_revenue", "sum"),
    )
    recency = (snapshot - stats["last_order"]).dt.days
    tenure = (snapshot - stats["first_order"]).dt.days

    stats["customer_segment"] = np.select(
        condlist=[
            (stats["n_orders"] >= int(cfg["champion_min_orders"]))
            & (recency <= int(cfg["champion_recency_days"])),
            tenure <= int(cfg["new_customer_window_days"]),
            (stats["n_orders"] >= int(cfg["loyal_min_orders"]))
            & (recency <= int(cfg["at_risk_recency_days"])),
            recency > int(cfg["at_risk_recency_days"]),
        ],
        choicelist=["Champion", "New", "Loyal", "At Risk"],
        default="Occasional",
    )

    out = df.copy()
    out["customer_segment"] = out["customer_id"].map(
        stats.set_index("customer_id")["customer_segment"])
    if out["customer_segment"].isna().any():
        raise DataQualityError(
            "segmentation left some orders without a segment — report as a bug")

    mix = out["customer_segment"].value_counts()
    logger.info("Segment mix (orders): %s",
                ", ".join(f"{k} {v:,}" for k, v in mix.items()))
    share = (out.groupby("customer_segment")["net_revenue"].sum()
             / out["net_revenue"].sum()).sort_values(ascending=False)
    logger.info("Net-revenue share by segment: %s",
                ", ".join(f"{k} {v:.0%}" for k, v in share.items()))
    return out


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    known = [c for c in FINAL_COLUMN_ORDER if c in df.columns]
    extra = [c for c in df.columns if c not in FINAL_COLUMN_ORDER]
    if extra or len(known) != len(FINAL_COLUMN_ORDER):
        logger.warning("Column drift vs FINAL_COLUMN_ORDER | extra=%s",
                       extra or "none")
    return df[known + extra]


def save_processed(df: pd.DataFrame, path: Path | str) -> Path:
    """Write the cleaned + enriched fact table as snappy-compressed Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = _reorder_columns(df)
    ordered.to_parquet(path, engine="pyarrow", compression="snappy")
    logger.info("Saved processed dataset -> %s (%d rows x %d cols, %.1f MB)",
                path, len(ordered), ordered.shape[1],
                path.stat().st_size / 1e6)
    return path


def _write_cleaning_report(report: dict[str, int], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "src.data_processing",
        "config_note": "counts refer to the run that produced "
                       "processed_sales.parquet",
        **report,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    logger.info("Cleaning audit report -> %s", path)
    return path


def _log_headline(df: pd.DataFrame) -> None:
    """Log the README section-6 'final' numbers for this run."""
    gross = float(df["gross_revenue"].sum())
    net = float(df["net_revenue"].sum())
    profit = float(df["net_profit"].sum())
    lost = float(df["revenue_lost_to_returns"].sum())
    logger.info("Headline | rows=%s | gross $%.2fM | net $%.2fM | profit "
                "$%.2fM | margin %.1f%% | return rate %.1f%% | revenue lost "
                "to returns $%.0fK", f"{len(df):,}", gross / 1e6, net / 1e6,
                profit / 1e6, 100 * profit / net, 100 * df["return_flag"].mean(),
                lost / 1e3)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def main() -> Path:
    """Run Step 2 end-to-end; return the parquet artifact path."""
    cfg = get_config()
    raw = load_raw_data(Path(cfg["paths"]["raw_data"]))

    report: dict[str, int] = {}
    cleaned = clean_data(raw, cfg["cleaning"], report=report)
    verify_cleaning(cleaned, cfg["cleaning"])   # evaluation gate

    featured = engineer_features(cleaned)
    enriched = segment_customers(featured, cfg["segmentation"])
    out_path = save_processed(enriched, Path(cfg["paths"]["processed_data"]))

    report_path = Path(cfg["paths"]["processed_data"]).with_name("cleaning_report.json")
    _write_cleaning_report(report, report_path)
    _log_headline(enriched)
    return out_path


if __name__ == "__main__":
    setup_logging()
    main()
