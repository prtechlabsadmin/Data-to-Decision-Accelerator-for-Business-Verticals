"""tests/test_data_processing.py — every planted issue, exact feature math,
segment priorities, persistence, and a full main() run into tmp_path."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import src.data_processing as dp
from helpers import CLEANING_CFG, SEG_CFG, raw_frame, raw_row
from src.data_processing import (
    UNKNOWN_RETURN_REASON,
    DataQualityError,
    SchemaError,
    clean_data,
    engineer_features,
    load_raw_data,
    save_processed,
    segment_customers,
    validate_schema,
    verify_cleaning,
)


def _write_csv(df: pd.DataFrame, tmp_path: Path, name: str = "raw.csv") -> Path:
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Schema & ingestion
# --------------------------------------------------------------------------- #

def test_validate_schema_missing_column_raises():
    df = raw_frame([raw_row("O1")]).drop(columns=["unit_cost"])
    with pytest.raises(SchemaError, match="unit_cost"):
        validate_schema(df)


def test_validate_schema_empty_raises():
    with pytest.raises(SchemaError, match="0 rows"):
        validate_schema(raw_frame([]))


def test_validate_schema_extra_column_warns_but_passes(caplog):
    df = raw_frame([raw_row("O1")]).assign(unexpected_column=1)
    with caplog.at_level("WARNING", logger="src.data_processing"):
        validate_schema(df)
    assert "unexpected_column" in caplog.text


def test_load_raw_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate it first"):
        load_raw_data(tmp_path / "missing.csv")


def test_load_raw_data_strict_date_parsing(tmp_path):
    """'2024/31/12' must NOT be leniently parsed — strict ISO or NaT."""
    path = tmp_path / "raw.csv"
    pd.DataFrame([
        raw_row("O1", order_date="2024-01-05"),
        raw_row("O2", order_date="2024/31/12"),
        raw_row("O3", order_date="not-a-date"),
    ]).to_csv(path, index=False)
    loaded = load_raw_data(path)
    assert loaded["order_date"].notna().sum() == 1
    assert loaded["order_date"].isna().sum() == 2


def test_load_raw_data_coerces_returned_strings(tmp_path):
    rows = [raw_row("O1"), raw_row("O2")]
    rows[0]["returned"] = "TRUE"
    rows[1]["returned"] = "maybe"
    loaded = load_raw_data(_write_csv(pd.DataFrame(rows), tmp_path))
    assert loaded["returned"].dtype == bool
    assert loaded["returned"].tolist() == [True, False]


def test_load_raw_data_coerces_numeric_junk_to_nan(tmp_path):
    row = raw_row("O1")
    row["quantity"] = "abc"
    loaded = load_raw_data(_write_csv(pd.DataFrame([row]), tmp_path))
    assert pd.isna(loaded["quantity"].iloc[0])


# --------------------------------------------------------------------------- #
# Cleaning — one focused test per planted issue
# --------------------------------------------------------------------------- #

def test_clean_data_fixes_duplicates_keep_first():
    df = raw_frame([
        raw_row("O1", quantity=1),
        raw_row("O1", quantity=3),   # duplicate id, later position
        raw_row("O2"),
    ])
    report: dict = {}
    out = clean_data(df, CLEANING_CFG, report=report)
    assert report["duplicates_removed"] == 1
    assert out["order_id"].is_unique
    assert out.set_index("order_id").loc["O1", "quantity"] == 1


def test_clean_data_negative_quantity_flagged_not_dropped():
    df = raw_frame([raw_row("O1", quantity=-2), raw_row("O2", quantity=3)])
    report: dict = {}
    out = clean_data(df, CLEANING_CFG, report=report)
    assert report["negative_quantities_fixed"] == 1
    assert int(out.loc[out["order_id"] == "O1", "quantity"].iloc[0]) == 2
    assert bool(out.loc[out["order_id"] == "O1", "quantity_adjusted"].iloc[0])


def test_clean_data_negative_quantity_left_when_fix_disabled():
    cfg = {**CLEANING_CFG, "fix_negative_quantities": False}
    df = raw_frame([raw_row("O1", quantity=-2)])
    out = clean_data(df, cfg)
    assert int(out["quantity"].iloc[0]) == -2
    assert bool(out["quantity_adjusted"].iloc[0])
    verify_cleaning(out, cfg)  # the negative check is config-gated: no raise


def test_clean_data_imputes_median_price():
    df = raw_frame([
        raw_row("O1", unit_price=10.0),
        raw_row("O2", unit_price=20.0),
        raw_row("O3", unit_price=30.0),
        raw_row("O4", unit_price=np.nan),
        raw_row("O5", unit_price=0.0),          # below min_valid_price
        raw_row("O6", unit_price=999_999.99),   # above max_valid_price
    ])
    report: dict = {}
    out = clean_data(df, CLEANING_CFG, report=report)
    assert report["prices_out_of_range_reset"] == 2
    assert report["prices_imputed"] == 3
    assert out["unit_price"].isna().sum() == 0
    imputed = float(out.loc[out["order_id"] == "O4", "unit_price"].iloc[0])
    assert imputed == pytest.approx(20.0)
    assert int(out["unit_price_imputed"].sum()) == 3


def test_clean_data_category_median_strategy():
    cfg = {**CLEANING_CFG, "missing_price_strategy": "category_median"}
    df = raw_frame([
        raw_row("O1", category="Apparel", unit_price=10.0),
        raw_row("O2", category="Apparel", unit_price=20.0),
        raw_row("O3", category="Apparel", unit_price=np.nan),
        raw_row("O4", category="Electronics", unit_price=100.0),
        raw_row("O5", category="Electronics", unit_price=300.0),
        raw_row("O6", category="Electronics", unit_price=np.nan),
    ])
    out = clean_data(df, cfg)
    apparel = float(out.loc[out["order_id"] == "O3", "unit_price"].iloc[0])
    electronics = float(out.loc[out["order_id"] == "O6", "unit_price"].iloc[0])
    assert apparel == pytest.approx(15.0)
    assert electronics == pytest.approx(200.0)


def test_clean_data_drop_price_strategy():
    cfg = {**CLEANING_CFG, "missing_price_strategy": "drop"}
    df = raw_frame([raw_row("O1", unit_price=10.0), raw_row("O2", unit_price=np.nan)])
    report: dict = {}
    out = clean_data(df, cfg, report=report)
    assert report["rows_dropped_missing_price"] == 1
    assert len(out) == 1
    assert not out["unit_price_imputed"].any()


def test_clean_data_region_fill_customer_mode_then_global():
    df = raw_frame([
        raw_row("O1", customer_id="C1", customer_region="South"),
        raw_row("O2", customer_id="C1", customer_region="South"),
        raw_row("O3", customer_id="C1", customer_region=np.nan),
        raw_row("O4", customer_id="C3", customer_region="North"),
        raw_row("O5", customer_id="C3", customer_region="North"),
        raw_row("O6", customer_id="C3", customer_region="North"),
        raw_row("O7", customer_id="C2", customer_region=np.nan),
    ])
    report: dict = {}
    out = clean_data(df, CLEANING_CFG, report=report)
    assert report["regions_filled_by_customer_mode"] == 1
    assert report["regions_filled_by_global_mode"] == 1
    assert out.loc[out["order_id"] == "O3", "customer_region"].iloc[0] == "South"
    assert out.loc[out["order_id"] == "O7", "customer_region"].iloc[0] == "North"


def test_clean_data_drops_bad_dates():
    df = raw_frame([raw_row("O1"), raw_row("O2", order_date="not-a-date")])
    report: dict = {}
    out = clean_data(df, CLEANING_CFG, report=report)
    assert report["rows_dropped_bad_dates"] == 1
    assert out["order_id"].tolist() == ["O1"]


def test_clean_data_drops_unrecoverable_rows():
    df = raw_frame([
        raw_row("O1"),
        raw_row("O2", quantity=np.nan),
        raw_row("O3", unit_cost=np.nan),
    ])
    report: dict = {}
    out = clean_data(df, CLEANING_CFG, report=report)
    assert report["rows_dropped_unrecoverable"] == 2
    assert out["order_id"].tolist() == ["O1"]


def test_clean_data_return_reason_semantics():
    df = raw_frame([
        raw_row("O1"),                                           # NaN reason stays
        raw_row("O2", returned=True, return_reason="Wrong size or fit"),
        raw_row("O3", returned=True),                            # orphan -> Unknown
        raw_row("O4", return_reason="Changed mind"),             # stray: kept as-is
    ])
    report: dict = {}
    out = clean_data(df, CLEANING_CFG, report=report)
    by_id = out.set_index("order_id")
    assert pd.isna(by_id.loc["O1", "return_reason"])
    assert by_id.loc["O2", "return_reason"] == "Wrong size or fit"
    assert by_id.loc["O3", "return_reason"] == UNKNOWN_RETURN_REASON
    assert by_id.loc["O4", "return_reason"] == "Changed mind"
    assert report["return_reasons_filled_unknown"] == 1


def test_clean_data_discount_fill_and_clip():
    df = raw_frame([
        raw_row("O1", discount_pct=np.nan),
        raw_row("O2", discount_pct=150.0),
        raw_row("O3", discount_pct=-5.0),
    ])
    report: dict = {}
    out = clean_data(df, CLEANING_CFG, report=report)
    assert out["discount_pct"].tolist() == [0.0, 100.0, 0.0]
    assert report["discounts_filled_or_clipped"] == 3


def test_clean_data_does_not_mutate_input():
    df = raw_frame([
        raw_row("O1", quantity=-1, unit_price=np.nan, customer_region=np.nan),
        raw_row("O2", order_date="junk"),
    ])
    snapshot = df.copy(deep=True)
    clean_data(df, CLEANING_CFG)
    assert_frame_equal(df, snapshot)


# --------------------------------------------------------------------------- #
# Verification gate
# --------------------------------------------------------------------------- #

def test_verify_cleaning_passes_on_clean_output():
    df = raw_frame([
        raw_row("O1"),
        raw_row("O2", returned=True, return_reason="Damaged on arrival"),
    ])
    verify_cleaning(clean_data(df, CLEANING_CFG), CLEANING_CFG)  # no raise


def test_verify_cleaning_collects_all_violations_at_once():
    df = raw_frame([raw_row("O1"), raw_row("O2")])
    out = clean_data(df, CLEANING_CFG)
    out.loc[out.index[0], "order_date"] = pd.NaT
    out.loc[out.index[0], "unit_price"] = np.nan
    out.loc[out.index[1], "quantity"] = -1
    out.loc[out.index[1], "order_id"] = out.loc[out.index[0], "order_id"]
    with pytest.raises(DataQualityError) as excinfo:
        verify_cleaning(out, CLEANING_CFG)
    message = str(excinfo.value)
    for fragment in ("invalid order_date", "missing unit_price",
                     "negative quantity", "duplicate order_id"):
        assert fragment in message


def test_verify_cleaning_empty_output_raises():
    out = clean_data(raw_frame([raw_row("O1", order_date="junk")]), CLEANING_CFG)
    assert out.empty
    with pytest.raises(DataQualityError, match="no rows survive"):
        verify_cleaning(out, CLEANING_CFG)


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"missing_price_strategy": "bogus"}, "missing_price_strategy"),
        ({"missing_region_strategy": "bogus"}, "missing_region_strategy"),
        ({"min_valid_price": 0.0}, "min_valid_price"),
        ({"max_valid_price": 0.001}, "max_valid_price"),
    ],
)
def test_clean_data_rejects_bad_config(override, fragment):
    with pytest.raises(ValueError, match=fragment):
        clean_data(raw_frame([raw_row("O1")]), {**CLEANING_CFG, **override})


def test_clean_data_rejects_missing_config_keys():
    incomplete = {k: v for k, v in CLEANING_CFG.items() if k != "min_valid_price"}
    with pytest.raises(ValueError, match="missing key"):
        clean_data(raw_frame([raw_row("O1")]), incomplete)


# --------------------------------------------------------------------------- #
# Feature engineering — exact math
# --------------------------------------------------------------------------- #

def test_engineer_features_exact_math():
    df = raw_frame([
        raw_row("O1", quantity=2, unit_price=100.0, unit_cost=60.0,
                discount_pct=10.0, product_id="PA"),
        raw_row("O2", quantity=1, unit_price=50.0, unit_cost=30.0,
                returned=True, return_reason="Changed mind", product_id="PB"),
        raw_row("O3", quantity=3, unit_price=20.0, unit_cost=10.0,
                discount_pct=50.0, product_id="PA"),
        raw_row("O4", quantity=3, unit_price=10.555, unit_cost=4.0),
    ])
    out = engineer_features(df).set_index("order_id")
    assert out.loc["O1", "discounted_unit_price"] == pytest.approx(90.0)
    assert out.loc["O1", "gross_revenue"] == pytest.approx(180.0)
    assert out.loc["O1", "cogs"] == pytest.approx(120.0)
    assert out.loc["O1", "net_revenue"] == pytest.approx(180.0)
    assert out.loc["O1", "revenue_lost_to_returns"] == pytest.approx(0.0)
    assert out.loc["O1", "net_profit"] == pytest.approx(60.0)
    assert out.loc["O1", "margin_pct"] == pytest.approx(60 / 180, rel=1e-3)
    # returned row: write-off model — zero revenue, full COGS, undefined margin
    assert out.loc["O2", "net_revenue"] == pytest.approx(0.0)
    assert out.loc["O2", "revenue_lost_to_returns"] == pytest.approx(50.0)
    assert out.loc["O2", "net_profit"] == pytest.approx(-30.0)
    assert pd.isna(out.loc["O2", "margin_pct"])
    # 50% discount lands exactly on zero margin
    assert out.loc["O3", "net_profit"] == pytest.approx(0.0)
    assert out.loc["O3", "margin_pct"] == pytest.approx(0.0)
    # money columns are rounded to 2dp
    assert out.loc["O4", "gross_revenue"] == pytest.approx(31.66, abs=0.01)
    # SKU-level return rate + time features
    assert out.loc["O1", "sku_return_rate"] == pytest.approx(0.0)
    assert out.loc["O2", "sku_return_rate"] == pytest.approx(1.0)
    assert out["return_flag"].tolist() == [0, 1, 0, 0]
    assert (out["order_year_month"] == "2024-01").all()
    assert (out["month"] == 1).all()


# --------------------------------------------------------------------------- #
# Segmentation — the priority matrix
# --------------------------------------------------------------------------- #

def _segmentation_frame() -> pd.DataFrame:
    rows = (
        [raw_row(f"C1-{i}", order_date=f"2024-12-{10 + i}", customer_id="C1")
         for i in range(6)]                                   # Champion (beats New)
        + [raw_row("C2-1", order_date="2024-12-30", customer_id="C2")]   # New
        + [raw_row(f"C3-{i}", order_date=d, customer_id="C3")
           for d in ("2024-06-01", "2024-09-01", "2024-11-01")]          # Loyal
        + [raw_row("C4-1", order_date="2024-08-01", customer_id="C4")]   # At Risk
        + [raw_row("C5-1", order_date="2024-10-01", customer_id="C5"),
           raw_row("C5-2", order_date="2024-11-15", customer_id="C5")]   # Occasional
    )
    return engineer_features(clean_data(raw_frame(rows), CLEANING_CFG))


def test_segment_customers_priority_order():
    out = segment_customers(_segmentation_frame(), SEG_CFG)
    by_customer = out.groupby("customer_id")["customer_segment"].first()
    assert by_customer["C1"] == "Champion"   # beats New despite short tenure
    assert by_customer["C2"] == "New"
    assert by_customer["C3"] == "Loyal"
    assert by_customer["C4"] == "At Risk"
    assert by_customer["C5"] == "Occasional"


def test_segment_customers_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        segment_customers(raw_frame([]), SEG_CFG)


def test_segment_customers_rejects_bad_config():
    df = _segmentation_frame()
    with pytest.raises(ValueError, match="missing key"):
        segment_customers(df, {"champion_min_orders": 5})
    with pytest.raises(ValueError, match="non-negative"):
        segment_customers(df, {**SEG_CFG, "loyal_min_orders": -1})


# --------------------------------------------------------------------------- #
# Persistence & full main()
# --------------------------------------------------------------------------- #

def test_save_processed_roundtrip(tmp_path):
    df = engineer_features(clean_data(raw_frame([
        raw_row("O1"),
        raw_row("O2", returned=True, return_reason="Damaged on arrival"),
    ]), CLEANING_CFG))
    path = save_processed(df, tmp_path / "processed.parquet")
    back = pd.read_parquet(path)
    assert path.is_file()
    assert len(back) == len(df)
    assert back["returned"].dtype == bool
    assert set(dp.REQUIRED_COLUMNS) <= set(back.columns)


def _integration_raw_rows() -> list[dict]:
    return [
        raw_row("O1", order_date="2024-02-10", customer_id="C1",
                customer_region="North", unit_price=50.0, unit_cost=25.0),
        raw_row("O2", order_date="2024-05-15", customer_id="C1",
                customer_region="North", unit_price=60.0, unit_cost=30.0),
        raw_row("O3", order_date="2024-08-20", customer_id="C2",
                customer_region="South", unit_price=80.0, unit_cost=40.0),
        raw_row("O4", order_date="2024-11-25", customer_id="C3",
                customer_region="East", unit_price=40.0, unit_cost=20.0),
        raw_row("O5", order_date="2024-03-12", customer_id="C2",
                customer_region="South", unit_price=30.0, unit_cost=15.0),
        raw_row("O6", order_date="2024-07-07", customer_id="C1",
                customer_region="North", unit_price=25.0, unit_cost=12.0),
        raw_row("O7", order_date="2024-09-30", customer_id="C3",
                customer_region="East", unit_price=90.0, unit_cost=45.0),
        raw_row("O8", order_date="2024-12-05", customer_id="C2",
                customer_region="South", unit_price=70.0, unit_cost=35.0),
        raw_row("O1", order_date="2024-02-10", customer_id="C1",       # duplicate
                customer_region="North", unit_price=50.0, unit_cost=25.0),
        raw_row("O9", order_date="2024-04-04", customer_id="C1",       # negative qty
                customer_region="North", quantity=-2, unit_price=20.0, unit_cost=10.0),
        raw_row("O10", order_date="2024-06-06", customer_id="C2",      # missing price
                customer_region="South", unit_price=np.nan, unit_cost=15.0),
        raw_row("O11", order_date="2024-10-10", customer_id="C1",      # missing region
                customer_region=np.nan, unit_price=55.0, unit_cost=27.0),
        raw_row("O12", order_date="garbage-date", customer_id="C3",    # bad date
                customer_region="East", unit_price=65.0, unit_cost=32.0),
    ]


@pytest.mark.integration
def test_main_end_to_end(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.csv"
    raw_frame(_integration_raw_rows()).to_csv(raw_path, index=False)
    processed_path = tmp_path / "processed.parquet"
    cfg = {
        "paths": {"raw_data": str(raw_path), "processed_data": str(processed_path)},
        "cleaning": CLEANING_CFG,
        "segmentation": SEG_CFG,
    }
    monkeypatch.setattr(dp, "get_config", lambda: cfg)
    returned = dp.main()
    assert returned == processed_path
    assert processed_path.is_file()
    report = json.loads((tmp_path / "cleaning_report.json").read_text(encoding="utf-8"))
    assert report["rows_in"] == 13
    assert report["rows_dropped_bad_dates"] == 1
    assert report["duplicates_removed"] == 1
    assert report["prices_imputed"] == 1
    assert report["negative_quantities_fixed"] == 1
    assert report["regions_filled_by_customer_mode"] == 1
    assert report["rows_out"] == len(pd.read_parquet(processed_path)) == 11


def test_main_missing_raw_raises(tmp_path, monkeypatch):
    cfg = {
        "paths": {"raw_data": str(tmp_path / "nope.csv"),
                  "processed_data": str(tmp_path / "p.parquet")},
        "cleaning": CLEANING_CFG,
        "segmentation": SEG_CFG,
    }
    monkeypatch.setattr(dp, "get_config", lambda: cfg)
    with pytest.raises(FileNotFoundError):
        dp.main()
