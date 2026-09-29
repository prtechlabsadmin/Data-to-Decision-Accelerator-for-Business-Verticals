"""tests/test_data_generation.py — determinism, invariants, planted stories,
injected dirt. Exact counts come from ISSUE_MIX itself, not magic numbers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from helpers import MINI_GEN, raw_frame, raw_row
from src.data_generation import (
    CATEGORY_PROFILE,
    DEFAULT_PROFILE,
    DECLINING_REGION,
    ISSUE_MIX,
    MALFORMED_DATES,
    PROBLEM_SKU_RATE_CAP,
    _profile,
    generate_customers,
    generate_orders,
    generate_products,
    inject_data_issues,
    save_raw_data,
)

ORDER_COLUMNS = {
    "order_id", "order_date", "customer_id", "customer_region", "product_id",
    "product_name", "category", "quantity", "unit_price", "unit_cost",
    "discount_pct", "returned", "return_reason",
}


def _build(seed: int, n_orders: int = 3_000, frac: float = 0.1):
    """Generate products/customers/orders and inject dirt from one rng stream."""
    rng = np.random.default_rng(seed)
    products = generate_products(MINI_GEN["n_products"], MINI_GEN["categories"], rng)
    customers = generate_customers(MINI_GEN["n_customers"], MINI_GEN["regions"], rng)
    cfg = {**MINI_GEN, "n_orders": n_orders}
    orders = generate_orders(products, customers, cfg, rng)
    dirty = inject_data_issues(orders, rng, frac)
    return orders, dirty


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

def test_generation_is_deterministic():
    assert_frame_equal(_build(7)[1], _build(7)[1])


def test_different_seed_changes_data():
    with pytest.raises(AssertionError):
        assert_frame_equal(_build(7)[1], _build(8)[1])


# --------------------------------------------------------------------------- #
# Clean-order invariants
# --------------------------------------------------------------------------- #

def test_clean_orders_schema_and_invariants():
    orders, _ = _build(7)
    assert set(orders.columns) == ORDER_COLUMNS
    assert orders["order_id"].is_unique
    assert orders["returned"].dtype == bool
    assert orders["quantity"].between(1, 5).all()
    # structural missingness: a reason exists exactly when the order returned
    assert (orders["return_reason"].isna() == ~orders["returned"]).all()
    assert len(orders) == 3_000
    dates = pd.to_datetime(orders["order_date"])
    assert str(dates.min()) >= "2024-01-01"
    assert str(dates.max()) <= "2024-12-31"


def test_seasonality_and_region_stories_are_planted():
    orders, _ = _build(7)
    month = pd.to_datetime(orders["order_date"]).dt.month
    assert (month == 12).sum() > (month == 8).sum()      # Dec peak vs Aug trough
    east = orders["customer_region"] == DECLINING_REGION
    assert (east & (month == 12)).sum() < (east & (month == 1)).sum()


# --------------------------------------------------------------------------- #
# Products & customers
# --------------------------------------------------------------------------- #

def test_products_catalog_invariants_and_problem_skus():
    rng = np.random.default_rng(7)
    products = generate_products(60, MINI_GEN["categories"], rng)
    assert products["product_id"].is_unique
    assert set(products["category"]) == set(MINI_GEN["categories"])
    assert (products["unit_cost"] < products["unit_price"]).all()
    assert products["base_return_rate"].between(0.0, 0.9).all()
    for category in MINI_GEN["categories"]:
        expected = min(PROBLEM_SKU_RATE_CAP,
                       CATEGORY_PROFILE[category]["return_rate"] * 3.0)
        group = products[products["category"] == category]
        assert group["base_return_rate"].max() == pytest.approx(expected, abs=1e-9)
        top = group["popularity"].idxmax()
        assert products.loc[top, "base_return_rate"] == pytest.approx(expected, abs=1e-9)


def test_products_require_one_per_category():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="n_products"):
        generate_products(2, MINI_GEN["categories"], rng)


def test_unknown_category_falls_back_to_default_profile():
    assert _profile("Does Not Exist") is DEFAULT_PROFILE


def test_customers_invariants():
    rng = np.random.default_rng(7)
    customers = generate_customers(150, MINI_GEN["regions"], rng)
    assert customers["customer_id"].is_unique
    assert set(customers["region"]) <= set(MINI_GEN["regions"])
    assert (customers["activity_weight"] > 0).all()
    assert customers["churn_frac"].between(0.0, 1.0).all()


def test_customers_reject_empty_regions_and_zero_count():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="regions"):
        generate_customers(10, [], rng)
    with pytest.raises(ValueError, match="positive"):
        generate_customers(0, ["North"], rng)


def test_generate_orders_rejects_bad_config():
    rng = np.random.default_rng(1)
    products = generate_products(12, ["Electronics", "Apparel"], rng)
    customers = generate_customers(20, ["North"], rng)
    bad_seasonality = {**MINI_GEN, "monthly_seasonality": [1.0] * 11}
    with pytest.raises(ValueError, match="monthly_seasonality"):
        generate_orders(products, customers, bad_seasonality, rng)
    bad_dates = {**MINI_GEN, "end_date": "2023-12-31"}
    with pytest.raises(ValueError, match="end_date"):
        generate_orders(products, customers, bad_dates, rng)
    zero_orders = {**MINI_GEN, "n_orders": 0}
    with pytest.raises(ValueError, match="n_orders"):
        generate_orders(products, customers, zero_orders, rng)


# --------------------------------------------------------------------------- #
# Issue injection
# --------------------------------------------------------------------------- #

def _expected_issue_counts(budget: int) -> dict[str, int]:
    """Mirror the largest-remainder split used by inject_data_issues."""
    names = list(ISSUE_MIX)
    fracs = np.array([ISSUE_MIX[k] for k in names], dtype=float)
    counts = np.floor(fracs / fracs.sum() * budget).astype(int)
    counts[0] += budget - int(counts.sum())  # remainder lands on the first issue
    return dict(zip(names, counts.tolist()))


def _small_orders(seed: int, n_orders: int):
    rng = np.random.default_rng(seed)
    products = generate_products(24, ["Electronics", "Apparel", "Beauty"], rng)
    customers = generate_customers(80, MINI_GEN["regions"], rng)
    orders = generate_orders(products, customers, {**MINI_GEN, "n_orders": n_orders}, rng)
    return orders, rng


def test_inject_data_issues_budget_split_and_counts():
    orders, rng = _small_orders(11, 1_000)
    dirty = inject_data_issues(orders, rng, 0.10)
    expected = _expected_issue_counts(100)
    assert len(dirty) == 1_000 + expected["duplicate_row"]
    assert dirty["order_id"].duplicated().sum() == expected["duplicate_row"]
    assert dirty["unit_price"].isna().sum() == (
        expected["missing_price"] + expected["out_of_range_price"]
    )
    assert (dirty["quantity"] < 0).sum() == expected["negative_quantity"]
    assert dirty["order_date"].isin(MALFORMED_DATES).sum() == expected["malformed_date"]
    assert dirty["customer_region"].isna().sum() == expected["missing_region"]
    # structural missingness survives the injection untouched
    assert (dirty["return_reason"].isna() == ~dirty["returned"]).all()


def test_inject_data_issues_does_not_mutate_input():
    orders, rng = _small_orders(11, 500)
    snapshot = orders.copy(deep=True)
    inject_data_issues(orders, rng, 0.10)
    assert_frame_equal(orders, snapshot)


@pytest.mark.parametrize("frac", [-0.1, 1.5])
def test_inject_data_issues_rejects_bad_fraction(frac):
    rng = np.random.default_rng(1)
    df = raw_frame([raw_row("O1"), raw_row("O2")])
    with pytest.raises(ValueError, match="dirty_fraction"):
        inject_data_issues(df, rng, frac)


def test_inject_data_issues_zero_fraction_is_noop():
    rng = np.random.default_rng(1)
    df = raw_frame([raw_row("O1"), raw_row("O2")])
    assert_frame_equal(inject_data_issues(df, rng, 0.0), df)


def test_save_raw_data_roundtrip(tmp_path):
    df = raw_frame([
        raw_row("O1"),
        raw_row("O2", returned=True, return_reason="Changed mind"),
    ])
    path = save_raw_data(df, tmp_path / "raw.csv")
    back = pd.read_csv(path)
    assert path.is_file()
    assert list(back.columns) == list(df.columns)
    assert back["returned"].dtype == bool
    assert back["order_id"].tolist() == ["O1", "O2"]
    returned = back.loc[back["order_id"] == "O2", "return_reason"].iloc[0]
    assert returned == "Changed mind"
