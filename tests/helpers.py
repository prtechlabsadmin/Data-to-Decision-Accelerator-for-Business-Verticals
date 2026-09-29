"""tests/helpers.py — handcrafted-frame builders and shared test configs.

Testing philosophy for this repo:
    * UNIT tests pin EXACT behaviour on tiny frames with known numbers.
    * ONE module-scoped "mini pipeline" (build_mini_processed) covers
      integration: seeded generation -> cleaning -> features -> segmentation.
    * Tests never touch the repo's real data/ artifacts (the only exception
      is the AppTest smoke test, which skips itself when they are absent).

Every function under test accepts explicit config dicts, so tests inject
settings directly instead of fighting the lru_cache'd global config.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Explicit config dicts (mirror config.yaml defaults)
# ---------------------------------------------------------------------------

CLEANING_CFG: dict = {
    "drop_duplicate_keys": True,
    "fix_negative_quantities": True,
    "missing_price_strategy": "median",
    "missing_region_strategy": "mode",
    "min_valid_price": 0.01,
    "max_valid_price": 100_000.0,
}

SEG_CFG: dict = {
    "champion_min_orders": 5,
    "champion_recency_days": 30,
    "loyal_min_orders": 3,
    "at_risk_recency_days": 90,
    "new_customer_window_days": 30,
}

ANALYSIS_CFG: dict = {
    "high_return_threshold": 0.15,
    "min_category_volume": 50,
    "min_sku_volume": 20,
    "top_n_return_drivers": 5,
    "top_n_insights": 5,
    "mom_decline_threshold": -0.10,
}

MINI_GEN: dict = {
    "seed": 7,
    "n_products": 60,
    "n_customers": 150,
    "n_orders": 3_000,
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "dirty_row_fraction": 0.0,
    "monthly_seasonality": [0.85, 0.90, 1.00, 0.95, 1.05, 1.10,
                            0.92, 0.88, 1.02, 1.08, 1.15, 1.30],
    "categories": ["Electronics", "Apparel", "Home & Kitchen",
                   "Beauty", "Sports", "Toys"],
    "regions": ["North", "South", "East", "West", "Central"],
}

# ---------------------------------------------------------------------------
# Raw-frame builder (the 13-column Step-1 schema, post-load state)
# ---------------------------------------------------------------------------

RAW_COLS = [
    "order_id", "order_date", "customer_id", "customer_region",
    "product_id", "product_name", "category", "quantity",
    "unit_price", "unit_cost", "discount_pct", "returned", "return_reason",
]


def raw_row(
    order_id: str = "O1",
    order_date: str = "2024-01-05",
    customer_id: str = "C1",
    customer_region: str | None = "North",
    product_id: str = "P1",
    product_name: str = "Widget",
    category: str = "Electronics",
    quantity: int | float = 1,
    unit_price: float = 100.0,
    unit_cost: float = 60.0,
    discount_pct: float = 0.0,
    returned: bool = False,
    return_reason: object = np.nan,
) -> dict:
    """One raw-schema row; junk values are passed through untouched."""
    return {
        "order_id": order_id,
        "order_date": order_date,
        "customer_id": customer_id,
        "customer_region": customer_region,
        "product_id": product_id,
        "product_name": product_name,
        "category": category,
        "quantity": quantity,
        "unit_price": unit_price,
        "unit_cost": unit_cost,
        "discount_pct": discount_pct,
        "returned": returned,
        "return_reason": return_reason,
    }


def raw_frame(rows: list[dict]) -> pd.DataFrame:
    """Raw frame in the post-load state: datetimes (junk -> NaT), bool returned."""
    df = pd.DataFrame(rows, columns=RAW_COLS)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Processed-frame builder (Step-2 output schema, consistent money columns)
# ---------------------------------------------------------------------------

FACT_COLS = [
    "order_id", "order_date", "month", "customer_id", "customer_region",
    "customer_segment", "product_id", "product_name", "category",
    "quantity", "unit_cost", "discount_pct", "returned", "return_flag",
    "return_reason", "gross_revenue", "net_revenue", "cogs",
    "revenue_lost_to_returns", "net_profit",
]


def fact_row(
    order_id: str = "O1",
    date: str = "2024-01-05",
    customer_id: str = "C1",
    customer_region: str = "North",
    customer_segment: str = "Champion",
    product_id: str = "P1",
    product_name: str = "Widget",
    category: str = "Electronics",
    quantity: int = 1,
    unit_price: float = 100.0,
    unit_cost: float = 60.0,
    discount_pct: float = 0.0,
    returned: bool = False,
    return_reason: str | None = None,
) -> dict:
    """One processed-schema row with internally consistent money columns
    (net revenue, COGS, write-off profit model)."""
    gross = round(quantity * unit_price * (1 - discount_pct / 100), 2)
    cogs = round(quantity * unit_cost, 2)
    flag = int(bool(returned))
    ts = pd.Timestamp(date)
    reason = np.nan
    if returned and return_reason:
        reason = return_reason
    return {
        "order_id": order_id,
        "order_date": ts,
        "month": ts.month,
        "customer_id": customer_id,
        "customer_region": customer_region,
        "customer_segment": customer_segment,
        "product_id": product_id,
        "product_name": product_name,
        "category": category,
        "quantity": quantity,
        "unit_cost": unit_cost,
        "discount_pct": discount_pct,
        "returned": bool(returned),
        "return_flag": flag,
        "return_reason": reason,
        "gross_revenue": gross,
        "net_revenue": round(gross * (1 - flag), 2),
        "cogs": cogs,
        "revenue_lost_to_returns": round(gross * flag, 2),
        "net_profit": round(gross * (1 - flag) - cogs, 2),
    }


def fact_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=FACT_COLS)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


# ---------------------------------------------------------------------------
# The mini pipeline (integration fixture backing)
# ---------------------------------------------------------------------------

def build_mini_processed() -> pd.DataFrame:
    """Seeded mini pipeline: generate -> clean -> verify -> engineer -> segment."""
    from src.data_generation import (
        generate_customers,
        generate_orders,
        generate_products,
    )
    from src.data_processing import (
        clean_data,
        engineer_features,
        segment_customers,
        verify_cleaning,
    )

    rng = np.random.default_rng(MINI_GEN["seed"])
    products = generate_products(MINI_GEN["n_products"], MINI_GEN["categories"], rng)
    customers = generate_customers(MINI_GEN["n_customers"], MINI_GEN["regions"], rng)
    orders = generate_orders(products, customers, MINI_GEN, rng)
    # load_raw_data's job, done in-memory so the mini pipeline needs no disk.
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    cleaned = clean_data(orders, CLEANING_CFG)
    verify_cleaning(cleaned, CLEANING_CFG)
    featured = engineer_features(cleaned)
    return segment_customers(featured, SEG_CFG)
