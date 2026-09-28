"""src/data_generation.py — Step 1: synthetic raw e-commerce sales data.

Builds a deterministic (seeded) fact table of orders, products, customers and
returns over the configured period, then deliberately injects realistic
data-quality issues that Step 2 (src/data_processing.py) must detect and fix.

Planted business stories — the analysis stage is designed to find these:
    1. Category return-rate spread — Apparel bleeds margin (~24% returns),
       Beauty is clean (~5%).
    2. Problem SKUs — one high-volume SKU per category with a ~3x elevated
       return rate (these become the "margin bleed" insights).
    3. Seasonality — straight from config.yaml (Aug trough, Dec peak).
    4. Declining region — 'East' loses ~60% of order share across the year.
    5. Customer heterogeneity — whale customers + ~15% mid-year churners,
       powering the RFM-lite segmentation later.

Output schema (data/raw_sales.csv, 13 columns) — Step 2 validates against this:
    order_id          str    unique per order (duplicates injected afterwards)
    order_date        str    ISO date; a few malformed values injected
    customer_id       str
    customer_region   str    customer's home region (NaN injected)
    product_id        str
    product_name      str
    category          str
    quantity          int    items ordered (negative values injected)
    unit_price        float  list price (NaN + out-of-range values injected)
    unit_cost         float  cost per unit (always < unit_price)
    discount_pct      float  0–30, heavier in Nov/Dec
    returned          bool   True if the order came back
    return_reason     str    NaN when returned is False — STRUCTURAL
                             missingness (a reason cannot exist), NOT dirt:
                             the cleaning stage must leave it alone.

Module contract (see run_pipeline.py):  main() -> Path  returns the artifact.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_config, get_logger, setup_logging

logger = get_logger(__name__)

__all__ = [
    "generate_products",
    "generate_customers",
    "generate_orders",
    "inject_data_issues",
    "save_raw_data",
    "main",
]

# --------------------------------------------------------------------------- #
# Data stories — module constants, NOT tunables (tunables live in config.yaml)
# --------------------------------------------------------------------------- #

CATEGORY_PROFILE: dict[str, dict[str, Any]] = {
    #                       list-price range  cost/price ratio  base return  order share
    "Electronics":    {"price_range": (19.0, 799.0), "cost_ratio": (0.60, 0.78),
                       "return_rate": 0.12, "order_share": 0.15},
    "Apparel":        {"price_range": (9.0, 89.0),   "cost_ratio": (0.35, 0.55),
                       "return_rate": 0.24, "order_share": 0.30},
    "Home & Kitchen": {"price_range": (7.0, 149.0),  "cost_ratio": (0.45, 0.65),
                       "return_rate": 0.08, "order_share": 0.20},
    "Beauty":         {"price_range": (4.0, 59.0),   "cost_ratio": (0.25, 0.45),
                       "return_rate": 0.05, "order_share": 0.15},
    "Sports":         {"price_range": (11.0, 199.0), "cost_ratio": (0.50, 0.70),
                       "return_rate": 0.10, "order_share": 0.10},
    "Toys":           {"price_range": (4.0, 69.0),   "cost_ratio": (0.40, 0.60),
                       "return_rate": 0.04, "order_share": 0.10},
}
DEFAULT_PROFILE: dict[str, Any] = {
    "price_range": (10.0, 120.0), "cost_ratio": (0.40, 0.65),
    "return_rate": 0.10, "order_share": 0.10,
}

REGION_WEIGHTS: dict[str, float] = {
    "North": 0.30, "South": 0.22, "West": 0.18, "East": 0.15, "Central": 0.15,
}
DEFAULT_REGION_WEIGHT = 0.15
DECLINING_REGION = "East"
REGION_DECAY_PER_MONTH = 0.055  # East's weight: 1.00 -> ~0.40 by the final month
CHURN_FRACTION = 0.15           # share of customers who stop ordering mid-year

PROBLEM_SKU_MULTIPLIER = 3.0    # most popular SKU per category: ~3x returns
PROBLEM_SKU_RATE_CAP = 0.50

BRANDS: tuple[str, ...] = (
    "Aurora", "Nimbus", "Vertex", "Lumen", "Cascade", "Orbit",
    "Zephyr", "Pinnacle", "Solstice", "Meridian", "Halcyon", "Quartz",
)
PRODUCT_TYPES: dict[str, tuple[str, ...]] = {
    "Electronics": ("Wireless Earbuds", "Smart Speaker", "4K Action Camera",
                    "USB-C Hub", "Mechanical Keyboard", "Wireless Mouse",
                    "Fast-Charge Power Bank", "Noise-Cancelling Headphones",
                    "Fitness Smartwatch", "HD Webcam"),
    "Apparel": ("Cotton T-Shirt", "Slim Jeans", "Hoodie", "Running Jacket",
                "Summer Dress", "Casual Sneakers", "Wool Sweater",
                "Linen Shirt", "Yoga Leggings", "Rain Coat"),
    "Home & Kitchen": ("Stainless Steel Kettle", "Ceramic Dinner Set", "Blender",
                       "Non-Stick Pan", "Memory Foam Pillow", "Cotton Bedsheet Set",
                       "Air Fryer", "Cutlery Set", "Table Lamp", "Storage Organiser"),
    "Beauty": ("Vitamin C Serum", "Hydrating Face Cream", "Shampoo & Conditioner Set",
               "Electric Toothbrush", "Lipstick Set", "Sunscreen SPF 50",
               "Hair Dryer", "Face Mask Pack", "Perfume", "Makeup Brush Set"),
    "Sports": ("Yoga Mat", "Adjustable Dumbbell", "Resistance Band Set",
               "Cycling Helmet", "Trail Running Shoes", "Tennis Racket",
               "Jump Rope", "Insulated Water Bottle", "Gym Gloves", "Backpack"),
    "Toys": ("Building Block Set", "Remote Control Car", "Board Game", "Plush Toy",
             "Science Kit", "Puzzle Set", "Toy Train Set", "Art & Craft Kit",
             "Action Figure", "Doll House"),
}
DEFAULT_PRODUCT_TYPES: tuple[str, ...] = (
    "Generic Product", "Basic Model", "Standard Edition", "Classic Version",
)
MODEL_SUFFIXES: tuple[str, ...] = (
    "Pro", "Max", "Lite", "Plus", "Series 3", "Series 5", "X2", "X4", "Elite", "v2",
)

QUANTITY_CHOICES: list[int] = [1, 2, 3, 4, 5]
QUANTITY_P: list[float] = [0.52, 0.27, 0.12, 0.06, 0.03]

DISCOUNT_PROB_BASE = 0.35
DISCOUNT_PROB_PEAK = 0.55  # promo pressure in Nov/Dec
PEAK_MONTHS = frozenset({10, 11})  # 0-based month indices
DISCOUNT_RANGE = (5.0, 30.0)
WEEKEND_WEIGHT = 1.35

DEFAULT_RETURN_REASONS: dict[str, float] = {
    "Changed mind": 0.30, "Not as described": 0.25, "Damaged on arrival": 0.20,
    "Found better price": 0.15, "Quality below expectations": 0.10,
}
CATEGORY_RETURN_REASONS: dict[str, dict[str, float]] = {
    "Apparel": {"Wrong size or fit": 0.45, "Changed mind": 0.20,
                "Not as described": 0.15, "Found better price": 0.10,
                "Quality below expectations": 0.10},
    "Electronics": {"Not as described": 0.30, "Damaged on arrival": 0.25,
                    "Changed mind": 0.20, "Found better price": 0.15,
                    "Quality below expectations": 0.10},
    "Beauty": {"Changed mind": 0.40, "Not as described": 0.25,
               "Found better price": 0.15, "Quality below expectations": 0.10,
               "Damaged on arrival": 0.10},
}

# How the dirty-row budget (config: dirty_row_fraction) is split across issues.
ISSUE_MIX: dict[str, float] = {
    "missing_price": 0.30,
    "missing_region": 0.25,
    "negative_quantity": 0.15,
    "out_of_range_price": 0.10,
    "malformed_date": 0.10,
    "duplicate_row": 0.10,
}
MALFORMED_DATES: tuple[str, ...] = ("2024-13-45", "not-a-date", "2024/31/12", "N/A", "")


# --------------------------------------------------------------------------- #
# Small helpers (pure -> easy unit-test targets)
# --------------------------------------------------------------------------- #

def _profile(category: str) -> dict[str, Any]:
    """Category profile, falling back to a default for unknown categories."""
    prof = CATEGORY_PROFILE.get(category)
    if prof is None:
        logger.warning("Category '%s' not in CATEGORY_PROFILE — using default", category)
        return DEFAULT_PROFILE
    return prof


def _product_types(category: str) -> tuple[str, ...]:
    return PRODUCT_TYPES.get(category, DEFAULT_PRODUCT_TYPES)


def _reason_weights(category: str) -> tuple[list[str], np.ndarray]:
    """Return (reason labels, normalised probabilities) for a category."""
    table = CATEGORY_RETURN_REASONS.get(category, DEFAULT_RETURN_REASONS)
    names = list(table)
    probs = np.array([table[k] for k in names], dtype=float)
    return names, probs / probs.sum()


def _allocate(n_total: int, weights: np.ndarray) -> np.ndarray:
    """Split n_total across groups proportionally to weights (largest remainder)."""
    if len(weights) == 0 or (weights <= 0).any():
        raise ValueError("allocation weights must be non-empty and positive")
    quota = weights / weights.sum() * n_total
    counts = np.floor(quota).astype(int)
    shortfall = n_total - int(counts.sum())
    if shortfall > 0:
        for pos in np.argsort(quota - counts)[::-1][:shortfall]:
            counts[pos] += 1
    return counts


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def generate_products(
    n_products: int, categories: list[str], rng: np.random.Generator
) -> pd.DataFrame:
    """Build the product catalogue: one row per SKU with price, cost and
    popularity (drives how often each SKU is ordered)."""
    if n_products < len(categories):
        raise ValueError(
            f"n_products ({n_products}) must be >= number of categories "
            f"({len(categories)}); every category needs at least one SKU."
        )
    shares = np.array([float(_profile(c)["order_share"]) for c in categories])
    per_category = _allocate(n_products, shares)

    frames: list[pd.DataFrame] = []
    sku_start = 0
    for category, count in zip(categories, per_category):
        prof = _profile(category)
        lo, hi = prof["price_range"]
        clo, chi = prof["cost_ratio"]
        count = int(count)

        prices = np.round(rng.uniform(lo, hi, count), 2)
        costs = np.round(prices * rng.uniform(clo, chi, count), 2)
        popularity = rng.lognormal(0.0, 0.8, count)
        rates = np.clip(float(prof["return_rate"]) * rng.uniform(0.6, 1.4, count), 0.0, 0.9)
        types = rng.choice(np.array(_product_types(category), dtype=object), size=count)
        brands = rng.choice(np.array(BRANDS, dtype=object), size=count)
        models = rng.choice(np.array(MODEL_SUFFIXES, dtype=object), size=count)

        frames.append(pd.DataFrame({
            "product_id": [f"SKU-{sku_start + i + 1:04d}" for i in range(count)],
            "product_name": [f"{b} {t} {m}" for b, t, m in zip(brands, types, models)],
            "category": category,
            "unit_price": prices,
            "unit_cost": costs,
            "base_return_rate": rates,
            "popularity": popularity,
        }))
        sku_start += count

    products = pd.concat(frames, ignore_index=True)

    # Plant one problem SKU per category: its most popular product returns ~3x.
    for category, group in products.groupby("category", sort=False):
        top = group["popularity"].idxmax()
        planted = min(
            PROBLEM_SKU_RATE_CAP,
            float(_profile(category)["return_rate"]) * PROBLEM_SKU_MULTIPLIER,
        )
        products.at[top, "base_return_rate"] = planted
        logger.info("Planted problem SKU %s (%s) | return rate %.0f%%",
                    products.at[top, "product_id"], category, planted * 100)

    logger.info("Product catalogue: %d SKUs across %d categories",
                len(products), products["category"].nunique())
    return products


def generate_customers(
    n_customers: int, regions: list[str], rng: np.random.Generator
) -> pd.DataFrame:
    """Customer master: id, region, activity weight, and churn point.

    ``churn_frac`` in (0, 1] is the fraction of the timeline the customer
    stays active; 1.0 means they never churn.
    """
    if n_customers <= 0:
        raise ValueError("n_customers must be positive")
    if not regions:
        raise ValueError("config 'generation.regions' is empty — add at least one region")

    region_arr = np.array(regions, dtype=object)
    weights = np.array([REGION_WEIGHTS.get(r, DEFAULT_REGION_WEIGHT) for r in regions])
    sampled_regions = rng.choice(region_arr, size=n_customers, p=weights / weights.sum())

    activity = rng.lognormal(0.0, 1.0, n_customers)  # whales vs occasional buyers
    u = rng.random(n_customers)
    churn_frac = np.where(u < CHURN_FRACTION, rng.random(n_customers), 1.0)

    customers = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(1, n_customers + 1)],
        "region": sampled_regions,
        "activity_weight": activity,
        "churn_frac": churn_frac,
    })

    if DECLINING_REGION in regions:
        logger.info("Regional story: '%s' order share decays to ~%.0f%% of its "
                    "start by the final month", DECLINING_REGION,
                    (1.0 - REGION_DECAY_PER_MONTH * 11) * 100)
    else:
        logger.warning("Declining-region story disabled: '%s' not in configured "
                       "regions", DECLINING_REGION)
    logger.info("Customer master: %d customers | %.0f%% churn mid-year",
                n_customers, CHURN_FRACTION * 100)
    return customers


def generate_orders(
    products: pd.DataFrame,
    customers: pd.DataFrame,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate the clean order fact table (before any issues are injected)."""
    start, end = pd.Timestamp(cfg["start_date"]), pd.Timestamp(cfg["end_date"])
    n_orders = int(cfg["n_orders"])
    if n_orders <= 0:
        raise ValueError("generation.n_orders must be positive")
    if end <= start:
        raise ValueError(f"end_date ({cfg['end_date']}) must be after "
                         f"start_date ({cfg['start_date']})")

    months = pd.period_range(start, end, freq="M")
    seasonality = np.asarray(cfg["monthly_seasonality"], dtype=float)
    if len(seasonality) != len(months):
        raise ValueError(
            f"monthly_seasonality has {len(seasonality)} entries but "
            f"{start.date()} -> {end.date()} spans {len(months)} months — "
            f"update config.yaml so the lengths match."
        )
    if (seasonality <= 0).any():
        raise ValueError("monthly_seasonality entries must all be positive")

    counts = rng.multinomial(n_orders, seasonality / seasonality.sum())

    # Product sampling weights: popularity x category order share.
    cat_share = {c: float(_profile(c)["order_share"]) for c in products["category"].unique()}
    w = products["popularity"].to_numpy() * np.array(
        [cat_share[c] for c in products["category"]])
    p_prod = w / w.sum()

    cust_ids = customers["customer_id"].to_numpy(dtype=object)
    cust_region = customers["region"].to_numpy(dtype=object)
    activity = customers["activity_weight"].to_numpy(dtype=float)
    churn_frac = customers["churn_frac"].to_numpy(dtype=float)

    prod_ids = products["product_id"].to_numpy(dtype=object)
    prod_names = products["product_name"].to_numpy(dtype=object)
    prod_cat = products["category"].to_numpy(dtype=object)
    prod_price = products["unit_price"].to_numpy(dtype=float)
    prod_cost = products["unit_cost"].to_numpy(dtype=float)
    prod_rate = products["base_return_rate"].to_numpy(dtype=float)

    blocks: list[pd.DataFrame] = []
    n_months = len(months)
    for m, (period, n_m) in enumerate(zip(months, counts)):
        n_m = int(n_m)
        if n_m == 0:
            continue

        # Who orders this month: activity x region decay x churn eligibility.
        active = churn_frac >= (m + 1) / n_months
        decay = np.where(
            cust_region == DECLINING_REGION,
            max(0.0, 1.0 - REGION_DECAY_PER_MONTH * m),
            1.0,
        )
        w_cust = activity * decay * active
        if w_cust.sum() <= 0:
            raise ValueError(f"No eligible customers in {period}; check churn/region settings.")
        cust_idx = rng.choice(len(customers), size=n_m, p=w_cust / w_cust.sum())

        # Which day: weekends slightly heavier.
        days = pd.date_range(period.start_time, period.end_time)
        day_w = np.where(days.dayofweek.to_numpy() >= 5, WEEKEND_WEIGHT, 1.0)
        day_pos = rng.choice(len(days), size=n_m, p=day_w / day_w.sum())

        # What they buy, how many, at what discount.
        prod_idx = rng.choice(len(products), size=n_m, p=p_prod)
        qty = rng.choice(QUANTITY_CHOICES, size=n_m, p=QUANTITY_P)
        p_disc = DISCOUNT_PROB_PEAK if m in PEAK_MONTHS else DISCOUNT_PROB_BASE
        disc = np.where(
            rng.random(n_m) < p_disc,
            np.round(rng.uniform(DISCOUNT_RANGE[0], DISCOUNT_RANGE[1], n_m), 1),
            0.0,
        )

        # Returns and their reasons (NaN reason = not returned: structural).
        ret = rng.random(n_m) < prod_rate[prod_idx]
        reason = np.full(n_m, np.nan, dtype=object)
        for cat in np.unique(prod_cat):
            mask = ret & (prod_cat[prod_idx] == cat)
            if not mask.any():
                continue
            names, probs = _reason_weights(cat)
            reason[mask] = rng.choice(
                np.array(names, dtype=object), size=int(mask.sum()), p=probs)

        blocks.append(pd.DataFrame({
            "order_date": days.to_numpy()[day_pos],
            "customer_id": cust_ids[cust_idx],
            "customer_region": cust_region[cust_idx],
            "product_id": prod_ids[prod_idx],
            "product_name": prod_names[prod_idx],
            "category": prod_cat[prod_idx],
            "quantity": qty,
            "unit_price": prod_price[prod_idx],
            "unit_cost": prod_cost[prod_idx],
            "discount_pct": disc,
            "returned": ret,
            "return_reason": reason,
        }))

    if not blocks:
        raise ValueError("No orders generated — check n_orders and the date range.")

    orders = pd.concat(blocks, ignore_index=True)
    orders = orders.sort_values("order_date", kind="stable").reset_index(drop=True)
    orders.insert(0, "order_id", [f"ORD-{i:06d}" for i in range(1, len(orders) + 1)])
    orders["order_date"] = orders["order_date"].dt.strftime("%Y-%m-%d")

    logger.info("Generated %d clean orders across %d months", len(orders), n_months)
    return orders


def inject_data_issues(
    df: pd.DataFrame, rng: np.random.Generator, dirty_fraction: float
) -> pd.DataFrame:
    """Inject realistic data-quality issues into ~``dirty_fraction`` of rows.

    Each dirty row receives exactly one issue type (chunks are disjoint);
    duplicates are appended, so the file grows. Returns a new frame — the
    caller's DataFrame is never mutated.
    """
    if not 0.0 <= dirty_fraction <= 1.0:
        raise ValueError(f"dirty_fraction must be in [0, 1], got {dirty_fraction}")

    out = df.copy()
    n = len(out)
    budget = int(round(n * dirty_fraction))
    if budget == 0:
        logger.warning("dirty_row_fraction gave a zero budget — file saved clean")
        return out

    targets = rng.choice(np.arange(n), size=budget, replace=False)

    names = list(ISSUE_MIX)
    fracs = np.array([ISSUE_MIX[k] for k in names], dtype=float)
    counts = np.floor(fracs / fracs.sum() * budget).astype(int)
    counts[0] += budget - int(counts.sum())  # rounding remainder -> first issue

    cursor, appended = 0, 0
    for name, cnt in zip(names, counts):
        idx = targets[cursor:cursor + cnt]
        cursor += cnt
        if len(idx) == 0:
            continue
        if name == "missing_price":
            out.loc[idx, "unit_price"] = np.nan
        elif name == "missing_region":
            out.loc[idx, "customer_region"] = np.nan
        elif name == "negative_quantity":
            out.loc[idx, "quantity"] = -out.loc[idx, "quantity"].abs().to_numpy()
        elif name == "out_of_range_price":
            half = len(idx) // 2
            out.loc[idx[:half], "unit_price"] = 0.0        # below min_valid_price
            out.loc[idx[half:], "unit_price"] = 999_999.99  # above max_valid_price
        elif name == "malformed_date":
            junk = rng.choice(np.array(MALFORMED_DATES, dtype=object), size=len(idx))
            out.loc[idx, "order_date"] = junk
        elif name == "duplicate_row":
            out = pd.concat([out, out.loc[idx]], ignore_index=True)
            appended += len(idx)
        logger.info("Injected issue %-18s -> %d rows", name, len(idx))

    out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    logger.info("Data issues: %d/%d rows (%.1f%%) | %d duplicates appended "
                "| file now %d rows", budget, n, 100 * budget / n, appended, len(out))
    return out


def save_raw_data(df: pd.DataFrame, path: Path | str) -> Path:
    """Write the raw fact table to CSV and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved raw dataset -> %s (%d rows x %d cols, %.1f MB)",
                path, len(df), df.shape[1], path.stat().st_size / (1024 * 1024))
    return path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def _validate_generation_config(g: dict[str, Any]) -> None:
    """Fail fast on impossible config instead of dying mid-generation."""
    required = ("seed", "n_products", "n_customers", "n_orders", "start_date",
                "end_date", "dirty_row_fraction", "monthly_seasonality",
                "categories", "regions")
    missing = [k for k in required if k not in g]
    if missing:
        raise ValueError(f"config.yaml is missing generation keys: {', '.join(missing)}")
    if not g["categories"] or not g["regions"]:
        raise ValueError("generation.categories and generation.regions must be non-empty")
    if int(g["n_orders"]) <= 0 or int(g["n_customers"]) <= 0:
        raise ValueError("n_orders and n_customers must be positive")
    if int(g["n_products"]) < len(g["categories"]):
        raise ValueError("n_products must be >= number of categories")
    if pd.Timestamp(g["end_date"]) <= pd.Timestamp(g["start_date"]):
        raise ValueError("end_date must be after start_date")
    if not 0.0 <= float(g["dirty_row_fraction"]) <= 1.0:
        raise ValueError("dirty_row_fraction must be in [0, 1]")


def _log_clean_summary(orders: pd.DataFrame) -> None:
    """Log the 'true' baseline before injection (for README section 6)."""
    gross = float((orders["quantity"] * orders["unit_price"]
                   * (1 - orders["discount_pct"] / 100)).sum())
    logger.info("Clean baseline | rows=%d | customers=%d | SKUs=%d | "
                "span=%s -> %s | gross revenue ~ $%.2fM | return rate %.1f%%",
                len(orders), orders["customer_id"].nunique(),
                orders["product_id"].nunique(), orders["order_date"].min(),
                orders["order_date"].max(), gross / 1e6,
                100 * orders["returned"].mean())


def main() -> Path:
    """Run Step 1 end-to-end; return the raw CSV artifact path."""
    cfg = get_config()
    g = cfg["generation"]
    _validate_generation_config(g)

    rng = np.random.default_rng(int(g["seed"]))
    logger.info("Generating dataset | seed=%s | %s products | %s customers | "
                "%s orders | %s -> %s", g["seed"], g["n_products"],
                g["n_customers"], g["n_orders"], g["start_date"], g["end_date"])

    products = generate_products(int(g["n_products"]), list(g["categories"]), rng)
    customers = generate_customers(int(g["n_customers"]), list(g["regions"]), rng)
    orders = generate_orders(products, customers, g, rng)
    _log_clean_summary(orders)

    dirty = inject_data_issues(orders, rng, float(g["dirty_row_fraction"]))
    return save_raw_data(dirty, Path(cfg["paths"]["raw_data"]))


if __name__ == "__main__":
    setup_logging()
    main()
