"""src/analysis.py — Step 3: KPIs, trends, ranked insights, executive memo.

Pipeline position: consumes data/processed_sales.parquet (src.data_processing),
produces insights/executive_memo.md, plus an in-memory AnalysisResult that
src.visualize and app.py render (one computation, one truth).

Insight engine — six candidate generators, ranked by dollars at risk:

    kind                 fires when                                        config knob
    -------------------  ------------------------------------------------  ---------------------
    returns_category     category return rate >= high_return_threshold     high_return_threshold
    returns_sku          SKU in top_return_drivers (rate + volume filter)  min_sku_volume
    region_decline       H1 -> H2 revenue-share drop >= 2pp                (module floor)
    seasonal_drop        MoM net revenue <= mom_decline_threshold          mom_decline_threshold
    churn_at_risk        'At Risk' segment holds >= 0.5% of revenue        (module floors)
    discount_compression Nov-Dec margin >= 1.5pp thinner than rest of year  (module floor)

Ranking rule: sort by impact_usd desc, keep top N (top_n_insights). One slot
is GUARANTEED to the top returns_sku candidate (product-level actionability
is this portfolio's core promise) — the swap is logged, and can be disabled
via GUARANTEE_SKU_SLOT. Per-kind impact definitions are printed in the memo's
methodology section so every dollar figure is defensible.

Module contract (see run_pipeline.py): main() -> Path (memo artifact).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_config, get_logger, setup_logging
from src.data_processing import DataQualityError

logger = get_logger(__name__)

__all__ = [
    "Insight", "AnalysisResult",
    "load_processed_data", "compute_kpis", "monthly_trend",
    "detect_seasonal_drops", "category_performance", "regional_performance",
    "top_return_drivers", "segment_summary", "rank_insights", "analyze",
    "generate_executive_memo", "main",
]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REQUIRED_PROCESSED_COLUMNS: tuple[str, ...] = (
    "order_id", "order_date", "month", "customer_id", "customer_region",
    "customer_segment", "product_id", "product_name", "category", "quantity",
    "unit_cost", "discount_pct", "returned", "return_flag", "return_reason",
    "gross_revenue", "net_revenue", "cogs", "revenue_lost_to_returns",
    "net_profit",
)

# Insight-generator floors (module constants: judgment calls, not tunables).
REGION_SHARE_DROP_FLOOR_PP = 2.0     # H1->H2 share loss needed to fire
DISCOUNT_COMPRESSION_FLOOR_PP = 1.5  # peak-vs-rest margin gap needed to fire
CHURN_MIN_CUSTOMERS = 10
CHURN_MIN_REVENUE_SHARE = 0.005
SEASONAL_MAX_INSIGHTS = 2            # don't flood the memo with drop months
PEAK_MONTHS = (11, 12)               # Nov, Dec (1-based month numbers)
GUARANTEE_SKU_SLOT = True             # reserve one top-N slot for top SKU

#: Per-kind definition of "dollars at risk" — printed in the memo methodology.
KIND_IMPACT_NOTES: dict[str, str] = {
    "returns_category": "refunded revenue + written-off COGS on that category's returns",
    "returns_sku": "refunded revenue + written-off COGS on that SKU's returns",
    "region_decline": "H2 net-revenue gap vs holding the region's H1 share",
    "seasonal_drop": "net-revenue shortfall vs the prior month",
    "churn_at_risk": "period net revenue of customers inactive 90+ days",
    "discount_compression": "margin points lost in Nov-Dec x peak net revenue",
}

#: Top return reason -> the commercial action it implies (data-driven actions).
REASON_ACTIONS: dict[str, str] = {
    "Wrong size or fit":
        "Deploy accurate size charts and fit notes on every listing; add a "
        "pre-purchase size quiz",
    "Damaged on arrival":
        "Audit packaging and carrier handling; sample-inspect 20 units from "
        "the affected vendor",
    "Not as described":
        "Tighten listing accuracy (photos, dimensions, materials); enforce a "
        "content SLA with sellers",
    "Changed mind":
        "Add richer product imagery and video to reduce regret purchases",
    "Found better price":
        "Review price competitiveness and the price-match policy on top movers",
    "Quality below expectations":
        "Re-negotiate vendor QC thresholds; batch-test incoming inventory",
}


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Insight:
    """One ranked, actionable finding. impact_usd is the ranking key."""
    key: str            # stable slug, e.g. "returns_sku::SKU-00008"
    kind: str           # one of KIND_IMPACT_NOTES
    title: str          # headline for the memo
    tldr: str           # one-line action summary for the TL;DR list
    impact_usd: float   # dollars at risk (definition depends on kind)
    evidence: tuple[str, ...]
    action: str         # the concrete "do this"
    owner: str
    horizon: str = "This week"


@dataclass(frozen=True)
class AnalysisResult:
    """Everything downstream (memo, charts, dashboard) renders from this.

    Built once by analyze(); app.py and visualize.py consume it so all
    artifacts show identical numbers.
    """
    kpis: dict[str, Any]
    monthly: pd.DataFrame
    seasonal_drops: list[dict[str, Any]]
    categories: pd.DataFrame
    regions: pd.DataFrame
    return_drivers: pd.DataFrame
    segments: pd.DataFrame
    insights: list[Insight]
    meta: dict[str, Any]


# --------------------------------------------------------------------------- #
# Formatting helpers (pure -> easy unit-test targets)
# --------------------------------------------------------------------------- #

def _usd(x: float) -> str:
    x = float(x)
    if pd.isna(x):
        return "—"
    if abs(x) >= 1_000_000:
        return f"${x / 1e6:.2f}M"
    if abs(x) >= 10_000:
        return f"${x / 1e3:.0f}K"
    if abs(x) >= 1_000:
        return f"${x / 1e3:.1f}K"
    return f"${x:,.0f}"


def _pct(x: float, dp: int = 1) -> str:
    x = float(x)
    return f"{x:.{dp}f}%" if pd.notna(x) else "—"


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def _mode_reason(s: pd.Series) -> str:
    """Most common non-null return reason in a group ('' if none)."""
    s = s.dropna()
    if s.empty:
        return ""
    return str(s.value_counts().index[0])


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #

def load_processed_data(path: Path | str) -> pd.DataFrame:
    """Read the processed parquet, validate columns, coerce dtypes."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"processed data not found: {path} | run it first: "
            f"python run_pipeline.py --stages process")
    df = pd.read_parquet(path)

    missing = [c for c in REQUIRED_PROCESSED_COLUMNS if c not in df.columns]
    if missing:
        raise DataQualityError(
            f"processed data is missing column(s): {', '.join(missing)} — "
            f"regenerate with: python run_pipeline.py --stages process")
    if df.empty:
        raise ValueError(f"processed data at {path} has 0 rows — regenerate it")

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["returned"] = df["returned"].astype(bool)
    logger.info("Loaded processed data: %d rows x %d cols from %s",
                len(df), df.shape[1], path)
    return df


# --------------------------------------------------------------------------- #
# Core metrics (public API, per README §3)
# --------------------------------------------------------------------------- #

def compute_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """Company-level headline metrics for the full dataset period."""
    net = float(df["net_revenue"].sum())
    profit = float(df["net_profit"].sum())
    refunded = float(df["revenue_lost_to_returns"].sum())
    writeoff = float((df["cogs"] * df["return_flag"]).sum())
    orders = int(len(df))
    return {
        "period_start": df["order_date"].min(),
        "period_end": df["order_date"].max(),
        "months": int(df["order_date"].dt.to_period("M").nunique()),
        "orders": orders,
        "customers": int(df["customer_id"].nunique()),
        "products": int(df["product_id"].nunique()),
        "categories": int(df["category"].nunique()),
        "units_sold": int(df["quantity"].sum()),
        "gross_revenue": float(df["gross_revenue"].sum()),
        "net_revenue": net,
        "cogs": float(df["cogs"].sum()),
        "net_profit": profit,
        "margin_pct": 100.0 * _safe_ratio(profit, net),
        "aov": _safe_ratio(net, orders),
        "returned_orders": int(df["return_flag"].sum()),
        "return_rate_pct": 100.0 * _safe_ratio(float(df["return_flag"].sum()), orders),
        "refunded_revenue": refunded,
        "writeoff_cogs": writeoff,
        "returns_cost": refunded + writeoff,
        "returns_cost_pct_of_net": 100.0 * _safe_ratio(refunded + writeoff, net),
        "avg_discount_pct": float(df["discount_pct"].mean()),
    }


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly revenue/profit/return trend with MoM change and prior values."""
    period = df["order_date"].dt.to_period("M")
    g = (df.groupby(period)
           .agg(orders=("order_id", "count"),
                gross_revenue=("gross_revenue", "sum"),
                net_revenue=("net_revenue", "sum"),
                net_profit=("net_profit", "sum"),
                returned_orders=("return_flag", "sum"),
                avg_discount_pct=("discount_pct", "mean")))
    g.index.name = "order_year_month"
    g = g.sort_index()

    out = g.reset_index()
    out["order_year_month"] = out["order_year_month"].astype(str)
    out["prev_net_revenue"] = g["net_revenue"].shift(1).to_numpy()
    out["prev_orders"] = g["orders"].shift(1).to_numpy()
    out["mom_net_pct"] = (g["net_revenue"].pct_change() * 100).to_numpy()
    out["return_rate_pct"] = [
        100.0 * _safe_ratio(r, o) for r, o in zip(out["returned_orders"], out["orders"])]
    out["month_label"] = pd.to_datetime(out["order_year_month"] + "-01").dt.strftime("%b %y")

    logger.info("Monthly trend: %d months, %s -> %s",
                len(out), out["order_year_month"].iat[0], out["order_year_month"].iat[-1])
    return out


def detect_seasonal_drops(
    monthly: pd.DataFrame, mom_threshold: float = -0.10
) -> list[dict[str, Any]]:
    """Flag months whose MoM net revenue <= mom_threshold (a fraction, negative).

    Returns dicts sorted by revenue shortfall, largest first.
    """
    if mom_threshold >= 0:
        raise ValueError("mom_threshold must be negative (it is a decline), e.g. -0.10")
    drops: list[dict[str, Any]] = []
    for row in monthly.itertuples():
        mom = row.mom_net_pct
        if pd.isna(mom) or mom > mom_threshold * 100:
            continue
        drops.append({
            "order_year_month": row.order_year_month,
            "month_label": row.month_label,
            "mom_pct": float(mom),
            "prev_net_revenue": float(row.prev_net_revenue),
            "net_revenue": float(row.net_revenue),
            "prev_orders": int(row.prev_orders),
            "orders": int(row.orders),
            "shortfall": float(row.prev_net_revenue - row.net_revenue),
        })
    drops.sort(key=lambda d: d["shortfall"], reverse=True)
    if drops:
        logger.info("Seasonal drops detected: %s",
                    ", ".join(f"{d['month_label']} ({d['mom_pct']:.1f}%)" for d in drops))
    else:
        logger.info("No month fell below the %.0f%% MoM decline threshold",
                    mom_threshold * 100)
    return drops


def category_performance(df: pd.DataFrame, min_volume: int = 50) -> pd.DataFrame:
    """Revenue / margin / return economics per category (with low-volume flag)."""
    tmp = df.assign(
        _writeoff=df["cogs"] * df["return_flag"],
        _reason=df["return_reason"].where(df["return_flag"] == 1),
    )
    g = tmp.groupby("category").agg(
        orders=("order_id", "count"),
        net_revenue=("net_revenue", "sum"),
        net_profit=("net_profit", "sum"),
        refunded=("revenue_lost_to_returns", "sum"),
        writeoff=("_writeoff", "sum"),
        returned_orders=("return_flag", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        top_return_reason=("_reason", _mode_reason),
    )
    g["returns_cost"] = g["refunded"] + g["writeoff"]
    total_net = float(g["net_revenue"].sum())
    g["net_revenue_share_pct"] = [100.0 * _safe_ratio(v, total_net) for v in g["net_revenue"]]
    g["margin_pct"] = [100.0 * _safe_ratio(p, r)
                       for p, r in zip(g["net_profit"], g["net_revenue"])]
    g["return_rate_pct"] = [100.0 * _safe_ratio(r, o)
                            for r, o in zip(g["returned_orders"], g["orders"])]
    g["low_volume"] = g["orders"] < min_volume

    out = g.reset_index().sort_values("net_revenue", ascending=False, ignore_index=True)
    logger.info("Category performance: %d categories | top: %s (%s net)",
                len(out), out["category"].iat[0], _usd(out["net_revenue"].iat[0]))
    return out


def regional_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Regional economics plus H1 vs H2 revenue share (the drop-off detector)."""
    g = df.groupby("customer_region").agg(
        orders=("order_id", "count"),
        customers=("customer_id", "nunique"),
        net_revenue=("net_revenue", "sum"),
        returned_orders=("return_flag", "sum"),
    )
    total_net = float(g["net_revenue"].sum())
    g["net_revenue_share_pct"] = [100.0 * _safe_ratio(v, total_net) for v in g["net_revenue"]]
    g["return_rate_pct"] = [100.0 * _safe_ratio(r, o)
                            for r, o in zip(g["returned_orders"], g["orders"])]

    halves = df.assign(_h=np.where(df["month"] <= 6, "H1", "H2"))
    piv = halves.groupby(["_h", "customer_region"])["net_revenue"].sum().unstack("customer_region")
    if {"H1", "H2"} <= set(piv.index):
        shares = piv.div(piv.sum(axis=1), axis=0) * 100
        g["share_h1_pct"] = shares.loc["H1"]
        g["share_h2_pct"] = shares.loc["H2"]
        g["share_change_pp"] = shares.loc["H2"] - shares.loc["H1"]
    else:
        for col in ("share_h1_pct", "share_h2_pct", "share_change_pp"):
            g[col] = np.nan
        logger.warning("Regional half-year shares skipped: data does not span both halves")

    out = g.reset_index().sort_values("net_revenue", ascending=False, ignore_index=True)
    logger.info("Regional performance: %d regions | top: %s (%s net)",
                len(out), out["customer_region"].iat[0], _usd(out["net_revenue"].iat[0]))
    return out


def top_return_drivers(
    df: pd.DataFrame,
    top_n: int = 5,
    min_volume: int = 20,
    high_return_threshold_pct: float = 15.0,
) -> pd.DataFrame:
    """SKUs ranked by returns cost (refund + write-off) among eligible SKUs.

    Eligibility: enough orders (min_volume) and a return rate at or above
    the threshold — the low-volume guard prevents noisy small-sample SKUs.
    """
    tmp = df.assign(
        _writeoff=df["cogs"] * df["return_flag"],
        _reason=df["return_reason"].where(df["return_flag"] == 1),
    )
    g = tmp.groupby(["product_id", "product_name", "category"]).agg(
        orders=("order_id", "count"),
        units=("quantity", "sum"),
        net_revenue=("net_revenue", "sum"),
        refunded=("revenue_lost_to_returns", "sum"),
        writeoff=("_writeoff", "sum"),
        returned_orders=("return_flag", "sum"),
        top_return_reason=("_reason", _mode_reason),
    )
    g["returns_cost"] = g["refunded"] + g["writeoff"]
    g["return_rate_pct"] = [100.0 * _safe_ratio(r, o)
                            for r, o in zip(g["returned_orders"], g["orders"])]

    eligible = g[(g["orders"] >= min_volume)
                 & (g["return_rate_pct"] >= high_return_threshold_pct)]
    out = (eligible.sort_values("returns_cost", ascending=False)
                  .head(top_n).reset_index())
    logger.info("Return drivers: %d SKUs above threshold -> keeping top %d "
                "(largest: %s, %s)",
                len(eligible), len(out),
                out["product_id"].iat[0] if len(out) else "none",
                _usd(out["returns_cost"].iat[0]) if len(out) else "—")
    return out


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """RFM-lite segment mix: customers, orders, revenue and revenue share."""
    g = df.groupby("customer_segment").agg(
        customers=("customer_id", "nunique"),
        orders=("order_id", "count"),
        net_revenue=("net_revenue", "sum"),
    )
    total_net = float(g["net_revenue"].sum())
    g["net_revenue_share_pct"] = [100.0 * _safe_ratio(v, total_net) for v in g["net_revenue"]]
    out = g.reset_index().sort_values("net_revenue", ascending=False, ignore_index=True)
    logger.info("Segment summary: %s",
                ", ".join(f"{r.customer_segment} {_usd(r.net_revenue)}"
                          for r in out.itertuples()))
    return out


# --------------------------------------------------------------------------- #
# Insight generators (each returns candidates; floors prevent noise)
# --------------------------------------------------------------------------- #

def _category_return_insights(
    cats: pd.DataFrame, kpis: dict[str, Any], threshold_pct: float
) -> list[Insight]:
    company = float(kpis["return_rate_pct"])
    total_returned = int(kpis["returned_orders"])
    out: list[Insight] = []
    for row in cats.itertuples():
        if row.return_rate_pct < threshold_pct or row.low_volume:
            continue
        mult = _safe_ratio(row.return_rate_pct, company)
        share_of_returns = 100.0 * _safe_ratio(row.returned_orders, total_returned)
        reason = row.top_return_reason or "Changed mind"
        action = REASON_ACTIONS.get(reason, REASON_ACTIONS["Changed mind"])
        out.append(Insight(
            key=f"returns_category::{row.category}",
            kind="returns_category",
            title=(f"{row.category} returns at {_pct(row.return_rate_pct)} — "
                   f"{mult:.1f}x the company average"),
            tldr=f"Fix {row.category} returns (top cause: '{reason}')",
            impact_usd=float(row.returns_cost),
            evidence=(
                f"Return rate {_pct(row.return_rate_pct)} vs {_pct(company)} "
                f"company average ({mult:.1f}x)",
                f"{share_of_returns:.0f}% of ALL company returns come from {row.category}",
                f"Cost this period: {_usd(row.refunded)} refunds + "
                f"{_usd(row.writeoff)} written-off inventory",
            ),
            action=(f"{action} — across all {row.category} listings; re-measure the "
                    f"return rate after 4 weeks; escalate to vendor renegotiation if "
                    f"it does not fall below {_pct(threshold_pct)}."),
            owner="Merchandising + Vendor QC",
        ))
    return out


def _sku_return_insights(
    drivers: pd.DataFrame, kpis: dict[str, Any]
) -> list[Insight]:
    company = float(kpis["return_rate_pct"])
    return [Insight(
        key=f"returns_sku::{row.product_id}",
        kind="returns_sku",
        title=(f"{row.product_name} ({row.product_id}) — "
               f"{_pct(row.return_rate_pct)} returns on {row.orders:,} orders"),
        tldr=f"Pull {row.product_id} from promos and audit its listing",
        impact_usd=float(row.returns_cost),
        evidence=(
            f"Return rate {_pct(row.return_rate_pct)} vs {_pct(company)} company average",
            f"Top return reason: '{row.top_return_reason or 'Not as described'}'",
            f"{_usd(row.returns_cost)} at risk: {_usd(row.refunded)} refunds + "
            f"{_usd(row.writeoff)} write-offs on {int(row.returned_orders):,} returned orders",
        ),
        action=(f"Pull {row.product_id} from promo rotation today; audit the listing "
                f"against the physical product (photos, specs, sizing); sample-QC 20 "
                f"units; open a vendor claim if defect-driven; delist if unresolved "
                f"in 3 weeks."),
        owner="Category merch lead",
    ) for row in drivers.itertuples()]


def _regional_decline_insight(df: pd.DataFrame) -> Insight | None:
    """Fire on the region with the largest H1 -> H2 share collapse."""
    h1 = df[df["month"] <= 6]
    h2 = df[df["month"] > 6]
    if h1.empty or h2.empty:
        logger.info("Regional-decline insight skipped: needs both halves of the year")
        return None

    r1 = h1.groupby("customer_region")["net_revenue"].sum()
    r2 = h2.groupby("customer_region")["net_revenue"].sum()
    s1, s2 = r1 / r1.sum(), r2 / r2.sum()
    gap = (s1 - s2) * float(r2.sum())  # expected-at-H1-share minus actual H2
    region = gap.idxmax()
    pp_drop = float((s1[region] - s2[region]) * 100)
    if pp_drop < REGION_SHARE_DROP_FLOOR_PP or float(gap[region]) <= 0:
        logger.info("Regional-decline insight skipped: max share drop %.1fpp "
                    "< %.1fpp floor", pp_drop, REGION_SHARE_DROP_FLOOR_PP)
        return None

    o1 = int(h1.loc[h1["customer_region"] == region, "order_id"].count())
    o2 = int(h2.loc[h2["customer_region"] == region, "order_id"].count())
    c1, c2 = int(len(h1)), int(len(h2))
    trend_word = "grew" if c2 > c1 else "shrank" if c2 < c1 else "held steady"
    h2r = h2[h2["customer_region"] == region]
    h2o = h2[h2["customer_region"] != region]
    rr_region = 100.0 * _safe_ratio(float(h2r["return_flag"].sum()), len(h2r))
    rr_others = 100.0 * _safe_ratio(float(h2o["return_flag"].sum()), len(h2o))

    return Insight(
        key=f"region_decline::{region}",
        kind="region_decline",
        title=f"{region} revenue share collapsed {pp_drop:.1f}pp from H1 to H2",
        tldr=f"Audit {region} ops, then run a regional win-back test",
        impact_usd=float(gap[region]),
        evidence=(
            f"Share of company net revenue: {_pct(s1[region] * 100)} in H1 → "
            f"{_pct(s2[region] * 100)} in H2 (−{pp_drop:.1f}pp)",
            f"Orders in {region}: {o1:,} (H1) → {o2:,} (H2), while the company "
            f"overall {trend_word} ({c1:,} → {c2:,})",
            f"H2 revenue gap vs holding H1 share: {_usd(gap[region])} · return rate "
            f"in {region} ({_pct(rr_region)}) is in line with the rest "
            f"({_pct(rr_others)}) → a demand/ops problem, not product quality",
        ),
        action=(f"Before blaming demand, audit {region} ops: delivery times vs other "
                f"regions, stockouts, local competitor promos. Then run a 4-week "
                f"win-back test (10% coupon) on lapsed {region} customers. Add a "
                f"monthly regional-share alert so this never slips silently again."),
        owner="Regional ops / Growth",
        horizon="This month",
    )


def _seasonal_drop_insights(drops: list[dict[str, Any]]) -> list[Insight]:
    out: list[Insight] = []
    for rank, d in enumerate(drops[:SEASONAL_MAX_INSIGHTS]):
        mom = float(d["mom_pct"])
        evidence = [
            f"Net revenue {_usd(d['prev_net_revenue'])} → {_usd(d['net_revenue'])} "
            f"({mom:.1f}% MoM)",
            f"Orders {d['prev_orders']:,} → {d['orders']:,}",
        ]
        if rank == 0:
            evidence.append(f"Shortfall {_usd(d['shortfall'])} — the largest "
                            f"single-month decline this period")
        out.append(Insight(
            key=f"seasonal_drop::{d['order_year_month']}",
            kind="seasonal_drop",
            title=f"{d['month_label']} net revenue fell {mom:.1f}% month-over-month",
            tldr=f"Plan ahead for the {d['month_label'].split()[0]} trough",
            impact_usd=float(d["shortfall"]),
            evidence=tuple(evidence),
            action=("Treat the trough as structural, not an accident: load a "
                    "bundle/loyalty campaign two weeks before it hits; trim "
                    "replenishment on return-heavy SKUs to protect cash; use the "
                    "quiet window for the vendor-QC fixes in the actions above."),
            owner="Planning / Merch",
            horizon="Next cycle",
        ))
    return out


def _churn_insight(df: pd.DataFrame, kpis: dict[str, Any]) -> Insight | None:
    """'At Risk' customers (90+ days silent) and the revenue they represent."""
    sub = df.loc[df["customer_segment"] == "At Risk"]
    n = int(sub["customer_id"].nunique()) if not sub.empty else 0
    rev = float(sub["net_revenue"].sum()) if not sub.empty else 0.0
    total_rev = float(kpis["net_revenue"])
    if n < CHURN_MIN_CUSTOMERS or _safe_ratio(rev, total_rev) < CHURN_MIN_REVENUE_SHARE:
        logger.info("Churn insight skipped: %d at-risk customers, %s revenue",
                    n, _usd(rev))
        return None

    snapshot = df["order_date"].max()
    last = sub.groupby("customer_id")["order_date"].max()
    avg_silence = float((snapshot - last).dt.days.mean())
    top_cat = str(sub.groupby("category")["net_revenue"].sum().idxmax())

    return Insight(
        key="churn_at_risk",
        kind="churn_at_risk",
        title=f"{n:,} customers worth {_usd(rev)} have gone quiet (90+ days)",
        tldr="Launch a win-back campaign for lapsed customers",
        impact_usd=rev,
        evidence=(
            f"{n:,} customers ({100.0 * _safe_ratio(n, kpis['customers']):.1f}% of "
            f"the base) have not ordered in 90+ days",
            f"They produced {_usd(rev)} net revenue this period "
            f"({100.0 * _safe_ratio(rev, total_rev):.1f}% of company total)",
            f"Average silence: {avg_silence:.0f} days · their favourite category: {top_cat}",
        ),
        action=("Launch a win-back sequence this week: email + 10% coupon targeted "
                "at each lapsed customer's favourite category; personal outreach "
                "for lapsed customers above $500 lifetime value; measure the "
                "30-day reactivation rate and double down on whatever reconverts."),
        owner="CRM / Lifecycle",
    )


def _discount_insight(df: pd.DataFrame) -> Insight | None:
    """Nov-Dec margin compression vs the rest of the year."""
    peak_mask = df["month"].isin(PEAK_MONTHS)
    peak, rest = df[peak_mask], df[~peak_mask]
    if peak.empty or rest.empty:
        return None

    peak_net = float(peak["net_revenue"].sum())
    m_peak = 100.0 * _safe_ratio(float(peak["net_profit"].sum()), peak_net)
    m_rest = 100.0 * _safe_ratio(float(rest["net_profit"].sum()),
                                 float(rest["net_revenue"].sum()))
    pp = m_rest - m_peak
    if pp < DISCOUNT_COMPRESSION_FLOOR_PP:
        logger.info("Discount insight skipped: margin compression %.1fpp < "
                    "%.1fpp floor", pp, DISCOUNT_COMPRESSION_FLOOR_PP)
        return None

    impact = pp / 100.0 * peak_net
    d_peak = float(peak["discount_pct"].mean())
    d_rest = float(rest["discount_pct"].mean())
    return Insight(
        key="discount_compression",
        kind="discount_compression",
        title=f"Holiday discounting compressed margin by {pp:.1f}pp",
        tldr="Cap holiday discounts; shift promo budget to bundles",
        impact_usd=impact,
        evidence=(
            f"Nov–Dec average discount {_pct(d_peak)} vs {_pct(d_rest)} rest of year",
            f"Margin {_pct(m_peak)} in Nov–Dec vs {_pct(m_rest)} rest of year "
            f"({pp:.1f}pp thinner)",
            f"{_usd(impact)} profit given up on {_usd(peak_net)} peak-season net revenue",
        ),
        action=("Next peak: cap discounts at 20%; move the remaining promo budget "
                "to bundles and free-shipping thresholds (margin-accretive); any "
                "promo above 25% requires a margin-floor sign-off from finance."),
        owner="Pricing / Promo",
        horizon="Before next peak",
    )


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #

def rank_insights(candidates: list[Insight], top_n: int = 5) -> list[Insight]:
    """Dedupe by key, sort by impact desc, keep top N (with the SKU guarantee)."""
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    unique: dict[str, Insight] = {}
    for ins in candidates:
        cur = unique.get(ins.key)
        if cur is None or ins.impact_usd > cur.impact_usd:
            unique[ins.key] = ins
    ranked = sorted(unique.values(), key=lambda i: i.impact_usd, reverse=True)
    top = list(ranked[:top_n])

    if GUARANTEE_SKU_SLOT and top and not any(i.kind == "returns_sku" for i in top):
        sku = next((i for i in ranked[top_n:] if i.kind == "returns_sku"), None)
        if sku is not None:
            logger.info("SKU-slot guarantee: '%s' (%s) replaces '%s' (%s) in the "
                        "top %d — product-level actionability is a core promise",
                        sku.key, _usd(sku.impact_usd), top[-1].key,
                        _usd(top[-1].impact_usd), top_n)
            top[-1] = sku

    logger.info("Insight ranking: %d candidates -> %d kept", len(candidates), len(top))
    for i, ins in enumerate(top, 1):
        logger.info("  #%d [%s] %-34s impact %s",
                    i, ins.kind, ins.key[:34], _usd(ins.impact_usd))
    return top


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def _validate_analysis_cfg(cfg: dict[str, Any]) -> None:
    required = ("high_return_threshold", "min_category_volume", "min_sku_volume",
                "top_n_return_drivers", "top_n_insights", "mom_decline_threshold")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config.yaml 'analysis' section is missing key(s): "
                         f"{', '.join(missing)}")
    if not 0.0 < float(cfg["high_return_threshold"]) < 1.0:
        raise ValueError("analysis.high_return_threshold must be in (0, 1), e.g. 0.15")
    if float(cfg["mom_decline_threshold"]) >= 0:
        raise ValueError("analysis.mom_decline_threshold must be negative, e.g. -0.10")
    for key in ("top_n_return_drivers", "top_n_insights",
                "min_category_volume", "min_sku_volume"):
        if int(cfg[key]) < 1:
            raise ValueError(f"analysis.{key} must be >= 1")


def _load_cleaning_report() -> dict[str, Any] | None:
    """Attach Step 2's audit to the memo if it exists (never fatal if not)."""
    try:
        path = Path(get_config()["paths"]["processed_data"]).with_name("cleaning_report.json")
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Cleaning report attached: %s rows in -> %s out",
                    data.get("rows_in", "?"), data.get("rows_out", "?"))
        return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read cleaning_report.json (%s) — memo omits it", exc)
        return None


def analyze(
    df: pd.DataFrame, analysis_cfg: dict[str, Any] | None = None
) -> AnalysisResult:
    """Run every metric + generator; return the single AnalysisResult object.

    app.py and src.visualize consume this, so memo, charts and dashboard all
    render identical numbers.
    """
    cfg = analysis_cfg if analysis_cfg is not None else get_config().get("analysis", {})
    _validate_analysis_cfg(cfg)
    threshold_pct = float(cfg["high_return_threshold"]) * 100

    kpis = compute_kpis(df)
    monthly = monthly_trend(df)
    drops = detect_seasonal_drops(monthly, float(cfg["mom_decline_threshold"]))
    cats = category_performance(df, int(cfg["min_category_volume"]))
    regions = regional_performance(df)
    drivers = top_return_drivers(df, int(cfg["top_n_return_drivers"]),
                                 int(cfg["min_sku_volume"]), threshold_pct)
    segments = segment_summary(df)

    candidates: list[Insight] = []
    candidates += _category_return_insights(cats, kpis, threshold_pct)
    candidates += _sku_return_insights(drivers, kpis)
    candidates += _seasonal_drop_insights(drops)
    for gen in (_regional_decline_insight, _discount_insight):
        if (ins := gen(df)) is not None:
            candidates.append(ins)
    if (ins := _churn_insight(df, kpis)) is not None:
        candidates.append(ins)

    insights = rank_insights(candidates, int(cfg["top_n_insights"]))
    meta = {"seed": get_config().get("generation", {}).get("seed"),
            "high_return_pct": threshold_pct,
            "cleaning_report": _load_cleaning_report()}

    logger.info("Analysis complete: %d KPIs, 5 tables, %d insights "
                "(from %d candidates)", len(kpis), len(insights), len(candidates))
    return AnalysisResult(kpis=kpis, monthly=monthly, seasonal_drops=drops,
                          categories=cats, regions=regions, return_drivers=drivers,
                          segments=segments, insights=insights, meta=meta)


# --------------------------------------------------------------------------- #
# Executive memo
# --------------------------------------------------------------------------- #

def generate_executive_memo(result: AnalysisResult, path: Path | str) -> Path:
    """Render the one-page leadership memo (Markdown) and write it to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    k, m = result.kpis, result.meta
    seed_txt = f", seed {m['seed']}" if m.get("seed") is not None else ""
    L: list[str] = []
    ap = L.append

    ap("# Executive Memo — E-commerce Sales Performance & Actions")
    ap("")
    ap(f"**Period:** {k['period_start']:%b %d, %Y} → {k['period_end']:%b %d, %Y} · "
       f"**Generated:** {datetime.now():%Y-%m-%d %H:%M} · **Data:** synthetic"
       f"{seed_txt} · {k['orders']:,} orders · {k['customers']:,} customers")
    ap("")
    ap("---")
    ap("")

    # ---- TL;DR ----
    ap("## TL;DR — this week's priorities")
    ap("")
    if result.insights:
        for i, ins in enumerate(result.insights, 1):
            ap(f"{i}. **{ins.title}** — {ins.tldr} · **{_usd(ins.impact_usd)} at "
               f"risk** *(owner: {ins.owner})*")
    else:
        ap("_No insight crossed the configured thresholds this period. If this "
           "persists, review the `analysis` thresholds in config.yaml._")
    ap("")

    # ---- KPIs ----
    ap("## 1. Business at a glance")
    ap("")
    pairs = [
        ("Net revenue", _usd(k["net_revenue"])), ("Net profit", _usd(k["net_profit"])),
        ("Margin", _pct(k["margin_pct"])), ("Return rate", _pct(k["return_rate_pct"])),
        ("Returns cost (refund + write-off)", _usd(k["returns_cost"])),
        ("Revenue lost to returns", _usd(k["refunded_revenue"])),
        ("Average order value", f"${k['aov']:,.2f}"), ("Orders", f"{k['orders']:,}"),
        ("Customers", f"{k['customers']:,}"), ("Active SKUs", f"{k['products']:,}"),
        ("Average discount", _pct(k["avg_discount_pct"])), ("Months of data", str(k["months"])),
    ]
    ap("| Metric | Value | Metric | Value |")
    ap("|---|---:|---|---:|")
    for (l1, v1), (l2, v2) in zip(pairs[::2], pairs[1::2]):
        ap(f"| {l1} | {v1} | {l2} | {v2} |")
    ap("")

    # ---- Actions ----
    ap("## 2. This week's action list — ranked by dollars at risk")
    ap("")
    if result.insights:
        for i, ins in enumerate(result.insights, 1):
            ap(f"### Action {i} — {ins.title}")
            ap("")
            ap(f"**{_usd(ins.impact_usd)} at risk** · Owner: {ins.owner} · "
               f"Horizon: {ins.horizon}")
            ap("")
            for ev in ins.evidence:
                ap(f"- {ev}")
            ap("")
            ap(f"**→ Do this:** {ins.action}")
            ap("")
    else:
        ap("_No actions above threshold this period._")
        ap("")

    # ---- Watchlist ----
    ap("## 3. Product watchlist — margin bleed (top return drivers)")
    ap("")
    d = result.return_drivers
    if d.empty:
        ap("_No SKU crossed the return thresholds — nothing needs delisting._")
    else:
        ap("| # | SKU | Product | Category | Return rate | Orders | $ at risk | Top reason |")
        ap("|---|---|---|---|---:|---:|---:|---|")
        for i, row in enumerate(d.itertuples(), 1):
            ap(f"| {i} | {row.product_id} | {row.product_name} | {row.category} | "
               f"**{_pct(row.return_rate_pct)}** | {row.orders:,} | "
               f"{_usd(row.returns_cost)} | {row.top_return_reason or '—'} |")
    ap("")

    # ---- Categories ----
    ap("## 4. Category scorecard")
    ap("")
    ap("| Category | Net revenue | Share | Margin | Return rate | Returns cost | Top return reason |")
    ap("|---|---:|---:|---:|---:|---:|---|")
    for row in result.categories.itertuples():
        flag = " ⚠️" if row.return_rate_pct >= float(m.get("high_return_pct", 15.0)) else ""
        ap(f"| {row.category} | {_usd(row.net_revenue)} | "
           f"{_pct(row.net_revenue_share_pct)} | {_pct(row.margin_pct)} | "
           f"{_pct(row.return_rate_pct)}{flag} | {_usd(row.returns_cost)} | "
           f"{row.top_return_reason or '—'} |")
    ap("")

    # ---- Regions ----
    ap("## 5. Regional performance — where the drop-offs are")
    ap("")
    ap("| Region | Net revenue | Share | Return rate | Share H1 | Share H2 | Δ |")
    ap("|---|---:|---:|---:|---:|---:|---:|")
    for row in result.regions.itertuples():
        if pd.isna(row.share_h1_pct):
            h1s = h2s = ds = "—"
        else:
            h1s, h2s = _pct(row.share_h1_pct), _pct(row.share_h2_pct)
            ds = f"{row.share_change_pp:+.1f}pp"
        ap(f"| {row.customer_region} | {_usd(row.net_revenue)} | "
           f"{_pct(row.net_revenue_share_pct)} | {_pct(row.return_rate_pct)} | "
           f"{h1s} | {h2s} | {ds} |")
    ap("")
    decliners = result.regions[
        result.regions["share_change_pp"] <= -REGION_SHARE_DROP_FLOOR_PP]
    if not decliners.empty:
        worst = decliners.sort_values("share_change_pp").iloc[0]
        ap(f"> ⚠️ **{worst['customer_region']}** lost "
           f"{abs(worst['share_change_pp']):.1f}pp of revenue share between halves — "
           f"see the action list.")
        ap("")

    # ---- Segments ----
    ap("## 6. Customer base (RFM-lite segments)")
    ap("")
    ap("| Segment | Customers | Orders | Net revenue | Revenue share |")
    ap("|---|---:|---:|---:|---:|")
    for row in result.segments.itertuples():
        ap(f"| {row.customer_segment} | {row.customers:,} | {row.orders:,} | "
           f"{_usd(row.net_revenue)} | {_pct(row.net_revenue_share_pct)} |")
    ap("")

    # ---- Methodology ----
    ap("## 7. Methodology & caveats")
    ap("")
    ap("- **Returns cost** = refunded revenue + written-off COGS (conservative "
       "write-off model — returned goods are assumed unsellable, so `net_profit` "
       "on a returned order is −COGS).")
    kinds_present = {i.kind for i in result.insights}
    notes = [f"*{kind}*: {KIND_IMPACT_NOTES[kind]}"
             for kind in sorted(kinds_present) if kind in KIND_IMPACT_NOTES]
    if notes:
        ap("- **Dollars at risk** is measured as:")
        for note in notes:
            ap(f"  - {note}")
    ap("- Impacts are **period totals** (this dataset's full window), not forecasts.")
    ap("- Ranking is by dollar impact; one slot is reserved for the top "
       "product-level return driver (product actionability is this dashboard's "
       "core promise).")
    if (cr := m.get("cleaning_report")):
        ap(f"- Cleaning audit: {int(cr.get('rows_in', 0)):,} raw rows → "
           f"{int(cr.get('rows_out', 0)):,} kept; full fix counts in "
           f"`data/cleaning_report.json`.")
    ap(f"- Data is **synthetic**{seed_txt} and fully reproducible via "
       f"`python run_pipeline.py`.")
    ap("")
    ap("---")
    ap("")
    ap("_Generated by `src/analysis.py` · charts in `insights/charts/` · "
       "interactive view: `streamlit run app.py`_")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    logger.info("Executive memo written -> %s (%d lines, %.1f KB)",
                path, len(L), path.stat().st_size / 1024)
    return path


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> Path:
    """Run Step 3 end-to-end; return the memo artifact path."""
    cfg = get_config()
    df = load_processed_data(Path(cfg["paths"]["processed_data"]))
    result = analyze(df)

    k = result.kpis
    logger.info("Headline | net %s | profit %s | margin %s | return rate %s | "
                "returns cost %s", _usd(k["net_revenue"]), _usd(k["net_profit"]),
                _pct(k["margin_pct"]), _pct(k["return_rate_pct"]),
                _usd(k["returns_cost"]))
    return generate_executive_memo(result, Path(cfg["paths"]["memo_file"]))


if __name__ == "__main__":
    setup_logging()
    main()
