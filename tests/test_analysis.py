"""tests/test_analysis.py — exact metric math, insight floors, the ranking
rules (incl. the SKU-slot guarantee), memo rendering, and analyze() integration."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

import src.analysis as an
from helpers import ANALYSIS_CFG, fact_frame, fact_row
from src.analysis import (
    SEASONAL_MAX_INSIGHTS,
    Insight,
    analyze,
    category_performance,
    compute_kpis,
    detect_seasonal_drops,
    generate_executive_memo,
    load_processed_data,
    monthly_trend,
    rank_insights,
    regional_performance,
    segment_summary,
    top_return_drivers,
)
from src.data_processing import DataQualityError

# --------------------------------------------------------------------------- #
# KPIs — exact math on a 4-row frame
# --------------------------------------------------------------------------- #

KPI_FRAME = fact_frame([
    fact_row("O1", date="2024-01-05", customer_id="C1", product_id="P1"),
    fact_row("O2", date="2024-01-10", customer_id="C2", product_id="P2",
             unit_price=50.0, unit_cost=30.0, returned=True,
             return_reason="Changed mind"),
    fact_row("O3", date="2024-02-01", customer_id="C3", product_id="P1",
             quantity=2, unit_price=100.0, unit_cost=60.0),
    fact_row("O4", date="2024-02-15", customer_id="C4", product_id="P3",
             unit_price=50.0, unit_cost=40.0),
])


def test_compute_kpis_exact_math():
    k = compute_kpis(KPI_FRAME)
    assert k["orders"] == 4
    assert k["customers"] == 4
    assert k["products"] == 3
    assert k["months"] == 2
    assert k["gross_revenue"] == pytest.approx(400.0)
    assert k["net_revenue"] == pytest.approx(350.0)
    assert k["cogs"] == pytest.approx(250.0)
    assert k["net_profit"] == pytest.approx(100.0)
    assert k["margin_pct"] == pytest.approx(100 * 100 / 350, rel=1e-3)
    assert k["aov"] == pytest.approx(87.5)
    assert k["return_rate_pct"] == pytest.approx(25.0)
    assert k["refunded_revenue"] == pytest.approx(50.0)
    assert k["writeoff_cogs"] == pytest.approx(30.0)
    assert k["returns_cost"] == pytest.approx(80.0)  # refund + write-off


# --------------------------------------------------------------------------- #
# Trend & seasonal drops
# --------------------------------------------------------------------------- #

TREND_FRAME = fact_frame([
    fact_row("O1", date="2024-01-15"),
    fact_row("O2", date="2024-02-15"),
    fact_row("O3", date="2024-03-15", unit_price=120.0, unit_cost=40.0),
])


def test_monthly_trend_math():
    m = monthly_trend(TREND_FRAME)
    assert m["month_label"].tolist() == ["Jan 24", "Feb 24", "Mar 24"]
    assert pd.isna(m["mom_net_pct"].iloc[0])
    assert m["mom_net_pct"].iloc[1] == pytest.approx(-20.0)
    assert m["mom_net_pct"].iloc[2] == pytest.approx(50.0)
    assert m["prev_net_revenue"].iloc[1] == pytest.approx(100.0)
    assert m["return_rate_pct"].tolist() == [0.0, 0.0, 0.0]


def test_detect_seasonal_drops_boundary_and_order():
    drops = detect_seasonal_drops(monthly_trend(fact_frame([
        fact_row("O1", date="2024-01-15"),
        fact_row("O2", date="2024-02-15", unit_price=80.0, unit_cost=30.0),  # -20%
        fact_row("O3", date="2024-03-15", unit_price=72.0, unit_cost=30.0),  # -10% boundary
        fact_row("O4", date="2024-04-15", unit_price=90.0, unit_cost=30.0),  # +25%
    ])), mom_threshold=-0.10)
    assert [d["order_year_month"] for d in drops] == ["2024-02", "2024-03"]
    assert drops[0]["shortfall"] == pytest.approx(20.0)   # sorted largest first
    assert drops[1]["mom_pct"] == pytest.approx(-10.0)    # at threshold: included


def test_detect_seasonal_drops_rejects_positive_threshold():
    with pytest.raises(ValueError, match="negative"):
        detect_seasonal_drops(monthly_trend(TREND_FRAME), mom_threshold=0.10)


# --------------------------------------------------------------------------- #
# Category / regional / SKU / segment breakdowns
# --------------------------------------------------------------------------- #

def _category_frame() -> pd.DataFrame:
    rows = []
    for i in range(70):
        rows.append(fact_row(f"A{i}", category="Apparel", product_id="PA",
                             unit_price=20.0, unit_cost=10.0))
    for i in range(30):
        reason = "Wrong size or fit" if i < 20 else "Changed mind"
        rows.append(fact_row(f"AR{i}", category="Apparel", product_id="PA",
                             unit_price=20.0, unit_cost=10.0, returned=True,
                             return_reason=reason))
    for i in range(54):
        rows.append(fact_row(f"E{i}", category="Electronics", product_id="PE",
                             unit_price=500.0, unit_cost=200.0))
    for i in range(6):
        rows.append(fact_row(f"ER{i}", category="Electronics", product_id="PE",
                             unit_price=500.0, unit_cost=200.0, returned=True,
                             return_reason="Not as described"))
    for i in range(10):
        rows.append(fact_row(f"B{i}", category="Beauty", product_id="PB",
                             unit_price=10.0, unit_cost=4.0))
    return fact_frame(rows)


def test_category_performance_rates_flags_and_mode_reason():
    cats = category_performance(_category_frame(), min_volume=50)
    assert cats["category"].tolist() == ["Electronics", "Apparel", "Beauty"]
    by_cat = cats.set_index("category")
    assert by_cat.loc["Apparel", "return_rate_pct"] == pytest.approx(30.0)
    assert by_cat.loc["Electronics", "return_rate_pct"] == pytest.approx(10.0)
    assert not by_cat.loc["Apparel", "low_volume"]
    assert by_cat.loc["Beauty", "low_volume"]           # 10 orders < 50
    assert by_cat.loc["Apparel", "top_return_reason"] == "Wrong size or fit"
    assert by_cat.loc["Beauty", "top_return_reason"] == ""   # no returns at all
    assert cats["net_revenue_share_pct"].sum() == pytest.approx(100.0)


def _regional_frame() -> pd.DataFrame:
    rows = []
    for month in range(1, 13):
        date = f"2024-{month:02d}-10"
        for i in range(3):
            rows.append(fact_row(f"N{month}-{i}", date=date, customer_region="North"))
        for i in range(9 if month <= 6 else 1):
            rows.append(fact_row(f"E{month}-{i}", date=date, customer_region="East"))
    return fact_frame(rows)


def test_regional_performance_half_year_shares():
    regions = regional_performance(_regional_frame())
    by_region = regions.set_index("customer_region")
    assert by_region.loc["East", "share_h1_pct"] == pytest.approx(75.0)
    assert by_region.loc["East", "share_h2_pct"] == pytest.approx(25.0)
    assert by_region.loc["East", "share_change_pp"] == pytest.approx(-50.0)
    assert by_region.loc["North", "share_change_pp"] == pytest.approx(50.0)


def test_regional_performance_single_half_yields_nan():
    regions = regional_performance(fact_frame([
        fact_row("O1", date="2024-03-01", customer_region="North"),
        fact_row("O2", date="2024-04-01", customer_region="South"),
    ]))
    assert regions["share_h1_pct"].isna().all()


def _sku_rows(product_id: str, n_orders: int, n_returned: int,
              price: float, cost: float) -> list[dict]:
    rows = []
    for i in range(n_orders - n_returned):
        rows.append(fact_row(f"{product_id}-c{i}", product_id=product_id,
                             product_name=f"Item {product_id}",
                             category="Electronics", unit_price=price,
                             unit_cost=cost))
    for i in range(n_returned):
        rows.append(fact_row(f"{product_id}-r{i}", product_id=product_id,
                             product_name=f"Item {product_id}",
                             category="Electronics", unit_price=price,
                             unit_cost=cost, returned=True,
                             return_reason="Not as described"))
    return rows


def _drivers_frame() -> pd.DataFrame:
    return fact_frame(
        _sku_rows("A", 30, 12, 100.0, 50.0)    # 40% rate, $1,800 returns cost
        + _sku_rows("B", 30, 15, 10.0, 5.0)    # 50% rate, $225 cost
        + _sku_rows("C", 5, 4, 100.0, 50.0)    # 80% rate but only 5 orders
        + _sku_rows("D", 30, 1, 100.0, 50.0)   # 3.3% rate: below threshold
    )


def test_top_return_drivers_filters_and_ranking():
    drivers = top_return_drivers(_drivers_frame(), top_n=2, min_volume=20,
                                 high_return_threshold_pct=15.0)
    assert drivers["product_id"].tolist() == ["A", "B"]
    by_sku = drivers.set_index("product_id")
    assert by_sku.loc["A", "returns_cost"] == pytest.approx(1_800.0)
    assert by_sku.loc["A", "return_rate_pct"] == pytest.approx(40.0)
    assert by_sku.loc["B", "top_return_reason"] == "Not as described"


def test_top_return_drivers_respects_top_n():
    drivers = top_return_drivers(_drivers_frame(), top_n=1, min_volume=20,
                                 high_return_threshold_pct=15.0)
    assert drivers["product_id"].tolist() == ["A"]


def test_segment_summary_orders_by_revenue():
    summary = segment_summary(fact_frame([
        fact_row("O1", customer_segment="Champion"),
        fact_row("O2", customer_segment="At Risk"),
        fact_row("O3", customer_segment="Champion"),
    ]))
    assert summary["customer_segment"].tolist()[0] == "Champion"
    assert summary["net_revenue_share_pct"].sum() == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Ranking — including the SKU-slot guarantee
# --------------------------------------------------------------------------- #

def _ins(key: str, kind: str, impact: float) -> Insight:
    return Insight(key=key, kind=kind, title=f"Title {key}", tldr="do it",
                   impact_usd=impact, evidence=("evidence",), action="act",
                   owner="owner")


def _aggregate_candidates() -> list[Insight]:
    specs = [("returns_category", 100.0), ("churn_at_risk", 90.0),
             ("region_decline", 80.0), ("seasonal_drop", 70.0),
             ("discount_compression", 60.0)]
    return [_ins(f"k{i}", kind, impact) for i, (kind, impact) in enumerate(specs)]


def test_rank_insights_orders_by_impact_desc():
    ranked = rank_insights([_ins("a", "seasonal_drop", 10.0),
                            _ins("b", "churn_at_risk", 100.0),
                            _ins("c", "region_decline", 50.0)], top_n=3)
    assert [i.key for i in ranked] == ["b", "c", "a"]


def test_rank_insights_dedupes_by_key_keeping_max_impact():
    ranked = rank_insights([_ins("a", "seasonal_drop", 50.0),
                            _ins("a", "seasonal_drop", 70.0),
                            _ins("b", "churn_at_risk", 10.0)], top_n=5)
    assert [i.key for i in ranked] == ["a", "b"]
    assert ranked[0].impact_usd == 70.0


def test_rank_insights_guarantees_one_sku_slot():
    candidates = _aggregate_candidates() + [_ins("sku", "returns_sku", 10.0)]
    kinds = [i.kind for i in rank_insights(candidates, top_n=5)]
    assert "returns_sku" in kinds
    assert "discount_compression" not in kinds   # smallest aggregate swapped out


def test_rank_insights_no_swap_when_sku_already_ranks():
    candidates = _aggregate_candidates() + [_ins("sku", "returns_sku", 95.0)]
    kinds = [i.kind for i in rank_insights(candidates, top_n=5)]
    assert kinds.count("returns_sku") == 1
    assert "discount_compression" in kinds


def test_rank_insights_sku_guarantee_can_be_disabled(monkeypatch):
    monkeypatch.setattr(an, "GUARANTEE_SKU_SLOT", False)
    candidates = _aggregate_candidates() + [_ins("sku", "returns_sku", 10.0)]
    assert "returns_sku" not in [i.kind for i in rank_insights(candidates, top_n=5)]


def test_rank_insights_validation_and_empty():
    with pytest.raises(ValueError, match="top_n"):
        rank_insights([], top_n=0)
    assert rank_insights([], top_n=5) == []


# --------------------------------------------------------------------------- #
# Insight generators — floors prevent noise
# --------------------------------------------------------------------------- #

def test_category_return_insight_maps_reason_to_action():
    cats = pd.DataFrame([
        {"category": "Apparel", "orders": 100, "net_revenue": 1_000.0,
         "net_profit": 300.0, "refunded": 250.0, "writeoff": 100.0,
         "returns_cost": 350.0, "returned_orders": 25, "avg_discount_pct": 5.0,
         "top_return_reason": "Wrong size or fit", "net_revenue_share_pct": 50.0,
         "margin_pct": 30.0, "return_rate_pct": 25.0, "low_volume": False},
        {"category": "Beauty", "orders": 200, "net_revenue": 900.0,
         "net_profit": 300.0, "refunded": 20.0, "writeoff": 10.0,
         "returns_cost": 30.0, "returned_orders": 8, "avg_discount_pct": 4.0,
         "top_return_reason": "Changed mind", "net_revenue_share_pct": 45.0,
         "margin_pct": 33.3, "return_rate_pct": 4.0, "low_volume": False},
        {"category": "Sports", "orders": 40, "net_revenue": 100.0,
         "net_profit": 30.0, "refunded": 15.0, "writeoff": 5.0,
         "returns_cost": 20.0, "returned_orders": 12, "avg_discount_pct": 6.0,
         "top_return_reason": "Changed mind", "net_revenue_share_pct": 5.0,
         "margin_pct": 30.0, "return_rate_pct": 30.0, "low_volume": True},
    ])
    kpis = {"return_rate_pct": 10.0, "returned_orders": 45}
    insights = an._category_return_insights(cats, kpis, threshold_pct=15.0)
    assert len(insights) == 1          # Beauty below rate, Sports low-volume
    insight = insights[0]
    assert insight.kind == "returns_category"
    assert insight.key == "returns_category::Apparel"
    assert "Apparel" in insight.title
    assert "size chart" in insight.action.lower()   # fit reason -> sizing action


def test_sku_return_insights_from_drivers():
    drivers = pd.DataFrame([{
        "product_id": "SKU-0008", "product_name": "Aurora Hoodie Elite",
        "category": "Apparel", "orders": 120, "units": 130,
        "net_revenue": 1_500.0, "refunded": 300.0, "writeoff": 120.0,
        "returns_cost": 420.0, "returned_orders": 36,
        "top_return_reason": "Wrong size or fit", "return_rate_pct": 30.0,
    }])
    insights = an._sku_return_insights(drivers, {"return_rate_pct": 10.0})
    assert len(insights) == 1
    assert insights[0].key == "returns_sku::SKU-0008"
    assert "Aurora Hoodie Elite" in insights[0].title
    assert insights[0].owner == "Category merch lead"


def _east_collapse_frame() -> pd.DataFrame:
    rows = []
    for month in range(1, 13):
        date = f"2024-{month:02d}-10"
        east_n = 30 if month <= 6 else 3
        rows += [fact_row(f"E{month}-{i}", date=date, customer_region="East")
                 for i in range(east_n)]
        rows += [fact_row(f"N{month}-{i}", date=date, customer_region="North")
                 for i in range(70)]
    return fact_frame(rows)


def test_regional_decline_insight_fires_on_share_collapse():
    insight = an._regional_decline_insight(_east_collapse_frame())
    assert insight is not None
    assert insight.kind == "region_decline"
    assert "East" in insight.title
    assert insight.impact_usd > 0
    assert insight.horizon == "This month"


def test_regional_decline_insight_skips_stable_shares():
    rows = []
    for month in range(1, 13):
        date = f"2024-{month:02d}-10"
        rows += [fact_row(f"E{month}-{i}", date=date, customer_region="East")
                 for i in range(50)]
        rows += [fact_row(f"N{month}-{i}", date=date, customer_region="North")
                 for i in range(50)]
    assert an._regional_decline_insight(fact_frame(rows)) is None


def test_discount_insight_fires_on_peak_compression():
    rows = [fact_row(f"R{m}", date=f"2024-{m:02d}-10", discount_pct=0.0,
                     unit_price=100.0, unit_cost=40.0) for m in range(1, 11)]
    rows += [fact_row(f"P{m}", date=f"2024-{m:02d}-10", discount_pct=30.0,
                      unit_price=100.0, unit_cost=40.0) for m in (11, 12)]
    insight = an._discount_insight(fact_frame(rows))
    assert insight is not None
    assert insight.kind == "discount_compression"
    assert insight.impact_usd > 0


def test_discount_insight_skips_flat_discounts():
    rows = [fact_row(f"R{m}", date=f"2024-{m:02d}-10", discount_pct=5.0,
                     unit_price=100.0, unit_cost=40.0) for m in range(1, 13)]
    assert an._discount_insight(fact_frame(rows)) is None


def _churn_frame(n_at_risk: int) -> pd.DataFrame:
    rows = [fact_row(f"AR{i}", date="2024-08-01", customer_id=f"CR{i}",
                     customer_segment="At Risk", category="Apparel")
            for i in range(n_at_risk)]
    rows += [fact_row(f"CH{i}", date="2024-12-20", customer_id=f"CH{i}",
                      customer_segment="Champion") for i in range(100)]
    return fact_frame(rows)


def test_churn_insight_quantifies_silent_customers():
    df = _churn_frame(12)
    insight = an._churn_insight(df, compute_kpis(df))
    assert insight is not None
    assert insight.kind == "churn_at_risk"
    assert insight.impact_usd == pytest.approx(1_200.0)   # 12 x $100
    assert "12" in insight.title


def test_churn_insight_skips_small_at_risk_base():
    df = _churn_frame(2)
    assert an._churn_insight(df, compute_kpis(df)) is None


def _drop(ym: str, mom: float, prev: float, net: float) -> dict:
    return {"order_year_month": ym, "month_label": f"{ym} label",
            "mom_pct": mom, "prev_net_revenue": prev, "net_revenue": net,
            "prev_orders": 12, "orders": 10, "shortfall": prev - net}


def test_seasonal_drop_insights_capped():
    drops = [_drop("2024-02", -20.0, 100.0, 80.0),
             _drop("2024-03", -18.0, 80.0, 65.6),
             _drop("2024-07", -15.0, 90.0, 76.5),
             _drop("2024-09", -12.0, 70.0, 61.6)]
    insights = an._seasonal_drop_insights(drops)
    assert len(insights) == SEASONAL_MAX_INSIGHTS
    assert all(i.kind == "seasonal_drop" for i in insights)
    assert insights[0].key == "seasonal_drop::2024-02"


# --------------------------------------------------------------------------- #
# Ingestion guards
# --------------------------------------------------------------------------- #

def test_load_processed_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="run it first"):
        load_processed_data(tmp_path / "missing.parquet")


def test_load_processed_data_missing_column(tmp_path):
    path = tmp_path / "bad.parquet"
    KPI_FRAME.drop(columns=["net_profit"]).to_parquet(path)
    with pytest.raises(DataQualityError, match="net_profit"):
        load_processed_data(path)


def test_load_processed_data_empty(tmp_path):
    path = tmp_path / "empty.parquet"
    pd.DataFrame(columns=list(an.REQUIRED_PROCESSED_COLUMNS)).to_parquet(path)
    with pytest.raises(ValueError, match="0 rows"):
        load_processed_data(path)


# --------------------------------------------------------------------------- #
# analyze() integration on the mini pipeline
# --------------------------------------------------------------------------- #

@pytest.mark.integration
def test_analyze_end_to_end(mini_result, mini_processed):
    assert mini_result.kpis["orders"] == len(mini_processed)
    assert len(mini_result.monthly) == 12
    assert mini_result.meta["high_return_pct"] == pytest.approx(15.0)
    assert mini_result.insights, "the planted stories must produce insights"
    impacts = [i.impact_usd for i in mini_result.insights]
    assert impacts == sorted(impacts, reverse=True)
    assert len(mini_result.insights) <= ANALYSIS_CFG["top_n_insights"]
    kinds = {i.kind for i in mini_result.insights}
    assert "returns_category" in kinds    # the Apparel margin-bleed story
    assert "returns_sku" in kinds         # the SKU-slot guarantee


@pytest.mark.integration
@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"mom_decline_threshold": 0.10}, "mom_decline_threshold"),
        ({"high_return_threshold": 1.5}, "high_return_threshold"),
    ],
)
def test_analyze_rejects_bad_config(mini_processed, override, fragment):
    with pytest.raises(ValueError, match=fragment):
        analyze(mini_processed, {**ANALYSIS_CFG, **override})


@pytest.mark.integration
def test_analyze_rejects_missing_config_keys(mini_processed):
    with pytest.raises(ValueError, match="missing key"):
        analyze(mini_processed, {"high_return_threshold": 0.15})


# --------------------------------------------------------------------------- #
# Executive memo
# --------------------------------------------------------------------------- #

@pytest.mark.integration
def test_executive_memo_content(mini_result, tmp_path):
    path = generate_executive_memo(mini_result, tmp_path / "memo.md")
    text = path.read_text(encoding="utf-8")
    assert "Executive Memo" in text
    assert "## TL;DR" in text
    assert "## 7. Methodology" in text
    assert "| Metric | Value |" in text
    for insight in mini_result.insights:
        assert insight.title in text
    assert text.endswith("\n")


@pytest.mark.integration
def test_executive_memo_empty_insights_fallback(mini_result, tmp_path):
    empty = replace(mini_result, insights=[])
    path = generate_executive_memo(empty, tmp_path / "memo.md")
    assert "No insight crossed" in path.read_text(encoding="utf-8")


@pytest.mark.integration
def test_executive_memo_includes_cleaning_audit(mini_result, tmp_path):
    meta = {**mini_result.meta, "cleaning_report": {"rows_in": 1_000, "rows_out": 990}}
    audited = replace(mini_result, meta=meta)
    path = generate_executive_memo(audited, tmp_path / "memo.md")
    text = path.read_text(encoding="utf-8")
    assert "1,000 raw rows" in text
    assert "990 kept" in text


@pytest.mark.integration
def test_executive_memo_empty_watchlist_fallback(mini_result, tmp_path):
    empty_drivers = mini_result.return_drivers.iloc[0:0]
    no_drivers = replace(mini_result, return_drivers=empty_drivers)
    path = generate_executive_memo(no_drivers, tmp_path / "memo.md")
    assert "No SKU crossed" in path.read_text(encoding="utf-8")
