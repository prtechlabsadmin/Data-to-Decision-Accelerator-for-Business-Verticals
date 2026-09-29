"""app.py — Step 5: the interactive decision dashboard (Streamlit).

Pipeline position: presentation layer. Reads data/processed_sales.parquet
(Step 2), recomputes the full analysis (Step 3's `analyze`) on whatever
slice the user filters, and renders KPIs, interactive Plotly charts and the
ranked action list. The executive memo (the Step 6 artifact) is viewable and
downloadable at the bottom.

This is NOT a pipeline stage — there is no `main() -> Path` contract here;
Streamlit owns the entrypoint: `streamlit run app.py` (or `make run`).

Layout (decision-first — actions BEFORE charts):
    sidebar   date-range + region + category filters, data status, regenerate
    header    KPI row (8 metrics, latest-vs-prior-month deltas)
    actions   "This week's action list" — ranked insights, evidence, owners
    tabs      Overview | Margin bleed | Regions & seasonality | Customers
    memo      executive memo (full-period run) — view + download

Caching strategy:
    @st.cache_resource  init_logging()   once per process (no duplicate
                                          handlers across Streamlit reruns)
    @st.cache_data      load_data()      parquet read, once per process
    @st.cache_data      build_analysis() keyed on the filter tuple — every
                                          widget rerun is instant; only a
                                          CHANGED filter state pays the
                                          ~1-2s analyze() cost

Filter semantics:
    - Everything (KPIs, charts, insights, action ranking) recomputes on the
      filtered slice — filter to one category and watch the action list
      re-rank live. That re-prioritization is the demo.
    - EXCEPTION: customer_segment labels were assigned once in Step 2 against
      the full-period snapshot (correct RFM semantics), so they do not change
      with date filters — captioned in the Customers tab.
    - An emptied multiselect means "all"; an over-tight filter gets a friendly
      warning, never a crash.

Self-bootstrapping: if the parquet is missing (e.g. a fresh Streamlit Cloud
deployment without committed data), the app offers a one-click "generate
now" that runs Steps 1-2 in-process, clears caches and reloads.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

# Make `src.*` importable regardless of the directory Streamlit was launched from.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from src.analysis import (  # noqa: E402
    REGION_SHARE_DROP_FLOOR_PP,
    AnalysisResult,
    Insight,
    analyze,
    load_processed_data,
)
from src.config import get_config, get_logger, setup_logging  # noqa: E402
from src.data_processing import DataQualityError  # noqa: E402

st.set_page_config(
    page_title="Data-to-Decision Accelerator",
    page_icon="🚀",
    layout="wide",
)

logger = get_logger("app")

# --------------------------------------------------------------------------- #
# Constants (colors mirror src/visualize.py so static & interactive match)
# --------------------------------------------------------------------------- #

ALERT_COLOR = "#d62728"
POSITIVE_COLOR = "#2e7d32"
NEUTRAL_DARK = "#3f3f3f"
MUTED_COLOR = "#7f7f7f"
STACK_COLOR = "#c9c9c9"   # revenue lost to returns
H1_COLOR = "#bdbdbd"      # "before" bars in the regional chart

PARETO_TOP_N = 12
PARETO_VALUE_LABELS = 5
CHART_CONFIG = {"displayModeBar": False}


# --------------------------------------------------------------------------- #
# Small formatting helpers (mirror src/analysis formatting for consistency)
# --------------------------------------------------------------------------- #

def fmt_usd(x: float) -> str:
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


def fmt_pct(x: float, dp: int = 1) -> str:
    return f"{x:.{dp}f}%" if pd.notna(x) else "—"


def fmt_delta_usd(d: float | None) -> str | None:
    return None if d is None else f"{'+' if d >= 0 else '−'}{fmt_usd(abs(d))}"


def fmt_delta_pp(d: float | None) -> str | None:
    return None if d is None else f"{d:+.1f}pp"


def _palette() -> list[str]:
    """Qualitative colors matching the configured matplotlib palette name."""
    name = str(get_config().get("visualization", {}).get("palette", "Set2"))
    try:
        return list(getattr(px.colors.qualitative, name))
    except AttributeError:
        logger.warning("Plotly has no qualitative palette '%s' — using Set2", name)
        return list(px.colors.qualitative.Set2)


def _empty_figure(message: str) -> go.Figure:
    """Labeled empty-state chart: never crash, always explain."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white", height=300,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=message, xref="paper", yref="paper", x=0.5,
                          y=0.5, showarrow=False,
                          font=dict(size=13, color=MUTED_COLOR))],
    )
    return fig


# --------------------------------------------------------------------------- #
# Cached data & analysis
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def init_logging() -> None:
    """Configure logging once per process (safe across Streamlit reruns)."""
    setup_logging()


@st.cache_data(show_spinner="Loading processed data…")
def load_data() -> pd.DataFrame:
    """Read the processed parquet once per process (validated by Step 3 loader)."""
    return load_processed_data(Path(get_config()["paths"]["processed_data"]))


@st.cache_data(show_spinner="Analyzing the filtered slice…")
def build_analysis(
    start: date, end: date, regions: tuple[str, ...], categories: tuple[str, ...]
) -> AnalysisResult:
    """Full analysis on the filtered slice, cached per filter state."""
    df = load_data()
    filtered = apply_filters(
        df, Filters(start=start, end=end, regions=regions, categories=categories))
    if filtered.empty:
        raise ValueError(
            "The current filters match no orders — widen the date range or "
            "clear the region/category filters in the sidebar.")
    return analyze(filtered)


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Filters:
    start: date
    end: date
    regions: tuple[str, ...]
    categories: tuple[str, ...]


def get_filters(df: pd.DataFrame) -> Filters:
    """Render the sidebar filter widgets; return the current filter state."""
    with st.sidebar:
        st.header("🎛 Filters")
        st.caption(
            "Every KPI, chart and action below recomputes live on the "
            "filtered slice — try a single category and watch the action "
            "list re-rank.")
        min_d = df["order_date"].min().date()
        max_d = df["order_date"].max().date()

        val = st.date_input(
            "Order date range", value=(min_d, max_d),
            min_value=min_d, max_value=max_d, format="YYYY-MM-DD")
        if val is None:
            start, end = min_d, max_d
        elif isinstance(val, (list, tuple)):
            dates = list(val)
            start = dates[0] if dates else min_d
            end = dates[-1] if len(dates) > 1 else start
        else:  # single date
            start = end = val
        if start > end:
            start, end = end, start

        region_opts = sorted(df["customer_region"].dropna().unique().tolist())
        regions = st.multiselect("Region", options=region_opts, default=region_opts)
        if not regions:  # emptied selection = all
            regions = region_opts

        cat_opts = sorted(df["category"].dropna().unique().tolist())
        categories = st.multiselect("Category", options=cat_opts, default=cat_opts)
        if not categories:
            categories = cat_opts

    return Filters(start, end, tuple(sorted(regions)), tuple(sorted(categories)))


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    """Pure slice for the current filter state (no mutation, no cache)."""
    mask = df["order_date"].between(pd.Timestamp(f.start), pd.Timestamp(f.end))
    if f.regions:
        mask &= df["customer_region"].isin(f.regions)
    if f.categories:
        mask &= df["category"].isin(f.categories)
    return df.loc[mask].copy()


# --------------------------------------------------------------------------- #
# In-app data bootstrap / regeneration
# --------------------------------------------------------------------------- #

def regenerate_dataset() -> bool:
    """Run Steps 1-2 in-process (lazy imports keep app startup fast)."""
    try:
        from src.data_generation import main as generate_main
        from src.data_processing import main as process_main

        with st.spinner("Regenerating dataset (generate + process)…"):
            generate_main()
            process_main()
    except Exception as exc:  # noqa: BLE001 — surface, never crash the app
        logger.exception("In-app regeneration failed")
        st.error(f"Regeneration failed: {exc} — full traceback in logs/pipeline.log")
        return False
    st.cache_data.clear()
    st.success("Fresh dataset generated and processed — reloading.")
    return True


def render_data_status(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Sidebar: data freshness + one-click regeneration."""
    with st.sidebar:
        st.divider()
        st.subheader("🧾 Data")
        p = Path(cfg["paths"]["processed_data"])
        st.caption(f"{len(df):,} orders · {df['order_date'].min():%b %d, %Y} → "
                   f"{df['order_date'].max():%b %d, %Y}")
        if p.is_file():
            stamp = datetime.fromtimestamp(p.stat().st_mtime)
            st.caption(f"Processed: {stamp:%Y-%m-%d %H:%M} · seed "
                       f"{cfg.get('generation', {}).get('seed')}")
        if st.button("♻️ Regenerate dataset",
                     help="Runs Steps 1-2, then reloads the dashboard (~10 s)"):
            if regenerate_dataset():
                st.rerun()


# --------------------------------------------------------------------------- #
# Interactive charts (Plotly counterparts of the static exports —
# same decision boundaries, same stories, now hoverable)
# --------------------------------------------------------------------------- #

def chart_revenue_trend(result: AnalysisResult) -> go.Figure:
    """Monthly net revenue + revenue lost to returns (stacked), return-rate
    line on the right axis, MoM-drop months flagged red."""
    m = result.monthly
    labels = m["month_label"].tolist()
    yms = m["order_year_month"].tolist()
    net = m["net_revenue"].to_numpy(dtype=float)
    lost = np.clip(m["gross_revenue"].to_numpy(dtype=float) - net, 0.0, None)
    rr = m["return_rate_pct"].to_numpy(dtype=float)
    drops = {d["order_year_month"]: d for d in result.seasonal_drops}
    base = _palette()[0]
    line_color = _palette()[1] if len(_palette()) > 1 else NEUTRAL_DARK

    fig = go.Figure()
    fig.add_bar(
        x=labels, y=net, name="Net revenue",
        marker_color=[ALERT_COLOR if ym in drops else base for ym in yms],
        hovertemplate="%{x}<br>Net revenue: %{y:$,.0f}<extra></extra>")
    fig.add_bar(
        x=labels, y=lost, name="Revenue lost to returns", marker_color=STACK_COLOR,
        hovertemplate="%{x}<br>Lost to returns: %{y:$,.0f}<extra></extra>")
    fig.add_trace(go.Scatter(
        x=labels, y=rr, name="Monthly return rate", mode="lines+markers",
        yaxis="y2", line=dict(color=line_color, width=2.5),
        marker=dict(size=6, color=line_color),
        hovertemplate="%{x}<br>Return rate: %{y:.1f}%<extra></extra>"))

    for ym, d in drops.items():
        if ym in yms:
            i = yms.index(ym)
            fig.add_annotation(
                x=labels[i], y=float(net[i] + lost[i]),
                text=f"−{abs(d['mom_pct']):.0f}% MoM", showarrow=False, yshift=12,
                font=dict(color=ALERT_COLOR, size=11))

    fig.update_layout(
        barmode="stack", template="plotly_white", height=430, hovermode="x unified",
        yaxis=dict(title="Revenue", tickprefix="$", tickformat=".2s"),
        yaxis2=dict(title="Return rate", overlaying="y", side="right",
                    showgrid=False, ticksuffix="%",
                    range=[0, max(float(rr.max()) * 1.5, 10.0)]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=20, b=10))
    return fig


def chart_category_quadrant(result: AnalysisResult) -> go.Figure:
    """The decision quadrant: return rate vs margin, bubble = net revenue,
    threshold and company-margin lines drawn on the chart."""
    cats = result.categories
    threshold = float(result.meta.get("high_return_pct", 15.0))
    company_margin = float(result.kpis["margin_pct"])
    rev = cats["net_revenue"].to_numpy(dtype=float)
    orders = cats["orders"].to_numpy(dtype=float)
    pal = _palette()
    sizeref = 2.0 * float(rev.max()) / (55 ** 2) if rev.max() > 0 else 1.0

    fig = go.Figure(go.Scatter(
        x=cats["return_rate_pct"], y=cats["margin_pct"], mode="markers+text",
        text=cats["category"], textposition="top center", textfont=dict(size=11),
        marker=dict(size=rev, sizemode="area", sizeref=sizeref, sizemin=10,
                    color=[pal[i % len(pal)] for i in range(len(cats))],
                    opacity=0.9, line=dict(color="white", width=1.5)),
        customdata=np.stack([rev, orders], axis=-1),
        hovertemplate=("<b>%{text}</b><br>Return rate: %{x:.1f}%<br>Net margin: "
                       "%{y:.1f}%<br>Net revenue: %{customdata[0]:$,.0f}<br>"
                       "Orders: %{customdata[1]:,}<extra></extra>")))

    xmax = float(np.nanmax(cats["return_rate_pct"].to_numpy(dtype=float)))
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = 20.0
    fig.update_xaxes(range=[0, xmax * 1.2 + 1.0], ticksuffix="%",
                     title="Return rate (% of orders)")
    fig.update_yaxes(ticksuffix="%", title="Net margin (% of net revenue)")

    fig.add_vline(x=threshold, line_dash="dash", line_color=ALERT_COLOR,
                  annotation_text=f"return threshold {threshold:.0f}%",
                  annotation_font=dict(color=ALERT_COLOR, size=10))
    if np.isfinite(company_margin):
        fig.add_hline(y=company_margin, line_dash="dash", line_color=NEUTRAL_DARK,
                      annotation_text=f"company margin {company_margin:.1f}%",
                      annotation_font=dict(color=NEUTRAL_DARK, size=10))
    fig.add_annotation(x=0.98, y=0.04, xref="paper", yref="paper", xanchor="right",
                       text="MARGIN BLEED", showarrow=False,
                       font=dict(color=ALERT_COLOR, size=11))
    fig.add_annotation(x=0.02, y=0.96, xref="paper", yref="paper", xanchor="left",
                       text="HEALTHY", showarrow=False,
                       font=dict(color=POSITIVE_COLOR, size=11))
    fig.update_layout(template="plotly_white", height=460, hovermode="closest",
                      showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def chart_margin_bleed(
    df: pd.DataFrame, result: AnalysisResult, top_n: int = PARETO_TOP_N
) -> go.Figure:
    """SKUs ranked by returns cost, cumulative-share line across ALL SKUs,
    80% Pareto line; red bars are the threshold-exceeding watchlist SKUs."""
    tmp = df.assign(_writeoff=df["cogs"] * df["return_flag"])
    per_sku = (tmp.groupby("product_id")
                  .agg(refunded=("revenue_lost_to_returns", "sum"),
                       writeoff=("_writeoff", "sum"),
                       orders=("order_id", "count")))
    per_sku["returns_cost"] = per_sku["refunded"] + per_sku["writeoff"]
    total = float(per_sku["returns_cost"].sum())
    if total <= 0:
        return _empty_figure("No returns in this view — no margin is "
                             "bleeding through returns.")

    per_sku = per_sku.sort_values("returns_cost", ascending=False)
    share = per_sku["returns_cost"] / total * 100.0
    top = per_sku.head(max(top_n, 1))
    cum = share.cumsum().to_numpy()[: len(top)]

    driver_ids = (set(result.return_drivers["product_id"])
                  if not result.return_drivers.empty else set())
    ids = top.index.tolist()
    costs = top["returns_cost"].to_numpy(dtype=float)
    orders_n = top["orders"].to_numpy(dtype=float)
    shares = share.to_numpy()[: len(top)]
    base = _palette()[0]

    fig = go.Figure()
    fig.add_bar(
        x=ids, y=costs, name="Returns cost",
        marker_color=[ALERT_COLOR if pid in driver_ids else base for pid in ids],
        text=[fmt_usd(v) if i < PARETO_VALUE_LABELS else ""
              for i, v in enumerate(costs)],
        textposition="outside", textfont=dict(size=9),
        customdata=np.stack([orders_n, shares], axis=-1),
        hovertemplate=("<b>%{x}</b><br>Returns cost: %{y:$,.0f}<br>Orders: "
                       "%{customdata[0]:,.0f}<br>Share of all returns cost: "
                       "%{customdata[1]:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=ids, y=cum, name="Cumulative share (all SKUs)", mode="lines+markers",
        yaxis="y2", line=dict(color=NEUTRAL_DARK, width=2), marker=dict(size=5),
        hovertemplate="%{x}<br>Cumulative: %{y:.1f}%<extra></extra>"))

    fig.add_shape(type="line", x0=0, x1=1, y0=80, y1=80, xref="paper", yref="y2",
                  line=dict(dash="dot", color=MUTED_COLOR, width=1))
    fig.add_annotation(x=1.0, y=80, xref="paper", yref="y2", xanchor="right",
                       yanchor="bottom", text="80% line", showarrow=False,
                       font=dict(color=MUTED_COLOR, size=9))
    hit = np.where(cum >= 80.0)[0]
    if hit.size:
        i = int(hit[0])
        fig.add_annotation(x=ids[i], y=float(cum[i]), xref="x", yref="y2",
                           text=f"{i + 1} SKUs = 80% of returns cost",
                           showarrow=True, arrowcolor=NEUTRAL_DARK, ay=-30,
                           font=dict(color=NEUTRAL_DARK, size=10))

    fig.update_layout(
        template="plotly_white", height=440,
        yaxis=dict(title="Returns cost (refunds + write-offs)",
                   tickprefix="$", tickformat=".2s"),
        yaxis2=dict(title="Cumulative share", overlaying="y", side="right",
                    showgrid=False, ticksuffix="%", range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=20, b=10))
    return fig


def chart_regional_shift(result: AnalysisResult) -> go.Figure:
    """H1 vs H2 revenue share per region, worst decline first, share-point
    change labeled above each pair (red = beyond the alert floor)."""
    regions = result.regions
    if regions.empty or regions["share_h1_pct"].isna().all():
        return _empty_figure("Regional share shift needs data from both "
                             "halves of the year (H1 and H2).")
    plot_df = regions.dropna(subset=["share_change_pp"]) \
                     .sort_values("share_change_pp")
    names = plot_df["customer_region"].tolist()
    h1 = plot_df["share_h1_pct"].to_numpy(dtype=float)
    h2 = plot_df["share_h2_pct"].to_numpy(dtype=float)
    ymax = float(max(h1.max(), h2.max())) or 1.0

    fig = go.Figure()
    fig.add_bar(x=names, y=h1, name="H1 share", marker_color=H1_COLOR,
                hovertemplate="%{x}<br>H1 share: %{y:.1f}%<extra></extra>")
    fig.add_bar(x=names, y=h2, name="H2 share", marker_color=_palette()[0],
                hovertemplate="%{x}<br>H2 share: %{y:.1f}%<extra></extra>")
    for i, row in enumerate(plot_df.itertuples()):
        delta = float(row.share_change_pp)
        color = ALERT_COLOR if delta <= -REGION_SHARE_DROP_FLOOR_PP else NEUTRAL_DARK
        fig.add_annotation(x=names[i], y=float(max(h1[i], h2[i])) + ymax * 0.04,
                           text=f"{delta:+.1f}pp", showarrow=False,
                           font=dict(color=color, size=11))

    fig.update_layout(
        barmode="group", template="plotly_white", height=430,
        yaxis=dict(title="Share of company net revenue", ticksuffix="%",
                   range=[0, ymax * 1.3]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=20, b=10))
    return fig


def chart_segments(result: AnalysisResult) -> go.Figure:
    """Two panels: customers per segment vs net revenue per segment; the
    At Risk segment is drawn in alert red."""
    segs = result.segments
    if segs.empty:
        return _empty_figure("No customer segment data available.")
    names = segs["customer_segment"].tolist()
    colors = [ALERT_COLOR if n == "At Risk" else _palette()[0] for n in names]
    customers = segs["customers"].to_numpy(dtype=float)
    revenue = segs["net_revenue"].to_numpy(dtype=float)
    shares = segs["net_revenue_share_pct"].to_numpy(dtype=float)

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.14)
    fig.add_bar(
        x=customers, y=names, orientation="h", marker_color=colors,
        name="Customers", text=[f"{int(v):,}" for v in customers],
        textposition="outside", showlegend=False,
        hovertemplate="%{y}<br>Customers: %{x:,}<extra></extra>")
    fig.add_bar(
        x=revenue, y=names, orientation="h", marker_color=colors,
        name="Net revenue", text=[f"{fmt_usd(r)} ({s:.0f}%)"
                                  for r, s in zip(revenue, shares)],
        textposition="outside", showlegend=False, row=1, col=2,
        hovertemplate="%{y}<br>Net revenue: %{x:$,.0f}<extra></extra>")

    fig.update_yaxes(autorange="reversed")  # biggest segment on top
    fig.update_xaxes(tickformat=",.0f")
    fig.update_xaxes(tickprefix="$", tickformat=".2s", row=1, col=2)
    fig.update_layout(template="plotly_white", height=380,
                      margin=dict(l=10, r=40, t=20, b=10))
    return fig


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #

def render_kpi_cards(result: AnalysisResult) -> None:
    """KPI header row with latest-vs-prior-month deltas (filtered view)."""
    k, m = result.kpis, result.monthly

    def series_delta(s: pd.Series) -> float | None:
        if len(s) < 2 or pd.isna(s.iloc[-2]) or pd.isna(s.iloc[-1]):
            return None
        return float(s.iloc[-1] - float(s.iloc[-2]))

    def ratio_delta(num: str, den: str) -> float | None:
        if len(m) < 2:
            return None
        r = m[num] / m[den].replace(0, np.nan)
        if pd.isna(r.iloc[-1]) or pd.isna(r.iloc[-2]):
            return None
        return float(r.iloc[-1] - r.iloc[-2])

    lost_series = m["gross_revenue"] - m["net_revenue"]
    row1, row2 = st.columns(4), st.columns(4)
    row1[0].metric("Net revenue", fmt_usd(k["net_revenue"]),
                   delta=fmt_delta_usd(series_delta(m["net_revenue"])))
    row1[1].metric("Net profit", fmt_usd(k["net_profit"]),
                   delta=fmt_delta_usd(series_delta(m["net_profit"])))
    row1[2].metric("Margin", fmt_pct(k["margin_pct"]),
                   delta=fmt_delta_pp(ratio_delta("net_profit", "net_revenue")))
    row1[3].metric("Return rate", fmt_pct(k["return_rate_pct"]),
                   delta=fmt_delta_pp(series_delta(m["return_rate_pct"])),
                   delta_color="inverse")
    row2[0].metric("Revenue lost to returns", fmt_usd(k["refunded_revenue"]),
                   delta=fmt_delta_usd(series_delta(lost_series)))
    row2[1].metric("Returns cost", fmt_usd(k["returns_cost"]),
                   help="Refunded revenue + written-off COGS on returns")
    row2[2].metric("Avg order value", fmt_usd(k["aov"]),
                   delta=fmt_delta_usd(ratio_delta("net_revenue", "orders")))
    row2[3].metric("Orders", f"{k['orders']:,}",
                   delta=(f"{int(series_delta(m['orders'])):+,}"
                          if series_delta(m["orders"]) is not None else None))
    if len(m) >= 2:
        st.caption(f"Deltas compare {m['month_label'].iloc[-1]} with "
                   f"{m['month_label'].iloc[-2]} (the last two months in view).")


def render_action_plan(insights: list[Insight]) -> None:
    """The decision layer: ranked insight table + expandable action cards."""
    st.subheader("🎯 This week's action list")
    st.caption("Ranked by dollars at risk · recomputed live for the current "
               "filters · one slot is reserved for the top product-level "
               "return driver")
    if not insights:
        st.info("No insight crossed the configured thresholds for this view. "
                "Widen the date range or clear filters — or review the "
                "`analysis` thresholds in config.yaml.")
        return

    summary = pd.DataFrame([
        {"#": i, "Insight": ins.title, "Kind": ins.kind.replace("_", " "),
         "$ at risk": ins.impact_usd, "Owner": ins.owner, "Horizon": ins.horizon}
        for i, ins in enumerate(insights, 1)])
    st.dataframe(
        summary, use_container_width=True, hide_index=True,
        column_config={"$ at risk": st.column_config.NumberColumn(
            "$ at risk", format="$%.0f")})

    for i, ins in enumerate(insights, 1):
        with st.expander(f"{i}. {ins.title} · {fmt_usd(ins.impact_usd)} at risk",
                         expanded=(i == 1)):
            impact, owner, horizon = st.columns([1.2, 1.6, 0.8])
            impact.metric("$ at risk", fmt_usd(ins.impact_usd))
            owner.markdown(f"**Owner:** {ins.owner}")
            horizon.markdown(f"**Horizon:** {ins.horizon}")
            st.markdown("**Evidence**")
            for ev in ins.evidence:
                st.markdown(f"- {ev}")
            st.success(f"➡️ **Do this:** {ins.action}")


def _category_table(result: AnalysisResult) -> None:
    st.dataframe(
        result.categories[["category", "net_revenue", "net_revenue_share_pct",
                           "margin_pct", "return_rate_pct", "returns_cost",
                           "top_return_reason"]],
        use_container_width=True, hide_index=True,
        column_config={
            "category": "Category",
            "net_revenue": st.column_config.NumberColumn("Net revenue", format="$%.0f"),
            "net_revenue_share_pct": st.column_config.NumberColumn("Share (%)", format="%.1f"),
            "margin_pct": st.column_config.NumberColumn("Margin (%)", format="%.1f"),
            "return_rate_pct": st.column_config.NumberColumn("Return rate (%)", format="%.1f"),
            "returns_cost": st.column_config.NumberColumn("Returns cost", format="$%.0f"),
            "top_return_reason": "Top return reason"})


def _watchlist_table(result: AnalysisResult) -> None:
    d = result.return_drivers
    if d.empty:
        st.success("No SKU crossed the return thresholds in this view — "
                   "nothing needs delisting.")
        return
    st.dataframe(
        d[["product_id", "product_name", "category", "return_rate_pct", "orders",
           "returned_orders", "returns_cost", "top_return_reason"]],
        use_container_width=True, hide_index=True,
        column_config={
            "product_id": "SKU", "product_name": "Product", "category": "Category",
            "return_rate_pct": st.column_config.NumberColumn("Return rate (%)", format="%.1f"),
            "orders": st.column_config.NumberColumn("Orders", format="%.0f"),
            "returned_orders": st.column_config.NumberColumn("Returned", format="%.0f"),
            "returns_cost": st.column_config.NumberColumn("$ at risk", format="$%.0f"),
            "top_return_reason": "Top return reason"})


def _monthly_table(result: AnalysisResult) -> None:
    st.dataframe(
        result.monthly[["month_label", "orders", "net_revenue", "mom_net_pct",
                        "return_rate_pct", "avg_discount_pct"]],
        use_container_width=True, hide_index=True,
        column_config={
            "month_label": "Month",
            "orders": st.column_config.NumberColumn("Orders", format="%.0f"),
            "net_revenue": st.column_config.NumberColumn("Net revenue", format="$%.0f"),
            "mom_net_pct": st.column_config.NumberColumn("MoM (%)", format="%+.1f"),
            "return_rate_pct": st.column_config.NumberColumn("Return rate (%)", format="%.1f"),
            "avg_discount_pct": st.column_config.NumberColumn("Avg discount (%)", format="%.1f")})


def _segment_table(result: AnalysisResult) -> None:
    st.dataframe(
        result.segments[["customer_segment", "customers", "orders",
                         "net_revenue", "net_revenue_share_pct"]],
        use_container_width=True, hide_index=True,
        column_config={
            "customer_segment": "Segment",
            "customers": st.column_config.NumberColumn("Customers", format="%.0f"),
            "orders": st.column_config.NumberColumn("Orders", format="%.0f"),
            "net_revenue": st.column_config.NumberColumn("Net revenue", format="$%.0f"),
            "net_revenue_share_pct": st.column_config.NumberColumn("Revenue share (%)", format="%.1f")})


def render_charts(df: pd.DataFrame, result: AnalysisResult) -> None:
    """The four analysis tabs: interactive charts + supporting tables."""
    threshold = float(result.meta.get("high_return_pct", 15.0))
    tabs = st.tabs(["📈 Overview", "🩸 Margin bleed",
                    "🗺️ Regions & seasonality", "👥 Customers"])

    with tabs[0]:
        st.markdown("#### Revenue trend & return pressure")
        st.caption("Bars: net revenue + revenue lost to returns (grey) · line: "
                   "monthly return rate · red: months with a large MoM decline")
        st.plotly_chart(chart_revenue_trend(result), use_container_width=True,
                        config=CHART_CONFIG)
        st.markdown("#### Category quadrant: margin vs returns")
        st.caption(f"Bubble size = net revenue · dashed lines: the "
                   f"{threshold:.0f}% return threshold and company-average "
                   f"margin · bottom-right = margin bleed")
        st.plotly_chart(chart_category_quadrant(result), use_container_width=True,
                        config=CHART_CONFIG)
        st.markdown("#### Category scorecard")
        _category_table(result)

    with tabs[1]:
        st.markdown("#### Margin bleed: top SKUs by returns cost")
        st.caption(f"Red bars: SKUs also above the {threshold:.0f}% return-rate "
                   f"threshold (the watchlist) · line: cumulative share across "
                   f"ALL SKUs · dotted: the 80% Pareto line")
        st.plotly_chart(chart_margin_bleed(df, result), use_container_width=True,
                        config=CHART_CONFIG)
        st.markdown("#### Product watchlist")
        _watchlist_table(result)

    with tabs[2]:
        st.markdown("#### Regional share shift")
        st.caption("Each region's slice of company net revenue, H1 vs H2 · "
                   f"label = change in share points · red = beyond the "
                   f"{REGION_SHARE_DROP_FLOOR_PP:.0f}pp alert floor")
        st.plotly_chart(chart_regional_shift(result), use_container_width=True,
                        config=CHART_CONFIG)
        for d in result.seasonal_drops:
            st.warning(f"⚠️ {d['month_label']}: net revenue fell "
                       f"{abs(d['mom_pct']):.1f}% MoM "
                       f"({fmt_usd(d['prev_net_revenue'])} → "
                       f"{fmt_usd(d['net_revenue'])}).")
        st.markdown("#### Monthly detail")
        _monthly_table(result)

    with tabs[3]:
        st.markdown("#### Customer segments: size vs value")
        st.caption("Red = At Risk (90+ days silent) · segments were assigned "
                   "once against the full-period snapshot (Step 2), so date "
                   "filters do not re-derive them")
        st.plotly_chart(chart_segments(result), use_container_width=True,
                        config=CHART_CONFIG)
        _segment_table(result)


def render_memo_section(cfg: dict[str, Any]) -> None:
    """View + download the executive memo from the full pipeline run."""
    path = Path(cfg["paths"]["memo_file"])
    with st.expander("📄 Executive memo (full-period pipeline run)", expanded=False):
        if path.is_file():
            st.caption("Generated by `python run_pipeline.py` for the FULL "
                       "period — not affected by the filters above.")
            text = path.read_text(encoding="utf-8")
            st.markdown(text)
            st.download_button("⬇️ Download executive memo (.md)", data=text,
                               file_name="executive_memo.md",
                               mime="text/markdown")
        else:
            st.info("No memo yet — run `python run_pipeline.py` to generate "
                    "it (along with the static charts).")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def main() -> None:
    init_logging()
    cfg = get_config()
    seed = cfg.get("generation", {}).get("seed")

    st.title("🚀 Data-to-Decision Accelerator")
    st.caption("E-commerce sales performance → this week's actions. Filters "
               "recompute every KPI, chart and action live on the selected "
               "slice.")

    # --- self-bootstrap if the processed data is missing --------------------
    parquet = Path(cfg["paths"]["processed_data"])
    if not parquet.is_file():
        st.error("Processed data not found — this deployment has no "
                 f"`{cfg['paths']['processed_data']}` yet.")
        if st.button("⚙️ Generate the dataset now (~10 s)"):
            if regenerate_dataset():
                st.rerun()
        st.stop()

    try:
        df = load_data()
    except (FileNotFoundError, DataQualityError, ValueError) as exc:
        st.error(f"Could not load the processed data: {exc}")
        st.info("Fix: `python run_pipeline.py --stages process` (or the "
                "Regenerate button after reloading), then refresh.")
        st.stop()

    filters = get_filters(df)
    render_data_status(df, cfg)

    df_view = apply_filters(df, filters)
    if df_view.empty:
        st.warning("The current filters match no orders — widen the date "
                   "range or clear the region/category filters in the "
                   "sidebar.")
        st.stop()

    try:
        result = build_analysis(filters.start, filters.end,
                                filters.regions, filters.categories)
    except ValueError as exc:
        st.warning(str(exc))
        st.stop()
    except Exception as exc:  # noqa: BLE001 — log it, keep the app alive
        logger.exception("Analysis failed for the current filter state")
        st.error(f"Analysis failed: {exc} — full traceback in "
                 f"{cfg['paths']['log_file']}")
        st.stop()

    render_kpi_cards(result)
    st.divider()
    render_action_plan(result.insights)
    st.divider()
    render_charts(df_view, result)
    render_memo_section(cfg)
    st.caption(f"Synthetic data (seed {seed}) · reproduce everything: "
               f"`python run_pipeline.py` · built with Streamlit + Plotly")


if __name__ == "__main__":
    main()
