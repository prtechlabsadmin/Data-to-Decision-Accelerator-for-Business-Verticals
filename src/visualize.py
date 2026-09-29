"""src/visualize.py — Step 4: decision-oriented chart export.

Pipeline position: consumes data/processed_sales.parquet (via
src.analysis.load_processed_data + analyze, recomputed for a single source of
truth) and exports static charts to insights/charts/.

Charts produced (names are numbered so they sort correctly):

    01_revenue_trend              monthly net revenue + revenue lost to returns
                                  (stacked), return-rate line (right axis),
                                  MoM-drop months flagged in red
    02_category_margin_vs_returns margin-vs-return quadrant; bubble size =
                                  net revenue; threshold + company-margin lines
    03_return_pareto              SKU margin bleed: top SKUs by returns cost,
                                  cumulative-share line, 80% Pareto line
    04_regional_share_shift       H1 vs H2 revenue share per region with
                                  change-in-share-point labels
    05_customer_segments          segment size (customers) vs value (revenue)

Design principles:
    - Every chart answers ONE leadership question; thresholds are drawn ON the
      chart (a chart without its decision boundary is decoration).
    - Empty/thin data renders a labeled placeholder, never a crash — the
      artifact always exists, so the pipeline stage always verifies.
    - One failing chart never kills the export (per-chart isolation).
    - matplotlib only (no seaborn, no browser engine), Agg backend: safe on
      headless CI. Styling/format/dpi/palette come from config.yaml.
    - Figures are owned by the caller until save_chart() persists + closes them.

Module contract (see run_pipeline.py): main() -> Path (charts directory).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")  # headless export — must run before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from src.analysis import (  # noqa: E402
    REGION_SHARE_DROP_FLOOR_PP,
    AnalysisResult,
    analyze,
    load_processed_data,
)
from src.config import get_config, get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

__all__ = [
    "VisualizationError",
    "plot_revenue_trend",
    "plot_category_margin",
    "plot_return_pareto",
    "plot_regional_performance",
    "plot_customer_segments",
    "save_chart",
    "export_charts",
    "main",
]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SUPPORTED_FORMATS: tuple[str, ...] = ("png", "svg", "pdf")
PARETO_TOP_N = 12                  # bars shown in the margin-bleed chart
PARETO_VALUE_LABELS = 5            # annotate $ values on the first N bars only

ALERT_COLOR = "#d62728"            # drops, thresholds, at-risk
POSITIVE_COLOR = "#2e7d32"
NEUTRAL_DARK = "#3f3f3f"
MUTED_COLOR = "#7f7f7f"
STACK_COLOR = "#c9c9c9"            # revenue lost to returns (stacked wedge)
H1_COLOR = "#bdbdbd"               # the "before" bars in the regional chart

_STYLE: dict[str, Any] = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
}


class VisualizationError(RuntimeError):
    """Chart export failed entirely (every chart raised)."""


# --------------------------------------------------------------------------- #
# Config, style & formatting helpers (pure -> easy unit-test targets)
# --------------------------------------------------------------------------- #

def _viz_cfg() -> dict[str, Any]:
    return get_config().get("visualization", {})


def _figsize(wf: float = 1.0, hf: float = 1.0) -> tuple[float, float]:
    fs = _viz_cfg().get("figure_size", [10, 6])
    return float(fs[0]) * wf, float(fs[1]) * hf


def _cmap():
    name = str(_viz_cfg().get("palette", "Set2"))
    try:
        return colormaps[name]
    except KeyError as exc:
        raise ValueError(
            f"visualization.palette '{name}' is not a known matplotlib "
            f"colormap") from exc


def _apply_style() -> None:
    """Idempotent rcParams styling so every figure looks consistent."""
    plt.rcParams.update(_STYLE)


def _money_fmt(x: float, _pos: int | None = None) -> str:
    if abs(x) >= 1_000_000:
        return f"${x / 1e6:.1f}M"
    if abs(x) >= 1_000:
        return f"${x / 1e3:.0f}K"
    return f"${x:,.0f}"


def _pct_fmt(x: float, _pos: int | None = None) -> str:
    return f"{x:.0f}%"


def _compact_usd(x: float) -> str:
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"${x / 1e6:.2f}M"
    if abs(x) >= 10_000:
        return f"${x / 1e3:.0f}K"
    if abs(x) >= 1_000:
        return f"${x / 1e3:.1f}K"
    return f"${x:,.0f}"


def _scale_sizes(
    values: "np.ndarray | list[float] | pd.Series",
    s_min: float = 90.0,
    s_max: float = 2400.0,
) -> np.ndarray:
    """Map revenue values to bubble areas (min-max, NaN/zero-safe)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    vmax = float(np.nanmax(v))
    if not np.isfinite(vmax) or vmax <= 0:
        return np.full(v.shape, s_min)
    return s_min + (np.nan_to_num(v, nan=0.0) / vmax) * (s_max - s_min)


def _category_colors(categories: list[str]) -> dict[str, Any]:
    cmap = _cmap()
    return {cat: cmap(i % cmap.N) for i, cat in enumerate(dict.fromkeys(categories))}


def _footnote(result: AnalysisResult | None) -> str:
    if result is None:
        return "Synthetic data | generated by src/visualize.py"
    k, m = result.kpis, result.meta
    seed = m.get("seed")
    seed_txt = f"seed {seed}" if seed is not None else "seed n/a"
    return (f"Synthetic data ({seed_txt}) | {k['period_start']:%b %Y}-"
            f"{k['period_end']:%b %Y} | {k['orders']:,} orders | "
            f"generated by src/visualize.py")


def _finalize(fig: Figure, title: str, subtitle: str, footnote: str) -> None:
    """Consistent title / subtitle / footnote treatment for every chart."""
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=9,
             color=MUTED_COLOR)
    fig.text(0.005, 0.005, footnote, ha="left", va="bottom", fontsize=7.5,
             color=MUTED_COLOR)
    fig.tight_layout(rect=(0, 0.04, 1, 0.89))


def _placeholder(message: str) -> Figure:
    """Labeled empty-state chart: the artifact exists, nothing crashes."""
    _apply_style()
    fig, ax = plt.subplots(figsize=_figsize())
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=MUTED_COLOR, wrap=True)
    return fig


def _validate_viz_cfg(cfg: dict[str, Any]) -> None:
    """Fail fast on impossible visualization config."""
    required = ("chart_format", "chart_dpi", "palette", "figure_size")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config.yaml 'visualization' section is missing "
                         f"key(s): {', '.join(missing)}")
    fmt = str(cfg["chart_format"]).lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"visualization.chart_format must be one of "
                         f"{SUPPORTED_FORMATS}, got '{cfg['chart_format']}'")
    try:
        dpi = int(cfg["chart_dpi"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"visualization.chart_dpi must be an integer, got "
                         f"{cfg['chart_dpi']!r}") from exc
    if dpi < 72:
        raise ValueError(f"visualization.chart_dpi must be >= 72, got {dpi}")
    fs = cfg["figure_size"]
    if not isinstance(fs, (list, tuple)) or len(fs) != 2:
        raise ValueError("visualization.figure_size must be a list of two "
                         "numbers, e.g. [10, 6]")
    try:
        w, h = float(fs[0]), float(fs[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"visualization.figure_size values must be numeric, "
                         f"got {fs!r}") from exc
    if w <= 0 or h <= 0:
        raise ValueError(f"visualization.figure_size values must be positive, "
                         f"got {[w, h]}")
    try:
        colormaps[str(cfg["palette"])]
    except KeyError as exc:
        raise ValueError(f"visualization.palette '{cfg['palette']}' is not a "
                         f"known matplotlib colormap") from exc


# --------------------------------------------------------------------------- #
# Chart 1 — revenue trend with return overlay
# --------------------------------------------------------------------------- #

def plot_revenue_trend(result: AnalysisResult) -> Figure:
    """Monthly net revenue (bars) + revenue lost to returns (stacked wedge)
    + monthly return rate (right-axis line); MoM-drop months flagged red."""
    monthly = result.monthly
    if monthly.empty or float(monthly["gross_revenue"].sum()) <= 0:
        return _placeholder("No monthly revenue data available to plot.")

    _apply_style()
    fig, ax = plt.subplots(figsize=_figsize(1.0, 0.95))

    x = np.arange(len(monthly))
    net = np.asarray(monthly["net_revenue"], dtype=float)
    lost = np.clip(np.asarray(monthly["gross_revenue"], dtype=float) - net,
                   0.0, None)

    cmap = _cmap()
    base = cmap(0)
    line_color = cmap(1)
    drop_months = {d["order_year_month"] for d in result.seasonal_drops}
    bar_colors = [ALERT_COLOR if ym in drop_months else base
                  for ym in monthly["order_year_month"]]

    ax.bar(x, net, width=0.72, color=bar_colors, label="Net revenue", zorder=2)
    ax.bar(x, lost, width=0.72, bottom=net, color=STACK_COLOR,
           label="Revenue lost to returns", zorder=2)

    ym_to_pos = {ym: i for i, ym in enumerate(monthly["order_year_month"])}
    for d in result.seasonal_drops:
        pos = ym_to_pos.get(d["order_year_month"])
        if pos is None:
            continue
        ax.annotate(f"-{abs(float(d['mom_pct'])):.0f}% MoM",
                    xy=(pos, net[pos] + lost[pos]), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    fontweight="bold", color=ALERT_COLOR)

    rot = 0 if len(x) <= 12 else 45
    ax.set_xticks(x)
    ax.set_xticklabels(monthly["month_label"], rotation=rot,
                       ha="right" if rot else "center")
    ax.set_ylabel("Revenue")
    ax.yaxis.set_major_formatter(FuncFormatter(_money_fmt))
    top = float((net + lost).max())
    ax.set_ylim(0, top * 1.22 if top > 0 else 1.0)

    ax2 = ax.twinx()
    ax2.plot(x, monthly["return_rate_pct"], color=line_color, marker="o",
             linewidth=2, markersize=4, label="Monthly return rate", zorder=3)
    ax2.set_ylabel("Return rate", color=line_color)
    ax2.tick_params(axis="y", labelcolor=line_color)
    ax2.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    rr_max = float(monthly["return_rate_pct"].max()) if len(monthly) else 0.0
    ax2.set_ylim(0, max(rr_max * 1.5, 10.0))
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(line_color)
    ax2.spines["top"].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3)

    mom_thr = 100.0 * abs(float(
        get_config().get("analysis", {}).get("mom_decline_threshold", -0.10)))
    subtitle = ("Bars: net revenue + revenue lost to returns (grey) | line: "
                "monthly return rate | red: months with a large MoM decline")
    _finalize(fig, "Revenue trend & return pressure", subtitle,
              _footnote(result) + f" | drop threshold: {mom_thr:.0f}% MoM")
    return fig


# --------------------------------------------------------------------------- #
# Chart 2 — category quadrant: margin vs returns
# --------------------------------------------------------------------------- #

def plot_category_margin(result: AnalysisResult) -> Figure:
    """The decision quadrant: return rate (x) vs net margin (y), bubble size =
    net revenue, with the return threshold and company-average margin drawn
    on the chart. Bottom-right bubbles are the margin bleed."""
    cats = result.categories
    if cats.empty:
        return _placeholder("No category data available to plot.")

    _apply_style()
    fig, ax = plt.subplots(figsize=_figsize())
    colors = _category_colors(cats["category"].tolist())
    threshold = float(result.meta.get("high_return_pct", 15.0))
    sizes = _scale_sizes(cats["net_revenue"])

    for i, row in enumerate(cats.itertuples()):
        ax.scatter(row.return_rate_pct, row.margin_pct, s=sizes[i],
                   color=colors[row.category], alpha=0.85, edgecolors="white",
                   linewidths=1.4, zorder=3)
        ax.annotate(f"{row.category}{' *' if row.low_volume else ''}",
                    (row.return_rate_pct, row.margin_pct), xytext=(8, 6),
                    textcoords="offset points", fontsize=9, zorder=4)

    rates = np.asarray(cats["return_rate_pct"], dtype=float)
    margins = np.asarray(cats["margin_pct"], dtype=float)
    ax.set_xlim(0.0, float(np.nanmax(rates)) * 1.18 + 1.0)

    company_margin = float(result.kpis["margin_pct"])
    y_lo, y_hi = float(np.nanmin(margins)), float(np.nanmax(margins))
    if np.isfinite(company_margin):
        y_lo, y_hi = min(y_lo, company_margin), max(y_hi, company_margin)
    y_span = (y_hi - y_lo) or abs(y_hi) or 1.0
    ax.set_ylim(y_lo - 0.25 * y_span, y_hi + 0.25 * y_span)

    ax.axvline(threshold, color=ALERT_COLOR, linestyle="--", linewidth=1.2,
               zorder=2)
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_lim = ax.get_ylim()
    y_range = y_lim[1] - y_lim[0]
    ax.text(threshold + 0.012 * x_span, y_lim[1] - 0.03 * y_range,
            f"return threshold {threshold:.0f}%", rotation=90, va="top",
            ha="left", fontsize=8.5, color=ALERT_COLOR)
    if np.isfinite(company_margin):
        ax.axhline(company_margin, color=NEUTRAL_DARK, linestyle="--",
                   linewidth=1.2, zorder=2)
        ax.text(ax.get_xlim()[1] - 0.012 * x_span,
                company_margin + 0.03 * y_range,
                f"company margin {company_margin:.1f}%", ha="right",
                va="bottom", fontsize=8.5, color=NEUTRAL_DARK)

    ax.text(0.97, 0.05, "MARGIN BLEED\nhigh returns | thin margin",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            fontweight="bold", color=ALERT_COLOR, alpha=0.8, zorder=1)
    ax.text(0.03, 0.95, "HEALTHY\nlow returns | strong margin",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            fontweight="bold", color=POSITIVE_COLOR, alpha=0.8, zorder=1)

    ax.set_xlabel("Return rate (% of orders)")
    ax.set_ylabel("Net margin (% of net revenue)")
    ax.xaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))

    min_orders = int(get_config().get("analysis", {})
                     .get("min_category_volume", 50))
    subtitle = (f"Bubble size = net revenue | dashed: {threshold:.0f}% return "
                f"threshold & company avg margin | * = fewer than {min_orders} "
                f"orders (low volume)")
    _finalize(fig, "Category quadrant: margin vs returns", subtitle,
              _footnote(result))
    return fig


# --------------------------------------------------------------------------- #
# Chart 3 — margin bleed Pareto (needs the order-level frame)
# --------------------------------------------------------------------------- #

def plot_return_pareto(
    df: pd.DataFrame, result: AnalysisResult | None = None, top_n: int = PARETO_TOP_N
) -> Figure:
    """SKUs ranked by returns cost (refunds + write-offs), cumulative share
    line across ALL SKUs, and the 80% Pareto line. Outlined bars are the SKUs
    that also exceed the return-rate threshold (the watchlist)."""
    if df.empty:
        return _placeholder("No order data available to plot.")

    tmp = df.assign(_writeoff=df["cogs"] * df["return_flag"])
    per_sku = (tmp.groupby("product_id")
                  .agg(refunded=("revenue_lost_to_returns", "sum"),
                       writeoff=("_writeoff", "sum"),
                       orders=("order_id", "count")))
    per_sku["returns_cost"] = per_sku["refunded"] + per_sku["writeoff"]
    total = float(per_sku["returns_cost"].sum())
    if total <= 0:
        return _placeholder("No returns in this period - no margin is "
                            "bleeding through returns.")

    per_sku = per_sku.sort_values("returns_cost", ascending=False)
    top_n = max(int(top_n), 1)
    top = per_sku.head(top_n)
    cum = (per_sku["returns_cost"].cumsum() / total).to_numpy()[: len(top)]

    driver_ids: set[str] = set()
    threshold = 15.0
    if result is not None:
        if not result.return_drivers.empty:
            driver_ids = set(result.return_drivers["product_id"])
        threshold = float(result.meta.get("high_return_pct", threshold))

    _apply_style()
    fig, ax = plt.subplots(figsize=_figsize())
    x = np.arange(len(top))
    base = _cmap()(0)
    edge_colors = [ALERT_COLOR if pid in driver_ids else "none"
                   for pid in top.index]
    ax.bar(x, top["returns_cost"], width=0.7, color=base,
           edgecolor=edge_colors, linewidth=1.8, zorder=2)

    for i, v in enumerate(top["returns_cost"].to_numpy()[:PARETO_VALUE_LABELS]):
        ax.annotate(_compact_usd(float(v)), (i, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=NEUTRAL_DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(top.index, rotation=40, ha="right")
    ax.set_ylabel("Returns cost (refunds + write-offs)")
    ax.yaxis.set_major_formatter(FuncFormatter(_money_fmt))
    ax.set_ylim(0, float(top["returns_cost"].max()) * 1.15)

    ax2 = ax.twinx()
    ax2.plot(x, cum, color=NEUTRAL_DARK, marker="o", markersize=4,
             linewidth=1.8, zorder=3)
    ax2.set_ylabel("Cumulative share of returns cost", color=NEUTRAL_DARK)
    ax2.tick_params(axis="y", labelcolor=NEUTRAL_DARK)
    ax2.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ylim_top = max(float(cum[-1]) * 1.15, 0.25)
    hit = np.where(cum >= 0.8)[0]
    if hit.size:
        ylim_top = max(ylim_top, 0.92)
    ax2.set_ylim(0, ylim_top)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)

    if ylim_top > 0.82:
        ax2.axhline(0.8, color=MUTED_COLOR, linestyle=":", linewidth=1)
        ax2.text(0.99, 0.8, " 80% line", transform=ax2.get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=8, color=MUTED_COLOR)
    if hit.size:
        i = int(hit[0])
        ax2.annotate(f"{i + 1} SKUs = 80% of all returns cost",
                     xy=(i, float(cum[i])), xytext=(10, -14),
                     textcoords="offset points", fontsize=8.5,
                     color=NEUTRAL_DARK,
                     arrowprops=dict(arrowstyle="->", color=NEUTRAL_DARK,
                                     lw=0.8))

    subtitle = (f"Top {len(top)} of {len(per_sku)} SKUs by returns cost | "
                f"line: cumulative share across all SKUs | outlined bars: "
                f"return rate at or above the {threshold:.0f}% threshold")
    _finalize(fig, "Margin bleed: top SKUs by returns cost", subtitle,
              _footnote(result))
    return fig


# --------------------------------------------------------------------------- #
# Chart 4 — regional share shift (the drop-off detector)
# --------------------------------------------------------------------------- #

def plot_regional_performance(result: AnalysisResult) -> Figure:
    """H1 vs H2 share of company net revenue per region, worst decline first,
    with the share-point change labeled above each pair."""
    regions = result.regions
    if regions.empty:
        return _placeholder("No regional data available to plot.")
    if regions["share_h1_pct"].isna().all():
        return _placeholder("Regional share shift needs data from both "
                            "halves of the year (H1 and H2).")

    plot_df = regions.dropna(subset=["share_change_pp"]) \
                     .sort_values("share_change_pp")  # worst decline first
    dropped_rows = len(regions) - len(plot_df)
    if dropped_rows:
        logger.warning("Regional chart: %d region(s) without half-year shares "
                       "were skipped", dropped_rows)

    _apply_style()
    fig, ax = plt.subplots(figsize=_figsize())
    x = np.arange(len(plot_df))
    w = 0.38
    h1 = np.asarray(plot_df["share_h1_pct"], dtype=float)
    h2 = np.asarray(plot_df["share_h2_pct"], dtype=float)

    ax.bar(x - w / 2, h1, width=w, color=H1_COLOR, label="H1 share", zorder=2)
    ax.bar(x + w / 2, h2, width=w, color=_cmap()(0), label="H2 share",
           zorder=2)

    y_max = float(max(h1.max(), h2.max()))
    ax.set_ylim(0, y_max * 1.32 if y_max > 0 else 1.0)
    for i, row in enumerate(plot_df.itertuples()):
        delta = float(row.share_change_pp)
        y = max(h1[i], h2[i]) + y_max * 0.04
        color = (ALERT_COLOR if delta <= -REGION_SHARE_DROP_FLOOR_PP
                 else NEUTRAL_DARK)
        ax.annotate(f"{delta:+.1f}pp", (i, y), ha="center", fontsize=9,
                    fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["customer_region"])
    ax.set_ylabel("Share of company net revenue")
    ax.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    ax.legend(loc="upper left")

    subtitle = ("Each region's slice of company net revenue, first half vs "
                "second half | label: change in share points | red = beyond "
                f"the {REGION_SHARE_DROP_FLOOR_PP:.0f}pp alert floor")
    _finalize(fig, "Regional share shift: where the drop-offs are", subtitle,
              _footnote(result))
    return fig


# --------------------------------------------------------------------------- #
# Chart 5 — customer segments: size vs value
# --------------------------------------------------------------------------- #

def plot_customer_segments(result: AnalysisResult) -> Figure:
    """Two panels: customers per RFM-lite segment (size) vs net revenue and
    revenue share (value). The At Risk segment is drawn in alert red."""
    segs = result.segments
    if segs.empty:
        return _placeholder("No customer segment data available to plot.")

    _apply_style()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=_figsize(1.1, 0.8),
                                     sharey=True)
    y = np.arange(len(segs))[::-1]  # biggest segment on top
    names = segs["customer_segment"].tolist()
    base = _cmap()(0)
    bar_colors = [ALERT_COLOR if s == "At Risk" else base for s in names]

    customers = np.asarray(segs["customers"], dtype=float)
    revenue = np.asarray(segs["net_revenue"], dtype=float)
    shares = np.asarray(segs["net_revenue_share_pct"], dtype=float)

    ax_l.barh(y, customers, color=bar_colors, zorder=2)
    ax_r.barh(y, revenue, color=bar_colors, zorder=2)
    for a in (ax_l, ax_r):  # barh charts want x-grid, not the default y-grid
        a.yaxis.grid(False)
        a.xaxis.grid(True, alpha=0.25)

    ax_l.set_yticks(y)
    ax_l.set_yticklabels(names)
    for i, v in enumerate(customers):
        ax_l.annotate(f"{int(v):,}", (v, y[i]), xytext=(5, 0),
                      textcoords="offset points", va="center", fontsize=8.5,
                      color=NEUTRAL_DARK)
    for i, v in enumerate(revenue):
        ax_r.annotate(f"{_compact_usd(float(v))} ({shares[i]:.0f}%)",
                      (v, y[i]), xytext=(5, 0), textcoords="offset points",
                      va="center", fontsize=8.5, color=NEUTRAL_DARK)

    ax_l.set_xlabel("Customers")
    ax_r.set_xlabel("Net revenue")
    ax_r.xaxis.set_major_formatter(FuncFormatter(_money_fmt))
    ax_l.set_xlim(0, customers.max() * 1.2 if customers.max() > 0 else 1.0)
    ax_r.set_xlim(0, revenue.max() * 1.28 if revenue.max() > 0 else 1.0)

    subtitle = ("Left: customers per RFM-lite segment | right: net revenue "
                "(share of company net revenue) | red = At Risk (90+ days "
                "silent)")
    _finalize(fig, "Customer segments: size vs value", subtitle,
              _footnote(result))
    return fig


# --------------------------------------------------------------------------- #
# Persistence & export
# --------------------------------------------------------------------------- #

def save_chart(fig: Figure, name: str, out_dir: Path | str) -> Path:
    """Save a figure using the configured format/dpi, then close it.

    Callers that do NOT save a figure are responsible for closing it
    (plt.close(fig)) to avoid leaking figures.
    """
    viz = _viz_cfg()
    fmt = str(viz.get("chart_format", "png")).lower()
    dpi = int(viz.get("chart_dpi", 150))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Chart saved -> %s (%.0f KB)", path,
                path.stat().st_size / 1024)
    return path


def export_charts(
    df: pd.DataFrame, result: AnalysisResult, out_dir: Path | str
) -> list[Path]:
    """Render and save all five charts; returns the saved paths.

    Per-chart failure isolation: one broken chart is logged and skipped; the
    export only raises if EVERY chart fails.
    """
    _validate_viz_cfg(_viz_cfg())
    specs: list[tuple[str, Callable[[], Figure]]] = [
        ("01_revenue_trend", lambda: plot_revenue_trend(result)),
        ("02_category_margin_vs_returns", lambda: plot_category_margin(result)),
        ("03_return_pareto", lambda: plot_return_pareto(df, result)),
        ("04_regional_share_shift", lambda: plot_regional_performance(result)),
        ("05_customer_segments", lambda: plot_customer_segments(result)),
    ]
    out_dir = Path(out_dir)
    saved: list[Path] = []
    failed: list[str] = []
    for name, plot in specs:
        try:
            saved.append(save_chart(plot(), name, out_dir))
        except Exception:  # one bad chart must not kill the stage
            failed.append(name)
            logger.exception("Chart '%s' failed to render/save", name)

    if not saved:
        raise VisualizationError(
            f"all {len(specs)} charts failed ({', '.join(failed)}) - full "
            f"tracebacks in {get_config()['paths']['log_file']}")
    if failed:
        logger.warning("%d/%d charts failed and were skipped: %s",
                       len(failed), len(specs), ", ".join(failed))
    logger.info("Exported %d/%d charts -> %s", len(saved), len(specs), out_dir)
    return saved


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def main() -> Path:
    """Run Step 4 end-to-end; return the charts directory path."""
    cfg = get_config()
    _validate_viz_cfg(cfg.get("visualization", {}))

    df = load_processed_data(Path(cfg["paths"]["processed_data"]))
    logger.info("Recomputing analysis for chart export (same single source "
                "of truth as the executive memo)")
    result = analyze(df)

    out_dir = Path(cfg["paths"]["charts_dir"])
    export_charts(df, result, out_dir)
    return out_dir


if __name__ == "__main__":
    setup_logging()
    main()
