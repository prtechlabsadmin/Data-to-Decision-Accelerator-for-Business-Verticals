"""tests/test_visualize.py — formatters, config validation, chart shapes,
placeholders over crashes, and per-chart export failure isolation."""
from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import src.visualize as viz  # noqa: E402
from helpers import fact_frame, fact_row  # noqa: E402
from src.analysis import AnalysisResult  # noqa: E402
from src.visualize import (  # noqa: E402
    VisualizationError,
    _compact_usd,
    _money_fmt,
    _pct_fmt,
    _scale_sizes,
    _validate_viz_cfg,
    export_charts,
    plot_category_margin,
    plot_customer_segments,
    plot_regional_performance,
    plot_return_pareto,
    plot_revenue_trend,
    save_chart,
)

VIZ_TEST_CFG = {
    "paths": {"log_file": "logs/pipeline.log"},
    "visualization": {"chart_format": "png", "chart_dpi": 72,
                      "palette": "Set2", "figure_size": [6, 4]},
    "analysis": {"mom_decline_threshold": -0.10, "min_category_volume": 50},
    "generation": {"seed": 7},
}


@pytest.fixture
def viz_config(monkeypatch):
    """Hermetic config for every chart function in this module."""
    monkeypatch.setattr(viz, "get_config", lambda: VIZ_TEST_CFG)
    return VIZ_TEST_CFG


def _empty_result() -> AnalysisResult:
    return AnalysisResult(kpis={}, monthly=pd.DataFrame(), seasonal_drops=[],
                          categories=pd.DataFrame(), regions=pd.DataFrame(),
                          return_drivers=pd.DataFrame(), segments=pd.DataFrame(),
                          insights=[], meta={})


# --------------------------------------------------------------------------- #
# Pure formatters
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, "$0"), (999.0, "$999"), (1_000.0, "$1K"), (12_345.0, "$12K"),
     (1_234_567.0, "$1.2M"), (-5_000.0, "-$5K")],
)
def test_money_fmt(value, expected):
    assert _money_fmt(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, "$0"), (999.0, "$999"), (1_000.0, "$1.0K"), (12_345.0, "$12K"),
     (45_678.0, "$46K"), (1_234_567.0, "$1.23M")],
)
def test_compact_usd(value, expected):
    assert _compact_usd(value) == expected


def test_pct_fmt():
    assert _pct_fmt(12.3) == "12%"
    assert _pct_fmt(0.0) == "0%"


def test_scale_sizes_maps_range_and_guards():
    assert _scale_sizes([]).size == 0
    assert (_scale_sizes([0.0, 0.0, 0.0]) == 90.0).all()
    scaled = _scale_sizes([0.0, 50.0, 100.0])
    assert scaled[0] == pytest.approx(90.0)
    assert scaled[1] == pytest.approx(90.0 + 0.5 * 2_310.0)
    assert scaled[2] == pytest.approx(2_400.0)
    assert _scale_sizes([np.nan]) == pytest.approx(90.0)


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #

VALID_VIZ = {"chart_format": "png", "chart_dpi": 150, "palette": "Set2",
             "figure_size": [10, 6]}


def test_validate_viz_cfg_accepts_valid():
    _validate_viz_cfg(VALID_VIZ)  # no raise


def test_validate_viz_cfg_missing_keys():
    with pytest.raises(ValueError, match="missing key"):
        _validate_viz_cfg({})


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"chart_format": "gif"}, "chart_format"),
        ({"chart_dpi": 36}, "chart_dpi"),
        ({"chart_dpi": "fast"}, "chart_dpi"),
        ({"figure_size": [10]}, "figure_size"),
        ({"figure_size": [0, 6]}, "positive"),
        ({"palette": "not_a_real_colormap"}, "colormap"),
    ],
)
def test_validate_viz_cfg_rejects_bad_values(override, fragment):
    with pytest.raises(ValueError, match=fragment):
        _validate_viz_cfg({**VALID_VIZ, **override})


# --------------------------------------------------------------------------- #
# Placeholders over crashes
# --------------------------------------------------------------------------- #

def test_placeholder_charts_instead_of_crashes(viz_config):
    empty = _empty_result()
    for fig in (plot_revenue_trend(empty),
                plot_category_margin(empty),
                plot_regional_performance(empty),
                plot_customer_segments(empty)):
        assert not fig.axes[0].axison      # axis-off placeholder, no crash
        plt.close(fig)


def test_pareto_placeholder_when_no_returns(viz_config):
    df = fact_frame([fact_row("O1"), fact_row("O2")])   # zero returns
    fig = plot_return_pareto(df, None)
    assert not fig.axes[0].axison
    plt.close(fig)


def test_regional_placeholder_when_single_half(viz_config, mini_result):
    regions = mini_result.regions.copy()
    regions[["share_h1_pct", "share_h2_pct", "share_change_pp"]] = np.nan
    single_half = replace(mini_result, regions=regions)
    fig = plot_regional_performance(single_half)
    assert not fig.axes[0].axison
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Chart shapes (mini pipeline)
# --------------------------------------------------------------------------- #

@pytest.mark.integration
def test_all_five_charts_render_with_expected_axes(viz_config, mini_result,
                                                   mini_processed):
    figures = {
        "trend": plot_revenue_trend(mini_result),
        "quadrant": plot_category_margin(mini_result),
        "pareto": plot_return_pareto(mini_processed, mini_result),
        "regional": plot_regional_performance(mini_result),
        "segments": plot_customer_segments(mini_result),
    }
    assert len(figures["trend"].axes) == 2      # bars + right-axis rate line
    assert len(figures["quadrant"].axes) == 1
    assert len(figures["pareto"].axes) == 2     # bars + cumulative share
    assert len(figures["regional"].axes) == 1
    assert len(figures["segments"].axes) == 2   # two side-by-side panels
    for fig in figures.values():
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Persistence & export isolation
# --------------------------------------------------------------------------- #

def test_save_chart_writes_and_closes(viz_config, tmp_path):
    fig = plt.figure()
    path = save_chart(fig, "unit_test_chart", tmp_path)
    assert path.is_file()
    assert plt.get_fignums() == []


PLOT_NAMES = ("plot_revenue_trend", "plot_category_margin",
              "plot_return_pareto", "plot_regional_performance",
              "plot_customer_segments")


@pytest.mark.integration
def test_export_charts_writes_five_files(viz_config, mini_result,
                                         mini_processed, tmp_path):
    saved = export_charts(mini_processed, mini_result, tmp_path)
    assert len(saved) == 5
    assert all(p.suffix == ".png" and p.is_file() for p in saved)


@pytest.mark.integration
def test_export_charts_isolates_single_failure(viz_config, mini_result,
                                               mini_processed, tmp_path,
                                               monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(viz, "plot_return_pareto", boom)
    saved = export_charts(mini_processed, mini_result, tmp_path)
    assert len(saved) == 4
    assert not (tmp_path / "03_return_pareto.png").exists()


@pytest.mark.integration
def test_export_charts_raises_only_when_all_fail(viz_config, mini_result,
                                                 mini_processed, tmp_path,
                                                 monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    for name in PLOT_NAMES:
        monkeypatch.setattr(viz, name, boom)
    with pytest.raises(VisualizationError, match="all 5 charts failed"):
        export_charts(mini_processed, mini_result, tmp_path)
