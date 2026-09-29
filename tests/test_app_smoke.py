"""tests/test_app_smoke.py — pure app helpers + one full AppTest smoke run."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from app import (  # noqa: E402
    Filters,
    _empty_figure,
    _palette,
    apply_filters,
    fmt_delta_pp,
    fmt_delta_usd,
    fmt_pct,
    fmt_usd,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1_500_000.0, "$1.50M"), (45_000.0, "$45K"), (1_500.0, "$1.5K"),
     (999.0, "$999"), (float("nan"), "—")],
)
def test_fmt_usd(value, expected):
    assert fmt_usd(value) == expected


def test_fmt_pct_handles_none():
    assert fmt_pct(12.34) == "12.3%"
    assert fmt_pct(None) == "—"


def test_delta_formatters():
    assert fmt_delta_usd(None) is None
    assert fmt_delta_usd(1_500.0) == "+$1.5K"
    assert fmt_delta_usd(-1_500.0) == "−$1.5K"   # unicode minus, as rendered
    assert fmt_delta_pp(1.5) == "+1.5pp"
    assert fmt_delta_pp(None) is None


def test_empty_figure_carries_message():
    fig = _empty_figure("nothing to show")
    assert not fig.data
    assert fig.layout.annotations[0].text == "nothing to show"


def test_palette_returns_color_list():
    palette = _palette()
    assert palette and all(isinstance(color, str) for color in palette)


def test_apply_filters_slices_the_frame():
    df = pd.DataFrame({
        "order_date": pd.to_datetime(["2024-01-10", "2024-06-15",
                                       "2024-06-20", "2024-11-01"]),
        "customer_region": ["North", "South", "North", "East"],
        "category": ["Apparel", "Apparel", "Beauty", "Apparel"],
        "value": [1, 2, 3, 4],
    })
    narrow = Filters(start=pd.Timestamp("2024-06-01").date(),
                     end=pd.Timestamp("2024-06-30").date(),
                     regions=("South",), categories=("Apparel",))
    assert apply_filters(df, narrow)["value"].tolist() == [2]
    june = Filters(start=pd.Timestamp("2024-06-01").date(),
                   end=pd.Timestamp("2024-06-30").date(),
                   regions=(), categories=())
    assert len(apply_filters(df, june)) == 2


@pytest.mark.integration
@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "processed_sales.parquet").is_file(),
    reason="run the pipeline first: python run_pipeline.py",
)
def test_dashboard_renders_end_to_end():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    assert len(at.metric) >= 8          # the KPI header rows
    assert len(at.dataframe) >= 1       # ranked action summary / tables
    assert len(at.expander) >= 2        # action cards + executive memo
