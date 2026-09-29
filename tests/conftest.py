"""tests/conftest.py — shared fixtures.

The autouse cache-reset keeps the lru_cache'd global config isolated between
tests, so monkeypatched environment variables can never leak across tests.
"""
from __future__ import annotations

import pytest

import helpers
from src.config import get_config


@pytest.fixture(autouse=True)
def _fresh_config_cache():
    get_config.cache_clear()
    yield
    get_config.cache_clear()


@pytest.fixture(scope="module")
def mini_processed():
    """Seeded mini dataset: generate -> clean -> verify -> engineer -> segment."""
    return helpers.build_mini_processed()


@pytest.fixture(scope="module")
def mini_result(mini_processed):
    """Full analysis of the mini dataset (the memo/chart/dashboard input)."""
    from src.analysis import analyze

    return analyze(mini_processed, helpers.ANALYSIS_CFG)
