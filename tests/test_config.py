"""tests/test_config.py — config loading, env-var precedence, logging setup."""
from __future__ import annotations

import logging
import logging.handlers

import pytest

from src.config import get_config, setup_logging


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Keep dev .env values from leaking into config tests."""
    for var in ("DATA_DIR", "SEED", "LOG_LEVEL", "CONFIG_PATH"):
        monkeypatch.delenv(var, raising=False)


def test_repo_config_has_required_sections():
    cfg = get_config()
    for section in ("paths", "generation", "cleaning", "analysis",
                    "segmentation", "visualization", "logging"):
        assert section in cfg, f"config.yaml is missing section '{section}'"
    assert cfg["cleaning"]["missing_price_strategy"] in ("median", "category_median", "drop")


def test_env_overrides_yaml(monkeypatch):
    monkeypatch.setenv("SEED", "7")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    cfg = get_config()
    assert cfg["generation"]["seed"] == 7
    assert cfg["logging"]["level"] == "DEBUG"


def test_data_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    cfg = get_config()
    assert cfg["paths"]["raw_data"] == str(tmp_path / "raw_sales.csv")
    assert cfg["paths"]["processed_data"] == str(tmp_path / "processed_sales.parquet")


def test_config_path_env(monkeypatch, tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text("generation:\n  seed: 99\n", encoding="utf-8")
    monkeypatch.setenv("CONFIG_PATH", str(custom))
    assert get_config() == {"generation": {"seed": 99}}


def test_setup_logging_configures_root(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    root = logging.getLogger()
    level_before = root.level
    handlers_before = [(h, h.formatter) for h in root.handlers]
    try:
        setup_logging()
        assert root.level == logging.WARNING
        assert any(isinstance(h, logging.handlers.RotatingFileHandler)
                   for h in root.handlers)
    finally:
        root.setLevel(level_before)
        root.handlers[:] = [h for h, _ in handlers_before]
        for handler, formatter in handlers_before:
            handler.setFormatter(formatter)
