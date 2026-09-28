"""Central config loader: merges config.yaml with .env overrides.

Every module imports from here — no module reads config/env directly.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Load config.yaml once, then apply environment-variable overrides."""
    config_path = Path(os.getenv("CONFIG_PATH", REPO_ROOT / "config.yaml"))
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- Env vars always win over YAML ---
    if data_dir := os.getenv("DATA_DIR"):
        for key in ("raw_data", "processed_data"):
            cfg["paths"][key] = str(Path(data_dir) / Path(cfg["paths"][key]).name)
    if seed := os.getenv("SEED"):
        cfg["generation"]["seed"] = int(seed)
    if level := os.getenv("LOG_LEVEL"):
        cfg["logging"]["level"] = level.upper()

    return cfg


def setup_logging() -> logging.Logger:
    """Configure console + rotating file logging from config."""
    cfg = get_config()["logging"]
    root = logging.getLogger()
    root.setLevel(cfg["level"])
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    if cfg.get("console", True) and not root.handlers:
        root.addHandler(logging.StreamHandler())

    if cfg.get("file", True):
        log_path = REPO_ROOT / get_config()["paths"]["log_file"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        root.addHandler(logging.handlers.RotatingFileHandler(
            log_path, maxBytes=5_000_000, backupCount=2, encoding="utf-8"
        ))

    for h in root.handlers:
        h.setFormatter(fmt)
    return logging.getLogger("pipeline")


def get_logger(name: str) -> logging.Logger:
    """Per-module logger. Call once at import: logger = get_logger(__name__)."""
    return logging.getLogger(name)
