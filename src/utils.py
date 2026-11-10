"""Shared helpers: logging, filesystem preparation, JSON I/O and timing."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

import config

LOGGER_NAME = "threat_detection"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger of the project logger."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def setup_logging(level: str = config.LOG_LEVEL, log_file: Path | None = config.LOG_FILE) -> None:
    """Configure console (and optional file) logging once per process."""
    root = logging.getLogger(LOGGER_NAME)
    if root.handlers:
        return
    root.setLevel(level.upper())
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                                  datefmt="%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def ensure_directories() -> None:
    """Create every project output/data directory if it does not exist."""
    for directory in (config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR, config.MODELS_DIR,
                      config.FIGURES_DIR, config.REPORTS_DIR, config.ALERTS_DIR,
                      config.LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _json_default(value: Any) -> Any:
    """Make numpy scalars/arrays JSON serialisable."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def save_json(payload: dict, path: Path) -> None:
    """Write ``payload`` to ``path`` as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def load_json(path: Path) -> dict:
    """Read a JSON file, raising a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@contextmanager
def timed(label: str) -> Iterator[dict]:
    """Context manager that logs and records elapsed wall-clock seconds."""
    holder: dict = {}
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder["seconds"] = time.perf_counter() - start
        get_logger("timer").info("%s finished in %.2fs", label, holder["seconds"])
