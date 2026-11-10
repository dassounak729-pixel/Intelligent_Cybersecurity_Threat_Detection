"""Data access layer: reading and writing flow datasets with validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from src.utils import get_logger

logger = get_logger("data_loader")


class DataLoadError(Exception):
    """Raised when an input dataset is missing, unreadable or malformed."""


def load_flows(path: Path, require_label: bool = False) -> pd.DataFrame:
    """Load a flow CSV and validate its schema.

    Args:
        path: Location of the CSV file.
        require_label: If True the ``label`` column must be present.

    Raises:
        DataLoadError: If the file is missing, empty or lacks required columns.
    """
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Input file not found: {path}")
    try:
        frame = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Could not parse {path}: {exc}") from exc
    if frame.empty:
        raise DataLoadError(f"Input file contains no rows: {path}")
    validate_schema(frame, require_label=require_label)
    logger.info("Loaded %d rows from %s", len(frame), path)
    return frame


def validate_schema(frame: pd.DataFrame, require_label: bool = False) -> None:
    """Ensure all required columns exist in ``frame``."""
    required = list(config.REQUIRED_COLUMNS)
    if require_label:
        required.append(config.LABEL_COLUMN)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataLoadError(f"Missing required columns: {missing}")


def save_flows(frame: pd.DataFrame, path: Path) -> None:
    """Persist a flow dataframe to CSV, creating parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Saved %d rows to %s", len(frame), path)
