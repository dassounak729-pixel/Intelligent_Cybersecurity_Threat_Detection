"""Cleaning, feature engineering and dataset splitting for flow records."""

from __future__ import annotations

import ipaddress
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import config
from src.utils import get_logger

logger = get_logger("preprocessing")
_EPSILON = 1e-6


def clean_flows(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates/invalid rows and coerce numeric columns safely.

    Rows lacking a parsable timestamp or source/destination IP are dropped.
    Negative numeric values are clipped to zero; ports are bounded to the
    valid range. Missing numeric values are left as NaN for the model
    pipeline's imputer (so no information leaks across the train/test split).
    """
    start_rows = len(frame)
    cleaned = frame.copy()
    if "flow_id" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset="flow_id")
    cleaned = cleaned.drop_duplicates()
    cleaned[config.TIMESTAMP_COLUMN] = pd.to_datetime(cleaned[config.TIMESTAMP_COLUMN], errors="coerce")
    cleaned = cleaned.dropna(subset=[config.TIMESTAMP_COLUMN, "src_ip", "dst_ip"])
    numeric = config.NON_NEGATIVE_COLUMNS + ["dst_port", "asset_criticality"]
    for column in numeric:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned[config.NON_NEGATIVE_COLUMNS] = cleaned[config.NON_NEGATIVE_COLUMNS].clip(lower=0)
    cleaned["dst_port"] = cleaned["dst_port"].clip(0, config.MAX_PORT)
    cleaned["asset_criticality"] = cleaned["asset_criticality"].fillna(1).clip(1, 4).astype(int)
    cleaned["protocol"] = cleaned["protocol"].astype(str).str.upper().str.strip()
    cleaned = cleaned.reset_index(drop=True)
    logger.debug("Cleaned flows: %d -> %d rows", start_rows, len(cleaned))
    return cleaned


@lru_cache(maxsize=100_000)
def _is_private(address: str) -> bool:
    """Return True for RFC1918/private IPs; invalid addresses count as external."""
    try:
        return ipaddress.ip_address(address).is_private
    except ValueError:
        return False


def _internal_flag(series: pd.Series) -> pd.Series:
    """Vectorised private-address check using the cached scalar helper."""
    return series.astype(str).map(_is_private).astype(int)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the model feature matrix (columns follow ``config.MODEL_FEATURES``)."""
    features = pd.DataFrame(index=frame.index)
    for column in ("duration", "dst_port", "src_bytes", "dst_bytes", "src_packets", "dst_packets",
                   "failed_logins", "conn_count_1m", "unique_dst_ports_1m", "unique_dst_hosts_1m",
                   "dns_query_rate", "syn_ratio", "beacon_interval_std"):
        features[column] = frame[column]
    hour = pd.to_datetime(frame[config.TIMESTAMP_COLUMN]).dt.hour
    start, end = config.BUSINESS_HOURS
    features["hour_of_day"] = hour
    features["is_off_hours"] = ((hour < start) | (hour >= end)).astype(int)
    features["src_is_internal"] = _internal_flag(frame["src_ip"])
    features["dst_is_internal"] = _internal_flag(frame["dst_ip"])
    total_bytes = frame["src_bytes"] + frame["dst_bytes"]
    total_packets = frame["src_packets"] + frame["dst_packets"]
    features["total_bytes"] = total_bytes
    features["bytes_ratio"] = (frame["src_bytes"] + 1) / (frame["dst_bytes"] + 1)
    features["avg_packet_size"] = total_bytes / (total_packets + _EPSILON)
    features["packets_per_second"] = total_packets / (frame["duration"] + 0.01)
    for protocol in config.PROTOCOLS:
        features[f"protocol_{protocol}"] = (frame["protocol"] == protocol).astype(int)
    return features[config.MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)


def split_dataset(frame: pd.DataFrame, test_size: float = config.TEST_SIZE,
                  seed: int = config.RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split.

    Event-based data (has ``episode_id``) is split by episode so that events from
    one attack or session never appear on both sides; otherwise the split is
    stratified on the label column.
    """
    if "episode_id" in frame.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(frame, groups=frame["episode_id"]))
        return frame.iloc[train_idx].reset_index(drop=True), frame.iloc[test_idx].reset_index(drop=True)
    train, test = train_test_split(frame, test_size=test_size, random_state=seed,
                                   stratify=frame[config.LABEL_COLUMN])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def prepare_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(features, labels)`` from a cleaned, labelled frame."""
    return engineer_features(frame), frame[config.LABEL_COLUMN].astype(str)
