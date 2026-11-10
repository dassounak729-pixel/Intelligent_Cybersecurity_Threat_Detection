"""Explainability: show which features made a flow look anomalous."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _format_value(value: float) -> str:
    """Compact human-readable number."""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.3g}"


def explain_flows(X: pd.DataFrame, importances: pd.Series, baseline_median: pd.Series,
                  baseline_log_mean: pd.Series, baseline_log_std: pd.Series,
                  top_k: int = config.EVIDENCE_TOP_K) -> list[str]:
    """Return one evidence string per row of ``X``.

    Each feature is scored by ``global importance x |z|`` where ``z`` is the
    deviation (in log space) from the benign baseline. The ``top_k`` features
    exceeding ``EVIDENCE_MIN_Z`` are reported with their value and the benign
    median for context.
    """
    if X.empty:
        return []
    columns = list(X.columns)
    z_scores = (np.log1p(X.clip(lower=0)) - baseline_log_mean[columns]) / baseline_log_std[columns]
    contribution = z_scores.abs().to_numpy() * importances[columns].to_numpy()
    z_values = z_scores.to_numpy()
    raw_values = X.to_numpy()
    medians = baseline_median[columns].to_numpy()
    order = np.argsort(-contribution, axis=1)[:, :top_k]
    evidence = []
    for row, picks in enumerate(order):
        parts = [
            f"{columns[col]}={_format_value(raw_values[row, col])} (normal~{_format_value(medians[col])})"
            for col in picks if abs(z_values[row, col]) >= config.EVIDENCE_MIN_Z
        ]
        evidence.append("; ".join(parts) if parts else "No dominant single-feature deviation")
    return evidence
