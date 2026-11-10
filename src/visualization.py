"""Matplotlib charts. Functions only plot what they are given and save to disk."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("visualization")


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=config.FIGURE_DPI)
    plt.close(figure)
    logger.info("Saved figure %s", path)
    return path


def plot_class_distribution(labels: pd.Series, path: Path) -> Path:
    """Bar chart of traffic class counts (log scale to show rare attacks)."""
    counts = labels.value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(counts.index, counts.values, color=config.COLOR_PRIMARY)
    ax.set_yscale("log")
    ax.set_title("Traffic Class Distribution")
    ax.set_ylabel("Flows (log scale)")
    ax.bar_label(bars, padding=2, fontsize=8)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def plot_model_comparison(comparison: pd.DataFrame, path: Path) -> Path:
    """Grouped bars of cross-validated accuracy and macro-F1 per model."""
    x = np.arange(len(comparison))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, comparison["cv_accuracy"], 0.4, label="Accuracy", color=config.COLOR_PRIMARY)
    ax.bar(x + 0.2, comparison["cv_f1_macro"], 0.4, label="Macro F1", color=config.COLOR_ACCENT)
    ax.set_xticks(x, comparison["model"], rotation=15)
    ax.set_ylim(max(0.0, comparison[["cv_accuracy", "cv_f1_macro"]].min().min() - 0.1), 1.0)
    ax.set_title("Model Comparison (cross-validation)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def plot_confusion_matrix(matrix: list[list[int]], labels: list[str], path: Path) -> Path:
    """Row-normalised confusion matrix heatmap with counts annotated."""
    cm = np.array(matrix)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)")
    fig.colorbar(image, ax=ax, fraction=0.046)
    return _save(fig, path)


def plot_feature_importance(importances: pd.Series, path: Path, top_n: int = 12) -> Path:
    """Horizontal bars of the most important features."""
    top = importances.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index, top.values, color=config.COLOR_PRIMARY)
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Relative importance")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def plot_score_distribution(results: pd.DataFrame, path: Path) -> Path:
    """Histogram of threat scores for benign vs malicious flows."""
    benign = results.loc[results[config.LABEL_COLUMN] == config.NORMAL_LABEL, "threat_score"]
    malicious = results.loc[results[config.LABEL_COLUMN] != config.NORMAL_LABEL, "threat_score"]
    bins = np.linspace(0, 100, 41)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(benign, bins=bins, alpha=0.75, label="Benign", color=config.COLOR_NEUTRAL)
    ax.hist(malicious, bins=bins, alpha=0.75, label="Malicious", color=config.COLOR_ACCENT)
    ax.axvline(config.ALERT_SCORE_THRESHOLD, color="black", linestyle="--", label="Alert threshold")
    ax.set_yscale("log")
    ax.set_xlabel("Threat score")
    ax.set_ylabel("Flows (log scale)")
    ax.set_title("Threat Score Distribution")
    ax.legend()
    return _save(fig, path)


def plot_alert_timeline(results: pd.DataFrame, path: Path) -> Path:
    """Line chart of alerts per hour over the observation window."""
    alerts = results[results["is_alert"]].copy()
    alerts["hour"] = pd.to_datetime(alerts[config.TIMESTAMP_COLUMN]).dt.floor("h")
    per_hour = alerts.groupby("hour").size()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(per_hour.index, per_hour.values, color=config.COLOR_ACCENT, linewidth=1.2)
    ax.set_title("Alerts per Hour")
    ax.set_xlabel("Time")
    ax.set_ylabel("Alerts")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_alert_breakdown(results: pd.DataFrame, path: Path) -> Path:
    """Pie chart of alert threat types and bar chart of recommended actions."""
    alerts = results[results["is_alert"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    types = alerts["threat_type"].value_counts()
    ax1.pie(types, labels=types.index, autopct="%1.1f%%", startangle=140, textprops={"fontsize": 8})
    ax1.set_title("Alerts by Threat Type")
    actions = alerts["action"].value_counts()
    ax2.bar(actions.index, actions.values, color=config.COLOR_PRIMARY)
    ax2.set_title("Recommended Actions")
    ax2.set_ylabel("Alerts")
    ax2.grid(axis="y", alpha=0.3)
    return _save(fig, path)
