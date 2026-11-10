"""Evaluation metrics for the classifier, the anomaly layer and the full system."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

import config


def evaluate_classifier(y_true: pd.Series, y_pred: np.ndarray, labels: list[str]) -> dict:
    """Accuracy, macro-F1, per-class report and confusion matrix."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=labels),
        "report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def evaluate_detection(results: pd.DataFrame) -> dict:
    """End-to-end alert quality against ground-truth labels.

    A flow is a true threat if its label is not Normal; it is *flagged* when
    the system raised an alert (threat score >= ``ALERT_SCORE_THRESHOLD``).
    """
    is_threat = (results[config.LABEL_COLUMN] != config.NORMAL_LABEL).to_numpy()
    flagged = results["is_alert"].to_numpy()
    benign = ~is_threat
    metrics = {
        "precision": precision_score(is_threat, flagged, zero_division=0),
        "recall": recall_score(is_threat, flagged, zero_division=0),
        "f1": f1_score(is_threat, flagged, zero_division=0),
        "false_positive_rate": float(flagged[benign].mean()) if benign.any() else 0.0,
        "roc_auc": roc_auc_score(is_threat, results["threat_score"]) if is_threat.any() and benign.any() else None,
        "alerts_raised": int(flagged.sum()),
        "flows_evaluated": int(len(results)),
    }
    per_class = (results[is_threat].groupby(config.LABEL_COLUMN)["is_alert"].mean().to_dict())
    metrics["detection_rate_by_class"] = {k: float(v) for k, v in per_class.items()}
    return metrics


def evaluate_action_distribution(results: pd.DataFrame) -> dict:
    """Counts of recommended actions and severities across alerts."""
    alerts = results[results["is_alert"]]
    return {"actions": alerts["action"].value_counts().to_dict(),
            "severities": alerts["severity"].value_counts().to_dict()}


def evaluate_novel(results: pd.DataFrame) -> dict:
    """Zero-day style evaluation on a held-out attack type never seen in training.

    Reports both the full system and the anomaly layer in isolation (an
    anomaly score at/above ``SUSPICIOUS_ANOMALY_THRESHOLD``), since the
    closed-set classifier will assign the nearest known class to novel traffic.
    """
    novel = results[results[config.LABEL_COLUMN] == config.NOVEL_LABEL]
    benign = results[results[config.LABEL_COLUMN] == config.NORMAL_LABEL]
    threshold = config.SUSPICIOUS_ANOMALY_THRESHOLD
    return {
        "novel_flows": int(len(novel)),
        "benign_flows": int(len(benign)),
        "system_detection_rate": float(novel["is_alert"].mean()) if len(novel) else None,
        "anomaly_only_detection_rate": float((novel["anomaly_score"] >= threshold).mean()) if len(novel) else None,
        "anomaly_only_benign_flag_rate": float((benign["anomaly_score"] >= threshold).mean()) if len(benign) else None,
        "mean_novel_anomaly_score": float(novel["anomaly_score"].mean()) if len(novel) else None,
        "assigned_types": novel["threat_type"].value_counts().to_dict(),
        "benign_false_positive_rate": float(benign["is_alert"].mean()) if len(benign) else None,
    }


def measure_throughput(latencies: list[float], batch_sizes: list[int]) -> dict:
    """Summarise streaming performance."""
    total_seconds = float(sum(latencies))
    return {
        "batches": len(latencies),
        "mean_batch_latency_ms": float(np.mean(latencies) * 1000) if latencies else 0.0,
        "p95_batch_latency_ms": float(np.percentile(latencies, 95) * 1000) if latencies else 0.0,
        "flows_per_second": float(sum(batch_sizes) / total_seconds) if total_seconds else 0.0,
    }
