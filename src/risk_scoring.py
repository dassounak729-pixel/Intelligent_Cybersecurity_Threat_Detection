"""Risk scoring: combine classifier belief, anomaly score and asset context."""

from __future__ import annotations

import numpy as np

import config


def malicious_probability(probabilities: np.ndarray, classes: list[str]) -> np.ndarray:
    """Probability that a flow is *not* normal (sum over all attack classes)."""
    normal_idx = classes.index(config.NORMAL_LABEL)
    return 1.0 - probabilities[:, normal_idx]


def assign_threat_types(probabilities: np.ndarray, classes: list[str],
                        anomaly_scores: np.ndarray) -> np.ndarray:
    """Pick the most likely class; relabel anomalous "Normal" flows as Suspicious Traffic."""
    predicted = np.array(classes, dtype=object)[probabilities.argmax(axis=1)]
    zero_day = (predicted == config.NORMAL_LABEL) & (anomaly_scores >= config.SUSPICIOUS_ANOMALY_THRESHOLD)
    predicted[zero_day] = config.SUSPICIOUS_LABEL
    return predicted


def compute_threat_scores(p_malicious: np.ndarray, anomaly_scores: np.ndarray,
                          criticality: np.ndarray, threat_types: np.ndarray) -> np.ndarray:
    """Threat score on a 0-100 scale.

    ``base = w_cls * P(malicious) + w_anom * anomaly``; anomaly-only detections
    are floored at ``ANOMALY_ONLY_SCORE_CAP * anomaly`` so unknown threats are
    not under-scored. The result is scaled by an asset-criticality multiplier.
    """
    base = config.CLASSIFIER_WEIGHT * p_malicious + config.ANOMALY_WEIGHT * anomaly_scores
    suspicious = threat_types == config.SUSPICIOUS_LABEL
    base = np.where(suspicious, np.maximum(base, config.ANOMALY_ONLY_SCORE_CAP * anomaly_scores), base)
    multiplier = np.array([config.CRITICALITY_MULTIPLIER.get(int(c), 1.0) for c in criticality])
    return np.clip(base * multiplier, 0.0, 1.0) * 100.0


def severity_for_scores(scores: np.ndarray) -> np.ndarray:
    """Map scores to severity labels using ``config.SEVERITY_BANDS``."""
    severity = np.empty(len(scores), dtype=object)
    lower = -np.inf
    for upper, label, _ in config.SEVERITY_BANDS:
        severity[(scores >= lower) & (scores < upper)] = label
        lower = upper
    return severity


def relabel_flagged_normals(probabilities: np.ndarray, classes: list[str], scores: np.ndarray,
                            threat_types: np.ndarray) -> np.ndarray:
    """Give alert-level flows labelled "Normal" their most likely attack class.

    A flow can reach the alert threshold while "Normal" is still the single most
    likely class (the attack probability is split across several classes). An alert
    called "Normal" is meaningless to an analyst, so use the best attack class.
    """
    flagged = (threat_types == config.NORMAL_LABEL) & (scores >= config.ALERT_SCORE_THRESHOLD)
    if not flagged.any():
        return threat_types
    attack_idx = [i for i, label in enumerate(classes) if label != config.NORMAL_LABEL]
    best = np.array(classes, dtype=object)[attack_idx][probabilities[flagged][:, attack_idx].argmax(axis=1)]
    relabelled = threat_types.copy()
    relabelled[flagged] = best
    return relabelled
