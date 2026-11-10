"""Detection engine: raw flows in, scored and explained alerts out."""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np
import pandas as pd

import config
from src import explainability, preprocessing, response, risk_scoring
from src.models import ModelBundle
from src.utils import get_logger

logger = get_logger("detection")

_PASSTHROUGH = ["flow_id", "timestamp", "src_ip", "src_port", "dst_ip", "dst_port",
                "protocol", "duration", "user", "device", "asset_criticality", config.LABEL_COLUMN]


class ThreatDetector:
    """Applies a trained :class:`ModelBundle` to network-flow records."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    def analyze(self, raw_flows: pd.DataFrame) -> pd.DataFrame:
        """Score every flow and return a result frame (one row per valid flow)."""
        flows = preprocessing.clean_flows(raw_flows)
        features = preprocessing.engineer_features(flows)
        if self.bundle.entity_baseline is not None:
            features = features.join(self.bundle.entity_baseline.transform(flows))
        features = features[self.bundle.feature_columns]
        probabilities = self.bundle.classifier.predict_proba(features)
        classes = list(self.bundle.classifier.classes_)
        anomaly = self.bundle.anomaly_detector.score(features)
        p_malicious = risk_scoring.malicious_probability(probabilities, classes)
        threat_types = risk_scoring.assign_threat_types(probabilities, classes, anomaly)
        scores = risk_scoring.compute_threat_scores(
            p_malicious, anomaly, flows["asset_criticality"].to_numpy(), threat_types)

        threat_types = risk_scoring.relabel_flagged_normals(probabilities, classes, scores, threat_types)

        result = flows[[c for c in _PASSTHROUGH if c in flows.columns]].copy()
        result["p_malicious"] = p_malicious
        result["anomaly_score"] = anomaly
        result["threat_score"] = scores
        result["threat_type"] = threat_types
        result["confidence"] = np.where(threat_types == config.SUSPICIOUS_LABEL, anomaly,
                                        probabilities.max(axis=1))
        result["severity"] = risk_scoring.severity_for_scores(scores)
        result["action"] = response.recommend_actions(scores, threat_types)
        result["mitre_tactic"], result["mitre_technique"] = response.mitre_columns(threat_types)
        if "entity_max_z" in features.columns:
            result["entity_deviation"] = features["entity_max_z"].to_numpy()
        result["is_alert"] = scores >= config.ALERT_SCORE_THRESHOLD
        result["evidence"] = ""
        self._attach_evidence(result, features)
        return result

    def _attach_evidence(self, result: pd.DataFrame, features: pd.DataFrame) -> None:
        """Compute explanations only for flows that raised an alert."""
        alert_idx = result.index[result["is_alert"]]
        if len(alert_idx) == 0:
            return
        b = self.bundle
        result.loc[alert_idx, "evidence"] = explainability.explain_flows(
            features.loc[alert_idx], b.importances, b.baseline_median,
            b.baseline_log_mean, b.baseline_log_std)

    def analyze_stream(self, raw_flows: pd.DataFrame,
                       batch_size: int = config.STREAM_BATCH_SIZE) -> Iterator[tuple[pd.DataFrame, float]]:
        """Simulate near-real-time processing, yielding ``(results, latency_seconds)`` per batch."""
        ordered = raw_flows.sort_values(config.TIMESTAMP_COLUMN).reset_index(drop=True)
        for start in range(0, len(ordered), batch_size):
            batch = ordered.iloc[start:start + batch_size]
            tick = time.perf_counter()
            result = self.analyze(batch)
            yield result, time.perf_counter() - tick
