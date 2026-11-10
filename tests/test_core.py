"""Unit tests for preprocessing, scoring, response policy and data loading."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import data_loader, preprocessing, response, risk_scoring, synthetic_data


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.flows = synthetic_data.generate_flows(600, seed=1)

    def test_feature_matrix_has_expected_columns_and_no_inf(self):
        features = preprocessing.engineer_features(preprocessing.clean_flows(self.flows))
        self.assertEqual(list(features.columns), config.MODEL_FEATURES)
        self.assertFalse(np.isinf(features.to_numpy()).any())

    def test_clean_flows_clips_negatives_and_drops_bad_rows(self):
        dirty = self.flows.copy()
        dirty["timestamp"] = dirty["timestamp"].astype(str)  # as read from CSV
        dirty.loc[0, "src_bytes"] = -50
        dirty.loc[1, "timestamp"] = "not-a-date"
        cleaned = preprocessing.clean_flows(dirty)
        self.assertEqual(len(cleaned), len(dirty) - 1)
        self.assertGreaterEqual(cleaned["src_bytes"].min(), 0)

    def test_split_is_stratified(self):
        train, test = preprocessing.split_dataset(preprocessing.clean_flows(self.flows))
        self.assertEqual(set(train.label), set(test.label))


class ScoringTests(unittest.TestCase):
    def test_anomaly_only_flows_are_not_underscored(self):
        score = risk_scoring.compute_threat_scores(
            np.array([0.0]), np.array([1.0]), np.array([2]), np.array([config.SUSPICIOUS_LABEL]))
        self.assertGreaterEqual(score[0], 90)

    def test_scores_are_bounded_and_scale_with_criticality(self):
        types = np.array(["DDoS", "DDoS"])
        scores = risk_scoring.compute_threat_scores(
            np.array([0.6, 0.6]), np.array([0.5, 0.5]), np.array([1, 4]), types)
        self.assertTrue(((scores >= 0) & (scores <= 100)).all())
        self.assertLess(scores[0], scores[1])

    def test_normal_traffic_scores_low(self):
        score = risk_scoring.compute_threat_scores(
            np.array([0.01]), np.array([0.1]), np.array([2]), np.array(["Normal"]))
        self.assertLess(score[0], config.ALERT_SCORE_THRESHOLD)


class FlaggedNormalTests(unittest.TestCase):
    def test_flagged_flow_called_normal_gets_best_attack_class(self):
        classes = ["Normal", "DDoS", "Port Scan"]
        probs = np.array([[0.45, 0.10, 0.45], [0.9, 0.05, 0.05]])
        scores = np.array([config.ALERT_SCORE_THRESHOLD + 5, 10.0])
        types = np.array(["Normal", "Normal"], dtype=object)
        out = risk_scoring.relabel_flagged_normals(probs, classes, scores, types)
        self.assertEqual(list(out), ["Port Scan", "Normal"])

    def test_other_types_are_left_alone(self):
        out = risk_scoring.relabel_flagged_normals(
            np.array([[0.1, 0.9]]), ["Normal", "DDoS"], np.array([99.0]), np.array(["DDoS"], dtype=object))
        self.assertEqual(list(out), ["DDoS"])


class ResponseTests(unittest.TestCase):
    def test_action_depends_on_score_band_and_type(self):
        scores = np.array([10.0, 40.0, 60.0, 95.0, 95.0, 95.0])
        types = np.array(["Normal", "Normal", "DDoS", "C2 Communication", "Lateral Movement",
                          config.SUSPICIOUS_LABEL], dtype=object)
        actions = list(response.recommend_actions(scores, types))
        self.assertEqual(actions, ["Allow", "Monitor", "Investigate", "Quarantine", "Escalate", "Investigate"])

    def test_severity_bands(self):
        severity = risk_scoring.severity_for_scores(np.array([5.0, 30.0, 55.0, 80.0, 99.0]))
        self.assertEqual(list(severity), ["Info", "Low", "Medium", "High", "Critical"])

    def test_every_threat_type_has_mitre_mapping(self):
        for label in config.ATTACK_LABELS:
            self.assertIn(label, config.MITRE_MAPPING)


class DataLoaderTests(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(data_loader.DataLoadError):
            data_loader.load_flows(Path("does_not_exist.csv"))

    def test_missing_columns_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            pd.DataFrame({"a": [1]}).to_csv(path, index=False)
            with self.assertRaises(data_loader.DataLoadError):
                data_loader.load_flows(path)

    def test_roundtrip(self):
        flows = synthetic_data.generate_flows(100, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flows.csv"
            data_loader.save_flows(flows, path)
            self.assertEqual(len(data_loader.load_flows(path, require_label=True)), len(flows))


if __name__ == "__main__":
    unittest.main()
