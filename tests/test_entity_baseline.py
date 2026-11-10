"""Unit tests for per-entity behavioural baselines."""

import unittest

import numpy as np
import pandas as pd

import config
from src import preprocessing, synthetic_data
from src.entity_baseline import EntityBaselineStore, cross_fit_features


def _flows(n=400, seed=0):
    return preprocessing.clean_flows(synthetic_data.generate_flows(n, seed=seed))


class EntityBaselineTests(unittest.TestCase):
    def setUp(self):
        flows = _flows(3000)
        self.benign = flows[flows[config.LABEL_COLUMN] == config.NORMAL_LABEL].reset_index(drop=True)
        self.store = EntityBaselineStore(min_flows=5).fit(self.benign)

    def test_output_columns_and_no_missing_values(self):
        features = self.store.transform(self.benign.head(50))
        self.assertEqual(list(features.columns), config.ENTITY_FEATURES)
        self.assertFalse(features.isna().any().any())

    def test_flow_far_from_own_baseline_scores_higher(self):
        row = self.benign.iloc[[0]].copy()
        normal_z = self.store.transform(row)["entity_max_z"].iloc[0]
        extreme = row.copy()
        extreme["src_bytes"] = row["src_bytes"] * 5000
        extreme["conn_count_1m"] = row["conn_count_1m"] * 200
        self.assertGreater(self.store.transform(extreme)["entity_max_z"].iloc[0], normal_z + 2)

    def test_unknown_entity_falls_back_to_global(self):
        row = self.benign.iloc[[0]].copy()
        row[config.ENTITY_KEY] = "203.0.113.250"
        features = self.store.transform(row)
        self.assertEqual(features["entity_known"].iloc[0], 0)
        self.assertEqual(features["entity_new_port"].iloc[0], 0)

    def test_new_port_flag_for_known_entity(self):
        row = self.benign.iloc[[0]].copy()
        row["dst_port"] = 49999
        self.assertEqual(self.store.transform(row)["entity_new_port"].iloc[0], 1)

    def test_unusual_hour_has_higher_surprisal(self):
        row = self.benign.iloc[[0]].copy()
        hours = self.store.hour_prob_[self.store.index_.get_loc(row[config.ENTITY_KEY].iloc[0])]
        common, rare = int(np.argmax(hours)), int(np.argmin(hours))
        out = []
        for hour in (common, rare):
            r = row.copy()
            r[config.TIMESTAMP_COLUMN] = pd.Timestamp("2026-01-05") + pd.Timedelta(hours=hour)
            out.append(self.store.transform(r)["entity_hour_surprisal"].iloc[0])
        self.assertGreater(out[1], out[0])

    def test_cross_fit_matches_shape_and_has_no_missing(self):
        flows = _flows(1500, seed=2)
        mask = (flows[config.LABEL_COLUMN] == config.NORMAL_LABEL).to_numpy()
        features = cross_fit_features(flows, mask, n_splits=3)
        self.assertEqual(features.shape, (len(flows), len(config.ENTITY_FEATURES)))
        self.assertFalse(features.isna().any().any())

    def test_fit_on_empty_raises(self):
        with self.assertRaises(ValueError):
            EntityBaselineStore().fit(self.benign.iloc[0:0])

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            EntityBaselineStore().transform(self.benign.head(2))


if __name__ == "__main__":
    unittest.main()
