"""Tests for dataset upload analysis and its display helpers."""

import unittest

import numpy as np
import pandas as pd

import config
from src import upload_analysis as ua
from src import upload_view as uv

try:
    from tests.helpers import make_upload_results
except ImportError:  # run via `unittest discover -s tests`
    from helpers import make_upload_results


def _events(n=40):
    times = pd.Timestamp("2026-01-06 08:00:00") + pd.to_timedelta(np.arange(n) * 2, unit="s")
    return pd.DataFrame({"timestamp": times, "src_ip": ["192.168.1.5"] * n, "dst_ip": [f"8.8.8.{i % 9}" for i in range(n)],
                         "dst_port": [443 + (i % 3) for i in range(n)], "protocol": ["TCP"] * n,
                         "src_bytes": [500] * n, "dst_bytes": [900] * n})


class ReadingTests(unittest.TestCase):
    def test_rejects_empty_and_unreadable_and_header_only(self):
        for data in (b"", b"\x00\x01\x02", b"timestamp,src_ip\n"):
            with self.assertRaises(ua.UploadError):
                ua.read_csv_bytes(data)

    def test_row_limit(self):
        csv = _events(30).to_csv(index=False).encode()
        self.assertEqual(len(ua.read_csv_bytes(csv, max_rows=30)), 30)
        with self.assertRaises(ua.UploadError) as caught:
            ua.read_csv_bytes(csv, max_rows=10)
        self.assertIn("more than", str(caught.exception))


class PreparingTests(unittest.TestCase):
    def test_missing_required_columns_are_listed(self):
        with self.assertRaises(ua.UploadError) as caught:
            ua.prepare_dataset(pd.DataFrame({"timestamp": ["2026-01-06"], "src_ip": ["1.1.1.1"]}))
        message = str(caught.exception)
        self.assertIn("dst_ip", message)
        self.assertIn("Download the sample", message)

    def test_events_mode_computes_counters_and_fills_defaults(self):
        prepared = ua.prepare_dataset(_events())
        self.assertEqual(prepared.mode, "events")
        self.assertTrue(set(config.REQUIRED_COLUMNS) <= set(prepared.flows.columns))
        self.assertEqual(int(prepared.flows["conn_count_1m"].iloc[-1]), 30)  # 40 events 2 s apart, 60 s window
        self.assertTrue(any("duration" in n for n in prepared.notes))
        self.assertFalse(prepared.flows.isna().any().any())

    def test_aggregated_mode_keeps_supplied_counters(self):
        frame = _events(5)
        for column in config.UPLOAD_COUNTER_COLUMNS:
            frame[column] = 7
        prepared = ua.prepare_dataset(frame)
        self.assertEqual(prepared.mode, "aggregated")
        self.assertTrue((prepared.flows["conn_count_1m"] == 7).all())

    def test_cicflowmeter_style_headers_are_recognised(self):
        frame = pd.DataFrame({
            " Source IP": ["10.0.0.1", "10.0.0.2"], " Destination IP": ["8.8.8.8", "8.8.4.4"],
            " Destination Port": [443, 53], " Protocol": [6, 17], " Flow Duration": [2_000_000, 500_000],
            " Total Fwd Packets": [4, 1], " Total Backward Packets": [3, 1],
            "Total Length of Fwd Packets": [800, 60], " Total Length of Bwd Packets": [1200, 120],
            " Timestamp": ["2026-01-06 08:00:00", "2026-01-06 08:00:05"], " Label": ["BENIGN", "DDoS"]})
        prepared = ua.prepare_dataset(frame)
        flows = prepared.flows
        self.assertEqual(list(flows["protocol"]), ["TCP", "UDP"])
        self.assertAlmostEqual(float(flows["duration"].iloc[0]), 2.0)
        self.assertEqual(int(flows["src_packets"].iloc[0]), 4)
        self.assertIn("label", flows.columns)

    def test_unreadable_timestamps_are_dropped_or_rejected(self):
        frame = _events(10)
        frame["timestamp"] = frame["timestamp"].astype(str)
        frame.loc[0, "timestamp"] = "not a date"
        prepared = ua.prepare_dataset(frame)
        self.assertEqual((len(prepared.flows), prepared.rows_dropped), (9, 1))
        frame["timestamp"] = "garbage"
        with self.assertRaises(ua.UploadError):
            ua.prepare_dataset(frame)

    def test_unknown_protocol_becomes_tcp(self):
        frame = _events(3)
        frame["protocol"] = ["TCP", "xyz", 99]
        self.assertEqual(list(ua.prepare_dataset(frame).flows["protocol"]), ["TCP", "TCP", "TCP"])


class EvaluationTests(unittest.TestCase):
    def test_counts_and_rates(self):
        evaluation = ua.evaluate_labels(make_upload_results(), ["Normal", "DDoS", "Port Scan", "C2 Communication"])
        self.assertEqual((evaluation["true_positives"], evaluation["false_positives"],
                          evaluation["false_negatives"], evaluation["true_negatives"]), (3, 1, 1, 3))
        self.assertAlmostEqual(evaluation["precision"], 0.75)
        self.assertAlmostEqual(evaluation["recall"], 0.75)
        self.assertEqual(evaluation["by_label"]["Brute-Force"]["alerted"], 0)
        self.assertFalse(evaluation["labels_match_model_classes"])

    def test_no_label_column_means_no_evaluation(self):
        self.assertIsNone(ua.evaluate_labels(make_upload_results().drop(columns=[config.LABEL_COLUMN]), []))

    def test_benign_spellings(self):
        results = make_upload_results()
        results[config.LABEL_COLUMN] = ["BENIGN"] * len(results)
        self.assertEqual(ua.evaluate_labels(results, [])["attack_events"], 0)

    def test_summary(self):
        summary = ua.summarize_results(make_upload_results())
        self.assertEqual((summary["rows"], summary["alerts"], summary["sources_alerting"]), (8, 4, 3))
        self.assertEqual(summary["types"]["DDoS"], 2)


class SampleTests(unittest.TestCase):
    def test_sample_has_expected_columns_and_labels(self):
        sample = ua.make_sample_events(hours=0.5)
        self.assertTrue(set(ua.REQUIRED_UPLOAD_COLUMNS) <= set(sample.columns))
        self.assertIn("Normal", set(sample["label"]))
        self.assertGreater(sample["label"].nunique(), 3)

    def test_sample_survives_prepare(self):
        prepared = ua.prepare_dataset(ua.make_sample_events(hours=0.5).head(500))
        self.assertEqual(prepared.mode, "events")


class ViewTests(unittest.TestCase):
    def setUp(self):
        self.results = make_upload_results()
        self.summary = ua.summarize_results(self.results)

    def test_cards(self):
        cards = {c["label"]: c["value"] for c in uv.kpi_cards(self.summary)}
        self.assertEqual(cards["Events analysed"], "8")
        self.assertEqual(cards["Alerts raised"], "4")
        self.assertEqual(cards["Alert rate"], "50.0%")
        self.assertEqual(cards["Most common threat"], "DDoS")

    def test_time_series_totals(self):
        series = uv.time_series(self.results)
        self.assertEqual((int(series["Events"].sum()), int(series["Alerts"].sum())), (8, 4))

    def test_score_histogram_counts_every_event(self):
        self.assertEqual(int(uv.score_histogram(self.results)["Events"].sum()), 8)

    def test_alerts_table_sorted_and_filtered(self):
        table = uv.alerts_table(self.results)
        self.assertEqual(list(table["Score"]), sorted(table["Score"], reverse=True))
        self.assertEqual(len(uv.alerts_table(self.results, severities=["Critical"])), 1)
        self.assertEqual(len(uv.alerts_table(self.results, min_score=80)), 2)
        self.assertEqual(len(uv.alerts_table(self.results, threat_types=["Port Scan"])), 1)
        self.assertEqual(len(uv.alerts_table(self.results, limit=2)), 2)

    def test_export_csv_contains_only_alerts(self):
        text = uv.export_csv(self.results).decode()
        self.assertEqual(len(text.strip().splitlines()), 1 + 4)
        self.assertIn("threat_score", text.splitlines()[0])

    def test_evaluation_helpers(self):
        evaluation = ua.evaluate_labels(self.results, [])
        self.assertEqual(len(uv.evaluation_cards(evaluation)), 4)
        self.assertIn("Detection rate (%)", uv.evaluation_table(evaluation).columns)


def _bundle_ready():
    if not config.MODEL_BUNDLE_FILE.exists():
        return False
    from src.models import load_bundle
    return load_bundle().metadata.get("data_source") == "events"


@unittest.skipUnless(_bundle_ready(), "needs a model trained with --source events")
class EndToEndTests(unittest.TestCase):
    def test_sample_upload_is_analysed(self):
        from src.models import load_bundle
        data = ua.make_sample_events(hours=0.5).to_csv(index=False).encode()
        analysis = ua.analyze_bytes(load_bundle(), data)
        self.assertGreater(analysis.summary["rows"], 1000)
        self.assertGreater(analysis.summary["alerts"], 0)
        self.assertIsNotNone(analysis.evaluation)
        self.assertGreater(analysis.evaluation["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
