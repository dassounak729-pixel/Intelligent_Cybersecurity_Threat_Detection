"""Tests for the display-preparation functions used by the dashboards."""

import unittest

import pandas as pd

import config
from src import dashboard_view as view
from src.dashboard_state import DashboardState

try:
    from tests.helpers import make_results
except ImportError:  # run via `unittest discover -s tests`
    from helpers import make_results

_SUMMARY = {
    "episodes_detected": 5, "episodes_total": 6, "episode_detection_rate": 5 / 6,
    "normal_event_false_alert_rate": 0.005,
    "detection_by_class": {
        "Port Scan": {"episodes": 3, "detected": 3, "detection_rate": 1.0, "median_seconds_to_detect": 0.0,
                      "first_alert_type_correct": 1.0},
        "DDoS": {"episodes": 3, "detected": 2, "detection_rate": 2 / 3, "median_seconds_to_detect": None,
                 "first_alert_type_correct": None},
    },
}


def _snapshot(finished=False, with_data=True):
    state = DashboardState()
    token = state.reset(hours=1, speed=100)
    if with_data:
        state.begin(token, 6)
        state.record_batch(token, make_results())
    if finished:
        state.finish(token, _SUMMARY)
    return state.snapshot()


class ViewTests(unittest.TestCase):
    def test_empty_snapshot_does_not_break_anything(self):
        snap = _snapshot(with_data=False)
        self.assertTrue(view.events_frame(snap).empty)
        self.assertTrue(view.alerts_per_minute_frame(snap).empty)
        self.assertTrue(view.latency_frame(snap).empty)
        self.assertTrue(view.alerts_table(snap).empty)
        self.assertIsNone(view.summary_table(snap))
        cards = view.kpi_cards(snap)
        self.assertEqual(len(cards), 6)
        self.assertTrue(all(isinstance(c["value"], str) for c in cards))

    def test_series_frames_cover_elapsed_time_only(self):
        snap = _snapshot()
        events = view.events_frame(snap)
        self.assertIsInstance(events.index, pd.DatetimeIndex)
        self.assertLessEqual(len(events), snap["total_minutes"])
        self.assertEqual(int(events["Events per minute"].sum()), 6)
        self.assertEqual(int(view.alerts_per_minute_frame(snap)["Alerts per minute"].sum()), 2)

    def test_finished_run_shows_the_whole_window(self):
        snap = _snapshot(finished=True)
        self.assertEqual(len(view.events_frame(snap)), snap["total_minutes"])

    def test_counts_frame_is_sorted_and_integer(self):
        frame = view.counts_frame({"Block": 1, "Investigate": 5, "Allow": 3}, "Alerts")
        self.assertEqual(list(frame.index), ["Investigate", "Allow", "Block"])
        self.assertTrue(pd.api.types.is_integer_dtype(frame["Alerts"]))

    def test_alerts_table_newest_first_with_expected_columns(self):
        table = view.alerts_table(_snapshot())
        self.assertEqual(list(table["Threat type"]), ["Port Scan", "DDoS"])
        self.assertIn("Evidence", table.columns)
        self.assertEqual(table["Destination"].iat[0], "10.0.0.1:80 TCP")

    def test_kpi_values(self):
        cards = {c["label"]: c["value"] for c in view.kpi_cards(_snapshot())}
        self.assertEqual(cards["Events processed"], "6")
        self.assertEqual(cards["Alerts shown to analyst"], "2")
        self.assertEqual(cards["Attack events flagged"], "67%")

    def test_summary_table_and_headline(self):
        snap = _snapshot(finished=True)
        table = view.summary_table(snap)
        self.assertEqual(list(table["Attack class"]), ["DDoS", "Port Scan"])
        self.assertEqual(int(table.set_index("Attack class").loc["Port Scan", "First alert type correct (%)"]), 100)
        self.assertIn("5 / 6", view.summary_headline(snap))


if __name__ == "__main__":
    unittest.main()
