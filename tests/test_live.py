"""Unit tests for the live-detection building blocks."""

import unittest

import numpy as np
import pandas as pd

import config
from src import event_simulator
from src.event_source import PacedEventSource, ReplayEventSource
from src.live_detector import apply_cooldown
from src.stream_features import SlidingWindowAggregator


class AggregatorTests(unittest.TestCase):
    def test_counts_unique_ports_and_hosts_in_window(self):
        agg = SlidingWindowAggregator(window_seconds=60)
        out = [agg.update(t, "10.0.0.1", f"1.1.1.{t}", 1000 + t, 0) for t in range(5)]
        self.assertEqual(out[-1][:3], (5, 5, 5))

    def test_old_events_leave_the_window(self):
        agg = SlidingWindowAggregator(window_seconds=60)
        for t in range(5):
            agg.update(t, "10.0.0.1", "1.1.1.1", 443, 0)
        conn, ports, hosts, *_ = agg.update(200, "10.0.0.1", "1.1.1.1", 443, 0)
        self.assertEqual((conn, ports, hosts), (1, 1, 1))

    def test_failed_logins_and_dns_are_counted_then_expire(self):
        agg = SlidingWindowAggregator(window_seconds=60)
        agg.update(0, "a", "b", 22, 1)
        agg.update(1, "a", "b", 22, 1)
        _, _, _, failed, dns, _ = agg.update(2, "a", "ns", 53, 0)
        self.assertEqual((failed, dns), (2, 1))
        _, _, _, failed, dns, _ = agg.update(100, "a", "b", 22, 0)
        self.assertEqual((failed, dns), (0, 0))

    def test_sources_are_independent(self):
        agg = SlidingWindowAggregator()
        agg.update(0, "a", "x", 80, 0)
        self.assertEqual(agg.update(1, "b", "x", 80, 0)[0], 1)

    def test_regular_beacon_has_lower_std_than_irregular_traffic(self):
        regular, irregular = SlidingWindowAggregator(), SlidingWindowAggregator()
        rng = np.random.default_rng(0)
        for i in range(10):
            reg = regular.update(i * 5.0, "a", "c2", 443, 0)[5]
            irr = irregular.update(float(np.cumsum(rng.uniform(1, 60, 10))[i]), "a", "c2", 443, 0)[5]
        self.assertLess(reg, 0.01)
        self.assertGreater(irr, reg)

    def test_short_channel_uses_default(self):
        agg = SlidingWindowAggregator()
        self.assertEqual(agg.update(0, "a", "b", 80, 0)[5], config.LIVE_BEACON_DEFAULT)

    def test_process_matches_incremental_updates_and_index(self):
        events = event_simulator.simulate_events(1, config.LIVE_START, {"Port Scan": 2}, 1).head(300)
        batch = SlidingWindowAggregator().process(events)
        self.assertEqual(list(batch.columns), SlidingWindowAggregator.OUTPUT_COLUMNS)
        self.assertTrue(batch.index.equals(events.index))
        self.assertFalse(batch.isna().any().any())

    def test_purge_releases_idle_sources(self):
        agg = SlidingWindowAggregator(window_seconds=60)
        agg.update(0, "a", "b", 80, 0)
        agg.purge(10_000)
        self.assertEqual(agg.active_sources, 0)


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.events = event_simulator.simulate_events(1, config.LIVE_START, {"Port Scan": 3, "C2 Communication": 2,
                                                                          "Insider Misuse": 2}, seed=5)

    def test_time_ordered_with_labels_and_episodes(self):
        self.assertTrue(self.events["timestamp"].is_monotonic_increasing)
        self.assertTrue({"Normal", "Port Scan", "C2 Communication", "Insider Misuse"} <= set(self.events["label"]))
        self.assertFalse(self.events["episode_id"].isna().any())

    def test_deterministic_for_same_seed(self):
        again = event_simulator.simulate_events(1, config.LIVE_START, {"Port Scan": 3, "C2 Communication": 2,
                                                                       "Insider Misuse": 2}, seed=5)
        pd.testing.assert_frame_equal(self.events, again)

    def test_port_scan_touches_many_ports(self):
        scan = self.events[self.events["label"] == "Port Scan"]
        self.assertGreater(scan.groupby("episode_id")["dst_port"].nunique().min(), 4)

    def test_required_columns_present_after_features(self):
        flows = event_simulator.with_window_features(self.events.head(200))
        self.assertTrue(set(config.REQUIRED_COLUMNS) <= set(flows.columns))


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.events = event_simulator.simulate_events(1, config.LIVE_START, {}, seed=2).head(60)

    def test_paced_source_yields_every_event_in_order(self):
        source = PacedEventSource(self.events, speed=1e6, poll_seconds=0.001)
        batches = list(source.batches())
        merged = pd.concat(batches)
        self.assertEqual(len(merged), len(self.events))
        self.assertTrue(merged["timestamp"].is_monotonic_increasing)
        self.assertTrue(merged["emitted_at"].is_monotonic_increasing)

    def test_paced_source_rejects_bad_speed(self):
        with self.assertRaises(ValueError):
            PacedEventSource(self.events, speed=0)

    def test_replay_source_batches(self):
        batches = list(ReplayEventSource(self.events, batch_size=25).batches())
        self.assertEqual([len(b) for b in batches], [25, 25, 10])


class CooldownTests(unittest.TestCase):
    def _frame(self, offsets, types=None):
        n = len(offsets)
        return pd.DataFrame({
            config.TIMESTAMP_COLUMN: pd.Timestamp("2026-01-06 08:00:00") + pd.to_timedelta(offsets, unit="s"),
            "src_ip": ["1.2.3.4"] * n, "threat_type": types or ["DDoS"] * n, "is_alert": [True] * n})

    def test_repeat_alerts_within_cooldown_are_suppressed(self):
        emit = apply_cooldown(self._frame([0, 10, 59, 61]), {}, 60)
        self.assertEqual(list(emit), [True, False, False, True])

    def test_different_threat_type_is_not_suppressed(self):
        emit = apply_cooldown(self._frame([0, 5], ["DDoS", "Port Scan"]), {}, 60)
        self.assertEqual(list(emit), [True, True])

    def test_state_persists_across_batches(self):
        state = {}
        apply_cooldown(self._frame([0]), state, 60)
        self.assertEqual(list(apply_cooldown(self._frame([30]), state, 60)), [False])

    def test_non_alerts_never_emit(self):
        frame = self._frame([0, 1])
        frame["is_alert"] = False
        self.assertFalse(apply_cooldown(frame, {}, 60).any())


if __name__ == "__main__":
    unittest.main()
