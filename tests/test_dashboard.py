"""Tests for the live dashboard: state tracking, web server and background runner."""

import json
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

import config
from src.dashboard_server import DashboardServer, validate_run_params
from src.dashboard_state import DashboardState
from src.live_runner import LiveRunner

try:
    from tests.helpers import make_results as _results
except ImportError:  # run via `unittest discover -s tests`
    from helpers import make_results as _results


class StateTests(unittest.TestCase):
    def setUp(self):
        self.state = DashboardState()
        self.token = self.state.reset(hours=1, speed=100)

    def test_fresh_snapshot_is_json_serialisable(self):
        snapshot = self.state.snapshot()
        json.dumps(snapshot)
        self.assertEqual(snapshot["events"], 0)
        self.assertEqual(snapshot["total_minutes"], 60)

    def test_record_batch_accumulates(self):
        self.state.begin(self.token, 100)
        self.state.record_batch(self.token, _results())
        s = self.state.snapshot()
        self.assertEqual((s["events"], s["alerts_raw"], s["alerts_emitted"]), (6, 3, 2))
        self.assertEqual(s["types"], {"DDoS": 1, "Port Scan": 1})
        self.assertEqual(s["actions"]["Block"], 1)
        self.assertEqual(sum(s["series"]["events"]), 6)
        self.assertEqual(sum(s["series"]["alerts"]), 2)
        self.assertEqual(s["status"], "running")

    def test_ground_truth_counters(self):
        self.state.record_batch(self.token, _results())
        truth = self.state.snapshot()["truth"]
        self.assertEqual((truth["attack_events"], truth["attack_flagged"]), (3, 2))
        self.assertEqual((truth["normal_events"], truth["false_alerts_emitted"]), (3, 1))

    def test_newest_alert_first_and_nan_is_json_safe(self):
        self.state.record_batch(self.token, _results())
        s = self.state.snapshot()
        json.dumps(s)
        self.assertEqual([a["threat_type"] for a in s["alerts"]], ["Port Scan", "DDoS"])

    def test_stale_token_is_ignored_after_reset(self):
        new_token = self.state.reset(hours=2, speed=50)
        self.state.record_batch(self.token, _results())
        self.state.finish(self.token, {"x": 1})
        s = self.state.snapshot()
        self.assertEqual((s["events"], s["status"], s["hours"]), (0, "starting", 2))
        self.state.record_batch(new_token, _results())
        self.assertEqual(self.state.snapshot()["events"], 6)

    def test_percentiles_and_failure_state(self):
        self.state.record_batch(self.token, _results())
        self.assertIsNotNone(self.state.snapshot()["latency_ms"]["p95"])
        self.state.fail(self.token, "boom")
        s = self.state.snapshot()
        self.assertEqual((s["status"], s["message"]), ("error", "boom"))


class ValidationTests(unittest.TestCase):
    def test_accepts_values_in_range(self):
        self.assertEqual(validate_run_params({"hours": 2, "speed": "120"}), (2.0, 120.0))

    def test_rejects_bad_input(self):
        for payload in ({}, {"hours": "x", "speed": 1}, {"hours": 99, "speed": 100},
                        {"hours": 2, "speed": 0.1}, {"hours": None, "speed": 100}):
            with self.assertRaises(ValueError):
                validate_run_params(payload)


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.restarts = []
        self.state = DashboardState()
        self.server = DashboardServer(self.state, lambda h, s: self.restarts.append((h, s)), "127.0.0.1", 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = "http://%s:%d" % self.server.address

    def tearDown(self):
        self.server.close()

    def _post(self, body, headers=None):
        request = urllib.request.Request(self.base + "/api/restart", data=body, method="POST",
                                         headers=headers if headers is not None else
                                         {"Content-Type": "application/json", "X-Requested-With": "dashboard"})
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def test_serves_page_and_state(self):
        with urllib.request.urlopen(self.base + "/") as response:
            page = response.read().decode()
            self.assertIn("Threat Detection", page)
            self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])
        with urllib.request.urlopen(self.base + "/api/state") as response:
            self.assertIn("status", json.load(response))

    def test_unknown_route_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(self.base + "/nope")
        self.assertEqual(caught.exception.code, 404)

    def test_restart_requires_header_and_valid_values(self):
        body = json.dumps({"hours": 1, "speed": 100}).encode()
        self.assertEqual(self._post(body, headers={"Content-Type": "application/json"})[0], 403)
        self.assertEqual(self._post(json.dumps({"hours": 99, "speed": 100}).encode())[0], 400)
        self.assertEqual(self._post(b"not json")[0], 400)
        self.assertEqual(self.restarts, [])
        status, payload = self._post(body)
        self.assertEqual((status, payload["ok"]), (200, True))
        self.assertEqual(self.restarts, [(1.0, 100.0)])


def _bundle_ready():
    if not config.MODEL_BUNDLE_FILE.exists():
        return False
    from src.models import load_bundle
    return load_bundle().metadata.get("data_source") == "events"


@unittest.skipUnless(_bundle_ready(), "needs a model trained with --source events")
class RunnerIntegrationTests(unittest.TestCase):
    def test_short_run_finishes_with_summary(self):
        from src.models import load_bundle
        state = DashboardState()
        runner = LiveRunner(load_bundle(), state)
        runner.start(hours=0.5, speed=config.DASHBOARD_SPEED_RANGE[1])
        runner.wait(timeout=180)
        snapshot = state.snapshot()
        self.assertEqual(snapshot["status"], "finished", snapshot["message"])
        self.assertEqual(snapshot["events"], snapshot["total_events"])
        self.assertIsNotNone(snapshot["summary"])
        self.assertGreater(snapshot["alerts_emitted"], 0)
        json.dumps(snapshot)

    def test_restart_replaces_run(self):
        from src.models import load_bundle
        state = DashboardState()
        runner = LiveRunner(load_bundle(), state)
        runner.start(hours=0.5, speed=10)  # slow: still running when restarted
        runner.start(hours=0.5, speed=config.DASHBOARD_SPEED_RANGE[1])
        runner.wait(timeout=180)
        self.assertEqual(state.snapshot()["status"], "finished")


if __name__ == "__main__":
    unittest.main()
