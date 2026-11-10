"""Runs the simulated live feed on a background thread and feeds the dashboard state."""

from __future__ import annotations

import threading
import time

import pandas as pd

import config
from src import event_simulator
from src.dashboard_state import DashboardState
from src.event_source import PacedEventSource
from src.live_detector import LiveDetector, summarize
from src.models import ModelBundle
from src.utils import get_logger

logger = get_logger("live_runner")
_SUMMARY_COLUMNS = [config.TIMESTAMP_COLUMN, "is_alert", "emit_alert", config.LABEL_COLUMN, "episode_id",
                    "threat_type", "latency_ms"]


class LiveRunner:
    """Starts, stops and restarts simulated live runs (one at a time)."""

    def __init__(self, bundle: ModelBundle, state: DashboardState, base_seed: int | None = None) -> None:
        self._bundle, self._state = bundle, state
        self._base_seed = config.RANDOM_SEED + config.LIVE_SEED_OFFSET if base_seed is None else base_seed
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._runs = 0

    def start(self, hours: float, speed: float) -> None:
        """Begin a new run (stopping any run in progress). Each run uses a new seed."""
        with self._lock:
            self._halt()
            token = self._state.reset(hours, speed)
            seed = self._base_seed + self._runs
            self._runs += 1
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(token, hours, speed, seed, self._stop),
                                            name="live-detector", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the current run, if any."""
        with self._lock:
            self._halt()

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current run ends (used by tests)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _halt(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=15)

    def _run(self, token: int, hours: float, speed: float, seed: int, stop: threading.Event) -> None:
        try:
            events = event_simulator.simulate_events(hours, config.LIVE_START,
                                                     event_simulator.scaled_episodes(hours), seed)
            source = PacedEventSource(events, speed)
            detector = LiveDetector(self._bundle)
            self._state.begin(token, len(events))
            frames: list[pd.DataFrame] = []
            started = time.perf_counter()
            for batch in source.batches():
                if stop.is_set():
                    return
                results = detector.process_batch(batch)
                self._state.record_batch(token, results)
                frames.append(results[[c for c in _SUMMARY_COLUMNS if c in results.columns]])
            summary = summarize(pd.concat(frames, ignore_index=True), time.perf_counter() - started,
                                config.ALERT_COOLDOWN_SECONDS)
            self._state.finish(token, summary)
        except Exception as exc:  # surfaced in the UI instead of dying silently
            logger.exception("Live run failed")
            self._state.fail(token, f"{type(exc).__name__}: {exc}")
