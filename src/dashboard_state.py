"""Thread-safe state behind the live dashboard.

The detector thread calls ``record_batch``; the web server thread calls
``snapshot``. Everything the browser needs is derived here, so the page only draws.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd

import config
from src.stream_features import epoch_seconds

_ALERT_FIELDS = ("src_ip", "dst_ip", "dst_port", "protocol", "threat_type", "severity", "action",
                 "device", "user", "evidence")


def _number(value, digits: int = 1):
    """Round a number for JSON; NaN/inf become None."""
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def json_safe(obj):
    """Recursively convert numpy types and NaN/inf into JSON-safe Python values."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


class DashboardState:
    """Accumulates live-run statistics and serves immutable snapshots."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token = 0
        self._clear(config.DASHBOARD_DEFAULT_HOURS, config.DASHBOARD_DEFAULT_SPEED)
        self._status = "idle"

    def _clear(self, hours: float, speed: float) -> None:
        self._status = "starting"
        self._message = ""
        self._hours, self._speed = hours, speed
        self._total_events = 0
        self._events = 0
        self._alerts_raw = 0
        self._alerts_emitted = 0
        self._start_epoch: float | None = None
        self._sim_epoch: float | None = None
        self._events_per_min: dict[int, int] = defaultdict(int)
        self._alerts_per_min: dict[int, int] = defaultdict(int)
        self._types: Counter = Counter()
        self._severity: Counter = Counter()
        self._actions: Counter = Counter()
        self._latencies: deque = deque(maxlen=config.DASHBOARD_LATENCY_WINDOW)
        self._batches: deque = deque(maxlen=config.DASHBOARD_BATCH_HISTORY)
        self._alerts: deque = deque(maxlen=config.DASHBOARD_MAX_ALERTS)
        self._truth = Counter()
        self._wall_start = self._wall_end = None
        self._summary: dict | None = None

    # ------------------------------------------------------------------ writes
    def reset(self, hours: float, speed: float) -> int:
        """Start a fresh run; returns a token the run must present on every write."""
        with self._lock:
            self._token += 1
            self._clear(hours, speed)
            return self._token

    def begin(self, token: int, total_events: int) -> None:
        with self._lock:
            if token == self._token:
                self._total_events, self._status = int(total_events), "running"
                self._wall_start = time.perf_counter()

    def finish(self, token: int, summary: dict) -> None:
        with self._lock:
            if token == self._token:
                self._status, self._summary, self._wall_end = "finished", summary, time.perf_counter()

    def fail(self, token: int, message: str) -> None:
        with self._lock:
            if token == self._token:
                self._status, self._message, self._wall_end = "error", message, time.perf_counter()

    def record_batch(self, token: int, results: pd.DataFrame) -> None:
        """Fold one scored micro-batch into the statistics."""
        if results.empty:
            return
        epoch = epoch_seconds(results[config.TIMESTAMP_COLUMN])
        emit = results["emit_alert"].to_numpy(dtype=bool)
        raw = results["is_alert"].to_numpy(dtype=bool)
        with self._lock:
            if token != self._token:
                return
            if self._start_epoch is None:
                self._start_epoch = float(epoch.min())
            self._sim_epoch = max(self._sim_epoch or 0.0, float(epoch.max()))
            minutes = np.maximum(((epoch - self._start_epoch) // 60).astype(int), 0)
            self._events += len(results)
            self._alerts_raw += int(raw.sum())
            self._alerts_emitted += int(emit.sum())
            for minute, count in zip(*np.unique(minutes, return_counts=True)):
                self._events_per_min[int(minute)] += int(count)
            for minute, count in zip(*np.unique(minutes[emit], return_counts=True)):
                self._alerts_per_min[int(minute)] += int(count)
            emitted = results[emit]
            self._types.update(emitted["threat_type"].tolist())
            self._severity.update(emitted["severity"].tolist())
            self._actions.update(emitted["action"].tolist())
            if "latency_ms" in results.columns:
                latency = results["latency_ms"].astype(float)
                latency = latency[np.isfinite(latency)]  # one missing value must not poison the percentiles
                if len(latency):
                    self._latencies.extend(latency.tolist())
                    self._batches.append((len(results), float(latency.mean()), float(latency.max())))
            if config.LABEL_COLUMN in results.columns:
                normal = (results[config.LABEL_COLUMN] == config.NORMAL_LABEL).to_numpy()
                self._truth["normal_events"] += int(normal.sum())
                self._truth["normal_alerts"] += int((emit & normal).sum())
                self._truth["attack_events"] += int((~normal).sum())
                self._truth["attack_flagged"] += int((raw & ~normal).sum())
            for _, row in emitted.iterrows():
                self._alerts.appendleft(self._alert_dict(row))

    @staticmethod
    def _alert_dict(row: pd.Series) -> dict:
        alert = {name: row[name] for name in _ALERT_FIELDS if name in row.index}
        alert["time"] = pd.Timestamp(row[config.TIMESTAMP_COLUMN]).strftime("%H:%M:%S")
        alert["threat_score"] = _number(row["threat_score"])
        alert["latency_ms"] = _number(row["latency_ms"], 0) if "latency_ms" in row.index else None
        alert["mitre"] = f"{row['mitre_tactic']} / {row['mitre_technique']}"
        return alert

    # ------------------------------------------------------------------- reads
    def snapshot(self) -> dict:
        """A JSON-safe picture of the current run."""
        with self._lock:
            elapsed_sim = (self._sim_epoch - self._start_epoch) if self._start_epoch is not None else 0.0
            wall = ((self._wall_end or time.perf_counter()) - self._wall_start) if self._wall_start else 0.0
            total_minutes = int(math.ceil(self._hours * 60))
            latencies = np.array(self._latencies, dtype=float)
            percentile = (lambda q: _number(np.percentile(latencies, q), 0)) if len(latencies) else (lambda q: None)
            normal, attack = self._truth["normal_events"], self._truth["attack_events"]
            hours_elapsed = max(elapsed_sim / 3600.0, 1e-9)
            start_clock = (pd.Timestamp(self._start_epoch, unit="s").strftime("%Y-%m-%dT%H:%M:%S")
                           if self._start_epoch is not None else None)
            return json_safe({
                "status": self._status, "message": self._message,
                "hours": self._hours, "speed": self._speed,
                "progress": (self._events / self._total_events) if self._total_events else 0.0,
                "events": self._events, "total_events": self._total_events,
                "alerts_raw": self._alerts_raw, "alerts_emitted": self._alerts_emitted,
                "events_per_second": (self._events / wall) if wall > 0 else 0.0,
                "latency_ms": {"p50": percentile(50), "p95": percentile(95), "p99": percentile(99),
                               "max": _number(latencies.max(), 0) if len(latencies) else None},
                "start_clock": start_clock,
                "sim_clock": (pd.Timestamp(self._sim_epoch, unit="s").strftime("%Y-%m-%dT%H:%M:%S")
                              if self._sim_epoch is not None else None),
                "total_minutes": total_minutes,
                "series": {"events": [self._events_per_min.get(m, 0) for m in range(total_minutes)],
                           "alerts": [self._alerts_per_min.get(m, 0) for m in range(total_minutes)]},
                "types": dict(self._types.most_common()),
                "severities": dict(self._severity),
                "actions": dict(self._actions.most_common()),
                "batches": [{"size": s, "mean_ms": _number(m, 0), "max_ms": _number(x, 0)} for s, m, x in self._batches],
                "alerts": list(self._alerts)[: config.DASHBOARD_ALERTS_SHOWN],
                "truth": {
                    "available": bool(normal + attack),
                    "attack_events": attack, "attack_flagged": self._truth["attack_flagged"],
                    "normal_events": normal, "false_alerts_emitted": self._truth["normal_alerts"],
                    "false_alerts_per_hour": self._truth["normal_alerts"] / hours_elapsed if normal else None,
                },
                "summary": self._summary,
            })
