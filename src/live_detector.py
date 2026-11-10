"""Live detector: raw events in, de-duplicated scored alerts out, with latency stats."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

import config
from src.detection import ThreatDetector
from src.models import ModelBundle
from src.stream_features import SlidingWindowAggregator, epoch_seconds
from src.utils import get_logger

logger = get_logger("live_detector")
AlertCallback = Callable[[pd.Series], None]


def apply_cooldown(results: pd.DataFrame, last_alert: dict, cooldown_seconds: float) -> np.ndarray:
    """Return a boolean mask of alerts to emit, suppressing repeats.

    An alert for a (source, threat type) pair is suppressed if one was already
    emitted for that pair within ``cooldown_seconds`` of event time. ``last_alert``
    is updated in place so suppression carries across batches.
    """
    emit = np.zeros(len(results), dtype=bool)
    times = epoch_seconds(results[config.TIMESTAMP_COLUMN]) if len(results) else np.array([])
    for position in np.flatnonzero(results["is_alert"].to_numpy()):
        key = (results["src_ip"].iat[position], results["threat_type"].iat[position])
        previous = last_alert.get(key)
        if previous is None or times[position] - previous >= cooldown_seconds:
            emit[position] = True
            last_alert[key] = times[position]
    return emit


class LiveDetector:
    """Stateful micro-batch detector (sliding windows + scoring + alert suppression)."""

    def __init__(self, bundle: ModelBundle, on_alert: AlertCallback | None = None,
                 cooldown_seconds: float = config.ALERT_COOLDOWN_SECONDS) -> None:
        self.detector = ThreatDetector(bundle)
        self.aggregator = SlidingWindowAggregator()
        self.on_alert = on_alert
        self.cooldown_seconds = cooldown_seconds
        self._last_alert: dict = {}

    def process_batch(self, events: pd.DataFrame) -> pd.DataFrame:
        """Score one micro-batch and return per-event results (including latency)."""
        events = events.sort_values(config.TIMESTAMP_COLUMN, kind="stable").reset_index(drop=True)
        flows = pd.concat([events, self.aggregator.process(events)], axis=1)
        results = self.detector.analyze(flows)
        finished = time.perf_counter()
        extra = [c for c in ("emitted_at", "episode_id") if c in events.columns]
        results = results.merge(events[["flow_id"] + extra], on="flow_id", how="left")
        if "emitted_at" in results.columns:
            results["latency_ms"] = (finished - results["emitted_at"]) * 1000.0
        results["emit_alert"] = apply_cooldown(results, self._last_alert, self.cooldown_seconds)
        if self.on_alert is not None:
            for _, alert in results[results["emit_alert"]].iterrows():
                self.on_alert(alert)
        return results


def _percentiles(values: pd.Series) -> dict:
    return {"p50": float(values.quantile(0.50)), "p95": float(values.quantile(0.95)),
            "p99": float(values.quantile(0.99)), "max": float(values.max())}


def summarize(results: pd.DataFrame, wall_seconds: float, cooldown_seconds: float) -> dict:
    """Run-level statistics: throughput, latency, alert volume, detection quality."""
    span = (results[config.TIMESTAMP_COLUMN].max() - results[config.TIMESTAMP_COLUMN].min()).total_seconds()
    summary: dict = {
        "events": int(len(results)), "wall_seconds": float(wall_seconds),
        "simulated_seconds": float(span),
        "events_per_second": float(len(results) / wall_seconds) if wall_seconds else 0.0,
        "alerts_raw": int(results["is_alert"].sum()), "alerts_emitted": int(results["emit_alert"].sum()),
        "alert_cooldown_seconds": cooldown_seconds,
    }
    if "latency_ms" in results.columns:
        summary["latency_ms_all_events"] = _percentiles(results["latency_ms"])
        emitted = results[results["emit_alert"]]
        if len(emitted):
            summary["latency_ms_emitted_alerts"] = _percentiles(emitted["latency_ms"])
    if config.LABEL_COLUMN in results.columns and "episode_id" in results.columns:
        summary.update(_ground_truth_summary(results, span))
    return summary


def _ground_truth_summary(results: pd.DataFrame, span_seconds: float) -> dict:
    """Episode-level detection, time-to-detect and false-alert statistics."""
    attacks = results[~results[config.LABEL_COLUMN].isin([config.NORMAL_LABEL])]
    rows = []
    for episode, group in attacks.groupby("episode_id"):
        alerted = group[group["is_alert"]].sort_values(config.TIMESTAMP_COLUMN)
        first = alerted.iloc[0] if len(alerted) else None
        rows.append({
            "label": group[config.LABEL_COLUMN].iat[0], "detected": first is not None,
            "seconds_to_detect": (first[config.TIMESTAMP_COLUMN] - group[config.TIMESTAMP_COLUMN].min()).total_seconds()
            if first is not None else np.nan,
            "type_match": bool(first is not None and first["threat_type"] == group[config.LABEL_COLUMN].iat[0]),
        })
    episodes = pd.DataFrame(rows)
    by_class = {}
    for label, group in episodes.groupby("label"):
        found = group[group["detected"]]
        by_class[label] = {
            "episodes": int(len(group)), "detected": int(len(found)),
            "detection_rate": float(len(found) / len(group)),
            "median_seconds_to_detect": float(found["seconds_to_detect"].median()) if len(found) else None,
            "first_alert_type_correct": float(found["type_match"].mean()) if len(found) else None,
        }
    normal = results[results[config.LABEL_COLUMN] == config.NORMAL_LABEL]
    hours = max(span_seconds / 3600.0, 1e-9)
    return {
        "episodes_total": int(len(episodes)), "episodes_detected": int(episodes["detected"].sum()),
        "episode_detection_rate": float(episodes["detected"].mean()) if len(episodes) else None,
        "detection_by_class": by_class,
        "normal_event_false_alert_rate": float(normal["is_alert"].mean()) if len(normal) else None,
        "false_alerts_emitted_per_hour": float((normal["emit_alert"]).sum() / hours) if len(normal) else None,
    }


def format_summary(summary: dict) -> str:
    """Human-readable run summary."""
    lines = [
        "=" * 64, "LIVE RUN SUMMARY", "=" * 64,
        f"Events processed        : {summary['events']:,} in {summary['wall_seconds']:.1f}s "
        f"({summary['events_per_second']:,.0f} events/s; {summary['simulated_seconds'] / 60:.0f} simulated minutes)",
        f"Alerts                  : {summary['alerts_emitted']:,} emitted "
        f"({summary['alerts_raw']:,} raw before {summary['alert_cooldown_seconds']:.0f}s duplicate suppression)",
    ]
    if "latency_ms_all_events" in summary:
        lat = summary["latency_ms_all_events"]
        lines.append(f"Latency, event->scored  : p50 {lat['p50']:.0f} ms | p95 {lat['p95']:.0f} ms | "
                     f"p99 {lat['p99']:.0f} ms | max {lat['max']:.0f} ms")
    if "episodes_total" in summary:
        lines.append(f"Attack episodes caught  : {summary['episodes_detected']}/{summary['episodes_total']} "
                     f"({summary['episode_detection_rate']:.0%})")
        lines.append(f"False alerts on normal  : {summary['normal_event_false_alert_rate']:.2%} of normal events; "
                     f"{summary['false_alerts_emitted_per_hour']:.1f} emitted per simulated hour")
        lines.append("")
        lines.append(f"{'Attack class':20s} {'caught':>9s} {'median time-to-detect':>23s} {'type right':>11s}")
        for label, row in sorted(summary["detection_by_class"].items()):
            ttd = "-" if row["median_seconds_to_detect"] is None else f"{row['median_seconds_to_detect']:.0f} s"
            ok = "-" if row["first_alert_type_correct"] is None else f"{row['first_alert_type_correct']:.0%}"
            lines.append(f"{label:20s} {row['detected']:>4d}/{row['episodes']:<4d} {ttd:>23s} {ok:>11s}")
    return "\n".join(lines)
