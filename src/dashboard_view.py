"""Turns a dashboard snapshot into tables and numbers ready to display.

Pure functions (no Streamlit import) so they can be tested without a UI.
"""

from __future__ import annotations

import pandas as pd


def _fmt(value) -> str:
    return "-" if value is None else f"{int(round(value)):,}"


def elapsed_minutes(snapshot: dict) -> int:
    """Number of simulated minutes reached so far (all of them once finished)."""
    total = len(snapshot["series"]["events"])
    if snapshot["status"] == "finished":
        return total
    if not (snapshot["start_clock"] and snapshot["sim_clock"]):
        return 0
    seconds = (pd.Timestamp(snapshot["sim_clock"]) - pd.Timestamp(snapshot["start_clock"])).total_seconds()
    return int(min(total, seconds // 60 + 1))


def _series_frame(snapshot: dict, key: str, column: str) -> pd.DataFrame:
    values = snapshot["series"][key]
    shown = elapsed_minutes(snapshot)
    if not snapshot["start_clock"] or shown == 0:
        return pd.DataFrame({column: []})
    index = pd.date_range(snapshot["start_clock"], periods=len(values), freq="min")[:shown]
    return pd.DataFrame({column: values[:shown]}, index=index)


def events_frame(snapshot: dict) -> pd.DataFrame:
    """Events per simulated minute, up to the current simulated time."""
    return _series_frame(snapshot, "events", "Events per minute")


def alerts_per_minute_frame(snapshot: dict) -> pd.DataFrame:
    """Alerts per simulated minute, up to the current simulated time."""
    return _series_frame(snapshot, "alerts", "Alerts per minute")


def latency_frame(snapshot: dict) -> pd.DataFrame:
    """Mean and maximum scoring latency of the most recent batches."""
    batches = snapshot["batches"]
    return pd.DataFrame({"Mean (ms)": [b["mean_ms"] for b in batches],
                         "Max (ms)": [b["max_ms"] for b in batches]})


def counts_frame(counts: dict, column: str) -> pd.DataFrame:
    """A one-column frame (largest first) for bar charts of category counts."""
    series = pd.Series(counts, dtype=float).sort_values(ascending=False)
    return pd.DataFrame({column: series.astype(int)})


def alerts_table(snapshot: dict) -> pd.DataFrame:
    """Recent alerts, newest first, with readable column names."""
    columns = ["Time", "Severity", "Source", "Destination", "Threat type", "Score", "Action",
               "MITRE ATT&CK", "Latency (ms)", "Device", "Evidence"]
    rows = [{
        "Time": a["time"], "Severity": a["severity"], "Source": a["src_ip"],
        "Destination": f"{a['dst_ip']}:{a['dst_port']} {a['protocol']}",
        "Threat type": a["threat_type"], "Score": a["threat_score"], "Action": a["action"],
        "MITRE ATT&CK": a["mitre"], "Latency (ms)": a["latency_ms"], "Device": a.get("device", ""),
        "Evidence": a.get("evidence", ""),
    } for a in snapshot["alerts"]]
    return pd.DataFrame(rows, columns=columns)


def summary_table(snapshot: dict) -> pd.DataFrame | None:
    """Per attack class results of a finished run (None until the run ends)."""
    summary = snapshot.get("summary")
    if not summary or "detection_by_class" not in summary:
        return None
    rows = []
    for label, row in sorted(summary["detection_by_class"].items()):
        rows.append({
            "Attack class": label, "Caught": row["detected"], "Episodes": row["episodes"],
            "Median time to detect (s)": row["median_seconds_to_detect"],
            "First alert type correct (%)": None if row["first_alert_type_correct"] is None
            else round(100 * row["first_alert_type_correct"]),
        })
    return pd.DataFrame(rows)


def summary_headline(snapshot: dict) -> str:
    """One-line result for a finished run."""
    s = snapshot["summary"]
    return (f"Attack episodes caught: {s['episodes_detected']} / {s['episodes_total']} "
            f"({s['episode_detection_rate']:.0%}). False alerts on normal traffic: "
            f"{s['normal_event_false_alert_rate']:.2%} of normal events.")


def kpi_cards(snapshot: dict) -> list[dict]:
    """The headline numbers as ``{label, value, help}`` dictionaries."""
    lat, truth = snapshot["latency_ms"], snapshot["truth"]
    total = snapshot["total_events"]
    flagged = (f"{100 * truth['attack_flagged'] / truth['attack_events']:.0f}%"
               if truth["available"] and truth["attack_events"] else "-")
    false_rate = ("-" if truth["false_alerts_per_hour"] is None
                  else f"{truth['false_alerts_per_hour']:.1f}")
    return [
        {"label": "Events processed", "value": _fmt(snapshot["events"]),
         "help": f"of {total:,} in this run" if total else "waiting for the run to start"},
        {"label": "Alerts shown to analyst", "value": _fmt(snapshot["alerts_emitted"]),
         "help": f"{snapshot['alerts_raw']:,} raw alerts before 60-second duplicate suppression"},
        {"label": "Throughput (events/s)", "value": _fmt(snapshot["events_per_second"]),
         "help": "events scored per wall-clock second"},
        {"label": "Latency p95 (ms)", "value": _fmt(lat["p95"]),
         "help": "time from event arrival to scored; p50 %s, p99 %s" % (_fmt(lat["p50"]), _fmt(lat["p99"]))},
        {"label": "Attack events flagged", "value": flagged,
         "help": "simulator ground truth: share of attack events that raised an alert"},
        {"label": "False alerts per hour", "value": false_rate,
         "help": "simulator ground truth: alerts shown on normal traffic per simulated hour"},
    ]
