"""Turns an upload analysis into tables and numbers ready to display (no UI imports)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.dashboard_view import counts_frame  # noqa: F401  (re-exported for the page)


def kpi_cards(summary: dict) -> list[dict]:
    """Headline cards for an analysed dataset."""
    top_type = next(iter(summary["types"]), "-")
    return [
        {"label": "Events analysed", "value": f"{summary['rows']:,}", "help": "rows scored by the detector"},
        {"label": "Alerts raised", "value": f"{summary['alerts']:,}",
         "help": f"events with a threat score of {summary['alert_threshold']:.0f} or more"},
        {"label": "Alert rate", "value": f"{summary['alert_rate']:.1%}", "help": "alerts as a share of all events"},
        {"label": "Sources alerting", "value": f"{summary['sources_alerting']:,}", "help": "distinct source IPs with at least one alert"},
        {"label": "Highest threat score", "value": f"{summary['max_score']:.0f}", "help": "out of 100"},
        {"label": "Most common threat", "value": top_type, "help": "among the alerts"},
    ]


def time_series(results: pd.DataFrame, buckets: int = 60) -> pd.DataFrame:
    """Events and alerts per time bucket (bucket size is chosen from the data span)."""
    start, end = results["timestamp"].min(), results["timestamp"].max()
    span = max((end - start).total_seconds(), 1.0)
    seconds = max(int(np.ceil(span / buckets / 60.0)) * 60, 60)
    index = ((results["timestamp"] - start).dt.total_seconds() // seconds).astype(int)
    grouped = pd.DataFrame({"bucket": index, "alert": results["is_alert"].astype(int)}).groupby("bucket")["alert"].agg(["size", "sum"])
    grouped = grouped.reindex(range(int(index.max()) + 1), fill_value=0)
    grouped.index = start + pd.to_timedelta(grouped.index * seconds, unit="s")
    return grouped.rename(columns={"size": "Events", "sum": "Alerts"})


def score_histogram(results: pd.DataFrame, bin_width: int = 10) -> pd.DataFrame:
    """Number of events per threat-score band (0-10, 10-20, ...)."""
    edges = list(range(0, 101, bin_width))
    bands = pd.cut(results["threat_score"], bins=edges, right=False, include_lowest=True,
                   labels=[f"{a}-{a + bin_width}" for a in edges[:-1]])
    return pd.DataFrame({"Events": bands.value_counts().sort_index()})


def alerts_table(results: pd.DataFrame, severities: list[str] | None = None, min_score: float | None = None,
                 threat_types: list[str] | None = None, limit: int = config.UPLOAD_ALERT_TABLE_ROWS) -> pd.DataFrame:
    """Alerts, highest score first, optionally filtered, with readable column names."""
    alerts = results[results["is_alert"]]
    if severities:
        alerts = alerts[alerts["severity"].isin(severities)]
    if threat_types:
        alerts = alerts[alerts["threat_type"].isin(threat_types)]
    if min_score is not None:
        alerts = alerts[alerts["threat_score"] >= min_score]
    alerts = alerts.sort_values(["threat_score", "timestamp"], ascending=[False, True]).head(limit)
    table = pd.DataFrame({
        "Time": alerts["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Severity": alerts["severity"], "Source": alerts["src_ip"],
        "Destination": alerts["dst_ip"].astype(str) + ":" + alerts["dst_port"].astype(int).astype(str) + " " + alerts["protocol"],
        "Threat type": alerts["threat_type"], "Score": alerts["threat_score"].round(1), "Action": alerts["action"],
        "MITRE ATT&CK": alerts["mitre_tactic"] + " / " + alerts["mitre_technique"], "Evidence": alerts["evidence"],
    })
    return table.reset_index(drop=True)


def export_csv(results: pd.DataFrame) -> bytes:
    """All alerts as CSV bytes for the download button."""
    columns = ["timestamp", "src_ip", "dst_ip", "dst_port", "protocol", "user", "device", "threat_score",
               "threat_type", "confidence", "severity", "action", "mitre_tactic", "mitre_technique", "evidence"]
    alerts = results[results["is_alert"]].sort_values("threat_score", ascending=False)
    return alerts[[c for c in columns if c in alerts.columns]].to_csv(index=False, float_format="%.4f").encode("utf-8")


def evaluation_cards(evaluation: dict) -> list[dict]:
    """Headline numbers when the upload contained a label column."""
    pct = lambda v: "-" if v is None else f"{v:.1%}"  # noqa: E731
    return [
        {"label": "Precision", "value": pct(evaluation["precision"]), "help": "share of alerts that were real attacks"},
        {"label": "Recall", "value": pct(evaluation["recall"]), "help": "share of attack events that raised an alert"},
        {"label": "False-positive rate", "value": pct(evaluation["false_positive_rate"]), "help": "share of normal events that raised an alert"},
        {"label": "Attack events", "value": f"{evaluation['attack_events']:,}", "help": "rows with a non-benign label"},
    ]


def evaluation_table(evaluation: dict) -> pd.DataFrame:
    """Detection rate for each attack label found in the file."""
    rows = [{"Label": label, "Events": row["events"], "Alerted": row["alerted"],
             "Detection rate (%)": round(100 * row["detection_rate"], 1)}
            for label, row in sorted(evaluation["by_label"].items())]
    return pd.DataFrame(rows, columns=["Label", "Events", "Alerted", "Detection rate (%)"])
