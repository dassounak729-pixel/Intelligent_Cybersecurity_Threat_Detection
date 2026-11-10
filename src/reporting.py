"""Alert export and human-readable alert formatting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from src.utils import get_logger

logger = get_logger("reporting")

ALERT_COLUMNS = ["timestamp", "src_ip", "dst_ip", "dst_port", "protocol", "user", "device",
                 "threat_score", "threat_type", "confidence", "severity", "action",
                 "mitre_tactic", "mitre_technique", "evidence"]


def format_alert_text(alert: pd.Series) -> str:
    """Render one alert in the project-brief output format, plus extended context."""
    return "\n".join([
        f"IP: {alert['src_ip']}",
        f"Threat Score: {alert['threat_score']:.0f}%",
        f"Threat Type: {alert['threat_type']}",
        f"Action: {alert['action']}",
        f"  Destination: {alert['dst_ip']}:{int(alert['dst_port'])} ({alert['protocol']})",
        f"  Entity: {alert.get('user', 'n/a')} / {alert.get('device', 'n/a')}",
        f"  Severity: {alert['severity']}  Confidence: {alert['confidence']:.0%}",
        f"  MITRE ATT&CK: {alert['mitre_tactic']} - {alert['mitre_technique']}",
        f"  Evidence: {alert['evidence']}",
    ])


def top_alerts(results: pd.DataFrame, count: int = 5) -> pd.DataFrame:
    """Highest-scoring alerts, one per distinct threat type first for variety."""
    alerts = results[results["is_alert"]].sort_values("threat_score", ascending=False)
    per_type = alerts.groupby("threat_type", sort=False).head(1)
    remainder = alerts.drop(per_type.index).head(max(count - len(per_type), 0))
    return pd.concat([per_type, remainder]).sort_values("threat_score", ascending=False).head(count)


def write_alerts(results: pd.DataFrame, path: Path = config.ALERTS_FILE) -> Path:
    """Write alert rows (only flows that raised an alert) to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    alerts = results[results["is_alert"]].sort_values("threat_score", ascending=False)
    columns = [c for c in ALERT_COLUMNS if c in alerts.columns]
    alerts[columns].to_csv(path, index=False, float_format="%.4f")
    logger.info("Wrote %d alerts to %s", len(alerts), path)
    return path


def write_sample_alerts(results: pd.DataFrame, path: Path = config.SAMPLE_ALERTS_FILE, count: int = 6) -> str:
    """Write a text file of formatted example alerts and return the text."""
    blocks = [format_alert_text(row) for _, row in top_alerts(results, count).iterrows()]
    text = "\n\n".join(blocks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return text
