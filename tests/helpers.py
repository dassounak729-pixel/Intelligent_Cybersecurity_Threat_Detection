"""Shared test helpers."""

import numpy as np
import pandas as pd

import config


def make_results(n=6, start="2026-01-06 08:00:00", step_s=30):
    """A minimal scored-batch frame with the columns the dashboard state reads."""
    times = pd.Timestamp(start) + pd.to_timedelta(np.arange(n) * step_s, unit="s")
    return pd.DataFrame({
        config.TIMESTAMP_COLUMN: times,
        "is_alert": [True, True, False, False, True, False][:n],
        "emit_alert": [True, False, False, False, True, False][:n],
        "threat_type": ["DDoS", "DDoS", "Normal", "Normal", "Port Scan", "Normal"][:n],
        "severity": ["High", "High", "Info", "Info", "Medium", "Info"][:n],
        "action": ["Block", "Block", "Allow", "Allow", "Investigate", "Allow"][:n],
        "latency_ms": [50.0, 60.0, 70.0, np.nan, 90.0, 100.0][:n],
        "label": ["DDoS", "DDoS", "Normal", "Normal", "Normal", "Port Scan"][:n],
        "src_ip": ["1.1.1.1"] * n, "dst_ip": ["10.0.0.1"] * n, "dst_port": [80] * n, "protocol": ["TCP"] * n,
        "device": ["EXTERNAL"] * n, "user": ["external"] * n, "evidence": ["x"] * n,
        "threat_score": [90.0, 88.0, 5.0, 5.0, 65.0, 4.0][:n],
        "mitre_tactic": ["Impact"] * n, "mitre_technique": ["T1498 Network Denial of Service"] * n,
    })


def make_upload_results(n=8, start="2026-01-06 08:00:00", step_s=60):
    """A minimal analysed-upload frame (output of ThreatDetector.analyze plus a label)."""
    times = pd.Timestamp(start) + pd.to_timedelta(np.arange(n) * step_s, unit="s")
    alert = np.array([True, True, False, False, True, False, False, True][:n])
    return pd.DataFrame({
        config.TIMESTAMP_COLUMN: times, "is_alert": alert,
        "threat_score": np.where(alert, [90.0, 75.0, 5.0, 8.0, 65.0, 3.0, 12.0, 82.0][:n], 5.0)[:n],
        "threat_type": np.where(alert, ["DDoS", "Port Scan", "Normal", "Normal", "DDoS", "Normal", "Normal", "C2 Communication"][:n], "Normal"),
        "severity": np.where(alert, ["Critical", "High", "Info", "Info", "Medium", "Info", "Info", "High"][:n], "Info"),
        "action": np.where(alert, ["Block", "Investigate", "Allow", "Allow", "Investigate", "Allow", "Allow", "Quarantine"][:n], "Allow"),
        "src_ip": ["1.1.1.1", "1.1.1.2", "10.0.0.5", "10.0.0.6", "1.1.1.1", "10.0.0.7", "10.0.0.8", "10.0.0.9"][:n],
        "dst_ip": ["10.0.0.1"] * n, "dst_port": [80] * n, "protocol": ["TCP"] * n,
        "mitre_tactic": ["Impact"] * n, "mitre_technique": ["T1498 Network Denial of Service"] * n,
        "evidence": ["conn_count_1m=90 (normal~5)"] * n, "user": ["external"] * n, "device": ["EXTERNAL"] * n,
        "confidence": [0.9] * n, config.LABEL_COLUMN: ["DDoS", "Port Scan", "Normal", "Normal", "Normal", "Normal", "Brute-Force", "C2 Communication"][:n],
    })
