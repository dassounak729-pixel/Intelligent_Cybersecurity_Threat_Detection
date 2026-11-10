"""Central configuration for the Intelligent Cybersecurity Threat Detection system.

Every path, threshold, model parameter and synthetic-data profile used by the
project lives here, so behaviour can be tuned without touching business logic.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
ALERTS_DIR = OUTPUTS_DIR / "alerts"
LOGS_DIR = OUTPUTS_DIR / "logs"

RAW_FLOWS_FILE = RAW_DATA_DIR / "network_flows.csv"
NOVEL_FLOWS_FILE = RAW_DATA_DIR / "novel_flows.csv"
TRAIN_FILE = PROCESSED_DATA_DIR / "train_flows.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test_flows.csv"
MODEL_BUNDLE_FILE = MODELS_DIR / "threat_model_bundle.joblib"
METRICS_FILE = REPORTS_DIR / "metrics.json"
MODEL_COMPARISON_FILE = REPORTS_DIR / "model_comparison.csv"
ALERTS_FILE = ALERTS_DIR / "alerts.csv"
SAMPLE_ALERTS_FILE = ALERTS_DIR / "sample_alerts.txt"
LOG_FILE = LOGS_DIR / "threat_detection.log"
LOG_LEVEL = os.getenv("THREAT_DETECTION_LOG_LEVEL", "INFO")

# --------------------------------------------------------------------------- #
# Reproducibility & data split
# --------------------------------------------------------------------------- #
RANDOM_SEED = int(os.getenv("THREAT_DETECTION_SEED", "42"))
TEST_SIZE = 0.20
CV_FOLDS = 3
N_JOBS = -1

# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
NORMAL_LABEL = "Normal"
SUSPICIOUS_LABEL = "Suspicious Traffic"  # anomaly with no known attack signature
NOVEL_LABEL = "Novel Anomaly"            # held-out zero-day style traffic
ATTACK_LABELS = [
    "DDoS",
    "Brute-Force",
    "Port Scan",
    "Data Exfiltration",
    "C2 Communication",
    "Lateral Movement",
    "Insider Misuse",
]
ALL_LABELS = [NORMAL_LABEL] + ATTACK_LABELS

# MITRE ATT&CK alignment (tactic, technique id, technique name)
MITRE_MAPPING = {
    "DDoS": ("Impact", "T1498", "Network Denial of Service"),
    "Brute-Force": ("Credential Access", "T1110", "Brute Force"),
    "Port Scan": ("Discovery", "T1046", "Network Service Discovery"),
    "Data Exfiltration": ("Exfiltration", "T1048", "Exfiltration Over Alternative Protocol"),
    "C2 Communication": ("Command and Control", "T1071", "Application Layer Protocol"),
    "Lateral Movement": ("Lateral Movement", "T1021", "Remote Services"),
    "Insider Misuse": ("Collection", "T1213", "Data from Information Repositories"),
    SUSPICIOUS_LABEL: ("Unknown", "N/A", "Behavioural anomaly - unmapped"),
    NORMAL_LABEL: ("None", "N/A", "Benign traffic"),
}

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
REQUIRED_COLUMNS = [
    "timestamp", "src_ip", "dst_ip", "protocol", "dst_port", "duration",
    "src_bytes", "dst_bytes", "src_packets", "dst_packets", "failed_logins",
    "conn_count_1m", "unique_dst_ports_1m", "unique_dst_hosts_1m",
    "dns_query_rate", "syn_ratio", "beacon_interval_std", "asset_criticality",
]
LABEL_COLUMN = "label"
TIMESTAMP_COLUMN = "timestamp"
IDENTITY_COLUMNS = ["flow_id", "timestamp", "src_ip", "dst_ip", "src_port",
                    "dst_port", "protocol", "duration", "user", "device"]
NON_NEGATIVE_COLUMNS = [
    "duration", "src_bytes", "dst_bytes", "src_packets", "dst_packets",
    "failed_logins", "conn_count_1m", "unique_dst_ports_1m",
    "unique_dst_hosts_1m", "dns_query_rate", "syn_ratio", "beacon_interval_std",
]
PROTOCOLS = ["TCP", "UDP", "ICMP"]
MAX_PORT = 65535
BUSINESS_HOURS = (8, 19)  # [start, end)

# Features fed to the models (asset criticality is deliberately excluded - it
# is business context used only by the risk-scoring layer).
MODEL_FEATURES = [
    "duration", "dst_port", "src_bytes", "dst_bytes", "src_packets",
    "dst_packets", "failed_logins", "conn_count_1m", "unique_dst_ports_1m",
    "unique_dst_hosts_1m", "dns_query_rate", "syn_ratio", "beacon_interval_std",
    "hour_of_day", "is_off_hours", "src_is_internal", "dst_is_internal",
    "total_bytes", "bytes_ratio", "avg_packet_size", "packets_per_second",
    "protocol_TCP", "protocol_UDP", "protocol_ICMP",
]

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
MODEL_PARAMS = {
    "Logistic Regression": {"max_iter": 2000, "class_weight": "balanced"},
    "Decision Tree": {"max_depth": 14, "class_weight": "balanced", "min_samples_leaf": 3},
    "Random Forest": {"n_estimators": 250, "min_samples_leaf": 2,
                      "class_weight": "balanced_subsample"},
    "Gradient Boosting": {"max_iter": 200, "learning_rate": 0.1,
                          "class_weight": "balanced"},
}
MODEL_SELECTION_METRIC = "f1_macro"
ANOMALY_PARAMS = {"n_estimators": 300, "contamination": "auto", "max_samples": 512}
ANOMALY_CALIBRATION_LOW_PCT = 50.0    # benign percentile mapped to anomaly score 0
ANOMALY_CALIBRATION_HIGH_PCT = 99.5   # benign percentile mapped to anomaly score 1

# --------------------------------------------------------------------------- #
# Risk scoring, severity & response policy
# --------------------------------------------------------------------------- #
CLASSIFIER_WEIGHT = 0.65
ANOMALY_WEIGHT = 0.35
ANOMALY_ONLY_SCORE_CAP = 0.95        # score ceiling for anomaly-only detections
SUSPICIOUS_ANOMALY_THRESHOLD = 0.99  # anomaly score above which "Normal" is re-labelled (benign calibration ceiling)
CRITICALITY_MULTIPLIER = {1: 0.90, 2: 1.00, 3: 1.10, 4: 1.20}
ALERT_SCORE_THRESHOLD = 60.0         # threat score (0-100) at/above which an alert is raised.
# Chosen on a validation stream (RANDOM_SEED+100), confirmed on another (RANDOM_SEED+200):
# 50 -> ~35-40 false alerts/hour, 99-100% of attack episodes caught
# 60 -> ~27-34 false alerts/hour, 98-100% caught   <- chosen
# 80 -> ~26-31 false alerts/hour, but only 92-94% caught (insider misuse/lateral movement slip)
# Re-tune on your own data; the right value depends on the model and the traffic.

# (upper bound exclusive, severity, generic action). Last band uses the
# threat-type specific action from ACTION_POLICY.
SEVERITY_BANDS = [
    (25.0, "Info", "Allow"),
    (50.0, "Low", "Monitor"),
    (70.0, "Medium", "Investigate"),
    (90.0, "High", None),
    (101.0, "Critical", None),
]
ACTION_POLICY = {
    "DDoS": "Block",
    "Brute-Force": "Block",
    "Port Scan": "Investigate",
    "Data Exfiltration": "Quarantine",
    "C2 Communication": "Quarantine",
    "Lateral Movement": "Escalate",
    "Insider Misuse": "Escalate",  # human review (security + HR/compliance), never auto-block staff
    SUSPICIOUS_LABEL: "Investigate",
}

# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
EVIDENCE_TOP_K = 3
EVIDENCE_MIN_Z = 2.0
PERMUTATION_SAMPLE = 2000

# --------------------------------------------------------------------------- #
# Streaming simulation
# --------------------------------------------------------------------------- #
STREAM_BATCH_SIZE = 500

# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #
N_FLOWS = 30000
N_NOVEL_FLOWS = 600
STEALTH_FRACTION = 0.12   # share of attack flows that mimic normal behaviour
STEALTH_BLEND = 0.40      # share of behavioural features replaced by normal values
SIM_START = "2026-01-05 00:00:00"
SIM_DAYS = 7

POOL_SIZES = {"internal_hosts": 220, "servers": 30, "compromised": 25,
              "external_users": 300, "attackers": 40, "c2_infra": 15}
PUBLIC_FIRST_OCTETS = [34, 45, 52, 91, 104, 151, 185, 203]

BEHAVIOUR_COLUMNS = [
    "duration", "src_bytes", "dst_bytes", "src_packets", "dst_packets",
    "failed_logins", "conn_count_1m", "unique_dst_ports_1m",
    "unique_dst_hosts_1m", "dns_query_rate", "syn_ratio", "beacon_interval_std",
]
MIN_ONE_COLUMNS = ["conn_count_1m", "unique_dst_ports_1m", "unique_dst_hosts_1m"]

HOUR_PROFILES = {
    "business": {"peak_hours": list(range(8, 19)), "peak_weight": 6.0},
    "night": {"peak_hours": [0, 1, 2, 3, 4, 5, 22, 23], "peak_weight": 5.0},
    "uniform": {"peak_hours": [], "peak_weight": 1.0},
    # Individual working patterns used by per-host personas.
    "day_shift": {"peak_hours": list(range(8, 19)), "peak_weight": 30.0},
    "night_shift": {"peak_hours": [22, 23, 0, 1, 2, 3, 4, 5], "peak_weight": 30.0},
}

# Sampler specs: ("lognormal", mu, sigma) | ("poisson", lam) | ("beta", a, b)
#                | ("gamma", shape, scale) | ("uniform", lo, hi)
SYNTHETIC_PROFILES = {
    "Normal": {
        "duration": ("lognormal", 0.7, 1.0), "src_bytes": ("lognormal", 7.0, 1.3),
        "dst_bytes": ("lognormal", 8.3, 1.5), "bytes_per_packet": ("uniform", 150, 900),
        "failed_logins": ("poisson", 0.15), "conn_count_1m": ("poisson", 8),
        "unique_dst_ports_1m": ("poisson", 1.5), "unique_dst_hosts_1m": ("poisson", 2.0),
        "dns_query_rate": ("poisson", 6), "syn_ratio": ("beta", 2, 9),
        "beacon_interval_std": ("lognormal", 3.4, 0.7),
        "protocols": {"TCP": 0.74, "UDP": 0.23, "ICMP": 0.03},
        "dst_ports": {443: 0.42, 80: 0.18, 53: 0.20, 8080: 0.04, 22: 0.03, 25: 0.03,
                      3306: 0.03, 993: 0.03, 445: 0.02, 3389: 0.02},
        "hours": "shift", "persona": True, "src_kind": "host", "dst_kind": "mixed",
    },
    "DDoS": {
        "duration": ("lognormal", -0.5, 0.8), "src_bytes": ("lognormal", 6.0, 0.8),
        "dst_bytes": ("lognormal", 4.5, 1.0), "bytes_per_packet": ("uniform", 40, 200),
        "failed_logins": ("poisson", 0.1), "conn_count_1m": ("lognormal", 6.0, 0.9),
        "unique_dst_ports_1m": ("poisson", 2.0), "unique_dst_hosts_1m": ("poisson", 1.2),
        "dns_query_rate": ("poisson", 5), "syn_ratio": ("beta", 8, 2),
        "beacon_interval_std": ("lognormal", 3.0, 0.8),
        "protocols": {"TCP": 0.60, "UDP": 0.35, "ICMP": 0.05},
        "dst_ports": {80: 0.5, 443: 0.4, 53: 0.1},
        "hours": "uniform", "src_kind": "attacker", "dst_kind": "server",
    },
    "Brute-Force": {
        "duration": ("lognormal", 0.2, 0.7), "src_bytes": ("lognormal", 6.5, 0.6),
        "dst_bytes": ("lognormal", 6.0, 0.6), "bytes_per_packet": ("uniform", 60, 300),
        "failed_logins": ("poisson", 22), "conn_count_1m": ("poisson", 45),
        "unique_dst_ports_1m": ("poisson", 1.3), "unique_dst_hosts_1m": ("poisson", 1.5),
        "dns_query_rate": ("poisson", 5), "syn_ratio": ("beta", 3, 6),
        "beacon_interval_std": ("lognormal", 1.8, 0.6),
        "protocols": {"TCP": 1.0},
        "dst_ports": {22: 0.5, 3389: 0.3, 21: 0.1, 445: 0.1},
        "hours": "night", "src_kind": "attacker", "dst_kind": "server",
    },
    "Port Scan": {
        "duration": ("lognormal", -1.5, 0.6), "src_bytes": ("lognormal", 4.5, 0.5),
        "dst_bytes": ("lognormal", 3.0, 1.0), "bytes_per_packet": ("uniform", 40, 80),
        "failed_logins": ("poisson", 0.3), "conn_count_1m": ("poisson", 120),
        "unique_dst_ports_1m": ("poisson", 90), "unique_dst_hosts_1m": ("poisson", 6),
        "dns_query_rate": ("poisson", 5), "syn_ratio": ("beta", 9, 1.5),
        "beacon_interval_std": ("lognormal", 1.0, 0.6),
        "protocols": {"TCP": 0.90, "UDP": 0.10},
        "dst_ports": {21: 0.12, 22: 0.12, 23: 0.12, 80: 0.14, 443: 0.14, 445: 0.12,
                      3389: 0.12, 8080: 0.12},
        "hours": "uniform", "src_kind": "attacker", "dst_kind": "server",
    },
    "Data Exfiltration": {
        "duration": ("lognormal", 4.5, 0.8), "src_bytes": ("lognormal", 13.5, 1.0),
        "dst_bytes": ("lognormal", 7.0, 1.0), "bytes_per_packet": ("uniform", 600, 1400),
        "failed_logins": ("poisson", 0.1), "conn_count_1m": ("poisson", 6),
        "unique_dst_ports_1m": ("poisson", 1.2), "unique_dst_hosts_1m": ("poisson", 1.2),
        "dns_query_rate": ("poisson", 20), "syn_ratio": ("beta", 2, 9),
        "beacon_interval_std": ("lognormal", 3.0, 0.7),
        "protocols": {"TCP": 0.85, "UDP": 0.15},
        "dst_ports": {443: 0.5, 21: 0.1, 22: 0.15, 53: 0.15, 8080: 0.1},
        "hours": "night", "src_kind": "compromised", "dst_kind": "c2",
    },
    "C2 Communication": {
        "duration": ("lognormal", 0.0, 0.6), "src_bytes": ("lognormal", 5.8, 0.5),
        "dst_bytes": ("lognormal", 6.2, 0.6), "bytes_per_packet": ("uniform", 80, 300),
        "failed_logins": ("poisson", 0.1), "conn_count_1m": ("poisson", 14),
        "unique_dst_ports_1m": ("poisson", 1.1), "unique_dst_hosts_1m": ("poisson", 1.1),
        "dns_query_rate": ("poisson", 35), "syn_ratio": ("beta", 2, 9),
        "beacon_interval_std": ("gamma", 1.8, 0.5),
        "protocols": {"TCP": 0.80, "UDP": 0.20},
        "dst_ports": {443: 0.5, 8080: 0.2, 53: 0.2, 80: 0.1},
        "hours": "uniform", "src_kind": "compromised", "dst_kind": "c2",
    },
    "Lateral Movement": {
        "duration": ("lognormal", 1.8, 0.8), "src_bytes": ("lognormal", 8.0, 1.0),
        "dst_bytes": ("lognormal", 8.0, 1.0), "bytes_per_packet": ("uniform", 100, 700),
        "failed_logins": ("poisson", 2.5), "conn_count_1m": ("poisson", 18),
        "unique_dst_ports_1m": ("poisson", 3.0), "unique_dst_hosts_1m": ("poisson", 10),
        "dns_query_rate": ("poisson", 6), "syn_ratio": ("beta", 3, 7),
        "beacon_interval_std": ("lognormal", 2.6, 0.7),
        "protocols": {"TCP": 1.0},
        "dst_ports": {445: 0.35, 3389: 0.25, 135: 0.15, 5985: 0.10, 22: 0.15},
        "hours": "night", "src_kind": "compromised", "dst_kind": "host",
    },
    # Held-out traffic never shown to the classifier (zero-day style ICMP tunnel).
    "Novel Anomaly": {
        "duration": ("lognormal", 5.5, 0.5), "src_bytes": ("lognormal", 11.0, 0.6),
        "dst_bytes": ("lognormal", 11.0, 0.6), "bytes_per_packet": ("uniform", 900, 1500),
        "failed_logins": ("poisson", 0.05), "conn_count_1m": ("poisson", 30),
        "unique_dst_ports_1m": ("poisson", 1.0), "unique_dst_hosts_1m": ("poisson", 25),
        "dns_query_rate": ("poisson", 2), "syn_ratio": ("beta", 1, 20),
        "beacon_interval_std": ("lognormal", 1.2, 0.5),
        "protocols": {"ICMP": 0.8, "UDP": 0.2},
        "dst_ports": {53: 0.5, 443: 0.5},
        "hours": "night", "src_kind": "compromised", "dst_kind": "host",
    },
}

# Per-host "personas": each internal host gets its own activity scale and shift,
# so "normal" differs between hosts (this is what per-entity baselines learn).
HOST_PERSONA = {
    "scale_sigma": 0.5,               # lognormal spread of per-host activity level
    "night_shift_fraction": 0.15,     # share of hosts that work night shifts
    "scaled_columns": ["src_bytes", "dst_bytes", "src_packets", "dst_packets",
                       "conn_count_1m", "dns_query_rate"],
}
PERSONA_HOUR_MODES = ("shift", "off_shift")  # profile "hours" values handled per host

# Insider misuse: a normal-looking host that suddenly hoards data outside its own
# working hours. Values multiply the host's usual behaviour - still unremarkable
# for the bank as a whole, but unusual for that host.
INSIDER_BOOST = {"src_bytes": 8.0, "src_packets": 8.0, "conn_count_1m": 3.0,
                 "unique_dst_hosts_1m": 3.0, "unique_dst_ports_1m": 2.0}
SYNTHETIC_PROFILES["Insider Misuse"] = {
    **SYNTHETIC_PROFILES["Normal"], "hours": "off_shift", "boost": INSIDER_BOOST,
}
CLASS_PRIORS = {
    "Normal": 0.78, "DDoS": 0.04, "Brute-Force": 0.04, "Port Scan": 0.045,
    "Data Exfiltration": 0.025, "C2 Communication": 0.03, "Lateral Movement": 0.02,
    "Insider Misuse": 0.02,
}

# --------------------------------------------------------------------------- #
# Per-entity behavioural baselines (UEBA)
# --------------------------------------------------------------------------- #
USE_ENTITY_BASELINES = True       # set False to train without per-entity features
ENTITY_KEY = "src_ip"             # entity to profile (use "user" if you have real user IDs)
ENTITY_PROFILE_COLUMNS = ["src_bytes", "dst_bytes", "duration", "conn_count_1m",
                          "unique_dst_ports_1m", "unique_dst_hosts_1m",
                          "dns_query_rate", "failed_logins"]
ENTITY_FEATURES = ["entity_max_z", "entity_mean_z", "entity_hour_surprisal",
                   "entity_new_port", "entity_known"]
ENTITY_MIN_FLOWS = 20             # flows needed before an entity counts as "known"
ENTITY_SHRINKAGE = 10.0           # pulls sparse entities toward the global baseline
ENTITY_MIN_STD = 0.25             # floor on per-feature spread (log units)
ENTITY_Z_CAP = 12.0               # cap on deviation scores
ENTITY_CV_FOLDS = 5               # cross-fitting folds for training features

# --------------------------------------------------------------------------- #
# Live / streaming detection
# --------------------------------------------------------------------------- #
# "events"     = simulate raw events and derive features with the SAME sliding-window
#                code used live (no training/serving skew). Required for `live`.
# "aggregated" = original generator that samples pre-aggregated flow records.
DATA_SOURCE = "events"
LIVE_WINDOW_SECONDS = 60          # feature names ending in _1m assume 60 seconds
LIVE_AGG_KEY = "src_ip"           # entity whose recent activity is counted
LIVE_BEACON_HISTORY = 10          # timestamps kept per (src, dst, port) channel
LIVE_BEACON_DEFAULT = 30.0        # beacon std used when a channel has < 3 events
LIVE_BEACON_STALE_SECONDS = 600   # channel history older than this is discarded
LIVE_PURGE_EVERY = 20000          # events between memory clean-ups of idle entities
LIVE_MAX_BATCH = 2000             # max events scored per micro-batch
LIVE_POLL_SECONDS = 0.02          # how often the paced source checks for due events
LIVE_DEFAULT_SPEED = 240.0        # simulated seconds per wall-clock second
LIVE_RUN_HOURS = 2.0              # simulated duration of a `live` run
LIVE_START = "2026-01-06 08:00:00"
LIVE_SEED_OFFSET = 100            # live stream seed = RANDOM_SEED + this
ALERT_COOLDOWN_SECONDS = 60       # suppress repeat alerts for the same (source, threat type)
LIVE_ALERTS_FILE = ALERTS_DIR / "live_alerts.jsonl"
LIVE_SUMMARY_FILE = REPORTS_DIR / "live_summary.json"

# Event-level simulator (used when DATA_SOURCE == "events")
EVENT_TRAIN_MAX_ROWS = 80000      # uniform random subsample keeps class proportions
EVENT_NOVEL_MAX_ROWS = 20000
EVENT_SIM = {
    "start": SIM_START, "sessions_per_active_hour": 6.0, "off_shift_activity": 0.05,
    "session_size_lambda": 6, "session_spread_seconds": 50,
    "session_hosts_lambda": 1.5, "session_ports_lambda": 1.0,
    "insider": {"size": 3.0, "hosts": 3, "ports": 2, "src_bytes": 8.0},
}
EVENT_NORMAL = {
    "src_bytes": ("lognormal", 7.0, 1.3), "dst_bytes": ("lognormal", 8.3, 1.5),
    "flow_duration": ("lognormal", 0.7, 1.0), "syn_ratio": ("beta", 2, 9),
    "bytes_per_packet": (150, 900), "fail_prob": 0.03, "dns_fraction": 0.20,
    "icmp_fraction": 0.02,
    "ports": {443: 0.525, 80: 0.225, 8080: 0.05, 22: 0.0375, 25: 0.0375,
              3306: 0.0375, 993: 0.0375, 445: 0.025, 3389: 0.025},
}
EVENT_DNS = {"src_bytes": ("lognormal", 4.2, 0.3), "dst_bytes": ("lognormal", 5.0, 0.4),
             "flow_duration": ("lognormal", -2.0, 0.4), "syn_ratio": ("beta", 1, 20)}
EVENT_EPISODES_PER_DAY = {
    "Port Scan": 60, "Brute-Force": 60, "DDoS": 30, "Data Exfiltration": 50,
    "C2 Communication": 40, "Lateral Movement": 40, "Insider Misuse": 60,
}
EVENT_SCENARIOS = {
    "Port Scan": {"n_events": ("poisson", 150), "duration_s": ("uniform", 8, 45),
                  "src_bytes": ("lognormal", 4.5, 0.5), "dst_bytes": ("lognormal", 3.0, 1.0),
                  "flow_duration": ("lognormal", -1.5, 0.6), "syn_ratio": ("beta", 9, 1.5)},
    "Brute-Force": {"n_events": ("poisson", 50), "duration_s": ("uniform", 40, 110),
                    "src_bytes": ("lognormal", 6.5, 0.6), "dst_bytes": ("lognormal", 6.0, 0.6),
                    "flow_duration": ("lognormal", 0.2, 0.7), "syn_ratio": ("beta", 3, 6),
                    "fail_prob": 0.9, "ports": {22: 0.5, 3389: 0.3, 21: 0.1, 445: 0.1}},
    "DDoS": {"n_events": ("lognormal", 5.0, 0.7), "duration_s": ("uniform", 20, 55),
             "src_bytes": ("lognormal", 6.0, 0.8), "dst_bytes": ("lognormal", 4.5, 1.0),
             "flow_duration": ("lognormal", -0.5, 0.8), "syn_ratio": ("beta", 8, 2),
             "ports": {80: 0.5, 443: 0.4, 53: 0.1}},
    "Data Exfiltration": {"n_events": ("poisson", 6), "duration_s": ("uniform", 60, 240),
                          "src_bytes": ("lognormal", 13.5, 1.0), "dst_bytes": ("lognormal", 7.0, 1.0),
                          "flow_duration": ("lognormal", 4.5, 0.8), "syn_ratio": ("beta", 2, 9),
                          "dns_events": ("poisson", 20), "bytes_per_packet": (600, 1400),
                          "ports": {443: 0.5, 21: 0.1, 22: 0.15, 53: 0.15, 8080: 0.1}},
    "C2 Communication": {"n_events": ("uniform", 30, 60), "interval_s": ("uniform", 3, 10),
                         "jitter_s": ("uniform", 0.2, 0.9),
                         "src_bytes": ("lognormal", 5.8, 0.5), "dst_bytes": ("lognormal", 6.2, 0.6),
                         "flow_duration": ("lognormal", 0.0, 0.6), "syn_ratio": ("beta", 2, 9),
                         "dns_events": ("poisson", 60), "bytes_per_packet": (80, 300),
                         "ports": {443: 0.5, 8080: 0.2, 80: 0.3}},
    "Lateral Movement": {"n_events": ("poisson", 18), "duration_s": ("uniform", 30, 80),
                         "src_bytes": ("lognormal", 8.0, 1.0), "dst_bytes": ("lognormal", 8.0, 1.0),
                         "flow_duration": ("lognormal", 1.8, 0.8), "syn_ratio": ("beta", 3, 7),
                         "fail_prob": 0.15, "n_targets": ("poisson", 9),
                         "ports": {445: 0.35, 3389: 0.25, 135: 0.15, 5985: 0.10, 22: 0.15}},
    # Held-out ICMP tunnel (never used for training) for the zero-day test.
    "Novel Anomaly": {"n_events": ("poisson", 30), "duration_s": ("uniform", 60, 120),
                      "src_bytes": ("lognormal", 11.0, 0.6), "dst_bytes": ("lognormal", 11.0, 0.6),
                      "flow_duration": ("lognormal", 5.5, 0.5), "syn_ratio": ("beta", 1, 20),
                      "n_targets": ("poisson", 25), "bytes_per_packet": (900, 1500)},
}

# --------------------------------------------------------------------------- #
# Live dashboard
# --------------------------------------------------------------------------- #
DASHBOARD_HOST = "127.0.0.1"      # local machine only; do not expose without authentication
DASHBOARD_PORT = 8050
DASHBOARD_DEFAULT_HOURS = 4.0     # simulated duration per run
DASHBOARD_DEFAULT_SPEED = 120.0   # simulated seconds per wall-clock second (4h -> 2 min)
DASHBOARD_HOURS_RANGE = (0.5, 8.0)
DASHBOARD_SPEED_RANGE = (10.0, 2000.0)
DASHBOARD_MAX_ALERTS = 300        # alerts kept in memory
DASHBOARD_ALERTS_SHOWN = 60       # newest alerts sent to the browser
DASHBOARD_LATENCY_WINDOW = 5000   # recent events used for latency percentiles
DASHBOARD_BATCH_HISTORY = 150     # recent batches shown on the latency chart
DASHBOARD_REFRESH_MS = 1000       # browser polling interval
DASHBOARD_PAGE = PROJECT_ROOT / "src" / "dashboard" / "index.html"
STREAMLIT_APP = PROJECT_ROOT / "src" / "streamlit_app.py"
STREAMLIT_SPEEDS = [60, 120, 240, 480, 960]   # speed choices offered in the sidebar

# Dataset upload (Streamlit "Analyze a dataset" page)
UPLOAD_MAX_ROWS = 100_000         # rows analysed per uploaded file
UPLOAD_MAX_MB = 50                # also enforced by .streamlit/config.toml
SAMPLE_DIR = DATA_DIR / "sample"
SAMPLE_FILE = SAMPLE_DIR / "sample_events.csv"
SAMPLE_HOURS = 0.5                # simulated duration of the downloadable sample
UPLOAD_ALERT_TABLE_ROWS = 500     # rows shown in the alerts table (the CSV download has all)
UPLOAD_BENIGN_LABELS = {"normal", "benign", "0", "none", "legit", "legitimate", "clean"}
UPLOAD_DEFAULTS = {               # used when an optional column is missing
    "duration": 1.0, "syn_ratio": 0.1, "auth_failed": 0, "asset_criticality": 2, "src_port": 0,
}
UPLOAD_COUNTER_COLUMNS = ["conn_count_1m", "unique_dst_ports_1m", "unique_dst_hosts_1m",
                          "failed_logins", "dns_query_rate", "beacon_interval_std"]
PROTOCOL_NUMBERS = {1: "ICMP", 6: "TCP", 17: "UDP"}

# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
FIGURE_DPI = 130
COLOR_PRIMARY = "#1F4E79"
COLOR_ACCENT = "#C0392B"
COLOR_NEUTRAL = "#7F8C8D"
