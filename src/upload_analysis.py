"""Analysis of user-uploaded datasets (backend of the Streamlit "Analyze a dataset" page).

An uploaded CSV can be:

* **raw events** - one row per connection, without behavioural counters. The
  counters are computed here with the same sliding-window code used in live
  detection, so the model sees features built the way it was trained.
* **aggregated flows** - rows that already carry the six counter columns
  (``config.UPLOAD_COUNTER_COLUMNS``); these are used as they are.

Common column names from other tools (for example CICFlowMeter exports) are
recognised, and missing optional columns get documented defaults. Every
adjustment is reported back in ``notes`` so the user can see what happened.
"""

from __future__ import annotations

import io
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from src import event_simulator
from src.detection import ThreatDetector
from src.models import ModelBundle
from src.stream_features import SlidingWindowAggregator, epoch_seconds
from src.utils import get_logger

logger = get_logger("upload_analysis")

REQUIRED_UPLOAD_COLUMNS = ["timestamp", "src_ip", "dst_ip", "dst_port", "protocol", "src_bytes", "dst_bytes"]
OPTIONAL_UPLOAD_COLUMNS = ["duration", "src_packets", "dst_packets", "syn_ratio", "auth_failed",
                           "asset_criticality", "src_port", "user", "device", "flow_id", "label"]

# normalised header (lower case, letters and digits only) -> canonical column
_ALIASES = {
    "timestamp": "timestamp", "time": "timestamp", "datetime": "timestamp", "starttime": "timestamp",
    "srcip": "src_ip", "sourceip": "src_ip", "sourceaddress": "src_ip", "srcaddr": "src_ip", "srcaddress": "src_ip",
    "dstip": "dst_ip", "destinationip": "dst_ip", "destip": "dst_ip", "destinationaddress": "dst_ip",
    "dstaddr": "dst_ip", "dstaddress": "dst_ip",
    "srcport": "src_port", "sourceport": "src_port",
    "dstport": "dst_port", "destinationport": "dst_port", "destport": "dst_port",
    "protocol": "protocol", "proto": "protocol",
    "flowduration": "duration", "duration": "duration",
    "totalfwdpackets": "src_packets", "srcpackets": "src_packets", "fwdpackets": "src_packets",
    "totalbackwardpackets": "dst_packets", "dstpackets": "dst_packets", "bwdpackets": "dst_packets",
    "totallengthoffwdpackets": "src_bytes", "srcbytes": "src_bytes", "sentbytes": "src_bytes",
    "totallengthofbwdpackets": "dst_bytes", "dstbytes": "dst_bytes", "receivedbytes": "dst_bytes",
    "synratio": "syn_ratio", "authfailed": "auth_failed", "failedlogin": "auth_failed",
    "assetcriticality": "asset_criticality", "label": "label", "class": "label", "attackcat": "label",
    "user": "user", "device": "device", "flowid": "flow_id",
}


class UploadError(Exception):
    """Raised when an uploaded file cannot be analysed; the message is shown to the user."""


@dataclass
class PreparedDataset:
    """A validated dataset ready for the detector."""

    flows: pd.DataFrame
    mode: str                       # "events" (counters computed here) or "aggregated"
    notes: list[str] = field(default_factory=list)
    rows_dropped: int = 0


@dataclass
class AnalysisResult:
    """Everything the page needs to display an analysis."""

    results: pd.DataFrame
    summary: dict
    evaluation: dict | None
    notes: list[str]


# --------------------------------------------------------------------------- reading
def read_csv_bytes(data: bytes, max_rows: int = config.UPLOAD_MAX_ROWS) -> pd.DataFrame:
    """Parse uploaded CSV bytes, enforcing size and row limits."""
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > config.UPLOAD_MAX_MB * 1024 * 1024:
        raise UploadError(f"The file is larger than {config.UPLOAD_MAX_MB} MB. Please upload a smaller file.")
    try:
        frame = pd.read_csv(io.BytesIO(data), nrows=max_rows + 1, low_memory=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        raise UploadError(f"The file is not a readable CSV: {exc}") from exc
    if frame.empty:
        raise UploadError("The file has a header but no rows.")
    if len(frame) > max_rows:
        raise UploadError(f"The file has more than {max_rows:,} rows. Please upload a sample of at most "
                          f"{max_rows:,} rows.")
    return frame


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def normalise_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Rename known alternative headers to the project's column names."""
    notes: list[str] = []
    existing = set(frame.columns)
    rename: dict[str, str] = {}
    for column in frame.columns:
        target = _ALIASES.get(_normalise_name(column))
        if target and column != target and target not in existing and target not in rename.values():
            rename[column] = target
    if rename:
        notes.append("Recognised columns: " + ", ".join(f"'{str(old).strip()}' -> {new}" for old, new in rename.items()) + ".")
    renamed = frame.rename(columns=rename)
    microseconds = any(new == "duration" and _normalise_name(old) == "flowduration" for old, new in rename.items())
    if microseconds:
        renamed["duration"] = pd.to_numeric(renamed["duration"], errors="coerce") / 1e6
        notes.append("'Flow Duration' is in microseconds in CICFlowMeter exports; converted to seconds.")
    return renamed, notes


# ------------------------------------------------------------------------ preparing
def prepare_dataset(frame: pd.DataFrame) -> PreparedDataset:
    """Validate an uploaded frame, fill optional columns and compute counters if needed."""
    frame, notes = normalise_columns(frame)
    missing = [c for c in REQUIRED_UPLOAD_COLUMNS if c not in frame.columns]
    if missing:
        raise UploadError(
            f"Missing required column(s): {', '.join(missing)}. Required: {', '.join(REQUIRED_UPLOAD_COLUMNS)}. "
            f"Columns found: {', '.join(map(str, frame.columns[:25]))}"
            + (" ..." if len(frame.columns) > 25 else "") + ". Download the sample file to see the expected layout.")
    flows = frame.copy()
    start_rows = len(flows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # pandas warns when it must infer mixed date formats
        flows["timestamp"] = pd.to_datetime(flows["timestamp"], errors="coerce")
    flows = flows.dropna(subset=["timestamp", "src_ip", "dst_ip"])
    if len(flows) < 0.5 * start_rows:
        raise UploadError("More than half of the rows have an unreadable timestamp or missing IP address. "
                          "Timestamps should look like 2026-01-06 08:00:00.")
    for column in ("src_bytes", "dst_bytes", "dst_port"):
        flows[column] = pd.to_numeric(flows[column], errors="coerce").fillna(0)
    flows["protocol"] = _clean_protocol(flows["protocol"])

    for column, default in config.UPLOAD_DEFAULTS.items():
        if column not in flows.columns:
            flows[column] = default
            notes.append(f"No '{column}' column; using {default} for every row.")
    for column, source in (("src_packets", "src_bytes"), ("dst_packets", "dst_bytes")):
        if column not in flows.columns:
            flows[column] = np.maximum(1, np.round(flows[source] / 600.0))
            notes.append(f"No '{column}' column; estimated as bytes / 600.")
    if "flow_id" not in flows.columns:
        flows["flow_id"] = [f"U{i:07d}" for i in range(len(flows))]
    flows["flow_id"] = flows["flow_id"].astype(str)
    if "user" not in flows.columns:
        flows["user"] = np.where(flows["src_ip"].astype(str).map(_is_private), "user_" + flows["src_ip"].astype(str), "external")
    if "device" not in flows.columns:
        flows["device"] = flows["src_ip"].astype(str)
    flows = flows.sort_values("timestamp", kind="stable").reset_index(drop=True)

    has_counters = all(c in flows.columns for c in config.UPLOAD_COUNTER_COLUMNS)
    if has_counters:
        mode = "aggregated"
        notes.append("The file already contains the behavioural counter columns; they are used as they are.")
    else:
        mode = "events"
        counters = SlidingWindowAggregator().process(flows)
        flows = flows.drop(columns=[c for c in config.UPLOAD_COUNTER_COLUMNS if c in flows.columns])
        flows = pd.concat([flows, counters], axis=1)
        notes.append("Behavioural counters (connections, distinct ports/hosts, failed logins, DNS rate, beacon "
                     f"regularity) were computed over a {config.LIVE_WINDOW_SECONDS}-second sliding window per source IP.")
    return PreparedDataset(flows=flows, mode=mode, notes=notes, rows_dropped=start_rows - len(flows))


def _is_private(address: str) -> bool:
    import ipaddress
    try:
        return ipaddress.ip_address(address).is_private
    except ValueError:
        return False


def _clean_protocol(series: pd.Series) -> pd.Series:
    """Accept names (TCP) or IANA numbers (6); anything else becomes TCP."""
    numeric = pd.to_numeric(series, errors="coerce")
    names = series.astype(str).str.upper().str.strip()
    mapped = numeric.map(lambda v: config.PROTOCOL_NUMBERS.get(int(v)) if pd.notna(v) else None)
    cleaned = mapped.where(mapped.notna(), names)
    return cleaned.where(cleaned.isin(config.PROTOCOLS), "TCP")


# ------------------------------------------------------------------------ analysing
def _is_attack_label(labels: pd.Series) -> np.ndarray:
    return ~labels.astype(str).str.strip().str.lower().isin(config.UPLOAD_BENIGN_LABELS).to_numpy()


def evaluate_labels(results: pd.DataFrame, bundle_classes: list[str]) -> dict | None:
    """Compare alerts with a user-supplied ``label`` column (any non-benign label counts as an attack)."""
    if config.LABEL_COLUMN not in results.columns or results[config.LABEL_COLUMN].isna().all():
        return None
    attack = _is_attack_label(results[config.LABEL_COLUMN])
    flagged = results["is_alert"].to_numpy(dtype=bool)
    tp, fp = int((attack & flagged).sum()), int((~attack & flagged).sum())
    fn, tn = int((attack & ~flagged).sum()), int((~attack & ~flagged).sum())
    evaluation = {
        "attack_events": int(attack.sum()), "normal_events": int((~attack).sum()),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "by_label": {},
    }
    labels = results[config.LABEL_COLUMN].astype(str).str.strip()
    for label, group in results[attack].groupby(labels[attack]):
        evaluation["by_label"][label] = {"events": int(len(group)), "alerted": int(group["is_alert"].sum()),
                                          "detection_rate": float(group["is_alert"].mean())}
    evaluation["labels_match_model_classes"] = bool(set(labels.unique()) <= set(bundle_classes))
    return evaluation


def summarize_results(results: pd.DataFrame) -> dict:
    """Headline numbers for an analysed dataset."""
    alerts = results[results["is_alert"]]
    span = (results["timestamp"].max() - results["timestamp"].min()).total_seconds()
    return {
        "rows": int(len(results)), "alerts": int(len(alerts)),
        "alert_rate": float(len(alerts) / len(results)) if len(results) else 0.0,
        "max_score": float(results["threat_score"].max()) if len(results) else 0.0,
        "time_start": results["timestamp"].min().isoformat(), "time_end": results["timestamp"].max().isoformat(),
        "span_seconds": float(span), "sources_alerting": int(alerts["src_ip"].nunique()),
        "types": alerts["threat_type"].value_counts().to_dict(),
        "severities": alerts["severity"].value_counts().to_dict(),
        "actions": alerts["action"].value_counts().to_dict(),
        "alert_threshold": config.ALERT_SCORE_THRESHOLD,
    }


def analyze_dataset(bundle: ModelBundle, prepared: PreparedDataset) -> AnalysisResult:
    """Score every row of a prepared dataset and summarise the outcome."""
    results = ThreatDetector(bundle).analyze(prepared.flows)
    if results.empty:
        raise UploadError("No rows could be analysed after cleaning. Check the column formats.")
    notes = list(prepared.notes)
    if bundle.metadata.get("data_source") != "events":
        notes.append("This model was trained on pre-aggregated records; retrain with --source events for best results.")
    return AnalysisResult(results=results, summary=summarize_results(results),
                          evaluation=evaluate_labels(results, list(bundle.classes)), notes=notes)


def analyze_bytes(bundle: ModelBundle, data: bytes) -> AnalysisResult:
    """Convenience: raw uploaded bytes -> analysis (read, validate, prepare, score)."""
    return analyze_dataset(bundle, prepare_dataset(read_csv_bytes(data)))


# -------------------------------------------------------------------------- samples
def make_sample_events(hours: float = config.SAMPLE_HOURS, seed: int = config.RANDOM_SEED + 500) -> pd.DataFrame:
    """A small raw-event dataset (with labels) that the upload page accepts as-is."""
    events = event_simulator.simulate_events(hours, config.LIVE_START, event_simulator.scaled_episodes(hours), seed)
    keep = ["flow_id", "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "duration", "src_bytes",
            "dst_bytes", "src_packets", "dst_packets", "syn_ratio", "auth_failed", "asset_criticality", "user",
            "device", "label"]
    return events[keep]
