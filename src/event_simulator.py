"""Event-level traffic simulator.

Unlike ``synthetic_data`` (which samples finished flow records), this module
emits *raw events* - individual connections in time order - from normal
host "sessions" and attack "episodes". Features are then derived by the
sliding-window aggregator, exactly as in live operation.

Every event carries the ground-truth ``label`` and an ``episode_id`` (a session
or attack episode) used for leakage-free splitting and for measuring how quickly
an attack is detected.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import config
from src.stream_features import SlidingWindowAggregator
from src.synthetic_data import (IpPools, attach_context, build_ip_pools, sample_distribution,
                                weighted_choice)
from src.utils import get_logger

logger = get_logger("event_simulator")
_RAW_COLUMNS = ["offset_s", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "duration",
                "src_bytes", "dst_bytes", "src_packets", "dst_packets", "syn_ratio",
                "auth_failed", "label", "episode_id"]


def scaled_episodes(hours: float) -> dict[str, int]:
    """Attack episode counts for a run of ``hours`` (at least 2 per class)."""
    return {label: max(2, round(count * hours / 24.0))
            for label, count in config.EVENT_EPISODES_PER_DAY.items()}


def _event_block(offsets, src_ip, dst_ip, dst_port, protocol, spec: dict, rng: np.random.Generator,
                 label: str, episode_id: str, src_scale: float = 1.0, dst_scale: float = 1.0,
                 fail_prob: float = 0.0) -> dict:
    """Build the columns for one group of events from a scenario spec."""
    n = len(offsets)
    low, high = spec.get("bytes_per_packet", config.EVENT_NORMAL["bytes_per_packet"])
    src_bytes = np.round(sample_distribution(spec["src_bytes"], n, rng) * src_scale)
    dst_bytes = np.round(sample_distribution(spec["dst_bytes"], n, rng) * dst_scale)
    return {
        "offset_s": np.asarray(offsets, dtype=float),
        "src_ip": np.full(n, src_ip, dtype=object) if isinstance(src_ip, str) else np.asarray(src_ip, dtype=object),
        "src_port": rng.integers(1024, config.MAX_PORT + 1, n),
        "dst_ip": np.full(n, dst_ip, dtype=object) if isinstance(dst_ip, str) else np.asarray(dst_ip, dtype=object),
        "dst_port": np.asarray(dst_port, dtype=int),
        "protocol": np.asarray(protocol, dtype=object),
        "duration": sample_distribution(spec["flow_duration"], n, rng),
        "src_bytes": src_bytes, "dst_bytes": dst_bytes,
        "src_packets": np.maximum(1, np.round(src_bytes / rng.uniform(low, high, n))),
        "dst_packets": np.maximum(1, np.round(dst_bytes / rng.uniform(low, high, n))),
        "syn_ratio": sample_distribution(spec["syn_ratio"], n, rng),
        "auth_failed": (rng.random(n) < fail_prob).astype(int),
        "label": np.full(n, label, dtype=object),
        "episode_id": np.full(n, episode_id, dtype=object),
    }


def _protocols_for(ports: np.ndarray) -> np.ndarray:
    """UDP for DNS, TCP otherwise."""
    return np.where(np.asarray(ports) == 53, "UDP", "TCP").astype(object)


def _dns_block(start_s: float, span_s: float, src_ip: str, resolver: str, count: int,
               rng: np.random.Generator, label: str, episode_id: str) -> dict:
    """DNS lookups spread across an episode (tunnelling / C2 name resolution)."""
    offsets = np.sort(start_s + rng.uniform(0, span_s, count))
    return _event_block(offsets, src_ip, resolver, np.full(count, 53), np.full(count, "UDP", dtype=object),
                        config.EVENT_DNS, rng, label, episode_id)


def _session_block(host: str, start_s: float, scale: float, pools: IpPools, rng: np.random.Generator,
                   label: str, episode_id: str, size_mult: float = 1.0, host_mult: int = 1,
                   port_mult: int = 1, src_boost: float = 1.0) -> dict:
    """A burst of related connections from one host (normal use, or insider misuse)."""
    sim, spec = config.EVENT_SIM, config.EVENT_NORMAL
    n = 1 + rng.poisson(sim["session_size_lambda"] * size_mult)
    offsets = start_s + np.sort(rng.uniform(0, sim["session_spread_seconds"], n))
    k_hosts = (1 + rng.poisson(sim["session_hosts_lambda"])) * host_mult
    candidates = np.where(rng.random(k_hosts) < 0.8, rng.choice(pools.pools["external"], k_hosts),
                          rng.choice(pools.pools["server"], k_hosts))
    k_ports = (1 + rng.poisson(sim["session_ports_lambda"])) * port_mult
    port_keys = list(spec["ports"])
    port_set = rng.choice(port_keys, size=k_ports, p=np.array(list(spec["ports"].values())))
    dst = candidates[rng.integers(0, k_hosts, n)].astype(object)
    port = port_set[rng.integers(0, k_ports, n)].astype(int)
    is_dns = rng.random(n) < spec["dns_fraction"]
    dst = np.where(is_dns, pools.pools["server"][0], dst).astype(object)
    port = np.where(is_dns, 53, port)
    icmp = rng.random(n) < spec["icmp_fraction"]
    protocol = _protocols_for(port)
    protocol[icmp] = "ICMP"
    port = np.where(icmp, 0, port)
    block = _event_block(offsets, host, dst, port, protocol, spec, rng, label, episode_id,
                         src_scale=scale * src_boost, dst_scale=scale, fail_prob=spec["fail_prob"])
    return _shrink_dns_events(block, is_dns & ~icmp, rng)


def _shrink_dns_events(block: dict, is_dns: np.ndarray, rng: np.random.Generator) -> dict:
    """DNS lookups are tiny and quick, unlike web traffic (uses ``config.EVENT_DNS``)."""
    if not is_dns.any():
        return block
    dns = _event_block(block["offset_s"], "", "", block["dst_port"], block["protocol"], config.EVENT_DNS,
                       rng, "", "")
    for column in ("duration", "src_bytes", "dst_bytes", "src_packets", "dst_packets", "syn_ratio"):
        block[column] = np.where(is_dns, dns[column], block[column])
    return block


def _normal_sessions(pools: IpPools, hours: float, start_hour: int, rng: np.random.Generator) -> list[dict]:
    """Poisson-arriving sessions per host, concentrated in each host's own shift."""
    sim = config.EVENT_SIM
    blocks, counter = [], 0
    for host in pools.pools["host"]:
        host = str(host)
        active = set(config.HOUR_PROFILES[pools.persona_shift[host]]["peak_hours"])
        scale = pools.persona_scale[host]
        for h in range(int(math.ceil(hours))):
            rate = sim["session_size_lambda"] and sim["sessions_per_active_hour"] * scale * (
                1.0 if (start_hour + h) % 24 in active else sim["off_shift_activity"])
            for _ in range(rng.poisson(rate)):
                start = h * 3600 + rng.uniform(0, 3600)
                if start < hours * 3600:
                    counter += 1
                    blocks.append(_session_block(host, start, scale, pools, rng, config.NORMAL_LABEL, f"S{counter:06d}"))
    return blocks


def _episode_start(label: str, hours: float, start_hour: int, rng: np.random.Generator, span: float = 150.0) -> float:
    """Pick an episode start offset (seconds) honouring the class's typical hours."""
    profile_name = config.SYNTHETIC_PROFILES.get(label, {}).get("hours", "night")
    profile = config.HOUR_PROFILES.get(profile_name, config.HOUR_PROFILES["uniform"])
    hour_slots = list(range(int(math.ceil(hours))))
    weights = np.array([profile["peak_weight"] if (start_hour + h) % 24 in profile["peak_hours"] else 1.0
                        for h in hour_slots])
    slot = rng.choice(hour_slots, p=weights / weights.sum())
    return float(min(slot * 3600 + rng.uniform(0, 3600), max(hours * 3600 - span, 0.0)))


def _episode_blocks(label: str, index: int, pools: IpPools, hours: float, start_hour: int,
                    rng: np.random.Generator) -> list[dict]:
    """Events (plus supporting DNS) for one attack episode."""
    spec = config.EVENT_SCENARIOS[label]
    episode_id = f"A-{label.replace(' ', '').replace('-', '')}-{index:04d}"
    start = _episode_start(label, hours, start_hour, rng)
    servers, hosts = pools.pools["server"], pools.pools["host"]
    n_arr = lambda key: max(5, int(sample_distribution(spec[key], 1, rng)[0]))  # noqa: E731

    if label in ("Port Scan", "Brute-Force", "DDoS"):
        src, target = str(rng.choice(pools.pools["attacker"])), str(rng.choice(servers))
        n = n_arr("n_events")
        offsets = np.sort(start + rng.uniform(*spec["duration_s"][1:], 1) * rng.random(n))
        if label == "Port Scan":
            ports = rng.choice(np.arange(1, 1025), size=n, replace=n > 1024)
        else:
            ports = rng.choice(list(spec["ports"]), size=n, p=np.array(list(spec["ports"].values())))
        return [_event_block(offsets, src, target, ports, _protocols_for(ports), spec, rng, label, episode_id,
                             fail_prob=spec.get("fail_prob", 0.0))]

    src = str(rng.choice(pools.pools["compromised"]))
    if label == "Data Exfiltration":
        target, n = str(rng.choice(pools.pools["c2"])), n_arr("n_events")
        span = float(rng.uniform(*spec["duration_s"][1:]))
        offsets = np.sort(start + rng.uniform(0, span, n))
        ports = rng.choice(list(spec["ports"]), size=n, p=np.array(list(spec["ports"].values())))
        blocks = [_event_block(offsets, src, target, ports, _protocols_for(ports), spec, rng, label, episode_id)]
        blocks.append(_dns_block(start, span, src, servers[0], int(sample_distribution(spec["dns_events"], 1, rng)[0]), rng, label, episode_id))
        return blocks
    if label == "C2 Communication":
        target, n = str(rng.choice(pools.pools["c2"])), int(sample_distribution(spec["n_events"], 1, rng)[0])
        interval, jitter = (float(sample_distribution(spec[k], 1, rng)[0]) for k in ("interval_s", "jitter_s"))
        offsets = np.sort(start + np.arange(n) * interval + rng.normal(0, jitter, n))
        port = int(rng.choice(list(spec["ports"]), p=np.array(list(spec["ports"].values()))))
        blocks = [_event_block(offsets, src, target, np.full(n, port), _protocols_for(np.full(n, port)), spec, rng, label, episode_id)]
        blocks.append(_dns_block(start, n * interval, src, servers[0], int(sample_distribution(spec["dns_events"], 1, rng)[0]), rng, label, episode_id))
        return blocks
    # Lateral Movement and Novel Anomaly: one source touching many internal hosts.
    n = n_arr("n_events")
    targets = rng.choice(hosts, size=max(2, min(n, int(sample_distribution(spec["n_targets"], 1, rng)[0]))), replace=False)
    dst = targets[rng.integers(0, len(targets), n)].astype(object)
    offsets = np.sort(start + rng.uniform(0, float(rng.uniform(*spec["duration_s"][1:])), n))
    if label == "Novel Anomaly":
        ports, protocol = np.zeros(n, dtype=int), np.full(n, "ICMP", dtype=object)
        src = str(rng.choice(pools.pools["compromised"]))
    else:
        ports = rng.choice(list(spec["ports"]), size=n, p=np.array(list(spec["ports"].values())))
        protocol = _protocols_for(ports)
    return [_event_block(offsets, src, dst, ports, protocol, spec, rng, label, episode_id, fail_prob=spec.get("fail_prob", 0.0))]


def _insider_blocks(index: int, pools: IpPools, hours: float, start_hour: int, rng: np.random.Generator) -> list[dict]:
    """A normal-looking host hoarding data outside its own working shift."""
    start = float(rng.uniform(0, max(hours * 3600 - 120, 1.0)))
    clock = (start_hour + int(start // 3600)) % 24
    day_hours = set(config.HOUR_PROFILES["day_shift"]["peak_hours"])
    wanted = "night_shift" if clock in day_hours else "day_shift"
    candidates = [h for h in pools.pools["host"] if pools.persona_shift[str(h)] == wanted] or list(pools.pools["host"])
    host = str(rng.choice(candidates))
    cfg = config.EVENT_SIM["insider"]
    return [_session_block(host, start, pools.persona_scale[host], pools, rng, "Insider Misuse",
                           f"A-InsiderMisuse-{index:04d}", size_mult=cfg["size"], host_mult=cfg["hosts"],
                           port_mult=cfg["ports"], src_boost=cfg["src_bytes"])]


def simulate_events(hours: float, start: str, episodes: dict[str, int], seed: int) -> pd.DataFrame:
    """Simulate a time-ordered stream of labelled raw events (without windowed features)."""
    rng = np.random.default_rng(seed)
    pools = build_ip_pools(np.random.default_rng(config.RANDOM_SEED))  # same hosts/personas everywhere
    start_ts = pd.Timestamp(start)
    start_hour = start_ts.hour
    blocks = _normal_sessions(pools, hours, start_hour, rng)
    for label, count in episodes.items():
        for i in range(count):
            blocks += (_insider_blocks(i, pools, hours, start_hour, rng) if label == "Insider Misuse"
                       else _episode_blocks(label, i, pools, hours, start_hour, rng))
    raw = pd.DataFrame({col: np.concatenate([b[col] for b in blocks]) for col in _RAW_COLUMNS})
    raw = raw.sort_values("offset_s", kind="stable").reset_index(drop=True)
    raw["timestamp"] = start_ts + pd.to_timedelta(raw.pop("offset_s"), unit="s")
    raw.insert(0, "flow_id", [f"E{i:07d}" for i in range(len(raw))])
    raw = attach_context(raw, pools)
    logger.info("Simulated %d events (%d episodes): %s", len(raw), raw["episode_id"].nunique(),
                raw["label"].value_counts().to_dict())
    return raw


def with_window_features(events: pd.DataFrame) -> pd.DataFrame:
    """Append the sliding-window features to an event frame."""
    return pd.concat([events, SlidingWindowAggregator().process(events)], axis=1)


def _cap_rows(flows: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Uniform random subsample (keeps class proportions), restored to time order."""
    if len(flows) <= max_rows:
        return flows
    return flows.sample(max_rows, random_state=seed).sort_values("timestamp").reset_index(drop=True)


def build_training_flows(seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """One simulated day of events with live-computed features (training source)."""
    events = simulate_events(24, config.EVENT_SIM["start"], dict(config.EVENT_EPISODES_PER_DAY), seed)
    return _cap_rows(with_window_features(events), config.EVENT_TRAIN_MAX_ROWS, seed)


def build_novel_flows(seed: int = config.RANDOM_SEED + 1) -> pd.DataFrame:
    """Held-out novel-attack stream (ICMP tunnel) mixed with normal traffic."""
    events = simulate_events(4, config.EVENT_SIM["start"], {config.NOVEL_LABEL: 25}, seed)
    return _cap_rows(with_window_features(events), config.EVENT_NOVEL_MAX_ROWS, seed)
