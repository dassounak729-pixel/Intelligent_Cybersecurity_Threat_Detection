"""Synthetic network-flow generator.

Real enterprise telemetry cannot be shipped with a project, so this module
produces flow records (NetFlow/IPFIX-style plus auth/DNS context) whose
statistical profiles are defined entirely in ``config.SYNTHETIC_PROFILES``.
A share of attack flows is made "stealthy" (partially blended with normal
behaviour) so the problem is not trivially separable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from src.utils import get_logger

logger = get_logger("synthetic_data")


@dataclass
class IpPools:
    """Pools of IP addresses by role, plus lookups for device name/criticality."""

    pools: dict[str, np.ndarray]
    criticality: dict[str, int]
    device_names: dict[str, str]
    persona_scale: dict[str, float] = field(default_factory=dict)
    persona_shift: dict[str, str] = field(default_factory=dict)


def sample_distribution(spec: tuple, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``size`` values from a distribution described by a config tuple."""
    kind, *params = spec
    if kind == "lognormal":
        return rng.lognormal(params[0], params[1], size)
    if kind == "poisson":
        return rng.poisson(params[0], size).astype(float)
    if kind == "beta":
        return rng.beta(params[0], params[1], size)
    if kind == "gamma":
        return rng.gamma(params[0], params[1], size)
    if kind == "uniform":
        return rng.uniform(params[0], params[1], size)
    raise ValueError(f"Unsupported distribution kind: {kind}")


def weighted_choice(weights: dict, size: int, rng: np.random.Generator) -> np.ndarray:
    """Weighted categorical draw from a ``{value: probability}`` mapping."""
    keys = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=float)
    return rng.choice(keys, size=size, p=probs / probs.sum())


def _public_ips(count: int, rng: np.random.Generator) -> np.ndarray:
    """Generate ``count`` unique public-looking IPv4 addresses."""
    ips: set[str] = set()
    while len(ips) < count:
        ips.add(f"{rng.choice(config.PUBLIC_FIRST_OCTETS)}.{rng.integers(1, 255)}."
                f"{rng.integers(1, 255)}.{rng.integers(2, 254)}")
    return np.array(sorted(ips))


def build_ip_pools(rng: np.random.Generator) -> IpPools:
    """Create IP pools for hosts, servers, compromised hosts and external actors."""
    sizes = config.POOL_SIZES
    hosts = np.array(sorted({f"192.168.{rng.integers(10, 30)}.{rng.integers(2, 251)}"
                             for _ in range(sizes["internal_hosts"] * 3)}))[: sizes["internal_hosts"]]
    servers = np.array([f"10.0.0.{i}" for i in range(10, 10 + sizes["servers"])])
    compromised = rng.choice(hosts, size=sizes["compromised"], replace=False)
    pools = {
        "host": hosts, "server": servers, "compromised": compromised,
        "external": _public_ips(sizes["external_users"], rng),
        "attacker": _public_ips(sizes["attackers"], rng),
        "c2": _public_ips(sizes["c2_infra"], rng),
    }
    criticality = {ip: int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])) for ip in hosts}
    criticality.update({ip: int(rng.choice([3, 4], p=[0.6, 0.4])) for ip in servers})
    names = {ip: f"WS-{i:03d}" for i, ip in enumerate(hosts)}
    names.update({ip: f"SRV-{i:02d}" for i, ip in enumerate(servers)})
    persona = config.HOST_PERSONA
    scales = rng.lognormal(0.0, persona["scale_sigma"], len(hosts))
    night = rng.random(len(hosts)) < persona["night_shift_fraction"]
    return IpPools(
        pools=pools, criticality=criticality, device_names=names,
        persona_scale={str(ip): float(scale) for ip, scale in zip(hosts, scales)},
        persona_shift={str(ip): ("night_shift" if is_night else "day_shift")
                       for ip, is_night in zip(hosts, night)},
    )


def _draw_endpoints(kind: str, size: int, ip_pools: IpPools,
                    rng: np.random.Generator) -> np.ndarray:
    """Pick endpoint IPs for a role (``mixed`` = mostly external, some servers)."""
    if kind == "mixed":
        external = rng.choice(ip_pools.pools["external"], size=size)
        internal = rng.choice(ip_pools.pools["server"], size=size)
        return np.where(rng.random(size) < 0.8, external, internal)
    return rng.choice(ip_pools.pools[kind], size=size)


def draw_hours(profile_name: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample hour-of-day according to a named activity profile."""
    spec = config.HOUR_PROFILES[profile_name]
    weights = np.ones(24)
    weights[spec["peak_hours"]] = spec["peak_weight"]
    return rng.choice(24, size=size, p=weights / weights.sum())


def _persona_hours(frame: pd.DataFrame, profile: dict, ip_pools: IpPools,
                   rng: np.random.Generator) -> np.ndarray:
    """Hours drawn from each source host's own shift (or the opposite shift for insiders)."""
    shift = frame["src_ip"].map(ip_pools.persona_shift).fillna("day_shift").to_numpy()
    if profile["hours"] == "off_shift":
        flip = {"day_shift": "night_shift", "night_shift": "day_shift"}
        shift = np.array([flip[name] for name in shift])
    hours = np.zeros(len(frame), dtype=int)
    for name in ("day_shift", "night_shift"):
        mask = shift == name
        if mask.any():
            hours[mask] = draw_hours(name, int(mask.sum()), rng)
    return hours


def _apply_persona(frame: pd.DataFrame, profile: dict, ip_pools: IpPools) -> pd.DataFrame:
    """Scale behaviour by the source host's activity level, then apply any boost."""
    scale = frame["src_ip"].map(ip_pools.persona_scale).fillna(1.0).to_numpy()
    factors = {column: scale for column in config.HOST_PERSONA["scaled_columns"]}
    for column, boost in profile.get("boost", {}).items():
        factors[column] = factors.get(column, 1.0) * boost
    for column, factor in factors.items():
        frame[column] = np.round(frame[column] * factor)
    for column in config.MIN_ONE_COLUMNS + ["src_packets", "dst_packets"]:
        frame[column] = np.maximum(1, frame[column])
    return frame


def _draw_behaviour(profile: dict, size: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample the numeric behavioural features for one traffic class."""
    frame = pd.DataFrame({
        name: sample_distribution(profile[name], size, rng)
        for name in ("duration", "src_bytes", "dst_bytes", "failed_logins", "conn_count_1m",
                     "unique_dst_ports_1m", "unique_dst_hosts_1m", "dns_query_rate",
                     "syn_ratio", "beacon_interval_std")
    })
    frame["src_packets"] = np.maximum(1, np.round(frame["src_bytes"] / sample_distribution(profile["bytes_per_packet"], size, rng)))
    frame["dst_packets"] = np.maximum(1, np.round(frame["dst_bytes"] / sample_distribution(profile["bytes_per_packet"], size, rng)))
    frame["src_bytes"] = np.round(frame["src_bytes"])
    frame["dst_bytes"] = np.round(frame["dst_bytes"])
    for column in config.MIN_ONE_COLUMNS:
        frame[column] = np.maximum(1, frame[column])
    return frame[config.BEHAVIOUR_COLUMNS]


def _apply_stealth(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Blend a fraction of attack flows with normal behaviour to create overlap."""
    stealth_idx = np.flatnonzero(rng.random(len(frame)) < config.STEALTH_FRACTION)
    if stealth_idx.size == 0:
        return frame
    normal = _draw_behaviour(config.SYNTHETIC_PROFILES[config.NORMAL_LABEL], stealth_idx.size, rng)
    mask = rng.random((stealth_idx.size, len(config.BEHAVIOUR_COLUMNS))) < config.STEALTH_BLEND
    blended = np.where(mask, normal.to_numpy(), frame.iloc[stealth_idx].to_numpy())
    frame = frame.copy()
    frame.iloc[stealth_idx] = blended
    return frame


def _build_class_frame(label: str, size: int, ip_pools: IpPools, rng: np.random.Generator,
                       apply_stealth: bool) -> pd.DataFrame:
    """Generate all columns for ``size`` flows of one traffic class."""
    profile = config.SYNTHETIC_PROFILES[label]
    behaviour = _draw_behaviour(profile, size, rng)
    if apply_stealth:
        behaviour = _apply_stealth(behaviour, rng)
    behaviour = behaviour.reset_index(drop=True)
    frame = behaviour.copy()
    frame["protocol"] = weighted_choice(profile["protocols"], size, rng)
    frame["dst_port"] = weighted_choice(profile["dst_ports"], size, rng).astype(int)
    frame["src_port"] = rng.integers(1024, config.MAX_PORT + 1, size)
    frame["src_ip"] = _draw_endpoints(profile["src_kind"], size, ip_pools, rng)
    frame["dst_ip"] = _draw_endpoints(profile["dst_kind"], size, ip_pools, rng)
    if profile.get("persona"):
        frame = _apply_persona(frame, profile, ip_pools)
    if profile["hours"] in config.PERSONA_HOUR_MODES:
        hours = _persona_hours(frame, profile, ip_pools, rng)
    else:
        hours = draw_hours(profile["hours"], size, rng)
    day = rng.integers(0, config.SIM_DAYS, size)
    seconds = rng.integers(0, 3600, size)
    start = pd.Timestamp(config.SIM_START)
    frame["timestamp"] = start + pd.to_timedelta(day * 86400 + hours * 3600 + seconds, unit="s")
    frame[config.LABEL_COLUMN] = label
    return frame


def attach_context(frame: pd.DataFrame, ip_pools: IpPools) -> pd.DataFrame:
    """Add asset criticality, user and device columns derived from the IPs."""
    crit = frame["dst_ip"].map(ip_pools.criticality)
    crit = crit.fillna(frame["src_ip"].map(ip_pools.criticality)).fillna(1)
    frame["asset_criticality"] = crit.astype(int)
    device = frame["src_ip"].map(ip_pools.device_names)
    frame["device"] = device.fillna(frame["dst_ip"].map(ip_pools.device_names)).fillna("EXTERNAL")
    frame["user"] = np.where(frame["src_ip"].isin(ip_pools.criticality),
                             "user_" + frame["src_ip"].str.replace(".", "_", regex=False), "external")
    return frame


def generate_flows(n_flows: int = config.N_FLOWS, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Generate a labelled, time-ordered flow dataset following ``CLASS_PRIORS``."""
    rng = np.random.default_rng(seed)
    ip_pools = build_ip_pools(rng)
    labels = list(config.CLASS_PRIORS.keys())
    counts = rng.multinomial(n_flows, list(config.CLASS_PRIORS.values()))
    frames = [_build_class_frame(label, int(count), ip_pools, rng, apply_stealth=(label != config.NORMAL_LABEL
                                                    and not config.SYNTHETIC_PROFILES[label].get("persona")))
              for label, count in zip(labels, counts) if count > 0]
    flows = pd.concat(frames, ignore_index=True)
    flows = attach_context(flows, ip_pools)
    flows = flows.sort_values("timestamp").reset_index(drop=True)
    flows.insert(0, "flow_id", [f"F{i:07d}" for i in range(len(flows))])
    logger.info("Generated %d flows: %s", len(flows), flows[config.LABEL_COLUMN].value_counts().to_dict())
    return flows


def generate_novel_flows(n_flows: int = config.N_NOVEL_FLOWS, seed: int = config.RANDOM_SEED + 1,
                         n_benign: int | None = None) -> pd.DataFrame:
    """Generate held-out novel-attack flows (mixed with benign flows) for zero-day tests."""
    rng = np.random.default_rng(seed)
    ip_pools = build_ip_pools(np.random.default_rng(config.RANDOM_SEED))
    n_benign = n_flows if n_benign is None else n_benign
    frames = [
        _build_class_frame(config.NOVEL_LABEL, n_flows, ip_pools, rng, apply_stealth=False),
        _build_class_frame(config.NORMAL_LABEL, n_benign, ip_pools, rng, apply_stealth=False),
    ]
    flows = attach_context(pd.concat(frames, ignore_index=True), ip_pools)
    flows = flows.sort_values("timestamp").reset_index(drop=True)
    flows.insert(0, "flow_id", [f"N{i:07d}" for i in range(len(flows))])
    return flows
