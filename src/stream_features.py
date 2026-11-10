"""Sliding-window feature computation for live event streams.

Flow records arriving from a collector do not carry behavioural counters such as
"connections in the last minute". This module computes them incrementally, one
event at a time, keeping a short memory per source.

The same code builds the training data (``DATA_SOURCE = "events"``) and runs in
live detection, so the model never sees features computed differently at serving
time than at training time.

Events must be fed in non-decreasing timestamp order.
"""

from __future__ import annotations

import math
from collections import Counter, deque

import numpy as np
import pandas as pd

import config
from src.utils import get_logger

logger = get_logger("stream_features")
_DNS_PORT = 53
_EPOCH = pd.Timestamp("1970-01-01")


def epoch_seconds(timestamps: pd.Series) -> np.ndarray:
    """Convert a datetime Series to float seconds (independent of datetime unit)."""
    return (pd.to_datetime(timestamps) - _EPOCH).dt.total_seconds().to_numpy()


class _SourceState:
    """Events inside the window for one source, with running counters."""

    __slots__ = ("events", "ports", "hosts", "failed", "dns")

    def __init__(self) -> None:
        self.events: deque = deque()
        self.ports: Counter = Counter()
        self.hosts: Counter = Counter()
        self.failed = 0
        self.dns = 0


def _decrement(counter: Counter, key) -> None:
    counter[key] -= 1
    if counter[key] <= 0:
        del counter[key]


class SlidingWindowAggregator:
    """Maintains per-source windows and produces the behavioural features."""

    OUTPUT_COLUMNS = ["conn_count_1m", "unique_dst_ports_1m", "unique_dst_hosts_1m",
                      "failed_logins", "dns_query_rate", "beacon_interval_std"]

    def __init__(self, window_seconds: float = config.LIVE_WINDOW_SECONDS) -> None:
        self.window = float(window_seconds)
        self._sources: dict[str, _SourceState] = {}
        self._channels: dict[tuple, deque] = {}
        self._seen = 0

    @property
    def active_sources(self) -> int:
        """Number of sources currently held in memory."""
        return len(self._sources)

    def update(self, ts: float, src: str, dst: str, port: int, failed: int) -> tuple:
        """Ingest one event and return its features (see ``OUTPUT_COLUMNS``)."""
        state = self._sources.get(src)
        if state is None:
            state = self._sources[src] = _SourceState()
        events = state.events
        horizon = ts - self.window
        while events and events[0][0] <= horizon:
            _, old_dst, old_port, old_failed, old_dns = events.popleft()
            _decrement(state.hosts, old_dst)
            _decrement(state.ports, old_port)
            state.failed -= old_failed
            state.dns -= old_dns
        is_dns = 1 if port == _DNS_PORT else 0
        events.append((ts, dst, port, failed, is_dns))
        state.hosts[dst] += 1
        state.ports[port] += 1
        state.failed += failed
        state.dns += is_dns
        return (len(events), len(state.ports), len(state.hosts), state.failed, state.dns,
                self._beacon_std(ts, src, dst, port))

    def _beacon_std(self, ts: float, src: str, dst: str, port: int) -> float:
        """Std-dev of inter-arrival gaps on a (src, dst, port) channel."""
        key = (src, dst, port)
        channel = self._channels.get(key)
        if channel is None:
            channel = self._channels[key] = deque(maxlen=config.LIVE_BEACON_HISTORY)
        elif channel and ts - channel[-1] > config.LIVE_BEACON_STALE_SECONDS:
            channel.clear()
        channel.append(ts)
        if len(channel) < 3:
            return config.LIVE_BEACON_DEFAULT
        gaps = [b - a for a, b in zip(channel, list(channel)[1:])]
        mean = sum(gaps) / len(gaps)
        return math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps))

    def purge(self, now: float) -> None:
        """Drop idle sources/channels to bound memory."""
        idle = [s for s, st in self._sources.items() if not st.events or now - st.events[-1][0] > self.window]
        for source in idle:
            del self._sources[source]
        stale = [k for k, ch in self._channels.items() if not ch or now - ch[-1] > config.LIVE_BEACON_STALE_SECONDS]
        for key in stale:
            del self._channels[key]

    def process(self, events: pd.DataFrame) -> pd.DataFrame:
        """Compute features for a time-ordered batch (index matches ``events``)."""
        if events.empty:
            return pd.DataFrame(columns=self.OUTPUT_COLUMNS, index=events.index)
        times = epoch_seconds(events["timestamp"])
        sources = events[config.LIVE_AGG_KEY].astype(str).tolist()
        dsts = events["dst_ip"].astype(str).tolist()
        ports = events["dst_port"].fillna(-1).astype(int).tolist()
        failed = (events["auth_failed"].fillna(0).astype(int).tolist()
                  if "auth_failed" in events.columns else [0] * len(events))
        rows = []
        for ts, src, dst, port, fail in zip(times, sources, dsts, ports, failed):
            rows.append(self.update(ts, src, dst, port, fail))
            self._seen += 1
            if self._seen % config.LIVE_PURGE_EVERY == 0:
                self.purge(ts)
        return pd.DataFrame(rows, columns=self.OUTPUT_COLUMNS, index=events.index)
