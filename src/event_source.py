"""Event sources: where live events come from.

Every source yields micro-batches (DataFrames) carrying an ``emitted_at`` column -
the wall-clock moment (``time.perf_counter`` basis) the event became available.
Detection latency is measured from that moment to the moment the event is scored.

To connect a real feed (Kafka, a syslog/NetFlow collector, a webhook), write a
new subclass whose ``batches()`` converts incoming messages into a DataFrame with
the same columns as the simulator's raw events and sets ``emitted_at``. Nothing
else in the detector changes.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np
import pandas as pd

import config
from src.stream_features import epoch_seconds


class EventSource(ABC):
    """Interface for anything that produces micro-batches of raw events."""

    @abstractmethod
    def batches(self) -> Iterator[pd.DataFrame]:
        """Yield time-ordered micro-batches until the source is exhausted."""


class PacedEventSource(EventSource):
    """Replays a simulated stream in (accelerated) real time.

    Event ``i`` becomes available ``(t_i - t_0) / speed`` wall-clock seconds after
    the run starts. If the detector falls behind, batches grow and latency rises,
    which is exactly what the latency statistics are meant to reveal.
    """

    def __init__(self, events: pd.DataFrame, speed: float = config.LIVE_DEFAULT_SPEED,
                 max_batch: int = config.LIVE_MAX_BATCH, poll_seconds: float = config.LIVE_POLL_SECONDS) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.events = events.sort_values(config.TIMESTAMP_COLUMN, kind="stable").reset_index(drop=True)
        seconds = epoch_seconds(self.events[config.TIMESTAMP_COLUMN])
        self.due = (seconds - seconds[0]) / speed
        self.max_batch = max_batch
        self.poll_seconds = poll_seconds

    def batches(self) -> Iterator[pd.DataFrame]:
        start, position, total = time.perf_counter(), 0, len(self.events)
        while position < total:
            elapsed = time.perf_counter() - start
            end = min(int(np.searchsorted(self.due, elapsed, side="right")), position + self.max_batch)
            if end > position:
                batch = self.events.iloc[position:end].copy()
                batch["emitted_at"] = start + self.due[position:end]
                position = end
                yield batch
            else:
                time.sleep(min(self.poll_seconds, max(0.0, self.due[position] - elapsed)))


class ReplayEventSource(EventSource):
    """Feeds events as fast as possible (throughput test; latency = processing time)."""

    def __init__(self, events: pd.DataFrame, batch_size: int = config.LIVE_MAX_BATCH) -> None:
        self.events = events.sort_values(config.TIMESTAMP_COLUMN, kind="stable").reset_index(drop=True)
        self.batch_size = batch_size

    def batches(self) -> Iterator[pd.DataFrame]:
        for start in range(0, len(self.events), self.batch_size):
            batch = self.events.iloc[start:start + self.batch_size].copy()
            batch["emitted_at"] = time.perf_counter()
            yield batch
