"""Per-entity behavioural baselines (UEBA).

A global model only knows what is normal for the whole network. This module
learns what is normal for *each* entity (by default each source IP; use
``config.ENTITY_KEY = "user"`` with real user IDs) from benign history, then
scores how far a new flow deviates from that entity's own habits.

Features produced (see ``config.ENTITY_FEATURES``):
    entity_max_z           largest deviation across the profiled behaviours
    entity_mean_z          average deviation across the profiled behaviours
    entity_hour_surprisal  how unusual this hour is for this entity
    entity_new_port        1 if the entity never used this destination port before
    entity_known           1 if the entity has enough history to trust its baseline

Entities with little history are blended toward the global baseline
(shrinkage), and unseen entities fall back to it entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import config
from src.utils import get_logger

logger = get_logger("entity_baseline")
_HOURS = 24


def _log_profile(flows: pd.DataFrame) -> np.ndarray:
    """Log-scaled behaviour matrix (rows = flows, columns = profiled behaviours)."""
    return np.log1p(flows[config.ENTITY_PROFILE_COLUMNS].clip(lower=0)).to_numpy(dtype=float)


def _entity_keys(flows: pd.DataFrame) -> pd.Series:
    """Entity identifier for each flow (missing values grouped as 'unknown')."""
    return flows[config.ENTITY_KEY].fillna("unknown").astype(str)


def _hours(flows: pd.DataFrame) -> np.ndarray:
    """Hour of day (0-23) for each flow."""
    return pd.to_datetime(flows[config.TIMESTAMP_COLUMN]).dt.hour.to_numpy()


def _ports(flows: pd.DataFrame) -> np.ndarray:
    """Destination ports as integers (missing -> -1)."""
    return flows["dst_port"].fillna(-1).astype(int).to_numpy()


class EntityBaselineStore:
    """Learns and applies per-entity baselines from benign flows."""

    def __init__(self, min_flows: int = config.ENTITY_MIN_FLOWS,
                 shrinkage: float = config.ENTITY_SHRINKAGE,
                 min_std: float = config.ENTITY_MIN_STD) -> None:
        self.min_flows = min_flows
        self.shrinkage = shrinkage
        self.min_std = min_std
        self.index_: pd.Index | None = None

    @property
    def n_entities(self) -> int:
        """Number of entities with a learned baseline."""
        return 0 if self.index_ is None else len(self.index_)

    def fit(self, benign_flows: pd.DataFrame) -> "EntityBaselineStore":
        """Learn baselines from benign (cleaned) flows."""
        if benign_flows.empty:
            raise ValueError("Cannot fit entity baselines on an empty dataset.")
        keys = _entity_keys(benign_flows).to_numpy()
        profile = pd.DataFrame(_log_profile(benign_flows), columns=config.ENTITY_PROFILE_COLUMNS)

        self.global_mean_ = profile.mean().to_numpy()
        self.global_std_ = np.maximum(profile.std().to_numpy(), self.min_std)
        grouped = profile.groupby(keys)
        counts = grouped.size()
        weight = (counts / (counts + self.shrinkage)).to_numpy()[:, None]
        self.index_ = counts.index
        self.counts_ = counts.to_numpy()
        self.means_ = weight * grouped.mean().to_numpy() + (1 - weight) * self.global_mean_
        entity_std = grouped.std().fillna(0.0).to_numpy()
        self.stds_ = np.maximum(weight * entity_std + (1 - weight) * self.global_std_, self.min_std)

        hour_counts = (pd.crosstab(keys, _hours(benign_flows))
                       .reindex(index=self.index_, columns=range(_HOURS), fill_value=0))
        global_hours = hour_counts.sum(axis=0).to_numpy() + 1.0
        self.global_hour_prob_ = global_hours / global_hours.sum()
        self.hour_prob_ = ((hour_counts.to_numpy() + self.shrinkage * self.global_hour_prob_)
                           / (self.counts_[:, None] + self.shrinkage))
        self.known_ports_ = set(zip(keys, _ports(benign_flows)))
        logger.info("Learned baselines for %d entities from %d benign flows",
                    self.n_entities, len(benign_flows))
        return self

    def transform(self, flows: pd.DataFrame) -> pd.DataFrame:
        """Return the entity features for ``flows`` (same index as ``flows``)."""
        if self.index_ is None:
            raise RuntimeError("EntityBaselineStore must be fitted before transform().")
        keys = _entity_keys(flows)
        position = self.index_.get_indexer(keys)
        seen = position >= 0
        safe = np.where(seen, position, 0)

        means = np.where(seen[:, None], self.means_[safe], self.global_mean_)
        stds = np.where(seen[:, None], self.stds_[safe], self.global_std_)
        z = np.nan_to_num(np.abs((_log_profile(flows) - means) / stds), nan=0.0)
        z = np.minimum(z, config.ENTITY_Z_CAP)

        known = seen & (self.counts_[safe] >= self.min_flows)
        hour_prob = np.where(seen, self.hour_prob_[safe, _hours(flows)],
                             self.global_hour_prob_[_hours(flows)])
        new_port = np.array([(key, port) not in self.known_ports_
                             for key, port in zip(keys, _ports(flows))], dtype=bool) & known
        return pd.DataFrame({
            "entity_max_z": z.max(axis=1),
            "entity_mean_z": z.mean(axis=1),
            "entity_hour_surprisal": -np.log(hour_prob),
            "entity_new_port": new_port.astype(int),
            "entity_known": known.astype(int),
        }, index=flows.index)[config.ENTITY_FEATURES]


def cross_fit_features(flows: pd.DataFrame, benign_mask: np.ndarray,
                       n_splits: int = config.ENTITY_CV_FOLDS,
                       seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Entity features for training data without self-comparison leakage.

    Each fold is scored against baselines learned from the *other* folds'
    benign flows, so a flow is never compared with a baseline that contains it.
    """
    benign_mask = np.asarray(benign_mask, dtype=bool)
    features = pd.DataFrame(index=flows.index, columns=config.ENTITY_FEATURES, dtype=float)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, apply_idx in splitter.split(flows):
        store = EntityBaselineStore().fit(flows.iloc[fit_idx][benign_mask[fit_idx]])
        features.iloc[apply_idx] = store.transform(flows.iloc[apply_idx]).to_numpy()
    return features.astype(float)
