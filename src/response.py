"""Response recommendation: translate threat score and type into an action."""

from __future__ import annotations

import numpy as np

import config


def recommend_actions(scores: np.ndarray, threat_types: np.ndarray) -> np.ndarray:
    """Return the recommended action for each flow.

    Low-risk bands use generic actions (Allow / Monitor / Investigate); in the
    high-risk bands the action is looked up from ``config.ACTION_POLICY`` by
    threat type, so e.g. a C2 channel is quarantined while a scan is investigated.
    """
    actions = np.empty(len(scores), dtype=object)
    lower = -np.inf
    for upper, _, generic_action in config.SEVERITY_BANDS:
        in_band = (scores >= lower) & (scores < upper)
        if generic_action is not None:
            actions[in_band] = generic_action
        else:
            for idx in np.flatnonzero(in_band):
                actions[idx] = config.ACTION_POLICY.get(threat_types[idx], "Investigate")
        lower = upper
    return actions


def mitre_columns(threat_types: np.ndarray) -> tuple[list[str], list[str]]:
    """Return (tactic, 'Txxxx name') lists aligned with ``threat_types``."""
    tactics, techniques = [], []
    for threat_type in threat_types:
        tactic, technique_id, name = config.MITRE_MAPPING.get(threat_type, ("Unknown", "N/A", "Unmapped"))
        tactics.append(tactic)
        techniques.append(f"{technique_id} {name}" if technique_id != "N/A" else name)
    return tactics, techniques
