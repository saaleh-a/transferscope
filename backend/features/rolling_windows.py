"""1000-min player rolling averages and 3000-min team rolling averages.

Weighted prior blend for low-data players.
RAG confidence: Red < 0.3, Amber 0.3-0.7, Green > 0.7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.data.sofascore_client import ALL_METRICS, CORE_METRICS

# ── Constants ────────────────────────────────────────────────────────────────

PLAYER_WINDOW_MINUTES = 1000
TEAM_WINDOW_MINUTES = 3000
PRIOR_CONSTANT = 1000  # C in the blend formula


@dataclass
class RollingFeatures:
    """Rolling averages for a player/team with confidence metadata."""

    per90: Dict[str, Optional[float]]
    minutes_used: float
    weight: float  # min(1, minutes / C)
    confidence: str  # "red", "amber", "green"


def compute_confidence(weight: float) -> str:
    """Map blend weight to RAG confidence status."""
    if weight < 0.3:
        return "red"
    elif weight <= 0.7:
        return "amber"
    return "green"


def blend_weight(minutes_played: float, constant: float = PRIOR_CONSTANT) -> float:
    """Compute the prior blend weight: min(1, minutes / C)."""
    return min(1.0, minutes_played / constant) if constant > 0 else 1.0


def blend_features(
    raw_per90: Dict[str, Optional[float]],
    prior_per90: Dict[str, Optional[float]],
    minutes_played: float,
    constant: float = PRIOR_CONSTANT,
) -> RollingFeatures:
    """Blend raw rolling averages with priors for low-data players.

    Formula: feature = (1 - weight) * prior + weight * raw_rolling_avg
    where weight = min(1, sum(minutes_played) / C)
    """
    w = blend_weight(minutes_played, constant)
    blended: Dict[str, Optional[float]] = {}

    for metric in ALL_METRICS:
        raw = raw_per90.get(metric)
        prior = prior_per90.get(metric)

        if raw is not None and prior is not None:
            blended[metric] = (1 - w) * prior + w * raw
        elif raw is not None:
            blended[metric] = raw
        elif prior is not None:
            blended[metric] = prior
        else:
            blended[metric] = None

    return RollingFeatures(
        per90=blended,
        minutes_used=minutes_played,
        weight=w,
        confidence=compute_confidence(w),
    )


def player_rolling_average(
    match_logs: List[Dict[str, Any]],
    window_minutes: float = PLAYER_WINDOW_MINUTES,
) -> Dict[str, Optional[float]]:
    """Compute per-90 rolling averages from match-level data.

    Parameters
    ----------
    match_logs : list[dict]
        Each dict must have ``minutes`` (int) and per-90 metric keys.
        Ordered most-recent first.
    window_minutes : float
        Rolling window size in minutes.

    Returns
    -------
    dict mapping metric name -> per-90 rolling average (or None).
    """
    if not match_logs:
        return {m: None for m in ALL_METRICS}

    totals: Dict[str, float] = {m: 0.0 for m in ALL_METRICS}
    counts: Dict[str, int] = {m: 0 for m in ALL_METRICS}
    minutes_accumulated = 0.0

    for log in match_logs:
        mins = log.get("minutes", 0)
        if mins is None:
            mins = 0
        if minutes_accumulated >= window_minutes:
            break
        minutes_accumulated += mins

        for metric in ALL_METRICS:
            val = log.get(metric)
            if val is not None:
                try:
                    totals[metric] += float(val) * mins
                    counts[metric] += mins
                except (ValueError, TypeError):
                    pass

    result: Dict[str, Optional[float]] = {}
    for metric in ALL_METRICS:
        if counts[metric] > 0:
            # Weighted average (minutes-weighted), then express as per-90
            result[metric] = totals[metric] / counts[metric]
        else:
            result[metric] = None

    return result


def team_rolling_average(
    team_match_logs: List[Dict[str, Any]],
    window_minutes: float = TEAM_WINDOW_MINUTES,
) -> Dict[str, Optional[float]]:
    """Compute per-90 rolling averages for team-level metrics.

    Same logic as player but with 3000-min window.
    """
    return player_rolling_average(team_match_logs, window_minutes)


def team_position_rolling_average(
    position_match_logs: List[Dict[str, Any]],
    window_minutes: float = TEAM_WINDOW_MINUTES,
) -> Dict[str, Optional[float]]:
    """Compute per-90 rolling averages for a specific position within a team.

    Same window as team (3000 mins).
    """
    return player_rolling_average(position_match_logs, window_minutes)


def compute_player_features(
    player_stats: Dict[str, Any],
    prior_per90: Optional[Dict[str, Optional[float]]] = None,
    match_logs: Optional[List[Dict[str, Any]]] = None,
) -> RollingFeatures:
    """Build player features from stats, optionally blended with priors.

    Parameters
    ----------
    player_stats : dict
        Output of ``sofascore_client.get_player_stats`` — must have
        ``per90`` dict and ``minutes_played``.
    prior_per90 : dict, optional
        League/position prior per-90 values. If None, no blending.
    match_logs : list, optional
        If provided, compute rolling average from match logs instead
        of using the season aggregate from player_stats.
    """
    minutes = player_stats.get("minutes_played", 0)
    if minutes is None:
        minutes = 0

    if match_logs:
        raw = player_rolling_average(match_logs)
    else:
        raw = player_stats.get("per90", {})

    if prior_per90 is None:
        prior_per90 = {m: None for m in ALL_METRICS}

    return blend_features(raw, prior_per90, minutes)
