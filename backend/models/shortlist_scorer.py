"""Weighted similarity scoring for shortlist generation.

normalized_target = (predicted_value - mean) / std
weighted_score = normalized_target * user_weight
final_score = sum(weighted_scores) / sum(weights)

Filters: age, market value, minutes played, position, league,
         club Power Ranking cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from backend.data.sofascore_client import CORE_METRICS


@dataclass
class Candidate:
    """A shortlist candidate with predicted metrics and metadata."""

    player_id: int
    name: str
    team: str
    position: str
    age: Optional[int] = None
    market_value: Optional[float] = None
    minutes_played: Optional[int] = None
    league: Optional[str] = None
    club_power_ranking: Optional[float] = None
    predicted_per90: Dict[str, float] = field(default_factory=dict)
    current_per90: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    metric_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class ShortlistFilters:
    """User-configurable filters for shortlist generation."""

    max_age: Optional[int] = None
    min_age: Optional[int] = None
    max_market_value: Optional[float] = None
    min_minutes_played: Optional[int] = None
    positions: Optional[List[str]] = None
    leagues: Optional[List[str]] = None
    max_power_ranking: Optional[float] = None


def filter_candidates(
    candidates: List[Candidate],
    filters: ShortlistFilters,
) -> List[Candidate]:
    """Apply filters to reduce candidate list."""
    result = []
    for c in candidates:
        if filters.max_age is not None and c.age is not None:
            if c.age > filters.max_age:
                continue
        if filters.min_age is not None and c.age is not None:
            if c.age < filters.min_age:
                continue
        if filters.max_market_value is not None and c.market_value is not None:
            if c.market_value > filters.max_market_value:
                continue
        if filters.min_minutes_played is not None and c.minutes_played is not None:
            if c.minutes_played < filters.min_minutes_played:
                continue
        if filters.positions is not None and c.position:
            if c.position not in filters.positions:
                continue
        if filters.leagues is not None and c.league:
            if c.league not in filters.leagues:
                continue
        if filters.max_power_ranking is not None and c.club_power_ranking is not None:
            if c.club_power_ranking > filters.max_power_ranking:
                continue
        result.append(c)
    return result


def score_candidates(
    candidates: List[Candidate],
    weights: Dict[str, float],
    filters: Optional[ShortlistFilters] = None,
) -> List[Candidate]:
    """Score and rank candidates using weighted similarity.

    Parameters
    ----------
    candidates : list[Candidate]
        Candidates with ``predicted_per90`` already filled.
    weights : dict[str, float]
        User weights per metric (0.0-1.0). Metrics not in this dict
        are ignored.
    filters : ShortlistFilters, optional
        If provided, filter candidates before scoring.

    Returns
    -------
    list[Candidate] sorted by score descending. Each candidate's
    ``score`` and ``metric_scores`` fields are updated.
    """
    if filters is not None:
        candidates = filter_candidates(candidates, filters)

    if not candidates or not weights:
        return candidates

    # Determine which metrics to use
    active_metrics = [m for m, w in weights.items() if w > 0 and m in CORE_METRICS]
    if not active_metrics:
        return candidates

    # Step 1 — Collect predicted values per metric
    values: Dict[str, List[float]] = {m: [] for m in active_metrics}
    for c in candidates:
        for m in active_metrics:
            values[m].append(c.predicted_per90.get(m, 0.0))

    # Step 2 — Compute mean and std per metric
    stats: Dict[str, tuple] = {}
    for m in active_metrics:
        arr = np.array(values[m])
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        stats[m] = (mean, std if std > 0 else 1.0)

    # Step 3 — Normalize and weight
    total_weight = sum(weights.get(m, 0) for m in active_metrics)
    if total_weight == 0:
        total_weight = 1.0

    for c in candidates:
        weighted_sum = 0.0
        c.metric_scores = {}
        for m in active_metrics:
            pred = c.predicted_per90.get(m, 0.0)
            mean, std = stats[m]
            normalized = (pred - mean) / std
            w = weights.get(m, 0.0)
            weighted = normalized * w
            c.metric_scores[m] = weighted
            weighted_sum += weighted

        c.score = weighted_sum / total_weight

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def compute_percentage_changes(
    current_per90: Dict[str, float],
    predicted_per90: Dict[str, float],
) -> Dict[str, float]:
    """Compute % change from current to predicted for each metric.

    Returns dict[metric -> percentage change].
    Positive = improvement, negative = decline.
    """
    changes = {}
    for metric in CORE_METRICS:
        current = current_per90.get(metric, 0)
        predicted = predicted_per90.get(metric, 0)
        if current and current != 0:
            changes[metric] = ((predicted - current) / abs(current)) * 100
        elif predicted:
            changes[metric] = 100.0
        else:
            changes[metric] = 0.0
    return changes
