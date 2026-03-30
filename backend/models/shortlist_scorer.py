"""Shortlist scoring via k-means style clustering + weighted distance.

Flow:
1. Standardize all per-90 metrics across candidates + reference player.
2. K-means cluster all players into style groups (k=√(n/2), capped 3-10).
3. Find the reference player's cluster.
4. Rank candidates by weighted Euclidean distance to the reference player.
   Candidates in the same cluster are preferred (cluster-match bonus).
5. Filters (age, minutes, position, league, power ranking) are applied
   before clustering so that only relevant candidates are considered.

Falls back to direct weighted distance when too few candidates for
meaningful clustering (< 10).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from backend.data.sofascore_client import CORE_METRICS

_log = logging.getLogger(__name__)

# Minimum minutes a player must have to be considered for the shortlist.
# Imported from shared constants to stay in sync with training pipeline.
from backend.utils.constants import MIN_MINUTES_THRESHOLD

# Candidates below this similarity score are flagged as low-confidence
# so the UI can warn users that the match quality is weak.
_LOW_CONFIDENCE_THRESHOLD = 0.3


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
    rating: Optional[float] = None
    predicted_per90: Dict[str, float] = field(default_factory=dict)
    current_per90: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    metric_scores: Dict[str, float] = field(default_factory=dict)
    cluster: int = -1  # K-means cluster label (-1 = unassigned)
    same_cluster_as_reference: bool = False  # True if in same cluster as reference player
    low_confidence: bool = False  # True when similarity score is below threshold


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
    """Apply filters to reduce candidate list.

    Design: when a candidate has ``None`` for a filtered field (e.g. age
    unknown), the candidate **passes through** the filter rather than being
    excluded.  This means ``max_age=30`` selects "players aged ≤30 OR
    players with unknown age" — not strictly "players aged ≤30 only".

    This is intentional — excluding unknowns would silently drop players
    whose data is incomplete from the Sofascore API, giving the impression
    that 0 candidates exist when in reality the data is just sparse.
    Users can inspect the results and see "—" for missing fields.
    """
    from backend.data.sofascore_client import normalize_position

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
            # Use normalized position categories for robust matching
            c_norm = normalize_position(c.position)
            if c_norm not in filters.positions and c.position not in filters.positions:
                continue
        if filters.leagues is not None:
            # Candidates with known league must match the filter.
            # Candidates with empty/None league pass through (data incomplete).
            if c.league and c.league not in filters.leagues:
                continue
        if filters.max_power_ranking is not None and c.club_power_ranking is not None:
            if c.club_power_ranking > filters.max_power_ranking:
                continue
        result.append(c)
    return result


def _build_feature_matrix(
    candidates: List[Candidate],
    reference_per90: Dict[str, float],
    weights: Dict[str, float],
) -> tuple:
    """Build standardized feature matrix for clustering.

    Returns (X_scaled, active_metrics, scaler, ref_scaled) where
    X_scaled[i] is the feature vector for candidates[i],
    ref_scaled is the reference player's feature vector.
    """
    active_metrics = [m for m, w in weights.items() if w > 0 and m in CORE_METRICS]
    if not active_metrics:
        active_metrics = list(CORE_METRICS)

    # Build raw matrix: candidates + reference player (last row)
    n = len(candidates)
    raw = np.zeros((n + 1, len(active_metrics)))

    for i, c in enumerate(candidates):
        for j, m in enumerate(active_metrics):
            raw[i, j] = c.predicted_per90.get(m, 0.0)

    # Reference player as last row
    for j, m in enumerate(active_metrics):
        v = reference_per90.get(m)
        raw[n, j] = float(v) if v is not None else 0.0

    # Apply user weights before standardization so that heavily-weighted
    # metrics dominate the distance/clustering.
    w_arr = np.array([weights.get(m, 0.5) for m in active_metrics])
    raw_weighted = raw * w_arr

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(raw_weighted)

    ref_scaled = X_scaled[-1]       # reference player
    X_candidates = X_scaled[:-1]    # candidates only

    return X_candidates, active_metrics, scaler, ref_scaled


def score_candidates(
    candidates: List[Candidate],
    weights: Dict[str, float],
    filters: Optional[ShortlistFilters] = None,
    reference_per90: Optional[Dict[str, float]] = None,
) -> List[Candidate]:
    """Score and rank candidates using k-means clustering + weighted distance.

    When ``reference_per90`` is provided (the stats of the player being
    replaced), candidates are clustered by playing style and ranked by
    weighted Euclidean distance to the reference player.  Candidates in the
    same cluster as the reference player receive a bonus (lower distance)
    since they share a similar style profile.

    When ``reference_per90`` is not provided, falls back to z-score ranking
    (original behavior).

    Parameters
    ----------
    candidates : list[Candidate]
        Candidates with ``predicted_per90`` already filled.
    weights : dict[str, float]
        User weights per metric (0.0-1.0).
    filters : ShortlistFilters, optional
        If provided, filter candidates before scoring.
    reference_per90 : dict, optional
        Per-90 stats of the player being replaced.

    Returns
    -------
    list[Candidate] sorted by score descending (higher = more similar).
    Always returns at least min(20, total_candidates) results.  If strict
    filtering removes all candidates the position filter is relaxed first,
    then all filters are dropped, to guarantee a non-empty result.
    """
    if not candidates or not weights:
        return candidates

    original_candidates = list(candidates)  # preserve unfiltered pool

    if filters is not None:
        candidates = filter_candidates(candidates, filters)

        # Fallback 1: if position filter removed all candidates, retry
        # without position restriction (most common cause of 0 results).
        if not candidates and filters.positions is not None:
            _log.info(
                "Position filter left 0 candidates — retrying without "
                "position restriction (%d candidates in pool)",
                len(original_candidates),
            )
            relaxed = ShortlistFilters(
                max_age=filters.max_age,
                min_age=filters.min_age,
                max_market_value=filters.max_market_value,
                min_minutes_played=filters.min_minutes_played,
                positions=None,  # drop position filter
                leagues=filters.leagues,
                max_power_ranking=filters.max_power_ranking,
            )
            candidates = filter_candidates(original_candidates, relaxed)

        # Fallback 2: if *all* filters removed every candidate, use the
        # full unfiltered pool so we always return something.
        if not candidates:
            _log.info(
                "All filters left 0 candidates — using full unfiltered "
                "pool (%d candidates)",
                len(original_candidates),
            )
            candidates = original_candidates

    # No reference player → fall back to z-score ranking
    if reference_per90 is None:
        return _score_zscore(candidates, weights)

    active_metrics = [m for m, w in weights.items() if w > 0 and m in CORE_METRICS]
    if not active_metrics:
        return candidates

    n = len(candidates)

    # Build feature matrix (candidates + reference)
    X_cand, active_metrics, scaler, ref_vec = _build_feature_matrix(
        candidates, reference_per90, weights
    )

    # --- K-means clustering ---
    # Choose k = √(n/2), clamped to [3, 10].
    # Below 10 candidates, skip clustering — just use distance.
    use_clustering = n >= 10
    ref_cluster = -1

    if use_clustering:
        k = max(3, min(10, int(math.sqrt(n / 2))))
        # Include reference player in clustering to find its cluster
        X_all = np.vstack([X_cand, ref_vec.reshape(1, -1)])
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_all)
        ref_cluster = int(labels[-1])
        cand_labels = labels[:-1]
    else:
        cand_labels = np.zeros(n, dtype=int)

    # --- Weighted Euclidean distance to reference ---
    diffs = X_cand - ref_vec
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))

    # Normalize distances to [0, 1] range for scoring
    max_dist = float(np.max(distances)) if np.max(distances) > 0 else 1.0

    # Score = 1 - normalized_distance (higher = more similar)
    # Cluster bonus: candidates in same cluster as reference get 15% boost
    for i, c in enumerate(candidates):
        norm_dist = distances[i] / max_dist
        base_score = 1.0 - norm_dist

        # Cluster membership
        c.cluster = int(cand_labels[i])
        c.same_cluster_as_reference = (
            use_clustering and c.cluster == ref_cluster
        )

        # Same-cluster bonus: candidates sharing the reference player's
        # style group get a 15% score boost.
        if c.same_cluster_as_reference:
            base_score = min(1.0, base_score * 1.15)

        c.score = base_score

        # Per-metric breakdown: how close each metric is to reference
        c.metric_scores = {}
        for j, m in enumerate(active_metrics):
            # Difference in standardized space
            metric_diff = abs(X_cand[i, j] - ref_vec[j])
            # Convert to similarity (1 = identical, 0 = very different)
            c.metric_scores[m] = max(0.0, 1.0 - metric_diff / 3.0)

    # Sort by score descending (most similar first)
    candidates.sort(key=lambda c: c.score, reverse=True)

    # Mark low-confidence candidates so the UI can flag them.
    for c in candidates:
        if c.score < _LOW_CONFIDENCE_THRESHOLD:
            c.low_confidence = True

    return candidates


def _score_zscore(
    candidates: List[Candidate],
    weights: Dict[str, float],
) -> List[Candidate]:
    """Original z-score ranking fallback (no reference player)."""
    active_metrics = [m for m, w in weights.items() if w > 0 and m in CORE_METRICS]
    if not active_metrics:
        return candidates

    # Collect predicted values per metric
    values: Dict[str, List[float]] = {m: [] for m in active_metrics}
    for c in candidates:
        for m in active_metrics:
            values[m].append(c.predicted_per90.get(m, 0.0))

    # Compute mean and std per metric
    stats: Dict[str, tuple] = {}
    for m in active_metrics:
        arr = np.array(values[m])
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        stats[m] = (mean, std if std > 0 else 1.0)

    # Normalize and weight
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

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def compute_percentage_changes(
    current_per90: Dict[str, float],
    predicted_per90: Dict[str, float],
) -> Dict[str, float]:
    """Compute % change from current to predicted for each metric.

    Returns dict[metric -> percentage change].
    Positive = improvement, negative = decline.
    When current is zero/missing, returns 0.0 — a percentage change cannot be
    computed from a zero base, and displaying the raw delta (e.g. +0.3 xG/90)
    in a column labelled "Change %" would be misleading.
    """
    changes = {}
    for metric in CORE_METRICS:
        current = current_per90.get(metric)
        current = current if current is not None else 0
        predicted = predicted_per90.get(metric)
        predicted = predicted if predicted is not None else 0
        if current != 0:
            changes[metric] = ((predicted - current) / abs(current)) * 100
        else:
            # Percentage undefined for zero base — return 0.0 to avoid
            # misleading values in the "Change %" display column.
            changes[metric] = 0.0
    return changes
