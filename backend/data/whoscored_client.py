"""WhoScored player data client — spatial event data via public JSON API.

Supplements StatsBomb open data with broader player coverage from
WhoScored's public endpoints.  Uses ``curl_cffi`` for HTTP requests
with Cloudflare TLS-fingerprint bypass (falls back to stdlib
``requests`` if unavailable).

All external calls are routed through :mod:`backend.data.cache` with a
7-day TTL, per ARCHITECTURE.md rules.

WhoScored pitch coordinates: 100 × 100 (0,0 = bottom-left).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote as _url_quote

from backend.data import cache

_log = logging.getLogger(__name__)

# ── HTTP session ─────────────────────────────────────────────────────────────

try:
    from curl_cffi.requests import Session as _CurlSession

    _http = _CurlSession(impersonate="chrome110")
except (ImportError, Exception):
    import requests as _stdlib_requests

    _http = _stdlib_requests  # type: ignore[assignment]

# ── constants ────────────────────────────────────────────────────────────────

_CACHE_NAMESPACE = "whoscored"
_CACHE_TTL = 86400 * 7  # 7 days

_BASE_URL = "https://www.whoscored.com"
_API_BASE = "https://www.whoscored.com/api/v1"

# WhoScored uses a 100×100 pitch coordinate system.
_PITCH_WIDTH = 100.0
_PITCH_HEIGHT = 100.0

# Goal centre on a 100×100 pitch (right-hand side goal).
_GOAL_CENTER: Tuple[float, float] = (100.0, 50.0)
_BOX_X_THRESHOLD = 83.0  # approx 18-yard box on 100-scale
_PROGRESSIVE_PASS_THRESHOLD = 10.0  # % of pitch toward goal

# Pitch-third boundaries (y-axis: 0–100).
_LEFT_THIRD_MAX = 100.0 / 3
_RIGHT_THIRD_MIN = 2 * 100.0 / 3

# Negative-cache sentinel for 404s / player-not-found.
_NEGATIVE_SENTINEL = {"error": "not_found"}
_NEGATIVE_CACHE_TTL = 3600  # 1 hour

# Request headers mimicking a browser session.
_HEADERS: Dict[str, str] = {
    "Accept": "application/json",
    "Referer": "https://www.whoscored.com/",
    "X-Requested-With": "XMLHttpRequest",
}


# ── helpers ──────────────────────────────────────────────────────────────────


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two pitch coordinates."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _safe_float(val: Any) -> float:
    """Convert to float, returning 0.0 on failure."""
    try:
        return float(val) if val is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def _get_json(url: str, **kwargs: Any) -> Optional[Any]:
    """Issue GET request, return parsed JSON or None on any error."""
    try:
        resp = _http.get(url, headers=_HEADERS, timeout=15, **kwargs)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        _log.debug("WhoScored request failed (%s): %s", url, exc)
        return None


# ── public API ───────────────────────────────────────────────────────────────


def search_player(name: str) -> List[Dict[str, Any]]:
    """Find players by name via WhoScored search.

    Returns a list of dicts:
    ``{id, name, team_name, team_id, region_code}``.
    Returns ``[]`` on failure or no results.  Cached 7 days.
    """
    if not name or not name.strip():
        return []

    name_key = name.lower().strip()
    key = cache.make_key(_CACHE_NAMESPACE, "search", name_key)
    cached = cache.get(key, max_age=_CACHE_TTL)
    if cached is not None:
        return cached

    url = f"{_API_BASE}/Search/Players/?term={_url_quote(name)}"
    data = _get_json(url)
    if data is None or not isinstance(data, list):
        return []

    results: List[Dict[str, Any]] = []
    for item in data:
        results.append({
            "id": int(item.get("PlayerId", 0)),
            "name": str(item.get("PlayerName", "")),
            "team_name": str(item.get("TeamName", "")),
            "team_id": int(item.get("TeamId", 0)),
            "region_code": str(item.get("RegionCode", "")),
        })

    cache.set(key, results)
    _log.info("WhoScored search '%s' -> %d results", name, len(results))
    return results


def get_player_season_stats(
    player_id: int,
    season_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Fetch per-90 season statistics for a WhoScored player.

    Parameters
    ----------
    player_id : int
        WhoScored player numeric ID.
    season_id : int, optional
        WhoScored season ID.  If *None*, returns the latest available.

    Returns a dict with per-90 metrics keyed by canonical names where
    possible, plus ``minutes_played``, ``appearances``, ``rating``.
    Returns ``{}`` on failure or player-not-found.  Cached 7 days.
    """
    if player_id <= 0:
        return {}

    season_key = str(season_id) if season_id is not None else "latest"
    key = cache.make_key(_CACHE_NAMESPACE, "season_stats", str(player_id), season_key)
    cached = cache.get(key, max_age=_CACHE_TTL)
    if cached is not None:
        return cached if cached != _NEGATIVE_SENTINEL else {}

    neg_key = cache.make_key(_CACHE_NAMESPACE, "neg", "season_stats", str(player_id))
    neg_cached = cache.get(neg_key, max_age=_NEGATIVE_CACHE_TTL)
    if neg_cached is not None:
        return {}

    url = f"{_API_BASE}/Players/{player_id}/Statistics"
    if season_id is not None:
        url += f"?seasonId={season_id}"
    data = _get_json(url)
    if data is None:
        cache.set(neg_key, _NEGATIVE_SENTINEL)
        return {}

    stats = _parse_season_stats(data)
    cache.set(key, stats)
    return stats


def get_player_match_history(
    player_id: int,
    season_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch match-level performance logs for a WhoScored player.

    Returns list of dicts with keys:
    ``{match_id, date, opponent, minutes, rating, events}``.
    Returns ``[]`` on failure.  Cached 7 days.
    """
    if player_id <= 0:
        return []

    season_key = str(season_id) if season_id is not None else "latest"
    key = cache.make_key(_CACHE_NAMESPACE, "match_history", str(player_id), season_key)
    cached = cache.get(key, max_age=_CACHE_TTL)
    if cached is not None:
        return cached

    url = f"{_API_BASE}/Players/{player_id}/MatchHistory"
    if season_id is not None:
        url += f"?seasonId={season_id}"
    data = _get_json(url)
    if data is None or not isinstance(data, list):
        return []

    results: List[Dict[str, Any]] = []
    for item in data:
        results.append({
            "match_id": int(item.get("MatchId", 0)),
            "date": str(item.get("Date", "")),
            "opponent": str(item.get("Opponent", "")),
            "minutes": int(item.get("MinutesPlayed", 0)),
            "rating": _safe_float(item.get("Rating")),
            "events": item.get("Events", []),
        })

    cache.set(key, results)
    _log.info("WhoScored match history for %d -> %d matches", player_id, len(results))
    return results


def get_player_heatmap_data(
    player_id: int,
    season_id: Optional[int] = None,
) -> List[Tuple[float, float]]:
    """Fetch aggregated touch/action locations for heatmap rendering.

    Returns list of ``(x, y)`` tuples in WhoScored's 100×100 coords.
    Returns ``[]`` if unavailable.  Cached 7 days.
    """
    if player_id <= 0:
        return []

    season_key = str(season_id) if season_id is not None else "latest"
    key = cache.make_key(_CACHE_NAMESPACE, "heatmap", str(player_id), season_key)
    cached = cache.get(key, max_age=_CACHE_TTL)
    if cached is not None:
        return cached

    url = f"{_API_BASE}/Players/{player_id}/Heatmap"
    if season_id is not None:
        url += f"?seasonId={season_id}"
    data = _get_json(url)
    if data is None or not isinstance(data, list):
        return []

    points: List[Tuple[float, float]] = []
    for item in data:
        x = _safe_float(item.get("x"))
        y = _safe_float(item.get("y"))
        if 0 <= x <= _PITCH_WIDTH and 0 <= y <= _PITCH_HEIGHT:
            points.append((x, y))

    cache.set(key, points)
    return points


def compute_spatial_features(
    player_id: int,
    season_id: Optional[int] = None,
) -> Dict[str, float]:
    """Compute aggregate spatial features from WhoScored event data.

    Returns dict with the same keys as
    :func:`statsbomb_client.compute_spatial_features`:

    - ``avg_shot_distance``: mean distance of shots from goal centre
    - ``shots_inside_box_pct``: % of shots from inside 18-yard box
    - ``avg_pass_length``: mean pass distance
    - ``progressive_pass_pct``: % of passes moving ball >10% toward goal
    - ``avg_carry_distance``: mean carry distance
    - ``touches_left_pct``, ``touches_center_pct``, ``touches_right_pct``
    - ``avg_defensive_distance``: mean distance of defensive actions

    Returns ``{}`` if no data available.  Cached 7 days.
    """
    if player_id <= 0:
        return {}

    season_key = str(season_id) if season_id is not None else "latest"
    key = cache.make_key(_CACHE_NAMESPACE, "spatial_features", str(player_id), season_key)
    cached = cache.get(key, max_age=_CACHE_TTL)
    if cached is not None:
        return cached if cached != _NEGATIVE_SENTINEL else {}

    # Fetch match history and extract events
    matches = get_player_match_history(player_id, season_id)
    if not matches:
        return {}

    # Collect all events across matches
    all_events: List[Dict[str, Any]] = []
    for match in matches:
        events = match.get("events", [])
        if isinstance(events, list):
            all_events.extend(events)

    if not all_events:
        return {}

    features = _compute_features_from_events(all_events)
    if features:
        cache.set(key, features)
    return features


# ── internal helpers ─────────────────────────────────────────────────────────


def _parse_season_stats(data: Any) -> Dict[str, Any]:
    """Parse WhoScored season stats response into canonical dict."""
    if not isinstance(data, dict):
        return {}

    stats = data.get("Statistics") or data
    if not isinstance(stats, dict):
        return {}

    # WhoScored key → canonical name
    _KEY_MAP = {
        "Rating": "rating",
        "MinutesPlayed": "minutes_played",
        "Appearances": "appearances",
        "Goals": "goals",
        "Assists": "assists",
        "ShotsPerGame": "shots",
        "KeyPassesPerGame": "chances_created",
        "DribblesWonPerGame": "successful_dribbles",
        "PassSuccessPercentage": "pass_completion_pct",
        "AerialWonPerGame": "aerial_duels_won",
        "TacklesPerGame": "tackles",
        "InterceptionsPerGame": "interceptions",
        "ClearancesPerGame": "clearances",
        "CrossesPerGame": "successful_crosses",
        "LongBallsPerGame": "accurate_long_balls",
        "ThroughBallsPerGame": "through_balls",
        "FoulsCommittedPerGame": "fouls_committed",
    }

    result: Dict[str, Any] = {}
    for ws_key, canonical in _KEY_MAP.items():
        val = stats.get(ws_key)
        if val is not None:
            result[canonical] = _safe_float(val)

    return result


def _compute_features_from_events(
    events: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute spatial features from a flat list of WhoScored events.

    WhoScored events typically have: type, x, y, end_x, end_y, qualifiers.
    """
    shot_distances: List[float] = []
    shots_in_box = 0
    total_shots = 0

    pass_lengths: List[float] = []
    progressive_passes = 0
    total_passes = 0

    carry_distances: List[float] = []

    left_count = 0
    center_count = 0
    right_count = 0
    total_touches = 0

    defensive_distances: List[float] = []

    _SHOT_TYPES = {"Shot", "ShotOnTarget", "Goal", "MissedShots",
                   "ShotBlocked", "AttemptSaved"}
    _PASS_TYPES = {"Pass", "KeyPass", "Assist", "CrossNotClaimed",
                   "CrossInaccurate", "CrossAccurate"}
    _CARRY_TYPES = {"TakeOn", "Carry", "BallTouch"}
    _DEFENSIVE_TYPES = {"Tackle", "Interception", "BlockedPass", "Clearance"}

    for ev in events:
        ev_type = str(ev.get("type", ""))
        x = _safe_float(ev.get("x"))
        y = _safe_float(ev.get("y"))

        if x <= 0 and y <= 0:
            continue

        # Touch distribution
        total_touches += 1
        if y < _LEFT_THIRD_MAX:
            left_count += 1
        elif y >= _RIGHT_THIRD_MIN:
            right_count += 1
        else:
            center_count += 1

        # Shots
        if ev_type in _SHOT_TYPES:
            total_shots += 1
            shot_distances.append(_distance(x, y, *_GOAL_CENTER))
            if x > _BOX_X_THRESHOLD:
                shots_in_box += 1

        # Passes
        if ev_type in _PASS_TYPES:
            end_x = _safe_float(ev.get("end_x") or ev.get("endX"))
            end_y = _safe_float(ev.get("end_y") or ev.get("endY"))
            if end_x > 0 or end_y > 0:
                total_passes += 1
                pass_lengths.append(_distance(x, y, end_x, end_y))
                if (end_x - x) > _PROGRESSIVE_PASS_THRESHOLD:
                    progressive_passes += 1

        # Carries / take-ons
        if ev_type in _CARRY_TYPES:
            end_x = _safe_float(ev.get("end_x") or ev.get("endX"))
            end_y = _safe_float(ev.get("end_y") or ev.get("endY"))
            if end_x > 0 or end_y > 0:
                carry_distances.append(_distance(x, y, end_x, end_y))

        # Defensive actions
        if ev_type in _DEFENSIVE_TYPES:
            # Distance from own goal (0, 50)
            defensive_distances.append(_distance(x, y, 0.0, 50.0))

    # Build result
    features: Dict[str, float] = {}

    if shot_distances:
        features["avg_shot_distance"] = round(
            sum(shot_distances) / len(shot_distances), 2
        )
    if total_shots > 0:
        features["shots_inside_box_pct"] = round(
            shots_in_box / total_shots * 100, 2
        )
    if pass_lengths:
        features["avg_pass_length"] = round(
            sum(pass_lengths) / len(pass_lengths), 2
        )
    if total_passes > 0:
        features["progressive_pass_pct"] = round(
            progressive_passes / total_passes * 100, 2
        )
    if carry_distances:
        features["avg_carry_distance"] = round(
            sum(carry_distances) / len(carry_distances), 2
        )
    if total_touches > 0:
        features["touches_left_pct"] = round(
            left_count / total_touches * 100, 2
        )
        features["touches_center_pct"] = round(
            center_count / total_touches * 100, 2
        )
        features["touches_right_pct"] = round(
            right_count / total_touches * 100, 2
        )
    if defensive_distances:
        features["avg_defensive_distance"] = round(
            sum(defensive_distances) / len(defensive_distances), 2
        )

    return features
