"""StatsBomb open-data client — event-level spatial data.

Fetches x/y-coordinate events (shots, passes, carries, defensive actions)
from StatsBomb's free open-data catalogue via the ``statsbombpy`` package.

Coverage is limited to specific competitions (La Liga 2004-2020, FAWSL,
Women's World Cup, etc.).  All functions gracefully return empty results
when data is unavailable for a player or team.

StatsBomb pitch coordinates: 120 × 80, goal at x = 120.

All external data is routed through the project cache layer with a 7-day
TTL (open data is static).
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

from backend.data import cache

_log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

_CACHE_NAMESPACE = "statsbomb"
_CACHE_TTL = 86400 * 7  # 7 days — open data doesn't change

_GOAL_CENTER: Tuple[float, float] = (120.0, 40.0)
_BOX_X_THRESHOLD = 102.0  # 18-yard box starts at x ≈ 102
_PROGRESSIVE_PASS_THRESHOLD = 10.0  # metres toward goal
_MAX_COMPETITIONS_SCAN = 5  # cap when searching across all competitions

# Pitch-third boundaries (y-axis: 0–80)
_LEFT_THIRD_MAX = 80.0 / 3
_RIGHT_THIRD_MIN = 2 * 80.0 / 3


# ── helpers ──────────────────────────────────────────────────────────────────


def _import_sb():
    """Import statsbombpy at call time to avoid hard dependency."""
    try:
        from statsbombpy import sb
        # Suppress the NoAuthWarning emitted on every API call when using
        # open data without credentials — expected behaviour, not actionable.
        try:
            from statsbombpy.api_client import NoAuthWarning
            warnings.filterwarnings("ignore", category=NoAuthWarning)
        except ImportError:
            pass
        return sb
    except ImportError:
        _log.warning("statsbombpy is not installed — StatsBomb data unavailable")
        return None


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two pitch coordinates."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _safe_float(val: Any) -> float:
    """Convert to float, returning 0.0 on failure."""
    try:
        return float(val) if val is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def _matches_player(event_player: Any, target_name: str) -> bool:
    """Case-insensitive player name match with partial support."""
    if event_player is None:
        return False
    event_str = str(event_player).lower().strip()
    target_lower = target_name.lower().strip()
    return target_lower in event_str or event_str in target_lower


def _extract_location(event: dict, key: str = "location") -> Optional[Tuple[float, float]]:
    """Extract an (x, y) tuple from an event dict, returning None on failure."""
    loc = event.get(key)
    if loc is None or not isinstance(loc, (list, tuple)) or len(loc) < 2:
        return None
    try:
        return (float(loc[0]), float(loc[1]))
    except (ValueError, TypeError):
        return None


# ── public API ───────────────────────────────────────────────────────────────


def get_available_competitions() -> List[Dict[str, Any]]:
    """Return list of competitions with free StatsBomb data.

    Each dict has: competition_id, competition_name, season_id, season_name.
    Cached 7 days.
    """
    key = cache.make_key(_CACHE_NAMESPACE, "competitions")
    cached = cache.get(key, max_age=_CACHE_TTL)
    if cached is not None:
        return cached

    sb = _import_sb()
    if sb is None:
        return []

    try:
        df = sb.competitions()
    except Exception as exc:
        _log.warning("Failed to fetch StatsBomb competitions: %s", exc)
        return []

    if df is None or df.empty:
        return []

    results: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        results.append({
            "competition_id": int(row.get("competition_id", 0)),
            "competition_name": str(row.get("competition_name", "")),
            "season_id": int(row.get("season_id", 0)),
            "season_name": str(row.get("season_name", "")),
        })

    cache.set(key, results)
    _log.info("Cached %d StatsBomb competition-seasons", len(results))
    return results


def get_player_events(
    player_name: str,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Get all events for a player from StatsBomb free data.

    If *competition_id* / *season_id* are not given, searches across all
    free competitions (capped at :data:`_MAX_COMPETITIONS_SCAN`).

    Returns a list of event dicts with ``type``, ``location`` [x, y], and
    event-specific fields.  Returns ``[]`` if the player is not found or no
    data is available.  Cached 7 days.
    """
    name_key = player_name.lower().strip()
    comp_key = str(competition_id) if competition_id is not None else "all"
    season_key = str(season_id) if season_id is not None else "all"
    key = cache.make_key(_CACHE_NAMESPACE, "player_events", name_key, comp_key, season_key)
    cached = cache.get(key, max_age=_CACHE_TTL)
    if cached is not None:
        return cached

    sb = _import_sb()
    if sb is None:
        return []

    comp_seasons = _resolve_competition_seasons(sb, competition_id, season_id)
    if not comp_seasons:
        return []

    all_events: List[Dict[str, Any]] = []

    for cid, sid in comp_seasons:
        try:
            matches_df = sb.matches(competition_id=cid, season_id=sid)
        except Exception as exc:
            _log.debug("Failed to fetch matches for comp %s season %s: %s", cid, sid, exc)
            continue

        if matches_df is None or matches_df.empty:
            continue

        for _, match_row in matches_df.iterrows():
            match_id = int(match_row.get("match_id", 0))
            if match_id == 0:
                continue

            events_df = _get_match_events(sb, match_id)
            if events_df is None or events_df.empty:
                continue

            player_mask = events_df.get("player")
            if player_mask is None:
                continue

            for _, event_row in events_df.iterrows():
                if not _matches_player(event_row.get("player"), player_name):
                    continue
                event_dict = _event_row_to_dict(event_row, match_id)
                all_events.append(event_dict)

    cache.set(key, all_events)
    _log.info("Found %d events for '%s'", len(all_events), player_name)
    return all_events


def get_player_shots(
    player_name: str,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Extract shot events with spatial data.

    Returns list of dicts:
    ``{x, y, xg, outcome, body_part, technique, minute, match_id}``.
    Uses StatsBomb pitch coordinates (120 × 80).
    Returns ``[]`` if no shots found.
    """
    events = get_player_events(player_name, competition_id, season_id)
    shots: List[Dict[str, float]] = []

    for ev in events:
        if ev.get("type") != "Shot":
            continue
        loc = _extract_location(ev)
        if loc is None:
            continue
        shots.append({
            "x": loc[0],
            "y": loc[1],
            "xg": _safe_float(ev.get("shot_statsbomb_xg")),
            "outcome": str(ev.get("shot_outcome", "")),
            "body_part": str(ev.get("shot_body_part", "")),
            "technique": str(ev.get("shot_technique", "")),
            "minute": _safe_float(ev.get("minute")),
            "match_id": ev.get("match_id", 0),
        })

    return shots


def get_player_passes(
    player_name: str,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extract pass events with spatial data.

    Returns list of dicts:
    ``{start_x, start_y, end_x, end_y, outcome, length, angle, type,
    recipient, minute, match_id}``.
    Returns ``[]`` if no passes found.
    """
    events = get_player_events(player_name, competition_id, season_id)
    passes: List[Dict[str, Any]] = []

    for ev in events:
        if ev.get("type") != "Pass":
            continue
        start = _extract_location(ev)
        if start is None:
            continue
        end_loc = ev.get("pass_end_location")
        if isinstance(end_loc, (list, tuple)) and len(end_loc) >= 2:
            end_x, end_y = float(end_loc[0]), float(end_loc[1])
        else:
            end_x, end_y = 0.0, 0.0

        passes.append({
            "start_x": start[0],
            "start_y": start[1],
            "end_x": end_x,
            "end_y": end_y,
            "outcome": str(ev.get("pass_outcome", "Complete")),
            "length": _safe_float(ev.get("pass_length")),
            "angle": _safe_float(ev.get("pass_angle")),
            "type": str(ev.get("pass_type", "")),
            "recipient": str(ev.get("pass_recipient", "")),
            "minute": _safe_float(ev.get("minute")),
            "match_id": ev.get("match_id", 0),
        })

    return passes


def get_player_heatmap_data(
    player_name: str,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
) -> List[Tuple[float, float]]:
    """Extract all touch/action locations for heatmap rendering.

    Returns list of ``(x, y)`` tuples in StatsBomb coordinates.
    Returns ``[]`` if no data.
    """
    events = get_player_events(player_name, competition_id, season_id)
    points: List[Tuple[float, float]] = []

    for ev in events:
        loc = _extract_location(ev)
        if loc is not None:
            points.append(loc)

    return points


def compute_spatial_features(
    player_name: str,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
) -> Dict[str, float]:
    """Compute aggregate spatial features from event data.

    Returns dict with keys:
      - ``avg_shot_distance``: mean distance of shots from goal center
      - ``shots_inside_box_pct``: % of shots from inside the 18-yard box
      - ``avg_pass_length``: mean pass distance
      - ``progressive_pass_pct``: % of passes that move ball >10 m toward goal
      - ``avg_carry_distance``: mean distance of carries
      - ``touches_left_pct``, ``touches_center_pct``, ``touches_right_pct``:
        pitch-third distribution (y-axis)
      - ``avg_defensive_distance``: mean distance of defensive actions from
        own goal

    Returns ``{}`` if no data available.
    """
    events = get_player_events(player_name, competition_id, season_id)
    if not events:
        return {}

    # ── shots ────────────────────────────────────────────────────────────
    shot_distances: List[float] = []
    shots_in_box = 0
    total_shots = 0
    for ev in events:
        if ev.get("type") != "Shot":
            continue
        loc = _extract_location(ev)
        if loc is None:
            continue
        total_shots += 1
        shot_distances.append(_distance(loc[0], loc[1], *_GOAL_CENTER))
        if loc[0] > _BOX_X_THRESHOLD:
            shots_in_box += 1

    # ── passes ───────────────────────────────────────────────────────────
    pass_lengths: List[float] = []
    progressive_passes = 0
    total_passes = 0
    for ev in events:
        if ev.get("type") != "Pass":
            continue
        start = _extract_location(ev)
        if start is None:
            continue
        end_raw = ev.get("pass_end_location")
        if not isinstance(end_raw, (list, tuple)) or len(end_raw) < 2:
            continue
        end_x, end_y = float(end_raw[0]), float(end_raw[1])
        total_passes += 1
        pass_lengths.append(_distance(start[0], start[1], end_x, end_y))
        # Progressive = moves ball >10 m toward goal (higher x)
        if (end_x - start[0]) > _PROGRESSIVE_PASS_THRESHOLD:
            progressive_passes += 1

    # ── carries ──────────────────────────────────────────────────────────
    carry_distances: List[float] = []
    for ev in events:
        if ev.get("type") != "Carry":
            continue
        start = _extract_location(ev)
        if start is None:
            continue
        end_raw = ev.get("carry_end_location")
        if not isinstance(end_raw, (list, tuple)) or len(end_raw) < 2:
            continue
        end_x, end_y = float(end_raw[0]), float(end_raw[1])
        carry_distances.append(_distance(start[0], start[1], end_x, end_y))

    # ── touches (pitch-third distribution by y-axis) ─────────────────────
    left_count = 0
    center_count = 0
    right_count = 0
    total_touches = 0
    for ev in events:
        loc = _extract_location(ev)
        if loc is None:
            continue
        total_touches += 1
        y = loc[1]
        if y < _LEFT_THIRD_MAX:
            left_count += 1
        elif y >= _RIGHT_THIRD_MIN:
            right_count += 1
        else:
            center_count += 1

    # ── defensive actions ────────────────────────────────────────────────
    _DEFENSIVE_TYPES = {"Tackle", "Interception", "Block", "Clearance"}
    defensive_distances: List[float] = []
    for ev in events:
        if ev.get("type") not in _DEFENSIVE_TYPES:
            continue
        loc = _extract_location(ev)
        if loc is None:
            continue
        # Distance from own goal (0, 40)
        defensive_distances.append(_distance(loc[0], loc[1], 0.0, 40.0))

    # ── build result ─────────────────────────────────────────────────────
    features: Dict[str, float] = {}

    if shot_distances:
        features["avg_shot_distance"] = round(sum(shot_distances) / len(shot_distances), 2)
    if total_shots > 0:
        features["shots_inside_box_pct"] = round(shots_in_box / total_shots * 100, 2)
    if pass_lengths:
        features["avg_pass_length"] = round(sum(pass_lengths) / len(pass_lengths), 2)
    if total_passes > 0:
        features["progressive_pass_pct"] = round(progressive_passes / total_passes * 100, 2)
    if carry_distances:
        features["avg_carry_distance"] = round(sum(carry_distances) / len(carry_distances), 2)
    if total_touches > 0:
        features["touches_left_pct"] = round(left_count / total_touches * 100, 2)
        features["touches_center_pct"] = round(center_count / total_touches * 100, 2)
        features["touches_right_pct"] = round(right_count / total_touches * 100, 2)
    if defensive_distances:
        features["avg_defensive_distance"] = round(
            sum(defensive_distances) / len(defensive_distances), 2
        )

    return features


# ── internal helpers ─────────────────────────────────────────────────────────


def _resolve_competition_seasons(
    sb: Any,
    competition_id: Optional[int],
    season_id: Optional[int],
) -> List[Tuple[int, int]]:
    """Return (competition_id, season_id) pairs to scan."""
    if competition_id is not None and season_id is not None:
        return [(competition_id, season_id)]

    comps = get_available_competitions()
    if not comps:
        return []

    pairs: List[Tuple[int, int]] = []
    for c in comps:
        cid = c["competition_id"]
        sid = c["season_id"]
        if competition_id is not None and cid != competition_id:
            continue
        if season_id is not None and sid != season_id:
            continue
        pairs.append((cid, sid))
        if len(pairs) >= _MAX_COMPETITIONS_SCAN:
            break

    return pairs


def _get_match_events(sb: Any, match_id: int):
    """Fetch events for a single match, with cache."""
    key = cache.make_key(_CACHE_NAMESPACE, "match_events", str(match_id))
    cached = cache.get(key, max_age=_CACHE_TTL)
    if cached is not None:
        return cached

    try:
        events_df = sb.events(match_id=match_id)
    except Exception as exc:
        _log.debug("Failed to fetch events for match %s: %s", match_id, exc)
        return None

    if events_df is not None and not events_df.empty:
        cache.set(key, events_df)
    return events_df


def _event_row_to_dict(row: Any, match_id: int) -> Dict[str, Any]:
    """Convert a DataFrame row to a plain dict, preserving key fields."""
    d: Dict[str, Any] = {}
    try:
        row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        for k, v in row_dict.items():
            # Skip NaN/None values to keep dicts compact
            if v is None:
                continue
            try:
                import pandas as pd
                if pd.isna(v):
                    continue
            except (TypeError, ValueError):
                pass
            d[k] = v
    except Exception:
        pass
    d["match_id"] = match_id
    return d
