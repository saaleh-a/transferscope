"""Sofascore player stats client via direct HTTP API calls.

Replaces mobfot/FotMob as the player stats source.
Public API is identical to fotmob_client to avoid cascading changes
in the frontend and feature pipeline.

Sofascore returns raw totals; per-90 values are computed here as:
    per_90 = total / (minutes_played / 90)
Percentage stats (e.g. pass_completion_pct) are stored as-is.
All external calls are routed through backend.data.cache.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from backend.data import cache

# ── Metric definitions (unchanged — canonical names are source-agnostic) ─────

CORE_METRICS: list[str] = [
    "expected_goals",
    "expected_assists",
    "shots",
    "successful_dribbles",
    "successful_crosses",
    "touches_in_opposition_box",
    "successful_passes",
    "pass_completion_pct",
    "accurate_long_balls",
    "chances_created",
    "clearances",
    "interceptions",
    "possession_won_final_3rd",
]

ADDITIONAL_METRICS: list[str] = [
    "xg_on_target",
    "non_penalty_xg",
    "dispossessed",
    "duels_won_pct",
    "aerial_duels_won_pct",
    "recoveries",
    "fouls_won",
    "touches",
    "goals_conceded_on_pitch",
    "xg_against_on_pitch",
]

ALL_METRICS: list[str] = CORE_METRICS + ADDITIONAL_METRICS

# ── Sofascore stat key → canonical name ──────────────────────────────────────
# Sofascore returns a flat dict of raw totals.  Multiple aliases are listed
# to guard against minor API key variations across seasons/versions.
_SOFASCORE_KEY_MAP: dict[str, str] = {
    # xG / xA
    "expectedGoals": "expected_goals",
    "xG": "expected_goals",
    # xA
    "expectedAssists": "expected_assists",
    "xA": "expected_assists",
    # Shots (total)
    "shots": "shots",
    "totalShots": "shots",
    "shotAttempts": "shots",
    # Dribbles
    "successfulDribbles": "successful_dribbles",
    "dribbles": "successful_dribbles",
    # Crosses
    "accurateCrosses": "successful_crosses",
    "crossesAccurate": "successful_crosses",
    # Touches in box
    "penaltyAreaTouches": "touches_in_opposition_box",
    "touchInBox": "touches_in_opposition_box",
    "touchesInOppositionBox": "touches_in_opposition_box",
    # Passes
    "accuratePasses": "successful_passes",
    "passesAccurate": "successful_passes",
    # Pass completion % (percentage — kept as-is, not converted to per-90)
    "accuratePassesPercentage": "pass_completion_pct",
    "passAccuracy": "pass_completion_pct",
    "passAccuracyPercentage": "pass_completion_pct",
    # Long balls
    "accurateLongBalls": "accurate_long_balls",
    "longBallsAccurate": "accurate_long_balls",
    # Chances created / key passes
    "keyPasses": "chances_created",
    "bigChancesCreated": "chances_created",
    "chancesCreated": "chances_created",
    # Defensive — own third (clearances)
    "clearances": "clearances",
    # Defensive — mid third (interceptions)
    "interceptions": "interceptions",
    # Defensive — att third (won tackles as proxy for possession won final 3rd)
    "wonTackles": "possession_won_final_3rd",
    "tacklesWon": "possession_won_final_3rd",
    "successfulTackles": "possession_won_final_3rd",
    # Additional metrics
    "expectedGoalsOnTarget": "xg_on_target",
    "xGOT": "xg_on_target",
    "expectedGoalsNoPenalty": "non_penalty_xg",
    "nonPenaltyXg": "non_penalty_xg",
    "npxG": "non_penalty_xg",
    "dispossessed": "dispossessed",
    # Duels won % (percentage — kept as-is)
    "duelsWonPercentage": "duels_won_pct",
    "duelsWon%": "duels_won_pct",
    # Aerial duels won % (percentage — kept as-is)
    "aerialDuelsWonPercentage": "aerial_duels_won_pct",
    "aerialDuelsWon%": "aerial_duels_won_pct",
    # Recoveries
    "ballRecovery": "recoveries",
    "recoveries": "recoveries",
    # Fouls won (drawn)
    "foulsDrawn": "fouls_won",
    "foulsWon": "fouls_won",
    # Touches
    "touches": "touches",
    # Goals conceded while on pitch
    "goalsConceded": "goals_conceded_on_pitch",
    "goalsConcededOnPitch": "goals_conceded_on_pitch",
    # xG against while on pitch (often unavailable in Sofascore — returns None)
    "xGAgainst": "xg_against_on_pitch",
    "expectedGoalsAgainst": "xg_against_on_pitch",
}

# Metrics that are percentages and must NOT be divided by minutes
_PERCENTAGE_METRICS: frozenset[str] = frozenset(
    ["pass_completion_pct", "duels_won_pct", "aerial_duels_won_pct"]
)

_BASE_URL = "https://api.sofascore.com/api/v1"
_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.sofascore.com",
    "Referer": "https://www.sofascore.com/",
}

_REQUEST_TIMEOUT = 10  # seconds


def _get(path: str) -> Optional[dict]:
    """Execute a GET request against the Sofascore API.

    Returns the parsed JSON dict, or None on any error.
    """
    url = f"{_BASE_URL}{path}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────


def search_player(name: str) -> List[Dict[str, Any]]:
    """Search Sofascore for a player by name.

    Returns a list of dicts, each with at least ``id`` and ``name``.
    Also caches ``tournament_id`` and ``season_id`` per player for use
    by ``get_player_stats``.
    """
    key = cache.make_key("sofascore_search", name.lower().strip())
    cached = cache.get(key, max_age=86400 * 7)
    if cached is not None:
        return cached

    raw = _get(f"/search/players?q={requests.utils.quote(name)}&page=0")
    players: list[dict] = []

    if isinstance(raw, dict):
        results = raw.get("results", [])
        for item in results:
            entity = item.get("entity") or item
            if not isinstance(entity, dict):
                continue
            player_id = entity.get("id")
            player_name = entity.get("name") or entity.get("shortName", "")
            if not player_id or not player_name:
                continue

            entry: dict[str, Any] = {"id": player_id, "name": player_name}

            # Extract and cache tournament/season info when available
            team = entity.get("team") or {}
            tournament = team.get("tournament") or {}
            tournament_id = tournament.get("id")
            if tournament_id:
                entry["tournament_id"] = tournament_id
                # Stash meta for get_player_stats fast-path
                _cache_player_meta(player_id, tournament_id)

            players.append(entry)

    cache.set(key, players)
    return players


def get_player_stats(
    player_id: int,
    season: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch full player stats from Sofascore and return structured data.

    Parameters
    ----------
    player_id : int
        Sofascore player ID.
    season : str, optional
        Season string like ``"2024/2025"`` (currently unused — always
        fetches the current season for the player's league).

    Returns
    -------
    dict with keys:
        - ``name``: player display name
        - ``team``: current team name
        - ``team_id``: current team Sofascore ID
        - ``position``: primary position label
        - ``minutes_played``: total minutes this season
        - ``appearances``: number of appearances
        - ``per90``: dict mapping each of the 23 metrics to its per-90 value
          (``None`` for unavailable metrics)
        - ``raw``: the original Sofascore statistics JSON
    """
    key = cache.make_key("sofascore_player", str(player_id), season or "current")
    cached = cache.get(key, max_age=86400)
    if cached is not None:
        return cached

    # Step 1 — Get player profile to resolve team + tournament
    profile_raw = _get(f"/player/{player_id}")
    result = _make_empty_result()

    if isinstance(profile_raw, dict):
        player_data = profile_raw.get("player") or profile_raw
        if isinstance(player_data, dict):
            result["name"] = player_data.get("name") or player_data.get("shortName", "")

            # Team
            team_data = player_data.get("team") or {}
            if isinstance(team_data, dict):
                result["team"] = team_data.get("name", "")
                result["team_id"] = team_data.get("id")

                # Tournament — stash for future calls
                tournament = team_data.get("tournament") or {}
                tournament_id = tournament.get("id")
                if tournament_id:
                    _cache_player_meta(player_id, tournament_id)

            # Position
            position_data = player_data.get("position") or {}
            result["position"] = _map_position(
                player_data.get("positionDescription", {}) or position_data
            )

    # Step 2 — Discover current tournament + season
    tournament_id = _get_cached_tournament_id(player_id)
    season_id = None
    if tournament_id:
        season_id = _get_current_season_id(tournament_id)

    # Step 3 — Fetch statistics if we have the required IDs
    stats_raw: Optional[dict] = None
    if tournament_id and season_id:
        stats_raw = _get(
            f"/player/{player_id}/unique-tournament/{tournament_id}"
            f"/season/{season_id}/statistics/overall"
        )

    if isinstance(stats_raw, dict):
        stats = stats_raw.get("statistics") or {}
        if isinstance(stats, dict):
            result["minutes_played"] = int(stats.get("minutesPlayed") or 0)
            result["appearances"] = int(
                stats.get("appearances") or stats.get("matchesStarted") or 0
            )
            result["per90"] = _parse_stats(stats, result["minutes_played"])
            result["raw"] = stats_raw
    else:
        result["raw"] = {}

    cache.set(key, result)
    return result


def get_team_players_stats(
    team_id: int,
    season: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch team squad and return a list of basic player entries.

    Returns a list of dicts with at least ``id``, ``name``, and ``position``.
    Detailed stats are fetched separately via ``get_player_stats``.
    """
    key = cache.make_key("sofascore_team", str(team_id), season or "current")
    cached = cache.get(key, max_age=86400)
    if cached is not None:
        return cached

    raw = _get(f"/team/{team_id}/players")
    players: list[dict] = []

    if isinstance(raw, dict):
        for group in raw.get("players", []):
            if not isinstance(group, dict):
                continue
            group_name = group.get("name") or group.get("title", "Unknown")
            for member in group.get("players") or group.get("members") or []:
                if not isinstance(member, dict):
                    continue
                player_id = member.get("id")
                player_name = member.get("name") or member.get("shortName", "")
                if player_id and player_name:
                    players.append(
                        {
                            "id": player_id,
                            "name": player_name,
                            "position": group_name,
                        }
                    )

    cache.set(key, players)
    return players


# ── Internal helpers ──────────────────────────────────────────────────────────


def _make_empty_result() -> Dict[str, Any]:
    """Return a blank result dict matching the public API shape."""
    return {
        "name": "",
        "team": "",
        "team_id": None,
        "position": "",
        "minutes_played": 0,
        "appearances": 0,
        "per90": {m: None for m in ALL_METRICS},
        "raw": {},
    }


def _parse_stats(stats: dict, minutes_played: int) -> Dict[str, Optional[float]]:
    """Map a Sofascore statistics dict to our canonical per-90 values.

    Raw totals are divided by (minutes_played / 90).
    Percentage metrics are stored as-is.
    """
    per90: Dict[str, Optional[float]] = {m: None for m in ALL_METRICS}
    nineties = minutes_played / 90.0 if minutes_played > 0 else None

    for raw_key, value in stats.items():
        canonical = _SOFASCORE_KEY_MAP.get(raw_key)
        if canonical is None or canonical not in per90:
            continue
        if value is None:
            continue
        try:
            fval = float(value)
        except (ValueError, TypeError):
            continue

        if canonical in _PERCENTAGE_METRICS:
            # Percentages stored directly
            per90[canonical] = fval
        elif nineties and nineties > 0:
            per90[canonical] = round(fval / nineties, 4)
        # If no minutes, leave as None (data unavailable)

    return per90


def _map_position(position_data: Any) -> str:
    """Extract a human-readable position label from Sofascore position data."""
    if isinstance(position_data, str):
        return position_data
    if isinstance(position_data, dict):
        # Try common Sofascore position description shapes
        for key in ("primaryPosition", "position", "name"):
            val = position_data.get(key)
            if val:
                if isinstance(val, dict):
                    return val.get("name") or val.get("label") or ""
                return str(val)
    return ""


def _cache_player_meta(player_id: int, tournament_id: int) -> None:
    """Cache the mapping of player_id → tournament_id."""
    meta_key = cache.make_key("sofascore_player_meta", str(player_id))
    existing = cache.get(meta_key, max_age=86400 * 30)
    if existing is None:
        cache.set(meta_key, {"tournament_id": tournament_id})


def _get_cached_tournament_id(player_id: int) -> Optional[int]:
    """Look up cached tournament_id for a player."""
    meta_key = cache.make_key("sofascore_player_meta", str(player_id))
    meta = cache.get(meta_key, max_age=86400 * 30)
    if isinstance(meta, dict):
        return meta.get("tournament_id")
    return None


def _get_current_season_id(tournament_id: int) -> Optional[int]:
    """Return the current (most recent) season ID for a Sofascore tournament."""
    key = cache.make_key("sofascore_seasons", str(tournament_id))
    cached = cache.get(key, max_age=86400)  # refresh daily
    if cached is not None:
        return cached

    raw = _get(f"/unique-tournament/{tournament_id}/seasons")
    if not isinstance(raw, dict):
        return None

    seasons = raw.get("seasons") or []
    if not seasons:
        return None

    # Sofascore returns seasons newest-first
    season_id = seasons[0].get("id")
    if season_id is not None:
        cache.set(key, season_id)
    return season_id
