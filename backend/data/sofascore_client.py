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

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from backend.data import cache

_log = logging.getLogger(__name__)

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

# ── Metric category sets (paper-aligned) ─────────────────────────────────────
# Used by prediction fallbacks to apply different adjustment rates.

OFFENSIVE_METRICS: frozenset[str] = frozenset({
    "expected_goals", "expected_assists", "shots",
    "successful_dribbles", "successful_crosses",
    "touches_in_opposition_box", "chances_created",
})

DEFENSIVE_METRICS: frozenset[str] = frozenset({
    "clearances", "interceptions", "possession_won_final_3rd",
})

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
_MAX_RETRIES = 3  # total attempts for retryable errors
_RETRY_BASE_DELAY = 1.0  # seconds — doubles each attempt (1, 2, 4)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _get(path: str) -> Optional[dict]:
    """Execute a GET request against the Sofascore API with retry.

    Retries up to ``_MAX_RETRIES`` times with exponential backoff for
    transient HTTP errors (429 rate-limit, 5xx server errors).
    Returns the parsed JSON dict, or None on any permanent error.
    """
    url = f"{_BASE_URL}{path}"
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
            if resp.status_code in _RETRYABLE_STATUS_CODES:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)  # 1s, 2s, 4s
                _log.info(
                    "Sofascore %d on %s — retry %d/%d in %.1fs",
                    resp.status_code, path, attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            delay = _RETRY_BASE_DELAY * (2 ** attempt)
            _log.info("Sofascore connection error on %s — retry in %.1fs", path, delay)
            time.sleep(delay)
        except Exception:
            return None
    _log.warning("Sofascore request failed after %d retries: %s", _MAX_RETRIES, path)
    return None


# ── Public API ────────────────────────────────────────────────────────────────


def search_team(name: str) -> List[Dict[str, Any]]:
    """Search Sofascore for a team by name.

    Returns a list of dicts with ``id``, ``name``, and optional
    ``tournament_id``.
    """
    key = cache.make_key("sofascore_team_search", name.lower().strip())
    cached = cache.get(key, max_age=86400 * 7)
    if cached is not None:
        return cached

    raw = _get(f"/search/teams?q={requests.utils.quote(name)}&page=0")
    teams: list[dict] = []

    if isinstance(raw, dict):
        results = raw.get("results", [])
        for item in results:
            entity = item.get("entity") or item
            if not isinstance(entity, dict):
                continue
            team_id = entity.get("id")
            team_name = entity.get("name") or entity.get("shortName", "")
            if not team_id or not team_name:
                continue

            entry: dict[str, Any] = {"id": team_id, "name": team_name}

            tournament_id = _extract_unique_tournament_id(entity)
            if tournament_id:
                entry["tournament_id"] = tournament_id

            country = entity.get("country") or {}
            if isinstance(country, dict) and country.get("name"):
                entry["country"] = country["name"]

            teams.append(entry)

    cache.set(key, teams)
    return teams


def get_player_transfer_history(player_id: int) -> List[Dict[str, Any]]:
    """Fetch a player's transfer history from Sofascore.

    Returns a list of transfer dicts (most recent first), each with:
        - ``transfer_date``: ISO date string or None
        - ``from_team``: dict with ``id`` and ``name``
        - ``to_team``: dict with ``id`` and ``name``
        - ``type``: transfer type string (e.g. "transfer", "loan")
    """
    key = cache.make_key("sofascore_transfers", str(player_id))
    cached = cache.get(key, max_age=86400 * 7)
    if cached is not None:
        return cached

    raw = _get(f"/player/{player_id}/transfer-history")
    transfers: list[dict] = []

    if isinstance(raw, dict):
        entries = raw.get("transferHistory", [])
        for entry in entries:
            if not isinstance(entry, dict):
                continue

            from_team = entry.get("transferFrom") or {}
            to_team = entry.get("transferTo") or {}

            t: dict[str, Any] = {
                "transfer_date": _unix_to_iso(entry.get("transferDateTimestamp")),
                "from_team": {
                    "id": from_team.get("id"),
                    "name": from_team.get("name", ""),
                },
                "to_team": {
                    "id": to_team.get("id"),
                    "name": to_team.get("name", ""),
                },
                "type": entry.get("type") or entry.get("transferType", ""),
            }
            transfers.append(t)

    cache.set(key, transfers)
    return transfers


def get_league_player_stats(
    tournament_id: int,
    season_id: Optional[int] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Fetch aggregated player stats for an entire league/tournament season.

    Uses Sofascore's top-players endpoint to retrieve season stats for
    all players in the tournament.  Results are cached for 1 day.

    Parameters
    ----------
    tournament_id : int
        Sofascore unique-tournament ID.
    season_id : int, optional
        Specific season ID.  If ``None``, the current season is fetched.
    limit : int
        Maximum number of players to fetch (default 200).

    Returns
    -------
    list[dict] — each dict has ``id``, ``name``, ``team``, ``team_id``,
    ``position``, ``minutes_played``, ``per90``.
    """
    if season_id is None:
        season_id = _get_current_season_id(tournament_id)
    if season_id is None:
        return []

    key = cache.make_key(
        "sofascore_league_stats", str(tournament_id), str(season_id)
    )
    cached = cache.get(key, max_age=86400)
    if cached is not None:
        return cached

    # Sofascore provides a "top players" endpoint per tournament+season
    # We query multiple stat types to get the broadest player coverage
    players_map: Dict[int, Dict[str, Any]] = {}

    # Fetch multiple stat types to get broad coverage
    for stat_type in ("rating", "expectedGoals", "accuratePasses"):
        page = 1
        while page <= (limit // 100 + 1):
            raw = _get(
                f"/unique-tournament/{tournament_id}/season/{season_id}"
                f"/top-players/{stat_type}"
                f"?accumulation=total&order=desc&group=overall&page={page}"
            )
            if not isinstance(raw, dict):
                # Fall back to alternate endpoint format
                raw = _get(
                    f"/unique-tournament/{tournament_id}/season/{season_id}"
                    f"/statistics?limit=100&order=-{stat_type}"
                    f"&accumulation=total&group=summary&page={page}"
                )
                if not isinstance(raw, dict):
                    break
            # Response may use "results" or "players" depending on endpoint
            results = raw.get("results") or raw.get("players") or []
            if not results:
                break
            for item in results:
                if not isinstance(item, dict):
                    continue
                player_data = item.get("player") or {}
                pid = player_data.get("id")
                if pid is None or pid in players_map:
                    continue
                team_data = item.get("team") or player_data.get("team") or {}
                # Stats may be nested under "statistics" or flat on the item
                stats_dict = item.get("statistics") or {}
                if not isinstance(stats_dict, dict):
                    stats_dict = {}
                # Minutes: try statistics sub-dict first, then top-level
                mins_raw = stats_dict.get("minutesPlayed")
                if mins_raw is None:
                    mins_raw = item.get("minutesPlayed")
                minutes = int(mins_raw) if mins_raw is not None else 0
                # Merge item-level stat keys into stats_dict for _parse_stats
                merged_stats = dict(stats_dict)
                for k, v in item.items():
                    if k not in ("player", "team", "statistics") and k not in merged_stats:
                        merged_stats[k] = v
                per90 = _parse_stats(merged_stats, minutes)

                players_map[pid] = {
                    "id": pid,
                    "name": player_data.get("name") or player_data.get("shortName", ""),
                    "team": team_data.get("name", ""),
                    "team_id": team_data.get("id"),
                    "position": _map_position(
                        player_data.get("position") or ""
                    ),
                    "minutes_played": minutes,
                    "per90": per90,
                }
                if len(players_map) >= limit:
                    break
            if len(players_map) >= limit:
                break
            page += 1
        if len(players_map) >= limit:
            break

    result = list(players_map.values())
    cache.set(key, result)
    return result


def get_season_list(tournament_id: int) -> List[Dict[str, Any]]:
    """Return the list of available seasons for a tournament.

    Each item has ``id`` (season_id) and ``name`` (e.g. ``"2024/2025"``).
    Newest season first.
    """
    key = cache.make_key("sofascore_season_list", str(tournament_id))
    cached = cache.get(key, max_age=86400)
    if cached is not None:
        return cached

    raw = _get(f"/unique-tournament/{tournament_id}/seasons")
    if not isinstance(raw, dict):
        return []

    seasons = raw.get("seasons") or []
    result = [
        {"id": s.get("id"), "name": s.get("name", "")}
        for s in seasons
        if isinstance(s, dict) and s.get("id") is not None
    ]

    cache.set(key, result)
    return result


def get_player_stats_for_season(
    player_id: int,
    tournament_id: int,
    season_id: int,
) -> Dict[str, Any]:
    """Fetch player stats for a specific tournament + season combination.

    Unlike ``get_player_stats`` which auto-discovers the current season,
    this function targets an explicit season.
    """
    key = cache.make_key(
        "sofascore_player_season",
        str(player_id),
        str(tournament_id),
        str(season_id),
    )
    cached = cache.get(key, max_age=86400)
    if cached is not None:
        return cached

    # Get player profile for name/team
    profile_raw = _get(f"/player/{player_id}")
    result = _make_empty_result()

    if isinstance(profile_raw, dict):
        player_data = profile_raw.get("player") or profile_raw
        if isinstance(player_data, dict):
            result["name"] = player_data.get("name") or player_data.get("shortName", "")
            team_data = player_data.get("team") or {}
            if isinstance(team_data, dict):
                result["team"] = team_data.get("name", "")
                result["team_id"] = team_data.get("id")
            position_data = player_data.get("position") or {}
            result["position"] = _map_position(
                player_data.get("positionDescription", {}) or position_data
            )

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


def search_player(name: str) -> List[Dict[str, Any]]:
    """Search Sofascore for a player by name.

    Returns a list of dicts, each with at least ``id`` and ``name``.
    Also includes ``age``, ``nationality``, and ``team_name`` when
    available from the search response.
    Caches ``tournament_id`` and ``season_id`` per player for use
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

            # Age — Sofascore may return dateOfBirthTimestamp
            dob_ts = entity.get("dateOfBirthTimestamp")
            if dob_ts is not None:
                try:
                    from datetime import datetime, timezone
                    born = datetime.fromtimestamp(int(dob_ts), tz=timezone.utc)
                    now = datetime.now(tz=timezone.utc)
                    entry["age"] = now.year - born.year - (
                        (now.month, now.day) < (born.month, born.day)
                    )
                except (ValueError, TypeError, OSError):
                    pass

            # Nationality
            country = entity.get("country") or {}
            if isinstance(country, dict) and country.get("name"):
                entry["nationality"] = country["name"]

            # Extract and cache tournament/season info when available
            team = entity.get("team") or {}
            if isinstance(team, dict) and team.get("name"):
                entry["team_name"] = team["name"]
                entry["team_id"] = team.get("id")
            tournament_id = _extract_unique_tournament_id(team, entity)
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

            # Tournament — check multiple locations in Sofascore response
            tournament_id = _extract_unique_tournament_id(
                team_data, player_data, profile_raw,
            )
            if tournament_id:
                _cache_player_meta(player_id, tournament_id)

            # Position
            position_data = player_data.get("position") or {}
            result["position"] = _map_position(
                player_data.get("positionDescription", {}) or position_data
            )

    # Step 2 — Discover current tournament + season
    tournament_id = _get_cached_tournament_id(player_id)

    # Fallback: if no tournament_id found yet, try the team's tournaments endpoint
    if not tournament_id and result.get("team_id"):
        tournament_id = _discover_tournament_for_team(result["team_id"])
        if tournament_id:
            _cache_player_meta(player_id, tournament_id)

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


def _extract_unique_tournament_id(*dicts: Any) -> Optional[int]:
    """Extract the unique tournament ID from Sofascore response dicts.

    Sofascore nests tournament info in several places depending on endpoint:
      - ``team.tournament.uniqueTournament.id`` (common in profiles)
      - ``team.tournament.id`` (occasionally, but often the *tournament* id, not unique)
      - ``uniqueTournament.id`` (top-level on stats responses)
      - ``tournament.uniqueTournament.id``

    Checks multiple dicts (team_data, player_data, profile_raw) in order.
    Returns the first valid integer ID found, or None.
    """
    for d in dicts:
        if not isinstance(d, dict):
            continue
        # Direct uniqueTournament at this level
        ut = d.get("uniqueTournament")
        if isinstance(ut, dict) and ut.get("id"):
            return int(ut["id"])
        # Nested under tournament.uniqueTournament
        tournament = d.get("tournament")
        if isinstance(tournament, dict):
            ut2 = tournament.get("uniqueTournament")
            if isinstance(ut2, dict) and ut2.get("id"):
                return int(ut2["id"])
            # Fallback: tournament.id itself (may be the unique tournament ID
            # in some API versions / mock data)
            tid = tournament.get("id")
            if tid is not None:
                return int(tid)
    return None


def _discover_tournament_for_team(team_id: int) -> Optional[int]:
    """Fetch the primary unique tournament ID for a team via Sofascore API.

    Uses ``/team/{team_id}/unique-tournaments`` and picks the first
    domestic league tournament (highest userCount, non-international).
    Falls back to the first result if none match.
    """
    key = cache.make_key("sofascore_team_tournament", str(team_id))
    cached = cache.get(key, max_age=86400 * 7)
    if cached is not None:
        return cached

    raw = _get(f"/team/{team_id}/unique-tournaments")
    if not isinstance(raw, dict):
        return None

    tournaments = raw.get("uniqueTournaments") or []
    if not tournaments:
        return None

    # Prefer domestic league (non-international, highest userCount)
    best = None
    best_count = -1
    for t in tournaments:
        if not isinstance(t, dict):
            continue
        tid = t.get("id")
        if tid is None:
            continue
        # Skip international tournaments (Champions League=7, Europa=679, etc.)
        # Heuristic: domestic leagues have a "category" with a country
        cat = t.get("category") or {}
        is_domestic = cat.get("flag") is not None or cat.get("alpha2") is not None
        user_count = t.get("userCount") or 0
        if is_domestic and user_count > best_count:
            best = int(tid)
            best_count = user_count

    # Fallback to first tournament if no domestic league found
    if best is None:
        first = tournaments[0]
        if isinstance(first, dict) and first.get("id"):
            best = int(first["id"])

    if best is not None:
        cache.set(key, best)
    return best


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


def _unix_to_iso(ts: Any) -> Optional[str]:
    """Convert a Unix timestamp to ISO-8601 date string, or return None."""
    if ts is None:
        return None
    try:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, TypeError, OSError):
        return None


def _get_current_season_id(tournament_id: int) -> Optional[int]:
    """Return the current (most recent) season ID for a Sofascore tournament.

    Reuses the ``get_season_list`` cache when available so the same
    ``/seasons`` endpoint is not fetched twice.
    """
    key = cache.make_key("sofascore_seasons", str(tournament_id))
    cached = cache.get(key, max_age=86400)  # refresh daily
    if cached is not None:
        return cached

    # Try the season_list cache first (populated by get_season_list)
    seasons = get_season_list(tournament_id)
    if seasons:
        season_id = seasons[0].get("id")
        if season_id is not None:
            cache.set(key, season_id)
        return season_id

    return None
