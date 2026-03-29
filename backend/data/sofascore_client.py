"""Sofascore player stats client via direct HTTP API calls.

Sofascore returns raw totals; per-90 values are computed here as:
    per_90 = total / (minutes_played / 90)
Percentage stats (e.g. pass_completion_pct) are stored as-is.
All external calls are routed through backend.data.cache.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote as _url_quote

import requests as _stdlib_requests

# Prefer tls_requests for Cloudflare bypass; fall back to stdlib requests
# when tls_requests is not installed or its native TLS library fails to load.
try:
    import tls_requests as _tls_requests
except ImportError:
    _tls_requests = None

# Active HTTP module — may be swapped to _stdlib_requests at runtime if
# tls_requests raises OSError (native library download failure).
_http = _tls_requests if _tls_requests is not None else _stdlib_requests

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
    # Touches in box — Sofascore uses varying key names across seasons/endpoints
    "penaltyAreaTouches": "touches_in_opposition_box",
    "touchInBox": "touches_in_opposition_box",
    "touchesInOppositionBox": "touches_in_opposition_box",
    "touchesInPenaltyArea": "touches_in_opposition_box",
    "penAreaEntries": "touches_in_opposition_box",
    "penaltyAreaEntries": "touches_in_opposition_box",
    "boxTouches": "touches_in_opposition_box",
    "touchesInTheBox": "touches_in_opposition_box",
    "touchInPenaltyArea": "touches_in_opposition_box",
    "penAreaTouches": "touches_in_opposition_box",
    "totalTouchesInPenaltyArea": "touches_in_opposition_box",
    "totalTouchInBox": "touches_in_opposition_box",
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
    "totalDuelsWonPercentage": "duels_won_pct",
    # Aerial duels won % (percentage — kept as-is)
    "aerialDuelsWonPercentage": "aerial_duels_won_pct",
    "aerialDuelsWon%": "aerial_duels_won_pct",
    # Recoveries
    "ballRecovery": "recoveries",
    "recoveries": "recoveries",
    # Fouls won (drawn)
    "foulsDrawn": "fouls_won",
    "foulsWon": "fouls_won",
    "wasFouled": "fouls_won",
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
_RETRYABLE_STATUS_CODES = {403, 429, 500, 502, 503, 504}
_LEAGUE_STATS_INTER_REQUEST_DELAY = 0.5  # seconds between bulk API calls


def _get(path: str) -> Optional[dict]:
    """Execute a GET request against the Sofascore API with retry.

    Retries up to ``_MAX_RETRIES`` times with exponential backoff for
    transient HTTP errors (429 rate-limit, 5xx server errors).
    Returns the parsed JSON dict, or None on any permanent error.

    Uses ``tls_requests`` when available (Cloudflare bypass) and
    falls back to stdlib ``requests`` if the native TLS library is
    unavailable at runtime.
    """
    global _http
    url = f"{_BASE_URL}{path}"
    for attempt in range(_MAX_RETRIES):
        try:
            resp = _http.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
            if resp.status_code in _RETRYABLE_STATUS_CODES:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)  # 1s, 2s, 4s
                _log.info(
                    "Sofascore %d on %s — retry %d/%d in %.1fs",
                    resp.status_code, path, attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue
            # Non-retryable HTTP errors — return None immediately.
            # Explicit check avoids raise_for_status() whose exception
            # hierarchy differs between tls_requests and stdlib requests.
            if resp.status_code >= 400:
                _log.warning("Sofascore HTTP %d on %s", resp.status_code, path)
                return None
            return resp.json()
        except (ConnectionError, OSError) as exc:
            # tls_requests raises OSError when its native TLS library is
            # unavailable (e.g. on Streamlit Cloud).  Fall back to stdlib
            # requests and retry this attempt.
            if _http is not _stdlib_requests:
                _log.warning(
                    "tls_requests unavailable (%s), falling back to stdlib requests",
                    exc,
                )
                _http = _stdlib_requests
                continue
            delay = _RETRY_BASE_DELAY * (2 ** attempt)
            _log.info(
                "Sofascore connection error on %s — retry %d/%d in %.1fs (%s)",
                path, attempt + 1, _MAX_RETRIES, delay, exc,
            )
            time.sleep(delay)
        except Exception as exc:
            # If tls_requests raises a non-standard exception, fall back once.
            if _http is not _stdlib_requests:
                _log.warning(
                    "tls_requests error (%s), falling back to stdlib requests",
                    exc,
                )
                _http = _stdlib_requests
                continue
            _log.warning("Sofascore permanent error on %s: %s", path, exc)
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
    if cached:
        return cached

    raw = _get(f"/search/teams?q={_url_quote(name)}&page=0")
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


# Sofascore numeric transfer-type codes → human-readable labels
_TRANSFER_TYPE_MAP: Dict[int, str] = {
    1: "Transfer",
    2: "Loan",
    3: "Loan return",
    4: "Free transfer",
    5: "Swap",
}


def _normalize_transfer_type(raw: Any) -> str:
    """Convert a Sofascore transfer type code to a readable label."""
    if isinstance(raw, int):
        return _TRANSFER_TYPE_MAP.get(raw, "Unknown")
    if isinstance(raw, str):
        # Already a string — title-case it for consistency
        return raw.strip().title() if raw.strip() else "N/A"
    return "N/A"


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
    if cached:
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
                "type": _normalize_transfer_type(
                    entry.get("type") or entry.get("transferType", "")
                ),
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

    Uses standings, team rosters, and individual player statistics endpoints
    to collect season stats for all players in the tournament.
    Results are cached for 1 day.

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
    ``position``, ``age``, ``minutes_played``, ``per90``, ``rating``.
    """
    if season_id is None:
        season_id = _get_current_season_id(tournament_id)
    if season_id is None:
        return []

    key = cache.make_key(
        "sofascore_league_stats", str(tournament_id), str(season_id),
        str(limit),
    )
    cached = cache.get(key, max_age=86400)
    if cached:
        return cached

    # Step 1 — Get all teams from league standings
    standings_raw = _get(
        f"/unique-tournament/{tournament_id}/season/{season_id}/standings/total"
    )
    if not isinstance(standings_raw, dict):
        _log.warning(
            "get_league_player_stats: standings endpoint returned %s for tid=%d sid=%d — not caching",
            type(standings_raw).__name__, tournament_id, season_id,
        )
        return []

    standings = standings_raw.get("standings") or []
    teams: List[Dict[str, Any]] = []
    for group in standings:
        if not isinstance(group, dict):
            continue
        for row in group.get("rows") or []:
            if not isinstance(row, dict):
                continue
            team_data = row.get("team") or {}
            team_id = team_data.get("id")
            team_name = team_data.get("name", "")
            if team_id is not None:
                teams.append({"id": team_id, "name": team_name})

    if not teams:
        _log.debug(
            "get_league_player_stats: no teams found for tid=%d sid=%d — caching",
            tournament_id, season_id,
        )
        cache.set(key, [])
        return []

    # Step 2 & 3 — For each team, get roster and then individual stats
    players_map: Dict[int, Dict[str, Any]] = {}

    for idx, team_info in enumerate(teams):
        if len(players_map) >= limit:
            break

        team_id = team_info["id"]
        team_name = team_info["name"]

        # Step 2 — Get team roster (delay between teams to avoid rate-limiting)
        if idx > 0:
            time.sleep(_LEAGUE_STATS_INTER_REQUEST_DELAY)
        roster_raw = _get(f"/team/{team_id}/players")
        if not isinstance(roster_raw, dict):
            continue

        roster_players: List[Dict[str, Any]] = []
        for group in roster_raw.get("players") or []:
            if not isinstance(group, dict):
                continue
            # Sofascore wraps each player in a group with a 'player' key
            player_entry = group.get("player")
            if isinstance(player_entry, dict):
                roster_players.append(player_entry)
            else:
                # Alternate roster format: players nested under 'members' key
                for member in group.get("players") or group.get("members") or []:
                    if isinstance(member, dict):
                        p = member.get("player") or member
                        if isinstance(p, dict):
                            roster_players.append(p)

        # Step 3 — Fetch individual stats for each player
        for player_data in roster_players:
            if len(players_map) >= limit:
                break

            pid = player_data.get("id")
            if pid is None or pid in players_map:
                continue

            time.sleep(_LEAGUE_STATS_INTER_REQUEST_DELAY)
            stats_raw = _get(
                f"/player/{pid}/unique-tournament/{tournament_id}"
                f"/season/{season_id}/statistics/overall"
            )
            if not isinstance(stats_raw, dict):
                continue

            stats = stats_raw.get("statistics") or {}
            if not isinstance(stats, dict):
                continue

            minutes = int(stats.get("minutesPlayed") or 0)
            if minutes == 0:
                continue

            # Step 4 — Parse stats via existing _parse_stats
            per90 = _parse_stats(stats, minutes)

            # Age from dateOfBirthTimestamp
            dob_ts = player_data.get("dateOfBirthTimestamp")
            player_age = None
            if dob_ts is not None:
                try:
                    age_seconds = time.time() - int(dob_ts)
                    if age_seconds > 0:
                        player_age = int(age_seconds / (365.25 * 86400))
                except (ValueError, TypeError):
                    pass

            # Rating (Sofascore 0-10 scale)
            avg_rating = stats.get("rating")
            if avg_rating is not None:
                try:
                    avg_rating = float(avg_rating)
                except (ValueError, TypeError):
                    avg_rating = None

            # Step 5 — Same return dict format
            players_map[pid] = {
                "id": pid,
                "name": player_data.get("name") or player_data.get("shortName", ""),
                "team": team_name,
                "team_id": team_id,
                "position": _map_position(
                    player_data.get("position") or ""
                ),
                "age": player_age,
                "minutes_played": minutes,
                "per90": per90,
                "rating": avg_rating,
            }

    result = list(players_map.values())
    if not result:
        _log.warning(
            "get_league_player_stats: 0 players collected for tid=%d sid=%d "
            "(%d teams found) — not caching",
            tournament_id, season_id, len(teams),
        )
        return result
    cache.set(key, result)
    return result


def get_season_list(tournament_id: int) -> List[Dict[str, Any]]:
    """Return the list of available seasons for a tournament.

    Each item has ``id`` (season_id) and ``name`` (e.g. ``"2024/2025"``).
    Newest season first.
    """
    key = cache.make_key("sofascore_season_list", str(tournament_id))
    cached = cache.get(key, max_age=86400)
    if cached:
        return cached

    raw = _get(f"/unique-tournament/{tournament_id}/seasons")
    if not isinstance(raw, dict):
        _log.warning(
            "get_season_list(%d): API returned %s instead of dict — not caching",
            tournament_id,
            type(raw).__name__,
        )
        return []

    seasons = raw.get("seasons") or []
    result = [
        {"id": s.get("id"), "name": s.get("name", "")}
        for s in seasons
        if isinstance(s, dict) and s.get("id") is not None
    ]

    if not result:
        _log.warning(
            "get_season_list(%d): API returned 0 valid seasons (raw keys: %s) — not caching",
            tournament_id,
            list(raw.keys()),
        )
        return []

    cache.set(key, result)
    return result


def get_player_match_logs(
    player_id: int,
    tournament_id: int,
    season_id: int,
) -> List[Dict[str, Any]]:
    """Fetch per-match player stats for a specific tournament + season.

    Uses the Sofascore events endpoint to retrieve match-by-match data.
    Paginates from page 0 (most recent) upward until empty or page > 10.

    Returns a list of match dicts sorted by ``match_date`` **ascending**
    (oldest first), suitable for rolling window accumulation.  Each dict:
    ``match_id``, ``match_date``, ``minutes_played``,
    ``per90`` (dict of canonical metric -> float).

    Matches with ``minutes_played`` of 0 or None are excluded.
    If fewer than 3 valid matches are found, returns ``[]``.
    """
    key = cache.make_key(
        "sofascore", "match_logs",
        str(player_id), str(tournament_id), str(season_id),
    )
    cached = cache.get(key, max_age=86400 * 7)
    if cached:
        return cached

    matches: List[Dict[str, Any]] = []
    max_page = 10  # safety ceiling

    for page in range(max_page + 1):
        raw = _get(
            f"/player/{player_id}/unique-tournament/{tournament_id}"
            f"/season/{season_id}/events/last/{page}"
        )
        if not isinstance(raw, dict):
            break

        events = raw.get("events") or []
        if not events:
            break

        for event in events:
            if not isinstance(event, dict):
                continue

            match_id = event.get("id")
            # Extract date from startTimestamp
            start_ts = event.get("startTimestamp")
            match_date = _unix_to_iso(start_ts)

            # Player statistics may be nested under "statistics" or "playerStatistics"
            stats_container = event.get("statistics") or event.get("playerStatistics") or {}
            if not isinstance(stats_container, dict):
                stats_container = {}

            minutes_played_raw = stats_container.get("minutesPlayed")
            if minutes_played_raw is None:
                # Try alternate locations
                minutes_played_raw = event.get("minutesPlayed")
            if minutes_played_raw is None or int(minutes_played_raw) <= 0:
                continue
            minutes_played = int(minutes_played_raw)

            per90 = _parse_stats(stats_container, minutes_played)

            matches.append({
                "match_id": match_id,
                "match_date": match_date or "",
                "minutes_played": minutes_played,
                "per90": {m: per90.get(m) for m in CORE_METRICS},
            })

    # Sort by match_date ascending (oldest first)
    matches.sort(key=lambda m: m.get("match_date", ""))

    # Fewer than 3 valid matches → unreliable data
    if len(matches) < 3:
        result: List[Dict[str, Any]] = []
    else:
        result = matches

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
    if cached:
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
    if cached:
        return cached

    raw = _get(f"/search/players?q={_url_quote(name)}&page=0")
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
    if cached:
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

            # Age — compute from dateOfBirthTimestamp if available
            dob_ts = player_data.get("dateOfBirthTimestamp")
            if dob_ts is not None:
                try:
                    age_seconds = time.time() - int(dob_ts)
                    if age_seconds > 0:
                        result["age"] = int(age_seconds / (365.25 * 86400))
                except (ValueError, TypeError):
                    pass

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
            # Extract average match rating (Sofascore 0-10 scale)
            avg_rating = stats.get("rating")
            if avg_rating is not None:
                try:
                    result["rating"] = float(avg_rating)
                except (ValueError, TypeError):
                    pass
    else:
        result["raw"] = {}

    # Step 4 — Multi-tournament fallback: if the primary tournament returned
    # 0 minutes, try ALL tournaments the player's team participates in and
    # use the one with the most minutes. This fixes players like Kroupi who
    # have significant minutes across cup/European competitions but 0 in the
    # primary domestic league season.
    if result["minutes_played"] == 0 and result.get("team_id"):
        best = _try_all_tournaments_for_player(
            player_id, result["team_id"], tournament_id,
        )
        if best is not None:
            best_stats, best_tid, best_sid = best
            b_stats = best_stats.get("statistics") or {}
            if isinstance(b_stats, dict):
                mins = int(b_stats.get("minutesPlayed") or 0)
                if mins > result["minutes_played"]:
                    result["minutes_played"] = mins
                    result["appearances"] = int(
                        b_stats.get("appearances")
                        or b_stats.get("matchesStarted")
                        or 0
                    )
                    result["per90"] = _parse_stats(b_stats, mins)
                    result["raw"] = best_stats
                    avg_rating = b_stats.get("rating")
                    if avg_rating is not None:
                        try:
                            result["rating"] = float(avg_rating)
                        except (ValueError, TypeError):
                            pass
                    # Cache the tournament where data was found, so
                    # subsequent calls and season selectors default to
                    # this tournament (note: the UI season selector
                    # currently only shows seasons for one tournament).
                    _cache_player_meta(player_id, best_tid)

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
    if cached:
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


# ── Position categories ──────────────────────────────────────────────────────

_POSITION_CATEGORIES: Dict[str, str] = {
    # Forward / Attacker variants
    "forward": "Forward", "forwards": "Forward", "attacker": "Forward",
    "attackers": "Forward", "striker": "Forward", "centre-forward": "Forward",
    "center forward": "Forward", "cf": "Forward", "st": "Forward",
    "right winger": "Forward", "left winger": "Forward", "winger": "Forward",
    "rw": "Forward", "lw": "Forward", "wing": "Forward",
    "f": "Forward",  # Sofascore single-letter code
    # Midfielder variants
    "midfielder": "Midfielder", "midfielders": "Midfielder",
    "central midfielder": "Midfielder", "attacking midfielder": "Midfielder",
    "defensive midfielder": "Midfielder", "cm": "Midfielder",
    "am": "Midfielder", "dm": "Midfielder", "cam": "Midfielder",
    "cdm": "Midfielder", "rm": "Midfielder", "lm": "Midfielder",
    "m": "Midfielder",  # Sofascore single-letter code
    # Defender variants
    "defender": "Defender", "defenders": "Defender",
    "centre-back": "Defender", "center back": "Defender",
    "right back": "Defender", "left back": "Defender",
    "right-back": "Defender", "left-back": "Defender",
    "cb": "Defender", "rb": "Defender", "lb": "Defender",
    "rwb": "Defender", "lwb": "Defender",
    "d": "Defender",  # Sofascore single-letter code
    # Goalkeeper variants
    "goalkeeper": "Goalkeeper", "goalkeepers": "Goalkeeper", "gk": "Goalkeeper",
    "g": "Goalkeeper",  # Sofascore single-letter code
}


def normalize_position(position: str) -> str:
    """Normalize a position string to one of: Forward, Midfielder, Defender, Goalkeeper."""
    return _POSITION_CATEGORIES.get(position.strip().lower(), "Unknown")


def get_team_position_averages(
    team_id: int,
    target_position: str,
    max_players: int = 8,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Compute average per-90 stats for players in a position at a team.

    Fetches the team squad, filters by position category, retrieves
    individual stats for matching players, and returns the mean per-90
    for each core metric.  This captures the team's tactical style for
    that position (paper Section 2.3: team-position features).

    Parameters
    ----------
    team_id : int
        Sofascore team ID.
    target_position : str
        Position to match (e.g. "Forward", "Right Winger", "Striker").
    max_players : int
        Cap on number of individual player stats to fetch.

    Returns
    -------
    (avg_per90, player_data_list)
        avg_per90: dict mapping metric -> average per-90 across position.
        player_data_list: list of dicts with ``per90``, ``position``, ``name``.
    """
    cache_key = cache.make_key(
        "team_pos_avg", str(team_id), normalize_position(target_position),
    )
    cached = cache.get(cache_key, max_age=86400)
    if cached:
        return cached  # type: ignore[return-value]

    target_cat = normalize_position(target_position)
    squad = get_team_players_stats(team_id)

    # Filter squad members whose position matches the same category
    matching = [
        p for p in squad
        if normalize_position(p.get("position", "")) == target_cat
    ]
    if not matching:
        # No exact position match — return empty averages rather than
        # mixing unrelated positions which would produce misleading data.
        import logging
        logging.getLogger(__name__).info(
            "No %s players found for team %s; returning empty position averages",
            target_cat, team_id,
        )
        empty_avg: Dict[str, float] = {m: 0.0 for m in CORE_METRICS}
        result_empty = (empty_avg, [])
        cache.set(cache_key, result_empty)
        return result_empty

    # Fetch individual stats (limited to avoid too many API calls)
    player_data: List[Dict[str, Any]] = []
    for p in matching[:max_players]:
        pid = p.get("id")
        if not pid:
            continue
        try:
            stats = get_player_stats(pid)
            if stats.get("per90"):
                player_data.append(stats)
        except Exception:
            continue

    # Compute average per-90 across matching players
    avg_per90: Dict[str, float] = {}
    for m in CORE_METRICS:
        values = []
        for pd_item in player_data:
            v = pd_item.get("per90", {}).get(m)
            if v is not None:
                values.append(v)
        avg_per90[m] = sum(values) / len(values) if values else 0.0

    result = (avg_per90, player_data)
    cache.set(cache_key, result)
    return result


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


def discover_tournament_for_team(team_id: int) -> Optional[int]:
    """Public wrapper — see :func:`_discover_tournament_for_team`."""
    return _discover_tournament_for_team(team_id)


# Known international club competition tournament IDs on Sofascore.
# These must never be returned as a team's "domestic" league.
_INTERNATIONAL_TOURNAMENT_IDS: frozenset = frozenset({
    7,    # UEFA Champions League
    679,  # UEFA Europa League
    73,   # UEFA Europa Conference League
    384,  # UEFA Super Cup
    498,  # FIFA Club World Cup
    480,  # Copa Libertadores
    133,  # Copa Sudamericana
    851,  # AFC Champions League
})

# Continental / international alpha2 codes used by Sofascore that do NOT
# represent a real country.  Champions League uses "EU", for example.
_NON_COUNTRY_ALPHA2: frozenset = frozenset({"EU", "INT", "WW"})


def _discover_tournament_for_team(team_id: int) -> Optional[int]:
    """Fetch the primary unique tournament ID for a team via Sofascore API.

    Uses ``/team/{team_id}/unique-tournaments`` and picks the first
    domestic league tournament (highest userCount, non-international).
    Falls back to the first result if none match.
    """
    key = cache.make_key("sofascore_team_tournament", str(team_id))
    cached = cache.get(key, max_age=86400 * 7)
    if cached:
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

        # Skip known international club competitions by ID
        if int(tid) in _INTERNATIONAL_TOURNAMENT_IDS:
            continue

        # Skip tournaments whose category uses a continental alpha2 code
        # (e.g. Champions League has alpha2="EU", not a real country)
        cat = t.get("category") or {}
        alpha2 = cat.get("alpha2") or ""
        if alpha2 in _NON_COUNTRY_ALPHA2:
            continue

        is_domestic = bool(alpha2)
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


def _try_all_tournaments_for_player(
    player_id: int,
    team_id: int,
    already_tried_tid: Optional[int] = None,
) -> Optional[tuple]:
    """Try all tournaments for a team and return stats from the one with most minutes.

    When the primary domestic tournament returns 0 minutes, this function
    iterates through every tournament the team participates in (cups,
    European competitions, secondary divisions) to find one where the
    player actually has data.

    Returns ``(stats_raw, tournament_id, season_id)`` for the tournament
    with the most minutes, or ``None`` if no tournament yields data.
    """
    raw = _get(f"/team/{team_id}/unique-tournaments")
    if not isinstance(raw, dict):
        return None

    tournaments = raw.get("uniqueTournaments") or []
    if not tournaments:
        return None

    best_mins = 0
    best_result: Optional[tuple] = None

    for t in tournaments:
        if not isinstance(t, dict):
            continue
        tid = t.get("id")
        if tid is None:
            continue
        tid = int(tid)
        if tid == already_tried_tid:
            continue  # already fetched this one

        sid = _get_current_season_id(tid)
        if sid is None:
            continue

        stats_raw = _get(
            f"/player/{player_id}/unique-tournament/{tid}"
            f"/season/{sid}/statistics/overall"
        )
        if not isinstance(stats_raw, dict):
            continue

        stats = stats_raw.get("statistics") or {}
        mins = int(stats.get("minutesPlayed") or 0)
        if mins > best_mins:
            best_mins = mins
            best_result = (stats_raw, tid, sid)

    return best_result


def _make_empty_result() -> Dict[str, Any]:
    """Return a blank result dict matching the public API shape."""
    return {
        "name": "",
        "team": "",
        "team_id": None,
        "position": "",
        "age": None,
        "minutes_played": 0,
        "appearances": 0,
        "per90": {m: None for m in ALL_METRICS},
        "rating": None,
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

    # Fallback for touches_in_opposition_box: Sofascore may not always
    # provide this stat directly.  When missing, estimate from total
    # touches using a position-based ratio (attackers ~15-20% of touches
    # are in the box, midfielders ~8-12%, defenders ~3-5%).
    #
    # Estimation constants (educated estimates calibrated to paper case studies):
    #   BOX_TOUCHES_PER_SHOT — each shot implies ~2.5 box touches on average
    #     (typical attackers have ~2-3x more box touches than shots per game)
    #   MAX_BOX_TOUCH_RATIO  — cap at 30% of total touches (realistic upper bound
    #     even for elite strikers; most attackers are 15-25%)
    #   DEFAULT_BOX_RATIO    — generic fallback when no shots data available
    #     (~10% is typical for midfielders/mixed positions)
    _BOX_TOUCHES_PER_SHOT = 2.5
    _MAX_BOX_TOUCH_RATIO = 0.30
    _DEFAULT_BOX_RATIO = 0.10

    if per90.get("touches_in_opposition_box") is None and per90.get("touches") is not None:
        total_touches = per90["touches"]
        if total_touches is not None and total_touches > 0:
            shots = per90.get("shots")
            if shots is not None and shots > 0:
                # Players who shoot more tend to be in the box more
                estimated_box_touches = round(shots * _BOX_TOUCHES_PER_SHOT, 4)
                per90["touches_in_opposition_box"] = min(
                    estimated_box_touches, total_touches * _MAX_BOX_TOUCH_RATIO
                )
            else:
                per90["touches_in_opposition_box"] = round(
                    total_touches * _DEFAULT_BOX_RATIO, 4
                )

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


def get_cached_tournament_id(player_id: int) -> Optional[int]:
    """Public wrapper — see :func:`_get_cached_tournament_id`."""
    return _get_cached_tournament_id(player_id)


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
    if cached:
        return cached

    # Try the season_list cache first (populated by get_season_list)
    seasons = get_season_list(tournament_id)
    if seasons:
        season_id = seasons[0].get("id")
        if season_id is not None:
            cache.set(key, season_id)
        return season_id

    return None
