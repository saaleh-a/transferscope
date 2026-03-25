"""FotMob API wrapper via mobfot.

Pulls player season stats for a given player ID and season.
Returns all 23 metrics from CLAUDE.md as per-90 values.
Every response is cached via backend.data.cache.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mobfot import MobFot

from backend.data import cache

# ── Metric definitions ──────────────────────────────────────────────────────
# 13 core metrics (paper) + 10 additional metrics
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

# FotMob stat key -> our canonical name mapping.
# FotMob nests stats under category groups; this maps the *stat key* strings
# that appear in the JSON payload to our internal names.
_FOTMOB_KEY_MAP: dict[str, str] = {
    "Expected goals (xG)": "expected_goals",
    "xG": "expected_goals",
    "Expected assists (xA)": "expected_assists",
    "xA": "expected_assists",
    "Total shots": "shots",
    "Shots": "shots",
    "Successful dribbles": "successful_dribbles",
    "Dribbles succeeded": "successful_dribbles",
    "Accurate crosses": "successful_crosses",
    "Crosses (accurate)": "successful_crosses",
    "Touches in opposition box": "touches_in_opposition_box",
    "Touches in opp. box": "touches_in_opposition_box",
    "Accurate passes": "successful_passes",
    "Passes (accurate)": "successful_passes",
    "Pass accuracy": "pass_completion_pct",
    "Pass completion": "pass_completion_pct",
    "Accurate long balls": "accurate_long_balls",
    "Long balls (accurate)": "accurate_long_balls",
    "Chances created": "chances_created",
    "Big chances created": "chances_created",
    "Clearances": "clearances",
    "Interceptions": "interceptions",
    "Possession won final 3rd": "possession_won_final_3rd",
    "Poss. won final 3rd": "possession_won_final_3rd",
    "xGOT": "xg_on_target",
    "Expected goals on target (xGOT)": "xg_on_target",
    "Non-penalty xG": "non_penalty_xg",
    "Non penalty xG": "non_penalty_xg",
    "npxG": "non_penalty_xg",
    "Dispossessed": "dispossessed",
    "Duels won": "duels_won_pct",
    "Duels won %": "duels_won_pct",
    "Aerial duels won": "aerial_duels_won_pct",
    "Aerial duels won %": "aerial_duels_won_pct",
    "Recoveries": "recoveries",
    "Fouls won": "fouls_won",
    "Touches": "touches",
    "Goals conceded": "goals_conceded_on_pitch",
    "Goals conceded on pitch": "goals_conceded_on_pitch",
    "xG against": "xg_against_on_pitch",
    "xG conceded": "xg_against_on_pitch",
    "xG against on pitch": "xg_against_on_pitch",
}

_client: Optional[MobFot] = None


def _get_client() -> MobFot:
    global _client
    if _client is None:
        _client = MobFot()
    return _client


# ── Public API ───────────────────────────────────────────────────────────────


def search_player(name: str) -> List[Dict[str, Any]]:
    """Search FotMob for a player by name. Returns list of matches.

    Each match dict has at least ``id`` and ``name``.
    """
    key = cache.make_key("fotmob_search", name.lower().strip())
    cached = cache.get(key, max_age=86400 * 7)  # 7-day cache for searches
    if cached is not None:
        return cached

    client = _get_client()
    raw = client.search(name)
    players: list[dict] = []
    # FotMob search returns grouped results
    if isinstance(raw, dict):
        for squad_or_player in raw.get("squad", []):
            players.append(
                {"id": squad_or_player.get("id"), "name": squad_or_player.get("name")}
            )
        # Also check 'players' key variant
        for p in raw.get("players", []):
            players.append({"id": p.get("id"), "name": p.get("name")})
        # Some versions nest under suggestions
        for suggestion in raw.get("suggestions", []):
            if suggestion.get("type") == "player":
                players.append(
                    {"id": suggestion.get("id"), "name": suggestion.get("text", suggestion.get("name"))}
                )

    cache.set(key, players)
    return players


def get_player_stats(
    player_id: int,
    season: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch full player data from FotMob and return structured stats.

    Parameters
    ----------
    player_id : int
        FotMob player ID.
    season : str, optional
        Season string like ``"2024/2025"``.  If *None* the current season
        is returned.

    Returns
    -------
    dict with keys:
        - ``name``: player display name
        - ``team``: current team name
        - ``team_id``: current team FotMob ID
        - ``position``: primary position
        - ``minutes_played``: total minutes this season
        - ``appearances``: number of appearances
        - ``per90``: dict mapping each of the 23 metrics to its per-90 value
          (``None`` for unavailable metrics)
        - ``raw``: the original FotMob JSON (for debugging / future use)
    """
    key = cache.make_key("fotmob_player", str(player_id), season or "current")
    cached = cache.get(key, max_age=86400)  # 1-day cache
    if cached is not None:
        return cached

    client = _get_client()
    raw = client.get_player(player_id)

    result = _parse_player_response(raw)
    result["raw"] = raw
    cache.set(key, result)
    return result


def get_team_players_stats(
    team_id: int,
    season: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch team squad data and return per-player stat summaries.

    Parameters
    ----------
    team_id : int
        FotMob team ID.
    season : str, optional
        Season string.

    Returns
    -------
    list[dict] — one entry per squad player, same shape as ``get_player_stats``
    minus the ``raw`` field.
    """
    key = cache.make_key("fotmob_team", str(team_id), season or "current")
    cached = cache.get(key, max_age=86400)
    if cached is not None:
        return cached

    client = _get_client()
    raw = client.get_team(team_id)

    players: list[dict] = []
    squad = []
    if isinstance(raw, dict):
        squad = raw.get("squad", [])
        # squad is a list of position groups
        for group in squad:
            if isinstance(group, dict):
                for member in group.get("members", []):
                    players.append(
                        {
                            "id": member.get("id"),
                            "name": member.get("name"),
                            "position": group.get("title", "Unknown"),
                        }
                    )

    cache.set(key, players)
    return players


# ── Internal parsing ─────────────────────────────────────────────────────────


def _parse_player_response(raw: Any) -> Dict[str, Any]:
    """Extract structured stats from a FotMob ``get_player`` response."""
    result: Dict[str, Any] = {
        "name": "",
        "team": "",
        "team_id": None,
        "position": "",
        "minutes_played": 0,
        "appearances": 0,
        "per90": {m: None for m in ALL_METRICS},
    }

    if not isinstance(raw, dict):
        return result

    result["name"] = raw.get("name", "")

    # Primary team
    primary_team = raw.get("primaryTeam", {})
    if isinstance(primary_team, dict):
        result["team"] = primary_team.get("teamName", "")
        result["team_id"] = primary_team.get("teamId")

    # Position
    origin = raw.get("origin", {})
    if isinstance(origin, dict):
        result["position"] = origin.get("positionDesc", {}).get("primaryPosition", {}).get("label", "")
    if not result["position"]:
        result["position"] = raw.get("positionDescription", {}).get("primaryPosition", {}).get("label", "")

    # Stats — FotMob nests stats in multiple layers
    stats_section = raw.get("mainLeague", {})
    if isinstance(stats_section, dict):
        result["appearances"] = stats_section.get("appearances", 0) or 0
        result["minutes_played"] = stats_section.get("minutesPlayed", 0) or 0

    # Per-90 stats are in the statSeasons / statsSection structure
    stat_seasons = raw.get("statSeasons", [])
    for season_block in stat_seasons:
        if not isinstance(season_block, dict):
            continue
        stat_items = season_block.get("statisticsOfSelectedSeason", [])
        if not isinstance(stat_items, list):
            continue
        for category in stat_items:
            if not isinstance(category, dict):
                continue
            for stat in category.get("stats", []):
                if not isinstance(stat, dict):
                    continue
                stat_name = stat.get("title", "") or stat.get("key", "")
                canonical = _FOTMOB_KEY_MAP.get(stat_name)
                if canonical is None:
                    continue
                # Percentage stats stored as-is (not per-90)
                if canonical.endswith("_pct"):
                    total = stat.get("per90") or stat.get("value")
                    if total is not None:
                        try:
                            result["per90"][canonical] = float(total)
                        except (ValueError, TypeError):
                            pass
                    continue

                # Prefer per90 value
                per90_val = stat.get("per90")
                if per90_val is not None:
                    try:
                        result["per90"][canonical] = float(per90_val)
                    except (ValueError, TypeError):
                        pass
                elif result["per90"][canonical] is None:
                    # Fallback: compute per-90 from total if minutes available
                    total = stat.get("value")
                    if total is not None and result["minutes_played"] and result["minutes_played"] > 0:
                        try:
                            total_f = float(total)
                            result["per90"][canonical] = round(
                                total_f / result["minutes_played"] * 90, 4
                            )
                        except (ValueError, TypeError):
                            pass

    return result
