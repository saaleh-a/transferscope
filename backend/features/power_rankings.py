"""Dynamic league Power Rankings from mean team Elo per league per day.

Normalize all clubs 0-100 globally daily.
Store per-league mean, std, and percentile bands (10th, 25th, 50th, 75th, 90th).
Compute relative_ability = team_score - league_mean_score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.data import cache, clubelo_client, elo_router, worldfootballelo_client
from backend.utils.league_registry import LEAGUES, LeagueInfo


_ONE_DAY = 86400


@dataclass
class LeagueSnapshot:
    """Per-league statistics for a single day."""

    league_code: str
    league_name: str
    date: date
    mean_elo: float
    std_elo: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    mean_normalized: float  # 0-100 normalized mean
    team_count: int


@dataclass
class TeamRanking:
    """A single team's normalized ranking on a date."""

    team_name: str
    league_code: str
    raw_elo: float
    normalized_score: float  # 0-100
    league_mean_normalized: float
    relative_ability: float  # team - league_mean


def compute_daily_rankings(
    query_date: Optional[date] = None,
) -> Tuple[Dict[str, TeamRanking], Dict[str, LeagueSnapshot]]:
    """Compute global Power Rankings for all known teams on a date.

    Returns
    -------
    (team_rankings, league_snapshots)
        team_rankings: dict[team_name -> TeamRanking]
        league_snapshots: dict[league_code -> LeagueSnapshot]
    """
    if query_date is None:
        query_date = date.today()

    key = cache.make_key("power_rankings", query_date.isoformat())
    cached = cache.get(key, max_age=_ONE_DAY)
    if cached is not None:
        return cached

    # Step 1 — Collect all team Elo scores
    all_teams: Dict[str, Tuple[float, str]] = {}  # team -> (elo, league_code)

    # European clubs from ClubElo
    try:
        ce_df = clubelo_client.get_all_by_date(query_date)
        if ce_df is not None and len(ce_df) > 0:
            for team_name in ce_df.index:
                elo_val = float(ce_df.loc[team_name, "elo"])
                ce_league = ce_df.loc[team_name, "league"]
                # Map ClubElo league to our code
                league_code = _clubelo_to_code(ce_league)
                if league_code:
                    all_teams[team_name] = (elo_val, league_code)
    except Exception:
        pass

    # Non-European clubs from WorldFootballElo
    for code, info in LEAGUES.items():
        if info.clubelo_league is not None:
            continue  # already covered by ClubElo
        if info.worldelo_slug is None:
            continue
        try:
            teams = worldfootballelo_client.get_league_teams(info.worldelo_slug)
            for t in teams:
                if t.get("elo") and t.get("name"):
                    all_teams[t["name"]] = (t["elo"], code)
        except Exception:
            pass

    if not all_teams:
        return {}, {}

    # Step 2 — Global normalization 0-100
    all_elos = [elo for elo, _ in all_teams.values()]
    global_min = min(all_elos)
    global_max = max(all_elos)

    def normalize(elo: float) -> float:
        return elo_router.normalize_elo(elo, global_min, global_max)

    # Step 3 — Build league snapshots
    league_teams: Dict[str, List[Tuple[str, float, float]]] = {}
    for team_name, (elo, code) in all_teams.items():
        norm = normalize(elo)
        league_teams.setdefault(code, []).append((team_name, elo, norm))

    league_snapshots: Dict[str, LeagueSnapshot] = {}
    for code, members in league_teams.items():
        elos = np.array([e for _, e, _ in members])
        norms = np.array([n for _, _, n in members])
        info = LEAGUES.get(code)
        league_snapshots[code] = LeagueSnapshot(
            league_code=code,
            league_name=info.name if info else code,
            date=query_date,
            mean_elo=float(np.mean(elos)),
            std_elo=float(np.std(elos)) if len(elos) > 1 else 0.0,
            p10=float(np.percentile(norms, 10)) if len(norms) > 1 else float(norms[0]),
            p25=float(np.percentile(norms, 25)) if len(norms) > 1 else float(norms[0]),
            p50=float(np.percentile(norms, 50)),
            p75=float(np.percentile(norms, 75)) if len(norms) > 1 else float(norms[0]),
            p90=float(np.percentile(norms, 90)) if len(norms) > 1 else float(norms[0]),
            mean_normalized=float(np.mean(norms)),
            team_count=len(members),
        )

    # Step 4 — Build team rankings with relative ability
    team_rankings: Dict[str, TeamRanking] = {}
    for team_name, (elo, code) in all_teams.items():
        norm = normalize(elo)
        league_mean = league_snapshots[code].mean_normalized
        team_rankings[team_name] = TeamRanking(
            team_name=team_name,
            league_code=code,
            raw_elo=elo,
            normalized_score=norm,
            league_mean_normalized=league_mean,
            relative_ability=norm - league_mean,
        )

    result = (team_rankings, league_snapshots)
    cache.set(key, result)
    return result


def get_team_ranking(
    team_name: str,
    query_date: Optional[date] = None,
) -> Optional[TeamRanking]:
    """Get a single team's Power Ranking.

    Uses fuzzy matching to handle name differences between data sources.
    For example, ClubElo returns ``"RealMadrid"`` while Sofascore returns
    ``"Real Madrid"``.
    """
    teams, _ = compute_daily_rankings(query_date)

    # 1. Exact match
    if team_name in teams:
        return teams[team_name]

    # 2. Build normalized lookup and try fuzzy match
    match = _fuzzy_find_team(team_name, teams)
    if match is not None:
        return teams[match]

    return None


def get_league_snapshot(
    league_code: str,
    query_date: Optional[date] = None,
) -> Optional[LeagueSnapshot]:
    """Get a single league's snapshot."""
    _, leagues = compute_daily_rankings(query_date)
    return leagues.get(league_code)


def get_relative_ability(
    team_name: str,
    query_date: Optional[date] = None,
) -> Optional[float]:
    """Return team_normalized - league_mean_normalized."""
    ranking = get_team_ranking(team_name, query_date)
    if ranking is None:
        return None
    return ranking.relative_ability


def get_change_in_relative_ability(
    team_from: str,
    team_to: str,
    query_date: Optional[date] = None,
) -> Optional[float]:
    """Compute change in relative ability for a transfer.

    Returns target_relative_ability - source_relative_ability.
    """
    ra_from = get_relative_ability(team_from, query_date)
    ra_to = get_relative_ability(team_to, query_date)
    if ra_from is None or ra_to is None:
        return None
    return ra_to - ra_from


# ── Internal ─────────────────────────────────────────────────────────────────

def _clubelo_to_code(clubelo_league: Optional[str]) -> Optional[str]:
    """Map a ClubElo league string to our league code."""
    if clubelo_league is None:
        return None
    for code, info in LEAGUES.items():
        if info.clubelo_league == clubelo_league:
            return code
    # Unknown European league — still track it with the raw string
    return None


# ── Fuzzy team name matching ─────────────────────────────────────────────────

import re
import unicodedata

# Only strip short abbreviation suffixes that never form the core name.
# Do NOT strip "City", "United", etc. — these are meaningful in names like
# "ManCity", "Man City", "Manchester City".
_STRIP_ABBREVS = re.compile(
    r"\b(FC|CF|SC|AC|AS|SS|SK|FK|BK|IF|SV|VfB|VfL|TSG|BSC|"
    r"1\.\s*FC|Calcio|Club|Futbol)\b",
    re.IGNORECASE,
)

# Curated alias map for ClubElo abbreviated names ↔ common full names.
# Keys and values are both lowercased-and-stripped forms.
_TEAM_ALIASES: Dict[str, str] = {
    "mancity": "manchestercity",
    "manchestercity": "mancity",
    "manutd": "manchesterunited",
    "manchesterunited": "manutd",
    "manunited": "manchesterunited",
    "atletico": "atleticomadrid",
    "atleticomadrid": "atletico",
    "spurs": "tottenhamhotspur",
    "tottenham": "tottenhamhotspur",
    "tottenhamhotspur": "tottenham",
    "intermilan": "inter",
    "inter": "intermilan",
    "acmilan": "milan",
    "milan": "acmilan",
    "psg": "parissaintgermain",
    "parissaintgermain": "psg",
    "realsociedad": "rsociedad",
    "realbetis": "betis",
    "betis": "realbetis",
    "westham": "westhamunited",
    "westhamunited": "westham",
    "newcastle": "newcastleunited",
    "newcastleunited": "newcastle",
    "astonvilla": "avilla",
    "wolverhampton": "wolves",
    "wolves": "wolverhampton",
    "wolverhamptonwanderers": "wolves",
    "nottingham": "nottinghamforest",
    "nottinghamforest": "nottingham",
    "leverkusen": "bayerleverkusen",
    "bayerleverkusen": "leverkusen",
    "dortmund": "borussiadortmund",
    "borussiadortmund": "dortmund",
    "gladbach": "borussiamgladbach",
    "borussiamgladbach": "gladbach",
    "leipzig": "rbleipzig",
    "rbleipzig": "leipzig",
}


def _normalize_team_name(name: str) -> str:
    """Reduce a team name to a canonical key for matching.

    Steps:
    - NFKD-decompose accented characters
    - Lowercase
    - Strip short abbreviations (FC, CF, SC, etc.) but NOT meaningful
      words like City, United, Town
    - Remove all non-alphanumeric characters
    """
    # Decompose accents (ü → u, é → e)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    # Strip only short abbreviation suffixes
    name = _STRIP_ABBREVS.sub("", name)
    # Remove non-alphanumeric
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def _fuzzy_find_team(
    query: str,
    teams: Dict[str, TeamRanking],
) -> Optional[str]:
    """Find the best-matching team name from *teams* for *query*.

    Matching strategy (in priority order):
    1. Direct normalized match
    2. Alias lookup (ManCity ↔ Manchester City, etc.)
    3. Substring containment (longer overlap wins)

    Returns the original key from *teams*, or None if no match.
    """
    q = _normalize_team_name(query)
    if not q:
        return None

    # Build normalised → original key map
    candidates: Dict[str, str] = {}
    for team_name in teams:
        norm = _normalize_team_name(team_name)
        candidates[norm] = team_name

    # 1. Direct normalised match
    if q in candidates:
        return candidates[q]

    # 2. Alias lookup: check if query has a known alias in the candidates
    alias = _TEAM_ALIASES.get(q)
    if alias and alias in candidates:
        return candidates[alias]

    # Also check reverse: does any candidate have an alias matching query?
    for norm, orig in candidates.items():
        candidate_alias = _TEAM_ALIASES.get(norm)
        if candidate_alias == q:
            return orig

    # 3. Substring match: query contains candidate or vice-versa
    # Prefer the longest overlapping match to avoid false positives
    # Require minimum 4 chars overlap to avoid spurious matches
    best: Optional[str] = None
    best_len = 0
    for norm, orig in candidates.items():
        if len(norm) < 4 or len(q) < 4:
            continue
        if q in norm or norm in q:
            overlap = min(len(q), len(norm))
            if overlap > best_len:
                best = orig
                best_len = overlap

    return best
