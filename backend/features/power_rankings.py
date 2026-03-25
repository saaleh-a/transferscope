"""Dynamic league Power Rankings from mean team Elo per league per day.

Normalize all clubs 0-100 globally daily.
Store per-league mean, std, and percentile bands (10th, 25th, 50th, 75th, 90th).
Compute relative_ability = team_score - league_mean_score.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

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
    match_type: str = "exact"  # "exact" or "fuzzy"


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
            _log.info("ClubElo loaded %d teams", len([t for t in all_teams]))
        else:
            _log.warning("ClubElo returned empty DataFrame for %s", query_date)
    except Exception as exc:
        _log.exception("ClubElo data fetch failed: %s", exc)

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
        except Exception as exc:
            _log.warning("WorldElo fetch failed for %s: %s", info.worldelo_slug, exc)

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

    The returned ``TeamRanking.match_type`` is ``"exact"`` for direct
    hits or ``"fuzzy"`` when the name was resolved via similarity.
    """
    teams, _ = compute_daily_rankings(query_date)

    if not teams:
        _log.warning(
            "Power Rankings empty — no teams loaded from any Elo source"
        )
        return None

    # 1. Exact match
    if team_name in teams:
        ranking = teams[team_name]
        ranking.match_type = "exact"
        return ranking

    # 2. Build normalized lookup and try fuzzy match
    match = _fuzzy_find_team(team_name, teams)
    if match is not None:
        _log.info("Fuzzy matched '%s' → '%s'", team_name, match)
        ranking = teams[match]
        ranking.match_type = "fuzzy"
        return ranking

    _log.warning(
        "No Power Ranking match for '%s' among %d teams", team_name, len(teams)
    )
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


def get_historical_rankings(
    team_name: str,
    months: int = 6,
) -> List[Tuple[date, float]]:
    """Return a team's normalized Power Ranking over the past *months* months.

    Returns a list of ``(date, normalized_score)`` tuples, one per month,
    oldest first.  Falls back to the current score for months where data
    is unavailable.
    """
    from datetime import timedelta

    today = date.today()
    history: List[Tuple[date, float]] = []

    for i in range(months, -1, -1):
        # Approximate month offsets (30-day intervals)
        query_date = today - timedelta(days=30 * i)
        ranking = get_team_ranking(team_name, query_date)
        if ranking is not None:
            history.append((query_date, ranking.normalized_score))

    return history


def compare_leagues(
    league_codes: List[str],
    query_date: Optional[date] = None,
) -> List[Dict[str, Any]]:
    """Compare multiple leagues by their Power Ranking statistics.

    Parameters
    ----------
    league_codes : list[str]
        List of league codes (e.g. ``["ENG1", "ESP1", "GER1"]``).
    query_date : date, optional
        Defaults to today.

    Returns
    -------
    list[dict] — sorted by ``mean_normalized`` descending.  Each dict:
      ``code``, ``name``, ``mean_normalized``, ``std_elo``, ``team_count``,
      ``p10``, ``p25``, ``p50``, ``p75``, ``p90``.
    """
    _, snapshots = compute_daily_rankings(query_date)
    result = []
    for code in league_codes:
        snap = snapshots.get(code)
        if snap is None:
            continue
        result.append({
            "code": code,
            "name": snap.league_name,
            "mean_normalized": round(snap.mean_normalized, 1),
            "std_elo": round(snap.std_elo, 1),
            "team_count": snap.team_count,
            "p10": round(snap.p10, 1),
            "p25": round(snap.p25, 1),
            "p50": round(snap.p50, 1),
            "p75": round(snap.p75, 1),
            "p90": round(snap.p90, 1),
        })
    result.sort(key=lambda x: x["mean_normalized"], reverse=True)
    return result


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

# Only strip short abbreviation suffixes that never form the core name.
_STRIP_ABBREVS = re.compile(
    r"\b(FC|CF|SC|AC|AS|SS|SK|FK|BK|IF|SV|VfB|VfL|TSG|BSC|"
    r"1\.\s*FC|Calcio|Club|Futbol)\b",
    re.IGNORECASE,
)

# Minimum similarity ratio (0-1) for SequenceMatcher to accept a match.
# 0.65 avoids false positives like "Arsenal" matching "Marseille" (0.625)
# while still catching most legitimate matches.
_FUZZY_THRESHOLD = 0.65

# Tiny abbreviation map ONLY for extreme cases where SequenceMatcher
# mathematically fails (ratio < 0.5).  These are ClubElo-specific
# abbreviations with no string similarity to the full name.
# Everything else is handled automatically by SequenceMatcher.
_EXTREME_ABBREVS: Dict[str, List[str]] = {
    "manchesterunited": ["manutd", "manunited"],
    "manutd": ["manchesterunited"],
    "parissaintgermain": ["psg"],
    "psg": ["parissaintgermain"],
    "wolverhamptonwanderers": ["wolves"],
    "wolves": ["wolverhamptonwanderers", "wolverhampton"],
}


def _normalize_team_name(name: str) -> str:
    """Reduce a team name to a canonical key for matching.

    Steps:
    - NFKD-decompose accented characters (ü → u, é → e)
    - Lowercase
    - Strip short abbreviations (FC, CF, SC, etc.)
    - Remove all non-alphanumeric characters
    """
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = _STRIP_ABBREVS.sub("", name)
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def _fuzzy_find_team(
    query: str,
    teams: Dict[str, TeamRanking],
) -> Optional[str]:
    """Find the best-matching team name from *teams* for *query*.

    Uses automatic fuzzy matching that scales to any number of teams
    from any league — no hardcoded alias list.

    Matching strategy (in priority order):
    1. Exact normalized match (fast path)
    2. Substring containment (one name contains the other)
    3. SequenceMatcher similarity ratio (handles abbreviations,
       reorderings, and partial names across all leagues)

    Returns the original key from *teams*, or None if no match.
    """
    q = _normalize_team_name(query)
    if not q:
        return None

    # Build normalised → original key map
    candidates: Dict[str, str] = {}
    for team_name in teams:
        norm = _normalize_team_name(team_name)
        if norm:
            candidates[norm] = team_name

    # 1. Exact normalised match
    if q in candidates:
        return candidates[q]

    # 2. Substring containment — one name fully inside the other
    #    e.g. "bayern" in "bayernmunich", "tottenham" in "tottenhamhotspur"
    #    Prefer longest overlap to avoid false positives.
    best_sub: Optional[str] = None
    best_sub_len = 0
    for norm, orig in candidates.items():
        if len(norm) < 4 or len(q) < 4:
            continue
        if q in norm or norm in q:
            overlap = min(len(q), len(norm))
            if overlap > best_sub_len:
                best_sub = orig
                best_sub_len = overlap
    if best_sub is not None:
        return best_sub

    # 3. Extreme abbreviation lookup — only for the handful of cases where
    #    SequenceMatcher fails (ManUtd ↔ Manchester United, PSG, Wolves)
    q_aliases = _EXTREME_ABBREVS.get(q, [])
    for alias in q_aliases:
        if alias in candidates:
            return candidates[alias]
    # Reverse: check if any candidate has an alias matching q
    for norm, orig in candidates.items():
        for alias in _EXTREME_ABBREVS.get(norm, []):
            if alias == q:
                return orig

    # 4. SequenceMatcher fuzzy matching — works for any team, any league
    #    Handles "ManCity" ↔ "manchestercity", "Flamengo" ↔ "Flamengo RJ",
    #    "São Paulo" ↔ "SaoPaulo", etc.
    best_match: Optional[str] = None
    best_ratio = 0.0
    for norm, orig in candidates.items():
        ratio = SequenceMatcher(None, q, norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = orig

    if best_ratio >= _FUZZY_THRESHOLD:
        return best_match

    return None
