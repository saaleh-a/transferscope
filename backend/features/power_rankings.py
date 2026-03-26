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


# ── ClubElo → Sofascore direct name mapping ──────────────────────────────────
# ClubElo uses abbreviated team names (e.g. "PSG", "ManCity") while Sofascore
# uses full display names (e.g. "Paris Saint-Germain", "Manchester City").
# This mapping canonicalizes ClubElo names at data-load time so that the
# ``teams`` dict keys match the names that come from the Sofascore dropdowns.
# Fuzzy matching is kept as a safety net for any unmapped teams.
_CLUBELO_TO_SOFASCORE: Dict[str, str] = {
    # England
    "ManCity": "Manchester City",
    "ManUtd": "Manchester United",
    "Tottenham": "Tottenham Hotspur",
    "Wolves": "Wolverhampton Wanderers",
    "NottmForest": "Nottingham Forest",
    "SheffUtd": "Sheffield United",
    "WestHam": "West Ham United",
    "Newcastle": "Newcastle United",
    "Brighton": "Brighton & Hove Albion",
    "Leicester": "Leicester City",
    "WestBrom": "West Bromwich Albion",
    "SheffWed": "Sheffield Wednesday",
    "QPR": "Queens Park Rangers",
    "Boro": "Middlesbrough",
    "Coventry": "Coventry City",
    "Stoke": "Stoke City",
    "Cardiff": "Cardiff City",
    "Swansea": "Swansea City",
    "Norwich": "Norwich City",
    "Leeds": "Leeds United",
    "Sunderland": "Sunderland AFC",
    "Huddersfield": "Huddersfield Town",
    "Hull": "Hull City",
    "Bournemouth": "AFC Bournemouth",
    # France
    "PSG": "Paris Saint-Germain",
    "Monaco": "AS Monaco",
    "Lyon": "Olympique Lyonnais",
    "Marseille": "Olympique de Marseille",
    "Lille": "Lille OSC",
    "Rennes": "Stade Rennais",
    "Nantes": "FC Nantes",
    "Nice": "OGC Nice",
    "Lens": "RC Lens",
    "Strasbourg": "RC Strasbourg Alsace",
    "StEtienne": "AS Saint-Étienne",
    "Montpellier": "Montpellier HSC",
    "Brest": "Stade Brestois 29",
    "Toulouse": "Toulouse FC",
    "Reims": "Stade de Reims",
    "Angers": "Angers SCO",
    "LeHavre": "Le Havre AC",
    # Germany
    "Bayern": "Bayern Munich",
    "BayernMunich": "Bayern Munich",
    "Dortmund": "Borussia Dortmund",
    "Leverkusen": "Bayer 04 Leverkusen",
    "Leipzig": "RB Leipzig",
    "Frankfurt": "Eintracht Frankfurt",
    "Gladbach": "Borussia Mönchengladbach",
    "Wolfsburg": "VfL Wolfsburg",
    "Freiburg": "SC Freiburg",
    "Hoffenheim": "TSG 1899 Hoffenheim",
    "Stuttgart": "VfB Stuttgart",
    "Mainz": "1. FSV Mainz 05",
    "Augsburg": "FC Augsburg",
    "Heidenheim": "1. FC Heidenheim 1846",
    "UnionBerlin": "1. FC Union Berlin",
    "HerthaBerlin": "Hertha BSC",
    "Bochum": "VfL Bochum 1848",
    "Koln": "1. FC Köln",
    # Spain
    "RealMadrid": "Real Madrid",
    "Atletico": "Atlético Madrid",
    "AtleticoMadrid": "Atlético Madrid",
    "AthleticBilbao": "Athletic Club",
    "AthleticClub": "Athletic Club",
    "Betis": "Real Betis",
    "RSociedad": "Real Sociedad",
    "RealSociedad": "Real Sociedad",
    "CeltaVigo": "Celta Vigo",
    "Alaves": "Deportivo Alavés",
    "Vallecano": "Rayo Vallecano",
    "RayoVallecano": "Rayo Vallecano",
    "LasPalmas": "UD Las Palmas",
    "Leganes": "CD Leganés",
    "RealValladolid": "Real Valladolid",
    "Osasuna": "CA Osasuna",
    # Italy
    "Inter": "Inter",
    "InterMilan": "Inter",
    "ACMilan": "AC Milan",
    "Milan": "AC Milan",
    "Napoli": "SSC Napoli",
    "Roma": "AS Roma",
    "Lazio": "SS Lazio",
    "Atalanta": "Atalanta BC",
    "Fiorentina": "ACF Fiorentina",
    "Torino": "Torino FC",
    "Bologna": "Bologna FC 1909",
    "Udinese": "Udinese Calcio",
    "Cagliari": "Cagliari Calcio",
    "Empoli": "Empoli FC",
    "Verona": "Hellas Verona",
    "Monza": "AC Monza",
    "Lecce": "US Lecce",
    "Parma": "Parma Calcio 1913",
    "Genoa": "Genoa CFC",
    "Sassuolo": "US Sassuolo",
    "Salernitana": "US Salernitana 1919",
    "Frosinone": "Frosinone Calcio",
    "Como": "Como 1907",
    "Venezia": "Venezia FC",
    # Portugal
    "Sporting": "Sporting CP",
    "SportingCP": "Sporting CP",
    # Netherlands
    "AZ": "AZ Alkmaar",
    "AZAlkmaar": "AZ Alkmaar",
    "PSV": "PSV Eindhoven",
    "PSVEindhoven": "PSV Eindhoven",
    "Feyenoord": "Feyenoord Rotterdam",
    "Twente": "FC Twente",
    # Turkey
    "Galatasaray": "Galatasaray SK",
    "Fenerbahce": "Fenerbahçe SK",
    "Besiktas": "Beşiktaş JK",
    # Scotland
    "Celtic": "Celtic FC",
    "Rangers": "Rangers FC",
    # Belgium
    "ClubBrugge": "Club Brugge KV",
    "Anderlecht": "RSC Anderlecht",
    # Austria
    "Salzburg": "FC Red Bull Salzburg",
    "RBSalzburg": "FC Red Bull Salzburg",
}

# Build reverse lookup (Sofascore → ClubElo) for history queries
_SOFASCORE_TO_CLUBELO: Dict[str, str] = {v: k for k, v in _CLUBELO_TO_SOFASCORE.items()}


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
            for raw_name in ce_df.index:
                elo_val = float(ce_df.loc[raw_name, "elo"])
                ce_league = ce_df.loc[raw_name, "league"]
                # Map ClubElo league to our code.  When soccerdata is used
                # it only maps 5 major leagues — the rest become NaN.
                # Fall back to our own mapping from country+level columns.
                league_code = _clubelo_to_code(ce_league)
                if league_code is None and "country" in ce_df.columns:
                    league_code = _clubelo_to_code_from_country(
                        ce_df.loc[raw_name, "country"],
                        ce_df.loc[raw_name, "level"] if "level" in ce_df.columns else 1,
                    )
                if league_code:
                    # Canonicalize ClubElo abbreviated name → Sofascore
                    # full name so that dropdown selections match directly.
                    canonical = _CLUBELO_TO_SOFASCORE.get(str(raw_name), str(raw_name))
                    all_teams[canonical] = (elo_val, league_code)
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

    # 1. Exact match (cheapest check)
    if team_name in teams:
        ranking = teams[team_name]
        ranking.match_type = "exact"
        return ranking

    # 2. Accent-normalized exact match — handles "Atlético Madrid" matching
    #    "Atletico Madrid" or vice-versa without needing a fuzzy pass.
    norm_query = _strip_accents(team_name)
    if norm_query != team_name:
        for key in teams:
            if _strip_accents(key) == norm_query:
                ranking = teams[key]
                ranking.match_type = "exact"
                return ranking

    # 3. Build normalized lookup and try fuzzy match
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

def _strip_accents(name: str) -> str:
    """Remove diacritics from *name* while preserving case and spacing."""
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def _clubelo_to_code(clubelo_league: Optional[str]) -> Optional[str]:
    """Map a ClubElo league string to our league code."""
    if clubelo_league is None:
        return None
    if isinstance(clubelo_league, float):  # NaN from soccerdata
        return None
    clubelo_league = str(clubelo_league)
    for code, info in LEAGUES.items():
        if info.clubelo_league == clubelo_league:
            return code
    # Unknown European league — still track it with the raw string
    return None


# Country+level → league code mapping for the soccerdata path where
# ``_translate_league`` only handles the Big-5 and sets everything else
# to NaN.  Uses the same country codes that ClubElo CSV data contains.
_COUNTRY_LEVEL_TO_CODE: Dict[str, Dict[int, str]] = {}
for _code, _info in LEAGUES.items():
    if _info.clubelo_league is not None:
        # Extract country prefix from e.g. "ENG-Premier League" → "ENG"
        _country = _info.clubelo_league.split("-")[0]
        # Derive level from league code suffix (e.g. "ENG1" → 1, "ENG2" → 2)
        _level = int(_code[-1]) if _code[-1].isdigit() else 1
        _COUNTRY_LEVEL_TO_CODE.setdefault(_country, {})[_level] = _code


def _clubelo_to_code_from_country(
    country: Optional[str],
    level: Optional[int] = 1,
) -> Optional[str]:
    """Derive league code from raw ClubElo country and level columns.

    This is a fallback for when ``soccerdata``'s ``_translate_league``
    sets the league column to NaN for non-Big-5 leagues.
    """
    if country is None or (isinstance(country, float) and pd.isna(country)):
        return None
    country = str(country).strip()
    try:
        level_int = int(level)
    except (ValueError, TypeError):
        level_int = 1
    lvl_map = _COUNTRY_LEVEL_TO_CODE.get(country, {})
    return lvl_map.get(level_int) or lvl_map.get(1)


# ── Fuzzy team name matching ─────────────────────────────────────────────────

# Only strip short abbreviation suffixes that never form the core name.
# NOTE: "Club" removed — stripping it turns "Athletic Club" into just
# "athletic" which loses identity.  "AC" is kept because "AC Milan" → "Milan"
# is unambiguous (there is no other "AC" team that conflicts).
_STRIP_ABBREVS = re.compile(
    r"\b(FC|CF|SC|AC|AS|SS|SK|FK|BK|IF|SV|VfB|VfL|TSG|BSC|"
    r"1\.\s*FC|Calcio|Futbol)\b",
    re.IGNORECASE,
)

# Minimum similarity ratio (0-1) for SequenceMatcher to accept a match.
# 0.65 rejects false positives like "Arsenal" matching "Marseille" (0.625).
# Edge cases below 0.65 (ManCity ↔ ManchesterCity = 0.667, ManUtd ↔
# ManchesterUnited = 0.545) are handled by _EXTREME_ABBREVS instead.
_FUZZY_THRESHOLD = 0.65

# Abbreviation map for cases where SequenceMatcher mathematically fails
# (ratio < 0.5) and substring matching can't help.  These cover common
# ClubElo ↔ Sofascore naming discrepancies.
_EXTREME_ABBREVS: Dict[str, List[str]] = {
    # England
    "manchesterunited": ["manutd", "manunited", "manu"],
    "manutd": ["manchesterunited"],
    "manunited": ["manchesterunited"],
    "manchestercity": ["mancity"],
    "mancity": ["manchestercity"],
    "wolverhamptonwanderers": ["wolves", "wolverhampton"],
    "wolves": ["wolverhamptonwanderers", "wolverhampton"],
    "wolverhampton": ["wolverhamptonwanderers", "wolves"],
    "nottinghamforest": ["nottmforest", "nforest"],
    "nottmforest": ["nottinghamforest"],
    "sheffieldunited": ["sheffutd", "sheffieldutd"],
    "sheffutd": ["sheffieldunited"],
    "westhamunited": ["westham"],
    "westham": ["westhamunited"],
    "tottenhamhotspur": ["tottenham", "spurs"],
    "tottenham": ["tottenhamhotspur"],
    "spurs": ["tottenhamhotspur"],
    "newcastleunited": ["newcastle"],
    "newcastle": ["newcastleunited"],
    "brightonhovealbion": ["brighton"],
    "brighton": ["brightonhovealbion"],
    "leicestercity": ["leicester"],
    "leicester": ["leicestercity"],
    # France
    "parissaintgermain": ["psg", "parissg", "parissggermain"],
    "psg": ["parissaintgermain", "parissg"],
    "parissg": ["parissaintgermain", "psg"],
    "olympiquelyonnais": ["lyon", "olympiquelyon"],
    "lyon": ["olympiquelyonnais"],
    "olympiquedemarseille": ["marseille", "olympiquemarseille", "om"],
    "marseille": ["olympiquedemarseille"],
    "staderennais": ["rennes"],
    "rennes": ["staderennais"],
    "asmonaco": ["monaco"],
    "monaco": ["asmonaco"],
    "lilleosc": ["lille"],
    "lille": ["lilleosc"],
    "rclens": ["lens"],
    "lens": ["rclens"],
    "ogcnice": ["nice"],
    "nice": ["ogcnice"],
    # Germany
    "borussiadortmund": ["dortmund", "bvb", "bvbdortmund"],
    "dortmund": ["borussiadortmund"],
    "bvb": ["borussiadortmund"],
    "bvbdortmund": ["borussiadortmund"],
    "bayernmunich": ["bayernmunchen", "bayern"],
    "bayernmunchen": ["bayernmunich", "bayern"],
    "bayern": ["bayernmunchen", "bayernmunich"],
    "borussiamonchengladbach": ["gladbach", "borussiamgladbach", "monchengladbach"],
    "borussiamgladbach": ["gladbach", "borussiamonchengladbach", "monchengladbach"],
    "gladbach": ["borussiamonchengladbach", "borussiamgladbach"],
    "monchengladbach": ["borussiamonchengladbach", "borussiamgladbach"],
    "bayerleverkusen": ["leverkusen", "bayer04leverkusen"],
    "bayer04leverkusen": ["leverkusen", "bayerleverkusen"],
    "leverkusen": ["bayerleverkusen", "bayer04leverkusen"],
    "rbleipzig": ["leipzig"],
    "leipzig": ["rbleipzig"],
    "eintrachtfrankfurt": ["frankfurt", "efrankfurt"],
    "frankfurt": ["eintrachtfrankfurt"],
    "vflwolfsburg": ["wolfsburg"],
    "wolfsburg": ["vflwolfsburg"],
    "scfreiburg": ["freiburg"],
    "freiburg": ["scfreiburg"],
    "vfbstuttgart": ["stuttgart"],
    "stuttgart": ["vfbstuttgart"],
    "tsg1899hoffenheim": ["hoffenheim"],
    "hoffenheim": ["tsg1899hoffenheim"],
    "1fcunionberlin": ["unionberlin"],
    "unionberlin": ["1fcunionberlin"],
    "1fsv mainz05": ["mainz", "mainz05"],
    "mainz": ["mainz05"],
    "1fckoln": ["koln"],
    "koln": ["1fckoln"],
    # Spain
    "atleticomadrid": ["atletico", "atleticodemadrid", "atlmadrid"],
    "atletico": ["atleticomadrid"],
    "atlmadrid": ["atleticomadrid"],
    "athleticclub": ["athleticbilbao", "athletic", "bilbao"],
    "athleticbilbao": ["athleticclub", "athletic", "bilbao"],
    "athletic": ["athleticclub", "athleticbilbao"],
    "realbetis": ["betis", "realbetisbalompie"],
    "betis": ["realbetis"],
    "realsociedad": ["rsociedad", "lasociedad"],
    "deportivoalaves": ["alaves"],
    "alaves": ["deportivoalaves"],
    "celtavigo": ["celta", "rceltadevigo"],
    "celta": ["celtavigo"],
    "rayovallecano": ["rayo"],
    "rayo": ["rayovallecano"],
    # Italy
    "internazionale": ["inter", "intermilan", "intermilanfc", "internazionalemilano"],
    "internazionalemilano": ["inter", "intermilan", "internazionale"],
    "inter": ["internazionale", "intermilan", "internazionalemilano"],
    "intermilan": ["internazionale", "inter", "internazionalemilano"],
    "acmilan": ["milan"],
    "milan": ["acmilan"],
    "sscnapoli": ["napoli"],
    "napoli": ["sscnapoli"],
    "asroma": ["roma"],
    "roma": ["asroma"],
    "sslazio": ["lazio"],
    "lazio": ["sslazio"],
    "atalantabc": ["atalanta"],
    "atalanta": ["atalantabc"],
    "acffiorentina": ["fiorentina"],
    "fiorentina": ["acffiorentina"],
    "hellasverona": ["verona"],
    "verona": ["hellasverona"],
    # Portugal
    "sportinglisbon": ["sportingcp", "sporting"],
    "sportingcp": ["sportinglisbon", "sporting"],
    "sporting": ["sportinglisbon", "sportingcp"],
    # Netherlands
    "azmaalkmaar": ["az", "azalkmaar"],
    "azalkmaar": ["az", "azmaalkmaar"],
    "az": ["azalkmaar", "azmaalkmaar"],
    "psv": ["psveindhoven"],
    "psveindhoven": ["psv"],
    # Turkey
    "galatasaray": ["galatasaraysk"],
    "galatasaraysk": ["galatasaray"],
    "fenerbahce": ["fenerbahcesk"],
    "fenerbahcesk": ["fenerbahce"],
    "besiktas": ["besiktasjk"],
    "besiktasjk": ["besiktas"],
    # Belgium
    "clubbrugge": ["clubbruggekv"],
    "clubbruggekv": ["clubbrugge"],
    "rscanderlecht": ["anderlecht"],
    "anderlecht": ["rscanderlecht"],
    # Scotland
    "celticfc": ["celtic"],
    "celtic": ["celticfc"],
    "rangersfc": ["rangers"],
    "rangers": ["rangersfc"],
    # Austria
    "fcredbullsalzburg": ["salzburg", "rbsalzburg", "redbullsalzburg"],
    "salzburg": ["fcredbullsalzburg", "rbsalzburg"],
    "rbsalzburg": ["fcredbullsalzburg", "salzburg"],
    # South America
    "flamengo": ["flamengobj", "crflamengo"],
    "palmeiras": ["sepalmeiras"],
    "corinthians": ["sccorinthians", "corinthianspaulista"],
    "saopaulfc": ["saopaulo"],
    "saopaulo": ["saopaulfc"],
    "atleticomineiro": ["atleticomg", "camatleticomineiro"],
    "riverplate": ["cariverplate"],
    "bocajuniors": ["cabocajuniors"],
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
    from any league — no hardcoded alias list (except extreme abbrevs).

    Matching strategy (in priority order):
    1. Exact normalized match (fast path)
    2. Extreme abbreviation lookup (PSG ↔ Paris Saint-Germain, etc.)
    3. Substring containment with overlap ratio guard
    4. Word-level matching (shared significant words)
    5. SequenceMatcher similarity ratio

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

    # 2. Extreme abbreviation lookup — must come before substring to prevent
    #    false positives like "Paris FC" beating "PSG" for "Paris Saint-Germain"
    q_aliases = _EXTREME_ABBREVS.get(q, [])
    for alias in q_aliases:
        if alias in candidates:
            return candidates[alias]
    # Reverse: check if any candidate has an alias matching q
    for norm, orig in candidates.items():
        for alias in _EXTREME_ABBREVS.get(norm, []):
            if alias == q:
                return orig

    # 3. Substring containment — one name fully inside the other.
    #    Guard: require the shorter name to be at least 6 chars AND
    #    represent >= 45% of the longer name.  This prevents
    #    "paris" (5 chars, 29% of "parissaintgermain") from matching.
    best_sub: Optional[str] = None
    best_sub_len = 0
    for norm, orig in candidates.items():
        shorter, longer = (norm, q) if len(norm) <= len(q) else (q, norm)
        if len(shorter) < 6:
            continue
        if shorter not in longer:
            continue
        ratio = len(shorter) / len(longer)
        if ratio < 0.45:
            continue
        if len(shorter) > best_sub_len:
            best_sub = orig
            best_sub_len = len(shorter)
    if best_sub is not None:
        return best_sub

    # 4. Word-level matching — tokenize into "words" (alpha runs ≥ 4 chars)
    #    and check if any significant word from the query matches a word
    #    in a candidate or vice versa.  This catches "Borussia Dortmund"
    #    sharing "dortmund" with just "Dortmund" even when full-string
    #    substring fails.
    q_words = set(re.findall(r"[a-z]{4,}", q))
    if q_words:
        best_word_match: Optional[str] = None
        best_word_overlap = 0
        for norm, orig in candidates.items():
            c_words = set(re.findall(r"[a-z]{4,}", norm))
            shared = q_words & c_words
            if shared:
                # Weight by total characters in shared words
                overlap = sum(len(w) for w in shared)
                if overlap > best_word_overlap:
                    best_word_overlap = overlap
                    best_word_match = orig
        # Only accept if shared words represent meaningful overlap
        if best_word_match is not None and best_word_overlap >= 5:
            return best_word_match

    # 5. SequenceMatcher fuzzy matching — works for any team, any league
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
