"""Football-data.co.uk client — historical match results for coefficient calibration.

Downloads season CSV files from football-data.co.uk for major European leagues.
Computes team-level per-game statistics (shots, shots on target, fouls, corners)
that can be used to derive empirical style coefficients for the adjustment models.

Data source: https://www.football-data.co.uk/data.php
CSV format: Div, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HS, AS, HST, AST, ...

Public API
----------
fetch_season(league, season)
    Download a single season CSV and return a DataFrame.
compute_team_stats(league, season)
    Aggregate per-game team stats for a season.
compute_league_style_profile(league, season)
    Return a dict of league-wide per-game averages (shots, fouls, corners, etc.).
compute_multi_season_profiles(leagues, seasons)
    Batch compute style profiles across multiple leagues/seasons.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.data import cache

_log = logging.getLogger(__name__)

# Cache TTL — 30 days (historical data doesn't change).
_CACHE_TTL = 86400 * 30

# football-data.co.uk uses 2-letter league codes in URLs.
# Maps our internal league codes to their URL path and division code.
_LEAGUE_URL_MAP: Dict[str, Tuple[str, str]] = {
    # (country_path, division_code)
    "ENG1": ("england", "E0"),
    "ENG2": ("england", "E1"),
    "ESP1": ("spain", "SP1"),
    "ESP2": ("spain", "SP2"),
    "GER1": ("germany", "D1"),
    "GER2": ("germany", "D2"),
    "ITA1": ("italy", "I1"),
    "ITA2": ("italy", "I2"),
    "FRA1": ("france", "F1"),
    "FRA2": ("france", "F2"),
    "NED1": ("netherlands", "N1"),
    "BEL1": ("belgium", "B1"),
    "POR1": ("portugal", "P1"),
    "TUR1": ("turkey", "T1"),
    "GRE1": ("greece", "G1"),
    "SCO1": ("scotland", "SC0"),
}

# Columns we extract from the CSV (when available).
_STAT_COLUMNS = [
    "HomeTeam", "AwayTeam",
    "FTHG", "FTAG",  # Full-time goals
    "HS", "AS",       # Shots
    "HST", "AST",     # Shots on target
    "HF", "AF",       # Fouls
    "HC", "AC",       # Corners
    "HY", "AY",       # Yellow cards
]


def _season_url(league_code: str, season: str) -> Optional[str]:
    """Build the download URL for a league/season CSV.

    *season* is e.g. "2324" for 2023-24, matching football-data.co.uk's
    naming convention.
    """
    mapping = _LEAGUE_URL_MAP.get(league_code)
    if mapping is None:
        return None
    _country, div_code = mapping
    return f"https://www.football-data.co.uk/mmz4281/{season}/{div_code}.csv"


def fetch_season(
    league_code: str,
    season: str,
) -> Optional[pd.DataFrame]:
    """Download a single season CSV and return a DataFrame.

    Parameters
    ----------
    league_code : str
        Internal code (e.g. ``"ENG1"``).
    season : str
        Two-digit season code (e.g. ``"2324"`` for 2023-24).

    Returns None on download failure or missing league.
    """
    import requests as _req

    url = _season_url(league_code, season)
    if url is None:
        _log.warning("No URL mapping for league %s", league_code)
        return None

    cache_key = f"footballdata:{league_code}:{season}"
    cached = cache.get(cache_key)
    if cached is not None:
        try:
            return pd.read_csv(io.StringIO(cached), encoding="utf-8")
        except Exception:
            pass

    _log.info("football-data.co.uk: fetching %s", url)
    try:
        resp = _req.get(url, timeout=20)
        resp.raise_for_status()
        raw = resp.text
    except Exception as exc:
        _log.warning("football-data.co.uk download failed: %s", exc)
        return None

    cache.set(cache_key, raw, ttl=_CACHE_TTL)
    try:
        return pd.read_csv(io.StringIO(raw), encoding="utf-8")
    except Exception as exc:
        _log.warning("football-data.co.uk CSV parse failed: %s", exc)
        return None


def compute_team_stats(
    league_code: str,
    season: str,
) -> Optional[pd.DataFrame]:
    """Aggregate per-game team statistics for a season.

    Returns a DataFrame indexed by team name with columns:
        games, goals_per_game, shots_per_game, shots_on_target_per_game,
        fouls_per_game, corners_per_game, yellows_per_game
    """
    df = fetch_season(league_code, season)
    if df is None:
        return None

    # Ensure required columns exist (some older seasons lack stats).
    required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not required.issubset(df.columns):
        return None

    records: Dict[str, Dict[str, float]] = {}

    def _add(team: str, goals: float, shots: float, sot: float,
             fouls: float, corners: float, yellows: float) -> None:
        if team not in records:
            records[team] = {
                "games": 0, "goals": 0, "shots": 0, "sot": 0,
                "fouls": 0, "corners": 0, "yellows": 0,
            }
        records[team]["games"] += 1
        records[team]["goals"] += goals
        records[team]["shots"] += shots
        records[team]["sot"] += sot
        records[team]["fouls"] += fouls
        records[team]["corners"] += corners
        records[team]["yellows"] += yellows

    for _, row in df.iterrows():
        home = str(row.get("HomeTeam", ""))
        away = str(row.get("AwayTeam", ""))
        if not home or not away:
            continue

        _add(
            home,
            _safe_float(row.get("FTHG")),
            _safe_float(row.get("HS")),
            _safe_float(row.get("HST")),
            _safe_float(row.get("HF")),
            _safe_float(row.get("HC")),
            _safe_float(row.get("HY")),
        )
        _add(
            away,
            _safe_float(row.get("FTAG")),
            _safe_float(row.get("AS")),
            _safe_float(row.get("AST")),
            _safe_float(row.get("AF")),
            _safe_float(row.get("AC")),
            _safe_float(row.get("AY")),
        )

    if not records:
        return None

    rows = []
    for team, stats in records.items():
        g = max(stats["games"], 1)
        rows.append({
            "team": team,
            "games": stats["games"],
            "goals_per_game": stats["goals"] / g,
            "shots_per_game": stats["shots"] / g,
            "shots_on_target_per_game": stats["sot"] / g,
            "fouls_per_game": stats["fouls"] / g,
            "corners_per_game": stats["corners"] / g,
            "yellows_per_game": stats["yellows"] / g,
        })

    return pd.DataFrame(rows).set_index("team")


def compute_league_style_profile(
    league_code: str,
    season: str,
) -> Dict[str, float]:
    """Return league-wide per-game averages for a single season.

    Keys: goals, shots, shots_on_target, fouls, corners, yellows.
    Returns empty dict on failure.
    """
    team_df = compute_team_stats(league_code, season)
    if team_df is None or team_df.empty:
        return {}

    return {
        "goals": float(team_df["goals_per_game"].mean()),
        "shots": float(team_df["shots_per_game"].mean()),
        "shots_on_target": float(team_df["shots_on_target_per_game"].mean()),
        "fouls": float(team_df["fouls_per_game"].mean()),
        "corners": float(team_df["corners_per_game"].mean()),
        "yellows": float(team_df["yellows_per_game"].mean()),
    }


def compute_multi_season_profiles(
    league_codes: Optional[List[str]] = None,
    seasons: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Batch compute league style profiles across leagues and seasons.

    Parameters
    ----------
    league_codes : list[str], optional
        Defaults to Big 5 (ENG1, ESP1, GER1, ITA1, FRA1).
    seasons : list[str], optional
        Defaults to last 3 seasons (``["2122", "2223", "2324"]``).

    Returns ``{league_code: averaged_profile_dict}``.
    """
    if league_codes is None:
        league_codes = ["ENG1", "ESP1", "GER1", "ITA1", "FRA1"]
    if seasons is None:
        seasons = ["2122", "2223", "2324"]

    result: Dict[str, Dict[str, float]] = {}

    for lc in league_codes:
        profiles: List[Dict[str, float]] = []
        for s in seasons:
            p = compute_league_style_profile(lc, s)
            if p:
                profiles.append(p)
        if not profiles:
            continue
        # Average across seasons
        avg: Dict[str, float] = {}
        for key in profiles[0]:
            avg[key] = sum(p.get(key, 0) for p in profiles) / len(profiles)
        result[lc] = avg

    return result


def _safe_float(val: Any) -> float:
    """Convert to float, returning 0.0 on failure."""
    try:
        return float(val) if pd.notna(val) else 0.0
    except (ValueError, TypeError):
        return 0.0
