"""ClubElo wrapper via soccerdata (European clubs).

Pulls team Elo snapshots by date and team history.
All responses cached with max_age = 1 day.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import soccerdata as sd

from backend.data import cache

_ONE_DAY = 86400  # seconds

_elo_instance: Optional[sd.ClubElo] = None


def _get_elo() -> sd.ClubElo:
    global _elo_instance
    if _elo_instance is None:
        _elo_instance = sd.ClubElo()
    return _elo_instance


def get_all_by_date(
    query_date: Optional[date] = None,
) -> pd.DataFrame:
    """Return all European club Elo ratings for a given date.

    Parameters
    ----------
    query_date : date, optional
        Defaults to today (UTC).

    Returns
    -------
    DataFrame with index ``team`` and columns:
        rank, country, level, elo, from, to, league
    """
    if query_date is None:
        query_date = date.today()
    date_str = query_date.isoformat()

    key = cache.make_key("clubelo_date", date_str)
    cached = cache.get(key, max_age=_ONE_DAY)
    if cached is not None:
        return cached

    ce = _get_elo()
    df = ce.read_by_date(date_str)

    cache.set(key, df)
    return df


def get_team_elo(
    team_name: str,
    query_date: Optional[date] = None,
) -> Optional[float]:
    """Return a single team's Elo on a given date, or None if not found."""
    df = get_all_by_date(query_date)
    if team_name in df.index:
        return float(df.loc[team_name, "elo"])
    return None


def get_team_history(team_name: str) -> pd.DataFrame:
    """Return full historical Elo series for a European club.

    Returns
    -------
    DataFrame indexed by ``from`` (date) with columns:
        rank, team, country, level, elo, to
    """
    key = cache.make_key("clubelo_history", team_name)
    cached = cache.get(key, max_age=_ONE_DAY * 7)  # history changes slowly
    if cached is not None:
        return cached

    ce = _get_elo()
    df = ce.read_team_history(team_name)

    cache.set(key, df)
    return df


def list_teams_by_league(
    league: str,
    query_date: Optional[date] = None,
) -> List[str]:
    """Return all team names in a given league on a date.

    Parameters
    ----------
    league : str
        League code as returned by ClubElo, e.g. ``"ENG-Premier League"``.
    """
    df = get_all_by_date(query_date)
    return list(df[df["league"] == league].index)


def list_leagues(query_date: Optional[date] = None) -> List[str]:
    """Return all unique league codes available on a date."""
    df = get_all_by_date(query_date)
    return sorted(df["league"].dropna().unique().tolist())


def is_covered(team_name: str, query_date: Optional[date] = None) -> bool:
    """Check whether ClubElo has data for *team_name* on *query_date*."""
    df = get_all_by_date(query_date)
    return team_name in df.index
