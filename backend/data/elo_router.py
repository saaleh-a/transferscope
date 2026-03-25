"""Elo source router — routes club to correct Elo source, merges scores.

ClubElo takes priority for European clubs.
WorldFootballElo as global fallback.
Returns normalized 0-100 score when requested.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Tuple

from backend.data import clubelo_client, worldfootballelo_client


def get_team_elo(
    team_name: str,
    query_date: Optional[date] = None,
) -> Optional[float]:
    """Get raw Elo for a club from the best available source.

    Priority: ClubElo (European) > WorldFootballElo (global fallback).

    Returns
    -------
    float or None — raw Elo score.
    """
    # Try ClubElo first (European clubs)
    elo = clubelo_client.get_team_elo(team_name, query_date)
    if elo is not None:
        return elo

    # Fallback to WorldFootballElo
    elo = worldfootballelo_client.get_team_elo(team_name, query_date)
    return elo


def get_team_elo_with_source(
    team_name: str,
    query_date: Optional[date] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Like ``get_team_elo`` but also returns which source was used.

    Returns
    -------
    (elo, source) where source is ``"clubelo"`` or ``"worldelo"`` or None.
    """
    elo = clubelo_client.get_team_elo(team_name, query_date)
    if elo is not None:
        return elo, "clubelo"

    elo = worldfootballelo_client.get_team_elo(team_name, query_date)
    if elo is not None:
        return elo, "worldelo"

    return None, None


def normalize_elo(
    raw_elo: float,
    global_min: float,
    global_max: float,
) -> float:
    """Scale a raw Elo to 0-100.

    ``normalized = (raw - min) / (max - min) * 100``
    """
    if global_max == global_min:
        return 50.0
    return (raw_elo - global_min) / (global_max - global_min) * 100.0


def is_covered(team_name: str, query_date: Optional[date] = None) -> bool:
    """Check whether any Elo source covers this team."""
    if clubelo_client.is_covered(team_name, query_date):
        return True
    return worldfootballelo_client.is_covered(team_name)
