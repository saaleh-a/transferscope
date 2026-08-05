"""Elo source router.

ClubElo is the only source of true-scale club Elo, covering roughly 600
European clubs.

There used to be a WorldFootballElo fallback here for everywhere else. It was
removed because it never worked: it scraped eloratings.net, which publishes
ratings for **national teams**, not clubs, so it returned ``None`` for every
club it was ever asked about.

Clubs outside ClubElo get an Elo rescaled from their Opta Power Ranking in
:func:`backend.features.power_rankings._opta_score_to_raw_elo`. That is what
actually provides global coverage — 13,791 clubs across 333 leagues.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Tuple

from backend.data import clubelo_client


def get_team_elo(
    team_name: str,
    query_date: Optional[date] = None,
) -> Optional[float]:
    """Get raw Elo for a club from ClubElo.

    Returns
    -------
    float or None — raw Elo, or None when ClubElo does not cover the club.
    Callers needing global coverage should go through power_rankings, which
    falls back to an Opta-derived Elo.
    """
    return clubelo_client.get_team_elo(team_name, query_date)


def get_team_elo_with_source(
    team_name: str,
    query_date: Optional[date] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """Like ``get_team_elo`` but also reports the source.

    Returns
    -------
    (elo, source) where source is ``"clubelo"`` or None.
    """
    elo = clubelo_client.get_team_elo(team_name, query_date)
    if elo is not None:
        return elo, "clubelo"
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
    """Check whether ClubElo covers this team."""
    return clubelo_client.is_covered(team_name, query_date)
