"""League ID mappings for Sofascore + Elo sources.

Central registry so every module resolves league/team names consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class LeagueInfo:
    """Metadata for a single league across all data sources."""

    name: str
    sofascore_tournament_id: Optional[int]  # Sofascore unique-tournament ID
    clubelo_league: Optional[str]  # None if not in ClubElo
    worldelo_slug: Optional[str]  # None if not on eloratings.net
    country: str
    continent: str


# ── Registry ─────────────────────────────────────────────────────────────────

LEAGUES: Dict[str, LeagueInfo] = {
    # ── Europe ───────────────────────────────────────────────────────────
    "ENG1": LeagueInfo(
        name="Premier League",
        sofascore_tournament_id=17,
        clubelo_league="ENG-Premier League",
        worldelo_slug="England",
        country="England",
        continent="Europe",
    ),
    "ENG2": LeagueInfo(
        name="Championship",
        sofascore_tournament_id=18,
        clubelo_league="ENG-Championship",
        worldelo_slug="England",
        country="England",
        continent="Europe",
    ),
    "ESP1": LeagueInfo(
        name="La Liga",
        sofascore_tournament_id=8,
        clubelo_league="ESP-La Liga",
        worldelo_slug="Spain",
        country="Spain",
        continent="Europe",
    ),
    "GER1": LeagueInfo(
        name="Bundesliga",
        sofascore_tournament_id=35,
        clubelo_league="GER-Bundesliga",
        worldelo_slug="Germany",
        country="Germany",
        continent="Europe",
    ),
    "ITA1": LeagueInfo(
        name="Serie A",
        sofascore_tournament_id=23,
        clubelo_league="ITA-Serie A",
        worldelo_slug="Italy",
        country="Italy",
        continent="Europe",
    ),
    "FRA1": LeagueInfo(
        name="Ligue 1",
        sofascore_tournament_id=34,
        clubelo_league="FRA-Ligue 1",
        worldelo_slug="France",
        country="France",
        continent="Europe",
    ),
    "NED1": LeagueInfo(
        name="Eredivisie",
        sofascore_tournament_id=37,
        clubelo_league="NED-Eredivisie",
        worldelo_slug="Netherlands",
        country="Netherlands",
        continent="Europe",
    ),
    "POR1": LeagueInfo(
        name="Primeira Liga",
        sofascore_tournament_id=238,
        clubelo_league="POR-Liga Portugal",
        worldelo_slug="Portugal",
        country="Portugal",
        continent="Europe",
    ),
    "BEL1": LeagueInfo(
        name="Belgian Pro League",
        sofascore_tournament_id=11,
        clubelo_league="BEL-First Division A",
        worldelo_slug="Belgium",
        country="Belgium",
        continent="Europe",
    ),
    "TUR1": LeagueInfo(
        name="Super Lig",
        sofascore_tournament_id=52,
        clubelo_league="TUR-Süper Lig",
        worldelo_slug="Turkey",
        country="Turkey",
        continent="Europe",
    ),
    # ── South America ────────────────────────────────────────────────────
    "BRA1": LeagueInfo(
        name="Brasileirao Serie A",
        sofascore_tournament_id=325,
        clubelo_league=None,
        worldelo_slug="Brazil",
        country="Brazil",
        continent="South America",
    ),
    "BRA2": LeagueInfo(
        name="Brasileirao Serie B",
        sofascore_tournament_id=390,
        clubelo_league=None,
        worldelo_slug="Brazil",
        country="Brazil",
        continent="South America",
    ),
    "ARG1": LeagueInfo(
        name="Argentine Primera Division",
        sofascore_tournament_id=155,
        clubelo_league=None,
        worldelo_slug="Argentina",
        country="Argentina",
        continent="South America",
    ),
    "COL1": LeagueInfo(
        name="Colombian Primera A",
        sofascore_tournament_id=109,
        clubelo_league=None,
        worldelo_slug="Colombia",
        country="Colombia",
        continent="South America",
    ),
    "CHI1": LeagueInfo(
        name="Chilean Primera Division",
        sofascore_tournament_id=19,
        clubelo_league=None,
        worldelo_slug="Chile",
        country="Chile",
        continent="South America",
    ),
    "URU1": LeagueInfo(
        name="Uruguayan Primera Division",
        sofascore_tournament_id=1064,
        clubelo_league=None,
        worldelo_slug="Uruguay",
        country="Uruguay",
        continent="South America",
    ),
    "ECU1": LeagueInfo(
        name="Ecuadorian Serie A",
        sofascore_tournament_id=240,
        clubelo_league=None,
        worldelo_slug="Ecuador",
        country="Ecuador",
        continent="South America",
    ),
    # ── Other ────────────────────────────────────────────────────────────
    "USA1": LeagueInfo(
        name="MLS",
        sofascore_tournament_id=242,
        clubelo_league=None,
        worldelo_slug="United_States",
        country="United States",
        continent="North America",
    ),
    "SAU1": LeagueInfo(
        name="Saudi Pro League",
        sofascore_tournament_id=955,
        clubelo_league=None,
        worldelo_slug="Saudi_Arabia",
        country="Saudi Arabia",
        continent="Asia",
    ),
    "JPN1": LeagueInfo(
        name="J-League",
        sofascore_tournament_id=196,
        clubelo_league=None,
        worldelo_slug="Japan",
        country="Japan",
        continent="Asia",
    ),
}


def get_by_sofascore_id(tournament_id: int) -> Optional[LeagueInfo]:
    """Look up league info by Sofascore tournament ID."""
    for info in LEAGUES.values():
        if info.sofascore_tournament_id == tournament_id:
            return info
    return None


def get_by_clubelo_league(clubelo_league: str) -> Optional[LeagueInfo]:
    """Look up league info by ClubElo league code."""
    for info in LEAGUES.values():
        if info.clubelo_league == clubelo_league:
            return info
    return None


def get_by_worldelo_slug(slug: str) -> Optional[LeagueInfo]:
    """Look up league info by WorldFootballElo country slug."""
    for info in LEAGUES.values():
        if info.worldelo_slug == slug:
            return info
    return None


def leagues_by_continent(continent: str) -> List[LeagueInfo]:
    """Return all leagues in a given continent."""
    return [l for l in LEAGUES.values() if l.continent == continent]


def european_leagues() -> List[LeagueInfo]:
    """Return all European leagues (covered by ClubElo)."""
    return [l for l in LEAGUES.values() if l.clubelo_league is not None]


def non_european_leagues() -> List[LeagueInfo]:
    """Return all non-European leagues (WorldFootballElo only)."""
    return [l for l in LEAGUES.values() if l.clubelo_league is None]


def all_league_codes() -> List[str]:
    """Return all league codes (e.g. 'ENG1', 'BRA1')."""
    return list(LEAGUES.keys())
