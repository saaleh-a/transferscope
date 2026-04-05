"""League ID mappings for Sofascore + Elo sources.

Central registry so every module resolves league/team names consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LeagueInfo:
    """Metadata for a single league across all data sources."""

    name: str
    sofascore_tournament_id: Optional[int]  # Sofascore unique-tournament ID
    clubelo_league: Optional[str]  # None if not in ClubElo
    worldelo_slug: Optional[str]  # None if not on eloratings.net
    country: str
    continent: str
    reep_competition_id: Optional[str] = None  # REEP competitions.csv reep_id


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
        reep_competition_id="reep_lb3d230cb",
    ),
    "ENG2": LeagueInfo(
        name="Championship",
        sofascore_tournament_id=18,
        clubelo_league="ENG-Championship",
        worldelo_slug="England",
        country="England",
        continent="Europe",
        reep_competition_id="reep_l5bdd9936",
    ),
    "ESP1": LeagueInfo(
        name="La Liga",
        sofascore_tournament_id=8,
        clubelo_league="ESP-La Liga",
        worldelo_slug="Spain",
        country="Spain",
        continent="Europe",
        reep_competition_id="reep_ld517cd4f",
    ),
    "GER1": LeagueInfo(
        name="Bundesliga",
        sofascore_tournament_id=35,
        clubelo_league="GER-Bundesliga",
        worldelo_slug="Germany",
        country="Germany",
        continent="Europe",
        reep_competition_id="reep_l15d39c26",
    ),
    "ITA1": LeagueInfo(
        name="Serie A",
        sofascore_tournament_id=23,
        clubelo_league="ITA-Serie A",
        worldelo_slug="Italy",
        country="Italy",
        continent="Europe",
        reep_competition_id="reep_l2176f46e",
    ),
    "FRA1": LeagueInfo(
        name="Ligue 1",
        sofascore_tournament_id=34,
        clubelo_league="FRA-Ligue 1",
        worldelo_slug="France",
        country="France",
        continent="Europe",
        reep_competition_id="reep_l55ddfaa7",
    ),
    "NED1": LeagueInfo(
        name="Eredivisie",
        sofascore_tournament_id=37,
        clubelo_league="NED-Eredivisie",
        worldelo_slug="Netherlands",
        country="Netherlands",
        continent="Europe",
        reep_competition_id="reep_lfe2d9011",
    ),
    "POR1": LeagueInfo(
        name="Primeira Liga",
        sofascore_tournament_id=238,
        clubelo_league="POR-Liga Portugal",
        worldelo_slug="Portugal",
        country="Portugal",
        continent="Europe",
        reep_competition_id="reep_l2e7b8b73",
    ),
    "BEL1": LeagueInfo(
        name="Belgian Pro League",
        sofascore_tournament_id=38,
        clubelo_league="BEL-First Division A",
        worldelo_slug="Belgium",
        country="Belgium",
        continent="Europe",
        reep_competition_id="reep_l64eccec3",
    ),
    "TUR1": LeagueInfo(
        name="Super Lig",
        sofascore_tournament_id=52,
        clubelo_league="TUR-Süper Lig",
        worldelo_slug="Turkey",
        country="Turkey",
        continent="Europe",
        reep_competition_id="reep_la41838d0",
    ),
    # ── Additional European leagues (in ClubElo) ─────────────────────────
    "SCO1": LeagueInfo(
        name="Scottish Premiership",
        sofascore_tournament_id=36,
        clubelo_league="SCO-Premiership",
        worldelo_slug="Scotland",
        country="Scotland",
        continent="Europe",
        reep_competition_id="reep_l8cc7cd9a",
    ),
    "AUT1": LeagueInfo(
        name="Austrian Bundesliga",
        sofascore_tournament_id=45,
        clubelo_league="AUT-Bundesliga",
        worldelo_slug="Austria",
        country="Austria",
        continent="Europe",
    ),
    "SUI1": LeagueInfo(
        name="Swiss Super League",
        sofascore_tournament_id=215,
        clubelo_league="SUI-Super League",
        worldelo_slug="Switzerland",
        country="Switzerland",
        continent="Europe",
        reep_competition_id="reep_lb7870013",
    ),
    "GRE1": LeagueInfo(
        name="Greek Super League",
        sofascore_tournament_id=65,
        clubelo_league="GRE-Super League",
        worldelo_slug="Greece",
        country="Greece",
        continent="Europe",
        reep_competition_id="reep_l2da1200b",
    ),
    "CZE1": LeagueInfo(
        name="Czech First League",
        sofascore_tournament_id=163,
        clubelo_league="CZE-First League",
        worldelo_slug="Czech_Republic",
        country="Czech Republic",
        continent="Europe",
        reep_competition_id="reep_l736a5eba",
    ),
    "DEN1": LeagueInfo(
        name="Danish Superliga",
        sofascore_tournament_id=40,
        clubelo_league="DEN-Superliga",
        worldelo_slug="Denmark",
        country="Denmark",
        continent="Europe",
        reep_competition_id="reep_lba72483a",
    ),
    "CRO1": LeagueInfo(
        name="Croatian 1. HNL",
        sofascore_tournament_id=72,
        clubelo_league="CRO-1. HNL",
        worldelo_slug="Croatia",
        country="Croatia",
        continent="Europe",
        reep_competition_id="reep_l0077ca9a",
    ),
    "SER1": LeagueInfo(
        name="Serbian Super Liga",
        sofascore_tournament_id=449,
        clubelo_league="SER-Super Liga",
        worldelo_slug="Serbia",
        country="Serbia",
        continent="Europe",
        reep_competition_id="reep_l6a8606a3",
    ),
    "NOR1": LeagueInfo(
        name="Norwegian Eliteserien",
        sofascore_tournament_id=60,
        clubelo_league="NOR-Eliteserien",
        worldelo_slug="Norway",
        country="Norway",
        continent="Europe",
        reep_competition_id="reep_la4c03fe3",
    ),
    "SWE1": LeagueInfo(
        name="Swedish Allsvenskan",
        sofascore_tournament_id=56,
        clubelo_league="SWE-Allsvenskan",
        worldelo_slug="Sweden",
        country="Sweden",
        continent="Europe",
        reep_competition_id="reep_l238e6416",
    ),
    "POL1": LeagueInfo(
        name="Polish Ekstraklasa",
        sofascore_tournament_id=202,
        clubelo_league="POL-Ekstraklasa",
        worldelo_slug="Poland",
        country="Poland",
        continent="Europe",
        reep_competition_id="reep_lc46c3862",
    ),
    "ROM1": LeagueInfo(
        name="Romanian Liga I",
        sofascore_tournament_id=172,
        clubelo_league="ROM-Liga I",
        worldelo_slug="Romania",
        country="Romania",
        continent="Europe",
    ),
    "UKR1": LeagueInfo(
        name="Ukrainian Premier League",
        sofascore_tournament_id=218,
        clubelo_league="UKR-Premier League",
        worldelo_slug="Ukraine",
        country="Ukraine",
        continent="Europe",
    ),
    "RUS1": LeagueInfo(
        name="Russian Premier League",
        sofascore_tournament_id=203,
        clubelo_league="RUS-Premier League",
        worldelo_slug="Russia",
        country="Russia",
        continent="Europe",
    ),
    "BUL1": LeagueInfo(
        name="Bulgarian First League",
        sofascore_tournament_id=75,
        clubelo_league="BUL-First League",
        worldelo_slug="Bulgaria",
        country="Bulgaria",
        continent="Europe",
    ),
    "HUN1": LeagueInfo(
        name="Hungarian NB I",
        sofascore_tournament_id=153,
        clubelo_league="HUN-NB I",
        worldelo_slug="Hungary",
        country="Hungary",
        continent="Europe",
    ),
    "CYP1": LeagueInfo(
        name="Cypriot First Division",
        sofascore_tournament_id=388,
        clubelo_league="CYP-First Division",
        worldelo_slug="Cyprus",
        country="Cyprus",
        continent="Europe",
    ),
    "FIN1": LeagueInfo(
        name="Finnish Veikkausliiga",
        sofascore_tournament_id=285,
        clubelo_league="FIN-Veikkausliiga",
        worldelo_slug="Finland",
        country="Finland",
        continent="Europe",
    ),
    "SVK1": LeagueInfo(
        name="Slovak Super Liga",
        sofascore_tournament_id=204,
        clubelo_league="SVK-Super Liga",
        worldelo_slug="Slovakia",
        country="Slovakia",
        continent="Europe",
    ),
    "SLO1": LeagueInfo(
        name="Slovenian PrvaLiga",
        sofascore_tournament_id=207,
        clubelo_league="SLO-PrvaLiga",
        worldelo_slug="Slovenia",
        country="Slovenia",
        continent="Europe",
    ),
    "BOS1": LeagueInfo(
        name="Bosnian Premier Liga",
        sofascore_tournament_id=190,
        clubelo_league="BOS-Premier Liga",
        worldelo_slug="Bosnia-Herzegovina",
        country="Bosnia and Herzegovina",
        continent="Europe",
    ),
    "ISR1": LeagueInfo(
        name="Israeli Premier League",
        sofascore_tournament_id=266,
        clubelo_league="ISR-Premier League",
        worldelo_slug="Israel",
        country="Israel",
        continent="Europe",
    ),
    "KAZ1": LeagueInfo(
        name="Kazakh Premier League",
        sofascore_tournament_id=356,
        clubelo_league="KAZ-Premier League",
        worldelo_slug="Kazakhstan",
        country="Kazakhstan",
        continent="Europe",
    ),
    "ISL1": LeagueInfo(
        name="Icelandic Úrvalsdeild",
        sofascore_tournament_id=350,
        clubelo_league="ISL-Úrvalsdeild",
        worldelo_slug="Iceland",
        country="Iceland",
        continent="Europe",
    ),
    "IRL1": LeagueInfo(
        name="Irish Premier Division",
        sofascore_tournament_id=373,
        clubelo_league="IRL-Premier Division",
        worldelo_slug="Ireland",
        country="Ireland",
        continent="Europe",
    ),
    "WAL1": LeagueInfo(
        name="Welsh Premier League",
        sofascore_tournament_id=308,
        clubelo_league="WAL-Premier League",
        worldelo_slug="Wales",
        country="Wales",
        continent="Europe",
    ),
    "GEO1": LeagueInfo(
        name="Georgian Erovnuli Liga",
        sofascore_tournament_id=313,
        clubelo_league="GEO-Erovnuli Liga",
        worldelo_slug="Georgia",
        country="Georgia",
        continent="Europe",
    ),
    "ESP2": LeagueInfo(
        name="La Liga 2",
        sofascore_tournament_id=492,
        clubelo_league="ESP-Segunda División",
        worldelo_slug="Spain",
        country="Spain",
        continent="Europe",
        reep_competition_id="reep_lb9ce8b5c",
    ),
    "GER2": LeagueInfo(
        name="2. Bundesliga",
        sofascore_tournament_id=44,
        clubelo_league="GER-2. Bundesliga",
        worldelo_slug="Germany",
        country="Germany",
        continent="Europe",
        reep_competition_id="reep_la13477c1",
    ),
    "ITA2": LeagueInfo(
        name="Serie B",
        sofascore_tournament_id=53,
        clubelo_league="ITA-Serie B",
        worldelo_slug="Italy",
        country="Italy",
        continent="Europe",
        reep_competition_id="reep_l6964918f",
    ),
    "FRA2": LeagueInfo(
        name="Ligue 2",
        sofascore_tournament_id=182,
        clubelo_league="FRA-Ligue 2",
        worldelo_slug="France",
        country="France",
        continent="Europe",
        reep_competition_id="reep_l4e26931d",
    ),
    # ── South America ────────────────────────────────────────────────────
    "BRA1": LeagueInfo(
        name="Brasileirao Serie A",
        sofascore_tournament_id=325,
        clubelo_league=None,
        worldelo_slug="Brazil",
        country="Brazil",
        continent="South America",
        reep_competition_id="reep_l34be2d9a",
    ),
    "BRA2": LeagueInfo(
        name="Brasileirao Serie B",
        sofascore_tournament_id=390,
        clubelo_league=None,
        worldelo_slug="Brazil",
        country="Brazil",
        continent="South America",
        reep_competition_id="reep_leeb62459",
    ),
    "ARG1": LeagueInfo(
        name="Argentine Primera Division",
        sofascore_tournament_id=155,
        clubelo_league=None,
        worldelo_slug="Argentina",
        country="Argentina",
        continent="South America",
        reep_competition_id="reep_l1ca6d473",
    ),
    "COL1": LeagueInfo(
        name="Colombian Primera A",
        sofascore_tournament_id=109,
        clubelo_league=None,
        worldelo_slug="Colombia",
        country="Colombia",
        continent="South America",
        reep_competition_id="reep_l2f7dcd96",
    ),
    "CHI1": LeagueInfo(
        name="Chilean Primera Division",
        sofascore_tournament_id=19,
        clubelo_league=None,
        worldelo_slug="Chile",
        country="Chile",
        continent="South America",
        reep_competition_id="reep_l6bbcd589",
    ),
    "URU1": LeagueInfo(
        name="Uruguayan Primera Division",
        sofascore_tournament_id=1064,
        clubelo_league=None,
        worldelo_slug="Uruguay",
        country="Uruguay",
        continent="South America",
        reep_competition_id="reep_l7eddf211",
    ),
    "ECU1": LeagueInfo(
        name="Ecuadorian Serie A",
        sofascore_tournament_id=240,
        clubelo_league=None,
        worldelo_slug="Ecuador",
        country="Ecuador",
        continent="South America",
        reep_competition_id="reep_l2cf8bed4",
    ),
    # ── Other ────────────────────────────────────────────────────────────
    "USA1": LeagueInfo(
        name="MLS",
        sofascore_tournament_id=242,
        clubelo_league=None,
        worldelo_slug="United_States",
        country="United States",
        continent="North America",
        reep_competition_id="reep_l9c9cdd75",
    ),
    "SAU1": LeagueInfo(
        name="Saudi Pro League",
        sofascore_tournament_id=955,
        clubelo_league=None,
        worldelo_slug="Saudi_Arabia",
        country="Saudi Arabia",
        continent="Asia",
        reep_competition_id="reep_l07e8fd55",
    ),
    "JPN1": LeagueInfo(
        name="J-League",
        sofascore_tournament_id=196,
        clubelo_league=None,
        worldelo_slug="Japan",
        country="Japan",
        continent="Asia",
        reep_competition_id="reep_l4d609984",
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


# ── REEP integration ────────────────────────────────────────────────────────


def get_by_reep_competition_id(reep_id: str) -> Optional[LeagueInfo]:
    """Look up league info by REEP competition ID."""
    for info in LEAGUES.values():
        if info.reep_competition_id == reep_id:
            return info
    return None


def enrich_from_reep(league_code: str) -> Optional[Dict[str, Any]]:
    """Return the full REEP competition row for a league code.

    Uses the ``reep_competition_id`` field in the league registry to fetch
    the full competition metadata from ``competitions.csv``, including
    provider keys (``key_fbref``, ``key_opta``, ``key_fotmob``, etc.).

    Returns *None* if the league has no REEP competition link or the
    REEP data is unavailable.
    """
    info = LEAGUES.get(league_code)
    if info is None or info.reep_competition_id is None:
        return None
    try:
        from backend.data.reep_registry import lookup_competition

        return lookup_competition(info.reep_competition_id)
    except Exception:
        return None


def get_seasons(league_code: str) -> List[Dict[str, Any]]:
    """Return all REEP seasons for a league code.

    Uses the ``reep_competition_id`` to fetch linked season rows from
    ``seasons.csv``.  Returns an empty list on miss or if REEP is
    unavailable.
    """
    info = LEAGUES.get(league_code)
    if info is None or info.reep_competition_id is None:
        return []
    try:
        from backend.data.reep_registry import get_seasons_for_competition

        return get_seasons_for_competition(info.reep_competition_id)
    except Exception:
        return []
