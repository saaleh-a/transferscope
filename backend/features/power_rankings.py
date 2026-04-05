"""Dynamic league Power Rankings from mean team Elo per league per day.

Normalize all clubs 0-100 globally daily.
Store per-league mean, std, and percentile bands (10th, 25th, 50th, 75th, 90th).
Compute relative_ability = team_score - league_mean_score.

**Inference path (Option B hybrid)**: When ``query_date`` is today/None,
Opta Power Rankings (0-100 natively) are used for normalized scores.  ClubElo
provides the ``raw_elo`` on the ~1000-2100 scale the model was trained on.
Teams not covered by ClubElo get a linear rescale from Opta's 0-100.

**Training path**: Historical dates always use ClubElo / WorldFootballElo
(Opta has no historical API).
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

# Prefer rapidfuzz for token_sort_ratio (faster, no GPL license issues).
# Falls back to SequenceMatcher (stdlib difflib) if rapidfuzz is not installed.
try:
    from rapidfuzz import fuzz as _rfuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False


_ONE_DAY = 86400

# ── Significant-token overlap helpers (Fix C) ────────────────────────────────
# Common short/generic tokens that should be ignored when comparing team names.
_IGNORE_TOKENS = frozenset({
    "fc", "afc", "sc", "cf", "sv", "fk", "sk", "ac", "ss", "rc", "as",
    "cd", "sl", "vv", "kv", "bk", "if", "united", "city", "town",
    "rovers", "wanderers", "athletic", "real", "club", "sporting",
})

# Threshold above which a fuzzy match is accepted even without token overlap
# (handles abbreviation pairs like "Manchester United" ↔ "Man United").
_HIGH_SIMILARITY_BYPASS = 0.85


def _significant_tokens(name: str) -> set:
    """Return the set of lowercased tokens in *name* that are ≥ 4 chars
    and not in the ignore list."""
    return {
        t.lower()
        for t in name.split()
        if len(t) >= 4 and t.lower() not in _IGNORE_TOKENS
    }


def _has_token_overlap(query: str, match: str) -> bool:
    """Return True if *query* and *match* share at least one significant token."""
    return bool(_significant_tokens(query) & _significant_tokens(match))


# ── Team alias map for known Sofascore↔REEP discrepancies (Problem 2) ────────
# Value = alternative name to attempt, or None if the team is confirmed absent.
TEAM_ALIASES: Dict[str, Optional[str]] = {
    # English Championship
    "Nottingham Forest": "Nottingham Forest",
    "Hull City": None,
    "Cardiff City": None,
    "Sheffield Wednesday": None,
    "Huddersfield Town": None,
    "Luton Town": None,
    "West Bromwich Albion": "West Brom",
    # Belgian
    "Royale Union Saint-Gilloise": "Union SG",
    "Oud-Heverlee Leuven": None,
    "Sint-Truidense VV": "St Truiden",
    "KAS Eupen": None,
    "KV Kortrijk": None,
    # Portuguese
    "Sporting Braga": "Braga",
    # Scottish
    "Ross County": None,
    "St. Johnstone": None,
    # Italian
    "Genoa": "Genoa CFC",
    "Lecce": "US Lecce",
    # French
    "Olympique Lyonnais": "Lyon",
}

# Mid-table fallback ranking for teams confirmed absent from all Elo sources.
# ClubElo ratings typically range ~1000-2100; 1500 is the median across ~526 top
# clubs (by design Elo 1500 ≈ average).
DEFAULT_RANKING = 1500.0

# ── Opta→ClubElo raw-Elo rescale (Option B) ──────────────────────────────────
# When using Opta for normalized scores at inference but ClubElo for raw_elo,
# teams not covered by ClubElo need a linear rescale from Opta's 0-100 into
# the ~1000-2100 range the model was trained on.
# Formula:  pseudo_raw_elo = opta_score / 100 * (_OPTA_ELO_MAX - _OPTA_ELO_MIN) + _OPTA_ELO_MIN
_OPTA_ELO_MIN = 1000.0
_OPTA_ELO_MAX = 2100.0

# Whether to prefer Opta for current-day rankings.  Set to False to disable
# the Opta path entirely (e.g. if the client is broken).
# Re-enabled: opta_client now extracts data directly from index.js (no Selenium).
_USE_OPTA_FOR_INFERENCE = True

# ── In-process rankings cache (keyed by date ISO string) ─────────────────────
# Avoids repeated diskcache lookups during training (one lookup per unique date
# instead of one per player call).  Much cheaper than hitting SQLite for every
# sample in the build loop.
_rankings_in_process_cache: Dict[str, tuple] = {}

# ── Opta league rating in-process cache ──────────────────────────────────────
# {league_name_lower: rating (0-100)}.  Populated lazily on first call to
# get_league_opta_rating().  Avoids re-fetching league-meta.json on every
# build_training_sample() call across a training run.
_opta_league_map: Optional[Dict[str, float]] = None

# ── Opta team alias map (short_name / club_name → canonical team name) ────────
# Built lazily from the Opta team list.  Allows resolving "Man City" or "MCFC"
# to "Manchester City" without fuzzy matching overhead.
_opta_alias_map: Optional[Dict[str, str]] = None

# ── Opta team→league rating flat index ───────────────────────────────────────
# {team_name_lower: league_rating}.  Avoids iterating 14k Opta teams on every
# get_league_opta_rating() call during training (was O(14k) per sample).
# Populated lazily alongside _opta_alias_map; same lifetime.
_opta_team_league_map: Optional[Dict[str, float]] = None

# ── LEAGUES code → Opta rating resolved cache ─────────────────────────────────
# Pre-resolved on first call to get_league_opta_rating(); avoids running the
# fuzzy SequenceMatcher loop over 446 leagues on every training sample.
_league_code_opta_rating_cache: Dict[str, float] = {}

# ── ClubElo canonical name aliases ───────────────────────────────────────────
# Maps incoming Sofascore/Opta display names → names closer to ClubElo's
# canonical form so the existing exact + fuzzy pipeline resolves them cleanly.
# Checked BEFORE fuzzy matching.
CLUBELO_ALIASES: Dict[str, str] = {
    "New York Red Bulls": "New York RB",
    "New York Red Bulls II": "New York RB",
    "Heart of Midlothian": "Hearts",
    "Royale Union Saint-Gilloise": "Union SG",
    "TSG Hoffenheim": "Hoffenheim",
    "FC Porto": "Porto",
    "AFC Ajax": "Ajax",
    "Bayer 04 Leverkusen": "Bayer Leverkusen",
    "FC Bayern München": "Bayern München",
    "1. FSV Mainz 05": "Mainz 05",
    "1. FC Heidenheim": "Heidenheim",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Bournemouth": "AFC Bournemouth",
    "Stade Rennais": "Rennes",
    "RC Strasbourg": "Strasbourg",
    "AS Monaco": "Monaco",
    "RSC Anderlecht": "Anderlecht",
    "Olympique de Marseille": "Olympique Marseille",
    "Atletico Madrid": "Atlético",
    "Deportivo La Coruña": "Deportivo La Coruña",
    "Celta Vigo": "Celta de Vigo",
}

# ── Youth / reserve suffix stripping ─────────────────────────────────────────
# Before fuzzy matching, strip suffixes that identify reserve/youth teams.
# The parent club's Elo is used as a proxy (better than None).
_YOUTH_SUFFIX_RE = re.compile(
    r"\s+(U1[5-9]|U2[0-3]|U\d+|II|2nd|2|B|Jong|Youth|Reserve|Reserves"
    r"|Academy|Atlètic|Atlético\s*B|2ème|FC\s*2)$",
    re.IGNORECASE,
)

# ── ClubElo per-team JSON API (replacement for Opta inference path) ───────────
# http://api.club-elo.com/api/getTeamStats/?team={name} returns JSON with Elo.
# Cached per-team per-day to avoid hammering the API per-player during build.
_CLUBELO_JSON_BASE = "http://api.club-elo.com"
_CLUBELO_JSON_TTL = 86400  # 24 h

try:
    from curl_cffi.requests import Session as _CurlSessionCls
    _clubelo_http = _CurlSessionCls(impersonate="chrome110")
except Exception:
    import requests as _clubelo_http  # type: ignore


def _strip_youth_suffix(name: str) -> str:
    """Strip youth/reserve suffixes from a team name (e.g. 'Chelsea U21' → 'Chelsea').

    Returns the stripped name if a suffix was found, otherwise returns *name*
    unchanged.  Only one suffix is stripped to avoid over-stripping.
    """
    return _YOUTH_SUFFIX_RE.sub("", name).strip()


def _fetch_clubelo_team_elo(team_name: str) -> Optional[float]:
    """Fetch current Elo for a single team via the club-elo.com JSON API.

    Endpoint: GET http://api.club-elo.com/api/getTeamStats/?team={team_name}
    Response: JSON object with an ``Elo`` field (float, ~1000-2100 scale).

    Results are cached per team per day so we never hit the endpoint more than
    once per team per training run.  Returns ``None`` on any failure.
    """
    cache_key = cache.make_key(
        "clubelo_json_elo", team_name, date.today().isoformat()
    )
    cached = cache.get(cache_key, max_age=_CLUBELO_JSON_TTL)
    if cached is not None:
        return float(cached)

    url = f"{_CLUBELO_JSON_BASE}/api/getTeamStats/?team={team_name}"
    try:
        resp = _clubelo_http.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            elo = data.get("Elo") or data.get("elo") or data.get("ELO")
            if elo is not None:
                elo_float = float(elo)
                cache.set(cache_key, elo_float)
                return elo_float
    except Exception as exc:
        _log.debug("ClubElo JSON fetch failed for '%s': %s", team_name, exc)
    return None


def _opta_score_to_raw_elo(opta_score: float) -> float:
    """Linearly rescale an Opta 0-100 score to the ~1000-2100 raw Elo range."""
    return opta_score / 100.0 * (_OPTA_ELO_MAX - _OPTA_ELO_MIN) + _OPTA_ELO_MIN


# ── ClubElo → Sofascore direct name mapping ──────────────────────────────────
# ClubElo uses abbreviated team names (e.g. "PSG", "ManCity") while Sofascore
# uses full display names (e.g. "Paris Saint-Germain", "Manchester City").
# This mapping canonicalizes ClubElo names at data-load time so that the
# ``teams`` dict keys match the names that come from the Sofascore dropdowns.
# Fuzzy matching is kept as a safety net for any unmapped teams.
# Exact (country.casefold(), domestic_league.casefold()) → league_code
# Built from actual Opta API field values (confirmed by inspection).
# Using both fields prevents "Premier League" (Wales/Belarus/Armenia) from
# polluting ENG1.  Keys are .casefold() so accent-variants match naturally.
_OPTA_COUNTRY_LEAGUE_TO_CODE: Dict[str, str] = {
    # ── Tier 1 ───────────────────────────────────────────────────────────
    ("england",       "premier league"):    "ENG1",
    ("spain",         "primera división"):  "ESP1",
    ("spain",         "primera division"):  "ESP1",  # no accent fallback
    ("germany",       "bundesliga"):        "GER1",
    ("italy",         "serie a"):           "ITA1",
    ("france",        "ligue 1"):           "FRA1",
    ("netherlands",   "eredivisie"):        "NED1",
    ("portugal",      "primeira liga"):     "POR1",
    ("belgium",       "first division a"):  "BEL1",
    ("türkiye",       "süper lig"):         "TUR1",
    ("turkey",        "süper lig"):         "TUR1",  # in case country varies
    ("turkey",        "super lig"):         "TUR1",
    ("scotland",      "premiership"):       "SCO1",
    ("austria",       "bundesliga"):        "AUT1",
    ("switzerland",   "super league"):      "SUI1",
    ("greece",        "super league 1"):    "GRE1",
    ("czechia",       "czech liga"):        "CZE1",
    ("czech republic","first league"):      "CZE1",
    ("denmark",       "superliga"):         "DEN1",
    ("croatia",       "hnl"):               "CRO1",
    ("croatia",       "1. hnl"):            "CRO1",
    ("serbia",        "super liga"):        "SER1",
    ("norway",        "eliteserien"):       "NOR1",
    ("sweden",        "allsvenskan"):       "SWE1",
    ("poland",        "ekstraklasa"):       "POL1",
    ("romania",       "liga i"):            "ROM1",
    ("ukraine",       "premier league"):    "UKR1",
    ("russia",        "premier league"):    "RUS1",
    ("bulgaria",      "first league"):      "BUL1",
    ("hungary",       "nb i"):              "HUN1",
    ("cyprus",        "1. division"):       "CYP1",
    ("finland",       "veikkausliiga"):     "FIN1",
    ("slovakia",      "1. liga"):           "SVK1",
    ("slovenia",      "1. snl"):            "SVN1",
    ("israel",        "premier league"):    "ISR1",
    ("kazakhstan",    "premier league"):    "KAZ1",
    ("azerbaijan",    "premyer liqa"):      "AZE1",
    # ── Tier 2 ───────────────────────────────────────────────────────────
    ("england",       "championship"):      "ENG2",
    ("spain",         "segunda división"):  "ESP2",
    ("spain",         "segunda division"):  "ESP2",
    ("germany",       "2. bundesliga"):     "GER2",
    ("italy",         "serie b"):           "ITA2",
    ("france",        "ligue 2"):           "FRA2",
    ("netherlands",   "eerste divisie"):    "NED2",
    ("portugal",      "segunda liga"):      "POR2",
}

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
    "RapidVienna": "SK Rapid Wien",
    "Rapid": "SK Rapid Wien",
    "AustriaVienna": "FK Austria Wien",
    "Sturm": "SK Sturm Graz",
    "SturmGraz": "SK Sturm Graz",
    "LASK": "LASK",
    "Wolfsberg": "Wolfsberger AC",
    "WAC": "Wolfsberger AC",
    "Hartberg": "TSV Hartberg",
    "Altach": "SCR Altach",
    "Rheindorf": "SCR Altach",
    "Klagenfurt": "SK Austria Klagenfurt",
    "Tirol": "WSG Tirol",
    "WSGTirol": "WSG Tirol",
    "Blau-Weiss": "FC Blau-Weiß Linz",
    "BWLinz": "FC Blau-Weiß Linz",
    # Switzerland
    "YoungBoys": "BSC Young Boys",
    "Basel": "FC Basel 1893",
    "Zurich": "FC Zürich",
    "Lugano": "FC Lugano",
    "Servette": "Servette FC",
    "StGallen": "FC St. Gallen 1879",
    "Luzern": "FC Luzern",
    "Sion": "FC Sion",
    "Winterthur": "FC Winterthur",
    "GCZurich": "Grasshopper Club Zürich",
    "Grasshoppers": "Grasshopper Club Zürich",
    "Lausanne": "FC Lausanne-Sport",
    "Yverdon": "Yverdon Sport FC",
    # Greece
    "Olympiacos": "Olympiacos FC",
    "Olympiakos": "Olympiacos FC",
    "Panathinaikos": "Panathinaikos FC",
    "PAOK": "PAOK FC",
    "AEK": "AEK Athens FC",
    "AEKAthens": "AEK Athens FC",
    "Aris": "Aris Thessaloniki FC",
    "ArisThessaloniki": "Aris Thessaloniki FC",
    "OFICrete": "OFI Crete",
    "Volos": "Volos NFC",
    "Asteras": "Asteras Tripolis FC",
    "Atromitos": "Atromitos FC",
    "Panetolikos": "Panetolikos FC",
    "Lamia": "PAS Lamia 1964",
    # Czech Republic
    "SpartaPrague": "Sparta Prague",
    "SlaviaPrague": "Slavia Prague",
    "Slavia": "Slavia Prague",
    "PlzenViktoria": "Viktoria Plzeň",
    "Viktoria": "Viktoria Plzeň",
    "Plzen": "Viktoria Plzeň",
    "BanikOstrava": "FC Baník Ostrava",
    "Ostrava": "FC Baník Ostrava",
    "SlovackoBrod": "1. FC Slovácko",
    "Slovacko": "1. FC Slovácko",
    "Jablonec": "FK Jablonec",
    "Mlada": "FK Mladá Boleslav",
    "MladaBoleslav": "FK Mladá Boleslav",
    "Liberec": "FC Slovan Liberec",
    "Hradec": "FC Hradec Králové",
    "HradecKralove": "FC Hradec Králové",
    "Bohemians1905": "Bohemians 1905",
    "Bohemians": "Bohemians 1905",
    "Teplice": "FK Teplice",
    # Denmark
    "Copenhagen": "FC Copenhagen",
    "FCCopenhagen": "FC Copenhagen",
    "Midtjylland": "FC Midtjylland",
    "FCMidtjylland": "FC Midtjylland",
    "Brondby": "Brøndby IF",
    "BrondbyIF": "Brøndby IF",
    "Nordsjaelland": "FC Nordsjælland",
    "FCNordsjaelland": "FC Nordsjælland",
    "AarhusBold": "Aarhus GF",
    "AGF": "Aarhus GF",
    "Aarhus": "Aarhus GF",
    "Silkeborg": "Silkeborg IF",
    "Randers": "Randers FC",
    "Viborg": "Viborg FF",
    "Lyngby": "Lyngby BK",
    "OdenseBK": "Odense Boldklub",
    "Odense": "Odense Boldklub",
    "Aalborg": "AaB",
    "AalborgBK": "AaB",
    # Croatia
    "DinamoZagreb": "GNK Dinamo Zagreb",
    "HajdukSplit": "HNK Hajduk Split",
    "Hajduk": "HNK Hajduk Split",
    "Rijeka": "HNK Rijeka",
    "Osijek": "NK Osijek",
    "Lokomotiva": "NK Lokomotiva Zagreb",
    "Gorica": "HNK Gorica",
    "Varazdin": "NK Varaždin",
    "Istra1961": "NK Istra 1961",
    # Serbia
    "CrvenaZvezda": "FK Crvena zvezda",
    "RedStarBelgrade": "FK Crvena zvezda",
    "RedStar": "FK Crvena zvezda",
    "Partizan": "FK Partizan",
    "PartizanBelgrade": "FK Partizan",
    "Vojvodina": "FK Vojvodina",
    "Cukaricki": "FK Čukarički",
    "TSC": "FK TSC Bačka Topola",
    "Backa": "FK TSC Bačka Topola",
    # Norway
    "Bodo": "FK Bodø/Glimt",
    "BodoGlimt": "FK Bodø/Glimt",
    "Rosenborg": "Rosenborg BK",
    "Molde": "Molde FK",
    "Viking": "Viking FK",
    "Brann": "SK Brann",
    "SKBrann": "SK Brann",
    "Lillestrom": "Lillestrøm SK",
    "ValerengaOslo": "Vålerenga Fotball",
    "Valerenga": "Vålerenga Fotball",
    "Tromso": "Tromsø IL",
    "Stromsgodset": "Strømsgodset IF",
    "Sarpsborg": "Sarpsborg 08 FF",
    "HamKam": "Hamarkameratene",
    # Sweden
    "Malmo": "Malmö FF",
    "MalmoFF": "Malmö FF",
    "AIK": "AIK",
    "Djurgarden": "Djurgårdens IF",
    "DjurgardenIF": "Djurgårdens IF",
    "IFElfsborg": "IF Elfsborg",
    "Elfsborg": "IF Elfsborg",
    "Hacken": "BK Häcken",
    "BKHacken": "BK Häcken",
    "Hammarby": "Hammarby IF",
    "HammarbyIF": "Hammarby IF",
    "IFKGoteborg": "IFK Göteborg",
    "Goteborg": "IFK Göteborg",
    "Norrkoping": "IFK Norrköping",
    "IFKNorrkoping": "IFK Norrköping",
    "Sirius": "IK Sirius",
    "Kalmar": "Kalmar FF",
    "Varberg": "Varbergs BoIS FC",
    "Mjallby": "Mjällby AIF",
    # Poland
    "Legia": "Legia Warsaw",
    "LegiaWarsaw": "Legia Warsaw",
    "LechPoznan": "Lech Poznań",
    "Lech": "Lech Poznań",
    "Rakow": "Raków Częstochowa",
    "RakowCzestochowa": "Raków Częstochowa",
    "PogonSzczecin": "Pogoń Szczecin",
    "Pogon": "Pogoń Szczecin",
    "Jagiellonia": "Jagiellonia Białystok",
    "JagielloniaBialystok": "Jagiellonia Białystok",
    "Gornik": "Górnik Zabrze",
    "GornikZabrze": "Górnik Zabrze",
    "SlaAskWroclaw": "Śląsk Wrocław",
    "Slask": "Śląsk Wrocław",
    "Wisla": "Wisła Kraków",
    "WislaKrakow": "Wisła Kraków",
    "Piast": "Piast Gliwice",
    "PiastGliwice": "Piast Gliwice",
    "Cracovia": "MKS Cracovia",
    "Warta": "Warta Poznań",
    "WartaPoznan": "Warta Poznań",
    "Zaglebie": "Zagłębie Lubin",
    "ZaglebieLubin": "Zagłębie Lubin",
    # Romania
    "FCSB": "FCSB",
    "SteauaBucharest": "FCSB",
    "CFRCluj": "CFR Cluj",
    "CRaiova": "Universitatea Craiova",
    "UCraiova": "Universitatea Craiova",
    "Craiova": "Universitatea Craiova",
    "RapidBucharest": "Rapid București",
    "DinamoBucharest": "Dinamo București",
    "Sepsi": "Sepsi OSK",
    # Ukraine
    "ShakhtarDonetsk": "Shakhtar Donetsk",
    "Shakhtar": "Shakhtar Donetsk",
    "DynamoKyiv": "Dynamo Kyiv",
    "Dnipro1": "SC Dnipro-1",
    "Vorskla": "Vorskla Poltava",
    "Zorya": "Zorya Luhansk",
    "ZoryaLuhansk": "Zorya Luhansk",
    "Oleksandriya": "FC Oleksandriya",
    # Russia
    "ZenitStPetersburg": "Zenit St. Petersburg",
    "Zenit": "Zenit St. Petersburg",
    "Spartak": "Spartak Moscow",
    "SpartakMoscow": "Spartak Moscow",
    "CSKA": "CSKA Moscow",
    "CSKAMoscow": "CSKA Moscow",
    "LokomotivMoscow": "Lokomotiv Moscow",
    "Lokomotiv": "Lokomotiv Moscow",
    "Krasnodar": "FK Krasnodar",
    "Rostov": "FK Rostov",
    "Rubin": "Rubin Kazan",
    "RubinKazan": "Rubin Kazan",
    "SochiFC": "PFC Sochi",
    "Sochi": "PFC Sochi",
    "Akhmat": "Akhmat Grozny",
    # Bulgaria
    "LudogoretsRazgrad": "Ludogorets Razgrad",
    "Ludogorets": "Ludogorets Razgrad",
    "LevskiSofia": "PFC Levski Sofia",
    "Levski": "PFC Levski Sofia",
    "CSKASofia": "CSKA Sofia",
    "Botev": "Botev Plovdiv",
    "BotevPlovdiv": "Botev Plovdiv",
    "Lokomotiv1929": "Lokomotiv Plovdiv",
    "LokomotivPlovdiv": "Lokomotiv Plovdiv",
    # Hungary
    "Ferencvaros": "Ferencvárosi TC",
    "FerencvarosTC": "Ferencvárosi TC",
    "MOLFehérvár": "Fehérvár FC",
    "Fehervar": "Fehérvár FC",
    "Puskas": "Puskás Akadémia FC",
    "PuskasAkademia": "Puskás Akadémia FC",
    "Ujpest": "Újpest FC",
    "UjpestFC": "Újpest FC",
    "Kecskemeti": "Kecskeméti TE",
    "Debrecen": "Debreceni VSC",
    "DebrecenVSC": "Debreceni VSC",
    # Cyprus
    "APOEL": "APOEL Nicosia",
    "APOELNicosia": "APOEL Nicosia",
    "Omonia": "AC Omonia",
    "OmoniaNicosia": "AC Omonia",
    "AnoOrthosis": "Anorthosis Famagusta",
    "Anorthosis": "Anorthosis Famagusta",
    "AELLimassol": "AEL Limassol",
    "Apollon": "Apollon Limassol",
    "ApollonLimassol": "Apollon Limassol",
    "PaphosFC": "Pafos FC",
    "Pafos": "Pafos FC",
    # Finland
    "HJK": "HJK Helsinki",
    "HJKHelsinki": "HJK Helsinki",
    "KuPS": "Kuopion Palloseura",
    "KuopionPS": "Kuopion Palloseura",
    "IlvesT": "Tampereen Ilves",
    "Ilves": "Tampereen Ilves",
    "SJK": "SJK Seinäjoki",
    "SJKSeinajoki": "SJK Seinäjoki",
    "InterTurku": "FC Inter Turku",
    "Honka": "FC Honka",
    "Haka": "FC Haka",
    # Slovakia
    "SlovanBratislava": "ŠK Slovan Bratislava",
    "Bratislava": "ŠK Slovan Bratislava",
    "SpartakTrnava": "FC Spartak Trnava",
    "Trnava": "FC Spartak Trnava",
    "DACDunajska": "FC DAC 1904 Dunajská Streda",
    "DAC": "FC DAC 1904 Dunajská Streda",
    "Zilina": "MŠK Žilina",
    "MSKZilina": "MŠK Žilina",
    "Ruzomberok": "MFK Ružomberok",
    # Slovenia
    "Maribor": "NK Maribor",
    "NKMaribor": "NK Maribor",
    "Olimpija": "NK Olimpija Ljubljana",
    "OlimpijaLjubljana": "NK Olimpija Ljubljana",
    "Celje": "NK Celje",
    "Domzale": "NK Domžale",
    "NKDomzale": "NK Domžale",
    "Mura": "NŠ Mura",
    "NSMura": "NŠ Mura",
    "Koper": "FC Koper",
    # Bosnia and Herzegovina
    "ZeljeznicarSarajevo": "FK Željezničar Sarajevo",
    "Zeljeznicar": "FK Željezničar Sarajevo",
    "SarajevoFK": "FK Sarajevo",
    "FKSarajevo": "FK Sarajevo",
    "ZrinjskiMostar": "HŠK Zrinjski Mostar",
    "Zrinjski": "HŠK Zrinjski Mostar",
    "BoraczBanjaluka": "FK Borac Banja Luka",
    "Borac": "FK Borac Banja Luka",
    "TuzlaCity": "FK Tuzla City",
    "VelezMostar": "FK Velež Mostar",
    # Israel
    "MaccabiTelAviv": "Maccabi Tel Aviv FC",
    "MaccabiTA": "Maccabi Tel Aviv FC",
    "MaccabiHaifa": "Maccabi Haifa FC",
    "HapoeiBeerSheva": "Hapoel Be'er Sheva FC",
    "HapoelBeerSheva": "Hapoel Be'er Sheva FC",
    "BeitarJerusalem": "Beitar Jerusalem FC",
    "Beitar": "Beitar Jerusalem FC",
    "MaccabiNetanya": "Maccabi Netanya FC",
    "HapoelTelAviv": "Hapoel Tel Aviv FC",
    "HapoelTA": "Hapoel Tel Aviv FC",
    # Kazakhstan
    "Astana": "FC Astana",
    "FCAstana": "FC Astana",
    "Kairat": "FC Kairat",
    "KairatAlmaty": "FC Kairat",
    "Tobol": "FC Tobol",
    "TobolKostanay": "FC Tobol",
    "Ordabasy": "FC Ordabasy",
    "Aktobe": "FC Aktobe",
    # Iceland
    "Vikingur": "Víkingur Reykjavík",
    "VikingurReykjavik": "Víkingur Reykjavík",
    "Valur": "Valur Reykjavík",
    "Breidablik": "Breiðablik",
    "FH": "FH Hafnarfjörður",
    "FHHafnarfjordur": "FH Hafnarfjörður",
    "KR": "KR Reykjavík",
    "KRReykjavik": "KR Reykjavík",
    "Stjarnan": "Stjarnan FC",
    # Ireland
    "ShamrockRovers": "Shamrock Rovers FC",
    "Shamrock": "Shamrock Rovers FC",
    "Dundalk": "Dundalk FC",
    "DundalkFC": "Dundalk FC",
    "Bohemian": "Bohemian FC",
    "BohemianFC": "Bohemian FC",
    "StPatricksAthletic": "St Patrick's Athletic FC",
    "StPats": "St Patrick's Athletic FC",
    "Shelbourne": "Shelbourne FC",
    "ShelFC": "Shelbourne FC",
    "Derry": "Derry City FC",
    "DerryCity": "Derry City FC",
    "Drogheda": "Drogheda United FC",
    "Sligo": "Sligo Rovers FC",
    "SligoRovers": "Sligo Rovers FC",
    # Wales
    "TNS": "The New Saints FC",
    "NewSaints": "The New Saints FC",
    "Connah": "Connah's Quay Nomads FC",
    "ConnahsQuay": "Connah's Quay Nomads FC",
    "BarryTown": "Barry Town United FC",
    "Bala": "Bala Town FC",
    "BalaTown": "Bala Town FC",
    "Caernarfon": "Caernarfon Town FC",
    "Penybont": "Penybont FC",
    # Georgia
    "DinamoTbilisi": "FC Dinamo Tbilisi",
    "DinamoT": "FC Dinamo Tbilisi",
    "TorpedoKutaisi": "FC Torpedo Kutaisi",
    "Torpedo": "FC Torpedo Kutaisi",
    "DinamoBatumi": "FC Dinamo Batumi",
    "Saburtalo": "FC Saburtalo Tbilisi",
    "Dila": "FC Dila Gori",
    "DilaGori": "FC Dila Gori",
    # Portugal (expansion)
    "Porto": "FC Porto",
    "FCPorto": "FC Porto",
    "Benfica": "SL Benfica",
    "SLBenfica": "SL Benfica",
    "Braga": "SC Braga",
    "SCBraga": "SC Braga",
    "Vitoria": "Vitória SC",
    "VitoriaSC": "Vitória SC",
    "Guimaraes": "Vitória SC",
    "Famalicao": "FC Famalicão",
    "GilVicente": "Gil Vicente FC",
    "Boavista": "Boavista FC",
    "CasaPia": "Casa Pia AC",
    "Arouca": "FC Arouca",
    "RioAve": "Rio Ave FC",
    "Estoril": "Estoril Praia",
    "Moreirense": "Moreirense FC",
    "AVS": "AVS Futebol SAD",
    # Belgium (expansion)
    "Genk": "KRC Genk",
    "KRCGenk": "KRC Genk",
    "Antwerp": "Royal Antwerp FC",
    "RoyalAntwerp": "Royal Antwerp FC",
    "StandardLiege": "Standard Liège",
    "Standard": "Standard Liège",
    "Gent": "KAA Gent",
    "KAAGent": "KAA Gent",
    "UnionSG": "Royale Union Saint-Gilloise",
    "UnionStGilloise": "Royale Union Saint-Gilloise",
    "CercleBrugge": "Cercle Brugge KSV",
    "Mechelen": "KV Mechelen",
    "KVMechelen": "KV Mechelen",
    "Charleroi": "Sporting Charleroi",
    "SportingCharleroi": "Sporting Charleroi",
    "Westerlo": "KVC Westerlo",
    "Kortrijk": "KV Kortrijk",
    "OHLeuven": "OH Leuven",
    "Eupen": "KAS Eupen",
    # Netherlands (expansion)
    "Ajax": "AFC Ajax",
    "AFCAjax": "AFC Ajax",
    "Vitesse": "Vitesse",
    "Utrecht": "FC Utrecht",
    "FCUtrecht": "FC Utrecht",
    "Heerenveen": "SC Heerenveen",
    "SCHeerenveen": "SC Heerenveen",
    "Groningen": "FC Groningen",
    "NEC": "NEC Nijmegen",
    "NECNijmegen": "NEC Nijmegen",
    "SpartaRotterdam": "Sparta Rotterdam",
    "GoAheadEagles": "Go Ahead Eagles",
    "Fortuna": "Fortuna Sittard",
    "FortunaSittard": "Fortuna Sittard",
    "Heracles": "Heracles Almelo",
    "HeraclesAlmelo": "Heracles Almelo",
    "Waalwijk": "RKC Waalwijk",
    "Volendam": "FC Volendam",
    "Almere": "Almere City FC",
    "Excelsior": "Excelsior Rotterdam",
    "Willem": "Willem II",
    "WillemII": "Willem II",
    # Scotland (expansion)
    "AberdeenFC": "Aberdeen FC",
    "Aberdeen": "Aberdeen FC",
    "HeartsFC": "Heart of Midlothian FC",
    "Hearts": "Heart of Midlothian FC",
    "Hibernian": "Hibernian FC",
    "HibernianFC": "Hibernian FC",
    "Hibs": "Hibernian FC",
    "Dundee": "Dundee FC",
    "DundeeFC": "Dundee FC",
    "DundeeUtd": "Dundee United FC",
    "DundeeUnited": "Dundee United FC",
    "StMirren": "St Mirren FC",
    "StJohnstone": "St Johnstone FC",
    "Kilmarnock": "Kilmarnock FC",
    "KilmarnockFC": "Kilmarnock FC",
    "Ross": "Ross County FC",
    "RossCounty": "Ross County FC",
    "Motherwell": "Motherwell FC",
    "MotherwellFC": "Motherwell FC",
    "Livingston": "Livingston FC",
    # Turkey (expansion)
    "Trabzonspor": "Trabzonspor",
    "Basaksehir": "İstanbul Başakşehir FK",
    "IstanbulBasaksehir": "İstanbul Başakşehir FK",
    "Sivasspor": "Sivasspor",
    "Antalyaspor": "Antalyaspor",
    "Konyaspor": "Konyaspor",
    "Kasimpasa": "Kasımpaşa SK",
    "KasimpasaSK": "Kasımpaşa SK",
    "Alanyaspor": "Alanyaspor",
    "Rizespor": "Çaykur Rizespor",
    "CaykurRizespor": "Çaykur Rizespor",
    "Hatayspor": "Hatayspor",
    "Kayserispor": "Kayserispor",
    "Adana": "Adana Demirspor",
    "AdanaDemirspor": "Adana Demirspor",
    "Gaziantep": "Gaziantep FK",
    "GaziantepFK": "Gaziantep FK",
    "Eyupspor": "Eyüpspor",
    "Pendikspor": "Pendikspor",
    # ── ClubElo API names (space-containing, as returned by api.clubelo.com) ──
    # The keys above were built for old soccerdata format (no spaces).
    # These entries cover the actual HTTP API names so the lookup always hits.
    # England
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "West Ham": "West Ham United",
    "West Brom": "West Bromwich Albion",
    "Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield United",
    "Sheffield Weds": "Sheffield Wednesday",
    "Aston Villa": "Aston Villa",
    "Crystal Palace": "Crystal Palace",
    "Ipswich": "Ipswich Town",
    "Derby": "Derby County",
    "Birmingham": "Birmingham City",
    "Middlesbrough": "Middlesbrough",
    # France
    "Paris SG": "Paris Saint-Germain",
    "Paris FC": "Paris FC",
    "Saint-Etienne": "AS Saint-Étienne",
    "Le Havre": "Le Havre AC",
    # Germany
    "RB Leipzig": "RB Leipzig",
    "Union Berlin": "1. FC Union Berlin",
    "St Pauli": "FC St. Pauli",
    "Koeln": "1. FC Köln",
    "Hertha": "Hertha BSC",
    # Spain
    "Real Madrid": "Real Madrid",
    "Bilbao": "Athletic Club",
    "Sociedad": "Real Sociedad",
    "Celta": "Celta Vigo",
    "Rayo Vallecano": "Rayo Vallecano",
    "Sociedad B": "Real Sociedad B",
    # Norway
    "Bodoe Glimt": "FK Bodø/Glimt",
    # Belgium
    "St Gillis": "Royale Union Saint-Gilloise",
    "St Truiden": "Sint-Truidense VV",
    "Cercle Brugge": "Cercle Brugge KSV",
    # Netherlands
    "Alkmaar": "AZ Alkmaar",
    "Nijmegen": "NEC Nijmegen",
    "Sparta Rotterdam": "Sparta Rotterdam",
    "Go Ahead Eagles": "Go Ahead Eagles",
    "Sittard": "Fortuna Sittard",
    # Serbia
    "Crvena Zvezda": "FK Crvena zvezda",
    "Red Star": "FK Crvena zvezda",
    "Novi Sad": "FK Vojvodina",
    "Backa Topola": "FK TSC Bačka Topola",
    # Russia
    "Lok Moskva": "Lokomotiv Moscow",
    "CSKA Moskva": "CSKA Moscow",
    "Spartak Moskva": "Spartak Moscow",
    "Dynamo Moskva": "FC Dynamo Moscow",
    "Kryliya Sovetov": "Krylia Sovetov Samara",
    "FC Krasnodar": "FK Krasnodar",
    "Nizhny Novgorod": "FC Nizhny Novgorod",
    # Ukraine
    "Dynamo Kyiv": "Dynamo Kyiv",
    # Denmark
    "FC Kobenhavn": "FC Copenhagen",
    "Aarhus": "Aarhus GF",
    "Nordsjaelland": "FC Nordsjælland",
    # Czech Republic
    "Slavia Praha": "Slavia Prague",
    "Sparta Praha": "Sparta Prague",
    "Bohemians Praha": "Bohemians 1905",
    # Turkey
    "Bueyueksehir": "İstanbul Başakşehir FK",
    "Kayseri": "Kayserispor",
    # Azerbaijan
    "Karabakh Agdam": "FK Qarabağ",
    # Scotland
    "St Mirren": "St Mirren FC",
    "Dundee United": "Dundee United FC",
    # Portugal
    "Gil Vicente": "Gil Vicente FC",
    "Santa Clara": "CS Santa Clara",
    "Rio Ave": "Rio Ave FC",
    "Famalicao": "FC Famalicão",
    # Sweden
    "Mjaellby": "Mjällby AIF",
    # Poland
    "Plock": "Wisła Płock",
    "Katowice": "GKS Katowice",
    "Lubin": "Zagłębie Lubin",
    "Gornik": "Górnik Zabrze",
    "Nieciecza": "Bruk-Bet Termalica Nieciecza",
    # Romania
    "Rapid Bucuresti": "Rapid București",
    "Dinamo Bucuresti": "Dinamo București",
    "Universitatea Cluj": "Universitatea Cluj",
    "Craiova": "Universitatea Craiova",
    # Israel
    "M Tel Aviv": "Maccabi Tel Aviv FC",
    "H Tel Aviv": "Hapoel Tel Aviv FC",
    "H Petach Tikva": "Hapoel Petach Tikva FC",
    "Beer-Sheva": "Hapoel Be'er Sheva FC",
    "Aris Limassol": "Aris Limassol FC",
}

# Build reverse lookup (Sofascore → ClubElo) for history queries
_SOFASCORE_TO_CLUBELO: Dict[str, str] = {v: k for k, v in _CLUBELO_TO_SOFASCORE.items()}


def _get_clubelo_sofascore_map() -> Dict[str, str]:
    """Return ClubElo→Sofascore name mapping, augmented by REEP register.

    The hardcoded ``_CLUBELO_TO_SOFASCORE`` is always included as a
    fallback.  When the REEP teams.csv is available, its
    ``key_clubelo → name`` mapping is merged in (REEP entries can
    override hardcoded ones since REEP names are more authoritative).
    """
    merged = dict(_CLUBELO_TO_SOFASCORE)  # start with hardcoded fallback
    try:
        from backend.data.reep_registry import build_clubelo_sofascore_map

        reep_map = build_clubelo_sofascore_map()
        if reep_map:
            merged.update(reep_map)
    except Exception as exc:
        _log.debug("REEP augmentation unavailable: %s", exc)
    return merged


# ── Dynamic alias generation from REEP ───────────────────────────────────────

_dynamic_aliases_cache: Optional[Dict[str, List[str]]] = None


def _build_dynamic_aliases() -> Dict[str, List[str]]:
    """Build fuzzy alias table dynamically from REEP teams.csv.

    For every team row in REEP that has multiple name columns
    (``name``, ``key_clubelo``, ``key_fbref``, ``key_transfermarkt``),
    we normalize each variant and create bidirectional alias links.

    This supplements the hardcoded ``_EXTREME_ABBREVS`` so we don't
    have to manually maintain aliases for thousands of clubs.  The REEP
    register covers ~45,000 teams globally.

    Results are cached in-process after first successful build.
    Returns empty dict on any failure (no network in tests, graceful degradation).
    """
    global _dynamic_aliases_cache
    if _dynamic_aliases_cache is not None:
        return _dynamic_aliases_cache

    try:
        from backend.data.reep_registry import get_teams_df

        df = get_teams_df()
        if df is None:
            # Don't cache failure — retry next time (e.g. network was down)
            return {}
    except Exception as exc:
        _log.debug("REEP teams unavailable for alias building: %s", exc)
        return {}

    # Columns that may contain team names across providers.
    # NOTE: key_clubelo is deliberately EXCLUDED — the upstream REEP repo has
    # misaligned values in that column (e.g. Lille OSC→"Fulham"), so using it
    # as a name variant creates poisonous cross-links.  The remaining columns
    # (key_fbref, key_transfermarkt) are reliable.
    name_columns = [
        c for c in ["name", "key_fbref", "key_transfermarkt"]
        if c in df.columns
    ]
    if not name_columns:
        return {}

    aliases: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        # Collect all non-null name variants for this team
        variants: List[str] = []
        for col in name_columns:
            val = row.get(col)
            if pd.notna(val):
                s = str(val).strip()
                if s:
                    variants.append(s)

        if len(variants) < 2:
            continue  # nothing to cross-link

        # Normalize all variants
        normed = []
        raw_for_norm: Dict[str, str] = {}  # norm → raw (for token check)
        for v in variants:
            n = _normalize_team_name(v)
            if n and len(n) >= 3:  # skip trivially short
                normed.append(n)
                raw_for_norm[n] = v

        # Remove duplicates but preserve order
        normed = list(dict.fromkeys(normed))
        if len(normed) < 2:
            continue

        # Create bidirectional links between all pairs, but only when
        # variants share at least one significant token OR have high
        # character-level similarity.  This prevents cross-links from
        # misaligned REEP columns (e.g. key_clubelo of Lille OSC being
        # "Fulham" due to a data error).
        for i, norm_a in enumerate(normed):
            for j, norm_b in enumerate(normed):
                if i == j:
                    continue
                raw_a = raw_for_norm.get(norm_a, norm_a)
                raw_b = raw_for_norm.get(norm_b, norm_b)
                sim = SequenceMatcher(None, norm_a, norm_b).ratio()
                if not _has_token_overlap(raw_a, raw_b) and sim < _HIGH_SIMILARITY_BYPASS:
                    continue  # skip bogus cross-link
                aliases.setdefault(norm_a, [])
                if norm_b not in aliases[norm_a]:
                    aliases[norm_a].append(norm_b)

    _log.info(
        "Built %d dynamic aliases from REEP teams (%d bidirectional links)",
        len(aliases),
        sum(len(v) for v in aliases.values()),
    )
    _dynamic_aliases_cache = aliases
    return aliases


def _get_merged_aliases() -> Dict[str, List[str]]:
    """Merge hardcoded ``_EXTREME_ABBREVS`` with dynamic REEP aliases.

    Hardcoded entries take priority (they are curated for edge cases
    like PSG, ManCity, BVB that REEP may not handle perfectly).
    Dynamic aliases fill in the gaps for the thousands of teams that
    aren't manually maintained.
    """
    dynamic = _build_dynamic_aliases()
    if not dynamic:
        return _EXTREME_ABBREVS

    # Start with dynamic, then overlay hardcoded
    merged = dict(dynamic)
    for key, val_list in _EXTREME_ABBREVS.items():
        existing = merged.get(key, [])
        # Merge: keep hardcoded entries + any dynamic extras
        combined = list(val_list)
        for v in existing:
            if v not in combined:
                combined.append(v)
        merged[key] = combined

    return merged


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


# ── Opta-based ranking builder (Option B) ─────────────────────────────────────

def _compute_rankings_from_opta() -> (
    Optional[Tuple[Dict[str, TeamRanking], Dict[str, LeagueSnapshot]]]
):
    """Build TeamRanking / LeagueSnapshot dicts from Opta Power Rankings.

    Opta provides normalized 0-100 scores natively.  For ``raw_elo`` the
    function first tries ClubElo (exact scale the model was trained on); if
    the team isn't in ClubElo it falls back to a linear rescale from Opta's
    0-100 → ~1000-2100.

    Returns ``None`` if the Opta scrape yields no data (caller should
    fall back to the legacy ClubElo pipeline).
    """
    from backend.data import opta_client

    opta_teams = opta_client.get_team_rankings()
    if not opta_teams:
        _log.warning("Opta team rankings empty — falling back to ClubElo")
        return None

    opta_leagues = opta_client.get_league_rankings()

    # Build a league-name → (rating, number_of_teams) lookup for Opta league
    # rankings.  These official values from league-meta.json are used for
    # league snapshot ``mean_normalized`` and ``team_count`` when available,
    # ensuring correctness regardless of how many teams from each league
    # we actually matched by name.
    opta_league_ratings: Dict[str, float] = {}
    opta_league_team_counts: Dict[str, int] = {}
    for lr in opta_leagues:
        opta_league_ratings[lr.league] = lr.rating
        if lr.number_of_teams > 0:
            opta_league_team_counts[lr.league] = lr.number_of_teams

    # Collect all ClubElo data (single fetch, reused for raw_elo + league mapping).
    clubelo_raw: Dict[str, float] = {}  # canonical_name → raw elo
    clubelo_league_map: Dict[str, str] = {}  # canonical_name → league_code
    clubelo_raw_name_to_canonical: Dict[str, str] = {}  # raw ClubElo name → canonical
    clubelo_name_map: Dict[str, str] = {}  # populated inside try; used for Opta name normalization
    try:
        clubelo_name_map = _get_clubelo_sofascore_map()
        ce_df = clubelo_client.get_all_by_date(date.today())
        if ce_df is not None and len(ce_df) > 0:
            for raw_name in ce_df.index:
                elo_val = float(ce_df.loc[raw_name, "elo"])
                canonical = clubelo_name_map.get(str(raw_name), str(raw_name))
                clubelo_raw[canonical] = elo_val
                clubelo_raw_name_to_canonical[str(raw_name)] = canonical
                # Also build league code mapping
                ce_league = ce_df.loc[raw_name, "league"]
                league_code = _clubelo_to_code(ce_league)
                if league_code is None and "country" in ce_df.columns:
                    league_code = _clubelo_to_code_from_country(
                        ce_df.loc[raw_name, "country"],
                        (
                            ce_df.loc[raw_name, "level"]
                            if "level" in ce_df.columns
                            else 1
                        ),
                    )
                if league_code:
                    clubelo_league_map[canonical] = league_code
    except Exception as exc:
        _log.warning("ClubElo fetch for raw_elo/league overlay failed: %s", exc)

    # Build a reverse lookup: opta_name → canonical ClubElo name.
    # Index by BOTH canonical name AND original ClubElo raw name so that:
    #   - "Arsenal F.C." (canonical, after REEP mapping) is found, AND
    #   - "Arsenal" (ClubElo raw name) is ALSO found when Opta provides "Arsenal"
    # Without the raw-name index, REEP-mapped teams like Arsenal, Bayern etc.
    # would all get league_code="UNK" because the Opta name matches the raw
    # ClubElo name, not the canonical "Arsenal F.C." form.
    clubelo_lower_index: Dict[str, str] = {}  # lower(name) → canonical
    for canonical in clubelo_raw:
        clubelo_lower_index[canonical.lower()] = canonical
    # Add raw ClubElo names as additional keys (e.g. "arsenal" → "Arsenal F.C.")
    for raw_name, canonical in clubelo_raw_name_to_canonical.items():
        raw_lower = raw_name.lower()
        if raw_lower not in clubelo_lower_index:
            clubelo_lower_index[raw_lower] = canonical

    all_teams_data: Dict[str, Tuple[float, float, str]] = {}
    # team_name -> (opta_rating, raw_elo, league_code)
    all_teams_rank: Dict[str, int] = {}
    # team_name -> best (lowest) Opta rank seen so far for that name.
    # When multiple Opta entries share the same contestantName (e.g. four
    # clubs all called "Arsenal"), we keep only the highest-quality one
    # (rank=1 = strongest) so that Arsenal London (rank~1, rating~100)
    # is never overwritten by Arsenal Guadeloupe (rank=6446, rating=51.6).

    for opta_team in opta_teams:
        team_name = opta_team.team
        # Normalise Opta team name via the ClubElo alias map so that
        # alternate spellings from Opta (e.g. "Wolves", "Leeds") are
        # collapsed to the canonical form ("Wolverhampton Wanderers",
        # "Leeds United") before the rank-dedup check below.
        team_name = clubelo_name_map.get(team_name, team_name)
        opta_rating = opta_team.rating

        # Skip if we already have a better-ranked team with this exact name.
        existing_rank = all_teams_rank.get(team_name, float("inf"))
        if opta_team.rank >= existing_rank:
            continue

        # raw_elo: prefer ClubElo actual Elo, fall back to rescale.
        # Try exact match first, then case-insensitive.
        raw_elo = clubelo_raw.get(team_name)
        if raw_elo is None:
            ce_canonical = clubelo_lower_index.get(team_name.lower())
            if ce_canonical is not None:
                raw_elo = clubelo_raw[ce_canonical]
        if raw_elo is None:
            raw_elo = _opta_score_to_raw_elo(opta_rating)

        # League code: ClubElo exact → ClubElo case-insensitive →
        # Opta (domestic_league + country) → "UNK".
        #
        # We deliberately do NOT fuzzy-match domestic_league here because
        # Opta has 14k+ teams and generic league names like "Premier League"
        # appear in dozens of countries (Malta, Wales, etc.) — fuzzy matching
        # at 70%+ pulls hundreds of weak teams into ENG1.
        #
        # However, the compound key (domestic_league, country) is safe for
        # exact matching: "Premier League"+"England" uniquely identifies ENG1,
        # while "Premier League"+"Wales" maps to WAL1.  This picks up teams
        # that ClubElo doesn't cover (e.g. newly promoted clubs) without the
        # false-positive risk of name-only matching.
        # Opta (country, domestic_league) exact lookup → "UNK".
        # The exact (country, league) lookup fills in teams ClubElo misses
        # (e.g. Bundesliga has 18 clubs but ClubElo only tracks ~10).
        # Using BOTH fields prevents pollution: "Premier League" in Wales,
        # Belarus, Armenia etc. never matches ENG1 because their country
        # doesn't equal "england".
        league_code = clubelo_league_map.get(team_name)
        if league_code is None:
            ce_canonical = clubelo_lower_index.get(team_name.lower())
            if ce_canonical is not None:
                league_code = clubelo_league_map.get(ce_canonical)
        if not league_code:
            league_code = "UNK"

        # Fallback: use Opta's own metadata to resolve the league.
        if league_code == "UNK":
            opta_resolved = _resolve_opta_league_code(
                opta_team.domestic_league, opta_team.country
            )
            if opta_resolved:
                league_code = opta_resolved

        all_teams_data[team_name] = (opta_rating, raw_elo, league_code)
        all_teams_rank[team_name] = opta_team.rank

    if not all_teams_data:
        return None

    # Build league snapshots from the Opta ratings.
    # Track which Opta domestic_league names map to each league code so we
    # can look up official seasonAverageRating and leagueSize from league-meta.
    league_teams: Dict[str, List[Tuple[str, float, float]]] = {}
    # league_code -> [(team_name, raw_elo, opta_rating)]
    league_code_to_opta_name: Dict[str, str] = {}
    # league_code -> Opta domesticLeagueName (for league-meta.json lookup)
    for opta_team in opta_teams:
        team_name = clubelo_name_map.get(opta_team.team, opta_team.team)
        if team_name not in all_teams_data:
            continue
        _, _, code = all_teams_data[team_name]
        if code != "UNK" and code not in league_code_to_opta_name:
            if opta_team.domestic_league:
                league_code_to_opta_name[code] = opta_team.domestic_league

    for team_name, (opta_rating, raw_elo, code) in all_teams_data.items():
        league_teams.setdefault(code, []).append((team_name, raw_elo, opta_rating))

    today = date.today()
    league_snapshots: Dict[str, LeagueSnapshot] = {}
    for code, members in league_teams.items():
        raw_elos = np.array([e for _, e, _ in members])
        norms = np.array([n for _, _, n in members])
        info = LEAGUES.get(code)
        league_snapshots[code] = LeagueSnapshot(
            league_code=code,
            league_name=info.name if info else code,
            date=today,
            mean_elo=float(np.mean(raw_elos)),
            std_elo=float(np.std(raw_elos)) if len(raw_elos) > 1 else 0.0,
            p10=float(np.percentile(norms, 10)) if len(norms) > 1 else float(norms[0]),
            p25=float(np.percentile(norms, 25)) if len(norms) > 1 else float(norms[0]),
            p50=float(np.percentile(norms, 50)),
            p75=float(np.percentile(norms, 75)) if len(norms) > 1 else float(norms[0]),
            p90=float(np.percentile(norms, 90)) if len(norms) > 1 else float(norms[0]),
            mean_normalized=float(np.mean(norms)),
            team_count=len(members),
        )

    # Build team rankings
    team_rankings: Dict[str, TeamRanking] = {}
    for team_name, (opta_rating, raw_elo, code) in all_teams_data.items():
        league_mean = league_snapshots[code].mean_normalized
        team_rankings[team_name] = TeamRanking(
            team_name=team_name,
            league_code=code,
            raw_elo=raw_elo,
            normalized_score=opta_rating,
            league_mean_normalized=league_mean,
            relative_ability=opta_rating - league_mean,
        )

    unk_count = sum(1 for _, (_, _, c) in all_teams_data.items() if c == "UNK")
    _log.info(
        "Built Opta-based rankings: %d teams, %d leagues "
        "(%d with ClubElo raw_elo, %d rescaled, %d UNK league)",
        len(team_rankings),
        len(league_snapshots),
        sum(1 for tn in all_teams_data if tn in clubelo_raw),
        sum(1 for tn in all_teams_data if tn not in clubelo_raw),
        unk_count,
    )
    return team_rankings, league_snapshots


def compute_daily_rankings(
    query_date: Optional[date] = None,
) -> Tuple[Dict[str, TeamRanking], Dict[str, LeagueSnapshot]]:
    """Compute global Power Rankings for all known teams on a date.

    **Option B hybrid behaviour**:

    *   When *query_date* is today (or ``None`` → today) **and**
        ``_USE_OPTA_FOR_INFERENCE`` is ``True``, Opta Power Rankings are
        used as the primary source for normalised scores (0-100).  ClubElo
        provides the ``raw_elo`` on the ~1000-2100 scale the model trained
        on; teams not in ClubElo get a linear rescale from Opta's 0-100.
    *   For historical dates the legacy ClubElo + WorldFootballElo pipeline
        is used (Opta has no historical archive).

    Returns
    -------
    (team_rankings, league_snapshots)
        team_rankings: dict[team_name -> TeamRanking]
        league_snapshots: dict[league_code -> LeagueSnapshot]
    """
    if query_date is None:
        query_date = date.today()

    # Cap to today — prevents future-date lookups that would time out or
    # return empty results from ClubElo (the API has no future data).
    today = date.today()
    if query_date > today:
        query_date = today

    date_str = query_date.isoformat()

    # In-process cache — fastest path, avoids diskcache I/O for repeated
    # dates within the same training run (e.g. all 2022-07-01 samples).
    if date_str in _rankings_in_process_cache:
        return _rankings_in_process_cache[date_str]

    key = cache.make_key("power_rankings", date_str)
    cached = cache.get(key, max_age=_ONE_DAY)
    if cached is not None:
        _rankings_in_process_cache[date_str] = cached
        return cached

    # ── Option B: try Opta for current-day inference ──────────────────────
    if _USE_OPTA_FOR_INFERENCE and query_date == today:
        try:
            opta_result = _compute_rankings_from_opta()
            if opta_result is not None:
                cache.set(key, opta_result)
                _rankings_in_process_cache[date_str] = opta_result
                return opta_result
        except Exception as exc:
            _log.warning(
                "Opta ranking build failed, falling back to ClubElo: %s", exc
            )

    # ── Legacy path: ClubElo + WorldFootballElo ─────────────────────────────
    # Used for historical dates (training) or when Opta is unavailable.

    # Step 1 — Collect all team Elo scores
    all_teams: Dict[str, Tuple[float, str]] = {}  # team -> (elo, league_code)

    # Build the ClubElo→Sofascore name map (hardcoded + REEP augmentation).
    clubelo_name_map = _get_clubelo_sofascore_map()

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
                    canonical = clubelo_name_map.get(str(raw_name), str(raw_name))
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
    _rankings_in_process_cache[date_str] = result
    return result


def _resolve_league_for_ranking(
    ranking: TeamRanking,
    tournament_id: Optional[int],
    league_snapshots: Dict[str, "LeagueSnapshot"],
) -> TeamRanking:
    """Patch a TeamRanking's league fields when tournament_id is available.

    If the ranking has ``league_code == "UNK"`` (team not in ClubElo) but
    the caller knows the Sofascore ``tournament_id``, we resolve the league
    via :func:`league_registry.get_by_sofascore_id` and recompute
    ``league_mean_normalized`` and ``relative_ability`` from the correct
    league snapshot.

    Returns the (possibly mutated) ranking.
    """
    if tournament_id is None:
        return ranking
    if ranking.league_code != "UNK":
        return ranking  # already resolved

    from backend.utils.league_registry import LEAGUES, get_by_sofascore_id

    info = get_by_sofascore_id(tournament_id)
    if info is None:
        return ranking

    # Find the matching league code
    resolved_code: Optional[str] = None
    for code, li in LEAGUES.items():
        if li.sofascore_tournament_id == tournament_id:
            resolved_code = code
            break

    if resolved_code is None:
        return ranking

    # If we have a league snapshot for this code, use its mean
    if resolved_code in league_snapshots:
        snap = league_snapshots[resolved_code]
        ranking.league_code = resolved_code
        ranking.league_mean_normalized = snap.mean_normalized
        ranking.relative_ability = ranking.normalized_score - snap.mean_normalized
        _log.debug(
            "Resolved league for '%s': UNK -> %s (mean=%.1f)",
            ranking.team_name, resolved_code, snap.mean_normalized,
        )
    else:
        # No snapshot for this league yet — use overall mean as approximation
        ranking.league_code = resolved_code
        _log.debug(
            "Resolved league code for '%s': UNK -> %s (no snapshot available)",
            ranking.team_name, resolved_code,
        )

    return ranking


# ── Opta league rating helpers ────────────────────────────────────────────────

def _get_opta_league_map() -> Dict[str, float]:
    """Return {league_name_lower: rating} from Opta league rankings.

    Populated lazily on first call; the underlying opta_client result is
    already cached for 24 h so this adds only a tiny dict-build overhead.
    """
    global _opta_league_map
    if _opta_league_map is not None:
        return _opta_league_map
    try:
        from backend.data import opta_client
        _opta_league_map = {
            lr.league.lower(): lr.rating
            for lr in opta_client.get_league_rankings()
            if lr.league
        }
        _log.debug("Built Opta league map: %d leagues", len(_opta_league_map))
    except Exception as exc:
        _log.warning("Failed to build Opta league map: %s — using empty dict", exc)
        _opta_league_map = {}
    return _opta_league_map


def _get_opta_alias_map() -> Dict[str, str]:
    """Return {alias_lower → canonical_team_name} from Opta short/club names.

    Includes ``short_name`` and ``club_name`` fields from every Opta team,
    mapped back to the canonical ``team`` field used in the team rankings dict.
    Only aliases that differ from the canonical name are included.

    Also populates ``_opta_team_league_map`` in the same pass to avoid a
    second full iteration of the 14k-entry Opta team list.
    """
    global _opta_alias_map, _opta_team_league_map
    if _opta_alias_map is not None:
        return _opta_alias_map
    try:
        from backend.data import opta_client
        league_map = _get_opta_league_map()
        _opta_alias_map = {}
        _opta_team_league_map = {}
        n_aliases = 0
        n_teams = 0
        for t in opta_client.get_team_rankings():
            n_teams += 1
            canonical = t.team
            # ── alias map ────────────────────────────────────────────────────
            for alt in (t.short_name, t.club_name):
                if alt and alt.lower() != canonical.lower():
                    _opta_alias_map[alt.lower()] = canonical
                    n_aliases += 1
            # ── team → league rating flat index ──────────────────────────────
            if t.domestic_league:
                rating = league_map.get(t.domestic_league.lower())
                if rating is not None:
                    _opta_team_league_map[canonical.lower()] = rating
                    if t.short_name:
                        _opta_team_league_map[t.short_name.lower()] = rating
                    if t.club_name:
                        _opta_team_league_map[t.club_name.lower()] = rating
        _log.info(
            "Built Opta alias map: %d aliases from %d teams; "
            "%d team→league rating entries",
            n_aliases, n_teams, len(_opta_team_league_map),
        )
    except Exception as exc:
        _log.warning("Failed to build Opta alias/league maps: %s — using empty dicts", exc)
        _opta_alias_map = {}
        _opta_team_league_map = {}
    return _opta_alias_map


def _get_opta_team_league_map() -> Dict[str, float]:
    """Return {team_name_lower: league_rating} — O(1) per lookup during training.

    Built in the same pass as ``_get_opta_alias_map()`` so there is never
    more than one full iteration of the Opta team list per process lifetime.
    """
    global _opta_team_league_map
    if _opta_team_league_map is not None:
        return _opta_team_league_map
    _get_opta_alias_map()  # populates _opta_team_league_map as a side effect
    return _opta_team_league_map or {}


def get_league_opta_rating(
    league_code: Optional[str] = None,
    team_name: Optional[str] = None,
) -> float:
    """Return Opta league rating (0-100) for a league or team.

    Lookup priority
    ---------------
    1. *team_name* → find in Opta team list → ``domestic_league`` field →
       look up in Opta league ratings.  Most accurate for clubs whose
       ``domestic_league`` matches the Opta league name exactly.
    2. *league_code* → ``LEAGUES`` registry name → fuzzy-match in Opta league
       list.  Covers the ~50 leagues in our registry even when team lookup
       fails (e.g. training samples without Opta team coverage).
    3. Returns 50.0 (scale midpoint) when neither resolves and logs at DEBUG.

    This replaces the ClubElo-derived ``league_mean_normalized`` which only
    covered ~50 European leagues.  Opta league ratings cover 446 leagues on
    the same 0-100 scale.

    Safe to call during training (leakage risk accepted per task spec): Opta
    league ratings are relatively stable season-to-season.
    """
    # ── Priority 1: team_name → league rating (O(1) dict lookup) ────────────
    # _get_opta_team_league_map() is built once per process from the Opta team
    # list and maps every team/short_name/club_name to its league rating.
    if team_name:
        team_league_map = _get_opta_team_league_map()
        name_lower = team_name.lower()
        rating = team_league_map.get(name_lower)
        if rating is None:
            # Try alias resolution (short_name / club_name)
            alias_map = _get_opta_alias_map()
            canonical = alias_map.get(name_lower)
            if canonical:
                rating = team_league_map.get(canonical.lower())
        if rating is not None:
            return rating

    # ── Priority 2: league_code → registry name → Opta league match ──────────
    # Results are cached in _league_code_opta_rating_cache so the fuzzy
    # SequenceMatcher loop only runs ONCE per league_code per process.
    if league_code:
        if league_code in _league_code_opta_rating_cache:
            return _league_code_opta_rating_cache[league_code]
        info = LEAGUES.get(league_code)
        if info and info.name:
            league_map = _get_opta_league_map()
            # Exact match
            rating = league_map.get(info.name.lower())
            if rating is None:
                # Fuzzy match — runs at most once per league_code
                best_score = 0.0
                for key, r in league_map.items():
                    score = SequenceMatcher(None, info.name.lower(), key).ratio()
                    if score > best_score and score >= 0.70:
                        best_score = score
                        rating = r
            if rating is not None:
                _league_code_opta_rating_cache[league_code] = rating
                return rating
        # Cache the miss too so we don't re-run on every sample
        _league_code_opta_rating_cache[league_code] = 50.0

    _log.debug(
        "No Opta league rating for code=%s, team=%s — returning 50.0",
        league_code, team_name,
    )
    return 50.0


def _opta_fallback_ranking(
    team_name: str,
    tournament_id: Optional[int],
    league_snapshots: Dict[str, LeagueSnapshot],
) -> Optional[TeamRanking]:
    """Try Opta Power Rankings when ClubElo has no coverage for *team_name*.

    Priority:
    1. Exact or fuzzy name match in Opta team list → use ``opta_team.rating``
    2. No team match → find the team's league via *tournament_id* and use
       the Opta league's average ``rating`` as a proxy.
    3. Still nothing → return ``None``.

    ``match_type`` is set to ``"opta"`` for a team match and
    ``"opta_league_avg"`` for a league-average proxy.
    """
    try:
        from backend.data import opta_client
    except ImportError:
        return None

    opta_teams = opta_client.get_team_rankings()
    if not opta_teams:
        return None

    # ── Step 1: find team in Opta ────────────────────────────────────────────
    opta_team = None

    # Exact match — collect ALL matches and pick the highest-ranked one
    # (rank=1 is best).  This prevents low-quality clubs with the same name
    # (e.g. "Arsenal Guadeloupe", rank=6446) from shadowing the real club
    # (e.g. "Arsenal" London, rank=1).
    exact_matches = [
        t for t in opta_teams
        if (t.team == team_name
            or (t.short_name and t.short_name == team_name)
            or (t.club_name and t.club_name == team_name))
    ]
    if exact_matches:
        opta_team = min(exact_matches, key=lambda t: t.rank)

    # Case-insensitive if exact failed (includes short_name / club_name)
    if opta_team is None:
        name_lower = team_name.lower()
        # Check alias map built from short_name / club_name
        alias_map = _get_opta_alias_map()
        canonical_name = alias_map.get(name_lower)
        if canonical_name:
            alias_matches = [t for t in opta_teams if t.team == canonical_name]
            if alias_matches:
                opta_team = min(alias_matches, key=lambda t: t.rank)
    if opta_team is None:
        name_lower = team_name.lower()
        ci_matches = [
            t for t in opta_teams
            if (t.team.lower() == name_lower
                or (t.short_name and t.short_name.lower() == name_lower)
                or (t.club_name and t.club_name.lower() == name_lower))
        ]
        if ci_matches:
            opta_team = min(ci_matches, key=lambda t: t.rank)

    # Fuzzy if still no match
    if opta_team is None:
        best_score = 0.0
        for t in opta_teams:
            if _RAPIDFUZZ_AVAILABLE:
                score = float(_rfuzz.token_sort_ratio(team_name, t.team))
            else:
                score = SequenceMatcher(None, team_name.lower(), t.team.lower()).ratio() * 100.0
            if score > best_score and score >= 80.0:
                best_score = score
                opta_team = t

    if opta_team is not None:
        raw_elo = _opta_score_to_raw_elo(opta_team.rating)
        # Determine league code — prefer league_snapshots resolution via
        # tournament_id; fall back to fuzzy match on opta_team.domestic_league;
        # last resort "UNK".
        league_code = "UNK"
        league_mean = 50.0
        if tournament_id is not None:
            for code, li in LEAGUES.items():
                if li.sofascore_tournament_id == tournament_id:
                    league_code = code
                    if code in league_snapshots:
                        league_mean = league_snapshots[code].mean_normalized
                    break
        # If tournament_id didn't resolve a league, try matching
        # opta_team.domestic_league against our LEAGUES display names.
        if league_code == "UNK" and opta_team.domestic_league:
            dl_lower = opta_team.domestic_league.lower()
            best_lc_score = 0.0
            for code, li in LEAGUES.items():
                if _RAPIDFUZZ_AVAILABLE:
                    sc = float(_rfuzz.token_sort_ratio(dl_lower, li.name.lower()))
                else:
                    sc = SequenceMatcher(None, dl_lower, li.name.lower()).ratio() * 100.0
                if sc > best_lc_score and sc >= 70.0:
                    best_lc_score = sc
                    league_code = code
            if league_code != "UNK" and league_code in league_snapshots:
                league_mean = league_snapshots[league_code].mean_normalized
        ranking = TeamRanking(
            team_name=team_name,
            league_code=league_code,
            raw_elo=raw_elo,
            normalized_score=opta_team.rating,
            league_mean_normalized=league_mean,
            relative_ability=opta_team.rating - league_mean,
            match_type="opta",
        )
        return ranking

    # ── Step 2: league-average proxy via tournament_id ───────────────────────
    if tournament_id is None:
        return None

    league_code = "UNK"
    for code, li in LEAGUES.items():
        if li.sofascore_tournament_id == tournament_id:
            league_code = code
            break

    if league_code == "UNK":
        return None

    opta_leagues = opta_client.get_league_rankings()
    if not opta_leagues:
        return None

    # Look up by league name similarity
    league_info = LEAGUES.get(league_code)
    league_display = league_info.name if league_info else league_code
    best_lr = None
    best_score = 0.0
    for lr in opta_leagues:
        score = SequenceMatcher(None, league_display.lower(), lr.league.lower()).ratio() * 100.0
        if score > best_score and score >= 60.0:
            best_score = score
            best_lr = lr

    if best_lr is None:
        return None

    raw_elo = _opta_score_to_raw_elo(best_lr.rating)
    league_mean = best_lr.rating
    if league_code in league_snapshots:
        league_mean = league_snapshots[league_code].mean_normalized
    return TeamRanking(
        team_name=team_name,
        league_code=league_code,
        raw_elo=raw_elo,
        normalized_score=best_lr.rating,
        league_mean_normalized=league_mean,
        relative_ability=0.0,
        match_type="opta_league_avg",
    )


def get_team_ranking(
    team_name: str,
    query_date: Optional[date] = None,
    tournament_id: Optional[int] = None,
) -> Optional[TeamRanking]:
    """Get a single team's Power Ranking.

    Uses fuzzy matching to handle name differences between data sources.
    For example, ClubElo returns ``"RealMadrid"`` while Sofascore returns
    ``"Real Madrid"``.

    The returned ``TeamRanking.match_type`` reflects how the team was
    resolved: ``"exact"`` / ``"fuzzy"`` for ClubElo, ``"opta"`` for an
    Opta team match, ``"opta_league_avg"`` for an Opta league-average proxy.
    Returns ``None`` when no source (ClubElo or Opta) has coverage.

    Parameters
    ----------
    team_name : str
        Display name of the team (e.g. ``"Arsenal"``).
    query_date : date, optional
        Date for historical lookup.  ``None`` → today.
    tournament_id : int, optional
        Sofascore tournament ID for the team's league.  When provided and
        the team's ``league_code`` would otherwise be ``"UNK"``, this is
        used to resolve the correct league via ``league_registry``.  This
        also enables accurate ``league_mean_normalized`` and
        ``relative_ability`` for Opta teams not covered by ClubElo.
    """
    teams, league_snapshots = compute_daily_rankings(query_date)

    if not teams:
        _log.warning(
            "Power Rankings empty — no teams loaded from any Elo source"
        )
        return None

    # 0. Check TEAM_ALIASES before any matching.
    #    - alias is None → confirmed absent, skip immediately (return None)
    #    - alias is a string → remap the lookup name and continue matching
    effective_name = team_name
    if team_name in TEAM_ALIASES:
        alias = TEAM_ALIASES[team_name]
        if alias is None:
            _log.debug(
                "Team '%s' confirmed absent from Elo sources — returning None",
                team_name,
            )
            return None
        effective_name = alias

    # 0a. CLUBELO_ALIASES — maps display names to ClubElo canonical names
    #     before fuzzy matching (exact lookups for known high-value variants).
    if effective_name in CLUBELO_ALIASES:
        effective_name = CLUBELO_ALIASES[effective_name]

    # 1. Exact match (cheapest check) — try both effective and original name
    for lookup in dict.fromkeys([effective_name, team_name]):
        if lookup in teams:
            ranking = teams[lookup]
            ranking.match_type = "exact"
            return _resolve_league_for_ranking(ranking, tournament_id, league_snapshots)

    # 2. Accent-normalized exact match — handles "Atlético Madrid" matching
    #    "Atletico Madrid" or vice-versa without needing a fuzzy pass.
    norm_query = _strip_accents(effective_name)
    if norm_query != effective_name:
        for key in teams:
            if _strip_accents(key) == norm_query:
                ranking = teams[key]
                ranking.match_type = "exact"
                return _resolve_league_for_ranking(ranking, tournament_id, league_snapshots)

    # 2a. Youth-suffix stripping — try the parent club name when the team is
    #     a reserve/youth side (e.g. "Sporting CP B" → "Sporting CP").
    stripped = _strip_youth_suffix(effective_name)
    if stripped != effective_name:
        for lookup in dict.fromkeys([stripped, _strip_accents(stripped)]):
            if lookup in teams:
                ranking = teams[lookup]
                ranking.match_type = "exact"
                return _resolve_league_for_ranking(ranking, tournament_id, league_snapshots)
        # Also try fuzzy on the stripped name
        match = _fuzzy_find_team(stripped, teams)
        if match is not None:
            _log.info("Youth-stripped fuzzy '%s' -> '%s'", team_name, match)
            ranking = teams[match]
            ranking.match_type = "fuzzy"
            return _resolve_league_for_ranking(ranking, tournament_id, league_snapshots)

    # 3. Fuzzy match on effective name
    match = _fuzzy_find_team(effective_name, teams)
    if match is not None:
        try:
            _log.info("Fuzzy matched '%s' -> '%s'", team_name, match)
        except UnicodeEncodeError:
            _log.info(
                "Fuzzy matched '%s' -> '%s'",
                team_name.encode("ascii", "replace").decode("ascii"),
                match.encode("ascii", "replace").decode("ascii"),
            )
        ranking = teams[match]
        ranking.match_type = "fuzzy"
        return _resolve_league_for_ranking(ranking, tournament_id, league_snapshots)

    # 4. Opta fallback — try Opta Power Rankings when ClubElo has no coverage.
    opta_result = _opta_fallback_ranking(team_name, tournament_id, league_snapshots)
    if opta_result is not None:
        _log.info(
            "No ClubElo for '%s' — resolved via Opta (score=%.1f, type=%s)",
            team_name, opta_result.normalized_score, opta_result.match_type,
        )
        return opta_result

    _log.warning(
        "No Power Ranking match for '%s' among %d teams (ClubElo + Opta checked)",
        team_name, len(teams)
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


# ── Opta (domestic_league, country) → league_code fallback ────────────────────
# Used when ClubElo doesn't cover a team.  The compound key avoids the
# ambiguity problem of bare league names: "Premier League" appears in
# England, Wales, Ukraine, etc. — by requiring the country to match too,
# only the correct league is selected.
#
# Built from the LEAGUES registry (exact name + country) plus a handful of
# known Opta-specific aliases where ``domesticLeagueName`` differs from our
# canonical ``LeagueInfo.name`` (e.g. "LaLiga" vs "La Liga").
_OPTA_DL_COUNTRY_TO_CODE: Dict[Tuple[str, str], str] = {}
for _code, _info in LEAGUES.items():
    _OPTA_DL_COUNTRY_TO_CODE[(_info.name.lower(), _info.country.lower())] = _code

# Known Opta domesticLeagueName variants that differ from our registry names.
_OPTA_DL_ALIASES: Dict[Tuple[str, str], str] = {
    # Spain — Opta uses the official rebrand "LaLiga"
    ("laliga", "spain"): "ESP1",
    ("laliga santander", "spain"): "ESP1",
    ("laliga ea sports", "spain"): "ESP1",
    ("laliga hypermotion", "spain"): "ESP2",
    ("laliga 2", "spain"): "ESP2",
    ("segunda división", "spain"): "ESP2",
    # France — league title sponsorship names
    ("ligue 1 uber eats", "france"): "FRA1",
    ("ligue 1 mcdonald's", "france"): "FRA1",
    # Italy — league title sponsorship names
    ("serie a tim", "italy"): "ITA1",
    ("serie a enilive", "italy"): "ITA1",
    ("serie bkt", "italy"): "ITA2",
    # Germany
    ("1. bundesliga", "germany"): "GER1",
    ("2. bundesliga", "germany"): "GER2",
    # England
    ("efl championship", "england"): "ENG2",
    # Portugal — name variant
    ("liga portugal", "portugal"): "POR1",
    ("liga portugal betclic", "portugal"): "POR1",
    # Netherlands
    ("eredivisie", "netherlands"): "NED1",
    # Belgium
    ("first division a", "belgium"): "BEL1",
    ("jupiler pro league", "belgium"): "BEL1",
    # Turkey — country name variants
    ("super lig", "turkey"): "TUR1",
    ("süper lig", "turkey"): "TUR1",
    ("super lig", "türkiye"): "TUR1",
    ("süper lig", "türkiye"): "TUR1",
    # Scotland
    ("premiership", "scotland"): "SCO1",
    ("scottish premiership", "scotland"): "SCO1",
    # Switzerland
    ("super league", "switzerland"): "SUI1",
}
_OPTA_DL_COUNTRY_TO_CODE.update(_OPTA_DL_ALIASES)


def _resolve_opta_league_code(domestic_league: str, country: str) -> Optional[str]:
    """Resolve a league code from Opta's ``domesticLeagueName`` + ``country``.

    Returns the league code if the (name, country) pair matches a known
    league in the registry or alias table, otherwise ``None``.
    """
    if not domestic_league or not country:
        return None
    key = (domestic_league.lower().strip(), country.lower().strip())
    return _OPTA_DL_COUNTRY_TO_CODE.get(key)


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
# 0.85 prevents false positives like "Brentford" matching "AFC Bournemouth"
# or "Sivasspor" matching "Samsunspor".  Edge cases below 0.85
# (ManCity ↔ ManchesterCity = 0.667, ManUtd ↔ ManchesterUnited = 0.545)
# are handled by _EXTREME_ABBREVS instead.
_FUZZY_THRESHOLD = 0.85

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
    "1fsvmainz05": ["mainz", "mainz05"],
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
    # MLS — prevent false positives with European "City" / "United" teams
    "orlandocity": ["orlandocitysc", "orlando"],
    "newyorkcity": ["newyorkcityfc", "nycfc", "nyc"],
    "nycfc": ["newyorkcity", "newyorkcityfc"],
    "intermiami": ["intermiamifc", "intermiamicf"],
    "intermiamifc": ["intermiami"],
    "intermiamicf": ["intermiami"],
    "losangelesfc": ["lafc"],
    "lafc": ["losangelesfc"],
    "losangelesgalaxy": ["lagalaxy", "galaxy"],
    "lagalaxy": ["losangelesgalaxy"],
    "galaxy": ["losangelesgalaxy"],
    "atlantaunited": ["atlantaunitedfc", "atlanta"],
    "atlantaunitedfc": ["atlantaunited"],
    "dcunited": ["dcunitedfc"],
    "dcunitedfc": ["dcunited"],
    "portlandtimbers": ["portland", "timbers"],
    "seattlesounders": ["seattle", "sounders", "seattlesoundersfc"],
    "seattlesoundersfc": ["seattlesounders"],
    "cincinnati": ["fccincinnati"],
    "fccincinnati": ["cincinnati"],
    "nashvillesc": ["nashville"],
    "charlottefc": ["charlotte"],
    "columbusc": ["columbuscrew", "crew"],
    "columbuscrew": ["columbusc", "crew"],
    "minnesotaunited": ["minnesota"],
    "newyorkredbulls": ["nyredbulls", "redbulls", "nyrb"],
    "redbulls": ["newyorkredbulls"],
    "sportingkansascity": ["sportingkc", "kansascity"],
    "sportingkc": ["sportingkansascity"],
    "houstondy": ["houstondynamo"],
    "houstondynamo": ["houstondy", "dynamo"],
    # Saudi Arabia
    "alhilal": ["alhilalfc", "hilal"],
    "alahli": ["alahlifc", "ahli"],
    "alnassr": ["alnassrfc", "nassr"],
    "alittihad": ["alittihadfc", "ittihad"],
    # Japan
    "urawareddiamonds": ["urawareds", "urawa"],
    "yokohamamarinos": ["yokohamafmarinos", "marinos"],
    "visselkobe": ["kobe", "vissel"],
    "kawasakifrontale": ["kawasaki"],
    # Portugal
    "fcporto": ["porto"],
    "porto": ["fcporto"],
    "slbenfica": ["benfica"],
    "benfica": ["slbenfica"],
    "scbraga": ["braga"],
    "braga": ["scbraga"],
    "vitoriasc": ["vitoria", "guimaraes", "vitoriaguimaraes"],
    "vitoria": ["vitoriasc"],
    "guimaraes": ["vitoriasc"],
    # Belgium (expanded)
    "krcgenk": ["genk"],
    "genk": ["krcgenk"],
    "royalantwerpfc": ["antwerp", "royalantwerp"],
    "antwerp": ["royalantwerpfc"],
    "standardliege": ["standard"],
    "standard": ["standardliege"],
    "kaagent": ["gent"],
    "gent": ["kaagent"],
    "royaleunionstgilloise": ["unionsg", "unionstgilloise"],
    "unionsg": ["royaleunionstgilloise"],
    "unionstgilloise": ["royaleunionstgilloise"],
    "sportingcharleroi": ["charleroi"],
    "charleroi": ["sportingcharleroi"],
    # Netherlands (expanded)
    "afcajax": ["ajax"],
    "ajax": ["afcajax"],
    "fcutrecht": ["utrecht"],
    "utrecht": ["fcutrecht"],
    "scheerenveen": ["heerenveen"],
    "heerenveen": ["scheerenveen"],
    "necnijmegen": ["nec"],
    "nec": ["necnijmegen"],
    "spartarotterdam": ["spartardam"],
    "fortunasittard": ["fortuna"],
    "fortuna": ["fortunasittard"],
    "heraclesalmelo": ["heracles"],
    "heracles": ["heraclesalmelo"],
    # Scotland (expanded)
    "heartofmidlothianfc": ["hearts", "heartsfc"],
    "hearts": ["heartofmidlothianfc"],
    "hibernianfc": ["hibs", "hibernian"],
    "hibernian": ["hibernianfc"],
    "hibs": ["hibernianfc"],
    "dundeeunitedfc": ["dundeeutd", "dundeeunited"],
    "dundeeutd": ["dundeeunitedfc"],
    "dundeeunited": ["dundeeunitedfc"],
    "stmirrenfc": ["stmirren"],
    "stmirren": ["stmirrenfc"],
    "stjohnstonefc": ["stjohnstone"],
    "stjohnstone": ["stjohnstonefc"],
    "kilmarnockfc": ["kilmarnock"],
    "kilmarnock": ["kilmarnockfc"],
    "rosscountyfc": ["rosscounty", "ross"],
    "rosscounty": ["rosscountyfc"],
    "motherwellfc": ["motherwell"],
    "motherwell": ["motherwellfc"],
    # Turkey (expanded)
    "istanbulbasaksehirfk": ["basaksehir", "istanbulbasaksehir"],
    "basaksehir": ["istanbulbasaksehirfk"],
    "kasimpasask": ["kasimpasa"],
    "kasimpasa": ["kasimpasask"],
    "caykurrizespor": ["rizespor"],
    "rizespor": ["caykurrizespor"],
    "adanademirspor": ["adana"],
    "adana": ["adanademirspor"],
    "gaziantepfk": ["gaziantep"],
    "gaziantep": ["gaziantepfk"],
    # Switzerland
    "bscyoungboys": ["youngboys"],
    "youngboys": ["bscyoungboys"],
    "fcbasel1893": ["basel"],
    "basel": ["fcbasel1893"],
    "fczurich": ["zurich"],
    "zurich": ["fczurich"],
    "grasshopperclubzurich": ["grasshoppers", "gczurich"],
    "grasshoppers": ["grasshopperclubzurich"],
    "gczurich": ["grasshopperclubzurich"],
    "fcstgallen1879": ["stgallen"],
    "stgallen": ["fcstgallen1879"],
    "fclausannesport": ["lausanne"],
    "lausanne": ["fclausannesport"],
    # Greece
    "olympiacosfc": ["olympiacos", "olympiakos"],
    "olympiacos": ["olympiacosfc"],
    "olympiakos": ["olympiacosfc"],
    "panathinaikosfc": ["panathinaikos"],
    "panathinaikos": ["panathinaikosfc"],
    "paokfc": ["paok"],
    "paok": ["paokfc"],
    "aekathensfc": ["aek", "aekathens"],
    "aek": ["aekathensfc"],
    "aekathens": ["aekathensfc"],
    "aristhessalonikifc": ["aris", "aristhessaloniki"],
    "aris": ["aristhessalonikifc"],
    # Czech Republic
    "spartaprague": ["spartaprag"],
    "slaviaprague": ["slavia"],
    "slavia": ["slaviaprague"],
    "viktoriaplzen": ["plzen", "viktoria"],
    "plzen": ["viktoriaplzen"],
    "fcbanikostravafc": ["ostrava", "banikostravafc"],
    "ostrava": ["banikostravafc"],
    "fkmladaboleslav": ["mlada", "mladaboleslav"],
    "mlada": ["fkmladaboleslav"],
    "fcslovanliberec": ["liberec"],
    "liberec": ["fcslovanliberec"],
    # Denmark
    "fccopenhagen": ["copenhagen"],
    "copenhagen": ["fccopenhagen"],
    "fcmidtjylland": ["midtjylland"],
    "midtjylland": ["fcmidtjylland"],
    "brondbyif": ["brondby"],
    "brondby": ["brondbyif"],
    "fcnordsjaelland": ["nordsjaelland"],
    "nordsjaelland": ["fcnordsjaelland"],
    "aarhusgf": ["agf", "aarhus"],
    "agf": ["aarhusgf"],
    "aarhus": ["aarhusgf"],
    # Croatia
    "gnkdinamozagreb": ["dinamozagreb"],
    "dinamozagreb": ["gnkdinamozagreb"],
    "hnkhajduksplit": ["hajduk", "hajduksplit"],
    "hajduk": ["hnkhajduksplit"],
    "hajduksplit": ["hnkhajduksplit"],
    "hnkrijeka": ["rijeka"],
    "rijeka": ["hnkrijeka"],
    "nkosijek": ["osijek"],
    "osijek": ["nkosijek"],
    # Serbia
    "fkcrvenazvedza": ["redstarbelgrade", "redstar", "crvenazvedza"],
    "redstarbelgrade": ["fkcrvenazvedza"],
    "redstar": ["fkcrvenazvedza"],
    "crvenazvedza": ["fkcrvenazvedza"],
    "fkpartizan": ["partizan", "partizanbelgrade"],
    "partizan": ["fkpartizan"],
    "partizanbelgrade": ["fkpartizan"],
    "fkvojvodina": ["vojvodina"],
    "vojvodina": ["fkvojvodina"],
    # Norway
    "fkbodoglimt": ["bodo", "bodoglimt"],
    "bodo": ["fkbodoglimt"],
    "bodoglimt": ["fkbodoglimt"],
    "rosenborgbk": ["rosenborg"],
    "rosenborg": ["rosenborgbk"],
    "moldefk": ["molde"],
    "molde": ["moldefk"],
    "vikingfk": ["viking"],
    "viking": ["vikingfk"],
    "skbrann": ["brann"],
    "brann": ["skbrann"],
    "lillestromsk": ["lillestrom"],
    "lillestrom": ["lillestromsk"],
    "valerengafotball": ["valerenga"],
    "valerenga": ["valerengafotball"],
    "stromsgodsetif": ["stromsgodset"],
    "stromsgodset": ["stromsgodsetif"],
    # Sweden
    "malmoff": ["malmo"],
    "malmo": ["malmoff"],
    "djurgardensif": ["djurgarden"],
    "djurgarden": ["djurgardensif"],
    "ifelfsborg": ["elfsborg"],
    "elfsborg": ["ifelfsborg"],
    "bkhacken": ["hacken"],
    "hacken": ["bkhacken"],
    "hammarbyif": ["hammarby"],
    "hammarby": ["hammarbyif"],
    "ifkgoteborg": ["goteborg"],
    "goteborg": ["ifkgoteborg"],
    "ifknorrkoping": ["norrkoping"],
    "norrkoping": ["ifknorrkoping"],
    # Poland
    "legiawarsaw": ["legia"],
    "legia": ["legiawarsaw"],
    "lechpoznan": ["lech"],
    "lech": ["lechpoznan"],
    "rakowczestochowa": ["rakow"],
    "rakow": ["rakowczestochowa"],
    "pogonszczecin": ["pogon"],
    "pogon": ["pogonszczecin"],
    "jagielloniabialystok": ["jagiellonia"],
    "jagiellonia": ["jagielloniabialystok"],
    "gornikzabrze": ["gornik"],
    "gornik": ["gornikzabrze"],
    "slaskwroclaw": ["slask"],
    "slask": ["slaskwroclaw"],
    "wislakrakow": ["wisla"],
    "wisla": ["wislakrakow"],
    "piastgliwice": ["piast"],
    "piast": ["piastgliwice"],
    "zaglebielubin": ["zaglebie"],
    "zaglebie": ["zaglebielubin"],
    # Romania
    "fcsb": ["steauabucharest", "steaua"],
    "steauabucharest": ["fcsb"],
    "steaua": ["fcsb"],
    "universitateacraiova": ["craiova", "ucraiova"],
    "craiova": ["universitateacraiova"],
    "rapidbucuresti": ["rapidbucharest"],
    "rapidbucharest": ["rapidbucuresti"],
    # Ukraine
    "shakhtardonetsk": ["shakhtar"],
    "shakhtar": ["shakhtardonetsk"],
    "dynamokyiv": ["dynamokiev"],
    "dynamokiev": ["dynamokyiv"],
    "zoryaluhansk": ["zorya"],
    "zorya": ["zoryaluhansk"],
    # Russia
    "zenitstpetersburg": ["zenit"],
    "zenit": ["zenitstpetersburg"],
    "spartakmoscow": ["spartak"],
    "spartak": ["spartakmoscow"],
    "cskamoscow": ["cska"],
    "cska": ["cskamoscow"],
    "lokomotivmoscow": ["lokomotiv"],
    "rubinkazan": ["rubin"],
    "rubin": ["rubinkazan"],
    # Bulgaria
    "ludogoretsrazgrad": ["ludogorets"],
    "ludogorets": ["ludogoretsrazgrad"],
    "pfclevskisofia": ["levski", "levskisofia"],
    "levski": ["pfclevskisofia"],
    "levskisofia": ["pfclevskisofia"],
    "cskasofia": ["cska1948sofia"],
    "botevplovdiv": ["botev"],
    "botev": ["botevplovdiv"],
    # Hungary
    "ferencvarostc": ["ferencvaros"],
    "ferencvaros": ["ferencvarostc"],
    "fehervárfc": ["fehervar"],
    "fehervar": ["fehervárfc"],
    "puskasademiafc": ["puskas", "puskasakademia"],
    "puskas": ["puskasademiafc"],
    "ujpestfc": ["ujpest"],
    "ujpest": ["ujpestfc"],
    "debrecenivsc": ["debrecen"],
    "debrecen": ["debrecenivsc"],
    # Cyprus
    "apoelnicosia": ["apoel"],
    "apoel": ["apoelnicosia"],
    "acomonia": ["omonia", "omonianicosia"],
    "omonia": ["acomonia"],
    "anorthosisfamagusta": ["anorthosis"],
    "anorthosis": ["anorthosisfamagusta"],
    "apollonlimassol": ["apollon"],
    "apollon": ["apollonlimassol"],
    # Finland
    "hjkhelsinki": ["hjk"],
    "hjk": ["hjkhelsinki"],
    "kuopionpalloseura": ["kups"],
    "kups": ["kuopionpalloseura"],
    # Slovakia
    "skslovanbratislava": ["slovanbratislava", "bratislava"],
    "slovanbratislava": ["skslovanbratislava"],
    "fcspartaktrnava": ["trnava", "spartaktrnava"],
    "trnava": ["fcspartaktrnava"],
    "mskzilina": ["zilina"],
    "zilina": ["mskzilina"],
    # Slovenia
    "nkmaribor": ["maribor"],
    "maribor": ["nkmaribor"],
    "nkolimpijaljubljana": ["olimpija", "olimpijaljubljana"],
    "olimpija": ["nkolimpijaljubljana"],
    "nkdomzale": ["domzale"],
    "domzale": ["nkdomzale"],
    "nsmura": ["mura"],
    "mura": ["nsmura"],
    # Bosnia
    "fkzeljeznicarsarajevo": ["zeljeznicar"],
    "zeljeznicar": ["fkzeljeznicarsarajevo"],
    "fksarajevo": ["sarajevo"],
    "sarajevo": ["fksarajevo"],
    "hskzrinjskimostar": ["zrinjski", "zrinjskimostar"],
    "zrinjski": ["hskzrinjskimostar"],
    "fkvelezmostar": ["velezmostar"],
    "velezmostar": ["fkvelezmostar"],
    # Israel
    "maccabitelavivfc": ["maccabitelaviv", "maccabita"],
    "maccabitelaviv": ["maccabitelavivfc"],
    "maccabita": ["maccabitelavivfc"],
    "maccabihaifafc": ["maccabihaifa"],
    "maccabihaifa": ["maccabihaifafc"],
    "hapoelbeersheva": ["hapoelbeershevafc"],
    "hapoelbeershevafc": ["hapoelbeersheva"],
    "beitarjerusalemfc": ["beitar", "beitarjerusalem"],
    "beitar": ["beitarjerusalemfc"],
    "hapoeltelavivfc": ["hapoeltelaviv", "hapoelta"],
    "hapoeltelaviv": ["hapoeltelavivfc"],
    "hapoelta": ["hapoeltelavivfc"],
    # Kazakhstan
    "fcastana": ["astana"],
    "astana": ["fcastana"],
    "fckairat": ["kairat", "kairatalmaty"],
    "kairat": ["fckairat"],
    "kairatalmaty": ["fckairat"],
    "fctobol": ["tobol"],
    "tobol": ["fctobol"],
    # Iceland
    "vikingurreykjavik": ["vikingur"],
    "vikingur": ["vikingurreykjavik"],
    "valurreykjavik": ["valur"],
    "valur": ["valurreykjavik"],
    "breidablik": ["breidablikubr"],
    "fhhafnarfjordur": ["fh"],
    "fh": ["fhhafnarfjordur"],
    "krreykjavik": ["kr"],
    "kr": ["krreykjavik"],
    # Ireland
    "shamrockroversfc": ["shamrock", "shamrockrovers"],
    "shamrock": ["shamrockroversfc"],
    "shamrockrovers": ["shamrockroversfc"],
    "dundalkfc": ["dundalk"],
    "dundalk": ["dundalkfc"],
    "bohemianfc": ["bohemian"],
    "stpatricksathleticfc": ["stpats", "stpatricksathletic"],
    "stpats": ["stpatricksathleticfc"],
    "shelbournefc": ["shelbourne"],
    "shelbourne": ["shelbournefc"],
    "derrycityfc": ["derry", "derrycity"],
    "derry": ["derrycityfc"],
    "sligoroversfc": ["sligo", "sligorovers"],
    "sligo": ["sligoroversfc"],
    # Wales
    "thenewsaintsfc": ["tns", "newsaints"],
    "tns": ["thenewsaintsfc"],
    "newsaints": ["thenewsaintsfc"],
    "connahsquaynomadsfc": ["connah", "connahsquay"],
    "connah": ["connahsquaynomadsfc"],
    # Georgia
    "fcdinamotbilisi": ["dinamotbilisi"],
    "dinamotbilisi": ["fcdinamotbilisi"],
    "fctorpedokutaisi": ["torpedo", "torpedokutaisi"],
    "torpedokutaisi": ["fctorpedokutaisi"],
    "fcdinamobatumi": ["dinamobatumi"],
    "dinamobatumi": ["fcdinamobatumi"],
    # South America (expanded)
    "gremio": ["gremiofbpa", "gremioporealegrense"],
    "internacional": ["scinternacional", "internacionalpoa"],
    "santosfc": ["santos"],
    "santos": ["santosfc"],
    "fluminense": ["fluminensefc"],
    "fluminensefc": ["fluminense"],
    "vascoda": ["vascodagama", "crvascodagama"],
    "vascodagama": ["vascoda"],
    "botafogo": ["botafogofr"],
    "botafogofr": ["botafogo"],
    "independiente": ["caindependiente"],
    "caindependiente": ["independiente"],
    "racingclub": ["racingclubavellaneda"],
    "sanlorenzo": ["casanlorenzo"],
    "casanlorenzo": ["sanlorenzo"],
    "velez": ["velezsarsfield"],
    "velezsarsfield": ["velez"],
    "estudiantes": ["estudiantesdelaplata"],
    "estudiantesdelaplata": ["estudiantes"],
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
    #    Uses merged aliases: hardcoded _EXTREME_ABBREVS + dynamic REEP aliases.
    merged_aliases = _get_merged_aliases()
    q_aliases = merged_aliases.get(q, [])
    for alias in q_aliases:
        if alias in candidates:
            return candidates[alias]
    # Reverse: check if any candidate has an alias matching q
    for norm, orig in candidates.items():
        for alias in merged_aliases.get(norm, []):
            if alias == q:
                return orig

    # 3. Substring containment — one name fully inside the other.
    #    Guard: require the shorter name to be at least 6 chars AND
    #    represent >= 45% of the longer name.  This prevents
    #    "paris" (5 chars, 29% of "parissaintgermain") from matching.
    #    Additional guard: require token overlap so "angers" ⊂ "rangers"
    #    does not produce a false match (no shared significant tokens).
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
        if not _has_token_overlap(query, orig):
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

    # 5. token_sort_ratio fuzzy matching on ORIGINAL names.
    #    Comparing original (non-normalised) strings preserves word structure
    #    so that multi-word names like "Angers SCO" score much lower against
    #    "Rangers FC" (~52) than against "Angers" (~100).  Using normalised
    #    names collapsed both to "angerso"/"rangers" giving a misleadingly
    #    high SequenceMatcher ratio of 0.857 which caused false matches.
    #
    #    Minimum accepted score: 80 (fuzz 0-100 scale).
    #    Token-overlap bypass: only for score >= 95 (handles abbreviation
    #    pairs like "Man United" ↔ "Manchester United" that share no 4-char
    #    token after normalisation but are unambiguous at high similarity).
    _MIN_FUZZY_SCORE = 80

    best_match: Optional[str] = None
    best_score = 0
    for orig in teams:
        if _RAPIDFUZZ_AVAILABLE:
            score = _rfuzz.token_sort_ratio(query, orig)
        else:
            # Graceful fallback to SequenceMatcher on normalised names.
            norm = _normalize_team_name(orig)
            score = int(SequenceMatcher(None, q, norm).ratio() * 100)
        if score > best_score:
            best_score = score
            best_match = orig

    if best_match is not None and best_score >= _MIN_FUZZY_SCORE:
        if not _has_token_overlap(query, best_match):
            if best_score < 95:
                _log.warning(
                    "Fuzzy match rejected (no token overlap): "
                    "'%s' -> '%s' (score=%d)",
                    query, best_match, best_score,
                )
                return None
        return best_match

    if best_match is not None:
        _log.warning(
            "No fuzzy match for '%s' (best='%s', score=%d < %d) — returning None",
            query, best_match, best_score, _MIN_FUZZY_SCORE,
        )
    return None
