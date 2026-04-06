"""Opta Power Rankings client — extracts team and league rankings from The Analyst.

The dataviz SPA at https://dataviz.theanalyst.com/opta-power-rankings/ is a
React app that *bundles its entire dataset directly into index.js* as two
``JSON.parse(...)`` calls:

    f6  = JSON.parse(`[{"rank":1,"contestantName":"Arsenal", ...}, ...]`)   ← men's
    C0  = JSON.parse(`[{"rank":1,"contestantName":"Barcelona", ...}, ...]`) ← women's

League metadata is a plain static JSON file served alongside the app:

    league-meta.json         ← men's (and mixed) leagues
    women-league-meta.json   ← women's leagues

This client:
1. Fetches index.js once per day (via curl_cffi → requests fallback).
2. Extracts the embedded JSON with a string-search (no regex engine overhead).
3. Caches the *parsed* Python objects for 24 h so subsequent calls are instant.
4. Fetches league-meta.json the same way and caches it separately.

All existing public interfaces (``get_team_rankings``, ``get_league_rankings``,
``get_team_rankings_dict``, ``get_league_rankings_dict``) are preserved.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

from backend.data import cache

_log = logging.getLogger(__name__)

# ── Endpoints ─────────────────────────────────────────────────────────────────
_BASE_URL = "https://dataviz.theanalyst.com/opta-power-rankings"
_INDEX_JS_URL = f"{_BASE_URL}/index.js"
_LEAGUE_META_URL = f"{_BASE_URL}/league-meta.json"
_WOMEN_LEAGUE_META_URL = f"{_BASE_URL}/women-league-meta.json"

# The JS bundle is ~17 MB and only regenerated when rankings change (roughly
# weekly, Mon-Fri).  Cache the *parsed* result for 7 days to avoid re-fetching
# the large bundle unnecessarily.
_CACHE_TTL = 604_800  # 7 days in seconds
_HTTP_TIMEOUT = 30   # seconds — large bundle needs generous timeout

# ── HTTP session (curl_cffi preferred for TLS fingerprint bypass) ─────────────
try:
    from curl_cffi.requests import Session as _CurlSession
    _http = _CurlSession(impersonate="chrome110")
except Exception:
    import requests as _http  # type: ignore[assignment]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class OptaTeamRanking:
    """A single team from the Opta Power Rankings."""

    rank: int
    team: str                       # contestantName
    rating: float                   # currentRating, 0-100 scale
    ranking_change_7d: Optional[str]  # e.g. "+3", "-1", "0"
    opta_id: str                    # contestantId (Opta's internal GUID)
    # Extra fields available from the bundle (not used by downstream code but
    # useful for debugging / future features).
    short_name: str = ""            # contestantShortName
    club_name: str = ""             # contestantClubName
    domestic_league: str = ""       # domesticLeagueName
    domestic_league_id: str = ""    # domesticLeagueId
    country: str = ""
    confederation: str = ""
    season_avg_rating: float = 0.0  # seasonAverageRating


@dataclass
class OptaLeagueRanking:
    """A single league from the Opta League Rankings."""

    rank: int
    league: str
    rating: float                   # seasonAverageRating, same 0-100 scale
    ranking_change_7d: Optional[str]
    country: str = ""
    number_of_teams: int = 0


# ── Internal fetchers ─────────────────────────────────────────────────────────

def _fetch_text(url: str) -> Optional[str]:
    """GET *url* and return response text, or None on any failure."""
    try:
        resp = _http.get(url, timeout=_HTTP_TIMEOUT)
        if resp.status_code == 200:
            return resp.text
        _log.warning("Opta fetch HTTP %d for %s", resp.status_code, url)
    except Exception as exc:
        _log.warning("Opta fetch failed for %s: %s", url, exc)
    return None


def _extract_json_parse(js_text: str, marker: Optional[str] = None) -> Optional[str]:
    """Find ``JSON.parse(\\`...\\`)`` in *js_text* and return the JSON string.

    If *marker* is given (e.g. ``"f6"``), searches for that exact variable
    assignment: ``<marker>=JSON.parse(...)``.  If *marker* is ``None``,
    returns ``None`` — use :func:`_extract_all_json_parse` for positional
    extraction.

    **Escape fix**: the ``comps`` field embeds Python-style list-of-dicts with
    competition names that were double-escaped during bundle generation.  Any
    competition name containing a double-quote (e.g. "Ligat Ha'al") is stored as
    the 3-byte sequence ``\\\\\"`` (backslash + backslash + double-quote).  In JSON
    that means *escaped-backslash* followed by *end-of-string*, which breaks
    ``json.loads``.  Collapsing it to the 2-byte ``\\"`` (escaped double-quote)
    fixes the parse without affecting any other field.

    Returns ``None`` if the marker is not found.
    """
    if marker is None:
        return None
    needle = f"{marker}=JSON.parse(`"
    idx = js_text.find(needle)
    if idx < 0:
        return None
    start = idx + len(needle)
    end = js_text.find("`", start)
    if end < 0:
        return None
    raw = js_text[start:end]
    # Collapse \\\" (3 bytes) → \" (2 bytes) — fixes double-escaped dquotes in
    # competition names inside the comps string field.
    _DBL_ESC = chr(92) + chr(92) + chr(34)   # \\"  (backslash backslash dquote)
    _SGL_ESC = chr(92) + chr(34)             # \"   (backslash dquote)
    return raw.replace(_DBL_ESC, _SGL_ESC)


def _extract_all_json_parse(js_text: str) -> List[str]:
    """Extract ALL ``JSON.parse(\\`[...]\\`)`` blobs from the JS bundle.

    Returns a list of raw JSON strings in the order they appear in the bundle.
    The bundle typically contains 4 blobs:
      [0] = men's team rankings
      [1] = men's search index (not used)
      [2] = women's team rankings
      [3] = women's search index (not used)

    Uses regex to avoid depending on minified variable names which change
    between deploys.
    """
    # Match JSON.parse(`[...]`) — the array content between backticks.
    # Use [^`]* instead of .*? to avoid backtracking on the ~17 MB bundle.
    pattern = r'JSON\.parse\(`(\[[^`]*\])`\)'
    matches = re.findall(pattern, js_text)

    result: List[str] = []
    _DBL_ESC = chr(92) + chr(92) + chr(34)
    _SGL_ESC = chr(92) + chr(34)
    for raw in matches:
        result.append(raw.replace(_DBL_ESC, _SGL_ESC))
    return result


def _parse_float(value: str, default: float = 0.0) -> float:
    """Parse a string to float, stripping commas and whitespace."""
    try:
        return float(str(value).strip().replace(",", ""))
    except (ValueError, TypeError):
        return default


def _parse_int(value: str, default: int = 0) -> int:
    """Parse a string to int, stripping commas, '#' prefixes and whitespace."""
    try:
        cleaned = str(value).strip().lstrip("#").replace(",", "")
        return int(float(cleaned))
    except (ValueError, TypeError):
        return default


def _parse_change(last_week: Optional[str], current: Optional[str]) -> str:
    """Convert lastWeekGlobalRank / currentGlobalRank to a ±N string."""
    try:
        delta = int(last_week or 0) - int(current or 0)
        if delta > 0:
            return f"+{delta}"
        return str(delta)
    except (TypeError, ValueError):
        return "0"


def _scrape_team_rankings() -> List[OptaTeamRanking]:
    """Fetch index.js and extract the men's team ranking dataset.

    Extraction strategy:
    1. Try regex-based ``_extract_all_json_parse`` which finds all
       ``JSON.parse(\\`[...]\\`)`` blobs by position (index 0 = men's teams).
       This is resilient to minified variable name changes between deploys.
    2. Fall back to the legacy ``f6`` marker if regex fails.

    Returns an empty list on any parse / network failure.
    """
    js = _fetch_text(_INDEX_JS_URL)
    if not js:
        return []

    # Strategy 1: positional regex extraction (preferred).
    blobs = _extract_all_json_parse(js)
    raw = blobs[0] if blobs else None

    # Strategy 2: legacy marker-based extraction (fallback).
    if not raw:
        raw = _extract_json_parse(js, "f6")

    if not raw:
        _log.warning(
            "Could not extract team rankings from Opta index.js — "
            "the bundle structure may have changed after a deploy"
        )
        return []

    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        _log.error("Failed to parse Opta team JSON from bundle: %s", exc)
        return []

    rankings: List[OptaTeamRanking] = []
    for e in entries:
        try:
            rankings.append(OptaTeamRanking(
                rank=int(e.get("rank", 0)),
                team=e.get("contestantName", ""),
                rating=float(e.get("currentRating", 0)),
                ranking_change_7d=_parse_change(
                    e.get("lastWeekGlobalRank"), e.get("currentGlobalRank")
                ),
                opta_id=e.get("contestantId", ""),
                short_name=e.get("contestantShortName", ""),
                club_name=e.get("contestantClubName", ""),
                domestic_league=e.get("domesticLeagueName", ""),
                domestic_league_id=e.get("domesticLeagueId", ""),
                country=e.get("country", ""),
                confederation=e.get("confederation", ""),
                season_avg_rating=float(e.get("seasonAverageRating", 0)),
            ))
        except Exception as exc:
            _log.debug("Skipping malformed Opta team entry: %s — %s", e, exc)

    _log.info("Parsed %d Opta team rankings from bundle", len(rankings))
    return rankings


def _scrape_league_rankings() -> List[OptaLeagueRanking]:
    """Fetch league-meta.json and return a sorted list of OptaLeagueRanking.

    Uses ``seasonAverageRating`` as the rating field (same 0-100 scale as
    team ratings).  Leagues with ``globalRank`` are sorted ascending so that
    rank 1 is the strongest league.
    """
    text = _fetch_text(_LEAGUE_META_URL)
    if not text:
        return []

    try:
        entries = json.loads(text)
    except json.JSONDecodeError as exc:
        _log.error("Failed to parse Opta league-meta.json: %s", exc)
        return []

    rankings: List[OptaLeagueRanking] = []
    for e in entries:
        try:
            # globalRank is descending in the raw data (rank 1 = weakest at
            # bottom of the list), so we invert to get a conventional rank
            # where rank 1 = strongest league.
            raw_global = e.get("globalRank") or e.get("lastWeekGlobalRank") or 0
            total = int(e.get("globalSize", 0)) or 1
            rank = total - int(float(raw_global)) + 1

            # 7-day change: compare globalRank to lastWeekGlobalRank
            last_w = e.get("lastWeekGlobalRank")
            curr_r = e.get("globalRank")
            change = _parse_change(
                str(int(float(last_w))) if last_w else None,
                str(int(float(curr_r))) if curr_r else None,
            )

            rankings.append(OptaLeagueRanking(
                rank=max(1, rank),
                league=e.get("leagueName", ""),
                rating=float(e.get("seasonAverageRating") or 0),
                ranking_change_7d=change,
                country=e.get("countryName", ""),
                number_of_teams=int(e.get("leagueSize", 0) or 0),
            ))
        except Exception as exc:
            _log.debug("Skipping malformed Opta league entry: %s — %s", e, exc)

    # Sort strongest → weakest (highest seasonAverageRating first) and
    # re-assign sequential ranks so callers get a clean 1-N ordering.
    rankings.sort(key=lambda r: r.rating, reverse=True)
    for i, r in enumerate(rankings, start=1):
        r.rank = i

    _log.info("Parsed %d Opta league rankings from league-meta.json", len(rankings))
    return rankings


# ── Public API (cached) ───────────────────────────────────────────────────────

def get_team_rankings(force_refresh: bool = False) -> List[OptaTeamRanking]:
    """Return today's Opta team rankings, cached for 7 days.

    Fetches ``index.js``, extracts the team rankings JSON blob, and parses it.
    On failure returns ``[]`` so callers can fall back gracefully.

    Parameters
    ----------
    force_refresh : bool
        Bypass the cache and re-fetch.
    """
    key = cache.make_key("opta_team_rankings_v3", date.today().isoformat())
    if not force_refresh:
        cached = cache.get(key, max_age=_CACHE_TTL)
        if cached is not None:
            _log.debug("Opta team rankings from cache (%d teams)", len(cached))
            return cached

    rankings = _scrape_team_rankings()
    if rankings:
        cache.set(key, rankings)
    return rankings


def get_league_rankings(force_refresh: bool = False) -> List[OptaLeagueRanking]:
    """Return today's Opta league rankings from ``league-meta.json``, cached 7 days.

    Parameters
    ----------
    force_refresh : bool
        Bypass the cache and re-fetch.
    """
    key = cache.make_key("opta_league_rankings_v3", date.today().isoformat())
    if not force_refresh:
        cached = cache.get(key, max_age=_CACHE_TTL)
        if cached is not None:
            _log.debug("Opta league rankings from cache (%d leagues)", len(cached))
            return cached

    rankings = _scrape_league_rankings()
    if rankings:
        cache.set(key, rankings)
    return rankings


def get_team_rankings_dict(
    force_refresh: bool = False,
) -> Dict[str, OptaTeamRanking]:
    """Return Opta team rankings keyed by ``contestantName`` (case-sensitive)."""
    return {r.team: r for r in get_team_rankings(force_refresh)}


def get_league_rankings_dict(
    force_refresh: bool = False,
) -> Dict[str, OptaLeagueRanking]:
    """Return Opta league rankings keyed by league name."""
    return {r.league: r for r in get_league_rankings(force_refresh)}
