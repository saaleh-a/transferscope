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

# The JS bundle is ~17 MB but only regenerated when rankings change (roughly
# weekly).  Cache the *parsed* result for 24 h so we never re-parse in prod.
_CACHE_TTL = 86_400  # 24 h in seconds
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
    country: str = ""               # country the league belongs to
    number_of_teams: int = 0        # official team count for the league


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


def _extract_json_parse(js_text: str, marker: str) -> Optional[str]:
    """Find ``<marker>=JSON.parse(\\`...\\`)`` in *js_text* and return the JSON
    string, ready for ``json.loads()``.

    Uses plain string search — no regex — so it's fast even on 17 MB of JS.

    **Escape fix**: the ``comps`` field embeds Python-style list-of-dicts with
    competition names that were double-escaped during bundle generation.  Any
    competition name containing a double-quote (e.g. "Ligat Ha'al") is stored as
    the 3-byte sequence ``\\\\\"`` (backslash + backslash + double-quote).  In JSON
    that means *escaped-backslash* followed by *end-of-string*, which breaks
    ``json.loads``.  Collapsing it to the 2-byte ``\\"`` (escaped double-quote)
    fixes the parse without affecting any other field.

    Returns ``None`` if the marker is not found.
    """
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


def _parse_change(last_week: Optional[str], current: Optional[str]) -> str:
    """Convert lastWeekGlobalRank / currentGlobalRank to a ±N string."""
    try:
        delta = int(last_week or 0) - int(current or 0)
        if delta > 0:
            return f"+{delta}"
        return str(delta)
    except (TypeError, ValueError):
        return "0"


def _load_team_rankings_from_bundle() -> List[OptaTeamRanking]:
    """Fetch index.js and extract the men's team ranking dataset (``f6``).

    Falls back to an empty list on any parse / network failure.
    """
    js = _fetch_text(_INDEX_JS_URL)
    if not js:
        return []

    raw = _extract_json_parse(js, "f6")
    if not raw:
        _log.warning(
            "Could not find 'f6=JSON.parse(...)' in Opta index.js — "
            "the bundle variable name may have changed after a deploy"
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
                country=e.get("country", ""),
                confederation=e.get("confederation", ""),
                season_avg_rating=float(e.get("seasonAverageRating", 0)),
            ))
        except Exception as exc:
            _log.debug("Skipping malformed Opta team entry: %s — %s", e, exc)

    _log.info("Parsed %d Opta team rankings from bundle", len(rankings))
    return rankings


def _load_league_rankings_from_meta() -> List[OptaLeagueRanking]:
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
                country=str(e.get("country", "") or ""),
                number_of_teams=int(float(e.get("numberOfTeams", 0) or 0)),
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
    """Return today's Opta team rankings, cached for 24 h.

    Fetches ``index.js``, extracts the ``f6`` JSON blob, and parses it.
    On failure returns ``[]`` so callers can fall back gracefully.

    Parameters
    ----------
    force_refresh : bool
        Bypass the cache and re-fetch.
    """
    key = cache.make_key("opta_team_rankings_v2", date.today().isoformat())
    if not force_refresh:
        cached = cache.get(key, max_age=_CACHE_TTL)
        if cached is not None:
            _log.debug("Opta team rankings from cache (%d teams)", len(cached))
            return cached

    rankings = _load_team_rankings_from_bundle()
    if rankings:
        cache.set(key, rankings)
    return rankings


def get_league_rankings(force_refresh: bool = False) -> List[OptaLeagueRanking]:
    """Return today's Opta league rankings from ``league-meta.json``, cached 24 h.

    Parameters
    ----------
    force_refresh : bool
        Bypass the cache and re-fetch.
    """
    key = cache.make_key("opta_league_rankings_v2", date.today().isoformat())
    if not force_refresh:
        cached = cache.get(key, max_age=_CACHE_TTL)
        if cached is not None:
            _log.debug("Opta league rankings from cache (%d leagues)", len(cached))
            return cached

    rankings = _load_league_rankings_from_meta()
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
