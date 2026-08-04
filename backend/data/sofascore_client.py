"""Sofascore player stats client via direct HTTP API calls.

Sofascore returns raw totals; per-90 values are computed here as:
    per_90 = total / (minutes_played / 90)
Percentage stats (e.g. pass_completion_pct) are stored as-is.
All external calls are routed through backend.data.cache.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote as _url_quote

import requests as _stdlib_requests

# Prefer curl_cffi for Cloudflare bypass (TLS fingerprint impersonation).
# Sofascore hard-blocks stdlib requests with HTTP 403 regardless of headers,
# because it fingerprints the TLS/JA3 handshake rather than the User-Agent.
# curl_cffi therefore is not an optimisation — it is the only working transport.
_CURL_IMPERSONATE = os.environ.get("SOFASCORE_IMPERSONATE", "chrome120")
try:
    from curl_cffi.requests import Session as _CurlSession
    _CURL_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover - depends on install
    _CurlSession = None
    _CURL_AVAILABLE = False
    logging.getLogger(__name__).error(
        "curl_cffi is not installed (%s). Sofascore blocks stdlib requests with "
        "HTTP 403, so every player/stat lookup will fail. Install it with "
        "`pip install curl-cffi`.",
        _exc,
    )

from backend.data import cache

_log = logging.getLogger(__name__)

# ── HTTP transport ───────────────────────────────────────────────────────────
# The session is held in thread-local storage: curl_cffi Sessions are not
# documented as thread-safe, and Streamlit reruns scripts on worker threads.
# Sharing one module-level Session across threads produced sporadic exceptions
# which the old code treated as "curl_cffi is broken" and responded to by
# permanently downgrading the whole process to stdlib requests — guaranteeing
# HTTP 403 on every subsequent call.  Sessions are now per-thread and are
# rebuilt on error rather than abandoned.
_thread_local = threading.local()


def _get_session():
    """Return this thread's HTTP session, creating it on first use."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = _new_session()
        _thread_local.session = session
    return session


def _new_session():
    """Create a fresh session, preferring curl_cffi's TLS impersonation."""
    if _CURL_AVAILABLE:
        try:
            return _CurlSession(impersonate=_CURL_IMPERSONATE)
        except Exception as exc:  # pragma: no cover - defensive
            _log.error(
                "Could not create curl_cffi session (%s) — falling back to stdlib "
                "requests, which Sofascore blocks with HTTP 403.", exc,
            )
    return _stdlib_requests


def _reset_session() -> None:
    """Discard this thread's session so the next call builds a clean one."""
    _thread_local.session = None


def using_curl_impersonation() -> bool:
    """Return True when the working transport can bypass Sofascore's TLS block."""
    return _CURL_AVAILABLE


# ── Request counter (timing instrumentation) ─────────────────────────────────
# Incremented on every live HTTP call (not cache hits).  Read by training
# pipeline every 500 samples to measure Sofascore network pressure.
http_call_count: int = 0

# ── Metric definitions (unchanged — canonical names are source-agnostic) ─────

CORE_METRICS: list[str] = [
    "expected_goals",
    "expected_assists",
    "shots",
    "successful_dribbles",
    "successful_crosses",
    "touches_in_opposition_box",
    "successful_passes",
    "pass_completion_pct",
    "accurate_long_balls",
    "chances_created",
    "clearances",
    "interceptions",
    "possession_won_final_3rd",
]

ADDITIONAL_METRICS: list[str] = [
    "xg_on_target",
    "non_penalty_xg",
    "dispossessed",
    "duels_won_pct",
    "aerial_duels_won_pct",
    "recoveries",
    "fouls_won",
    "touches",
    "goals_conceded_on_pitch",
    "xg_against_on_pitch",
]

ALL_METRICS: list[str] = CORE_METRICS + ADDITIONAL_METRICS

# ── Metric category sets (paper-aligned) ─────────────────────────────────────
# Used by prediction fallbacks to apply different adjustment rates.

OFFENSIVE_METRICS: frozenset[str] = frozenset({
    "expected_goals", "expected_assists", "shots",
    "successful_dribbles", "successful_crosses",
    "touches_in_opposition_box", "chances_created",
})

DEFENSIVE_METRICS: frozenset[str] = frozenset({
    "clearances", "interceptions", "possession_won_final_3rd",
})

# ── Sofascore stat key → canonical name ──────────────────────────────────────
# Sofascore returns a flat dict of raw totals.  Multiple aliases are listed
# to guard against minor API key variations across seasons/versions.
_SOFASCORE_KEY_MAP: dict[str, str] = {
    # xG / xA
    "expectedGoals": "expected_goals",
    "xG": "expected_goals",
    # xA
    "expectedAssists": "expected_assists",
    "xA": "expected_assists",
    # Shots (total)
    "shots": "shots",
    "totalShots": "shots",
    "shotAttempts": "shots",
    # Dribbles
    "successfulDribbles": "successful_dribbles",
    "dribbles": "successful_dribbles",
    # Crosses
    "accurateCrosses": "successful_crosses",
    "crossesAccurate": "successful_crosses",
    # Touches in box — Sofascore uses varying key names across seasons/endpoints
    "penaltyAreaTouches": "touches_in_opposition_box",
    "touchInBox": "touches_in_opposition_box",
    "touchesInOppositionBox": "touches_in_opposition_box",
    "touchesInPenaltyArea": "touches_in_opposition_box",
    "penAreaEntries": "touches_in_opposition_box",
    "penaltyAreaEntries": "touches_in_opposition_box",
    "boxTouches": "touches_in_opposition_box",
    "touchesInTheBox": "touches_in_opposition_box",
    "touchInPenaltyArea": "touches_in_opposition_box",
    "penAreaTouches": "touches_in_opposition_box",
    "totalTouchesInPenaltyArea": "touches_in_opposition_box",
    "totalTouchInBox": "touches_in_opposition_box",
    # Passes
    "accuratePasses": "successful_passes",
    "passesAccurate": "successful_passes",
    # Pass completion % (percentage — kept as-is, not converted to per-90)
    "accuratePassesPercentage": "pass_completion_pct",
    "passAccuracy": "pass_completion_pct",
    "passAccuracyPercentage": "pass_completion_pct",
    # Long balls
    "accurateLongBalls": "accurate_long_balls",
    "longBallsAccurate": "accurate_long_balls",
    # Chances created / key passes
    "keyPasses": "chances_created",
    "bigChancesCreated": "chances_created",
    "chancesCreated": "chances_created",
    # Defensive — own third (clearances)
    "clearances": "clearances",
    # Defensive — mid third (interceptions)
    "interceptions": "interceptions",
    # Defensive — att third (won tackles as proxy for possession won final 3rd)
    "wonTackles": "possession_won_final_3rd",
    "tacklesWon": "possession_won_final_3rd",
    "successfulTackles": "possession_won_final_3rd",
    # Additional metrics
    "expectedGoalsOnTarget": "xg_on_target",
    "xGOT": "xg_on_target",
    "expectedGoalsNoPenalty": "non_penalty_xg",
    "nonPenaltyXg": "non_penalty_xg",
    "npxG": "non_penalty_xg",
    "dispossessed": "dispossessed",
    # Duels won % (percentage — kept as-is)
    "duelsWonPercentage": "duels_won_pct",
    "duelsWon%": "duels_won_pct",
    "totalDuelsWonPercentage": "duels_won_pct",
    # Aerial duels won % (percentage — kept as-is)
    "aerialDuelsWonPercentage": "aerial_duels_won_pct",
    "aerialDuelsWon%": "aerial_duels_won_pct",
    # Recoveries
    "ballRecovery": "recoveries",
    "recoveries": "recoveries",
    # Fouls won (drawn)
    "foulsDrawn": "fouls_won",
    "foulsWon": "fouls_won",
    "wasFouled": "fouls_won",
    # Touches
    "touches": "touches",
    # Goals conceded while on pitch
    "goalsConceded": "goals_conceded_on_pitch",
    "goalsConcededOnPitch": "goals_conceded_on_pitch",
    # xG against while on pitch (often unavailable in Sofascore — returns None)
    "xGAgainst": "xg_against_on_pitch",
    "expectedGoalsAgainst": "xg_against_on_pitch",
}

# Metrics that are percentages and must NOT be divided by minutes
_PERCENTAGE_METRICS: frozenset[str] = frozenset(
    ["pass_completion_pct", "duels_won_pct", "aerial_duels_won_pct"]
)

_BASE_URL = "https://api.sofascore.com/api/v1"
_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}

_REQUEST_TIMEOUT = int(os.environ.get("SOFASCORE_REQUEST_TIMEOUT", "10"))
_MAX_RETRIES = int(os.environ.get("SOFASCORE_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY = float(os.environ.get("SOFASCORE_RETRY_BASE_DELAY", "1.0"))
_RETRY_MAX_DELAY = 30.0   # cap per-retry backoff to avoid indefinite hangs
_RETRY_JITTER_MAX = 0.5   # random jitter range added to each retry delay
_RETRYABLE_STATUS_CODES = {403, 429, 500, 502, 503, 504}
_DEFAULT_INTER_REQUEST_DELAY = float(
    os.environ.get("SOFASCORE_INTER_REQUEST_DELAY", "0.5")
)

# ── Adaptive rate-limiting state ─────────────────────────────────────────────
# When 403/429 responses are encountered, the inter-request delay is
# automatically increased so that subsequent calls back off without the
# caller needing to manually adjust --api-delay.
_adaptive_delay: float = _DEFAULT_INTER_REQUEST_DELAY
_adaptive_delay_floor: float = _DEFAULT_INTER_REQUEST_DELAY
_ADAPTIVE_DELAY_MULTIPLIER = float(
    os.environ.get("SOFASCORE_ADAPTIVE_DELAY_MULTIPLIER", "2.0")
)
_ADAPTIVE_DELAY_MAX = float(os.environ.get("SOFASCORE_ADAPTIVE_DELAY_MAX", "4.0"))
_has_made_request = False  # skip delay before the very first request


def set_inter_request_delay(seconds: float) -> None:
    """Set the base inter-request delay (called by ``--api-delay``).

    Also resets the adaptive delay so callers can start fresh.
    """
    global _adaptive_delay, _adaptive_delay_floor, _has_made_request
    _adaptive_delay_floor = max(seconds, 0.0)
    _adaptive_delay = _adaptive_delay_floor
    _has_made_request = False


def _bump_adaptive_delay() -> None:
    """Increase the adaptive inter-request delay after a rate-limit hit."""
    global _adaptive_delay
    # Floor at 0.5s so multiplying can't get stuck at zero
    _adaptive_delay = min(
        max(_adaptive_delay, 0.5) * _ADAPTIVE_DELAY_MULTIPLIER,
        _ADAPTIVE_DELAY_MAX,
    )
    _log.info("Adaptive rate-limit delay increased to %.1fs", _adaptive_delay)


_ADAPTIVE_JITTER_MAX = 0.3  # jitter range for inter-request throttle


def _inter_request_delay() -> None:
    """Sleep between API calls to avoid triggering rate limits.

    Skips the very first call so startup is not delayed.
    """
    global _has_made_request
    if not _has_made_request:
        _has_made_request = True
        return
    # base delay + small random jitter to avoid predictable patterns
    time.sleep(_adaptive_delay + random.uniform(0, _ADAPTIVE_JITTER_MAX))


def _retry_delay(attempt: int) -> float:
    """Compute retry delay: exponential backoff capped + jitter."""
    return min(_RETRY_BASE_DELAY * (2 ** attempt), _RETRY_MAX_DELAY) + random.uniform(0, _RETRY_JITTER_MAX)


_NEGATIVE_CACHE_TTL = 86400  # 24 hours — cache dead endpoints to avoid re-fetching
_NEGATIVE_SENTINEL = "__NEGATIVE__"  # Marker stored in cache for None results
# Squad profiles (market value, contract) move slowly — cache for a week.
_SQUAD_PROFILE_TTL = 604800


def _get(path: str) -> Optional[dict]:
    """Execute a GET request against the Sofascore API with retry.

    Retries up to ``_MAX_RETRIES`` times with exponential backoff for
    transient HTTP errors (403 TLS block, 429 rate-limit, 5xx server errors).
    Returns the parsed JSON dict, or None on any permanent error.

    HTTP 404 and other non-retryable failures are cached for 24 hours
    so the same dead endpoint is not re-fetched on the next pipeline run.
    Transient errors (403, 429, 5xx) are NOT cached.

    Transport errors rebuild this thread's session rather than downgrading to
    stdlib ``requests``: Sofascore blocks stdlib with HTTP 403, so a permanent
    downgrade turns one transient blip into total, silent failure.
    """
    # Check negative cache — avoid re-fetching known-dead endpoints
    neg_key = cache.make_key("sofascore_neg", path)
    neg_cached = cache.get(neg_key, max_age=_NEGATIVE_CACHE_TTL)
    if neg_cached is not None:
        return None

    url = f"{_BASE_URL}{path}"
    _had_transient_failure = False  # Track if failure was due to transient errors (403/429/5xx/connection)
    for attempt in range(_MAX_RETRIES):
        try:
            # Adaptive throttle before every outbound request (including
            # retries) so the bumped delay is honoured immediately.
            _inter_request_delay()
            global http_call_count
            http_call_count += 1
            resp = _get_session().get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
            if resp.status_code in _RETRYABLE_STATUS_CODES:
                _had_transient_failure = True
                if resp.status_code in (403, 429):
                    _bump_adaptive_delay()
                if resp.status_code == 403:
                    # 403 here is a TLS-fingerprint block, not rate limiting.
                    # Sleeping alone never clears it — rebuild the session so
                    # the next attempt presents a fresh handshake.
                    _reset_session()
                    if not _CURL_AVAILABLE:
                        _log.error(
                            "Sofascore returned 403 and curl_cffi is not installed — "
                            "install it with `pip install curl-cffi`.",
                        )
                # Exponential backoff: base * 2^attempt + jitter (cap 30s)
                delay = _retry_delay(attempt)
                _log.info(
                    "Sofascore %d on %s — retry %d/%d in %.1fs",
                    resp.status_code, path, attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue
            # Non-retryable HTTP errors — cache and return None immediately.
            # Explicit check avoids raise_for_status() whose exception
            # hierarchy differs between curl_cffi and stdlib requests.
            if resp.status_code >= 400:
                _log.warning("Sofascore HTTP %d on %s", resp.status_code, path)
                cache.set(neg_key, _NEGATIVE_SENTINEL)
                return None
            return resp.json()
        except Exception as exc:
            # Any transport-level failure (connection reset, timeout, or a
            # curl_cffi-specific error) is treated as transient.  Rebuild the
            # session and back off; never abandon TLS impersonation, because
            # the stdlib fallback is guaranteed to be blocked.
            _had_transient_failure = True
            _reset_session()
            delay = _retry_delay(attempt)
            _log.info(
                "Sofascore transport error on %s — retry %d/%d in %.1fs (%s: %s)",
                path, attempt + 1, _MAX_RETRIES, delay, type(exc).__name__, exc,
            )
            time.sleep(delay)
    _log.warning("Sofascore request failed after %d retries: %s", _MAX_RETRIES, path)
    # Do NOT cache transient failures (403/429/5xx/connection) — they should be retried next run
    if not _had_transient_failure:
        cache.set(neg_key, _NEGATIVE_SENTINEL)
    return None


# ── Public API ────────────────────────────────────────────────────────────────


def search_team(name: str) -> List[Dict[str, Any]]:
    """Search Sofascore for a team by name.

    Returns a list of dicts with ``id``, ``name``, and optional
    ``tournament_id``.
    """
    key = cache.make_key("sofascore_team_search", name.lower().strip())
    cached = cache.get(key, max_age=86400 * 7)
    if cached:
        return cached

    raw = _get(f"/search/teams?q={_url_quote(name)}&page=0")
    teams: list[dict] = []

    if isinstance(raw, dict):
        results = raw.get("results", [])
        for item in results:
            entity = item.get("entity") or item
            if not isinstance(entity, dict):
                continue
            team_id = entity.get("id")
            team_name = entity.get("name") or entity.get("shortName", "")
            if not team_id or not team_name:
                continue

            entry: dict[str, Any] = {"id": team_id, "name": team_name}

            tournament_id = _extract_unique_tournament_id(entity)
            if tournament_id:
                entry["tournament_id"] = tournament_id

            country = entity.get("country") or {}
            if isinstance(country, dict) and country.get("name"):
                entry["country"] = country["name"]

            teams.append(entry)

    cache.set(key, teams)
    return teams


# Sofascore numeric transfer-type codes → human-readable labels
_TRANSFER_TYPE_MAP: Dict[int, str] = {
    1: "Transfer",
    2: "Loan",
    3: "Loan return",
    4: "Free transfer",
    5: "Swap",
}


def _normalize_transfer_type(raw: Any) -> str:
    """Convert a Sofascore transfer type code to a readable label."""
    if isinstance(raw, int):
        return _TRANSFER_TYPE_MAP.get(raw, "Unknown")
    if isinstance(raw, str):
        # Already a string — title-case it for consistency
        return raw.strip().title() if raw.strip() else "N/A"
    return "N/A"


def get_player_heatmap(
    player_id: int,
    tournament_id: int,
    season_id: int,
) -> List[Dict[str, int]]:
    """Fetch a player's season touch heatmap.

    Returns a list of ``{"x": int, "y": int, "count": int}`` points on a
    0-100 pitch grid (0,0 = own goal-line, left touchline), or ``[]`` when
    Sofascore has no heatmap for that player/season.

    This is the only working source of positional data in the project.  The
    WhoScored client is dead (every endpoint returns 404/406) and StatsBomb's
    open data covers only a small fraction of current players, whereas this
    endpoint resolved for 7 of 8 sampled Premier League players — the eighth
    had left the league, so the 404 was correct.
    """
    if player_id <= 0 or tournament_id <= 0 or season_id <= 0:
        return []

    raw = _get(
        f"/player/{player_id}/unique-tournament/{tournament_id}"
        f"/season/{season_id}/heatmap/overall"
    )
    if not isinstance(raw, dict):
        return []

    points = raw.get("points")
    if not isinstance(points, list):
        return []

    cleaned: List[Dict[str, int]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        try:
            cleaned.append({
                "x": int(point["x"]),
                "y": int(point["y"]),
                "count": int(point.get("count", 1)),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return cleaned


# Pitch-territory thresholds on Sofascore's 0-100 x-axis, where x is distance
# toward the opposition goal.
_FINAL_THIRD_X = 200.0 / 3      # 66.7
_OWN_THIRD_X = 100.0 / 3        # 33.3
_BOX_X = 83.0                   # approx 18-yard line
# The y-axis runs right-to-left from the attacking player's perspective:
# low y is the RIGHT flank, high y is the LEFT.  Verified against known
# wingers — Saka (right) sits 84% in the low-y band, Martinelli (left) 60%
# in the high-y band.  Labelling these the intuitive way round would have
# reported every winger on the wrong flank.
_RIGHT_Y_MAX = 100.0 / 3
_LEFT_Y_MIN = 200.0 / 3


def compute_territory_features(
    player_id: int,
    tournament_id: int,
    season_id: int,
) -> Dict[str, float]:
    """Summarise where on the pitch a player actually operates.

    Derived from the season heatmap, weighting each grid cell by its touch
    count.  Returns fractions in 0-1:

    - ``territory_final_third`` / ``territory_middle_third`` / ``territory_own_third``
    - ``territory_box`` — share of touches inside the opposition box
    - ``territory_left`` / ``territory_central`` / ``territory_right``
    - ``territory_avg_x`` / ``territory_avg_y`` — mean position (0-100)
    - ``territory_width`` — spread across the pitch, as a standard deviation

    Returns ``{}`` when no heatmap exists, so callers can distinguish "we do
    not know" from "this player never enters the final third".
    """
    points = get_player_heatmap(player_id, tournament_id, season_id)
    if not points:
        return {}

    total = sum(p["count"] for p in points)
    if total <= 0:
        return {}

    def share(predicate) -> float:
        return sum(p["count"] for p in points if predicate(p)) / total

    mean_x = sum(p["x"] * p["count"] for p in points) / total
    mean_y = sum(p["y"] * p["count"] for p in points) / total
    variance_y = sum(((p["y"] - mean_y) ** 2) * p["count"] for p in points) / total

    return {
        "territory_final_third": round(share(lambda p: p["x"] >= _FINAL_THIRD_X), 4),
        "territory_middle_third": round(
            share(lambda p: _OWN_THIRD_X < p["x"] < _FINAL_THIRD_X), 4
        ),
        "territory_own_third": round(share(lambda p: p["x"] <= _OWN_THIRD_X), 4),
        "territory_box": round(share(lambda p: p["x"] >= _BOX_X), 4),
        "territory_right": round(share(lambda p: p["y"] <= _RIGHT_Y_MAX), 4),
        "territory_central": round(
            share(lambda p: _RIGHT_Y_MAX < p["y"] < _LEFT_Y_MIN), 4
        ),
        "territory_left": round(share(lambda p: p["y"] >= _LEFT_Y_MIN), 4),
        "territory_avg_x": round(mean_x, 2),
        "territory_avg_y": round(mean_y, 2),
        "territory_width": round(variance_y ** 0.5, 2),
    }


def get_player_transfer_history(player_id: int) -> List[Dict[str, Any]]:
    """Fetch a player's transfer history from Sofascore.

    Returns a list of transfer dicts (most recent first), each with:
        - ``transfer_date``: ISO date string or None
        - ``from_team``: dict with ``id`` and ``name``
        - ``to_team``: dict with ``id`` and ``name``
        - ``type``: transfer type string (e.g. "transfer", "loan")
    """
    key = cache.make_key("sofascore_transfers", str(player_id))
    cached = cache.get(key, max_age=86400 * 7)
    if cached:
        return cached

    raw = _get(f"/player/{player_id}/transfer-history")
    transfers: list[dict] = []

    if isinstance(raw, dict):
        entries = raw.get("transferHistory", [])
        for entry in entries:
            if not isinstance(entry, dict):
                continue

            from_team = entry.get("transferFrom") or {}
            to_team = entry.get("transferTo") or {}

            t: dict[str, Any] = {
                "transfer_date": _unix_to_iso(entry.get("transferDateTimestamp")),
                "from_team": {
                    "id": from_team.get("id"),
                    "name": from_team.get("name", ""),
                },
                "to_team": {
                    "id": to_team.get("id"),
                    "name": to_team.get("name", ""),
                },
                "type": _normalize_transfer_type(
                    entry.get("type") or entry.get("transferType", "")
                ),
            }
            transfers.append(t)

    cache.set(key, transfers)
    return transfers


def get_league_player_stats(
    tournament_id: int,
    season_id: Optional[int] = None,
    limit: int = 200,
    enrich_profiles: bool = True,
) -> List[Dict[str, Any]]:
    """Fetch aggregated player stats for an entire league/tournament season.

    Attempts the batch statistics endpoint first.  When it returns 404
    (common for many Sofascore tournament/season combos), falls back to
    discovering teams via the standings endpoint, fetching each team's
    roster, and calling the per-player statistics endpoint which is
    reliably available.

    Results are cached for 1 hour.

    Parameters
    ----------
    tournament_id : int
        Sofascore unique-tournament ID.
    season_id : int, optional
        Specific season ID.  If ``None``, the current season is fetched.
    limit : int
        Maximum number of players to return (default 200).
    enrich_profiles : bool
        When True (default), join squad-level profile data — market value,
        contract expiry, height and preferred foot.  The statistics endpoints
        carry none of these, so they come from one request per club (cached
        for a week).  Set False to skip that cost when the caller only needs
        per-90 output.

    Returns
    -------
    list[dict] — each dict has ``id``, ``name``, ``team``, ``team_id``,
    ``position``, ``age``, ``minutes_played``, ``per90``, ``rating``, and
    (when enriched) ``market_value``, ``contract_until``,
    ``contract_years_left``, ``height_cm``, ``weight_kg``, ``preferred_foot``.
    Enriched fields are ``None`` when Sofascore has no data for that player —
    never 0, so that "unknown price" is never mistaken for "cheap".
    """
    if season_id is None:
        season_id = _get_current_season_id(tournament_id)
    if season_id is None:
        return []

    # Check result-level cache (includes limit and enrichment flag, since
    # enriched and unenriched results have different shapes)
    key = cache.make_key(
        "sofascore_league_stats", str(tournament_id), str(season_id),
        str(limit), "enriched" if enrich_profiles else "plain",
    )
    cached = cache.get(key, max_age=3600)
    if cached is not None:
        return cached

    # ── Attempt 1: batch statistics endpoint ─────────────────────────────
    batch_key = cache.make_key(
        "sofascore_league_batch", str(tournament_id), str(season_id),
    )
    batch_cached = cache.get(batch_key, max_age=3600)

    if batch_cached is not None:
        batch_raw = batch_cached
    else:
        batch_raw = _get(
            f"/unique-tournament/{tournament_id}/season/{season_id}"
            f"/statistics/overall"
        )
        if isinstance(batch_raw, dict):
            cache.set(batch_key, batch_raw)

    if isinstance(batch_raw, dict):
        result = _parse_batch_league_stats(batch_raw, limit)
        if result:
            if enrich_profiles:
                result = _enrich_with_squad_profiles(
                    result, tournament_id, season_id,
                )
            cache.set(key, result)
            return result

    # ── Attempt 2: per-player fallback via standings + roster ────────────
    _log.info(
        "get_league_player_stats: batch endpoint unavailable for tid=%d "
        "sid=%d, falling back to per-player stats",
        tournament_id, season_id,
    )
    result = _league_stats_per_player_fallback(
        tournament_id, season_id, limit,
    )
    if result:
        if enrich_profiles:
            result = _enrich_with_squad_profiles(
                result, tournament_id, season_id,
            )
        cache.set(key, result)
    return result


# Profile fields joined onto league player stats.  Always present after
# enrichment (as None when unknown) so downstream consumers can rely on the key.
_PROFILE_FIELDS = (
    "market_value",
    "contract_until",
    "contract_years_left",
    "height_cm",
    "weight_kg",
    "preferred_foot",
)


def _enrich_with_squad_profiles(
    players: List[Dict[str, Any]],
    tournament_id: int,
    season_id: int,
) -> List[Dict[str, Any]]:
    """Join market value / contract data onto league player stats.

    Sofascore's statistics endpoints omit market value entirely, so it is
    fetched per club (one request per team, cached for a week) and joined by
    player id here.  Enrichment is best-effort: if the squad lookup fails the
    players are returned unchanged with ``None`` profile fields rather than
    the whole league failing.

    Missing values stay ``None``.  They must never be coerced to 0, which
    would make an unpriced player look free to any value-based ranking.
    """
    try:
        profiles = get_league_squad_profiles(tournament_id, season_id)
    except Exception as exc:  # pragma: no cover - defensive
        _log.warning(
            "Squad profile enrichment failed for tid=%s sid=%s: %s",
            tournament_id, season_id, exc,
        )
        profiles = {}

    matched = 0
    for player in players:
        pid = player.get("id")
        profile = profiles.get(pid) if pid is not None else None
        if profile:
            matched += 1
        for field in _PROFILE_FIELDS:
            # Preserve any value already present; fill the rest from the
            # squad profile, defaulting to None rather than 0.
            if player.get(field) is None:
                player[field] = (profile or {}).get(field)
        # Squad profiles carry an exact date of birth, so prefer that age.
        if profile and profile.get("age") is not None:
            player["age"] = profile["age"]

    if players:
        _log.info(
            "Squad profile enrichment: %d/%d players matched (tid=%s)",
            matched, len(players), tournament_id,
        )
    return players


def _parse_batch_league_stats(
    batch_raw: dict,
    limit: int,
) -> List[Dict[str, Any]]:
    """Parse the batch statistics response into a list of player dicts."""
    entries = batch_raw.get("results") or batch_raw.get("players") or []

    players_map: Dict[int, Dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if len(players_map) >= limit:
            break

        player_data = entry.get("player") or {}
        if not isinstance(player_data, dict):
            continue

        pid = player_data.get("id")
        if pid is None or pid in players_map:
            continue

        team_data = entry.get("team") or {}
        team_id = team_data.get("id")
        team_name = team_data.get("name", "")

        # Statistics may be nested under "statistics" or at top level
        stats = entry.get("statistics") or {}
        if not isinstance(stats, dict):
            continue

        # minutesPlayed may also be at the entry level
        minutes = int(
            stats.get("minutesPlayed")
            or entry.get("minutesPlayed")
            or 0
        )
        if minutes == 0:
            continue

        per90 = _parse_stats(stats, minutes)

        # Age from dateOfBirthTimestamp
        dob_ts = player_data.get("dateOfBirthTimestamp")
        player_age = None
        if dob_ts is not None:
            try:
                age_seconds = time.time() - int(dob_ts)
                if age_seconds > 0:
                    player_age = int(age_seconds / (365.25 * 86400))
            except (ValueError, TypeError):
                pass

        # Rating (Sofascore 0-10 scale)
        avg_rating = stats.get("rating")
        if avg_rating is not None:
            try:
                avg_rating = float(avg_rating)
            except (ValueError, TypeError):
                avg_rating = None

        players_map[pid] = {
            "id": pid,
            "name": player_data.get("name") or player_data.get("shortName", ""),
            "team": team_name,
            "team_id": team_id,
            "position": _map_position(
                player_data.get("position") or ""
            ),
            "age": player_age,
            "minutes_played": minutes,
            "per90": per90,
            "rating": avg_rating,
        }

    return list(players_map.values())


def get_team_squad_profiles(team_id: int) -> Dict[int, Dict[str, Any]]:
    """Return squad profile data for one club, keyed by player id.

    The ``/team/{id}/players`` endpoint returns market value, contract expiry,
    height, weight and preferred foot for an entire squad in a **single**
    request.  The bulk league-statistics endpoint carries none of these, and
    fetching them per player would cost hundreds of calls per league, so
    enrichment is done per club instead (~20 calls per league).

    Values are returned as-is from Sofascore:

    - ``market_value``      : EUR, or None when Sofascore has no valuation
    - ``contract_until``    : unix timestamp of contract expiry, or None
    - ``contract_years_left``: float years from now, or None
    - ``height_cm`` / ``weight_kg`` / ``preferred_foot`` / ``age``

    Coverage is roughly 90% at major clubs and degrades in minor leagues, so
    every consumer must treat missing values as *unknown*, never as zero.
    """
    cache_key = cache.make_key("sofascore_squad_profiles", str(team_id))
    cached = cache.get(cache_key, max_age=_SQUAD_PROFILE_TTL)
    if cached is not None:
        return {int(k): v for k, v in cached.items()}

    raw = _get(f"/team/{team_id}/players")
    profiles: Dict[int, Dict[str, Any]] = {}

    if isinstance(raw, dict):
        entries = raw.get("players") or []
        now = time.time()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            player = entry.get("player") or {}
            if not isinstance(player, dict):
                continue
            pid = player.get("id")
            if pid is None:
                continue

            market_value = player.get("proposedMarketValue")
            if market_value is None:
                raw_mv = player.get("proposedMarketValueRaw")
                if isinstance(raw_mv, dict):
                    market_value = raw_mv.get("value")
            try:
                market_value = float(market_value) if market_value else None
            except (TypeError, ValueError):
                market_value = None

            contract_until = player.get("contractUntilTimestamp")
            contract_years_left = None
            if contract_until:
                try:
                    contract_until = int(contract_until)
                    contract_years_left = round(
                        (contract_until - now) / (365.25 * 86400), 2
                    )
                except (TypeError, ValueError):
                    contract_until = None

            age = None
            dob_ts = player.get("dateOfBirthTimestamp")
            if dob_ts is not None:
                try:
                    age_seconds = now - int(dob_ts)
                    if age_seconds > 0:
                        age = int(age_seconds / (365.25 * 86400))
                except (TypeError, ValueError):
                    pass

            def _num(value):
                try:
                    return float(value) if value else None
                except (TypeError, ValueError):
                    return None

            profiles[int(pid)] = {
                "market_value": market_value,
                "contract_until": contract_until,
                "contract_years_left": contract_years_left,
                "height_cm": _num(player.get("height")),
                "weight_kg": _num(player.get("weight")),
                "preferred_foot": player.get("preferredFoot") or None,
                "age": age,
            }

    if profiles:
        cache.set(cache_key, {str(k): v for k, v in profiles.items()})
    return profiles


def get_league_squad_profiles(
    tournament_id: int,
    season_id: int,
    max_teams: int = 40,
) -> Dict[int, Dict[str, Any]]:
    """Aggregate squad profiles for every club in a league.

    Costs one request per club (cached for a week), rather than one per player.
    Returns an empty dict when the league's teams cannot be resolved, so
    callers degrade to "no value data" instead of failing.
    """
    teams = _get_league_team_ids(tournament_id, season_id)
    combined: Dict[int, Dict[str, Any]] = {}
    for team in teams[:max_teams]:
        tid = team.get("id")
        if not tid:
            continue
        try:
            combined.update(get_team_squad_profiles(int(tid)))
        except Exception as exc:  # pragma: no cover - defensive
            _log.warning("Squad profile fetch failed for team %s: %s", tid, exc)
    return combined


def _get_league_team_ids(
    tournament_id: int,
    season_id: int,
) -> List[Dict[str, Any]]:
    """Get teams in a league via the standings endpoint.

    Returns list of dicts with ``id`` and ``name`` for each team.
    """
    standings_key = cache.make_key(
        "sofascore_standings", str(tournament_id), str(season_id),
    )
    cached = cache.get(standings_key, max_age=86400)
    if cached is not None:
        return cached

    raw = _get(
        f"/unique-tournament/{tournament_id}/season/{season_id}"
        f"/standings/total"
    )

    teams: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()

    if isinstance(raw, dict):
        for group in raw.get("standings", []):
            if not isinstance(group, dict):
                continue
            for row in group.get("rows", []):
                if not isinstance(row, dict):
                    continue
                team_data = row.get("team") or {}
                tid = team_data.get("id")
                tname = team_data.get("name", "")
                if tid and tid not in seen_ids:
                    teams.append({"id": tid, "name": tname})
                    seen_ids.add(tid)

    if teams:
        cache.set(standings_key, teams)
    return teams


def _league_stats_per_player_fallback(
    tournament_id: int,
    season_id: int,
    limit: int,
) -> List[Dict[str, Any]]:
    """Fallback: build league player stats from per-player API calls.

    1. Discover teams via standings endpoint.
    2. Fetch each team's roster via ``get_team_players_stats()``.
    3. For each rostered player, call ``get_player_stats_for_season()``.
    4. Return results in the same format as the batch endpoint.
    """
    teams = _get_league_team_ids(tournament_id, season_id)
    if not teams:
        _log.warning(
            "get_league_player_stats fallback: no teams found for tid=%d sid=%d",
            tournament_id, season_id,
        )
        return []

    players_map: Dict[int, Dict[str, Any]] = {}

    for team_info in teams:
        if len(players_map) >= limit:
            break

        team_id = team_info["id"]
        team_name = team_info["name"]
        roster = get_team_players_stats(team_id)

        for player in roster:
            if len(players_map) >= limit:
                break

            pid = player.get("id")
            if not pid or pid in players_map:
                continue

            stats = get_player_stats_for_season(pid, tournament_id, season_id)
            if not stats:
                continue

            minutes = stats.get("minutes_played", 0)
            if minutes == 0:
                continue

            per90 = stats.get("per90") or {}
            position = stats.get("position") or _map_position(
                player.get("position", "")
            )

            players_map[pid] = {
                "id": pid,
                "name": stats.get("name") or player.get("name", ""),
                "team": team_name,
                "team_id": team_id,
                "position": position,
                "age": stats.get("age"),
                "minutes_played": minutes,
                "per90": per90,
                "rating": stats.get("rating"),
            }

    return list(players_map.values())


def get_player_season_stats(
    player_id: int,
    tournament_id: int,
    season_id: int,
) -> Optional[Dict[str, Any]]:
    """Look up a single player's season stats from the batch league endpoint.

    Calls :func:`get_league_player_stats` (which caches the full batch
    response) and returns the stats dict for *player_id*, or ``None`` if
    the player is not found in the batch results.

    Returns
    -------
    dict or None — the player dict with ``id``, ``name``, ``team``,
    ``team_id``, ``position``, ``age``, ``minutes_played``, ``per90``,
    ``rating``, or ``None`` if not found.
    """
    all_players = get_league_player_stats(tournament_id, season_id=season_id)
    for p in all_players:
        if p.get("id") == player_id:
            return p
    return None


def get_season_list(tournament_id: int) -> List[Dict[str, Any]]:
    """Return the list of available seasons for a tournament.

    Each item has ``id`` (season_id) and ``name`` (e.g. ``"2024/2025"``).
    Newest season first.
    """
    key = cache.make_key("sofascore_season_list", str(tournament_id))
    cached = cache.get(key, max_age=86400)
    if cached:
        return cached

    raw = _get(f"/unique-tournament/{tournament_id}/seasons")
    if not isinstance(raw, dict):
        _log.warning(
            "get_season_list(%d): API returned %s instead of dict — not caching",
            tournament_id,
            type(raw).__name__,
        )
        return []

    seasons = raw.get("seasons") or []
    result = [
        {"id": s.get("id"), "name": s.get("name", "")}
        for s in seasons
        if isinstance(s, dict) and s.get("id") is not None
    ]

    if not result:
        _log.warning(
            "get_season_list(%d): API returned 0 valid seasons (raw keys: %s) — not caching",
            tournament_id,
            list(raw.keys()),
        )
        return []

    cache.set(key, result)
    return result


def get_player_match_logs(
    player_id: int,
    tournament_id: int,
    season_id: int,
) -> List[Dict[str, Any]]:
    """Fetch per-match player stats for a specific tournament + season.

    Uses the Sofascore events endpoint to retrieve match-by-match data.
    Paginates from page 0 (most recent) upward until empty or page > 10.

    Returns a list of match dicts sorted by ``match_date`` **ascending**
    (oldest first), suitable for rolling window accumulation.  Each dict:
    ``match_id``, ``match_date``, ``minutes_played``,
    ``per90`` (dict of canonical metric -> float).

    Matches with ``minutes_played`` of 0 or None are excluded.
    If fewer than 3 valid matches are found, returns ``[]``.
    """
    key = cache.make_key(
        "sofascore", "match_logs",
        str(player_id), str(tournament_id), str(season_id),
    )
    cached = cache.get(key, max_age=86400 * 7)
    if cached:
        return cached

    matches: List[Dict[str, Any]] = []
    max_page = 10  # safety ceiling

    for page in range(max_page + 1):
        raw = _get(
            f"/player/{player_id}/unique-tournament/{tournament_id}"
            f"/season/{season_id}/events/last/{page}"
        )
        if not isinstance(raw, dict):
            break

        events = raw.get("events") or []
        if not events:
            break

        for event in events:
            if not isinstance(event, dict):
                continue

            match_id = event.get("id")
            # Extract date from startTimestamp
            start_ts = event.get("startTimestamp")
            match_date = _unix_to_iso(start_ts)

            # Player statistics may be nested under "statistics" or "playerStatistics"
            stats_container = event.get("statistics") or event.get("playerStatistics") or {}
            if not isinstance(stats_container, dict):
                stats_container = {}

            minutes_played_raw = stats_container.get("minutesPlayed")
            if minutes_played_raw is None:
                # Try alternate locations
                minutes_played_raw = event.get("minutesPlayed")
            try:
                minutes_played = int(minutes_played_raw)
            except (ValueError, TypeError):
                continue
            if minutes_played <= 0:
                continue

            per90 = _parse_stats(stats_container, minutes_played)

            matches.append({
                "match_id": match_id,
                "match_date": match_date or "",
                "minutes_played": minutes_played,
                "per90": {m: per90.get(m) for m in CORE_METRICS},
            })

    # Sort by match_date ascending (oldest first)
    matches.sort(key=lambda m: m.get("match_date", ""))

    # Fewer than 3 valid matches → unreliable data
    if len(matches) < 3:
        result: List[Dict[str, Any]] = []
    else:
        result = matches

    cache.set(key, result)
    return result


def get_player_stats_for_season(
    player_id: int,
    tournament_id: int,
    season_id: int,
) -> Dict[str, Any]:
    """Fetch player stats for a specific tournament + season combination.

    Unlike ``get_player_stats`` which auto-discovers the current season,
    this function targets an explicit season.
    """
    key = cache.make_key(
        "sofascore_player_season",
        str(player_id),
        str(tournament_id),
        str(season_id),
    )
    cached = cache.get(key, max_age=86400)
    if cached:
        return cached

    # Get player profile for name/team
    profile_raw = _get(f"/player/{player_id}")
    result = _make_empty_result()

    if isinstance(profile_raw, dict):
        player_data = profile_raw.get("player") or profile_raw
        if isinstance(player_data, dict):
            result["name"] = player_data.get("name") or player_data.get("shortName", "")
            team_data = player_data.get("team") or {}
            if isinstance(team_data, dict):
                result["team"] = team_data.get("name", "")
                result["team_id"] = team_data.get("id")
            position_data = player_data.get("position") or {}
            result["position"] = _map_position(
                player_data.get("positionDescription", {}) or position_data
            )

    stats_raw = _get(
        f"/player/{player_id}/unique-tournament/{tournament_id}"
        f"/season/{season_id}/statistics/overall"
    )

    if isinstance(stats_raw, dict):
        stats = stats_raw.get("statistics") or {}
        if isinstance(stats, dict):
            result["minutes_played"] = int(stats.get("minutesPlayed") or 0)
            result["appearances"] = int(
                stats.get("appearances") or stats.get("matchesStarted") or 0
            )
            result["per90"] = _parse_stats(stats, result["minutes_played"])
            result["raw"] = stats_raw
    else:
        result["raw"] = {}

    cache.set(key, result)
    return result


def search_player(name: str) -> List[Dict[str, Any]]:
    """Search Sofascore for a player by name.

    Returns a list of dicts, each with at least ``id`` and ``name``.
    Also includes ``age``, ``nationality``, and ``team_name`` when
    available from the search response.
    Caches ``tournament_id`` and ``season_id`` per player for use
    by ``get_player_stats``.
    """
    key = cache.make_key("sofascore_search", name.lower().strip())
    cached = cache.get(key, max_age=86400 * 7)
    if cached:
        return cached

    raw = _get(f"/search/players?q={_url_quote(name)}&page=0")
    players: list[dict] = []

    if isinstance(raw, dict):
        results = raw.get("results", [])
        for item in results:
            entity = item.get("entity") or item
            if not isinstance(entity, dict):
                continue
            player_id = entity.get("id")
            player_name = entity.get("name") or entity.get("shortName", "")
            if not player_id or not player_name:
                continue

            entry: dict[str, Any] = {"id": player_id, "name": player_name}

            # Age — Sofascore may return dateOfBirthTimestamp
            dob_ts = entity.get("dateOfBirthTimestamp")
            if dob_ts is not None:
                try:
                    from datetime import datetime, timezone
                    born = datetime.fromtimestamp(int(dob_ts), tz=timezone.utc)
                    now = datetime.now(tz=timezone.utc)
                    entry["age"] = now.year - born.year - (
                        (now.month, now.day) < (born.month, born.day)
                    )
                except (ValueError, TypeError, OSError):
                    pass

            # Nationality
            country = entity.get("country") or {}
            if isinstance(country, dict) and country.get("name"):
                entry["nationality"] = country["name"]

            # Extract and cache tournament/season info when available
            team = entity.get("team") or {}
            if isinstance(team, dict) and team.get("name"):
                entry["team_name"] = team["name"]
                entry["team_id"] = team.get("id")
            tournament_id = _extract_unique_tournament_id(team, entity)
            if tournament_id:
                entry["tournament_id"] = tournament_id
                # Stash meta for get_player_stats fast-path
                _cache_player_meta(player_id, tournament_id)

            players.append(entry)

    cache.set(key, players)
    return players


def get_player_stats(
    player_id: int,
    season: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch full player stats from Sofascore and return structured data.

    Parameters
    ----------
    player_id : int
        Sofascore player ID.
    season : str, optional
        Season string like ``"2024/2025"`` (currently unused — always
        fetches the current season for the player's league).

    Returns
    -------
    dict with keys:
        - ``name``: player display name
        - ``team``: current team name
        - ``team_id``: current team Sofascore ID
        - ``position``: primary position label
        - ``minutes_played``: total minutes this season
        - ``appearances``: number of appearances
        - ``per90``: dict mapping each of the 23 metrics to its per-90 value
          (``None`` for unavailable metrics)
        - ``raw``: the original Sofascore statistics JSON
    """
    key = cache.make_key("sofascore_player", str(player_id), season or "current")
    cached = cache.get(key, max_age=86400)
    if cached:
        return cached

    # Step 1 — Get player profile to resolve team + tournament
    profile_raw = _get(f"/player/{player_id}")
    result = _make_empty_result()

    if isinstance(profile_raw, dict):
        player_data = profile_raw.get("player") or profile_raw
        if isinstance(player_data, dict):
            result["name"] = player_data.get("name") or player_data.get("shortName", "")

            # Team
            team_data = player_data.get("team") or {}
            if isinstance(team_data, dict):
                result["team"] = team_data.get("name", "")
                result["team_id"] = team_data.get("id")

            # Tournament — check multiple locations in Sofascore response
            tournament_id = _extract_unique_tournament_id(
                team_data, player_data, profile_raw,
            )
            if tournament_id:
                _cache_player_meta(player_id, tournament_id)

            # Position
            position_data = player_data.get("position") or {}
            result["position"] = _map_position(
                player_data.get("positionDescription", {}) or position_data
            )

            # Age — compute from dateOfBirthTimestamp if available
            dob_ts = player_data.get("dateOfBirthTimestamp")
            if dob_ts is not None:
                try:
                    age_seconds = time.time() - int(dob_ts)
                    if age_seconds > 0:
                        result["age"] = int(age_seconds / (365.25 * 86400))
                except (ValueError, TypeError):
                    pass

    # Step 2 — Discover current tournament + season
    tournament_id = _get_cached_tournament_id(player_id)

    # Fallback: if no tournament_id found yet, try the team's tournaments endpoint
    if not tournament_id and result.get("team_id"):
        tournament_id = _discover_tournament_for_team(result["team_id"])
        if tournament_id:
            _cache_player_meta(player_id, tournament_id)

    season_id = None
    if tournament_id:
        season_id = _get_current_season_id(tournament_id)

    # Step 3 — Fetch statistics if we have the required IDs.
    #
    # The newest season in Sofascore's list starts existing before a ball is
    # kicked, so between roughly May and August the "current" season returns
    # 404 / empty stats for everyone.  That is precisely the transfer window,
    # when this tool is most used, so fall back through recent seasons until
    # one has real minutes rather than returning an empty player.
    stats_raw: Optional[dict] = None
    if tournament_id and season_id:
        for candidate_season in _recent_season_ids(tournament_id, season_id):
            candidate_raw = _get(
                f"/player/{player_id}/unique-tournament/{tournament_id}"
                f"/season/{candidate_season}/statistics/overall"
            )
            if _has_usable_minutes(candidate_raw):
                stats_raw = candidate_raw
                if candidate_season != season_id:
                    _log.info(
                        "Player %d has no stats for season %s — using %s instead",
                        player_id, season_id, candidate_season,
                    )
                    season_id = candidate_season
                break
        result["season_id"] = season_id

    if isinstance(stats_raw, dict):
        stats = stats_raw.get("statistics") or {}
        if isinstance(stats, dict):
            result["minutes_played"] = int(stats.get("minutesPlayed") or 0)
            result["appearances"] = int(
                stats.get("appearances") or stats.get("matchesStarted") or 0
            )
            result["per90"] = _parse_stats(stats, result["minutes_played"])
            result["raw"] = stats_raw
            # Extract average match rating (Sofascore 0-10 scale)
            avg_rating = stats.get("rating")
            if avg_rating is not None:
                try:
                    result["rating"] = float(avg_rating)
                except (ValueError, TypeError):
                    pass
    else:
        result["raw"] = {}

    # Step 4 — Multi-tournament fallback: if the primary tournament returned
    # 0 minutes, try ALL tournaments the player's team participates in and
    # use the one with the most minutes. This fixes players like Kroupi who
    # have significant minutes across cup/European competitions but 0 in the
    # primary domestic league season.
    if result["minutes_played"] == 0 and result.get("team_id"):
        best = _try_all_tournaments_for_player(
            player_id, result["team_id"], tournament_id,
        )
        if best is not None:
            best_stats, best_tid, best_sid = best
            b_stats = best_stats.get("statistics") or {}
            if isinstance(b_stats, dict):
                mins = int(b_stats.get("minutesPlayed") or 0)
                if mins > result["minutes_played"]:
                    result["minutes_played"] = mins
                    result["appearances"] = int(
                        b_stats.get("appearances")
                        or b_stats.get("matchesStarted")
                        or 0
                    )
                    result["per90"] = _parse_stats(b_stats, mins)
                    result["raw"] = best_stats
                    avg_rating = b_stats.get("rating")
                    if avg_rating is not None:
                        try:
                            result["rating"] = float(avg_rating)
                        except (ValueError, TypeError):
                            pass
                    # Cache the tournament where data was found, so
                    # subsequent calls and season selectors default to
                    # this tournament (note: the UI season selector
                    # currently only shows seasons for one tournament).
                    _cache_player_meta(player_id, best_tid)

    # Only cache results that have actual data.  If minutes_played == 0
    # the API call likely failed or returned the wrong season — caching
    # that empty result would poison every subsequent call for 24 hours.
    if result["minutes_played"] > 0:
        cache.set(key, result)
    return result


def get_team_players_stats(
    team_id: int,
    season: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch team squad and return a list of basic player entries.

    Returns a list of dicts with at least ``id``, ``name``, and ``position``.
    Detailed stats are fetched separately via ``get_player_stats``.
    """
    key = cache.make_key("sofascore_team", str(team_id), season or "current")
    cached = cache.get(key, max_age=86400)
    if cached:
        return cached

    raw = _get(f"/team/{team_id}/players")
    players: list[dict] = []

    if isinstance(raw, dict):
        for entry in raw.get("players", []):
            if not isinstance(entry, dict):
                continue

            # ── Format A (current API): flat list with "player" key ──────
            if "player" in entry and isinstance(entry["player"], dict):
                pdata = entry["player"]
                player_id = pdata.get("id")
                player_name = (
                    pdata.get("name")
                    or pdata.get("shortName", "")
                )
                pos = normalize_position(
                    _map_position(pdata.get("position", ""))
                )
                if player_id and player_name:
                    players.append(
                        {
                            "id": player_id,
                            "name": player_name,
                            "position": pos,
                        }
                    )
                continue

            # ── Format B (legacy/grouped): groups with nested "players" ──
            group_name = entry.get("name") or entry.get("title", "Unknown")
            for member in entry.get("players") or entry.get("members") or []:
                if not isinstance(member, dict):
                    continue
                player_id = member.get("id")
                player_name = member.get("name") or member.get("shortName", "")
                if player_id and player_name:
                    players.append(
                        {
                            "id": player_id,
                            "name": player_name,
                            "position": group_name,
                        }
                    )

    cache.set(key, players)
    return players


# ── Position categories ──────────────────────────────────────────────────────

_POSITION_CATEGORIES: Dict[str, str] = {
    # Forward / Attacker variants
    "forward": "Forward", "forwards": "Forward", "attacker": "Forward",
    "attackers": "Forward", "striker": "Forward", "centre-forward": "Forward",
    "center forward": "Forward", "cf": "Forward", "st": "Forward",
    "right winger": "Forward", "left winger": "Forward", "winger": "Forward",
    "rw": "Forward", "lw": "Forward", "wing": "Forward",
    "f": "Forward",  # Sofascore single-letter code
    # Midfielder variants
    "midfielder": "Midfielder", "midfielders": "Midfielder",
    "central midfielder": "Midfielder", "attacking midfielder": "Midfielder",
    "defensive midfielder": "Midfielder", "cm": "Midfielder",
    "am": "Midfielder", "dm": "Midfielder", "cam": "Midfielder",
    "cdm": "Midfielder", "rm": "Midfielder", "lm": "Midfielder",
    "m": "Midfielder",  # Sofascore single-letter code
    # Defender variants
    "defender": "Defender", "defenders": "Defender",
    "centre-back": "Defender", "center back": "Defender",
    "right back": "Defender", "left back": "Defender",
    "right-back": "Defender", "left-back": "Defender",
    "cb": "Defender", "rb": "Defender", "lb": "Defender",
    "rwb": "Defender", "lwb": "Defender",
    "d": "Defender",  # Sofascore single-letter code
    # Goalkeeper variants
    "goalkeeper": "Goalkeeper", "goalkeepers": "Goalkeeper", "gk": "Goalkeeper",
    "g": "Goalkeeper",  # Sofascore single-letter code
}


def normalize_position(position: str) -> str:
    """Normalize a position string to one of: Forward, Midfielder, Defender, Goalkeeper."""
    return _POSITION_CATEGORIES.get(position.strip().lower(), "Unknown")


def get_team_position_averages(
    team_id: int,
    target_position: str,
    max_players: int = 8,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Compute average per-90 stats for players in a position at a team.

    Fetches the team squad, filters by position category, retrieves
    individual stats for matching players, and returns the mean per-90
    for each core metric.  This captures the team's tactical style for
    that position (paper Section 2.3: team-position features).

    Parameters
    ----------
    team_id : int
        Sofascore team ID.
    target_position : str
        Position to match (e.g. "Forward", "Right Winger", "Striker").
    max_players : int
        Cap on number of individual player stats to fetch.

    Returns
    -------
    (avg_per90, player_data_list)
        avg_per90: dict mapping metric -> average per-90 across position.
        player_data_list: list of dicts with ``per90``, ``position``, ``name``.
    """
    cache_key = cache.make_key(
        "team_pos_avg", str(team_id), normalize_position(target_position),
    )
    cached = cache.get(cache_key, max_age=86400)
    if cached:
        return cached  # type: ignore[return-value]

    target_cat = normalize_position(target_position)
    squad = get_team_players_stats(team_id)

    # Filter squad members whose position matches the same category
    matching = [
        p for p in squad
        if normalize_position(p.get("position", "")) == target_cat
    ]
    if not matching:
        # No exact position match — return empty averages rather than
        # mixing unrelated positions which would produce misleading data.
        import logging
        logging.getLogger(__name__).info(
            "No %s players found for team %s; returning empty position averages",
            target_cat, team_id,
        )
        empty_avg: Dict[str, float] = {m: 0.0 for m in CORE_METRICS}
        result_empty = (empty_avg, [])
        cache.set(cache_key, result_empty)
        return result_empty

    # Fetch individual stats (limited to avoid too many API calls)
    player_data: List[Dict[str, Any]] = []
    for p in matching[:max_players]:
        pid = p.get("id")
        if not pid:
            continue
        try:
            stats = get_player_stats(pid)
            if stats.get("per90"):
                player_data.append(stats)
        except Exception:
            continue

    # Compute average per-90 across matching players
    avg_per90: Dict[str, float] = {}
    for m in CORE_METRICS:
        values = []
        for pd_item in player_data:
            v = pd_item.get("per90", {}).get(m)
            if v is not None:
                values.append(v)
        avg_per90[m] = sum(values) / len(values) if values else 0.0

    result = (avg_per90, player_data)
    cache.set(cache_key, result)
    return result


# ── Internal helpers ──────────────────────────────────────────────────────────


def _extract_unique_tournament_id(*dicts: Any) -> Optional[int]:
    """Extract the unique tournament ID from Sofascore response dicts.

    Sofascore nests tournament info in several places depending on endpoint:
      - ``team.tournament.uniqueTournament.id`` (common in profiles)
      - ``team.tournament.id`` (occasionally, but often the *tournament* id, not unique)
      - ``uniqueTournament.id`` (top-level on stats responses)
      - ``tournament.uniqueTournament.id``

    Checks multiple dicts (team_data, player_data, profile_raw) in order.
    Returns the first valid integer ID found, or None.
    """
    for d in dicts:
        if not isinstance(d, dict):
            continue
        # Direct uniqueTournament at this level
        ut = d.get("uniqueTournament")
        if isinstance(ut, dict) and ut.get("id"):
            return int(ut["id"])
        # Nested under tournament.uniqueTournament
        tournament = d.get("tournament")
        if isinstance(tournament, dict):
            ut2 = tournament.get("uniqueTournament")
            if isinstance(ut2, dict) and ut2.get("id"):
                return int(ut2["id"])
            # Fallback: tournament.id itself (may be the unique tournament ID
            # in some API versions / mock data)
            tid = tournament.get("id")
            if tid is not None:
                return int(tid)
    return None


def discover_tournament_for_team(team_id: int) -> Optional[int]:
    """Public wrapper — see :func:`_discover_tournament_for_team`."""
    return _discover_tournament_for_team(team_id)


# Known international club competition tournament IDs on Sofascore.
# These must never be returned as a team's "domestic" league.
_INTERNATIONAL_TOURNAMENT_IDS: frozenset = frozenset({
    7,    # UEFA Champions League
    679,  # UEFA Europa League
    73,   # UEFA Europa Conference League
    384,  # UEFA Super Cup
    498,  # FIFA Club World Cup
    480,  # Copa Libertadores
    133,  # Copa Sudamericana
    851,  # AFC Champions League
})

# Continental / international alpha2 codes used by Sofascore that do NOT
# represent a real country.  Champions League uses "EU", for example.
_NON_COUNTRY_ALPHA2: frozenset = frozenset({"EU", "INT", "WW"})


def _discover_tournament_for_team(team_id: int) -> Optional[int]:
    """Fetch the primary unique tournament ID for a team via Sofascore API.

    Uses ``/team/{team_id}/unique-tournaments`` and picks the first
    domestic league tournament (highest userCount, non-international).
    Falls back to the first result if none match.
    """
    key = cache.make_key("sofascore_team_tournament", str(team_id))
    cached = cache.get(key, max_age=86400 * 7)
    if cached:
        return cached

    raw = _get(f"/team/{team_id}/unique-tournaments")
    if not isinstance(raw, dict):
        return None

    tournaments = raw.get("uniqueTournaments") or []
    if not tournaments:
        return None

    # Prefer domestic league (non-international, highest userCount)
    best = None
    best_count = -1
    for t in tournaments:
        if not isinstance(t, dict):
            continue
        tid = t.get("id")
        if tid is None:
            continue

        # Skip known international club competitions by ID
        if int(tid) in _INTERNATIONAL_TOURNAMENT_IDS:
            continue

        # Skip tournaments whose category uses a continental alpha2 code
        # (e.g. Champions League has alpha2="EU", not a real country)
        cat = t.get("category") or {}
        alpha2 = cat.get("alpha2") or ""
        if alpha2 in _NON_COUNTRY_ALPHA2:
            continue

        is_domestic = bool(alpha2)
        user_count = t.get("userCount") or 0
        if is_domestic and user_count > best_count:
            best = int(tid)
            best_count = user_count

    # Fallback to first tournament if no domestic league found
    if best is None and tournaments:
        first = tournaments[0]
        if isinstance(first, dict) and first.get("id"):
            best = int(first["id"])

    if best is not None:
        cache.set(key, best)
    return best


def _try_all_tournaments_for_player(
    player_id: int,
    team_id: int,
    already_tried_tid: Optional[int] = None,
) -> Optional[tuple]:
    """Try all tournaments for a team and return stats from the one with most minutes.

    When the primary domestic tournament returns 0 minutes, this function
    iterates through every tournament the team participates in (cups,
    European competitions, secondary divisions) to find one where the
    player actually has data.

    Returns ``(stats_raw, tournament_id, season_id)`` for the tournament
    with the most minutes, or ``None`` if no tournament yields data.
    """
    raw = _get(f"/team/{team_id}/unique-tournaments")
    if not isinstance(raw, dict):
        return None

    tournaments = raw.get("uniqueTournaments") or []
    if not tournaments:
        return None

    best_mins = 0
    best_result: Optional[tuple] = None

    for t in tournaments:
        if not isinstance(t, dict):
            continue
        tid = t.get("id")
        if tid is None:
            continue
        tid = int(tid)
        if tid == already_tried_tid:
            continue  # already fetched this one

        sid = _get_current_season_id(tid)
        if sid is None:
            continue

        stats_raw = _get(
            f"/player/{player_id}/unique-tournament/{tid}"
            f"/season/{sid}/statistics/overall"
        )
        if not isinstance(stats_raw, dict):
            continue

        stats = stats_raw.get("statistics") or {}
        mins = int(stats.get("minutesPlayed") or 0)
        if mins > best_mins:
            best_mins = mins
            best_result = (stats_raw, tid, sid)

    return best_result


def _make_empty_result() -> Dict[str, Any]:
    """Return a blank result dict matching the public API shape."""
    return {
        "name": "",
        "team": "",
        "team_id": None,
        "position": "",
        "age": None,
        "minutes_played": 0,
        "appearances": 0,
        "per90": {m: None for m in ALL_METRICS},
        "rating": None,
        "raw": {},
    }


def _parse_stats(stats: dict, minutes_played: int) -> Dict[str, Optional[float]]:
    """Map a Sofascore statistics dict to our canonical per-90 values.

    Raw totals are divided by (minutes_played / 90).
    Percentage metrics are stored as-is.
    """
    per90: Dict[str, Optional[float]] = {m: None for m in ALL_METRICS}
    nineties = minutes_played / 90.0 if minutes_played > 0 else None

    for raw_key, value in stats.items():
        canonical = _SOFASCORE_KEY_MAP.get(raw_key)
        if canonical is None or canonical not in per90:
            continue
        if value is None:
            continue
        try:
            fval = float(value)
        except (ValueError, TypeError):
            continue

        if canonical in _PERCENTAGE_METRICS:
            # Percentages stored directly
            per90[canonical] = fval
        elif nineties and nineties > 0:
            per90[canonical] = round(fval / nineties, 4)
        # If no minutes, leave as None (data unavailable)

    # Fallback for touches_in_opposition_box: Sofascore may not always
    # provide this stat directly.  When missing, estimate from total
    # touches using a position-based ratio (attackers ~15-20% of touches
    # are in the box, midfielders ~8-12%, defenders ~3-5%).
    #
    # Estimation constants (educated estimates calibrated to paper case studies):
    #   BOX_TOUCHES_PER_SHOT — each shot implies ~2.5 box touches on average
    #     (typical attackers have ~2-3x more box touches than shots per game)
    #   MAX_BOX_TOUCH_RATIO  — cap at 30% of total touches (realistic upper bound
    #     even for elite strikers; most attackers are 15-25%)
    #   DEFAULT_BOX_RATIO    — generic fallback when no shots data available
    #     (~10% is typical for midfielders/mixed positions)
    _BOX_TOUCHES_PER_SHOT = 2.5
    _MAX_BOX_TOUCH_RATIO = 0.30
    _DEFAULT_BOX_RATIO = 0.10

    if per90.get("touches_in_opposition_box") is None and per90.get("touches") is not None:
        total_touches = per90["touches"]
        if total_touches is not None and total_touches > 0:
            shots = per90.get("shots")
            if shots is not None and shots > 0:
                # Players who shoot more tend to be in the box more
                estimated_box_touches = round(shots * _BOX_TOUCHES_PER_SHOT, 4)
                per90["touches_in_opposition_box"] = min(
                    estimated_box_touches, total_touches * _MAX_BOX_TOUCH_RATIO
                )
            else:
                per90["touches_in_opposition_box"] = round(
                    total_touches * _DEFAULT_BOX_RATIO, 4
                )

    return per90


def _map_position(position_data: Any) -> str:
    """Extract a human-readable position label from Sofascore position data."""
    if isinstance(position_data, str):
        return position_data
    if isinstance(position_data, dict):
        # Try common Sofascore position description shapes
        for key in ("primaryPosition", "position", "name"):
            val = position_data.get(key)
            if val:
                if isinstance(val, dict):
                    return val.get("name") or val.get("label") or ""
                return str(val)
    return ""


def _cache_player_meta(player_id: int, tournament_id: int) -> None:
    """Cache the mapping of player_id → tournament_id."""
    meta_key = cache.make_key("sofascore_player_meta", str(player_id))
    existing = cache.get(meta_key, max_age=86400 * 30)
    if existing is None:
        cache.set(meta_key, {"tournament_id": tournament_id})


def _get_cached_tournament_id(player_id: int) -> Optional[int]:
    """Look up cached tournament_id for a player."""
    meta_key = cache.make_key("sofascore_player_meta", str(player_id))
    meta = cache.get(meta_key, max_age=86400 * 30)
    if isinstance(meta, dict):
        return meta.get("tournament_id")
    return None


def get_cached_tournament_id(player_id: int) -> Optional[int]:
    """Public wrapper — see :func:`_get_cached_tournament_id`."""
    return _get_cached_tournament_id(player_id)


def _unix_to_iso(ts: Any) -> Optional[str]:
    """Convert a Unix timestamp to ISO-8601 date string, or return None."""
    if ts is None:
        return None
    try:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, TypeError, OSError):
        return None


def _has_usable_minutes(stats_raw: Any) -> bool:
    """Return True when a statistics payload contains real playing time."""
    if not isinstance(stats_raw, dict):
        return False
    stats = stats_raw.get("statistics")
    if not isinstance(stats, dict):
        return False
    try:
        return int(stats.get("minutesPlayed") or 0) > 0
    except (TypeError, ValueError):
        return False


# How many seasons to walk back when the newest one has no data yet.
_SEASON_FALLBACK_DEPTH = int(os.environ.get("SOFASCORE_SEASON_FALLBACK_DEPTH", "3"))


def _recent_season_ids(
    tournament_id: int,
    preferred_season_id: int,
) -> List[int]:
    """Return ``preferred_season_id`` followed by the seasons just before it.

    Used to survive the pre-season gap, when the newest season exists in
    Sofascore's season list but no matches have been played yet.
    """
    ordered: List[int] = [int(preferred_season_id)]
    try:
        seasons = get_season_list(tournament_id) or []
    except Exception:  # pragma: no cover - defensive
        return ordered

    ids: List[int] = []
    for season in seasons:
        sid = season.get("id") if isinstance(season, dict) else None
        if sid is None:
            continue
        try:
            ids.append(int(sid))
        except (TypeError, ValueError):
            continue

    # get_season_list is newest-first; start immediately after the preferred one.
    if int(preferred_season_id) in ids:
        start = ids.index(int(preferred_season_id)) + 1
    else:
        start = 0

    for sid in ids[start:]:
        if sid not in ordered:
            ordered.append(sid)
        if len(ordered) >= _SEASON_FALLBACK_DEPTH:
            break
    return ordered


def _get_current_season_id(tournament_id: int) -> Optional[int]:
    """Return the current (most recent) season ID for a Sofascore tournament.

    Reuses the ``get_season_list`` cache when available so the same
    ``/seasons`` endpoint is not fetched twice.
    """
    key = cache.make_key("sofascore_seasons", str(tournament_id))
    cached = cache.get(key, max_age=86400)  # refresh daily
    if cached:
        return cached

    # Try the season_list cache first (populated by get_season_list)
    seasons = get_season_list(tournament_id)
    if seasons:
        season_id = seasons[0].get("id")
        if season_id is not None:
            cache.set(key, season_id)
        return season_id

    return None
