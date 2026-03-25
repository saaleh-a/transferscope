"""ClubElo wrapper for European clubs.

Primary path: ``soccerdata.ClubElo`` (uses the full soccerdata stack).
Fallback path: direct HTTP to ``http://api.clubelo.com`` CSV API — used
automatically when soccerdata fails (e.g. TLS library download issues on
Streamlit Community Cloud).

All responses cached with max_age = 1 day.
"""

from __future__ import annotations

import io
import logging
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from backend.data import cache

_log = logging.getLogger(__name__)

_ONE_DAY = 86400  # seconds
_API_BASE = "http://api.clubelo.com"
_HTTP_TIMEOUT = 15

# ── soccerdata primary path ──────────────────────────────────────────────────

_soccerdata_available: Optional[bool] = None  # lazy probe
_soccerdata_last_failure_at: float = 0.0  # timestamp of last soccerdata failure
_SOCCERDATA_RETRY_INTERVAL = 3600  # retry soccerdata every hour after failure


def _try_soccerdata(date_str: str) -> Optional[pd.DataFrame]:
    """Try fetching via soccerdata.ClubElo.  Returns None on any failure."""
    global _soccerdata_available, _soccerdata_last_failure_at
    if _soccerdata_available is False:
        # Retry after the interval expires
        import time
        if (time.time() - _soccerdata_last_failure_at) < _SOCCERDATA_RETRY_INTERVAL:
            return None
        _log.info("Retrying soccerdata after %.0fs cooldown", _SOCCERDATA_RETRY_INTERVAL)
        _soccerdata_available = None  # reset for retry
    try:
        import soccerdata as sd
        ce = sd.ClubElo()
        df = ce.read_by_date(date_str)
        _soccerdata_available = True
        return df
    except Exception as exc:
        _log.info("soccerdata ClubElo unavailable, using HTTP fallback: %s", exc)
        import time
        _soccerdata_available = False
        _soccerdata_last_failure_at = time.time()
        return None


def _try_soccerdata_history(team_name: str) -> Optional[pd.DataFrame]:
    """Try fetching team history via soccerdata."""
    global _soccerdata_available, _soccerdata_last_failure_at
    if _soccerdata_available is False:
        import time
        if (time.time() - _soccerdata_last_failure_at) < _SOCCERDATA_RETRY_INTERVAL:
            return None
        _soccerdata_available = None  # reset for retry
    try:
        import soccerdata as sd
        ce = sd.ClubElo()
        df = ce.read_team_history(team_name)
        _soccerdata_available = True
        return df
    except Exception:
        import time
        _soccerdata_available = False
        _soccerdata_last_failure_at = time.time()
        return None


# ── Direct HTTP fallback ─────────────────────────────────────────────────────

_LEAGUE_NAMES: Dict[str, Dict[int, str]] = {
    "ENG": {1: "ENG-Premier League", 2: "ENG-Championship"},
    "ESP": {1: "ESP-La Liga", 2: "ESP-Segunda División"},
    "GER": {1: "GER-Bundesliga", 2: "GER-2. Bundesliga"},
    "ITA": {1: "ITA-Serie A", 2: "ITA-Serie B"},
    "FRA": {1: "FRA-Ligue 1", 2: "FRA-Ligue 2"},
    "NED": {1: "NED-Eredivisie"},
    "POR": {1: "POR-Liga Portugal"},
    "BEL": {1: "BEL-First Division A"},
    "TUR": {1: "TUR-Süper Lig"},
    "SCO": {1: "SCO-Premiership"},
    "RUS": {1: "RUS-Premier League"},
    "UKR": {1: "UKR-Premier League"},
    "AUT": {1: "AUT-Bundesliga"},
    "SUI": {1: "SUI-Super League"},
    "GRE": {1: "GRE-Super League"},
    "CZE": {1: "CZE-First League"},
    "DEN": {1: "DEN-Superliga"},
    "CRO": {1: "CRO-1. HNL"},
    "SER": {1: "SER-Super Liga"},
    "NOR": {1: "NOR-Eliteserien"},
    "SWE": {1: "SWE-Allsvenskan"},
    "POL": {1: "POL-Ekstraklasa"},
    "ROM": {1: "ROM-Liga I"},
    # ── Additional ClubElo countries ─────────────────────────────────────
    "BUL": {1: "BUL-First League"},
    "HUN": {1: "HUN-NB I"},
    "SVK": {1: "SVK-Super Liga"},
    "SLO": {1: "SLO-PrvaLiga"},
    "BOS": {1: "BOS-Premier Liga"},
    "CYP": {1: "CYP-First Division"},
    "ISR": {1: "ISR-Premier League"},
    "KAZ": {1: "KAZ-Premier League"},
    "FIN": {1: "FIN-Veikkausliiga"},
    "ISL": {1: "ISL-Úrvalsdeild"},
    "IRL": {1: "IRL-Premier Division"},
    "WAL": {1: "WAL-Premier League"},
    "GEO": {1: "GEO-Erovnuli Liga"},
}


def _league_label(row: pd.Series) -> str:
    """Build a ClubElo-style league label like ``ENG-Premier League``."""
    country = str(row.get("country", "")).strip()
    level = row.get("level", 1)
    try:
        level_int = int(level)
    except (ValueError, TypeError):
        level_int = 1
    return _LEAGUE_NAMES.get(country, {}).get(
        level_int, f"{country}-Level {level_int}"
    )


def _is_valid_csv(text: str) -> bool:
    """Return True if *text* looks like valid ClubElo CSV data."""
    if not text:
        return False
    # ClubElo returns HTML error pages on bad requests
    lower = text[:200].lower()
    if "<html" in lower or "<!doctype" in lower:
        return False
    # Verify expected CSV header is present
    first_line = text.split("\n", 1)[0].lower()
    return "club" in first_line and "elo" in first_line


def _fetch_csv(date_str: str) -> Optional[str]:
    """GET the ClubElo CSV for a specific date via direct HTTP."""
    url = f"{_API_BASE}/{date_str}"
    try:
        resp = requests.get(url, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        text = resp.text.strip()
        if not _is_valid_csv(text):
            return None
        return text
    except Exception as exc:
        _log.debug("ClubElo HTTP fetch failed for %s: %s", date_str, exc)
        return None


def _parse_csv(csv_text: str) -> pd.DataFrame:
    """Parse ClubElo CSV into a DataFrame indexed by team name."""
    df = pd.read_csv(io.StringIO(csv_text), header=0)
    df.columns = [c.strip().lower() for c in df.columns]
    if "club" not in df.columns:
        return pd.DataFrame()
    df = df.rename(columns={"club": "team"})
    df["team"] = df["team"].astype(str).str.strip()
    df = df.set_index("team")
    if "country" in df.columns and "level" in df.columns:
        df["league"] = df.apply(_league_label, axis=1)
    return df


def _http_fallback_by_date(date_str: str) -> pd.DataFrame:
    """Fetch all Elo ratings via direct HTTP CSV API."""
    csv_text = _fetch_csv(date_str)
    if csv_text is None:
        return pd.DataFrame()
    return _parse_csv(csv_text)


def _http_fallback_history(team_name: str) -> pd.DataFrame:
    """Fetch team history via direct HTTP CSV API."""
    url = f"{_API_BASE}/{team_name}"
    try:
        resp = requests.get(url, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        text = resp.text.strip()
        if not _is_valid_csv(text):
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(text), header=0)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


# ── Public API ───────────────────────────────────────────────────────────────


def get_all_by_date(
    query_date: Optional[date] = None,
) -> pd.DataFrame:
    """Return all European club Elo ratings for a given date.

    Tries soccerdata first; falls back to direct HTTP CSV on failure.

    Parameters
    ----------
    query_date : date, optional
        Defaults to today (UTC).

    Returns
    -------
    DataFrame with index ``team`` and columns including ``elo`` and ``league``.
    """
    if query_date is None:
        query_date = date.today()
    date_str = query_date.isoformat()

    key = cache.make_key("clubelo_date", date_str)
    cached = cache.get(key, max_age=_ONE_DAY)
    if cached is not None:
        return cached

    # Primary: soccerdata
    df = _try_soccerdata(date_str)

    # Fallback: direct HTTP
    if df is None or df.empty:
        df = _http_fallback_by_date(date_str)

    if df is not None and not df.empty:
        cache.set(key, df)
        return df

    return pd.DataFrame()


def get_team_elo(
    team_name: str,
    query_date: Optional[date] = None,
) -> Optional[float]:
    """Return a single team's Elo on a given date, or None if not found."""
    df = get_all_by_date(query_date)
    if df.empty:
        return None
    if team_name in df.index:
        return float(df.loc[team_name, "elo"])
    return None


def get_team_history(team_name: str) -> pd.DataFrame:
    """Return full historical Elo series for a European club.

    Returns
    -------
    DataFrame with Elo history (may be empty if unavailable).
    """
    key = cache.make_key("clubelo_history", team_name)
    cached = cache.get(key, max_age=_ONE_DAY * 7)
    if cached is not None:
        return cached

    # Primary: soccerdata
    df = _try_soccerdata_history(team_name)

    # Fallback: direct HTTP
    if df is None or df.empty:
        df = _http_fallback_history(team_name)

    if df is not None and not df.empty:
        cache.set(key, df)
        return df

    return pd.DataFrame()


def list_teams_by_league(
    league: str,
    query_date: Optional[date] = None,
) -> List[str]:
    """Return all team names in a given league on a date."""
    df = get_all_by_date(query_date)
    if df.empty or "league" not in df.columns:
        return []
    return list(df[df["league"] == league].index)


def list_leagues(query_date: Optional[date] = None) -> List[str]:
    """Return all unique league codes available on a date."""
    df = get_all_by_date(query_date)
    if df.empty or "league" not in df.columns:
        return []
    return sorted(df["league"].dropna().unique().tolist())


def is_covered(team_name: str, query_date: Optional[date] = None) -> bool:
    """Check whether ClubElo has data for *team_name* on *query_date*."""
    df = get_all_by_date(query_date)
    if df.empty:
        return False
    return team_name in df.index
