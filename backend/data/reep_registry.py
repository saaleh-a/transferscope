"""REEP Register — CSV-based entity resolution for teams and people.

Uses the `withqwerty/reep` open-data register to map identifiers across
data providers (Sofascore, ClubElo, Transfermarkt, FBref, etc.).

Data files:
    - teams.csv  (~45,000 clubs, keyed by Wikidata QID)
    - people.csv (~430,000 players/coaches, keyed by Wikidata QID)

Both files are downloaded once from GitHub and cached locally via
*diskcache* with a 7-day TTL.  Subsequent imports reuse the cache.

Public API
----------
get_teams_df()
    Returns the full teams DataFrame (cached).
get_people_df()
    Returns the full people DataFrame (cached).
clubelo_to_sofascore_name(clubelo_key)
    Resolve a ClubElo slug (e.g. ``"Arsenal"``) to the Sofascore display
    name (e.g. ``"Arsenal"``).  Returns *None* on miss.
build_clubelo_sofascore_map()
    Returns ``Dict[str, str]`` mapping every ClubElo key found in REEP to
    the corresponding team name, suitable as a drop-in replacement for the
    hand-maintained ``_CLUBELO_TO_SOFASCORE`` dict.
sofascore_team_aliases(sofascore_id)
    Returns a list of known name variants for a Sofascore team ID,
    useful for supplementing ``_EXTREME_ABBREVS``.
enrich_player(sofascore_player_id)
    Returns a dict with ``nationality``, ``height_cm``, ``date_of_birth``,
    and ``position`` from REEP people.csv (or empty dict on miss).
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import requests as _requests_lib

from backend.data import cache

_log = logging.getLogger(__name__)

_TEAMS_URL = (
    "https://raw.githubusercontent.com/withqwerty/reep/main/data/teams.csv"
)
_PEOPLE_URL = (
    "https://raw.githubusercontent.com/withqwerty/reep/main/data/people.csv"
)

# Cache TTL — 7 days (same as player stats).
_CACHE_TTL = 86400 * 7

# ── In-memory singletons (populated on first access) ────────────────────────
_teams_df_mem: Optional[pd.DataFrame] = None
_people_df_mem: Optional[pd.DataFrame] = None
# Pre-indexed lookup: sofascore_player_id (str) → dict of row values.
_people_index: Optional[Dict[str, Dict[str, Any]]] = None
# Cached ClubElo → display-name mapping.
_clubelo_map_mem: Optional[Dict[str, str]] = None

# ── Internal helpers ─────────────────────────────────────────────────────────


def clear_memory_cache() -> None:
    """Reset all in-memory caches.  Useful in tests."""
    global _teams_df_mem, _people_df_mem, _people_index, _clubelo_map_mem
    _teams_df_mem = None
    _people_df_mem = None
    _people_index = None
    _clubelo_map_mem = None


def _fetch_csv(url: str, cache_key: str) -> Optional[pd.DataFrame]:
    """Download a CSV from *url*, cache the raw text, return a DataFrame."""
    cached = cache.get(cache_key, max_age=_CACHE_TTL)
    if cached is not None:
        try:
            return pd.read_csv(io.StringIO(cached), low_memory=False)
        except Exception:
            pass  # stale / corrupt — refetch

    _log.info("REEP: downloading %s …", url)
    try:
        resp = _requests_lib.get(url, timeout=30)
        resp.raise_for_status()
        raw = resp.text
    except Exception as exc:
        _log.warning("REEP download failed (%s): %s", url, exc)
        return None

    cache.set(cache_key, raw)
    try:
        return pd.read_csv(io.StringIO(raw), low_memory=False)
    except Exception as exc:
        _log.warning("REEP CSV parse failed (%s): %s", url, exc)
        return None


# ── Public API ───────────────────────────────────────────────────────────────


def get_teams_df() -> Optional[pd.DataFrame]:
    """Return the REEP teams DataFrame (cached in-memory after first load)."""
    global _teams_df_mem
    if _teams_df_mem is not None:
        return _teams_df_mem
    df = _fetch_csv(_TEAMS_URL, "reep:teams_csv")
    if df is not None:
        _teams_df_mem = df
    return df


def get_people_df() -> Optional[pd.DataFrame]:
    """Return the REEP people DataFrame (cached in-memory after first load)."""
    global _people_df_mem
    if _people_df_mem is not None:
        return _people_df_mem
    df = _fetch_csv(_PEOPLE_URL, "reep:people_csv")
    if df is not None:
        _people_df_mem = df
    return df


def build_clubelo_sofascore_map() -> Dict[str, str]:
    """Build a ``{ClubElo key → team display name}`` dict from REEP.

    Uses the ``key_clubelo`` and ``name`` columns from teams.csv.
    This can replace the hand-maintained ``_CLUBELO_TO_SOFASCORE`` dict
    in ``power_rankings.py``.

    The result is cached in-memory after the first call.
    Returns an empty dict if the download fails.
    """
    global _clubelo_map_mem
    if _clubelo_map_mem is not None:
        return _clubelo_map_mem

    df = get_teams_df()
    if df is None:
        return {}

    # Keep only rows where both columns are present
    mask = df["key_clubelo"].notna() & df["name"].notna()
    subset = df.loc[mask, ["key_clubelo", "name"]]
    result = dict(zip(subset["key_clubelo"].astype(str), subset["name"].astype(str)))
    _clubelo_map_mem = result
    return result


def clubelo_to_sofascore_name(clubelo_key: str) -> Optional[str]:
    """Resolve a single ClubElo key to the team's display name."""
    mapping = build_clubelo_sofascore_map()
    return mapping.get(clubelo_key)


def sofascore_team_aliases(sofascore_id: int) -> List[str]:
    """Return known name variants for a Sofascore team ID.

    Collects: REEP canonical ``name``, plus any Transfermarkt / FBref /
    ClubElo names that can be inferred from the same row.
    Useful for supplementing the fuzzy-matching alias table.
    """
    df = get_teams_df()
    if df is None:
        return []

    mask = df["key_sofascore"].astype(str) == str(sofascore_id)
    rows = df.loc[mask]
    if rows.empty:
        return []

    aliases: List[str] = []
    for _, row in rows.iterrows():
        name = row.get("name")
        if pd.notna(name) and str(name).strip():
            aliases.append(str(name).strip())
        clubelo = row.get("key_clubelo")
        if pd.notna(clubelo) and str(clubelo).strip():
            aliases.append(str(clubelo).strip())
    return list(dict.fromkeys(aliases))  # deduplicate, preserve order


def _build_people_index(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Build an O(1) lookup dict keyed by sofascore player ID (str)."""
    index: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        sid = row.get("key_sofascore")
        if pd.notna(sid):
            index[str(int(float(sid)))] = {
                "nationality": row.get("nationality") if pd.notna(row.get("nationality")) else None,
                "height_cm": _safe_int(row.get("height_cm")) if pd.notna(row.get("height_cm")) else None,
                "date_of_birth": row.get("date_of_birth") if pd.notna(row.get("date_of_birth")) else None,
                "position": row.get("position") if pd.notna(row.get("position")) else None,
                "whoscored_id": _safe_int(row.get("key_whoscored")) if pd.notna(row.get("key_whoscored")) else None,
            }
    return index


def enrich_player(sofascore_player_id: int) -> Dict[str, Any]:
    """Return metadata from REEP people.csv for a Sofascore player ID.

    Returns a dict with keys ``nationality``, ``height_cm``,
    ``date_of_birth``, ``position``, ``whoscored_id`` (any may be
    *None*).  Returns an empty dict on miss or download failure.

    Uses a pre-built in-memory index for O(1) lookups.
    """
    global _people_index
    if _people_index is None:
        df = get_people_df()
        if df is None:
            return {}
        _people_index = _build_people_index(df)

    entry = _people_index.get(str(sofascore_player_id))
    if entry is None:
        return {}
    # Return a copy so callers can't mutate the cached data.
    return dict(entry)


def _safe_int(val: Any) -> Optional[int]:
    """Convert to int, returning None on failure."""
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None
