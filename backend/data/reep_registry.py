"""REEP Register — CSV-based entity resolution for teams and people.

Uses the `withqwerty/reep` open-data register to map identifiers across
data providers (Sofascore, ClubElo, Transfermarkt, FBref, etc.).

Data files:
    - teams.csv  (~45 000 clubs, keyed by Wikidata QID)
    - people.csv (~430 000 players/coaches, keyed by Wikidata QID)

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

# ── Internal helpers ─────────────────────────────────────────────────────────


def _fetch_csv(url: str, cache_key: str) -> Optional[pd.DataFrame]:
    """Download a CSV from *url*, cache the raw text, return a DataFrame."""
    cached = cache.get(cache_key)
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

    cache.set(cache_key, raw, ttl=_CACHE_TTL)
    try:
        return pd.read_csv(io.StringIO(raw), low_memory=False)
    except Exception as exc:
        _log.warning("REEP CSV parse failed (%s): %s", url, exc)
        return None


# ── Public API ───────────────────────────────────────────────────────────────


def get_teams_df() -> Optional[pd.DataFrame]:
    """Return the REEP teams DataFrame (cached for 7 days)."""
    return _fetch_csv(_TEAMS_URL, "reep:teams_csv")


def get_people_df() -> Optional[pd.DataFrame]:
    """Return the REEP people DataFrame (cached for 7 days)."""
    return _fetch_csv(_PEOPLE_URL, "reep:people_csv")


def build_clubelo_sofascore_map() -> Dict[str, str]:
    """Build a ``{ClubElo key → team display name}`` dict from REEP.

    Uses the ``key_clubelo`` and ``name`` columns from teams.csv.
    This can replace the hand-maintained ``_CLUBELO_TO_SOFASCORE`` dict
    in ``power_rankings.py``.

    Returns an empty dict if the download fails.
    """
    df = get_teams_df()
    if df is None:
        return {}

    # Keep only rows where both columns are present
    mask = df["key_clubelo"].notna() & df["name"].notna()
    subset = df.loc[mask, ["key_clubelo", "name"]]
    return dict(zip(subset["key_clubelo"].astype(str), subset["name"].astype(str)))


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


def enrich_player(sofascore_player_id: int) -> Dict[str, Any]:
    """Return metadata from REEP people.csv for a Sofascore player ID.

    Returns a dict with keys ``nationality``, ``height_cm``,
    ``date_of_birth``, ``position`` (any may be *None*).
    Returns an empty dict on miss or download failure.
    """
    df = get_people_df()
    if df is None:
        return {}

    mask = df["key_sofascore"].astype(str) == str(sofascore_player_id)
    rows = df.loc[mask]
    if rows.empty:
        return {}

    row = rows.iloc[0]
    result: Dict[str, Any] = {}

    for col, key in [
        ("nationality", "nationality"),
        ("height_cm", "height_cm"),
        ("date_of_birth", "date_of_birth"),
        ("position", "position"),
    ]:
        val = row.get(col)
        if pd.notna(val):
            result[key] = val if col != "height_cm" else _safe_int(val)
        else:
            result[key] = None

    return result


def _safe_int(val: Any) -> Optional[int]:
    """Convert to int, returning None on failure."""
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None
