"""REEP Register — CSV-based entity resolution for teams and people.

Uses the `withqwerty/reep` open-data register (v2) to map identifiers
across data providers (Sofascore, ClubElo, Transfermarkt, FBref, FotMob,
SkillCorner, Impect, TheSportsDB, Understat, and 30+ more).

Each entity has a stable ``reep_id`` primary key (e.g. ``reep_t03df16bf``
for teams, ``reep_pbdf32f0e`` for people) that is decoupled from Wikidata,
allowing REEP to house entities that only exist in provider-specific
systems.

Data files:
    - data/reep/teams.csv        (~45,000 clubs)
    - data/reep/people.csv       (~430,000 players/coaches)
    - data/reep/competitions.csv (~200 competitions)
    - data/reep/seasons.csv      (~1,200 seasons)

All files are bundled in the repository and loaded from disk on first
access, then cached in-memory for the process lifetime.

Public API
----------
get_teams_df()
    Returns the full teams DataFrame (cached).
get_people_df()
    Returns the full people DataFrame (cached).
get_competitions_df()
    Returns the full competitions DataFrame (cached).
get_seasons_df()
    Returns the full seasons DataFrame (cached).
lookup_team(reep_id)
    Look up a team row by its ``reep_id``.  Returns a dict or None.
lookup_person(reep_id)
    Look up a person row by their ``reep_id``.  Returns a dict or None.
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

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

_log = logging.getLogger(__name__)

# ── Local CSV paths (bundled in data/reep/) ──────────────────────────────────
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "reep",
)
_TEAMS_PATH = os.path.join(_DATA_DIR, "teams.csv")
_PEOPLE_PATH = os.path.join(_DATA_DIR, "people.csv")
_COMPETITIONS_PATH = os.path.join(_DATA_DIR, "competitions.csv")
_SEASONS_PATH = os.path.join(_DATA_DIR, "seasons.csv")

# ── In-memory singletons (populated on first access) ────────────────────────
_teams_df_mem: Optional[pd.DataFrame] = None
_people_df_mem: Optional[pd.DataFrame] = None
_competitions_df_mem: Optional[pd.DataFrame] = None
_seasons_df_mem: Optional[pd.DataFrame] = None
# Pre-indexed lookup: sofascore_player_id (str) → dict of row values.
_people_index: Optional[Dict[str, Dict[str, Any]]] = None
# Cached ClubElo → display-name mapping.
_clubelo_map_mem: Optional[Dict[str, str]] = None

# ── Internal helpers ─────────────────────────────────────────────────────────


def clear_memory_cache() -> None:
    """Reset all in-memory caches.  Useful in tests."""
    global _teams_df_mem, _people_df_mem, _competitions_df_mem, _seasons_df_mem
    global _people_index, _clubelo_map_mem
    _teams_df_mem = None
    _people_df_mem = None
    _competitions_df_mem = None
    _seasons_df_mem = None
    _people_index = None
    _clubelo_map_mem = None


def _load_csv(path: str) -> Optional[pd.DataFrame]:
    """Read a local CSV file and return a DataFrame.

    Uses ``dtype=str`` to prevent pandas from auto-coercing sparse ID
    columns to float64 (which would add spurious ``.0`` suffixes to
    values like ``key_sofascore`` or ``height_cm``).
    """
    if not os.path.isfile(path):
        _log.warning("REEP file not found: %s", path)
        return None
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    except Exception as exc:
        _log.warning("REEP CSV parse failed (%s): %s", path, exc)
        return None


# ── Public API ───────────────────────────────────────────────────────────────


def get_teams_df() -> Optional[pd.DataFrame]:
    """Return the REEP teams DataFrame (cached in-memory after first load)."""
    global _teams_df_mem
    if _teams_df_mem is not None:
        return _teams_df_mem
    df = _load_csv(_TEAMS_PATH)
    if df is not None:
        _teams_df_mem = df
    return df


def get_people_df() -> Optional[pd.DataFrame]:
    """Return the REEP people DataFrame (cached in-memory after first load)."""
    global _people_df_mem
    if _people_df_mem is not None:
        return _people_df_mem
    df = _load_csv(_PEOPLE_PATH)
    if df is not None:
        _people_df_mem = df
    return df


def get_competitions_df() -> Optional[pd.DataFrame]:
    """Return the REEP competitions DataFrame (cached after first load)."""
    global _competitions_df_mem
    if _competitions_df_mem is not None:
        return _competitions_df_mem
    df = _load_csv(_COMPETITIONS_PATH)
    if df is not None:
        _competitions_df_mem = df
    return df


def get_seasons_df() -> Optional[pd.DataFrame]:
    """Return the REEP seasons DataFrame (cached after first load)."""
    global _seasons_df_mem
    if _seasons_df_mem is not None:
        return _seasons_df_mem
    df = _load_csv(_SEASONS_PATH)
    if df is not None:
        _seasons_df_mem = df
    return df


def lookup_team(reep_id: str) -> Optional[Dict[str, Any]]:
    """Look up a team by its stable ``reep_id``.  Returns a dict or None."""
    df = get_teams_df()
    if df is None or "reep_id" not in df.columns:
        return None
    rows = df[df["reep_id"] == reep_id]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {k: (v if v else None) for k, v in row.to_dict().items()}


def lookup_person(reep_id: str) -> Optional[Dict[str, Any]]:
    """Look up a person by their stable ``reep_id``.  Returns a dict or None."""
    df = get_people_df()
    if df is None or "reep_id" not in df.columns:
        return None
    rows = df[df["reep_id"] == reep_id]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {k: (v if v else None) for k, v in row.to_dict().items()}


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

    # Keep only rows where both columns are present (non-empty string)
    mask = (df["key_clubelo"] != "") & (df["name"] != "")
    subset = df.loc[mask, ["key_clubelo", "name"]]
    result = dict(zip(subset["key_clubelo"], subset["name"]))
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

    sid_col = pd.to_numeric(df["key_sofascore"], errors="coerce")
    mask = sid_col == int(sofascore_id)
    rows = df.loc[mask]
    if rows.empty:
        return []

    aliases: List[str] = []
    for _, row in rows.iterrows():
        name = row.get("name", "")
        if name.strip():
            aliases.append(name.strip())
        clubelo = row.get("key_clubelo", "")
        if clubelo.strip():
            aliases.append(clubelo.strip())
    return list(dict.fromkeys(aliases))  # deduplicate, preserve order


def _build_people_index(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Build an O(1) lookup dict keyed by sofascore player ID (str).

    Skips rows where ``key_sofascore`` is missing or non-numeric.
    Uses vectorised pandas operations for speed (~430k rows in <1s).
    """
    subset = df[df["key_sofascore"] != ""].copy()
    # Coerce non-numeric IDs (e.g. 'francesco-conti') to NaN, then drop.
    subset["_sid"] = pd.to_numeric(subset["key_sofascore"], errors="coerce")
    subset = subset[subset["_sid"].notna()]
    subset["_sid"] = subset["_sid"].astype(int).astype(str)

    index: Dict[str, Dict[str, Any]] = {}
    for sid, nat, hcm, dob, pos, ws in zip(
        subset["_sid"],
        subset["nationality"],
        subset["height_cm"],
        subset["date_of_birth"],
        subset["position"],
        subset["key_whoscored"],
    ):
        index[sid] = {
            "nationality": nat if nat else None,
            "height_cm": _safe_int(hcm) if hcm else None,
            "date_of_birth": dob if dob else None,
            "position": pos if pos else None,
            "whoscored_id": _safe_int(ws) if ws else None,
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
