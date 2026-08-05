"""Shared prediction inputs for the Streamlit pages.

Every page that calls ``build_feature_dict`` has to assemble the same inputs.
When each page did that independently, they drifted: Transfer Impact passed
per-metric league means, while Hot or Not and the Shortlist Generator did not,
so ``build_feature_dict`` filled 26 features with its defaults (0.0 for the 13
``league_norm_*`` and 1.0 for the 13 ``league_mean_ratio_*``). Training fills
those with real values, so the same player-to-club move produced different
predictions depending on which page you opened — mean |z| 2.03 against the
shipped scaler, worst -8.96.

Keeping the assembly in one place is the point of this module.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import streamlit as st

from backend.data import sofascore_client
from backend.models.transfer_portal import _compute_league_means_from_stats

_log = logging.getLogger(__name__)


@st.cache_data(ttl=86400, show_spinner=False)
def get_league_players_cached(tournament_id: int, season_id) -> list:
    """Fetch league player stats once per day — shared across all sessions."""
    return sofascore_client.get_league_player_stats(
        tournament_id, season_id, limit=100
    )


def league_means_for_tournament(
    tournament_id: Optional[int], season_id: Any,
) -> Optional[Dict[str, float]]:
    """Return per-metric league averages, or None when they can't be computed.

    None rather than an empty dict, so ``build_feature_dict`` can tell "no
    league data" apart from "a league where every average is zero".
    """
    if not tournament_id or not season_id:
        return None
    try:
        players = get_league_players_cached(tournament_id, season_id)
    except Exception as exc:  # noqa: BLE001
        _log.warning("League player fetch failed for tournament %s: %s", tournament_id, exc)
        return None
    if not players:
        return None
    try:
        return _compute_league_means_from_stats(players)
    except Exception as exc:  # noqa: BLE001
        _log.warning("League mean computation failed: %s", exc)
        return None


def resolve_target_tournament(target_team_id: Optional[int]) -> Optional[int]:
    """Look up which tournament a team plays in, or None."""
    if not target_team_id:
        return None
    try:
        return sofascore_client.discover_tournament_for_team(target_team_id)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Tournament discovery failed for team %s: %s", target_team_id, exc)
        return None


def source_and_target_league_means(
    source_tournament_id: Optional[int],
    target_team_id: Optional[int],
    season_id: Any,
) -> tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    """Return ``(source_means, target_means)`` for a transfer.

    Either may be None. Callers should pass both through to
    ``build_feature_dict`` regardless — it handles None by falling back to its
    defaults, which is the documented behaviour, whereas silently omitting the
    arguments is what caused the page-to-page disagreement.
    """
    source = league_means_for_tournament(source_tournament_id, season_id)
    target_tournament_id = resolve_target_tournament(target_team_id)
    target = league_means_for_tournament(target_tournament_id, season_id)
    return source, target
