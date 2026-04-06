"""TransferScope Training Pipeline — single entry point for all training.

Collects historical transfer data from Sofascore, builds feature/label
pairs, trains the TransferPortalModel neural network and both adjustment
model types, and generates a backtesting validation report.

Usage:
    python backend/models/training_pipeline.py
    python backend/models/training_pipeline.py --seasons-back 3
    python backend/models/training_pipeline.py --skip-discovery --skip-training
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ── Force UTF-8 on stdout/stderr before any logging call ────────────────────
# Windows defaults to cp1252 which cannot encode Turkish (ş, ı, ğ), Polish
# (ć, ź), or arrow (→) characters, producing "--- Logging error ---" blocks.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Ensure repo root is on sys.path ─────────────────────────────────────────
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.data import cache, sofascore_client
from backend.data.sofascore_client import CORE_METRICS, normalize_position
from backend.features import power_rankings
from backend.features.adjustment_models import (
    PlayerAdjustmentModel,
    TeamAdjustmentModel,
)
from backend.features.rolling_windows import blend_weight
from backend.models.transfer_portal import (
    FEATURE_DIM,
    GROUP_FEATURE_SUBSETS,
    MODEL_GROUPS,
    TransferPortalModel,
    _feature_keys,
    build_feature_dict,
)
from backend.utils.league_registry import LEAGUES

_log = logging.getLogger(__name__)

_MODELS_DIR = os.path.join(_REPO_ROOT, "data", "models")

# Top 11 leagues for default discovery
DEFAULT_LEAGUE_CODES = [
    "ENG1",  # Premier League
    "ESP1",  # La Liga
    "GER1",  # Bundesliga
    "ITA1",  # Serie A
    "FRA1",  # Ligue 1
    "NED1",  # Eredivisie
    "POR1",  # Primeira Liga
    "BEL1",  # Belgian Pro League
    "ENG2",  # Championship
    "TUR1",  # Super Lig
    "SCO1",  # Scottish Premiership
]

from backend.utils.constants import MIN_MINUTES_THRESHOLD
API_CALL_DELAY_SECONDS = float(os.environ.get("SOFASCORE_API_CALL_DELAY", "0"))


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class TransferRecord:
    """A single player transfer with pre/post season references."""

    player_id: int
    player_name: str
    position: str
    from_club_id: int
    from_club_name: str
    from_league_id: int
    to_club_id: int
    to_club_name: str
    to_league_id: int
    transfer_date: str  # ISO date string or season boundary
    pre_transfer_season_id: int
    post_transfer_season_id: int
    pre_transfer_tournament_id: int
    post_transfer_tournament_id: int


@dataclass
class NonTransferRecord:
    """A player who stayed at the same club across consecutive seasons."""

    player_id: int
    player_name: str
    position: str
    club_id: int
    club_name: str
    league_id: int
    pre_season_id: int
    post_season_id: int
    pre_tournament_id: int
    post_tournament_id: int
    is_transfer: bool = False
    # Cached league-scan per90 and minutes (fallback when match logs / season
    # stats API calls return 404).  Populated by discover_non_transfers() from
    # the get_league_player_stats() data that was already fetched.
    cached_pre_per90: Optional[Dict[str, float]] = None
    cached_pre_minutes: int = 0


# ── Step 2: Transfer Discovery ───────────────────────────────────────────────


def discover_transfers(
    league_codes: Optional[List[str]] = None,
    seasons_back: int = 5,
) -> List[TransferRecord]:
    """Discover historical transfers across leagues with pre/post season data.

    Parameters
    ----------
    league_codes : list[str], optional
        League codes from the registry (default: top 11).
    seasons_back : int
        How many seasons of history to scan per league.

    Returns
    -------
    list[TransferRecord]
    """
    if league_codes is None:
        league_codes = DEFAULT_LEAGUE_CODES

    records: List[TransferRecord] = []
    seen_transfers: set = set()  # (player_id, from_club_id, to_club_id) dedup
    stats_by_league: Dict[str, int] = {}
    stats_by_season: Dict[str, int] = {}
    skipped_minutes = 0
    skipped_no_post = 0
    skipped_same_club = 0

    # Build team_id → tournament_id mapping from all league scans so that
    # _try_resolve_league can detect cross-league transfers without calling
    # the broken /team/{id}/unique-tournaments endpoint.
    team_id_to_league: Dict[int, int] = {}

    # Cache season lists per tournament_id to avoid repeated API calls
    # when resolving cross-league season IDs.
    _season_list_cache: Dict[int, List[Dict[str, Any]]] = {}

    for league_code in league_codes:
        info = LEAGUES.get(league_code)
        if info is None or info.sofascore_tournament_id is None:
            _log.warning("Unknown league code: %s", league_code)
            continue

        tid = info.sofascore_tournament_id
        _log.info("Scanning league: %s (tid=%d)", info.name, tid)

        # Get seasons
        seasons = sofascore_client.get_season_list(tid)
        if not seasons:
            _log.warning(
                "No seasons found for %s (tid=%d). "
                "This usually means the Sofascore API is unreachable "
                "(blocked by Cloudflare, rate-limited, or no internet). "
                "Check your network connection and try again with a higher --api-delay.",
                info.name, tid,
            )
            continue

        _season_list_cache[tid] = seasons

        # Take the last seasons_back seasons (newest first from API).
        # +1 so idx==0 is always a buffer season whose only purpose is to
        # serve as the post-transfer season for idx==1 (the newest
        # *usable* season).  This avoids silently dropping transfers from
        # the most recent season.
        target_seasons = seasons[:seasons_back + 1]

        # Build season lookup: season_id -> season_name
        season_ids = [s["id"] for s in target_seasons]

        # For each season, get all players
        for idx, season in enumerate(target_seasons):
            sid = season["id"]
            sname = season.get("name", str(sid))
            _log.info("  Season: %s (sid=%d)", sname, sid)

            players = sofascore_client.get_league_player_stats(tid, sid, limit=300)
            _log.info("  Found %d players in %s %s", len(players), info.name, sname)

            # Populate team_id → tournament_id from this scan
            for p in players:
                t_id = p.get("team_id")
                if t_id is not None:
                    team_id_to_league[t_id] = tid

            _first_player_logged = False
            for player in players:
                pid = player.get("id")
                if pid is None:
                    continue

                minutes = player.get("minutes_played", 0)
                if not _first_player_logged:
                    _log.info(
                        "First player in %s %s: id=%s name=%r minutes_played=%s",
                        info.name, sname, pid,
                        player.get("name", "?"), minutes,
                    )
                    _first_player_logged = True
                if minutes < MIN_MINUTES_THRESHOLD:
                    skipped_minutes += 1
                    continue

                # Get transfer history
                transfers = sofascore_client.get_player_transfer_history(pid)
                if not transfers:
                    continue

                player_position = player.get("position", "Unknown")

                # Check each transfer for matching seasons
                for transfer in transfers:
                    from_team = transfer.get("from_team") or {}
                    to_team = transfer.get("to_team") or {}

                    from_id = from_team.get("id")
                    to_id = to_team.get("id")
                    from_name = from_team.get("name", "")
                    to_name = to_team.get("name", "")

                    if not from_id or not to_id:
                        continue
                    if from_id == to_id:
                        skipped_same_club += 1
                        continue

                    # Dedup
                    dedup_key = (pid, from_id, to_id)
                    if dedup_key in seen_transfers:
                        continue

                    transfer_date = transfer.get("transfer_date", "")

                    # Try to identify pre and post transfer seasons
                    # The player was at from_club in the current season (pre)
                    # and should be at to_club in the next season (post)
                    # We need a next season in our data
                    if idx == 0:
                        # Newest season — no post-transfer season in our data
                        # unless we match by transfer date
                        continue

                    # Pre-transfer = current season, post-transfer = previous
                    # index (newer). seasons are newest-first.
                    pre_sid = sid
                    post_sid = target_seasons[idx - 1]["id"]

                    # Determine league IDs for from/to
                    # The from_league is the current tournament
                    # The to_league might be different — check transfer
                    from_league_id = tid
                    to_league_id = tid  # Same league by default

                    # Try to detect cross-league transfers
                    resolved_tid = _try_resolve_league(
                        to_id, to_name, team_id_to_league
                    )
                    if resolved_tid is not None and resolved_tid != tid:
                        to_league_id = resolved_tid

                    # Fetch pre-transfer stats directly — the league scan
                    # only contains current roster so players who have
                    # since left the club are missing.
                    pre_stats = sofascore_client.get_player_stats_for_season(
                        pid, from_league_id, pre_sid
                    )
                    pre_minutes = pre_stats.get("minutes_played", 0)
                    if pre_minutes < MIN_MINUTES_THRESHOLD:
                        skipped_minutes += 1
                        continue

                    # Resolve the correct post-transfer season ID.
                    # For cross-league transfers the destination league
                    # has different season IDs for the same calendar year.
                    # Match by season name (e.g. "24/25") to find the
                    # correct post_sid.
                    post_sname = target_seasons[idx - 1].get("name", "")
                    if to_league_id != from_league_id:
                        resolved_post_sid = _resolve_cross_league_post_sid(
                            to_league_id, post_sname, _season_list_cache
                        )
                        if resolved_post_sid is not None:
                            post_sid = resolved_post_sid
                        else:
                            _log.debug(
                                "Could not resolve post season for %s in tid=%d, "
                                "falling back to source league post_sid=%d",
                                post_sname, to_league_id, post_sid,
                            )

                    # Verify the player has enough minutes at the target club
                    # in the post-transfer season — use to_league_id so
                    # cross-league transfers fetch from the correct league.
                    post_stats = sofascore_client.get_player_stats_for_season(
                        pid, to_league_id, post_sid
                    )
                    post_minutes = post_stats.get("minutes_played", 0)
                    if post_minutes < MIN_MINUTES_THRESHOLD:
                        skipped_no_post += 1
                        continue

                    record = TransferRecord(
                        player_id=pid,
                        player_name=player.get("name", ""),
                        position=player_position,
                        from_club_id=from_id,
                        from_club_name=from_name,
                        from_league_id=from_league_id,
                        to_club_id=to_id,
                        to_club_name=to_name,
                        to_league_id=to_league_id,
                        transfer_date=transfer_date or sname,
                        pre_transfer_season_id=pre_sid,
                        post_transfer_season_id=post_sid,
                        pre_transfer_tournament_id=tid,
                        post_transfer_tournament_id=to_league_id,
                    )
                    records.append(record)
                    seen_transfers.add(dedup_key)

                    # Stats tracking
                    stats_by_league[league_code] = (
                        stats_by_league.get(league_code, 0) + 1
                    )
                    stats_by_season[sname] = stats_by_season.get(sname, 0) + 1

    # Print summary
    print(f"\n{'='*60}")
    print(f"Transfer Discovery Summary")
    print(f"{'='*60}")
    print(f"Total transfers found: {len(records)}")
    print(f"Skipped (same club):   {skipped_same_club}")
    print(f"Skipped (low minutes): {skipped_minutes}")
    print(f"Skipped (no post):     {skipped_no_post}")
    print(f"\nBy league:")
    for code, count in sorted(stats_by_league.items(), key=lambda x: -x[1]):
        print(f"  {code}: {count}")
    print(f"\nBy season:")
    for sname, count in sorted(stats_by_season.items()):
        print(f"  {sname}: {count}")
    print(f"{'='*60}\n")

    return records


def _try_resolve_league(
    team_id: int,
    team_name: str,
    team_id_to_league: Optional[Dict[int, int]] = None,
) -> Optional[int]:
    """Try to resolve a team's league tournament ID using the league registry.

    Uses a pre-built mapping of ``team_id → tournament_id`` gathered from
    league player-stats scans.  Falls back to ``None`` (same-league assumed)
    when the team is not found.  Does **not** call the Sofascore
    ``/team/{id}/unique-tournaments`` endpoint (which returns 404).
    """
    if team_id_to_league and team_id in team_id_to_league:
        return team_id_to_league[team_id]
    return None


def _resolve_cross_league_post_sid(
    to_league_id: int,
    post_season_name: str,
    season_list_cache: Dict[int, List[Dict[str, Any]]],
) -> Optional[int]:
    """Find the destination league's season ID that matches *post_season_name*.

    When a player transfers from league A to league B, the post-transfer
    season ID must come from league B's season list (they differ per
    league).  Match by season name (e.g. ``"24/25"`` or ``"2024/2025"``).

    Populates *season_list_cache* on first call per tournament to avoid
    repeated API requests.
    """
    if not post_season_name:
        return None

    if to_league_id not in season_list_cache:
        season_list_cache[to_league_id] = sofascore_client.get_season_list(
            to_league_id
        )

    for s in season_list_cache.get(to_league_id, []):
        if s.get("name") == post_season_name:
            return s["id"]
    return None


# ── Step 3: Feature and Label Extraction ─────────────────────────────────────

# Rolling-window target window size (paper: first 1000 minutes at new club)
_TARGET_WINDOW_MINUTES = 1000
_FEATURE_WINDOW_MINUTES = 1000


def _accumulate_first_n_minutes(
    match_logs: List[Dict[str, Any]],
    target_minutes: int = _TARGET_WINDOW_MINUTES,
) -> Optional[Dict[str, float]]:
    """Accumulate per-90 stats from the first N minutes of match logs.

    match_logs must be sorted ascending (oldest first).
    Returns a dict of metric -> weighted-average per-90, or None if
    insufficient minutes.
    """
    totals: Dict[str, float] = {m: 0.0 for m in CORE_METRICS}
    counts: Dict[str, int] = {m: 0 for m in CORE_METRICS}
    total_minutes = 0

    for match in match_logs:
        mins = match.get("minutes_played", 0)
        if mins is None or mins <= 0:
            continue
        if total_minutes >= target_minutes:
            break
        total_minutes += mins
        per90 = match.get("per90") or {}
        for m in CORE_METRICS:
            v = per90.get(m)
            if v is not None:
                try:
                    totals[m] += float(v) * mins
                    counts[m] += mins
                except (ValueError, TypeError):
                    pass

    if total_minutes < MIN_MINUTES_THRESHOLD:
        return None

    result: Dict[str, float] = {}
    for m in CORE_METRICS:
        if counts[m] > 0:
            result[m] = totals[m] / counts[m]
        else:
            result[m] = 0.0
    return result


def _accumulate_last_n_minutes(
    match_logs: List[Dict[str, Any]],
    target_minutes: int = _FEATURE_WINDOW_MINUTES,
) -> Optional[Dict[str, float]]:
    """Accumulate per-90 stats from the last N minutes of match logs.

    match_logs must be sorted ascending (oldest first).
    Walks backwards from the end.
    Returns a dict of metric -> weighted-average per-90, or None if
    insufficient minutes.
    """
    totals: Dict[str, float] = {m: 0.0 for m in CORE_METRICS}
    counts: Dict[str, int] = {m: 0 for m in CORE_METRICS}
    total_minutes = 0

    for match in reversed(match_logs):
        mins = match.get("minutes_played", 0)
        if mins is None or mins <= 0:
            continue
        if total_minutes >= target_minutes:
            break
        total_minutes += mins
        per90 = match.get("per90") or {}
        for m in CORE_METRICS:
            v = per90.get(m)
            if v is not None:
                try:
                    totals[m] += float(v) * mins
                    counts[m] += mins
                except (ValueError, TypeError):
                    pass

    if total_minutes < MIN_MINUTES_THRESHOLD:
        return None

    result: Dict[str, float] = {}
    for m in CORE_METRICS:
        if counts[m] > 0:
            result[m] = totals[m] / counts[m]
        else:
            result[m] = 0.0
    return result


def build_training_sample(
    record: TransferRecord,
) -> Optional[Dict[str, Any]]:
    """Build a single training sample (features + labels) from a transfer record.

    Returns
    -------
    dict with keys: features (46,), labels (13,), player_id, transfer_date,
    confidence, from_club, to_club — or None if data is insufficient.
    """
    # Block 1 — Player pre-transfer per-90 (13 values)
    # Strategy: try match logs first (rolling window of last 1000 minutes)
    pre_match_logs = sofascore_client.get_player_match_logs(
        record.player_id,
        record.pre_transfer_tournament_id,
        record.pre_transfer_season_id,
    )

    used_match_logs_for_features = False
    if pre_match_logs:
        rolling_per90 = _accumulate_last_n_minutes(pre_match_logs)
        if rolling_per90 is not None:
            pre_per90 = rolling_per90
            pre_minutes = sum(m.get("minutes_played", 0) for m in pre_match_logs)
            used_match_logs_for_features = True
        else:
            pre_per90 = None
    else:
        pre_per90 = None

    # Fallback: full season aggregate
    if pre_per90 is None:
        pre_stats = sofascore_client.get_player_stats_for_season(
            record.player_id,
            record.pre_transfer_tournament_id,
            record.pre_transfer_season_id,
        )
        if not pre_stats:
            return None
        pre_per90 = pre_stats.get("per90") or {}
        pre_minutes = pre_stats.get("minutes_played", 0)

    if pre_minutes < MIN_MINUTES_THRESHOLD:
        _log.warning(
            "Player %d pre-transfer minutes %d < %d, skipping",
            record.player_id, pre_minutes, MIN_MINUTES_THRESHOLD,
        )
        return None

    # Compute blend weight for confidence
    weight = blend_weight(pre_minutes)

    # Extract 13 core metrics in fixed order, 0.0 for missing
    player_metrics = []
    for m in CORE_METRICS:
        v = pre_per90.get(m)
        if v is None:
            _log.debug("Player %d missing metric %s, using 0.0", record.player_id, m)
            val = 0.0
        elif np.isnan(v):
            _log.warning("Player %d metric %s is NaN, using 0.0", record.player_id, m)
            val = 0.0
        else:
            val = float(v)
        player_metrics.append(val)

    # Block 2 — Team ability (2 values)
    # Parse transfer_date so historical Elo is used instead of today's.
    # Falling back to None (= today) only when the date is missing/unparseable.
    _ranking_date = None
    if record.transfer_date:
        try:
            _ranking_date = date.fromisoformat(record.transfer_date[:10])
        except (ValueError, TypeError):
            pass
    # No need to call compute_daily_rankings() explicitly here — the
    # coverage pre-check in build_full_dataset() already warmed the
    # in-process cache for this date.  get_team_ranking() calls below
    # hit that cache directly (O(1) diskcache skip).

    from_ranking = power_rankings.get_team_ranking(
        record.from_club_name,
        query_date=_ranking_date,
        tournament_id=record.from_league_id,
    )
    to_ranking = power_rankings.get_team_ranking(
        record.to_club_name,
        query_date=_ranking_date,
        tournament_id=record.to_league_id,
    )

    if from_ranking is None or to_ranking is None:
        _missing = []
        if from_ranking is None:
            _missing.append(f"source={record.from_club_name!r}")
        if to_ranking is None:
            _missing.append(f"dest={record.to_club_name!r}")
        _log.debug(
            "Player %d: no power ranking for %s — dropping",
            record.player_id, ", ".join(_missing),
        )
        return None

    team_ability_current = from_ranking.normalized_score
    raw_elo_current = from_ranking.raw_elo
    team_ability_target = to_ranking.normalized_score
    raw_elo_target = to_ranking.raw_elo

    # Block 3 — League ability (2 values)
    # Use Opta league ratings (446 leagues, 0-100) instead of ClubElo-derived
    # league means (~50 European leagues only).  Leakage risk accepted per
    # task spec: Opta league ratings are stable enough season-to-season.
    from_league_code = _find_league_code(record.from_league_id)
    to_league_code = _find_league_code(record.to_league_id)

    league_ability_current = power_rankings.get_league_opta_rating(
        from_league_code, record.from_club_name
    )
    league_ability_target = power_rankings.get_league_opta_rating(
        to_league_code, record.to_club_name
    )

    # Block 3b — REEP player metadata
    player_height_cm = 0.0
    player_age = 0.0
    try:
        from backend.data import reep_registry
        reep_data = reep_registry.enrich_player(record.player_id)
        if reep_data.get("reep_id"):
            _log.debug("REEP player %s → %s", record.player_id, reep_data["reep_id"])
        if reep_data.get("height_cm"):
            player_height_cm = float(reep_data["height_cm"])
        if reep_data.get("date_of_birth"):
            try:
                from datetime import datetime
                dob = datetime.strptime(str(reep_data["date_of_birth"])[:10], "%Y-%m-%d").date()
                player_age = (date.today() - dob).days / 365.25
            except Exception:
                pass
    except Exception:
        pass

    # Block 4+5 — Team-position per-90 (13 values each, placeholder zeros)
    # These are overwritten in build_full_dataset() with averages computed
    # from the training samples themselves — zero API calls needed here.
    position = normalize_position(record.position) or "Forward"
    team_pos_current = [0.0] * len(CORE_METRICS)
    team_pos_target = [0.0] * len(CORE_METRICS)

    # Block 6 — League means (source + target) for per-metric normalisation
    source_league_means = _compute_league_means(
        record.pre_transfer_tournament_id, record.pre_transfer_season_id
    )
    target_league_means = _compute_league_means(
        record.post_transfer_tournament_id, record.post_transfer_season_id
    )

    # Per-metric league-normalised features (Phase 5)
    league_norm_features = []
    league_mean_ratio_features = []
    for m in CORE_METRICS:
        player_val = player_metrics[CORE_METRICS.index(m)]
        src_mean = source_league_means.get(m, 0.0)
        tgt_mean = target_league_means.get(m, 0.0)
        # How many multiples of source league average the player is
        if src_mean > 1e-6:
            league_norm_features.append(min(player_val / src_mean, 20.0))
        else:
            league_norm_features.append(0.0)
        # Ratio of source-to-target league means
        if tgt_mean > 1e-6:
            league_mean_ratio_features.append(
                min(src_mean / tgt_mean, 5.0) if src_mean > 1e-6 else 1.0
            )
        else:
            league_mean_ratio_features.append(1.0)

    # Assemble the 79-feature vector
    ability_gap = team_ability_target - team_ability_current
    rel_current = team_ability_current - league_ability_current
    rel_target = team_ability_target - league_ability_target
    features = np.array(
        player_metrics
        + [team_ability_current, team_ability_target,
           league_ability_current, league_ability_target]
        + [raw_elo_current, raw_elo_target]
        + [player_height_cm, player_age]
        + team_pos_current
        + team_pos_target
        + [ability_gap, ability_gap ** 2,
           league_ability_target - league_ability_current]
        + [rel_current, rel_target, rel_target - rel_current]
        + league_norm_features
        + league_mean_ratio_features,
        dtype=np.float32,
    )

    if features.shape != (FEATURE_DIM,):
        raise ValueError(
            f"Feature vector shape {features.shape} != expected ({FEATURE_DIM},)"
        )

    # Replace any NaN with 0.0
    features = np.nan_to_num(features, nan=0.0)

    # ── Labels: post-transfer per-90 (first 1000 minutes — paper target) ──
    # Strategy A: match logs available → accumulate first 1000 minutes
    post_match_logs = sofascore_client.get_player_match_logs(
        record.player_id,
        record.post_transfer_tournament_id,
        record.post_transfer_season_id,
    )

    used_match_logs_for_labels = False
    minutes_accumulated = 0
    if post_match_logs:
        first_1000 = _accumulate_first_n_minutes(post_match_logs)
        if first_1000 is not None:
            post_per90 = first_1000
            cumulative = 0
            for m in post_match_logs:
                mins = m.get("minutes_played", 0)
                cumulative += mins
                if cumulative > _TARGET_WINDOW_MINUTES + 90:
                    break
                minutes_accumulated += mins
            used_match_logs_for_labels = True
        else:
            post_per90 = None
    else:
        post_per90 = None

    # Strategy B: fallback to full-season stats
    if post_per90 is None:
        post_stats = sofascore_client.get_player_stats_for_season(
            record.player_id,
            record.post_transfer_tournament_id,
            record.post_transfer_season_id,
        )
        if not post_stats:
            return None
        post_per90 = post_stats.get("per90") or {}
        post_minutes = post_stats.get("minutes_played", 0)
        minutes_accumulated = post_minutes

        # Apply minutes filter for fallback
        if post_minutes < 700:
            _log.warning(
                "Player %d post-transfer minutes %d < 700 (fallback), skipping",
                record.player_id, post_minutes,
            )
            return None
        if post_minutes > 1300:
            _log.warning(
                "Player %d post-transfer minutes %d > 1300, using full-season as-is",
                record.player_id, post_minutes,
            )

    labels = []
    for m in CORE_METRICS:
        v = post_per90.get(m)
        if v is None:
            val = 0.0
        elif np.isnan(v):
            _log.warning(
                "Player %d post-transfer metric %s is NaN, using 0.0",
                record.player_id, m,
            )
            val = 0.0
        else:
            val = float(v)
        labels.append(val)

    labels_arr = np.array(labels, dtype=np.float32)
    if labels_arr.shape != (len(CORE_METRICS),):
        raise ValueError(
            f"Labels shape {labels_arr.shape} != expected ({len(CORE_METRICS)},)"
        )
    labels_arr = np.nan_to_num(labels_arr, nan=0.0)

    # source_league_means and target_league_means already computed above
    # (Block 6) for per-metric normalisation features.

    return {
        "features": features,
        "labels": labels_arr,
        "player_id": record.player_id,
        "player_name": record.player_name,
        "transfer_date": record.transfer_date,
        "confidence": float(weight),
        "from_club": record.from_club_name,
        "to_club": record.to_club_name,
        "position": record.position,
        "is_transfer": True,
        # Extra metadata for adjustment model training
        "team_ability_current": float(team_ability_current),
        "team_ability_target": float(team_ability_target),
        "league_ability_current": float(league_ability_current),
        "league_ability_target": float(league_ability_target),
        "pre_per90": {m: float(player_metrics[i]) for i, m in enumerate(CORE_METRICS)},
        "from_pos_avg": {m: float(team_pos_current[i]) for i, m in enumerate(CORE_METRICS)},
        "to_pos_avg": {m: float(team_pos_target[i]) for i, m in enumerate(CORE_METRICS)},
        "league_means": target_league_means,
        "source_league_means": source_league_means,
        "used_match_logs_features": used_match_logs_for_features,
        "used_match_logs_labels": used_match_logs_for_labels,
        "minutes_accumulated": minutes_accumulated,
        "source_club_id": record.from_club_id,
        "target_club_id": record.to_club_id,
    }


def _find_league_code(tournament_id: int) -> Optional[str]:
    """Find league code by Sofascore tournament ID."""
    for code, info in LEAGUES.items():
        if info.sofascore_tournament_id == tournament_id:
            return code
    return None


def _rate_limit_delay() -> None:
    """Sleep to respect API rate limits during data collection."""
    import time
    time.sleep(API_CALL_DELAY_SECONDS)


def _compute_league_means(
    tournament_id: int,
    season_id: int,
) -> Dict[str, float]:
    """Compute league mean per-90 for players with >= 450 minutes.

    Falls back to empty dict on failure.
    """
    try:
        players = sofascore_client.get_league_player_stats(tournament_id, season_id, limit=300)
        if not players:
            return {}

        metric_sums: Dict[str, float] = {m: 0.0 for m in CORE_METRICS}
        metric_counts: Dict[str, int] = {m: 0 for m in CORE_METRICS}

        for p in players:
            if p.get("minutes_played", 0) < MIN_MINUTES_THRESHOLD:
                continue
            per90 = p.get("per90") or {}
            for m in CORE_METRICS:
                v = per90.get(m)
                if v is not None:
                    try:
                        metric_sums[m] += float(v)
                        metric_counts[m] += 1
                    except (ValueError, TypeError):
                        pass

        return {
            m: metric_sums[m] / metric_counts[m] if metric_counts[m] > 0 else 0.0
            for m in CORE_METRICS
        }
    except Exception as exc:
        _log.warning("Failed to compute league means for tid=%d sid=%d: %s",
                     tournament_id, season_id, exc)
        return {}


_NT_MIN_MINUTES = 300  # Minimum minutes for non-transfer candidates (~3.3 full matches)


def discover_non_transfers(
    league_codes: Optional[List[str]] = None,
    seasons_back: int = 5,
    exclude_player_seasons: Optional[set] = None,
    min_minutes: int = _NT_MIN_MINUTES,
    target_count: Optional[int] = None,
) -> List[NonTransferRecord]:
    """Discover players who stayed at the same club across consecutive seasons.

    For each league & consecutive season pair, finds players with >= min_minutes
    in both seasons and no intervening transfer to a different club.

    Parameters
    ----------
    exclude_player_seasons : set[tuple[int, int]], optional
        ``(player_id, season_id)`` pairs to skip.  A player is only
        excluded for the specific season in which they appear in the
        transfer dataset, so the same player can still contribute
        non-transfer samples for other season pairs.
    min_minutes : int
        Minimum minutes played in each season. Default 300 (~3.3 full matches).
    target_count : int, optional
        Target number of non-transfer samples. If the initial pass yields
        fewer candidates, a second pass with ``min_minutes // 2`` is attempted.
    """
    if league_codes is None:
        league_codes = DEFAULT_LEAGUE_CODES

    if exclude_player_seasons is None:
        exclude_player_seasons = set()

    records: List[NonTransferRecord] = []
    seen: set = set()

    _raw_candidates = 0
    _skipped_excluded = 0
    _skipped_minutes = 0
    _skipped_moved = 0
    _skipped_post_minutes = 0

    for league_code in league_codes:
        info = LEAGUES.get(league_code)
        if info is None or info.sofascore_tournament_id is None:
            continue

        tid = info.sofascore_tournament_id
        seasons = sofascore_client.get_season_list(tid)
        if not seasons:
            continue

        target_seasons = seasons[:seasons_back]

        for idx in range(len(target_seasons) - 1):
            pre_season = target_seasons[idx + 1]  # older
            post_season = target_seasons[idx]       # newer
            pre_sid = pre_season["id"]
            post_sid = post_season["id"]

            players = sofascore_client.get_league_player_stats(tid, pre_sid, limit=300)
            _log.info(
                "Non-transfer scan: %s season %s -> %d players before filtering",
                league_code, pre_season.get("name", "?"), len(players),
            )

            for player in players:
                pid = player.get("id")
                if pid is None:
                    continue

                _raw_candidates += 1

                if (pid, pre_sid) in exclude_player_seasons:
                    _skipped_excluded += 1
                    continue
                if player.get("minutes_played", 0) < min_minutes:
                    _skipped_minutes += 1
                    continue

                dedup_key = (pid, pre_sid, post_sid)
                if dedup_key in seen:
                    continue

                # Check transfer history — did the player move between these seasons?
                transfers = sofascore_client.get_player_transfer_history(pid)
                moved = False
                pre_season_name = pre_season.get("name", "")
                post_season_name = post_season.get("name", "")
                if transfers:
                    for t in transfers:
                        from_team = (t.get("from_team") or {}).get("id")
                        to_team = (t.get("to_team") or {}).get("id")
                        if not from_team or not to_team or from_team == to_team:
                            continue
                        # Check if transfer date overlaps with the season window
                        t_date = t.get("transfer_date") or ""
                        if t_date:
                            t_year = t_date[:4]
                            pre_years = pre_season_name.replace("/", " ").split()
                            post_years = post_season_name.replace("/", " ").split()
                            relevant_years = set()
                            for y in pre_years + post_years:
                                if len(y) == 2:
                                    relevant_years.add("20" + y)
                                elif len(y) == 4:
                                    relevant_years.add(y)
                            if t_year in relevant_years:
                                moved = True
                                break
                        # else: no date — skip this entry, verify via team_id below

                if moved:
                    _skipped_moved += 1
                    continue

                # Verify minutes in post season AND same team
                post_stats = sofascore_client.get_player_stats_for_season(
                    pid, tid, post_sid
                )
                if post_stats.get("minutes_played", 0) < min_minutes:
                    _skipped_post_minutes += 1
                    continue

                # Cross-check: if post-season team_id differs from pre-season,
                # the player moved despite no dated transfer record.
                pre_team_id = player.get("team_id") or player.get("teamId") or 0
                post_team_id = post_stats.get("team_id") or post_stats.get("teamId") or 0
                if pre_team_id and post_team_id and pre_team_id != post_team_id:
                    _skipped_moved += 1
                    continue

                # get_league_player_stats returns "team"/"team_id"; fallbacks for safety
                club_id = player.get("team_id") or player.get("teamId") or 0
                club_name = player.get("team") or player.get("team_name") or ""

                records.append(NonTransferRecord(
                    player_id=pid,
                    player_name=player.get("name", ""),
                    position=player.get("position", "Unknown"),
                    club_id=club_id,
                    club_name=club_name,
                    league_id=tid,
                    pre_season_id=pre_sid,
                    post_season_id=post_sid,
                    pre_tournament_id=tid,
                    post_tournament_id=tid,
                    cached_pre_per90=player.get("per90"),
                    cached_pre_minutes=player.get("minutes_played", 0),
                ))
                seen.add(dedup_key)

    _log.info(
        "Non-transfer discovery: %d raw candidates, %d excluded (transfer players), "
        "%d skipped (low minutes), %d skipped (moved), %d skipped (post-season minutes), "
        "%d accepted",
        _raw_candidates, _skipped_excluded, _skipped_minutes,
        _skipped_moved, _skipped_post_minutes, len(records),
    )

    # If we haven't hit the target, retry with relaxed minutes threshold
    if target_count is not None and len(records) < target_count and min_minutes > 250:
        relaxed_min = max(250, min_minutes // 2)
        _log.info(
            "Non-transfer retry ATTEMPTED: count %d < target %d; retrying with min_minutes=%d",
            len(records), target_count, relaxed_min,
        )
        extra = discover_non_transfers(
            league_codes=league_codes,
            seasons_back=seasons_back,
            exclude_player_seasons=exclude_player_seasons | {
                (r.player_id, r.pre_season_id) for r in records
            },
            min_minutes=relaxed_min,
            target_count=None,  # no recursive retry
        )
        records.extend(extra)
        _log.info("After relaxed retry: %d total non-transfer samples", len(records))
    elif target_count is not None and len(records) < target_count:
        _log.info(
            "Non-transfer retry NOT attempted: count %d < target %d but "
            "min_minutes=%d is already at or below 250",
            len(records), target_count, min_minutes,
        )

    _log.info("Discovered %d non-transfer samples", len(records))
    return records


def build_non_transfer_sample(
    record: NonTransferRecord,
    min_minutes: int = MIN_MINUTES_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    """Build a training sample for a player who stayed at the same club.

    Same structure as build_training_sample() but from_club == to_club,
    change_relative_ability = 0, and team_pos_current == team_pos_target.
    """
    # Pre-transfer features (match logs first, then season aggregate)
    pre_match_logs = sofascore_client.get_player_match_logs(
        record.player_id,
        record.pre_tournament_id,
        record.pre_season_id,
    )

    used_match_logs_for_features = False
    if pre_match_logs:
        rolling_per90 = _accumulate_last_n_minutes(pre_match_logs)
        if rolling_per90 is not None:
            pre_per90 = rolling_per90
            pre_minutes = sum(m.get("minutes_played", 0) for m in pre_match_logs)
            used_match_logs_for_features = True
        else:
            pre_per90 = None
    else:
        pre_per90 = None

    if pre_per90 is None:
        pre_stats = sofascore_client.get_player_stats_for_season(
            record.player_id,
            record.pre_tournament_id,
            record.pre_season_id,
        )
        if pre_stats:
            pre_per90 = pre_stats.get("per90") or {}
            pre_minutes = pre_stats.get("minutes_played", 0)
            if pre_minutes < min_minutes:
                pre_per90 = None
        else:
            pre_per90 = None

    # Fallback: use cached per90 from league-wide scan (already fetched during
    # discover_non_transfers → get_league_player_stats) when both match logs
    # and season stats API calls fail or return insufficient data.
    if pre_per90 is None or not pre_per90:
        if record.cached_pre_per90:
            _log.info(
                "Non-transfer player %d (%s): using cached league-scan per90 "
                "(%d min) as fallback for pre-features",
                record.player_id, record.player_name, record.cached_pre_minutes,
            )
            pre_per90 = record.cached_pre_per90
            pre_minutes = record.cached_pre_minutes
        else:
            _log.debug(
                "Non-transfer player %d (%s): no cached data available, skipping",
                record.player_id, record.player_name,
            )
            return None

    if pre_minutes < min_minutes:
        return None

    weight = blend_weight(pre_minutes)

    player_metrics = []
    for m in CORE_METRICS:
        v = pre_per90.get(m)
        if v is None:
            val = 0.0
        elif np.isnan(v):
            val = 0.0
        else:
            val = float(v)
        player_metrics.append(val)

    # Power rankings — same club so current == target
    try:
        team_rankings, league_snapshots = power_rankings.compute_daily_rankings()
    except Exception:
        team_rankings, league_snapshots = {}, {}

    ranking = power_rankings.get_team_ranking(
        record.club_name,
        tournament_id=record.league_id,
    )
    if ranking is None:
        _log.debug(
            "Non-transfer player %d (%s): no power ranking coverage — dropping",
            record.player_id, record.club_name,
        )
        return None
    team_ability = ranking.normalized_score
    raw_elo = ranking.raw_elo

    league_code = _find_league_code(record.league_id)
    league_snap = league_snapshots.get(league_code) if league_code else None
    league_ability = league_snap.mean_normalized if league_snap else 50.0

    # REEP player metadata
    nt_height_cm = 0.0
    nt_age = 0.0
    try:
        from backend.data import reep_registry
        reep_data = reep_registry.enrich_player(record.player_id)
        if reep_data.get("reep_id"):
            _log.debug("REEP player %s → %s", record.player_id, reep_data["reep_id"])
        if reep_data.get("height_cm"):
            nt_height_cm = float(reep_data["height_cm"])
        if reep_data.get("date_of_birth"):
            try:
                from datetime import datetime
                dob = datetime.strptime(str(reep_data["date_of_birth"])[:10], "%Y-%m-%d").date()
                nt_age = (date.today() - dob).days / 365.25
            except Exception:
                pass
    except Exception:
        pass

    # Team-position per-90 — placeholder zeros, overwritten in
    # build_full_dataset() with averages from the training data itself.
    position = normalize_position(record.position) or "Forward"
    team_pos = [0.0] * len(CORE_METRICS)

    # League means for per-metric normalisation (same league for non-transfers)
    league_means = _compute_league_means(record.league_id, record.post_season_id)

    # Per-metric league-normalised features (Phase 5)
    league_norm_features = []
    league_mean_ratio_features = []
    for m in CORE_METRICS:
        player_val = player_metrics[CORE_METRICS.index(m)]
        lm = league_means.get(m, 0.0)
        # How many multiples of league average the player is
        if lm > 1e-6:
            league_norm_features.append(min(player_val / lm, 20.0))
        else:
            league_norm_features.append(0.0)
        # Same league → ratio = 1.0
        league_mean_ratio_features.append(1.0)

    # Same team → current == target for all ability and position features
    # Interaction features are all zero (no gap when staying at same club)
    nt_rel = team_ability - league_ability
    features = np.array(
        player_metrics
        + [team_ability, team_ability, league_ability, league_ability]
        + [raw_elo, raw_elo]
        + [nt_height_cm, nt_age]
        + team_pos
        + team_pos  # team_pos_target == team_pos_current
        + [0.0, 0.0, 0.0]  # interaction: gap=0, gap²=0, league_gap=0
        + [nt_rel, nt_rel, 0.0]  # relative: current==target, gap=0
        + league_norm_features
        + league_mean_ratio_features,
        dtype=np.float32,
    )

    if features.shape != (FEATURE_DIM,):
        raise ValueError(
            f"Feature vector shape {features.shape} != expected ({FEATURE_DIM},)"
        )
    features = np.nan_to_num(features, nan=0.0)

    # Labels: post-season per-90 (first 1000 minutes)
    post_match_logs = sofascore_client.get_player_match_logs(
        record.player_id,
        record.post_tournament_id,
        record.post_season_id,
    )

    used_match_logs_for_labels = False
    if post_match_logs:
        first_1000 = _accumulate_first_n_minutes(post_match_logs)
        if first_1000 is not None:
            post_per90 = first_1000
            used_match_logs_for_labels = True
        else:
            post_per90 = None
    else:
        post_per90 = None

    if post_per90 is None:
        post_stats = sofascore_client.get_player_stats_for_season(
            record.player_id,
            record.post_tournament_id,
            record.post_season_id,
        )
        if post_stats:
            post_per90 = post_stats.get("per90") or {}
            post_minutes = post_stats.get("minutes_played", 0)
            if post_minutes < min_minutes:
                post_per90 = None
        else:
            post_per90 = None

    # Fallback for post-season labels: use pre-season per90 (same club →
    # performance should be similar across consecutive seasons).  This is
    # an approximation, but better than dropping the candidate entirely.
    if post_per90 is None or not post_per90:
        if pre_per90:
            _log.info(
                "Non-transfer player %d (%s): using pre-season per90 as "
                "fallback for post-season labels",
                record.player_id, record.player_name,
            )
            post_per90 = pre_per90
        else:
            return None

    labels = []
    for m in CORE_METRICS:
        v = post_per90.get(m)
        if v is None:
            val = 0.0
        elif np.isnan(v):
            val = 0.0
        else:
            val = float(v)
        labels.append(val)

    labels_arr = np.array(labels, dtype=np.float32)
    labels_arr = np.nan_to_num(labels_arr, nan=0.0)

    # league_means already computed above (Block: League means) for features.

    return {
        "features": features,
        "labels": labels_arr,
        "player_id": record.player_id,
        "player_name": record.player_name,
        "transfer_date": "",  # no transfer date for non-transfers
        "confidence": float(weight),
        "from_club": record.club_name,
        "to_club": record.club_name,
        "position": record.position,
        "is_transfer": False,
        "team_ability_current": float(team_ability),
        "team_ability_target": float(team_ability),
        "league_ability_current": float(league_ability),
        "league_ability_target": float(league_ability),
        "pre_per90": {m: float(player_metrics[i]) for i, m in enumerate(CORE_METRICS)},
        "from_pos_avg": {m: float(team_pos[i]) for i, m in enumerate(CORE_METRICS)},
        "to_pos_avg": {m: float(team_pos[i]) for i, m in enumerate(CORE_METRICS)},
        "league_means": league_means,
        "source_league_means": league_means,  # same league for non-transfers
        "used_match_logs_features": used_match_logs_for_features,
        "used_match_logs_labels": used_match_logs_for_labels,
        "source_club_id": record.club_id,
        "target_club_id": record.club_id,
    }


def compute_team_position_averages(
    samples: list[dict],
) -> dict[tuple[int, str], dict[str, float]]:
    """Compute per-90 averages for each (club_id, position) pair
    from a list of already-built training samples.

    Parameters
    ----------
    samples : list[dict]
        Each dict must contain:
          - 'source_club_id'   (int)
          - 'position'         (str, e.g. 'Midfielder')
          - 'pre_per90'        (dict of metric -> float)

    Returns
    -------
    dict keyed by (club_id, position) -> dict of metric -> mean float
    """
    from collections import defaultdict

    buckets: dict = defaultdict(list)
    for s in samples:
        club_id = s.get("source_club_id")
        position = s.get("position")
        per90 = s.get("pre_per90", {})
        if club_id and position and per90:
            buckets[(club_id, position)].append(per90)

    averages: dict = {}
    for key, records in buckets.items():
        if not records:
            continue
        avg = {}
        all_metrics = set().union(*[r.keys() for r in records])
        for metric in all_metrics:
            vals = [r[metric] for r in records if metric in r]
            avg[metric] = float(np.mean(vals)) if vals else 0.0
        averages[key] = avg
    return averages


def inject_team_pos_averages(
    X: np.ndarray,
    metadata: List[Dict[str, Any]],
    train_metadata: List[Dict[str, Any]],
) -> np.ndarray:
    """Inject team-position per-90 averages into feature matrices.

    Computes averages from *train_metadata* only (not val/test) to
    prevent data leakage, then writes them into the team_pos_current
    and team_pos_target slots of *X* for all splits.

    Parameters
    ----------
    X : ndarray, shape (N, FEATURE_DIM)
        Feature matrix to update in-place.
    metadata : list[dict]
        Metadata for the rows in *X* (needs source_club_id, target_club_id,
        position).
    train_metadata : list[dict]
        Metadata for training rows only — used to compute averages.

    Returns
    -------
    ndarray — updated *X* (modified in-place and also returned).

    Side Effects
    ------------
    *metadata* dicts are also updated in-place with 'from_pos_avg' and
    'to_pos_avg' keys for downstream consumers (e.g. adjustment models).
    """
    team_pos_lookup = compute_team_position_averages(train_metadata)

    # Layout: [0:13] player, [13:17] abilities, [17:19] raw_elo,
    #          [19:21] reep, [21:34] pos_current,
    #          [34:47] pos_target, [47:50] interactions.
    _POS_CURRENT_OFFSET = len(CORE_METRICS) + 4 + 2 + 2   # 21
    _POS_TARGET_OFFSET = _POS_CURRENT_OFFSET + len(CORE_METRICS)  # 34
    assert _POS_CURRENT_OFFSET == 21, f"Expected 21, got {_POS_CURRENT_OFFSET}"
    assert _POS_TARGET_OFFSET == 34, f"Expected 34, got {_POS_TARGET_OFFSET}"
    assert _POS_TARGET_OFFSET + len(CORE_METRICS) <= FEATURE_DIM, (
        f"team_pos slots exceed FEATURE_DIM ({FEATURE_DIM})"
    )

    for i, m_dict in enumerate(metadata):
        src_key = (m_dict.get("source_club_id"), m_dict.get("position"))
        tgt_key = (m_dict.get("target_club_id"), m_dict.get("position"))

        src_avg = team_pos_lookup.get(src_key, {})
        tgt_avg = team_pos_lookup.get(tgt_key, {})

        for j, metric in enumerate(CORE_METRICS):
            X[i, _POS_CURRENT_OFFSET + j] = src_avg.get(metric, 0.0)
            X[i, _POS_TARGET_OFFSET + j] = tgt_avg.get(metric, 0.0)

        # Also update metadata dicts for downstream consumers
        m_dict["from_pos_avg"] = {metric: src_avg.get(metric, 0.0) for metric in CORE_METRICS}
        m_dict["to_pos_avg"] = {metric: tgt_avg.get(metric, 0.0) for metric in CORE_METRICS}

    return X


def build_full_dataset(
    transfer_records: List[TransferRecord],
    non_transfer_records: Optional[List[NonTransferRecord]] = None,
    nt_min_minutes: int = MIN_MINUTES_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Build the full training dataset from transfer records.

    Returns
    -------
    (X, y, metadata)
        X: shape (N, 50) features
        y: shape (N, 13) labels
        metadata: list of dicts with player_id, transfer_date, from_club, to_club, etc.
    """
    samples = []
    drop_reasons: Dict[str, int] = {}
    opta_resolved_source = 0
    opta_resolved_dest = 0
    total = len(transfer_records)

    _OPTA_TYPES = frozenset({"opta", "opta_league_avg"})

    _log.info(f"TIMER: dataset build loop start — {datetime.now().strftime('%H:%M:%S')}")
    for i, record in enumerate(transfer_records):
        if (i + 1) % 50 == 0 or i == 0:
            _log.info("Building sample %d / %d ...", i + 1, total)
        if (i + 1) % 500 == 0:
            from backend.data import sofascore_client as _sc
            from backend.data import clubelo_client as _cc
            _log.info(
                "SAMPLE %d — Sofascore HTTP calls: %d | ClubElo cache hits: %d / misses: %d",
                i + 1,
                _sc.http_call_count,
                _cc.cache_hits,
                _cc.cache_misses,
            )

        # ── Coverage pre-check ───────────────────────────────────────────────
        # Resolve the transfer date for date-accurate ranking lookups.
        _pre_date = None
        if record.transfer_date:
            try:
                _pre_date = date.fromisoformat(record.transfer_date[:10])
            except (ValueError, TypeError):
                pass

        from_r = power_rankings.get_team_ranking(
            record.from_club_name, _pre_date, record.from_league_id
        )
        to_r = power_rankings.get_team_ranking(
            record.to_club_name, _pre_date, record.to_league_id
        )

        if from_r is None or to_r is None:
            drop_reasons["no_coverage_at_all"] = (
                drop_reasons.get("no_coverage_at_all", 0) + 1
            )
            continue

        if from_r.match_type in _OPTA_TYPES:
            opta_resolved_source += 1
        if to_r.match_type in _OPTA_TYPES:
            opta_resolved_dest += 1

        try:
            sample = build_training_sample(record)
        except Exception as exc:
            _log.warning(
                "Error building sample for player %d: %s", record.player_id, exc
            )
            drop_reasons["error"] = drop_reasons.get("error", 0) + 1
            continue

        if sample is None:
            drop_reasons["insufficient_data"] = (
                drop_reasons.get("insufficient_data", 0) + 1
            )
            continue

        samples.append(sample)

    # Process non-transfer records
    n_transfer_samples = len(samples)
    if non_transfer_records:
        nt_total = len(non_transfer_records)
        for i, nt_record in enumerate(non_transfer_records):
            if (i + 1) % 50 == 0 or i == 0:
                _log.info("Building non-transfer sample %d / %d ...", i + 1, nt_total)

            try:
                nt_sample = build_non_transfer_sample(nt_record, min_minutes=nt_min_minutes)
            except Exception as exc:
                _log.warning(
                    "Error building non-transfer sample for player %d: %s",
                    nt_record.player_id, exc,
                )
                drop_reasons["nt_error"] = drop_reasons.get("nt_error", 0) + 1
                continue

            if nt_sample is None:
                _log.info(
                    "Non-transfer player %d (%s) dropped: insufficient data "
                    "(cached_pre_minutes=%d, has_cached_per90=%s)",
                    nt_record.player_id, nt_record.player_name,
                    nt_record.cached_pre_minutes,
                    bool(nt_record.cached_pre_per90),
                )
                drop_reasons["nt_insufficient_data"] = (
                    drop_reasons.get("nt_insufficient_data", 0) + 1
                )
                continue

            samples.append(nt_sample)

    if not samples:
        _log.error("No valid training samples built!")
        return np.empty((0, FEATURE_DIM)), np.empty((0, len(CORE_METRICS))), []

    # ── Team-position averages — leave as zeros here ─────────────────────
    # Previously computed from ALL samples (including future test data),
    # causing mild data leakage.  Now deferred to inject_team_pos_averages()
    # which is called AFTER the temporal split using training data only.
    # The placeholder zeros are overwritten before training begins.

    X = np.stack([s["features"] for s in samples])
    y = np.stack([s["labels"] for s in samples])
    metadata = [
        {k: v for k, v in s.items() if k not in ("features", "labels")}
        for s in samples
    ]

    # Verify no NaNs
    nan_count_X = int(np.isnan(X).sum())
    nan_count_y = int(np.isnan(y).sum())
    if nan_count_X > 0 or nan_count_y > 0:
        _log.warning("NaN values found -- X: %d, y: %d. Replacing with 0.", nan_count_X, nan_count_y)
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

    print(f"\nDataset Summary:")
    print(f"  Valid samples: {len(samples)}")
    print(f"    Transfer samples: {n_transfer_samples}")
    print(f"    Non-transfer samples: {len(samples) - n_transfer_samples}")
    print(f"  Transfer coverage (of {total} records):")
    print(f"    Resolved via ClubElo+Opta fallback — source: {opta_resolved_source}, dest: {opta_resolved_dest}")
    print(f"  Dropped: {sum(drop_reasons.values())}")
    for reason, count in sorted(drop_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    _log.info(f"TIMER: dataset build loop end (after Dataset Summary) — {datetime.now().strftime('%H:%M:%S')}")

    return X, y, metadata


# ── Step 4: Train/Validate/Test Split ────────────────────────────────────────


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    metadata: List[Dict[str, Any]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]],
]:
    """Split dataset temporally (NOT random) — most recent transfers go to test.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test,
    meta_train, meta_val, meta_test
    """
    n = len(metadata)
    if n == 0:
        empty_X = np.empty((0, X.shape[1] if X.ndim > 1 else FEATURE_DIM))
        empty_y = np.empty((0, y.shape[1] if y.ndim > 1 else len(CORE_METRICS)))
        return empty_X, empty_y, empty_X, empty_y, empty_X, empty_y, [], [], []

    # Sort by transfer_date ascending
    indices = list(range(n))
    indices.sort(key=lambda i: metadata[i].get("transfer_date", ""))

    sorted_X = X[indices]
    sorted_y = y[indices]
    sorted_meta = [metadata[i] for i in indices]

    # Temporal split
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_test - n_val

    if n_train < 1:
        # Not enough data — put everything in training
        _log.warning("Not enough data for 3-way split, using all for training")
        return sorted_X, sorted_y, sorted_X[:0], sorted_y[:0], sorted_X[:0], sorted_y[:0], sorted_meta, [], []

    X_train = sorted_X[:n_train]
    y_train = sorted_y[:n_train]
    meta_train = sorted_meta[:n_train]

    X_val = sorted_X[n_train:n_train + n_val]
    y_val = sorted_y[n_train:n_train + n_val]
    meta_val = sorted_meta[n_train:n_train + n_val]

    X_test = sorted_X[n_train + n_val:]
    y_test = sorted_y[n_train + n_val:]
    meta_test = sorted_meta[n_train + n_val:]

    # Player-level deduplication: remove from train+val any player_id in test set
    test_player_ids = {m.get("player_id") for m in meta_test if m.get("player_id") is not None}
    if test_player_ids:
        def _filter_by_player_ids(
            X_split: np.ndarray, y_split: np.ndarray,
            meta_split: List[Dict[str, Any]], excluded_ids: set, label: str,
        ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
            """Remove samples whose player_id is in *excluded_ids*."""
            mask = [m.get("player_id") not in excluded_ids for m in meta_split]
            removed = sum(1 for keep in mask if not keep)
            if removed == 0:
                return X_split, y_split, meta_split
            _log.info("Removed %d %s samples (player overlap with test)", removed, label)
            indices = [i for i, keep in enumerate(mask) if keep]
            if indices:
                return X_split[indices], y_split[indices], [meta_split[i] for i in indices]
            return (
                np.empty((0, X.shape[1] if X.ndim > 1 else FEATURE_DIM)),
                np.empty((0, y.shape[1] if y.ndim > 1 else len(CORE_METRICS))),
                [],
            )

        X_train, y_train, meta_train = _filter_by_player_ids(
            X_train, y_train, meta_train, test_player_ids, "training",
        )
        X_val, y_val, meta_val = _filter_by_player_ids(
            X_val, y_val, meta_val, test_player_ids, "validation",
        )

    # Print split summary
    def _date_range(meta: List[Dict]) -> str:
        if not meta:
            return "N/A"
        dates = [m["transfer_date"] for m in meta if m.get("transfer_date")]
        if not dates:
            return "N/A"
        return f"{min(dates)} to {max(dates)}"

    print(f"\nTemporal Split Summary:")
    print(f"  Training:   {len(meta_train)} samples, {_date_range(meta_train)}")
    print(f"  Validation: {len(meta_val)} samples, {_date_range(meta_val)}")
    print(f"  Test:       {len(meta_test)} samples, {_date_range(meta_test)}")

    return (
        X_train, y_train, X_val, y_val, X_test, y_test,
        meta_train, meta_val, meta_test,
    )


# ── Step 5: Train Adjustment Models ─────────────────────────────────────────


def train_adjustment_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    meta_train: List[Dict[str, Any]],
) -> Tuple[TeamAdjustmentModel, PlayerAdjustmentModel]:
    """Train both adjustment model types from the training set.

    Returns (team_model, player_model) — both fitted and saved.
    """
    os.makedirs(_MODELS_DIR, exist_ok=True)

    # Build training rows for TeamAdjustmentModel
    team_rows: List[Dict[str, Any]] = []
    player_rows: List[Dict[str, Any]] = []

    for i, meta in enumerate(meta_train):
        team_ability_current = meta.get("team_ability_current", 50.0)
        team_ability_target = meta.get("team_ability_target", 50.0)
        league_ability_current = meta.get("league_ability_current", 50.0)
        league_ability_target = meta.get("league_ability_target", 50.0)

        # Relative ability = team_score - league_mean
        from_ra = team_ability_current - league_ability_current
        to_ra = team_ability_target - league_ability_target
        change_ra = to_ra - from_ra

        pre_per90 = meta.get("pre_per90", {})
        from_pos_avg = meta.get("from_pos_avg", {})
        to_pos_avg = meta.get("to_pos_avg", {})
        position = normalize_position(meta.get("position", "Unknown"))

        for j, m in enumerate(CORE_METRICS):
            actual = float(y_train[i, j])
            player_prev = pre_per90.get(m, 0.0)
            avg_pos_new = to_pos_avg.get(m, 0.0)
            avg_pos_old = from_pos_avg.get(m, 0.0)

            # Team adjustment row:
            # naive_league_expectation approximates what the player's metric would be
            # at the target league's average level. league_ability is 0-100 normalized,
            # so dividing by 100 creates a scaling factor (0.0-1.0) applied to the
            # player's current per-90.
            # Use league mean per-90 if available (Improvement 4); fall back to
            # player's own per-90 as a neutral baseline when league stats unavailable
            league_means = meta.get("league_means", {})
            naive_expectation = league_means.get(m, player_prev)

            team_rows.append({
                "metric": m,
                # Both source and target team relative abilities are predictive:
                # from_ra captures the caliber of environment the player developed
                # in; to_ra captures the quality of the destination team.
                "from_ra": from_ra,
                "to_ra": to_ra,
                "naive_league_expectation": naive_expectation,
                "actual": actual,
            })

            # Player adjustment row
            player_rows.append({
                "position": position,
                "metric": m,
                "player_previous_per90": player_prev,
                "avg_position_feature_new_team": avg_pos_new,
                "diff_avg_position_old_vs_new": avg_pos_new - avg_pos_old,
                # Normalize change_ra to [-1, 1] range for polynomial stability
                "change_relative_ability": change_ra / 50.0,
                "actual": actual,
            })

    # ── Bug 4: Validate team rating features before training ─────────────────
    # If ClubElo was unavailable during dataset build, all teams fall back to
    # league-mean ability (relative_ability = 0 for every row).  Training on
    # flat features makes R² ≈ 0.  Fail loudly so this is never silent.
    if team_rows:
        from_ras = np.array([r["from_ra"] for r in team_rows])
        to_ras = np.array([r["to_ra"] for r in team_rows])
        ra_std = float(np.std(from_ras))
        # A row where both from_ra AND to_ra are exactly 0 signals a fallback.
        n_zero = int(np.sum((np.abs(from_ras) < 1e-3) & (np.abs(to_ras) < 1e-3)))
        fallback_pct = n_zero / len(team_rows)
        _log.info(
            "TeamAdjustmentModel feature check: from_ra std=%.3f, "
            "zero_ra rows=%d/%d (%.1f%%)",
            ra_std, n_zero, len(team_rows), fallback_pct * 100,
        )
        print(f"\n  Team rating feature check:")
        print(f"    from_ra std = {ra_std:.4f}")
        print(f"    Zero relative_ability: {n_zero}/{len(team_rows)} ({fallback_pct:.1%})")
        if ra_std < 0.1:
            raise RuntimeError(
                "TeamAdjustmentModel: zero variance in team rating feature "
                "(from_ra std < 0.1). All teams likely returned default 1500 — "
                "check ClubElo API connectivity and power_rankings logs."
            )
        if fallback_pct > 0.20:
            raise RuntimeError(
                f"TeamAdjustmentModel: {fallback_pct:.1%} of rows have "
                f"zero relative_ability (threshold 20%). Too many teams are "
                f"on default rating — re-run after fixing ClubElo connectivity."
            )

    # Fit team model
    team_model = TeamAdjustmentModel()
    team_model.fit(team_rows)
    team_path = team_model.save(os.path.join(_MODELS_DIR, "adjustment_team.pkl"))

    # Fit player model
    player_model = PlayerAdjustmentModel()
    player_model.fit(player_rows)
    player_path = player_model.save(
        os.path.join(_MODELS_DIR, "adjustment_player.pkl")
    )

    # Report R² scores
    print(f"\nAdjustment Model Training Report:")
    print(f"  Team model saved to: {team_path}")
    print(f"  Player model saved to: {player_path}")

    # Compute R² for team model
    print(f"\n  TeamAdjustmentModel R2 per metric:")
    for m in CORE_METRICS:
        if m in team_model.models:
            model = team_model.models[m]
            # Get training data for this metric
            m_rows = [r for r in team_rows if r["metric"] == m]
            if len(m_rows) >= 2:
                X_m = np.array([[r["from_ra"], r["to_ra"]] for r in m_rows])
                y_m = np.array([r["actual"] - r["naive_league_expectation"] for r in m_rows])
                r2 = model.score(X_m, y_m)
                flag = " [!]" if r2 < 0.1 else ""
                print(f"    {m}: R2={r2:.4f}{flag}")

    # Compute R² for player model
    print(f"\n  PlayerAdjustmentModel mean R2 per metric:")
    for m in CORE_METRICS:
        r2_values = []
        for pos, models in player_model.models.items():
            if m in models:
                # Get training data for this position + metric
                pm_rows = [
                    r for r in player_rows
                    if r["metric"] == m and r["position"] == pos
                ]
                if len(pm_rows) >= 2:
                    cra_vals = [r["change_relative_ability"] for r in pm_rows]
                    X_pm = np.array([
                        [
                            r["player_previous_per90"],
                            r["avg_position_feature_new_team"],
                            r["diff_avg_position_old_vs_new"],
                            r["change_relative_ability"],
                            r["change_relative_ability"] ** 2,
                            r["change_relative_ability"] ** 3,
                        ]
                        for r in pm_rows
                    ])
                    y_pm = np.array([r["actual"] for r in pm_rows])
                    try:
                        # Apply scaler if one was fitted (match how model was trained)
                        scaler = player_model._scalers.get(pos, {}).get(m)
                        if scaler is not None:
                            X_pm = scaler.transform(X_pm)
                        r2 = models[m].score(X_pm, y_pm)
                        r2_values.append(r2)
                    except Exception:
                        pass
        if r2_values:
            mean_r2 = np.mean(r2_values)
            flag = " [!]" if mean_r2 < 0.1 else ""
            print(f"    {m}: mean R2={mean_r2:.4f} (across {len(r2_values)} positions){flag}")

    return team_model, player_model


# ── Step 6: Train Neural Network ────────────────────────────────────────────


def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    meta_train: Optional[List[Dict[str, Any]]] = None,
    meta_val: Optional[List[Dict[str, Any]]] = None,
) -> TransferPortalModel:
    """Train the 4-group TransferPortalModel neural network.

    Parameters
    ----------
    meta_train : list[dict], optional
        If provided, 'confidence' values are used as sample weights so
        high-quality samples (more pre-transfer minutes) contribute more.

    Returns the fitted model (already saved to data/models/).
    """
    import joblib
    from sklearn.preprocessing import StandardScaler

    os.makedirs(_MODELS_DIR, exist_ok=True)

    # Normalize features (not targets)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Save scaler
    scaler_path = os.path.join(_MODELS_DIR, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"\nFeature scaler saved to: {scaler_path}")

    # Build model
    model = TransferPortalModel()
    model.build(FEATURE_DIM)

    # Map metrics to column indices in y
    metric_to_idx = {m: i for i, m in enumerate(CORE_METRICS)}

    # Train each group separately
    import tensorflow as tf

    print(f"\nNeural Network Training:")
    print(f"  Input dim: {FEATURE_DIM}")
    print(f"  Training samples: {X_train_scaled.shape[0]}")
    print(f"  Validation samples: {X_val_scaled.shape[0]}")

    # Per-group target scalers to normalise y across different scales
    # (e.g. successful_passes 30-80 vs expected_goals 0-1)
    target_scalers: Dict[str, StandardScaler] = {}

    # ── Data quality: per-metric loss masking for missing xG ────────────────
    # Sofascore tracks shots for almost all leagues but only tracks xG for
    # the top ~30.  When pre_xG = 0 the "delta" label for expected_goals
    # becomes y_delta = post_xG − 0 = absolute post_xG (not a real delta).
    # Instead of dropping these samples entirely (losing ~17% of all
    # passing/defending/dribbling labels), we keep them and use per-sample
    # weight masking: the *shooting* group zeros out the sample weight for
    # affected rows so only the xG/shots head ignores them, while all other
    # groups still learn from the full 12 non-shooting metrics.
    # Goalkeepers legitimately have xG=0 so we exempt position == "G".
    XG_COL = CORE_METRICS.index("expected_goals")
    xg_zero_mask_train = np.zeros(len(X_train), dtype=bool)
    xg_zero_mask_val = np.zeros(len(X_val), dtype=bool)

    if meta_train:
        gk_mask = np.array([m.get("position", "?") == "G" for m in meta_train])
        xg_zero_mask_train = (X_train[:, XG_COL] == 0) & ~gk_mask
        n_masked = int(xg_zero_mask_train.sum())
        if n_masked:
            _log.info(
                "Per-metric loss masking: %d / %d training samples have "
                "missing pre-transfer xG (non-GK) — shooting group will "
                "zero-weight these; other groups keep them",
                n_masked, len(X_train),
            )

    if meta_val:
        gk_mask_val = np.array([m.get("position", "?") == "G" for m in meta_val])
        xg_zero_mask_val = (X_val[:, XG_COL] == 0) & ~gk_mask_val
        n_masked_val = int(xg_zero_mask_val.sum())
        if n_masked_val:
            _log.info(
                "Per-metric loss masking (val): %d / %d validation samples "
                "have missing pre-transfer xG",
                n_masked_val, len(X_val),
            )

    # Compute deltas: train model to predict (post - pre) instead of absolute
    # post-transfer values.  Pre-transfer per-90 is already an excellent
    # predictor of post-transfer per-90, so the model only needs to learn
    # small adjustments.  predict() adds pre back to recover absolute values.
    # X[:, :len(CORE_METRICS)] are the unscaled pre-transfer per-90 stats.
    y_train_delta = y_train - X_train[:, :len(CORE_METRICS)]
    y_val_delta = y_val - X_val[:, :len(CORE_METRICS)]

    # Sample weights: combine class-balance correction (10:1 transfer ratio)
    # with confidence weighting (more pre-transfer minutes → more reliable).
    # Without class balancing the model ignores non-transfer samples entirely.
    sample_weights = None
    if meta_train:
        n_total = len(meta_train)
        n_transfer = sum(1 for m in meta_train if m.get("is_transfer", True))
        n_non_transfer = n_total - n_transfer

        if n_non_transfer > 0 and n_transfer > 0:
            # Inverse-frequency class weights so both classes contribute equally.
            w_transfer = n_total / (2.0 * n_transfer)
            w_non_transfer = n_total / (2.0 * n_non_transfer)
            _log.info(
                "Class balance: %d transfer (w=%.3f), %d non-transfer (w=%.3f)",
                n_transfer, w_transfer, n_non_transfer, w_non_transfer,
            )
            class_w = np.array(
                [w_transfer if m.get("is_transfer", True) else w_non_transfer
                 for m in meta_train],
                dtype=np.float32,
            )
        else:
            class_w = np.ones(n_total, dtype=np.float32)

        confidence_w = np.array(
            [m.get("confidence", 1.0) for m in meta_train], dtype=np.float32,
        )
        raw_weights = class_w * confidence_w
        mean_w = raw_weights.mean()
        if mean_w > 0:
            sample_weights = raw_weights / mean_w
            _log.info(
                "Sample weights (class×confidence): min=%.3f, max=%.3f, mean=%.3f",
                sample_weights.min(), sample_weights.max(), sample_weights.mean(),
            )
        else:
            # All-zero confidence → fall back to uniform weighting
            _log.warning("All confidence values are zero -- using uniform sample weights")
            sample_weights = None

    # ── LR Warmup callback ─────────────────────────────────────────────────
    # Linear warmup from a low learning rate to the target over `warmup_epochs`
    # helps the network find a better loss basin before ReduceLROnPlateau
    # kicks in.
    _WARMUP_EPOCHS = 10
    _WARMUP_START_LR = 1e-5
    _TARGET_LR = 5e-4

    class _LinearWarmup(tf.keras.callbacks.Callback):
        """Linear LR warmup from *start_lr* to *target_lr* over *warmup_epochs*."""

        def __init__(self, warmup_epochs: int, start_lr: float, target_lr: float):
            super().__init__()
            self.warmup_epochs = warmup_epochs
            self.start_lr = start_lr
            self.target_lr = target_lr

        def on_epoch_begin(self, epoch: int, logs: Any = None) -> None:
            if epoch < self.warmup_epochs:
                lr = self.start_lr + (self.target_lr - self.start_lr) * (
                    epoch / max(1, self.warmup_epochs)
                )
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)

    for group_name, targets in MODEL_GROUPS.items():
        # Get target columns
        target_indices = [metric_to_idx[t] for t in targets]
        y_group_train_raw = y_train_delta[:, target_indices]
        y_group_val_raw = y_val_delta[:, target_indices]

        # Scale targets so all groups train on comparable loss scales
        y_scaler = StandardScaler()
        y_group_train = y_scaler.fit_transform(y_group_train_raw)
        y_group_val = y_scaler.transform(y_group_val_raw)
        target_scalers[group_name] = y_scaler

        # Extract per-group feature subset from the full scaled array
        all_keys = _feature_keys()
        key_to_idx = {k: i for i, k in enumerate(all_keys)}
        group_indices = [key_to_idx[k] for k in GROUP_FEATURE_SUBSETS[group_name]]
        X_group_train = X_train_scaled[:, group_indices]
        X_group_val = X_val_scaled[:, group_indices]

        print(f"\n  Group: {group_name} ({len(targets)} targets, {len(group_indices)} features)")

        # Per-group sample weights: start from base weights, then apply
        # xG-zero mask for the shooting group only.
        group_weights = sample_weights
        if group_name == "shooting" and xg_zero_mask_train.any():
            # Zero out sample weight for rows with missing pre-xG so the
            # shooting model ignores their corrupted delta labels while
            # other groups still learn from these samples' valid metrics.
            if sample_weights is not None:
                group_weights = sample_weights.copy()
            else:
                group_weights = np.ones(len(X_group_train), dtype=np.float32)
            group_weights[xg_zero_mask_train] = 0.0
            n_zero = int(xg_zero_mask_train.sum())
            _log.info(
                "Shooting group: zero-weighting %d samples with missing xG "
                "(keeping them in other groups)",
                n_zero,
            )

        # Recompile with Huber loss (robust to outlier deltas) and lower
        # learning rate for stable convergence on small delta targets.
        # Start at warmup LR; the _LinearWarmup callback ramps it up.
        model.models[group_name].compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=_WARMUP_START_LR),
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=["mae"],
        )

        # Validation data: for the shooting group, exclude xG-zero samples
        # from validation too so val_loss is computed on clean deltas only.
        if group_name == "shooting" and xg_zero_mask_val.any():
            val_keep = ~xg_zero_mask_val
            val_data = (X_group_val[val_keep], y_group_val[val_keep])
        else:
            val_data = (X_group_val, y_group_val)

        # Early stopping + LR warmup + LR reduction on plateau
        callbacks = [
            _LinearWarmup(_WARMUP_EPOCHS, _WARMUP_START_LR, _TARGET_LR),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-5,
                verbose=0,
            ),
        ]

        history = model.models[group_name].fit(
            X_group_train,
            y_group_train,
            sample_weight=group_weights,
            epochs=150,
            batch_size=32,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=0,
        )

        # Report (losses are in scaled space — also report original-scale MSE)
        epochs_trained = len(history.history.get("loss", []))
        if epochs_trained == 0:
            _log.warning("Group %s: 0 epochs trained -- check data", group_name)
            print(f"    WARNING: 0 epochs trained - skipping report")
            continue

        final_train_loss = history.history["loss"][-1]
        final_val_loss = history.history["val_loss"][-1]
        best_val_loss = min(history.history["val_loss"])

        print(f"    Epochs: {epochs_trained}")
        print(f"    Train loss (scaled): {final_train_loss:.6f}")
        print(f"    Val loss (scaled):   {final_val_loss:.6f}")
        print(f"    Best val loss (scaled): {best_val_loss:.6f}")

        if final_val_loss > final_train_loss * 1.5:
            print(f"    WARNING: Potential overfitting detected!")

    model.fitted = True

    # Save target scalers alongside model weights
    target_scaler_path = os.path.join(_MODELS_DIR, "target_scalers.pkl")
    joblib.dump(target_scalers, target_scaler_path)
    print(f"\nTarget scalers saved to: {target_scaler_path}")

    # Save model
    save_dir = model.save()
    print(f"\nModel saved to: {save_dir}")
    _log.info(f"TIMER: model saved — {datetime.now().strftime('%H:%M:%S')}")

    # Compare trained model vs heuristic baseline on validation set
    print(f"\nModel vs Heuristic Baseline (Validation Set):")
    _compare_model_vs_heuristic(model, scaler, X_val, y_val, target_scalers)

    return model


def _compare_model_vs_heuristic(
    model: TransferPortalModel,
    scaler: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_scalers: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare trained model MSE vs heuristic MSE on validation data."""
    from backend.features.adjustment_models import paper_heuristic_predict

    metric_to_idx = {m: i for i, m in enumerate(CORE_METRICS)}

    trained_errors = {m: [] for m in CORE_METRICS}
    heuristic_errors = {m: [] for m in CORE_METRICS}

    # Attach scalers to the model so predict() handles scaling internally.
    # This avoids manual pre-scaling which risks double-scaling if
    # model._scaler is set elsewhere.
    old_scaler = model._scaler
    old_target_scalers = model._target_scalers
    model._scaler = scaler
    if target_scalers is not None:
        model._target_scalers = target_scalers

    keys = _feature_keys_list()

    for i in range(len(X_val)):
        # Trained model prediction — pass UNSCALED features;
        # model.predict() handles scaling internally via model._scaler.
        feature_dict = {}
        for j, key in enumerate(keys):
            feature_dict[key] = float(X_val[i, j])

        trained_preds = model.predict(feature_dict)

        # Heuristic prediction — build a feature dict from raw (unscaled) features
        raw_feature_dict = {}
        for j, key in enumerate(keys):
            raw_feature_dict[key] = float(X_val[i, j])

        # Extract components for heuristic using key-based lookup (not hardcoded indices)
        key_to_idx = {k: j for j, k in enumerate(keys)}
        player_per90 = {}
        team_pos_current = {}
        team_pos_target = {}
        for m in CORE_METRICS:
            player_per90[m] = float(X_val[i, key_to_idx[f"player_{m}"]])
            team_pos_current[m] = float(X_val[i, key_to_idx[f"team_pos_current_{m}"]])
            team_pos_target[m] = float(X_val[i, key_to_idx[f"team_pos_target_{m}"]])

        ra_current = (
            float(X_val[i, key_to_idx["team_ability_current"]])
            - float(X_val[i, key_to_idx["league_ability_current"]])
        )
        ra_target = (
            float(X_val[i, key_to_idx["team_ability_target"]])
            - float(X_val[i, key_to_idx["league_ability_target"]])
        )
        change_ra = ra_target - ra_current

        try:
            heuristic_preds = paper_heuristic_predict(
                player_per90=player_per90,
                source_pos_avg=team_pos_current,
                target_pos_avg=team_pos_target,
                change_relative_ability=change_ra,
            )
        except Exception as exc:
            _log.warning("paper_heuristic_predict failed: %s", exc)
            heuristic_preds = player_per90.copy()

        # Compute errors
        for m in CORE_METRICS:
            actual = float(y_val[i, metric_to_idx[m]])
            t_pred = trained_preds.get(m, 0.0)
            h_pred = heuristic_preds.get(m, player_per90.get(m, 0.0))

            trained_errors[m].append((t_pred - actual) ** 2)
            heuristic_errors[m].append((h_pred - actual) ** 2)

    # Restore original scalers
    model._scaler = old_scaler
    model._target_scalers = old_target_scalers

    # Report
    report = {}
    print(f"\n  {'Metric':<30} {'Trained MSE':>12} {'Heuristic MSE':>14} {'Improvement':>12}")
    print(f"  {'-'*68}")

    for m in CORE_METRICS:
        t_mse = np.mean(trained_errors[m]) if trained_errors[m] else 0.0
        h_mse = np.mean(heuristic_errors[m]) if heuristic_errors[m] else 0.0
        improvement = ((h_mse - t_mse) / h_mse * 100) if h_mse > 0 else 0.0
        flag = " [!]" if improvement < -20 else ""
        print(f"  {m:<30} {t_mse:>12.6f} {h_mse:>14.6f} {improvement:>10.1f}%{flag}")
        report[m] = {"trained_mse": float(t_mse), "heuristic_mse": float(h_mse), "improvement_pct": float(improvement)}

    overall_t = np.mean([np.mean(trained_errors[m]) for m in CORE_METRICS])
    overall_h = np.mean([np.mean(heuristic_errors[m]) for m in CORE_METRICS])
    overall_imp = ((overall_h - overall_t) / overall_h * 100) if overall_h > 0 else 0.0
    print(f"\n  Overall: Trained MSE={overall_t:.6f}, Heuristic MSE={overall_h:.6f}, Improvement={overall_imp:.1f}%")

    return report


def _feature_keys_list() -> List[str]:
    """Return the ordered list of feature keys matching the 79-feature vector.

    Must stay in sync with ``_feature_keys()`` in transfer_portal.py —
    includes raw Elo, REEP metadata, interaction features, relative
    dominance features, and per-metric league-normalised features.
    """
    keys = []
    for m in CORE_METRICS:
        keys.append(f"player_{m}")
    keys.extend([
        "team_ability_current", "team_ability_target",
        "league_ability_current", "league_ability_target",
    ])
    # Raw Elo scores (absolute scale)
    keys.append("raw_elo_current")
    keys.append("raw_elo_target")
    # REEP player metadata
    keys.append("player_height_cm")
    keys.append("player_age")
    for m in CORE_METRICS:
        keys.append(f"team_pos_current_{m}")
    for m in CORE_METRICS:
        keys.append(f"team_pos_target_{m}")
    # Interaction features (must match transfer_portal._feature_keys())
    keys.append("interaction_ability_gap")
    keys.append("interaction_gap_squared")
    keys.append("interaction_league_gap")
    # Relative team dominance within league (Phase 6)
    keys.append("relative_ability_current")
    keys.append("relative_ability_target")
    keys.append("relative_ability_gap")
    # Per-metric league-normalised features (Phase 5)
    for m in CORE_METRICS:
        keys.append(f"league_norm_{m}")
    for m in CORE_METRICS:
        keys.append(f"league_mean_ratio_{m}")
    return keys


# ── Step 9: Main Entry Point ────────────────────────────────────────────────


def run_pipeline(
    *,
    league_codes: Optional[List[str]] = None,
    seasons_back: int = 5,
    skip_discovery: bool = False,
    skip_build: bool = False,
    skip_training: bool = False,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    api_delay: float = API_CALL_DELAY_SECONDS,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> bool:
    """Run the full training pipeline programmatically.

    Parameters
    ----------
    league_codes : list of str, optional
        League codes to scan. Defaults to ``DEFAULT_LEAGUE_CODES``.
    seasons_back : int
        Number of historical seasons to use.
    skip_discovery : bool
        If True, load cached transfer records instead of discovering.
    skip_build : bool
        If True, load pre-built feature matrices from
        ``data/models/matrices/`` instead of rebuilding from API.
    skip_training : bool
        If True, skip model training (backtest only).
    val_ratio, test_ratio : float
        Validation and test split ratios.
    api_delay : float
        Seconds between Sofascore API calls (default: 0).
    progress_callback : callable, optional
        Called with ``(step: str, detail: str)`` to report progress.
        When *None*, messages go to ``print()``.

    Returns
    -------
    bool
        True if training completed successfully, False otherwise.
    """
    global API_CALL_DELAY_SECONDS
    API_CALL_DELAY_SECONDS = api_delay
    sofascore_client.set_inter_request_delay(api_delay)

    if league_codes is None:
        league_codes = list(DEFAULT_LEAGUE_CODES)

    def _report(step: str, detail: str = "") -> None:
        if progress_callback is not None:
            progress_callback(step, detail)
        else:
            msg = step if not detail else f"{step} — {detail}"
            print(msg)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # ── Bulletproof Unicode safety for Windows cp1252 consoles ───────────────
    # reconfigure() fails silently on Windows Store Python / Windows console
    # streams that don't support it.  Instead we patch every handler's emit()
    # so that UnicodeEncodeError is caught at the last moment and non-ASCII
    # chars are replaced with '?'.  Works regardless of the stream type.
    def _make_unicode_safe(handler: logging.Handler) -> None:
        _orig_emit = handler.emit

        def _safe_emit(record: logging.LogRecord) -> None:
            try:
                _orig_emit(record)
            except UnicodeEncodeError:
                # Re-encode message and args to ASCII with replacement
                safe = logging.makeLogRecord(record.__dict__)
                safe.msg = str(record.msg).encode("ascii", "replace").decode("ascii")
                if record.args:
                    if isinstance(record.args, dict):
                        safe.args = {
                            k: str(v).encode("ascii", "replace").decode("ascii")
                            if isinstance(v, str) else v
                            for k, v in record.args.items()
                        }
                    else:
                        args = record.args if isinstance(record.args, tuple) else (record.args,)
                        safe.args = tuple(
                            str(a).encode("ascii", "replace").decode("ascii")
                            if isinstance(a, str) else a
                            for a in args
                        )
                try:
                    _orig_emit(safe)
                except Exception:
                    pass

        handler.emit = _safe_emit  # type: ignore[method-assign]

    for _h in logging.root.handlers:
        _make_unicode_safe(_h)

    _report("Starting", "TransferScope Training Pipeline")

    records_path = os.path.join(_MODELS_DIR, "transfer_records.json")

    # Step 2: Discover transfers
    if skip_discovery and os.path.exists(records_path):
        _report("Loading cached transfers")
        with open(records_path, "r") as f:
            raw_records = json.load(f)
        records = [TransferRecord(**r) for r in raw_records]
        _report("Loaded transfers", f"{len(records)} records from cache")
    else:
        _report("Discovering transfers", f"{len(league_codes)} leagues, {seasons_back} seasons")
        records = discover_transfers(league_codes, seasons_back)

        os.makedirs(_MODELS_DIR, exist_ok=True)
        with open(records_path, "w") as f:
            json.dump([asdict(r) for r in records], f, indent=2, default=str)
        _report("Transfers found", f"{len(records)} records saved")

    if len(records) < 10:
        _report("Insufficient data", f"Only {len(records)} transfers found - need >=10")
        return False

    # Step 3: Build full dataset (or load pre-built matrices)
    _MATRICES_DIR = os.path.join(_MODELS_DIR, "matrices")

    if skip_build:
        X_path = os.path.join(_MATRICES_DIR, "X.npy")
        y_path = os.path.join(_MATRICES_DIR, "y.npy")
        meta_path = os.path.join(_MATRICES_DIR, "metadata.json")
        if os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(meta_path):
            X = np.load(X_path)
            y = np.load(y_path)
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            _report("Loaded feature matrices", f"{len(X)} samples from data/models/matrices/")
        else:
            _report("WARNING: --skip-build set but no matrices found — building from API")
            skip_build = False

    if not skip_build:
        _report("Building dataset", "Discovering non-transfer samples…")
        # Build (player_id, season_id) exclusion pairs — a player is only
        # excluded for the specific season in which they transferred, so
        # they can still contribute non-transfer samples for other seasons.
        transfer_player_seasons = set()
        for r in records:
            transfer_player_seasons.add((r.player_id, r.pre_transfer_season_id))
            transfer_player_seasons.add((r.player_id, r.post_transfer_season_id))
        # Target approximately 20% of transfer samples as non-transfer controls
        nt_target = max(1, len(records) // 5)
        non_transfer_records = discover_non_transfers(
            league_codes, seasons_back,
            exclude_player_seasons=transfer_player_seasons,
            target_count=nt_target,
        )
        _report("Non-transfers found", f"{len(non_transfer_records)} records (target: {nt_target})")

        # Use relaxed minutes threshold for non-transfer samples to match
        # discover_non_transfers retry logic (max(250, _NT_MIN_MINUTES // 2))
        nt_effective_min = max(250, _NT_MIN_MINUTES // 2)
        X, y, metadata = build_full_dataset(
            records, non_transfer_records, nt_min_minutes=nt_effective_min,
        )

        # Save feature matrices to disk for --skip-build on subsequent runs
        os.makedirs(_MATRICES_DIR, exist_ok=True)
        np.save(os.path.join(_MATRICES_DIR, "X.npy"), X)
        np.save(os.path.join(_MATRICES_DIR, "y.npy"), y)
        with open(os.path.join(_MATRICES_DIR, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        _report("Saved feature matrices", f"{len(X)} samples -> data/models/matrices/")

    if len(metadata) < 10:
        _report("Insufficient data", f"Only {len(metadata)} valid samples")
        return False

    # Step 4: Split dataset
    _report("Splitting dataset", "Temporal train/val/test split")
    (
        X_train, y_train, X_val, y_val, X_test, y_test,
        meta_train, meta_val, meta_test,
    ) = split_dataset(X, y, metadata, val_ratio, test_ratio)

    # Step 4b: Inject team-position averages from TRAINING data only
    # (avoids data leakage — test/val team-pos comes from train distribution)
    _report("Injecting team-position averages", "From training data only (no leakage)")
    inject_team_pos_averages(X_train, meta_train, meta_train)
    inject_team_pos_averages(X_val, meta_val, meta_train)
    inject_team_pos_averages(X_test, meta_test, meta_train)

    if not skip_training:
        # Step 5: Train adjustment models
        _report("Training adjustment models", "sklearn LinearRegression × 13 metrics")
        train_adjustment_models(X_train, y_train, meta_train)

        # Step 6: Train neural network
        _log.info(f"TIMER: neural network training start — {datetime.now().strftime('%H:%M:%S')}")
        _report("Training neural network", "4-group multi-head TensorFlow model")
        train_neural_network(X_train, y_train, X_val, y_val, meta_train=meta_train, meta_val=meta_val)
    else:
        _report("Skipping training", "--skip-training flag set")

    # Step 8: Backtesting
    if len(meta_test) > 0:
        _report("Running backtest", f"{len(meta_test)} test samples")
        from backend.models.backtester import run_backtest, show_example_predictions

        _log.info(f"TIMER: backtest start — {datetime.now().strftime('%H:%M:%S')}")
        run_backtest(X_test, y_test, meta_test, meta_train=meta_train)
        show_example_predictions(meta_test, n=10)
        _log.info(f"TIMER: backtest end — {datetime.now().strftime('%H:%M:%S')}")
    else:
        _report("Backtesting", "No test data available")

    _report("Complete", "Models saved to data/models/")
    return True


def main() -> None:
    """Run the full training pipeline (CLI entry point)."""
    parser = argparse.ArgumentParser(
        description="TransferScope Training Pipeline"
    )
    parser.add_argument(
        "--leagues",
        type=str,
        default=",".join(DEFAULT_LEAGUE_CODES),
        help="Comma-separated league codes (default: top 11)",
    )
    parser.add_argument(
        "--seasons-back",
        type=int,
        default=5,
        help="Number of historical seasons to use (default: 5)",
    )
    parser.add_argument(
        "--skip-discovery",
        action="store_true",
        help="Load cached transfer records from data/models/transfer_records.json",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, run backtesting only (requires existing weights)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Load pre-built feature matrices from data/models/matrices/ instead of rebuilding from API",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Test split ratio (default: 0.10)",
    )
    parser.add_argument(
        "--api-delay",
        type=float,
        default=API_CALL_DELAY_SECONDS,
        help="Delay between API calls in seconds (default: from env or 0)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TransferScope Training Pipeline")
    print("=" * 60)

    league_codes = [c.strip() for c in args.leagues.split(",")]

    success = run_pipeline(
        league_codes=league_codes,
        seasons_back=args.seasons_back,
        skip_discovery=args.skip_discovery,
        skip_build=args.skip_build,
        skip_training=args.skip_training,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        api_delay=args.api_delay,
    )

    if success:
        print("\n" + "=" * 60)
        print("Training complete. Models saved to data/models/")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Training failed -- see messages above.")
        print("=" * 60)


if __name__ == "__main__":
    main()
