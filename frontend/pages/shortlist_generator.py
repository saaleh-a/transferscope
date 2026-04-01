"""Shortlist Generator — replacement candidate ranking (paper Fig 2).

Inputs: player to replace, metric weight sliders (0.0-1.0),
        filters (age, value, league, position, minutes, Power Ranking cap).
Output: ranked candidate table with similarity scores.
Click any candidate to open their full transfer impact dashboard.

Now supports multi-league search via Sofascore tournament stats.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from backend.data import sofascore_client
from backend.data.sofascore_client import (
    CORE_METRICS, OFFENSIVE_METRICS, DEFENSIVE_METRICS, normalize_position,
)
from backend.features import power_rankings
from backend.features.adjustment_models import paper_heuristic_predict
from backend.models.shortlist_scorer import (
    Candidate,
    MIN_MINUTES_THRESHOLD,
    ShortlistFilters,
    compute_percentage_changes,
    filter_candidates as _filter_candidates_fn,
    score_candidates,
)
from backend.utils.league_registry import LEAGUES
from frontend.theme import section_header, player_info_card

_log = logging.getLogger(__name__)

# Delay between league API calls to avoid Sofascore rate-limiting (seconds).
# Sofascore returns 403/429 after rapid sequential requests.  A small delay
# between leagues improves success rate.  The adaptive backoff in
# sofascore_client will increase the delay automatically if needed.
_INTER_LEAGUE_DELAY = float(os.environ.get("SOFASCORE_INTER_LEAGUE_DELAY", "0.5"))

from frontend.constants import METRIC_LABELS as _LABELS

_WEIGHTS_EXPLANATION = (
    "Each weight controls how important a metric is when finding similar players. "
    "A weight of **1.0** means the metric is fully considered; **0.0** ignores it. "
    "Candidates are scored by:\n\n"
    "1. **Standardize** each weighted metric across all candidates\n"
    "2. **Cluster** players into style groups using k-means\n"
    "3. **Rank** by weighted Euclidean distance to the player being replaced\n"
    "4. **Bonus** for candidates in the same style cluster\n\n"
    "Higher similarity % = closer match to the player's style profile. "
    "For example, set xG and xA to 1.0 and everything else to 0.0 to find "
    "the most prolific attackers."
)


def _collect_league_candidates(
    tid: int,
    season_id: Optional[int],
    player_id: int,
    source_norm: float,
    source_league: float,
    source_pos_avg: Dict[str, float],
    league_name: str,
) -> List[Candidate]:
    """Fetch player stats for one league and build Candidate objects.

    Separated from render() for clarity and testability.
    """
    try:
        league_players = sofascore_client.get_league_player_stats(
            tid, season_id, limit=100,
        )
    except Exception as exc:
        _log.info("Failed to fetch league %s (tid=%d): %s", league_name, tid, exc)
        return []

    candidates: List[Candidate] = []
    skipped = 0
    for lp in league_players:
        lp_id = lp.get("id")
        if lp_id == player_id:
            continue

        lp_per90 = lp.get("per90") or {}
        # Use `is not None` to avoid skipping legitimate 0.0 values
        if not any(lp_per90.get(m) is not None for m in CORE_METRICS):
            continue

        try:
            lp_current = {}
            for m in CORE_METRICS:
                v = lp_per90.get(m)
                lp_current[m] = v if v is not None else 0

            # Paper-aligned prediction: use team-position style data
            # to predict how this candidate would perform at the source team.
            lp_team = lp.get("team", "")
            lp_ranking = power_rankings.get_team_ranking(lp_team)
            lp_norm = lp_ranking.normalized_score if lp_ranking else 50.0
            lp_league = lp_ranking.league_mean_normalized if lp_ranking else 50.0
            # Relative ability change if player moved to the source team
            delta_ra = (source_norm - source_league) - (lp_norm - lp_league)

            # Use the candidate's own stats as a proxy for their team's
            # position average (fetching each candidate's team would be
            # too many API calls).  The source team's position average
            # is the real tactical style target.
            predicted = paper_heuristic_predict(
                player_per90=lp_current,
                source_pos_avg=lp_current,
                target_pos_avg=source_pos_avg,
                change_relative_ability=delta_ra,
                player_rating=lp.get("rating"),
                source_league_mean=lp_league,
                target_league_mean=source_league,
            )

            # Normalize position for consistent filtering
            raw_pos = lp.get("position", "Unknown")
            norm_pos = normalize_position(raw_pos)

            candidates.append(Candidate(
                player_id=lp_id,
                name=lp.get("name", "Unknown"),
                team=lp_team,
                position=norm_pos if norm_pos != "Unknown" else raw_pos,
                age=lp.get("age"),
                minutes_played=lp.get("minutes_played"),
                league=league_name,
                predicted_per90=predicted,
                current_per90=lp_current,
                club_power_ranking=lp_norm,
                rating=lp.get("rating"),
            ))
        except Exception as exc:
            skipped += 1
            _log.debug(
                "Skipped player %s (id=%s) in %s: %s",
                lp.get("name", "?"), lp_id, league_name, exc,
            )

    if skipped:
        _log.info(
            "League %s: %d players built, %d skipped due to errors",
            league_name, len(candidates), skipped,
        )

    return candidates


def render():
    st.header("Shortlist Generator")
    st.caption("Find replacement candidates ranked by weighted similarity across leagues")

    # ── Player to replace ────────────────────────────────────────────────
    player_query = st.text_input(
        "Player to replace",
        placeholder="e.g. Bukayo Saka",
        key="shortlist_player",
    )

    if not player_query:
        st.info("Enter a player name to generate a replacement shortlist.")
        return

    with st.spinner("Searching..."):
        try:
            results = sofascore_client.search_player(player_query)
        except Exception as e:
            st.error(f"Search failed: {e}")
            return

    if not results:
        st.warning("No players found.")
        return

    options = {
        f"{p['name']}"
        + (f" · Age {p['age']}" if p.get("age") else "")
        + (f" · {p['nationality']}" if p.get("nationality") else "")
        + (f" · {p['team_name']}" if p.get("team_name") else "")
        : p for p in results
    }
    selected = st.selectbox("Select player", list(options.keys()), key="shortlist_select")
    if selected is None or selected not in options:
        st.info("Select a player from the dropdown.")
        return
    player = options[selected]

    with st.spinner("Fetching stats..."):
        try:
            player_stats = sofascore_client.get_player_stats(player["id"])
        except Exception as e:
            st.error(f"Failed to fetch stats: {e}")
            return

    player_name = player_stats.get("name", "Unknown")
    current_per90 = player_stats.get("per90", {})
    position = player_stats.get("position", "Unknown")
    current_team = player_stats.get("team", "")
    minutes_played = player_stats.get("minutes_played", 0)
    if minutes_played is None:
        minutes_played = 0

    player_info_card(
        f"Replacing: {player_name}", current_team, position, minutes_played
    )

    # ── Metric weight sliders ────────────────────────────────────────────
    section_header("Metric Weights", "Set importance of each metric — 0 = ignore, 1 = max")
    with st.expander("ℹ️ How weights work", expanded=False):
        st.markdown(_WEIGHTS_EXPLANATION)

    weights: Dict[str, float] = {}
    cols = st.columns(3)
    for i, metric in enumerate(CORE_METRICS):
        label = _LABELS.get(metric, metric)
        with cols[i % 3]:
            weights[metric] = st.slider(
                label, 0.0, 1.0, 0.5, 0.1, key=f"w_{metric}"
            )

    # ── Filters ──────────────────────────────────────────────────────────
    section_header("Filters", "Narrow the candidate pool")
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        # Defaults: age 35 (wide enough for experienced targets), min 270 mins
        # (≈3 games — low to avoid excluding sparse-data candidates.
        # None-passthrough filter design means unknowns pass through anyway).
        max_age = st.number_input("Max age", 16, 45, 35, key="f_age")
        min_minutes = st.number_input(
            "Min minutes played", 0, 5000, MIN_MINUTES_THRESHOLD, key="f_mins",
        )
    with fcol2:
        league_names = ["Any"] + [l.name for l in LEAGUES.values()]
        selected_leagues = st.multiselect("Leagues", league_names, default=["Any"], key="f_leagues")
        positions = st.multiselect(
            "Positions",
            ["Any", "Forward", "Midfielder", "Defender", "Goalkeeper", "Right Winger",
             "Left Winger", "Centre-Forward", "Central Midfielder", "Centre-Back"],
            default=["Any"],
            key="f_positions",
        )
    with fcol3:
        max_pr = st.slider("Max club Power Ranking", 0, 100, 100, key="f_pr")

    # Normalize positions for consistent matching between filter and data
    normalized_filter_positions = None
    if "Any" not in positions:
        # Build a set of normalized categories from the selected positions
        normalized_filter_positions = list(set(
            normalize_position(p) for p in positions
        ))
        # Remove "Unknown" if it crept in from unrecognized input
        normalized_filter_positions = [p for p in normalized_filter_positions if p != "Unknown"]
        if not normalized_filter_positions:
            normalized_filter_positions = None

    filters = ShortlistFilters(
        max_age=max_age if max_age < 45 else None,
        min_minutes_played=min_minutes if min_minutes > 0 else None,
        leagues=None if "Any" in selected_leagues else selected_leagues,
        positions=normalized_filter_positions,
        max_power_ranking=max_pr if max_pr < 100 else None,
    )

    # ── Season selector ──────────────────────────────────────────────────
    tournament_id = player.get("tournament_id")
    selected_season_id: Optional[int] = None

    if tournament_id:
        seasons = sofascore_client.get_season_list(tournament_id)
        if seasons:
            season_labels = {s["name"]: s["id"] for s in seasons}
            chosen_season = st.selectbox(
                "Season",
                ["Current"] + list(season_labels.keys()),
                key="sg_season",
            )
            if chosen_season != "Current":
                selected_season_id = season_labels[chosen_season]

    # ── Generate shortlist ───────────────────────────────────────────────
    if not st.button("Generate Shortlist", type="primary"):
        return

    with st.spinner("Generating shortlist... This may take a moment."):
        team_id = player_stats.get("team_id")
        if not team_id:
            st.error("Cannot determine player's team.")
            return

        # Get current team's power ranking for context
        source_ranking = power_rankings.get_team_ranking(
            player_stats.get("team", "")
        )
        source_norm = source_ranking.normalized_score if source_ranking else 50.0
        source_league = source_ranking.league_mean_normalized if source_ranking else 50.0

        # Fetch source team position averages once (paper Section 2.3)
        # This represents the tactical style of the team the player is joining.
        source_pos_avg: Dict[str, float] = {}
        try:
            source_pos_avg, _ = sofascore_client.get_team_position_averages(
                team_id, position
            )
        except Exception as e:
            st.warning(f"Could not fetch team position data: {e}")
        if not source_pos_avg:
            source_pos_avg = {m: current_per90.get(m, 0) if current_per90.get(m) is not None else 0.0
                              for m in CORE_METRICS}

        candidates: List[Candidate] = []

        # ── Multi-league search ──────────────────────────────────────────
        # Determine which leagues to scan
        if "Any" not in (selected_leagues or ["Any"]):
            target_leagues = [
                (code, info) for code, info in LEAGUES.items()
                if info.name in selected_leagues
            ]
        else:
            # Default: scan Big 5 only for reliability and speed.
            # Scanning 11+ leagues triggers Sofascore rate-limiting (403/429)
            # which causes 0 results.  Users can select more leagues explicitly.
            _DEFAULT_LEAGUES = {
                "ENG1", "ESP1", "GER1", "ITA1", "FRA1",
            }
            target_leagues = [
                (code, info) for code, info in LEAGUES.items()
                if code in _DEFAULT_LEAGUES
            ]

        # Always include the player's own league first (most likely to succeed
        # since season resolution is already cached from the player search).
        player_league_tid = player.get("tournament_id")
        if player_league_tid:
            from backend.utils.league_registry import get_by_sofascore_id
            player_league_info = get_by_sofascore_id(player_league_tid)
            if player_league_info:
                # Insert at front so it's scanned first
                already = {code for code, _ in target_leagues}
                for code, info in LEAGUES.items():
                    if info is player_league_info and code not in already:
                        target_leagues.insert(0, (code, info))
                        break

        league_progress = st.progress(0, text="Scanning leagues...")
        total_leagues = len(target_leagues)
        leagues_with_data = 0
        league_diagnostics: List[str] = []

        for li, (league_code, league_info) in enumerate(target_leagues):
            league_progress.progress(
                (li + 1) / total_leagues,
                text=f"Scanning {league_info.name}... ({len(candidates)} candidates found)",
            )
            tid = league_info.sofascore_tournament_id
            if tid is None:
                continue

            # Rate-limit protection: delay between league API calls.
            # Without this, Sofascore returns 403/429 after 2-3 rapid calls,
            # causing all subsequent leagues to fail → 0 candidates.
            if li > 0:
                time.sleep(_INTER_LEAGUE_DELAY)

            try:
                league_candidates = _collect_league_candidates(
                    tid=tid,
                    season_id=selected_season_id,
                    player_id=player["id"],
                    source_norm=source_norm,
                    source_league=source_league,
                    source_pos_avg=source_pos_avg,
                    league_name=league_info.name,
                )
            except Exception as exc:
                _log.warning(
                    "Unhandled error scanning league %s: %s",
                    league_info.name, exc,
                )
                league_candidates = []

            if league_candidates:
                leagues_with_data += 1
                candidates.extend(league_candidates)
                league_diagnostics.append(
                    f"✅ {league_info.name}: {len(league_candidates)} players"
                )
            else:
                league_diagnostics.append(f"⚠️ {league_info.name}: 0 players")

        league_progress.empty()

        # Show diagnostic info so user knows what happened per league
        with st.expander(
            f"📊 League scan results — {leagues_with_data}/{total_leagues} leagues returned data",
            expanded=leagues_with_data == 0,
        ):
            for diag in league_diagnostics:
                st.text(diag)
            if leagues_with_data == 0:
                st.warning(
                    "**No data from any league.** This is usually caused by "
                    "Sofascore API rate-limiting.\n\n"
                    "**Try:**\n"
                    "1. Wait 2-3 minutes and try again\n"
                    "2. Select just 1-2 specific leagues instead of 'Any'\n"
                    "3. Clear the cache (sidebar) if data seems stale"
                )

        # Also include current teammates for completeness
        try:
            team_players = sofascore_client.get_team_players_stats(team_id)
        except Exception as e:
            _log.warning("Failed to fetch team players: %s", e)
            team_players = []

        seen_ids = {c.player_id for c in candidates}
        for tp in team_players:
            tp_id = tp.get("id")
            if tp_id == player["id"] or tp_id in seen_ids:
                continue

            try:
                tp_stats = sofascore_client.get_player_stats(tp_id)
            except Exception as e:
                _log.warning("Failed to fetch teammate %s stats: %s", tp_id, e)
                continue

            tp_per90 = tp_stats.get("per90", {})
            if not any(tp_per90.get(m) is not None for m in CORE_METRICS):
                continue

            tp_current = {}
            for m in CORE_METRICS:
                v = tp_per90.get(m)
                tp_current[m] = v if v is not None else 0

            raw_pos = tp_stats.get("position", tp.get("position", "Unknown"))
            norm_pos = normalize_position(raw_pos)

            # Teammates are already on the same team — no transfer adjustment
            teammate_league = ""
            if source_ranking:
                li_info = LEAGUES.get(source_ranking.league_code)
                teammate_league = li_info.name if li_info else ""

            # Extract teammate age for consistent filter behavior
            tp_age = tp_stats.get("age")

            candidates.append(Candidate(
                player_id=tp_id,
                name=tp_stats.get("name", tp.get("name", "Unknown")),
                team=tp_stats.get("team", ""),
                position=norm_pos if norm_pos != "Unknown" else raw_pos,
                age=tp_age,
                minutes_played=tp_stats.get("minutes_played"),
                league=teammate_league,
                predicted_per90=tp_current.copy(),
                current_per90=tp_current,
                club_power_ranking=source_norm,
            ))

    if not candidates:
        st.warning(
            "No candidates found. This usually means Sofascore API is rate-limiting requests.\n\n"
            "**Try:**\n"
            "- Wait 2-3 minutes and try again\n"
            "- Select just 1-2 specific leagues\n"
            "- Clear the cache using the sidebar button"
        )
        return

    # ── Enrich candidates with REEP metadata ─────────────────────────────
    try:
        from backend.data.reep_registry import enrich_player as _reep_enrich

        for c in candidates:
            meta = _reep_enrich(c.player_id)
            if meta:
                if c.nationality is None and meta.get("nationality"):
                    c.nationality = meta["nationality"]
                if c.height_cm is None and meta.get("height_cm"):
                    c.height_cm = meta["height_cm"]
    except Exception:
        pass  # REEP unavailable — no enrichment, not critical

    # Build reference per90 for the player being replaced (clean None → 0.0)
    reference_per90 = {
        m: (current_per90.get(m) if current_per90.get(m) is not None else 0.0)
        for m in CORE_METRICS
    }

    # ── Pre-filter diagnostics ───────────────────────────────────────────
    total_before_filter = len(candidates)

    # Count how many candidates would survive each filter independently
    # (for the debug expander — gives users actionable info).
    _diag_minutes_pass = len(_filter_candidates_fn(
        candidates,
        ShortlistFilters(min_minutes_played=filters.min_minutes_played),
    )) if filters.min_minutes_played else total_before_filter
    _diag_position_pass = len(_filter_candidates_fn(
        candidates,
        ShortlistFilters(positions=filters.positions),
    )) if filters.positions else total_before_filter

    # Score candidates using k-means clustering + weighted distance
    scored = score_candidates(candidates, weights, filters, reference_per90=reference_per90)

    # Guarantee at least top 20 results (the scorer already handles
    # filter relaxation internally, but double-check here).
    display_count = min(len(scored), 20) if scored else 0

    # ── Debug expander ────────────────────────────────────────────────────
    with st.expander(
        "🔍 Debug: Pipeline Diagnostics",
        expanded=display_count == 0,
    ):
        st.markdown(
            f"- **Leagues scanned:** {total_leagues}\n"
            f"- **Players fetched:** {total_before_filter}\n"
            f"- **Passing minutes filter** (≥{filters.min_minutes_played or 0} min): "
            f"{_diag_minutes_pass}\n"
            f"- **Passing position filter** "
            f"({', '.join(filters.positions) if filters.positions else 'Any'}): "
            f"{_diag_position_pass}\n"
            f"- **Final shortlist count:** {len(scored)}\n"
            f"- **Displayed:** {display_count}"
        )

    if not scored:
        st.warning(
            f"No candidates match your filters ({total_before_filter} candidates "
            f"found, all filtered out).\n\n"
            f"**Try:** increase Max Age, decrease Min Minutes, or set Positions to 'Any'."
        )
        return

    # ── Display results ──────────────────────────────────────────────────
    section_header(
        f"Top {display_count} of {len(scored)} Candidates",
        f"Ranked by style similarity (k-means clustering) — {total_before_filter} scanned, "
        f"{len(scored)} passed filters",
    )

    rows = []
    for c in scored[:20]:
        changes = compute_percentage_changes(c.current_per90, c.predicted_per90)
        top_changes = sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        top_str = ", ".join(
            f"{_LABELS.get(m, m)}: {v:+.1f}%" for m, v in top_changes
        )
        confidence = "⚠️ Low" if c.low_confidence else "✅ Good"
        rows.append({
            "Rank": len(rows) + 1,
            "Player": c.name,
            "Team": c.team,
            "League": c.league or "",
            "Position": c.position,
            "Age": c.age if c.age is not None else "—",
            "Nat.": c.nationality or "—",
            "Height": f"{c.height_cm} cm" if c.height_cm else "—",
            "Rating": f"{c.rating:.2f}" if c.rating is not None else "—",
            "Similarity": f"{c.score:.1%}",
            "Confidence": confidence,
            "Cluster": "✓ Same" if c.same_cluster_as_reference else ("○ Diff" if c.cluster >= 0 else "—"),
            "Top Changes": top_str,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Detailed view for selected candidate ─────────────────────────────
    if scored:
        candidate_names = [f"{c.name} ({c.team})" for c in scored[:20]]
        detail_selection = st.selectbox(
            "View detailed prediction for:", candidate_names, key="detail_candidate"
        )
        if detail_selection is not None and detail_selection in candidate_names:
            idx = candidate_names.index(detail_selection)
            detail = scored[idx]

            st.markdown(f"### {detail.name} — Detailed Prediction")
            from frontend.components import metric_bar
            changes = compute_percentage_changes(detail.current_per90, detail.predicted_per90)
            metric_bar.show(
                detail.current_per90, detail.predicted_per90, changes,
                title=f"Predicted Changes: {detail.name}",
            )

            # Pizza chart: predicted profile vs reference player
            try:
                from frontend.components import player_pizza
                st.markdown("#### Style Profile Comparison")
                pizza_col1, pizza_col2 = st.columns(2)
                with pizza_col1:
                    player_pizza.show(
                        detail.predicted_per90,
                        player_name=detail.name,
                        comparison_per90=reference_per90,
                        comparison_name=player_name,
                    )
                with pizza_col2:
                    player_pizza.show(
                        reference_per90,
                        player_name=f"{player_name} (Reference)",
                    )
            except Exception as exc:
                _log.debug("Pizza chart unavailable: %s", exc)
