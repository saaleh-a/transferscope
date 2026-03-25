"""Shortlist Generator — replacement candidate ranking (paper Fig 2).

Inputs: player to replace, metric weight sliders (0.0-1.0),
        filters (age, value, league, position, minutes, Power Ranking cap).
Output: ranked candidate table with similarity scores.
Click any candidate to open their full transfer impact dashboard.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import streamlit as st

from backend.data import fotmob_client
from backend.data.fotmob_client import CORE_METRICS
from backend.features import power_rankings
from backend.models.shortlist_scorer import (
    Candidate,
    ShortlistFilters,
    compute_percentage_changes,
    score_candidates,
)
from backend.models.transfer_portal import (
    TransferPortalModel,
    build_feature_dict,
    FEATURE_DIM,
)
from backend.utils.league_registry import LEAGUES

_LABELS: Dict[str, str] = {
    "expected_goals": "xG",
    "expected_assists": "xA",
    "shots": "Shots",
    "successful_dribbles": "Take-ons",
    "successful_crosses": "Crosses",
    "touches_in_opposition_box": "Pen. Area Entries",
    "successful_passes": "Total Passes",
    "pass_completion_pct": "Short Pass %",
    "accurate_long_balls": "Long Passes",
    "chances_created": "Passes Att 3rd",
    "clearances": "Def Own 3rd",
    "interceptions": "Def Mid 3rd",
    "possession_won_final_3rd": "Def Att 3rd",
}


def render():
    st.header("Shortlist Generator")
    st.caption("Find replacement candidates ranked by weighted similarity")

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
            results = fotmob_client.search_player(player_query)
        except Exception as e:
            st.error(f"Search failed: {e}")
            return

    if not results:
        st.warning("No players found.")
        return

    options = {f"{p['name']} (ID: {p['id']})": p for p in results}
    selected = st.selectbox("Select player", list(options.keys()), key="shortlist_select")
    player = options[selected]

    with st.spinner("Fetching stats..."):
        try:
            player_stats = fotmob_client.get_player_stats(player["id"])
        except Exception as e:
            st.error(f"Failed to fetch stats: {e}")
            return

    player_name = player_stats.get("name", "Unknown")
    current_per90 = player_stats.get("per90", {})
    position = player_stats.get("position", "Unknown")

    st.subheader(f"Replacing: {player_name} ({position})")

    # ── Metric weight sliders ────────────────────────────────────────────
    st.markdown("### Metric Weights")
    st.caption("Set importance of each metric (0 = ignore, 1 = maximum weight)")

    weights: Dict[str, float] = {}
    cols = st.columns(3)
    for i, metric in enumerate(CORE_METRICS):
        label = _LABELS.get(metric, metric)
        with cols[i % 3]:
            weights[metric] = st.slider(
                label, 0.0, 1.0, 0.5, 0.1, key=f"w_{metric}"
            )

    # ── Filters ──────────────────────────────────────────────────────────
    st.markdown("### Filters")
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        max_age = st.number_input("Max age", 16, 45, 30, key="f_age")
        min_minutes = st.number_input("Min minutes played", 0, 5000, 500, key="f_mins")
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

    filters = ShortlistFilters(
        max_age=max_age if max_age < 45 else None,
        min_minutes_played=min_minutes if min_minutes > 0 else None,
        leagues=None if "Any" in selected_leagues else selected_leagues,
        positions=None if "Any" in positions else positions,
        max_power_ranking=max_pr if max_pr < 100 else None,
    )

    # ── Generate shortlist ───────────────────────────────────────────────
    if not st.button("Generate Shortlist", type="primary"):
        return

    with st.spinner("Generating shortlist... This may take a moment."):
        team_id = player_stats.get("team_id")
        if not team_id:
            st.error("Cannot determine player's team.")
            return

        # Get candidates from the same team and comparable teams
        try:
            team_players = fotmob_client.get_team_players_stats(team_id)
        except Exception:
            team_players = []

        # Build candidate list from team players (in a real deployment,
        # this would search across multiple leagues/teams)
        candidates: List[Candidate] = []

        # Get current team's power ranking for context
        source_ranking = power_rankings.get_team_ranking(
            player_stats.get("team", "")
        )
        source_norm = source_ranking.normalized_score if source_ranking else 50.0
        source_league = source_ranking.league_mean_normalized if source_ranking else 50.0

        for tp in team_players:
            if tp.get("id") == player["id"]:
                continue  # Skip the player being replaced

            try:
                tp_stats = fotmob_client.get_player_stats(tp["id"])
            except Exception:
                continue

            tp_per90 = tp_stats.get("per90", {})
            if not any(tp_per90.get(m) for m in CORE_METRICS):
                continue

            predicted = {}
            for m in CORE_METRICS:
                predicted[m] = tp_per90.get(m, 0) or 0

            candidates.append(Candidate(
                player_id=tp["id"],
                name=tp_stats.get("name", tp.get("name", "Unknown")),
                team=tp_stats.get("team", ""),
                position=tp_stats.get("position", tp.get("position", "Unknown")),
                minutes_played=tp_stats.get("minutes_played"),
                predicted_per90=predicted,
                current_per90={m: (tp_per90.get(m) or 0) for m in CORE_METRICS},
            ))

    if not candidates:
        st.warning("No candidates found. Try broadening your filters or searching more leagues.")
        return

    # Score candidates
    scored = score_candidates(candidates, weights, filters)

    if not scored:
        st.warning("No candidates match your filters.")
        return

    # ── Display results ──────────────────────────────────────────────────
    st.subheader(f"Top {min(len(scored), 20)} Candidates")

    rows = []
    for c in scored[:20]:
        changes = compute_percentage_changes(c.current_per90, c.predicted_per90)
        top_changes = sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        top_str = ", ".join(
            f"{_LABELS.get(m, m)}: {v:+.1f}%" for m, v in top_changes
        )
        rows.append({
            "Rank": len(rows) + 1,
            "Player": c.name,
            "Team": c.team,
            "Position": c.position,
            "Score": f"{c.score:.3f}",
            "Top Changes": top_str,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Detailed view for selected candidate ─────────────────────────────
    if scored:
        candidate_names = [f"{c.name} ({c.team})" for c in scored[:20]]
        detail_selection = st.selectbox(
            "View detailed prediction for:", candidate_names, key="detail_candidate"
        )
        idx = candidate_names.index(detail_selection)
        detail = scored[idx]

        st.markdown(f"### {detail.name} — Detailed Prediction")
        from frontend.components import metric_bar
        changes = compute_percentage_changes(detail.current_per90, detail.predicted_per90)
        metric_bar.show(
            detail.current_per90, detail.predicted_per90, changes,
            title=f"Predicted Changes: {detail.name}",
        )
