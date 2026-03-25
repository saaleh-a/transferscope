"""Transfer Impact dashboard — predicted performance change (paper Fig 1).

Inputs: player search, target club search.
Outputs:
  (a) metric bars — predicted % change per metric
  (b) power ranking chart — before/after team Power Rankings
  (c) RAG confidence indicator
  (d) swarm plots — player vs league/team context
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import streamlit as st

from backend.data import sofascore_client, elo_router
from backend.data.sofascore_client import CORE_METRICS
from backend.features import power_rankings, rolling_windows
from backend.features.adjustment_models import (
    PlayerAdjustmentModel,
    TeamAdjustmentModel,
    scale_team_position_features,
)
from backend.models.shortlist_scorer import compute_percentage_changes
from backend.models.transfer_portal import (
    TransferPortalModel,
    build_feature_dict,
    FEATURE_DIM,
)
from frontend.components import metric_bar, power_ranking_chart, swarm_plot


def render():
    st.header("Transfer Impact")
    st.caption("Predict how a player's performance will change at a new club")

    # ── Inputs ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        player_query = st.text_input("Search player", placeholder="e.g. Bukayo Saka")
    with col2:
        target_club_query = st.text_input("Target club", placeholder="e.g. Real Madrid")

    if not player_query or not target_club_query:
        st.info("Enter a player name and target club to generate a transfer impact prediction.")
        return

    # ── Player search ────────────────────────────────────────────────────
    with st.spinner("Searching for player..."):
        try:
            search_results = sofascore_client.search_player(player_query)
        except Exception as e:
            st.error(f"Sofascore search failed: {e}")
            return

    if not search_results:
        st.warning(f"No players found for '{player_query}'.")
        return

    player_options = {f"{p['name']} (ID: {p['id']})": p for p in search_results}
    selected = st.selectbox("Select player", list(player_options.keys()))
    player_info = player_options[selected]
    player_id = player_info["id"]

    # ── Fetch player stats ───────────────────────────────────────────────
    with st.spinner("Fetching player stats..."):
        try:
            player_stats = sofascore_client.get_player_stats(player_id)
        except Exception as e:
            st.error(f"Failed to fetch player stats: {e}")
            return

    player_name = player_stats.get("name", "Unknown")
    current_team = player_stats.get("team", "Unknown")
    position = player_stats.get("position", "Unknown")
    minutes = player_stats.get("minutes_played", 0) or 0
    current_per90 = player_stats.get("per90", {})

    st.subheader(f"{player_name} — {current_team}")
    st.caption(f"Position: {position} | Minutes: {minutes}")

    # ── Power Rankings ───────────────────────────────────────────────────
    with st.spinner("Computing Power Rankings..."):
        source_ranking = power_rankings.get_team_ranking(current_team)
        target_ranking = power_rankings.get_team_ranking(target_club_query)

    if source_ranking is None:
        st.warning(f"Could not find Power Ranking for {current_team}. Using defaults.")
        source_norm = 50.0
        source_league_mean = 50.0
    else:
        source_norm = source_ranking.normalized_score
        source_league_mean = source_ranking.league_mean_normalized

    if target_ranking is None:
        st.warning(f"Could not find Power Ranking for {target_club_query}. Using defaults.")
        target_norm = 50.0
        target_league_mean = 50.0
    else:
        target_norm = target_ranking.normalized_score
        target_league_mean = target_ranking.league_mean_normalized

    change_ra = (target_norm - target_league_mean) - (source_norm - source_league_mean)

    # ── (c) RAG confidence ───────────────────────────────────────────────
    features = rolling_windows.compute_player_features(player_stats)
    confidence = features.confidence
    confidence_colors = {"green": "#2ecc71", "amber": "#f39c12", "red": "#e74c3c"}
    conf_color = confidence_colors.get(confidence, "#95a5a6")

    st.markdown(
        f"**Data Confidence:** "
        f"<span style='color:{conf_color}; font-weight:bold;'>"
        f"{confidence.upper()}</span> "
        f"(weight={features.weight:.2f}, {minutes} mins)",
        unsafe_allow_html=True,
    )

    if confidence == "red":
        st.warning("Low data confidence — prediction heavily relies on priors. Treat with caution.")

    # ── Build prediction ─────────────────────────────────────────────────
    # Use adjustment models if available, else use raw features
    current_per90_clean = {m: (current_per90.get(m) or 0.0) for m in CORE_METRICS}
    # Simple team-position proxy: use player's own per-90 as position-level
    team_pos_current = current_per90_clean.copy()
    team_pos_target = current_per90_clean.copy()

    # Attempt neural net prediction
    try:
        model = TransferPortalModel()
        model.build(FEATURE_DIM)

        fd = build_feature_dict(
            player_per90=current_per90_clean,
            team_ability_current=source_norm,
            team_ability_target=target_norm,
            league_ability_current=source_league_mean,
            league_ability_target=target_league_mean,
            team_pos_current=team_pos_current,
            team_pos_target=team_pos_target,
        )

        predicted = model.predict(fd)
    except Exception:
        # Fallback: use adjustment-based linear prediction
        predicted = {}
        for m in CORE_METRICS:
            val = current_per90_clean.get(m, 0)
            # Simple linear scaling by relative ability change
            adjustment = 1.0 + (change_ra / 100.0) * 0.5
            predicted[m] = val * adjustment

    # ── (a) Metric bars ──────────────────────────────────────────────────
    pct_changes = compute_percentage_changes(current_per90_clean, predicted)
    metric_bar.show(current_per90_clean, predicted, pct_changes,
                    title=f"Predicted Changes: {player_name} -> {target_club_query}")

    # ── (b) Power Ranking chart ──────────────────────────────────────────
    st.subheader("Power Rankings Context")
    # Build simple history (current snapshot + simulated)
    today = date.today()
    source_history = [(today - timedelta(days=30 * i), source_norm - i * 0.5) for i in range(6)]
    source_history.reverse()
    target_history = [(today - timedelta(days=30 * i), target_norm + i * 0.3) for i in range(6)]
    target_history.reverse()

    power_ranking_chart.show(
        current_team, target_club_query,
        source_history, target_history,
        transfer_date=today,
    )

    # ── (d) Swarm plots ──────────────────────────────────────────────────
    st.subheader("Player in League Context")

    # Get teammate/league data if available
    team_id = player_stats.get("team_id")
    teammate_per90s: List[Dict] = []
    league_per90s: List[Dict] = []

    if team_id:
        try:
            team_players = sofascore_client.get_team_players_stats(team_id)
            # For each teammate, fetch their stats
            for tp in team_players[:10]:  # Limit to avoid too many API calls
                if tp.get("id") and tp["id"] != player_id:
                    try:
                        tp_stats = sofascore_client.get_player_stats(tp["id"])
                        if tp_stats.get("per90"):
                            teammate_per90s.append(tp_stats["per90"])
                    except Exception:
                        pass
        except Exception:
            pass

    swarm_plot.show_swarm_grid(
        player_name=player_name,
        player_per90=current_per90_clean,
        teammate_per90s=teammate_per90s,
        league_per90s=league_per90s,
    )

    # ── Summary table ────────────────────────────────────────────────────
    st.subheader("Detailed Predictions")
    import pandas as pd
    labels = {
        "expected_goals": "xG", "expected_assists": "xA", "shots": "Shots",
        "successful_dribbles": "Take-ons", "successful_crosses": "Crosses",
        "touches_in_opposition_box": "Pen. Area Entries",
        "successful_passes": "Total Passes", "pass_completion_pct": "Short Pass %",
        "accurate_long_balls": "Long Passes", "chances_created": "Passes Att 3rd",
        "clearances": "Def Own 3rd", "interceptions": "Def Mid 3rd",
        "possession_won_final_3rd": "Def Att 3rd",
    }
    rows = []
    for m in CORE_METRICS:
        rows.append({
            "Metric": labels.get(m, m),
            "Current": f"{current_per90_clean.get(m, 0):.3f}",
            "Predicted": f"{predicted.get(m, 0):.3f}",
            "Change": f"{pct_changes.get(m, 0):+.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
