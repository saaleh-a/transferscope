"""Hot or Not — quick rumour validator (paper Section 5).

Input: player name + target club (with Sofascore team search),
       optional season selector.
Output: Hot / Tepid / Not verdict with top 3 predicted metric changes.
"""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

from backend.data import sofascore_client
from backend.data.sofascore_client import CORE_METRICS, OFFENSIVE_METRICS, DEFENSIVE_METRICS
from backend.features import power_rankings, rolling_windows
from backend.features.adjustment_models import paper_heuristic_predict
from backend.models.shortlist_scorer import compute_percentage_changes
from backend.models.transfer_portal import (
    TransferPortalModel,
    build_feature_dict,
)
from frontend.theme import (
    section_header, confidence_badge, verdict_display, player_info_card, COLORS,
)

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


def _verdict(avg_change: float) -> tuple:
    """Return (verdict, color, emoji) based on average % change."""
    if avg_change > 5:
        return "HOT", "#2ecc71", "fire"
    elif avg_change > -5:
        return "TEPID", "#f39c12", "thinking_face"
    else:
        return "NOT", "#e74c3c", "x"


def render():
    st.header("Hot or Not")
    st.caption("Quick rumour validator — is this transfer a good move?")

    # ── Inputs ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        player_query = st.text_input("Player", placeholder="e.g. Victor Osimhen", key="hon_player")
    with col2:
        target_club_query = st.text_input("Target club", placeholder="e.g. Arsenal", key="hon_club")

    if not player_query or not target_club_query:
        st.info("Enter a player and target club to validate the rumour.")
        return

    # ── Player search + dropdown ─────────────────────────────────────────
    with st.spinner("Searching for player..."):
        try:
            search_results = sofascore_client.search_player(player_query)
        except Exception as e:
            st.error(f"Search failed: {e}")
            return

    if not search_results:
        st.warning(f"No player found for '{player_query}'.")
        return

    player_options = {
        f"{p['name']}"
        + (f" · Age {p['age']}" if p.get("age") else "")
        + (f" · {p['nationality']}" if p.get("nationality") else "")
        + (f" · {p['team_name']}" if p.get("team_name") else "")
        : p for p in search_results
    }
    selected_player = st.selectbox(
        "Select player", list(player_options.keys()), key="hon_player_select"
    )
    player_info = player_options[selected_player]

    # ── Target club autocomplete ─────────────────────────────────────────
    try:
        club_results = sofascore_client.search_team(target_club_query)
    except Exception:
        club_results = []

    target_club = target_club_query
    target_team_id: Optional[int] = None
    if club_results:
        club_options = {
            f"{c['name']}" + (f" ({c.get('country', '')})" if c.get("country") else ""): c
            for c in club_results
        }
        selected_club = st.selectbox(
            "Select target club", list(club_options.keys()), key="hon_club_select"
        )
        target_club = club_options[selected_club]["name"]
        target_team_id = club_options[selected_club].get("id")

    # ── Season selector ──────────────────────────────────────────────────
    tournament_id = player_info.get("tournament_id")
    selected_season_id: Optional[int] = None

    if tournament_id:
        seasons = sofascore_client.get_season_list(tournament_id)
        if seasons:
            season_labels = {s["name"]: s["id"] for s in seasons}
            chosen_season = st.selectbox(
                "Season",
                ["Current"] + list(season_labels.keys()),
                key="hon_season",
            )
            if chosen_season != "Current":
                selected_season_id = season_labels[chosen_season]

    if not st.button("Validate Rumour", type="primary"):
        return

    # ── Fetch stats ──────────────────────────────────────────────────────
    with st.spinner("Analysing transfer rumour..."):
        try:
            if selected_season_id and tournament_id:
                player_stats = sofascore_client.get_player_stats_for_season(
                    player_info["id"], tournament_id, selected_season_id
                )
            else:
                player_stats = sofascore_client.get_player_stats(player_info["id"])
        except Exception as e:
            st.error(f"Failed to fetch stats: {e}")
            return

    player_name = player_stats.get("name", "Unknown")
    current_team = player_stats.get("team", "Unknown")
    position = player_stats.get("position", "Unknown")
    minutes = player_stats.get("minutes_played", 0)
    if minutes is None:
        minutes = 0
    current_per90 = player_stats.get("per90", {})
    current_per90_clean = {m: (current_per90.get(m) if current_per90.get(m) is not None else 0.0) for m in CORE_METRICS}

    # ── Player info card ─────────────────────────────────────────────────
    player_info_card(player_name, current_team, position, minutes,
                     rating=player_stats.get("rating"))

    # Power Rankings
    source_ranking = power_rankings.get_team_ranking(current_team)
    target_ranking = power_rankings.get_team_ranking(target_club)

    source_norm = source_ranking.normalized_score if source_ranking else 50.0
    source_league = source_ranking.league_mean_normalized if source_ranking else 50.0
    target_norm = target_ranking.normalized_score if target_ranking else 50.0
    target_league = target_ranking.league_mean_normalized if target_ranking else 50.0

    change_ra = (target_norm - target_league) - (source_norm - source_league)

    # Confidence
    features = rolling_windows.compute_player_features(player_stats)

    # Fetch team-position averages for tactical style context (paper Sec 2.3)
    source_pos_avg: dict = {}
    target_pos_avg: dict = {}
    team_id = player_stats.get("team_id")

    if team_id:
        try:
            source_pos_avg, _ = sofascore_client.get_team_position_averages(
                team_id, position
            )
        except Exception:
            pass
    if target_team_id:
        try:
            target_pos_avg, _ = sofascore_client.get_team_position_averages(
                target_team_id, position
            )
        except Exception:
            pass

    if not source_pos_avg:
        source_pos_avg = current_per90_clean.copy()
    if not target_pos_avg:
        target_pos_avg = current_per90_clean.copy()

    # Prediction — only use TF model if trained weights exist
    predicted = {}
    try:
        model = TransferPortalModel()
        model.load()
        if model.fitted:
            fd = build_feature_dict(
                player_per90=current_per90_clean,
                team_ability_current=source_norm,
                team_ability_target=target_norm,
                league_ability_current=source_league,
                league_ability_target=target_league,
                team_pos_current=source_pos_avg,
                team_pos_target=target_pos_avg,
            )
            predicted = model.predict(fd)
    except Exception:
        predicted = {}

    # Paper-aligned heuristic: per-metric predictions using team style + ability
    if not predicted:
        predicted = paper_heuristic_predict(
            player_per90=current_per90_clean,
            source_pos_avg=source_pos_avg,
            target_pos_avg=target_pos_avg,
            change_relative_ability=change_ra,
            player_rating=player_stats.get("rating"),
            source_league_mean=source_league,
            target_league_mean=target_league,
        )

    pct_changes = compute_percentage_changes(current_per90_clean, predicted)

    # ── Verdict ──────────────────────────────────────────────────────────
    valid_changes = [v for v in pct_changes.values() if v != 0]
    avg_change = sum(valid_changes) / len(valid_changes) if valid_changes else 0
    verdict, color, emoji = _verdict(avg_change)

    st.markdown("---")

    # Big verdict display
    verdict_display(verdict, player_name, current_team, target_club)

    # Confidence badge
    confidence_badge(features.confidence, features.weight, minutes)

    # ── Transfer History context ─────────────────────────────────────────
    try:
        transfer_history = sofascore_client.get_player_transfer_history(player_info["id"])
        if transfer_history:
            section_header("Transfer History", "Previous career moves")
            import pandas as pd
            th_rows = []
            for t in transfer_history[:10]:
                th_rows.append({
                    "Date": t.get("transfer_date", "N/A"),
                    "From": t.get("from_team", {}).get("name", "N/A"),
                    "To": t.get("to_team", {}).get("name", "N/A"),
                    "Type": t.get("type", "N/A"),
                })
            st.dataframe(pd.DataFrame(th_rows), use_container_width=True, hide_index=True)
    except Exception:
        pass

    # ── Top 3 changes ────────────────────────────────────────────────────
    section_header("Key Predicted Changes", "Top metric movements")

    sorted_changes = sorted(pct_changes.items(), key=lambda x: abs(x[1]), reverse=True)
    top3 = sorted_changes[:3]

    cols = st.columns(3)
    for i, (metric, change) in enumerate(top3):
        with cols[i]:
            label = _LABELS.get(metric, metric)
            current_val = current_per90_clean.get(metric, 0)
            pred_val = predicted.get(metric, 0)
            delta_color = "normal" if change >= 0 else "inverse"
            st.metric(
                label=label,
                value=f"{pred_val:.2f}",
                delta=f"{change:+.1f}%",
                delta_color=delta_color,
            )

    # ── Quick summary ────────────────────────────────────────────────────
    st.markdown("---")
    improving = sum(1 for v in pct_changes.values() if v > 2)
    declining = sum(1 for v in pct_changes.values() if v < -2)
    stable = len(CORE_METRICS) - improving - declining

    st.markdown(
        f"**Summary:** {improving} metrics improving, {stable} stable, "
        f"{declining} declining across 13 core metrics."
    )

    if source_ranking and target_ranking:
        pr_diff = target_norm - source_norm
        direction = "stronger" if pr_diff > 0 else "weaker"
        st.markdown(
            f"**Club context:** {target_club} is rated "
            f"{abs(pr_diff):.1f} points {direction} than {current_team} "
            f"in global Power Rankings."
        )
