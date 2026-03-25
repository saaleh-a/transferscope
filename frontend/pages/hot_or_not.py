"""Hot or Not — quick rumour validator (paper Section 5).

Input: player name + target club.
Output: Hot / Tepid / Not verdict with top 3 predicted metric changes.
"""

from __future__ import annotations

from typing import Dict

import streamlit as st

from backend.data import fotmob_client
from backend.data.fotmob_client import CORE_METRICS
from backend.features import power_rankings, rolling_windows
from backend.models.shortlist_scorer import compute_percentage_changes
from backend.models.transfer_portal import (
    TransferPortalModel,
    build_feature_dict,
    FEATURE_DIM,
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
        target_club = st.text_input("Target club", placeholder="e.g. Arsenal", key="hon_club")

    if not player_query or not target_club:
        st.info("Enter a player and target club to validate the rumour.")
        return

    if not st.button("Validate Rumour", type="primary"):
        return

    # ── Search & fetch ───────────────────────────────────────────────────
    with st.spinner("Analysing transfer rumour..."):
        try:
            results = fotmob_client.search_player(player_query)
        except Exception as e:
            st.error(f"Search failed: {e}")
            return

        if not results:
            st.warning(f"No player found for '{player_query}'.")
            return

        # Auto-select best match
        player_info = results[0]
        try:
            player_stats = fotmob_client.get_player_stats(player_info["id"])
        except Exception as e:
            st.error(f"Failed to fetch stats: {e}")
            return

    player_name = player_stats.get("name", "Unknown")
    current_team = player_stats.get("team", "Unknown")
    position = player_stats.get("position", "Unknown")
    minutes = player_stats.get("minutes_played", 0) or 0
    current_per90 = player_stats.get("per90", {})
    current_per90_clean = {m: (current_per90.get(m) or 0.0) for m in CORE_METRICS}

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

    # Prediction
    try:
        model = TransferPortalModel()
        model.build(FEATURE_DIM)
        fd = build_feature_dict(
            player_per90=current_per90_clean,
            team_ability_current=source_norm,
            team_ability_target=target_norm,
            league_ability_current=source_league,
            league_ability_target=target_league,
            team_pos_current=current_per90_clean,
            team_pos_target=current_per90_clean,
        )
        predicted = model.predict(fd)
    except Exception:
        predicted = {}
        for m in CORE_METRICS:
            val = current_per90_clean.get(m, 0)
            adjustment = 1.0 + (change_ra / 100.0) * 0.5
            predicted[m] = val * adjustment

    pct_changes = compute_percentage_changes(current_per90_clean, predicted)

    # ── Verdict ──────────────────────────────────────────────────────────
    valid_changes = [v for v in pct_changes.values() if v != 0]
    avg_change = sum(valid_changes) / len(valid_changes) if valid_changes else 0
    verdict, color, emoji = _verdict(avg_change)

    st.markdown("---")

    # Big verdict display
    st.markdown(
        f"<div style='text-align:center; padding:20px;'>"
        f"<h1 style='color:{color}; font-size:4em; margin:0;'>{verdict}</h1>"
        f"<p style='font-size:1.3em;'>"
        f"{player_name} ({current_team}) → {target_club}"
        f"</p></div>",
        unsafe_allow_html=True,
    )

    # Confidence badge
    conf_colors = {"green": "#2ecc71", "amber": "#f39c12", "red": "#e74c3c"}
    conf_color = conf_colors.get(features.confidence, "#95a5a6")
    st.markdown(
        f"<p style='text-align:center;'>Data Confidence: "
        f"<span style='color:{conf_color}; font-weight:bold;'>"
        f"{features.confidence.upper()}</span> ({minutes} mins)</p>",
        unsafe_allow_html=True,
    )

    # ── Top 3 changes ────────────────────────────────────────────────────
    st.markdown("### Key Predicted Changes")

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
