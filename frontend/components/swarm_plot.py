"""Plotly strip plot: player vs league/team context.

Tactical Noir styled — dark background, amber player marker,
teal teammates, muted league scatter.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from frontend.theme import COLORS, PLOTLY_LAYOUT


def render_swarm_plot(
    metric_name: str,
    metric_label: str,
    player_value: float,
    teammate_values: List[float],
    league_values: List[float],
    player_name: str = "Target Player",
    percentile: Optional[float] = None,
) -> go.Figure:
    """Create a strip plot showing player in context of team and league."""
    fig = go.Figure()

    # League (muted)
    if league_values:
        fig.add_trace(go.Box(
            x=league_values,
            name="League",
            marker_color="#30363D",
            boxpoints="all",
            jitter=0.5,
            pointpos=0,
            marker_size=3,
            marker_opacity=0.4,
            line_width=0,
            fillcolor="rgba(0,0,0,0)",
        ))

    # Teammates (teal)
    if teammate_values:
        fig.add_trace(go.Box(
            x=teammate_values,
            name="Teammates",
            marker_color=COLORS["accent_teal"],
            boxpoints="all",
            jitter=0.5,
            pointpos=0,
            marker_size=5,
            marker_opacity=0.7,
            line_width=0,
            fillcolor="rgba(0,0,0,0)",
        ))

    # Target player (gold diamond)
    fig.add_trace(go.Scatter(
        x=[player_value],
        y=["League" if league_values else "Teammates"],
        mode="markers",
        name=player_name,
        marker=dict(
            color=COLORS["accent_gold"],
            size=14,
            symbol="diamond",
            line=dict(width=1.5, color=COLORS["accent_amber"]),
        ),
    ))

    title = f"<b>{metric_label}</b>"
    if percentile is not None:
        pct_color = COLORS["accent_green"] if percentile >= 75 else (
            COLORS["accent_amber"] if percentile >= 50 else COLORS["accent_crimson"]
        )
        title += (
            f'  <span style="color:{pct_color}; font-size:0.85em;">'
            f'{percentile:.0f}th</span>'
        )

    layout = dict(PLOTLY_LAYOUT)
    layout.update(
        title=dict(text=title, **PLOTLY_LAYOUT["title"]),
        xaxis_title="Per 90",
        showlegend=False,
        height=190,
        margin=dict(l=10, r=10, t=45, b=25),
    )
    fig.update_layout(**layout)

    return fig


def show_swarm_grid(
    player_name: str,
    player_per90: Dict[str, float],
    teammate_per90s: List[Dict[str, float]],
    league_per90s: List[Dict[str, float]],
    metrics: Optional[List[str]] = None,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Render a grid of swarm plots in Streamlit (2 columns)."""
    from backend.data.sofascore_client import CORE_METRICS

    if metrics is None:
        metrics = [m for m in CORE_METRICS if player_per90.get(m) is not None]
    if labels is None:
        labels = {}

    default_labels = {
        "expected_goals": "xG", "expected_assists": "xA", "shots": "Shots",
        "successful_dribbles": "Take-ons", "successful_crosses": "Crosses",
        "touches_in_opposition_box": "Pen. Area Entries",
        "successful_passes": "Total Passes", "pass_completion_pct": "Short Pass %",
        "accurate_long_balls": "Long Passes", "chances_created": "Passes Att 3rd",
        "clearances": "Def Own 3rd", "interceptions": "Def Mid 3rd",
        "possession_won_final_3rd": "Def Att 3rd",
    }

    cols = st.columns(2)
    for i, metric in enumerate(metrics):
        player_val = pv if (pv := player_per90.get(metric)) is not None else 0
        tm_vals = [tv if (tv := d.get(metric)) is not None else 0 for d in teammate_per90s if d.get(metric) is not None]
        lg_vals = [lv if (lv := d.get(metric)) is not None else 0 for d in league_per90s if d.get(metric) is not None]

        # Compute percentile
        all_vals = lg_vals + tm_vals + [player_val]
        if all_vals:
            pct = float(np.sum(np.array(all_vals) <= player_val) / len(all_vals) * 100)
        else:
            pct = None

        label = labels.get(metric, default_labels.get(metric, metric))
        fig = render_swarm_plot(
            metric, label, player_val, tm_vals, lg_vals,
            player_name=player_name, percentile=pct,
        )
        with cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)
