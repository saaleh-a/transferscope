"""Plotly strip plot: player vs league/team context.

Target player in red. Teammates in orange. Rest of league in grey.
League percentile shown as header.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_swarm_plot(
    metric_name: str,
    metric_label: str,
    player_value: float,
    teammate_values: List[float],
    league_values: List[float],
    player_name: str = "Target Player",
    percentile: Optional[float] = None,
) -> go.Figure:
    """Create a strip plot showing player in context of team and league.

    Parameters
    ----------
    metric_name : str
        Internal metric key.
    metric_label : str
        Human-readable label for display.
    player_value : float
        The target player's per-90 value.
    teammate_values : list[float]
        Per-90 values for teammates (same team).
    league_values : list[float]
        Per-90 values for rest of league (excluding teammates).
    player_name : str
        Name to show in legend.
    percentile : float, optional
        Player's league percentile for this metric.

    Returns
    -------
    plotly Figure.
    """
    fig = go.Figure()

    # League (grey)
    if league_values:
        fig.add_trace(go.Box(
            x=league_values,
            name="League",
            marker_color="#bdc3c7",
            boxpoints="all",
            jitter=0.5,
            pointpos=0,
            marker_size=4,
            line_width=0,
            fillcolor="rgba(0,0,0,0)",
        ))

    # Teammates (orange)
    if teammate_values:
        fig.add_trace(go.Box(
            x=teammate_values,
            name="Teammates",
            marker_color="#f39c12",
            boxpoints="all",
            jitter=0.5,
            pointpos=0,
            marker_size=6,
            line_width=0,
            fillcolor="rgba(0,0,0,0)",
        ))

    # Target player (red)
    fig.add_trace(go.Scatter(
        x=[player_value],
        y=["League" if league_values else "Teammates"],
        mode="markers",
        name=player_name,
        marker=dict(color="#e74c3c", size=14, symbol="diamond"),
    ))

    title = f"{metric_label}"
    if percentile is not None:
        title += f" — {percentile:.0f}th percentile"

    fig.update_layout(
        title=title,
        xaxis_title="Per 90",
        showlegend=True,
        height=200,
        margin=dict(l=10, r=10, t=40, b=30),
        template="plotly_white",
    )

    return fig


def show_swarm_grid(
    player_name: str,
    player_per90: Dict[str, float],
    teammate_per90s: List[Dict[str, float]],
    league_per90s: List[Dict[str, float]],
    metrics: Optional[List[str]] = None,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    """Render a grid of swarm plots in Streamlit (2 columns).

    Parameters
    ----------
    player_name : str
    player_per90 : dict of per-90 values for the player.
    teammate_per90s : list of per-90 dicts for teammates.
    league_per90s : list of per-90 dicts for league players.
    metrics : list of metric keys to show (defaults to all available).
    labels : dict mapping metric key to display label.
    """
    from backend.data.fotmob_client import CORE_METRICS

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
        player_val = player_per90.get(metric, 0) or 0
        tm_vals = [d.get(metric, 0) or 0 for d in teammate_per90s if d.get(metric) is not None]
        lg_vals = [d.get(metric, 0) or 0 for d in league_per90s if d.get(metric) is not None]

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
