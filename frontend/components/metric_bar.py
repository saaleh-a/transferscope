"""Plotly horizontal bar chart: predicted % change per metric.

Tactical Noir styled — dark background, amber/green/crimson accents.
"""

from __future__ import annotations

from typing import Dict, Optional

import plotly.graph_objects as go
import streamlit as st

from backend.data.sofascore_client import CORE_METRICS
from frontend.theme import COLORS, PLOTLY_LAYOUT

# Human-readable labels
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


def render_metric_bars(
    current_per90: Dict[str, float],
    predicted_per90: Dict[str, float],
    pct_changes: Dict[str, float],
    title: str = "Predicted Metric Changes",
) -> go.Figure:
    """Create horizontal bar chart comparing current vs predicted per-90."""
    metrics = [m for m in CORE_METRICS if m in current_per90 or m in predicted_per90]
    labels = [_LABELS.get(m, m) for m in metrics]
    current_vals = [current_per90.get(m, 0) or 0 for m in metrics]
    predicted_vals = [predicted_per90.get(m, 0) or 0 for m in metrics]
    changes = [pct_changes.get(m, 0) for m in metrics]

    colors = [COLORS["accent_green"] if c >= 0 else COLORS["accent_crimson"] for c in changes]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=labels,
        x=current_vals,
        name="Current",
        orientation="h",
        marker_color=COLORS["text_muted"],
        opacity=0.35,
        marker_line_width=0,
    ))

    fig.add_trace(go.Bar(
        y=labels,
        x=predicted_vals,
        name="Predicted",
        orientation="h",
        marker_color=colors,
        opacity=0.9,
        text=[f"{c:+.1f}%" for c in changes],
        textposition="outside",
        textfont=dict(
            family="'JetBrains Mono', monospace",
            size=10,
            color=[COLORS["accent_green"] if c >= 0 else COLORS["accent_crimson"] for c in changes],
        ),
        marker_line_width=0,
    ))

    layout = dict(PLOTLY_LAYOUT)
    layout.update(
        title=dict(text=title, **PLOTLY_LAYOUT["title"]),
        barmode="group",
        xaxis_title="Per 90",
        yaxis=dict(autorange="reversed", **PLOTLY_LAYOUT["yaxis"]),
        height=max(420, len(metrics) * 42),
        margin=dict(l=120, r=80, t=50, b=30),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text_secondary"], size=11),
        ),
        bargap=0.25,
        bargroupgap=0.08,
    )
    fig.update_layout(**layout)

    return fig


def show(
    current_per90: Dict[str, float],
    predicted_per90: Dict[str, float],
    pct_changes: Dict[str, float],
    title: str = "Predicted Metric Changes",
) -> None:
    """Render the metric bar chart in Streamlit."""
    fig = render_metric_bars(current_per90, predicted_per90, pct_changes, title)
    st.plotly_chart(fig, use_container_width=True)
