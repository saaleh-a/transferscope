"""Plotly line chart: before/after team Power Rankings.

Tactical Noir styled — dark background, amber source line, teal target line.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st

from frontend.theme import COLORS, PLOTLY_LAYOUT


def render_power_ranking_chart(
    source_club: str,
    target_club: str,
    source_history: List[Tuple[date, float]],
    target_history: List[Tuple[date, float]],
    transfer_date: Optional[date] = None,
    title: str = "Team Power Rankings",
) -> go.Figure:
    """Create a dual-line Power Rankings timeline."""
    fig = go.Figure()

    if source_history:
        dates_s, vals_s = zip(*source_history)
        fig.add_trace(go.Scatter(
            x=list(dates_s),
            y=list(vals_s),
            name=source_club,
            mode="lines+markers",
            line=dict(color=COLORS["accent_blue"], width=2.5),
            marker=dict(size=5, color=COLORS["accent_blue"]),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.06)",
        ))

    if target_history:
        dates_t, vals_t = zip(*target_history)
        fig.add_trace(go.Scatter(
            x=list(dates_t),
            y=list(vals_t),
            name=target_club,
            mode="lines+markers",
            line=dict(color=COLORS["accent_teal"], width=2.5),
            marker=dict(size=5, color=COLORS["accent_teal"]),
            fill="tozeroy",
            fillcolor="rgba(57,210,192,0.06)",
        ))

    if transfer_date is not None:
        fig.add_vline(
            x=transfer_date.isoformat(),
            line_dash="dot",
            line_color=COLORS["accent_gold"],
            line_width=1.5,
        )
        fig.add_annotation(
            x=transfer_date.isoformat(),
            y=1.05,
            yref="paper",
            text="TRANSFER",
            showarrow=False,
            font=dict(
                family="'Outfit', sans-serif",
                size=10,
                color=COLORS["accent_gold"],
            ),
        )

    layout = dict(PLOTLY_LAYOUT)
    layout.update(
        title=dict(text=title, **PLOTLY_LAYOUT["title"]),
        yaxis=dict(
            range=[0, 100],
            **PLOTLY_LAYOUT["yaxis"],
        ),
        xaxis_title="",
        yaxis_title="Power Ranking",
        height=380,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text_secondary"], size=11),
        ),
    )
    fig.update_layout(**layout)

    return fig


def show(
    source_club: str,
    target_club: str,
    source_history: List[Tuple[date, float]],
    target_history: List[Tuple[date, float]],
    transfer_date: Optional[date] = None,
) -> None:
    """Render the Power Rankings chart in Streamlit."""
    fig = render_power_ranking_chart(
        source_club, target_club,
        source_history, target_history,
        transfer_date,
    )
    st.plotly_chart(fig, use_container_width=True)
