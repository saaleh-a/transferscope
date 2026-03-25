"""Plotly line chart: before/after team Power Rankings.

Both clubs on same axis with transfer date marked.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st


def render_power_ranking_chart(
    source_club: str,
    target_club: str,
    source_history: List[Tuple[date, float]],
    target_history: List[Tuple[date, float]],
    transfer_date: Optional[date] = None,
    title: str = "Team Power Rankings",
) -> go.Figure:
    """Create a dual-line Power Rankings timeline.

    Parameters
    ----------
    source_club, target_club : str
        Club names.
    source_history, target_history : list[(date, normalized_score)]
        Historical normalized 0-100 scores over time.
    transfer_date : date, optional
        If given, a vertical line is drawn at this date.
    title : str
        Chart title.

    Returns
    -------
    plotly Figure.
    """
    fig = go.Figure()

    if source_history:
        dates_s, vals_s = zip(*source_history)
        fig.add_trace(go.Scatter(
            x=list(dates_s),
            y=list(vals_s),
            name=source_club,
            mode="lines+markers",
            line=dict(color="#3498db", width=2),
            marker=dict(size=4),
        ))

    if target_history:
        dates_t, vals_t = zip(*target_history)
        fig.add_trace(go.Scatter(
            x=list(dates_t),
            y=list(vals_t),
            name=target_club,
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2),
            marker=dict(size=4),
        ))

    if transfer_date is not None:
        fig.add_vline(
            x=transfer_date.isoformat(),
            line_dash="dash",
            line_color="#2c3e50",
            annotation_text="Transfer",
            annotation_position="top",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Power Ranking (0-100)",
        yaxis=dict(range=[0, 100]),
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

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
