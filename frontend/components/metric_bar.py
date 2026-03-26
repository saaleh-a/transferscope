"""Plotly diverging bar chart: predicted % change per metric.

Redesigned as a butterfly/diverging chart centered on zero — the % change
is the primary visual element, not the raw per-90 values. Negative changes
fan left (crimson), positive fan right (emerald). A ghosted tooltip shows
the actual per-90 values for reference.

Tactical Noir styled — deep charcoal base, high-contrast color coding.
Paper reference: Figure 15 (Doku comparison strips).
"""

from __future__ import annotations

from typing import Dict, Optional

import plotly.graph_objects as go
import streamlit as st

from backend.data.sofascore_client import CORE_METRICS
from frontend.theme import COLORS, PLOTLY_LAYOUT

# Human-readable labels — grouped by paper category
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

# Paper Table 1 group markers for visual grouping
_GROUP_MARKERS: Dict[str, str] = {
    "expected_goals": "⚡",
    "shots": "⚡",
    "expected_assists": "◈",
    "successful_crosses": "◈",
    "successful_passes": "◈",
    "pass_completion_pct": "◈",
    "accurate_long_balls": "◈",
    "chances_created": "◈",
    "touches_in_opposition_box": "◈",
    "successful_dribbles": "◎",
    "clearances": "◆",
    "interceptions": "◆",
    "possession_won_final_3rd": "◆",
}

# Colors with intensity scaling
_CLR_POSITIVE = "#2DD4A8"       # Emerald — distinctive, not generic green
_CLR_NEGATIVE = "#F45B69"       # Coral-crimson — warm, urgent
_CLR_NEUTRAL = "#485363"        # Muted slate
_CLR_GHOST = "rgba(200,210,220,0.12)"


def _intensity_color(value: float, positive_color: str, negative_color: str) -> str:
    """Return color with opacity proportional to magnitude."""
    opacity = min(1.0, abs(value) / 30.0 * 0.6 + 0.4)
    if value >= 0:
        r, g, b = int(positive_color[1:3], 16), int(positive_color[3:5], 16), int(positive_color[5:7], 16)
    else:
        r, g, b = int(negative_color[1:3], 16), int(negative_color[3:5], 16), int(negative_color[5:7], 16)
    return f"rgba({r},{g},{b},{opacity:.2f})"


def render_metric_bars(
    current_per90: Dict[str, float],
    predicted_per90: Dict[str, float],
    pct_changes: Dict[str, float],
    title: str = "Predicted Metric Changes",
) -> go.Figure:
    """Create diverging bar chart centered on zero (% change is primary)."""
    metrics = [m for m in CORE_METRICS if m in current_per90 or m in predicted_per90]
    # Reverse so first metric appears at top
    metrics = list(reversed(metrics))

    labels = [f"{_GROUP_MARKERS.get(m, '')} {_LABELS.get(m, m)}" for m in metrics]
    changes = [pct_changes.get(m, 0) for m in metrics]
    current_vals = [cv if (cv := current_per90.get(m)) is not None else 0 for m in metrics]
    predicted_vals = [pv if (pv := predicted_per90.get(m)) is not None else 0 for m in metrics]

    bar_colors = [
        _intensity_color(c, _CLR_POSITIVE, _CLR_NEGATIVE) for c in changes
    ]
    border_colors = [
        _CLR_POSITIVE if c >= 0 else _CLR_NEGATIVE for c in changes
    ]

    fig = go.Figure()

    # Main diverging bars — % change
    fig.add_trace(go.Bar(
        y=labels,
        x=changes,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color=border_colors, width=1.5),
        ),
        text=[
            f"<b>{c:+.1f}%</b>  ·  {cv:.2f} → {pv:.2f}"
            for c, cv, pv in zip(changes, current_vals, predicted_vals)
        ],
        textposition=["outside" if abs(c) > 3 else "auto" for c in changes],
        textfont=dict(
            family="'JetBrains Mono', monospace",
            size=11,
            color=[_CLR_POSITIVE if c >= 0 else _CLR_NEGATIVE for c in changes],
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Change: %{x:+.1f}%<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    # Zero line emphasis
    fig.add_vline(
        x=0, line_color=COLORS["text_muted"],
        line_width=1.5, line_dash="solid",
    )

    # Group separator annotations
    fig.add_annotation(
        x=0, y=1.02, xref="paper", yref="paper",
        text="<b>⚡ Shooting   ◈ Passing   ◎ Dribbling   ◆ Defending</b>",
        showarrow=False,
        font=dict(
            family="'DM Sans', sans-serif", size=10,
            color=COLORS["text_muted"],
        ),
        xanchor="left",
    )

    max_abs = max(abs(c) for c in changes) if changes else 10
    x_range = max(max_abs * 1.4, 8)  # At least ±8% visible

    layout = dict(PLOTLY_LAYOUT)
    layout.update(
        title=dict(
            text=title,
            font=dict(
                family="'Outfit', sans-serif",
                size=16,
                color=COLORS["text_primary"],
            ),
            x=0, xanchor="left",
        ),
        xaxis=dict(
            range=[-x_range, x_range],
            title="% Change",
            zeroline=True,
            zerolinecolor=COLORS["border"],
            zerolinewidth=2,
            gridcolor="rgba(48,54,61,0.4)",
            tickfont=dict(color=COLORS["text_secondary"], size=10),
            title_font=dict(color=COLORS["text_muted"], size=10),
            ticksuffix="%",
        ),
        yaxis=dict(
            tickfont=dict(
                family="'DM Sans', sans-serif",
                color=COLORS["text_primary"],
                size=12,
            ),
            gridcolor="rgba(0,0,0,0)",
        ),
        height=max(560, len(metrics) * 46),
        margin=dict(l=160, r=140, t=60, b=30),
        bargap=0.22,
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
