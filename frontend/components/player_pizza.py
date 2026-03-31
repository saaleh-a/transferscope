"""mplsoccer radar (pizza) chart for player per-90 profiles.

Renders a "pizza chart" (filled radar plot) using mplsoccer's PyPizza
class.  The chart shows a player's per-90 metrics as slices, color-coded
by paper model group (Shooting, Passing, Dribbling, Defending).

Designed to be rendered to a PNG buffer and displayed in Streamlit via
``st.image()`` — avoids Matplotlib's interactive backend which conflicts
with Streamlit's event loop.

Tactical Noir styled — dark background, gold/emerald/crimson accents.
"""

from __future__ import annotations

import io
import logging
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import streamlit as st

from backend.data.sofascore_client import CORE_METRICS

_log = logging.getLogger(__name__)

_MIN_SCALE_THRESHOLD = 1e-6  # Near-zero guard for normalization

# ── Metric grouping (paper Table 1) ─────────────────────────────────────────

_GROUP_COLORS = {
    "Shooting": "#F45B69",    # Coral-crimson
    "Passing": "#D4A843",     # Gold
    "Dribbling": "#2DD4A8",   # Emerald
    "Defending": "#58A6FF",   # Blue
}

_METRIC_GROUPS: Dict[str, str] = {
    "expected_goals": "Shooting",
    "shots": "Shooting",
    "expected_assists": "Passing",
    "successful_crosses": "Passing",
    "successful_passes": "Passing",
    "pass_completion_pct": "Passing",
    "accurate_long_balls": "Passing",
    "chances_created": "Passing",
    "touches_in_opposition_box": "Passing",
    "successful_dribbles": "Dribbling",
    "clearances": "Defending",
    "interceptions": "Defending",
    "possession_won_final_3rd": "Defending",
}

_LABELS: Dict[str, str] = {
    "expected_goals": "xG",
    "expected_assists": "xA",
    "shots": "Shots",
    "successful_dribbles": "Take-ons",
    "successful_crosses": "Crosses",
    "touches_in_opposition_box": "Pen. Area",
    "successful_passes": "Passes",
    "pass_completion_pct": "Pass %",
    "accurate_long_balls": "Long Balls",
    "chances_created": "Key Passes",
    "clearances": "Clearances",
    "interceptions": "Intercepts",
    "possession_won_final_3rd": "Tackles 3rd",
}


def render_pizza(
    per90: Dict[str, float],
    player_name: str = "",
    comparison_per90: Optional[Dict[str, float]] = None,
    comparison_name: str = "Reference",
    figsize: Tuple[float, float] = (8, 8),
) -> Optional[io.BytesIO]:
    """Render a pizza chart to a PNG buffer.

    Parameters
    ----------
    per90 : dict
        Player's per-90 metrics (13 core metrics).
    player_name : str
        Name displayed in the chart title.
    comparison_per90 : dict, optional
        If provided, renders a second layer showing the reference player.
    comparison_name : str
        Label for the comparison player.
    figsize : tuple
        Figure dimensions in inches.

    Returns
    -------
    io.BytesIO containing a PNG image, or None on error.
    """
    try:
        from mplsoccer import PyPizza
    except ImportError:
        _log.warning("mplsoccer not installed — pizza charts unavailable")
        return None

    # Order metrics by group for visual coherence
    ordered_metrics = [m for m in CORE_METRICS if m in per90 or m in (comparison_per90 or {})]
    if not ordered_metrics:
        return None

    params = [_LABELS.get(m, m) for m in ordered_metrics]
    values = [max(0.0, per90.get(m) or 0.0) for m in ordered_metrics]

    # Normalize to 0-100 scale for the pizza chart.
    # Use simple min-max across both players if comparison exists.
    all_vals = list(values)
    if comparison_per90:
        comp_values = [max(0.0, comparison_per90.get(m) or 0.0) for m in ordered_metrics]
        all_vals.extend(comp_values)
    else:
        comp_values = None

    max_val = max(all_vals) if all_vals else 1.0
    if max_val < _MIN_SCALE_THRESHOLD:
        max_val = 1.0

    normalized = [min(100, (v / max_val) * 100) for v in values]
    if comp_values is not None:
        comp_normalized = [min(100, (v / max_val) * 100) for v in comp_values]

    # Slice colors by group
    slice_colors = [_GROUP_COLORS.get(_METRIC_GROUPS.get(m, ""), "#D4A843")
                    for m in ordered_metrics]
    text_colors = ["#C9D1D9"] * len(ordered_metrics)

    try:
        baker = PyPizza(
            params=params,
            background_color="#0E1117",
            straight_line_color="#30363D",
            straight_line_lw=1,
            last_circle_color="#30363D",
            last_circle_lw=1,
            other_circle_color="#21262D",
            other_circle_lw=0.5,
            inner_circle_size=20,
        )

        # Build kwargs for comparison layer if provided
        compare_kwargs: Dict = {}
        if comp_values is not None and comp_normalized:
            compare_kwargs = dict(
                compare_values=comp_normalized,
                compare_colors=["#8B949E"] * len(ordered_metrics),
                compare_value_colors=["#C9D1D9"] * len(ordered_metrics),
                compare_value_bck_colors=["#484F58"] * len(ordered_metrics),
                kwargs_compare=dict(
                    edgecolor="#484F58", zorder=1, linewidth=0.5,
                ),
                kwargs_compare_values=dict(
                    color="#C9D1D9", fontsize=7,
                    fontfamily="monospace",
                    bbox=dict(edgecolor="none", facecolor="none"),
                ),
            )

        fig, ax = baker.make_pizza(
            normalized,
            figsize=figsize,
            color_blank_space="same",
            slice_colors=slice_colors,
            value_colors=text_colors,
            value_bck_colors=slice_colors,
            blank_alpha=0.15,
            kwargs_slices=dict(edgecolor="#30363D", zorder=2, linewidth=1),
            kwargs_params=dict(
                color="#C9D1D9", fontsize=10,
                fontfamily="sans-serif", zorder=5,
            ),
            kwargs_values=dict(
                color="#0E1117", fontsize=9,
                fontfamily="monospace", zorder=3,
                bbox=dict(
                    edgecolor="none", facecolor="none",
                    boxstyle="round,pad=0.2",
                ),
            ),
            **compare_kwargs,
        )

        # Title
        title_parts = [player_name] if player_name else ["Player"]
        if comparison_per90:
            title_parts.append(f"vs {comparison_name}")
        fig.text(
            0.515, 0.975, " ".join(title_parts),
            size=16, ha="center", fontfamily="sans-serif",
            color="#C9D1D9", weight="bold",
        )

        # Legend — group colors
        legend_text = "  ".join(
            f"● {group}" for group in ["Shooting", "Passing", "Dribbling", "Defending"]
        )
        fig.text(
            0.515, 0.02, legend_text,
            size=9, ha="center", fontfamily="sans-serif",
            color="#8B949E",
        )

        # Render to buffer
        buf = io.BytesIO()
        fig.savefig(
            buf, format="png", dpi=150,
            bbox_inches="tight", facecolor="#0E1117",
            edgecolor="none",
        )
        plt.close(fig)
        buf.seek(0)
        return buf

    except Exception as exc:
        _log.warning("Pizza chart render failed: %s", exc)
        plt.close("all")
        return None


def show(
    per90: Dict[str, float],
    player_name: str = "",
    comparison_per90: Optional[Dict[str, float]] = None,
    comparison_name: str = "Reference",
) -> None:
    """Render a pizza chart in Streamlit."""
    buf = render_pizza(per90, player_name, comparison_per90, comparison_name)
    if buf is not None:
        st.image(buf, use_container_width=True)
    else:
        st.info("Pizza chart unavailable (mplsoccer not installed or no data).")
