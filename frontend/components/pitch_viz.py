"""mplsoccer pitch visualizations for TransferScope.

Renders shot maps, pass networks, and touch heatmaps on football pitches
using mplsoccer.  Each chart is rendered to a PNG buffer and displayed in
Streamlit via ``st.image()`` — avoids Matplotlib's interactive backend
which conflicts with Streamlit's event loop.

Designed for the Transfer Impact and Shortlist pages.  All coordinates use
StatsBomb convention (120×80).  Tactical Noir styled — dark background,
gold/emerald/crimson accents.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

_log = logging.getLogger(__name__)

# ── Tactical Noir palette ────────────────────────────────────────────────────

_BG = "#0E1117"
_LINE = "#30363D"
_TEXT = "#C9D1D9"
_MUTED = "#8B949E"
_GOLD = "#D4A843"
_EMERALD = "#2DD4A8"
_CRIMSON = "#F45B69"
_BLUE = "#58A6FF"

# ── Outcome → colour mappings ────────────────────────────────────────────────

_SHOT_COLORS: Dict[str, str] = {
    "Goal": _EMERALD,
    "Saved": _GOLD,
}
_SHOT_DEFAULT_COLOR = _CRIMSON

_PASS_COLORS: Dict[str, str] = {
    "Complete": _GOLD,
}
_PASS_DEFAULT_COLOR = _CRIMSON

# Custom colormap for heatmap: dark background → gold
_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "tactical_heat",
    [_BG, "#3D2E10", "#7A5A1E", _GOLD],
    N=256,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_PASSES_DISPLAY = 200
_MIN_HEATMAP_LOCATIONS = 10
_PROGRESSIVE_THRESHOLD = 10.0  # metres toward goal for a "progressive" pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_buf(fig: plt.Figure) -> io.BytesIO:
    """Save *fig* to a PNG buffer and close it."""
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor=_BG,
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return buf


def _is_progressive(start_x: float, end_x: float) -> bool:
    """A pass is progressive if it moves ≥ threshold metres toward goal."""
    return (end_x - start_x) >= _PROGRESSIVE_THRESHOLD


# ═════════════════════════════════════════════════════════════════════════════
# Shot Map
# ═════════════════════════════════════════════════════════════════════════════

def render_shot_map(
    shots: List[Dict[str, float]],
    player_name: str = "Player",
    figsize: Tuple[int, int] = (10, 7),
) -> Optional[io.BytesIO]:
    """Render a shot map on a half-pitch.

    Parameters
    ----------
    shots : list[dict]
        Each dict must contain ``x``, ``y``, ``xg`` (expected goals) and
        ``outcome`` (``'Goal'``, ``'Saved'``, or other).  Coordinates in
        StatsBomb space (120×80).
    player_name : str
        Displayed in the chart title.
    figsize : tuple
        Figure dimensions in inches.

    Returns
    -------
    io.BytesIO containing a PNG image, or *None* on error / empty data.
    """
    try:
        from mplsoccer import VerticalPitch
    except ImportError:
        _log.warning("mplsoccer not installed — shot map unavailable")
        return None

    if not shots:
        return None

    xs = [s["x"] for s in shots]
    ys = [s["y"] for s in shots]
    xgs = [s.get("xg", 0.05) for s in shots]
    outcomes = [s.get("outcome", "") for s in shots]
    colors = [_SHOT_COLORS.get(o, _SHOT_DEFAULT_COLOR) for o in outcomes]
    sizes = [max(50, xg * 800) for xg in xgs]

    try:
        pitch = VerticalPitch(
            half=True,
            pitch_color=_BG,
            line_color=_LINE,
            linewidth=1,
        )
        fig, ax = pitch.draw(figsize=figsize)
        fig.patch.set_facecolor(_BG)

        pitch.scatter(
            xs, ys, s=sizes, c=colors,
            edgecolors=_LINE, linewidth=0.5,
            alpha=0.85, zorder=2, ax=ax,
        )

        # Title
        fig.text(
            0.515, 0.96, f"{player_name} — Shot Map",
            size=16, ha="center", fontfamily="sans-serif",
            color=_GOLD, weight="bold",
        )

        # Subtitle
        n_goals = sum(1 for o in outcomes if o == "Goal")
        total_xg = sum(xgs)
        fig.text(
            0.515, 0.93,
            f"{len(shots)} shots  ·  {n_goals} goals  ·  {total_xg:.2f} xG",
            size=10, ha="center", fontfamily="sans-serif",
            color=_MUTED,
        )

        # Legend
        fig.text(
            0.515, 0.02,
            "● Goal    ● Saved    ● Other        size = xG",
            size=9, ha="center", fontfamily="sans-serif",
            color=_MUTED,
        )

        return _to_buf(fig)

    except Exception as exc:
        _log.warning("Shot map render failed: %s", exc)
        plt.close("all")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Pass Network
# ═════════════════════════════════════════════════════════════════════════════

def render_pass_network(
    passes: List[Dict[str, Any]],
    player_name: str = "Player",
    figsize: Tuple[int, int] = (12, 8),
) -> Optional[io.BytesIO]:
    """Render a pass map showing pass origins and destinations.

    Parameters
    ----------
    passes : list[dict]
        Each dict must contain ``start_x``, ``start_y``, ``end_x``,
        ``end_y``, ``outcome`` (``'Complete'`` / ``'Incomplete'``), and
        optionally ``type``.  Coordinates in StatsBomb space (120×80).
    player_name : str
        Displayed in the chart title.
    figsize : tuple
        Figure dimensions in inches.

    Returns
    -------
    io.BytesIO containing a PNG image, or *None* on error / empty data.
    """
    try:
        from mplsoccer import Pitch
    except ImportError:
        _log.warning("mplsoccer not installed — pass network unavailable")
        return None

    if not passes:
        return None

    # Cap at _MAX_PASSES_DISPLAY for readability
    display_passes = passes[:_MAX_PASSES_DISPLAY]

    start_xs = np.array([p["start_x"] for p in display_passes])
    start_ys = np.array([p["start_y"] for p in display_passes])
    end_xs = np.array([p["end_x"] for p in display_passes])
    end_ys = np.array([p["end_y"] for p in display_passes])
    outcomes = [p.get("outcome", "Complete") for p in display_passes]

    # Build category masks: (complete, progressive) combos for layered drawing
    is_complete = np.array([o == "Complete" for o in outcomes])
    is_prog = np.array([
        _is_progressive(sx, ex)
        for sx, ex in zip(start_xs, end_xs)
    ])

    try:
        pitch = Pitch(
            pitch_color=_BG,
            line_color=_LINE,
            linewidth=1,
        )
        fig, ax = pitch.draw(figsize=figsize)
        fig.patch.set_facecolor(_BG)

        # Draw passes in 4 layers: incomplete-normal, incomplete-progressive,
        # complete-normal, complete-progressive (last on top).
        _layers = [
            (~is_complete & ~is_prog, _PASS_DEFAULT_COLOR, 1.2, 0.50, 1),
            (~is_complete & is_prog,  _PASS_DEFAULT_COLOR, 2.5, 0.60, 2),
            (is_complete & ~is_prog,  _GOLD,               1.2, 0.65, 3),
            (is_complete & is_prog,   _GOLD,               2.5, 0.80, 4),
        ]
        for mask, color, width, alpha, zorder in _layers:
            if not mask.any():
                continue
            pitch.arrows(
                start_xs[mask], start_ys[mask],
                end_xs[mask], end_ys[mask],
                width=width,
                headwidth=5, headlength=4, headaxislength=3,
                color=color,
                alpha=alpha,
                zorder=zorder,
                ax=ax,
            )

        # Title
        fig.text(
            0.515, 0.96, f"{player_name} — Pass Map",
            size=16, ha="center", fontfamily="sans-serif",
            color=_GOLD, weight="bold",
        )

        # Subtitle
        n_complete = sum(1 for o in outcomes if o == "Complete")
        n_prog = sum(
            1 for p in display_passes
            if _is_progressive(p["start_x"], p["end_x"])
        )
        fig.text(
            0.515, 0.93,
            f"{len(display_passes)} passes  ·  "
            f"{n_complete} complete  ·  {n_prog} progressive",
            size=10, ha="center", fontfamily="sans-serif",
            color=_MUTED,
        )

        # Legend
        fig.text(
            0.515, 0.02,
            "— Complete    — Incomplete        thick = progressive (≥10 m forward)",
            size=9, ha="center", fontfamily="sans-serif",
            color=_MUTED,
        )

        return _to_buf(fig)

    except Exception as exc:
        _log.warning("Pass network render failed: %s", exc)
        plt.close("all")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Heatmap
# ═════════════════════════════════════════════════════════════════════════════

def render_heatmap(
    locations: List[Tuple[float, float]],
    player_name: str = "Player",
    figsize: Tuple[int, int] = (12, 8),
) -> Optional[io.BytesIO]:
    """Render a touch/action heatmap on a full pitch.

    Parameters
    ----------
    locations : list[tuple[float, float]]
        (x, y) pairs in StatsBomb space (120×80).  Requires at least
        ``_MIN_HEATMAP_LOCATIONS`` points for KDE to work.
    player_name : str
        Displayed in the chart title.
    figsize : tuple
        Figure dimensions in inches.

    Returns
    -------
    io.BytesIO containing a PNG image, or *None* on error / insufficient data.
    """
    try:
        from mplsoccer import Pitch
    except ImportError:
        _log.warning("mplsoccer not installed — heatmap unavailable")
        return None

    if not locations or len(locations) < _MIN_HEATMAP_LOCATIONS:
        _log.info(
            "Heatmap requires ≥%d locations, got %d",
            _MIN_HEATMAP_LOCATIONS,
            len(locations) if locations else 0,
        )
        return None

    xs = np.array([loc[0] for loc in locations], dtype=float)
    ys = np.array([loc[1] for loc in locations], dtype=float)

    try:
        pitch = Pitch(
            pitch_color=_BG,
            line_color=_LINE,
            linewidth=1,
        )
        fig, ax = pitch.draw(figsize=figsize)
        fig.patch.set_facecolor(_BG)

        pitch.kdeplot(
            xs, ys,
            ax=ax,
            fill=True,
            cmap=_HEATMAP_CMAP,
            levels=100,
            thresh=0.05,
            zorder=1,
        )

        # Title
        fig.text(
            0.515, 0.96, f"{player_name} — Touch Heatmap",
            size=16, ha="center", fontfamily="sans-serif",
            color=_GOLD, weight="bold",
        )

        # Subtitle
        fig.text(
            0.515, 0.93,
            f"{len(locations)} touch locations",
            size=10, ha="center", fontfamily="sans-serif",
            color=_MUTED,
        )

        return _to_buf(fig)

    except Exception as exc:
        _log.warning("Heatmap render failed: %s", exc)
        plt.close("all")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Streamlit wrappers
# ═════════════════════════════════════════════════════════════════════════════

def show_shot_map(
    shots: List[Dict[str, float]],
    player_name: str = "Player",
) -> None:
    """Render and display a shot map in Streamlit."""
    buf = render_shot_map(shots, player_name)
    if buf is not None:
        st.image(buf, use_container_width=True)
    else:
        st.info("Shot map unavailable (mplsoccer not installed or no data).")


def show_pass_network(
    passes: List[Dict[str, Any]],
    player_name: str = "Player",
) -> None:
    """Render and display a pass network in Streamlit."""
    buf = render_pass_network(passes, player_name)
    if buf is not None:
        st.image(buf, use_container_width=True)
    else:
        st.info("Pass map unavailable (mplsoccer not installed or no data).")


def show_heatmap(
    locations: List[Tuple[float, float]],
    player_name: str = "Player",
) -> None:
    """Render and display a touch heatmap in Streamlit."""
    buf = render_heatmap(locations, player_name)
    if buf is not None:
        st.image(buf, use_container_width=True)
    else:
        st.info("Heatmap unavailable (mplsoccer not installed or insufficient data).")
