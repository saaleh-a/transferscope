"""Backtest Validator — compare predictions against actual post-transfer per-90.

Search any player via Sofascore, pick a transfer, and compare model predictions
against actual post-transfer per-90 stats from the correct season.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.data import sofascore_client
from backend.data.sofascore_client import (
    CORE_METRICS,
    discover_tournament_for_team,
    get_player_stats_for_season,
    get_player_transfer_history,
    get_season_list,
    normalize_position,
    search_player,
)
from backend.features import power_rankings
from backend.features.adjustment_models import paper_heuristic_predict
from backend.models.transfer_portal import (
    TransferPortalModel,
    build_feature_dict,
)
from frontend.theme import (
    COLORS,
    PLOTLY_LAYOUT,
    player_info_card,
    section_header,
)

from frontend.constants import METRIC_LABELS as _LABELS

_EPSILON = 0.001  # Minimum meaningful change threshold


# ── Season matching helpers ──────────────────────────────────────────────────


def _parse_season_years(name: str) -> Tuple[int, int]:
    """Parse season name to ``(start_year, end_year)``.

    Handles season names with text prefixes (e.g. from Sofascore API)::

        "24/25"                  → (2024, 2025)
        "2024/2025"              → (2024, 2025)
        "Liga Portugal 25/26"   → (2025, 2026)
        "Premier League 2024/2025" → (2024, 2025)
        "2024"                   → (2024, 2024)
        "MLS 2025"               → (2025, 2025)
    """
    name = name.strip()
    # Search for split-year pattern anywhere in the string (e.g. "25/26",
    # "2024/2025", or "Liga Portugal 25/26").
    m = re.search(r"(\d{2,4})\s*/\s*(\d{2,4})", name)
    if m:
        s, e = m.group(1), m.group(2)
        try:
            start = int(s) if len(s) == 4 else 2000 + int(s)
            end = int(e) if len(e) == 4 else 2000 + int(e)
            return (start, end)
        except ValueError:
            pass  # Fall through to standalone year check below
    # Search for a standalone 4-digit year anywhere in the string.
    m = re.search(r"(\d{4})", name)
    if m:
        year = int(m.group(1))
        return (year, year)
    return (0, 0)


def _match_season(
    seasons: List[Dict[str, Any]],
    transfer_year: int,
    transfer_month: int,
    mode: str,
) -> Optional[Dict[str, Any]]:
    """Pick the correct season for a transfer date.

    Parameters
    ----------
    seasons : list
        Season dicts with ``id`` and ``name``, newest first.
    transfer_year, transfer_month : int
        Year and month from the transfer date.
    mode : ``"pre"`` or ``"post"``

    Returns
    -------
    Season dict or *None*.
    """
    if not seasons:
        return None

    is_summer = 5 <= transfer_month <= 9

    parsed: List[Tuple[Dict[str, Any], int, int]] = []
    for s in seasons:
        sy, ey = _parse_season_years(s.get("name", ""))
        if sy > 0:
            parsed.append((s, sy, ey))

    if not parsed:
        return seasons[0]

    # Calendar-year leagues: prefer exact year, then nearest
    # Split-year leagues: use summer/winter logic
    best: Optional[Dict[str, Any]] = None

    for s, sy, ey in parsed:
        if sy == ey:
            # Calendar-year league (MLS "2025")
            if mode == "pre":
                if sy == transfer_year:
                    return s  # Same year — best match
                if sy == transfer_year - 1 and best is None:
                    best = s
            else:  # post
                if sy == transfer_year:
                    return s  # Same year — best match
                if sy == transfer_year + 1 and best is None:
                    best = s
        else:
            # Split-year league ("24/25")
            if is_summer:
                if mode == "pre" and ey == transfer_year:
                    return s  # Season that just ended
                if mode == "post" and sy == transfer_year:
                    return s  # Season about to start
            else:
                # Winter transfer — same season for both pre and post
                if sy == transfer_year - 1 and ey == transfer_year:
                    return s
                if sy == transfer_year and transfer_month >= 8:
                    return s

    if best is not None:
        return best

    # Fallback: newest for post, second-newest for pre
    if mode == "post":
        return parsed[0][0]
    return parsed[min(1, len(parsed) - 1)][0]


def _resolve_transfer_context(
    player_id: int,
    from_team_id: Optional[int],
    to_team_id: Optional[int],
    transfer_date: str,
) -> Optional[Dict[str, Any]]:
    """Dynamically resolve tournament/season IDs for a transfer.

    Returns dict with ``pre_tournament_id``, ``pre_season_id``,
    ``post_tournament_id``, ``post_season_id``, and season name labels,
    or *None* on failure.
    """
    if not from_team_id or not to_team_id:
        return None

    from_tid = discover_tournament_for_team(from_team_id)
    to_tid = discover_tournament_for_team(to_team_id)
    if not from_tid or not to_tid:
        return None

    from_seasons = get_season_list(from_tid)
    to_seasons = get_season_list(to_tid)
    if not from_seasons or not to_seasons:
        return None

    try:
        year = int(transfer_date[:4])
        month = int(transfer_date[5:7])
    except (ValueError, IndexError):
        return None

    pre_season = _match_season(from_seasons, year, month, "pre")
    post_season = _match_season(to_seasons, year, month, "post")
    if not pre_season or not post_season:
        return None

    return {
        "pre_tournament_id": from_tid,
        "pre_season_id": pre_season["id"],
        "pre_season_name": pre_season.get("name", ""),
        "post_tournament_id": to_tid,
        "post_season_id": post_season["id"],
        "post_season_name": post_season.get("name", ""),
    }


# ── Display helpers ──────────────────────────────────────────────────────────


def _color_pct_error(val: float) -> str:
    """Return CSS color for a percentage error value."""
    if val < 10:
        return COLORS["accent_green"]
    if val < 20:
        return COLORS["accent_amber"]
    return COLORS["accent_crimson"]


def _is_direction_match(predicted_change: float, actual_change: float) -> bool:
    """Return True if predicted and actual changes share the same direction."""
    return (predicted_change > 0 and actual_change > 0) or \
           (predicted_change < 0 and actual_change < 0)


def _direction_icon(predicted_change: float, actual_change: float) -> str:
    """Return ✅ or ❌ based on whether predicted direction matches actual."""
    if abs(actual_change) < _EPSILON:
        return "➖"
    return "✅" if _is_direction_match(predicted_change, actual_change) else "❌"


# ── Main render ──────────────────────────────────────────────────────────────


def render():
    st.header("Backtest Validator")
    st.caption(
        "Compare model predictions against actual post-transfer per-90 stats. "
        "Search any player — stats are fetched live from Sofascore."
    )

    with st.expander("ℹ️ Data sources & limitations"):
        st.markdown(
            "**Predicted** — generated by the paper-aligned heuristic model "
            "(or trained TF model when available). Uses the player's pre-transfer "
            "per-90 stats adjusted by Power Rankings (team quality gap), "
            "team-position averages, and per-metric sensitivity coefficients.\n\n"
            "**Actual** — fetched from the Sofascore season statistics API for "
            "the *first season after the transfer*, resolved dynamically from "
            "the transfer date.\n\n"
            "**Direction ✅/❌** — compares the *change from pre-transfer* baseline: "
            "✅ means both predicted and actual moved in the same direction (both up "
            "or both down); ❌ means they moved in opposite directions.\n\n"
            "⚠️ **Limitation:** Sofascore returns aggregate season stats per "
            "tournament — there is no per-club filter. If a player transferred "
            "mid-season, the figures may include time at both clubs."
        )

    # ── Step 1: Player search ────────────────────────────────────────────
    player_query = st.text_input(
        "🔍 Search player",
        placeholder="e.g. Bukayo Saka, Boniface, Salah",
        key="bt_player_query",
    )

    if not player_query or len(player_query.strip()) < 2:
        st.info("Enter a player name above to search.")
        return

    with st.spinner("Searching…"):
        results = search_player(player_query.strip())

    if not results:
        st.warning("No players found. Try a different spelling.")
        return

    # ── Step 2: Select player ────────────────────────────────────────────
    player_labels = [
        f"{p['name']} — {p.get('team_name', 'Unknown')}"
        + (f" (age {p['age']})" if p.get("age") else "")
        for p in results
    ]
    selected_player_idx = st.selectbox(
        "Select player",
        range(len(player_labels)),
        format_func=lambda i: player_labels[i],
        key="bt_player_select",
    )
    # Guard against stale selectbox state after search query changes
    if selected_player_idx >= len(results):
        selected_player_idx = 0
    player = results[selected_player_idx]
    player_id = player["id"]
    player_name = player["name"]

    # Transfermarkt verification link
    _tm_query = player_name.replace(" ", "+")
    st.caption(
        f"[Verify transfer history on Transfermarkt ↗]"
        f"(https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"
        f"?query={_tm_query})"
    )

    # ── Step 3: Fetch transfer history ───────────────────────────────────
    with st.spinner("Loading transfer history…"):
        transfers = get_player_transfer_history(player_id)

    if not transfers:
        st.warning(f"No transfer history found for {player_name}.")
        return

    # Filter out entries with missing team info
    valid_transfers = [
        t for t in transfers
        if t.get("from_team", {}).get("id") and t.get("to_team", {}).get("id")
        and t.get("from_team", {}).get("name") and t.get("to_team", {}).get("name")
    ]
    if not valid_transfers:
        st.warning("No valid transfers with complete team data.")
        return

    # NOTE: Sofascore transfer type labels (e.g. "Loan return") are
    # frequently inaccurate — permanent transfers may be labelled as loans
    # and vice-versa.  We omit the type from the dropdown to avoid
    # confusion and provide a Transfermarkt link above for verification.
    transfer_labels = [
        f"{t['from_team']['name']} → {t['to_team']['name']}  "
        f"({t.get('transfer_date', 'Unknown date')})"
        for t in valid_transfers
    ]
    selected_transfer_idx = st.selectbox(
        "Select transfer",
        range(len(transfer_labels)),
        format_func=lambda i: transfer_labels[i],
        key="bt_transfer_select",
    )
    # Guard against stale selectbox state after player/search changes
    if selected_transfer_idx >= len(valid_transfers):
        selected_transfer_idx = 0
    transfer = valid_transfers[selected_transfer_idx]

    from_team_id = transfer["from_team"]["id"]
    from_club = transfer["from_team"]["name"]
    to_team_id = transfer["to_team"]["id"]
    to_club = transfer["to_team"]["name"]
    transfer_date = transfer.get("transfer_date", "")

    st.markdown("---")

    # ── Step 4: Resolve seasons dynamically ──────────────────────────────
    with st.spinner("Resolving seasons…"):
        ctx = _resolve_transfer_context(
            player_id, from_team_id, to_team_id, transfer_date
        )

    if ctx is None:
        st.error(
            "Could not resolve tournament/season for this transfer. "
            "The teams may not be in a recognized domestic league on Sofascore."
        )
        return

    pre_tournament_id = ctx["pre_tournament_id"]
    pre_season_id = ctx["pre_season_id"]
    post_tournament_id = ctx["post_tournament_id"]
    post_season_id = ctx["post_season_id"]

    st.caption(
        f"Pre-transfer: tournament {pre_tournament_id}, "
        f"season {ctx['pre_season_name']} ({pre_season_id})  ·  "
        f"Post-transfer: tournament {post_tournament_id}, "
        f"season {ctx['post_season_name']} ({post_season_id})"
    )

    # ── Pre-transfer stats ───────────────────────────────────────────────
    with st.spinner("Fetching pre-transfer stats…"):
        try:
            pre_stats = get_player_stats_for_season(
                player_id, pre_tournament_id, pre_season_id
            )
        except Exception as e:
            st.error(f"Failed to fetch pre-transfer stats: {e}")
            return

    pre_per90 = pre_stats.get("per90", {})
    pre_per90_clean: Dict[str, float] = {
        m: (pre_per90.get(m) if pre_per90.get(m) is not None else 0.0)
        for m in CORE_METRICS
    }
    minutes = pre_stats.get("minutes_played", 0) or 0

    player_position = pre_stats.get("position") or normalize_position(
        player.get("position", "F")
    )

    player_info_card(
        player_name, from_club, player_position, minutes,
        season_label=f"Pre-transfer · {ctx['pre_season_name']} ({transfer_date})",
        rating=pre_stats.get("rating"),
    )

    if minutes == 0:
        st.warning(
            "⚠️ No pre-transfer minutes found in this season. "
            "The player may not have been registered in this league/season."
        )
        return

    # ── Power Rankings ───────────────────────────────────────────────────
    source_ranking = power_rankings.get_team_ranking(from_club)
    target_ranking = power_rankings.get_team_ranking(to_club)

    source_norm = source_ranking.normalized_score if source_ranking else 50.0
    source_league = source_ranking.league_mean_normalized if source_ranking else 50.0
    target_norm = target_ranking.normalized_score if target_ranking else 50.0
    target_league = target_ranking.league_mean_normalized if target_ranking else 50.0

    if source_ranking is None or target_ranking is None:
        missing = []
        if source_ranking is None:
            missing.append(from_club)
        if target_ranking is None:
            missing.append(to_club)
        st.warning(
            f"⚠️ Could not find Power Ranking for: {', '.join(missing)}. "
            "Using default (50.0) — predictions may be less accurate."
        )

    change_ra = (target_norm - target_league) - (source_norm - source_league)

    # ── Team-position averages ───────────────────────────────────────────
    source_pos_avg: Dict[str, float] = {}
    target_pos_avg: Dict[str, float] = {}

    try:
        source_pos_avg, _ = sofascore_client.get_team_position_averages(
            from_team_id, player_position
        )
    except Exception:
        pass
    try:
        target_pos_avg, _ = sofascore_client.get_team_position_averages(
            to_team_id, player_position
        )
    except Exception:
        pass

    if not source_pos_avg:
        source_pos_avg = pre_per90_clean.copy()
    if not target_pos_avg:
        target_pos_avg = pre_per90_clean.copy()

    # ── Run prediction ───────────────────────────────────────────────────
    with st.spinner("Running prediction pipeline…"):
        predicted: Dict[str, float] = {}
        try:
            model = TransferPortalModel()
            model.load()
            if model.fitted:
                fd = build_feature_dict(
                    player_per90=pre_per90_clean,
                    team_ability_current=source_norm,
                    team_ability_target=target_norm,
                    league_ability_current=source_league,
                    league_ability_target=target_league,
                    team_pos_current=source_pos_avg,
                    team_pos_target=target_pos_avg,
                )
                predicted = model.predict(fd)
        except Exception:
            pass

        if not predicted:
            player_rating = pre_stats.get("rating")
            predicted = paper_heuristic_predict(
                player_per90=pre_per90_clean,
                source_pos_avg=source_pos_avg,
                target_pos_avg=target_pos_avg,
                change_relative_ability=change_ra,
                player_rating=player_rating,
                source_league_mean=source_league,
                target_league_mean=target_league,
            )

    # ── Fetch actual post-transfer stats ─────────────────────────────────
    with st.spinner("Fetching actual post-transfer stats…"):
        try:
            post_stats = get_player_stats_for_season(
                player_id, post_tournament_id, post_season_id
            )
        except Exception as e:
            st.error(f"Failed to fetch post-transfer stats: {e}")
            post_stats = {}

    post_per90 = post_stats.get("per90", {}) if post_stats else {}
    post_minutes = (post_stats.get("minutes_played", 0) or 0) if post_stats else 0

    has_post_data = post_minutes > 0 and any(
        post_per90.get(m) is not None for m in CORE_METRICS
    )

    if not has_post_data:
        st.warning(
            "⚠️ No post-transfer performance data available. "
            f"Season: {ctx['post_season_name']} "
            f"(tournament {post_tournament_id}, season {post_season_id}). "
            "Predictions are shown below without actual comparison."
        )
        section_header("Predicted Per-90 (no actuals available)")
        pred_rows = []
        for m in CORE_METRICS:
            pred_rows.append({
                "Metric": _LABELS.get(m, m),
                "Pre-Transfer": f"{pre_per90_clean.get(m, 0.0):.3f}",
                "Predicted": f"{predicted.get(m, 0.0):.3f}",
            })
        st.dataframe(
            pd.DataFrame(pred_rows),
            use_container_width=True,
            hide_index=True,
        )
        return

    actual_per90: Dict[str, float] = {
        m: (post_per90.get(m) if post_per90.get(m) is not None else 0.0)
        for m in CORE_METRICS
    }

    # ── Comparison table ─────────────────────────────────────────────────
    section_header(
        "Prediction vs Actual",
        f"Post-transfer: {ctx['post_season_name']} · {post_minutes:,} mins played",
    )

    table_rows = []
    pct_errors: List[float] = []
    direction_correct = 0
    direction_total = 0

    for m in CORE_METRICS:
        pred_val = predicted.get(m, 0.0)
        actual_val = actual_per90.get(m, 0.0)
        pre_val = pre_per90_clean.get(m, 0.0)
        diff = pred_val - actual_val

        if abs(actual_val) > _EPSILON:
            pct_err = abs(diff) / abs(actual_val) * 100
        else:
            pct_err = 0.0 if abs(pred_val) < _EPSILON else 100.0
        pct_errors.append(pct_err)

        predicted_change = pred_val - pre_val
        actual_change = actual_val - pre_val
        direction = _direction_icon(predicted_change, actual_change)
        if abs(actual_change) > _EPSILON:
            direction_total += 1
            if _is_direction_match(predicted_change, actual_change):
                direction_correct += 1

        color = _color_pct_error(pct_err)
        table_rows.append({
            "Metric": _LABELS.get(m, m),
            "Pre": f"{pre_val:.3f}",
            "Predicted": f"{pred_val:.3f}",
            "Actual": f"{actual_val:.3f}",
            "Difference": f"{diff:+.3f}",
            "% Error": pct_err,
            "Direction": direction,
            "_pct_err_color": color,
        })

    _render_comparison_table(table_rows)
    st.caption(
        "**Direction** compares change from pre-transfer: ✅ = both predicted "
        "and actual moved the same way (↑↑ or ↓↓), ❌ = opposite directions."
    )

    # ── Summary cards ────────────────────────────────────────────────────
    st.markdown("---")
    section_header("Summary", "Aggregate accuracy metrics")

    mean_pct_error = sum(pct_errors) / len(pct_errors) if pct_errors else 0.0
    within_10 = sum(1 for e in pct_errors if e < 10)
    within_20 = sum(1 for e in pct_errors if e < 20)
    dir_accuracy_label = "N/A"
    if direction_total > 0:
        dir_accuracy = direction_correct / direction_total * 100
        dir_accuracy_label = f"{dir_accuracy:.0f}%"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Abs % Error", f"{mean_pct_error:.1f}%")
    with col2:
        st.metric("Metrics < 10% Error", f"{within_10} / {len(CORE_METRICS)}")
    with col3:
        st.metric("Metrics < 20% Error", f"{within_20} / {len(CORE_METRICS)}")
    with col4:
        st.metric("Direction Accuracy", dir_accuracy_label)

    # ── Plotly grouped bar chart ─────────────────────────────────────────
    st.markdown("---")
    section_header("Predicted vs Actual", "Per-90 comparison by metric")

    metric_labels = [_LABELS.get(m, m) for m in CORE_METRICS]
    pred_vals = [predicted.get(m, 0.0) for m in CORE_METRICS]
    actual_vals = [actual_per90.get(m, 0.0) for m in CORE_METRICS]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Predicted",
        x=metric_labels,
        y=pred_vals,
        marker_color=COLORS["accent_gold"],
        opacity=0.9,
    ))
    fig.add_trace(go.Bar(
        name="Actual",
        x=metric_labels,
        y=actual_vals,
        marker_color=COLORS["accent_green"],
        opacity=0.9,
    ))
    layout = dict(PLOTLY_LAYOUT)
    layout["title"] = dict(text="Predicted vs Actual Per-90", **PLOTLY_LAYOUT["title"])
    fig.update_layout(
        **layout,
        barmode="group",
        xaxis_title="Metric",
        yaxis_title="Per-90 Value",
        height=450,
    )
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # ── Direction accuracy detail ────────────────────────────────────────
    st.markdown("---")
    section_header(
        "Direction Accuracy",
        "Did the model predict whether each metric would rise or fall?",
    )

    dir_rows = []
    for m in CORE_METRICS:
        pre_val = pre_per90_clean.get(m, 0.0)
        pred_val = predicted.get(m, 0.0)
        actual_val = actual_per90.get(m, 0.0)
        pred_change = pred_val - pre_val
        actual_change = actual_val - pre_val

        if abs(actual_change) < _EPSILON:
            pred_dir = "↑" if pred_change > 0 else ("↓" if pred_change < 0 else "→")
            dir_rows.append({
                "Metric": _LABELS.get(m, m),
                "Pre": f"{pre_val:.3f}",
                "Predicted": f"{pred_val:.3f}",
                "Actual": f"{actual_val:.3f}",
                "Pred Δ": f"{pred_change:+.3f} {pred_dir}",
                "Actual Δ": "→ (no change)",
                "Result": "➖",
            })
        else:
            pred_dir = "↑" if pred_change > 0 else "↓"
            actual_dir = "↑" if actual_change > 0 else "↓"
            correct = _is_direction_match(pred_change, actual_change)
            dir_rows.append({
                "Metric": _LABELS.get(m, m),
                "Pre": f"{pre_val:.3f}",
                "Predicted": f"{pred_val:.3f}",
                "Actual": f"{actual_val:.3f}",
                "Pred Δ": f"{pred_change:+.3f} {pred_dir}",
                "Actual Δ": f"{actual_change:+.3f} {actual_dir}",
                "Result": "✅" if correct else "❌",
            })

    st.dataframe(
        pd.DataFrame(dir_rows),
        use_container_width=True,
        hide_index=True,
    )


def _render_comparison_table(rows: List[Dict[str, Any]]) -> None:
    """Render the comparison table with color-coded % Error column."""
    c_border = COLORS["border"]
    c_text = COLORS["text_primary"]
    c_sec = COLORS["text_secondary"]
    c_muted = COLORS["text_muted"]
    c_gold = COLORS["accent_gold"]
    c_green = COLORS["accent_green"]

    header = (
        "<table style='width:100%; border-collapse:collapse; "
        "font-family: \"JetBrains Mono\", \"Fira Code\", monospace; font-size:0.85rem;'>"
        f"<thead><tr style='border-bottom:2px solid {c_border};'>"
        f"<th style='text-align:left; padding:8px; color:{c_sec};'>Metric</th>"
        f"<th style='text-align:right; padding:8px; color:{c_sec};'>Pre</th>"
        f"<th style='text-align:right; padding:8px; color:{c_sec};'>Predicted</th>"
        f"<th style='text-align:right; padding:8px; color:{c_sec};'>Actual</th>"
        f"<th style='text-align:right; padding:8px; color:{c_sec};'>Diff</th>"
        f"<th style='text-align:right; padding:8px; color:{c_sec};'>% Error</th>"
        f"<th style='text-align:center; padding:8px; color:{c_sec};'>Dir</th>"
        "</tr></thead><tbody>"
    )

    body = ""
    for r in rows:
        color = r["_pct_err_color"]
        body += (
            f"<tr style='border-bottom:1px solid {c_border};'>"
            f"<td style='padding:8px; color:{c_text};'>{r['Metric']}</td>"
            f"<td style='text-align:right; padding:8px; color:{c_muted};'>"
            f"{r['Pre']}</td>"
            f"<td style='text-align:right; padding:8px; color:{c_gold};'>"
            f"{r['Predicted']}</td>"
            f"<td style='text-align:right; padding:8px; color:{c_green};'>"
            f"{r['Actual']}</td>"
            f"<td style='text-align:right; padding:8px; color:{c_text};'>"
            f"{r['Difference']}</td>"
            f"<td style='text-align:right; padding:8px; color:{color}; font-weight:600;'>"
            f"{r['% Error']:.1f}%</td>"
            f"<td style='text-align:center; padding:8px;'>{r['Direction']}</td>"
            f"</tr>"
        )

    html = header + body + "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)
