"""Transfer Impact dashboard — predicted performance change (paper Fig 1).

Inputs: player search, target club search (with Sofascore autocomplete),
        optional season selector.
Outputs:
  (a) metric bars — predicted % change per metric
  (b) power ranking chart — before/after team Power Rankings
  (c) RAG confidence indicator
  (d) swarm plots — player vs league/team context (real league data)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import streamlit as st

from backend.data import sofascore_client, elo_router


@st.cache_data(ttl=86400)
def _get_league_players_cached(tournament_id: int, season_id) -> list:
    """Fetch league player stats once per day — shared across all sessions."""
    return sofascore_client.get_league_player_stats(
        tournament_id, season_id, limit=100
    )
from backend.data.sofascore_client import CORE_METRICS, OFFENSIVE_METRICS, DEFENSIVE_METRICS
from backend.features import power_rankings, rolling_windows
from backend.features.adjustment_models import paper_heuristic_predict
from backend.models.shortlist_scorer import compute_percentage_changes
from backend.models.transfer_portal import (
    TransferPortalModel,
    build_feature_dict,
)
from backend.utils.league_registry import LEAGUES
from frontend.components import metric_bar, power_ranking_chart, swarm_plot
from frontend.theme import section_header, confidence_badge, player_info_card, COLORS

_log = logging.getLogger(__name__)


def render():
    st.header("Transfer Impact")
    st.caption("Predict how a player's performance will change at a new club")

    # ── Inputs ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        player_query = st.text_input("Search player", placeholder="e.g. Bukayo Saka")
    with col2:
        target_club_query = st.text_input("Target club", placeholder="e.g. Real Madrid")

    if not player_query or not target_club_query:
        st.info("Enter a player name and target club to generate a transfer impact prediction.")
        return

    # Gate all search/prediction behind an explicit button click so that
    # Streamlit doesn't fire mid-keystroke with a partial query.
    analyse_clicked = st.button("Analyse Transfer", type="primary")

    # Persist last successful results across reruns so they survive
    # Streamlit's re-execution cycle after the button click.
    if analyse_clicked:
        with st.spinner("Searching for player..."):
            try:
                search_results = sofascore_client.search_player(player_query)
            except Exception as e:
                st.error(f"Sofascore search failed: {e}")
                return
        if not search_results:
            st.warning(f"No players found for '{player_query}'.")
            return
        st.session_state["ti_search_results"] = search_results
        st.session_state["ti_player_query"] = player_query
        st.session_state["ti_target_club_query"] = target_club_query

    # Use cached results on subsequent reruns (e.g. widget interaction)
    search_results = st.session_state.get("ti_search_results")
    stored_query = st.session_state.get("ti_player_query", "")
    if search_results and stored_query != player_query:
        st.session_state.pop("ti_search_results", None)
        st.caption("Query changed — click **Analyse Transfer** to search.")
        return
    if not search_results:
        st.caption("Click **Analyse Transfer** to search.")
        return

    def _player_label(p: dict) -> str:
        parts = [p.get("name", "Unknown")]
        if p.get("age"):
            parts.append(f"Age {p['age']}")
        if p.get("nationality"):
            parts.append(p["nationality"])
        if p.get("team_name"):
            parts.append(p["team_name"])
        return " · ".join(parts)

    player_options = {_player_label(p): p for p in search_results}
    selected = st.selectbox("Select player", list(player_options.keys()), key="ti_player_select")
    if selected is None or selected not in player_options:
        st.info("Select a player from the dropdown.")
        return
    player_info = player_options[selected]
    player_id = player_info["id"]

    # ── Target club autocomplete (Sofascore search) ──────────────────────
    with st.spinner("Searching for target club..."):
        try:
            club_results = sofascore_client.search_team(target_club_query)
        except Exception as e:
            st.warning(f"Club search failed: {e}")
            club_results = []
    target_club_display = target_club_query
    target_team_id = None  # Initialize before conditional to prevent NameError

    if club_results:
        club_options = {
            f"{c['name']}" + (f" ({c.get('country', '')})" if c.get("country") else ""): c
            for c in club_results
        }
        selected_club = st.selectbox(
            "Select target club", list(club_options.keys()), key="ti_club_select"
        )
        if selected_club is not None and selected_club in club_options:
            chosen = club_options[selected_club]
            target_club_display = chosen["name"]
            target_team_id = chosen.get("id")
    else:
        st.caption(f"Using '{target_club_query}' as target club name (no Sofascore results).")

    # ── Season selector ──────────────────────────────────────────────────
    tournament_id = player_info.get("tournament_id")
    selected_season_id: Optional[int] = None
    selected_season_label = "Current"

    if tournament_id:
        seasons = sofascore_client.get_season_list(tournament_id)
        if seasons:
            season_labels = {s["name"]: s["id"] for s in seasons}
            chosen_season = st.selectbox(
                "Season",
                ["Current"] + list(season_labels.keys()),
                key="ti_season",
            )
            if chosen_season != "Current":
                selected_season_id = season_labels[chosen_season]
                selected_season_label = chosen_season

    # ── Fetch player stats ───────────────────────────────────────────────
    with st.spinner("Fetching player stats..."):
        try:
            if selected_season_id and tournament_id:
                player_stats = sofascore_client.get_player_stats_for_season(
                    player_id, tournament_id, selected_season_id
                )
            else:
                player_stats = sofascore_client.get_player_stats(player_id)
        except Exception as e:
            st.error(f"Failed to fetch player stats: {e}")
            return

    # ── Resolve tournament_id if search didn't provide it ────────────────
    # get_player_stats() discovers and caches the tournament internally;
    # retrieve it so that the league-context section can use it later.
    if not tournament_id:
        tournament_id = sofascore_client.get_cached_tournament_id(player_id)
    if not tournament_id and player_stats.get("team_id"):
        tournament_id = sofascore_client.discover_tournament_for_team(
            player_stats["team_id"]
        )

    player_name = player_stats.get("name", "Unknown")
    current_team = player_stats.get("team", "Unknown")
    position = player_stats.get("position", "Unknown")
    minutes = player_stats.get("minutes_played", 0)
    if minutes is None:
        minutes = 0
    current_per90 = player_stats.get("per90", {})

    player_info_card(player_name, current_team, position, minutes, selected_season_label,
                     rating=player_stats.get("rating"))

    # ── Power Rankings ───────────────────────────────────────────────────
    with st.spinner("Computing Power Rankings..."):
        source_ranking = power_rankings.get_team_ranking(current_team)
        target_ranking = power_rankings.get_team_ranking(target_club_display)

    if source_ranking is None:
        st.warning(f"Could not find Power Ranking for {current_team}. Using defaults.")
        source_norm = 50.0
        source_league_mean = 50.0
    else:
        source_norm = source_ranking.normalized_score
        source_league_mean = source_ranking.league_mean_normalized
        if source_ranking.match_type == "fuzzy":
            st.info(
                f"⚠️ Approximate match: '{current_team}' → "
                f"'{source_ranking.team_name}' (fuzzy)"
            )

    if target_ranking is None:
        st.warning(f"Could not find Power Ranking for {target_club_display}. Using defaults.")
        target_norm = 50.0
        target_league_mean = 50.0
    else:
        target_norm = target_ranking.normalized_score
        target_league_mean = target_ranking.league_mean_normalized
        if target_ranking.match_type == "fuzzy":
            st.info(
                f"⚠️ Approximate match: '{target_club_display}' → "
                f"'{target_ranking.team_name}' (fuzzy)"
            )

    # Show data source status in a compact expander
    with st.expander("ℹ️ Data source status", expanded=False):
        # Reuses the cached result from get_team_ranking() calls above
        all_teams, all_snapshots = power_rankings.compute_daily_rankings()
        n_teams = len(all_teams)
        n_leagues = len(all_snapshots)
        src_match = source_ranking.match_type if source_ranking else "not found"
        tgt_match = target_ranking.match_type if target_ranking else "not found"
        st.markdown(
            f"**Elo teams loaded:** {n_teams} across {n_leagues} leagues  \n"
            f"**{current_team}:** {src_match} match"
            f" → score {source_norm:.1f}  \n"
            f"**{target_club_display}:** {tgt_match} match"
            f" → score {target_norm:.1f}"
        )

    change_ra = (target_norm - target_league_mean) - (source_norm - source_league_mean)

    # ── Opta analyst context (UI-only — never fed to model) ──────────────
    # Fetch raw Opta team objects for form and momentum signals.
    # These use current-date data only; they are display context, not features.
    _src_opta = None
    _tgt_opta = None
    try:
        from backend.data import opta_client as _opta_client
        _opta_teams = _opta_client.get_team_rankings()
        _opta_by_name = {t.team.lower(): t for t in _opta_teams}
        _src_opta = _opta_by_name.get(current_team.lower())
        _tgt_opta = _opta_by_name.get(target_club_display.lower())
        # Fallback: case-insensitive search if exact lower didn't match
        if _src_opta is None:
            for t in _opta_teams:
                if t.team.lower() == current_team.lower():
                    _src_opta = t
                    break
        if _tgt_opta is None:
            for t in _opta_teams:
                if t.team.lower() == target_club_display.lower():
                    _tgt_opta = t
                    break
    except Exception as _opta_exc:
        _log.debug("Opta analyst context unavailable: %s", _opta_exc)

    # form_delta = current rating − season average (positive = in form)
    _src_form = (
        _src_opta.rating - _src_opta.season_avg_rating
        if _src_opta and _src_opta.season_avg_rating
        else None
    )
    _tgt_form = (
        _tgt_opta.rating - _tgt_opta.season_avg_rating
        if _tgt_opta and _tgt_opta.season_avg_rating
        else None
    )

    # Opta league ratings for the ambition (league step-up) metric
    _src_league_code = source_ranking.league_code if source_ranking else None
    _tgt_league_code = target_ranking.league_code if target_ranking else None
    _src_opta_league = power_rankings.get_league_opta_rating(_src_league_code, current_team)
    _tgt_opta_league = power_rankings.get_league_opta_rating(_tgt_league_code, target_club_display)
    _league_step = _tgt_opta_league - _src_opta_league

    # ── (c) RAG confidence ───────────────────────────────────────────────
    features = rolling_windows.compute_player_features(player_stats)
    confidence = features.confidence
    confidence_badge(confidence, features.weight, minutes)

    if confidence == "red":
        st.warning("Low data confidence — prediction heavily relies on priors.")

    # ── Build prediction ─────────────────────────────────────────────────
    # Replace None with 0.0 for model input; track which metrics have data
    current_per90_clean = {m: (current_per90.get(m) if current_per90.get(m) is not None else 0.0) for m in CORE_METRICS}
    has_real_data = any(current_per90.get(m) is not None for m in CORE_METRICS)

    if not has_real_data:
        st.warning(
            "⚠️ No per-90 stats available for this player/season. "
            "Stats may not have loaded from Sofascore — try a different season "
            "or check that the player has played enough minutes."
        )

    # ── Fetch team-position averages (paper Section 2.3) ─────────────────
    # These capture tactical style: how each team uses players in this position.
    source_pos_avg: Dict[str, float] = {}
    target_pos_avg: Dict[str, float] = {}
    source_pos_players: list = []
    target_pos_players: list = []

    team_id = player_stats.get("team_id")
    if team_id:
        with st.spinner("Fetching source team position data..."):
            try:
                source_pos_avg, source_pos_players = (
                    sofascore_client.get_team_position_averages(team_id, position)
                )
            except Exception as e:
                st.warning(f"Could not fetch source team position data: {e}")

    if target_team_id:
        with st.spinner("Fetching target team position data..."):
            try:
                target_pos_avg, target_pos_players = (
                    sofascore_client.get_team_position_averages(target_team_id, position)
                )
            except Exception as e:
                st.warning(f"Could not fetch target team position data: {e}")

    # Fallback: use player's own stats if position averages are empty
    if not source_pos_avg:
        source_pos_avg = current_per90_clean.copy()
    if not target_pos_avg:
        target_pos_avg = current_per90_clean.copy()

    # Only use the TF model if trained weights have been saved to disk.
    predicted_target = {}
    predicted_current = {}
    try:
        model = TransferPortalModel()
        model.load()  # load() sets self.fitted = True only if files exist
        if model.fitted:
            # Paper Section 4: simulate at TARGET club
            fd_target = build_feature_dict(
                player_per90=current_per90_clean,
                team_ability_current=source_norm,
                team_ability_target=target_norm,
                league_ability_current=source_league_mean,
                league_ability_target=target_league_mean,
                team_pos_current=source_pos_avg,
                team_pos_target=target_pos_avg,
            )
            predicted_target = model.predict(fd_target)
            # Paper Section 4: simulate at CURRENT club as baseline
            # "we generate performance predictions using Transfer Portal
            # for players at their current club too"
            fd_current = build_feature_dict(
                player_per90=current_per90_clean,
                team_ability_current=source_norm,
                team_ability_target=source_norm,
                league_ability_current=source_league_mean,
                league_ability_target=source_league_mean,
                team_pos_current=source_pos_avg,
                team_pos_target=source_pos_avg,
            )
            predicted_current = model.predict(fd_current)
    except Exception as e:
        st.warning(f"TF model prediction failed, using heuristic fallback: {e}")
        predicted_target = {}
        predicted_current = {}

    # Paper-aligned heuristic fallback: uses team-position style data +
    # relative ability polynomial to give per-metric predictions
    # (not flat group adjustments).
    if not predicted_target:
        player_rating = player_stats.get("rating")
        predicted_target = paper_heuristic_predict(
            player_per90=current_per90_clean,
            source_pos_avg=source_pos_avg,
            target_pos_avg=target_pos_avg,
            change_relative_ability=change_ra,
            player_rating=player_rating,
            source_league_mean=source_league_mean,
            target_league_mean=target_league_mean,
        )
        # Paper Section 4: baseline = simulate at current club (ra=0, same team)
        predicted_current = paper_heuristic_predict(
            player_per90=current_per90_clean,
            source_pos_avg=source_pos_avg,
            target_pos_avg=source_pos_avg,
            change_relative_ability=0.0,
            player_rating=player_rating,
            source_league_mean=source_league_mean,
            target_league_mean=source_league_mean,
        )

    # Simulated Current = model's prediction at current club (for reference)
    baseline = predicted_current if predicted_current else current_per90_clean

    # ── Re-anchor model adjustments to actual per-90 ─────────────────────
    # The model produces predicted_target and predicted_current, both of
    # which may diverge from the player's actual stats (model bias).
    # We extract the model's DELTA (the adjustment it thinks the transfer
    # causes) and apply that delta to the player's ACTUAL per-90 values.
    # This way predictions are grounded in reality:
    #   predicted = actual + (model_target - model_current)
    if predicted_current and predicted_target:
        reanchored = {}
        for _m in CORE_METRICS:
            _actual = current_per90_clean.get(_m, 0.0)
            _sim_current = predicted_current.get(_m, 0.0)
            _sim_target = predicted_target.get(_m, 0.0)
            _delta = _sim_target - _sim_current
            reanchored[_m] = max(0.0, _actual + _delta)
        predicted_target = reanchored

    # ── Compute prediction confidence (Monte Carlo dropout) ──────────────
    prediction_confidence: Dict[str, float] = {}
    try:
        if model.fitted and fd_target:  # type: ignore[possibly-undefined]
            _, std_devs = model.predict_with_confidence(fd_target, current_per90_clean)
            if std_devs:
                prediction_confidence = std_devs
    except (NameError, AttributeError, Exception):
        pass  # confidence estimation is non-critical

    # ── (a) Metric bars ──────────────────────────────────────────────────
    # Percentage changes are computed relative to the player's ACTUAL per-90
    # stats — the model delta has been re-anchored to actual above.
    pct_changes = compute_percentage_changes(current_per90_clean, predicted_target)

    # Transfer context summary — paper Section 4.3 style
    _ra_label = "stronger" if change_ra > 0 else ("weaker" if change_ra < 0 else "equivalent")
    st.markdown(
        f'<div style="display:flex; gap:1.5rem; margin:0.8rem 0 1.2rem; flex-wrap:wrap;">'
        f'<div class="ts-stat-card" style="flex:1; min-width:160px;">'
        f'<span class="ts-stat-label">Relative Ability Δ</span>'
        f'<span class="ts-stat-value" style="font-size:1.3rem;">{change_ra:+.1f}</span>'
        f'<span style="font-size:0.72rem; color:var(--text-muted);">{_ra_label} position</span>'
        f'</div>'
        f'<div class="ts-stat-card" style="flex:1; min-width:160px;">'
        f'<span class="ts-stat-label">Source Power</span>'
        f'<span class="ts-stat-value" style="font-size:1.3rem;">{source_norm:.0f}</span>'
        f'<span style="font-size:0.72rem; color:var(--text-muted);">league avg {source_league_mean:.0f}</span>'
        f'</div>'
        f'<div class="ts-stat-card" style="flex:1; min-width:160px;">'
        f'<span class="ts-stat-label">Target Power</span>'
        f'<span class="ts-stat-value" style="font-size:1.3rem;">{target_norm:.0f}</span>'
        f'<span style="font-size:0.72rem; color:var(--text-muted);">league avg {target_league_mean:.0f}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Opta analyst context cards ───────────────────────────────────────────
    # Form, momentum, and ambition signals — inference-only, not model features.
    _form_cards = []

    def _form_html(label: str, club: str, form_val: float | None, change_7d: str | None) -> str:
        # form_delta card
        if form_val is not None:
            _sign = "+" if form_val >= 0 else ""
            _col = "#2DD4A8" if form_val >= 0 else "#F45B69"
            _desc = "above season avg" if form_val >= 0 else "below season avg"
            form_part = (
                f'<span style="font-size:1.1rem; color:{_col}; font-weight:600;">'
                f'{_sign}{form_val:.1f}</span>'
                f'<span style="font-size:0.72rem; color:var(--text-muted);"> {_desc}</span>'
            )
        else:
            form_part = '<span style="font-size:0.8rem; color:var(--text-muted);">No Opta data</span>'

        # 7-day momentum
        if change_7d and change_7d not in ("0", "None", ""):
            try:
                _delta = int(change_7d)
                _arrow = "↑" if _delta > 0 else "↓"
                _mcol = "#2DD4A8" if _delta > 0 else "#F45B69"
                _places = abs(_delta)
                momentum_part = (
                    f'<span style="color:{_mcol}; font-size:0.75rem;">'
                    f'{_arrow}{_places} place{"s" if _places != 1 else ""} (7d)</span>'
                )
            except (ValueError, TypeError):
                momentum_part = ""
        else:
            momentum_part = '<span style="font-size:0.72rem; color:var(--text-muted);">stable (7d)</span>'

        return (
            f'<div class="ts-stat-card" style="flex:1; min-width:180px;">'
            f'<span class="ts-stat-label">Current Form — {club}</span>'
            f'{form_part}<br/>{momentum_part}'
            f'</div>'
        )

    _src_change_7d = _src_opta.ranking_change_7d if _src_opta else None
    _tgt_change_7d = _tgt_opta.ranking_change_7d if _tgt_opta else None

    _form_html_src = _form_html("Origin", current_team, _src_form, _src_change_7d)
    _form_html_tgt = _form_html("Destination", target_club_display, _tgt_form, _tgt_change_7d)

    # Ambition card (league step-up/down)
    _amb_sign = "+" if _league_step >= 0 else ""
    _amb_col = "#2DD4A8" if _league_step > 2 else ("#F45B69" if _league_step < -2 else "var(--text-muted)")
    _amb_label = "step up" if _league_step > 2 else ("step down" if _league_step < -2 else "lateral")
    _amb_card = (
        f'<div class="ts-stat-card" style="flex:1; min-width:180px;">'
        f'<span class="ts-stat-label">League Step-Up</span>'
        f'<span class="ts-stat-value" style="font-size:1.3rem; color:{_amb_col};">'
        f'{_amb_sign}{_league_step:.1f}</span>'
        f'<span style="font-size:0.72rem; color:{_amb_col};">{_amb_label} '
        f'({_src_opta_league:.0f} → {_tgt_opta_league:.0f})</span>'
        f'</div>'
    )

    st.markdown(
        f'<div style="display:flex; gap:1.5rem; margin:0.6rem 0 1.2rem; flex-wrap:wrap;">'
        f'{_form_html_src}{_form_html_tgt}{_amb_card}'
        f'</div>',
        unsafe_allow_html=True,
    )

    metric_bar.show(current_per90_clean, predicted_target, pct_changes,
                    title=f"Predicted Changes: {player_name} → {target_club_display}")

    # ── Prediction confidence indicator ──────────────────────────────────
    if prediction_confidence:
        avg_std = (
            sum(prediction_confidence.values()) / len(prediction_confidence)
            if prediction_confidence
            else 0
        )
        confidence_level = (
            "green" if avg_std < 0.05
            else ("amber" if avg_std < 0.15 else "red")
        )
        _conf_color = {
            "green": COLORS["accent_green"],
            "amber": COLORS["accent_amber"],
            "red": COLORS["accent_crimson"],
        }.get(confidence_level, COLORS["text_muted"])
        _conf_label = confidence_level.upper()
        st.markdown(
            f'<div class="ts-confidence-badge">'
            f'<span class="ts-confidence-dot" style="background:{_conf_color};'
            f' box-shadow:0 0 8px {_conf_color}40;"></span>'
            f'<span class="ts-confidence-text">'
            f'<strong style="color:{_conf_color};">MODEL CONFIDENCE: {_conf_label}</strong>'
            f' — Avg uncertainty: ±{avg_std:.3f} per 90'
            f'</span></div>',
            unsafe_allow_html=True,
        )

    # ── Summary table — right after metric bars for easy comparison ──────
    _LABELS = {
        "expected_goals": "xG", "expected_assists": "xA", "shots": "Shots",
        "successful_dribbles": "Take-ons", "successful_crosses": "Crosses",
        "touches_in_opposition_box": "Pen. Area Entries",
        "successful_passes": "Total Passes", "pass_completion_pct": "Short Pass %",
        "accurate_long_balls": "Long Passes", "chances_created": "Passes Att 3rd",
        "clearances": "Def Own 3rd", "interceptions": "Def Mid 3rd",
        "possession_won_final_3rd": "Def Att 3rd",
    }

    # Paper Table 1 group labels
    _GROUPS = {
        "expected_goals": "Shooting", "shots": "Shooting",
        "expected_assists": "Passing", "successful_crosses": "Passing",
        "successful_passes": "Passing", "pass_completion_pct": "Passing",
        "accurate_long_balls": "Passing", "chances_created": "Passing",
        "touches_in_opposition_box": "Passing",
        "successful_dribbles": "Dribbling",
        "clearances": "Defending", "interceptions": "Defending",
        "possession_won_final_3rd": "Defending",
    }

    section_header("Detailed Predictions", "Per-90 breakdown across all 13 core metrics (paper Table 1 groups)")
    import pandas as pd
    rows = []
    for m in CORE_METRICS:
        change = pct_changes.get(m, 0)
        row = {
            "Group": _GROUPS.get(m, ""),
            "Metric": _LABELS.get(m, m),
            "Actual (per 90)": round(current_per90_clean.get(m, 0), 3),
            "Simulated Current": round(baseline.get(m, 0), 3),
            "Predicted (per 90)": round(predicted_target.get(m, 0), 3),
            "± Std": round(prediction_confidence[m], 3) if prediction_confidence.get(m) is not None else "—",
            "Change %": round(change, 1),
            "Direction": "📈" if change > 2 else ("📉" if change < -2 else "➡️"),
        }
        rows.append(row)
    df_table = pd.DataFrame(rows)
    try:
        styled = df_table.style.map(
            lambda v: "color: #2DD4A8" if isinstance(v, (int, float)) and v > 0
            else ("color: #F45B69" if isinstance(v, (int, float)) and v < 0 else ""),
            subset=["Change %"],
        )
    except (KeyError, ValueError, TypeError):
        styled = df_table
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
    )

    # ── (b) Power Ranking chart ──────────────────────────────────────────
    section_header("Power Rankings", "Club strength comparison over time")
    today = date.today()

    # Use real historical power rankings when available
    source_history = power_rankings.get_historical_rankings(current_team, months=6)
    target_history = power_rankings.get_historical_rankings(target_club_display, months=6)

    # Fallback to synthetic trend if history is empty
    if not source_history:
        source_history = [(today - timedelta(days=30 * i), source_norm) for i in range(6)]
        source_history.reverse()
    if not target_history:
        target_history = [(today - timedelta(days=30 * i), target_norm) for i in range(6)]
        target_history.reverse()

    power_ranking_chart.show(
        current_team, target_club_display,
        source_history, target_history,
        transfer_date=today,
    )

    # ── League comparison panel ──────────────────────────────────────────
    section_header("League Comparison", "How do the source and target leagues compare?")

    src_league = source_ranking.league_code if source_ranking else None
    tgt_league = target_ranking.league_code if target_ranking else None
    comparison_codes = list(dict.fromkeys(
        [c for c in [src_league, tgt_league] if c]
    ))
    # Always include the big five for context
    for big5 in ["ENG1", "ESP1", "GER1", "ITA1", "FRA1"]:
        if big5 not in comparison_codes:
            comparison_codes.append(big5)

    comparison = power_rankings.compare_leagues(comparison_codes)
    if comparison:
        import pandas as pd
        df_leagues = pd.DataFrame(comparison)
        df_leagues = df_leagues.rename(columns={
            "name": "League",
            "mean_normalized": "Avg Rating",
            "team_count": "Teams",
            "p50": "Median",
            "p10": "P10 (Weakest)",
            "p90": "P90 (Strongest)",
        })
        cols_to_show = ["League", "Avg Rating", "Teams", "Median", "P10 (Weakest)", "P90 (Strongest)"]
        st.dataframe(
            df_leagues[[c for c in cols_to_show if c in df_leagues.columns]],
            use_container_width=True,
            hide_index=True,
        )

    # ── (d) Swarm plots — with real league context data ──────────────────
    section_header("League Context", "Player positioning vs teammates and league")

    # ── League snapshot lookups (null-guarded) ────────────────────────────
    source_snapshot = None
    target_snapshot = None

    if src_league:
        source_snapshot = power_rankings.get_league_snapshot(src_league)
        if source_snapshot is None:
            _log.warning(
                "League snapshot unavailable for source league '%s'", src_league
            )
    else:
        _log.warning(
            "Source league code is None (team: '%s') — cannot look up league snapshot",
            current_team,
        )

    if tgt_league:
        target_snapshot = power_rankings.get_league_snapshot(tgt_league)
        if target_snapshot is None:
            _log.warning(
                "League snapshot unavailable for target league '%s'", tgt_league
            )
    else:
        _log.warning(
            "Target league code is None (team: '%s') — cannot look up league snapshot",
            target_club_display,
        )

    # Show league context summary cards
    if source_snapshot or target_snapshot:
        lc1, lc2 = st.columns(2)
        with lc1:
            if source_snapshot:
                st.metric(
                    f"📊 {source_snapshot.league_name} (origin)",
                    f"{source_snapshot.mean_normalized:.1f}",
                    help="Average league Power Rating (0–100 scale)",
                )
            else:
                st.info("Origin league data unavailable")
        with lc2:
            if target_snapshot:
                st.metric(
                    f"📊 {target_snapshot.league_name} (target)",
                    f"{target_snapshot.mean_normalized:.1f}",
                    help="Average league Power Rating (0–100 scale)",
                )
            else:
                st.info("Target league data unavailable")
    else:
        st.info(
            "League data unavailable — league snapshots could not be loaded "
            "for the origin or target league."
        )
        _log.warning(
            "No league snapshots available: source='%s', target='%s'",
            src_league,
            tgt_league,
        )

    # Reuse teammate data from the position-average fetch where possible
    teammate_per90s: List[Dict] = []
    league_per90s: List[Dict] = []

    if source_pos_players:
        # Already fetched during position average computation
        for tp_stats in source_pos_players:
            p90 = tp_stats.get("per90")
            if p90 and any(v is not None for v in p90.values()):
                teammate_per90s.append(p90)
    elif team_id:
        try:
            team_players = sofascore_client.get_team_players_stats(team_id)
            for tp in team_players[:15]:  # Cap to limit API calls per run
                if tp.get("id") and tp["id"] != player_id:
                    try:
                        tp_stats = sofascore_client.get_player_stats(tp["id"])
                        p90 = tp_stats.get("per90")
                        if p90 and any(v is not None for v in p90.values()):
                            teammate_per90s.append(p90)
                    except Exception as e:
                        _log.warning("Failed to fetch teammate stats: %s", e)
        except Exception as e:
            _log.warning("Failed to fetch team players: %s", e)
    else:
        _log.warning(
            "No team_id available for '%s' — cannot fetch teammate data",
            current_team,
        )

    # Populate league-level data from Sofascore tournament stats
    if tournament_id:
        try:
            league_players = _get_league_players_cached(
                tournament_id, selected_season_id
            )
            for lp in league_players:
                lp_per90 = lp.get("per90")
                if lp_per90 and lp.get("id") != player_id and any(
                    v is not None for v in lp_per90.values()
                ):
                    league_per90s.append(lp_per90)
        except Exception as e:
            _log.warning("Failed to fetch league player stats: %s", e)
    else:
        _log.warning(
            "No tournament_id for player — cannot fetch league player stats"
        )

    if not teammate_per90s and not league_per90s:
        st.warning(
            "⚠️ League context data unavailable — could not load teammate or "
            "league-level per-90 stats. Swarm plots cannot be rendered."
        )
        _log.warning(
            "No teammate or league per-90 data for swarm plots "
            "(team_id=%s, tournament_id=%s)",
            team_id,
            tournament_id,
        )
    else:
        swarm_plot.show_swarm_grid(
            player_name=player_name,
            player_per90=current_per90_clean,
            teammate_per90s=teammate_per90s,
            league_per90s=league_per90s,
        )

    # ── (e) Spatial visualizations (StatsBomb data when available) ────────
    try:
        from backend.data import statsbomb_client
        from frontend.components import pitch_viz

        shots = statsbomb_client.get_player_shots(player_name)
        passes = statsbomb_client.get_player_passes(player_name)
        heatmap_locs = statsbomb_client.get_player_heatmap_data(player_name)

        has_spatial = bool(shots or passes or heatmap_locs)
        if has_spatial:
            section_header(
                "Spatial Profile",
                "Event-level data from StatsBomb open data (where available)",
            )
            viz_cols = st.columns(3)
            with viz_cols[0]:
                if shots:
                    pitch_viz.show_shot_map(shots, player_name=player_name)
                else:
                    st.caption("No shot data available")
            with viz_cols[1]:
                if passes:
                    pitch_viz.show_pass_network(passes, player_name=player_name)
                else:
                    st.caption("No pass data available")
            with viz_cols[2]:
                if heatmap_locs:
                    pitch_viz.show_heatmap(heatmap_locs, player_name=player_name)
                else:
                    st.caption("No heatmap data available")

            # Spatial feature summary (StatsBomb first, WhoScored fallback)
            spatial = statsbomb_client.compute_spatial_features(player_name)
            if not spatial:
                try:
                    from backend.data import reep_registry, whoscored_client
                    reep_data = reep_registry.enrich_player(player_id)
                    ws_id = reep_data.get("whoscored_id")
                    if ws_id:
                        spatial = whoscored_client.compute_spatial_features(ws_id)
                except Exception:
                    pass
            if spatial:
                st.markdown(
                    '<div style="display:flex; gap:1rem; margin:0.8rem 0; flex-wrap:wrap;">',
                    unsafe_allow_html=True,
                )
                _spatial_labels = {
                    "avg_shot_distance": ("Avg Shot Dist", "m"),
                    "shots_inside_box_pct": ("Shots in Box", "%"),
                    "progressive_pass_pct": ("Prog. Passes", "%"),
                    "touches_left_pct": ("Left Third", "%"),
                    "touches_center_pct": ("Center", "%"),
                    "touches_right_pct": ("Right Third", "%"),
                }
                cards = []
                for sk, (label, unit) in _spatial_labels.items():
                    val = spatial.get(sk)
                    if val is not None:
                        fmt = f"{val:.1f}{unit}" if unit == "m" else f"{val:.0f}{unit}"
                        cards.append(
                            f'<div class="ts-stat-card" style="flex:1; min-width:120px;">'
                            f'<span class="ts-stat-label">{label}</span>'
                            f'<span class="ts-stat-value">{fmt}</span></div>'
                        )
                st.markdown("".join(cards) + "</div>", unsafe_allow_html=True)
    except Exception as exc:
        _log.debug("Spatial viz unavailable: %s", exc)
