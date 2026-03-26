"""sklearn LinearRegression adjustment models for team and player priors.

Team adjustment: 13 models (one per core metric).
Player adjustment: 13 models per position.

``build_training_data_from_transfers`` turns Sofascore transfer-history
records into the rows these models expect, enabling auto-training.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

from backend.data.sofascore_client import CORE_METRICS

_log = logging.getLogger(__name__)

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "models",
)


# ── Team adjustment models ───────────────────────────────────────────────────

class TeamAdjustmentModel:
    """13 sklearn LinearRegression models — one per core metric.

    target = naive_league_expectation (offset)
           + beta * team_relative_feature_in_previous_league
           + error
    """

    def __init__(self):
        self.models: Dict[str, LinearRegression] = {}
        self.fitted = False

    def fit(self, training_data: List[Dict[str, Any]]) -> None:
        """Fit all 13 models from historical transfer data.

        Parameters
        ----------
        training_data : list[dict]
            Each dict has:
                - ``metric``: str (metric name)
                - ``team_relative_feature``: float (team's relative per-90
                  in previous league context)
                - ``naive_league_expectation``: float (league-average per-90
                  at target league)
                - ``actual``: float (observed per-90 at new club)
        """
        by_metric: Dict[str, Tuple[List, List]] = {m: ([], []) for m in CORE_METRICS}

        for row in training_data:
            metric = row.get("metric")
            if metric not in by_metric:
                continue
            X_val = row.get("team_relative_feature")
            y_val = row.get("actual")
            offset = row.get("naive_league_expectation", 0)
            if X_val is None or y_val is None:
                continue
            by_metric[metric][0].append([X_val])
            by_metric[metric][1].append(y_val - offset)

        for metric in CORE_METRICS:
            xs, ys = by_metric[metric]
            model = LinearRegression()
            if len(xs) >= 2:
                model.fit(np.array(xs), np.array(ys))
            else:
                # Not enough data — identity model (predict 0 adjustment)
                model.coef_ = np.array([0.0])
                model.intercept_ = 0.0
            self.models[metric] = model

        self.fitted = True

    def predict(
        self,
        team_relative_feature: float,
        naive_expectation: float,
        metric: str,
    ) -> float:
        """Predict adjusted team-level per-90 for one metric.

        Returns naive_expectation + model adjustment.
        """
        if metric not in self.models:
            return naive_expectation
        model = self.models[metric]
        adjustment = model.predict(np.array([[team_relative_feature]]))[0]
        return naive_expectation + adjustment

    def predict_all(
        self,
        team_relative_features: Dict[str, float],
        naive_expectations: Dict[str, float],
    ) -> Dict[str, float]:
        """Predict adjusted per-90 for all 13 core metrics."""
        result = {}
        for metric in CORE_METRICS:
            rel = team_relative_features.get(metric, 0.0)
            naive = naive_expectations.get(metric, 0.0)
            result[metric] = self.predict(rel, naive, metric)
        return result

    def save(self, path: Optional[str] = None) -> str:
        """Save fitted models to disk."""
        if path is None:
            os.makedirs(_MODELS_DIR, exist_ok=True)
            path = os.path.join(_MODELS_DIR, "team_adjustment.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.models, f)
        return path

    def load(self, path: Optional[str] = None) -> None:
        """Load models from disk."""
        if path is None:
            path = os.path.join(_MODELS_DIR, "team_adjustment.pkl")
        with open(path, "rb") as f:
            self.models = pickle.load(f)
        self.fitted = True


# ── Team-position scaling ────────────────────────────────────────────────────

def scale_team_position_features(
    current_position_per90: Dict[str, float],
    current_team_per90: Dict[str, float],
    adjusted_team_per90: Dict[str, float],
) -> Dict[str, float]:
    """Scale team-position features by the same % change as team adjustment.

    If team xG drops 40%, striker xG and CB xG both drop 40%.
    """
    scaled = {}
    for metric in CORE_METRICS:
        current_team_val = current_team_per90.get(metric, 0)
        adjusted_val = adjusted_team_per90.get(metric, 0)
        current_pos_val = current_position_per90.get(metric, 0)

        if current_team_val != 0:
            pct_change = adjusted_val / current_team_val
        else:
            pct_change = 1.0

        scaled[metric] = current_pos_val * pct_change

    return scaled


# ── Player adjustment models ────────────────────────────────────────────────

class PlayerAdjustmentModel:
    """13 sklearn LinearRegression models per position.

    target = intercept
           + b1 * player_previous_per90
           + b2 * avg_position_feature_new_team
           + b3 * diff_avg_position_feature_old_vs_new_team
           + b4 * change_in_relative_ability
           + b5 * change_in_relative_ability^2
           + b6 * change_in_relative_ability^3
           + error
    """

    def __init__(self):
        self.models: Dict[str, Dict[str, LinearRegression]] = {}
        self.fitted = False

    def fit(self, training_data: List[Dict[str, Any]]) -> None:
        """Fit models from historical player transfer data.

        Parameters
        ----------
        training_data : list[dict]
            Each dict has:
                - ``position``: str
                - ``metric``: str
                - ``player_previous_per90``: float
                - ``avg_position_feature_new_team``: float
                - ``diff_avg_position_old_vs_new``: float
                - ``change_relative_ability``: float
                - ``actual``: float (observed per-90 at new club)
        """
        grouped: Dict[str, Dict[str, Tuple[List, List]]] = {}

        for row in training_data:
            pos = row.get("position", "Unknown")
            metric = row.get("metric")
            if metric not in CORE_METRICS:
                continue

            grouped.setdefault(pos, {}).setdefault(metric, ([], []))

            cra = row.get("change_relative_ability", 0)
            features = [
                row.get("player_previous_per90", 0),
                row.get("avg_position_feature_new_team", 0),
                row.get("diff_avg_position_old_vs_new", 0),
                cra,
                cra ** 2,
                cra ** 3,
            ]
            actual = row.get("actual")
            if actual is None:
                continue

            grouped[pos][metric][0].append(features)
            grouped[pos][metric][1].append(actual)

        for pos, metrics in grouped.items():
            self.models[pos] = {}
            for metric in CORE_METRICS:
                model = LinearRegression()
                if metric in metrics and len(metrics[metric][0]) >= 2:
                    xs, ys = metrics[metric]
                    model.fit(np.array(xs), np.array(ys))
                else:
                    model.coef_ = np.zeros(6)
                    model.intercept_ = 0.0
                self.models[pos][metric] = model

        self.fitted = True

    def predict(
        self,
        position: str,
        metric: str,
        player_previous_per90: float,
        avg_position_feature_new_team: float,
        diff_avg_position_old_vs_new: float,
        change_relative_ability: float,
    ) -> float:
        """Predict a player's per-90 at the target club for one metric."""
        cra = change_relative_ability
        features = np.array([[
            player_previous_per90,
            avg_position_feature_new_team,
            diff_avg_position_old_vs_new,
            cra,
            cra ** 2,
            cra ** 3,
        ]])

        if position in self.models and metric in self.models[position]:
            return float(self.models[position][metric].predict(features)[0])

        # Fallback: try generic position or return current per-90
        if "Unknown" in self.models and metric in self.models["Unknown"]:
            return float(self.models["Unknown"][metric].predict(features)[0])

        return player_previous_per90

    def predict_all(
        self,
        position: str,
        player_per90: Dict[str, float],
        avg_position_new: Dict[str, float],
        avg_position_old: Dict[str, float],
        change_relative_ability: float,
    ) -> Dict[str, float]:
        """Predict all 13 core metrics for a player transfer."""
        result = {}
        for metric in CORE_METRICS:
            pp = player_per90.get(metric, 0)
            apn = avg_position_new.get(metric, 0)
            diff = apn - avg_position_old.get(metric, 0)
            result[metric] = self.predict(
                position, metric, pp, apn, diff, change_relative_ability
            )
        return result

    def save(self, path: Optional[str] = None) -> str:
        if path is None:
            os.makedirs(_MODELS_DIR, exist_ok=True)
            path = os.path.join(_MODELS_DIR, "player_adjustment.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.models, f)
        return path

    def load(self, path: Optional[str] = None) -> None:
        if path is None:
            path = os.path.join(_MODELS_DIR, "player_adjustment.pkl")
        with open(path, "rb") as f:
            self.models = pickle.load(f)
        self.fitted = True


# ── Auto-training from transfer history ──────────────────────────────────────


def build_training_data_from_transfers(
    player_id: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build training rows for the Team and Player adjustment models.

    Uses the player's Sofascore transfer history to find clubs where the
    player has stats both *before* and *after* a move.  For each such
    transfer the function emits training rows (one per core metric).

    Returns
    -------
    (team_rows, player_rows)
        team_rows:  list[dict] suitable for ``TeamAdjustmentModel.fit``
        player_rows: list[dict] suitable for ``PlayerAdjustmentModel.fit``
    """
    from backend.data import sofascore_client
    from backend.features import power_rankings

    transfers = sofascore_client.get_player_transfer_history(player_id)
    if not transfers:
        return [], []

    team_rows: List[Dict[str, Any]] = []
    player_rows: List[Dict[str, Any]] = []

    # We need at least two entries (a departure and an arrival) to build a pair
    if len(transfers) < 2:
        return [], []

    # Use the most recent transfer: transfers[0] = arrival at new club,
    # transfers[1] = previous club context.
    to_transfer = transfers[0]
    from_transfer = transfers[1]

    new_team = to_transfer.get("to_team") or {}
    old_team = from_transfer.get("to_team") or from_transfer.get("from_team") or {}

    new_team_id = new_team.get("id")
    old_team_id = old_team.get("id")
    new_team_name = new_team.get("name", "")
    old_team_name = old_team.get("name", "")

    if not new_team_id or not old_team_id:
        return [], []

    try:
        new_stats = sofascore_client.get_player_stats(player_id)
    except Exception:
        return [], []

    # Power rankings for relative ability
    new_ranking = power_rankings.get_team_ranking(new_team_name)
    old_ranking = power_rankings.get_team_ranking(old_team_name)

    if new_ranking is None or old_ranking is None:
        return [], []

    change_ra = new_ranking.relative_ability - old_ranking.relative_ability
    new_per90 = new_stats.get("per90") or {}
    position = new_stats.get("position", "Unknown")

    for metric in CORE_METRICS:
        actual = new_per90.get(metric)
        if actual is None:
            continue

        # Team adjustment row
        team_rows.append({
            "metric": metric,
            "team_relative_feature": new_ranking.relative_ability,
            "naive_league_expectation": new_ranking.league_mean_normalized,
            "actual": actual,
        })

        # Player adjustment row
        player_rows.append({
            "position": position,
            "metric": metric,
            "player_previous_per90": actual,  # approximation
            "avg_position_feature_new_team": actual,
            "diff_avg_position_old_vs_new": 0.0,
            "change_relative_ability": change_ra,
            "actual": actual,
        })

    return team_rows, player_rows


def auto_train_from_player_history(
    player_ids: List[int],
) -> Tuple[TeamAdjustmentModel, PlayerAdjustmentModel]:
    """Convenience function: collect training data and fit both models.

    Parameters
    ----------
    player_ids : list[int]
        Sofascore player IDs with transfer histories.

    Returns
    -------
    (team_model, player_model) — both fitted.
    """
    all_team_rows: List[Dict[str, Any]] = []
    all_player_rows: List[Dict[str, Any]] = []

    for pid in player_ids:
        try:
            t_rows, p_rows = build_training_data_from_transfers(pid)
            all_team_rows.extend(t_rows)
            all_player_rows.extend(p_rows)
        except Exception as exc:
            _log.warning("Failed to build training data for player %s: %s", pid, exc)

    team_model = TeamAdjustmentModel()
    team_model.fit(all_team_rows)

    player_model = PlayerAdjustmentModel()
    player_model.fit(all_player_rows)

    return team_model, player_model


# ── Paper-aligned heuristic prediction ────────────────────────────────────────

# Per-metric team influence: how much a metric is driven by team tactics
# vs individual skill (paper Sections 2.3, 4.2).
# Higher = more team-dependent (adapts more to new team's style).
# Lower = more individual (retains player's own level).
_TEAM_INFLUENCE: Dict[str, float] = {
    "expected_goals": 0.35,           # Team creates chances, moderate influence
    "expected_assists": 0.40,         # Passing system drives assists
    "shots": 0.30,                    # Team creates shooting situations
    "successful_dribbles": 0.15,      # Highly individual, "irreducible" per paper
    "successful_crosses": 0.45,       # Tactical role dictates crossing volume
    "touches_in_opposition_box": 0.50,  # Team attacking approach drives box presence
    "successful_passes": 0.50,        # Passing style is heavily tactical
    "pass_completion_pct": 0.35,      # Mix of individual quality + system
    "accurate_long_balls": 0.45,      # Tactical approach to build-up play
    "chances_created": 0.40,          # Team's attacking system
    "clearances": 0.50,               # Defensive approach/press height
    "interceptions": 0.45,            # Press intensity and defensive line
    "possession_won_final_3rd": 0.35,  # Press style, moderate team influence
}

# Per-metric sensitivity to relative ability changes (paper Section 2.2).
# Offensive metrics scale positively (better team → more output).
# Defensive workload scales inversely (better team → less defending).
_ABILITY_SENSITIVITY: Dict[str, float] = {
    "expected_goals": 0.50,
    "expected_assists": 0.42,         # Slightly higher than shots — creative output scales more
    "shots": 0.38,                    # Volume metric, moderate scaling
    "successful_dribbles": 0.15,      # Barely affected by team quality
    "successful_crosses": 0.30,
    "touches_in_opposition_box": 0.50,
    "successful_passes": 0.22,        # Possession teams pass more, but also individual
    "pass_completion_pct": 0.10,
    "accurate_long_balls": 0.18,      # Direct play slightly decreases at top teams
    "chances_created": 0.45,          # Creative output scales strongly with team quality
    "clearances": -0.40,              # Better team → less clearances
    "interceptions": -0.30,           # Better team → less interceptions
    "possession_won_final_3rd": 0.25, # Better team → more high press
}


# Per-metric coefficients for estimating how team-position averages vary
# with relative ability when real team-position data is unavailable.
# Used by paper_heuristic_predict() (line ~552) as a fallback style estimator.
# Derived from the paper's observations (Sections 4.2, 4.3):
# - Stronger teams: higher attacking outputs, lower defensive workload
# - Different metrics respond to team quality at different rates
# - Individual skills (dribbling) barely change between leagues
_LEAGUE_STYLE_COEFF: Dict[str, float] = {
    "expected_goals": 0.12,           # Team creates more chances
    "expected_assists": 0.18,         # Better passing systems produce more xA
    "shots": 0.08,                    # More attacking play at stronger teams
    "successful_dribbles": 0.02,      # Near-pure individual skill
    "successful_crosses": 0.11,       # Depends on team's width of play
    "touches_in_opposition_box": 0.14,  # More box presence at attacking teams
    "successful_passes": 0.07,        # More possession at stronger teams
    "pass_completion_pct": 0.03,      # Slightly better pass accuracy
    "accurate_long_balls": -0.04,     # Direct teams use MORE long balls (often weaker)
    "chances_created": 0.16,          # Team attacking quality drives this
    "clearances": -0.14,              # Less defending at stronger teams
    "interceptions": -0.09,           # Less defensive work needed
    "possession_won_final_3rd": 0.09,  # More high pressing at stronger teams
}


# Per-metric sensitivity to opposition quality (paper Section 4.3, Appendix A.3).
#
# The paper's trained model receives league_ability_current and
# league_ability_target as separate features (build_feature_dict), so it
# learns that moving to a weaker league means facing weaker opposition,
# which boosts offensive per-90 output.  Our heuristic only receives
# change_relative_ability = (tgt_team−tgt_league) − (src_team−src_league),
# which collapses the absolute league quality gap.
#
# This dict restores the paper's opposition-quality signal:
#   league_gap = (source_league_mean − target_league_mean) / 100
#   opposition_boost = player_val * _OPP_QUALITY_SENS[m] * league_gap
#
# Paper evidence (Section 4.3.1):
#   - Doku xG INCREASES at Gwangju (much weaker league) despite weak team
#   - Doku take-ons barely change (individual / "irreducible" per paper)
#   - Final-third passes drop slightly but Doku still in 73rd percentile
#   - Mbappé: 8-15% decrease moving to HARDER league (La Liga)
#
# Positive values: weaker opposition → higher per-90 output.
# Near-zero: individual metrics unaffected by opposition quality.
# Negative: defensive metrics increase when facing weaker opposition
#           (team has less defending to do).
_OPP_QUALITY_SENS: Dict[str, float] = {
    "expected_goals": 0.55,           # Weaker defenders/GK → more xG per 90
    "expected_assists": 0.35,         # Less pressing → more creative freedom
    "shots": 0.40,                    # More space → more shooting opportunities
    "successful_dribbles": 0.08,      # "Irreducible" — barely affected by opposition
    "successful_crosses": 0.25,       # Somewhat more space on the flanks
    "touches_in_opposition_box": 0.45,  # More dominant possession in box
    "successful_passes": 0.15,        # Slightly easier passing against weaker press
    "pass_completion_pct": 0.08,      # Marginal improvement
    "accurate_long_balls": 0.05,      # Barely affected
    "chances_created": 0.38,          # More creative opportunities vs weaker defence
    "clearances": -0.25,              # Less defending when opposition is weaker
    "interceptions": -0.20,           # Less defending needed
    "possession_won_final_3rd": 0.15,  # Easier to win ball vs weaker opposition
}


def paper_heuristic_predict(
    player_per90: Dict[str, float],
    source_pos_avg: Dict[str, float],
    target_pos_avg: Dict[str, float],
    change_relative_ability: float,
    player_rating: Optional[float] = None,
) -> Dict[str, float]:
    """Paper-aligned heuristic when no trained model is available.

    Mirrors the structure of the PlayerAdjustmentModel (paper Appendix A.3)
    but with reasonable default coefficients instead of trained weights.

    For each metric the prediction uses:

    1. **Style shift** — difference between target and source team's position
       averages, weighted by how team-influenced the metric is.  When real
       team-position data is unavailable (both averages are the same),
       estimates per-metric style differences from the league quality gap
       so that metrics respond differently rather than changing by the same
       flat percentage.
    2. **Ability adjustment** — polynomial in the change of relative ability,
       with per-metric sensitivity.  This captures the paper's observation
       that offensive output scales with team quality while defensive
       workload decreases.
    3. **Player quality modifier** — when a Sofascore average match rating
       is available, it modulates how much the player adapts vs retains
       their own level.  Elite players (rating >= 7.5) retain more of
       their individual output; weaker players are more team-dependent.

    Parameters
    ----------
    player_per90 : dict
        Player's current per-90 values (13 core metrics).
    source_pos_avg : dict
        Average per-90 for the player's position at the **current** team.
    target_pos_avg : dict
        Average per-90 for the player's position at the **target** team.
    change_relative_ability : float
        ``(target_norm - target_league_mean) - (source_norm - source_league_mean)``
        Positive means moving to a relatively stronger position.
    player_rating : float, optional
        Sofascore average match rating (0-10 scale).  Higher-rated players
        retain more of their individual output when changing teams.

    Returns
    -------
    dict[str, float] — predicted per-90 values at the target club.
    """
    # Scale to roughly [-1, 1] for typical transfers; extreme league
    # jumps can exceed this range (e.g. ±1.5).
    ra = change_relative_ability / 100.0

    # Detect whether we have real team-position data or just fallback
    # (both source and target are the same → no style data available).
    _has_style_data = _check_has_style_data(player_per90, source_pos_avg, target_pos_avg)

    # Player quality modifier: elite players retain more individual output.
    # Centered at 6.5 (average Sofascore rating), scaled multiplicatively so
    # that higher-rated players have reduced (but never zero) team influence.
    # e.g. rating=7.5 → quality_scale=0.85 (15% less team-dependent)
    #      rating=6.0 → quality_scale=1.075 (slightly more team-dependent)
    quality_scale = 1.0
    if player_rating is not None and isinstance(player_rating, (int, float)):
        raw_mod = (player_rating - 6.5) * 0.15
        # Clamp so team influence is scaled between 0.7x and 1.3x
        quality_scale = max(0.7, min(1.3, 1.0 - raw_mod))

    predicted: Dict[str, float] = {}
    for m in CORE_METRICS:
        player_val = player_per90.get(m, 0.0)
        src_avg = source_pos_avg.get(m, player_val)
        tgt_avg = target_pos_avg.get(m, player_val)

        # 1. Style shift: how much does the target team's tactical use
        #    of this position differ from the source team?
        style_diff = tgt_avg - src_avg
        team_inf = _TEAM_INFLUENCE.get(m, 0.3)

        # When no real team-position data is available, estimate style
        # differences from the league quality gap (paper Section 4.2).
        # Each metric gets a unique coefficient so that metrics with the
        # same ability sensitivity still produce different predictions.
        if not _has_style_data:
            league_coeff = _LEAGUE_STYLE_COEFF.get(m, 0.05)
            estimated_style_diff = src_avg * league_coeff * ra
            style_diff = estimated_style_diff

        # Modulate team influence by player quality: elite players are
        # less team-dependent (lower effective team_inf), multiplicatively.
        effective_team_inf = team_inf * quality_scale

        # Player-system-reliance scaling (paper-aligned correction).
        #
        # The paper's trained b3 coefficient on diff_pos_old_new
        # implicitly learns how much of the team-average change applies
        # to an individual player.  Our fixed coefficient (team_inf) is
        # the same for all players, so we need an explicit correction.
        #
        # A player performing well BELOW their team's positional average
        # was less boosted by the tactical system — their output is more
        # individual.  They lose less when the team system changes.
        # Example: a 0.15 xG/90 forward at Man City (avg 0.40) is at 37%
        # of team level — the style shift should be scaled to 37%, not 100%.
        #
        # Conversely, a player at or above the team average gets the full
        # shift (capped at 1.0 to avoid over-amplifying star players).
        if abs(src_avg) > 0.01:
            system_reliance = min(1.0, max(0.3, player_val / src_avg))
        else:
            system_reliance = 1.0

        # Base prediction: player's stats shifted toward target team's style,
        # scaled by how much the player relied on the source team's system.
        base = player_val + effective_team_inf * style_diff * system_reliance

        # 2. Ability adjustment: polynomial (linear + quadratic + cubic).
        #    Maps to paper Appendix A.3: β4*ra + β5*ra² + β6*ra³
        sensitivity = _ABILITY_SENSITIVITY.get(m, 0.2)
        ability_factor = (
            1.0
            + sensitivity * ra                    # linear term
            - 0.15 * sensitivity * (ra ** 2)      # dampens extremes
            + 0.02 * sensitivity * (ra ** 3)      # asymmetric tails
        )

        pred = base * ability_factor
        predicted[m] = max(pred, 0.0)  # per-90 can't be negative

    return predicted


def _check_has_style_data(
    player_per90: Dict[str, float],
    source_pos_avg: Dict[str, float],
    target_pos_avg: Dict[str, float],
) -> bool:
    """Return True if source/target position averages carry real team style info.

    Returns False when both position average dicts are empty, all zeros,
    or effectively equal to the player's own stats (indicating fallback
    data rather than genuine team-position data).
    """
    diffs = 0
    for m in CORE_METRICS:
        src = source_pos_avg.get(m, 0.0)
        tgt = target_pos_avg.get(m, 0.0)
        player = player_per90.get(m, 0.0)

        # Check if source and target are genuinely different from each other
        if abs(src - tgt) > 1e-6:
            diffs += 1
        # Or at least different from the player's own stats
        elif abs(src - player) > 1e-6 or abs(tgt - player) > 1e-6:
            diffs += 1

    # Need at least 2 metrics with different averages to count as real data
    return diffs >= 2
