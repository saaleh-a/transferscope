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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

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

    Paper Appendix A.1 formulation:

        y_{i,j} = x_{i,j} + α_j + β_j · z_{i,j} + ε_{i,j}

    Where:
        x_{i,j} = naive league expectation (target league mean per-90 for
                   metric j).  Used as an offset so the model only needs to
                   learn the *adjustment*.
        z_{i,j} = team's relative feature value in previous league.  This is
                   a *per-metric* signal: the source team's actual per-90 for
                   metric j relative to the source league distribution.

    The model also accepts the legacy ``from_ra`` / ``to_ra`` features
    (Elo-derived relative ability) as supplementary inputs.  When
    ``team_relative_feature`` is provided in training data, it is used
    as the primary feature (paper-aligned).  Legacy features are included
    alongside for backward compatibility.
    """

    # Number of features: team_relative_feature + from_ra + to_ra
    _N_FEATURES = 3

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
                - ``from_ra``: float (source team relative ability)
                - ``to_ra``: float (target team relative ability)
                - ``naive_league_expectation``: float (league-average per-90
                  at target league)
                - ``actual``: float (observed per-90 at new club)
                - ``team_relative_feature``: float, optional (paper A.1 z_{i,j}
                  — source team's per-90 for this metric relative to source
                  league distribution; 0.0 when unavailable)
        """
        by_metric: Dict[str, Tuple[List, List]] = {m: ([], []) for m in CORE_METRICS}

        for row in training_data:
            metric = row.get("metric")
            if metric not in by_metric:
                continue
            from_ra = row.get("from_ra")
            to_ra = row.get("to_ra")
            y_val = row.get("actual")
            offset = row.get("naive_league_expectation", 0)
            # Paper A.1: z_{i,j} — per-metric team feature relative to league
            team_rel = row.get("team_relative_feature", 0.0)
            if from_ra is None or to_ra is None or y_val is None:
                continue
            by_metric[metric][0].append([team_rel, from_ra, to_ra])
            by_metric[metric][1].append(y_val - offset)

        for metric in CORE_METRICS:
            xs, ys = by_metric[metric]
            model = LinearRegression()
            if len(xs) >= 2:
                model.fit(np.array(xs), np.array(ys))
            else:
                # Not enough data — identity model (predict 0 adjustment)
                model.coef_ = np.array([0.0] * self._N_FEATURES)
                model.intercept_ = 0.0
            self.models[metric] = model

        self.fitted = True

    def predict(
        self,
        from_ra: float,
        to_ra: float,
        naive_expectation: float,
        metric: str,
        team_relative_feature: float = 0.0,
    ) -> float:
        """Predict adjusted team-level per-90 for one metric.

        Returns naive_expectation + model adjustment.
        """
        if metric not in self.models:
            return naive_expectation
        model = self.models[metric]
        features = np.array([[team_relative_feature, from_ra, to_ra]])
        # Guard against models fitted with legacy 2-feature format
        if hasattr(model, "coef_") and len(model.coef_) == 2:
            features = np.array([[from_ra, to_ra]])
        adjustment = model.predict(features)[0]
        return naive_expectation + adjustment

    def predict_all(
        self,
        from_ra: float,
        to_ra: float,
        naive_expectations: Dict[str, float],
        team_relative_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Predict adjusted per-90 for all 13 core metrics."""
        result = {}
        for metric in CORE_METRICS:
            naive = naive_expectations.get(metric, 0.0)
            trf = 0.0
            if team_relative_features is not None:
                trf = team_relative_features.get(metric, 0.0)
            result[metric] = self.predict(from_ra, to_ra, naive, metric, trf)
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

_PLAYER_MIN_SAMPLES = 30  # Minimum samples per position/metric for fitting
_PLAYER_TARGET_STD_THRESHOLD = 0.01  # Skip fitting if target std is below this


class PlayerAdjustmentModel:
    """13 sklearn Ridge models per position.

    Uses Ridge regression (alpha=1.0) instead of OLS to handle
    multicollinearity from the polynomial change_relative_ability
    features (cra, cra², cra³).

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
        self.models: Dict[str, Dict[str, Ridge]] = {}
        self._scalers: Dict[str, Dict[str, StandardScaler]] = {}
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
                - ``change_relative_ability``: float (already normalised)
                - ``actual``: float (observed per-90 at new club)
        """
        grouped: Dict[str, Dict[str, Tuple[List, List]]] = {}

        for row in training_data:
            pos = row.get("position", "Unknown")
            metric = row.get("metric")
            if metric not in CORE_METRICS:
                continue

            grouped.setdefault(pos, {}).setdefault(metric, ([], []))

            # change_relative_ability is already normalised by the caller
            # (training_pipeline divides by 50); do NOT divide again.
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
            self._scalers[pos] = {}
            for metric in CORE_METRICS:
                model = Ridge(alpha=1.0)
                scaler = StandardScaler()
                if (
                    metric in metrics
                    and len(metrics[metric][0]) >= _PLAYER_MIN_SAMPLES
                ):
                    xs, ys = metrics[metric]
                    X_arr = np.array(xs)
                    y_arr = np.array(ys)
                    _log.info(
                        "Position %s metric %s: %d samples",
                        pos, metric, len(xs),
                    )
                    # Near-zero variance in target → constant prediction fallback
                    if np.std(y_arr) < _PLAYER_TARGET_STD_THRESHOLD:
                        _log.info(
                            "Position %s metric %s: target std=%.6f < %.4f, "
                            "skipping regression (constant prediction fallback)",
                            pos, metric, np.std(y_arr),
                            _PLAYER_TARGET_STD_THRESHOLD,
                        )
                        model.coef_ = np.zeros(6)
                        model.intercept_ = float(np.mean(y_arr))
                        scaler = None  # type: ignore[assignment]
                    else:
                        X_scaled = scaler.fit_transform(X_arr)
                        model.fit(X_scaled, y_arr)
                elif metric in metrics and len(metrics[metric][0]) >= 2:
                    # Too few samples for reliable regression; log a warning
                    # and fall back to identity (predict player_previous_per90).
                    _log.info(
                        "Position %s metric %s: only %d samples (< %d), "
                        "skipping regression (will use identity fallback)",
                        pos, metric, len(metrics[metric][0]),
                        _PLAYER_MIN_SAMPLES,
                    )
                    model.coef_ = np.zeros(6)
                    model.intercept_ = 0.0
                    scaler = None  # type: ignore[assignment]
                else:
                    model.coef_ = np.zeros(6)
                    model.intercept_ = 0.0
                    scaler = None  # type: ignore[assignment]
                self.models[pos][metric] = model
                self._scalers[pos][metric] = scaler

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
        cra = change_relative_ability / 50.0
        raw_features = np.array([[
            player_previous_per90,
            avg_position_feature_new_team,
            diff_avg_position_old_vs_new,
            cra,
            cra ** 2,
            cra ** 3,
        ]])

        if position in self.models and metric in self.models[position]:
            scaler = self._scalers.get(position, {}).get(metric)
            if scaler is not None:
                features = scaler.transform(raw_features)
            else:
                features = raw_features
            return float(self.models[position][metric].predict(features)[0])

        # Fallback: try generic position or return current per-90
        if "Unknown" in self.models and metric in self.models["Unknown"]:
            scaler = self._scalers.get("Unknown", {}).get(metric)
            if scaler is not None:
                features = scaler.transform(raw_features)
            else:
                features = raw_features
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
            pickle.dump({"models": self.models, "scalers": self._scalers}, f)
        return path

    def load(self, path: Optional[str] = None) -> None:
        if path is None:
            path = os.path.join(_MODELS_DIR, "player_adjustment.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Support both old format (dict of models) and new format
        if isinstance(data, dict) and "models" in data:
            self.models = data["models"]
            self._scalers = data.get("scalers", {})
        else:
            # Legacy format: data is the models dict directly
            self.models = data
            self._scalers = {}
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

        # Team adjustment row.
        # naive_league_expectation is the league-average per-90 for this metric,
        # used as the regression offset in TeamAdjustmentModel.  When building
        # training data from a single player's transfer history we don't have the
        # full league-average per-90 distribution, so we use 0.0 (neutral) rather
        # than league_mean_normalized which is an Elo score on a 0–100 scale and
        # would produce nonsensical regression targets (e.g. 0.35 xG - 65.0 = -64.65).
        team_rows.append({
            "metric": metric,
            "from_ra": old_ranking.relative_ability,
            "to_ra": new_ranking.relative_ability,
            "naive_league_expectation": 0.0,
            "team_relative_feature": 0.0,  # unavailable from single-player history
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
#
# Calibration evidence from paper:
#   - De Jong to Man Utd: up to 50% drop in creative metrics — team style
#     dominance for passing/creative metrics (paper Section 5)
#   - Doku take-ons: barely change across destinations — "irreducible"
#     individual skill (paper Section 4.3.1, Fig 15e)
#   - Sterling to Barcelona: pattern carries over — team-relative production
#     retained (paper Section 5)
_TEAM_INFLUENCE: Dict[str, float] = {
    "expected_goals": 0.40,           # Team creates chances, moderate influence
    "expected_assists": 0.50,         # Passing system drives assists strongly
    "shots": 0.35,                    # Team creates shooting situations
    "successful_dribbles": 0.15,      # Highly individual, "irreducible" per paper
    "successful_crosses": 0.55,       # Tactical role dictates crossing volume
    "touches_in_opposition_box": 0.55,  # Team attacking approach drives box presence
    "successful_passes": 0.60,        # Passing style is heavily tactical
    "pass_completion_pct": 0.40,      # Mix of individual quality + system
    "accurate_long_balls": 0.50,      # Tactical approach to build-up play
    "chances_created": 0.50,          # Team's attacking system
    "clearances": 0.55,               # Defensive approach/press height
    "interceptions": 0.50,            # Press intensity and defensive line
    "possession_won_final_3rd": 0.40,  # Press style, moderate team influence
}

# Per-metric sensitivity to relative ability changes (paper Section 2.2).
# Offensive metrics scale positively (better team → more output).
# Defensive workload scales inversely (better team → less defending).
#
# Calibration targets from paper:
#   - Mbappé to Real Madrid: 8-15% decrease (Ligue 1 → La Liga, Section 5)
#   - Healey to Brighton: ~30% drop in shooting (Ligue 2 → PL, Section 5)
#   - Adeyemi to Dortmund: top 20% → top 33% (Austrian BL → Bundesliga, Section 5)
#   - De Jong to Man Utd: up to 50% in creative (large style gap, Section 5)
#   - Doku take-ons: near-zero change ("irreducible", Section 4.3.1)
_ABILITY_SENSITIVITY: Dict[str, float] = {
    "expected_goals": 0.82,
    "expected_assists": 0.70,         # Creative output scales strongly with team quality
    "shots": 0.62,                    # Volume metric, good scaling
    "successful_dribbles": 0.18,      # Barely affected by team quality — "irreducible"
    "successful_crosses": 0.50,
    "touches_in_opposition_box": 0.85,
    "successful_passes": 0.35,        # Possession teams pass more, but also individual
    "pass_completion_pct": 0.15,
    "accurate_long_balls": 0.28,      # Direct play slightly decreases at top teams
    "chances_created": 0.75,          # Creative output scales strongly with team quality
    "clearances": -0.65,              # Better team → much less clearances
    "interceptions": -0.50,           # Better team → less interceptions
    "possession_won_final_3rd": 0.40, # Better team → more high press
}


# Per-metric coefficients for estimating how team-position averages vary
# with relative ability when real team-position data is unavailable.
# Used by paper_heuristic_predict() (line ~552) as a fallback style estimator.
# Derived from the paper's observations (Sections 4.2, 4.3):
# - Stronger teams: higher attacking outputs, lower defensive workload
# - Different metrics respond to team quality at different rates
# - Individual skills (dribbling) barely change between leagues
#
# Increased from prior values to produce meaningful per-metric differentiation
# when real team-position data is unavailable.  Without these, all 13 metrics
# would change by nearly identical percentages (flat, uninformative).
_LEAGUE_STYLE_COEFF: Dict[str, float] = {
    "expected_goals": 0.30,           # Team creates more chances
    "expected_assists": 0.40,         # Better passing systems produce more xA
    "shots": 0.20,                    # More attacking play at stronger teams
    "successful_dribbles": 0.04,      # Near-pure individual skill
    "successful_crosses": 0.25,       # Depends on team's width of play
    "touches_in_opposition_box": 0.30,  # More box presence at attacking teams
    "successful_passes": 0.15,        # More possession at stronger teams
    "pass_completion_pct": 0.06,      # Slightly better pass accuracy
    "accurate_long_balls": -0.08,     # Direct teams use MORE long balls (often weaker)
    "chances_created": 0.35,          # Team attacking quality drives this
    "clearances": -0.30,              # Less defending at stronger teams
    "interceptions": -0.20,           # Less defensive work needed
    "possession_won_final_3rd": 0.20,  # More high pressing at stronger teams
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
# Paper evidence (Section 4.3.1, Section 5):
#   - Doku xG INCREASES at Gwangju (much weaker league) despite weak team
#   - Doku take-ons barely change (individual / "irreducible" per paper)
#   - Mbappé: 8-15% decrease moving to HARDER league (La Liga)
#   - Healey: ~30% drop in shooting metrics (Ligue 2 → Premier League)
#   - Mooy: top metrics "hold up well" despite weaker → stronger league
#
# Positive values: weaker opposition → higher per-90 output.
# Near-zero: individual metrics unaffected by opposition quality.
# Negative: defensive metrics increase when facing weaker opposition
#           (team has less defending to do).
_OPP_QUALITY_SENS: Dict[str, float] = {
    "expected_goals": 1.30,           # Weaker defenders/GK → significantly more xG per 90
    "expected_assists": 0.85,         # Less pressing → more creative freedom
    "shots": 1.00,                    # More space → more shooting opportunities
    "successful_dribbles": 0.12,      # "Irreducible" — barely affected by opposition
    "successful_crosses": 0.60,       # More space on the flanks
    "touches_in_opposition_box": 1.10,  # More dominant possession in box
    "successful_passes": 0.30,        # Slightly easier passing against weaker press
    "pass_completion_pct": 0.15,      # Marginal improvement
    "accurate_long_balls": 0.08,      # Barely affected
    "chances_created": 0.90,          # More creative opportunities vs weaker defence
    "clearances": -0.55,              # Less defending when opposition is weaker
    "interceptions": -0.45,           # Less defending needed
    "possession_won_final_3rd": 0.30,  # Easier to win ball vs weaker opposition
}


# ── Structural parameters for paper_heuristic_predict ─────────────────────────
# These control the formula shape, not per-metric behavior. Extracted as named
# constants for clarity and future tuning (code review feedback).

# Paper A.3 β2 coefficient: how strongly a player conforms to the new team's
# position-level averages. Higher = more adaptation to new team's style.
# Increased from 0.15 to 0.25 to produce paper-scale style effects.
_CONFORMITY_COEFF = 0.25

# Quadratic damping on the team quality polynomial (paper A.3 x4² term).
# Prevents unrealistically large effects for extreme transfers.
# Now asymmetric: less damping for downgrades (large drops are realistic —
# paper shows 30-50% changes for extreme moves like de Jong Barça→Man Utd).
# More damping for upgrades (ceiling effect — player can't exceed their talent).
_DAMPING_FACTOR_DOWN = 0.05   # moving to weaker team: allow bigger drops
_DAMPING_FACTOR_UP = 0.10     # moving to stronger team: modest ceiling

# League-quality attenuation: scales down style_diff for cross-league moves
# where position-average differences reflect league quality, not tactics.
# Factor: how aggressively style is attenuated per unit of league gap.
# Floor: minimum style influence even for extreme cross-league moves.
# Reduced from (2.0, 0.15) to (1.5, 0.25) — paper shows style effects persist
# in cross-league moves (Doku's xA differs at Liverpool vs Barcelona, Section 4.3.1).
_LEAGUE_ATTN_FACTOR = 1.5
_LEAGUE_ATTN_FLOOR = 0.25


def paper_heuristic_predict(
    player_per90: Dict[str, float],
    source_pos_avg: Dict[str, float],
    target_pos_avg: Dict[str, float],
    change_relative_ability: float,
    player_rating: Optional[float] = None,
    source_league_mean: Optional[float] = None,
    target_league_mean: Optional[float] = None,
) -> Dict[str, float]:
    """Paper-aligned heuristic when no trained model is available.

    Faithfully mirrors the structure of the Player Adjustment Model
    (paper Appendix A.3) using the same four input signals:

    1. **Player's previous per-90** (x1) — the player's current output,
       weighted to retain individual quality.
    2. **New team position average** (x2) — the player will partly conform
       to how the target team uses players in this position.
    3. **Position-average difference old→new** (x3) — the style shift
       between source and target team's tactical use of this position.
    4. **Change in relative ability** (x4, polynomial) — how the player's
       relative positioning within the league changes.

    Additionally models the **opposition quality** effect that the paper's
    trained neural network learns from separate ``league_ability_current``
    and ``league_ability_target`` features: moving to a weaker league means
    facing weaker opposition, which boosts per-90 offensive output even if
    the team is weaker (paper Section 4.3.1 — Doku's xG increases at both
    Liverpool *and* Gwangju).

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
    source_league_mean : float, optional
        Normalized (0-100) mean Elo score of the **source** league.
    target_league_mean : float, optional
        Normalized (0-100) mean Elo score of the **target** league.

    Returns
    -------
    dict[str, float] — predicted per-90 values at the target club.
    """
    # Scale to roughly [-1, 1] for typical transfers (change_ra ranges
    # from roughly -50 to +50 for same-league moves, per ARCHITECTURE.md polynomial
    # normalization which maps -50..+50 to -1..+1 using / 50.0, consistent
    # with PlayerAdjustmentModel.predict()).  Extreme cross-league jumps can
    # exceed this range (e.g. ±1.5).
    ra = change_relative_ability / 50.0

    # Detect whether we have real team-position data or just fallback
    # (both source and target are the same → no style data available).
    _has_style_data = _check_has_style_data(player_per90, source_pos_avg, target_pos_avg)

    # ── Decompose the transfer into two orthogonal forces ────────────────
    # The paper's TF model receives team_ability and league_ability as
    # separate features, learning two distinct effects:
    #
    #   1. TEAM QUALITY: moving to a better/worse team changes how the
    #      player's position is used (better teams create more chances).
    #      Recovered as: team_gap = ra - league_gap
    #
    #   2. OPPOSITION QUALITY: moving to a weaker/stronger league means
    #      facing weaker/stronger opponents, boosting/reducing per-90.
    #      league_gap = (source_league_mean - target_league_mean) / 100
    #
    # Paper Section 4.3.1 evidence:
    #   Liverpool:  team_gap=+0.30, league_gap=-0.20 → xG INCREASES
    #               (big team quality boost outweighs harder league)
    #   Gwangju:    team_gap=-0.40, league_gap=+0.30 → xG INCREASES
    #               (much weaker opposition outweighs worse team)
    #   Barcelona:  team_gap=+0.27, league_gap=-0.15 → xG roughly same
    #               (team boost and harder league roughly cancel)
    league_gap = 0.0
    if source_league_mean is not None and target_league_mean is not None:
        league_gap = (source_league_mean - target_league_mean) / 100.0
    team_gap = ra - league_gap  # absolute team quality change (positive = better team)

    # Player quality modifier: elite players retain more individual output
    # when moving UP (to stronger teams) but are not fully protected when
    # moving DOWN (to much weaker teams where the system won't support them).
    # Centered at 6.5 (average Sofascore rating), scaled multiplicatively.
    # Asymmetric: upgrades reduce team influence (retain individual quality),
    # downgrades only partially reduce it (even elites suffer in bad systems).
    quality_scale = 1.0
    if player_rating is not None and isinstance(player_rating, (int, float)):
        raw_mod = (player_rating - 6.5) * 0.15
        if ra < -0.1:
            # Moving to a weaker team: elite protection is halved
            # (Mbappe at Ipswich WOULD suffer — weaker system, fewer chances)
            raw_mod *= 0.5
        # Clamp so team influence is scaled between 0.7x and 1.3x
        quality_scale = max(0.7, min(1.3, 1.0 - raw_mod))

    predicted: Dict[str, float] = {}
    for m in CORE_METRICS:
        player_val = player_per90.get(m, 0.0)
        src_avg = source_pos_avg.get(m, player_val)
        tgt_avg = target_pos_avg.get(m, player_val)

        team_inf = _TEAM_INFLUENCE.get(m, 0.3)

        # Modulate team influence by player quality: elite players are
        # less team-dependent (lower effective team_inf), multiplicatively.
        effective_team_inf = team_inf * quality_scale

        # ── Paper A.3: y = α + β1·x1 + β2·x2 + β3·x3 + β4·x4 + β5·x4² + β6·x4³
        #
        # x1 = player's previous per-90 (dominant — player retains ~70-90%)
        # x2 = avg feature for position at NEW team (small pull toward team)
        # x3 = diff in position avg old→new (style adaptation)
        # x4 = change in relative ability (polynomial)
        #
        # The paper's trained β1 >> β2, β3 — the player's own level is
        # the primary predictor.  β2 provides a small conformity pull
        # toward the new team.  β3 captures style differences.
        #
        # Key insight from Section 4.3.1: Doku's xG INCREASES at Gwangju
        # (much weaker league) despite Gwangju being a relegation candidate.
        # This means the opposition quality effect (from league_ability
        # features in the trained model) outweighs any pull toward the
        # weaker team's position average.

        # x3: position-average difference old→new (style shift)
        style_diff = tgt_avg - src_avg

        # When no real team-position data is available, estimate style
        # differences from the relative ability change (paper Section 4.2).
        # For extreme transfers (large |ra|), quality differences dominate
        # over style differences — attenuate the estimate proportionally.
        # This prevents double-counting: team_effect + opp_effect already
        # handle quality; the style estimate should capture only the
        # residual tactical differences between teams.
        if not _has_style_data:
            league_coeff = _LEAGUE_STYLE_COEFF.get(m, 0.05)
            # Attenuate for extreme moves: |ra|=0.15 → 70% retained,
            # |ra|=0.30 → 40% retained, |ra|≥0.35 → 30% floor.
            style_scale = max(0.3, 1.0 - abs(ra) * 2.0)
            estimated_style_diff = src_avg * league_coeff * ra * style_scale
            style_diff = estimated_style_diff

        # League-quality adjustment: when teams are in different leagues,
        # much of the position-average difference is due to league quality
        # (e.g., Gwangju wingers have lower xG because of weaker league,
        # not because of tactical style).  The paper's trained model handles
        # this via the separate league_ability features.  We approximate by
        # attenuating style_diff proportionally to the league gap.
        # Same-league (league_gap≈0): full style_diff (league_attn≈1.0).
        # Large cross-league gap (|league_gap|≥0.40): mostly league quality,
        # so attn drops to 0.40 or below.
        # Factor of 1.5 chosen so that a 30-point gap (e.g., Ligue 1 → K-League)
        # reduces style influence to ~55%, while a 40-point gap (extreme) drops
        # to ~40%.  This is less aggressive than before — the paper's examples
        # show style effects persist even in cross-league moves (Doku's xA
        # changes differently at Liverpool vs Barcelona, Section 4.3.1).
        # Floor of 0.25 ensures meaningful style adaptation always occurs.
        league_attn = max(_LEAGUE_ATTN_FLOOR, 1.0 - abs(league_gap) * _LEAGUE_ATTN_FACTOR)

        # Paper A.3 β3·x3: style adaptation
        style_shift = effective_team_inf * style_diff * league_attn

        # Paper A.3 β2·x2: conformity to new team's position average.
        # The paper notes players partly conform to how the target team uses
        # their position (Section 2.3, Appendix A.3).  Attenuated for
        # cross-league moves where position averages reflect league quality
        # more than tactical style.
        conformity_pull = _CONFORMITY_COEFF * effective_team_inf * (tgt_avg - player_val) * league_attn

        # Base: player retains their level + small style/conformity adjustments
        base = player_val + style_shift + conformity_pull

        # ── Team quality factor (paper A.3 x4 = change_relative_ability) ──
        # The paper uses a polynomial in change_relative_ability. We decompose
        # into team quality change and opposition quality change, then combine
        # ADDITIVELY to prevent compounding (the paper's linear regression
        # also adds these effects, not multiplies them).
        sensitivity = _ABILITY_SENSITIVITY.get(m, 0.2)
        opp_sens = _OPP_QUALITY_SENS.get(m, 0.0)

        # Team quality: moving to a better/worse team.
        # Uses team_gap (positive = better team → more output for offensive).
        # Asymmetric damping: less damping for downgrades (large drops are
        # realistic — Mbappe at Ipswich, de Jong at Man Utd) vs upgrades
        # (player can't exceed their talent ceiling).
        damp = _DAMPING_FACTOR_UP if team_gap > 0 else _DAMPING_FACTOR_DOWN
        team_effect = sensitivity * team_gap * (1.0 - damp * abs(team_gap))

        # Opposition quality: moving to a weaker/stronger league.
        # Uses league_gap (positive = weaker league → more per-90 output).
        opp_effect = opp_sens * league_gap

        # Combined adjustment: multiplicative scaling (team_effect and
        # opp_effect are fractional multipliers, so the adjustment is
        # proportional to the player's base level — consistent with the
        # paper's β1·x1 term where predicted output scales with the
        # player's previous per-90).
        combined_factor = 1.0 + team_effect + opp_effect

        pred = base * combined_factor
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


# ── Data-driven coefficient calibration ──────────────────────────────────────

_CALIBRATION_DATA_WEIGHT = 0.6   # blend: data-driven portion
_CALIBRATION_PRIOR_WEIGHT = 0.4  # blend: original defaults portion
_OPP_BASE_WEIGHT = 0.7           # opposition sensitivity: base portion
_OPP_SCALE_WEIGHT = 0.3          # opposition sensitivity: coefficient-of-variation scaled portion


def calibrate_style_coefficients(
    profiles: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Calibrate heuristic coefficients from football-data.co.uk league profiles.

    Uses empirical league-wide match statistics (goals/game, shots/game,
    corners/game, fouls/game) to derive relative league style differences.
    These differences are used to refine ``_LEAGUE_STYLE_COEFF`` and
    ``_OPP_QUALITY_SENS`` so they reflect real-world data rather than
    hand-tuned guesses.

    Parameters
    ----------
    profiles : dict, optional
        ``{league_code: {goals, shots, shots_on_target, fouls, corners}}``
        as returned by ``footballdata_client.compute_multi_season_profiles()``.
        If None, attempts to fetch from football-data.co.uk (requires network).

    Returns
    -------
    dict with keys ``"league_style_coeff"`` and ``"opp_quality_sens"``,
    each mapping metric → calibrated float.  Returns current defaults
    if calibration data is unavailable.
    """
    if profiles is None:
        try:
            from backend.data.footballdata_client import compute_multi_season_profiles
            profiles = compute_multi_season_profiles()
        except Exception as exc:
            _log.warning("Cannot fetch calibration data: %s", exc)
            return {
                "league_style_coeff": dict(_LEAGUE_STYLE_COEFF),
                "opp_quality_sens": dict(_OPP_QUALITY_SENS),
            }

    if not profiles or len(profiles) < 2:
        return {
            "league_style_coeff": dict(_LEAGUE_STYLE_COEFF),
            "opp_quality_sens": dict(_OPP_QUALITY_SENS),
        }

    # Compute cross-league variability for each stat.
    # Higher variability = metrics that differ more across leagues =
    # higher league_style influence.
    stats_keys = ["goals", "shots", "shots_on_target", "fouls", "corners"]
    league_values: Dict[str, List[float]] = {k: [] for k in stats_keys}
    for _lc, profile in profiles.items():
        for k in stats_keys:
            if k in profile:
                league_values[k].append(profile[k])

    # Coefficient of variation (CV) per stat
    cvs: Dict[str, float] = {}
    for k, vals in league_values.items():
        if len(vals) >= 2:
            mean_v = sum(vals) / len(vals)
            std_v = (sum((v - mean_v) ** 2 for v in vals) / len(vals)) ** 0.5
            cvs[k] = std_v / mean_v if mean_v > 0 else 0.0
        else:
            cvs[k] = 0.0

    # Map football-data stats to our core metrics.
    # Higher CV → higher league style coefficient.
    _STAT_TO_METRICS = {
        "goals": ["expected_goals"],
        "shots": ["shots", "touches_in_opposition_box"],
        "shots_on_target": ["expected_assists", "chances_created"],
        "fouls": ["possession_won_final_3rd"],
        "corners": ["successful_crosses", "accurate_long_balls"],
    }

    # Scale CVs to coefficient range [0.02, 0.50].
    max_cv = max(cvs.values()) if cvs else 1.0
    if max_cv < 1e-6:
        max_cv = 1.0

    calibrated_style: Dict[str, float] = dict(_LEAGUE_STYLE_COEFF)
    calibrated_opp: Dict[str, float] = dict(_OPP_QUALITY_SENS)

    for stat_key, metrics in _STAT_TO_METRICS.items():
        cv = cvs.get(stat_key, 0.0)
        # Normalized CV → coefficient in [0.02, 0.50]
        scaled = 0.02 + (cv / max_cv) * 0.48
        for m in metrics:
            if m in calibrated_style:
                calibrated_style[m] = round(
                    _CALIBRATION_DATA_WEIGHT * scaled
                    + _CALIBRATION_PRIOR_WEIGHT * calibrated_style[m], 3
                )

    # Opposition quality sensitivity: proportional to goals CV
    # (leagues with more goal variance = more opposition impact)
    goals_cv = cvs.get("goals", 0.0)
    if goals_cv > 1e-6:
        opp_scale = goals_cv / max_cv  # 0..1
        for m in calibrated_opp:
            # Scale opposition sensitivity by relative league variability.
            # Offensive metrics scale up; defensive scale down (more negative).
            sign = 1.0 if calibrated_opp[m] >= 0 else -1.0
            base_mag = abs(calibrated_opp[m])
            calibrated_opp[m] = round(
                sign * (_OPP_BASE_WEIGHT * base_mag + _OPP_SCALE_WEIGHT * base_mag * opp_scale), 3
            )

    return {
        "league_style_coeff": calibrated_style,
        "opp_quality_sens": calibrated_opp,
    }
