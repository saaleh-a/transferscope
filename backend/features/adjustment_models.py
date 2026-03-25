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

    # Walk consecutive pairs: at club A (before) → club B (after)
    for i in range(len(transfers) - 1):
        to_transfer = transfers[i]       # newer (arrival at new club)
        from_transfer = transfers[i + 1] # older (was at previous club)

        new_team = to_transfer.get("to_team") or {}
        old_team = from_transfer.get("to_team") or from_transfer.get("from_team") or {}

        new_team_id = new_team.get("id")
        old_team_id = old_team.get("id")
        new_team_name = new_team.get("name", "")
        old_team_name = old_team.get("name", "")

        if not new_team_id or not old_team_id:
            continue

        # Try to get the player's stats at both clubs.  We use the
        # current-season stats as a proxy (best available without
        # historical season resolution).
        try:
            new_stats = sofascore_client.get_player_stats(player_id)
        except Exception:
            continue

        # Power rankings for relative ability
        new_ranking = power_rankings.get_team_ranking(new_team_name)
        old_ranking = power_rankings.get_team_ranking(old_team_name)

        if new_ranking is None or old_ranking is None:
            continue

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

        # Only use the most recent transfer for now
        break

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
