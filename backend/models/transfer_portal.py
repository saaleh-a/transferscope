"""TensorFlow multi-head neural network — 4 target groups from CLAUDE.md.

Group 1 - Shooting: xG, Shots (2 heads)
Group 2 - Passing: xA, Crosses, Total Passes, Short Passes, Long Passes,
                    Passes Att Third, Penalty Area Entries (7 heads)
Group 3 - Dribbling: Take-ons (1 head)
Group 4 - Defending: Def own third, Def mid third, Def att third (3 heads)

Architecture per group:
  Input -> Dense(128, relu) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.3)
  -> [Linear output head per target]
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from backend.data.sofascore_client import CORE_METRICS

_log = logging.getLogger(__name__)

# ── Target group definitions ─────────────────────────────────────────────────

MODEL_GROUPS: Dict[str, List[str]] = {
    "shooting": ["expected_goals", "shots"],
    "passing": [
        "expected_assists",
        "successful_crosses",
        "successful_passes",
        "pass_completion_pct",
        "accurate_long_balls",
        "chances_created",
        "touches_in_opposition_box",
    ],
    "dribbling": ["successful_dribbles"],
    "defending": ["clearances", "interceptions", "possession_won_final_3rd"],
}

# ── Per-group feature subsets (Improvement 6) ────────────────────────────────
# Each group uses only the features most relevant to its target metrics.
# The external interface does NOT change — slicing is internal.

GROUP_FEATURE_SUBSETS: Dict[str, List[str]] = {
    "shooting": [
        "player_expected_goals",
        "player_shots",
        "player_touches_in_opposition_box",
        "player_chances_created",
        "team_ability_current",
        "team_ability_target",
        "league_ability_current",
        "league_ability_target",
        "team_pos_current_expected_goals",
        "team_pos_current_shots",
        "team_pos_current_touches_in_opposition_box",
        "team_pos_current_chances_created",
        "team_pos_target_expected_goals",
        "team_pos_target_shots",
        "team_pos_target_touches_in_opposition_box",
        "team_pos_target_chances_created",
    ],
    "passing": [
        "player_expected_assists",
        "player_successful_crosses",
        "player_successful_passes",
        "player_pass_completion_pct",
        "player_accurate_long_balls",
        "player_chances_created",
        "player_touches_in_opposition_box",
        "team_ability_current",
        "team_ability_target",
        "league_ability_current",
        "league_ability_target",
        "team_pos_current_expected_assists",
        "team_pos_current_successful_crosses",
        "team_pos_current_successful_passes",
        "team_pos_current_pass_completion_pct",
        "team_pos_current_accurate_long_balls",
        "team_pos_current_chances_created",
        "team_pos_current_touches_in_opposition_box",
        "team_pos_target_expected_assists",
        "team_pos_target_successful_crosses",
        "team_pos_target_successful_passes",
        "team_pos_target_pass_completion_pct",
        "team_pos_target_accurate_long_balls",
        "team_pos_target_chances_created",
        "team_pos_target_touches_in_opposition_box",
    ],
    "dribbling": [
        "player_successful_dribbles",
        "team_ability_current",
        "team_ability_target",
        "league_ability_current",
        "league_ability_target",
        "team_pos_current_successful_dribbles",
        "team_pos_target_successful_dribbles",
    ],
    "defending": [
        "player_clearances",
        "player_interceptions",
        "player_possession_won_final_3rd",
        "team_ability_current",
        "team_ability_target",
        "league_ability_current",
        "league_ability_target",
        "team_pos_current_clearances",
        "team_pos_current_interceptions",
        "team_pos_current_possession_won_final_3rd",
        "team_pos_target_clearances",
        "team_pos_target_interceptions",
        "team_pos_target_possession_won_final_3rd",
    ],
}

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "models",
)


def _build_group_model(input_dim: int, num_targets: int, group_name: str) -> tf.keras.Model:
    """Build a multi-head model for one target group."""
    inp = tf.keras.Input(shape=(input_dim,), name=f"{group_name}_input")
    x = tf.keras.layers.Dense(128, activation="relu", name=f"{group_name}_dense1")(inp)
    x = tf.keras.layers.Dropout(0.3, name=f"{group_name}_drop1")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name=f"{group_name}_dense2")(x)
    x = tf.keras.layers.Dropout(0.3, name=f"{group_name}_drop2")(x)

    outputs = []
    targets = MODEL_GROUPS[group_name]
    for target in targets:
        out = tf.keras.layers.Dense(1, activation="linear", name=f"head_{target}")(x)
        outputs.append(out)

    if len(outputs) > 1:
        combined = tf.keras.layers.Concatenate(name=f"{group_name}_out")(outputs)
    else:
        combined = outputs[0]

    model = tf.keras.Model(inputs=inp, outputs=combined, name=f"{group_name}_model")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


class TransferPortalModel:
    """4-group multi-head neural network for transfer performance prediction."""

    def __init__(self, input_dim: Optional[int] = None):
        self.input_dim = input_dim
        self.models: Dict[str, tf.keras.Model] = {}
        self.fitted = False
        self._scaler: Any = None  # StandardScaler, loaded from data/models/ if trained
        self._target_scalers: Dict[str, Any] = {}  # Per-group target scalers

    def build(self, input_dim: int) -> None:
        """Build all 4 group models with per-group feature dimensions."""
        self.input_dim = input_dim
        for group_name, targets in MODEL_GROUPS.items():
            group_dim = len(GROUP_FEATURE_SUBSETS[group_name])
            self.models[group_name] = _build_group_model(
                group_dim, len(targets), group_name
            )
        self.fitted = False

    def _prepare_features(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """Flatten a feature dict into a consistent numpy vector.

        Input features (from CLAUDE.md):
        - Player per-90 metrics (current club): 13 values
        - Team ability current: 1 value (normalized 0-100)
        - Team ability target: 1 value
        - League ability current: 1 value
        - League ability target: 1 value
        - Team-position per-90 current: 13 values
        - Team-position per-90 target: 13 values
        Total: 43 features
        """
        keys = _feature_keys()
        vec = [feature_dict.get(k, 0.0) for k in keys]
        return np.array(vec, dtype=np.float32)

    def _prepare_group_features(
        self, feature_dict: Dict[str, float], group_name: str,
    ) -> np.ndarray:
        """Extract only the feature subset relevant to a given group."""
        keys = GROUP_FEATURE_SUBSETS[group_name]
        vec = [feature_dict.get(k, 0.0) for k in keys]
        return np.array(vec, dtype=np.float32)

    def fit(
        self,
        X: List[Dict[str, float]],
        y: Dict[str, List[float]],
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.15,
    ) -> Dict[str, Any]:
        """Train all 4 group models.

        Parameters
        ----------
        X : list[dict]
            Feature dicts, one per training sample.
        y : dict[str, list[float]]
            Targets keyed by metric name. Each list has length = len(X).
        epochs, batch_size, validation_split : training params.

        Returns
        -------
        dict of training history per group.
        """
        from sklearn.preprocessing import StandardScaler

        # Store full feature array for compatibility
        X_arr_full = np.array([self._prepare_features(fd) for fd in X], dtype=np.float32)

        if self.input_dim is None:
            self.build(X_arr_full.shape[1])
        elif not self.models:
            self.build(self.input_dim)

        histories = {}
        for group_name, targets in MODEL_GROUPS.items():
            X_group = np.array(
                [self._prepare_group_features(fd, group_name) for fd in X],
                dtype=np.float32,
            )
            y_group_raw = np.column_stack([
                np.array(y.get(t, [0.0] * len(X)), dtype=np.float32)
                for t in targets
            ])

            # Scale targets per-group to equalise loss across groups
            y_scaler = StandardScaler()
            y_group = y_scaler.fit_transform(y_group_raw)
            self._target_scalers[group_name] = y_scaler

            hist = self.models[group_name].fit(
                X_group, y_group,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=0,
            )
            histories[group_name] = hist.history

        self.fitted = True
        return histories

    def is_trained(self) -> bool:
        """Return True if saved TF weights and feature scaler both exist."""
        model_dir = os.path.join(_MODELS_DIR, "transfer_portal")
        scaler_path = os.path.join(_MODELS_DIR, "feature_scaler.pkl")
        if not os.path.exists(scaler_path):
            return False
        if not os.path.isdir(model_dir):
            return False
        # Check at least one .keras file exists
        for group_name in MODEL_GROUPS:
            if os.path.exists(os.path.join(model_dir, f"{group_name}.keras")):
                return True
        return False

    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, float]:
        """Predict per-90 metrics for a single transfer scenario.

        Returns dict mapping metric name -> predicted per-90 value.

        If trained weights and a feature scaler exist in data/models/,
        loads them and runs the neural network.  Otherwise falls back
        to paper_heuristic_predict() with a logged warning.
        """
        # Try to use trained weights if available and model not already loaded
        if not self.models:
            if self.is_trained():
                try:
                    model_dir = os.path.join(_MODELS_DIR, "transfer_portal")
                    self.load(model_dir)
                except Exception as exc:
                    _log.warning("Failed to load trained model: %s", exc)

            if not self.models:
                _log.warning("No trained model found, using heuristic fallback")
                return self._heuristic_fallback(feature_dict)

        # Build full feature vector and optionally scale it
        full_X = self._prepare_features(feature_dict).reshape(1, -1)
        if self._scaler is not None:
            full_X = self._scaler.transform(full_X)

        # Pre-compute column indices for each group's feature subset
        all_keys = _feature_keys()
        key_to_idx = {k: i for i, k in enumerate(all_keys)}

        # Build safe pre_val lookup.  The model predicts a delta (post − pre),
        # so the anchor matters: if the API returned 0 for a metric (missing
        # advanced stats, partial season, etc.) anchoring to 0 gives a
        # garbage prediction.  Fall back to the training-distribution mean
        # for that metric so predictions remain plausible even when the live
        # API omits certain stats (xG, shots, dribbles are often missing).
        safe_pre: Dict[str, float] = {}
        for idx, key in enumerate(all_keys):
            if not key.startswith("player_"):
                continue
            metric = key[len("player_"):]
            raw = feature_dict.get(key, 0.0)
            if raw <= 0.0 and self._scaler is not None:
                # Stat missing or zero — use training mean as anchor
                safe_pre[metric] = float(self._scaler.mean_[idx])
            else:
                safe_pre[metric] = raw

        result = {}

        for group_name, targets in MODEL_GROUPS.items():
            if group_name not in self.models:
                continue
            group_indices = [key_to_idx[k] for k in GROUP_FEATURE_SUBSETS[group_name]]
            X_group = full_X[:, group_indices]
            preds = self.models[group_name].predict(X_group, verbose=0)

            # Inverse-transform predictions if target scalers are available
            target_scaler = self._target_scalers.get(group_name)
            if target_scaler is not None:
                preds = target_scaler.inverse_transform(preds)

            preds = preds.flatten()
            for i, target in enumerate(targets):
                if i < len(preds):
                    # Model predicts delta (post − pre); add pre-transfer value
                    # back to recover the absolute post-transfer per-90 stat.
                    # Uses safe_pre which falls back to training mean if the
                    # API returned an implausibly low value (> 3σ below mean).
                    pre_val = safe_pre.get(target, 0.0)
                    result[target] = max(0.0, pre_val + float(preds[i]))

        return result

    @staticmethod
    def _heuristic_fallback(feature_dict: Dict[str, float]) -> Dict[str, float]:
        """Fall back to paper_heuristic_predict() when no trained model exists."""
        from backend.features.adjustment_models import paper_heuristic_predict

        # Extract components from feature dict
        player_per90 = {}
        src_pos_avg = {}
        tgt_pos_avg = {}
        for m in CORE_METRICS:
            player_per90[m] = feature_dict.get(f"player_{m}", 0.0)
            src_pos_avg[m] = feature_dict.get(f"team_pos_current_{m}", 0.0)
            tgt_pos_avg[m] = feature_dict.get(f"team_pos_target_{m}", 0.0)

        team_current = feature_dict.get("team_ability_current", 50.0)
        team_target = feature_dict.get("team_ability_target", 50.0)
        league_current = feature_dict.get("league_ability_current", 50.0)
        league_target = feature_dict.get("league_ability_target", 50.0)

        ra_current = team_current - league_current
        ra_target = team_target - league_target
        change_ra = ra_target - ra_current

        return paper_heuristic_predict(
            player_per90=player_per90,
            source_pos_avg=src_pos_avg,
            target_pos_avg=tgt_pos_avg,
            change_relative_ability=change_ra,
        )

    def predict_batch(self, feature_dicts: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Predict for multiple transfer scenarios at once."""
        if not self.models:
            raise RuntimeError("Model not built. Call build() or fit() first.")

        n = len(feature_dicts)
        results = [{} for _ in range(n)]

        for group_name, targets in MODEL_GROUPS.items():
            if group_name not in self.models:
                continue
            X_group = np.array(
                [self._prepare_group_features(fd, group_name) for fd in feature_dicts],
                dtype=np.float32,
            )
            preds = self.models[group_name].predict(X_group, verbose=0)

            # Inverse-transform predictions if target scalers are available
            target_scaler = self._target_scalers.get(group_name)
            if target_scaler is not None:
                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)
                preds = target_scaler.inverse_transform(preds)
            elif preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            for i in range(n):
                for j, target in enumerate(targets):
                    if j < preds.shape[1]:
                        # Model predicts delta (post − pre); add pre-transfer
                        # value back to recover absolute post-transfer per-90.
                        pre_val = feature_dicts[i].get(f"player_{target}", 0.0)
                        results[i][target] = max(0.0, pre_val + float(preds[i, j]))

        return results

    def save(self, directory: Optional[str] = None) -> str:
        """Save all group models to directory."""
        if directory is None:
            directory = os.path.join(_MODELS_DIR, "transfer_portal")
        os.makedirs(directory, exist_ok=True)

        for group_name, model in self.models.items():
            model.save(os.path.join(directory, f"{group_name}.keras"))

        return directory

    def load(self, directory: Optional[str] = None) -> None:
        """Load all group models and associated scalers from directory.

        Scalers (``feature_scaler.pkl`` and ``target_scalers.pkl``) live in
        the parent of *directory* (i.e. ``data/models/``).  They are loaded
        automatically so that ``predict()`` produces correctly scaled output.
        """
        import joblib

        if directory is None:
            directory = os.path.join(_MODELS_DIR, "transfer_portal")

        self.models = {}
        for group_name in MODEL_GROUPS:
            path = os.path.join(directory, f"{group_name}.keras")
            if os.path.exists(path):
                self.models[group_name] = tf.keras.models.load_model(path)

        self.fitted = bool(self.models)

        # Load feature scaler and per-group target scalers that live alongside
        # the model directory.  Without these, predict() would feed unscaled
        # features and return z-scores instead of real per-90 values.
        parent_dir = os.path.dirname(directory)
        scaler_path = os.path.join(parent_dir, "feature_scaler.pkl")
        if not self._scaler and os.path.exists(scaler_path):
            self._scaler = joblib.load(scaler_path)

        target_scaler_path = os.path.join(parent_dir, "target_scalers.pkl")
        if not self._target_scalers and os.path.exists(target_scaler_path):
            self._target_scalers = joblib.load(target_scaler_path)


def _feature_keys() -> List[str]:
    """Return ordered list of feature keys for the input vector."""
    keys = []
    # Player per-90 (current club)
    for m in CORE_METRICS:
        keys.append(f"player_{m}")
    # Team abilities
    keys.append("team_ability_current")
    keys.append("team_ability_target")
    keys.append("league_ability_current")
    keys.append("league_ability_target")
    # Team-position per-90 (current)
    for m in CORE_METRICS:
        keys.append(f"team_pos_current_{m}")
    # Team-position per-90 (target)
    for m in CORE_METRICS:
        keys.append(f"team_pos_target_{m}")
    return keys


def build_feature_dict(
    player_per90: Dict[str, float],
    team_ability_current: float,
    team_ability_target: float,
    league_ability_current: float,
    league_ability_target: float,
    team_pos_current: Dict[str, float],
    team_pos_target: Dict[str, float],
) -> Dict[str, float]:
    """Assemble a feature dict from components, ready for predict().

    This is the convenience function that maps the conceptual inputs
    to the flat feature vector the model expects.
    """
    fd: Dict[str, float] = {}

    for m in CORE_METRICS:
        v = player_per90.get(m)
        fd[f"player_{m}"] = float(v) if v is not None else 0.0

    fd["team_ability_current"] = team_ability_current
    fd["team_ability_target"] = team_ability_target
    fd["league_ability_current"] = league_ability_current
    fd["league_ability_target"] = league_ability_target

    for m in CORE_METRICS:
        v = team_pos_current.get(m)
        fd[f"team_pos_current_{m}"] = float(v) if v is not None else 0.0
    for m in CORE_METRICS:
        v = team_pos_target.get(m)
        fd[f"team_pos_target_{m}"] = float(v) if v is not None else 0.0

    return fd


FEATURE_DIM = len(_feature_keys())  # 43


# ── Improvement 9: Inference-time feature builder ────────────────────────────


def build_feature_dict_from_player(
    player_id: int,
    tournament_id: int,
    season_id: int,
    target_club_id: int,
    target_league_id: int,
    position: str,
    target_team_name: str = "",
    query_date: Optional[date] = None,
) -> Dict[str, float]:
    """Build a full feature dict for inference by fetching live data.

    Steps:
    1. Get match logs → rolling window (last 1000 min) for player per-90.
       Falls back to season aggregate if logs unavailable.
    2. Get power rankings for source (player's current club) and target.
    3. Get team-position averages for both clubs.
    4. Call build_feature_dict() with assembled components.

    Parameters
    ----------
    target_team_name : str
        Display name of the target club (e.g. "Arsenal") used for power
        ranking lookup.  Preferred over resolving from *target_club_id*
        via search.
    """
    from backend.data import sofascore_client
    from backend.data.sofascore_client import normalize_position
    from backend.features import power_rankings
    from backend.features.rolling_windows import player_rolling_average

    norm_pos = normalize_position(position) or "Forward"

    # Step 1: Player per-90 (match logs → rolling window, or season aggregate)
    match_logs = sofascore_client.get_player_match_logs(
        player_id, tournament_id, season_id,
    )

    player_per90: Dict[str, float] = {}
    source_club_id: Optional[int] = None

    if match_logs:
        rolling = player_rolling_average(match_logs)
        player_per90 = {m: (rolling.get(m) or 0.0) for m in CORE_METRICS}
        # Infer source club from season stats
        stats = sofascore_client.get_player_stats_for_season(
            player_id, tournament_id, season_id,
        )
        if stats:
            source_club_id = stats.get("team_id")
    else:
        # Fallback: season aggregate
        stats = sofascore_client.get_player_stats_for_season(
            player_id, tournament_id, season_id,
        )
        if stats:
            per90 = stats.get("per90") or {}
            player_per90 = {m: float(per90.get(m, 0.0) or 0.0) for m in CORE_METRICS}
            source_club_id = stats.get("team_id")

    if not player_per90:
        player_per90 = {m: 0.0 for m in CORE_METRICS}

    # Step 2: Power rankings
    try:
        team_rankings, league_snapshots = power_rankings.compute_daily_rankings(
            query_date
        )
    except Exception:
        team_rankings, league_snapshots = {}, {}

    # Find source team name for ranking lookup
    source_team_name = ""
    if source_club_id:
        stats_for_name = sofascore_client.get_player_stats_for_season(
            player_id, tournament_id, season_id,
        )
        if stats_for_name:
            source_team_name = stats_for_name.get("team", "")

    src_ranking = power_rankings.get_team_ranking(source_team_name) if source_team_name else None
    team_ability_current = src_ranking.normalized_score if src_ranking else 50.0
    league_ability_current = src_ranking.league_mean_normalized if src_ranking else 50.0

    # Target rankings
    tgt_ranking = power_rankings.get_team_ranking(target_team_name) if target_team_name else None
    team_ability_target = tgt_ranking.normalized_score if tgt_ranking else 50.0
    league_ability_target = tgt_ranking.league_mean_normalized if tgt_ranking else 50.0

    # Step 3: Team-position averages
    try:
        src_pos_avg, _ = sofascore_client.get_team_position_averages(
            source_club_id, norm_pos
        ) if source_club_id else ({m: 0.0 for m in CORE_METRICS}, [])
    except Exception:
        src_pos_avg = {m: 0.0 for m in CORE_METRICS}

    try:
        tgt_pos_avg, _ = sofascore_client.get_team_position_averages(
            target_club_id, norm_pos
        )
    except Exception:
        tgt_pos_avg = {m: 0.0 for m in CORE_METRICS}

    # Step 4: Assemble via build_feature_dict
    return build_feature_dict(
        player_per90=player_per90,
        team_ability_current=team_ability_current,
        team_ability_target=team_ability_target,
        league_ability_current=league_ability_current,
        league_ability_target=league_ability_target,
        team_pos_current=src_pos_avg,
        team_pos_target=tgt_pos_avg,
    )
