"""TensorFlow multi-head neural network — 4 target groups from ARCHITECTURE.md.

Group 1 - Shooting: xG, Shots (2 heads)
Group 2 - Passing: xA, Crosses, Total Passes, Short Passes, Long Passes,
                    Passes Att Third, Penalty Area Entries (7 heads)
Group 3 - Dribbling: Take-ons (1 head)
Group 4 - Defending: Def own third, Def mid third, Def att third (3 heads)

Default architecture per group:
  Input -> Dense(128, relu) -> BatchNormalization -> Dropout(0.3)
  -> Dense(64, relu) -> BatchNormalization -> Dropout(0.3)
  -> [Linear output head per target]

Dribbling group override (smaller feature/target ratio):
  Input -> Dense(64, relu) -> BatchNormalization -> Dropout(0.4)
  -> Dense(32, relu) -> BatchNormalization -> Dropout(0.4)
  -> [Linear output head]
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from backend.data.sofascore_client import ADDITIONAL_METRICS, CORE_METRICS

_log = logging.getLogger(__name__)

# ── Delta clipping parameters ────────────────────────────────────────────────
# Cap absolute per-90 delta to ±DELTA_CLIP_MULTIPLIER × pre-transfer value.
# For small metrics (pre_val near zero), use _METRIC_CLIP_FLOORS to set a
# plausible per-metric maximum delta.  Prevents runaway TF model predictions.
DELTA_CLIP_MULTIPLIER = 2.0

# Per-metric clip floors — calibrated to ~40% of each metric's typical per-90
# upper bound.  This represents the maximum plausible single-transfer change
# for a player.  For metrics with small absolute values (xG, xA), tight floors
# prevent the model from making absurdly large predictions when pre_val ≈ 0.
_METRIC_CLIP_FLOORS: Dict[str, float] = {
    "expected_goals": 0.15,              # typical range: 0.05–0.40
    "expected_assists": 0.10,            # typical range: 0.03–0.25
    "shots": 0.80,                       # typical range: 0.5–4.0
    "successful_dribbles": 0.60,         # typical range: 0.2–3.0
    "successful_crosses": 0.30,          # typical range: 0.1–1.5
    "touches_in_opposition_box": 1.00,   # typical range: 0.5–5.0
    "successful_passes": 8.00,           # typical range: 10–80
    "pass_completion_pct": 3.00,         # typical range: 70–92
    "accurate_long_balls": 0.60,         # typical range: 0.5–5.0
    "chances_created": 0.30,             # typical range: 0.1–2.0
    "clearances": 0.80,                  # typical range: 0.5–5.0
    "interceptions": 0.40,               # typical range: 0.2–2.0
    "possession_won_final_3rd": 0.40,    # typical range: 0.2–2.0
}
# Conservative fallback for metrics not in _METRIC_CLIP_FLOORS.
# Set to the median of documented floors (~0.5) to avoid overly tight
# or overly loose clipping on unknown metrics.
DELTA_CLIP_FLOOR = 0.5

# ── Delta shrinkage ──────────────────────────────────────────────────────────
# Multiply predicted deltas by this factor to pull toward naive baseline
# (zero delta).  Prevents systematic overshoot when the model is uncertain.
# Applied *before* clipping so that the shrunken delta is what gets compared
# against the clip threshold — the two guards compose naturally.
DELTA_SHRINKAGE = 0.85

# ── Position labels for one-hot encoding ─────────────────────────────────────
# Order matters — must be stable across training and inference.
POSITION_LABELS: List[str] = ["F", "M", "D", "G"]

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
        # Additional metrics — shooting-relevant enrichment
        "player_xg_on_target",
        "player_non_penalty_xg",
        "player_touches",
        "team_ability_current",
        "team_ability_target",
        "league_ability_current",
        "league_ability_target",
        "raw_elo_current",
        "raw_elo_target",
        "player_height_cm",
        "team_pos_current_expected_goals",
        "team_pos_current_shots",
        "team_pos_current_touches_in_opposition_box",
        "team_pos_current_chances_created",
        "team_pos_target_expected_goals",
        "team_pos_target_shots",
        "team_pos_target_touches_in_opposition_box",
        "team_pos_target_chances_created",
        "interaction_ability_gap",
        "interaction_gap_squared",
        "interaction_league_gap",
        # Relative team dominance within league (Phase 6)
        "relative_ability_current",
        "relative_ability_target",
        "relative_ability_gap",
        # Per-metric league-normalised (Phase 5)
        "league_norm_expected_goals",
        "league_norm_shots",
        "league_norm_touches_in_opposition_box",
        "league_norm_chances_created",
        "league_mean_ratio_expected_goals",
        "league_mean_ratio_shots",
        "league_mean_ratio_touches_in_opposition_box",
        "league_mean_ratio_chances_created",
        # Position one-hot (Phase 8)
        "position_F",
        "position_M",
        "position_D",
        "position_G",
    ],
    "passing": [
        "player_expected_assists",
        "player_successful_crosses",
        "player_successful_passes",
        "player_pass_completion_pct",
        "player_accurate_long_balls",
        "player_chances_created",
        "player_touches_in_opposition_box",
        # Additional metrics — passing-relevant enrichment
        "player_touches",
        "player_fouls_won",
        "team_ability_current",
        "team_ability_target",
        "league_ability_current",
        "league_ability_target",
        "raw_elo_current",
        "raw_elo_target",
        "player_height_cm",
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
        "interaction_ability_gap",
        "interaction_gap_squared",
        "interaction_league_gap",
        # Relative team dominance within league (Phase 6)
        "relative_ability_current",
        "relative_ability_target",
        "relative_ability_gap",
        # Per-metric league-normalised (Phase 5)
        "league_norm_expected_assists",
        "league_norm_successful_crosses",
        "league_norm_successful_passes",
        "league_norm_pass_completion_pct",
        "league_norm_accurate_long_balls",
        "league_norm_chances_created",
        "league_norm_touches_in_opposition_box",
        "league_mean_ratio_expected_assists",
        "league_mean_ratio_successful_crosses",
        "league_mean_ratio_successful_passes",
        "league_mean_ratio_pass_completion_pct",
        "league_mean_ratio_accurate_long_balls",
        "league_mean_ratio_chances_created",
        "league_mean_ratio_touches_in_opposition_box",
        # Position one-hot (Phase 8)
        "position_F",
        "position_M",
        "position_D",
        "position_G",
    ],
    "dribbling": [
        "player_successful_dribbles",
        # Additional metrics — dribbling-relevant enrichment
        "player_dispossessed",
        "player_duels_won_pct",
        "player_fouls_won",
        "player_touches",
        "team_ability_current",
        "team_ability_target",
        "league_ability_current",
        "league_ability_target",
        "raw_elo_current",
        "raw_elo_target",
        "player_age",
        "team_pos_current_successful_dribbles",
        "team_pos_target_successful_dribbles",
        "interaction_ability_gap",
        "interaction_gap_squared",
        "interaction_league_gap",
        # Relative team dominance within league (Phase 6)
        "relative_ability_current",
        "relative_ability_target",
        "relative_ability_gap",
        # Per-metric league-normalised (Phase 5)
        "league_norm_successful_dribbles",
        "league_mean_ratio_successful_dribbles",
        # Position one-hot (Phase 8)
        "position_F",
        "position_M",
        "position_D",
        "position_G",
    ],
    "defending": [
        "player_clearances",
        "player_interceptions",
        "player_possession_won_final_3rd",
        # Additional metrics — defending-relevant enrichment
        "player_recoveries",
        "player_aerial_duels_won_pct",
        "player_duels_won_pct",
        "player_goals_conceded_on_pitch",
        "player_xg_against_on_pitch",
        "team_ability_current",
        "team_ability_target",
        "league_ability_current",
        "league_ability_target",
        "raw_elo_current",
        "raw_elo_target",
        "player_height_cm",
        "team_pos_current_clearances",
        "team_pos_current_interceptions",
        "team_pos_current_possession_won_final_3rd",
        "team_pos_target_clearances",
        "team_pos_target_interceptions",
        "team_pos_target_possession_won_final_3rd",
        "interaction_ability_gap",
        "interaction_gap_squared",
        "interaction_league_gap",
        # Relative team dominance within league (Phase 6)
        "relative_ability_current",
        "relative_ability_target",
        "relative_ability_gap",
        # Per-metric league-normalised (Phase 5)
        "league_norm_clearances",
        "league_norm_interceptions",
        "league_norm_possession_won_final_3rd",
        "league_mean_ratio_clearances",
        "league_mean_ratio_interceptions",
        "league_mean_ratio_possession_won_final_3rd",
        # Position one-hot (Phase 8)
        "position_F",
        "position_M",
        "position_D",
        "position_G",
    ],
}

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "models",
)


# Per-group architecture overrides.  Groups not listed use the defaults
# (hidden_units=[128, 64], dropout=0.3, l2=1e-4).  Smaller networks with
# higher dropout / stronger L2 are used for groups where the signal-to-noise
# ratio is low — the default 128→64 architecture tends to overfit noise in
# these groups.
#
# Backtest SNR analysis (delta_mean / delta_std):
#   shooting: 0.003–0.152 (very low), passing: 0.003–0.063 (very low),
#   dribbling: 0.247 (moderate), defending: 0.040–0.251 (mixed).
_GROUP_ARCH_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "shooting": {"hidden_units": [32, 16], "dropout": 0.45, "l2": 5e-4},
    "dribbling": {"hidden_units": [64, 32], "dropout": 0.4, "l2": 3e-4},
    "defending": {"hidden_units": [96, 48], "dropout": 0.35},
}


def _build_group_model(input_dim: int, num_targets: int, group_name: str) -> tf.keras.Model:
    """Build a multi-head model for one target group.

    Uses BatchNormalization + L2 regularization on Dense layers and Huber
    loss for robustness to outlier deltas in training data.

    ``group_name`` selects per-group architecture overrides from
    ``_GROUP_ARCH_OVERRIDES``.  Groups without an override entry use
    the defaults (128→64 hidden units, 0.3 dropout, 1e-4 L2).
    """
    overrides = _GROUP_ARCH_OVERRIDES.get(group_name, {})
    hidden_units = overrides.get("hidden_units", [128, 64])
    dropout_rate = overrides.get("dropout", 0.3)
    l2_strength = overrides.get("l2", 1e-4)

    l2_reg = tf.keras.regularizers.l2(l2_strength)

    inp = tf.keras.Input(shape=(input_dim,), name=f"{group_name}_input")
    x = tf.keras.layers.Dense(
        hidden_units[0], activation="relu", kernel_regularizer=l2_reg,
        name=f"{group_name}_dense1",
    )(inp)
    x = tf.keras.layers.BatchNormalization(name=f"{group_name}_bn1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name=f"{group_name}_drop1")(x)
    x = tf.keras.layers.Dense(
        hidden_units[1], activation="relu", kernel_regularizer=l2_reg,
        name=f"{group_name}_dense2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{group_name}_bn2")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name=f"{group_name}_drop2")(x)

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
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )
    return model


class TransferPortalModel:
    """4-group multi-head neural network for transfer performance prediction."""

    def __init__(self, input_dim: Optional[int] = None):
        self.input_dim = input_dim
        self.models: Dict[str, tf.keras.Model] = {}
        self.fitted = False
        self._scaler: Any = None  # StandardScaler, loaded from data/models/ if trained
        self._target_scalers: Dict[str, Any] = {}  # Per-group target scalers

    @staticmethod
    def _clip_delta(delta: float, pre_val: float, target: str = "") -> float:
        """Clip an extreme model delta to a plausible range.

        Uses per-metric clip floors from _METRIC_CLIP_FLOORS for tight,
        metric-appropriate bounds.  Falls back to DELTA_CLIP_FLOOR for
        unknown metrics.
        """
        floor = _METRIC_CLIP_FLOORS.get(target, DELTA_CLIP_FLOOR)
        max_delta = max(DELTA_CLIP_MULTIPLIER * abs(pre_val), floor)
        if abs(delta) > max_delta:
            _log.warning(
                "Clipping extreme delta for %s: %.3f -> %.3f (pre_val=%.3f)",
                target, delta,
                max_delta if delta > 0 else -max_delta,
                pre_val,
            )
            return max(-max_delta, min(max_delta, delta))
        return delta

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

        Input features (from ARCHITECTURE.md):
        - Player per-90 metrics (current club): 13 values
        - Team ability current: 1 value (normalized 0-100)
        - Team ability target: 1 value
        - League ability current: 1 value
        - League ability target: 1 value
        - Team-position per-90 current: 13 values
        - Team-position per-90 target: 13 values
        - Interaction features: 3 values (ability_gap, gap², league_gap)
        Total: 46 features
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
            # Use the raw pre-transfer value as the delta anchor.
            # Do NOT substitute the training mean for zero values: defenders and
            # players from leagues without advanced stat tracking legitimately have
            # 0 for metrics like xG and shots.  Substituting the training mean
            # causes systematic overestimation for these players and makes
            # predictions worse than the naive baseline.
            safe_pre[metric] = max(0.0, raw)

        result = {}

        for group_name, targets in MODEL_GROUPS.items():
            if group_name not in self.models:
                continue
            group_indices = [key_to_idx[k] for k in GROUP_FEATURE_SUBSETS[group_name]]
            X_group = full_X[:, group_indices]
            # Use direct model call instead of model.predict() for
            # single-sample inference to avoid tf.function retracing
            # warnings caused by varying input shapes.
            preds = self.models[group_name](X_group, training=False).numpy()

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
                    raw_delta = float(preds[i]) * DELTA_SHRINKAGE
                    delta = self._clip_delta(raw_delta, pre_val, target)
                    result[target] = max(0.0, pre_val + delta)

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
            source_league_mean=league_current,
            target_league_mean=league_target,
        )

    def predict_batch(self, feature_dicts: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Predict for multiple transfer scenarios at once."""
        if not self.models:
            raise RuntimeError("Model not built. Call build() or fit() first.")

        n = len(feature_dicts)
        results = [{} for _ in range(n)]

        # Build and scale the full feature matrix first (mirrors predict()).
        # predict_batch previously built X_group from raw feature dicts without
        # applying self._scaler, feeding unscaled inputs to a network trained on
        # StandardScaler-scaled features and producing garbage predictions.
        all_keys = _feature_keys()
        key_to_idx = {k: i for i, k in enumerate(all_keys)}
        X_full = np.array(
            [self._prepare_features(fd) for fd in feature_dicts], dtype=np.float32
        )
        if self._scaler is not None:
            X_full = self._scaler.transform(X_full)

        for group_name, targets in MODEL_GROUPS.items():
            if group_name not in self.models:
                continue
            group_indices = [key_to_idx[k] for k in GROUP_FEATURE_SUBSETS[group_name]]
            X_group = X_full[:, group_indices]
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
                        raw_delta = float(preds[i, j]) * DELTA_SHRINKAGE
                        delta = self._clip_delta(raw_delta, pre_val, target)
                        results[i][target] = max(0.0, pre_val + delta)

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

    def compute_feature_importance(
        self, feature_dict: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-group feature importance via gradient-based sensitivity.

        For each model group, computes the absolute gradient of each output
        with respect to each input feature. Higher gradient = more important.

        Returns: {group_name: {feature_name: importance_score}}
        Returns empty dict if model is not trained.
        """
        if not self.models:
            return {}

        all_keys = _feature_keys()
        key_to_idx = {k: i for i, k in enumerate(all_keys)}

        full_X = self._prepare_features(feature_dict)
        if self._scaler is not None:
            full_X = self._scaler.transform(full_X.reshape(1, -1)).flatten()

        result: Dict[str, Dict[str, float]] = {}

        for group_name, targets in MODEL_GROUPS.items():
            if group_name not in self.models:
                continue

            group_keys = GROUP_FEATURE_SUBSETS[group_name]
            group_indices = [key_to_idx[k] for k in group_keys]
            x_np = full_X[group_indices].astype(np.float32)
            x_tensor = tf.Variable(x_np.reshape(1, -1))

            with tf.GradientTape() as tape:
                preds = self.models[group_name](x_tensor, training=False)
                # Sum all output heads to get a scalar for gradient computation
                total = tf.reduce_sum(preds)

            grads = tape.gradient(total, x_tensor)
            if grads is None:
                result[group_name] = {k: 0.0 for k in group_keys}
                continue

            abs_grads = np.abs(grads.numpy().flatten())
            grad_sum = abs_grads.sum()
            if grad_sum > 0:
                normalized = abs_grads / grad_sum
            else:
                normalized = np.zeros_like(abs_grads)

            result[group_name] = {
                k: float(normalized[i]) for i, k in enumerate(group_keys)
            }

        return result

    def predict_with_confidence(
        self,
        feature_dict: Dict[str, float],
        pre_per90: Dict[str, float],
        n_samples: int = 20,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Predict with confidence intervals via Monte Carlo dropout.

        Runs the model multiple times with dropout enabled (training=True)
        to get a distribution of predictions. Returns (prediction, std_dev).

        The prediction is the mean across samples.
        The std_dev indicates model uncertainty — higher = less confident.

        Falls back to regular predict() with zero std if model isn't trained.
        """
        if not self.models:
            preds = self.predict(feature_dict)
            std_devs = {k: 0.0 for k in preds}
            return preds, std_devs

        all_keys = _feature_keys()
        key_to_idx = {k: i for i, k in enumerate(all_keys)}

        full_X = self._prepare_features(feature_dict).reshape(1, -1)
        if self._scaler is not None:
            full_X = self._scaler.transform(full_X)

        # Collect per-metric samples across Monte Carlo passes
        samples: Dict[str, List[float]] = {
            t: [] for targets in MODEL_GROUPS.values() for t in targets
        }

        for _ in range(n_samples):
            for group_name, targets in MODEL_GROUPS.items():
                if group_name not in self.models:
                    continue

                group_indices = [
                    key_to_idx[k] for k in GROUP_FEATURE_SUBSETS[group_name]
                ]
                X_group = full_X[:, group_indices]

                # training=True keeps dropout active for MC sampling
                preds = self.models[group_name](X_group, training=True).numpy()

                target_scaler = self._target_scalers.get(group_name)
                if target_scaler is not None:
                    preds = target_scaler.inverse_transform(preds)

                preds = preds.flatten()
                for i, target in enumerate(targets):
                    if i < len(preds):
                        pre_val = pre_per90.get(target, 0.0)
                        raw_delta = float(preds[i]) * DELTA_SHRINKAGE
                        delta = self._clip_delta(raw_delta, pre_val, target)
                        samples[target].append(max(0.0, pre_val + delta))

        mean_preds: Dict[str, float] = {}
        std_preds: Dict[str, float] = {}
        for metric, vals in samples.items():
            if vals:
                mean_preds[metric] = float(np.mean(vals))
                std_preds[metric] = float(np.std(vals))
            else:
                mean_preds[metric] = 0.0
                std_preds[metric] = 0.0

        return mean_preds, std_preds


def _feature_keys() -> List[str]:
    """Return ordered list of feature keys for the input vector."""
    keys = []
    # Player per-90 (current club) — 13 core metrics
    for m in CORE_METRICS:
        keys.append(f"player_{m}")
    # Player per-90 — 10 additional metrics (enrichment features, not targets)
    for m in ADDITIONAL_METRICS:
        keys.append(f"player_{m}")
    # Team abilities (normalized 0-100)
    keys.append("team_ability_current")
    keys.append("team_ability_target")
    keys.append("league_ability_current")
    keys.append("league_ability_target")
    # Raw Elo scores (absolute scale, preserves cross-league strength)
    keys.append("raw_elo_current")
    keys.append("raw_elo_target")
    # REEP player metadata
    keys.append("player_height_cm")
    keys.append("player_age")
    # Team-position per-90 (current)
    for m in CORE_METRICS:
        keys.append(f"team_pos_current_{m}")
    # Team-position per-90 (target)
    for m in CORE_METRICS:
        keys.append(f"team_pos_target_{m}")
    # Interaction features (Phase 4)
    keys.append("interaction_ability_gap")
    keys.append("interaction_gap_squared")
    keys.append("interaction_league_gap")
    # Relative team dominance within league (Phase 6)
    keys.append("relative_ability_current")
    keys.append("relative_ability_target")
    keys.append("relative_ability_gap")
    # Per-metric league-normalised features (Phase 5)
    # How many multiples of league average the player is per metric
    for m in CORE_METRICS:
        keys.append(f"league_norm_{m}")
    # Ratio of source-to-target league means per metric
    for m in CORE_METRICS:
        keys.append(f"league_mean_ratio_{m}")
    # Position one-hot encoding (Phase 8)
    for pos in POSITION_LABELS:
        keys.append(f"position_{pos}")
    return keys


def build_feature_dict(
    player_per90: Dict[str, float],
    team_ability_current: float,
    team_ability_target: float,
    league_ability_current: float,
    league_ability_target: float,
    team_pos_current: Dict[str, float],
    team_pos_target: Dict[str, float],
    raw_elo_current: float = 1500.0,
    raw_elo_target: float = 1500.0,
    player_height_cm: float = 0.0,
    player_age: float = 0.0,
    source_league_means: Optional[Dict[str, float]] = None,
    target_league_means: Optional[Dict[str, float]] = None,
    position: str = "",
) -> Dict[str, float]:
    """Assemble a feature dict from components, ready for predict().

    This is the convenience function that maps the conceptual inputs
    to the flat feature vector the model expects.

    Parameters
    ----------
    raw_elo_current / raw_elo_target : float
        Raw Elo scores (absolute scale).  Defaults to 1500.0 (neutral)
        when unavailable.  These preserve cross-league strength that
        normalized scores lose.
    player_height_cm : float
        Player height in cm from REEP.  0.0 when unavailable.
    player_age : float
        Player age in years from REEP.  0.0 when unavailable.
    source_league_means / target_league_means : dict, optional
        Per-metric league averages for the source/target league.
        When provided, enables per-metric league-normalised features.
    position : str
        Single-letter position code ('F', 'M', 'D', 'G') for one-hot
        encoding.  Empty or unknown positions produce all-zero one-hot.
    """
    fd: Dict[str, float] = {}

    for m in CORE_METRICS:
        v = player_per90.get(m)
        fd[f"player_{m}"] = float(v) if v is not None else 0.0

    # Additional metrics — enrichment inputs (not prediction targets)
    for m in ADDITIONAL_METRICS:
        v = player_per90.get(m)
        fd[f"player_{m}"] = float(v) if v is not None else 0.0

    fd["team_ability_current"] = team_ability_current
    fd["team_ability_target"] = team_ability_target
    fd["league_ability_current"] = league_ability_current
    fd["league_ability_target"] = league_ability_target

    # Raw Elo — preserves absolute league strength across sources
    fd["raw_elo_current"] = raw_elo_current
    fd["raw_elo_target"] = raw_elo_target

    # REEP player metadata
    fd["player_height_cm"] = player_height_cm
    fd["player_age"] = player_age

    for m in CORE_METRICS:
        v = team_pos_current.get(m)
        fd[f"team_pos_current_{m}"] = float(v) if v is not None else 0.0
    for m in CORE_METRICS:
        v = team_pos_target.get(m)
        fd[f"team_pos_target_{m}"] = float(v) if v is not None else 0.0

    # Interaction features (Phase 4)
    ability_gap = team_ability_target - team_ability_current
    fd["interaction_ability_gap"] = ability_gap
    fd["interaction_gap_squared"] = ability_gap ** 2
    fd["interaction_league_gap"] = league_ability_target - league_ability_current

    # Relative team dominance within league (team_ability - league_ability)
    rel_current = team_ability_current - league_ability_current
    rel_target = team_ability_target - league_ability_target
    fd["relative_ability_current"] = rel_current
    fd["relative_ability_target"] = rel_target
    fd["relative_ability_gap"] = rel_target - rel_current

    # Per-metric league-normalised features (Phase 5)
    _src_means = source_league_means or {}
    _tgt_means = target_league_means or {}
    for m in CORE_METRICS:
        player_val = fd.get(f"player_{m}", 0.0)
        src_mean = _src_means.get(m, 0.0)
        tgt_mean = _tgt_means.get(m, 0.0)
        # How many multiples of source league average the player is
        # (capped to avoid extreme values when league mean ≈ 0)
        if src_mean > 1e-6:
            fd[f"league_norm_{m}"] = min(player_val / src_mean, 20.0)
        else:
            fd[f"league_norm_{m}"] = 0.0
        # Ratio of source-to-target league means (how different are the leagues
        # on this specific metric).  1.0 = identical, >1 = source league has
        # higher raw averages.
        if tgt_mean > 1e-6:
            fd[f"league_mean_ratio_{m}"] = min(src_mean / tgt_mean, 5.0) if src_mean > 1e-6 else 1.0
        else:
            fd[f"league_mean_ratio_{m}"] = 1.0

    # Position one-hot encoding (Phase 8)
    pos_upper = position.strip().upper()[:1] if position else ""
    for p in POSITION_LABELS:
        fd[f"position_{p}"] = 1.0 if pos_upper == p else 0.0

    return fd


FEATURE_DIM = len(_feature_keys())  # 93 (13 player_core + 10 player_additional + 4 team/league + 2 raw_elo + 2 reep + 26 team_pos + 3 interaction + 3 relative + 13 league_norm + 13 league_mean_ratio + 4 position_one_hot)
_log.info("Feature vector dimension: %d", FEATURE_DIM)

# Minimum minutes threshold for league mean computation (matches training pipeline)
_MIN_MINUTES_THRESHOLD = 450


def _compute_league_means_from_stats(
    players: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute league mean per-90 from a list of player stat dicts.

    Mirrors ``training_pipeline._compute_league_means`` but operates on
    already-fetched data instead of calling Sofascore.
    """
    metric_sums: Dict[str, float] = {m: 0.0 for m in CORE_METRICS}
    metric_counts: Dict[str, int] = {m: 0 for m in CORE_METRICS}

    for p in players:
        if p.get("minutes_played", 0) < _MIN_MINUTES_THRESHOLD:
            continue
        per90 = p.get("per90") or {}
        for m in CORE_METRICS:
            v = per90.get(m)
            if v is not None:
                try:
                    metric_sums[m] += float(v)
                    metric_counts[m] += 1
                except (ValueError, TypeError):
                    pass

    return {
        m: metric_sums[m] / metric_counts[m] if metric_counts[m] > 0 else 0.0
        for m in CORE_METRICS
    }


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
    player_name: str = "",
) -> Dict[str, float]:
    """Build a full feature dict for inference by fetching live data.

    Steps:
    1. Get match logs → rolling window (last 1000 min) for player per-90.
       Falls back to season aggregate if logs unavailable.
    2. Get power rankings for source (player's current club) and target.
    3. Get team-position averages for both clubs.
    4. Fetch REEP player metadata (height, age).
    5. Fetch StatsBomb spatial features (when available).
    6. Call build_feature_dict() with assembled components.

    Parameters
    ----------
    target_team_name : str
        Display name of the target club (e.g. "Arsenal") used for power
        ranking lookup.  Preferred over resolving from *target_club_id*
        via search.
    player_name : str
        Player display name for StatsBomb spatial feature lookup.
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
        # get_player_match_logs returns ascending (oldest first); reverse so
        # player_rolling_average picks the MOST RECENT 1000 minutes of form.
        rolling = player_rolling_average(list(reversed(match_logs)))
        player_per90 = {m: (rolling.get(m) or 0.0) for m in CORE_METRICS + ADDITIONAL_METRICS}
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
            player_per90 = {m: float(per90.get(m, 0.0) or 0.0) for m in CORE_METRICS + ADDITIONAL_METRICS}
            source_club_id = stats.get("team_id")

    if not player_per90:
        player_per90 = {m: 0.0 for m in CORE_METRICS + ADDITIONAL_METRICS}

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

    src_ranking = power_rankings.get_team_ranking(
        source_team_name, tournament_id=tournament_id,
    ) if source_team_name else None
    team_ability_current = src_ranking.normalized_score if src_ranking else 50.0
    raw_elo_current = src_ranking.raw_elo if src_ranking else 1500.0
    league_ability_current = power_rankings.get_league_opta_rating(
        src_ranking.league_code if src_ranking else None,
        source_team_name or None,
    )

    # Target rankings
    tgt_ranking = power_rankings.get_team_ranking(
        target_team_name, tournament_id=target_league_id,
    ) if target_team_name else None
    team_ability_target = tgt_ranking.normalized_score if tgt_ranking else 50.0
    raw_elo_target = tgt_ranking.raw_elo if tgt_ranking else 1500.0
    league_ability_target = power_rankings.get_league_opta_rating(
        tgt_ranking.league_code if tgt_ranking else None,
        target_team_name or None,
    )

    # REEP team enrichment — resolve Sofascore club IDs to stable reep_ids
    try:
        from backend.data import reep_registry as _rr
        if source_club_id:
            src_team = _rr.enrich_team(source_club_id)
            if src_team.get("reep_id"):
                _log.debug("REEP source team %s → %s", source_club_id, src_team["reep_id"])
        if target_club_id:
            tgt_team = _rr.enrich_team(target_club_id)
            if tgt_team.get("reep_id"):
                _log.debug("REEP target team %s → %s", target_club_id, tgt_team["reep_id"])
    except Exception:
        pass

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

    # Step 4: REEP player metadata
    player_height_cm = 0.0
    player_age = 0.0
    try:
        from backend.data import reep_registry
        reep_data = reep_registry.enrich_player(player_id)
        if reep_data.get("reep_id"):
            _log.debug("REEP player %s → %s", player_id, reep_data["reep_id"])
        if reep_data.get("height_cm"):
            player_height_cm = float(reep_data["height_cm"])
        if reep_data.get("date_of_birth"):
            try:
                from datetime import datetime
                dob = datetime.strptime(str(reep_data["date_of_birth"])[:10], "%Y-%m-%d").date()
                ref = query_date or date.today()
                player_age = (ref - dob).days / 365.25
            except Exception:
                pass
    except Exception:
        pass

    # Step 5: League means for per-metric normalisation
    source_league_means: Optional[Dict[str, float]] = None
    target_league_means: Optional[Dict[str, float]] = None
    try:
        src_league_stats = sofascore_client.get_league_player_stats(
            tournament_id, season_id, limit=300,
        )
        if src_league_stats:
            source_league_means = _compute_league_means_from_stats(src_league_stats)
    except Exception:
        _log.debug("Could not fetch source league means for tournament %s", tournament_id)
    try:
        # For target league, use target_league_id with current season
        # (season_id may differ, but league means are stable across seasons)
        tgt_league_stats = sofascore_client.get_league_player_stats(
            target_league_id, season_id, limit=300,
        )
        if tgt_league_stats:
            target_league_means = _compute_league_means_from_stats(tgt_league_stats)
    except Exception:
        _log.debug("Could not fetch target league means for tournament %s", target_league_id)

    # Step 6: Assemble via build_feature_dict
    return build_feature_dict(
        player_per90=player_per90,
        team_ability_current=team_ability_current,
        team_ability_target=team_ability_target,
        league_ability_current=league_ability_current,
        league_ability_target=league_ability_target,
        team_pos_current=src_pos_avg,
        team_pos_target=tgt_pos_avg,
        raw_elo_current=raw_elo_current,
        raw_elo_target=raw_elo_target,
        player_height_cm=player_height_cm,
        player_age=player_age,
        source_league_means=source_league_means,
        target_league_means=target_league_means,
        position=position,
    )
