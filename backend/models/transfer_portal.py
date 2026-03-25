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

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from backend.data.fotmob_client import CORE_METRICS

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

    def build(self, input_dim: int) -> None:
        """Build all 4 group models."""
        self.input_dim = input_dim
        for group_name, targets in MODEL_GROUPS.items():
            self.models[group_name] = _build_group_model(
                input_dim, len(targets), group_name
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
        X_arr = np.array([self._prepare_features(fd) for fd in X], dtype=np.float32)

        if self.input_dim is None:
            self.build(X_arr.shape[1])
        elif not self.models:
            self.build(self.input_dim)

        histories = {}
        for group_name, targets in MODEL_GROUPS.items():
            y_group = np.column_stack([
                np.array(y.get(t, [0.0] * len(X)), dtype=np.float32)
                for t in targets
            ])
            hist = self.models[group_name].fit(
                X_arr, y_group,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=0,
            )
            histories[group_name] = hist.history

        self.fitted = True
        return histories

    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, float]:
        """Predict per-90 metrics for a single transfer scenario.

        Returns dict mapping metric name -> predicted per-90 value.
        """
        if not self.models:
            raise RuntimeError("Model not built. Call build() or fit() first.")

        X = self._prepare_features(feature_dict).reshape(1, -1)
        result = {}

        for group_name, targets in MODEL_GROUPS.items():
            if group_name not in self.models:
                continue
            preds = self.models[group_name].predict(X, verbose=0)
            preds = preds.flatten()
            for i, target in enumerate(targets):
                if i < len(preds):
                    result[target] = float(preds[i])

        return result

    def predict_batch(self, feature_dicts: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Predict for multiple transfer scenarios at once."""
        if not self.models:
            raise RuntimeError("Model not built. Call build() or fit() first.")

        X = np.array(
            [self._prepare_features(fd) for fd in feature_dicts],
            dtype=np.float32,
        )
        n = len(feature_dicts)
        results = [{} for _ in range(n)]

        for group_name, targets in MODEL_GROUPS.items():
            if group_name not in self.models:
                continue
            preds = self.models[group_name].predict(X, verbose=0)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            for i in range(n):
                for j, target in enumerate(targets):
                    if j < preds.shape[1]:
                        results[i][target] = float(preds[i, j])

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
        """Load all group models from directory."""
        if directory is None:
            directory = os.path.join(_MODELS_DIR, "transfer_portal")

        self.models = {}
        for group_name in MODEL_GROUPS:
            path = os.path.join(directory, f"{group_name}.keras")
            if os.path.exists(path):
                self.models[group_name] = tf.keras.models.load_model(path)

        self.fitted = bool(self.models)


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
        fd[f"player_{m}"] = player_per90.get(m, 0.0) or 0.0

    fd["team_ability_current"] = team_ability_current
    fd["team_ability_target"] = team_ability_target
    fd["league_ability_current"] = league_ability_current
    fd["league_ability_target"] = league_ability_target

    for m in CORE_METRICS:
        fd[f"team_pos_current_{m}"] = team_pos_current.get(m, 0.0) or 0.0
    for m in CORE_METRICS:
        fd[f"team_pos_target_{m}"] = team_pos_target.get(m, 0.0) or 0.0

    return fd


FEATURE_DIM = len(_feature_keys())  # 43
