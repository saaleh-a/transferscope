#!/usr/bin/env python3
"""Demo: Train the TransferPortalModel neural network and predict transfers.

This script shows the full trained-model pipeline — NO heuristic fallback.
It generates realistic synthetic training data (no API calls), trains all
4 TensorFlow model groups, and predicts two transfer scenarios:

  1. Upgrade: A mid-table PL forward moves to a top-4 PL club
  2. Downgrade: A top-4 PL midfielder moves to a mid-table Bundesliga club

Run:
    python demo_trained_model.py
"""

from __future__ import annotations

import os
import sys

# Ensure repo root is on path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

np.random.seed(42)

from backend.data.sofascore_client import CORE_METRICS
from backend.models.transfer_portal import (
    MODEL_GROUPS,
    TransferPortalModel,
    build_feature_dict,
)

# ── 1. Generate realistic synthetic training data ────────────────────────────
#
# Each sample represents a historical transfer: we know the player's per-90
# stats at their old club, both clubs' Power Rankings, the team-position
# averages at both clubs, and the player's ACTUAL per-90 at the new club
# (the training label).
#
# The synthetic data encodes the key real-world pattern:
#   - Moving to a stronger team → offensive stats tend to improve
#   - Moving to a weaker team → offensive stats tend to decline
#   - Defensive stats move inversely (stronger team = less defending to do)

# Baseline per-90 values for a "typical" player at each position
_METRIC_BASELINES = {
    "expected_goals": 0.35,
    "expected_assists": 0.20,
    "shots": 2.5,
    "successful_dribbles": 1.2,
    "successful_crosses": 0.8,
    "touches_in_opposition_box": 4.0,
    "successful_passes": 38.0,
    "pass_completion_pct": 82.0,
    "accurate_long_balls": 2.5,
    "chances_created": 1.5,
    "clearances": 2.0,
    "interceptions": 1.3,
    "possession_won_final_3rd": 0.9,
}

# How much each metric responds to team quality gap (positive = improves
# when moving to a better team).  Offensive metrics are positive,
# defensive metrics are negative (better team → less defending).
_QUALITY_SENSITIVITY = {
    "expected_goals": 0.008,
    "expected_assists": 0.005,
    "shots": 0.04,
    "successful_dribbles": 0.015,
    "successful_crosses": 0.012,
    "touches_in_opposition_box": 0.06,
    "successful_passes": 0.5,
    "pass_completion_pct": 0.15,
    "accurate_long_balls": 0.03,
    "chances_created": 0.025,
    "clearances": -0.04,
    "interceptions": -0.025,
    "possession_won_final_3rd": -0.01,
}


def generate_training_data(n_samples: int = 500):
    """Generate n synthetic transfer samples with feature dicts and targets."""
    X_features = []
    y_targets = {m: [] for m in CORE_METRICS}

    for _ in range(n_samples):
        # Random team abilities (0-100 Power Ranking scale)
        team_current = np.random.uniform(20, 95)
        team_target = np.random.uniform(20, 95)
        league_current = np.random.uniform(40, 80)
        league_target = np.random.uniform(40, 80)

        quality_gap = team_target - team_current  # e.g., +20 = upgrade

        # Player per-90 at their current club (baseline + noise + team effect)
        player_per90 = {}
        for m in CORE_METRICS:
            base = _METRIC_BASELINES[m]
            # Better players at better teams
            team_boost = _QUALITY_SENSITIVITY[m] * (team_current - 50)
            noise = np.random.normal(0, base * 0.15)
            player_per90[m] = max(0, base + team_boost + noise)

        # Team-position averages (what the team's forwards/midfielders average)
        team_pos_current = {}
        team_pos_target = {}
        for m in CORE_METRICS:
            base = _METRIC_BASELINES[m]
            team_pos_current[m] = max(0, base + _QUALITY_SENSITIVITY[m] * (team_current - 50) + np.random.normal(0, base * 0.08))
            team_pos_target[m] = max(0, base + _QUALITY_SENSITIVITY[m] * (team_target - 50) + np.random.normal(0, base * 0.08))

        # Build the 43-feature dict
        fd = build_feature_dict(
            player_per90=player_per90,
            team_ability_current=team_current,
            team_ability_target=team_target,
            league_ability_current=league_current,
            league_ability_target=league_target,
            team_pos_current=team_pos_current,
            team_pos_target=team_pos_target,
        )
        X_features.append(fd)

        # LABELS: What the player actually achieved at the new club
        # Key pattern: stats shift toward the target team's profile,
        # modulated by quality gap
        for m in CORE_METRICS:
            # Start from player's current per-90
            current_val = player_per90[m]
            # Pull toward target team's positional average (style adaptation)
            style_pull = 0.3 * (team_pos_target[m] - current_val)
            # Quality effect
            quality_effect = _QUALITY_SENSITIVITY[m] * quality_gap
            # Some noise (real transfers are noisy)
            noise = np.random.normal(0, current_val * 0.08)
            actual = max(0, current_val + style_pull + quality_effect + noise)
            y_targets[m].append(actual)

    return X_features, y_targets


# ── 2. Train the model ───────────────────────────────────────────────────────

print("=" * 70)
print("  TransferScope — Neural Network Training Demo")
print("=" * 70)
print()

print("📊 Generating 500 synthetic transfer samples...")
X_train, y_train = generate_training_data(500)
print(f"   ✅ {len(X_train)} samples × 43 features each")
print(f"   ✅ Targets: {len(CORE_METRICS)} metrics per sample")
print()

print("🧠 Training 4 TensorFlow model groups...")
print("   (Shooting: 2 heads | Passing: 7 heads | Dribbling: 1 head | Defending: 3 heads)")
print()

model = TransferPortalModel()
histories = model.fit(X_train, y_train, epochs=80, batch_size=32, validation_split=0.15)

print("   Training results:")
for group_name, hist in histories.items():
    targets = MODEL_GROUPS[group_name]
    final_loss = hist["loss"][-1]
    final_val_loss = hist["val_loss"][-1]
    print(f"   {'⚡' if group_name == 'shooting' else '◈' if group_name == 'passing' else '◎' if group_name == 'dribbling' else '◆'} {group_name.capitalize():12s} — "
          f"loss: {final_loss:.4f}  val_loss: {final_val_loss:.4f}  "
          f"({len(targets)} output{'s' if len(targets) > 1 else ''})")

print()
print(f"   ✅ Model trained — model.fitted = {model.fitted}")
print(f"   ✅ 4 Keras models built, no heuristic fallback involved")
print()

# ── 3. Predict: Scenario A — Upgrade transfer ────────────────────────────────
#
# A mid-table PL forward (team rating 55) moves to a top-4 club (rating 85).
# We expect: offensive stats UP, defensive stats DOWN.

print("─" * 70)
print("  SCENARIO A: Upgrade Transfer")
print("  Mid-table PL forward → Top-4 PL club")
print("  Team Power Ranking: 55 → 85 (+30 jump)")
print("─" * 70)
print()

player_A = {
    "expected_goals": 0.32,
    "expected_assists": 0.15,
    "shots": 2.8,
    "successful_dribbles": 1.5,
    "successful_crosses": 0.6,
    "touches_in_opposition_box": 3.8,
    "successful_passes": 35.0,
    "pass_completion_pct": 80.0,
    "accurate_long_balls": 2.0,
    "chances_created": 1.2,
    "clearances": 2.5,
    "interceptions": 1.5,
    "possession_won_final_3rd": 1.0,
}

# Mid-table team's forward averages
pos_avg_A_current = {m: v * 0.95 for m, v in player_A.items()}
# Top-4 team's forward averages (higher offensive, lower defensive)
pos_avg_A_target = {
    "expected_goals": 0.45, "expected_assists": 0.25, "shots": 3.5,
    "successful_dribbles": 1.8, "successful_crosses": 1.0,
    "touches_in_opposition_box": 5.5, "successful_passes": 45.0,
    "pass_completion_pct": 86.0, "accurate_long_balls": 3.0,
    "chances_created": 2.0, "clearances": 1.5, "interceptions": 0.9,
    "possession_won_final_3rd": 0.7,
}

fd_A = build_feature_dict(
    player_per90=player_A,
    team_ability_current=55.0,
    team_ability_target=85.0,
    league_ability_current=65.0,
    league_ability_target=65.0,  # same league
    team_pos_current=pos_avg_A_current,
    team_pos_target=pos_avg_A_target,
)

pred_A = model.predict(fd_A)

print(f"  {'Metric':<30s} {'Current':>10s} {'Predicted':>10s} {'Change':>10s}")
print(f"  {'─' * 30} {'─' * 10} {'─' * 10} {'─' * 10}")
for m in CORE_METRICS:
    curr = player_A[m]
    pred = pred_A.get(m, 0.0)
    pct = ((pred - curr) / curr * 100) if curr > 0 else 0
    arrow = "📈" if pct > 2 else "📉" if pct < -2 else "➡️"
    print(f"  {arrow} {m:<27s} {curr:>10.2f} {pred:>10.2f} {pct:>+9.1f}%")

print()

# ── 4. Predict: Scenario B — Downgrade transfer ──────────────────────────────
#
# A top-4 PL midfielder (team rating 82) moves to a mid-table Bundesliga
# club (rating 52). Cross-league, significant step down.

print("─" * 70)
print("  SCENARIO B: Downgrade Transfer")
print("  Top-4 PL midfielder → Mid-table Bundesliga club")
print("  Team Power Ranking: 82 → 52 (-30 drop)")
print("─" * 70)
print()

player_B = {
    "expected_goals": 0.12,
    "expected_assists": 0.22,
    "shots": 1.5,
    "successful_dribbles": 1.0,
    "successful_crosses": 0.9,
    "touches_in_opposition_box": 2.5,
    "successful_passes": 52.0,
    "pass_completion_pct": 89.0,
    "accurate_long_balls": 3.5,
    "chances_created": 2.0,
    "clearances": 1.8,
    "interceptions": 1.6,
    "possession_won_final_3rd": 1.2,
}

pos_avg_B_current = {m: v * 1.05 for m, v in player_B.items()}
pos_avg_B_target = {
    "expected_goals": 0.08, "expected_assists": 0.14, "shots": 1.2,
    "successful_dribbles": 0.8, "successful_crosses": 0.7,
    "touches_in_opposition_box": 1.8, "successful_passes": 38.0,
    "pass_completion_pct": 81.0, "accurate_long_balls": 2.8,
    "chances_created": 1.3, "clearances": 2.8, "interceptions": 2.0,
    "possession_won_final_3rd": 1.4,
}

fd_B = build_feature_dict(
    player_per90=player_B,
    team_ability_current=82.0,
    team_ability_target=52.0,
    league_ability_current=65.0,
    league_ability_target=58.0,  # Bundesliga slightly lower avg
    team_pos_current=pos_avg_B_current,
    team_pos_target=pos_avg_B_target,
)

pred_B = model.predict(fd_B)

print(f"  {'Metric':<30s} {'Current':>10s} {'Predicted':>10s} {'Change':>10s}")
print(f"  {'─' * 30} {'─' * 10} {'─' * 10} {'─' * 10}")
for m in CORE_METRICS:
    curr = player_B[m]
    pred = pred_B.get(m, 0.0)
    pct = ((pred - curr) / curr * 100) if curr > 0 else 0
    arrow = "📈" if pct > 2 else "📉" if pct < -2 else "➡️"
    print(f"  {arrow} {m:<27s} {curr:>10.2f} {pred:>10.2f} {pct:>+9.1f}%")

print()
print("=" * 70)
print("  ✅ Both predictions used the TRAINED neural network.")
print("  ✅ No heuristic fallback (paper_heuristic_predict) was involved.")
print(f"  ✅ Model groups: {list(model.models.keys())}")
print("=" * 70)
