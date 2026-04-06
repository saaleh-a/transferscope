"""Backtesting framework for TransferScope prediction validation.

Compares trained model predictions against actual post-transfer per-90 stats
for a held-out test set of historical transfers.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from backend.data.sofascore_client import CORE_METRICS
from backend.features import power_rankings
from backend.models.transfer_portal import (
    FEATURE_DIM,
    MODEL_GROUPS,
    TransferPortalModel,
)

_log = logging.getLogger(__name__)

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "models",
)

# Typical per-90 upper bounds for normalising pre-value magnitude.
# Used by _prediction_confidence() to judge whether a player's stats
# are rich enough to support a confident prediction.
_TYPICAL_MAX: Dict[str, float] = {
    "expected_goals": 0.6, "expected_assists": 0.4, "shots": 4.0,
    "successful_dribbles": 3.0, "successful_crosses": 2.0,
    "touches_in_opposition_box": 6.0, "successful_passes": 80.0,
    "pass_completion_pct": 95.0, "accurate_long_balls": 5.0,
    "chances_created": 3.0, "clearances": 6.0,
    "interceptions": 3.0, "possession_won_final_3rd": 2.0,
}


# Confidence score component weights.  Metric coverage is the strongest
# signal (missing stats directly degrade prediction quality); pre-value
# magnitude and context completeness contribute equally to the remainder.
_CONF_W_METRIC_COVERAGE = 0.4
_CONF_W_PRE_MAGNITUDE = 0.3
_CONF_W_CTX_COMPLETENESS = 0.3


def _prediction_confidence(X_row: np.ndarray) -> float:
    """Compute a [0, 1] prediction confidence score from a feature row.

    Parameters
    ----------
    X_row : ndarray, shape (FEATURE_DIM,)
        A single row from the unscaled feature matrix.  The first
        ``len(CORE_METRICS)`` columns are player per-90 stats; the
        remaining columns are contextual features (team ability, Elo,
        REEP metadata, team-position averages, interactions).

    Combines three independent signals so that the score has meaningful
    variance across the test set:

    1. **Player metric coverage** -- fraction of the core per-90 stats
       that are non-zero.  Zeros may indicate missing data rather than
       genuinely zero contribution.
    2. **Pre-value magnitude** -- average of each per-90 stat normalised
       by its typical upper bound.  Players with very low baselines
       (e.g. defenders with near-zero xG) get lower scores because the
       model has less signal to anchor its prediction.
    3. **Context feature completeness** -- fraction of non-player features
       (team ability, Elo, REEP metadata, team-position averages,
       interactions) that are non-zero.  Missing context degrades
       prediction quality.
    """
    n_player = len(CORE_METRICS)

    # Component 1: Player metric coverage
    player_vals = X_row[:n_player]
    n_nonzero = int(np.count_nonzero(player_vals))
    metric_coverage = n_nonzero / n_player

    # Component 2: Pre-value magnitude
    magnitude_scores = []
    for j, m in enumerate(CORE_METRICS):
        v = float(X_row[j])
        max_v = _TYPICAL_MAX.get(m, 1.0)
        magnitude_scores.append(min(v / max_v, 1.0) if max_v > 0 else 0.0)
    pre_magnitude = float(np.mean(magnitude_scores))

    # Component 3: Context feature completeness
    context_features = X_row[n_player:]
    n_ctx = len(context_features)
    ctx_nonzero = int(np.count_nonzero(context_features))
    ctx_completeness = ctx_nonzero / n_ctx if n_ctx > 0 else 0.0

    confidence = (
        _CONF_W_METRIC_COVERAGE * metric_coverage
        + _CONF_W_PRE_MAGNITUDE * pre_magnitude
        + _CONF_W_CTX_COMPLETENESS * ctx_completeness
    )
    return float(np.clip(confidence, 0.0, 1.0))


def _feature_keys_list() -> List[str]:
    """Return the ordered list of feature keys matching the 76-feature vector.

    Must stay in sync with ``_feature_keys()`` in transfer_portal.py —
    includes raw Elo, REEP metadata, interaction features, and
    per-metric league-normalised features.
    """
    keys = []
    for m in CORE_METRICS:
        keys.append(f"player_{m}")
    keys.extend([
        "team_ability_current", "team_ability_target",
        "league_ability_current", "league_ability_target",
    ])
    # Raw Elo scores (absolute scale)
    keys.append("raw_elo_current")
    keys.append("raw_elo_target")
    # REEP player metadata
    keys.append("player_height_cm")
    keys.append("player_age")
    for m in CORE_METRICS:
        keys.append(f"team_pos_current_{m}")
    for m in CORE_METRICS:
        keys.append(f"team_pos_target_{m}")
    # Interaction features (must match transfer_portal._feature_keys())
    keys.append("interaction_ability_gap")
    keys.append("interaction_gap_squared")
    keys.append("interaction_league_gap")
    # Per-metric league-normalised features (Phase 5)
    for m in CORE_METRICS:
        keys.append(f"league_norm_{m}")
    for m in CORE_METRICS:
        keys.append(f"league_mean_ratio_{m}")
    return keys


def run_backtest(
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_test: List[Dict[str, Any]],
    meta_train: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run backtest on held-out test set.

    Compares trained model predictions and naive baseline against actual
    post-transfer per-90 stats.

    Parameters
    ----------
    X_test : ndarray, shape (N, 46)
    y_test : ndarray, shape (N, 13)
    meta_test : list[dict]
    meta_train : list[dict], optional
        If provided, used to check for data leakage (overlapping player IDs).

    Returns
    -------
    dict — backtest report, also saved to data/models/backtest_report.json
    """
    # Data-leakage guard: hard stop if any test players appear in training data.
    if meta_train is not None:
        train_ids = {m.get("player_id") for m in meta_train if m.get("player_id") is not None}
        test_ids = {m.get("player_id") for m in meta_test if m.get("player_id") is not None}
        overlap = train_ids & test_ids
        if overlap:
            raise ValueError(
                f"DATA LEAKAGE: {len(overlap)} player(s) appear in both train and test sets: "
                f"{list(overlap)[:5]}... Backtest aborted. Re-run with a clean temporal split."
            )

    metric_to_idx = {m: i for i, m in enumerate(CORE_METRICS)}
    keys = _feature_keys_list()

    # Load trained model — model.load() handles both model weights and
    # scaler loading (feature_scaler.pkl + target_scalers.pkl) internally.
    model = TransferPortalModel()
    model_dir = os.path.join(_MODELS_DIR, "transfer_portal")

    if os.path.isdir(model_dir):
        model.load(model_dir)

    has_trained = model.fitted and model._scaler is not None

    # Per-metric collectors
    trained_abs_errors: Dict[str, List[float]] = {m: [] for m in CORE_METRICS}
    trained_pct_errors: Dict[str, List[float]] = {m: [] for m in CORE_METRICS}
    naive_abs_errors: Dict[str, List[float]] = {m: [] for m in CORE_METRICS}
    naive_pct_errors: Dict[str, List[float]] = {m: [] for m in CORE_METRICS}
    trained_within_10: Dict[str, int] = {m: 0 for m in CORE_METRICS}
    trained_within_20: Dict[str, int] = {m: 0 for m in CORE_METRICS}
    trained_direction: Dict[str, int] = {m: 0 for m in CORE_METRICS}
    direction_total: Dict[str, int] = {m: 0 for m in CORE_METRICS}

    n = len(X_test)
    skipped_no_coverage = 0

    for i in range(n):
        # Coverage guard — skip samples where neither ClubElo nor Opta covers
        # both clubs.  Samples built after the training_pipeline pre-filter will
        # never hit this, but it protects against independently-assembled test sets.
        meta_i = meta_test[i]
        from_club = meta_i.get("from_club")
        to_club = meta_i.get("to_club")
        if from_club or to_club:
            _bt_date = None
            raw_date = meta_i.get("transfer_date")
            if raw_date:
                try:
                    from datetime import date as _date_cls
                    _bt_date = _date_cls.fromisoformat(str(raw_date)[:10])
                except (ValueError, TypeError):
                    pass
            from_r = power_rankings.get_team_ranking(from_club, _bt_date) if from_club else None
            to_r = power_rankings.get_team_ranking(to_club, _bt_date) if to_club else None
            if from_club and from_r is None or to_club and to_r is None:
                skipped_no_coverage += 1
                continue

        # Naive baseline: player's pre-transfer stats (from X_test, first 13 cols)
        naive_pred = {m: float(X_test[i, j]) for j, m in enumerate(CORE_METRICS)}

        # Trained model prediction — pass UNSCALED features; model.predict()
        # handles scaling internally via model._scaler.
        if has_trained:
            feature_dict = {key: float(X_test[i, j]) for j, key in enumerate(keys)}
            try:
                trained_pred = model.predict(feature_dict)
            except Exception:
                trained_pred = naive_pred.copy()
        else:
            trained_pred = naive_pred.copy()

        # Compute prediction confidence from raw (unscaled) features and
        # store it in the metadata so show_example_predictions() can
        # display a meaningful score instead of the always-1.0 blend_weight.
        meta_test[i]["prediction_confidence"] = _prediction_confidence(X_test[i])

        for m in CORE_METRICS:
            actual = float(y_test[i, metric_to_idx[m]])
            t_pred = trained_pred.get(m, 0.0)
            n_pred = naive_pred.get(m, 0.0)

            # Absolute errors
            t_abs = abs(t_pred - actual)
            n_abs = abs(n_pred - actual)
            trained_abs_errors[m].append(t_abs)
            naive_abs_errors[m].append(n_abs)

            # Percentage errors (avoid div by zero)
            if abs(actual) > 0.001:
                t_pct = abs(t_pred - actual) / abs(actual) * 100
                n_pct = abs(n_pred - actual) / abs(actual) * 100
                trained_pct_errors[m].append(t_pct)
                naive_pct_errors[m].append(n_pct)

                if t_pct <= 10:
                    trained_within_10[m] += 1
                if t_pct <= 20:
                    trained_within_20[m] += 1

            # Direction accuracy
            actual_change = actual - n_pred
            pred_change = t_pred - n_pred
            if abs(actual_change) > 0.001:
                direction_total[m] += 1
                if (actual_change > 0 and pred_change > 0) or \
                   (actual_change < 0 and pred_change < 0):
                    trained_direction[m] += 1

    # Aggregate report
    report: Dict[str, Any] = {
        "n_samples": n,
        "skipped_no_coverage": skipped_no_coverage,
        "per_metric": {},
    }

    print(f"\n{'='*80}")
    print(f"Backtest Report ({n} test samples, {skipped_no_coverage} skipped no coverage)")
    print(f"{'='*80}")
    print(f"\n{'Metric':<30} {'MAE':>8} {'MPE%':>8} {'<10%':>6} {'<20%':>6} {'Dir%':>6} {'Naive MAE':>10} {'Improv%':>8}")
    print(f"{'-'*88}")

    overall_trained_mse = []
    overall_naive_mse = []

    for m in CORE_METRICS:
        mae = float(np.mean(trained_abs_errors[m])) if trained_abs_errors[m] else 0.0
        mpe = float(np.mean(trained_pct_errors[m])) if trained_pct_errors[m] else 0.0
        n_pct_total = len(trained_pct_errors[m])
        w10 = trained_within_10[m] / n_pct_total * 100 if n_pct_total > 0 else 0.0
        w20 = trained_within_20[m] / n_pct_total * 100 if n_pct_total > 0 else 0.0
        dir_acc = trained_direction[m] / direction_total[m] * 100 if direction_total[m] > 0 else 0.0

        naive_mae = float(np.mean(naive_abs_errors[m])) if naive_abs_errors[m] else 0.0
        improvement = ((naive_mae - mae) / naive_mae * 100) if naive_mae > 0 else 0.0

        trained_mse = float(np.mean([e**2 for e in trained_abs_errors[m]])) if trained_abs_errors[m] else 0.0
        naive_mse = float(np.mean([e**2 for e in naive_abs_errors[m]])) if naive_abs_errors[m] else 0.0
        overall_trained_mse.append(trained_mse)
        overall_naive_mse.append(naive_mse)

        print(f"{m:<30} {mae:>8.4f} {mpe:>7.1f}% {w10:>5.1f}% {w20:>5.1f}% {dir_acc:>5.1f}% {naive_mae:>10.4f} {improvement:>7.1f}%")

        report["per_metric"][m] = {
            "mae": mae,
            "mean_pct_error": mpe,
            "within_10_pct": w10,
            "within_20_pct": w20,
            "direction_accuracy": dir_acc,
            "naive_mae": naive_mae,
            "improvement_vs_naive": improvement,
            "trained_mse": trained_mse,
            "naive_mse": naive_mse,
        }

    # Overall summary
    mean_trained_mse = float(np.mean(overall_trained_mse)) if overall_trained_mse else 0.0
    mean_naive_mse = float(np.mean(overall_naive_mse)) if overall_naive_mse else 0.0
    overall_improvement = (
        (mean_naive_mse - mean_trained_mse) / mean_naive_mse * 100
        if mean_naive_mse > 0 else 0.0
    )

    metrics_improved = sum(
        1 for m in CORE_METRICS
        if report["per_metric"][m]["improvement_vs_naive"] > 0
    )

    report["overall"] = {
        "mean_trained_mse": mean_trained_mse,
        "mean_naive_mse": mean_naive_mse,
        "overall_improvement_pct": overall_improvement,
        "metrics_improved": metrics_improved,
        "metrics_total": len(CORE_METRICS),
    }

    report["meta"] = {
        "n_train": len(meta_train) if meta_train else None,
        "n_test": n,
        "leakage_check": "passed" if meta_train is not None else "skipped — meta_train not provided",
    }

    print(f"\n{'='*80}")
    print(f"Overall: Trained MSE={mean_trained_mse:.6f}, Naive MSE={mean_naive_mse:.6f}")
    print(f"Improvement vs naive: {overall_improvement:.1f}%")
    print(f"Metrics improved: {metrics_improved}/{len(CORE_METRICS)}")
    print(f"{'='*80}")

    # Save report
    os.makedirs(_MODELS_DIR, exist_ok=True)
    report_path = os.path.join(_MODELS_DIR, "backtest_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    return report


def show_example_predictions(
    meta_test: List[Dict[str, Any]],
    n: int = 10,
) -> None:
    """Show example predictions for the highest-confidence test transfers.

    Parameters
    ----------
    meta_test : list[dict]
        Metadata for test samples (must include from_club, to_club, etc.).
        If ``run_backtest`` has been called first, each entry also has
        ``prediction_confidence`` -- a feature-based quality score with
        meaningful variance across players.
    n : int
        Number of examples to show.
    """
    # Prefer prediction_confidence (from run_backtest) over the
    # blend_weight-based "confidence" which saturates at 1.0 for most
    # professional players.
    conf_key = "prediction_confidence"
    if not any(m.get(conf_key) is not None for m in meta_test):
        conf_key = "confidence"  # fall back to blend_weight

    # Sort by confidence descending
    sorted_meta = sorted(
        meta_test,
        key=lambda m: m.get(conf_key, 0),
        reverse=True,
    )

    # Deduplicate by player_id — keep the highest-confidence entry per player.
    # This prevents the same player appearing multiple times when a transfer
    # is recorded bidirectionally or across multiple seasons.
    seen_players: set = set()
    deduped: List[Dict[str, Any]] = []
    for m in sorted_meta:
        pid = m.get("player_id")
        if pid is not None and pid in seen_players:
            continue
        if pid is not None:
            seen_players.add(pid)
        deduped.append(m)

    # Filter to high-confidence (weight > 0.7)
    high_conf = [m for m in deduped if m.get(conf_key, 0) > 0.7]
    if not high_conf:
        high_conf = deduped  # Fall back to all (deduped) if none above threshold

    examples = high_conf[:n]

    print(f"\n{'='*60}")
    print(f"Example Predictions (top {len(examples)} by confidence)")
    print(f"{'='*60}")

    # Top 5 metrics to show
    top_metrics = [
        "expected_goals", "expected_assists", "shots",
        "successful_dribbles", "successful_passes",
    ]

    for i, meta in enumerate(examples):
        print(f"\n  {i+1}. {meta.get('player_name', 'Unknown')}")
        print(f"     {meta.get('from_club', '?')} -> {meta.get('to_club', '?')}")
        print(f"     Confidence: {meta.get(conf_key, 0):.2f}")

        pre_per90 = meta.get("pre_per90", {})
        for m in top_metrics:
            pre_val = pre_per90.get(m, 0.0)
            print(f"     {m}: pre={pre_val:.3f}")
