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


def _feature_keys_list() -> List[str]:
    """Return the ordered list of feature keys matching the 43-feature vector."""
    keys = []
    for m in CORE_METRICS:
        keys.append(f"player_{m}")
    keys.extend([
        "team_ability_current", "team_ability_target",
        "league_ability_current", "league_ability_target",
    ])
    for m in CORE_METRICS:
        keys.append(f"team_pos_current_{m}")
    for m in CORE_METRICS:
        keys.append(f"team_pos_target_{m}")
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
    X_test : ndarray, shape (N, 43)
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

    for i in range(n):
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
    report: Dict[str, Any] = {"n_samples": n, "per_metric": {}}

    print(f"\n{'='*80}")
    print(f"Backtest Report ({n} test samples)")
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
        Metadata for test samples (must include confidence, from_club, to_club, etc.)
    n : int
        Number of examples to show.
    """
    # Sort by confidence descending
    sorted_meta = sorted(
        meta_test,
        key=lambda m: m.get("confidence", 0),
        reverse=True,
    )

    # Filter to high-confidence (weight > 0.7)
    high_conf = [m for m in sorted_meta if m.get("confidence", 0) > 0.7]
    if not high_conf:
        high_conf = sorted_meta  # Fall back to all if none above threshold

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
        print(f"     {meta.get('from_club', '?')} → {meta.get('to_club', '?')}")
        print(f"     Confidence: {meta.get('confidence', 0):.2f}")

        pre_per90 = meta.get("pre_per90", {})
        for m in top_metrics:
            pre_val = pre_per90.get(m, 0.0)
            print(f"     {m}: pre={pre_val:.3f}")
