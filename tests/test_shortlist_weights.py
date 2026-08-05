"""The metric-weight sliders must actually change the ranking.

Weights were applied *before* ``StandardScaler``, where they cancel exactly:
scaling a column by a positive constant w scales its mean and std by w too, so
(w·x − w·μ)/(w·σ) = (x − μ)/σ. Measured difference between a uniform and a
heavily skewed weight vector was 1.3e-15 — floating-point noise — and the
resulting candidate order was identical. The sliders were an on/off switch
while the UI said "1.0 means the metric is fully considered".
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.models.shortlist_scorer import CORE_METRICS, Candidate, _build_feature_matrix


def _candidate(name: str, per90: dict) -> Candidate:
    return Candidate(
        player_id=abs(hash(name)) % 100000,
        name=name,
        team="Test FC",
        league="ENG1",
        position="F",
        age=25,
        minutes_played=2000,
        predicted_per90=per90,
        current_per90=per90,
    )


def _fixture():
    """Two metrics that disagree about who is best."""
    rng = np.random.default_rng(7)
    candidates = []
    for i in range(12):
        candidates.append(
            _candidate(
                f"p{i}",
                {
                    "expected_goals": 0.1 + i * 0.05,
                    "successful_passes": 60.0 - i * 3.0,
                    "clearances": float(rng.integers(0, 6)),
                    "interceptions": float(rng.integers(0, 4)),
                },
            )
        )
    reference = {
        "expected_goals": 0.55,
        "successful_passes": 30.0,
        "clearances": 2.0,
        "interceptions": 1.0,
    }
    return candidates, reference


class TestWeightsAffectGeometry:
    def test_weights_change_the_feature_matrix(self):
        candidates, reference = _fixture()
        uniform = {m: 0.5 for m in CORE_METRICS}
        skewed = dict.fromkeys(CORE_METRICS, 0.5)
        skewed["expected_goals"] = 1.0
        skewed["successful_passes"] = 0.01

        Xa, ma, _, _ = _build_feature_matrix(candidates, reference, uniform)
        Xb, mb, _, _ = _build_feature_matrix(candidates, reference, skewed)

        assert ma == mb, "same active metrics, so shapes are comparable"
        assert np.abs(Xa - Xb).max() > 1e-3, (
            "weights had no effect on the feature matrix — they cancelled "
            "inside StandardScaler"
        )

    def test_weights_change_distance_ordering(self):
        candidates, reference = _fixture()

        def order(weights):
            X, _, _, ref = _build_feature_matrix(candidates, reference, weights)
            d = np.linalg.norm(X - ref, axis=1)
            return [candidates[i].name for i in np.argsort(d)]

        goals_first = dict.fromkeys(CORE_METRICS, 0.01)
        goals_first["expected_goals"] = 1.0
        passes_first = dict.fromkeys(CORE_METRICS, 0.01)
        passes_first["successful_passes"] = 1.0

        assert order(goals_first) != order(passes_first), (
            "ranking is identical whether the user prioritises goals or "
            "passes — the sliders do nothing"
        )

    def test_higher_weight_pulls_ranking_toward_that_metric(self):
        """Weighting xG heavily should favour the closest xG match."""
        candidates, reference = _fixture()
        weights = dict.fromkeys(CORE_METRICS, 0.01)
        weights["expected_goals"] = 1.0

        X, metrics, _, ref = _build_feature_matrix(candidates, reference, weights)
        d = np.linalg.norm(X - ref, axis=1)
        winner = candidates[int(np.argmin(d))]

        closest_xg = min(
            candidates,
            key=lambda c: abs(c.predicted_per90["expected_goals"] - reference["expected_goals"]),
        )
        assert winner.name == closest_xg.name

    def test_zero_weight_metrics_are_dropped_entirely(self):
        candidates, reference = _fixture()
        weights = dict.fromkeys(CORE_METRICS, 0.0)
        weights["expected_goals"] = 1.0
        weights["clearances"] = 0.5

        _, metrics, _, _ = _build_feature_matrix(candidates, reference, weights)
        assert set(metrics) == {"expected_goals", "clearances"}

    def test_uniform_weights_preserve_relative_geometry(self):
        """Scaling everything equally must not reorder anything."""
        candidates, reference = _fixture()

        def order(w):
            X, _, _, ref = _build_feature_matrix(
                candidates, reference, dict.fromkeys(CORE_METRICS, w)
            )
            d = np.linalg.norm(X - ref, axis=1)
            return [candidates[i].name for i in np.argsort(d)]

        assert order(0.25) == order(1.0)

    @pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
    def test_no_nan_for_any_weight(self, weight):
        candidates, reference = _fixture()
        weights = dict.fromkeys(CORE_METRICS, 0.5)
        weights["expected_goals"] = weight
        X, _, _, ref = _build_feature_matrix(candidates, reference, weights)
        assert np.isfinite(X).all()
        assert np.isfinite(ref).all()
