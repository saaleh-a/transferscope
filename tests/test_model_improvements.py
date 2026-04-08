"""Tests for Phase 4 model architecture improvements:

- Interaction features in build_feature_dict()
- Feature importance analysis via gradient-based sensitivity
- Prediction confidence estimation via Monte Carlo dropout
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from backend.data.sofascore_client import CORE_METRICS
from backend.models.transfer_portal import (
    FEATURE_DIM,
    GROUP_FEATURE_SUBSETS,
    MODEL_GROUPS,
    TransferPortalModel,
    _feature_keys,
    build_feature_dict,
)


def _make_dummy_inputs():
    """Return dummy inputs for build_feature_dict()."""
    player_per90 = {m: float(i + 1) for i, m in enumerate(CORE_METRICS)}
    team_pos_current = {m: float(i + 0.5) for i, m in enumerate(CORE_METRICS)}
    team_pos_target = {m: float(i + 1.5) for i, m in enumerate(CORE_METRICS)}
    return dict(
        player_per90=player_per90,
        team_ability_current=60.0,
        team_ability_target=75.0,
        league_ability_current=55.0,
        league_ability_target=70.0,
        team_pos_current=team_pos_current,
        team_pos_target=team_pos_target,
    )


def _make_feature_dict(**overrides):
    """Build a feature dict from dummy inputs with optional overrides."""
    inputs = _make_dummy_inputs()
    inputs.update(overrides)
    return build_feature_dict(**inputs)


# ── Interaction features ────────────────────────────────────────────────────


class TestInteractionFeatures(unittest.TestCase):
    """Tests for the 3 interaction features added to build_feature_dict()."""

    def test_interaction_features_present(self):
        """build_feature_dict returns 93 keys including interaction features."""
        fd = _make_feature_dict()
        self.assertEqual(len(fd), 93)
        self.assertIn("interaction_ability_gap", fd)
        self.assertIn("interaction_gap_squared", fd)
        self.assertIn("interaction_league_gap", fd)

    def test_interaction_ability_gap(self):
        """interaction_ability_gap = target - current team ability."""
        fd = _make_feature_dict(team_ability_current=40.0, team_ability_target=80.0)
        self.assertAlmostEqual(fd["interaction_ability_gap"], 40.0)

    def test_interaction_ability_gap_negative(self):
        """interaction_ability_gap is negative for a downgrade move."""
        fd = _make_feature_dict(team_ability_current=80.0, team_ability_target=40.0)
        self.assertAlmostEqual(fd["interaction_ability_gap"], -40.0)

    def test_interaction_gap_squared(self):
        """interaction_gap_squared = (target - current)²."""
        fd = _make_feature_dict(team_ability_current=40.0, team_ability_target=80.0)
        self.assertAlmostEqual(fd["interaction_gap_squared"], 1600.0)

    def test_interaction_gap_squared_same_for_sign(self):
        """Squared gap is identical regardless of move direction."""
        fd_up = _make_feature_dict(team_ability_current=40.0, team_ability_target=80.0)
        fd_down = _make_feature_dict(team_ability_current=80.0, team_ability_target=40.0)
        self.assertAlmostEqual(
            fd_up["interaction_gap_squared"],
            fd_down["interaction_gap_squared"],
        )

    def test_interaction_league_gap(self):
        """interaction_league_gap = target - current league ability."""
        fd = _make_feature_dict(league_ability_current=50.0, league_ability_target=70.0)
        self.assertAlmostEqual(fd["interaction_league_gap"], 20.0)

    def test_interaction_zero_gap(self):
        """Interaction features are zero when moving within same team quality."""
        fd = _make_feature_dict(
            team_ability_current=60.0, team_ability_target=60.0,
            league_ability_current=55.0, league_ability_target=55.0,
        )
        self.assertAlmostEqual(fd["interaction_ability_gap"], 0.0)
        self.assertAlmostEqual(fd["interaction_gap_squared"], 0.0)
        self.assertAlmostEqual(fd["interaction_league_gap"], 0.0)


# ── Feature dimension & keys ────────────────────────────────────────────────


class TestFeatureDimension(unittest.TestCase):
    """Tests that FEATURE_DIM and _feature_keys() reflect the 93 features."""

    def test_feature_dim_is_93(self):
        """FEATURE_DIM must be 93 (13 core + 10 additional + 4 team/league + 2 raw_elo + 2 reep + 26 team_pos + 3 interaction + 3 relative + 13 league_norm + 13 league_mean_ratio + 4 position_one_hot)."""
        self.assertEqual(FEATURE_DIM, 93)

    def test_feature_keys_length(self):
        """_feature_keys() returns exactly 93 items."""
        self.assertEqual(len(_feature_keys()), 93)

    def test_feature_keys_contain_interactions(self):
        """_feature_keys() includes all 3 interaction feature names."""
        keys = _feature_keys()
        self.assertIn("interaction_ability_gap", keys)
        self.assertIn("interaction_gap_squared", keys)
        self.assertIn("interaction_league_gap", keys)

    def test_interaction_keys_at_expected_positions(self):
        """Interaction features are at indices 57-59 in the key list (shifted +10 for additional metrics)."""
        keys = _feature_keys()
        self.assertEqual(keys[57], "interaction_ability_gap")
        self.assertEqual(keys[58], "interaction_gap_squared")
        self.assertEqual(keys[59], "interaction_league_gap")


# ── GROUP_FEATURE_SUBSETS ────────────────────────────────────────────────────


class TestGroupFeatureSubsets(unittest.TestCase):
    """Tests that all 4 model groups include interaction features."""

    def test_group_feature_subsets_include_interactions(self):
        """Every group in GROUP_FEATURE_SUBSETS contains the 3 interaction keys."""
        interaction_keys = [
            "interaction_ability_gap",
            "interaction_gap_squared",
            "interaction_league_gap",
        ]
        for group_name in MODEL_GROUPS:
            subset = GROUP_FEATURE_SUBSETS[group_name]
            for key in interaction_keys:
                self.assertIn(
                    key, subset,
                    f"{key} missing from {group_name} feature subset",
                )

    def test_shooting_subset_size(self):
        """Shooting group: 22 original + 3 additional + 3 relative + 4 league_norm + 4 league_mean_ratio + 4 position = 40 features."""
        self.assertEqual(len(GROUP_FEATURE_SUBSETS["shooting"]), 40)

    def test_passing_subset_size(self):
        """Passing group: 31 original + 2 additional + 3 relative + 7 league_norm + 7 league_mean_ratio + 4 position = 54 features."""
        self.assertEqual(len(GROUP_FEATURE_SUBSETS["passing"]), 54)

    def test_dribbling_subset_size(self):
        """Dribbling group: 13 original + 4 additional + 3 relative + 1 league_norm + 1 league_mean_ratio + 4 position = 26 features."""
        self.assertEqual(len(GROUP_FEATURE_SUBSETS["dribbling"]), 26)

    def test_defending_subset_size(self):
        """Defending group: 19 original + 5 additional + 3 relative + 3 league_norm + 3 league_mean_ratio + 4 position = 37 features."""
        self.assertEqual(len(GROUP_FEATURE_SUBSETS["defending"]), 37)


# ── Backward compatibility ──────────────────────────────────────────────────


class TestBackwardCompatibility(unittest.TestCase):
    """Ensure old callers of build_feature_dict() still work."""

    def test_build_feature_dict_backward_compatible(self):
        """Same API signature, old callers get interaction features for free."""
        fd = _make_feature_dict()
        # All original 43 keys still present
        for m in CORE_METRICS:
            self.assertIn(f"player_{m}", fd)
            self.assertIn(f"team_pos_current_{m}", fd)
            self.assertIn(f"team_pos_target_{m}", fd)
        self.assertIn("team_ability_current", fd)
        self.assertIn("team_ability_target", fd)
        self.assertIn("league_ability_current", fd)
        self.assertIn("league_ability_target", fd)
        # Plus 3 new interaction keys
        self.assertIn("interaction_ability_gap", fd)
        self.assertIn("interaction_gap_squared", fd)
        self.assertIn("interaction_league_gap", fd)

    def test_missing_metrics_default_to_zero(self):
        """Player metrics not in dict default to 0.0 (unchanged behavior)."""
        fd = build_feature_dict(
            player_per90={},
            team_ability_current=50.0,
            team_ability_target=50.0,
            league_ability_current=50.0,
            league_ability_target=50.0,
            team_pos_current={},
            team_pos_target={},
        )
        for m in CORE_METRICS:
            self.assertEqual(fd[f"player_{m}"], 0.0)
        # Interaction features still computed correctly
        self.assertAlmostEqual(fd["interaction_ability_gap"], 0.0)


# ── Feature importance (untrained) ──────────────────────────────────────────


class TestFeatureImportanceUntrained(unittest.TestCase):
    """Tests for compute_feature_importance when model is not trained."""

    def test_compute_feature_importance_untrained(self):
        """Returns empty dict when no model is built/trained."""
        model = TransferPortalModel()
        fd = _make_feature_dict()
        result = model.compute_feature_importance(fd)
        self.assertEqual(result, {})


# ── Feature importance (mocked model) ───────────────────────────────────────


class TestFeatureImportanceTrained(unittest.TestCase):
    """Tests for compute_feature_importance with a mocked trained model."""

    def _build_model(self):
        """Build a real (untrained) model to test gradient flow."""
        model = TransferPortalModel()
        model.build(FEATURE_DIM)
        return model

    def test_returns_all_groups(self):
        """Feature importance dict has an entry for each model group."""
        model = self._build_model()
        fd = _make_feature_dict()
        result = model.compute_feature_importance(fd)
        for group_name in MODEL_GROUPS:
            self.assertIn(group_name, result)

    def test_importance_sums_to_one(self):
        """Per-group importances are normalized to sum to 1.0."""
        model = self._build_model()
        fd = _make_feature_dict()
        result = model.compute_feature_importance(fd)
        for group_name, importances in result.items():
            total = sum(importances.values())
            self.assertAlmostEqual(total, 1.0, places=5,
                                   msg=f"{group_name} importance sum != 1.0")

    def test_importance_keys_match_group_subset(self):
        """Feature names in importance dict match GROUP_FEATURE_SUBSETS."""
        model = self._build_model()
        fd = _make_feature_dict()
        result = model.compute_feature_importance(fd)
        for group_name, importances in result.items():
            self.assertEqual(
                set(importances.keys()),
                set(GROUP_FEATURE_SUBSETS[group_name]),
            )

    def test_importance_values_non_negative(self):
        """All importance values are >= 0 (absolute gradients)."""
        model = self._build_model()
        fd = _make_feature_dict()
        result = model.compute_feature_importance(fd)
        for group_name, importances in result.items():
            for feat, val in importances.items():
                self.assertGreaterEqual(val, 0.0, f"{group_name}/{feat} < 0")


# ── Predict with confidence (untrained fallback) ────────────────────────────


class TestPredictWithConfidenceUntrained(unittest.TestCase):
    """Tests for predict_with_confidence fallback path."""

    @patch.object(TransferPortalModel, "is_trained", return_value=False)
    @patch("backend.features.adjustment_models.paper_heuristic_predict")
    def test_predict_with_confidence_untrained_fallback(
        self, mock_heuristic, _mock_trained,
    ):
        """Falls back to predict() (heuristic) with zero std when untrained."""
        mock_heuristic.return_value = {"expected_goals": 0.5, "shots": 2.0}
        model = TransferPortalModel()
        fd = _make_feature_dict()
        pre_per90 = {m: 1.0 for m in CORE_METRICS}

        means, stds = model.predict_with_confidence(fd, pre_per90)

        self.assertIn("expected_goals", means)
        for metric, std_val in stds.items():
            self.assertEqual(std_val, 0.0, f"std for {metric} should be 0")

    @patch.object(TransferPortalModel, "is_trained", return_value=False)
    @patch("backend.features.adjustment_models.paper_heuristic_predict")
    def test_predict_with_confidence_returns_tuple(
        self, mock_heuristic, _mock_trained,
    ):
        """Return value is a (dict, dict) tuple."""
        mock_heuristic.return_value = {"expected_goals": 0.5}
        model = TransferPortalModel()
        fd = _make_feature_dict()
        result = model.predict_with_confidence(fd, {})
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], dict)
        self.assertIsInstance(result[1], dict)


# ── Predict with confidence (mocked trained model) ──────────────────────────


class TestPredictWithConfidenceTrained(unittest.TestCase):
    """Tests for predict_with_confidence with a built (random-weight) model."""

    def _build_model(self):
        model = TransferPortalModel()
        model.build(FEATURE_DIM)
        return model

    def test_returns_all_metrics(self):
        """Mean predictions contain all 13 target metrics."""
        model = self._build_model()
        fd = _make_feature_dict()
        pre_per90 = {m: 1.0 for m in CORE_METRICS}
        means, stds = model.predict_with_confidence(fd, pre_per90, n_samples=5)
        all_targets = [t for targets in MODEL_GROUPS.values() for t in targets]
        for t in all_targets:
            self.assertIn(t, means, f"Missing metric {t} in means")
            self.assertIn(t, stds, f"Missing metric {t} in stds")

    def test_std_non_negative(self):
        """All std_dev values are >= 0."""
        model = self._build_model()
        fd = _make_feature_dict()
        pre_per90 = {m: 1.0 for m in CORE_METRICS}
        _, stds = model.predict_with_confidence(fd, pre_per90, n_samples=5)
        for metric, std_val in stds.items():
            self.assertGreaterEqual(std_val, 0.0, f"Negative std for {metric}")

    def test_means_non_negative(self):
        """All mean predictions are >= 0 (per-90 values can't be negative)."""
        model = self._build_model()
        fd = _make_feature_dict()
        pre_per90 = {m: 1.0 for m in CORE_METRICS}
        means, _ = model.predict_with_confidence(fd, pre_per90, n_samples=5)
        for metric, val in means.items():
            self.assertGreaterEqual(val, 0.0, f"Negative mean for {metric}")


if __name__ == "__main__":
    unittest.main()
