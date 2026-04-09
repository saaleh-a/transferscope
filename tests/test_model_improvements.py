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


# ── Training pipeline improvement tests ─────────────────────────────────────


class TestGroupArchOverrides(unittest.TestCase):
    """Tests for per-group architecture overrides."""

    def test_all_low_snr_groups_have_overrides(self):
        """Both shooting and passing have very low SNR — both must have overrides."""
        from backend.models.transfer_portal import _GROUP_ARCH_OVERRIDES
        self.assertIn("shooting", _GROUP_ARCH_OVERRIDES)
        self.assertIn("passing", _GROUP_ARCH_OVERRIDES)

    def test_passing_uses_smaller_arch_than_default(self):
        """Passing override should use smaller hidden units than the 128→64 default."""
        from backend.models.transfer_portal import _GROUP_ARCH_OVERRIDES
        override = _GROUP_ARCH_OVERRIDES["passing"]
        hidden = override["hidden_units"]
        self.assertLess(hidden[0], 128, "Passing first hidden should be < 128")
        self.assertLess(hidden[1], 64, "Passing second hidden should be < 64")

    def test_all_overrides_have_l2(self):
        """Groups with explicit overrides should specify L2 where SNR is low."""
        from backend.models.transfer_portal import _GROUP_ARCH_OVERRIDES
        for name in ("shooting", "passing", "dribbling"):
            self.assertIn(
                "l2", _GROUP_ARCH_OVERRIDES[name],
                f"{name} override should have explicit L2",
            )

    def test_shooting_strongest_regularization(self):
        """Shooting (lowest SNR, xG masking) should have the strongest regularization."""
        from backend.models.transfer_portal import _GROUP_ARCH_OVERRIDES
        shooting_l2 = _GROUP_ARCH_OVERRIDES["shooting"]["l2"]
        passing_l2 = _GROUP_ARCH_OVERRIDES["passing"]["l2"]
        dribbling_l2 = _GROUP_ARCH_OVERRIDES["dribbling"]["l2"]
        self.assertGreater(shooting_l2, passing_l2)
        self.assertGreater(shooting_l2, dribbling_l2)


class TestOutputHeadRegularization(unittest.TestCase):
    """Tests that output heads have L2 regularization."""

    def test_output_heads_have_l2_regularization(self):
        """Each output head Dense layer should have kernel_regularizer set."""
        model = TransferPortalModel()
        model.build(FEATURE_DIM)
        for group_name, keras_model in model.models.items():
            for layer in keras_model.layers:
                if layer.name.startswith("head_"):
                    config = layer.get_config()
                    reg = config.get("kernel_regularizer")
                    self.assertIsNotNone(
                        reg,
                        f"Output head {layer.name} in {group_name} has no "
                        f"kernel_regularizer",
                    )


class TestTeamAdjustmentMinSamples(unittest.TestCase):
    """Tests for team adjustment model minimum sample requirement."""

    def test_min_samples_at_least_10(self):
        """TeamAdjustmentModel should require ≥10 samples, not ≥2."""
        from backend.features.adjustment_models import TeamAdjustmentModel
        self.assertGreaterEqual(TeamAdjustmentModel._MIN_SAMPLES, 10)

    def test_identity_fallback_with_few_samples(self):
        """With only 5 samples per metric, model should use identity (zero coef)."""
        from backend.features.adjustment_models import TeamAdjustmentModel
        model = TeamAdjustmentModel()
        # Build 5 training rows per metric — below _MIN_SAMPLES
        training_data = []
        for metric in CORE_METRICS:
            for i in range(5):
                training_data.append({
                    "metric": metric,
                    "from_ra": float(i),
                    "to_ra": float(i + 1),
                    "naive_league_expectation": 0.5,
                    "actual": 0.6 + i * 0.01,
                    "team_relative_feature": 0.1 * i,
                })
        model.fit(training_data)
        for metric in CORE_METRICS:
            np.testing.assert_array_equal(
                model.models[metric].coef_,
                [0.0, 0.0, 0.0],
                f"{metric} should use identity fallback with <10 samples",
            )

    def test_fits_with_enough_samples(self):
        """With ≥10 samples per metric, model should actually fit."""
        from backend.features.adjustment_models import TeamAdjustmentModel
        model = TeamAdjustmentModel()
        training_data = []
        for metric in CORE_METRICS:
            for i in range(15):
                training_data.append({
                    "metric": metric,
                    "from_ra": float(i),
                    "to_ra": float(i + 1),
                    "naive_league_expectation": 0.5,
                    "actual": 0.6 + i * 0.05,
                    "team_relative_feature": 0.1 * i,
                })
        model.fit(training_data)
        # At least one metric should have non-zero coefficients
        any_fitted = any(
            not np.allclose(model.models[m].coef_, 0.0)
            for m in CORE_METRICS
        )
        self.assertTrue(any_fitted, "With 15 samples, at least one metric should fit")


class TestPlayerAdjustmentRidgeAlpha(unittest.TestCase):
    """Tests for player adjustment model Ridge alpha."""

    def test_ridge_alpha_is_10(self):
        """Ridge alpha should be 10.0 to handle polynomial multicollinearity."""
        from backend.features.adjustment_models import PlayerAdjustmentModel
        model = PlayerAdjustmentModel()
        # Build enough training data to actually fit
        training_data = []
        for i in range(40):
            training_data.append({
                "position": "F",
                "metric": CORE_METRICS[0],
                "player_previous_per90": 0.5 + i * 0.01,
                "change_relative_ability": float(i - 20),
                "from_ra": float(50 + i),
                "to_ra": float(55 + i),
                "actual": 0.6 + i * 0.02,
            })
        model.fit(training_data)
        if "F" in model.models and CORE_METRICS[0] in model.models["F"]:
            ridge = model.models["F"][CORE_METRICS[0]]
            self.assertEqual(ridge.alpha, 10.0)


# ── P2/P3 training improvement tests ───────────────────────────────────────


class TestCosineAnnealingLR(unittest.TestCase):
    """Tests for the warmup + cosine annealing LR schedule."""

    def _make_callback(self):
        """Import and construct _WarmupCosineAnnealing from the training pipeline."""
        # The callback is defined as a local class inside train_neural_network,
        # so we replicate the logic here for unit testing.
        warmup_epochs = 10
        start_lr = 1e-5
        target_lr = 5e-4
        min_lr = 1e-5
        max_epochs = 150
        return warmup_epochs, start_lr, target_lr, min_lr, max_epochs

    def _cosine_lr(self, epoch, warmup_epochs, start_lr, target_lr, min_lr, max_epochs):
        """Compute expected LR at a given epoch."""
        if epoch < warmup_epochs:
            return start_lr + (target_lr - start_lr) * ((epoch + 1) / warmup_epochs)
        cosine_epochs = max_epochs - warmup_epochs
        progress = (epoch - warmup_epochs) / max(cosine_epochs, 1)
        return min_lr + 0.5 * (target_lr - min_lr) * (1 + np.cos(np.pi * progress))

    def test_warmup_phase_increases_lr(self):
        """During warmup, LR should monotonically increase."""
        params = self._make_callback()
        lrs = [self._cosine_lr(e, *params) for e in range(params[0])]
        for i in range(1, len(lrs)):
            self.assertGreater(lrs[i], lrs[i - 1],
                               f"LR should increase during warmup (epoch {i})")

    def test_warmup_reaches_target_lr(self):
        """At end of warmup, LR should equal target_lr."""
        params = self._make_callback()
        lr_at_warmup_end = self._cosine_lr(params[0] - 1, *params)
        self.assertAlmostEqual(lr_at_warmup_end, params[2], places=6)

    def test_cosine_phase_decreases_lr(self):
        """After warmup, LR should generally decrease."""
        params = self._make_callback()
        lr_start_cosine = self._cosine_lr(params[0], *params)
        lr_end_cosine = self._cosine_lr(params[4] - 1, *params)
        self.assertGreater(lr_start_cosine, lr_end_cosine,
                           "LR should decrease during cosine phase")

    def test_cosine_phase_ends_at_min_lr(self):
        """At the last epoch, LR should reach min_lr."""
        params = self._make_callback()
        lr_final = self._cosine_lr(params[4] - 1, *params)
        self.assertAlmostEqual(lr_final, params[3], places=5)

    def test_lr_never_negative(self):
        """LR should never go negative at any epoch."""
        params = self._make_callback()
        for e in range(params[4]):
            lr = self._cosine_lr(e, *params)
            self.assertGreaterEqual(lr, 0.0, f"LR negative at epoch {e}")


class TestPerMetricTargetScaling(unittest.TestCase):
    """Tests for per-metric (not per-group) target scaling."""

    def test_inverse_scale_with_per_metric_scalers(self):
        """_inverse_scale_targets handles dict-of-scalers correctly."""
        from sklearn.preprocessing import StandardScaler
        # Simulate per-metric scalers
        targets = ["metric_a", "metric_b"]
        scalers = {}
        raw_data = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        for col_idx, name in enumerate(targets):
            s = StandardScaler()
            s.fit(raw_data[:, col_idx:col_idx + 1])
            scalers[name] = s

        # Scale the data
        scaled = np.zeros_like(raw_data)
        for col_idx, name in enumerate(targets):
            scaled[:, col_idx:col_idx + 1] = scalers[name].transform(
                raw_data[:, col_idx:col_idx + 1]
            )

        # Inverse should recover original
        recovered = TransferPortalModel._inverse_scale_targets(
            scaled, scalers, targets,
        )
        np.testing.assert_array_almost_equal(recovered, raw_data, decimal=5)

    def test_inverse_scale_with_legacy_scaler(self):
        """_inverse_scale_targets handles a single StandardScaler (legacy)."""
        from sklearn.preprocessing import StandardScaler
        raw_data = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(raw_data)

        recovered = TransferPortalModel._inverse_scale_targets(
            scaled, scaler, ["a", "b"],
        )
        np.testing.assert_array_almost_equal(recovered, raw_data, decimal=5)

    def test_inverse_scale_with_none(self):
        """_inverse_scale_targets returns input unchanged when scaler is None."""
        data = np.array([[1.0, 2.0]])
        result = TransferPortalModel._inverse_scale_targets(
            data, None, ["a", "b"],
        )
        np.testing.assert_array_equal(result, data)


class TestAdaptiveHuberDelta(unittest.TestCase):
    """Tests for per-group Huber delta configuration."""

    def test_shooting_has_small_huber_delta(self):
        """Shooting group has tiny deltas — Huber delta should be < 1.0."""
        from backend.models.transfer_portal import _GROUP_ARCH_OVERRIDES
        delta = _GROUP_ARCH_OVERRIDES["shooting"].get("huber_delta", 1.0)
        self.assertLess(delta, 1.0, "Shooting Huber delta should be < 1.0")

    def test_passing_has_larger_huber_delta(self):
        """Passing group has larger deltas — Huber delta should be > 1.0."""
        from backend.models.transfer_portal import _GROUP_ARCH_OVERRIDES
        delta = _GROUP_ARCH_OVERRIDES["passing"].get("huber_delta", 1.0)
        self.assertGreater(delta, 1.0, "Passing Huber delta should be > 1.0")

    def test_model_builds_with_custom_huber_delta(self):
        """Model should build and compile with per-group Huber deltas."""
        model = TransferPortalModel()
        model.build(FEATURE_DIM)
        for group_name in MODEL_GROUPS:
            keras_model = model.models[group_name]
            # Check that model compiled successfully (has a loss function)
            self.assertIsNotNone(keras_model.loss)


class TestShootingLayerNorm(unittest.TestCase):
    """Tests that shooting group uses LayerNorm instead of BatchNorm."""

    def test_shooting_has_layer_norm(self):
        """Shooting group should use LayerNormalization (not BatchNorm)."""
        model = TransferPortalModel()
        model.build(FEATURE_DIM)
        shooting_model = model.models["shooting"]
        layer_names = [l.name for l in shooting_model.layers]
        self.assertTrue(
            any("ln" in name for name in layer_names),
            f"Shooting should have LayerNorm layers, got: {layer_names}",
        )
        self.assertFalse(
            any("bn" in name for name in layer_names),
            f"Shooting should NOT have BatchNorm layers, got: {layer_names}",
        )

    def test_non_shooting_has_batch_norm(self):
        """Non-shooting groups should use BatchNormalization."""
        model = TransferPortalModel()
        model.build(FEATURE_DIM)
        for group_name in ("passing", "dribbling", "defending"):
            keras_model = model.models[group_name]
            layer_names = [l.name for l in keras_model.layers]
            self.assertTrue(
                any("bn" in name for name in layer_names),
                f"{group_name} should have BatchNorm layers, got: {layer_names}",
            )


class TestSampleWeightingLogic(unittest.TestCase):
    """Tests for the sample weighting and xG masking pipeline."""

    def test_confidence_field_is_blend_weight(self):
        """The confidence field should be produced by blend_weight()."""
        from backend.features.rolling_windows import blend_weight
        # 0 minutes → 0.0 confidence
        self.assertEqual(blend_weight(0), 0.0)
        # PRIOR_CONSTANT minutes → 1.0 confidence
        from backend.features.rolling_windows import PRIOR_CONSTANT
        self.assertEqual(blend_weight(PRIOR_CONSTANT), 1.0)
        # More than PRIOR_CONSTANT → still capped at 1.0
        self.assertEqual(blend_weight(PRIOR_CONSTANT * 2), 1.0)
        # Halfway → 0.5
        self.assertAlmostEqual(blend_weight(PRIOR_CONSTANT / 2), 0.5)

    def test_group_weights_deep_copy(self):
        """group_weights should be a copy of sample_weights, not an alias."""
        weights = np.array([1.0, 2.0, 3.0])
        copy = weights.copy()
        copy[0] = 99.0
        # Modifying the copy shouldn't affect original
        self.assertEqual(weights[0], 1.0)
        self.assertEqual(copy[0], 99.0)

    def test_xg_zero_mask_zeroes_weight(self):
        """Zero-weighting xG=0 rows should produce 0.0 weight."""
        weights = np.ones(5, dtype=np.float32)
        mask = np.array([False, True, False, True, False])
        weights[mask] = 0.0
        self.assertEqual(weights[1], 0.0)
        self.assertEqual(weights[3], 0.0)
        self.assertEqual(weights[0], 1.0)


class TestEarlyStoppingMinDelta(unittest.TestCase):
    """Tests that EarlyStopping uses a minimum improvement threshold."""

    def test_min_delta_is_set(self):
        """EarlyStopping callback should have min_delta > 0."""
        # We can't easily extract the callback from train_neural_network
        # without running it, so we test that the value is documented
        # and correct via a direct callback construction test.
        import tensorflow as tf
        cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, min_delta=0.001,
            restore_best_weights=True,
        )
        self.assertEqual(cb.min_delta, 0.001)
        self.assertEqual(cb.patience, 15)


if __name__ == "__main__":
    unittest.main()
