"""Tests for artefact consistency guards and the shared theme components.

The artefact tests exist because the app once shipped a 4-group / 93-feature
set of saved weights against 6-group / 94-feature code, which surfaced only at
prediction time as an opaque sklearn ValueError.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from backend.models.transfer_portal import (
    FEATURE_DIM,
    GROUP_FEATURE_SUBSETS,
    MODEL_GROUPS,
    TransferPortalModel,
    _feature_keys,
)


class TestFeatureContract(unittest.TestCase):
    """The feature vector and group definitions must agree with each other."""

    def test_feature_dim_matches_feature_keys(self):
        self.assertEqual(len(_feature_keys()), FEATURE_DIM)

    def test_feature_keys_are_unique(self):
        keys = _feature_keys()
        self.assertEqual(len(keys), len(set(keys)))

    def test_every_group_has_a_feature_subset(self):
        self.assertEqual(set(MODEL_GROUPS), set(GROUP_FEATURE_SUBSETS))

    def test_group_subsets_reference_real_features(self):
        known = set(_feature_keys())
        for group, subset in GROUP_FEATURE_SUBSETS.items():
            unknown = [f for f in subset if f not in known]
            self.assertEqual(unknown, [], f"group {group} references {unknown}")

    def test_all_core_metrics_are_covered_exactly_once(self):
        covered = [m for targets in MODEL_GROUPS.values() for m in targets]
        self.assertEqual(len(covered), len(set(covered)), "a metric is in two groups")
        self.assertEqual(len(covered), 13)


class TestIsTrained(unittest.TestCase):
    """A partial set of weights must count as untrained, not trained."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.group_dir = os.path.join(self.root, "transfer_portal")
        os.makedirs(self.group_dir)
        # A scaler must exist for is_trained() to even consider the weights.
        open(os.path.join(self.root, "feature_scaler.pkl"), "wb").close()
        self._patch = patch(
            "backend.models.transfer_portal._MODELS_DIR", self.root
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def _write(self, *group_names):
        for name in group_names:
            open(os.path.join(self.group_dir, f"{name}.keras"), "wb").close()

    def test_single_group_is_not_enough(self):
        """This is the exact state that shipped broken: 1 of 6 groups present."""
        self._write("shooting")
        self.assertFalse(TransferPortalModel().is_trained())

    def test_stale_four_group_layout_is_rejected(self):
        self._write("shooting", "passing", "dribbling", "defending")
        model = TransferPortalModel()
        self.assertFalse(model.is_trained())
        self.assertEqual(
            sorted(model.missing_groups()),
            ["creation", "crossing", "distribution"],
        )

    def test_all_groups_present_is_trained(self):
        self._write(*MODEL_GROUPS)
        model = TransferPortalModel()
        self.assertTrue(model.is_trained())
        self.assertEqual(model.missing_groups(), [])

    def test_ensemble_seed_layout_counts_as_present(self):
        for name in MODEL_GROUPS:
            open(os.path.join(self.group_dir, f"{name}_seed0.keras"), "wb").close()
        self.assertTrue(TransferPortalModel().is_trained())

    def test_missing_scaler_is_untrained(self):
        self._write(*MODEL_GROUPS)
        os.remove(os.path.join(self.root, "feature_scaler.pkl"))
        self.assertFalse(TransferPortalModel().is_trained())


class TestScalerMismatchFallback(unittest.TestCase):
    """A stale scaler must degrade to heuristics, not raise ValueError."""

    def test_dimension_mismatch_falls_back_to_heuristic(self):
        model = TransferPortalModel()
        model.models = {"shooting": MagicMock()}  # bypass the load path

        stale_scaler = MagicMock()
        stale_scaler.n_features_in_ = FEATURE_DIM - 1  # simulates the 93 vs 94 bug
        model._scaler = stale_scaler

        fd = {k: 0.5 for k in _feature_keys()}
        sentinel = {"expected_goals": 0.123}
        with patch.object(model, "_heuristic_fallback", return_value=sentinel) as fb:
            result = model.predict(fd)

        fb.assert_called_once()
        self.assertEqual(result, sentinel)
        stale_scaler.transform.assert_not_called()

    def test_zero_variance_columns_are_neutralised(self):
        """Constant training features must not leak live values into the net.

        ``pre_minutes_per_match`` is constant-zero whenever the cached matrices
        predate Phase 9.  StandardScaler leaves zero-variance columns unscaled
        (scale_=1), so without this guard a live value of e.g. 85 minutes would
        be fed to a network that only ever saw 0.
        """
        model = TransferPortalModel()

        class FakeScaler:
            n_features_in_ = FEATURE_DIM
            # Final column (pre_minutes_per_match) was constant in training.
            var_ = np.array([1.0] * (FEATURE_DIM - 1) + [0.0])

            def transform(self, X):
                return X.copy()

        model._scaler = FakeScaler()

        captured: dict = {}

        def _fake_model(X_group, training=False):
            captured["X"] = np.asarray(X_group)
            out = MagicMock()
            out.numpy.return_value = np.zeros((1, 1), dtype=np.float32)
            return out

        model.models = {"minutes_probe": _fake_model}
        model._target_scalers = {}

        probe_features = ["player_expected_goals", "pre_minutes_per_match"]
        with patch.dict(
            "backend.models.transfer_portal.MODEL_GROUPS",
            {"minutes_probe": ["expected_goals"]},
            clear=True,
        ), patch.dict(
            "backend.models.transfer_portal.GROUP_FEATURE_SUBSETS",
            {"minutes_probe": probe_features},
            clear=True,
        ):
            fd = {k: 1.0 for k in _feature_keys()}
            fd["pre_minutes_per_match"] = 85.0  # a real, live, in-range value
            model.predict(fd)

        self.assertIn("X", captured)
        # Column 0 (a normal feature) passes through untouched...
        self.assertEqual(captured["X"][0, 0], 1.0)
        # ...but the zero-variance column is forced back to the training constant.
        self.assertEqual(captured["X"][0, 1], 0.0)


class TestThemeComponents(unittest.TestCase):
    """The borrowed score-ring / badge components must be well-formed HTML."""

    def test_score_ring_is_valid_svg(self):
        from frontend.theme import score_ring

        html = score_ring(12.4, label="net change", sublabel="13 metrics")
        self.assertIn("<svg", html)
        self.assertIn("</svg>", html)
        self.assertIn("stroke-dasharray", html)
        self.assertIn("+12.4%", html)
        self.assertIn("NET CHANGE", html)

    def test_score_ring_clamps_out_of_range_values(self):
        from frontend.theme import score_ring

        for value in (-9999.0, 9999.0):
            html = score_ring(value)
            self.assertIn("<svg", html)
            # dasharray must never be negative or NaN
            dash = html.split('stroke-dasharray="')[1].split(" ")[0]
            self.assertGreaterEqual(float(dash), 0.0)

    def test_score_ring_handles_degenerate_range(self):
        from frontend.theme import score_ring

        html = score_ring(5.0, vmin=1.0, vmax=1.0)  # zero span
        self.assertIn("<svg", html)

    def test_tone_for_value(self):
        from frontend.theme import tone_for_value

        self.assertEqual(tone_for_value(20.0), "positive")
        self.assertEqual(tone_for_value(-20.0), "negative")
        self.assertEqual(tone_for_value(0.0), "warning")

    def test_badge_contains_text_and_tone_colour(self):
        from frontend.theme import badge

        html = badge("HOT", "positive")
        self.assertIn("HOT", html)
        self.assertIn("<span", html)

    def test_apply_plotly_theme_sets_title_and_transparency(self):
        import plotly.graph_objects as go

        from frontend.theme import apply_plotly_theme

        fig = apply_plotly_theme(go.Figure(), title="Predicted vs Actual")
        self.assertEqual(fig.layout.title.text, "Predicted vs Actual")
        self.assertEqual(fig.layout.paper_bgcolor, "rgba(0,0,0,0)")


if __name__ == "__main__":
    unittest.main()
