"""Tests for shortlist prediction routing.

The shortlist called paper_heuristic_predict directly and never touched the
trained network, so its clustering and similarity were both built on the weaker
predictor. Measured on the 1,344-transfer temporal test split, the heuristic has
higher MAE than the network on all 13 metrics (mean +32.5%) and is also worse
than assuming no change at all.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from backend.data.sofascore_client import CORE_METRICS
from frontend.pages import shortlist_generator as sg


class TestPredictionRouting(unittest.TestCase):
    @staticmethod
    def _args(model):
        return {
            "model": model,
            "lp_current": {m: 0.5 for m in CORE_METRICS},
            "source_pos_avg": {m: 0.4 for m in CORE_METRICS},
            "lp_norm": 70.0,
            "lp_league": 60.0,
            "source_norm": 90.0,
            "source_league": 85.0,
            "position": "Forward",
        }

    def test_uses_the_network_when_available(self):
        model = MagicMock()
        model.predict.return_value = {m: 0.9 for m in CORE_METRICS}

        with patch.object(sg, "paper_heuristic_predict") as heuristic:
            result = sg._predict_at_source_team(**self._args(model))

        model.predict.assert_called_once()
        heuristic.assert_not_called()
        self.assertEqual(result["expected_goals"], 0.9)

    def test_falls_back_to_heuristic_without_a_model(self):
        with patch.object(
            sg, "paper_heuristic_predict",
            return_value={m: 0.3 for m in CORE_METRICS},
        ) as heuristic:
            result = sg._predict_at_source_team(**self._args(None))

        heuristic.assert_called_once()
        self.assertEqual(result["expected_goals"], 0.3)

    def test_falls_back_when_the_network_raises(self):
        model = MagicMock()
        model.predict.side_effect = RuntimeError("scaler mismatch")

        with patch.object(
            sg, "paper_heuristic_predict",
            return_value={m: 0.3 for m in CORE_METRICS},
        ) as heuristic:
            result = sg._predict_at_source_team(**self._args(model))

        heuristic.assert_called_once()
        self.assertEqual(result["expected_goals"], 0.3)

    def test_falls_back_when_the_network_returns_nothing(self):
        model = MagicMock()
        model.predict.return_value = {}

        with patch.object(
            sg, "paper_heuristic_predict",
            return_value={m: 0.3 for m in CORE_METRICS},
        ) as heuristic:
            sg._predict_at_source_team(**self._args(model))

        heuristic.assert_called_once()

    def test_builds_a_full_feature_vector(self):
        """The network needs all 94 features, not the heuristic's arguments."""
        from backend.models.transfer_portal import FEATURE_DIM, _feature_keys

        captured = {}

        def capture(fd):
            captured["fd"] = fd
            return {m: 0.5 for m in CORE_METRICS}

        model = MagicMock()
        model.predict.side_effect = capture
        sg._predict_at_source_team(**self._args(model))

        self.assertIn("fd", captured)
        keys = set(_feature_keys())
        missing = keys - set(captured["fd"])
        self.assertEqual(missing, set(), f"feature dict missing {len(missing)} keys")
        self.assertEqual(len(keys), FEATURE_DIM)


class TestLoadModel(unittest.TestCase):
    def test_returns_none_when_untrained(self):
        sg._load_model.clear()
        with patch.object(sg, "TransferPortalModel") as cls:
            cls.return_value.is_trained.return_value = False
            self.assertIsNone(sg._load_model())
        sg._load_model.clear()

    def test_returns_none_on_load_failure(self):
        """A broken model must degrade to the heuristic, not crash the page."""
        sg._load_model.clear()
        with patch.object(sg, "TransferPortalModel") as cls:
            cls.return_value.is_trained.return_value = True
            cls.return_value.load.side_effect = OSError("corrupt weights")
            self.assertIsNone(sg._load_model())
        sg._load_model.clear()


if __name__ == "__main__":
    unittest.main()
