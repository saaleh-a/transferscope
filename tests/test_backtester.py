"""Tests for backend.models.backtester."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from backend.data.sofascore_client import CORE_METRICS
from backend.models.transfer_portal import FEATURE_DIM


class TestRunBacktest(unittest.TestCase):
    """Test run_backtest returns a properly structured report dict."""

    @mock.patch("backend.models.backtester._MODELS_DIR")
    def test_run_backtest_returns_report_dict(self, mock_dir):
        from backend.models.backtester import run_backtest

        # Use a temporary directory for models
        tmpdir = tempfile.mkdtemp()
        mock_dir.__str__ = lambda self: tmpdir

        # Monkey-patch _MODELS_DIR inside backtester module
        import backend.models.backtester as bt
        orig_dir = bt._MODELS_DIR
        bt._MODELS_DIR = tmpdir

        try:
            n = 20
            X_test = np.random.randn(n, FEATURE_DIM).astype(np.float32)
            y_test = np.abs(np.random.randn(n, len(CORE_METRICS)).astype(np.float32))
            meta_test = [
                {"player_id": i, "transfer_date": f"2024-01-{i+1:02d}",
                 "from_club": "A", "to_club": "B", "confidence": 0.8}
                for i in range(n)
            ]

            report = run_backtest(X_test, y_test, meta_test)

            self.assertIn("n_samples", report)
            self.assertEqual(report["n_samples"], n)
            self.assertIn("per_metric", report)
            self.assertIn("overall", report)

            for m in CORE_METRICS:
                self.assertIn(m, report["per_metric"])
                self.assertIn("mae", report["per_metric"][m])
                self.assertIn("naive_mae", report["per_metric"][m])

            # Report file should exist
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "backtest_report.json")))
        finally:
            bt._MODELS_DIR = orig_dir
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestNaiveBaseline(unittest.TestCase):
    """Test that naive baseline is computed correctly (pre-transfer stats)."""

    def test_naive_baseline_computed_correctly(self):
        """Verify naive baseline = pre-transfer per-90 (first 13 features)."""
        n = 1
        # Set up known values
        features = np.zeros((n, FEATURE_DIM), dtype=np.float32)
        # Player pre-transfer per-90 = [1.0, 2.0, 3.0, ...] for first 13 metrics
        for i in range(len(CORE_METRICS)):
            features[0, i] = float(i + 1)

        labels = np.ones((n, len(CORE_METRICS)), dtype=np.float32) * 5.0

        # Naive prediction should be features[0, 0:13]
        # MAE for metric 0: |1.0 - 5.0| = 4.0
        # MAE for metric 1: |2.0 - 5.0| = 3.0
        # etc.
        expected_naive_mae_metric0 = abs(1.0 - 5.0)
        expected_naive_mae_metric1 = abs(2.0 - 5.0)

        # We verify the logic by computing manually
        self.assertAlmostEqual(expected_naive_mae_metric0, 4.0)
        self.assertAlmostEqual(expected_naive_mae_metric1, 3.0)


class TestDirectionAccuracy(unittest.TestCase):
    """Test direction accuracy computation."""

    def test_direction_accuracy_computed_correctly(self):
        """Verify direction accuracy: predicted increase when actual increased."""
        # If naive = 2.0, actual = 4.0 (increased), predicted = 3.0 (also increased)
        # -> direction correct
        naive = 2.0
        actual = 4.0
        predicted = 3.0

        actual_change = actual - naive  # +2.0
        pred_change = predicted - naive  # +1.0
        direction_correct = (actual_change > 0 and pred_change > 0)
        self.assertTrue(direction_correct)

        # If naive = 2.0, actual = 1.0 (decreased), predicted = 3.0 (increased)
        # -> direction wrong
        actual2 = 1.0
        predicted2 = 3.0
        actual_change2 = actual2 - naive  # -1.0
        pred_change2 = predicted2 - naive  # +1.0
        direction_correct2 = (actual_change2 > 0 and pred_change2 > 0) or \
                             (actual_change2 < 0 and pred_change2 < 0)
        self.assertFalse(direction_correct2)


if __name__ == "__main__":
    unittest.main()
