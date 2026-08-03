"""Tests for the regression-to-the-mean baseline and significance testing.

Persistence (post = pre) is a weak opponent for noisy per-90 metrics: a player
with an unusually high season regresses downward whether or not he transfers.
These tests cover the stronger baseline that distinguishes transfer signal from
that inevitability, and the bootstrap that stops a fractional improvement being
reported as skill.
"""

from __future__ import annotations

import unittest

import numpy as np

from backend.data.sofascore_client import CORE_METRICS
from backend.models.backtester import (
    _paired_bootstrap_ci,
    _target_league_mean,
    fit_mean_reversion,
)


class TestTargetLeagueMean(unittest.TestCase):
    def test_reads_target_league_mean(self):
        record = {"league_means": {"expected_goals": 0.15}}
        self.assertEqual(_target_league_mean(record, "expected_goals"), 0.15)

    def test_missing_returns_none_not_zero(self):
        """A missing mean must not silently become 0, which would invent a
        target of zero output and make the baseline look terrible."""
        self.assertIsNone(_target_league_mean({}, "expected_goals"))
        self.assertIsNone(_target_league_mean({"league_means": {}}, "expected_goals"))
        self.assertIsNone(
            _target_league_mean({"league_means": {"expected_goals": None}}, "expected_goals")
        )

    def test_bad_value_returns_none(self):
        record = {"league_means": {"expected_goals": "abc"}}
        self.assertIsNone(_target_league_mean(record, "expected_goals"))


def _make_dataset(n, shrink, noise=0.0, seed=0):
    """Build data where post = pre + shrink*(mean - pre) + noise."""
    rng = np.random.default_rng(seed)
    n_metrics = len(CORE_METRICS)
    X = np.zeros((n, 100), dtype=np.float32)
    y = np.zeros((n, n_metrics), dtype=np.float32)
    meta = []
    league_mean = 1.0

    for i in range(n):
        pre = float(rng.uniform(0.0, 2.0))
        for j in range(n_metrics):
            X[i, j] = pre
            post = pre + shrink * (league_mean - pre)
            if noise:
                post += rng.normal(0, noise)
            y[i, j] = post
        meta.append({
            "league_means": {m: league_mean for m in CORE_METRICS},
            "player_id": i,
        })
    return X, y, meta


class TestFitMeanReversion(unittest.TestCase):
    def test_recovers_known_shrinkage(self):
        X, y, meta = _make_dataset(400, shrink=0.30)
        lambdas = fit_mean_reversion(X, y, meta)
        for m in CORE_METRICS:
            self.assertAlmostEqual(lambdas[m], 0.30, delta=0.03)

    def test_recovers_zero_shrinkage(self):
        """Perfectly persistent data must fit lambda ~ 0."""
        X, y, meta = _make_dataset(400, shrink=0.0)
        lambdas = fit_mean_reversion(X, y, meta)
        for m in CORE_METRICS:
            self.assertLess(lambdas[m], 0.05)

    def test_recovers_full_shrinkage(self):
        """Data that fully reverts must fit lambda ~ 1."""
        X, y, meta = _make_dataset(400, shrink=1.0)
        lambdas = fit_mean_reversion(X, y, meta)
        for m in CORE_METRICS:
            self.assertGreater(lambdas[m], 0.95)

    def test_survives_noise(self):
        X, y, meta = _make_dataset(800, shrink=0.40, noise=0.15)
        lambdas = fit_mean_reversion(X, y, meta)
        self.assertAlmostEqual(lambdas["expected_goals"], 0.40, delta=0.10)

    def test_lambda_is_bounded(self):
        X, y, meta = _make_dataset(200, shrink=0.5, noise=1.0, seed=3)
        for lam in fit_mean_reversion(X, y, meta).values():
            self.assertGreaterEqual(lam, 0.0)
            self.assertLessEqual(lam, 1.0)

    def test_falls_back_to_persistence_without_enough_data(self):
        """Too few samples must yield lambda=0, not an overfitted value."""
        X, y, meta = _make_dataset(10, shrink=0.5)
        lambdas = fit_mean_reversion(X, y, meta, min_samples=50)
        for m in CORE_METRICS:
            self.assertEqual(lambdas[m], 0.0)

    def test_records_without_league_means_are_skipped(self):
        X, y, meta = _make_dataset(100, shrink=0.5)
        for record in meta:
            record.pop("league_means")
        lambdas = fit_mean_reversion(X, y, meta)
        for m in CORE_METRICS:
            self.assertEqual(lambdas[m], 0.0)

    def test_empty_input(self):
        lambdas = fit_mean_reversion(np.zeros((0, 100)), np.zeros((0, 13)), [])
        self.assertEqual(set(lambdas), set(CORE_METRICS))
        self.assertTrue(all(v == 0.0 for v in lambdas.values()))


class TestPairedBootstrap(unittest.TestCase):
    def test_clear_improvement_is_significant(self):
        baseline = [1.0] * 300
        model = [0.5] * 300
        low, high, sig = _paired_bootstrap_ci(baseline, model)
        self.assertTrue(sig)
        self.assertGreater(low, 0)

    def test_no_difference_is_not_significant(self):
        rng = np.random.default_rng(1)
        errors = list(rng.normal(1.0, 0.3, 300))
        low, high, sig = _paired_bootstrap_ci(errors, list(errors))
        self.assertFalse(sig)
        self.assertAlmostEqual(low, 0.0, places=6)

    def test_tiny_improvement_in_noise_is_rejected(self):
        """The case that matters: a small delta must not read as skill."""
        rng = np.random.default_rng(2)
        baseline = rng.normal(1.0, 1.0, 200)
        model = baseline - rng.normal(0.001, 1.0, 200)  # negligible, very noisy
        low, high, sig = _paired_bootstrap_ci(list(baseline), list(model))
        self.assertFalse(sig)
        self.assertLess(low, 0)

    def test_model_worse_is_not_significant(self):
        baseline = [0.5] * 200
        model = [1.0] * 200
        low, high, sig = _paired_bootstrap_ci(baseline, model)
        self.assertFalse(sig)
        self.assertLess(high, 0)

    def test_ci_brackets_the_point_estimate(self):
        rng = np.random.default_rng(4)
        baseline = np.array(rng.normal(1.0, 0.2, 400))
        # Vary the improvement so the difference has real spread — a constant
        # gap would collapse the interval to a single point.
        model = baseline - rng.normal(0.1, 0.05, 400)
        point = float(np.mean(baseline - model))
        low, high, _ = _paired_bootstrap_ci(list(baseline), list(model))
        self.assertLess(low, point)
        self.assertGreater(high, point)

    def test_constant_difference_collapses_interval(self):
        """An exactly constant gap has no sampling variance to report."""
        baseline = [1.0] * 200
        model = [0.9] * 200
        low, high, sig = _paired_bootstrap_ci(baseline, model)
        self.assertAlmostEqual(low, 0.1, places=9)
        self.assertAlmostEqual(high, 0.1, places=9)
        self.assertTrue(sig)

    def test_is_deterministic(self):
        rng = np.random.default_rng(5)
        baseline = list(rng.normal(1.0, 0.3, 200))
        model = list(np.array(baseline) - 0.05)
        self.assertEqual(
            _paired_bootstrap_ci(baseline, model),
            _paired_bootstrap_ci(baseline, model),
        )

    def test_degenerate_inputs(self):
        self.assertEqual(_paired_bootstrap_ci([], []), (None, None, False))
        self.assertEqual(_paired_bootstrap_ci([1.0], [0.5]), (None, None, False))
        # Mismatched lengths must not silently compare misaligned pairs
        self.assertEqual(_paired_bootstrap_ci([1.0, 2.0], [0.5]), (None, None, False))


class TestBacktestReportShape(unittest.TestCase):
    """run_backtest must report against the strong baseline when given train data.

    Both tests redirect ``_MODELS_DIR`` to a temp directory: run_backtest writes
    backtest_report.json as a side effect, and a test must never overwrite the
    real trained artefacts.
    """

    @staticmethod
    def _isolated_run(*args, **kwargs):
        """Call run_backtest with the models dir pointed at a temp folder."""
        import shutil
        import tempfile
        from unittest.mock import patch

        from backend.models import backtester as bt

        tmpdir = tempfile.mkdtemp()
        original = bt._MODELS_DIR
        bt._MODELS_DIR = tmpdir
        try:
            with patch.object(bt.power_rankings, "get_team_ranking", return_value=None):
                return bt.run_backtest(*args, **kwargs)
        finally:
            bt._MODELS_DIR = original
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_report_includes_mean_reversion_fields(self):
        X_tr, y_tr, m_tr = _make_dataset(200, shrink=0.3, seed=7)
        X_te, y_te, m_te = _make_dataset(60, shrink=0.3, seed=8)
        for i, record in enumerate(m_te):
            record["player_id"] = 10_000 + i  # avoid the leakage guard

        report = self._isolated_run(
            X_te, y_te, m_te, meta_train=m_tr, X_train=X_tr, y_train=y_tr,
        )

        overall = report["overall"]
        self.assertIn("metrics_beating_mean_reversion", overall)
        self.assertIn("metrics_beating_mean_reversion_significantly", overall)
        self.assertIn("inconclusive_metrics", overall)

        per_metric = report["per_metric"]["expected_goals"]
        for field in (
            "mean_reversion_lambda",
            "mean_reversion_mae",
            "improvement_vs_mean_reversion",
            "beats_mean_reversion",
        ):
            self.assertIn(field, per_metric)

    def test_report_omits_mean_reversion_without_training_data(self):
        X_te, y_te, m_te = _make_dataset(40, shrink=0.3, seed=9)
        report = self._isolated_run(X_te, y_te, m_te)

        self.assertNotIn("metrics_beating_mean_reversion", report["overall"])
        self.assertNotIn(
            "improvement_vs_mean_reversion", report["per_metric"]["expected_goals"]
        )

    def test_does_not_write_to_the_real_models_dir(self):
        """Guard against the pollution this class exists to avoid."""
        import os

        from backend.models import backtester as bt

        real_report = os.path.join(bt._MODELS_DIR, "backtest_report.json")
        before = os.path.getmtime(real_report) if os.path.exists(real_report) else None

        X_te, y_te, m_te = _make_dataset(30, shrink=0.3, seed=11)
        self._isolated_run(X_te, y_te, m_te)

        after = os.path.getmtime(real_report) if os.path.exists(real_report) else None
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
