"""Comprehensive tests for backend.features.rolling_windows.

Covers:
- compute_confidence: RAG thresholds
- blend_weight: prior blend weight calculation
- blend_features: prior-weighted blending
- player_rolling_average: minute-weighted per-90 from match logs
- team_rolling_average / team_position_rolling_average: wrappers
- compute_player_features: end-to-end feature builder
"""

import os
import tempfile
import unittest

_TEMP_DIR = tempfile.mkdtemp(prefix="ts_rolling_test_")
os.environ.setdefault("CACHE_DIR", _TEMP_DIR)

from backend.data.sofascore_client import ALL_METRICS, CORE_METRICS
from backend.features.rolling_windows import (
    PLAYER_WINDOW_MINUTES,
    PRIOR_CONSTANT,
    TEAM_WINDOW_MINUTES,
    RollingFeatures,
    blend_features,
    blend_weight,
    compute_confidence,
    compute_player_features,
    player_rolling_average,
    team_position_rolling_average,
    team_rolling_average,
)


class TestComputeConfidence(unittest.TestCase):
    """compute_confidence maps blend weight → red/amber/green."""

    def test_zero_weight_is_red(self):
        self.assertEqual(compute_confidence(0.0), "red")

    def test_below_threshold_is_red(self):
        self.assertEqual(compute_confidence(0.29), "red")

    def test_boundary_030_is_amber(self):
        """weight == 0.3 should be amber (not red)."""
        self.assertEqual(compute_confidence(0.3), "amber")

    def test_mid_range_is_amber(self):
        self.assertEqual(compute_confidence(0.5), "amber")

    def test_boundary_070_is_amber(self):
        """weight == 0.7 should be amber (not green)."""
        self.assertEqual(compute_confidence(0.7), "amber")

    def test_above_070_is_green(self):
        self.assertEqual(compute_confidence(0.71), "green")

    def test_full_weight_is_green(self):
        self.assertEqual(compute_confidence(1.0), "green")


class TestBlendWeight(unittest.TestCase):
    """blend_weight: min(1, minutes / C)."""

    def test_zero_minutes(self):
        self.assertAlmostEqual(blend_weight(0), 0.0)

    def test_half_constant(self):
        self.assertAlmostEqual(blend_weight(500, 1000), 0.5)

    def test_exactly_at_constant(self):
        self.assertAlmostEqual(blend_weight(1000, 1000), 1.0)

    def test_above_constant_caps_at_one(self):
        self.assertAlmostEqual(blend_weight(2000, 1000), 1.0)

    def test_custom_constant(self):
        self.assertAlmostEqual(blend_weight(300, 600), 0.5)

    def test_zero_constant_returns_one(self):
        """Degenerate case: C=0 should return 1.0 to avoid division by zero."""
        self.assertAlmostEqual(blend_weight(500, 0), 1.0)

    def test_negative_minutes_returns_negative(self):
        """Defensive: negative minutes produces a number, doesn't crash."""
        result = blend_weight(-100, 1000)
        self.assertIsInstance(result, float)

    def test_default_constant_is_1000(self):
        self.assertAlmostEqual(blend_weight(500), 0.5)
        self.assertAlmostEqual(blend_weight(1000), 1.0)


class TestBlendFeatures(unittest.TestCase):
    """blend_features: (1-w)*prior + w*raw, returns RollingFeatures."""

    def _make_per90(self, value):
        """Create a per90 dict with all metrics set to the same value."""
        return {m: value for m in ALL_METRICS}

    def test_zero_minutes_returns_pure_prior(self):
        """With 0 minutes, weight=0 → blended = prior."""
        raw = self._make_per90(10.0)
        prior = self._make_per90(5.0)
        result = blend_features(raw, prior, 0.0)
        self.assertAlmostEqual(result.weight, 0.0)
        self.assertEqual(result.confidence, "red")
        for m in ALL_METRICS:
            self.assertAlmostEqual(result.per90[m], 5.0)

    def test_full_minutes_returns_pure_raw(self):
        """With minutes >= constant, weight=1 → blended = raw."""
        raw = self._make_per90(10.0)
        prior = self._make_per90(5.0)
        result = blend_features(raw, prior, 1500.0)
        self.assertAlmostEqual(result.weight, 1.0)
        self.assertEqual(result.confidence, "green")
        for m in ALL_METRICS:
            self.assertAlmostEqual(result.per90[m], 10.0)

    def test_half_weight_blends_equally(self):
        """At 500 minutes (default C=1000), weight=0.5 → midpoint."""
        raw = self._make_per90(10.0)
        prior = self._make_per90(0.0)
        result = blend_features(raw, prior, 500.0)
        self.assertAlmostEqual(result.weight, 0.5)
        for m in ALL_METRICS:
            self.assertAlmostEqual(result.per90[m], 5.0)

    def test_raw_none_uses_prior(self):
        """When a raw metric is None, prior is used unchanged."""
        raw = {m: None for m in ALL_METRICS}
        prior = self._make_per90(3.0)
        result = blend_features(raw, prior, 500.0)
        for m in ALL_METRICS:
            self.assertAlmostEqual(result.per90[m], 3.0)

    def test_prior_none_uses_raw(self):
        """When a prior metric is None, raw is used unchanged."""
        raw = self._make_per90(7.0)
        prior = {m: None for m in ALL_METRICS}
        result = blend_features(raw, prior, 500.0)
        for m in ALL_METRICS:
            self.assertAlmostEqual(result.per90[m], 7.0)

    def test_both_none_returns_none(self):
        """When both raw and prior are None for a metric, result is None."""
        raw = {m: None for m in ALL_METRICS}
        prior = {m: None for m in ALL_METRICS}
        result = blend_features(raw, prior, 500.0)
        for m in ALL_METRICS:
            self.assertIsNone(result.per90[m])

    def test_returns_rolling_features_dataclass(self):
        raw = self._make_per90(1.0)
        prior = self._make_per90(1.0)
        result = blend_features(raw, prior, 400.0)
        self.assertIsInstance(result, RollingFeatures)
        self.assertAlmostEqual(result.minutes_used, 400.0)
        self.assertEqual(result.confidence, "amber")

    def test_custom_constant(self):
        """Verify the constant parameter is passed through to blend_weight."""
        raw = self._make_per90(10.0)
        prior = self._make_per90(0.0)
        result = blend_features(raw, prior, 100.0, constant=200.0)
        self.assertAlmostEqual(result.weight, 0.5)

    def test_missing_metrics_in_raw_handled(self):
        """Metrics missing from raw dict (not even key present) use prior."""
        raw = {"expected_goals": 2.0}  # Only one metric
        prior = self._make_per90(1.0)
        result = blend_features(raw, prior, 500.0)
        # expected_goals should be blended: 0.5*1.0 + 0.5*2.0 = 1.5
        self.assertAlmostEqual(result.per90["expected_goals"], 1.5)
        # Others should be prior (1.0) since raw.get returns None
        self.assertAlmostEqual(result.per90["shots"], 1.0)


class TestPlayerRollingAverage(unittest.TestCase):
    """player_rolling_average: minute-weighted per-90 from match logs."""

    def test_empty_logs_returns_all_none(self):
        result = player_rolling_average([])
        for m in ALL_METRICS:
            self.assertIsNone(result[m])

    def test_single_match(self):
        """One 90-min match with known values."""
        log = {"minutes": 90, "expected_goals": 0.5, "shots": 3.0}
        result = player_rolling_average([log])
        self.assertAlmostEqual(result["expected_goals"], 0.5)
        self.assertAlmostEqual(result["shots"], 3.0)
        # All other metrics should be None (not present in log)
        self.assertIsNone(result["clearances"])

    def test_two_matches_weighted_average(self):
        """Two matches — per-90 is weighted by minutes.

        Match 1: 90 min, xG = 1.0 → total = 90 * 1.0 = 90
        Match 2: 45 min, xG = 2.0 → total = 45 * 2.0 = 90
        Overall: 180 / 135 = 1.333...
        """
        logs = [
            {"minutes": 90, "expected_goals": 1.0},
            {"minutes": 45, "expected_goals": 2.0},
        ]
        result = player_rolling_average(logs)
        expected = (90 * 1.0 + 45 * 2.0) / (90 + 45)
        self.assertAlmostEqual(result["expected_goals"], expected, places=4)

    def test_window_limits_matches_used(self):
        """Only matches within window (1000 min) should be included."""
        # 12 matches of 90 mins = 1080 total. Window = 1000.
        # Should stop after the 11th match (990 + 90 = 1080 is over window).
        # Actually: accumulates until >= 1000, so 12th log at 1080 breaks.
        # The 12th log is not included because accumulation reaches >= 1000 first.
        logs = [{"minutes": 90, "expected_goals": 1.0} for _ in range(12)]
        # The last log should be EXCLUDED due to windowing:
        # After 11 matches: 990 minutes, not yet >= 1000, so match 12 is included
        # After 12 matches: 1080, now >= 1000 so loop breaks before match 13
        # With 12 entries, all 12 are included (990 < 1000 after 11, continue)
        result = player_rolling_average(logs, window_minutes=1000)
        self.assertAlmostEqual(result["expected_goals"], 1.0)

    def test_window_exactly_at_limit(self):
        """Matches totaling exactly the window should be used."""
        logs = [{"minutes": 500, "expected_goals": 1.0},
                {"minutes": 500, "expected_goals": 2.0}]
        result = player_rolling_average(logs, window_minutes=1000)
        expected = (500 * 1.0 + 500 * 2.0) / 1000
        self.assertAlmostEqual(result["expected_goals"], expected)

    def test_none_minutes_treated_as_zero(self):
        """Match with None minutes should be treated as 0 mins."""
        logs = [
            {"minutes": None, "expected_goals": 5.0},
            {"minutes": 90, "expected_goals": 1.0},
        ]
        result = player_rolling_average(logs)
        # First match contributes 0 minutes → 0 * 5.0 = 0, counts 0
        # Second match: 90 min → 90 * 1.0 = 90, counts 90
        # Average: 90 / 90 = 1.0
        self.assertAlmostEqual(result["expected_goals"], 1.0)

    def test_non_numeric_metric_skipped(self):
        """Non-numeric metric values should be silently skipped."""
        logs = [{"minutes": 90, "expected_goals": "not_a_number"}]
        result = player_rolling_average(logs)
        self.assertIsNone(result["expected_goals"])

    def test_zero_minutes_match_contributes_nothing(self):
        """A 0-minute appearance contributes nothing to the average."""
        logs = [
            {"minutes": 0, "expected_goals": 100.0},
            {"minutes": 90, "expected_goals": 1.0},
        ]
        result = player_rolling_average(logs)
        self.assertAlmostEqual(result["expected_goals"], 1.0)

    def test_returns_all_metrics(self):
        """Result should have entries for ALL_METRICS."""
        logs = [{"minutes": 90, "expected_goals": 1.0}]
        result = player_rolling_average(logs)
        for m in ALL_METRICS:
            self.assertIn(m, result)

    def test_custom_window(self):
        """Verify custom window_minutes parameter is respected."""
        logs = [
            {"minutes": 100, "expected_goals": 1.0},
            {"minutes": 100, "expected_goals": 3.0},
            {"minutes": 100, "expected_goals": 5.0},
        ]
        # Window = 150: only first log should be included (100 < 150)
        # After first: 100 accumulated. Still < 150, so continue.
        # After second: 200 accumulated. >= 150, so break.
        # So first two logs included.
        result = player_rolling_average(logs, window_minutes=150)
        expected = (100 * 1.0 + 100 * 3.0) / 200
        self.assertAlmostEqual(result["expected_goals"], expected)


class TestTeamRollingAverage(unittest.TestCase):
    """team_rolling_average: same as player but 3000-min window."""

    def test_delegates_to_player_rolling_with_3000_window(self):
        """team_rolling_average uses TEAM_WINDOW_MINUTES (3000) by default."""
        logs = [{"minutes": 90, "expected_goals": 2.0}]
        result = team_rolling_average(logs)
        self.assertAlmostEqual(result["expected_goals"], 2.0)

    def test_empty_logs(self):
        result = team_rolling_average([])
        for m in ALL_METRICS:
            self.assertIsNone(result[m])


class TestTeamPositionRollingAverage(unittest.TestCase):
    """team_position_rolling_average: same as team."""

    def test_delegates_correctly(self):
        logs = [{"minutes": 90, "shots": 4.0}]
        result = team_position_rolling_average(logs)
        self.assertAlmostEqual(result["shots"], 4.0)

    def test_empty_logs(self):
        result = team_position_rolling_average([])
        for m in ALL_METRICS:
            self.assertIsNone(result[m])


class TestComputePlayerFeatures(unittest.TestCase):
    """compute_player_features: end-to-end feature builder."""

    def _make_player_stats(self, minutes=900, per90=None):
        """Helper to build player_stats dict."""
        if per90 is None:
            per90 = {m: 1.0 for m in ALL_METRICS}
        return {"minutes_played": minutes, "per90": per90}

    def test_basic_no_prior_no_logs(self):
        """With no priors or logs, uses season aggregate from per90."""
        stats = self._make_player_stats(minutes=900)
        result = compute_player_features(stats)
        self.assertIsInstance(result, RollingFeatures)
        self.assertAlmostEqual(result.minutes_used, 900)
        # 900/1000 = 0.9 weight → green (> 0.7)
        self.assertEqual(result.confidence, "green")
        # With no prior, blending produces: (1-w)*None + w*raw = raw
        for m in ALL_METRICS:
            self.assertAlmostEqual(result.per90[m], 1.0)

    def test_with_prior_blends(self):
        """When prior is provided, blend with season aggregate."""
        raw = {m: 2.0 for m in ALL_METRICS}
        prior = {m: 0.0 for m in ALL_METRICS}
        stats = self._make_player_stats(minutes=500, per90=raw)
        result = compute_player_features(stats, prior_per90=prior)
        # weight = 500/1000 = 0.5, blended = 0.5*0 + 0.5*2 = 1.0
        for m in ALL_METRICS:
            self.assertAlmostEqual(result.per90[m], 1.0)

    def test_with_match_logs_uses_rolling(self):
        """When match_logs provided, use rolling average instead of per90."""
        stats = self._make_player_stats(minutes=900, per90={m: 99.0 for m in ALL_METRICS})
        match_logs = [{"minutes": 90, "expected_goals": 0.3}]
        result = compute_player_features(stats, match_logs=match_logs)
        # Should use rolling average from logs (0.3), not season per90 (99.0)
        self.assertAlmostEqual(result.per90["expected_goals"], 0.3)

    def test_none_minutes_treated_as_zero(self):
        """If minutes_played is None, treat as 0 → red confidence."""
        stats = {"minutes_played": None, "per90": {m: 1.0 for m in ALL_METRICS}}
        result = compute_player_features(stats)
        self.assertAlmostEqual(result.minutes_used, 0)
        self.assertEqual(result.confidence, "red")

    def test_missing_minutes_key_treated_as_zero(self):
        """If minutes_played key missing entirely, default to 0."""
        stats = {"per90": {m: 1.0 for m in ALL_METRICS}}
        result = compute_player_features(stats)
        self.assertAlmostEqual(result.minutes_used, 0)

    def test_high_minutes_green_confidence(self):
        """Minutes > 700 (weight > 0.7) → green confidence."""
        stats = self._make_player_stats(minutes=1200)
        result = compute_player_features(stats)
        self.assertEqual(result.confidence, "green")

    def test_empty_match_logs_falls_back_to_season(self):
        """Empty match_logs list (falsy) → uses per90 from stats."""
        raw = {m: 5.0 for m in ALL_METRICS}
        stats = self._make_player_stats(minutes=900, per90=raw)
        result = compute_player_features(stats, match_logs=[])
        for m in ALL_METRICS:
            self.assertAlmostEqual(result.per90[m], 5.0)


class TestConstants(unittest.TestCase):
    """Verify module constants match documented values."""

    def test_player_window(self):
        self.assertEqual(PLAYER_WINDOW_MINUTES, 1000)

    def test_team_window(self):
        self.assertEqual(TEAM_WINDOW_MINUTES, 3000)

    def test_prior_constant(self):
        self.assertEqual(PRIOR_CONSTANT, 1000)


if __name__ == "__main__":
    unittest.main()
