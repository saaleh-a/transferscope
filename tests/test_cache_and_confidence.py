"""Tests for cache limits, pruning, and the shortlist evidence signal.

Both address the same failure: something looked fine because nothing measured
it. The cache grew to 1 GB unbounded, and the shortlist scored a 200-minute
cameo identically to a full season with no way to tell them apart.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest

from backend.data import cache
from backend.models.shortlist_scorer import data_confidence


class TestCacheLimits(unittest.TestCase):
    def setUp(self):
        cache.close()
        self._tmp = tempfile.mkdtemp(prefix="ts_cache_limit_")
        os.environ["CACHE_DIR"] = self._tmp

    def tearDown(self):
        cache.close()
        os.environ.pop("CACHE_SIZE_LIMIT_MB", None)

    def test_stats_reports_size_and_limit(self):
        cache.set(cache.make_key("ns", "a"), {"x": 1})
        stats = cache.stats()
        self.assertGreaterEqual(stats["entries"], 1)
        self.assertIn("limit_mb", stats)
        self.assertEqual(stats["eviction_policy"], "least-recently-used")

    def test_limit_is_configurable(self):
        cache.close()
        os.environ["CACHE_SIZE_LIMIT_MB"] = "128"
        cache.set(cache.make_key("ns", "a"), 1)
        self.assertEqual(cache.stats()["limit_mb"], 128)

    def test_invalid_limit_falls_back_to_default(self):
        cache.close()
        os.environ["CACHE_SIZE_LIMIT_MB"] = "not-a-number"
        cache.set(cache.make_key("ns", "a"), 1)
        self.assertEqual(cache.stats()["limit_mb"], 2048)

    def test_limit_has_a_floor(self):
        """A tiny limit would thrash; the floor prevents that."""
        cache.close()
        os.environ["CACHE_SIZE_LIMIT_MB"] = "1"
        cache.set(cache.make_key("ns", "a"), 1)
        self.assertGreaterEqual(cache.stats()["limit_mb"], 64)

    def test_namespace_breakdown_counts_and_sorts(self):
        for i in range(3):
            cache.set(cache.make_key("alpha", str(i)), i)
        cache.set(cache.make_key("beta", "0"), 0)
        breakdown = cache.namespace_breakdown()
        self.assertEqual(breakdown["alpha"], 3)
        self.assertEqual(breakdown["beta"], 1)
        self.assertEqual(list(breakdown)[0], "alpha")  # largest first


class TestPruneExpired(unittest.TestCase):
    def setUp(self):
        cache.close()
        self._tmp = tempfile.mkdtemp(prefix="ts_cache_prune_")
        os.environ["CACHE_DIR"] = self._tmp

    def tearDown(self):
        cache.close()

    def test_fresh_entries_survive(self):
        cache.set(cache.make_key("ns", "fresh"), "value")
        self.assertEqual(cache.prune_expired(), 0)
        self.assertEqual(cache.get(cache.make_key("ns", "fresh")), "value")

    def test_old_entries_are_removed(self):
        key = cache.make_key("ns", "old")
        # Write directly with an ancient timestamp.
        cache._get_cache().set(key, (time.time() - cache.MAX_TTL_SECONDS - 100, "v"))
        self.assertEqual(cache.prune_expired(), 1)
        self.assertIsNone(cache.get(key))

    def test_prune_respects_custom_age(self):
        key = cache.make_key("ns", "hour_old")
        cache._get_cache().set(key, (time.time() - 3700, "v"))
        self.assertEqual(cache.prune_expired(max_age=7200), 0)  # younger than 2h
        self.assertEqual(cache.prune_expired(max_age=3600), 1)  # older than 1h

    def test_malformed_entries_are_skipped_not_crashed(self):
        cache._get_cache().set("weird", "not-a-tuple")
        cache._get_cache().set("weird2", ("not-a-number", "v"))
        cache.prune_expired()  # must not raise

    def test_empty_cache(self):
        self.assertEqual(cache.prune_expired(), 0)


class TestDataConfidence(unittest.TestCase):
    """The shortlist must distinguish a cameo from a season."""

    FULL = {m: 1.0 for m in [
        "expected_goals", "expected_assists", "shots", "successful_dribbles",
        "successful_crosses", "touches_in_opposition_box", "successful_passes",
        "pass_completion_pct", "accurate_long_balls", "chances_created",
        "clearances", "interceptions", "possession_won_final_3rd",
    ]}

    def test_full_season_with_full_coverage_is_high(self):
        result = data_confidence(3000, self.FULL)
        self.assertEqual(result["level"], "high")
        self.assertEqual(result["label"], "Good")
        self.assertEqual(result["coverage"], 1.0)

    def test_thin_minutes_is_low(self):
        """200 minutes of per-90 is mostly noise."""
        result = data_confidence(200, self.FULL)
        self.assertEqual(result["level"], "low")
        self.assertEqual(result["label"], "Thin")
        self.assertIn("200", result["reason"])

    def test_partial_season_is_medium(self):
        result = data_confidence(1200, self.FULL)
        self.assertEqual(result["level"], "medium")
        self.assertEqual(result["label"], "Partial")

    def test_missing_metrics_downgrade_confidence(self):
        sparse = {"expected_goals": 1.0, "shots": 1.0}
        result = data_confidence(3000, sparse)
        self.assertEqual(result["level"], "low")
        self.assertLess(result["coverage"], 0.5)

    def test_unknown_minutes_is_low_not_assumed_good(self):
        """Unknown must never read as fine."""
        result = data_confidence(None, self.FULL)
        self.assertEqual(result["level"], "low")
        self.assertEqual(result["label"], "Unknown")
        self.assertIsNone(result["minutes"])

    def test_empty_per90_is_low(self):
        result = data_confidence(3000, {})
        self.assertEqual(result["level"], "low")
        self.assertEqual(result["coverage"], 0.0)

    def test_reason_is_always_populated(self):
        for minutes in (None, 100, 1200, 3000):
            result = data_confidence(minutes, self.FULL)
            self.assertTrue(result["reason"])

    def test_zero_minutes_is_low(self):
        self.assertEqual(data_confidence(0, self.FULL)["level"], "low")

    def test_confidence_is_monotonic_in_minutes(self):
        order = {"low": 0, "medium": 1, "high": 2}
        levels = [
            order[data_confidence(m, self.FULL)["level"]]
            for m in (300, 1200, 2500)
        ]
        self.assertEqual(levels, sorted(levels))


if __name__ == "__main__":
    unittest.main()
