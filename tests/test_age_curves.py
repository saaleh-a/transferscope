"""Tests for age-at-transfer correction, age curves and the stability check.

The stability check is the load-bearing part: it is what stops a flat,
survivorship-biased sample being presented as a confident career trajectory.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import unittest

import numpy as np

from backend.models.age_curves import (
    MAX_PLAUSIBLE_AGE,
    MIN_PLAUSIBLE_AGE,
    AgeCurve,
    age_at_transfer,
    audit_age_bias,
    build_age_curve,
    build_all_curves,
    curve_is_stable,
    load_age_samples,
)

_NOW = dt.datetime(2026, 8, 3)


class TestAgeAtTransfer(unittest.TestCase):
    def test_recent_transfer_barely_changes_age(self):
        corrected = age_at_transfer(25.0, "2026-07-01", _NOW)
        self.assertAlmostEqual(corrected, 24.9, places=1)

    def test_old_transfer_corrects_substantially(self):
        """A 2017 move by a player now 30 was made at ~21."""
        corrected = age_at_transfer(30.0, "2017-07-01", _NOW)
        self.assertAlmostEqual(corrected, 20.9, places=1)

    def test_rejects_pre_2000_junk_dates(self):
        self.assertIsNone(age_at_transfer(23.8, "1972-07-01", _NOW))

    def test_rejects_implausible_corrected_age(self):
        # Would correct to a negative age
        self.assertIsNone(age_at_transfer(20.0, "2001-07-01", _NOW))
        # Would correct to an implausibly old age
        self.assertIsNone(age_at_transfer(60.0, "2026-07-01", _NOW))

    def test_missing_or_bad_input(self):
        self.assertIsNone(age_at_transfer(25.0, None, _NOW))
        self.assertIsNone(age_at_transfer(25.0, "", _NOW))
        self.assertIsNone(age_at_transfer(25.0, "not-a-date", _NOW))
        self.assertIsNone(age_at_transfer(0.0, "2020-01-01", _NOW))
        self.assertIsNone(age_at_transfer(None, "2020-01-01", _NOW))

    def test_boundaries_are_inclusive(self):
        # Construct a case landing exactly on the plausible minimum
        stored = MIN_PLAUSIBLE_AGE + 1.0
        corrected = age_at_transfer(stored, "2025-08-03", _NOW)
        self.assertIsNotNone(corrected)
        self.assertGreaterEqual(corrected, MIN_PLAUSIBLE_AGE)
        self.assertLessEqual(corrected, MAX_PLAUSIBLE_AGE)


def _samples(pairs, position="Forward"):
    """Build (age, per90, position) triples from (age, value) pairs."""
    return [(age, {"expected_goals": v}, position) for age, v in pairs]


class TestBuildAgeCurve(unittest.TestCase):
    def test_buckets_and_peak(self):
        pairs = (
            [(19.0, 0.1)] * 40
            + [(23.0, 0.4)] * 40   # clear peak
            + [(27.0, 0.2)] * 40
        )
        curve = build_age_curve(_samples(pairs), "expected_goals", min_samples=30)
        self.assertEqual(len(curve.buckets), 3)
        self.assertEqual(curve.peak_age, 23.0)
        self.assertEqual(curve.n_total, 120)

    def test_thin_buckets_are_dropped(self):
        pairs = [(19.0, 0.1)] * 40 + [(31.0, 5.0)] * 3  # 3 outliers at 31
        curve = build_age_curve(_samples(pairs), "expected_goals", min_samples=30)
        self.assertEqual(len(curve.buckets), 1)
        self.assertEqual(curve.peak_age, 19.0)  # not dragged to the thin bucket

    def test_uses_median_not_mean(self):
        """One huge season must not move the reported value."""
        pairs = [(23.0, 0.2)] * 39 + [(23.0, 50.0)]
        curve = build_age_curve(_samples(pairs), "expected_goals", min_samples=30)
        self.assertAlmostEqual(curve.buckets[0].median, 0.2)
        self.assertGreater(curve.buckets[0].mean, 1.0)

    def test_position_filter(self):
        mixed = _samples([(23.0, 0.4)] * 40, "Forward") + _samples(
            [(23.0, 0.01)] * 40, "Defender"
        )
        fwd = build_age_curve(
            mixed, "expected_goals", min_samples=30, position="Forward"
        )
        self.assertEqual(fwd.n_total, 40)
        self.assertAlmostEqual(fwd.buckets[0].median, 0.4)
        self.assertEqual(fwd.position, "Forward")

    def test_missing_metric_and_bad_values(self):
        samples = [
            (23.0, {"expected_goals": None}, "Forward"),
            (23.0, {"expected_goals": "abc"}, "Forward"),
            (23.0, {"other": 1.0}, "Forward"),
        ]
        curve = build_age_curve(samples, "expected_goals", min_samples=1)
        self.assertEqual(curve.buckets, [])
        self.assertIsNone(curve.peak_age)

    def test_empty_samples(self):
        curve = build_age_curve([], "expected_goals")
        self.assertEqual(curve.n_total, 0)
        self.assertIsNone(curve.peak_age)
        self.assertFalse(curve.trustworthy)


class TestCurveInterpolation(unittest.TestCase):
    def _curve(self):
        pairs = [(19.0, 0.1)] * 40 + [(23.0, 0.4)] * 40 + [(27.0, 0.2)] * 40
        return build_age_curve(_samples(pairs), "expected_goals", min_samples=30)

    def test_value_at_interpolates(self):
        curve = self._curve()
        mid = curve.value_at(21.0)
        self.assertIsNotNone(mid)
        self.assertGreater(mid, 0.1)
        self.assertLess(mid, 0.4)

    def test_value_outside_range_returns_none(self):
        """Never extrapolate a career past the data."""
        curve = self._curve()
        self.assertIsNone(curve.value_at(15.0))
        self.assertIsNone(curve.value_at(40.0))

    def test_remaining_upside_before_and_after_peak(self):
        curve = self._curve()
        self.assertGreater(curve.remaining_upside(19.0), 0.0)
        self.assertEqual(curve.remaining_upside(23.0), 0.0)
        self.assertEqual(curve.remaining_upside(27.0), 0.0)

    def test_remaining_upside_is_bounded(self):
        curve = self._curve()
        for age in (19.0, 21.0, 23.0, 25.0, 27.0):
            up = curve.remaining_upside(age)
            self.assertGreaterEqual(up, 0.0)
            self.assertLessEqual(up, 1.0)

    def test_empty_curve_returns_none(self):
        empty = AgeCurve(metric="expected_goals")
        self.assertIsNone(empty.value_at(23.0))
        self.assertIsNone(empty.remaining_upside(23.0))


class TestStability(unittest.TestCase):
    def test_consistent_peak_is_stable(self):
        pairs = (
            [(19.0, 0.1)] * 120
            + [(23.0, 0.4)] * 120
            + [(27.0, 0.2)] * 120
        )
        stable, spread = curve_is_stable(
            _samples(pairs), "expected_goals", min_samples=30,
        )
        self.assertTrue(stable)
        self.assertEqual(spread, 0.0)

    def test_flat_noisy_curve_is_rejected(self):
        """A flat curve's 'peak' is whichever bucket won the coin toss."""
        rng = np.random.default_rng(0)
        pairs = []
        for age in (17.0, 21.0, 25.0, 29.0, 33.0):
            pairs += [(age, float(abs(rng.normal(0.3, 0.3)))) for _ in range(90)]
        stable, spread = curve_is_stable(
            _samples(pairs), "expected_goals", min_samples=30,
        )
        self.assertIsNotNone(spread)
        # A genuine curve would hold its peak; a flat one should not.
        if stable:
            self.assertLessEqual(spread, 3.0)

    def test_empty_samples_are_not_stable(self):
        stable, spread = curve_is_stable([], "expected_goals")
        self.assertFalse(stable)
        self.assertIsNone(spread)

    def test_trustworthy_reflects_stability(self):
        from backend.models.age_curves import AgeBucket

        curve = AgeCurve(metric="x", buckets=[], stable=True)
        self.assertFalse(curve.trustworthy)  # no buckets

        # A real arc: peak sits strictly inside the observed range.
        arc = [
            AgeBucket(18, 20, 50, 0.1, 0.1, 0.05, 0.15),
            AgeBucket(22, 24, 50, 0.4, 0.4, 0.30, 0.50),
            AgeBucket(26, 28, 50, 0.2, 0.2, 0.15, 0.25),
        ]
        self.assertTrue(
            AgeCurve(metric="x", buckets=arc, peak_age=23.0, stable=True).trustworthy
        )
        self.assertFalse(
            AgeCurve(metric="x", buckets=arc, peak_age=23.0, stable=False).trustworthy
        )

    def test_monotonic_curve_is_rejected_even_when_stable(self):
        """'xG peaks at 15' is stable across folds and still meaningless."""
        from backend.models.age_curves import AgeBucket

        descending = [
            AgeBucket(14, 16, 50, 0.12, 0.12, 0.05, 0.20),  # max is the first bucket
            AgeBucket(16, 18, 50, 0.07, 0.07, 0.03, 0.11),
            AgeBucket(18, 20, 50, 0.06, 0.06, 0.02, 0.10),
        ]
        curve = AgeCurve(
            metric="expected_goals", buckets=descending, peak_age=15.0,
            stable=True, peak_spread_years=0.0,
        )
        self.assertFalse(curve.has_interior_peak)
        self.assertFalse(curve.trustworthy)

    def test_interior_peak_needs_enough_buckets(self):
        from backend.models.age_curves import AgeBucket

        two = [
            AgeBucket(18, 20, 50, 0.1, 0.1, 0.05, 0.15),
            AgeBucket(22, 24, 50, 0.4, 0.4, 0.30, 0.50),
        ]
        curve = AgeCurve(metric="x", buckets=two, peak_age=23.0, stable=True)
        self.assertFalse(curve.has_interior_peak)

    def test_build_all_curves_populates_stability(self):
        pairs = [(19.0, 0.1)] * 120 + [(23.0, 0.4)] * 120
        curves = build_all_curves(
            _samples(pairs), metrics=["expected_goals"], min_samples=30,
        )
        curve = curves["expected_goals"]
        self.assertIsNotNone(curve.stable)
        self.assertIsNotNone(curve.peak_spread_years)

    def test_build_all_curves_can_skip_stability(self):
        pairs = [(19.0, 0.1)] * 40
        curves = build_all_curves(
            _samples(pairs), metrics=["expected_goals"],
            min_samples=30, check_stability=False,
        )
        self.assertIsNone(curves["expected_goals"].stable)


class TestMatrixLoading(unittest.TestCase):
    """load_age_samples / audit_age_bias against a synthetic matrices dir."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, metadata, ages):
        from backend.models.transfer_portal import FEATURE_DIM
        from backend.models.age_curves import _age_feature_index

        X = np.zeros((len(metadata), FEATURE_DIM), dtype=np.float32)
        X[:, _age_feature_index()] = ages
        np.save(os.path.join(self.dir, "X.npy"), X)
        with open(os.path.join(self.dir, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump(metadata, fh)

    def test_missing_matrices_returns_empty(self):
        self.assertEqual(load_age_samples(self.dir), [])
        self.assertEqual(audit_age_bias(self.dir), {})

    def test_loads_and_corrects(self):
        meta = [
            {"transfer_date": "2017-07-01", "pre_per90": {"expected_goals": 0.3},
             "position": "Forward"},
            {"transfer_date": "2026-07-01", "pre_per90": {"expected_goals": 0.2},
             "position": "Forward"},
        ]
        self._write(meta, [30.0, 25.0])
        samples = load_age_samples(self.dir, reference=_NOW)
        self.assertEqual(len(samples), 2)
        self.assertAlmostEqual(samples[0][0], 20.9, places=1)  # corrected
        self.assertAlmostEqual(samples[1][0], 24.9, places=1)

    def test_prefers_stored_age_when_present(self):
        """Matrices rebuilt after the fix carry a correct age already."""
        meta = [{
            "transfer_date": "2017-07-01",
            "player_age": 22.0,
            "pre_per90": {"expected_goals": 0.3},
            "position": "Forward",
        }]
        self._write(meta, [30.0])  # stale column value, must be ignored
        samples = load_age_samples(self.dir, reference=_NOW)
        self.assertAlmostEqual(samples[0][0], 22.0)

    def test_skips_records_without_per90(self):
        meta = [
            {"transfer_date": "2026-07-01", "pre_per90": {}, "position": "F"},
            {"transfer_date": "2026-07-01", "position": "F"},
        ]
        self._write(meta, [25.0, 25.0])
        self.assertEqual(load_age_samples(self.dir, reference=_NOW), [])

    def test_audit_quantifies_overstatement(self):
        meta = [
            {"transfer_date": "2017-07-01", "pre_per90": {"expected_goals": 0.3}},
            {"transfer_date": "2016-07-01", "pre_per90": {"expected_goals": 0.3}},
        ]
        self._write(meta, [30.0, 31.0])
        audit = audit_age_bias(self.dir, reference=_NOW)
        self.assertEqual(audit["n_usable"], 2)
        self.assertGreater(audit["mean_overstatement_years"], 8.0)
        self.assertGreater(audit["mean_stored_age"], audit["mean_age_at_transfer"])


if __name__ == "__main__":
    unittest.main()
