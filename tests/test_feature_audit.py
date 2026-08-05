"""Tests for feature-degeneracy detection.

A constant column trains without complaint and contributes nothing. Four were
found by hand in the shipped matrix, three of them because Sofascore does not
serve those stats at all. These tests turn that manual sweep into a guard, so
the next one is caught before it trains into a shipped model.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import tempfile
import unittest

import numpy as np

from backend.models.feature_audit import (
    KNOWN_DEAD_FEATURES,
    SPARSE_THRESHOLD,
    STALE_MATRIX_PENDING,
    audit_features,
    audit_saved_matrices,
    format_report,
    matrix_predates_current_pipeline,
    unexpected_dead_features,
)


class TestAuditFeatures(unittest.TestCase):
    def test_detects_constant_column(self):
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        audits = audit_features(X, ["varies", "constant"])
        self.assertFalse(audits[0].is_constant)
        self.assertTrue(audits[1].is_constant)
        self.assertEqual(audits[1].constant_value, 5.0)
        self.assertFalse(audits[1].is_healthy)

    def test_constant_zero_is_the_classic_case(self):
        """pre_minutes_per_match was all zeros after a migration."""
        X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        audits = audit_features(X, ["ok", "dead"])
        self.assertTrue(audits[1].is_constant)
        self.assertEqual(audits[1].constant_value, 0.0)

    def test_detects_sparse_but_not_constant(self):
        col = np.zeros(100)
        col[0] = 1.0  # 99% zeros, still varies
        X = np.column_stack([np.arange(100), col])
        audits = audit_features(X, ["dense", "sparse"])
        self.assertFalse(audits[1].is_constant)
        self.assertTrue(audits[1].is_sparse)
        self.assertGreater(audits[1].zero_fraction, SPARSE_THRESHOLD)

    def test_sparse_and_constant_are_mutually_exclusive(self):
        X = np.zeros((50, 1))
        audits = audit_features(X, ["allzero"])
        self.assertTrue(audits[0].is_constant)
        self.assertFalse(audits[0].is_sparse)

    def test_detects_non_finite(self):
        X = np.array([[1.0], [np.nan], [3.0]])
        audits = audit_features(X, ["nan_col"])
        self.assertTrue(audits[0].has_non_finite)
        self.assertFalse(audits[0].is_healthy)

    def test_detects_inf(self):
        X = np.array([[1.0], [np.inf], [3.0]])
        self.assertTrue(audit_features(X, ["inf_col"])[0].has_non_finite)

    def test_handles_more_names_than_columns(self):
        X = np.ones((5, 2))
        audits = audit_features(X, ["a", "b", "c", "d"])
        self.assertEqual(len(audits), 2)

    def test_empty_matrix(self):
        self.assertEqual(audit_features(np.zeros((0, 0)), []), [])


class TestUnexpectedDeadFeatures(unittest.TestCase):
    def test_known_gaps_are_not_flagged(self):
        """Documented unavailability is not a regression."""
        X = np.zeros((10, 1))
        name = sorted(KNOWN_DEAD_FEATURES)[0]
        audits = audit_features(X, [name])
        self.assertEqual(unexpected_dead_features(audits), [])

    def test_new_dead_feature_is_flagged(self):
        """This is the regression signal the guard exists for."""
        X = np.zeros((10, 1))
        audits = audit_features(X, ["player_expected_goals"])
        self.assertEqual(
            unexpected_dead_features(audits), ["player_expected_goals"]
        )

    def test_healthy_features_are_not_flagged(self):
        X = np.random.default_rng(0).normal(size=(50, 3))
        audits = audit_features(X, ["a", "b", "c"])
        self.assertEqual(unexpected_dead_features(audits), [])

    def test_migration_gap_is_no_longer_whitelisted(self):
        """``pre_minutes_per_match`` must not be exempt any more.

        It was in ``KNOWN_MIGRATION_GAPS``, which meant the one guard built to
        catch "a migration silently zero-filled a column" was configured to
        stay quiet about the column it had happened to. Inference now computes
        the value, so a constant column is a genuine regression.
        """
        X = np.zeros((10, 1))
        audits = audit_features(X, ["pre_minutes_per_match"])
        self.assertEqual(
            unexpected_dead_features(audits), ["pre_minutes_per_match"]
        )

    def test_nothing_is_whitelisted_as_a_migration_gap(self):
        from backend.models.feature_audit import KNOWN_MIGRATION_GAPS

        self.assertEqual(
            set(KNOWN_MIGRATION_GAPS), set(),
            "a migration gap left in this set disables the dead-feature guard "
            "for that column; clear it once the rebuild lands",
        )


class TestStalenessDetection(unittest.TestCase):
    """The stale-matrix escape hatch must be content-based and self-clearing.

    It gates the dead-feature guard, so a detector that is wrong in the
    permissive direction silently disables the check. mtime cannot be used:
    the matrices are git-tracked and a fresh CI clone rewrites every timestamp,
    which would make every checkout look freshly built.
    """

    def _write(self, tmpdir, records):
        path = pathlib.Path(tmpdir) / "metadata.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        return tmpdir

    def test_metadata_without_the_marker_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, [{"player_id": 1}, {"player_id": 2}])
            self.assertTrue(matrix_predates_current_pipeline(tmp))

    def test_metadata_with_the_marker_is_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, [{"player_id": 1, "pre_minutes_per_match": 74.2}])
            self.assertFalse(matrix_predates_current_pipeline(tmp))

    def test_partial_marker_counts_as_current(self):
        """One record carrying it means the pipeline emits it."""
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, [
                {"player_id": 1},
                {"player_id": 2, "pre_minutes_per_match": 80.0},
            ])
            self.assertFalse(matrix_predates_current_pipeline(tmp))

    def test_missing_metadata_is_not_stale(self):
        """Unknown provenance must not silence the guard."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(matrix_predates_current_pipeline(tmp))

    def test_unreadable_metadata_is_not_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            (pathlib.Path(tmp) / "metadata.json").write_text("{not json", encoding="utf-8")
            self.assertFalse(matrix_predates_current_pipeline(tmp))

    def test_empty_metadata_is_not_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, [])
            self.assertFalse(matrix_predates_current_pipeline(tmp))

    def test_does_not_depend_on_file_mtime(self):
        """A backdated file with the marker must still read as current."""
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, [{"pre_minutes_per_match": 74.2}])
            path = pathlib.Path(tmp) / "metadata.json"
            old = datetime.datetime(2020, 1, 1).timestamp()
            os.utime(path, (old, old))
            self.assertFalse(matrix_predates_current_pipeline(tmp))

    def test_only_pre_minutes_per_match_is_ever_excused(self):
        """The escape hatch must not widen into a general whitelist."""
        self.assertEqual(set(STALE_MATRIX_PENDING), {"pre_minutes_per_match"})


class TestSavedMatrices(unittest.TestCase):
    """Guard the real matrices, which is the point of the module."""

    @classmethod
    def setUpClass(cls):
        cls.report = audit_saved_matrices()

    def test_no_unexpected_dead_features(self):
        if self.report is None:
            self.skipTest("no saved matrices")
        if matrix_predates_current_pipeline():
            self.skipTest(
                "matrix metadata predates the current pipeline — rebuild "
                f"pending; currently dead: {self.report['unexpected_dead']}"
            )
        self.assertEqual(
            self.report["unexpected_dead"], [],
            "a feature went constant that is not a documented gap — the source "
            "stopped supplying it, a key mapping broke, or a migration "
            "zero-filled a column",
        )

    def test_no_non_finite_values(self):
        if self.report is None:
            self.skipTest("no saved matrices")
        self.assertEqual(self.report["non_finite"], [])

    def test_most_features_are_healthy(self):
        if self.report is None:
            self.skipTest("no saved matrices")
        self.assertGreater(self.report["healthy_fraction"], 0.90)

    def test_report_formats_without_error(self):
        if self.report is None:
            self.skipTest("no saved matrices")
        text = format_report(self.report)
        self.assertIn("features:", text)
        self.assertIn("healthy:", text)

    def test_missing_matrices_returns_none(self):
        self.assertIsNone(audit_saved_matrices("/definitely/not/a/path"))


if __name__ == "__main__":
    unittest.main()
