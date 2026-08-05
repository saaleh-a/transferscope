"""Tests for the field-availability gate.

Three feature slots in the shipped matrix are permanently constant because
Sofascore never served those stats, and each cost a full rebuild to discover.
This gate answers the question before the rebuild.

The subtle case it exists for: a field can be well populated *today* and absent
from every historical season. kilometersCovered is 98% populated for 25/26 and
missing from 24/25 through 20/21, so it would train as constant zero while
looking healthy in a spot check.
"""

from __future__ import annotations

import importlib.util
import os
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "check_field_availability",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "check_field_availability.py",
    ),
)
cfa = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cfa)


class TestThresholds(unittest.TestCase):
    def test_thresholds_are_sane(self):
        self.assertGreater(cfa.MIN_PLAYER_COVERAGE, 0.5)
        self.assertGreater(cfa.MIN_SEASON_COVERAGE, 0.5)
        self.assertLessEqual(cfa.MIN_PLAYER_COVERAGE, 1.0)
        self.assertLessEqual(cfa.MIN_SEASON_COVERAGE, 1.0)

    def test_candidate_list_excludes_physical_metrics(self):
        """The physical fields fail the historical test, so they are not listed."""
        for field in ("kilometersCovered", "numberOfSprints", "topSpeed"):
            self.assertNotIn(field, cfa.CANDIDATE_FIELDS)

    def test_candidate_list_holds_the_verified_fields(self):
        for field in ("tackles", "bigChancesCreated", "accurateFinalThirdPasses"):
            self.assertIn(field, cfa.CANDIDATE_FIELDS)

    def test_candidate_list_has_no_duplicates(self):
        self.assertEqual(
            len(cfa.CANDIDATE_FIELDS), len(set(cfa.CANDIDATE_FIELDS))
        )


class TestUnreachableApi(unittest.TestCase):
    """A rate-limited run must not read as evidence against a field."""

    def test_no_data_returns_empty(self):
        from unittest.mock import patch

        with patch.object(cfa, "_session"), patch.object(cfa, "_get", return_value=None):
            self.assertEqual(cfa.check_fields(["tackles"]), {})

    def test_missing_seasons_payload_returns_empty(self):
        from unittest.mock import patch

        with patch.object(cfa, "_session"), \
             patch.object(cfa, "_get", return_value={"seasons": []}):
            self.assertEqual(cfa.check_fields(["tackles"]), {})


class TestClassification(unittest.TestCase):
    """The verdict logic, exercised through a stubbed network layer."""

    @staticmethod
    def _run(season_present: bool, player_present: bool):
        from unittest.mock import patch

        seasons = {"seasons": [{"id": i} for i in range(6)]}
        squad = {"players": [{"player": {"id": 900 + i}} for i in range(5)]}
        stats_present = {
            "statistics": {"minutesPlayed": 2000, "tackles": 40}
        }
        stats_absent = {"statistics": {"minutesPlayed": 2000}}

        def fake_get(_session, url, attempts=3):
            if "/seasons" in url:
                return seasons
            if "/players" in url:
                return squad
            if "/statistics/overall" in url:
                # Player-coverage calls use season id 0 (the newest).
                is_current = "/season/0/" in url
                if is_current:
                    return stats_present if player_present else stats_absent
                return stats_present if season_present else stats_absent
            return None

        with patch.object(cfa, "_session"), patch.object(cfa, "_get", fake_get):
            return cfa.check_fields(["tackles"])["tackles"]

    def test_present_everywhere_is_safe(self):
        result = self._run(season_present=True, player_present=True)
        self.assertTrue(result["safe"])
        self.assertEqual(result["verdict"], "safe to add")

    def test_current_season_only_is_rejected(self):
        """The kilometersCovered case: healthy today, absent historically."""
        result = self._run(season_present=False, player_present=True)
        self.assertFalse(result["safe"])
        self.assertIn("CURRENT SEASON ONLY", result["verdict"])

    def test_sparse_across_players_is_rejected(self):
        result = self._run(season_present=True, player_present=False)
        self.assertFalse(result["safe"])
        self.assertIn("sparse", result["verdict"])


if __name__ == "__main__":
    unittest.main()
