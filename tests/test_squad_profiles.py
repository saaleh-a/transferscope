"""Tests for Sofascore squad profiles and pre-season season fallback.

Both cover failures that only appear during the summer transfer window — the
period when a transfer tool is most used:

* the newest season exists in Sofascore's season list before any match is
  played, so "current season" stats 404 for every player;
* market value and contract data live on the squad endpoint, not the bulk
  league-statistics endpoint.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from unittest.mock import patch

from backend.data import cache
from backend.data import sofascore_client as sc

_TEMP_DIR = tempfile.mkdtemp()


class TestSeasonFallback(unittest.TestCase):
    """A newly-created season with no matches must not blank out a player."""

    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR
        for ns in ("sofascore_player", "sofascore_seasons", "sofascore_neg"):
            cache.clear_namespace(ns)

    def tearDown(self):
        cache.close()

    def test_recent_season_ids_starts_with_preferred(self):
        seasons = [{"id": 96668}, {"id": 76986}, {"id": 61627}, {"id": 52186}]
        with patch.object(sc, "get_season_list", return_value=seasons):
            ids = sc._recent_season_ids(17, 96668)
        self.assertEqual(ids[0], 96668)
        self.assertEqual(ids[1], 76986)

    def test_recent_season_ids_respects_depth(self):
        seasons = [{"id": i} for i in range(20)]
        with patch.object(sc, "get_season_list", return_value=seasons):
            ids = sc._recent_season_ids(17, 0)
        self.assertLessEqual(len(ids), sc._SEASON_FALLBACK_DEPTH)

    def test_recent_season_ids_handles_missing_preferred(self):
        seasons = [{"id": 111}, {"id": 222}]
        with patch.object(sc, "get_season_list", return_value=seasons):
            ids = sc._recent_season_ids(17, 999)
        self.assertEqual(ids[0], 999)
        self.assertIn(111, ids)

    def test_recent_season_ids_survives_bad_payload(self):
        with patch.object(sc, "get_season_list", return_value=None):
            self.assertEqual(sc._recent_season_ids(17, 5), [5])
        with patch.object(sc, "get_season_list", side_effect=RuntimeError("boom")):
            self.assertEqual(sc._recent_season_ids(17, 5), [5])
        with patch.object(sc, "get_season_list", return_value=[{"no_id": 1}]):
            self.assertEqual(sc._recent_season_ids(17, 5), [5])

    def test_has_usable_minutes(self):
        self.assertTrue(sc._has_usable_minutes({"statistics": {"minutesPlayed": 900}}))
        self.assertFalse(sc._has_usable_minutes({"statistics": {"minutesPlayed": 0}}))
        self.assertFalse(sc._has_usable_minutes({"statistics": {}}))
        self.assertFalse(sc._has_usable_minutes({}))
        self.assertFalse(sc._has_usable_minutes(None))
        self.assertFalse(sc._has_usable_minutes("nonsense"))
        self.assertFalse(
            sc._has_usable_minutes({"statistics": {"minutesPlayed": "abc"}})
        )

    def test_player_stats_fall_back_to_previous_season(self):
        """The regression: pre-season returns nothing, so walk back a season."""
        profile = {
            "player": {
                "id": 1,
                "name": "Test Player",
                "team": {"id": 44, "name": "Test FC",
                         "tournament": {"uniqueTournament": {"id": 17}}},
            }
        }
        current_empty = {"statistics": {"minutesPlayed": 0}}
        previous_good = {
            "statistics": {"minutesPlayed": 2000, "rating": 7.2, "goals": 10}
        }

        def fake_get(path):
            if path == "/player/1":
                return profile
            if "/season/96668/" in path:
                return current_empty
            if "/season/76986/" in path:
                return previous_good
            return None

        seasons = [{"id": 96668}, {"id": 76986}, {"id": 61627}]
        with patch.object(sc, "_get", side_effect=fake_get), \
             patch.object(sc, "get_season_list", return_value=seasons), \
             patch.object(sc, "_get_current_season_id", return_value=96668):
            result = sc.get_player_stats(1)

        self.assertEqual(result["minutes_played"], 2000)
        self.assertEqual(result["season_id"], 76986)

    def test_player_stats_prefers_current_season_when_it_has_data(self):
        profile = {
            "player": {
                "id": 2,
                "name": "In Season",
                "team": {"id": 44, "name": "Test FC",
                         "tournament": {"uniqueTournament": {"id": 17}}},
            }
        }
        current_good = {"statistics": {"minutesPlayed": 1500}}
        calls = []

        def fake_get(path):
            calls.append(path)
            if path == "/player/2":
                return profile
            if "/season/96668/" in path:
                return current_good
            return {"statistics": {"minutesPlayed": 999}}

        seasons = [{"id": 96668}, {"id": 76986}]
        with patch.object(sc, "_get", side_effect=fake_get), \
             patch.object(sc, "get_season_list", return_value=seasons), \
             patch.object(sc, "_get_current_season_id", return_value=96668):
            result = sc.get_player_stats(2)

        self.assertEqual(result["minutes_played"], 1500)
        self.assertEqual(result["season_id"], 96668)
        # Must not waste a call on the older season once the current one works.
        self.assertFalse(any("/season/76986/" in c for c in calls))


class TestSquadProfiles(unittest.TestCase):
    """Market value / contract enrichment from the team squad endpoint."""

    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR
        cache.clear_namespace("sofascore_squad_profiles")

    def tearDown(self):
        cache.close()

    @staticmethod
    def _payload():
        future = int(time.time() + 365.25 * 86400 * 2)  # ~2 years out
        dob = int(time.time() - 365.25 * 86400 * 24)    # ~24 years old
        return {
            "players": [
                {"player": {
                    "id": 100, "name": "Rich Player",
                    "proposedMarketValue": 88_000_000,
                    "contractUntilTimestamp": future,
                    "dateOfBirthTimestamp": dob,
                    "height": 191, "weight": 82, "preferredFoot": "Right",
                }},
                {"player": {
                    "id": 101, "name": "No Value",
                    "dateOfBirthTimestamp": dob,
                }},
                {"player": {
                    "id": 102, "name": "Raw Value Only",
                    "proposedMarketValueRaw": {"value": 5_000_000, "currency": "EUR"},
                }},
            ]
        }

    def test_extracts_value_contract_and_profile(self):
        with patch.object(sc, "_get", return_value=self._payload()):
            profiles = sc.get_team_squad_profiles(44)

        self.assertEqual(len(profiles), 3)
        rich = profiles[100]
        self.assertEqual(rich["market_value"], 88_000_000.0)
        self.assertAlmostEqual(rich["contract_years_left"], 2.0, places=1)
        self.assertEqual(rich["height_cm"], 191.0)
        self.assertEqual(rich["preferred_foot"], "Right")
        self.assertEqual(rich["age"], 24)

    def test_missing_value_is_none_not_zero(self):
        """Critical: unknown price must never read as free."""
        with patch.object(sc, "_get", return_value=self._payload()):
            profiles = sc.get_team_squad_profiles(44)
        self.assertIsNone(profiles[101]["market_value"])
        self.assertIsNone(profiles[101]["contract_years_left"])

    def test_falls_back_to_raw_market_value_field(self):
        with patch.object(sc, "_get", return_value=self._payload()):
            profiles = sc.get_team_squad_profiles(44)
        self.assertEqual(profiles[102]["market_value"], 5_000_000.0)

    def test_bad_payload_returns_empty(self):
        for payload in (None, {}, {"players": []}, {"players": ["nonsense"]}):
            with patch.object(sc, "_get", return_value=payload):
                self.assertEqual(sc.get_team_squad_profiles(999), {})

    def test_league_profiles_aggregate_across_teams(self):
        teams = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
        with patch.object(sc, "_get_league_team_ids", return_value=teams), \
             patch.object(
                 sc, "get_team_squad_profiles",
                 side_effect=[{1: {"market_value": 10.0}}, {2: {"market_value": 20.0}}],
             ):
            combined = sc.get_league_squad_profiles(17, 61627)
        self.assertEqual(set(combined), {1, 2})

    def test_league_profiles_survive_one_failing_team(self):
        teams = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
        with patch.object(sc, "_get_league_team_ids", return_value=teams), \
             patch.object(
                 sc, "get_team_squad_profiles",
                 side_effect=[RuntimeError("boom"), {2: {"market_value": 20.0}}],
             ):
            combined = sc.get_league_squad_profiles(17, 61627)
        self.assertEqual(set(combined), {2})

    def test_league_profiles_with_no_teams(self):
        with patch.object(sc, "_get_league_team_ids", return_value=[]):
            self.assertEqual(sc.get_league_squad_profiles(17, 61627), {})


if __name__ == "__main__":
    unittest.main()
