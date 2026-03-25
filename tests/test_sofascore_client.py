"""Unit tests for backend.data.sofascore_client using mock responses."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

_TEMP_DIR = tempfile.mkdtemp(prefix="ts_sofascore_test_")
os.environ["CACHE_DIR"] = _TEMP_DIR

from backend.data import cache, sofascore_client


# ── Mock Sofascore API responses ─────────────────────────────────────────────

MOCK_SEARCH_RESPONSE = {
    "results": [
        {
            "entity": {
                "id": 961995,
                "name": "Bukayo Saka",
                "team": {
                    "id": 789,
                    "name": "Arsenal",
                    "tournament": {"id": 17},
                },
            }
        },
        {
            "entity": {
                "id": 123456,
                "name": "Saka Test",
                "team": {"id": 999, "name": "Test FC"},
            }
        },
    ]
}

MOCK_PLAYER_PROFILE = {
    "player": {
        "id": 961995,
        "name": "Bukayo Saka",
        "team": {
            "id": 789,
            "name": "Arsenal",
            "tournament": {"id": 17},
        },
        "positionDescription": "Right Winger",
    }
}

# Raw totals — per-90 values will be computed using minutesPlayed=1650
# nineties = 1650 / 90 = 18.333...
# e.g. expectedGoals=6.4 → per90 = 6.4 / 18.333 ≈ 0.349
MOCK_STATS_RESPONSE = {
    "statistics": {
        "minutesPlayed": 1650,
        "appearances": 20,
        "matchesStarted": 20,
        "expectedGoals": 6.4,
        "expectedAssists": 4.1,
        "shots": 51,
        "successfulDribbles": 28,
        "accurateCrosses": 11,
        "penaltyAreaTouches": 77,
        "accuratePasses": 697,
        "accuratePassesPercentage": 84.5,
        "accurateLongBalls": 22,
        "keyPasses": 38,
        "clearances": 5,
        "interceptions": 9,
        "wonTackles": 20,
        "expectedGoalsOnTarget": 5.1,
        "expectedGoalsNoPenalty": 5.5,
        "dispossessed": 18,
        "duelsWonPercentage": 55.4,
        "aerialDuelsWonPercentage": 45.5,
        "ballRecovery": 58,
        "foulsDrawn": 25,
        "touches": 1008,
        "goalsConceded": 13,
    }
}

MOCK_SEASONS_RESPONSE = {
    "seasons": [
        {"id": 61627, "name": "2024/2025"},
        {"id": 52186, "name": "2023/2024"},
    ]
}

MOCK_TEAM_RESPONSE = {
    "players": [
        {
            "name": "Forwards",
            "players": [
                {"id": 961995, "name": "Bukayo Saka"},
                {"id": 111111, "name": "Gabriel Jesus"},
            ],
        },
        {
            "name": "Midfielders",
            "players": [
                {"id": 222222, "name": "Martin Odegaard"},
            ],
        },
    ]
}


class TestSofascoreClient(unittest.TestCase):
    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR
        cache.clear_namespace("sofascore_search")
        cache.clear_namespace("sofascore_player")
        cache.clear_namespace("sofascore_team")
        cache.clear_namespace("sofascore_player_meta")
        cache.clear_namespace("sofascore_seasons")

    def tearDown(self):
        cache.close()

    @classmethod
    def tearDownClass(cls):
        cache.close()
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)

    @patch.object(sofascore_client, "_get")
    def test_search_player(self, mock_get):
        mock_get.return_value = MOCK_SEARCH_RESPONSE

        results = sofascore_client.search_player("Saka")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], 961995)
        self.assertEqual(results[0]["name"], "Bukayo Saka")

    @patch.object(sofascore_client, "_get")
    def test_search_player_cached(self, mock_get):
        mock_get.return_value = MOCK_SEARCH_RESPONSE

        sofascore_client.search_player("Saka cached test")
        sofascore_client.search_player("Saka cached test")
        # Only called once due to cache
        mock_get.assert_called_once()

    @patch.object(sofascore_client, "_get")
    def test_search_stores_tournament_meta(self, mock_get):
        mock_get.return_value = MOCK_SEARCH_RESPONSE

        sofascore_client.search_player("Saka")
        # Player 961995 had tournament_id=17 in the search result
        tid = sofascore_client._get_cached_tournament_id(961995)
        self.assertEqual(tid, 17)

    @patch.object(sofascore_client, "_get")
    def test_get_player_stats(self, mock_get):
        def side_effect(path):
            if "/player/961995" == path and "unique-tournament" not in path:
                return MOCK_PLAYER_PROFILE
            if "/seasons" in path:
                return MOCK_SEASONS_RESPONSE
            if "statistics" in path:
                return MOCK_STATS_RESPONSE
            return None

        mock_get.side_effect = side_effect

        # Pre-seed the player meta so profile lookup is skipped for tournament
        sofascore_client._cache_player_meta(961995, 17)

        stats = sofascore_client.get_player_stats(961995)

        self.assertEqual(stats["name"], "Bukayo Saka")
        self.assertEqual(stats["team"], "Arsenal")
        self.assertEqual(stats["team_id"], 789)
        self.assertEqual(stats["minutes_played"], 1650)
        self.assertEqual(stats["appearances"], 20)

    @patch.object(sofascore_client, "_get")
    def test_get_player_stats_per90_computation(self, mock_get):
        def side_effect(path):
            if "/player/961995" == path and "unique-tournament" not in path:
                return MOCK_PLAYER_PROFILE
            if "/seasons" in path:
                return MOCK_SEASONS_RESPONSE
            if "statistics" in path:
                return MOCK_STATS_RESPONSE
            return None

        mock_get.side_effect = side_effect
        sofascore_client._cache_player_meta(961995, 17)

        stats = sofascore_client.get_player_stats(961995)
        per90 = stats["per90"]

        # Verify all 23 metrics are present
        self.assertEqual(len(per90), 23)

        # Validate per-90 computation: value / (1650/90)
        nineties = 1650 / 90
        self.assertAlmostEqual(per90["expected_goals"], 6.4 / nineties, places=3)
        self.assertAlmostEqual(per90["expected_assists"], 4.1 / nineties, places=3)
        self.assertAlmostEqual(per90["shots"], 51 / nineties, places=3)
        self.assertAlmostEqual(per90["successful_dribbles"], 28 / nineties, places=3)
        self.assertAlmostEqual(per90["successful_crosses"], 11 / nineties, places=3)
        self.assertAlmostEqual(per90["touches_in_opposition_box"], 77 / nineties, places=3)
        self.assertAlmostEqual(per90["successful_passes"], 697 / nineties, places=3)
        self.assertAlmostEqual(per90["accurate_long_balls"], 22 / nineties, places=3)
        self.assertAlmostEqual(per90["chances_created"], 38 / nineties, places=3)
        self.assertAlmostEqual(per90["clearances"], 5 / nineties, places=3)
        self.assertAlmostEqual(per90["interceptions"], 9 / nineties, places=3)
        self.assertAlmostEqual(per90["recoveries"], 58 / nineties, places=3)
        self.assertAlmostEqual(per90["touches"], 1008 / nineties, places=3)

    @patch.object(sofascore_client, "_get")
    def test_percentage_metrics_not_divided(self, mock_get):
        def side_effect(path):
            if "/player/961995" == path and "unique-tournament" not in path:
                return MOCK_PLAYER_PROFILE
            if "/seasons" in path:
                return MOCK_SEASONS_RESPONSE
            if "statistics" in path:
                return MOCK_STATS_RESPONSE
            return None

        mock_get.side_effect = side_effect
        sofascore_client._cache_player_meta(961995, 17)

        stats = sofascore_client.get_player_stats(961995)
        per90 = stats["per90"]

        # Percentages stored as-is
        self.assertAlmostEqual(per90["pass_completion_pct"], 84.5)
        self.assertAlmostEqual(per90["duels_won_pct"], 55.4)
        self.assertAlmostEqual(per90["aerial_duels_won_pct"], 45.5)

    @patch.object(sofascore_client, "_get")
    def test_get_team_players_stats(self, mock_get):
        mock_get.return_value = MOCK_TEAM_RESPONSE

        players = sofascore_client.get_team_players_stats(789)
        self.assertEqual(len(players), 3)
        self.assertEqual(players[0]["name"], "Bukayo Saka")
        self.assertEqual(players[0]["position"], "Forwards")
        self.assertEqual(players[2]["name"], "Martin Odegaard")
        self.assertEqual(players[2]["position"], "Midfielders")

    @patch.object(sofascore_client, "_get")
    def test_get_player_stats_no_minutes(self, mock_get):
        no_minutes_stats = {
            "statistics": {
                "minutesPlayed": 0,
                "appearances": 0,
                "expectedGoals": 2.0,
            }
        }

        def side_effect(path):
            if "/player/1" == path and "unique-tournament" not in path:
                return {"player": {"id": 1, "name": "Test", "team": {}}}
            if "/seasons" in path:
                return MOCK_SEASONS_RESPONSE
            if "statistics" in path:
                return no_minutes_stats
            return None

        mock_get.side_effect = side_effect
        sofascore_client._cache_player_meta(1, 17)

        stats = sofascore_client.get_player_stats(1)
        # All per-90 should be None when minutes=0 (can't compute per-90)
        self.assertTrue(all(v is None for v in stats["per90"].values()))

    @patch.object(sofascore_client, "_get")
    def test_get_player_stats_api_failure(self, mock_get):
        mock_get.return_value = None

        stats = sofascore_client.get_player_stats(9999)
        self.assertEqual(stats["name"], "")
        self.assertEqual(stats["minutes_played"], 0)
        self.assertTrue(all(v is None for v in stats["per90"].values()))

    def test_all_metrics_defined(self):
        self.assertEqual(len(sofascore_client.CORE_METRICS), 13)
        self.assertEqual(len(sofascore_client.ADDITIONAL_METRICS), 10)
        self.assertEqual(len(sofascore_client.ALL_METRICS), 23)

    def test_parse_stats_empty(self):
        result = sofascore_client._parse_stats({}, 0)
        self.assertTrue(all(v is None for v in result.values()))

    def test_parse_stats_correct_per90(self):
        stats = {"expectedGoals": 9.0, "accuratePassesPercentage": 80.0}
        result = sofascore_client._parse_stats(stats, 900)
        # 9.0 / (900/90) = 9.0 / 10 = 0.9
        self.assertAlmostEqual(result["expected_goals"], 0.9)
        # Percentage kept as-is
        self.assertAlmostEqual(result["pass_completion_pct"], 80.0)

    def test_get_current_season_id_caches(self):
        with patch.object(sofascore_client, "_get") as mock_get:
            mock_get.return_value = MOCK_SEASONS_RESPONSE
            sid1 = sofascore_client._get_current_season_id(17)
            sid2 = sofascore_client._get_current_season_id(17)
            self.assertEqual(sid1, 61627)
            self.assertEqual(sid2, 61627)
            # Only one actual API call
            mock_get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
