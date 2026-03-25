"""Unit tests for the new sofascore_client features:
- search_team
- get_player_transfer_history
- get_league_player_stats
- get_season_list
- get_player_stats_for_season
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

_TEMP_DIR = tempfile.mkdtemp(prefix="ts_new_features_test_")
os.environ["CACHE_DIR"] = _TEMP_DIR

from backend.data import cache, sofascore_client


def tearDownModule():
    """Clean up the temp cache directory after all tests."""
    cache.close()
    shutil.rmtree(_TEMP_DIR, ignore_errors=True)


# ── Mock API responses ───────────────────────────────────────────────────────

MOCK_TEAM_SEARCH_RESPONSE = {
    "results": [
        {
            "entity": {
                "id": 42,
                "name": "Arsenal",
                "tournament": {"id": 17},
                "country": {"name": "England"},
            }
        },
        {
            "entity": {
                "id": 2697,
                "name": "Arsenal de Sarandí",
                "tournament": {"id": 155},
                "country": {"name": "Argentina"},
            }
        },
    ]
}

MOCK_TRANSFER_HISTORY_RESPONSE = {
    "transferHistory": [
        {
            "transferDateTimestamp": 1656633600,  # 2022-07-01
            "transferFrom": {"id": 100, "name": "AC Milan"},
            "transferTo": {"id": 42, "name": "Arsenal"},
            "type": "transfer",
        },
        {
            "transferDateTimestamp": 1625097600,  # 2021-07-01
            "transferFrom": {"id": 200, "name": "Lille"},
            "transferTo": {"id": 100, "name": "AC Milan"},
            "type": "transfer",
        },
    ]
}

MOCK_SEASONS_RESPONSE = {
    "seasons": [
        {"id": 61627, "name": "2024/2025"},
        {"id": 52186, "name": "2023/2024"},
        {"id": 41886, "name": "2022/2023"},
    ]
}

MOCK_LEAGUE_STATS_RESPONSE = {
    "results": [
        {
            "player": {"id": 961995, "name": "Bukayo Saka", "position": "F"},
            "team": {"id": 42, "name": "Arsenal"},
            "minutesPlayed": 1800,
            "statistics": {
                "minutesPlayed": 1800,
                "expectedGoals": 9.0,
                "accuratePasses": 500,
                "accuratePassesPercentage": 85.0,
            },
        },
        {
            "player": {"id": 111111, "name": "Mohamed Salah", "position": "F"},
            "team": {"id": 44, "name": "Liverpool"},
            "minutesPlayed": 2100,
            "statistics": {
                "minutesPlayed": 2100,
                "expectedGoals": 12.3,
                "accuratePasses": 600,
                "accuratePassesPercentage": 82.0,
            },
        },
    ]
}

MOCK_PLAYER_PROFILE = {
    "player": {
        "id": 961995,
        "name": "Bukayo Saka",
        "team": {"id": 42, "name": "Arsenal", "tournament": {"id": 17}},
        "positionDescription": "Right Winger",
    }
}

MOCK_SEASON_STATS_RESPONSE = {
    "statistics": {
        "minutesPlayed": 1200,
        "appearances": 15,
        "expectedGoals": 4.5,
        "accuratePassesPercentage": 83.0,
    }
}


class TestSearchTeam(unittest.TestCase):
    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR
        cache.clear_namespace("sofascore_team_search")

    def tearDown(self):
        cache.close()

    @patch.object(sofascore_client, "_get")
    def test_search_team_returns_results(self, mock_get):
        mock_get.return_value = MOCK_TEAM_SEARCH_RESPONSE

        results = sofascore_client.search_team("Arsenal")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], 42)
        self.assertEqual(results[0]["name"], "Arsenal")
        self.assertEqual(results[0]["tournament_id"], 17)
        self.assertEqual(results[0]["country"], "England")

    @patch.object(sofascore_client, "_get")
    def test_search_team_cached(self, mock_get):
        mock_get.return_value = MOCK_TEAM_SEARCH_RESPONSE

        sofascore_client.search_team("Arsenal cached")
        sofascore_client.search_team("Arsenal cached")
        mock_get.assert_called_once()

    @patch.object(sofascore_client, "_get")
    def test_search_team_empty_response(self, mock_get):
        mock_get.return_value = None

        results = sofascore_client.search_team("Nonexistent FC")
        self.assertEqual(results, [])


class TestTransferHistory(unittest.TestCase):
    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR
        cache.clear_namespace("sofascore_transfers")

    def tearDown(self):
        cache.close()

    @patch.object(sofascore_client, "_get")
    def test_get_transfer_history(self, mock_get):
        mock_get.return_value = MOCK_TRANSFER_HISTORY_RESPONSE

        transfers = sofascore_client.get_player_transfer_history(961995)
        self.assertEqual(len(transfers), 2)
        self.assertEqual(transfers[0]["to_team"]["name"], "Arsenal")
        self.assertEqual(transfers[0]["from_team"]["name"], "AC Milan")
        self.assertEqual(transfers[0]["type"], "transfer")
        self.assertEqual(transfers[0]["transfer_date"], "2022-07-01")

    @patch.object(sofascore_client, "_get")
    def test_transfer_history_cached(self, mock_get):
        mock_get.return_value = MOCK_TRANSFER_HISTORY_RESPONSE

        sofascore_client.get_player_transfer_history(888888)
        sofascore_client.get_player_transfer_history(888888)
        mock_get.assert_called_once()

    @patch.object(sofascore_client, "_get")
    def test_transfer_history_empty(self, mock_get):
        mock_get.return_value = None

        transfers = sofascore_client.get_player_transfer_history(999999)
        self.assertEqual(transfers, [])


class TestLeaguePlayerStats(unittest.TestCase):
    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR
        cache.clear_namespace("sofascore_league_stats")
        cache.clear_namespace("sofascore_seasons")

    def tearDown(self):
        cache.close()

    @patch.object(sofascore_client, "_get")
    def test_get_league_player_stats(self, mock_get):
        def side_effect(path):
            if "/seasons" in path:
                return MOCK_SEASONS_RESPONSE
            if "/statistics" in path:
                return MOCK_LEAGUE_STATS_RESPONSE
            return None

        mock_get.side_effect = side_effect

        players = sofascore_client.get_league_player_stats(17, limit=10)
        self.assertEqual(len(players), 2)
        self.assertEqual(players[0]["name"], "Bukayo Saka")
        self.assertEqual(players[0]["team"], "Arsenal")
        self.assertEqual(players[0]["minutes_played"], 1800)
        # Per-90 should be computed
        nineties = 1800 / 90
        self.assertIsNotNone(players[0]["per90"].get("expected_goals"))
        self.assertAlmostEqual(
            players[0]["per90"]["expected_goals"], 9.0 / nineties, places=3
        )

    @patch.object(sofascore_client, "_get")
    def test_get_league_player_stats_no_season(self, mock_get):
        mock_get.return_value = None

        players = sofascore_client.get_league_player_stats(9999)
        self.assertEqual(players, [])


class TestSeasonList(unittest.TestCase):
    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR
        cache.clear_namespace("sofascore_season_list")

    def tearDown(self):
        cache.close()

    @patch.object(sofascore_client, "_get")
    def test_get_season_list(self, mock_get):
        mock_get.return_value = MOCK_SEASONS_RESPONSE

        seasons = sofascore_client.get_season_list(17)
        self.assertEqual(len(seasons), 3)
        self.assertEqual(seasons[0]["name"], "2024/2025")
        self.assertEqual(seasons[0]["id"], 61627)

    @patch.object(sofascore_client, "_get")
    def test_get_season_list_cached(self, mock_get):
        mock_get.return_value = MOCK_SEASONS_RESPONSE

        sofascore_client.get_season_list(17)
        sofascore_client.get_season_list(17)
        mock_get.assert_called_once()

    @patch.object(sofascore_client, "_get")
    def test_get_season_list_empty(self, mock_get):
        mock_get.return_value = None

        seasons = sofascore_client.get_season_list(9999)
        self.assertEqual(seasons, [])


class TestPlayerStatsForSeason(unittest.TestCase):
    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR
        cache.clear_namespace("sofascore_player_season")

    def tearDown(self):
        cache.close()

    @patch.object(sofascore_client, "_get")
    def test_get_player_stats_for_season(self, mock_get):
        def side_effect(path):
            if "/player/961995" == path:
                return MOCK_PLAYER_PROFILE
            if "statistics" in path:
                return MOCK_SEASON_STATS_RESPONSE
            return None

        mock_get.side_effect = side_effect

        stats = sofascore_client.get_player_stats_for_season(961995, 17, 52186)
        self.assertEqual(stats["name"], "Bukayo Saka")
        self.assertEqual(stats["team"], "Arsenal")
        self.assertEqual(stats["minutes_played"], 1200)
        self.assertEqual(stats["appearances"], 15)
        # Verify per-90 computation
        nineties = 1200 / 90
        self.assertAlmostEqual(
            stats["per90"]["expected_goals"], 4.5 / nineties, places=3
        )

    @patch.object(sofascore_client, "_get")
    def test_get_player_stats_for_season_api_failure(self, mock_get):
        mock_get.return_value = None

        stats = sofascore_client.get_player_stats_for_season(9999, 17, 52186)
        self.assertEqual(stats["name"], "")
        self.assertEqual(stats["minutes_played"], 0)


class TestUnixToIso(unittest.TestCase):
    def test_valid_timestamp(self):
        result = sofascore_client._unix_to_iso(1656633600)
        self.assertEqual(result, "2022-07-01")

    def test_none_timestamp(self):
        result = sofascore_client._unix_to_iso(None)
        self.assertIsNone(result)

    def test_invalid_timestamp(self):
        result = sofascore_client._unix_to_iso("not_a_number")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
