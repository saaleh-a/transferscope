"""Unit tests for backend.data.fotmob_client using mock responses."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

_TEMP_DIR = tempfile.mkdtemp(prefix="ts_fotmob_test_")
os.environ["CACHE_DIR"] = _TEMP_DIR

from backend.data import cache, fotmob_client


# ── Mock FotMob responses ────────────────────────────────────────────────────

MOCK_SEARCH_RESPONSE = {
    "squad": [],
    "players": [],
    "suggestions": [
        {"type": "player", "id": 961995, "text": "Bukayo Saka"},
        {"type": "player", "id": 123456, "text": "Saka Test"},
    ],
}

MOCK_PLAYER_RESPONSE = {
    "name": "Bukayo Saka",
    "primaryTeam": {"teamName": "Arsenal", "teamId": 9825},
    "origin": {
        "positionDesc": {
            "primaryPosition": {"label": "Right Winger"}
        }
    },
    "mainLeague": {
        "appearances": 20,
        "minutesPlayed": 1650,
    },
    "statSeasons": [
        {
            "statisticsOfSelectedSeason": [
                {
                    "stats": [
                        {"title": "Expected goals (xG)", "per90": 0.35, "value": 6.4},
                        {"title": "Expected assists (xA)", "per90": 0.22, "value": 4.1},
                        {"title": "Total shots", "per90": 2.8, "value": 51},
                        {"title": "Successful dribbles", "per90": 1.5, "value": 28},
                        {"title": "Accurate crosses", "per90": 0.6, "value": 11},
                        {"title": "Touches in opposition box", "per90": 4.2, "value": 77},
                        {"title": "Accurate passes", "per90": 38.0, "value": 697},
                        {"title": "Pass accuracy", "value": 84.5},
                        {"title": "Accurate long balls", "per90": 1.2, "value": 22},
                        {"title": "Chances created", "per90": 2.1, "value": 38},
                        {"title": "Clearances", "per90": 0.3, "value": 5},
                        {"title": "Interceptions", "per90": 0.5, "value": 9},
                        {"title": "Possession won final 3rd", "per90": 1.1, "value": 20},
                        {"title": "xGOT", "per90": 0.28, "value": 5.1},
                        {"title": "Non-penalty xG", "per90": 0.30, "value": 5.5},
                        {"title": "Dispossessed", "per90": 1.0, "value": 18},
                        {"title": "Duels won", "per90": 4.5, "value": 82},
                        {"title": "Aerial duels won", "per90": 0.8, "value": 15},
                        {"title": "Recoveries", "per90": 3.2, "value": 58},
                        {"title": "Fouls won", "per90": 1.4, "value": 25},
                        {"title": "Touches", "per90": 55.0, "value": 1008},
                        {"title": "Goals conceded on pitch", "per90": 0.7, "value": 13},
                        {"title": "xG against on pitch", "per90": 0.65, "value": 12},
                    ]
                }
            ]
        }
    ],
}

MOCK_TEAM_RESPONSE = {
    "squad": [
        {
            "title": "Forwards",
            "members": [
                {"id": 961995, "name": "Bukayo Saka"},
                {"id": 111111, "name": "Gabriel Jesus"},
            ],
        },
        {
            "title": "Midfielders",
            "members": [
                {"id": 222222, "name": "Martin Odegaard"},
            ],
        },
    ]
}


class TestFotMobClient(unittest.TestCase):
    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR
        cache.clear_namespace("fotmob_search")
        cache.clear_namespace("fotmob_player")
        cache.clear_namespace("fotmob_team")

    def tearDown(self):
        cache.close()

    @classmethod
    def tearDownClass(cls):
        cache.close()
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)

    @patch.object(fotmob_client, "_get_client")
    def test_search_player(self, mock_get_client):
        client = MagicMock()
        client.search.return_value = MOCK_SEARCH_RESPONSE
        mock_get_client.return_value = client

        results = fotmob_client.search_player("Saka")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], 961995)
        self.assertEqual(results[0]["name"], "Bukayo Saka")

    @patch.object(fotmob_client, "_get_client")
    def test_search_player_cached(self, mock_get_client):
        client = MagicMock()
        client.search.return_value = MOCK_SEARCH_RESPONSE
        mock_get_client.return_value = client

        fotmob_client.search_player("Saka cached test")
        fotmob_client.search_player("Saka cached test")
        # Only called once due to cache
        client.search.assert_called_once()

    @patch.object(fotmob_client, "_get_client")
    def test_get_player_stats(self, mock_get_client):
        client = MagicMock()
        client.get_player.return_value = MOCK_PLAYER_RESPONSE
        mock_get_client.return_value = client

        stats = fotmob_client.get_player_stats(961995)

        self.assertEqual(stats["name"], "Bukayo Saka")
        self.assertEqual(stats["team"], "Arsenal")
        self.assertEqual(stats["team_id"], 9825)
        self.assertEqual(stats["position"], "Right Winger")
        self.assertEqual(stats["minutes_played"], 1650)
        self.assertEqual(stats["appearances"], 20)

        # Check all 23 metrics are present
        per90 = stats["per90"]
        self.assertEqual(len(per90), 23)

        # Core metrics
        self.assertAlmostEqual(per90["expected_goals"], 0.35)
        self.assertAlmostEqual(per90["expected_assists"], 0.22)
        self.assertAlmostEqual(per90["shots"], 2.8)
        self.assertAlmostEqual(per90["successful_dribbles"], 1.5)
        self.assertAlmostEqual(per90["successful_crosses"], 0.6)
        self.assertAlmostEqual(per90["touches_in_opposition_box"], 4.2)
        self.assertAlmostEqual(per90["successful_passes"], 38.0)
        self.assertAlmostEqual(per90["clearances"], 0.3)
        self.assertAlmostEqual(per90["interceptions"], 0.5)
        self.assertAlmostEqual(per90["possession_won_final_3rd"], 1.1)

        # Additional metrics
        self.assertAlmostEqual(per90["xg_on_target"], 0.28)
        self.assertAlmostEqual(per90["recoveries"], 3.2)
        self.assertAlmostEqual(per90["touches"], 55.0)

    @patch.object(fotmob_client, "_get_client")
    def test_get_player_stats_percentage_handling(self, mock_get_client):
        client = MagicMock()
        client.get_player.return_value = MOCK_PLAYER_RESPONSE
        mock_get_client.return_value = client

        stats = fotmob_client.get_player_stats(961995)
        # pass_completion_pct is a percentage, stored as-is
        self.assertAlmostEqual(stats["per90"]["pass_completion_pct"], 84.5)

    @patch.object(fotmob_client, "_get_client")
    def test_get_team_players_stats(self, mock_get_client):
        client = MagicMock()
        client.get_team.return_value = MOCK_TEAM_RESPONSE
        mock_get_client.return_value = client

        players = fotmob_client.get_team_players_stats(9825)
        self.assertEqual(len(players), 3)
        self.assertEqual(players[0]["name"], "Bukayo Saka")
        self.assertEqual(players[0]["position"], "Forwards")
        self.assertEqual(players[2]["name"], "Martin Odegaard")
        self.assertEqual(players[2]["position"], "Midfielders")

    def test_parse_empty_response(self):
        result = fotmob_client._parse_player_response({})
        self.assertEqual(result["name"], "")
        self.assertEqual(result["minutes_played"], 0)
        self.assertTrue(all(v is None for v in result["per90"].values()))

    def test_parse_non_dict_response(self):
        result = fotmob_client._parse_player_response(None)
        self.assertEqual(result["name"], "")

    def test_all_metrics_defined(self):
        self.assertEqual(len(fotmob_client.CORE_METRICS), 13)
        self.assertEqual(len(fotmob_client.ADDITIONAL_METRICS), 10)
        self.assertEqual(len(fotmob_client.ALL_METRICS), 23)


if __name__ == "__main__":
    unittest.main()
