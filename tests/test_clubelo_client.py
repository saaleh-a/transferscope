"""Unit tests for backend.data.clubelo_client using mock responses."""

import os
import shutil
import tempfile
import unittest
from datetime import date
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd

_TEMP_DIR = tempfile.mkdtemp(prefix="ts_clubelo_test_")
os.environ["CACHE_DIR"] = _TEMP_DIR

from backend.data import cache, clubelo_client


# ── Mock ClubElo data ────────────────────────────────────────────────────────

def _mock_elo_df():
    """Build a DataFrame mimicking soccerdata ClubElo output."""
    data = {
        "rank": [1, 2, 3, 4, 5],
        "country": ["ENG", "ENG", "ESP", "ESP", "GER"],
        "level": [1, 1, 1, 1, 1],
        "elo": [2050.0, 1980.0, 2010.0, 1890.0, 1970.0],
        "from": ["2025-01-01"] * 5,
        "to": ["2025-12-31"] * 5,
        "league": [
            "ENG-Premier League",
            "ENG-Premier League",
            "ESP-La Liga",
            "ESP-La Liga",
            "GER-Bundesliga",
        ],
    }
    df = pd.DataFrame(data, index=["Arsenal", "Man City", "Real Madrid", "Barcelona", "Bayern Munich"])
    df.index.name = "team"
    return df


def _mock_history_df():
    data = {
        "rank": [1, 2, 3],
        "team": ["Arsenal", "Arsenal", "Arsenal"],
        "country": ["ENG", "ENG", "ENG"],
        "level": [1, 1, 1],
        "elo": [2000.0, 2020.0, 2050.0],
        "to": ["2025-06-01", "2025-09-01", "2025-12-31"],
    }
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(["2025-01-01", "2025-06-01", "2025-09-01"])
    df.index.name = "from"
    return df


class TestClubEloClient(unittest.TestCase):
    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR
        cache.clear_namespace("clubelo_date")
        cache.clear_namespace("clubelo_history")
        clubelo_client._elo_instance = None

    def tearDown(self):
        cache.close()

    @classmethod
    def tearDownClass(cls):
        cache.close()
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)

    @patch.object(clubelo_client, "_get_elo")
    def test_get_all_by_date(self, mock_get_elo):
        mock_ce = MagicMock()
        mock_ce.read_by_date.return_value = _mock_elo_df()
        mock_get_elo.return_value = mock_ce

        df = clubelo_client.get_all_by_date(date(2025, 6, 15))
        self.assertEqual(len(df), 5)
        self.assertIn("Arsenal", df.index)
        self.assertAlmostEqual(df.loc["Arsenal", "elo"], 2050.0)

    @patch.object(clubelo_client, "_get_elo")
    def test_get_all_by_date_cached(self, mock_get_elo):
        mock_ce = MagicMock()
        mock_ce.read_by_date.return_value = _mock_elo_df()
        mock_get_elo.return_value = mock_ce

        clubelo_client.get_all_by_date(date(2025, 7, 1))
        clubelo_client.get_all_by_date(date(2025, 7, 1))
        mock_ce.read_by_date.assert_called_once()

    @patch.object(clubelo_client, "_get_elo")
    def test_get_team_elo(self, mock_get_elo):
        mock_ce = MagicMock()
        mock_ce.read_by_date.return_value = _mock_elo_df()
        mock_get_elo.return_value = mock_ce

        elo = clubelo_client.get_team_elo("Arsenal", date(2025, 6, 15))
        self.assertAlmostEqual(elo, 2050.0)

    @patch.object(clubelo_client, "_get_elo")
    def test_get_team_elo_not_found(self, mock_get_elo):
        mock_ce = MagicMock()
        mock_ce.read_by_date.return_value = _mock_elo_df()
        mock_get_elo.return_value = mock_ce

        elo = clubelo_client.get_team_elo("Flamengo", date(2025, 6, 15))
        self.assertIsNone(elo)

    @patch.object(clubelo_client, "_get_elo")
    def test_get_team_history(self, mock_get_elo):
        mock_ce = MagicMock()
        mock_ce.read_team_history.return_value = _mock_history_df()
        mock_get_elo.return_value = mock_ce

        df = clubelo_client.get_team_history("Arsenal")
        self.assertEqual(len(df), 3)
        self.assertIn("elo", df.columns)

    @patch.object(clubelo_client, "_get_elo")
    def test_list_teams_by_league(self, mock_get_elo):
        mock_ce = MagicMock()
        mock_ce.read_by_date.return_value = _mock_elo_df()
        mock_get_elo.return_value = mock_ce

        teams = clubelo_client.list_teams_by_league("ENG-Premier League", date(2025, 6, 15))
        self.assertEqual(set(teams), {"Arsenal", "Man City"})

    @patch.object(clubelo_client, "_get_elo")
    def test_list_leagues(self, mock_get_elo):
        mock_ce = MagicMock()
        mock_ce.read_by_date.return_value = _mock_elo_df()
        mock_get_elo.return_value = mock_ce

        leagues = clubelo_client.list_leagues(date(2025, 6, 15))
        self.assertIn("ENG-Premier League", leagues)
        self.assertIn("ESP-La Liga", leagues)

    @patch.object(clubelo_client, "_get_elo")
    def test_is_covered(self, mock_get_elo):
        mock_ce = MagicMock()
        mock_ce.read_by_date.return_value = _mock_elo_df()
        mock_get_elo.return_value = mock_ce

        self.assertTrue(clubelo_client.is_covered("Arsenal", date(2025, 6, 15)))
        self.assertFalse(clubelo_client.is_covered("Boca Juniors", date(2025, 6, 15)))


if __name__ == "__main__":
    unittest.main()
