"""Unit tests for backend.data.clubelo_client.

Tests both the soccerdata primary path and the direct HTTP fallback path.
"""

import os
import shutil
import tempfile
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

_TEMP_DIR = tempfile.mkdtemp(prefix="ts_clubelo_test_")
os.environ["CACHE_DIR"] = _TEMP_DIR

from backend.data import cache, clubelo_client


# ── Mock CSV data matching the real ClubElo API format ───────────────────────

_MOCK_CSV = (
    "Rank,Club,Country,Level,Elo,From,To\n"
    "1,Arsenal,ENG,1,2050.0,2025-01-01,2025-12-31\n"
    "2,Man City,ENG,1,1980.0,2025-01-01,2025-12-31\n"
    "3,Real Madrid,ESP,1,2010.0,2025-01-01,2025-12-31\n"
    "4,Barcelona,ESP,1,1890.0,2025-01-01,2025-12-31\n"
    "5,Bayern Munich,GER,1,1970.0,2025-01-01,2025-12-31\n"
)

_MOCK_HISTORY_CSV = (
    "Rank,Club,Country,Level,Elo,From,To\n"
    "1,Arsenal,ENG,1,2000.0,2025-01-01,2025-06-01\n"
    "2,Arsenal,ENG,1,2020.0,2025-06-01,2025-09-01\n"
    "3,Arsenal,ENG,1,2050.0,2025-09-01,2025-12-31\n"
)


def _make_soccerdata_df():
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
    df = pd.DataFrame(
        data,
        index=["Arsenal", "Man City", "Real Madrid", "Barcelona", "Bayern Munich"],
    )
    df.index.name = "team"
    return df


class TestClubEloClient(unittest.TestCase):
    """Test with soccerdata primary path (mocked)."""

    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR
        cache.clear_namespace("clubelo_date")
        cache.clear_namespace("clubelo_history")
        # Reset the availability probe so each test is independent
        clubelo_client._soccerdata_available = None

    def tearDown(self):
        cache.close()

    @classmethod
    def tearDownClass(cls):
        cache.close()
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)

    # ── soccerdata primary path ──────────────────────────────────────────

    @patch.object(clubelo_client, "_try_soccerdata", return_value=_make_soccerdata_df())
    def test_get_all_by_date(self, mock_sd):
        df = clubelo_client.get_all_by_date(date(2025, 6, 15))
        self.assertEqual(len(df), 5)
        self.assertIn("Arsenal", df.index)
        self.assertAlmostEqual(df.loc["Arsenal", "elo"], 2050.0)

    @patch.object(clubelo_client, "_try_soccerdata", return_value=_make_soccerdata_df())
    def test_get_all_by_date_cached(self, mock_sd):
        clubelo_client.get_all_by_date(date(2025, 7, 1))
        clubelo_client.get_all_by_date(date(2025, 7, 1))
        mock_sd.assert_called_once()

    @patch.object(clubelo_client, "_try_soccerdata", return_value=_make_soccerdata_df())
    def test_get_team_elo(self, mock_sd):
        elo = clubelo_client.get_team_elo("Arsenal", date(2025, 6, 15))
        self.assertAlmostEqual(elo, 2050.0)

    @patch.object(clubelo_client, "_try_soccerdata", return_value=_make_soccerdata_df())
    def test_get_team_elo_not_found(self, mock_sd):
        elo = clubelo_client.get_team_elo("Flamengo", date(2025, 6, 15))
        self.assertIsNone(elo)

    @patch.object(clubelo_client, "_try_soccerdata_history")
    def test_get_team_history(self, mock_hist):
        hist_df = pd.DataFrame({
            "rank": [1, 2, 3],
            "team": ["Arsenal"] * 3,
            "country": ["ENG"] * 3,
            "level": [1, 1, 1],
            "elo": [2000.0, 2020.0, 2050.0],
            "to": ["2025-06-01", "2025-09-01", "2025-12-31"],
        })
        hist_df.index = pd.to_datetime(
            ["2025-01-01", "2025-06-01", "2025-09-01"]
        )
        hist_df.index.name = "from"
        mock_hist.return_value = hist_df

        df = clubelo_client.get_team_history("Arsenal")
        self.assertEqual(len(df), 3)
        self.assertIn("elo", df.columns)

    @patch.object(clubelo_client, "_try_soccerdata", return_value=_make_soccerdata_df())
    def test_list_teams_by_league(self, mock_sd):
        teams = clubelo_client.list_teams_by_league(
            "ENG-Premier League", date(2025, 6, 15)
        )
        self.assertEqual(set(teams), {"Arsenal", "Man City"})

    @patch.object(clubelo_client, "_try_soccerdata", return_value=_make_soccerdata_df())
    def test_list_leagues(self, mock_sd):
        leagues = clubelo_client.list_leagues(date(2025, 6, 15))
        self.assertIn("ENG-Premier League", leagues)
        self.assertIn("ESP-La Liga", leagues)

    @patch.object(clubelo_client, "_try_soccerdata", return_value=_make_soccerdata_df())
    def test_is_covered(self, mock_sd):
        self.assertTrue(
            clubelo_client.is_covered("Arsenal", date(2025, 6, 15))
        )
        self.assertFalse(
            clubelo_client.is_covered("Boca Juniors", date(2025, 6, 15))
        )

    # ── HTTP fallback path ───────────────────────────────────────────────

    @patch.object(clubelo_client, "_try_soccerdata", return_value=None)
    @patch.object(clubelo_client, "_fetch_csv", return_value=_MOCK_CSV)
    def test_http_fallback_get_all(self, mock_csv, mock_sd):
        df = clubelo_client.get_all_by_date(date(2025, 8, 1))
        self.assertEqual(len(df), 5)
        self.assertIn("Arsenal", df.index)
        self.assertAlmostEqual(df.loc["Arsenal", "elo"], 2050.0)

    @patch.object(clubelo_client, "_try_soccerdata", return_value=None)
    @patch.object(clubelo_client, "_fetch_csv", return_value=_MOCK_CSV)
    def test_http_fallback_leagues(self, mock_csv, mock_sd):
        leagues = clubelo_client.list_leagues(date(2025, 8, 1))
        self.assertIn("ENG-Premier League", leagues)

    @patch.object(clubelo_client, "_try_soccerdata_history", return_value=None)
    @patch("requests.get")
    def test_http_fallback_history(self, mock_get, mock_sd_hist):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _MOCK_HISTORY_CSV
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        df = clubelo_client.get_team_history("Arsenal")
        self.assertEqual(len(df), 3)
        self.assertIn("elo", df.columns)


if __name__ == "__main__":
    unittest.main()
