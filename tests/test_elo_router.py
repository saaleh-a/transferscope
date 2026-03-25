"""Unit tests for backend.data.elo_router."""

import os
import shutil
import tempfile
import unittest
from datetime import date
from unittest.mock import patch

_TEMP_DIR = tempfile.mkdtemp(prefix="ts_router_test_")
os.environ["CACHE_DIR"] = _TEMP_DIR

from backend.data import cache, elo_router


class TestEloRouter(unittest.TestCase):
    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR

    def tearDown(self):
        cache.close()

    @classmethod
    def tearDownClass(cls):
        cache.close()
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)

    @patch("backend.data.clubelo_client.get_team_elo", return_value=2050.0)
    def test_european_club_uses_clubelo(self, mock_ce):
        elo = elo_router.get_team_elo("Arsenal")
        self.assertAlmostEqual(elo, 2050.0)
        mock_ce.assert_called_once()

    @patch("backend.data.clubelo_client.get_team_elo", return_value=None)
    @patch("backend.data.worldfootballelo_client.get_team_elo", return_value=1834.0)
    def test_non_european_uses_worldelo(self, mock_we, mock_ce):
        elo = elo_router.get_team_elo("Flamengo")
        self.assertAlmostEqual(elo, 1834.0)
        mock_ce.assert_called_once()
        mock_we.assert_called_once()

    @patch("backend.data.clubelo_client.get_team_elo", return_value=None)
    @patch("backend.data.worldfootballelo_client.get_team_elo", return_value=None)
    def test_unknown_club_returns_none(self, mock_we, mock_ce):
        elo = elo_router.get_team_elo("Unknown FC")
        self.assertIsNone(elo)

    @patch("backend.data.clubelo_client.get_team_elo", return_value=2050.0)
    def test_with_source_european(self, mock_ce):
        elo, source = elo_router.get_team_elo_with_source("Arsenal")
        self.assertEqual(source, "clubelo")
        self.assertAlmostEqual(elo, 2050.0)

    @patch("backend.data.clubelo_client.get_team_elo", return_value=None)
    @patch("backend.data.worldfootballelo_client.get_team_elo", return_value=1834.0)
    def test_with_source_worldelo(self, mock_we, mock_ce):
        elo, source = elo_router.get_team_elo_with_source("Flamengo")
        self.assertEqual(source, "worldelo")

    @patch("backend.data.clubelo_client.get_team_elo", return_value=None)
    @patch("backend.data.worldfootballelo_client.get_team_elo", return_value=None)
    def test_with_source_none(self, mock_we, mock_ce):
        elo, source = elo_router.get_team_elo_with_source("Unknown")
        self.assertIsNone(elo)
        self.assertIsNone(source)

    def test_normalize_elo(self):
        self.assertAlmostEqual(elo_router.normalize_elo(1500, 1000, 2000), 50.0)
        self.assertAlmostEqual(elo_router.normalize_elo(2000, 1000, 2000), 100.0)
        self.assertAlmostEqual(elo_router.normalize_elo(1000, 1000, 2000), 0.0)

    def test_normalize_elo_equal_bounds(self):
        self.assertAlmostEqual(elo_router.normalize_elo(1500, 1500, 1500), 50.0)

    @patch("backend.data.clubelo_client.is_covered", return_value=True)
    def test_is_covered_european(self, mock_ce):
        self.assertTrue(elo_router.is_covered("Arsenal"))

    @patch("backend.data.clubelo_client.is_covered", return_value=False)
    @patch("backend.data.worldfootballelo_client.is_covered", return_value=True)
    def test_is_covered_worldelo(self, mock_we, mock_ce):
        self.assertTrue(elo_router.is_covered("Flamengo"))


if __name__ == "__main__":
    unittest.main()
