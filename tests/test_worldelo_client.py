"""Unit tests for backend.data.worldfootballelo_client using mock responses."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

_TEMP_DIR = tempfile.mkdtemp(prefix="ts_worldelo_test_")
os.environ["CACHE_DIR"] = _TEMP_DIR

from backend.data import cache, worldfootballelo_client


MOCK_TEAM_HTML = """
<html><body>
<h1>Flamengo</h1>
<table>
<tr><td>Rating: 1834</td></tr>
</table>
</body></html>
"""

MOCK_LEAGUE_HTML = """
<html><body>
<table>
<tr><td><a href="/Flamengo">Flamengo</a></td><td>1834</td></tr>
<tr><td><a href="/Palmeiras">Palmeiras</a></td><td>1810</td></tr>
<tr><td><a href="/Atletico_Mineiro">Atlético Mineiro</a></td><td>1756</td></tr>
</table>
</body></html>
"""


class TestWorldEloClient(unittest.TestCase):
    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR
        cache.clear_namespace("worldelo_team")
        cache.clear_namespace("worldelo_league")

    def tearDown(self):
        cache.close()

    @classmethod
    def tearDownClass(cls):
        cache.close()
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)

    def test_team_slug(self):
        self.assertEqual(worldfootballelo_client._team_slug("Boca Juniors"), "Boca_Juniors")
        self.assertEqual(worldfootballelo_client._team_slug("  River Plate  "), "River_Plate")

    def test_parse_elo_from_html_rating_pattern(self):
        elo = worldfootballelo_client._parse_elo_from_html(MOCK_TEAM_HTML)
        self.assertAlmostEqual(elo, 1834.0)

    def test_parse_elo_from_html_td_pattern(self):
        html = '<table><tr><td class="r">1920</td></tr></table>'
        elo = worldfootballelo_client._parse_elo_from_html(html)
        self.assertAlmostEqual(elo, 1920.0)

    def test_parse_elo_from_html_no_match(self):
        elo = worldfootballelo_client._parse_elo_from_html("<html><body>No data</body></html>")
        self.assertIsNone(elo)

    @patch("backend.data.worldfootballelo_client.requests.get")
    def test_get_team_elo(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = MOCK_TEAM_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        elo = worldfootballelo_client.get_team_elo("Flamengo")
        self.assertAlmostEqual(elo, 1834.0)

    @patch("backend.data.worldfootballelo_client.requests.get")
    def test_get_team_elo_cached(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = MOCK_TEAM_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        worldfootballelo_client.get_team_elo("Flamengo cache test")
        worldfootballelo_client.get_team_elo("Flamengo cache test")
        # URL slug differs so test with exact slug
        self.assertEqual(mock_get.call_count, 1)

    @patch("backend.data.worldfootballelo_client.requests.get")
    def test_get_team_elo_network_error(self, mock_get):
        import requests as req
        mock_get.side_effect = req.RequestException("timeout")

        elo = worldfootballelo_client.get_team_elo("NetworkFail")
        self.assertIsNone(elo)

    @patch("backend.data.worldfootballelo_client.requests.get")
    def test_get_league_teams(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = MOCK_LEAGUE_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        teams = worldfootballelo_client.get_league_teams("Brazil")
        self.assertEqual(len(teams), 3)
        self.assertEqual(teams[0]["name"], "Flamengo")
        self.assertAlmostEqual(teams[0]["elo"], 1834.0)
        self.assertEqual(teams[1]["name"], "Palmeiras")

    @patch("backend.data.worldfootballelo_client.requests.get")
    def test_is_covered(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = MOCK_TEAM_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        self.assertTrue(worldfootballelo_client.is_covered("Flamengo"))


if __name__ == "__main__":
    unittest.main()
