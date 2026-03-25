"""Tests for the new improvement features:
- Sofascore retry with exponential backoff
- Sofascore player age/nationality in search results
- Sofascore shared season cache (_get_current_season_id reuses get_season_list)
- ClubElo expanded _LEAGUE_NAMES
- ClubElo soccerdata retry with expiration
- Power Rankings match_type (exact/fuzzy), historical rankings, league comparison
"""

import os
import shutil
import tempfile
import time
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

_TEMP_DIR = tempfile.mkdtemp(prefix="ts_improvements_test_")
os.environ["CACHE_DIR"] = _TEMP_DIR

from backend.data import cache, clubelo_client, sofascore_client
from backend.features import power_rankings
from backend.utils.league_registry import LEAGUES


def tearDownModule():
    cache.close()
    shutil.rmtree(_TEMP_DIR, ignore_errors=True)


# ── Sofascore retry ─────────────────────────────────────────────────────────

class TestSofascoreRetry(unittest.TestCase):
    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR

    def tearDown(self):
        cache.close()

    @patch("requests.get")
    def test_retries_on_429(self, mock_get):
        """_get should retry on HTTP 429 and eventually return data."""
        resp_429 = MagicMock()
        resp_429.status_code = 429

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.json.return_value = {"data": "ok"}
        resp_ok.raise_for_status.return_value = None

        mock_get.side_effect = [resp_429, resp_ok]

        with patch("time.sleep"):  # don't actually sleep
            result = sofascore_client._get("/test")

        self.assertEqual(result, {"data": "ok"})
        self.assertEqual(mock_get.call_count, 2)

    @patch("requests.get")
    def test_retries_on_500(self, mock_get):
        """_get should retry on HTTP 500."""
        resp_500 = MagicMock()
        resp_500.status_code = 500

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.json.return_value = {"result": True}
        resp_ok.raise_for_status.return_value = None

        mock_get.side_effect = [resp_500, resp_500, resp_ok]

        with patch("time.sleep"):
            result = sofascore_client._get("/test500")

        self.assertEqual(result, {"result": True})
        self.assertEqual(mock_get.call_count, 3)

    @patch("requests.get")
    def test_gives_up_after_max_retries(self, mock_get):
        """_get returns None after exhausting retries."""
        resp_503 = MagicMock()
        resp_503.status_code = 503

        mock_get.return_value = resp_503

        with patch("time.sleep"):
            result = sofascore_client._get("/always_fail")

        self.assertIsNone(result)
        self.assertEqual(mock_get.call_count, sofascore_client._MAX_RETRIES)

    @patch("requests.get")
    def test_no_retry_on_404(self, mock_get):
        """_get should NOT retry on 404 (non-retryable)."""
        import requests as req
        resp_404 = MagicMock()
        resp_404.status_code = 404
        resp_404.raise_for_status.side_effect = req.exceptions.HTTPError("404")

        mock_get.return_value = resp_404

        result = sofascore_client._get("/not_found")
        self.assertIsNone(result)
        mock_get.assert_called_once()


# ── Sofascore player age/nationality ────────────────────────────────────────

MOCK_SEARCH_WITH_METADATA = {
    "results": [
        {
            "entity": {
                "id": 961995,
                "name": "Bukayo Saka",
                "dateOfBirthTimestamp": 999820800,  # 2001-09-05
                "country": {"name": "England"},
                "team": {
                    "id": 789,
                    "name": "Arsenal",
                    "tournament": {"id": 17},
                },
            }
        }
    ]
}


class TestSofascorePlayerMetadata(unittest.TestCase):
    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR
        cache.clear_namespace("sofascore_search")
        cache.clear_namespace("sofascore_player_meta")

    def tearDown(self):
        cache.close()

    @patch.object(sofascore_client, "_get")
    def test_search_includes_age(self, mock_get):
        mock_get.return_value = MOCK_SEARCH_WITH_METADATA
        results = sofascore_client.search_player("Saka metadata test")
        self.assertIn("age", results[0])
        self.assertIsInstance(results[0]["age"], int)
        # Saka born 2001-09-05, should be 24 or 25 depending on current date
        self.assertGreaterEqual(results[0]["age"], 23)
        self.assertLessEqual(results[0]["age"], 30)

    @patch.object(sofascore_client, "_get")
    def test_search_includes_nationality(self, mock_get):
        mock_get.return_value = MOCK_SEARCH_WITH_METADATA
        results = sofascore_client.search_player("Saka nationality test")
        self.assertEqual(results[0]["nationality"], "England")

    @patch.object(sofascore_client, "_get")
    def test_search_includes_team_name(self, mock_get):
        mock_get.return_value = MOCK_SEARCH_WITH_METADATA
        results = sofascore_client.search_player("Saka team test")
        self.assertEqual(results[0]["team_name"], "Arsenal")


# ── Sofascore shared season cache ───────────────────────────────────────────

MOCK_SEASONS_RESPONSE = {
    "seasons": [
        {"id": 61627, "name": "2024/2025"},
        {"id": 52186, "name": "2023/2024"},
    ]
}


class TestSofascoreSharedSeasonCache(unittest.TestCase):
    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR
        cache.clear_namespace("sofascore_season_list")
        cache.clear_namespace("sofascore_seasons")

    def tearDown(self):
        cache.close()

    @patch.object(sofascore_client, "_get")
    def test_get_current_season_reuses_season_list_cache(self, mock_get):
        """_get_current_season_id should use get_season_list cache."""
        mock_get.return_value = MOCK_SEASONS_RESPONSE

        # First call populates season_list cache
        seasons = sofascore_client.get_season_list(17)
        self.assertEqual(len(seasons), 2)

        # Second call should hit cache — no additional API call
        sid = sofascore_client._get_current_season_id(17)
        self.assertEqual(sid, 61627)

        # Should only have called _get once (for the season_list)
        mock_get.assert_called_once()


# ── ClubElo expanded leagues ────────────────────────────────────────────────

class TestClubEloExpandedLeagues(unittest.TestCase):
    def test_league_names_has_additional_countries(self):
        """_LEAGUE_NAMES should have at least 35 countries."""
        self.assertGreaterEqual(len(clubelo_client._LEAGUE_NAMES), 35)

    def test_new_countries_present(self):
        """Newly added countries should be in _LEAGUE_NAMES."""
        new_countries = ["BUL", "HUN", "SVK", "SLO", "CYP", "FIN", "GEO", "ISR"]
        for country in new_countries:
            self.assertIn(country, clubelo_client._LEAGUE_NAMES,
                          f"Missing country: {country}")

    def test_second_divisions_present(self):
        """Second divisions should be in _LEAGUE_NAMES."""
        self.assertIn(2, clubelo_client._LEAGUE_NAMES["ESP"])
        self.assertIn(2, clubelo_client._LEAGUE_NAMES["GER"])
        self.assertIn(2, clubelo_client._LEAGUE_NAMES["ITA"])
        self.assertIn(2, clubelo_client._LEAGUE_NAMES["FRA"])


# ── ClubElo soccerdata retry with expiration ────────────────────────────────

class TestSoccerdataRetryExpiration(unittest.TestCase):
    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR
        cache.clear_namespace("clubelo_date")
        # Reset state
        clubelo_client._soccerdata_available = None
        clubelo_client._soccerdata_disabled_at = 0.0

    def tearDown(self):
        cache.close()
        clubelo_client._soccerdata_available = None
        clubelo_client._soccerdata_disabled_at = 0.0

    def test_retry_after_interval(self):
        """soccerdata should be retried after the interval expires."""
        clubelo_client._soccerdata_available = False
        # Set disabled time to 2 hours ago (longer than 1-hour interval)
        clubelo_client._soccerdata_disabled_at = time.time() - 7200

        # It should attempt soccerdata again (reset _soccerdata_available)
        with patch("soccerdata.ClubElo") as mock_ce_class:
            mock_ce = MagicMock()
            mock_ce.read_by_date.side_effect = Exception("still broken")
            mock_ce_class.return_value = mock_ce

            result = clubelo_client._try_soccerdata("2025-01-01")
            self.assertIsNone(result)
            # Should have tried (not skipped)
            mock_ce.read_by_date.assert_called_once()

    def test_no_retry_within_interval(self):
        """soccerdata should NOT be retried within the interval."""
        clubelo_client._soccerdata_available = False
        clubelo_client._soccerdata_disabled_at = time.time()  # just now

        result = clubelo_client._try_soccerdata("2025-01-01")
        self.assertIsNone(result)
        # Should have been skipped — no import of soccerdata


# ── League registry expanded ────────────────────────────────────────────────

class TestLeagueRegistryExpanded(unittest.TestCase):
    def test_new_european_leagues_in_registry(self):
        new_codes = [
            "SCO1", "AUT1", "SUI1", "GRE1", "CZE1", "DEN1",
            "CRO1", "SER1", "NOR1", "SWE1", "POL1", "ROM1",
            "UKR1", "RUS1", "BUL1", "HUN1", "CYP1", "FIN1",
            "ESP2", "GER2", "ITA2", "FRA2",
        ]
        for code in new_codes:
            self.assertIn(code, LEAGUES, f"Missing league code: {code}")
            info = LEAGUES[code]
            self.assertIsNotNone(info.clubelo_league,
                                 f"{code} should have clubelo_league set")

    def test_new_leagues_have_sofascore_ids(self):
        """All new leagues should have sofascore_tournament_id."""
        for code in ["SCO1", "AUT1", "SUI1", "ESP2", "GER2"]:
            info = LEAGUES[code]
            self.assertIsNotNone(info.sofascore_tournament_id,
                                 f"{code} should have sofascore_tournament_id")


# ── Power Rankings match_type ───────────────────────────────────────────────

class TestPowerRankingsMatchType(unittest.TestCase):
    def test_team_ranking_has_match_type_field(self):
        """TeamRanking dataclass should have match_type field."""
        from backend.features.power_rankings import TeamRanking
        r = TeamRanking(
            team_name="Arsenal",
            league_code="ENG1",
            raw_elo=2000,
            normalized_score=90,
            league_mean_normalized=70,
            relative_ability=20,
        )
        self.assertEqual(r.match_type, "exact")  # default
        r.match_type = "fuzzy"
        self.assertEqual(r.match_type, "fuzzy")


# ── Power Rankings compare_leagues ──────────────────────────────────────────

class TestComparLeagues(unittest.TestCase):
    def setUp(self):
        cache.close()
        os.environ["CACHE_DIR"] = _TEMP_DIR
        cache.clear_namespace("power_rankings")

    def tearDown(self):
        cache.close()

    @patch.object(power_rankings, "compute_daily_rankings")
    def test_compare_leagues_returns_sorted(self, mock_compute):
        from backend.features.power_rankings import LeagueSnapshot
        snap_eng = LeagueSnapshot(
            league_code="ENG1", league_name="Premier League",
            date=date.today(), mean_elo=1900, std_elo=80,
            p10=50, p25=60, p50=70, p75=80, p90=90,
            mean_normalized=75.0, team_count=20,
        )
        snap_esp = LeagueSnapshot(
            league_code="ESP1", league_name="La Liga",
            date=date.today(), mean_elo=1850, std_elo=90,
            p10=40, p25=50, p50=65, p75=75, p90=85,
            mean_normalized=70.0, team_count=20,
        )
        mock_compute.return_value = ({}, {"ENG1": snap_eng, "ESP1": snap_esp})

        result = power_rankings.compare_leagues(["ESP1", "ENG1"])
        self.assertEqual(len(result), 2)
        # Should be sorted by mean_normalized descending
        self.assertEqual(result[0]["code"], "ENG1")
        self.assertEqual(result[1]["code"], "ESP1")
        self.assertEqual(result[0]["mean_normalized"], 75.0)

    @patch.object(power_rankings, "compute_daily_rankings")
    def test_compare_leagues_unknown_code(self, mock_compute):
        mock_compute.return_value = ({}, {})
        result = power_rankings.compare_leagues(["UNKNOWN1"])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
