"""Tests for backend.data.opta_client and Opta integration in power_rankings."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

from backend.data.opta_client import (
    OptaLeagueRanking,
    OptaTeamRanking,
    _parse_float,
    _parse_int,
)


# ── Helper factories ──────────────────────────────────────────────────────────

def _make_opta_team(
    rank: int = 1,
    team: str = "Manchester City",
    rating: float = 95.2,
    opta_id: str = "abc123",
    change: str = "+2",
) -> OptaTeamRanking:
    return OptaTeamRanking(
        rank=rank,
        team=team,
        rating=rating,
        ranking_change_7d=change,
        opta_id=opta_id,
    )


def _make_opta_league(
    rank: int = 1,
    league: str = "Premier League",
    rating: float = 88.0,
    change: str = "",
) -> OptaLeagueRanking:
    return OptaLeagueRanking(
        rank=rank,
        league=league,
        rating=rating,
        ranking_change_7d=change,
    )


# ── Unit tests: parse helpers ─────────────────────────────────────────────────

class TestParseHelpers(unittest.TestCase):
    """Test the _parse_float and _parse_int utility functions."""

    def test_parse_float_normal(self):
        self.assertAlmostEqual(_parse_float("95.2"), 95.2)

    def test_parse_float_with_comma(self):
        self.assertAlmostEqual(_parse_float("1,234.5"), 1234.5)

    def test_parse_float_empty(self):
        self.assertAlmostEqual(_parse_float(""), 0.0)

    def test_parse_float_invalid(self):
        self.assertAlmostEqual(_parse_float("N/A"), 0.0)

    def test_parse_float_whitespace(self):
        self.assertAlmostEqual(_parse_float("  42.0  "), 42.0)

    def test_parse_int_normal(self):
        self.assertEqual(_parse_int("42"), 42)

    def test_parse_int_with_hash(self):
        self.assertEqual(_parse_int("#42"), 42)

    def test_parse_int_with_comma(self):
        self.assertEqual(_parse_int("1,234"), 1234)

    def test_parse_int_empty(self):
        self.assertEqual(_parse_int(""), 0)

    def test_parse_int_invalid(self):
        self.assertEqual(_parse_int("abc"), 0)


# ── Unit tests: dataclass construction ────────────────────────────────────────

class TestOptaDataclasses(unittest.TestCase):
    """Test OptaTeamRanking and OptaLeagueRanking dataclasses."""

    def test_team_ranking_fields(self):
        t = _make_opta_team()
        self.assertEqual(t.rank, 1)
        self.assertEqual(t.team, "Manchester City")
        self.assertAlmostEqual(t.rating, 95.2)
        self.assertEqual(t.opta_id, "abc123")

    def test_league_ranking_fields(self):
        l = _make_opta_league()
        self.assertEqual(l.rank, 1)
        self.assertEqual(l.league, "Premier League")
        self.assertAlmostEqual(l.rating, 88.0)


# ── Unit tests: caching layer ─────────────────────────────────────────────────

class TestOptaCaching(unittest.TestCase):
    """Test that get_team_rankings / get_league_rankings use the cache."""

    @patch("backend.data.opta_client._scrape_team_rankings")
    @patch("backend.data.opta_client.cache")
    def test_team_rankings_cache_hit(self, mock_cache, mock_scrape):
        """When cache has data, scraper should not be called."""
        from backend.data.opta_client import get_team_rankings

        mock_cache.make_key.return_value = "opta_team_rankings:2026-04-02"
        mock_cache.get.return_value = [_make_opta_team()]

        result = get_team_rankings()
        self.assertEqual(len(result), 1)
        mock_scrape.assert_not_called()

    @patch("backend.data.opta_client._scrape_team_rankings")
    @patch("backend.data.opta_client.cache")
    def test_team_rankings_cache_miss(self, mock_cache, mock_scrape):
        """When cache is empty, scraper should be called and result cached."""
        from backend.data.opta_client import get_team_rankings

        mock_cache.make_key.return_value = "opta_team_rankings:2026-04-02"
        mock_cache.get.return_value = None
        mock_scrape.return_value = [_make_opta_team()]

        result = get_team_rankings()
        self.assertEqual(len(result), 1)
        mock_scrape.assert_called_once()
        mock_cache.set.assert_called_once()

    @patch("backend.data.opta_client._scrape_team_rankings")
    @patch("backend.data.opta_client.cache")
    def test_force_refresh_bypasses_cache(self, mock_cache, mock_scrape):
        """force_refresh=True should bypass cache."""
        from backend.data.opta_client import get_team_rankings

        mock_cache.make_key.return_value = "opta_team_rankings:2026-04-02"
        mock_scrape.return_value = [_make_opta_team()]

        result = get_team_rankings(force_refresh=True)
        self.assertEqual(len(result), 1)
        # cache.get should NOT have been called
        mock_cache.get.assert_not_called()

    @patch("backend.data.opta_client._scrape_league_rankings")
    @patch("backend.data.opta_client.cache")
    def test_league_rankings_cache_hit(self, mock_cache, mock_scrape):
        from backend.data.opta_client import get_league_rankings

        mock_cache.make_key.return_value = "opta_league_rankings:2026-04-02"
        mock_cache.get.return_value = [_make_opta_league()]

        result = get_league_rankings()
        self.assertEqual(len(result), 1)
        mock_scrape.assert_not_called()


# ── Unit tests: dict helpers ──────────────────────────────────────────────────

class TestOptaDictHelpers(unittest.TestCase):
    """Test get_team_rankings_dict / get_league_rankings_dict."""

    @patch("backend.data.opta_client.get_team_rankings")
    def test_team_dict_keyed_by_name(self, mock_get):
        from backend.data.opta_client import get_team_rankings_dict

        mock_get.return_value = [
            _make_opta_team(rank=1, team="Manchester City", rating=95.2),
            _make_opta_team(rank=2, team="Arsenal", rating=91.0),
        ]
        d = get_team_rankings_dict()
        self.assertIn("Manchester City", d)
        self.assertIn("Arsenal", d)
        self.assertEqual(d["Arsenal"].rank, 2)

    @patch("backend.data.opta_client.get_league_rankings")
    def test_league_dict_keyed_by_name(self, mock_get):
        from backend.data.opta_client import get_league_rankings_dict

        mock_get.return_value = [
            _make_opta_league(rank=1, league="Premier League", rating=88.0),
            _make_opta_league(rank=2, league="La Liga", rating=82.0),
        ]
        d = get_league_rankings_dict()
        self.assertIn("Premier League", d)
        self.assertIn("La Liga", d)


# ── Integration tests: _opta_score_to_raw_elo ─────────────────────────────────

class TestOptaRescale(unittest.TestCase):
    """Test the linear rescale from Opta 0-100 to ClubElo ~1000-2100."""

    def test_rescale_zero(self):
        from backend.features.power_rankings import _opta_score_to_raw_elo

        self.assertAlmostEqual(_opta_score_to_raw_elo(0.0), 1000.0)

    def test_rescale_hundred(self):
        from backend.features.power_rankings import _opta_score_to_raw_elo

        self.assertAlmostEqual(_opta_score_to_raw_elo(100.0), 2100.0)

    def test_rescale_fifty(self):
        from backend.features.power_rankings import _opta_score_to_raw_elo

        self.assertAlmostEqual(_opta_score_to_raw_elo(50.0), 1550.0)

    def test_rescale_typical_value(self):
        from backend.features.power_rankings import _opta_score_to_raw_elo

        # 87.3 -> 87.3/100 * 1100 + 1000 = 960.3 + 1000 = 1960.3
        self.assertAlmostEqual(_opta_score_to_raw_elo(87.3), 1960.3)


# ── Integration tests: _compute_rankings_from_opta ────────────────────────────

class TestComputeRankingsFromOpta(unittest.TestCase):
    """Test _compute_rankings_from_opta with mocked opta_client and clubelo."""

    def _mock_opta_teams(self) -> List[OptaTeamRanking]:
        return [
            _make_opta_team(rank=1, team="Manchester City", rating=95.0, opta_id="mc1"),
            _make_opta_team(rank=2, team="Arsenal", rating=91.0, opta_id="ars1"),
            _make_opta_team(rank=28, team="Fulham", rating=65.0, opta_id="ful1"),
            _make_opta_team(rank=193, team="Hull City", rating=30.0, opta_id="hul1"),
        ]

    def _mock_opta_leagues(self) -> List[OptaLeagueRanking]:
        return [
            _make_opta_league(rank=1, league="Premier League", rating=88.0),
        ]

    @patch("backend.features.power_rankings.clubelo_client")
    @patch("backend.features.power_rankings._get_clubelo_sofascore_map")
    @patch("backend.data.opta_client.get_team_rankings")
    @patch("backend.data.opta_client.get_league_rankings")
    def test_opta_builds_team_rankings(
        self, mock_opta_leagues, mock_opta_teams, mock_ce_map, mock_ce_client
    ):
        from backend.features.power_rankings import _compute_rankings_from_opta

        mock_opta_teams.return_value = self._mock_opta_teams()
        mock_opta_leagues.return_value = self._mock_opta_leagues()
        mock_ce_map.return_value = {}
        mock_ce_client.get_all_by_date.return_value = None

        result = _compute_rankings_from_opta()
        self.assertIsNotNone(result)

        team_rankings, league_snapshots = result
        self.assertEqual(len(team_rankings), 4)
        self.assertIn("Manchester City", team_rankings)
        self.assertIn("Hull City", team_rankings)

        mc = team_rankings["Manchester City"]
        self.assertAlmostEqual(mc.normalized_score, 95.0)
        # Without ClubElo, raw_elo should be rescaled from Opta
        expected_raw = 95.0 / 100.0 * 1100.0 + 1000.0
        self.assertAlmostEqual(mc.raw_elo, expected_raw)

    @patch("backend.features.power_rankings.clubelo_client")
    @patch("backend.features.power_rankings._get_clubelo_sofascore_map")
    @patch("backend.data.opta_client.get_team_rankings")
    @patch("backend.data.opta_client.get_league_rankings")
    def test_opta_uses_clubelo_raw_elo_when_available(
        self, mock_opta_leagues, mock_opta_teams, mock_ce_map, mock_ce_client
    ):
        """When ClubElo has a team, raw_elo should come from ClubElo, not rescale."""
        import pandas as pd
        from backend.features.power_rankings import _compute_rankings_from_opta

        mock_opta_teams.return_value = self._mock_opta_teams()
        mock_opta_leagues.return_value = self._mock_opta_leagues()

        # ClubElo covers Manchester City with raw elo 2050
        mock_ce_map.return_value = {"ManCity": "Manchester City"}
        ce_df = pd.DataFrame(
            {"elo": [2050.0], "league": ["ENG_1"]},
            index=["ManCity"],
        )
        mock_ce_client.get_all_by_date.return_value = ce_df

        result = _compute_rankings_from_opta()
        self.assertIsNotNone(result)

        team_rankings, _ = result
        mc = team_rankings["Manchester City"]
        # normalized_score should still be from Opta
        self.assertAlmostEqual(mc.normalized_score, 95.0)
        # raw_elo should be from ClubElo
        self.assertAlmostEqual(mc.raw_elo, 2050.0)

        # Hull City is NOT in ClubElo — raw_elo should be rescaled
        hull = team_rankings["Hull City"]
        expected_raw = 30.0 / 100.0 * 1100.0 + 1000.0
        self.assertAlmostEqual(hull.raw_elo, expected_raw)

    @patch("backend.data.opta_client.get_team_rankings")
    @patch("backend.data.opta_client.get_league_rankings")
    def test_opta_returns_none_when_empty(self, mock_opta_leagues, mock_opta_teams):
        from backend.features.power_rankings import _compute_rankings_from_opta

        mock_opta_teams.return_value = []
        mock_opta_leagues.return_value = []

        result = _compute_rankings_from_opta()
        self.assertIsNone(result)


# ── Integration tests: compute_daily_rankings Opta path ───────────────────────

class TestComputeDailyRankingsOptaPath(unittest.TestCase):
    """Test that compute_daily_rankings tries Opta for today's date."""

    @patch("backend.features.power_rankings._compute_rankings_from_opta")
    @patch("backend.features.power_rankings.cache")
    def test_today_uses_opta_when_available(self, mock_cache, mock_opta_fn):
        from backend.features.power_rankings import (
            LeagueSnapshot,
            TeamRanking,
            compute_daily_rankings,
        )

        mock_cache.make_key.return_value = "power_rankings:today"
        mock_cache.get.return_value = None

        fake_team = TeamRanking(
            team_name="Test FC",
            league_code="TST",
            raw_elo=1800.0,
            normalized_score=80.0,
            league_mean_normalized=60.0,
            relative_ability=20.0,
        )
        fake_league = LeagueSnapshot(
            league_code="TST",
            league_name="Test League",
            date=date.today(),
            mean_elo=1600.0,
            std_elo=100.0,
            p10=40.0,
            p25=50.0,
            p50=60.0,
            p75=70.0,
            p90=80.0,
            mean_normalized=60.0,
            team_count=1,
        )
        mock_opta_fn.return_value = ({"Test FC": fake_team}, {"TST": fake_league})

        result = compute_daily_rankings(date.today())
        teams, leagues = result
        self.assertIn("Test FC", teams)
        self.assertAlmostEqual(teams["Test FC"].normalized_score, 80.0)
        mock_opta_fn.assert_called_once()

    @patch("backend.features.power_rankings._compute_rankings_from_opta")
    @patch("backend.features.power_rankings.clubelo_client")
    @patch("backend.features.power_rankings.worldfootballelo_client")
    @patch("backend.features.power_rankings.cache")
    def test_today_falls_back_to_clubelo_when_opta_empty(
        self, mock_cache, mock_welo, mock_ce, mock_opta_fn
    ):
        """If Opta returns None, should proceed to ClubElo path."""
        from backend.features.power_rankings import compute_daily_rankings

        mock_cache.make_key.return_value = "power_rankings:today"
        mock_cache.get.return_value = None
        mock_opta_fn.return_value = None  # Opta failed

        # ClubElo also returns nothing (for simplicity)
        mock_ce.get_all_by_date.return_value = None
        mock_welo.get_league_teams.return_value = []

        teams, leagues = compute_daily_rankings(date.today())
        # With no data from either source, should be empty
        self.assertEqual(len(teams), 0)

    @patch("backend.features.power_rankings._compute_rankings_from_opta")
    @patch("backend.features.power_rankings.clubelo_client")
    @patch("backend.features.power_rankings.worldfootballelo_client")
    @patch("backend.features.power_rankings.cache")
    def test_historical_date_skips_opta(
        self, mock_cache, mock_welo, mock_ce, mock_opta_fn
    ):
        """Historical dates should NOT try Opta."""
        from backend.features.power_rankings import compute_daily_rankings

        mock_cache.make_key.return_value = "power_rankings:2023-01-01"
        mock_cache.get.return_value = None
        mock_ce.get_all_by_date.return_value = None
        mock_welo.get_league_teams.return_value = []

        compute_daily_rankings(date(2023, 1, 1))
        mock_opta_fn.assert_not_called()


# ── Verify USE_OPTA_FOR_INFERENCE toggle ──────────────────────────────────────

class TestOptaToggle(unittest.TestCase):
    """Test that _USE_OPTA_FOR_INFERENCE can be toggled."""

    @patch("backend.features.power_rankings._compute_rankings_from_opta")
    @patch("backend.features.power_rankings.clubelo_client")
    @patch("backend.features.power_rankings.worldfootballelo_client")
    @patch("backend.features.power_rankings.cache")
    def test_toggle_off_skips_opta(
        self, mock_cache, mock_welo, mock_ce, mock_opta_fn
    ):
        import backend.features.power_rankings as pr

        original = pr._USE_OPTA_FOR_INFERENCE
        try:
            pr._USE_OPTA_FOR_INFERENCE = False

            mock_cache.make_key.return_value = "power_rankings:today"
            mock_cache.get.return_value = None
            mock_ce.get_all_by_date.return_value = None
            mock_welo.get_league_teams.return_value = []

            pr.compute_daily_rankings(date.today())
            mock_opta_fn.assert_not_called()
        finally:
            pr._USE_OPTA_FOR_INFERENCE = original


if __name__ == "__main__":
    unittest.main()
