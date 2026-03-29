"""Tests for frontend.pages.backtest_validator season-matching helpers."""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from frontend.pages.backtest_validator import _match_season, _parse_season_years


class TestParseSeasonYears(unittest.TestCase):
    """_parse_season_years must handle plain and prefixed season names."""

    # Plain formats (no text prefix)
    def test_short_split(self):
        self.assertEqual(_parse_season_years("24/25"), (2024, 2025))

    def test_long_split(self):
        self.assertEqual(_parse_season_years("2024/2025"), (2024, 2025))

    def test_calendar_year(self):
        self.assertEqual(_parse_season_years("2024"), (2024, 2024))

    # Prefixed formats (Sofascore tournament-prefixed season names)
    def test_liga_portugal_prefix(self):
        self.assertEqual(_parse_season_years("Liga Portugal 25/26"), (2025, 2026))

    def test_premier_league_prefix(self):
        self.assertEqual(
            _parse_season_years("Premier League 2024/2025"), (2024, 2025)
        )

    def test_mls_calendar_prefix(self):
        self.assertEqual(_parse_season_years("MLS 2025"), (2025, 2025))

    def test_serie_a_prefix(self):
        self.assertEqual(_parse_season_years("Serie A 23/24"), (2023, 2024))

    # Edge cases
    def test_empty_string(self):
        self.assertEqual(_parse_season_years(""), (0, 0))

    def test_no_numbers(self):
        self.assertEqual(_parse_season_years("Season Unknown"), (0, 0))

    def test_whitespace_around_slash(self):
        self.assertEqual(_parse_season_years("24 / 25"), (2024, 2025))


class TestMatchSeason(unittest.TestCase):
    """_match_season must select the correct pre/post season."""

    def _seasons(self, names: List[str]) -> List[Dict[str, Any]]:
        """Build season dicts from a list of names (newest first)."""
        return [{"id": 1000 + i, "name": n} for i, n in enumerate(names)]

    # ── Summer transfer, split-year league ─────────────────────────────
    def test_summer_pre_split_year(self):
        """July 2025 transfer → pre-transfer season = 24/25."""
        seasons = self._seasons(["25/26", "24/25", "23/24"])
        result = _match_season(seasons, 2025, 7, "pre")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "24/25")

    def test_summer_post_split_year(self):
        """July 2025 transfer → post-transfer season = 25/26."""
        seasons = self._seasons(["25/26", "24/25", "23/24"])
        result = _match_season(seasons, 2025, 7, "post")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "25/26")

    # ── Summer transfer, prefixed season names ─────────────────────────
    def test_summer_pre_prefixed_names(self):
        """Prefixed season names (e.g. 'Liga Portugal 25/26') must parse."""
        seasons = self._seasons(["Liga Portugal 25/26", "Liga Portugal 24/25"])
        result = _match_season(seasons, 2025, 7, "pre")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Liga Portugal 24/25")

    def test_summer_post_prefixed_names(self):
        seasons = self._seasons(["Liga Portugal 25/26", "Liga Portugal 24/25"])
        result = _match_season(seasons, 2025, 7, "post")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Liga Portugal 25/26")

    # ── Winter transfer, split-year league ─────────────────────────────
    def test_winter_pre_split_year(self):
        """Jan 2025 transfer → pre-transfer season = 24/25."""
        seasons = self._seasons(["25/26", "24/25", "23/24"])
        result = _match_season(seasons, 2025, 1, "pre")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "24/25")

    def test_winter_post_split_year(self):
        """Jan 2025 transfer → post-transfer season = 24/25 (same season)."""
        seasons = self._seasons(["25/26", "24/25", "23/24"])
        result = _match_season(seasons, 2025, 1, "post")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "24/25")

    # ── Calendar-year league (MLS) ─────────────────────────────────────
    def test_summer_pre_calendar_year(self):
        seasons = self._seasons(["2025", "2024", "2023"])
        result = _match_season(seasons, 2025, 7, "pre")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "2025")

    def test_summer_post_calendar_year(self):
        seasons = self._seasons(["2025", "2024", "2023"])
        result = _match_season(seasons, 2025, 7, "post")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "2025")

    # ── Fallback when only one season available ────────────────────────
    def test_single_season_pre_fallback(self):
        """Only newest season available → fallback for pre should still pick it."""
        seasons = self._seasons(["25/26"])
        result = _match_season(seasons, 2025, 7, "pre")
        # Only 25/26 available; can't find 24/25.  Fallback returns what's there.
        self.assertIsNotNone(result)

    def test_empty_seasons(self):
        self.assertIsNone(_match_season([], 2025, 7, "pre"))


if __name__ == "__main__":
    unittest.main()
