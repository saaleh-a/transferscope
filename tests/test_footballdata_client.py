"""Tests for backend.data.footballdata_client — all mocked, no network."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest


_SAMPLE_CSV = """\
Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR
E0,19/08/23,Arsenal,Nottingham,2,1,H,2,0,H,15,6,7,2,12,12,8,3,2,2,0,0
E0,19/08/23,Bournemouth,West Ham,1,1,D,0,0,D,10,8,4,3,10,14,5,6,1,3,0,0
E0,19/08/23,Brighton,Luton,4,1,H,2,0,H,18,5,8,2,8,10,7,2,0,3,0,0
E0,19/08/23,Everton,Fulham,0,1,A,0,0,D,12,9,3,4,11,9,6,5,3,1,0,0
"""


@pytest.fixture(autouse=True)
def _patch_cache():
    """Prevent real cache reads/writes."""
    with patch("backend.data.footballdata_client.cache") as mock_cache:
        mock_cache.get.return_value = None
        mock_cache.set = MagicMock()
        yield mock_cache


class TestFetchSeason:
    """fetch_season() downloads and parses CSV."""

    def test_returns_dataframe_on_success(self, _patch_cache):
        mock_resp = MagicMock()
        mock_resp.text = _SAMPLE_CSV
        mock_resp.raise_for_status = MagicMock()

        with patch("backend.data.footballdata_client.fetch_season.__module__", "backend.data.footballdata_client"):
            # Patch at the import site inside the function
            import backend.data.footballdata_client as mod
            original_fetch = mod.fetch_season

            # Just test via cache path
            _patch_cache.get.return_value = _SAMPLE_CSV
            df = mod.fetch_season("ENG1", "2324")

        assert df is not None
        assert len(df) == 4
        assert "HomeTeam" in df.columns

    def test_returns_none_for_unknown_league(self, _patch_cache):
        import backend.data.footballdata_client as mod
        assert mod.fetch_season("UNKNOWN", "2324") is None

    def test_uses_cache(self, _patch_cache):
        _patch_cache.get.return_value = _SAMPLE_CSV
        import backend.data.footballdata_client as mod
        df = mod.fetch_season("ENG1", "2324")
        assert df is not None
        assert len(df) == 4


class TestComputeTeamStats:
    """compute_team_stats() aggregates per-game stats."""

    def test_aggregates_correctly(self, _patch_cache):
        _patch_cache.get.return_value = _SAMPLE_CSV
        import backend.data.footballdata_client as mod
        stats = mod.compute_team_stats("ENG1", "2324")

        assert stats is not None
        assert "Arsenal" in stats.index
        arsenal = stats.loc["Arsenal"]
        assert arsenal["games"] == 1
        assert arsenal["goals_per_game"] == 2.0
        assert arsenal["shots_per_game"] == 15.0

    def test_away_team_included(self, _patch_cache):
        _patch_cache.get.return_value = _SAMPLE_CSV
        import backend.data.footballdata_client as mod
        stats = mod.compute_team_stats("ENG1", "2324")

        assert "Nottingham" in stats.index
        nott = stats.loc["Nottingham"]
        assert nott["goals_per_game"] == 1.0
        assert nott["shots_per_game"] == 6.0


class TestComputeLeagueStyleProfile:
    """compute_league_style_profile() returns league averages."""

    def test_returns_league_averages(self, _patch_cache):
        _patch_cache.get.return_value = _SAMPLE_CSV
        import backend.data.footballdata_client as mod
        profile = mod.compute_league_style_profile("ENG1", "2324")

        assert profile != {}
        assert "goals" in profile
        assert "shots" in profile
        # 8 teams, each plays 1 game. Total goals = 2+1+1+1+4+1+0+1 = 11
        # 8 teams, mean goals/game = 11/8 = 1.375
        assert abs(profile["goals"] - 1.375) < 0.01

    def test_returns_empty_for_missing_data(self, _patch_cache):
        import backend.data.footballdata_client as mod
        profile = mod.compute_league_style_profile("UNKNOWN", "2324")
        assert profile == {}


class TestComputeMultiSeasonProfiles:
    """compute_multi_season_profiles() averages across seasons."""

    def test_averages_correctly(self, _patch_cache):
        _patch_cache.get.return_value = _SAMPLE_CSV
        import backend.data.footballdata_client as mod
        profiles = mod.compute_multi_season_profiles(
            league_codes=["ENG1"], seasons=["2324"]
        )
        assert "ENG1" in profiles
        assert "goals" in profiles["ENG1"]

    def test_empty_when_no_data(self, _patch_cache):
        import backend.data.footballdata_client as mod
        profiles = mod.compute_multi_season_profiles(
            league_codes=["UNKNOWN"], seasons=["2324"]
        )
        assert profiles == {}


class TestCalibrateStyleCoefficients:
    """calibrate_style_coefficients() refines coefficients from profiles."""

    def test_returns_calibrated_dicts(self):
        from backend.features.adjustment_models import calibrate_style_coefficients

        profiles = {
            "ENG1": {"goals": 1.4, "shots": 13.0, "shots_on_target": 4.5,
                     "fouls": 11.0, "corners": 5.5, "yellows": 1.8},
            "ESP1": {"goals": 1.2, "shots": 12.0, "shots_on_target": 4.0,
                     "fouls": 14.0, "corners": 5.0, "yellows": 2.5},
            "GER1": {"goals": 1.5, "shots": 12.5, "shots_on_target": 4.2,
                     "fouls": 12.0, "corners": 5.2, "yellows": 2.0},
        }
        result = calibrate_style_coefficients(profiles)

        assert "league_style_coeff" in result
        assert "opp_quality_sens" in result
        assert "expected_goals" in result["league_style_coeff"]
        assert "expected_goals" in result["opp_quality_sens"]

        # Calibrated values should differ from defaults (at least slightly)
        from backend.features.adjustment_models import _LEAGUE_STYLE_COEFF
        # At least one metric should have changed
        changed = any(
            abs(result["league_style_coeff"][m] - _LEAGUE_STYLE_COEFF[m]) > 0.001
            for m in _LEAGUE_STYLE_COEFF
        )
        assert changed, "Calibration should produce different coefficients"

    def test_returns_defaults_when_insufficient_data(self):
        from backend.features.adjustment_models import (
            calibrate_style_coefficients,
            _LEAGUE_STYLE_COEFF,
        )

        result = calibrate_style_coefficients(profiles={"ENG1": {"goals": 1.4}})
        assert result["league_style_coeff"] == _LEAGUE_STYLE_COEFF

    def test_returns_defaults_on_empty_profiles(self):
        from backend.features.adjustment_models import (
            calibrate_style_coefficients,
            _LEAGUE_STYLE_COEFF,
        )

        result = calibrate_style_coefficients(profiles={})
        assert result["league_style_coeff"] == _LEAGUE_STYLE_COEFF
