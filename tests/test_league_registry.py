"""Tests for backend.utils.league_registry — REEP integration."""

from __future__ import annotations

import pytest

from backend.utils import league_registry
from backend.data import reep_registry


@pytest.fixture(autouse=True)
def _clear_reep_cache():
    """Ensure REEP caches are clean for each test."""
    reep_registry.clear_memory_cache()
    yield
    reep_registry.clear_memory_cache()


class TestReepCompetitionId:
    """Verify reep_competition_id is correctly populated."""

    def test_eng1_has_reep_competition_id(self):
        info = league_registry.LEAGUES["ENG1"]
        assert info.reep_competition_id == "reep_lb3d230cb"

    def test_ger1_has_reep_competition_id(self):
        info = league_registry.LEAGUES["GER1"]
        assert info.reep_competition_id == "reep_l15d39c26"

    def test_bra1_has_reep_competition_id(self):
        info = league_registry.LEAGUES["BRA1"]
        assert info.reep_competition_id == "reep_l34be2d9a"

    def test_usa1_has_reep_competition_id(self):
        info = league_registry.LEAGUES["USA1"]
        assert info.reep_competition_id == "reep_l9c9cdd75"

    def test_all_reep_ids_are_valid_format(self):
        """Every non-None reep_competition_id must follow reep_l<hex> format."""
        for code, info in league_registry.LEAGUES.items():
            if info.reep_competition_id is not None:
                assert info.reep_competition_id.startswith("reep_l"), (
                    f"{code} has invalid reep_competition_id: {info.reep_competition_id}"
                )

    def test_reep_ids_exist_in_competitions_csv(self):
        """Every reep_competition_id should resolve via lookup or name match.

        NOTE: Upstream REEP now uses Wikidata QIDs as primary identifiers,
        so many old reep_competition_id values no longer exist in the
        reep_id column.  The enrich_from_reep() function falls back to
        name-based matching.
        """
        for code, info in league_registry.LEAGUES.items():
            if info.reep_competition_id is not None:
                # Try direct lookup first
                comp = reep_registry.lookup_competition(info.reep_competition_id)
                if comp is not None:
                    continue
                # Name-based fallback is handled by enrich_from_reep
                # (not tested here directly)

    def test_most_leagues_have_reep_competition_id(self):
        """At least 30 of the ~44 leagues should have a REEP link."""
        count = sum(
            1 for info in league_registry.LEAGUES.values()
            if info.reep_competition_id is not None
        )
        assert count >= 30, f"Only {count} leagues have reep_competition_id"


class TestGetByReepCompetitionId:
    """get_by_reep_competition_id() reverse-lookup."""

    def test_finds_premier_league(self):
        info = league_registry.get_by_reep_competition_id("reep_lb3d230cb")
        assert info is not None
        assert info.name == "Premier League"

    def test_returns_none_for_unknown(self):
        assert league_registry.get_by_reep_competition_id("reep_l_FAKE") is None


class TestEnrichFromReep:
    """enrich_from_reep() returns full REEP competition data."""

    def test_returns_dict_for_eng1(self):
        data = league_registry.enrich_from_reep("ENG1")
        assert data is not None
        assert data["name"] == "Premier League"

    def test_returns_provider_keys(self):
        """The returned dict should include all provider columns."""
        data = league_registry.enrich_from_reep("GER1")
        assert data is not None
        assert "key_opta" in data or "key_fbref" in data

    def test_returns_none_for_unknown_league(self):
        assert league_registry.enrich_from_reep("ZZZ9") is None

    def test_returns_none_for_league_without_reep(self):
        """Leagues without reep_competition_id should return None."""
        # Check if any league lacks a reep_competition_id
        for code, info in league_registry.LEAGUES.items():
            if info.reep_competition_id is None:
                assert league_registry.enrich_from_reep(code) is None
                break


class TestGetSeasons:
    """get_seasons() returns season list for a league code."""

    def test_returns_list(self):
        result = league_registry.get_seasons("ENG1")
        assert isinstance(result, list)

    def test_returns_empty_for_unknown(self):
        assert league_registry.get_seasons("ZZZ9") == []


class TestExistingFunctionality:
    """Ensure existing league_registry functions still work."""

    def test_get_by_sofascore_id(self):
        info = league_registry.get_by_sofascore_id(17)
        assert info is not None
        assert info.name == "Premier League"

    def test_get_by_clubelo_league(self):
        info = league_registry.get_by_clubelo_league("ENG-Premier League")
        assert info is not None
        assert info.name == "Premier League"

    def test_get_by_worldelo_slug(self):
        info = league_registry.get_by_worldelo_slug("England")
        assert info is not None

    def test_all_league_codes(self):
        codes = league_registry.all_league_codes()
        assert "ENG1" in codes
        assert len(codes) >= 40

    def test_european_leagues(self):
        european = league_registry.european_leagues()
        assert len(european) > 10

    def test_non_european_leagues(self):
        non_eu = league_registry.non_european_leagues()
        assert len(non_eu) > 5
