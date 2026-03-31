"""Tests for backend.data.reep_registry — all mocked, no network."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

# ── Minimal CSV test data ────────────────────────────────────────────────────
# Column order matches the real REEP teams.csv header (23 columns).
# Fields: key_wikidata(0),name(1),country(2),founded(3),stadium(4),
#   key_transfermarkt(5),key_fbref(6),key_soccerway(7),key_opta(8),
#   key_kicker(9),key_flashscore(10),key_sofascore(11),key_soccerbase(12),
#   key_uefa(13),key_footballdatabase_eu(14),key_worldfootball(15),
#   key_espn(16),key_playmakerstats(17),key_clubelo(18),key_sportmonks(19),
#   key_api_football(20),key_sofifa(21),key_fotmob(22)
_TEAMS_CSV = (
    "key_wikidata,name,country,founded,stadium,key_transfermarkt,key_fbref,"
    "key_soccerway,key_opta,key_kicker,key_flashscore,key_sofascore,"
    "key_soccerbase,key_uefa,key_footballdatabase_eu,key_worldfootball,"
    "key_espn,key_playmakerstats,key_clubelo,key_sportmonks,key_api_football,"
    "key_sofifa,key_fotmob\n"
    # Arsenal: sofascore=42, clubelo=Arsenal
    #  0       1              2       3    4 5  6        7 8 9 10 11 12 13 14 15 16 17 18      19 20 21   22
    "Q9616,Arsenal F.C.,England,1886,,11,18bb7c10,,,,,42,,,,,,,Arsenal,,,,8687\n"
    # Manchester City: sofascore=17, clubelo=ManCity
    "Q50602,Manchester City F.C.,England,1880,,281,b8fd03ef,,,,,17,,,,,,,ManCity,,,,8456\n"
    # PSG: sofascore=1644, clubelo=PSG
    "Q483020,Paris Saint-Germain F.C.,France,1970,,583,e2d8892c,,,,,1644,,,,,,,PSG,,,,10251\n"
    # Bayern: sofascore=2672, clubelo=Bayern
    "Q15862,FC Bayern München,Germany,1900,,27,054efa67,,,,,2672,,,,,,,Bayern,,,,9823\n"
    # Dortmund: sofascore=2673, clubelo=Dortmund
    "Q12217,Borussia Dortmund,Germany,1909,,16,e4a775cb,,,,,2673,,,,,,,Dortmund,,,,9789\n"
)

# Column order matches people.csv header (41 columns).
_PEOPLE_CSV = (
    "key_wikidata,type,name,full_name,date_of_birth,nationality,position,"
    "height_cm,key_transfermarkt,key_transfermarkt_manager,key_fbref,"
    "key_soccerway,key_sofascore,key_flashscore,key_opta,key_premier_league,"
    "key_11v11,key_espn,key_national_football_teams,key_worldfootball,"
    "key_soccerbase,key_kicker,key_uefa,key_lequipe,key_fff_fr,key_serie_a,"
    "key_besoccer,key_footballdatabase_eu,key_eu_football_info,key_hugman,"
    "key_german_fa,key_statmuse_pl,key_sofifa,key_soccerdonna,key_dongqiudi,"
    "key_understat,key_whoscored,key_fbref_verified,key_sportmonks,"
    "key_api_football,key_fotmob\n"
    # Haaland: sofascore=839956 (col 12)
    # Cols 0-12 populated, cols 13-40 empty (28 trailing commas)
    "Q67890,player,Erling Haaland,Erling Braut Haaland,2000-07-21,Norway,forward,195,418560,,1f44ac21,,839956"
    ",,,,,,,,,,,,,,,,,,,,,,,,,,,,\n"
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure each test starts with a clean cache state."""
    with patch("backend.data.reep_registry.cache") as mock_cache:
        mock_cache.get.return_value = None
        mock_cache.set = MagicMock()
        yield mock_cache


class TestGetTeamsDf:
    """get_teams_df() downloads and parses teams.csv."""

    def test_returns_dataframe_on_success(self, _clear_cache):
        mock_resp = MagicMock()
        mock_resp.text = _TEAMS_CSV
        mock_resp.raise_for_status = MagicMock()

        with patch("backend.data.reep_registry._requests_lib.get", return_value=mock_resp):
            from backend.data.reep_registry import get_teams_df

            df = get_teams_df()

        assert df is not None
        assert len(df) == 5
        assert "key_clubelo" in df.columns
        assert "key_sofascore" in df.columns

    def test_returns_none_on_network_failure(self, _clear_cache):
        with patch(
            "backend.data.reep_registry._requests_lib.get",
            side_effect=ConnectionError("no network"),
        ):
            from backend.data.reep_registry import get_teams_df

            df = get_teams_df()

        assert df is None

    def test_uses_cache_when_available(self, _clear_cache):
        _clear_cache.get.return_value = _TEAMS_CSV

        from backend.data.reep_registry import get_teams_df

        df = get_teams_df()
        assert df is not None
        assert len(df) == 5


class TestBuildClubEloSofascoreMap:
    """build_clubelo_sofascore_map() maps ClubElo key → display name."""

    def test_produces_correct_mapping(self, _clear_cache):
        _clear_cache.get.return_value = _TEAMS_CSV

        from backend.data.reep_registry import build_clubelo_sofascore_map

        mapping = build_clubelo_sofascore_map()

        assert mapping["Arsenal"] == "Arsenal F.C."
        assert mapping["ManCity"] == "Manchester City F.C."
        assert mapping["PSG"] == "Paris Saint-Germain F.C."
        assert mapping["Bayern"] == "FC Bayern München"
        assert mapping["Dortmund"] == "Borussia Dortmund"

    def test_returns_empty_dict_on_failure(self, _clear_cache):
        with patch(
            "backend.data.reep_registry._requests_lib.get",
            side_effect=ConnectionError("no network"),
        ):
            from backend.data.reep_registry import build_clubelo_sofascore_map

            mapping = build_clubelo_sofascore_map()

        assert mapping == {}


class TestClubEloToSofascoreName:
    """clubelo_to_sofascore_name() resolves a single key."""

    def test_resolves_known_key(self, _clear_cache):
        _clear_cache.get.return_value = _TEAMS_CSV

        from backend.data.reep_registry import clubelo_to_sofascore_name

        assert clubelo_to_sofascore_name("ManCity") == "Manchester City F.C."

    def test_returns_none_for_unknown_key(self, _clear_cache):
        _clear_cache.get.return_value = _TEAMS_CSV

        from backend.data.reep_registry import clubelo_to_sofascore_name

        assert clubelo_to_sofascore_name("NonExistentTeam") is None


class TestSofascoreTeamAliases:
    """sofascore_team_aliases() returns name variants."""

    def test_returns_aliases(self, _clear_cache):
        _clear_cache.get.return_value = _TEAMS_CSV

        from backend.data.reep_registry import sofascore_team_aliases

        aliases = sofascore_team_aliases(17)  # Man City
        assert "Manchester City F.C." in aliases
        assert "ManCity" in aliases

    def test_returns_empty_for_unknown_id(self, _clear_cache):
        _clear_cache.get.return_value = _TEAMS_CSV

        from backend.data.reep_registry import sofascore_team_aliases

        assert sofascore_team_aliases(999999) == []


class TestEnrichPlayer:
    """enrich_player() returns metadata from people.csv."""

    def test_returns_metadata_for_known_player(self, _clear_cache):
        _clear_cache.get.return_value = _PEOPLE_CSV

        from backend.data.reep_registry import enrich_player

        info = enrich_player(839956)  # Haaland
        assert info["nationality"] == "Norway"
        assert info["height_cm"] == 195
        assert info["date_of_birth"] == "2000-07-21"
        assert info["position"] == "forward"

    def test_returns_empty_for_unknown_player(self, _clear_cache):
        _clear_cache.get.return_value = _PEOPLE_CSV

        from backend.data.reep_registry import enrich_player

        assert enrich_player(999999) == {}

    def test_returns_empty_on_download_failure(self, _clear_cache):
        with patch(
            "backend.data.reep_registry._requests_lib.get",
            side_effect=ConnectionError("no network"),
        ):
            from backend.data.reep_registry import enrich_player

            assert enrich_player(839956) == {}
