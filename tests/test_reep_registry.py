"""Tests for backend.data.reep_registry — uses real bundled CSV files."""

from __future__ import annotations

import os
from unittest.mock import patch
import pytest

from backend.data import reep_registry


# Sanity-check: the CSV files must exist in the repo.
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "reep",
)


@pytest.fixture(autouse=True)
def _clear_mem():
    """Ensure each test starts with a clean in-memory cache."""
    reep_registry.clear_memory_cache()
    yield
    reep_registry.clear_memory_cache()


class TestCsvFilesExist:
    """Guard: bundled CSV files must be present."""

    def test_teams_csv_exists(self):
        assert os.path.isfile(os.path.join(_DATA_DIR, "teams.csv"))

    def test_people_csv_exists(self):
        assert os.path.isfile(os.path.join(_DATA_DIR, "people.csv"))

    def test_competitions_csv_exists(self):
        assert os.path.isfile(os.path.join(_DATA_DIR, "competitions.csv"))

    def test_seasons_csv_exists(self):
        assert os.path.isfile(os.path.join(_DATA_DIR, "seasons.csv"))


class TestGetTeamsDf:
    """get_teams_df() loads and parses the real teams.csv."""

    def test_returns_dataframe(self):
        df = reep_registry.get_teams_df()
        assert df is not None
        assert len(df) > 40_000  # ~45k rows
        assert "reep_id" in df.columns
        assert "key_wikidata" in df.columns
        assert "key_clubelo" in df.columns
        assert "key_sofascore" in df.columns
        assert "key_thesportsdb" in df.columns
        assert "key_understat" in df.columns

    def test_caches_in_memory(self):
        df1 = reep_registry.get_teams_df()
        df2 = reep_registry.get_teams_df()
        assert df1 is df2  # exact same object, no re-parse

    def test_returns_none_when_file_missing(self):
        with patch.object(reep_registry, "_TEAMS_PATH", "/nonexistent/teams.csv"):
            assert reep_registry.get_teams_df() is None


class TestGetPeopleDf:
    """get_people_df() loads and parses the real people.csv."""

    def test_returns_dataframe(self):
        df = reep_registry.get_people_df()
        assert df is not None
        assert len(df) > 400_000  # ~430k rows
        assert "reep_id" in df.columns
        assert "key_wikidata" in df.columns
        assert "key_sofascore" in df.columns
        assert "key_skillcorner" in df.columns
        assert "key_impect" in df.columns

    def test_caches_in_memory(self):
        df1 = reep_registry.get_people_df()
        df2 = reep_registry.get_people_df()
        assert df1 is df2


class TestBuildClubEloSofascoreMap:
    """build_clubelo_sofascore_map() maps ClubElo key → display name."""

    def test_contains_known_clubs(self):
        mapping = reep_registry.build_clubelo_sofascore_map()
        assert len(mapping) > 0
        # Spot-check a club present in the real data
        assert "Arsenal" in mapping

    def test_caches_in_memory(self):
        m1 = reep_registry.build_clubelo_sofascore_map()
        m2 = reep_registry.build_clubelo_sofascore_map()
        assert m1 is m2

    def test_returns_empty_dict_when_file_missing(self):
        with patch.object(reep_registry, "_TEAMS_PATH", "/nonexistent/teams.csv"):
            assert reep_registry.build_clubelo_sofascore_map() == {}


class TestClubEloToSofascoreName:
    """clubelo_to_sofascore_name() resolves a single key."""

    def test_resolves_known_key(self):
        name = reep_registry.clubelo_to_sofascore_name("ManCity")
        assert name is not None
        assert "Manchester" in name or "City" in name

    def test_returns_none_for_unknown_key(self):
        assert reep_registry.clubelo_to_sofascore_name("ZZZ_FAKE_999") is None


class TestSofascoreTeamAliases:
    """sofascore_team_aliases() returns name variants."""

    def test_returns_aliases_for_known_team(self):
        # Use a sofascore ID that actually exists in the real data.
        # First find one dynamically.
        df = reep_registry.get_teams_df()
        valid = df[(df["key_sofascore"] != "") & df["key_sofascore"].str.match(r"^\d+$")]
        if valid.empty:
            pytest.skip("No sofascore IDs in teams.csv")
        sid = int(valid.iloc[0]["key_sofascore"])
        aliases = reep_registry.sofascore_team_aliases(sid)
        assert len(aliases) > 0

    def test_returns_empty_for_unknown_id(self):
        assert reep_registry.sofascore_team_aliases(999999999) == []


class TestEnrichPlayer:
    """enrich_player() returns metadata from people.csv."""

    def test_returns_metadata_for_known_player(self):
        # Erling Haaland — sofascore 839956
        info = reep_registry.enrich_player(839956)
        assert info.get("nationality") == "Norway"
        assert info.get("height_cm") == 194
        assert info.get("position") == "forward"
        assert info.get("date_of_birth") is not None

    def test_returns_reep_id_for_known_player(self):
        """enrich_player() must include the stable reep_id."""
        info = reep_registry.enrich_player(839956)
        assert info.get("reep_id") is not None
        assert info["reep_id"].startswith("reep_p")

    def test_returns_empty_for_unknown_player(self):
        assert reep_registry.enrich_player(999999999) == {}

    def test_returns_empty_when_file_missing(self):
        with patch.object(reep_registry, "_PEOPLE_PATH", "/nonexistent/people.csv"):
            assert reep_registry.enrich_player(839956) == {}

    def test_uses_indexed_lookup(self):
        """Second call should use the pre-built index (no re-parse)."""
        info1 = reep_registry.enrich_player(839956)
        info2 = reep_registry.enrich_player(839956)
        assert info1 == info2
        # Returned dicts are copies, not the same object
        assert info1 is not info2


class TestGetCompetitionsDf:
    """get_competitions_df() loads and parses the real competitions.csv."""

    def test_returns_dataframe(self):
        df = reep_registry.get_competitions_df()
        assert df is not None
        assert len(df) > 100  # ~200 competitions
        assert "reep_id" in df.columns
        assert "name" in df.columns

    def test_caches_in_memory(self):
        df1 = reep_registry.get_competitions_df()
        df2 = reep_registry.get_competitions_df()
        assert df1 is df2


class TestGetSeasonsDf:
    """get_seasons_df() loads and parses the real seasons.csv."""

    def test_returns_dataframe(self):
        df = reep_registry.get_seasons_df()
        assert df is not None
        assert len(df) > 500  # ~1200 seasons
        assert "reep_id" in df.columns
        assert "competition_reep_id" in df.columns

    def test_caches_in_memory(self):
        df1 = reep_registry.get_seasons_df()
        df2 = reep_registry.get_seasons_df()
        assert df1 is df2


class TestLookupTeam:
    """lookup_team() finds a team by reep_id."""

    def test_finds_known_team(self):
        df = reep_registry.get_teams_df()
        # Get the first team with a non-empty reep_id
        valid = df[df["reep_id"] != ""]
        if valid.empty:
            pytest.skip("No reep_id in teams.csv")
        rid = valid.iloc[0]["reep_id"]
        result = reep_registry.lookup_team(rid)
        assert result is not None
        assert result["reep_id"] == rid

    def test_returns_none_for_unknown(self):
        assert reep_registry.lookup_team("reep_t_NONEXISTENT") is None


class TestLookupPerson:
    """lookup_person() finds a person by reep_id."""

    def test_finds_known_person(self):
        df = reep_registry.get_people_df()
        valid = df[df["reep_id"] != ""]
        if valid.empty:
            pytest.skip("No reep_id in people.csv")
        rid = valid.iloc[0]["reep_id"]
        result = reep_registry.lookup_person(rid)
        assert result is not None
        assert result["reep_id"] == rid

    def test_returns_none_for_unknown(self):
        assert reep_registry.lookup_person("reep_p_NONEXISTENT") is None


class TestEnrichTeam:
    """enrich_team() returns metadata from teams.csv by Sofascore team ID."""

    def test_returns_metadata_for_known_team(self):
        # Find a team with a valid sofascore ID dynamically
        df = reep_registry.get_teams_df()
        valid = df[
            (df["key_sofascore"] != "") & df["key_sofascore"].str.match(r"^\d+$")
        ]
        if valid.empty:
            pytest.skip("No sofascore IDs in teams.csv")
        sid = int(valid.iloc[0]["key_sofascore"])
        info = reep_registry.enrich_team(sid)
        assert info.get("reep_id") is not None
        assert info["reep_id"].startswith("reep_t")
        assert info.get("name") is not None

    def test_returns_empty_for_unknown_team(self):
        assert reep_registry.enrich_team(999999999) == {}

    def test_returns_empty_when_file_missing(self):
        with patch.object(reep_registry, "_TEAMS_PATH", "/nonexistent/teams.csv"):
            assert reep_registry.enrich_team(1) == {}

    def test_uses_indexed_lookup(self):
        """Second call should reuse the pre-built index."""
        df = reep_registry.get_teams_df()
        valid = df[
            (df["key_sofascore"] != "") & df["key_sofascore"].str.match(r"^\d+$")
        ]
        if valid.empty:
            pytest.skip("No sofascore IDs in teams.csv")
        sid = int(valid.iloc[0]["key_sofascore"])
        info1 = reep_registry.enrich_team(sid)
        info2 = reep_registry.enrich_team(sid)
        assert info1 == info2
        assert info1 is not info2  # copies, not same object


class TestLookupCompetition:
    """lookup_competition() finds a competition by reep_id."""

    def test_finds_known_competition(self):
        df = reep_registry.get_competitions_df()
        valid = df[df["reep_id"] != ""]
        if valid.empty:
            pytest.skip("No reep_id in competitions.csv")
        rid = valid.iloc[0]["reep_id"]
        result = reep_registry.lookup_competition(rid)
        assert result is not None
        assert result["reep_id"] == rid
        assert result.get("name") is not None

    def test_returns_none_for_unknown(self):
        assert reep_registry.lookup_competition("reep_l_NONEXISTENT") is None


class TestLookupSeason:
    """lookup_season() finds a season by reep_id."""

    def test_finds_known_season(self):
        df = reep_registry.get_seasons_df()
        valid = df[df["reep_id"] != ""]
        if valid.empty:
            pytest.skip("No reep_id in seasons.csv")
        rid = valid.iloc[0]["reep_id"]
        result = reep_registry.lookup_season(rid)
        assert result is not None
        assert result["reep_id"] == rid

    def test_returns_none_for_unknown(self):
        assert reep_registry.lookup_season("reep_s_NONEXISTENT") is None


class TestGetCompetitionByProvider:
    """get_competition_by_provider() finds a competition by provider key."""

    def test_finds_by_fbref(self):
        # fbref=9 → Premier League
        result = reep_registry.get_competition_by_provider("key_fbref", "9")
        assert result is not None
        assert "Premier League" in result["name"]

    def test_finds_by_opta(self):
        # key_opta=8 → Premier League
        result = reep_registry.get_competition_by_provider("key_opta", "8")
        assert result is not None
        assert "Premier League" in result["name"]

    def test_returns_none_for_unknown_value(self):
        assert reep_registry.get_competition_by_provider("key_fbref", "99999") is None

    def test_returns_none_for_unknown_column(self):
        assert reep_registry.get_competition_by_provider("key_nonexistent", "1") is None


class TestGetSeasonsForCompetition:
    """get_seasons_for_competition() returns linked seasons."""

    def test_returns_list(self):
        # Even if no seasons are linked, should return a list
        result = reep_registry.get_seasons_for_competition("reep_lb3d230cb")
        assert isinstance(result, list)

    def test_returns_empty_for_unknown(self):
        result = reep_registry.get_seasons_for_competition("reep_l_NONEXISTENT")
        assert result == []


# ── Data Quality Validation ──────────────────────────────────────────────────


class TestTeamsCsvDataQuality:
    """Validate teams.csv has no misaligned / malformed values."""

    def test_reep_id_format(self):
        """reep_id values must follow the reep_t<hex> pattern."""
        df = reep_registry.get_teams_df()
        filled = df[df["reep_id"] != ""]
        assert len(filled) > 40_000  # every row should have one
        bad = []
        for _, row in filled.iterrows():
            val = row["reep_id"]
            if not val.startswith("reep_t"):
                bad.append(val)
                if len(bad) >= 5:
                    break
        assert len(bad) == 0, f"Bad reep_id format: {bad}"

    def test_key_sofascore_all_numeric(self):
        """key_sofascore entries must be clean integer strings (no '.0')."""
        df = reep_registry.get_teams_df()
        filled = df[df["key_sofascore"].notna()]
        for _, row in filled.iterrows():
            val = str(row["key_sofascore"]).strip()
            if val:
                assert val.isdigit(), (
                    f"key_sofascore '{val}' for {row['name']} is not a clean integer"
                )

    def test_key_transfermarkt_no_urls(self):
        """key_transfermarkt entries should be numeric IDs (tiny exceptions allowed).

        A handful of upstream entries may contain URLs; guard against mass
        corruption rather than zero tolerance.
        """
        df = reep_registry.get_teams_df()
        filled = df[df["key_transfermarkt"] != ""]
        url_count = sum(
            1 for v in filled["key_transfermarkt"] if str(v).startswith("http")
        )
        assert url_count < 10, (
            f"Too many URL key_transfermarkt values ({url_count}); "
            "upstream data may be corrupted"
        )

    def test_key_clubelo_no_misalignment(self):
        """Spot-check known teams have correct key_clubelo."""
        df = reep_registry.get_teams_df()
        checks = {
            "Lille OSC": "Lille",
            "AFC Bournemouth": "Bournemouth",
            "Brentford F.C.": "Brentford",
            "Fulham F.C.": "Fulham",
        }
        for team_name, expected_ce in checks.items():
            rows = df[df["name"] == team_name]
            if rows.empty:
                continue
            for _, row in rows.iterrows():
                ce = row.get("key_clubelo")
                if ce is not None and str(ce).strip() and str(ce) != "nan":
                    assert str(ce).strip() == expected_ce, (
                        f"{team_name} has key_clubelo='{ce}', expected '{expected_ce}'"
                    )


class TestPeopleCsvDataQuality:
    """Validate people.csv has no malformed values."""

    def test_reep_id_format(self):
        """reep_id values must follow the reep_<type><hex> pattern."""
        df = reep_registry.get_people_df()
        filled = df[df["reep_id"] != ""]
        assert len(filled) > 400_000
        bad = []
        for _, row in filled.iterrows():
            val = row["reep_id"]
            # People file includes players (reep_p) and coaches (reep_c)
            if not (val.startswith("reep_p") or val.startswith("reep_c")):
                bad.append(val)
                if len(bad) >= 5:
                    break
        assert len(bad) == 0, f"Bad reep_id format: {bad}"

    def test_height_cm_all_integer(self):
        """height_cm values must be numeric (integer or float-coercible)."""
        df = reep_registry.get_people_df()
        filled = df[df["height_cm"].notna()]
        bad = []
        for _, row in filled.iterrows():
            val = str(row["height_cm"]).strip()
            if not val or val == "nan":
                continue
            # Accept clean integers ("180") and float-style ("180.0")
            # which _safe_int() converts correctly.
            try:
                int(float(val))
            except (ValueError, TypeError):
                bad.append((row["key_wikidata"], val))
                if len(bad) >= 5:
                    break
        assert len(bad) == 0, f"Non-numeric height_cm values found: {bad}"

    def test_date_of_birth_no_urls(self):
        """date_of_birth should be YYYY-MM-DD or empty — URL count should be tiny.

        A small number of Wikidata blank-node URLs may exist upstream;
        the code treats them as missing.  Guard against mass corruption.
        """
        df = reep_registry.get_people_df()
        filled = df[df["date_of_birth"] != ""]
        url_count = sum(
            1 for v in filled["date_of_birth"] if str(v).startswith("http")
        )
        # Allow a known set of upstream Wikidata blank-node artefacts (<500),
        # but fail if many rows are corrupted (indicating a data pipeline issue).
        assert url_count < 500, (
            f"Too many URL date_of_birth values ({url_count}); "
            "upstream data may be corrupted"
        )

    def test_key_sofascore_all_numeric(self):
        """key_sofascore entries must be numeric strings (tiny exceptions allowed).

        A handful of upstream slug-style IDs (e.g. 'francesco-conti') may
        exist; the code skips them during index construction.
        """
        df = reep_registry.get_people_df()
        filled = df[(df["key_sofascore"] != "") & (df["key_sofascore"] != "nan")]
        bad = [v for v in filled["key_sofascore"] if not str(v).strip().isdigit()]
        assert len(bad) < 10, (
            f"Too many non-numeric key_sofascore values ({len(bad)}): "
            f"{bad[:5]}"
        )
