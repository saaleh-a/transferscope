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


class TestGetTeamsDf:
    """get_teams_df() loads and parses the real teams.csv."""

    def test_returns_dataframe(self):
        df = reep_registry.get_teams_df()
        assert df is not None
        assert len(df) > 40_000  # ~45k rows
        assert "key_clubelo" in df.columns
        assert "key_sofascore" in df.columns

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
        assert "key_sofascore" in df.columns

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


# ── Data Quality Validation ──────────────────────────────────────────────────


class TestTeamsCsvDataQuality:
    """Validate teams.csv has no misaligned / malformed values."""

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

    def test_key_transfermarkt_all_numeric(self):
        """key_transfermarkt entries must be numeric IDs (no URLs)."""
        df = reep_registry.get_teams_df()
        filled = df[df["key_transfermarkt"].notna()]
        for _, row in filled.iterrows():
            val = str(row["key_transfermarkt"]).strip()
            if val and val != "nan":
                assert not val.startswith("http"), (
                    f"key_transfermarkt for {row['name']} is a URL: {val}"
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

    def test_height_cm_all_integer(self):
        """height_cm values must be clean integers (no '.0' or decimals)."""
        df = reep_registry.get_people_df()
        filled = df[df["height_cm"].notna()]
        bad = []
        for _, row in filled.iterrows():
            val = str(row["height_cm"]).strip()
            if val and val != "nan" and not val.isdigit():
                bad.append((row["key_wikidata"], val))
                if len(bad) >= 5:
                    break
        assert len(bad) == 0, f"Non-integer height_cm values found: {bad}"

    def test_date_of_birth_no_urls(self):
        """date_of_birth must be YYYY-MM-DD or empty — never a URL."""
        df = reep_registry.get_people_df()
        filled = df[df["date_of_birth"].notna()]
        bad = []
        for _, row in filled.iterrows():
            val = str(row["date_of_birth"]).strip()
            if val and val.startswith("http"):
                bad.append((row["key_wikidata"], val[:60]))
                if len(bad) >= 5:
                    break
        assert len(bad) == 0, f"URL date_of_birth values found: {bad}"

    def test_key_sofascore_all_numeric(self):
        """key_sofascore entries must be clean integer strings."""
        df = reep_registry.get_people_df()
        filled = df[df["key_sofascore"].notna()]
        bad = []
        for _, row in filled.iterrows():
            val = str(row["key_sofascore"]).strip()
            if val and val != "nan" and not val.isdigit():
                bad.append((row["key_wikidata"], val))
                if len(bad) >= 5:
                    break
        assert len(bad) == 0, f"Non-numeric key_sofascore values: {bad}"
