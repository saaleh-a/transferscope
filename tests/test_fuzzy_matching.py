"""Unit tests for fuzzy team name matching in power_rankings.

Tests _normalize_team_name and _fuzzy_find_team against real-world
ClubElo ↔ Sofascore naming discrepancies.
"""

import os
import shutil
import tempfile
import unittest

_TEMP_DIR = tempfile.mkdtemp(prefix="ts_fuzzy_test_")
os.environ["CACHE_DIR"] = _TEMP_DIR

from backend.features.power_rankings import (
    TeamRanking,
    _fuzzy_find_team,
    _normalize_team_name,
)


def _make_ranking(name: str, code: str = "ENG1") -> TeamRanking:
    """Helper to build a minimal TeamRanking for testing."""
    return TeamRanking(
        team_name=name,
        league_code=code,
        raw_elo=1800.0,
        normalized_score=60.0,
        league_mean_normalized=50.0,
        relative_ability=10.0,
    )


# Simulate ClubElo team names (as they appear in the real API)
_CLUBELO_TEAMS = {
    # England
    "Arsenal": _make_ranking("Arsenal"),
    "Man City": _make_ranking("Man City"),
    "Man Utd": _make_ranking("Man Utd"),
    "Liverpool": _make_ranking("Liverpool"),
    "Chelsea": _make_ranking("Chelsea"),
    "Tottenham": _make_ranking("Tottenham"),
    "Newcastle": _make_ranking("Newcastle"),
    "Brighton": _make_ranking("Brighton"),
    "Aston Villa": _make_ranking("Aston Villa"),
    "West Ham": _make_ranking("West Ham"),
    "Wolves": _make_ranking("Wolves"),
    "Nottm Forest": _make_ranking("Nottm Forest"),
    "Crystal Palace": _make_ranking("Crystal Palace"),
    "Fulham": _make_ranking("Fulham"),
    "Brentford": _make_ranking("Brentford"),
    "Everton": _make_ranking("Everton"),
    "Bournemouth": _make_ranking("Bournemouth"),
    "Leicester": _make_ranking("Leicester"),
    # France
    "PSG": _make_ranking("PSG", "FRA1"),
    "Paris FC": _make_ranking("Paris FC", "FRA2"),
    "Marseille": _make_ranking("Marseille", "FRA1"),
    "Monaco": _make_ranking("Monaco", "FRA1"),
    "Lyon": _make_ranking("Lyon", "FRA1"),
    "Lille": _make_ranking("Lille", "FRA1"),
    # Germany
    "Bayern Munich": _make_ranking("Bayern Munich", "GER1"),
    "Dortmund": _make_ranking("Dortmund", "GER1"),
    "Leverkusen": _make_ranking("Leverkusen", "GER1"),
    "RB Leipzig": _make_ranking("RB Leipzig", "GER1"),
    "Frankfurt": _make_ranking("Frankfurt", "GER1"),
    "Gladbach": _make_ranking("Gladbach", "GER1"),
    # Spain
    "Real Madrid": _make_ranking("Real Madrid", "ESP1"),
    "Barcelona": _make_ranking("Barcelona", "ESP1"),
    "Atletico": _make_ranking("Atletico", "ESP1"),
    "Athletic Bilbao": _make_ranking("Athletic Bilbao", "ESP1"),
    "Real Sociedad": _make_ranking("Real Sociedad", "ESP1"),
    "Betis": _make_ranking("Betis", "ESP1"),
    "Villarreal": _make_ranking("Villarreal", "ESP1"),
    "Celta Vigo": _make_ranking("Celta Vigo", "ESP1"),
    # Italy
    "Inter": _make_ranking("Inter", "ITA1"),
    "AC Milan": _make_ranking("AC Milan", "ITA1"),
    "Juventus": _make_ranking("Juventus", "ITA1"),
    "Napoli": _make_ranking("Napoli", "ITA1"),
    "AS Roma": _make_ranking("AS Roma", "ITA1"),
    "Lazio": _make_ranking("Lazio", "ITA1"),
    "Atalanta": _make_ranking("Atalanta", "ITA1"),
    # Portugal
    "Benfica": _make_ranking("Benfica", "POR1"),
    "Porto": _make_ranking("Porto", "POR1"),
    "Sporting CP": _make_ranking("Sporting CP", "POR1"),
    # Netherlands
    "Ajax": _make_ranking("Ajax", "NED1"),
    "PSV": _make_ranking("PSV", "NED1"),
    "Feyenoord": _make_ranking("Feyenoord", "NED1"),
    "AZ": _make_ranking("AZ", "NED1"),
}


class TestNormalizeTeamName(unittest.TestCase):
    """Test the _normalize_team_name helper."""

    def test_strips_fc(self):
        self.assertEqual(_normalize_team_name("Arsenal FC"), "arsenal")

    def test_strips_ac(self):
        self.assertEqual(_normalize_team_name("AC Milan"), "milan")

    def test_strips_as(self):
        self.assertEqual(_normalize_team_name("AS Roma"), "roma")

    def test_does_not_strip_club(self):
        """'Club' was removed from strip list — Athletic Club keeps identity."""
        result = _normalize_team_name("Athletic Club")
        self.assertIn("club", result)

    def test_accent_removal(self):
        result = _normalize_team_name("Atlético Madrid")
        self.assertEqual(result, "atleticomadrid")

    def test_removes_spaces_and_hyphens(self):
        result = _normalize_team_name("Paris Saint-Germain")
        self.assertEqual(result, "parissaintgermain")

    def test_lowercase(self):
        result = _normalize_team_name("REAL MADRID")
        self.assertEqual(result, "realmadrid")


class TestFuzzyFindTeam(unittest.TestCase):
    """Test _fuzzy_find_team against real-world naming discrepancies."""

    # ── Exact matches (step 1) ───────────────────────────────────────────

    def test_exact_match(self):
        self.assertEqual(
            _fuzzy_find_team("Arsenal", _CLUBELO_TEAMS), "Arsenal"
        )

    def test_exact_match_with_fc(self):
        """'Arsenal FC' normalizes to 'arsenal' — same as 'Arsenal'."""
        self.assertEqual(
            _fuzzy_find_team("Arsenal FC", _CLUBELO_TEAMS), "Arsenal"
        )

    # ── PSG vs Paris FC (the critical bug) ───────────────────────────────

    def test_psg_from_full_name(self):
        """Paris Saint-Germain must match PSG, NOT Paris FC."""
        result = _fuzzy_find_team("Paris Saint-Germain", _CLUBELO_TEAMS)
        self.assertEqual(result, "PSG")

    def test_psg_from_abbreviation(self):
        """'PSG' direct lookup."""
        result = _fuzzy_find_team("PSG", _CLUBELO_TEAMS)
        self.assertEqual(result, "PSG")

    def test_paris_fc_still_works(self):
        """Paris FC should still match Paris FC (exact normalized)."""
        result = _fuzzy_find_team("Paris FC", _CLUBELO_TEAMS)
        self.assertEqual(result, "Paris FC")

    # ── English clubs ────────────────────────────────────────────────────

    def test_manchester_city(self):
        result = _fuzzy_find_team("Manchester City", _CLUBELO_TEAMS)
        self.assertEqual(result, "Man City")

    def test_manchester_united(self):
        result = _fuzzy_find_team("Manchester United", _CLUBELO_TEAMS)
        self.assertEqual(result, "Man Utd")

    def test_wolverhampton_wanderers(self):
        result = _fuzzy_find_team("Wolverhampton Wanderers", _CLUBELO_TEAMS)
        self.assertEqual(result, "Wolves")

    def test_tottenham_hotspur(self):
        result = _fuzzy_find_team("Tottenham Hotspur", _CLUBELO_TEAMS)
        self.assertEqual(result, "Tottenham")

    def test_nottingham_forest(self):
        result = _fuzzy_find_team("Nottingham Forest", _CLUBELO_TEAMS)
        self.assertEqual(result, "Nottm Forest")

    def test_newcastle_united(self):
        result = _fuzzy_find_team("Newcastle United", _CLUBELO_TEAMS)
        self.assertEqual(result, "Newcastle")

    def test_brighton_hove_albion(self):
        result = _fuzzy_find_team("Brighton & Hove Albion", _CLUBELO_TEAMS)
        self.assertEqual(result, "Brighton")

    def test_west_ham_united(self):
        result = _fuzzy_find_team("West Ham United", _CLUBELO_TEAMS)
        self.assertEqual(result, "West Ham")

    def test_leicester_city(self):
        result = _fuzzy_find_team("Leicester City", _CLUBELO_TEAMS)
        self.assertEqual(result, "Leicester")

    # ── German clubs ─────────────────────────────────────────────────────

    def test_bayern_munchen(self):
        """Sofascore uses 'Bayern München', ClubElo uses 'Bayern Munich'."""
        result = _fuzzy_find_team("Bayern München", _CLUBELO_TEAMS)
        self.assertEqual(result, "Bayern Munich")

    def test_borussia_dortmund(self):
        result = _fuzzy_find_team("Borussia Dortmund", _CLUBELO_TEAMS)
        self.assertEqual(result, "Dortmund")

    def test_bayer_leverkusen(self):
        result = _fuzzy_find_team("Bayer 04 Leverkusen", _CLUBELO_TEAMS)
        self.assertEqual(result, "Leverkusen")

    def test_eintracht_frankfurt(self):
        result = _fuzzy_find_team("Eintracht Frankfurt", _CLUBELO_TEAMS)
        self.assertEqual(result, "Frankfurt")

    def test_borussia_mgladbach(self):
        result = _fuzzy_find_team("Borussia Mönchengladbach", _CLUBELO_TEAMS)
        self.assertEqual(result, "Gladbach")

    # ── Spanish clubs ────────────────────────────────────────────────────

    def test_atletico_madrid(self):
        result = _fuzzy_find_team("Atlético Madrid", _CLUBELO_TEAMS)
        self.assertEqual(result, "Atletico")

    def test_athletic_club(self):
        """Athletic Club (Bilbao) — 'Club' must NOT be stripped."""
        result = _fuzzy_find_team("Athletic Club", _CLUBELO_TEAMS)
        self.assertEqual(result, "Athletic Bilbao")

    def test_real_betis(self):
        result = _fuzzy_find_team("Real Betis", _CLUBELO_TEAMS)
        self.assertEqual(result, "Betis")

    def test_celta_de_vigo(self):
        result = _fuzzy_find_team("Celta de Vigo", _CLUBELO_TEAMS)
        self.assertEqual(result, "Celta Vigo")

    # ── Italian clubs ────────────────────────────────────────────────────

    def test_inter_milan(self):
        result = _fuzzy_find_team("Inter Milan", _CLUBELO_TEAMS)
        self.assertEqual(result, "Inter")

    def test_internazionale(self):
        result = _fuzzy_find_team("FC Internazionale Milano", _CLUBELO_TEAMS)
        self.assertEqual(result, "Inter")

    def test_ac_milan(self):
        """AC is stripped, so 'AC Milan' → 'milan' matches 'AC Milan' → 'milan'."""
        result = _fuzzy_find_team("AC Milan", _CLUBELO_TEAMS)
        self.assertEqual(result, "AC Milan")

    def test_roma(self):
        result = _fuzzy_find_team("Roma", _CLUBELO_TEAMS)
        self.assertEqual(result, "AS Roma")

    # ── Portuguese clubs ─────────────────────────────────────────────────

    def test_sporting_lisbon(self):
        result = _fuzzy_find_team("Sporting Lisbon", _CLUBELO_TEAMS)
        self.assertEqual(result, "Sporting CP")

    # ── False positive guard ─────────────────────────────────────────────

    def test_no_false_positive_arsenal_marseille(self):
        """Arsenal should NOT match Marseille (ratio ~0.63)."""
        result = _fuzzy_find_team("Arsenal", _CLUBELO_TEAMS)
        self.assertNotEqual(result, "Marseille")

    def test_no_false_positive_inter_miami_for_inter(self):
        """If 'Inter Miami' is searched, it should NOT match 'Inter' (Milan)
        since the overlap ratio is too low."""
        # Inter Miami normalizes to "intermiami" (10 chars)
        # Inter normalizes to "inter" (5 chars) — 5/10 = 0.5 but < 6 chars
        # This should fall through to SequenceMatcher at best
        result = _fuzzy_find_team("Inter Miami CF", _CLUBELO_TEAMS)
        # Either matches Inter (acceptable) or None — but must not crash
        self.assertIn(result, ["Inter", None])

    def test_returns_none_for_unknown(self):
        """Completely unknown team returns None."""
        result = _fuzzy_find_team("Fictional FC United", _CLUBELO_TEAMS)
        self.assertIsNone(result)


class TestFuzzyEdgeCases(unittest.TestCase):
    """Edge cases and regression tests."""

    def test_empty_query(self):
        self.assertIsNone(_fuzzy_find_team("", _CLUBELO_TEAMS))

    def test_empty_teams(self):
        self.assertIsNone(_fuzzy_find_team("Arsenal", {}))

    def test_accented_query(self):
        """Accents in query are stripped for matching."""
        result = _fuzzy_find_team("Nápoli", _CLUBELO_TEAMS)
        self.assertEqual(result, "Napoli")

    def test_case_insensitive(self):
        result = _fuzzy_find_team("LIVERPOOL", _CLUBELO_TEAMS)
        self.assertEqual(result, "Liverpool")

    def test_rb_leipzig(self):
        result = _fuzzy_find_team("RB Leipzig", _CLUBELO_TEAMS)
        self.assertEqual(result, "RB Leipzig")

    def test_psv_eindhoven(self):
        result = _fuzzy_find_team("PSV Eindhoven", _CLUBELO_TEAMS)
        self.assertEqual(result, "PSV")

    def test_orlando_city_does_not_match_man_city(self):
        """Regression: Orlando City SC must NOT match Man City (0.667 ratio)."""
        result = _fuzzy_find_team("Orlando City SC", _CLUBELO_TEAMS)
        # Orlando City isn't in _CLUBELO_TEAMS, so should be None — NOT Man City
        self.assertNotEqual(result, "Man City")

    def test_inter_miami_does_not_match_inter_milan(self):
        """Regression: Inter Miami CF must NOT match Inter (Milan)."""
        result = _fuzzy_find_team("Inter Miami CF", _CLUBELO_TEAMS)
        self.assertNotEqual(result, "Inter")

    def test_new_york_city_fc_does_not_match_man_city(self):
        """Regression: New York City FC must NOT match Man City."""
        result = _fuzzy_find_team("New York City FC", _CLUBELO_TEAMS)
        self.assertNotEqual(result, "Man City")

    def test_orlando_city_matches_when_in_dict(self):
        """Orlando City SC matches Orlando City when it's in the teams dict."""
        teams_with_orlando = dict(_CLUBELO_TEAMS)
        teams_with_orlando["Orlando City"] = _make_ranking("Orlando City", "USA1")
        result = _fuzzy_find_team("Orlando City SC", teams_with_orlando)
        self.assertEqual(result, "Orlando City")

    def test_inter_miami_matches_when_in_dict(self):
        """Inter Miami CF matches Inter Miami when it's in the teams dict."""
        teams_with_miami = dict(_CLUBELO_TEAMS)
        teams_with_miami["Inter Miami"] = _make_ranking("Inter Miami", "USA1")
        result = _fuzzy_find_team("Inter Miami CF", teams_with_miami)
        self.assertEqual(result, "Inter Miami")


class TestDirectMapping(unittest.TestCase):
    """Test the _CLUBELO_TO_SOFASCORE direct mapping and canonicalization."""

    def test_psg_canonical(self):
        from backend.features.power_rankings import _CLUBELO_TO_SOFASCORE
        self.assertEqual(_CLUBELO_TO_SOFASCORE["PSG"], "Paris Saint-Germain")

    def test_mancity_canonical(self):
        from backend.features.power_rankings import _CLUBELO_TO_SOFASCORE
        self.assertEqual(_CLUBELO_TO_SOFASCORE["ManCity"], "Manchester City")

    def test_bayern_canonical(self):
        from backend.features.power_rankings import _CLUBELO_TO_SOFASCORE
        self.assertEqual(_CLUBELO_TO_SOFASCORE["Bayern"], "Bayern Munich")

    def test_atletico_canonical(self):
        from backend.features.power_rankings import _CLUBELO_TO_SOFASCORE
        self.assertEqual(_CLUBELO_TO_SOFASCORE["Atletico"], "Atlético Madrid")

    def test_reverse_mapping_exists(self):
        from backend.features.power_rankings import _SOFASCORE_TO_CLUBELO
        self.assertIn("Paris Saint-Germain", _SOFASCORE_TO_CLUBELO)
        self.assertIn("Manchester City", _SOFASCORE_TO_CLUBELO)

    def test_canonicalized_teams_match_sofascore_names(self):
        """After canonicalization, Sofascore dropdown names should match exactly."""
        from backend.features.power_rankings import _CLUBELO_TO_SOFASCORE

        # Simulate building teams dict with canonical names
        raw_clubelo = ["PSG", "ManCity", "ManUtd", "Bayern", "Dortmund",
                       "RealMadrid", "Atletico", "Wolves", "Tottenham"]
        canonical_names = {_CLUBELO_TO_SOFASCORE.get(n, n) for n in raw_clubelo}

        # These Sofascore names should all be in the canonical set
        sofascore_names = [
            "Paris Saint-Germain", "Manchester City", "Manchester United",
            "Bayern Munich", "Borussia Dortmund", "Real Madrid",
            "Atlético Madrid", "Wolverhampton Wanderers", "Tottenham Hotspur",
        ]
        for name in sofascore_names:
            self.assertIn(name, canonical_names, f"{name!r} not in canonical set")

    def test_parissg_alias_in_extreme_abbrevs(self):
        """'parissg' alias handles ClubElo returning 'Paris SG' variant."""
        from backend.features.power_rankings import _EXTREME_ABBREVS
        self.assertIn("parissg", _EXTREME_ABBREVS)
        self.assertIn("parissaintgermain", _EXTREME_ABBREVS["parissg"])


class TestCountryLevelFallback(unittest.TestCase):
    """Test _clubelo_to_code_from_country for the soccerdata NaN league fix."""

    def test_france_level1(self):
        from backend.features.power_rankings import _clubelo_to_code_from_country
        self.assertEqual(_clubelo_to_code_from_country("FRA", 1), "FRA1")

    def test_england_level2(self):
        from backend.features.power_rankings import _clubelo_to_code_from_country
        self.assertEqual(_clubelo_to_code_from_country("ENG", 2), "ENG2")

    def test_netherlands_level1(self):
        from backend.features.power_rankings import _clubelo_to_code_from_country
        self.assertEqual(_clubelo_to_code_from_country("NED", 1), "NED1")

    def test_turkey_level1(self):
        from backend.features.power_rankings import _clubelo_to_code_from_country
        self.assertEqual(_clubelo_to_code_from_country("TUR", 1), "TUR1")

    def test_portugal_level1(self):
        from backend.features.power_rankings import _clubelo_to_code_from_country
        self.assertEqual(_clubelo_to_code_from_country("POR", 1), "POR1")

    def test_unknown_country_returns_none(self):
        from backend.features.power_rankings import _clubelo_to_code_from_country
        self.assertIsNone(_clubelo_to_code_from_country("ZZZ", 1))

    def test_nan_country_returns_none(self):
        import math
        from backend.features.power_rankings import _clubelo_to_code_from_country
        self.assertIsNone(_clubelo_to_code_from_country(float("nan"), 1))

    def test_none_country_returns_none(self):
        from backend.features.power_rankings import _clubelo_to_code_from_country
        self.assertIsNone(_clubelo_to_code_from_country(None, 1))


class TestStripAccents(unittest.TestCase):
    """Test accent normalization helper."""

    def test_atletico(self):
        from backend.features.power_rankings import _strip_accents
        self.assertEqual(_strip_accents("Atlético Madrid"), "Atletico Madrid")

    def test_monchengladbach(self):
        from backend.features.power_rankings import _strip_accents
        self.assertEqual(_strip_accents("Borussia Mönchengladbach"),
                         "Borussia Monchengladbach")

    def test_fenerbahce(self):
        from backend.features.power_rankings import _strip_accents
        self.assertEqual(_strip_accents("Fenerbahçe SK"), "Fenerbahce SK")

    def test_no_accents_unchanged(self):
        from backend.features.power_rankings import _strip_accents
        self.assertEqual(_strip_accents("Arsenal"), "Arsenal")

    def test_preserves_case_and_spacing(self):
        from backend.features.power_rankings import _strip_accents
        self.assertEqual(_strip_accents("São Paulo FC"), "Sao Paulo FC")


class TestExtremeAbbrevsFixes(unittest.TestCase):
    """Regression tests for bugs in _EXTREME_ABBREVS keys."""

    def test_mainz_key_has_no_space(self):
        """Bug fix: '1fsv mainz05' had a space — must be '1fsvmainz05'.

        The space caused the reverse abbreviation lookup to fail because
        _normalize_team_name removes all spaces, making '1fsvmainz05' (no
        space) the lookup key, which did not match the old spaced key.
        """
        from backend.features.power_rankings import _EXTREME_ABBREVS, _normalize_team_name

        full_name = "1. FSV Mainz 05"
        normalized = _normalize_team_name(full_name)
        self.assertEqual(normalized, "1fsvmainz05")
        # Key must exist without a space
        self.assertIn(normalized, _EXTREME_ABBREVS)
        self.assertNotIn("1fsv mainz05", _EXTREME_ABBREVS)
        # Aliases must include 'mainz' and 'mainz05'
        self.assertIn("mainz", _EXTREME_ABBREVS[normalized])
        self.assertIn("mainz05", _EXTREME_ABBREVS[normalized])

    def test_mainz_reverse_lookup_works(self):
        """'Mainz' query must resolve to '1. FSV Mainz 05' via reverse lookup."""
        teams = {"1. FSV Mainz 05": _make_ranking("1. FSV Mainz 05", "GER1")}
        result = _fuzzy_find_team("Mainz", teams)
        self.assertEqual(result, "1. FSV Mainz 05")

    def test_mainz05_reverse_lookup_works(self):
        """'Mainz 05' query must resolve to '1. FSV Mainz 05' via reverse lookup."""
        teams = {"1. FSV Mainz 05": _make_ranking("1. FSV Mainz 05", "GER1")}
        result = _fuzzy_find_team("Mainz 05", teams)
        self.assertEqual(result, "1. FSV Mainz 05")

    def test_cincinnati_key_is_correct(self):
        """Bug fix: 'cincinnatitied' was a typo — must be 'cincinnati'.

        The old key 'cincinnatitied' (with 'tied' appended) and alias
        'cincinnatidied' were both typos and would never match any real
        team name.  The correct key is 'cincinnati' (the normalized form
        of 'FC Cincinnati' and 'Cincinnati').
        """
        from backend.features.power_rankings import _EXTREME_ABBREVS

        self.assertIn("cincinnati", _EXTREME_ABBREVS)
        self.assertNotIn("cincinnatitied", _EXTREME_ABBREVS)
        # Must not contain the typo alias
        self.assertNotIn("cincinnatidied", _EXTREME_ABBREVS.get("cincinnati", []))
        # Must include the fccincinnati alias
        self.assertIn("fccincinnati", _EXTREME_ABBREVS["cincinnati"])

    def test_cincinnati_forward_lookup_works(self):
        """'Cincinnati' must resolve to 'FCCincinnati' via abbreviation lookup."""
        # Simulate a data source that stores the team without space (no FC stripping)
        teams = {"FCCincinnati": _make_ranking("FCCincinnati", "USA1")}
        result = _fuzzy_find_team("Cincinnati", teams)
        self.assertEqual(result, "FCCincinnati")


class TestDynamicAliases(unittest.TestCase):
    """Test the dynamic alias system powered by REEP data."""

    def test_build_dynamic_aliases_returns_dict(self):
        """_build_dynamic_aliases always returns a dict (even if REEP unavailable)."""
        from backend.features.power_rankings import _build_dynamic_aliases

        result = _build_dynamic_aliases()
        self.assertIsInstance(result, dict)

    def test_get_merged_aliases_includes_hardcoded(self):
        """_get_merged_aliases must always include the hardcoded _EXTREME_ABBREVS."""
        from backend.features.power_rankings import (
            _EXTREME_ABBREVS,
            _get_merged_aliases,
        )

        merged = _get_merged_aliases()
        # All hardcoded keys must be present in merged
        for key in _EXTREME_ABBREVS:
            self.assertIn(key, merged)
            # All hardcoded aliases must be present
            for alias in _EXTREME_ABBREVS[key]:
                self.assertIn(alias, merged[key])

    def test_merged_aliases_psg_still_works(self):
        """PSG alias chain must survive merging with dynamic aliases."""
        from backend.features.power_rankings import _get_merged_aliases

        merged = _get_merged_aliases()
        self.assertIn("psg", merged)
        self.assertIn("parissaintgermain", merged["psg"])

    def test_merged_aliases_mancity_still_works(self):
        """ManCity alias chain must survive merging."""
        from backend.features.power_rankings import _get_merged_aliases

        merged = _get_merged_aliases()
        self.assertIn("mancity", merged)
        self.assertIn("manchestercity", merged["mancity"])

    def test_dynamic_aliases_with_mock_reep_data(self):
        """When REEP provides team data, dynamic aliases are built correctly."""
        import io
        from unittest.mock import patch

        import pandas as pd

        from backend.features import power_rankings

        # Mock REEP teams data with a few teams
        mock_data = pd.DataFrame({
            "name": ["Real Madrid CF", "FC Barcelona", "Olympique de Marseille"],
            "key_clubelo": ["RealMadrid", "Barcelona", "Marseille"],
            "key_fbref": ["Real-Madrid", "Barcelona", "Marseille"],
            "key_transfermarkt": ["real-madrid", "fc-barcelona", "olympique-marseille"],
        })

        # Reset cache to force rebuild
        old_cache = power_rankings._dynamic_aliases_cache
        power_rankings._dynamic_aliases_cache = None

        try:
            with patch(
                "backend.data.reep_registry.get_teams_df",
                return_value=mock_data,
            ):
                result = power_rankings._build_dynamic_aliases()

            # Should have created cross-links between normalized names
            self.assertIsInstance(result, dict)
            self.assertGreater(len(result), 0)

            # "olympiquedemarseille" (from name) should alias to
            # "marseille" (from key_clubelo) — these normalize differently
            # enough to produce cross-links
            has_any_link = any(
                "marseille" in k or "olympique" in k or "barcelona" in k
                for k in result
            )
            self.assertTrue(has_any_link, f"Expected aliases for known teams, got: {list(result.keys())[:20]}")
        finally:
            power_rankings._dynamic_aliases_cache = old_cache


if __name__ == "__main__":
    unittest.main()
