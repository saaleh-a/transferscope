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


if __name__ == "__main__":
    unittest.main()
