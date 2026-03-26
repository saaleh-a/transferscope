"""Unit tests for shortlist_scorer: filtering, scoring, and edge cases."""

import os
import tempfile
import unittest

_TEMP_DIR = tempfile.mkdtemp(prefix="ts_shortlist_test_")
os.environ.setdefault("CACHE_DIR", _TEMP_DIR)

from backend.models.shortlist_scorer import (
    Candidate,
    ShortlistFilters,
    compute_percentage_changes,
    filter_candidates,
    score_candidates,
)
from backend.data.sofascore_client import CORE_METRICS


def _make_candidate(**kwargs) -> Candidate:
    """Create a Candidate with sensible defaults for testing."""
    defaults = {
        "player_id": 1,
        "name": "Test Player",
        "team": "Test FC",
        "position": "Forward",
        "age": 25,
        "minutes_played": 1000,
        "league": "Premier League",
        "club_power_ranking": 70.0,
        "predicted_per90": {m: 0.5 for m in CORE_METRICS},
        "current_per90": {m: 0.5 for m in CORE_METRICS},
        "rating": 7.0,
    }
    defaults.update(kwargs)
    return Candidate(**defaults)


class TestFilterCandidates(unittest.TestCase):
    """Test that filter_candidates correctly applies all filter types."""

    def test_no_filters_passes_all(self):
        """All candidates pass when no filters are set."""
        candidates = [_make_candidate(player_id=i) for i in range(5)]
        filters = ShortlistFilters()
        result = filter_candidates(candidates, filters)
        self.assertEqual(len(result), 5)

    def test_max_age_filters_old_players(self):
        young = _make_candidate(player_id=1, age=22)
        old = _make_candidate(player_id=2, age=35)
        filters = ShortlistFilters(max_age=30)
        result = filter_candidates([young, old], filters)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].player_id, 1)

    def test_none_age_passes_through_age_filter(self):
        """Candidates with unknown age should NOT be excluded."""
        unknown = _make_candidate(player_id=1, age=None)
        young = _make_candidate(player_id=2, age=22)
        filters = ShortlistFilters(max_age=30)
        result = filter_candidates([unknown, young], filters)
        self.assertEqual(len(result), 2)

    def test_min_minutes_filters_low_minutes(self):
        high = _make_candidate(player_id=1, minutes_played=1500)
        low = _make_candidate(player_id=2, minutes_played=100)
        filters = ShortlistFilters(min_minutes_played=500)
        result = filter_candidates([high, low], filters)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].player_id, 1)

    def test_none_minutes_passes_through(self):
        """Candidates with unknown minutes should NOT be excluded."""
        unknown = _make_candidate(player_id=1, minutes_played=None)
        filters = ShortlistFilters(min_minutes_played=500)
        result = filter_candidates([unknown], filters)
        self.assertEqual(len(result), 1)

    def test_position_filter(self):
        fwd = _make_candidate(player_id=1, position="Forward")
        mid = _make_candidate(player_id=2, position="Midfielder")
        filters = ShortlistFilters(positions=["Forward"])
        result = filter_candidates([fwd, mid], filters)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].player_id, 1)

    def test_league_filter_known_leagues(self):
        epl = _make_candidate(player_id=1, league="Premier League")
        laliga = _make_candidate(player_id=2, league="La Liga")
        filters = ShortlistFilters(leagues=["Premier League"])
        result = filter_candidates([epl, laliga], filters)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].player_id, 1)

    def test_empty_league_passes_through_league_filter(self):
        """Candidates with empty league pass through (data incomplete)."""
        no_league = _make_candidate(player_id=1, league="")
        epl = _make_candidate(player_id=2, league="Premier League")
        filters = ShortlistFilters(leagues=["Premier League"])
        result = filter_candidates([no_league, epl], filters)
        self.assertEqual(len(result), 2)

    def test_none_league_passes_through(self):
        """Candidates with None league pass through."""
        no_league = _make_candidate(player_id=1, league=None)
        filters = ShortlistFilters(leagues=["Premier League"])
        result = filter_candidates([no_league], filters)
        self.assertEqual(len(result), 1)

    def test_power_ranking_filter(self):
        strong = _make_candidate(player_id=1, club_power_ranking=80.0)
        weak = _make_candidate(player_id=2, club_power_ranking=30.0)
        filters = ShortlistFilters(max_power_ranking=50.0)
        result = filter_candidates([strong, weak], filters)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].player_id, 2)


class TestComputePercentageChanges(unittest.TestCase):
    """Test compute_percentage_changes edge cases."""

    def test_normal_change(self):
        current = {"expected_goals": 0.5}
        predicted = {"expected_goals": 0.6}
        changes = compute_percentage_changes(current, predicted)
        self.assertAlmostEqual(changes["expected_goals"], 20.0, places=1)

    def test_zero_current_returns_zero(self):
        current = {"expected_goals": 0.0}
        predicted = {"expected_goals": 0.3}
        changes = compute_percentage_changes(current, predicted)
        self.assertEqual(changes["expected_goals"], 0.0)

    def test_none_values_treated_as_zero(self):
        current = {"expected_goals": None}
        predicted = {"expected_goals": None}
        changes = compute_percentage_changes(current, predicted)
        self.assertEqual(changes["expected_goals"], 0.0)


class TestScoreCandidates(unittest.TestCase):
    """Test score_candidates with reference player."""

    def test_similar_player_scores_higher(self):
        """A player with identical stats should score higher than a different one."""
        reference = {m: 1.0 for m in CORE_METRICS}
        similar = _make_candidate(
            player_id=1,
            predicted_per90={m: 1.0 for m in CORE_METRICS},
        )
        different = _make_candidate(
            player_id=2,
            predicted_per90={m: 5.0 for m in CORE_METRICS},
        )
        weights = {m: 1.0 for m in CORE_METRICS}
        scored = score_candidates(
            [similar, different], weights, reference_per90=reference
        )
        self.assertEqual(scored[0].player_id, 1)
        self.assertGreater(scored[0].score, scored[1].score)

    def test_empty_candidates_returns_empty(self):
        weights = {m: 1.0 for m in CORE_METRICS}
        scored = score_candidates([], weights, reference_per90={m: 0 for m in CORE_METRICS})
        self.assertEqual(len(scored), 0)


if __name__ == "__main__":
    unittest.main()
