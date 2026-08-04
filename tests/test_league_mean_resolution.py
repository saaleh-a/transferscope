"""Tests for league-mean resolution in power rankings.

Our LEAGUES registry maps 51 leagues; Opta publishes 333. Roughly 97% of teams
therefore resolved to "UNK" and were compared against a global mean of ~51
instead of their own league. Since relative_ability = score - league_mean is a
model input, that turned a Brazilian Serie A club's +6 into +39.

Opta puts the league average on every team record as seasonAverageRating, so
the fix is to read it rather than infer it from however many teams of that
league happened to match by name.
"""

from __future__ import annotations

import unittest

from backend.data import opta_client


class TestOptaSeasonAverageCoverage(unittest.TestCase):
    """The field the fix relies on must actually be populated."""

    @classmethod
    def setUpClass(cls):
        # Served from the 7-day cache in normal runs; skip when unavailable
        # so the suite stays offline-safe.
        try:
            cls.teams = opta_client.get_team_rankings()
        except Exception:
            cls.teams = []

    def test_season_avg_rating_is_widely_populated(self):
        if not self.teams:
            self.skipTest("Opta bundle not cached")
        have = sum(1 for t in self.teams if t.season_avg_rating)
        self.assertGreater(have / len(self.teams), 0.95)

    def test_season_avg_matches_published_league_rating(self):
        """A team's seasonAverageRating should equal its league's rating."""
        if not self.teams:
            self.skipTest("Opta bundle not cached")
        leagues = opta_client.get_league_rankings()
        if not leagues:
            self.skipTest("league-meta not cached")

        by_country_name = {
            ((lr.country or "").lower(), (lr.league or "").lower()): lr.rating
            for lr in leagues
        }
        checked = 0
        for team in self.teams[:200]:
            if not (team.season_avg_rating and team.domestic_league and team.country):
                continue
            key = (team.country.lower(), team.domestic_league.lower())
            published = by_country_name.get(key)
            if published is None:
                continue
            self.assertAlmostEqual(
                float(team.season_avg_rating), published, places=1,
                msg=f"{team.club_name}: team says {team.season_avg_rating}, "
                    f"league-meta says {published}",
            )
            checked += 1
        if checked == 0:
            self.skipTest("no directly comparable league names in sample")


class TestLeagueMeanIsNotGlobalFallback(unittest.TestCase):
    """Clubs outside the mapped leagues must not share one global mean."""

    @classmethod
    def setUpClass(cls):
        from backend.features import power_rankings

        cls.power_rankings = power_rankings
        try:
            cls.rankings, cls.snapshots = power_rankings.compute_daily_rankings()
        except Exception:
            cls.rankings = {}

    def test_non_european_leagues_have_distinct_means(self):
        if not self.rankings:
            self.skipTest("rankings unavailable offline")

        samples = ["Flamengo", "Al Hilal", "LA Galaxy", "Boca Juniors"]
        means = []
        for club in samples:
            ranking = self.power_rankings.get_team_ranking(club)
            if ranking is not None:
                means.append(round(ranking.league_mean_normalized, 1))

        if len(means) < 3:
            self.skipTest("sample clubs not resolvable offline")

        # Before the fix these were all identical (the UNK global mean).
        self.assertGreater(
            len(set(means)), 1,
            f"all sampled leagues share one mean {means} — the global "
            "fallback has returned",
        )

    def test_relative_ability_is_plausible_outside_europe(self):
        """A domestic champion should sit single digits above its league."""
        if not self.rankings:
            self.skipTest("rankings unavailable offline")

        ranking = self.power_rankings.get_team_ranking("Flamengo")
        if ranking is None:
            self.skipTest("Flamengo not resolvable offline")

        # Comparing against the global ~51 mean produced roughly +39.
        self.assertLess(
            abs(ranking.relative_ability), 25.0,
            f"relative_ability={ranking.relative_ability:.1f} looks like the "
            "global-mean bug rather than a real league comparison",
        )

    def test_english_clubs_still_use_the_premier_league_mean(self):
        """The fix must not regress the leagues that already worked."""
        if not self.rankings:
            self.skipTest("rankings unavailable offline")

        ranking = self.power_rankings.get_team_ranking("Arsenal")
        if ranking is None:
            self.skipTest("Arsenal not resolvable offline")
        self.assertGreater(ranking.league_mean_normalized, 85.0)


if __name__ == "__main__":
    unittest.main()
