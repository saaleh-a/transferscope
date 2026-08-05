"""League-ability resolution must land on the right league.

Every value here was taken from the live Opta bundle (2026-08-05, 13,791 teams
/ 333 leagues). Before the fix:

    get_league_opta_rating('ENG1', 'Manchester City') = 65.50   (Bahrain!)
    get_league_opta_rating('ENG1', 'Arsenal')         = 60.43   (Guadeloupe!)
    get_league_opta_rating(league_code='GER1')        = 45.02   (a German 6th tier)
    get_league_opta_rating(league_code='ITA1')        = 53.29   ('serie d')

Two independent causes:

1. Opta's *team* records name leagues short ("Premier League") while
   league-meta names them long ("English Premier League"), so the
   country-qualified lookup missed for every Big-5 club and fell through to
   "highest-rated league of that bare name anywhere in the world".
2. The team-name index was last-write-wins over a rank-ascending list, so for
   genuine homonyms the *worst*-ranked club on earth won the key.

The feature this corrupts (``league_ability_current`` / ``_target`` and the
four derived relative-ability features) appears in all six model groups.
"""

from __future__ import annotations

import pytest

from backend.data.opta_client import OptaLeagueRanking, OptaTeamRanking


def _team(rank, name, rating, league, country, season_avg, short="", club=""):
    return OptaTeamRanking(
        rank=rank,
        team=name,
        rating=rating,
        ranking_change_7d="0",
        opta_id=f"id{rank}",
        short_name=short,
        club_name=club,
        domestic_league=league,
        domestic_league_id="",
        country=country,
        confederation="",
        season_avg_rating=season_avg,
    )


def _league(rank, name, rating, country, n_teams=20):
    return OptaLeagueRanking(
        rank=rank,
        league=name,
        rating=rating,
        ranking_change_7d="0",
        country=country,
        number_of_teams=n_teams,
    )


# Rank-ascending, exactly as the real bundle ships it. Note that the
# Guadeloupe and Ecuador clubs come *after* their famous namesakes.
TEAMS = [
    _team(1, "Manchester City", 95.0, "Premier League", "England", 92.91),
    _team(2, "Arsenal", 94.5, "Premier League", "England", 92.91),
    _team(5, "Barcelona", 93.0, "La Liga", "Spain", 87.10),
    _team(9, "Internazionale", 91.0, "Serie A", "Italy", 87.02),
    _team(12, "Paris Saint-Germain", 90.0, "Ligue 1", "France", 86.24),
    _team(2186, "Arsenal", 65.0, "Premier League", "Belarus", 62.00),
    _team(4100, "Barcelona", 61.0, "Liga Pro", "Ecuador", 79.14),
    _team(6336, "Arsenal", 58.0, "Division d'Honneur", "Guadeloupe", 53.94),
]

# "Premier League" and "Ligue 1" exist as bare names in other countries and
# outrank nothing — but the old name-only map picked them anyway.
LEAGUES = [
    _league(1, "English Premier League", 92.91, "England"),
    _league(2, "German Bundesliga", 87.14, "Germany"),
    _league(3, "Spanish La Liga", 87.10, "Spain"),
    _league(4, "Italian Serie A", 87.02, "Italy"),
    _league(5, "French Ligue 1", 86.24, "France"),
    _league(20, "English Football League - Championship", 81.22, "England"),
    _league(24, "Spanish Segunda Division", 78.79, "Spain"),
    _league(28, "German Bundesliga Zwei", 77.84, "Germany"),
    _league(31, "Italian Serie B", 75.49, "Italy"),
    _league(33, "French Ligue 2", 75.35, "France"),
    _league(60, "Premier League", 65.50, "Bahrain"),
    _league(70, "Ligue 1", 69.52, "Tunisia"),
    _league(140, "Landesliga", 45.02, "Germany"),
    _league(155, "Serie D", 53.29, "Italy"),
    _league(180, "Division d'Honneur", 53.94, "Guadeloupe"),
]


@pytest.fixture(autouse=True)
def _opta(monkeypatch):
    """Serve the fixture bundle and reset every lazily-built module global."""
    from backend.data import opta_client
    from backend.features import power_rankings as pr

    monkeypatch.setattr(opta_client, "get_team_rankings", lambda *a, **k: list(TEAMS))
    monkeypatch.setattr(opta_client, "get_league_rankings", lambda *a, **k: list(LEAGUES))

    for name in (
        "_opta_alias_map",
        "_opta_team_league_map",
        "_opta_league_map",
        "_opta_league_country_map",
        "_opta_league_team_counts",
        "_opta_country_leagues",
    ):
        monkeypatch.setattr(pr, name, None, raising=False)
    monkeypatch.setattr(pr, "_league_code_opta_rating_cache", {}, raising=False)
    yield


class TestTeamNameResolution:
    """A club must get its own league's rating, not a namesake's."""

    @pytest.mark.parametrize(
        "code,club,expected",
        [
            ("ENG1", "Manchester City", 92.91),
            ("ENG1", "Arsenal", 92.91),
            ("ESP1", "Barcelona", 87.10),
            ("ITA1", "Internazionale", 87.02),
            ("FRA1", "Paris Saint-Germain", 86.24),
        ],
    )
    def test_big5_clubs_get_their_own_league(self, code, club, expected):
        from backend.features.power_rankings import get_league_opta_rating

        assert get_league_opta_rating(code, club) == pytest.approx(expected)

    def test_clubs_in_the_same_league_agree(self):
        """The regression that made the feature meaningless.

        Arsenal read 60.43 and Manchester City 65.50 — same league, same
        season, two different values, both wrong.
        """
        from backend.features.power_rankings import get_league_opta_rating

        assert get_league_opta_rating("ENG1", "Arsenal") == get_league_opta_rating(
            "ENG1", "Manchester City"
        )

    def test_best_ranked_homonym_wins(self):
        """Arsenal (England, rank 2) must beat Arsenal (Guadeloupe, rank 6336)."""
        from backend.features.power_rankings import get_league_opta_rating

        assert get_league_opta_rating(team_name="Arsenal") == pytest.approx(92.91)
        assert get_league_opta_rating(team_name="Barcelona") == pytest.approx(87.10)

    def test_never_returns_a_foreign_bare_name_league(self):
        from backend.features.power_rankings import get_league_opta_rating

        assert get_league_opta_rating("ENG1", "Manchester City") != pytest.approx(65.50)
        assert get_league_opta_rating("FRA1", "Paris Saint-Germain") != pytest.approx(69.52)


class TestLeagueCodeResolution:
    """The no-team fallback must not fuzzy-match into a lower division."""

    @pytest.mark.parametrize(
        "code,expected",
        [
            ("ENG1", 92.91),
            ("GER1", 87.14),
            ("ESP1", 87.10),
            ("ITA1", 87.02),
            ("FRA1", 86.24),
        ],
    )
    def test_top_flights(self, code, expected):
        from backend.features.power_rankings import get_league_opta_rating

        assert get_league_opta_rating(league_code=code) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "code,expected",
        [
            ("ENG2", 81.22),
            ("ESP2", 78.79),
            ("GER2", 77.84),
            ("ITA2", 75.49),
        ],
    )
    def test_second_tiers_resolve_to_the_second_tier(self, code, expected):
        from backend.features.power_rankings import get_league_opta_rating

        assert get_league_opta_rating(league_code=code) == pytest.approx(expected)

    def test_bundesliga_does_not_match_landesliga(self):
        """SequenceMatcher scored these 0.80 — a top flight vs a sixth tier."""
        from backend.features.power_rankings import get_league_opta_rating

        assert get_league_opta_rating(league_code="GER1") != pytest.approx(45.02)

    def test_serie_a_does_not_match_serie_d(self):
        from backend.features.power_rankings import get_league_opta_rating

        assert get_league_opta_rating(league_code="ITA1") != pytest.approx(53.29)

    def test_first_tier_always_outranks_second_tier(self):
        from backend.features.power_rankings import get_league_opta_rating

        for first, second in [("ENG1", "ENG2"), ("ESP1", "ESP2"),
                              ("GER1", "GER2"), ("ITA1", "ITA2")]:
            assert get_league_opta_rating(league_code=first) > get_league_opta_rating(
                league_code=second
            ), f"{first} should rate above {second}"


class TestTrainServeParity:
    """Training and inference must agree on league ability.

    Training reads ``get_league_opta_rating``; the Streamlit pages pass
    ``TeamRanking.league_mean_normalized``, which comes from the team's own
    ``seasonAverageRating``. Those were 20-35 points apart on a 0-100 feature,
    which put every served prediction ~2 sigma out of distribution.
    """

    @pytest.mark.parametrize(
        "code,club",
        [
            ("ENG1", "Manchester City"),
            ("ESP1", "Barcelona"),
            ("ITA1", "Internazionale"),
            ("FRA1", "Paris Saint-Germain"),
        ],
    )
    def test_training_path_matches_the_served_season_average(self, code, club):
        from backend.features.power_rankings import get_league_opta_rating

        served = next(
            t.season_avg_rating
            for t in TEAMS
            if t.team == club and t.rank < 100
        )
        assert get_league_opta_rating(code, club) == pytest.approx(served)


class TestMapConstruction:
    def test_alias_map_is_published_atomically(self):
        """Built into a local and assigned once.

        Streamlit runs each rerun on a worker thread against these plain module
        globals. Publishing an empty dict *before* a 13,791-iteration fill lets a
        second thread read a truncated map and keep it for the life of the
        process, because it is no longer ``None``.
        """
        import inspect

        from backend.features import power_rankings as pr

        src = inspect.getsource(pr._get_opta_alias_map)
        publish = src.index("_opta_alias_map = alias_map")
        loop = src.index("for t in opta_client.get_team_rankings()")
        assert loop < publish, "map must be filled before it is published"

    def test_every_registry_league_resolves(self):
        """No registry league may silently fall back to the 50.0 midpoint."""
        from backend.features.power_rankings import get_league_opta_rating
        from backend.utils.league_registry import LEAGUES as REGISTRY

        # Only the countries present in this fixture can resolve; assert on those.
        covered = {lg.country for lg in LEAGUES}
        checked = 0
        for code, info in REGISTRY.items():
            if info.country not in covered:
                continue
            checked += 1
            assert get_league_opta_rating(league_code=code) != pytest.approx(50.0), (
                f"{code} ({info.name}) fell back to the scale midpoint"
            )
        assert checked >= 8
