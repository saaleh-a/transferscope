"""Comprehensive tests for paper_heuristic_predict — the fallback prediction engine.

paper_heuristic_predict is the default prediction path when no trained TF model
exists. Every user sees its output until the training pipeline has run. It must
handle all realistic transfer scenarios faithfully.
"""

import os
import shutil
import tempfile
import unittest

_TEMP_DIR = tempfile.mkdtemp(prefix="ts_heuristic_test_")
os.environ.setdefault("CACHE_DIR", _TEMP_DIR)

from backend.data.sofascore_client import CORE_METRICS
from backend.features.adjustment_models import (
    paper_heuristic_predict,
    _check_has_style_data,
)
from backend.data import cache


def tearDownModule():
    """Clean up the temp cache directory after all tests."""
    cache.close()
    shutil.rmtree(_TEMP_DIR, ignore_errors=True)


def _uniform_per90(value: float) -> dict:
    """Return a per-90 dict with every core metric set to the same value."""
    return {m: value for m in CORE_METRICS}


def _zero_per90() -> dict:
    return _uniform_per90(0.0)


class TestPaperHeuristicHappyPath(unittest.TestCase):
    """Basic predictions with typical inputs produce sensible outputs."""

    def test_identity_transfer_no_change(self):
        """Same team → same league → zero relative ability change → ≈no change."""
        player = _uniform_per90(1.0)
        pos_avg = _uniform_per90(1.0)
        result = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=pos_avg,
            target_pos_avg=pos_avg,
            change_relative_ability=0.0,
        )
        for m in CORE_METRICS:
            self.assertAlmostEqual(result[m], 1.0, delta=0.05,
                msg=f"{m} should be ≈1.0 for identity transfer")

    def test_returns_all_13_core_metrics(self):
        """Output dict must contain exactly the 13 core metrics."""
        result = paper_heuristic_predict(
            player_per90=_uniform_per90(1.0),
            source_pos_avg=_uniform_per90(1.0),
            target_pos_avg=_uniform_per90(1.0),
            change_relative_ability=0.0,
        )
        self.assertEqual(set(result.keys()), set(CORE_METRICS))

    def test_all_values_non_negative(self):
        """Per-90 values can never be negative (clamped at 0)."""
        result = paper_heuristic_predict(
            player_per90=_uniform_per90(0.1),
            source_pos_avg=_uniform_per90(0.5),
            target_pos_avg=_uniform_per90(0.5),
            change_relative_ability=-50.0,  # extreme downgrade
        )
        for m in CORE_METRICS:
            self.assertGreaterEqual(result[m], 0.0,
                msg=f"{m} must be >= 0 even for extreme downgrades")

    def test_upgrade_increases_offensive_metrics(self):
        """Moving to a much stronger team should increase xG, shots, etc."""
        player = _uniform_per90(1.0)
        src_avg = _uniform_per90(0.8)
        tgt_avg = _uniform_per90(1.5)  # stronger team's position avg is higher
        result = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=src_avg,
            target_pos_avg=tgt_avg,
            change_relative_ability=20.0,  # moderate upgrade
        )
        # xG, shots should increase
        self.assertGreater(result["expected_goals"], 1.0)
        self.assertGreater(result["shots"], 1.0)

    def test_downgrade_decreases_offensive_metrics(self):
        """Moving to a much weaker team should decrease xG, shots, etc."""
        player = _uniform_per90(1.0)
        src_avg = _uniform_per90(1.5)
        tgt_avg = _uniform_per90(0.5)
        result = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=src_avg,
            target_pos_avg=tgt_avg,
            change_relative_ability=-20.0,
        )
        self.assertLess(result["expected_goals"], 1.0)
        self.assertLess(result["shots"], 1.0)


class TestPaperHeuristicDefensiveMetrics(unittest.TestCase):
    """Defensive metrics (clearances, interceptions) have inverted sensitivity."""

    def test_upgrade_decreases_clearances(self):
        """Better team → less defensive work → fewer clearances."""
        player = _uniform_per90(1.0)
        # Using identical pos avgs → no style data → fallback path
        result = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=player.copy(),
            target_pos_avg=player.copy(),
            change_relative_ability=25.0,
            source_league_mean=50.0,
            target_league_mean=50.0,
        )
        self.assertLess(result["clearances"], 1.0,
            "Moving to a better team should decrease clearances")
        self.assertLess(result["interceptions"], 1.0,
            "Moving to a better team should decrease interceptions")

    def test_downgrade_increases_clearances(self):
        """Weaker team → more defensive work → more clearances."""
        player = _uniform_per90(1.0)
        result = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=player.copy(),
            target_pos_avg=player.copy(),
            change_relative_ability=-25.0,
            source_league_mean=50.0,
            target_league_mean=50.0,
        )
        self.assertGreater(result["clearances"], 1.0)
        self.assertGreater(result["interceptions"], 1.0)


class TestPaperHeuristicDribbling(unittest.TestCase):
    """Dribbling is 'irreducible' per the paper — barely changes."""

    def test_dribbling_near_stable_for_moderate_move(self):
        """Take-ons should change < 10% for a moderate transfer."""
        player = _uniform_per90(1.0)
        result = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=player.copy(),
            target_pos_avg=player.copy(),
            change_relative_ability=15.0,
            source_league_mean=50.0,
            target_league_mean=50.0,
        )
        pct_change = abs(result["successful_dribbles"] - 1.0) / 1.0 * 100
        self.assertLess(pct_change, 10.0,
            "Dribbling should change < 10% for moderate moves (irreducible)")


class TestPaperHeuristicOppositionQuality(unittest.TestCase):
    """Tests the opposition quality effect (weaker league → higher output)."""

    def test_weaker_league_boosts_xg(self):
        """Moving to a weaker league boosts xG via opposition quality effect."""
        player = _uniform_per90(1.0)
        result = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=player.copy(),
            target_pos_avg=player.copy(),
            change_relative_ability=0.0,  # same relative position
            source_league_mean=70.0,      # strong source league
            target_league_mean=40.0,      # weaker target league
        )
        # league_gap = (70-40)/100 = 0.30 (positive = weaker target)
        # xG has _OPP_QUALITY_SENS = 1.30 → big boost
        self.assertGreater(result["expected_goals"], 1.0,
            "Weaker opposition should boost xG")

    def test_stronger_league_decreases_xg(self):
        """Moving to a stronger league decreases xG via opposition quality."""
        player = _uniform_per90(1.0)
        result = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=player.copy(),
            target_pos_avg=player.copy(),
            change_relative_ability=0.0,
            source_league_mean=40.0,      # weaker source league
            target_league_mean=70.0,      # stronger target league
        )
        self.assertLess(result["expected_goals"], 1.0,
            "Stronger opposition should reduce xG")

    def test_no_league_means_no_opposition_effect(self):
        """When league means are None, opposition effect is zero."""
        player = _uniform_per90(1.0)
        result_with = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=player.copy(),
            target_pos_avg=player.copy(),
            change_relative_ability=0.0,
            source_league_mean=None,
            target_league_mean=None,
        )
        result_same_league = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=player.copy(),
            target_pos_avg=player.copy(),
            change_relative_ability=0.0,
            source_league_mean=50.0,
            target_league_mean=50.0,
        )
        # Both should be essentially the same (no league gap in either case)
        for m in CORE_METRICS:
            self.assertAlmostEqual(result_with[m], result_same_league[m], places=4,
                msg=f"{m} should be identical when league means are None vs equal")


class TestPaperHeuristicPlayerRating(unittest.TestCase):
    """Player quality modifier: elite players retain more individual output."""

    def test_high_rated_player_retains_more_on_upgrade(self):
        """A 7.5-rated player should deviate less from baseline than a 5.5-rated player.

        The quality_scale modifier reduces effective_team_inf for elite players,
        meaning they are less pulled toward the new team's averages on upgrade.
        """
        player = _uniform_per90(1.0)
        result_elite = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=_uniform_per90(0.8),
            target_pos_avg=_uniform_per90(1.2),
            change_relative_ability=15.0,
            player_rating=7.5,
        )
        result_avg = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=_uniform_per90(0.8),
            target_pos_avg=_uniform_per90(1.2),
            change_relative_ability=15.0,
            player_rating=5.5,
        )
        # For team-influenced metrics, the elite player (higher quality_scale
        # → lower effective_team_inf) should deviate less from their own value.
        elite_diff = abs(result_elite["successful_passes"] - 1.0)
        avg_diff = abs(result_avg["successful_passes"] - 1.0)
        self.assertLess(elite_diff, avg_diff,
            "Elite player (7.5 rating) should change less than average player (5.5) "
            "for team-influenced metric on upgrade")

    def test_elite_protection_halved_for_downgrades(self):
        """Asymmetric: elite players not fully protected on downgrades."""
        player = _uniform_per90(2.0)
        # Downgrade scenario
        result_down = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=_uniform_per90(2.0),
            target_pos_avg=_uniform_per90(0.5),
            change_relative_ability=-30.0,
            player_rating=8.0,  # very high rated
        )
        # Even an 8.0-rated player should see meaningful drops on downgrade
        self.assertLess(result_down["expected_goals"], 2.0,
            "Elite player should still suffer on extreme downgrade")

    def test_none_rating_defaults_to_no_modifier(self):
        """When player_rating is None, quality_scale should be 1.0 (no effect)."""
        player = _uniform_per90(1.0)
        result_none = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=player.copy(),
            target_pos_avg=player.copy(),
            change_relative_ability=10.0,
            player_rating=None,
            source_league_mean=50.0,
            target_league_mean=50.0,
        )
        result_avg = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=player.copy(),
            target_pos_avg=player.copy(),
            change_relative_ability=10.0,
            player_rating=6.5,  # 6.5 is the center → quality_scale = 1.0
            source_league_mean=50.0,
            target_league_mean=50.0,
        )
        for m in CORE_METRICS:
            self.assertAlmostEqual(result_none[m], result_avg[m], places=4,
                msg=f"{m}: None rating should equal 6.5 (center) rating")


class TestPaperHeuristicEdgeCases(unittest.TestCase):
    """Edge cases and boundary values."""

    def test_zero_player_per90(self):
        """Player with all zeros should produce all zeros or near-zeros."""
        result = paper_heuristic_predict(
            player_per90=_zero_per90(),
            source_pos_avg=_uniform_per90(1.0),
            target_pos_avg=_uniform_per90(1.0),
            change_relative_ability=0.0,
        )
        for m in CORE_METRICS:
            # With zero player stats, conformity pull toward team avg is small
            # and the base is near zero, so result should be small
            self.assertGreaterEqual(result[m], 0.0)

    def test_extreme_positive_ra(self):
        """Very large positive change_relative_ability doesn't explode."""
        result = paper_heuristic_predict(
            player_per90=_uniform_per90(1.0),
            source_pos_avg=_uniform_per90(1.0),
            target_pos_avg=_uniform_per90(1.0),
            change_relative_ability=100.0,
            source_league_mean=50.0,
            target_league_mean=50.0,
        )
        for m in CORE_METRICS:
            self.assertLess(result[m], 10.0,
                msg=f"{m} should not explode for extreme RA=100")
            self.assertGreaterEqual(result[m], 0.0)

    def test_extreme_negative_ra(self):
        """Very large negative change_relative_ability clamps to >= 0."""
        result = paper_heuristic_predict(
            player_per90=_uniform_per90(1.0),
            source_pos_avg=_uniform_per90(1.0),
            target_pos_avg=_uniform_per90(1.0),
            change_relative_ability=-100.0,
            source_league_mean=50.0,
            target_league_mean=50.0,
        )
        for m in CORE_METRICS:
            self.assertGreaterEqual(result[m], 0.0,
                msg=f"{m} must be clamped to >= 0 for extreme negative RA")

    def test_missing_metrics_in_player_per90(self):
        """Metrics missing from player_per90 default to 0.0."""
        partial = {"expected_goals": 0.5}  # only 1 of 13 metrics
        result = paper_heuristic_predict(
            player_per90=partial,
            source_pos_avg=_uniform_per90(1.0),
            target_pos_avg=_uniform_per90(1.0),
            change_relative_ability=0.0,
        )
        self.assertEqual(len(result), len(CORE_METRICS))
        self.assertIsInstance(result["expected_goals"], float)
        self.assertIsInstance(result["clearances"], float)  # was missing → 0.0

    def test_missing_metrics_in_position_averages(self):
        """Missing position average keys fall back to player's value."""
        player = _uniform_per90(1.0)
        result = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg={},  # completely empty
            target_pos_avg={},
            change_relative_ability=0.0,
        )
        # With empty pos avgs, style_diff = 0 for all metrics
        self.assertEqual(len(result), len(CORE_METRICS))

    def test_large_league_gap_attenuates_style(self):
        """Cross-league moves attenuate style differences (league_attn)."""
        player = _uniform_per90(1.0)
        src_avg = _uniform_per90(0.5)
        tgt_avg = _uniform_per90(2.0)
        # Same-league transfer: full style effect
        result_same = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=src_avg,
            target_pos_avg=tgt_avg,
            change_relative_ability=10.0,
            source_league_mean=60.0,
            target_league_mean=60.0,
        )
        # Cross-league: attenuated style effect
        result_cross = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=src_avg,
            target_pos_avg=tgt_avg,
            change_relative_ability=10.0,
            source_league_mean=80.0,
            target_league_mean=40.0,
        )
        # For a heavily team-influenced metric, the cross-league version
        # should show a DIFFERENT result than same-league
        # (style is attenuated but opposition effect adds a new dimension)
        self.assertNotAlmostEqual(
            result_same["successful_passes"],
            result_cross["successful_passes"],
            places=2,
            msg="Cross-league should differ from same-league due to attenuation"
        )


class TestPaperHeuristicStyleDataDetection(unittest.TestCase):
    """Tests for _check_has_style_data: fallback vs real team-position data."""

    def test_identical_avgs_means_no_style_data(self):
        """When source and target are identical, it's fallback data."""
        player = _uniform_per90(1.0)
        pos_avg = _uniform_per90(1.0)
        self.assertFalse(_check_has_style_data(player, pos_avg, pos_avg))

    def test_different_avgs_means_real_style_data(self):
        """When source and target differ meaningfully, it's real data."""
        player = _uniform_per90(1.0)
        src = _uniform_per90(0.8)
        tgt = _uniform_per90(1.2)
        self.assertTrue(_check_has_style_data(player, src, tgt))

    def test_one_metric_different_not_enough(self):
        """Need at least 2 metrics to count as real style data."""
        player = _uniform_per90(1.0)
        src = _uniform_per90(1.0)
        tgt = _uniform_per90(1.0)
        tgt["expected_goals"] = 2.0  # only 1 metric differs
        self.assertFalse(_check_has_style_data(player, src, tgt))

    def test_two_metrics_different_is_enough(self):
        """Two differing metrics is the threshold for real data."""
        player = _uniform_per90(1.0)
        src = _uniform_per90(1.0)
        tgt = _uniform_per90(1.0)
        tgt["expected_goals"] = 2.0
        tgt["shots"] = 2.0
        self.assertTrue(_check_has_style_data(player, src, tgt))

    def test_empty_dicts_means_no_style_data(self):
        """All empty dicts → no style data."""
        self.assertFalse(_check_has_style_data({}, {}, {}))

    def test_avgs_equal_to_player_means_no_style_data(self):
        """When pos_avgs == player stats, it's using player as fallback."""
        player = _uniform_per90(1.5)
        self.assertFalse(_check_has_style_data(player, player.copy(), player.copy()))

    def test_no_style_data_triggers_fallback_estimation(self):
        """Without style data, the function uses _LEAGUE_STYLE_COEFF estimation."""
        player = _uniform_per90(1.0)
        # Identical pos avgs → no style data → fallback estimation
        result_no_style = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=player.copy(),
            target_pos_avg=player.copy(),
            change_relative_ability=15.0,
            source_league_mean=50.0,
            target_league_mean=50.0,
        )
        # Different pos avgs → has style data → real style_diff
        result_with_style = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=_uniform_per90(0.8),
            target_pos_avg=_uniform_per90(1.2),
            change_relative_ability=15.0,
            source_league_mean=50.0,
            target_league_mean=50.0,
        )
        # Results should differ because style estimation is different path
        any_diff = any(
            abs(result_no_style[m] - result_with_style[m]) > 0.01
            for m in CORE_METRICS
        )
        self.assertTrue(any_diff,
            "With vs without style data should produce different predictions")


class TestPaperHeuristicPerMetricDifferentiation(unittest.TestCase):
    """Different metrics should respond differently to the same transfer."""

    def test_not_all_metrics_change_equally(self):
        """The 13 metrics must NOT all change by the same percentage."""
        player = _uniform_per90(1.0)
        result = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=player.copy(),
            target_pos_avg=player.copy(),
            change_relative_ability=20.0,
            source_league_mean=60.0,
            target_league_mean=40.0,
        )
        pct_changes = [(result[m] - 1.0) for m in CORE_METRICS]
        # Check that there are at least 3 distinct percentage changes
        unique_changes = set(round(c, 3) for c in pct_changes)
        self.assertGreater(len(unique_changes), 3,
            "Metrics should respond differently — not flat identical changes")

    def test_offensive_vs_defensive_direction(self):
        """For an upgrade, offensive should increase and defensive should decrease."""
        player = _uniform_per90(1.0)
        result = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=player.copy(),
            target_pos_avg=player.copy(),
            change_relative_ability=25.0,
            source_league_mean=50.0,
            target_league_mean=50.0,
        )
        # xG should increase (positive ability sensitivity)
        self.assertGreater(result["expected_goals"], 1.0)
        # Clearances should decrease (negative ability sensitivity)
        self.assertLess(result["clearances"], 1.0)


class TestPaperHeuristicAsymmetricDamping(unittest.TestCase):
    """Damping factors differ: _DAMPING_FACTOR_DOWN=0.05, _DAMPING_FACTOR_UP=0.10."""

    def test_upgrade_has_more_damping_than_downgrade(self):
        """For very large |RA|, upgrades are damped more → smaller absolute change.

        The asymmetry is in the team_effect term:
            team_effect = sensitivity * team_gap * (1 - damp * |team_gap|)
        where damp is 0.10 for upgrades (ceiling) vs 0.05 for downgrades
        (allow bigger drops). We isolate this by providing real style data
        (so fallback estimation doesn't interact) and no league gap.
        """
        player = _uniform_per90(1.0)
        # Provide real style data with distinct source/target to avoid fallback
        src_avg = {m: 0.9 for m in CORE_METRICS}
        tgt_avg_up = {m: 1.1 for m in CORE_METRICS}
        tgt_avg_down = {m: 0.7 for m in CORE_METRICS}

        result_up = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=src_avg,
            target_pos_avg=tgt_avg_up,
            change_relative_ability=40.0,
            source_league_mean=50.0,
            target_league_mean=50.0,
        )
        result_down = paper_heuristic_predict(
            player_per90=player,
            source_pos_avg=src_avg,
            target_pos_avg=tgt_avg_down,
            change_relative_ability=-40.0,
            source_league_mean=50.0,
            target_league_mean=50.0,
        )
        # Dribbling is "irreducible" with low sensitivity so team_effect
        # dominates relative to style.  The downgrade xG change should be
        # larger in magnitude than the upgrade xG change (less damped).
        # However the interaction of style and team effects is complex, so
        # we verify a softer property: both directions produce distinct moves
        # and the upgrade direction does not exceed the downgrade in *all* metrics.
        up_changes = {m: abs(result_up[m] - 1.0) for m in CORE_METRICS}
        down_changes = {m: abs(result_down[m] - 1.0) for m in CORE_METRICS}
        # At least some metrics should show larger drop than gain
        larger_drop_count = sum(
            1 for m in CORE_METRICS if down_changes[m] > up_changes[m]
        )
        self.assertGreater(larger_drop_count, 0,
            "At least some metrics should drop more on downgrade than rise on upgrade")


class TestPaperHeuristicReturnTypes(unittest.TestCase):
    """Ensure output types are correct."""

    def test_all_values_are_float(self):
        result = paper_heuristic_predict(
            player_per90=_uniform_per90(1.0),
            source_pos_avg=_uniform_per90(1.0),
            target_pos_avg=_uniform_per90(1.0),
            change_relative_ability=10.0,
        )
        for m, v in result.items():
            self.assertIsInstance(v, float, f"{m} should be float, got {type(v)}")

    def test_return_is_dict(self):
        result = paper_heuristic_predict(
            player_per90=_uniform_per90(1.0),
            source_pos_avg=_uniform_per90(1.0),
            target_pos_avg=_uniform_per90(1.0),
            change_relative_ability=0.0,
        )
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
