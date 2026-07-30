"""Tests for the Value Opportunity Score.

The guiding rule for this module is that missing data must never be imputed as
zero — an unknown market value must not make a player look cheap.
"""

from __future__ import annotations

import unittest

from backend.models.value_score import (
    MAX_USEFUL_CONTRACT_YEARS,
    PEAK_AGE,
    VALUE_WEIGHTS,
    ValueCandidate,
    _age_runway,
    _contract_leverage,
    _position_group,
    composite_output,
    percentile_ranks,
    score_candidates,
)


class TestPercentileRanks(unittest.TestCase):
    def test_orders_from_zero_to_one(self):
        self.assertEqual(percentile_ranks([10, 20, 30]), [0.0, 0.5, 1.0])

    def test_preserves_none_positions(self):
        out = percentile_ranks([10, None, 30])
        self.assertIsNone(out[1])
        self.assertEqual(out[0], 0.0)
        self.assertEqual(out[2], 1.0)

    def test_all_none_returns_all_none(self):
        self.assertEqual(percentile_ranks([None, None]), [None, None])

    def test_single_value_is_midpoint(self):
        self.assertEqual(percentile_ranks([42]), [0.5])

    def test_ties_share_average_rank(self):
        out = percentile_ranks([5, 5, 9])
        self.assertEqual(out[0], out[1])
        self.assertGreater(out[2], out[0])

    def test_robust_to_extreme_outlier(self):
        """A single huge value must not flatten the rest (min-max would)."""
        out = percentile_ranks([1, 2, 3, 1_000_000])
        self.assertGreater(out[1], 0.0)
        self.assertGreater(out[2], out[1])

    def test_empty_input(self):
        self.assertEqual(percentile_ranks([]), [])


class TestCompositeOutput(unittest.TestCase):
    def test_weighted_average(self):
        result = composite_output(
            {"expected_goals": 1.0, "shots": 3.0},
            {"expected_goals": 0.5, "shots": 0.5},
        )
        self.assertAlmostEqual(result, 2.0)

    def test_missing_metrics_are_skipped_not_zeroed(self):
        both = composite_output(
            {"expected_goals": 1.0, "shots": 3.0},
            {"expected_goals": 1.0, "shots": 1.0},
        )
        one = composite_output(
            {"expected_goals": 1.0},
            {"expected_goals": 1.0, "shots": 1.0},
        )
        self.assertAlmostEqual(one, 1.0)
        self.assertNotAlmostEqual(one, both)

    def test_no_data_returns_none(self):
        self.assertIsNone(composite_output({}, {"expected_goals": 1.0}))


class TestContractLeverage(unittest.TestCase):
    def test_expiring_contract_is_max_leverage(self):
        self.assertAlmostEqual(_contract_leverage(0.0), 1.0)

    def test_long_contract_is_zero_leverage(self):
        self.assertAlmostEqual(_contract_leverage(MAX_USEFUL_CONTRACT_YEARS), 0.0)

    def test_beyond_cap_does_not_go_negative(self):
        self.assertAlmostEqual(_contract_leverage(25.0), 0.0)

    def test_unknown_contract_is_none_not_zero(self):
        self.assertIsNone(_contract_leverage(None))

    def test_monotonically_decreasing(self):
        a = _contract_leverage(1.0)
        b = _contract_leverage(3.0)
        self.assertGreater(a, b)


class TestAgeRunway(unittest.TestCase):
    def test_young_player_has_runway(self):
        self.assertGreater(_age_runway(18), 0.5)

    def test_peak_age_has_no_runway(self):
        self.assertEqual(_age_runway(PEAK_AGE), 0.0)

    def test_past_peak_is_zero_not_negative(self):
        self.assertEqual(_age_runway(34), 0.0)

    def test_unknown_age_is_none(self):
        self.assertIsNone(_age_runway(None))

    def test_bad_input_is_none(self):
        self.assertIsNone(_age_runway("abc"))
        self.assertIsNone(_age_runway(0))


class TestScoreCandidates(unittest.TestCase):
    @staticmethod
    def _cand(pid, name, **kwargs):
        base = dict(
            market_value=20_000_000.0,
            contract_years_left=3.0,
            age=24,
            output=0.5,
            projected_improvement_pct=5.0,
        )
        base.update(kwargs)
        return ValueCandidate(player_id=pid, name=name, **base)

    def test_empty_input(self):
        self.assertEqual(score_candidates([]), [])

    def test_cheaper_player_with_same_output_scores_higher(self):
        cheap = self._cand(1, "Cheap", market_value=5_000_000.0)
        dear = self._cand(2, "Expensive", market_value=80_000_000.0)
        scores = {s.name: s.score for s in score_candidates([cheap, dear])}
        self.assertGreater(scores["Cheap"], scores["Expensive"])

    def test_expiring_contract_scores_higher(self):
        expiring = self._cand(1, "Expiring", contract_years_left=0.5)
        tied = self._cand(2, "Tied", contract_years_left=5.0)
        scores = {s.name: s.score for s in score_candidates([expiring, tied])}
        self.assertGreater(scores["Expiring"], scores["Tied"])

    def test_missing_market_value_is_not_rewarded(self):
        """The critical guard: unknown price must never look like a bargain."""
        unknown = self._cand(1, "Unknown", market_value=None)
        known_cheap = self._cand(2, "KnownCheap", market_value=1_000_000.0)
        results = {s.name: s for s in score_candidates([unknown, known_cheap])}
        self.assertIn("output_per_value", results["Unknown"].missing)
        self.assertGreater(
            results["KnownCheap"].score, results["Unknown"].score
        )

    def test_zero_market_value_is_treated_as_unknown(self):
        zero = self._cand(1, "Zero", market_value=0.0)
        results = score_candidates([zero, self._cand(2, "Other")])
        zero_result = next(r for r in results if r.name == "Zero")
        # Must not divide by zero or produce inf
        self.assertIn("output_per_value", zero_result.missing)

    def test_insufficient_coverage_returns_none_score(self):
        sparse = ValueCandidate(player_id=1, name="Sparse", age=20)
        results = score_candidates([sparse], min_coverage=0.5)
        self.assertIsNone(results[0].score)
        self.assertLess(results[0].coverage, 0.5)

    def test_results_are_sorted_with_unscoreable_last(self):
        sparse = ValueCandidate(player_id=9, name="Sparse")
        good = self._cand(1, "Good", market_value=2_000_000.0)
        mid = self._cand(2, "Mid", market_value=60_000_000.0)
        results = score_candidates([sparse, mid, good])
        self.assertEqual(results[0].name, "Good")
        self.assertIsNone(results[-1].score)

    def test_reasons_are_populated_and_human_readable(self):
        results = score_candidates(
            [self._cand(1, "A", market_value=2_000_000.0), self._cand(2, "B")]
        )
        top = results[0]
        self.assertTrue(top.reasons)
        self.assertLessEqual(len(top.reasons), 2)
        for reason in top.reasons:
            self.assertIsInstance(reason, str)
            self.assertNotIn("_", reason)  # display text, not a field name

    def test_score_is_bounded_0_to_100(self):
        cands = [
            self._cand(1, "A", market_value=1_000_000.0, contract_years_left=0.0, age=16),
            self._cand(2, "B", market_value=200_000_000.0, contract_years_left=5.0, age=36),
        ]
        for result in score_candidates(cands):
            if result.score is not None:
                self.assertGreaterEqual(result.score, 0.0)
                self.assertLessEqual(result.score, 100.0)

    def test_partial_data_is_renormalised_not_diluted(self):
        """A player missing one input should not be scored as if it were zero."""
        full = self._cand(1, "Full")
        no_contract = self._cand(2, "NoContract", contract_years_left=None)
        results = {s.name: s for s in score_candidates([full, no_contract])}
        self.assertIn("contract_leverage", results["NoContract"].missing)
        self.assertIsNotNone(results["NoContract"].score)
        self.assertLess(results["NoContract"].coverage, 1.0)

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(VALUE_WEIGHTS.values()), 1.0)

    def test_components_are_exposed_for_auditing(self):
        result = score_candidates([self._cand(1, "A"), self._cand(2, "B")])[0]
        self.assertTrue(set(result.components).issubset(set(VALUE_WEIGHTS)))


class TestPositionGrouping(unittest.TestCase):
    def test_maps_common_labels(self):
        self.assertEqual(_position_group("G"), "Goalkeeper")
        self.assertEqual(_position_group("Goalkeeper"), "Goalkeeper")
        self.assertEqual(_position_group("D"), "Defender")
        self.assertEqual(_position_group("Midfielder"), "Midfielder")
        self.assertEqual(_position_group("F"), "Forward")
        self.assertEqual(_position_group("Attacker"), "Forward")

    def test_unknown_and_empty(self):
        self.assertEqual(_position_group(None), "Unknown")
        self.assertEqual(_position_group(""), "Unknown")
        self.assertEqual(_position_group("???"), "Unknown")

    def test_by_position_scores_within_cohort(self):
        """A cheap keeper must not out-rank forwards on attacking output.

        This is the exact failure seen on real data: goalkeepers have tiny
        market values, so output-per-euro explodes and they top an attacking
        ranking. Grouping compares like with like.
        """
        keeper = ValueCandidate(
            player_id=1, name="CheapKeeper", position="G",
            market_value=500_000.0, contract_years_left=1.0, age=33,
            output=0.05, projected_improvement_pct=0.0,
        )
        forwards = [
            ValueCandidate(
                player_id=10 + i, name=f"Fwd{i}", position="F",
                market_value=40_000_000.0 + i * 1_000_000,
                contract_years_left=3.0, age=25,
                output=0.8 - i * 0.05, projected_improvement_pct=4.0,
            )
            for i in range(3)
        ]

        flat = score_candidates([keeper] + forwards, by_position=False)
        self.assertEqual(flat[0].name, "CheapKeeper")  # the bug

        grouped = score_candidates([keeper] + forwards, by_position=True)
        by_name = {r.name: r for r in grouped}
        # Every player is still scored...
        self.assertEqual(len(grouped), 4)
        # ...but the keeper is now scored against keepers only, so its
        # output_per_value percentile is mid-cohort rather than best-in-show.
        self.assertEqual(by_name["CheapKeeper"].components["output_per_value"], 0.5)
        # ...and the best forward beats the worst forward within its cohort.
        self.assertGreater(by_name["Fwd0"].score, by_name["Fwd2"].score)

    def test_by_position_preserves_all_candidates(self):
        cands = [
            ValueCandidate(player_id=i, name=f"P{i}", position=p,
                           market_value=10_000_000.0, contract_years_left=2.0,
                           age=24, output=0.4, projected_improvement_pct=1.0)
            for i, p in enumerate(["G", "D", "M", "F", "", "D"])
        ]
        grouped = score_candidates(cands, by_position=True)
        self.assertEqual(len(grouped), len(cands))
        self.assertEqual({r.name for r in grouped}, {c.name for c in cands})

    def test_by_position_is_sorted_descending(self):
        cands = [
            ValueCandidate(player_id=i, name=f"P{i}", position="F",
                           market_value=(i + 1) * 10_000_000.0,
                           contract_years_left=2.0, age=24, output=0.5,
                           projected_improvement_pct=1.0)
            for i in range(4)
        ]
        scores = [r.score for r in score_candidates(cands, by_position=True)]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
