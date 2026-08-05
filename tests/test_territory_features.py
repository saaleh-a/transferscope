"""Tests for Sofascore heatmap territory features.

These replace a dead spatial path: the WhoScored client returns 404/406 on
every endpoint, and StatsBomb open data covers only a fraction of current
players. The Sofascore season heatmap is the working source.

The axis-orientation tests matter most. Sofascore's y-axis runs right-to-left
from the attacking player's perspective, so labelling it the intuitive way
round reports every winger on the wrong flank.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.data import sofascore_client as sc


def _points(*triples):
    """Build heatmap points from (x, y, count) triples."""
    return [{"x": x, "y": y, "count": c} for x, y, c in triples]


class TestGetPlayerHeatmap(unittest.TestCase):
    def test_parses_points(self):
        payload = {"points": [{"x": 50, "y": 40, "count": 3}], "matches": 10}
        with patch.object(sc, "_get", return_value=payload):
            points = sc.get_player_heatmap(1, 17, 76986)
        self.assertEqual(points, [{"x": 50, "y": 40, "count": 3}])

    def test_defaults_missing_count_to_one(self):
        with patch.object(sc, "_get", return_value={"points": [{"x": 1, "y": 2}]}):
            points = sc.get_player_heatmap(1, 17, 76986)
        self.assertEqual(points[0]["count"], 1)

    def test_skips_malformed_points(self):
        payload = {"points": [
            {"x": 10, "y": 20, "count": 2},
            {"y": 5},                      # no x
            "nonsense",                    # not a dict
            {"x": "abc", "y": 1},          # unparseable
        ]}
        with patch.object(sc, "_get", return_value=payload):
            points = sc.get_player_heatmap(1, 17, 76986)
        self.assertEqual(len(points), 1)

    def test_bad_payloads_return_empty(self):
        for payload in (None, {}, {"points": None}, {"points": "x"}, "nope"):
            with patch.object(sc, "_get", return_value=payload):
                self.assertEqual(sc.get_player_heatmap(1, 17, 76986), [])

    def test_invalid_ids_short_circuit(self):
        with patch.object(sc, "_get") as mock_get:
            self.assertEqual(sc.get_player_heatmap(0, 17, 76986), [])
            self.assertEqual(sc.get_player_heatmap(1, 0, 76986), [])
            self.assertEqual(sc.get_player_heatmap(1, 17, 0), [])
        mock_get.assert_not_called()


class TestTerritoryFeatures(unittest.TestCase):
    @staticmethod
    def _features(points):
        with patch.object(sc, "get_player_heatmap", return_value=points):
            return sc.compute_territory_features(1, 17, 76986)

    def test_no_heatmap_returns_empty_not_zeros(self):
        """Absent data must be distinguishable from 'never enters final third'."""
        self.assertEqual(self._features([]), {})

    def test_zero_total_count_returns_empty(self):
        self.assertEqual(self._features(_points((50, 50, 0))), {})

    def test_thirds_sum_to_one(self):
        feats = self._features(_points(
            (10, 50, 5), (50, 50, 5), (90, 50, 5),
        ))
        total = (
            feats["territory_final_third"]
            + feats["territory_middle_third"]
            + feats["territory_own_third"]
        )
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_lanes_sum_to_one(self):
        feats = self._features(_points(
            (50, 10, 3), (50, 50, 3), (50, 90, 3),
        ))
        total = (
            feats["territory_left"]
            + feats["territory_central"]
            + feats["territory_right"]
        )
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_low_y_is_the_right_flank(self):
        """Verified against real data: Saka (right winger) sits 84% in low y."""
        feats = self._features(_points((60, 10, 10)))
        self.assertEqual(feats["territory_right"], 1.0)
        self.assertEqual(feats["territory_left"], 0.0)

    def test_high_y_is_the_left_flank(self):
        """Verified against real data: Martinelli (left winger) sits 60% in high y."""
        feats = self._features(_points((60, 90, 10)))
        self.assertEqual(feats["territory_left"], 1.0)
        self.assertEqual(feats["territory_right"], 0.0)

    def test_counts_are_weighted(self):
        """A cell visited 9 times must outweigh one visited once."""
        feats = self._features(_points((90, 50, 9), (10, 50, 1)))
        self.assertAlmostEqual(feats["territory_final_third"], 0.9)
        self.assertAlmostEqual(feats["territory_own_third"], 0.1)

    def test_box_is_a_subset_of_the_final_third(self):
        feats = self._features(_points((70, 50, 5), (90, 50, 5)))
        self.assertLessEqual(feats["territory_box"], feats["territory_final_third"])
        self.assertAlmostEqual(feats["territory_box"], 0.5)

    def test_goalkeeper_profile(self):
        """A keeper must read as own-third and central, with no box presence."""
        feats = self._features(_points(
            (5, 50, 50), (10, 45, 30), (15, 55, 20),
        ))
        self.assertGreater(feats["territory_own_third"], 0.95)
        self.assertEqual(feats["territory_box"], 0.0)
        self.assertGreater(feats["territory_central"], 0.95)
        self.assertLess(feats["territory_avg_x"], 20)

    def test_average_position(self):
        feats = self._features(_points((20, 30, 1), (80, 70, 1)))
        self.assertAlmostEqual(feats["territory_avg_x"], 50.0)
        self.assertAlmostEqual(feats["territory_avg_y"], 50.0)

    def test_width_reflects_lateral_spread(self):
        narrow = self._features(_points((50, 49, 1), (50, 51, 1)))
        wide = self._features(_points((50, 5, 1), (50, 95, 1)))
        self.assertLess(narrow["territory_width"], wide["territory_width"])

    def test_all_fractions_are_bounded(self):
        feats = self._features(_points(
            (0, 0, 3), (50, 50, 7), (100, 100, 2),
        ))
        for key, value in feats.items():
            if key.startswith("territory_avg") or key == "territory_width":
                continue
            self.assertGreaterEqual(value, 0.0, key)
            self.assertLessEqual(value, 1.0, key)


class TestWhoScoredIsRemoved(unittest.TestCase):
    """The dead client is deleted, not merely labelled."""

    def test_module_is_gone(self):
        with self.assertRaises(ImportError):
            from backend.data import whoscored_client  # noqa: F401

    def test_transfer_impact_no_longer_references_it(self):
        import pathlib

        page = pathlib.Path("frontend/pages/transfer_impact.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("whoscored_client", page)


if __name__ == "__main__":
    unittest.main()
