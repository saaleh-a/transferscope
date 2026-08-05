"""Tests for Sofascore player attribute overviews.

The open question was whether the 0-100 ratings are absolute or scaled within
position group, since that decides whether they are usable as model features.
Measured on unambiguous players, mean defending runs CB 78 -> DM 63 -> AM 34 ->
W 32 -> ST 27 while attacking runs the other way, which is only possible on a
shared scale.

The second trap was the ``position`` field appearing wrong: Saka's current-year
row reads "D". There are two different position fields, and the per-year one
describes the position played that year from a season still in progress.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.data import player_attributes as pa


def _payload(rows, average=None):
    return {
        "playerAttributeOverviews": rows,
        "averageAttributeOverviews": average or [],
    }


def _row(year_shift, position="M", **attrs):
    base = {
        "yearShift": year_shift,
        "position": position,
        "attacking": 70,
        "technical": 70,
        "tactical": 70,
        "defending": 40,
        "creativity": 70,
    }
    base.update(attrs)
    return base


class TestAttributeHistory(unittest.TestCase):
    def setUp(self):
        from backend.data import cache
        cache.clear_namespace("sofascore_attributes")

    def test_excludes_the_partial_current_season(self):
        """yearShift 0 covers a season in progress, so it is not comparable."""
        payload = _payload([_row(0), _row(1), _row(2)])
        with patch.object(pa, "_get", return_value=payload):
            history = pa.get_attribute_history(1)
        self.assertEqual([h["years_ago"] for h in history], [2, 1])

    def test_can_include_current_season_explicitly(self):
        payload = _payload([_row(0), _row(1)])
        with patch.object(pa, "_get", return_value=payload):
            history = pa.get_attribute_history(2, include_current_year=True)
        self.assertEqual([h["years_ago"] for h in history], [1, 0])
        self.assertTrue(history[-1]["is_partial_season"])

    def test_sorted_oldest_first(self):
        payload = _payload([_row(1), _row(3), _row(2)])
        with patch.object(pa, "_get", return_value=payload):
            history = pa.get_attribute_history(3)
        self.assertEqual([h["years_ago"] for h in history], [3, 2, 1])

    def test_missing_attribute_stays_none_not_zero(self):
        """Goalkeepers only receive tactical; zero would read as 'no ability'."""
        keeper = _row(1, "G", attacking=None, technical=None,
                      defending=None, creativity=None, tactical=84)
        with patch.object(pa, "_get", return_value=_payload([keeper])):
            history = pa.get_attribute_history(4)
        self.assertIsNone(history[0]["attacking"])
        self.assertEqual(history[0]["tactical"], 84.0)

    def test_bad_payloads_return_empty(self):
        for payload in (None, {}, {"playerAttributeOverviews": None}, "nope"):
            with patch.object(pa, "_get", return_value=payload):
                self.assertEqual(pa.get_attribute_history(5), [])

    def test_malformed_rows_are_skipped(self):
        payload = _payload([_row(1), "nonsense", {"no_year_shift": 1}])
        with patch.object(pa, "_get", return_value=payload):
            self.assertEqual(len(pa.get_attribute_history(6)), 1)

    def test_invalid_player_id(self):
        with patch.object(pa, "_get") as mock_get:
            self.assertEqual(pa.get_attribute_history(0), [])
        mock_get.assert_not_called()


class TestPositionalAverage(unittest.TestCase):
    def setUp(self):
        from backend.data import cache
        cache.clear_namespace("sofascore_attributes")

    def test_returns_the_comparison_row(self):
        """This position is the player's own, unlike the per-year rows."""
        payload = _payload([_row(1, "D")], average=[_row(0, "F", attacking=61)])
        with patch.object(pa, "_get", return_value=payload):
            avg = pa.get_positional_average(10)
        self.assertEqual(avg["position"], "F")
        self.assertEqual(avg["attacking"], 61.0)

    def test_missing_average_returns_none(self):
        with patch.object(pa, "_get", return_value=_payload([_row(1)])):
            self.assertIsNone(pa.get_positional_average(11))

    def test_invalid_id(self):
        self.assertIsNone(pa.get_positional_average(0))


class TestTrajectory(unittest.TestCase):
    def setUp(self):
        from backend.data import cache
        cache.clear_namespace("sofascore_attributes")

    def test_needs_two_seasons(self):
        """One point is not a trend."""
        with patch.object(pa, "_get", return_value=_payload([_row(1)])):
            self.assertEqual(pa.compute_trajectory(20), {})

    def test_computes_deltas(self):
        payload = _payload([
            _row(3, attacking=70),
            _row(2, attacking=75),
            _row(1, attacking=80),
        ])
        with patch.object(pa, "_get", return_value=payload):
            traj = pa.compute_trajectory(21)
        attacking = traj["attributes"]["attacking"]
        self.assertEqual(attacking["earliest"], 70.0)
        self.assertEqual(attacking["latest"], 80.0)
        self.assertEqual(attacking["delta"], 10.0)
        self.assertEqual(attacking["series"], [70.0, 75.0, 80.0])

    def test_flags_position_change(self):
        """A midfielder moved forward gains attacking without improving."""
        payload = _payload([_row(3, "M"), _row(2, "M"), _row(1, "F")])
        with patch.object(pa, "_get", return_value=payload):
            traj = pa.compute_trajectory(22)
        self.assertTrue(traj["position_changed"])

    def test_stable_position_is_not_flagged(self):
        payload = _payload([_row(3, "M"), _row(2, "M"), _row(1, "M")])
        with patch.object(pa, "_get", return_value=payload):
            self.assertFalse(pa.compute_trajectory(23)["position_changed"])

    def test_emits_no_categorical_label(self):
        """No 'Rising Star' verdicts — three points cannot support one."""
        payload = _payload([_row(3, attacking=60), _row(1, attacking=90)])
        with patch.object(pa, "_get", return_value=payload):
            traj = pa.compute_trajectory(24)
        serialised = str(traj).lower()
        for banned in ("rising", "declining", "star", "prospect"):
            self.assertNotIn(banned, serialised)

    def test_attribute_with_one_point_is_skipped(self):
        payload = _payload([
            _row(2, attacking=70, defending=None),
            _row(1, attacking=75, defending=50),
        ])
        with patch.object(pa, "_get", return_value=payload):
            traj = pa.compute_trajectory(25)
        self.assertIn("attacking", traj["attributes"])
        self.assertNotIn("defending", traj["attributes"])


if __name__ == "__main__":
    unittest.main()
