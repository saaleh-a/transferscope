"""Tests for backend.data.statsbomb_client — StatsBomb open-data client."""

from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd


# ── Sample data ──────────────────────────────────────────────────────────────

_COMPETITIONS_DF = pd.DataFrame([
    {"competition_id": 11, "competition_name": "La Liga", "season_id": 90, "season_name": "2019/2020"},
    {"competition_id": 43, "competition_name": "FIFA World Cup", "season_id": 3, "season_name": "2018"},
])

_MATCHES_DF = pd.DataFrame([
    {"match_id": 100, "home_team": "Barcelona", "away_team": "Real Madrid"},
    {"match_id": 101, "home_team": "Valencia", "away_team": "Barcelona"},
])

_EMPTY_DF = pd.DataFrame()

_SHOT_EVENT = {
    "player": "Lionel Messi",
    "type": "Shot",
    "location": [105.0, 35.0],
    "shot_statsbomb_xg": 0.32,
    "shot_outcome": "Goal",
    "shot_body_part": "Left Foot",
    "shot_technique": "Normal",
    "minute": 23,
}

_PASS_EVENT = {
    "player": "Lionel Messi",
    "type": "Pass",
    "location": [50.0, 40.0],
    "pass_end_location": [75.0, 35.0],
    "pass_outcome": "Complete",
    "pass_length": 26.0,
    "pass_angle": 0.2,
    "pass_type": "Ground Pass",
    "pass_recipient": "Luis Suárez",
    "minute": 10,
}

_CARRY_EVENT = {
    "player": "Lionel Messi",
    "type": "Carry",
    "location": [60.0, 40.0],
    "carry_end_location": [70.0, 42.0],
    "minute": 15,
}

_TACKLE_EVENT = {
    "player": "Lionel Messi",
    "type": "Tackle",
    "location": [30.0, 25.0],
    "minute": 55,
}

_OTHER_PLAYER_EVENT = {
    "player": "Sergio Busquets",
    "type": "Pass",
    "location": [40.0, 40.0],
    "pass_end_location": [55.0, 38.0],
    "pass_outcome": "Complete",
    "minute": 5,
}


def _make_events_df(events):
    """Build a DataFrame from a list of event dicts, like statsbombpy returns."""
    return pd.DataFrame(events)


# ═════════════════════════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestGetAvailableCompetitions(unittest.TestCase):
    """get_available_competitions() fetches and caches competition list."""

    @patch("backend.data.statsbomb_client._import_sb")
    @patch("backend.data.statsbomb_client.cache")
    def test_returns_competition_list(self, mock_cache, mock_sb_fn):
        from backend.data.statsbomb_client import get_available_competitions

        mock_cache.get.return_value = None
        mock_sb = MagicMock()
        mock_sb_fn.return_value = mock_sb
        mock_sb.competitions.return_value = _COMPETITIONS_DF

        result = get_available_competitions()

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["competition_name"], "La Liga")
        self.assertEqual(result[1]["season_name"], "2018")
        mock_cache.set.assert_called_once()

    @patch("backend.data.statsbomb_client._import_sb")
    @patch("backend.data.statsbomb_client.cache")
    def test_cache_hit_returns_cached(self, mock_cache, mock_sb_fn):
        from backend.data.statsbomb_client import get_available_competitions

        cached_data = [{"competition_id": 11, "competition_name": "La Liga",
                        "season_id": 90, "season_name": "2019/2020"}]
        mock_cache.get.return_value = cached_data

        result = get_available_competitions()

        self.assertEqual(result, cached_data)
        mock_sb_fn.assert_not_called()

    @patch("backend.data.statsbomb_client._import_sb")
    @patch("backend.data.statsbomb_client.cache")
    def test_sb_not_installed_returns_empty(self, mock_cache, mock_sb_fn):
        from backend.data.statsbomb_client import get_available_competitions

        mock_cache.get.return_value = None
        mock_sb_fn.return_value = None  # statsbombpy not installed

        result = get_available_competitions()

        self.assertEqual(result, [])

    @patch("backend.data.statsbomb_client._import_sb")
    @patch("backend.data.statsbomb_client.cache")
    def test_empty_df_returns_empty(self, mock_cache, mock_sb_fn):
        from backend.data.statsbomb_client import get_available_competitions

        mock_cache.get.return_value = None
        mock_sb = MagicMock()
        mock_sb_fn.return_value = mock_sb
        mock_sb.competitions.return_value = _EMPTY_DF

        result = get_available_competitions()

        self.assertEqual(result, [])

    @patch("backend.data.statsbomb_client._import_sb")
    @patch("backend.data.statsbomb_client.cache")
    def test_exception_returns_empty(self, mock_cache, mock_sb_fn):
        from backend.data.statsbomb_client import get_available_competitions

        mock_cache.get.return_value = None
        mock_sb = MagicMock()
        mock_sb_fn.return_value = mock_sb
        mock_sb.competitions.side_effect = RuntimeError("network error")

        result = get_available_competitions()

        self.assertEqual(result, [])


class TestGetPlayerEvents(unittest.TestCase):
    """get_player_events() gathers events across competitions & matches."""

    @patch("backend.data.statsbomb_client._import_sb")
    @patch("backend.data.statsbomb_client.cache")
    def test_returns_player_events(self, mock_cache, mock_sb_fn):
        from backend.data.statsbomb_client import get_player_events

        events_df = _make_events_df([_SHOT_EVENT, _PASS_EVENT, _OTHER_PLAYER_EVENT])

        mock_cache.get.return_value = None
        mock_sb = MagicMock()
        mock_sb_fn.return_value = mock_sb
        mock_sb.competitions.return_value = _COMPETITIONS_DF
        mock_sb.matches.return_value = _MATCHES_DF
        mock_sb.events.return_value = events_df

        result = get_player_events("Lionel Messi", competition_id=11, season_id=90)

        # Should only contain Messi events, not Busquets
        self.assertGreater(len(result), 0)
        for ev in result:
            player = str(ev.get("player", "")).lower()
            self.assertIn("messi", player)

    @patch("backend.data.statsbomb_client._import_sb")
    @patch("backend.data.statsbomb_client.cache")
    def test_cache_hit_returns_cached(self, mock_cache, mock_sb_fn):
        from backend.data.statsbomb_client import get_player_events

        cached_events = [{"type": "Shot", "location": [100, 40], "match_id": 1}]
        mock_cache.get.return_value = cached_events

        result = get_player_events("Lionel Messi")

        self.assertEqual(result, cached_events)
        mock_sb_fn.assert_not_called()

    @patch("backend.data.statsbomb_client._import_sb")
    @patch("backend.data.statsbomb_client.cache")
    def test_sb_not_installed_returns_empty(self, mock_cache, mock_sb_fn):
        from backend.data.statsbomb_client import get_player_events

        mock_cache.get.return_value = None
        mock_sb_fn.return_value = None

        result = get_player_events("Lionel Messi")

        self.assertEqual(result, [])

    @patch("backend.data.statsbomb_client._import_sb")
    @patch("backend.data.statsbomb_client.cache")
    def test_no_matches_returns_empty(self, mock_cache, mock_sb_fn):
        from backend.data.statsbomb_client import get_player_events

        mock_cache.get.return_value = None
        mock_sb = MagicMock()
        mock_sb_fn.return_value = mock_sb
        mock_sb.competitions.return_value = _COMPETITIONS_DF
        mock_sb.matches.return_value = _EMPTY_DF

        result = get_player_events("Lionel Messi", competition_id=11, season_id=90)

        self.assertEqual(result, [])


class TestGetPlayerShots(unittest.TestCase):
    """get_player_shots() extracts shot events with spatial data."""

    @patch("backend.data.statsbomb_client._import_sb")
    @patch("backend.data.statsbomb_client.cache")
    def test_extracts_shots(self, mock_cache, mock_sb_fn):
        from backend.data.statsbomb_client import get_player_shots

        events_df = _make_events_df([_SHOT_EVENT, _PASS_EVENT])
        mock_cache.get.return_value = None
        mock_sb = MagicMock()
        mock_sb_fn.return_value = mock_sb
        mock_sb.competitions.return_value = _COMPETITIONS_DF
        mock_sb.matches.return_value = _MATCHES_DF
        mock_sb.events.return_value = events_df

        shots = get_player_shots("Lionel Messi", competition_id=11, season_id=90)

        self.assertGreater(len(shots), 0)
        shot = shots[0]
        self.assertIn("x", shot)
        self.assertIn("y", shot)
        self.assertIn("xg", shot)
        self.assertIn("outcome", shot)
        self.assertEqual(shot["outcome"], "Goal")
        self.assertAlmostEqual(shot["xg"], 0.32, places=2)

    @patch("backend.data.statsbomb_client.get_player_events")
    def test_no_shot_events_returns_empty(self, mock_events):
        from backend.data.statsbomb_client import get_player_shots

        mock_events.return_value = [
            {"type": "Pass", "location": [50, 40], "match_id": 1}
        ]

        shots = get_player_shots("Test Player")

        self.assertEqual(shots, [])

    @patch("backend.data.statsbomb_client.get_player_events")
    def test_shot_without_location_skipped(self, mock_events):
        from backend.data.statsbomb_client import get_player_shots

        mock_events.return_value = [
            {"type": "Shot", "location": None, "shot_statsbomb_xg": 0.1,
             "shot_outcome": "Saved", "match_id": 1}
        ]

        shots = get_player_shots("Test Player")

        self.assertEqual(shots, [])

    @patch("backend.data.statsbomb_client.get_player_events")
    def test_empty_events_returns_empty(self, mock_events):
        from backend.data.statsbomb_client import get_player_shots

        mock_events.return_value = []

        shots = get_player_shots("Test Player")

        self.assertEqual(shots, [])


class TestGetPlayerPasses(unittest.TestCase):
    """get_player_passes() extracts pass events with start/end locations."""

    @patch("backend.data.statsbomb_client.get_player_events")
    def test_extracts_passes(self, mock_events):
        from backend.data.statsbomb_client import get_player_passes

        mock_events.return_value = [
            {
                "type": "Pass",
                "location": [50.0, 40.0],
                "pass_end_location": [75.0, 35.0],
                "pass_outcome": "Complete",
                "pass_length": 26.0,
                "pass_angle": 0.2,
                "pass_type": "Ground Pass",
                "pass_recipient": "Luis Suárez",
                "minute": 10,
                "match_id": 100,
            },
        ]

        passes = get_player_passes("Lionel Messi")

        self.assertEqual(len(passes), 1)
        p = passes[0]
        self.assertAlmostEqual(p["start_x"], 50.0)
        self.assertAlmostEqual(p["start_y"], 40.0)
        self.assertAlmostEqual(p["end_x"], 75.0)
        self.assertAlmostEqual(p["end_y"], 35.0)
        self.assertEqual(p["outcome"], "Complete")

    @patch("backend.data.statsbomb_client.get_player_events")
    def test_pass_without_start_location_skipped(self, mock_events):
        from backend.data.statsbomb_client import get_player_passes

        mock_events.return_value = [
            {"type": "Pass", "location": None, "pass_end_location": [75, 35],
             "pass_outcome": "Complete", "match_id": 1}
        ]

        passes = get_player_passes("Test Player")

        self.assertEqual(passes, [])

    @patch("backend.data.statsbomb_client.get_player_events")
    def test_pass_without_end_location_defaults_to_zero(self, mock_events):
        from backend.data.statsbomb_client import get_player_passes

        mock_events.return_value = [
            {"type": "Pass", "location": [50.0, 40.0], "pass_end_location": None,
             "pass_outcome": "Incomplete", "match_id": 1}
        ]

        passes = get_player_passes("Test Player")

        self.assertEqual(len(passes), 1)
        self.assertAlmostEqual(passes[0]["end_x"], 0.0)
        self.assertAlmostEqual(passes[0]["end_y"], 0.0)

    @patch("backend.data.statsbomb_client.get_player_events")
    def test_empty_events_returns_empty(self, mock_events):
        from backend.data.statsbomb_client import get_player_passes

        mock_events.return_value = []

        passes = get_player_passes("Test Player")

        self.assertEqual(passes, [])


class TestGetPlayerHeatmapData(unittest.TestCase):
    """get_player_heatmap_data() extracts (x,y) tuples from all events."""

    @patch("backend.data.statsbomb_client.get_player_events")
    def test_extracts_locations(self, mock_events):
        from backend.data.statsbomb_client import get_player_heatmap_data

        mock_events.return_value = [
            {"type": "Shot", "location": [105.0, 35.0], "match_id": 1},
            {"type": "Pass", "location": [50.0, 40.0], "match_id": 1},
            {"type": "Carry", "location": [60.0, 45.0], "match_id": 1},
        ]

        points = get_player_heatmap_data("Test Player")

        self.assertEqual(len(points), 3)
        self.assertEqual(points[0], (105.0, 35.0))
        self.assertEqual(points[1], (50.0, 40.0))

    @patch("backend.data.statsbomb_client.get_player_events")
    def test_skips_events_without_location(self, mock_events):
        from backend.data.statsbomb_client import get_player_heatmap_data

        mock_events.return_value = [
            {"type": "Shot", "location": [100.0, 40.0], "match_id": 1},
            {"type": "Pass", "location": None, "match_id": 1},
            {"type": "Carry", "match_id": 1},  # no location key at all
        ]

        points = get_player_heatmap_data("Test Player")

        self.assertEqual(len(points), 1)

    @patch("backend.data.statsbomb_client.get_player_events")
    def test_empty_events_returns_empty(self, mock_events):
        from backend.data.statsbomb_client import get_player_heatmap_data

        mock_events.return_value = []

        points = get_player_heatmap_data("Test Player")

        self.assertEqual(points, [])


class TestComputeSpatialFeatures(unittest.TestCase):
    """compute_spatial_features() computes aggregate spatial stats."""

    @patch("backend.data.statsbomb_client.get_player_events")
    def test_computes_shot_features(self, mock_events):
        from backend.data.statsbomb_client import compute_spatial_features

        # Shot inside the box (x > 102)
        mock_events.return_value = [
            {"type": "Shot", "location": [110.0, 40.0], "match_id": 1},
            {"type": "Shot", "location": [90.0, 30.0], "match_id": 1},
        ]

        features = compute_spatial_features("Test Player")

        self.assertIn("avg_shot_distance", features)
        self.assertIn("shots_inside_box_pct", features)
        # One shot inside box out of two = 50%
        self.assertAlmostEqual(features["shots_inside_box_pct"], 50.0, places=1)

    @patch("backend.data.statsbomb_client.get_player_events")
    def test_computes_pass_features(self, mock_events):
        from backend.data.statsbomb_client import compute_spatial_features

        # Progressive pass (end_x - start_x > 10)
        mock_events.return_value = [
            {"type": "Pass", "location": [40.0, 40.0],
             "pass_end_location": [65.0, 38.0], "match_id": 1},
            {"type": "Pass", "location": [50.0, 40.0],
             "pass_end_location": [55.0, 42.0], "match_id": 1},
        ]

        features = compute_spatial_features("Test Player")

        self.assertIn("avg_pass_length", features)
        self.assertIn("progressive_pass_pct", features)
        # 1 progressive out of 2 = 50%
        self.assertAlmostEqual(features["progressive_pass_pct"], 50.0, places=1)

    @patch("backend.data.statsbomb_client.get_player_events")
    def test_computes_carry_features(self, mock_events):
        from backend.data.statsbomb_client import compute_spatial_features

        mock_events.return_value = [
            {"type": "Carry", "location": [60.0, 40.0],
             "carry_end_location": [70.0, 40.0], "match_id": 1},
        ]

        features = compute_spatial_features("Test Player")

        self.assertIn("avg_carry_distance", features)
        self.assertAlmostEqual(features["avg_carry_distance"], 10.0, places=1)

    @patch("backend.data.statsbomb_client.get_player_events")
    def test_computes_touch_distribution(self, mock_events):
        from backend.data.statsbomb_client import compute_spatial_features

        # left third: y < 80/3 ≈ 26.67
        # center third: 26.67 <= y < 53.33
        # right third: y >= 53.33
        mock_events.return_value = [
            {"type": "Pass", "location": [50.0, 10.0],
             "pass_end_location": [60.0, 10.0], "match_id": 1},   # left
            {"type": "Shot", "location": [100.0, 40.0], "match_id": 1},  # center
            {"type": "Carry", "location": [60.0, 70.0],
             "carry_end_location": [65.0, 70.0], "match_id": 1},  # right
        ]

        features = compute_spatial_features("Test Player")

        self.assertIn("touches_left_pct", features)
        self.assertIn("touches_center_pct", features)
        self.assertIn("touches_right_pct", features)
        # Each third gets one touch → 33.33% each
        self.assertAlmostEqual(features["touches_left_pct"], 33.33, places=1)
        self.assertAlmostEqual(features["touches_center_pct"], 33.33, places=1)
        self.assertAlmostEqual(features["touches_right_pct"], 33.33, places=1)

    @patch("backend.data.statsbomb_client.get_player_events")
    def test_computes_defensive_features(self, mock_events):
        from backend.data.statsbomb_client import compute_spatial_features

        mock_events.return_value = [
            {"type": "Tackle", "location": [30.0, 40.0], "match_id": 1},
        ]

        features = compute_spatial_features("Test Player")

        self.assertIn("avg_defensive_distance", features)
        # Distance from own goal (0, 40) to (30, 40) = 30
        self.assertAlmostEqual(features["avg_defensive_distance"], 30.0, places=1)

    @patch("backend.data.statsbomb_client.get_player_events")
    def test_empty_events_returns_empty_dict(self, mock_events):
        from backend.data.statsbomb_client import compute_spatial_features

        mock_events.return_value = []

        features = compute_spatial_features("Test Player")

        self.assertEqual(features, {})

    @patch("backend.data.statsbomb_client.get_player_events")
    def test_all_features_computed_together(self, mock_events):
        from backend.data.statsbomb_client import compute_spatial_features

        mock_events.return_value = [
            _SHOT_EVENT,
            _PASS_EVENT,
            _CARRY_EVENT,
            _TACKLE_EVENT,
        ]

        # Patch the nested call to _matches_player — not needed since
        # compute_spatial_features just iterates events directly.
        features = compute_spatial_features("Lionel Messi")

        # Shot at (105, 35) is inside box (x > 102)
        self.assertAlmostEqual(features["shots_inside_box_pct"], 100.0, places=1)

        # Pass from (50, 40) to (75, 35): progressive (75-50=25 > 10)
        self.assertAlmostEqual(features["progressive_pass_pct"], 100.0, places=1)

        # Carry from (60, 40) to (70, 42)
        expected_carry = math.sqrt((70 - 60)**2 + (42 - 40)**2)
        self.assertAlmostEqual(features["avg_carry_distance"], round(expected_carry, 2), places=1)

        # Tackle at (30, 25) → distance from (0, 40)
        expected_def = math.sqrt(30**2 + (25 - 40)**2)
        self.assertAlmostEqual(features["avg_defensive_distance"], round(expected_def, 2), places=1)


class TestHelpers(unittest.TestCase):
    """Test internal helper functions."""

    def test_distance(self):
        from backend.data.statsbomb_client import _distance

        self.assertAlmostEqual(_distance(0, 0, 3, 4), 5.0)
        self.assertAlmostEqual(_distance(10, 10, 10, 10), 0.0)

    def test_safe_float(self):
        from backend.data.statsbomb_client import _safe_float

        self.assertEqual(_safe_float(3.14), 3.14)
        self.assertEqual(_safe_float("2.5"), 2.5)
        self.assertEqual(_safe_float(None), 0.0)
        self.assertEqual(_safe_float("not-a-number"), 0.0)
        self.assertEqual(_safe_float([1, 2]), 0.0)

    def test_matches_player_case_insensitive(self):
        from backend.data.statsbomb_client import _matches_player

        self.assertTrue(_matches_player("Lionel Messi", "lionel messi"))
        self.assertTrue(_matches_player("Lionel Andrés Messi", "Lionel"))
        self.assertFalse(_matches_player("Sergio Busquets", "Messi"))
        self.assertFalse(_matches_player(None, "Messi"))

    def test_extract_location_valid(self):
        from backend.data.statsbomb_client import _extract_location

        self.assertEqual(_extract_location({"location": [10.0, 20.0]}), (10.0, 20.0))

    def test_extract_location_missing(self):
        from backend.data.statsbomb_client import _extract_location

        self.assertIsNone(_extract_location({}))
        self.assertIsNone(_extract_location({"location": None}))
        self.assertIsNone(_extract_location({"location": [10]}))
        self.assertIsNone(_extract_location({"location": "bad"}))


if __name__ == "__main__":
    unittest.main()
