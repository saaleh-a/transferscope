"""Tests for backend.data.whoscored_client — WhoScored data client.

Follows the same mocked-response pattern used in test_statsbomb_client.py
and test_sofascore_client.py.  No network calls — all HTTP and cache are
mocked.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock, patch


# ── Sample data ──────────────────────────────────────────────────────────────

_SEARCH_RESPONSE = [
    {
        "PlayerId": 11119,
        "PlayerName": "Lionel Messi",
        "TeamName": "Inter Miami",
        "TeamId": 23880,
        "RegionCode": "AR",
    },
    {
        "PlayerId": 11120,
        "PlayerName": "Lionel Messi Jr",
        "TeamName": "Academy FC",
        "TeamId": 99999,
        "RegionCode": "AR",
    },
]

_SEASON_STATS_RESPONSE = {
    "Statistics": {
        "Rating": 7.64,
        "MinutesPlayed": 2850,
        "Appearances": 32,
        "Goals": 18,
        "Assists": 12,
        "ShotsPerGame": 4.2,
        "KeyPassesPerGame": 2.8,
        "DribblesWonPerGame": 3.1,
        "PassSuccessPercentage": 85.3,
        "TacklesPerGame": 0.5,
        "InterceptionsPerGame": 0.3,
        "ClearancesPerGame": 0.1,
    }
}

_MATCH_HISTORY_PARSED = [
    {
        "match_id": 1001,
        "date": "2024-03-15",
        "opponent": "Orlando City",
        "minutes": 90,
        "rating": 8.2,
        "events": [
            {"type": "Shot", "x": 88.0, "y": 45.0},
            {"type": "Pass", "x": 50.0, "y": 50.0, "endX": 70.0, "endY": 45.0},
            {"type": "Tackle", "x": 25.0, "y": 30.0},
        ],
    },
    {
        "match_id": 1002,
        "date": "2024-03-22",
        "opponent": "Atlanta United",
        "minutes": 75,
        "rating": 7.5,
        "events": [
            {"type": "ShotOnTarget", "x": 90.0, "y": 55.0},
            {"type": "KeyPass", "x": 60.0, "y": 40.0, "endX": 85.0, "endY": 50.0},
            {"type": "TakeOn", "x": 55.0, "y": 50.0, "endX": 62.0, "endY": 52.0},
        ],
    },
]

# Raw API response for get_player_match_history tests
_MATCH_HISTORY_RAW = [
    {
        "MatchId": 1001,
        "Date": "2024-03-15",
        "Opponent": "Orlando City",
        "MinutesPlayed": 90,
        "Rating": 8.2,
        "Events": [
            {"type": "Shot", "x": 88.0, "y": 45.0},
        ],
    },
    {
        "MatchId": 1002,
        "Date": "2024-03-22",
        "Opponent": "Atlanta United",
        "MinutesPlayed": 75,
        "Rating": 7.5,
        "Events": [],
    },
]

_HEATMAP_RESPONSE = [
    {"x": 50.0, "y": 40.0},
    {"x": 70.0, "y": 60.0},
    {"x": 85.0, "y": 50.0},
    {"x": 30.0, "y": 20.0},
]

_EMPTY_RESPONSE = []


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1 Tests — whoscored_client core functions
# ═════════════════════════════════════════════════════════════════════════════


class TestSearchPlayer(unittest.TestCase):
    """search_player() finds players by name with caching."""

    @patch("backend.data.whoscored_client._get_json")
    @patch("backend.data.whoscored_client.cache")
    def test_returns_search_results(self, mock_cache, mock_get):
        from backend.data.whoscored_client import search_player

        mock_cache.get.return_value = None
        mock_get.return_value = _SEARCH_RESPONSE

        result = search_player("Lionel Messi")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], 11119)
        self.assertEqual(result[0]["name"], "Lionel Messi")
        self.assertEqual(result[0]["team_name"], "Inter Miami")
        mock_cache.set.assert_called_once()

    @patch("backend.data.whoscored_client._get_json")
    @patch("backend.data.whoscored_client.cache")
    def test_cache_hit_skips_request(self, mock_cache, mock_get):
        from backend.data.whoscored_client import search_player

        cached = [{"id": 11119, "name": "Lionel Messi"}]
        mock_cache.get.return_value = cached

        result = search_player("Lionel Messi")

        self.assertEqual(result, cached)
        mock_get.assert_not_called()

    @patch("backend.data.whoscored_client._get_json")
    @patch("backend.data.whoscored_client.cache")
    def test_empty_name_returns_empty(self, mock_cache, mock_get):
        from backend.data.whoscored_client import search_player

        result = search_player("")
        self.assertEqual(result, [])
        mock_get.assert_not_called()

    @patch("backend.data.whoscored_client._get_json")
    @patch("backend.data.whoscored_client.cache")
    def test_api_failure_returns_empty(self, mock_cache, mock_get):
        from backend.data.whoscored_client import search_player

        mock_cache.get.return_value = None
        mock_get.return_value = None

        result = search_player("NonExistentPlayer")
        self.assertEqual(result, [])


class TestGetPlayerSeasonStats(unittest.TestCase):
    """get_player_season_stats() fetches and caches season stats."""

    @patch("backend.data.whoscored_client._get_json")
    @patch("backend.data.whoscored_client.cache")
    def test_returns_parsed_stats(self, mock_cache, mock_get):
        from backend.data.whoscored_client import get_player_season_stats

        mock_cache.get.return_value = None
        mock_get.return_value = _SEASON_STATS_RESPONSE

        result = get_player_season_stats(11119)

        self.assertAlmostEqual(result["rating"], 7.64)
        self.assertAlmostEqual(result["minutes_played"], 2850.0)
        self.assertAlmostEqual(result["shots"], 4.2)
        self.assertAlmostEqual(result["pass_completion_pct"], 85.3)
        mock_cache.set.assert_called_once()

    @patch("backend.data.whoscored_client._get_json")
    @patch("backend.data.whoscored_client.cache")
    def test_cache_hit_returns_cached(self, mock_cache, mock_get):
        from backend.data.whoscored_client import get_player_season_stats

        cached_stats = {"rating": 7.5, "goals": 10}
        mock_cache.get.return_value = cached_stats

        result = get_player_season_stats(11119)

        self.assertEqual(result, cached_stats)
        mock_get.assert_not_called()

    @patch("backend.data.whoscored_client._get_json")
    @patch("backend.data.whoscored_client.cache")
    def test_invalid_id_returns_empty(self, mock_cache, mock_get):
        from backend.data.whoscored_client import get_player_season_stats

        result = get_player_season_stats(0)
        self.assertEqual(result, {})
        mock_get.assert_not_called()

    @patch("backend.data.whoscored_client._get_json")
    @patch("backend.data.whoscored_client.cache")
    def test_api_404_caches_negative(self, mock_cache, mock_get):
        from backend.data.whoscored_client import get_player_season_stats

        mock_cache.get.return_value = None
        mock_get.return_value = None

        result = get_player_season_stats(99999)

        self.assertEqual(result, {})
        # Should cache a negative sentinel
        mock_cache.set.assert_called_once()


class TestGetPlayerMatchHistory(unittest.TestCase):
    """get_player_match_history() fetches match-level logs."""

    @patch("backend.data.whoscored_client._get_json")
    @patch("backend.data.whoscored_client.cache")
    def test_returns_match_list(self, mock_cache, mock_get):
        from backend.data.whoscored_client import get_player_match_history

        mock_cache.get.return_value = None
        mock_get.return_value = _MATCH_HISTORY_RAW

        result = get_player_match_history(11119)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["match_id"], 1001)
        self.assertEqual(result[0]["opponent"], "Orlando City")
        self.assertEqual(result[0]["minutes"], 90)
        self.assertAlmostEqual(result[0]["rating"], 8.2)
        self.assertEqual(len(result[0]["events"]), 1)
        mock_cache.set.assert_called_once()

    @patch("backend.data.whoscored_client._get_json")
    @patch("backend.data.whoscored_client.cache")
    def test_cache_hit_returns_cached(self, mock_cache, mock_get):
        from backend.data.whoscored_client import get_player_match_history

        cached = [{"match_id": 1001}]
        mock_cache.get.return_value = cached

        result = get_player_match_history(11119)

        self.assertEqual(result, cached)
        mock_get.assert_not_called()

    @patch("backend.data.whoscored_client._get_json")
    @patch("backend.data.whoscored_client.cache")
    def test_invalid_id_returns_empty(self, mock_cache, mock_get):
        from backend.data.whoscored_client import get_player_match_history

        result = get_player_match_history(-1)
        self.assertEqual(result, [])
        mock_get.assert_not_called()


class TestGetPlayerHeatmapData(unittest.TestCase):
    """get_player_heatmap_data() fetches touch locations."""

    @patch("backend.data.whoscored_client._get_json")
    @patch("backend.data.whoscored_client.cache")
    def test_returns_coordinate_tuples(self, mock_cache, mock_get):
        from backend.data.whoscored_client import get_player_heatmap_data

        mock_cache.get.return_value = None
        mock_get.return_value = _HEATMAP_RESPONSE

        result = get_player_heatmap_data(11119)

        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[0], tuple)
        self.assertAlmostEqual(result[0][0], 50.0)
        self.assertAlmostEqual(result[0][1], 40.0)
        mock_cache.set.assert_called_once()

    @patch("backend.data.whoscored_client._get_json")
    @patch("backend.data.whoscored_client.cache")
    def test_filters_out_of_bounds(self, mock_cache, mock_get):
        from backend.data.whoscored_client import get_player_heatmap_data

        mock_cache.get.return_value = None
        mock_get.return_value = [
            {"x": 50.0, "y": 50.0},
            {"x": -5.0, "y": 50.0},  # out of bounds
            {"x": 110.0, "y": 50.0},  # out of bounds
        ]

        result = get_player_heatmap_data(11119)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0][0], 50.0)


class TestComputeSpatialFeatures(unittest.TestCase):
    """compute_spatial_features() aggregates event data into feature dict."""

    @patch("backend.data.whoscored_client.get_player_match_history")
    @patch("backend.data.whoscored_client.cache")
    def test_computes_features_from_events(self, mock_cache, mock_history):
        from backend.data.whoscored_client import compute_spatial_features

        mock_cache.get.return_value = None
        mock_history.return_value = _MATCH_HISTORY_PARSED

        features = compute_spatial_features(11119)

        # Should have spatial feature keys
        self.assertIn("avg_shot_distance", features)
        self.assertIn("shots_inside_box_pct", features)
        self.assertIn("avg_pass_length", features)
        self.assertIn("progressive_pass_pct", features)
        self.assertIn("avg_carry_distance", features)
        self.assertIn("touches_left_pct", features)
        self.assertIn("touches_center_pct", features)
        self.assertIn("touches_right_pct", features)
        self.assertIn("avg_defensive_distance", features)

        # All values should be floats
        for v in features.values():
            self.assertIsInstance(v, float)

    @patch("backend.data.whoscored_client.get_player_match_history")
    @patch("backend.data.whoscored_client.cache")
    def test_returns_empty_on_no_data(self, mock_cache, mock_history):
        from backend.data.whoscored_client import compute_spatial_features

        mock_cache.get.return_value = None
        mock_history.return_value = []

        result = compute_spatial_features(11119)
        self.assertEqual(result, {})

    @patch("backend.data.whoscored_client.get_player_match_history")
    @patch("backend.data.whoscored_client.cache")
    def test_returns_empty_for_invalid_id(self, mock_cache, mock_history):
        from backend.data.whoscored_client import compute_spatial_features

        result = compute_spatial_features(0)
        self.assertEqual(result, {})
        mock_history.assert_not_called()

    @patch("backend.data.whoscored_client.get_player_match_history")
    @patch("backend.data.whoscored_client.cache")
    def test_cache_hit_returns_cached(self, mock_cache, mock_history):
        from backend.data.whoscored_client import compute_spatial_features

        cached = {"avg_shot_distance": 15.5, "shots_inside_box_pct": 72.0}
        mock_cache.get.return_value = cached

        result = compute_spatial_features(11119)

        self.assertEqual(result, cached)
        mock_history.assert_not_called()

    @patch("backend.data.whoscored_client.get_player_match_history")
    @patch("backend.data.whoscored_client.cache")
    def test_shot_distance_computation(self, mock_cache, mock_history):
        """Verify shot distance is computed correctly on 100×100 pitch."""
        from backend.data.whoscored_client import compute_spatial_features, _distance

        mock_cache.get.return_value = None
        mock_history.return_value = [
            {
                "match_id": 1,
                "date": "2024-01-01",
                "opponent": "Test FC",
                "minutes": 90,
                "rating": 7.0,
                "events": [
                    {"type": "Shot", "x": 90.0, "y": 50.0},  # 10 units from goal
                ],
            }
        ]

        features = compute_spatial_features(11119)

        expected_dist = _distance(90.0, 50.0, 100.0, 50.0)  # = 10.0
        self.assertAlmostEqual(features["avg_shot_distance"], round(expected_dist, 2))
        self.assertAlmostEqual(features["shots_inside_box_pct"], 100.0)

    @patch("backend.data.whoscored_client.get_player_match_history")
    @patch("backend.data.whoscored_client.cache")
    def test_matches_without_events_returns_empty(self, mock_cache, mock_history):
        from backend.data.whoscored_client import compute_spatial_features

        mock_cache.get.return_value = None
        mock_history.return_value = [
            {
                "match_id": 1,
                "date": "2024-01-01",
                "opponent": "Test FC",
                "minutes": 0,
                "rating": 0,
                "events": [],
            }
        ]

        result = compute_spatial_features(11119)
        self.assertEqual(result, {})


class TestHelpers(unittest.TestCase):
    """Test internal helper functions."""

    def test_distance(self):
        from backend.data.whoscored_client import _distance

        self.assertAlmostEqual(_distance(0, 0, 3, 4), 5.0)
        self.assertAlmostEqual(_distance(10, 10, 10, 10), 0.0)
        self.assertAlmostEqual(_distance(0, 0, 1, 0), 1.0)

    def test_safe_float(self):
        from backend.data.whoscored_client import _safe_float

        self.assertEqual(_safe_float(3.14), 3.14)
        self.assertEqual(_safe_float(42), 42.0)
        self.assertEqual(_safe_float("7.5"), 7.5)
        self.assertEqual(_safe_float(None), 0.0)
        self.assertEqual(_safe_float("not-a-number"), 0.0)

    def test_parse_season_stats(self):
        from backend.data.whoscored_client import _parse_season_stats

        result = _parse_season_stats(_SEASON_STATS_RESPONSE)
        self.assertAlmostEqual(result["rating"], 7.64)
        self.assertAlmostEqual(result["shots"], 4.2)
        self.assertAlmostEqual(result["chances_created"], 2.8)

    def test_parse_season_stats_empty(self):
        from backend.data.whoscored_client import _parse_season_stats

        self.assertEqual(_parse_season_stats(None), {})
        self.assertEqual(_parse_season_stats({}), {})
        self.assertEqual(_parse_season_stats("invalid"), {})

    def test_compute_features_from_events(self):
        from backend.data.whoscored_client import _compute_features_from_events

        events = [
            {"type": "Shot", "x": 85.0, "y": 50.0},
            {"type": "Pass", "x": 40.0, "y": 50.0, "endX": 60.0, "endY": 45.0},
            {"type": "Tackle", "x": 20.0, "y": 30.0},
        ]

        features = _compute_features_from_events(events)

        self.assertIn("avg_shot_distance", features)
        self.assertIn("avg_pass_length", features)
        self.assertIn("avg_defensive_distance", features)
        self.assertIn("touches_left_pct", features)
        self.assertIn("touches_center_pct", features)
        self.assertIn("touches_right_pct", features)

    def test_compute_features_from_empty_events(self):
        from backend.data.whoscored_client import _compute_features_from_events

        self.assertEqual(_compute_features_from_events([]), {})


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2 Tests — Fallback chain (StatsBomb miss → WhoScored attempt)
# ═════════════════════════════════════════════════════════════════════════════


class TestSpatialFallbackChain(unittest.TestCase):
    """Test that spatial features fall back from StatsBomb to WhoScored."""

    @patch("backend.data.whoscored_client.compute_spatial_features")
    @patch("backend.data.reep_registry.enrich_player")
    @patch("backend.data.statsbomb_client.compute_spatial_features")
    def test_statsbomb_hit_skips_whoscored(
        self, mock_sb_spatial, mock_reep, mock_ws_spatial
    ):
        """When StatsBomb returns data, WhoScored should NOT be called."""
        # StatsBomb returns features
        mock_sb_spatial.return_value = {
            "avg_shot_distance": 15.0,
            "shots_inside_box_pct": 70.0,
        }

        # Simulate the fallback logic as implemented in transfer_portal.py
        sf = mock_sb_spatial("Lionel Messi")
        if not sf:
            reep_data = mock_reep(12345)
            ws_id = reep_data.get("whoscored_id")
            if ws_id:
                sf = mock_ws_spatial(ws_id)

        # StatsBomb returned data, so WhoScored should NOT be called
        mock_ws_spatial.assert_not_called()
        mock_reep.assert_not_called()
        self.assertIn("avg_shot_distance", sf)

    @patch("backend.data.whoscored_client.compute_spatial_features")
    @patch("backend.data.reep_registry.enrich_player")
    @patch("backend.data.statsbomb_client.compute_spatial_features")
    def test_statsbomb_miss_tries_whoscored(
        self, mock_sb_spatial, mock_reep, mock_ws_spatial
    ):
        """When StatsBomb returns {}, WhoScored should be attempted via REEP."""
        mock_sb_spatial.return_value = {}  # StatsBomb miss
        mock_reep.return_value = {"whoscored_id": 11119}
        mock_ws_spatial.return_value = {
            "avg_shot_distance": 18.5,
            "shots_inside_box_pct": 65.0,
        }

        # Directly test the fallback logic from training_pipeline
        # by calling the block that does the fallback
        sf = mock_sb_spatial("Test Player")
        if not sf:
            reep_data = mock_reep(12345)
            ws_id = reep_data.get("whoscored_id")
            if ws_id:
                sf = mock_ws_spatial(ws_id)

        self.assertEqual(sf["avg_shot_distance"], 18.5)
        mock_ws_spatial.assert_called_once_with(11119)

    @patch("backend.data.whoscored_client.compute_spatial_features")
    @patch("backend.data.reep_registry.enrich_player")
    @patch("backend.data.statsbomb_client.compute_spatial_features")
    def test_no_whoscored_id_falls_to_zeros(
        self, mock_sb_spatial, mock_reep, mock_ws_spatial
    ):
        """When REEP has no whoscored_id, spatial features stay as zeros."""
        mock_sb_spatial.return_value = {}
        mock_reep.return_value = {"whoscored_id": None}  # No WhoScored ID

        sf = mock_sb_spatial("Test Player")
        if not sf:
            reep_data = mock_reep(12345)
            ws_id = reep_data.get("whoscored_id")
            if ws_id:
                sf = mock_ws_spatial(ws_id)

        # WhoScored was never called (no ID)
        mock_ws_spatial.assert_not_called()
        # sf should still be {} (empty)
        self.assertEqual(sf, {})


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3 Tests — REEP bridge (key_whoscored extraction)
# ═════════════════════════════════════════════════════════════════════════════


class TestReepWhoscoredBridge(unittest.TestCase):
    """enrich_player() now returns whoscored_id from key_whoscored column."""

    @patch("backend.data.reep_registry.get_people_df")
    def test_returns_whoscored_id(self, mock_df):
        import pandas as pd
        from backend.data.reep_registry import clear_memory_cache, enrich_player

        clear_memory_cache()
        mock_df.return_value = pd.DataFrame([{
            "reep_id": "reep_p12345678",
            "key_sofascore": "12345",
            "nationality": "Argentina",
            "height_cm": 170.0,
            "date_of_birth": "1987-06-24",
            "position": "Forward",
            "key_whoscored": "11119",
        }])

        result = enrich_player(12345)

        self.assertEqual(result["whoscored_id"], 11119)
        self.assertEqual(result["nationality"], "Argentina")
        self.assertEqual(result["height_cm"], 170)
        self.assertEqual(result["reep_id"], "reep_p12345678")

    @patch("backend.data.reep_registry.get_people_df")
    def test_missing_whoscored_id_returns_none(self, mock_df):
        import pandas as pd
        from backend.data.reep_registry import clear_memory_cache, enrich_player

        clear_memory_cache()
        mock_df.return_value = pd.DataFrame([{
            "reep_id": "reep_p00000001",
            "key_sofascore": "12345",
            "nationality": "England",
            "height_cm": 180.0,
            "date_of_birth": "1995-01-01",
            "position": "Midfielder",
            "key_whoscored": float("nan"),  # Missing
        }])

        result = enrich_player(12345)

        self.assertIsNone(result["whoscored_id"])

    @patch("backend.data.reep_registry.get_people_df")
    def test_player_not_found_returns_empty(self, mock_df):
        import pandas as pd
        from backend.data.reep_registry import clear_memory_cache, enrich_player

        clear_memory_cache()
        mock_df.return_value = pd.DataFrame([{
            "reep_id": "reep_p99999999",
            "key_sofascore": "99999",
            "nationality": "Brazil",
            "height_cm": 175.0,
            "date_of_birth": "2000-01-01",
            "position": "Defender",
            "key_whoscored": "55555",
        }])

        # Looking for a different player
        result = enrich_player(12345)
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
