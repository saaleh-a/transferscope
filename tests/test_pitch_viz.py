"""Tests for frontend.components.pitch_viz — mplsoccer pitch visualizations."""

from __future__ import annotations

import io
import unittest


# ── Sample data ──────────────────────────────────────────────────────────────

_SAMPLE_SHOTS = [
    {"x": 105.0, "y": 35.0, "xg": 0.32, "outcome": "Goal"},
    {"x": 100.0, "y": 42.0, "xg": 0.08, "outcome": "Saved"},
    {"x": 110.0, "y": 38.0, "xg": 0.55, "outcome": "Blocked"},
    {"x": 95.0, "y": 45.0, "xg": 0.12, "outcome": "Off T"},
]

_SAMPLE_PASSES = [
    {"start_x": 30.0, "start_y": 40.0, "end_x": 55.0, "end_y": 38.0, "outcome": "Complete"},
    {"start_x": 50.0, "start_y": 25.0, "end_x": 70.0, "end_y": 30.0, "outcome": "Complete"},
    {"start_x": 60.0, "start_y": 50.0, "end_x": 62.0, "end_y": 48.0, "outcome": "Incomplete"},
    {"start_x": 45.0, "start_y": 60.0, "end_x": 80.0, "end_y": 55.0, "outcome": "Complete"},
]

_SAMPLE_LOCATIONS = [
    (30.0, 40.0), (35.0, 42.0), (40.0, 38.0), (50.0, 45.0),
    (55.0, 50.0), (60.0, 35.0), (65.0, 42.0), (70.0, 48.0),
    (75.0, 40.0), (80.0, 38.0), (85.0, 44.0), (90.0, 42.0),
]  # 12 points, above the minimum of 10


def _is_valid_png(buf: io.BytesIO) -> bool:
    """Check that a BytesIO buffer contains a valid PNG."""
    data = buf.getvalue()
    return len(data) > 500 and data[:4] == b"\x89PNG"


# ═════════════════════════════════════════════════════════════════════════════
# Shot Map
# ═════════════════════════════════════════════════════════════════════════════


class TestRenderShotMap(unittest.TestCase):
    """render_shot_map() produces a valid PNG buffer."""

    def test_basic_render(self):
        from frontend.components.pitch_viz import render_shot_map

        buf = render_shot_map(_SAMPLE_SHOTS, "Test Player")

        self.assertIsNotNone(buf)
        self.assertIsInstance(buf, io.BytesIO)
        self.assertTrue(_is_valid_png(buf))

    def test_empty_shots_returns_none(self):
        from frontend.components.pitch_viz import render_shot_map

        result = render_shot_map([], "Test Player")

        self.assertIsNone(result)

    def test_single_shot(self):
        from frontend.components.pitch_viz import render_shot_map

        single = [{"x": 105.0, "y": 35.0, "xg": 0.45, "outcome": "Goal"}]
        buf = render_shot_map(single, "Solo Striker")

        self.assertIsNotNone(buf)
        self.assertTrue(_is_valid_png(buf))

    def test_goal_outcome(self):
        from frontend.components.pitch_viz import render_shot_map

        goals_only = [
            {"x": 108.0, "y": 38.0, "xg": 0.72, "outcome": "Goal"},
            {"x": 112.0, "y": 42.0, "xg": 0.91, "outcome": "Goal"},
        ]
        buf = render_shot_map(goals_only, "Clinical Finisher")

        self.assertIsNotNone(buf)

    def test_saved_and_blocked(self):
        from frontend.components.pitch_viz import render_shot_map

        mixed = [
            {"x": 100.0, "y": 40.0, "xg": 0.15, "outcome": "Saved"},
            {"x": 98.0, "y": 35.0, "xg": 0.10, "outcome": "Blocked"},
        ]
        buf = render_shot_map(mixed, "Unlucky Striker")

        self.assertIsNotNone(buf)

    def test_missing_xg_defaults(self):
        from frontend.components.pitch_viz import render_shot_map

        no_xg = [{"x": 105.0, "y": 40.0, "outcome": "Saved"}]
        buf = render_shot_map(no_xg, "No xG")

        self.assertIsNotNone(buf)


# ═════════════════════════════════════════════════════════════════════════════
# Pass Network
# ═════════════════════════════════════════════════════════════════════════════


class TestRenderPassNetwork(unittest.TestCase):
    """render_pass_network() produces a valid PNG buffer."""

    def test_basic_render(self):
        from frontend.components.pitch_viz import render_pass_network

        buf = render_pass_network(_SAMPLE_PASSES, "Test Midfielder")

        self.assertIsNotNone(buf)
        self.assertIsInstance(buf, io.BytesIO)
        self.assertTrue(_is_valid_png(buf))

    def test_empty_passes_returns_none(self):
        from frontend.components.pitch_viz import render_pass_network

        result = render_pass_network([], "Test Player")

        self.assertIsNone(result)

    def test_single_pass(self):
        from frontend.components.pitch_viz import render_pass_network

        single = [{"start_x": 40.0, "start_y": 40.0, "end_x": 60.0,
                    "end_y": 38.0, "outcome": "Complete"}]
        buf = render_pass_network(single, "One Pass")

        self.assertIsNotNone(buf)

    def test_mixed_complete_incomplete(self):
        from frontend.components.pitch_viz import render_pass_network

        mixed = [
            {"start_x": 30.0, "start_y": 40.0, "end_x": 55.0,
             "end_y": 38.0, "outcome": "Complete"},
            {"start_x": 50.0, "start_y": 25.0, "end_x": 52.0,
             "end_y": 30.0, "outcome": "Incomplete"},
        ]
        buf = render_pass_network(mixed, "Mixed Passer")

        self.assertIsNotNone(buf)

    def test_progressive_passes(self):
        """Passes moving ≥10 m toward goal are drawn thicker."""
        from frontend.components.pitch_viz import render_pass_network

        progressive = [
            {"start_x": 30.0, "start_y": 40.0, "end_x": 80.0,
             "end_y": 40.0, "outcome": "Complete"},  # +50 m = progressive
        ]
        buf = render_pass_network(progressive, "Progressive Passer")

        self.assertIsNotNone(buf)


# ═════════════════════════════════════════════════════════════════════════════
# Heatmap
# ═════════════════════════════════════════════════════════════════════════════


class TestRenderHeatmap(unittest.TestCase):
    """render_heatmap() produces a valid PNG buffer."""

    def test_basic_render(self):
        from frontend.components.pitch_viz import render_heatmap

        buf = render_heatmap(_SAMPLE_LOCATIONS, "Test Player")

        self.assertIsNotNone(buf)
        self.assertIsInstance(buf, io.BytesIO)
        self.assertTrue(_is_valid_png(buf))

    def test_empty_locations_returns_none(self):
        from frontend.components.pitch_viz import render_heatmap

        result = render_heatmap([], "Test Player")

        self.assertIsNone(result)

    def test_fewer_than_10_locations_returns_none(self):
        from frontend.components.pitch_viz import render_heatmap

        too_few = [(30.0, 40.0), (40.0, 50.0), (50.0, 60.0)]
        result = render_heatmap(too_few, "Too Few Touches")

        self.assertIsNone(result)

    def test_exactly_10_locations_works(self):
        from frontend.components.pitch_viz import render_heatmap

        exactly_10 = [
            (20.0 + i * 8, 30.0 + i * 2) for i in range(10)
        ]
        buf = render_heatmap(exactly_10, "Exact Minimum")

        self.assertIsNotNone(buf)

    def test_large_dataset(self):
        """Heatmap renders with many points without error."""
        from frontend.components.pitch_viz import render_heatmap

        many = [(float(i % 120), float(i % 80)) for i in range(200)]
        buf = render_heatmap(many, "Busy Player")

        self.assertIsNotNone(buf)


if __name__ == "__main__":
    unittest.main()
