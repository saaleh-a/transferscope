"""Tests for frontend.components.player_pizza — pizza chart rendering."""

from __future__ import annotations

import io

import pytest


_SAMPLE_PER90 = {
    "expected_goals": 0.45,
    "expected_assists": 0.30,
    "shots": 3.2,
    "successful_dribbles": 1.8,
    "successful_crosses": 0.9,
    "touches_in_opposition_box": 5.1,
    "successful_passes": 42.0,
    "pass_completion_pct": 82.0,
    "accurate_long_balls": 3.5,
    "chances_created": 1.7,
    "clearances": 1.2,
    "interceptions": 0.8,
    "possession_won_final_3rd": 0.6,
}

_COMPARISON_PER90 = {
    "expected_goals": 0.55,
    "expected_assists": 0.20,
    "shots": 4.0,
    "successful_dribbles": 1.2,
    "successful_crosses": 0.5,
    "touches_in_opposition_box": 6.0,
    "successful_passes": 38.0,
    "pass_completion_pct": 78.0,
    "accurate_long_balls": 2.8,
    "chances_created": 1.4,
    "clearances": 0.9,
    "interceptions": 0.6,
    "possession_won_final_3rd": 0.4,
}


class TestRenderPizza:
    """render_pizza() produces a valid PNG buffer."""

    def test_basic_render(self):
        from frontend.components.player_pizza import render_pizza
        buf = render_pizza(_SAMPLE_PER90, "Test Player")
        assert buf is not None
        assert isinstance(buf, io.BytesIO)
        data = buf.getvalue()
        assert len(data) > 1000  # valid PNG is at least a few KB
        assert data[:4] == b"\x89PNG"  # PNG magic bytes

    def test_render_with_comparison(self):
        from frontend.components.player_pizza import render_pizza
        buf = render_pizza(
            _SAMPLE_PER90,
            player_name="Player A",
            comparison_per90=_COMPARISON_PER90,
            comparison_name="Player B",
        )
        assert buf is not None
        data = buf.getvalue()
        assert data[:4] == b"\x89PNG"

    def test_returns_none_for_empty_data(self):
        from frontend.components.player_pizza import render_pizza
        result = render_pizza({}, "Empty Player")
        assert result is None

    def test_handles_partial_metrics(self):
        from frontend.components.player_pizza import render_pizza
        partial = {"expected_goals": 0.5, "shots": 2.0}
        buf = render_pizza(partial, "Partial Player")
        assert buf is not None

    def test_handles_zero_values(self):
        from frontend.components.player_pizza import render_pizza
        zeros = {m: 0.0 for m in _SAMPLE_PER90}
        buf = render_pizza(zeros, "Zero Player")
        # All zeros → still renders (max_val fallback to 1.0)
        assert buf is not None

    def test_handles_none_values(self):
        from frontend.components.player_pizza import render_pizza
        with_none = dict(_SAMPLE_PER90)
        with_none["shots"] = None  # type: ignore
        buf = render_pizza(with_none, "None-value Player")
        assert buf is not None
