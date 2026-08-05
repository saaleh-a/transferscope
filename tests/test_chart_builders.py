"""Chart builders must not raise.

A Plotly kwarg collision (passing ``**PLOTLY_LAYOUT`` *and* an explicit
``title=``) raises TypeError only when the chart is actually built, which in
Streamlit means only when a user opens that page with data present.  The
Diagnostics feature-importance chart shipped broken for exactly this reason.
These tests build every figure directly so the collision surfaces in CI.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from frontend.theme import PLOTLY_LAYOUT, apply_plotly_theme


def _bar() -> go.Figure:
    return go.Figure(go.Bar(x=[1.0, 2.0], y=["a", "b"], orientation="h"))


class TestApplyPlotlyTheme:
    def test_applies_without_title(self):
        fig = apply_plotly_theme(_bar())
        assert fig.layout.plot_bgcolor == "#161B22"

    def test_applies_with_title(self):
        fig = apply_plotly_theme(_bar(), title="Shooting Group")
        assert fig.layout.title.text == "Shooting Group"

    def test_overrides_do_not_collide_with_base_layout(self):
        """The exact signature that broke Diagnostics."""
        fig = apply_plotly_theme(
            _bar(),
            title="Shooting Group",
            height=300,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=40, b=20),
        )
        assert fig.layout.yaxis.autorange == "reversed"
        assert fig.layout.height == 300

    def test_override_wins_over_base(self):
        fig = apply_plotly_theme(_bar(), plot_bgcolor="#000000")
        assert fig.layout.plot_bgcolor == "#000000"

    def test_does_not_mutate_shared_layout(self):
        before = dict(PLOTLY_LAYOUT["title"])
        apply_plotly_theme(_bar(), title="One")
        apply_plotly_theme(_bar(), title="Two")
        assert dict(PLOTLY_LAYOUT["title"]) == before
        assert "text" not in PLOTLY_LAYOUT["title"]


class TestPageChartsBuild:
    """Build the figures each page constructs, with the same kwargs."""

    def test_diagnostics_feature_importance_chart(self):
        fig = go.Figure(go.Bar(x=[0.3, 0.2], y=["f1", "f2"], orientation="h"))
        apply_plotly_theme(
            fig,
            title="Shooting Group",
            height=max(200, 28 * 2),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=40, b=20),
        )
        assert fig.layout.title.text == "Shooting Group"

    def test_backtest_predicted_vs_actual_chart(self):
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Predicted", x=["shots"], y=[2.0]))
        fig.add_trace(go.Bar(name="Actual", x=["shots"], y=[2.4]))
        layout = dict(PLOTLY_LAYOUT)
        layout["title"] = dict(text="Predicted vs Actual Per-90", **PLOTLY_LAYOUT["title"])
        fig.update_layout(
            **layout,
            barmode="group",
            xaxis_title="Metric",
            yaxis_title="Per-90 Value",
            height=450,
        )
        assert fig.layout.barmode == "group"

    def test_power_ranking_chart_builds(self):
        from datetime import date

        from frontend.components.power_ranking_chart import render_power_ranking_chart

        fig = render_power_ranking_chart(
            source_club="Arsenal",
            target_club="Real Madrid",
            source_history=[(date(2025, 1, 1), 88.0), (date(2025, 6, 1), 89.5)],
            target_history=[(date(2025, 1, 1), 92.0), (date(2025, 6, 1), 93.1)],
            transfer_date=date(2025, 7, 1),
        )
        assert fig is not None

    def test_swarm_plot_builds(self):
        from frontend.components.swarm_plot import render_swarm_plot

        fig = render_swarm_plot(
            metric_name="shots",
            metric_label="Shots per 90",
            player_value=2.4,
            teammate_values=[1.1, 1.8, 2.0],
            league_values=[0.5, 1.2, 2.9, 3.4],
            player_name="Bukayo Saka",
            percentile=78.0,
        )
        assert fig is not None


class TestPlotlyLayoutContrast:
    """Axis and legend text must clear WCAG AA on the plot background."""

    @staticmethod
    def _ratio(fg: str, bg: str) -> float:
        def lin(c: float) -> float:
            c /= 255.0
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        def lum(h: str) -> float:
            h = h.lstrip("#")
            r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
            return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)

        a, b = lum(fg), lum(bg)
        lo, hi = min(a, b), max(a, b)
        return (hi + 0.05) / (lo + 0.05)

    @pytest.mark.parametrize("axis", ["xaxis", "yaxis"])
    def test_axis_tick_text_passes_aa(self, axis):
        colour = PLOTLY_LAYOUT[axis]["tickfont"]["color"]
        assert self._ratio(colour, PLOTLY_LAYOUT["plot_bgcolor"]) >= 4.5

    def test_legend_text_passes_aa(self):
        colour = PLOTLY_LAYOUT["legend"]["font"]["color"]
        assert self._ratio(colour, PLOTLY_LAYOUT["plot_bgcolor"]) >= 4.5
