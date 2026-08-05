"""Tests for the live data-source health probes.

These exist because the Diagnostics page previously reported a source healthy
whenever its Python module imported. WhoScored and WorldFootballElo both import
cleanly and both return nothing, so both showed a green tick indefinitely.

The tests run offline (the suite blocks sockets), so they cover probe
mechanics — statuses, failure isolation, degraded signalling — rather than
live reachability.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.data import source_health as sh


class TestProbeMechanics(unittest.TestCase):
    def test_data_returns_live(self):
        result = sh._probe("X", "why", lambda: "42 things")
        self.assertEqual(result.status, sh.LIVE)
        self.assertEqual(result.detail, "42 things")
        self.assertTrue(result.is_ok)

    def test_empty_returns_dead(self):
        result = sh._probe("X", "why", lambda: "")
        self.assertEqual(result.status, sh.DEAD)
        self.assertFalse(result.is_ok)

    def test_bang_prefix_signals_degraded(self):
        result = sh._probe("X", "why", lambda: "!partial coverage")
        self.assertEqual(result.status, sh.DEGRADED)
        self.assertEqual(result.detail, "partial coverage")  # marker stripped
        self.assertFalse(result.is_ok)

    def test_exception_becomes_unknown_not_a_crash(self):
        """A source being down must never take the Diagnostics page with it."""
        def boom():
            raise ConnectionError("network unreachable")

        result = sh._probe("X", "why", boom)
        self.assertEqual(result.status, sh.UNKNOWN)
        self.assertIn("ConnectionError", result.detail)

    def test_elapsed_is_recorded(self):
        result = sh._probe("X", "why", lambda: "ok")
        self.assertGreaterEqual(result.elapsed_s, 0.0)

    def test_used_for_is_carried_through(self):
        result = sh._probe("X", "explains why we care", lambda: "ok")
        self.assertEqual(result.used_for, "explains why we care")


class TestProbeAll(unittest.TestCase):
    def test_probe_all_never_raises(self):
        """Every probe fails offline; probe_all must still return cleanly."""
        results = sh.probe_all()
        self.assertEqual(len(results), len(sh._PROBES))
        for result in results:
            self.assertIn(
                result.status, {sh.LIVE, sh.DEGRADED, sh.DEAD, sh.UNKNOWN}
            )

    def test_include_slow_false_skips_heavy_probes(self):
        full = sh.probe_all(include_slow=True)
        quick = sh.probe_all(include_slow=False)
        self.assertLess(len(quick), len(full))

    def test_every_probe_documents_what_it_is_used_for(self):
        for name, used_for, _fn in sh._PROBES:
            self.assertTrue(used_for, f"{name} has no 'used for' description")

    def test_known_dead_sources_are_recorded_after_removal(self):
        """The two dead sources are deleted, not silently forgotten."""
        self.assertIn("WhoScored", sh.REMOVED_SOURCES)
        self.assertIn("WorldFootballElo", sh.REMOVED_SOURCES)
        for name, reason in sh.REMOVED_SOURCES.items():
            self.assertTrue(reason, f"{name} has no removal reason")

    def test_removed_sources_are_not_probed(self):
        """Probing a deleted module would raise ImportError every run."""
        probed = {name for name, _, _ in sh._PROBES}
        for removed in sh.REMOVED_SOURCES:
            self.assertNotIn(removed, probed)


class TestSummarise(unittest.TestCase):
    def test_counts_each_status(self):
        results = [
            sh.SourceHealth("a", sh.LIVE, "", ""),
            sh.SourceHealth("b", sh.LIVE, "", ""),
            sh.SourceHealth("c", sh.DEGRADED, "", ""),
            sh.SourceHealth("d", sh.DEAD, "", ""),
        ]
        summary = sh.summarise(results)
        self.assertIn("2 live", summary)
        self.assertIn("1 degraded", summary)
        self.assertIn("1 dead", summary)

    def test_empty(self):
        self.assertIn("0 live", sh.summarise([]))


class TestIndividualProbeLogic(unittest.TestCase):
    """Probe bodies, with the underlying clients mocked."""

    def test_sofascore_probe_dead_when_no_metrics(self):
        with patch("backend.data.sofascore_client.get_player_stats",
                   return_value={"per90": {}, "minutes_played": 0}):
            self.assertEqual(sh._probe_sofascore(), "")

    def test_sofascore_probe_live_with_metrics(self):
        with patch("backend.data.sofascore_client.get_player_stats",
                   return_value={"per90": {"expected_goals": 0.3},
                                 "minutes_played": 900}):
            self.assertIn("1 metrics", sh._probe_sofascore())

    def test_value_probe_degrades_when_squad_has_no_prices(self):
        """Squad resolves but nothing is priced — that is degraded, not dead."""
        with patch("backend.data.sofascore_client.get_team_squad_profiles",
                   return_value={1: {"market_value": None}}):
            self.assertTrue(sh._probe_sofascore_value().startswith("!"))

    def test_opta_probe_degrades_without_league_averages(self):
        class FakeTeam:
            season_avg_rating = None

        with patch("backend.data.opta_client.get_team_rankings",
                   return_value=[FakeTeam()] * 10), \
             patch("backend.data.opta_client.get_league_rankings",
                   return_value=[]):
            self.assertTrue(sh._probe_opta().startswith("!"))

    def test_statsbomb_probe_degrades_on_partial_coverage(self):
        calls = {"n": 0}

        def partial(_name):
            calls["n"] += 1
            return {"avg_shot_distance": 1.0} if calls["n"] == 1 else {}

        with patch("backend.data.statsbomb_client.compute_spatial_features",
                   side_effect=partial):
            self.assertTrue(sh._probe_statsbomb().startswith("!"))

    def test_worldelo_probe_is_dead_by_construction(self):
        """The removed clients must stay removed."""
        with self.assertRaises(ImportError):
            from backend.data import worldfootballelo_client  # noqa: F401
        with self.assertRaises(ImportError):
            from backend.data import whoscored_client  # noqa: F401


if __name__ == "__main__":
    unittest.main()
