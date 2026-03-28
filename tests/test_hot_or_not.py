"""Tests for Hot or Not verdict logic."""

from __future__ import annotations

import pytest

from frontend.pages.hot_or_not import _verdict


# ── Basic threshold tests ────────────────────────────────────────────────────

def test_hot_above_threshold():
    """Weighted avg > +5% with no mixed signals → HOT."""
    pct = {"a": 8.0, "b": 6.0, "c": 2.0}
    verdict, _, _ = _verdict(avg_change=7.0, pct_changes=pct)
    assert verdict == "HOT"


def test_not_below_threshold():
    """Weighted avg < -5% with no mixed signals → NOT."""
    pct = {"a": -8.0, "b": -6.0, "c": -2.0}
    verdict, _, _ = _verdict(avg_change=-7.0, pct_changes=pct)
    assert verdict == "NOT"


def test_tepid_within_threshold():
    """Weighted avg between -5% and +5% → TEPID."""
    pct = {"a": 3.0, "b": -2.0, "c": 1.0}
    verdict, _, _ = _verdict(avg_change=2.0, pct_changes=pct)
    assert verdict == "TEPID"


def test_unknown_no_data():
    """No data → UNKNOWN regardless of changes."""
    pct = {"a": 50.0}
    verdict, _, _ = _verdict(avg_change=50.0, pct_changes=pct, has_data=False)
    assert verdict == "UNKNOWN"


# ── Mixed signals with moderate avg → TEPID ──────────────────────────────────

def test_mixed_signals_moderate_avg_tepid():
    """≥2 strong up, ≥2 strong down, avg ≤ 10% → TEPID override."""
    pct = {
        "a": 15.0, "b": 12.0,    # 2 strong up
        "c": -15.0, "d": -12.0,  # 2 strong down
        "e": 1.0,
    }
    verdict, _, _ = _verdict(avg_change=7.0, pct_changes=pct)
    assert verdict == "TEPID"


def test_mixed_signals_negative_moderate_avg_tepid():
    """Mixed signals with moderate negative avg → TEPID override."""
    pct = {
        "a": 15.0, "b": 12.0,
        "c": -20.0, "d": -15.0,
        "e": -3.0,
    }
    verdict, _, _ = _verdict(avg_change=-8.0, pct_changes=pct)
    assert verdict == "TEPID"


# ── Mixed signals with strong avg → respect the avg ─────────────────────────

def test_mixed_signals_strong_positive_avg_hot():
    """Mixed signals but avg > 10% → HOT (not overridden).

    This is the Osimhen→Arsenal case: style trade-offs exist but the
    position-weighted net effect is overwhelmingly positive.
    """
    pct = {
        "expected_goals": 18.5,
        "chances_created": 42.6,
        "possession_won_final_3rd": 485.3,  # inflated defensive metric
        "successful_crosses": -48.9,
        "successful_dribbles": -21.4,
        "pass_completion_pct": -1.2,
        "shots": 5.0,
    }
    verdict, _, _ = _verdict(avg_change=17.9, pct_changes=pct)
    assert verdict == "HOT"


def test_mixed_signals_strong_negative_avg_not():
    """Mixed signals but avg < -10% → NOT (not overridden)."""
    pct = {
        "a": 15.0, "b": 12.0,
        "c": -30.0, "d": -25.0, "e": -20.0,
    }
    verdict, _, _ = _verdict(avg_change=-15.0, pct_changes=pct)
    assert verdict == "NOT"


# ── Edge cases around the mixed override boundary ────────────────────────────

def test_mixed_at_exactly_threshold_is_tepid():
    """avg_change exactly at ±10% → mixed override still applies."""
    pct = {"a": 15.0, "b": 12.0, "c": -15.0, "d": -12.0}
    verdict, _, _ = _verdict(avg_change=10.0, pct_changes=pct)
    assert verdict == "TEPID"


def test_mixed_just_above_threshold_is_hot():
    """avg_change just above +10% → mixed override does NOT apply."""
    pct = {"a": 15.0, "b": 12.0, "c": -15.0, "d": -12.0}
    verdict, _, _ = _verdict(avg_change=10.1, pct_changes=pct)
    assert verdict == "HOT"


def test_mixed_just_below_neg_threshold_is_not():
    """avg_change just below -10% → mixed override does NOT apply."""
    pct = {"a": 15.0, "b": 12.0, "c": -15.0, "d": -12.0}
    verdict, _, _ = _verdict(avg_change=-10.1, pct_changes=pct)
    assert verdict == "NOT"


# ── Insufficient mixed metrics → no override ─────────────────────────────────

def test_only_one_strong_down_not_mixed():
    """Only 1 strong decliner (need ≥2) → no mixed override → HOT."""
    pct = {"a": 15.0, "b": 12.0, "c": -15.0, "d": 3.0}
    verdict, _, _ = _verdict(avg_change=7.0, pct_changes=pct)
    assert verdict == "HOT"


def test_only_one_strong_up_not_mixed():
    """Only 1 strong improver (need ≥2) → no mixed override → NOT."""
    pct = {"a": 15.0, "b": -15.0, "c": -12.0, "d": -3.0}
    verdict, _, _ = _verdict(avg_change=-7.0, pct_changes=pct)
    assert verdict == "NOT"
