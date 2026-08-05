"""Tests for the Diagnostics page."""

from __future__ import annotations

from unittest.mock import ANY, MagicMock, patch
import sys

import pandas as pd
import pytest

from backend.models.transfer_portal import MODEL_GROUPS, FEATURE_DIM


# ── Test render() doesn't crash ──────────────────────────────────────────────


@patch("frontend.pages.diagnostics.st")
@patch("frontend.pages.diagnostics.page_header")
@patch("frontend.pages.diagnostics._render_system_info")
@patch("frontend.pages.diagnostics._render_data_source_status")
@patch("frontend.pages.diagnostics._render_cache_health")
@patch("frontend.pages.diagnostics._render_feature_importance")
@patch("frontend.pages.diagnostics._render_model_status")
def test_render_calls_all_sections(
    mock_model_status,
    mock_feat_imp,
    mock_cache,
    mock_data_src,
    mock_sys_info,
    mock_page_header,
    mock_st,
):
    """render() calls all five section renderers without error."""
    from frontend.pages.diagnostics import render

    render()

    mock_model_status.assert_called_once()
    mock_feat_imp.assert_called_once()
    mock_cache.assert_called_once()
    mock_data_src.assert_called_once()
    mock_sys_info.assert_called_once()
    mock_page_header.assert_called_once()
    assert mock_page_header.call_args.args[0] == "Diagnostics"


# ── Test model status: untrained ─────────────────────────────────────────────


@patch("frontend.pages.diagnostics.st")
@patch("backend.models.transfer_portal.TransferPortalModel.is_trained", return_value=False)
def test_model_status_untrained(mock_trained, mock_st):
    """Model status section renders without crashing for untrained model."""
    from frontend.pages.diagnostics import _render_model_status

    _render_model_status()

    mock_st.markdown.assert_called()
    mock_st.dataframe.assert_called_once()


# ── Test model status: trained ───────────────────────────────────────────────


@patch("frontend.pages.diagnostics.st")
@patch("backend.models.transfer_portal.TransferPortalModel.is_trained", return_value=True)
def test_model_status_trained(mock_trained, mock_st):
    """Model status section renders correctly for trained model."""
    from frontend.pages.diagnostics import _render_model_status

    _render_model_status()

    mock_st.markdown.assert_called()
    html_calls = [str(c) for c in mock_st.markdown.call_args_list]
    assert any("ts-stat-card" in h for h in html_calls)


# ── Test feature importance: untrained shows info ────────────────────────────


@patch("frontend.pages.diagnostics.st")
@patch("backend.models.transfer_portal.TransferPortalModel.is_trained", return_value=False)
def test_feature_importance_untrained(mock_trained, mock_st):
    """Feature importance shows info message when model is not trained."""
    from frontend.pages.diagnostics import _render_feature_importance

    _render_feature_importance()

    mock_st.info.assert_called_once()


# ── Test feature importance: trained with exception ──────────────────────────


@patch("frontend.pages.diagnostics.st")
@patch("backend.models.transfer_portal.TransferPortalModel.is_trained", return_value=True)
@patch("backend.models.transfer_portal.TransferPortalModel.load", side_effect=RuntimeError("bad"))
def test_feature_importance_handles_load_error(mock_load, mock_trained, mock_st):
    """Feature importance handles model load errors gracefully."""
    from frontend.pages.diagnostics import _render_feature_importance

    _render_feature_importance()

    mock_st.warning.assert_called_once()


# ── Test cache health: unavailable cache ─────────────────────────────────────


@patch("frontend.pages.diagnostics.st")
def test_cache_health_missing_dir(mock_st):
    """Cache health degrades to an info box when the cache cannot be read.

    The section used to walk the cache directory itself; it now asks the cache
    module for stats, so this covers the module reporting an error rather than
    a missing folder.
    """
    from frontend.pages.diagnostics import _render_cache_health

    with patch("backend.data.cache.stats", return_value={"error": "unavailable"}):
        _render_cache_health()
    mock_st.info.assert_called_once()


# ── Test cache health: populated cache ───────────────────────────────────────


@patch("frontend.pages.diagnostics.st")
def test_cache_health_existing_dir(mock_st):
    """Cache health renders size, limit and namespace breakdown."""
    from frontend.pages.diagnostics import _render_cache_health

    mock_st.button.return_value = False
    stats = {
        "entries": 1000,
        "bytes": 500 * 1024 * 1024,
        "mb": 500.0,
        "limit_mb": 2048,
        "pct_of_limit": 24.4,
        "eviction_policy": "least-recently-used",
    }
    breakdown = {"sofascore": 800, "clubelo": 200}

    with patch("backend.data.cache.stats", return_value=stats), \
         patch("backend.data.cache.namespace_breakdown", return_value=breakdown):
        _render_cache_health()

    mock_st.markdown.assert_called()
    html_calls = [str(c) for c in mock_st.markdown.call_args_list]
    assert any("ts-stat-card" in h for h in html_calls)
    mock_st.dataframe.assert_called_once()


@patch("frontend.pages.diagnostics.st")
def test_cache_health_warns_when_near_limit(mock_st):
    """A cache about to start evicting must say so."""
    from frontend.pages.diagnostics import _render_cache_health

    mock_st.button.return_value = False
    stats = {
        "entries": 100_000,
        "bytes": 1900 * 1024 * 1024,
        "mb": 1900.0,
        "limit_mb": 2048,
        "pct_of_limit": 92.8,
        "eviction_policy": "least-recently-used",
    }
    with patch("backend.data.cache.stats", return_value=stats), \
         patch("backend.data.cache.namespace_breakdown", return_value={"sofascore": 1}):
        _render_cache_health()

    mock_st.warning.assert_called()


# ── Test data source status ──────────────────────────────────────────────────


@patch("frontend.pages.diagnostics.st")
def test_data_source_status_renders(mock_st):
    """Data source status renders a table of availability."""
    from frontend.pages.diagnostics import _render_data_source_status

    _render_data_source_status()
    mock_st.dataframe.assert_called_once()


# ── Test data source detects available packages ──────────────────────────────


@patch("frontend.pages.diagnostics.st")
def test_data_source_shows_available_packages(mock_st):
    """The health table renders real probe results, not import checks.

    The old version of this section reported a source healthy whenever its
    Python module imported, which marked two permanently dead sources green.
    It now runs live probes on demand, so the table only appears once results
    exist in session state.
    """
    from backend.data.source_health import DEAD, LIVE, SourceHealth
    from frontend.pages.diagnostics import _render_data_source_status

    # No probe run yet — the page must prompt rather than render a stale table.
    mock_st.session_state = {}
    mock_st.button.return_value = False
    _render_data_source_status()
    mock_st.dataframe.assert_not_called()
    mock_st.info.assert_called()

    # With results present, the table reflects them.
    mock_st.reset_mock()
    mock_st.session_state = {
        "_diag_health": [
            SourceHealth("Sofascore", LIVE, "20 metrics", "model inputs", 0.5),
            SourceHealth("WhoScored", DEAD, "returned nothing", "Nothing", 0.2),
        ]
    }
    mock_st.button.return_value = False
    _render_data_source_status()

    mock_st.dataframe.assert_called_once()
    df = mock_st.dataframe.call_args[0][0]
    statuses = df["Status"].tolist()
    assert any("Live" in s for s in statuses)
    assert any("Dead" in s for s in statuses)
    # A dead source must be called out, not buried in the table.
    mock_st.warning.assert_called()


# ── Test system info renders ─────────────────────────────────────────────────


@patch("frontend.pages.diagnostics.st")
def test_system_info_renders(mock_st):
    """System info renders a table with Python version."""
    from frontend.pages.diagnostics import _render_system_info

    _render_system_info()
    mock_st.dataframe.assert_called_once()
    args = mock_st.dataframe.call_args
    df = args[0][0]
    components = df["Component"].tolist()
    assert "Python" in components


# ── Test system info includes TF version ─────────────────────────────────────


@patch("frontend.pages.diagnostics.st")
def test_system_info_includes_tensorflow(mock_st):
    """System info table includes TensorFlow."""
    from frontend.pages.diagnostics import _render_system_info

    _render_system_info()
    args = mock_st.dataframe.call_args
    df = args[0][0]
    components = df["Component"].tolist()
    assert "TensorFlow" in components


# ── Test model status stat card values ───────────────────────────────────────


@patch("frontend.pages.diagnostics.st")
@patch("backend.models.transfer_portal.TransferPortalModel.is_trained", return_value=False)
def test_model_status_shows_feature_dim(mock_trained, mock_st):
    """Model status section displays the FEATURE_DIM value."""
    from frontend.pages.diagnostics import _render_model_status

    _render_model_status()

    html_calls = " ".join(str(c) for c in mock_st.markdown.call_args_list)
    assert "Feature Dimension" in html_calls


# ── Test model groups constant ───────────────────────────────────────────────


def test_model_groups_has_six_groups():
    """MODEL_GROUPS constant has exactly 6 groups."""
    assert len(MODEL_GROUPS) == 6
    assert set(MODEL_GROUPS.keys()) == {
        "shooting", "creation", "distribution", "crossing",
        "dribbling", "defending",
    }
