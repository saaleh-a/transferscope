"""Tests for the Diagnostics page."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import sys

import pandas as pd
import pytest

from backend.models.transfer_portal import MODEL_GROUPS, FEATURE_DIM


# ── Test render() doesn't crash ──────────────────────────────────────────────


@patch("frontend.pages.diagnostics.st")
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
    mock_st.header.assert_called_once_with("Diagnostics")


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


# ── Test cache health: missing dir ───────────────────────────────────────────


@patch("frontend.pages.diagnostics.st")
@patch("frontend.pages.diagnostics.os.path.exists", return_value=False)
def test_cache_health_missing_dir(mock_exists, mock_st):
    """Cache health shows info when cache dir doesn't exist."""
    from frontend.pages.diagnostics import _render_cache_health

    _render_cache_health()
    mock_st.info.assert_called_once()


# ── Test cache health: existing dir with files ───────────────────────────────


@patch("frontend.pages.diagnostics.st")
@patch("frontend.pages.diagnostics.os.path.exists", return_value=True)
@patch(
    "frontend.pages.diagnostics.os.walk",
    return_value=[("/cache", [], ["file1.db", "file2.db"])],
)
@patch("frontend.pages.diagnostics.os.path.getsize", return_value=1024)
def test_cache_health_existing_dir(mock_size, mock_walk, mock_exists, mock_st):
    """Cache health renders size and file count cards."""
    from frontend.pages.diagnostics import _render_cache_health

    mock_cache_inst = MagicMock()
    mock_cache_inst.__iter__ = MagicMock(
        return_value=iter(["sofascore:key1", "clubelo:key2"])
    )
    with patch("diskcache.Cache", return_value=mock_cache_inst):
        _render_cache_health()

    mock_st.markdown.assert_called()
    html_calls = [str(c) for c in mock_st.markdown.call_args_list]
    assert any("ts-stat-card" in h for h in html_calls)


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
    """At least some packages should show as available."""
    from frontend.pages.diagnostics import _render_data_source_status

    _render_data_source_status()
    args = mock_st.dataframe.call_args
    df = args[0][0]
    statuses = df["Status"].tolist()
    assert any("Available" in s for s in statuses)


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


def test_model_groups_has_four_groups():
    """MODEL_GROUPS constant has exactly 4 groups."""
    assert len(MODEL_GROUPS) == 4
    assert set(MODEL_GROUPS.keys()) == {"shooting", "passing", "dribbling", "defending"}
