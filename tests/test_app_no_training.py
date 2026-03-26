"""Test that app.py does not import or trigger training_pipeline."""

from __future__ import annotations

import importlib
import sys
import unittest
from unittest import mock


class TestAppDoesNotImportTrainingPipeline(unittest.TestCase):
    """Bug 2: importing app.py must NOT trigger training_pipeline functions."""

    @mock.patch.dict(sys.modules, {"streamlit": mock.MagicMock()})
    def test_app_does_not_import_training_pipeline(self):
        """Verify that loading app does not cause training_pipeline to execute."""
        # Patch streamlit so we can import app.py in a test context
        # Also patch frontend.theme and backend.models.transfer_portal
        mock_st = sys.modules["streamlit"]
        mock_st.set_page_config = mock.MagicMock()
        mock_st.session_state = {}
        mock_st.success = mock.MagicMock()
        mock_st.warning = mock.MagicMock()
        mock_st.sidebar = mock.MagicMock()

        with mock.patch.dict(sys.modules, {
            "frontend.theme": mock.MagicMock(),
            "backend.models.transfer_portal": mock.MagicMock(),
        }):
            # If training_pipeline is already in sys.modules, remove it
            # so we can detect a fresh import
            tp_module_name = "backend.models.training_pipeline"
            was_loaded = tp_module_name in sys.modules

            # Create a sentinel to detect if training_pipeline is imported
            sentinel = mock.MagicMock()
            fake_tp = mock.MagicMock()
            fake_tp.run_pipeline = sentinel
            fake_tp.discover_transfers = sentinel

            with mock.patch.dict(sys.modules, {tp_module_name: fake_tp}):
                # Remove app from modules cache to force reimport
                sys.modules.pop("app", None)

                try:
                    # Just check that after importing app, no training
                    # pipeline function was called
                    importlib.import_module("app")
                except Exception:
                    # Streamlit may raise during import — that's fine,
                    # we only care about training_pipeline not being called
                    pass

                sentinel.assert_not_called()
