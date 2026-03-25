"""Unit tests for adjustment model training utilities."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

_TEMP_DIR = tempfile.mkdtemp(prefix="ts_adj_model_test_")
os.environ["CACHE_DIR"] = _TEMP_DIR

from backend.data.sofascore_client import CORE_METRICS
from backend.features.adjustment_models import (
    TeamAdjustmentModel,
    PlayerAdjustmentModel,
    build_training_data_from_transfers,
    auto_train_from_player_history,
    scale_team_position_features,
)
from backend.data import cache


def tearDownModule():
    """Clean up the temp cache directory after all tests."""
    cache.close()
    shutil.rmtree(_TEMP_DIR, ignore_errors=True)


class TestBuildTrainingData(unittest.TestCase):
    @patch("backend.features.power_rankings.get_team_ranking")
    @patch("backend.data.sofascore_client.get_player_stats")
    @patch("backend.data.sofascore_client.get_player_transfer_history")
    def test_builds_rows_from_transfer_history(self, mock_transfers, mock_stats, mock_ranking):
        mock_transfers.return_value = [
            {
                "transfer_date": "2023-07-01",
                "from_team": {"id": 100, "name": "AC Milan"},
                "to_team": {"id": 42, "name": "Arsenal"},
                "type": "transfer",
            },
            {
                "transfer_date": "2021-07-01",
                "from_team": {"id": 200, "name": "Lille"},
                "to_team": {"id": 100, "name": "AC Milan"},
                "type": "transfer",
            },
        ]

        per90 = {m: 1.5 for m in CORE_METRICS}
        mock_stats.return_value = {
            "per90": per90,
            "position": "Forward",
            "minutes_played": 1500,
        }

        ranking_new = MagicMock()
        ranking_new.relative_ability = 10.0
        ranking_new.league_mean_normalized = 60.0
        ranking_new.normalized_score = 70.0

        ranking_old = MagicMock()
        ranking_old.relative_ability = 5.0
        ranking_old.league_mean_normalized = 55.0
        ranking_old.normalized_score = 60.0

        def mock_get_team_ranking(name, *a, **kw):
            if name == "Arsenal":
                return ranking_new
            elif name == "AC Milan":
                return ranking_old
            return None

        mock_ranking.side_effect = mock_get_team_ranking

        team_rows, player_rows = build_training_data_from_transfers(961995)

        # Should produce 13 rows (one per core metric) for each model
        self.assertEqual(len(team_rows), 13)
        self.assertEqual(len(player_rows), 13)

        # Verify team row structure
        tr = team_rows[0]
        self.assertIn("metric", tr)
        self.assertIn("team_relative_feature", tr)
        self.assertIn("naive_league_expectation", tr)
        self.assertIn("actual", tr)

        # Verify player row structure
        pr = player_rows[0]
        self.assertIn("position", pr)
        self.assertIn("metric", pr)
        self.assertIn("change_relative_ability", pr)
        self.assertEqual(pr["change_relative_ability"], 5.0)  # 10.0 - 5.0

    @patch("backend.features.power_rankings.get_team_ranking")
    @patch("backend.data.sofascore_client.get_player_transfer_history")
    def test_empty_transfer_history(self, mock_transfers, mock_ranking):
        mock_transfers.return_value = []

        team_rows, player_rows = build_training_data_from_transfers(999)
        self.assertEqual(team_rows, [])
        self.assertEqual(player_rows, [])


class TestAutoTrain(unittest.TestCase):
    @patch("backend.features.power_rankings.get_team_ranking")
    @patch("backend.data.sofascore_client.get_player_stats")
    @patch("backend.data.sofascore_client.get_player_transfer_history")
    def test_auto_train_produces_fitted_models(self, mock_transfers, mock_stats, mock_ranking):
        mock_transfers.return_value = [
            {
                "transfer_date": "2023-07-01",
                "from_team": {"id": 100, "name": "Team A"},
                "to_team": {"id": 42, "name": "Team B"},
                "type": "transfer",
            },
            {
                "transfer_date": "2021-07-01",
                "from_team": {"id": 200, "name": "Team C"},
                "to_team": {"id": 100, "name": "Team A"},
                "type": "transfer",
            },
        ]

        per90 = {m: float(i) + 0.5 for i, m in enumerate(CORE_METRICS)}
        mock_stats.return_value = {
            "per90": per90,
            "position": "Midfielder",
            "minutes_played": 2000,
        }

        ranking = MagicMock()
        ranking.relative_ability = 5.0
        ranking.league_mean_normalized = 50.0
        ranking.normalized_score = 55.0
        mock_ranking.return_value = ranking

        team_model, player_model = auto_train_from_player_history([111, 222])

        self.assertIsInstance(team_model, TeamAdjustmentModel)
        self.assertIsInstance(player_model, PlayerAdjustmentModel)
        self.assertTrue(team_model.fitted)
        self.assertTrue(player_model.fitted)


class TestScaleTeamPositionFeatures(unittest.TestCase):
    def test_scaling(self):
        current_pos = {"expected_goals": 2.0, "expected_assists": 1.0}
        current_team = {"expected_goals": 4.0, "expected_assists": 2.0}
        adjusted_team = {"expected_goals": 2.0, "expected_assists": 3.0}

        result = scale_team_position_features(current_pos, current_team, adjusted_team)

        # xG: team dropped 50% (4→2), so pos should drop 50% (2→1)
        self.assertAlmostEqual(result["expected_goals"], 1.0)
        # xA: team increased 50% (2→3), so pos should increase 50% (1→1.5)
        self.assertAlmostEqual(result["expected_assists"], 1.5)


if __name__ == "__main__":
    unittest.main()
