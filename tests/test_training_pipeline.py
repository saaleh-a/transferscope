"""Tests for backend.models.training_pipeline."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from backend.data.sofascore_client import CORE_METRICS
from backend.models.transfer_portal import FEATURE_DIM

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mock_per90(base: float = 0.5) -> dict:
    """Return a per90 dict with all metrics set to base value."""
    return {m: base for m in CORE_METRICS}


def _make_mock_season_stats(minutes: int = 900, base: float = 0.5) -> dict:
    """Return a mock result from get_player_stats_for_season."""
    return {
        "name": "Test Player",
        "team": "Test FC",
        "team_id": 100,
        "position": "Forward",
        "minutes_played": minutes,
        "appearances": 10,
        "per90": _make_mock_per90(base),
        "raw": {},
    }


def _make_mock_transfer_record():
    """Return a TransferRecord for testing."""
    from backend.models.training_pipeline import TransferRecord

    return TransferRecord(
        player_id=1001,
        player_name="Test Player",
        position="Forward",
        from_club_id=200,
        from_club_name="Old FC",
        from_league_id=17,
        to_club_id=300,
        to_club_name="New FC",
        to_league_id=8,
        transfer_date="2023-07-01",
        pre_transfer_season_id=40000,
        post_transfer_season_id=50000,
        pre_transfer_tournament_id=17,
        post_transfer_tournament_id=8,
    )


def _make_mock_team_ranking(name, score=60.0, league_code="ENG1", league_mean=50.0):
    from backend.features.power_rankings import TeamRanking
    return TeamRanking(
        team_name=name,
        league_code=league_code,
        raw_elo=1500.0,
        normalized_score=score,
        league_mean_normalized=league_mean,
        relative_ability=score - league_mean,
    )


def _make_mock_league_snapshot(code="ENG1"):
    from backend.features.power_rankings import LeagueSnapshot
    from datetime import date
    return LeagueSnapshot(
        league_code=code,
        league_name="Test League",
        date=date.today(),
        mean_elo=1500.0,
        std_elo=100.0,
        p10=30.0, p25=40.0, p50=50.0, p75=60.0, p90=70.0,
        mean_normalized=50.0,
        team_count=20,
    )


# ── Test Cases ───────────────────────────────────────────────────────────────


class TestDiscoverTransfers(unittest.TestCase):
    """Test discover_transfers returns a list of TransferRecord."""

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    def test_discover_transfers_returns_list(self, mock_client):
        from backend.models.training_pipeline import discover_transfers

        mock_client.get_season_list.return_value = [
            {"id": 100, "name": "2023/2024"},
            {"id": 200, "name": "2022/2023"},
        ]
        mock_client.get_league_player_stats.return_value = [
            {
                "id": 1001,
                "name": "Player A",
                "position": "Forward",
                "minutes_played": 1000,
            }
        ]
        mock_client.get_player_transfer_history.return_value = [
            {
                "transfer_date": "2023-07-01",
                "from_team": {"id": 10, "name": "Club A"},
                "to_team": {"id": 20, "name": "Club B"},
            }
        ]
        mock_client.get_player_stats_for_season.return_value = _make_mock_season_stats(900)

        result = discover_transfers(["ENG1"], seasons_back=2)
        self.assertIsInstance(result, list)


class TestBuildTrainingSample(unittest.TestCase):
    """Test build_training_sample produces correct shapes."""

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    @mock.patch("backend.models.training_pipeline.power_rankings")
    def test_build_training_sample_correct_shape(self, mock_pr, mock_client):
        from backend.models.training_pipeline import build_training_sample

        record = _make_mock_transfer_record()

        mock_client.get_player_stats_for_season.return_value = _make_mock_season_stats(900)
        mock_client.get_team_position_averages.return_value = (
            _make_mock_per90(0.3), []
        )

        mock_pr.compute_daily_rankings.return_value = (
            {
                "Old FC": _make_mock_team_ranking("Old FC", 65.0),
                "New FC": _make_mock_team_ranking("New FC", 55.0, "ESP1"),
            },
            {
                "ENG1": _make_mock_league_snapshot("ENG1"),
                "ESP1": _make_mock_league_snapshot("ESP1"),
            },
        )
        mock_pr.get_team_ranking.side_effect = lambda name: {
            "Old FC": _make_mock_team_ranking("Old FC", 65.0),
            "New FC": _make_mock_team_ranking("New FC", 55.0, "ESP1"),
        }.get(name)

        result = build_training_sample(record)
        self.assertIsNotNone(result)
        self.assertEqual(result["features"].shape, (FEATURE_DIM,))
        self.assertEqual(result["labels"].shape, (len(CORE_METRICS),))

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    @mock.patch("backend.models.training_pipeline.power_rankings")
    def test_build_training_sample_handles_missing_metrics(self, mock_pr, mock_client):
        from backend.models.training_pipeline import build_training_sample

        record = _make_mock_transfer_record()

        # Per90 with some metrics missing (None)
        partial_per90 = {m: None for m in CORE_METRICS}
        partial_per90["expected_goals"] = 0.5
        partial_per90["shots"] = 2.0

        stats = _make_mock_season_stats(900)
        stats["per90"] = partial_per90
        mock_client.get_player_stats_for_season.return_value = stats
        mock_client.get_team_position_averages.return_value = (
            _make_mock_per90(0.3), []
        )

        mock_pr.compute_daily_rankings.return_value = ({}, {})
        mock_pr.get_team_ranking.return_value = None

        result = build_training_sample(record)
        self.assertIsNotNone(result)
        # No NaN values
        self.assertFalse(np.any(np.isnan(result["features"])))
        self.assertFalse(np.any(np.isnan(result["labels"])))

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    @mock.patch("backend.models.training_pipeline.power_rankings")
    def test_build_training_sample_returns_none_for_low_minutes(self, mock_pr, mock_client):
        from backend.models.training_pipeline import build_training_sample

        record = _make_mock_transfer_record()

        # Pre-transfer has enough minutes but post-transfer has too few
        pre_stats = _make_mock_season_stats(900)
        post_stats = _make_mock_season_stats(400)  # Below 450 threshold

        mock_client.get_player_stats_for_season.side_effect = [pre_stats, post_stats]
        mock_client.get_team_position_averages.return_value = (
            _make_mock_per90(0.3), []
        )
        mock_pr.compute_daily_rankings.return_value = ({}, {})
        mock_pr.get_team_ranking.return_value = None

        result = build_training_sample(record)
        self.assertIsNone(result)


class TestSplitDataset(unittest.TestCase):
    """Test temporal splitting."""

    def test_split_dataset_is_temporal(self):
        from backend.models.training_pipeline import split_dataset

        n = 100
        X = np.random.randn(n, FEATURE_DIM).astype(np.float32)
        y = np.random.randn(n, len(CORE_METRICS)).astype(np.float32)

        # Create metadata with known dates incrementing across years
        base_year = 2018
        dates = []
        for i in range(n):
            year = base_year + i // 12
            month = (i % 12) + 1
            dates.append(f"{year}-{month:02d}-15")
        metadata = [{"transfer_date": d, "player_id": i} for i, d in enumerate(dates)]

        result = split_dataset(X, y, metadata, val_ratio=0.15, test_ratio=0.10)
        (X_train, y_train, X_val, y_val, X_test, y_test,
         meta_train, meta_val, meta_test) = result

        # Verify temporal ordering: all test dates >= all val dates >= all train dates
        if meta_train and meta_val:
            max_train = max(m["transfer_date"] for m in meta_train)
            min_val = min(m["transfer_date"] for m in meta_val)
            self.assertLessEqual(max_train, min_val)

        if meta_val and meta_test:
            max_val = max(m["transfer_date"] for m in meta_val)
            min_test = min(m["transfer_date"] for m in meta_test)
            self.assertLessEqual(max_val, min_test)

    def test_split_sizes(self):
        from backend.models.training_pipeline import split_dataset

        n = 100
        X = np.random.randn(n, FEATURE_DIM).astype(np.float32)
        y = np.random.randn(n, len(CORE_METRICS)).astype(np.float32)
        metadata = [{"transfer_date": f"2020-01-{i+1:02d}", "player_id": i} for i in range(n)]

        result = split_dataset(X, y, metadata, val_ratio=0.15, test_ratio=0.10)
        (X_train, _, X_val, _, X_test, _, meta_train, meta_val, meta_test) = result

        self.assertEqual(len(meta_train) + len(meta_val) + len(meta_test), n)
        self.assertGreater(len(meta_test), 0)
        self.assertGreater(len(meta_val), 0)
        self.assertGreater(len(meta_train), 0)


class TestFullDatasetNoNans(unittest.TestCase):
    """Test that build_full_dataset produces no NaN values."""

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    @mock.patch("backend.models.training_pipeline.power_rankings")
    def test_full_dataset_no_nans(self, mock_pr, mock_client):
        from backend.models.training_pipeline import TransferRecord, build_full_dataset

        mock_client.get_player_stats_for_season.return_value = _make_mock_season_stats(900)
        mock_client.get_team_position_averages.return_value = (
            _make_mock_per90(0.3), []
        )
        mock_pr.compute_daily_rankings.return_value = ({}, {})
        mock_pr.get_team_ranking.return_value = None

        records = [
            TransferRecord(
                player_id=i,
                player_name=f"Player {i}",
                position="Forward",
                from_club_id=i * 10,
                from_club_name=f"From {i}",
                from_league_id=17,
                to_club_id=i * 10 + 1,
                to_club_name=f"To {i}",
                to_league_id=17,
                transfer_date=f"2023-07-{i+1:02d}",
                pre_transfer_season_id=40000,
                post_transfer_season_id=50000,
                pre_transfer_tournament_id=17,
                post_transfer_tournament_id=17,
            )
            for i in range(5)
        ]

        X, y, metadata = build_full_dataset(records)
        self.assertGreater(len(metadata), 0)
        self.assertFalse(np.any(np.isnan(X)))
        self.assertFalse(np.any(np.isnan(y)))


if __name__ == "__main__":
    unittest.main()
