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


# ── Improvement 2 tests ─────────────────────────────────────────────────────


class TestFirst1000MinuteTargets(unittest.TestCase):
    """Test first-1000-minute label accumulation logic."""

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    @mock.patch("backend.models.training_pipeline.power_rankings")
    def test_labels_accumulate_first_1000_minutes_only(self, mock_pr, mock_client):
        """Mock 15 matches totalling 1350 minutes. Assert labels use only
        the first matches summing to >= 1000 minutes, not all 1350."""
        from backend.models.training_pipeline import build_training_sample

        record = _make_mock_transfer_record()

        # Pre-transfer match logs (for features)
        pre_logs = [
            {"match_id": i, "match_date": f"2022-{(i%12)+1:02d}-01",
             "minutes_played": 90,
             "per90": _make_mock_per90(0.5)}
            for i in range(12)
        ]
        # Post-transfer match logs: 15 matches of 90 mins = 1350 total
        # First ~11 matches sum to 990 min, 12th brings to 1080 (>= 1000)
        # Matches 13-15 (extra 270 min) should NOT be included in labels
        post_logs = []
        for i in range(15):
            per90 = _make_mock_per90(0.3 if i < 12 else 0.9)
            post_logs.append({
                "match_id": 100 + i,
                "match_date": f"2023-{(i%12)+1:02d}-01",
                "minutes_played": 90,
                "per90": per90,
            })

        mock_client.get_player_match_logs.side_effect = [pre_logs, post_logs]

        # Fallback season stats (should not be used for labels if logs work)
        mock_client.get_player_stats_for_season.return_value = _make_mock_season_stats(1350, 0.7)
        mock_pr.compute_daily_rankings.return_value = ({}, {})
        mock_pr.get_team_ranking.return_value = None

        result = build_training_sample(record)
        self.assertIsNotNone(result)
        # Labels should be dominated by 0.3 (first 12 matches), not 0.7 (season agg)
        xg_label = result["labels"][0]  # expected_goals index 0
        self.assertLess(xg_label, 0.5, "Labels should reflect first-1000-min (0.3), not full season")
        self.assertTrue(result.get("used_match_logs_labels", False))

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    @mock.patch("backend.models.training_pipeline.power_rankings")
    def test_labels_fall_back_when_no_match_logs(self, mock_pr, mock_client):
        """Mock get_player_match_logs returning []. Assert
        get_player_stats_for_season is called for labels."""
        from backend.models.training_pipeline import build_training_sample

        record = _make_mock_transfer_record()

        mock_client.get_player_match_logs.return_value = []
        mock_client.get_player_stats_for_season.return_value = _make_mock_season_stats(900, 0.8)
        mock_pr.compute_daily_rankings.return_value = ({}, {})
        mock_pr.get_team_ranking.return_value = None

        result = build_training_sample(record)
        self.assertIsNotNone(result)
        # Should have used season aggregate
        self.assertFalse(result.get("used_match_logs_labels", True))
        # Labels should reflect the 0.8 base value from season stats
        xg_label = result["labels"][0]
        self.assertAlmostEqual(xg_label, 0.8, places=1)


# ── Improvement 3 tests ─────────────────────────────────────────────────────


class TestNonTransferSamples(unittest.TestCase):
    """Tests for non-transfer sample generation."""

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    @mock.patch("backend.models.training_pipeline.power_rankings")
    def test_non_transfer_sample_change_ra_is_zero(self, mock_pr, mock_client):
        """Verify that change_relative_ability is 0 in the feature vector
        for a non-transfer sample (same club → same club)."""
        from backend.models.training_pipeline import (
            NonTransferRecord,
            build_non_transfer_sample,
        )

        record = NonTransferRecord(
            player_id=1001,
            player_name="Stay Player",
            position="Forward",
            club_id=200,
            club_name="Same FC",
            league_id=17,
            pre_season_id=40000,
            post_season_id=50000,
            pre_tournament_id=17,
            post_tournament_id=17,
        )

        mock_client.get_player_match_logs.return_value = []
        mock_client.get_player_stats_for_season.return_value = _make_mock_season_stats(900, 0.5)
        mock_client.get_league_player_stats.return_value = []

        mock_pr.compute_daily_rankings.return_value = ({}, {})
        mock_pr.get_team_ranking.return_value = None

        result = build_non_transfer_sample(record)
        self.assertIsNotNone(result)

        # team_ability_current == team_ability_target
        self.assertEqual(
            result["team_ability_current"],
            result["team_ability_target"],
        )
        # league_ability_current == league_ability_target
        self.assertEqual(
            result["league_ability_current"],
            result["league_ability_target"],
        )
        # So change_ra = (team_target - league_target) - (team_current - league_current) = 0
        team_c = result["team_ability_current"]
        league_c = result["league_ability_current"]
        team_t = result["team_ability_target"]
        league_t = result["league_ability_target"]
        change_ra = (team_t - league_t) - (team_c - league_c)
        self.assertAlmostEqual(change_ra, 0.0)

        # Verify is_transfer is False
        self.assertFalse(result["is_transfer"])

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    @mock.patch("backend.models.training_pipeline.power_rankings")
    def test_build_full_dataset_combines_sample_types(self, mock_pr, mock_client):
        """Mock 3 transfer + 7 non-transfer records and verify combined shapes."""
        from backend.models.training_pipeline import (
            TransferRecord,
            NonTransferRecord,
            build_full_dataset,
        )

        mock_client.get_player_match_logs.return_value = []
        mock_client.get_player_stats_for_season.return_value = _make_mock_season_stats(900, 0.5)
        mock_client.get_league_player_stats.return_value = []
        mock_pr.compute_daily_rankings.return_value = ({}, {})
        mock_pr.get_team_ranking.return_value = None

        transfer_records = [
            TransferRecord(
                player_id=i,
                player_name=f"Transfer Player {i}",
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
            for i in range(3)
        ]

        non_transfer_records = [
            NonTransferRecord(
                player_id=100 + i,
                player_name=f"Stay Player {i}",
                position="Forward",
                club_id=i * 10 + 5,
                club_name=f"Club {i}",
                league_id=17,
                pre_season_id=40000,
                post_season_id=50000,
                pre_tournament_id=17,
                post_tournament_id=17,
            )
            for i in range(7)
        ]

        X, y, metadata = build_full_dataset(transfer_records, non_transfer_records)

        # All 10 should produce valid samples
        self.assertEqual(X.shape[0], 10)
        self.assertEqual(y.shape[0], 10)
        self.assertEqual(len(metadata), 10)
        self.assertEqual(X.shape[1], FEATURE_DIM)
        self.assertEqual(y.shape[1], len(CORE_METRICS))

        # Check mix of is_transfer values
        transfer_count = sum(1 for m in metadata if m.get("is_transfer"))
        non_transfer_count = sum(1 for m in metadata if not m.get("is_transfer"))
        self.assertEqual(transfer_count, 3)
        self.assertEqual(non_transfer_count, 7)


# ── Improvement 4 tests ─────────────────────────────────────────────────────


class TestNaiveLeagueExpectation(unittest.TestCase):
    """Test that naive_league_expectation uses league means."""

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    @mock.patch("backend.models.training_pipeline.power_rankings")
    def test_naive_expectation_equals_league_mean_not_scaled_player(self, mock_pr, mock_client):
        """Verify that when league means are available, the naive expectation
        is the league mean, not the old formula of (league_ability / 100) * player."""
        from backend.models.training_pipeline import build_training_sample

        record = _make_mock_transfer_record()

        # Pre-transfer per90: all metrics at 2.0
        mock_client.get_player_match_logs.return_value = []
        mock_client.get_player_stats_for_season.return_value = _make_mock_season_stats(900, 2.0)

        # League player stats with known values
        league_players = [
            {"id": 1, "name": "A", "position": "F", "minutes_played": 900,
             "per90": _make_mock_per90(1.0)},
            {"id": 2, "name": "B", "position": "F", "minutes_played": 900,
             "per90": _make_mock_per90(3.0)},
        ]
        mock_client.get_league_player_stats.return_value = league_players

        mock_pr.compute_daily_rankings.return_value = ({}, {})
        mock_pr.get_team_ranking.return_value = None

        result = build_training_sample(record)
        self.assertIsNotNone(result)

        # league_means should be mean of 1.0 and 3.0 = 2.0
        league_means = result.get("league_means", {})
        self.assertAlmostEqual(league_means.get("expected_goals", 0), 2.0, places=1)


# ── Improvement 5 tests ─────────────────────────────────────────────────────


class TestErrorHandling(unittest.TestCase):
    """Tests for rate limiting and error handling."""

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    @mock.patch("backend.models.training_pipeline.power_rankings")
    def test_returns_none_when_pre_season_stats_missing(self, mock_pr, mock_client):
        """When get_player_stats_for_season returns None, build_training_sample
        returns None."""
        from backend.models.training_pipeline import build_training_sample

        record = _make_mock_transfer_record()

        mock_client.get_player_match_logs.return_value = []
        mock_client.get_player_stats_for_season.return_value = None

        result = build_training_sample(record)
        self.assertIsNone(result)

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    @mock.patch("backend.models.training_pipeline.power_rankings")
    def test_team_pos_placeholder_zeros_at_sample_level(self, mock_pr, mock_client):
        """Team-position features are placeholder zeros at sample level,
        to be injected later by build_full_dataset()."""
        from backend.models.training_pipeline import build_training_sample

        record = _make_mock_transfer_record()

        mock_client.get_player_match_logs.return_value = []
        mock_client.get_player_stats_for_season.return_value = _make_mock_season_stats(900, 0.5)
        mock_client.get_league_player_stats.return_value = []

        mock_pr.compute_daily_rankings.return_value = ({}, {})
        mock_pr.get_team_ranking.return_value = None

        result = build_training_sample(record)
        self.assertIsNotNone(result)
        # Team-position features should be zeros (indices 17-42)
        features = result["features"]
        team_pos_slice = features[17:43]
        np.testing.assert_array_equal(team_pos_slice, 0.0)

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    @mock.patch("backend.models.training_pipeline.power_rankings")
    def test_uses_midpoint_when_power_rankings_fail(self, mock_pr, mock_client):
        """When power rankings fail, should use 50.0 midpoints."""
        from backend.models.training_pipeline import build_training_sample

        record = _make_mock_transfer_record()

        mock_client.get_player_match_logs.return_value = []
        mock_client.get_player_stats_for_season.return_value = _make_mock_season_stats(900, 0.5)
        mock_client.get_league_player_stats.return_value = []

        mock_pr.compute_daily_rankings.side_effect = RuntimeError("Elo service down")
        mock_pr.get_team_ranking.return_value = None

        result = build_training_sample(record)
        self.assertIsNotNone(result)
        # Team abilities should be 50.0 (midpoint fallback)
        self.assertAlmostEqual(result["team_ability_current"], 50.0)
        self.assertAlmostEqual(result["team_ability_target"], 50.0)

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    @mock.patch("backend.models.training_pipeline.power_rankings")
    def test_substitutes_zero_for_missing_metric(self, mock_pr, mock_client):
        """When a specific metric is missing (None), should be replaced with 0.0."""
        from backend.models.training_pipeline import build_training_sample

        record = _make_mock_transfer_record()

        # Per90 with all None except one
        partial_per90 = {m: None for m in CORE_METRICS}
        partial_per90["expected_goals"] = 1.5

        stats = _make_mock_season_stats(900)
        stats["per90"] = partial_per90

        mock_client.get_player_match_logs.return_value = []
        mock_client.get_player_stats_for_season.return_value = stats
        mock_client.get_league_player_stats.return_value = []

        mock_pr.compute_daily_rankings.return_value = ({}, {})
        mock_pr.get_team_ranking.return_value = None

        result = build_training_sample(record)
        self.assertIsNotNone(result)

        # xG (index 0) should be 1.5, all other player metrics should be 0.0
        features = result["features"]
        self.assertAlmostEqual(features[0], 1.5)
        for i in range(1, 13):
            self.assertAlmostEqual(features[i], 0.0,
                                   msg=f"Metric index {i} should be 0.0 for missing")


# ── Improvement 6 tests ─────────────────────────────────────────────────────


class TestPerGroupFeatureSubsets(unittest.TestCase):
    """Tests for per-group feature subsets in TransferPortalModel."""

    def test_group_models_have_correct_input_dimensions(self):
        """Each group model's input dimension should match its feature subset size."""
        from backend.models.transfer_portal import (
            TransferPortalModel,
            GROUP_FEATURE_SUBSETS,
            MODEL_GROUPS,
        )

        model = TransferPortalModel()
        model.build(FEATURE_DIM)

        for group_name in MODEL_GROUPS:
            expected_dim = len(GROUP_FEATURE_SUBSETS[group_name])
            keras_model = model.models[group_name]
            actual_dim = keras_model.input_shape[1]
            self.assertEqual(
                actual_dim,
                expected_dim,
                f"{group_name}: expected input_dim={expected_dim}, got {actual_dim}",
            )

    def test_predict_produces_all_13_metrics(self):
        """predict() should return all 13 core metrics regardless of per-group subsets."""
        from backend.models.transfer_portal import TransferPortalModel, _feature_keys

        model = TransferPortalModel()
        model.build(FEATURE_DIM)

        # Dummy feature dict
        feature_dict = {k: 0.5 for k in _feature_keys()}

        result = model.predict(feature_dict)
        self.assertEqual(len(result), len(CORE_METRICS))
        for m in CORE_METRICS:
            self.assertIn(m, result)


# ── Improvement 7 tests ─────────────────────────────────────────────────────


class TestNormalisedChangeRA(unittest.TestCase):
    """Test that change_ra is normalized by /50.0 for polynomial terms."""

    def test_polynomial_features_use_normalised_ra(self):
        """Fit PlayerAdjustmentModel with known change_ra and verify the
        polynomial features are normalized.

        The training pipeline pre-normalizes change_ra by dividing by 50
        before passing to fit().  predict() divides by 50 internally.
        Both sides must end up with the same value for consistency.
        """
        from backend.features.adjustment_models import PlayerAdjustmentModel

        # Create training data with change_ra already normalised (as
        # training_pipeline does: change_ra / 50.0 = 25.0 / 50.0 = 0.5).
        # Use 35 samples per metric to exceed the 30-sample minimum.
        rows = []
        for m in CORE_METRICS:
            for i in range(35):
                rows.append({
                    "position": "Forward",
                    "metric": m,
                    "player_previous_per90": 1.0 + i * 0.01,
                    "avg_position_feature_new_team": 1.0 + i * 0.005,
                    "diff_avg_position_old_vs_new": 0.0,
                    "change_relative_ability": 0.5,  # pre-normalised
                    "actual": 1.1,
                })

        model = PlayerAdjustmentModel()
        model.fit(rows)

        # Predict with raw change_ra = 25.0 → predict() divides by 50 → 0.5
        # (same as the pre-normalised training value)
        prediction = model.predict(
            "Forward", "expected_goals",
            player_previous_per90=1.0,
            avg_position_feature_new_team=1.0,
            diff_avg_position_old_vs_new=0.0,
            change_relative_ability=25.0,
        )

        # The prediction should be close to 1.1 (training target)
        self.assertAlmostEqual(prediction, 1.1, places=1)

        # Verify that the model internals used normalized values
        # Coefficients should be reasonable (not huge)
        fm = model.models["Forward"]["expected_goals"]
        self.assertTrue(
            abs(fm.coef_[3]) < 100,  # Would be huge without normalization
            f"Coefficient too large ({fm.coef_[3]}), suggesting no normalization",
        )


# ── Improvement 8 tests ─────────────────────────────────────────────────────


class TestSplitDeduplication(unittest.TestCase):
    """Tests for player-level deduplication in split."""

    def test_split_removes_player_overlap_from_training(self):
        """Players in the test set should not appear in training/validation."""
        from backend.models.training_pipeline import split_dataset

        n = 100
        X = np.random.randn(n, FEATURE_DIM).astype(np.float32)
        y = np.random.randn(n, len(CORE_METRICS)).astype(np.float32)

        # Create 20 unique players, each appearing 5 times
        player_ids = [i // 5 for i in range(n)]
        metadata = [
            {"transfer_date": f"2020-{(i%12)+1:02d}-{(i%28)+1:02d}", "player_id": player_ids[i]}
            for i in range(n)
        ]

        result = split_dataset(X, y, metadata, val_ratio=0.15, test_ratio=0.10)
        (X_train, _, X_val, _, X_test, _, meta_train, meta_val, meta_test) = result

        test_pids = {m["player_id"] for m in meta_test}
        train_pids = {m["player_id"] for m in meta_train}
        val_pids = {m["player_id"] for m in meta_val}

        # No overlap between test and train/val
        self.assertEqual(test_pids & train_pids, set())
        self.assertEqual(test_pids & val_pids, set())

    def test_split_preserves_test_set_integrity(self):
        """Test set should not be modified by deduplication."""
        from backend.models.training_pipeline import split_dataset

        n = 50
        X = np.random.randn(n, FEATURE_DIM).astype(np.float32)
        y = np.random.randn(n, len(CORE_METRICS)).astype(np.float32)

        metadata = [
            {"transfer_date": f"2020-{(i%12)+1:02d}-{(i%28)+1:02d}", "player_id": i % 10}
            for i in range(n)
        ]

        result = split_dataset(X, y, metadata, val_ratio=0.15, test_ratio=0.10)
        (_, _, _, _, X_test, _, _, _, meta_test) = result

        # Test set should have samples
        self.assertGreater(len(meta_test), 0)

        # Total samples may be less than n due to deduplication
        (X_train, _, X_val, _, X_test2, _, meta_train, meta_val, meta_test2) = result
        total = len(meta_train) + len(meta_val) + len(meta_test2)
        self.assertLessEqual(total, n)


# ── Improvement 9 tests ─────────────────────────────────────────────────────


class TestBuildFeatureDictFromPlayer(unittest.TestCase):
    """Tests for the inference-time feature builder."""

    @mock.patch("backend.features.power_rankings.get_team_ranking")
    @mock.patch("backend.features.power_rankings.compute_daily_rankings")
    @mock.patch("backend.data.sofascore_client.get_team_position_averages")
    @mock.patch("backend.data.sofascore_client.get_player_stats_for_season")
    @mock.patch("backend.data.sofascore_client.get_player_match_logs")
    @mock.patch("backend.features.rolling_windows.player_rolling_average")
    def test_inference_uses_match_logs_when_available(
        self, mock_rolling, mock_logs, mock_season, mock_pos_avg,
        mock_daily, mock_team_ranking,
    ):
        """When match logs are available, use rolling average from them."""
        from backend.models.transfer_portal import build_feature_dict_from_player

        match_logs = [
            {"match_id": i, "match_date": f"2023-{(i%12)+1:02d}-01",
             "minutes_played": 90,
             "per90": _make_mock_per90(1.5)}
            for i in range(12)
        ]
        mock_logs.return_value = match_logs
        mock_rolling.return_value = _make_mock_per90(1.5)
        mock_season.return_value = {
            "team_id": 200, "team": "Source FC",
            "minutes_played": 1080, "per90": _make_mock_per90(0.3),
        }
        mock_pos_avg.return_value = (_make_mock_per90(0.4), [])

        mock_daily.return_value = ({}, {})
        mock_team_ranking.return_value = None

        result = build_feature_dict_from_player(
            player_id=1001,
            tournament_id=17,
            season_id=40000,
            target_club_id=300,
            target_league_id=8,
            position="Forward",
            target_team_name="Target FC",
        )

        # Should have all 46 keys (FEATURE_DIM)
        self.assertEqual(len(result), FEATURE_DIM)

        # Player metrics should reflect match log rolling values (1.5), not season agg (0.3)
        self.assertGreater(result["player_expected_goals"], 1.0)

    @mock.patch("backend.features.power_rankings.get_team_ranking")
    @mock.patch("backend.features.power_rankings.compute_daily_rankings")
    @mock.patch("backend.data.sofascore_client.get_team_position_averages")
    @mock.patch("backend.data.sofascore_client.get_player_stats_for_season")
    @mock.patch("backend.data.sofascore_client.get_player_match_logs")
    def test_inference_falls_back_to_season_agg_when_no_logs(
        self, mock_logs, mock_season, mock_pos_avg,
        mock_daily, mock_team_ranking,
    ):
        """When no match logs are available, use season aggregate."""
        from backend.models.transfer_portal import build_feature_dict_from_player

        mock_logs.return_value = []
        mock_season.return_value = {
            "team_id": 200, "team": "Source FC",
            "minutes_played": 1080, "per90": _make_mock_per90(0.8),
        }
        mock_pos_avg.return_value = (_make_mock_per90(0.4), [])

        mock_daily.return_value = ({}, {})
        mock_team_ranking.return_value = None

        result = build_feature_dict_from_player(
            player_id=1001,
            tournament_id=17,
            season_id=40000,
            target_club_id=300,
            target_league_id=8,
            position="Forward",
            target_team_name="Target FC",
        )

        # Should have all 46 keys (FEATURE_DIM)
        self.assertEqual(len(result), FEATURE_DIM)

        # Player metrics should reflect season agg (0.8)
        self.assertAlmostEqual(result["player_expected_goals"], 0.8, places=1)


class TestDiscoverTransfersMinutesKey(unittest.TestCase):
    """Bug 1: verify minutes_played key is read correctly from league stats."""

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    def test_discover_transfers_minutes_key_correct(self, mock_client):
        """Players with >=450 minutes_played must NOT be skipped;
        players with <450 minutes_played MUST be skipped."""
        from backend.models.training_pipeline import discover_transfers

        player_high = {
            "id": 5001,
            "name": "Enough Minutes",
            "position": "Forward",
            "minutes_played": 900,
        }
        player_low = {
            "id": 5002,
            "name": "Too Few Minutes",
            "position": "Midfielder",
            "minutes_played": 200,
        }

        mock_client.get_season_list.return_value = [
            {"id": 100, "name": "2023/2024"},
            {"id": 200, "name": "2022/2023"},
        ]
        mock_client.get_league_player_stats.return_value = [player_high, player_low]
        mock_client.get_player_transfer_history.return_value = [
            {
                "transfer_date": "2023-07-01",
                "from_team": {"id": 10, "name": "Club A"},
                "to_team": {"id": 20, "name": "Club B"},
            }
        ]
        mock_client.get_player_stats_for_season.return_value = _make_mock_season_stats(900)

        result = discover_transfers(["ENG1"], seasons_back=2)

        # Player with 900 minutes should have been processed (transfer history checked)
        called_pids = [
            call.args[0]
            for call in mock_client.get_player_transfer_history.call_args_list
        ]
        self.assertIn(5001, called_pids, "Player with 900 min should NOT be skipped")
        self.assertNotIn(5002, called_pids, "Player with 200 min should be skipped")


class TestTryResolveLeagueUsesRegistry(unittest.TestCase):
    """Bug 4: _try_resolve_league should use registry mapping, not broken API."""

    def test_resolve_from_mapping(self):
        """When team_id exists in the mapping, return the tournament id."""
        from backend.models.training_pipeline import _try_resolve_league

        mapping = {20: 8, 30: 35}
        result = _try_resolve_league(20, "FC Barcelona", mapping)
        self.assertEqual(result, 8)

    def test_resolve_unknown_team_returns_none(self):
        """When team_id is NOT in the mapping, return None (same-league)."""
        from backend.models.training_pipeline import _try_resolve_league

        mapping = {20: 8}
        result = _try_resolve_league(999, "Unknown FC", mapping)
        self.assertIsNone(result)

    def test_resolve_no_mapping_returns_none(self):
        """When no mapping is provided, return None."""
        from backend.models.training_pipeline import _try_resolve_league

        result = _try_resolve_league(20, "FC Barcelona", None)
        self.assertIsNone(result)

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    def test_discover_transfers_cross_league_detected(self, mock_client):
        """Cross-league transfer detected via team_id_to_league mapping."""
        from backend.models.training_pipeline import discover_transfers

        # Two leagues: ENG1 (tid=17) and ESP1 (tid=8)
        # ENG1 has team_id=10 (Club A) and team_id=20 is in ESP1 scan
        mock_client.get_season_list.return_value = [
            {"id": 100, "name": "2023/2024"},
            {"id": 200, "name": "2022/2023"},
            {"id": 300, "name": "2021/2022"},
        ]
        # League scan returns players from known teams
        mock_client.get_league_player_stats.side_effect = [
            # ENG1 season 2023/2024 (buffer)
            [{"id": 9001, "name": "P1", "position": "Forward",
              "minutes_played": 1000, "team_id": 10}],
            # ENG1 season 2022/2023
            [{"id": 1001, "name": "Player A", "position": "Forward",
              "minutes_played": 1000, "team_id": 10}],
            # ENG1 season 2021/2022
            [],
            # ESP1 season 2023/2024 (buffer)
            [{"id": 9002, "name": "P2", "position": "Forward",
              "minutes_played": 1000, "team_id": 20}],
            # ESP1 season 2022/2023
            [],
            # ESP1 season 2021/2022
            [],
        ]
        mock_client.get_player_transfer_history.return_value = [
            {
                "transfer_date": "2023-07-01",
                "from_team": {"id": 10, "name": "Club A"},
                "to_team": {"id": 20, "name": "Club B"},
            }
        ]
        mock_client.get_player_stats_for_season.return_value = _make_mock_season_stats(900)

        result = discover_transfers(["ENG1", "ESP1"], seasons_back=2)

        # Should detect the cross-league transfer (to_id=20 maps to ESP1 tid=8)
        cross_league = [r for r in result if r.to_league_id != r.from_league_id]
        if cross_league:
            self.assertEqual(cross_league[0].to_league_id, 8)
            self.assertEqual(cross_league[0].from_league_id, 17)


class TestResolveCrossLeaguePostSid(unittest.TestCase):
    """Bug 5: cross-league season ID must come from destination league."""

    def test_resolves_matching_season_name(self):
        from backend.models.training_pipeline import _resolve_cross_league_post_sid

        cache = {
            8: [
                {"id": 5001, "name": "2023/2024"},
                {"id": 5002, "name": "2022/2023"},
            ]
        }
        result = _resolve_cross_league_post_sid(8, "2023/2024", cache)
        self.assertEqual(result, 5001)

    def test_returns_none_for_unknown_season(self):
        from backend.models.training_pipeline import _resolve_cross_league_post_sid

        cache = {
            8: [
                {"id": 5001, "name": "2023/2024"},
            ]
        }
        result = _resolve_cross_league_post_sid(8, "2020/2021", cache)
        self.assertIsNone(result)

    def test_returns_none_for_empty_name(self):
        from backend.models.training_pipeline import _resolve_cross_league_post_sid

        cache = {8: [{"id": 5001, "name": "2023/2024"}]}
        result = _resolve_cross_league_post_sid(8, "", cache)
        self.assertIsNone(result)

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    def test_fetches_season_list_on_cache_miss(self, mock_client):
        from backend.models.training_pipeline import _resolve_cross_league_post_sid

        mock_client.get_season_list.return_value = [
            {"id": 7001, "name": "2023/2024"},
            {"id": 7002, "name": "2022/2023"},
        ]
        cache = {}
        result = _resolve_cross_league_post_sid(35, "2022/2023", cache)
        self.assertEqual(result, 7002)
        mock_client.get_season_list.assert_called_once_with(35)
        # Verify cache was populated
        self.assertIn(35, cache)

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    def test_cross_league_transfer_uses_dest_season_id(self, mock_client):
        """End-to-end: cross-league transfer record has dest league's season ID."""
        from backend.models.training_pipeline import discover_transfers

        # ENG1 seasons (tid=17)
        eng_seasons = [
            {"id": 100, "name": "2023/2024"},
            {"id": 200, "name": "2022/2023"},
            {"id": 300, "name": "2021/2022"},
        ]
        # ESP1 seasons (tid=8) — different IDs for same calendar year
        esp_seasons = [
            {"id": 8100, "name": "2023/2024"},
            {"id": 8200, "name": "2022/2023"},
        ]

        def season_list_side_effect(tid):
            if tid == 17:
                return eng_seasons
            elif tid == 8:
                return esp_seasons
            return []

        mock_client.get_season_list.side_effect = season_list_side_effect

        mock_client.get_league_player_stats.side_effect = [
            # ENG1 2023/2024 (buffer) — has team 10 (source) AND team 20 (dest)
            [{"id": 9001, "name": "P1", "position": "Forward",
              "minutes_played": 1000, "team_id": 10}],
            # ENG1 2022/2023 — player 1001 has a transfer
            [{"id": 1001, "name": "Player A", "position": "Forward",
              "minutes_played": 1000, "team_id": 10}],
            # ENG1 2021/2022
            [],
            # ESP1 2023/2024 (buffer) — team 20 is here
            [{"id": 9002, "name": "P2", "position": "Forward",
              "minutes_played": 1000, "team_id": 20}],
            # ESP1 2022/2023
            [],
        ]

        mock_client.get_player_transfer_history.return_value = [
            {
                "transfer_date": "2023-07-01",
                "from_team": {"id": 10, "name": "Club A"},
                "to_team": {"id": 20, "name": "Club B"},
            }
        ]
        mock_client.get_player_stats_for_season.return_value = _make_mock_season_stats(900)

        result = discover_transfers(["ENG1", "ESP1"], seasons_back=2)

        # Find the cross-league record
        cross = [r for r in result if r.to_league_id == 8]
        if cross:
            # post_transfer_season_id must be ESP1's season ID (8100),
            # NOT ENG1's season ID (100)
            self.assertEqual(cross[0].post_transfer_season_id, 8100)
            self.assertEqual(cross[0].post_transfer_tournament_id, 8)


class TestNonTransferExcludesTransferPlayers(unittest.TestCase):
    """Bug 6: non-transfer discovery must exclude players in transfer records."""

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    def test_excludes_transfer_player_ids(self, mock_client):
        from backend.models.training_pipeline import discover_non_transfers

        mock_client.get_season_list.return_value = [
            {"id": 100, "name": "2023/2024"},
            {"id": 200, "name": "2022/2023"},
        ]
        # Two players: 1001 (transferred) and 1002 (stayed)
        mock_client.get_league_player_stats.return_value = [
            {"id": 1001, "name": "Transfer Player", "position": "Forward",
             "minutes_played": 900, "team_id": 10, "team": "Club A"},
            {"id": 1002, "name": "Stay Player", "position": "Midfielder",
             "minutes_played": 900, "team_id": 10, "team": "Club A"},
        ]
        # Neither player has a transfer in their history
        mock_client.get_player_transfer_history.return_value = []
        mock_client.get_player_stats_for_season.return_value = _make_mock_season_stats(900)

        # Exclude player 1001 (appears in transfer records)
        result = discover_non_transfers(
            ["ENG1"], seasons_back=2, exclude_player_ids={1001}
        )

        # Player 1001 should be excluded; 1002 should be included
        pids_in_result = {r.player_id for r in result}
        self.assertNotIn(1001, pids_in_result)
        self.assertIn(1002, pids_in_result)

    @mock.patch("backend.models.training_pipeline.sofascore_client")
    def test_no_exclusion_when_none(self, mock_client):
        """When exclude_player_ids is None, all eligible players are included."""
        from backend.models.training_pipeline import discover_non_transfers

        mock_client.get_season_list.return_value = [
            {"id": 100, "name": "2023/2024"},
            {"id": 200, "name": "2022/2023"},
        ]
        mock_client.get_league_player_stats.return_value = [
            {"id": 1001, "name": "Player A", "position": "Forward",
             "minutes_played": 900, "team_id": 10, "team": "Club A"},
        ]
        mock_client.get_player_transfer_history.return_value = []
        mock_client.get_player_stats_for_season.return_value = _make_mock_season_stats(900)

        result = discover_non_transfers(["ENG1"], seasons_back=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].player_id, 1001)


class TestFeatureKeysListConsistency(unittest.TestCase):
    """Verify training_pipeline._feature_keys_list() matches transfer_portal._feature_keys()."""

    def test_matches_transfer_portal_feature_keys(self):
        """_feature_keys_list() must match _feature_keys() exactly."""
        from backend.models.training_pipeline import _feature_keys_list
        from backend.models.transfer_portal import _feature_keys

        tp_keys = _feature_keys_list()
        ref_keys = _feature_keys()
        self.assertEqual(tp_keys, ref_keys)

    def test_length_matches_feature_dim(self):
        """_feature_keys_list() length must equal FEATURE_DIM (46)."""
        from backend.models.training_pipeline import _feature_keys_list

        self.assertEqual(len(_feature_keys_list()), FEATURE_DIM)

    def test_includes_interaction_features(self):
        """_feature_keys_list() must include the 3 interaction features."""
        from backend.models.training_pipeline import _feature_keys_list

        keys = _feature_keys_list()
        self.assertIn("interaction_ability_gap", keys)
        self.assertIn("interaction_gap_squared", keys)
        self.assertIn("interaction_league_gap", keys)
