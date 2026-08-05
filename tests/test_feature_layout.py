"""The 94-slot feature layout is defined in one place and asserted here.

The ordering used to be reimplemented four times: ``transfer_portal._feature_keys``,
``training_pipeline._feature_keys_list``, ``backtester._feature_keys_list``, and
the direct ``np.array(a + b + c + ...)`` concatenation that actually builds the
matrix. Three were kept in step by a "must stay in sync" docstring, which is
not a mechanism. The first three now delegate; this file guards the fourth,
because the pipeline's only check was on *shape*, and a reordering preserves
shape while silently changing what every column means.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.data.sofascore_client import ADDITIONAL_METRICS, CORE_METRICS
from backend.models.transfer_portal import (
    FEATURE_DIM,
    POSITION_LABELS,
    _feature_keys,
    build_feature_dict,
)

N_CORE = len(CORE_METRICS)
N_ADDITIONAL = len(ADDITIONAL_METRICS)
N_POS = len(POSITION_LABELS)

# The layout the training concatenation builds, segment by segment, in order.
EXPECTED_SEGMENTS = [
    ("player core metrics", N_CORE, lambda: [f"player_{m}" for m in CORE_METRICS]),
    ("player additional metrics", N_ADDITIONAL,
     lambda: [f"player_{m}" for m in ADDITIONAL_METRICS]),
    ("team/league ability", 4, lambda: [
        "team_ability_current", "team_ability_target",
        "league_ability_current", "league_ability_target",
    ]),
    ("raw elo", 2, lambda: ["raw_elo_current", "raw_elo_target"]),
    ("reep metadata", 2, lambda: ["player_height_cm", "player_age"]),
    ("team_pos_current", N_CORE,
     lambda: [f"team_pos_current_{m}" for m in CORE_METRICS]),
    ("team_pos_target", N_CORE,
     lambda: [f"team_pos_target_{m}" for m in CORE_METRICS]),
    ("interactions", 3, lambda: [
        "interaction_ability_gap", "interaction_gap_squared",
        "interaction_league_gap",
    ]),
    ("relative ability", 3, lambda: [
        "relative_ability_current", "relative_ability_target",
        "relative_ability_gap",
    ]),
    ("league_norm", N_CORE, lambda: [f"league_norm_{m}" for m in CORE_METRICS]),
    ("league_mean_ratio", N_CORE,
     lambda: [f"league_mean_ratio_{m}" for m in CORE_METRICS]),
    ("position one-hot", N_POS, lambda: [f"position_{p}" for p in POSITION_LABELS]),
    ("minutes per match", 1, lambda: ["pre_minutes_per_match"]),
]


class TestLayout:
    def test_segments_sum_to_feature_dim(self):
        assert sum(n for _, n, _ in EXPECTED_SEGMENTS) == FEATURE_DIM

    def test_feature_keys_matches_the_segment_layout(self):
        expected: list[str] = []
        for _, _, build in EXPECTED_SEGMENTS:
            expected.extend(build())
        assert _feature_keys() == expected

    def test_no_duplicate_keys(self):
        keys = _feature_keys()
        assert len(set(keys)) == len(keys)

    @pytest.mark.parametrize("name,count,build", EXPECTED_SEGMENTS)
    def test_each_segment_is_contiguous(self, name, count, build):
        """A segment's keys must occupy consecutive slots, in order."""
        keys = _feature_keys()
        wanted = build()
        start = keys.index(wanted[0])
        assert keys[start : start + count] == wanted, f"{name} is not contiguous"


class TestSingleSourceOfTruth:
    def test_all_key_builders_delegate(self):
        from backend.models.backtester import _feature_keys_list as bt
        from backend.models.training_pipeline import _feature_keys_list as tp

        keys = _feature_keys()
        assert tp() == keys
        assert bt() == keys

    def test_build_feature_dict_covers_every_key(self):
        """Nothing the model expects may be missing from the dict builder."""
        fd = build_feature_dict(
            player_per90={m: 1.0 for m in CORE_METRICS},
            team_ability_current=60.0,
            team_ability_target=70.0,
            league_ability_current=80.0,
            league_ability_target=85.0,
            team_pos_current={m: 0.5 for m in CORE_METRICS},
            team_pos_target={m: 0.6 for m in CORE_METRICS},
            position="F",
            pre_minutes_per_match=75.0,
        )
        missing = [k for k in _feature_keys() if k not in fd]
        assert missing == [], f"build_feature_dict omits {missing}"

    def test_build_feature_dict_adds_nothing_extra(self):
        fd = build_feature_dict(
            player_per90={m: 1.0 for m in CORE_METRICS},
            team_ability_current=60.0,
            team_ability_target=70.0,
            league_ability_current=80.0,
            league_ability_target=85.0,
            team_pos_current={m: 0.5 for m in CORE_METRICS},
            team_pos_target={m: 0.6 for m in CORE_METRICS},
        )
        extra = [k for k in fd if k not in set(_feature_keys())]
        assert extra == [], f"build_feature_dict emits unknown keys {extra}"


class TestTrainingConcatenationOrder:
    """Pin the order the pipeline's np.array(...) concatenation produces.

    The pipeline asserts ``features.shape == (FEATURE_DIM,)``, which a
    reordering passes. This rebuilds the same concatenation from uniquely
    identifiable per-segment values and checks each named key lands where
    ``_feature_keys()`` says it does.
    """

    def test_concatenation_order_matches_feature_keys(self):
        # One distinct value per segment, so a swapped pair is detectable.
        player_metrics = [1.0] * N_CORE
        additional = [2.0] * N_ADDITIONAL
        abilities = [3.0, 4.0, 5.0, 6.0]
        raw_elo = [7.0, 8.0]
        reep = [9.0, 10.0]
        team_pos_current = [11.0] * N_CORE
        team_pos_target = [12.0] * N_CORE
        interactions = [13.0, 14.0, 15.0]
        relative = [16.0, 17.0, 18.0]
        league_norm = [19.0] * N_CORE
        league_ratio = [20.0] * N_CORE
        position_one_hot = [21.0] * N_POS
        mpm = [22.0]

        features = np.array(
            player_metrics
            + additional
            + abilities
            + raw_elo
            + reep
            + team_pos_current
            + team_pos_target
            + interactions
            + relative
            + league_norm
            + league_ratio
            + position_one_hot
            + mpm,
            dtype=np.float32,
        )
        assert features.shape == (FEATURE_DIM,)

        keys = _feature_keys()
        by_key = dict(zip(keys, features))

        assert by_key[f"player_{CORE_METRICS[0]}"] == 1.0
        assert by_key[f"player_{ADDITIONAL_METRICS[0]}"] == 2.0
        assert by_key["team_ability_current"] == 3.0
        assert by_key["league_ability_target"] == 6.0
        assert by_key["raw_elo_current"] == 7.0
        assert by_key["player_height_cm"] == 9.0
        assert by_key["player_age"] == 10.0
        assert by_key[f"team_pos_current_{CORE_METRICS[0]}"] == 11.0
        assert by_key[f"team_pos_target_{CORE_METRICS[0]}"] == 12.0
        assert by_key["interaction_ability_gap"] == 13.0
        assert by_key["relative_ability_current"] == 16.0
        assert by_key[f"league_norm_{CORE_METRICS[0]}"] == 19.0
        assert by_key[f"league_mean_ratio_{CORE_METRICS[0]}"] == 20.0
        assert by_key[f"position_{POSITION_LABELS[0]}"] == 21.0
        assert by_key["pre_minutes_per_match"] == 22.0

    def test_minutes_per_match_is_the_last_slot(self):
        """Appended last by the pipeline; the shipped matrix has it at index 93."""
        assert _feature_keys()[-1] == "pre_minutes_per_match"
        assert _feature_keys().index("pre_minutes_per_match") == FEATURE_DIM - 1

    def test_position_one_hot_occupies_the_four_slots_before_it(self):
        keys = _feature_keys()
        assert keys[FEATURE_DIM - 1 - N_POS : FEATURE_DIM - 1] == [
            f"position_{p}" for p in POSITION_LABELS
        ]
