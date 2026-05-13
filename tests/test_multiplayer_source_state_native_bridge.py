from dataclasses import replace

import numpy as np
import pytest

from curvyzero.training.multiplayer_source_state_native_bridge import (
    SOURCE_STATE_NATIVE_GAME_SEGMENTS_CONTRACT_ID,
)
from curvyzero.training.multiplayer_source_state_native_bridge import (
    build_source_state_native_game_segment_specs_v0,
)
from curvyzero.training.multiplayer_source_state_native_bridge import (
    build_source_state_native_game_segments_v0,
)
from curvyzero.training.multiplayer_source_state_target_rows import DEFAULT_TO_PLAY
from curvyzero.training.multiplayer_source_state_target_rows import (
    PROJECT_HELPER_RESTRICTED_SOURCE_FIDELITY_CLAIM,
)
from curvyzero.training.multiplayer_source_state_target_rows import (
    SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID,
)
from curvyzero.training.multiplayer_source_state_target_rows import (
    SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_HASH,
)
from curvyzero.training.multiplayer_source_state_target_rows import (
    SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_ID,
)
from curvyzero.training.multiplayer_source_state_target_rows import (
    SourceStateMultiplayerTargetRowsV0,
)
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError


def test_native_bridge_maps_rows_to_injected_game_segment_calls():
    rows = _target_rows(player_count=2)
    config = {"game_segment_length": 2}

    native = build_source_state_native_game_segments_v0(
        rows,
        game_segment_cls=FakeGameSegment,
        config=config,
    )

    assert native.metadata["bridge_contract_id"] == (
        SOURCE_STATE_NATIVE_GAME_SEGMENTS_CONTRACT_ID
    )
    assert native.metadata["source_target_contract_id"] == (
        SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID
    )
    assert native.metadata["segment_count"] == 2
    assert native.config is config

    segment_0 = native.game_segments[0]
    assert segment_0.config is config
    assert segment_0.calls[0][0] == "reset"
    np.testing.assert_array_equal(segment_0.calls[0][1], _obs(0, 0))

    assert segment_0.calls[1][0] == "store_search_stats"
    np.testing.assert_allclose(segment_0.calls[1][1], [1.0, 0.0, 0.0])
    assert segment_0.calls[1][2] == pytest.approx(0.5)
    assert segment_0.calls[2][0] == "append"
    assert segment_0.calls[2][1] == 0
    np.testing.assert_array_equal(segment_0.calls[2][2], _obs(0, 1))
    assert segment_0.calls[2][3] == pytest.approx(1.0)
    np.testing.assert_array_equal(segment_0.calls[2][4], [True, True, False])
    assert segment_0.calls[2][5] == DEFAULT_TO_PLAY
    assert segment_0.calls[2][6] == 0

    assert segment_0.calls[3][0] == "store_search_stats"
    np.testing.assert_allclose(segment_0.calls[3][1], [0.0, 1.0, 0.0])
    assert segment_0.calls[3][2] == pytest.approx(1.5)
    assert segment_0.calls[4][0] == "append"
    assert segment_0.calls[4][1] == 1
    np.testing.assert_array_equal(segment_0.calls[4][2], _obs(0, 3))
    assert segment_0.calls[4][3] == pytest.approx(3.0)
    np.testing.assert_array_equal(segment_0.calls[4][4], [True, True, True])
    assert segment_0.calls[4][5] == DEFAULT_TO_PLAY
    assert segment_0.calls[4][6] == 2

    spec_0 = native.specs[0]
    assert spec_0.env_row == 0
    assert spec_0.player == 0
    assert spec_0.row_id == (1, 0)
    assert spec_0.actions == (0, 1)
    assert spec_0.rewards == (1.0, 3.0)
    assert spec_0.root_values == (0.5, 1.5)
    assert spec_0.record_indices == (0, 2)
    assert spec_0.to_play == (DEFAULT_TO_PLAY, DEFAULT_TO_PLAY)
    assert spec_0.terminal is True
    np.testing.assert_array_equal(spec_0.policy_target[0], spec_0.visit_distributions[0])


def test_native_bridge_groups_by_player_and_sorts_record_index_for_p2_and_p4():
    p2_specs = build_source_state_native_game_segment_specs_v0(_target_rows(player_count=2))
    p4_specs = build_source_state_native_game_segment_specs_v0(_target_rows(player_count=4))

    assert [(spec.env_row, spec.player) for spec in p2_specs] == [(0, 0), (0, 1)]
    assert [spec.record_indices for spec in p2_specs] == [(0, 2), (0, 2)]
    assert [spec.row_id for spec in p2_specs] == [(1, 0), (3, 2)]

    assert [(spec.env_row, spec.player) for spec in p4_specs] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
    ]
    assert [spec.record_indices for spec in p4_specs] == [
        (0, 2),
        (0, 2),
        (0, 2),
        (0, 2),
    ]


def test_native_bridge_preserves_helper_metadata_and_keeps_claims_false():
    rows = _target_rows(
        player_count=2,
        metadata={
            "death_mode": "profile_no_death",
            "death_suppression_for_profile": True,
            "death_suppression_claim": "profile_only_not_source_fidelity",
            "original_curvytron_behavior_claim": False,
            "source_fidelity_claim": PROJECT_HELPER_RESTRICTED_SOURCE_FIDELITY_CLAIM,
            "project_training_helper_active": True,
            "project_training_helper_metadata": {
                "death_mode": "profile_no_death",
                "death_suppression_for_profile": True,
                "source_fidelity_claim": PROJECT_HELPER_RESTRICTED_SOURCE_FIDELITY_CLAIM,
            },
        },
    )

    native = build_source_state_native_game_segments_v0(
        rows,
        game_segment_cls=FakeGameSegment,
    )

    assert native.metadata["death_mode"] == "profile_no_death"
    assert native.metadata["death_suppression_for_profile"] is True
    assert native.metadata["death_suppression_claim"] == "profile_only_not_source_fidelity"
    assert native.metadata["original_curvytron_behavior_claim"] is False
    assert native.metadata["source_fidelity_claim"] == (
        PROJECT_HELPER_RESTRICTED_SOURCE_FIDELITY_CLAIM
    )
    assert native.metadata["project_training_helper_active"] is True
    assert native.metadata["project_training_helper_metadata"]["death_mode"] == (
        "profile_no_death"
    )
    assert native.metadata["native_game_segment_claim"] is False
    assert native.metadata["lightzero_native_game_segment_claim"] is False
    assert native.metadata["lightzero_training_integration_claim"] is False
    assert native.metadata["lightzero_training_claim"] is False
    assert native.metadata["muzero_game_buffer_claim"] is False
    assert native.metadata["learner_update_claim"] is False
    assert native.metadata["policy_improvement_claim"] is False


def test_native_bridge_rejects_target_rows_with_non_default_to_play():
    rows = _target_rows(player_count=2)
    bad_to_play = rows.to_play.copy()
    bad_to_play[0] = 0

    with pytest.raises(ReplayCompatibilityError, match="to_play == -1"):
        build_source_state_native_game_segment_specs_v0(
            replace(rows, to_play=bad_to_play),
        )


class FakeGameSegment:
    def __init__(self, *, config=None):
        self.config = config
        self.calls = []

    def reset(self, observation):
        self.calls.append(("reset", np.asarray(observation, dtype=np.float32).copy()))

    def store_search_stats(self, policy, *, root_value):
        self.calls.append(
            (
                "store_search_stats",
                np.asarray(policy, dtype=np.float32).copy(),
                float(root_value),
            )
        )

    def append(
        self,
        action,
        next_observation,
        reward,
        *,
        action_mask,
        to_play,
        timestep,
    ):
        self.calls.append(
            (
                "append",
                int(action),
                np.asarray(next_observation, dtype=np.float32).copy(),
                float(reward),
                np.asarray(action_mask, dtype=bool).copy(),
                int(to_play),
                int(timestep),
            )
        )


def _target_rows(*, player_count: int, metadata: dict | None = None):
    row_defs = []
    for player in range(player_count):
        row_defs.extend(
            [
                {
                    "record_index": 2,
                    "player": player,
                    "action": (player + 1) % 3,
                    "reward": float(player + 3),
                    "root_value": float(player) + 1.5,
                    "done": True,
                },
                {
                    "record_index": 0,
                    "player": player,
                    "action": player % 3,
                    "reward": float(player + 1),
                    "root_value": float(player) + 0.5,
                    "done": False,
                },
            ]
        )
    row_count = len(row_defs)
    target_metadata = {
        "target_contract_id": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID,
        "target_schema_id": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_ID,
        "target_schema_hash": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_HASH,
        "native_game_segment_claim": False,
        "lightzero_native_game_segment_claim": False,
        "lightzero_training_integration_claim": False,
        "lightzero_training_claim": False,
        "muzero_game_buffer_claim": False,
        "learner_update_claim": False,
        "policy_improvement_claim": False,
    }
    if metadata:
        target_metadata.update(metadata)

    return SourceStateMultiplayerTargetRowsV0(
        metadata=target_metadata,
        observation=np.stack(
            [_obs(row["player"], row["record_index"]) for row in row_defs],
            axis=0,
        ).astype(np.float32),
        action=np.asarray([row["action"] for row in row_defs], dtype=np.int16),
        action_mask=np.asarray(
            [_mask(row["action"]) for row in row_defs],
            dtype=bool,
        ),
        policy_target=np.asarray(
            [_policy(row["action"]) for row in row_defs],
            dtype=np.float32,
        ),
        root_value=np.asarray([row["root_value"] for row in row_defs], dtype=np.float32),
        reward=np.asarray([row["reward"] for row in row_defs], dtype=np.float32),
        final_reward=np.asarray([row["reward"] for row in row_defs], dtype=np.float32),
        done=np.asarray([row["done"] for row in row_defs], dtype=bool),
        terminated=np.asarray([row["done"] for row in row_defs], dtype=bool),
        truncated=np.zeros((row_count,), dtype=bool),
        next_observation=np.stack(
            [_obs(row["player"], row["record_index"] + 1) for row in row_defs],
            axis=0,
        ).astype(np.float32),
        to_play=np.full((row_count,), DEFAULT_TO_PLAY, dtype=np.int64),
        env_row=np.zeros((row_count,), dtype=np.int32),
        player=np.asarray([row["player"] for row in row_defs], dtype=np.int16),
        record_index=np.asarray(
            [row["record_index"] for row in row_defs],
            dtype=np.int32,
        ),
        next_record_index=np.asarray(
            [row["record_index"] + 1 for row in row_defs],
            dtype=np.int32,
        ),
        policy_row=np.arange(row_count, dtype=np.int32),
        policy_source=tuple("unit_test" for _ in row_defs),
        source_record_ref=tuple(f"row:{index}" for index in range(row_count)),
    )


def _obs(player: int, record_index: int) -> np.ndarray:
    return np.asarray([float(player), float(record_index)], dtype=np.float32)


def _mask(action: int) -> np.ndarray:
    mask = np.ones((3,), dtype=bool)
    if action == 0:
        mask[2] = False
    return mask


def _policy(action: int) -> np.ndarray:
    policy = np.zeros((3,), dtype=np.float32)
    policy[action] = 1.0
    return policy
