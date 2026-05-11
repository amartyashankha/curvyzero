import numpy as np
import pytest

from curvyzero.env.multiplayer_ego_wrapper import (
    MULTIPLAYER_EGO_ACTION_SIDECAR_SCHEMA_ID,
    MULTIPLAYER_EGO_ACTION_MAP_POLICY_ID,
    MULTIPLAYER_EGO_WRAPPER_ID,
    MetadataOnlyMultiplayerEgoWrapper,
    build_multiplayer_ego_action_map,
    build_multiplayer_ego_policy_rows,
)
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.training.multiplayer_opponent_policy import (
    FIXED_ACTION_OPPONENT_POLICY_ID,
    NO_OPPONENT_ACTION,
    RANDOM_LEGAL_OPPONENT_POLICY_ID,
    FixedActionOpponentPolicy,
    SeededRandomOpponentPolicy,
)
from curvyzero.training.multiplayer_replay_contract import (
    MULTIPLAYER_REPLAY_OPPONENT_POLICY_SIDECAR_SCHEMA_ID,
)


def test_ego_policy_rows_include_only_configured_live_ego_seats():
    observation = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    legal_action_mask = np.zeros((2, 3, 3), dtype=bool)
    legal_action_mask[0, 0] = True
    legal_action_mask[0, 1] = True
    legal_action_mask[1, 2] = True

    rows = build_multiplayer_ego_policy_rows(
        observation,
        legal_action_mask,
        ego_player_id=np.asarray([0, 1], dtype=np.int16),
        observation_schema_id="curvyzero_debug_metadata_only/v0",
        observation_schema_hash="debug-hash",
        pad_to=3,
    )

    assert rows.metadata_only is True
    assert rows.learned_observation_claim is False
    assert rows.joint_action_mcts_claim is False
    assert rows.observation_schema_id == "curvyzero_debug_metadata_only/v0"
    assert rows.mapping.active_count == 1
    assert rows.mapping.capacity == 3
    np.testing.assert_array_equal(rows.mapping.env_row_id, np.asarray([0, -1, -1]))
    np.testing.assert_array_equal(rows.mapping.player_id, np.asarray([0, -1, -1]))
    np.testing.assert_array_equal(rows.mapping.observations[0], observation[0, 0])


def test_fixed_opponent_policy_fills_full_action_map_and_sidecars():
    observation = np.zeros((2, 3, 6), dtype=np.float32)
    legal_action_mask = np.zeros((2, 3, 3), dtype=bool)
    legal_action_mask[0, 0] = True
    legal_action_mask[0, 1] = True
    legal_action_mask[0, 2] = True
    legal_action_mask[1, 1] = True
    legal_action_mask[1, 2] = True
    present = np.asarray([[True, True, True], [False, True, True]], dtype=bool)
    alive = np.asarray([[True, True, True], [False, True, True]], dtype=bool)

    rows = build_multiplayer_ego_policy_rows(
        observation,
        legal_action_mask,
        ego_player_id=np.asarray([0, 1], dtype=np.int16),
    )
    action_map = build_multiplayer_ego_action_map(
        rows,
        np.asarray([2, 0], dtype=np.int16),
        legal_action_mask,
        opponent_policy=FixedActionOpponentPolicy(action_id=1),
        decision_index=7,
        present=present,
        alive=alive,
    )

    np.testing.assert_array_equal(
        action_map.joint_action,
        np.asarray([[2, 1, 1], [1, 0, 1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        action_map.opponent_selection.actions,
        np.asarray(
            [
                [NO_OPPONENT_ACTION, 1, 1],
                [NO_OPPONENT_ACTION, NO_OPPONENT_ACTION, 1],
            ],
            dtype=np.int16,
        ),
    )
    assert action_map.action_sidecar["schema_id"] == MULTIPLAYER_EGO_ACTION_SIDECAR_SCHEMA_ID
    assert action_map.action_sidecar["wrapper_id"] == MULTIPLAYER_EGO_WRAPPER_ID
    assert action_map.action_sidecar["action_map_policy_id"] == (
        MULTIPLAYER_EGO_ACTION_MAP_POLICY_ID
    )
    assert action_map.action_sidecar["joint_action_semantics"] == (
        "full_simultaneous_env_action_map_not_mcts/v0"
    )
    assert action_map.action_sidecar["branching_policy"] == (
        "ego_rows_only_opponents_are_sidecar_fills/v0"
    )
    assert action_map.action_sidecar["joint_action_mcts_claim"] is False
    assert action_map.action_sidecar["opponent_policy_id"] == FIXED_ACTION_OPPONENT_POLICY_ID
    np.testing.assert_array_equal(
        action_map.action_sidecar["action_source"],
        np.asarray(
            [
                ["ego_policy", "opponent_policy", "opponent_policy"],
                ["absent_noop", "ego_policy", "opponent_policy"],
            ],
            dtype=object,
        ),
    )

    opponent_sidecar = action_map.opponent_policy_sidecar
    assert opponent_sidecar["schema_id"] == MULTIPLAYER_REPLAY_OPPONENT_POLICY_SIDECAR_SCHEMA_ID
    assert opponent_sidecar["policy_id"] == FIXED_ACTION_OPPONENT_POLICY_ID
    assert opponent_sidecar["policy_version"]
    np.testing.assert_array_equal(opponent_sidecar["seed"], np.asarray([0, 0]))
    np.testing.assert_array_equal(
        opponent_sidecar["actions"],
        action_map.opponent_selection.actions,
    )


def test_seeded_random_opponent_policy_is_deterministic_and_records_slot_seeds():
    observation = np.zeros((1, 4, 6), dtype=np.float32)
    legal_action_mask = np.ones((1, 4, 3), dtype=bool)
    legal_action_mask[0, 2] = np.asarray([False, True, False], dtype=bool)
    rows = build_multiplayer_ego_policy_rows(
        observation,
        legal_action_mask,
        ego_player_id=0,
    )
    policy = SeededRandomOpponentPolicy(seed=1234)

    action_map_a = build_multiplayer_ego_action_map(
        rows,
        np.asarray([2], dtype=np.int16),
        legal_action_mask,
        opponent_policy=policy,
        decision_index=3,
    )
    action_map_b = build_multiplayer_ego_action_map(
        rows,
        np.asarray([2], dtype=np.int16),
        legal_action_mask,
        opponent_policy=policy,
        decision_index=3,
    )
    action_map_c = build_multiplayer_ego_action_map(
        rows,
        np.asarray([2], dtype=np.int16),
        legal_action_mask,
        opponent_policy=policy,
        decision_index=4,
    )

    assert action_map_a.opponent_policy_sidecar["policy_id"] == RANDOM_LEGAL_OPPONENT_POLICY_ID
    np.testing.assert_array_equal(action_map_a.joint_action, action_map_b.joint_action)
    np.testing.assert_array_equal(
        action_map_a.opponent_policy_sidecar["action_seed"],
        action_map_b.opponent_policy_sidecar["action_seed"],
    )
    assert not np.array_equal(
        action_map_a.opponent_policy_sidecar["action_seed"],
        action_map_c.opponent_policy_sidecar["action_seed"],
    )
    assert int(action_map_a.joint_action[0, 2]) == 1


def test_ego_action_map_rejects_illegal_ego_action():
    observation = np.zeros((1, 2, 6), dtype=np.float32)
    legal_action_mask = np.ones((1, 2, 3), dtype=bool)
    legal_action_mask[0, 0] = np.asarray([True, False, True], dtype=bool)
    rows = build_multiplayer_ego_policy_rows(
        observation,
        legal_action_mask,
        ego_player_id=0,
    )

    with pytest.raises(ValueError, match="illegal active-row actions"):
        build_multiplayer_ego_action_map(
            rows,
            np.asarray([1], dtype=np.int16),
            legal_action_mask,
            opponent_policy=FixedActionOpponentPolicy(),
        )


def test_metadata_only_wrapper_steps_public_env_with_full_joint_action_sidecar():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=3,
        seed=42,
        decision_ms=50.0,
        body_capacity=64,
        event_capacity=16,
    )
    wrapper = MetadataOnlyMultiplayerEgoWrapper(
        env,
        ego_player_id=0,
        opponent_policy=FixedActionOpponentPolicy(action_id=1),
    )
    reset_batch = wrapper.reset(seed=42)
    rows = wrapper.observe()

    assert reset_batch.info["metadata_only"] is True
    assert rows.mapping.active_count == 1
    np.testing.assert_array_equal(rows.mapping.player_id, np.asarray([0], dtype=np.int16))

    step_batch = wrapper.step(np.asarray([0], dtype=np.int16))

    assert step_batch.info["metadata_only"] is True
    assert step_batch.info["trainer_observation_claim"] is False
    assert step_batch.info["trainer_replay_claim"] is False
    assert step_batch.info["learned_observation_claim"] is False
    assert step_batch.info["joint_action_mcts_claim"] is False
    assert step_batch.info["multiplayer_ego_wrapper_id"] == MULTIPLAYER_EGO_WRAPPER_ID
    np.testing.assert_array_equal(
        step_batch.info["wrapper_joint_action"],
        np.asarray([[0, 1, 1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        step_batch.info["joint_action"],
        step_batch.info["wrapper_joint_action"],
    )
    assert step_batch.info["opponent_policy_sidecar"]["policy_id"] == (
        FIXED_ACTION_OPPONENT_POLICY_ID
    )
    np.testing.assert_array_equal(
        step_batch.info["opponent_policy_sidecar"]["actions"],
        np.asarray([[NO_OPPONENT_ACTION, 1, 1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        step_batch.info["multiplayer_ego_action_sidecar"]["action_source"],
        np.asarray([["ego_policy", "opponent_policy", "opponent_policy"]], dtype=object),
    )
