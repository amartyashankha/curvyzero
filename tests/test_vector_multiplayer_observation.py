import numpy as np

from curvyzero.env import trainer_contract
from curvyzero.env import vector_multiplayer_observation as obs
from curvyzero.env.vector_multiplayer_env import DEBUG_METADATA_OBSERVATION_SCHEMA_ID
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv


def test_4p_public_state_packs_present_alive_ego_rows_with_masks_and_ids():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        seed=123,
        timer_capacity=4,
        random_tape_capacity=32,
    )
    env.reset(
        seed=np.asarray([7], dtype=np.uint64),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )

    packed = obs.pack_vector_multiplayer_observation_rows_v0(
        env.state,
        max_ticks=env.max_ticks,
    )

    assert packed.schema_id == obs.MULTIPLAYER_OBSERVATION_SCHEMA_ID
    assert packed.schema_hash == obs.MULTIPLAYER_OBSERVATION_SCHEMA_HASH
    assert packed.source_shape == (1, 4)
    assert packed.active_count == 4
    assert packed.capacity == 4
    assert packed.observation.shape == (4, *obs.MULTIPLAYER_OBSERVATION_SHAPE)
    assert packed.observation.dtype == np.float32
    assert np.isfinite(packed.observation).all()
    assert packed.action_mask.dtype == np.bool_
    assert packed.lightzero_action_mask.dtype == np.int8
    np.testing.assert_array_equal(packed.action_mask, np.ones((4, 3), dtype=bool))
    np.testing.assert_array_equal(packed.lightzero_action_mask, np.ones((4, 3), dtype=np.int8))
    np.testing.assert_array_equal(packed.to_play, np.full(4, -1, dtype=np.int64))
    np.testing.assert_array_equal(packed.env_row_id, np.zeros(4, dtype=np.int32))
    np.testing.assert_array_equal(packed.ego_player_id, np.arange(4, dtype=np.int16))
    np.testing.assert_array_equal(packed.row_mask, np.ones(4, dtype=bool))


def test_3p_absent_dead_and_terminal_slots_are_padded_without_legal_actions():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=3,
        seed=123,
        timer_capacity=3,
        random_tape_capacity=32,
    )
    env.reset(
        seed=np.asarray([11], dtype=np.uint64),
        present=np.asarray([[True, False, True]], dtype=bool),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    env.state["alive"][0, 2] = False
    env.state["done"][0] = True

    packed = obs.pack_vector_multiplayer_observation_rows_v0(
        env.state,
        max_ticks=env.max_ticks,
        pad_to=3,
    )

    assert packed.source_shape == (1, 3)
    assert packed.active_count == 1
    assert packed.capacity == 3
    np.testing.assert_array_equal(packed.row_mask, np.asarray([True, False, False]))
    np.testing.assert_array_equal(packed.env_row_id, np.asarray([0, -1, -1], dtype=np.int32))
    np.testing.assert_array_equal(packed.ego_player_id, np.asarray([0, -1, -1], dtype=np.int16))
    np.testing.assert_array_equal(packed.action_mask, np.zeros((3, 3), dtype=bool))
    np.testing.assert_array_equal(packed.lightzero_action_mask, np.zeros((3, 3), dtype=np.int8))
    np.testing.assert_array_equal(
        packed.observation[1:],
        np.zeros((2, *obs.MULTIPLAYER_OBSERVATION_SHAPE), dtype=np.float32),
    )
    assert packed.observation[0, obs.FEATURE_INDEX["opponent_0_slot_present_alive"]] == 0.0


def test_multiplayer_observation_schema_hash_and_non_claims_are_pinned():
    assert obs.MULTIPLAYER_OBSERVATION_SCHEMA_ID != trainer_contract.OBSERVATION_SCHEMA_ID
    assert obs.MULTIPLAYER_OBSERVATION_SCHEMA_ID != DEBUG_METADATA_OBSERVATION_SCHEMA_ID
    assert obs.MULTIPLAYER_OBSERVATION_SCHEMA_HASH == "0414896a73d123f9"
    assert (
        trainer_contract.stable_contract_hash(obs.MULTIPLAYER_OBSERVATION_SCHEMA)
        == obs.MULTIPLAYER_OBSERVATION_SCHEMA_HASH
    )

    schema = obs.MULTIPLAYER_OBSERVATION_SCHEMA
    assert schema["source"] == "VectorMultiplayerEnv.state arrays"
    assert schema["separate_env_implementation"] is False
    assert schema["bonus_policy"] == "no_bonus_state_projection_only"
    assert schema["claims"]["learned_observation_schema"] is True
    assert schema["claims"]["trainer_ready_env_claim"] is False
    assert schema["claims"]["replay_writer_claim"] is False
    assert schema["claims"]["visual_or_pixel_claim"] is False
    assert schema["claims"]["trail_ray_claim"] is False
    assert "score" in schema["hidden_state_policy"]["excluded_from_float_observation"]
