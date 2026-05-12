import json

import numpy as np

from curvyzero.env.vector_multiplayer_env import JOINT_ACTION_SCHEMA_ID
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SHAPE
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SURFACE
from curvyzero.env.vector_visual_observation import SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID
from curvyzero.env.vector_visual_observation import rgb_canvas_like_to_gray64
from curvyzero.env.vector_visual_observation import render_source_state_canvas_gray64
from curvyzero.env.vector_visual_observation import render_source_state_rgb_canvas_like
from curvyzero.training.curvyzero_source_state_visual_turn_commit_lightzero_env import (
    CurvyZeroSourceStateVisualTurnCommitLightZeroEnv,
    CurvyZeroSourceStateVisualTurnCommitLightZeroLocalEnv,
    ENV_VARIANT_SOURCE_STATE_TURN_COMMIT,
    LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_ID,
    LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_TYPE,
    SOURCE_STATE_CANVAS_LIKE_RAW_SCHEMA_HASH,
    SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE,
    SOURCE_STATE_TURN_COMMIT_RUNTIME_TOPOLOGY,
    STACKED_SOURCE_STATE_GRAY64_SCHEMA_HASH,
    STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
    STACKED_SOURCE_STATE_GRAY64_SHAPE,
    TURN_COMMIT_REWARD_CREDIT_STATUS,
    TURN_COMMIT_TRAINING_STATUS,
    _normalize_player_perspective,
    _player_perspective_lut,
)


def test_source_state_turn_commit_reset_shape_and_metadata():
    env = CurvyZeroSourceStateVisualTurnCommitLightZeroLocalEnv(
        {"seed": 13, "source_max_steps": 8}
    )

    observation = env.reset(seed=13)

    assert observation["observation"].shape == STACKED_SOURCE_STATE_GRAY64_SHAPE
    assert observation["observation"].dtype == np.float32
    assert float(observation["observation"].min()) >= 0.0
    assert float(observation["observation"].max()) <= 1.0
    np.testing.assert_array_equal(observation["observation"][:3], np.zeros((3, 64, 64)))
    assert observation["action_mask"].dtype == np.int8
    np.testing.assert_array_equal(observation["action_mask"], np.array([1, 1, 1], dtype=np.int8))
    assert observation["to_play"] == -1
    assert observation["timestep"] == 0
    assert env.active_player_index == 0
    assert env.active_player_id == "player_0"

    assert env.last_reset_info is not None
    assert env.last_reset_info["env_variant"] == ENV_VARIANT_SOURCE_STATE_TURN_COMMIT
    assert env.last_reset_info["observation_schema_id"] == STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID
    assert env.last_reset_info["observation_schema_hash"] == STACKED_SOURCE_STATE_GRAY64_SCHEMA_HASH
    assert env.last_reset_info["schema_hash"] == STACKED_SOURCE_STATE_GRAY64_SCHEMA_HASH
    assert env.last_reset_info["single_frame_schema_id"] == SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID
    assert (
        env.last_reset_info["single_frame_schema_hash"]
        == SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH
    )
    assert env.last_reset_info["raw_observation_schema_id"] == SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID
    assert env.last_reset_info["raw_observation_schema_hash"] == SOURCE_STATE_CANVAS_LIKE_RAW_SCHEMA_HASH
    assert env.last_reset_info["raw_observation_available"] is True
    assert env.last_reset_info["raw_observation_dtype"] == "uint8"
    assert env.last_reset_info["raw_observation_color_space"] == "RGB"
    assert env.last_reset_info["raw_frame_shape"] == list(SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE)
    assert env.last_reset_info["grayscale_frame_shape"] == list(SOURCE_STATE_CANVAS_GRAY64_SHAPE)
    assert (
        "render_source_state_canvas_gray64"
        in env.last_reset_info["grayscale_observation_source"]
    )
    assert env.last_reset_info["turn_commit_adapter"] is True
    assert env.last_reset_info["turn_commit_rule"] == (
        "physical_env_advances_only_after_all_players_commit"
    )
    assert env.last_reset_info["runtime_topology"] == SOURCE_STATE_TURN_COMMIT_RUNTIME_TOPOLOGY
    assert env.last_reset_info["debug_fidelity_only"] is False
    assert env.last_reset_info["source_fidelity_claim"] == "source_state_backed_non_browser_pixel"
    assert env.last_reset_info["visual_surface"] == SOURCE_STATE_CANVAS_GRAY64_SURFACE
    assert env.last_reset_info["visual_source_state_backed"] is True
    assert env.last_reset_info["uses_ale"] is False
    assert env.last_reset_info["current_policy_self_play"] is True
    assert env.last_reset_info["trusted_current_policy_self_play"] is False
    assert env.last_reset_info["learning_quality_claim"] is False
    assert env.last_reset_info["training_status"] == TURN_COMMIT_TRAINING_STATUS
    assert env.last_reset_info["reward_credit_status"] == TURN_COMMIT_REWARD_CREDIT_STATUS
    assert env.last_reset_info["two_seat_self_play"] is False
    assert env.last_reset_info["current_policy_two_seat_action_collection"] is True
    assert env.last_reset_info["two_seat_self_play_status"] == TURN_COMMIT_TRAINING_STATUS
    assert env.last_reset_info["source_tick_index"] == 0
    assert env.last_reset_info["adapter_timestep"] == 0

    raw = env.raw_observation()
    raw_from_render = env.render("source_state_raw_visual_tensor")
    perspective_raw = env.raw_observation(player_perspective=True)
    assert raw is not None
    assert raw_from_render is not None
    assert perspective_raw is not None
    assert raw.shape == SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE
    assert raw.dtype == np.uint8
    np.testing.assert_array_equal(raw, raw_from_render)
    np.testing.assert_array_equal(raw, env.render("source_state_rgb_canvas_like"))
    np.testing.assert_array_equal(
        raw,
        render_source_state_rgb_canvas_like(env._env.state, row=0),
    )
    human_rgb = env.human_rgb_observation(frame_size=128)
    assert human_rgb is not None
    assert human_rgb.shape == (128, 128, 3)
    assert human_rgb.dtype == np.uint8
    np.testing.assert_array_equal(
        human_rgb,
        render_source_state_rgb_canvas_like(env._env.state, row=0, frame_size=128),
    )
    assert perspective_raw.shape == SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE
    assert perspective_raw.dtype == np.uint8
    gray64 = rgb_canvas_like_to_gray64(perspective_raw)
    np.testing.assert_array_equal(
        gray64,
        render_source_state_canvas_gray64(env._env.state, row=0),
    )
    np.testing.assert_array_equal(
        env.render("source_state_grayscale64_visual_tensor"),
        gray64,
    )
    np.testing.assert_allclose(
        observation["observation"][-1],
        gray64[0].astype(np.float32) / np.float32(255.0),
        rtol=0.0,
        atol=1e-7,
    )


def test_source_state_turn_commit_pending_then_commit_advances_once():
    env = CurvyZeroSourceStateVisualTurnCommitLightZeroLocalEnv(
        {"seed": 21, "source_max_steps": 8}
    )
    env.reset(seed=21)
    before = env._env._public_info()

    pending = env.step(0)
    after_pending = env._env._public_info()

    assert pending.reward == 0.0
    assert pending.done is False
    assert pending.info["acting_player_id"] == "player_0"
    assert pending.info["next_active_player_id"] == "player_1"
    assert pending.info["physical_env_advanced"] is False
    assert pending.info["pending_action_count"] == 1
    assert pending.info["source_tick_index"] == 0
    assert pending.info["adapter_timestep"] == 0
    assert pending.info["reward_perspective"] == "bookkeeping_pending_action_no_physical_reward"
    assert env.active_player_index == 1
    assert int(after_pending["tick_index"][0]) == int(before["tick_index"][0])
    assert float(after_pending["elapsed_ms"][0]) == float(before["elapsed_ms"][0])
    pending_perspective_raw = env.raw_observation(player_perspective=True)
    assert pending_perspective_raw is not None
    pending_gray64 = rgb_canvas_like_to_gray64(pending_perspective_raw)
    np.testing.assert_array_equal(
        pending_gray64,
        render_source_state_canvas_gray64(
            env._env.state,
            row=0,
            player_rgb=env._perspective_rgb_palettes[1],
        ),
    )
    np.testing.assert_allclose(
        pending.obs["observation"][-1],
        pending_gray64[0].astype(np.float32) / np.float32(255.0),
        rtol=0.0,
        atol=1e-7,
    )

    committed = env.step(2)
    after_commit = env._env._public_info()

    assert committed.info["acting_player_id"] == "player_1"
    assert committed.info["next_active_player_id"] == "player_0"
    assert committed.info["physical_env_advanced"] is True
    assert committed.info["pending_action_count"] == 0
    assert committed.info["source_tick_index"] == 1
    assert committed.info["adapter_timestep"] == 1
    assert committed.info["joint_action"] == {"player_0": 0, "player_1": 2}
    assert committed.info["joint_action_schema_id"] == JOINT_ACTION_SCHEMA_ID
    assert committed.reward in {0.0, 1.0}
    assert env.active_player_index == 0
    assert int(after_commit["tick_index"][0]) == int(before["tick_index"][0]) + int(
        committed.info["decision_source_frames"]
    )
    assert float(after_commit["elapsed_ms"][0]) > float(before["elapsed_ms"][0])


def test_source_state_turn_commit_reset_clears_pending_state():
    env = CurvyZeroSourceStateVisualTurnCommitLightZeroLocalEnv(
        {"seed": 31, "source_max_steps": 8}
    )

    env.reset(seed=31)
    pending = env.step(1)
    assert pending.info["pending_action_count"] == 1
    assert env._pending_actions == {0: 1}

    observation = env.reset(seed=32)

    assert env.active_player_index == 0
    assert env._pending_actions == {}
    assert observation["timestep"] == 0
    assert env.last_reset_info["episode_seed"] == 32
    assert env.last_reset_info["source_tick_index"] == 0
    assert env.last_reset_info["adapter_timestep"] == 0


def test_source_state_turn_commit_telemetry_separates_pending_and_physical_rows(tmp_path):
    telemetry_path = tmp_path / "turn_commit_steps.jsonl"
    env = CurvyZeroSourceStateVisualTurnCommitLightZeroLocalEnv(
        {
            "seed": 41,
            "source_max_steps": 8,
            "telemetry_path": telemetry_path,
        }
    )
    env.reset(seed=41)

    env.step(0)
    env.step(2)

    rows = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 2
    assert rows[0]["schema_id"] == "curvyzero_source_state_visual_turn_commit_env_step/v0"
    assert rows[0]["env_variant"] == ENV_VARIANT_SOURCE_STATE_TURN_COMMIT
    assert rows[0]["physical_env_advanced"] is False
    assert rows[0]["pending_action_count"] == 1
    assert rows[0]["acting_player_id"] == "player_0"
    assert rows[0]["next_active_player_id"] == "player_1"
    assert rows[0]["reward"] == 0.0
    assert rows[1]["physical_env_advanced"] is True
    assert rows[1]["pending_action_count"] == 0
    assert rows[1]["acting_player_id"] == "player_1"
    assert rows[1]["next_active_player_id"] == "player_0"
    assert rows[1]["joint_action"] == {"player_0": 0, "player_1": 2}
    assert rows[1]["training_status"] == TURN_COMMIT_TRAINING_STATUS
    assert rows[1]["reward_credit_status"] == TURN_COMMIT_REWARD_CREDIT_STATUS


def test_registered_source_state_turn_commit_env_reuses_local_semantics():
    registered = CurvyZeroSourceStateVisualTurnCommitLightZeroEnv(
        {"seed": 51, "source_max_steps": 8}
    )
    local = CurvyZeroSourceStateVisualTurnCommitLightZeroLocalEnv(
        {"seed": 51, "source_max_steps": 8}
    )

    registered_reset = registered.reset(seed=51)
    local_reset = local.reset(seed=51)

    assert repr(registered)
    assert registered.env_id == LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_ID
    assert registered.lightzero_env_type == LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_TYPE
    np.testing.assert_allclose(registered_reset["observation"], local_reset["observation"])
    np.testing.assert_array_equal(registered_reset["action_mask"], local_reset["action_mask"])
    assert registered_reset["to_play"] == local_reset["to_play"] == -1

    registered_timestep = registered.step(0)
    local_timestep = local.step(0)

    np.testing.assert_allclose(
        registered_timestep.obs["observation"],
        local_timestep.obs["observation"],
    )
    np.testing.assert_array_equal(
        registered_timestep.obs["action_mask"],
        local_timestep.obs["action_mask"],
    )
    assert registered_timestep.reward == local_timestep.reward
    assert registered_timestep.done == local_timestep.done
    assert registered_timestep.info["physical_env_advanced"] is False
    assert registered_timestep.info["joint_action"] == local_timestep.info["joint_action"]


def test_source_state_turn_commit_render_does_not_mutate_active_stack():
    env = CurvyZeroSourceStateVisualTurnCommitLightZeroLocalEnv({"seed": 61})
    observation = env.reset(seed=61)

    first_render = env.render()
    second_render = env.render()

    assert first_render is not None
    np.testing.assert_array_equal(first_render, second_render)
    np.testing.assert_array_equal(first_render, observation["observation"])


def test_turn_commit_player_perspective_lut_remaps_only_player_body_and_head_values():
    frame = np.asarray(
        [[0, 80, 96, 128, 224, 232, 240, 248, 255]],
        dtype=np.uint8,
    ).reshape(1, 1, 9)
    out = np.zeros_like(frame)

    player_0 = _normalize_player_perspective(
        frame,
        controlled_player=0,
        out=out.copy(),
        lut=_player_perspective_lut(0),
    )
    player_1 = _normalize_player_perspective(
        frame,
        controlled_player=1,
        out=out.copy(),
        lut=_player_perspective_lut(1),
    )

    np.testing.assert_array_equal(
        player_0.reshape(-1),
        np.array([0, 80, 96, 128, 224, 232, 240, 248, 255], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        player_1.reshape(-1),
        np.array([0, 80, 128, 96, 232, 224, 240, 248, 255], dtype=np.uint8),
    )
