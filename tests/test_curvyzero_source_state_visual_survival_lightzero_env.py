import json

import numpy as np
import pytest

from curvyzero.training import curvyzero_source_state_visual_survival_lightzero_env as env_mod
from curvyzero.env import vector_runtime
from curvyzero.env.trainer_contract import ACTION_ID_TO_SOURCE_MOVE
from curvyzero.env.trainer_contract import ACTION_SPACE_ID
from curvyzero.env.vector_multiplayer_env import JOINT_ACTION_SCHEMA_ID
from curvyzero.env.vector_multiplayer_env import NATURAL_BONUS_EFFECT_TYPE_NAMES
from curvyzero.env.vector_multiplayer_env import NATURAL_BONUS_ENV_IMPL_ID
from curvyzero.env.vector_multiplayer_env import PUBLIC_NATURAL_BONUS_ENV_CONTRACT_ID
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_RGB_CANVAS_LIKE_RENDERER_IMPL_ID,
)
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
)
from curvyzero.env.vector_visual_observation import SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID
from curvyzero.env.vector_visual_observation import rgb_canvas_like_to_gray64
from curvyzero.env.vector_visual_observation import render_source_state_canvas_gray64
from curvyzero.env.vector_visual_observation import render_source_state_rgb_canvas_like
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    CurvyZeroSourceStateVisualJointActionLightZeroEnv,
    CurvyZeroSourceStateVisualJointActionLightZeroLocalEnv,
    CurvyZeroSourceStateVisualSurvivalLightZeroEnv,
    CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv,
    LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_ID,
    LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_TYPE,
    LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_ID,
    LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_TYPE,
    LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_IMPORT_NAMES,
    OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION,
    SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_HASH,
    SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_ID,
    SOURCE_STATE_CANVAS_LIKE_GRAY64_SURFACE,
    SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SCHEMA_HASH,
    SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SHAPE,
    SOURCE_STATE_DEFAULT_TRAIL_RENDER_MODE,
    SOURCE_STATE_FIXED_OPPONENT_ENV_VARIANT,
    SOURCE_STATE_FIXED_OPPONENT_RUNTIME_TOPOLOGY,
    SOURCE_STATE_FIXED_OPPONENT_TWO_SEAT_STATUS,
    SOURCE_STATE_FIXED_OPPONENT_UNDERLYING_ENV_CLASS,
    SOURCE_STATE_JOINT_ACTION_ENV_VARIANT,
    SOURCE_STATE_JOINT_ACTION_RUNTIME_TOPOLOGY,
    SOURCE_STATE_JOINT_ACTION_TRAINING_STATUS,
    SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES,
    SOURCE_STATE_TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
    SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES,
    STACKED_SOURCE_STATE_GRAY64_SCHEMA_HASH,
    STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
    STACKED_SOURCE_STATE_GRAY64_SHAPE,
    _normalize_player_perspective,
    _player_perspective_lut,
)
from curvyzero.training.multiplayer_opponent_policy import OpponentPolicySelection


def test_source_state_visual_survival_reset_shape_and_metadata():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
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
    np.testing.assert_array_equal(env.legal_actions, np.array([0, 1, 2], dtype=np.int64))
    assert env.config["action_space_size"] == 3
    assert env._action_space == {"type": "Discrete", "n": 3}

    assert env.last_reset_info is not None
    assert env.last_reset_info["observation_schema_id"] == STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID
    assert env.last_reset_info["observation_schema_hash"] == STACKED_SOURCE_STATE_GRAY64_SCHEMA_HASH
    assert env.last_reset_info["schema_hash"] == STACKED_SOURCE_STATE_GRAY64_SCHEMA_HASH
    assert env.last_reset_info["single_frame_schema_id"] == SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_ID
    assert (
        env.last_reset_info["single_frame_schema_hash"]
        == SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_HASH
    )
    assert env.last_reset_info["raw_observation_schema_id"] == SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID
    assert env.last_reset_info["raw_observation_schema_hash"] == SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SCHEMA_HASH
    assert env.last_reset_info["raw_observation_available"] is True
    assert env.last_reset_info["raw_observation_dtype"] == "uint8"
    assert env.last_reset_info["raw_observation_color_space"] == "RGB"
    assert env.last_reset_info["raw_frame_shape"] == list(SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SHAPE)
    assert env.last_reset_info["raw_frame_shape"] == [
        SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        3,
    ]
    assert (
        f"frame_size={SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE}"
        in env.last_reset_info["raw_observation_source"]
    )
    assert (
        "render_source_state_canvas_gray64"
        in env.last_reset_info["grayscale_observation_source"]
    )
    assert (
        env.last_reset_info["default_trail_render_mode"]
        == SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES
    )
    assert env.last_reset_info["default_trail_render_mode"] == (
        SOURCE_STATE_DEFAULT_TRAIL_RENDER_MODE
    )
    assert env.last_reset_info["supported_trail_render_modes"] == list(
        SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES
    )
    assert (
        env.last_reset_info["trail_render_mode"]
        == SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES
    )
    assert env.last_reset_info["trail_renderer_kind"] == "connected_rounded_lines"
    assert (
        env.last_reset_info["trail_renderer_truth_level"]
        == "source_state_browser_style_lines_non_pixel_parity"
    )
    assert env.last_reset_info["trail_renderer_is_approximation"] is False
    assert env.last_reset_info["browser_style_trail_renderer"] is True
    assert (
        env.last_reset_info["browser_trail_semantics"]
        == "persistent_background_canvas_round_line_caps"
    )
    assert "persisted body points" in env.last_reset_info[
        "browser_client_trail_point_caveat"
    ]
    assert (
        env.last_reset_info["browser_pixel_fidelity_claim"]
        == "not_validated_against_browser_canvas"
    )
    assert env.last_reset_info["debug_fidelity_only"] is False
    assert env.last_reset_info["source_fidelity_claim"] == "source_state_backed_non_browser_pixel"
    assert env.last_reset_info["visual_surface"] == SOURCE_STATE_CANVAS_LIKE_GRAY64_SURFACE
    assert (
        env.last_reset_info["visual_truth_level"]
        == "source_state_backed_browser_like_non_pixel_parity"
    )
    assert env.last_reset_info["visual_source_state_backed"] is True
    assert env.last_reset_info["browser_pixel_fidelity"] is False
    assert env.last_reset_info["env_variant"] == SOURCE_STATE_FIXED_OPPONENT_ENV_VARIANT
    assert env.last_reset_info["runtime_topology"] == SOURCE_STATE_FIXED_OPPONENT_RUNTIME_TOPOLOGY
    assert env.last_reset_info["underlying_env_class"] == (
        SOURCE_STATE_FIXED_OPPONENT_UNDERLYING_ENV_CLASS
    )
    assert env.last_reset_info["runtime_env_impl_id"] == NATURAL_BONUS_ENV_IMPL_ID
    assert env.last_reset_info["env_impl_id"] == NATURAL_BONUS_ENV_IMPL_ID
    assert (
        env.last_reset_info["public_env_contract_id"]
        == PUBLIC_NATURAL_BONUS_ENV_CONTRACT_ID
    )
    assert env.last_reset_info["natural_bonus_spawn"] is True
    assert env.last_reset_info["bonus_support_mode"] == "natural_spawn"
    assert env.last_reset_info["natural_bonus_type_codes"] == env._env.natural_bonus_type_codes.tolist()
    assert env.last_reset_info["natural_bonus_type_names"] == [
        "BonusSelfSmall",
        "BonusSelfSlow",
        "BonusSelfFast",
        "BonusSelfMaster",
        "BonusEnemySlow",
        "BonusEnemyFast",
        "BonusEnemyBig",
        "BonusEnemyInverse",
        "BonusEnemyStraightAngle",
        "BonusGameBorderless",
        "BonusAllColor",
        "BonusGameClear",
    ]
    assert env.last_reset_info["supported_natural_bonus_effect_types"] == list(
        NATURAL_BONUS_EFFECT_TYPE_NAMES
    )
    assert "BonusSelfMaster" in env.last_reset_info["supported_natural_bonus_effect_types"]


def test_source_state_visual_survival_bonus_spawn_can_be_disabled_from_cfg():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {"seed": 13, "source_max_steps": 8, "natural_bonus_spawn": False}
    )

    observation = env.reset(seed=13)

    assert env.config["natural_bonus_spawn"] is True
    assert env.last_reset_info is not None
    assert env.last_reset_info["natural_bonus_spawn"] is False
    assert "BonusAllColor" in env.last_reset_info["supported_natural_bonus_effect_types"]
    assert "BonusSelfMaster" not in env.last_reset_info[
        "unsupported_natural_bonus_effects"
    ]
    assert "BonusAllColor" not in env.last_reset_info[
        "unsupported_natural_bonus_effects"
    ]
    assert "BonusEnemyStraightAngle" not in env.last_reset_info[
        "unsupported_natural_bonus_effects"
    ]
    assert env.last_reset_info["death_mode"] == "normal"
    assert env.last_reset_info["death_suppression_for_profile"] is False
    assert env.last_reset_info["uses_ale"] is False
    assert env.last_reset_info["turn_commit_adapter"] is False
    assert env.last_reset_info["current_policy_self_play"] is False
    assert env.last_reset_info["trusted_current_policy_self_play"] is False
    assert env.last_reset_info["simultaneous_game_theory_claim"] is False
    assert env.last_reset_info["two_seat_self_play"] is False
    assert env.last_reset_info["two_seat_self_play_status"] == (
        SOURCE_STATE_FIXED_OPPONENT_TWO_SEAT_STATUS
    )
    assert env.last_reset_info["fixed_opponent_is_two_seat_self_play"] is False
    assert "two-seat" in env.last_reset_info["current_policy_self_play_blocker"]
    assert env.last_reset_info["opponent_policy_kind"] == "fixed_straight"
    assert env.last_reset_info["opponent_training_relation"] == "learner_vs_fixed_straight"
    assert env.last_reset_info["episode_seed"] == 13
    assert type(env._env) is VectorMultiplayerEnv
    assert env._env.player_count == 2
    assert env._env.batch_size == 1

    raw = env.raw_observation()
    raw_from_render = env.render("source_state_raw_visual_tensor")
    perspective_raw = env.raw_observation(player_perspective=True)
    assert raw is not None
    assert raw_from_render is not None
    assert perspective_raw is not None
    assert raw.shape == SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SHAPE
    assert raw.dtype == np.uint8
    np.testing.assert_array_equal(raw, raw_from_render)
    np.testing.assert_array_equal(raw, env.render("source_state_rgb_canvas_like"))
    human_rgb = env.human_rgb_observation(frame_size=128)
    assert human_rgb is not None
    assert human_rgb.shape == (128, 128, 3)
    assert human_rgb.dtype == np.uint8
    np.testing.assert_array_equal(
        human_rgb,
        render_source_state_rgb_canvas_like(
            env._env.state,
            row=0,
            frame_size=128,
            trail_render_mode=SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES,
        ),
    )
    np.testing.assert_array_equal(
        env.render("source_state_player_perspective_raw_visual_tensor"),
        perspective_raw,
    )
    np.testing.assert_array_equal(raw, perspective_raw)
    gray64 = rgb_canvas_like_to_gray64(raw)
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


def test_source_state_visual_survival_terminal_snapshots_survive_manual_reset():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {"seed": 19, "source_max_steps": 1}
    )
    env.reset(seed=19)

    timestep = env.step(2)

    assert timestep.done is True
    assert timestep.info["needs_reset"] is True
    terminal_final_observation = timestep.info["final_observation"]
    assert terminal_final_observation is not None
    assert terminal_final_observation["observation"] is not timestep.obs["observation"]
    saved_final_stack = terminal_final_observation["observation"].copy()
    saved_terminal_raw = env.raw_observation()
    assert saved_terminal_raw is not None
    np.testing.assert_array_equal(
        saved_terminal_raw,
        render_source_state_rgb_canvas_like(env._env.state, row=0),
    )

    returned_raw_copy = env.raw_observation()
    assert returned_raw_copy is not None
    returned_raw_copy.fill(0)
    np.testing.assert_array_equal(
        env.raw_observation(),
        saved_terminal_raw,
    )

    with pytest.raises(RuntimeError, match="reset must be called before stepping after done"):
        env.step(1)
    np.testing.assert_array_equal(
        env.raw_observation(),
        saved_terminal_raw,
    )

    env.reset(seed=20)

    np.testing.assert_array_equal(
        terminal_final_observation["observation"],
        saved_final_stack,
    )


def test_source_state_visual_survival_step_and_terminal_telemetry(tmp_path):
    telemetry_path = tmp_path / "source_state_steps.jsonl"
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 21,
            "source_max_steps": 1,
            "telemetry_path": telemetry_path,
        }
    )
    reset_observation = env.reset(seed=21)

    timestep = env.step(0)

    assert timestep.done is True
    assert timestep.reward in {0.0, 1.0}
    assert timestep.obs["observation"].shape == STACKED_SOURCE_STATE_GRAY64_SHAPE
    np.testing.assert_array_equal(
        timestep.obs["action_mask"],
        np.array([0, 0, 0], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        timestep.obs["observation"][-2],
        reset_observation["observation"][-1],
    )
    np.testing.assert_array_equal(
        timestep.info["final_observation"]["action_mask"],
        np.array([0, 0, 0], dtype=np.int8),
    )
    assert timestep.info["requested_ego_action"] == 0
    assert timestep.info["executed_ego_action"] == 0
    assert timestep.info["joint_action"] == {"player_0": 0, "player_1": 1}
    assert timestep.info["joint_action_schema_id"] == JOINT_ACTION_SCHEMA_ID
    assert timestep.info["opponent_action_id"] == 1
    assert timestep.info["opponent_policy_kind"] == "fixed_straight"
    assert timestep.info["opponent_training_relation"] == "learner_vs_fixed_straight"
    assert timestep.info["physical_env_advanced"] is True
    assert timestep.info["reward_perspective"] == "ego_player_sparse_round_outcome"
    assert "terminal_reason" in timestep.info
    assert "death_count" in timestep.info
    assert "death_player" in timestep.info
    assert "death_cause" in timestep.info
    assert "death_cause_name" in timestep.info
    assert "death_hit_owner" in timestep.info
    assert timestep.info["final_observation"] is not None
    np.testing.assert_array_equal(
        timestep.info["final_observation"]["observation"],
        timestep.obs["observation"],
    )
    assert timestep.info["eval_episode_return"] == timestep.reward
    assert timestep.info["episode_training_return"] == timestep.reward

    terminal_raw = env.raw_observation()
    terminal_raw_from_render = env.render("source_state_raw_visual_tensor")
    terminal_player_perspective_raw = env.raw_observation(player_perspective=True)
    assert terminal_raw is not None
    assert terminal_raw_from_render is not None
    assert terminal_player_perspective_raw is not None
    assert terminal_raw.shape == SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SHAPE
    assert terminal_raw.dtype == np.uint8
    np.testing.assert_array_equal(terminal_raw, terminal_raw_from_render)
    np.testing.assert_array_equal(
        terminal_raw,
        render_source_state_rgb_canvas_like(env._env.state, row=0),
    )
    terminal_gray64 = rgb_canvas_like_to_gray64(terminal_player_perspective_raw)
    np.testing.assert_array_equal(
        terminal_gray64,
        render_source_state_canvas_gray64(env._env.state, row=0),
    )
    np.testing.assert_allclose(
        timestep.info["final_observation"]["observation"][-1],
        terminal_gray64[0].astype(np.float32) / np.float32(255.0),
        rtol=0.0,
        atol=1e-7,
    )

    assert timestep.info["single_frame_schema_id"] == SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_ID
    assert (
        timestep.info["single_frame_schema_hash"]
        == SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_HASH
    )
    assert timestep.info["raw_observation_schema_id"] == SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID
    assert timestep.info["raw_observation_schema_hash"] == SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SCHEMA_HASH
    assert timestep.info["raw_observation_available"] is True
    assert timestep.info["raw_observation_accessors"] == [
        "raw_observation()",
        "render('source_state_raw_visual_tensor')",
        "render('source_state_rgb_canvas_like')",
    ]
    assert timestep.info["raw_observation_dtype"] == "uint8"
    assert timestep.info["raw_observation_color_space"] == "RGB"
    assert timestep.info["raw_renderer_impl_id"] == SOURCE_STATE_RGB_CANVAS_LIKE_RENDERER_IMPL_ID
    assert (
        timestep.info["default_trail_render_mode"]
        == SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES
    )
    assert timestep.info["supported_trail_render_modes"] == list(
        SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES
    )
    assert timestep.info["trail_render_mode"] == SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES
    assert timestep.info["trail_renderer_kind"] == "connected_rounded_lines"
    assert (
        timestep.info["trail_renderer_truth_level"]
        == "source_state_browser_style_lines_non_pixel_parity"
    )
    assert timestep.info["trail_renderer_is_approximation"] is False
    assert timestep.info["browser_style_trail_renderer"] is True
    assert timestep.info["browser_pixel_fidelity"] is False
    assert (
        timestep.info["browser_pixel_fidelity_claim"]
        == "not_validated_against_browser_canvas"
    )
    assert timestep.info["uses_ale"] is False
    assert timestep.info["ale_usage"] == "none"

    runtime_info = env._env.last_step_info
    assert runtime_info is not None
    np.testing.assert_array_equal(
        runtime_info["joint_action"],
        np.array([[0, 1]], dtype=np.int16),
    )
    action_sidecar = runtime_info["action_sidecar"]
    assert action_sidecar["action_space_id"] == ACTION_SPACE_ID
    assert action_sidecar["joint_action_schema_id"] == JOINT_ACTION_SCHEMA_ID
    np.testing.assert_array_equal(
        action_sidecar["player_action"],
        np.array([[0, 1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        action_sidecar["player_action_mask"],
        np.ones((1, 2, 3), dtype=bool),
    )
    np.testing.assert_array_equal(
        action_sidecar["action_required"],
        np.array([[True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        action_sidecar["native_control_value"],
        np.array(
            [[ACTION_ID_TO_SOURCE_MOVE[0], ACTION_ID_TO_SOURCE_MOVE[1]]],
            dtype=np.int8,
        ),
    )
    np.testing.assert_array_equal(
        action_sidecar["action_source"],
        np.array([["external_joint_action", "external_joint_action"]], dtype=object),
    )

    rows = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 1
    assert rows[0]["schema_id"] == "curvyzero_source_state_visual_survival_env_step/v0"
    assert rows[0]["source_fidelity_claim"] == "source_state_backed_non_browser_pixel"
    assert rows[0]["visual_surface"] == SOURCE_STATE_CANVAS_LIKE_GRAY64_SURFACE
    assert rows[0]["visual_truth_level"] == "source_state_backed_browser_like_non_pixel_parity"
    assert rows[0]["visual_source_state_backed"] is True
    assert rows[0]["default_trail_render_mode"] == SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES
    assert rows[0]["trail_render_mode"] == SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES
    assert rows[0]["trail_renderer_kind"] == "connected_rounded_lines"
    assert (
        rows[0]["trail_renderer_truth_level"]
        == "source_state_browser_style_lines_non_pixel_parity"
    )
    assert rows[0]["trail_renderer_is_approximation"] is False
    assert rows[0]["browser_style_trail_renderer"] is True
    assert rows[0]["browser_pixel_fidelity_claim"] == "not_validated_against_browser_canvas"
    assert rows[0]["uses_ale"] is False
    assert rows[0]["env_variant"] == SOURCE_STATE_FIXED_OPPONENT_ENV_VARIANT
    assert rows[0]["runtime_topology"] == SOURCE_STATE_FIXED_OPPONENT_RUNTIME_TOPOLOGY
    assert rows[0]["underlying_env_class"] == SOURCE_STATE_FIXED_OPPONENT_UNDERLYING_ENV_CLASS
    assert rows[0]["runtime_env_impl_id"] == NATURAL_BONUS_ENV_IMPL_ID
    assert rows[0]["natural_bonus_spawn"] is True
    assert rows[0]["bonus_support_mode"] == "natural_spawn"
    assert rows[0]["death_mode"] == "normal"
    assert rows[0]["death_suppression_for_profile"] is False
    assert rows[0]["debug_fidelity_only"] is False
    assert rows[0]["current_policy_self_play"] is False
    assert rows[0]["simultaneous_game_theory_claim"] is False
    assert rows[0]["two_seat_self_play"] is False
    assert rows[0]["two_seat_self_play_status"] == SOURCE_STATE_FIXED_OPPONENT_TWO_SEAT_STATUS
    assert rows[0]["fixed_opponent_is_two_seat_self_play"] is False
    assert "terminal_reason" in rows[0]
    assert "death_count" in rows[0]
    assert "death_player" in rows[0]
    assert "death_cause" in rows[0]
    assert "death_cause_name" in rows[0]
    assert "death_hit_owner" in rows[0]
    assert rows[0]["eval_episode_return"] == timestep.reward


def test_source_state_visual_survival_profile_env_timing_is_opt_in(tmp_path):
    telemetry_path = tmp_path / "source_state_timed_steps.jsonl"
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 22,
            "source_max_steps": 2,
            "telemetry_path": telemetry_path,
            "telemetry_stride": 1,
            "profile_env_timing_enabled": True,
        }
    )
    env.reset(seed=22)

    timestep = env.step(1)

    timing = timestep.info["profile_env_timing_sec"]
    assert timing["opponent_action_sec"] >= 0.0
    assert timing["vector_step_sec"] >= 0.0
    assert timing["observation_sec"] >= 0.0
    assert timing["step_total_before_info_sec"] >= timing["observation_sec"]

    rows = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 1
    assert rows[0]["profile_env_timing_sec"]["opponent_action_sec"] >= 0.0
    assert rows[0]["profile_env_timing_sec"]["observation_sec"] >= 0.0


def test_source_state_visual_survival_can_use_frozen_checkpoint_opponent(
    monkeypatch,
    tmp_path,
):
    calls = []
    telemetry_path = tmp_path / "frozen_source_state_steps.jsonl"

    class FakeFrozenOpponentPolicy:
        policy_id = "fake_snapshot_policy"
        policy_version = "v-test"
        seed = 99

        def select_actions(
            self,
            legal_action_mask,
            opponent_mask,
            *,
            decision_index=0,
            observation=None,
        ):
            calls.append(
                {
                    "legal_action_mask": legal_action_mask.copy(),
                    "opponent_mask": opponent_mask.copy(),
                    "decision_index": decision_index,
                    "observation": observation.copy(),
                }
            )
            actions = np.full((1, 2), -1, dtype=np.int16)
            actions[0, 1] = 2
            action_seed = np.full((1, 2), -1, dtype=np.int64)
            action_seed[0, 1] = 123
            action_logp = np.full((1, 2), np.nan, dtype=np.float32)
            action_logp[0, 1] = -0.25
            return OpponentPolicySelection(
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                seed=self.seed,
                actions=actions,
                action_seed=action_seed,
                action_logp=action_logp,
                opponent_mask=opponent_mask,
                decision_index=decision_index,
                policy_metadata={
                    "checkpoint_ref": "runs/checkpoints/iteration_7.pth.tar",
                    "snapshot_ref": "stage-007",
                    "provider_id": "fake_provider",
                    "provider_version": "v-test",
                    "provider_load_summary": {
                        "ok": True,
                        "strict": True,
                        "candidate": "as_is",
                    },
                },
            )

    def fake_builder(**kwargs):
        assert kwargs["checkpoint_path"] == "/tmp/frozen.pth.tar"
        assert kwargs["checkpoint_ref"] == "runs/checkpoints/iteration_7.pth.tar"
        assert kwargs["snapshot_ref"] == "stage-007"
        assert kwargs["state_key"] == "model"
        return FakeFrozenOpponentPolicy()

    monkeypatch.setattr(
        env_mod,
        "snapshot_backed_lightzero_checkpoint_opponent_policy",
        fake_builder,
    )
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 21,
            "source_max_steps": 2,
            "opponent_policy_kind": "frozen_lightzero_checkpoint",
            "opponent_checkpoint_path": "/tmp/frozen.pth.tar",
            "opponent_checkpoint_ref": "runs/checkpoints/iteration_7.pth.tar",
            "opponent_snapshot_ref": "stage-007",
            "opponent_checkpoint_state_key": "model",
            "opponent_policy_seed": 99,
            "telemetry_path": telemetry_path,
        }
    )

    env.reset(seed=21)
    timestep = env.step(0)

    assert timestep.info["joint_action"] == {"player_0": 0, "player_1": 2}
    assert timestep.info["opponent_action_id"] == 2
    assert timestep.info["opponent_policy_kind"] == "frozen_lightzero_checkpoint"
    assert (
        timestep.info["opponent_training_relation"]
        == "learner_vs_frozen_lightzero_checkpoint"
    )
    assert timestep.info["opponent_policy_id"] == "fake_snapshot_policy"
    assert timestep.info["opponent_policy_version"] == "v-test"
    sidecar = timestep.info["opponent_policy_sidecar"]
    assert sidecar["policy_metadata"]["checkpoint_ref"] == (
        "runs/checkpoints/iteration_7.pth.tar"
    )
    assert calls[0]["decision_index"] == 0
    np.testing.assert_array_equal(calls[0]["opponent_mask"], np.array([[False, True]]))
    assert calls[0]["observation"].shape == (1, 2, *STACKED_SOURCE_STATE_GRAY64_SHAPE)
    assert calls[0]["legal_action_mask"].shape == (1, 2, 3)
    rows = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert rows[0]["opponent_checkpoint_ref"] == "runs/checkpoints/iteration_7.pth.tar"
    assert rows[0]["opponent_snapshot_ref"] == "stage-007"
    assert rows[0]["opponent_provider_id"] == "fake_provider"
    assert rows[0]["opponent_provider_load_ok"] is True
    assert rows[0]["opponent_provider_load_strict"] is True
    assert rows[0]["opponent_provider_load_candidate"] == "as_is"


def test_source_state_fixed_opponent_config_names_non_self_play_runtime_contract():
    config = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv.config

    assert config["env_variant"] == SOURCE_STATE_FIXED_OPPONENT_ENV_VARIANT
    assert config["runtime_topology"] == SOURCE_STATE_FIXED_OPPONENT_RUNTIME_TOPOLOGY
    assert config["underlying_env_class"] == SOURCE_STATE_FIXED_OPPONENT_UNDERLYING_ENV_CLASS
    assert config["source_fidelity_claim"] == "source_state_backed_non_browser_pixel"
    assert config["source_state_trail_render_mode"] == (
        SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES
    )
    assert config["default_trail_render_mode"] == SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES
    assert config["supported_trail_render_modes"] == SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES
    assert config["two_seat_self_play"] is False
    assert config["two_seat_self_play_status"] == SOURCE_STATE_FIXED_OPPONENT_TWO_SEAT_STATUS
    assert config["fixed_opponent_is_two_seat_self_play"] is False


def test_source_state_visual_survival_allows_explicit_fast_trail_mode_metadata():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 23,
            "source_max_steps": 1,
            "source_state_trail_render_mode": SOURCE_STATE_TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
        }
    )

    env.reset(seed=23)
    timestep = env.step(1)

    assert (
        env.last_reset_info["default_trail_render_mode"]
        == SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES
    )
    assert (
        env.last_reset_info["trail_render_mode"]
        == SOURCE_STATE_TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
    )
    assert env.last_reset_info["trail_renderer_kind"] == "circle_per_body"
    assert (
        env.last_reset_info["trail_renderer_truth_level"]
        == "source_state_fast_body_circle_approximation"
    )
    assert env.last_reset_info["trail_renderer_is_approximation"] is True
    assert env.last_reset_info["browser_style_trail_renderer"] is False
    assert env.raw_observation().shape == SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SHAPE
    assert timestep.info["trail_render_mode"] == (
        SOURCE_STATE_TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
    )
    assert timestep.info["trail_renderer_is_approximation"] is True
    assert timestep.info["final_observation"] is not None


def test_source_state_visual_survival_rejects_unknown_trail_mode():
    with pytest.raises(ValueError, match="source_state_trail_render_mode"):
        CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
            {"source_state_trail_render_mode": "mystery_trails"}
        )


def test_source_state_visual_survival_profile_no_death_exercises_natural_bonus_timer():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 55,
            "source_max_steps": 20,
            "decision_ms": 1000.0,
            "disable_death_for_profile": True,
        }
    )

    env.reset(seed=55)
    timestep = None
    for _ in range(12):
        timestep = env.step(1)
        if timestep.info["natural_bonus_pop_count"] >= 1:
            break

    assert timestep is not None
    assert timestep.done is False
    assert timestep.info["death_mode"] == "profile_no_death"
    assert timestep.info["death_suppression_for_profile"] is True
    assert timestep.info["death_suppression_claim"] == "profile_only_not_source_fidelity"
    assert timestep.info["natural_bonus_spawn"] is True
    assert timestep.info["natural_bonus_pop_count"] >= 1
    assert bool(env._env.state["alive"].all()) is True


def test_source_state_visual_survival_profile_no_death_handles_long_natural_bonus_stack():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 1172,
            "source_max_steps": 400,
            "decision_ms": 300.0,
            "disable_death_for_profile": True,
        }
    )

    env.reset(seed=1172)
    rng = np.random.default_rng(1172)
    timestep = None
    for _ in range(400):
        timestep = env.step(int(rng.integers(0, 3)))

    assert timestep is not None
    assert timestep.done
    assert timestep.info["terminal_reason"] == "timeout"
    assert timestep.info["death_mode"] == "profile_no_death"
    assert timestep.info["unsupported_natural_bonus_effects"] == []
    assert timestep.info["natural_bonus_pop_count"] > 0
    assert (
        env._env.state["bonus_stack_id"].shape[2]
        >= vector_runtime.SOURCE_MAX_ACTIVE_BONUSES
    )


def test_source_state_visual_survival_profile_no_death_handles_crowded_bonus_positions():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 1181,
            "source_max_steps": 240,
            "decision_ms": 300.0,
            "disable_death_for_profile": True,
        }
    )

    env.reset(seed=1181)
    rng = np.random.default_rng(1181)
    timestep = None
    for _ in range(240):
        timestep = env.step(int(rng.integers(0, 3)))

    assert timestep is not None
    assert timestep.done
    assert timestep.info["terminal_reason"] == "timeout"
    assert timestep.info["death_mode"] == "profile_no_death"
    assert timestep.info["unsupported_natural_bonus_effects"] == []
    assert timestep.info["natural_bonus_pop_count"] > 0
    assert env._env.natural_bonus_position_attempt_capacity == 256


def test_source_state_visual_survival_action_override_is_explicit_and_seeded():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 31,
            "source_max_steps": 4,
            "ego_action_straight_override_probability": 1.0,
        }
    )

    env.reset(seed=31)
    timestep = env.step(0)

    assert timestep.info["requested_ego_action"] == 0
    assert timestep.info["executed_ego_action"] == 1
    assert timestep.info["ego_action_override_applied"] is True
    assert timestep.info["ego_action_straight_override_probability"] == 1.0
    assert timestep.info["ego_action_straight_override_seed"] == 1040
    assert timestep.info["control_noise_profile_id"] == "straight_override"


def test_source_state_visual_survival_render_does_not_mutate_stack():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv({"seed": 41})
    observation = env.reset(seed=41)

    first_render = env.render()
    second_render = env.render()

    assert first_render is not None
    np.testing.assert_array_equal(first_render, second_render)
    np.testing.assert_array_equal(first_render, observation["observation"])


def test_registered_source_state_visual_survival_env_reuses_local_semantics():
    registered = CurvyZeroSourceStateVisualSurvivalLightZeroEnv(
        {"seed": 51, "source_max_steps": 8}
    )
    local = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {"seed": 51, "source_max_steps": 8}
    )

    registered_reset = registered.reset(seed=51)
    local_reset = local.reset(seed=51)

    assert repr(registered)
    assert registered.env_id == LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_ID
    assert registered.lightzero_env_type == LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_TYPE
    assert CurvyZeroSourceStateVisualSurvivalLightZeroEnv.config["env_id"] == (
        LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_ID
    )
    assert CurvyZeroSourceStateVisualSurvivalLightZeroEnv.config["lightzero_import_names"] == (
        LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_IMPORT_NAMES
    )
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
    assert registered_timestep.info["joint_action"] == local_timestep.info["joint_action"]
    assert registered_timestep.info["trace_hash"] == local_timestep.info["trace_hash"]


def test_registered_source_state_visual_survival_env_exposes_spaces_and_random_action():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroEnv({"seed": 61})
    env.reset(seed=61)

    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.reward_space is not None
    assert _space_shape(env.observation_space) == STACKED_SOURCE_STATE_GRAY64_SHAPE
    assert _space_shape(env.reward_space) == ()
    assert _space_action_count(env.action_space) == 3
    np.testing.assert_array_equal(env.legal_actions, np.array([0, 1, 2], dtype=np.int64))
    assert env.random_action() in {0, 1, 2}


def test_source_state_joint_action_accepts_none_centralized_opponent_kind():
    env = CurvyZeroSourceStateVisualJointActionLightZeroLocalEnv(
        {
            "seed": 71,
            "source_max_steps": 8,
            "opponent_policy_kind": OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION,
        }
    )

    observation = env.reset(seed=71)
    timestep = env.step(4)

    assert env.env_id == LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_ID
    assert env.last_reset_info["env_variant"] == SOURCE_STATE_JOINT_ACTION_ENV_VARIANT
    assert env.last_reset_info["runtime_topology"] == (
        SOURCE_STATE_JOINT_ACTION_RUNTIME_TOPOLOGY
    )
    assert env.last_reset_info["opponent_policy_kind"] == (
        OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION
    )
    assert env.last_reset_info["opponent_training_relation"] == (
        "centralized_policy_controls_both_players"
    )
    assert observation["action_mask"].shape == (9,)
    np.testing.assert_array_equal(env.legal_actions, np.arange(9, dtype=np.int64))
    assert timestep.info["joint_action_scalar"] == 4
    assert timestep.info["joint_action"] == {"player_0": 1, "player_1": 1}
    assert timestep.info["centralized_joint_action_control"] is True
    assert timestep.info["true_competitive_self_play"] is False
    assert timestep.info["two_seat_self_play"] is False
    assert timestep.info["two_seat_self_play_status"] == (
        SOURCE_STATE_JOINT_ACTION_TRAINING_STATUS
    )
    assert timestep.info["current_policy_self_play"] is False
    assert timestep.info["current_policy_self_play_blocker"] == (
        "centralized_joint_action_control_is_not_true_competitive_self_play"
    )


def test_registered_source_state_joint_action_env_exposes_nine_actions():
    env = CurvyZeroSourceStateVisualJointActionLightZeroEnv(
        {
            "seed": 73,
            "source_max_steps": 8,
            "opponent_policy_kind": OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION,
        }
    )
    env.reset(seed=73)

    assert env.env_id == LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_ID
    assert env.lightzero_env_type == LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_TYPE
    assert _space_action_count(env.action_space) == 9
    assert env.random_action() in set(range(9))


def test_player_perspective_lut_remaps_only_player_body_and_head_values():
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


def _space_shape(space):
    if isinstance(space, dict):
        return space["shape"]
    return space.shape


def _space_action_count(space):
    if isinstance(space, dict):
        return space["n"]
    return space.n
