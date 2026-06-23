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
from curvyzero.env.vector_multiplayer_env import SOURCE_PHYSICS_STEP_MS
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.env.vector_visual_observation import (
    BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
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
    OPPONENT_DEATH_MODE_IMMORTAL,
    OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
    OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION,
    OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
    SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_HASH,
    SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_ID,
    SOURCE_STATE_CANVAS_LIKE_GRAY64_SURFACE,
    SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SCHEMA_HASH,
    SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SHAPE,
    SOURCE_STATE_DEFAULT_BONUS_RENDER_MODE,
    SOURCE_STATE_DEFAULT_TRAIL_RENDER_MODE,
    SOURCE_STATE_FIXED_OPPONENT_ENV_VARIANT,
    SOURCE_STATE_FIXED_OPPONENT_RUNTIME_TOPOLOGY,
    SOURCE_STATE_FIXED_OPPONENT_TWO_SEAT_STATUS,
    SOURCE_STATE_FIXED_OPPONENT_UNDERLYING_ENV_CLASS,
    SOURCE_STATE_JOINT_ACTION_ENV_VARIANT,
    SOURCE_STATE_JOINT_ACTION_RUNTIME_TOPOLOGY,
    SOURCE_STATE_JOINT_ACTION_TRAINING_STATUS,
    SOURCE_STATE_SUPPORTED_BONUS_RENDER_MODES,
    SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES,
    SOURCE_STATE_TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
    SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES,
    STACKED_SOURCE_STATE_GRAY64_SCHEMA_HASH,
    STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
    STACKED_SOURCE_STATE_GRAY64_SHAPE,
    SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD,
    SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_ID,
    SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA_ID,
    _normalize_player_perspective,
    _player_perspective_lut,
)
from curvyzero.training.multiplayer_opponent_policy import OpponentPolicySelection


def _ego_player_rgb(env: CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv):
    return env_mod._player_perspective_rgb_palette(
        env._render_state_view(),
        row=0,
        controlled_player=env.ego_player_index,
        player_count=2,
    )


def test_jax_scalar_gpu_trail_slot_profile_uses_active_prefix_bucket():
    state = {
        "visual_trail_active": np.zeros((1, 4096), dtype=bool),
        "visual_trail_write_cursor": np.asarray([933], dtype=np.int32),
    }
    state["visual_trail_active"][0, 930] = True
    renderer = env_mod._JaxScalarPolicyObservationRenderer(
        player_count=2,
        min_trail_slots=1024,
    )

    profile = renderer._trail_slot_profile(state)

    assert profile["visual_trail_capacity"] == 4096
    assert profile["visual_trail_last_active_exclusive"] == 931
    assert profile["render_trail_slots"] == 1024
    assert profile["render_trail_slots_reduced_from_capacity"] is True


def test_jax_scalar_gpu_trail_slot_profile_grows_bucket_without_truncating():
    state = {
        "visual_trail_active": np.zeros((1, 4096), dtype=bool),
        "visual_trail_write_cursor": np.asarray([1501], dtype=np.int32),
    }
    state["visual_trail_active"][0, 1500] = True
    renderer = env_mod._JaxScalarPolicyObservationRenderer(
        player_count=2,
        min_trail_slots=1024,
    )

    profile = renderer._trail_slot_profile(state)

    assert profile["render_trail_slots"] == 2048


def test_jax_scalar_gpu_trail_slot_profile_ignores_stale_slots_after_cursor():
    state = {
        "visual_trail_active": np.zeros((1, 4096), dtype=bool),
        "visual_trail_write_cursor": np.asarray([3], dtype=np.int32),
    }
    state["visual_trail_active"][0, 2] = True
    state["visual_trail_active"][0, 1500] = True
    renderer = env_mod._JaxScalarPolicyObservationRenderer(
        player_count=2,
        min_trail_slots=1024,
    )

    profile = renderer._trail_slot_profile(state)

    assert profile["visual_trail_active_count"] == 1
    assert profile["visual_trail_last_active_exclusive"] == 3
    assert profile["render_trail_slots"] == 1024


def _joint_action_for_env(
    env: CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv,
    *,
    ego_action: int,
    opponent_action: int,
) -> np.ndarray:
    actions = np.full((1, 2), 1, dtype=np.int16)
    actions[0, env.ego_player_index] = int(ego_action)
    actions[0, env.opponent_player_index] = int(opponent_action)
    return actions


def _joint_action_dict_for_env(
    env: CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv,
    *,
    ego_action: int,
    opponent_action: int,
) -> dict[str, int]:
    actions = _joint_action_for_env(
        env,
        ego_action=ego_action,
        opponent_action=opponent_action,
    )
    return {"player_0": int(actions[0, 0]), "player_1": int(actions[0, 1])}


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
    assert (
        env.last_reset_info["source_state_bonus_render_mode"]
        == BONUS_RENDER_MODE_SIMPLE_SYMBOLS
    )
    assert env.last_reset_info["default_bonus_render_mode"] == (
        SOURCE_STATE_DEFAULT_BONUS_RENDER_MODE
    )
    assert env.last_reset_info["supported_bonus_render_modes"] == list(
        SOURCE_STATE_SUPPORTED_BONUS_RENDER_MODES
    )
    assert (
        env.last_reset_info["model_observation_bonus_render_mode"]
        == BONUS_RENDER_MODE_SIMPLE_SYMBOLS
    )
    assert (
        env.last_reset_info["raw_observation_bonus_render_mode"]
        == BONUS_RENDER_MODE_SIMPLE_SYMBOLS
    )
    assert env.last_reset_info["bonus_render_mode"] == BONUS_RENDER_MODE_SIMPLE_SYMBOLS
    assert env.last_reset_info["bonus_renderer_kind"] == "simple_symbol_masks"
    assert env.last_reset_info["bonus_renderer_is_approximation"] is True
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
            env._render_state_view(),
            row=0,
            frame_size=128,
            trail_render_mode=SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES,
            bonus_render_mode=env._raw_observation_bonus_render_mode,
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
        render_source_state_canvas_gray64(
            env._render_state_view(),
            row=0,
            bonus_render_mode=env._raw_observation_bonus_render_mode,
        ),
    )
    model_gray64 = render_source_state_canvas_gray64(
        env._render_state_view(),
        row=0,
        player_rgb=_ego_player_rgb(env),
        bonus_render_mode=env._model_bonus_render_mode,
    )
    np.testing.assert_array_equal(
        env.render("source_state_grayscale64_visual_tensor"),
        model_gray64,
    )
    np.testing.assert_allclose(
        observation["observation"][-1],
        model_gray64[0].astype(np.float32) / np.float32(255.0),
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
        render_source_state_rgb_canvas_like(
            env._render_state_view(),
            row=0,
            bonus_render_mode=env._raw_observation_bonus_render_mode,
        ),
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
    assert timestep.info["joint_action"] == _joint_action_dict_for_env(
        env,
        ego_action=0,
        opponent_action=1,
    )
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
        render_source_state_rgb_canvas_like(
            env._render_state_view(),
            row=0,
            bonus_render_mode=env._raw_observation_bonus_render_mode,
        ),
    )
    terminal_gray64 = rgb_canvas_like_to_gray64(terminal_player_perspective_raw)
    np.testing.assert_array_equal(
        terminal_gray64,
        render_source_state_canvas_gray64(
            env._render_state_view(),
            row=0,
            bonus_render_mode=env._raw_observation_bonus_render_mode,
        ),
    )
    terminal_model_gray64 = render_source_state_canvas_gray64(
        env._render_state_view(),
        row=0,
        player_rgb=_ego_player_rgb(env),
        bonus_render_mode=env._model_bonus_render_mode,
    )
    np.testing.assert_allclose(
        timestep.info["final_observation"]["observation"][-1],
        terminal_model_gray64[0].astype(np.float32) / np.float32(255.0),
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
    assert timestep.info["bonus_render_mode"] == BONUS_RENDER_MODE_SIMPLE_SYMBOLS
    assert (
        timestep.info["model_observation_bonus_render_mode"]
        == BONUS_RENDER_MODE_SIMPLE_SYMBOLS
    )
    assert (
        timestep.info["raw_observation_bonus_render_mode"]
        == BONUS_RENDER_MODE_SIMPLE_SYMBOLS
    )
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
        _joint_action_for_env(env, ego_action=0, opponent_action=1),
    )
    action_sidecar = runtime_info["action_sidecar"]
    assert action_sidecar["action_space_id"] == ACTION_SPACE_ID
    assert action_sidecar["joint_action_schema_id"] == JOINT_ACTION_SCHEMA_ID
    np.testing.assert_array_equal(
        action_sidecar["player_action"],
        _joint_action_for_env(env, ego_action=0, opponent_action=1),
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
        np.asarray(ACTION_ID_TO_SOURCE_MOVE, dtype=np.int8)[
            _joint_action_for_env(env, ego_action=0, opponent_action=1)
        ],
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
    assert rows[0]["decision_ms"] == pytest.approx(SOURCE_PHYSICS_STEP_MS)
    assert rows[0]["decision_source_frames"] == 1
    assert rows[0]["source_physics_step_ms"] == pytest.approx(SOURCE_PHYSICS_STEP_MS)
    assert rows[0]["max_ticks"] == 1
    assert rows[0]["max_source_ticks"] == 1
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
    assert rows[0]["policy_action_repeat_min"] == 1
    assert rows[0]["policy_action_repeat_max"] == 1
    assert rows[0]["policy_action_repeat_extra_probability"] == 0.0
    assert rows[0]["policy_action_repeat_requested"] == 1
    assert rows[0]["policy_action_repeat_executed"] == 1
    assert rows[0]["policy_action_repeat_extra_steps"] == 0
    assert rows[0]["physical_decision_ms_total"] == pytest.approx(SOURCE_PHYSICS_STEP_MS)
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
    assert timing["update_stack_sec"] >= 0.0
    assert timing["observation_stack_copy_sec"] >= 0.0
    assert timing["action_mask_copy_sec"] >= 0.0
    assert timing["base_info_sec"] >= 0.0
    assert timing["step_total_before_info_sec"] >= timing["observation_sec"]

    rows = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 1
    assert rows[0]["profile_env_timing_sec"]["opponent_action_sec"] >= 0.0
    assert rows[0]["profile_env_timing_sec"]["observation_sec"] >= 0.0
    assert rows[0]["profile_env_timing_sec"]["update_stack_sec"] >= 0.0
    assert rows[0]["profile_env_timing_sec"]["action_mask_copy_sec"] >= 0.0


def test_registered_source_state_env_profile_records_base_timestep_payload(tmp_path):
    telemetry_path = tmp_path / "registered_source_state_timed_steps.jsonl"
    env = CurvyZeroSourceStateVisualSurvivalLightZeroEnv(
        {
            "seed": 23,
            "source_max_steps": 2,
            "telemetry_path": telemetry_path,
            "telemetry_stride": 1,
            "profile_env_timing_enabled": True,
        }
    )
    env.reset(seed=23)

    timestep = env.step(1)

    timing = timestep.info["profile_env_timing_sec"]
    assert timing["base_env_timestep_construct_sec"] >= 0.0
    assert timing["base_env_timestep_pickle_sec"] >= 0.0
    assert timing["base_env_timestep_pickle_bytes"] > 0.0
    expected_obs_bytes = float(np.prod(STACKED_SOURCE_STATE_GRAY64_SHAPE) * 4)
    assert timing["base_env_timestep_array_bytes"] >= expected_obs_bytes
    rows = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 1
    row_timing = rows[0]["profile_env_timing_sec"]
    assert row_timing["base_env_timestep_construct_sec"] >= 0.0
    assert row_timing["base_env_timestep_pickle_bytes"] == timing["base_env_timestep_pickle_bytes"]


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
            actions[opponent_mask] = 2
            action_seed = np.full((1, 2), -1, dtype=np.int64)
            action_seed[opponent_mask] = 123
            action_logp = np.full((1, 2), np.nan, dtype=np.float32)
            action_logp[opponent_mask] = -0.25
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

    assert timestep.info["joint_action"] == _joint_action_dict_for_env(
        env,
        ego_action=0,
        opponent_action=2,
    )
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
    expected_opponent_mask = np.zeros((1, 2), dtype=bool)
    expected_opponent_mask[0, env.opponent_player_index] = True
    np.testing.assert_array_equal(calls[0]["opponent_mask"], expected_opponent_mask)
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


def test_source_state_env_rejects_body_circles_fast_trail_mode():
    with pytest.raises(ValueError, match="source_state_trail_render_mode"):
        CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
            {
                "source_state_trail_render_mode": SOURCE_STATE_TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
            }
        )


def test_source_state_survival_plus_bonus_no_outcome_keeps_survival_without_catch():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 91,
            "source_max_steps": 4,
            "natural_bonus_spawn": False,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        }
    )

    env.reset(seed=91)
    timestep = env.step(1)

    assert timestep.done is False
    assert timestep.reward == 1.0
    assert timestep.info["reward_schema_id"] == (
        SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_ID
    )
    assert timestep.info["reward_perspective"] == (
        "ego_player_dense_survival_plus_same_step_bonus_no_outcome"
    )
    assert timestep.info["dense_survival_helper_for_ego"] == 1.0
    assert timestep.info["bonus_catch_count_step_for_ego"] == 0
    assert timestep.info["bonus_pickup_reward_for_ego"] == 0.0
    assert timestep.info["sparse_outcome_reward_for_ego"] == 0.0
    assert timestep.info["terminal_outcome_bonus"] == 0.0
    assert timestep.info["winner_bonus"] == 0.0
    assert timestep.info["loser_penalty"] == 0.0


def test_source_state_survival_plus_bonus_no_outcome_rewards_same_step_bonus():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 92,
            "source_max_steps": 4,
            "natural_bonus_spawn": False,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        }
    )

    env.reset(seed=92)
    ego = env.ego_player_index
    env._env.seed_active_bonus(
        row=0,
        bonus_type="BonusSelfSmall",
        x=float(env._env.state["pos"][0, ego, 0]),
        y=float(env._env.state["pos"][0, ego, 1]),
        radius=20.0,
    )

    catch_timestep = env.step(1)
    next_timestep = env.step(1)

    assert catch_timestep.done is False
    assert catch_timestep.info["bonus_catch_count_step_for_ego"] == 1
    assert catch_timestep.info["bonus_pickup_reward_per_catch"] == (
        SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD
    )
    assert catch_timestep.info["bonus_pickup_reward_for_ego"] == (
        SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD
    )
    assert catch_timestep.info["dense_survival_helper_for_ego"] == 1.0
    assert catch_timestep.reward == (
        1.0 + SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD
    )
    assert next_timestep.info["bonus_catch_count_step_for_ego"] == 0
    assert next_timestep.info["bonus_pickup_reward_for_ego"] == 0.0
    assert next_timestep.reward == 1.0


def test_source_state_survival_plus_bonus_no_outcome_excludes_terminal_outcome():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 93,
            "source_max_steps": 64,
            "natural_bonus_spawn": False,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        }
    )

    env.reset(seed=93)
    ego = env.ego_player_index
    ego_key = f"player_{ego}"
    env._env.state["pos"][0, ego] = np.array([-1.0, -1.0], dtype=np.float64)
    timestep = env.step(1)

    assert timestep.done is True
    assert timestep.reward == 0.0
    assert timestep.info["trainer_reward"] == 0.0
    assert timestep.info["dense_survival_helper_for_ego"] == 0.0
    assert timestep.info["bonus_pickup_reward_for_ego"] == 0.0
    assert timestep.info["sparse_outcome_reward_for_ego"] == -1.0
    assert timestep.info["terminal_outcome_bonus"] == 0.0
    assert timestep.info["final_step_training_reward_map"][ego_key] == 0.0
    assert timestep.info["source_terminal_reward_map"][ego_key] == -1.0
    assert timestep.info["final_reward_map"][ego_key] == -1.0


def test_source_state_survival_plus_bonus_plus_outcome_keeps_survival_without_catch():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 94,
            "source_max_steps": 4,
            "natural_bonus_spawn": False,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
        }
    )

    env.reset(seed=94)
    timestep = env.step(1)

    assert timestep.done is False
    assert timestep.reward == 1.0
    assert timestep.info["reward_schema_id"] == (
        SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA_ID
    )
    assert timestep.info["reward_perspective"] == (
        "ego_player_dense_survival_plus_same_step_bonus_plus_scaled_outcome"
    )
    assert timestep.info["dense_survival_helper_for_ego"] == 1.0
    assert timestep.info["bonus_catch_count_step_for_ego"] == 0
    assert timestep.info["bonus_pickup_reward_for_ego"] == 0.0
    assert timestep.info["terminal_outcome_reward_for_ego"] == 0.0
    assert timestep.info["sparse_outcome_reward_for_ego"] == 0.0
    assert timestep.info["terminal_outcome_bonus"] == 1.0
    assert timestep.info["terminal_outcome_scale"] == "accumulated_non_outcome_training_reward"
    assert timestep.info["winner_bonus"] == 1.0
    assert timestep.info["loser_penalty"] == -1.0


def test_source_state_survival_plus_bonus_plus_outcome_rewards_same_step_bonus():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 95,
            "source_max_steps": 4,
            "natural_bonus_spawn": False,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
        }
    )

    env.reset(seed=95)
    ego = env.ego_player_index
    env._env.seed_active_bonus(
        row=0,
        bonus_type="BonusSelfSmall",
        x=float(env._env.state["pos"][0, ego, 0]),
        y=float(env._env.state["pos"][0, ego, 1]),
        radius=20.0,
    )

    timestep = env.step(1)

    assert timestep.done is False
    assert timestep.info["bonus_catch_count_step_for_ego"] == 1
    assert timestep.info["bonus_pickup_reward_per_catch"] == (
        SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD
    )
    assert timestep.info["bonus_pickup_reward_for_ego"] == (
        SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD
    )
    assert timestep.info["terminal_outcome_reward_for_ego"] == 0.0
    assert timestep.info["terminal_outcome_reward_enabled"] is True
    assert timestep.info["terminal_outcome_scale"] == "accumulated_non_outcome_training_reward"
    assert timestep.reward == (
        1.0 + SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD
    )


def test_source_state_survival_plus_bonus_plus_outcome_scales_terminal_loss():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 95,
            "source_max_steps": 64,
            "natural_bonus_spawn": False,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
        }
    )

    env.reset(seed=95)
    first = env.step(1)
    ego = env.ego_player_index
    ego_key = f"player_{ego}"
    env._env.state["pos"][0, ego] = np.array([-1.0, -1.0], dtype=np.float64)
    terminal = env.step(1)

    assert first.reward == 1.0
    assert terminal.done is True
    assert terminal.info["source_tick_index"] == 2
    assert terminal.info["dense_survival_helper_for_ego"] == 0.0
    assert terminal.info["bonus_pickup_reward_for_ego"] == 0.0
    assert terminal.info["sparse_outcome_reward_for_ego"] == -1.0
    assert terminal.info["terminal_outcome_reward_for_ego"] == -1.0
    assert terminal.reward == -1.0
    assert terminal.info["trainer_reward"] == -1.0
    assert terminal.info["episode_training_return"] == 0.0
    assert terminal.info["final_step_training_reward_map"][ego_key] == -1.0
    assert terminal.info["source_terminal_reward_map"][ego_key] == -1.0


def test_source_state_survival_plus_bonus_plus_outcome_allows_weaker_terminal_outcome():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 96,
            "source_max_steps": 64,
            "natural_bonus_spawn": False,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
            "reward_outcome_alpha": 0.5,
        }
    )

    env.reset(seed=96)
    first = env.step(1)
    ego = env.ego_player_index
    ego_key = f"player_{ego}"
    env._env.state["pos"][0, ego] = np.array([-1.0, -1.0], dtype=np.float64)
    terminal = env.step(1)

    assert first.reward == 1.0
    assert first.info["reward_outcome_alpha"] == 0.5
    assert first.info["terminal_outcome_bonus"] == 0.5
    assert terminal.done is True
    assert terminal.info["source_tick_index"] == 2
    assert terminal.info["sparse_outcome_reward_for_ego"] == -1.0
    assert terminal.info["terminal_outcome_reward_for_ego"] == -0.5
    assert terminal.reward == -0.5
    assert terminal.info["trainer_reward"] == -0.5
    assert terminal.info["episode_training_return"] == 0.5
    assert terminal.info["final_step_training_reward_map"][ego_key] == -0.5
    assert terminal.info["source_terminal_reward_map"][ego_key] == -1.0


def test_source_state_visual_survival_allows_explicit_bonus_render_mode_metadata():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 24,
            "source_max_steps": 1,
            "source_state_bonus_render_mode": BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
        }
    )

    env.reset(seed=24)
    timestep = env.step(1)

    assert (
        env.last_reset_info["source_state_bonus_render_mode"]
        == BONUS_RENDER_MODE_SIMPLE_SYMBOLS
    )
    assert (
        env.last_reset_info["model_observation_bonus_render_mode"]
        == BONUS_RENDER_MODE_SIMPLE_SYMBOLS
    )
    assert (
        env.last_reset_info["raw_observation_bonus_render_mode"]
        == BONUS_RENDER_MODE_SIMPLE_SYMBOLS
    )
    assert env.last_reset_info["bonus_render_mode"] == BONUS_RENDER_MODE_SIMPLE_SYMBOLS
    assert env.last_reset_info["bonus_renderer_kind"] == "simple_symbol_masks"
    assert env.last_reset_info["bonus_renderer_is_approximation"] is True
    assert timestep.info["bonus_render_mode"] == BONUS_RENDER_MODE_SIMPLE_SYMBOLS
    assert timestep.info["final_observation"] is not None


def test_source_state_visual_survival_rejects_unknown_trail_mode():
    with pytest.raises(ValueError, match="source_state_trail_render_mode"):
        CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
            {"source_state_trail_render_mode": "mystery_trails"}
        )


def test_source_state_visual_survival_rejects_unknown_bonus_mode():
    with pytest.raises(ValueError, match="source_state_bonus_render_mode"):
        CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
            {"source_state_bonus_render_mode": "mystery_bonus"}
        )


def test_source_state_visual_survival_opponent_immortal_still_allows_ego_wall_death():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 70,
            "source_max_steps": 64,
            "natural_bonus_spawn": False,
            "opponent_death_mode": OPPONENT_DEATH_MODE_IMMORTAL,
        }
    )

    env.reset(seed=70)
    ego = env.ego_player_index
    opponent = env.opponent_player_index
    env._env.state["pos"][0, ego] = np.array([-1.0, -1.0], dtype=np.float64)
    env._env.state["pos"][0, opponent] = np.array([-1.0, -1.0], dtype=np.float64)
    timestep = env.step(1)

    assert timestep.done is True
    assert timestep.info["death_mode"] == "normal"
    assert timestep.info["opponent_death_mode"] == OPPONENT_DEATH_MODE_IMMORTAL
    assert timestep.info["opponent_death_mode_diagnostic"] is True
    assert timestep.info["opponent_death_mode_claim"] == (
        "diagnostic_opponent_immortal_not_source_faithful"
    )
    assert bool(env._env.state["alive"][0, ego]) is False
    assert bool(env._env.state["alive"][0, opponent]) is True
    assert timestep.info["death_player_ids"] == (f"player_{ego}",)
    assert timestep.info["death_cause_name"][0][0] == "wall"


def test_source_state_visual_survival_opponent_immortal_blocks_opponent_wall_death():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 71,
            "source_max_steps": 64,
            "natural_bonus_spawn": False,
            "opponent_death_mode": OPPONENT_DEATH_MODE_IMMORTAL,
        }
    )

    env.reset(seed=71)
    ego = env.ego_player_index
    opponent = env.opponent_player_index
    env._env.state["pos"][0, opponent] = np.array([-1.0, -1.0], dtype=np.float64)
    timestep = env.step(1)

    assert timestep.done is False
    assert timestep.info["death_mode"] == "normal"
    assert timestep.info["opponent_death_mode"] == OPPONENT_DEATH_MODE_IMMORTAL
    assert bool(env._env.state["alive"][0, ego]) is True
    assert bool(env._env.state["alive"][0, opponent]) is True
    assert timestep.info["death_player_ids"] == ()


def test_source_state_visual_survival_opponent_immortal_blocks_opponent_body_death():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 72,
            "source_max_steps": 64,
            "natural_bonus_spawn": False,
            "opponent_death_mode": OPPONENT_DEATH_MODE_IMMORTAL,
        }
    )

    env.reset(seed=72)
    ego = env.ego_player_index
    opponent = env.opponent_player_index
    opponent_pos = env._env.state["pos"][0, opponent].copy()
    env._env.state["body_active"][0, 0] = True
    env._env.state["body_pos"][0, 0] = opponent_pos
    env._env.state["body_radius"][0, 0] = 8.0
    env._env.state["body_owner"][0, 0] = ego
    env._env.state["body_num"][0, 0] = 0
    env._env.state["body_insert_tick"][0, 0] = 0
    env._env.state["body_insert_kind"][0, 0] = vector_runtime.BODY_KIND_NORMAL
    env._env.state["body_write_cursor"][0] = 1
    env._env.state["world_body_count"][0] = 1
    env._env.state["body_count"][0, ego] = 1

    timestep = env.step(1)

    assert timestep.done is False
    assert timestep.info["opponent_death_mode"] == OPPONENT_DEATH_MODE_IMMORTAL
    assert bool(env._env.state["alive"][0, ego]) is True
    assert bool(env._env.state["alive"][0, opponent]) is True
    assert timestep.info["death_player_ids"] == ()
    assert env._last_batch is not None
    assert env._last_batch.info["step_counters"]["body_hits"] >= 1


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


def test_source_state_visual_survival_action_repeat_is_one_policy_transition():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 37,
            "source_max_steps": 32,
            "natural_bonus_spawn": False,
            "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
            "policy_action_repeat_min": 3,
            "policy_action_repeat_max": 3,
            "policy_action_repeat_extra_probability": 0.0,
        }
    )

    env.reset(seed=37)
    start_tick = int(env._env.state["tick"][0])
    timestep = env.step(1)

    assert timestep.done is False
    assert timestep.info["step_index"] == 0
    assert timestep.info["physical_step_index"] == 3
    assert timestep.info["source_tick_index"] == 3
    assert timestep.info["decision_source_frames"] == 1
    assert timestep.info["decision_ms"] == pytest.approx(SOURCE_PHYSICS_STEP_MS)
    assert timestep.info["policy_action_repeat_min"] == 3
    assert timestep.info["policy_action_repeat_max"] == 3
    assert timestep.info["policy_action_repeat_requested"] == 3
    assert timestep.info["policy_action_repeat_executed"] == 3
    assert timestep.info["policy_action_repeat_extra_steps"] == 2
    assert timestep.info["policy_observation_after_skipped_steps"] == 2
    assert timestep.info["physical_decision_ms_total"] == pytest.approx(
        3.0 * SOURCE_PHYSICS_STEP_MS
    )
    assert timestep.info["control_noise_profile_id"] == "policy_action_repeat"
    assert timestep.info["trainer_reward"] == 3.0
    assert timestep.info["dense_survival_helper_for_ego"] == 3.0
    assert timestep.info["bonus_catch_count_step_for_ego"] == 0
    assert int(env._env.state["tick"][0]) - start_tick == 3
    assert env._last_batch is not None
    assert int(env._last_batch.info["source_physics_substeps_executed"][0]) == 1


def test_source_state_visual_survival_default_is_one_source_frame_per_policy_action():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 38,
            "source_max_steps": 32,
            "natural_bonus_spawn": False,
            "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
        }
    )

    env.reset(seed=38)
    timestep = env.step(1)

    assert timestep.info["decision_source_frames"] == 1
    assert timestep.info["decision_ms"] == pytest.approx(SOURCE_PHYSICS_STEP_MS)
    assert timestep.info["policy_action_repeat_requested"] == 1
    assert timestep.info["policy_action_repeat_executed"] == 1
    assert timestep.info["policy_action_repeat_extra_steps"] == 0
    assert timestep.info["physical_step_index"] == 1
    assert env._last_batch is not None
    assert int(env._last_batch.info["source_physics_substeps_executed"][0]) == 1


def test_source_state_visual_survival_source_max_steps_caps_granular_source_ticks():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 39,
            "source_max_steps": 3,
            "natural_bonus_spawn": False,
            "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
        }
    )

    env.reset(seed=39)
    timestep = None
    step_count = 0
    while timestep is None or not timestep.done:
        timestep = env.step(1)
        step_count += 1
        assert step_count <= 4

    assert step_count == 3
    assert timestep.info["terminal_reason"] == "timeout"
    assert timestep.info["decision_source_frames"] == 1
    assert timestep.info["max_ticks"] == 3
    assert timestep.info["max_source_ticks"] == 3
    assert timestep.info["physical_step_index"] == 3
    assert timestep.info["source_tick_index"] == 3
    assert int(env._env.state["tick"][0]) == 3


def test_source_state_visual_survival_repeat_stops_at_source_max_steps_cap():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 40,
            "source_max_steps": 2,
            "natural_bonus_spawn": False,
            "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
            "policy_action_repeat_min": 3,
            "policy_action_repeat_max": 3,
            "policy_action_repeat_extra_probability": 0.0,
        }
    )

    env.reset(seed=40)
    timestep = env.step(1)

    assert timestep.done is True
    assert timestep.info["terminal_reason"] == "timeout"
    assert timestep.info["decision_source_frames"] == 1
    assert timestep.info["policy_action_repeat_requested"] == 3
    assert timestep.info["policy_action_repeat_executed"] == 2
    assert timestep.info["policy_action_repeat_extra_steps"] == 1
    assert timestep.info["physical_decision_ms_total"] == pytest.approx(
        2.0 * SOURCE_PHYSICS_STEP_MS
    )
    assert timestep.info["physical_step_index"] == 2
    assert timestep.info["source_tick_index"] == 2
    assert int(env._env.state["tick"][0]) == 2


def test_source_state_visual_survival_proactive_wall_avoidant_turns_from_left_wall():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 117,
            "source_max_steps": 64,
            "natural_bonus_spawn": False,
            "opponent_policy_kind": OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
        }
    )

    env.reset(seed=117)
    state = env._env.state
    map_size = float(state["map_size"][0])
    opponent = env.opponent_player_index
    opponent_radius = float(state["radius"][0, opponent])
    opponent_start = np.asarray(
        [opponent_radius + 5.0, map_size / 2.0],
        dtype=np.float64,
    )
    state["pos"][0, opponent] = opponent_start
    state["prev_pos"][0, opponent] = opponent_start
    state["heading"][0, opponent] = np.pi / 2.0
    ego_start = np.asarray([map_size * 0.75, map_size / 2.0], dtype=np.float64)
    state["pos"][0, env.ego_player_index] = ego_start
    state["prev_pos"][0, env.ego_player_index] = ego_start

    timestep = env.step(1)

    assert timestep.done is False
    assert timestep.info["opponent_policy_kind"] == (
        OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT
    )
    assert timestep.info["opponent_training_relation"] == (
        "learner_vs_proactive_wall_avoidant"
    )
    assert timestep.info["opponent_action_id"] == 0
    assert bool(state["alive"][0, opponent]) is True
    assert float(state["pos"][0, opponent, 0]) > float(opponent_start[0])
    sidecar = timestep.info["opponent_policy_sidecar"]
    metadata = sidecar["policy_metadata"]
    assert metadata["opponent_policy_variant"] == "proactive_force_field"
    assert metadata["nearest_wall_name"] == "left"
    assert metadata["selected_from_legal_actions"] is True
    assert metadata["uses_legal_left_straight_right_actions_only"] is True
    assert metadata["no_teleport_or_bounce"] is True


def test_blank_canvas_noop_still_ignores_proactive_wall_avoidant_policy():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 118,
            "source_max_steps": 64,
            "natural_bonus_spawn": False,
            "opponent_policy_kind": OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
            "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        }
    )

    env.reset(seed=118)
    timestep = env.step(1)

    assert timestep.info["blank_canvas_noop"] is True
    assert timestep.info["opponent_action_id"] == 1
    assert timestep.info["opponent_policy_sidecar"] == {
        "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
        "action_ignored": True,
    }


def test_opponent_mixture_selects_once_per_reset_not_per_step():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 119,
            "source_max_steps": 64,
            "natural_bonus_spawn": False,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
            "opponent_mixture": {
                "seed": 5,
                "entries": [
                    {
                        "name": "blank",
                        "weight": 1,
                        "opponent_policy_kind": "fixed_straight",
                        "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
                        "opponent_immortal": True,
                    }
                ],
            },
        }
    )

    env.reset(seed=119)
    first = env.step(1)
    second = env.step(2)

    assert first.info["opponent_mixture_enabled"] is True
    assert first.info["opponent_mixture_selection_unit"] == "episode_reset"
    assert first.info["opponent_mixture_entry_name"] == "blank"
    assert first.info["opponent_split_unit"] is None
    assert first.info["opponent_split_mode"] is None
    assert first.info["opponent_split_plan_sha256"] is None
    assert first.info["opponent_split_env_index"] is None
    assert first.info["opponent_split_env_num"] is None
    assert first.info["opponent_split_entry_name"] is None
    assert first.info["opponent_split_entry_count"] is None
    assert second.info["opponent_mixture_entry_name"] == "blank"
    assert first.info["opponent_runtime_mode"] == OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP
    assert second.info["opponent_runtime_mode"] == OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP
    assert first.info["opponent_policy_sidecar"]["action_ignored"] is True
    assert second.info["opponent_policy_sidecar"]["action_ignored"] is True


def test_opponent_mixture_refresh_applies_on_reset_and_records_assignment_context():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 119,
            "source_max_steps": 64,
            "natural_bonus_spawn": False,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
            "opponent_mixture": {
                "seed": 5,
                "entries": [
                    {
                        "name": "blank",
                        "weight": 1,
                        "opponent_policy_kind": "fixed_straight",
                        "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
                        "opponent_immortal": True,
                    }
                ],
            },
            "opponent_assignment_context": {
                "assignment_id": "assignment-a",
                "assignment_ref": "training/a/assignment.json",
                "assignment_sha256": "a" * 64,
                "refresh_index": 0,
            },
        }
    )

    env.reset(seed=119)
    before = env.step(1)
    after_reset_obs = env.reset(
        seed=120,
        opponent_mixture={
            "seed": 6,
            "entries": [
                {
                    "name": "wall",
                    "weight": 1,
                    "opponent_policy_kind": OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
                    "opponent_immortal": True,
                }
            ],
        },
        opponent_assignment_context={
            "assignment_id": "assignment-b",
            "assignment_ref": "training/b/assignment.json",
            "assignment_sha256": "b" * 64,
            "refresh_index": 1,
        },
    )
    assert env.last_reset_info is not None
    assert env.last_reset_info["opponent_assignment_id"] == "assignment-b"
    assert env.last_reset_info["opponent_assignment_ref"] == "training/b/assignment.json"
    assert env.last_reset_info["opponent_assignment_sha256"] == "b" * 64
    assert env.last_reset_info["opponent_assignment_refresh_index"] == 1
    after = env.step(1)

    assert before.info["opponent_mixture_entry_name"] == "blank"
    assert before.info["opponent_runtime_mode"] == OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP
    assert before.info["opponent_assignment_id"] == "assignment-a"
    assert before.info["opponent_assignment_refresh_index"] == 0
    assert after_reset_obs["observation"].shape == STACKED_SOURCE_STATE_GRAY64_SHAPE
    assert after.info["opponent_mixture_entry_name"] == "wall"
    assert after.info["opponent_policy_kind"] == OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT
    assert after.info["opponent_death_mode"] == OPPONENT_DEATH_MODE_IMMORTAL
    assert after.info["opponent_assignment_id"] == "assignment-b"
    assert after.info["opponent_assignment_ref"] == "training/b/assignment.json"
    assert after.info["opponent_assignment_sha256"] == "b" * 64
    assert after.info["opponent_assignment_refresh_index"] == 1


def test_opponent_mixture_refresh_clears_loaded_frozen_slot_with_same_name(monkeypatch):
    built_paths = []

    class FakeFrozenOpponent:
        def __init__(self, checkpoint_path):
            self.checkpoint_path = checkpoint_path
            self.policy_id = f"fake:{checkpoint_path}"
            self.policy_version = "test"

    def fake_build(cfg, *, seed):
        del seed
        path = cfg["opponent_checkpoint_path"]
        built_paths.append(path)
        return FakeFrozenOpponent(path)

    monkeypatch.setattr(
        env_mod,
        "_build_source_state_frozen_lightzero_opponent_policy",
        fake_build,
    )
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 121,
            "source_max_steps": 64,
            "natural_bonus_spawn": False,
            "opponent_mixture": {
                "entries": [
                    {
                        "name": "slot_champion",
                        "weight": 1,
                        "opponent_policy_kind": "frozen_lightzero_checkpoint",
                        "opponent_checkpoint_path": "/tmp/champion-a.pth.tar",
                    }
                ]
            },
        }
    )

    env.reset(seed=121)
    first_policy = env.opponent_policy
    env.reset(
        seed=122,
        opponent_mixture={
            "entries": [
                {
                    "name": "slot_champion",
                    "weight": 1,
                    "opponent_policy_kind": "frozen_lightzero_checkpoint",
                    "opponent_checkpoint_path": "/tmp/champion-b.pth.tar",
                }
            ]
        },
    )
    second_policy = env.opponent_policy

    assert built_paths == ["/tmp/champion-a.pth.tar", "/tmp/champion-b.pth.tar"]
    assert first_policy is not second_policy
    assert first_policy.checkpoint_path == "/tmp/champion-a.pth.tar"
    assert second_policy.checkpoint_path == "/tmp/champion-b.pth.tar"
    assert list(env._opponent_policy_cache) == ["slot_champion"]


def test_opponent_mixture_supports_passive_immortal_dirty_control():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 120,
            "source_max_steps": 64,
            "natural_bonus_spawn": False,
            "opponent_mixture": {
                "entries": [
                    {
                        "name": "passive_immortal",
                        "weight": 1,
                        "opponent_policy_kind": "fixed_straight",
                        "opponent_immortal": True,
                    }
                ]
            },
        }
    )

    env.reset(seed=120)
    opponent = env.opponent_player_index
    env._env.state["pos"][0, opponent] = np.array([-1.0, -1.0], dtype=np.float64)
    timestep = env.step(1)

    assert timestep.done is False
    assert timestep.info["opponent_mixture_entry_name"] == "passive_immortal"
    assert timestep.info["opponent_policy_kind"] == "fixed_straight"
    assert timestep.info["opponent_death_mode"] == OPPONENT_DEATH_MODE_IMMORTAL
    assert bool(env._env.state["alive"][0, opponent]) is True


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


def test_blank_canvas_noop_reset_hides_player_1_but_keeps_public_lifecycle():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 101,
            "source_max_steps": 32,
            "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
            "natural_bonus_spawn": False,
        }
    )

    observation = env.reset(seed=101)

    opponent = env.opponent_player_index
    assert bool(env._env.state["present"][0, opponent])
    assert bool(env._env.state["alive"][0, opponent])
    assert env.last_reset_info["opponent_runtime_mode"] == "blank_canvas_noop"
    assert env.last_reset_info["blank_canvas_noop"] is True
    assert env.last_reset_info["blank_canvas_noop_uses_remove_player"] is False
    assert env.last_reset_info["blank_canvas_noop_public_player_1_present_alive"] is True
    assert observation["observation"].shape == STACKED_SOURCE_STATE_GRAY64_SHAPE
    assert float(observation["observation"][-1].max()) > 0.0
    assert not bool(env._render_state_view()["present"][0, env.opponent_player_index])
    np.testing.assert_array_equal(
        env.render("source_state_grayscale64_visual_tensor"),
        render_source_state_canvas_gray64(
            env._render_state_view(),
            row=0,
            player_rgb=_ego_player_rgb(env),
            trail_render_mode=env._source_state_trail_render_mode,
            bonus_render_mode=env._raw_observation_bonus_render_mode,
        ),
    )


def test_blank_canvas_noop_steps_without_player_1_artifacts_or_terminal():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 103,
            "source_max_steps": 128,
            "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
            "natural_bonus_spawn": False,
        }
    )
    env.reset(seed=103)
    opponent = env.opponent_player_index
    env._env.seed_active_bonus(
        row=0,
        bonus_type="BonusSelfSmall",
        x=float(env._env.state["pos"][0, opponent, 0]),
        y=float(env._env.state["pos"][0, opponent, 1]),
        radius=20.0,
    )

    last = None
    for _ in range(8):
        last = env.step(1)
        assert not last.done

    assert last is not None
    assert bool(env._env.state["present"][0, opponent])
    assert bool(env._env.state["alive"][0, opponent])
    assert not _owner_active(env._env.state, "body", opponent)
    assert not _owner_active(env._env.state, "visual_trail", opponent)
    assert int(env._env.state["bonus_catch_count_step"][0, opponent]) == 0
    deaths = env._env.state["death_player"][0, : int(env._env.state["death_count"][0])]
    assert opponent not in deaths
    assert last.info["opponent_runtime_mode"] == "blank_canvas_noop"
    assert last.info["sparse_outcome_reward_for_ego"] == 0.0


def test_source_state_scalar_dirty_render_cache_matches_full_renderer_after_steps():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 111,
            "source_max_steps": 128,
            "death_mode": vector_runtime.DEATH_MODE_PROFILE_NO_DEATH,
            "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
            "natural_bonus_spawn": False,
            "source_state_trail_render_mode": SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES,
        }
    )
    env.reset(seed=111)

    for _ in range(12):
        timestep = env.step(1)
        assert not timestep.done

    state_view = env._render_state_view()
    expected_rgb = render_source_state_rgb_canvas_like(
        state_view,
        row=0,
        trail_render_mode=SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES,
        bonus_render_mode=env._raw_observation_bonus_render_mode,
    )
    expected_gray64 = render_source_state_canvas_gray64(
        state_view,
        row=0,
        player_rgb=_ego_player_rgb(env),
        trail_render_mode=SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES,
        bonus_render_mode=env._raw_observation_bonus_render_mode,
    )

    np.testing.assert_array_equal(env.raw_observation(), expected_rgb)
    np.testing.assert_array_equal(
        env.render("source_state_grayscale64_visual_tensor"),
        expected_gray64,
    )
    np.testing.assert_allclose(
        timestep.obs["observation"][-1],
        expected_gray64[0].astype(np.float32) / np.float32(255.0),
        rtol=0.0,
        atol=1e-7,
    )
    assert env._scalar_dirty_render_cache.stats.hits > 0
    assert env._scalar_dirty_render_cache.stats.fallbacks == 0


def test_blank_canvas_noop_scrubs_seeded_player_1_body_before_collision():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 107,
            "source_max_steps": 64,
            "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
            "natural_bonus_spawn": False,
        }
    )
    env.reset(seed=107)
    state = env._env.state
    ego = env.ego_player_index
    opponent = env.opponent_player_index
    state["body_active"][0, 0] = True
    state["body_owner"][0, 0] = opponent
    state["body_pos"][0, 0] = state["pos"][0, ego]
    state["body_radius"][0, 0] = 30.0
    state["body_count"][0, opponent] = 1
    state["world_body_count"][0] = int(state["body_active"][0].sum())

    timestep = env.step(1)

    assert not timestep.done
    assert bool(state["alive"][0, ego])
    assert not _owner_active(state, "body", opponent)


def test_blank_canvas_noop_player_0_wall_death_still_works():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 109,
            "source_max_steps": 64,
            "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
            "natural_bonus_spawn": False,
        }
    )
    env.reset(seed=109)
    ego = env.ego_player_index
    opponent = env.opponent_player_index
    env._env.state["pos"][0, ego] = np.asarray([-10.0, -10.0], dtype=np.float64)

    timestep = env.step(1)

    assert timestep.done
    assert not bool(env._env.state["alive"][0, ego])
    assert bool(env._env.state["alive"][0, opponent])
    deaths = env._env.state["death_player"][0, : int(env._env.state["death_count"][0])]
    assert opponent not in deaths
    assert timestep.reward == 0.0


def _owner_active(state: dict[str, np.ndarray], prefix: str, owner: int) -> bool:
    active = state[f"{prefix}_active"]
    owners = state[f"{prefix}_owner"]
    return bool((active & (owners == owner)).any())


def _space_shape(space):
    if isinstance(space, dict):
        return space["shape"]
    return space.shape


def _space_action_count(space):
    if isinstance(space, dict):
        return space["n"]
    return space.n
