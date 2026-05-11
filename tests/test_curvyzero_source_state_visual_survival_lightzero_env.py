import json

import numpy as np

from curvyzero.env import vector_runtime
from curvyzero.env.trainer_contract import ACTION_ID_TO_SOURCE_MOVE
from curvyzero.env.trainer_contract import ACTION_SPACE_ID
from curvyzero.env.vector_multiplayer_env import JOINT_ACTION_SCHEMA_ID
from curvyzero.env.vector_multiplayer_env import NATURAL_BONUS_EFFECT_TYPE_NAMES
from curvyzero.env.vector_multiplayer_env import NATURAL_BONUS_ENV_IMPL_ID
from curvyzero.env.vector_multiplayer_env import PUBLIC_NATURAL_BONUS_ENV_CONTRACT_ID
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.env.vector_visual_observation import SOURCE_STATE_GRAY64_RENDERER_IMPL_ID
from curvyzero.env.vector_visual_observation import SOURCE_STATE_GRAY64_SCHEMA_HASH
from curvyzero.env.vector_visual_observation import SOURCE_STATE_GRAY64_SCHEMA_ID
from curvyzero.env.vector_visual_observation import SourceStateGray64Renderer
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    CurvyZeroSourceStateVisualSurvivalLightZeroEnv,
    CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv,
    LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_ID,
    LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_TYPE,
    LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_IMPORT_NAMES,
    SOURCE_STATE_FIXED_OPPONENT_ENV_VARIANT,
    SOURCE_STATE_FIXED_OPPONENT_RUNTIME_TOPOLOGY,
    SOURCE_STATE_FIXED_OPPONENT_TWO_SEAT_STATUS,
    SOURCE_STATE_FIXED_OPPONENT_UNDERLYING_ENV_CLASS,
    STACKED_SOURCE_STATE_GRAY64_SCHEMA_HASH,
    STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
    STACKED_SOURCE_STATE_GRAY64_SHAPE,
    _normalize_player_perspective,
    _player_perspective_lut,
)


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
    assert env.last_reset_info["raw_observation_available"] is True
    assert env.last_reset_info["raw_observation_dtype"] == "uint8"
    assert env.last_reset_info["debug_fidelity_only"] is False
    assert env.last_reset_info["source_fidelity_claim"] == "source_state_backed_non_browser_pixel"
    assert env.last_reset_info["visual_surface"] == "source_state_visual_tensor"
    assert env.last_reset_info["visual_truth_level"] == "source_state_backed_non_browser_pixel"
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
    assert raw.shape == (1, 64, 64)
    assert raw.dtype == np.uint8
    np.testing.assert_array_equal(raw, raw_from_render)
    np.testing.assert_array_equal(
        env.render("source_state_player_perspective_raw_visual_tensor"),
        perspective_raw,
    )
    np.testing.assert_allclose(
        observation["observation"][-1],
        perspective_raw[0].astype(np.float32) / np.float32(255.0),
        rtol=0.0,
        atol=1e-7,
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
    assert timestep.info["reward_perspective"] == "ego_player_survival_after_step"
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

    terminal_raw = env.raw_observation()
    terminal_raw_from_render = env.render("source_state_raw_visual_tensor")
    terminal_player_perspective_raw = env.raw_observation(player_perspective=True)
    assert terminal_raw is not None
    assert terminal_raw_from_render is not None
    assert terminal_player_perspective_raw is not None
    assert terminal_raw.shape == (1, 64, 64)
    assert terminal_raw.dtype == np.uint8
    np.testing.assert_array_equal(terminal_raw, terminal_raw_from_render)
    np.testing.assert_array_equal(
        terminal_raw,
        SourceStateGray64Renderer(validate_state=False).render(env._env.state, row=0),
    )
    np.testing.assert_allclose(
        timestep.info["final_observation"]["observation"][-1],
        terminal_player_perspective_raw[0].astype(np.float32) / np.float32(255.0),
        rtol=0.0,
        atol=1e-7,
    )

    assert timestep.info["single_frame_schema_id"] == SOURCE_STATE_GRAY64_SCHEMA_ID
    assert timestep.info["single_frame_schema_hash"] == SOURCE_STATE_GRAY64_SCHEMA_HASH
    assert timestep.info["raw_observation_schema_id"] == SOURCE_STATE_GRAY64_SCHEMA_ID
    assert timestep.info["raw_observation_schema_hash"] == SOURCE_STATE_GRAY64_SCHEMA_HASH
    assert timestep.info["raw_observation_available"] is True
    assert timestep.info["raw_observation_accessors"] == [
        "raw_observation()",
        "render('source_state_raw_visual_tensor')",
    ]
    assert timestep.info["raw_observation_dtype"] == "uint8"
    assert timestep.info["renderer_impl_id"] == SOURCE_STATE_GRAY64_RENDERER_IMPL_ID
    assert timestep.info["browser_pixel_fidelity"] is False
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
    assert rows[0]["visual_surface"] == "source_state_visual_tensor"
    assert rows[0]["visual_truth_level"] == "source_state_backed_non_browser_pixel"
    assert rows[0]["visual_source_state_backed"] is True
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


def test_source_state_fixed_opponent_config_names_non_self_play_runtime_contract():
    config = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv.config

    assert config["env_variant"] == SOURCE_STATE_FIXED_OPPONENT_ENV_VARIANT
    assert config["runtime_topology"] == SOURCE_STATE_FIXED_OPPONENT_RUNTIME_TOPOLOGY
    assert config["underlying_env_class"] == SOURCE_STATE_FIXED_OPPONENT_UNDERLYING_ENV_CLASS
    assert config["source_fidelity_claim"] == "source_state_backed_non_browser_pixel"
    assert config["two_seat_self_play"] is False
    assert config["two_seat_self_play_status"] == SOURCE_STATE_FIXED_OPPONENT_TWO_SEAT_STATUS
    assert config["fixed_opponent_is_two_seat_self_play"] is False


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
