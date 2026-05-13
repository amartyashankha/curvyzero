import numpy as np

from curvyzero.env.trainer_contract import ACTION_ID_TO_SOURCE_MOVE
from curvyzero.env.trainer_contract import ACTION_SPACE_ID
from curvyzero.env.vector_multiplayer_env import JOINT_ACTION_SCHEMA_ID
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.env.vector_visual_observation import render_source_state_canvas_gray64
from curvyzero.env.vector_visual_observation import render_source_state_rgb_canvas_like
from curvyzero.env.vector_visual_observation import rgb_canvas_like_to_gray64
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    CurvyZeroSourceStateVisualJointActionLightZeroLocalEnv,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_ID,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SHAPE,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    SOURCE_STATE_JOINT_ACTION_ENV_VARIANT,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    SOURCE_STATE_JOINT_ACTION_RUNTIME_TOPOLOGY,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    SOURCE_STATE_JOINT_ACTION_TRAINING_STATUS,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    STACKED_SOURCE_STATE_GRAY64_SHAPE,
)


SEED = 777
DECISION_SOURCE_FRAMES = 12
SOURCE_MAX_STEPS = 64
LEFT_RIGHT_SCALAR_ACTION = 2
STRAIGHT_STRAIGHT_SCALAR_ACTION = 4

UNSUPPORTED_REPLAY_TRAINER_ARRAY_GAP = (
    "source_state_lightzero_wrapper_exposes_lightzero_timestep_dicts_and_"
    "vector_batch_sidecars_but_not_multiplayer_trainer_replay_arrays/v0"
)


def test_lightzero_source_state_joint_action_wrapper_routes_controls_visuals_and_terminal():
    """Focused wrapper-route proof.

    Current unsupported replay/trainer-array gap:
    ``UNSUPPORTED_REPLAY_TRAINER_ARRAY_GAP``. The LightZero wrapper exposes
    final LightZero observations through ``timestep.info`` and the underlying
    public ``VectorMultiplayerEnv`` sidecars through ``env._last_batch.info``;
    it does not yet emit full multiplayer trainer replay arrays.
    """

    env = _new_joint_action_env()
    reset_observation = env.reset(seed=SEED)

    assert type(env._env) is VectorMultiplayerEnv
    assert env.env_id == LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_ID
    assert env.last_reset_info["underlying_env_class"] == "VectorMultiplayerEnv"
    assert env.last_reset_info["env_variant"] == SOURCE_STATE_JOINT_ACTION_ENV_VARIANT
    assert env.last_reset_info["runtime_topology"] == (
        SOURCE_STATE_JOINT_ACTION_RUNTIME_TOPOLOGY
    )
    assert env.last_reset_info["opponent_policy_kind"] == (
        OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION
    )
    assert env.last_reset_info["centralized_joint_action_control"] is True
    assert env.last_reset_info["two_seat_self_play_status"] == (
        SOURCE_STATE_JOINT_ACTION_TRAINING_STATUS
    )
    assert env.last_reset_info["source_frame_decision"] is True
    assert env.last_reset_info["decision_source_frames"] == DECISION_SOURCE_FRAMES
    assert env._env.source_frame_decision is True
    assert env._env.decision_source_frames == DECISION_SOURCE_FRAMES
    assert reset_observation["observation"].shape == STACKED_SOURCE_STATE_GRAY64_SHAPE
    np.testing.assert_array_equal(
        reset_observation["action_mask"],
        np.ones(9, dtype=np.int8),
    )

    _assert_raw_rgb_to_gray64_stack_path(env, reset_observation)

    before_tick = int(env._env.state["tick"][0])
    timestep = env.step(LEFT_RIGHT_SCALAR_ACTION)
    tick_delta = int(env._env.state["tick"][0]) - before_tick

    assert timestep.done is False
    assert timestep.info["joint_action_scalar"] == LEFT_RIGHT_SCALAR_ACTION
    assert timestep.info["joint_action_decode_rule"] == (
        "scalar // 3 -> player_0, scalar % 3 -> player_1"
    )
    assert timestep.info["joint_action"] == {"player_0": 0, "player_1": 2}
    assert timestep.info["joint_action_schema_id"] == JOINT_ACTION_SCHEMA_ID
    assert timestep.info["source_frame_decision"] is True
    assert timestep.info["decision_source_frames"] == DECISION_SOURCE_FRAMES
    assert timestep.info["physical_env_advanced"] is True
    assert tick_delta == DECISION_SOURCE_FRAMES
    np.testing.assert_allclose(
        timestep.obs["observation"][-2],
        reset_observation["observation"][-1],
        rtol=0.0,
        atol=0.0,
    )

    runtime_info = env._last_batch.info
    assert env._env.last_step_info is runtime_info
    _assert_native_control_sidecar(
        runtime_info,
        player_actions=np.array([[0, 2]], dtype=np.int16),
        native_controls=np.array(
            [[ACTION_ID_TO_SOURCE_MOVE[0], ACTION_ID_TO_SOURCE_MOVE[2]]],
            dtype=np.int8,
        ),
        substeps=DECISION_SOURCE_FRAMES,
    )
    np.testing.assert_allclose(
        runtime_info["source_physics_elapsed_ms"],
        np.array([env._env.source_physics_step_ms * DECISION_SOURCE_FRAMES]),
        rtol=0.0,
        atol=1e-6,
    )
    assert runtime_info["action_sidecar"]["metadata_only"] is True
    assert runtime_info["action_sidecar"]["joint_action_mcts_claim"] is False

    terminal_env = _new_joint_action_env()
    terminal_env.reset(seed=SEED)
    terminal_timestep = None
    terminal_delta = None
    step_count = 0
    while step_count < 80:
        before_tick = int(terminal_env._env.state["tick"][0])
        current = terminal_env.step(STRAIGHT_STRAIGHT_SCALAR_ACTION)
        terminal_delta = int(terminal_env._env.state["tick"][0]) - before_tick
        step_count += 1
        if current.done:
            terminal_timestep = current
            break
        assert terminal_delta == DECISION_SOURCE_FRAMES
        _assert_native_control_sidecar(
            terminal_env._last_batch.info,
            player_actions=np.array([[1, 1]], dtype=np.int16),
            native_controls=np.array([[0, 0]], dtype=np.int8),
            substeps=DECISION_SOURCE_FRAMES,
        )

    assert terminal_timestep is not None
    assert step_count == 15
    assert int(terminal_env._env.state["tick"][0]) == 177
    assert terminal_delta is not None
    assert 0 < terminal_delta < DECISION_SOURCE_FRAMES
    assert terminal_timestep.done is True
    assert terminal_timestep.reward == 0.0
    assert terminal_timestep.info["terminal_reason"] == "round_survivor_win"
    assert terminal_timestep.info["winner_ids"] == ("player_1",)
    assert terminal_timestep.info["loser_ids"] == ("player_0",)
    assert terminal_timestep.info["death_count"] == [1]
    assert terminal_timestep.info["death_player"] == [[0, -1]]
    assert terminal_timestep.info["source_terminal_reward_map"] == {
        "player_0": -1.0,
        "player_1": 1.0,
    }
    assert terminal_timestep.info["final_reward_map"] == {
        "player_0": -1.0,
        "player_1": 1.0,
    }
    assert terminal_timestep.info["final_step_training_reward_map"] == {
        "player_0": 0.0,
        "player_1": 0.0,
    }
    assert terminal_timestep.info["final_observation"] is not None
    assert (
        terminal_timestep.info["final_observation"]["observation"].shape
        == STACKED_SOURCE_STATE_GRAY64_SHAPE
    )
    np.testing.assert_array_equal(
        terminal_timestep.info["final_observation"]["action_mask"],
        np.zeros(9, dtype=np.int8),
    )
    np.testing.assert_allclose(
        terminal_timestep.info["final_observation"]["observation"],
        terminal_timestep.obs["observation"],
        rtol=0.0,
        atol=0.0,
    )

    terminal_runtime_info = terminal_env._last_batch.info
    _assert_native_control_sidecar(
        terminal_runtime_info,
        player_actions=np.array([[1, 1]], dtype=np.int16),
        native_controls=np.array([[0, 0]], dtype=np.int8),
        max_substeps=DECISION_SOURCE_FRAMES - 1,
    )
    np.testing.assert_array_equal(
        terminal_runtime_info["final_observation_row_mask"],
        np.array([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_runtime_info["final_reward_row_mask"],
        np.array([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_runtime_info["terminal_rows"],
        np.array([0], dtype=np.int32),
    )
    assert terminal_runtime_info["final_observation_row_policy"]["source_claim"] == (
        "debug_metadata_only_public_terminal_rows/v0"
    )
    np.testing.assert_array_equal(
        terminal_env._last_batch.reward,
        np.array([[-1.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        terminal_env._last_batch.final_reward,
        np.array([[-1.0, 1.0]], dtype=np.float32),
    )
    assert terminal_runtime_info["action_sidecar"]["metadata_only"] is True
    assert terminal_runtime_info["action_sidecar"]["joint_action_mcts_claim"] is False
    assert "trainer_replay_claim" not in terminal_timestep.info
    assert UNSUPPORTED_REPLAY_TRAINER_ARRAY_GAP.endswith("_arrays/v0")


def _new_joint_action_env() -> CurvyZeroSourceStateVisualJointActionLightZeroLocalEnv:
    return CurvyZeroSourceStateVisualJointActionLightZeroLocalEnv(
        {
            "seed": SEED,
            "source_max_steps": SOURCE_MAX_STEPS,
            "decision_source_frames": DECISION_SOURCE_FRAMES,
            "natural_bonus_spawn": False,
            "opponent_policy_kind": OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION,
        }
    )


def _assert_raw_rgb_to_gray64_stack_path(
    env: CurvyZeroSourceStateVisualJointActionLightZeroLocalEnv,
    observation: dict,
) -> None:
    raw = env.raw_observation()
    raw_from_render = env.render("source_state_rgb_canvas_like")
    assert raw is not None
    assert raw_from_render is not None
    assert raw.shape == SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SHAPE
    assert raw.dtype == np.uint8
    np.testing.assert_array_equal(raw, raw_from_render)
    np.testing.assert_array_equal(
        raw,
        render_source_state_rgb_canvas_like(env._env.state, row=0),
    )

    gray64_from_raw = rgb_canvas_like_to_gray64(raw)
    gray64_direct = render_source_state_canvas_gray64(env._env.state, row=0)
    np.testing.assert_array_equal(gray64_from_raw, gray64_direct)
    np.testing.assert_allclose(
        observation["observation"][-1],
        gray64_from_raw[0].astype(np.float32) / np.float32(255.0),
        rtol=0.0,
        atol=1e-7,
    )


def _assert_native_control_sidecar(
    runtime_info: dict,
    *,
    player_actions: np.ndarray,
    native_controls: np.ndarray,
    substeps: int | None = None,
    max_substeps: int | None = None,
) -> None:
    action_sidecar = runtime_info["action_sidecar"]
    assert runtime_info["source_frame_decision"] is True
    assert runtime_info["decision_source_frames"] == DECISION_SOURCE_FRAMES
    assert action_sidecar["action_space_id"] == ACTION_SPACE_ID
    assert action_sidecar["joint_action_schema_id"] == JOINT_ACTION_SCHEMA_ID
    np.testing.assert_array_equal(runtime_info["joint_action"], player_actions)
    np.testing.assert_array_equal(runtime_info["source_moves"], native_controls)
    np.testing.assert_array_equal(action_sidecar["player_action"], player_actions)
    np.testing.assert_array_equal(
        action_sidecar["player_action_mask"],
        np.ones((1, 2, 3), dtype=bool),
    )
    np.testing.assert_array_equal(
        action_sidecar["action_required"],
        np.ones((1, 2), dtype=bool),
    )
    np.testing.assert_array_equal(action_sidecar["native_control_value"], native_controls)
    np.testing.assert_array_equal(
        action_sidecar["action_source"],
        np.array([["external_joint_action", "external_joint_action"]], dtype=object),
    )
    executed = int(runtime_info["source_physics_substeps_executed"][0])
    if substeps is not None:
        assert executed == substeps
    if max_substeps is not None:
        assert 0 < executed <= max_substeps
