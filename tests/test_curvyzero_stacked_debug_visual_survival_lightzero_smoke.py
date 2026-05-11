import json

import numpy as np

from curvyzero.training.curvytron_visual_observation import DEBUG_OCCUPANCY_GRAY64_LABEL
from curvyzero.training.curvytron_visual_observation import (
    DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_LABEL,
)
from curvyzero.training.curvytron_visual_observation import (
    DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_SCHEMA_HASH,
)
from curvyzero.training.curvytron_visual_observation import DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv,
    CurvyZeroStackedDebugVisualTurnCommitLightZeroLocalSmokeEnv,
    STACKED_DEBUG_VISUAL_SURVIVAL_FRAME_STACK_OWNER,
    STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_HASH,
    STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_ID,
    TURN_COMMIT_REWARD_CREDIT_CAVEAT,
    run_stacked_debug_visual_survival_smoke,
)


def test_stacked_debug_visual_reset_shape_dtype_and_policy_metadata():
    env = CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv({"seed": 13})

    observation = env.reset(seed=13)

    assert observation["observation"].shape == (4, 64, 64)
    assert observation["observation"].dtype == np.float32
    assert float(observation["observation"].min()) >= 0.0
    assert float(observation["observation"].max()) <= 1.0
    np.testing.assert_array_equal(observation["observation"][:3], np.zeros((3, 64, 64)))
    assert int(np.count_nonzero(observation["observation"][-1])) >= 1
    assert observation["action_mask"].dtype == np.int8
    np.testing.assert_array_equal(observation["action_mask"], np.array([1, 1, 1], dtype=np.int8))
    assert observation["to_play"] == -1

    assert env.last_reset_info is not None
    assert env.last_reset_info["observation_schema_id"] == (
        STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_ID
    )
    assert env.last_reset_info["observation_schema_hash"] == (
        STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_HASH
    )
    assert env.last_reset_info["schema_hash"] == STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_HASH
    assert env.last_reset_info["raw_observation_schema_id"] == (
        DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_LABEL
    )
    assert env.last_reset_info["raw_observation_schema_hash"] == (
        DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_SCHEMA_HASH
    )
    assert env.last_reset_info["legacy_anonymous_raw_observation_schema_id"] == (
        DEBUG_OCCUPANCY_GRAY64_LABEL
    )
    assert env.last_reset_info["renderer_impl_id"] == DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID
    assert env.last_reset_info["truth_level"] == "debug_non_fidelity"
    assert env.last_reset_info["source_fidelity_level"] == "none"
    assert env.last_reset_info["shape"] == [4, 64, 64]
    assert env.last_reset_info["dtype"] == "float32"
    assert env.last_reset_info["range"] == [0.0, 1.0]
    assert env.last_reset_info["raw_frame_shape"] == [1, 64, 64]
    assert env.last_reset_info["lightzero_payload_shape"] == [4, 64, 64]
    assert env.last_reset_info["model_observation_shape"] == [4, 64, 64]
    assert env.last_reset_info["frame_stack_owner"] == (
        STACKED_DEBUG_VISUAL_SURVIVAL_FRAME_STACK_OWNER
    )
    assert env.last_reset_info["frame_stack_proof"] == (
        "wrapper_owned_fifo_stack; not LightZero env-manager stacking"
    )
    assert env.last_reset_info["debug_fidelity_only"] is True
    assert env.last_reset_info["source_fidelity_claim"] == "none"
    assert env.last_reset_info["source_fidelity_level"] == "none"
    assert env.last_reset_info["browser_pixel_fidelity"] is False
    assert env.last_reset_info["uses_ale"] is False
    assert env.last_reset_info["ale_usage"] == "none"


def test_stacked_debug_visual_step_shifts_previous_frame_and_final_observation():
    env = CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv(
        {"seed": 21, "source_max_steps": 1}
    )
    reset_observation = env.reset(seed=21)

    timestep = env.step(1)

    assert timestep.done is True
    assert timestep.obs["observation"].shape == (4, 64, 64)
    assert timestep.obs["observation"].dtype == np.float32
    np.testing.assert_array_equal(
        timestep.obs["observation"][-2],
        reset_observation["observation"][-1],
    )
    assert timestep.info["final_observation"]["observation"].shape == (4, 64, 64)
    assert timestep.info["final_observation"]["observation"].dtype == np.float32
    np.testing.assert_array_equal(
        timestep.info["final_observation"]["action_mask"],
        np.array([0, 0, 0], dtype=np.int8),
    )
    assert timestep.info["frame_stack_owner"] == STACKED_DEBUG_VISUAL_SURVIVAL_FRAME_STACK_OWNER
    assert timestep.info["frame_stack_proof"] == (
        "wrapper_owned_fifo_stack; not LightZero env-manager stacking"
    )
    assert timestep.info["source_fidelity_claim"] == "none"


def test_stacked_debug_visual_telemetry_records_terminal_death_fields(tmp_path):
    telemetry_path = tmp_path / "steps.jsonl"
    env = CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv(
        {
            "seed": 1297473639,
            "source_max_steps": 1024,
            "telemetry_path": telemetry_path,
        }
    )
    env.reset(seed=1297473639)

    timestep = None
    for _ in range(33):
        timestep = env.step(0)
        if timestep.done:
            break

    assert timestep is not None
    assert timestep.done is True
    rows = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert rows[-1]["terminal_reason"] == "survivor_win"
    assert rows[-1]["winner_ids"] == ["player_1"]
    assert rows[-1]["loser_ids"] == ["player_0"]
    assert rows[-1]["death_player_ids"] == ["player_0"]
    assert rows[-1]["death_count"] == [1]
    assert rows[-1]["death_player"] == [[0, -1]]
    assert rows[-1]["death_cause_name"] == [["wall", "none"]]
    assert rows[-1]["death_hit_owner"] == [[-1, -1]]
    assert rows[-1]["trace_hash"] == "30c77a4dedae3f35"


def test_stacked_debug_visual_collect_smoke_reports_shape_without_training():
    result = run_stacked_debug_visual_survival_smoke(seed=2, steps=2)

    assert result["ok"] is True
    assert result["call_policy"] == "does_not_train; does_not_call_lzero_entrypoints"
    assert result["observation_schema_id"] == STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_ID
    assert result["observation_schema_hash"] == STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_HASH
    assert result["raw_observation_schema_id"] == DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_LABEL
    assert result["raw_observation_schema_hash"] == DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_SCHEMA_HASH
    assert result["legacy_anonymous_raw_observation_schema_id"] == DEBUG_OCCUPANCY_GRAY64_LABEL
    assert result["renderer_impl_id"] == DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID
    assert result["truth_level"] == "debug_non_fidelity"
    assert result["source_fidelity_level"] == "none"
    assert result["observation_shape"] == [4, 64, 64]
    assert result["observation_dtype"] == "float32"
    assert result["value_range"] == [0.0, 1.0]
    assert result["raw_frame_shape"] == [1, 64, 64]
    assert result["frame_stack_owner"] == STACKED_DEBUG_VISUAL_SURVIVAL_FRAME_STACK_OWNER
    assert result["uses_ale"] is False
    assert result["ale_usage"] == "none"
    assert result["mcts_search"]["status"] == "not_run"
    assert result["learner_profile"]["status"] == "not_run"


def test_turn_commit_first_step_stores_private_pending_action_without_physics_step():
    env = CurvyZeroStackedDebugVisualTurnCommitLightZeroLocalSmokeEnv(
        {"seed": 31, "source_max_steps": 10}
    )
    reset_observation = env.reset(seed=31)

    timestep = env.step(0)

    assert timestep.done is False
    assert timestep.reward == 0.0
    assert timestep.info["turn_commit_adapter"] is True
    assert timestep.info["physical_env_advanced"] is False
    assert timestep.info["acting_player_id"] == "player_0"
    assert timestep.info["active_player_id"] == "player_1"
    assert timestep.info["pending_action_count"] == 1
    assert timestep.info["pending_actions_private"] is True
    assert timestep.info["trusted_current_policy_self_play"] is False
    assert timestep.info["simultaneous_game_theory_claim"] is False
    assert timestep.info["scalar_steps_per_source_tick"] == 2
    assert timestep.info["reward_perspective"] == "just_controlled_player_after_commit"
    assert timestep.info["joint_action"] == {"player_0": 0}
    assert timestep.info["reward_credit_caveat"] == TURN_COMMIT_REWARD_CREDIT_CAVEAT
    assert env.active_player_id == "player_1"
    assert timestep.obs["observation"].shape == reset_observation["observation"].shape
    assert timestep.obs["timestep"] == 0


def test_turn_commit_second_step_advances_once_and_returns_player_zero_view():
    env = CurvyZeroStackedDebugVisualTurnCommitLightZeroLocalSmokeEnv(
        {"seed": 32, "source_max_steps": 10}
    )
    reset_observation = env.reset(seed=32)
    pending = env.step(0)

    committed = env.step(2)

    assert pending.info["physical_env_advanced"] is False
    assert committed.info["physical_env_advanced"] is True
    assert committed.info["acting_player_id"] == "player_1"
    assert committed.info["active_player_id"] == "player_0"
    assert committed.info["pending_action_count"] == 0
    assert committed.info["trusted_current_policy_self_play"] is False
    assert committed.info["simultaneous_game_theory_claim"] is False
    assert committed.info["joint_action"] == {"player_0": 0, "player_1": 2}
    assert committed.info["joint_source_move"] == {
        "player_0": -1.0,
        "player_1": 1.0,
    }
    assert committed.reward in {0.0, 1.0}
    assert committed.obs["timestep"] == 1
    assert env.active_player_id == "player_0"
    np.testing.assert_array_equal(
        committed.obs["observation"][-2],
        reset_observation["observation"][-1],
    )


def test_turn_commit_reset_clears_pending_state():
    env = CurvyZeroStackedDebugVisualTurnCommitLightZeroLocalSmokeEnv(
        {"seed": 33, "source_max_steps": 10}
    )
    env.reset(seed=33)
    env.step(0)

    observation = env.reset(seed=34)

    assert env.active_player_id == "player_0"
    assert observation["timestep"] == 0
    assert env.last_reset_info is not None
    assert env.last_reset_info["pending_action_count"] == 0
