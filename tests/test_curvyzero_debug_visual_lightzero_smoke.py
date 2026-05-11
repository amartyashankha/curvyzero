import numpy as np
import pytest

from curvyzero.training.curvyzero_debug_visual_lightzero_smoke import (
    ACTION_ID_TO_SOURCE_MOVE,
)
from curvyzero.training.curvyzero_debug_visual_lightzero_smoke import (
    OPPONENT_POLICY_ID,
)
from curvyzero.training.curvyzero_debug_visual_lightzero_smoke import (
    OPPONENT_POLICY_VERSION,
)
from curvyzero.training.curvyzero_debug_visual_lightzero_smoke import (
    CurvyZeroDebugVisualLightZeroLocalSmokeEnv,
)
from curvyzero.training.curvyzero_debug_visual_lightzero_smoke import (
    LocalDebugVisualLightZeroTimestep,
)
from curvyzero.training.curvyzero_debug_visual_lightzero_smoke import (
    optional_base_env_timestep_cls,
)
from curvyzero.training.curvyzero_debug_visual_lightzero_smoke import (
    to_base_env_timestep,
)
from curvyzero.training.curvytron_visual_observation import DEBUG_OCCUPANCY_GRAY64_LABEL
from curvyzero.training.curvytron_visual_observation import DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID
from curvyzero.training.curvytron_visual_observation import DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH


class _RecordingBaseEnvTimestep:
    def __init__(self, obs, reward, done, info):
        self.obs = obs
        self.reward = reward
        self.done = done
        self.info = info


def test_reset_returns_debug_visual_lightzero_observation_and_metadata():
    env = CurvyZeroDebugVisualLightZeroLocalSmokeEnv({"seed": 17})

    observation = env.reset()

    assert set(observation) == {"observation", "action_mask", "to_play", "timestep"}
    assert observation["observation"].shape == (1, 64, 64)
    assert observation["observation"].dtype == np.float32
    assert float(observation["observation"].min()) >= 0.0
    assert float(observation["observation"].max()) <= 1.0
    assert int(np.count_nonzero(observation["observation"])) >= 1
    assert observation["action_mask"].dtype == np.int8
    np.testing.assert_array_equal(observation["action_mask"], np.array([1, 1, 1], dtype=np.int8))
    assert observation["to_play"] == -1
    assert observation["timestep"] == 0

    assert env.last_reset_info is not None
    assert env.last_reset_info["surface"] == "debug_visual_tensor"
    assert env.last_reset_info["observation_schema_id"] == DEBUG_OCCUPANCY_GRAY64_LABEL
    assert env.last_reset_info["observation_schema_hash"] == DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH
    assert env.last_reset_info["schema_hash"] == DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH
    assert env.last_reset_info["renderer_impl_id"] == DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID
    assert env.last_reset_info["truth_level"] == "debug_non_fidelity"
    assert env.last_reset_info["source_fidelity_level"] == "none"
    assert env.last_reset_info["source_backed_observation_fidelity"] is False
    assert env.last_reset_info["shape"] == [1, 64, 64]
    assert env.last_reset_info["dtype"] == "float32"
    assert env.last_reset_info["range"] == [0.0, 1.0]
    assert env.last_reset_info["uses_ale"] is False
    assert env.last_reset_info["ale_usage"] == "none"
    assert env.last_reset_info["frame_stack_owner"] == "optimizer"
    assert env.last_reset_info["opponent_policy_id"] == OPPONENT_POLICY_ID
    assert env.last_reset_info["opponent_policy_version"] == OPPONENT_POLICY_VERSION
    assert env.last_reset_info["source_game_started"] is True
    assert env.last_reset_info["source_world_active"] is True
    assert env.last_reset_info["source_reset_warmup_ms"] == 0.0
    assert env.last_reset_info["source_reset_advance_timers_ms"] == 0.0


def test_step_maps_turn3_actions_to_source_moves_and_keeps_opponent_straight():
    env = CurvyZeroDebugVisualLightZeroLocalSmokeEnv({"seed": 23})
    reset = env.reset()

    timestep = env.step(0)

    assert timestep.obs["observation"].shape == (1, 64, 64)
    assert timestep.obs["observation"].dtype == np.float32
    assert not np.array_equal(timestep.obs["observation"], reset["observation"])
    np.testing.assert_array_equal(timestep.obs["action_mask"], np.array([1, 1, 1], dtype=np.int8))
    assert timestep.reward == 0.0
    assert timestep.done is False
    assert timestep.obs["timestep"] == 1
    assert timestep.info["adapter_timestep"] == 1
    assert timestep.info["tick_index"] == 1
    assert timestep.info["joint_action"] == {"player_0": 0, "player_1": 1}
    assert timestep.info["joint_source_move"] == {"player_0": -1.0, "player_1": 0.0}
    assert timestep.info["opponent_action_id"] == 1
    assert timestep.info["opponent_policy_id"] == OPPONENT_POLICY_ID
    assert timestep.info["opponent_policy_version"] == OPPONENT_POLICY_VERSION
    assert timestep.info["source_step_pre_advance_timers_ms"] == 0.0
    assert timestep.info["source_step_ms"] > 0.0
    assert timestep.info["source_timer_clock_advances_on_step"] is False
    assert timestep.info["source_at_ms"] == 0
    assert len(timestep.info["trace_hash"]) == 16
    assert tuple(ACTION_ID_TO_SOURCE_MOVE) == (-1, 0, 1)
    assert timestep.info["death_count"] == [0]
    assert timestep.info["death_player"] == [[-1, -1]]
    assert timestep.info["death_cause_name"] == [["none", "none"]]
    assert timestep.info["death_hit_owner"] == [[-1, -1]]


def test_replayed_old_all_left_episode_reports_wall_death():
    env = CurvyZeroDebugVisualLightZeroLocalSmokeEnv(
        {"seed": 1297473639, "source_max_steps": 1024}
    )
    env.reset(seed=1297473639)

    timestep = None
    for _ in range(33):
        timestep = env.step(0)
        if timestep.done:
            break

    assert timestep is not None
    assert timestep.done is True
    assert timestep.info["terminal_reason"] == "survivor_win"
    assert timestep.info["winner_ids"] == ("player_1",)
    assert timestep.info["loser_ids"] == ("player_0",)
    assert timestep.info["death_player_ids"] == ("player_0",)
    assert timestep.info["death_count"] == [1]
    assert timestep.info["death_player"] == [[0, -1]]
    assert timestep.info["death_cause_name"] == [["wall", "none"]]
    assert timestep.info["death_hit_owner"] == [[-1, -1]]
    assert timestep.info["death_source_avatar"] == [[1, -1]]
    assert timestep.info["death_source_killer_avatar"] == [[-1, -1]]


def test_source_max_steps_truncates_with_final_observation_and_blocks_autoreset():
    env = CurvyZeroDebugVisualLightZeroLocalSmokeEnv({"seed": 9, "source_max_steps": 1})
    env.reset()

    timestep = env.step(2)

    assert timestep.done is True
    assert timestep.info["done"] is True
    assert timestep.info["terminated"] is False
    assert timestep.info["truncated"] is True
    assert timestep.info["timeout"] is True
    assert timestep.info["truncation_reason"] == "source_max_steps"
    assert timestep.info["eval_episode_return"] == 0.0
    assert timestep.info["needs_reset"] is True
    assert timestep.info["final_observation"]["observation"].shape == (1, 64, 64)
    np.testing.assert_array_equal(
        timestep.info["final_observation"]["action_mask"],
        np.array([0, 0, 0], dtype=np.int8),
    )

    with pytest.raises(RuntimeError, match="reset must be called"):
        env.step(1)


def test_same_seed_and_actions_are_deterministic_through_debug_visual_adapter():
    first = CurvyZeroDebugVisualLightZeroLocalSmokeEnv({"seed": 31})
    second = CurvyZeroDebugVisualLightZeroLocalSmokeEnv({"seed": 31})

    np.testing.assert_allclose(first.reset()["observation"], second.reset()["observation"])
    first_step = first.step(2)
    second_step = second.step(2)

    np.testing.assert_allclose(first_step.obs["observation"], second_step.obs["observation"])
    np.testing.assert_array_equal(first_step.obs["action_mask"], second_step.obs["action_mask"])
    assert first_step.info["joint_action"] == second_step.info["joint_action"]
    assert first_step.info["joint_source_move"] == second_step.info["joint_source_move"]
    assert first_step.info["trace_hash"] == second_step.info["trace_hash"]


def test_local_timestep_converts_through_explicit_base_env_timestep_boundary():
    timestep = LocalDebugVisualLightZeroTimestep(
        obs={"observation": np.array([1.0], dtype=np.float32)},
        reward=0.5,
        done=False,
        info={"step_index": 7},
    )

    converted = timestep.to_base_env_timestep(_RecordingBaseEnvTimestep)

    assert isinstance(converted, _RecordingBaseEnvTimestep)
    assert converted.obs is timestep.obs
    assert converted.reward == 0.5
    assert converted.done is False
    assert converted.info is timestep.info


def test_missing_real_base_env_timestep_import_has_clear_boundary():
    if optional_base_env_timestep_cls() is not None:
        pytest.skip("DI-engine BaseEnvTimestep is installed; real conversion test covers this")
    timestep = LocalDebugVisualLightZeroTimestep(obs={}, reward=0.0, done=False, info={})

    with pytest.raises(ImportError, match="ding.envs.BaseEnvTimestep is not available"):
        to_base_env_timestep(timestep)


def test_real_base_env_timestep_conversion_when_di_engine_is_available():
    base_env_timestep_cls = optional_base_env_timestep_cls()
    if base_env_timestep_cls is None:
        pytest.skip("DI-engine/LightZero runtime is not installed locally")
    env = CurvyZeroDebugVisualLightZeroLocalSmokeEnv()
    env.reset(seed=41)

    local_timestep = env.step(1)
    real_timestep = local_timestep.to_base_env_timestep(base_env_timestep_cls)

    assert isinstance(real_timestep, base_env_timestep_cls)
    np.testing.assert_allclose(
        real_timestep.obs["observation"],
        local_timestep.obs["observation"],
    )
    np.testing.assert_array_equal(
        real_timestep.obs["action_mask"],
        local_timestep.obs["action_mask"],
    )
    assert real_timestep.reward == local_timestep.reward
    assert real_timestep.done == local_timestep.done
    assert real_timestep.info is local_timestep.info
