import numpy as np
import pytest

from curvyzero.env import CurvyTronConfig
from curvyzero.env import CurvyTronEnv
from curvyzero.env import trainer_contract as contract
from curvyzero.env.trainer_observation import observe_egocentric_rays_v0
from curvyzero.training.curvyzero_lightzero_smoke import OPPONENT_POLICY_ID
from curvyzero.training.curvyzero_lightzero_smoke import OPPONENT_POLICY_VERSION
from curvyzero.training.curvyzero_lightzero_smoke import CurvyZeroLightZeroLocalSmokeEnv
from curvyzero.training.curvyzero_lightzero_smoke import LocalLightZeroTimestep
from curvyzero.training.curvyzero_lightzero_smoke import optional_base_env_timestep_cls
from curvyzero.training.curvyzero_lightzero_smoke import to_base_env_timestep


class _RecordingBaseEnvTimestep:
    def __init__(self, obs, reward, done, info):
        self.obs = obs
        self.reward = reward
        self.done = done
        self.info = info


def test_reset_returns_pinned_lightzero_observation_and_metadata():
    env = CurvyZeroLightZeroLocalSmokeEnv(
        {
            "env_config": CurvyTronConfig(action_repeat=1),
            "ego_player_id": "player_0",
            "seed": 17,
        }
    )

    observation = env.reset()

    assert set(observation) == {"observation", "action_mask", "to_play", "timestep"}
    assert observation["observation"].shape == contract.LIGHTZERO_FLAT_OBSERVATION_SHAPE
    assert observation["observation"].dtype == np.float32
    assert observation["action_mask"].dtype == np.int8
    np.testing.assert_array_equal(observation["action_mask"], np.array([1, 1, 1], dtype=np.int8))
    assert observation["to_play"] == -1
    assert observation["timestep"] == 0

    assert env.last_reset_info is not None
    assert env.last_reset_info["observation_schema_id"] == contract.OBSERVATION_SCHEMA_ID
    assert env.last_reset_info["observation_schema_hash"] == contract.OBSERVATION_SCHEMA_HASH
    assert env.last_reset_info["action_space_id"] == contract.ACTION_SPACE_ID
    assert env.last_reset_info["action_space_hash"] == contract.ACTION_SPACE_HASH
    assert env.last_reset_info["reward_schema_id"] == contract.REWARD_SCHEMA_ID
    assert env.last_reset_info["reward_schema_hash"] == contract.REWARD_SCHEMA_HASH
    assert env.last_reset_info["trainer_adapter_contract_hash"] == (
        contract.TRAINER_ADAPTER_CONTRACT_HASH
    )
    assert env.last_reset_info["opponent_policy_id"] == OPPONENT_POLICY_ID
    assert env.last_reset_info["opponent_policy_version"] == OPPONENT_POLICY_VERSION
    assert env.last_reset_info["needs_reset"] is False


def test_step_uses_fixed_opponent_and_matches_direct_joint_env_step():
    config = CurvyTronConfig(action_repeat=1)
    wrapper = CurvyZeroLightZeroLocalSmokeEnv({"env_config": config})
    wrapper.reset(seed=23)

    timestep = wrapper.step(0)

    direct = CurvyTronEnv(config)
    direct.reset(seed=23)
    direct.step({"player_0": 0, "player_1": 1})
    assert direct.state is not None
    assert wrapper.curvyzero_env.state is not None

    np.testing.assert_allclose(wrapper.curvyzero_env.state.positions, direct.state.positions)
    np.testing.assert_allclose(wrapper.curvyzero_env.state.headings, direct.state.headings)
    np.testing.assert_array_equal(wrapper.curvyzero_env.state.alive, direct.state.alive)
    assert wrapper.curvyzero_env.state.tick == direct.state.tick

    expected_observation = observe_egocentric_rays_v0(
        direct.state,
        direct.config,
        "player_0",
        player_ids=direct.agents,
    )
    np.testing.assert_allclose(timestep.obs["observation"], expected_observation.observation)
    np.testing.assert_array_equal(
        timestep.obs["action_mask"],
        expected_observation.lightzero_action_mask,
    )
    assert timestep.reward == 0.0
    assert timestep.done is False
    assert timestep.info["joint_action"] == {"player_0": 0, "player_1": 1}
    assert timestep.info["opponent_action_id"] == 1
    assert timestep.info["opponent_policy_id"] == OPPONENT_POLICY_ID
    assert timestep.info["opponent_policy_version"] == OPPONENT_POLICY_VERSION
    assert len(timestep.info["trace_hash"]) == 16


def test_terminal_step_returns_final_observation_reward_map_and_blocks_autoreset():
    config = CurvyTronConfig(
        width=10,
        height=30,
        max_ticks=100,
        action_repeat=1,
        speed=9.1,
        turn_rate_radians=np.pi / 2,
        spawn_margin=1.0,
    )
    env = CurvyZeroLightZeroLocalSmokeEnv({"env_config": config})
    env.reset(seed=5)

    timestep = env.step(0)

    assert timestep.done is True
    assert timestep.reward == 1.0
    assert timestep.info["done"] is True
    assert timestep.info["terminated"] is True
    assert timestep.info["truncated"] is False
    assert timestep.info["terminal_reason"] == "survivor_win"
    assert timestep.info["winner_ids"] == ("player_0",)
    assert timestep.info["loser_ids"] == ("player_1",)
    assert timestep.info["final_reward_map"] == {"player_0": 1.0, "player_1": -1.0}
    assert timestep.info["eval_episode_return"] == 1.0
    assert timestep.info["needs_reset"] is True
    assert timestep.info["final_observation"]["observation"].shape == (
        contract.LIGHTZERO_FLAT_OBSERVATION_SHAPE
    )
    np.testing.assert_array_equal(
        timestep.info["final_observation"]["action_mask"],
        np.array([0, 0, 0], dtype=np.int8),
    )
    for key in contract.LIGHTZERO_TERMINAL_INFO_KEYS:
        assert key in timestep.info

    with pytest.raises(RuntimeError, match="reset must be called"):
        env.step(1)

    reset_observation = env.reset(seed=5)
    np.testing.assert_array_equal(
        reset_observation["action_mask"],
        np.array([1, 1, 1], dtype=np.int8),
    )


def test_truncation_uses_done_timeout_metadata_and_zero_terminal_mask():
    env = CurvyZeroLightZeroLocalSmokeEnv(
        {
            "env_config": CurvyTronConfig(max_ticks=1, action_repeat=1),
            "seed": 9,
        }
    )
    env.reset()

    timestep = env.step(1)

    assert timestep.done is True
    assert timestep.reward == 0.0
    assert timestep.info["terminated"] is False
    assert timestep.info["truncated"] is True
    assert timestep.info["timeout"] is True
    assert timestep.info["terminal_reason"] == "timeout"
    assert timestep.info["truncation_reason"] == "max_ticks"
    assert timestep.info["eval_episode_return"] == 0.0
    np.testing.assert_array_equal(
        timestep.obs["action_mask"],
        np.array([0, 0, 0], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        timestep.info["final_observation"]["action_mask"],
        np.array([0, 0, 0], dtype=np.int8),
    )


def test_same_seed_and_actions_are_deterministic_through_the_adapter():
    config = CurvyTronConfig(action_repeat=1)
    first = CurvyZeroLightZeroLocalSmokeEnv({"env_config": config})
    second = CurvyZeroLightZeroLocalSmokeEnv({"env_config": config})

    np.testing.assert_allclose(first.reset(seed=31)["observation"], second.reset(seed=31)["observation"])
    first_step = first.step(2)
    second_step = second.step(2)

    np.testing.assert_allclose(first_step.obs["observation"], second_step.obs["observation"])
    np.testing.assert_array_equal(first_step.obs["action_mask"], second_step.obs["action_mask"])
    assert first_step.info["joint_action"] == second_step.info["joint_action"]
    assert first_step.info["trace_hash"] == second_step.info["trace_hash"]


def test_local_timestep_converts_through_explicit_base_env_timestep_boundary():
    timestep = LocalLightZeroTimestep(
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
    timestep = LocalLightZeroTimestep(obs={}, reward=0.0, done=False, info={})

    with pytest.raises(ImportError, match="ding.envs.BaseEnvTimestep is not available"):
        to_base_env_timestep(timestep)


def test_real_base_env_timestep_conversion_when_di_engine_is_available():
    base_env_timestep_cls = optional_base_env_timestep_cls()
    if base_env_timestep_cls is None:
        pytest.skip("DI-engine/LightZero runtime is not installed locally")
    env = CurvyZeroLightZeroLocalSmokeEnv({"env_config": CurvyTronConfig(action_repeat=1)})
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
