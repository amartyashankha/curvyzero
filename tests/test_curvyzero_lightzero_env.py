import numpy as np

from curvyzero.env import CurvyTronConfig
from curvyzero.env import trainer_contract as contract
from curvyzero.training.curvyzero_lightzero_env import CurvyZeroLightZeroEnv
from curvyzero.training.curvyzero_lightzero_env import LIGHTZERO_CURVYZERO_ENV_ID
from curvyzero.training.curvyzero_lightzero_env import LIGHTZERO_CURVYZERO_IMPORT_NAMES
from curvyzero.training.curvyzero_lightzero_env import LIGHTZERO_CURVYZERO_ENV_TYPE
from curvyzero.training.curvyzero_lightzero_smoke import CurvyZeroLightZeroLocalSmokeEnv


def test_registered_lightzero_env_reuses_local_smoke_semantics():
    config = CurvyTronConfig(action_repeat=1)
    registered = CurvyZeroLightZeroEnv({"env_config": config})
    local = CurvyZeroLightZeroLocalSmokeEnv({"env_config": config})

    registered_reset = registered.reset(seed=23)
    local_reset = local.reset(seed=23)

    assert registered.env_id == LIGHTZERO_CURVYZERO_ENV_ID
    assert registered.lightzero_env_type == LIGHTZERO_CURVYZERO_ENV_TYPE
    assert CurvyZeroLightZeroEnv.config["env_id"] == LIGHTZERO_CURVYZERO_ENV_ID
    assert CurvyZeroLightZeroEnv.config["lightzero_env_type"] == LIGHTZERO_CURVYZERO_ENV_TYPE
    assert CurvyZeroLightZeroEnv.config["lightzero_import_names"] == (
        LIGHTZERO_CURVYZERO_IMPORT_NAMES
    )
    assert LIGHTZERO_CURVYZERO_IMPORT_NAMES == ("curvyzero.training.curvyzero_lightzero_env",)
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


def test_registered_lightzero_env_exposes_spaces_and_random_action():
    env = CurvyZeroLightZeroEnv({"env_config": CurvyTronConfig(action_repeat=1)})
    env.reset(seed=7)

    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.reward_space is not None
    assert _space_shape(env.observation_space) == contract.LIGHTZERO_FLAT_OBSERVATION_SHAPE
    assert _space_shape(env.reward_space) == ()
    assert _space_action_count(env.action_space) == 3
    np.testing.assert_array_equal(env.legal_actions, np.array([0, 1, 2], dtype=np.int64))
    assert env.random_action() in {0, 1, 2}


def _space_shape(space):
    if isinstance(space, dict):
        return space["shape"]
    return space.shape


def _space_action_count(space):
    if isinstance(space, dict):
        return space["n"]
    return space.n
