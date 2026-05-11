import numpy as np

from curvyzero.training.curvyzero_debug_visual_lightzero_env import (
    LIGHTZERO_DEBUG_VISUAL_ENV_ID,
)
from curvyzero.training.curvyzero_debug_visual_lightzero_env import (
    LIGHTZERO_DEBUG_VISUAL_ENV_TYPE,
)
from curvyzero.training.curvyzero_debug_visual_lightzero_env import (
    LIGHTZERO_DEBUG_VISUAL_IMPORT_NAMES,
)
from curvyzero.training.curvyzero_debug_visual_lightzero_env import (
    CurvyZeroDebugVisualLightZeroEnv,
)
from curvyzero.training.curvyzero_debug_visual_lightzero_smoke import (
    CurvyZeroDebugVisualLightZeroLocalSmokeEnv,
)


def test_registered_debug_visual_env_reuses_local_smoke_semantics():
    registered = CurvyZeroDebugVisualLightZeroEnv({"seed": 23})
    local = CurvyZeroDebugVisualLightZeroLocalSmokeEnv({"seed": 23})

    registered_reset = registered.reset(seed=23)
    local_reset = local.reset(seed=23)

    assert registered.env_id == LIGHTZERO_DEBUG_VISUAL_ENV_ID
    assert registered.lightzero_env_type == LIGHTZERO_DEBUG_VISUAL_ENV_TYPE
    assert CurvyZeroDebugVisualLightZeroEnv.config["env_id"] == LIGHTZERO_DEBUG_VISUAL_ENV_ID
    assert CurvyZeroDebugVisualLightZeroEnv.config["lightzero_env_type"] == (
        LIGHTZERO_DEBUG_VISUAL_ENV_TYPE
    )
    assert CurvyZeroDebugVisualLightZeroEnv.config["lightzero_import_names"] == (
        LIGHTZERO_DEBUG_VISUAL_IMPORT_NAMES
    )
    assert LIGHTZERO_DEBUG_VISUAL_IMPORT_NAMES == (
        "curvyzero.training.curvyzero_debug_visual_lightzero_env",
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
    assert registered_timestep.info["joint_source_move"] == (
        local_timestep.info["joint_source_move"]
    )
    assert registered_timestep.info["trace_hash"] == local_timestep.info["trace_hash"]


def test_registered_debug_visual_env_exposes_spaces_and_random_action():
    env = CurvyZeroDebugVisualLightZeroEnv()
    env.reset(seed=7)

    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.reward_space is not None
    assert _space_shape(env.observation_space) == (1, 64, 64)
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
