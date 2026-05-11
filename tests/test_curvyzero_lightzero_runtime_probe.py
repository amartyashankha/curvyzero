import numpy as np

from curvyzero.env import trainer_contract as contract
from curvyzero.training.curvyzero_lightzero_runtime_probe import (
    CURVYZERO_LIGHTZERO_ENV_TYPE,
)
from curvyzero.training.curvyzero_lightzero_runtime_probe import (
    run_curvyzero_lightzero_runtime_probe,
)


def test_runtime_probe_reports_custom_env_surface_without_training_locally():
    result = run_curvyzero_lightzero_runtime_probe(
        seed=3,
        include_env_factory=False,
        include_terminal=True,
        require_installed_runtime=False,
    )

    assert result["ok"] is True
    assert result["call_policy"] == "does_not_train; does_not_call_lzero_entrypoints"
    assert result["config_surface"]["create_config"]["env"]["type"] == (
        CURVYZERO_LIGHTZERO_ENV_TYPE
    )
    assert result["config_surface"]["simulator_kind"] == "project_owned_curvytron_simulator"
    assert result["config_surface"]["emulator"] == "none"
    assert result["config_surface"]["ale_usage"].startswith("none")
    assert result["config_surface"]["atari_usage"] == "none"
    assert result["identity"]["not_cartpole_atari_ale"] is True

    reset = result["direct_env"]["exercise"]["reset"]
    assert reset["observation"]["shape"] == list(contract.LIGHTZERO_FLAT_OBSERVATION_SHAPE)
    assert reset["observation"]["dtype"] == "float32"
    assert reset["action_mask"]["shape"] == [len(contract.ACTION_NAMES)]
    assert reset["action_mask"]["dtype"] == "int8"
    np.testing.assert_array_equal(
        reset["action_mask_values"],
        np.array([1, 1, 1], dtype=np.int8),
    )
    assert reset["to_play"] == -1

    step = result["direct_env"]["exercise"]["step"]
    assert step["base_env_timestep_like"] is True
    assert step["obs"]["observation"]["shape"] == list(contract.LIGHTZERO_FLAT_OBSERVATION_SHAPE)
    assert step["obs"]["action_mask"]["dtype"] == "int8"
    assert step["obs"]["to_play"] == -1

    terminal = result["terminal_env"]
    assert terminal["ok"] is True
    assert terminal["terminal_checks"]["done"] is True
    assert terminal["terminal_checks"]["terminated"] is True
    assert terminal["terminal_checks"]["truncated"] is False
    assert terminal["terminal_checks"]["missing_terminal_info_keys"] == []
    assert terminal["terminal_checks"]["final_observation"]["action_mask"]["dtype"] == "int8"
