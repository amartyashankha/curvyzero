import numpy as np

from curvyzero.training.curvyzero_debug_visual_lightzero_runtime_probe import (
    CURVYZERO_DEBUG_VISUAL_ENV_TYPE,
)
from curvyzero.training.curvyzero_debug_visual_lightzero_runtime_probe import (
    run_curvyzero_debug_visual_lightzero_runtime_probe,
)
from curvyzero.training.curvytron_visual_observation import DEBUG_OCCUPANCY_GRAY64_LABEL
from curvyzero.training.curvytron_visual_observation import DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID
from curvyzero.training.curvytron_visual_observation import DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH


def test_runtime_probe_reports_debug_visual_env_surface_without_training_locally():
    result = run_curvyzero_debug_visual_lightzero_runtime_probe(
        seed=3,
        include_env_factory=False,
        require_installed_runtime=False,
    )

    assert result["ok"] is True
    assert result["call_policy"] == "does_not_train; does_not_call_lzero_entrypoints"
    assert result["config_surface"]["create_config"]["env"]["type"] == (
        CURVYZERO_DEBUG_VISUAL_ENV_TYPE
    )
    assert result["config_surface"]["simulator_kind"] == "project_owned_curvytron_source_env"
    assert result["config_surface"]["emulator"] == "none"
    assert result["config_surface"]["ale_usage"] == "none"
    assert result["config_surface"]["uses_ale"] is False
    assert result["config_surface"]["surface"] == "debug_visual_tensor"
    assert result["config_surface"]["observation_schema_id"] == DEBUG_OCCUPANCY_GRAY64_LABEL
    assert result["config_surface"]["observation_schema_hash"] == (
        DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH
    )
    assert result["config_surface"]["renderer_impl_id"] == DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID
    assert result["config_surface"]["truth_level"] == "debug_non_fidelity"
    assert result["config_surface"]["source_fidelity_level"] == "none"
    assert result["config_surface"]["source_backed_observation_fidelity"] is False
    assert result["config_surface"]["shape"] == [1, 64, 64]
    assert result["config_surface"]["dtype"] == "float32"
    assert result["config_surface"]["range"] == [0.0, 1.0]
    assert result["config_surface"]["env_raw_frame_shape"] == [1, 64, 64]
    assert result["config_surface"]["model_observation_shape"] == [4, 64, 64]
    assert result["config_surface"]["frame_stack"] == 4
    assert result["config_surface"]["frame_stack_owner"] == "optimizer"
    assert result["config_surface"]["action_space_n"] == 3
    assert result["config_surface"]["model_kind"] == "conv"
    assert result["identity"]["identity_clean"] is True

    reset = result["direct_env"]["exercise"]["reset"]
    assert reset["observation"]["shape"] == [1, 64, 64]
    assert reset["observation"]["dtype"] == "float32"
    assert reset["observation_min"] >= 0.0
    assert reset["observation_max"] <= 1.0
    assert reset["action_mask"]["shape"] == [3]
    assert reset["action_mask"]["dtype"] == "int8"
    np.testing.assert_array_equal(
        reset["action_mask_values"],
        np.array([1, 1, 1], dtype=np.int8),
    )
    assert reset["to_play"] == -1

    step = result["direct_env"]["exercise"]["step"]
    assert step["base_env_timestep_like"] is True
    assert step["obs"]["observation"]["shape"] == [1, 64, 64]
    assert step["obs"]["action_mask"]["dtype"] == "int8"
    assert step["obs"]["to_play"] == -1
    assert step["info"]["selected"]["surface"] == "debug_visual_tensor"
    assert step["info"]["selected"]["observation_schema_id"] == DEBUG_OCCUPANCY_GRAY64_LABEL
    assert step["info"]["selected"]["observation_schema_hash"] == (
        DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH
    )
    assert step["info"]["selected"]["renderer_impl_id"] == DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID
    assert step["info"]["selected"]["shape"] == [1, 64, 64]
    assert step["info"]["selected"]["dtype"] == "float32"
    assert step["info"]["selected"]["range"] == [0.0, 1.0]
    assert step["info"]["selected"]["uses_ale"] is False
    assert step["info"]["selected"]["ale_usage"] == "none"
