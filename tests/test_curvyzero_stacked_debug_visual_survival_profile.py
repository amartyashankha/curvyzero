from curvyzero.training.curvyzero_stacked_debug_visual_survival_profile import (
    PROFILE_SCHEMA_ID,
    run_curvyzero_stacked_debug_visual_survival_profile,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    STACKED_DEBUG_VISUAL_SURVIVAL_FRAME_STACK_OWNER,
    STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_ID,
)
from curvyzero.training.curvyzero_survival_time_lightzero_smoke import (
    SURVIVAL_TIME_REWARD_SCHEMA_ID,
)


def test_stacked_debug_visual_survival_profile_local_shape_and_timing():
    result = run_curvyzero_stacked_debug_visual_survival_profile(
        seed=4,
        steps=2,
        num_simulations=2,
        require_installed_lightzero=False,
        attempt_installed_lightzero=False,
    )

    assert result["ok"] is True
    assert result["schema"] == PROFILE_SCHEMA_ID
    assert result["surface"]["observation_schema_id"] == STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_ID
    assert result["surface"]["env_import_names"] == [
        "curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env"
    ]
    assert result["surface"]["reward_schema_id"] == SURVIVAL_TIME_REWARD_SCHEMA_ID
    assert result["surface"]["frame_stack_owner"] == (
        STACKED_DEBUG_VISUAL_SURVIVAL_FRAME_STACK_OWNER
    )
    assert result["surface"]["debug_fidelity_only"] is True
    assert result["surface"]["uses_ale"] is False
    assert result["timed_components"]["reset"] is True
    assert result["timed_components"]["env_step"] is True
    assert result["timed_components"]["render"] is True
    assert result["timed_components"]["stack"] is True
    assert result["timed_components"]["replay"] is True
    assert result["timed_components"]["sample"] is True
    assert result["timed_components"]["train_muzero"] is False
    assert result["timed_components"]["optimizer_step"] is False
    assert result["lightzero_policy"]["status"] == "not_requested"
    assert result["replay"]["row_count"] == 2
    assert result["replay"]["sample"]["observation_batch"]["shape"] == [2, 4, 64, 64]
    assert result["steps"][0]["observation_shape"] == [4, 64, 64]


def test_stacked_debug_visual_survival_profile_reports_missing_lightzero_when_required():
    result = run_curvyzero_stacked_debug_visual_survival_profile(
        seed=4,
        steps=1,
        num_simulations=2,
        require_installed_lightzero=True,
        attempt_installed_lightzero=True,
    )

    if result["packages"]["LightZero"] == "missing":
        assert result["ok"] is False
        assert result["lightzero_policy"]["status"] == "blocked"
        assert "installed LightZero policy/search setup did not complete" in result["problems"]
