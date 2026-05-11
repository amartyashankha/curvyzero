import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "benchmark_vector_trainer_actor_loop_profile.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "benchmark_vector_trainer_actor_loop_profile",
    _SCRIPT_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
profile_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = profile_script
_SPEC.loader.exec_module(profile_script)


def test_vector_trainer_actor_loop_profile_writes_valid_replay(tmp_path):
    report = profile_script.run_profile(
        batch_size=2,
        rollout_steps=2,
        decision_ms=1000.0 / 60.0,
        max_ticks=1000,
        seed=123,
        hidden_dim=8,
        event_mode="no-event",
        action_mode="straight",
        artifact_root=tmp_path,
        profile_observation_phases=True,
    )

    assert report["schema_id"] == "curvyzero_vector_trainer_actor_loop_profile/v0"
    assert report["optimizer_profile_schema"] == "curvyzero_optimizer_profile_report/v0"
    assert report["status"] == "ok"
    assert report["run"]["debug_event_mode"] == "no-event"
    assert report["contracts"]["environment_impl_id"] == "VectorTrainerEnv1v1NoBonus"
    assert report["contracts"]["feature_flags"] == ["strict_1v1", "no_bonus", "P=2"]
    assert report["loop_shape"]["observation_shape"] == [2, 2, 2, 106]
    assert report["integrity"]["replay_schema_read_validated"] is True
    assert report["integrity"]["masked_action_violations"] == 0
    assert report["integrity"]["selected_action_positive_weight_violations"] == 0
    assert report["policy_search"]["policy_forward_used_for_action_selection"] is False
    assert report["observation_phase_profile"]["enabled"] is True
    assert report["observation_phase_profile"]["timing_sec"]["ray_cast_sec"] > 0.0
    assert Path(report["artifacts"]["report_json"]).exists()
    replay_path = Path(report["artifacts"]["replay_v0_chunk"])
    assert replay_path.exists()

    with np.load(replay_path, allow_pickle=False) as payload:
        assert payload["reset_source"].dtype != np.dtype("<U1")
        np.testing.assert_array_equal(
            payload["reset_source"],
            np.asarray(["manual", "manual"], dtype="<U6"),
        )


def test_straight_action_outputs_handles_mixed_illegal_straight_rows():
    mapping = SimpleNamespace(
        capacity=3,
        action_count=3,
        row_mask=np.asarray([True, True, False], dtype=bool),
        legal_action_mask=np.asarray(
            [
                [False, True, True],
                [True, False, True],
                [False, False, False],
            ],
            dtype=bool,
        ),
    )

    selected, probs = profile_script._straight_action_outputs(mapping)

    np.testing.assert_array_equal(selected, np.asarray([1, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(
        probs,
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_vector_trainer_actor_loop_profile_rejects_negative_seed(tmp_path):
    with pytest.raises(ValueError, match="seed must be nonnegative"):
        profile_script.run_profile(
            batch_size=2,
            rollout_steps=2,
            decision_ms=1000.0 / 60.0,
            max_ticks=1000,
            seed=-1,
            hidden_dim=8,
            event_mode="no-event",
            action_mode="straight",
            artifact_root=tmp_path,
        )
