import importlib.util
from pathlib import Path
import sys


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "repo_native_ppo_actor_loop_dry_run.py"
)
_SPEC = importlib.util.spec_from_file_location("repo_native_ppo_actor_loop_dry_run", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
dry_run = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = dry_run
_SPEC.loader.exec_module(dry_run)


def test_repo_native_dry_run_reports_optimizer_profile_contract_fields(tmp_path):
    report = dry_run.run_dry_actor_loop(
        batch_size=2,
        rollout_steps=3,
        seed=123,
        artifact_root=tmp_path,
    )

    assert report["schema_id"] == "curvyzero_repo_native_ppo_actor_loop_dry_run/v0"
    assert report["optimizer_profile_schema"] == "curvyzero_optimizer_profile_report/v0"
    assert report["lane"] == "repo_native_actor_loop"
    assert report["denominators"]["env_transitions"] == 6
    assert report["denominators"]["player_ticks"] == 12
    assert any("shape-only dry run" in caveat for caveat in report["caveats"])
    assert report["contracts"]["observation_schema_id"] == "curvyzero_egocentric_rays/v0"
    assert report["contracts"]["reward_schema_id"] == "curvyzero_sparse_round_outcome/v0"
    assert report["contracts"]["environment_impl_id"] == "curvyzero_python_toy_v0_env/v0"
    assert report["loop_shape"]["observation_shape"] == [3, 2, 2, 106]
    assert report["loop_shape"]["action_shape"] == [3, 2, 2]
    assert "joint_action_shape" not in report["loop_shape"]
    assert report["replay_or_rollout"]["storage_mode"] == "npz_chunk"
    assert report["replay_or_rollout"]["final_observation_included"] is True
    assert report["replay_or_rollout"]["rows"] == 12
    assert report["replay_or_rollout"]["field_specs"]["observation"]["shape"] == [3, 2, 2, 106]
    assert report["integrity"]["done_terminated_truncated_invariant_failures"] == 0
    assert report["integrity"]["nan_or_inf_count"] == 0
    assert report["integrity"]["final_rows_staged_before_next_reset"] is True
    assert report["status"] == "ok"
    assert report["policy_search"]["policy_kind"] == "masked_uniform_random"
    assert report["policy_staleness"] == {"mode": "synchronous", "max_version_lag": 0}
    assert report["learner"] == {"ran": False, "update_count": 0}
    assert report["denominators"]["learner_updates"] == 0
    assert "counters" not in report
    assert "policy_and_checkpoint" not in report
    assert report["throughput"]["env_transitions_per_sec"] > 0.0
    assert report["throughput"]["ego_decisions_per_sec"] > 0.0
    assert report["timing_sec"]["replay_write_or_learner_handoff_sec"] > 0.0
    assert report["timing_sec"]["search_sec"] == 0.0
    assert report["timing_sec"]["wall_elapsed_sec"] > 0.0
    assert report["latency_sec"]["policy_action"]["count"] == 3
    assert "action_p95" not in report["latency_sec"]
    assert Path(report["artifacts"]["rollout_npz"]).exists()
    assert Path(report["artifacts"]["report_json"]).exists()
