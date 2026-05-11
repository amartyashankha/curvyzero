import importlib.util
import json
import sys
from pathlib import Path

_BATCH_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_fidelity_batch.py"
_BATCH_SPEC = importlib.util.spec_from_file_location("run_fidelity_batch", _BATCH_PATH)
assert _BATCH_SPEC is not None
assert _BATCH_SPEC.loader is not None
run_fidelity_batch = importlib.util.module_from_spec(_BATCH_SPEC)
sys.modules[_BATCH_SPEC.name] = run_fidelity_batch
_BATCH_SPEC.loader.exec_module(run_fidelity_batch)


def test_run_batch_defaults_to_common_trace_and_allows_mismatches(tmp_path):
    scenario_a = _write_scenario(tmp_path / "pass.json", "pass_case")
    scenario_b = _write_scenario(tmp_path / "fail.json", "fail_case")
    manifest = tmp_path / "batch.json"
    manifest.write_text(json.dumps([str(scenario_a), str(scenario_b)]), encoding="utf-8")
    calls = []

    def fake_run_loop(scenario, **kwargs):
        calls.append((Path(scenario), kwargs))
        scenario_id = run_fidelity_batch.fidelity_loop.scenario_id_from_file(Path(scenario))
        if scenario_id == "pass_case":
            return _loop_result(tmp_path, scenario_id, status="match", diff_status="pass")
        return _loop_result(
            tmp_path,
            scenario_id,
            status="mismatch",
            diff_status="fail",
            first_mismatch={
                "path": "$.steps[0].players[0].x",
                "message": "First mismatch at $.steps[0].players[0].x.",
            },
        )

    result = run_fidelity_batch.run_batch(
        manifest,
        artifact_root=tmp_path / "artifacts",
        node_executable="node-test",
        python_executable="python-test",
        run_loop=fake_run_loop,
    )

    written_summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert written_summary == result.summary
    assert result.summary["schema"] == "curvyzero_local_fidelity_batch/v1"
    assert result.summary["diff_mode"] == "common-trace"
    assert result.summary["counts"] == {"pass": 1, "fail": 1, "blocked": 0}
    assert [entry["status"] for entry in result.summary["scenarios"]] == ["pass", "fail"]
    assert result.summary["scenarios"][1]["first_mismatch"] == {
        "path": "$.steps[0].players[0].x",
        "message": "First mismatch at $.steps[0].players[0].x.",
    }
    assert [call[0] for call in calls] == [scenario_a.resolve(), scenario_b.resolve()]
    assert [call[1]["common_trace"] for call in calls] == [True, True]
    assert [call[1]["fail_on_mismatch"] for call in calls] == [False, False]
    assert calls[0][1]["node_executable"] == "node-test"
    assert calls[0][1]["python_executable"] == "python-test"


def test_run_batch_accepts_object_manifest_and_raw_strict_mode(tmp_path):
    scenario = _write_scenario(tmp_path / "scenario.json", "strict_case")
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps({"scenarios": [{"path": scenario.name}]}),
        encoding="utf-8",
    )
    calls = []

    def fake_run_loop(scenario_path, **kwargs):
        calls.append((Path(scenario_path), kwargs))
        return _loop_result(
            tmp_path,
            "strict_case",
            status="mismatch",
            diff_status="fail",
            first_mismatch={"path": "$.tick", "message": "First mismatch at $.tick."},
            exit_code=1,
        )

    result = run_fidelity_batch.run_batch(
        manifest,
        artifact_root=tmp_path / "artifacts",
        python_runner="source-kinematics",
        raw_diff=True,
        fail_on_mismatch=True,
        run_loop=fake_run_loop,
    )

    assert result.exit_code == 1
    assert result.summary["diff_mode"] == "raw"
    assert result.summary["counts"] == {"pass": 0, "fail": 1, "blocked": 0}
    assert calls == [
        (
            scenario.resolve(),
            {
                "artifact_root": (tmp_path / "artifacts").resolve(),
                "node_executable": "node",
                "python_executable": sys.executable,
                "python_runner": "source-kinematics",
                "common_trace": False,
                "fail_on_mismatch": True,
            },
        )
    ]


def test_run_batch_copies_observability_output_paths(tmp_path):
    scenario = _write_scenario(tmp_path / "scenario.json", "sidecar_case")
    manifest = tmp_path / "batch.json"
    manifest.write_text(json.dumps([str(scenario)]), encoding="utf-8")

    def fake_run_loop(_scenario_path, **_kwargs):
        return _loop_result(
            tmp_path,
            "sidecar_case",
            status="match",
            diff_status="pass",
            include_common_trace_outputs=True,
        )

    result = run_fidelity_batch.run_batch(
        manifest,
        artifact_root=tmp_path / "artifacts",
        run_loop=fake_run_loop,
    )

    outputs = result.summary["scenarios"][0]["outputs"]

    assert result.exit_code == 0
    assert outputs["js_common_trace"] == "artifacts/sidecar_case/js.common_trace.json"
    assert outputs["python_common_trace"] == "artifacts/sidecar_case/python.common_trace.json"
    assert outputs["js_timeline"] == "artifacts/sidecar_case/js.timeline.txt"
    assert outputs["python_timeline"] == "artifacts/sidecar_case/python.timeline.txt"


def test_loads_source_body_canary_batch_manifest():
    repo_root = Path(__file__).resolve().parents[1]
    paths = run_fidelity_batch.load_scenario_paths(
        repo_root / "scenarios" / "environment" / "source_body_canary_batch.json"
    )

    assert [path.name for path in paths] == [
        "source_body_opponent_tangent_safe_step.json",
        "source_body_opponent_overlap_kills_step.json",
        "source_body_own_delta3_safe_step.json",
        "source_body_own_delta4_kills_step.json",
        "source_body_same_frame_point_kills_step.json",
        "source_body_same_frame_point_control_safe_step.json",
    ]


def test_loads_source_body_old_metadata_batch_manifest():
    repo_root = Path(__file__).resolve().parents[1]
    paths = run_fidelity_batch.load_scenario_paths(
        repo_root / "scenarios" / "environment" / "source_body_old_metadata_batch.json"
    )

    assert [path.name for path in paths] == [
        "source_body_old_opponent_overlap_kills_step.json",
    ]


def test_loads_source_kinematics_batch_manifest():
    repo_root = Path(__file__).resolve().parents[1]
    paths = run_fidelity_batch.load_scenario_paths(
        repo_root / "scenarios" / "environment" / "source_kinematics_batch.json"
    )

    assert [path.name for path in paths] == [
        "source_kinematics_straight_step.json",
        "source_kinematics_straight_multistep.json",
        "source_kinematics_turn_multistep.json",
        "source_kinematics_varied_elapsed_multistep.json",
        "source_kinematics_left_turn_step.json",
        "source_kinematics_right_turn_step.json",
        "forced_two_player_turn_step.json",
    ]


def test_loads_source_border_batch_manifest():
    repo_root = Path(__file__).resolve().parents[1]
    paths = run_fidelity_batch.load_scenario_paths(
        repo_root / "scenarios" / "environment" / "source_border_batch.json"
    )

    assert [path.name for path in paths] == [
        "source_normal_wall_death_step.json",
        "source_normal_wall_same_frame_draw_step.json",
        "source_borderless_wrap_step.json",
        "source_borderless_print_manager_wrap_toggle_step.json",
        "source_borderless_wrap_skips_destination_body_then_next_frame_kills.json",
        "source_borderless_exact_edge_corner_axis_step.json",
    ]


def test_loads_source_print_manager_batch_manifest():
    repo_root = Path(__file__).resolve().parents[1]
    paths = run_fidelity_batch.load_scenario_paths(
        repo_root / "scenarios" / "environment" / "source_print_manager_batch.json"
    )

    assert [path.name for path in paths] == [
        "source_print_manager_print_to_hole_step.json",
        "source_print_manager_hole_to_print_step.json",
        "source_print_manager_exact_zero_toggle_step.json",
        "source_print_manager_no_toggle_control_step.json",
        "source_print_manager_delayed_start_timer_step.json",
        "source_print_manager_active_stop_on_death_step.json",
        "source_print_manager_active_hole_stop_on_death_step.json",
        "source_print_manager_body_collision_stop_on_death_step.json",
    ]


def test_loads_source_print_manager_random_batch_manifest():
    repo_root = Path(__file__).resolve().parents[1]
    paths = run_fidelity_batch.load_scenario_paths(
        repo_root / "scenarios" / "environment" / "source_print_manager_random_batch.json"
    )

    assert [path.name for path in paths] == [
        "source_print_manager_random_call_order_step.json",
        "source_print_manager_random_cadence_multistep.json",
    ]


def test_loads_source_trail_batch_manifest():
    repo_root = Path(__file__).resolve().parents[1]
    paths = run_fidelity_batch.load_scenario_paths(
        repo_root / "scenarios" / "environment" / "source_trail_batch.json"
    )

    assert [path.name for path in paths] == [
        "source_trail_normal_point_step.json",
        "source_trail_no_point_below_radius_step.json",
    ]


def test_loads_source_trail_gap_batch_manifest():
    repo_root = Path(__file__).resolve().parents[1]
    paths = run_fidelity_batch.load_scenario_paths(
        repo_root / "scenarios" / "environment" / "source_trail_gap_batch.json"
    )

    assert [path.name for path in paths] == [
        "source_trail_gap_hole_space_safe_step.json",
        "source_trail_gap_stored_body_still_kills_step.json",
        "source_trail_gap_print_to_hole_boundary_kills_step.json",
        "source_trail_gap_hole_to_print_boundary_kills_step.json",
    ]


def test_loads_source_collision_order_batch_manifest():
    repo_root = Path(__file__).resolve().parents[1]
    paths = run_fidelity_batch.load_scenario_paths(
        repo_root / "scenarios" / "environment" / "source_collision_order_batch.json"
    )

    assert [path.name for path in paths] == [
        "source_collision_death_point_kills_later_player_step.json",
        "source_collision_head_head_reverse_order_single_death_step.json",
    ]


def test_run_batch_returns_nonzero_for_blocked_loop_result(tmp_path):
    scenario = _write_scenario(tmp_path / "blocked.json", "blocked_case")
    manifest = tmp_path / "batch.json"
    manifest.write_text(json.dumps({"scenarios": [str(scenario)]}), encoding="utf-8")

    def fake_run_loop(_scenario_path, **_kwargs):
        return _loop_result(
            tmp_path,
            "blocked_case",
            status="diff_failed",
            diff_status="blocked",
            exit_code=2,
        )

    result = run_fidelity_batch.run_batch(
        manifest,
        artifact_root=tmp_path / "artifacts",
        run_loop=fake_run_loop,
    )

    assert result.exit_code == 2
    assert result.summary["counts"] == {"pass": 0, "fail": 0, "blocked": 1}
    assert result.summary["scenarios"][0]["status"] == "blocked"
    assert result.summary["scenarios"][0]["first_mismatch"] is None


def test_run_batch_writes_blocked_entry_when_run_loop_raises(tmp_path):
    scenario = _write_scenario(tmp_path / "crash.json", "crash/case")
    manifest = tmp_path / "batch.json"
    manifest.write_text(json.dumps([str(scenario)]), encoding="utf-8")

    def fake_run_loop(_scenario_path, **_kwargs):
        raise OSError("runner unavailable")

    result = run_fidelity_batch.run_batch(
        manifest,
        artifact_root=tmp_path / "artifacts",
        run_loop=fake_run_loop,
    )

    scenario_summary = json.loads(
        (tmp_path / "artifacts" / "crash_case" / "summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert result.exit_code == 2
    assert result.summary["counts"] == {"pass": 0, "fail": 0, "blocked": 1}
    assert result.summary["scenarios"][0]["scenario_id"] == "crash/case"
    assert result.summary["scenarios"][0]["summary_path"].endswith(
        "artifacts/crash_case/summary.json"
    )
    assert scenario_summary["diff_status"] == "blocked"
    assert scenario_summary["error"] == {
        "type": "OSError",
        "message": "runner unavailable",
    }


def _write_scenario(path: Path, scenario_id: str) -> Path:
    path.write_text(json.dumps({"scenario_id": scenario_id}), encoding="utf-8")
    return path


def _loop_result(
    tmp_path: Path,
    scenario_id: str,
    *,
    status: str,
    diff_status: str,
    first_mismatch=None,
    exit_code: int = 0,
    include_common_trace_outputs: bool = False,
):
    summary_path = tmp_path / "artifacts" / scenario_id / "summary.json"
    outputs = {
        "js": f"artifacts/{scenario_id}/js.json",
        "python": f"artifacts/{scenario_id}/python.json",
        "diff": f"artifacts/{scenario_id}/diff.json",
        "summary": f"artifacts/{scenario_id}/summary.json",
        "js_stderr": f"artifacts/{scenario_id}/js.stderr.txt",
        "python_stderr": f"artifacts/{scenario_id}/python.stderr.txt",
        "diff_stderr": f"artifacts/{scenario_id}/diff.stderr.txt",
    }
    if include_common_trace_outputs:
        outputs["js_common_trace"] = f"artifacts/{scenario_id}/js.common_trace.json"
        outputs["python_common_trace"] = f"artifacts/{scenario_id}/python.common_trace.json"
        outputs["js_timeline"] = f"artifacts/{scenario_id}/js.timeline.txt"
        outputs["python_timeline"] = f"artifacts/{scenario_id}/python.timeline.txt"
    summary = {
        "scenario_id": scenario_id,
        "status": status,
        "diff_status": diff_status,
        "match": diff_status == "pass",
        "first_mismatch": first_mismatch,
        "outputs": outputs,
    }
    return run_fidelity_batch.fidelity_loop.LoopResult(
        summary=summary,
        summary_path=summary_path,
        exit_code=exit_code,
    )
