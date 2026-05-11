import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from curvyzero.env.trace_compare import project_common_trace

_LOOP_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_fidelity_loop.py"
_LOOP_SPEC = importlib.util.spec_from_file_location("run_fidelity_loop", _LOOP_PATH)
assert _LOOP_SPEC is not None
assert _LOOP_SPEC.loader is not None
run_fidelity_loop = importlib.util.module_from_spec(_LOOP_SPEC)
sys.modules[_LOOP_SPEC.name] = run_fidelity_loop
_LOOP_SPEC.loader.exec_module(run_fidelity_loop)


def test_builds_common_trace_diff_command_by_default(tmp_path):
    command = run_fidelity_loop.build_diff_command(
        tmp_path / "js.json",
        tmp_path / "python.json",
        python_executable="python",
    )

    assert command == (
        "python",
        str(run_fidelity_loop.DIFF_TOOL),
        str(tmp_path / "js.json"),
        str(tmp_path / "python.json"),
        "--json",
        "--common-trace",
    )


def test_builds_raw_diff_command_when_requested(tmp_path):
    command = run_fidelity_loop.build_diff_command(
        tmp_path / "js.json",
        tmp_path / "python.json",
        python_executable="python",
        common_trace=False,
    )

    assert command == (
        "python",
        str(run_fidelity_loop.DIFF_TOOL),
        str(tmp_path / "js.json"),
        str(tmp_path / "python.json"),
        "--json",
    )


def test_builds_python_command_with_scenario_default(tmp_path):
    command = run_fidelity_loop.build_python_command(
        tmp_path / "scenario.json",
        python_executable="python",
    )

    assert command == (
        "python",
        "-m",
        "curvyzero.env.scenarios",
        str(tmp_path / "scenario.json"),
        "--compact",
    )


def test_builds_python_command_with_source_kinematics_runner(tmp_path):
    command = run_fidelity_loop.build_python_command(
        tmp_path / "scenario.json",
        python_executable="python",
        python_runner="source-kinematics",
    )

    assert command == (
        "python",
        "-m",
        "curvyzero.env.scenarios",
        str(tmp_path / "scenario.json"),
        "--runner",
        "source-kinematics",
        "--compact",
    )


def test_builds_python_command_with_source_body_canary_runner(tmp_path):
    command = run_fidelity_loop.build_python_command(
        tmp_path / "scenario.json",
        python_executable="python",
        python_runner="source-body-canary",
    )

    assert command == (
        "python",
        "-m",
        "curvyzero.env.scenarios",
        str(tmp_path / "scenario.json"),
        "--runner",
        "source-body-canary",
        "--compact",
    )


def test_run_loop_writes_common_trace_summary_for_first_mismatch_without_node(tmp_path):
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({"scenario_id": "demo/local case"}), encoding="utf-8")
    js_payload = _js_runner_payload("demo/local case", x=20.0)
    python_payload = _python_runner_payload("demo/local case", x=21.0)
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((tuple(command), kwargs))
        if command[0] == "node":
            return subprocess.CompletedProcess(command, 0, _json_line(js_payload), "")
        if command[1:3] == ["-m", "curvyzero.env.scenarios"]:
            return subprocess.CompletedProcess(command, 0, _json_line(python_payload), "")
        return subprocess.CompletedProcess(
            command,
            1,
            json.dumps(
                {
                    "match": False,
                    "path": "$.steps[0].players[0].x",
                    "left": 20.0,
                    "right": 21.0,
                    "reason": "values differ",
                    "message": "First mismatch at $.steps[0].players[0].x.",
                }
            ),
            "",
        )

    result = run_fidelity_loop.run_loop(
        scenario,
        artifact_root=tmp_path / "artifacts",
        node_executable="node",
        python_executable="python",
        runner=fake_runner,
    )

    artifact_dir = tmp_path / "artifacts" / "demo_local_case"
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert summary["status"] == "mismatch"
    assert summary["diff_status"] == "fail"
    assert summary["diff"]["status"] == "fail"
    assert summary["diff_mode"] == "common-trace"
    assert summary["first_mismatch"] == {
        "path": "$.steps[0].players[0].x",
        "left": 20.0,
        "right": 21.0,
        "reason": "values differ",
        "message": "First mismatch at $.steps[0].players[0].x.",
    }
    assert json.loads((artifact_dir / "js.json").read_text(encoding="utf-8")) == js_payload
    assert json.loads((artifact_dir / "js.common_trace.json").read_text(encoding="utf-8")) == (
        project_common_trace(js_payload)
    )
    assert json.loads(
        (artifact_dir / "python.common_trace.json").read_text(encoding="utf-8")
    ) == project_common_trace(python_payload)
    assert summary["outputs"]["js_common_trace"] == str(
        artifact_dir / "js.common_trace.json"
    )
    assert summary["outputs"]["python_common_trace"] == str(
        artifact_dir / "python.common_trace.json"
    )
    assert summary["outputs"]["js_timeline"] == str(artifact_dir / "js.timeline.txt")
    assert summary["outputs"]["python_timeline"] == str(
        artifact_dir / "python.timeline.txt"
    )
    assert (artifact_dir / "js.timeline.txt").read_text(encoding="utf-8") == (
        "scenario=demo/local case\n"
        "step=0 step_ms=16 "
        "players=player_a[alive=true,pos=(20,30),angle=1.25,score=0,roundScore=0]\n"
    )
    assert calls[1][0] == (
        "python",
        "-m",
        "curvyzero.env.scenarios",
        str(scenario.resolve()),
        "--compact",
    )
    assert calls[2][0] == (
        "python",
        str(run_fidelity_loop.DIFF_TOOL),
        str(artifact_dir / "js.json"),
        str(artifact_dir / "python.json"),
        "--json",
        "--common-trace",
    )
    assert calls[0][1]["capture_output"] is True
    assert str(run_fidelity_loop.REPO_ROOT / "src") in calls[1][1]["env"]["PYTHONPATH"]


def test_run_loop_writes_common_trace_sidecars_on_pass(tmp_path):
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({"scenario_id": "demo/pass"}), encoding="utf-8")
    js_payload = _js_runner_payload("demo/pass")
    python_payload = _python_runner_payload("demo/pass")

    def fake_runner(command, **_kwargs):
        if command[0] == "node":
            return subprocess.CompletedProcess(command, 0, _json_line(js_payload), "")
        if command[1:3] == ["-m", "curvyzero.env.scenarios"]:
            return subprocess.CompletedProcess(command, 0, _json_line(python_payload), "")
        return subprocess.CompletedProcess(command, 0, json.dumps({"match": True}), "")

    result = run_fidelity_loop.run_loop(
        scenario,
        artifact_root=tmp_path / "artifacts",
        node_executable="node",
        python_executable="python",
        runner=fake_runner,
    )

    artifact_dir = tmp_path / "artifacts" / "demo_pass"
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert summary["status"] == "match"
    assert summary["outputs"]["js_common_trace"] == str(
        artifact_dir / "js.common_trace.json"
    )
    assert summary["outputs"]["python_common_trace"] == str(
        artifact_dir / "python.common_trace.json"
    )
    assert summary["outputs"]["js_timeline"] == str(artifact_dir / "js.timeline.txt")
    assert summary["outputs"]["python_timeline"] == str(
        artifact_dir / "python.timeline.txt"
    )
    assert json.loads((artifact_dir / "js.common_trace.json").read_text(encoding="utf-8")) == (
        project_common_trace(js_payload)
    )
    assert json.loads(
        (artifact_dir / "python.common_trace.json").read_text(encoding="utf-8")
    ) == project_common_trace(python_payload)


def test_run_loop_timeline_includes_body_counters_and_events(tmp_path):
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({"scenario_id": "source_body_timeline"}), encoding="utf-8")
    js_payload = _js_runner_payload(
        "source_body_timeline",
        include_body=True,
        include_events=True,
    )
    python_payload = _python_runner_payload(
        "source_body_timeline",
        include_body=True,
        include_events=True,
    )

    def fake_runner(command, **_kwargs):
        if command[0] == "node":
            return subprocess.CompletedProcess(command, 0, _json_line(js_payload), "")
        if command[1:3] == ["-m", "curvyzero.env.scenarios"]:
            return subprocess.CompletedProcess(command, 0, _json_line(python_payload), "")
        return subprocess.CompletedProcess(command, 0, json.dumps({"match": True}), "")

    run_fidelity_loop.run_loop(
        scenario,
        artifact_root=tmp_path / "artifacts",
        node_executable="node",
        python_executable="python",
        runner=fake_runner,
    )

    artifact_dir = tmp_path / "artifacts" / "source_body_timeline"
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    js_timeline = (artifact_dir / "js.timeline.txt").read_text(encoding="utf-8")
    python_timeline = (artifact_dir / "python.timeline.txt").read_text(encoding="utf-8")

    assert summary["outputs"]["js_timeline"] == str(artifact_dir / "js.timeline.txt")
    assert summary["outputs"]["python_timeline"] == str(
        artifact_dir / "python.timeline.txt"
    )
    assert "worldBodyCount=7" in js_timeline
    assert "trailPointCount=3" in js_timeline
    assert "bodyNum=2" in js_timeline
    assert "bodyCount=4" in js_timeline
    assert "events=die(player_id=player_a,killer_id=null,old=false)" in js_timeline
    assert "worldBodyCount=7" in python_timeline
    assert "events=die(player_id=player_a,killer_id=null,old=false)" in python_timeline


def test_run_loop_passes_source_kinematics_runner(tmp_path):
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({"scenario_id": "demo/source"}), encoding="utf-8")
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((tuple(command), kwargs))
        if command[0] == "node":
            return subprocess.CompletedProcess(command, 0, '{"trace":[]}\n', "")
        if command[1:3] == ["-m", "curvyzero.env.scenarios"]:
            return subprocess.CompletedProcess(command, 0, '{"trace":[]}\n', "")
        return subprocess.CompletedProcess(command, 0, json.dumps({"match": True}), "")

    result = run_fidelity_loop.run_loop(
        scenario,
        artifact_root=tmp_path / "artifacts",
        node_executable="node",
        python_executable="python",
        python_runner="source-kinematics",
        runner=fake_runner,
    )

    artifact_dir = tmp_path / "artifacts" / "demo_source"
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert summary["status"] == "match"
    assert calls[1][0] == (
        "python",
        "-m",
        "curvyzero.env.scenarios",
        str(scenario.resolve()),
        "--runner",
        "source-kinematics",
        "--compact",
    )
    assert summary["commands"]["python"] == [
        "python",
        "-m",
        "curvyzero.env.scenarios",
        str(scenario.resolve()),
        "--runner",
        "source-kinematics",
        "--compact",
    ]


def test_run_loop_can_use_raw_diff_when_requested(tmp_path):
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({"scenario_id": "demo/raw"}), encoding="utf-8")
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((tuple(command), kwargs))
        if command[0] == "node":
            return subprocess.CompletedProcess(command, 0, '{"trace":[]}\n', "")
        if command[1:3] == ["-m", "curvyzero.env.scenarios"]:
            return subprocess.CompletedProcess(command, 0, '{"trace":[]}\n', "")
        return subprocess.CompletedProcess(command, 0, json.dumps({"match": True}), "")

    result = run_fidelity_loop.run_loop(
        scenario,
        artifact_root=tmp_path / "artifacts",
        node_executable="node",
        python_executable="python",
        common_trace=False,
        runner=fake_runner,
    )

    artifact_dir = tmp_path / "artifacts" / "demo_raw"
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert summary["diff_mode"] == "raw"
    assert "js_common_trace" not in summary["outputs"]
    assert "python_common_trace" not in summary["outputs"]
    assert "js_timeline" not in summary["outputs"]
    assert "python_timeline" not in summary["outputs"]
    assert not (artifact_dir / "js.common_trace.json").exists()
    assert not (artifact_dir / "python.common_trace.json").exists()
    assert not (artifact_dir / "js.timeline.txt").exists()
    assert not (artifact_dir / "python.timeline.txt").exists()
    assert calls[2][0][-1] == "--json"


def test_run_loop_records_blocked_diff_status(tmp_path):
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({"scenario_id": "demo/blocked"}), encoding="utf-8")

    def fake_runner(command, **kwargs):
        if command[0] == "node":
            return subprocess.CompletedProcess(command, 0, '{"trace":"bad"}\n', "")
        if command[1:3] == ["-m", "curvyzero.env.scenarios"]:
            return subprocess.CompletedProcess(command, 0, '{"trace":[]}\n', "")
        return subprocess.CompletedProcess(
            command,
            2,
            json.dumps(
                {
                    "match": False,
                    "status": "blocked",
                    "reason": "trace normalization error",
                    "message": "Trace normalization error: payload.trace must be valid.",
                }
            ),
            "",
        )

    result = run_fidelity_loop.run_loop(
        scenario,
        artifact_root=tmp_path / "artifacts",
        node_executable="node",
        python_executable="python",
        common_trace=True,
        runner=fake_runner,
    )

    artifact_dir = tmp_path / "artifacts" / "demo_blocked"
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))

    assert result.exit_code == 2
    assert summary["status"] == "diff_failed"
    assert summary["diff_status"] == "blocked"
    assert summary["match"] is False
    assert summary["first_mismatch"] is None
    assert summary["diff"]["status"] == "blocked"
    assert "js_common_trace" not in summary["outputs"]
    assert "python_common_trace" not in summary["outputs"]
    assert "js_timeline" not in summary["outputs"]
    assert "python_timeline" not in summary["outputs"]


def _json_line(payload):
    return json.dumps(payload) + "\n"


def _js_runner_payload(scenario_id, *, x=20.0, include_body=False, include_events=False):
    player = {
        "id": "avatar-a",
        "name": "player_a",
        "x": x,
        "y": 30.0,
        "angle": 1.25,
        "alive": True,
        "score": 0,
        "roundScore": 0,
    }
    frame = {
        "tick": 0,
        "stepMs": 16,
        "avatars": [player],
    }
    payload = {
        "scenario": scenario_id,
        "trace": [frame],
    }
    if include_body:
        frame["game"] = {"worldBodyCount": 7}
        player["trailPointCount"] = 3
        player["bodyNum"] = 2
        player["bodyCount"] = 4
    if include_events:
        payload["comparison"] = {"include_events": True}
        frame["events"] = [
            {
                "event": "die",
                "data": {"avatar": "avatar-a", "killer": None, "old": False},
            }
        ]
    return payload


def _python_runner_payload(scenario_id, *, x=20.0, include_body=False, include_events=False):
    scenario = {
        "scenario_id": scenario_id,
        "time_policy": {"step_ms": 16},
        "initial_state": {"players": [{"id": "player_a"}]},
    }
    frame = {
        "positions": [[x, 30.0]],
        "headings": [1.25],
        "alive": [True],
        "scores": [0],
        "roundScores": [0],
    }
    payload = {
        "scenario_id": scenario_id,
        "scenario": scenario,
        "trace": {"frames": [frame]},
    }
    if include_body:
        frame["worldBodyCount"] = 7
        frame["trailPointCounts"] = [3]
        frame["bodyNums"] = [2]
        frame["bodyCounts"] = [4]
    if include_events:
        scenario["comparison"] = {"include_events": True}
        frame["events"] = [
            {
                "event": "die",
                "player_id": "player_a",
                "killer_id": None,
                "old": False,
            }
        ]
    return payload
