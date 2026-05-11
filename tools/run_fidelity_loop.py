"""Run one local JS/Python scenario fidelity loop.

The loop is intentionally small:

1. run the JS reference scenario runner
2. run the Python toy-v0 scenario runner
3. run the first-mismatch diff tool, using the common trace by default
4. write a JSON summary next to the artifacts
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    from curvyzero.env.trace_compare import TraceCompareError, normalize_trace_payload
except ModuleNotFoundError:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from curvyzero.env.trace_compare import TraceCompareError, normalize_trace_payload

DEFAULT_SCENARIO = REPO_ROOT / "scenarios" / "environment" / "forced_two_player_turn_step.json"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "local" / "fidelity"
JS_RUNNER = REPO_ROOT / "tools" / "reference_oracle" / "scenario_runner.js"
DIFF_TOOL = REPO_ROOT / "tools" / "fidelity_diff.py"
DIFF_STATUSES = {"pass", "fail", "blocked"}
BLOCKED_DIFF_REASONS = {
    "diff output parse error",
    "invalid input",
    "trace normalization error",
}

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class LoopPaths:
    scenario: Path
    artifact_dir: Path
    js_output: Path
    js_common_trace_output: Path
    js_timeline_output: Path
    python_output: Path
    python_common_trace_output: Path
    python_timeline_output: Path
    diff_output: Path
    summary_output: Path
    js_stderr: Path
    python_stderr: Path
    diff_stderr: Path


@dataclass(frozen=True, slots=True)
class CommandRecord:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class LoopResult:
    summary: dict[str, Any]
    summary_path: Path
    exit_code: int


def scenario_id_from_file(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("scenario JSON must be an object")
    value = payload.get("scenario_id", payload.get("id", path.stem))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("scenario_id or id must be a non-empty string")
    return value


def build_js_command(scenario_path: Path, *, node_executable: str = "node") -> tuple[str, ...]:
    return (node_executable, str(JS_RUNNER), str(scenario_path))


def build_python_command(
    scenario_path: Path,
    *,
    python_executable: str = sys.executable,
    python_runner: str | None = None,
) -> tuple[str, ...]:
    command = [
        python_executable,
        "-m",
        "curvyzero.env.scenarios",
        str(scenario_path),
    ]
    if python_runner is not None:
        command.extend(("--runner", python_runner))
    command.append("--compact")
    return tuple(command)


def build_diff_command(
    js_output: Path,
    python_output: Path,
    *,
    python_executable: str = sys.executable,
    common_trace: bool = True,
) -> tuple[str, ...]:
    command = [
        python_executable,
        str(DIFF_TOOL),
        str(js_output),
        str(python_output),
        "--json",
    ]
    if common_trace:
        command.append("--common-trace")
    return tuple(command)


def run_loop(
    scenario_path: str | Path = DEFAULT_SCENARIO,
    *,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    node_executable: str = "node",
    python_executable: str = sys.executable,
    python_runner: str | None = None,
    common_trace: bool = True,
    fail_on_mismatch: bool = False,
    runner: Runner = subprocess.run,
) -> LoopResult:
    scenario = _resolve_path(scenario_path)
    artifact_root_path = _resolve_path(artifact_root)
    scenario_id = scenario_id_from_file(scenario)
    artifact_id = _safe_path_component(scenario_id)
    paths = _loop_paths(scenario, artifact_root_path, artifact_id)
    paths.artifact_dir.mkdir(parents=True, exist_ok=True)

    summary = _base_summary(
        scenario_id=scenario_id,
        artifact_id=artifact_id,
        paths=paths,
        common_trace=common_trace,
    )

    js_command = build_js_command(scenario, node_executable=node_executable)
    python_command = build_python_command(
        scenario,
        python_executable=python_executable,
        python_runner=python_runner,
    )
    summary["commands"] = {
        "js": _display_command(js_command),
        "python": _display_command(python_command),
    }

    js_result = _run_command(js_command, runner)
    paths.js_output.write_text(js_result.stdout, encoding="utf-8")
    paths.js_stderr.write_text(js_result.stderr, encoding="utf-8")
    _record_step(summary, "js", js_result, paths.js_output, paths.js_stderr)
    if js_result.returncode != 0:
        return _finish(paths, summary, "js_failed", exit_code=js_result.returncode or 1)

    python_result = _run_command(python_command, runner)
    paths.python_output.write_text(python_result.stdout, encoding="utf-8")
    paths.python_stderr.write_text(python_result.stderr, encoding="utf-8")
    _record_step(summary, "python", python_result, paths.python_output, paths.python_stderr)
    if python_result.returncode != 0:
        return _finish(paths, summary, "python_failed", exit_code=python_result.returncode or 1)

    if common_trace:
        _write_common_trace_sidecars(paths, summary, js_result.stdout, python_result.stdout)

    diff_command = build_diff_command(
        paths.js_output,
        paths.python_output,
        python_executable=python_executable,
        common_trace=common_trace,
    )
    summary["commands"]["diff"] = _display_command(diff_command)
    diff_result = _run_command(diff_command, runner)
    paths.diff_stderr.write_text(diff_result.stderr, encoding="utf-8")
    _record_step(summary, "diff", diff_result, paths.diff_output, paths.diff_stderr)

    diff_payload, parse_error = _parse_diff_payload(diff_result.stdout)
    if parse_error is not None:
        diff_payload = {
            "match": False,
            "status": "blocked",
            "reason": "diff output parse error",
            "message": parse_error,
            "stdout": diff_result.stdout,
        }
        paths.diff_output.write_text(_json_text(diff_payload), encoding="utf-8")
        _record_diff_summary(summary, diff_payload)
        return _finish(paths, summary, "diff_failed", exit_code=2)

    diff_status = _record_diff_summary(summary, diff_payload)
    paths.diff_output.write_text(_json_text(diff_payload), encoding="utf-8")

    if diff_result.returncode not in (0, 1) or diff_status == "blocked":
        exit_code = diff_result.returncode if diff_result.returncode else 2
        return _finish(paths, summary, "diff_failed", exit_code=exit_code)

    if diff_status == "pass":
        return _finish(paths, summary, "match", exit_code=0)

    exit_code = 1 if fail_on_mismatch else 0
    return _finish(paths, summary, "mismatch", exit_code=exit_code)


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def _loop_paths(scenario: Path, artifact_root: Path, artifact_id: str) -> LoopPaths:
    artifact_dir = artifact_root / artifact_id
    return LoopPaths(
        scenario=scenario,
        artifact_dir=artifact_dir,
        js_output=artifact_dir / "js.json",
        js_common_trace_output=artifact_dir / "js.common_trace.json",
        js_timeline_output=artifact_dir / "js.timeline.txt",
        python_output=artifact_dir / "python.json",
        python_common_trace_output=artifact_dir / "python.common_trace.json",
        python_timeline_output=artifact_dir / "python.timeline.txt",
        diff_output=artifact_dir / "diff.json",
        summary_output=artifact_dir / "summary.json",
        js_stderr=artifact_dir / "js.stderr.txt",
        python_stderr=artifact_dir / "python.stderr.txt",
        diff_stderr=artifact_dir / "diff.stderr.txt",
    )


def _base_summary(
    *,
    scenario_id: str,
    artifact_id: str,
    paths: LoopPaths,
    common_trace: bool,
) -> dict[str, Any]:
    return {
        "schema": "curvyzero_local_fidelity_loop/v1",
        "scenario_id": scenario_id,
        "artifact_id": artifact_id,
        "scenario_path": _display_path(paths.scenario),
        "artifact_dir": _display_path(paths.artifact_dir),
        "diff_mode": "common-trace" if common_trace else "raw",
        "outputs": {
            "js": _display_path(paths.js_output),
            "python": _display_path(paths.python_output),
            "diff": _display_path(paths.diff_output),
            "summary": _display_path(paths.summary_output),
            "js_stderr": _display_path(paths.js_stderr),
            "python_stderr": _display_path(paths.python_stderr),
            "diff_stderr": _display_path(paths.diff_stderr),
        },
        "results": {},
        "commands": {},
        "match": None,
        "diff_status": None,
        "first_mismatch": None,
    }


def _run_command(command: Sequence[str], runner: Runner) -> CommandRecord:
    completed = runner(
        list(command),
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
    )
    return CommandRecord(
        argv=tuple(command),
        returncode=int(completed.returncode),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    src_path = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
    return env


def _record_step(
    summary: dict[str, Any],
    name: str,
    result: CommandRecord,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    summary["results"][name] = {
        "returncode": result.returncode,
        "stdout_path": _display_path(stdout_path),
        "stderr_path": _display_path(stderr_path),
    }


def _parse_diff_payload(stdout: str) -> tuple[dict[str, Any], str | None]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        return {}, f"Could not parse diff JSON: {error}"
    if not isinstance(payload, dict):
        return {}, "Diff JSON must be an object."
    return payload, None


def _write_common_trace_sidecars(
    paths: LoopPaths,
    summary: dict[str, Any],
    js_stdout: str,
    python_stdout: str,
) -> None:
    try:
        js_common_trace = _normalize_common_trace_text(js_stdout)
        python_common_trace = _normalize_common_trace_text(python_stdout)
    except (json.JSONDecodeError, TraceCompareError):
        return

    paths.js_common_trace_output.write_text(_json_text(js_common_trace), encoding="utf-8")
    paths.python_common_trace_output.write_text(
        _json_text(python_common_trace),
        encoding="utf-8",
    )
    paths.js_timeline_output.write_text(_timeline_text(js_common_trace), encoding="utf-8")
    paths.python_timeline_output.write_text(
        _timeline_text(python_common_trace),
        encoding="utf-8",
    )
    summary["outputs"]["js_common_trace"] = _display_path(paths.js_common_trace_output)
    summary["outputs"]["python_common_trace"] = _display_path(paths.python_common_trace_output)
    summary["outputs"]["js_timeline"] = _display_path(paths.js_timeline_output)
    summary["outputs"]["python_timeline"] = _display_path(paths.python_timeline_output)


def _normalize_common_trace_text(raw_text: str) -> dict[str, object]:
    return normalize_trace_payload(json.loads(raw_text))


def _timeline_text(common_trace: Mapping[str, Any]) -> str:
    lines = []
    scenario_id = common_trace.get("scenario_id")
    if scenario_id is not None:
        lines.append(f"scenario={_timeline_value(scenario_id)}")

    steps = common_trace.get("steps")
    if not isinstance(steps, list):
        return "\n".join(lines + ["steps=<invalid>"]) + "\n"

    for fallback_index, raw_step in enumerate(steps):
        if not isinstance(raw_step, Mapping):
            lines.append(f"step={fallback_index} <invalid>")
            continue

        step_index = raw_step.get("step_index", fallback_index)
        parts = [f"step={_timeline_value(step_index)}"]
        if "step_ms" in raw_step:
            parts.append(f"step_ms={_timeline_value(raw_step['step_ms'])}")
        if "worldBodyCount" in raw_step:
            parts.append(f"worldBodyCount={_timeline_value(raw_step['worldBodyCount'])}")

        player_text = _timeline_players(raw_step.get("players"))
        if player_text:
            parts.append(f"players={player_text}")

        if "events" in raw_step:
            parts.append(f"events={_timeline_events(raw_step.get('events'))}")

        lines.append(" ".join(parts))

    return "\n".join(lines) + "\n"


def _timeline_players(raw_players: Any) -> str:
    if not isinstance(raw_players, list):
        return ""

    players = []
    for player_index, raw_player in enumerate(raw_players):
        if not isinstance(raw_player, Mapping):
            continue

        player_id = _timeline_value(raw_player.get("player_id", f"player_{player_index}"))
        fields = []
        if "alive" in raw_player:
            fields.append(f"alive={_timeline_value(raw_player['alive'])}")
        if "x" in raw_player and "y" in raw_player:
            fields.append(
                f"pos=({_timeline_value(raw_player['x'])},{_timeline_value(raw_player['y'])})"
            )
        if "angle" in raw_player:
            fields.append(f"angle={_timeline_value(raw_player['angle'])}")
        for key in ("score", "roundScore", "trailPointCount", "bodyNum", "bodyCount"):
            if key in raw_player:
                fields.append(f"{key}={_timeline_value(raw_player[key])}")
        players.append(f"{player_id}[{','.join(fields)}]" if fields else player_id)

    return "|".join(players)


def _timeline_events(raw_events: Any) -> str:
    if not isinstance(raw_events, list):
        return "<invalid>"
    if not raw_events:
        return "[]"

    events = []
    for raw_event in raw_events:
        if not isinstance(raw_event, Mapping):
            continue
        name = _timeline_value(raw_event.get("event", "<unknown>"))
        fields = []
        for key in (
            "player_id",
            "winner_id",
            "killer_id",
            "x",
            "y",
            "important",
            "old",
            "score",
            "roundScore",
        ):
            if key in raw_event:
                fields.append(f"{key}={_timeline_value(raw_event[key])}")
        events.append(f"{name}({','.join(fields)})" if fields else name)
    return "|".join(events) if events else "[]"


def _timeline_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _record_diff_summary(summary: dict[str, Any], diff_payload: dict[str, Any]) -> str:
    diff_status = _diff_status(diff_payload)
    diff_payload["status"] = diff_status
    diff_payload.setdefault("match", diff_status == "pass")
    summary["diff"] = diff_payload
    summary["diff_status"] = diff_status
    summary["match"] = bool(diff_payload.get("match"))
    summary["first_mismatch"] = _first_mismatch(diff_payload)
    return diff_status


def _diff_status(diff_payload: dict[str, Any]) -> str:
    status = diff_payload.get("status")
    if status in DIFF_STATUSES:
        return str(status)
    if diff_payload.get("reason") in BLOCKED_DIFF_REASONS:
        return "blocked"
    if diff_payload.get("match") is True:
        return "pass"
    if diff_payload.get("match") is False:
        return "fail"
    return "blocked"


def _first_mismatch(diff_payload: dict[str, Any]) -> dict[str, Any] | None:
    if _diff_status(diff_payload) != "fail":
        return None
    keys = ("path", "left", "right", "reason", "message")
    mismatch = {key: diff_payload[key] for key in keys if key in diff_payload}
    return mismatch or None


def _finish(
    paths: LoopPaths,
    summary: dict[str, Any],
    status: str,
    *,
    exit_code: int,
) -> LoopResult:
    summary["status"] = status
    summary["exit_code"] = exit_code
    paths.summary_output.write_text(_json_text(summary), encoding="utf-8")
    return LoopResult(summary=summary, summary_path=paths.summary_output, exit_code=exit_code)


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _safe_path_component(value: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in value.strip()
    ).strip("._-")
    if not safe:
        raise ValueError("scenario id cannot be used as an artifact directory")
    return safe


def _display_command(command: Sequence[str]) -> list[str]:
    return [_display_arg(arg) for arg in command]


def _display_arg(arg: str) -> str:
    path = Path(arg)
    if path.is_absolute():
        return _display_path(path)
    return arg


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run JS and Python scenario traces, then write a local diff summary."
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default=str(DEFAULT_SCENARIO),
        help="Scenario JSON path",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help="Root directory for local fidelity artifacts",
    )
    parser.add_argument("--node", default="node", help="Node executable")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for the Python runner and diff tool",
    )
    parser.add_argument(
        "--python-runner",
        choices=(
            "toy-v0",
            "source-kinematics",
            "source-normal-wall",
            "source-borderless-wrap",
            "source-body-canary",
            "source-print-manager-canary",
            "source-trail-cadence-canary",
            "source-trail-gap-canary",
            "source-border-rules",
        ),
        help=(
            "Python scenario runner mode to pass through as --runner. "
            "Omit this to keep the scenario runner default."
        ),
    )
    diff_mode = parser.add_mutually_exclusive_group()
    diff_mode.add_argument(
        "--common-trace",
        action="store_true",
        help="Normalize JS/Python runner outputs before diffing. This is the default.",
    )
    diff_mode.add_argument(
        "--raw-diff",
        action="store_true",
        help="Compare raw JS/Python outputs instead of common-trace output.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Return exit 1 when the diff completes with a mismatch",
    )
    args = parser.parse_args(argv)

    try:
        result = run_loop(
            args.scenario,
            artifact_root=args.artifact_root,
            node_executable=args.node,
            python_executable=args.python,
            python_runner=args.python_runner,
            common_trace=not args.raw_diff,
            fail_on_mismatch=args.fail_on_mismatch,
        )
    except (OSError, ValueError) as error:
        print(f"fidelity loop error: {error}", file=sys.stderr)
        return 2

    sys.stdout.write(_json_text(result.summary))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
