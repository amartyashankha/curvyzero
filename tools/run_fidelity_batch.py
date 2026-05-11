"""Run a local batch of JS/Python scenario fidelity loops."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

try:
    from tools import run_fidelity_loop as fidelity_loop
except ModuleNotFoundError:
    _LOOP_PATH = Path(__file__).resolve().with_name("run_fidelity_loop.py")
    _LOOP_SPEC = importlib.util.spec_from_file_location("run_fidelity_loop", _LOOP_PATH)
    if _LOOP_SPEC is None or _LOOP_SPEC.loader is None:
        raise
    fidelity_loop = importlib.util.module_from_spec(_LOOP_SPEC)
    sys.modules[_LOOP_SPEC.name] = fidelity_loop
    _LOOP_SPEC.loader.exec_module(fidelity_loop)

RunLoop = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class BatchResult:
    summary: dict[str, Any]
    summary_path: Path
    exit_code: int


def load_scenario_paths(batch_path: str | Path) -> list[Path]:
    manifest_path = _resolve_manifest_path(batch_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    entries = _scenario_entries(payload)
    return [
        _scenario_path_from_entry(entry, manifest_path.parent, index)
        for index, entry in enumerate(entries)
    ]


def run_batch(
    batch_path: str | Path,
    *,
    artifact_root: str | Path = fidelity_loop.DEFAULT_ARTIFACT_ROOT,
    node_executable: str = "node",
    python_executable: str = sys.executable,
    python_runner: str | None = None,
    raw_diff: bool = False,
    fail_on_mismatch: bool = False,
    run_loop: RunLoop = fidelity_loop.run_loop,
) -> BatchResult:
    scenarios = load_scenario_paths(batch_path)
    artifact_root_path = _resolve_artifact_root(artifact_root)
    artifact_root_path.mkdir(parents=True, exist_ok=True)

    diff_mode = "raw" if raw_diff else "common-trace"
    counts = {"pass": 0, "fail": 0, "blocked": 0}
    scenario_entries = []

    for scenario in scenarios:
        try:
            result = run_loop(
                scenario,
                artifact_root=artifact_root_path,
                node_executable=node_executable,
                python_executable=python_executable,
                python_runner=python_runner,
                common_trace=not raw_diff,
                fail_on_mismatch=fail_on_mismatch,
            )
            entry = _entry_from_loop_result(scenario, result)
        except Exception as error:  # noqa: BLE001 - one bad scenario should be summarized.
            entry = _blocked_entry(scenario, artifact_root_path, diff_mode, error)

        counts[entry["status"]] += 1
        scenario_entries.append(entry)

    summary = {
        "schema": "curvyzero_local_fidelity_batch/v1",
        "artifact_root": _display_path(artifact_root_path),
        "diff_mode": diff_mode,
        "counts": counts,
        "scenarios": scenario_entries,
    }
    summary_path = artifact_root_path / "summary.json"
    summary_path.write_text(_json_text(summary), encoding="utf-8")

    return BatchResult(
        summary=summary,
        summary_path=summary_path,
        exit_code=_batch_exit_code(counts, fail_on_mismatch=fail_on_mismatch),
    )


def _scenario_entries(payload: Any) -> Sequence[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        scenarios = payload.get("scenarios")
        if isinstance(scenarios, list):
            return scenarios
        raise ValueError("batch object must include a scenarios list")
    raise ValueError("batch JSON must be a list or an object with a scenarios list")


def _scenario_path_from_entry(entry: Any, manifest_dir: Path, index: int) -> Path:
    if isinstance(entry, str):
        path_value = entry
    elif isinstance(entry, dict):
        path_value = entry.get("path", entry.get("scenario_path"))
    else:
        raise ValueError(f"scenario entry {index} must be a path string or object")

    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"scenario entry {index} must include a non-empty path")
    return _resolve_scenario_path(path_value, manifest_dir)


def _resolve_scenario_path(path_value: str, manifest_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()

    manifest_relative = (manifest_dir / path).resolve()
    if manifest_relative.exists():
        return manifest_relative
    return (fidelity_loop.REPO_ROOT / path).resolve()


def _entry_from_loop_result(scenario: Path, result: Any) -> dict[str, Any]:
    summary = result.summary
    status = _batch_status(summary)
    outputs = summary.get("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}

    return {
        "scenario_id": _scenario_id(summary, scenario),
        "status": status,
        "summary_path": _display_path(Path(result.summary_path)),
        "first_mismatch": summary.get("first_mismatch"),
        "outputs": outputs,
    }


def _batch_status(loop_summary: dict[str, Any]) -> str:
    diff_status = loop_summary.get("diff_status")
    loop_status = loop_summary.get("status")
    if diff_status == "pass" or loop_status == "match" or loop_summary.get("match") is True:
        return "pass"
    if diff_status == "fail" or loop_status == "mismatch":
        return "fail"
    return "blocked"


def _blocked_entry(
    scenario: Path,
    artifact_root: Path,
    diff_mode: str,
    error: Exception,
) -> dict[str, Any]:
    scenario_id = _fallback_scenario_id(scenario)
    artifact_id = _safe_path_component(scenario_id)
    artifact_dir = artifact_root / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "summary.json"
    outputs = {"summary": _display_path(summary_path)}
    error_payload = {
        "type": type(error).__name__,
        "message": str(error) or type(error).__name__,
    }

    scenario_summary = {
        "schema": "curvyzero_local_fidelity_loop/v1",
        "scenario_id": scenario_id,
        "artifact_id": artifact_id,
        "scenario_path": _display_path(scenario),
        "artifact_dir": _display_path(artifact_dir),
        "diff_mode": diff_mode,
        "outputs": outputs,
        "results": {},
        "commands": {},
        "match": None,
        "diff_status": "blocked",
        "first_mismatch": None,
        "status": "blocked",
        "exit_code": 2,
        "error": error_payload,
    }
    summary_path.write_text(_json_text(scenario_summary), encoding="utf-8")

    return {
        "scenario_id": scenario_id,
        "status": "blocked",
        "summary_path": _display_path(summary_path),
        "first_mismatch": None,
        "outputs": outputs,
    }


def _fallback_scenario_id(scenario: Path) -> str:
    try:
        return fidelity_loop.scenario_id_from_file(scenario)
    except Exception:  # noqa: BLE001 - fall back when the scenario itself is broken.
        return scenario.stem or "scenario"


def _scenario_id(loop_summary: dict[str, Any], scenario: Path) -> str:
    scenario_id = loop_summary.get("scenario_id")
    if isinstance(scenario_id, str) and scenario_id.strip():
        return scenario_id
    return _fallback_scenario_id(scenario)


def _batch_exit_code(counts: dict[str, int], *, fail_on_mismatch: bool) -> int:
    if counts["blocked"]:
        return 2
    if fail_on_mismatch and counts["fail"]:
        return 1
    return 0


def _resolve_manifest_path(batch_path: str | Path) -> Path:
    path = Path(batch_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    cwd_relative = path.resolve()
    if cwd_relative.exists():
        return cwd_relative
    return (fidelity_loop.REPO_ROOT / path).resolve()


def _resolve_artifact_root(artifact_root: str | Path) -> Path:
    path = Path(artifact_root).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (fidelity_loop.REPO_ROOT / path).resolve()


def _safe_path_component(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    safe = safe.strip("._-")
    return safe or "scenario"


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(fidelity_loop.REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a local batch of scenario fidelity loops."
    )
    parser.add_argument(
        "batch",
        help="Batch JSON path. Use a list of scenario paths or an object with scenarios.",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(fidelity_loop.DEFAULT_ARTIFACT_ROOT),
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
    parser.add_argument(
        "--raw-diff",
        action="store_true",
        help="Compare raw JS/Python outputs instead of common-trace output",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Return exit 1 when any scenario has a completed mismatch",
    )
    args = parser.parse_args(argv)

    try:
        result = run_batch(
            args.batch,
            artifact_root=args.artifact_root,
            node_executable=args.node,
            python_executable=args.python,
            python_runner=args.python_runner,
            raw_diff=args.raw_diff,
            fail_on_mismatch=args.fail_on_mismatch,
        )
    except (OSError, ValueError) as error:
        print(f"fidelity batch error: {error}", file=sys.stderr)
        return 2

    sys.stdout.write(_json_text(result.summary))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
