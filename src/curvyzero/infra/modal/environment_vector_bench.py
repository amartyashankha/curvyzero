"""Coarse Modal CPU smoke for environment vectorization benchmark scripts.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.environment_vector_bench
    uv run --extra modal modal run -m curvyzero.infra.modal.environment_vector_bench \
      --kind cpu-smoke --run-id env-vector-smoke-20260509

This wrapper runs whole benchmark scripts inside one CPU Modal Function and
writes coarse artifacts to the ``curvyzero-runs`` Volume. It does not call Modal
from an environment step loop.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs

APP_NAME = "curvyzero-env-vector-bench"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")
ROOT_REF = PurePosixPath("environment") / "vectorization"

SOURCE_BENCHMARK_SCRIPT = "scripts/benchmark_source_fidelity.py"
VECTOR_BENCHMARK_SCRIPT = "scripts/benchmark_vectorization_prototype.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("numpy>=1.26")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(Path.cwd() / "scripts", remote_path=str(REMOTE_ROOT / "scripts"), copy=True)
    .add_local_dir(Path.cwd() / "scenarios", remote_path=str(REMOTE_ROOT / "scenarios"), copy=True)
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _run_root_ref(run_id: str) -> PurePosixPath:
    return ROOT_REF / runs.clean_id(run_id, label="run_id")


def _attempt_root_ref(run_id: str, attempt_id: str) -> PurePosixPath:
    return _run_root_ref(run_id) / "attempts" / runs.clean_id(attempt_id, label="attempt_id")


def _artifact_ref(run_id: str, attempt_id: str, *parts: str) -> PurePosixPath:
    ref = _attempt_root_ref(run_id, attempt_id)
    for part in parts:
        ref /= runs.clean_id(part, label="artifact_path_segment")
    return ref


def _command_result(argv: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=REMOTE_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    elapsed_sec = time.perf_counter() - started
    result: dict[str, Any] = {
        "argv": argv,
        "cwd": str(REMOTE_ROOT),
        "returncode": completed.returncode,
        "elapsed_sec": elapsed_sec,
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        result["stdout_tail"] = completed.stdout[-4000:]
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    try:
        result["summary"] = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        result["stdout_tail"] = completed.stdout[-4000:]
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True)) from exc
    return result


def _source_command(*, repeat: int, warmup: int, profile: bool, profile_limit: int) -> list[str]:
    argv = [
        sys.executable,
        SOURCE_BENCHMARK_SCRIPT,
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
        "--format",
        "json",
    ]
    if profile:
        argv.extend(["--profile", "--profile-limit", str(profile_limit)])
    return argv


def _vector_command(
    *,
    batch: int,
    players: int,
    body_capacity: int,
    steps: int,
    warmup: int,
    dtype: str,
    action_pattern: str,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        VECTOR_BENCHMARK_SCRIPT,
        "--batch",
        str(batch),
        "--players",
        str(players),
        "--body-capacity",
        str(body_capacity),
        "--steps",
        str(steps),
        "--warmup",
        str(warmup),
        "--dtype",
        dtype,
        "--action-pattern",
        action_pattern,
        "--seed",
        str(seed),
        "--format",
        "json",
    ]


def _write_json(ref: PurePosixPath, payload: Any, *, exclusive: bool = False) -> dict[str, Any]:
    path = runs.volume_path(RUNS_MOUNT, ref)
    runs.write_json(path, payload, exclusive=exclusive)
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _write_attempt_state(
    *,
    run_id: str,
    attempt_id: str,
    status: str,
    started_at: str,
    ended_at: str | None,
    summary_ref: str | None,
    spec: dict[str, Any],
    exclusive: bool = False,
) -> dict[str, Any]:
    return _write_json(
        _attempt_root_ref(run_id, attempt_id) / "attempt.json",
        {
            "schema": "curvyzero_env_vector_modal_attempt/v1",
            "app_name": APP_NAME,
            "volume_name": VOLUME_NAME,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "status": status,
            "started_at": started_at,
            "ended_at": ended_at,
            "summary_ref": summary_ref,
            "spec": spec,
        },
        exclusive=exclusive,
    )


def _spec_payload(
    *,
    kind: str,
    run_id: str,
    attempt_id: str,
    source_repeat: int,
    source_warmup: int,
    source_profile: bool,
    source_profile_limit: int,
    vector_batch: int,
    vector_players: int,
    vector_body_capacity: int,
    vector_steps: int,
    vector_warmup: int,
    vector_dtype: str,
    vector_action_pattern: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema": "curvyzero_env_vector_modal_cpu_smoke_spec/v1",
        "kind": kind,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "modal_shape": "one coarse CPU Function runs whole benchmark scripts",
        "not_modal_shape": "no per-step, per-player, per-collision-row, or MCTS hot-loop calls",
        "source_fidelity": {
            "script": SOURCE_BENCHMARK_SCRIPT,
            "repeat": source_repeat,
            "warmup": source_warmup,
            "profile": source_profile,
            "profile_limit": source_profile_limit,
        },
        "vector_numpy": {
            "script": VECTOR_BENCHMARK_SCRIPT,
            "batch": vector_batch,
            "players": vector_players,
            "body_capacity": vector_body_capacity,
            "steps": vector_steps,
            "warmup": vector_warmup,
            "dtype": vector_dtype,
            "action_pattern": vector_action_pattern,
            "seed": seed,
            "equivalence_status": "not_run",
        },
        "gpu_status": "not_in_scope_for_this_cpu_smoke",
    }


def _run_manifest(
    *,
    run_id: str,
    attempt_id: str,
    status: str,
    started_at: str,
    ended_at: str,
    spec_ref: str,
    complete_ref: str,
) -> dict[str, Any]:
    return {
        "schema": "curvyzero_env_vector_modal_run/v1",
        "app_name": APP_NAME,
        "volume_name": VOLUME_NAME,
        "run_id": run_id,
        "status": status,
        "created_at": started_at,
        "updated_at": ended_at,
        "latest_attempt_id": attempt_id,
        "spec_ref": spec_ref,
        "complete_ref": complete_ref,
        "fast_lane_status": "Modal CPU smoke only; no GPU/tensor backend yet",
    }


@app.function(image=image, volumes={RUNS_MOUNT: runs_volume}, timeout=10 * 60)
def cpu_smoke(
    run_id: str | None = None,
    attempt_id: str | None = None,
    source_repeat: int = 1,
    source_warmup: int = 0,
    source_profile: bool = False,
    source_profile_limit: int = 15,
    vector_batch: int = 32,
    vector_players: int = 3,
    vector_body_capacity: int = 128,
    vector_steps: int = 50,
    vector_warmup: int = 5,
    vector_dtype: str = "float64",
    vector_action_pattern: str = "straight",
    seed: int = 0,
) -> dict[str, Any]:
    if source_repeat <= 0:
        raise ValueError("source_repeat must be greater than zero")
    if source_warmup < 0 or vector_warmup < 0:
        raise ValueError("warmup values must be zero or greater")
    if vector_batch <= 0 or vector_players <= 0 or vector_steps <= 0:
        raise ValueError("vector_batch, vector_players, and vector_steps must be greater than zero")
    if vector_body_capacity < 0:
        raise ValueError("vector_body_capacity must be zero or greater")
    if vector_dtype not in {"float64", "float32"}:
        raise ValueError("vector_dtype must be float64 or float32")
    if vector_action_pattern not in {"straight", "alternating-turns", "weave"}:
        raise ValueError("unsupported vector_action_pattern")

    started = time.perf_counter()
    started_at = runs.utc_timestamp()
    clean_run_id = runs.clean_id(run_id or runs.new_run_id("env-vector"), label="run_id")
    clean_attempt_id = runs.clean_id(
        attempt_id or runs.new_attempt_id("attempt"),
        label="attempt_id",
    )
    spec = _spec_payload(
        kind="cpu-smoke",
        run_id=clean_run_id,
        attempt_id=clean_attempt_id,
        source_repeat=source_repeat,
        source_warmup=source_warmup,
        source_profile=source_profile,
        source_profile_limit=source_profile_limit,
        vector_batch=vector_batch,
        vector_players=vector_players,
        vector_body_capacity=vector_body_capacity,
        vector_steps=vector_steps,
        vector_warmup=vector_warmup,
        vector_dtype=vector_dtype,
        vector_action_pattern=vector_action_pattern,
        seed=seed,
    )

    spec_file = _write_json(_run_root_ref(clean_run_id) / "spec.json", spec)
    attempt_file = _write_attempt_state(
        run_id=clean_run_id,
        attempt_id=clean_attempt_id,
        status="running",
        started_at=started_at,
        ended_at=None,
        summary_ref=None,
        spec=spec,
        exclusive=True,
    )

    try:
        source_command = _source_command(
            repeat=source_repeat,
            warmup=source_warmup,
            profile=source_profile,
            profile_limit=source_profile_limit,
        )
        vector_command = _vector_command(
            batch=vector_batch,
            players=vector_players,
            body_capacity=vector_body_capacity,
            steps=vector_steps,
            warmup=vector_warmup,
            dtype=vector_dtype,
            action_pattern=vector_action_pattern,
            seed=seed,
        )
        source_result = _command_result(source_command)
        vector_result = _command_result(vector_command)

        source_summary = {
            **source_result["summary"],
            "modal_wrapper": {
                "app_name": APP_NAME,
                "volume_name": VOLUME_NAME,
                "command_elapsed_sec": source_result["elapsed_sec"],
                "stderr_tail": source_result["stderr_tail"],
            },
        }
        vector_summary = {
            **vector_result["summary"],
            "modal_wrapper": {
                "app_name": APP_NAME,
                "volume_name": VOLUME_NAME,
                "command_elapsed_sec": vector_result["elapsed_sec"],
                "stderr_tail": vector_result["stderr_tail"],
            },
        }

        source_file = _write_json(
            _artifact_ref(clean_run_id, clean_attempt_id, "source_fidelity")
            / "batch-000.summary.json",
            source_summary,
        )
        vector_file = _write_json(
            _artifact_ref(clean_run_id, clean_attempt_id, "vector_numpy")
            / "profile-shard-000.json",
            vector_summary,
        )

        sweep_summary = {
            "schema": "curvyzero_env_vector_modal_numpy_sweep_summary/v1",
            "backend": "numpy-prototype",
            "profile_count": 1,
            "equivalence_status": "not_run",
            "profile_refs": [vector_file["ref"]],
            "rates": vector_summary.get("rates", {}),
            "counts": vector_summary.get("counts", {}),
            "memory": vector_summary.get("memory", {}),
        }
        sweep_file = _write_json(
            _artifact_ref(clean_run_id, clean_attempt_id, "vector_numpy")
            / "sweep_summary.json",
            sweep_summary,
        )

        ended_at = runs.utc_timestamp()
        complete = {
            "schema": "curvyzero_env_vector_modal_cpu_smoke_complete/v1",
            "app_name": APP_NAME,
            "volume_name": VOLUME_NAME,
            "run_id": clean_run_id,
            "attempt_id": clean_attempt_id,
            "status": "completed",
            "started_at": started_at,
            "ended_at": ended_at,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            },
            "artifacts": {
                "spec": spec_file,
                "attempt": attempt_file,
                "source_fidelity": source_file,
                "vector_profile": vector_file,
                "vector_sweep": sweep_file,
            },
            "source_fidelity": {
                "scenario_count": source_summary.get("workload", {}).get("scenario_count"),
                "scenario_iterations": source_summary.get("workload", {}).get(
                    "scenario_iterations"
                ),
                "elapsed_sec": source_summary.get("elapsed_sec"),
            },
            "vector_numpy": {
                "profile": vector_summary.get("workload", {}),
                "elapsed_sec": vector_summary.get("elapsed_sec"),
                "rates": vector_summary.get("rates", {}),
                "overflow_envs": vector_summary.get("counts", {}).get("overflow_envs"),
                "equivalence_status": "not_run",
            },
            "gpu_status": "not_started; needs a real JAX or PyTorch tensor rollout first",
            "remote_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        complete_file = _write_json(
            _attempt_root_ref(clean_run_id, clean_attempt_id) / "complete.json",
            complete,
        )
        manifest_file = _write_json(
            _run_root_ref(clean_run_id) / "manifest.json",
            _run_manifest(
                run_id=clean_run_id,
                attempt_id=clean_attempt_id,
                status="completed",
                started_at=started_at,
                ended_at=ended_at,
                spec_ref=spec_file["ref"],
                complete_ref=complete_file["ref"],
            ),
        )
        attempt_file = _write_attempt_state(
            run_id=clean_run_id,
            attempt_id=clean_attempt_id,
            status="completed",
            started_at=started_at,
            ended_at=ended_at,
            summary_ref=complete_file["ref"],
            spec=spec,
        )
        runs_volume.commit()

        return {
            "schema": "curvyzero_env_vector_modal_cpu_smoke_result/v1",
            "app_name": APP_NAME,
            "volume_name": VOLUME_NAME,
            "run_id": clean_run_id,
            "attempt_id": clean_attempt_id,
            "status": "completed",
            "manifest_ref": manifest_file["ref"],
            "complete_ref": complete_file["ref"],
            "attempt_ref": attempt_file["ref"],
            "source_fidelity_ref": source_file["ref"],
            "vector_profile_ref": vector_file["ref"],
            "vector_sweep_ref": sweep_file["ref"],
            "source_scenario_count": complete["source_fidelity"]["scenario_count"],
            "vector_env_steps_per_sec": complete["vector_numpy"]["rates"].get(
                "env_steps_per_sec"
            ),
            "gpu_status": complete["gpu_status"],
            "remote_elapsed_ms": complete["remote_elapsed_ms"],
        }
    except Exception as exc:
        ended_at = runs.utc_timestamp()
        failure = {
            "schema": "curvyzero_env_vector_modal_cpu_smoke_complete/v1",
            "app_name": APP_NAME,
            "volume_name": VOLUME_NAME,
            "run_id": clean_run_id,
            "attempt_id": clean_attempt_id,
            "status": "failed",
            "started_at": started_at,
            "ended_at": ended_at,
            "error": repr(exc),
            "remote_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        complete_file = _write_json(
            _attempt_root_ref(clean_run_id, clean_attempt_id) / "complete.json",
            failure,
        )
        _write_attempt_state(
            run_id=clean_run_id,
            attempt_id=clean_attempt_id,
            status="failed",
            started_at=started_at,
            ended_at=ended_at,
            summary_ref=complete_file["ref"],
            spec=spec,
        )
        runs_volume.commit()
        raise


@app.local_entrypoint()
def main(
    kind: str = "cpu-smoke",
    run_id: str | None = None,
    attempt_id: str | None = None,
    source_repeat: int = 1,
    source_warmup: int = 0,
    source_profile: bool = False,
    source_profile_limit: int = 15,
    vector_batch: int = 32,
    vector_players: int = 3,
    vector_body_capacity: int = 128,
    vector_steps: int = 50,
    vector_warmup: int = 5,
    vector_dtype: str = "float64",
    vector_action_pattern: str = "straight",
    seed: int = 0,
) -> None:
    if kind != "cpu-smoke":
        raise ValueError(f"unknown kind {kind!r}; only 'cpu-smoke' exists")
    result = cpu_smoke.remote(
        run_id=run_id,
        attempt_id=attempt_id,
        source_repeat=source_repeat,
        source_warmup=source_warmup,
        source_profile=source_profile,
        source_profile_limit=source_profile_limit,
        vector_batch=vector_batch,
        vector_players=vector_players,
        vector_body_capacity=vector_body_capacity,
        vector_steps=vector_steps,
        vector_warmup=vector_warmup,
        vector_dtype=vector_dtype,
        vector_action_pattern=vector_action_pattern,
        seed=seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
