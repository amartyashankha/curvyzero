"""Modal wrapper for the CurvyTron two-seat LightZero smoke.

Run from the repository root:

    uv run --extra modal modal run \
      -m curvyzero.infra.modal.lightzero_curvytron_two_seat_train_smoke \
      --seed 0 --batch-size 1 --outer-iterations 2 \
      --collect-steps-per-iteration 4 --updates-per-iteration 1 \
      --num-simulations 2 --replay-scope accumulated \
      --learner-sample-size 32

This is not ``train_muzero``. It runs one installed LightZero ``MuZeroPolicy``
object for both CurvyTron seats before stepping the public multiplayer env.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    DEFAULT_CHECKPOINT_EVERY_ITERATIONS,
    DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY,
    DEFAULT_POLICY_ACTION_REPEAT_MAX,
    DEFAULT_POLICY_ACTION_REPEAT_MIN,
    DEFAULT_POLICY_ACTION_REPEAT_WARMUP_ITERATIONS,
    DEFAULT_PROGRESS_EVERY_ITERATIONS,
    compact_curvytron_two_seat_lightzero_train_smoke_summary,
    run_curvytron_two_seat_lightzero_train_smoke,
)


APP_NAME = "curvyzero-lightzero-curvytron-two-seat-train-smoke"
TASK_ID = "lightzero-curvytron-visual-survival"
VOLUME_NAME = "curvyzero-runs"
LIGHTZERO_VERSION = "0.2.0"
REMOTE_ROOT = Path("/repo")
RUNS_MOUNT = Path("/runs")
CHEAP_GPU_RESOURCE = ["L4", "T4"]
COMPUTE_CPU = "cpu"
COMPUTE_GPU = "gpu-l4-t4"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(f"LightZero=={LIGHTZERO_VERSION}", "numpy>=1.26")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)

app = modal.App(APP_NAME)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

REMOTE_CALL_ARG_NAMES = (
    "seed",
    "batch_size",
    "steps",
    "outer_iterations",
    "collect_steps_per_iteration",
    "updates_per_iteration",
    "num_simulations",
    "learner_updates",
    "allow_optimizer_step",
    "replay_scope",
    "learner_sample_size",
    "max_replay_rows",
    "record_log_limit",
    "replay_row_log_limit",
    "max_ticks",
    "decision_ms",
    "alive_reward",
    "dead_reward",
    "action_selection_mode",
    "collect_temperature",
    "collect_epsilon",
    "action_noop_probability",
    "action_noop_warmup_iterations",
    "policy_action_repeat_min",
    "policy_action_repeat_max",
    "policy_action_repeat_extra_probability",
    "policy_action_repeat_warmup_iterations",
    "observation_noise_std",
    "checkpoint_every_iterations",
    "save_initial_checkpoint",
    "progress_every_iterations",
    "run_id",
    "attempt_id",
    "output",
)


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=12 * 60 * 60,
    cpu=40.0,
    memory=65536,
)
def lightzero_curvytron_two_seat_train_smoke(
    seed: int = 0,
    batch_size: int = 1,
    steps: int = 4,
    outer_iterations: int = 1,
    collect_steps_per_iteration: int | None = None,
    updates_per_iteration: int | None = None,
    num_simulations: int = 2,
    learner_updates: int = 1,
    allow_optimizer_step: bool = False,
    replay_scope: str = "current_iteration",
    learner_sample_size: int | None = None,
    max_replay_rows: int | None = 4096,
    record_log_limit: int = 512,
    replay_row_log_limit: int = 256,
    max_ticks: int | None = None,
    decision_ms: float = 300.0,
    alive_reward: float = 1.0,
    dead_reward: float = 0.0,
    action_selection_mode: str = "collect",
    collect_temperature: float = 1.0,
    collect_epsilon: float = 0.25,
    action_noop_probability: float = 0.0,
    action_noop_warmup_iterations: int = 0,
    policy_action_repeat_min: int = DEFAULT_POLICY_ACTION_REPEAT_MIN,
    policy_action_repeat_max: int = DEFAULT_POLICY_ACTION_REPEAT_MAX,
    policy_action_repeat_extra_probability: float = (
        DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
    ),
    policy_action_repeat_warmup_iterations: int = (
        DEFAULT_POLICY_ACTION_REPEAT_WARMUP_ITERATIONS
    ),
    observation_noise_std: float = 0.0,
    checkpoint_every_iterations: int = DEFAULT_CHECKPOINT_EVERY_ITERATIONS,
    save_initial_checkpoint: bool = False,
    progress_every_iterations: int = DEFAULT_PROGRESS_EVERY_ITERATIONS,
    run_id: str = "curvytron-two-seat-smoke-s0",
    attempt_id: str = "two-seat-smoke",
    output: str = "none",
) -> dict[str, Any]:
    checkpoint_ref = runs.checkpoints_root_ref(TASK_ID, run_id) / "lightzero"
    checkpoint_dir = runs.volume_path(RUNS_MOUNT, checkpoint_ref)
    train_ref = runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
    progress_ref = train_ref / "progress.jsonl"
    progress_path = runs.volume_path(RUNS_MOUNT, progress_ref)
    result = run_curvytron_two_seat_lightzero_train_smoke(
        seed=seed,
        batch_size=batch_size,
        steps=steps,
        outer_iterations=outer_iterations,
        collect_steps_per_iteration=collect_steps_per_iteration,
        updates_per_iteration=updates_per_iteration,
        num_simulations=num_simulations,
        learner_updates=learner_updates,
        allow_optimizer_step=allow_optimizer_step,
        replay_scope=replay_scope,
        learner_sample_size=learner_sample_size,
        max_replay_rows=max_replay_rows,
        record_log_limit=record_log_limit,
        replay_row_log_limit=replay_row_log_limit,
        max_ticks=max_ticks,
        decision_ms=decision_ms,
        alive_reward=alive_reward,
        dead_reward=dead_reward,
        action_selection_mode=action_selection_mode,
        collect_temperature=collect_temperature,
        collect_epsilon=collect_epsilon,
        action_noop_probability=action_noop_probability,
        action_noop_warmup_iterations=action_noop_warmup_iterations,
        policy_action_repeat_min=policy_action_repeat_min,
        policy_action_repeat_max=policy_action_repeat_max,
        policy_action_repeat_extra_probability=policy_action_repeat_extra_probability,
        policy_action_repeat_warmup_iterations=(
            policy_action_repeat_warmup_iterations
        ),
        observation_noise_std=observation_noise_std,
        use_cuda=False,
        checkpoint_every_iterations=checkpoint_every_iterations,
        save_initial_checkpoint=save_initial_checkpoint,
        progress_path=progress_path,
        progress_every_iterations=progress_every_iterations,
        progress_commit_callback=None,
        progress_print=True,
        checkpoint_dir=checkpoint_dir if allow_optimizer_step else None,
        checkpoint_metadata={
            "task_id": TASK_ID,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "checkpoint_root_ref": checkpoint_ref.as_posix(),
            "modal_app": APP_NAME,
            "compute": COMPUTE_CPU,
        },
        require_installed_lightzero=True,
    )
    summary_path = runs.volume_path(RUNS_MOUNT, train_ref / "summary.json")
    runs.write_json(summary_path, result)
    result["summary_ref"] = runs.file_ref(summary_path, mount=RUNS_MOUNT)
    result["progress_ref"] = progress_ref.as_posix()
    result["checkpoint_root_ref"] = checkpoint_ref.as_posix()
    result["volume_name"] = VOLUME_NAME
    if output == "summary":
        payload = compact_curvytron_two_seat_lightzero_train_smoke_summary(result)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif output == "full":
        print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=12 * 60 * 60,
    cpu=40.0,
    memory=65536,
    gpu=CHEAP_GPU_RESOURCE,
)
def lightzero_curvytron_two_seat_train_smoke_gpu(
    seed: int = 0,
    batch_size: int = 1,
    steps: int = 4,
    outer_iterations: int = 1,
    collect_steps_per_iteration: int | None = None,
    updates_per_iteration: int | None = None,
    num_simulations: int = 2,
    learner_updates: int = 1,
    allow_optimizer_step: bool = False,
    replay_scope: str = "current_iteration",
    learner_sample_size: int | None = None,
    max_replay_rows: int | None = 4096,
    record_log_limit: int = 512,
    replay_row_log_limit: int = 256,
    max_ticks: int | None = None,
    decision_ms: float = 300.0,
    alive_reward: float = 1.0,
    dead_reward: float = 0.0,
    action_selection_mode: str = "collect",
    collect_temperature: float = 1.0,
    collect_epsilon: float = 0.25,
    action_noop_probability: float = 0.0,
    action_noop_warmup_iterations: int = 0,
    policy_action_repeat_min: int = DEFAULT_POLICY_ACTION_REPEAT_MIN,
    policy_action_repeat_max: int = DEFAULT_POLICY_ACTION_REPEAT_MAX,
    policy_action_repeat_extra_probability: float = (
        DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
    ),
    policy_action_repeat_warmup_iterations: int = (
        DEFAULT_POLICY_ACTION_REPEAT_WARMUP_ITERATIONS
    ),
    observation_noise_std: float = 0.0,
    checkpoint_every_iterations: int = DEFAULT_CHECKPOINT_EVERY_ITERATIONS,
    save_initial_checkpoint: bool = False,
    progress_every_iterations: int = DEFAULT_PROGRESS_EVERY_ITERATIONS,
    run_id: str = "curvytron-two-seat-smoke-s0",
    attempt_id: str = "two-seat-smoke",
    output: str = "none",
) -> dict[str, Any]:
    checkpoint_ref = runs.checkpoints_root_ref(TASK_ID, run_id) / "lightzero"
    checkpoint_dir = runs.volume_path(RUNS_MOUNT, checkpoint_ref)
    train_ref = runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
    progress_ref = train_ref / "progress.jsonl"
    progress_path = runs.volume_path(RUNS_MOUNT, progress_ref)
    result = run_curvytron_two_seat_lightzero_train_smoke(
        seed=seed,
        batch_size=batch_size,
        steps=steps,
        outer_iterations=outer_iterations,
        collect_steps_per_iteration=collect_steps_per_iteration,
        updates_per_iteration=updates_per_iteration,
        num_simulations=num_simulations,
        learner_updates=learner_updates,
        allow_optimizer_step=allow_optimizer_step,
        replay_scope=replay_scope,
        learner_sample_size=learner_sample_size,
        max_replay_rows=max_replay_rows,
        record_log_limit=record_log_limit,
        replay_row_log_limit=replay_row_log_limit,
        max_ticks=max_ticks,
        decision_ms=decision_ms,
        alive_reward=alive_reward,
        dead_reward=dead_reward,
        action_selection_mode=action_selection_mode,
        collect_temperature=collect_temperature,
        collect_epsilon=collect_epsilon,
        action_noop_probability=action_noop_probability,
        action_noop_warmup_iterations=action_noop_warmup_iterations,
        policy_action_repeat_min=policy_action_repeat_min,
        policy_action_repeat_max=policy_action_repeat_max,
        policy_action_repeat_extra_probability=policy_action_repeat_extra_probability,
        policy_action_repeat_warmup_iterations=(
            policy_action_repeat_warmup_iterations
        ),
        observation_noise_std=observation_noise_std,
        use_cuda=True,
        checkpoint_every_iterations=checkpoint_every_iterations,
        save_initial_checkpoint=save_initial_checkpoint,
        progress_path=progress_path,
        progress_every_iterations=progress_every_iterations,
        progress_commit_callback=None,
        progress_print=True,
        checkpoint_dir=checkpoint_dir if allow_optimizer_step else None,
        checkpoint_metadata={
            "task_id": TASK_ID,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "checkpoint_root_ref": checkpoint_ref.as_posix(),
            "modal_app": APP_NAME,
            "compute": COMPUTE_GPU,
        },
        require_installed_lightzero=True,
    )
    summary_path = runs.volume_path(RUNS_MOUNT, train_ref / "summary.json")
    runs.write_json(summary_path, result)
    result["summary_ref"] = runs.file_ref(summary_path, mount=RUNS_MOUNT)
    result["progress_ref"] = progress_ref.as_posix()
    result["checkpoint_root_ref"] = checkpoint_ref.as_posix()
    result["volume_name"] = VOLUME_NAME
    if output == "summary":
        payload = compact_curvytron_two_seat_lightzero_train_smoke_summary(result)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif output == "full":
        print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def main(
    compute: str = COMPUTE_GPU,
    seed: int = 0,
    batch_size: int = 1,
    steps: int = 4,
    outer_iterations: int = 1,
    collect_steps_per_iteration: int | None = None,
    updates_per_iteration: int | None = None,
    num_simulations: int = 2,
    learner_updates: int = 1,
    allow_optimizer_step: bool = False,
    replay_scope: str = "current_iteration",
    learner_sample_size: int | None = None,
    max_replay_rows: int | None = 4096,
    record_log_limit: int = 512,
    replay_row_log_limit: int = 256,
    max_ticks: int | None = None,
    decision_ms: float = 300.0,
    alive_reward: float = 1.0,
    dead_reward: float = 0.0,
    action_selection_mode: str = "collect",
    collect_temperature: float = 1.0,
    collect_epsilon: float = 0.25,
    action_noop_probability: float = 0.0,
    action_noop_warmup_iterations: int = 0,
    policy_action_repeat_min: int = DEFAULT_POLICY_ACTION_REPEAT_MIN,
    policy_action_repeat_max: int = DEFAULT_POLICY_ACTION_REPEAT_MAX,
    policy_action_repeat_extra_probability: float = (
        DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
    ),
    policy_action_repeat_warmup_iterations: int = (
        DEFAULT_POLICY_ACTION_REPEAT_WARMUP_ITERATIONS
    ),
    observation_noise_std: float = 0.0,
    checkpoint_every_iterations: int = DEFAULT_CHECKPOINT_EVERY_ITERATIONS,
    save_initial_checkpoint: bool = False,
    progress_every_iterations: int = DEFAULT_PROGRESS_EVERY_ITERATIONS,
    run_id: str = "curvytron-two-seat-smoke-s0",
    attempt_id: str = "two-seat-smoke",
    output: str = "summary",
    remote_output: str = "none",
    background: bool = False,
    parallel_runs: int = 1,
    seed_stride: int = 1,
) -> None:
    if compute not in {COMPUTE_CPU, COMPUTE_GPU}:
        raise ValueError(f"unknown compute: {compute!r}; expected cpu or gpu-l4-t4")
    if parallel_runs < 1:
        raise ValueError("parallel_runs must be >= 1")
    if seed_stride < 1:
        raise ValueError("seed_stride must be >= 1")
    remote_function = (
        lightzero_curvytron_two_seat_train_smoke_gpu
        if compute == COMPUTE_GPU
        else lightzero_curvytron_two_seat_train_smoke
    )
    call_kwargs = {
        "seed": seed,
        "batch_size": batch_size,
        "steps": steps,
        "outer_iterations": outer_iterations,
        "collect_steps_per_iteration": collect_steps_per_iteration,
        "updates_per_iteration": updates_per_iteration,
        "num_simulations": num_simulations,
        "learner_updates": learner_updates,
        "allow_optimizer_step": allow_optimizer_step,
        "replay_scope": replay_scope,
        "learner_sample_size": learner_sample_size,
        "max_replay_rows": max_replay_rows,
        "record_log_limit": record_log_limit,
        "replay_row_log_limit": replay_row_log_limit,
        "max_ticks": max_ticks,
        "decision_ms": decision_ms,
        "alive_reward": alive_reward,
        "dead_reward": dead_reward,
        "action_selection_mode": action_selection_mode,
        "collect_temperature": collect_temperature,
        "collect_epsilon": collect_epsilon,
        "action_noop_probability": action_noop_probability,
        "action_noop_warmup_iterations": action_noop_warmup_iterations,
        "policy_action_repeat_min": policy_action_repeat_min,
        "policy_action_repeat_max": policy_action_repeat_max,
        "policy_action_repeat_extra_probability": policy_action_repeat_extra_probability,
        "policy_action_repeat_warmup_iterations": (
            policy_action_repeat_warmup_iterations
        ),
        "observation_noise_std": observation_noise_std,
        "checkpoint_every_iterations": checkpoint_every_iterations,
        "save_initial_checkpoint": save_initial_checkpoint,
        "progress_every_iterations": progress_every_iterations,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "output": remote_output,
    }
    if parallel_runs > 1:
        parallel_kwargs = [
            _parallel_call_kwargs(
                call_kwargs,
                index=index,
                seed_stride=seed_stride,
            )
            for index in range(parallel_runs)
        ]
        if background:
            calls = [
                remote_function.spawn(**one_call_kwargs)
                for one_call_kwargs in parallel_kwargs
            ]
            print(
                json.dumps(
                    {
                        "spawned": True,
                        "parallel_runs": int(parallel_runs),
                        "compute": compute,
                        "function_call_ids": [
                            getattr(call, "object_id", None) for call in calls
                        ],
                        "runs": [
                            _run_refs(one_call_kwargs)
                            for one_call_kwargs in parallel_kwargs
                        ],
                        "volume_name": VOLUME_NAME,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return

        wall_started = time.perf_counter()
        results = list(
            remote_function.starmap(
                [_remote_call_args(one_call_kwargs) for one_call_kwargs in parallel_kwargs],
                order_outputs=False,
                return_exceptions=True,
            )
        )
        payload = _parallel_summary(
            results,
            compute=compute,
            parallel_runs=parallel_runs,
            seed=seed,
            seed_stride=seed_stride,
            wall_elapsed_sec=time.perf_counter() - wall_started,
            full=output == "full",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        if payload["ok_count"] != parallel_runs:
            raise SystemExit(1)
        return

    if background:
        call = remote_function.spawn(**call_kwargs)
        train_ref = runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
        checkpoint_ref = runs.checkpoints_root_ref(TASK_ID, run_id) / "lightzero"
        print(
            json.dumps(
                {
                    "spawned": True,
                    "compute": compute,
                    "function_call_id": getattr(call, "object_id", None),
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "progress_ref": (train_ref / "progress.jsonl").as_posix(),
                    "progress_latest_ref": (
                        train_ref / "progress_latest.json"
                    ).as_posix(),
                    "summary_ref": (train_ref / "summary.json").as_posix(),
                    "checkpoint_root_ref": checkpoint_ref.as_posix(),
                    "volume_name": VOLUME_NAME,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    result = remote_function.remote(**call_kwargs)
    payload = (
        compact_curvytron_two_seat_lightzero_train_smoke_summary(result)
        if output == "summary"
        else result
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


def _remote_call_args(call_kwargs: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(call_kwargs[name] for name in REMOTE_CALL_ARG_NAMES)


def _parallel_call_kwargs(
    call_kwargs: dict[str, Any],
    *,
    index: int,
    seed_stride: int,
) -> dict[str, Any]:
    seed = int(call_kwargs["seed"]) + int(index) * int(seed_stride)
    result = dict(call_kwargs)
    result["seed"] = seed
    result["run_id"] = f"{call_kwargs['run_id']}-p{index:03d}-s{seed}"
    result["attempt_id"] = f"{call_kwargs['attempt_id']}-p{index:03d}"
    return result


def _run_refs(call_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(call_kwargs["run_id"])
    attempt_id = str(call_kwargs["attempt_id"])
    train_ref = runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
    checkpoint_ref = runs.checkpoints_root_ref(TASK_ID, run_id) / "lightzero"
    return {
        "seed": int(call_kwargs["seed"]),
        "run_id": run_id,
        "attempt_id": attempt_id,
        "progress_ref": (train_ref / "progress.jsonl").as_posix(),
        "summary_ref": (train_ref / "summary.json").as_posix(),
        "checkpoint_root_ref": checkpoint_ref.as_posix(),
    }


def _parallel_summary(
    results: list[Any],
    *,
    compute: str,
    parallel_runs: int,
    seed: int,
    seed_stride: int,
    wall_elapsed_sec: float,
    full: bool,
) -> dict[str, Any]:
    ok_results = [result for result in results if isinstance(result, dict) and result.get("ok")]
    dict_results = [result for result in results if isinstance(result, dict)]
    total_steps = sum(int(result.get("steps_survived") or 0) for result in dict_results)
    replay_rows = [
        int(result.get("replay", {}).get("row_count") or 0)
        for result in dict_results
        if isinstance(result.get("replay"), dict)
    ]
    elapsed_values = [
        float(result.get("elapsed_sec"))
        for result in dict_results
        if result.get("elapsed_sec") is not None
    ]
    problems = [
        {
            "index": index,
            "seed": result.get("inputs", {}).get("seed")
            if isinstance(result, dict)
            else None,
            "problems": result.get("problems", [])
            if isinstance(result, dict)
            else [repr(result)],
        }
        for index, result in enumerate(results)
        if not (isinstance(result, dict) and result.get("ok"))
    ]
    return {
        "mode": "parallel_curvytron_two_seat_runs",
        "compute": compute,
        "parallel_runs": int(parallel_runs),
        "seed_start": int(seed),
        "seed_stride": int(seed_stride),
        "wall_elapsed_sec": float(wall_elapsed_sec),
        "ok_count": len(ok_results),
        "result_count": len(results),
        "problem_count": len(problems),
        "problems": problems,
        "total_steps_collected": int(total_steps),
        "total_replay_rows": int(sum(replay_rows)),
        "effective_steps_per_wall_sec": _safe_rate(total_steps, wall_elapsed_sec),
        "effective_replay_rows_per_wall_sec": _safe_rate(sum(replay_rows), wall_elapsed_sec),
        "remote_elapsed_sec_sum": float(sum(elapsed_values)),
        "remote_elapsed_sec_max": max(elapsed_values) if elapsed_values else None,
        "remote_elapsed_sec_mean": (
            float(sum(elapsed_values) / len(elapsed_values)) if elapsed_values else None
        ),
        "results": results
        if full
        else [
            compact_curvytron_two_seat_lightzero_train_smoke_summary(result)
            if isinstance(result, dict)
            else {"ok": False, "error": repr(result)}
            for result in results
        ],
    }


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    denominator_float = float(denominator)
    if denominator_float <= 0.0:
        return None
    return float(numerator) / denominator_float
