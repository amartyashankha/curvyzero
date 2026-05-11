"""Brutally capped stock LightZero CartPole MuZero trainer smoke.

Run the default dry/config-patch smoke from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_cartpole_tiny_train_smoke

This is separate from CurvyZero's trainers and separate from Pong. The default
mode does not call the trainer; it imports the stock LightZero CartPole MuZero
config and reports the tiny CPU patches that would be passed to LightZero's
own train_muzero entrypoint. Use ``--mode train`` only when intentionally
starting the capped remote trainer smoke, and ``--mode progression`` when
intentionally running the slightly longer CPU progression probe that records
learner/evaluator signals and checkpoint artifacts.
"""

from __future__ import annotations

import copy
import contextlib
import io
import importlib
import inspect
import json
import os
import re
import shutil
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs

APP_NAME = "curvyzero-lightzero-cartpole-tiny-train-smoke"
TASK_ID = "lightzero-official-cartpole"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")
LIGHTZERO_VERSION = "0.2.0"

DEFAULT_MAX_ENV_STEP = 4
DEFAULT_MAX_TRAIN_ITER = 1
DEFAULT_COLLECTOR_ENV_NUM = 1
DEFAULT_EVALUATOR_ENV_NUM = 1
DEFAULT_NUM_SIMULATIONS = 2
DEFAULT_BATCH_SIZE = 4
DEFAULT_UPDATE_PER_COLLECT = 1

PROGRESSION_MAX_ENV_STEP = 128
PROGRESSION_MAX_TRAIN_ITER = 4
PROGRESSION_NUM_SIMULATIONS = 5
PROGRESSION_BATCH_SIZE = 16
PROGRESSION_UPDATE_PER_COLLECT = 4

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(f"LightZero=={LIGHTZERO_VERSION}")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App(APP_NAME)


def _version_or_missing(*packages: str) -> str:
    for package in packages:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return "missing"


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _set_path(mapping: Any, path: tuple[str, ...], value: Any) -> dict[str, Any]:
    current = mapping
    for part in path[:-1]:
        current = current[part]
    key = path[-1]
    old_value = current[key]
    current[key] = value
    return {
        "path": ".".join(path),
        "old": _to_plain(old_value),
        "new": _to_plain(value),
    }


def _set_path_if_present(mapping: Any, path: tuple[str, ...], value: Any) -> dict[str, Any]:
    current = mapping
    try:
        for part in path[:-1]:
            current = current[part]
        old_value = current[path[-1]]
    except KeyError:
        return {
            "path": ".".join(path),
            "old": "<missing>",
            "new": _to_plain(value),
            "skipped": True,
        }
    current[path[-1]] = value
    return {
        "path": ".".join(path),
        "old": _to_plain(old_value),
        "new": _to_plain(value),
    }


def _extract_surface(main_config: Any, create_config: Any, *, max_env_step: int) -> dict[str, Any]:
    return {
        "env_id": main_config["env"]["env_id"],
        "policy_type": create_config["policy"]["type"],
        "env_type": create_config["env"]["type"],
        "model_type": main_config["policy"]["model"]["model_type"],
        "observation_shape": _to_plain(main_config["policy"]["model"]["observation_shape"]),
        "action_space_size": main_config["policy"]["model"]["action_space_size"],
        "collector_env_num": main_config["env"]["collector_env_num"],
        "evaluator_env_num": main_config["env"]["evaluator_env_num"],
        "n_evaluator_episode": main_config["env"]["n_evaluator_episode"],
        "num_simulations": main_config["policy"]["num_simulations"],
        "batch_size": main_config["policy"]["batch_size"],
        "update_per_collect": main_config["policy"]["update_per_collect"],
        "n_episode": main_config["policy"].get("n_episode"),
        "eval_freq": main_config["policy"].get("eval_freq"),
        "cuda": main_config["policy"]["cuda"],
        "max_env_step": max_env_step,
        "exp_name": str(main_config["exp_name"]),
    }


def _patched_cartpole_configs(
    *,
    seed: int,
    max_env_step: int,
    collector_env_num: int,
    evaluator_env_num: int,
    num_simulations: int,
    batch_size: int,
    update_per_collect: int,
    eval_freq: int | None,
) -> dict[str, Any]:
    module_name = "zoo.classic_control.cartpole.config.cartpole_muzero_config"
    module = importlib.import_module(module_name)
    main_config = copy.deepcopy(module.main_config)
    create_config = copy.deepcopy(module.create_config)

    original_surface = _extract_surface(
        module.main_config,
        module.create_config,
        max_env_step=int(getattr(module, "max_env_step")),
    )

    patches = [
        _set_path(
            main_config,
            ("exp_name",),
            str(Path("/tmp") / "curvyzero-lightzero-cartpole-tiny" / f"seed-{seed}"),
        ),
        _set_path(main_config, ("env", "collector_env_num"), collector_env_num),
        _set_path(main_config, ("env", "evaluator_env_num"), evaluator_env_num),
        _set_path(main_config, ("env", "n_evaluator_episode"), evaluator_env_num),
        _set_path(main_config, ("policy", "cuda"), False),
        _set_path(main_config, ("policy", "collector_env_num"), collector_env_num),
        _set_path(main_config, ("policy", "evaluator_env_num"), evaluator_env_num),
        _set_path(main_config, ("policy", "n_episode"), 1),
        _set_path(main_config, ("policy", "num_simulations"), num_simulations),
        _set_path(main_config, ("policy", "batch_size"), batch_size),
        _set_path(main_config, ("policy", "update_per_collect"), update_per_collect),
    ]
    if eval_freq is not None:
        patches.append(_set_path_if_present(main_config, ("policy", "eval_freq"), eval_freq))

    patched_surface = _extract_surface(
        main_config,
        create_config,
        max_env_step=max_env_step,
    )
    return {
        "module": module_name,
        "main_config": main_config,
        "create_config": create_config,
        "original_surface": original_surface,
        "patched_surface": patched_surface,
        "patches": patches,
    }


def _validate_patched_surface(surface: dict[str, Any], *, max_train_iter: int, mode: str) -> list[str]:
    problems: list[str] = []
    expected = {
        "env_id": "CartPole-v0",
        "policy_type": "muzero",
        "env_type": "cartpole_lightzero",
        "model_type": "mlp",
        "action_space_size": 2,
        "cuda": False,
    }
    for key, value in expected.items():
        if surface[key] != value:
            problems.append(f"patched CartPole surface {key}={surface[key]!r}, expected {value!r}")
    caps = (
        {
            "max_env_step": PROGRESSION_MAX_ENV_STEP,
            "collector_env_num": 1,
            "evaluator_env_num": 1,
            "n_evaluator_episode": 1,
            "num_simulations": PROGRESSION_NUM_SIMULATIONS,
            "batch_size": PROGRESSION_BATCH_SIZE,
            "update_per_collect": PROGRESSION_UPDATE_PER_COLLECT,
            "n_episode": 1,
        }
        if mode == "progression"
        else {
            "max_env_step": 8,
            "collector_env_num": 1,
            "evaluator_env_num": 1,
            "n_evaluator_episode": 1,
            "num_simulations": 2,
            "batch_size": 8,
            "update_per_collect": 1,
            "n_episode": 1,
        }
    )
    for key, ceiling in caps.items():
        if int(surface[key]) > ceiling:
            problems.append(f"patched CartPole cap {key}={surface[key]!r} exceeds {ceiling}")
    max_train_iter_cap = PROGRESSION_MAX_TRAIN_ITER if mode == "progression" else 1
    if max_train_iter > max_train_iter_cap:
        problems.append(f"max_train_iter={max_train_iter!r} exceeds {mode} cap {max_train_iter_cap}")
    return problems


def _compact_log_tail(text: str, *, limit: int = 60) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def _parse_training_signals(stdout_text: str, stderr_text: str) -> dict[str, Any]:
    text = stdout_text + "\n" + stderr_text
    final_rewards = [float(value) for value in re.findall(r"final reward:\s*([-+]?\d+(?:\.\d+)?)", text)]
    training_iterations = [int(value) for value in re.findall(r"Training Iteration\s+(\d+)\s+Result", text)]
    checkpoint_saves = re.findall(r"learner save ckpt in\s+([^\n]+)", text)
    checkpoint_iterations = sorted(
        {int(value) for value in re.findall(r"iteration_(\d+)\.pth\.tar", "\n".join(checkpoint_saves))}
    )
    table_metrics: dict[str, float] = {}
    pending_names: list[str] | None = None
    for line in text.splitlines():
        parts = [part.strip() for part in line.split("|")]
        parts = [part for part in parts if part]
        if len(parts) >= 2 and parts[0] == "Name":
            pending_names = parts[1:]
        elif pending_names and len(parts) >= 2 and parts[0] == "Value":
            for name, raw_value in zip(pending_names, parts[1:]):
                try:
                    table_metrics[name] = float(raw_value)
                except ValueError:
                    pass
            pending_names = None
    metric_mentions = {
        key: len(re.findall(re.escape(key), text))
        for key in (
            "reward_mean",
            "envstep_count",
            "total_loss_avg",
            "policy_loss_avg",
            "value_loss_avg",
            "target_reward_avg",
        )
    }
    return {
        "final_rewards": final_rewards,
        "training_iterations": training_iterations,
        "checkpoint_iterations": checkpoint_iterations,
        "max_checkpoint_iteration": max(checkpoint_iterations) if checkpoint_iterations else None,
        "table_metrics": table_metrics,
        "checkpoint_saves": checkpoint_saves[-10:],
        "metric_mentions": metric_mentions,
        "stdout_line_count": len(stdout_text.splitlines()),
        "stderr_line_count": len(stderr_text.splitlines()),
        "stdout_tail": _compact_log_tail(stdout_text),
        "stderr_tail": _compact_log_tail(stderr_text, limit=20),
    }


def _scan_one_artifact_root(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"exists": False, "root": str(root), "files": [], "checkpoint_files": []}

    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": stat.st_size,
                    "mtime": round(stat.st_mtime, 3),
                }
            )
    checkpoint_files = [item for item in files if item["path"].endswith((".pth.tar", ".pt", ".pth"))]
    log_files = [item for item in files if item["path"].endswith((".txt", ".log", ".json", ".jsonl"))]
    return {
        "exists": True,
        "root": str(root),
        "file_count": len(files),
        "files_sample": files[:30],
        "checkpoint_files": checkpoint_files,
        "log_files": log_files[:30],
    }


def _scan_training_artifacts(exp_name: str) -> dict[str, Any]:
    root = Path(exp_name)
    roots = [root]
    if exp_name.startswith("/"):
        roots.append(Path("." + exp_name))
    scanned = [_scan_one_artifact_root(candidate) for candidate in roots]
    existing = [item for item in scanned if item["exists"]]
    checkpoint_files = [
        {"root": item["root"], **checkpoint}
        for item in existing
        for checkpoint in item["checkpoint_files"]
    ]
    return {
        "exists": bool(existing),
        "roots": scanned,
        "file_count": sum(int(item.get("file_count", 0)) for item in existing),
        "checkpoint_files": checkpoint_files,
    }


def _write_text(path: Path, text: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _write_run_manifest_once(*, run_id: str, config: dict[str, Any]) -> dict[str, Any]:
    path = runs.volume_path(RUNS_MOUNT, runs.run_manifest_ref(TASK_ID, run_id))
    if path.exists():
        return runs.file_summary(path, mount=RUNS_MOUNT)
    runs.write_json(path, runs.run_manifest(task_id=TASK_ID, run_id=run_id, config=config), exclusive=True)
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _write_attempt_state(
    *,
    run_id: str,
    attempt_id: str,
    status: str,
    started_at: str,
    ended_at: str | None,
    summary_ref: str | None,
    config: dict[str, Any],
    modal_task_id: str | None,
    exclusive: bool = False,
) -> dict[str, Any]:
    path = runs.volume_path(RUNS_MOUNT, runs.attempt_manifest_ref(TASK_ID, run_id, attempt_id))
    payload = runs.attempt_manifest(
        task_id=TASK_ID,
        run_id=run_id,
        attempt_id=attempt_id,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        modal_task_id=modal_task_id,
        summary_ref=summary_ref,
        config=config,
    )
    runs.write_json(path, payload, exclusive=exclusive)
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _write_latest_attempt(
    *,
    run_id: str,
    attempt_id: str,
    status: str,
    started_at: str,
    ended_at: str | None,
    summary_ref: str | None,
    modal_task_id: str | None,
) -> dict[str, Any]:
    path = runs.volume_path(RUNS_MOUNT, runs.latest_attempt_ref(TASK_ID, run_id))
    payload = runs.latest_attempt_pointer(
        task_id=TASK_ID,
        run_id=run_id,
        attempt_id=attempt_id,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        modal_task_id=modal_task_id,
        summary_ref=summary_ref,
    )
    runs.write_json(path, payload)
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _mirror_lightzero_checkpoints(
    *,
    run_id: str,
    artifact_summary: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_root = runs.volume_path(RUNS_MOUNT, runs.checkpoints_root_ref(TASK_ID, run_id)) / "lightzero"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    copied = []
    for item in artifact_summary.get("checkpoint_files", []):
        source = Path(item["root"]) / item["path"]
        if not source.is_file():
            continue
        dest = checkpoint_root / source.name
        shutil.copy2(source, dest)
        copied.append(runs.file_summary(dest, mount=RUNS_MOUNT))
    manifest = {
        "schema": "curvyzero_lightzero_official_checkpoint_manifest/v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "copied_checkpoints": copied,
    }
    manifest_path = checkpoint_root / "manifest.json"
    runs.write_json(manifest_path, manifest)
    return {
        "checkpoint_root_ref": runs.file_ref(checkpoint_root, mount=RUNS_MOUNT),
        "manifest": runs.file_summary(manifest_path, mount=RUNS_MOUNT),
        "copied_checkpoints": copied,
    }


def _persist_run_artifacts(
    *,
    run_id: str,
    attempt_id: str,
    config: dict[str, Any],
    result: dict[str, Any],
    patched: dict[str, Any],
    stdout_text: str,
    stderr_text: str,
    artifact_summary: dict[str, Any],
    checkpoint_mirror: dict[str, Any],
) -> dict[str, Any]:
    attempt_root = runs.volume_path(RUNS_MOUNT, runs.attempt_root_ref(TASK_ID, run_id, attempt_id))
    train_root = runs.volume_path(RUNS_MOUNT, runs.attempt_train_ref(TASK_ID, run_id, attempt_id))
    summary_path = train_root / "summary.json"
    signals_path = train_root / "lightzero_training_signals.json"
    artifacts_path = train_root / "lightzero_artifacts_manifest.json"
    config_path = attempt_root / "config.json"
    command_path = attempt_root / "command.json"
    stdout_path = attempt_root / "stdout_tail.txt"
    stderr_path = attempt_root / "stderr_tail.txt"

    train_result = result.get("train_result") or {}
    runs.write_json(summary_path, result)
    runs.write_json(signals_path, train_result.get("log_signals", {}))
    runs.write_json(artifacts_path, artifact_summary)
    runs.write_json(
        config_path,
        _to_plain(
            {
                "command": config,
                "main_config": patched["main_config"],
                "create_config": patched["create_config"],
            }
        ),
    )
    runs.write_json(command_path, config)
    _write_text(stdout_path, "\n".join(_compact_log_tail(stdout_text)) + ("\n" if stdout_text else ""))
    _write_text(stderr_path, "\n".join(_compact_log_tail(stderr_text, limit=30)) + ("\n" if stderr_text else ""))

    return {
        "summary": runs.file_summary(summary_path, mount=RUNS_MOUNT),
        "training_signals": runs.file_summary(signals_path, mount=RUNS_MOUNT),
        "lightzero_artifacts": runs.file_summary(artifacts_path, mount=RUNS_MOUNT),
        "config": runs.file_summary(config_path, mount=RUNS_MOUNT),
        "command": runs.file_summary(command_path, mount=RUNS_MOUNT),
        "stdout_tail": runs.file_summary(stdout_path, mount=RUNS_MOUNT),
        "stderr_tail": runs.file_summary(stderr_path, mount=RUNS_MOUNT),
        "checkpoint_mirror": checkpoint_mirror,
    }


def _resolve_mode_defaults(
    *,
    mode: str,
    max_env_step: int,
    max_train_iter: int,
    num_simulations: int,
    batch_size: int,
    update_per_collect: int,
) -> dict[str, int]:
    if mode != "progression":
        return {
            "max_env_step": max_env_step,
            "max_train_iter": max_train_iter,
            "num_simulations": num_simulations,
            "batch_size": batch_size,
            "update_per_collect": update_per_collect,
        }
    return {
        "max_env_step": PROGRESSION_MAX_ENV_STEP if max_env_step == DEFAULT_MAX_ENV_STEP else max_env_step,
        "max_train_iter": (
            PROGRESSION_MAX_TRAIN_ITER if max_train_iter == DEFAULT_MAX_TRAIN_ITER else max_train_iter
        ),
        "num_simulations": (
            PROGRESSION_NUM_SIMULATIONS
            if num_simulations == DEFAULT_NUM_SIMULATIONS
            else num_simulations
        ),
        "batch_size": PROGRESSION_BATCH_SIZE if batch_size == DEFAULT_BATCH_SIZE else batch_size,
        "update_per_collect": (
            PROGRESSION_UPDATE_PER_COLLECT
            if update_per_collect == DEFAULT_UPDATE_PER_COLLECT
            else update_per_collect
        ),
    }


def _run_lightzero_cartpole_tiny_smoke(
    *,
    mode: str,
    seed: int,
    max_env_step: int,
    max_train_iter: int,
    collector_env_num: int,
    evaluator_env_num: int,
    num_simulations: int,
    batch_size: int,
    update_per_collect: int,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    problems: list[str] = []
    run_id = run_id or runs.new_run_id("lz-official-cartpole")
    attempt_id = attempt_id or runs.new_attempt_id("attempt")
    started_at = runs.utc_timestamp()
    modal_task_id = os.environ.get("MODAL_TASK_ID")
    artifact_summary: dict[str, Any] = {"exists": False, "roots": [], "file_count": 0, "checkpoint_files": []}
    checkpoint_mirror: dict[str, Any] | None = None
    artifact_refs: dict[str, Any] | None = None
    stdout_text = ""
    stderr_text = ""
    packages = {
        "LightZero": _version_or_missing("LightZero", "lightzero"),
        "DI-engine": _version_or_missing("DI-engine", "ding"),
        "torch": _version_or_missing("torch"),
        "easydict": _version_or_missing("easydict"),
    }

    if mode not in {"dry", "train", "progression"}:
        raise ValueError(f"unknown mode: {mode!r}; expected 'dry', 'train', or 'progression'")

    resolved = _resolve_mode_defaults(
        mode=mode,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
    )
    max_env_step = resolved["max_env_step"]
    max_train_iter = resolved["max_train_iter"]
    num_simulations = resolved["num_simulations"]
    batch_size = resolved["batch_size"]
    update_per_collect = resolved["update_per_collect"]
    config = {
        "job_kind": "lightzero_official_cartpole_muzero_tiny_train",
        "official_example": "zoo.classic_control.cartpole.config.cartpole_muzero_config",
        "algorithm": "LightZero MuZero",
        "mode": mode,
        "seed": seed,
        "max_env_step": max_env_step,
        "max_train_iter": max_train_iter,
        "collector_env_num": collector_env_num,
        "evaluator_env_num": evaluator_env_num,
        "num_simulations": num_simulations,
        "batch_size": batch_size,
        "update_per_collect": update_per_collect,
    }
    _write_run_manifest_once(run_id=run_id, config=config)
    _write_attempt_state(
        run_id=run_id,
        attempt_id=attempt_id,
        status="running",
        started_at=started_at,
        ended_at=None,
        summary_ref=None,
        config=config,
        modal_task_id=modal_task_id,
        exclusive=True,
    )
    _write_latest_attempt(
        run_id=run_id,
        attempt_id=attempt_id,
        status="running",
        started_at=started_at,
        ended_at=None,
        summary_ref=None,
        modal_task_id=modal_task_id,
    )

    patched = _patched_cartpole_configs(
        seed=seed,
        max_env_step=max_env_step,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        eval_freq=1 if mode == "progression" else None,
    )
    patched_surface = patched["patched_surface"]
    problems.extend(_validate_patched_surface(patched_surface, max_train_iter=max_train_iter, mode=mode))

    entry_module = importlib.import_module("lzero.entry")
    train_muzero = entry_module.train_muzero
    trainer_signature = str(inspect.signature(train_muzero))
    train_result: dict[str, Any] | None = None

    if mode in {"train", "progression"} and not problems:
        train_started = time.perf_counter()
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                output = train_muzero(
                    [patched["main_config"], patched["create_config"]],
                    seed=seed,
                    model_path=patched["main_config"]["policy"]["model_path"],
                    max_train_iter=max_train_iter,
                    max_env_step=max_env_step,
                )
            time.sleep(1.0)
            stdout_text = stdout_buffer.getvalue()
            stderr_text = stderr_buffer.getvalue()
            log_signals = _parse_training_signals(stdout_text, stderr_text)
            artifact_summary = _scan_training_artifacts(str(patched["main_config"]["exp_name"]))
            train_result = {
                "ok": True,
                "return_type": type(output).__name__,
                "elapsed_sec": round(time.perf_counter() - train_started, 6),
                "log_signals": log_signals,
                "artifact_summary": artifact_summary,
            }
        except Exception as exc:  # pragma: no cover - remote trainer diagnosis.
            stdout_text = stdout_buffer.getvalue()
            stderr_text = stderr_buffer.getvalue()
            problems.append(f"stock LightZero train_muzero failed: {type(exc).__name__}: {exc}")
            train_result = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_sec": round(time.perf_counter() - train_started, 6),
                "log_signals": _parse_training_signals(stdout_text, stderr_text),
                "artifact_summary": _scan_training_artifacts(str(patched["main_config"]["exp_name"])),
            }
            artifact_summary = train_result["artifact_summary"]

    if mode in {"train", "progression"} and train_result:
        if train_result.get("ok") and not artifact_summary.get("checkpoint_files"):
            problems.append("no LightZero checkpoint artifacts were discovered")
        checkpoint_mirror = _mirror_lightzero_checkpoints(
            run_id=run_id,
            artifact_summary=artifact_summary,
        )
        if train_result.get("ok") and not checkpoint_mirror.get("copied_checkpoints"):
            problems.append("no LightZero checkpoints were mirrored to curvyzero-runs")

    result = {
        "ok": not problems,
        "label": "stock LightZero CartPole MuZero tiny trainer smoke",
        "task_id": TASK_ID,
        "mode": mode,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "call_policy": (
            "dry_config_patch_only"
            if mode == "dry"
            else "calls_stock_lzero.entry.train_muzero_with_cpu_caps"
        ),
        "problems": problems,
        "packages": packages,
        "stock_example": {
            "task": "CartPole-v0",
            "algorithm": "MuZero",
            "module": patched["module"],
            "trainer_entrypoint": "lzero.entry.train_muzero",
            "trainer_signature": trainer_signature,
            "original_surface": patched["original_surface"],
            "patched_surface": patched_surface,
            "patches": patched["patches"],
            "trainer_args": {
                "seed": seed,
                "max_train_iter": max_train_iter,
                "max_env_step": max_env_step,
            },
        },
        "train_result": train_result,
        "checkpoint_mirror": checkpoint_mirror,
        "note": (
            "This module is only for the stock LightZero CartPole path. It is "
            "not CurvyZero's trainer and it is not the LightZero Pong path."
        ),
        "remote_elapsed_sec": round(time.perf_counter() - started, 6),
    }
    if mode in {"train", "progression"}:
        artifact_refs = _persist_run_artifacts(
            run_id=run_id,
            attempt_id=attempt_id,
            config=config,
            result=_to_plain(result),
            patched=patched,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            artifact_summary=artifact_summary,
            checkpoint_mirror=checkpoint_mirror or {},
        )
        result["artifact_refs"] = artifact_refs
    ended_at = runs.utc_timestamp()
    status = "completed" if result["ok"] else "failed"
    summary_ref = None
    if artifact_refs:
        summary_ref = artifact_refs["summary"]["ref"]
    attempt_manifest = _write_attempt_state(
        run_id=run_id,
        attempt_id=attempt_id,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        summary_ref=summary_ref,
        config=config,
        modal_task_id=modal_task_id,
    )
    latest_attempt = _write_latest_attempt(
        run_id=run_id,
        attempt_id=attempt_id,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        summary_ref=summary_ref,
        modal_task_id=modal_task_id,
    )
    result["status"] = status
    result["summary_ref"] = summary_ref
    result["attempt_manifest"] = attempt_manifest
    result["latest_attempt"] = latest_attempt
    runs_volume.commit()
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=8 * 60)
def lightzero_cartpole_tiny_train_smoke(
    mode: str = "dry",
    seed: int = 0,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    return _run_lightzero_cartpole_tiny_smoke(
        mode=mode,
        seed=seed,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        run_id=run_id,
        attempt_id=attempt_id,
    )


@app.local_entrypoint()
def main(
    mode: str = "dry",
    seed: int = 0,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> None:
    result = lightzero_cartpole_tiny_train_smoke.remote(
        mode=mode,
        seed=seed,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
