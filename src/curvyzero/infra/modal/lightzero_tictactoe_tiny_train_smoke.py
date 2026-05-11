"""Tiny official LightZero TicTacToe MuZero trainer smoke on Modal.

Run the dry config-patch smoke:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_tictactoe_tiny_train_smoke

Run the tiny remote train smoke:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_tictactoe_tiny_train_smoke --mode train

This uses the stock LightZero TicTacToe MuZero bot-mode config. TicTacToe has a
delayed terminal reward and the official config uses td_steps=9 with
discount_factor=1 for final-outcome targets.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import inspect
import io
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

APP_NAME = "curvyzero-lightzero-tictactoe-tiny-train-smoke"
TASK_ID = "lightzero-official-tictactoe"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")
LIGHTZERO_VERSION = "0.2.0"

DEFAULT_MAX_ENV_STEP = 64
DEFAULT_MAX_TRAIN_ITER = 4
DEFAULT_COLLECTOR_ENV_NUM = 1
DEFAULT_EVALUATOR_ENV_NUM = 1
DEFAULT_N_EPISODE = 1
DEFAULT_NUM_SIMULATIONS = 5
DEFAULT_BATCH_SIZE = 4
DEFAULT_UPDATE_PER_COLLECT = 10

PROGRESSION_MAX_ENV_STEP = 64
PROGRESSION_MAX_TRAIN_ITER = 4
PROGRESSION_NUM_SIMULATIONS = 5

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
    old_value = current[path[-1]]
    current[path[-1]] = value
    return {"path": ".".join(path), "old": _to_plain(old_value), "new": _to_plain(value)}


def _set_path_if_present(mapping: Any, path: tuple[str, ...], value: Any) -> dict[str, Any]:
    current = mapping
    try:
        for part in path[:-1]:
            current = current[part]
        old_value = current[path[-1]]
    except KeyError:
        return {"path": ".".join(path), "old": "<missing>", "new": _to_plain(value), "skipped": True}
    current[path[-1]] = value
    return {"path": ".".join(path), "old": _to_plain(old_value), "new": _to_plain(value)}


def _set_or_add_path(mapping: Any, path: tuple[str, ...], value: Any) -> dict[str, Any]:
    current = mapping
    for part in path[:-1]:
        current = current[part]
    old_value = current.get(path[-1], "<missing>")
    current[path[-1]] = value
    return {"path": ".".join(path), "old": _to_plain(old_value), "new": _to_plain(value)}


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
    return {
        "final_rewards": final_rewards,
        "training_iterations": training_iterations,
        "checkpoint_iterations": checkpoint_iterations,
        "max_checkpoint_iteration": max(checkpoint_iterations) if checkpoint_iterations else None,
        "table_metrics": table_metrics,
        "checkpoint_saves": checkpoint_saves[-10:],
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
            stat = path.stat()
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": stat.st_size,
                    "mtime": round(stat.st_mtime, 3),
                }
            )
    return {
        "exists": True,
        "root": str(root),
        "file_count": len(files),
        "files_sample": files[:30],
        "checkpoint_files": [item for item in files if item["path"].endswith((".pth.tar", ".pt", ".pth"))],
        "log_files": [item for item in files if item["path"].endswith((".txt", ".log", ".json", ".jsonl"))][:30],
    }


def _scan_training_artifacts(exp_name: str) -> dict[str, Any]:
    roots = [Path(exp_name)]
    if exp_name.startswith("/"):
        roots.append(Path("." + exp_name))
    scanned = [_scan_one_artifact_root(root) for root in roots]
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


def _extract_surface(main_config: Any, create_config: Any, *, max_env_step: int) -> dict[str, Any]:
    return {
        "env_id": main_config["env"].get("env_id", "TicTacToe"),
        "battle_mode": main_config["env"]["battle_mode"],
        "bot_action_type": main_config["env"].get("bot_action_type", "v0"),
        "policy_type": create_config["policy"]["type"],
        "env_type": create_config["env"]["type"],
        "env_manager_type": create_config["env_manager"]["type"],
        "observation_shape": _to_plain(main_config["policy"]["model"]["observation_shape"]),
        "action_space_size": main_config["policy"]["model"]["action_space_size"],
        "reward_support_range": _to_plain(main_config["policy"]["model"].get("reward_support_range")),
        "value_support_range": _to_plain(main_config["policy"]["model"].get("value_support_range")),
        "collector_env_num": main_config["env"]["collector_env_num"],
        "evaluator_env_num": main_config["env"]["evaluator_env_num"],
        "n_evaluator_episode": main_config["env"]["n_evaluator_episode"],
        "n_episode": main_config["policy"]["n_episode"],
        "num_simulations": main_config["policy"]["num_simulations"],
        "batch_size": main_config["policy"]["batch_size"],
        "update_per_collect": main_config["policy"]["update_per_collect"],
        "td_steps": main_config["policy"]["td_steps"],
        "num_unroll_steps": main_config["policy"]["num_unroll_steps"],
        "discount_factor": main_config["policy"]["discount_factor"],
        "game_segment_length": main_config["policy"]["game_segment_length"],
        "eval_freq": main_config["policy"].get("eval_freq"),
        "cuda": main_config["policy"]["cuda"],
        "max_env_step": max_env_step,
        "exp_name": str(main_config["exp_name"]),
    }


def _patched_tictactoe_configs(
    *,
    seed: int,
    max_env_step: int,
    collector_env_num: int,
    evaluator_env_num: int,
    n_episode: int,
    num_simulations: int,
    batch_size: int,
    update_per_collect: int,
) -> dict[str, Any]:
    module_name = "zoo.board_games.tictactoe.config.tictactoe_muzero_bot_mode_config"
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
            str(Path("/tmp") / "curvyzero-lightzero-tictactoe-tiny" / f"seed-{seed}"),
        ),
        _set_path(main_config, ("env", "collector_env_num"), collector_env_num),
        _set_path(main_config, ("env", "evaluator_env_num"), evaluator_env_num),
        _set_path(main_config, ("env", "n_evaluator_episode"), evaluator_env_num),
        _set_path(main_config, ("env", "battle_mode"), "play_with_bot_mode"),
        _set_or_add_path(main_config, ("env", "env_id"), "TicTacToe"),
        _set_or_add_path(main_config, ("env", "bot_action_type"), "v0"),
        _set_path_if_present(main_config, ("env", "manager", "shared_memory"), False),
        _set_path(main_config, ("policy", "cuda"), False),
        _set_or_add_path(main_config, ("policy", "model", "reward_support_range"), (-10.0, 11.0, 1.0)),
        _set_or_add_path(main_config, ("policy", "model", "value_support_range"), (-10.0, 11.0, 1.0)),
        _set_path(main_config, ("policy", "collector_env_num"), collector_env_num),
        _set_path(main_config, ("policy", "evaluator_env_num"), evaluator_env_num),
        _set_path(main_config, ("policy", "n_episode"), n_episode),
        _set_path(main_config, ("policy", "num_simulations"), num_simulations),
        _set_path(main_config, ("policy", "batch_size"), batch_size),
        _set_path(main_config, ("policy", "update_per_collect"), update_per_collect),
        _set_path(main_config, ("policy", "eval_freq"), 1),
        _set_path(main_config, ("policy", "game_segment_length"), 5),
        _set_path(main_config, ("policy", "td_steps"), 9),
        _set_path(main_config, ("policy", "num_unroll_steps"), 3),
        _set_path(main_config, ("policy", "discount_factor"), 1),
        _set_path_if_present(main_config, ("policy", "use_wandb"), False),
    ]
    return {
        "module": module_name,
        "main_config": main_config,
        "create_config": create_config,
        "original_surface": original_surface,
        "patched_surface": _extract_surface(main_config, create_config, max_env_step=max_env_step),
        "patches": patches,
    }


def _validate_patched_surface(surface: dict[str, Any], *, max_train_iter: int, mode: str) -> list[str]:
    problems: list[str] = []
    expected = {
        "battle_mode": "play_with_bot_mode",
        "bot_action_type": "v0",
        "policy_type": "muzero",
        "env_type": "tictactoe",
        "env_manager_type": "subprocess",
        "action_space_size": 9,
        "td_steps": 9,
        "num_unroll_steps": 3,
        "discount_factor": 1,
        "game_segment_length": 5,
        "num_simulations": 5,
        "batch_size": 4,
        "update_per_collect": 10,
        "cuda": False,
    }
    for key, value in expected.items():
        if surface[key] != value:
            problems.append(f"patched TicTacToe surface {key}={surface[key]!r}, expected {value!r}")
    caps = {
        "max_env_step": PROGRESSION_MAX_ENV_STEP if mode == "progression" else DEFAULT_MAX_ENV_STEP,
        "collector_env_num": 1,
        "evaluator_env_num": 1,
        "n_evaluator_episode": 1,
        "n_episode": 1,
        "num_simulations": DEFAULT_NUM_SIMULATIONS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "update_per_collect": DEFAULT_UPDATE_PER_COLLECT,
    }
    for key, ceiling in caps.items():
        if int(surface[key]) > ceiling:
            problems.append(f"patched TicTacToe cap {key}={surface[key]!r} exceeds {ceiling}")
    if tuple(surface["reward_support_range"]) != (-10.0, 11.0, 1.0):
        problems.append(f"reward_support_range={surface['reward_support_range']!r}, expected (-10, 11, 1)")
    if tuple(surface["value_support_range"]) != (-10.0, 11.0, 1.0):
        problems.append(f"value_support_range={surface['value_support_range']!r}, expected (-10, 11, 1)")
    train_iter_cap = PROGRESSION_MAX_TRAIN_ITER if mode == "progression" else DEFAULT_MAX_TRAIN_ITER
    if max_train_iter > train_iter_cap:
        problems.append(f"max_train_iter={max_train_iter!r} exceeds {mode} cap {train_iter_cap}")
    return problems


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


def _write_text(path: Path, text: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _mirror_lightzero_checkpoints(*, run_id: str, artifact_summary: dict[str, Any]) -> dict[str, Any]:
    checkpoint_root = runs.volume_path(RUNS_MOUNT, runs.checkpoints_root_ref(TASK_ID, run_id)) / "lightzero"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    copied = []
    for item in artifact_summary.get("checkpoint_files", []):
        source = Path(item["root"]) / item["path"]
        if source.is_file():
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
) -> dict[str, int]:
    if mode != "progression":
        return {
            "max_env_step": max_env_step,
            "max_train_iter": max_train_iter,
            "num_simulations": num_simulations,
        }
    return {
        "max_env_step": PROGRESSION_MAX_ENV_STEP if max_env_step == DEFAULT_MAX_ENV_STEP else max_env_step,
        "max_train_iter": (
            PROGRESSION_MAX_TRAIN_ITER if max_train_iter == DEFAULT_MAX_TRAIN_ITER else max_train_iter
        ),
        "num_simulations": (
            PROGRESSION_NUM_SIMULATIONS if num_simulations == DEFAULT_NUM_SIMULATIONS else num_simulations
        ),
    }


def _run_lightzero_tictactoe_tiny_smoke(
    *,
    mode: str,
    seed: int,
    max_env_step: int,
    max_train_iter: int,
    collector_env_num: int,
    evaluator_env_num: int,
    n_episode: int,
    num_simulations: int,
    batch_size: int,
    update_per_collect: int,
    run_id: str | None,
    attempt_id: str | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if mode not in {"dry", "train", "progression"}:
        raise ValueError(f"unknown mode: {mode!r}; expected 'dry', 'train', or 'progression'")
    resolved = _resolve_mode_defaults(
        mode=mode,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        num_simulations=num_simulations,
    )
    max_env_step = resolved["max_env_step"]
    max_train_iter = resolved["max_train_iter"]
    num_simulations = resolved["num_simulations"]

    run_id = run_id or runs.new_run_id("lz-official-tictactoe")
    attempt_id = attempt_id or runs.new_attempt_id("attempt")
    started_at = runs.utc_timestamp()
    modal_task_id = os.environ.get("MODAL_TASK_ID")
    problems: list[str] = []
    stdout_text = ""
    stderr_text = ""
    artifact_summary: dict[str, Any] = {"exists": False, "roots": [], "file_count": 0, "checkpoint_files": []}
    checkpoint_mirror: dict[str, Any] | None = None
    artifact_refs: dict[str, Any] | None = None
    config = {
        "job_kind": "lightzero_official_tictactoe_muzero_tiny_train",
        "official_example": "zoo.board_games.tictactoe.config.tictactoe_muzero_bot_mode_config",
        "algorithm": "LightZero MuZero",
        "environment": "TicTacToe",
        "reward_shape": "delayed_terminal_outcome",
        "mode": mode,
        "seed": seed,
        "max_env_step": max_env_step,
        "max_train_iter": max_train_iter,
        "collector_env_num": collector_env_num,
        "evaluator_env_num": evaluator_env_num,
        "n_episode": n_episode,
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

    patched = _patched_tictactoe_configs(
        seed=seed,
        max_env_step=max_env_step,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_episode=n_episode,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
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
            artifact_summary = _scan_training_artifacts(str(patched["main_config"]["exp_name"]))
            train_result = {
                "ok": True,
                "return_type": type(output).__name__,
                "elapsed_sec": round(time.perf_counter() - train_started, 6),
                "log_signals": _parse_training_signals(stdout_text, stderr_text),
                "artifact_summary": artifact_summary,
            }
        except Exception as exc:  # pragma: no cover - remote trainer diagnosis.
            stdout_text = stdout_buffer.getvalue()
            stderr_text = stderr_buffer.getvalue()
            artifact_summary = _scan_training_artifacts(str(patched["main_config"]["exp_name"]))
            problems.append(f"stock LightZero train_muzero failed: {type(exc).__name__}: {exc}")
            train_result = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_sec": round(time.perf_counter() - train_started, 6),
                "log_signals": _parse_training_signals(stdout_text, stderr_text),
                "artifact_summary": artifact_summary,
            }

    if mode in {"train", "progression"} and train_result:
        if train_result.get("ok") and not artifact_summary.get("checkpoint_files"):
            problems.append("no LightZero checkpoint artifacts were discovered")
        checkpoint_mirror = _mirror_lightzero_checkpoints(run_id=run_id, artifact_summary=artifact_summary)
        if train_result.get("ok") and not checkpoint_mirror.get("copied_checkpoints"):
            problems.append("no LightZero checkpoints were mirrored to curvyzero-runs")

    result = {
        "ok": not problems,
        "label": "stock LightZero TicTacToe MuZero tiny trainer smoke",
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
        "packages": {
            "LightZero": _version_or_missing("LightZero", "lightzero"),
            "DI-engine": _version_or_missing("DI-engine", "ding"),
            "torch": _version_or_missing("torch"),
            "easydict": _version_or_missing("easydict"),
        },
        "stock_example": {
            "task": "TicTacToe",
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
        "reward_sparsity": {
            "source": "zoo.board_games.tictactoe.envs.tictactoe_env.TicTacToeEnv._player_step",
            "shape": "reward is 1.0 only when winner == current_player; draw/non-terminal reward is 0.0",
            "terminal_target_config": {"td_steps": patched_surface["td_steps"], "discount_factor": 1},
        },
        "train_result": train_result,
        "checkpoint_mirror": checkpoint_mirror,
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
    summary_ref = artifact_refs["summary"]["ref"] if artifact_refs else None
    result["status"] = status
    result["summary_ref"] = summary_ref
    result["attempt_manifest"] = _write_attempt_state(
        run_id=run_id,
        attempt_id=attempt_id,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        summary_ref=summary_ref,
        config=config,
        modal_task_id=modal_task_id,
    )
    result["latest_attempt"] = _write_latest_attempt(
        run_id=run_id,
        attempt_id=attempt_id,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        summary_ref=summary_ref,
        modal_task_id=modal_task_id,
    )
    runs_volume.commit()
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=8 * 60)
def lightzero_tictactoe_tiny_train_smoke(
    mode: str = "dry",
    seed: int = 0,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    n_episode: int = DEFAULT_N_EPISODE,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    return _run_lightzero_tictactoe_tiny_smoke(
        mode=mode,
        seed=seed,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_episode=n_episode,
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
    n_episode: int = DEFAULT_N_EPISODE,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> None:
    result = lightzero_tictactoe_tiny_train_smoke.remote(
        mode=mode,
        seed=seed,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_episode=n_episode,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
