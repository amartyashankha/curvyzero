"""Brutally capped stock LightZero Atari Pong MuZero trainer smoke.

Run the default dry/config-patch smoke from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke

Run the tiny trainer intentionally:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke --mode train

Run the bounded L4/T4 trainer intentionally:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke --compute gpu-l4-t4 --mode train

This is the stock ALE ``PongNoFrameskip-v4`` visual path via LightZero's
official ``zoo.atari.config.atari_muzero_config`` and ``train_muzero``. It is
mechanical infrastructure only: one collector/evaluator env, tiny MCTS, tiny
batch, and short Atari episode caps. It is not a quality run.
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
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_atari_rom_image import (
    ATARI_ROM_LICENSE_NOTICE,
    build_lightzero_atari_rom_image,
)
from curvyzero.infra.modal.lightzero_pong_dry_config_smoke import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTOR_ENV_NUM,
    DEFAULT_ENV_ID,
    DEFAULT_EVALUATOR_ENV_NUM,
    DEFAULT_MAX_ENV_STEP,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_SEED,
    DEFAULT_UPDATE_PER_COLLECT,
    LIGHTZERO_VERSION,
)

APP_NAME = "curvyzero-lightzero-pong-tiny-train-smoke"
TASK_ID = "lightzero-official-visual-pong"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

DEFAULT_MODE = "dry"
DEFAULT_COMPUTE = "cpu"
DEFAULT_MAX_TRAIN_ITER = 1
DEFAULT_MAX_EPISODE_STEPS = 64
DEFAULT_GAME_SEGMENT_LENGTH = 16
STOCK_AUTO_UPDATE_PER_COLLECT_SENTINEL = -1
CHEAP_GPU_RESOURCE = ["L4", "T4"]

image = (
    build_lightzero_atari_rom_image(lightzero_version=LIGHTZERO_VERSION)
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


def _runtime_compute_summary(*, requested_compute: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "requested_compute": requested_compute,
        "cheap_gpu_resource": CHEAP_GPU_RESOURCE,
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        summary.update(
            {
                "torch_cuda_available": cuda_available,
                "torch_cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            }
        )
        if cuda_available:
            current_device = int(torch.cuda.current_device())
            summary.update(
                {
                    "torch_cuda_current_device": current_device,
                    "torch_cuda_device_name": torch.cuda.get_device_name(current_device),
                    "torch_cuda_capability": list(torch.cuda.get_device_capability(current_device)),
                }
            )
    except Exception as exc:  # pragma: no cover - remote runtime diagnosis only.
        summary["torch_cuda_probe_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def _set_path(mapping: Any, path: tuple[str, ...], value: Any) -> dict[str, Any]:
    current = mapping
    for part in path[:-1]:
        current = current[part]
    key = path[-1]
    old_value = current[key]
    current[key] = value
    return {"path": ".".join(path), "old": _to_plain(old_value), "new": _to_plain(value)}


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
    return {"path": ".".join(path), "old": _to_plain(old_value), "new": _to_plain(value)}


def _set_path_creating_dicts(mapping: Any, path: tuple[str, ...], value: Any) -> dict[str, Any]:
    try:
        from easydict import EasyDict
    except Exception:  # pragma: no cover - only used inside LightZero image.
        EasyDict = dict  # type: ignore[assignment]

    current = mapping
    created: list[str] = []
    for index, part in enumerate(path[:-1]):
        if part not in current or current[part] is None:
            current[part] = EasyDict()
            created.append(".".join(path[: index + 1]))
        current = current[part]
    key = path[-1]
    old_value = current.get(key, "<missing>") if hasattr(current, "get") else "<missing>"
    current[key] = value
    result = {"path": ".".join(path), "old": _to_plain(old_value), "new": _to_plain(value)}
    if created:
        result["created"] = created
    return result


def _exception_result(exc: BaseException) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _compact_log_tail(text: str, *, limit: int = 80) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def _parse_training_signals(stdout_text: str, stderr_text: str) -> dict[str, Any]:
    text = stdout_text + "\n" + stderr_text
    checkpoint_saves = re.findall(r"learner save ckpt in\s+([^\n]+)", text)
    checkpoint_iterations = sorted(
        {int(value) for value in re.findall(r"iteration_(\d+)\.pth\.tar", "\n".join(checkpoint_saves))}
    )
    training_iterations = [int(value) for value in re.findall(r"Training Iteration\s+(\d+)", text)]
    final_rewards = [float(value) for value in re.findall(r"final reward:\s*([-+]?\d+(?:\.\d+)?)", text)]
    metric_mentions = {
        key: len(re.findall(re.escape(key), text))
        for key in (
            "reward_mean",
            "envstep_count",
            "total_loss_avg",
            "policy_loss_avg",
            "value_loss_avg",
            "target_reward_avg",
            "collect episode number",
        )
    }
    return {
        "training_iterations": training_iterations,
        "checkpoint_iterations": checkpoint_iterations,
        "max_checkpoint_iteration": max(checkpoint_iterations) if checkpoint_iterations else None,
        "checkpoint_saves": checkpoint_saves[-10:],
        "final_rewards": final_rewards,
        "metric_mentions": metric_mentions,
        "stdout_line_count": len(stdout_text.splitlines()),
        "stderr_line_count": len(stderr_text.splitlines()),
        "stdout_tail": _compact_log_tail(stdout_text),
        "stderr_tail": _compact_log_tail(stderr_text, limit=30),
    }


def _scan_one_artifact_root(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    if not root.exists():
        return {
            "root": str(root),
            "exists": False,
            "file_count": 0,
            "files_sample": [],
            "checkpoint_files": [],
            "log_files": [],
        }
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            relative_path = path.relative_to(root)
            files.append(
                {
                    "path": relative_path.as_posix(),
                    "size_bytes": stat.st_size,
                }
            )
    checkpoint_files = [
        item for item in files if item["path"].endswith((".pth.tar", ".pt", ".pth"))
    ]
    log_files = [
        item
        for item in files
        if item["path"].endswith((".txt", ".log", ".json", ".jsonl"))
        or "events.out.tfevents" in item["path"]
    ]
    return {
        "root": str(root),
        "exists": True,
        "file_count": len(files),
        "files_sample": files[:60],
        "checkpoint_files": checkpoint_files,
        "log_files": log_files[:60],
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
        "schema": "curvyzero_lightzero_official_visual_pong_checkpoint_manifest/v1",
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


def _extract_surface(main_config: Any, create_config: Any, *, max_env_step: int, max_train_iter: int) -> dict[str, Any]:
    return {
        "env_id": main_config["env"]["env_id"],
        "policy_type": create_config["policy"]["type"],
        "env_type": create_config["env"]["type"],
        "env_manager_type": create_config["env_manager"]["type"],
        "model_type": main_config["policy"]["model"]["model_type"],
        "observation_shape": _to_plain(main_config["policy"]["model"]["observation_shape"]),
        "action_space_size": main_config["policy"]["model"]["action_space_size"],
        "collector_env_num": main_config["env"]["collector_env_num"],
        "evaluator_env_num": main_config["env"]["evaluator_env_num"],
        "n_evaluator_episode": main_config["env"].get("n_evaluator_episode"),
        "collect_max_episode_steps": main_config["env"].get("collect_max_episode_steps"),
        "eval_max_episode_steps": main_config["env"].get("eval_max_episode_steps"),
        "num_simulations": main_config["policy"]["num_simulations"],
        "batch_size": main_config["policy"]["batch_size"],
        "update_per_collect": main_config["policy"].get("update_per_collect"),
        "n_episode": main_config["policy"].get("n_episode"),
        "game_segment_length": main_config["policy"].get("game_segment_length"),
        "eval_freq": main_config["policy"].get("eval_freq"),
        "cuda": main_config["policy"]["cuda"],
        "max_env_step": max_env_step,
        "max_train_iter": max_train_iter,
        "exp_name": str(main_config["exp_name"]),
    }


def _patched_stock_atari_pong_configs(
    *,
    env_id: str,
    seed: int,
    run_id: str,
    attempt_id: str,
    max_env_step: int,
    max_train_iter: int,
    collector_env_num: int,
    evaluator_env_num: int,
    num_simulations: int,
    batch_size: int,
    update_per_collect: int,
    max_episode_steps: int,
    game_segment_length: int,
    use_cuda: bool,
) -> dict[str, Any]:
    module_name = "zoo.atari.config.atari_muzero_config"
    module = importlib.import_module(module_name)
    action_map_module = importlib.import_module("zoo.atari.config.atari_env_action_space_map")
    action_map = getattr(action_map_module, "atari_env_action_space_map")
    main_config = copy.deepcopy(module.main_config)
    create_config = copy.deepcopy(module.create_config)
    original_surface = _extract_surface(
        module.main_config,
        module.create_config,
        max_env_step=int(getattr(module, "max_env_step")),
        max_train_iter=int(1e10),
    )
    exp_name = Path("/tmp") / "curvyzero-lightzero-visual-pong" / run_id / attempt_id / "exp"
    patched_update_per_collect = (
        None
        if update_per_collect == STOCK_AUTO_UPDATE_PER_COLLECT_SENTINEL
        else update_per_collect
    )
    patches = [
        _set_path(main_config, ("exp_name",), str(exp_name)),
        _set_path(main_config, ("env", "env_id"), env_id),
        _set_path(main_config, ("env", "collector_env_num"), collector_env_num),
        _set_path(main_config, ("env", "evaluator_env_num"), evaluator_env_num),
        _set_path_if_present(main_config, ("env", "n_evaluator_episode"), evaluator_env_num),
        _set_path_creating_dicts(main_config, ("env", "collect_max_episode_steps"), max_episode_steps),
        _set_path_creating_dicts(main_config, ("env", "eval_max_episode_steps"), max_episode_steps),
        _set_path_if_present(main_config, ("policy", "collector_env_num"), collector_env_num),
        _set_path_if_present(main_config, ("policy", "evaluator_env_num"), evaluator_env_num),
        _set_path_if_present(main_config, ("policy", "n_episode"), collector_env_num),
        _set_path(main_config, ("policy", "cuda"), use_cuda),
        _set_path(main_config, ("policy", "model", "action_space_size"), action_map[env_id]),
        _set_path(main_config, ("policy", "num_simulations"), num_simulations),
        _set_path(main_config, ("policy", "batch_size"), batch_size),
        _set_path_if_present(main_config, ("policy", "update_per_collect"), patched_update_per_collect),
        _set_path_if_present(main_config, ("policy", "game_segment_length"), game_segment_length),
        _set_path_if_present(main_config, ("policy", "eval_freq"), 1),
        _set_path_creating_dicts(
            main_config,
            ("policy", "learn", "learner", "hook", "save_ckpt_after_iter"),
            1,
        ),
    ]
    return {
        "module": module_name,
        "main_config": main_config,
        "create_config": create_config,
        "original_surface": original_surface,
        "patched_surface": _extract_surface(
            main_config,
            create_config,
            max_env_step=max_env_step,
            max_train_iter=max_train_iter,
        ),
        "patches": patches,
        "default_env_action_space_size": action_map[env_id],
    }


def _validate_patched_surface(surface: dict[str, Any], *, use_cuda: bool) -> list[str]:
    problems: list[str] = []
    expected = {
        "env_id": DEFAULT_ENV_ID,
        "policy_type": "muzero",
        "env_type": "atari_lightzero",
        "model_type": "conv",
        "action_space_size": 6,
        "cuda": use_cuda,
    }
    for key, value in expected.items():
        if surface[key] != value:
            problems.append(f"patched Pong surface {key}={surface[key]!r}, expected {value!r}")
    caps = {
        "max_env_step": 8192,
        "max_train_iter": 64,
        "collector_env_num": 4,
        "evaluator_env_num": 1,
        "n_evaluator_episode": 1,
        "collect_max_episode_steps": 1024,
        "eval_max_episode_steps": 1024,
        "num_simulations": 25,
        "batch_size": 64,
        "game_segment_length": 128,
    }
    for key, ceiling in caps.items():
        if surface[key] is not None and int(surface[key]) > ceiling:
            problems.append(f"patched Pong cap {key}={surface[key]!r} exceeds {ceiling}")
    update_per_collect = surface.get("update_per_collect")
    if update_per_collect is not None and int(update_per_collect) > 4:
        problems.append(f"patched Pong cap update_per_collect={update_per_collect!r} exceeds 4")
    if surface["env_manager_type"] != "subprocess":
        problems.append(f"unexpected env_manager_type={surface['env_manager_type']!r}")
    return problems


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


def _run_lightzero_pong_tiny_train_smoke(
    *,
    mode: str,
    compute: str,
    env_id: str,
    seed: int,
    max_env_step: int,
    max_train_iter: int,
    collector_env_num: int,
    evaluator_env_num: int,
    num_simulations: int,
    batch_size: int,
    update_per_collect: int,
    max_episode_steps: int,
    game_segment_length: int,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    problems: list[str] = []
    if mode not in {"dry", "train"}:
        raise ValueError(f"unknown mode: {mode!r}; expected 'dry' or 'train'")
    if compute not in {"cpu", "gpu-l4-t4"}:
        raise ValueError(f"unknown compute: {compute!r}; expected 'cpu' or 'gpu-l4-t4'")
    use_cuda = compute == "gpu-l4-t4"
    run_id = run_id or runs.new_run_id("lz-visual-pong")
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
        "gym": _version_or_missing("gym"),
        "gymnasium": _version_or_missing("gymnasium"),
        "ale-py": _version_or_missing("ale-py", "ale_py"),
        "opencv-python-headless": _version_or_missing("opencv-python-headless"),
        "AutoROM": _version_or_missing("AutoROM"),
    }
    runtime_compute = _runtime_compute_summary(requested_compute=compute)
    config = {
        "job_kind": "lightzero_official_visual_pong_muzero_tiny_train",
        "official_example": "zoo.atari.config.atari_muzero_config",
        "algorithm": "LightZero MuZero",
        "mode": mode,
        "compute": compute,
        "use_cuda": use_cuda,
        "env_id": env_id,
        "seed": seed,
        "max_env_step": max_env_step,
        "max_train_iter": max_train_iter,
        "collector_env_num": collector_env_num,
        "evaluator_env_num": evaluator_env_num,
        "num_simulations": num_simulations,
        "batch_size": batch_size,
        "update_per_collect": update_per_collect,
        "update_per_collect_note": (
            "CLI value -1 means restore stock LightZero update_per_collect=None "
            "and use the config replay_ratio accounting."
            if update_per_collect == STOCK_AUTO_UPDATE_PER_COLLECT_SENTINEL
            else "explicit update_per_collect cap"
        ),
        "max_episode_steps": max_episode_steps,
        "game_segment_length": game_segment_length,
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

    patched = _patched_stock_atari_pong_configs(
        env_id=env_id,
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        max_episode_steps=max_episode_steps,
        game_segment_length=game_segment_length,
        use_cuda=use_cuda,
    )
    patched_surface = patched["patched_surface"]
    problems.extend(_validate_patched_surface(patched_surface, use_cuda=use_cuda))

    entry_module = importlib.import_module("lzero.entry")
    train_muzero = entry_module.train_muzero
    trainer_signature = str(inspect.signature(train_muzero))
    train_result: dict[str, Any] | None = None

    if mode == "train" and not problems:
        train_started = time.perf_counter()
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                output = train_muzero(
                    [patched["main_config"], patched["create_config"]],
                    seed=seed,
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
            artifact_summary = _scan_training_artifacts(str(patched["main_config"]["exp_name"]))
            problems.append(f"stock LightZero Atari Pong train_muzero failed: {type(exc).__name__}: {exc}")
            train_result = {
                "ok": False,
                "elapsed_sec": round(time.perf_counter() - train_started, 6),
                "log_signals": _parse_training_signals(stdout_text, stderr_text),
                "artifact_summary": artifact_summary,
            }
            train_result.update(_exception_result(exc))

    if mode == "train":
        checkpoint_mirror = _mirror_lightzero_checkpoints(
            run_id=run_id,
            artifact_summary=artifact_summary,
        )

    result = {
        "ok": not problems and (mode == "dry" or bool(train_result and train_result.get("ok"))),
        "label": "stock LightZero Atari Pong MuZero tiny trainer smoke",
        "task_id": TASK_ID,
        "mode": mode,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "compute": compute,
        "use_cuda": use_cuda,
        "call_policy": (
            "dry_config_patch_only"
            if mode == "dry"
            else f"calls_stock_lzero.entry.train_muzero_with_{compute}_caps"
        ),
        "problems": problems,
        "packages": packages,
        "runtime_compute": runtime_compute,
        "stock_example": {
            "task": env_id,
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
        "lightzero_artifacts": artifact_summary,
        "checkpoint_mirror": checkpoint_mirror,
        "rom_unblocker": {
            "license_acceptance": ATARI_ROM_LICENSE_NOTICE,
            "modal_image_step": [
                'uv_pip_install("AutoROM[accept-rom-license]==0.6.1")',
                'run_commands("AutoROM --accept-license")',
            ],
        },
        "note": (
            "This is a trainer infrastructure smoke for stock ALE visual Pong. "
            "It caps Atari episode length and training budget aggressively; "
            "success means the official visual train path runs, not that the "
            "policy learned Pong."
        ),
        "remote_elapsed_sec": round(time.perf_counter() - started, 6),
    }
    if mode == "train":
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


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=8 * 60, cpu=1.0)
def lightzero_pong_tiny_train_smoke(
    mode: str = DEFAULT_MODE,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    return _run_lightzero_pong_tiny_train_smoke(
        mode=mode,
        compute="cpu",
        env_id=env_id,
        seed=seed,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        max_episode_steps=max_episode_steps,
        game_segment_length=game_segment_length,
        run_id=run_id,
        attempt_id=attempt_id,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=60 * 60,
    cpu=2.0,
    gpu=CHEAP_GPU_RESOURCE,
)
def lightzero_pong_tiny_train_smoke_gpu(
    mode: str = DEFAULT_MODE,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    return _run_lightzero_pong_tiny_train_smoke(
        mode=mode,
        compute="gpu-l4-t4",
        env_id=env_id,
        seed=seed,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        max_episode_steps=max_episode_steps,
        game_segment_length=game_segment_length,
        run_id=run_id,
        attempt_id=attempt_id,
    )


@app.local_entrypoint()
def main(
    compute: str = DEFAULT_COMPUTE,
    mode: str = DEFAULT_MODE,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> None:
    if compute == "cpu":
        train_fn = lightzero_pong_tiny_train_smoke
    elif compute == "gpu-l4-t4":
        train_fn = lightzero_pong_tiny_train_smoke_gpu
    else:
        raise ValueError(f"unknown compute: {compute!r}; expected 'cpu' or 'gpu-l4-t4'")
    result = train_fn.remote(
        mode=mode,
        env_id=env_id,
        seed=seed,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        max_episode_steps=max_episode_steps,
        game_segment_length=game_segment_length,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
