"""Model-card-style LightZero MuZeroAgent Atari Pong reproduction on Modal.

This is intentionally separate from ``lightzero_pong_exact_reproduction``. That
module calls ``lzero.entry.train_muzero`` on the installed 64x64
``zoo.atari.config.atari_muzero_config`` surface. This module uses the
``lzero.agent.MuZeroAgent`` API shown on the OpenDILabCommunity
``PongNoFrameskip-v4-MuZero`` model card, whose bundled Agent config is the
older 96x96/downsample visual Atari Pong surface.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import os
import threading
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
from curvyzero.infra.modal.lightzero_pong_dry_config_smoke import LIGHTZERO_VERSION

APP_NAME = "curvyzero-lightzero-pong-muzero-agent-reproduction"
TASK_ID = "lightzero-official-visual-pong-muzero-agent96"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

DEFAULT_MODE = "dry"
DEFAULT_COMPUTE = "cpu"
DEFAULT_ENV_ID = "PongNoFrameskip-v4"
DEFAULT_SEED = 0
DEFAULT_RUN_ID = "lz-visual-pong-muzero-agent96-s0"
DEFAULT_ATTEMPT_ID = "dry-agent96-model-card-surface"
DEFAULT_TRAIN_STEP = 500_000
DEFAULT_PROGRESS_INTERVAL_SEC = 300
DEFAULT_SAVE_CKPT_AFTER_ITER_OVERRIDE: int | None = None

CHEAP_GPU_RESOURCE = ["L4", "T4"]
H100_GPU_RESOURCE = "H100"
GPU_RESOURCE_BY_COMPUTE = {
    "gpu-l4-t4-cpu40": CHEAP_GPU_RESOURCE,
    "gpu-h100-cpu40": H100_GPU_RESOURCE,
}
CPU_COUNT_BY_COMPUTE = {
    "cpu": 1.0,
    "gpu-l4-t4-cpu40": 40.0,
    "gpu-h100-cpu40": 40.0,
}
COMPUTE_CHOICES = ("cpu", *GPU_RESOURCE_BY_COMPUTE)

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


def _exception_result(exc: BaseException) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _set_path_creating_dicts(mapping: Any, path: tuple[str, ...], value: Any) -> dict[str, Any]:
    current = mapping
    for part in path[:-1]:
        if part not in current or current[part] is None:
            current[part] = {}
        current = current[part]
    old_value = current.get(path[-1], "<missing>")
    current[path[-1]] = value
    return {
        "path": ".".join(path),
        "old": _to_plain(old_value),
        "new": _to_plain(value),
        "reason": "Checkpoint cadence observability override for short Agent-path controls.",
    }


def _compute_choices_label() -> str:
    return ", ".join(COMPUTE_CHOICES)


def _extract_agent_surface(cfg: Any, *, train_step: int) -> dict[str, Any]:
    main_config = cfg.main_config
    create_config = cfg.create_config
    policy = main_config.policy
    env = main_config.env
    model = policy.model
    return {
        "env_id": env.env_id,
        "exp_name": str(main_config.exp_name),
        "env_type": create_config.env.type,
        "env_import_names": _to_plain(create_config.env.get("import_names")),
        "env_manager_type": create_config.env_manager.type,
        "policy_type": create_config.policy.type,
        "policy_import_names": _to_plain(create_config.policy.get("import_names")),
        "model_type": model.get("model_type", "conv"),
        "observation_shape": _to_plain(model.observation_shape),
        "env_obs_shape": _to_plain(env.obs_shape),
        "action_space_size": model.action_space_size,
        "downsample": bool(model.get("downsample", False)),
        "self_supervised_learning_loss": bool(model.get("self_supervised_learning_loss", False)),
        "collector_env_num": env.collector_env_num,
        "policy_collector_env_num": policy.collector_env_num,
        "n_episode": policy.n_episode,
        "evaluator_env_num": env.evaluator_env_num,
        "policy_evaluator_env_num": policy.evaluator_env_num,
        "n_evaluator_episode": env.n_evaluator_episode,
        "num_simulations": policy.num_simulations,
        "batch_size": policy.batch_size,
        "update_per_collect": policy.update_per_collect,
        "game_segment_length": policy.game_segment_length,
        "eval_freq": policy.eval_freq,
        "cuda": policy.cuda,
        "learning_rate": policy.learning_rate,
        "replay_buffer_size": policy.replay_buffer_size,
        "reanalyze_ratio": policy.get("reanalyze_ratio"),
        "eps_greedy_exploration_in_collect": policy.eps.eps_greedy_exploration_in_collect,
        "save_ckpt_after_iter": policy.get("learn", {}).get("learner", {}).get("hook", {}).get(
            "save_ckpt_after_iter"
        ),
        "train_step_argument": train_step,
    }


def _validate_agent_surface(surface: dict[str, Any], *, expected_train_step: int) -> list[str]:
    expected = {
        "env_id": "PongNoFrameskip-v4",
        "env_type": "atari_lightzero",
        "env_import_names": ["zoo.atari.envs.atari_lightzero_env"],
        "env_manager_type": "subprocess",
        "policy_type": "muzero",
        "policy_import_names": ["lzero.policy.muzero"],
        "model_type": "conv",
        "observation_shape": [4, 96, 96],
        "env_obs_shape": [4, 96, 96],
        "action_space_size": 6,
        "downsample": True,
        "self_supervised_learning_loss": True,
        "collector_env_num": 8,
        "policy_collector_env_num": 8,
        "n_episode": 8,
        "evaluator_env_num": 3,
        "policy_evaluator_env_num": 3,
        "n_evaluator_episode": 3,
        "num_simulations": 50,
        "batch_size": 256,
        "update_per_collect": 1000,
        "game_segment_length": 400,
        "eval_freq": 2000,
        "cuda": True,
        "learning_rate": 0.2,
        "replay_buffer_size": 1_000_000,
        "reanalyze_ratio": 0.0,
        "eps_greedy_exploration_in_collect": False,
        "train_step_argument": expected_train_step,
    }
    problems = []
    for key, value in expected.items():
        if surface.get(key) != value:
            problems.append(f"MuZeroAgent surface {key}={surface.get(key)!r}, expected {value!r}")
    return problems


def _scan_one_exp_dir(root: Path) -> dict[str, Any]:
    file_count = 0
    total_bytes = 0
    checkpoint_count = 0
    checkpoint_bytes = 0
    newest_checkpoints: list[dict[str, Any]] = []
    largest_files: list[dict[str, Any]] = []
    if root.exists():
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            stat = path.stat()
            item = {
                "path": path.relative_to(root).as_posix(),
                "bytes": int(stat.st_size),
                "mtime": float(stat.st_mtime),
            }
            file_count += 1
            total_bytes += int(stat.st_size)
            largest_files.append(item)
            if path.name.endswith((".pth.tar", ".pth", ".pt")):
                checkpoint_count += 1
                checkpoint_bytes += int(stat.st_size)
                newest_checkpoints.append(item)
    return {
        "path": str(root),
        "exists": root.exists(),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "checkpoint_count": checkpoint_count,
        "checkpoint_bytes": checkpoint_bytes,
        "newest_checkpoints": sorted(
            newest_checkpoints, key=lambda item: item["mtime"], reverse=True
        )[:12],
        "largest_files": sorted(largest_files, key=lambda item: item["bytes"], reverse=True)[:12],
    }


def _scan_agent_exp_dir(exp_name: Path) -> dict[str, Any]:
    roots = [exp_name]
    if exp_name.is_absolute():
        roots.append(Path.cwd() / str(exp_name).lstrip("/"))
    scans = []
    seen = set()
    for root in roots:
        key = os.path.abspath(os.fspath(root))
        if key in seen:
            continue
        seen.add(key)
        scans.append(_scan_one_exp_dir(root))
    return {
        "exp_name": str(exp_name),
        "cwd": str(Path.cwd()),
        "root_scans": scans,
        "file_count": sum(int(scan["file_count"]) for scan in scans),
        "total_bytes": sum(int(scan["total_bytes"]) for scan in scans),
        "checkpoint_count": sum(int(scan["checkpoint_count"]) for scan in scans),
        "checkpoint_bytes": sum(int(scan["checkpoint_bytes"]) for scan in scans),
        "newest_checkpoints": sorted(
            [
                {**item, "root": scan["path"]}
                for scan in scans
                for item in scan["newest_checkpoints"]
            ],
            key=lambda item: item["mtime"],
            reverse=True,
        )[:12],
    }


def _write_json(path: Path, payload: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    runs.write_json(path, _to_plain(payload))
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _write_progress_snapshot(
    *,
    exp_name: Path,
    run_id: str,
    attempt_id: str,
    phase: str,
    progress_interval_sec: int,
    train_started_at: str | None,
    train_elapsed_sec: float | None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": "curvyzero_lightzero_muzero_agent96_progress/v0",
        "ok": error is None,
        "phase": phase,
        "timestamp": runs.utc_timestamp(),
        "task_id": TASK_ID,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "progress_interval_sec": progress_interval_sec,
        "train_started_at": train_started_at,
        "train_elapsed_sec": train_elapsed_sec,
        "scan": _scan_agent_exp_dir(exp_name),
    }
    if error is not None:
        payload["error"] = error
    progress_root = (
        runs.volume_path(RUNS_MOUNT, runs.attempt_train_ref(TASK_ID, run_id, attempt_id))
        / "progress"
    )
    latest_path = progress_root / "latest.json"
    history_path = progress_root / f"progress_{phase}_{runs.utc_stamp()}.json"
    return {
        "latest": _write_json(latest_path, payload),
        "history": _write_json(history_path, payload),
        "payload": _to_plain(payload),
    }


def _start_progress_watcher(
    *,
    exp_name: Path,
    run_id: str,
    attempt_id: str,
    progress_interval_sec: int,
    train_started_at: str,
    train_started_perf: float,
    snapshots: list[dict[str, Any]],
) -> tuple[threading.Event, threading.Thread] | None:
    if progress_interval_sec <= 0:
        return None

    stop_event = threading.Event()

    def _watch() -> None:
        while not stop_event.wait(progress_interval_sec):
            try:
                snapshots.append(
                    _write_progress_snapshot(
                        exp_name=exp_name,
                        run_id=run_id,
                        attempt_id=attempt_id,
                        phase="running",
                        progress_interval_sec=progress_interval_sec,
                        train_started_at=train_started_at,
                        train_elapsed_sec=time.perf_counter() - train_started_perf,
                    )
                )
                runs_volume.commit()
            except Exception:
                pass

    thread = threading.Thread(target=_watch, name="muzero-agent96-progress", daemon=True)
    thread.start()
    return stop_event, thread


def _runtime_compute_summary(*, requested_compute: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "requested_compute": requested_compute,
        "requested_modal_cpu_count": CPU_COUNT_BY_COMPUTE.get(requested_compute),
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
            device = int(torch.cuda.current_device())
            summary.update(
                {
                    "torch_cuda_current_device": device,
                    "torch_cuda_device_name": torch.cuda.get_device_name(device),
                    "torch_cuda_capability": list(torch.cuda.get_device_capability(device)),
                }
            )
    except Exception as exc:
        summary["torch_cuda_probe_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def _run_muzero_agent_reproduction(
    *,
    mode: str,
    compute: str,
    env_id: str,
    seed: int,
    train_step: int,
    run_id: str,
    attempt_id: str,
    progress_interval_sec: int,
    save_ckpt_after_iter_override: int | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if mode not in {"dry", "train"}:
        raise ValueError(f"unknown mode: {mode!r}; expected 'dry' or 'train'")
    if compute not in COMPUTE_CHOICES:
        raise ValueError(
            f"unknown compute: {compute!r}; expected one of: {_compute_choices_label()}"
        )
    if train_step <= 0:
        raise ValueError("train_step must be positive")
    if save_ckpt_after_iter_override is not None and save_ckpt_after_iter_override <= 0:
        raise ValueError("save_ckpt_after_iter_override must be positive when provided")

    packages = {
        "LightZero": _version_or_missing("LightZero", "lightzero"),
        "DI-engine": _version_or_missing("DI-engine", "ding"),
        "torch": _version_or_missing("torch"),
        "gym": _version_or_missing("gym"),
        "gymnasium": _version_or_missing("gymnasium"),
        "ale-py": _version_or_missing("ale-py", "ale_py"),
        "opencv-python-headless": _version_or_missing("opencv-python-headless"),
        "AutoROM": _version_or_missing("AutoROM"),
    }
    problems: list[str] = []
    if packages["LightZero"] != LIGHTZERO_VERSION:
        problems.append(
            f"installed LightZero version is {packages['LightZero']!r}, expected {LIGHTZERO_VERSION!r}"
        )

    config_module_name = "lzero.agent.config.muzero"
    agent_module_name = "lzero.agent"
    config_module = importlib.import_module(config_module_name)
    agent_module = importlib.import_module(agent_module_name)
    MuZeroAgent = getattr(agent_module, "MuZeroAgent")
    supported_env_cfg = getattr(config_module, "supported_env_cfg")
    if env_id not in supported_env_cfg:
        problems.append(f"{env_id!r} not present in MuZeroAgent supported_env_cfg")
        cfg = None
    else:
        cfg = copy.deepcopy(supported_env_cfg[env_id])

    exp_name_ref = runs.attempt_train_ref(TASK_ID, run_id, attempt_id) / "agent_exp"
    exp_name = Path(exp_name_ref.as_posix())
    patches: list[dict[str, Any]] = [
        {
            "path": "main_config.exp_name",
            "old": _to_plain(cfg.main_config.exp_name) if cfg is not None else None,
            "new": str(exp_name),
            "reason": "Modal Volume artifact root only; no environment/training semantics changed.",
        }
    ]
    if cfg is not None:
        cfg.main_config.exp_name = str(exp_name)
        if save_ckpt_after_iter_override is not None:
            patches.append(
                _set_path_creating_dicts(
                    cfg.main_config.policy,
                    ("learn", "learner", "hook", "save_ckpt_after_iter"),
                    save_ckpt_after_iter_override,
                )
            )

    surface = _extract_agent_surface(cfg, train_step=train_step) if cfg is not None else None
    if surface is not None:
        problems.extend(_validate_agent_surface(surface, expected_train_step=train_step))

    cpu_train_blocked = mode == "train" and compute == "cpu"
    if cpu_train_blocked:
        problems.append("CPU training is blocked; use a GPU compute for mode=train.")

    train_result: dict[str, Any] | None = None
    progress_snapshots: list[dict[str, Any]] = []
    original_cwd = Path.cwd()
    if mode == "train" and not problems:
        os.chdir(RUNS_MOUNT)
        train_started = time.perf_counter()
        train_started_at = runs.utc_timestamp()
        try:
            progress_snapshots.append(
                _write_progress_snapshot(
                    exp_name=exp_name,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    phase="starting",
                    progress_interval_sec=progress_interval_sec,
                    train_started_at=train_started_at,
                    train_elapsed_sec=0.0,
                )
            )
            runs_volume.commit()
        except Exception:
            pass
        watcher = _start_progress_watcher(
            exp_name=exp_name,
            run_id=run_id,
            attempt_id=attempt_id,
            progress_interval_sec=progress_interval_sec,
            train_started_at=train_started_at,
            train_started_perf=train_started,
            snapshots=progress_snapshots,
        )
        try:
            agent = MuZeroAgent(env_id=env_id, seed=seed, cfg=cfg)
            output = agent.train(step=train_step)
            train_result = {
                "ok": True,
                "return_type": type(output).__name__,
                "elapsed_sec": round(time.perf_counter() - train_started, 6),
            }
        except Exception as exc:
            problems.append(f"MuZeroAgent train failed: {type(exc).__name__}: {exc}")
            train_result = {"ok": False, "elapsed_sec": round(time.perf_counter() - train_started, 6)}
            train_result.update(_exception_result(exc))
        finally:
            if watcher is not None:
                stop_event, thread = watcher
                stop_event.set()
                thread.join(timeout=5)
            try:
                progress_snapshots.append(
                    _write_progress_snapshot(
                        exp_name=exp_name,
                        run_id=run_id,
                        attempt_id=attempt_id,
                        phase="completed" if train_result and train_result.get("ok") else "failed",
                        progress_interval_sec=progress_interval_sec,
                        train_started_at=train_started_at,
                        train_elapsed_sec=time.perf_counter() - train_started,
                        error=None if train_result and train_result.get("ok") else train_result,
                    )
                )
                runs_volume.commit()
            except Exception:
                pass

    artifact_scan = None
    artifact_scan_error = None
    try:
        artifact_scan = _scan_agent_exp_dir(exp_name)
    except Exception as exc:
        artifact_scan_error = _exception_result(exc)

    result = {
        "schema": "curvyzero_lightzero_pong_muzero_agent96_reproduction/v0",
        "ok": not problems and (mode == "dry" or bool(train_result and train_result.get("ok"))),
        "label": "LightZero MuZeroAgent 96x96 Atari Pong model-card reproduction path",
        "claim": (
            "This run uses the installed LightZero MuZeroAgent API and its "
            "PongNoFrameskip-v4 96x96/downsample visual Atari config."
        ),
        "non_claim": (
            "This does not prove CurvyTron readiness, simultaneous self-play, or parity "
            "with the separate 64x64 train_muzero config path."
        ),
        "mode": mode,
        "compute": compute,
        "task_id": TASK_ID,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "env_id": env_id,
        "seed": seed,
        "train_step": train_step,
        "save_ckpt_after_iter_override": save_ckpt_after_iter_override,
        "is_model_card_step": train_step == DEFAULT_TRAIN_STEP,
        "is_near_stock": save_ckpt_after_iter_override is None,
        "packages": packages,
        "runtime_compute": _runtime_compute_summary(requested_compute=compute),
        "agent_example": {
            "module": agent_module_name,
            "class": "MuZeroAgent",
            "constructor_signature": str(inspect.signature(MuZeroAgent)),
            "train_signature": str(inspect.signature(MuZeroAgent.train)),
            "model_card_snippet": (
                "agent = MuZeroAgent(env_id='PongNoFrameskip-v4', "
                "exp_name='PongNoFrameskip-v4-MuZero'); agent.train(step=500000)"
            ),
            "config_module": "lzero.agent.config.muzero.gym_pongnoframeskip_v4",
            "surface": surface,
            "patches": patches,
        },
        "working_directory": {
            "original_cwd": str(original_cwd),
            "current_cwd": str(Path.cwd()),
            "train_workdir": str(RUNS_MOUNT),
            "exp_name_ref": exp_name_ref.as_posix(),
            "exp_name_config_value": str(exp_name),
        },
        "problems": problems,
        "train_result": train_result,
        "progress": {
            "interval_sec": progress_interval_sec,
            "enabled": bool(mode == "train" and not cpu_train_blocked and progress_interval_sec > 0),
            "snapshot_count": len(progress_snapshots),
            "latest": progress_snapshots[-1].get("latest") if progress_snapshots else None,
        },
        "artifact_scan": artifact_scan,
        "artifact_scan_error": artifact_scan_error,
        "rom_unblocker": {
            "license_acceptance": ATARI_ROM_LICENSE_NOTICE,
            "modal_image_step": [
                'uv_pip_install("AutoROM[accept-rom-license]==0.6.1")',
                'run_commands("AutoROM --accept-license")',
            ],
        },
        "remote_elapsed_sec": round(time.perf_counter() - started, 6),
    }

    summary_root = (
        runs.volume_path(RUNS_MOUNT, runs.attempt_train_ref(TASK_ID, run_id, attempt_id))
        if mode == "train"
        else runs.volume_path(RUNS_MOUNT, runs.attempt_root_ref(TASK_ID, run_id, attempt_id))
    )
    summary_path = summary_root / ("summary.json" if mode == "train" else "dry_summary.json")
    result["artifact_refs"] = {"summary": _write_json(summary_path, result)}
    summary_ref = result["artifact_refs"]["summary"]["ref"]

    status = "completed" if result["ok"] else "failed"
    _write_json(
        runs.volume_path(RUNS_MOUNT, runs.attempt_manifest_ref(TASK_ID, run_id, attempt_id)),
        runs.attempt_manifest(
            task_id=TASK_ID,
            run_id=run_id,
            attempt_id=attempt_id,
            status=status,
            started_at=runs.utc_timestamp(),
            ended_at=runs.utc_timestamp(),
            modal_task_id=os.environ.get("MODAL_TASK_ID"),
            summary_ref=summary_ref,
            config={
                "mode": mode,
                "compute": compute,
                "env_id": env_id,
                "seed": seed,
                "train_step": train_step,
                "lightzero_version": LIGHTZERO_VERSION,
                "api": "lzero.agent.MuZeroAgent",
                "surface": "agent96_model_card",
                "save_ckpt_after_iter_override": save_ckpt_after_iter_override,
            },
        ),
    )
    _write_json(
        runs.volume_path(RUNS_MOUNT, runs.latest_attempt_ref(TASK_ID, run_id)),
        runs.latest_attempt_pointer(
            task_id=TASK_ID,
            run_id=run_id,
            attempt_id=attempt_id,
            status=status,
            started_at=runs.utc_timestamp(),
            ended_at=runs.utc_timestamp(),
            modal_task_id=os.environ.get("MODAL_TASK_ID"),
            summary_ref=summary_ref,
        ),
    )
    runs_volume.commit()
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=8 * 60, cpu=1.0)
def lightzero_pong_muzero_agent_cpu(
    mode: str = DEFAULT_MODE,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    train_step: int = DEFAULT_TRAIN_STEP,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    progress_interval_sec: int = DEFAULT_PROGRESS_INTERVAL_SEC,
    save_ckpt_after_iter_override: int | None = DEFAULT_SAVE_CKPT_AFTER_ITER_OVERRIDE,
) -> dict[str, Any]:
    return _run_muzero_agent_reproduction(
        mode=mode,
        compute="cpu",
        env_id=env_id,
        seed=seed,
        train_step=train_step,
        run_id=run_id,
        attempt_id=attempt_id,
        progress_interval_sec=progress_interval_sec,
        save_ckpt_after_iter_override=save_ckpt_after_iter_override,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=18 * 60 * 60,
    cpu=40.0,
    memory=32768,
    gpu=CHEAP_GPU_RESOURCE,
)
def lightzero_pong_muzero_agent_l4_cpu40(
    mode: str = DEFAULT_MODE,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    train_step: int = DEFAULT_TRAIN_STEP,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    progress_interval_sec: int = DEFAULT_PROGRESS_INTERVAL_SEC,
    save_ckpt_after_iter_override: int | None = DEFAULT_SAVE_CKPT_AFTER_ITER_OVERRIDE,
) -> dict[str, Any]:
    return _run_muzero_agent_reproduction(
        mode=mode,
        compute="gpu-l4-t4-cpu40",
        env_id=env_id,
        seed=seed,
        train_step=train_step,
        run_id=run_id,
        attempt_id=attempt_id,
        progress_interval_sec=progress_interval_sec,
        save_ckpt_after_iter_override=save_ckpt_after_iter_override,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=18 * 60 * 60,
    cpu=40.0,
    memory=32768,
    gpu=H100_GPU_RESOURCE,
)
def lightzero_pong_muzero_agent_h100_cpu40(
    mode: str = DEFAULT_MODE,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    train_step: int = DEFAULT_TRAIN_STEP,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    progress_interval_sec: int = DEFAULT_PROGRESS_INTERVAL_SEC,
    save_ckpt_after_iter_override: int | None = DEFAULT_SAVE_CKPT_AFTER_ITER_OVERRIDE,
) -> dict[str, Any]:
    return _run_muzero_agent_reproduction(
        mode=mode,
        compute="gpu-h100-cpu40",
        env_id=env_id,
        seed=seed,
        train_step=train_step,
        run_id=run_id,
        attempt_id=attempt_id,
        progress_interval_sec=progress_interval_sec,
        save_ckpt_after_iter_override=save_ckpt_after_iter_override,
    )


@app.local_entrypoint()
def main(
    mode: str = DEFAULT_MODE,
    compute: str = DEFAULT_COMPUTE,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    train_step: int = DEFAULT_TRAIN_STEP,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    progress_interval_sec: int = DEFAULT_PROGRESS_INTERVAL_SEC,
    save_ckpt_after_iter_override: int | None = DEFAULT_SAVE_CKPT_AFTER_ITER_OVERRIDE,
    wait_for_train: bool = False,
) -> None:
    if compute == "cpu":
        train_fn = lightzero_pong_muzero_agent_cpu
    elif compute == "gpu-l4-t4-cpu40":
        train_fn = lightzero_pong_muzero_agent_l4_cpu40
    elif compute == "gpu-h100-cpu40":
        train_fn = lightzero_pong_muzero_agent_h100_cpu40
    else:
        raise ValueError(
            f"unknown compute: {compute!r}; expected one of: {_compute_choices_label()}"
        )

    call_kwargs = {
        "mode": mode,
        "env_id": env_id,
        "seed": seed,
        "train_step": train_step,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "progress_interval_sec": progress_interval_sec,
        "save_ckpt_after_iter_override": save_ckpt_after_iter_override,
    }
    if mode == "train" and not wait_for_train:
        function_call = train_fn.spawn(**call_kwargs)
        call_id = getattr(function_call, "object_id", None) or getattr(function_call, "id", None)
        print(
            json.dumps(
                {
                    "schema": "curvyzero_lightzero_pong_muzero_agent96_background_launch/v1",
                    "status": "spawned",
                    "mode": mode,
                    "compute": compute,
                    "env_id": env_id,
                    "seed": seed,
                    "train_step": train_step,
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "function_call_id": call_id,
                    "progress_ref": (
                        runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
                        / "progress"
                        / "latest.json"
                    ).as_posix(),
                    "note": (
                        "Training was launched with Modal Function.spawn; use "
                        "--wait-for-train only for short calls that should return a full summary."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = train_fn.remote(**call_kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))
