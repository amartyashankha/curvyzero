"""Pinned GitHub LightZero Atari Pong MuZero dry-exact check.

This module is deliberately dry-only. It builds a Modal image from a pinned
``opendilab/LightZero`` GitHub commit, imports the upstream
``zoo.atari.config.atari_muzero_config`` module, captures the stock config
surface, patches only ``exp_name`` for artifact placement, and exits before
Atari env construction or ``train_muzero``.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_github_upstream_dry_check
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import os
import time
import traceback
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import modal

APP_NAME = "curvyzero-lightzero-pong-github-upstream-dry-check"
TASK_ID = "lightzero-official-visual-pong-github-upstream"
RUNS_MOUNT = Path("/runs")

LIGHTZERO_GITHUB_REPO = "https://github.com/opendilab/LightZero.git"
LIGHTZERO_GITHUB_COMMIT = "de74055298068f53b70e07bc38c41101fce51766"
DEFAULT_RUN_ID = "lz-visual-pong-github-upstream-exact-20260511-s0-dry"
DEFAULT_ATTEMPT_ID = "dry-exact-github-de740552-config-surface"
DEFAULT_SEED = 0
EXPECTED_MAX_ENV_STEP = 500000
DEFAULT_MODE = "plain-dry"
DEFAULT_COMPUTE = "cpu"
DEFAULT_ENV_ID = "ALE/Pong-v5"
DEFAULT_SEGMENT_RUN_ID = "lz-visual-pong-github-upstream-segment-20260511-s0-short"
DEFAULT_SEGMENT_ATTEMPT_ID = "train-muzero-segment-ale-pong-v5-50k-ckpt1000-l4cpu40"
DEFAULT_MAX_ENV_STEP_OVERRIDE = 50000
DEFAULT_SAVE_CKPT_AFTER_ITER_OVERRIDE = 1000
OPENCV_PYTHON_HEADLESS_VERSION = "4.11.0.86"
AUTOROM_VERSION = "0.6.1"
ATARI_ROM_LICENSE_NOTICE = (
    "This Modal image installs AutoROM[accept-rom-license] and runs "
    "`AutoROM --accept-license` during image build so ALE can load Atari ROMs."
)
SAFE_ID_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")
ATTEMPT_SCHEMA = "curvyzero_modal_training_attempt/v1"
LATEST_ATTEMPT_SCHEMA = "curvyzero_modal_training_latest_attempt/v1"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "g++")
    .uv_pip_install("Cython>=3", "numpy>=1.24.1,<2")
    .uv_pip_install(
        f"git+{LIGHTZERO_GITHUB_REPO}@{LIGHTZERO_GITHUB_COMMIT}",
        f"opencv-python-headless=={OPENCV_PYTHON_HEADLESS_VERSION}",
        f"AutoROM[accept-rom-license]=={AUTOROM_VERSION}",
    )
    .run_commands("AutoROM --accept-license")
)
runs_volume = modal.Volume.from_name("curvyzero-runs", create_if_missing=True)

app = modal.App(APP_NAME)


class _SegmentDryStop(RuntimeError):
    """Intentional stop after official segment config construction."""


def _version_or_missing(*packages: str) -> str:
    for package in packages:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return "missing"


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _clean_id(raw: str, *, label: str) -> str:
    if (
        not raw
        or len(raw) > 96
        or raw in {".", ".."}
        or not raw[0].isalnum()
        or any(char not in SAFE_ID_CHARS for char in raw)
    ):
        raise ValueError(
            f"{label} must be 1-96 chars of letters, numbers, dash, underscore, or dot"
        )
    return raw


def _run_root_ref(task_id: str, run_id: str) -> PurePosixPath:
    return PurePosixPath("training") / _clean_id(task_id, label="task_id") / _clean_id(
        run_id, label="run_id"
    )


def _attempt_root_ref(task_id: str, run_id: str, attempt_id: str) -> PurePosixPath:
    return _run_root_ref(task_id, run_id) / "attempts" / _clean_id(
        attempt_id, label="attempt_id"
    )


def _attempt_manifest_ref(task_id: str, run_id: str, attempt_id: str) -> PurePosixPath:
    return _attempt_root_ref(task_id, run_id, attempt_id) / "attempt.json"


def _attempt_train_ref(task_id: str, run_id: str, attempt_id: str) -> PurePosixPath:
    return _attempt_root_ref(task_id, run_id, attempt_id) / "train"


def _latest_attempt_ref(task_id: str, run_id: str) -> PurePosixPath:
    return _run_root_ref(task_id, run_id) / "latest_attempt.json"


def _volume_path(mount: Path, ref: PurePosixPath | str) -> Path:
    path = PurePosixPath(ref)
    if path.is_absolute() or not path.parts:
        raise ValueError("ref must be a non-empty relative path")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("ref must not contain empty, dot, or parent segments")
    return mount / Path(*path.parts)


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_summary(path: Path, *, mount: Path) -> dict[str, Any]:
    return {
        "ref": path.relative_to(mount).as_posix(),
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _attempt_manifest(
    *,
    task_id: str,
    run_id: str,
    attempt_id: str,
    status: str,
    started_at: str,
    ended_at: str,
    modal_task_id: str | None,
    summary_ref: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": ATTEMPT_SCHEMA,
        "task_id": task_id,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "modal_task_id": modal_task_id,
        "summary_ref": summary_ref,
        "config": config,
    }


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
        "traceback_tail": traceback.format_exc().splitlines()[-14:],
    }


def _write_json_artifact(path: Path, payload: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(_to_plain(payload)))
    return _file_summary(path, mount=RUNS_MOUNT)


def _get_path(mapping: Any, path: tuple[str, ...]) -> Any:
    current = mapping
    try:
        for part in path:
            current = current[part]
    except KeyError:
        return None
    return current


def _set_exp_name(main_config: Any, exp_name: Path) -> dict[str, Any]:
    old_value = main_config["exp_name"]
    main_config["exp_name"] = exp_name.as_posix()
    return {
        "path": "main_config.exp_name",
        "old": _to_plain(old_value),
        "new": exp_name.as_posix(),
        "reason": "artifact path metadata only; dry check does not train",
    }


def _extract_surface(main_config: Any, create_config: Any, *, max_env_step: int) -> dict[str, Any]:
    policy = main_config["policy"]
    env = main_config["env"]
    model = policy["model"]
    return {
        "exp_name": str(main_config["exp_name"]),
        "env_id": env["env_id"],
        "env_type": create_config["env"]["type"],
        "env_import_names": _to_plain(create_config["env"].get("import_names")),
        "env_manager_type": create_config["env_manager"]["type"],
        "policy_type": create_config["policy"]["type"],
        "policy_import_names": _to_plain(create_config["policy"].get("import_names")),
        "model_type": model["model_type"],
        "observation_shape": _to_plain(model["observation_shape"]),
        "env_observation_shape": _to_plain(env.get("observation_shape")),
        "action_space_size": model["action_space_size"],
        "collector_env_num": env["collector_env_num"],
        "policy_collector_env_num": policy.get("collector_env_num"),
        "n_episode": policy.get("n_episode"),
        "evaluator_env_num": env["evaluator_env_num"],
        "policy_evaluator_env_num": policy.get("evaluator_env_num"),
        "n_evaluator_episode": env.get("n_evaluator_episode"),
        "num_simulations": policy["num_simulations"],
        "batch_size": policy["batch_size"],
        "update_per_collect": policy.get("update_per_collect"),
        "replay_ratio": policy.get("replay_ratio"),
        "game_segment_length": policy.get("game_segment_length"),
        "eval_freq": policy.get("eval_freq"),
        "cuda": policy["cuda"],
        "learning_rate": policy.get("learning_rate"),
        "target_update_freq": policy.get("target_update_freq"),
        "replay_buffer_size": policy.get("replay_buffer_size"),
        "save_ckpt_after_iter": _get_path(
            policy, ("learn", "learner", "hook", "save_ckpt_after_iter")
        ),
        "collect_max_episode_steps": env.get("collect_max_episode_steps"),
        "eval_max_episode_steps": env.get("eval_max_episode_steps"),
        "frame_stack_num": env.get("frame_stack_num"),
        "gray_scale": env.get("gray_scale"),
        "image_channel": env.get("image_channel"),
        "max_env_step": max_env_step,
    }


def _validate_upstream_surface(surface: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    expected = {
        "env_id": "PongNoFrameskip-v4",
        "env_type": "atari_lightzero",
        "env_import_names": ["zoo.atari.envs.atari_lightzero_env"],
        "env_manager_type": "subprocess",
        "policy_type": "muzero",
        "policy_import_names": ["lzero.policy.muzero"],
        "model_type": "conv",
        "observation_shape": [4, 64, 64],
        "env_observation_shape": [4, 64, 64],
        "action_space_size": 6,
        "collector_env_num": 8,
        "policy_collector_env_num": 8,
        "n_episode": 8,
        "evaluator_env_num": 3,
        "policy_evaluator_env_num": 3,
        "n_evaluator_episode": 3,
        "num_simulations": 50,
        "batch_size": 256,
        "update_per_collect": None,
        "replay_ratio": 0.25,
        "game_segment_length": 400,
        "eval_freq": 2000,
        "cuda": True,
        "learning_rate": 0.2,
        "target_update_freq": 100,
        "replay_buffer_size": 1000000,
        "frame_stack_num": 4,
        "gray_scale": True,
        "image_channel": None,
        "max_env_step": EXPECTED_MAX_ENV_STEP,
    }
    for key, value in expected.items():
        if surface.get(key) != value:
            problems.append(f"upstream surface {key}={surface.get(key)!r}, expected {value!r}")
    if surface.get("collect_max_episode_steps") is not None:
        problems.append("upstream surface unexpectedly sets collect_max_episode_steps")
    if surface.get("eval_max_episode_steps") is not None:
        problems.append("upstream surface unexpectedly sets eval_max_episode_steps")
    return problems


def _extract_segment_surface(
    main_config: Any,
    create_config: Any,
    *,
    max_env_step: int,
) -> dict[str, Any]:
    surface = _extract_surface(main_config, create_config, max_env_step=max_env_step)
    policy = main_config["policy"]
    surface.update(
        {
            "num_segments": policy.get("num_segments"),
            "train_start_after_envsteps": policy.get("train_start_after_envsteps"),
            "td_steps": policy.get("td_steps"),
            "num_unroll_steps": policy.get("num_unroll_steps"),
            "buffer_reanalyze_freq": policy.get("buffer_reanalyze_freq"),
            "reanalyze_batch_size": policy.get("reanalyze_batch_size"),
            "reanalyze_partition": policy.get("reanalyze_partition"),
        }
    )
    return surface


def _validate_segment_surface(surface: dict[str, Any], *, expected_max_env_step: int) -> list[str]:
    problems: list[str] = []
    expected = {
        "env_id": "ALE/Pong-v5",
        "env_type": "atari_lightzero",
        "env_import_names": ["zoo.atari.envs.atari_lightzero_env"],
        "env_manager_type": "subprocess",
        "policy_type": "muzero",
        "policy_import_names": ["lzero.policy.muzero"],
        "model_type": "conv",
        "observation_shape": [4, 64, 64],
        "env_observation_shape": [4, 64, 64],
        "action_space_size": 6,
        "collector_env_num": 8,
        "policy_collector_env_num": 8,
        "evaluator_env_num": 3,
        "policy_evaluator_env_num": 3,
        "n_evaluator_episode": 3,
        "num_simulations": 50,
        "batch_size": 256,
        "update_per_collect": None,
        "replay_ratio": 0.25,
        "game_segment_length": 20,
        "num_segments": 8,
        "train_start_after_envsteps": 2000,
        "td_steps": 5,
        "eval_freq": 5000,
        "cuda": True,
        "learning_rate": 0.2,
        "target_update_freq": 100,
        "replay_buffer_size": 1000000,
        "save_ckpt_after_iter": 1000000,
        "frame_stack_num": 4,
        "gray_scale": True,
        "image_channel": None,
        "max_env_step": expected_max_env_step,
    }
    for key, value in expected.items():
        if surface.get(key) != value:
            problems.append(
                f"upstream segment surface {key}={surface.get(key)!r}, expected {value!r}"
            )
    if surface.get("collect_max_episode_steps") is not None:
        problems.append("upstream segment surface unexpectedly sets collect_max_episode_steps")
    if surface.get("eval_max_episode_steps") is not None:
        problems.append("upstream segment surface unexpectedly sets eval_max_episode_steps")
    return problems


def _patch_save_ckpt_after_iter(main_config: Any, value: int) -> dict[str, Any]:
    old_value = main_config["policy"]["learn"]["learner"]["hook"]["save_ckpt_after_iter"]
    main_config["policy"]["learn"]["learner"]["hook"]["save_ckpt_after_iter"] = value
    return {
        "path": "main_config.policy.learn.learner.hook.save_ckpt_after_iter",
        "old": _to_plain(old_value),
        "new": int(value),
        "reason": "faithful-short observability patch; not exact upstream",
    }


def _write_attempt_status(
    *,
    run_id: str,
    attempt_id: str,
    status: str,
    summary_ref: str,
    config: dict[str, Any],
) -> None:
    now = _utc_timestamp()
    attempt_path = _volume_path(RUNS_MOUNT, _attempt_manifest_ref(TASK_ID, run_id, attempt_id))
    latest_path = _volume_path(RUNS_MOUNT, _latest_attempt_ref(TASK_ID, run_id))
    _write_json_artifact(
        attempt_path,
        _attempt_manifest(
            task_id=TASK_ID,
            run_id=run_id,
            attempt_id=attempt_id,
            status=status,
            started_at=now,
            ended_at=now,
            modal_task_id=os.environ.get("MODAL_TASK_ID"),
            summary_ref=summary_ref,
            config=config,
        ),
    )
    _write_json_artifact(
        latest_path,
        {
            "schema": LATEST_ATTEMPT_SCHEMA,
            "task_id": TASK_ID,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "status": status,
            "updated_at": _utc_timestamp(),
            "summary_ref": summary_ref,
        },
    )


def _run_segment_official(
    *,
    mode: str,
    env_id: str,
    seed: int,
    run_id: str,
    attempt_id: str,
    max_env_step_override: int | None,
    save_ckpt_after_iter_override: int | None,
) -> dict[str, Any]:
    if mode not in {"segment-dry", "segment-train"}:
        raise ValueError("mode must be 'segment-dry' or 'segment-train'")
    if env_id != "ALE/Pong-v5":
        raise ValueError("current faithful segment wrapper is only validated for env_id='ALE/Pong-v5'")
    if max_env_step_override is not None and max_env_step_override <= 0:
        raise ValueError("max_env_step_override must be positive when provided")
    if save_ckpt_after_iter_override is not None and save_ckpt_after_iter_override <= 0:
        raise ValueError("save_ckpt_after_iter_override must be positive when provided")

    started = time.perf_counter()
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
    captured: dict[str, Any] = {}
    train_result: dict[str, Any] | None = None
    module_name = "zoo.atari.config.atari_muzero_segment_config"
    exp_name_ref = _attempt_train_ref(TASK_ID, run_id, attempt_id) / "lightzero_segment_exp"

    module = importlib.import_module(module_name)
    entry_module = importlib.import_module("lzero.entry")
    original_train = entry_module.train_muzero_segment
    original_cwd = Path.cwd()

    def intercepted_train(configs: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal train_result
        main_config, create_config = configs
        stock_max_env_step = int(kwargs.get("max_env_step"))
        original_surface = _extract_segment_surface(
            main_config,
            create_config,
            max_env_step=stock_max_env_step,
        )
        patches = [_set_exp_name(main_config, Path(exp_name_ref.as_posix()))]
        if save_ckpt_after_iter_override is not None:
            patches.append(_patch_save_ckpt_after_iter(main_config, save_ckpt_after_iter_override))
        actual_max_env_step = (
            stock_max_env_step if max_env_step_override is None else int(max_env_step_override)
        )
        patched_surface = _extract_segment_surface(
            main_config,
            create_config,
            max_env_step=actual_max_env_step,
        )
        captured.update(
            {
                "stock_max_env_step": stock_max_env_step,
                "actual_max_env_step": actual_max_env_step,
                "original_surface": original_surface,
                "patched_surface": patched_surface,
                "patches": patches,
                "trainer_signature": str(inspect.signature(original_train)),
            }
        )
        problems.extend(
            _validate_segment_surface(original_surface, expected_max_env_step=EXPECTED_MAX_ENV_STEP)
        )
        if max_env_step_override is None and save_ckpt_after_iter_override is None:
            problems.extend(
                _validate_segment_surface(patched_surface, expected_max_env_step=EXPECTED_MAX_ENV_STEP)
            )
        if mode == "segment-dry":
            raise _SegmentDryStop("dry segment config captured before env creation/train")
        if problems:
            raise RuntimeError("; ".join(problems))
        kwargs["max_env_step"] = actual_max_env_step
        os.chdir(RUNS_MOUNT)
        progress_path = (
            _volume_path(RUNS_MOUNT, _attempt_train_ref(TASK_ID, run_id, attempt_id))
            / "progress"
            / "latest.json"
        )
        _write_json_artifact(
            progress_path,
            {
                "phase": "starting",
                "updated_at": _utc_timestamp(),
                "env_id": env_id,
                "max_env_step": actual_max_env_step,
                "save_ckpt_after_iter_override": save_ckpt_after_iter_override,
                "source_commit": LIGHTZERO_GITHUB_COMMIT,
            },
        )
        runs_volume.commit()
        train_started = time.perf_counter()
        output = original_train(configs, *args, **kwargs)
        train_result = {
            "ok": True,
            "return_type": type(output).__name__,
            "elapsed_sec": round(time.perf_counter() - train_started, 6),
        }
        return output

    try:
        entry_module.train_muzero_segment = intercepted_train
        try:
            module.main(env_id, seed)
        except _SegmentDryStop:
            train_result = {"ok": True, "dry_intercepted_before_train": True}
    except Exception as exc:  # pragma: no cover - remote diagnosis only.
        problems.append(f"upstream segment run failed: {type(exc).__name__}: {exc}")
        train_result = {"ok": False, **_exception_result(exc)}
    finally:
        entry_module.train_muzero_segment = original_train
        os.chdir(original_cwd)

    result = {
        "ok": not problems and bool(train_result and train_result.get("ok")),
        "label": "pinned GitHub upstream LightZero Atari Pong MuZero segment faithful-short",
        "mode": mode,
        "env_id": env_id,
        "seed": seed,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "task_id": TASK_ID,
        "source": {
            "kind": "github",
            "repo": LIGHTZERO_GITHUB_REPO,
            "commit": LIGHTZERO_GITHUB_COMMIT,
            "install_spec": f"git+{LIGHTZERO_GITHUB_REPO}@{LIGHTZERO_GITHUB_COMMIT}",
            "official_source_module": module_name,
            "official_readme_command": "python3 -u zoo/atari/config/atari_muzero_segment_config.py",
            "official_env_arg_used": "--env ALE/Pong-v5",
        },
        "packages": packages,
        "call_policy": (
            "official_segment_config_intercepted_before_env_no_train"
            if mode == "segment-dry"
            else "official_segment_config_with_artifact_path_and_faithful_short_overrides"
        ),
        "max_env_step_override": max_env_step_override,
        "save_ckpt_after_iter_override": save_ckpt_after_iter_override,
        "stock_example": captured,
        "train_result": train_result,
        "problems": problems,
        "remote_elapsed_sec": round(time.perf_counter() - started, 6),
    }
    summary_name = (
        "dry_segment_github_upstream_summary.json"
        if mode == "segment-dry"
        else "train_segment_github_upstream_summary.json"
    )
    summary_root = (
        _attempt_root_ref(TASK_ID, run_id, attempt_id)
        if mode == "segment-dry"
        else _attempt_train_ref(TASK_ID, run_id, attempt_id)
    )
    summary_path = _volume_path(RUNS_MOUNT, summary_root) / summary_name
    result["artifact_refs"] = {"summary": _write_json_artifact(summary_path, result)}
    _write_attempt_status(
        run_id=run_id,
        attempt_id=attempt_id,
        status="completed" if result["ok"] else "failed",
        summary_ref=result["artifact_refs"]["summary"]["ref"],
        config={
            "mode": mode,
            "env_id": env_id,
            "seed": seed,
            "source_commit": LIGHTZERO_GITHUB_COMMIT,
            "max_env_step_override": max_env_step_override,
            "save_ckpt_after_iter_override": save_ckpt_after_iter_override,
        },
    )
    runs_volume.commit()
    return result


def _run_dry_check(*, seed: int, run_id: str, attempt_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    problems: list[str] = []
    module_name = "zoo.atari.config.atari_muzero_config"
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

    try:
        module = importlib.import_module(module_name)
        entry_module = importlib.import_module("lzero.entry")
        train_muzero_signature = str(inspect.signature(entry_module.train_muzero))
        stock_max_env_step = int(getattr(module, "max_env_step"))
        original_main_config = module.main_config
        original_create_config = module.create_config
        original_surface = _extract_surface(
            original_main_config,
            original_create_config,
            max_env_step=stock_max_env_step,
        )
        main_config = copy.deepcopy(original_main_config)
        create_config = copy.deepcopy(original_create_config)
        exp_name_ref = _attempt_train_ref(TASK_ID, run_id, attempt_id) / "lightzero_exp"
        patch = _set_exp_name(main_config, Path(exp_name_ref.as_posix()))
        patched_surface = _extract_surface(main_config, create_config, max_env_step=stock_max_env_step)
        problems.extend(_validate_upstream_surface(original_surface))
        problems.extend(_validate_upstream_surface(patched_surface))
    except Exception as exc:  # pragma: no cover - remote diagnosis only.
        stock_max_env_step = None
        train_muzero_signature = None
        original_surface = None
        patched_surface = None
        patch = None
        problems.append(f"dry exact import/config capture failed: {type(exc).__name__}: {exc}")
        import_error = _exception_result(exc)
    else:
        import_error = None

    result = {
        "ok": not problems,
        "label": "pinned GitHub upstream LightZero Atari Pong MuZero dry-exact config capture",
        "source": {
            "kind": "github",
            "repo": LIGHTZERO_GITHUB_REPO,
            "commit": LIGHTZERO_GITHUB_COMMIT,
            "install_spec": f"git+{LIGHTZERO_GITHUB_REPO}@{LIGHTZERO_GITHUB_COMMIT}",
            "version_note": (
                "LightZero upstream setup.py still reports package version 0.2.0; "
                "source identity is the pinned Git commit."
            ),
        },
        "mode": "dry-exact",
        "seed": seed,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "task_id": TASK_ID,
        "packages": packages,
        "stock_example": {
            "module": module_name,
            "trainer_entrypoint": "lzero.entry.train_muzero",
            "trainer_signature": train_muzero_signature,
            "trainer_args_not_called": {"seed": seed, "max_env_step": stock_max_env_step},
            "stock_max_env_step": stock_max_env_step,
            "original_surface": original_surface,
            "patched_surface": patched_surface,
            "patches": [patch] if patch is not None else [],
        },
        "call_policy": "dry_import_config_patch_exp_name_only_no_env_no_train",
        "problems": problems,
        "import_error": import_error,
        "rom_unblocker": {
            "license_acceptance": ATARI_ROM_LICENSE_NOTICE,
            "modal_image_step": [
                f'uv_pip_install("AutoROM[accept-rom-license]=={AUTOROM_VERSION}")',
                'run_commands("AutoROM --accept-license")',
            ],
        },
        "remote_elapsed_sec": round(time.perf_counter() - started, 6),
    }

    summary_path = (
        _volume_path(RUNS_MOUNT, _attempt_root_ref(TASK_ID, run_id, attempt_id))
        / "dry_exact_github_upstream_summary.json"
    )
    result["artifact_refs"] = {"dry_summary": _write_json_artifact(summary_path, result)}

    status = "completed" if result["ok"] else "failed"
    attempt_path = _volume_path(RUNS_MOUNT, _attempt_manifest_ref(TASK_ID, run_id, attempt_id))
    latest_path = _volume_path(RUNS_MOUNT, _latest_attempt_ref(TASK_ID, run_id))
    summary_ref = result["artifact_refs"]["dry_summary"]["ref"]
    _write_json_artifact(
        attempt_path,
        _attempt_manifest(
            task_id=TASK_ID,
            run_id=run_id,
            attempt_id=attempt_id,
            status=status,
            started_at=_utc_timestamp(),
            ended_at=_utc_timestamp(),
            modal_task_id=os.environ.get("MODAL_TASK_ID"),
            summary_ref=summary_ref,
            config={
                "mode": "dry-exact",
                "seed": seed,
                "source_commit": LIGHTZERO_GITHUB_COMMIT,
                "stock_max_env_step": stock_max_env_step,
            },
        ),
    )
    _write_json_artifact(
        latest_path,
        {
            "schema": LATEST_ATTEMPT_SCHEMA,
            "task_id": TASK_ID,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "status": status,
            "updated_at": _utc_timestamp(),
            "summary_ref": summary_ref,
        },
    )
    runs_volume.commit()
    return result


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=20 * 60, cpu=1.0)
def lightzero_pong_github_upstream_dry_check(
    seed: int = DEFAULT_SEED,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
) -> dict[str, Any]:
    return _run_dry_check(seed=seed, run_id=run_id, attempt_id=attempt_id)


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=20 * 60, cpu=1.0)
def lightzero_pong_github_upstream_segment_cpu(
    mode: str,
    env_id: str,
    seed: int,
    run_id: str,
    attempt_id: str,
    max_env_step_override: int | None,
    save_ckpt_after_iter_override: int | None,
) -> dict[str, Any]:
    if mode == "segment-train":
        raise ValueError("segment-train requires GPU compute")
    return _run_segment_official(
        mode=mode,
        env_id=env_id,
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        max_env_step_override=max_env_step_override,
        save_ckpt_after_iter_override=save_ckpt_after_iter_override,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=6 * 60 * 60,
    gpu=["L4", "T4"],
    cpu=40.0,
)
def lightzero_pong_github_upstream_segment_gpu_l4_cpu40(
    mode: str,
    env_id: str,
    seed: int,
    run_id: str,
    attempt_id: str,
    max_env_step_override: int | None,
    save_ckpt_after_iter_override: int | None,
) -> dict[str, Any]:
    return _run_segment_official(
        mode=mode,
        env_id=env_id,
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        max_env_step_override=max_env_step_override,
        save_ckpt_after_iter_override=save_ckpt_after_iter_override,
    )


@app.local_entrypoint()
def main(
    mode: str = DEFAULT_MODE,
    compute: str = DEFAULT_COMPUTE,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    run_id: str | None = None,
    attempt_id: str | None = None,
    max_env_step_override: int | None = DEFAULT_MAX_ENV_STEP_OVERRIDE,
    save_ckpt_after_iter_override: int | None = DEFAULT_SAVE_CKPT_AFTER_ITER_OVERRIDE,
    wait_for_train: bool = False,
) -> None:
    if mode == "plain-dry":
        result = lightzero_pong_github_upstream_dry_check.remote(
            seed=seed,
            run_id=run_id or DEFAULT_RUN_ID,
            attempt_id=attempt_id or DEFAULT_ATTEMPT_ID,
        )
        print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
        return

    if mode not in {"segment-dry", "segment-train"}:
        raise ValueError("mode must be one of: plain-dry, segment-dry, segment-train")
    actual_run_id = run_id or (
        DEFAULT_SEGMENT_RUN_ID if mode == "segment-train" else f"{DEFAULT_SEGMENT_RUN_ID}-dry"
    )
    actual_attempt_id = attempt_id or (
        DEFAULT_SEGMENT_ATTEMPT_ID
        if mode == "segment-train"
        else "dry-muzero-segment-ale-pong-v5-config-surface"
    )
    train_fn = (
        lightzero_pong_github_upstream_segment_gpu_l4_cpu40
        if compute == "gpu-l4-t4-cpu40"
        else lightzero_pong_github_upstream_segment_cpu
        if compute == "cpu"
        else None
    )
    if train_fn is None:
        raise ValueError("compute must be 'cpu' or 'gpu-l4-t4-cpu40'")
    call_kwargs = {
        "mode": mode,
        "env_id": env_id,
        "seed": seed,
        "run_id": actual_run_id,
        "attempt_id": actual_attempt_id,
        "max_env_step_override": max_env_step_override if mode == "segment-train" else None,
        "save_ckpt_after_iter_override": (
            save_ckpt_after_iter_override if mode == "segment-train" else None
        ),
    }
    if mode == "segment-train" and not wait_for_train:
        function_call = train_fn.spawn(**call_kwargs)
        call_id = getattr(function_call, "object_id", None) or getattr(function_call, "id", None)
        print(
            json.dumps(
                {
                    "schema": "curvyzero_lightzero_github_upstream_segment_launch/v1",
                    "status": "spawned",
                    "mode": mode,
                    "compute": compute,
                    "env_id": env_id,
                    "seed": seed,
                    "run_id": actual_run_id,
                    "attempt_id": actual_attempt_id,
                    "source_commit": LIGHTZERO_GITHUB_COMMIT,
                    "function_call_id": call_id,
                    "progress_ref": (
                        _attempt_train_ref(TASK_ID, actual_run_id, actual_attempt_id)
                        / "progress"
                        / "latest.json"
                    ).as_posix(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    result = train_fn.remote(**call_kwargs)
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
