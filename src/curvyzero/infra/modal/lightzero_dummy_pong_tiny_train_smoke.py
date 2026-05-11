"""Tiny Modal trainer smoke for LightZero MuZero on CurvyZero dummy Pong."""

from __future__ import annotations

import contextlib
import io
import importlib
import json
import os
import re
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTOR_ENV_NUM,
    DEFAULT_ENV,
    DEFAULT_EVALUATOR_ENV_NUM,
    DEFAULT_EPS_DECAY,
    DEFAULT_EPS_END,
    DEFAULT_EPS_GREEDY_EXPLORATION_IN_COLLECT,
    DEFAULT_EPS_START,
    DEFAULT_FEATURE_MODE,
    DEFAULT_FIXED_TEMPERATURE_VALUE,
    DEFAULT_GAME_SEGMENT_LENGTH,
    DEFAULT_DISCOUNT_FACTOR,
    DEFAULT_N_EPISODE,
    DEFAULT_N_EVALUATOR_EPISODE,
    DEFAULT_NUM_UNROLL_STEPS,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_PONG_EPISODE_MAX_STEPS,
    DEFAULT_PONG_RESET_PRESSURE_AGENT,
    DEFAULT_PONG_RESET_PROFILE,
    DEFAULT_RANDOM_COLLECT_EPISODE_NUM,
    DEFAULT_REWARD_SUPPORT_DELTA,
    DEFAULT_REWARD_SUPPORT_MAX,
    DEFAULT_REWARD_SUPPORT_MIN,
    DEFAULT_SEED,
    DEFAULT_SUPPORT_SCALE,
    DEFAULT_TD_STEPS,
    DEFAULT_UPDATE_PER_COLLECT,
    DEFAULT_VALUE_SUPPORT_DELTA,
    DEFAULT_VALUE_SUPPORT_MAX,
    DEFAULT_VALUE_SUPPORT_MIN,
    LIGHTZERO_VERSION,
    patched_dummy_pong_configs,
    resolve_pong_episode_max_steps,
    validate_dummy_pong_surface,
)

APP_NAME = "curvyzero-lightzero-dummy-pong-tiny-train-smoke"
TASK_ID = "lightzero-dummy-pong"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

DEFAULT_MAX_ENV_STEP = 64
DEFAULT_MAX_TRAIN_ITER = 2

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(f"LightZero=={LIGHTZERO_VERSION}")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


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


def _compact_log_tail(text: str, *, limit: int = 80) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def _exception_result(exc: BaseException) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


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


def _scan_lightzero_artifacts(exp_name: str) -> dict[str, Any]:
    root = Path(exp_name)
    roots = [root]
    if exp_name.startswith("/"):
        roots.append(Path("." + exp_name))
    files: list[dict[str, Any]] = []
    for candidate in roots:
        if not candidate.exists():
            continue
        for path in sorted(candidate.rglob("*")):
            if path.is_file():
                stat = path.stat()
                files.append(
                    {
                        "root": str(candidate),
                        "path": str(path),
                        "relative_path": str(path.relative_to(candidate)),
                        "size_bytes": stat.st_size,
                    }
                )
    checkpoint_files = [
        item for item in files if item["relative_path"].endswith((".pth.tar", ".pt", ".pth"))
    ]
    log_files = [
        item
        for item in files
        if item["relative_path"].endswith((".txt", ".log", ".json", ".jsonl"))
        or "events.out.tfevents" in item["relative_path"]
    ]
    return {
        "exists": bool(files),
        "file_count": len(files),
        "files_sample": files[:40],
        "checkpoint_files": checkpoint_files,
        "log_files": log_files[:40],
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
        source = Path(item["path"])
        if not source.is_file():
            continue
        dest = checkpoint_root / source.name
        shutil.copy2(source, dest)
        copied.append(runs.file_summary(dest, mount=RUNS_MOUNT))
    manifest = {
        "schema": "curvyzero_lightzero_checkpoint_manifest/v1",
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


def _copy_if_exists(source: Path, dest: Path) -> dict[str, Any] | None:
    if not source.is_file():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return runs.file_summary(dest, mount=RUNS_MOUNT)


def _target_replay_config_snapshot(
    *,
    config: dict[str, Any],
    patched_surface: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "curvyzero_lightzero_dummy_pong_target_replay_config/v1",
        "algorithm": "LightZero MuZero",
        "num_simulations": config["num_simulations"],
        "support_scale": {
            "support_scale": patched_surface.get("support_scale"),
            "reward_support_size": patched_surface.get("reward_support_size"),
            "value_support_size": patched_surface.get("value_support_size"),
            "reward_support_range": patched_surface.get("reward_support_range"),
            "value_support_range": patched_surface.get("value_support_range"),
            "categorical_distribution": patched_surface.get("categorical_distribution"),
            "support_scale_requested": config.get("support_scale"),
            "reward_support_requested": {
                "min": config.get("reward_support_min"),
                "max": config.get("reward_support_max"),
                "delta": config.get("reward_support_delta"),
            },
            "value_support_requested": {
                "min": config.get("value_support_min"),
                "max": config.get("value_support_max"),
                "delta": config.get("value_support_delta"),
            },
        },
        "feature": {
            "env": config["env"],
            "feature_mode": config["feature_mode"],
            "observation_shape": patched_surface.get("observation_shape"),
            "action_space_size": patched_surface.get("action_space_size"),
        },
        "reset": {
            "pong_reset_profile": config["pong_reset_profile"],
            "pong_reset_pressure_agent": config["pong_reset_pressure_agent"],
            "pong_episode_max_steps": config["pong_episode_max_steps"],
        },
        "opponent": {
            "opponent_policy": config["opponent_policy"],
            "ego_agent": config["ego_agent"],
            "opponent_checkpoint": config.get("opponent_checkpoint"),
            "opponent_checkpoint_label": config.get("opponent_checkpoint_label"),
            "opponent_checkpoint_adapter": config.get("opponent_checkpoint_adapter"),
            "opponent_checkpoint_num_simulations": config.get(
                "opponent_checkpoint_num_simulations"
            ),
            "opponent_checkpoint_state_key": config.get("opponent_checkpoint_state_key"),
            "opponent_checkpoint_input": config.get("opponent_checkpoint_input"),
        },
    }


def _len_or_zero(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def _segment_item(segment: Any, field: str, index: int, default: Any = None) -> Any:
    values = getattr(segment, field, None)
    if values is None:
        return default
    try:
        return values[index]
    except (IndexError, TypeError):
        return default


def _target_replay_step_rows(
    *,
    collect_call_index: int,
    collect_kwargs: dict[str, Any],
    segments: list[Any],
    metadata: list[dict[str, Any]],
    config_snapshot: dict[str, Any],
    first_global_episode_index: int,
    first_global_step_index: int,
) -> tuple[list[dict[str, Any]], int, int]:
    from curvyzero.training.dummy_pong import ACTION_LABELS

    rows: list[dict[str, Any]] = []
    global_episode_index = first_global_episode_index
    global_step_index = first_global_step_index
    for segment_index, segment in enumerate(segments):
        segment_metadata = metadata[segment_index] if segment_index < len(metadata) else {}
        segment_done = bool(segment_metadata.get("done", False))
        action_segment = getattr(segment, "action_segment", [])
        segment_length = _len_or_zero(action_segment)
        for step_index in range(segment_length):
            action = _to_plain(_segment_item(segment, "action_segment", step_index))
            try:
                action_id = int(action)
            except (TypeError, ValueError):
                action_id = None
            row_done = segment_done and step_index == segment_length - 1
            rows.append(
                {
                    "schema": "curvyzero_lightzero_dummy_pong_target_replay_step/v1",
                    "collect_call_index": collect_call_index,
                    "collect_train_iter": collect_kwargs.get("train_iter"),
                    "collect_n_episode": collect_kwargs.get("n_episode"),
                    "collect_with_pure_policy": bool(
                        collect_kwargs.get("collect_with_pure_policy", False)
                    ),
                    "global_episode_index": global_episode_index,
                    "global_step_index": global_step_index,
                    "segment_index": segment_index,
                    "step_index_in_segment": step_index,
                    "action_segment": action,
                    "action_label": (
                        ACTION_LABELS[action_id]
                        if action_id is not None and 0 <= action_id < len(ACTION_LABELS)
                        else None
                    ),
                    "child_visit_segment": _to_plain(
                        _segment_item(segment, "child_visit_segment", step_index, [])
                    ),
                    "visit_count_distribution": _to_plain(
                        _segment_item(segment, "child_visit_segment", step_index, [])
                    ),
                    "reward": _to_plain(_segment_item(segment, "reward_segment", step_index)),
                    "root_value": _to_plain(
                        _segment_item(segment, "root_value_segment", step_index)
                    ),
                    "done": row_done,
                    "truncated": None,
                    "truncation_source": (
                        "terminal episodes.jsonl sidecar"
                        if row_done
                        else "not_applicable_before_terminal_step"
                    ),
                    "segment_done": segment_done,
                    "segment_metadata": _to_plain(segment_metadata),
                    "target_config": config_snapshot,
                }
            )
            global_step_index += 1
        if segment_done:
            global_episode_index += 1
    return rows, global_episode_index, global_step_index


@contextlib.contextmanager
def _mirror_lightzero_collector_targets(
    *,
    path: Path,
    config_snapshot: dict[str, Any],
):
    """Mirror compact GameSegment target rows returned by LightZero collection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    state = {
        "collect_call_index": 0,
        "global_episode_index": 0,
        "global_step_index": 0,
    }

    from lzero.worker import MuZeroCollector

    original_collect = MuZeroCollector.collect

    def collect_and_mirror(self, *args: Any, **kwargs: Any):  # noqa: ANN001
        result = original_collect(self, *args, **kwargs)
        result_items = list(result) if isinstance(result, (list, tuple)) else []
        segments = list(result_items[0]) if result_items else []
        metadata = list(result_items[1]) if len(result_items) > 1 else []
        collect_kwargs = dict(kwargs)
        if args:
            collect_kwargs["args"] = _to_plain(args)
        rows, next_episode_index, next_step_index = _target_replay_step_rows(
            collect_call_index=int(state["collect_call_index"]),
            collect_kwargs=collect_kwargs,
            segments=segments,
            metadata=metadata,
            config_snapshot=config_snapshot,
            first_global_episode_index=int(state["global_episode_index"]),
            first_global_step_index=int(state["global_step_index"]),
        )
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(_to_plain(row), sort_keys=True) + "\n")
        state["collect_call_index"] = int(state["collect_call_index"]) + 1
        state["global_episode_index"] = next_episode_index
        state["global_step_index"] = next_step_index
        return result

    MuZeroCollector.collect = collect_and_mirror
    try:
        yield state
    finally:
        MuZeroCollector.collect = original_collect


def _summarize_target_replay_rows(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": "curvyzero_lightzero_dummy_pong_target_replay_summary/v1",
            "rows": 0,
            "episodes": 0,
            "collect_calls": 0,
            "terminal_steps": 0,
            "action_counts": {},
        }
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    action_counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("action_label"))
        action_counts[label] = action_counts.get(label, 0) + 1
    return {
        "schema": "curvyzero_lightzero_dummy_pong_target_replay_summary/v1",
        "rows": len(rows),
        "episodes": len({int(row["global_episode_index"]) for row in rows}) if rows else 0,
        "collect_calls": len({int(row["collect_call_index"]) for row in rows}) if rows else 0,
        "terminal_steps": sum(1 for row in rows if bool(row.get("done"))),
        "action_counts": action_counts,
        "has_child_visit_segment": any(bool(row.get("child_visit_segment")) for row in rows),
        "has_reward": any(row.get("reward") is not None for row in rows),
        "truncation_note": (
            "GameSegment does not expose truncation per transition; terminal "
            "done/truncated remains in episodes.jsonl."
        ),
    }


def _resolve_opponent_checkpoint_input(
    *,
    opponent_checkpoint: str | None,
    opponent_policy: str,
) -> dict[str, Any] | None:
    checkpoint_policy_ids = {
        "lightzero_policy_head_checkpoint",
        "lightzero_mcts_checkpoint",
    }
    if opponent_policy in checkpoint_policy_ids and not opponent_checkpoint:
        raise ValueError(f"{opponent_policy} requires --opponent-checkpoint")
    if not opponent_checkpoint:
        return None

    path, source = runs.resolve_mounted_ref_or_path(
        opponent_checkpoint,
        mount=RUNS_MOUNT,
        remote_root=REMOTE_ROOT,
    )
    if not path.is_file():
        raise FileNotFoundError(f"opponent checkpoint file not found: {path}")
    return {
        "checkpoint_path_or_ref": opponent_checkpoint,
        "resolved_checkpoint_path": str(path),
        **source,
        "file": runs.file_summary_any_mount(path, mount=RUNS_MOUNT),
    }


def _run_lightzero_dummy_pong_tiny_train_smoke(
    *,
    mode: str,
    env: str,
    feature_mode: str,
    seed: int,
    opponent_policy: str,
    max_env_step: int,
    pong_episode_max_steps: int | None,
    max_train_iter: int,
    collector_env_num: int,
    evaluator_env_num: int,
    n_evaluator_episode: int,
    num_simulations: int,
    batch_size: int,
    update_per_collect: int,
    pong_reset_profile: str = DEFAULT_PONG_RESET_PROFILE,
    pong_reset_pressure_agent: str = DEFAULT_PONG_RESET_PRESSURE_AGENT,
    n_episode: int = DEFAULT_N_EPISODE,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
    random_collect_episode_num: int = DEFAULT_RANDOM_COLLECT_EPISODE_NUM,
    eps_greedy_exploration_in_collect: bool = DEFAULT_EPS_GREEDY_EXPLORATION_IN_COLLECT,
    eps_start: float = DEFAULT_EPS_START,
    eps_end: float = DEFAULT_EPS_END,
    eps_decay: int = DEFAULT_EPS_DECAY,
    fixed_temperature_value: float = DEFAULT_FIXED_TEMPERATURE_VALUE,
    td_steps: int | None = DEFAULT_TD_STEPS,
    num_unroll_steps: int | None = DEFAULT_NUM_UNROLL_STEPS,
    discount_factor: float | None = DEFAULT_DISCOUNT_FACTOR,
    support_scale: int | None = DEFAULT_SUPPORT_SCALE,
    reward_support_min: float | None = DEFAULT_REWARD_SUPPORT_MIN,
    reward_support_max: float | None = DEFAULT_REWARD_SUPPORT_MAX,
    reward_support_delta: float | None = DEFAULT_REWARD_SUPPORT_DELTA,
    value_support_min: float | None = DEFAULT_VALUE_SUPPORT_MIN,
    value_support_max: float | None = DEFAULT_VALUE_SUPPORT_MAX,
    value_support_delta: float | None = DEFAULT_VALUE_SUPPORT_DELTA,
    run_id: str | None = None,
    attempt_id: str | None = None,
    ego_agent: str = "player_0",
    opponent_checkpoint: str | None = None,
    opponent_checkpoint_label: str | None = None,
    opponent_checkpoint_adapter: str | None = None,
    opponent_checkpoint_num_simulations: int | None = None,
    opponent_checkpoint_state_key: str | None = None,
    max_allowed_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_allowed_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    job_kind: str = "lightzero_custom_dummy_pong_muzero_tiny_train",
    result_label: str = "LightZero custom-env dummy Pong MuZero tiny train smoke",
) -> dict[str, Any]:
    if mode not in {"dry", "train", "progression"}:
        raise ValueError("mode must be 'dry', 'train', or 'progression'")
    if max_env_step > max_allowed_env_step:
        raise ValueError(f"max_env_step must stay <={max_allowed_env_step} for this smoke")
    if max_train_iter > max_allowed_train_iter:
        raise ValueError(f"max_train_iter must stay <={max_allowed_train_iter} for this smoke")
    effective_pong_episode_max_steps = resolve_pong_episode_max_steps(
        max_env_step=max_env_step,
        pong_episode_max_steps=pong_episode_max_steps,
    )

    run_id = run_id or runs.new_run_id("lz-dpong")
    attempt_id = attempt_id or runs.new_attempt_id("attempt")
    started_at = runs.utc_timestamp()
    modal_task_id = os.environ.get("MODAL_TASK_ID")
    opponent_checkpoint_input = _resolve_opponent_checkpoint_input(
        opponent_checkpoint=opponent_checkpoint,
        opponent_policy=opponent_policy,
    )
    config = {
        "job_kind": job_kind,
        "algorithm": "LightZero MuZero",
        "mode": mode,
        "env": env,
        "feature_mode": feature_mode,
        "seed": seed,
        "opponent_policy": opponent_policy,
        "ego_agent": ego_agent,
        "opponent_checkpoint": opponent_checkpoint,
        "opponent_checkpoint_label": opponent_checkpoint_label,
        "opponent_checkpoint_adapter": opponent_checkpoint_adapter,
        "opponent_checkpoint_num_simulations": opponent_checkpoint_num_simulations,
        "opponent_checkpoint_state_key": opponent_checkpoint_state_key or "model",
        "opponent_checkpoint_input": opponent_checkpoint_input,
        "max_env_step": max_env_step,
        "max_env_step_role": "lightzero_training_budget",
        "requested_pong_episode_max_steps": pong_episode_max_steps,
        "pong_episode_max_steps": effective_pong_episode_max_steps,
        "effective_pong_episode_max_steps": effective_pong_episode_max_steps,
        "pong_reset_profile": pong_reset_profile,
        "pong_reset_pressure_agent": pong_reset_pressure_agent,
        "pong_episode_max_steps_source": (
            "max_env_step_legacy_default" if pong_episode_max_steps is None else "explicit"
        ),
        "max_train_iter": max_train_iter,
        "max_allowed_env_step": max_allowed_env_step,
        "max_allowed_train_iter": max_allowed_train_iter,
        "collector_env_num": collector_env_num,
        "evaluator_env_num": evaluator_env_num,
        "n_evaluator_episode": n_evaluator_episode,
        "num_simulations": num_simulations,
        "batch_size": batch_size,
        "update_per_collect": update_per_collect,
        "n_episode": n_episode,
        "game_segment_length": game_segment_length,
        "random_collect_episode_num": random_collect_episode_num,
        "eps_greedy_exploration_in_collect": eps_greedy_exploration_in_collect,
        "eps_start": eps_start,
        "eps_end": eps_end,
        "eps_decay": eps_decay,
        "fixed_temperature_value": fixed_temperature_value,
        "td_steps": td_steps,
        "num_unroll_steps": num_unroll_steps,
        "discount_factor": discount_factor,
        "support_scale": support_scale,
        "reward_support_min": reward_support_min,
        "reward_support_max": reward_support_max,
        "reward_support_delta": reward_support_delta,
        "value_support_min": value_support_min,
        "value_support_max": value_support_max,
        "value_support_delta": value_support_delta,
    }
    attempt_train_root = runs.volume_path(
        RUNS_MOUNT,
        runs.attempt_train_ref(TASK_ID, run_id, attempt_id),
    )
    local_telemetry_path = Path("/tmp") / "curvyzero-lightzero-dummy-pong-episodes.jsonl"
    local_target_replay_path = (
        Path("/tmp") / "curvyzero-lightzero-dummy-pong-target-replay-steps.jsonl"
    )
    if local_telemetry_path.exists():
        local_telemetry_path.unlink()
    if local_target_replay_path.exists():
        local_target_replay_path.unlink()
    patched = patched_dummy_pong_configs(
        env=env,
        feature_mode=feature_mode,
        seed=seed,
        opponent_policy=opponent_policy,
        max_env_step=max_env_step,
        pong_episode_max_steps=pong_episode_max_steps,
        pong_reset_profile=pong_reset_profile,
        pong_reset_pressure_agent=pong_reset_pressure_agent,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=n_evaluator_episode,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        n_episode=n_episode,
        game_segment_length=game_segment_length,
        random_collect_episode_num=random_collect_episode_num,
        eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        fixed_temperature_value=fixed_temperature_value,
        td_steps=td_steps,
        num_unroll_steps=num_unroll_steps,
        discount_factor=discount_factor,
        support_scale=support_scale,
        reward_support_min=reward_support_min,
        reward_support_max=reward_support_max,
        reward_support_delta=reward_support_delta,
        value_support_min=value_support_min,
        value_support_max=value_support_max,
        value_support_delta=value_support_delta,
        telemetry_path=str(local_telemetry_path),
        ego_agent=ego_agent,
        opponent_checkpoint_path=(
            str(opponent_checkpoint_input["resolved_checkpoint_path"])
            if opponent_checkpoint_input
            else None
        ),
        opponent_checkpoint_label=opponent_checkpoint_label,
        opponent_checkpoint_adapter=opponent_checkpoint_adapter,
        opponent_checkpoint_num_simulations=opponent_checkpoint_num_simulations,
        opponent_checkpoint_sha256=(
            opponent_checkpoint_input["file"]["sha256"]
            if opponent_checkpoint_input
            else None
        ),
        opponent_checkpoint_source_ref=(
            opponent_checkpoint_input.get("source_ref")
            if opponent_checkpoint_input
            else None
        ),
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
    )
    problems = validate_dummy_pong_surface(
        patched["patched_surface"],
        max_env_step=max_env_step,
        pong_episode_max_steps=pong_episode_max_steps,
        pong_reset_profile=pong_reset_profile,
    )
    config["patched_terminal_target_config"] = {
        "td_steps": patched["patched_surface"].get("td_steps"),
        "num_unroll_steps": patched["patched_surface"].get("num_unroll_steps"),
        "discount_factor": patched["patched_surface"].get("discount_factor"),
        "support_scale": patched["patched_surface"].get("support_scale"),
        "reward_support_size": patched["patched_surface"].get("reward_support_size"),
        "value_support_size": patched["patched_surface"].get("value_support_size"),
        "reward_support_range": patched["patched_surface"].get("reward_support_range"),
        "value_support_range": patched["patched_surface"].get("value_support_range"),
        "categorical_distribution": patched["patched_surface"].get("categorical_distribution"),
    }
    target_replay_config = _target_replay_config_snapshot(
        config=config,
        patched_surface=patched["patched_surface"],
    )
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
    if max_train_iter > max_allowed_train_iter:
        problems.append(f"max_train_iter={max_train_iter!r} exceeds cap {max_allowed_train_iter}")

    entry_module = importlib.import_module("lzero.entry")
    train_muzero = entry_module.train_muzero
    train_result: dict[str, Any] | None = None
    called_train_muzero = False
    stdout_text = ""
    stderr_text = ""
    artifact_summary = _scan_lightzero_artifacts(str(patched["main_config"]["exp_name"]))

    if mode in {"train", "progression"} and not problems:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        train_started = time.perf_counter()
        try:
            with (
                contextlib.redirect_stdout(stdout_buffer),
                contextlib.redirect_stderr(stderr_buffer),
                _mirror_lightzero_collector_targets(
                    path=local_target_replay_path,
                    config_snapshot=target_replay_config,
                ),
            ):
                called_train_muzero = True
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
            artifact_summary = _scan_lightzero_artifacts(str(patched["main_config"]["exp_name"]))
            train_result = {
                "ok": True,
                "return_type": type(output).__name__,
                "elapsed_sec": round(time.perf_counter() - train_started, 6),
                "log_signals": _parse_training_signals(stdout_text, stderr_text),
            }
        except Exception as exc:  # pragma: no cover - remote trainer diagnosis.
            stdout_text = stdout_buffer.getvalue()
            stderr_text = stderr_buffer.getvalue()
            artifact_summary = _scan_lightzero_artifacts(str(patched["main_config"]["exp_name"]))
            problems.append(f"LightZero train_muzero failed: {type(exc).__name__}: {exc}")
            train_result = {
                "ok": False,
                "elapsed_sec": round(time.perf_counter() - train_started, 6),
                "log_signals": _parse_training_signals(stdout_text, stderr_text),
            }
            train_result.update(_exception_result(exc))

    from curvyzero.training.lightzero_dummy_pong_env import load_episode_rows
    from curvyzero.training.lightzero_dummy_pong_env import summarize_episode_rows

    episode_rows = load_episode_rows(local_telemetry_path)
    scorecard = summarize_episode_rows(episode_rows)
    target_replay_summary = _summarize_target_replay_rows(local_target_replay_path)
    checkpoint_mirror = _mirror_lightzero_checkpoints(
        run_id=run_id,
        artifact_summary=artifact_summary,
    )
    if mode in {"train", "progression"} and train_result and train_result.get("ok"):
        log_signals = train_result.get("log_signals", {})
        if not log_signals.get("training_iterations"):
            problems.append("LightZero logs did not expose any Training Iteration lines")
        if not episode_rows:
            problems.append("no env-side dummy Pong episode telemetry rows were written")
        if not artifact_summary.get("checkpoint_files"):
            problems.append("no LightZero checkpoint artifacts were discovered")
        if not checkpoint_mirror.get("copied_checkpoints"):
            problems.append("no LightZero checkpoints were mirrored to curvyzero-runs")
        if not target_replay_summary.get("rows"):
            problems.append("no LightZero replay target telemetry rows were mirrored")
    summary = {
        "schema": "curvyzero_lightzero_dummy_pong_tiny_train_summary/v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "algorithm": "LightZero MuZero",
        "called_train_muzero": called_train_muzero,
        "mode": mode,
        "ok": not problems and (mode == "dry" or bool(train_result and train_result.get("ok"))),
        "problems": problems,
        "command": config,
        "config_surface": patched["patched_surface"],
        "train_result": train_result,
        "lightzero_artifacts": artifact_summary,
        "checkpoint_mirror": checkpoint_mirror,
        "pong_scorecard": scorecard,
        "target_replay": target_replay_summary,
        "independent_scorecard": {
            "status": "blocked",
            "reason": (
                "This smoke records CurvyZero episode rows from the env sidecar, "
                "but LightZero checkpoint inference in the standalone CurvyZero "
                "scoreboard is not implemented yet."
            ),
        },
    }
    summary_path = attempt_train_root / "summary.json"
    episodes_path = attempt_train_root / "episodes.jsonl"
    target_replay_path = attempt_train_root / "target_replay_steps.jsonl"
    target_replay_summary_path = attempt_train_root / "target_replay_summary.json"
    signals_path = attempt_train_root / "lightzero_training_signals.json"
    artifacts_path = attempt_train_root / "lightzero_artifacts_manifest.json"
    config_path = runs.volume_path(RUNS_MOUNT, runs.attempt_root_ref(TASK_ID, run_id, attempt_id)) / "config.json"
    command_path = runs.volume_path(RUNS_MOUNT, runs.attempt_root_ref(TASK_ID, run_id, attempt_id)) / "command.json"
    stdout_path = runs.volume_path(RUNS_MOUNT, runs.attempt_root_ref(TASK_ID, run_id, attempt_id)) / "stdout_tail.txt"
    stderr_path = runs.volume_path(RUNS_MOUNT, runs.attempt_root_ref(TASK_ID, run_id, attempt_id)) / "stderr_tail.txt"

    runs.write_json(summary_path, summary)
    if local_telemetry_path.exists():
        _copy_if_exists(local_telemetry_path, episodes_path)
    else:
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        episodes_path.write_text("", encoding="utf-8")
    if local_target_replay_path.exists():
        _copy_if_exists(local_target_replay_path, target_replay_path)
    else:
        target_replay_path.parent.mkdir(parents=True, exist_ok=True)
        target_replay_path.write_text("", encoding="utf-8")
    runs.write_json(target_replay_summary_path, target_replay_summary)
    runs.write_json(signals_path, train_result["log_signals"] if train_result else {})
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

    ended_at = runs.utc_timestamp()
    status = "completed" if summary["ok"] else "failed"
    summary_ref = runs.file_ref(summary_path, mount=RUNS_MOUNT)
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
    runs_volume.commit()
    result = {
        "ok": summary["ok"],
        "label": result_label,
        "algorithm": "LightZero MuZero",
        "mode": mode,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "status": status,
        "problems": problems,
        "called_train_muzero": called_train_muzero,
        "summary_ref": summary_ref,
        "attempt_manifest": attempt_manifest,
        "latest_attempt": latest_attempt,
        "pong_scorecard": scorecard,
        "target_replay": target_replay_summary,
        "checkpoint_mirror": checkpoint_mirror,
        "artifact_refs": {
            "summary": runs.file_summary(summary_path, mount=RUNS_MOUNT),
            "episodes": runs.file_summary(episodes_path, mount=RUNS_MOUNT),
            "target_replay_steps": runs.file_summary(target_replay_path, mount=RUNS_MOUNT),
            "target_replay_summary": runs.file_summary(
                target_replay_summary_path,
                mount=RUNS_MOUNT,
            ),
            "training_signals": runs.file_summary(signals_path, mount=RUNS_MOUNT),
            "lightzero_artifacts": runs.file_summary(artifacts_path, mount=RUNS_MOUNT),
        },
    }
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=10 * 60)
def lightzero_dummy_pong_tiny_train_smoke(
    mode: str = "dry",
    env: str = DEFAULT_ENV,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    seed: int = DEFAULT_SEED,
    opponent_policy: str = "random_uniform",
    ego_agent: str = "player_0",
    opponent_checkpoint: str | None = None,
    opponent_checkpoint_label: str | None = None,
    opponent_checkpoint_adapter: str | None = None,
    opponent_checkpoint_num_simulations: int | None = None,
    opponent_checkpoint_state_key: str | None = None,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    pong_episode_max_steps: int | None = DEFAULT_PONG_EPISODE_MAX_STEPS,
    pong_reset_profile: str = DEFAULT_PONG_RESET_PROFILE,
    pong_reset_pressure_agent: str = DEFAULT_PONG_RESET_PRESSURE_AGENT,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    n_evaluator_episode: int = DEFAULT_N_EVALUATOR_EPISODE,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    n_episode: int = DEFAULT_N_EPISODE,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
    random_collect_episode_num: int = DEFAULT_RANDOM_COLLECT_EPISODE_NUM,
    eps_greedy_exploration_in_collect: bool = DEFAULT_EPS_GREEDY_EXPLORATION_IN_COLLECT,
    eps_start: float = DEFAULT_EPS_START,
    eps_end: float = DEFAULT_EPS_END,
    eps_decay: int = DEFAULT_EPS_DECAY,
    fixed_temperature_value: float = DEFAULT_FIXED_TEMPERATURE_VALUE,
    td_steps: int | None = DEFAULT_TD_STEPS,
    num_unroll_steps: int | None = DEFAULT_NUM_UNROLL_STEPS,
    discount_factor: float | None = DEFAULT_DISCOUNT_FACTOR,
    support_scale: int | None = DEFAULT_SUPPORT_SCALE,
    reward_support_min: float | None = DEFAULT_REWARD_SUPPORT_MIN,
    reward_support_max: float | None = DEFAULT_REWARD_SUPPORT_MAX,
    reward_support_delta: float | None = DEFAULT_REWARD_SUPPORT_DELTA,
    value_support_min: float | None = DEFAULT_VALUE_SUPPORT_MIN,
    value_support_max: float | None = DEFAULT_VALUE_SUPPORT_MAX,
    value_support_delta: float | None = DEFAULT_VALUE_SUPPORT_DELTA,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    return _run_lightzero_dummy_pong_tiny_train_smoke(
        mode=mode,
        env=env,
        feature_mode=feature_mode,
        seed=seed,
        opponent_policy=opponent_policy,
        ego_agent=ego_agent,
        opponent_checkpoint=opponent_checkpoint,
        opponent_checkpoint_label=opponent_checkpoint_label,
        opponent_checkpoint_adapter=opponent_checkpoint_adapter,
        opponent_checkpoint_num_simulations=opponent_checkpoint_num_simulations,
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
        max_env_step=max_env_step,
        pong_episode_max_steps=pong_episode_max_steps,
        pong_reset_profile=pong_reset_profile,
        pong_reset_pressure_agent=pong_reset_pressure_agent,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=n_evaluator_episode,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        n_episode=n_episode,
        game_segment_length=game_segment_length,
        random_collect_episode_num=random_collect_episode_num,
        eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        fixed_temperature_value=fixed_temperature_value,
        td_steps=td_steps,
        num_unroll_steps=num_unroll_steps,
        discount_factor=discount_factor,
        support_scale=support_scale,
        reward_support_min=reward_support_min,
        reward_support_max=reward_support_max,
        reward_support_delta=reward_support_delta,
        value_support_min=value_support_min,
        value_support_max=value_support_max,
        value_support_delta=value_support_delta,
        run_id=run_id,
        attempt_id=attempt_id,
    )


@app.local_entrypoint()
def main(
    mode: str = "dry",
    env: str = DEFAULT_ENV,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    seed: int = DEFAULT_SEED,
    opponent_policy: str = "random_uniform",
    ego_agent: str = "player_0",
    opponent_checkpoint: str | None = None,
    opponent_checkpoint_label: str | None = None,
    opponent_checkpoint_adapter: str | None = None,
    opponent_checkpoint_num_simulations: int | None = None,
    opponent_checkpoint_state_key: str | None = None,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    pong_episode_max_steps: int | None = DEFAULT_PONG_EPISODE_MAX_STEPS,
    pong_reset_profile: str = DEFAULT_PONG_RESET_PROFILE,
    pong_reset_pressure_agent: str = DEFAULT_PONG_RESET_PRESSURE_AGENT,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    n_evaluator_episode: int = DEFAULT_N_EVALUATOR_EPISODE,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    n_episode: int = DEFAULT_N_EPISODE,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
    random_collect_episode_num: int = DEFAULT_RANDOM_COLLECT_EPISODE_NUM,
    eps_greedy_exploration_in_collect: bool = DEFAULT_EPS_GREEDY_EXPLORATION_IN_COLLECT,
    eps_start: float = DEFAULT_EPS_START,
    eps_end: float = DEFAULT_EPS_END,
    eps_decay: int = DEFAULT_EPS_DECAY,
    fixed_temperature_value: float = DEFAULT_FIXED_TEMPERATURE_VALUE,
    td_steps: int | None = DEFAULT_TD_STEPS,
    num_unroll_steps: int | None = DEFAULT_NUM_UNROLL_STEPS,
    discount_factor: float | None = DEFAULT_DISCOUNT_FACTOR,
    support_scale: int | None = DEFAULT_SUPPORT_SCALE,
    reward_support_min: float | None = DEFAULT_REWARD_SUPPORT_MIN,
    reward_support_max: float | None = DEFAULT_REWARD_SUPPORT_MAX,
    reward_support_delta: float | None = DEFAULT_REWARD_SUPPORT_DELTA,
    value_support_min: float | None = DEFAULT_VALUE_SUPPORT_MIN,
    value_support_max: float | None = DEFAULT_VALUE_SUPPORT_MAX,
    value_support_delta: float | None = DEFAULT_VALUE_SUPPORT_DELTA,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> None:
    result = lightzero_dummy_pong_tiny_train_smoke.remote(
        mode=mode,
        env=env,
        feature_mode=feature_mode,
        seed=seed,
        opponent_policy=opponent_policy,
        ego_agent=ego_agent,
        opponent_checkpoint=opponent_checkpoint,
        opponent_checkpoint_label=opponent_checkpoint_label,
        opponent_checkpoint_adapter=opponent_checkpoint_adapter,
        opponent_checkpoint_num_simulations=opponent_checkpoint_num_simulations,
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
        max_env_step=max_env_step,
        pong_episode_max_steps=pong_episode_max_steps,
        pong_reset_profile=pong_reset_profile,
        pong_reset_pressure_agent=pong_reset_pressure_agent,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=n_evaluator_episode,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        n_episode=n_episode,
        game_segment_length=game_segment_length,
        random_collect_episode_num=random_collect_episode_num,
        eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        fixed_temperature_value=fixed_temperature_value,
        td_steps=td_steps,
        num_unroll_steps=num_unroll_steps,
        discount_factor=discount_factor,
        support_scale=support_scale,
        reward_support_min=reward_support_min,
        reward_support_max=reward_support_max,
        reward_support_delta=reward_support_delta,
        value_support_min=value_support_min,
        value_support_max=value_support_max,
        value_support_delta=value_support_delta,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
