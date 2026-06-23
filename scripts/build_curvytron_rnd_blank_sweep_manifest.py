#!/usr/bin/env python3
"""Build a no-tournament blank-canvas RND exploration-bonus sweep manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from curvyzero.contracts.curvytron import (
    COMPUTE_L4_T4_CPU40,
    CURVYTRON_BACKGROUND_GIF_FPS,
    CURVYTRON_COMMIT_ON_CHECKPOINT,
    CURVYTRON_DECISION_MS,
    CURVYTRON_DEFAULT_COLLECTOR_ENV_NUM,
    CURVYTRON_DEFAULT_MAX_ENV_STEP,
    CURVYTRON_DEFAULT_MAX_TRAIN_ITER,
    CURVYTRON_DEFAULT_N_EPISODE,
    CURVYTRON_DEFAULT_NUM_SIMULATIONS,
    CURVYTRON_DEFAULT_TRAIN_BATCH_SIZE,
    CURVYTRON_POLICY_BONUS_RENDER_MODE,
    CURVYTRON_POLICY_TRAIL_RENDER_MODE,
    CURVYTRON_SAVE_CKPT_AFTER_ITER,
    CURVYTRON_SOURCE_MAX_STEPS,
    CURVYTRON_TRAINING_TASK_ID,
    DEFAULT_CURVYTRON_TRAIN_APP_NAME,
    DEFAULT_LEARNER_SEAT_MODE,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
    TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT,
)
from curvyzero.training import exploration_bonus as xb


SCHEMA_ID = "curvyzero_curvytron_rnd_blank_sweep_manifest/v0"
ROW_SCHEMA_ID = "curvyzero_curvytron_rnd_blank_sweep_row/v0"
MODULE = "curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train"
TASK_ID = CURVYTRON_TRAINING_TASK_ID
DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_rnd_blank_sweep_manifests")
DEFAULT_MATRIX_NAME = "rnd-blank-sweep-20260519a"
MAX_MODAL_RUN_ID_LEN = 96
PROFILE_SWEEP = "rnd_blank_sweep"
PROFILE_METER_GATE = "rnd_blank_meter_gate"
PROFILE_CHOICES = (PROFILE_SWEEP, PROFILE_METER_GATE)


@dataclass(frozen=True)
class SweepPoint:
    label: str
    mode: str
    weight: float
    require_rnd_metrics: bool
    description: str


DEFAULT_SWEEP_POINTS: tuple[SweepPoint, ...] = (
    SweepPoint(
        "no-bonus",
        xb.EXPLORATION_BONUS_MODE_NONE,
        0.0,
        False,
        "no exploration bonus",
    ),
    SweepPoint(
        "measure-only",
        xb.EXPLORATION_BONUS_MODE_RND_METER_V0,
        0.0,
        True,
        "record exploration-bonus estimates without changing training targets",
    ),
    SweepPoint(
        "bonus-0p003",
        xb.EXPLORATION_BONUS_MODE_RND_REPLAY_TARGET_V0,
        0.003,
        True,
        "exploration bonus weight 0.003",
    ),
    SweepPoint(
        "bonus-0p01",
        xb.EXPLORATION_BONUS_MODE_RND_REPLAY_TARGET_V0,
        0.01,
        True,
        "exploration bonus weight 0.01",
    ),
    SweepPoint(
        "bonus-0p03",
        xb.EXPLORATION_BONUS_MODE_RND_REPLAY_TARGET_V0,
        0.03,
        True,
        "exploration bonus weight 0.03",
    ),
    SweepPoint(
        "bonus-0p10",
        xb.EXPLORATION_BONUS_MODE_RND_REPLAY_TARGET_V0,
        0.10,
        True,
        "exploration bonus weight 0.10",
    ),
    SweepPoint(
        "bonus-0p30",
        xb.EXPLORATION_BONUS_MODE_RND_REPLAY_TARGET_V0,
        0.30,
        True,
        "exploration bonus weight 0.30",
    ),
    SweepPoint(
        "bonus-0p60",
        xb.EXPLORATION_BONUS_MODE_RND_REPLAY_TARGET_V0,
        0.60,
        True,
        "exploration bonus weight 0.60",
    ),
    SweepPoint(
        "bonus-1p00",
        xb.EXPLORATION_BONUS_MODE_RND_REPLAY_TARGET_V0,
        1.00,
        True,
        "exploration bonus weight 1.00",
    ),
)


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_id(raw: str, *, label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-")
    if not safe:
        raise ValueError(f"{label} became empty")
    if len(safe) > MAX_MODAL_RUN_ID_LEN:
        digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:10]
        prefix_len = MAX_MODAL_RUN_ID_LEN - len(digest) - 1
        safe = f"{safe[:prefix_len].rstrip('-_.')}-{digest}"
    return safe


def _ref(*parts: str | Path) -> str:
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def _weight_label(weight: float) -> str:
    if weight < 0.01:
        text = f"{weight:.3f}"
    else:
        text = f"{weight:.2f}"
    return text.rstrip("0").rstrip(".")


def _weight_slug(weight: float) -> str:
    text = _weight_label(weight)
    if "." not in text:
        text = f"{text}.00"
    elif len(text.split(".", 1)[1]) == 1:
        text = f"{text}0"
    return text.replace(".", "p")


def _plain_point_name(point: SweepPoint, *, replica_index: int) -> str:
    copy_suffix = f" / copy {replica_index}" if replica_index else ""
    if point.mode == xb.EXPLORATION_BONUS_MODE_NONE:
        return f"No exploration bonus{copy_suffix}"
    if point.mode == xb.EXPLORATION_BONUS_MODE_RND_METER_V0:
        return f"Measure bonus only{copy_suffix}"
    return f"Exploration bonus {_weight_label(point.weight)}{copy_suffix}"


def _train_function_name(compute: str) -> str:
    if compute == "cpu":
        return "lightzero_curvytron_visual_survival_cpu"
    if compute == COMPUTE_L4_T4_CPU40:
        return "lightzero_curvytron_visual_survival_gpu_cpu40"
    if compute == "gpu-h100-cpu40":
        return "lightzero_curvytron_visual_survival_h100_cpu40"
    raise ValueError(f"unsupported compute {compute!r}")


def _sweep_points_from_args(args: argparse.Namespace) -> tuple[SweepPoint, ...]:
    if args.profile == PROFILE_METER_GATE:
        return DEFAULT_SWEEP_POINTS[:2]
    if not args.weights:
        return DEFAULT_SWEEP_POINTS
    points = [
        SweepPoint(
            "no-bonus",
            xb.EXPLORATION_BONUS_MODE_NONE,
            0.0,
            False,
            "no exploration bonus",
        ),
        SweepPoint(
            "measure-only",
            xb.EXPLORATION_BONUS_MODE_RND_METER_V0,
            0.0,
            True,
            "record exploration-bonus estimates without changing training targets",
        ),
    ]
    seen_positive: set[float] = set()
    for raw_weight in args.weights:
        weight = float(raw_weight)
        if weight <= 0.0:
            continue
        if weight in seen_positive:
            continue
        seen_positive.add(weight)
        points.append(
            SweepPoint(
                f"bonus-{_weight_slug(weight)}",
                xb.EXPLORATION_BONUS_MODE_RND_REPLAY_TARGET_V0,
                weight,
                True,
                f"exploration bonus weight {_weight_label(weight)}",
            )
        )
    return tuple(points)


def _train_kwargs(
    args: argparse.Namespace,
    *,
    run_id: str,
    attempt_id: str,
    seed: int,
    point: SweepPoint,
) -> dict[str, Any]:
    kwargs = {
        "mode": "train",
        "seed": seed,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "max_env_step": args.max_env_step,
        "max_train_iter": args.max_train_iter,
        "source_max_steps": CURVYTRON_SOURCE_MAX_STEPS,
        "decision_ms": CURVYTRON_DECISION_MS,
        "collector_env_num": args.collector_env_num,
        "evaluator_env_num": 1,
        "n_evaluator_episode": 1,
        "n_episode": args.n_episode,
        "num_simulations": args.num_simulations,
        "batch_size": args.batch_size,
        "model_support_cap": args.model_support_cap,
        "td_steps": args.td_steps,
        "lightzero_eval_freq": 0,
        "skip_lightzero_eval_in_profile": True,
        "profile_cuda_sync_enabled": False,
        "profile_allow_auto_resume": False,
        "profile_volume_commit": False,
        "lightzero_multi_gpu": False,
        "save_ckpt_after_iter": args.save_ckpt_after_iter,
        "commit_on_checkpoint": CURVYTRON_COMMIT_ON_CHECKPOINT,
        "stop_after_learner_train_calls": args.stop_after_learner_train_calls,
        "env_variant": "source_state_fixed_opponent",
        "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        "reward_outcome_alpha": 0.0,
        "source_state_trail_render_mode": CURVYTRON_POLICY_TRAIL_RENDER_MODE,
        "source_state_bonus_render_mode": CURVYTRON_POLICY_BONUS_RENDER_MODE,
        "learner_seat_mode": args.learner_seat_mode,
        "ego_action_straight_override_probability": 0.0,
        "policy_action_repeat_min": 1,
        "policy_action_repeat_max": 1,
        "policy_action_repeat_extra_probability": 0.0,
        "control_noise_profile_id": "none",
        "disable_death_for_profile": False,
        "opponent_death_mode": "normal",
        "opponent_runtime_mode": "blank_canvas_noop",
        "env_telemetry_stride": 1,
        "env_manager_type": args.env_manager_type,
        "opponent_policy_kind": "fixed_straight",
        "opponent_use_cuda": False,
        "opponent_checkpoint_ref": None,
        "opponent_snapshot_ref": None,
        "opponent_checkpoint_report_ref": None,
        "opponent_checkpoint_state_key": None,
        "initial_policy_checkpoint_ref": None,
        "initial_policy_checkpoint_state_key": None,
        "initial_policy_checkpoint_load_mode": "matching_shape",
        "opponent_mixture_spec": None,
        "opponent_assignment_ref": None,
        "opponent_assignment_refresh_interval_train_iter": 0,
        "opponent_assignment_refresh_ref": None,
        "own_checkpoint_opponent_refresh_enabled": False,
        "background_eval_enabled": True,
        "background_eval_launch_kind": "poller",
        "background_eval_compute": "cpu",
        "background_eval_id_prefix": "live_checkpoint",
        "background_eval_seed_count": args.background_eval_seed_count,
        "background_eval_seed_rng_seed": seed + 10_000_000,
        "background_eval_max_steps": CURVYTRON_SOURCE_MAX_STEPS,
        "background_eval_step_detail_limit": 4,
        "background_eval_num_simulations": args.background_eval_num_simulations,
        "background_eval_batch_size": args.background_eval_batch_size,
        "background_gif_enabled": True,
        "background_gif_seed_offset": 10_000,
        "background_gif_max_steps": args.background_gif_max_steps,
        "background_gif_frame_stride": args.background_gif_frame_stride,
        "background_gif_fps": CURVYTRON_BACKGROUND_GIF_FPS,
        "background_gif_scale": 4,
        "background_gif_frame_size": 128,
        "background_gif_collect_temperature": 1.0,
        "background_gif_collect_epsilon": 0.25,
        "exploration_bonus_mode": point.mode,
        "exploration_bonus_weight": float(point.weight),
        "exploration_bonus_feature_source": xb.RND_FEATURE_SOURCE_POLICY_GRAY64_LATEST_V0,
        "exploration_bonus_rnd_batch_size": args.rnd_batch_size,
        "exploration_bonus_rnd_update_per_collect": args.rnd_update_per_collect,
        "exploration_bonus_rnd_buffer_size": args.rnd_buffer_size,
        "exploration_bonus_rnd_learning_rate": args.rnd_learning_rate,
        "exploration_bonus_rnd_weight_decay": args.rnd_weight_decay,
        "exploration_bonus_rnd_input_norm": False,
        "require_rnd_metrics": point.require_rnd_metrics,
    }
    if point.mode == xb.EXPLORATION_BONUS_MODE_NONE:
        kwargs["require_rnd_metrics"] = False
    return kwargs


def _poller_kwargs(
    args: argparse.Namespace,
    *,
    run_id: str,
    attempt_id: str,
    seed: int,
) -> dict[str, Any]:
    exp_name_ref = _ref(
        "training",
        TASK_ID,
        run_id,
        "attempts",
        attempt_id,
        "train",
        "lightzero_exp",
    )
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "exp_name_ref": exp_name_ref,
        "seed": seed,
        "source_max_steps": CURVYTRON_SOURCE_MAX_STEPS,
        "env_variant": "source_state_fixed_opponent",
        "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        "reward_outcome_alpha": 0.0,
        "opponent_policy_kind": "fixed_straight",
        "opponent_checkpoint_ref": None,
        "opponent_snapshot_ref": None,
        "opponent_checkpoint_state_key": None,
        "opponent_mixture_spec": None,
        "opponent_assignment_ref": None,
        "opponent_death_mode": "normal",
        "opponent_runtime_mode": "blank_canvas_noop",
        "background_eval_compute": "cpu",
        "background_eval_id_prefix": "live_checkpoint",
        "background_eval_seed_count": args.background_eval_seed_count,
        "background_eval_seed_rng_seed": seed + 10_000_000,
        "background_eval_max_steps": CURVYTRON_SOURCE_MAX_STEPS,
        "background_eval_step_detail_limit": 4,
        "background_eval_num_simulations": args.background_eval_num_simulations,
        "background_eval_batch_size": args.background_eval_batch_size,
        "background_gif_enabled": True,
        "background_gif_seed_offset": 10_000,
        "background_gif_max_steps": args.background_gif_max_steps,
        "background_gif_frame_stride": args.background_gif_frame_stride,
        "background_gif_fps": CURVYTRON_BACKGROUND_GIF_FPS,
        "background_gif_scale": 4,
        "background_gif_frame_size": 128,
        "background_gif_collect_temperature": 1.0,
        "background_gif_collect_epsilon": 0.25,
        "poll_interval_sec": args.background_eval_poll_interval_sec,
        "stable_polls": 1,
        "max_runtime_sec": args.background_eval_poller_max_runtime_sec,
        "idle_after_train_done_sec": 60.0,
    }


def _row(
    args: argparse.Namespace,
    *,
    row_number: int,
    replica_index: int,
    point: SweepPoint,
) -> dict[str, Any]:
    seed = int(args.seed) + int(replica_index)
    row_id = f"r{row_number:03d}"
    label = f"{point.label}-copy{replica_index:02d}"
    run_id = _safe_id(
        f"{args.run_prefix}-{point.label}-copy{replica_index:02d}-s{seed}",
        label="run_id",
    )
    attempt_id = _safe_id(
        f"{args.attempt_prefix}-{point.label}-copy{replica_index:02d}-s{seed}",
        label="attempt_id",
    )
    train_kwargs = _train_kwargs(
        args,
        run_id=run_id,
        attempt_id=attempt_id,
        seed=seed,
        point=point,
    )
    poller_kwargs = _poller_kwargs(
        args,
        run_id=run_id,
        attempt_id=attempt_id,
        seed=seed,
    )
    spec = xb.normalize_exploration_bonus_spec(
        mode=point.mode,
        weight=point.weight,
        feature_source=train_kwargs["exploration_bonus_feature_source"],
        rnd_batch_size=train_kwargs["exploration_bonus_rnd_batch_size"],
        rnd_update_per_collect=train_kwargs["exploration_bonus_rnd_update_per_collect"],
        rnd_buffer_size=train_kwargs["exploration_bonus_rnd_buffer_size"],
        rnd_learning_rate=train_kwargs["exploration_bonus_rnd_learning_rate"],
        rnd_weight_decay=train_kwargs["exploration_bonus_rnd_weight_decay"],
        rnd_input_norm=train_kwargs["exploration_bonus_rnd_input_norm"],
    )
    train_ref = _ref("training", TASK_ID, run_id, "attempts", attempt_id, "train")
    return {
        "schema_id": ROW_SCHEMA_ID,
        "matrix_name": args.matrix_name,
        "row_id": row_id,
        "row_kind": "rnd_blank_sweep_training",
        "status": "ready_for_operator_launch_gate",
        "label": label,
        "plain_name": _plain_point_name(point, replica_index=replica_index),
        "run_id": run_id,
        "attempt_id": attempt_id,
        "mode": "train",
        "canonical_launcher": "scripts/submit_curvytron_survivaldiag_manifest.py",
        "calls_stock_train_muzero": point.mode == xb.EXPLORATION_BONUS_MODE_NONE,
        "trainer_entrypoint": xb.lightzero_trainer_entrypoint_ref(spec),
        "env_variant": "source_state_fixed_opponent",
        "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        "opponent_runtime_mode": "blank_canvas_noop",
        "opponent_death_mode": "normal",
        "opponent_source": "top_level_blank_canvas_noop",
        "initial_policy_checkpoint_ref": None,
        "initial_policy_checkpoint_source": {
            "source": "scratch_random_initialization",
            "checkpoint_ref": None,
        },
        "exploration_bonus": spec.as_dict(),
        "exploration_bonus_label": point.label,
        "exploration_bonus_description": point.description,
        "exploration_bonus_weight": float(point.weight),
        "replica_index": int(replica_index),
        "training_seed": seed,
        "deployed_app_submission": {
            "app_name": DEFAULT_CURVYTRON_TRAIN_APP_NAME,
            "train_function": _train_function_name(args.compute),
            "poller_function": "lightzero_curvytron_visual_survival_checkpoint_eval_poller",
            "spawn_order": ["poller", "train"],
        },
        "train_kwargs": train_kwargs,
        "poller_kwargs": poller_kwargs,
        "artifact_refs": {
            "summary": _ref(train_ref, "summary.json"),
            "progress_latest": _ref(train_ref, "progress_latest.json"),
            "checkpoint_root": _ref(train_ref, "lightzero_exp", "ckpt"),
            "background_eval_status": _ref(train_ref, "checkpoint_eval_poller.json"),
            "background_gif_status": _ref(train_ref, "background_gif_jobs.json"),
            "rnd_reward_model_metrics_latest": _ref(
                train_ref,
                "rnd_reward_model_metrics_latest.json",
            ),
            "rnd_reward_model_metrics_jsonl": _ref(
                train_ref,
                "rnd_reward_model_metrics.jsonl",
            ),
        },
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    points = _sweep_points_from_args(args)
    if args.profile == PROFILE_SWEEP and len(points) < 3:
        raise ValueError("RND blank sweep must include baseline, meter, and positive weights")
    if args.profile == PROFILE_METER_GATE and len(points) != 2:
        raise ValueError("RND blank meter gate must include exactly stock and meter rows")
    rows = []
    row_number = 0
    for replica_index in range(int(args.replicas)):
        for point in points:
            row_number += 1
            rows.append(
                _row(
                    args,
                    row_number=row_number,
                    replica_index=replica_index,
                    point=point,
                )
            )
    manifest = {
        "schema_id": SCHEMA_ID,
        "status": "ready_for_operator_launch_gate",
        "matrix_name": args.matrix_name,
        "matrix_profile": args.profile,
        "generated_at": _utc_timestamp(),
        "row_count": len(rows),
        "tournament": {
            "enabled": False,
            "intake_enabled": False,
            "reason": "explicit user request: no tournament",
        },
        "axes": {
            "exploration_bonus_points": [
                {
                    "label": point.label,
                    "mode": point.mode,
                    "weight": float(point.weight),
                    "require_rnd_metrics": bool(point.require_rnd_metrics),
                }
                for point in points
            ],
            "replicas": int(args.replicas),
        },
        "fixed_knobs": {
            "compute": args.compute,
            "seed": int(args.seed),
            "collector_env_num": args.collector_env_num,
            "n_episode": args.n_episode,
            "num_simulations": args.num_simulations,
            "batch_size": args.batch_size,
            "model_support_cap": args.model_support_cap,
            "td_steps": args.td_steps,
            "max_train_iter": args.max_train_iter,
            "max_env_step": args.max_env_step,
            "source_max_steps": CURVYTRON_SOURCE_MAX_STEPS,
            "save_ckpt_after_iter": args.save_ckpt_after_iter,
            "source_state_trail_render_mode": CURVYTRON_POLICY_TRAIL_RENDER_MODE,
            "source_state_bonus_render_mode": CURVYTRON_POLICY_BONUS_RENDER_MODE,
            "learner_seat_mode": args.learner_seat_mode,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
            "opponent_runtime_mode": "blank_canvas_noop",
            "opponent_death_mode": "normal",
            "opponent_assignment_refresh_interval_train_iter": 0,
            "background_eval_enabled": True,
            "background_gif_enabled": True,
            "background_gif_max_steps": args.background_gif_max_steps,
            "background_gif_frame_stride": args.background_gif_frame_stride,
        },
        "guards": {
            "deployed_app_name": DEFAULT_CURVYTRON_TRAIN_APP_NAME,
            "launch_script": "scripts/submit_curvytron_survivaldiag_manifest.py",
            "operator_launch_gate_required": True,
            "expected_row_count": len(points) * int(args.replicas),
            "modal_launch_performed": False,
            "tournament_enabled": False,
            "assignment_refresh_enabled": False,
            "all_rows_blank_canvas_noop": True,
            "all_rows_background_gif_enabled": True,
            "positive_rnd_mode_is_experimental": args.profile == PROFILE_SWEEP,
        },
        "rows": rows,
    }
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    rows = list(manifest["rows"])
    expected = int(manifest["guards"]["expected_row_count"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, got {len(rows)}")
    profile = str(manifest["matrix_profile"])
    if profile == PROFILE_SWEEP and expected < 3:
        raise ValueError("RND blank sweep expected at least three rows")
    if profile == PROFILE_METER_GATE and expected < 2:
        raise ValueError("RND blank meter gate expected at least two rows")
    run_ids = [str(row["run_id"]) for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate run_id values")
    modes = [str(row["train_kwargs"]["exploration_bonus_mode"]) for row in rows]
    weights = [float(row["train_kwargs"]["exploration_bonus_weight"]) for row in rows]
    if xb.EXPLORATION_BONUS_MODE_NONE not in modes:
        raise ValueError("sweep lacks stock zero baseline")
    if xb.EXPLORATION_BONUS_MODE_RND_METER_V0 not in modes:
        raise ValueError("sweep lacks zero-weight RND meter row")
    has_positive_mode = any(
        mode == xb.EXPLORATION_BONUS_MODE_RND_REPLAY_TARGET_V0 for mode in modes
    )
    if profile == PROFILE_SWEEP and not has_positive_mode:
        raise ValueError("sweep lacks positive RND target rows")
    if profile == PROFILE_METER_GATE and has_positive_mode:
        raise ValueError("meter gate must not include positive RND target rows")
    if profile == PROFILE_SWEEP and (min(weights) != 0.0 or max(weights) < 1.0):
        raise ValueError("sweep must span from zero through weight 1.0")
    if profile == PROFILE_METER_GATE and set(weights) != {0.0}:
        raise ValueError("meter gate weights must all be zero")
    for row in rows:
        train_kwargs = row["train_kwargs"]
        poller_kwargs = row["poller_kwargs"]
        missing = [
            key for key in TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT if key not in train_kwargs
        ]
        if missing:
            raise ValueError(f"row {row['row_id']} missing train kwargs {missing}")
        if len(row["run_id"]) > MAX_MODAL_RUN_ID_LEN:
            raise ValueError(f"row {row['row_id']} overlong run_id")
        if len(row["attempt_id"]) > MAX_MODAL_RUN_ID_LEN:
            raise ValueError(f"row {row['row_id']} overlong attempt_id")
        if train_kwargs.get("initial_policy_checkpoint_ref") is not None:
            raise ValueError(f"row {row['row_id']} must start from scratch")
        if row.get("initial_policy_checkpoint_source", {}).get("source") != (
            "scratch_random_initialization"
        ):
            raise ValueError(f"row {row['row_id']} must declare scratch initialization")
        for key, expected_value in (
            ("opponent_runtime_mode", "blank_canvas_noop"),
            ("opponent_death_mode", "normal"),
            ("opponent_policy_kind", "fixed_straight"),
            ("reward_variant", REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME),
        ):
            if train_kwargs.get(key) != expected_value:
                raise ValueError(f"row {row['row_id']} train {key} mismatch")
            if poller_kwargs.get(key) != expected_value:
                raise ValueError(f"row {row['row_id']} poller {key} mismatch")
        if train_kwargs.get("opponent_assignment_ref") is not None:
            raise ValueError(f"row {row['row_id']} must not use assignment refs")
        if train_kwargs.get("opponent_assignment_refresh_interval_train_iter") != 0:
            raise ValueError(f"row {row['row_id']} must disable assignment refresh")
        if train_kwargs.get("own_checkpoint_opponent_refresh_enabled") is not False:
            raise ValueError(f"row {row['row_id']} must disable own-checkpoint refresh")
        if train_kwargs.get("background_gif_enabled") is not True:
            raise ValueError(f"row {row['row_id']} train GIFs are disabled")
        if poller_kwargs.get("background_gif_enabled") is not True:
            raise ValueError(f"row {row['row_id']} poller GIFs are disabled")
        spec = xb.normalize_exploration_bonus_spec(
            mode=train_kwargs["exploration_bonus_mode"],
            weight=train_kwargs["exploration_bonus_weight"],
            feature_source=train_kwargs["exploration_bonus_feature_source"],
            rnd_batch_size=train_kwargs["exploration_bonus_rnd_batch_size"],
            rnd_update_per_collect=train_kwargs["exploration_bonus_rnd_update_per_collect"],
            rnd_buffer_size=train_kwargs["exploration_bonus_rnd_buffer_size"],
            rnd_learning_rate=train_kwargs["exploration_bonus_rnd_learning_rate"],
            rnd_weight_decay=train_kwargs["exploration_bonus_rnd_weight_decay"],
            rnd_input_norm=train_kwargs["exploration_bonus_rnd_input_norm"],
        )
        if row["exploration_bonus"]["config_hash"] != spec.config_hash():
            raise ValueError(f"row {row['row_id']} exploration config hash mismatch")
        if spec.enabled and train_kwargs.get("require_rnd_metrics") is not True:
            raise ValueError(f"row {row['row_id']} RND row must require metrics")
        if not spec.enabled and train_kwargs.get("require_rnd_metrics") is not False:
            raise ValueError(f"row {row['row_id']} stock row must not require RND metrics")


def _write_outputs(manifest: Mapping[str, Any], output_root: Path) -> dict[str, str]:
    output_dir = output_root / str(manifest["matrix_name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{manifest['matrix_name']}.json"
    rows_path = output_dir / f"{manifest['matrix_name']}.rows.jsonl"
    commands_path = output_dir / f"{manifest['matrix_name']}.submit.commands.txt"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in manifest["rows"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    commands_path.write_text(
        "\n".join(
            [
                "# Dry-run review:",
                (
                    "uv run --extra modal python "
                    f"scripts/submit_curvytron_survivaldiag_manifest.py {manifest_path}"
                ),
                "",
                "# Launch all rows after review:",
                (
                    "uv run --extra modal python scripts/submit_curvytron_survivaldiag_manifest.py "
                    f"{manifest_path} --allow-launch"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "manifest_json": str(manifest_path),
        "rows_jsonl": str(rows_path),
        "submit_commands_txt": str(commands_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default=PROFILE_SWEEP)
    parser.add_argument("--matrix-name", default=DEFAULT_MATRIX_NAME)
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="Run-id prefix. Defaults to --matrix-name so separate manifests do not collide.",
    )
    parser.add_argument(
        "--attempt-prefix",
        default=None,
        help="Attempt-id prefix. Defaults to try-{run-prefix}.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stdout-only", action="store_true")
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--compute", default=COMPUTE_L4_T4_CPU40)
    parser.add_argument("--collector-env-num", type=int, default=CURVYTRON_DEFAULT_COLLECTOR_ENV_NUM)
    parser.add_argument("--n-episode", type=int, default=CURVYTRON_DEFAULT_N_EPISODE)
    parser.add_argument("--num-simulations", type=int, default=CURVYTRON_DEFAULT_NUM_SIMULATIONS)
    parser.add_argument("--batch-size", type=int, default=CURVYTRON_DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--model-support-cap", type=int, default=None)
    parser.add_argument("--td-steps", type=int, default=None)
    parser.add_argument("--max-env-step", type=int, default=CURVYTRON_DEFAULT_MAX_ENV_STEP)
    parser.add_argument("--max-train-iter", type=int, default=CURVYTRON_DEFAULT_MAX_TRAIN_ITER)
    parser.add_argument("--save-ckpt-after-iter", type=int, default=CURVYTRON_SAVE_CKPT_AFTER_ITER)
    parser.add_argument("--stop-after-learner-train-calls", type=int, default=0)
    parser.add_argument("--env-manager-type", choices=("base", "subprocess"), default="subprocess")
    parser.add_argument("--learner-seat-mode", default=DEFAULT_LEARNER_SEAT_MODE)
    parser.add_argument("--background-eval-seed-count", type=int, default=3)
    parser.add_argument("--background-eval-num-simulations", type=int, default=8)
    parser.add_argument("--background-eval-batch-size", type=int, default=64)
    parser.add_argument("--background-eval-poll-interval-sec", type=float, default=120.0)
    parser.add_argument("--background-eval-poller-max-runtime-sec", type=float, default=6 * 60 * 60)
    parser.add_argument("--background-gif-max-steps", type=int, default=4096)
    parser.add_argument("--background-gif-frame-stride", type=int, default=4)
    parser.add_argument("--rnd-batch-size", type=int, default=64)
    parser.add_argument("--rnd-update-per-collect", type=int, default=1)
    parser.add_argument("--rnd-buffer-size", type=int, default=100_000)
    parser.add_argument("--rnd-learning-rate", type=float, default=3e-4)
    parser.add_argument("--rnd-weight-decay", type=float, default=1e-4)
    args = parser.parse_args(argv)
    if args.run_prefix is None:
        args.run_prefix = str(args.matrix_name)
    if args.attempt_prefix is None:
        args.attempt_prefix = f"try-{args.run_prefix}"
    if int(args.replicas) < 1:
        raise ValueError("--replicas must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_manifest(args)
    if args.stdout_only:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    outputs = _write_outputs(manifest, args.output_root)
    print(
        json.dumps(
            {
                "ok": True,
                "matrix_name": manifest["matrix_name"],
                "profile": manifest["matrix_profile"],
                "row_count": manifest["row_count"],
                "outputs": outputs,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
