#!/usr/bin/env python3
"""Build the May 14 tonight 18-run CurvyTron training manifest.

The manifest is dry-run/review material. Launch, if desired, should go through
``scripts/submit_curvytron_survivaldiag_manifest.py``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from curvyzero.contracts.curvytron import (
    COMPUTE_H100_CPU40,
    CURVYTRON_ASSIGNMENT_REFRESH_INTERVAL_TRAIN_ITER as ASSIGNMENT_REFRESH_INTERVAL_TRAIN_ITER,
    CURVYTRON_BACKGROUND_GIF_FPS as BACKGROUND_GIF_FPS,
    CURVYTRON_COMMIT_ON_CHECKPOINT as COMMIT_ON_CHECKPOINT,
    CURVYTRON_DECISION_MS as DECISION_MS,
    CURVYTRON_MAIN_REWARD_VARIANTS,
    CURVYTRON_POLICY_BONUS_RENDER_MODE as BONUS_RENDER_SIMPLE_SYMBOLS,
    CURVYTRON_POLICY_TRAIL_RENDER_MODE as RENDER_POLICY,
    CURVYTRON_SAVE_CKPT_AFTER_ITER as SAVE_CKPT_AFTER_ITER,
    CURVYTRON_SOURCE_MAX_STEPS as SOURCE_MAX_STEPS,
    CURVYTRON_TRAINING_TASK_ID,
    DEFAULT_LEARNER_SEAT_MODE,
    LEARNER_SEAT_MODE_CHOICES,
    TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT,
    curvytron_train_app_name,
)
from curvyzero.training.opponent_mixture import (
    OPPONENT_MIXTURE_SCHEMA_ID,
    OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
    OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
    OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
    OPPONENT_RUNTIME_MODE_NORMAL,
    parse_opponent_mixture_spec,
)
from curvyzero.training.opponent_registry import (
    OPPONENT_ASSIGNMENT_SCHEMA_ID,
    canonical_assignment_json_sha256,
    parse_opponent_assignment_snapshot,
)


MODULE = "curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train"
APP_NAME = curvytron_train_app_name()
TASK_ID = CURVYTRON_TRAINING_TASK_ID
SCHEMA_ID = "curvyzero_curvytron_tonight18_manifest/v0"
ROW_SCHEMA_ID = "curvyzero_curvytron_tonight18_manifest_row/v0"
OPPONENT_SOURCE_MIXTURE = "mixture"
OPPONENT_SOURCE_ASSIGNMENT = "assignment"
DEFAULT_OPPONENT_SOURCE = OPPONENT_SOURCE_ASSIGNMENT

DEFAULT_MATRIX_NAME = "curvy-restart18-allv2-20260515a"
DEFAULT_RUN_PREFIX = "curvy-r18v2"
DEFAULT_ATTEMPT_PREFIX = "try-r18v2"
DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_tonight18_manifests")
DEFAULT_ASSIGNMENT_TARGET_VOLUME = "control"
DEFAULT_ASSIGNMENT_REFRESH_POINTER_VOLUME = "control"

MODE_TRAIN = "train"
ENV_SOURCE_STATE_FIXED_OPPONENT = "source_state_fixed_opponent"
MAX_MODAL_RUN_ID_LEN = 96

REWARD_VARIANTS: tuple[str, ...] = CURVYTRON_MAIN_REWARD_VARIANTS


@dataclass(frozen=True)
class OpponentRecipe:
    recipe_id: str
    description: str
    weights: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class NoiseMode:
    mode_id: str
    tag: str
    description: str
    ego_action_straight_override_probability: float
    policy_action_repeat_min: int
    policy_action_repeat_max: int
    policy_action_repeat_extra_probability: float
    control_noise_profile_id: str


OPPONENT_RECIPES: tuple[OpponentRecipe, ...] = (
    OpponentRecipe(
        recipe_id="blank10-wall10-rank2_25-rank1_55",
        description=(
            "10% blank canvas, 10% proactive wall-avoidant immortal, "
            "25% rank2 frozen, 55% rank1 frozen"
        ),
        weights=(
            ("blank", 10),
            ("wall_avoidant_immortal", 10),
            ("rank2", 25),
            ("rank1", 55),
        ),
    ),
    OpponentRecipe(
        recipe_id="blank10-wall15-rank4_10-rank3_15-rank2_20-rank1_30",
        description=(
            "10% blank canvas, 15% proactive wall-avoidant immortal, "
            "10% rank4 frozen, 15% rank3 frozen, 20% rank2 frozen, "
            "30% rank1 frozen"
        ),
        weights=(
            ("blank", 10),
            ("wall_avoidant_immortal", 15),
            ("rank4", 10),
            ("rank3", 15),
            ("rank2", 20),
            ("rank1", 30),
        ),
    ),
    OpponentRecipe(
        recipe_id="blank20-wall20-rank1_60",
        description=(
            "20% blank canvas, 20% proactive wall-avoidant immortal, "
            "60% rank1 frozen"
        ),
        weights=(("blank", 20), ("wall_avoidant_immortal", 20), ("rank1", 60)),
    ),
)

NOISE_MODES: tuple[NoiseMode, ...] = (
    NoiseMode(
        mode_id="clean",
        tag="clean",
        description="no straight override and no held-action repeat",
        ego_action_straight_override_probability=0.0,
        policy_action_repeat_min=1,
        policy_action_repeat_max=1,
        policy_action_repeat_extra_probability=0.0,
        control_noise_profile_id="none",
    ),
    NoiseMode(
        mode_id="straight_override_p10_repeat_p10",
        tag="so10rep10",
        description="10% straight override plus 10% one-extra-step action repeat",
        ego_action_straight_override_probability=0.10,
        policy_action_repeat_min=1,
        policy_action_repeat_max=2,
        policy_action_repeat_extra_probability=0.10,
        control_noise_profile_id="straight_override_p10_repeat_p10",
    ),
)

REWARD_TAGS = {
    "sparse_outcome": "sparse",
    "survival_plus_bonus_no_outcome": "survbonusnoout",
    "survival_plus_bonus_plus_outcome": "survbonusout",
}


def _safe_id(value: str, *, label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    if not safe:
        raise ValueError(f"{label} became empty")
    if len(safe) > MAX_MODAL_RUN_ID_LEN:
        digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:10]
        prefix_len = MAX_MODAL_RUN_ID_LEN - len(digest) - 1
        safe = f"{safe[:prefix_len].rstrip('-_.')}-{digest}"
    return safe


def _ref(*parts: str | Path) -> str:
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def _derived_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 2_000_000_000


def _load_top_checkpoints(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ratings = payload.get("ratings")
    if ratings is None:
        ratings = payload.get("rows")
    if not isinstance(ratings, list) or len(ratings) < 4:
        raise ValueError(f"{path} must contain at least four ratings or leaderboard rows")
    top: dict[str, dict[str, Any]] = {}
    for expected_rank, row in enumerate(ratings[:4], start=1):
        if not isinstance(row, dict):
            raise ValueError(f"rating row {expected_rank} is not an object")
        rank = int(row.get("rank", expected_rank))
        if rank != expected_rank:
            raise ValueError(
                f"ratings[:4] must be ordered ranks 1-4; got rank {rank} at slot {expected_rank}"
            )
        status = str(row.get("status") or "")
        if status != "active":
            raise ValueError(
                f"rank{rank} must be active before tonight18 launch; got status {status!r}"
            )
        checkpoint_ref = str(row.get("checkpoint_ref") or "")
        if "latest" in checkpoint_ref or "ckpt_best" in checkpoint_ref:
            raise ValueError(f"rank{rank} checkpoint ref is mutable: {checkpoint_ref}")
        if not re.fullmatch(r"iteration_\d+\.pth\.tar", Path(checkpoint_ref).name):
            raise ValueError(f"rank{rank} ref must end in iteration_N.pth.tar: {checkpoint_ref}")
        top[f"rank{rank}"] = {
            "rank": rank,
            "checkpoint_id": row.get("checkpoint_id"),
            "rating": row.get("rating"),
            "status": status,
            "run_id": row.get("run_id"),
            "attempt_id": row.get("attempt_id"),
            "iteration": row.get("iteration"),
            "checkpoint_ref": checkpoint_ref,
        }
    return top


def _assignment_bank_run_id(args: argparse.Namespace) -> str:
    if args.assignment_bank_run_id:
        return _safe_id(args.assignment_bank_run_id, label="assignment_bank_run_id")
    return _safe_id(f"{args.run_prefix}-assignments", label="assignment_bank_run_id")


def _assignment_bank_attempt_id(args: argparse.Namespace) -> str:
    if args.assignment_bank_attempt_id:
        return _safe_id(args.assignment_bank_attempt_id, label="assignment_bank_attempt_id")
    return _safe_id(f"{args.attempt_prefix}-assignments", label="assignment_bank_attempt_id")


def _assignment_id(args: argparse.Namespace, recipe: OpponentRecipe) -> str:
    return _safe_id(
        f"{args.matrix_name}-{recipe.recipe_id}",
        label="assignment_id",
    )


def _assignment_ref(
    *,
    args: argparse.Namespace,
    assignment_id: str,
) -> str:
    prefix = "control:" if args.assignment_target_volume == "control" else ""
    return _ref(
        f"{prefix}training",
        TASK_ID,
        _assignment_bank_run_id(args),
        "attempts",
        _assignment_bank_attempt_id(args),
        "opponents",
        "assignments",
        assignment_id,
        "assignment.json",
    )


def _assignment_refresh_pointer_run_id(args: argparse.Namespace) -> str:
    value = args.assignment_refresh_pointer_run_id or f"{args.run_prefix}-control"
    return _safe_id(value, label="assignment_refresh_pointer_run_id")


def _assignment_refresh_pointer_attempt_id(args: argparse.Namespace) -> str:
    value = args.assignment_refresh_pointer_attempt_id or f"{args.attempt_prefix}-control"
    return _safe_id(value, label="assignment_refresh_pointer_attempt_id")


def _assignment_refresh_pointer_ref(
    *,
    args: argparse.Namespace,
    recipe: OpponentRecipe,
) -> str:
    volume_prefix = (
        "control:" if args.assignment_refresh_pointer_volume == "control" else "runs:"
    )
    return _ref(
        f"{volume_prefix}training",
        TASK_ID,
        _assignment_refresh_pointer_run_id(args),
        "attempts",
        _assignment_refresh_pointer_attempt_id(args),
        "opponents",
        "refresh_pointers",
        recipe.recipe_id,
        "refresh_pointer.json",
    )


def _assignment_source_ref(args: argparse.Namespace) -> str:
    return args.assignment_source_ref or str(args.ratings_snapshot)


def _assignment_artifact(
    *,
    args: argparse.Namespace,
    recipe: OpponentRecipe,
    top_checkpoints: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    assignment_id = _assignment_id(args, recipe)
    seed = _derived_seed(args.matrix_name, recipe.recipe_id, "assignment")
    mixture = _mixture_spec(recipe, top_checkpoints=top_checkpoints, seed=seed)
    assignment = {
        "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
        "assignment_id": assignment_id,
        "source_epoch": None,
        "source_ref": _assignment_source_ref(args),
        "seed": int(mixture["seed"]),
        "entries": mixture["entries"],
    }
    parse_opponent_assignment_snapshot(assignment)
    assignment_sha256 = canonical_assignment_json_sha256(assignment)
    audit = {
        "schema_id": "curvyzero_opponent_assignment_audit/v0",
        "assignment_id": assignment_id,
        "assignment_sha256": assignment_sha256,
        "source_leaderboard": {
            "leaderboard_id": args.leaderboard_id,
            "snapshot_id": args.leaderboard_snapshot_id,
            "snapshot_ref": _assignment_source_ref(args),
            "snapshot_sha256": args.leaderboard_snapshot_sha256,
        },
        "selection": {
            "strategy_id": "tonight18_recipe_slots_v0",
            "recipe_id": recipe.recipe_id,
            "recipe_description": recipe.description,
            "recipe_weights": dict(recipe.weights),
            "seed": seed,
        },
    }
    if not all(audit["source_leaderboard"].values()):
        audit.pop("source_leaderboard")
    return {
        "assignment_id": assignment_id,
        "assignment_ref": _assignment_ref(args=args, assignment_id=assignment_id),
        "assignment_sha256": assignment_sha256,
        "recipe_id": recipe.recipe_id,
        "assignment": assignment,
        "audit": audit,
    }


def _checkpoint_entry(
    name: str,
    *,
    weight: int,
    top_checkpoints: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    checkpoint = top_checkpoints[name]
    checkpoint_ref = str(checkpoint["checkpoint_ref"])
    return {
        "name": name,
        "age_label": name,
        "weight": weight,
        "opponent_policy_kind": "frozen_lightzero_checkpoint",
        "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_NORMAL,
        "opponent_immortal": False,
        "opponent_checkpoint_ref": checkpoint_ref,
        "opponent_snapshot_ref": f"tonight18_{name}_{Path(checkpoint_ref).stem}",
        "opponent_policy_seed": _derived_seed("opponent", name, seed),
        "tags": {
            "rank": checkpoint["rank"],
            "rating": checkpoint["rating"],
            "checkpoint_id": checkpoint["checkpoint_id"],
        },
    }


def _mixture_spec(
    recipe: OpponentRecipe,
    *,
    top_checkpoints: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for component, weight in recipe.weights:
        if component == "blank":
            entries.append(
                {
                    "name": "blank",
                    "age_label": "blank_canvas",
                    "weight": weight,
                    "opponent_policy_kind": "fixed_straight",
                    "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
                    "opponent_immortal": True,
                }
            )
            continue
        if component == "wall_avoidant_immortal":
            entries.append(
                {
                    "name": "wall_avoidant_immortal",
                    "age_label": "hardcoded_wall_avoidant_immortal",
                    "weight": weight,
                    "opponent_policy_kind": OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
                    "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_NORMAL,
                    "opponent_immortal": True,
                    "opponent_wall_avoidant_safe_margin": 20.0,
                }
            )
            continue
        entries.append(
            _checkpoint_entry(
                component,
                weight=weight,
                top_checkpoints=top_checkpoints,
                seed=seed,
            )
        )
    spec = {
        "schema_id": OPPONENT_MIXTURE_SCHEMA_ID,
        "selection_unit": "episode_reset",
        "seed": _derived_seed("mixture", recipe.recipe_id, seed),
        "entries": entries,
    }
    parsed = parse_opponent_mixture_spec(spec)
    if parsed is None:
        raise ValueError(f"recipe {recipe.recipe_id} produced an empty mixture")
    return _public_mixture_spec(parsed)


def _public_mixture_spec(parsed: dict[str, Any]) -> dict[str, Any]:
    """Keep manifest slot intent clean; env-only fields are derived later."""

    public = copy.deepcopy(parsed)
    for entry in public["entries"]:
        entry.pop("opponent_death_mode", None)
    return public


def _train_function_name(compute: str) -> str:
    if compute == COMPUTE_H100_CPU40:
        return "lightzero_curvytron_visual_survival_h100_cpu40"
    raise ValueError(f"unsupported compute for tonight18 manifest: {compute!r}")


def _train_kwargs(
    *,
    args: argparse.Namespace,
    run_id: str,
    attempt_id: str,
    seed: int,
    reward_variant: str,
    noise: NoiseMode,
    mixture_spec: dict[str, Any] | None,
    opponent_assignment_ref: str | None,
    opponent_assignment_refresh_ref: str | None,
    initial_policy_checkpoint_ref: str | None,
) -> dict[str, Any]:
    kwargs = {
        "mode": MODE_TRAIN,
        "seed": seed,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "max_env_step": args.max_env_step,
        "max_train_iter": args.max_train_iter,
        "source_max_steps": SOURCE_MAX_STEPS,
        "decision_ms": DECISION_MS,
        "collector_env_num": 256,
        "evaluator_env_num": 1,
        "n_evaluator_episode": 1,
        "n_episode": 256,
        "num_simulations": 8,
        "batch_size": 32,
        "lightzero_eval_freq": 0,
        "skip_lightzero_eval_in_profile": True,
        "profile_cuda_sync_enabled": False,
        "profile_allow_auto_resume": False,
        "profile_volume_commit": False,
        "lightzero_multi_gpu": False,
        "save_ckpt_after_iter": SAVE_CKPT_AFTER_ITER,
        "commit_on_checkpoint": COMMIT_ON_CHECKPOINT,
        "stop_after_learner_train_calls": args.stop_after_learner_train_calls,
        "env_variant": ENV_SOURCE_STATE_FIXED_OPPONENT,
        "reward_variant": reward_variant,
        "source_state_trail_render_mode": RENDER_POLICY,
        "source_state_bonus_render_mode": BONUS_RENDER_SIMPLE_SYMBOLS,
        "learner_seat_mode": args.learner_seat_mode,
        "ego_action_straight_override_probability": (
            noise.ego_action_straight_override_probability
        ),
        "policy_action_repeat_min": noise.policy_action_repeat_min,
        "policy_action_repeat_max": noise.policy_action_repeat_max,
        "policy_action_repeat_extra_probability": noise.policy_action_repeat_extra_probability,
        "control_noise_profile_id": noise.control_noise_profile_id,
        "disable_death_for_profile": False,
        "opponent_death_mode": "normal",
        "opponent_runtime_mode": "normal",
        "env_telemetry_stride": 1,
        "env_manager_type": args.env_manager_type,
        "opponent_policy_kind": "fixed_straight",
        "opponent_use_cuda": False,
        "opponent_checkpoint_ref": None,
        "opponent_snapshot_ref": None,
        "opponent_checkpoint_report_ref": None,
        "opponent_checkpoint_state_key": None,
        "initial_policy_checkpoint_ref": initial_policy_checkpoint_ref,
        "initial_policy_checkpoint_state_key": None,
        "initial_policy_checkpoint_load_mode": "matching_shape",
        "opponent_mixture_spec": mixture_spec,
        "opponent_assignment_ref": opponent_assignment_ref,
        "background_eval_enabled": True,
        "background_eval_launch_kind": "poller",
        "background_eval_compute": "cpu",
        "background_eval_id_prefix": "live_checkpoint",
        "background_eval_seed_count": args.background_eval_seed_count,
        "background_eval_seed_rng_seed": _derived_seed("eval", seed),
        "background_eval_max_steps": SOURCE_MAX_STEPS,
        "background_eval_step_detail_limit": 4,
        "background_eval_num_simulations": 8,
        "background_eval_batch_size": 64,
        "background_gif_enabled": True,
        "background_gif_seed_offset": 10_000,
        "background_gif_max_steps": 4096,
        "background_gif_frame_stride": 4,
        "background_gif_fps": BACKGROUND_GIF_FPS,
        "background_gif_scale": 4,
        "background_gif_frame_size": 128,
        "background_gif_collect_temperature": 1.0,
        "background_gif_collect_epsilon": 0.25,
    }
    if args.assignment_refresh_interval_train_iter > 0:
        if not opponent_assignment_ref:
            raise ValueError(
                "assignment refresh requires --opponent-source assignment for tonight18"
            )
        if not opponent_assignment_refresh_ref:
            raise ValueError("assignment refresh requires a non-empty refresh pointer ref")
        kwargs["opponent_assignment_refresh_interval_train_iter"] = (
            args.assignment_refresh_interval_train_iter
        )
        kwargs["opponent_assignment_refresh_ref"] = opponent_assignment_refresh_ref
    return kwargs


def _poller_kwargs(
    *,
    args: argparse.Namespace,
    run_id: str,
    attempt_id: str,
    seed: int,
    reward_variant: str,
    mixture_spec: dict[str, Any] | None,
    opponent_assignment_ref: str | None,
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
        "source_max_steps": SOURCE_MAX_STEPS,
        "env_variant": ENV_SOURCE_STATE_FIXED_OPPONENT,
        "reward_variant": reward_variant,
        "opponent_policy_kind": "fixed_straight",
        "opponent_checkpoint_ref": None,
        "opponent_snapshot_ref": None,
        "opponent_checkpoint_state_key": None,
        "opponent_mixture_spec": mixture_spec,
        "opponent_assignment_ref": opponent_assignment_ref,
        "opponent_death_mode": "normal",
        "opponent_runtime_mode": "normal",
        "background_eval_compute": "cpu",
        "background_eval_id_prefix": "live_checkpoint",
        "background_eval_seed_count": args.background_eval_seed_count,
        "background_eval_seed_rng_seed": _derived_seed("eval", seed),
        "background_eval_max_steps": SOURCE_MAX_STEPS,
        "background_eval_step_detail_limit": 4,
        "background_eval_num_simulations": 8,
        "background_eval_batch_size": 64,
        "background_gif_enabled": True,
        "background_gif_seed_offset": 10_000,
        "background_gif_max_steps": 4096,
        "background_gif_frame_stride": 4,
        "background_gif_fps": BACKGROUND_GIF_FPS,
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
    *,
    args: argparse.Namespace,
    row_number: int,
    reward_variant: str,
    recipe: OpponentRecipe,
    noise: NoiseMode,
    top_checkpoints: dict[str, dict[str, Any]],
    assignment_artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    reward_tag = REWARD_TAGS[reward_variant]
    seed = _derived_seed(args.matrix_name, reward_variant, recipe.recipe_id, noise.mode_id)
    preview_mixture = _mixture_spec(recipe, top_checkpoints=top_checkpoints, seed=seed)
    opponent_assignment_ref = (
        str(assignment_artifact["assignment_ref"]) if assignment_artifact is not None else None
    )
    opponent_assignment_refresh_ref = (
        _assignment_refresh_pointer_ref(args=args, recipe=recipe)
        if opponent_assignment_ref and args.assignment_refresh_interval_train_iter > 0
        else None
    )
    initial_policy_checkpoint_ref = str(top_checkpoints["rank1"]["checkpoint_ref"])
    mixture_spec = None if opponent_assignment_ref else preview_mixture
    row_id = f"r{row_number:03d}"
    label = f"{reward_tag}-{recipe.recipe_id}-{noise.tag}"
    run_id = _safe_id(f"{args.run_prefix}-{label}-s{seed}", label="run_id")
    attempt_id = _safe_id(f"{args.attempt_prefix}-{label}-s{seed}", label="attempt_id")
    train_kwargs = _train_kwargs(
        args=args,
        run_id=run_id,
        attempt_id=attempt_id,
        seed=seed,
        reward_variant=reward_variant,
        noise=noise,
        mixture_spec=mixture_spec,
        opponent_assignment_ref=opponent_assignment_ref,
        opponent_assignment_refresh_ref=opponent_assignment_refresh_ref,
        initial_policy_checkpoint_ref=initial_policy_checkpoint_ref,
    )
    poller_kwargs = _poller_kwargs(
        args=args,
        run_id=run_id,
        attempt_id=attempt_id,
        seed=seed,
        reward_variant=reward_variant,
        mixture_spec=mixture_spec,
        opponent_assignment_ref=opponent_assignment_ref,
    )
    train_ref = _ref("training", TASK_ID, run_id, "attempts", attempt_id, "train")
    submit_command = [
        "uv",
        "run",
        "python",
        "scripts/submit_curvytron_survivaldiag_manifest.py",
        "<manifest.json>",
        "--row-id",
        row_id,
        "--allow-launch",
    ]
    return {
        "schema_id": ROW_SCHEMA_ID,
        "matrix_name": args.matrix_name,
        "row_id": row_id,
        "row_kind": "tonight18_training",
        "status": "ready_for_operator_launch_gate",
        "label": label,
        "plain_name": label,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "mode": MODE_TRAIN,
        "canonical_launcher": MODULE,
        "calls_stock_train_muzero": True,
        "env_variant": ENV_SOURCE_STATE_FIXED_OPPONENT,
        "reward_variant": reward_variant,
        "reward_tag": reward_tag,
        "opponent_recipe_id": recipe.recipe_id,
        "opponent_recipe_description": recipe.description,
        "opponent_recipe_weights": dict(recipe.weights),
        "opponent_source": (
            OPPONENT_SOURCE_ASSIGNMENT
            if opponent_assignment_ref
            else OPPONENT_SOURCE_MIXTURE
        ),
        "opponent_assignment_id": (
            assignment_artifact["assignment_id"] if assignment_artifact is not None else None
        ),
        "opponent_assignment_ref": opponent_assignment_ref,
        "opponent_assignment_refresh_ref": opponent_assignment_refresh_ref,
        "opponent_mixture_enabled": opponent_assignment_ref is None,
        "opponent_mixture_spec": mixture_spec,
        "opponent_assignment_preview": assignment_artifact["assignment"]
        if assignment_artifact is not None
        else None,
        "opponent_components": [entry["name"] for entry in preview_mixture["entries"]],
        "initial_policy_checkpoint_ref": initial_policy_checkpoint_ref,
        "initial_policy_checkpoint_source": {
            "source": "leaderboard_rank1_at_manifest_build_time",
            **top_checkpoints["rank1"],
        },
        "noise_mode": noise.mode_id,
        "noise_description": noise.description,
        "training_seed": seed,
        "eval_seed": _derived_seed("eval", seed),
        "compute": args.compute,
        "source_max_steps": SOURCE_MAX_STEPS,
        "source_state_trail_render_mode": RENDER_POLICY,
        "source_state_bonus_render_mode": BONUS_RENDER_SIMPLE_SYMBOLS,
        "learner_seat_mode": args.learner_seat_mode,
        "deployed_app_submission": {
            "app_name": APP_NAME,
            "train_function": _train_function_name(args.compute),
            "poller_function": "lightzero_curvytron_visual_survival_checkpoint_eval_poller",
            "spawn_order": ["poller", "train"],
        },
        "train_kwargs": train_kwargs,
        "poller_kwargs": poller_kwargs,
        "flags": {
            "compute": args.compute,
            "collector_env_num": train_kwargs["collector_env_num"],
            "num_simulations": train_kwargs["num_simulations"],
            "batch_size": train_kwargs["batch_size"],
            "save_ckpt_after_iter": train_kwargs["save_ckpt_after_iter"],
            "commit_on_checkpoint": train_kwargs["commit_on_checkpoint"],
            "max_train_iter": train_kwargs["max_train_iter"],
            "max_env_step": train_kwargs["max_env_step"],
            "source_state_trail_render_mode": train_kwargs["source_state_trail_render_mode"],
            "source_state_bonus_render_mode": train_kwargs["source_state_bonus_render_mode"],
            "learner_seat_mode": train_kwargs["learner_seat_mode"],
            "ego_action_straight_override_probability": train_kwargs[
                "ego_action_straight_override_probability"
            ],
            "policy_action_repeat_extra_probability": train_kwargs[
                "policy_action_repeat_extra_probability"
            ],
            "control_noise_profile_id": train_kwargs["control_noise_profile_id"],
        },
        "artifact_refs": {
            "run_manifest": _ref("training", TASK_ID, run_id, "run.json"),
            "attempt_manifest": _ref(
                "training", TASK_ID, run_id, "attempts", attempt_id, "attempt.json"
            ),
            "latest_attempt": _ref("training", TASK_ID, run_id, "latest_attempt.json"),
            "summary": _ref(train_ref, "summary.json"),
            "progress_latest": _ref(train_ref, "progress_latest.json"),
            "checkpoint_root": _ref("training", TASK_ID, run_id, "checkpoints", "lightzero"),
            "background_eval_status": _ref(train_ref, "checkpoint_eval_poller.json"),
            "background_gif_status": _ref(train_ref, "background_gif_jobs.json"),
        },
        "command": submit_command,
        "command_text": shlex.join(submit_command),
        "review_command_only": True,
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    top_checkpoints = _load_top_checkpoints(args.ratings_snapshot)
    assignment_artifacts_by_recipe: dict[str, dict[str, Any]] = {}
    if args.opponent_source == OPPONENT_SOURCE_ASSIGNMENT:
        assignment_artifacts_by_recipe = {
            recipe.recipe_id: _assignment_artifact(
                args=args,
                recipe=recipe,
                top_checkpoints=top_checkpoints,
            )
            for recipe in OPPONENT_RECIPES
        }
    rows: list[dict[str, Any]] = []
    row_number = 0
    for reward_variant in REWARD_VARIANTS:
        for recipe in OPPONENT_RECIPES:
            for noise in NOISE_MODES:
                row_number += 1
                rows.append(
                    _row(
                        args=args,
                        row_number=row_number,
                        reward_variant=reward_variant,
                        recipe=recipe,
                        noise=noise,
                        top_checkpoints=top_checkpoints,
                        assignment_artifact=assignment_artifacts_by_recipe.get(
                            recipe.recipe_id
                        ),
                    )
                )
    assignment_bank = None
    if assignment_artifacts_by_recipe:
        assignment_bank = {
            "run_id": _assignment_bank_run_id(args),
            "attempt_id": _assignment_bank_attempt_id(args),
            "source_ref": _assignment_source_ref(args),
            "target_volume": args.assignment_target_volume,
            "assignments": assignment_artifacts_by_recipe,
        }
        if args.assignment_refresh_interval_train_iter > 0:
            assignment_bank["refresh_pointer_volume"] = (
                args.assignment_refresh_pointer_volume
            )
            assignment_bank["refresh_pointer_run_id"] = (
                _assignment_refresh_pointer_run_id(args)
            )
            assignment_bank["refresh_pointer_attempt_id"] = (
                _assignment_refresh_pointer_attempt_id(args)
            )
            refresh_pointers: dict[str, dict[str, Any]] = {}
            for recipe in OPPONENT_RECIPES:
                artifact = assignment_artifacts_by_recipe[recipe.recipe_id]
                refresh_pointers[recipe.recipe_id] = {
                    "schema_id": "curvyzero_opponent_assignment_refresh_pointer/v0",
                    "pointer_ref": _assignment_refresh_pointer_ref(
                        args=args,
                        recipe=recipe,
                    ),
                    "pointer_volume": args.assignment_refresh_pointer_volume,
                    "assignment_ref": artifact["assignment_ref"],
                    "assignment_sha256": artifact["assignment_sha256"],
                    "recipe_id": recipe.recipe_id,
                    "audit": {
                        "schema_id": "curvyzero_opponent_assignment_refresh_pointer_audit/v0",
                        "reason": "initial tonight18 per-recipe refresh pointer",
                        "assignment_id": artifact["assignment_id"],
                        "assignment_sha256": artifact["assignment_sha256"],
                        "assignment_ref": artifact["assignment_ref"],
                        "matrix_name": args.matrix_name,
                        "recipe_id": recipe.recipe_id,
                    },
                }
            assignment_bank["refresh_pointers"] = refresh_pointers

    manifest = {
        "schema_id": SCHEMA_ID,
        "status": "ready_for_operator_launch_gate",
        "matrix_name": args.matrix_name,
        "matrix_profile": "tonight18_reward_opponent_noise",
        "generated_at": datetime.now(UTC).isoformat(),
        "ratings_snapshot_path": str(args.ratings_snapshot),
        "opponent_source": args.opponent_source,
        "top_checkpoint_source": top_checkpoints,
        "assignment_bank": assignment_bank,
        "axes": {
            "reward_variants": list(REWARD_VARIANTS),
            "opponent_recipes": [
                {
                    "recipe_id": recipe.recipe_id,
                    "description": recipe.description,
                    "weights": dict(recipe.weights),
                }
                for recipe in OPPONENT_RECIPES
            ],
            "noise_modes": [
                {
                    "mode_id": noise.mode_id,
                    "description": noise.description,
                    "ego_action_straight_override_probability": (
                        noise.ego_action_straight_override_probability
                    ),
                    "policy_action_repeat_min": noise.policy_action_repeat_min,
                    "policy_action_repeat_max": noise.policy_action_repeat_max,
                    "policy_action_repeat_extra_probability": (
                        noise.policy_action_repeat_extra_probability
                    ),
                }
                for noise in NOISE_MODES
            ],
        },
        "fixed_knobs": {
            "compute": args.compute,
            "collector_env_num": 256,
            "num_simulations": 8,
            "batch_size": 32,
            "source_state_trail_render_mode": RENDER_POLICY,
            "source_state_bonus_render_mode": BONUS_RENDER_SIMPLE_SYMBOLS,
            "learner_seat_mode": args.learner_seat_mode,
            "save_ckpt_after_iter": SAVE_CKPT_AFTER_ITER,
            "commit_on_checkpoint": COMMIT_ON_CHECKPOINT,
            "source_max_steps": SOURCE_MAX_STEPS,
            "max_train_iter": args.max_train_iter,
            "max_env_step": args.max_env_step,
            "assignment_refresh_interval_train_iter": (
                args.assignment_refresh_interval_train_iter
            ),
            "assignment_target_volume": args.assignment_target_volume,
            "assignment_refresh_pointer_volume": args.assignment_refresh_pointer_volume,
            "initial_policy_checkpoint_source": "rank1_checkpoint_from_ratings_snapshot",
            "initial_policy_checkpoint_load_mode": "matching_shape",
        },
        "guards": {
            "deployed_app_name": APP_NAME,
            "launch_script": "scripts/submit_curvytron_survivaldiag_manifest.py",
            "operator_launch_gate_required": True,
            "expected_row_count": 18,
            "modal_launch_performed": False,
        },
        "rows": rows,
    }
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: dict[str, Any]) -> None:
    rows = manifest["rows"]
    if len(rows) != 18:
        raise ValueError(f"expected 18 rows, got {len(rows)}")
    run_ids = [row["run_id"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate run_id values in manifest")
    too_long_ids = [
        (row["row_id"], "run_id", row["run_id"])
        for row in rows
        if len(row["run_id"]) > MAX_MODAL_RUN_ID_LEN
    ] + [
        (row["row_id"], "attempt_id", row["attempt_id"])
        for row in rows
        if len(row["attempt_id"]) > MAX_MODAL_RUN_ID_LEN
    ]
    if too_long_ids:
        raise ValueError(f"manifest ids exceed {MAX_MODAL_RUN_ID_LEN} chars: {too_long_ids}")
    observed_axes = {
        (row["reward_variant"], row["opponent_recipe_id"], row["noise_mode"])
        for row in rows
    }
    expected_axes = {
        (reward, recipe.recipe_id, noise.mode_id)
        for reward in REWARD_VARIANTS
        for recipe in OPPONENT_RECIPES
        for noise in NOISE_MODES
    }
    if observed_axes != expected_axes:
        raise ValueError("manifest rows do not cover exactly the requested 3x3x2 axes")
    initial_refs = {
        str(row["train_kwargs"].get("initial_policy_checkpoint_ref") or "") for row in rows
    }
    if len(initial_refs) != 1 or not next(iter(initial_refs)):
        raise ValueError("all rows must share one non-empty initial policy checkpoint ref")
    for row in rows:
        train_kwargs = row["train_kwargs"]
        poller_kwargs = row["poller_kwargs"]
        missing = [
            key for key in TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT if key not in train_kwargs
        ]
        if missing:
            raise ValueError(f"row {row['row_id']} train_kwargs missing {missing}")
        if train_kwargs["opponent_mixture_spec"] != poller_kwargs["opponent_mixture_spec"]:
            raise ValueError(f"row {row['row_id']} train/poller mixture specs differ")
        if train_kwargs.get("opponent_assignment_ref") != poller_kwargs.get(
            "opponent_assignment_ref"
        ):
            raise ValueError(f"row {row['row_id']} train/poller assignment refs differ")
        if train_kwargs.get("opponent_assignment_ref") and train_kwargs.get(
            "opponent_mixture_spec"
        ) is not None:
            raise ValueError(f"row {row['row_id']} combines assignment ref and mixture")
        initial_ref = str(train_kwargs.get("initial_policy_checkpoint_ref") or "")
        if not initial_ref:
            raise ValueError(f"row {row['row_id']} lacks initial policy checkpoint ref")
        if "latest" in initial_ref or "ckpt_best" in initial_ref:
            raise ValueError(f"row {row['row_id']} initial checkpoint ref is mutable")
        if not re.fullmatch(r"iteration_\d+\.pth\.tar", Path(initial_ref).name):
            raise ValueError(
                f"row {row['row_id']} initial checkpoint must use iteration_N.pth.tar"
            )
        if train_kwargs.get("initial_policy_checkpoint_load_mode") != "matching_shape":
            raise ValueError(f"row {row['row_id']} initial checkpoint load mode mismatch")
        preview_assignment = row.get("opponent_assignment_preview")
        if train_kwargs.get("opponent_assignment_ref"):
            if not preview_assignment:
                raise ValueError(f"row {row['row_id']} assignment row lacks preview")
            parsed_assignment = parse_opponent_assignment_snapshot(preview_assignment)
            mixture = parsed_assignment["opponent_mixture"]
        else:
            mixture = parse_opponent_mixture_spec(train_kwargs["opponent_mixture_spec"])
        if mixture is None:
            raise ValueError(f"row {row['row_id']} has an empty opponent source")
        for entry in mixture["entries"]:
            if entry["opponent_policy_kind"] != OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT:
                continue
            if bool(entry.get("opponent_immortal")):
                raise ValueError(
                    f"row {row['row_id']} frozen leaderboard entries must not be immortal"
                )
            checkpoint_ref = str(entry.get("opponent_checkpoint_ref") or "")
            if "latest" in checkpoint_ref or "ckpt_best" in checkpoint_ref:
                raise ValueError(f"row {row['row_id']} contains mutable checkpoint ref")
            if not re.fullmatch(r"iteration_\d+\.pth\.tar", Path(checkpoint_ref).name):
                raise ValueError(
                    f"row {row['row_id']} frozen entry must use iteration_N.pth.tar"
                )
        wall_entries = [
            entry for entry in mixture["entries"] if entry["name"] == "wall_avoidant_immortal"
        ]
        if len(wall_entries) != 1:
            raise ValueError(
                f"row {row['row_id']} must contain one wall_avoidant_immortal sentinel"
            )
        wall_entry = wall_entries[0]
        if (
            wall_entry["opponent_policy_kind"] != OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT
            or wall_entry["opponent_runtime_mode"] != OPPONENT_RUNTIME_MODE_NORMAL
            or not bool(wall_entry.get("opponent_immortal"))
            or wall_entry["opponent_wall_avoidant_safe_margin"] != 20.0
        ):
            raise ValueError(
                f"row {row['row_id']} wall sentinel must be proactive/normal with opponent_immortal=true"
            )
        pressure = sum(
            float(entry["weight"])
            for entry in mixture["entries"]
            if entry["name"] in {"blank", "wall_avoidant_immortal"}
        )
        if pressure < 20.0:
            raise ValueError(f"row {row['row_id']} blank/immortal pressure below 20%")
        if train_kwargs["collector_env_num"] != 256:
            raise ValueError(f"row {row['row_id']} collector_env_num mismatch")
        if train_kwargs["num_simulations"] != 8 or train_kwargs["batch_size"] != 32:
            raise ValueError(f"row {row['row_id']} LightZero size mismatch")
        if train_kwargs["source_state_trail_render_mode"] != RENDER_POLICY:
            raise ValueError(f"row {row['row_id']} trail render mismatch")
        if train_kwargs["source_state_bonus_render_mode"] != BONUS_RENDER_SIMPLE_SYMBOLS:
            raise ValueError(f"row {row['row_id']} bonus render mismatch")
        if row.get("learner_seat_mode") != train_kwargs.get("learner_seat_mode"):
            raise ValueError(f"row {row['row_id']} learner seat mode row/train mismatch")
        if train_kwargs["learner_seat_mode"] not in LEARNER_SEAT_MODE_CHOICES:
            raise ValueError(f"row {row['row_id']} learner seat mode mismatch")
        if "learner_seat_mode" in poller_kwargs:
            raise ValueError(f"row {row['row_id']} poller must not inherit learner seat mode")
        if train_kwargs["save_ckpt_after_iter"] != SAVE_CKPT_AFTER_ITER:
            raise ValueError(f"row {row['row_id']} checkpoint cadence mismatch")
        if train_kwargs["commit_on_checkpoint"] is not COMMIT_ON_CHECKPOINT:
            raise ValueError(f"row {row['row_id']} checkpoint commit flag mismatch")
        refresh_interval = int(
            manifest["fixed_knobs"].get("assignment_refresh_interval_train_iter") or 0
        )
        refresh_ref = str(train_kwargs.get("opponent_assignment_refresh_ref") or "")
        if refresh_interval > 0:
            if not refresh_ref:
                raise ValueError(f"row {row['row_id']} lacks refresh pointer ref")
            if refresh_ref == str(train_kwargs.get("opponent_assignment_ref") or ""):
                raise ValueError(
                    f"row {row['row_id']} refresh ref must be mutable pointer, not assignment"
                )
            expected_prefix = (
                "control:"
                if manifest["fixed_knobs"]["assignment_refresh_pointer_volume"] == "control"
                else "runs:"
            )
            if not refresh_ref.startswith(expected_prefix):
                raise ValueError(
                    f"row {row['row_id']} refresh ref must start with {expected_prefix!r}"
                )


def _write_outputs(manifest: dict[str, Any], *, output_root: Path, matrix_name: str) -> dict[str, str]:
    output_dir = output_root / matrix_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{matrix_name}.json"
    rows_path = output_dir / f"{matrix_name}.rows.jsonl"
    commands_path = output_dir / f"{matrix_name}.submit.commands.txt"
    assignments_dir = output_dir / "assignments"
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in manifest["rows"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    submit_command = [
        "uv",
        "run",
        "python",
        "scripts/submit_curvytron_survivaldiag_manifest.py",
        str(manifest_path),
    ]
    launch_command = [*submit_command, "--allow-launch"]
    commands_path.write_text(
        "# Dry-run validation command\n"
        f"{shlex.join(submit_command)}\n\n"
        "# Launch command; do not run until reward support and operator gate are clear\n"
        f"{shlex.join(launch_command)}\n",
        encoding="utf-8",
    )
    outputs = {
        "manifest_json": str(manifest_path),
        "rows_jsonl": str(rows_path),
        "submit_commands_txt": str(commands_path),
    }
    assignment_bank = manifest.get("assignment_bank")
    if isinstance(assignment_bank, dict) and assignment_bank.get("assignments"):
        assignments_dir.mkdir(parents=True, exist_ok=True)
        index: dict[str, Any] = {}
        for recipe_id, artifact in assignment_bank["assignments"].items():
            recipe_dir = assignments_dir / str(recipe_id)
            recipe_dir.mkdir(parents=True, exist_ok=True)
            assignment_path = recipe_dir / "assignment.json"
            audit_path = recipe_dir / "audit.json"
            assignment_path.write_text(
                json.dumps(artifact["assignment"], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            audit_path.write_text(
                json.dumps(artifact["audit"], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            index[recipe_id] = {
                "assignment_json": str(assignment_path),
                "audit_json": str(audit_path),
                "assignment_ref": artifact["assignment_ref"],
            }
            refresh_pointers = assignment_bank.get("refresh_pointers")
            if isinstance(refresh_pointers, dict) and recipe_id in refresh_pointers:
                pointer_payload = copy.deepcopy(refresh_pointers[recipe_id])
                pointer_payload.pop("pointer_ref", None)
                pointer_payload.pop("pointer_volume", None)
                pointer_payload.pop("recipe_id", None)
                pointer_path = recipe_dir / "refresh_pointer.json"
                pointer_path.write_text(
                    json.dumps(pointer_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                index[recipe_id]["refresh_pointer_json"] = str(pointer_path)
                index[recipe_id]["refresh_pointer_ref"] = refresh_pointers[recipe_id][
                    "pointer_ref"
                ]
                index[recipe_id]["refresh_pointer_volume"] = refresh_pointers[recipe_id][
                    "pointer_volume"
                ]
        index_path = assignments_dir / "index.json"
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        outputs["assignments_index_json"] = str(index_path)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ratings-snapshot",
        type=Path,
        required=True,
        help=(
            "Leaderboard/rating snapshot to freeze into initial assignments. "
            "This is required so a real launch cannot silently use a stale snapshot."
        ),
    )
    parser.add_argument(
        "--opponent-source",
        choices=(OPPONENT_SOURCE_MIXTURE, OPPONENT_SOURCE_ASSIGNMENT),
        default=DEFAULT_OPPONENT_SOURCE,
    )
    parser.add_argument("--assignment-bank-run-id", default="")
    parser.add_argument("--assignment-bank-attempt-id", default="")
    parser.add_argument(
        "--assignment-target-volume",
        choices=("runs", "control"),
        default=DEFAULT_ASSIGNMENT_TARGET_VOLUME,
    )
    parser.add_argument("--assignment-refresh-pointer-run-id", default="")
    parser.add_argument("--assignment-refresh-pointer-attempt-id", default="")
    parser.add_argument(
        "--assignment-refresh-pointer-volume",
        choices=("runs", "control"),
        default=DEFAULT_ASSIGNMENT_REFRESH_POINTER_VOLUME,
    )
    parser.add_argument("--assignment-source-ref", default="")
    parser.add_argument("--leaderboard-id", default="")
    parser.add_argument("--leaderboard-snapshot-id", default="")
    parser.add_argument("--leaderboard-snapshot-sha256", default="")
    parser.add_argument(
        "--assignment-refresh-interval-train-iter",
        type=int,
        default=ASSIGNMENT_REFRESH_INTERVAL_TRAIN_ITER,
    )
    parser.add_argument("--matrix-name", default=DEFAULT_MATRIX_NAME)
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--attempt-prefix", default=DEFAULT_ATTEMPT_PREFIX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--compute", choices=(COMPUTE_H100_CPU40,), default=COMPUTE_H100_CPU40)
    parser.add_argument("--max-train-iter", type=int, default=300_000)
    parser.add_argument("--max-env-step", type=int, default=30_000_000)
    parser.add_argument("--env-manager-type", default="subprocess")
    parser.add_argument(
        "--learner-seat-mode",
        choices=LEARNER_SEAT_MODE_CHOICES,
        default=DEFAULT_LEARNER_SEAT_MODE,
        help=(
            "Learner seat assignment for training rows. Fresh real manifests "
            "default to random_per_episode."
        ),
    )
    parser.add_argument("--background-eval-seed-count", type=int, default=8)
    parser.add_argument("--background-eval-poll-interval-sec", type=float, default=10.0)
    parser.add_argument(
        "--background-eval-poller-max-runtime-sec",
        type=float,
        default=18 * 60 * 60,
    )
    parser.add_argument("--stop-after-learner-train-calls", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_manifest(args)
    outputs = _write_outputs(
        manifest,
        output_root=args.output_root,
        matrix_name=args.matrix_name,
    )
    print(json.dumps({"row_count": len(manifest["rows"]), **outputs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
