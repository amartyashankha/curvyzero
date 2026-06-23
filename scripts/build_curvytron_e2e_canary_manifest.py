#!/usr/bin/env python3
"""Build a tiny CurvyTron deployed E2E canary manifest.

The canary is intentionally one row:

- writes one clean control-volume starter assignment;
- writes one mutable control-volume refresh pointer;
- launches one trainer from an exact checkpoint through the grouped submitter;
- checkpoints frequently enough for tournament intake to see new files quickly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from curvyzero.contracts.curvytron import (
    CURVYTRON_BACKGROUND_GIF_FPS,
    CURVYTRON_DECISION_MS,
    CURVYTRON_POLICY_BONUS_RENDER_MODE,
    CURVYTRON_POLICY_TRAIL_RENDER_MODE,
    CURVYTRON_SOURCE_MAX_STEPS,
    CURVYTRON_TRAINING_TASK_ID,
    DEFAULT_CURVYTRON_TRAIN_APP_NAME,
    DEFAULT_LEARNER_SEAT_MODE,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
)
from curvyzero.contracts.curvytron_naming import (
    CURVYTRON_CANARY_BATCH,
    action_noise_tag,
    curvytron_attempt_id,
    curvytron_run_id,
    leaderboard_immortal_tag,
    reward_alpha_tag,
)
from curvyzero.training.opponent_mixture import (
    OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
    OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
    OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
    OPPONENT_RUNTIME_MODE_NORMAL,
)
from curvyzero.training.opponent_registry import (
    OPPONENT_ASSIGNMENT_SCHEMA_ID,
    canonical_assignment_json_sha256,
    parse_opponent_assignment_snapshot,
)


SCHEMA_ID = "curvyzero_curvytron_e2e_canary_manifest/v0"
ROW_SCHEMA_ID = "curvyzero_curvytron_e2e_canary_row/v0"
MAX_MODAL_RUN_ID_LEN = 96

COMPUTE_CPU = "cpu"
COMPUTE_H100_CPU40 = "gpu-h100-cpu40"
COMPUTE_CHOICES = (COMPUTE_CPU, COMPUTE_H100_CPU40)

TRAIN_FUNCTION_BY_COMPUTE = {
    COMPUTE_CPU: "lightzero_curvytron_visual_survival_cpu",
    COMPUTE_H100_CPU40: "lightzero_curvytron_visual_survival_h100_cpu40",
}
DEFAULT_RECIPE_CODE = "b50r1"
DEFAULT_CANARY_RUN_ID = curvytron_run_id(
    batch=CURVYTRON_CANARY_BATCH,
    row_number=1,
    reward_tag=reward_alpha_tag(1.0),
    noise_tag=action_noise_tag(0.0),
    immortal_tag=leaderboard_immortal_tag(0.0),
    recipe_code=DEFAULT_RECIPE_CODE,
)


def _safe_id(value: str, *, label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-")
    safe = "-".join(part for part in safe.split("-") if part)
    if not safe:
        raise ValueError(f"{label} became empty")
    if len(safe) > MAX_MODAL_RUN_ID_LEN:
        digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:10]
        safe = f"{safe[: MAX_MODAL_RUN_ID_LEN - len(digest) - 1].rstrip('-_.')}-{digest}"
    return safe


def _ref(*parts: str | Path) -> str:
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def _assignment_ref(*, run_id: str, attempt_id: str, assignment_id: str) -> str:
    return _ref(
        "control:training",
        CURVYTRON_TRAINING_TASK_ID,
        run_id,
        "attempts",
        attempt_id,
        "opponents",
        "assignments",
        assignment_id,
        "assignment.json",
    )


def _pointer_ref(*, run_id: str, attempt_id: str) -> str:
    return _ref(
        "control:training",
        CURVYTRON_TRAINING_TASK_ID,
        run_id,
        "attempts",
        attempt_id,
        "opponents",
        "current_assignment_pointer.json",
    )


def _assignment(
    *,
    assignment_id: str,
    seed: int,
    starter_checkpoint_ref: str,
) -> dict[str, Any]:
    payload = {
        "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
        "assignment_id": assignment_id,
        "source_epoch": 0,
        "source_ref": "operator_seed:rank_slot_canary",
        "seed": seed,
        "entries": [
            {
                "name": "canary_blank_canvas",
                "weight": 1,
                "age_label": "blank_canvas",
                "tags": ["canary", "blank", "immortal"],
                "opponent_policy_kind": OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
                "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
                "opponent_immortal": True,
            },
            {
                "name": "rank1",
                "weight": 1,
                "age_label": "rank1",
                "tags": {"canary": True, "rank": 1, "source_slot": "rank1"},
                "opponent_policy_kind": OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
                "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_NORMAL,
                "opponent_immortal": False,
                "opponent_checkpoint_ref": starter_checkpoint_ref,
            },
        ],
    }
    parse_opponent_assignment_snapshot(payload)
    return payload


def _train_kwargs(
    args: argparse.Namespace, *, assignment_ref: str, pointer_ref: str
) -> dict[str, Any]:
    return {
        "mode": "train",
        "seed": args.seed,
        "run_id": args.run_id,
        "attempt_id": args.attempt_id,
        "max_env_step": args.max_env_step,
        "max_train_iter": args.max_train_iter,
        "source_max_steps": CURVYTRON_SOURCE_MAX_STEPS,
        "decision_ms": CURVYTRON_DECISION_MS,
        "collector_env_num": args.collector_env_num,
        "evaluator_env_num": 1,
        "n_evaluator_episode": 1,
        "n_episode": args.collector_env_num,
        "num_simulations": args.num_simulations,
        "batch_size": args.batch_size,
        "lightzero_eval_freq": 0,
        "skip_lightzero_eval_in_profile": True,
        "profile_cuda_sync_enabled": False,
        "profile_allow_auto_resume": False,
        "profile_volume_commit": False,
        "lightzero_multi_gpu": False,
        "save_ckpt_after_iter": args.save_ckpt_after_iter,
        "commit_on_checkpoint": True,
        "stop_after_learner_train_calls": args.stop_after_learner_train_calls,
        "env_variant": "source_state_fixed_opponent",
        "reward_variant": args.reward_variant,
        "reward_outcome_alpha": args.reward_outcome_alpha,
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
        "opponent_runtime_mode": "normal",
        "env_telemetry_stride": 1,
        "env_manager_type": args.env_manager_type,
        "opponent_policy_kind": OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        "opponent_use_cuda": False,
        "opponent_checkpoint_ref": None,
        "opponent_snapshot_ref": None,
        "opponent_checkpoint_report_ref": None,
        "opponent_checkpoint_state_key": None,
        "initial_policy_checkpoint_ref": args.initial_policy_checkpoint_ref,
        "initial_policy_checkpoint_state_key": None,
        "initial_policy_checkpoint_load_mode": "matching_shape",
        "opponent_mixture_spec": None,
        "opponent_assignment_ref": assignment_ref,
        "opponent_assignment_refresh_interval_train_iter": args.refresh_interval_train_iter,
        "opponent_assignment_refresh_ref": pointer_ref,
        "background_eval_enabled": False,
        "background_eval_launch_kind": "poller",
        "background_eval_compute": "cpu",
        "background_eval_id_prefix": "e2e_canary_checkpoint",
        "background_eval_seed_count": 1,
        "background_eval_seed_rng_seed": args.seed + 1000,
        "background_eval_max_steps": CURVYTRON_SOURCE_MAX_STEPS,
        "background_eval_step_detail_limit": 4,
        "background_eval_num_simulations": args.num_simulations,
        "background_eval_batch_size": 8,
        "background_gif_enabled": False,
        "background_gif_seed_offset": 10_000,
        "background_gif_max_steps": 4096,
        "background_gif_frame_stride": 4,
        "background_gif_fps": CURVYTRON_BACKGROUND_GIF_FPS,
        "background_gif_scale": 4,
        "background_gif_frame_size": 128,
        "background_gif_collect_temperature": 1.0,
        "background_gif_collect_epsilon": 0.25,
    }


def _poller_kwargs(args: argparse.Namespace, *, assignment_ref: str) -> dict[str, Any]:
    exp_name_ref = _ref(
        "training",
        CURVYTRON_TRAINING_TASK_ID,
        args.run_id,
        "attempts",
        args.attempt_id,
        "train",
        "lightzero_exp",
    )
    return {
        "run_id": args.run_id,
        "attempt_id": args.attempt_id,
        "exp_name_ref": exp_name_ref,
        "seed": args.seed,
        "source_max_steps": CURVYTRON_SOURCE_MAX_STEPS,
        "env_variant": "source_state_fixed_opponent",
        "reward_variant": args.reward_variant,
        "reward_outcome_alpha": args.reward_outcome_alpha,
        "opponent_policy_kind": OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        "opponent_checkpoint_ref": None,
        "opponent_snapshot_ref": None,
        "opponent_checkpoint_state_key": None,
        "opponent_mixture_spec": None,
        "opponent_assignment_ref": assignment_ref,
        "opponent_death_mode": "normal",
        "opponent_runtime_mode": "normal",
        "background_eval_compute": "cpu",
        "background_eval_id_prefix": "e2e_canary_checkpoint",
        "background_eval_seed_count": 1,
        "background_eval_seed_rng_seed": args.seed + 1000,
        "background_eval_max_steps": CURVYTRON_SOURCE_MAX_STEPS,
        "background_eval_step_detail_limit": 4,
        "background_eval_num_simulations": args.num_simulations,
        "background_eval_batch_size": 8,
        "background_gif_enabled": False,
        "background_gif_seed_offset": 10_000,
        "background_gif_max_steps": 4096,
        "background_gif_frame_stride": 4,
        "background_gif_fps": CURVYTRON_BACKGROUND_GIF_FPS,
        "background_gif_scale": 4,
        "background_gif_frame_size": 128,
        "background_gif_collect_temperature": 1.0,
        "background_gif_collect_epsilon": 0.25,
        "poll_interval_sec": 30.0,
        "stable_polls": 1,
        "max_runtime_sec": 1800.0,
        "idle_after_train_done_sec": 60.0,
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    args.run_id = _safe_id(args.run_id, label="run_id")
    args.attempt_id = _safe_id(args.attempt_id, label="attempt_id")
    assignment_bank_run_id = _safe_id(args.assignment_bank_run_id, label="assignment_bank_run_id")
    assignment_bank_attempt_id = _safe_id(
        args.assignment_bank_attempt_id,
        label="assignment_bank_attempt_id",
    )
    pointer_run_id = _safe_id(args.pointer_run_id, label="pointer_run_id")
    pointer_attempt_id = _safe_id(args.pointer_attempt_id, label="pointer_attempt_id")
    assignment_id = _safe_id(args.assignment_id, label="assignment_id")
    assignment = _assignment(
        assignment_id=assignment_id,
        seed=args.seed,
        starter_checkpoint_ref=args.initial_policy_checkpoint_ref,
    )
    assignment_sha256 = canonical_assignment_json_sha256(assignment)
    assignment_ref = _assignment_ref(
        run_id=assignment_bank_run_id,
        attempt_id=assignment_bank_attempt_id,
        assignment_id=assignment_id,
    )
    pointer_ref = _pointer_ref(run_id=pointer_run_id, attempt_id=pointer_attempt_id)
    audit = {
        "schema_id": "curvyzero_opponent_assignment_audit/v0",
        "assignment_id": assignment_id,
        "assignment_sha256": assignment_sha256,
        "selection": {
            "strategy_id": "e2e_canary_rank_slot_v0",
            "reason": "starter assignment with a replaceable rank1 slot for deployed full-loop proof",
            "seed": args.seed,
        },
    }
    row = {
        "schema_id": ROW_SCHEMA_ID,
        "row_id": "r001",
        "row_kind": "e2e_canary_training",
        "status": "ready_for_operator_launch_gate",
        "label": args.run_id,
        "plain_name": "cz26c canary: blank plus rank1",
        "run_id": args.run_id,
        "attempt_id": args.attempt_id,
        "mode": "train",
        "compute": args.compute,
        "canonical_launcher": "scripts/submit_curvytron_survivaldiag_manifest.py",
        "calls_stock_train_muzero": True,
        "env_variant": "source_state_fixed_opponent",
        "reward_variant": args.reward_variant,
        "reward_outcome_alpha": args.reward_outcome_alpha,
        "opponent_source": "assignment",
        "opponent_assignment_id": assignment_id,
        "opponent_assignment_ref": assignment_ref,
        "opponent_assignment_refresh_ref": pointer_ref,
        "initial_policy_checkpoint_ref": args.initial_policy_checkpoint_ref,
        "source_state_trail_render_mode": CURVYTRON_POLICY_TRAIL_RENDER_MODE,
        "source_state_bonus_render_mode": CURVYTRON_POLICY_BONUS_RENDER_MODE,
        "learner_seat_mode": args.learner_seat_mode,
        "deployed_app_submission": {
            "app_name": DEFAULT_CURVYTRON_TRAIN_APP_NAME,
            "train_function": TRAIN_FUNCTION_BY_COMPUTE[args.compute],
            "poller_function": "lightzero_curvytron_visual_survival_checkpoint_eval_poller",
            "spawn_order": ["poller", "train"],
        },
        "train_kwargs": _train_kwargs(
            args,
            assignment_ref=assignment_ref,
            pointer_ref=pointer_ref,
        ),
        "poller_kwargs": _poller_kwargs(args, assignment_ref=assignment_ref),
        "artifact_refs": {
            "summary": _ref(
                "training",
                CURVYTRON_TRAINING_TASK_ID,
                args.run_id,
                "attempts",
                args.attempt_id,
                "train",
                "summary.json",
            ),
            "progress_latest": _ref(
                "training",
                CURVYTRON_TRAINING_TASK_ID,
                args.run_id,
                "attempts",
                args.attempt_id,
                "train",
                "progress_latest.json",
            ),
            "checkpoint_root": _ref(
                "training",
                CURVYTRON_TRAINING_TASK_ID,
                args.run_id,
                "attempts",
                args.attempt_id,
                "train",
                "lightzero_exp",
                "ckpt",
            ),
            "refresh_events": _ref(
                "training",
                CURVYTRON_TRAINING_TASK_ID,
                args.run_id,
                "attempts",
                args.attempt_id,
                "train",
                "opponent_assignment_refresh_events.jsonl",
            ),
            "env_steps": _ref(
                "training",
                CURVYTRON_TRAINING_TASK_ID,
                args.run_id,
                "attempts",
                args.attempt_id,
                "train",
                "env_steps.jsonl",
            ),
        },
    }
    return {
        "schema_id": SCHEMA_ID,
        "status": "ready_for_operator_launch_gate",
        "matrix_name": args.matrix_name,
        "matrix_profile": "single_row_deployed_full_loop_canary",
        "generated_at": datetime.now(UTC).isoformat(),
        "assignment_bank": {
            "run_id": assignment_bank_run_id,
            "attempt_id": assignment_bank_attempt_id,
            "source_ref": "operator_seed:rank_slot_canary",
            "target_volume": "control",
            "assignments": {
                DEFAULT_RECIPE_CODE: {
                    "assignment_id": assignment_id,
                    "assignment_ref": assignment_ref,
                    "assignment_sha256": assignment_sha256,
                    "recipe_id": DEFAULT_RECIPE_CODE,
                    "assignment": assignment,
                    "audit": audit,
                }
            },
            "refresh_pointer_volume": "control",
            "refresh_pointer_run_id": pointer_run_id,
            "refresh_pointer_attempt_id": pointer_attempt_id,
            "refresh_pointers": {
                DEFAULT_RECIPE_CODE: {
                    "schema_id": "curvyzero_opponent_assignment_refresh_pointer/v0",
                    "pointer_ref": pointer_ref,
                    "pointer_volume": "control",
                    "assignment_ref": assignment_ref,
                    "assignment_sha256": assignment_sha256,
                    "recipe_id": DEFAULT_RECIPE_CODE,
                    "audit": {
                        "schema_id": "curvyzero_opponent_assignment_refresh_pointer_audit/v0",
                        "reason": "initial E2E canary pointer",
                        "assignment_id": assignment_id,
                        "assignment_ref": assignment_ref,
                        "assignment_sha256": assignment_sha256,
                    },
                }
            },
        },
        "guards": {
            "deployed_app_name": DEFAULT_CURVYTRON_TRAIN_APP_NAME,
            "launch_script": "scripts/submit_curvytron_survivaldiag_manifest.py",
            "operator_launch_gate_required": True,
            "expected_row_count": 1,
            "modal_launch_performed": False,
            "requires_fresh_tournament_ids": True,
        },
        "rows": [row],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix-name", default=CURVYTRON_CANARY_BATCH)
    parser.add_argument("--run-id", default=DEFAULT_CANARY_RUN_ID)
    parser.add_argument("--attempt-id", default=curvytron_attempt_id(DEFAULT_CANARY_RUN_ID))
    parser.add_argument(
        "--assignment-id",
        default=f"{CURVYTRON_CANARY_BATCH}-r001-{DEFAULT_RECIPE_CODE}-initial",
    )
    parser.add_argument(
        "--assignment-bank-run-id",
        default=f"{CURVYTRON_CANARY_BATCH}-assignments",
    )
    parser.add_argument(
        "--assignment-bank-attempt-id",
        default=f"try-{CURVYTRON_CANARY_BATCH}-assignments",
    )
    parser.add_argument("--pointer-run-id", default=f"{CURVYTRON_CANARY_BATCH}-control")
    parser.add_argument(
        "--pointer-attempt-id",
        default=f"try-{CURVYTRON_CANARY_BATCH}-control",
    )
    parser.add_argument("--initial-policy-checkpoint-ref", required=True)
    parser.add_argument("--compute", choices=COMPUTE_CHOICES, default=COMPUTE_H100_CPU40)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--max-train-iter", type=int, default=4000)
    parser.add_argument("--max-env-step", type=int, default=200000)
    parser.add_argument("--collector-env-num", type=int, default=8)
    parser.add_argument("--num-simulations", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-ckpt-after-iter", type=int, default=50)
    parser.add_argument("--refresh-interval-train-iter", type=int, default=25)
    parser.add_argument("--stop-after-learner-train-calls", type=int, default=0)
    parser.add_argument("--env-manager-type", choices=("base", "subprocess"), default="base")
    parser.add_argument("--learner-seat-mode", default=DEFAULT_LEARNER_SEAT_MODE)
    parser.add_argument(
        "--reward-variant",
        default=REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
    )
    parser.add_argument("--reward-outcome-alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"manifest": args.output.as_posix(), "row_count": len(manifest["rows"])}, indent=2
        )
    )


if __name__ == "__main__":
    main()
