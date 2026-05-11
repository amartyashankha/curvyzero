"""Build a staged frozen-opponent refresh plan for CurvyTron LightZero.

This is a planning helper, not a trainer.  Each stage trains one ego policy
against one frozen LightZero opponent checkpoint.  The next stage can point at
the previous stage's mirrored checkpoint.  That is useful prep for self-play,
but it is not current-policy self-play.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    CURRENT_POLICY_SELF_PLAY_BLOCKER,
    CURRENT_POLICY_SELF_PLAY_CLAIM,
    OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
    OPPONENT_TRAINING_RELATION_FROZEN_LIGHTZERO_CHECKPOINT,
)


TRAIN_MODULE = "curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train"
EVAL_MODULE = "curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval"
TASK_ROOT = "training/lightzero-curvytron-visual-survival"
PLAN_SCHEMA_ID = "curvyzero_curvytron_frozen_opponent_refresh_plan/v0"


@dataclass(frozen=True, slots=True)
class RefreshPlanRequest:
    initial_opponent_checkpoint_ref: str
    initial_snapshot_ref: str
    stage_count: int = 2
    run_id_prefix: str = "curvytron-visual-survival-debug-lz-refresh"
    attempt_id_prefix: str = "train-gpu-l4t4-refresh"
    seed_start: int = 50
    max_env_step: int = 4096
    max_train_iter: int = 32
    source_max_steps: int = 1024
    collector_env_num: int = 1
    evaluator_env_num: int = 1
    n_evaluator_episode: int = 1
    n_episode: int = 1
    num_simulations: int = 4
    batch_size: int = 16
    save_ckpt_after_iter: int = 4
    refresh_checkpoint_name: str = "ckpt_best.pth.tar"
    eval_selected_iterations: str = "0"
    eval_compute: str = "gpu-l4-t4-cpu40"
    eval_seed_count: int = 5
    eval_max_steps: int = 1024
    eval_num_simulations: int = 2


def build_refresh_plan(request: RefreshPlanRequest) -> dict[str, Any]:
    """Return a JSON-ready staged frozen-opponent plan."""

    _validate_request(request)
    stages: list[dict[str, Any]] = []
    opponent_checkpoint_ref = request.initial_opponent_checkpoint_ref
    opponent_snapshot_ref = request.initial_snapshot_ref

    for stage_index in range(request.stage_count):
        seed = request.seed_start + stage_index
        run_id = f"{request.run_id_prefix}-stage{stage_index:02d}-s{seed}"
        attempt_id = f"{request.attempt_id_prefix}-stage{stage_index:02d}-s{seed}"
        trained_checkpoint_ref = _stage_checkpoint_ref(
            run_id=run_id,
            checkpoint_name=request.refresh_checkpoint_name,
        )
        stage_snapshot_ref = f"{run_id}_{_checkpoint_snapshot_label(request.refresh_checkpoint_name)}"
        stage = {
            "stage_index": stage_index,
            "seed": seed,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "opponent_policy_kind": OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
            "opponent_training_relation": OPPONENT_TRAINING_RELATION_FROZEN_LIGHTZERO_CHECKPOINT,
            "opponent_checkpoint_ref": opponent_checkpoint_ref,
            "opponent_snapshot_ref": opponent_snapshot_ref,
            "current_policy_self_play": CURRENT_POLICY_SELF_PLAY_CLAIM,
            "current_policy_self_play_blocker": CURRENT_POLICY_SELF_PLAY_BLOCKER,
            "train_command": _train_command(
                request=request,
                seed=seed,
                run_id=run_id,
                attempt_id=attempt_id,
                opponent_checkpoint_ref=opponent_checkpoint_ref,
                opponent_snapshot_ref=opponent_snapshot_ref,
            ),
            "fixed_baseline_eval_command": _eval_command(
                request=request,
                run_id=run_id,
                attempt_id=attempt_id,
                eval_id=f"stage{stage_index:02d}_fixed_baseline",
                opponent_checkpoint_ref=None,
                opponent_snapshot_ref=None,
            ),
            "matched_frozen_opponent_eval_command": _eval_command(
                request=request,
                run_id=run_id,
                attempt_id=attempt_id,
                eval_id=f"stage{stage_index:02d}_matched_frozen_opponent",
                opponent_checkpoint_ref=opponent_checkpoint_ref,
                opponent_snapshot_ref=opponent_snapshot_ref,
            ),
            "next_stage_opponent_checkpoint_ref": trained_checkpoint_ref,
            "next_stage_snapshot_ref": stage_snapshot_ref,
            "claim": (
                "learner_vs_frozen_checkpoint_stage; useful staged refresh prep, "
                "not current-policy self-play"
            ),
        }
        stages.append(stage)
        opponent_checkpoint_ref = trained_checkpoint_ref
        opponent_snapshot_ref = stage_snapshot_ref

    return {
        "schema_id": PLAN_SCHEMA_ID,
        "stage_count": request.stage_count,
        "current_policy_self_play": CURRENT_POLICY_SELF_PLAY_CLAIM,
        "current_policy_self_play_blocker": CURRENT_POLICY_SELF_PLAY_BLOCKER,
        "opponent_training_relation": OPPONENT_TRAINING_RELATION_FROZEN_LIGHTZERO_CHECKPOINT,
        "plain_language_status": (
            "This is staged learner-versus-frozen-opponent refresh. It does not "
            "put the live learner policy in both seats during one train_muzero run."
        ),
        "refresh_rule": (
            "After a stage completes and publishes checkpoints, use its mirrored "
            f"{request.refresh_checkpoint_name} as the frozen opponent for the next stage."
        ),
        "request": _dataclass_to_plain(request),
        "stages": stages,
    }


def _validate_request(request: RefreshPlanRequest) -> None:
    if not request.initial_opponent_checkpoint_ref.strip():
        raise ValueError("initial_opponent_checkpoint_ref is required")
    if not request.initial_snapshot_ref.strip():
        raise ValueError("initial_snapshot_ref is required")
    for name in (
        "stage_count",
        "max_env_step",
        "max_train_iter",
        "source_max_steps",
        "collector_env_num",
        "evaluator_env_num",
        "n_evaluator_episode",
        "n_episode",
        "num_simulations",
        "batch_size",
        "save_ckpt_after_iter",
        "eval_seed_count",
        "eval_max_steps",
        "eval_num_simulations",
    ):
        if int(getattr(request, name)) < 1:
            raise ValueError(f"{name} must be at least 1")
    if "/" in request.refresh_checkpoint_name or "\\" in request.refresh_checkpoint_name:
        raise ValueError("refresh_checkpoint_name must be a checkpoint file name")


def _stage_checkpoint_ref(*, run_id: str, checkpoint_name: str) -> str:
    return f"{TASK_ROOT}/{run_id}/checkpoints/lightzero/{checkpoint_name}"


def _checkpoint_snapshot_label(checkpoint_name: str) -> str:
    label = checkpoint_name
    for suffix in (".pth.tar", ".tar", ".pth", ".pt"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
            break
    return label.replace(".", "_").replace("-", "_")


def _train_command(
    *,
    request: RefreshPlanRequest,
    seed: int,
    run_id: str,
    attempt_id: str,
    opponent_checkpoint_ref: str,
    opponent_snapshot_ref: str,
) -> list[str]:
    return [
        "uv",
        "run",
        "--extra",
        "modal",
        "modal",
        "run",
        "-m",
        TRAIN_MODULE,
        "--mode",
        "train",
        "--compute",
        "gpu-l4-t4",
        "--seed",
        str(seed),
        "--run-id",
        run_id,
        "--attempt-id",
        attempt_id,
        "--max-env-step",
        str(request.max_env_step),
        "--max-train-iter",
        str(request.max_train_iter),
        "--source-max-steps",
        str(request.source_max_steps),
        "--collector-env-num",
        str(request.collector_env_num),
        "--evaluator-env-num",
        str(request.evaluator_env_num),
        "--n-evaluator-episode",
        str(request.n_evaluator_episode),
        "--n-episode",
        str(request.n_episode),
        "--num-simulations",
        str(request.num_simulations),
        "--batch-size",
        str(request.batch_size),
        "--save-ckpt-after-iter",
        str(request.save_ckpt_after_iter),
        "--opponent-policy-kind",
        OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
        "--opponent-checkpoint-ref",
        opponent_checkpoint_ref,
        "--snapshot-ref",
        opponent_snapshot_ref,
        "--wait-for-train",
    ]


def _eval_command(
    *,
    request: RefreshPlanRequest,
    run_id: str,
    attempt_id: str,
    eval_id: str,
    opponent_checkpoint_ref: str | None,
    opponent_snapshot_ref: str | None,
) -> list[str]:
    command = [
        "uv",
        "run",
        "--extra",
        "modal",
        "modal",
        "run",
        "-m",
        EVAL_MODULE,
        "--run-id",
        run_id,
        "--attempt-id",
        attempt_id,
        "--selected-iterations",
        request.eval_selected_iterations,
        "--eval-id",
        eval_id,
        "--compute",
        request.eval_compute,
        "--eval-seed-count",
        str(request.eval_seed_count),
        "--max-eval-steps",
        str(request.eval_max_steps),
        "--num-simulations",
        str(request.eval_num_simulations),
        "--parallel",
        "--summary-only",
        "--quiet-framework-logs",
    ]
    if opponent_checkpoint_ref is not None:
        command.extend(
            [
                "--opponent-policy-kind",
                OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
                "--opponent-checkpoint-ref",
                opponent_checkpoint_ref,
            ]
        )
        if opponent_snapshot_ref is not None:
            command.extend(["--opponent-snapshot-ref", opponent_snapshot_ref])
    return command


def _dataclass_to_plain(request: RefreshPlanRequest) -> dict[str, Any]:
    return {
        field: getattr(request, field)
        for field in RefreshPlanRequest.__dataclass_fields__
    }


def _parse_args() -> argparse.Namespace:
    defaults = RefreshPlanRequest(
        initial_opponent_checkpoint_ref="unused",
        initial_snapshot_ref="unused",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Print a staged frozen-opponent refresh plan. This is not "
            "current-policy self-play."
        )
    )
    parser.add_argument("--initial-opponent-checkpoint-ref", required=True)
    parser.add_argument("--initial-snapshot-ref", required=True)
    parser.add_argument("--stage-count", type=int, default=2)
    parser.add_argument("--run-id-prefix", default=defaults.run_id_prefix)
    parser.add_argument("--attempt-id-prefix", default=defaults.attempt_id_prefix)
    parser.add_argument("--seed-start", type=int, default=defaults.seed_start)
    parser.add_argument("--max-env-step", type=int, default=defaults.max_env_step)
    parser.add_argument("--max-train-iter", type=int, default=defaults.max_train_iter)
    parser.add_argument("--source-max-steps", type=int, default=defaults.source_max_steps)
    parser.add_argument("--collector-env-num", type=int, default=defaults.collector_env_num)
    parser.add_argument("--evaluator-env-num", type=int, default=defaults.evaluator_env_num)
    parser.add_argument(
        "--n-evaluator-episode",
        type=int,
        default=defaults.n_evaluator_episode,
    )
    parser.add_argument("--n-episode", type=int, default=defaults.n_episode)
    parser.add_argument("--num-simulations", type=int, default=defaults.num_simulations)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--save-ckpt-after-iter",
        type=int,
        default=defaults.save_ckpt_after_iter,
    )
    parser.add_argument(
        "--refresh-checkpoint-name",
        default=defaults.refresh_checkpoint_name,
    )
    parser.add_argument(
        "--eval-selected-iterations",
        default=defaults.eval_selected_iterations,
    )
    parser.add_argument("--eval-compute", default=defaults.eval_compute)
    parser.add_argument("--eval-seed-count", type=int, default=defaults.eval_seed_count)
    parser.add_argument("--eval-max-steps", type=int, default=defaults.eval_max_steps)
    parser.add_argument(
        "--eval-num-simulations",
        type=int,
        default=defaults.eval_num_simulations,
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    request = RefreshPlanRequest(
        initial_opponent_checkpoint_ref=args.initial_opponent_checkpoint_ref,
        initial_snapshot_ref=args.initial_snapshot_ref,
        stage_count=args.stage_count,
        run_id_prefix=args.run_id_prefix,
        attempt_id_prefix=args.attempt_id_prefix,
        seed_start=args.seed_start,
        max_env_step=args.max_env_step,
        max_train_iter=args.max_train_iter,
        source_max_steps=args.source_max_steps,
        collector_env_num=args.collector_env_num,
        evaluator_env_num=args.evaluator_env_num,
        n_evaluator_episode=args.n_evaluator_episode,
        n_episode=args.n_episode,
        num_simulations=args.num_simulations,
        batch_size=args.batch_size,
        save_ckpt_after_iter=args.save_ckpt_after_iter,
        refresh_checkpoint_name=args.refresh_checkpoint_name,
        eval_selected_iterations=args.eval_selected_iterations,
        eval_compute=args.eval_compute,
        eval_seed_count=args.eval_seed_count,
        eval_max_steps=args.eval_max_steps,
        eval_num_simulations=args.eval_num_simulations,
    )
    plan = build_refresh_plan(request)
    text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
