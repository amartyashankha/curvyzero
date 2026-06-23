#!/usr/bin/env python3
"""Build a local current-chain eval/GIF/tournament-load evidence artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path.cwd() / "artifacts" / "local" / "matplotlib").resolve()),
)

from curvyzero.env.vector_runtime import DEATH_MODE_NORMAL
from curvyzero.infra.modal import lightzero_curvytron_visual_survival_eval as eval_mod
from curvyzero.infra.modal import (
    lightzero_curvyzero_stacked_debug_visual_survival_train as train_mod,
)
from curvyzero.infra.modal import run_management as runs
from curvyzero.tournament import curvytron_checkpoint_tournament as arena
from curvyzero.training.compact_coach_compatibility import (
    COMPACT_COACH_GATE_COACH_SPEED_ROW,
)
from curvyzero.training.compact_coach_compatibility import (
    COMPACT_COACH_LIFECYCLE_GATES,
)
from curvyzero.training.compact_coach_compatibility import (
    COMPACT_COACH_REQUIRED_PROMOTION_GATES,
)
from curvyzero.training.compact_coach_compatibility import (
    COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
)
from curvyzero.training.compact_coach_compatibility import (
    build_compact_coach_compatibility_report_v1,
)
from curvyzero.training.compact_eval_gif_tournament_load import (
    compact_current_chain_eval_gif_tournament_load_evidence_ref,
)
from curvyzero.training.compact_eval_gif_tournament_load import (
    save_compact_current_chain_eval_gif_tournament_load_evidence_v1,
)
from curvyzero.training.compact_muzero_learner import (
    CompactMuZeroLearnerConfigV1,
)
from curvyzero.training.compact_muzero_learner import CompactMuZeroLearnerEdgeV1
from curvyzero.training.compact_owned_loop import CompactPolicyVersionRefV1
from curvyzero.training.compact_owned_loop import (
    compact_owned_loop_replay_store_metadata,
)
from curvyzero.training.compact_policy_row_bridge import CompactReplayIndexRowsV1
from curvyzero.training.compact_policy_refresh_handoff import (
    build_compact_policy_refresh_handoff_v1,
)
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_model_state_digest_v1,
)
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_policy_refresh_metadata_from_state_v1,
)
from curvyzero.training.compact_search_service import (
    COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    COMPACT_STOCK_TOURNAMENT_LOADER_SMOKE_SCHEMA_ID,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    build_current_policy_observation_sidecar_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    save_compact_stock_export_evidence_bundle_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    save_compact_stock_export_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    verify_compact_stock_export_model_contract_v1,
)
from curvyzero.training.compact_torch_search_service import (
    CompactTorchCompileConfig,
)
from curvyzero.training.compact_torch_search_service import CompactTorchSearchServiceV1
from curvyzero.training.compact_trainer_checkpoint import (
    CompactOwnedLoopCountersV1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    CompactOwnedLoopRuntimeStateV1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    CompactTrainerResumeStateV1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    build_compact_trainer_checkpoint_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    save_compact_trainer_checkpoint_v1,
)
from curvyzero.training.compact_training_metrics_lineage import (
    build_compact_training_metrics_lineage_v1,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    _CompactReplayRingV1,
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/local/curvytron_compact_current_chain_eval_gif_tournament_results"
)
DEFAULT_RUN_ID = "optimizer-compact-current-chain-eval-gif-tournament-smoke-20260530"
GAMEPLAY_SMOKE_SCHEMA_ID = "curvyzero_compact_stock_export_one_game_gif_smoke/v1"
POLICY_SOURCE = "compact_current_chain_eval_gif_tournament_smoke"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--game-steps", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=8)
    parser.add_argument("--step-detail-limit", type=int, default=4)
    parser.add_argument("--num-simulations", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--gif-fps", type=float, default=12.0)
    parser.add_argument(
        "--unified-lifecycle-profile-result",
        type=Path,
        default=None,
        help=(
            "Optional normal-death hybrid profile result JSON. When supplied, "
            "the smoke builds one LightZero-shaped compact checkpoint carrying "
            "reward/RND, normal-death, policy-refresh, and metrics-lineage "
            "contracts before attaching current-chain eval/GIF evidence."
        ),
    )
    parser.add_argument(
        "--source-max-steps",
        type=int,
        default=train_mod.DEFAULT_SOURCE_MAX_STEPS,
    )
    parser.add_argument(
        "--decision-source-frames",
        type=int,
        default=train_mod.DEFAULT_DECISION_SOURCE_FRAMES,
    )
    parser.add_argument(
        "--source-physics-step-ms",
        type=float,
        default=train_mod.DEFAULT_SOURCE_PHYSICS_STEP_MS,
    )
    parser.add_argument(
        "--source-max-steps-semantics",
        default="source_physics_steps",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    surface_path = output_dir / "stock_lightzero_model_surface.json"
    compact_checkpoint_path = output_dir / "compact_stock_lightzero_checkpoint.pt"
    stock_export_path = output_dir / "iteration_0.pth.tar"
    verification_report_path = output_dir / "verification_report.json"
    tournament_loader_report_path = output_dir / "tournament_loader_report.json"
    gameplay_report_path = output_dir / "one_game_gif_smoke_report.json"
    eval_summary_path = output_dir / "eval" / "summary.json"
    unified_lifecycle_report_path = output_dir / "unified_lifecycle_report.json"

    model, optimizer, surface = _build_untrained_stock_lightzero_model(
        args=args,
        output_dir=output_dir,
    )
    _write_json(surface_path, surface)

    resume_state = _resume_state(args.run_id)
    replay_store_state = None
    metrics: dict[str, Any] = {"current_chain_smoke_loss": 0.0}
    loop_runtime_state = None
    policy_refresh_handoff = None
    training_metrics_lineage = None
    normal_death_profile_payload = None
    normal_death_evidence_refs: list[str] = []
    trainer_config: dict[str, Any] = {
        "schema_id": "compact_current_chain_eval_gif_tournament_smoke_config/v1",
        "model_source": "fresh_untrained_stock_lightzero_muzero_policy_model",
        "model_surface_path": _relative_ref(surface_path, repo_root),
        "env_variant": train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        "reward_variant": train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        "source_max_steps": int(args.source_max_steps),
        "decision_source_frames": int(args.decision_source_frames),
        "source_physics_step_ms": float(args.source_physics_step_ms),
    }
    if args.unified_lifecycle_profile_result is not None:
        normal_death_profile_path = (
            repo_root / args.unified_lifecycle_profile_result
            if not args.unified_lifecycle_profile_result.is_absolute()
            else args.unified_lifecycle_profile_result
        ).resolve()
        normal_death_profile_payload = _load_profile_payload(normal_death_profile_path)
        normal_death_evidence_refs = [
            _relative_ref(normal_death_profile_path, repo_root),
            "scripts/build_compact_current_chain_eval_gif_tournament_smoke.py:unified_lifecycle",
        ]
        lifecycle = _prepare_unified_lifecycle_contracts(
            checkpoint_id=str(args.run_id),
            resume_state=resume_state,
            model=model,
            optimizer=optimizer,
            surface=surface,
            args=args,
            normal_death_profile_path=normal_death_profile_path,
            repo_root=repo_root,
        )
        replay_store_state = lifecycle["replay_store_state"]
        metrics = lifecycle["metrics"]
        loop_runtime_state = lifecycle["loop_runtime_state"]
        policy_refresh_handoff = lifecycle["policy_refresh_handoff"]
        training_metrics_lineage = lifecycle["training_metrics_lineage"]
        trainer_config.update(
            {
                "unified_lifecycle_local_smoke": True,
                "normal_death_profile_result": _relative_ref(
                    normal_death_profile_path,
                    repo_root,
                ),
                "policy_refresh_handoff_contract": True,
                "training_metrics_lineage_contract": True,
            }
        )

    if replay_store_state is None:
        replay_store_state = _owned_replay_state(
            policy_version_ref=resume_state.policy_version_ref,
            model_version_ref=resume_state.model_version_ref,
        )
    compact_checkpoint = build_compact_trainer_checkpoint_v1(
        checkpoint_id=str(args.run_id),
        trainer_config=trainer_config,
        resume_state=resume_state,
        model=model,
        optimizer=optimizer,
        replay_store_state=replay_store_state,
        metrics=metrics,
        loop_runtime_state=loop_runtime_state,
        death_mode=(
            DEATH_MODE_NORMAL
            if normal_death_profile_payload is not None
            else "profile_no_death"
        ),
        normal_collision_death_profile_result=normal_death_profile_payload,
        normal_collision_death_evidence_id=(
            f"{args.run_id}:normal-death-profile"
            if normal_death_profile_payload is not None
            else ""
        ),
        normal_collision_death_evidence_refs=normal_death_evidence_refs,
        policy_refresh_handoff=policy_refresh_handoff,
        training_metrics_lineage=training_metrics_lineage,
    )
    save_compact_trainer_checkpoint_v1(compact_checkpoint, compact_checkpoint_path)

    sidecar = build_current_policy_observation_sidecar_v1(
        model_env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        model_reward_variant=train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        decision_source_frames=int(args.decision_source_frames),
        source_physics_step_ms=float(args.source_physics_step_ms),
        source_max_steps=int(args.source_max_steps),
        source_max_steps_semantics=str(args.source_max_steps_semantics),
    )
    export = save_compact_stock_export_v1(
        compact_checkpoint,
        stock_export_path,
        policy_metadata=sidecar,
    )

    verification_report = verify_compact_stock_export_model_contract_v1(
        stock_export_path,
        seed=int(args.seed),
        num_simulations=int(args.num_simulations),
        batch_size=int(args.batch_size),
        use_cuda=False,
        state_key=COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
        raise_on_failure=True,
    )
    _write_json(verification_report_path, verification_report)

    stock_export_ref = _relative_ref(stock_export_path, repo_root)
    loader_report = _tournament_loader_report(
        stock_export_ref=stock_export_ref,
        stock_export_path=stock_export_path,
        seed=int(args.seed) + 101,
        source_max_steps=int(args.source_max_steps),
        num_simulations=int(args.num_simulations),
        batch_size=int(args.batch_size),
        telemetry_path=output_dir / "tournament_loader_telemetry.jsonl",
        mount=repo_root,
    )
    _write_json(tournament_loader_report_path, loader_report)

    stock_bundle = save_compact_stock_export_evidence_bundle_v1(
        stock_export_path,
        verification_report_path=verification_report_path,
        tournament_loader_report_path=tournament_loader_report_path,
    )

    game_summary = arena.run_checkpoint_game(
        _game_spec(
            run_id=str(args.run_id),
            stock_export_ref=stock_export_ref,
            sidecar=sidecar,
            seed=int(args.seed) + 202,
            game_steps=int(args.game_steps),
            num_simulations=int(args.num_simulations),
            batch_size=int(args.batch_size),
            frame_stride=int(args.frame_stride),
            gif_fps=float(args.gif_fps),
        ),
        checkpoint_mount=repo_root,
        artifact_mount=output_dir,
    )
    if game_summary.get("ok") is not True:
        raise RuntimeError(f"tournament one-game smoke failed: {game_summary.get('failure')}")

    eval_result = _run_standalone_eval(
        checkpoint_ref=stock_export_ref,
        output_ref=_relative_ref(eval_summary_path, repo_root),
        run_id=str(args.run_id),
        seed=int(args.seed) + 303,
        eval_steps=int(args.eval_steps),
        step_detail_limit=int(args.step_detail_limit),
        source_max_steps=int(args.source_max_steps),
        decision_source_frames=int(args.decision_source_frames),
        source_physics_step_ms=float(args.source_physics_step_ms),
        source_max_steps_semantics=str(args.source_max_steps_semantics),
        num_simulations=int(args.num_simulations),
        batch_size=int(args.batch_size),
        repo_root=repo_root,
    )
    if eval_result.get("ok") is not True:
        raise RuntimeError(f"standalone eval failed: {eval_result.get('wrapper_error')}")

    gameplay_report = _gameplay_report(
        checkpoint_ref=stock_export_ref,
        sidecar_ref=str(export["sidecar_path"]),
        evidence_bundle_ref=str(stock_bundle["path"]),
        verification_report_ref=str(verification_report_path),
        tournament_loader_report_ref=str(tournament_loader_report_path),
        game_summary=game_summary,
        standalone_eval=eval_result,
    )
    _write_json(gameplay_report_path, gameplay_report)

    current_chain = save_compact_current_chain_eval_gif_tournament_load_evidence_v1(
        compact_checkpoint_path=compact_checkpoint_path,
        stock_export_path=stock_export_path,
        stock_export_evidence_bundle_path=stock_bundle["path"],
        gameplay_smoke_report_path=gameplay_report_path,
        profile_result_path=(
            None
            if args.unified_lifecycle_profile_result is None
            else normal_death_profile_path
        ),
    )
    evidence_ref = compact_current_chain_eval_gif_tournament_load_evidence_ref(
        current_chain["evidence"],
    )
    compatibility_report_path = _write_current_chain_coach_compatibility_report(
        output_dir=output_dir,
        current_chain_evidence_ref=evidence_ref,
        compact_checkpoint_metadata=compact_checkpoint.metadata,
        unified_lifecycle=args.unified_lifecycle_profile_result is not None,
        compact_checkpoint_path=compact_checkpoint_path,
        stock_export_path=stock_export_path,
        sidecar_path=export["sidecar_path"],
        verification_report_path=verification_report_path,
        tournament_loader_report_path=tournament_loader_report_path,
        stock_export_evidence_bundle_path=stock_bundle["path"],
        gameplay_report_path=gameplay_report_path,
        standalone_eval_summary_path=eval_summary_path,
        current_chain_evidence_path=current_chain["path"],
    )
    unified_lifecycle_report = None
    if args.unified_lifecycle_profile_result is not None:
        unified_lifecycle_report = _write_unified_lifecycle_report(
            path=unified_lifecycle_report_path,
            compatibility_report_path=compatibility_report_path,
            compact_checkpoint=compact_checkpoint,
            compact_checkpoint_path=compact_checkpoint_path,
            current_chain_evidence=current_chain["evidence"],
            current_chain_evidence_path=current_chain["path"],
            current_chain_evidence_ref=evidence_ref,
            normal_death_profile_result=normal_death_profile_path,
            stock_export_path=stock_export_path,
            gameplay_report_path=gameplay_report_path,
            standalone_eval_summary_path=eval_summary_path,
        )

    report_path = output_dir / "current_chain_eval_gif_tournament_smoke_report.json"
    report = {
        "schema_id": "curvyzero_compact_current_chain_eval_gif_tournament_smoke/v1",
        "ok": True,
        "run_id": str(args.run_id),
        "compact_checkpoint_path": str(compact_checkpoint_path),
        "stock_export_path": str(stock_export_path),
        "stock_export_sidecar_path": str(export["sidecar_path"]),
        "verification_report_path": str(verification_report_path),
        "tournament_loader_report_path": str(tournament_loader_report_path),
        "stock_export_evidence_bundle_path": str(stock_bundle["path"]),
        "gameplay_smoke_report_path": str(gameplay_report_path),
        "game_summary_ref": game_summary["summary_ref"],
        "gif_ref": game_summary["gif_ref"],
        "frames_ref": game_summary["frames_ref"],
        "standalone_eval_summary_path": str(eval_summary_path),
        "current_chain_evidence_path": str(current_chain["path"]),
        "current_chain_evidence_ref": evidence_ref,
        "compatibility_report_path": str(compatibility_report_path),
        "unified_lifecycle_report_path": (
            None
            if unified_lifecycle_report is None
            else str(unified_lifecycle_report_path)
        ),
        "unified_lifecycle_complete": (
            False
            if unified_lifecycle_report is None
            else bool(unified_lifecycle_report["lifecycle_gates_complete"])
        ),
        "physical_steps": int(gameplay_report["physical_steps"]),
        "standalone_eval_steps_survived": int(
            eval_result["status"]["steps_survived"]
        ),
        "promotion_claim": False,
        "training_speed_claim": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
    }
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "current_chain_evidence_path": str(current_chain["path"]),
                "current_chain_evidence_ref": evidence_ref,
                "unified_lifecycle_report_path": (
                    None
                    if unified_lifecycle_report is None
                    else str(unified_lifecycle_report_path)
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def _build_untrained_stock_lightzero_model(
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from ding.config import compile_config
    from lzero.policy.muzero import MuZeroPolicy

    torch.manual_seed(int(args.seed))
    decision_ms = float(args.decision_source_frames) * float(
        args.source_physics_step_ms
    )
    patched = eval_mod._build_visual_survival_configs(
        seed=int(args.seed),
        exp_name=output_dir / "lightzero_model_build_exp",
        telemetry_path=output_dir / "lightzero_model_build_telemetry.jsonl",
        cuda=False,
        max_env_step=int(args.source_max_steps),
        source_max_steps=int(args.source_max_steps),
        decision_ms=decision_ms,
        decision_source_frames=int(args.decision_source_frames),
        source_physics_step_ms=float(args.source_physics_step_ms),
        source_max_steps_semantics=str(args.source_max_steps_semantics),
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        n_episode=1,
        num_simulations=int(args.num_simulations),
        batch_size=int(args.batch_size),
        lightzero_eval_freq=0,
        lightzero_multi_gpu=False,
        max_train_iter=1,
        save_ckpt_after_iter=1,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        reward_outcome_alpha=train_mod.DEFAULT_REWARD_OUTCOME_ALPHA,
        ego_action_straight_override_probability=0.0,
        control_noise_profile_id=train_mod.DEFAULT_CONTROL_NOISE_PROFILE_ID,
        disable_death_for_profile=False,
        env_telemetry_stride=1,
        env_manager_type="base",
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_use_cuda=False,
        opponent_checkpoint=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        natural_bonus_spawn=True,
        opponent_death_mode=train_mod.DEFAULT_OPPONENT_DEATH_MODE,
        opponent_runtime_mode=train_mod.DEFAULT_OPPONENT_RUNTIME_MODE,
    )
    cfg = compile_config(
        copy.deepcopy(patched["main_config"]),
        seed=int(args.seed),
        auto=True,
        create_cfg=copy.deepcopy(patched["create_config"]),
        save_cfg=False,
    )
    cfg.policy.cuda = False
    cfg.policy.device = "cpu"
    policy = MuZeroPolicy(cfg.policy)
    model = getattr(policy, "_model", None)
    if model is None:
        raise AttributeError("MuZeroPolicy has no _model attribute")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    surface = {
        "schema_id": "curvyzero_stock_lightzero_model_build_surface/v1",
        "policy": {
            "cuda": bool(cfg.policy.cuda),
            "device": str(cfg.policy.device),
            "num_simulations": int(cfg.policy.num_simulations),
            "batch_size": int(cfg.policy.batch_size),
            "model_type": str(cfg.policy.model.model_type),
            "observation_shape": _jsonable(cfg.policy.model.observation_shape),
            "image_channel": int(cfg.policy.model.image_channel),
            "frame_stack_num": int(cfg.policy.model.frame_stack_num),
            "action_space_size": int(cfg.policy.model.action_space_size),
            "support_scale": int(cfg.policy.model.support_scale),
        },
        "patched_surface": patched["surface"],
    }
    return model, optimizer, surface


def _resume_state(run_id: str) -> CompactTrainerResumeStateV1:
    return CompactTrainerResumeStateV1(
        trainer_id=f"{run_id}:compact-current-chain-trainer",
        train_step=1,
        learner_update_count=1,
        sample_batch_count=1,
        policy_version_ref=f"{run_id}:policy-update-1",
        model_version_ref=f"{run_id}:model-update-1",
        policy_source=POLICY_SOURCE,
        loop_counters={},
    )


def _owned_replay_state(
    *,
    policy_version_ref: str,
    model_version_ref: str,
    extra: Mapping[str, Any] | None = None,
    populate_rows: int = 0,
):
    policy = CompactPolicyVersionRefV1(
        policy_version_ref=policy_version_ref,
        policy_source=POLICY_SOURCE,
        model_version_ref=model_version_ref,
    )
    metadata = compact_owned_loop_replay_store_metadata(policy, extra=extra)
    ring = _CompactReplayRingV1(capacity=2, metadata=metadata)
    row_count = int(populate_rows)
    if row_count > 0:
        ring.append(
            previous_step=_compact_replay_step(row_count=row_count, actions_offset=0),
            current_step=_compact_replay_step(row_count=row_count, actions_offset=1),
            index_rows=_compact_replay_rows(
                row_count=row_count,
                metadata=metadata,
            ),
        )
    return ring.snapshot_durable_state(
        policy_version_ref=policy.policy_version_ref,
        policy_source=policy.policy_source,
        model_version_ref=policy.model_version_ref,
        metadata=metadata,
    )


def _compact_replay_step(*, row_count: int, actions_offset: int) -> SimpleNamespace:
    rows = int(row_count)
    actions = (
        np.arange(rows, dtype=np.int16) + np.int16(actions_offset)
    ) % np.int16(3)
    rewards = np.zeros((rows,), dtype=np.float32)
    return SimpleNamespace(
        observation=np.zeros((rows, 1, 4, 64, 64), dtype=np.float32),
        action_mask=np.ones((rows, 1, 3), dtype=np.bool_),
        reward=rewards.reshape(rows, 1),
        final_reward_map=rewards.reshape(rows, 1),
        done=np.zeros((rows,), dtype=np.bool_),
        payload={"joint_action": actions.reshape(rows, 1)},
        compact_batch=None,
    )


def _compact_replay_rows(
    *,
    row_count: int,
    metadata: Mapping[str, Any],
) -> CompactReplayIndexRowsV1:
    rows = int(row_count)
    actions = np.arange(rows, dtype=np.int16) % np.int16(3)
    policy_target = np.eye(3, dtype=np.float32)[actions.astype(np.int64)]
    return CompactReplayIndexRowsV1(
        metadata=dict(metadata),
        record_index=1,
        next_record_index=2,
        compact_root_row=np.arange(rows, dtype=np.int32),
        policy_env_id=np.arange(rows, dtype=np.int64),
        policy_row=np.arange(rows, dtype=np.int32),
        env_row=np.arange(rows, dtype=np.int32),
        player=np.zeros((rows,), dtype=np.int16),
        action=actions,
        action_mask=np.ones((rows, 3), dtype=np.bool_),
        policy_target=policy_target,
        root_value=np.zeros((rows,), dtype=np.float32),
        reward=np.zeros((rows,), dtype=np.float32),
        final_reward=np.zeros((rows,), dtype=np.float32),
        done=np.zeros((rows,), dtype=np.bool_),
        terminated=np.zeros((rows,), dtype=np.bool_),
        truncated=np.zeros((rows,), dtype=np.bool_),
        next_final_observation_row=np.zeros((rows,), dtype=np.bool_),
        to_play=np.full((rows,), -1, dtype=np.int64),
        policy_source=POLICY_SOURCE,
    )


def _tournament_loader_report(
    *,
    stock_export_ref: str,
    stock_export_path: Path,
    seed: int,
    source_max_steps: int,
    num_simulations: int,
    batch_size: int,
    telemetry_path: Path,
    mount: Path,
) -> dict[str, Any]:
    loaded = arena._load_policy_from_checkpoint(
        checkpoint_ref=stock_export_ref,
        checkpoint_state_key=COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
        seed=int(seed),
        source_max_steps=int(source_max_steps),
        num_simulations=int(num_simulations),
        batch_size=int(batch_size),
        telemetry_path=telemetry_path,
        mount=mount,
        remote_root=None,
        model_env_variant=None,
        model_reward_variant=None,
    )
    policy = loaded.pop("policy", None)
    del policy
    return {
        "schema_id": COMPACT_STOCK_TOURNAMENT_LOADER_SMOKE_SCHEMA_ID,
        "ok": True,
        "smoke_scope": (
            "local_current_chain_tournament_policy_loader_real_stock_model_"
            "construction_no_game_or_gif"
        ),
        "checkpoint_ref": stock_export_ref,
        "checkpoint_path": str(stock_export_path),
        **loaded,
    }


def _game_spec(
    *,
    run_id: str,
    stock_export_ref: str,
    sidecar: Mapping[str, Any],
    seed: int,
    game_steps: int,
    num_simulations: int,
    batch_size: int,
    frame_stride: int,
    gif_fps: float,
) -> dict[str, Any]:
    player_base = {
        "checkpoint_ref": stock_export_ref,
        "checkpoint_state_key": COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
        "model_env_variant": sidecar["model_env_variant"],
        "model_reward_variant": sidecar["model_reward_variant"],
        "policy_trail_render_mode": sidecar["policy_trail_render_mode"],
        "policy_bonus_render_mode": sidecar["policy_bonus_render_mode"],
        "policy_observation_backend": sidecar["policy_observation_backend"],
        "policy_observation_contract_id": sidecar["policy_observation_contract_id"],
        "observation_contract": sidecar["observation_contract"],
    }
    return {
        "tournament_id": run_id,
        "battle_id": "compact-current-chain-self-play-smoke",
        "pair_index": 0,
        "game_id": "game-000000",
        "players": [
            {"checkpoint_id": "current-chain-seat-0", **player_base},
            {"checkpoint_id": "current-chain-seat-1", **player_base},
        ],
        "games_per_pair": 1,
        "seed": int(seed),
        "max_steps": int(game_steps),
        "num_simulations": int(num_simulations),
        "policy_batch_size": int(batch_size),
        "policy_mode": arena.POLICY_MODE_EVAL,
        "save_gif": True,
        "save_frames_npz": True,
        "frame_stride": int(frame_stride),
        "gif_fps": float(gif_fps),
        "decision_ms": float(sidecar["decision_ms"]),
        "decision_source_frames": int(sidecar["decision_source_frames"]),
        "source_physics_step_ms": float(sidecar["source_physics_step_ms"]),
        "source_max_steps_semantics": str(sidecar["source_max_steps_semantics"]),
        "natural_bonus_spawn": True,
    }


def _run_standalone_eval(
    *,
    checkpoint_ref: str,
    output_ref: str,
    run_id: str,
    seed: int,
    eval_steps: int,
    step_detail_limit: int,
    source_max_steps: int,
    decision_source_frames: int,
    source_physics_step_ms: float,
    source_max_steps_semantics: str,
    num_simulations: int,
    batch_size: int,
    repo_root: Path,
) -> dict[str, Any]:
    eval_mod.RUNS_MOUNT = repo_root
    return eval_mod._run_eval(
        compute="cpu",
        checkpoint_ref=checkpoint_ref,
        output_ref=output_ref,
        run_id=run_id,
        attempt_id="local-current-chain-smoke",
        seed=int(seed),
        max_eval_steps=int(eval_steps),
        step_detail_limit=int(step_detail_limit),
        source_max_steps=int(source_max_steps),
        decision_ms=float(decision_source_frames) * float(source_physics_step_ms),
        decision_source_frames=int(decision_source_frames),
        source_physics_step_ms=float(source_physics_step_ms),
        source_max_steps_semantics=str(source_max_steps_semantics),
        num_simulations=int(num_simulations),
        batch_size=int(batch_size),
        emit_result_json=False,
        quiet_framework_logs=True,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        model_reward_variant=train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_checkpoint_ref=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        natural_bonus_spawn=True,
        opponent_death_mode=train_mod.DEFAULT_OPPONENT_DEATH_MODE,
        opponent_runtime_mode=train_mod.DEFAULT_OPPONENT_RUNTIME_MODE,
        commit=False,
    )


def _gameplay_report(
    *,
    checkpoint_ref: str,
    sidecar_ref: str,
    evidence_bundle_ref: str,
    verification_report_ref: str,
    tournament_loader_report_ref: str,
    game_summary: Mapping[str, Any],
    standalone_eval: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_id": GAMEPLAY_SMOKE_SCHEMA_ID,
        "ok": True,
        "checkpoint_ref": checkpoint_ref,
        "sidecar_ref": sidecar_ref,
        "evidence_bundle_ref": evidence_bundle_ref,
        "verification_report_ref": verification_report_ref,
        "tournament_loader_report_ref": tournament_loader_report_ref,
        "policy_loads": game_summary["policy_loads"],
        "first_action_trace": list(game_summary["action_trace"][:1]),
        "physical_steps": int(game_summary["physical_steps"]),
        "action_counts": game_summary["action_counts"],
        "done": bool(game_summary["done"]),
        "truncated": bool(game_summary["truncated"]),
        "failure": game_summary.get("failure"),
        "gif_ref": str(game_summary["gif_ref"]),
        "frames_ref": str(game_summary["frames_ref"]),
        "game_summary_ref": str(game_summary["summary_ref"]),
        "standalone_eval": {
            "ok": bool(standalone_eval["ok"]),
            "status": standalone_eval["status"],
            "load_state_dict": standalone_eval["surface"]["load_state_dict"],
            "artifact": standalone_eval["artifact"],
            "telemetry_artifact": standalone_eval.get("telemetry_artifact"),
            "wrapper_error": standalone_eval.get("wrapper_error"),
        },
        "promotion_claim": False,
        "training_speed_claim": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
    }


def _prepare_unified_lifecycle_contracts(
    *,
    checkpoint_id: str,
    resume_state: CompactTrainerResumeStateV1,
    model: Any,
    optimizer: Any,
    surface: Mapping[str, Any],
    args: argparse.Namespace,
    normal_death_profile_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    import torch

    policy_surface = dict(surface["policy"])
    observation_shape = tuple(int(dim) for dim in policy_surface["observation_shape"])
    action_space_size = int(policy_surface["action_space_size"])
    support_scale = int(policy_surface["support_scale"])
    search_metadata = _unified_search_metadata(
        checkpoint_id=checkpoint_id,
        num_simulations=int(args.num_simulations),
        active_root_count=max(2, int(args.batch_size)),
    )
    sample = _resident_lightzero_sample(
        observation_shape=observation_shape,
        action_space_size=action_space_size,
        row_count=max(2, int(args.batch_size)),
        metadata=search_metadata,
    )
    learner = CompactMuZeroLearnerEdgeV1(
        model=model,
        optimizer=optimizer,
        config=CompactMuZeroLearnerConfigV1(
            device="cpu",
            support_scale=support_scale,
            num_unroll_steps=1,
            require_device_replay_rows=True,
        ),
    )
    step_result = learner.train_on_sample_batch(sample, train_steps=1)
    learner_telemetry = dict(step_result.telemetry)
    metrics = {
        "unified_lifecycle_local_loss": float(
            learner_telemetry["compact_muzero_learner_loss"]
        ),
        "last_learner_telemetry": learner_telemetry,
    }
    search_worker_model = copy.deepcopy(model)
    search_service = CompactTorchSearchServiceV1(
        policy=_ModelPolicy(search_worker_model),
        num_simulations=max(1, int(args.num_simulations)),
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
        ),
    )
    learner_digest = compact_model_state_digest_v1(model)
    search_worker_state = search_service.refresh_model_state(
        model_state_dict=model.state_dict(),
        policy_version_ref=resume_state.policy_version_ref,
        model_version_ref=resume_state.model_version_ref,
        policy_source=resume_state.policy_source,
        learner_update_count=resume_state.learner_update_count,
        expected_model_state_digest=learner_digest,
    )
    row_metadata = compact_policy_refresh_metadata_from_state_v1(
        search_worker_state
    )
    stamped_metadata = {**search_metadata, **row_metadata}
    replay_state = _owned_replay_state(
        policy_version_ref=resume_state.policy_version_ref,
        model_version_ref=resume_state.model_version_ref,
        extra=stamped_metadata,
        populate_rows=max(1, int(args.batch_size)),
    )
    loop_runtime_state = _unified_loop_runtime_state(
        learner_telemetry=learner_telemetry,
        sample_metadata=stamped_metadata,
    )
    normal_death_ref = _relative_ref(normal_death_profile_path, repo_root)
    policy_refresh_handoff = build_compact_policy_refresh_handoff_v1(
        checkpoint_id=checkpoint_id,
        resume_state=resume_state,
        learner_model=model,
        search_worker_state=search_worker_state,
        root_metadata=stamped_metadata,
        action_metadata=stamped_metadata,
        replay_metadata=stamped_metadata,
        sample_metadata=stamped_metadata,
        evidence_refs=(
            "local_lightzero_compact_learner_update",
            "local_compact_torch_search_worker_refresh",
            normal_death_ref,
        ),
    )
    training_metrics_lineage = build_compact_training_metrics_lineage_v1(
        checkpoint_id=checkpoint_id,
        resume_state=resume_state,
        replay_store_state=replay_state,
        metrics=metrics,
        loop_runtime_state=loop_runtime_state,
        evidence_refs=(
            "local_lightzero_compact_learner_update",
            "local_resident_sample_metadata",
            normal_death_ref,
        ),
    )
    del torch
    return {
        "replay_store_state": replay_state,
        "metrics": metrics,
        "loop_runtime_state": loop_runtime_state,
        "policy_refresh_handoff": policy_refresh_handoff,
        "training_metrics_lineage": training_metrics_lineage,
    }


class _ModelPolicy:
    def __init__(self, model: Any) -> None:
        self._model = model


def _resident_lightzero_sample(
    *,
    observation_shape: tuple[int, ...],
    action_space_size: int,
    row_count: int,
    metadata: Mapping[str, Any],
) -> Any:
    import torch

    rows = int(row_count)
    actions = torch.arange(rows, dtype=torch.int64) % int(action_space_size)
    next_actions = (actions + 1) % int(action_space_size)
    policy_target = torch.zeros((rows, int(action_space_size)), dtype=torch.float32)
    next_policy_target = torch.zeros_like(policy_target)
    policy_target[torch.arange(rows), actions] = 1.0
    next_policy_target[torch.arange(rows), next_actions] = 1.0
    observation = torch.zeros((rows, *observation_shape), dtype=torch.uint8)
    return argparse.Namespace(
        metadata=dict(metadata),
        observation=observation,
        action=actions.to(dtype=torch.int16),
        action_mask=torch.ones((rows, int(action_space_size)), dtype=torch.bool),
        policy_target=policy_target,
        root_value=torch.zeros((rows,), dtype=torch.float32),
        reward=torch.zeros((rows,), dtype=torch.float32),
        weights=torch.ones((rows,), dtype=torch.float32),
        next_action_mask=torch.ones((rows, int(action_space_size)), dtype=torch.bool),
        next_policy_target=next_policy_target,
        next_root_value=torch.zeros((rows,), dtype=torch.float32),
    )


def _unified_search_metadata(
    *,
    checkpoint_id: str,
    num_simulations: int,
    active_root_count: int,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(b"curvyzero_unified_lifecycle_synthetic_search_stamp/v1")
    digest.update(str(checkpoint_id).encode("utf-8"))
    digest.update(str(num_simulations).encode("utf-8"))
    digest.update(str(active_root_count).encode("utf-8"))
    return {
        "resident_device_sample_batch": True,
        "device_replay_index_rows_sample": True,
        "search_impl": CompactTorchSearchServiceV1.search_impl,
        "num_simulations": int(num_simulations),
        "active_root_count": int(active_root_count),
        "root_batch_schema_id": "curvyzero_compact_root_batch/v1",
        "search_result_schema_id": "curvyzero_compact_search_result/v1",
        "replay_payload_schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
        "search_replay_payload_digest": digest.hexdigest(),
    }


def _unified_loop_runtime_state(
    *,
    learner_telemetry: Mapping[str, Any],
    sample_metadata: Mapping[str, Any],
) -> CompactOwnedLoopRuntimeStateV1:
    sample_rows = int(learner_telemetry["compact_muzero_learner_sample_rows"])
    input_bytes = int(learner_telemetry.get("compact_muzero_learner_input_bytes", 0))
    counters = CompactOwnedLoopCountersV1(
        sample_gate_calls=1,
        sample_gate_opportunities=1,
        sample_gate_skipped_count=0,
        sample_gate_index_rows=sample_rows,
        sample_gate_target_rows=sample_rows,
        sample_gate_sample_rows=sample_rows,
        sample_gate_sec=0.0,
        sample_gate_last_telemetry={
            "compact_rollout_slab_sample_gate_enabled": True,
            "compact_rollout_slab_sample_gate_resident_sample_batch": True,
            "compact_rollout_slab_sample_gate_device_replay_index_rows": True,
        },
        sample_gate_last_sample_metadata=dict(sample_metadata),
        learner_gate_calls=1,
        learner_gate_updates=1,
        learner_gate_sample_rows=sample_rows,
        learner_gate_input_bytes=input_bytes,
        learner_gate_sec=float(learner_telemetry.get("compact_muzero_learner_sec", 0.0)),
        learner_gate_last_telemetry=dict(learner_telemetry),
    )
    return CompactOwnedLoopRuntimeStateV1(previous_step=None, counters=counters)


def _write_current_chain_coach_compatibility_report(
    *,
    output_dir: Path,
    current_chain_evidence_ref: str,
    compact_checkpoint_metadata: Mapping[str, Any],
    unified_lifecycle: bool,
    compact_checkpoint_path: Path,
    stock_export_path: Path,
    sidecar_path: Path,
    verification_report_path: Path,
    tournament_loader_report_path: Path,
    stock_export_evidence_bundle_path: Path,
    gameplay_report_path: Path,
    standalone_eval_summary_path: Path,
    current_chain_evidence_path: Path,
) -> Path:
    gates = {gate: False for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES}
    evidence: dict[str, str]
    if unified_lifecycle:
        for gate in COMPACT_COACH_LIFECYCLE_GATES:
            gates[gate] = bool(
                compact_checkpoint_metadata.get(
                    f"compact_coach_compatibility_gate_{gate}",
                    False,
                )
            )
        gates["eval_gif_tournament_load"] = True
        gates[COMPACT_COACH_GATE_COACH_SPEED_ROW] = False
        evidence = {
            str(key): str(value)
            for key, value in dict(
                compact_checkpoint_metadata.get(
                    "compact_coach_compatibility_evidence",
                    {},
                )
                or {}
            ).items()
        }
        evidence["eval_gif_tournament_load"] = current_chain_evidence_ref
    else:
        gates.update(
            {
                "trainer_entrypoint": True,
                "checkpoint_save_load": True,
                "resume_metadata": True,
                "eval_gif_tournament_load": True,
            }
        )
        evidence = {
            "trainer_entrypoint": (
                "scripts/build_compact_current_chain_eval_gif_tournament_smoke.py"
            ),
            "checkpoint_save_load": str(compact_checkpoint_path),
            "resume_metadata": str(compact_checkpoint_path),
            "eval_gif_tournament_load": current_chain_evidence_ref,
        }
    report = build_compact_coach_compatibility_report_v1(
        route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        profile_only=False,
        calls_train_muzero=False,
        touches_live_runs=False,
        speed_currency="compact_trainer_lifecycle_evidence_no_speed",
        gates=gates,
        evidence=evidence,
        promotion_claim=False,
    )
    metadata = report.as_metadata()
    passed = [gate for gate in report.required_gates if gate not in report.missing_required_gates]
    lifecycle_missing = [
        gate
        for gate in COMPACT_COACH_LIFECYCLE_GATES
        if not bool(gates.get(gate)) or not str(evidence.get(gate, "")).strip()
    ]
    payload = {
        "schema_id": "curvyzero_compact_coach_compatibility_refresh/v1",
        "ok": True,
        "refresh_scope": (
            "post_unified_lifecycle_current_chain_eval_gif_tournament_load"
            if unified_lifecycle
            else "post_current_chain_eval_gif_tournament_load_fail_closed"
        ),
        "route": report.route,
        "profile_only": report.profile_only,
        "calls_train_muzero": report.calls_train_muzero,
        "touches_live_runs": report.touches_live_runs,
        "compatibility_metadata": metadata,
        "lifecycle_gates_complete": not lifecycle_missing,
        "lifecycle_missing_gates_or_evidence": lifecycle_missing,
        "passed_required_gates": passed,
        "missing_required_gates": list(report.missing_required_gates),
        "missing_required_evidence": list(report.missing_required_evidence),
        "promotion_eligible": report.promotion_eligible,
        "promotion_blocker": report.promotion_blocker,
        "promotion_claim": report.promotion_claim,
        "selected_next_missing_gate": (
            report.missing_required_gates[0]
            if report.missing_required_gates
            else None
        ),
        "selected_next_missing_gate_reason": (
            (
                "All local lifecycle gates are bound to this stock-shaped compact "
                "checkpoint plus sibling current-chain eval/GIF evidence. The "
                "remaining required promotion gate is an explicit real Coach "
                "speed/training row."
            )
            if unified_lifecycle
            else (
                "Current-chain eval/GIF/tournament-load is now evidenced for this "
                "stock-shaped compact checkpoint. The remaining gates must be bound "
                "to a unified compact checkpoint lifecycle before any promotion or "
                "Coach speed claim."
            )
        ),
        "evidence_refs": {
            "compact_checkpoint": str(compact_checkpoint_path),
            "stock_export": str(stock_export_path),
            "policy_sidecar": str(sidecar_path),
            "strict_verifier_report": str(verification_report_path),
            "tournament_loader_report": str(tournament_loader_report_path),
            "stock_export_evidence_bundle": str(stock_export_evidence_bundle_path),
            "one_game_gif_eval_smoke": str(gameplay_report_path),
            "standalone_eval_summary": str(standalone_eval_summary_path),
            "current_chain_evidence": str(current_chain_evidence_path),
            "current_chain_evidence_ref": current_chain_evidence_ref,
        },
        "non_claims": {
            "all_lifecycle_gates_same_candidate": bool(
                unified_lifecycle and not lifecycle_missing
            ),
            "all_required_gates_same_checkpoint": False,
            "coach_speed_row": False,
            "rating_or_promotion_quality_claim": False,
            "training_speed_claim": False,
            "stock_resume_claim": False,
            "base_export_claims_mutated": False,
            "live_run_safety_claim": False,
        },
    }
    path = output_dir / "compatibility_report.json"
    _write_json(path, payload)
    return path


def _write_unified_lifecycle_report(
    *,
    path: Path,
    compatibility_report_path: Path,
    compact_checkpoint: Any,
    compact_checkpoint_path: Path,
    current_chain_evidence: Mapping[str, Any],
    current_chain_evidence_path: Path,
    current_chain_evidence_ref: str,
    normal_death_profile_result: Path,
    stock_export_path: Path,
    gameplay_report_path: Path,
    standalone_eval_summary_path: Path,
) -> dict[str, Any]:
    metadata = compact_checkpoint.metadata
    identity = current_chain_evidence["current_chain_identity"]
    if str(identity["checkpoint_id"]) != str(metadata["checkpoint_id"]):
        raise RuntimeError("current-chain checkpoint_id does not match checkpoint")
    checkpoint_evidence = dict(metadata["compact_coach_compatibility_evidence"])
    gates = {gate: False for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES}
    for gate in COMPACT_COACH_LIFECYCLE_GATES:
        gates[gate] = bool(
            metadata.get(f"compact_coach_compatibility_gate_{gate}", False)
        )
    gates["eval_gif_tournament_load"] = True
    gates[COMPACT_COACH_GATE_COACH_SPEED_ROW] = False
    evidence = {
        **checkpoint_evidence,
        "eval_gif_tournament_load": current_chain_evidence_ref,
    }
    report = build_compact_coach_compatibility_report_v1(
        route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        profile_only=False,
        calls_train_muzero=False,
        touches_live_runs=False,
        speed_currency="compact_trainer_lifecycle_evidence_no_speed",
        gates=gates,
        evidence=evidence,
        promotion_claim=False,
    )
    lifecycle_missing = [
        gate
        for gate in COMPACT_COACH_LIFECYCLE_GATES
        if not bool(gates.get(gate)) or not str(evidence.get(gate, "")).strip()
    ]
    payload = {
        "schema_id": "curvyzero_compact_unified_lifecycle_smoke/v1",
        "ok": True,
        "run_id": str(metadata["checkpoint_id"]),
        "checkpoint_id": str(metadata["checkpoint_id"]),
        "compact_checkpoint_path": str(compact_checkpoint_path),
        "normal_death_profile_result": str(normal_death_profile_result),
        "stock_export_path": str(stock_export_path),
        "gameplay_smoke_report_path": str(gameplay_report_path),
        "standalone_eval_summary_path": str(standalone_eval_summary_path),
        "current_chain_evidence_path": str(current_chain_evidence_path),
        "current_chain_evidence_ref": current_chain_evidence_ref,
        "base_current_chain_compatibility_report_path": str(compatibility_report_path),
        "lifecycle_gates": {gate: bool(gates.get(gate)) for gate in COMPACT_COACH_LIFECYCLE_GATES},
        "lifecycle_gates_complete": not lifecycle_missing,
        "lifecycle_missing_gates_or_evidence": lifecycle_missing,
        "coach_speed_row_gate": False,
        "compatibility_metadata": report.as_metadata(),
        "passed_required_gates": [
            gate for gate in report.required_gates if gate not in report.missing_required_gates
        ],
        "missing_required_gates": list(report.missing_required_gates),
        "missing_required_evidence": list(report.missing_required_evidence),
        "promotion_eligible": report.promotion_eligible,
        "promotion_blocker": report.promotion_blocker,
        "promotion_claim": report.promotion_claim,
        "non_claims": {
            "coach_speed_row": False,
            "training_speed_claim": False,
            "rating_or_promotion_quality_claim": False,
            "stock_resume_claim": False,
            "live_run_safety_claim": False,
            "base_checkpoint_eval_metadata_mutated": False,
        },
    }
    _write_json(path, payload)
    return payload


def _load_profile_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("compact"), dict):
        return payload["compact"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("profile result must be a JSON object")


def _relative_ref(path: Path, base: Path) -> str:
    return runs.require_relative_ref(path.resolve().relative_to(base.resolve()).as_posix()).as_posix()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
