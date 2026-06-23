#!/usr/bin/env python3
"""Produce compact-only longer-horizon learning metrics evidence."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from curvyzero.env.vector_multiplayer_env import ACTION_COUNT
from curvyzero.training.compact_longer_horizon_learning_metrics import (
    COMPACT_RECORD_STEP_ENTRYPOINT,
)
from curvyzero.training.compact_longer_horizon_learning_metrics import (
    build_compact_longer_horizon_learning_metrics_v1,
)
from curvyzero.training.compact_longer_horizon_learning_metrics import (
    validate_compact_longer_horizon_learning_metrics_v1,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_CANDIDATE_ROLE,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_OWNED_TRAINER_ROUTE,
)
from curvyzero.training.compact_trainer_checkpoint import (
    save_compact_trainer_checkpoint_v1,
)


def _load_sibling_script(module_name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


candidate = _load_sibling_script(
    "build_compact_matched_learning_quality_compact_candidate_producer_for_longer_horizon",
    "build_compact_matched_learning_quality_compact_candidate_producer.py",
)


DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_compact_promotion_readiness_results")
DEFAULT_UNIFIED_LIFECYCLE_REPORT = candidate.DEFAULT_UNIFIED_LIFECYCLE_REPORT
DEFAULT_COMPATIBILITY_REPORT = Path(
    "artifacts/local/curvytron_compact_coach_compatibility_results"
    "/optimizer-compact-coach-compatibility-after-speed-row-h100-threshold-20260530"
    "/compatibility_report.json"
)
DEFAULT_RUN_ID = "optimizer-compact-longer-horizon-learning-metrics-20260530"
PRODUCER_SCHEMA_ID = "curvyzero_compact_longer_horizon_learning_metrics_producer/v1"
PRODUCER_ID = "scripts/build_compact_longer_horizon_learning_metrics_producer.py"


@dataclass(frozen=True, slots=True)
class CompactLongerHorizonSnapshot:
    checkpoint_index: int
    checkpoint_step: int
    checkpoint_id: str
    checkpoint: Any
    model_state_digest: str
    digest_changed_from_previous: bool
    trainer_counters: Mapping[str, Any]
    interval_denominators: Mapping[str, Any]
    cumulative_denominators: Mapping[str, Any]
    learner_metrics: Mapping[str, Any]
    loop_metrics: Mapping[str, Any]
    training_metrics_lineage: Mapping[str, Any] | None
    sample_source: str = COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE
    compact_training_entrypoint: str = COMPACT_RECORD_STEP_ENTRYPOINT


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path.cwd()
    output_dir = (repo_root / args.output_root / args.run_id).resolve()
    candidate._prepare_fresh_output_dir(output_dir)

    lifecycle_path = candidate._resolve_path(args.unified_lifecycle_report, repo_root)
    lifecycle = candidate._read_json_mapping(lifecycle_path, "unified lifecycle report")
    candidate._validate_lifecycle_report(lifecycle)
    compatibility_path = candidate._resolve_path(args.compatibility_report, repo_root)
    context = candidate._load_compact_candidate_context(
        args=args,
        repo_root=repo_root,
        output_dir=output_dir,
        lifecycle=lifecycle,
        lifecycle_path=lifecycle_path,
    )
    snapshots = _train_compact_longer_horizon_trace(
        args=args,
        context=context,
        run_id=str(args.run_id),
    )
    checkpoint_series = _write_checkpoint_series_outputs(
        args=args,
        repo_root=repo_root,
        output_dir=output_dir,
        context=context,
        snapshots=snapshots,
    )
    report_path = output_dir / "longer_horizon_learning_metrics_report.json"
    report = build_compact_longer_horizon_learning_metrics_v1(
        run_id=str(args.run_id),
        compatibility_report_path=compatibility_path,
        unified_lifecycle_report_path=lifecycle_path,
        checkpoint_series=checkpoint_series,
        source_fingerprint=_source_fingerprint(
            args=args,
            repo_root=repo_root,
            lifecycle_path=lifecycle_path,
            context=context,
        ),
        eval_settings=candidate._eval_settings(args=args),
        training_settings=_training_settings(args=args),
    )
    _write_json(report_path, report)
    validate_compact_longer_horizon_learning_metrics_v1(report)

    manifest_path = output_dir / "longer_horizon_learning_metrics_producer_manifest.json"
    manifest = _producer_manifest(
        args=args,
        lifecycle_path=lifecycle_path,
        compatibility_path=compatibility_path,
        context=context,
        report_path=report_path,
        snapshots=snapshots,
    )
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "ok": True,
                "role": COMPACT_CANDIDATE_ROLE,
                "run_id": str(args.run_id),
                "report_path": str(report_path),
                "manifest_path": str(manifest_path),
                "checkpoint_count": len(snapshots),
            },
            sort_keys=True,
        )
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--unified-lifecycle-report",
        type=Path,
        default=DEFAULT_UNIFIED_LIFECYCLE_REPORT,
    )
    parser.add_argument(
        "--compatibility-report",
        type=Path,
        default=DEFAULT_COMPATIBILITY_REPORT,
    )
    parser.add_argument("--compact-checkpoint", type=Path)
    parser.add_argument("--candidate-checkpoint-id")
    parser.add_argument("--checkpoint-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--eval-seed", type=int, default=20260833)
    parser.add_argument("--eval-seed-count", type=int, default=2)
    parser.add_argument("--eval-seed-rng-seed", type=int)
    parser.add_argument("--source-max-steps", type=int, default=1_048_576)
    parser.add_argument("--eval-steps", type=int, default=128)
    parser.add_argument("--step-detail-limit", type=int, default=0)
    parser.add_argument("--decision-source-frames", type=int, default=1)
    parser.add_argument("--source-physics-step-ms", type=float, default=50.0 / 3.0)
    parser.add_argument(
        "--source-max-steps-semantics",
        default="source_physics_steps",
    )
    parser.add_argument("--num-simulations", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument(
        "--compact-training-mode",
        choices=(candidate.COMPACT_TRAINING_MODE_ENV_SEARCH_REPLAY,),
        default=candidate.COMPACT_TRAINING_MODE_ENV_SEARCH_REPLAY,
    )
    parser.add_argument("--compact-env-steps", type=int, default=8)
    parser.add_argument("--compact-warmup-steps", type=int, default=1)
    parser.add_argument("--compact-sample-batch-size", type=int, default=2)
    parser.add_argument("--compact-sample-interval", type=int, default=1)
    parser.add_argument("--compact-replay-pair-capacity", type=int, default=32)
    parser.add_argument(
        "--learner-device",
        default="cpu",
        choices=("cpu", "cuda", "auto"),
    )
    parser.add_argument("--hardware-class", default="local-cpu-producer-smoke")
    args = parser.parse_args(argv)
    if int(args.checkpoint_count) < 3:
        raise ValueError("checkpoint_count must be at least 3")
    return args


def _train_compact_longer_horizon_trace(
    *,
    args: argparse.Namespace,
    context: Any,
    run_id: str,
) -> list[CompactLongerHorizonSnapshot]:
    import torch

    policy_surface = candidate._policy_surface(context.surface)
    support_scale = int(policy_surface["support_scale"])
    resume_state = context.checkpoint.resume_state
    policy_source = str(resume_state.policy_source)
    learner_edge = candidate.CompactMuZeroLearnerEdgeV1(
        model=context.model,
        optimizer=context.optimizer,
        config=candidate.CompactMuZeroLearnerConfigV1(
            device=str(args.learner_device),
            support_scale=support_scale,
            num_unroll_steps=1,
            require_resident_sample=True,
            require_device_replay_rows=True,
        ),
    )
    loop_learner = candidate._CompactMuZeroLoopLearnerAdapter(learner_edge)
    search_device = candidate._resident_renderer_device(str(args.learner_device))
    search_model = copy.deepcopy(context.model).to(torch.device(search_device))
    search_service = candidate.CompactTorchSearchServiceV1(
        policy=candidate._CompactCandidateSearchPolicy(search_model),
        num_simulations=int(args.num_simulations),
        root_noise_weight=0.0,
        compile_config=candidate.CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
        ),
        require_resident_observation=True,
    )
    search_service.refresh_model_state(
        model_state_dict=context.model.state_dict(),
        policy_version_ref=str(resume_state.policy_version_ref),
        model_version_ref=str(resume_state.model_version_ref),
        policy_source=policy_source,
        learner_update_count=max(1, int(resume_state.learner_update_count)),
        expected_model_state_digest=candidate.compact_model_state_digest_v1(
            context.model
        ),
    )
    slab = candidate.CompactRolloutSlab(
        batch_size=int(args.batch_size),
        player_count=2,
        search_service=search_service,
        search_lane="compact_longer_horizon_learning_metrics",
        policy_source=policy_source,
        copy_root_observation=False,
    )
    replay_store = candidate._CompactReplayRingV1(
        capacity=int(args.compact_replay_pair_capacity),
        metadata=candidate.compact_owned_loop_replay_store_metadata(
            candidate.CompactPolicyVersionRefV1(
                policy_version_ref=str(resume_state.policy_version_ref),
                policy_source=policy_source,
                model_version_ref=str(resume_state.model_version_ref),
            ),
            extra={
                "compact_longer_horizon_learning_metrics_producer": True,
                "sample_source": COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE,
            },
        ),
    )
    loop = candidate.CompactOwnedLoopV1(
        config=candidate.CompactOwnedLoopConfigV1(
            sample_batch_size=int(args.compact_sample_batch_size),
            sample_interval=int(args.compact_sample_interval),
            replay_capacity=int(args.compact_replay_pair_capacity),
            learner_train_steps=int(args.train_steps),
            num_unroll_steps=1,
            sample_seed_base=int(args.seed),
            learner_impl=candidate.COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_COMPACT_MUZERO,
            require_next_targets=True,
            capture_replay_store_state=True,
        ),
        policy_version=candidate.CompactPolicyVersionRefV1(
            policy_version_ref=str(resume_state.policy_version_ref),
            policy_source=policy_source,
            model_version_ref=str(resume_state.model_version_ref),
        ),
        replay_store=replay_store,
        learner=loop_learner,
    )
    trainer = candidate.CompactOwnedTrainerV1(
        config=candidate.CompactOwnedTrainerConfigV1(
            trainer_id=f"{run_id}:compact-longer-horizon-trainer",
            policy_source=policy_source,
            initial_policy_version_ref=str(resume_state.policy_version_ref),
            initial_model_version_ref=str(resume_state.model_version_ref),
        ),
        learner=loop_learner,
        loop=loop,
    )
    candidate._seed_trainer_from_resume_state(trainer, resume_state)
    counters_before = candidate._trainer_counters(trainer)
    initial_digest = candidate.compact_model_state_digest_v1(context.model)
    zero_denominators = _zero_denominators()
    snapshots: list[CompactLongerHorizonSnapshot] = [
        CompactLongerHorizonSnapshot(
            checkpoint_index=0,
            checkpoint_step=0,
            checkpoint_id=str(context.checkpoint_id),
            checkpoint=context.checkpoint,
            model_state_digest=initial_digest,
            digest_changed_from_previous=False,
            trainer_counters=counters_before,
            interval_denominators=zero_denominators,
            cumulative_denominators=zero_denominators,
            learner_metrics={},
            loop_metrics={},
            training_metrics_lineage=None,
        )
    ]
    manager = candidate.HybridBatchedObservationProfileManager(
        candidate.HybridObservationProfileConfig(
            batch_size=int(args.batch_size),
            actor_count=1,
            steps=int(args.compact_env_steps),
            warmup_steps=int(args.compact_warmup_steps),
            seed=int(args.seed),
            max_ticks=int(args.source_max_steps),
            decision_source_frames=int(args.decision_source_frames),
            source_physics_step_ms=float(args.source_physics_step_ms),
            stack_storage_dtype=candidate.HYBRID_STACK_STORAGE_DTYPE_UINT8,
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
            compact_rollout_slab_sample_gate=True,
            compact_rollout_slab_sample_gate_batch_size=int(
                args.compact_sample_batch_size
            ),
            compact_rollout_slab_sample_gate_interval=int(args.compact_sample_interval),
            compact_rollout_slab_sample_gate_replay_pair_capacity=int(
                args.compact_replay_pair_capacity
            ),
            compact_rollout_slab_learner_gate=False,
            compact_owned_loop_entrypoint=False,
        ),
        observation_renderer=candidate._PersistentDeviceRenderer(device=search_device),
        compact_rollout_slab=slab,
    )
    rng = np.random.default_rng(int(args.seed) + 7919)
    next_action: np.ndarray | None = None
    committed_index_rows = 0
    started = time.perf_counter()
    target_count = int(args.checkpoint_count)
    for iteration in range(int(args.compact_warmup_steps) + int(args.compact_env_steps)):
        if next_action is None:
            action = rng.integers(
                0,
                ACTION_COUNT,
                size=(int(args.batch_size), 2),
                dtype=np.int16,
            )
        else:
            action = next_action.astype(np.int16, copy=True)
        step = manager.step(action)
        slab_step = step.compact_rollout_slab_step
        if slab_step is None:
            raise ValueError("compact longer-horizon training produced no slab step")
        next_action = slab_step.next_joint_action.astype(np.int16, copy=True)
        if iteration < int(args.compact_warmup_steps):
            loop.prime_previous_step(step)
            continue
        index_rows = slab_step.committed_index_rows
        if index_rows is None:
            continue
        row_count = int(getattr(index_rows.action, "shape", (0,))[0])
        committed_index_rows += row_count
        previous_updates = int(trainer.learner_update_count)
        trainer.record_step(current_step=step, index_rows=index_rows)
        if int(trainer.learner_update_count) <= previous_updates:
            continue
        digest = candidate.compact_model_state_digest_v1(context.model)
        search_service.refresh_model_state(
            model_state_dict=context.model.state_dict(),
            policy_version_ref=trainer.policy_version_ref,
            model_version_ref=trainer.model_version_ref,
            policy_source=policy_source,
            learner_update_count=int(trainer.learner_update_count),
            expected_model_state_digest=digest,
        )
        snapshot_index = len(snapshots)
        loop_telemetry = loop.telemetry()
        telemetry = dict(
            loop_telemetry.get("compact_owned_loop_learner_gate_last_telemetry") or {}
        )
        if not telemetry:
            telemetry = dict(trainer.last_learner_telemetry)
        replay_state = loop.snapshot_replay_store_state()
        learner_loss = candidate._learner_loss_from_telemetry(telemetry)
        checkpoint = trainer.checkpoint(
            checkpoint_id=f"{run_id}:compact-longer-horizon-checkpoint-{snapshot_index}",
            replay_store_state=replay_state,
            metrics={
                "compact_longer_horizon_learning_metrics": True,
                "last_learner_telemetry": telemetry,
                "compact_owned_loop_telemetry": loop_telemetry,
                "learner_loss_mean": learner_loss,
            },
            training_metrics_lineage_evidence_refs=(
                f"compact_longer_horizon_learning_metrics:{run_id}:"
                f"checkpoint_{snapshot_index}",
            ),
            extra_metadata={
                "compact_longer_horizon_learning_metrics_producer": True,
                "matched_learning_quality_sample_source": (
                    COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE
                ),
            },
        )
        cumulative = _cumulative_denominators(
            trainer=trainer,
            counters_before=counters_before,
            loop_telemetry=loop_telemetry,
            committed_index_rows=committed_index_rows,
            elapsed_sec=max(0.0, time.perf_counter() - started),
        )
        snapshots.append(
            CompactLongerHorizonSnapshot(
                checkpoint_index=snapshot_index,
                checkpoint_step=snapshot_index,
                checkpoint_id=f"{run_id}:compact-longer-horizon-checkpoint-{snapshot_index}",
                checkpoint=checkpoint,
                model_state_digest=digest,
                digest_changed_from_previous=(
                    digest != snapshots[-1].model_state_digest
                ),
                trainer_counters=candidate._trainer_counters(trainer),
                interval_denominators=_interval_denominators(
                    cumulative,
                    snapshots[-1].cumulative_denominators,
                ),
                cumulative_denominators=cumulative,
                learner_metrics=_learner_metrics_from_telemetry(telemetry),
                loop_metrics=_loop_metrics(loop_telemetry),
                training_metrics_lineage=checkpoint.metadata.get(
                    "compact_training_metrics_lineage"
                ),
            )
        )
        if len(snapshots) >= target_count:
            break
    if len(snapshots) < target_count:
        raise RuntimeError(
            "compact longer-horizon producer did not observe enough learner "
            f"updates for {target_count} checkpoints; observed {len(snapshots)}"
        )
    return snapshots


def _write_checkpoint_series_outputs(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    output_dir: Path,
    context: Any,
    snapshots: Sequence[CompactLongerHorizonSnapshot],
) -> list[dict[str, Any]]:
    sidecar = candidate.build_current_policy_observation_sidecar_v1(
        decision_source_frames=int(args.decision_source_frames),
        source_physics_step_ms=float(args.source_physics_step_ms),
        source_max_steps=int(args.source_max_steps),
        source_max_steps_semantics=str(args.source_max_steps_semantics),
    )
    series: list[dict[str, Any]] = []
    for snapshot in snapshots:
        index = int(snapshot.checkpoint_index)
        point_id = (
            "pre_train"
            if index == 0
            else "post_train"
            if index == len(snapshots) - 1
            else f"checkpoint_{index}"
        )
        compact_path = output_dir / "compact_checkpoints" / f"iteration_{index}.compact.pth.tar"
        stock_path = output_dir / "stock_exports" / f"iteration_{index}.pth.tar"
        eval_path = output_dir / "eval" / f"iteration_{index}_eval_summary.json"
        _save_compact_checkpoint(snapshot.checkpoint, compact_path)
        candidate._save_stock_checkpoint_export(
            checkpoint=snapshot.checkpoint,
            path=stock_path,
            policy_metadata=sidecar,
            extra_metadata={
                "compact_longer_horizon_learning_metrics_checkpoint": True,
                "compact_longer_horizon_checkpoint_index": index,
                "producer_id": PRODUCER_ID,
            },
        )
        eval_summary = candidate._run_eval_summary(
            args=args,
            checkpoint_path=stock_path,
            output_path=eval_path,
            run_id=str(args.run_id),
            seed=int(args.eval_seed),
            repo_root=repo_root,
        )
        candidate._assert_eval_summary_bound_to_checkpoint(
            eval_summary,
            expected_checkpoint_path=stock_path,
            label=f"checkpoint {index} eval summary",
        )
        series.append(
            {
                "candidate_checkpoint_id": str(context.checkpoint_id),
                "point_id": point_id,
                "checkpoint_index": index,
                "checkpoint_step": int(snapshot.checkpoint_step),
                "checkpoint_id": str(snapshot.checkpoint_id),
                "compact_checkpoint_path": str(compact_path),
                "compact_checkpoint_sha256": _file_sha256(compact_path),
                "stock_export_path": str(stock_path),
                "stock_export_sha256": _file_sha256(stock_path),
                "eval_summary_path": str(eval_path),
                "eval_summary_sha256": _file_sha256(eval_path),
                "model_state_digest": str(snapshot.model_state_digest),
                "digest_changed_from_previous": bool(
                    snapshot.digest_changed_from_previous
                ),
                "trainer_counters": dict(snapshot.trainer_counters),
                "interval_denominators": dict(snapshot.interval_denominators),
                "cumulative_denominators": dict(snapshot.cumulative_denominators),
                "learner_metrics": dict(snapshot.learner_metrics),
                "loop_metrics": dict(snapshot.loop_metrics),
                "training_metrics_lineage": (
                    None
                    if snapshot.training_metrics_lineage is None
                    else dict(snapshot.training_metrics_lineage)
                ),
                "sample_source": snapshot.sample_source,
                "compact_training_entrypoint": snapshot.compact_training_entrypoint,
            }
        )
    return series


def _save_compact_checkpoint(checkpoint: Any, path: Path) -> Path:
    return save_compact_trainer_checkpoint_v1(checkpoint, path)


def _source_fingerprint(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    lifecycle_path: Path,
    context: Any,
) -> dict[str, Any]:
    source = dict(
        candidate._source_fingerprint(
            args=args,
            repo_root=repo_root,
            lifecycle_path=lifecycle_path,
            context=context,
        )
    )
    source.update(
        {
            "producer_script": PRODUCER_ID,
            "producer_route": "compact_longer_horizon_env_search_replay_producer",
        }
    )
    return source


def _training_settings(*, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "route": COMPACT_OWNED_TRAINER_ROUTE,
        "sample_source": COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE,
        "compact_training_entrypoint": COMPACT_RECORD_STEP_ENTRYPOINT,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "training_speedup_claim": False,
        "wall_sec_used_for_speed_claim": False,
        "checkpoint_count": int(args.checkpoint_count),
        "compact_env_steps": int(args.compact_env_steps),
        "compact_warmup_steps": int(args.compact_warmup_steps),
        "compact_sample_batch_size": int(args.compact_sample_batch_size),
        "compact_sample_interval": int(args.compact_sample_interval),
        "compact_replay_pair_capacity": int(args.compact_replay_pair_capacity),
        "learner_device": str(args.learner_device),
        "train_steps_per_learner_update": int(args.train_steps),
    }


def _zero_denominators() -> dict[str, Any]:
    return {
        "train_step_delta": 0,
        "learner_update_count_delta": 0,
        "sample_batch_count_delta": 0,
        "record_step_calls": 0,
        "appended_replay_entry_count": 0,
        "sampled_count": 0,
        "trained_count": 0,
        "compact_rollout_rows": 0,
        "compact_sample_rows": 0,
        "replay_store_entry_count": 0,
        "replay_store_index_row_count": 0,
        "training_wall_sec": 0.0,
    }


def _cumulative_denominators(
    *,
    trainer: Any,
    counters_before: Mapping[str, Any],
    loop_telemetry: Mapping[str, Any],
    committed_index_rows: int,
    elapsed_sec: float,
) -> dict[str, Any]:
    return {
        "train_step_delta": int(trainer.train_step) - int(counters_before["train_step"]),
        "learner_update_count_delta": int(trainer.learner_update_count)
        - int(counters_before["learner_update_count"]),
        "sample_batch_count_delta": int(trainer.sample_batch_count)
        - int(counters_before["sample_batch_count"]),
        "record_step_calls": int(trainer.record_step_calls),
        "appended_replay_entry_count": int(trainer.appended_replay_entry_count),
        "sampled_count": int(trainer.sampled_count),
        "trained_count": int(trainer.trained_count),
        "compact_rollout_rows": int(committed_index_rows),
        "compact_sample_rows": int(
            loop_telemetry.get("compact_owned_loop_sample_gate_sample_row_count", 0)
        ),
        "replay_store_entry_count": int(
            loop_telemetry.get("compact_owned_loop_replay_store_entry_count", 0)
        ),
        "replay_store_index_row_count": int(
            loop_telemetry.get("compact_owned_loop_replay_store_index_row_count", 0)
        ),
        "training_wall_sec": float(max(elapsed_sec, 0.0)),
    }


def _interval_denominators(
    current: Mapping[str, Any],
    previous: Mapping[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in current.items():
        if isinstance(value, float):
            out[key] = float(value) - float(previous.get(key, 0.0))
        else:
            out[key] = int(value) - int(previous.get(key, 0))
    return out


def _learner_metrics_from_telemetry(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "learner_loss": float(telemetry["compact_muzero_learner_loss"]),
        "learner_policy_loss": float(telemetry["compact_muzero_learner_policy_loss"]),
        "learner_value_loss": float(telemetry["compact_muzero_learner_value_loss"]),
        "learner_reward_loss": float(telemetry["compact_muzero_learner_reward_loss"]),
        "learner_grad_norm_before_clip": float(
            telemetry["compact_muzero_learner_grad_norm_before_clip"]
        ),
        "learner_sample_rows": int(telemetry["compact_muzero_learner_sample_rows"]),
        "learner_train_steps": int(telemetry["compact_muzero_learner_train_steps"]),
    }


def _loop_metrics(loop_telemetry: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "compact_owned_loop_sample_gate_calls",
        "compact_owned_loop_sample_gate_sample_row_count",
        "compact_owned_loop_learner_gate_calls",
        "compact_owned_loop_learner_gate_updates",
        "compact_owned_loop_learner_gate_sample_row_count",
        "compact_owned_loop_replay_store_entry_count",
        "compact_owned_loop_replay_store_index_row_count",
    )
    return {key: loop_telemetry.get(key) for key in keys if key in loop_telemetry}


def _producer_manifest(
    *,
    args: argparse.Namespace,
    lifecycle_path: Path,
    compatibility_path: Path,
    context: Any,
    report_path: Path,
    snapshots: Sequence[CompactLongerHorizonSnapshot],
) -> dict[str, Any]:
    return {
        "schema_id": PRODUCER_SCHEMA_ID,
        "ok": True,
        "role": COMPACT_CANDIDATE_ROLE,
        "producer_id": PRODUCER_ID,
        "run_id": str(args.run_id),
        "created_at": datetime.now(UTC).isoformat(),
        "support_only": False,
        "profile_only": False,
        "touches_live_runs": False,
        "calls_train_muzero": False,
        "real_compact_owned_training_work": True,
        "sample_source": COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE,
        "compact_training_entrypoint": COMPACT_RECORD_STEP_ENTRYPOINT,
        "checkpoint_count": len(snapshots),
        "unified_lifecycle_report_path": str(lifecycle_path),
        "compatibility_report_path": str(compatibility_path),
        "loaded_checkpoint_path": str(context.checkpoint_path),
        "loaded_checkpoint_sha256": context.checkpoint_sha256,
        "report_path": str(report_path),
        "report_sha256": _file_sha256(report_path),
        "attached_claims": {
            "longer_horizon_compact_learning_metrics": True,
            "promotion_claim": False,
            "training_speedup_claim": False,
            "live_run_safety_claim": False,
            "rating_or_promotion_quality_claim": False,
            "leaderboard_claim": False,
            "touches_live_runs": False,
            "calls_train_muzero": False,
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
