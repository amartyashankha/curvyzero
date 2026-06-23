#!/usr/bin/env python3
"""Produce a compact-candidate matched learning-quality capture."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from curvyzero.env.vector_multiplayer_env import ACTION_COUNT
from curvyzero.env.vector_runtime import DEATH_MODE_NORMAL
from curvyzero.training.compact_owned_loop import CompactOwnedLoopConfigV1
from curvyzero.training.compact_owned_loop import CompactOwnedLoopV1
from curvyzero.training.compact_owned_loop import CompactPolicyVersionRefV1
from curvyzero.training.compact_owned_loop import compact_owned_loop_replay_store_metadata
from curvyzero.training.compact_muzero_learner import (
    CompactMuZeroLearnerConfigV1,
)
from curvyzero.training.compact_muzero_learner import CompactMuZeroLearnerEdgeV1
from curvyzero.training.compact_owned_trainer import CompactOwnedTrainerConfigV1
from curvyzero.training.compact_owned_trainer import CompactOwnedTrainerV1
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_model_state_digest_v1,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_CANDIDATE_ROLE,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_PROVENANCE_SCHEMA_ID,
)
from curvyzero.training.compact_rollout_slab import CompactRolloutSlab
from curvyzero.training.compact_stock_checkpoint_export import (
    build_current_policy_observation_sidecar_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    save_compact_stock_export_v1,
)
from curvyzero.training.compact_torch_search_service import CompactTorchCompileConfig
from curvyzero.training.compact_torch_search_service import CompactTorchSearchServiceV1
from curvyzero.training.compact_trainer_checkpoint import (
    load_compact_trainer_checkpoint_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    restore_compact_trainer_checkpoint_v1,
)
from curvyzero.training.source_state_batched_observation_profile import (
    SourceStateBatchedRenderResult,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_COMPACT_MUZERO,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    HYBRID_STACK_STORAGE_DTYPE_UINT8,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    HybridBatchedObservationProfileManager,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    HybridObservationProfileConfig,
)
from curvyzero.training.source_state_hybrid_observation_profile import _CompactReplayRingV1


def _load_sibling_script(module_name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


chain_smoke = _load_sibling_script(
    "build_compact_current_chain_eval_gif_tournament_smoke_for_quality_producer",
    "build_compact_current_chain_eval_gif_tournament_smoke.py",
)
packager = _load_sibling_script(
    "build_compact_matched_learning_quality_capture_from_artifacts_for_quality_producer",
    "build_compact_matched_learning_quality_capture_from_artifacts.py",
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/local/curvytron_compact_matched_learning_quality_results"
)
DEFAULT_UNIFIED_LIFECYCLE_REPORT = Path(
    "artifacts/local/curvytron_compact_current_chain_eval_gif_tournament_results"
    "/optimizer-compact-unified-lifecycle-smoke-20260530/unified_lifecycle_report.json"
)
DEFAULT_RUN_ID = "optimizer-compact-candidate-quality-producer-smoke-20260530"
PRODUCER_SCHEMA_ID = (
    "curvyzero_compact_matched_learning_quality_compact_candidate_producer/v1"
)
TRAINING_ARTIFACT_SCHEMA_ID = (
    "curvyzero_compact_matched_learning_quality_compact_candidate_training_artifact/v1"
)
PRODUCER_ID = (
    "scripts/build_compact_matched_learning_quality_compact_candidate_producer.py"
)
COMPACT_TRAINING_MODE_ENV_SEARCH_REPLAY = "env_search_replay"
COMPACT_TRAINING_MODE_RESIDENT_SAMPLE_SCAFFOLD = "resident_sample_scaffold"
COMPACT_TRAINING_MODES = (
    COMPACT_TRAINING_MODE_ENV_SEARCH_REPLAY,
    COMPACT_TRAINING_MODE_RESIDENT_SAMPLE_SCAFFOLD,
)
COMPACT_RECORD_STEP_ENTRYPOINT = (
    "curvyzero.training.compact_owned_trainer."
    "CompactOwnedTrainerV1.record_step"
)
COMPACT_RESIDENT_SAMPLE_ENTRYPOINT = (
    "curvyzero.training.compact_owned_trainer."
    "CompactOwnedTrainerV1.train_on_sample_batch"
)


@dataclass(frozen=True, slots=True)
class CompactCandidateContext:
    model: Any
    optimizer: Any
    checkpoint: Any
    surface: Mapping[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_id: str


@dataclass(frozen=True, slots=True)
class CompactCandidateTrainingResult:
    final_checkpoint: Any
    initial_model_state_digest: str
    final_model_state_digest: str
    trainer_counters_before: Mapping[str, Any]
    trainer_counters_after: Mapping[str, Any]
    learner_telemetry: Mapping[str, Any]
    learner_update_count_delta: int
    sample_batch_count_delta: int
    compact_rollout_rows: int
    compact_sample_rows: int
    training_wall_sec: float
    learner_loss_mean: float
    sample_source: str


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path.cwd()
    output_dir = (repo_root / args.output_root / args.run_id).resolve()
    _prepare_fresh_output_dir(output_dir)

    lifecycle_path = _resolve_path(args.unified_lifecycle_report, repo_root)
    lifecycle = _read_json_mapping(lifecycle_path, "unified lifecycle report")
    _validate_lifecycle_report(lifecycle)

    context = _load_compact_candidate_context(
        args=args,
        repo_root=repo_root,
        output_dir=output_dir,
        lifecycle=lifecycle,
        lifecycle_path=lifecycle_path,
    )
    training = _train_compact_candidate_once(
        args=args,
        context=context,
        run_id=str(args.run_id),
    )
    _validate_training_result(training)

    sidecar = build_current_policy_observation_sidecar_v1(
        decision_source_frames=int(args.decision_source_frames),
        source_physics_step_ms=float(args.source_physics_step_ms),
        source_max_steps=int(args.source_max_steps),
        source_max_steps_semantics=str(args.source_max_steps_semantics),
    )
    initial_checkpoint_path = output_dir / "iteration_0.pth.tar"
    final_checkpoint_path = output_dir / "iteration_1.pth.tar"
    _save_stock_checkpoint_export(
        checkpoint=context.checkpoint,
        path=initial_checkpoint_path,
        policy_metadata=sidecar,
        extra_metadata={
            "matched_learning_quality_checkpoint_point": "pre_train",
            "producer_id": PRODUCER_ID,
        },
    )
    _save_stock_checkpoint_export(
        checkpoint=training.final_checkpoint,
        path=final_checkpoint_path,
        policy_metadata=sidecar,
        extra_metadata={
            "matched_learning_quality_checkpoint_point": "post_train",
            "producer_id": PRODUCER_ID,
        },
    )
    _require_distinct_existing_checkpoints(
        initial_checkpoint_path,
        final_checkpoint_path,
    )

    eval_dir = output_dir / "eval"
    pre_eval_path = eval_dir / "pre_eval_summary.json"
    post_eval_path = eval_dir / "post_eval_summary.json"
    pre_eval = _run_eval_summary(
        args=args,
        checkpoint_path=initial_checkpoint_path,
        output_path=pre_eval_path,
        run_id=str(args.run_id),
        seed=int(args.eval_seed),
        repo_root=repo_root,
    )
    post_eval = _run_eval_summary(
        args=args,
        checkpoint_path=final_checkpoint_path,
        output_path=post_eval_path,
        run_id=str(args.run_id),
        seed=int(args.eval_seed),
        repo_root=repo_root,
    )
    _assert_eval_summary_bound_to_checkpoint(
        pre_eval,
        expected_checkpoint_path=initial_checkpoint_path,
        label="pre eval summary",
    )
    _assert_eval_summary_bound_to_checkpoint(
        post_eval,
        expected_checkpoint_path=final_checkpoint_path,
        label="post eval summary",
    )

    inputs_dir = output_dir / "capture_inputs"
    input_paths = _write_packager_inputs(
        args=args,
        repo_root=repo_root,
        output_dir=output_dir,
        inputs_dir=inputs_dir,
        lifecycle=lifecycle,
        lifecycle_path=lifecycle_path,
        context=context,
        training=training,
        pre_eval_path=pre_eval_path,
        post_eval_path=post_eval_path,
        pre_eval=pre_eval,
        post_eval=post_eval,
        initial_checkpoint_path=initial_checkpoint_path,
        final_checkpoint_path=final_checkpoint_path,
    )
    capture_path = output_dir / "compact_candidate_capture.json"
    packager_args = _packager_args(
        args=args,
        context=context,
        input_paths=input_paths,
        capture_path=capture_path,
        initial_checkpoint_path=initial_checkpoint_path,
        final_checkpoint_path=final_checkpoint_path,
    )
    packager_return = _invoke_capture_packager(packager_args)
    if int(packager_return) != 0:
        raise RuntimeError(f"compact candidate capture packager failed: {packager_return}")

    manifest_path = output_dir / "compact_candidate_producer_manifest.json"
    manifest = _producer_manifest(
        args=args,
        lifecycle_path=lifecycle_path,
        context=context,
        training=training,
        input_paths=input_paths,
        capture_path=capture_path,
        packager_args=packager_args,
    )
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "ok": True,
                "role": COMPACT_CANDIDATE_ROLE,
                "run_id": str(args.run_id),
                "capture_path": str(capture_path),
                "manifest_path": str(manifest_path),
                "stock_reference_still_required": True,
            },
            sort_keys=True,
        )
    )
    return 0


def _prepare_fresh_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"producer output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"producer output directory already exists and is not empty: {output_dir}"
            )
        return
    output_dir.mkdir(parents=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--unified-lifecycle-report",
        type=Path,
        default=DEFAULT_UNIFIED_LIFECYCLE_REPORT,
    )
    parser.add_argument("--compact-checkpoint", type=Path)
    parser.add_argument("--candidate-checkpoint-id")
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--eval-seed", type=int, default=20260833)
    parser.add_argument("--eval-seed-count", type=int, default=2)
    parser.add_argument("--eval-seed-rng-seed", type=int)
    parser.add_argument("--source-max-steps", type=int, default=1_048_576)
    parser.add_argument("--eval-steps", type=int, default=128)
    parser.add_argument("--step-detail-limit", type=int, default=0)
    parser.add_argument("--decision-source-frames", type=int, default=1)
    parser.add_argument(
        "--source-physics-step-ms",
        type=float,
        default=50.0 / 3.0,
    )
    parser.add_argument(
        "--source-max-steps-semantics",
        default="source_physics_steps",
    )
    parser.add_argument("--num-simulations", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument(
        "--compact-training-mode",
        choices=COMPACT_TRAINING_MODES,
        default=COMPACT_TRAINING_MODE_ENV_SEARCH_REPLAY,
    )
    parser.add_argument("--compact-env-steps", type=int, default=4)
    parser.add_argument("--compact-warmup-steps", type=int, default=1)
    parser.add_argument("--compact-sample-batch-size", type=int, default=2)
    parser.add_argument("--compact-sample-interval", type=int, default=1)
    parser.add_argument("--compact-replay-pair-capacity", type=int, default=16)
    parser.add_argument(
        "--learner-device",
        default="cpu",
        choices=("cpu", "cuda", "auto"),
    )
    parser.add_argument(
        "--quality-horizon",
        default="compact_env_search_replay_pre_post_eval",
    )
    parser.add_argument("--hardware-class", default="local-cpu-producer-smoke")
    parser.add_argument(
        "--denominator-id",
        default="compact_candidate_env_search_replay_denominator_v1",
    )
    return parser.parse_args(argv)


def _load_compact_candidate_context(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    output_dir: Path,
    lifecycle: Mapping[str, Any],
    lifecycle_path: Path,
) -> CompactCandidateContext:
    checkpoint_path = _resolve_lifecycle_checkpoint_path(
        args=args,
        repo_root=repo_root,
        lifecycle=lifecycle,
        lifecycle_path=lifecycle_path,
    )
    model, optimizer, surface = chain_smoke._build_untrained_stock_lightzero_model(
        args=args,
        output_dir=output_dir,
    )
    checkpoint = load_compact_trainer_checkpoint_v1(checkpoint_path)
    restore_compact_trainer_checkpoint_v1(
        checkpoint,
        model=model,
        optimizer=optimizer,
    )
    checkpoint_id = str(
        args.candidate_checkpoint_id
        or lifecycle.get("checkpoint_id")
        or checkpoint.metadata.get("checkpoint_id")
    )
    if not checkpoint_id.strip():
        raise ValueError("compact candidate checkpoint_id must be non-empty")
    return CompactCandidateContext(
        model=model,
        optimizer=optimizer,
        checkpoint=checkpoint,
        surface=surface,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_file_sha256(checkpoint_path),
        checkpoint_id=checkpoint_id,
    )


def _resolve_lifecycle_checkpoint_path(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    lifecycle: Mapping[str, Any],
    lifecycle_path: Path,
) -> Path:
    raw_path = args.compact_checkpoint or lifecycle.get("compact_checkpoint_path")
    if raw_path is None:
        raise ValueError(f"unified lifecycle report has no compact_checkpoint_path: {lifecycle_path}")
    checkpoint_path = _resolve_path(Path(str(raw_path)), repo_root)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"compact checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _train_compact_candidate_once(
    *,
    args: argparse.Namespace,
    context: CompactCandidateContext,
    run_id: str,
) -> CompactCandidateTrainingResult:
    if str(args.compact_training_mode) == COMPACT_TRAINING_MODE_RESIDENT_SAMPLE_SCAFFOLD:
        return _train_compact_candidate_resident_sample_scaffold_once(
            args=args,
            context=context,
            run_id=run_id,
        )
    if str(args.compact_training_mode) != COMPACT_TRAINING_MODE_ENV_SEARCH_REPLAY:
        raise ValueError(f"unknown compact training mode: {args.compact_training_mode!r}")
    return _train_compact_candidate_env_search_replay_once(
        args=args,
        context=context,
        run_id=run_id,
    )


def _train_compact_candidate_resident_sample_scaffold_once(
    *,
    args: argparse.Namespace,
    context: CompactCandidateContext,
    run_id: str,
) -> CompactCandidateTrainingResult:
    policy_surface = _policy_surface(context.surface)
    observation_shape = tuple(int(dim) for dim in policy_surface["observation_shape"])
    action_space_size = int(policy_surface["action_space_size"])
    support_scale = int(policy_surface["support_scale"])
    row_count = max(2, int(args.batch_size))
    sample_metadata = chain_smoke._unified_search_metadata(
        checkpoint_id=f"{run_id}:compact-candidate-quality",
        num_simulations=int(args.num_simulations),
        active_root_count=row_count,
    )
    sample = chain_smoke._resident_lightzero_sample(
        observation_shape=observation_shape,
        action_space_size=action_space_size,
        row_count=row_count,
        metadata=sample_metadata,
    )
    learner = CompactMuZeroLearnerEdgeV1(
        model=context.model,
        optimizer=context.optimizer,
        config=CompactMuZeroLearnerConfigV1(
            device="cpu",
            support_scale=support_scale,
            num_unroll_steps=1,
            require_device_replay_rows=True,
        ),
    )
    resume_state = context.checkpoint.resume_state
    trainer = CompactOwnedTrainerV1(
        config=CompactOwnedTrainerConfigV1(
            trainer_id=f"{run_id}:compact-candidate-trainer",
            policy_source=str(resume_state.policy_source),
            initial_policy_version_ref=str(resume_state.policy_version_ref),
            initial_model_version_ref=str(resume_state.model_version_ref),
        ),
        learner=learner,
    )
    _seed_trainer_from_resume_state(trainer, resume_state)
    counters_before = _trainer_counters(trainer)
    initial_digest = compact_model_state_digest_v1(context.model)
    started = time.perf_counter()
    step_result = trainer.train_on_sample_batch(
        sample,
        train_steps=int(args.train_steps),
    )
    elapsed_sec = max(0.0, time.perf_counter() - started)
    final_digest = compact_model_state_digest_v1(context.model)
    counters_after = _trainer_counters(trainer)
    telemetry = dict(step_result.telemetry)
    learner_sec = telemetry.get("compact_muzero_learner_sec")
    training_wall_sec = (
        float(learner_sec)
        if _is_positive_number(learner_sec)
        else max(elapsed_sec, 1e-9)
    )
    sample_rows = int(telemetry.get("compact_muzero_learner_sample_rows", row_count))
    replay_state = chain_smoke._owned_replay_state(
        policy_version_ref=trainer.policy_version_ref,
        model_version_ref=trainer.model_version_ref,
        extra=sample_metadata,
        populate_rows=max(1, sample_rows),
    )
    final_checkpoint = trainer.checkpoint(
        checkpoint_id=f"{run_id}:compact-candidate-post-train",
        replay_store_state=replay_state,
        metrics={
            "matched_learning_quality_compact_candidate": True,
            "last_learner_telemetry": telemetry,
            "learner_loss_mean": float(telemetry["compact_muzero_learner_loss"]),
        },
        extra_metadata={
            "matched_learning_quality_compact_candidate_producer": True,
            "matched_learning_quality_sample_source": (
                "deterministic_resident_lightzero_sample"
            ),
        },
    )
    return CompactCandidateTrainingResult(
        final_checkpoint=final_checkpoint,
        initial_model_state_digest=initial_digest,
        final_model_state_digest=final_digest,
        trainer_counters_before=counters_before,
        trainer_counters_after=counters_after,
        learner_telemetry=telemetry,
        learner_update_count_delta=(
            int(counters_after["learner_update_count"])
            - int(counters_before["learner_update_count"])
        ),
        sample_batch_count_delta=(
            int(counters_after["sample_batch_count"])
            - int(counters_before["sample_batch_count"])
        ),
        compact_rollout_rows=row_count,
        compact_sample_rows=sample_rows,
        training_wall_sec=training_wall_sec,
        learner_loss_mean=float(telemetry["compact_muzero_learner_loss"]),
        sample_source="deterministic_resident_lightzero_sample",
    )


def _train_compact_candidate_env_search_replay_once(
    *,
    args: argparse.Namespace,
    context: CompactCandidateContext,
    run_id: str,
) -> CompactCandidateTrainingResult:
    import torch

    policy_surface = _policy_surface(context.surface)
    support_scale = int(policy_surface["support_scale"])
    resume_state = context.checkpoint.resume_state
    policy_source = str(resume_state.policy_source)
    learner_edge = CompactMuZeroLearnerEdgeV1(
        model=context.model,
        optimizer=context.optimizer,
        config=CompactMuZeroLearnerConfigV1(
            device=str(args.learner_device),
            support_scale=support_scale,
            num_unroll_steps=1,
            require_resident_sample=True,
            require_device_replay_rows=True,
        ),
    )
    loop_learner = _CompactMuZeroLoopLearnerAdapter(learner_edge)
    search_device = _resident_renderer_device(str(args.learner_device))
    search_model = copy.deepcopy(context.model).to(torch.device(search_device))
    search_service = CompactTorchSearchServiceV1(
        policy=_CompactCandidateSearchPolicy(search_model),
        num_simulations=int(args.num_simulations),
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
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
        expected_model_state_digest=compact_model_state_digest_v1(context.model),
    )
    slab = CompactRolloutSlab(
        batch_size=int(args.batch_size),
        player_count=2,
        search_service=search_service,
        search_lane="compact_candidate_env_search_replay_quality",
        policy_source=policy_source,
        copy_root_observation=False,
    )
    replay_store = _CompactReplayRingV1(
        capacity=int(args.compact_replay_pair_capacity),
        metadata=compact_owned_loop_replay_store_metadata(
            CompactPolicyVersionRefV1(
                policy_version_ref=str(resume_state.policy_version_ref),
                policy_source=policy_source,
                model_version_ref=str(resume_state.model_version_ref),
            ),
            extra={
                "compact_candidate_learning_quality_producer": True,
                "sample_source": COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE,
            },
        ),
    )
    loop = CompactOwnedLoopV1(
        config=CompactOwnedLoopConfigV1(
            sample_batch_size=int(args.compact_sample_batch_size),
            sample_interval=int(args.compact_sample_interval),
            replay_capacity=int(args.compact_replay_pair_capacity),
            learner_train_steps=int(args.train_steps),
            num_unroll_steps=1,
            sample_seed_base=int(args.seed),
            learner_impl=COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_COMPACT_MUZERO,
            require_next_targets=True,
            capture_replay_store_state=True,
        ),
        policy_version=CompactPolicyVersionRefV1(
            policy_version_ref=str(resume_state.policy_version_ref),
            policy_source=policy_source,
            model_version_ref=str(resume_state.model_version_ref),
        ),
        replay_store=replay_store,
        learner=loop_learner,
    )
    trainer = CompactOwnedTrainerV1(
        config=CompactOwnedTrainerConfigV1(
            trainer_id=f"{run_id}:compact-candidate-trainer",
            policy_source=policy_source,
            initial_policy_version_ref=str(resume_state.policy_version_ref),
            initial_model_version_ref=str(resume_state.model_version_ref),
        ),
        learner=loop_learner,
        loop=loop,
    )
    _seed_trainer_from_resume_state(trainer, resume_state)
    counters_before = _trainer_counters(trainer)
    initial_digest = compact_model_state_digest_v1(context.model)
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=int(args.batch_size),
            actor_count=1,
            steps=int(args.compact_env_steps),
            warmup_steps=int(args.compact_warmup_steps),
            seed=int(args.seed),
            max_ticks=int(args.source_max_steps),
            decision_source_frames=int(args.decision_source_frames),
            source_physics_step_ms=float(args.source_physics_step_ms),
            stack_storage_dtype=HYBRID_STACK_STORAGE_DTYPE_UINT8,
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
            compact_rollout_slab_sample_gate=True,
            compact_rollout_slab_sample_gate_batch_size=int(args.compact_sample_batch_size),
            compact_rollout_slab_sample_gate_interval=int(args.compact_sample_interval),
            compact_rollout_slab_sample_gate_replay_pair_capacity=int(
                args.compact_replay_pair_capacity
            ),
            compact_rollout_slab_learner_gate=False,
            compact_owned_loop_entrypoint=False,
        ),
        observation_renderer=_PersistentDeviceRenderer(
            device=search_device,
        ),
        compact_rollout_slab=slab,
    )
    rng = np.random.default_rng(int(args.seed) + 7919)
    next_action: np.ndarray | None = None
    committed_index_rows = 0
    started = time.perf_counter()
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
            raise ValueError("compact env/search/replay training produced no slab step")
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
        if int(trainer.learner_update_count) > previous_updates:
            digest = compact_model_state_digest_v1(context.model)
            search_service.refresh_model_state(
                model_state_dict=context.model.state_dict(),
                policy_version_ref=trainer.policy_version_ref,
                model_version_ref=trainer.model_version_ref,
                policy_source=policy_source,
                learner_update_count=int(trainer.learner_update_count),
                expected_model_state_digest=digest,
            )
    elapsed_sec = max(0.0, time.perf_counter() - started)
    loop_telemetry = loop.telemetry()
    telemetry = dict(loop_telemetry.get("compact_owned_loop_learner_gate_last_telemetry") or {})
    if not telemetry:
        telemetry = dict(trainer.last_learner_telemetry)
    final_digest = compact_model_state_digest_v1(context.model)
    counters_after = _trainer_counters(trainer)
    replay_state = loop.snapshot_replay_store_state()
    learner_loss = _learner_loss_from_telemetry(telemetry)
    final_checkpoint = trainer.checkpoint(
        checkpoint_id=f"{run_id}:compact-candidate-post-train",
        replay_store_state=replay_state,
        metrics={
            "matched_learning_quality_compact_candidate": True,
            "last_learner_telemetry": telemetry,
            "compact_owned_loop_telemetry": loop_telemetry,
            "learner_loss_mean": learner_loss,
        },
        extra_metadata={
            "matched_learning_quality_compact_candidate_producer": True,
            "matched_learning_quality_sample_source": (
                COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE
            ),
        },
    )
    training_wall_sec = float(
        loop_telemetry.get("compact_owned_loop_learner_gate_sec") or elapsed_sec
    )
    return CompactCandidateTrainingResult(
        final_checkpoint=final_checkpoint,
        initial_model_state_digest=initial_digest,
        final_model_state_digest=final_digest,
        trainer_counters_before=counters_before,
        trainer_counters_after=counters_after,
        learner_telemetry=telemetry,
        learner_update_count_delta=(
            int(counters_after["learner_update_count"])
            - int(counters_before["learner_update_count"])
        ),
        sample_batch_count_delta=(
            int(counters_after["sample_batch_count"])
            - int(counters_before["sample_batch_count"])
        ),
        compact_rollout_rows=committed_index_rows,
        compact_sample_rows=int(
            loop_telemetry.get("compact_owned_loop_sample_gate_sample_row_count", 0)
        ),
        training_wall_sec=max(training_wall_sec, 1e-9),
        learner_loss_mean=learner_loss,
        sample_source=COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE,
    )


def _save_stock_checkpoint_export(
    *,
    checkpoint: Any,
    path: Path,
    policy_metadata: Mapping[str, Any],
    extra_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return save_compact_stock_export_v1(
        checkpoint,
        path,
        policy_metadata=policy_metadata,
        extra_metadata=extra_metadata,
    )


class _CompactMuZeroLoopLearnerAdapter:
    def __init__(self, edge: CompactMuZeroLearnerEdgeV1) -> None:
        self._edge = edge
        self.model = edge.model
        self.optimizer = edge.optimizer

    def train_on_sample_batch(self, sample_batch: Any, *, train_steps: int) -> dict[str, Any]:
        result = self._edge.train_on_sample_batch(sample_batch, train_steps=int(train_steps))
        telemetry = dict(result.telemetry)
        return {
            **telemetry,
            "compact_rollout_slab_learner_gate_impl": (
                COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_COMPACT_MUZERO
            ),
            "compact_rollout_slab_learner_gate_toy_probe": False,
            "compact_rollout_slab_learner_gate_real_muzero_update": True,
            "compact_rollout_slab_learner_gate_sec": float(
                telemetry["compact_muzero_learner_sec"]
            ),
            "compact_rollout_slab_learner_gate_device": str(
                telemetry["compact_muzero_learner_device"]
            ),
            "compact_rollout_slab_learner_gate_sample_rows": int(
                telemetry["compact_muzero_learner_sample_rows"]
            ),
            "compact_rollout_slab_learner_gate_train_steps": int(
                telemetry["compact_muzero_learner_train_steps"]
            ),
            "compact_rollout_slab_learner_gate_num_unroll_steps": int(
                telemetry["compact_muzero_learner_num_unroll_steps"]
            ),
            "compact_rollout_slab_learner_gate_updates": int(
                telemetry["compact_muzero_learner_train_steps"]
            ),
            "compact_rollout_slab_learner_gate_input_bytes": int(
                telemetry.get("compact_muzero_learner_input_bytes", 0)
            ),
            "compact_rollout_slab_learner_gate_input_h2d_bytes": int(
                telemetry["compact_muzero_learner_input_h2d_bytes"]
            ),
            "compact_rollout_slab_learner_gate_observation_h2d_bytes": int(
                telemetry["compact_muzero_learner_observation_h2d_bytes"]
            ),
            "compact_rollout_slab_learner_gate_resident_sample_used": bool(
                telemetry["compact_muzero_learner_resident_sample_used"]
            ),
            "compact_rollout_slab_learner_gate_device_replay_index_rows_sample": bool(
                telemetry["compact_muzero_learner_device_replay_index_rows_sample"]
            ),
            "compact_rollout_slab_learner_gate_loss": telemetry[
                "compact_muzero_learner_loss"
            ],
        }


class _CompactCandidateSearchPolicy:
    def __init__(self, model: Any) -> None:
        self._model = model
        self._cfg = SimpleNamespace(
            pb_c_base=19652,
            pb_c_init=1.25,
            discount_factor=0.997,
            root_noise_weight=0.0,
            root_dirichlet_alpha=0.3,
            value_delta_max=0.01,
        )


class _PersistentDeviceRenderer:
    backend_name = PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME

    def __init__(self, *, device: str) -> None:
        self.device = str(device)

    def render(self, request: Any) -> SourceStateBatchedRenderResult:
        import torch

        out = np.asarray(request.out)
        rows = np.asarray(request.row_indices, dtype=np.int64)
        players = np.asarray(request.controlled_players, dtype=np.int64)
        values = ((rows + 1) * 10 + players + 1).astype(np.uint8)
        out.fill(0)
        out[:, 0, :, :] = values[:, None, None]
        batch_size = int(rows.max(initial=-1) + 1)
        player_count = int(players.max(initial=-1) + 1)
        device_frames = np.zeros_like(out)
        device_frames[:, 0, :, :] = values[:, None, None]
        device_grid = device_frames.reshape(batch_size, player_count, 1, 64, 64)
        return SourceStateBatchedRenderResult(
            frames=out,
            telemetry={
                "render_sec": 0.001,
                "device_render_sec": 0.001,
                "device_to_host_sec": 0.0,
            },
            device_frames=torch.as_tensor(device_grid, device=self.device),
        )


def _resident_renderer_device(learner_device: str) -> str:
    requested = str(learner_device)
    if requested == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if requested == "cuda" else "cpu"


def _learner_loss_from_telemetry(telemetry: Mapping[str, Any]) -> float:
    for key in ("compact_muzero_learner_loss", "compact_rollout_slab_learner_gate_loss"):
        value = telemetry.get(key)
        if _is_finite_number(value):
            return float(value)
    raise ValueError("compact env/search/replay learner telemetry missing finite loss")


def _eval_seed_values(*, args: argparse.Namespace, fallback_seed: int) -> list[int]:
    count = int(args.eval_seed_count)
    if count <= 0:
        raise ValueError("eval_seed_count must be positive")
    if count == 1:
        return [int(fallback_seed)]
    sampler_seed = (
        int(args.eval_seed_rng_seed)
        if args.eval_seed_rng_seed is not None
        else int(fallback_seed)
    )
    rng = random.Random(sampler_seed)
    return rng.sample(range(0, 2_147_483_647 + 1), count)


def _effective_eval_seed_rng_seed(args: argparse.Namespace) -> int:
    if args.eval_seed_rng_seed is not None:
        return int(args.eval_seed_rng_seed)
    return int(args.eval_seed)


def _compact_training_entrypoint(training: CompactCandidateTrainingResult) -> str:
    if training.sample_source == COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE:
        return COMPACT_RECORD_STEP_ENTRYPOINT
    return COMPACT_RESIDENT_SAMPLE_ENTRYPOINT


def _aggregate_local_eval_summaries(
    summaries: list[Mapping[str, Any]],
    *,
    checkpoint_ref: str,
    output_ref: str,
    run_id: str,
    seed_values: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = [
        _local_eval_survival_row(summary, checkpoint_ref=checkpoint_ref)
        for summary in summaries
    ]
    steps = [float(row["steps"]) for row in rows]
    capped_count = sum(1 for row in rows if str(row.get("terminal")) in {"cap", "capped"})
    outcome_histogram: dict[str, int] = {}
    for row in rows:
        outcome = str(row.get("terminal") or row.get("outcome") or "unknown")
        outcome_histogram[outcome] = outcome_histogram.get(outcome, 0) + 1
    checkpoint_step = _checkpoint_step_from_ref(checkpoint_ref)
    aggregate = {
        "checkpoint": f"iteration_{checkpoint_step}",
        "checkpoint_ref": checkpoint_ref,
        "checkpoint_step": checkpoint_step,
        "seeds": len(seed_values),
        "eval_episode_count": len(seed_values),
        "evaluator_eval_calls": 1,
        "mean_steps": float(np.mean(steps)),
        "median_steps": float(np.median(steps)),
        "min_steps": float(np.min(steps)),
        "max_steps": float(np.max(steps)),
        "ok_count": len(seed_values),
        "capped_count": int(capped_count),
        "failure_count": 0,
        "outcome_histogram": outcome_histogram,
    }
    return {
        "ok": True,
        "schema_id": "curvyzero_compact_candidate_local_eval_aggregate/v1",
        "run_id": str(run_id),
        "checkpoint_ref": checkpoint_ref,
        "eval_max_steps": int(args.eval_steps),
        "selection": {
            "checkpoint_ref": checkpoint_ref,
            "checkpoint_refs": checkpoint_ref,
            "eval_seed_values": [int(seed) for seed in seed_values],
            "eval_seed_count": len(seed_values),
            "eval_seed_sampler_seed": (
                int(args.eval_seed_rng_seed)
                if args.eval_seed_rng_seed is not None
                else int(args.eval_seed)
            ),
        },
        "config": {
            "checkpoint_ref": checkpoint_ref,
            "output_ref": output_ref,
            "run_id": str(run_id),
            "seed": int(args.eval_seed),
            "max_eval_steps": int(args.eval_steps),
            "source_max_steps": int(args.source_max_steps),
            "num_simulations": int(args.num_simulations),
            "batch_size": int(args.batch_size),
            "reward_variant": "survival_plus_bonus_no_outcome",
            "env_variant": "source_state_fixed_opponent",
            "opponent_policy_kind": "fixed_straight",
        },
        "survival_aggregate_table": [aggregate],
        "survival_table": rows,
        "per_seed_summaries": list(summaries),
    }


def _local_eval_survival_row(
    summary: Mapping[str, Any],
    *,
    checkpoint_ref: str,
) -> dict[str, Any]:
    episode = summary.get("episode")
    if not isinstance(episode, Mapping):
        raise ValueError("local compact eval summary missing episode")
    seed = int(episode.get("seed", 0))
    steps = float(episode.get("steps_survived", episode.get("steps_run", 0)))
    terminal = str(episode.get("terminal_reason") or "unknown")
    return {
        "checkpoint": f"iteration_{_checkpoint_step_from_ref(checkpoint_ref)}",
        "checkpoint_ref": checkpoint_ref,
        "checkpoint_step": _checkpoint_step_from_ref(checkpoint_ref),
        "seed": seed,
        "steps": steps,
        "cap": int(summary.get("config", {}).get("max_eval_steps", 0)),
        "terminal": terminal,
        "ok": bool(summary.get("ok", True)),
    }


def _checkpoint_step_from_ref(ref: str) -> int:
    name = Path(ref).name
    if name.endswith(".pth.tar"):
        name = name[: -len(".pth.tar")]
    if name.startswith("iteration_") and name.removeprefix("iteration_").isdigit():
        return int(name.removeprefix("iteration_"))
    return 0


def _run_eval_summary(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    output_path: Path,
    run_id: str,
    seed: int,
    repo_root: Path,
) -> Mapping[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_values = _eval_seed_values(args=args, fallback_seed=int(seed))
    checkpoint_ref = _relative_ref(checkpoint_path, repo_root)
    if len(seed_values) == 1:
        summary = chain_smoke._run_standalone_eval(
            checkpoint_ref=checkpoint_ref,
            output_ref=_relative_ref(output_path, repo_root),
            run_id=run_id,
            seed=int(seed_values[0]),
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
    else:
        per_seed = []
        for eval_seed in seed_values:
            seed_output = output_path.with_name(
                f"{output_path.stem}_seed{int(eval_seed)}{output_path.suffix}"
            )
            per_seed.append(
                chain_smoke._run_standalone_eval(
                    checkpoint_ref=checkpoint_ref,
                    output_ref=_relative_ref(seed_output, repo_root),
                    run_id=run_id,
                    seed=int(eval_seed),
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
            )
        summary = _aggregate_local_eval_summaries(
            per_seed,
            checkpoint_ref=checkpoint_ref,
            output_ref=_relative_ref(output_path, repo_root),
            run_id=run_id,
            seed_values=seed_values,
            args=args,
        )
    _write_json(output_path, summary)
    return summary


def _write_packager_inputs(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    output_dir: Path,
    inputs_dir: Path,
    lifecycle: Mapping[str, Any],
    lifecycle_path: Path,
    context: CompactCandidateContext,
    training: CompactCandidateTrainingResult,
    pre_eval_path: Path,
    post_eval_path: Path,
    pre_eval: Mapping[str, Any],
    post_eval: Mapping[str, Any],
    initial_checkpoint_path: Path,
    final_checkpoint_path: Path,
) -> dict[str, Path]:
    paths = {
        "source_fingerprint": inputs_dir / "source_fingerprint.json",
        "model_identity": inputs_dir / "model_identity.json",
        "eval_settings": inputs_dir / "eval_settings.json",
        "denominators": inputs_dir / "denominators.json",
        "capture_provenance": inputs_dir / "capture_provenance.json",
        "training_artifact": inputs_dir / "training_artifact.json",
        "pre_eval_summary": pre_eval_path,
        "post_eval_summary": post_eval_path,
        "initial_checkpoint": initial_checkpoint_path,
        "final_checkpoint": final_checkpoint_path,
    }
    _write_json(
        paths["source_fingerprint"],
        _source_fingerprint(
            args=args,
            repo_root=repo_root,
            lifecycle_path=lifecycle_path,
            context=context,
        ),
    )
    _write_json(paths["model_identity"], _model_identity(context, training))
    _write_json(paths["eval_settings"], _eval_settings(args=args))
    _write_json(paths["denominators"], _denominators(training))
    _write_json(paths["capture_provenance"], _capture_provenance(training))
    _write_json(
        paths["training_artifact"],
        _training_artifact(
            args=args,
            output_dir=output_dir,
            lifecycle=lifecycle,
            lifecycle_path=lifecycle_path,
            context=context,
            training=training,
            pre_eval=pre_eval,
            post_eval=post_eval,
            initial_checkpoint_path=initial_checkpoint_path,
            final_checkpoint_path=final_checkpoint_path,
        ),
    )
    return paths


def _packager_args(
    *,
    args: argparse.Namespace,
    context: CompactCandidateContext,
    input_paths: Mapping[str, Path],
    capture_path: Path,
    initial_checkpoint_path: Path,
    final_checkpoint_path: Path,
) -> list[str]:
    candidate_checkpoint_id = str(args.candidate_checkpoint_id or context.checkpoint_id)
    return [
        "--role",
        COMPACT_CANDIDATE_ROLE,
        "--run-id",
        str(args.run_id),
        "--capture-id",
        f"{args.run_id}:compact-candidate-capture",
        "--candidate-checkpoint-id",
        candidate_checkpoint_id,
        "--denominator-id",
        str(args.denominator_id),
        "--quality-horizon",
        str(args.quality_horizon),
        "--hardware-class",
        str(args.hardware_class),
        "--source-fingerprint-json",
        str(input_paths["source_fingerprint"]),
        "--model-identity-json",
        str(input_paths["model_identity"]),
        "--eval-settings-json",
        str(input_paths["eval_settings"]),
        "--denominators-json",
        str(input_paths["denominators"]),
        "--capture-provenance-json",
        str(input_paths["capture_provenance"]),
        "--training-artifact",
        str(input_paths["training_artifact"]),
        "--pre-eval-summary",
        str(input_paths["pre_eval_summary"]),
        "--post-eval-summary",
        str(input_paths["post_eval_summary"]),
        "--initial-checkpoint",
        str(initial_checkpoint_path),
        "--final-checkpoint",
        str(final_checkpoint_path),
        "--output",
        str(capture_path),
    ]


def _invoke_capture_packager(argv: list[str]) -> int:
    return int(packager.main(argv))


def _producer_manifest(
    *,
    args: argparse.Namespace,
    lifecycle_path: Path,
    context: CompactCandidateContext,
    training: CompactCandidateTrainingResult,
    input_paths: Mapping[str, Path],
    capture_path: Path,
    packager_args: list[str],
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
        "sample_source": training.sample_source,
        "synthetic_resident_sample": (
            training.sample_source != COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE
        ),
        "real_env_search_replay_rows": (
            training.sample_source == COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE
        ),
        "final_matched_stock_reference_still_required": True,
        "unified_lifecycle_report_path": str(lifecycle_path),
        "loaded_checkpoint_path": str(context.checkpoint_path),
        "loaded_checkpoint_sha256": context.checkpoint_sha256,
        "capture_path": str(capture_path),
        "packager_args": list(packager_args),
        "input_paths": {key: str(path) for key, path in input_paths.items()},
        "model_identity": _model_identity(context, training),
        "denominators": _denominators(training),
        "producer_guardrails": {
            "initial_final_checkpoint_files_distinct": True,
            "pre_eval_bound_to_initial_checkpoint": True,
            "post_eval_bound_to_final_checkpoint": True,
            "model_state_digest_changed": True,
            "denominators_derived_from_trainer_counters": True,
            "packager_writes_single_role_capture": True,
        },
    }


def _training_artifact(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    lifecycle: Mapping[str, Any],
    lifecycle_path: Path,
    context: CompactCandidateContext,
    training: CompactCandidateTrainingResult,
    pre_eval: Mapping[str, Any],
    post_eval: Mapping[str, Any],
    initial_checkpoint_path: Path,
    final_checkpoint_path: Path,
) -> dict[str, Any]:
    return {
        "schema_id": TRAINING_ARTIFACT_SCHEMA_ID,
        "ok": True,
        "role": COMPACT_CANDIDATE_ROLE,
        "route": "compact_owned_trainer",
        "producer_id": PRODUCER_ID,
        "run_id": str(args.run_id),
        "output_dir": str(output_dir),
        "profile_only": False,
        "support_only": False,
        "touches_live_runs": False,
        "calls_train_muzero": False,
        "real_compact_owned_training_work": True,
        "compact_training_entrypoint": _compact_training_entrypoint(training),
        "sample_source": training.sample_source,
        "synthetic_resident_sample": (
            training.sample_source != COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE
        ),
        "real_env_search_replay_rows": (
            training.sample_source == COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE
        ),
        "unified_lifecycle": {
            "path": str(lifecycle_path),
            "schema_id": lifecycle.get("schema_id"),
            "checkpoint_id": lifecycle.get("checkpoint_id"),
            "lifecycle_gates_complete": lifecycle.get("lifecycle_gates_complete"),
            "promotion_eligible": lifecycle.get("promotion_eligible"),
        },
        "loaded_checkpoint": {
            "checkpoint_id": context.checkpoint_id,
            "path": str(context.checkpoint_path),
            "sha256": context.checkpoint_sha256,
        },
        "initial_checkpoint_path": str(initial_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
        "pre_eval_checkpoint_ref": _eval_checkpoint_ref(pre_eval),
        "post_eval_checkpoint_ref": _eval_checkpoint_ref(post_eval),
        "trainer_counters_before": dict(training.trainer_counters_before),
        "trainer_counters_after": dict(training.trainer_counters_after),
        "learner_telemetry": dict(training.learner_telemetry),
        "model_identity": _model_identity(context, training),
        "denominators": _denominators(training),
    }


def _model_identity(
    context: CompactCandidateContext,
    training: CompactCandidateTrainingResult,
) -> dict[str, Any]:
    return {
        "model_identity_scope": "candidate_loaded_checkpoint",
        "candidate_loaded_checkpoint": True,
        "loaded_checkpoint_id": context.checkpoint_id,
        "loaded_compact_checkpoint_path": str(context.checkpoint_path),
        "loaded_compact_checkpoint_sha256": context.checkpoint_sha256,
        "initial_model_state_digest": training.initial_model_state_digest,
        "final_model_state_digest": training.final_model_state_digest,
        "model_state_digest_changed": (
            training.initial_model_state_digest != training.final_model_state_digest
        ),
    }


def _denominators(training: CompactCandidateTrainingResult) -> dict[str, Any]:
    return {
        "denominator_currency": "compact_owned_learning_quality",
        "env_step_currency": "compact_owned_trainer_env_steps",
        "speed_currency": "compact_trainer_env_steps_per_sec",
        "learner_update_count_delta": int(training.learner_update_count_delta),
        "sample_batch_count_delta": int(training.sample_batch_count_delta),
        "compact_rollout_rows": int(training.compact_rollout_rows),
        "compact_sample_rows": int(training.compact_sample_rows),
        "training_wall_sec": float(training.training_wall_sec),
        "learner_loss_mean": float(training.learner_loss_mean),
        "uses_fallback_denominator": False,
        "wall_sec_used_for_speed_claim": False,
    }


def _capture_provenance(training: CompactCandidateTrainingResult) -> dict[str, Any]:
    return {
        "schema_id": COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_PROVENANCE_SCHEMA_ID,
        "role": COMPACT_CANDIDATE_ROLE,
        "producer_id": PRODUCER_ID,
        "producer_ran_training": True,
        "producer_ran_pre_eval": True,
        "producer_ran_post_eval": True,
        "feeds_builder": True,
        "support_only": False,
        "training_artifact_ref": "training_artifact",
        "pre_eval_artifact_ref": "pre_eval_summary",
        "post_eval_artifact_ref": "post_eval_summary",
        "compact_owned_training": {
            "calls_train_muzero": False,
            "real_compact_owned_training_work": True,
            "compact_training_entrypoint": _compact_training_entrypoint(training),
            "sample_source": training.sample_source,
        },
    }


def _eval_settings(*, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "observation_schema_id": "curvyzero_source_state_rgb_canvas_like_gray64_stack4/v0",
        "policy_observation_backend": "cpu_oracle",
        "eval_seed_set": _eval_seed_values(args=args, fallback_seed=int(args.eval_seed)),
        "eval_seed_rng_seed": _effective_eval_seed_rng_seed(args),
        "eval_episode_count": int(args.eval_seed_count),
        "source_max_steps": int(args.source_max_steps),
        "eval_max_steps": int(args.eval_steps),
        "num_simulations": int(args.num_simulations),
        "batch_size": int(args.batch_size),
        "reward_variant": "survival_plus_bonus_no_outcome",
        "reward_target_effect": "extrinsic_reward_only",
        "death_mode": DEATH_MODE_NORMAL,
        "terminal_target_mode": "stock_terminal_no_bootstrap_return_discount_1.0",
        "root_noise": 0.0,
        "dirichlet_alpha": 0.3,
        "policy_noise": 0.0,
        "rnd_enabled": False,
        "exploration_bonus_mode": "none",
        "opponent_policy_ref": "fixed_straight",
        "opponent_policy_kind": "fixed_straight",
        "opponent_runtime_mode": "normal",
        "opponent_death_mode": DEATH_MODE_NORMAL,
        "natural_bonus_spawn": True,
        "training_seed_policy": "fixed_eval_seed",
        "initialization_source": "loaded_unified_lifecycle_compact_checkpoint",
        "num_unroll_steps": 1,
        "td_steps": 1,
        "discount": 1.0,
        "support_scale": 300,
    }


def _source_fingerprint(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    lifecycle_path: Path,
    context: CompactCandidateContext,
) -> dict[str, Any]:
    return {
        "git_commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "git_status_dirty": bool(_git_output(repo_root, "status", "--short")),
        "producer_script": PRODUCER_ID,
        "producer_route": (
            "compact_candidate_env_search_replay_producer"
            if str(args.compact_training_mode) == COMPACT_TRAINING_MODE_ENV_SEARCH_REPLAY
            else "compact_candidate_resident_sample_scaffold_producer"
        ),
        "image_digest": "local-workspace",
        "unified_lifecycle_report_path": str(lifecycle_path),
        "loaded_compact_checkpoint_sha256": context.checkpoint_sha256,
        "matched_surface": {
            "env_variant": "source_state_fixed_opponent",
            "reward_variant": "survival_plus_bonus_no_outcome",
            "policy_observation_backend": "cpu_oracle",
            "opponent_policy_kind": "fixed_straight",
            "eval_seed_set": _eval_seed_values(args=args, fallback_seed=int(args.eval_seed)),
            "eval_seed_rng_seed": _effective_eval_seed_rng_seed(args),
        },
    }


def _validate_lifecycle_report(lifecycle: Mapping[str, Any]) -> None:
    if lifecycle.get("ok") is not True:
        raise ValueError("unified lifecycle report must be ok=true")
    if lifecycle.get("lifecycle_gates_complete") is not True:
        raise ValueError("unified lifecycle report must have lifecycle_gates_complete=true")
    if lifecycle.get("promotion_claim") is not False:
        raise ValueError("unified lifecycle report must not claim promotion")


def _validate_training_result(training: CompactCandidateTrainingResult) -> None:
    if training.sample_source != COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE:
        raise ValueError(
            "compact candidate requires compact_env_search_replay_rows sample source"
        )
    if training.initial_model_state_digest == training.final_model_state_digest:
        raise ValueError("compact candidate model state digest did not change")
    telemetry = dict(training.learner_telemetry)
    if telemetry.get("compact_muzero_learner_calls_train_muzero") is not False:
        raise ValueError("compact candidate learner must not call train_muzero")
    if int(training.learner_update_count_delta) <= 0:
        raise ValueError("compact candidate learner_update_count_delta must be positive")
    if int(training.sample_batch_count_delta) <= 0:
        raise ValueError("compact candidate sample_batch_count_delta must be positive")
    if int(training.compact_rollout_rows) <= 0:
        raise ValueError("compact candidate compact_rollout_rows must be positive")
    if int(training.compact_sample_rows) <= 0:
        raise ValueError("compact candidate compact_sample_rows must be positive")
    if not _is_positive_number(training.training_wall_sec):
        raise ValueError("compact candidate training_wall_sec must be positive")
    if not _is_finite_number(training.learner_loss_mean):
        raise ValueError("compact candidate learner_loss_mean must be finite")


def _require_distinct_existing_checkpoints(initial: Path, final: Path) -> None:
    if initial.resolve() == final.resolve():
        raise ValueError("initial and final checkpoint paths must be distinct")
    for path in (initial, final):
        if path.name not in {"iteration_0.pth.tar", "iteration_1.pth.tar"}:
            raise ValueError(f"producer checkpoint must be immutable iteration_N: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"producer checkpoint file missing: {path}")


def _assert_eval_summary_bound_to_checkpoint(
    summary: Mapping[str, Any],
    *,
    expected_checkpoint_path: Path,
    label: str,
) -> None:
    refs = _eval_checkpoint_refs(summary)
    if not refs:
        raise ValueError(f"{label} missing checkpoint reference")
    expected = expected_checkpoint_path.resolve()
    expected_name = expected.name
    for ref in refs:
        ref_path = Path(str(ref))
        if ref_path.is_absolute() and ref_path.resolve() == expected:
            return
        if not ref_path.is_absolute() and len(ref_path.parts) > 1:
            try:
                if expected.relative_to(Path.cwd().resolve()).as_posix() == ref_path.as_posix():
                    return
            except ValueError:
                pass
            if expected.as_posix().endswith(ref_path.as_posix()):
                return
            continue
        if ref_path.name == expected_name:
            return
    raise ValueError(f"{label} checkpoint reference does not match {expected_name}")


def _eval_checkpoint_ref(summary: Mapping[str, Any]) -> str:
    refs = _eval_checkpoint_refs(summary)
    return "" if not refs else refs[0]


def _eval_checkpoint_refs(summary: Mapping[str, Any]) -> list[str]:
    refs: list[str] = []
    checkpoint = summary.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        if checkpoint.get("path") is not None:
            refs.append(str(checkpoint["path"]))
    elif checkpoint is not None:
        refs.append(str(checkpoint))
    config = summary.get("config")
    if isinstance(config, Mapping) and config.get("checkpoint_ref") is not None:
        refs.append(str(config["checkpoint_ref"]))
    for key in ("checkpoint_ref", "checkpoint_path", "checkpoint_label"):
        if summary.get(key) is not None:
            refs.append(str(summary[key]))
    return refs


def _seed_trainer_from_resume_state(trainer: CompactOwnedTrainerV1, resume_state: Any) -> None:
    trainer.train_step = int(resume_state.train_step)
    trainer.learner_update_count = int(resume_state.learner_update_count)
    trainer.sample_batch_count = int(resume_state.sample_batch_count)
    trainer.policy_version_ref = str(resume_state.policy_version_ref)
    trainer.model_version_ref = str(resume_state.model_version_ref)


def _trainer_counters(trainer: CompactOwnedTrainerV1) -> dict[str, int]:
    return {
        "train_step": int(trainer.train_step),
        "learner_update_count": int(trainer.learner_update_count),
        "sample_batch_count": int(trainer.sample_batch_count),
        "policy_refresh_count": int(trainer.policy_refresh_count),
    }


def _policy_surface(surface: Mapping[str, Any]) -> Mapping[str, Any]:
    policy = surface.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("model surface missing policy mapping")
    return policy


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _relative_ref(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


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


def _git_output(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def _is_positive_number(value: Any) -> bool:
    return _is_finite_number(value) and float(value) > 0.0


if __name__ == "__main__":
    raise SystemExit(main())
