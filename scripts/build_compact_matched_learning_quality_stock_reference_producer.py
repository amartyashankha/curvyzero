#!/usr/bin/env python3
"""Produce a stock-reference matched learning-quality capture from Modal evidence."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from curvyzero.contracts.curvytron import curvytron_train_app_name
from curvyzero.contracts.curvytron import modal_volume_kwargs_for_name
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_PROVENANCE_SCHEMA_ID,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import STOCK_REFERENCE_ROLE


def _load_sibling_script(module_name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


packager = _load_sibling_script(
    "build_compact_matched_learning_quality_capture_from_artifacts_for_stock_producer",
    "build_compact_matched_learning_quality_capture_from_artifacts.py",
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/local/curvytron_compact_matched_learning_quality_results"
)
DEFAULT_RUN_ID = "optimizer-stock-reference-quality-producer-smoke-20260530"
DEFAULT_ATTEMPT_ID = "stock-reference-quality"
PRODUCER_SCHEMA_ID = "curvyzero_compact_matched_learning_quality_stock_reference_producer/v1"
TRAINING_ARTIFACT_SCHEMA_ID = (
    "curvyzero_compact_matched_learning_quality_stock_reference_training_artifact/v1"
)
PRODUCER_ID = "scripts/build_compact_matched_learning_quality_stock_reference_producer.py"
DEFAULT_TRAIN_APP_NAME = curvytron_train_app_name()
DEFAULT_STATUS_APP_NAME = "curvyzero-lightzero-curvytron-run-status"
ITERATION_CHECKPOINT_RE = re.compile(r"\Aiteration_(\d+)\.pth\.tar\Z")
MUTABLE_CHECKPOINT_NAMES = frozenset({"latest.pth.tar", "ckpt_best.pth.tar"})


@dataclass(frozen=True, slots=True)
class StockCheckpointSelection:
    initial_ref: str
    initial_iteration: int
    final_ref: str
    final_iteration: int
    ignored_mutable_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StockCheckpointFiles:
    initial_path: Path
    final_path: Path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_supported_stock_producer_args(args)
    repo_root = Path.cwd()
    output_dir = (repo_root / args.output_root / args.run_id).resolve()
    _prepare_fresh_output_dir(output_dir)

    train_result = _run_stock_train(args)
    _validate_stock_train_result(train_result, args=args)
    train_summary = _load_stock_train_summary(args, train_result=train_result)
    status = _load_stock_status(args, train_result=train_result)
    selection = _select_stock_checkpoints(train_result)
    checkpoint_files = _download_stock_checkpoints(
        args=args,
        output_dir=output_dir,
        selection=selection,
    )
    model_identity = _model_identity(checkpoint_files)
    _validate_model_identity(model_identity)

    pre_eval_raw = _run_stock_eval(
        args=args,
        checkpoint_ref=selection.initial_ref,
        checkpoint_iteration=selection.initial_iteration,
        point_id="pre_train",
    )
    post_eval_raw = _run_stock_eval(
        args=args,
        checkpoint_ref=selection.final_ref,
        checkpoint_iteration=selection.final_iteration,
        point_id="post_train",
    )
    _assert_eval_summary_bound_to_checkpoint(
        pre_eval_raw,
        expected_checkpoint_ref=selection.initial_ref,
        expected_iteration=selection.initial_iteration,
        label="pre eval summary",
    )
    _assert_eval_summary_bound_to_checkpoint(
        post_eval_raw,
        expected_checkpoint_ref=selection.final_ref,
        expected_iteration=selection.final_iteration,
        label="post eval summary",
    )
    eval_seed_set = _validate_and_extract_matched_eval_seed_set(
        pre_eval_raw,
        post_eval_raw,
        args=args,
    )
    _assert_eval_summaries_match_requested_horizon(
        pre_eval_raw,
        post_eval_raw,
        args=args,
    )
    _assert_stock_quality_eval_not_saturated(pre_eval_raw, post_eval_raw)
    pre_eval = _normalized_eval_summary(
        pre_eval_raw,
        checkpoint_ref=selection.initial_ref,
        checkpoint_iteration=selection.initial_iteration,
        eval_episode_count=len(eval_seed_set),
    )
    post_eval = _normalized_eval_summary(
        post_eval_raw,
        checkpoint_ref=selection.final_ref,
        checkpoint_iteration=selection.final_iteration,
        eval_episode_count=len(eval_seed_set),
    )

    inputs_dir = output_dir / "capture_inputs"
    input_paths = _write_packager_inputs(
        args=args,
        repo_root=repo_root,
        output_dir=output_dir,
        inputs_dir=inputs_dir,
        train_result=train_result,
        train_summary=train_summary,
        status=status,
        selection=selection,
        checkpoint_files=checkpoint_files,
        model_identity=model_identity,
        pre_eval=pre_eval,
        post_eval=post_eval,
        eval_seed_set=eval_seed_set,
    )
    capture_path = output_dir / "stock_reference_capture.json"
    packager_args = _packager_args(
        args=args,
        input_paths=input_paths,
        capture_path=capture_path,
        checkpoint_files=checkpoint_files,
    )
    packager_return = _invoke_capture_packager(packager_args)
    if int(packager_return) != 0:
        raise RuntimeError(f"stock reference capture packager failed: {packager_return}")

    manifest_path = output_dir / "stock_reference_producer_manifest.json"
    manifest = _producer_manifest(
        args=args,
        train_result=train_result,
        train_summary=train_summary,
        status=status,
        selection=selection,
        checkpoint_files=checkpoint_files,
        input_paths=input_paths,
        capture_path=capture_path,
        packager_args=packager_args,
    )
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "ok": True,
                "role": STOCK_REFERENCE_ROLE,
                "run_id": str(args.run_id),
                "capture_path": str(capture_path),
                "manifest_path": str(manifest_path),
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
    parser.add_argument("--attempt-id", default=DEFAULT_ATTEMPT_ID)
    parser.add_argument("--candidate-checkpoint-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--compute",
        choices=("gpu-l4-t4-cpu40", "gpu-h100-cpu40"),
        default="gpu-l4-t4-cpu40",
    )
    parser.add_argument("--eval-compute", default="cpu")
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--eval-seed", type=int, default=20260833)
    parser.add_argument("--eval-seed-count", type=int, default=2)
    parser.add_argument("--eval-seed-rng-seed", type=int)
    parser.add_argument("--max-train-iter", type=int, default=1)
    parser.add_argument("--max-env-step", type=int, default=512)
    parser.add_argument("--source-max-steps", type=int, default=1_048_576)
    parser.add_argument("--eval-steps", type=int, default=128)
    parser.add_argument("--step-detail-limit", type=int, default=4)
    parser.add_argument("--decision-ms", type=float, default=50.0 / 3.0)
    parser.add_argument("--decision-source-frames", type=int, default=1)
    parser.add_argument("--source-physics-step-ms", type=float, default=50.0 / 3.0)
    parser.add_argument("--num-simulations", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--collector-env-num", type=int, default=2)
    parser.add_argument("--evaluator-env-num", type=int, default=2)
    parser.add_argument("--n-episode", type=int, default=2)
    parser.add_argument("--n-evaluator-episode", type=int, default=2)
    parser.add_argument("--save-ckpt-after-iter", type=int, default=1)
    parser.add_argument("--env-variant", default="source_state_fixed_opponent")
    parser.add_argument("--reward-variant", default="survival_plus_bonus_no_outcome")
    parser.add_argument("--policy-observation-backend", default="cpu_oracle")
    parser.add_argument("--collect-search-backend", default="stock")
    parser.add_argument("--collect-search-ctree-backend", default="lightzero")
    parser.add_argument("--env-manager-type", default="base")
    parser.add_argument("--opponent-policy-kind", default="fixed_straight")
    parser.add_argument("--opponent-runtime-mode", default="normal")
    parser.add_argument("--opponent-death-mode", default="normal")
    parser.add_argument("--natural-bonus-spawn", dest="natural_bonus_spawn", action="store_true")
    parser.add_argument(
        "--no-natural-bonus-spawn",
        dest="natural_bonus_spawn",
        action="store_false",
    )
    parser.set_defaults(natural_bonus_spawn=True)
    parser.add_argument("--quality-horizon", default="stock_train_muzero_pre_post_eval")
    parser.add_argument("--hardware-class", default="modal-gpu-l4-t4-cpu40")
    parser.add_argument("--denominator-id", default="stock_reference_train_muzero_denominator_v1")
    parser.add_argument("--train-app-name", default=DEFAULT_TRAIN_APP_NAME)
    parser.add_argument("--status-app-name", default=DEFAULT_STATUS_APP_NAME)
    parser.add_argument("--modal-env", default=None)
    parser.add_argument("--runs-volume-name", default="curvyzero-runs-v2")
    parser.add_argument("--allow-auto-resume", action="store_true")
    return parser.parse_args(argv)


def _validate_supported_stock_producer_args(args: argparse.Namespace) -> None:
    if not bool(args.natural_bonus_spawn):
        raise ValueError(
            "stock train path currently supports natural_bonus_spawn=true only"
        )
    if int(args.decision_source_frames) != 1:
        raise ValueError(
            "stock train path currently supports decision_source_frames=1 only"
        )
    if abs(float(args.source_physics_step_ms) - (50.0 / 3.0)) > 1.0e-9:
        raise ValueError(
            "stock train path currently supports source_physics_step_ms=50/3 only"
        )
    if int(args.n_evaluator_episode) < int(args.evaluator_env_num):
        raise ValueError("n_evaluator_episode must be >= evaluator_env_num")
    if int(args.n_evaluator_episode) != int(args.evaluator_env_num):
        raise ValueError(
            "stock evaluator surface requires n_evaluator_episode == evaluator_env_num"
        )
    if int(args.eval_seed_count) >= 32 and int(args.evaluator_env_num) != int(
        args.eval_seed_count
    ):
        raise ValueError(
            "larger stock evaluator surface requires evaluator_env_num == eval_seed_count"
        )


def _run_stock_train(args: argparse.Namespace) -> Mapping[str, Any]:
    train_fn = _stock_train_function(args, compute=str(args.compute))
    return _require_mapping(
        train_fn.remote(**_stock_train_kwargs(args)),
        "stock train result",
    )


def _stock_train_function(args: argparse.Namespace, *, compute: str) -> Any:
    if compute == "gpu-l4-t4-cpu40":
        function_name = "lightzero_curvytron_visual_survival_gpu_cpu40"
    elif compute == "gpu-h100-cpu40":
        function_name = "lightzero_curvytron_visual_survival_h100_cpu40"
    else:
        raise ValueError(f"unsupported stock compute: {compute}")
    return _deployed_modal_function(
        app_name=str(args.train_app_name),
        function_name=function_name,
        modal_env=args.modal_env,
    )


def _deployed_modal_function(
    *,
    app_name: str,
    function_name: str,
    modal_env: str | None,
) -> Any:
    import modal

    return modal.Function.from_name(
        str(app_name),
        str(function_name),
        environment_name=modal_env,
    )


def _stock_train_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": "train",
        "seed": int(args.seed),
        "run_id": str(args.run_id),
        "attempt_id": str(args.attempt_id),
        "max_env_step": int(args.max_env_step),
        "max_train_iter": int(args.max_train_iter),
        "source_max_steps": int(args.source_max_steps),
        "decision_ms": float(args.decision_ms),
        "collector_env_num": int(args.collector_env_num),
        "evaluator_env_num": int(args.evaluator_env_num),
        "n_evaluator_episode": int(args.n_evaluator_episode),
        "n_episode": int(args.n_episode),
        "num_simulations": int(args.num_simulations),
        "batch_size": int(args.batch_size),
        "lightzero_eval_freq": 0,
        "skip_lightzero_eval_in_profile": False,
        "profile_cuda_sync_enabled": False,
        "profile_allow_auto_resume": False,
        "profile_volume_commit": False,
        "lightzero_multi_gpu": False,
        "save_ckpt_after_iter": int(args.save_ckpt_after_iter),
        "commit_on_checkpoint": True,
        "stop_after_learner_train_calls": 0,
        "env_variant": str(args.env_variant),
        "reward_variant": str(args.reward_variant),
        "policy_observation_backend": str(args.policy_observation_backend),
        "collect_search_backend": str(args.collect_search_backend),
        "collect_search_ctree_backend": str(args.collect_search_ctree_backend),
        "env_manager_type": str(args.env_manager_type),
        "opponent_policy_kind": str(args.opponent_policy_kind),
        "opponent_runtime_mode": str(args.opponent_runtime_mode),
        "opponent_death_mode": str(args.opponent_death_mode),
        "background_eval_enabled": False,
        "background_gif_enabled": False,
        "exploration_bonus_mode": "none",
        "exploration_bonus_weight": 0.0,
    }


def _validate_stock_train_result(
    train_result: Mapping[str, Any],
    *,
    args: argparse.Namespace,
) -> None:
    if train_result.get("ok") is not True:
        raise ValueError("stock train result must be ok=true")
    if train_result.get("mode") != "train":
        raise ValueError("stock train result mode must be train")
    if train_result.get("called_train_muzero") is not True:
        raise ValueError("stock train result must call train_muzero")
    if train_result.get("trainer_entrypoint") != "lzero.entry.train_muzero":
        raise ValueError("stock train result trainer_entrypoint must be lzero.entry.train_muzero")
    if train_result.get("run_id") != str(args.run_id):
        raise ValueError("stock train result run_id mismatch")
    if train_result.get("attempt_id") != str(args.attempt_id):
        raise ValueError("stock train result attempt_id mismatch")
    if train_result.get("compute") != str(args.compute):
        raise ValueError("stock train result compute mismatch")
    command = _required_mapping(train_result.get("command"), "stock train command")
    _assert_command_int(command, "seed", int(args.seed))
    _assert_command_int(command, "max_env_step", int(args.max_env_step))
    _assert_command_int(command, "max_train_iter", int(args.max_train_iter))
    _assert_command_int(command, "source_max_steps", int(args.source_max_steps))
    _assert_command_int(command, "collector_env_num", int(args.collector_env_num))
    _assert_command_int(command, "evaluator_env_num", int(args.evaluator_env_num))
    _assert_command_int(command, "n_episode", int(args.n_episode))
    _assert_command_int(command, "n_evaluator_episode", int(args.n_evaluator_episode))
    _assert_command_int(command, "num_simulations", int(args.num_simulations))
    _assert_command_int(command, "batch_size", int(args.batch_size))
    _assert_command_int(command, "save_ckpt_after_iter", int(args.save_ckpt_after_iter))
    _assert_command_float(command, "decision_ms", float(args.decision_ms))
    _assert_command_str(command, "reward_variant", str(args.reward_variant))
    _assert_command_str(
        command,
        "policy_observation_backend",
        str(args.policy_observation_backend),
    )
    _assert_command_str(command, "opponent_policy_kind", str(args.opponent_policy_kind))
    _assert_command_str(command, "opponent_runtime_mode", str(args.opponent_runtime_mode))
    _assert_command_str(command, "opponent_death_mode", str(args.opponent_death_mode))
    if command.get("env_variant") != str(args.env_variant):
        raise ValueError("stock train command env_variant mismatch")
    if command.get("env_variant") != "source_state_fixed_opponent":
        raise ValueError("stock train must use source_state_fixed_opponent")
    if command.get("env_manager_type") != "base":
        raise ValueError("stock train must use env_manager_type=base")
    if command.get("collect_search_backend") != "stock":
        raise ValueError("stock train must use collect_search_backend=stock")
    if command.get("collect_search_ctree_backend") != "lightzero":
        raise ValueError("stock train must use collect_search_ctree_backend=lightzero")
    if int(command.get("stop_after_learner_train_calls") or 0) != 0:
        raise ValueError("stock train stop_after_learner_train_calls must be 0")
    if command.get("background_eval_enabled") is not False:
        raise ValueError("stock train background eval must be disabled for matched producer")
    if int(command.get("decision_source_frames") or 0) != int(args.decision_source_frames):
        raise ValueError("stock train command decision_source_frames mismatch")
    if abs(
        float(command.get("source_physics_step_ms") or 0.0)
        - float(args.source_physics_step_ms)
    ) > 1.0e-9:
        raise ValueError("stock train command source_physics_step_ms mismatch")
    if command.get("natural_bonus_spawn") is not bool(args.natural_bonus_spawn):
        raise ValueError("stock train command natural_bonus_spawn mismatch")
    auto_resume = train_result.get("auto_resume")
    if (
        isinstance(auto_resume, Mapping)
        and auto_resume.get("found") is True
        and not bool(args.allow_auto_resume)
    ):
        raise ValueError("stock reference producer requires a fresh run; auto_resume found")
    target_counts = _target_audit_counts(train_result)
    _require_positive_int(target_counts.get("replay_sample_calls"), "replay_sample_calls")


def _assert_command_int(command: Mapping[str, Any], key: str, expected: int) -> None:
    value = command.get(key)
    try:
        observed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stock train command {key} mismatch") from exc
    if observed != int(expected):
        raise ValueError(f"stock train command {key} mismatch")


def _assert_command_float(command: Mapping[str, Any], key: str, expected: float) -> None:
    value = command.get(key)
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stock train command {key} mismatch") from exc
    if abs(observed - float(expected)) > 1.0e-9:
        raise ValueError(f"stock train command {key} mismatch")


def _assert_command_str(command: Mapping[str, Any], key: str, expected: str) -> None:
    if command.get(key) != str(expected):
        raise ValueError(f"stock train command {key} mismatch")


def _load_stock_status(
    args: argparse.Namespace,
    *,
    train_result: Mapping[str, Any],
) -> Mapping[str, Any]:
    del train_result
    status_fn = _deployed_modal_function(
        app_name=str(args.status_app_name),
        function_name="curvytron_run_status",
        modal_env=args.modal_env,
    )
    rows = status_fn.remote(
        [str(args.run_id)],
        [str(args.attempt_id)],
    )
    if not rows:
        raise ValueError("stock run status returned no rows")
    return _require_mapping(rows[0], "stock run status row")


def _load_stock_train_summary(
    args: argparse.Namespace,
    *,
    train_result: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        _training_wall_sec_from_result(train_result)
        return train_result
    except ValueError:
        pass
    summary_ref = train_result.get("summary_ref")
    if summary_ref is None or not str(summary_ref).strip():
        return train_result
    return _download_volume_json_ref(
        str(summary_ref),
        volume_name=str(args.runs_volume_name),
        modal_env=args.modal_env,
    )


def _select_stock_checkpoints(train_result: Mapping[str, Any]) -> StockCheckpointSelection:
    mirror = _required_mapping(train_result.get("checkpoint_mirror"), "checkpoint_mirror")
    copied = mirror.get("copied_checkpoints")
    if not isinstance(copied, Sequence) or isinstance(copied, (str, bytes)):
        raise ValueError("checkpoint_mirror.copied_checkpoints must be a list")
    by_iteration: dict[int, str] = {}
    ignored_mutable_refs: list[str] = []
    for index, item_obj in enumerate(copied):
        item = _required_mapping(item_obj, f"copied_checkpoints[{index}]")
        ref = _checkpoint_ref_from_record(item)
        if ref is None:
            continue
        name = Path(ref).name
        if name in MUTABLE_CHECKPOINT_NAMES or "latest" in name or "ckpt_best" in name:
            ignored_mutable_refs.append(ref)
            continue
        iteration = _checkpoint_iteration_from_ref(ref)
        if iteration is None:
            continue
        existing = by_iteration.get(iteration)
        if existing is not None and existing != ref:
            raise ValueError(f"duplicate stock checkpoint iteration {iteration}")
        by_iteration[iteration] = ref
    if 0 not in by_iteration:
        raise ValueError("stock reference requires immutable iteration_0.pth.tar")
    later_iterations = [iteration for iteration in by_iteration if iteration > 0]
    if not later_iterations:
        raise ValueError("stock reference requires a later immutable iteration_N.pth.tar")
    final_iteration = max(later_iterations)
    initial_ref = by_iteration[0]
    final_ref = by_iteration[final_iteration]
    if initial_ref == final_ref:
        raise ValueError("stock reference initial/final checkpoints must be distinct")
    return StockCheckpointSelection(
        initial_ref=initial_ref,
        initial_iteration=0,
        final_ref=final_ref,
        final_iteration=final_iteration,
        ignored_mutable_refs=tuple(sorted(set(ignored_mutable_refs))),
    )


def _checkpoint_ref_from_record(record: Mapping[str, Any]) -> str | None:
    for key in ("ref", "checkpoint_ref", "path"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _checkpoint_iteration_from_ref(ref: str) -> int | None:
    match = ITERATION_CHECKPOINT_RE.fullmatch(Path(ref).name)
    if match is None:
        return None
    return int(match.group(1))


def _download_stock_checkpoints(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    selection: StockCheckpointSelection,
) -> StockCheckpointFiles:
    checkpoint_dir = output_dir / "checkpoints"
    initial_path = checkpoint_dir / "iteration_0.pth.tar"
    final_path = checkpoint_dir / f"iteration_{selection.final_iteration}.pth.tar"
    _download_volume_ref(
        selection.initial_ref,
        initial_path,
        volume_name=str(args.runs_volume_name),
        modal_env=args.modal_env,
    )
    _download_volume_ref(
        selection.final_ref,
        final_path,
        volume_name=str(args.runs_volume_name),
        modal_env=args.modal_env,
    )
    for path in (initial_path, final_path):
        if not path.is_file():
            raise FileNotFoundError(f"downloaded stock checkpoint missing: {path}")
        if path.stat().st_size <= 0:
            raise ValueError(f"downloaded stock checkpoint is empty: {path}")
    return StockCheckpointFiles(initial_path=initial_path, final_path=final_path)


def _download_volume_ref(
    ref: str,
    output_path: Path,
    *,
    volume_name: str,
    modal_env: str | None,
) -> None:
    import modal

    clean_ref = str(ref).removeprefix("runs:").removeprefix("ref:").lstrip("/")
    remote_path = "/" + clean_ref
    volume = modal.Volume.from_name(
        volume_name,
        environment_name=modal_env,
        **modal_volume_kwargs_for_name(volume_name, create_if_missing=False),
    )
    buffer = io.BytesIO()
    volume.read_file_into_fileobj(remote_path, buffer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(buffer.getvalue())


def _download_volume_json_ref(
    ref: str,
    *,
    volume_name: str,
    modal_env: str | None,
) -> Mapping[str, Any]:
    import modal

    clean_ref = str(ref).removeprefix("runs:").removeprefix("ref:").lstrip("/")
    remote_path = "/" + clean_ref
    volume = modal.Volume.from_name(
        volume_name,
        environment_name=modal_env,
        **modal_volume_kwargs_for_name(volume_name, create_if_missing=False),
    )
    buffer = io.BytesIO()
    volume.read_file_into_fileobj(remote_path, buffer)
    return _require_mapping(json.loads(buffer.getvalue().decode("utf-8")), f"volume JSON {ref}")


def _run_stock_eval(
    *,
    args: argparse.Namespace,
    checkpoint_ref: str,
    checkpoint_iteration: int,
    point_id: str,
) -> Mapping[str, Any]:
    label = f"iteration_{int(checkpoint_iteration)}"
    eval_fn = _deployed_modal_function(
        app_name=str(args.train_app_name),
        function_name="lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect",
        modal_env=args.modal_env,
    )
    result = eval_fn.remote(
        checkpoint_ref=str(checkpoint_ref),
        checkpoint_label=label,
        eval_id=f"{args.run_id}-{point_id}",
        run_id=str(args.run_id),
        attempt_id=str(args.attempt_id),
        compute=str(args.eval_compute),
        seed=int(args.eval_seed),
        eval_seed_count=int(args.eval_seed_count),
        eval_seed_rng_seed=_effective_eval_seed_rng_seed(args),
        max_eval_steps=int(args.eval_steps),
        step_detail_limit=int(args.step_detail_limit),
        source_max_steps=int(args.source_max_steps),
        decision_ms=float(args.decision_ms),
        decision_source_frames=int(args.decision_source_frames),
        source_physics_step_ms=float(args.source_physics_step_ms),
        source_max_steps_semantics="source_physics_steps",
        num_simulations=int(args.num_simulations),
        batch_size=int(args.batch_size),
        env_variant=str(args.env_variant),
        reward_variant=str(args.reward_variant),
        reward_outcome_alpha=0.0,
        model_reward_variant=None,
        opponent_policy_kind=str(args.opponent_policy_kind),
        opponent_checkpoint_ref=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        opponent_mixture_spec=None,
        opponent_assignment_ref=None,
        opponent_death_mode=str(args.opponent_death_mode),
        opponent_runtime_mode=str(args.opponent_runtime_mode),
        natural_bonus_spawn=bool(args.natural_bonus_spawn),
    )
    return _require_mapping(result, f"{point_id} eval summary")


def _assert_eval_summary_bound_to_checkpoint(
    summary: Mapping[str, Any],
    *,
    expected_checkpoint_ref: str,
    expected_iteration: int,
    label: str,
) -> None:
    refs = _eval_checkpoint_refs(summary)
    if expected_checkpoint_ref not in refs:
        raise ValueError(f"{label} checkpoint_ref does not match selected checkpoint")
    for ref in refs:
        iteration = _checkpoint_iteration_from_ref(ref)
        if iteration is not None and iteration != int(expected_iteration):
            raise ValueError(f"{label} checkpoint iteration mismatch")
    if summary.get("ok") is not True:
        raise ValueError(f"{label} must be ok=true")


def _eval_checkpoint_refs(summary: Mapping[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in ("checkpoint_ref", "checkpoint_path"):
        value = summary.get(key)
        if value is not None:
            refs.add(str(value))
    selection = summary.get("selection")
    if isinstance(selection, Mapping):
        for key in ("checkpoint_ref", "checkpoint_refs"):
            value = selection.get(key)
            if value is not None:
                refs.add(str(value))
        jobs = selection.get("jobs")
        if isinstance(jobs, Sequence) and not isinstance(jobs, (str, bytes)):
            for job in jobs:
                if isinstance(job, Mapping) and job.get("checkpoint_ref") is not None:
                    refs.add(str(job["checkpoint_ref"]))
    config = summary.get("config")
    if isinstance(config, Mapping) and config.get("checkpoint_ref") is not None:
        refs.add(str(config["checkpoint_ref"]))
    return refs


def _validate_and_extract_matched_eval_seed_set(
    pre_eval: Mapping[str, Any],
    post_eval: Mapping[str, Any],
    *,
    args: argparse.Namespace,
) -> list[int]:
    pre = _eval_seed_set(pre_eval, args=args)
    post = _eval_seed_set(post_eval, args=args)
    if pre != post:
        raise ValueError("pre/post eval seed sets must match")
    if len(pre) != int(args.eval_seed_count):
        raise ValueError("eval seed count mismatch")
    return pre


def _eval_seed_set(summary: Mapping[str, Any], *, args: argparse.Namespace) -> list[int]:
    selection = summary.get("selection")
    if isinstance(selection, Mapping):
        for key in ("eval_seed_values", "eval_seed_set", "eval_seeds"):
            values = selection.get(key)
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                return [int(value) for value in values]
    seeds_from_table = {
        int(row["seed"])
        for row in summary.get("survival_table", [])
        if isinstance(row, Mapping) and row.get("seed") is not None
    }
    if seeds_from_table:
        return sorted(seeds_from_table)
    if int(args.eval_seed_count) == 1:
        return [int(args.eval_seed)]
    raise ValueError("eval summary missing explicit seed set")


def _assert_stock_quality_eval_not_saturated(
    pre_eval: Mapping[str, Any],
    post_eval: Mapping[str, Any],
) -> None:
    if _eval_summary_all_capped(pre_eval) and _eval_summary_all_capped(post_eval):
        raise ValueError("stock_reference saturated eval horizon")


def _assert_eval_summaries_match_requested_horizon(
    pre_eval: Mapping[str, Any],
    post_eval: Mapping[str, Any],
    *,
    args: argparse.Namespace,
) -> None:
    requested = int(args.eval_steps)
    for label, summary in (("pre_train", pre_eval), ("post_train", post_eval)):
        observed = _eval_summary_max_steps(summary)
        if observed is None:
            raise ValueError(f"{label} eval summary missing eval_max_steps")
        if int(observed) != requested:
            raise ValueError(
                f"{label} eval_max_steps mismatch: expected {requested}, got {observed}"
            )


def _eval_summary_max_steps(summary: Mapping[str, Any]) -> int | None:
    for container in (summary, summary.get("selection"), summary.get("config")):
        if not isinstance(container, Mapping):
            continue
        for key in ("eval_max_steps", "max_eval_steps", "eval_steps"):
            value = container.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    rows = summary.get("survival_table")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        caps = {
            int(row["cap"])
            for row in rows
            if isinstance(row, Mapping) and row.get("cap") is not None
        }
        if len(caps) == 1:
            return caps.pop()
    return None


def _eval_summary_all_capped(summary: Mapping[str, Any]) -> bool:
    rows = summary.get("survival_aggregate_table")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) and rows:
        row = next((item for item in rows if isinstance(item, Mapping)), {})
    else:
        row = summary
    episode_count = row.get("eval_episode_count") or row.get("seeds") or row.get("ok_count")
    capped_count = row.get("capped_count") or row.get("cap_count")
    histogram = row.get("outcome_histogram")
    if capped_count is None and isinstance(histogram, Mapping):
        capped_count = histogram.get("cap") or histogram.get("capped")
    if episode_count is None or capped_count is None:
        return False
    try:
        return int(capped_count) >= int(episode_count) > 0
    except (TypeError, ValueError):
        return False


def _normalized_eval_summary(
    summary: Mapping[str, Any],
    *,
    checkpoint_ref: str,
    checkpoint_iteration: int,
    eval_episode_count: int,
) -> dict[str, Any]:
    normalized = _plain_json_mapping(summary)
    normalized["checkpoint_ref"] = str(checkpoint_ref)
    normalized["checkpoint_step"] = int(checkpoint_iteration)
    aggregate = normalized.get("survival_aggregate_table")
    if isinstance(aggregate, list) and len(aggregate) == 1 and isinstance(aggregate[0], dict):
        aggregate[0].setdefault("checkpoint_ref", str(checkpoint_ref))
        aggregate[0].setdefault("checkpoint_step", int(checkpoint_iteration))
        aggregate[0].setdefault("eval_episode_count", int(eval_episode_count))
        aggregate[0].setdefault("evaluator_eval_calls", 1)
    return normalized


def _write_packager_inputs(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    output_dir: Path,
    inputs_dir: Path,
    train_result: Mapping[str, Any],
    train_summary: Mapping[str, Any],
    status: Mapping[str, Any],
    selection: StockCheckpointSelection,
    checkpoint_files: StockCheckpointFiles,
    model_identity: Mapping[str, Any],
    pre_eval: Mapping[str, Any],
    post_eval: Mapping[str, Any],
    eval_seed_set: Sequence[int],
) -> dict[str, Path]:
    paths = {
        "source_fingerprint": inputs_dir / "source_fingerprint.json",
        "model_identity": inputs_dir / "model_identity.json",
        "eval_settings": inputs_dir / "eval_settings.json",
        "denominators": inputs_dir / "denominators.json",
        "capture_provenance": inputs_dir / "capture_provenance.json",
        "training_artifact": inputs_dir / "training_artifact.json",
        "pre_eval_summary": output_dir / "eval" / "pre_eval_summary.json",
        "post_eval_summary": output_dir / "eval" / "post_eval_summary.json",
        "initial_checkpoint": checkpoint_files.initial_path,
        "final_checkpoint": checkpoint_files.final_path,
    }
    denominators = _denominators(
        train_result=train_result,
        train_summary=train_summary,
        status=status,
        selection=selection,
    )
    _write_json(
        paths["source_fingerprint"],
        _source_fingerprint(
            args=args,
            repo_root=repo_root,
            train_result=train_result,
            eval_seed_set=eval_seed_set,
        ),
    )
    _write_json(paths["model_identity"], model_identity)
    _write_json(paths["eval_settings"], _eval_settings(args=args, eval_seed_set=eval_seed_set))
    _write_json(paths["denominators"], denominators)
    _write_json(paths["capture_provenance"], _capture_provenance(train_result))
    _write_json(
        paths["training_artifact"],
        _training_artifact(
            args=args,
            output_dir=output_dir,
            train_result=train_result,
            status=status,
            selection=selection,
            checkpoint_files=checkpoint_files,
            model_identity=model_identity,
            denominators=denominators,
        ),
    )
    _write_json(paths["pre_eval_summary"], pre_eval)
    _write_json(paths["post_eval_summary"], post_eval)
    return paths


def _packager_args(
    *,
    args: argparse.Namespace,
    input_paths: Mapping[str, Path],
    capture_path: Path,
    checkpoint_files: StockCheckpointFiles,
) -> list[str]:
    return [
        "--role",
        STOCK_REFERENCE_ROLE,
        "--run-id",
        str(args.run_id),
        "--capture-id",
        f"{args.run_id}:stock-reference-capture",
        "--candidate-checkpoint-id",
        str(args.candidate_checkpoint_id),
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
        str(checkpoint_files.initial_path),
        "--final-checkpoint",
        str(checkpoint_files.final_path),
        "--output",
        str(capture_path),
    ]


def _invoke_capture_packager(argv: list[str]) -> int:
    return int(packager.main(argv))


def _producer_manifest(
    *,
    args: argparse.Namespace,
    train_result: Mapping[str, Any],
    train_summary: Mapping[str, Any],
    status: Mapping[str, Any],
    selection: StockCheckpointSelection,
    checkpoint_files: StockCheckpointFiles,
    input_paths: Mapping[str, Path],
    capture_path: Path,
    packager_args: list[str],
) -> dict[str, Any]:
    return {
        "schema_id": PRODUCER_SCHEMA_ID,
        "ok": True,
        "role": STOCK_REFERENCE_ROLE,
        "producer_id": PRODUCER_ID,
        "run_id": str(args.run_id),
        "attempt_id": str(args.attempt_id),
        "created_at": datetime.now(UTC).isoformat(),
        "support_only": False,
        "profile_only": False,
        "touches_live_runs": False,
        "called_train_muzero": True,
        "train_summary_ref": train_result.get("summary_ref"),
        "capture_path": str(capture_path),
        "packager_args": list(packager_args),
        "input_paths": {key: str(path) for key, path in input_paths.items()},
        "checkpoint_selection": {
            "initial_ref": selection.initial_ref,
            "initial_iteration": selection.initial_iteration,
            "final_ref": selection.final_ref,
            "final_iteration": selection.final_iteration,
            "ignored_mutable_refs": list(selection.ignored_mutable_refs),
        },
        "downloaded_checkpoints": {
            "initial": str(checkpoint_files.initial_path),
            "final": str(checkpoint_files.final_path),
        },
        "denominators": _denominators(
            train_result=train_result,
            train_summary=train_summary,
            status=status,
            selection=selection,
        ),
        "producer_guardrails": {
            "modal_train_muzero_called": True,
            "mode_train": True,
            "fresh_run_auto_resume_absent": True,
            "immutable_iteration_0_selected": True,
            "immutable_later_iteration_selected": True,
            "eval_bound_to_exact_checkpoint_refs": True,
            "stock_internal_evaluator_env_num_matches_episode_count": True,
            "denominators_derived_from_learner_metrics_and_target_audit": True,
            "packager_writes_single_role_capture": True,
        },
    }


def _training_artifact(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    train_result: Mapping[str, Any],
    status: Mapping[str, Any],
    selection: StockCheckpointSelection,
    checkpoint_files: StockCheckpointFiles,
    model_identity: Mapping[str, Any],
    denominators: Mapping[str, Any],
) -> dict[str, Any]:
    command = _required_mapping(train_result.get("command"), "stock train command")
    return {
        "schema_id": TRAINING_ARTIFACT_SCHEMA_ID,
        "ok": True,
        "role": STOCK_REFERENCE_ROLE,
        "route": "stock_train_muzero_reference",
        "producer_id": PRODUCER_ID,
        "run_id": str(args.run_id),
        "attempt_id": str(args.attempt_id),
        "output_dir": str(output_dir),
        "profile_only": False,
        "support_only": False,
        "touches_live_runs": False,
        "called_train_muzero": True,
        "train_muzero_entrypoint": train_result.get("trainer_entrypoint"),
        "train_summary_ref": train_result.get("summary_ref"),
        "command": _plain_json_mapping(command),
        "stock_internal_evaluator_surface": _stock_internal_evaluator_surface(
            args=args
        ),
        "run_status": _plain_json_mapping(status),
        "checkpoint_selection": {
            "initial_ref": selection.initial_ref,
            "initial_iteration": selection.initial_iteration,
            "final_ref": selection.final_ref,
            "final_iteration": selection.final_iteration,
            "ignored_mutable_refs": list(selection.ignored_mutable_refs),
        },
        "downloaded_checkpoints": {
            "initial_path": str(checkpoint_files.initial_path),
            "initial_sha256": _file_sha256(checkpoint_files.initial_path),
            "final_path": str(checkpoint_files.final_path),
            "final_sha256": _file_sha256(checkpoint_files.final_path),
        },
        "model_identity": _plain_json_mapping(model_identity),
        "denominators": _plain_json_mapping(denominators),
        "train_result_compact": {
            key: _plain_json_value(train_result.get(key))
            for key in (
                "ok",
                "status",
                "mode",
                "compute",
                "called_train_muzero",
                "trainer_entrypoint",
                "train_result",
                "target_audit",
                "checkpoint_mirror",
                "auto_resume",
                "runtime_compute",
                "artifact_refs",
            )
        },
    }


def _model_identity(checkpoint_files: StockCheckpointFiles) -> dict[str, Any]:
    initial_digest = _stock_model_state_digest_from_checkpoint(checkpoint_files.initial_path)
    final_digest = _stock_model_state_digest_from_checkpoint(checkpoint_files.final_path)
    return {
        "model_identity_scope": "stock_lightzero_iteration_checkpoint_file",
        "digest_material": "downloaded_checkpoint_file_sha256",
        "initial_model_state_digest": initial_digest,
        "final_model_state_digest": final_digest,
        "model_state_digest_changed": initial_digest != final_digest,
        "initial_checkpoint_sha256": _file_sha256(checkpoint_files.initial_path),
        "final_checkpoint_sha256": _file_sha256(checkpoint_files.final_path),
    }


def _stock_model_state_digest_from_checkpoint(path: Path) -> str:
    return _file_sha256(path)


def _validate_model_identity(identity: Mapping[str, Any]) -> None:
    if identity.get("initial_model_state_digest") == identity.get("final_model_state_digest"):
        raise ValueError("stock reference model state digest did not change")
    if identity.get("model_state_digest_changed") is not True:
        raise ValueError("stock reference model_state_digest_changed must be true")


def _denominators(
    *,
    train_result: Mapping[str, Any],
    train_summary: Mapping[str, Any] | None = None,
    status: Mapping[str, Any],
    selection: StockCheckpointSelection,
) -> dict[str, Any]:
    target_counts = _target_audit_counts(train_result)
    if status.get("learner_metrics_latest_exists") is not True:
        raise ValueError("learner_metrics_latest must exist for stock denominators")
    learner_train_calls = _require_positive_int(
        status.get("learner_train_call_index"),
        "learner_train_call_index",
    )
    collector_envstep_delta = _require_positive_int(
        status.get("learner_collector_envstep"),
        "learner_collector_envstep",
    )
    replay_sample_calls = _require_positive_int(
        target_counts.get("replay_sample_calls"),
        "replay_sample_calls",
    )
    training_wall_sec = _training_wall_sec_from_result(
        train_result,
        train_summary=train_summary,
    )
    return {
        "denominator_currency": "stock_train_muzero_learning_quality",
        "env_step_currency": "stock_train_muzero_raw_env_steps",
        "speed_currency": "stock_train_muzero_raw_env_steps_per_sec",
        "collector_envstep_delta": collector_envstep_delta,
        "learner_train_calls": learner_train_calls,
        "replay_sample_calls": replay_sample_calls,
        "checkpoint_iteration_delta": (
            int(selection.final_iteration) - int(selection.initial_iteration)
        ),
        "training_wall_sec": training_wall_sec,
        "uses_fallback_denominator": False,
        "wall_sec_used_for_speed_claim": False,
    }


def _training_wall_sec_from_result(
    train_result: Mapping[str, Any],
    *,
    train_summary: Mapping[str, Any] | None = None,
) -> float:
    train_payload = train_result.get("train_result")
    if isinstance(train_payload, Mapping):
        value = train_payload.get("elapsed_sec")
        if value is not None:
            return _require_positive_number(value, "train_result.elapsed_sec")
    if train_summary is not None and train_summary is not train_result:
        try:
            return _training_wall_sec_from_result(train_summary)
        except ValueError:
            pass
    timers = train_result.get("timers_sec")
    if isinstance(timers, Mapping):
        value = timers.get("train_muzero_wall") or timers.get("train_muzero_wall_sec")
        if value is not None:
            return _require_positive_number(value, "timers_sec.train_muzero_wall")
    if train_result.get("elapsed_sec") is not None:
        return _require_positive_number(train_result.get("elapsed_sec"), "elapsed_sec")
    raise ValueError("stock train result missing positive training wall seconds")


def _target_audit_counts(train_result: Mapping[str, Any]) -> Mapping[str, Any]:
    target_audit = _required_mapping(train_result.get("target_audit"), "target_audit")
    return _required_mapping(target_audit.get("counts"), "target_audit.counts")


def _capture_provenance(train_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_id": COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_PROVENANCE_SCHEMA_ID,
        "role": STOCK_REFERENCE_ROLE,
        "producer_id": PRODUCER_ID,
        "producer_ran_training": True,
        "producer_ran_pre_eval": True,
        "producer_ran_post_eval": True,
        "feeds_builder": True,
        "support_only": False,
        "training_artifact_ref": "training_artifact",
        "pre_eval_artifact_ref": "pre_eval_summary",
        "post_eval_artifact_ref": "post_eval_summary",
        "stock_train_muzero": {
            "called_train_muzero": True,
            "train_muzero_entrypoint": str(train_result.get("trainer_entrypoint")),
            "summary_ref": train_result.get("summary_ref"),
        },
    }


def _eval_settings(
    *,
    args: argparse.Namespace,
    eval_seed_set: Sequence[int],
) -> dict[str, Any]:
    return {
        "observation_schema_id": "curvyzero_source_state_rgb_canvas_like_gray64_stack4/v0",
        "policy_observation_backend": str(args.policy_observation_backend),
        "eval_seed_set": [int(seed) for seed in eval_seed_set],
        "eval_seed_rng_seed": _effective_eval_seed_rng_seed(args),
        "eval_episode_count": len(eval_seed_set),
        "stock_internal_evaluator_surface": _stock_internal_evaluator_surface(
            args=args
        ),
        "source_max_steps": int(args.source_max_steps),
        "eval_max_steps": int(args.eval_steps),
        "num_simulations": int(args.num_simulations),
        "batch_size": int(args.batch_size),
        "reward_variant": str(args.reward_variant),
        "reward_target_effect": "extrinsic_reward_only",
        "death_mode": "normal",
        "terminal_target_mode": "stock_terminal_no_bootstrap_return_discount_1.0",
        "root_noise": 0.0,
        "dirichlet_alpha": 0.3,
        "policy_noise": 0.0,
        "rnd_enabled": False,
        "exploration_bonus_mode": "none",
        "opponent_policy_ref": "fixed_straight",
        "opponent_policy_kind": str(args.opponent_policy_kind),
        "opponent_runtime_mode": str(args.opponent_runtime_mode),
        "opponent_death_mode": str(args.opponent_death_mode),
        "natural_bonus_spawn": bool(args.natural_bonus_spawn),
        "training_seed_policy": "fixed_train_and_eval_seed",
        "initialization_source": "fresh_stock_train_muzero_run",
        "num_unroll_steps": 1,
        "td_steps": 1,
        "discount": 1.0,
        "support_scale": 300,
    }


def _source_fingerprint(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    train_result: Mapping[str, Any],
    eval_seed_set: Sequence[int],
) -> dict[str, Any]:
    command = _required_mapping(train_result.get("command"), "stock train command")
    return {
        "git_commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "git_status_dirty": bool(_git_output(repo_root, "status", "--short")),
        "producer_script": PRODUCER_ID,
        "producer_route": "stock_reference_modal_train_muzero_producer",
        "train_summary_ref": train_result.get("summary_ref"),
        "runtime_compute": _plain_json_value(train_result.get("runtime_compute")),
        "matched_surface": {
            "env_variant": command.get("env_variant"),
            "reward_variant": command.get("reward_variant"),
            "policy_observation_backend": command.get("policy_observation_backend"),
            "opponent_policy_kind": command.get("opponent_policy_kind"),
            "eval_seed_set": [int(seed) for seed in eval_seed_set],
            "eval_seed_rng_seed": _effective_eval_seed_rng_seed(args),
            "stock_internal_evaluator_surface": _stock_internal_evaluator_surface(
                args=args
            ),
        },
    }


def _effective_eval_seed_rng_seed(args: argparse.Namespace) -> int:
    if args.eval_seed_rng_seed is not None:
        return int(args.eval_seed_rng_seed)
    return int(args.eval_seed)


def _stock_internal_evaluator_surface(args: argparse.Namespace) -> dict[str, Any]:
    evaluator_env_num = int(args.evaluator_env_num)
    n_evaluator_episode = int(args.n_evaluator_episode)
    return {
        "evaluator_env_num": evaluator_env_num,
        "n_evaluator_episode": n_evaluator_episode,
        "env_num_matches_episode_count": evaluator_env_num == n_evaluator_episode,
        "eval_seed_count": int(args.eval_seed_count),
        "evalenv_full_episode_surface": True,
        "empty_ready_set_workaround_formalized": True,
    }


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    return _require_mapping(value, label)


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{label} must be > 0")
    return result


def _require_positive_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be positive") from exc
    if result <= 0.0 or result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{label} must be positive")
    return result


def _plain_json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _plain_json_value(item)
        for key, item in value.items()
    }


def _plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_json_value(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain_json_mapping(payload), indent=2, sort_keys=True) + "\n",
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


if __name__ == "__main__":
    raise SystemExit(main())
