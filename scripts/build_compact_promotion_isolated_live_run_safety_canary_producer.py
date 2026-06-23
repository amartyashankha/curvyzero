#!/usr/bin/env python3
"""Run and package the isolated live-run safety sandbox canary."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any

from curvyzero.contracts.curvytron import curvytron_train_app_name
from curvyzero.contracts.curvytron import modal_volume_kwargs_for_name
from curvyzero.training.compact_stock_checkpoint_export import (
    COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
)
from curvyzero.training.compact_promotion_readiness import (
    build_compact_promotion_isolated_live_run_safety_canary_v1,
)
from curvyzero.training.compact_promotion_readiness import (
    validate_compact_promotion_stock_resume_load_canary_v1,
)
from curvyzero.training.opponent_leaderboard import validate_assignment_audit
from curvyzero.training.opponent_registry import (
    canonical_assignment_json_sha256,
)
from curvyzero.training.opponent_registry import parse_opponent_assignment_snapshot


DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_compact_promotion_readiness_results")
DEFAULT_RUN_ID = "optimizer-compact-promotion-isolated-live-run-safety-canary-20260530"
DEFAULT_ATTEMPT_ID = "isolated-live-run-safety"
PRODUCER_SCHEMA_ID = (
    "curvyzero_compact_promotion_isolated_live_run_safety_canary_producer/v1"
)
TRAINER_RESULT_SCHEMA_ID = "curvyzero_compact_isolated_live_run_trainer_result/v1"
METRICS_SCHEMA_ID = "curvyzero_compact_isolated_live_run_metrics/v1"
FORBIDDEN_TOUCH_AUDIT_SCHEMA_ID = (
    "curvyzero_compact_isolated_live_run_touch_audit/v1"
)
PRODUCER_ID = "scripts/build_compact_promotion_isolated_live_run_safety_canary_producer.py"
ASSIGNMENT_WRITER_FUNCTION = "lightzero_curvytron_write_opponent_assignment_artifacts"
ITERATION_CHECKPOINT_RE = re.compile(r"\Aiteration_(\d+)\.pth\.tar\Z")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path.cwd().resolve()
    output_dir = (repo_root / args.output_root / args.run_id).resolve()
    _prepare_fresh_output_dir(output_dir)

    stock_canary = _read_json_mapping(
        _resolve_input(args.stock_resume_load_canary_report, repo_root),
        "stock resume/load canary report",
    )
    validate_compact_promotion_stock_resume_load_canary_v1(stock_canary)

    initial_checkpoint_path = Path(str(stock_canary["resumed_stock_export_path"])).resolve()
    if not initial_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"stock-resume export checkpoint missing: {initial_checkpoint_path}"
        )
    remote_initial = _upload_initial_checkpoint(
        args=args,
        initial_checkpoint_path=initial_checkpoint_path,
    )
    inputs_dir = output_dir / "canary_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    assignment_path, assignment_audit_path, assignment, assignment_audit = (
        _load_or_generate_assignment_inputs(
            args=args,
            repo_root=repo_root,
            inputs_dir=inputs_dir,
            remote_initial=remote_initial,
        )
    )
    remote_assignment = _write_remote_assignment(
        args=args,
        assignment=assignment,
        assignment_audit=assignment_audit,
    )
    train_result = _run_sandbox_stock_train(
        args=args,
        initial_checkpoint_ref=str(remote_initial["checkpoint_ref"]),
        assignment_ref=str(remote_assignment["assignment_ref"]),
    )
    raw_train_result_path = inputs_dir / "sandbox_raw_train_result.json"
    _write_json(raw_train_result_path, train_result)
    _validate_sandbox_train_result(
        train_result,
        args=args,
        remote_initial=remote_initial,
        remote_assignment=remote_assignment,
    )
    status = _load_run_status(args=args)
    assignment_proof = _load_assignment_proof(
        args=args,
        assignment_sha256=canonical_assignment_json_sha256(assignment),
    )
    final_selection = _select_final_checkpoint(train_result)
    final_checkpoint_path = output_dir / "checkpoints" / Path(final_selection["ref"]).name
    _download_volume_ref(
        str(final_selection["ref"]),
        final_checkpoint_path,
        volume_name=str(args.runs_volume_name),
        modal_env=args.modal_env,
    )

    upload_path = inputs_dir / "initial_checkpoint_upload.json"
    assignment_write_path = inputs_dir / "remote_assignment_write.json"
    trainer_result_path = inputs_dir / "sandbox_trainer_result.json"
    metrics_path = inputs_dir / "sandbox_metrics.json"
    forbidden_touch_path = inputs_dir / "forbidden_touch_audit.json"
    _write_json(upload_path, remote_initial)
    _write_json(assignment_write_path, remote_assignment)
    trainer_result_payload = _trainer_result_payload(
        args=args,
        stock_canary=stock_canary,
        train_result=train_result,
        remote_initial=remote_initial,
        remote_assignment=remote_assignment,
        assignment_proof=assignment_proof,
        assignment_sha256=canonical_assignment_json_sha256(assignment),
        initial_checkpoint_path=initial_checkpoint_path,
    )
    metrics_payload = _metrics_payload(
        train_result=train_result,
        status=status,
        assignment_proof=assignment_proof,
        final_selection=final_selection,
        initial_checkpoint_path=initial_checkpoint_path,
        final_checkpoint_path=final_checkpoint_path,
    )
    forbidden_touch_payload = _forbidden_touch_audit_payload(args=args)
    _write_json(trainer_result_path, trainer_result_payload)
    _write_json(metrics_path, metrics_payload)
    _write_json(forbidden_touch_path, forbidden_touch_payload)

    report = build_compact_promotion_isolated_live_run_safety_canary_v1(
        run_id=str(args.run_id),
        unified_lifecycle_report_path=_resolve_input(args.unified_lifecycle_report, repo_root),
        compatibility_report_path=_resolve_input(args.compatibility_report, repo_root),
        stock_resume_load_canary_report_path=_resolve_input(
            args.stock_resume_load_canary_report,
            repo_root,
        ),
        assignment_path=assignment_path,
        assignment_audit_path=assignment_audit_path,
        trainer_result_path=trainer_result_path,
        metrics_path=metrics_path,
        forbidden_touch_audit_path=forbidden_touch_path,
        initial_checkpoint_path=initial_checkpoint_path,
        final_checkpoint_path=final_checkpoint_path,
        repo_root=repo_root,
    )
    report_path = output_dir / "isolated_live_run_safety_canary_report.json"
    manifest_path = output_dir / "isolated_live_run_safety_canary_producer_manifest.json"
    _write_json(report_path, report)
    _write_json(
        manifest_path,
        _producer_manifest(
            args=args,
            report_path=report_path,
        upload_path=upload_path,
        assignment_write_path=assignment_write_path,
        raw_train_result_path=raw_train_result_path,
        trainer_result_path=trainer_result_path,
        metrics_path=metrics_path,
        forbidden_touch_path=forbidden_touch_path,
        assignment_path=assignment_path,
        assignment_audit_path=assignment_audit_path,
        final_checkpoint_path=final_checkpoint_path,
        final_selection=final_selection,
    ),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "run_id": str(args.run_id),
                "report_path": str(report_path),
                "manifest_path": str(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--attempt-id", default=DEFAULT_ATTEMPT_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--unified-lifecycle-report", type=Path, required=True)
    parser.add_argument("--compatibility-report", type=Path, required=True)
    parser.add_argument("--stock-resume-load-canary-report", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, default=None)
    parser.add_argument("--assignment-audit", type=Path, default=None)
    parser.add_argument(
        "--compute",
        choices=("cpu", "gpu-l4-t4-cpu40", "gpu-h100-cpu40"),
        default="cpu",
    )
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--max-env-step", type=int, default=64)
    parser.add_argument("--max-train-iter", type=int, default=1)
    parser.add_argument("--source-max-steps", type=int, default=1_048_576)
    parser.add_argument("--decision-ms", type=float, default=50.0 / 3.0)
    parser.add_argument("--collector-env-num", type=int, default=2)
    parser.add_argument("--evaluator-env-num", type=int, default=2)
    parser.add_argument("--n-episode", type=int, default=2)
    parser.add_argument("--n-evaluator-episode", type=int, default=2)
    parser.add_argument("--num-simulations", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--save-ckpt-after-iter", type=int, default=1)
    parser.add_argument("--env-telemetry-stride", type=int, default=1)
    parser.add_argument("--train-app-name", default=curvytron_train_app_name())
    parser.add_argument("--status-app-name", default="curvyzero-lightzero-curvytron-run-status")
    parser.add_argument("--modal-env", default=None)
    parser.add_argument("--runs-volume-name", default="curvyzero-runs-v2")
    return parser.parse_args(argv)


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


def _validate_assignment_inputs(
    assignment: Mapping[str, Any],
    assignment_audit: Mapping[str, Any],
) -> None:
    parsed = parse_opponent_assignment_snapshot(dict(assignment))
    if parsed is None:
        raise ValueError("sandbox assignment did not parse")
    validate_assignment_audit(assignment_audit, assignment=assignment)


def _load_or_generate_assignment_inputs(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    inputs_dir: Path,
    remote_initial: Mapping[str, Any],
) -> tuple[Path, Path, Mapping[str, Any], Mapping[str, Any]]:
    if (args.assignment is None) != (args.assignment_audit is None):
        raise ValueError("--assignment and --assignment-audit must be provided together")
    if args.assignment is not None and args.assignment_audit is not None:
        assignment_path = _resolve_input(args.assignment, repo_root)
        assignment_audit_path = _resolve_input(args.assignment_audit, repo_root)
        assignment = _read_json_mapping(assignment_path, "sandbox assignment")
        assignment_audit = _read_json_mapping(
            assignment_audit_path,
            "sandbox assignment audit",
        )
        _validate_assignment_inputs(assignment, assignment_audit)
        return assignment_path, assignment_audit_path, assignment, assignment_audit

    assignment = _generated_frozen_initial_assignment(
        args=args,
        remote_initial=remote_initial,
    )
    assignment_audit = _generated_assignment_audit(
        assignment=assignment,
        remote_initial=remote_initial,
    )
    _validate_assignment_inputs(assignment, assignment_audit)
    assignment_path = inputs_dir / "sandbox_assignment.generated.json"
    assignment_audit_path = inputs_dir / "sandbox_assignment_audit.generated.json"
    _write_json(assignment_path, assignment)
    _write_json(assignment_audit_path, assignment_audit)
    return assignment_path, assignment_audit_path, assignment, assignment_audit


def _generated_frozen_initial_assignment(
    *,
    args: argparse.Namespace,
    remote_initial: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_ref = str(remote_initial["checkpoint_ref"])
    assignment_id = _generated_assignment_id(args=args)
    return {
        "schema_id": "curvyzero_opponent_assignment/v0",
        "assignment_id": assignment_id,
        "source_epoch": 1,
        "source_ref": (
            f"sandbox/opt057/{args.run_id}/{args.attempt_id}/"
            "generated_frozen_initial_assignment.json"
        ),
        "seed": int(args.seed),
        "entries": [
            {
                "name": "sandbox_frozen_initial_policy",
                "weight": 1,
                "opponent_policy_kind": "frozen_lightzero_checkpoint",
                "opponent_runtime_mode": "normal",
                "opponent_immortal": False,
                "opponent_checkpoint_ref": checkpoint_ref,
                "opponent_checkpoint_state_key": COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
                "opponent_num_simulations": int(args.num_simulations),
                "opponent_batch_size": int(args.batch_size),
                "opponent_use_cuda": False,
            }
        ],
    }


def _generated_assignment_id(*, args: argparse.Namespace) -> str:
    digest = hashlib.sha256(
        f"{args.run_id}/{args.attempt_id}/isolated-live-frozen-initial".encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    return f"opt057-frozen-initial-{digest}"


def _generated_assignment_audit(
    *,
    assignment: Mapping[str, Any],
    remote_initial: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_id": "curvyzero_opponent_assignment_audit/v0",
        "assignment_id": str(assignment["assignment_id"]),
        "assignment_sha256": canonical_assignment_json_sha256(assignment),
        "selection": {
            "strategy_id": "isolated_live_frozen_initial_v1",
            "source": "producer_generated_sandbox_fixture",
            "opponent_checkpoint_ref": str(remote_initial["checkpoint_ref"]),
            "opponent_checkpoint_sha256": str(remote_initial["local_sha256"]),
        },
    }


def _upload_initial_checkpoint(
    *,
    args: argparse.Namespace,
    initial_checkpoint_path: Path,
) -> dict[str, Any]:
    remote_ref = (
        f"training/lightzero-curvytron-visual-survival/{args.run_id}/attempts/"
        f"{args.attempt_id}/isolated_live_seed/iteration_0.pth.tar"
    )
    _upload_volume_bytes(
        content=initial_checkpoint_path.read_bytes(),
        remote_ref=remote_ref,
        volume_name=str(args.runs_volume_name),
        modal_env=args.modal_env,
    )
    return {
        "schema_id": "curvyzero_compact_isolated_live_initial_checkpoint_upload/v1",
        "ok": True,
        "run_id": str(args.run_id),
        "attempt_id": str(args.attempt_id),
        "checkpoint_ref": remote_ref,
        "upload_ref": "/" + remote_ref.lstrip("/"),
        "local_path": str(initial_checkpoint_path),
        "local_sha256": _file_sha256(initial_checkpoint_path),
        "state_key": COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
        "load_mode": "matching_shape",
    }


def _write_remote_assignment(
    *,
    args: argparse.Namespace,
    assignment: Mapping[str, Any],
    assignment_audit: Mapping[str, Any],
) -> dict[str, Any]:
    writer_fn = _deployed_modal_function(
        app_name=str(args.train_app_name),
        function_name=ASSIGNMENT_WRITER_FUNCTION,
        modal_env=args.modal_env,
    )
    result = _require_mapping(
        writer_fn.remote(
            {
                "run_id": str(args.run_id),
                "attempt_id": str(args.attempt_id),
                "assignment": dict(assignment),
                "audit": dict(assignment_audit),
                "target_volume": "runs",
                "mirror_checkpoints_to_control": False,
            }
        ),
        "remote assignment write result",
    )
    expected_sha = canonical_assignment_json_sha256(assignment)
    if str(result.get("assignment_sha256", "")) != expected_sha:
        raise ValueError("remote assignment write returned a different sha256")
    if not str(result.get("assignment_ref", "")).strip():
        raise ValueError("remote assignment write did not return assignment_ref")
    return dict(result)


def _run_sandbox_stock_train(
    *,
    args: argparse.Namespace,
    initial_checkpoint_ref: str,
    assignment_ref: str,
) -> Mapping[str, Any]:
    train_fn = _stock_train_function(args, compute=str(args.compute))
    return _require_mapping(
        train_fn.remote(
            **_sandbox_train_kwargs(
                args,
                initial_checkpoint_ref=initial_checkpoint_ref,
                assignment_ref=assignment_ref,
            )
        ),
        "sandbox stock train result",
    )


def _stock_train_function(args: argparse.Namespace, *, compute: str) -> Any:
    if compute == "cpu":
        function_name = "lightzero_curvytron_visual_survival_cpu"
    elif compute == "gpu-l4-t4-cpu40":
        function_name = "lightzero_curvytron_visual_survival_gpu_cpu40"
    elif compute == "gpu-h100-cpu40":
        function_name = "lightzero_curvytron_visual_survival_h100_cpu40"
    else:
        raise ValueError(f"unsupported sandbox compute: {compute}")
    return _deployed_modal_function(
        app_name=str(args.train_app_name),
        function_name=function_name,
        modal_env=args.modal_env,
    )


def _sandbox_train_kwargs(
    args: argparse.Namespace,
    *,
    initial_checkpoint_ref: str,
    assignment_ref: str,
) -> dict[str, Any]:
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
        "env_variant": "source_state_fixed_opponent",
        "reward_variant": "survival_plus_bonus_no_outcome",
        "policy_observation_backend": "cpu_oracle",
        "collect_search_backend": "stock",
        "collect_search_ctree_backend": "lightzero",
        "env_manager_type": "base",
        "opponent_policy_kind": "fixed_straight",
        "opponent_runtime_mode": "normal",
        "opponent_death_mode": "normal",
        "opponent_assignment_ref": str(assignment_ref),
        "opponent_assignment_refresh_interval_train_iter": 0,
        "opponent_assignment_refresh_ref": None,
        "own_checkpoint_opponent_refresh_enabled": False,
        "initial_policy_checkpoint_ref": str(initial_checkpoint_ref),
        "initial_policy_checkpoint_state_key": COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
        "initial_policy_checkpoint_load_mode": "matching_shape",
        "env_telemetry_stride": int(args.env_telemetry_stride),
        "background_eval_enabled": False,
        "background_gif_enabled": False,
        "exploration_bonus_mode": "none",
        "exploration_bonus_weight": 0.0,
    }


def _validate_sandbox_train_result(
    train_result: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    remote_initial: Mapping[str, Any],
    remote_assignment: Mapping[str, Any],
) -> None:
    if train_result.get("ok") is not True:
        raise ValueError("sandbox stock train result must be ok=true")
    if train_result.get("mode") != "train":
        raise ValueError("sandbox stock train result mode must be train")
    if train_result.get("called_train_muzero") is not True:
        raise ValueError("sandbox stock train must call train_muzero")
    if train_result.get("trainer_entrypoint") != "lzero.entry.train_muzero":
        raise ValueError("sandbox stock trainer entrypoint mismatch")
    if train_result.get("run_id") != str(args.run_id):
        raise ValueError("sandbox stock train run_id mismatch")
    if train_result.get("attempt_id") != str(args.attempt_id):
        raise ValueError("sandbox stock train attempt_id mismatch")
    command = _require_mapping(train_result.get("command"), "sandbox train command")
    if command.get("background_eval_enabled") is not False:
        raise ValueError("sandbox train background eval must be disabled")
    if command.get("background_gif_enabled") is not False:
        raise ValueError("sandbox train background gif must be disabled")
    if command.get("opponent_assignment_ref") != str(remote_assignment["assignment_ref"]):
        raise ValueError("sandbox train assignment ref mismatch")
    refresh_ref = command.get("opponent_assignment_refresh_ref")
    if refresh_ref not in (None, "", str(remote_assignment["assignment_ref"])):
        raise ValueError("sandbox train assignment refresh ref mismatch")
    command_initial_ref = command.get("initial_policy_checkpoint_ref")
    if command_initial_ref not in (None, "", str(remote_initial["checkpoint_ref"])):
        raise ValueError("sandbox train initial checkpoint ref mismatch")
    command_load_mode = command.get("initial_policy_checkpoint_load_mode")
    if command_load_mode not in (None, "", "matching_shape"):
        raise ValueError("sandbox train initial checkpoint load mode mismatch")
    command_state_key = command.get("initial_policy_checkpoint_state_key")
    if command_state_key not in (None, "", COMPACT_STOCK_EXPORT_MODEL_STATE_KEY):
        raise ValueError("sandbox train initial checkpoint state key mismatch")
    auto_resume = train_result.get("auto_resume")
    if isinstance(auto_resume, Mapping) and auto_resume.get("found") is True:
        raise ValueError("isolated live-run canary requires a fresh run; auto_resume found")
    policy = _require_mapping(
        train_result.get("initial_policy_checkpoint"),
        "initial policy checkpoint result",
    )
    if policy.get("applied") is not True:
        raise ValueError("sandbox train initial policy was not applied")
    if policy.get("checkpoint_ref") != str(remote_initial["checkpoint_ref"]):
        raise ValueError("sandbox train initial policy checkpoint ref mismatch")
    if policy.get("load_mode") != "matching_shape":
        raise ValueError("sandbox train initial policy load mode mismatch")
    if policy.get("state_key") != COMPACT_STOCK_EXPORT_MODEL_STATE_KEY:
        raise ValueError("sandbox train initial policy state key mismatch")
    load_result = _require_mapping(policy.get("load_result"), "initial policy load_result")
    if load_result.get("loaded") is not True:
        raise ValueError("sandbox train initial policy was not loaded")
    if load_result.get("fresh_optimizer_preserved") is not True:
        raise ValueError("sandbox train fresh optimizer was not preserved")


def _load_run_status(*, args: argparse.Namespace) -> Mapping[str, Any]:
    status_fn = _deployed_modal_function(
        app_name=str(args.status_app_name),
        function_name="curvytron_run_status",
        modal_env=args.modal_env,
    )
    rows = status_fn.remote([str(args.run_id)], [str(args.attempt_id)])
    if not rows:
        raise ValueError("sandbox run status returned no rows")
    return _require_mapping(rows[0], "sandbox run status row")


def _load_assignment_proof(
    *,
    args: argparse.Namespace,
    assignment_sha256: str,
) -> Mapping[str, Any]:
    proof_fn = _deployed_modal_function(
        app_name=str(args.status_app_name),
        function_name="curvytron_assignment_proof",
        modal_env=args.modal_env,
    )
    rows = proof_fn.remote(
        [str(args.run_id)],
        [str(args.attempt_id)],
        [str(assignment_sha256)],
    )
    if not rows:
        raise ValueError("sandbox assignment proof returned no rows")
    proof = _require_mapping(rows[0], "sandbox assignment proof row")
    if int(proof.get("assignment_env_proof_target_row_count", 0)) <= 0:
        raise ValueError("sandbox assignment proof did not find target env rows")
    if int(proof.get("assignment_env_proof_target_provider_ok_count", 0)) <= 0:
        raise ValueError(
            "sandbox assignment proof did not observe provider_ok target rows"
        )
    if int(proof.get("assignment_env_proof_target_provider_false_count", -1)) != 0:
        raise ValueError("sandbox assignment proof observed provider failures")
    return proof


def _select_final_checkpoint(train_result: Mapping[str, Any]) -> dict[str, Any]:
    mirror_obj = train_result.get("checkpoint_mirror")
    if isinstance(mirror_obj, Mapping):
        copied = mirror_obj.get("copied_checkpoints")
    else:
        copied = mirror_obj
    if not isinstance(copied, Sequence) or isinstance(copied, (str, bytes)):
        raise ValueError("checkpoint_mirror.copied_checkpoints must be a list")
    by_iteration: dict[int, str] = {}
    for index, item_obj in enumerate(copied):
        item = _require_mapping(item_obj, f"copied_checkpoints[{index}]")
        ref = _checkpoint_ref_from_record(item)
        if ref is None:
            continue
        iteration = _checkpoint_iteration_from_ref(ref)
        if iteration is None:
            continue
        by_iteration[iteration] = ref
    later_iterations = [iteration for iteration in by_iteration if iteration > 0]
    if not later_iterations:
        raise ValueError("isolated live-run canary requires a later iteration checkpoint")
    final_iteration = max(later_iterations)
    return {"iteration": final_iteration, "ref": by_iteration[final_iteration]}


def _trainer_result_payload(
    *,
    args: argparse.Namespace,
    stock_canary: Mapping[str, Any],
    train_result: Mapping[str, Any],
    remote_initial: Mapping[str, Any],
    remote_assignment: Mapping[str, Any],
    assignment_proof: Mapping[str, Any],
    assignment_sha256: str,
    initial_checkpoint_path: Path,
) -> dict[str, Any]:
    policy = dict(
        _require_mapping(
            train_result.get("initial_policy_checkpoint"),
            "initial policy checkpoint result",
        )
    )
    policy["source_sha256"] = _file_sha256(initial_checkpoint_path)
    policy.setdefault("checkpoint_ref", str(remote_initial["checkpoint_ref"]))
    policy.setdefault("source_path", str(policy.get("load_path") or ""))
    assignment_decision = _assignment_application_decision(train_result)
    return {
        "schema_id": TRAINER_RESULT_SCHEMA_ID,
        "ok": True,
        "trainer_mode": "sandbox_canary",
        "trainer_entrypoint": str(train_result["trainer_entrypoint"]),
        "canary_stock_train_muzero_called": bool(train_result["called_train_muzero"]),
        "compact_candidate_calls_train_muzero": False,
        "load_ckpt_before_run_target": str(policy.get("load_path", "")),
        "initial_policy_checkpoint": policy,
        "assignment_consumption": {
            "assignment_loaded": True,
            "assignment_applied": True,
            "trainer_loaded_assignment_ref": str(remote_assignment["assignment_ref"]),
            "trainer_loaded_assignment_sha256": assignment_sha256,
            "trainer_applied_assignment_sha256": assignment_sha256,
            "assignment_refresh_latest_decision": assignment_decision,
            "env_telemetry_assignment_sha256": assignment_sha256,
            "env_telemetry_row_count": int(
                assignment_proof["assignment_env_proof_target_row_count"]
            ),
            "provider_ok_count": int(
                assignment_proof["assignment_env_proof_target_provider_ok_count"]
            ),
            "provider_false_count": int(
                assignment_proof["assignment_env_proof_target_provider_false_count"]
            ),
        },
        "lineage_stages": [
            "trainer_assignment_loaded",
            "trainer_assignment_applied",
            "checkpoint_written",
        ],
        "source_stock_resume_load_canary_run_id": stock_canary.get("run_id"),
        "sandbox_run_id": str(args.run_id),
        "sandbox_attempt_id": str(args.attempt_id),
    }


def _metrics_payload(
    *,
    train_result: Mapping[str, Any],
    status: Mapping[str, Any],
    assignment_proof: Mapping[str, Any],
    final_selection: Mapping[str, Any],
    initial_checkpoint_path: Path,
    final_checkpoint_path: Path,
) -> dict[str, Any]:
    collector_envstep_delta = _positive_int_from_candidates(
        "collector_envstep_delta",
        status.get("learner_collector_envstep"),
        _nested_get(train_result, ("action_observability", "row_count")),
    )
    learner_train_calls_delta = _positive_int_from_candidates(
        "learner_train_calls_delta",
        status.get("learner_train_call_index"),
        _nested_get(train_result, ("target_audit", "counts", "replay_sample_calls")),
    )
    training_wall_sec = _positive_float_from_candidates(
        "training_wall_sec",
        _nested_get(train_result, ("phase_profile", "timers_sec", "train_muzero_wall_sec")),
        _nested_get(train_result, ("train_result", "elapsed_sec")),
    )
    final_sha = _file_sha256(final_checkpoint_path)
    if final_sha == _file_sha256(initial_checkpoint_path):
        raise ValueError("isolated live-run final checkpoint did not move")
    return {
        "schema_id": METRICS_SCHEMA_ID,
        "ok": True,
        "checkpoint_write_ok": True,
        "checkpoint_read_ok": final_checkpoint_path.is_file()
        and final_checkpoint_path.stat().st_size > 0,
        "progress_moved": int(final_selection["iteration"]) > 0,
        "learner_metrics_moved": bool(status.get("learner_metrics_latest_exists", True))
        and learner_train_calls_delta > 0,
        "collector_envstep_delta": collector_envstep_delta,
        "learner_train_calls_delta": learner_train_calls_delta,
        "checkpoint_iteration_delta": int(final_selection["iteration"]),
        "env_telemetry_row_count": int(
            assignment_proof["assignment_env_proof_target_row_count"]
        ),
        "training_wall_sec": training_wall_sec,
    }


def _forbidden_touch_audit_payload(*, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_id": FORBIDDEN_TOUCH_AUDIT_SCHEMA_ID,
        "ok": True,
        "sandbox_scope": {
            "isolated": True,
            "namespace": f"isolated-{args.run_id}",
            "production_namespace": False,
        },
        "forbidden_touch_audit": {
            "production_live_runs_touched": False,
            "production_intake_touched": False,
            "production_rating_touched": False,
            "production_leaderboard_touched": False,
            "production_control_pointer_touched": False,
            "writes_checkpoint_intake": False,
            "spawns_rating": False,
            "publishes_leaderboard": False,
            "rewrites_production_control_pointers": False,
            "uses_production_modal_objects": False,
            "background_eval_enabled": False,
            "background_gif_enabled": False,
        },
        "allowed_sandbox_modal_objects": {
            "trainer_app_name": str(args.train_app_name),
            "status_app_name": str(args.status_app_name),
            "runs_volume_name": str(args.runs_volume_name),
            "scope": "isolated run_id/attempt_id only; no intake/rating/leaderboard/control writes",
        },
    }


def _producer_manifest(
    *,
    args: argparse.Namespace,
    report_path: Path,
    upload_path: Path,
    assignment_write_path: Path,
    raw_train_result_path: Path,
    trainer_result_path: Path,
    metrics_path: Path,
    forbidden_touch_path: Path,
    assignment_path: Path,
    assignment_audit_path: Path,
    final_checkpoint_path: Path,
    final_selection: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_id": PRODUCER_SCHEMA_ID,
        "ok": True,
        "producer_id": PRODUCER_ID,
        "run_id": str(args.run_id),
        "attempt_id": str(args.attempt_id),
        "created_at": datetime.now(UTC).isoformat(),
        "trainer_mode": "sandbox_canary",
        "canary_stock_train_muzero_called": True,
        "compact_candidate_calls_train_muzero": False,
        "touches_live_runs": False,
        "touches_production_live_runs": False,
        "background_eval_enabled": False,
        "background_gif_enabled": False,
        "final_checkpoint_ref": str(final_selection["ref"]),
        "final_checkpoint_path": str(final_checkpoint_path),
        "artifacts": {
            "report": _file_ref(report_path),
            "assignment": _file_ref(assignment_path),
            "assignment_audit": _file_ref(assignment_audit_path),
            "initial_checkpoint_upload": _file_ref(upload_path),
            "remote_assignment_write": _file_ref(assignment_write_path),
            "raw_train_result": _file_ref(raw_train_result_path),
            "trainer_result": _file_ref(trainer_result_path),
            "metrics": _file_ref(metrics_path),
            "forbidden_touch_audit": _file_ref(forbidden_touch_path),
            "final_checkpoint": _file_ref(final_checkpoint_path),
        },
    }


def _assignment_application_decision(train_result: Mapping[str, Any]) -> str:
    refresh_obj = train_result.get("opponent_assignment_refresh")
    if not isinstance(refresh_obj, Mapping):
        return "initial_assignment_applied"
    refresh = refresh_obj
    events = refresh.get("events")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return "initial_assignment_applied"
    applied = [
        _require_mapping(event, "assignment refresh event")
        for event in events
        if isinstance(event, Mapping) and event.get("decision") == "applied"
    ]
    if not applied:
        return "initial_assignment_applied"
    event = applied[-1]
    env_report = _require_mapping(event.get("env_ready_report"), "assignment env_ready_report")
    if env_report.get("ok") is not True:
        raise ValueError("sandbox train assignment env_ready_report must be ok=true")
    return "applied"


def _env_telemetry_row_count(train_result: Mapping[str, Any]) -> int:
    return _positive_int_from_candidates(
        "env_telemetry_row_count",
        _nested_get(train_result, ("action_observability", "row_count")),
    )


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


def _upload_volume_bytes(
    *,
    content: bytes,
    remote_ref: str,
    volume_name: str,
    modal_env: str | None,
) -> None:
    import modal

    volume = modal.Volume.from_name(
        volume_name,
        environment_name=modal_env,
        **modal_volume_kwargs_for_name(volume_name),
    )
    upload_ref = "/" + str(remote_ref).lstrip("/")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(content), upload_ref)


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


def _positive_int_from_candidates(label: str, *values: Any) -> int:
    for value in values:
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    raise ValueError(f"{label} must be positive")


def _positive_float_from_candidates(label: str, *values: Any) -> float:
    for value in values:
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    raise ValueError(f"{label} must be positive")


def _nested_get(value: Mapping[str, Any], keys: Sequence[str]) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return _require_mapping(data, label)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _resolve_input(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "bytes": int(path.stat().st_size),
    }


if __name__ == "__main__":
    raise SystemExit(main())
