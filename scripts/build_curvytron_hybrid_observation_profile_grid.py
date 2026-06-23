#!/usr/bin/env python3
"""Build raw Modal commands for the hybrid observation profile canary.

This is deliberately separate from the stock-train profile manifest builder.
The hybrid canary does not call ``train_muzero`` and uses the boundary profile
app's compute names: ``gpu-h100`` and ``gpu-l4-t4``.
"""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODULE = "curvyzero.infra.modal.source_state_batched_observation_boundary_profile"
DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_hybrid_observation_profile_manifests")
LAUNCH_MODE_BLOCKING_STDOUT_JSON = "blocking_stdout_json"
LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT = "detached_function_call_result"
RESULT_CAPTURE_STDOUT_JSON = "stdout_json"
RESULT_CAPTURE_MODAL_FUNCTION_CALL_GET = "modal_function_call_get"
COMPUTE_ALIASES = {
    "h100": "gpu-h100",
    "gpu-h100": "gpu-h100",
    "l4": "gpu-l4-t4",
    "gpu-l4": "gpu-l4-t4",
    "gpu-l4-t4": "gpu-l4-t4",
}
COMPUTE_CHOICES = {"gpu-h100", "gpu-l4-t4"}
DEATH_MODE_PROFILE_NO_DEATH = "profile_no_death"
DEATH_MODE_NORMAL = "normal"
DEATH_MODE_CHOICES = (DEATH_MODE_PROFILE_NO_DEATH, DEATH_MODE_NORMAL)
MCTS_ARRAYS_BOUNDARY_IMPL_CHOICES = (
    "stock_facade",
    "direct_ctree_arrays",
    "direct_ctree_gpu_latent",
    "direct_ctree_gpu_latent_precomputed_recurrent",
)
LIGHTZERO_ARRAY_CEILING_MODE_CHOICES = (
    "policy_arrays",
    "mock_search_service",
    "service_tax_probe",
    "recurrent_toy",
    "dense_torch_mcts",
    "dense_torch_mcts_compile_spike",
    "compact_torch_search_service",
    "fixed_shape_search_owner",
)
COMPACT_TORCH_MODEL_COMPILE_MODE_CHOICES = (
    "default",
    "reduce-overhead",
    "max-autotune-no-cudagraphs",
    "max-autotune",
)
COMPACT_TORCH_INITIAL_INFERENCE_MODE_CHOICES = ("model_method", "direct_core")
COMPACT_TORCH_MEMORY_FORMAT_CHOICES = ("contiguous", "channels_last")
COMPACT_REPLAY_ARRAY_CEILING_MODES = frozenset(
    {
        "mock_search_service",
        "service_tax_probe",
        "dense_torch_mcts",
        "dense_torch_mcts_compile_spike",
        "compact_torch_search_service",
        "fixed_shape_search_owner",
    }
)
NEXT_DIRECT_CTREE_COMPARISON_IMPLS = [
    "stock_facade",
    "direct_ctree_arrays",
    "direct_ctree_gpu_latent",
]
COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_CHOICES = (
    "toy_probe",
    "compact_muzero",
)
MATCHED_DENOMINATOR_ID = "curvytron-stock-vs-compact-owned-no-rnd-h100-20260528"
MATCHED_DENOMINATOR_ROW_PURPOSE = "matched_denominator_speed"
MATCHED_COMPACT_SPEED_CURRENCY = "compact_profile_active_roots_per_sec"
MATCHED_STOCK_MANIFEST_REF = (
    "artifacts/local/curvytron_optimizer_profile_manifests/"
    "optimizer-matched-denominator-stock-20260528.json"
)


def _csv_strings(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def _csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _csv_bools(raw: str) -> list[bool]:
    out: list[bool] = []
    for part in _csv_strings(raw):
        lowered = part.lower()
        if lowered in {"1", "true", "yes", "on"}:
            out.append(True)
        elif lowered in {"0", "false", "no", "off"}:
            out.append(False)
        else:
            raise argparse.ArgumentTypeError(
                f"expected booleans like true,false or yes,no; got {part!r}"
            )
    return out


def _csv_computes(raw: str) -> list[str]:
    values: list[str] = []
    for part in _csv_strings(raw):
        compute = COMPUTE_ALIASES.get(part, part)
        if compute not in COMPUTE_CHOICES:
            allowed = ", ".join(sorted(set(COMPUTE_ALIASES).union(COMPUTE_CHOICES)))
            raise argparse.ArgumentTypeError(
                f"unknown compute {part!r}; expected one of: {allowed}"
            )
        values.append(compute)
    return values


def _csv_mcts_arrays_boundary_impls(raw: str) -> list[str]:
    values: list[str] = []
    for part in _csv_strings(raw):
        if part not in MCTS_ARRAYS_BOUNDARY_IMPL_CHOICES:
            allowed = ", ".join(MCTS_ARRAYS_BOUNDARY_IMPL_CHOICES)
            raise argparse.ArgumentTypeError(
                f"unknown MCTS arrays boundary impl {part!r}; expected one of: {allowed}"
            )
        values.append(part)
    return values


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_id(raw: str) -> str:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")
    clean = "".join(char if char in allowed else "-" for char in raw).strip("-.")
    if not clean or not clean[0].isalnum():
        raise ValueError(f"cannot make a safe id from {raw!r}")
    return clean


def apply_next_direct_ctree_comparison_preset(args: argparse.Namespace) -> None:
    """Mutate parsed args to the current fixed-denominator comparison grid."""

    args.computes = ["gpu-h100", "gpu-l4-t4"] if args.include_l4 else ["gpu-h100"]
    args.batch_sizes = [512]
    args.actor_count = 16
    args.steps = 60
    args.warmup_steps = 15
    args.probe_simulations = [8]
    args.materialize_scalar_timestep = [False]
    args.device_latest = [False]
    args.resident_chunk_probe = False
    args.lightzero_collect_forward_probe = False
    args.lightzero_initial_inference_probe = False
    args.lightzero_array_ceiling_probe = False
    args.lightzero_mcts_arrays_boundary_probe = True
    args.lightzero_mcts_arrays_boundary_impls = NEXT_DIRECT_CTREE_COMPARISON_IMPLS
    args.lightzero_mcts_arrays_boundary_input_mode = "host_uint8"


def apply_next_fixed_root_tape_preset(args: argparse.Namespace) -> None:
    """Mutate parsed args to the current fixed-root-tape comparator smoke grid."""

    args.computes = ["gpu-h100", "gpu-l4-t4"] if args.include_l4 else ["gpu-h100"]
    args.batch_sizes = [128]
    args.actor_count = 8
    args.steps = 20
    args.warmup_steps = 10
    args.probe_simulations = [8]
    args.materialize_scalar_timestep = [False]
    args.device_latest = [False]
    args.resident_chunk_probe = False
    args.compact_rollout_slab_probe = True
    args.compact_root_tape_compare = True
    args.compact_root_tape_max_records = 4
    args.compact_root_tape_compare_fixed_shape_floor = True
    args.compact_root_tape_reference_label = "primary"
    args.hybrid_resident_observation_search = False
    args.lightzero_collect_forward_probe = False
    args.lightzero_initial_inference_probe = False
    args.lightzero_mcts_arrays_boundary_probe = False
    args.mctx_compact_search_probe = False
    args.lightzero_array_ceiling_probe = True
    args.lightzero_array_ceiling_mode = "compact_torch_search_service"
    args.lightzero_array_ceiling_input_mode = "host_uint8"
    args.lightzero_consumer_root_noise_weight = 0.0


def apply_next_fixed_root_tape_large_preset(args: argparse.Namespace) -> None:
    """Mutate parsed args to the next durable fixed-root-tape H100 grid."""

    apply_next_fixed_root_tape_preset(args)
    args.computes = ["gpu-h100"]
    args.batch_sizes = [512]
    args.actor_count = 16
    args.steps = 60
    args.warmup_steps = 15
    args.compact_root_tape_max_records = 16
    args.launch_mode = LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT
    args.result_capture = RESULT_CAPTURE_MODAL_FUNCTION_CALL_GET
    args.row_timeout_sec = 300
    args.result_timeout_sec = 7200
    args.captured_result_required = True


def apply_next_fixed_root_tape_mctx_preset(args: argparse.Namespace) -> None:
    """Mutate parsed args to the next fixed-root MCTX comparator H100 grid."""

    apply_next_fixed_root_tape_large_preset(args)
    args.compact_root_tape_compare_mctx = True


def apply_next_fixed_root_tape_compile_preset(args: argparse.Namespace) -> None:
    """Mutate parsed args to the next fixed-root eager-vs-compile comparator."""

    apply_next_fixed_root_tape_preset(args)
    args.computes = ["gpu-h100"]
    args.probe_simulations = [1]
    args.compact_root_tape_compare_fixed_shape_floor = False
    args.compact_root_tape_compare_model_compile = True
    args.compact_root_tape_model_compile_mode = "default"
    args.compact_root_tape_require_model_compile = True
    args.compact_torch_compile_model_inference = False
    args.compact_torch_require_model_compile = False
    args.compact_torch_model_compile_mode = "reduce-overhead"
    args.compact_torch_compile_search = False


def apply_next_fixed_root_tape_direct_core_preset(args: argparse.Namespace) -> None:
    """Mutate parsed args to the next public-vs-direct-core root-tape gate."""

    apply_next_fixed_root_tape_preset(args)
    args.computes = ["gpu-h100"]
    args.probe_simulations = [1]
    args.compact_root_tape_compare_fixed_shape_floor = False
    args.compact_root_tape_compare_direct_core = True
    args.compact_torch_initial_inference_mode = "model_method"
    args.compact_torch_compile_model_inference = False
    args.compact_torch_require_model_compile = False
    args.compact_torch_model_compile_mode = "reduce-overhead"
    args.compact_torch_compile_search = False


def apply_next_terminal_nstep_compact_owned_preset(args: argparse.Namespace) -> None:
    """Mutate parsed args to the next terminal-safe compact-owned H100 row."""

    args.computes = ["gpu-h100"]
    args.batch_sizes = [128]
    args.actor_count = 8
    args.steps = 6
    args.warmup_steps = 0
    args.max_ticks = 3
    args.probe_simulations = [8]
    args.materialize_scalar_timestep = [False]
    args.device_latest = [False]
    args.resident_chunk_probe = False
    args.compact_service_replay_proof = False
    args.compact_rollout_slab_probe = True
    args.compact_rollout_slab_sample_gate = True
    args.compact_rollout_slab_sample_gate_batch_size = 0
    args.compact_rollout_slab_sample_gate_interval = 1
    args.compact_rollout_slab_sample_gate_replay_pair_capacity = 64
    args.compact_rollout_slab_learner_gate = True
    args.compact_rollout_slab_learner_gate_train_steps = 1
    args.compact_rollout_slab_learner_gate_device = "cuda"
    args.compact_rollout_slab_learner_gate_include_rnd = False
    args.compact_rollout_slab_learner_gate_impl = "compact_muzero"
    args.compact_rollout_slab_learner_gate_support_scale = 9
    args.compact_rollout_slab_learner_gate_num_unroll_steps = 2
    args.compact_rollout_slab_action_mode = "search_feedback"
    args.compact_owned_loop_entrypoint = True
    args.compact_owned_loop_policy_version_ref = "terminal-nstep-compact-owned-policy-v1"
    args.compact_owned_loop_model_version_ref = "terminal-nstep-compact-owned-model-v1"
    args.compact_owned_loop_policy_source = "terminal_nstep_compact_owned_profile"
    args.compact_owned_loop_capture_replay_store_state = True
    args.compact_root_tape_compare = False
    args.compact_root_tape_compare_mctx = False
    args.hybrid_device_only_stack = True
    args.hybrid_refresh_observation_stack = True
    args.hybrid_resident_observation_search = True
    args.hybrid_native_actor_buffer = True
    args.hybrid_persistent_compact_render_state_buffer = False
    args.lightzero_collect_forward_probe = False
    args.lightzero_initial_inference_probe = False
    args.lightzero_mcts_arrays_boundary_probe = False
    args.mctx_compact_search_probe = False
    args.lightzero_array_ceiling_probe = True
    args.lightzero_array_ceiling_mode = "compact_torch_search_service"
    args.lightzero_array_ceiling_input_mode = "host_uint8"
    args.lightzero_consumer_root_noise_weight = 0.0
    args.require_terminal_compact_owned_nstep = True
    args.launch_mode = LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT
    args.result_capture = RESULT_CAPTURE_MODAL_FUNCTION_CALL_GET
    args.row_timeout_sec = 300
    args.result_timeout_sec = 7200
    args.captured_result_required = True


def apply_next_normal_death_compact_owned_preset(args: argparse.Namespace) -> None:
    """Mutate parsed args to the next normal-death compact-owned H100 row."""

    apply_next_terminal_nstep_compact_owned_preset(args)
    args.death_mode = DEATH_MODE_NORMAL
    args.steps = 64
    args.max_ticks = 2000
    args.compact_rollout_slab_sample_gate_batch_size = 256
    args.compact_rollout_slab_sample_gate_interval = 8
    args.compact_rollout_slab_sample_gate_replay_pair_capacity = 8192
    args.compact_rollout_slab_learner_gate_support_scale = 300
    args.compact_torch_compile_search = False
    args.require_normal_death_terminal_contract = True
    args.normal_death_terminal_contract_evidence_id = "optimizer-normal-death-compact-owned-profile"
    args.normal_death_terminal_contract_evidence_refs = [
        "source_collision_head_head_reverse_order_single_death_step.json",
        "payload-derived-normal-death-evidence-local-20260530",
    ]


def apply_next_borrowed_normal_death_compact_owned_preset(args: argparse.Namespace) -> None:
    """Mutate parsed args to the next borrowed render-state normal-death row."""

    apply_next_normal_death_compact_owned_preset(args)
    args.batch_sizes = [1024]
    args.actor_count = 1
    args.compact_rollout_slab_sample_gate_batch_size = 512
    args.hybrid_borrow_single_actor_render_state = True
    args.compact_torch_initial_inference_mode = "direct_core"


def apply_next_matched_denominator_compact_owned_preset(
    args: argparse.Namespace,
) -> None:
    """Mutate parsed args to the selected compact side of the denominator pair."""

    args.computes = ["gpu-h100"]
    args.batch_sizes = [1024]
    args.actor_count = 16
    args.steps = 60
    args.warmup_steps = 15
    args.max_ticks = 2000
    args.probe_simulations = [8]
    args.materialize_scalar_timestep = [False]
    args.device_latest = [False]
    args.resident_chunk_probe = False
    args.compact_service_replay_proof = False
    args.compact_rollout_slab_probe = True
    args.compact_rollout_slab_sample_gate = True
    args.compact_rollout_slab_sample_gate_batch_size = 512
    args.compact_rollout_slab_sample_gate_interval = 8
    args.compact_rollout_slab_sample_gate_replay_pair_capacity = 4096
    args.compact_rollout_slab_learner_gate = True
    args.compact_rollout_slab_learner_gate_train_steps = 1
    args.compact_rollout_slab_learner_gate_device = "cuda"
    args.compact_rollout_slab_learner_gate_include_rnd = False
    args.compact_rollout_slab_learner_gate_impl = "compact_muzero"
    args.compact_rollout_slab_learner_gate_support_scale = 300
    args.compact_rollout_slab_learner_gate_num_unroll_steps = 1
    args.compact_rollout_slab_action_mode = "search_feedback"
    args.compact_owned_loop_entrypoint = True
    args.compact_owned_loop_policy_version_ref = "matched-denominator-compact-owned-policy-v1"
    args.compact_owned_loop_model_version_ref = "matched-denominator-compact-owned-model-v1"
    args.compact_owned_loop_policy_source = "matched_denominator_compact_owned_profile"
    args.compact_owned_loop_capture_replay_store_state = True
    args.compact_root_tape_compare = False
    args.compact_root_tape_compare_mctx = False
    args.hybrid_device_only_stack = True
    args.hybrid_refresh_observation_stack = True
    args.hybrid_resident_observation_search = True
    args.hybrid_native_actor_buffer = True
    args.hybrid_persistent_compact_render_state_buffer = False
    args.lightzero_collect_forward_probe = False
    args.lightzero_initial_inference_probe = False
    args.lightzero_mcts_arrays_boundary_probe = False
    args.mctx_compact_search_probe = False
    args.lightzero_array_ceiling_probe = True
    args.lightzero_array_ceiling_mode = "compact_torch_search_service"
    args.lightzero_array_ceiling_input_mode = "host_uint8"
    args.lightzero_consumer_root_noise_weight = 0.0
    args.require_terminal_compact_owned_nstep = False
    args.launch_mode = LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT
    args.result_capture = RESULT_CAPTURE_MODAL_FUNCTION_CALL_GET
    args.row_timeout_sec = 300
    args.result_timeout_sec = 7200
    args.captured_result_required = True
    args.matched_denominator_id = MATCHED_DENOMINATOR_ID
    args.matched_pair_role = "compact_candidate"
    args.matched_speed_currency = MATCHED_COMPACT_SPEED_CURRENCY
    args.matched_counterpart_manifest_ref = MATCHED_STOCK_MANIFEST_REF
    args.matched_counterpart_row_id = "001"
    args.matched_row_purpose = MATCHED_DENOMINATOR_ROW_PURPOSE
    args.matched_promotion_claim = False


def _command(args: argparse.Namespace, *, row: dict[str, Any]) -> list[str]:
    synthetic_probe_simulations = (
        0
        if (
            args.lightzero_collect_forward_probe
            or args.lightzero_initial_inference_probe
            or args.lightzero_array_ceiling_probe
            or args.lightzero_mcts_arrays_boundary_probe
            or args.mctx_compact_search_probe
        )
        else row["probe_simulations"]
    )
    command = [
        "uv",
        "run",
        "--extra",
        "modal",
        "modal",
        "run",
        "-m",
        MODULE,
        "--hybrid-observation-canary",
        "--compute",
        row["compute"],
        "--batch-size",
        str(row["batch_size"]),
        "--actor-count",
        str(row["actor_count"]),
        "--steps",
        str(row["steps"]),
        "--warmup-steps",
        str(row["warmup_steps"]),
        "--max-ticks",
        str(row["max_ticks"]),
        "--death-mode",
        str(row["death_mode"]),
        "--trail-slots",
        str(row["trail_slots"]),
        "--body-capacity",
        str(row["body_capacity"]),
        "--render-surface",
        args.render_surface,
        "--observation-renderer-backend",
        args.observation_renderer_backend,
        "--hybrid-stack-storage-dtype",
        args.stack_storage_dtype,
        "--hybrid-batched-stack-probe-simulations",
        str(synthetic_probe_simulations),
        "--hybrid-batched-stack-probe-channels",
        str(args.probe_channels),
    ]
    if row.get("launch_mode") == LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT:
        modal_run_index = next(
            index
            for index, part in enumerate(command[:-1])
            if part == "modal" and command[index + 1] == "run"
        )
        command.insert(modal_run_index + 2, "--detach")
        command.append("--hybrid-profile-spawn-result")
    command.append(
        "--hybrid-materialize-scalar-timestep"
        if row["materialize_scalar_timestep"]
        else "--no-hybrid-materialize-scalar-timestep"
    )
    if row["device_latest"]:
        command.append("--hybrid-batched-stack-probe-device-latest")
    if row.get("compact_service_replay_proof", False):
        command.append("--hybrid-compact-service-replay-proof")
    if row.get("compact_rollout_slab_probe", False):
        command.append("--hybrid-compact-rollout-slab-probe")
        command.extend(
            [
                "--hybrid-compact-rollout-slab-action-mode",
                str(row["compact_rollout_slab_action_mode"]),
            ]
        )
    if row.get("compact_root_tape_compare", False):
        command.append("--hybrid-compact-root-tape-compare")
        command.extend(
            [
                "--hybrid-compact-root-tape-max-records",
                str(row["compact_root_tape_max_records"]),
                "--hybrid-compact-root-tape-reference-label",
                str(row["compact_root_tape_reference_label"]),
            ]
        )
        if row.get("compact_root_tape_allow_resident_host_snapshot", False):
            command.append("--hybrid-compact-root-tape-allow-resident-host-snapshot")
        if not row.get("compact_root_tape_compare_fixed_shape_floor", True):
            command.append("--no-hybrid-compact-root-tape-compare-fixed-shape-floor")
        if row.get("compact_root_tape_compare_mctx", False):
            command.append("--hybrid-compact-root-tape-compare-mctx")
            if not args.mctx_compact_search_probe:
                command.extend(
                    [
                        "--hybrid-mctx-num-simulations",
                        str(row["probe_simulations"]),
                        "--hybrid-mctx-hidden-dim",
                        str(args.mctx_hidden_dim),
                        "--hybrid-mctx-visual-channels",
                        str(args.mctx_visual_channels),
                    ]
                )
        if row.get("compact_root_tape_compare_model_compile", False):
            command.append("--hybrid-compact-root-tape-compare-model-compile")
            command.extend(
                [
                    "--hybrid-compact-root-tape-model-compile-mode",
                    str(row["compact_root_tape_model_compile_mode"]),
                ]
            )
            if not row.get("compact_root_tape_require_model_compile", True):
                command.append("--no-hybrid-compact-root-tape-require-model-compile")
                if not args.mctx_require_gpu_backend:
                    command.append("--no-hybrid-mctx-require-gpu-backend")
        if row.get("compact_root_tape_compare_direct_core", False):
            command.append("--hybrid-compact-root-tape-compare-direct-core")
    if row.get("compact_rollout_slab_sample_gate", False):
        command.append("--hybrid-compact-rollout-slab-sample-gate")
        command.extend(
            [
                "--hybrid-compact-rollout-slab-sample-gate-batch-size",
                str(args.compact_rollout_slab_sample_gate_batch_size),
                "--hybrid-compact-rollout-slab-sample-gate-interval",
                str(args.compact_rollout_slab_sample_gate_interval),
                "--hybrid-compact-rollout-slab-sample-gate-replay-pair-capacity",
                str(args.compact_rollout_slab_sample_gate_replay_pair_capacity),
            ]
        )
    if row.get("compact_rollout_slab_learner_gate", False):
        command.append("--hybrid-compact-rollout-slab-learner-gate")
        command.extend(
            [
                "--hybrid-compact-rollout-slab-learner-gate-train-steps",
                str(args.compact_rollout_slab_learner_gate_train_steps),
                "--hybrid-compact-rollout-slab-learner-gate-device",
                args.compact_rollout_slab_learner_gate_device,
                "--hybrid-compact-rollout-slab-learner-gate-impl",
                args.compact_rollout_slab_learner_gate_impl,
                "--hybrid-compact-rollout-slab-learner-gate-support-scale",
                str(args.compact_rollout_slab_learner_gate_support_scale),
                "--hybrid-compact-rollout-slab-learner-gate-num-unroll-steps",
                str(args.compact_rollout_slab_learner_gate_num_unroll_steps),
            ]
        )
        if args.compact_rollout_slab_learner_gate_include_rnd:
            command.append("--hybrid-compact-rollout-slab-learner-gate-include-rnd")
    if row.get("compact_owned_loop_entrypoint", False):
        command.append("--hybrid-compact-owned-loop-entrypoint")
        command.extend(
            [
                "--hybrid-compact-owned-loop-policy-version-ref",
                str(row["compact_owned_loop_policy_version_ref"]),
                "--hybrid-compact-owned-loop-policy-source",
                str(row["compact_owned_loop_policy_source"]),
            ]
        )
        if row.get("compact_owned_loop_model_version_ref"):
            command.extend(
                [
                    "--hybrid-compact-owned-loop-model-version-ref",
                    str(row["compact_owned_loop_model_version_ref"]),
                ]
            )
        if row.get("compact_owned_loop_capture_replay_store_state", False):
            command.append("--hybrid-compact-owned-loop-capture-replay-store-state")
    if args.hybrid_device_only_stack:
        command.append("--hybrid-device-only-stack")
    if not args.hybrid_refresh_observation_stack:
        command.append("--no-hybrid-refresh-observation-stack")
    if args.hybrid_resident_observation_search:
        command.append("--hybrid-resident-observation-search")
    if args.hybrid_native_actor_buffer:
        command.append("--hybrid-native-actor-buffer")
    if args.hybrid_persistent_compact_render_state_buffer:
        command.append("--hybrid-persistent-compact-render-state-buffer")
    if args.hybrid_borrow_single_actor_render_state:
        command.append("--hybrid-borrow-single-actor-render-state")
    if args.mctx_compact_search_probe:
        command.extend(
            [
                "--hybrid-mctx-compact-search-probe",
                "--hybrid-mctx-num-simulations",
                str(row["probe_simulations"]),
                "--hybrid-mctx-hidden-dim",
                str(args.mctx_hidden_dim),
                "--hybrid-mctx-visual-channels",
                str(args.mctx_visual_channels),
            ]
        )
        if not args.mctx_require_gpu_backend:
            command.append("--no-hybrid-mctx-require-gpu-backend")
    if row["resident_chunk_probe"]:
        command.extend(
            [
                "--hybrid-resident-chunk-probe",
                "--hybrid-resident-replay-steps",
                str(args.resident_replay_steps),
                "--hybrid-resident-sample-batch-size",
                str(args.resident_sample_batch_size),
                "--hybrid-resident-replay-train-steps",
                str(args.resident_replay_train_steps),
            ]
        )
        if args.no_resident_readback_checksum:
            command.append("--no-hybrid-resident-readback-checksum")
    if (
        args.lightzero_collect_forward_probe
        or args.lightzero_initial_inference_probe
        or args.lightzero_array_ceiling_probe
        or args.lightzero_mcts_arrays_boundary_probe
    ):
        if args.lightzero_collect_forward_probe:
            command.extend(
                [
                    "--hybrid-lightzero-collect-forward-probe",
                    "--hybrid-lightzero-consumer-num-simulations",
                    str(row["probe_simulations"]),
                    "--hybrid-lightzero-consumer-temperature",
                    str(args.lightzero_consumer_temperature),
                    "--hybrid-lightzero-consumer-epsilon",
                    str(args.lightzero_consumer_epsilon),
                ]
            )
        elif args.lightzero_initial_inference_probe:
            command.append("--hybrid-lightzero-initial-inference-probe")
        elif args.lightzero_array_ceiling_probe:
            command.extend(
                [
                    "--hybrid-lightzero-array-ceiling-probe",
                    "--hybrid-lightzero-array-ceiling-mode",
                    args.lightzero_array_ceiling_mode,
                    "--hybrid-lightzero-array-ceiling-input-mode",
                    args.lightzero_array_ceiling_input_mode,
                ]
            )
            if args.lightzero_mock_service_materialize_public_output:
                command.append("--hybrid-lightzero-mock-service-materialize-public-output")
        else:
            command.extend(
                [
                    "--hybrid-lightzero-mcts-arrays-boundary-probe",
                    "--hybrid-lightzero-mcts-arrays-boundary-impl",
                    row["lightzero_mcts_arrays_boundary_impl"],
                    "--hybrid-lightzero-mcts-arrays-boundary-input-mode",
                    args.lightzero_mcts_arrays_boundary_input_mode,
                ]
            )
        command.extend(
            ["--hybrid-lightzero-consumer-num-simulations", str(row["probe_simulations"])]
            if (
                args.lightzero_initial_inference_probe
                or args.lightzero_array_ceiling_probe
                or args.lightzero_mcts_arrays_boundary_probe
            )
            else []
        )
        if args.lightzero_consumer_root_noise_weight >= 0.0:
            command.extend(
                [
                    "--hybrid-lightzero-consumer-root-noise-weight",
                    str(args.lightzero_consumer_root_noise_weight),
                ]
            )
        if not args.lightzero_consumer_use_cuda:
            command.append("--no-hybrid-lightzero-consumer-use-cuda")
        if args.lightzero_consumer_collect_with_pure_policy:
            command.append("--hybrid-lightzero-consumer-collect-with-pure-policy")
        if getattr(args, "compact_torch_compile_model_inference", False):
            command.append("--hybrid-compact-torch-compile-model-inference")
        if getattr(args, "compact_torch_require_model_compile", False):
            command.append("--hybrid-compact-torch-require-model-compile")
        model_compile_mode = getattr(
            args,
            "compact_torch_model_compile_mode",
            "reduce-overhead",
        )
        if model_compile_mode != "reduce-overhead":
            command.extend(
                [
                    "--hybrid-compact-torch-model-compile-mode",
                    model_compile_mode,
                ]
            )
        if not getattr(args, "compact_torch_compile_search", True):
            command.append("--no-hybrid-compact-torch-compile-search")
        recurrent_action_shape_mode = getattr(
            args,
            "compact_torch_recurrent_action_shape_mode",
            "auto",
        )
        if recurrent_action_shape_mode != "auto":
            command.extend(
                [
                    "--hybrid-compact-torch-recurrent-action-shape-mode",
                    recurrent_action_shape_mode,
                ]
            )
        initial_inference_mode = getattr(
            args,
            "compact_torch_initial_inference_mode",
            "model_method",
        )
        if initial_inference_mode != "model_method":
            command.extend(
                [
                    "--hybrid-compact-torch-initial-inference-mode",
                    initial_inference_mode,
                ]
            )
        observation_memory_format = getattr(
            args,
            "compact_torch_observation_memory_format",
            "contiguous",
        )
        if observation_memory_format != "contiguous":
            command.extend(
                [
                    "--hybrid-compact-torch-observation-memory-format",
                    observation_memory_format,
                ]
            )
        model_memory_format = getattr(
            args,
            "compact_torch_model_memory_format",
            "contiguous",
        )
        if model_memory_format != "contiguous":
            command.extend(
                [
                    "--hybrid-compact-torch-model-memory-format",
                    model_memory_format,
                ]
            )
    if args.quiet:
        command.insert(6, "--quiet")
    return command


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.launch_mode not in {
        LAUNCH_MODE_BLOCKING_STDOUT_JSON,
        LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT,
    }:
        raise ValueError("unknown launch_mode")
    if args.result_capture not in {
        RESULT_CAPTURE_STDOUT_JSON,
        RESULT_CAPTURE_MODAL_FUNCTION_CALL_GET,
    }:
        raise ValueError("unknown result_capture")
    if (
        args.launch_mode == LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT
        and args.result_capture != RESULT_CAPTURE_MODAL_FUNCTION_CALL_GET
    ):
        raise ValueError(
            "detached_function_call_result requires result_capture=modal_function_call_get"
        )
    if (
        args.launch_mode == LAUNCH_MODE_BLOCKING_STDOUT_JSON
        and args.result_capture != RESULT_CAPTURE_STDOUT_JSON
    ):
        raise ValueError("blocking_stdout_json requires result_capture=stdout_json")
    if args.row_timeout_sec is not None and args.row_timeout_sec <= 0:
        raise ValueError("--row-timeout-sec must be positive when set")
    if args.result_timeout_sec is not None and args.result_timeout_sec <= 0:
        raise ValueError("--result-timeout-sec must be positive when set")
    if args.max_ticks <= 0:
        raise ValueError("--max-ticks must be positive")
    lightzero_probe_count = sum(
        bool(value)
        for value in (
            args.lightzero_collect_forward_probe,
            args.lightzero_initial_inference_probe,
            args.lightzero_array_ceiling_probe,
            args.lightzero_mcts_arrays_boundary_probe,
        )
    )
    if lightzero_probe_count > 1:
        raise ValueError("at most one LightZero probe flag may be enabled")
    if args.mctx_compact_search_probe and lightzero_probe_count:
        raise ValueError("--mctx-compact-search-probe cannot be combined with LightZero probes")
    lightzero_or_mctx_probe = bool(lightzero_probe_count or args.mctx_compact_search_probe)
    if args.lightzero_mock_service_materialize_public_output and (
        not args.lightzero_array_ceiling_probe
        or args.lightzero_array_ceiling_mode != "mock_search_service"
    ):
        raise ValueError(
            "--lightzero-mock-service-materialize-public-output requires "
            "--lightzero-array-ceiling-probe and "
            "--lightzero-array-ceiling-mode=mock_search_service"
        )
    if args.hybrid_persistent_compact_render_state_buffer and not args.hybrid_native_actor_buffer:
        raise ValueError(
            "--hybrid-persistent-compact-render-state-buffer requires --hybrid-native-actor-buffer"
        )
    if args.hybrid_borrow_single_actor_render_state:
        if not args.hybrid_native_actor_buffer:
            raise ValueError(
                "--hybrid-borrow-single-actor-render-state requires --hybrid-native-actor-buffer"
            )
        if int(args.actor_count) != 1:
            raise ValueError("--hybrid-borrow-single-actor-render-state requires --actor-count 1")
        if args.hybrid_persistent_compact_render_state_buffer:
            raise ValueError(
                "--hybrid-borrow-single-actor-render-state cannot be combined with "
                "--hybrid-persistent-compact-render-state-buffer"
            )
        if not args.hybrid_refresh_observation_stack:
            raise ValueError(
                "--hybrid-borrow-single-actor-render-state requires "
                "--hybrid-refresh-observation-stack"
            )
    if lightzero_or_mctx_probe:
        if True in args.device_latest:
            raise ValueError("LightZero/MCTX profile probes require --device-latest false")
        if args.stack_storage_dtype != "uint8":
            raise ValueError("LightZero/MCTX profile probes require --stack-storage-dtype uint8")
        if args.render_surface != "direct_gray64":
            raise ValueError("LightZero/MCTX profile probes require --render-surface direct_gray64")
        if args.observation_renderer_backend != "jax_gpu_persistent_policy_framebuffer_profile":
            raise ValueError(
                "LightZero/MCTX profile probes require "
                "--observation-renderer-backend "
                "jax_gpu_persistent_policy_framebuffer_profile"
            )
    compact_replay_from_direct = bool(args.lightzero_mcts_arrays_boundary_probe)
    compact_replay_from_array_ceiling = bool(
        args.lightzero_array_ceiling_probe
        and args.lightzero_array_ceiling_mode in COMPACT_REPLAY_ARRAY_CEILING_MODES
    )
    compact_replay_from_mctx = bool(args.mctx_compact_search_probe)
    if args.compact_rollout_slab_probe and args.compact_service_replay_proof:
        raise ValueError(
            "--compact-rollout-slab-probe owns replay-index commits; "
            "do not combine it with --compact-service-replay-proof"
        )
    if args.mctx_compact_search_probe and not args.compact_rollout_slab_probe:
        raise ValueError(
            "--mctx-compact-search-probe currently requires --compact-rollout-slab-probe"
        )
    if args.mctx_compact_search_probe and args.compact_service_replay_proof:
        raise ValueError(
            "--mctx-compact-search-probe does not support "
            "--compact-service-replay-proof; use --compact-rollout-slab-probe"
        )
    if args.compact_rollout_slab_probe and not (
        compact_replay_from_direct or compact_replay_from_array_ceiling or compact_replay_from_mctx
    ):
        allowed = ", ".join(sorted(COMPACT_REPLAY_ARRAY_CEILING_MODES))
        raise ValueError(
            "--compact-rollout-slab-probe requires --lightzero-mcts-arrays-boundary-probe "
            "or --lightzero-array-ceiling-probe with mode in "
            f"{allowed}, or --mctx-compact-search-probe"
        )
    if args.compact_rollout_slab_sample_gate and not args.compact_rollout_slab_probe:
        raise ValueError("--compact-rollout-slab-sample-gate requires --compact-rollout-slab-probe")
    if args.compact_rollout_slab_learner_gate and not args.compact_rollout_slab_sample_gate:
        raise ValueError(
            "--compact-rollout-slab-learner-gate requires --compact-rollout-slab-sample-gate"
        )
    if args.compact_rollout_slab_sample_gate and True in args.materialize_scalar_timestep:
        raise ValueError(
            "--compact-rollout-slab-sample-gate is a no-scalar proof; "
            "use --materialize-scalar-timestep false"
        )
    if args.compact_rollout_slab_sample_gate_batch_size < 0:
        raise ValueError("--compact-rollout-slab-sample-gate-batch-size must be non-negative")
    if args.compact_rollout_slab_sample_gate_interval <= 0:
        raise ValueError("--compact-rollout-slab-sample-gate-interval must be positive")
    if args.compact_rollout_slab_sample_gate_replay_pair_capacity <= 0:
        raise ValueError("--compact-rollout-slab-sample-gate-replay-pair-capacity must be positive")
    if args.compact_rollout_slab_learner_gate_train_steps <= 0:
        raise ValueError("--compact-rollout-slab-learner-gate-train-steps must be positive")
    if args.compact_rollout_slab_learner_gate_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("--compact-rollout-slab-learner-gate-device must be auto, cpu, or cuda")
    if (
        args.compact_rollout_slab_learner_gate_impl
        not in COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_CHOICES
    ):
        allowed = ", ".join(COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_CHOICES)
        raise ValueError(f"--compact-rollout-slab-learner-gate-impl must be one of {allowed}")
    if args.compact_rollout_slab_learner_gate_support_scale <= 0:
        raise ValueError("--compact-rollout-slab-learner-gate-support-scale must be positive")
    if args.compact_rollout_slab_learner_gate_num_unroll_steps <= 0:
        raise ValueError("--compact-rollout-slab-learner-gate-num-unroll-steps must be positive")
    if (
        args.compact_rollout_slab_learner_gate_num_unroll_steps != 1
        and args.compact_rollout_slab_learner_gate_impl != "compact_muzero"
    ):
        raise ValueError(
            "--compact-rollout-slab-learner-gate-num-unroll-steps > 1 "
            "requires --compact-rollout-slab-learner-gate-impl compact_muzero"
        )
    if (
        args.compact_rollout_slab_learner_gate_impl == "compact_muzero"
        and args.compact_rollout_slab_learner_gate_include_rnd
    ):
        raise ValueError(
            "--compact-rollout-slab-learner-gate-impl compact_muzero does not "
            "support --compact-rollout-slab-learner-gate-include-rnd yet"
        )
    if args.compact_owned_loop_entrypoint:
        if not args.compact_rollout_slab_probe:
            raise ValueError(
                "--compact-owned-loop-entrypoint requires --compact-rollout-slab-probe"
            )
        if not args.compact_rollout_slab_sample_gate:
            raise ValueError(
                "--compact-owned-loop-entrypoint requires --compact-rollout-slab-sample-gate"
            )
        if not args.compact_rollout_slab_learner_gate:
            raise ValueError(
                "--compact-owned-loop-entrypoint requires --compact-rollout-slab-learner-gate"
            )
        if True in args.materialize_scalar_timestep:
            raise ValueError(
                "--compact-owned-loop-entrypoint is a no-scalar loop; "
                "use --materialize-scalar-timestep false"
            )
        if not str(args.compact_owned_loop_policy_version_ref).strip():
            raise ValueError("--compact-owned-loop-policy-version-ref must be non-empty")
        if not str(args.compact_owned_loop_policy_source).strip():
            raise ValueError("--compact-owned-loop-policy-source must be non-empty")
    if args.require_terminal_compact_owned_nstep:
        terminal_unroll_steps = int(args.compact_rollout_slab_learner_gate_num_unroll_steps)
        min_terminal_ticks = terminal_unroll_steps + 1
        min_terminal_steps = int(args.max_ticks) + terminal_unroll_steps + 1
        normal_death_contract = bool(args.require_normal_death_terminal_contract)
        base_terminal_shape_ok = (
            args.compact_rollout_slab_probe
            and args.compact_rollout_slab_sample_gate
            and args.compact_rollout_slab_learner_gate
            and args.compact_rollout_slab_learner_gate_impl == "compact_muzero"
            and terminal_unroll_steps > 1
            and args.hybrid_resident_observation_search
        )
        if normal_death_contract:
            terminal_window_ok = (
                int(args.max_ticks) > int(args.steps)
                and int(args.steps) >= terminal_unroll_steps + 1
            )
        else:
            terminal_window_ok = (
                int(args.max_ticks) >= min_terminal_ticks and int(args.steps) >= min_terminal_steps
            )
        if not (base_terminal_shape_ok and terminal_window_ok):
            window_help = (
                "for normal-death rows, max_ticks must be greater than steps "
                "so max-tick truncations cannot enter the learner sample"
                if normal_death_contract
                else (
                    "plus enough measured steps and max_ticks to create active, "
                    "terminal, and post-terminal successor replay rows"
                )
            )
            raise ValueError(
                "--require-terminal-compact-owned-nstep requires resident compact "
                "slab sample+compact_muzero learner gate with num_unroll_steps > 1 "
                f"{window_help}"
            )
    if args.require_normal_death_terminal_contract:
        if args.death_mode != DEATH_MODE_NORMAL:
            raise ValueError(
                "--require-normal-death-terminal-contract requires --death-mode normal"
            )
        if not args.require_terminal_compact_owned_nstep:
            raise ValueError(
                "--require-normal-death-terminal-contract requires "
                "--require-terminal-compact-owned-nstep"
            )
        if not args.compact_owned_loop_entrypoint:
            raise ValueError(
                "--require-normal-death-terminal-contract requires --compact-owned-loop-entrypoint"
            )
        if int(args.compact_rollout_slab_sample_gate_batch_size) <= 0:
            raise ValueError(
                "--require-normal-death-terminal-contract requires a bounded "
                "--compact-rollout-slab-sample-gate-batch-size"
            )
    if (
        args.compact_rollout_slab_action_mode != "search_feedback"
        and not args.compact_rollout_slab_probe
    ):
        raise ValueError(
            "--compact-rollout-slab-action-mode is only meaningful with "
            "--compact-rollout-slab-probe"
        )
    if args.hybrid_resident_observation_search:
        if not args.lightzero_consumer_use_cuda:
            raise ValueError(
                "--hybrid-resident-observation-search requires --lightzero-consumer-use-cuda"
            )
        if not args.hybrid_device_only_stack:
            raise ValueError(
                "--hybrid-resident-observation-search requires --hybrid-device-only-stack"
            )
        if True in args.materialize_scalar_timestep:
            raise ValueError(
                "--hybrid-resident-observation-search requires --materialize-scalar-timestep false"
            )
        if args.stack_storage_dtype != "uint8":
            raise ValueError(
                "--hybrid-resident-observation-search requires --stack-storage-dtype uint8"
            )
        if args.observation_renderer_backend != "jax_gpu_persistent_policy_framebuffer_profile":
            raise ValueError(
                "--hybrid-resident-observation-search requires "
                "--observation-renderer-backend "
                "jax_gpu_persistent_policy_framebuffer_profile"
            )
        if not args.lightzero_array_ceiling_probe or (
            args.lightzero_array_ceiling_mode != "compact_torch_search_service"
        ):
            raise ValueError(
                "--hybrid-resident-observation-search currently requires "
                "--lightzero-array-ceiling-probe with "
                "--lightzero-array-ceiling-mode compact_torch_search_service"
            )
    if args.compact_root_tape_compare and not args.compact_rollout_slab_probe:
        raise ValueError("--compact-root-tape-compare requires --compact-rollout-slab-probe")
    if (
        args.compact_root_tape_compare
        and not args.compact_root_tape_compare_fixed_shape_floor
        and not args.compact_root_tape_compare_mctx
        and not args.compact_root_tape_compare_model_compile
        and not args.compact_root_tape_compare_direct_core
    ):
        raise ValueError(
            "--compact-root-tape-compare currently requires "
            "at least one secondary service: fixed-shape-floor, MCTX, or "
            "model-compile/direct-core"
        )
    if args.compact_root_tape_compare_mctx and not args.compact_root_tape_compare:
        raise ValueError("--compact-root-tape-compare-mctx requires --compact-root-tape-compare")
    if args.compact_root_tape_compare_model_compile and not args.compact_root_tape_compare:
        raise ValueError(
            "--compact-root-tape-compare-model-compile requires --compact-root-tape-compare"
        )
    if args.compact_root_tape_compare_direct_core and not args.compact_root_tape_compare:
        raise ValueError(
            "--compact-root-tape-compare-direct-core requires --compact-root-tape-compare"
        )
    if args.compact_root_tape_compare_mctx and args.mctx_compact_search_probe:
        raise ValueError(
            "--compact-root-tape-compare-mctx is for non-MCTX primary rows; "
            "do not combine it with --mctx-compact-search-probe"
        )
    if args.compact_root_tape_compare_model_compile and args.compact_torch_compile_model_inference:
        raise ValueError(
            "--compact-root-tape-compare-model-compile expects an eager primary; "
            "do not combine it with --compact-torch-compile-model-inference"
        )
    if args.compact_root_tape_compare_model_compile and (
        not args.lightzero_array_ceiling_probe
        or args.lightzero_array_ceiling_mode != "compact_torch_search_service"
    ):
        raise ValueError(
            "--compact-root-tape-compare-model-compile requires "
            "--lightzero-array-ceiling-mode compact_torch_search_service"
        )
    if args.compact_root_tape_compare_direct_core and (
        not args.lightzero_array_ceiling_probe
        or args.lightzero_array_ceiling_mode != "compact_torch_search_service"
    ):
        raise ValueError(
            "--compact-root-tape-compare-direct-core requires "
            "--lightzero-array-ceiling-mode compact_torch_search_service"
        )
    if args.compact_root_tape_compare_direct_core:
        if args.compact_torch_compile_model_inference:
            raise ValueError("--compact-root-tape-compare-direct-core expects model compile off")
        if args.compact_torch_initial_inference_mode != "model_method":
            raise ValueError(
                "--compact-root-tape-compare-direct-core expects a model_method primary"
            )
        if args.lightzero_consumer_root_noise_weight != 0.0:
            raise ValueError(
                "--compact-root-tape-compare-direct-core requires "
                "--lightzero-consumer-root-noise-weight 0.0"
            )
    compact_torch_observation_memory_format = getattr(
        args,
        "compact_torch_observation_memory_format",
        "contiguous",
    )
    compact_torch_model_memory_format = getattr(
        args,
        "compact_torch_model_memory_format",
        "contiguous",
    )
    if compact_torch_model_memory_format != "contiguous":
        raise ValueError(
            "compact Torch model_memory_format=channels_last is parked for the "
            "current LightZero MuZero model because recurrent dynamics uses "
            ".view(); use --compact-torch-model-memory-format contiguous"
        )
    if compact_torch_observation_memory_format != "contiguous" or (
        compact_torch_model_memory_format != "contiguous"
    ):
        if not (
            args.lightzero_array_ceiling_probe
            and args.lightzero_array_ceiling_mode == "compact_torch_search_service"
        ):
            raise ValueError(
                "compact Torch memory-format probes require "
                "--lightzero-array-ceiling-mode compact_torch_search_service"
            )
    if args.compact_root_tape_max_records <= 0:
        raise ValueError("--compact-root-tape-max-records must be positive")
    if not args.compact_root_tape_reference_label:
        raise ValueError("--compact-root-tape-reference-label must be non-empty")
    if args.compact_root_tape_compare and args.hybrid_resident_observation_search:
        raise ValueError(
            "--compact-root-tape-compare does not yet support resident observation "
            "search; wire a real explicit device-to-host root snapshot first"
        )
    if args.compact_service_replay_proof and not (
        compact_replay_from_direct or compact_replay_from_array_ceiling
    ):
        allowed = ", ".join(sorted(COMPACT_REPLAY_ARRAY_CEILING_MODES))
        raise ValueError(
            "--compact-service-replay-proof requires --lightzero-mcts-arrays-boundary-probe "
            "or --lightzero-array-ceiling-probe with mode in "
            f"{allowed}"
        )
    compact_replay_input_mode = (
        "host_uint8"
        if compact_replay_from_mctx
        else (
            args.lightzero_array_ceiling_input_mode
            if compact_replay_from_array_ceiling
            else args.lightzero_mcts_arrays_boundary_input_mode
        )
    )
    if (
        args.compact_service_replay_proof or args.compact_rollout_slab_probe
    ) and compact_replay_input_mode == "resident_torch_reuse":
        raise ValueError(
            "compact search proof modes require fresh inputs; "
            "resident_torch_reuse is a stale-input ceiling"
        )
    if (
        args.lightzero_array_ceiling_probe
        and args.lightzero_array_ceiling_mode == "compact_torch_search_service"
        and args.lightzero_array_ceiling_input_mode != "host_uint8"
    ):
        raise ValueError(
            "compact_torch_search_service currently consumes CompactRootBatchV1 "
            "host uint8 observations directly; use --lightzero-array-ceiling-input-mode host_uint8"
        )
    if (
        args.lightzero_array_ceiling_probe
        and args.lightzero_array_ceiling_mode == "fixed_shape_search_owner"
        and args.lightzero_array_ceiling_input_mode != "host_uint8"
    ):
        raise ValueError(
            "fixed_shape_search_owner currently consumes CompactRootBatchV1 "
            "host uint8 observations directly; use --lightzero-array-ceiling-input-mode host_uint8"
        )
    if (
        (args.compact_service_replay_proof or args.compact_rollout_slab_probe)
        and compact_replay_from_direct
        and args.lightzero_mcts_arrays_boundary_impl == "stock_facade"
        and not args.lightzero_mcts_arrays_boundary_impls
    ):
        raise ValueError("--compact-service-replay-proof requires a direct CTree arrays impl")
    if (
        (args.compact_service_replay_proof or args.compact_rollout_slab_probe)
        and compact_replay_from_direct
        and args.lightzero_mcts_arrays_boundary_impls
    ):
        invalid = [
            impl for impl in args.lightzero_mcts_arrays_boundary_impls if impl == "stock_facade"
        ]
        if invalid:
            raise ValueError("compact search proof modes cannot be crossed with stock_facade")

    rows: list[dict[str, Any]] = []
    row_number = 1
    for compute in args.computes:
        for batch_size in args.batch_sizes:
            for probe_simulations in args.probe_simulations:
                for materialize in args.materialize_scalar_timestep:
                    for device_latest in args.device_latest:
                        if args.resident_chunk_probe and device_latest:
                            continue
                        impls = (
                            getattr(args, "lightzero_mcts_arrays_boundary_impls", None)
                            or [args.lightzero_mcts_arrays_boundary_impl]
                            if args.lightzero_mcts_arrays_boundary_probe
                            else [args.lightzero_mcts_arrays_boundary_impl]
                        )
                        for mcts_arrays_impl in impls:
                            row_id = f"{row_number:03d}"
                            label = _safe_id(
                                f"{compute}-b{batch_size}-a{args.actor_count}"
                                f"-s{probe_simulations}"
                                f"-scalar{'on' if materialize else 'off'}"
                                f"-latest{'on' if device_latest else 'off'}"
                                f"{'-resident' if args.resident_chunk_probe else ''}"
                                f"{'-compactreplay' if args.compact_service_replay_proof else ''}"
                                f"{'-compactslab' if args.compact_rollout_slab_probe else ''}"
                                f"{'-rootmctx' if args.compact_root_tape_compare_mctx else ''}"
                                f"{'-action-' + args.compact_rollout_slab_action_mode if args.compact_rollout_slab_probe and args.compact_rollout_slab_action_mode != 'search_feedback' else ''}"
                                f"{'-samplegate-b' + str(args.compact_rollout_slab_sample_gate_batch_size) + '-i' + str(args.compact_rollout_slab_sample_gate_interval) + '-cap' + str(args.compact_rollout_slab_sample_gate_replay_pair_capacity) if args.compact_rollout_slab_sample_gate else ''}"
                                f"{'-learnergate-' + args.compact_rollout_slab_learner_gate_impl + '-' + args.compact_rollout_slab_learner_gate_device + '-t' + str(args.compact_rollout_slab_learner_gate_train_steps) + '-ss' + str(args.compact_rollout_slab_learner_gate_support_scale) + '-u' + str(args.compact_rollout_slab_learner_gate_num_unroll_steps) if args.compact_rollout_slab_learner_gate else ''}"
                                f"{'-ownedloop' if args.compact_owned_loop_entrypoint else ''}"
                                f"{'-rnd' if args.compact_rollout_slab_learner_gate and args.compact_rollout_slab_learner_gate_include_rnd else ''}"
                                f"{'-devicestack' if args.hybrid_device_only_stack else ''}"
                                f"{'-norefresh' if not args.hybrid_refresh_observation_stack else ''}"
                                f"{'-nativeactor' if args.hybrid_native_actor_buffer else ''}"
                                f"{'-persistentrenderstate' if args.hybrid_persistent_compact_render_state_buffer else ''}"
                                f"{'-borrowrenderstate' if args.hybrid_borrow_single_actor_render_state else ''}"
                                f"{'-lzcf' if args.lightzero_collect_forward_probe else ''}"
                                f"{'-lzii' if args.lightzero_initial_inference_probe else ''}"
                                f"{'-lzarr-' + args.lightzero_array_ceiling_mode + '-in' + args.lightzero_array_ceiling_input_mode if args.lightzero_array_ceiling_probe else ''}"
                                f"{'-lzmctsarr-' + mcts_arrays_impl + '-in' + args.lightzero_mcts_arrays_boundary_input_mode if args.lightzero_mcts_arrays_boundary_probe else ''}"
                                f"{'-mctx-h' + str(args.mctx_hidden_dim) + '-vc' + str(args.mctx_visual_channels) if args.mctx_compact_search_probe else ''}"
                                f"{'-rootmodelcompile-' + args.compact_root_tape_model_compile_mode if args.compact_root_tape_compare_model_compile else ''}"
                                f"{'-rootdirectcore' if args.compact_root_tape_compare_direct_core else ''}"
                            )
                            input_mode = (
                                "mctx_jax_host_uint8"
                                if args.mctx_compact_search_probe
                                else (
                                    args.lightzero_array_ceiling_input_mode
                                    if args.lightzero_array_ceiling_probe
                                    else args.lightzero_mcts_arrays_boundary_input_mode
                                )
                            )
                            fixed_denominator = {
                                "batch_size": batch_size,
                                "actor_count": args.actor_count,
                                "probe_simulations": probe_simulations,
                                "steps": args.steps,
                                "warmup_steps": args.warmup_steps,
                                "max_ticks": args.max_ticks,
                                "death_mode": args.death_mode,
                                "input_mode": input_mode,
                                "materialize_scalar_timestep": materialize,
                                "lightzero_consumer_root_noise_weight": (
                                    args.lightzero_consumer_root_noise_weight
                                ),
                            }
                            if args.compact_rollout_slab_probe:
                                fixed_denominator["compact_rollout_slab_action_mode"] = (
                                    args.compact_rollout_slab_action_mode
                                )
                            if args.compact_root_tape_compare:
                                fixed_denominator["compact_root_tape_compare"] = True
                                fixed_denominator["compact_root_tape_max_records"] = (
                                    args.compact_root_tape_max_records
                                )
                                fixed_denominator["compact_root_tape_compare_fixed_shape_floor"] = (
                                    args.compact_root_tape_compare_fixed_shape_floor
                                )
                                fixed_denominator["compact_root_tape_reference_label"] = (
                                    args.compact_root_tape_reference_label
                                )
                                fixed_denominator["compact_root_tape_compare_mctx"] = (
                                    args.compact_root_tape_compare_mctx
                                )
                                fixed_denominator["compact_root_tape_compare_model_compile"] = (
                                    args.compact_root_tape_compare_model_compile
                                )
                                fixed_denominator["compact_root_tape_compare_direct_core"] = (
                                    args.compact_root_tape_compare_direct_core
                                )
                                if args.compact_root_tape_compare_model_compile:
                                    fixed_denominator["compact_root_tape_model_compile_mode"] = (
                                        args.compact_root_tape_model_compile_mode
                                    )
                                    fixed_denominator["compact_root_tape_require_model_compile"] = (
                                        args.compact_root_tape_require_model_compile
                                    )
                                if args.compact_root_tape_compare_mctx:
                                    fixed_denominator["compact_root_tape_mctx_hidden_dim"] = (
                                        args.mctx_hidden_dim
                                    )
                                    fixed_denominator["compact_root_tape_mctx_visual_channels"] = (
                                        args.mctx_visual_channels
                                    )
                                    fixed_denominator[
                                        "compact_root_tape_mctx_require_gpu_backend"
                                    ] = args.mctx_require_gpu_backend
                            if args.compact_rollout_slab_sample_gate:
                                fixed_denominator["compact_rollout_slab_sample_gate_batch_size"] = (
                                    args.compact_rollout_slab_sample_gate_batch_size
                                )
                                fixed_denominator["compact_rollout_slab_sample_gate_interval"] = (
                                    args.compact_rollout_slab_sample_gate_interval
                                )
                                fixed_denominator[
                                    "compact_rollout_slab_sample_gate_replay_pair_capacity"
                                ] = args.compact_rollout_slab_sample_gate_replay_pair_capacity
                            if args.compact_rollout_slab_learner_gate:
                                fixed_denominator["compact_rollout_slab_learner_gate"] = True
                                fixed_denominator[
                                    "compact_rollout_slab_learner_gate_train_steps"
                                ] = args.compact_rollout_slab_learner_gate_train_steps
                                fixed_denominator["compact_rollout_slab_learner_gate_device"] = (
                                    args.compact_rollout_slab_learner_gate_device
                                )
                                fixed_denominator[
                                    "compact_rollout_slab_learner_gate_include_rnd"
                                ] = args.compact_rollout_slab_learner_gate_include_rnd
                                fixed_denominator["compact_rollout_slab_learner_gate_impl"] = (
                                    args.compact_rollout_slab_learner_gate_impl
                                )
                                fixed_denominator[
                                    "compact_rollout_slab_learner_gate_support_scale"
                                ] = args.compact_rollout_slab_learner_gate_support_scale
                                fixed_denominator[
                                    "compact_rollout_slab_learner_gate_num_unroll_steps"
                                ] = args.compact_rollout_slab_learner_gate_num_unroll_steps
                            if args.compact_owned_loop_entrypoint:
                                fixed_denominator["compact_owned_loop_entrypoint"] = True
                                fixed_denominator["compact_owned_loop_policy_version_ref"] = (
                                    args.compact_owned_loop_policy_version_ref
                                )
                                fixed_denominator["compact_owned_loop_model_version_ref"] = (
                                    args.compact_owned_loop_model_version_ref
                                )
                                fixed_denominator["compact_owned_loop_policy_source"] = (
                                    args.compact_owned_loop_policy_source
                                )
                                fixed_denominator[
                                    "compact_owned_loop_capture_replay_store_state"
                                ] = args.compact_owned_loop_capture_replay_store_state
                            fixed_denominator["hybrid_device_only_stack"] = (
                                args.hybrid_device_only_stack
                            )
                            fixed_denominator["hybrid_refresh_observation_stack"] = (
                                args.hybrid_refresh_observation_stack
                            )
                            fixed_denominator["hybrid_resident_observation_search"] = (
                                args.hybrid_resident_observation_search
                            )
                            fixed_denominator["hybrid_native_actor_buffer"] = (
                                args.hybrid_native_actor_buffer
                            )
                            fixed_denominator["hybrid_persistent_compact_render_state_buffer"] = (
                                args.hybrid_persistent_compact_render_state_buffer
                            )
                            fixed_denominator["hybrid_borrow_single_actor_render_state"] = (
                                args.hybrid_borrow_single_actor_render_state
                            )
                            matched_denominator_id = str(
                                getattr(args, "matched_denominator_id", "") or ""
                            )
                            if matched_denominator_id:
                                fixed_denominator["matched_denominator_id"] = matched_denominator_id
                                fixed_denominator["matched_pair_role"] = str(
                                    getattr(
                                        args,
                                        "matched_pair_role",
                                        "compact_candidate",
                                    )
                                )
                                fixed_denominator["speed_currency"] = str(
                                    getattr(
                                        args,
                                        "matched_speed_currency",
                                        MATCHED_COMPACT_SPEED_CURRENCY,
                                    )
                                )
                                fixed_denominator["row_purpose"] = str(
                                    getattr(
                                        args,
                                        "matched_row_purpose",
                                        MATCHED_DENOMINATOR_ROW_PURPOSE,
                                    )
                                )
                            row = {
                                "schema_id": "curvyzero_hybrid_observation_profile_row/v0",
                                "experiment_id": args.experiment_id,
                                "row_id": row_id,
                                "label": label,
                                "compute": compute,
                                "batch_size": batch_size,
                                "actor_count": args.actor_count,
                                "steps": args.steps,
                                "warmup_steps": args.warmup_steps,
                                "max_ticks": args.max_ticks,
                                "death_mode": args.death_mode,
                                "trail_slots": args.trail_slots,
                                "body_capacity": args.body_capacity,
                                "probe_simulations": probe_simulations,
                                "probe_channels": args.probe_channels,
                                "materialize_scalar_timestep": materialize,
                                "compact_service_replay_proof": (args.compact_service_replay_proof),
                                "compact_rollout_slab_probe": args.compact_rollout_slab_probe,
                                "compact_rollout_slab_sample_gate": (
                                    args.compact_rollout_slab_sample_gate
                                ),
                                "compact_rollout_slab_sample_gate_batch_size": (
                                    args.compact_rollout_slab_sample_gate_batch_size
                                ),
                                "compact_rollout_slab_sample_gate_interval": (
                                    args.compact_rollout_slab_sample_gate_interval
                                ),
                                "compact_rollout_slab_sample_gate_replay_pair_capacity": (
                                    args.compact_rollout_slab_sample_gate_replay_pair_capacity
                                ),
                                "compact_rollout_slab_learner_gate": (
                                    args.compact_rollout_slab_learner_gate
                                ),
                                "compact_rollout_slab_learner_gate_train_steps": (
                                    args.compact_rollout_slab_learner_gate_train_steps
                                ),
                                "compact_rollout_slab_learner_gate_device": (
                                    args.compact_rollout_slab_learner_gate_device
                                ),
                                "compact_rollout_slab_learner_gate_include_rnd": (
                                    args.compact_rollout_slab_learner_gate_include_rnd
                                ),
                                "compact_rollout_slab_learner_gate_impl": (
                                    args.compact_rollout_slab_learner_gate_impl
                                ),
                                "compact_rollout_slab_learner_gate_support_scale": (
                                    args.compact_rollout_slab_learner_gate_support_scale
                                ),
                                "compact_rollout_slab_learner_gate_num_unroll_steps": (
                                    args.compact_rollout_slab_learner_gate_num_unroll_steps
                                ),
                                "compact_rollout_slab_action_mode": (
                                    args.compact_rollout_slab_action_mode
                                ),
                                "compact_owned_loop_entrypoint": (
                                    args.compact_owned_loop_entrypoint
                                ),
                                "compact_owned_loop_policy_version_ref": (
                                    args.compact_owned_loop_policy_version_ref
                                ),
                                "compact_owned_loop_model_version_ref": (
                                    args.compact_owned_loop_model_version_ref
                                ),
                                "compact_owned_loop_policy_source": (
                                    args.compact_owned_loop_policy_source
                                ),
                                "compact_owned_loop_capture_replay_store_state": (
                                    args.compact_owned_loop_capture_replay_store_state
                                ),
                                "compact_root_tape_compare": (args.compact_root_tape_compare),
                                "compact_root_tape_max_records": (
                                    args.compact_root_tape_max_records
                                ),
                                "compact_root_tape_allow_resident_host_snapshot": (
                                    args.compact_root_tape_allow_resident_host_snapshot
                                ),
                                "compact_root_tape_compare_fixed_shape_floor": (
                                    args.compact_root_tape_compare_fixed_shape_floor
                                ),
                                "compact_root_tape_compare_mctx": (
                                    args.compact_root_tape_compare_mctx
                                ),
                                "compact_root_tape_compare_model_compile": (
                                    args.compact_root_tape_compare_model_compile
                                ),
                                "compact_root_tape_compare_direct_core": (
                                    args.compact_root_tape_compare_direct_core
                                ),
                                "compact_root_tape_model_compile_mode": (
                                    args.compact_root_tape_model_compile_mode
                                ),
                                "compact_root_tape_require_model_compile": (
                                    args.compact_root_tape_require_model_compile
                                ),
                                "compact_root_tape_reference_label": (
                                    args.compact_root_tape_reference_label
                                ),
                                "hybrid_device_only_stack": args.hybrid_device_only_stack,
                                "hybrid_refresh_observation_stack": (
                                    args.hybrid_refresh_observation_stack
                                ),
                                "hybrid_resident_observation_search": (
                                    args.hybrid_resident_observation_search
                                ),
                                "hybrid_native_actor_buffer": args.hybrid_native_actor_buffer,
                                "hybrid_persistent_compact_render_state_buffer": (
                                    args.hybrid_persistent_compact_render_state_buffer
                                ),
                                "hybrid_borrow_single_actor_render_state": (
                                    args.hybrid_borrow_single_actor_render_state
                                ),
                                "device_latest": device_latest,
                                "resident_chunk_probe": args.resident_chunk_probe,
                                "lightzero_collect_forward_probe": (
                                    args.lightzero_collect_forward_probe
                                ),
                                "lightzero_initial_inference_probe": (
                                    args.lightzero_initial_inference_probe
                                ),
                                "lightzero_array_ceiling_probe": (
                                    args.lightzero_array_ceiling_probe
                                ),
                                "lightzero_mcts_arrays_boundary_probe": (
                                    args.lightzero_mcts_arrays_boundary_probe
                                ),
                                "mctx_compact_search_probe": args.mctx_compact_search_probe,
                                "mctx_hidden_dim": args.mctx_hidden_dim,
                                "mctx_visual_channels": args.mctx_visual_channels,
                                "mctx_require_gpu_backend": args.mctx_require_gpu_backend,
                                "lightzero_mcts_arrays_boundary_impl": mcts_arrays_impl,
                                "lightzero_mcts_arrays_boundary_input_mode": (
                                    args.lightzero_mcts_arrays_boundary_input_mode
                                ),
                                "lightzero_consumer_root_noise_weight": (
                                    args.lightzero_consumer_root_noise_weight
                                ),
                                "lightzero_array_ceiling_mode": args.lightzero_array_ceiling_mode,
                                "lightzero_array_ceiling_input_mode": (
                                    args.lightzero_array_ceiling_input_mode
                                ),
                                "lightzero_mock_service_materialize_public_output": (
                                    args.lightzero_mock_service_materialize_public_output
                                ),
                                "compact_torch_compile_model_inference": (
                                    getattr(args, "compact_torch_compile_model_inference", False)
                                ),
                                "compact_torch_compile_search": getattr(
                                    args,
                                    "compact_torch_compile_search",
                                    True,
                                ),
                                "compact_torch_require_model_compile": (
                                    getattr(args, "compact_torch_require_model_compile", False)
                                ),
                                "compact_torch_model_compile_mode": (
                                    getattr(
                                        args,
                                        "compact_torch_model_compile_mode",
                                        "reduce-overhead",
                                    )
                                ),
                                "compact_torch_recurrent_action_shape_mode": (
                                    getattr(
                                        args,
                                        "compact_torch_recurrent_action_shape_mode",
                                        "auto",
                                    )
                                ),
                                "compact_torch_initial_inference_mode": (
                                    getattr(
                                        args,
                                        "compact_torch_initial_inference_mode",
                                        "model_method",
                                    )
                                ),
                                "compact_torch_observation_memory_format": (
                                    getattr(
                                        args,
                                        "compact_torch_observation_memory_format",
                                        "contiguous",
                                    )
                                ),
                                "compact_torch_model_memory_format": (
                                    getattr(
                                        args,
                                        "compact_torch_model_memory_format",
                                        "contiguous",
                                    )
                                ),
                                "lightzero_mcts_arrays_boundary_input_freshness": (
                                    "resident_reuse_mode"
                                    if (
                                        args.lightzero_mcts_arrays_boundary_probe
                                        and args.lightzero_mcts_arrays_boundary_input_mode
                                        == "resident_torch_reuse"
                                    )
                                    else "fresh_each_step_expected"
                                ),
                                "lightzero_array_ceiling_input_freshness": (
                                    "resident_reuse_mode"
                                    if (
                                        args.lightzero_array_ceiling_probe
                                        and args.lightzero_array_ceiling_input_mode
                                        == "resident_torch_reuse"
                                    )
                                    else "fresh_each_step_expected"
                                ),
                                "comparison_group": (
                                    "matched_stock_compact_denominator"
                                    if matched_denominator_id
                                    else "compact_root_tape_fixed_denominator"
                                    if args.compact_root_tape_compare
                                    else (
                                        "mcts_arrays_boundary_fixed_denominator"
                                        if args.lightzero_mcts_arrays_boundary_probe
                                        else None
                                    )
                                ),
                                "fixed_denominator": fixed_denominator,
                                "launch_mode": args.launch_mode,
                                "result_capture": args.result_capture,
                                "row_timeout_sec": args.row_timeout_sec,
                                "result_timeout_sec": args.result_timeout_sec,
                                "captured_result_required": args.captured_result_required,
                                "require_terminal_compact_owned_nstep": (
                                    args.require_terminal_compact_owned_nstep
                                ),
                                "require_normal_death_terminal_contract": (
                                    args.require_normal_death_terminal_contract
                                ),
                                "normal_death_terminal_contract_evidence_id": (
                                    args.normal_death_terminal_contract_evidence_id
                                ),
                                "normal_death_terminal_contract_evidence_refs": list(
                                    args.normal_death_terminal_contract_evidence_refs
                                ),
                                "require_payload_label_triad": bool(
                                    args.captured_result_required or args.compact_root_tape_compare
                                ),
                                "profile_only": True,
                                "calls_train_muzero": False,
                                "touches_live_runs": False,
                            }
                            if matched_denominator_id:
                                row["matched_denominator_id"] = matched_denominator_id
                                row["matched_pair_role"] = str(
                                    getattr(
                                        args,
                                        "matched_pair_role",
                                        "compact_candidate",
                                    )
                                )
                                row["speed_currency"] = str(
                                    getattr(
                                        args,
                                        "matched_speed_currency",
                                        MATCHED_COMPACT_SPEED_CURRENCY,
                                    )
                                )
                                row["counterpart_manifest_ref"] = str(
                                    getattr(args, "matched_counterpart_manifest_ref", "") or ""
                                )
                                row["counterpart_row_id"] = str(
                                    getattr(args, "matched_counterpart_row_id", "001")
                                )
                                row["row_purpose"] = str(
                                    getattr(
                                        args,
                                        "matched_row_purpose",
                                        MATCHED_DENOMINATOR_ROW_PURPOSE,
                                    )
                                )
                                row["promotion_claim"] = bool(
                                    getattr(args, "matched_promotion_claim", False)
                                )
                            row["command"] = _command(args, row=row)
                            row["command_text"] = shlex.join(row["command"])
                            rows.append(row)
                            row_number += 1
    matched_denominator_id = str(getattr(args, "matched_denominator_id", "") or "")
    manifest = {
        "schema_id": "curvyzero_hybrid_observation_profile_manifest/v0",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "experiment_id": args.experiment_id,
        "module": MODULE,
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "comparison_group": (
            "matched_stock_compact_denominator"
            if matched_denominator_id
            else "compact_root_tape_fixed_denominator"
            if args.compact_root_tape_compare
            else (
                "mcts_arrays_boundary_fixed_denominator"
                if args.lightzero_mcts_arrays_boundary_probe
                else None
            )
        ),
        "launch_mode": args.launch_mode,
        "result_capture": args.result_capture,
        "rows": rows,
    }
    if matched_denominator_id:
        manifest["matched_denominator"] = {
            "counterpart_manifest_ref": str(
                getattr(args, "matched_counterpart_manifest_ref", "") or ""
            ),
            "counterpart_row_id": str(getattr(args, "matched_counterpart_row_id", "001")),
            "id": matched_denominator_id,
            "promotion_claim": bool(getattr(args, "matched_promotion_claim", False)),
            "role": str(getattr(args, "matched_pair_role", "compact_candidate")),
            "row_purpose": str(
                getattr(args, "matched_row_purpose", MATCHED_DENOMINATOR_ROW_PURPOSE)
            ),
            "speed_currency": str(
                getattr(args, "matched_speed_currency", MATCHED_COMPACT_SPEED_CURRENCY)
            ),
        }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=f"hybrid-observation-{_utc_stamp()}")
    parser.add_argument("--computes", type=_csv_computes, default=["gpu-h100", "gpu-l4-t4"])
    parser.add_argument("--batch-sizes", type=_csv_ints, default=[512])
    parser.add_argument("--actor-count", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--max-ticks", type=int, default=2000)
    parser.add_argument(
        "--death-mode",
        choices=DEATH_MODE_CHOICES,
        default=DEATH_MODE_PROFILE_NO_DEATH,
    )
    parser.add_argument("--trail-slots", type=int, default=1024)
    parser.add_argument("--body-capacity", type=int, default=1024)
    parser.add_argument("--probe-simulations", type=_csv_ints, default=[8])
    parser.add_argument("--probe-channels", type=int, default=16)
    parser.add_argument("--materialize-scalar-timestep", type=_csv_bools, default=[False, True])
    parser.add_argument("--device-latest", type=_csv_bools, default=[False])
    parser.add_argument("--resident-chunk-probe", action="store_true")
    parser.add_argument("--resident-replay-steps", type=int, default=64)
    parser.add_argument("--resident-sample-batch-size", type=int, default=256)
    parser.add_argument("--resident-replay-train-steps", type=int, default=1)
    parser.add_argument("--no-resident-readback-checksum", action="store_true")
    parser.add_argument("--compact-service-replay-proof", action="store_true")
    parser.add_argument("--compact-rollout-slab-probe", action="store_true")
    parser.add_argument("--compact-rollout-slab-sample-gate", action="store_true")
    parser.add_argument("--compact-rollout-slab-sample-gate-batch-size", type=int, default=0)
    parser.add_argument("--compact-rollout-slab-sample-gate-interval", type=int, default=1)
    parser.add_argument(
        "--compact-rollout-slab-sample-gate-replay-pair-capacity",
        type=int,
        default=4096,
    )
    parser.add_argument("--compact-rollout-slab-learner-gate", action="store_true")
    parser.add_argument("--compact-rollout-slab-learner-gate-train-steps", type=int, default=1)
    parser.add_argument(
        "--compact-rollout-slab-learner-gate-device",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument(
        "--compact-rollout-slab-learner-gate-impl",
        choices=COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_CHOICES,
        default="toy_probe",
    )
    parser.add_argument(
        "--compact-rollout-slab-learner-gate-support-scale",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--compact-rollout-slab-learner-gate-num-unroll-steps",
        type=int,
        default=1,
    )
    parser.add_argument("--compact-rollout-slab-learner-gate-include-rnd", action="store_true")
    parser.add_argument(
        "--compact-rollout-slab-action-mode",
        choices=["search_feedback", "scripted_random"],
        default="search_feedback",
    )
    parser.add_argument("--compact-owned-loop-entrypoint", action="store_true")
    parser.add_argument("--compact-owned-loop-policy-version-ref", default="")
    parser.add_argument("--compact-owned-loop-model-version-ref", default="")
    parser.add_argument("--compact-owned-loop-policy-source", default="")
    parser.add_argument(
        "--compact-owned-loop-capture-replay-store-state",
        action="store_true",
    )
    parser.add_argument("--compact-root-tape-compare", action="store_true")
    parser.add_argument("--compact-root-tape-max-records", type=int, default=4)
    parser.add_argument(
        "--compact-root-tape-allow-resident-host-snapshot",
        action="store_true",
    )
    parser.add_argument(
        "--compact-root-tape-compare-fixed-shape-floor",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--compact-root-tape-compare-mctx", action="store_true")
    parser.add_argument(
        "--compact-root-tape-compare-model-compile",
        action="store_true",
    )
    parser.add_argument(
        "--compact-root-tape-compare-direct-core",
        action="store_true",
    )
    parser.add_argument(
        "--compact-root-tape-model-compile-mode",
        choices=COMPACT_TORCH_MODEL_COMPILE_MODE_CHOICES,
        default="default",
    )
    parser.add_argument(
        "--compact-root-tape-require-model-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--compact-root-tape-reference-label", default="primary")
    parser.add_argument("--hybrid-device-only-stack", action="store_true")
    parser.add_argument(
        "--hybrid-refresh-observation-stack",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hybrid-resident-observation-search", action="store_true")
    parser.add_argument("--hybrid-native-actor-buffer", action="store_true")
    parser.add_argument("--hybrid-persistent-compact-render-state-buffer", action="store_true")
    parser.add_argument("--hybrid-borrow-single-actor-render-state", action="store_true")
    parser.add_argument("--lightzero-collect-forward-probe", action="store_true")
    parser.add_argument("--lightzero-initial-inference-probe", action="store_true")
    parser.add_argument("--lightzero-array-ceiling-probe", action="store_true")
    parser.add_argument("--lightzero-mcts-arrays-boundary-probe", action="store_true")
    parser.add_argument("--mctx-compact-search-probe", action="store_true")
    parser.add_argument("--mctx-hidden-dim", type=int, default=64)
    parser.add_argument("--mctx-visual-channels", type=int, default=8)
    parser.add_argument(
        "--mctx-require-gpu-backend",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lightzero-mcts-arrays-boundary-impl",
        choices=MCTS_ARRAYS_BOUNDARY_IMPL_CHOICES,
        default="stock_facade",
    )
    parser.add_argument(
        "--lightzero-mcts-arrays-boundary-impls",
        type=_csv_mcts_arrays_boundary_impls,
        default=None,
        help=(
            "Comma-separated impls to cross when --lightzero-mcts-arrays-boundary-probe is enabled."
        ),
    )
    parser.add_argument(
        "--lightzero-mcts-arrays-boundary-input-mode",
        choices=[
            "host_uint8",
            "host_uint8_pinned",
            "host_float32",
            "resident_torch_reuse",
        ],
        default="host_uint8",
    )
    parser.add_argument(
        "--lightzero-array-ceiling-mode",
        choices=LIGHTZERO_ARRAY_CEILING_MODE_CHOICES,
        default="policy_arrays",
    )
    parser.add_argument(
        "--lightzero-array-ceiling-input-mode",
        choices=[
            "host_uint8",
            "host_uint8_pinned",
            "host_float32",
            "resident_torch_reuse",
        ],
        default="host_uint8",
    )
    parser.add_argument(
        "--lightzero-mock-service-materialize-public-output",
        action="store_true",
        help=(
            "When --lightzero-array-ceiling-mode=mock_search_service, also build "
            "LightZero-shaped public collect dicts from compact arrays to price "
            "the scalar/object edge."
        ),
    )
    parser.add_argument("--lightzero-consumer-temperature", type=float, default=1.0)
    parser.add_argument("--lightzero-consumer-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--lightzero-consumer-root-noise-weight",
        type=float,
        default=-1.0,
        help="Set to >=0 to override LightZero root noise; -1 keeps the config default.",
    )
    parser.add_argument(
        "--lightzero-consumer-use-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lightzero-consumer-collect-with-pure-policy",
        action="store_true",
    )
    parser.add_argument("--compact-torch-compile-model-inference", action="store_true")
    parser.add_argument("--compact-torch-require-model-compile", action="store_true")
    parser.add_argument(
        "--compact-torch-model-compile-mode",
        choices=COMPACT_TORCH_MODEL_COMPILE_MODE_CHOICES,
        default="reduce-overhead",
    )
    parser.add_argument(
        "--compact-torch-compile-search",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compact-torch-recurrent-action-shape-mode",
        choices=["auto", "flat", "column"],
        default="auto",
    )
    parser.add_argument(
        "--compact-torch-initial-inference-mode",
        choices=COMPACT_TORCH_INITIAL_INFERENCE_MODE_CHOICES,
        default="model_method",
    )
    parser.add_argument(
        "--compact-torch-observation-memory-format",
        choices=COMPACT_TORCH_MEMORY_FORMAT_CHOICES,
        default="contiguous",
    )
    parser.add_argument(
        "--compact-torch-model-memory-format",
        choices=COMPACT_TORCH_MEMORY_FORMAT_CHOICES,
        default="contiguous",
    )
    parser.add_argument("--stack-storage-dtype", default="uint8", choices=["uint8", "float32"])
    parser.add_argument(
        "--launch-mode",
        choices=[
            LAUNCH_MODE_BLOCKING_STDOUT_JSON,
            LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT,
        ],
        default=LAUNCH_MODE_BLOCKING_STDOUT_JSON,
    )
    parser.add_argument(
        "--result-capture",
        choices=[RESULT_CAPTURE_STDOUT_JSON, RESULT_CAPTURE_MODAL_FUNCTION_CALL_GET],
        default=RESULT_CAPTURE_STDOUT_JSON,
    )
    parser.add_argument("--row-timeout-sec", type=float, default=None)
    parser.add_argument("--result-timeout-sec", type=float, default=None)
    parser.add_argument("--captured-result-required", action="store_true")
    parser.add_argument("--require-terminal-compact-owned-nstep", action="store_true")
    parser.add_argument("--require-normal-death-terminal-contract", action="store_true")
    parser.add_argument("--normal-death-terminal-contract-evidence-id", default="")
    parser.add_argument(
        "--normal-death-terminal-contract-evidence-refs",
        type=_csv_strings,
        default=[],
    )
    parser.add_argument("--render-surface", default="direct_gray64")
    parser.add_argument(
        "--observation-renderer-backend",
        default="jax_gpu_persistent_policy_framebuffer_profile",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--next-direct-ctree-comparison-preset",
        action="store_true",
        help=(
            "Emit the next fixed-denominator stock_facade vs direct_ctree_arrays "
            "vs direct_ctree_gpu_latent comparison: H100 B512/A16/sim8, "
            "60 measured steps, 15 warmup steps."
        ),
    )
    parser.add_argument(
        "--next-fixed-root-tape-preset",
        action="store_true",
        help=(
            "Emit the current fixed-root-tape comparator grid: compact Torch "
            "search service with fixed-shape floor, root noise forced off, "
            "profile-only."
        ),
    )
    parser.add_argument(
        "--next-fixed-root-tape-large-preset",
        action="store_true",
        help=(
            "Emit the next durable larger fixed-root-tape row: H100 "
            "B512/A16/sim8, 60 measured, 15 warmup, 16 records, detached "
            "FunctionCall result capture."
        ),
    )
    parser.add_argument(
        "--next-fixed-root-tape-mctx-preset",
        action="store_true",
        help=(
            "Emit the next durable fixed-root-tape MCTX comparator row: the "
            "large H100 compact Torch root-tape preset plus an independent "
            "MCTX service label."
        ),
    )
    parser.add_argument(
        "--next-fixed-root-tape-compile-preset",
        action="store_true",
        help=(
            "Emit the next fixed-root-tape eager-vs-model-compile comparator "
            "row for compact Torch, with root noise forced off."
        ),
    )
    parser.add_argument(
        "--next-fixed-root-tape-direct-core-preset",
        action="store_true",
        help=(
            "Emit the next fixed-root-tape public-vs-direct-core initial "
            "inference comparator row for compact Torch, with root noise "
            "forced off."
        ),
    )
    parser.add_argument(
        "--next-terminal-nstep-compact-owned-preset",
        action="store_true",
        help=(
            "Emit the next durable terminal-safe compact-owned N-step row: "
            "H100 B128/A8/sim8, max_ticks=3, resident/device-only compact "
            "Torch, sample-all gate, compact MuZero num_unroll_steps=2, "
            "detached FunctionCall result capture."
        ),
    )
    parser.add_argument(
        "--next-normal-death-compact-owned-preset",
        action="store_true",
        help=(
            "Emit the next normal-death compact-owned proof row: the terminal "
            "N-step compact-owned preset plus death-mode=normal and the "
            "payload-derived normal collision death contract gate, using a "
            "bounded learner sample instead of sample-all."
        ),
    )
    parser.add_argument(
        "--next-borrowed-normal-death-compact-owned-preset",
        action="store_true",
        help=(
            "Emit the normal-death compact-owned proof row in the fast borrowed "
            "single-actor render-state shape: H100 B1024/A1/sim8, resident "
            "device-only compact Torch, direct-core initial inference, and "
            "the normal collision death contract gate."
        ),
    )
    parser.add_argument(
        "--next-matched-denominator-compact-owned-preset",
        action="store_true",
        help=(
            "Emit the selected compact side of the matched denominator pair: "
            "H100 B1024/A16/sim8, 60 measured, 15 warmup, resident/device-only "
            "compact Torch, sample gate B512 every 8, compact MuZero, split "
            "entrypoint, detached FunctionCall result capture."
        ),
    )
    parser.add_argument("--matched-denominator-id", default="")
    parser.add_argument("--matched-pair-role", default="compact_candidate")
    parser.add_argument("--matched-speed-currency", default=MATCHED_COMPACT_SPEED_CURRENCY)
    parser.add_argument("--matched-counterpart-manifest-ref", default="")
    parser.add_argument("--matched-counterpart-row-id", default="001")
    parser.add_argument("--matched-row-purpose", default=MATCHED_DENOMINATOR_ROW_PURPOSE)
    parser.add_argument(
        "--matched-promotion-claim",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--include-l4",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include L4/T4 rows when using --next-direct-ctree-comparison-preset.",
    )
    parser.add_argument("--stdout-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.next_direct_ctree_comparison_preset:
        apply_next_direct_ctree_comparison_preset(args)
    if args.next_fixed_root_tape_preset:
        apply_next_fixed_root_tape_preset(args)
    if args.next_fixed_root_tape_large_preset:
        apply_next_fixed_root_tape_large_preset(args)
    if args.next_fixed_root_tape_mctx_preset:
        apply_next_fixed_root_tape_mctx_preset(args)
    if args.next_fixed_root_tape_compile_preset:
        apply_next_fixed_root_tape_compile_preset(args)
    if args.next_fixed_root_tape_direct_core_preset:
        apply_next_fixed_root_tape_direct_core_preset(args)
    if args.next_terminal_nstep_compact_owned_preset:
        apply_next_terminal_nstep_compact_owned_preset(args)
    if args.next_normal_death_compact_owned_preset:
        apply_next_normal_death_compact_owned_preset(args)
    if args.next_borrowed_normal_death_compact_owned_preset:
        apply_next_borrowed_normal_death_compact_owned_preset(args)
    if args.next_matched_denominator_compact_owned_preset:
        apply_next_matched_denominator_compact_owned_preset(args)
    consumer_count = sum(
        [
            bool(args.resident_chunk_probe),
            bool(args.lightzero_collect_forward_probe),
            bool(args.lightzero_initial_inference_probe),
            bool(args.lightzero_array_ceiling_probe),
            bool(args.lightzero_mcts_arrays_boundary_probe),
            bool(args.mctx_compact_search_probe),
        ]
    )
    if consumer_count > 1:
        raise SystemExit(
            "choose only one of --resident-chunk-probe, "
            "--lightzero-collect-forward-probe, --lightzero-initial-inference-probe, "
            "--lightzero-array-ceiling-probe, --lightzero-mcts-arrays-boundary-probe, "
            "or --mctx-compact-search-probe"
        )
    if args.lightzero_mock_service_materialize_public_output and (
        not args.lightzero_array_ceiling_probe
        or args.lightzero_array_ceiling_mode != "mock_search_service"
    ):
        raise SystemExit(
            "--lightzero-mock-service-materialize-public-output requires "
            "--lightzero-array-ceiling-probe and "
            "--lightzero-array-ceiling-mode=mock_search_service"
        )
    if args.hybrid_persistent_compact_render_state_buffer and not args.hybrid_native_actor_buffer:
        raise SystemExit(
            "--hybrid-persistent-compact-render-state-buffer requires --hybrid-native-actor-buffer"
        )
    if (
        args.lightzero_array_ceiling_probe
        and args.lightzero_array_ceiling_mode == "compact_torch_search_service"
        and args.lightzero_array_ceiling_input_mode != "host_uint8"
    ):
        raise SystemExit(
            "compact_torch_search_service currently consumes CompactRootBatchV1 "
            "host uint8 observations directly; use --lightzero-array-ceiling-input-mode host_uint8"
        )
    if (
        args.lightzero_array_ceiling_probe
        and args.lightzero_array_ceiling_mode == "fixed_shape_search_owner"
        and args.lightzero_array_ceiling_input_mode != "host_uint8"
    ):
        raise SystemExit(
            "fixed_shape_search_owner currently consumes CompactRootBatchV1 "
            "host uint8 observations directly; use --lightzero-array-ceiling-input-mode host_uint8"
        )

    manifest = build_manifest(args)
    if args.stdout_only:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    output_dir = args.output_root / _safe_id(args.experiment_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    commands_path = output_dir / "commands.sh"
    commands_path.write_text("\n".join(row["command_text"] for row in manifest["rows"]) + "\n")
    print(output_path)
    print(commands_path)
    print(
        "durable runner: "
        f"uv run python scripts/run_curvytron_hybrid_observation_profile_manifest.py "
        f"--manifest {output_path}"
    )
    print("debug/raw commands only; use the durable runner for captured metrics")


if __name__ == "__main__":
    main()
