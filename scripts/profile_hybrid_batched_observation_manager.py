#!/usr/bin/env python3
"""Run the profile-only hybrid actor plus zero-observation manager."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from curvyzero.training.source_state_hybrid_observation_profile import (
    CpuOracleBatchedObservationRenderer,
    HybridBatchedStackProbeResult,
    HybridCompactBatch,
    HybridObservationProfileConfig,
    HYBRID_STACK_STORAGE_DTYPES,
    run_hybrid_observation_profile,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_policy_row_records_from_compact_search_v0,
)
from curvyzero.training.compact_policy_row_bridge import validate_compact_policy_search_arrays_v0
from curvyzero.training.exploration_bonus import (
    extract_policy_gray64_latest_for_rnd_from_compact_observation,
)
from curvyzero.training.multiplayer_source_state_target_rows import ACTION_COUNT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--actor-count", type=int, default=4)
    parser.add_argument("--player-count", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-ticks", type=int, default=2_000)
    parser.add_argument("--no-pickle-payload", action="store_true")
    parser.add_argument(
        "--stack-storage-dtype",
        choices=HYBRID_STACK_STORAGE_DTYPES,
        default="float32",
        help="Internal profile stack dtype. uint8 tests memory-bandwidth pressure.",
    )
    parser.add_argument(
        "--observation-mode",
        choices=("zero", "cpu-oracle"),
        default="zero",
        help="zero is topology-only; cpu-oracle uses the local CPU renderer seam.",
    )
    parser.add_argument(
        "--materialize-scalar-timestep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Materialize scalar LightZero-shaped timesteps at the edge.",
    )
    parser.add_argument(
        "--native-vector-boundary-probe",
        action="store_true",
        help=(
            "Consume the batched stack and action mask as contiguous arrays, "
            "return compact action/visit/value arrays, and skip real search."
        ),
    )
    parser.add_argument(
        "--closed-compact-consumer-probe",
        action="store_true",
        help=(
            "Consume HybridCompactBatch through one compact mock search output, "
            "RND latest-frame input, and target-record construction edge."
        ),
    )
    parser.add_argument(
        "--closed-compact-target-mode",
        choices=("records", "arrays"),
        default="records",
        help=(
            "records builds PolicyRowRecordV0 objects; arrays validates the same "
            "compact target fields without per-root record allocation."
        ),
    )
    parser.add_argument(
        "--native-actor-buffer",
        action="store_true",
        help=(
            "Profile actors writing scalar fields directly into parent-owned "
            "compact arrays. Zero-observation rows only."
        ),
    )
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="Print full arrays and per-row info instead of the compact profile summary.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Print a small metrics payload for quick profile grids.",
    )
    args = parser.parse_args()

    observation_renderer = None
    if args.observation_mode == "cpu-oracle":
        observation_renderer = CpuOracleBatchedObservationRenderer()
    if args.native_vector_boundary_probe and args.closed_compact_consumer_probe:
        raise SystemExit(
            "--native-vector-boundary-probe and --closed-compact-consumer-probe "
            "are mutually exclusive"
        )
    batched_stack_probe = None
    if args.native_vector_boundary_probe:
        batched_stack_probe = _NativeVectorBoundaryProbe()
    if args.closed_compact_consumer_probe:
        batched_stack_probe = _ClosedCompactConsumerProbe(
            target_mode=args.closed_compact_target_mode
        )

    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=args.batch_size,
            actor_count=args.actor_count,
            player_count=args.player_count,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            max_ticks=args.max_ticks,
            pickle_payload=not args.no_pickle_payload,
            stack_storage_dtype=args.stack_storage_dtype,
            materialize_scalar_timestep=bool(args.materialize_scalar_timestep),
            native_actor_buffer=bool(args.native_actor_buffer),
        ),
        observation_renderer=observation_renderer,
        batched_stack_probe=batched_stack_probe,
    )
    if args.metrics_only:
        result = _metrics_profile_result(result)
    elif not args.full_output:
        result = _compact_profile_result(result)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))


class _NativeVectorBoundaryProbe:
    """Profile-only Puffer-style compact array consumer.

    This is not MCTS and not a trainer. It prices the boundary shape where a
    batched observation stack and legal-action mask stay as arrays and produce
    compact action/visit/value arrays without scalar LightZero timesteps.
    """

    backend_name = "native_vector_boundary_mock"
    semantics = "puffer_style_contiguous_arrays_mock_not_search"

    def run(
        self, observation: np.ndarray, action_mask: np.ndarray
    ) -> HybridBatchedStackProbeResult:
        return self._run_arrays(
            observation=observation,
            action_mask=action_mask,
            extra_telemetry={
                "native_vector_boundary_batch_contract": "legacy_observation_action_mask_v0",
            },
        )

    def run_compact_batch(self, batch: HybridCompactBatch) -> HybridBatchedStackProbeResult:
        final_present = batch.final_observation is not None
        extra_telemetry = {
            "native_vector_boundary_batch_contract": "compact_row_player_sidecar_v1",
            "native_vector_boundary_reward_checksum": float(batch.reward.sum(dtype=np.float64)),
            "native_vector_boundary_done_count": float(batch.done.sum()),
            "native_vector_boundary_done_root_count": float(batch.done_root.sum()),
            "native_vector_boundary_active_root_count_from_sidecar": float(
                batch.active_root_mask.sum()
            ),
            "native_vector_boundary_to_play_checksum": float(batch.to_play.astype(np.int64).sum()),
            "native_vector_boundary_target_reward_shape": list(batch.target_reward.shape),
            "native_vector_boundary_policy_env_id_checksum": float(
                batch.policy_env_id.astype(np.int64).sum()
            ),
            "native_vector_boundary_policy_player_checksum": float(
                batch.policy_player.astype(np.int64).sum()
            ),
            "native_vector_boundary_terminal_count": float(batch.terminal_global_rows.size),
            "native_vector_boundary_autoreset_count": float(batch.autoreset_global_rows.size),
            "native_vector_boundary_final_observation_present": bool(final_present),
            "native_vector_boundary_final_observation_rows": float(
                batch.final_observation_row_mask.sum()
            ),
            "native_vector_boundary_episode_step_checksum": float(
                batch.episode_step.astype(np.int64).sum()
            ),
            "native_vector_boundary_alive_count": float(batch.alive.sum()),
            "native_vector_boundary_joint_action_checksum": float(
                batch.joint_action.astype(np.int64).sum()
            ),
        }
        return self._run_arrays(
            observation=batch.observation,
            action_mask=batch.action_mask,
            extra_telemetry=extra_telemetry,
        )

    def _run_arrays(
        self,
        *,
        observation: np.ndarray,
        action_mask: np.ndarray,
        extra_telemetry: dict[str, Any],
    ) -> HybridBatchedStackProbeResult:
        import time

        started = time.perf_counter()
        pack_started = time.perf_counter()
        obs = np.asarray(observation)
        mask = np.asarray(action_mask, dtype=np.bool_)
        root_count = int(obs.shape[0]) * int(obs.shape[1])
        flat_mask = mask.reshape(root_count, -1)
        flat_obs = obs.reshape(root_count, *obs.shape[2:])
        pack_sec = time.perf_counter() - pack_started

        action_started = time.perf_counter()
        active = flat_mask.any(axis=1)
        visits = flat_mask.astype(np.float32, copy=True)
        visit_denominator = visits.sum(axis=1, keepdims=True)
        np.divide(visits, np.maximum(visit_denominator, 1.0), out=visits)
        actions = visits.argmax(axis=1).astype(np.int16, copy=False)
        latest = flat_obs[:, -1].astype(np.float32, copy=False)
        if obs.dtype == np.uint8:
            latest = latest * np.float32(1.0 / 255.0)
        values = latest.mean(axis=(1, 2), dtype=np.float32)
        action_sec = time.perf_counter() - action_started

        output_started = time.perf_counter()
        illegal_actions = int(
            np.logical_and(active, ~flat_mask[np.arange(root_count), actions]).sum()
        )
        output_bytes = int(actions.nbytes + visits.nbytes + values.nbytes)
        checksum = int(actions.astype(np.int64).sum())
        output_sec = time.perf_counter() - output_started
        total_sec = time.perf_counter() - started

        telemetry = {
            "total_sec": total_sec,
            "native_vector_boundary_total_sec": total_sec,
            "native_vector_boundary_pack_sec": pack_sec,
            "native_vector_boundary_action_sec": action_sec,
            "native_vector_boundary_output_sec": output_sec,
            "native_vector_boundary_root_count": float(root_count),
            "native_vector_boundary_active_root_count": float(active.sum()),
            "native_vector_boundary_input_bytes": float(obs.nbytes + mask.nbytes),
            "native_vector_boundary_output_bytes": float(output_bytes),
            "native_vector_boundary_illegal_action_count": float(illegal_actions),
            "native_vector_boundary_action_checksum": float(checksum),
            "native_vector_boundary_visit_shape": list(visits.shape),
            "native_vector_boundary_semantics": self.semantics,
        }
        telemetry.update(extra_telemetry)
        return HybridBatchedStackProbeResult(telemetry=telemetry)


class _ClosedCompactConsumerProbe:
    """Profile-only closed compact sidecar consumer.

    This is deliberately not real MCTS. It prices the compact batch path after
    observation: legal compact search-output arrays, RND latest-frame input,
    and the target-record construction edge from one `HybridCompactBatch`.
    """

    backend_name = "closed_compact_consumer_mock_search"
    semantics = "compact_batch_to_mock_search_rnd_target_records_profile_only"

    def __init__(self, *, target_mode: str = "records") -> None:
        if target_mode not in {"records", "arrays"}:
            raise ValueError("target_mode must be 'records' or 'arrays'")
        self._target_mode = target_mode
        self.semantics = f"compact_batch_to_mock_search_rnd_target_{target_mode}_profile_only"

    def run_compact_batch(self, batch: HybridCompactBatch) -> HybridBatchedStackProbeResult:
        import time

        started = time.perf_counter()

        rnd_started = time.perf_counter()
        rnd_input = extract_policy_gray64_latest_for_rnd_from_compact_observation(
            batch.observation,
            batch.target_reward,
        )
        rnd_latest_sec = time.perf_counter() - rnd_started

        search_started = time.perf_counter()
        flat_mask = np.asarray(batch.action_mask, dtype=np.bool_).reshape(-1, ACTION_COUNT)
        active_mask = np.asarray(batch.active_root_mask, dtype=np.bool_).reshape(-1)
        active_indices = np.flatnonzero(active_mask)
        active_mask_rows = flat_mask[active_indices]
        visit_policy = active_mask_rows.astype(np.float32, copy=True)
        visit_denominator = visit_policy.sum(axis=1, keepdims=True)
        np.divide(visit_policy, np.maximum(visit_denominator, 1.0), out=visit_policy)
        selected_action = visit_policy.argmax(axis=1).astype(np.int16, copy=False)
        flat_observation = np.asarray(batch.observation).reshape(
            (-1,) + tuple(np.asarray(batch.observation).shape[2:])
        )
        active_latest = flat_observation[active_indices, -1].astype(np.float32, copy=False)
        if np.asarray(batch.observation).dtype == np.uint8:
            active_latest = active_latest * np.float32(1.0 / 255.0)
        root_value = active_latest.mean(axis=(1, 2), dtype=np.float32)
        search_output_sec = time.perf_counter() - search_started

        target_started = time.perf_counter()
        if self._target_mode == "records":
            records_or_arrays = build_policy_row_records_from_compact_search_v0(
                batch,
                selected_action=selected_action,
                visit_policy=visit_policy,
                root_value=root_value,
                record_index=0,
                policy_source=self.backend_name,
            )
            record_count = len(records_or_arrays)
        else:
            records_or_arrays = validate_compact_policy_search_arrays_v0(
                batch,
                selected_action=selected_action,
                visit_policy=visit_policy,
                root_value=root_value,
                record_index=0,
                policy_source=self.backend_name,
            )
            record_count = int(records_or_arrays.action.shape[0])
        target_record_sec = time.perf_counter() - target_started

        output_bytes = int(selected_action.nbytes + visit_policy.nbytes + root_value.nbytes)
        total_sec = time.perf_counter() - started
        illegal_actions = int(
            np.logical_and(
                np.ones(selected_action.shape, dtype=np.bool_),
                ~active_mask_rows[np.arange(selected_action.size), selected_action],
            ).sum()
        )

        return HybridBatchedStackProbeResult(
            telemetry={
                "total_sec": total_sec,
                "closed_compact_consumer_total_sec": total_sec,
                "closed_compact_consumer_rnd_latest_sec": rnd_latest_sec,
                "closed_compact_consumer_search_output_sec": search_output_sec,
                "closed_compact_consumer_target_record_sec": target_record_sec,
                "closed_compact_consumer_target_mode": self._target_mode,
                "closed_compact_consumer_root_count": float(flat_mask.shape[0]),
                "closed_compact_consumer_active_root_count": float(active_indices.size),
                "closed_compact_consumer_rnd_input_shape": list(rnd_input.shape),
                "closed_compact_consumer_rnd_input_bytes": float(rnd_input.nbytes),
                "closed_compact_consumer_output_bytes": float(output_bytes),
                "closed_compact_consumer_record_count": float(record_count),
                "closed_compact_consumer_illegal_action_count": float(illegal_actions),
                "closed_compact_consumer_visit_shape": list(visit_policy.shape),
                "closed_compact_consumer_semantics": self.semantics,
                "closed_compact_consumer_non_claims": [
                    "not_real_mcts",
                    "not_checked_target_rows_without_replay_chunk",
                    "not_lightzero_training_integration",
                    "not_native_game_segment",
                    "not_learner_update",
                ],
            }
        )

    def run(
        self, observation: np.ndarray, action_mask: np.ndarray
    ) -> HybridBatchedStackProbeResult:
        raise ValueError("closed compact consumer probe requires HybridCompactBatch sidecars")


def _compact_profile_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return the fields that matter for sweep dashboards and terminal reads."""

    policy_env_id = [int(value) for value in result.get("last_policy_env_id", [])]
    policy_env_row = [int(value) for value in result.get("last_policy_env_row", [])]
    policy_player = [int(value) for value in result.get("last_policy_player", [])]
    return {
        "schema_id": result.get("schema_id"),
        "impl_id": result.get("impl_id"),
        "profile_only": result.get("profile_only"),
        "calls_train_muzero": result.get("calls_train_muzero"),
        "stock_lightzero_integrated": result.get("stock_lightzero_integrated"),
        "trainer_defaults_changed": result.get("trainer_defaults_changed"),
        "touches_live_runs": result.get("touches_live_runs"),
        "observation_mode": result.get("observation_mode"),
        "renderer_backend_name": result.get("renderer_backend_name"),
        "stack_storage_dtype": result.get("stack_storage_dtype"),
        "materialize_scalar_timestep": result.get("materialize_scalar_timestep"),
        "native_actor_buffer": result.get("native_actor_buffer"),
        "policy_search_probe_backend_name": result.get("policy_search_probe_backend_name"),
        "policy_search_probe_semantics": result.get("policy_search_probe_semantics"),
        "policy_search_probe_calls": result.get("policy_search_probe_calls"),
        "policy_search_probe_total_roots": result.get("policy_search_probe_total_roots"),
        "policy_search_probe_roots_per_call": result.get("policy_search_probe_roots_per_call"),
        "policy_search_probe_input_shape": result.get("policy_search_probe_input_shape"),
        "policy_search_probe_input_dtype": result.get("policy_search_probe_input_dtype"),
        "policy_search_probe_input_bytes_total": result.get(
            "policy_search_probe_input_bytes_total"
        ),
        "policy_search_probe_last_telemetry": result.get("policy_search_probe_last_telemetry"),
        "batched_stack_probe_backend_name": result.get("batched_stack_probe_backend_name"),
        "batched_stack_probe_semantics": result.get("batched_stack_probe_semantics"),
        "batched_stack_probe_calls": result.get("batched_stack_probe_calls"),
        "batched_stack_probe_total_roots": result.get("batched_stack_probe_total_roots"),
        "batched_stack_probe_roots_per_call": result.get("batched_stack_probe_roots_per_call"),
        "batched_stack_probe_input_shape": result.get("batched_stack_probe_input_shape"),
        "batched_stack_probe_input_dtype": result.get("batched_stack_probe_input_dtype"),
        "batched_stack_probe_input_bytes_total": result.get(
            "batched_stack_probe_input_bytes_total"
        ),
        "batched_stack_probe_last_telemetry": result.get("batched_stack_probe_last_telemetry"),
        "batch_size": result.get("batch_size"),
        "actor_count": result.get("actor_count"),
        "player_count": result.get("player_count"),
        "steps": result.get("steps"),
        "warmup_steps": result.get("warmup_steps"),
        "rows_per_step": result.get("rows_per_step"),
        "ready_count": result.get("ready_count"),
        "timestep_count": result.get("timestep_count"),
        "materialized_timestep_count": result.get("materialized_timestep_count"),
        "live_physical_row_count": result.get("live_physical_row_count"),
        "terminal_row_count": result.get("terminal_row_count"),
        "autoreset_row_count": result.get("autoreset_row_count"),
        "done_rows": result.get("done_rows"),
        "total_sec": result.get("total_sec"),
        "measured_sec": result.get("measured_sec"),
        "warmup_sec": result.get("warmup_sec"),
        "steps_per_sec": result.get("steps_per_sec"),
        "physical_rows_per_sec": result.get("physical_rows_per_sec"),
        "timings": result.get("timings"),
        "timing_per_timestep_sec": result.get("timing_per_timestep_sec"),
        "compact_payload_bytes_per_step": result.get("compact_payload_bytes_per_step"),
        "compact_payload_bytes_per_timestep": result.get("compact_payload_bytes_per_timestep"),
        "compact_payload_bytes_total": result.get("compact_payload_bytes_total"),
        "rendered_stack_bytes_per_step": result.get("rendered_stack_bytes_per_step"),
        "compact_vs_rendered_stack_ratio": result.get("compact_vs_rendered_stack_ratio"),
        "last_observation_shape": result.get("last_observation_shape"),
        "last_observation_dtype": result.get("last_observation_dtype"),
        "last_flat_obs_shape": result.get("last_flat_obs_shape"),
        "last_flat_obs_dtype": result.get("last_flat_obs_dtype"),
        "last_target_reward_shape": result.get("last_target_reward_shape"),
        "last_policy_env_id_head": policy_env_id[:16],
        "last_policy_env_id_tail": policy_env_id[-16:],
        "last_policy_env_row_head": policy_env_row[:16],
        "last_policy_env_row_tail": policy_env_row[-16:],
        "last_policy_player_head": policy_player[:16],
        "last_policy_player_tail": policy_player[-16:],
        "contract": result.get("contract"),
    }


def _metrics_profile_result(result: dict[str, Any]) -> dict[str, Any]:
    """Small terminal payload for quick local profile comparisons."""

    timings = dict(result.get("timings") or {})
    probe = dict(result.get("batched_stack_probe_last_telemetry") or {})
    return {
        "schema_id": result.get("schema_id"),
        "profile_only": result.get("profile_only"),
        "calls_train_muzero": result.get("calls_train_muzero"),
        "touches_live_runs": result.get("touches_live_runs"),
        "observation_mode": result.get("observation_mode"),
        "stack_storage_dtype": result.get("stack_storage_dtype"),
        "materialize_scalar_timestep": result.get("materialize_scalar_timestep"),
        "native_actor_buffer": result.get("native_actor_buffer"),
        "batch_size": result.get("batch_size"),
        "actor_count": result.get("actor_count"),
        "steps": result.get("steps"),
        "warmup_steps": result.get("warmup_steps"),
        "timestep_count": result.get("timestep_count"),
        "steps_per_sec": result.get("steps_per_sec"),
        "physical_rows_per_sec": result.get("physical_rows_per_sec"),
        "measured_sec": result.get("measured_sec"),
        "batched_stack_probe_backend_name": result.get("batched_stack_probe_backend_name"),
        "batched_stack_probe_semantics": result.get("batched_stack_probe_semantics"),
        "batched_stack_probe_roots_per_call": result.get("batched_stack_probe_roots_per_call"),
        "timings": {
            "actor_step_wall_sec": timings.get("actor_step_wall_sec"),
            "gather_merge_sec": timings.get("gather_merge_sec"),
            "observation_sec": timings.get("observation_sec"),
            "scalar_materialization_sec": timings.get("scalar_materialization_sec"),
            "batched_stack_probe_sec": timings.get("batched_stack_probe_sec"),
        },
        "probe": probe,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
