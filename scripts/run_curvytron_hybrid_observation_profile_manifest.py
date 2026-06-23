#!/usr/bin/env python3
"""Run CurvyTron hybrid observation profile manifest rows and save results.

This is the profile-only companion to ``build_curvytron_hybrid_observation_profile_grid.py``.
It captures blocking stdout JSON rows and the explicit detached FunctionCall
result-capture rows emitted by the builder. Detached rows are only accepted
when the manifest says how the result will be collected.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from curvyzero.training.compact_death_terminal_contract import (
    CompactDeathTerminalContractError,
)
from curvyzero.training.compact_death_terminal_contract import (
    build_normal_collision_death_contract_from_profile_result_v1,
)

try:
    import modal
except ImportError:  # pragma: no cover - exercised only without the modal extra.
    modal = None  # type: ignore[assignment]


DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_hybrid_observation_profile_results")
BOUNDARY_MODULE = "curvyzero.infra.modal.source_state_batched_observation_boundary_profile"
MANIFEST_SCHEMA_ID = "curvyzero_hybrid_observation_profile_manifest/v0"
RESULT_SCHEMA_ID = "curvyzero_hybrid_observation_profile_collected_result/v0"
SPAWN_SCHEMA_ID = "curvyzero_hybrid_observation_profile_spawn/v0"
COMPACT_ROOT_TAPE_COMPARISON_SCHEMA_ID = "curvyzero_compact_root_tape_comparison/v1"
COMPACT_OWNED_LOOP_SCHEMA_ID = "curvyzero_compact_owned_loop/v1"
COMPACT_REPLAY_STORE_STATE_SCHEMA_ID = "curvyzero_compact_replay_store_state/v1"
LAUNCH_MODE_BLOCKING_STDOUT_JSON = "blocking_stdout_json"
LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT = "detached_function_call_result"
RESULT_CAPTURE_STDOUT_JSON = "stdout_json"
RESULT_CAPTURE_MODAL_FUNCTION_CALL_GET = "modal_function_call_get"
MATCHED_DENOMINATOR_ROW_PURPOSE = "matched_denominator_speed"
MATCHED_COMPACT_SPEED_CURRENCY = "compact_profile_active_roots_per_sec"
DEATH_MODE_CHOICES = {"profile_no_death", "normal"}
MATCHED_COMPACT_EXPECTED_ROW = {
    "actor_count": 16,
    "batch_size": 1024,
    "compact_rollout_slab_action_mode": "search_feedback",
    "compact_rollout_slab_learner_gate_device": "cuda",
    "compact_rollout_slab_learner_gate_impl": "compact_muzero",
    "compact_rollout_slab_learner_gate_include_rnd": False,
    "compact_rollout_slab_learner_gate_num_unroll_steps": 1,
    "compact_rollout_slab_learner_gate_support_scale": 300,
    "compact_rollout_slab_learner_gate_train_steps": 1,
    "compact_rollout_slab_sample_gate_batch_size": 512,
    "compact_rollout_slab_sample_gate_interval": 8,
    "compact_rollout_slab_sample_gate_replay_pair_capacity": 4096,
    "compact_owned_loop_capture_replay_store_state": True,
    "compact_owned_loop_entrypoint": True,
    "compute": "gpu-h100",
    "death_mode": "profile_no_death",
    "device_latest": False,
    "hybrid_device_only_stack": True,
    "hybrid_native_actor_buffer": True,
    "hybrid_persistent_compact_render_state_buffer": False,
    "hybrid_refresh_observation_stack": True,
    "hybrid_resident_observation_search": True,
    "launch_mode": LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT,
    "lightzero_array_ceiling_input_mode": "host_uint8",
    "lightzero_array_ceiling_mode": "compact_torch_search_service",
    "lightzero_array_ceiling_probe": True,
    "lightzero_consumer_root_noise_weight": 0.0,
    "materialize_scalar_timestep": False,
    "max_ticks": 2000,
    "probe_simulations": 8,
    "require_terminal_compact_owned_nstep": False,
    "result_capture": RESULT_CAPTURE_MODAL_FUNCTION_CALL_GET,
    "steps": 60,
    "warmup_steps": 15,
}


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _safe_id(raw: str) -> str:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")
    clean = "".join(char if char in allowed else "-" for char in raw).strip("-.")
    if not clean or not clean[0].isalnum():
        raise ValueError(f"cannot make a safe id from {raw!r}")
    return clean


def _row_id_aliases(row_id: str) -> set[str]:
    aliases = {row_id}
    if row_id.isdigit():
        number = str(int(row_id))
        aliases.update({number, number.zfill(2), number.zfill(3)})
    return aliases


def _parse_rows(raw: list[str]) -> set[str] | None:
    if not raw:
        return None
    rows: set[str] = set()
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if part:
                rows.add(str(int(part)) if part.isdigit() else part)
    return rows


def _selected_rows(manifest: dict[str, Any], row_ids: set[str] | None) -> list[dict[str, Any]]:
    rows = list(manifest.get("rows") or [])
    if row_ids is None:
        return rows
    selected: list[dict[str, Any]] = []
    matched: set[str] = set()
    for row in rows:
        aliases = _row_id_aliases(str(row.get("row_id")))
        hits = row_ids.intersection(aliases)
        if hits:
            selected.append(row)
            matched.update(hits)
    missing = sorted(row_ids.difference(matched))
    if missing:
        raise SystemExit(f"unknown row id(s): {', '.join(missing)}")
    return selected


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_id") != MANIFEST_SCHEMA_ID:
        raise SystemExit(f"manifest schema must be {MANIFEST_SCHEMA_ID}")
    if manifest.get("profile_only") is not True:
        raise SystemExit("manifest must be profile_only=true")
    if manifest.get("calls_train_muzero") is not False:
        raise SystemExit("hybrid boundary manifest must not call train_muzero")
    if manifest.get("touches_live_runs") is not False:
        raise SystemExit("hybrid boundary manifest must not touch live runs")
    rows = list(manifest.get("rows") or [])
    if not rows:
        raise SystemExit("manifest has no rows")

    problems: list[str] = []
    for row in rows:
        row_id = row.get("row_id")
        command = [str(part) for part in row.get("command") or []]
        launch_mode = _row_launch_mode(row)
        result_capture = _row_result_capture(row)
        if not command:
            problems.append(f"row {row_id}: missing command")
            continue
        if launch_mode not in {
            LAUNCH_MODE_BLOCKING_STDOUT_JSON,
            LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT,
        }:
            problems.append(f"row {row_id}: unknown launch_mode {launch_mode!r}")
        if result_capture not in {
            RESULT_CAPTURE_STDOUT_JSON,
            RESULT_CAPTURE_MODAL_FUNCTION_CALL_GET,
        }:
            problems.append(f"row {row_id}: unknown result_capture {result_capture!r}")
        if launch_mode == LAUNCH_MODE_BLOCKING_STDOUT_JSON:
            if result_capture != RESULT_CAPTURE_STDOUT_JSON:
                problems.append(f"row {row_id}: blocking rows require stdout_json capture")
            if "--detach" in command:
                problems.append(
                    f"row {row_id}: remove --detach; this runner captures blocking JSON"
                )
            if "--hybrid-profile-spawn-result" in command:
                problems.append(
                    f"row {row_id}: blocking rows must not spawn a FunctionCall result"
                )
        if launch_mode == LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT:
            if result_capture != RESULT_CAPTURE_MODAL_FUNCTION_CALL_GET:
                problems.append(
                    f"row {row_id}: detached rows require modal_function_call_get capture"
                )
            if "--detach" not in command:
                problems.append(f"row {row_id}: detached rows must include --detach")
            if "--hybrid-profile-spawn-result" not in command:
                problems.append(
                    f"row {row_id}: detached rows must include --hybrid-profile-spawn-result"
                )
        if "-m" not in command or BOUNDARY_MODULE not in command:
            problems.append(f"row {row_id}: command must run -m {BOUNDARY_MODULE}")
        if "--hybrid-observation-canary" not in command:
            problems.append(f"row {row_id}: missing --hybrid-observation-canary")
        problems.extend(_death_mode_preflight_problems(row, command))
        if row.get("profile_only") is not True:
            problems.append(f"row {row_id}: profile_only must be true")
        if row.get("calls_train_muzero") is not False:
            problems.append(f"row {row_id}: calls_train_muzero must be false")
        if row.get("touches_live_runs") is not False:
            problems.append(f"row {row_id}: touches_live_runs must be false")
        problems.extend(_fixed_root_tape_preflight_problems(row, command))
        problems.extend(_compact_owned_candidate_preflight_problems(row, command))
        problems.extend(_matched_denominator_preflight_problems(row, command))
    if problems:
        raise SystemExit("manifest preflight failed:\n- " + "\n- ".join(problems))


def _extract_last_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    if not candidates:
        raise ValueError("no JSON object found in command output")
    for value in reversed(candidates):
        if value.get("schema_id") == SPAWN_SCHEMA_ID and value.get("function_call_id"):
            return value
    for value in reversed(candidates):
        if value.get("ok") is not None and value.get("profile_only") is not None:
            return value
    return candidates[-1]


def _compact_line(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    telemetry = payload.get("batched_stack_probe_last_telemetry")
    if not isinstance(telemetry, dict):
        telemetry = {}
    slab_telemetry = payload.get("compact_rollout_slab_last_telemetry")
    if not isinstance(slab_telemetry, dict):
        slab_telemetry = {}
    sample_gate_last = payload.get("compact_rollout_slab_sample_gate_last_telemetry")
    if not isinstance(sample_gate_last, dict):
        sample_gate_last = {}
    learner_gate_last = payload.get("compact_rollout_slab_learner_gate_last_telemetry")
    if not isinstance(learner_gate_last, dict):
        learner_gate_last = {}
    learner_muzero = learner_gate_last.get(
        "compact_rollout_slab_learner_gate_compact_muzero_telemetry"
    )
    if not isinstance(learner_muzero, dict):
        learner_muzero = {}
    slab_totals = payload.get("compact_rollout_slab_telemetry_totals")
    if not isinstance(slab_totals, dict):
        slab_totals = {}
    slab_has_totals = bool(slab_totals)
    slab_profile_telemetry = slab_telemetry.get("compact_rollout_slab_profile_telemetry")
    if not isinstance(slab_profile_telemetry, dict):
        slab_profile_telemetry = {}
    timings = payload.get("timings")
    if not isinstance(timings, dict):
        timings = {}
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        contract = {}
    ledger_totals = payload.get("batched_stack_probe_ledger_totals")
    if not isinstance(ledger_totals, dict):
        ledger_totals = {}
    array_mode = str(row.get("lightzero_array_ceiling_mode") or "")
    profile_only = _payload_or_row_bool(row, payload, "profile_only", default=True)
    calls_train_muzero = _payload_or_row_bool(
        row,
        payload,
        "calls_train_muzero",
        default=False,
    )
    touches_live_runs = _payload_or_row_bool(
        row,
        payload,
        "touches_live_runs",
        default=False,
    )

    def first_present(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    def telemetry_value(*keys: str) -> Any:
        for key in keys:
            if key in telemetry:
                return telemetry.get(key)
        return None

    def slab_profile_value(*keys: str) -> Any:
        for key in keys:
            if key in slab_profile_telemetry:
                return slab_profile_telemetry.get(key)
        return None

    def _float_or_zero(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def slab_model_sec_from_profile() -> float | None:
        consumer_total = _float_or_zero(
            slab_profile_value("lightzero_consumer_model_total_sec")
        )
        if consumer_total:
            return consumer_total
        direct_total = _float_or_zero(
            slab_profile_value("lightzero_mcts_arrays_boundary_initial_inference_sec")
        ) + _float_or_zero(
            slab_profile_value("lightzero_mcts_arrays_boundary_recurrent_inference_sec")
        )
        array_total = _float_or_zero(
            slab_profile_value("lightzero_array_ceiling_initial_inference_sec")
        ) + _float_or_zero(
            slab_profile_value("lightzero_array_ceiling_recurrent_inference_sec")
        )
        total = direct_total + array_total
        return total if total else None

    row_root_noise = row.get("lightzero_consumer_root_noise_weight")
    if row_root_noise is not None and float(row_root_noise) < 0.0:
        row_root_noise = None

    def ledger_value(array_key: str, mcts_key: str) -> Any:
        selected_key = mcts_key if mcts_probe else array_key
        if selected_key in ledger_totals:
            return ledger_totals.get(selected_key)
        return telemetry.get(selected_key)

    mcts_probe = bool(row.get("lightzero_mcts_arrays_boundary_probe"))
    array_probe = bool(row.get("lightzero_array_ceiling_probe"))
    slab_enabled = bool(
        row.get("compact_rollout_slab_probe")
        or payload.get("compact_rollout_slab_enabled")
    )
    if slab_enabled:
        probe_total_sec = timings.get("compact_rollout_slab_sec")
        model_sec = (
            slab_totals.get("compact_rollout_slab_model_sec") if slab_has_totals else None
        )
        search_sec = (
            slab_totals.get("compact_rollout_slab_search_sec") if slab_has_totals else None
        )
        h2d_sec = (
            slab_totals.get("compact_rollout_slab_h2d_sec") if slab_has_totals else None
        )
    else:
        probe_total_sec = (
            timings.get("lightzero_mcts_arrays_boundary_total_sec")
            if mcts_probe
            else timings.get("lightzero_array_ceiling_total_sec")
            if array_probe
            else timings.get("batched_stack_probe_sec")
        )
        model_sec = (
            timings.get("lightzero_consumer_model_total_sec")
            or (
                float(timings.get("lightzero_mcts_arrays_boundary_initial_inference_sec", 0.0))
                + float(timings.get("lightzero_mcts_arrays_boundary_recurrent_inference_sec", 0.0))
                if mcts_probe
                else None
            )
            or (
                float(timings.get("lightzero_array_ceiling_initial_inference_sec", 0.0))
                + float(timings.get("lightzero_array_ceiling_recurrent_inference_sec", 0.0))
                if array_probe
                else None
            )
        )
        search_sec = (
            timings.get("lightzero_mcts_arrays_boundary_search_sec")
            if mcts_probe
            else timings.get("lightzero_array_ceiling_search_update_sec")
            if array_probe
            else None
        )
        h2d_sec = (
            timings.get("lightzero_mcts_arrays_boundary_input_prepare_sec")
            if mcts_probe
            else timings.get("lightzero_array_ceiling_h2d_sec")
            if array_probe
            else timings.get("batched_stack_probe_host_to_device_sec")
        )
    total_roots = (
        payload.get("compact_rollout_slab_total_roots")
        if slab_enabled
        else payload.get("batched_stack_probe_total_roots")
    )
    probe_roots_per_sec = (
        float(total_roots) / float(probe_total_sec)
        if total_roots is not None and probe_total_sec is not None and float(probe_total_sec) > 0
        else None
    )
    mode = (
        f"compact_rollout_slab:{slab_telemetry.get('compact_rollout_slab_search_impl')}"
        if slab_enabled
        else row.get("lightzero_array_ceiling_mode")
        if row.get("lightzero_array_ceiling_probe")
        else row.get("lightzero_mcts_arrays_boundary_impl")
        if row.get("lightzero_mcts_arrays_boundary_probe")
        else "hybrid_observation"
    )
    measured_sec = payload.get("measured_sec")
    probe_accounting_sec = _float_or_zero(probe_total_sec)
    actor_wall_sec = _float_or_zero(timings.get("actor_step_wall_sec"))
    gather_merge_sec = _float_or_zero(timings.get("gather_merge_sec"))
    observation_sec = _float_or_zero(timings.get("observation_sec"))
    compact_batch_build_sec = _float_or_zero(timings.get("compact_batch_build_sec"))
    compact_sample_gate_sec = _float_or_zero(
        timings.get("compact_rollout_slab_sample_gate_sec")
    )
    scalar_materialization_sec = _float_or_zero(timings.get("scalar_materialization_sec"))
    compact_payload_pickle_sec = _float_or_zero(timings.get("compact_payload_pickle_sec"))
    compact_learner_gate_sec = _float_or_zero(
        timings.get("compact_rollout_slab_learner_gate_sec")
    )
    known_accounting_sec = (
        actor_wall_sec
        + gather_merge_sec
        + observation_sec
        + compact_batch_build_sec
        + probe_accounting_sec
        + compact_sample_gate_sec
        + scalar_materialization_sec
        + compact_payload_pickle_sec
        + compact_learner_gate_sec
    )
    measured_float = _float_or_zero(measured_sec)
    other_accounting_sec = max(0.0, measured_float - known_accounting_sec)
    summary = {
        "row_id": row.get("row_id"),
        "label": row.get("label"),
        "status": "complete" if payload.get("ok") is True else "failed",
        "profile_only": profile_only,
        "calls_train_muzero": calls_train_muzero,
        "touches_live_runs": touches_live_runs,
        "promotion_eligible": False,
        "promotion_blocker": "profile_only_boundary_probe",
        "compute": row.get("compute"),
        "mode": mode,
        "simulations": row.get("probe_simulations"),
        "batch_size": first_present(payload.get("batch_size"), row.get("batch_size")),
        "actor_count": first_present(payload.get("actor_count"), row.get("actor_count")),
        "death_mode": first_present(payload.get("death_mode"), row.get("death_mode")),
        "render_state_handoff_mode": contract.get("render_state_handoff_mode"),
        "borrow_single_actor_render_state": contract.get(
            "borrow_single_actor_render_state"
        ),
        "render_state_copy_steps": contract.get("render_state_copy_steps"),
        "render_state_borrowed_steps": contract.get("render_state_borrowed_steps"),
        "done_semantics_verified": payload.get("done_semantics_verified"),
        "terminated_row_count": payload.get("terminated_row_count"),
        "truncated_row_count": payload.get("truncated_row_count"),
        "death_row_count": payload.get("death_row_count"),
        "death_count_total": payload.get("death_count_total"),
        "death_cause_count_by_name": payload.get("death_cause_count_by_name"),
        "normal_collision_death_causes": payload.get(
            "normal_collision_death_causes"
        ),
        "normal_collision_death_hit_owner_present": payload.get(
            "normal_collision_death_hit_owner_present"
        ),
        "normal_collision_death_evidence_rows": payload.get(
            "normal_collision_death_evidence_rows"
        ),
        "terminal_final_observation_row_count": payload.get(
            "terminal_final_observation_row_count"
        ),
        "terminal_final_observation_before_autoreset_verified": payload.get(
            "terminal_final_observation_before_autoreset_verified"
        ),
        "terminal_final_reward_map_row_count": payload.get(
            "terminal_final_reward_map_row_count"
        ),
        "terminal_final_reward_map_matches_reward_row_count": payload.get(
            "terminal_final_reward_map_matches_reward_row_count"
        ),
        "terminal_final_reward_map_verified": payload.get(
            "terminal_final_reward_map_verified"
        ),
        "matched_denominator_id": row.get("matched_denominator_id"),
        "matched_pair_role": row.get("matched_pair_role"),
        "row_purpose": row.get("row_purpose"),
        "speed_currency": row.get("speed_currency"),
        "counterpart_manifest_ref": row.get("counterpart_manifest_ref"),
        "counterpart_row_id": row.get("counterpart_row_id"),
        "promotion_claim": row.get("promotion_claim"),
        "steps_per_sec": payload.get("steps_per_sec"),
        "physical_rows_per_sec": payload.get("physical_rows_per_sec"),
        "rows_per_step": payload.get("rows_per_step"),
        "measured_sec": measured_sec,
        "accounting_actor_wall_sec": actor_wall_sec,
        "accounting_gather_merge_sec": gather_merge_sec,
        "accounting_observation_sec": observation_sec,
        "accounting_compact_batch_build_sec": compact_batch_build_sec,
        "accounting_probe_sec": probe_accounting_sec,
        "accounting_compact_sample_gate_sec": compact_sample_gate_sec,
        "accounting_scalar_materialization_sec": scalar_materialization_sec,
        "accounting_compact_payload_pickle_sec": compact_payload_pickle_sec,
        "accounting_compact_learner_gate_sec": compact_learner_gate_sec,
        "accounting_known_sec": known_accounting_sec,
        "accounting_other_sec": other_accounting_sec,
        "accounting_other_fraction": (
            other_accounting_sec / measured_float if measured_float > 0.0 else None
        ),
        "total_roots": total_roots,
        "probe_total_sec": probe_total_sec,
        "probe_roots_per_sec": probe_roots_per_sec,
        "compact_rollout_slab_enabled": slab_enabled,
        "compact_rollout_slab_calls": payload.get("compact_rollout_slab_calls"),
        "compact_rollout_slab_roots_per_call": payload.get(
            "compact_rollout_slab_roots_per_call"
        ),
        "compact_rollout_slab_committed_index_rows": payload.get(
            "compact_rollout_slab_committed_index_row_count"
        ),
        "compact_rollout_slab_action_mode": payload.get("compact_rollout_slab_action_mode"),
        "compact_rollout_slab_action_override_drop_count": payload.get(
            "compact_rollout_slab_action_override_drop_count"
        ),
        "env_action_checksum_total": payload.get("env_action_checksum_total"),
        "env_trajectory_checksum_total": payload.get("env_trajectory_checksum_total"),
        "last_env_action_checksum": payload.get("last_env_action_checksum"),
        "last_env_trajectory_checksum": payload.get("last_env_trajectory_checksum"),
        "compact_rollout_slab_search_impl": slab_telemetry.get(
            "compact_rollout_slab_search_impl"
        ),
        "compact_rollout_slab_num_simulations": slab_telemetry.get(
            "compact_rollout_slab_num_simulations"
        ),
        "compact_rollout_slab_last_search_service_total_sec": slab_telemetry.get(
            "compact_rollout_slab_search_service_total_sec"
        ),
        "compact_rollout_slab_last_model_sec": slab_model_sec_from_profile()
        or slab_telemetry.get("compact_rollout_slab_model_sec"),
        "compact_rollout_slab_last_search_sec": slab_telemetry.get(
            "compact_rollout_slab_search_sec"
        ),
        "compact_rollout_slab_last_h2d_sec": first_present(
            slab_profile_value(
                "lightzero_consumer_h2d_sec",
                "host_to_device_sec",
                "lightzero_mcts_arrays_boundary_h2d_sec",
                "lightzero_array_ceiling_h2d_sec",
                "lightzero_mcts_arrays_boundary_input_prepare_sec",
            ),
            slab_telemetry.get("compact_rollout_slab_h2d_sec"),
        ),
        "compact_rollout_slab_sample_gate_enabled": payload.get(
            "compact_rollout_slab_sample_gate_enabled"
        ),
        "compact_rollout_slab_sample_gate_calls": payload.get(
            "compact_rollout_slab_sample_gate_calls"
        ),
        "compact_rollout_slab_sample_gate_index_rows": payload.get(
            "compact_rollout_slab_sample_gate_index_row_count"
        ),
        "compact_rollout_slab_sample_gate_target_rows": payload.get(
            "compact_rollout_slab_sample_gate_target_row_count"
        ),
        "compact_rollout_slab_sample_gate_sample_rows": payload.get(
            "compact_rollout_slab_sample_gate_sample_row_count"
        ),
        "compact_rollout_slab_sample_gate_batch_size": payload.get(
            "compact_rollout_slab_sample_gate_batch_size"
        ),
        "compact_rollout_slab_sample_gate_interval": payload.get(
            "compact_rollout_slab_sample_gate_interval"
        ),
        "compact_rollout_slab_sample_gate_opportunities": payload.get(
            "compact_rollout_slab_sample_gate_opportunities"
        ),
        "compact_rollout_slab_sample_gate_skipped_count": payload.get(
            "compact_rollout_slab_sample_gate_skipped_count"
        ),
        "compact_rollout_slab_sample_gate_sec": payload.get(
            "compact_rollout_slab_sample_gate_sec"
        ),
        "compact_rollout_slab_sample_gate_mock_base_env_timestep_rows": payload.get(
            "compact_rollout_slab_sample_gate_mock_base_env_timestep_rows"
        ),
        "compact_rollout_slab_sample_gate_terminal_sample_rows": sample_gate_last.get(
            "compact_rollout_slab_sample_gate_terminal_sample_row_count"
        ),
        "compact_rollout_slab_sample_gate_next_final_observation_rows": (
            sample_gate_last.get(
                "compact_rollout_slab_sample_gate_next_final_observation_row_count"
            )
        ),
        "compact_rollout_slab_sample_gate_resident_terminal_final_observation_used": (
            sample_gate_last.get(
                "compact_rollout_slab_sample_gate_resident_terminal_final_observation_used"
            )
        ),
        "compact_rollout_slab_sample_gate_terminal_unroll_value_target_rows": (
            sample_gate_last.get(
                "compact_rollout_slab_sample_gate_terminal_unroll_value_target_row_count"
            )
        ),
        "compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode": (
            sample_gate_last.get(
                "compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode"
            )
        ),
        "compact_rollout_slab_learner_gate_enabled": payload.get(
            "compact_rollout_slab_learner_gate_enabled"
        ),
        "compact_rollout_slab_learner_gate_calls": payload.get(
            "compact_rollout_slab_learner_gate_calls"
        ),
        "compact_rollout_slab_learner_gate_updates": payload.get(
            "compact_rollout_slab_learner_gate_updates"
        ),
        "compact_rollout_slab_learner_gate_sample_rows": payload.get(
            "compact_rollout_slab_learner_gate_sample_row_count"
        ),
        "compact_rollout_slab_learner_gate_input_bytes": payload.get(
            "compact_rollout_slab_learner_gate_input_bytes"
        ),
        "compact_rollout_slab_learner_gate_sec": payload.get(
            "compact_rollout_slab_learner_gate_sec"
        ),
        "compact_rollout_slab_learner_gate_train_steps": payload.get(
            "compact_rollout_slab_learner_gate_train_steps"
        ),
        "compact_rollout_slab_learner_gate_device": payload.get(
            "compact_rollout_slab_learner_gate_device"
        ),
        "compact_rollout_slab_learner_gate_include_rnd": payload.get(
            "compact_rollout_slab_learner_gate_include_rnd"
        ),
        "compact_rollout_slab_learner_gate_impl": first_present(
            payload.get("compact_rollout_slab_learner_gate_impl"),
            row.get("compact_rollout_slab_learner_gate_impl"),
        ),
        "compact_rollout_slab_learner_gate_toy_probe": payload.get(
            "compact_rollout_slab_learner_gate_toy_probe"
        ),
        "compact_rollout_slab_learner_gate_real_muzero_update": payload.get(
            "compact_rollout_slab_learner_gate_real_muzero_update"
        ),
        "compact_rollout_slab_learner_gate_support_scale": first_present(
            payload.get("compact_rollout_slab_learner_gate_support_scale"),
            row.get("compact_rollout_slab_learner_gate_support_scale"),
        ),
        "compact_rollout_slab_learner_gate_num_unroll_steps": first_present(
            payload.get("compact_rollout_slab_learner_gate_num_unroll_steps"),
            row.get("compact_rollout_slab_learner_gate_num_unroll_steps"),
        ),
        "compact_muzero_cuda_before_backward_free_bytes": learner_muzero.get(
            "compact_muzero_learner_cuda_before_backward_mem_get_info_free_bytes"
        ),
        "compact_muzero_cuda_after_backward_free_bytes": learner_muzero.get(
            "compact_muzero_learner_cuda_after_backward_mem_get_info_free_bytes"
        ),
        "compact_muzero_cuda_after_train_peak_allocated_bytes": learner_muzero.get(
            "compact_muzero_learner_cuda_after_train_memory_peak_allocated_bytes"
        ),
        "compact_muzero_cuda_after_train_peak_reserved_bytes": learner_muzero.get(
            "compact_muzero_learner_cuda_after_train_memory_peak_reserved_bytes"
        ),
        "compact_owned_loop_entrypoint_enabled": payload.get(
            "compact_owned_loop_entrypoint_enabled"
        ),
        "compact_owned_loop_schema_id": payload.get("compact_owned_loop_schema_id"),
        "compact_owned_loop_profile_only": payload.get("compact_owned_loop_profile_only"),
        "compact_owned_loop_calls_train_muzero": payload.get(
            "compact_owned_loop_calls_train_muzero"
        ),
        "compact_owned_loop_touches_live_runs": payload.get(
            "compact_owned_loop_touches_live_runs"
        ),
        "compact_owned_loop_replay_store_owned": payload.get(
            "compact_owned_loop_replay_store_owned"
        ),
        "compact_owned_loop_policy_version_handoff": payload.get(
            "compact_owned_loop_policy_version_handoff"
        ),
        "compact_owned_loop_policy_version_ref": first_present(
            payload.get("compact_owned_loop_policy_version_ref"),
            row.get("compact_owned_loop_policy_version_ref"),
        ),
        "compact_owned_loop_model_version_ref": first_present(
            payload.get("compact_owned_loop_model_version_ref"),
            row.get("compact_owned_loop_model_version_ref"),
        ),
        "compact_owned_loop_policy_source": first_present(
            payload.get("compact_owned_loop_policy_source"),
            row.get("compact_owned_loop_policy_source"),
        ),
        "compact_service_replay_proof_calls": payload.get(
            "compact_service_replay_proof_calls"
        ),
        "compact_service_replay_proof_warmup_seeded_calls": payload.get(
            "compact_service_replay_proof_warmup_seeded_calls"
        ),
        "compact_service_replay_proof_target_rows": payload.get(
            "compact_service_replay_proof_target_row_count"
        ),
        "compact_service_replay_proof_sec": payload.get("compact_service_replay_proof_sec"),
        "search_sec": search_sec,
        "model_sec": model_sec,
        "h2d_sec": h2d_sec,
        "obs_h2d_bytes": first_present(
            slab_totals.get("compact_rollout_slab_obs_h2d_bytes")
            if slab_enabled and slab_has_totals
            else None,
            ledger_value(
                "lightzero_array_ceiling_obs_h2d_bytes",
                "lightzero_mcts_arrays_boundary_obs_h2d_bytes",
            ),
            telemetry_value(
                "host_to_device_bytes",
                "lightzero_array_ceiling_input_bytes",
                f"{array_mode}_input_bytes",
            ),
        ),
        "resident_observation_used": first_present(
            slab_profile_value("resident_observation_used") if slab_enabled else None,
            payload.get("resident_observation_used"),
        ),
        "resident_observation_generation_id": first_present(
            slab_profile_value("resident_observation_generation_id")
            if slab_enabled
            else None,
            payload.get("resident_observation_generation_id"),
        ),
        "resident_observation_host_fallback_count": first_present(
            slab_profile_value("resident_observation_host_fallback_count")
            if slab_enabled
            else None,
            payload.get("resident_observation_host_fallback_count"),
        ),
        "resident_observation_h2d_bytes": first_present(
            slab_profile_value("resident_observation_h2d_bytes") if slab_enabled else None,
            payload.get("resident_observation_h2d_bytes"),
        ),
        "resident_observation_d2h_bytes": first_present(
            slab_profile_value("resident_observation_d2h_bytes") if slab_enabled else None,
            payload.get("resident_observation_d2h_bytes"),
        ),
        "resident_obs_reused": first_present(
            slab_profile_value("compact_torch_search_service_resident_obs_reused")
            if slab_enabled
            else None,
            payload.get("resident_obs_reused"),
            ledger_value(
                "lightzero_array_ceiling_resident_obs_reused",
                "lightzero_mcts_arrays_boundary_resident_reused",
            ),
        ),
        "mask_h2d_bytes": first_present(
            slab_totals.get("compact_rollout_slab_mask_h2d_bytes")
            if slab_enabled and slab_has_totals
            else None,
            ledger_value(
                "lightzero_array_ceiling_mask_h2d_bytes",
                "lightzero_mcts_arrays_boundary_mask_h2d_bytes",
            ),
        ),
        "action_d2h_bytes": first_present(
            slab_totals.get("compact_rollout_slab_action_d2h_bytes")
            if slab_enabled and slab_has_totals
            else None,
            ledger_value(
                "lightzero_array_ceiling_action_d2h_bytes",
                "lightzero_mcts_arrays_boundary_action_d2h_bytes",
            ),
        ),
        "replay_payload_d2h_bytes": first_present(
            slab_totals.get("compact_rollout_slab_replay_payload_d2h_bytes")
            if slab_enabled and slab_has_totals
            else None,
            ledger_value(
                "lightzero_array_ceiling_replay_payload_d2h_bytes",
                "lightzero_mcts_arrays_boundary_replay_payload_d2h_bytes",
            ),
        ),
        "committed_replay_payload_d2h_bytes": (
            slab_totals.get("compact_rollout_slab_committed_replay_payload_d2h_bytes")
            if slab_enabled and slab_has_totals
            else None
        ),
        "root_observation_copy_bytes": first_present(
            slab_totals.get("compact_rollout_slab_root_observation_copy_bytes")
            if slab_enabled and slab_has_totals
            else None,
            ledger_value(
                "lightzero_array_ceiling_root_observation_copy_bytes",
                "lightzero_mcts_arrays_boundary_root_observation_copy_bytes",
            ),
        ),
        "python_rows_materialized": first_present(
            slab_totals.get("compact_rollout_slab_python_rows_materialized")
            if slab_enabled and slab_has_totals
            else None,
            ledger_value(
                "lightzero_array_ceiling_python_rows_materialized",
                "lightzero_mcts_arrays_boundary_python_rows_materialized",
            ),
        ),
        "rnd_materialized_rows": first_present(
            slab_totals.get("compact_rollout_slab_rnd_materialized_rows")
            if slab_enabled and slab_has_totals
            else None,
            ledger_value(
                "lightzero_array_ceiling_rnd_materialized_rows",
                "lightzero_mcts_arrays_boundary_rnd_materialized_rows",
            ),
        ),
        "model_output_d2h_bytes": (
            telemetry.get("lightzero_mcts_arrays_boundary_model_output_d2h_bytes")
            if mcts_probe
            else None
        ),
        "root_noise_weight": first_present(
            slab_profile_value(
                "lightzero_array_ceiling_root_noise_weight",
                "lightzero_mcts_arrays_boundary_root_noise_weight",
            )
            if slab_enabled
            else None,
            telemetry.get("lightzero_array_ceiling_root_noise_weight")
            if array_probe
            else telemetry.get("lightzero_mcts_arrays_boundary_root_noise_weight"),
            row_root_noise,
        ),
        "compile_status": first_present(
            slab_profile_value(
                "compact_torch_search_service_compile_status",
                "compact_torch_search_compile_status",
                "lightzero_array_ceiling_compile_status",
                "lightzero_mcts_arrays_boundary_compile_status",
            )
            if slab_enabled
            else None,
            telemetry.get("lightzero_array_ceiling_compile_status")
            if array_probe
            else telemetry.get("lightzero_mcts_arrays_boundary_compile_status"),
        ),
        "compile_reason": first_present(
            slab_profile_value(
                "compact_torch_search_service_compile_reason",
                "compact_torch_search_compile_reason",
                "lightzero_array_ceiling_compile_reason",
                "lightzero_mcts_arrays_boundary_compile_reason",
            )
            if slab_enabled
            else None,
            telemetry.get("lightzero_array_ceiling_compile_reason")
            if array_probe
            else telemetry.get("lightzero_mcts_arrays_boundary_compile_reason"),
        ),
        "compile_attempted": (
            slab_profile_value("compact_torch_search_compile_attempted")
            if slab_enabled
            else None
        ),
        "compile_used": (
            slab_profile_value("compact_torch_search_compile_used") if slab_enabled else None
        ),
        "compile_cache_hit": (
            slab_profile_value("compact_torch_search_compile_cache_hit")
            if slab_enabled
            else None
        ),
        "compile_runtime_status": (
            slab_profile_value("compact_torch_search_compile_runtime_status")
            if slab_enabled
            else None
        ),
        "model_compile_requested": (
            slab_profile_value("compact_torch_search_model_compile_requested")
            if slab_enabled
            else None
        ),
        "model_compile_attempted": (
            slab_profile_value("compact_torch_search_model_compile_attempted")
            if slab_enabled
            else None
        ),
        "model_compile_used": (
            slab_profile_value("compact_torch_search_model_compile_used")
            if slab_enabled
            else None
        ),
        "model_compile_cache_hit": (
            slab_profile_value("compact_torch_search_model_compile_cache_hit")
            if slab_enabled
            else None
        ),
        "model_compile_runtime_status": (
            slab_profile_value("compact_torch_search_model_compile_runtime_status")
            if slab_enabled
            else None
        ),
        "recurrent_action_shape_mode_effective": (
            slab_profile_value("compact_torch_search_recurrent_action_shape_mode_effective")
            if slab_enabled
            else None
        ),
        "recurrent_action_shape_fallback_count": (
            slab_profile_value(
                "compact_torch_search_recurrent_action_shape_exception_fallback_count"
            )
            if slab_enabled
            else None
        ),
        "semantics": first_present(
            slab_telemetry.get("compact_rollout_slab_semantics")
            if slab_enabled
            else None,
            slab_profile_value("profile_semantics") if slab_enabled else None,
            slab_profile_value(
                "lightzero_array_ceiling_semantics",
                "lightzero_mcts_arrays_boundary_semantics",
                "compact_torch_search_semantics",
            )
            if slab_enabled
            else None,
            telemetry.get("lightzero_array_ceiling_semantics"),
            telemetry.get("lightzero_mcts_arrays_boundary_semantics"),
            payload.get("batched_stack_probe_semantics"),
        ),
        "terminal_row_count": payload.get("terminal_row_count"),
    }
    summary.update(_root_tape_summary_fields(payload))
    return summary


def _payload_or_row_bool(
    row: dict[str, Any],
    payload: dict[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    if key in payload:
        return bool(payload[key])
    return bool(row.get(key, default))


def _row_launch_mode(row: dict[str, Any]) -> str:
    return str(row.get("launch_mode") or LAUNCH_MODE_BLOCKING_STDOUT_JSON)


def _row_result_capture(row: dict[str, Any]) -> str:
    return str(row.get("result_capture") or RESULT_CAPTURE_STDOUT_JSON)


def _requires_payload_label_triad(row: dict[str, Any]) -> bool:
    return bool(
        row.get("require_payload_label_triad")
        or row.get("captured_result_required")
        or row.get("compact_root_tape_compare")
        or _is_compact_owned_split_entrypoint_row(row)
        or _is_compact_owned_candidate_row(row)
        or _row_launch_mode(row) == LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT
    )


def _is_compact_owned_split_entrypoint_row(row: dict[str, Any]) -> bool:
    return row.get("compact_owned_loop_entrypoint") is True


def _is_compact_owned_candidate_row(row: dict[str, Any]) -> bool:
    return bool(
        row.get("compact_rollout_slab_probe") is True
        and row.get("compact_rollout_slab_sample_gate") is True
        and row.get("compact_rollout_slab_learner_gate") is True
        and str(row.get("compact_rollout_slab_learner_gate_impl") or "")
        == "compact_muzero"
    )


def _payload_label_problems(row: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    expected = {
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
    }
    problems: list[str] = []
    require = _requires_payload_label_triad(row)
    for key, expected_value in expected.items():
        row_value = row.get(key, expected_value)
        if not isinstance(row_value, bool) or row_value is not expected_value:
            problems.append(f"{key}: row must be {expected_value!r}, got {row_value!r}")
            continue
        if key not in payload:
            if require:
                problems.append(f"{key}: missing from payload")
            continue
        payload_value = payload[key]
        if not isinstance(payload_value, bool) or payload_value is not expected_value:
            problems.append(f"{key}: row={expected_value!r}, payload={payload_value!r}")
    return problems


def _fixed_root_tape_preflight_problems(
    row: dict[str, Any],
    command: list[str],
) -> list[str]:
    if row.get("compact_root_tape_compare") is not True:
        return []
    row_id = row.get("row_id")
    problems: list[str] = []
    if row.get("compact_rollout_slab_probe") is not True:
        problems.append(f"row {row_id}: root-tape compare requires compact slab probe")
    if "--hybrid-compact-rollout-slab-probe" not in command:
        problems.append(f"row {row_id}: missing compact slab command flag")
    if "--hybrid-compact-root-tape-compare" not in command:
        problems.append(f"row {row_id}: missing compact root-tape compare command flag")
    if row.get("compact_root_tape_compare_mctx") is True:
        if "--hybrid-compact-root-tape-compare-mctx" not in command:
            problems.append(f"row {row_id}: missing root-tape MCTX command flag")
    if row.get("compact_root_tape_compare_model_compile") is True:
        if "--hybrid-compact-root-tape-compare-model-compile" not in command:
            problems.append(
                f"row {row_id}: missing root-tape model-compile command flag"
            )
        if row.get("compact_torch_compile_model_inference") is True:
            problems.append(
                f"row {row_id}: root-tape model-compile comparison expects eager primary"
            )
    if row.get("compact_root_tape_compare_direct_core") is True:
        if "--hybrid-compact-root-tape-compare-direct-core" not in command:
            problems.append(
                f"row {row_id}: missing root-tape direct-core command flag"
            )
        if row.get("compact_torch_compile_model_inference") is True:
            problems.append(
                f"row {row_id}: root-tape direct-core comparison expects model compile off"
            )
        if str(row.get("compact_torch_initial_inference_mode") or "model_method") != (
            "model_method"
        ):
            problems.append(
                f"row {row_id}: root-tape direct-core comparison expects model_method primary"
            )
    try:
        max_records = int(row.get("compact_root_tape_max_records"))
    except (TypeError, ValueError):
        max_records = 0
    if max_records <= 0:
        problems.append(f"row {row_id}: compact_root_tape_max_records must be positive")
    if str(row.get("compact_root_tape_reference_label") or "") != "primary":
        problems.append(f"row {row_id}: root-tape reference label must be primary")
    if (
        row.get("compact_root_tape_compare_fixed_shape_floor") is not True
        and row.get("compact_root_tape_compare_mctx") is not True
        and row.get("compact_root_tape_compare_model_compile") is not True
        and row.get("compact_root_tape_compare_direct_core") is not True
    ):
        problems.append(f"row {row_id}: root-tape compare requires a secondary service")
    if row.get("compact_root_tape_allow_resident_host_snapshot") is True:
        problems.append(f"row {row_id}: root-tape resident host snapshot fallback disabled")
    if row.get("hybrid_resident_observation_search") is True:
        problems.append(f"row {row_id}: root-tape compare forbids resident observation search")
    try:
        root_noise = float(row.get("lightzero_consumer_root_noise_weight"))
    except (TypeError, ValueError):
        root_noise = None
    if root_noise != 0.0:
        problems.append(f"row {row_id}: root-tape compare requires root noise weight 0.0")
    return problems


def _command_flag_value(command: list[str], flag: str) -> str | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def _death_mode_preflight_problems(row: dict[str, Any], command: list[str]) -> list[str]:
    row_id = row.get("row_id")
    if "death_mode" not in row:
        return [f"row {row_id}: missing death_mode"]
    death_mode = str(row.get("death_mode"))
    if death_mode not in DEATH_MODE_CHOICES:
        allowed = ", ".join(sorted(DEATH_MODE_CHOICES))
        return [f"row {row_id}: death_mode must be one of {allowed}; got {death_mode!r}"]
    command_value = _command_flag_value(command, "--death-mode")
    if command_value != death_mode:
        return [
            f"row {row_id}: command --death-mode {command_value!r} "
            f"does not match row death_mode {death_mode!r}"
        ]
    return []


def _compact_owned_candidate_preflight_problems(
    row: dict[str, Any],
    command: list[str],
) -> list[str]:
    row_id = row.get("row_id")
    problems: list[str] = []
    is_candidate = _is_compact_owned_candidate_row(row)
    is_split_entrypoint = _is_compact_owned_split_entrypoint_row(row)
    require_normal_death = row.get("require_normal_death_terminal_contract") is True
    if require_normal_death:
        if str(row.get("death_mode") or "") != "normal":
            problems.append(
                f"row {row_id}: normal-death terminal contract requires "
                "death_mode=normal"
            )
        if _command_flag_value(command, "--death-mode") != "normal":
            problems.append(
                f"row {row_id}: normal-death terminal contract requires "
                "command --death-mode normal"
            )
        if not is_candidate:
            problems.append(
                f"row {row_id}: normal-death terminal contract requires "
                "compact-owned candidate slab/sample/compact_muzero learner gates"
            )
        if not is_split_entrypoint:
            problems.append(
                f"row {row_id}: normal-death terminal contract requires "
                "compact-owned split entrypoint"
            )
        if row.get("require_terminal_compact_owned_nstep") is not True:
            problems.append(
                f"row {row_id}: normal-death terminal contract requires "
                "terminal compact-owned N-step proof"
            )
        sample_gate_batch_size = _as_int(
            row.get("compact_rollout_slab_sample_gate_batch_size")
        )
        if sample_gate_batch_size is None or sample_gate_batch_size <= 0:
            problems.append(
                f"row {row_id}: normal-death terminal contract requires a bounded "
                "compact rollout slab sample gate batch size"
            )
    if not (is_candidate or is_split_entrypoint):
        return problems
    if row.get("materialize_scalar_timestep") is not False:
        problems.append(
            f"row {row_id}: compact-owned candidate requires "
            "materialize_scalar_timestep=false"
        )
    if row.get("hybrid_device_only_stack") is not True:
        problems.append(
            f"row {row_id}: compact-owned candidate requires hybrid_device_only_stack"
        )
    if row.get("hybrid_resident_observation_search") is not True:
        problems.append(
            f"row {row_id}: compact-owned candidate requires "
            "hybrid_resident_observation_search"
        )
    if row.get("compact_rollout_slab_probe") is not True:
        problems.append(f"row {row_id}: compact-owned candidate requires compact slab")
    if row.get("compact_rollout_slab_sample_gate") is not True:
        problems.append(f"row {row_id}: compact-owned candidate requires sample gate")
    if row.get("compact_rollout_slab_learner_gate") is not True:
        problems.append(f"row {row_id}: compact-owned candidate requires learner gate")
    if str(row.get("compact_rollout_slab_learner_gate_impl") or "") != "compact_muzero":
        problems.append(
            f"row {row_id}: compact-owned candidate row learner gate impl must be "
            "compact_muzero"
        )
    required_flags = {
        "--no-hybrid-materialize-scalar-timestep": "no-scalar command flag",
        "--hybrid-device-only-stack": "device-only stack command flag",
        "--hybrid-resident-observation-search": "resident observation command flag",
        "--hybrid-compact-rollout-slab-probe": "compact slab command flag",
        "--hybrid-compact-rollout-slab-sample-gate": "sample gate command flag",
        "--hybrid-compact-rollout-slab-learner-gate": "learner gate command flag",
    }
    for flag, label in required_flags.items():
        if flag not in command:
            problems.append(f"row {row_id}: compact-owned candidate missing {label}")
    if (
        _command_flag_value(command, "--hybrid-compact-rollout-slab-learner-gate-impl")
        != "compact_muzero"
    ):
        problems.append(
            f"row {row_id}: compact-owned candidate learner gate impl must be "
            "compact_muzero"
        )
    expected_unroll_steps = str(row.get("compact_rollout_slab_learner_gate_num_unroll_steps", 1))
    if (
        _command_flag_value(
            command,
            "--hybrid-compact-rollout-slab-learner-gate-num-unroll-steps",
        )
        != expected_unroll_steps
    ):
        problems.append(
            f"row {row_id}: compact-owned candidate learner gate num_unroll_steps "
            f"must be {expected_unroll_steps}"
        )
    if "--hybrid-compact-rollout-slab-learner-gate-include-rnd" in command:
        problems.append(
            f"row {row_id}: compact-owned candidate compact_muzero gate forbids RND"
        )
    if is_split_entrypoint:
        if "--hybrid-compact-owned-loop-entrypoint" not in command:
            problems.append(
                f"row {row_id}: compact-owned split entrypoint missing command flag"
            )
        policy_ref = str(row.get("compact_owned_loop_policy_version_ref") or "").strip()
        if not policy_ref:
            problems.append(
                f"row {row_id}: compact-owned split entrypoint missing policy version"
            )
        elif (
            _command_flag_value(
                command,
                "--hybrid-compact-owned-loop-policy-version-ref",
            )
            != policy_ref
        ):
            problems.append(
                f"row {row_id}: compact-owned split entrypoint policy version "
                f"must be {policy_ref}"
            )
        policy_source = str(row.get("compact_owned_loop_policy_source") or "").strip()
        if not policy_source:
            problems.append(
                f"row {row_id}: compact-owned split entrypoint missing policy source"
            )
        elif (
            _command_flag_value(command, "--hybrid-compact-owned-loop-policy-source")
            != policy_source
        ):
            problems.append(
                f"row {row_id}: compact-owned split entrypoint policy source "
                f"must be {policy_source}"
            )
        model_ref = str(row.get("compact_owned_loop_model_version_ref") or "").strip()
        if model_ref and (
            _command_flag_value(
                command,
                "--hybrid-compact-owned-loop-model-version-ref",
            )
            != model_ref
        ):
            problems.append(
                f"row {row_id}: compact-owned split entrypoint model version "
                f"must be {model_ref}"
            )
        if (
            row.get("compact_owned_loop_capture_replay_store_state") is True
            and "--hybrid-compact-owned-loop-capture-replay-store-state" not in command
        ):
            problems.append(
                f"row {row_id}: compact-owned split entrypoint must capture "
                "replay-store state"
            )
    if row.get("require_terminal_compact_owned_nstep") is True:
        unroll_steps = _as_int(row.get("compact_rollout_slab_learner_gate_num_unroll_steps"))
        max_ticks = _as_int(row.get("max_ticks"))
        steps = _as_int(row.get("steps"))
        normal_death_contract = row.get("require_normal_death_terminal_contract") is True
        if unroll_steps is None or unroll_steps <= 1:
            problems.append(
                f"row {row_id}: terminal compact-owned N-step proof requires "
                "num_unroll_steps > 1"
            )
        if max_ticks is None or max_ticks <= 0:
            problems.append(
                f"row {row_id}: terminal compact-owned N-step proof requires max_ticks"
            )
        elif unroll_steps is not None and max_ticks < unroll_steps + 1:
            problems.append(
                f"row {row_id}: terminal compact-owned N-step proof requires "
                "max_ticks >= num_unroll_steps + 1 so at least one active root "
                "precedes the terminal row"
            )
        elif (
            normal_death_contract
            and steps is not None
            and max_ticks <= max(1, steps)
        ):
            problems.append(
                f"row {row_id}: normal-death terminal proof requires "
                "max_ticks > steps so max-tick truncations cannot enter the "
                "compact MuZero learner sample"
            )
        elif (
            not normal_death_contract
            and steps is not None
            and max_ticks > max(1, steps)
        ):
            problems.append(
                f"row {row_id}: terminal compact-owned N-step proof requires "
                "max_ticks <= steps"
            )
        if (
            not normal_death_contract
            and
            steps is not None
            and max_ticks is not None
            and unroll_steps is not None
            and steps < max_ticks + unroll_steps + 1
        ):
            problems.append(
                f"row {row_id}: terminal compact-owned N-step proof requires "
                "steps >= max_ticks + num_unroll_steps + 1 so reset-successor "
                "rows can enter the replay ring"
            )
        if (
            normal_death_contract
            and steps is not None
            and unroll_steps is not None
            and steps < unroll_steps + 1
        ):
            problems.append(
                f"row {row_id}: normal-death terminal proof requires "
                "steps >= num_unroll_steps + 1"
            )
        if _command_flag_value(command, "--max-ticks") != str(row.get("max_ticks")):
            problems.append(
                f"row {row_id}: terminal compact-owned N-step proof command must "
                f"set --max-ticks {row.get('max_ticks')}"
            )
    return problems


def _matched_denominator_preflight_problems(
    row: dict[str, Any],
    command: list[str],
) -> list[str]:
    matched_id = str(row.get("matched_denominator_id") or "").strip()
    if not matched_id:
        return []
    row_id = row.get("row_id")
    problems: list[str] = []
    if row.get("matched_pair_role") != "compact_candidate":
        problems.append(f"row {row_id}: matched denominator role must be compact_candidate")
    if row.get("speed_currency") != MATCHED_COMPACT_SPEED_CURRENCY:
        problems.append(
            f"row {row_id}: matched denominator speed_currency must be "
            f"{MATCHED_COMPACT_SPEED_CURRENCY}"
        )
    if row.get("row_purpose") != MATCHED_DENOMINATOR_ROW_PURPOSE:
        problems.append(
            f"row {row_id}: matched denominator row_purpose must be "
            f"{MATCHED_DENOMINATOR_ROW_PURPOSE}"
        )
    if row.get("promotion_claim") is not False:
        problems.append(f"row {row_id}: matched denominator promotion_claim must be false")
    if not str(row.get("counterpart_manifest_ref") or "").strip():
        problems.append(f"row {row_id}: matched denominator counterpart manifest missing")
    if not str(row.get("counterpart_row_id") or "").strip():
        problems.append(f"row {row_id}: matched denominator counterpart row missing")

    for key, expected in MATCHED_COMPACT_EXPECTED_ROW.items():
        actual = row.get(key)
        if actual != expected:
            problems.append(
                f"row {row_id}: matched denominator {key} "
                f"{actual!r} != expected {expected!r}"
            )

    fixed = row.get("fixed_denominator")
    if not isinstance(fixed, dict):
        problems.append(f"row {row_id}: matched denominator fixed_denominator missing")
    else:
        for key in (
            "matched_denominator_id",
            "matched_pair_role",
            "row_purpose",
            "speed_currency",
        ):
            if fixed.get(key) != row.get(key):
                problems.append(
                    f"row {row_id}: fixed_denominator.{key} must match row"
                )

    expected_flag_values = {
        "--batch-size": row.get("batch_size"),
        "--actor-count": row.get("actor_count"),
        "--steps": row.get("steps"),
        "--warmup-steps": row.get("warmup_steps"),
        "--max-ticks": row.get("max_ticks"),
        "--hybrid-lightzero-array-ceiling-mode": "compact_torch_search_service",
        "--hybrid-lightzero-array-ceiling-input-mode": "host_uint8",
        "--hybrid-lightzero-consumer-root-noise-weight": "0.0",
        "--hybrid-compact-rollout-slab-sample-gate-batch-size": 512,
        "--hybrid-compact-rollout-slab-sample-gate-interval": 8,
        "--hybrid-compact-rollout-slab-sample-gate-replay-pair-capacity": 4096,
        "--hybrid-compact-rollout-slab-learner-gate-support-scale": 300,
        "--hybrid-compact-rollout-slab-learner-gate-num-unroll-steps": 1,
        "--hybrid-compact-rollout-slab-action-mode": "search_feedback",
    }
    for flag, expected in expected_flag_values.items():
        actual = _command_flag_value(command, flag)
        if actual != str(expected):
            problems.append(
                f"row {row_id}: matched denominator command {flag} "
                f"{actual!r} != expected {str(expected)!r}"
            )

    required_flags = (
        "--detach",
        "--hybrid-profile-spawn-result",
        "--no-hybrid-materialize-scalar-timestep",
        "--hybrid-lightzero-array-ceiling-probe",
        "--hybrid-device-only-stack",
        "--hybrid-resident-observation-search",
        "--hybrid-native-actor-buffer",
        "--hybrid-compact-owned-loop-entrypoint",
        "--hybrid-compact-owned-loop-capture-replay-store-state",
    )
    for flag in required_flags:
        if flag not in command:
            problems.append(f"row {row_id}: matched denominator missing {flag}")
    forbidden_flags = (
        "--hybrid-materialize-scalar-timestep",
        "--hybrid-compact-rollout-slab-learner-gate-include-rnd",
        "--require-terminal-compact-owned-nstep",
    )
    for flag in forbidden_flags:
        if flag in command:
            problems.append(f"row {row_id}: matched denominator forbids {flag}")
    return problems


def _expected_root_tape_service_labels(row: dict[str, Any]) -> set[str]:
    labels = {"primary"}
    if row.get("compact_root_tape_compare_fixed_shape_floor") is True:
        labels.add("fixed_shape_floor")
    if row.get("compact_root_tape_compare_mctx") is True:
        labels.add("mctx")
    if row.get("compact_root_tape_compare_model_compile") is True:
        mode = str(row.get("compact_root_tape_model_compile_mode") or "default")
        labels.add(f"model_compile_{mode.replace('-', '_')}")
    if row.get("compact_root_tape_compare_direct_core") is True:
        labels.add("initial_inference_direct_core")
    return labels


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fixed_root_tape_invariant_problems(
    row: dict[str, Any],
    payload: dict[str, Any],
) -> list[str]:
    if row.get("compact_root_tape_compare") is not True:
        return []
    problems: list[str] = []
    expected_records = _as_int(row.get("compact_root_tape_max_records"))
    if payload.get("compact_root_tape_compare_enabled") is not True:
        problems.append("compact_root_tape_compare_enabled is not true")
    error = payload.get("compact_root_tape_error")
    if error not in ("", None):
        problems.append(f"compact_root_tape_error is non-empty: {error!r}")
    record_count = _as_int(payload.get("compact_root_tape_record_count"))
    if expected_records is None or expected_records <= 0:
        problems.append("row compact_root_tape_max_records is invalid")
    elif record_count != expected_records:
        problems.append(
            "compact_root_tape_record_count "
            f"{record_count!r} != expected {expected_records!r}"
        )
    if payload.get("compact_root_tape_reference_label") != "primary":
        problems.append("compact_root_tape_reference_label must be primary")
    service_labels = payload.get("compact_root_tape_service_labels")
    if not isinstance(service_labels, list):
        problems.append("compact_root_tape_service_labels must be a list")
        service_labels = []
    expected_service_labels = _expected_root_tape_service_labels(row)
    if set(service_labels) != expected_service_labels:
        problems.append(
            "compact_root_tape_service_labels mismatch: expected "
            f"{sorted(expected_service_labels)!r}, got {sorted(service_labels)!r}"
        )
    report = payload.get("compact_root_tape_comparison")
    if not isinstance(report, dict):
        problems.append("compact_root_tape_comparison must be a dict")
        return problems
    if report.get("schema_id") != COMPACT_ROOT_TAPE_COMPARISON_SCHEMA_ID:
        problems.append("compact_root_tape_comparison schema_id mismatch")
    if _as_int(report.get("record_count")) != record_count:
        problems.append("comparison record_count does not match payload")
    if report.get("reference_label") != "primary":
        problems.append("comparison reference_label must be primary")
    if (
        row.get("compact_root_tape_compare_model_compile") is True
        and row.get("compact_root_tape_require_model_compile", True) is not False
    ):
        backend = report.get("backend")
        mode = str(row.get("compact_root_tape_model_compile_mode") or "default")
        label = f"model_compile_{mode.replace('-', '_')}"
        stats = backend.get(label) if isinstance(backend, dict) else None
        if not isinstance(stats, dict):
            problems.append(f"missing backend stats for {label}")
        else:
            used_count = _as_int(stats.get("model_compile_used_count"))
            if used_count != record_count:
                problems.append(
                    f"{label} model_compile_used_count {used_count!r} != "
                    f"record_count {record_count!r}"
                )
    metadata = report.get("tape_metadata")
    if not isinstance(metadata, dict):
        problems.append("comparison tape_metadata must be a dict")
        metadata = {}
    if _as_int(metadata.get("record_count")) != record_count:
        problems.append("tape_metadata record_count does not match payload")
    if expected_records is not None and _as_int(metadata.get("max_records")) != expected_records:
        problems.append("tape_metadata max_records does not match row")
    if metadata.get("profile_only") is not True:
        problems.append("tape_metadata profile_only must be true")
    if metadata.get("calls_train_muzero") is not False:
        problems.append("tape_metadata calls_train_muzero must be false")
    if metadata.get("action_mode") != row.get("compact_rollout_slab_action_mode"):
        problems.append("tape_metadata action_mode does not match row")
    if _as_float(metadata.get("root_noise_weight")) != 0.0:
        problems.append("tape_metadata root_noise_weight must be 0.0")
    backend = report.get("backend")
    if not isinstance(backend, dict):
        problems.append("comparison backend must be a dict")
        backend = {}
    for label in sorted(expected_service_labels):
        stats = backend.get(label)
        if not isinstance(stats, dict):
            problems.append(f"comparison backend missing {label}")
            continue
        if record_count is not None and _as_int(stats.get("run_count")) != record_count:
            problems.append(f"comparison backend {label} run_count mismatch")
        active = _as_int(stats.get("active_root_count"))
        if active is None or active <= 0:
            problems.append(f"comparison backend {label} active_root_count must be > 0")
    comparison = report.get("comparison")
    if not isinstance(comparison, dict):
        problems.append("comparison section must be a dict")
        comparison = {}
    for label in sorted(expected_service_labels.difference({"primary"})):
        comparison_key = f"{label}_vs_primary"
        stats = comparison.get(comparison_key)
        if not isinstance(stats, dict):
            problems.append(f"missing {comparison_key} comparison")
            continue
        if record_count is not None and _as_int(stats.get("record_count")) != record_count:
            problems.append(f"{comparison_key} record_count mismatch")
        for key in (
            "active_root_count",
            "action_match_fraction",
            "visit_l1_mean",
            "visit_l1_max",
            "root_value_abs_diff_mean",
            "root_value_abs_diff_max",
        ):
            if stats.get(key) is None:
                problems.append(f"{comparison_key} missing {key}")
    return problems


def _compact_owned_candidate_invariant_problems(
    row: dict[str, Any],
    payload: dict[str, Any],
    summary: dict[str, Any],
) -> list[str]:
    if not _is_compact_owned_candidate_row(row) or payload.get("ok") is not True:
        return []
    problems: list[str] = []

    def require_exact(key: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            problems.append(f"{key} {actual!r} != expected {expected!r}")

    def require_positive(key: str, actual: Any) -> None:
        value = _as_float(actual)
        if value is None or value <= 0.0:
            problems.append(f"{key} must be > 0, got {actual!r}")

    def require_zero(key: str, actual: Any) -> None:
        value = _as_float(actual)
        if value is None or value != 0.0:
            problems.append(f"{key} must be 0, got {actual!r}")

    require_exact("profile_only", payload.get("profile_only"), True)
    require_exact("calls_train_muzero", payload.get("calls_train_muzero"), False)
    require_exact("touches_live_runs", payload.get("touches_live_runs"), False)
    require_exact("death_mode", payload.get("death_mode"), row.get("death_mode"))
    require_exact(
        "compact_rollout_slab_enabled",
        payload.get("compact_rollout_slab_enabled"),
        True,
    )
    require_positive(
        "compact_rollout_slab_calls", payload.get("compact_rollout_slab_calls")
    )
    require_positive(
        "compact_rollout_slab_total_roots",
        payload.get("compact_rollout_slab_total_roots"),
    )
    require_positive(
        "compact_rollout_slab_committed_index_rows",
        summary.get("compact_rollout_slab_committed_index_rows"),
    )
    if row.get("compact_rollout_slab_action_mode") is not None:
        require_exact(
            "compact_rollout_slab_action_mode",
            payload.get("compact_rollout_slab_action_mode"),
            row.get("compact_rollout_slab_action_mode"),
        )

    require_exact(
        "resident_observation_used",
        summary.get("resident_observation_used"),
        True,
    )
    require_zero(
        "resident_observation_host_fallback_count",
        summary.get("resident_observation_host_fallback_count"),
    )
    require_zero(
        "resident_observation_h2d_bytes",
        summary.get("resident_observation_h2d_bytes"),
    )
    require_zero(
        "resident_observation_d2h_bytes",
        summary.get("resident_observation_d2h_bytes"),
    )
    require_zero("obs_h2d_bytes", summary.get("obs_h2d_bytes"))
    require_zero(
        "committed_replay_payload_d2h_bytes",
        summary.get("committed_replay_payload_d2h_bytes"),
    )
    require_zero(
        "replay_payload_d2h_bytes",
        summary.get("replay_payload_d2h_bytes"),
    )
    require_zero(
        "accounting_scalar_materialization_sec",
        summary.get("accounting_scalar_materialization_sec"),
    )
    require_zero("python_rows_materialized", summary.get("python_rows_materialized"))

    require_exact(
        "compact_rollout_slab_sample_gate_enabled",
        payload.get("compact_rollout_slab_sample_gate_enabled"),
        True,
    )
    require_positive(
        "compact_rollout_slab_sample_gate_calls",
        payload.get("compact_rollout_slab_sample_gate_calls"),
    )
    require_positive(
        "compact_rollout_slab_sample_gate_opportunities",
        payload.get("compact_rollout_slab_sample_gate_opportunities"),
    )
    sample_calls = _as_int(payload.get("compact_rollout_slab_sample_gate_calls"))
    sample_skips = _as_int(payload.get("compact_rollout_slab_sample_gate_skipped_count"))
    sample_opportunities = _as_int(
        payload.get("compact_rollout_slab_sample_gate_opportunities")
    )
    if (
        sample_calls is None
        or sample_skips is None
        or sample_opportunities is None
        or sample_calls + sample_skips != sample_opportunities
    ):
        problems.append(
            "compact_rollout_slab_sample_gate_calls + skipped_count must equal "
            "opportunities"
        )
    require_positive(
        "compact_rollout_slab_sample_gate_index_rows",
        summary.get("compact_rollout_slab_sample_gate_index_rows"),
    )
    require_positive(
        "compact_rollout_slab_sample_gate_target_rows",
        summary.get("compact_rollout_slab_sample_gate_target_rows"),
    )
    require_positive(
        "compact_rollout_slab_sample_gate_sample_rows",
        summary.get("compact_rollout_slab_sample_gate_sample_rows"),
    )
    require_zero(
        "compact_rollout_slab_sample_gate_mock_base_env_timestep_rows",
        payload.get("compact_rollout_slab_sample_gate_mock_base_env_timestep_rows"),
    )

    require_exact(
        "compact_rollout_slab_learner_gate_enabled",
        payload.get("compact_rollout_slab_learner_gate_enabled"),
        True,
    )
    require_exact(
        "compact_rollout_slab_learner_gate_impl",
        payload.get("compact_rollout_slab_learner_gate_impl"),
        "compact_muzero",
    )
    require_exact(
        "compact_rollout_slab_learner_gate_toy_probe",
        payload.get("compact_rollout_slab_learner_gate_toy_probe"),
        False,
    )
    require_exact(
        "compact_rollout_slab_learner_gate_real_muzero_update",
        payload.get("compact_rollout_slab_learner_gate_real_muzero_update"),
        True,
    )
    require_exact(
        "compact_rollout_slab_learner_gate_include_rnd",
        payload.get("compact_rollout_slab_learner_gate_include_rnd"),
        False,
    )
    require_positive(
        "compact_rollout_slab_learner_gate_calls",
        payload.get("compact_rollout_slab_learner_gate_calls"),
    )
    require_positive(
        "compact_rollout_slab_learner_gate_updates",
        payload.get("compact_rollout_slab_learner_gate_updates"),
    )
    require_positive(
        "compact_rollout_slab_learner_gate_sample_rows",
        summary.get("compact_rollout_slab_learner_gate_sample_rows"),
    )
    if (
        _as_int(summary.get("compact_rollout_slab_learner_gate_sample_rows"))
        != _as_int(summary.get("compact_rollout_slab_sample_gate_sample_rows"))
    ):
        problems.append(
            "compact_rollout_slab_learner_gate_sample_rows must match "
            "compact_rollout_slab_sample_gate_sample_rows"
        )
    require_exact(
        "compact_rollout_slab_learner_gate_train_steps",
        payload.get("compact_rollout_slab_learner_gate_train_steps"),
        row.get("compact_rollout_slab_learner_gate_train_steps"),
    )
    require_exact(
        "compact_rollout_slab_learner_gate_num_unroll_steps",
        payload.get("compact_rollout_slab_learner_gate_num_unroll_steps"),
        row.get("compact_rollout_slab_learner_gate_num_unroll_steps", 1),
    )
    row_unroll_steps = _as_int(row.get("compact_rollout_slab_learner_gate_num_unroll_steps", 1))
    last_sample = None
    last_learner = None
    if row_unroll_steps is not None and row_unroll_steps > 1:
        last_sample = payload.get("compact_rollout_slab_sample_gate_last_telemetry")
        if not isinstance(last_sample, dict):
            problems.append("compact_rollout_slab_sample_gate_last_telemetry must be present")
            last_sample = None
        else:
            require_exact(
                "compact_rollout_slab_sample_gate_explicit_unroll_targets",
                last_sample.get("compact_rollout_slab_sample_gate_explicit_unroll_targets"),
                True,
            )
            require_exact(
                "compact_rollout_slab_sample_gate_num_unroll_steps",
                last_sample.get("compact_rollout_slab_sample_gate_num_unroll_steps"),
                row_unroll_steps,
            )
        last_learner = payload.get("compact_rollout_slab_learner_gate_last_telemetry")
        if not isinstance(last_learner, dict):
            problems.append("compact_rollout_slab_learner_gate_last_telemetry must be present")
            last_learner = None
        else:
            require_exact(
                "compact_rollout_slab_learner_gate_last_num_unroll_steps",
                last_learner.get("compact_rollout_slab_learner_gate_num_unroll_steps"),
                row_unroll_steps,
            )
    if row.get("require_terminal_compact_owned_nstep") is True:
        require_positive("terminal_row_count", payload.get("terminal_row_count"))
        if last_sample is None:
            problems.append(
                "terminal compact-owned N-step row requires sample gate last telemetry"
            )
        else:
            require_exact(
                "compact_rollout_slab_sample_gate_terminal_unroll_windows_supported",
                last_sample.get(
                    "compact_rollout_slab_sample_gate_terminal_unroll_windows_supported"
                ),
                True,
            )
            require_positive(
                "compact_rollout_slab_sample_gate_terminal_sample_row_count",
                last_sample.get(
                    "compact_rollout_slab_sample_gate_terminal_sample_row_count"
                ),
            )
            require_positive(
                "compact_rollout_slab_sample_gate_next_final_observation_row_count",
                last_sample.get(
                    "compact_rollout_slab_sample_gate_next_final_observation_row_count"
                ),
            )
            require_exact(
                "compact_rollout_slab_sample_gate_resident_terminal_final_observation_used",
                last_sample.get(
                    "compact_rollout_slab_sample_gate_resident_terminal_final_observation_used"
                ),
                True,
            )
            require_positive(
                "compact_rollout_slab_sample_gate_terminal_unroll_value_target_row_count",
                last_sample.get(
                    "compact_rollout_slab_sample_gate_terminal_unroll_value_target_row_count"
                ),
            )
            require_exact(
                "compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode",
                last_sample.get(
                    "compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode"
                ),
                "stock_terminal_no_bootstrap_return_discount_1.0",
            )
        if last_learner is None:
            problems.append(
                "terminal compact-owned N-step row requires learner gate last telemetry"
            )
        else:
            learner_telemetry = last_learner.get(
                "compact_rollout_slab_learner_gate_compact_muzero_telemetry"
            )
            if not isinstance(learner_telemetry, dict):
                problems.append(
                    "compact_rollout_slab_learner_gate_compact_muzero_telemetry "
                    "must be present"
                )
            else:
                require_positive(
                    "compact_muzero_learner_done_count",
                    learner_telemetry.get("compact_muzero_learner_done_count"),
                )
                require_positive(
                    "compact_muzero_learner_value_valid_count",
                    learner_telemetry.get("compact_muzero_learner_value_valid_count"),
                )
    if row.get("require_normal_death_terminal_contract") is True:
        evidence_id = str(
            row.get("normal_death_terminal_contract_evidence_id")
            or f"{row.get('label') or row.get('row_id')}:normal_death"
        )
        evidence_refs = row.get("normal_death_terminal_contract_evidence_refs")
        if not isinstance(evidence_refs, list) or not evidence_refs:
            evidence_refs = [
                str(row.get("result_artifact_ref") or row.get("label") or row.get("row_id"))
            ]
        try:
            contract = build_normal_collision_death_contract_from_profile_result_v1(
                payload,
                evidence_id=evidence_id,
                evidence_refs=tuple(str(ref) for ref in evidence_refs),
            )
        except CompactDeathTerminalContractError as exc:
            problems.append(f"normal_death_terminal_contract: {exc}")
        else:
            summary["normal_death_terminal_contract"] = contract
            summary["normal_death_terminal_contract_schema_id"] = contract.get(
                "compact_death_terminal_contract_schema_id"
            )
            summary["normal_death_terminal_contract_evidence"] = contract.get(
                "normal_collision_death_evidence"
            )
            summary["normal_death_terminal_contract_evidence_id"] = contract.get(
                "normal_collision_death_evidence_id"
            )
            summary["normal_death_terminal_contract_evidence_refs"] = list(
                contract.get("normal_collision_death_evidence_refs") or []
            )
            summary["normal_death_terminal_contract_promotion_gate_satisfied"] = (
                contract.get("compact_death_terminal_contract_promotion_gate_satisfied")
            )
            if (
                contract.get("compact_death_terminal_contract_promotion_gate_satisfied")
                is not True
            ):
                problems.append("normal_death_terminal_contract promotion gate not satisfied")
    if _is_compact_owned_split_entrypoint_row(row):
        require_exact(
            "compact_owned_loop_entrypoint_enabled",
            payload.get("compact_owned_loop_entrypoint_enabled"),
            True,
        )
        require_exact(
            "compact_owned_loop_schema_id",
            payload.get("compact_owned_loop_schema_id"),
            COMPACT_OWNED_LOOP_SCHEMA_ID,
        )
        require_exact(
            "compact_owned_loop_profile_only",
            payload.get("compact_owned_loop_profile_only"),
            True,
        )
        require_exact(
            "compact_owned_loop_calls_train_muzero",
            payload.get("compact_owned_loop_calls_train_muzero"),
            False,
        )
        require_exact(
            "compact_owned_loop_touches_live_runs",
            payload.get("compact_owned_loop_touches_live_runs"),
            False,
        )
        require_exact(
            "compact_owned_loop_replay_store_owned",
            payload.get("compact_owned_loop_replay_store_owned"),
            True,
        )
        require_exact(
            "compact_owned_loop_policy_version_handoff",
            payload.get("compact_owned_loop_policy_version_handoff"),
            True,
        )
        require_exact(
            "compact_owned_loop_policy_version_ref",
            payload.get("compact_owned_loop_policy_version_ref"),
            row.get("compact_owned_loop_policy_version_ref"),
        )
        expected_model_ref = row.get("compact_owned_loop_model_version_ref") or None
        require_exact(
            "compact_owned_loop_model_version_ref",
            payload.get("compact_owned_loop_model_version_ref"),
            expected_model_ref,
        )
        require_exact(
            "compact_owned_loop_policy_source",
            payload.get("compact_owned_loop_policy_source"),
            row.get("compact_owned_loop_policy_source"),
        )
        telemetry = payload.get("compact_owned_loop_telemetry")
        if not isinstance(telemetry, dict):
            problems.append("compact_owned_loop_telemetry must be present")
        else:
            require_exact(
                "compact_owned_loop_telemetry_policy_version_ref",
                telemetry.get("compact_owned_loop_policy_version_ref"),
                row.get("compact_owned_loop_policy_version_ref"),
            )
            sample_metadata = telemetry.get(
                "compact_owned_loop_sample_gate_last_sample_metadata"
            )
            if not isinstance(sample_metadata, dict):
                problems.append(
                    "compact_owned_loop_sample_gate_last_sample_metadata must be present"
                )
            else:
                require_exact(
                    "compact_owned_loop_sample_metadata_replay_store_owned",
                    sample_metadata.get("compact_owned_loop_replay_store_owned"),
                    True,
                )
                require_exact(
                    "compact_owned_loop_sample_metadata_policy_version_ref",
                    sample_metadata.get("compact_owned_loop_policy_version_ref"),
                    row.get("compact_owned_loop_policy_version_ref"),
                )
        if row.get("compact_owned_loop_capture_replay_store_state") is True:
            state_metadata = payload.get("compact_owned_loop_replay_store_state_metadata")
            if not isinstance(state_metadata, dict):
                problems.append(
                    "compact_owned_loop_replay_store_state_metadata must be present"
                )
            else:
                require_exact(
                    "compact_owned_loop_replay_store_state_schema_id",
                    state_metadata.get("schema_id"),
                    COMPACT_REPLAY_STORE_STATE_SCHEMA_ID,
                )
                require_exact(
                    "compact_owned_loop_state_replay_store_owned",
                    state_metadata.get("compact_owned_loop_replay_store_owned"),
                    True,
                )
    require_positive(
        "compact_rollout_slab_learner_gate_input_bytes",
        payload.get("compact_rollout_slab_learner_gate_input_bytes"),
    )
    return problems


def _matched_denominator_invariant_problems(
    row: dict[str, Any],
    payload: dict[str, Any],
    summary: dict[str, Any],
) -> list[str]:
    if not str(row.get("matched_denominator_id") or "").strip():
        return []
    problems: list[str] = []

    def require_exact(key: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            problems.append(f"{key} {actual!r} != expected {expected!r}")

    def require_positive(key: str, actual: Any) -> None:
        value = _as_float(actual)
        if value is None or value <= 0.0:
            problems.append(f"{key} must be > 0, got {actual!r}")

    require_exact("payload.ok", payload.get("ok"), True)
    require_exact(
        "summary.speed_currency",
        summary.get("speed_currency"),
        MATCHED_COMPACT_SPEED_CURRENCY,
    )
    require_exact(
        "summary.matched_pair_role",
        summary.get("matched_pair_role"),
        "compact_candidate",
    )
    require_exact(
        "summary.row_purpose",
        summary.get("row_purpose"),
        MATCHED_DENOMINATOR_ROW_PURPOSE,
    )
    require_exact("summary.promotion_claim", summary.get("promotion_claim"), False)
    require_exact("summary.batch_size", summary.get("batch_size"), 1024)
    require_exact("summary.actor_count", summary.get("actor_count"), 16)
    require_exact("payload.steps", payload.get("steps"), 60)
    require_exact("payload.warmup_steps", payload.get("warmup_steps"), 15)
    require_exact("payload.max_ticks", payload.get("max_ticks"), 2000)
    require_exact(
        "compact_rollout_slab_sample_gate_batch_size",
        summary.get("compact_rollout_slab_sample_gate_batch_size"),
        512,
    )
    require_exact(
        "compact_rollout_slab_sample_gate_interval",
        summary.get("compact_rollout_slab_sample_gate_interval"),
        8,
    )
    require_exact(
        "compact_rollout_slab_learner_gate_support_scale",
        summary.get("compact_rollout_slab_learner_gate_support_scale"),
        300,
    )
    require_exact(
        "compact_rollout_slab_learner_gate_num_unroll_steps",
        summary.get("compact_rollout_slab_learner_gate_num_unroll_steps"),
        1,
    )
    require_exact("root_noise_weight", summary.get("root_noise_weight"), 0.0)
    require_positive("steps_per_sec", summary.get("steps_per_sec"))
    require_positive("physical_rows_per_sec", summary.get("physical_rows_per_sec"))
    require_positive("rows_per_step", summary.get("rows_per_step"))
    require_positive("total_roots", summary.get("total_roots"))
    terminal_count = _as_int(payload.get("terminal_row_count"))
    if terminal_count not in (None, 0):
        problems.append(
            "matched denominator speed row must be non-terminal; "
            f"terminal_row_count={terminal_count!r}"
        )
    search_impl = str(summary.get("compact_rollout_slab_search_impl") or "")
    if "compact_torch" not in search_impl:
        problems.append(
            "compact_rollout_slab_search_impl must be compact Torch, "
            f"got {search_impl!r}"
        )
    return problems


def _root_tape_summary_fields(payload: dict[str, Any]) -> dict[str, Any]:
    report = payload.get("compact_root_tape_comparison")
    fields: dict[str, Any] = {
        "compact_root_tape_compare_enabled": payload.get(
            "compact_root_tape_compare_enabled"
        ),
        "compact_root_tape_record_count": payload.get("compact_root_tape_record_count"),
        "compact_root_tape_skipped_record_count": payload.get(
            "compact_root_tape_skipped_record_count"
        ),
        "compact_root_tape_reference_label": payload.get(
            "compact_root_tape_reference_label"
        ),
        "compact_root_tape_service_labels": payload.get(
            "compact_root_tape_service_labels"
        ),
        "compact_root_tape_error": payload.get("compact_root_tape_error"),
    }
    if not isinstance(report, dict):
        return fields
    metadata = report.get("tape_metadata")
    if isinstance(metadata, dict):
        fields.update(
            {
                "compact_root_tape_metadata_profile_only": metadata.get("profile_only"),
                "compact_root_tape_metadata_calls_train_muzero": metadata.get(
                    "calls_train_muzero"
                ),
                "compact_root_tape_metadata_action_mode": metadata.get("action_mode"),
                "compact_root_tape_metadata_root_noise_weight": metadata.get(
                    "root_noise_weight"
                ),
                "compact_root_tape_metadata_max_records": metadata.get("max_records"),
            }
        )
    backend = report.get("backend")
    if isinstance(backend, dict):
        for label, stats in sorted(backend.items()):
            if not isinstance(stats, dict):
                continue
            prefix = f"compact_root_tape_backend_{label}"
            fields.update(
                {
                    f"{prefix}_run_count": stats.get("run_count"),
                    f"{prefix}_active_root_count": stats.get("active_root_count"),
                    f"{prefix}_run_sec": stats.get("run_sec"),
                    f"{prefix}_run_sec_per_active_root": stats.get(
                        "run_sec_per_active_root"
                    ),
                    f"{prefix}_h2d_bytes": stats.get("h2d_bytes"),
                    f"{prefix}_d2h_bytes": stats.get("d2h_bytes"),
                    f"{prefix}_model_compile_requested_count": stats.get(
                        "model_compile_requested_count"
                    ),
                    f"{prefix}_model_compile_used_count": stats.get(
                        "model_compile_used_count"
                    ),
                    f"{prefix}_model_compile_cache_hit_count": stats.get(
                        "model_compile_cache_hit_count"
                    ),
                    f"{prefix}_model_compile_runtime_status_counts": stats.get(
                        "model_compile_runtime_status_counts"
                    ),
                }
            )
    comparison = report.get("comparison")
    if isinstance(comparison, dict):
        for comparison_key, comparison_stats in sorted(comparison.items()):
            if not isinstance(comparison_stats, dict):
                continue
            prefix = f"compact_root_tape_{comparison_key}"
            fields.update(
                {
                    f"{prefix}_record_count": comparison_stats.get("record_count"),
                    f"{prefix}_active_root_count": comparison_stats.get(
                        "active_root_count"
                    ),
                    f"{prefix}_action_match_count": comparison_stats.get(
                        "action_match_count"
                    ),
                    f"{prefix}_action_match_fraction": comparison_stats.get(
                        "action_match_fraction"
                    ),
                    f"{prefix}_visit_l1_mean": comparison_stats.get("visit_l1_mean"),
                    f"{prefix}_visit_l1_max": comparison_stats.get("visit_l1_max"),
                    f"{prefix}_root_value_abs_diff_mean": comparison_stats.get(
                        "root_value_abs_diff_mean"
                    ),
                    f"{prefix}_root_value_abs_diff_max": comparison_stats.get(
                        "root_value_abs_diff_max"
                    ),
                }
            )
    return fields


def _base_record(
    row: dict[str, Any],
    output_dir: Path,
    stdout_path: Path,
    started_at: str,
) -> dict[str, Any]:
    return {
        "schema_id": RESULT_SCHEMA_ID,
        "row_id": str(row.get("row_id")),
        "label": row.get("label"),
        "started_at": started_at,
        "collected_at": _utc_timestamp(),
        "stdout_path": str(stdout_path),
        "row": row,
        "output_dir": str(output_dir),
    }


def _finalize_profile_payload(
    row: dict[str, Any],
    record: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    label_problems = _payload_label_problems(row, payload)
    if label_problems:
        record.update(
            {
                "status": "profile_label_mismatch",
                "problem": "manifest/payload label mismatch: "
                + "; ".join(label_problems),
                "compact": payload,
            }
        )
        return record
    root_tape_problems = _fixed_root_tape_invariant_problems(row, payload)
    if root_tape_problems:
        record.update(
            {
                "status": "fixed_root_tape_invariant_failed",
                "problem": "; ".join(root_tape_problems),
                "compact": payload,
            }
        )
        return record
    summary = _compact_line(row, payload)
    compact_owned_problems = _compact_owned_candidate_invariant_problems(
        row,
        payload,
        summary,
    )
    if compact_owned_problems:
        record.update(
            {
                "status": "compact_owned_candidate_invariant_failed",
                "problem": "; ".join(compact_owned_problems),
                "compact": payload,
            }
        )
        return record
    matched_denominator_problems = _matched_denominator_invariant_problems(
        row,
        payload,
        summary,
    )
    if matched_denominator_problems:
        record.update(
            {
                "status": "matched_denominator_invariant_failed",
                "problem": "; ".join(matched_denominator_problems),
                "compact": payload,
            }
        )
        return record
    record.update(
        {
            "status": "complete" if payload.get("ok") is True else "profile_failed",
            "problem": None if payload.get("ok") is True else payload.get("error"),
            "compact": payload,
            "summary": summary,
        }
    )
    return record


def _timeout_seconds(row: dict[str, Any], key: str, default: float | None) -> float | None:
    value = row.get(key, default)
    if value is None:
        return None
    return float(value)


def _decode_timeout_output(exc: subprocess.TimeoutExpired) -> str:
    output = exc.stdout if exc.stdout is not None else exc.output
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return str(output)


def _collect_modal_function_call(function_call_id: str, timeout: float | None) -> Any:
    if modal is None:
        raise RuntimeError("modal package is not available; install the modal extra")
    call = modal.FunctionCall.from_id(function_call_id)
    if timeout is None:
        return call.get()
    return call.get(timeout=timeout)


def _run_row(
    row: dict[str, Any],
    output_dir: Path,
    *,
    default_row_timeout_sec: float | None = None,
    default_result_timeout_sec: float | None = None,
) -> dict[str, Any]:
    row_id = str(row.get("row_id"))
    stdout_path = output_dir / f"row_{row_id}_stdout.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(part) for part in row.get("command") or []]
    started_at = _utc_timestamp()
    row_timeout = _timeout_seconds(row, "row_timeout_sec", default_row_timeout_sec)
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=row_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_timeout_output(exc)
        stdout_path.write_text(stdout, encoding="utf-8")
        record = _base_record(row, output_dir, stdout_path, started_at)
        record.update(
            {
                "status": "command_timeout",
                "problem": f"command timed out after {row_timeout} sec",
                "returncode": None,
                "compact": None,
            }
        )
        return record
    stdout = completed.stdout or ""
    stdout_path.write_text(stdout, encoding="utf-8")
    record = _base_record(row, output_dir, stdout_path, started_at)
    record["returncode"] = completed.returncode
    if completed.returncode != 0:
        record.update(
            {
                "status": "command_failed",
                "problem": f"command exited {completed.returncode}",
                "compact": None,
            }
        )
        return record
    try:
        payload = _extract_last_json_object(stdout)
    except ValueError as exc:
        record.update({"status": "parse_failed", "problem": str(exc), "compact": None})
        return record
    if _row_launch_mode(row) != LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT:
        return _finalize_profile_payload(row, record, payload)

    if payload.get("schema_id") != SPAWN_SCHEMA_ID:
        record.update(
            {
                "status": "launch_parse_failed",
                "problem": "detached row did not print a spawn payload",
                "compact": payload,
            }
        )
        return record
    launch_label_problems = _payload_label_problems(row, payload)
    if launch_label_problems:
        record.update(
            {
                "status": "launch_label_mismatch",
                "problem": "manifest/launch label mismatch: "
                + "; ".join(launch_label_problems),
                "compact": None,
                "launch": payload,
            }
        )
        return record
    function_call_id = str(payload.get("function_call_id") or "")
    if not function_call_id:
        record.update(
            {
                "status": "launch_parse_failed",
                "problem": "spawn payload missing function_call_id",
                "compact": None,
                "launch": payload,
            }
        )
        return record
    record["function_call_id"] = function_call_id
    record["launch"] = payload
    result_timeout = _timeout_seconds(
        row,
        "result_timeout_sec",
        default_result_timeout_sec,
    )
    try:
        result_payload = _collect_modal_function_call(function_call_id, result_timeout)
    except Exception as exc:  # noqa: BLE001 - preserve the capture failure in JSON.
        record.update(
            {
                "status": "collect_failed",
                "problem": f"modal FunctionCall result collection failed: {exc}",
                "compact": None,
            }
        )
        return record
    if not isinstance(result_payload, dict):
        record.update(
            {
                "status": "collect_failed",
                "problem": (
                    "modal FunctionCall result was not a JSON object: "
                    f"{type(result_payload).__name__}"
                ),
                "compact": None,
            }
        )
        return record
    return _finalize_profile_payload(row, record, result_payload)


def _write_result(output_dir: Path, record: dict[str, Any]) -> None:
    row_id = str(record.get("row_id"))
    _write_json(output_dir / f"row_{row_id}_result.json", record)
    if isinstance(record.get("summary"), dict):
        print(json.dumps(record["summary"], sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "row_id": row_id,
                    "status": record.get("status"),
                    "problem": record.get("problem"),
                    "returncode": record.get("returncode"),
                },
                sort_keys=True,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rows", action="append", default=[])
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--row-timeout-sec", type=float, default=None)
    parser.add_argument("--collect-timeout-sec", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = _load_json(args.manifest)
    _validate_manifest(manifest)
    row_ids = _parse_rows(args.rows)
    rows = _selected_rows(manifest, row_ids)
    output_dir = args.output_root / _safe_id(str(manifest["experiment_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest.json", manifest)

    if args.dry_run:
        for row in rows:
            print(row.get("command_text") or " ".join(str(part) for part in row["command"]))
        return

    max_workers = max(1, int(args.parallel))
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_row,
                row,
                output_dir,
                default_row_timeout_sec=args.row_timeout_sec,
                default_result_timeout_sec=args.collect_timeout_sec,
            )
            for row in rows
        ]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            _write_result(output_dir, record)
            _append_jsonl(output_dir / "rows.jsonl", record)
            results.append(record)
    results.sort(key=lambda item: str(item.get("row_id")))
    _write_json(output_dir / "collected_results.json", results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
