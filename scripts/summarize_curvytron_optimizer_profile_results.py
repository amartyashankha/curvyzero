#!/usr/bin/env python3
"""Summarize local CurvyTron optimizer profile result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_DIR = Path(
    "artifacts/local/curvytron_optimizer_profile_results/"
    "opt-stock-frozen-profile-first-wave-20260512e"
)

PROFILE_ATTESTATION_REQUIRED_COMPACT_KEYS = (
    "schema_id",
    "mode",
    "ok",
    "status",
    "called_train_muzero",
    "trainer_entrypoint",
    "semantic_identity",
)
PROFILE_ATTESTATION_REQUIRED_COMMAND_KEYS = (
    "env_variant",
    "env_manager_type",
    "collector_env_num",
    "num_simulations",
    "batch_size",
    "collect_search_backend",
    "collect_search_ctree_backend",
    "disable_death_for_profile",
    "opponent_death_mode",
    "policy_observation_backend",
    "policy_trail_render_mode",
    "policy_bonus_render_mode",
    "source_state_trail_render_mode",
    "source_state_bonus_render_mode",
    "policy_observation_contract_id",
    "observation_contract",
    "exploration_bonus",
    "skip_lightzero_eval_in_profile",
)
PROFILE_ATTESTATION_REQUIRED_OBSERVATION_CONTRACT_KEYS = (
    "contract_id",
    "backend",
    "surface_label",
    "stack_shape",
    "single_frame_shape",
    "target_frame_size",
    "source_frame_size",
    "grayscale_method",
    "downsample_method",
    "perspective_schema_id",
)
PROFILE_ATTESTATION_REQUIRED_COUNT_KEYS = (
    "env_steps_collected",
    "env_steps_collected_raw",
    "env_steps_collected_source",
    "env_steps_collected_uses_fallback",
    "mcts_search_calls",
    "mcts_search_root_sum",
)
PROFILE_ATTESTATION_REQUIRED_DERIVED_KEYS = (
    "steps_per_sec",
    "steps_per_sec_currency",
    "steps_per_sec_source",
    "steps_per_sec_uses_fallback_denominator",
    "mcts_root_batch_mean",
)
PROFILE_ATTESTATION_REQUIRED_TIMER_KEYS = (
    "train_muzero_wall",
    "collector_collect",
    "policy_forward_collect",
    "mcts_search",
)
BATCHED_PROFILE_REQUIRED_TIMER_KEYS = (
    "batched_profile_env_manager_step",
    "batched_profile_renderer_render",
    "batched_profile_surface_stack_update",
)
PROFILE_ATTESTATION_REQUIRED_SEMANTIC_IDENTITY_KEYS = (
    "schema_id",
    "observation_raw_dtype",
    "observation_model_dtype",
    "scalar_materialization_semantics",
    "lightzero_to_play_mode",
    "zero_mask_filtering_semantics",
    "consumer_semantics",
    "collect_search_backend",
    "collect_search_ctree_backend",
    "env_steps_collected_source",
    "speed_currency",
)
DIRECT_CTREE_GPU_LATENT_BACKEND = "direct_ctree_gpu_latent"
FLAT_A3_CTREE_BACKEND = "flat_a3"
COLLECTOR_ENVSTEP_SOURCE = "collector_envstep_delta"
DIRECT_CTREE_REQUIRED_COUNT_KEYS = (
    "collect_search_backend_direct_ctree_gpu_latent_calls",
    "collect_search_backend_fallback_calls",
    "collect_search_backend_output_rows",
)
GATE_A_SPEED_CURRENCY = "stock_train_muzero_profile_env_steps_per_sec"
FALLBACK_STOCK_PROFILE_SPEED_CURRENCY = (
    "stock_train_muzero_profile_mcts_roots_per_sec_fallback"
)
GATE_A_DENOMINATOR_COMMAND_KEYS = (
    "env_variant",
    "env_manager_type",
    "collector_env_num",
    "n_episode",
    "num_simulations",
    "batch_size",
    "source_max_steps",
    "max_train_iter",
    "max_env_step",
    "save_ckpt_after_iter",
    "stop_after_learner_train_calls",
    "disable_death_for_profile",
    "opponent_death_mode",
    "policy_observation_backend",
    "policy_trail_render_mode",
    "policy_bonus_render_mode",
    "source_state_trail_render_mode",
    "source_state_bonus_render_mode",
    "policy_observation_contract_id",
    "reward_variant",
    "exploration_bonus",
    "skip_lightzero_eval_in_profile",
    "background_eval_enabled",
    "background_gif_enabled",
)
GATE_A_OPTIONAL_DENOMINATOR_COMMAND_KEYS = (
    "max_train_iter",
    "max_env_step",
    "stop_after_learner_train_calls",
    "background_eval_enabled",
    "background_gif_enabled",
)
GATE_A_RUNTIME_KEYS = (
    "compute",
)
GATE_A_GPU_KEYS = (
    "requested_compute",
)
GATE_A_REQUIRED_COUNT_KEYS = (
    "learner_train_calls",
    "replay_sample_calls",
)


def _expected_speed_currency_for_source(source: Any) -> str:
    if source == COLLECTOR_ENVSTEP_SOURCE:
        return GATE_A_SPEED_CURRENCY
    if source == "mcts_search_root_sum_profile_fallback":
        return FALLBACK_STOCK_PROFILE_SPEED_CURRENCY
    return "unknown_profile_steps_per_sec"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _round(value: Any, digits: int = 2) -> Any:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, digits)
    return value


def _is_present(mapping: Any, key: str) -> bool:
    return isinstance(mapping, dict) and key in mapping and mapping.get(key) is not None


def _missing(prefix: str, mapping: Any, keys: tuple[str, ...]) -> list[str]:
    return [f"{prefix}.{key}" for key in keys if not _is_present(mapping, key)]


def _profile_attestation_problems(payload: dict[str, Any]) -> list[str]:
    """Return missing semantic identity fields for a collected profile result.

    This is deliberately small and conservative. It does not prove a row is
    correct; it proves the row says enough about what it measured before we use
    its speed as launch advice.
    """

    compact = payload.get("compact")
    if not isinstance(compact, dict):
        return ["compact"]

    command = compact.get("command")
    counts = compact.get("counts")
    derived = compact.get("derived")
    timers = compact.get("timers_sec")
    semantic_identity = compact.get("semantic_identity")
    search_backend_proof = compact.get("search_backend_proof")
    problems: list[str] = []
    problems.extend(_missing("compact", compact, PROFILE_ATTESTATION_REQUIRED_COMPACT_KEYS))
    problems.extend(_missing("command", command, PROFILE_ATTESTATION_REQUIRED_COMMAND_KEYS))
    problems.extend(_missing("counts", counts, PROFILE_ATTESTATION_REQUIRED_COUNT_KEYS))
    problems.extend(_missing("derived", derived, PROFILE_ATTESTATION_REQUIRED_DERIVED_KEYS))
    problems.extend(_missing("timers_sec", timers, PROFILE_ATTESTATION_REQUIRED_TIMER_KEYS))
    problems.extend(
        _missing(
            "semantic_identity",
            semantic_identity,
            PROFILE_ATTESTATION_REQUIRED_SEMANTIC_IDENTITY_KEYS,
        )
    )

    if isinstance(command, dict):
        observation_contract = command.get("observation_contract")
        problems.extend(
            _missing(
                "command.observation_contract",
                observation_contract,
                PROFILE_ATTESTATION_REQUIRED_OBSERVATION_CONTRACT_KEYS,
            )
        )
        exploration_bonus = command.get("exploration_bonus")
        if not isinstance(exploration_bonus, dict) or not _is_present(
            exploration_bonus,
            "mode",
        ):
            problems.append("command.exploration_bonus.mode")
        if command.get("env_manager_type") == "curvyzero_batched_profile":
            problems.extend(
                _missing(
                    "timers_sec",
                    timers,
                    BATCHED_PROFILE_REQUIRED_TIMER_KEYS,
                )
            )
        if command.get("skip_lightzero_eval_in_profile") is not True:
            problems.append("command.skip_lightzero_eval_in_profile=true")
        if command.get("collect_search_backend") == DIRECT_CTREE_GPU_LATENT_BACKEND:
            problems.extend(
                _missing(
                    "counts",
                    counts,
                    DIRECT_CTREE_REQUIRED_COUNT_KEYS,
                )
            )

    if compact.get("mode") != "profile":
        problems.append("compact.mode=profile")
    if compact.get("ok") is not True:
        problems.append("compact.ok=true")
    if compact.get("status") != "completed":
        problems.append("compact.status=completed")
    if compact.get("called_train_muzero") is not True:
        problems.append("compact.called_train_muzero=true")
    if isinstance(semantic_identity, dict):
        if semantic_identity.get("schema_id") != "curvyzero_optimizer_profile_semantic_identity/v0":
            problems.append("semantic_identity.schema_id=current")
        if semantic_identity.get("env_steps_collected_source") != counts.get(
            "env_steps_collected_source"
        ):
            problems.append("semantic_identity.env_steps_collected_source=counts")
        if isinstance(derived, dict) and semantic_identity.get(
            "speed_currency"
        ) != derived.get("steps_per_sec_currency"):
            problems.append("semantic_identity.speed_currency=derived")
        if isinstance(command, dict) and semantic_identity.get(
            "collect_search_backend"
        ) != command.get("collect_search_backend"):
            problems.append("semantic_identity.collect_search_backend=command")
        if isinstance(command, dict) and semantic_identity.get(
            "collect_search_ctree_backend"
        ) != command.get("collect_search_ctree_backend"):
            problems.append("semantic_identity.collect_search_ctree_backend=command")
    if isinstance(counts, dict) and isinstance(derived, dict):
        env_steps_source = counts.get("env_steps_collected_source")
        if derived.get("steps_per_sec_source") != env_steps_source:
            problems.append("derived.steps_per_sec_source=counts")
        if counts.get("env_steps_collected_uses_fallback") != derived.get(
            "steps_per_sec_uses_fallback_denominator"
        ):
            problems.append("counts.env_steps_collected_uses_fallback=derived")
        if derived.get("steps_per_sec_currency") != _expected_speed_currency_for_source(
            env_steps_source
        ):
            problems.append("derived.steps_per_sec_currency=source")
    if isinstance(command, dict) and command.get(
        "collect_search_backend"
    ) == DIRECT_CTREE_GPU_LATENT_BACKEND:
        if counts.get("collect_search_backend_fallback_calls") != 0:
            problems.append("counts.collect_search_backend_fallback_calls=0")
        if not isinstance(
            counts.get("collect_search_backend_direct_ctree_gpu_latent_calls"),
            int | float,
        ) or counts.get("collect_search_backend_direct_ctree_gpu_latent_calls", 0) <= 0:
            problems.append(
                "counts.collect_search_backend_direct_ctree_gpu_latent_calls>0"
            )
        if not isinstance(
            counts.get("collect_search_backend_output_rows"),
            int | float,
        ) or counts.get("collect_search_backend_output_rows", 0) <= 0:
            problems.append("counts.collect_search_backend_output_rows>0")
    if (
        isinstance(command, dict)
        and command.get("collect_search_ctree_backend") == FLAT_A3_CTREE_BACKEND
    ):
        observed_ctree = (
            search_backend_proof.get("observed_collect_search_ctree_backends")
            if isinstance(search_backend_proof, dict)
            else None
        )
        if not isinstance(observed_ctree, list) or FLAT_A3_CTREE_BACKEND not in observed_ctree:
            problems.append("search_backend_proof.observed_collect_search_ctree_backends=flat_a3")
        flat_payload_timer_present = (
            search_backend_proof.get("flat_payload_timer_present")
            if isinstance(search_backend_proof, dict)
            else None
        )
        if flat_payload_timer_present is not True:
            problems.append("search_backend_proof.flat_payload_timer_present=true")
    if payload.get("status") != "complete":
        problems.append("payload.status=complete")
    return sorted(set(problems))


def _profile_row(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    return _profile_row_from_payload(path, payload)


def _profile_row_from_payload(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    compact = payload.get("compact") or {}
    command = compact.get("command") or {}
    counts = compact.get("counts") or {}
    derived = compact.get("derived") or {}
    timers = compact.get("timers_sec") or {}
    gpu = compact.get("gpu") or {}
    telemetry = compact.get("telemetry") or {}
    env_timing = telemetry.get("profile_env_timing_sec") or {}
    sampled_sum = env_timing.get("sampled_sum") or {}
    attestation_problems = _profile_attestation_problems(payload)
    gate_a_fields = _gate_a_summary_fields(compact)
    return {
        "row": str(payload.get("row_id") or path.stem.removeprefix("row_").removesuffix("_result")),
        "status": payload.get("status"),
        "attested": not attestation_problems,
        "attestation_problems": attestation_problems,
        **gate_a_fields,
        "collectors": command.get("collector_env_num"),
        "manager": command.get("env_manager_type"),
        "ctree": command.get("collect_search_ctree_backend"),
        "sims": command.get("num_simulations"),
        "death": "nodeath" if command.get("disable_death_for_profile") else "normal",
        "render_mode": command.get("source_state_trail_render_mode"),
        "reward": command.get("reward_variant"),
        "steps": counts.get("env_steps_collected"),
        "steps_source": counts.get("env_steps_collected_source"),
        "speed_currency": derived.get("steps_per_sec_currency"),
        "wall": timers.get("train_muzero_wall"),
        "steps_s": derived.get("steps_per_sec"),
        "collect": timers.get("collector_collect"),
        "manager_step": timers.get("batched_profile_env_manager_step"),
        "surface_env": timers.get("batched_profile_surface_env_step"),
        "surface_stack": timers.get("batched_profile_surface_stack_update"),
        "render_sec": timers.get("batched_profile_renderer_render"),
        "device_render": timers.get("batched_profile_renderer_device_render"),
        "mcts": timers.get("mcts_search"),
        "learner": timers.get("learner_train"),
        "policy_collect": timers.get("policy_forward_collect"),
        "root_batch": derived.get("mcts_root_batch_mean"),
        "ready_before": derived.get("batched_profile_ready_obs_before_step_mean"),
        "ready_after": derived.get("batched_profile_ready_obs_after_step_mean"),
        "timesteps": derived.get("batched_profile_timestep_count_mean"),
        "omitted_rows": derived.get("batched_profile_complete_row_omission_count_mean"),
        "partial_render": derived.get("batched_profile_partial_render_request_mean"),
        "render_outputs": derived.get("batched_profile_render_output_count_mean"),
        "obs_sum": sampled_sum.get("observation_sec"),
        "opp_sum": sampled_sum.get("opponent_action_sec"),
        "vec_sum": sampled_sum.get("vector_step_sec"),
        "gpu_max": gpu.get("max_util_percent"),
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        ("row", "row"),
        ("attested", "attested"),
        ("promotion_status", "promotion"),
        ("speed_currency", "currency"),
        ("stock_path_replaced", "replaces"),
        ("collectors", "C"),
        ("manager", "manager"),
        ("ctree", "ctree"),
        ("sims", "sims"),
        ("death", "death"),
        ("render_mode", "render mode"),
        ("reward", "reward"),
        ("steps", "steps"),
        ("steps_source", "step source"),
        ("wall", "wall"),
        ("steps_s", "steps/s"),
        ("collect", "collect"),
        ("manager_step", "manager"),
        ("surface_env", "env"),
        ("surface_stack", "stack"),
        ("render_sec", "render"),
        ("device_render", "dev render"),
        ("mcts", "MCTS"),
        ("policy_collect", "policy"),
        ("root_batch", "root mean"),
        ("ready_before", "ready before"),
        ("ready_after", "ready after"),
        ("timesteps", "timesteps"),
        ("omitted_rows", "omit rows"),
        ("partial_render", "partial"),
        ("render_outputs", "render out"),
        ("learner", "learner"),
        ("obs_sum", "obs"),
        ("opp_sum", "opp"),
        ("gpu_max", "GPU max"),
    ]
    header = "| " + " | ".join(label for _key, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in sorted(rows, key=lambda item: str(item["row"])):
        cells = []
        for key, _label in columns:
            value = _round(row.get(key))
            cells.append("" if value is None else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _gate_a_summary_fields(compact: dict[str, Any]) -> dict[str, str]:
    command = compact.get("command") or {}
    counts = compact.get("counts") or {}
    derived = compact.get("derived") or {}
    if compact.get("called_train_muzero") is not True or compact.get("mode") != "profile":
        return {
            "speed_currency": "not_gate_a",
            "promotion_status": "not_gate_a",
            "trainer_insertion_point": "none",
            "stock_path_replaced": "none",
        }
    replaced: list[str] = []
    if command.get("collect_search_backend") != "stock":
        replaced.append("collect_search_backend")
    if command.get("collect_search_ctree_backend") != "lightzero":
        replaced.append("collect_search_ctree_backend")
    stock_path_replaced = ",".join(replaced) if replaced else "none"
    if counts.get("env_steps_collected_source") != COLLECTOR_ENVSTEP_SOURCE:
        return {
            "speed_currency": str(
                derived.get("steps_per_sec_currency")
                or FALLBACK_STOCK_PROFILE_SPEED_CURRENCY
            ),
            "promotion_status": "gate_a_ineligible_fallback_steps",
            "trainer_insertion_point": str(compact.get("trainer_entrypoint") or ""),
            "stock_path_replaced": stock_path_replaced,
        }
    promotion_status = (
        "gate_a_stock_baseline"
        if stock_path_replaced == "none"
        else "gate_a_candidate_profile_only"
    )
    return {
        "speed_currency": GATE_A_SPEED_CURRENCY,
        "promotion_status": promotion_status,
        "trainer_insertion_point": str(compact.get("trainer_entrypoint") or ""),
        "stock_path_replaced": stock_path_replaced,
    }


def _gate_a_comparison_problems(payloads: list[dict[str, Any]]) -> list[str]:
    if len(payloads) < 2:
        return ["gate_a requires at least two rows"]
    problems: list[str] = []
    baseline_count = 0
    candidate_count = 0
    reference_values: dict[str, Any] = {}
    reference_runtime_values: dict[str, Any] = {}
    reference_gpu_values: dict[str, Any] = {}
    for index, payload in enumerate(payloads):
        row_label = str(payload.get("row_id", index))
        attestation = _profile_attestation_problems(payload)
        if attestation:
            problems.append(f"row {row_label} attestation failed: {', '.join(attestation)}")
            continue
        compact = payload.get("compact") or {}
        command = compact.get("command") or {}
        counts = compact.get("counts") or {}
        gpu = compact.get("gpu") or {}
        if command.get("collect_search_backend") == "stock" and command.get(
            "collect_search_ctree_backend"
        ) == "lightzero":
            baseline_count += 1
        else:
            candidate_count += 1
        for key in GATE_A_REQUIRED_COUNT_KEYS:
            value = counts.get(key)
            if not isinstance(value, int | float) or value <= 0:
                problems.append(f"row {row_label} counts.{key}>0")
        if counts.get("evaluator_eval_calls") not in (None, 0):
            problems.append(f"row {row_label} counts.evaluator_eval_calls=0")
        if counts.get("env_steps_collected_source") != "collector_envstep_delta":
            problems.append(
                f"row {row_label} counts.env_steps_collected_source=collector_envstep_delta"
            )
        if command.get("lightzero_eval_freq") not in (None, 0):
            problems.append(f"row {row_label} command.lightzero_eval_freq=0")
        for key in GATE_A_DENOMINATOR_COMMAND_KEYS:
            if key in GATE_A_OPTIONAL_DENOMINATOR_COMMAND_KEYS and key not in command:
                continue
            if key not in command or command.get(key) is None:
                problems.append(f"row {row_label} command.{key}")
                continue
            current = _stable_gate_value(command.get(key))
            if key not in reference_values:
                reference_values[key] = current
            elif reference_values[key] != current:
                problems.append(
                    f"row {row_label} command.{key} mismatch: "
                    f"{current!r} != {reference_values[key]!r}"
                )
        for key in GATE_A_RUNTIME_KEYS:
            if key not in compact or compact.get(key) is None:
                problems.append(f"row {row_label} compact.{key}")
                continue
            current = _stable_gate_value(compact.get(key))
            if key not in reference_runtime_values:
                reference_runtime_values[key] = current
            elif reference_runtime_values[key] != current:
                problems.append(
                    f"row {row_label} compact.{key} mismatch: "
                    f"{current!r} != {reference_runtime_values[key]!r}"
                )
        for key in GATE_A_GPU_KEYS:
            if key not in gpu or gpu.get(key) is None:
                continue
            current = _stable_gate_value(gpu.get(key))
            if key not in reference_gpu_values:
                reference_gpu_values[key] = current
            elif reference_gpu_values[key] != current:
                problems.append(
                    f"row {row_label} gpu.{key} mismatch: "
                    f"{current!r} != {reference_gpu_values[key]!r}"
                )
    if baseline_count <= 0:
        problems.append("gate_a requires at least one stock/lightzero baseline row")
    if candidate_count <= 0:
        problems.append("gate_a requires at least one candidate row")
    return sorted(set(problems))


def _stable_gate_value(value: Any) -> Any:
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument(
        "--require-attestation",
        action="store_true",
        help="fail if any profile row is missing required semantic identity fields",
    )
    parser.add_argument(
        "--gate-a-compare",
        action="store_true",
        help="fail unless all rows form a matched stock-vs-candidate Gate A comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.results_dir.glob("row_*_result.json"))
    rows = [_profile_row(path) for path in paths]
    if args.gate_a_compare:
        gate_a_problems = _gate_a_comparison_problems([_load_json(path) for path in paths])
        if gate_a_problems:
            raise SystemExit("gate A comparison failed:\n- " + "\n- ".join(gate_a_problems))
    if args.require_attestation:
        problems = [
            f"row {row['row']}: {', '.join(row['attestation_problems'])}"
            for row in rows
            if row.get("attestation_problems")
        ]
        if problems:
            raise SystemExit("profile attestation failed:\n- " + "\n- ".join(problems))
    if args.format == "json":
        text = json.dumps(rows, indent=2, sort_keys=True) + "\n"
    else:
        text = _markdown_table(rows) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
