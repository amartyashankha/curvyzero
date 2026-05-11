"""Benchmark a first local vector actor-loop bridge.

This is an end-to-end timing scout for the current fixture-seeded vector row
path. It connects batched array steps, debug obs/reward packing, a synthetic
NumPy policy/search stand-in, and fixed replay chunk staging.

It is deliberately not a final training benchmark: observations and rewards are
the debug packer schema, actions come from a synthetic local stand-in, replay is
only in-memory chunk staging, and source fidelity is preflighted only for the
current supported fixture transition slice.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for root in (SCRIPT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import benchmark_vector_batch_rows as batch_rows  # noqa: E402
import benchmark_vector_obs_reward_packing as debug_pack  # noqa: E402
import compare_vector_arrays_to_fidelity as vector_compare  # noqa: E402
import seed_vector_state_from_fixtures as seed_bridge  # noqa: E402
from curvyzero.training import debug_actor_loop_replay as replay_io  # noqa: E402
from curvyzero.training import replay_chunk_v0 as replay_v0  # noqa: E402
from curvyzero.training.policy_row_mapping import (  # noqa: E402
    POLICY_ROW_MAPPING_SCHEMA,
    PolicyRowMapping,
    build_policy_row_mapping,
    policy_rows_to_joint_action,
)


SCHEMA_VERSION = "curvyzero_vector_actor_loop_bridge/v1"
BENCHMARK_ID = "fixture_seeded_numpy_vector_actor_loop_bridge"
SAMPLE_SCHEMA_VERSION = "curvyzero_vector_actor_loop_bridge_sample/v1"
SAMPLE_CONTRACT_METADATA_SCHEMA_VERSION = (
    "curvyzero_vector_actor_loop_bridge_sample_contract_metadata/v1"
)
ACTION_SPACE_ID = "curvyzero_source_move_action_space/v0"
ENV_IMPL_ID = "fixture_seeded_numpy_vector_actor_loop_bridge"
DEFAULT_PATHS = batch_rows.DEFAULT_PATHS
DEFAULT_BATCH_SIZES = (32, 128)
ACTION_COUNT = debug_pack.ACTION_COUNT
ACTION_ID_TO_SOURCE_MOVE = np.asarray([-1, 0, 1], dtype=np.int8)
DEFAULT_ACTION_ID = 1

TIMER_DEFINITIONS = {
    "reset_copy": (
        "Reset the mutable batch state from fixture-seeded initial arrays before each "
        "timed rollout block."
    ),
    "previous_delta_snapshot": (
        "Copy the minimal previous alive/score/round_score arrays needed by the debug "
        "reward delta packer before continuous synthetic-feedback steps."
    ),
    "env_step": (
        "One in-place fixture-seeded vector array step over B rows. The first "
        "step in each rollout block uses fixture source moves; later steps feed "
        "back the synthetic selected actions as source-move ids -1/0/1."
    ),
    "debug_pack": (
        "Pack debug obs[B,P,9], reward[B,P], masks, legal action mask, and ego ids "
        "from previous/current vector states."
    ),
    "policy_root": (
        "Synthetic NumPy root model over flattened ego rows. This is not a learned "
        "checkpoint."
    ),
    "policy_search": (
        "Small synthetic recurrent/search-shaped loop that builds fake visit counts. "
        "This is not MCTS, Mctx, JAX, or GPU work."
    ),
    "action_select": "Mask fake policy/search outputs and choose one action id per ego row.",
    "action_encode": "Map action ids left/straight/right to source moves -1/0/1 for the next step.",
    "replay_chunk_stage": (
        "Copy obs, reward, action ids, fake action weights, fake root values, done, "
        "and ego masks into a fixed in-memory replay chunk ring."
    ),
    "autoreset": (
        "Internal debug actor-bridge row reset after the final transition has "
        "already been staged into the in-memory replay chunk."
    ),
    "loop_overhead": "Timed loop wall time not covered by the buckets above.",
}
FIXED_SHAPE_COST_SCHEMA_VERSION = "curvyzero_vector_actor_loop_fixed_shape_costs/v1"
TRAINING_RATE_SCHEMA_VERSION = "curvyzero_vector_actor_loop_training_rates/v1"
LATENCY_SCHEMA_VERSION = "curvyzero_vector_actor_loop_latency/v1"
FIXED_SHAPE_BUCKET_KEYS = (
    ("reset_copy", "reset_copy_sec"),
    ("previous_delta_snapshot", "previous_delta_snapshot_sec"),
    ("env_step", "env_step_sec"),
    ("debug_pack", "debug_pack_sec"),
    ("policy_root", "policy_root_sec"),
    ("policy_search", "policy_search_sec"),
    ("action_select", "action_select_sec"),
    ("action_encode", "action_encode_sec"),
    ("replay_chunk_stage", "replay_chunk_stage_sec"),
    ("autoreset", "autoreset_sec"),
    ("loop_overhead", "loop_overhead_sec"),
)

KNOWN_FAKE_OR_INCOMPLETE = [
    "timed loop is fixture-reset rollout blocks with debug-only internal autoreset after replay staging, not a production reset/autoreset system",
    "debug obs/reward packer is used as-is; it is not the final training observation or reward schema",
    "synthetic NumPy policy/search stand-in is local shape work, not a learned model and not real MCTS",
    "timed replay chunk staging is an in-memory fixed ring; sample-only can write one local debug .npz chunk, not a production replay stream",
    "source/common-trace preflight covers the supported fixture source moves, not arbitrary policy moves",
    "no-event comparison skips debug event arrays and lets the debug reward packer use its alive-transition fallback",
    "P=1, P=2, and P=3 fixture groups are timed separately instead of one padded mixed-player production batch",
    "the env step is CPU/NumPy only and includes optional lightweight phase timing instrumentation",
    "internal actor-bridge autoreset is debug-only and happens after replay staging; it is not a public reset_many/step_many contract",
    "no horizon truncation, learner handoff, device transfer, JAX compile, or GPU timing is measured",
]


@dataclass
class PolicyState:
    representation_w: np.ndarray
    representation_b: np.ndarray
    dynamics_w: np.ndarray
    dynamics_b: np.ndarray
    action_embed: np.ndarray
    policy_w: np.ndarray
    policy_b: np.ndarray
    value_w: np.ndarray
    reward_w: np.ndarray
    row_index: np.ndarray
    checksum: float = 0.0


@dataclass
class ReplayChunk:
    obs: np.ndarray
    reward: np.ndarray
    action: np.ndarray
    action_weights: np.ndarray
    root_value: np.ndarray
    done: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    ego_mask: np.ndarray
    cursor: int = 0
    chunks_completed: int = 0
    checksum: float = 0.0


def benchmark_inputs(
    paths: Sequence[str | Path],
    *,
    body_capacity: int = 4,
    step_index: int = 0,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    event_modes: Sequence[str] = batch_rows.DEFAULT_EVENT_MODES,
    repeat: int = 1_000,
    warmup: int = 100,
    rollout_steps: int = 1,
    hidden_dim: int = 32,
    simulations: int = 4,
    chunk_steps: int = 32,
    seed: int = 0,
    require_verified: bool = True,
) -> dict[str, Any]:
    """Benchmark the current vector bridge from env rows to staged replay chunks."""

    if body_capacity < 0:
        raise vector_compare.VectorCompareError("body_capacity must be zero or greater")
    if step_index < 0:
        raise vector_compare.VectorCompareError("step_index must be zero or greater")
    if repeat <= 0:
        raise vector_compare.VectorCompareError("repeat must be greater than zero")
    if warmup < 0:
        raise vector_compare.VectorCompareError("warmup must be zero or greater")
    if rollout_steps <= 0:
        raise vector_compare.VectorCompareError("rollout_steps must be greater than zero")
    if hidden_dim <= 0:
        raise vector_compare.VectorCompareError("hidden_dim must be greater than zero")
    if simulations <= 0:
        raise vector_compare.VectorCompareError("simulations must be greater than zero")
    if chunk_steps <= 0:
        raise vector_compare.VectorCompareError("chunk_steps must be greater than zero")
    batch_sizes = tuple(int(size) for size in batch_sizes)
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise vector_compare.VectorCompareError("batch_sizes must contain positive integers")
    event_modes = batch_rows._normalize_event_modes(event_modes)

    wall_started = time.perf_counter()
    timers = {
        "seed_inputs_sec": 0.0,
        "source_preflight_sec": 0.0,
        "array_prepare_sec": 0.0,
        "batch_state_preflight_sec": 0.0,
    }

    started = time.perf_counter()
    seeded = seed_bridge.seed_inputs(paths, body_capacity=body_capacity)
    timers["seed_inputs_sec"] = time.perf_counter() - started

    fixture_results: list[dict[str, Any]] = []
    templates: list[dict[str, Any]] = []
    for fixture in seeded["fixtures"]:
        scenario_id = _string(fixture.get("scenario_id"), "fixture.scenario_id")

        started = time.perf_counter()
        preflight = vector_compare.compare_fixture_seed(
            fixture,
            step_index=step_index,
            require_verified=require_verified,
        )
        timers["source_preflight_sec"] += time.perf_counter() - started

        fixture_result = {
            "scenario_id": scenario_id,
            "path": fixture.get("path"),
            "status": preflight["status"],
            "match": preflight["match"],
            "compared_fields": len(preflight["compared_fields"]),
            "skipped_fields": len(preflight["skipped_fields"]),
            "event_rows_compared": f"$.steps[{step_index}].events"
            not in preflight["skipped_fields"],
            "covered_mechanics": preflight["covered_mechanics"],
            "unsupported_mechanics": preflight["unsupported_mechanics"],
        }
        if preflight["status"] == "fail":
            fixture_result["mismatches"] = preflight["mismatches"]
        fixture_results.append(fixture_result)
        if preflight["status"] != "pass":
            continue

        started = time.perf_counter()
        initial_state = batch_rows._array_state_from_seed(fixture)
        prepared_step = batch_rows._prepare_fixture_array_step(
            fixture,
            step_index=step_index,
        )
        timers["array_prepare_sec"] += time.perf_counter() - started

        profile = _mapping(fixture.get("profile"), "fixture.profile")
        templates.append(
            {
                "fixture": fixture,
                "scenario_id": scenario_id,
                "group_key": batch_rows._group_key(profile, initial_state),
                "player_count": _int(profile.get("P"), "fixture.profile.P"),
                "body_capacity": _int(profile.get("K"), "fixture.profile.K"),
                "initial_state": initial_state,
                "prepared_step": prepared_step,
            }
        )

    passed = sum(result["status"] == "pass" for result in fixture_results)
    failed = sum(result["status"] == "fail" for result in fixture_results)
    unsupported = sum(result["status"] == "unsupported" for result in fixture_results)

    groups_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for template in templates:
        groups_by_key[template["group_key"]].append(template)

    groups = []
    can_time = failed == 0 and bool(templates)
    for group_index, (group_key, group_templates) in enumerate(sorted(groups_by_key.items())):
        started = time.perf_counter()
        preflight = batch_rows._batch_state_preflight(group_templates)
        timers["batch_state_preflight_sec"] += time.perf_counter() - started
        no_event_preflight = None
        if batch_rows.EVENT_MODE_NONE in event_modes:
            started = time.perf_counter()
            no_event_preflight = batch_rows._no_event_state_preflight(group_templates)
            timers["batch_state_preflight_sec"] += time.perf_counter() - started

        batch_results = []
        can_time_group = can_time and preflight["match"] and (
            no_event_preflight is None or no_event_preflight["match"]
        )
        if can_time_group:
            for batch_index, batch_size in enumerate(batch_sizes):
                for event_mode in event_modes:
                    batch_results.append(
                        _benchmark_group_actor_loop(
                            group_templates,
                            batch_size=batch_size,
                            event_mode=event_mode,
                            repeat=repeat,
                            warmup=warmup,
                            rollout_steps=rollout_steps,
                            hidden_dim=hidden_dim,
                            simulations=simulations,
                            chunk_steps=chunk_steps,
                            seed=seed + 10_000 * group_index + 100 * batch_index,
                        )
                    )

        groups.append(
            {
                "group_id": group_key,
                "player_count": group_templates[0]["player_count"],
                "body_capacity": group_templates[0]["body_capacity"],
                "supported_fixture_count": len(group_templates),
                "fixture_ids": [template["scenario_id"] for template in group_templates],
                "preflight": preflight,
                "no_event_preflight": no_event_preflight,
                "batches": batch_results,
                "event_mode_comparisons": _actor_event_mode_comparisons(batch_results),
            }
        )

    batch_preflight_failed = any(
        not group["preflight"]["match"]
        or (
            group["no_event_preflight"] is not None
            and not group["no_event_preflight"]["match"]
        )
        for group in groups
    )
    wall_elapsed_sec = time.perf_counter() - wall_started
    return {
        "schema": SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "source_fidelity_claim": (
            "source/common-trace state and fixed event rows are compared once per "
            "supported fixture before timing; B>1 batch preflight compares stacked "
            "fixture source moves against scalar comparator output. Timed bridge "
            "iterations may feed synthetic selected moves back into the same vector "
            "step path after the first step of each rollout block and are not "
            "source-compared."
        ),
        "trust_level": (
            "First local actor-loop bridge timing only. It connects the current "
            "fixture-seeded NumPy batch step, debug obs/reward packer, synthetic "
            "policy/search stand-in, and in-memory replay chunk staging. It is not "
            "final training throughput."
        ),
        "bridge_contract": {
            "loop_shape": (
                "reset fixture batch once per rollout block -> one or more vector "
                "steps -> debug pack -> synthetic policy/search -> action encode for "
                "the next step -> replay chunk stage"
            ),
            "rollout_steps": rollout_steps,
            "obs_schema": debug_pack.DEBUG_OBS_SCHEMA,
            "reward_schema": debug_pack.DEBUG_REWARD_SCHEMA,
            "obs_shape": "[B,P,9]",
            "reward_shape": "[B,P]",
            "action_count": ACTION_COUNT,
            "action_id_to_source_move": ACTION_ID_TO_SOURCE_MOVE.tolist(),
            "policy_row_mapping_schema": POLICY_ROW_MAPPING_SCHEMA,
            "policy_row_mapping": (
                "debug obs/legal/live masks are compacted to active ego policy rows, "
                "then selected action ids are mapped back to joint_action[B,P]"
            ),
            "replay_chunk_status": (
                "timed benchmark uses a fixed in-memory array ring; sample-only can "
                "optionally write one local debug .npz chunk, no learner handoff"
            ),
        },
        "config": {
            "paths": [str(path) for path in paths],
            "body_capacity": body_capacity,
            "step_index": step_index,
            "batch_sizes": list(batch_sizes),
            "event_modes": list(event_modes),
            "repeat": repeat,
            "warmup": warmup,
            "rollout_steps": rollout_steps,
            "hidden_dim": hidden_dim,
            "simulations": simulations,
            "chunk_steps": chunk_steps,
            "seed": seed,
            "require_verified": require_verified,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "timer_schema": TIMER_DEFINITIONS,
        "input_count": seeded["input_count"],
        "fixture_count": len(fixture_results),
        "supported_fixture_count": len(templates),
        "summary": {
            "passed": passed,
            "failed": failed,
            "unsupported": unsupported,
            "batch_preflight_failed": batch_preflight_failed,
            "status": (
                "fail"
                if failed or batch_preflight_failed
                else "pass"
                if passed and not unsupported
                else "mixed"
            ),
        },
        "timing_sec": {
            **timers,
            "setup_sec": timers["seed_inputs_sec"] + timers["array_prepare_sec"],
            "preflight_total_sec": timers["source_preflight_sec"]
            + timers["batch_state_preflight_sec"],
            "wall_elapsed_sec": wall_elapsed_sec,
        },
        "fixtures": fixture_results,
        "groups": groups,
        "known_fake_or_incomplete": KNOWN_FAKE_OR_INCOMPLETE,
    }


def build_fixture_seeded_actor_bridge_sample(
    paths: Sequence[str | Path] = DEFAULT_PATHS,
    *,
    body_capacity: int = 4,
    step_index: int = 0,
    batch_size: int = 1,
    player_count: int | None = None,
    group_id: str | None = None,
    event_mode: str = batch_rows.EVENT_MODE_DEBUG,
    rollout_steps: int = 2,
    hidden_dim: int = 32,
    simulations: int = 4,
    seed: int = 0,
    require_verified: bool = True,
    replay_chunk_path: str | Path | None = None,
    replay_v0_chunk_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build one fixed-shape sample from the local vector actor bridge."""

    if body_capacity < 0:
        raise vector_compare.VectorCompareError("body_capacity must be zero or greater")
    if step_index < 0:
        raise vector_compare.VectorCompareError("step_index must be zero or greater")
    if batch_size <= 0:
        raise vector_compare.VectorCompareError("batch_size must be greater than zero")
    if player_count is not None and player_count <= 0:
        raise vector_compare.VectorCompareError("player_count must be greater than zero")
    if group_id is not None and not group_id:
        raise vector_compare.VectorCompareError("group_id must be non-empty when provided")
    if rollout_steps <= 0:
        raise vector_compare.VectorCompareError("rollout_steps must be greater than zero")
    if hidden_dim <= 0:
        raise vector_compare.VectorCompareError("hidden_dim must be greater than zero")
    if simulations <= 0:
        raise vector_compare.VectorCompareError("simulations must be greater than zero")
    event_mode = batch_rows._normalize_event_mode(event_mode)

    wall_started = time.perf_counter()
    timers = {
        "seed_inputs_sec": 0.0,
        "source_preflight_sec": 0.0,
        "array_prepare_sec": 0.0,
        "batch_state_preflight_sec": 0.0,
        "actor_rollout_sec": 0.0,
    }

    started = time.perf_counter()
    seeded = seed_bridge.seed_inputs(paths, body_capacity=body_capacity)
    timers["seed_inputs_sec"] = time.perf_counter() - started

    fixture_results: list[dict[str, Any]] = []
    templates: list[dict[str, Any]] = []
    for fixture in seeded["fixtures"]:
        scenario_id = _string(fixture.get("scenario_id"), "fixture.scenario_id")

        started = time.perf_counter()
        preflight = vector_compare.compare_fixture_seed(
            fixture,
            step_index=step_index,
            require_verified=require_verified,
        )
        timers["source_preflight_sec"] += time.perf_counter() - started

        fixture_result = {
            "scenario_id": scenario_id,
            "path": fixture.get("path"),
            "status": preflight["status"],
            "match": preflight["match"],
            "compared_fields": len(preflight["compared_fields"]),
            "skipped_fields": len(preflight["skipped_fields"]),
            "event_rows_compared": f"$.steps[{step_index}].events"
            not in preflight["skipped_fields"],
            "covered_mechanics": preflight["covered_mechanics"],
            "unsupported_mechanics": preflight["unsupported_mechanics"],
        }
        if preflight["status"] == "fail":
            fixture_result["mismatches"] = preflight["mismatches"]
        fixture_results.append(fixture_result)
        if preflight["status"] != "pass":
            continue

        started = time.perf_counter()
        initial_state = batch_rows._array_state_from_seed(fixture)
        prepared_step = batch_rows._prepare_fixture_array_step(
            fixture,
            step_index=step_index,
        )
        timers["array_prepare_sec"] += time.perf_counter() - started

        profile = _mapping(fixture.get("profile"), "fixture.profile")
        templates.append(
            {
                "fixture": fixture,
                "scenario_id": scenario_id,
                "group_key": batch_rows._group_key(profile, initial_state),
                "player_count": _int(profile.get("P"), "fixture.profile.P"),
                "body_capacity": _int(profile.get("K"), "fixture.profile.K"),
                "initial_state": initial_state,
                "prepared_step": prepared_step,
            }
        )

    passed = sum(result["status"] == "pass" for result in fixture_results)
    failed = sum(result["status"] == "fail" for result in fixture_results)
    unsupported = sum(result["status"] == "unsupported" for result in fixture_results)
    if failed:
        raise vector_compare.VectorCompareError(
            f"cannot build actor bridge sample with {failed} source preflight failure(s)"
        )

    groups_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for template in templates:
        groups_by_key[template["group_key"]].append(template)

    selected_group_key = ""
    selected_templates: list[dict[str, Any]] = []
    selected_preflight: dict[str, Any] | None = None
    selected_no_event_preflight: dict[str, Any] | None = None
    skipped_groups: list[dict[str, Any]] = []
    for candidate_group_key, group_templates in sorted(groups_by_key.items()):
        candidate_player_count = group_templates[0]["player_count"]
        if player_count is not None and candidate_player_count != player_count:
            skipped_groups.append(
                {
                    "group_id": candidate_group_key,
                    "reason": "player_count_mismatch",
                    "player_count": candidate_player_count,
                }
            )
            continue
        if group_id is not None and candidate_group_key != group_id:
            skipped_groups.append(
                {
                    "group_id": candidate_group_key,
                    "reason": "group_id_mismatch",
                    "player_count": candidate_player_count,
                }
            )
            continue

        started = time.perf_counter()
        preflight = batch_rows._batch_state_preflight(group_templates)
        timers["batch_state_preflight_sec"] += time.perf_counter() - started
        if not preflight["match"]:
            skipped_groups.append(
                {
                    "group_id": candidate_group_key,
                    "reason": "batch_state_preflight_failed",
                    "player_count": candidate_player_count,
                    "mismatches": preflight["mismatches"],
                }
            )
            continue

        no_event_preflight = None
        if event_mode == batch_rows.EVENT_MODE_NONE:
            started = time.perf_counter()
            no_event_preflight = batch_rows._no_event_state_preflight(group_templates)
            timers["batch_state_preflight_sec"] += time.perf_counter() - started
            if not no_event_preflight["match"]:
                no_event_mismatches = [
                    *no_event_preflight["state_mismatches"],
                    *no_event_preflight["counter_mismatches"],
                ]
                skipped_groups.append(
                    {
                        "group_id": candidate_group_key,
                        "reason": "no_event_state_preflight_failed",
                        "player_count": candidate_player_count,
                        "mismatches": no_event_mismatches,
                    }
                )
                continue

        selected_group_key = candidate_group_key
        selected_templates = list(group_templates)
        selected_preflight = preflight
        selected_no_event_preflight = no_event_preflight
        break

    if selected_preflight is None:
        available_groups = sorted(groups_by_key)
        raise vector_compare.VectorCompareError(
            "no fixture group matched actor bridge sample request; "
            f"requested player_count={player_count!r}, group_id={group_id!r}, "
            f"available_groups={available_groups!r}, skipped_groups={skipped_groups!r}"
        )

    batch = batch_rows._build_batch(selected_templates, batch_size)
    selected_player_count = _int(
        batch["prepared_batch"]["player_count"],
        "prepared_batch.player_count",
    )
    working_state = vector_compare.copy_array_state(batch["initial_state"])
    prepared_for_loop = {
        "player_count": selected_player_count,
        "step_ms": np.asarray(batch["prepared_batch"]["step_ms"], dtype=np.float64),
        "source_moves": np.asarray(batch["prepared_batch"]["source_moves"], dtype=np.int8).copy(),
        "print_manager_mode": np.asarray(
            batch["prepared_batch"].get("print_manager_mode", []),
            dtype=object,
        ).copy(),
        "timer_advance_ms": np.asarray(
            batch["prepared_batch"].get("timer_advance_ms", []),
            dtype=np.float64,
        ).copy(),
    }
    fixture_source_moves = prepared_for_loop["source_moves"].copy()
    pack_state = _debug_pack_state_for_event_mode(working_state, event_mode)
    policy_state = _make_policy_state(
        batch_size=batch_size,
        player_count=selected_player_count,
        obs_dim=len(debug_pack.DEBUG_OBS_FEATURE_NAMES),
        hidden_dim=hidden_dim,
        seed=seed,
    )

    next_source_moves = fixture_source_moves
    step_summaries: list[dict[str, Any]] = []
    total_step_counters: dict[str, int] = defaultdict(int)
    final_surfaces: Mapping[str, Any] | None = None
    final_action_grid: np.ndarray | None = None
    final_action_weights: np.ndarray | None = None
    final_searched_value: np.ndarray | None = None
    final_policy_mapping: PolicyRowMapping | None = None
    sample_replay = _make_replay_chunk(
        chunk_steps=rollout_steps,
        batch_size=batch_size,
        player_count=selected_player_count,
        obs_dim=len(debug_pack.DEBUG_OBS_FEATURE_NAMES),
    )

    started = time.perf_counter()
    for rollout_step_index in range(rollout_steps):
        if rollout_step_index == 0:
            previous_delta_state = batch["initial_state"]
            source_kind = "fixture_source_moves"
        else:
            previous_delta_state = _snapshot_previous_delta_state(working_state)
            source_kind = "synthetic_feedback_moves"

        prepared_for_loop["source_moves"] = next_source_moves
        step_counters = batch_rows.step_batched_arrays(
            working_state,
            prepared_for_loop,
            event_mode=event_mode,
        )
        for name, value in step_counters.items():
            total_step_counters[name] += int(value)

        surfaces = debug_pack.pack_debug_obs_reward(
            previous_delta_state,
            pack_state,
            player_count=selected_player_count,
        )
        obs = np.asarray(surfaces["obs"], dtype=np.float32)
        legal_action_mask = np.asarray(surfaces["legal_action_mask"], dtype=bool)
        ego_mask = np.asarray(surfaces["ego_mask"], dtype=bool)
        policy_mapping = build_policy_row_mapping(obs, ego_mask, legal_action_mask)
        _validate_policy_mapping(policy_mapping)
        hidden, logits, root_value = _policy_root(policy_mapping.observations, policy_state)
        visits, value_accum = _policy_search_standin(
            hidden,
            logits,
            root_value,
            policy_state,
            simulations=simulations,
        )
        policy_action_weights, policy_searched_value, action_ids = _select_actions(
            visits,
            value_accum,
            policy_mapping.legal_action_mask,
            simulations=simulations,
        )
        selected_action_grid, action_weights, searched_value = _policy_rows_to_joint_surfaces(
            policy_mapping,
            action_ids,
            policy_action_weights,
            policy_searched_value,
        )
        next_source_moves = ACTION_ID_TO_SOURCE_MOVE[selected_action_grid].astype(
            np.int8,
            copy=False,
        )
        _stage_replay_chunk(
            sample_replay,
            surfaces,
            selected_action_grid,
            action_weights,
            searched_value,
        )

        step_summaries.append(
            {
                "rollout_step_index": rollout_step_index,
                "source_kind": source_kind,
                "step_counters": step_counters,
                "sample": _actor_bridge_surface_sample_summary(
                    surfaces,
                    selected_action_grid,
                    action_weights,
                    searched_value,
                    policy_mapping,
                ),
            }
        )
        final_surfaces = surfaces
        final_action_grid = selected_action_grid
        final_action_weights = action_weights
        final_searched_value = searched_value
        final_policy_mapping = policy_mapping

    timers["actor_rollout_sec"] = time.perf_counter() - started
    timers["wall_elapsed_sec"] = time.perf_counter() - wall_started
    if (
        final_surfaces is None
        or final_action_grid is None
        or final_action_weights is None
        or final_searched_value is None
        or final_policy_mapping is None
    ):
        raise vector_compare.VectorCompareError("actor bridge sample did not run any steps")

    sample_contract_metadata = _sample_contract_metadata(
        selected_templates,
        batch_size=batch_size,
        player_count=selected_player_count,
        obs_dim=len(debug_pack.DEBUG_OBS_FEATURE_NAMES),
        body_capacity=body_capacity,
        step_index=step_index,
        event_mode=event_mode,
    )
    sample_replay_chunk = _sample_replay_chunk_report(
        replay_chunk_path,
        sample_replay,
        sample_contract_metadata,
    )
    sample_replay_v0_chunk = _sample_replay_v0_chunk_report(
        replay_v0_chunk_path,
        sample_replay,
        sample_contract_metadata,
        selected_templates=selected_templates,
        selected_group_key=selected_group_key,
        seed=seed,
    )

    return {
        "surfaces": final_surfaces,
        "source": {
            "schema": SAMPLE_SCHEMA_VERSION,
            "benchmark_id": BENCHMARK_ID,
            "source": "fixture_seeded_cpu_actor_loop_bridge",
            "source_fidelity_claim": (
                "source/common-trace state and fixed event rows are compared once per "
                "supported fixture before building this sample; B>1 batch preflight "
                "compares stacked fixture source moves against scalar comparator output. "
                "Only rollout step 0 uses source fixture moves. Later rollout steps, "
                "when requested, feed synthetic selected actions back into the vector "
                "step path and are not source-compared."
            ),
            "trust_level": (
                "One fixed-shape sample from the current local actor bridge. It uses "
                "real fixture-seeded NumPy vector env steps and the debug packer, but "
                "synthetic policy/search feedback and no production learner/replay stream."
            ),
            "config": {
                "paths": [str(path) for path in paths],
                "body_capacity": body_capacity,
                "step_index": step_index,
                "batch_size": batch_size,
                "player_count": player_count,
                "group_id": group_id,
                "event_mode": event_mode,
                "rollout_steps": rollout_steps,
                "hidden_dim": hidden_dim,
                "simulations": simulations,
                "seed": seed,
                "require_verified": require_verified,
                "replay_chunk_path": str(replay_chunk_path)
                if replay_chunk_path is not None
                else None,
                "replay_v0_chunk_path": str(replay_v0_chunk_path)
                if replay_v0_chunk_path is not None
                else None,
            },
            "summary": {
                "passed": passed,
                "failed": failed,
                "unsupported": unsupported,
                "status": "pass" if passed and not unsupported else "mixed",
            },
            "input_count": seeded["input_count"],
            "fixture_count": len(fixture_results),
            "supported_fixture_count": len(templates),
            "fixtures": fixture_results,
            "selected_group": {
                "group_id": selected_group_key,
                "player_count": selected_player_count,
                "body_capacity": selected_templates[0]["body_capacity"],
                "supported_fixture_count": len(selected_templates),
                "fixture_ids": [
                    str(template["scenario_id"]) for template in selected_templates
                ],
                "row_source_counts": batch["row_source_counts"],
                "preflight": selected_preflight,
                "no_event_preflight": selected_no_event_preflight,
            },
            "skipped_groups": skipped_groups,
            "step_counters": dict(total_step_counters),
            "step_summaries": step_summaries,
            "sample": _actor_bridge_surface_sample_summary(
                final_surfaces,
                final_action_grid,
                final_action_weights,
                final_searched_value,
                final_policy_mapping,
            ),
            "sample_contract_metadata": sample_contract_metadata,
            "sample_replay_chunk": sample_replay_chunk,
            "sample_replay_v0_chunk": sample_replay_v0_chunk,
            "timing_sec": timers,
            "known_fake_or_incomplete": KNOWN_FAKE_OR_INCOMPLETE,
        },
    }


def print_plain(summary: Mapping[str, Any]) -> None:
    counts = _mapping(summary.get("summary"), "summary")
    timing = _mapping(summary.get("timing_sec"), "timing_sec")
    config = _mapping(summary.get("config"), "config")
    print(f"benchmark={summary['benchmark_id']}")
    print(f"trust_level={summary['trust_level']}")
    print(f"source_fidelity_claim={summary['source_fidelity_claim']}")
    print(
        "summary="
        f"passed:{counts['passed']} failed:{counts['failed']} "
        f"unsupported:{counts['unsupported']} "
        f"batch_preflight_failed:{counts['batch_preflight_failed']}"
    )
    print(
        "profile="
        f"repeat:{config['repeat']} warmup:{config['warmup']} hidden:{config['hidden_dim']} "
        f"synthetic_sims:{config['simulations']} chunk_steps:{config['chunk_steps']} "
        f"rollout_steps:{config['rollout_steps']} "
        f"event_modes:{','.join(str(mode) for mode in config['event_modes'])}"
    )
    print(
        "preflight_sec="
        f"source:{timing['source_preflight_sec']:.6f} "
        f"batch_state:{timing['batch_state_preflight_sec']:.6f} "
        f"setup:{timing['setup_sec']:.6f}"
    )
    for group in _list(summary.get("groups"), "groups"):
        group = _mapping(group, "group")
        preflight = _mapping(group["preflight"], "group.preflight")
        print(
            "group="
            f"{group['group_id']} P={group['player_count']} K={group['body_capacity']} "
            f"fixtures={group['supported_fixture_count']} "
            f"batch_state_match={preflight['state_match']} "
            f"batch_event_match={preflight['event_match']}"
        )
        no_event_preflight = group.get("no_event_preflight")
        if no_event_preflight is not None:
            no_event_preflight = _mapping(
                no_event_preflight,
                "group.no_event_preflight",
            )
            print(
                "no_event_preflight="
                f"state_match:{no_event_preflight['state_match']} "
                f"match:{no_event_preflight['match']}"
            )
        if not preflight["match"]:
            print(f"batch_preflight_mismatches={preflight['mismatches']}")
            continue
        for batch in _list(group.get("batches"), "group.batches"):
            batch = _mapping(batch, "batch")
            rates = _mapping(batch["rates"], "batch.rates")
            timers = _mapping(batch["timing_sec"], "batch.timing_sec")
            sample = _mapping(batch["sample"], "batch.sample")
            counts = _mapping(batch["counts"], "batch.counts")
            cost_report = _mapping(
                batch["fixed_shape_cost_report"],
                "batch.fixed_shape_cost_report",
            )
            cost_buckets = _mapping(cost_report["buckets"], "batch.fixed_shape_cost_report.buckets")
            env_step_cost = _mapping(cost_buckets["env_step"], "cost.env_step")
            debug_pack_cost = _mapping(cost_buckets["debug_pack"], "cost.debug_pack")
            policy_cost = _mapping(
                cost_report["aggregates"]["policy_total"],
                "cost.policy_total",
            )
            replay_cost = _mapping(cost_buckets["replay_chunk_stage"], "cost.replay")
            event_cost = _mapping(cost_report["event_cost"], "cost.event_cost")
            training_report = _mapping(
                batch["training_rate_report"],
                "batch.training_rate_report",
            )
            training_rates = _mapping(
                training_report["loop_rates"],
                "batch.training_rate_report.loop_rates",
            )
            amdahl = _mapping(
                training_report["amdahl_breakdown_pct_loop"],
                "batch.training_rate_report.amdahl_breakdown_pct_loop",
            )
            latency = _mapping(batch["latency_sec"], "batch.latency_sec")
            actor_step_latency = _mapping(
                latency["actor_step_total_sec"],
                "batch.latency_sec.actor_step_total_sec",
            )
            env_step_latency = _mapping(
                latency["env_step_sec"],
                "batch.latency_sec.env_step_sec",
            )
            policy_latency = _mapping(
                latency["synthetic_policy_search_total_sec"],
                "batch.latency_sec.synthetic_policy_search_total_sec",
            )
            replay_latency = _mapping(
                latency["replay_chunk_stage_sec"],
                "batch.latency_sec.replay_chunk_stage_sec",
            )
            print(
                "batch="
                f"event_mode:{batch['event_mode']} "
                f"B:{batch['batch_size']} repeats:{batch['repeat']} "
                f"rollout_steps:{batch['rollout_steps']} "
                f"env_rows:{counts['env_rows']} ego_rows:{counts['ego_rows']} "
                f"env_step_calls:{counts['env_step_calls']} "
                f"elapsed:{timers['loop_elapsed_sec']:.6f} "
                f"env_rows_per_sec:{rates['env_rows_per_sec_total_loop']:.1f} "
                f"ego_rows_per_sec:{rates['ego_rows_per_sec_total_loop']:.1f} "
                f"reset_sec:{timers['reset_copy_sec']:.6f} "
                f"prev_snapshot_sec:{timers['previous_delta_snapshot_sec']:.6f} "
                f"env_step_sec:{timers['env_step_sec']:.6f} "
                f"debug_pack_sec:{timers['debug_pack_sec']:.6f} "
                f"policy_sec:{timers['policy_total_sec']:.6f} "
                f"action_encode_sec:{timers['action_encode_sec']:.6f} "
                f"replay_sec:{timers['replay_chunk_stage_sec']:.6f} "
                f"overhead_sec:{timers['loop_overhead_sec']:.6f} "
                f"top_bucket:{batch['top_bucket']} "
                f"top_env_phase:{batch['top_env_phase']} "
                f"active_ego:{counts['active_ego_rows']} "
                f"chunks:{counts['chunks_completed']} "
                f"chunk_bytes:{sample['bytes_per_chunk']} "
                f"checksum:{sample['checksum']:.6f}"
            )
            print(
                "training_rate="
                f"event_mode:{batch['event_mode']} "
                f"B:{batch['batch_size']} "
                f"staged_transition_ego_rows_per_sec:"
                f"{training_rates['staged_transition_ego_rows_per_sec_total_loop']:.1f} "
                f"staged_transition_ego_rows_per_min:"
                f"{training_rates['staged_transition_ego_rows_per_min_total_loop']:.1f} "
                f"final_transition_env_rows_per_min:"
                f"{training_rates['final_transition_env_rows_per_min_total_loop']:.1f} "
                f"completed_game_rows_per_min_proxy:"
                f"{training_rates['completed_game_rows_per_min_total_loop']:.1f} "
                "completion_proxy:final_transition_env_rows_before_debug_autoreset"
            )
            print(
                "latency="
                f"event_mode:{batch['event_mode']} "
                f"B:{batch['batch_size']} "
                f"actor_step_p50_ms:{actor_step_latency['p50'] * 1000.0:.3f} "
                f"actor_step_p95_ms:{actor_step_latency['p95'] * 1000.0:.3f} "
                f"actor_step_p99_ms:{actor_step_latency['p99'] * 1000.0:.3f} "
                f"env_step_p50_us:{env_step_latency['p50'] * 1_000_000.0:.3f} "
                f"synthetic_policy_p50_us:{policy_latency['p50'] * 1_000_000.0:.3f} "
                f"replay_chunk_stage_p50_us:{replay_latency['p50'] * 1_000_000.0:.3f} "
                "scope:per_timed_vector_actor_step"
            )
            print(
                "amdahl="
                f"event_mode:{batch['event_mode']} "
                f"B:{batch['batch_size']} "
                f"env_step_pct_loop:{amdahl['env_step']:.1f} "
                f"debug_pack_pct_loop:{amdahl['debug_pack']:.1f} "
                f"synthetic_policy_pct_loop:{amdahl['synthetic_policy_search_total']:.1f} "
                f"replay_pct_loop:{amdahl['replay_chunk_stage']:.1f} "
                f"non_env_step_pct_loop:{amdahl['non_env_step']:.1f} "
                "scope:this_fixture_reset_synthetic_actor_loop_only"
            )
            print(
                "fixed_shape_cost="
                f"event_mode:{batch['event_mode']} "
                f"B:{batch['batch_size']} P:{group['player_count']} "
                f"env_step_pct_loop:{env_step_cost['pct_loop']:.1f} "
                f"debug_pack_pct_loop:{debug_pack_cost['pct_loop']:.1f} "
                f"policy_pct_loop:{policy_cost['pct_loop']:.1f} "
                f"replay_pct_loop:{replay_cost['pct_loop']:.1f} "
                f"env_step_us_per_env_row:{env_step_cost['us_per_env_row']:.3f} "
                f"debug_pack_us_per_ego_row:{debug_pack_cost['us_per_ego_row']:.3f} "
                f"policy_us_per_ego_row:{policy_cost['us_per_ego_row']:.3f} "
                f"replay_us_per_ego_row:{replay_cost['us_per_ego_row']:.3f} "
                f"event_emit_pct_env_step:{event_cost['pct_env_step']:.1f} "
                f"events_per_env_row:{event_cost['events_per_env_row']:.3f}"
            )
        for comparison in _list(
            group.get("event_mode_comparisons", []),
            "group.event_mode_comparisons",
        ):
            comparison = _mapping(comparison, "event_mode_comparison")
            print(
                "event_compare="
                f"B:{comparison['batch_size']} rollout_steps:{comparison['rollout_steps']} "
                f"env_rows:{comparison['env_rows']} "
                f"debug_env_step_sec:{comparison['debug_event_env_step_sec']:.6f} "
                f"no_event_env_step_sec:{comparison['no_event_env_step_sec']:.6f} "
                f"debug_minus_no_event_env_step_sec:"
                f"{comparison['debug_minus_no_event_env_step_sec']:.6f} "
                f"debug_minus_no_event_env_step_pct:"
                f"{comparison['debug_minus_no_event_env_step_pct']:.1f} "
                f"debug_loop_sec:{comparison['debug_event_loop_sec']:.6f} "
                f"no_event_loop_sec:{comparison['no_event_loop_sec']:.6f} "
                f"debug_env_rows_per_sec:"
                f"{comparison['debug_event_env_rows_per_sec_total_loop']:.1f} "
                f"no_event_env_rows_per_sec:"
                f"{comparison['no_event_env_rows_per_sec_total_loop']:.1f} "
                f"no_event_total_loop_speedup_vs_debug:"
                f"{comparison['no_event_total_loop_speedup_vs_debug']:.3f} "
                f"no_event_env_step_speedup_vs_debug:"
                f"{comparison['no_event_env_step_speedup_vs_debug']:.3f} "
                f"debug_pack_delta_sec:{comparison['debug_minus_no_event_debug_pack_sec']:.6f} "
                f"policy_delta_sec:{comparison['debug_minus_no_event_policy_total_sec']:.6f} "
                f"replay_delta_sec:{comparison['debug_minus_no_event_replay_chunk_stage_sec']:.6f}"
            )
    print("known_fake_or_incomplete=" + "; ".join(summary["known_fake_or_incomplete"]))


def _benchmark_group_actor_loop(
    templates: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    event_mode: str,
    repeat: int,
    warmup: int,
    rollout_steps: int,
    hidden_dim: int,
    simulations: int,
    chunk_steps: int,
    seed: int,
) -> dict[str, Any]:
    event_mode = batch_rows._normalize_event_mode(event_mode)
    batch = batch_rows._build_batch(templates, batch_size)
    player_count = _int(batch["prepared_batch"]["player_count"], "prepared_batch.player_count")
    obs_dim = len(debug_pack.DEBUG_OBS_FEATURE_NAMES)

    if warmup:
        warmup_state = vector_compare.copy_array_state(batch["initial_state"])
        warmup_policy = _make_policy_state(
            batch_size=batch_size,
            player_count=player_count,
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            seed=seed + 1_000_000,
        )
        warmup_replay = _make_replay_chunk(
            chunk_steps=chunk_steps,
            batch_size=batch_size,
            player_count=player_count,
            obs_dim=obs_dim,
        )
        warmup_started = time.perf_counter()
        _run_actor_loop(
            working_state=warmup_state,
            initial_state=batch["initial_state"],
            prepared_batch=batch["prepared_batch"],
            policy_state=warmup_policy,
            replay=warmup_replay,
            player_count=player_count,
            iterations=warmup,
            rollout_steps=rollout_steps,
            simulations=simulations,
            collect_phase_timing=False,
            event_mode=event_mode,
        )
        warmup_sec = time.perf_counter() - warmup_started
    else:
        warmup_sec = 0.0

    working_state = vector_compare.copy_array_state(batch["initial_state"])
    policy_state = _make_policy_state(
        batch_size=batch_size,
        player_count=player_count,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        seed=seed,
    )
    replay = _make_replay_chunk(
        chunk_steps=chunk_steps,
        batch_size=batch_size,
        player_count=player_count,
        obs_dim=obs_dim,
    )
    timers, counters, sample, latency_samples = _run_actor_loop(
        working_state=working_state,
        initial_state=batch["initial_state"],
        prepared_batch=batch["prepared_batch"],
        policy_state=policy_state,
        replay=replay,
        player_count=player_count,
        iterations=repeat,
        rollout_steps=rollout_steps,
        simulations=simulations,
        collect_phase_timing=True,
        event_mode=event_mode,
    )

    loop_elapsed = timers["loop_elapsed_sec"]
    policy_total_sec = timers["policy_root_sec"] + timers["policy_search_sec"] + timers[
        "action_select_sec"
    ]
    timers["policy_total_sec"] = policy_total_sec
    event_overhead_sec = batch_rows._event_phase_sec(timers["env_phase_timing_sec"])
    timers["env_event_overhead_sec"] = event_overhead_sec

    env_rows = counters["env_rows"]
    ego_rows = counters["ego_rows"]
    final_transition_rows = counters["final_transition_rows_before_autoreset"]
    staged_transition_rows = counters["replay_rows"]
    training_rate_report = _training_rate_report(timers=timers, counters=counters)
    return {
        "batch_size": batch_size,
        "event_mode": event_mode,
        "events_enabled": batch_rows._events_enabled(event_mode),
        "repeat": repeat,
        "warmup": warmup,
        "rollout_steps": rollout_steps,
        "hidden_dim": hidden_dim,
        "simulations": simulations,
        "chunk_steps": chunk_steps,
        "row_source_counts": batch["row_source_counts"],
        "timing_sec": timers,
        "env_phase_timing_sec": timers["env_phase_timing_sec"],
        "rates": {
            "warmup_env_rows": batch_size * warmup * rollout_steps,
            "timed_env_rows": env_rows,
            "timed_ego_rows": ego_rows,
            "env_rows_per_sec_total_loop": _rate(env_rows, loop_elapsed),
            "ego_rows_per_sec_total_loop": _rate(ego_rows, loop_elapsed),
            "active_ego_rows_per_sec_total_loop": _rate(counters["active_ego_rows"], loop_elapsed),
            "staged_transition_ego_rows_per_sec_total_loop": _rate(
                staged_transition_rows,
                loop_elapsed,
            ),
            "staged_transition_ego_rows_per_min_total_loop": _rate(
                staged_transition_rows * 60.0,
                loop_elapsed,
            ),
            "final_transition_env_rows_per_sec_total_loop": _rate(
                final_transition_rows,
                loop_elapsed,
            ),
            "final_transition_env_rows_per_min_total_loop": _rate(
                final_transition_rows * 60.0,
                loop_elapsed,
            ),
            "completed_game_rows_per_sec_total_loop": _rate(
                final_transition_rows,
                loop_elapsed,
            ),
            "completed_game_rows_per_min_total_loop": _rate(
                final_transition_rows * 60.0,
                loop_elapsed,
            ),
            "env_rows_per_sec_env_step_bucket": _rate(env_rows, timers["env_step_sec"]),
            "ego_rows_per_sec_debug_pack_bucket": _rate(ego_rows, timers["debug_pack_sec"]),
            "ego_rows_per_sec_policy_bucket": _rate(ego_rows, policy_total_sec),
            "ego_rows_per_sec_replay_bucket": _rate(ego_rows, timers["replay_chunk_stage_sec"]),
            "env_event_overhead_step_pct": _rate(
                event_overhead_sec * 100.0,
                timers["env_step_sec"],
            ),
        },
        "counts": counters,
        "sample": sample,
        "fixed_shape_cost_report": _fixed_shape_cost_report(
            batch_size=batch_size,
            player_count=player_count,
            event_mode=event_mode,
            timers=timers,
            counters=counters,
            event_overhead_sec=event_overhead_sec,
        ),
        "training_rate_report": training_rate_report,
        "latency_sec": _latency_report(latency_samples),
        "warmup_sec": warmup_sec,
        "top_bucket": _top_bucket_label(timers),
        "top_env_phase": _top_phase_label(timers["env_phase_timing_sec"]),
    }


def _run_actor_loop(
    *,
    working_state: dict[str, np.ndarray],
    initial_state: Mapping[str, np.ndarray],
    prepared_batch: Mapping[str, Any],
    policy_state: PolicyState,
    replay: ReplayChunk,
    player_count: int,
    iterations: int,
    rollout_steps: int,
    simulations: int,
    collect_phase_timing: bool,
    event_mode: str,
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any], dict[str, list[float]]]:
    batch_size = int(working_state["tick"].shape[0])
    prepared_for_loop = {
        "player_count": player_count,
        "step_ms": np.asarray(prepared_batch["step_ms"], dtype=np.float64),
        "source_moves": np.asarray(prepared_batch["source_moves"], dtype=np.int8).copy(),
        "print_manager_mode": np.asarray(
            prepared_batch.get("print_manager_mode", []),
            dtype=object,
        ).copy(),
        "timer_advance_ms": np.asarray(
            prepared_batch.get("timer_advance_ms", []),
            dtype=np.float64,
        ).copy(),
    }
    fixture_source_moves = prepared_for_loop["source_moves"].copy()
    pack_state = _debug_pack_state_for_event_mode(working_state, event_mode)
    timers: dict[str, Any] = {
        "reset_copy_sec": 0.0,
        "previous_delta_snapshot_sec": 0.0,
        "env_step_sec": 0.0,
        "debug_pack_sec": 0.0,
        "policy_root_sec": 0.0,
        "policy_search_sec": 0.0,
        "action_select_sec": 0.0,
        "action_encode_sec": 0.0,
        "replay_chunk_stage_sec": 0.0,
        "autoreset_sec": 0.0,
        "loop_overhead_sec": 0.0,
        "loop_elapsed_sec": 0.0,
        "env_phase_timing_sec": batch_rows._empty_phase_timers(),
    }
    counters = {
        "rollout_blocks": 0,
        "state_reset_blocks": 0,
        "state_reset_env_rows": 0,
        "env_step_calls": 0,
        "env_rows": 0,
        "ego_rows": 0,
        "active_ego_rows": 0,
        "policy_row_mapping_calls": 0,
        "policy_rows": 0,
        "active_policy_rows": 0,
        "padded_policy_rows": 0,
        "replay_rows": 0,
        "done_rows": 0,
        "truncated_rows": 0,
        "autoreset_rows": 0,
        "final_transition_rows_before_autoreset": 0,
        "chunks_completed": 0,
        "chunk_stage_calls": 0,
        "synthetic_recurrent_model_calls": 0,
        "synthetic_visit_updates": 0,
        "fixture_source_env_rows": 0,
        "synthetic_feedback_env_rows": 0,
        "env_events_emitted": 0,
        "env_event_overflow_attempts": 0,
    }
    latency_samples: dict[str, list[float]] = {
        "actor_step_total_sec": [],
        "env_step_sec": [],
        "policy_total_sec": [],
        "replay_chunk_stage_sec": [],
    }
    sample: dict[str, Any] = {}

    started_loop = time.perf_counter()
    for block_index in range(iterations):
        started = time.perf_counter()
        vector_compare.reset_array_state(working_state, initial_state)
        timers["reset_copy_sec"] += time.perf_counter() - started
        counters["rollout_blocks"] += 1
        counters["state_reset_blocks"] += 1
        counters["state_reset_env_rows"] += batch_size

        next_source_moves = fixture_source_moves
        for step_index in range(rollout_steps):
            actor_step_started = time.perf_counter()
            if step_index == 0:
                previous_delta_state = initial_state
                counters["fixture_source_env_rows"] += batch_size
            else:
                started = time.perf_counter()
                previous_delta_state = _snapshot_previous_delta_state(working_state)
                timers["previous_delta_snapshot_sec"] += time.perf_counter() - started
                counters["synthetic_feedback_env_rows"] += batch_size

            prepared_for_loop["source_moves"] = next_source_moves
            phase_timers = timers["env_phase_timing_sec"] if collect_phase_timing else None
            started = time.perf_counter()
            step_counters = batch_rows.step_batched_arrays(
                working_state,
                prepared_for_loop,
                phase_timers=phase_timers,
                event_mode=event_mode,
            )
            env_step_elapsed = time.perf_counter() - started
            timers["env_step_sec"] += env_step_elapsed

            started = time.perf_counter()
            surfaces = debug_pack.pack_debug_obs_reward(
                previous_delta_state,
                pack_state,
                player_count=player_count,
            )
            timers["debug_pack_sec"] += time.perf_counter() - started

            obs = np.asarray(surfaces["obs"], dtype=np.float32)
            legal_action_mask = np.asarray(surfaces["legal_action_mask"], dtype=bool)
            ego_mask = np.asarray(surfaces["ego_mask"], dtype=bool)
            policy_mapping = build_policy_row_mapping(obs, ego_mask, legal_action_mask)
            _validate_policy_mapping(policy_mapping)

            started = time.perf_counter()
            hidden, logits, root_value = _policy_root(policy_mapping.observations, policy_state)
            policy_root_elapsed = time.perf_counter() - started
            timers["policy_root_sec"] += policy_root_elapsed

            started = time.perf_counter()
            visits, value_accum = _policy_search_standin(
                hidden,
                logits,
                root_value,
                policy_state,
                simulations=simulations,
            )
            policy_search_elapsed = time.perf_counter() - started
            timers["policy_search_sec"] += policy_search_elapsed

            started = time.perf_counter()
            policy_action_weights, policy_searched_value, action_ids = _select_actions(
                visits,
                value_accum,
                policy_mapping.legal_action_mask,
                simulations=simulations,
            )
            action_select_elapsed = time.perf_counter() - started
            timers["action_select_sec"] += action_select_elapsed

            started = time.perf_counter()
            selected_action_grid, action_weights, searched_value = _policy_rows_to_joint_surfaces(
                policy_mapping,
                action_ids,
                policy_action_weights,
                policy_searched_value,
            )
            next_source_moves = ACTION_ID_TO_SOURCE_MOVE[selected_action_grid].astype(
                np.int8,
                copy=False,
            )
            timers["action_encode_sec"] += time.perf_counter() - started

            started = time.perf_counter()
            replay_rows = _stage_replay_chunk(
                replay,
                surfaces,
                selected_action_grid,
                action_weights,
                searched_value,
            )
            replay_chunk_stage_elapsed = time.perf_counter() - started
            timers["replay_chunk_stage_sec"] += replay_chunk_stage_elapsed

            done = np.asarray(surfaces["done"], dtype=bool)
            truncated = np.asarray(surfaces["truncated"], dtype=bool)
            final_transition_mask = vector_compare.final_transition_mask(done, truncated)
            final_transition_count = int(final_transition_mask.sum())

            counters["env_step_calls"] += 1
            counters["env_rows"] += batch_size
            counters["ego_rows"] += batch_size * player_count
            counters["active_ego_rows"] += int(ego_mask.sum())
            counters["policy_row_mapping_calls"] += 1
            counters["policy_rows"] += policy_mapping.capacity
            counters["active_policy_rows"] += policy_mapping.active_count
            counters["padded_policy_rows"] += policy_mapping.capacity - policy_mapping.active_count
            counters["replay_rows"] += replay_rows
            counters["done_rows"] += int(done.sum())
            counters["truncated_rows"] += int(truncated.sum())
            counters["final_transition_rows_before_autoreset"] += final_transition_count
            counters["chunk_stage_calls"] += 1
            counters["synthetic_recurrent_model_calls"] += (
                policy_mapping.active_count * simulations
            )
            counters["synthetic_visit_updates"] += policy_mapping.active_count * simulations
            counters["env_events_emitted"] += int(step_counters.get("events_emitted", 0))
            counters["env_event_overflow_attempts"] += int(
                step_counters.get("event_overflow_attempts", 0)
            )
            counters["chunks_completed"] = replay.chunks_completed

            if block_index == iterations - 1 and step_index == rollout_steps - 1:
                final_sample_inputs = (
                    surfaces,
                    selected_action_grid,
                    action_weights,
                    searched_value,
                    policy_mapping,
                )
            else:
                final_sample_inputs = None

            if final_transition_count:
                started = time.perf_counter()
                counters["autoreset_rows"] += int(
                    vector_compare.reset_array_rows(
                        working_state,
                        initial_state,
                        final_transition_mask,
                    )
                )
                timers["autoreset_sec"] += time.perf_counter() - started

            if collect_phase_timing:
                latency_samples["actor_step_total_sec"].append(
                    time.perf_counter() - actor_step_started
                )
                latency_samples["env_step_sec"].append(env_step_elapsed)
                latency_samples["policy_total_sec"].append(
                    policy_root_elapsed + policy_search_elapsed + action_select_elapsed
                )
                latency_samples["replay_chunk_stage_sec"].append(
                    replay_chunk_stage_elapsed
                )

            if final_sample_inputs is not None:
                sample = _sample_summary(
                    *final_sample_inputs,
                    replay,
                    policy_state,
                )

    elapsed = time.perf_counter() - started_loop
    measured = sum(
        value
        for key, value in timers.items()
        if key.endswith("_sec") and isinstance(value, float)
    )
    measured -= timers["loop_overhead_sec"]
    measured -= timers["loop_elapsed_sec"]
    timers["loop_elapsed_sec"] = elapsed
    timers["loop_overhead_sec"] = max(0.0, elapsed - measured)
    return timers, counters, sample, latency_samples


def _snapshot_previous_delta_state(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "alive": np.asarray(state["alive"], dtype=bool).copy(),
        "score": np.asarray(state["score"], dtype=np.int32).copy(),
        "round_score": np.asarray(state["round_score"], dtype=np.int32).copy(),
    }


def _debug_pack_state_for_event_mode(
    state: dict[str, np.ndarray],
    event_mode: str,
) -> Mapping[str, np.ndarray]:
    if batch_rows._events_enabled(event_mode):
        return state
    return {name: value for name, value in state.items() if name not in batch_rows.EVENT_ARRAY_NAMES}


def _make_policy_state(
    *,
    batch_size: int,
    player_count: int,
    obs_dim: int,
    hidden_dim: int,
    seed: int,
) -> PolicyState:
    rng = np.random.default_rng(seed)
    ego_rows = batch_size * player_count

    def matrix(rows: int, cols: int, scale: float) -> np.ndarray:
        return rng.normal(0.0, scale, size=(rows, cols)).astype(np.float32)

    return PolicyState(
        representation_w=matrix(obs_dim, hidden_dim, 0.08),
        representation_b=rng.normal(0.0, 0.01, size=hidden_dim).astype(np.float32),
        dynamics_w=matrix(hidden_dim, hidden_dim, 0.025),
        dynamics_b=rng.normal(0.0, 0.01, size=hidden_dim).astype(np.float32),
        action_embed=matrix(ACTION_COUNT, hidden_dim, 0.06),
        policy_w=matrix(hidden_dim, ACTION_COUNT, 0.05),
        policy_b=rng.normal(0.0, 0.01, size=ACTION_COUNT).astype(np.float32),
        value_w=rng.normal(0.0, 0.04, size=hidden_dim).astype(np.float32),
        reward_w=rng.normal(0.0, 0.03, size=hidden_dim).astype(np.float32),
        row_index=np.arange(ego_rows),
    )


def _make_replay_chunk(
    *,
    chunk_steps: int,
    batch_size: int,
    player_count: int,
    obs_dim: int,
) -> ReplayChunk:
    return ReplayChunk(
        obs=np.empty((chunk_steps, batch_size, player_count, obs_dim), dtype=np.float32),
        reward=np.empty((chunk_steps, batch_size, player_count), dtype=np.float32),
        action=np.empty((chunk_steps, batch_size, player_count), dtype=np.int8),
        action_weights=np.empty(
            (chunk_steps, batch_size, player_count, ACTION_COUNT),
            dtype=np.float32,
        ),
        root_value=np.empty((chunk_steps, batch_size, player_count), dtype=np.float32),
        done=np.empty((chunk_steps, batch_size), dtype=bool),
        terminated=np.empty((chunk_steps, batch_size), dtype=bool),
        truncated=np.empty((chunk_steps, batch_size), dtype=bool),
        ego_mask=np.empty((chunk_steps, batch_size, player_count), dtype=bool),
    )


def _policy_root(
    obs: np.ndarray,
    state: PolicyState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = obs.reshape(-1, obs.shape[-1])
    hidden = np.tanh(rows @ state.representation_w + state.representation_b)
    logits = hidden @ state.policy_w + state.policy_b
    root_value = np.tanh(hidden @ state.value_w)
    return hidden, logits, root_value.astype(np.float32, copy=False)


def _policy_search_standin(
    hidden: np.ndarray,
    logits: np.ndarray,
    root_value: np.ndarray,
    state: PolicyState,
    *,
    simulations: int,
) -> tuple[np.ndarray, np.ndarray]:
    visits = np.zeros((hidden.shape[0], ACTION_COUNT), dtype=np.float32)
    value_accum = root_value.astype(np.float32, copy=True)
    current_hidden = hidden
    current_logits = logits
    row_index = np.arange(hidden.shape[0])
    for sim_index in range(simulations):
        tie_break = np.float32((sim_index + 1) * 1.0e-4)
        action = np.argmax(
            current_logits + tie_break * np.arange(ACTION_COUNT, dtype=np.float32),
            axis=1,
        )
        np.add.at(visits, (row_index, action), 1.0)
        current_hidden = np.tanh(
            current_hidden
            + state.action_embed[action]
            + current_hidden @ state.dynamics_w
            + state.dynamics_b
        )
        current_logits = current_hidden @ state.policy_w + state.policy_b
        value = np.tanh(current_hidden @ state.value_w)
        reward = np.float32(0.05) * np.tanh(current_hidden @ state.reward_w)
        value_accum += reward + np.float32(0.99) * value
    state.checksum += float(value_accum[: min(4, value_accum.size)].sum(dtype=np.float64))
    return visits, value_accum


def _validate_policy_mapping(mapping: PolicyRowMapping) -> None:
    if mapping.schema != POLICY_ROW_MAPPING_SCHEMA:
        raise vector_compare.VectorCompareError(
            f"unexpected policy row mapping schema: {mapping.schema!r}"
        )
    if mapping.action_count != ACTION_COUNT:
        raise vector_compare.VectorCompareError(
            f"policy mapping action_count must be {ACTION_COUNT}, got {mapping.action_count}"
        )


def _select_actions(
    visits: np.ndarray,
    value_accum: np.ndarray,
    legal_action_mask: np.ndarray,
    *,
    simulations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_legal = legal_action_mask.reshape(-1, ACTION_COUNT)
    denom = np.float32(max(1, simulations))
    action_weights = visits / denom
    action_weights = np.where(flat_legal, action_weights, 0.0).astype(np.float32, copy=False)
    legal_any = flat_legal.any(axis=1)
    selected = np.argmax(action_weights, axis=1).astype(np.int16)
    selected[~legal_any] = DEFAULT_ACTION_ID
    root_value = (value_accum / np.float32(simulations + 1)).astype(np.float32, copy=False)
    root_value[~legal_any] = 0.0
    return action_weights, root_value, selected


def _policy_rows_to_joint_surfaces(
    mapping: PolicyRowMapping,
    selected_action_ids: np.ndarray,
    action_weights: np.ndarray,
    searched_value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_action_grid = policy_rows_to_joint_action(
        mapping,
        selected_action_ids,
        noop_action_id=DEFAULT_ACTION_ID,
        dtype=np.int8,
    )
    batch_size, player_count = mapping.source_shape
    action_weights_grid = np.zeros((batch_size, player_count, ACTION_COUNT), dtype=np.float32)
    searched_value_grid = np.zeros((batch_size, player_count), dtype=np.float32)

    active_rows = np.asarray(mapping.row_mask, dtype=bool)
    if active_rows.any():
        env_ids = np.asarray(mapping.env_row_id, dtype=np.int64)[active_rows]
        player_ids = np.asarray(mapping.player_id, dtype=np.int64)[active_rows]
        action_weights_grid[env_ids, player_ids] = np.asarray(
            action_weights,
            dtype=np.float32,
        )[active_rows]
        searched_value_grid[env_ids, player_ids] = np.asarray(
            searched_value,
            dtype=np.float32,
        )[active_rows]
    return selected_action_grid, action_weights_grid, searched_value_grid


def _stage_replay_chunk(
    replay: ReplayChunk,
    surfaces: Mapping[str, Any],
    action_ids: np.ndarray,
    action_weights: np.ndarray,
    root_value: np.ndarray,
) -> int:
    slot = replay.cursor
    replay.obs[slot] = np.asarray(surfaces["obs"], dtype=np.float32)
    replay.reward[slot] = np.asarray(surfaces["reward"], dtype=np.float32)
    replay.action[slot] = action_ids.astype(np.int8, copy=False)
    replay.action_weights[slot] = action_weights.astype(np.float32, copy=False)
    replay.root_value[slot] = root_value.astype(np.float32, copy=False)
    terminated = np.asarray(surfaces["done"], dtype=bool)
    truncated = np.asarray(surfaces["truncated"], dtype=bool)
    replay.done[slot] = terminated
    replay.terminated[slot] = terminated
    replay.truncated[slot] = truncated
    replay.ego_mask[slot] = np.asarray(surfaces["ego_mask"], dtype=bool)
    replay.checksum += float(
        replay.obs[slot, :, :, 0].sum(dtype=np.float64)
        + replay.reward[slot].sum(dtype=np.float64)
        + replay.root_value[slot].sum(dtype=np.float64)
        + replay.action[slot].sum(dtype=np.float64)
    )
    replay.cursor += 1
    if replay.cursor == replay.obs.shape[0]:
        replay.cursor = 0
        replay.chunks_completed += 1
    return int(action_ids.size)


def _sample_summary(
    surfaces: Mapping[str, Any],
    action_ids: np.ndarray,
    action_weights: np.ndarray,
    searched_value: np.ndarray,
    policy_mapping: PolicyRowMapping,
    replay: ReplayChunk,
    policy_state: PolicyState,
) -> dict[str, Any]:
    obs = np.asarray(surfaces["obs"])
    reward = np.asarray(surfaces["reward"])
    ego_mask = np.asarray(surfaces["ego_mask"], dtype=bool)
    return {
        "obs_shape": list(obs.shape),
        "reward_shape": list(reward.shape),
        "action_shape": list(action_ids.shape),
        "action_weights_shape": list(action_weights.shape),
        "root_value_shape": list(searched_value.shape),
        "obs_dtype": str(obs.dtype),
        "reward_dtype": str(reward.dtype),
        "action_dtype": str(action_ids.dtype),
        "bytes_per_chunk": _chunk_nbytes(replay),
        "replay_chunk_shapes": _chunk_shapes(replay),
        "replay_chunk_dtypes": _chunk_dtypes(replay),
        "policy_row_mapping_schema": policy_mapping.schema,
        "policy_source_shape": list(policy_mapping.source_shape),
        "policy_rows": policy_mapping.capacity,
        "active_policy_rows": policy_mapping.active_count,
        "padded_policy_rows": policy_mapping.capacity - policy_mapping.active_count,
        "reward_sum": float(reward.sum(dtype=np.float64)),
        "live_ego_count": int(ego_mask.sum()),
        "done_count": int(np.asarray(surfaces["done"], dtype=bool).sum()),
        "truncated_count": int(np.asarray(surfaces["truncated"], dtype=bool).sum()),
        "selected_action_histogram": _histogram(action_ids, ACTION_COUNT),
        "checksum": float(
            obs.sum(dtype=np.float64)
            + reward.sum(dtype=np.float64)
            + action_weights.sum(dtype=np.float64)
            + searched_value.sum(dtype=np.float64)
            + replay.checksum
            + policy_state.checksum
        ),
    }


def _actor_bridge_surface_sample_summary(
    surfaces: Mapping[str, Any],
    action_ids: np.ndarray,
    action_weights: np.ndarray,
    searched_value: np.ndarray,
    policy_mapping: PolicyRowMapping,
) -> dict[str, Any]:
    obs = np.asarray(surfaces["obs"])
    reward = np.asarray(surfaces["reward"])
    done = np.asarray(surfaces["done"])
    truncated = np.asarray(surfaces["truncated"])
    ego_mask = np.asarray(surfaces["ego_mask"], dtype=bool)
    legal_action_mask = np.asarray(surfaces["legal_action_mask"])
    return {
        "obs_shape": list(obs.shape),
        "reward_shape": list(reward.shape),
        "done_shape": list(done.shape),
        "truncated_shape": list(truncated.shape),
        "legal_action_mask_shape": list(legal_action_mask.shape),
        "ego_mask_shape": list(ego_mask.shape),
        "ego_row_id_shape": list(np.asarray(surfaces["ego_row_id"]).shape),
        "ego_env_id_shape": list(np.asarray(surfaces["ego_env_id"]).shape),
        "ego_player_id_shape": list(np.asarray(surfaces["ego_player_id"]).shape),
        "action_shape": list(action_ids.shape),
        "action_weights_shape": list(action_weights.shape),
        "root_value_shape": list(searched_value.shape),
        "policy_row_mapping_schema": policy_mapping.schema,
        "policy_source_shape": list(policy_mapping.source_shape),
        "policy_rows": policy_mapping.capacity,
        "active_policy_rows": policy_mapping.active_count,
        "padded_policy_rows": policy_mapping.capacity - policy_mapping.active_count,
        "obs_dtype": str(obs.dtype),
        "reward_dtype": str(reward.dtype),
        "done_dtype": str(done.dtype),
        "truncated_dtype": str(truncated.dtype),
        "legal_action_mask_dtype": str(legal_action_mask.dtype),
        "ego_mask_dtype": str(ego_mask.dtype),
        "bytes_per_sample": _surface_nbytes(surfaces),
        "reward_sum": float(reward.sum(dtype=np.float64)),
        "live_ego_count": int(ego_mask.sum()),
        "done_count": int(done.sum()),
        "truncated_count": int(truncated.sum()),
        "legal_true_count": int(legal_action_mask.sum()),
        "selected_action_histogram": _histogram(action_ids, ACTION_COUNT),
        "reward_died_source": str(surfaces["reward_components"]["died_source"]),
        "checksum": float(
            obs.sum(dtype=np.float64)
            + reward.sum(dtype=np.float64)
            + action_weights.sum(dtype=np.float64)
            + searched_value.sum(dtype=np.float64)
        ),
    }


def _sample_contract_metadata(
    selected_templates: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    player_count: int,
    obs_dim: int,
    body_capacity: int,
    step_index: int,
    event_mode: str,
) -> dict[str, Any]:
    fixture_ids = [str(template["scenario_id"]) for template in selected_templates]
    ruleset_ids = sorted(
        {
            str(template["fixture"].get("ruleset_id", "unknown"))
            for template in selected_templates
        }
    )
    ruleset_id = ruleset_ids[0] if len(ruleset_ids) == 1 else "mixed"
    rules_hash = _stable_contract_hash(
        {
            "scope": "fixture_sample_selection",
            "ruleset_ids": ruleset_ids,
            "fixture_ids": fixture_ids,
            "body_capacity": body_capacity,
            "step_index": step_index,
            "event_mode": event_mode,
        }
    )
    observation_schema_hash = _stable_contract_hash(
        {
            "schema_id": debug_pack.DEBUG_OBS_SCHEMA,
            "shape": ["B", "P", obs_dim],
            "dtype": "float32",
            "feature_names": list(debug_pack.DEBUG_OBS_FEATURE_NAMES),
        }
    )
    action_space_hash = _stable_contract_hash(
        {
            "schema_id": ACTION_SPACE_ID,
            "action_count": ACTION_COUNT,
            "action_id_to_source_move": ACTION_ID_TO_SOURCE_MOVE.tolist(),
            "default_action_id": DEFAULT_ACTION_ID,
        }
    )
    reward_schema_hash = _stable_contract_hash(
        {
            "schema_id": debug_pack.DEBUG_REWARD_SCHEMA,
            "shape": ["B", "P"],
            "dtype": "float32",
            "components": [
                "score_delta",
                "round_score_delta",
                "died_this_step",
            ],
        }
    )
    return {
        "schema": SAMPLE_CONTRACT_METADATA_SCHEMA_VERSION,
        "status": "sample_metadata_only",
        "metadata_scope": (
            "This is deterministic metadata for the actor bridge sample output. "
            "It is a local compatibility guard, not a production replay writer contract."
        ),
        "chunk_level_metadata": {
            "replay_schema_id": replay_io.REPLAY_SCHEMA_ID,
            "replay_schema_hash": replay_io.replay_schema_hash(
                obs_dim=obs_dim,
                action_count=ACTION_COUNT,
            ),
            "ruleset_id": ruleset_id,
            "ruleset_ids": ruleset_ids,
            "rules_hash": rules_hash,
            "rules_hash_scope": (
                "selected fixture ids, ruleset ids, body capacity, step index, and "
                "event mode only; not a full source rules hash"
            ),
            "observation_schema_id": debug_pack.DEBUG_OBS_SCHEMA,
            "observation_schema_hash": observation_schema_hash,
            "action_space_id": ACTION_SPACE_ID,
            "action_space_hash": action_space_hash,
            "reward_schema_id": debug_pack.DEBUG_REWARD_SCHEMA,
            "reward_schema_hash": reward_schema_hash,
            "env_impl_id": ENV_IMPL_ID,
            "env_impl_version": SCHEMA_VERSION,
            "producer": BENCHMARK_ID,
            "created_at": None,
            "created_at_status": "omitted_for_deterministic_sample",
        },
        "replay_v0_contract": {
            "decision": "sample_only_debug_schema_supported_for_p2",
            "production_training_decision": "blocked",
            "blocked_reason": (
                "actor bridge sample still emits debug obs/reward surfaces, not "
                "curvyzero trainer observations or a production replay stream"
            ),
            "replay_contract_id": replay_v0.REPLAY_CONTRACT_ID,
            "metadata_schema_id": replay_v0.REPLAY_METADATA_SCHEMA_ID,
            "replay_schema_id": replay_v0.REPLAY_SCHEMA_ID,
            "replay_schema_hash": replay_v0.replay_schema_hash(
                obs_dim=obs_dim,
                action_count=ACTION_COUNT,
            ),
            "player_count_required": replay_v0.PLAYER_COUNT,
            "observation_payload_policy": (
                "debug obs float32[T,B,P,9]; this is not final trainer observation"
            ),
            "reward_payload_policy": (
                "debug score/round-score/death-penalty reward; not sparse "
                "trainer reward"
            ),
            "final_observation_policy": (
                "last staged debug obs per env row; not final trainer observation"
            ),
            "final_reward_map_policy": (
                "sample-only 1v1 no-bonus terminal map derived from final debug "
                "alive/done/truncated surfaces; zeros for nonterminal rows"
            ),
        },
        "replay_v0_chunk_level_metadata": {
            "replay_contract_id": replay_v0.REPLAY_CONTRACT_ID,
            "metadata_schema_id": replay_v0.REPLAY_METADATA_SCHEMA_ID,
            "replay_schema_id": replay_v0.REPLAY_SCHEMA_ID,
            "replay_schema_hash": replay_v0.replay_schema_hash(
                obs_dim=obs_dim,
                action_count=ACTION_COUNT,
            ),
            "ruleset_id": ruleset_id,
            "ruleset_ids": ruleset_ids,
            "rules_hash": rules_hash,
            "rules_hash_scope": (
                "selected fixture ids, ruleset ids, body capacity, step index, and "
                "event mode only; not a full source rules hash"
            ),
            "observation_schema_id": debug_pack.DEBUG_OBS_SCHEMA,
            "observation_schema_hash": observation_schema_hash,
            "action_space_id": ACTION_SPACE_ID,
            "action_space_hash": action_space_hash,
            "reward_schema_id": debug_pack.DEBUG_REWARD_SCHEMA,
            "reward_schema_hash": reward_schema_hash,
            "producer": BENCHMARK_ID,
            "created_at": None,
            "created_at_status": "omitted_for_deterministic_sample",
        },
        "sample_shape": {
            "batch_size": batch_size,
            "player_count": player_count,
            "obs_dim": obs_dim,
            "action_count": ACTION_COUNT,
            "event_mode": event_mode,
        },
        "fixture_ids": fixture_ids,
        "missing_before_production_replay": [
            "real source rules hash",
            "production replay manifest, compaction, and reader integration",
            "final learned observation and reward schemas",
            "actor bridge trainer observation adapter for replay v0 promotion",
        ],
    }


def _sample_replay_chunk_report(
    replay_chunk_path: str | Path | None,
    replay: ReplayChunk,
    sample_contract_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if replay_chunk_path is None:
        return {
            "status": "not_requested",
            "scope": "sample-only debug replay chunk output",
        }
    if replay.cursor != 0 or replay.chunks_completed != 1:
        raise vector_compare.VectorCompareError(
            "sample replay chunk must contain exactly one complete rollout chunk"
        )

    arrays = _replay_arrays(replay)
    chunk_metadata = _mapping(
        sample_contract_metadata["chunk_level_metadata"],
        "sample_contract_metadata.chunk_level_metadata",
    )
    metadata = replay_io.build_debug_actor_loop_replay_metadata(
        arrays,
        ruleset_id=_string(chunk_metadata["ruleset_id"], "chunk_level_metadata.ruleset_id"),
        rules_hash=_string(chunk_metadata["rules_hash"], "chunk_level_metadata.rules_hash"),
        observation_schema_id=_string(
            chunk_metadata["observation_schema_id"],
            "chunk_level_metadata.observation_schema_id",
        ),
        observation_schema_hash=_string(
            chunk_metadata["observation_schema_hash"],
            "chunk_level_metadata.observation_schema_hash",
        ),
        action_space_id=_string(
            chunk_metadata["action_space_id"],
            "chunk_level_metadata.action_space_id",
        ),
        action_space_hash=_string(
            chunk_metadata["action_space_hash"],
            "chunk_level_metadata.action_space_hash",
        ),
        reward_schema_id=_string(
            chunk_metadata["reward_schema_id"],
            "chunk_level_metadata.reward_schema_id",
        ),
        reward_schema_hash=_string(
            chunk_metadata["reward_schema_hash"],
            "chunk_level_metadata.reward_schema_hash",
        ),
        env_impl_id=ENV_IMPL_ID,
        env_impl_version=SCHEMA_VERSION,
        producer=BENCHMARK_ID,
        created_at=None,
    )
    started = time.perf_counter()
    replay_io.write_debug_actor_loop_replay_chunk(
        replay_chunk_path,
        arrays=arrays,
        metadata=metadata,
    )
    write_elapsed_sec = time.perf_counter() - started
    file_bytes = _file_bytes(replay_chunk_path)
    return {
        "status": "written",
        "path": str(replay_chunk_path),
        "metadata_schema_id": replay_io.REPLAY_METADATA_SCHEMA_ID,
        "replay_schema_id": replay_io.REPLAY_SCHEMA_ID,
        "compatibility_metadata": replay_io.compatibility_metadata(metadata),
        "array_shapes": _chunk_shapes(replay),
        "array_dtypes": _chunk_dtypes(replay),
        "write_elapsed_sec": write_elapsed_sec,
        "file_bytes": file_bytes,
        "write_mb_per_sec": _mb_per_sec(file_bytes, write_elapsed_sec),
    }


def _sample_replay_v0_chunk_report(
    replay_v0_chunk_path: str | Path | None,
    replay: ReplayChunk,
    sample_contract_metadata: Mapping[str, Any],
    *,
    selected_templates: Sequence[Mapping[str, Any]],
    selected_group_key: str,
    seed: int,
) -> dict[str, Any]:
    if replay_v0_chunk_path is None:
        return {
            "status": "not_requested",
            "scope": "sample-only replay v0 chunk output",
            "production_training_decision": "blocked",
            "blocked_reason": (
                "actor bridge sample still emits debug obs/reward surfaces, not "
                "curvyzero trainer observations or a production replay stream"
            ),
        }
    if replay.cursor != 0 or replay.chunks_completed != 1:
        raise vector_compare.VectorCompareError(
            "sample replay v0 chunk must contain exactly one complete rollout chunk"
        )
    if replay.obs.shape[2] != replay_v0.PLAYER_COUNT:
        raise vector_compare.VectorCompareError(
            "sample replay v0 chunk is only supported for 1v1/P=2 samples"
        )
    if seed < 0:
        raise vector_compare.VectorCompareError(
            "sample replay v0 chunk reset_seed metadata requires a nonnegative seed"
        )

    arrays = _replay_v0_arrays(
        replay,
        selected_group_key=selected_group_key,
        selected_templates=selected_templates,
        seed=seed,
    )
    chunk_metadata = _mapping(
        sample_contract_metadata["replay_v0_chunk_level_metadata"],
        "sample_contract_metadata.replay_v0_chunk_level_metadata",
    )
    metadata = replay_v0.build_replay_chunk_v0_metadata(
        arrays,
        rules_hash=_string(chunk_metadata["rules_hash"], "replay_v0.rules_hash"),
        observation_schema_hash=_string(
            chunk_metadata["observation_schema_hash"],
            "replay_v0.observation_schema_hash",
        ),
        action_space_hash=_string(
            chunk_metadata["action_space_hash"],
            "replay_v0.action_space_hash",
        ),
        reward_schema_hash=_string(
            chunk_metadata["reward_schema_hash"],
            "replay_v0.reward_schema_hash",
        ),
        ruleset_id=_string(chunk_metadata["ruleset_id"], "replay_v0.ruleset_id"),
        observation_schema_id=_string(
            chunk_metadata["observation_schema_id"],
            "replay_v0.observation_schema_id",
        ),
        action_space_id=_string(
            chunk_metadata["action_space_id"],
            "replay_v0.action_space_id",
        ),
        reward_schema_id=_string(
            chunk_metadata["reward_schema_id"],
            "replay_v0.reward_schema_id",
        ),
        producer=BENCHMARK_ID,
        created_at=None,
    )
    started = time.perf_counter()
    replay_v0.write_replay_chunk_v0(
        replay_v0_chunk_path,
        arrays=arrays,
        metadata=metadata,
    )
    write_elapsed_sec = time.perf_counter() - started
    file_bytes = _file_bytes(replay_v0_chunk_path)
    return {
        "status": "written",
        "path": str(replay_v0_chunk_path),
        "metadata_schema_id": replay_v0.REPLAY_METADATA_SCHEMA_ID,
        "replay_contract_id": replay_v0.REPLAY_CONTRACT_ID,
        "replay_schema_id": replay_v0.REPLAY_SCHEMA_ID,
        "production_training_decision": "blocked",
        "blocked_reason": (
            "written chunk is valid replay v0 shape metadata, but carries debug "
            "actor-bridge obs/reward payloads"
        ),
        "compatibility_metadata": replay_v0.compatibility_metadata(metadata),
        "array_shapes": metadata["array_shapes"],
        "array_dtypes": metadata["array_dtypes"],
        "write_elapsed_sec": write_elapsed_sec,
        "file_bytes": file_bytes,
        "write_mb_per_sec": _mb_per_sec(file_bytes, write_elapsed_sec),
    }


def _stable_contract_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _file_bytes(path: str | Path) -> int:
    return int(Path(path).stat().st_size)


def _mb_per_sec(byte_count: int, elapsed_sec: float) -> float | None:
    if elapsed_sec <= 0:
        return None
    return float(byte_count) / (1024.0 * 1024.0) / elapsed_sec


def _chunk_nbytes(replay: ReplayChunk) -> int:
    return int(
        replay.obs.nbytes
        + replay.reward.nbytes
        + replay.action.nbytes
        + replay.action_weights.nbytes
        + replay.root_value.nbytes
        + replay.done.nbytes
        + replay.ego_mask.nbytes
    )


def _replay_arrays(replay: ReplayChunk) -> dict[str, np.ndarray]:
    return {
        "obs": replay.obs,
        "reward": replay.reward,
        "action": replay.action,
        "action_weights": replay.action_weights,
        "root_value": replay.root_value,
        "done": replay.done,
        "ego_mask": replay.ego_mask,
    }


def _replay_v0_arrays(
    replay: ReplayChunk,
    *,
    selected_group_key: str,
    selected_templates: Sequence[Mapping[str, Any]],
    seed: int,
) -> dict[str, np.ndarray]:
    done = np.logical_or(replay.terminated, replay.truncated)
    batch_size = int(replay.obs.shape[1])
    obs_dim = int(replay.obs.shape[3])
    last_slot = replay.obs.shape[0] - 1
    fixture_ids = [str(template["scenario_id"]) for template in selected_templates]
    episode_id = np.asarray(
        [
            f"{selected_group_key}:sample:{row}:fixtures={','.join(fixture_ids)}"
            for row in range(batch_size)
        ],
        dtype="<U256",
    )
    reset_source = np.asarray(
        [f"fixture_sample:{selected_group_key}" for _ in range(batch_size)],
        dtype="<U128",
    )
    reset_seed = np.asarray(
        [int(seed) + row for row in range(batch_size)],
        dtype=np.int64,
    )
    final_reward_map = np.zeros(
        (batch_size, replay_v0.PLAYER_COUNT),
        dtype=np.float32,
    )
    for row in range(batch_size):
        done_slots = np.flatnonzero(done[:, row])
        if done_slots.size:
            final_reward_map[row] = replay.reward[int(done_slots[-1]), row]

    return {
        "observation": replay.obs.astype(np.float32, copy=True),
        "reward": replay.reward.astype(np.float32, copy=True),
        "action": replay.action.astype(np.int16, copy=True),
        "action_weights": replay.action_weights.astype(np.float32, copy=True),
        "root_value": replay.root_value.astype(np.float32, copy=True),
        "done": done.astype(bool, copy=True),
        "terminated": replay.terminated.astype(bool, copy=True),
        "truncated": replay.truncated.astype(bool, copy=True),
        "episode_id": episode_id,
        "reset_seed": reset_seed,
        "reset_source": reset_source,
        "final_observation": replay.obs[last_slot].reshape(
            batch_size,
            replay_v0.PLAYER_COUNT,
            obs_dim,
        ).astype(np.float32, copy=True),
        "final_reward_map": final_reward_map,
    }


def _chunk_shapes(replay: ReplayChunk) -> dict[str, list[int]]:
    return {
        "obs": list(replay.obs.shape),
        "reward": list(replay.reward.shape),
        "action": list(replay.action.shape),
        "action_weights": list(replay.action_weights.shape),
        "root_value": list(replay.root_value.shape),
        "done": list(replay.done.shape),
        "ego_mask": list(replay.ego_mask.shape),
    }


def _chunk_dtypes(replay: ReplayChunk) -> dict[str, str]:
    return {
        "obs": str(replay.obs.dtype),
        "reward": str(replay.reward.dtype),
        "action": str(replay.action.dtype),
        "action_weights": str(replay.action_weights.dtype),
        "root_value": str(replay.root_value.dtype),
        "done": str(replay.done.dtype),
        "ego_mask": str(replay.ego_mask.dtype),
    }


def _histogram(values: np.ndarray, bins: int) -> list[int]:
    counts = np.bincount(values.reshape(-1).astype(np.int64), minlength=bins)
    return [int(count) for count in counts[:bins]]


def _surface_nbytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(_surface_nbytes(child) for child in value.values())
    return 0


def _fixed_shape_cost_report(
    *,
    batch_size: int,
    player_count: int,
    event_mode: str,
    timers: Mapping[str, Any],
    counters: Mapping[str, int],
    event_overhead_sec: float,
) -> dict[str, Any]:
    loop_sec = float(timers["loop_elapsed_sec"])
    env_step_sec = float(timers["env_step_sec"])
    env_rows = int(counters["env_rows"])
    ego_rows = int(counters["ego_rows"])
    active_ego_rows = int(counters["active_ego_rows"])
    env_step_calls = int(counters["env_step_calls"])
    rollout_blocks = int(counters["rollout_blocks"])
    chunk_stage_calls = int(counters["chunk_stage_calls"])
    policy_total_sec = float(timers["policy_total_sec"])

    def bucket(sec: float) -> dict[str, float]:
        return {
            "sec": sec,
            "pct_loop": _rate(sec * 100.0, loop_sec),
            "us_per_env_row": _rate(sec * 1_000_000.0, env_rows),
            "us_per_ego_row": _rate(sec * 1_000_000.0, ego_rows),
            "us_per_active_ego_row": _rate(sec * 1_000_000.0, active_ego_rows),
            "us_per_env_step_call": _rate(sec * 1_000_000.0, env_step_calls),
            "us_per_rollout_block": _rate(sec * 1_000_000.0, rollout_blocks),
            "env_rows_per_sec": _rate(env_rows, sec),
            "ego_rows_per_sec": _rate(ego_rows, sec),
        }

    buckets = {
        name: bucket(float(timers[timer_key]))
        for name, timer_key in FIXED_SHAPE_BUCKET_KEYS
    }
    event_count = int(counters["env_events_emitted"])
    event_overflow_attempts = int(counters["env_event_overflow_attempts"])
    return {
        "schema": FIXED_SHAPE_COST_SCHEMA_VERSION,
        "normalization": (
            "all per-row costs are normalized over the timed loop's fixed "
            "env_rows and ego_rows; reset-copy is also shown per rollout block"
        ),
        "shape": {
            "batch_size": batch_size,
            "player_count": player_count,
            "event_mode": event_mode,
            "events_enabled": batch_rows._events_enabled(event_mode),
            "rollout_blocks": rollout_blocks,
            "env_step_calls": env_step_calls,
            "chunk_stage_calls": chunk_stage_calls,
            "env_rows": env_rows,
            "ego_rows": ego_rows,
            "active_ego_rows": active_ego_rows,
        },
        "buckets": buckets,
        "aggregates": {
            "policy_total": bucket(policy_total_sec),
        },
        "event_cost": {
            "sec": event_overhead_sec,
            "pct_loop": _rate(event_overhead_sec * 100.0, loop_sec),
            "pct_env_step": _rate(event_overhead_sec * 100.0, env_step_sec),
            "us_per_env_row": _rate(event_overhead_sec * 1_000_000.0, env_rows),
            "events_emitted": event_count,
            "event_overflow_attempts": event_overflow_attempts,
            "events_per_env_row": _rate(event_count, env_rows),
        },
    }


def _training_rate_report(
    *,
    timers: Mapping[str, Any],
    counters: Mapping[str, int],
) -> dict[str, Any]:
    loop_sec = float(timers["loop_elapsed_sec"])
    env_step_sec = float(timers["env_step_sec"])
    debug_pack_sec = float(timers["debug_pack_sec"])
    policy_total_sec = float(timers["policy_total_sec"])
    replay_sec = float(timers["replay_chunk_stage_sec"])
    reset_sec = float(timers["reset_copy_sec"])
    autoreset_sec = float(timers["autoreset_sec"])
    overhead_sec = float(timers["loop_overhead_sec"])
    env_rows = int(counters["env_rows"])
    ego_rows = int(counters["ego_rows"])
    active_ego_rows = int(counters["active_ego_rows"])
    staged_transition_rows = int(counters["replay_rows"])
    final_transition_rows = int(counters["final_transition_rows_before_autoreset"])
    done_rows = int(counters["done_rows"])
    truncated_rows = int(counters["truncated_rows"])
    autoreset_rows = int(counters["autoreset_rows"])

    return {
        "schema": TRAINING_RATE_SCHEMA_VERSION,
        "timing_scope": "timed actor loop only; excludes setup, source preflight, and warmup",
        "completion_proxy": (
            "final_transition_env_rows counts rows with done or truncated after replay "
            "staging and before debug internal autoreset; this is a fixture-reset "
            "games/min proxy, not production training completion throughput"
        ),
        "metric_caveats": [
            "staged transition rows/sec can rise with larger batches while action latency gets worse",
            "completed_game_rows/min is a terminal-row proxy inside fixture-reset rollout blocks, not production self-play games/min",
            "synthetic_policy_search_total is local NumPy shape work and can mislead about real model or MCTS cost",
            "Amdahl percentages show this benchmark's loop composition only; they do not predict GPU or async actor scaling",
        ],
        "counts": {
            "env_rows": env_rows,
            "ego_rows": ego_rows,
            "active_ego_rows": active_ego_rows,
            "staged_transition_ego_rows": staged_transition_rows,
            "final_transition_env_rows": final_transition_rows,
            "done_rows": done_rows,
            "truncated_rows": truncated_rows,
            "state_reset_blocks": int(counters["state_reset_blocks"]),
            "state_reset_env_rows": int(counters["state_reset_env_rows"]),
            "autoreset_rows": autoreset_rows,
        },
        "timing_sec": {
            "loop_elapsed_sec": loop_sec,
            "env_step_sec": env_step_sec,
            "debug_pack_sec": debug_pack_sec,
            "synthetic_policy_search_total_sec": policy_total_sec,
            "replay_chunk_stage_sec": replay_sec,
            "reset_copy_sec": reset_sec,
            "autoreset_sec": autoreset_sec,
            "loop_overhead_sec": overhead_sec,
        },
        "loop_rates": {
            "env_rows_per_sec_total_loop": _rate(env_rows, loop_sec),
            "ego_rows_per_sec_total_loop": _rate(ego_rows, loop_sec),
            "active_ego_rows_per_sec_total_loop": _rate(active_ego_rows, loop_sec),
            "staged_transition_ego_rows_per_sec_total_loop": _rate(
                staged_transition_rows,
                loop_sec,
            ),
            "staged_transition_ego_rows_per_min_total_loop": _rate(
                staged_transition_rows * 60.0,
                loop_sec,
            ),
            "final_transition_env_rows_per_sec_total_loop": _rate(
                final_transition_rows,
                loop_sec,
            ),
            "final_transition_env_rows_per_min_total_loop": _rate(
                final_transition_rows * 60.0,
                loop_sec,
            ),
            "completed_game_rows_per_sec_total_loop": _rate(
                final_transition_rows,
                loop_sec,
            ),
            "completed_game_rows_per_min_total_loop": _rate(
                final_transition_rows * 60.0,
                loop_sec,
            ),
            "done_rows_per_min_total_loop": _rate(done_rows * 60.0, loop_sec),
            "truncated_rows_per_min_total_loop": _rate(
                truncated_rows * 60.0,
                loop_sec,
            ),
            "autoreset_rows_per_min_total_loop": _rate(autoreset_rows * 60.0, loop_sec),
        },
        "bucket_rates": {
            "env_rows_per_sec_env_step_bucket": _rate(env_rows, env_step_sec),
            "ego_rows_per_sec_debug_pack_bucket": _rate(ego_rows, debug_pack_sec),
            "ego_rows_per_sec_synthetic_policy_bucket": _rate(
                ego_rows,
                policy_total_sec,
            ),
            "ego_rows_per_sec_replay_bucket": _rate(ego_rows, replay_sec),
        },
        "amdahl_breakdown_pct_loop": {
            "env_step": _rate(env_step_sec * 100.0, loop_sec),
            "debug_pack": _rate(debug_pack_sec * 100.0, loop_sec),
            "synthetic_policy_search_total": _rate(policy_total_sec * 100.0, loop_sec),
            "replay_chunk_stage": _rate(replay_sec * 100.0, loop_sec),
            "reset_copy": _rate(reset_sec * 100.0, loop_sec),
            "autoreset": _rate(autoreset_sec * 100.0, loop_sec),
            "loop_overhead": _rate(overhead_sec * 100.0, loop_sec),
            "non_env_step": _rate(max(0.0, loop_sec - env_step_sec) * 100.0, loop_sec),
        },
        "amdahl_ratios": {
            "synthetic_policy_to_env_step_time": _rate(policy_total_sec, env_step_sec),
            "env_step_to_loop_time": _rate(env_step_sec, loop_sec),
        },
    }


def _latency_report(samples: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    return {
        "schema": LATENCY_SCHEMA_VERSION,
        "units": "seconds",
        "sample_scope": (
            "per timed vector actor step; actor_step_total excludes final reporting "
            "sample construction and includes env step, debug pack, synthetic "
            "policy/search, action handling, replay staging, autoreset, and local "
            "loop overhead for that step"
        ),
        "metric_caveats": [
            "p50/p95/p99 are sampled from timed Python actor steps and can miss very rare stalls",
            "actor_step_total is a latency signal, so it should be read alongside batch size and transition throughput",
            "synthetic_policy_search_total latency is not learned-model inference or production MCTS latency",
        ],
        "actor_step_total_sec": _latency_summary(samples["actor_step_total_sec"]),
        "env_step_sec": _latency_summary(samples["env_step_sec"]),
        "synthetic_policy_search_total_sec": _latency_summary(
            samples["policy_total_sec"]
        ),
        "replay_chunk_stage_sec": _latency_summary(samples["replay_chunk_stage_sec"]),
    }


def _latency_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

    array = np.asarray(values, dtype=np.float64)
    p50, p95, p99 = np.percentile(array, [50.0, 95.0, 99.0])
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(array.max()),
    }


def _actor_event_mode_comparisons(batch_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_size: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for batch in batch_results:
        by_size[int(batch["batch_size"])][str(batch["event_mode"])] = batch

    comparisons = []
    for batch_size, by_mode in sorted(by_size.items()):
        debug_batch = by_mode.get(batch_rows.EVENT_MODE_DEBUG)
        no_event_batch = by_mode.get(batch_rows.EVENT_MODE_NONE)
        if debug_batch is None or no_event_batch is None:
            continue
        debug_timing = _mapping(debug_batch["timing_sec"], "debug_batch.timing_sec")
        no_event_timing = _mapping(no_event_batch["timing_sec"], "no_event_batch.timing_sec")
        debug_rates = _mapping(debug_batch["rates"], "debug_batch.rates")
        no_event_rates = _mapping(no_event_batch["rates"], "no_event_batch.rates")
        debug_counts = _mapping(debug_batch["counts"], "debug_batch.counts")
        debug_env_step_sec = float(debug_timing["env_step_sec"])
        no_event_env_step_sec = float(no_event_timing["env_step_sec"])
        debug_loop_sec = float(debug_timing["loop_elapsed_sec"])
        no_event_loop_sec = float(no_event_timing["loop_elapsed_sec"])
        env_step_delta = debug_env_step_sec - no_event_env_step_sec
        debug_pack_delta = float(debug_timing["debug_pack_sec"]) - float(
            no_event_timing["debug_pack_sec"]
        )
        policy_delta = float(debug_timing["policy_total_sec"]) - float(
            no_event_timing["policy_total_sec"]
        )
        replay_delta = float(debug_timing["replay_chunk_stage_sec"]) - float(
            no_event_timing["replay_chunk_stage_sec"]
        )
        comparisons.append(
            {
                "batch_size": batch_size,
                "rollout_steps": int(debug_batch["rollout_steps"]),
                "env_rows": int(debug_counts["env_rows"]),
                "ego_rows": int(debug_counts["ego_rows"]),
                "debug_events_emitted": int(debug_counts["env_events_emitted"]),
                "debug_event_overflow_attempts": int(
                    debug_counts["env_event_overflow_attempts"]
                ),
                "debug_event_env_step_sec": debug_env_step_sec,
                "no_event_env_step_sec": no_event_env_step_sec,
                "debug_minus_no_event_env_step_sec": env_step_delta,
                "debug_minus_no_event_env_step_pct": _rate(
                    env_step_delta * 100.0,
                    debug_env_step_sec,
                ),
                "debug_event_loop_sec": debug_loop_sec,
                "no_event_loop_sec": no_event_loop_sec,
                "debug_minus_no_event_loop_sec": debug_loop_sec - no_event_loop_sec,
                "debug_event_overhead_sec": float(debug_timing["env_event_overhead_sec"]),
                "debug_event_overhead_pct_env_step": float(
                    debug_rates["env_event_overhead_step_pct"]
                ),
                "debug_event_events_per_env_row": _rate(
                    int(debug_counts["env_events_emitted"]),
                    int(debug_counts["env_rows"]),
                ),
                "debug_event_debug_pack_sec": float(debug_timing["debug_pack_sec"]),
                "no_event_debug_pack_sec": float(no_event_timing["debug_pack_sec"]),
                "debug_minus_no_event_debug_pack_sec": debug_pack_delta,
                "debug_minus_no_event_debug_pack_pct": _rate(
                    debug_pack_delta * 100.0,
                    float(debug_timing["debug_pack_sec"]),
                ),
                "debug_event_policy_total_sec": float(debug_timing["policy_total_sec"]),
                "no_event_policy_total_sec": float(no_event_timing["policy_total_sec"]),
                "debug_minus_no_event_policy_total_sec": policy_delta,
                "debug_minus_no_event_policy_total_pct": _rate(
                    policy_delta * 100.0,
                    float(debug_timing["policy_total_sec"]),
                ),
                "debug_event_replay_chunk_stage_sec": float(
                    debug_timing["replay_chunk_stage_sec"]
                ),
                "no_event_replay_chunk_stage_sec": float(
                    no_event_timing["replay_chunk_stage_sec"]
                ),
                "debug_minus_no_event_replay_chunk_stage_sec": replay_delta,
                "debug_minus_no_event_replay_chunk_stage_pct": _rate(
                    replay_delta * 100.0,
                    float(debug_timing["replay_chunk_stage_sec"]),
                ),
                "debug_event_env_rows_per_sec_total_loop": float(
                    debug_rates["env_rows_per_sec_total_loop"]
                ),
                "no_event_env_rows_per_sec_total_loop": float(
                    no_event_rates["env_rows_per_sec_total_loop"]
                ),
                "no_event_total_loop_speedup_vs_debug": _rate(
                    float(no_event_rates["env_rows_per_sec_total_loop"]),
                    float(debug_rates["env_rows_per_sec_total_loop"]),
                ),
                "debug_event_env_rows_per_sec_env_step_bucket": float(
                    debug_rates["env_rows_per_sec_env_step_bucket"]
                ),
                "no_event_env_rows_per_sec_env_step_bucket": float(
                    no_event_rates["env_rows_per_sec_env_step_bucket"]
                ),
                "no_event_env_step_speedup_vs_debug": _rate(
                    float(no_event_rates["env_rows_per_sec_env_step_bucket"]),
                    float(debug_rates["env_rows_per_sec_env_step_bucket"]),
                ),
            }
        )
    return comparisons


def _top_bucket_label(timers: Mapping[str, Any]) -> str:
    candidates = {
        "reset_copy": float(timers["reset_copy_sec"]),
        "previous_delta_snapshot": float(timers["previous_delta_snapshot_sec"]),
        "env_step": float(timers["env_step_sec"]),
        "debug_pack": float(timers["debug_pack_sec"]),
        "policy_root": float(timers["policy_root_sec"]),
        "policy_search": float(timers["policy_search_sec"]),
        "action_select": float(timers["action_select_sec"]),
        "action_encode": float(timers["action_encode_sec"]),
        "replay_chunk_stage": float(timers["replay_chunk_stage_sec"]),
        "autoreset": float(timers["autoreset_sec"]),
        "loop_overhead": float(timers["loop_overhead_sec"]),
    }
    return max(candidates, key=candidates.get)


def _top_phase_label(phase_timers: Mapping[str, float]) -> str:
    if not phase_timers:
        return "none"
    name, value = max(phase_timers.items(), key=lambda item: item[1])
    return f"{name}:{value:.6f}s"


def _rate(numerator: int | float, denominator_sec: float) -> float:
    if denominator_sec <= 0:
        return 0.0
    return float(numerator) / denominator_sec


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise vector_compare.VectorCompareError(f"{field} must be an object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise vector_compare.VectorCompareError(f"{field} must be a list")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise vector_compare.VectorCompareError(f"{field} must be a non-empty string")
    return value


def _int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise vector_compare.VectorCompareError(f"{field} must be an integer")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the current fixture-seeded vector step -> debug pack -> "
            "synthetic policy/search -> replay chunk bridge."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=list(DEFAULT_PATHS),
        help="Scenario JSON files or batch manifests. Defaults to the current supported set.",
    )
    parser.add_argument(
        "--body-capacity",
        type=_nonnegative_int,
        default=4,
        help="Fixed K for the seeded body buffer. Defaults to the narrow K=4 profile.",
    )
    parser.add_argument("--step-index", type=_nonnegative_int, default=0)
    parser.add_argument("--batch-sizes", type=_positive_int, nargs="+", default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument(
        "--event-modes",
        choices=batch_rows.EVENT_MODE_CHOICES,
        nargs="+",
        default=list(batch_rows.DEFAULT_EVENT_MODES),
        help=(
            "Event logging modes to time. Use 'debug-event no-event' to isolate "
            "debug event row cost inside the env step. Defaults to debug-event."
        ),
    )
    parser.add_argument("--repeat", type=_positive_int, default=1_000)
    parser.add_argument("--warmup", type=_nonnegative_int, default=100)
    parser.add_argument(
        "--rollout-steps",
        type=_positive_int,
        default=1,
        help=(
            "Consecutive vector steps per timed block. Step 0 uses fixture source moves; "
            "later steps feed back synthetic selected actions. Defaults to 1."
        ),
    )
    parser.add_argument("--hidden-dim", type=_positive_int, default=32)
    parser.add_argument(
        "--simulations",
        type=_positive_int,
        default=4,
        help=(
            "Scale the synthetic NumPy recurrent/search-shaped stand-in inside the "
            "actor loop. Higher values emulate heavier model/search latency for "
            "latency-vs-throughput scouts, but this is still not real MCTS, Mctx, "
            "JAX, GPU, or learned-model timing."
        ),
    )
    parser.add_argument("--chunk-steps", type=_positive_int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help=(
            "Build one fixed-shape actor bridge output sample instead of running "
            "the timing benchmark. Uses the first batch size and first event mode."
        ),
    )
    parser.add_argument(
        "--sample-replay-chunk",
        default=None,
        help=(
            "Optional .npz path for --sample-only. Writes one validated local debug "
            "actor-loop replay chunk for the requested fixed-shape sample."
        ),
    )
    parser.add_argument(
        "--sample-replay-v0-chunk",
        default=None,
        help=(
            "Optional .npz path for --sample-only. Writes one validated replay v0 "
            "sample chunk for a 1v1/P=2 fixed-shape sample. This is still debug "
            "obs/reward payload data, not production training data."
        ),
    )
    parser.add_argument(
        "--player-count",
        type=_positive_int,
        default=None,
        help="Optional player-count filter for --sample-only.",
    )
    parser.add_argument(
        "--group-id",
        default=None,
        help="Optional fixed-shape group id filter for --sample-only, such as P2_K4.",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Allow fixtures that are not marked python-runner-verified.",
    )
    parser.add_argument(
        "--fail-on-unsupported",
        action="store_true",
        help="Exit nonzero when any fixture is unsupported.",
    )
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    args = parser.parse_args()

    if args.sample_replay_chunk is not None and not args.sample_only:
        parser.error("--sample-replay-chunk requires --sample-only")
    if args.sample_replay_v0_chunk is not None and not args.sample_only:
        parser.error("--sample-replay-v0-chunk requires --sample-only")

    if args.sample_only:
        sample_payload = build_fixture_seeded_actor_bridge_sample(
            args.paths,
            body_capacity=args.body_capacity,
            step_index=args.step_index,
            batch_size=args.batch_sizes[0],
            player_count=args.player_count,
            group_id=args.group_id,
            event_mode=args.event_modes[0],
            rollout_steps=args.rollout_steps,
            hidden_dim=args.hidden_dim,
            simulations=args.simulations,
            seed=args.seed,
            require_verified=not args.allow_unverified,
            replay_chunk_path=args.sample_replay_chunk,
            replay_v0_chunk_path=args.sample_replay_v0_chunk,
        )
        print(json.dumps(sample_payload["source"], indent=2, sort_keys=True))
        return

    summary = benchmark_inputs(
        args.paths,
        body_capacity=args.body_capacity,
        step_index=args.step_index,
        batch_sizes=args.batch_sizes,
        event_modes=args.event_modes,
        repeat=args.repeat,
        warmup=args.warmup,
        rollout_steps=args.rollout_steps,
        hidden_dim=args.hidden_dim,
        simulations=args.simulations,
        chunk_steps=args.chunk_steps,
        seed=args.seed,
        require_verified=not args.allow_unverified,
    )
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_plain(summary)

    counts = summary["summary"]
    if (
        counts["failed"]
        or counts["batch_preflight_failed"]
        or (args.fail_on_unsupported and counts["unsupported"])
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
