"""Benchmark debug observation/reward packing on current vector array shapes.

This is a prototype only. It reuses the current fixture-seeded B>1 state rows,
packs a small fixed debug surface, and reports pack-only timing. The observation
and reward here are not the final training schema.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
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
import compare_vector_arrays_to_fidelity as vector_compare  # noqa: E402
import seed_vector_state_from_fixtures as seed_bridge  # noqa: E402


SCHEMA_VERSION = "curvyzero_vector_debug_obs_reward_packing/v1"
BENCHMARK_ID = "fixture_seeded_numpy_debug_obs_reward_packing"
DEBUG_OBS_SCHEMA = "curvyzero_debug_global_player_obs/v0"
DEBUG_REWARD_SCHEMA = "curvyzero_debug_score_round_delta_death_penalty/v0"
ACTION_COUNT = 3
DEFAULT_PATHS = batch_rows.DEFAULT_PATHS
DEFAULT_BATCH_SIZES = (1, 32, 128)
DEBUG_OBS_FEATURE_NAMES = (
    "x_over_map_size",
    "y_over_map_size",
    "heading_sin",
    "heading_cos",
    "alive",
    "printing",
    "score",
    "round_score",
    "map_size_over_1000",
)
KNOWN_FAKE_OR_INCOMPLETE = [
    "obs is a tiny privileged debug player-state surface, not an egocentric training observation",
    "reward is a placeholder/narrow debug formula; die uses event rows when present",
    "score and round-score reward terms still come from state deltas, not a final reward schema",
    "done is derived from existing done plus alive_count <= 1; there is no reset/autoreset contract",
    "truncated is derived from overflow flags only; there is no horizon or rollout-limit policy",
    "legal masks expose the three source move actions as all legal for active live ego rows",
    "rows are made by cycling current supported fixture seeds inside fixed P/K groups",
    "no policy, MCTS/search, replay chunking, wrapper dict conversion, or training loop is measured",
]


def benchmark_inputs(
    paths: Sequence[str | Path],
    *,
    body_capacity: int = 4,
    step_index: int = 0,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    repeat: int = 10_000,
    warmup: int = 500,
    require_verified: bool = True,
) -> dict[str, Any]:
    """Benchmark packing debug surfaces from fixture-seeded B>1 state rows."""

    if body_capacity < 0:
        raise vector_compare.VectorCompareError("body_capacity must be zero or greater")
    if step_index < 0:
        raise vector_compare.VectorCompareError("step_index must be zero or greater")
    if repeat <= 0:
        raise vector_compare.VectorCompareError("repeat must be greater than zero")
    if warmup < 0:
        raise vector_compare.VectorCompareError("warmup must be zero or greater")
    batch_sizes = tuple(int(size) for size in batch_sizes)
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise vector_compare.VectorCompareError("batch_sizes must contain positive integers")

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
    for group_key, group_templates in sorted(groups_by_key.items()):
        started = time.perf_counter()
        preflight = batch_rows._batch_state_preflight(group_templates)
        timers["batch_state_preflight_sec"] += time.perf_counter() - started

        batch_results = []
        if can_time and preflight["match"]:
            for batch_size in batch_sizes:
                batch_results.append(
                    _benchmark_group_packing(
                        group_templates,
                        batch_size=batch_size,
                        repeat=repeat,
                        warmup=warmup,
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
                "batches": batch_results,
            }
        )

    batch_preflight_failed = any(not group["preflight"]["match"] for group in groups)
    wall_elapsed_sec = time.perf_counter() - wall_started
    return {
        "schema": SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "source_fidelity_claim": (
            "state and event rows are compared once per supported fixture by "
            "compare_vector_arrays_to_fidelity.py before timing; timed packing "
            "uses the B>1 batch-row state path and reads die event rows when present"
        ),
        "trust_level": (
            "Debug packing benchmark only. It proves fixed shapes and pack-only "
            "timing for the current fixture slice, not a final training observation "
            "or reward contract."
        ),
        "packing_contract": {
            "obs_schema": DEBUG_OBS_SCHEMA,
            "reward_schema": DEBUG_REWARD_SCHEMA,
            "obs_shape": "[B,P,9]",
            "obs_dtype": "float32",
            "obs_features": list(DEBUG_OBS_FEATURE_NAMES),
            "reward_shape": "[B,P]",
            "reward_dtype": "float32",
            "reward_formula": "score_delta + round_score_delta - died_this_step",
            "reward_status": (
                "placeholder/narrow debug reward; die uses event rows when present, "
                "but this is not a training reward contract"
            ),
            "legal_action_mask_shape": "[B,P,3]",
            "legal_action_values": "left, straight, right are all true for active live ego rows",
            "ego_row_id_shape": "[B,P]",
            "done_shape": "[B] derived from state.done OR alive_count <= 1",
            "truncated_shape": "[B] derived from overflow/event_overflow when present",
        },
        "config": {
            "paths": [str(path) for path in paths],
            "body_capacity": body_capacity,
            "step_index": step_index,
            "batch_sizes": list(batch_sizes),
            "repeat": repeat,
            "warmup": warmup,
            "require_verified": require_verified,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
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
        "hot_loop_exclusions": {
            "env_step_calls": 0,
            "source_trace_calls": 0,
            "projection_calls": 0,
            "comparison_calls": 0,
            "policy_calls": 0,
            "mcts_calls": 0,
            "replay_writes": 0,
            "note": (
                "timed loop packs already-stepped arrays only; the one batch step "
                "per B/profile is setup for reward deltas"
            ),
        },
        "known_fake_or_incomplete": KNOWN_FAKE_OR_INCOMPLETE,
    }


def build_fixture_seeded_debug_surfaces(
    paths: Sequence[str | Path] = DEFAULT_PATHS,
    *,
    body_capacity: int = 4,
    step_index: int = 0,
    batch_size: int = 1,
    player_count: int | None = None,
    group_id: str | None = None,
    require_verified: bool = True,
) -> dict[str, Any]:
    """Build one fixture-seeded B>1 debug obs/reward surface sample."""

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

    wall_started = time.perf_counter()
    timers = {
        "seed_inputs_sec": 0.0,
        "source_preflight_sec": 0.0,
        "array_prepare_sec": 0.0,
        "batch_state_preflight_sec": 0.0,
        "single_batch_step_sec": 0.0,
        "pack_sec": 0.0,
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
            f"cannot build debug surfaces with {failed} source preflight failure(s)"
        )

    groups_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for template in templates:
        groups_by_key[template["group_key"]].append(template)

    selected_group_key = ""
    selected_templates: list[dict[str, Any]] = []
    selected_preflight: dict[str, Any] | None = None
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

        selected_group_key = candidate_group_key
        selected_templates = list(group_templates)
        selected_preflight = preflight
        break

    if selected_preflight is None:
        available_groups = sorted(groups_by_key)
        raise vector_compare.VectorCompareError(
            "no fixture group matched debug surface request; "
            f"requested player_count={player_count!r}, group_id={group_id!r}, "
            f"available_groups={available_groups!r}, skipped_groups={skipped_groups!r}"
        )

    batch = batch_rows._build_batch(selected_templates, batch_size)
    previous_state = batch["initial_state"]
    next_state = vector_compare.copy_array_state(previous_state)
    selected_player_count = _int(
        batch["prepared_batch"]["player_count"],
        "prepared_batch.player_count",
    )

    started = time.perf_counter()
    step_counters = batch_rows.step_batched_arrays(next_state, batch["prepared_batch"])
    timers["single_batch_step_sec"] = time.perf_counter() - started

    started = time.perf_counter()
    surfaces = pack_debug_obs_reward(
        previous_state,
        next_state,
        player_count=selected_player_count,
    )
    timers["pack_sec"] = time.perf_counter() - started
    timers["wall_elapsed_sec"] = time.perf_counter() - wall_started

    return {
        "surfaces": surfaces,
        "source": {
            "schema": SCHEMA_VERSION,
            "benchmark_id": BENCHMARK_ID,
            "source": "fixture_seeded_cpu_debug_packer",
            "source_fidelity_claim": (
                "state and event rows are compared once per supported fixture by "
                "compare_vector_arrays_to_fidelity.py before building this sample"
            ),
            "trust_level": (
                "One fixture-seeded debug packer sample. This is not a rollout, "
                "trainer, final observation schema, or replay contract."
            ),
            "config": {
                "paths": [str(path) for path in paths],
                "body_capacity": body_capacity,
                "step_index": step_index,
                "batch_size": batch_size,
                "player_count": player_count,
                "group_id": group_id,
                "require_verified": require_verified,
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
            },
            "skipped_groups": skipped_groups,
            "step_counters": step_counters,
            "sample": _surface_sample_summary(surfaces),
            "timing_sec": timers,
            "known_fake_or_incomplete": KNOWN_FAKE_OR_INCOMPLETE,
        },
    }


def pack_debug_obs_reward(
    previous_state: Mapping[str, np.ndarray],
    state: Mapping[str, np.ndarray],
    *,
    player_count: int,
) -> dict[str, Any]:
    """Pack fixed debug obs/reward/masks for the current B,P state arrays."""

    pos = np.asarray(state["pos"][:, :player_count], dtype=np.float64)
    heading = np.asarray(state["heading"][:, :player_count], dtype=np.float64)
    alive = np.asarray(state["alive"][:, :player_count], dtype=bool)
    printing = np.asarray(state["printing"][:, :player_count], dtype=bool)
    previous_alive = np.asarray(previous_state["alive"][:, :player_count], dtype=bool)
    score = np.asarray(state["score"][:, :player_count], dtype=np.int32)
    previous_score = np.asarray(previous_state["score"][:, :player_count], dtype=np.int32)
    round_score = np.asarray(state["round_score"][:, :player_count], dtype=np.int32)
    previous_round_score = np.asarray(
        previous_state["round_score"][:, :player_count],
        dtype=np.int32,
    )
    map_size = np.asarray(state["map_size"], dtype=np.float64)
    batch_size = pos.shape[0]
    if pos.shape != (batch_size, player_count, 2):
        raise vector_compare.VectorCompareError("state.pos must have shape [B,P,2]")

    safe_map_size = np.where(map_size == 0.0, 1.0, map_size)
    obs = np.empty((batch_size, player_count, len(DEBUG_OBS_FEATURE_NAMES)), dtype=np.float32)
    obs[:, :, 0] = (pos[:, :, 0] / safe_map_size[:, None]).astype(np.float32)
    obs[:, :, 1] = (pos[:, :, 1] / safe_map_size[:, None]).astype(np.float32)
    obs[:, :, 2] = np.sin(heading).astype(np.float32)
    obs[:, :, 3] = np.cos(heading).astype(np.float32)
    obs[:, :, 4] = alive.astype(np.float32)
    obs[:, :, 5] = printing.astype(np.float32)
    obs[:, :, 6] = score.astype(np.float32)
    obs[:, :, 7] = round_score.astype(np.float32)
    obs[:, :, 8] = (map_size[:, None] / 1000.0).astype(np.float32)

    score_delta = (score - previous_score).astype(np.float32)
    round_score_delta = (round_score - previous_round_score).astype(np.float32)
    event_die_mask = _die_event_mask_from_rows(state, batch_size=batch_size, player_count=player_count)
    if event_die_mask is None:
        died_this_step = previous_alive & ~alive
        died_source = "alive_transition_fallback"
    else:
        died_this_step = event_die_mask
        died_source = "event_rows"
    reward = score_delta + round_score_delta - died_this_step.astype(np.float32)

    done = np.asarray(state["done"], dtype=bool).copy()
    done |= alive.sum(axis=1) <= 1
    truncated = np.asarray(state["overflow"], dtype=bool).copy()
    event_overflow = state.get("event_overflow")
    if event_overflow is not None:
        truncated |= np.asarray(event_overflow, dtype=bool)

    ego_row_id = np.arange(batch_size * player_count, dtype=np.int32).reshape(
        batch_size,
        player_count,
    )
    ego_env_id = np.repeat(np.arange(batch_size, dtype=np.int32)[:, None], player_count, axis=1)
    ego_player_id = np.repeat(
        np.arange(player_count, dtype=np.int16)[None, :],
        batch_size,
        axis=0,
    )
    ego_mask = alive & ~done[:, None] & ~truncated[:, None]
    legal_action_mask = np.repeat(ego_mask[:, :, None], ACTION_COUNT, axis=2)
    terminated_agent = done[:, None] | ~alive
    truncated_agent = np.repeat(truncated[:, None], player_count, axis=1)

    return {
        "obs": obs,
        "reward": reward,
        "done": done,
        "truncated": truncated,
        "terminated_agent": terminated_agent,
        "truncated_agent": truncated_agent,
        "legal_action_mask": legal_action_mask,
        "ego_row_id": ego_row_id,
        "ego_env_id": ego_env_id,
        "ego_player_id": ego_player_id,
        "ego_mask": ego_mask,
        "reward_components": {
            "score_delta": score_delta,
            "round_score_delta": round_score_delta,
            "died_this_step": died_this_step,
            "died_source": died_source,
        },
    }


def print_plain(summary: Mapping[str, Any]) -> None:
    counts = _mapping(summary.get("summary"), "summary")
    timing = _mapping(summary.get("timing_sec"), "timing_sec")
    contract = _mapping(summary.get("packing_contract"), "packing_contract")
    print(f"benchmark={summary['benchmark_id']}")
    print(f"trust_level={summary['trust_level']}")
    print(
        "summary="
        f"passed:{counts['passed']} failed:{counts['failed']} "
        f"unsupported:{counts['unsupported']} "
        f"batch_preflight_failed:{counts['batch_preflight_failed']}"
    )
    print(
        "contract="
        f"obs:{contract['obs_shape']} {contract['obs_dtype']} "
        f"reward:{contract['reward_shape']} {contract['reward_dtype']} "
        f"reward_status:{contract['reward_status']}"
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
            f"batch_state_match={preflight['match']}"
        )
        if not preflight["match"]:
            print(f"batch_preflight_mismatches={preflight['mismatches']}")
            continue
        for batch in _list(group.get("batches"), "group.batches"):
            batch = _mapping(batch, "batch")
            rates = _mapping(batch["rates"], "batch.rates")
            sample = _mapping(batch["sample"], "batch.sample")
            print(
                "batch="
                f"B:{batch['batch_size']} repeats:{batch['repeat']} "
                f"obs_shape:{sample['obs_shape']} "
                f"pack_sec:{batch['timing_sec']['pack_sec']:.6f} "
                f"env_rows_per_sec:{rates['env_rows_per_pack_sec']:.1f} "
                f"ego_rows_per_sec:{rates['ego_rows_per_pack_sec']:.1f} "
                f"live_ego:{sample['live_ego_count']} "
                f"done:{sample['done_count']} "
                f"truncated:{sample['truncated_count']} "
                f"reward_die_source:{sample['reward_died_source']}"
            )
    print("known_fake_or_incomplete=" + "; ".join(summary["known_fake_or_incomplete"]))


def _benchmark_group_packing(
    templates: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    batch = batch_rows._build_batch(templates, batch_size)
    previous_state = batch["initial_state"]
    next_state = vector_compare.copy_array_state(previous_state)
    player_count = _int(batch["prepared_batch"]["player_count"], "prepared_batch.player_count")

    started = time.perf_counter()
    step_counters = batch_rows.step_batched_arrays(next_state, batch["prepared_batch"])
    setup_step_sec = time.perf_counter() - started

    sample_surfaces = pack_debug_obs_reward(
        previous_state,
        next_state,
        player_count=player_count,
    )
    for _ in range(warmup):
        pack_debug_obs_reward(previous_state, next_state, player_count=player_count)

    started = time.perf_counter()
    timed_surfaces = sample_surfaces
    for _ in range(repeat):
        timed_surfaces = pack_debug_obs_reward(
            previous_state,
            next_state,
            player_count=player_count,
        )
    pack_sec = time.perf_counter() - started

    timed_env_rows = batch_size * repeat
    timed_ego_rows = batch_size * player_count * repeat
    return {
        "batch_size": batch_size,
        "repeat": repeat,
        "warmup": warmup,
        "row_source_counts": batch["row_source_counts"],
        "setup_single_batch_step_sec": setup_step_sec,
        "step_counters": step_counters,
        "timing_sec": {"pack_sec": pack_sec},
        "rates": {
            "warmup_env_rows": batch_size * warmup,
            "timed_env_rows": timed_env_rows,
            "timed_ego_rows": timed_ego_rows,
            "env_rows_per_pack_sec": _rate(timed_env_rows, pack_sec),
            "ego_rows_per_pack_sec": _rate(timed_ego_rows, pack_sec),
        },
        "sample": _surface_sample_summary(timed_surfaces),
    }


def _surface_sample_summary(surfaces: Mapping[str, Any]) -> dict[str, Any]:
    obs = np.asarray(surfaces["obs"])
    reward = np.asarray(surfaces["reward"])
    done = np.asarray(surfaces["done"])
    truncated = np.asarray(surfaces["truncated"])
    legal_action_mask = np.asarray(surfaces["legal_action_mask"])
    ego_mask = np.asarray(surfaces["ego_mask"])
    return {
        "obs_shape": list(obs.shape),
        "reward_shape": list(reward.shape),
        "done_shape": list(done.shape),
        "truncated_shape": list(truncated.shape),
        "legal_action_mask_shape": list(legal_action_mask.shape),
        "ego_row_id_shape": list(np.asarray(surfaces["ego_row_id"]).shape),
        "obs_dtype": str(obs.dtype),
        "reward_dtype": str(reward.dtype),
        "bytes_per_pack": _surface_nbytes(surfaces),
        "reward_sum": float(reward.sum()),
        "live_ego_count": int(ego_mask.sum()),
        "done_count": int(done.sum()),
        "truncated_count": int(truncated.sum()),
        "legal_true_count": int(legal_action_mask.sum()),
        "reward_died_source": str(surfaces["reward_components"]["died_source"]),
        "checksum": float(obs.sum(dtype=np.float64) + reward.sum(dtype=np.float64)),
    }


def _die_event_mask_from_rows(
    state: Mapping[str, np.ndarray],
    *,
    batch_size: int,
    player_count: int,
) -> np.ndarray | None:
    required = ("event_mask", "event_type", "event_player")
    if any(name not in state for name in required):
        return None

    event_mask = np.asarray(state["event_mask"], dtype=bool)
    event_type = np.asarray(state["event_type"], dtype=np.int16)
    event_player = np.asarray(state["event_player"], dtype=np.int16)
    if event_mask.shape != event_type.shape or event_mask.shape != event_player.shape:
        raise vector_compare.VectorCompareError("event arrays must have matching [B,L] shapes")
    if event_mask.shape[0] != batch_size:
        raise vector_compare.VectorCompareError("event arrays must use the same B as state.pos")

    died = np.zeros((batch_size, player_count), dtype=bool)
    rows, slots = np.nonzero(event_mask & (event_type == vector_compare.EVENT_DIE))
    if rows.size == 0:
        return died
    players = event_player[rows, slots].astype(np.int64, copy=False)
    valid = (players >= 0) & (players < player_count)
    died[rows[valid], players[valid]] = True
    return died


def _surface_nbytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(_surface_nbytes(child) for child in value.values())
    return 0


def _rate(numerator: int, denominator_sec: float) -> float:
    if denominator_sec <= 0:
        return 0.0
    return numerator / denominator_sec


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
        description="Benchmark debug obs/reward packing on fixture-seeded vector rows."
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
    parser.add_argument("--repeat", type=_positive_int, default=10_000)
    parser.add_argument("--warmup", type=_nonnegative_int, default=500)
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

    summary = benchmark_inputs(
        args.paths,
        body_capacity=args.body_capacity,
        step_index=args.step_index,
        batch_sizes=args.batch_sizes,
        repeat=args.repeat,
        warmup=args.warmup,
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
