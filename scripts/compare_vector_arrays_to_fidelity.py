"""Compare fixture-seeded vector array transitions to the source common trace.

This is a narrow bridge, not the production vector backend. It starts from the
JSON array seed emitted by ``seed_vector_state_from_fixtures.py``, runs one
source-ordered NumPy step for most supported fixtures, runs the documented
two-tick reset/timer trace for the delayed PrintManager start fixture, and
compares the supported state and event fields back to the Python
source-fidelity common trace.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
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

import seed_vector_state_from_fixtures as seed_bridge  # noqa: E402
from curvyzero.env.scenario_schema import load_scenario  # noqa: E402
from curvyzero.env.trace_compare import COMMON_TRACE_SCHEMA, project_common_trace  # noqa: E402
from curvyzero.fidelity.source_runners import (  # noqa: E402
    run_source_body_canary_scenario,
    run_source_borderless_wrap_scenario,
    run_source_normal_wall_scenario,
    run_source_print_manager_scenario,
    run_source_trail_gap_scenario,
)


SCHEMA_VERSION = "curvyzero_vector_fixture_step_compare/v1"
RESET_INFO_SCHEMA_ID = "curvyzero_vector_reset_info/v1"
TERMINAL_TRANSITION_SNAPSHOT_SCHEMA_ID = "curvyzero_vector_terminal_transition_snapshot/v1"
RULES_SCHEMA_ID = "curvyzero_source_fixture_rules/v1"
VECTOR_STATE_SCHEMA_ID = "curvyzero_vector_fixture_state/v1"
BENCHMARK_ID = (
    "fixture_seeded_numpy_body_canary_borderless_wall_print_manager_trail_gap"
    "_multiplayer_wall"
)
SOURCE_FIDELITY_CLAIM = (
    "narrow fixture array transition compared to Python source-body-canary, "
    "source-borderless-wrap, source-normal-wall, and source-print-manager common "
    "traces plus source-trail-gap canaries for the current body-canary batch "
    "plus selected border, promoted no-bonus 3P/4P normal-wall, PrintManager "
    "control/toggle/death-stop/delayed-start, forced trail-gap fixtures, and the "
    "separate natural taped multi-step trail-gap fixture"
)

BODY_KIND_NORMAL = 0
BODY_KIND_IMPORTANT = 1
BODY_KIND_DEATH = 2
DEFAULT_BODY_CAPACITY = 16
DEFAULT_EVENT_CAPACITY = 16
DEFAULT_TIMER_CAPACITY = 4
DEFAULT_RANDOM_TAPE_CAPACITY = 8
LIFECYCLE_RNG_TEMPLATE_ARRAY_NAMES = (
    "lifecycle_rng_call_index",
    "lifecycle_rng_call_site_code",
    "lifecycle_rng_call_avatar",
    "lifecycle_rng_call_value",
    "lifecycle_rng_call_at_ms",
)

EVENT_NONE = 0
EVENT_POSITION = 1
EVENT_POINT = 2
EVENT_DIE = 3
EVENT_SCORE_ROUND = 4
EVENT_SCORE = 5
EVENT_ROUND_END = 6
EVENT_PROPERTY = 7

PROPERTY_NONE = 0
PROPERTY_PRINTING = 1
PRINT_MANAGER_RANDOM_HALF_PRINT_DISTANCE = 39.0
PRINT_MANAGER_RANDOM_HALF_HOLE_DISTANCE = 5.25
PRINT_MANAGER_DELAYED_START_MS = 3000.0

TIMER_KIND_NONE = 0
TIMER_KIND_PRINT_MANAGER_START = 1

TERMINAL_REASON_NONE = 0
TERMINAL_REASON_SURVIVOR_WIN = 1
TERMINAL_REASON_ALL_DEAD_DRAW = 2
TERMINAL_REASON_TIMEOUT_TRUNCATED = 3
TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED = 4
TERMINAL_REASON_BODY_OVERFLOW_TRUNCATED = 5

RESET_SOURCE_MANUAL = 0
RESET_SOURCE_AUTORESET = 1
RESET_SOURCE_FIXTURE = 2
RESET_SOURCE_REPLAY = 3

EVENT_CODE_NAMES = {
    EVENT_NONE: "none",
    EVENT_POSITION: "position",
    EVENT_POINT: "point",
    EVENT_DIE: "die",
    EVENT_SCORE_ROUND: "score:round",
    EVENT_SCORE: "score",
    EVENT_ROUND_END: "round:end",
    EVENT_PROPERTY: "property",
}
PROPERTY_CODE_NAMES = {
    PROPERTY_NONE: "none",
    PROPERTY_PRINTING: "printing",
}
TERMINAL_REASON_CODE_NAMES = {
    TERMINAL_REASON_NONE: "none",
    TERMINAL_REASON_SURVIVOR_WIN: "survivor_win",
    TERMINAL_REASON_ALL_DEAD_DRAW: "all_dead_draw",
    TERMINAL_REASON_TIMEOUT_TRUNCATED: "timeout_truncated",
    TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED: "event_overflow_truncated",
    TERMINAL_REASON_BODY_OVERFLOW_TRUNCATED: "body_overflow_truncated",
}
RESET_SOURCE_CODE_NAMES = {
    RESET_SOURCE_MANUAL: "manual",
    RESET_SOURCE_AUTORESET: "autoreset",
    RESET_SOURCE_FIXTURE: "fixture",
    RESET_SOURCE_REPLAY: "replay",
}
SOURCE_TERMINAL_REASON_CODES = frozenset(
    {
        TERMINAL_REASON_SURVIVOR_WIN,
        TERMINAL_REASON_ALL_DEAD_DRAW,
    }
)
EVENT_SCHEMA = {
    "capacity": DEFAULT_EVENT_CAPACITY,
    "codes": {name: code for code, name in EVENT_CODE_NAMES.items() if name != "none"},
    "property_codes": {
        name: code for code, name in PROPERTY_CODE_NAMES.items() if name != "none"
    },
    "arrays": {
        "event_count": "[B] int16",
        "event_mask": "[B,L] bool",
        "event_type": "[B,L] int16",
        "event_player": "[B,L] int16 primary player or -1",
        "event_other": "[B,L] int16 killer/winner/related player or -1",
        "event_bool": "[B,L] int8; -1 null, 0 false, 1 true",
        "event_value_i": "[B,L,2] int32 score/roundScore or property/value slots",
        "event_value_f": "[B,L,2] float64 x/y payload",
        "event_overflow": "[B] bool",
    },
    "supported_now": [
        "position",
        "point",
        "die",
        "score:round",
        "score",
        "round:end",
        "property",
    ],
    "reserved_not_emitted_by_this_gate": [],
}

SUPPORTED_BODY_OPPONENT_SCENARIOS = frozenset(
    {
        "source_body_opponent_tangent_safe_step",
        "source_body_opponent_overlap_kills_step",
    }
)
SUPPORTED_BODY_OWN_SCENARIOS = frozenset(
    {
        "source_body_own_delta3_safe_step",
        "source_body_own_delta4_kills_step",
    }
)
SUPPORTED_BODY_SAME_FRAME_SCENARIOS = frozenset(
    {
        "source_body_same_frame_point_control_safe_step",
        "source_body_same_frame_point_kills_step",
    }
)
SUPPORTED_BODY_COLLISION_ORDER_SCENARIOS = frozenset(
    {
        "source_collision_death_point_kills_later_player_step",
        "source_collision_head_head_reverse_order_single_death_step",
    }
)
SUPPORTED_BODY_SCENARIOS = (
    SUPPORTED_BODY_OPPONENT_SCENARIOS
    | SUPPORTED_BODY_OWN_SCENARIOS
    | SUPPORTED_BODY_SAME_FRAME_SCENARIOS
)
SUPPORTED_BORDERLESS_SIMPLE_WRAP_SCENARIOS = frozenset({"source_borderless_wrap_step"})
SUPPORTED_BORDERLESS_PRINT_MANAGER_WRAP_SCENARIOS = frozenset(
    {"source_borderless_print_manager_wrap_toggle_step"}
)
SUPPORTED_BORDERLESS_BODY_SKIP_SCENARIOS = frozenset(
    {"source_borderless_wrap_skips_destination_body_then_next_frame_kills"}
)
SUPPORTED_BORDERLESS_WRAP_SCENARIOS = (
    SUPPORTED_BORDERLESS_SIMPLE_WRAP_SCENARIOS
    | SUPPORTED_BORDERLESS_PRINT_MANAGER_WRAP_SCENARIOS
    | SUPPORTED_BORDERLESS_BODY_SKIP_SCENARIOS
)
SUPPORTED_NORMAL_WALL_2P_SCENARIOS = frozenset({"source_normal_wall_death_step"})
SUPPORTED_NORMAL_WALL_3P_SCENARIOS = frozenset(
    {"source_normal_wall_3p_two_die_one_survivor_step"}
)
SUPPORTED_NORMAL_WALL_4P_SCENARIOS = frozenset(
    {
        "source_normal_wall_4p_ordered_deaths_survivor_score",
        "source_normal_wall_4p_two_prior_then_same_frame_terminal_draw",
    }
)
SUPPORTED_NORMAL_WALL_MULTISTEP_SCENARIOS = SUPPORTED_NORMAL_WALL_4P_SCENARIOS
SUPPORTED_NORMAL_WALL_SCENARIOS = (
    SUPPORTED_NORMAL_WALL_2P_SCENARIOS
    | SUPPORTED_NORMAL_WALL_3P_SCENARIOS
    | SUPPORTED_NORMAL_WALL_4P_SCENARIOS
)
SUPPORTED_PRINT_MANAGER_NO_TOGGLE_SCENARIOS = frozenset(
    {"source_print_manager_no_toggle_control_step"}
)
SUPPORTED_PRINT_MANAGER_TOGGLE_SCENARIOS = frozenset(
    {
        "source_print_manager_print_to_hole_step",
        "source_print_manager_hole_to_print_step",
        "source_print_manager_exact_zero_toggle_step",
    }
)
SUPPORTED_PRINT_MANAGER_DEATH_STOP_SCENARIOS = frozenset(
    {
        "source_print_manager_active_stop_on_death_step",
        "source_print_manager_active_hole_stop_on_death_step",
        "source_print_manager_body_collision_stop_on_death_step",
    }
)
SUPPORTED_PRINT_MANAGER_DELAYED_START_SCENARIOS = frozenset(
    {"source_print_manager_delayed_start_timer_step"}
)
SUPPORTED_PRINT_MANAGER_SCENARIOS = (
    SUPPORTED_PRINT_MANAGER_NO_TOGGLE_SCENARIOS
    | SUPPORTED_PRINT_MANAGER_TOGGLE_SCENARIOS
    | SUPPORTED_PRINT_MANAGER_DEATH_STOP_SCENARIOS
    | SUPPORTED_PRINT_MANAGER_DELAYED_START_SCENARIOS
)
SUPPORTED_TRAIL_GAP_NO_TOGGLE_SCENARIOS = frozenset(
    {
        "source_trail_gap_hole_space_safe_step",
        "source_trail_gap_stored_body_still_kills_step",
    }
)
SUPPORTED_TRAIL_GAP_PRINT_TO_HOLE_BOUNDARY_SCENARIOS = frozenset(
    {"source_trail_gap_print_to_hole_boundary_kills_step"}
)
SUPPORTED_TRAIL_GAP_HOLE_TO_PRINT_BOUNDARY_SCENARIOS = frozenset(
    {"source_trail_gap_hole_to_print_boundary_kills_step"}
)
SUPPORTED_TRAIL_GAP_NATURAL_SCENARIOS = frozenset(
    {"source_trail_gap_natural_multistep_hole_crossing"}
)
SUPPORTED_TRAIL_GAP_TOGGLE_SCENARIOS = (
    SUPPORTED_TRAIL_GAP_PRINT_TO_HOLE_BOUNDARY_SCENARIOS
    | SUPPORTED_TRAIL_GAP_HOLE_TO_PRINT_BOUNDARY_SCENARIOS
)
SUPPORTED_TRAIL_GAP_SCENARIOS = (
    SUPPORTED_TRAIL_GAP_NO_TOGGLE_SCENARIOS
    | SUPPORTED_TRAIL_GAP_TOGGLE_SCENARIOS
    | SUPPORTED_TRAIL_GAP_NATURAL_SCENARIOS
)
RUNTIME_SOURCE_VERIFIED_SCENARIOS = (
    SUPPORTED_BODY_OPPONENT_SCENARIOS
    | SUPPORTED_BODY_COLLISION_ORDER_SCENARIOS
    | SUPPORTED_BORDERLESS_WRAP_SCENARIOS
    | SUPPORTED_NORMAL_WALL_SCENARIOS
    | SUPPORTED_TRAIL_GAP_SCENARIOS
)
SUPPORTED_SCENARIOS = (
    SUPPORTED_BODY_SCENARIOS
    | SUPPORTED_BODY_COLLISION_ORDER_SCENARIOS
    | SUPPORTED_BORDERLESS_WRAP_SCENARIOS
    | SUPPORTED_NORMAL_WALL_SCENARIOS
    | SUPPORTED_PRINT_MANAGER_SCENARIOS
    | SUPPORTED_TRAIL_GAP_SCENARIOS
)

SUPPORTED_PLAYER_FIELDS = frozenset(
    {
        "player_id",
        "x",
        "y",
        "angle",
        "alive",
        "score",
        "roundScore",
        "printing",
        "printManager",
        "trailPointCount",
        "lastTrailPoint",
        "bodyNum",
        "bodyCount",
    }
)

GLOBAL_UNSUPPORTED_MECHANICS = [
    {
        "mechanic": "broader event types",
        "status": "partially_supported",
        "reason": (
            "this gate emits fixed rows for position, point, die, score:round, score, "
            "round:end, and the narrow PrintManager property event in the supported "
            "fixtures only; angle events remain outside this comparator slice"
        ),
    },
    {
        "mechanic": "broader wall and borderless cases",
        "status": "not_supported",
        "reason": (
            "source_normal_wall_death_step plus promoted 3P/4P no-bonus normal-wall "
            "fixtures and the simple, destination-body-skip, and PrintManager-toggle "
            "source borderless wrap fixtures are supported; broader wall and "
            "borderless variants remain unsupported"
        ),
    },
    {
        "mechanic": "PrintManager toggles and trail gaps",
        "status": "partially_supported",
        "reason": (
            "the active no-toggle PrintManager distance bookkeeping control and the "
            "print-to-hole, hole-to-print, exact-zero toggle/property, and active "
            "death-stop fixtures are supported; the delayed-start fixture is supported "
            "through its reset/timer/pre-step trace; the forced trail-gap hole-space, "
            "stored-body, print-to-hole boundary, and hole-to-print boundary canaries "
            "are supported; the separate natural multi-step trail-gap hole-crossing "
            "fixture is supported through a row-local random tape; broader natural "
            "gap semantics still need their own gates"
        ),
    },
    {
        "mechanic": "terminal scoring and round lifecycle",
        "status": "partially_supported",
        "reason": (
            "the one-survivor score update needed by source_normal_wall_death_step, "
            "promoted 3P/4P normal-wall survivor/draw score rows, and the "
            "collision-order all-dead draw/survivor rows are supported; broader "
            "round lifecycle remains unsupported"
        ),
    },
    {
        "mechanic": "observation, reward, reset, autoreset, policy, and MCTS arrays",
        "status": "not_supported",
        "reason": "this script only proves a one-step environment-state transition",
    },
]


class VectorCompareError(ValueError):
    """Raised when a fixture cannot be compared by this narrow bridge."""


def compare_inputs(
    paths: Sequence[str | Path],
    *,
    body_capacity: int = DEFAULT_BODY_CAPACITY,
    step_index: int = 0,
    require_verified: bool = True,
) -> dict[str, Any]:
    """Seed and compare each input scenario or batch manifest."""

    if body_capacity < 0:
        raise VectorCompareError("body_capacity must be zero or greater")
    if step_index < 0:
        raise VectorCompareError("step_index must be zero or greater")

    seeded = seed_bridge.seed_inputs(paths, body_capacity=body_capacity)
    fixtures = [
        compare_fixture_seed(
            fixture,
            step_index=step_index,
            require_verified=require_verified,
        )
        for fixture in seeded["fixtures"]
    ]
    passed = sum(result["status"] == "pass" for result in fixtures)
    failed = sum(result["status"] == "fail" for result in fixtures)
    unsupported = sum(result["status"] == "unsupported" for result in fixtures)
    return {
        "schema": SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "source_fidelity_claim": SOURCE_FIDELITY_CLAIM,
        "trust_level": (
            "This is the first fixture-seeded array-step proof. It is not a "
            "production backend and it does not support batched self-play."
        ),
        "seed_schema": seeded["schema"],
        "input_count": seeded["input_count"],
        "fixture_count": len(fixtures),
        "body_capacity": body_capacity,
        "event_schema": EVENT_SCHEMA,
        "step_index": step_index,
        "summary": {
            "passed": passed,
            "failed": failed,
            "unsupported": unsupported,
            "status": "fail" if failed else "pass" if passed and not unsupported else "mixed",
        },
        "fixtures": fixtures,
        "global_unsupported_mechanics": GLOBAL_UNSUPPORTED_MECHANICS,
        "next_speed_gate": (
            "Add broader natural trail-gap variants and broader wall/wrap "
            "state/event semantics before claiming production B>1 self-play batching."
        ),
    }


def compare_fixture(
    path: str | Path,
    *,
    body_capacity: int = DEFAULT_BODY_CAPACITY,
    step_index: int = 0,
    require_verified: bool = True,
) -> dict[str, Any]:
    """Seed and compare one fixture path."""

    fixture = seed_bridge.seed_fixture(path, body_capacity=body_capacity)
    return compare_fixture_seed(
        fixture,
        step_index=step_index,
        require_verified=require_verified,
    )


def compare_fixture_seed(
    fixture: Mapping[str, Any],
    *,
    step_index: int = 0,
    require_verified: bool = True,
) -> dict[str, Any]:
    scenario_id = _string(fixture.get("scenario_id"), "fixture.scenario_id")
    support = fixture_transition_support(
        fixture,
        step_index=step_index,
        require_verified=require_verified,
    )
    base = {
        "scenario_id": scenario_id,
        "path": fixture.get("path"),
        "verification": fixture.get("verification"),
        "supported_transition": support["supported"],
        "covered_mechanics": support["covered_mechanics"],
        "unsupported_mechanics": support["unsupported_mechanics"],
    }
    if not support["supported"]:
        return {
            **base,
            "status": "unsupported",
            "match": None,
            "mismatches": [],
            "compared_fields": [],
            "skipped_fields": [],
        }

    source_trace = source_common_trace_for_fixture(fixture)
    if _uses_full_fixture_trace_compare(scenario_id):
        return _compare_full_fixture_trace(
            base=base,
            fixture=fixture,
            source_trace=source_trace,
        )

    state = array_state_from_seed(fixture)
    prepared_step = prepare_fixture_array_step(fixture, step_index=step_index)
    started = time.perf_counter()
    counters = step_prepared_arrays(state, prepared_step)
    step_sec = time.perf_counter() - started
    actual_trace = project_array_state_to_common_trace(
        fixture,
        state,
        step_index=step_index,
    )
    compare = compare_common_trace_fields(
        actual_trace,
        source_trace,
        fixture.get("comparison", {}),
    )
    return {
        **base,
        "status": "pass" if compare["match"] else "fail",
        "match": compare["match"],
        "mismatches": compare["mismatches"],
        "compared_fields": compare["compared_fields"],
        "skipped_fields": compare["skipped_fields"],
        "step_timing_sec": step_sec,
        "array_counters": counters,
        "array_event_arrays": event_arrays_to_payload(state),
        "array_projection": actual_trace,
        "source_projection": source_trace,
    }


def _compare_full_fixture_trace(
    *,
    base: Mapping[str, Any],
    fixture: Mapping[str, Any],
    source_trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare supported fixtures whose source contract spans the whole fixture trace."""

    state = array_state_from_seed(fixture)
    action_schedule = _list(fixture.get("action_schedule"), "fixture.action_schedule")
    actual_steps: list[dict[str, Any]] = []
    counters_by_step: list[dict[str, int]] = []
    event_arrays_by_step: list[dict[str, Any]] = []

    started = time.perf_counter()
    for index in range(len(action_schedule)):
        counters = step_prepared_arrays(
            state,
            prepare_fixture_array_step(fixture, step_index=index),
        )
        counters_by_step.append({"step_index": index, **counters})
        actual_steps.append(
            project_array_state_to_common_trace(
                fixture,
                state,
                step_index=index,
            )["steps"][0]
        )
        event_arrays_by_step.append({"step_index": index, **event_arrays_to_payload(state)})
    step_sec = time.perf_counter() - started

    actual_trace = {
        "schema": COMMON_TRACE_SCHEMA,
        "scenario_id": _string(fixture.get("scenario_id"), "fixture.scenario_id"),
        "map_size": float(state["map_size"][0]),
        "steps": actual_steps,
    }
    compare = compare_common_trace_fields(
        actual_trace,
        source_trace,
        fixture.get("comparison", {}),
    )
    return {
        **base,
        "status": "pass" if compare["match"] else "fail",
        "match": compare["match"],
        "mismatches": compare["mismatches"],
        "compared_fields": compare["compared_fields"],
        "skipped_fields": compare["skipped_fields"],
        "step_timing_sec": step_sec,
        "array_counters": _aggregate_counters_by_step(counters_by_step),
        "array_counters_by_step": counters_by_step,
        "array_event_arrays": event_arrays_to_payload(state),
        "array_event_arrays_by_step": event_arrays_by_step,
        "array_projection": actual_trace,
        "source_projection": source_trace,
    }


def step_seeded_arrays(
    fixture: Mapping[str, Any],
    *,
    step_index: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Run one source-ordered fixture array tick."""

    state = array_state_from_seed(fixture)
    counters = step_prepared_arrays(
        state,
        prepare_fixture_array_step(fixture, step_index=step_index),
    )
    return state, counters


def fixture_transition_support(
    fixture: Mapping[str, Any],
    *,
    step_index: int = 0,
    require_verified: bool = True,
) -> dict[str, Any]:
    """Return whether this fixture step is inside the narrow supported lane."""

    support = _support_status(fixture, require_verified=require_verified)
    if not support["supported"]:
        return support

    scenario_id = _string(fixture.get("scenario_id"), "fixture.scenario_id")
    if _uses_full_fixture_trace_compare(scenario_id) and step_index != 0:
        if scenario_id in SUPPORTED_PRINT_MANAGER_DELAYED_START_SCENARIOS:
            mechanic = "delayed-start fixture trace"
            reason = (
                "this comparator supports the delayed-start fixture only as the "
                "full two-tick reset/timer/pre-step trace from step_index 0"
            )
        else:
            mechanic = "full fixture trace"
            reason = (
                "this comparator supports this multi-step fixture only as the "
                "full source/common trace from step_index 0"
            )
        return {
            **support,
            "supported": False,
            "unsupported_mechanics": [
                *support["unsupported_mechanics"],
                {
                    "mechanic": mechanic,
                    "status": "not_supported",
                    "reason": reason,
                },
            ],
        }

    action_schedule = _list(fixture.get("action_schedule"), "fixture.action_schedule")
    if step_index < len(action_schedule):
        return support

    return {
        **support,
        "supported": False,
        "unsupported_mechanics": [
            *support["unsupported_mechanics"],
            {
                "mechanic": "multi-step rollout",
                "status": "not_supported",
                "reason": f"fixture has {len(action_schedule)} step(s), requested step {step_index}",
            },
        ],
    }


def array_state_from_seed(fixture: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Materialize mutable NumPy arrays from a fixture seed payload."""

    return _state_from_seed(fixture)


def copy_array_state(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return a mutable deep copy of an array state mapping."""

    return {name: value.copy() for name, value in state.items()}


def reset_array_state(
    target: Mapping[str, np.ndarray],
    source: Mapping[str, np.ndarray],
) -> None:
    """Copy source arrays into an existing same-shaped mutable target state."""

    if target.keys() != source.keys():
        raise VectorCompareError("target and source state keys must match")
    for name, source_array in source.items():
        target_array = target[name]
        if target_array.shape != source_array.shape:
            raise VectorCompareError(f"state array {name!r} shape differs during reset")
        np.copyto(target_array, source_array)


def _bool_row_mask(value: np.ndarray, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.dtype != bool or mask.ndim != 1:
        raise VectorCompareError(f"{name} must be a bool array with shape [B]")
    return mask


def _int32_row_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.int32 or array.ndim != 1:
        raise VectorCompareError(f"{name} must be an int32 array with shape [B]")
    return array


def _int64_row_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.int64 or array.ndim != 1:
        raise VectorCompareError(f"{name} must be an int64 array with shape [B]")
    return array


def _int16_row_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.int16 or array.ndim != 1:
        raise VectorCompareError(f"{name} must be an int16 array with shape [B]")
    return array


def _uint64_row_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint64 or array.ndim != 1:
        raise VectorCompareError(f"{name} must be a uint64 array with shape [B]")
    return array


def terminal_transition_snapshot(
    state: Mapping[str, np.ndarray],
    final_mask: np.ndarray,
    *,
    array_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Copy selected final transition rows before the backing state is reset."""

    mask = _bool_row_mask(final_mask, "final_mask")
    batch_size = mask.shape[0]
    names = _snapshot_array_names(state, array_names)
    for name in names:
        _validate_leading_batch_dimension(state, name, batch_size, "terminal snapshot")

    return {
        "schema": TERMINAL_TRANSITION_SNAPSHOT_SCHEMA_ID,
        "final_mask": mask.copy(),
        "final_rows": np.flatnonzero(mask).astype(np.int32),
        "arrays": {
            name: np.asarray(state[name])[mask, ...].copy()
            for name in names
        },
    }


def reset_array_rows_with_info(
    target: Mapping[str, np.ndarray],
    source: Mapping[str, np.ndarray],
    reset_mask: np.ndarray,
    *,
    reset_episode_id: np.ndarray | int | None = None,
    reset_seed: np.ndarray | int | None = None,
    reset_source: np.ndarray | int = RESET_SOURCE_MANUAL,
    snapshot_array_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Snapshot selected terminal rows, reset them, and return reset metadata arrays."""

    mask = _bool_row_mask(reset_mask, "reset_mask")
    explicit_reset_episode_id = (
        _reset_metadata_array(
            reset_episode_id,
            "reset_episode_id",
            dtype=np.int64,
            mask=mask,
            default=None,
            scalar_default=0,
        )
        if reset_episode_id is not None
        else None
    )
    explicit_reset_seed = (
        _reset_metadata_array(
            reset_seed,
            "reset_seed",
            dtype=np.uint64,
            mask=mask,
            default=None,
            scalar_default=0,
        )
        if reset_seed is not None
        else None
    )
    reset_source_array = _reset_metadata_array(
        reset_source,
        "reset_source",
        dtype=np.int16,
        mask=mask,
        default=target.get("reset_source"),
        scalar_default=RESET_SOURCE_MANUAL,
    )
    if not bool(np.isin(reset_source_array[mask], tuple(RESET_SOURCE_CODE_NAMES)).all()):
        raise VectorCompareError("reset_source values must be known reset source codes")

    snapshot = terminal_transition_snapshot(
        target,
        mask,
        array_names=snapshot_array_names,
    )
    reset_count = reset_array_rows(target, source, mask)

    return {
        "schema": RESET_INFO_SCHEMA_ID,
        "reset_schema_id": RESET_INFO_SCHEMA_ID,
        "rules_schema_id": RULES_SCHEMA_ID,
        "state_schema_id": VECTOR_STATE_SCHEMA_ID,
        "reset_count": reset_count,
        "reset_mask": mask.copy(),
        "reset_rows": np.flatnonzero(mask).astype(np.int32),
        "reset_episode_id": explicit_reset_episode_id
        if explicit_reset_episode_id is not None
        else _reset_metadata_array(
            reset_episode_id,
            "reset_episode_id",
            dtype=np.int64,
            mask=mask,
            default=target.get("episode_id"),
            scalar_default=0,
        ),
        "reset_seed": explicit_reset_seed
        if explicit_reset_seed is not None
        else _reset_metadata_array(
            reset_seed,
            "reset_seed",
            dtype=np.uint64,
            mask=mask,
            default=target.get("reset_seed"),
            scalar_default=0,
        ),
        "reset_source": reset_source_array,
        "terminal_transition_snapshot": snapshot,
    }


def reset_arrays(
    target: Mapping[str, np.ndarray],
    source: Mapping[str, np.ndarray],
    reset_mask: np.ndarray,
    *,
    reset_seed: np.ndarray | int,
    reset_source: np.ndarray | int = RESET_SOURCE_MANUAL,
    snapshot_array_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Reset selected rows and stamp row-local lifecycle metadata.

    This is still a local comparator helper. It proves the order needed by a
    later production reset path: snapshot terminal rows, copy reset template
    rows, then mutate the lifecycle fields for the next episode.
    """

    mask = _bool_row_mask(reset_mask, "reset_mask")
    batch_size = mask.shape[0]
    _validate_reset_lifecycle_state(target, batch_size=batch_size, state_name="target")
    _validate_reset_lifecycle_state(source, batch_size=batch_size, state_name="source")

    previous_episode_id = target["episode_id"].copy()
    if (previous_episode_id < 0).any():
        raise VectorCompareError("episode_id values must be non-negative")
    if bool((previous_episode_id[mask] == np.iinfo(np.int64).max).any()):
        raise VectorCompareError("episode_id values cannot be incremented without overflow")

    reset_episode_id = previous_episode_id.copy()
    reset_episode_id[mask] += 1
    reset_seed_array = _row_reset_metadata_input_array(
        reset_seed,
        "reset_seed",
        dtype=np.uint64,
        mask=mask,
        current=target["reset_seed"],
    )
    reset_source_array = _row_reset_metadata_input_array(
        reset_source,
        "reset_source",
        dtype=np.int16,
        mask=mask,
        current=target["reset_source"],
    )
    if not bool(np.isin(reset_source_array, tuple(RESET_SOURCE_CODE_NAMES)).all()):
        raise VectorCompareError("reset_source values must be known reset source codes")

    info = reset_array_rows_with_info(
        target,
        source,
        mask,
        reset_episode_id=reset_episode_id,
        reset_seed=reset_seed_array,
        reset_source=reset_source_array,
        snapshot_array_names=snapshot_array_names,
    )

    target["episode_id"][mask] = reset_episode_id[mask]
    target["episode_step"][mask] = 0
    target["env_active"][mask] = True
    target["reset_pending"][mask] = False
    target["done"][mask] = False
    target["terminated"][mask] = False
    target["truncated"][mask] = False
    target["terminal_reason"][mask] = TERMINAL_REASON_NONE
    target["reset_seed"][mask] = reset_seed_array[mask]
    target["reset_source"][mask] = reset_source_array[mask]
    _reset_optional_clock_rows(target, mask)
    _reset_event_rows(target, mask)
    if "timer_fired_count" in target:
        _validate_leading_batch_dimension(target, "timer_fired_count", batch_size, "reset")
        target["timer_fired_count"][mask] = 0

    info.update(
        {
            "reset_episode_step": target["episode_step"].copy(),
            "reset_env_active": target["env_active"].copy(),
            "reset_pending": target["reset_pending"].copy(),
            "reset_done": target["done"].copy(),
            "reset_terminated": target["terminated"].copy(),
            "reset_truncated": target["truncated"].copy(),
            "reset_terminal_reason": target["terminal_reason"].copy(),
        }
    )
    return info


def reset_many(
    target: Mapping[str, np.ndarray],
    source: Mapping[str, np.ndarray],
    reset_mask: np.ndarray,
    *,
    reset_seed: np.ndarray | int,
    reset_source: np.ndarray | int = RESET_SOURCE_MANUAL,
    snapshot_array_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Public-ish comparator row reset wrapper, not a final trainer API."""

    return reset_arrays(
        target,
        source,
        reset_mask,
        reset_seed=reset_seed,
        reset_source=reset_source,
        snapshot_array_names=snapshot_array_names,
    )


def reset_array_rows(
    target: Mapping[str, np.ndarray],
    source: Mapping[str, np.ndarray],
    reset_mask: np.ndarray,
) -> int:
    """Copy selected batch rows from source into an existing mutable target state."""

    mask = _bool_row_mask(reset_mask, "reset_mask")
    if target.keys() != source.keys():
        raise VectorCompareError("target and source state keys must match")

    batch_size = mask.shape[0]
    for name, source_array in source.items():
        target_array = target[name]
        if target_array.shape != source_array.shape:
            raise VectorCompareError(f"state array {name!r} shape differs during row reset")
        if source_array.ndim < 1 or source_array.shape[0] != batch_size:
            raise VectorCompareError(
                f"state array {name!r} must have leading reset_mask dimension"
            )

    for name, source_array in source.items():
        target[name][mask, ...] = source_array[mask, ...]
    return int(mask.sum())


def _validate_reset_lifecycle_state(
    state: Mapping[str, np.ndarray],
    *,
    batch_size: int,
    state_name: str,
) -> None:
    for name, validator in (
        ("episode_id", _int64_row_array),
        ("episode_step", _int32_row_array),
        ("env_active", _bool_row_mask),
        ("reset_pending", _bool_row_mask),
        ("done", _bool_row_mask),
        ("terminated", _bool_row_mask),
        ("truncated", _bool_row_mask),
        ("terminal_reason", _int16_row_array),
        ("reset_seed", _uint64_row_array),
        ("reset_source", _int16_row_array),
    ):
        if name not in state:
            raise VectorCompareError(f"{state_name} state is missing {name!r} for reset")
        array = validator(state[name], name)
        if array.shape != (batch_size,):
            raise VectorCompareError(
                f"{state_name} state {name!r} must match reset_mask shape [B]"
            )

    terminal_reason = np.asarray(state["terminal_reason"])
    if not bool(np.isin(terminal_reason, tuple(TERMINAL_REASON_CODE_NAMES)).all()):
        raise VectorCompareError(
            f"{state_name} terminal_reason values must be known terminal reason codes"
        )
    reset_source = np.asarray(state["reset_source"])
    if not bool(np.isin(reset_source, tuple(RESET_SOURCE_CODE_NAMES)).all()):
        raise VectorCompareError(
            f"{state_name} reset_source values must be known reset source codes"
        )


def _row_reset_metadata_input_array(
    value: np.ndarray | int,
    name: str,
    *,
    dtype: Any,
    mask: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    current_array = np.asarray(current)
    if current_array.dtype != np.dtype(dtype) or current_array.shape != mask.shape:
        raise VectorCompareError(f"{name} current value must be a {np.dtype(dtype)} array [B]")

    array = np.asarray(value)
    if array.ndim == 0:
        scalar = _metadata_scalar_value(value, name, dtype)
        result = current_array.copy()
        result[mask] = scalar
        return result
    if array.dtype != np.dtype(dtype) or array.ndim != 1:
        raise VectorCompareError(f"{name} must be a {np.dtype(dtype)} array with shape [B]")
    if array.shape != mask.shape:
        raise VectorCompareError(f"{name} and reset_mask must have matching shape [B]")
    return array.copy()


def _reset_optional_clock_rows(
    state: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> None:
    if "tick" in state:
        tick = _int32_row_array(state["tick"], "tick")
        if tick.shape != mask.shape:
            raise VectorCompareError("tick and reset_mask must have matching shape [B]")
        state["tick"][mask] = 0
    if "elapsed_ms" in state:
        elapsed_ms = np.asarray(state["elapsed_ms"])
        if elapsed_ms.dtype != np.float64 or elapsed_ms.shape != mask.shape:
            raise VectorCompareError("elapsed_ms must be a float64 array with shape [B]")
        state["elapsed_ms"][mask] = 0.0


def _reset_event_rows(
    state: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> None:
    event_names = (
        "event_count",
        "event_mask",
        "event_type",
        "event_player",
        "event_other",
        "event_bool",
        "event_value_i",
        "event_value_f",
        "event_overflow",
        "event_overflow_attempts",
    )
    if not any(name in state for name in event_names):
        return
    for name in event_names:
        if name not in state:
            raise VectorCompareError(f"state is missing {name!r} for event row reset")
        _validate_leading_batch_dimension(state, name, mask.shape[0], "reset")

    state["event_count"][mask] = 0
    state["event_mask"][mask, ...] = False
    state["event_type"][mask, ...] = EVENT_NONE
    state["event_player"][mask, ...] = -1
    state["event_other"][mask, ...] = -1
    state["event_bool"][mask, ...] = -1
    state["event_value_i"][mask, ...] = 0
    state["event_value_f"][mask, ...] = 0.0
    state["event_overflow"][mask] = False
    state["event_overflow_attempts"][mask] = 0


def _snapshot_array_names(
    state: Mapping[str, np.ndarray],
    array_names: Sequence[str] | None,
) -> tuple[str, ...]:
    if array_names is None:
        return tuple(state.keys())
    if isinstance(array_names, str):
        raise VectorCompareError("array_names must be a sequence of state array names")

    names = tuple(array_names)
    for name in names:
        if not isinstance(name, str):
            raise VectorCompareError("array_names must contain only strings")
        if name not in state:
            raise VectorCompareError(f"state array {name!r} is missing for terminal snapshot")
    return names


def _validate_leading_batch_dimension(
    state: Mapping[str, np.ndarray],
    name: str,
    batch_size: int,
    context: str,
) -> None:
    array = np.asarray(state[name])
    if array.ndim < 1 or array.shape[0] != batch_size:
        raise VectorCompareError(
            f"state array {name!r} must have leading {context} mask dimension"
        )


def _reset_metadata_array(
    value: np.ndarray | int | None,
    name: str,
    *,
    dtype: Any,
    mask: np.ndarray,
    default: np.ndarray | None,
    scalar_default: int,
) -> np.ndarray:
    if value is None:
        if default is not None:
            array = np.asarray(default)
            if array.dtype != np.dtype(dtype) or array.ndim != 1:
                raise VectorCompareError(f"{name} default must be a {np.dtype(dtype)} array [B]")
            if array.shape != mask.shape:
                raise VectorCompareError(f"{name} default and reset_mask must match shape [B]")
            return array.copy()
        return np.full(mask.shape, scalar_default, dtype=dtype)

    array = np.asarray(value)
    if array.ndim == 0:
        scalar = _metadata_scalar_value(value, name, dtype)
        result = np.full(mask.shape, scalar_default, dtype=dtype)
        result[mask] = scalar
        return result
    if array.dtype != np.dtype(dtype) or array.ndim != 1:
        raise VectorCompareError(f"{name} must be a {np.dtype(dtype)} array with shape [B]")
    if array.shape != mask.shape:
        raise VectorCompareError(f"{name} and reset_mask must have matching shape [B]")
    return array.copy()


def _metadata_scalar_value(value: np.ndarray | int, name: str, dtype: Any) -> int:
    scalar_array = np.asarray(value)
    if scalar_array.dtype == bool or not np.issubdtype(scalar_array.dtype, np.integer):
        raise VectorCompareError(f"{name} scalar must be an integer")
    scalar = int(scalar_array)
    if np.dtype(dtype) == np.dtype(np.uint64) and scalar < 0:
        raise VectorCompareError(f"{name} scalar must be non-negative")
    return scalar


def horizon_truncation_mask(
    episode_step: np.ndarray,
    horizon_steps: np.ndarray,
) -> np.ndarray:
    """Return rows whose post-step episode counter reaches an active horizon."""

    episode_step_array = _int32_row_array(episode_step, "episode_step")
    horizon_steps_array = _int32_row_array(horizon_steps, "horizon_steps")
    if horizon_steps_array.shape != episode_step_array.shape:
        raise VectorCompareError("episode_step and horizon_steps must have matching shape [B]")
    if (episode_step_array < 0).any():
        raise VectorCompareError("episode_step values must be non-negative")
    if (horizon_steps_array < 0).any():
        raise VectorCompareError("horizon_steps values must be non-negative")

    return (horizon_steps_array > 0) & (episode_step_array >= horizon_steps_array)


def final_transition_mask(
    done: np.ndarray,
    truncated: np.ndarray | None = None,
) -> np.ndarray:
    """Return rows whose just-staged transition is terminal or truncated."""

    done_mask = _bool_row_mask(done, "done")
    if truncated is None:
        return done_mask.copy()

    truncated_mask = _bool_row_mask(truncated, "truncated")
    if truncated_mask.shape != done_mask.shape:
        raise VectorCompareError("done and truncated must have matching shape [B]")
    return done_mask | truncated_mask


def row_lifecycle_arrays(
    source_terminated: np.ndarray,
    source_terminal_reason: np.ndarray,
    episode_step: np.ndarray,
    horizon_steps: np.ndarray,
    *,
    event_overflow: np.ndarray | None = None,
    body_overflow: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build row-local terminal surfaces for a staged post-step transition.

    ``episode_step`` is the post-step episode counter. Source termination takes
    precedence over truncation. For non-source truncations, event overflow is the
    most specific reason, then body overflow, then horizon timeout.
    """

    terminated = _bool_row_mask(source_terminated, "source_terminated")
    source_reason = _int16_row_array(source_terminal_reason, "source_terminal_reason")
    if source_reason.shape != terminated.shape:
        raise VectorCompareError(
            "source_terminated and source_terminal_reason must have matching shape [B]"
        )

    known_reason = np.isin(source_reason, tuple(TERMINAL_REASON_CODE_NAMES))
    if not bool(known_reason.all()):
        raise VectorCompareError(
            "source_terminal_reason values must be known terminal reason codes"
        )
    source_reason_allowed = np.isin(source_reason, tuple(SOURCE_TERMINAL_REASON_CODES))
    if bool((terminated & ~source_reason_allowed).any()):
        raise VectorCompareError(
            "source_terminal_reason must be survivor_win or all_dead_draw "
            "for source_terminated rows"
        )
    if bool(((source_reason != TERMINAL_REASON_NONE) & ~terminated).any()):
        raise VectorCompareError(
            "source_terminal_reason must be none for non-source-terminated rows"
        )

    horizon_truncated = horizon_truncation_mask(episode_step, horizon_steps)
    if horizon_truncated.shape != terminated.shape:
        raise VectorCompareError(
            "source_terminated and episode_step must have matching shape [B]"
        )

    event_truncated = _optional_bool_row_mask(
        event_overflow,
        "event_overflow",
        expected_shape=terminated.shape,
    )
    body_truncated = _optional_bool_row_mask(
        body_overflow,
        "body_overflow",
        expected_shape=terminated.shape,
    )

    truncated = (horizon_truncated | event_truncated | body_truncated) & ~terminated
    done = terminated | truncated

    terminal_reason = np.zeros(terminated.shape, dtype=np.int16)
    terminal_reason[terminated] = source_reason[terminated]
    terminal_reason[truncated & horizon_truncated] = TERMINAL_REASON_TIMEOUT_TRUNCATED
    terminal_reason[truncated & body_truncated] = TERMINAL_REASON_BODY_OVERFLOW_TRUNCATED
    terminal_reason[truncated & event_truncated] = TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED

    return {
        "terminated": terminated.copy(),
        "truncated": truncated,
        "done": done,
        "terminal_reason": terminal_reason,
    }


def _optional_bool_row_mask(
    value: np.ndarray | None,
    name: str,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if value is None:
        return np.zeros(expected_shape, dtype=bool)

    mask = _bool_row_mask(value, name)
    if mask.shape != expected_shape:
        raise VectorCompareError(
            f"{name} and source_terminated must have matching shape [B]"
        )
    return mask


def prepare_fixture_array_step(
    fixture: Mapping[str, Any],
    *,
    step_index: int = 0,
) -> dict[str, Any]:
    """Compile one fixture action row into reusable array-step inputs."""

    profile = _mapping(fixture.get("profile"), "fixture.profile")
    player_count = _int(profile.get("P"), "fixture.profile.P")
    action_schedule = _list(fixture.get("action_schedule"), "fixture.action_schedule")
    if step_index >= len(action_schedule):
        raise VectorCompareError(
            f"fixture has {len(action_schedule)} step(s), requested step {step_index}"
        )
    raw_step = _mapping(action_schedule[step_index], f"fixture.action_schedule[{step_index}]")
    step_ms = _number(raw_step.get("step_ms"), "action_schedule.step_ms")
    moves = np.asarray(_list(raw_step.get("source_moves"), "action_schedule.source_moves"))
    if moves.shape != (player_count,):
        raise VectorCompareError("source_moves must have shape [P]")
    scenario_id = _string(fixture.get("scenario_id"), "fixture.scenario_id")
    return {
        "step_index": step_index,
        "scenario_id": scenario_id,
        "player_count": player_count,
        "step_ms": step_ms,
        "source_moves": moves.astype(np.int8, copy=False),
        "print_manager_mode": _print_manager_mode_for_scenario(scenario_id),
        "timer_advance_ms": _timer_advance_ms_for_fixture_step(
            fixture,
            step_index,
            raw_step=raw_step,
        ),
    }


def step_prepared_arrays(
    state: dict[str, np.ndarray],
    prepared_step: Mapping[str, Any],
) -> dict[str, int]:
    """Run one in-place source-ordered array tick from prepared step inputs."""

    player_count = _int(prepared_step.get("player_count"), "prepared_step.player_count")
    step_ms = _number(prepared_step.get("step_ms"), "prepared_step.step_ms")
    moves = np.asarray(prepared_step.get("source_moves"))
    if moves.shape != (player_count,):
        raise VectorCompareError("prepared_step.source_moves must have shape [P]")

    counters = {
        "movement_updates": 0,
        "events_emitted": 0,
        "event_overflow_attempts": 0,
        "normal_points_inserted": 0,
        "death_points_inserted": 0,
        "body_hits": 0,
        "body_scan_slots": 0,
        "body_candidates": 0,
        "body_overflow_attempts": 0,
        "borderless_wraps": 0,
        "normal_wall_deaths": 0,
        "terminal_score_rows": 0,
        "print_manager_no_toggle_updates": 0,
        "print_manager_toggle_updates": 0,
        "print_manager_toggle_rows_unhandled": 0,
        "print_manager_visual_clears": 0,
        "print_manager_death_stops": 0,
        "print_manager_death_stop_points": 0,
        "print_manager_death_stop_visual_clears": 0,
        "pre_step_timer_advances": 0,
        "pre_step_timer_fires": 0,
        "print_manager_delayed_start_fires": 0,
        "print_manager_delayed_start_points": 0,
        "random_tape_draws": 0,
        "random_tape_exhaustions": 0,
    }

    _ensure_random_tape_arrays(state)
    _reset_event_arrays(state)
    random_tape_draw_count_before = state["random_tape_draw_count"].copy()
    for name, value in _advance_pre_step_timers(state, prepared_step).items():
        counters[name] += value
    print_manager_mode = str(prepared_step.get("print_manager_mode", "none"))
    frame_start_deaths = player_count - state["alive"].sum(axis=1).astype(np.int32)
    death_rows = np.zeros(state["tick"].shape[0], dtype=bool)
    for player in range(player_count - 1, -1, -1):
        live_mask = state["alive"][:, player] & ~state["done"] & ~state["overflow"]
        if not live_mask.any():
            continue

        state["live_body_num"][live_mask, player] = state["body_count"][live_mask, player]
        old_pos = state["pos"][:, player].copy()
        old_heading = state["heading"][:, player].copy()
        angle_delta = state["angular_velocity_per_ms"][:, player] * step_ms * float(moves[player])
        new_heading = old_heading + angle_delta
        state["heading"][:, player] = np.where(live_mask, new_heading, old_heading)
        distance = state["speed"][:, player] * step_ms / 1000.0
        state["prev_pos"][live_mask, player] = old_pos[live_mask]
        state["pos"][:, player, 0] = np.where(
            live_mask,
            old_pos[:, 0] + np.cos(state["heading"][:, player]) * distance,
            state["pos"][:, player, 0],
        )
        state["pos"][:, player, 1] = np.where(
            live_mask,
            old_pos[:, 1] + np.sin(state["heading"][:, player]) * distance,
            state["pos"][:, player, 1],
        )
        counters["movement_updates"] += int(live_mask.sum())
        _emit_position_events(state, player, live_mask)

        cursor_dx = state["pos"][:, player, 0] - state["draw_cursor_pos"][:, player, 0]
        cursor_dy = state["pos"][:, player, 1] - state["draw_cursor_pos"][:, player, 1]
        cursor_dist_sq = cursor_dx * cursor_dx + cursor_dy * cursor_dy
        radius_sq = state["radius"][:, player] * state["radius"][:, player]
        should_draw = (
            live_mask
            & state["printing"][:, player]
            & (~state["has_draw_cursor"][:, player] | (cursor_dist_sq > radius_sq))
        )
        inserted, overflowed = _append_body_points(
            state,
            player=player,
            write_mask=should_draw,
            insert_kind=BODY_KIND_NORMAL,
        )
        counters["normal_points_inserted"] += inserted
        counters["body_overflow_attempts"] += overflowed
        _emit_point_events(state, player, should_draw, important=False)

        wrap_count, wrapped_mask = _apply_borderless_wrap(state, player, live_mask)
        counters["borderless_wraps"] += wrap_count
        _emit_position_events(state, player, wrapped_mask)

        wall_hit_mask = _normal_wall_hit_mask(state, player, live_mask & ~wrapped_mask)
        if wall_hit_mask.any():
            state["alive"][wall_hit_mask, player] = False
            state["death_tick"][wall_hit_mask, player] = state["tick"][wall_hit_mask]
            state["round_score"][wall_hit_mask, player] += frame_start_deaths[wall_hit_mask]
            death_rows |= wall_hit_mask
            counters["normal_wall_deaths"] += int(wall_hit_mask.sum())
            inserted, overflowed = _append_body_points(
                state,
                player=player,
                write_mask=wall_hit_mask,
                insert_kind=BODY_KIND_DEATH,
            )
            counters["death_points_inserted"] += inserted
            counters["body_overflow_attempts"] += overflowed
            _emit_point_events(state, player, wall_hit_mask, important=False)
            if print_manager_mode == "death_stop":
                stop_count, stop_points, visual_clears = _stop_print_manager_on_death(
                    state,
                    player=player,
                    death_mask=wall_hit_mask,
                )
                counters["print_manager_death_stops"] += stop_count
                counters["print_manager_death_stop_points"] += stop_points
                counters["print_manager_death_stop_visual_clears"] += visual_clears
            _emit_die_events(state, player, wall_hit_mask)
            _emit_score_events(
                state,
                player,
                wall_hit_mask,
                event_type=EVENT_SCORE_ROUND,
            )

        collision_live_mask = live_mask & ~wrapped_mask & ~wall_hit_mask
        hit_rows, candidate_count, scanned_slots = _body_collision_rows(
            state,
            player,
            collision_live_mask,
        )
        counters["body_candidates"] += candidate_count
        counters["body_scan_slots"] += scanned_slots
        body_hit_mask = np.zeros(state["tick"].shape[0], dtype=bool)
        if hit_rows.size:
            hit_mask = _rows_to_mask(hit_rows, state["tick"].shape[0])
            body_hit_mask = hit_mask
            hit_owners = _first_hit_body_owner(state, player, hit_rows)
            state["alive"][hit_rows, player] = False
            state["death_tick"][hit_rows, player] = state["tick"][hit_rows]
            death_rows |= hit_mask
            counters["body_hits"] += int(hit_rows.size)
            inserted, overflowed = _append_body_points(
                state,
                player=player,
                write_mask=hit_mask,
                insert_kind=BODY_KIND_DEATH,
            )
            counters["death_points_inserted"] += inserted
            counters["body_overflow_attempts"] += overflowed
            _emit_point_events(state, player, hit_mask, important=False)
            if print_manager_mode == "death_stop":
                stop_count, stop_points, visual_clears = _stop_print_manager_on_death(
                    state,
                    player=player,
                    death_mask=hit_mask,
                )
                counters["print_manager_death_stops"] += stop_count
                counters["print_manager_death_stop_points"] += stop_points
                counters["print_manager_death_stop_visual_clears"] += visual_clears
            _emit_die_events(
                state,
                player,
                hit_mask,
                other_player=hit_owners,
                old=False,
            )
            _emit_score_events(
                state,
                player,
                hit_mask,
                event_type=EVENT_SCORE_ROUND,
            )

        print_manager_live_mask = live_mask & ~wall_hit_mask & ~body_hit_mask
        if print_manager_mode in {"no_toggle_control", "delayed_start"}:
            no_toggle, unhandled_toggle = _update_print_manager_no_toggle(
                state,
                player=player,
                live_mask=print_manager_live_mask,
            )
            counters["print_manager_no_toggle_updates"] += no_toggle
            counters["print_manager_toggle_rows_unhandled"] += unhandled_toggle
        elif print_manager_mode in {"toggle", "natural_toggle"}:
            toggle, no_toggle, visual_clears = _update_print_manager_toggle(
                state,
                player=player,
                live_mask=print_manager_live_mask,
            )
            counters["print_manager_toggle_updates"] += toggle
            if print_manager_mode == "natural_toggle":
                counters["print_manager_no_toggle_updates"] += no_toggle
            else:
                counters["print_manager_toggle_rows_unhandled"] += no_toggle
            counters["print_manager_visual_clears"] += visual_clears

    counters["terminal_score_rows"] += _apply_terminal_score_for_rows(
        state,
        death_rows,
        player_count=player_count,
    )
    counters["random_tape_draws"] = int(
        (state["random_tape_draw_count"] - random_tape_draw_count_before).sum()
    )
    counters["random_tape_exhaustions"] = int(state["random_tape_exhausted"].sum())
    counters["events_emitted"] = int(state["event_count"].sum())
    counters["event_overflow_attempts"] = int(state["event_overflow_attempts"].sum())
    active_rows = ~state["done"]
    state["tick"][active_rows] += 1
    return counters


def source_common_trace_for_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Run the matching source runner and project it to the common trace schema."""

    return _source_common_trace(fixture)


def project_array_state_to_common_trace(
    fixture: Mapping[str, Any],
    state: Mapping[str, np.ndarray],
    *,
    step_index: int,
) -> dict[str, Any]:
    profile = _mapping(fixture.get("profile"), "fixture.profile")
    player_ids = _list(profile.get("player_ids"), "fixture.profile.player_ids")
    action_schedule = _list(fixture.get("action_schedule"), "fixture.action_schedule")
    raw_step = _mapping(action_schedule[step_index], f"fixture.action_schedule[{step_index}]")
    players = []
    for player_index, player_id in enumerate(player_ids):
        last_trail_point: list[float] | None = None
        if bool(state["has_visible_trail_last"][0, player_index]):
            last_trail_point = [
                float(state["visible_trail_last_pos"][0, player_index, 0]),
                float(state["visible_trail_last_pos"][0, player_index, 1]),
            ]
        players.append(
            {
                "player_id": str(player_id),
                "x": float(state["pos"][0, player_index, 0]),
                "y": float(state["pos"][0, player_index, 1]),
                "angle": float(state["heading"][0, player_index]),
                "alive": bool(state["alive"][0, player_index]),
                "score": int(state["score"][0, player_index]),
                "roundScore": int(state["round_score"][0, player_index]),
                "trailPointCount": int(state["visible_trail_count"][0, player_index]),
                "lastTrailPoint": last_trail_point,
                "bodyNum": int(state["live_body_num"][0, player_index]),
                "bodyCount": int(state["body_count"][0, player_index]),
                "printing": bool(state["printing"][0, player_index]),
                "printManager": {
                    "active": bool(state["print_manager_active"][0, player_index]),
                    "distance": float(state["print_manager_distance"][0, player_index]),
                    "lastX": float(state["print_manager_last_pos"][0, player_index, 0]),
                    "lastY": float(state["print_manager_last_pos"][0, player_index, 1]),
                },
            }
        )
    step = {
        "step_index": step_index,
        "step_ms": float(raw_step["step_ms"]),
        "players": players,
        "worldBodyCount": int(state["world_body_count"][0]),
        "events": decode_event_rows_to_common_trace(state, player_ids, row=0),
    }
    return {
        "schema": COMMON_TRACE_SCHEMA,
        "scenario_id": _string(fixture.get("scenario_id"), "fixture.scenario_id"),
        "map_size": float(state["map_size"][0]),
        "steps": [step],
    }


def event_arrays_to_payload(state: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Return compact JSON-friendly event debug arrays."""

    names = {str(code): name for code, name in EVENT_CODE_NAMES.items()}
    return {
        "schema": "curvyzero_vector_event_rows/v1",
        "capacity": int(state["event_type"].shape[1]),
        "event_code_names": names,
        "property_code_names": {str(code): name for code, name in PROPERTY_CODE_NAMES.items()},
        "event_count": state["event_count"].tolist(),
        "event_overflow": state["event_overflow"].tolist(),
        "event_overflow_attempts": state["event_overflow_attempts"].tolist(),
        "event_mask": state["event_mask"].tolist(),
        "event_type": state["event_type"].tolist(),
        "event_player": state["event_player"].tolist(),
        "event_other": state["event_other"].tolist(),
        "event_bool": state["event_bool"].tolist(),
        "event_value_i": state["event_value_i"].tolist(),
        "event_value_f": state["event_value_f"].tolist(),
    }


def decode_event_rows_to_common_trace(
    state: Mapping[str, np.ndarray],
    player_ids: Sequence[Any],
    *,
    row: int,
) -> list[dict[str, object]]:
    """Decode fixed event rows into the existing common-trace event shape."""

    events: list[dict[str, object]] = []
    count = int(state["event_count"][row])
    capacity = state["event_type"].shape[1]
    if count > capacity:
        count = capacity
    for slot in range(count):
        if not bool(state["event_mask"][row, slot]):
            continue
        event_type = int(state["event_type"][row, slot])
        player_index = int(state["event_player"][row, slot])
        other_index = int(state["event_other"][row, slot])
        if event_type == EVENT_POSITION:
            events.append(
                {
                    "event": "position",
                    "player_id": _player_id_for_event(player_ids, player_index),
                    "x": float(state["event_value_f"][row, slot, 0]),
                    "y": float(state["event_value_f"][row, slot, 1]),
                }
            )
            continue
        if event_type == EVENT_POINT:
            events.append(
                {
                    "event": "point",
                    "player_id": _player_id_for_event(player_ids, player_index),
                    "x": float(state["event_value_f"][row, slot, 0]),
                    "y": float(state["event_value_f"][row, slot, 1]),
                    "important": bool(int(state["event_bool"][row, slot]) == 1),
                }
            )
            continue
        if event_type == EVENT_DIE:
            events.append(
                {
                    "event": "die",
                    "player_id": _player_id_for_event(player_ids, player_index),
                    "killer_id": _optional_player_id_for_event(player_ids, other_index),
                    "old": _event_bool_to_optional_bool(int(state["event_bool"][row, slot])),
                }
            )
            continue
        if event_type in {EVENT_SCORE_ROUND, EVENT_SCORE}:
            events.append(
                {
                    "event": EVENT_CODE_NAMES[event_type],
                    "player_id": _player_id_for_event(player_ids, player_index),
                    "score": int(state["event_value_i"][row, slot, 0]),
                    "roundScore": int(state["event_value_i"][row, slot, 1]),
                }
            )
            continue
        if event_type == EVENT_ROUND_END:
            events.append(
                {
                    "event": "round:end",
                    "winner_id": _optional_player_id_for_event(player_ids, other_index),
                }
            )
            continue
        if event_type == EVENT_PROPERTY:
            property_name = PROPERTY_CODE_NAMES.get(
                int(state["event_value_i"][row, slot, 0]),
                "unsupported",
            )
            events.append(
                {
                    "event": "property",
                    "player_id": _player_id_for_event(player_ids, player_index),
                    "property": property_name,
                    "value": _event_bool_to_optional_bool(int(state["event_bool"][row, slot])),
                }
            )
            continue
        events.append({"event": EVENT_CODE_NAMES.get(event_type, f"unsupported:{event_type}")})
    return events


def compare_common_trace_fields(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    comparison: Any,
) -> dict[str, Any]:
    tolerances = _mapping(comparison, "comparison", default={}).get("tolerances", {})
    if not isinstance(tolerances, Mapping):
        tolerances = {}
    compared_fields: list[str] = []
    skipped_fields: list[str] = []
    mismatches: list[dict[str, Any]] = []

    expected_steps = _list(expected.get("steps"), "expected.steps")
    actual_steps = _list(actual.get("steps"), "actual.steps")
    if len(expected_steps) != len(actual_steps):
        mismatches.append(
            {
                "path": "$.steps",
                "expected": len(expected_steps),
                "actual": len(actual_steps),
                "reason": "step counts differ",
            }
        )
        return {
            "match": False,
            "mismatches": mismatches,
            "compared_fields": compared_fields,
            "skipped_fields": skipped_fields,
        }

    for step_index, expected_step in enumerate(expected_steps):
        actual_step = _mapping(actual_steps[step_index], f"actual.steps[{step_index}]")
        expected_step = _mapping(expected_step, f"expected.steps[{step_index}]")
        for key in ("step_index", "step_ms", "worldBodyCount"):
            if key in expected_step:
                path = f"$.steps[{step_index}].{key}"
                _compare_value(
                    actual_step.get(key),
                    expected_step[key],
                    path,
                    _tolerance_for_field(key, tolerances),
                    mismatches,
                )
                compared_fields.append(path)
        if "events" in expected_step:
            event_paths = _compare_event_list(
                actual_step.get("events"),
                expected_step["events"],
                f"$.steps[{step_index}].events",
                tolerances,
                mismatches,
            )
            compared_fields.extend(event_paths)

        expected_players = _list(expected_step.get("players"), f"expected.steps[{step_index}].players")
        actual_players = _list(actual_step.get("players"), f"actual.steps[{step_index}].players")
        if len(expected_players) != len(actual_players):
            mismatches.append(
                {
                    "path": f"$.steps[{step_index}].players",
                    "expected": len(expected_players),
                    "actual": len(actual_players),
                    "reason": "player counts differ",
                }
            )
            continue

        for player_index, expected_player in enumerate(expected_players):
            expected_player = _mapping(
                expected_player,
                f"expected.steps[{step_index}].players[{player_index}]",
            )
            actual_player = _mapping(
                actual_players[player_index],
                f"actual.steps[{step_index}].players[{player_index}]",
            )
            for key, expected_value in expected_player.items():
                path = f"$.steps[{step_index}].players[{player_index}].{key}"
                if key not in SUPPORTED_PLAYER_FIELDS:
                    skipped_fields.append(path)
                    continue
                _compare_value(
                    actual_player.get(key),
                    expected_value,
                    path,
                    _tolerance_for_field(key, tolerances),
                    mismatches,
                )
                compared_fields.append(path)

    return {
        "match": not mismatches,
        "mismatches": mismatches,
        "compared_fields": compared_fields,
        "skipped_fields": skipped_fields,
    }


def _compare_event_list(
    actual: Any,
    expected: Any,
    path: str,
    tolerances: Mapping[str, Any],
    mismatches: list[dict[str, Any]],
) -> list[str]:
    compared_fields: list[str] = []
    if not isinstance(actual, list) or not isinstance(expected, list):
        mismatches.append(
            {
                "path": path,
                "expected": expected,
                "actual": actual,
                "reason": "event list differs",
            }
        )
        return compared_fields
    if len(actual) != len(expected):
        mismatches.append(
            {
                "path": path,
                "expected": len(expected),
                "actual": len(actual),
                "reason": "event counts differ",
            }
        )
        return compared_fields
    for event_index, expected_event in enumerate(expected):
        event_path = f"{path}[{event_index}]"
        expected_event = _mapping(expected_event, f"expected{event_path}")
        actual_event = _mapping(actual[event_index], f"actual{event_path}")
        expected_keys = set(expected_event.keys())
        actual_keys = set(actual_event.keys())
        if actual_keys != expected_keys:
            mismatches.append(
                {
                    "path": event_path,
                    "expected": sorted(expected_keys),
                    "actual": sorted(actual_keys),
                    "reason": "event fields differ",
                }
            )
            continue
        for key, expected_value in expected_event.items():
            field_path = f"{event_path}.{key}"
            _compare_value(
                actual_event.get(key),
                expected_value,
                field_path,
                _tolerance_for_field(key, tolerances),
                mismatches,
            )
            compared_fields.append(field_path)
    return compared_fields


def print_plain(summary: Mapping[str, Any]) -> None:
    counts = _mapping(summary.get("summary"), "summary.summary")
    print(f"benchmark={summary['benchmark_id']}")
    print(f"source_fidelity_claim={summary['source_fidelity_claim']}")
    print(
        "summary="
        f"passed:{counts['passed']} failed:{counts['failed']} unsupported:{counts['unsupported']}"
    )
    for result in _list(summary.get("fixtures"), "summary.fixtures"):
        fixture = _mapping(result, "fixture_result")
        print(
            "fixture="
            f"{fixture['scenario_id']} status={fixture['status']} "
            f"compared_fields={len(fixture['compared_fields'])} "
            f"skipped_fields={len(fixture['skipped_fields'])}"
        )
        if fixture["status"] == "fail":
            first = _mapping(_list(fixture["mismatches"], "mismatches")[0], "first_mismatch")
            print(
                "first_mismatch="
                f"{first['path']} expected={first.get('expected')} actual={first.get('actual')} "
                f"reason={first.get('reason')}"
            )
        if fixture["status"] == "unsupported":
            mechanics = _list(fixture["unsupported_mechanics"], "unsupported_mechanics")
            reason = _mapping(mechanics[0], "unsupported_mechanics[0]") if mechanics else {}
            print(f"unsupported_reason={reason.get('reason', 'not supported by this gate')}")
    print(f"next_speed_gate={summary['next_speed_gate']}")


def _support_status(fixture: Mapping[str, Any], *, require_verified: bool) -> dict[str, Any]:
    scenario_id = _string(fixture.get("scenario_id"), "fixture.scenario_id")
    verification = _mapping(fixture.get("verification"), "fixture.verification", default={})
    unsupported: list[dict[str, Any]] = []
    if scenario_id not in SUPPORTED_SCENARIOS:
        unsupported.append(
            {
                "mechanic": "scenario selection",
                "status": "not_supported",
                "reason": (
                    "this array-step gate supports the 3-player source-body-canary "
                    "fixtures, the two 2-player source_collision_* order fixtures, "
                    "source_borderless_wrap_step, source_borderless_print_manager_"
                    "wrap_toggle_step, source_borderless_wrap_skips_destination_body_"
                    "then_next_frame_kills, source_normal_wall_death_step, the "
                    "promoted 3P/4P source_normal_wall_* no-bonus fixtures, "
                    "source_print_manager_no_toggle_control_step, "
                    "source_print_manager_print_to_hole_step, "
                    "source_print_manager_hole_to_print_step, "
                    "source_print_manager_exact_zero_toggle_step, the active "
                    "PrintManager death-stop fixtures, source_print_manager_"
                    "delayed_start_timer_step, the forced source trail-gap canaries, "
                    "and source_trail_gap_natural_multistep_hole_crossing"
                ),
            }
        )
    if (
        require_verified
        and not bool(verification.get("python_runner_verified"))
        and scenario_id not in RUNTIME_SOURCE_VERIFIED_SCENARIOS
    ):
        unsupported.append(
            {
                "mechanic": "fixture verification",
                "status": "not_supported",
                "reason": (
                    "fixture is not marked python-runner-verified and this scenario "
                    "does not have a narrow comparator source-runner proof"
                ),
            }
        )
    profile = _mapping(fixture.get("profile"), "fixture.profile", default={})
    if int(profile.get("B", 0)) != 1:
        unsupported.append(
            {
                "mechanic": "batched stacking",
                "status": "not_supported",
                "reason": "this comparator expects one B=1 fixture seed at a time",
            }
        )
    expected_players = _expected_player_count(scenario_id)
    if expected_players is not None and int(profile.get("P", 0)) != expected_players:
        unsupported.append(
            {
                "mechanic": "player count",
                "status": "not_supported",
                "reason": f"this fixture gate expects exactly {expected_players} source players",
            }
        )
    return {
        "supported": not unsupported,
        "covered_mechanics": [] if unsupported else _covered_mechanics(scenario_id),
        "unsupported_mechanics": unsupported,
    }


def _source_common_trace(fixture: Mapping[str, Any]) -> dict[str, Any]:
    path = _resolve_fixture_path(_string(fixture.get("path"), "fixture.path"))
    scenario = load_scenario(path)
    scenario_id = _string(fixture.get("scenario_id"), "fixture.scenario_id")
    if scenario_id in SUPPORTED_BORDERLESS_WRAP_SCENARIOS:
        payload = run_source_borderless_wrap_scenario(scenario).to_payload()
    elif scenario_id in SUPPORTED_NORMAL_WALL_SCENARIOS:
        payload = run_source_normal_wall_scenario(scenario).to_payload()
    elif scenario_id in SUPPORTED_PRINT_MANAGER_SCENARIOS:
        payload = run_source_print_manager_scenario(scenario).to_payload()
    elif scenario_id in SUPPORTED_TRAIL_GAP_SCENARIOS:
        payload = run_source_trail_gap_scenario(scenario).to_payload()
    else:
        payload = run_source_body_canary_scenario(scenario).to_payload()
    return project_common_trace(payload)


def _timer_advance_ms_for_fixture_step(
    fixture: Mapping[str, Any],
    step_index: int,
    *,
    raw_step: Mapping[str, Any],
) -> float:
    direct_value = raw_step.get("timer_advance_ms")
    if direct_value is not None:
        result = _number(direct_value, "action_schedule.timer_advance_ms")
        if result < 0.0:
            raise VectorCompareError("action_schedule.timer_advance_ms must be non-negative")
        return result

    scenario_id = _string(fixture.get("scenario_id"), "fixture.scenario_id")
    if scenario_id not in SUPPORTED_PRINT_MANAGER_DELAYED_START_SCENARIOS:
        return 0.0

    path = _resolve_fixture_path(_string(fixture.get("path"), "fixture.path"))
    scenario = load_scenario(path)
    raw_values = scenario.time_policy.get(
        "timer_advance_ms_sequence",
        scenario.time_policy.get("timerAdvanceMsSequence"),
    )
    if raw_values is None:
        return 0.0
    if not isinstance(raw_values, list):
        raise VectorCompareError("time_policy.timer_advance_ms_sequence must be a list")
    if step_index >= len(raw_values):
        raise VectorCompareError(
            "time_policy.timer_advance_ms_sequence is missing the requested step"
        )

    value = raw_values[step_index]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise VectorCompareError("time_policy.timer_advance_ms_sequence values must be numbers")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise VectorCompareError(
            "time_policy.timer_advance_ms_sequence values must be finite and non-negative"
        )
    return result


def _aggregate_counters_by_step(counters_by_step: Sequence[Mapping[str, int]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for counters in counters_by_step:
        for name, value in counters.items():
            if name == "step_index":
                continue
            totals[name] = totals.get(name, 0) + int(value)
    return totals


def _expected_player_count(scenario_id: str) -> int | None:
    if scenario_id in SUPPORTED_BODY_COLLISION_ORDER_SCENARIOS:
        return 2
    if scenario_id in SUPPORTED_BODY_SCENARIOS:
        return 3
    if scenario_id in SUPPORTED_BORDERLESS_SIMPLE_WRAP_SCENARIOS | SUPPORTED_NORMAL_WALL_2P_SCENARIOS:
        return 2
    if scenario_id in SUPPORTED_NORMAL_WALL_3P_SCENARIOS:
        return 3
    if scenario_id in SUPPORTED_NORMAL_WALL_4P_SCENARIOS:
        return 4
    if scenario_id in SUPPORTED_BORDERLESS_PRINT_MANAGER_WRAP_SCENARIOS:
        return 1
    if scenario_id in SUPPORTED_BORDERLESS_BODY_SKIP_SCENARIOS:
        return 3
    if scenario_id in SUPPORTED_PRINT_MANAGER_DEATH_STOP_SCENARIOS:
        return 3
    if scenario_id in SUPPORTED_PRINT_MANAGER_SCENARIOS:
        return 1
    if scenario_id in SUPPORTED_TRAIL_GAP_SCENARIOS:
        return 3
    return None


def _print_manager_mode_for_scenario(scenario_id: str) -> str:
    if scenario_id in SUPPORTED_PRINT_MANAGER_DELAYED_START_SCENARIOS:
        return "delayed_start"
    if scenario_id in SUPPORTED_TRAIL_GAP_NATURAL_SCENARIOS:
        return "natural_toggle"
    if scenario_id in (
        SUPPORTED_PRINT_MANAGER_NO_TOGGLE_SCENARIOS
        | SUPPORTED_TRAIL_GAP_NO_TOGGLE_SCENARIOS
    ):
        return "no_toggle_control"
    if scenario_id in (
        SUPPORTED_PRINT_MANAGER_TOGGLE_SCENARIOS | SUPPORTED_TRAIL_GAP_TOGGLE_SCENARIOS
        | SUPPORTED_BORDERLESS_PRINT_MANAGER_WRAP_SCENARIOS
    ):
        return "toggle"
    if scenario_id in SUPPORTED_PRINT_MANAGER_DEATH_STOP_SCENARIOS:
        return "death_stop"
    return "none"


def _uses_full_fixture_trace_compare(scenario_id: str) -> bool:
    return scenario_id in (
        SUPPORTED_PRINT_MANAGER_DELAYED_START_SCENARIOS
        | SUPPORTED_TRAIL_GAP_NATURAL_SCENARIOS
        | SUPPORTED_BORDERLESS_BODY_SKIP_SCENARIOS
        | SUPPORTED_NORMAL_WALL_MULTISTEP_SCENARIOS
    )


def _covered_mechanics(scenario_id: str) -> list[str]:
    common = [
        "fixture JSON to fixed arrays",
        "reverse player update order",
        "source-move kinematics for one step",
        "common-trace state-field projection",
    ]
    if scenario_id in SUPPORTED_BORDERLESS_SIMPLE_WRAP_SCENARIOS:
        return [
            *common,
            "simple source borderless wrap after movement",
            "source-borderless-wrap runner comparison",
        ]
    if scenario_id in SUPPORTED_BORDERLESS_PRINT_MANAGER_WRAP_SCENARIOS:
        return [
            *common,
            "source borderless wrap skips body lookup before PrintManager.test",
            "pre-wrap normal trail point insertion",
            "post-wrap PrintManager important point insertion",
            "PrintManager printing=false property event emission after wrap",
            "source-borderless-wrap runner comparison",
        ]
    if scenario_id in SUPPORTED_BORDERLESS_BODY_SKIP_SCENARIOS:
        return [
            *common,
            "full multi-step source borderless destination-body-skip trace",
            "wrap frame skips destination body collision",
            "next frame body lookup kills at wrapped position",
            "source-borderless-wrap runner comparison",
        ]
    if scenario_id in SUPPORTED_NORMAL_WALL_SCENARIOS:
        mechanics = [
            *common,
            "normal wall death after movement",
            "death point body insertion for wall hits",
            "source-normal-wall runner comparison",
        ]
        if scenario_id in SUPPORTED_NORMAL_WALL_2P_SCENARIOS:
            return [
                *mechanics,
                "one-survivor terminal score update",
            ]
        if scenario_id in SUPPORTED_NORMAL_WALL_3P_SCENARIOS:
            return [
                *mechanics,
                "three-player same-frame wall deaths in reverse source order",
                "three-player one-survivor terminal score update",
            ]
        if scenario_id == "source_normal_wall_4p_ordered_deaths_survivor_score":
            return [
                *mechanics,
                "full multi-step four-player ordered wall death trace",
                "frame-start death score timing",
                "four-player one-survivor terminal score update",
            ]
        return [
            *mechanics,
            "full multi-step four-player terminal draw trace",
            "same-frame terminal draw score order",
            "no-survivor round:end winner null",
        ]
    if scenario_id in SUPPORTED_PRINT_MANAGER_NO_TOGGLE_SCENARIOS:
        return [
            *common,
            "active PrintManager distance bookkeeping without toggle",
            "PrintManager last position update after collision checks",
            "no property event emitted for no-toggle control",
            "source-print-manager runner comparison",
        ]
    if scenario_id in SUPPORTED_PRINT_MANAGER_DELAYED_START_SCENARIOS:
        return [
            *common,
            "reset-scheduled PrintManager start timer",
            "pre-step timer advancement before movement events",
            "two captured delayed-start ticks compared from reset",
            "PrintManager important start point insertion",
            "PrintManager printing property event emission",
            "fixed source Math.random=0.5 start-distance draw",
            "source-print-manager runner comparison",
        ]
    if scenario_id in SUPPORTED_PRINT_MANAGER_TOGGLE_SCENARIOS:
        mechanics = [
            *common,
            "active PrintManager distance bookkeeping through one toggle",
            "PrintManager important boundary point insertion",
            "PrintManager printing property event emission",
            "fixed source Math.random=0.5 next-distance draw",
            "source-print-manager runner comparison",
        ]
        if scenario_id == "source_print_manager_print_to_hole_step":
            return [
                *mechanics,
                "print-to-hole visible trail and draw cursor clear",
                "property printing=false event emission",
            ]
        if scenario_id == "source_print_manager_exact_zero_toggle_step":
            return [
                *mechanics,
                "exact-zero distance threshold toggles at distance <= 0",
            ]
        return [
            *mechanics,
            "hole-to-print visible trail boundary retention",
            "property printing=true event emission",
        ]
    if scenario_id in SUPPORTED_PRINT_MANAGER_DEATH_STOP_SCENARIOS:
        mechanics = [
            *common,
            "active PrintManager stop on death before die event",
            "PrintManager state clear after death",
            "property printing=false event emission",
            "source-print-manager runner comparison",
        ]
        if scenario_id == "source_print_manager_active_hole_stop_on_death_step":
            return [
                *mechanics,
                "already-hole death stop emits no important stop point",
                "death trail point remains visible after already-hole stop",
            ]
        if scenario_id == "source_print_manager_body_collision_stop_on_death_step":
            return [
                *mechanics,
                "seeded body-collision death before PrintManager test",
                "body-collision death stop important point before die",
            ]
        return [
            *mechanics,
            "active printing wall death stop important point before die",
            "death-stop visible trail and draw cursor clear",
        ]
    if scenario_id in SUPPORTED_TRAIL_GAP_SCENARIOS:
        mechanics = [
            *common,
            "source-trail-gap runner comparison",
            "forced trail-gap three-player update order",
            "visual trail gap state stays separate from world bodies",
        ]
        if scenario_id in SUPPORTED_TRAIL_GAP_NATURAL_SCENARIOS:
            return [
                *common,
                "source-trail-gap runner comparison",
                "full multi-step natural trail-gap trace",
                "row-local Math.random tape for PrintManager distances",
                "natural PrintManager hole crossing remains body-free for the crossing player",
                "visual trail gap state stays separate from world bodies",
            ]
        if scenario_id in SUPPORTED_TRAIL_GAP_NO_TOGGLE_SCENARIOS:
            mechanics = [
                *mechanics,
                "active PrintManager distance bookkeeping inside a hole",
                "no normal point emitted while printing is false",
            ]
            if scenario_id == "source_trail_gap_stored_body_still_kills_step":
                return [
                    *mechanics,
                    "stored body in visual hole still kills later player",
                    "body-collision death event from stored gap body",
                ]
            return [
                *mechanics,
                "forced hole-space endpoint remains safe without stored body",
            ]
        if scenario_id in SUPPORTED_TRAIL_GAP_PRINT_TO_HOLE_BOUNDARY_SCENARIOS:
            return [
                *mechanics,
                "print-to-hole boundary important point inserts collision body",
                "print-to-hole visual trail and draw cursor clear",
                "same-frame boundary body kills later player",
                "property printing=false event emission",
            ]
        return [
            *mechanics,
            "hole-to-print boundary important point inserts collision body",
            "hole-to-print visible trail boundary retention",
            "same-frame boundary body kills later player",
            "property printing=true event emission",
        ]
    body_common = [
        *common,
        "normal point body insertion before collision",
        "strict body overlap",
        "source-body-canary runner comparison",
    ]
    if scenario_id in SUPPORTED_BODY_OPPONENT_SCENARIOS:
        mechanics = [
            *body_common,
            "opponent seeded-body collision scan",
        ]
        if scenario_id == "source_body_opponent_tangent_safe_step":
            return [*mechanics, "strict tangent non-overlap remains safe"]
        return [
            *mechanics,
            "opponent strict-overlap body hit",
            "death point body insertion for one nonterminal body hit",
        ]
    if scenario_id in SUPPORTED_BODY_OWN_SCENARIOS:
        return [
            *body_common,
            "own-body latency mask",
            "death point body insertion for one nonterminal body hit",
        ]
    if scenario_id in SUPPORTED_BODY_COLLISION_ORDER_SCENARIOS:
        mechanics = [
            *common,
            "source-body-canary runner comparison",
            "two-player collision-order source update",
            "same-frame body insertion visible to later players",
        ]
        if scenario_id == "source_collision_death_point_kills_later_player_step":
            return [
                *mechanics,
                "death point side effect kills later player",
                "all-dead terminal draw scoring",
            ]
        return [
            *mechanics,
            "head-head endpoint asymmetry via emitted point body",
            "one-survivor terminal score update",
        ]
    return [
        *body_common,
        "same-frame point materialization before later collision checks",
        "death point body insertion for one nonterminal body hit",
    ]


def _state_from_seed(fixture: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays = _mapping(fixture.get("arrays"), "fixture.arrays")
    state = {
        name: _array_from_payload(payload, name)
        for name, payload in arrays.items()
    }
    _ensure_random_tape_arrays(state)
    state.update(_empty_event_arrays(int(state["tick"].shape[0]), DEFAULT_EVENT_CAPACITY))
    state.update(_empty_timer_arrays(int(state["tick"].shape[0]), DEFAULT_TIMER_CAPACITY))
    scenario_id = _string(fixture.get("scenario_id"), "fixture.scenario_id")
    if scenario_id in SUPPORTED_PRINT_MANAGER_DELAYED_START_SCENARIOS:
        profile = _mapping(fixture.get("profile"), "fixture.profile")
        _seed_print_manager_start_timers(
            state,
            player_count=_int(profile.get("P"), "fixture.profile.P"),
        )
    return state


def _ensure_random_tape_arrays(state: dict[str, np.ndarray]) -> None:
    batch_size = int(state["tick"].shape[0])
    if "random_tape_values" not in state:
        state["random_tape_values"] = np.zeros(
            (batch_size, DEFAULT_RANDOM_TAPE_CAPACITY),
            dtype=np.float64,
        )
    if "random_tape_length" not in state:
        state["random_tape_length"] = np.zeros(batch_size, dtype=np.int32)
    if "random_tape_cursor" not in state:
        state["random_tape_cursor"] = np.zeros(batch_size, dtype=np.int32)
    if "random_tape_exhausted" not in state:
        state["random_tape_exhausted"] = np.zeros(batch_size, dtype=bool)
    if "random_tape_draw_count" not in state:
        state["random_tape_draw_count"] = np.zeros(batch_size, dtype=np.int32)


def _array_from_payload(payload: Any, name: str) -> np.ndarray:
    data = _mapping(payload, f"arrays.{name}")
    dtype = _numpy_dtype(_string(data.get("dtype"), f"arrays.{name}.dtype"))
    return np.asarray(data.get("values"), dtype=dtype)


def _empty_event_arrays(batch_size: int, capacity: int) -> dict[str, np.ndarray]:
    return {
        "event_count": np.zeros(batch_size, dtype=np.int16),
        "event_mask": np.zeros((batch_size, capacity), dtype=bool),
        "event_type": np.zeros((batch_size, capacity), dtype=np.int16),
        "event_player": np.full((batch_size, capacity), -1, dtype=np.int16),
        "event_other": np.full((batch_size, capacity), -1, dtype=np.int16),
        "event_bool": np.full((batch_size, capacity), -1, dtype=np.int8),
        "event_value_i": np.zeros((batch_size, capacity, 2), dtype=np.int32),
        "event_value_f": np.zeros((batch_size, capacity, 2), dtype=np.float64),
        "event_overflow": np.zeros(batch_size, dtype=bool),
        "event_overflow_attempts": np.zeros(batch_size, dtype=np.int32),
    }


def _empty_timer_arrays(batch_size: int, capacity: int) -> dict[str, np.ndarray]:
    return {
        "timer_active": np.zeros((batch_size, capacity), dtype=bool),
        "timer_remaining_ms": np.zeros((batch_size, capacity), dtype=np.float64),
        "timer_kind": np.zeros((batch_size, capacity), dtype=np.int16),
        "timer_player": np.full((batch_size, capacity), -1, dtype=np.int16),
        "timer_seq": np.zeros((batch_size, capacity), dtype=np.int32),
        "timer_overflow": np.zeros(batch_size, dtype=bool),
    }


def _seed_print_manager_start_timers(
    state: dict[str, np.ndarray],
    *,
    player_count: int,
) -> None:
    capacity = state["timer_active"].shape[1]
    if player_count > capacity:
        state["timer_overflow"][:] = True
        state["overflow"][:] = True
        return

    for row in range(state["timer_active"].shape[0]):
        for seq, player in enumerate(range(player_count - 1, -1, -1)):
            state["timer_active"][row, seq] = True
            state["timer_remaining_ms"][row, seq] = PRINT_MANAGER_DELAYED_START_MS
            state["timer_kind"][row, seq] = TIMER_KIND_PRINT_MANAGER_START
            state["timer_player"][row, seq] = player
            state["timer_seq"][row, seq] = seq


def _reset_event_arrays(state: dict[str, np.ndarray]) -> None:
    state["event_count"].fill(0)
    state["event_mask"].fill(False)
    state["event_type"].fill(EVENT_NONE)
    state["event_player"].fill(-1)
    state["event_other"].fill(-1)
    state["event_bool"].fill(-1)
    state["event_value_i"].fill(0)
    state["event_value_f"].fill(0.0)
    state["event_overflow"].fill(False)
    state["event_overflow_attempts"].fill(0)


def _advance_pre_step_timers(
    state: dict[str, np.ndarray],
    prepared_step: Mapping[str, Any],
) -> dict[str, int]:
    advance_ms = _number(
        prepared_step.get("timer_advance_ms", 0.0),
        "prepared_step.timer_advance_ms",
    )
    if advance_ms < 0.0:
        raise VectorCompareError("prepared_step.timer_advance_ms must be non-negative")

    counters = {
        "pre_step_timer_advances": 0,
        "pre_step_timer_fires": 0,
        "print_manager_delayed_start_fires": 0,
        "print_manager_delayed_start_points": 0,
        "body_overflow_attempts": 0,
    }
    if "timer_active" not in state or not state["timer_active"].any():
        return counters

    active_rows = np.flatnonzero(~state["done"] & ~state["overflow"])
    for row in active_rows:
        row_int = int(row)
        active_slots = np.flatnonzero(state["timer_active"][row_int])
        if active_slots.size == 0:
            continue

        counters["pre_step_timer_advances"] += 1
        state["timer_remaining_ms"][row_int, active_slots] -= advance_ms
        due_slots = [
            int(slot)
            for slot in active_slots
            if float(state["timer_remaining_ms"][row_int, int(slot)]) <= 0.0
        ]
        due_slots.sort(key=lambda slot: int(state["timer_seq"][row_int, slot]))

        for slot in due_slots:
            kind = int(state["timer_kind"][row_int, slot])
            if kind == TIMER_KIND_PRINT_MANAGER_START:
                player = int(state["timer_player"][row_int, slot])
                fired, points, overflowed = _fire_print_manager_start_timer(
                    state,
                    row=row_int,
                    player=player,
                )
                counters["pre_step_timer_fires"] += fired
                counters["print_manager_delayed_start_fires"] += fired
                counters["print_manager_delayed_start_points"] += points
                counters["body_overflow_attempts"] += overflowed
            _clear_timer_slot(state, row=row_int, slot=slot)

    return counters


def _fire_print_manager_start_timer(
    state: dict[str, np.ndarray],
    *,
    row: int,
    player: int,
) -> tuple[int, int, int]:
    if player < 0 or player >= state["print_manager_active"].shape[1]:
        raise VectorCompareError(f"timer references unknown player index {player}")
    if bool(state["print_manager_active"][row, player]):
        return 0, 0, 0

    state["print_manager_active"][row, player] = True
    state["print_manager_last_pos"][row, player] = state["pos"][row, player]

    row_mask = np.zeros(state["tick"].shape[0], dtype=bool)
    row_mask[row] = True
    points = 0
    overflowed = 0
    if not bool(state["printing"][row, player]):
        state["printing"][row, player] = True
        points, overflowed = _append_body_points(
            state,
            player=player,
            write_mask=row_mask,
            insert_kind=BODY_KIND_IMPORTANT,
        )
        _emit_point_events(state, player, row_mask, important=True)

    _emit_printing_property_events(state, player, row_mask)
    state["print_manager_distance"][row, player] = _next_print_manager_random_distance(
        state,
        row=row,
        printing=bool(state["printing"][row, player]),
    )
    return 1, points, overflowed


def _clear_timer_slot(
    state: dict[str, np.ndarray],
    *,
    row: int,
    slot: int,
) -> None:
    state["timer_active"][row, slot] = False
    state["timer_remaining_ms"][row, slot] = 0.0
    state["timer_kind"][row, slot] = TIMER_KIND_NONE
    state["timer_player"][row, slot] = -1
    state["timer_seq"][row, slot] = 0


def _emit_position_events(
    state: dict[str, np.ndarray],
    player: int,
    mask: np.ndarray,
) -> None:
    rows = np.flatnonzero(mask)
    for row in rows:
        _emit_event_row(
            state,
            row=int(row),
            event_type=EVENT_POSITION,
            player=player,
            float_values=state["pos"][row, player],
        )


def _emit_point_events(
    state: dict[str, np.ndarray],
    player: int,
    mask: np.ndarray,
    *,
    important: bool,
) -> None:
    rows = np.flatnonzero(mask)
    bool_value = 1 if important else 0
    for row in rows:
        _emit_event_row(
            state,
            row=int(row),
            event_type=EVENT_POINT,
            player=player,
            bool_value=bool_value,
            float_values=state["pos"][row, player],
        )


def _emit_printing_property_events(
    state: dict[str, np.ndarray],
    player: int,
    mask: np.ndarray,
) -> None:
    rows = np.flatnonzero(mask)
    for row in rows:
        _emit_event_row(
            state,
            row=int(row),
            event_type=EVENT_PROPERTY,
            player=player,
            bool_value=1 if bool(state["printing"][row, player]) else 0,
            int_values=(PROPERTY_PRINTING, 0),
        )


def _emit_die_events(
    state: dict[str, np.ndarray],
    player: int,
    mask: np.ndarray,
    *,
    other_player: np.ndarray | None = None,
    old: bool | None = None,
) -> None:
    rows = np.flatnonzero(mask)
    bool_value = -1 if old is None else 1 if old else 0
    for row in rows:
        other = -1 if other_player is None else int(other_player[row])
        _emit_event_row(
            state,
            row=int(row),
            event_type=EVENT_DIE,
            player=player,
            other=other,
            bool_value=bool_value,
        )


def _emit_score_events(
    state: dict[str, np.ndarray],
    player: int,
    mask: np.ndarray,
    *,
    event_type: int,
) -> None:
    rows = np.flatnonzero(mask)
    for row in rows:
        _emit_score_row(state, row=int(row), player=player, event_type=event_type)


def _emit_score_row(
    state: dict[str, np.ndarray],
    *,
    row: int,
    player: int,
    event_type: int,
) -> None:
    _emit_event_row(
        state,
        row=row,
        event_type=event_type,
        player=player,
        int_values=(int(state["score"][row, player]), int(state["round_score"][row, player])),
    )


def _emit_round_end_row(
    state: dict[str, np.ndarray],
    *,
    row: int,
    winner: int,
) -> None:
    _emit_event_row(
        state,
        row=row,
        event_type=EVENT_ROUND_END,
        player=-1,
        other=winner,
    )


def _emit_event_row(
    state: dict[str, np.ndarray],
    *,
    row: int,
    event_type: int,
    player: int = -1,
    other: int = -1,
    bool_value: int = -1,
    int_values: tuple[int, int] | None = None,
    float_values: Sequence[float] | np.ndarray | None = None,
) -> None:
    cursor = int(state["event_count"][row])
    capacity = state["event_type"].shape[1]
    if cursor >= capacity:
        state["event_overflow"][row] = True
        state["event_overflow_attempts"][row] += 1
        return

    state["event_count"][row] += 1
    state["event_mask"][row, cursor] = True
    state["event_type"][row, cursor] = event_type
    state["event_player"][row, cursor] = player
    state["event_other"][row, cursor] = other
    state["event_bool"][row, cursor] = bool_value
    if int_values is not None:
        state["event_value_i"][row, cursor, 0] = int_values[0]
        state["event_value_i"][row, cursor, 1] = int_values[1]
    if float_values is not None:
        state["event_value_f"][row, cursor, 0] = float(float_values[0])
        state["event_value_f"][row, cursor, 1] = float(float_values[1])


def _first_hit_body_owner(
    state: Mapping[str, np.ndarray],
    player: int,
    hit_rows: np.ndarray,
) -> np.ndarray:
    owners = np.full(state["tick"].shape[0], -1, dtype=np.int16)
    capacity = state["body_active"].shape[1]
    for row in hit_rows:
        row_int = int(row)
        for slot in range(capacity):
            if not bool(state["body_active"][row_int, slot]):
                continue
            body_owner = int(state["body_owner"][row_int, slot])
            own_body = body_owner == player
            own_delta = int(state["live_body_num"][row_int, player]) - int(
                state["body_num"][row_int, slot]
            )
            if own_body and own_delta <= int(state["trail_latency"][row_int, player]):
                continue
            dx = state["body_pos"][row_int, slot, 0] - state["pos"][row_int, player, 0]
            dy = state["body_pos"][row_int, slot, 1] - state["pos"][row_int, player, 1]
            radius = state["radius"][row_int, player] + state["body_radius"][row_int, slot]
            if dx * dx + dy * dy < radius * radius:
                owners[row_int] = body_owner
                break
    return owners


def _append_body_points(
    state: dict[str, np.ndarray],
    *,
    player: int,
    write_mask: np.ndarray,
    insert_kind: int,
) -> tuple[int, int]:
    rows = np.flatnonzero(write_mask)
    if rows.size == 0:
        return 0, 0

    inserted = 0
    overflowed = 0
    capacity = state["body_active"].shape[1]
    for row in rows:
        cursor = int(state["body_write_cursor"][row])
        if cursor >= capacity:
            state["body_overflow"][row] = True
            state["overflow"][row] = True
            overflowed += 1
            continue

        body_num = int(state["body_count"][row, player])
        state["body_active"][row, cursor] = True
        state["body_pos"][row, cursor] = state["pos"][row, player]
        state["body_radius"][row, cursor] = state["radius"][row, player]
        state["body_owner"][row, cursor] = player
        state["body_num"][row, cursor] = body_num
        state["body_insert_tick"][row, cursor] = state["tick"][row]
        state["body_insert_kind"][row, cursor] = insert_kind
        state["body_write_cursor"][row] += 1
        state["world_body_count"][row] += 1
        state["body_count"][row, player] += 1
        state["visible_trail_count"][row, player] += 1
        state["has_visible_trail_last"][row, player] = True
        state["visible_trail_last_pos"][row, player] = state["pos"][row, player]
        state["has_draw_cursor"][row, player] = True
        state["draw_cursor_pos"][row, player] = state["pos"][row, player]
        inserted += 1

    return inserted, overflowed


def _apply_borderless_wrap(
    state: dict[str, np.ndarray],
    player: int,
    live_mask: np.ndarray,
) -> tuple[int, np.ndarray]:
    active = live_mask & state["borderless"]
    if not active.any():
        return 0, np.zeros(state["tick"].shape[0], dtype=bool)

    x = state["pos"][:, player, 0]
    y = state["pos"][:, player, 1]
    map_size = state["map_size"]
    x_low = active & (x < 0)
    x_high = active & (x > map_size)
    x_wrapped = x_low | x_high
    y_low = active & ~x_wrapped & (y < 0)
    y_high = active & ~x_wrapped & (y > map_size)
    wrapped = x_wrapped | y_low | y_high

    state["pos"][x_low, player, 0] = map_size[x_low]
    state["pos"][x_high, player, 0] = 0.0
    state["pos"][y_low, player, 1] = map_size[y_low]
    state["pos"][y_high, player, 1] = 0.0
    state["live_body_num"][wrapped, player] = state["body_count"][wrapped, player]
    return int(wrapped.sum()), wrapped


def _normal_wall_hit_mask(
    state: Mapping[str, np.ndarray],
    player: int,
    live_mask: np.ndarray,
) -> np.ndarray:
    active = live_mask & ~state["borderless"]
    if not active.any():
        return np.zeros(state["tick"].shape[0], dtype=bool)

    x = state["pos"][:, player, 0]
    y = state["pos"][:, player, 1]
    radius = state["radius"][:, player]
    map_size = state["map_size"]
    hit = (
        (x - radius < 0.0)
        | (x + radius > map_size)
        | (y - radius < 0.0)
        | (y + radius > map_size)
    )
    return active & hit


def _apply_terminal_score_for_rows(
    state: dict[str, np.ndarray],
    death_rows: np.ndarray,
    *,
    player_count: int,
) -> int:
    terminal_rows = death_rows & (state["alive"].sum(axis=1) <= 1)
    rows = np.flatnonzero(terminal_rows)
    for row in rows:
        live_players = np.flatnonzero(state["alive"][row])
        if live_players.size == 1:
            winner = int(live_players[0])
            state["round_score"][row, winner] += max(player_count - 1, 1)
            _emit_score_row(
                state,
                row=row,
                player=winner,
                event_type=EVENT_SCORE_ROUND,
            )
        state["score"][row] += state["round_score"][row]
        for player in range(player_count - 1, -1, -1):
            _emit_score_row(
                state,
                row=row,
                player=player,
                event_type=EVENT_SCORE,
            )
        state["round_score"][row] = 0
        winner_value = int(live_players[0]) if live_players.size == 1 else -1
        _emit_round_end_row(state, row=row, winner=winner_value)
    return int(rows.size)


def _body_collision_rows(
    state: Mapping[str, np.ndarray],
    player: int,
    live_mask: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    capacity = state["body_active"].shape[1]
    if capacity == 0:
        return np.asarray([], dtype=np.int64), 0, 0

    radius = state["radius"][:, player][:, None]
    dx = state["body_pos"][:, :, 0] - state["pos"][:, player, 0][:, None]
    dy = state["body_pos"][:, :, 1] - state["pos"][:, player, 1][:, None]
    dist_sq = dx * dx + dy * dy
    hit_radius_sq = (radius + state["body_radius"]) ** 2
    own_body = state["body_owner"] == player
    own_delta = state["live_body_num"][:, player][:, None] - state["body_num"]
    own_too_young = own_body & (own_delta <= state["trail_latency"][:, player][:, None])
    candidate = state["body_active"] & ~own_too_young
    hit_mask = candidate & (dist_sq < hit_radius_sq)
    hit_rows = np.flatnonzero(live_mask & hit_mask.any(axis=1))
    scanned_slots = int(live_mask.sum()) * capacity
    return hit_rows, int(candidate.sum()), scanned_slots


def _update_print_manager_no_toggle(
    state: dict[str, np.ndarray],
    *,
    player: int,
    live_mask: np.ndarray,
) -> tuple[int, int]:
    active = live_mask & state["print_manager_active"][:, player]
    if not active.any():
        return 0, 0

    dx = state["print_manager_last_pos"][:, player, 0] - state["pos"][:, player, 0]
    dy = state["print_manager_last_pos"][:, player, 1] - state["pos"][:, player, 1]
    next_distance = state["print_manager_distance"][:, player] - np.hypot(dx, dy)
    no_toggle = active & (next_distance > 0.0)
    unhandled_toggle = active & ~no_toggle

    state["print_manager_distance"][no_toggle, player] = next_distance[no_toggle]
    state["print_manager_last_pos"][no_toggle, player] = state["pos"][no_toggle, player]

    return int(no_toggle.sum()), int(unhandled_toggle.sum())


def _update_print_manager_toggle(
    state: dict[str, np.ndarray],
    *,
    player: int,
    live_mask: np.ndarray,
) -> tuple[int, int, int]:
    active = live_mask & state["print_manager_active"][:, player]
    if not active.any():
        return 0, 0, 0

    dx = state["print_manager_last_pos"][:, player, 0] - state["pos"][:, player, 0]
    dy = state["print_manager_last_pos"][:, player, 1] - state["pos"][:, player, 1]
    next_distance = state["print_manager_distance"][:, player] - np.hypot(dx, dy)
    toggle = active & (next_distance <= 0.0)
    no_toggle = active & ~toggle
    old_printing = state["printing"][:, player].copy()
    toggled_to_hole = toggle & old_printing

    state["print_manager_distance"][no_toggle, player] = next_distance[no_toggle]
    state["print_manager_last_pos"][active, player] = state["pos"][active, player]

    if toggle.any():
        state["printing"][toggle, player] = ~old_printing[toggle]
        _append_body_points(
            state,
            player=player,
            write_mask=toggle,
            insert_kind=BODY_KIND_IMPORTANT,
        )
        _emit_point_events(state, player, toggle, important=True)
        if toggled_to_hole.any():
            _clear_visible_trail(state, player, toggled_to_hole)
        _emit_printing_property_events(state, player, toggle)
        _assign_print_manager_random_distances(
            state,
            player=player,
            mask=toggle,
        )

    return int(toggle.sum()), int(no_toggle.sum()), int(toggled_to_hole.sum())


def _assign_print_manager_random_distances(
    state: dict[str, np.ndarray],
    *,
    player: int,
    mask: np.ndarray,
) -> None:
    for row in np.flatnonzero(mask):
        row_int = int(row)
        state["print_manager_distance"][row_int, player] = (
            _next_print_manager_random_distance(
                state,
                row=row_int,
                printing=bool(state["printing"][row_int, player]),
            )
        )


def _next_print_manager_random_distance(
    state: dict[str, np.ndarray],
    *,
    row: int,
    printing: bool,
) -> float:
    length = int(state["random_tape_length"][row])
    if length <= 0:
        state["random_tape_draw_count"][row] += 1
        return (
            PRINT_MANAGER_RANDOM_HALF_PRINT_DISTANCE
            if printing
            else PRINT_MANAGER_RANDOM_HALF_HOLE_DISTANCE
        )

    cursor = int(state["random_tape_cursor"][row])
    if cursor >= length:
        state["random_tape_exhausted"][row] = True
        raise VectorCompareError(f"row {row} Math.random tape exhausted after {cursor} calls")
    random_value = float(state["random_tape_values"][row, cursor])
    if not math.isfinite(random_value) or random_value < 0.0 or random_value >= 1.0:
        raise VectorCompareError(f"row {row} Math.random tape value must be in [0, 1)")
    state["random_tape_cursor"][row] = cursor + 1
    state["random_tape_draw_count"][row] += 1
    if printing:
        return 60.0 * (0.3 + random_value * 0.7)
    return 5.0 * (0.8 + random_value * 0.5)


def _stop_print_manager_on_death(
    state: dict[str, np.ndarray],
    *,
    player: int,
    death_mask: np.ndarray,
) -> tuple[int, int, int]:
    active = death_mask & state["print_manager_active"][:, player]
    if not active.any():
        return 0, 0, 0

    was_printing = state["printing"][:, player].copy()
    important_stop = active & was_printing

    if important_stop.any():
        _append_body_points(
            state,
            player=player,
            write_mask=important_stop,
            insert_kind=BODY_KIND_IMPORTANT,
        )
        _emit_point_events(state, player, important_stop, important=True)

    state["printing"][active, player] = False
    if important_stop.any():
        _clear_visible_trail(state, player, important_stop)
    _emit_printing_property_events(state, player, active)

    state["print_manager_active"][active, player] = False
    state["print_manager_distance"][active, player] = 0.0
    state["print_manager_last_pos"][active, player] = 0.0

    return int(active.sum()), int(important_stop.sum()), int(important_stop.sum())


def _clear_visible_trail(
    state: dict[str, np.ndarray],
    player: int,
    mask: np.ndarray,
) -> None:
    state["visible_trail_count"][mask, player] = 0
    state["has_visible_trail_last"][mask, player] = False
    state["visible_trail_last_pos"][mask, player] = 0.0
    state["has_draw_cursor"][mask, player] = False
    state["draw_cursor_pos"][mask, player] = 0.0


def _rows_to_mask(rows: np.ndarray, size: int) -> np.ndarray:
    mask = np.zeros(size, dtype=bool)
    mask[rows] = True
    return mask


def _player_id_for_event(player_ids: Sequence[Any], player_index: int) -> str:
    if player_index < 0 or player_index >= len(player_ids):
        raise VectorCompareError(f"event references unknown player index {player_index}")
    return str(player_ids[player_index])


def _optional_player_id_for_event(player_ids: Sequence[Any], player_index: int) -> str | None:
    if player_index < 0:
        return None
    return _player_id_for_event(player_ids, player_index)


def _event_bool_to_optional_bool(value: int) -> bool | None:
    if value < 0:
        return None
    return bool(value)


def _compare_value(
    actual: Any,
    expected: Any,
    path: str,
    tolerance: float,
    mismatches: list[dict[str, Any]],
) -> None:
    if isinstance(expected, int | float) and not isinstance(expected, bool):
        if not isinstance(actual, int | float) or isinstance(actual, bool):
            mismatches.append(
                {"path": path, "expected": expected, "actual": actual, "reason": "type differs"}
            )
            return
        if math.isfinite(float(expected)) and abs(float(actual) - float(expected)) <= tolerance:
            return
        mismatches.append(
            {
                "path": path,
                "expected": expected,
                "actual": actual,
                "tolerance": tolerance,
                "reason": "numeric values differ",
            }
        )
        return

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            mismatches.append(
                {"path": path, "expected": expected, "actual": actual, "reason": "list differs"}
            )
            return
        for index, expected_item in enumerate(expected):
            _compare_value(
                actual[index],
                expected_item,
                f"{path}[{index}]",
                tolerance,
                mismatches,
            )
        return

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            mismatches.append(
                {"path": path, "expected": expected, "actual": actual, "reason": "type differs"}
            )
            return
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        if actual_keys != expected_keys:
            mismatches.append(
                {
                    "path": path,
                    "expected": sorted(expected_keys),
                    "actual": sorted(actual_keys),
                    "reason": "object fields differ",
                }
            )
            return
        for key, expected_item in expected.items():
            key_text = str(key)
            _compare_value(
                actual.get(key),
                expected_item,
                f"{path}.{key_text}",
                _tolerance_for_field(key_text, {}),
                mismatches,
            )
        return

    if actual != expected:
        mismatches.append(
            {"path": path, "expected": expected, "actual": actual, "reason": "values differ"}
        )


def _tolerance_for_field(field: str, tolerances: Mapping[str, Any]) -> float:
    if field in {"x", "y", "lastTrailPoint", "lastX", "lastY"}:
        return float(tolerances.get("position", 1e-6))
    if field == "distance":
        return float(tolerances.get("distance", tolerances.get("position", 1e-6)))
    if field in {"angle", "step_ms"}:
        return float(tolerances.get("angle", 1e-6))
    return 0.0


def _resolve_fixture_path(path: str) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    return (REPO_ROOT / raw).resolve()


def _numpy_dtype(name: str) -> Any:
    dtypes = {
        "bool": bool,
        "float64": np.float64,
        "float32": np.float32,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
    }
    if name not in dtypes:
        raise VectorCompareError(f"unsupported array dtype {name!r}")
    return dtypes[name]


def _mapping(value: Any, field: str, *, default: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    if value is None:
        if default is not None:
            return default
        raise VectorCompareError(f"{field} is required")
    if not isinstance(value, Mapping):
        raise VectorCompareError(f"{field} must be an object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise VectorCompareError(f"{field} must be a list")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise VectorCompareError(f"{field} must be a non-empty string")
    return value


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VectorCompareError(f"{field} must be an integer")
    return int(value)


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise VectorCompareError(f"{field} must be a number")
    return float(value)


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare one fixture-seeded vector array step to common traces."
    )
    parser.add_argument("paths", nargs="+", help="Scenario JSON files or batch manifests.")
    parser.add_argument(
        "--body-capacity",
        type=_nonnegative_int,
        default=DEFAULT_BODY_CAPACITY,
        help="Fixed K for the seeded body buffer. Defaults to 16 for one-step append room.",
    )
    parser.add_argument("--step-index", type=_nonnegative_int, default=0)
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

    summary = compare_inputs(
        args.paths,
        body_capacity=args.body_capacity,
        step_index=args.step_index,
        require_verified=not args.allow_unverified,
    )
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_plain(summary)

    counts = summary["summary"]
    if counts["failed"] or (args.fail_on_unsupported and counts["unsupported"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
