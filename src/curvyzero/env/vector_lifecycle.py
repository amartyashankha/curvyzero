"""Small vector reset/spawn composition boundary.

This module intentionally stops short of full lifecycle support. It only wires
the production-facing masked reset boundary to the narrow source-shaped spawn
helper so selected rows can be reset from a template and immediately spawned
from row-local random tape.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import vector_reset, vector_runtime, vector_spawn
from .config import CurvyTronReferenceDefaults


RESET_SPAWN_INFO_SCHEMA_ID = "curvyzero_vector_reset_spawn_info/v1"
RESET_SPAWN_WARMUP_NO_BONUS_INFO_SCHEMA_ID = (
    "curvyzero_vector_reset_spawn_warmup_no_bonus_info/v1"
)
RESET_SPAWN_WARMUP_INFO_SCHEMA_ID = (
    "curvyzero_vector_reset_spawn_warmup_1v1_no_bonus_info/v1"
)
WARMDOWN_ADVANCE_NO_BONUS_INFO_SCHEMA_ID = (
    "curvyzero_vector_warmdown_no_bonus_advance_info/v1"
)
WARMUP_START_STEP_1V1_NO_BONUS_INFO_SCHEMA_ID = (
    "curvyzero_vector_warmup_start_step_1v1_no_bonus_info/v1"
)
RESET_SPAWN_SURFACE = "reset_spawn_rows_only"
RESET_SPAWN_LIFECYCLE_SURFACE = "optional_1v1_delayed_start_metadata"
RESET_SPAWN_WARMUP_NO_BONUS_SURFACE = "reset_spawn_warmup_no_bonus_rows"
RESET_SPAWN_WARMUP_SURFACE = "reset_spawn_warmup_1v1_no_bonus_rows"
WARMDOWN_ADVANCE_NO_BONUS_SURFACE = "advance_warmdown_no_bonus_rows"
WARMUP_START_STEP_1V1_NO_BONUS_SURFACE = "run_warmup_start_step_1v1_no_bonus_rows"
SOURCE_ROUND_WARMUP_MS = 3_000.0
SOURCE_ROUND_WARMDOWN_MS = vector_runtime.SOURCE_ROUND_WARMDOWN_MS
SOURCE_TRAIL_START_DELAY_MS = vector_runtime.SOURCE_TRAIL_START_DELAY_MS
TIMER_KIND_NONE = vector_runtime.TIMER_KIND_NONE
TIMER_KIND_PRINT_MANAGER_START = vector_runtime.TIMER_KIND_PRINT_MANAGER_START
TIMER_KIND_GAME_START = vector_runtime.TIMER_KIND_GAME_START
TIMER_KIND_WARMDOWN_END = vector_runtime.TIMER_KIND_WARMDOWN_END
TIMER_PLAYER_NONE = vector_runtime.TIMER_PLAYER_NONE
SUPPORTED_LIFECYCLE_PLAYER_COUNT = 2
SUPPORTED_NO_BONUS_WARMUP_PLAYER_COUNTS = (2, 3, 4)

_REQUIRED_RESET_ARRAYS: tuple[str, ...] = (
    "episode_id",
    "episode_step",
    "env_active",
    "reset_pending",
    "done",
    "terminated",
    "truncated",
    "terminal_reason",
    "reset_seed",
    "reset_source",
)

_REQUIRED_SPAWN_ARRAYS: tuple[str, ...] = (
    "pos",
    "heading",
    "alive",
    "present",
    "map_size",
    "random_tape_values",
    "random_tape_length",
    "random_tape_cursor",
    "random_tape_draw_count",
)

_REQUIRED_ARRAYS: tuple[str, ...] = tuple(
    dict.fromkeys((*_REQUIRED_RESET_ARRAYS, *_REQUIRED_SPAWN_ARRAYS))
)

_OPTIONAL_ROUND_LIFECYCLE_ARRAYS: tuple[str, ...] = (
    "started",
    "in_round",
    "world_active",
    "world_body_count",
)

_OPTIONAL_DELAYED_START_TIMER_ARRAYS: tuple[str, ...] = (
    "timer_active",
    "timer_remaining_ms",
    "timer_kind",
    "timer_player",
    "timer_seq",
    "timer_overflow",
)

_REQUIRED_1V1_WARMUP_ARRAYS: tuple[str, ...] = (
    *_OPTIONAL_ROUND_LIFECYCLE_ARRAYS,
    *_OPTIONAL_DELAYED_START_TIMER_ARRAYS,
    "present",
    "alive",
)


class VectorLifecycleError(ValueError):
    """Raised when wrapper-level vector lifecycle inputs are invalid."""


def reset_and_spawn_round_rows(
    target: Mapping[str, np.ndarray],
    reset_template: Mapping[str, np.ndarray],
    row_mask: np.ndarray,
    *,
    player_count: int,
    reset_seed: np.ndarray | int,
    reset_source: np.ndarray | int = vector_reset.RESET_SOURCE_MANUAL,
    snapshot_array_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Reset selected rows from a template, then spawn their present players.

    The reset template owns the selected rows' row-local random tape. After
    ``reset_arrays`` copies those template rows into ``target``,
    ``spawn_round_rows`` consumes the selected rows' tape and advances only the
    selected random cursors.

    If the target and template cannot compose because arrays are missing or the
    key sets differ, this returns a non-mutating metadata report with
    ``can_compose=False`` and exact missing array names.
    """

    mask = _bool_row_mask(row_mask, "row_mask")
    rows = np.flatnonzero(mask).astype(np.int32)
    array_report = reset_spawn_array_report(target, reset_template)

    base_info: dict[str, Any] = {
        "schema": RESET_SPAWN_INFO_SCHEMA_ID,
        "surface": RESET_SPAWN_SURFACE,
        "full_lifecycle": False,
        "row_count": int(mask.sum()),
        "row_mask": mask.copy(),
        "rows": rows,
        **array_report,
    }
    if not array_report["can_compose"]:
        return {
            **base_info,
            "scheduled_timer_count": 0,
            "lifecycle_schedule_info": _lifecycle_schedule_report(
                target,
                mask,
                player_count=player_count,
                can_compose=False,
            ),
            "reset_info": None,
            "spawn_info": None,
            "terminal_transition_snapshot": None,
        }

    lifecycle_schedule_info = _lifecycle_schedule_report(
        target,
        mask,
        player_count=player_count,
        can_compose=True,
    )
    _validate_optional_lifecycle_arrays(
        target,
        mask,
        player_count=player_count,
        schedule_info=lifecycle_schedule_info,
    )

    reset_info = vector_reset.reset_arrays(
        target,
        reset_template,
        mask,
        reset_seed=reset_seed,
        reset_source=reset_source,
        snapshot_array_names=snapshot_array_names,
    )
    spawn_info = vector_spawn.spawn_round_rows(
        target,
        mask,
        player_count=player_count,
    )
    lifecycle_schedule_info = _apply_optional_lifecycle_schedule(
        target,
        mask,
        player_count=player_count,
        schedule_info=lifecycle_schedule_info,
    )

    return {
        **base_info,
        "reset_count": int(reset_info["reset_count"]),
        "spawn_count": int(spawn_info["spawn_count"]),
        "reset_rows": np.asarray(reset_info["reset_rows"]).copy(),
        "spawn_rows": np.asarray(spawn_info["spawn_rows"]).copy(),
        "random_draw_count_delta": np.asarray(
            spawn_info["random_draw_count_delta"]
        ).copy(),
        "random_tape_cursor": np.asarray(spawn_info["random_tape_cursor"]).copy(),
        "random_tape_draw_count": np.asarray(
            spawn_info["random_tape_draw_count"]
        ).copy(),
        "world_body_insert_count": int(spawn_info["world_body_insert_count"]),
        "scheduled_timer_count": int(
            lifecycle_schedule_info["delayed_start_timers"]["scheduled_timer_count"]
        ),
        "lifecycle_schedule_info": lifecycle_schedule_info,
        "terminal_transition_snapshot": reset_info["terminal_transition_snapshot"],
        "reset_info": reset_info,
        "spawn_info": spawn_info,
    }


def reset_spawn_warmup_no_bonus_rows(
    target: Mapping[str, np.ndarray],
    reset_template: Mapping[str, np.ndarray],
    row_mask: np.ndarray,
    *,
    player_count: int,
    reset_seed: np.ndarray | int,
    reset_source: np.ndarray | int = vector_reset.RESET_SOURCE_MANUAL,
    first_warmup_ms: float = SOURCE_ROUND_WARMUP_MS,
    snapshot_array_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Reset, spawn, and schedule the first source warmup timer for 2P/3P/4P.

    This no-bonus helper deliberately schedules only one ``GAME_START`` timer
    per selected row. It does not start PrintManager timers yet; those belong
    to the runtime timer-advance step that fires ``game:start``.
    """

    return _reset_spawn_warmup_no_bonus_rows_impl(
        target,
        reset_template,
        row_mask,
        player_count=player_count,
        reset_seed=reset_seed,
        reset_source=reset_source,
        first_warmup_ms=first_warmup_ms,
        snapshot_array_names=snapshot_array_names,
        schema=RESET_SPAWN_WARMUP_NO_BONUS_INFO_SCHEMA_ID,
        surface=RESET_SPAWN_WARMUP_NO_BONUS_SURFACE,
        require_all_present=False,
    )


def reset_spawn_warmup_1v1_no_bonus_rows(
    target: Mapping[str, np.ndarray],
    reset_template: Mapping[str, np.ndarray],
    row_mask: np.ndarray,
    *,
    reset_seed: np.ndarray | int,
    reset_source: np.ndarray | int = vector_reset.RESET_SOURCE_MANUAL,
    first_warmup_ms: float = SOURCE_ROUND_WARMUP_MS,
    snapshot_array_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for strict 1v1/no-bonus warmup reset.

    This is a stricter 1v1/no-bonus row-start helper than
    ``reset_and_spawn_round_rows``. It deliberately schedules only a
    ``GAME_START`` timer. It does not start PrintManager timers yet; those
    belong to the later timer-advance step that fires ``game:start``.
    """

    return _reset_spawn_warmup_no_bonus_rows_impl(
        target,
        reset_template,
        row_mask,
        player_count=SUPPORTED_LIFECYCLE_PLAYER_COUNT,
        reset_seed=reset_seed,
        reset_source=reset_source,
        first_warmup_ms=first_warmup_ms,
        snapshot_array_names=snapshot_array_names,
        schema=RESET_SPAWN_WARMUP_INFO_SCHEMA_ID,
        surface=RESET_SPAWN_WARMUP_SURFACE,
        require_all_present=True,
    )


def advance_warmdown_no_bonus_rows(
    target: Mapping[str, np.ndarray],
    advance_ms: Any,
    *,
    player_count: int,
    next_warmup_ms: float = SOURCE_ROUND_WARMUP_MS,
    max_timer_callbacks: int = 16,
) -> dict[str, Any]:
    """Advance no-bonus round warmdown and spawn the next round for 2P/3P/4P.

    This continues rows that opted into runtime round lifecycle metadata
    (``round_done``/``warmdown_pending`` plus the standard timer arrays). When
    a ``WARMDOWN_END`` timer fires and no match winner is present, it applies
    the source ``game:stop``/``round:new`` slice: clear round-local state,
    spawn present players from the row-local random tape, and schedule the next
    ``GAME_START`` warmup timer.
    """

    player_count = _validate_no_bonus_warmup_player_count(player_count)
    _validate_first_warmup_ms(next_warmup_ms)
    if (
        not isinstance(max_timer_callbacks, int)
        or isinstance(max_timer_callbacks, bool)
        or max_timer_callbacks < 1
    ):
        raise VectorLifecycleError("max_timer_callbacks must be a positive integer")

    tick = np.asarray(target["tick"])
    if tick.ndim != 1 or not np.issubdtype(tick.dtype, np.integer):
        raise VectorLifecycleError("tick must be an integer array with shape [B]")
    batch_size = int(tick.shape[0])
    advance = _row_float_input(advance_ms, batch_size, "advance_ms")
    mask = np.ones(batch_size, dtype=bool)
    _validate_warmup_arrays(target, mask, player_count=player_count)
    _validate_round_local_warmup_arrays(target, mask, player_count=player_count)
    _validate_warmdown_arrays(target, batch_size=batch_size)

    random_draw_count_before = np.asarray(target["random_tape_draw_count"]).copy()
    timer_active = target["timer_active"]
    timer_remaining_ms = target["timer_remaining_ms"]
    timer_kind = target["timer_kind"]
    timer_overflow = target["timer_overflow"]

    counters = {
        "pre_step_timer_advances": 0,
        "pre_step_timer_fires": 0,
        "warmdown_end_fires": 0,
        "game_stop_fires": 0,
        "next_round_count": 0,
        "match_end_count": 0,
        "round_clear_print_manager_stops": 0,
        "scheduled_timer_count": 0,
        "random_tape_draws": 0,
        "random_tape_exhaustions": 0,
    }
    warmdown_rows: list[int] = []
    game_stop_rows: list[int] = []
    next_round_rows: list[int] = []
    match_end_rows: list[int] = []
    stopped_rows: list[int] = []
    stopped_players: list[int] = []
    spawn_infos: list[dict[str, Any]] = []
    scheduled_rows: list[int] = []
    scheduled_slots: list[int] = []
    callback_count = 0

    for row in range(batch_size):
        if not bool(target["warmdown_pending"][row]):
            continue
        active_slots = np.flatnonzero(timer_active[row])
        if active_slots.size == 0:
            continue

        counters["pre_step_timer_advances"] += 1
        budget_ms = float(advance[row])
        while bool(timer_active[row].any()):
            active_slots = np.flatnonzero(timer_active[row])
            active_remaining = timer_remaining_ms[row, active_slots]
            due_after_ms = max(0.0, float(active_remaining.min()))
            if due_after_ms > budget_ms:
                timer_remaining_ms[row, active_slots] -= budget_ms
                break

            timer_remaining_ms[row, active_slots] -= due_after_ms
            budget_ms -= due_after_ms
            due_slots = [
                int(slot)
                for slot in active_slots
                if float(timer_remaining_ms[row, int(slot)]) <= 0.0
            ]
            due_slots.sort(key=lambda slot: int(target["timer_seq"][row, slot]))
            if not due_slots:
                break

            for slot in due_slots:
                if not bool(timer_active[row, slot]):
                    continue
                callback_count += 1
                if callback_count > max_timer_callbacks:
                    raise VectorLifecycleError(
                        f"timer advance exceeded {max_timer_callbacks} callbacks"
                    )
                kind = int(timer_kind[row, slot])
                if kind != TIMER_KIND_WARMDOWN_END:
                    raise VectorLifecycleError(
                        f"unsupported warmdown timer kind {kind}"
                    )

                _clear_timer_slot(target, row=row, slot=slot)
                counters["pre_step_timer_fires"] += 1
                counters["warmdown_end_fires"] += 1
                counters["game_stop_fires"] += 1
                warmdown_rows.append(row)
                game_stop_rows.append(row)

                match_done, match_winner = _match_done_for_warmdown_row(
                    target,
                    row=row,
                    player_count=player_count,
                )
                if match_done:
                    _end_match_row(target, row=row, match_winner=match_winner)
                    counters["match_end_count"] += 1
                    match_end_rows.append(row)
                    budget_ms = 0.0
                    continue

                stopped = _stop_active_print_managers_for_round_clear(
                    target,
                    row=row,
                    player_count=player_count,
                )
                for player in stopped:
                    stopped_rows.append(row)
                    stopped_players.append(player)
                counters["round_clear_print_manager_stops"] += len(stopped)

                row_mask = np.zeros(batch_size, dtype=bool)
                row_mask[row] = True
                _resize_row_arena_to_present_count(
                    target,
                    row=row,
                    player_count=player_count,
                )
                _clear_selected_round_local_arrays(
                    target,
                    row_mask,
                    clear_present_players_only=True,
                    player_count=player_count,
                )
                target["round_done"][row] = False
                target["warmdown_pending"][row] = False
                target["match_done"][row] = False
                if "round_winner" in target:
                    target["round_winner"][row] = -1
                if "match_winner" in target:
                    target["match_winner"][row] = -1
                if "terminal_reason" in target:
                    target["terminal_reason"][row] = vector_reset.TERMINAL_REASON_NONE
                if "winner" in target:
                    target["winner"][row] = -1
                if "draw" in target:
                    target["draw"][row] = False

                spawn_info = vector_spawn.spawn_round_rows(
                    target,
                    row_mask,
                    player_count=player_count,
                )
                spawn_infos.append(spawn_info)
                timer_info = _schedule_game_start_timers(
                    target,
                    row_mask,
                    first_warmup_ms=float(next_warmup_ms),
                )
                scheduled_rows.extend(int(value) for value in timer_info["scheduled_timer_rows"])
                scheduled_slots.extend(int(value) for value in timer_info["scheduled_timer_slots"])
                counters["scheduled_timer_count"] += int(timer_info["scheduled_timer_count"])
                counters["next_round_count"] += 1
                next_round_rows.append(row)
                budget_ms = 0.0

            if budget_ms <= 0.0:
                break

    random_draw_delta = (
        np.asarray(target["random_tape_draw_count"]).astype(np.int64)
        - random_draw_count_before.astype(np.int64)
    )
    if bool((random_draw_delta < 0).any()):
        raise VectorLifecycleError("random_tape_draw_count cannot decrease")
    counters["random_tape_draws"] = int(random_draw_delta.sum())
    if "random_tape_exhausted" in target:
        counters["random_tape_exhaustions"] = int(
            np.asarray(target["random_tape_exhausted"], dtype=bool).sum()
        )

    return {
        "schema": WARMDOWN_ADVANCE_NO_BONUS_INFO_SCHEMA_ID,
        "surface": WARMDOWN_ADVANCE_NO_BONUS_SURFACE,
        "player_count": player_count,
        "supported_player_counts": list(SUPPORTED_NO_BONUS_WARMUP_PLAYER_COUNTS),
        **counters,
        "advance_ms": advance.copy(),
        "warmdown_end_rows": np.asarray(warmdown_rows, dtype=np.int32),
        "game_stop_rows": np.asarray(game_stop_rows, dtype=np.int32),
        "next_round_rows": np.asarray(next_round_rows, dtype=np.int32),
        "match_end_rows": np.asarray(match_end_rows, dtype=np.int32),
        "round_clear_print_manager_stop_rows": np.asarray(stopped_rows, dtype=np.int32),
        "round_clear_print_manager_stop_players": np.asarray(
            stopped_players,
            dtype=np.int16,
        ),
        "scheduled_timer_rows": np.asarray(scheduled_rows, dtype=np.int32),
        "scheduled_timer_slots": np.asarray(scheduled_slots, dtype=np.int16),
        "scheduled_timer_kind": "game:start",
        "scheduled_timer_kind_code": TIMER_KIND_GAME_START,
        "spawn_infos": spawn_infos,
        "random_draw_count_delta": random_draw_delta.astype(np.int32),
        "timer_overflow_rows": np.flatnonzero(timer_overflow).astype(np.int32),
    }


def _reset_spawn_warmup_no_bonus_rows_impl(
    target: Mapping[str, np.ndarray],
    reset_template: Mapping[str, np.ndarray],
    row_mask: np.ndarray,
    *,
    player_count: int,
    reset_seed: np.ndarray | int,
    reset_source: np.ndarray | int,
    first_warmup_ms: float,
    snapshot_array_names: Sequence[str] | None,
    schema: str,
    surface: str,
    require_all_present: bool,
) -> dict[str, Any]:
    player_count = _validate_no_bonus_warmup_player_count(player_count)
    mask = _bool_row_mask(row_mask, "row_mask")
    rows = np.flatnonzero(mask).astype(np.int32)
    _validate_first_warmup_ms(first_warmup_ms)
    array_report = reset_spawn_array_report(target, reset_template)
    warmup_array_report = _warmup_array_report(target, reset_template)
    can_compose = bool(array_report["can_compose"] and warmup_array_report["can_compose"])
    base_info: dict[str, Any] = {
        "schema": schema,
        "surface": surface,
        "full_lifecycle": False,
        "player_count": player_count,
        "supported_player_counts": list(SUPPORTED_NO_BONUS_WARMUP_PLAYER_COUNTS),
        "row_count": int(mask.sum()),
        "row_mask": mask.copy(),
        "rows": rows,
        **array_report,
        "warmup_required_arrays": list(_REQUIRED_1V1_WARMUP_ARRAYS),
        "warmup_missing_target_arrays": warmup_array_report["missing_target_arrays"],
        "warmup_missing_reset_template_arrays": warmup_array_report[
            "missing_reset_template_arrays"
        ],
    }
    if not can_compose:
        return {
            **base_info,
            "can_compose": False,
            "scheduled_timer_count": 0,
            "scheduled_timer_rows": np.asarray([], dtype=np.int32),
            "scheduled_timer_slots": np.asarray([], dtype=np.int16),
            "reset_info": None,
            "spawn_info": None,
            "terminal_transition_snapshot": None,
        }

    _validate_warmup_arrays(target, mask, player_count=player_count)
    _validate_warmup_arrays(reset_template, mask, player_count=player_count)
    _validate_round_local_warmup_arrays(target, mask, player_count=player_count)
    _validate_round_local_warmup_arrays(reset_template, mask, player_count=player_count)

    reset_info = vector_reset.reset_arrays(
        target,
        reset_template,
        mask,
        reset_seed=reset_seed,
        reset_source=reset_source,
        snapshot_array_names=snapshot_array_names,
    )
    if require_all_present:
        _validate_all_present_rows(target, mask, player_count=player_count)
    _clear_selected_round_local_arrays(target, mask)
    spawn_info = vector_spawn.spawn_round_rows(
        target,
        mask,
        player_count=player_count,
    )
    timer_info = _schedule_game_start_timers(
        target,
        mask,
        first_warmup_ms=float(first_warmup_ms),
    )

    return {
        **base_info,
        "can_compose": True,
        "reset_count": int(reset_info["reset_count"]),
        "spawn_count": int(spawn_info["spawn_count"]),
        "reset_rows": np.asarray(reset_info["reset_rows"]).copy(),
        "spawn_rows": np.asarray(spawn_info["spawn_rows"]).copy(),
        "world_flags": {
            "started": True,
            "in_round": True,
            "world_active": False,
            "world_body_count": 0,
            "rows": rows.copy(),
        },
        "first_warmup_ms": float(first_warmup_ms),
        **timer_info,
        "terminal_transition_snapshot": reset_info["terminal_transition_snapshot"],
        "reset_info": reset_info,
        "spawn_info": spawn_info,
    }


def run_warmup_start_step_1v1_no_bonus_rows(
    target: Mapping[str, np.ndarray],
    reset_template: Mapping[str, np.ndarray],
    row_mask: np.ndarray,
    *,
    reset_seed: np.ndarray | int,
    runtime_steps: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    reset_source: np.ndarray | int = vector_reset.RESET_SOURCE_MANUAL,
    first_warmup_ms: float = SOURCE_ROUND_WARMUP_MS,
    start_advance_ms: Any | None = None,
    event_mode: str = vector_runtime.EVENT_MODE_DEBUG,
    snapshot_array_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the strict reset/spawn, warmup start, and runtime step slice.

    ``runtime_steps`` uses the same prepared-batch mapping shape accepted by
    ``vector_runtime.VectorStepInput.from_mapping``. Row-leading arrays may be
    full-batch arrays or already narrowed to the selected rows.
    """

    mask = _bool_row_mask(row_mask, "row_mask")
    rows = np.flatnonzero(mask).astype(np.int32)
    step_mappings = _runtime_step_mappings(runtime_steps)

    reset_spawn_info = reset_spawn_warmup_1v1_no_bonus_rows(
        target,
        reset_template,
        mask,
        reset_seed=reset_seed,
        reset_source=reset_source,
        first_warmup_ms=first_warmup_ms,
        snapshot_array_names=snapshot_array_names,
    )
    base_info: dict[str, Any] = {
        "schema": WARMUP_START_STEP_1V1_NO_BONUS_INFO_SCHEMA_ID,
        "surface": WARMUP_START_STEP_1V1_NO_BONUS_SURFACE,
        "full_lifecycle": False,
        "player_count": SUPPORTED_LIFECYCLE_PLAYER_COUNT,
        "row_count": int(mask.sum()),
        "row_mask": mask.copy(),
        "rows": rows.copy(),
        "reset_spawn_info": reset_spawn_info,
    }
    if not reset_spawn_info["can_compose"] or rows.size == 0:
        return {
            **base_info,
            "warmup_timer_info": None,
            "runtime_step_counters": [],
            "runtime_step_count": 0,
        }

    selected_state = _row_subset_state(target, rows, batch_size=mask.shape[0])
    selected_advance_ms = _selected_row_value(
        (
            float(first_warmup_ms) + SOURCE_TRAIL_START_DELAY_MS
            if start_advance_ms is None
            else start_advance_ms
        ),
        rows,
        batch_size=mask.shape[0],
        selected_count=rows.size,
        field="start_advance_ms",
    )
    warmup_timer_info = vector_runtime.advance_warmup_1v1_no_bonus_timers(
        selected_state,
        selected_advance_ms,
    )

    runtime_step_counters: list[dict[str, int]] = []
    for index, step_mapping in enumerate(step_mappings):
        selected_step = _selected_runtime_step_mapping(
            step_mapping,
            rows,
            batch_size=mask.shape[0],
            selected_count=rows.size,
            index=index,
        )
        step_input = vector_runtime.VectorStepInput.from_mapping(
            selected_state,
            selected_step,
            event_mode=event_mode,
        )
        runtime_step_counters.append(dict(vector_runtime.step_many(step_input)))

    _copy_row_subset_back(target, selected_state, rows, batch_size=mask.shape[0])

    return {
        **base_info,
        "warmup_timer_info": warmup_timer_info,
        "runtime_step_counters": runtime_step_counters,
        "runtime_step_count": len(runtime_step_counters),
    }


def reset_spawn_array_report(
    target: Mapping[str, np.ndarray],
    reset_template: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Return the exact array-key report needed before reset/spawn composition."""

    target_keys = set(target.keys())
    template_keys = set(reset_template.keys())
    required_keys = set(_REQUIRED_ARRAYS)

    missing_target_arrays = sorted((required_keys | template_keys) - target_keys)
    missing_reset_template_arrays = sorted((required_keys | target_keys) - template_keys)

    return {
        "can_compose": not missing_target_arrays and not missing_reset_template_arrays,
        "required_reset_arrays": list(_REQUIRED_RESET_ARRAYS),
        "required_spawn_arrays": list(_REQUIRED_SPAWN_ARRAYS),
        "missing_target_arrays": missing_target_arrays,
        "missing_reset_template_arrays": missing_reset_template_arrays,
        "missing_target_required_arrays": sorted(required_keys - target_keys),
        "missing_reset_template_required_arrays": sorted(required_keys - template_keys),
        "target_only_arrays": sorted(target_keys - template_keys),
        "reset_template_only_arrays": sorted(template_keys - target_keys),
    }


def _bool_row_mask(value: np.ndarray, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.dtype != bool or mask.ndim != 1:
        raise VectorLifecycleError(f"{name} must be a bool array with shape [B]")
    return mask


def _runtime_step_mappings(
    runtime_steps: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(runtime_steps, Mapping):
        steps = (runtime_steps,)
    else:
        steps = tuple(runtime_steps)
    if not steps:
        raise VectorLifecycleError("runtime_steps must contain at least one step mapping")
    if not all(isinstance(step, Mapping) for step in steps):
        raise VectorLifecycleError("runtime_steps entries must be mappings")
    return steps


def _row_subset_state(
    state: Mapping[str, np.ndarray],
    rows: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    subset: dict[str, np.ndarray] = {}
    for name, value in state.items():
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] == batch_size:
            subset[name] = array[rows, ...].copy()
        else:
            subset[name] = array.copy()
    return subset


def _copy_row_subset_back(
    target: Mapping[str, np.ndarray],
    subset: Mapping[str, np.ndarray],
    rows: np.ndarray,
    *,
    batch_size: int,
) -> None:
    for name, selected_value in subset.items():
        if name not in target:
            continue
        target_array = np.asarray(target[name])
        selected_array = np.asarray(selected_value)
        if (
            target_array.ndim >= 1
            and target_array.shape[0] == batch_size
            and selected_array.shape == (rows.size, *target_array.shape[1:])
        ):
            target_array[rows, ...] = selected_array


def _selected_runtime_step_mapping(
    step_mapping: Mapping[str, Any],
    rows: np.ndarray,
    *,
    batch_size: int,
    selected_count: int,
    index: int,
) -> dict[str, Any]:
    player_count = step_mapping.get("player_count")
    if player_count != SUPPORTED_LIFECYCLE_PLAYER_COUNT:
        raise VectorLifecycleError(
            f"runtime_steps[{index}].player_count must be 2 for 1v1/no-bonus rows"
        )
    selected = dict(step_mapping)
    selected["step_ms"] = _selected_row_value(
        step_mapping.get("step_ms"),
        rows,
        batch_size=batch_size,
        selected_count=selected_count,
        field=f"runtime_steps[{index}].step_ms",
    )
    selected["source_moves"] = _selected_row_value(
        step_mapping.get("source_moves"),
        rows,
        batch_size=batch_size,
        selected_count=selected_count,
        field=f"runtime_steps[{index}].source_moves",
        allow_scalar=False,
    )
    if "print_manager_mode" in step_mapping:
        selected["print_manager_mode"] = _selected_row_value(
            step_mapping["print_manager_mode"],
            rows,
            batch_size=batch_size,
            selected_count=selected_count,
            field=f"runtime_steps[{index}].print_manager_mode",
        )
    if "timer_advance_ms" in step_mapping:
        selected["timer_advance_ms"] = _selected_row_value(
            step_mapping["timer_advance_ms"],
            rows,
            batch_size=batch_size,
            selected_count=selected_count,
            field=f"runtime_steps[{index}].timer_advance_ms",
        )
    return selected


def _selected_row_value(
    value: Any,
    rows: np.ndarray,
    *,
    batch_size: int,
    selected_count: int,
    field: str,
    allow_scalar: bool = True,
) -> Any:
    if value is None:
        raise VectorLifecycleError(f"{field} is required")
    array = np.asarray(value)
    if array.ndim == 0:
        if allow_scalar:
            return np.full(selected_count, array.item(), dtype=array.dtype)
        raise VectorLifecycleError(
            f"{field} must have leading shape [B] or selected row count"
        )
    if array.shape[0] == batch_size:
        return array[rows, ...].copy()
    if array.shape[0] == selected_count:
        return array.copy()
    raise VectorLifecycleError(
        f"{field} must have leading shape [B] or selected row count"
    )


def _row_float_input(value: Any, row_count: int, field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise VectorLifecycleError(f"{field} cannot be converted to float64") from exc
    if array.ndim == 0:
        array = np.full(row_count, float(array), dtype=np.float64)
    if array.shape != (row_count,):
        raise VectorLifecycleError(f"{field} must be a numeric array with shape [B]")
    if not bool(np.isfinite(array).all()):
        raise VectorLifecycleError(f"{field} values must be finite")
    if bool((array < 0.0).any()):
        raise VectorLifecycleError(f"{field} values must be non-negative")
    return array


def _validate_first_warmup_ms(value: float) -> None:
    if not np.isfinite(float(value)) or float(value) < 0.0:
        raise VectorLifecycleError("first_warmup_ms must be finite and nonnegative")


def _validate_no_bonus_warmup_player_count(player_count: int) -> int:
    if not isinstance(player_count, int) or isinstance(player_count, bool):
        raise VectorLifecycleError("player_count must be 2, 3, or 4 for no-bonus warmup")
    if player_count not in SUPPORTED_NO_BONUS_WARMUP_PLAYER_COUNTS:
        raise VectorLifecycleError("player_count must be 2, 3, or 4 for no-bonus warmup")
    return player_count


def _warmup_array_report(
    target: Mapping[str, np.ndarray],
    reset_template: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    required = set(_REQUIRED_1V1_WARMUP_ARRAYS)
    target_keys = set(target)
    template_keys = set(reset_template)
    missing_target_arrays = sorted(required - target_keys)
    missing_reset_template_arrays = sorted(required - template_keys)
    return {
        "can_compose": not missing_target_arrays and not missing_reset_template_arrays,
        "missing_target_arrays": missing_target_arrays,
        "missing_reset_template_arrays": missing_reset_template_arrays,
    }


def _validate_warmup_arrays(
    state: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    player_count: int,
) -> None:
    batch_size = mask.shape[0]
    _typed_array(state, "started", np.bool_, (batch_size,))
    _typed_array(state, "in_round", np.bool_, (batch_size,))
    _typed_array(state, "world_active", np.bool_, (batch_size,))
    _typed_array(state, "world_body_count", np.int32, (batch_size,))
    timer_shape = _typed_array(
        state,
        "timer_active",
        np.bool_,
        ndim=2,
        batch_size=batch_size,
    ).shape
    _typed_array(state, "timer_remaining_ms", np.float64, timer_shape)
    _typed_array(state, "timer_kind", np.int16, timer_shape)
    _typed_array(state, "timer_player", np.int16, timer_shape)
    _typed_array(state, "timer_seq", np.int32, timer_shape)
    _typed_array(state, "timer_overflow", np.bool_, (batch_size,))
    _typed_array(
        state,
        "present",
        np.bool_,
        (batch_size, player_count),
    )
    _typed_array(
        state,
        "alive",
        np.bool_,
        (batch_size, player_count),
    )


def _validate_round_local_warmup_arrays(
    state: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    player_count: int,
) -> None:
    batch_size = mask.shape[0]
    player_shape = (batch_size, player_count)
    for name in ("printing", "print_manager_active"):
        if name in state:
            _typed_array(state, name, np.bool_, player_shape)
    for name in ("print_manager_distance",):
        if name in state:
            _numeric_array(state, name, player_shape)
    if "print_manager_last_pos" in state:
        _numeric_array(state, "print_manager_last_pos", (*player_shape, 2))
    for name in ("score", "round_score", "body_count"):
        if name in state:
            _integer_array(state, name, player_shape)
    if ("death_count" in state) != ("death_player" in state):
        raise VectorLifecycleError("death_count and death_player must be supplied together")
    if "death_count" in state:
        _integer_array(state, "death_count", (batch_size,))
        _integer_array(state, "death_player", player_shape)


def _validate_warmdown_arrays(
    state: Mapping[str, np.ndarray],
    *,
    batch_size: int,
) -> None:
    for name in ("round_done", "warmdown_pending", "match_done"):
        _typed_array(state, name, np.bool_, (batch_size,))
    if "round_winner" in state:
        _integer_array(state, "round_winner", (batch_size,))
    if "match_winner" in state:
        _integer_array(state, "match_winner", (batch_size,))
    if "winner" in state:
        _integer_array(state, "winner", (batch_size,))
    if "draw" in state:
        _typed_array(state, "draw", np.bool_, (batch_size,))
    if "terminal_reason" in state:
        _integer_array(state, "terminal_reason", (batch_size,))
    if "max_score" in state:
        array = np.asarray(state["max_score"])
        if array.shape != (batch_size,) or not np.issubdtype(array.dtype, np.number):
            raise VectorLifecycleError("max_score must have shape [B]")


def _clear_timer_slot(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    slot: int,
) -> None:
    state["timer_active"][row, slot] = False
    state["timer_remaining_ms"][row, slot] = 0.0
    state["timer_kind"][row, slot] = TIMER_KIND_NONE
    state["timer_player"][row, slot] = TIMER_PLAYER_NONE
    state["timer_seq"][row, slot] = 0


def _match_done_for_warmdown_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player_count: int,
) -> tuple[bool, int]:
    if "max_score" not in state:
        return False, -1
    max_score = float(np.asarray(state["max_score"])[row])
    if not np.isfinite(max_score) or max_score <= 0.0:
        return False, -1

    present = np.asarray(state["present"], dtype=bool)[row, :player_count]
    present_count = int(present.sum())
    if present_count <= 0:
        return True, -1
    if player_count > 1 and present_count <= 1:
        leaders = np.flatnonzero(present)
        return True, int(leaders[0]) if leaders.size else -1

    scores = np.asarray(state["score"])[row, :player_count].astype(np.float64)
    leader_players = np.flatnonzero(present & (scores >= max_score))
    if leader_players.size == 0:
        return False, -1
    if leader_players.size == 1:
        return True, int(leader_players[0])
    ordered = sorted(
        (int(player) for player in leader_players),
        key=lambda player: float(scores[player]),
        reverse=True,
    )
    if float(scores[ordered[0]]) == float(scores[ordered[1]]):
        return False, -1
    return True, ordered[0]


def _resize_row_arena_to_present_count(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player_count: int,
) -> None:
    if "map_size" not in state:
        return
    present = np.asarray(state["present"], dtype=bool)[row, :player_count]
    present_count = int(present.sum())
    if present_count <= 0:
        return
    state["map_size"][row] = float(
        CurvyTronReferenceDefaults().arena_size_for_players(present_count)
    )


def _end_match_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    match_winner: int,
) -> None:
    state["warmdown_pending"][row] = False
    state["match_done"][row] = True
    state["started"][row] = False
    state["in_round"][row] = False
    state["world_active"][row] = False
    state["world_body_count"][row] = 0
    if "done" in state:
        state["done"][row] = True
    if "terminated" in state:
        state["terminated"][row] = True
    if "reset_pending" in state:
        state["reset_pending"][row] = True
    if "match_winner" in state:
        state["match_winner"][row] = match_winner


def _stop_active_print_managers_for_round_clear(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player_count: int,
) -> list[int]:
    stopped: list[int] = []
    row_mask = np.zeros(np.asarray(state["tick"]).shape[0], dtype=bool)
    row_mask[row] = True
    for player in range(player_count):
        if "alive" in state and not bool(state["alive"][row, player]):
            continue
        if not bool(state["print_manager_active"][row, player]):
            continue
        state["print_manager_active"][row, player] = False
        state["printing"][row, player] = False
        vector_runtime.assign_print_manager_random_distances(
            state,
            player=player,
            mask=row_mask,
        )
        state["print_manager_distance"][row, player] = 0.0
        state["print_manager_last_pos"][row, player] = 0.0
        stopped.append(player)
    return stopped


def _validate_all_present_rows(
    state: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    player_count: int,
) -> None:
    present = np.asarray(state["present"], dtype=bool)
    if not bool(present[mask].all()):
        if player_count == SUPPORTED_LIFECYCLE_PLAYER_COUNT:
            raise VectorLifecycleError(
                "1v1/no-bonus warmup reset requires both players present"
            )
        raise VectorLifecycleError(
            f"{player_count}P/no-bonus warmup reset requires all players present"
        )


def _clear_selected_round_local_arrays(
    target: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    clear_present_players_only: bool = False,
    player_count: int | None = None,
) -> None:
    if clear_present_players_only and (player_count is None or "present" not in target):
        raise VectorLifecycleError(
            "present and player_count are required to clear only present players"
        )
    for name in ("started", "in_round"):
        if name in target:
            target[name][mask] = True
    if "world_active" in target:
        target["world_active"][mask] = False
    if "world_body_count" in target:
        target["world_body_count"][mask] = 0
    _clear_player_round_array(
        target,
        "printing",
        mask,
        False,
        clear_present_players_only=clear_present_players_only,
        player_count=player_count,
    )
    _clear_player_round_array(
        target,
        "print_manager_active",
        mask,
        False,
        clear_present_players_only=clear_present_players_only,
        player_count=player_count,
    )
    _clear_player_round_array(
        target,
        "print_manager_distance",
        mask,
        0.0,
        clear_present_players_only=clear_present_players_only,
        player_count=player_count,
    )
    _clear_player_round_array(
        target,
        "print_manager_last_pos",
        mask,
        0.0,
        clear_present_players_only=clear_present_players_only,
        player_count=player_count,
    )
    if "death_count" in target:
        target["death_count"][mask] = 0
    if "death_player" in target:
        target["death_player"][mask, ...] = -1
    _clear_player_round_array(
        target,
        "round_score",
        mask,
        0,
        clear_present_players_only=clear_present_players_only,
        player_count=player_count,
    )
    for name in (
        "body_active",
        "body_pos",
        "body_radius",
        "body_owner",
        "body_num",
        "body_insert_tick",
        "body_insert_kind",
    ):
        if name not in target:
            continue
        if target[name].dtype == bool:
            target[name][mask, ...] = False
        elif np.issubdtype(target[name].dtype, np.floating):
            target[name][mask, ...] = 0.0
        else:
            target[name][mask, ...] = -1
    if "body_write_cursor" in target:
        target["body_write_cursor"][mask] = 0
    _clear_player_round_array(
        target,
        "body_count",
        mask,
        0,
        clear_present_players_only=clear_present_players_only,
        player_count=player_count,
    )


def _clear_player_round_array(
    target: Mapping[str, np.ndarray],
    name: str,
    mask: np.ndarray,
    value: bool | int | float,
    *,
    clear_present_players_only: bool,
    player_count: int | None,
) -> None:
    if name not in target:
        return
    if not clear_present_players_only:
        target[name][mask, ...] = value
        return

    present = np.asarray(target["present"], dtype=bool)
    for row in np.flatnonzero(mask):
        row_int = int(row)
        players = np.flatnonzero(present[row_int, : int(player_count)])
        if players.size:
            target[name][row_int, players, ...] = value


def _schedule_game_start_timers(
    target: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    first_warmup_ms: float,
) -> dict[str, Any]:
    timer_active = target["timer_active"]
    timer_remaining_ms = target["timer_remaining_ms"]
    timer_kind = target["timer_kind"]
    timer_player = target["timer_player"]
    timer_seq = target["timer_seq"]
    timer_overflow = target["timer_overflow"]

    rows: list[int] = []
    slots: list[int] = []
    overflow_rows: list[int] = []
    capacity = int(timer_active.shape[1])
    for row in np.flatnonzero(mask):
        row_int = int(row)
        timer_active[row_int, :] = False
        timer_remaining_ms[row_int, :] = 0.0
        timer_kind[row_int, :] = TIMER_KIND_NONE
        timer_player[row_int, :] = TIMER_PLAYER_NONE
        timer_seq[row_int, :] = 0
        timer_overflow[row_int] = False

        if capacity < 1:
            timer_overflow[row_int] = True
            overflow_rows.append(row_int)
            continue

        timer_active[row_int, 0] = True
        timer_remaining_ms[row_int, 0] = first_warmup_ms
        timer_kind[row_int, 0] = TIMER_KIND_GAME_START
        timer_player[row_int, 0] = TIMER_PLAYER_NONE
        timer_seq[row_int, 0] = 0
        rows.append(row_int)
        slots.append(0)

    return {
        "scheduled_timer_count": len(rows),
        "scheduled_timer_rows": np.asarray(rows, dtype=np.int32),
        "scheduled_timer_slots": np.asarray(slots, dtype=np.int16),
        "timer_overflow_rows": np.asarray(overflow_rows, dtype=np.int32),
        "scheduled_timer_kind": "game:start",
        "scheduled_timer_kind_code": TIMER_KIND_GAME_START,
    }


def _lifecycle_schedule_report(
    target: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    player_count: int,
    can_compose: bool,
) -> dict[str, Any]:
    rows = np.flatnonzero(mask).astype(np.int32)
    present_lifecycle_arrays = [
        name for name in _OPTIONAL_ROUND_LIFECYCLE_ARRAYS if name in target
    ]
    missing_lifecycle_arrays = [
        name for name in _OPTIONAL_ROUND_LIFECYCLE_ARRAYS if name not in target
    ]
    present_timer_arrays = [
        name for name in _OPTIONAL_DELAYED_START_TIMER_ARRAYS if name in target
    ]
    missing_timer_arrays = [
        name for name in _OPTIONAL_DELAYED_START_TIMER_ARRAYS if name not in target
    ]

    lifecycle_reasons = _lifecycle_unsupported_reasons(
        can_compose=can_compose,
        player_count=player_count,
    )
    timer_reasons = list(lifecycle_reasons)
    if missing_timer_arrays:
        timer_reasons.append("missing_delayed_start_timer_arrays")

    return {
        "surface": RESET_SPAWN_LIFECYCLE_SURFACE,
        "full_lifecycle": False,
        "player_count": int(player_count),
        "supported_player_count": SUPPORTED_LIFECYCLE_PLAYER_COUNT,
        "rows": rows.copy(),
        "world_flags": {
            "requested_arrays": list(_OPTIONAL_ROUND_LIFECYCLE_ARRAYS),
            "present_arrays": present_lifecycle_arrays,
            "missing_arrays": missing_lifecycle_arrays,
            "applied": False,
            "applied_arrays": [],
            "rows": np.asarray([], dtype=np.int32),
            "values": {
                "started": True,
                "in_round": True,
                "world_active": False,
                "world_body_count": 0,
            },
            "unsupported_reasons": lifecycle_reasons,
        },
        "delayed_start_timers": {
            "required_arrays": list(_OPTIONAL_DELAYED_START_TIMER_ARRAYS),
            "present_arrays": present_timer_arrays,
            "missing_arrays": missing_timer_arrays,
            "scheduled": False,
            "scheduled_timer_count": 0,
            "scheduled_timer_rows": np.asarray([], dtype=np.int32),
            "scheduled_timer_slots": np.asarray([], dtype=np.int16),
            "scheduled_timer_players": np.asarray([], dtype=np.int16),
            "timer_overflow_rows": np.asarray([], dtype=np.int32),
            "timer_kind": "print_manager_start",
            "timer_kind_code": TIMER_KIND_PRINT_MANAGER_START,
            "delay_ms": SOURCE_TRAIL_START_DELAY_MS,
            "unsupported_reasons": timer_reasons,
        },
    }


def _lifecycle_unsupported_reasons(
    *,
    can_compose: bool,
    player_count: int,
) -> list[str]:
    reasons: list[str] = []
    if not can_compose:
        reasons.append("reset_spawn_arrays_unavailable")
    if player_count != SUPPORTED_LIFECYCLE_PLAYER_COUNT:
        reasons.append("player_count_not_2")
    return reasons


def _validate_optional_lifecycle_arrays(
    target: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    player_count: int,
    schedule_info: Mapping[str, Any],
) -> None:
    batch_size = mask.shape[0]
    for name in schedule_info["world_flags"]["present_arrays"]:
        if name in ("started", "in_round", "world_active"):
            _typed_array(target, name, np.bool_, (batch_size,))
        elif name == "world_body_count":
            _typed_array(target, name, np.int32, (batch_size,))

    timer_info = schedule_info["delayed_start_timers"]
    if timer_info["unsupported_reasons"]:
        return

    capacity_shape = _typed_array(
        target,
        "timer_active",
        np.bool_,
        ndim=2,
        batch_size=batch_size,
    ).shape
    _typed_array(target, "timer_remaining_ms", np.float64, capacity_shape)
    _typed_array(target, "timer_kind", np.int16, capacity_shape)
    _typed_array(target, "timer_player", np.int16, capacity_shape)
    _typed_array(target, "timer_seq", np.int32, capacity_shape)
    _typed_array(target, "timer_overflow", np.bool_, (batch_size,))

    present = _typed_array(
        target,
        "present",
        np.bool_,
        (batch_size, player_count),
    )
    if player_count == SUPPORTED_LIFECYCLE_PLAYER_COUNT and present.ndim != 2:
        raise VectorLifecycleError("present must be a bool array with shape [B,P]")


def _apply_optional_lifecycle_schedule(
    target: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    player_count: int,
    schedule_info: dict[str, Any],
) -> dict[str, Any]:
    if not schedule_info["world_flags"]["unsupported_reasons"]:
        _apply_optional_world_flags(target, mask, schedule_info=schedule_info)
    if not schedule_info["delayed_start_timers"]["unsupported_reasons"]:
        _schedule_optional_delayed_start_timers(
            target,
            mask,
            player_count=player_count,
            schedule_info=schedule_info,
        )
    return schedule_info


def _apply_optional_world_flags(
    target: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    schedule_info: dict[str, Any],
) -> None:
    applied_arrays: list[str] = []
    if "started" in target:
        target["started"][mask] = True
        applied_arrays.append("started")
    if "in_round" in target:
        target["in_round"][mask] = True
        applied_arrays.append("in_round")
    if "world_active" in target:
        target["world_active"][mask] = False
        applied_arrays.append("world_active")
    if "world_body_count" in target:
        target["world_body_count"][mask] = 0
        applied_arrays.append("world_body_count")

    schedule_info["world_flags"].update(
        {
            "applied": bool(applied_arrays),
            "applied_arrays": applied_arrays,
            "rows": np.flatnonzero(mask).astype(np.int32),
        }
    )


def _schedule_optional_delayed_start_timers(
    target: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    player_count: int,
    schedule_info: dict[str, Any],
) -> None:
    timer_active = target["timer_active"]
    timer_remaining_ms = target["timer_remaining_ms"]
    timer_kind = target["timer_kind"]
    timer_player = target["timer_player"]
    timer_seq = target["timer_seq"]
    timer_overflow = target["timer_overflow"]
    present = target["present"]

    rows: list[int] = []
    slots: list[int] = []
    players: list[int] = []
    overflow_rows: list[int] = []
    capacity = int(timer_active.shape[1])

    for row in np.flatnonzero(mask):
        row_int = int(row)
        timer_active[row_int, :] = False
        timer_remaining_ms[row_int, :] = 0.0
        timer_kind[row_int, :] = TIMER_KIND_NONE
        timer_player[row_int, :] = TIMER_PLAYER_NONE
        timer_seq[row_int, :] = 0
        timer_overflow[row_int] = False

        row_players = [
            player
            for player in range(player_count - 1, -1, -1)
            if bool(present[row_int, player])
        ]
        if len(row_players) > capacity:
            timer_overflow[row_int] = True
            overflow_rows.append(row_int)
            continue

        for slot, player in enumerate(row_players):
            timer_active[row_int, slot] = True
            timer_remaining_ms[row_int, slot] = SOURCE_TRAIL_START_DELAY_MS
            timer_kind[row_int, slot] = TIMER_KIND_PRINT_MANAGER_START
            timer_player[row_int, slot] = player
            timer_seq[row_int, slot] = slot
            rows.append(row_int)
            slots.append(slot)
            players.append(player)

    schedule_info["delayed_start_timers"].update(
        {
            "scheduled": bool(rows),
            "scheduled_timer_count": len(rows),
            "scheduled_timer_rows": np.asarray(rows, dtype=np.int32),
            "scheduled_timer_slots": np.asarray(slots, dtype=np.int16),
            "scheduled_timer_players": np.asarray(players, dtype=np.int16),
            "timer_overflow_rows": np.asarray(overflow_rows, dtype=np.int32),
        }
    )


def _typed_array(
    state: Mapping[str, np.ndarray],
    name: str,
    dtype: Any,
    shape: tuple[int, ...] | None = None,
    *,
    ndim: int | None = None,
    batch_size: int | None = None,
) -> np.ndarray:
    array = np.asarray(state[name])
    expected_dtype = np.dtype(dtype)
    if array.dtype != expected_dtype:
        raise VectorLifecycleError(
            f"{name} must be a {expected_dtype.name} array for lifecycle scheduling"
        )
    if shape is not None and array.shape != shape:
        raise VectorLifecycleError(f"{name} must have shape {_shape_phrase(shape)}")
    if ndim is not None and array.ndim != ndim:
        raise VectorLifecycleError(f"{name} must have {ndim} dimensions")
    if batch_size is not None and (array.ndim < 1 or array.shape[0] != batch_size):
        raise VectorLifecycleError(f"{name} must have leading shape [B]")
    return array


def _numeric_array(
    state: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(state[name])
    if array.shape != shape or not np.issubdtype(array.dtype, np.number):
        raise VectorLifecycleError(f"{name} must have shape {_shape_phrase(shape)}")
    return array


def _integer_array(
    state: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(state[name])
    if array.shape != shape or not np.issubdtype(array.dtype, np.integer):
        raise VectorLifecycleError(f"{name} must have shape {_shape_phrase(shape)}")
    return array


def _shape_phrase(shape: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(part) for part in shape) + "]"


__all__ = [
    "RESET_SPAWN_INFO_SCHEMA_ID",
    "RESET_SPAWN_LIFECYCLE_SURFACE",
    "RESET_SPAWN_SURFACE",
    "RESET_SPAWN_WARMUP_NO_BONUS_INFO_SCHEMA_ID",
    "RESET_SPAWN_WARMUP_NO_BONUS_SURFACE",
    "RESET_SPAWN_WARMUP_INFO_SCHEMA_ID",
    "RESET_SPAWN_WARMUP_SURFACE",
    "SOURCE_ROUND_WARMUP_MS",
    "SOURCE_ROUND_WARMDOWN_MS",
    "SOURCE_TRAIL_START_DELAY_MS",
    "SUPPORTED_LIFECYCLE_PLAYER_COUNT",
    "SUPPORTED_NO_BONUS_WARMUP_PLAYER_COUNTS",
    "TIMER_KIND_GAME_START",
    "TIMER_KIND_NONE",
    "TIMER_KIND_PRINT_MANAGER_START",
    "TIMER_KIND_WARMDOWN_END",
    "TIMER_PLAYER_NONE",
    "VectorLifecycleError",
    "WARMDOWN_ADVANCE_NO_BONUS_INFO_SCHEMA_ID",
    "WARMDOWN_ADVANCE_NO_BONUS_SURFACE",
    "WARMUP_START_STEP_1V1_NO_BONUS_INFO_SCHEMA_ID",
    "WARMUP_START_STEP_1V1_NO_BONUS_SURFACE",
    "advance_warmdown_no_bonus_rows",
    "reset_and_spawn_round_rows",
    "reset_spawn_warmup_no_bonus_rows",
    "reset_spawn_warmup_1v1_no_bonus_rows",
    "reset_spawn_array_report",
    "run_warmup_start_step_1v1_no_bonus_rows",
]
