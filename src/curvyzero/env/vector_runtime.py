"""Production-facing vector step contract.

The source-ordered NumPy batch step now lives here. The scripts still own
fixture loading, source/common-trace comparison, timing reports, and benchmark
CLI glue.

This module is the runtime-owned boundary for the batched NumPy transition. It
validates the unified ``step_many`` batch shape and owns reusable runtime
accounting helpers without importing script or benchmark glue.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import time
from typing import Any, TypedDict

import numpy as np

from curvyzero.env import vector_spawn
from curvyzero.env.vector_reset import (
    TERMINAL_REASON_NONE,
    TERMINAL_REASON_ALL_DEAD_DRAW,
    TERMINAL_REASON_SURVIVOR_WIN,
)


EVENT_MODE_DEBUG = "debug-event"
EVENT_MODE_NONE = "no-event"
EVENT_MODES = frozenset((EVENT_MODE_DEBUG, EVENT_MODE_NONE))
DEATH_MODE_NORMAL = "normal"
DEATH_MODE_PROFILE_NO_DEATH = "profile_no_death"
DEATH_MODES = frozenset((DEATH_MODE_NORMAL, DEATH_MODE_PROFILE_NO_DEATH))

PRINT_MANAGER_RANDOM_HALF_PRINT_DISTANCE = 39.0
PRINT_MANAGER_RANDOM_HALF_HOLE_DISTANCE = 5.25
SOURCE_TRAIL_START_DELAY_MS = 3_000.0
SOURCE_ROUND_WARMDOWN_MS = 5_000.0
EVENT_NONE = 0
EVENT_POSITION = 1
EVENT_POINT = 2
EVENT_DIE = 3
EVENT_SCORE_ROUND = 4
EVENT_SCORE = 5
EVENT_ROUND_END = 6
EVENT_PROPERTY = 7
EVENT_BONUS_CLEAR = 8
EVENT_BONUS_STACK = 9
EVENT_CLEAR = 10
EVENT_BORDERLESS = 11
EVENT_BONUS_POP = 12
PROPERTY_PRINTING = 1
PROPERTY_RADIUS = 2
PROPERTY_VELOCITY = 3
PROPERTY_INVERSE = 4
PROPERTY_INVINCIBLE = 5
PROPERTY_COLOR = 6
BONUS_TYPE_NONE = 0
BONUS_TYPE_SELF_SMALL = 1
BONUS_TYPE_SELF_SLOW = 2
BONUS_TYPE_SELF_FAST = 3
BONUS_TYPE_SELF_MASTER = 4
BONUS_TYPE_ENEMY_SLOW = 5
BONUS_TYPE_ENEMY_FAST = 6
BONUS_TYPE_ENEMY_BIG = 7
BONUS_TYPE_ENEMY_INVERSE = 8
BONUS_TYPE_ENEMY_STRAIGHT_ANGLE = 9
BONUS_TYPE_GAME_BORDERLESS = 10
BONUS_TYPE_ALL_COLOR = 11
BONUS_TYPE_GAME_CLEAR = 12
SOURCE_DEFAULT_BONUS_TYPE_CODES = (
    BONUS_TYPE_SELF_SMALL,
    BONUS_TYPE_SELF_SLOW,
    BONUS_TYPE_SELF_FAST,
    BONUS_TYPE_SELF_MASTER,
    BONUS_TYPE_ENEMY_SLOW,
    BONUS_TYPE_ENEMY_FAST,
    BONUS_TYPE_ENEMY_BIG,
    BONUS_TYPE_ENEMY_INVERSE,
    BONUS_TYPE_ENEMY_STRAIGHT_ANGLE,
    BONUS_TYPE_GAME_BORDERLESS,
    BONUS_TYPE_ALL_COLOR,
    BONUS_TYPE_GAME_CLEAR,
)
BONUS_TYPE_NAME_BY_CODE = (
    "None",
    "BonusSelfSmall",
    "BonusSelfSlow",
    "BonusSelfFast",
    "BonusSelfMaster",
    "BonusEnemySlow",
    "BonusEnemyFast",
    "BonusEnemyBig",
    "BonusEnemyInverse",
    "BonusEnemyStraightAngle",
    "BonusGameBorderless",
    "BonusAllColor",
    "BonusGameClear",
)
BONUS_TYPE_SELECTION_METADATA_SCHEMA_ID = "curvyzero_vector_bonus_type_selection_metadata/v1"
BONUS_TYPE_SELECTION_METADATA_SURFACE = "select_bonus_type_metadata"
BONUS_SPAWN_CAP_METADATA_SCHEMA_ID = "curvyzero_vector_bonus_spawn_cap_metadata/v1"
BONUS_SPAWN_CAP_METADATA_SURFACE = "bonus_spawn_cap_metadata"
BONUS_SPAWN_DUE_ROWS_SCHEMA_ID = "curvyzero_vector_bonus_spawn_due_rows/v1"
BONUS_SPAWN_DUE_ROWS_SURFACE = "bonus_spawn_due_rows"
SOURCE_MAX_ACTIVE_BONUSES = 20
SOURCE_BONUS_RADIUS = 3.0
SOURCE_BONUS_POSITION_MARGIN_FRACTION = 0.01
SOURCE_AVATAR_SPEED = 16.0
SOURCE_AVATAR_ANGULAR_VELOCITY_PER_MS = 2.8 / 1000.0
SOURCE_STRAIGHT_ANGLE_RADIANS = float(np.pi / 2.0)
SOURCE_DEFAULT_BONUS_DURATION_MS = 5_000
BONUS_SELF_SMALL_DURATION_MS = 7_500
BONUS_SELF_SMALL_RADIUS_POWER = -1
BONUS_SELF_SLOW_DURATION_MS = SOURCE_DEFAULT_BONUS_DURATION_MS
BONUS_SELF_SLOW_VELOCITY_DELTA = -SOURCE_AVATAR_SPEED / 2.0
BONUS_SELF_FAST_DURATION_MS = 4_000
BONUS_SELF_FAST_VELOCITY_DELTA = 0.75 * SOURCE_AVATAR_SPEED
BONUS_SELF_MASTER_DURATION_MS = 7_500
BONUS_SELF_MASTER_INVINCIBLE_DELTA = 1
BONUS_SELF_MASTER_PRINTING_DELTA = -1
BONUS_ENEMY_SLOW_DURATION_MS = SOURCE_DEFAULT_BONUS_DURATION_MS
BONUS_ENEMY_SLOW_VELOCITY_DELTA = -SOURCE_AVATAR_SPEED / 2.0
BONUS_ENEMY_FAST_DURATION_MS = 6_000
BONUS_ENEMY_FAST_VELOCITY_DELTA = 0.75 * SOURCE_AVATAR_SPEED
BONUS_ENEMY_BIG_DURATION_MS = 7_500
BONUS_ENEMY_BIG_RADIUS_POWER = 1
BONUS_ENEMY_INVERSE_DURATION_MS = SOURCE_DEFAULT_BONUS_DURATION_MS
BONUS_ENEMY_INVERSE_DELTA = 1
BONUS_ENEMY_STRAIGHT_ANGLE_DURATION_MS = SOURCE_DEFAULT_BONUS_DURATION_MS
BONUS_ALL_COLOR_DURATION_MS = 7_500
BONUS_GAME_BORDERLESS_DURATION_MS = 10_000
BONUS_GAME_BORDERLESS_STACK_VALUE = 1
BONUS_STACK_METHOD_REMOVE = 0
BONUS_STACK_METHOD_ADD = 1
SOURCE_AVATAR_RADIUS = 0.6
BONUS_AVATAR_STACK_TYPES = frozenset(
    (
        BONUS_TYPE_SELF_SMALL,
        BONUS_TYPE_SELF_SLOW,
        BONUS_TYPE_SELF_FAST,
        BONUS_TYPE_SELF_MASTER,
        BONUS_TYPE_ENEMY_SLOW,
        BONUS_TYPE_ENEMY_FAST,
        BONUS_TYPE_ENEMY_BIG,
        BONUS_TYPE_ENEMY_INVERSE,
        BONUS_TYPE_ENEMY_STRAIGHT_ANGLE,
        BONUS_TYPE_ALL_COLOR,
    )
)
BONUS_SELF_STACK_TYPES = frozenset(
    (
        BONUS_TYPE_SELF_SMALL,
        BONUS_TYPE_SELF_SLOW,
        BONUS_TYPE_SELF_FAST,
        BONUS_TYPE_SELF_MASTER,
    )
)
BONUS_ENEMY_STACK_TYPES = frozenset(
    (
        BONUS_TYPE_ENEMY_SLOW,
        BONUS_TYPE_ENEMY_FAST,
        BONUS_TYPE_ENEMY_BIG,
        BONUS_TYPE_ENEMY_INVERSE,
        BONUS_TYPE_ENEMY_STRAIGHT_ANGLE,
    )
)
BONUS_ALL_STACK_TYPES = frozenset((BONUS_TYPE_ALL_COLOR,))
BONUS_AVATAR_STACK_DURATION_MS_BY_TYPE = {
    BONUS_TYPE_SELF_SMALL: BONUS_SELF_SMALL_DURATION_MS,
    BONUS_TYPE_SELF_SLOW: BONUS_SELF_SLOW_DURATION_MS,
    BONUS_TYPE_SELF_FAST: BONUS_SELF_FAST_DURATION_MS,
    BONUS_TYPE_SELF_MASTER: BONUS_SELF_MASTER_DURATION_MS,
    BONUS_TYPE_ENEMY_SLOW: BONUS_ENEMY_SLOW_DURATION_MS,
    BONUS_TYPE_ENEMY_FAST: BONUS_ENEMY_FAST_DURATION_MS,
    BONUS_TYPE_ENEMY_BIG: BONUS_ENEMY_BIG_DURATION_MS,
    BONUS_TYPE_ENEMY_INVERSE: BONUS_ENEMY_INVERSE_DURATION_MS,
    BONUS_TYPE_ENEMY_STRAIGHT_ANGLE: BONUS_ENEMY_STRAIGHT_ANGLE_DURATION_MS,
    BONUS_TYPE_ALL_COLOR: BONUS_ALL_COLOR_DURATION_MS,
}
BONUS_CATCH_COUNTER_BY_TYPE = {
    BONUS_TYPE_SELF_SMALL: "bonus_self_small_catches",
    BONUS_TYPE_SELF_SLOW: "bonus_self_slow_catches",
    BONUS_TYPE_SELF_FAST: "bonus_self_fast_catches",
    BONUS_TYPE_SELF_MASTER: "bonus_self_master_catches",
    BONUS_TYPE_ENEMY_SLOW: "bonus_enemy_slow_catches",
    BONUS_TYPE_ENEMY_FAST: "bonus_enemy_fast_catches",
    BONUS_TYPE_ENEMY_BIG: "bonus_enemy_big_catches",
    BONUS_TYPE_ENEMY_INVERSE: "bonus_enemy_inverse_catches",
    BONUS_TYPE_ENEMY_STRAIGHT_ANGLE: "bonus_enemy_straight_angle_catches",
    BONUS_TYPE_GAME_BORDERLESS: "bonus_game_borderless_catches",
    BONUS_TYPE_ALL_COLOR: "bonus_all_color_catches",
    BONUS_TYPE_GAME_CLEAR: "bonus_game_clear_catches",
}
BONUS_EXPIRY_COUNTER_BY_TYPE = {
    BONUS_TYPE_SELF_SMALL: "bonus_self_small_expiries",
    BONUS_TYPE_SELF_SLOW: "bonus_self_slow_expiries",
    BONUS_TYPE_SELF_FAST: "bonus_self_fast_expiries",
    BONUS_TYPE_SELF_MASTER: "bonus_self_master_expiries",
    BONUS_TYPE_ENEMY_SLOW: "bonus_enemy_slow_expiries",
    BONUS_TYPE_ENEMY_FAST: "bonus_enemy_fast_expiries",
    BONUS_TYPE_ENEMY_BIG: "bonus_enemy_big_expiries",
    BONUS_TYPE_ENEMY_INVERSE: "bonus_enemy_inverse_expiries",
    BONUS_TYPE_ENEMY_STRAIGHT_ANGLE: "bonus_enemy_straight_angle_expiries",
    BONUS_TYPE_ALL_COLOR: "bonus_all_color_expiries",
}


@dataclass(frozen=True, slots=True)
class BonusRuntimeEffect:
    target_group: str
    catch_counter: str
    expiry_counter: str | None = None
    duration_ms: int = 0
    radius_power: int = 0
    velocity_delta: float = 0.0
    inverse_delta: int = 0
    angular_velocity_per_ms: float = 0.0
    invincible_delta: int = 0
    printing_delta: int = 0
    rotates_color: bool = False
    borderless: int = 0


BONUS_TARGET_SELF = "self"
BONUS_TARGET_ENEMY = "enemy"
BONUS_TARGET_ALL = "all"
BONUS_TARGET_GAME = "game"
BONUS_TARGET_CLEAR = "clear"
BONUS_RUNTIME_EFFECT_BY_TYPE = {
    BONUS_TYPE_SELF_SMALL: BonusRuntimeEffect(
        target_group=BONUS_TARGET_SELF,
        catch_counter=BONUS_CATCH_COUNTER_BY_TYPE[BONUS_TYPE_SELF_SMALL],
        expiry_counter=BONUS_EXPIRY_COUNTER_BY_TYPE[BONUS_TYPE_SELF_SMALL],
        duration_ms=BONUS_SELF_SMALL_DURATION_MS,
        radius_power=BONUS_SELF_SMALL_RADIUS_POWER,
    ),
    BONUS_TYPE_SELF_SLOW: BonusRuntimeEffect(
        target_group=BONUS_TARGET_SELF,
        catch_counter=BONUS_CATCH_COUNTER_BY_TYPE[BONUS_TYPE_SELF_SLOW],
        expiry_counter=BONUS_EXPIRY_COUNTER_BY_TYPE[BONUS_TYPE_SELF_SLOW],
        duration_ms=BONUS_SELF_SLOW_DURATION_MS,
        velocity_delta=BONUS_SELF_SLOW_VELOCITY_DELTA,
    ),
    BONUS_TYPE_SELF_FAST: BonusRuntimeEffect(
        target_group=BONUS_TARGET_SELF,
        catch_counter=BONUS_CATCH_COUNTER_BY_TYPE[BONUS_TYPE_SELF_FAST],
        expiry_counter=BONUS_EXPIRY_COUNTER_BY_TYPE[BONUS_TYPE_SELF_FAST],
        duration_ms=BONUS_SELF_FAST_DURATION_MS,
        velocity_delta=BONUS_SELF_FAST_VELOCITY_DELTA,
    ),
    BONUS_TYPE_SELF_MASTER: BonusRuntimeEffect(
        target_group=BONUS_TARGET_SELF,
        catch_counter=BONUS_CATCH_COUNTER_BY_TYPE[BONUS_TYPE_SELF_MASTER],
        expiry_counter=BONUS_EXPIRY_COUNTER_BY_TYPE[BONUS_TYPE_SELF_MASTER],
        duration_ms=BONUS_SELF_MASTER_DURATION_MS,
        invincible_delta=BONUS_SELF_MASTER_INVINCIBLE_DELTA,
        printing_delta=BONUS_SELF_MASTER_PRINTING_DELTA,
    ),
    BONUS_TYPE_ENEMY_SLOW: BonusRuntimeEffect(
        target_group=BONUS_TARGET_ENEMY,
        catch_counter=BONUS_CATCH_COUNTER_BY_TYPE[BONUS_TYPE_ENEMY_SLOW],
        expiry_counter=BONUS_EXPIRY_COUNTER_BY_TYPE[BONUS_TYPE_ENEMY_SLOW],
        duration_ms=BONUS_ENEMY_SLOW_DURATION_MS,
        velocity_delta=BONUS_ENEMY_SLOW_VELOCITY_DELTA,
    ),
    BONUS_TYPE_ENEMY_FAST: BonusRuntimeEffect(
        target_group=BONUS_TARGET_ENEMY,
        catch_counter=BONUS_CATCH_COUNTER_BY_TYPE[BONUS_TYPE_ENEMY_FAST],
        expiry_counter=BONUS_EXPIRY_COUNTER_BY_TYPE[BONUS_TYPE_ENEMY_FAST],
        duration_ms=BONUS_ENEMY_FAST_DURATION_MS,
        velocity_delta=BONUS_ENEMY_FAST_VELOCITY_DELTA,
    ),
    BONUS_TYPE_ENEMY_BIG: BonusRuntimeEffect(
        target_group=BONUS_TARGET_ENEMY,
        catch_counter=BONUS_CATCH_COUNTER_BY_TYPE[BONUS_TYPE_ENEMY_BIG],
        expiry_counter=BONUS_EXPIRY_COUNTER_BY_TYPE[BONUS_TYPE_ENEMY_BIG],
        duration_ms=BONUS_ENEMY_BIG_DURATION_MS,
        radius_power=BONUS_ENEMY_BIG_RADIUS_POWER,
    ),
    BONUS_TYPE_ENEMY_INVERSE: BonusRuntimeEffect(
        target_group=BONUS_TARGET_ENEMY,
        catch_counter=BONUS_CATCH_COUNTER_BY_TYPE[BONUS_TYPE_ENEMY_INVERSE],
        expiry_counter=BONUS_EXPIRY_COUNTER_BY_TYPE[BONUS_TYPE_ENEMY_INVERSE],
        duration_ms=BONUS_ENEMY_INVERSE_DURATION_MS,
        inverse_delta=BONUS_ENEMY_INVERSE_DELTA,
    ),
    BONUS_TYPE_ENEMY_STRAIGHT_ANGLE: BonusRuntimeEffect(
        target_group=BONUS_TARGET_ENEMY,
        catch_counter=BONUS_CATCH_COUNTER_BY_TYPE[BONUS_TYPE_ENEMY_STRAIGHT_ANGLE],
        expiry_counter=BONUS_EXPIRY_COUNTER_BY_TYPE[BONUS_TYPE_ENEMY_STRAIGHT_ANGLE],
        duration_ms=BONUS_ENEMY_STRAIGHT_ANGLE_DURATION_MS,
        angular_velocity_per_ms=SOURCE_STRAIGHT_ANGLE_RADIANS,
    ),
    BONUS_TYPE_GAME_BORDERLESS: BonusRuntimeEffect(
        target_group=BONUS_TARGET_GAME,
        catch_counter=BONUS_CATCH_COUNTER_BY_TYPE[BONUS_TYPE_GAME_BORDERLESS],
        expiry_counter="bonus_game_borderless_expiries",
        duration_ms=BONUS_GAME_BORDERLESS_DURATION_MS,
        borderless=BONUS_GAME_BORDERLESS_STACK_VALUE,
    ),
    BONUS_TYPE_ALL_COLOR: BonusRuntimeEffect(
        target_group=BONUS_TARGET_ALL,
        catch_counter=BONUS_CATCH_COUNTER_BY_TYPE[BONUS_TYPE_ALL_COLOR],
        expiry_counter=BONUS_EXPIRY_COUNTER_BY_TYPE[BONUS_TYPE_ALL_COLOR],
        duration_ms=BONUS_ALL_COLOR_DURATION_MS,
        rotates_color=True,
    ),
    BONUS_TYPE_GAME_CLEAR: BonusRuntimeEffect(
        target_group=BONUS_TARGET_CLEAR,
        catch_counter=BONUS_CATCH_COUNTER_BY_TYPE[BONUS_TYPE_GAME_CLEAR],
    ),
}
BONUS_RUNTIME_CATCH_COUNTER_NAMES = tuple(
    effect.catch_counter for effect in BONUS_RUNTIME_EFFECT_BY_TYPE.values()
)
BONUS_RUNTIME_EXPIRY_COUNTER_NAMES = tuple(
    effect.expiry_counter
    for effect in BONUS_RUNTIME_EFFECT_BY_TYPE.values()
    if effect.expiry_counter is not None
)
TIMER_KIND_NONE = 0
TIMER_KIND_PRINT_MANAGER_START = 1
TIMER_KIND_GAME_START = 2
TIMER_KIND_WARMDOWN_END = 3
TIMER_PLAYER_NONE = -1
BODY_KIND_NORMAL = 0
BODY_KIND_IMPORTANT = 1
BODY_KIND_DEATH = 2
DEATH_CAUSE_NONE = 0
DEATH_CAUSE_WALL = 1
DEATH_CAUSE_OWN_TRAIL = 2
DEATH_CAUSE_OPPONENT_TRAIL = 3
DEATH_CAUSE_BODY_UNKNOWN = 4
DEATH_CAUSE_NAMES = (
    "none",
    "wall",
    "own_trail",
    "opponent_trail",
    "body_unknown",
)
WARMUP_TIMER_ADVANCE_NO_BONUS_INFO_SCHEMA_ID = (
    "curvyzero_vector_warmup_no_bonus_timer_advance_info/v1"
)
WARMUP_TIMER_ADVANCE_INFO_SCHEMA_ID = "curvyzero_vector_warmup_1v1_no_bonus_timer_advance_info/v1"
WARMDOWN_TIMER_ADVANCE_NO_BONUS_INFO_SCHEMA_ID = (
    "curvyzero_vector_warmdown_no_bonus_timer_advance_info/v1"
)
WARMUP_TIMER_ADVANCE_NO_BONUS_SURFACE = "advance_warmup_no_bonus_timers"
WARMUP_TIMER_ADVANCE_SURFACE = "advance_warmup_1v1_no_bonus_timers"
WARMDOWN_TIMER_ADVANCE_NO_BONUS_SURFACE = "advance_warmdown_no_bonus_timers"
SUPPORTED_WARMUP_PLAYER_COUNT = 2
SUPPORTED_NO_BONUS_WARMUP_PLAYER_COUNTS = (2, 3, 4)

PRINT_MANAGER_MODES = frozenset(
    (
        "none",
        "no_toggle_control",
        "delayed_start",
        "toggle",
        "natural_toggle",
        "death_stop",
    )
)

STEP_COUNTER_NAMES = (
    "movement_updates",
    "events_emitted",
    "event_overflow_attempts",
    "normal_points_inserted",
    "death_points_inserted",
    "body_hits",
    "body_scan_slots",
    "body_candidates",
    "body_overflow_attempts",
    "borderless_wraps",
    "normal_wall_deaths",
    "terminal_score_rows",
    "print_manager_no_toggle_updates",
    "print_manager_toggle_updates",
    "print_manager_toggle_rows_unhandled",
    "print_manager_visual_clears",
    "print_manager_death_stops",
    "print_manager_death_stop_points",
    "print_manager_death_stop_visual_clears",
    "pre_step_timer_advances",
    "pre_step_timer_fires",
    "print_manager_delayed_start_fires",
    "print_manager_delayed_start_points",
    "bonus_self_small_catches",
    "bonus_self_slow_catches",
    "bonus_self_fast_catches",
    "bonus_self_master_catches",
    "bonus_enemy_slow_catches",
    "bonus_enemy_fast_catches",
    "bonus_enemy_big_catches",
    "bonus_enemy_inverse_catches",
    "bonus_enemy_straight_angle_catches",
    "bonus_game_clear_catches",
    "bonus_game_borderless_catches",
    "bonus_all_color_catches",
    "bonus_self_small_expiries",
    "bonus_self_slow_expiries",
    "bonus_self_fast_expiries",
    "bonus_self_master_expiries",
    "bonus_enemy_slow_expiries",
    "bonus_enemy_fast_expiries",
    "bonus_enemy_big_expiries",
    "bonus_enemy_inverse_expiries",
    "bonus_enemy_straight_angle_expiries",
    "bonus_game_borderless_expiries",
    "bonus_all_color_expiries",
    "bonus_stack_appends",
    "random_tape_draws",
    "random_tape_exhaustions",
)


class VectorRuntimeError(ValueError):
    """Raised when vector runtime inputs do not match the step contract."""


class VectorStepCounters(TypedDict):
    movement_updates: int
    events_emitted: int
    event_overflow_attempts: int
    normal_points_inserted: int
    death_points_inserted: int
    body_hits: int
    body_scan_slots: int
    body_candidates: int
    body_overflow_attempts: int
    borderless_wraps: int
    normal_wall_deaths: int
    terminal_score_rows: int
    print_manager_no_toggle_updates: int
    print_manager_toggle_updates: int
    print_manager_toggle_rows_unhandled: int
    print_manager_visual_clears: int
    print_manager_death_stops: int
    print_manager_death_stop_points: int
    print_manager_death_stop_visual_clears: int
    pre_step_timer_advances: int
    pre_step_timer_fires: int
    print_manager_delayed_start_fires: int
    print_manager_delayed_start_points: int
    bonus_self_small_catches: int
    bonus_self_slow_catches: int
    bonus_self_fast_catches: int
    bonus_self_master_catches: int
    bonus_enemy_slow_catches: int
    bonus_enemy_fast_catches: int
    bonus_enemy_big_catches: int
    bonus_enemy_inverse_catches: int
    bonus_enemy_straight_angle_catches: int
    bonus_game_clear_catches: int
    bonus_game_borderless_catches: int
    bonus_all_color_catches: int
    bonus_self_small_expiries: int
    bonus_self_slow_expiries: int
    bonus_self_fast_expiries: int
    bonus_self_master_expiries: int
    bonus_enemy_slow_expiries: int
    bonus_enemy_fast_expiries: int
    bonus_enemy_big_expiries: int
    bonus_enemy_inverse_expiries: int
    bonus_enemy_straight_angle_expiries: int
    bonus_game_borderless_expiries: int
    bonus_all_color_expiries: int
    bonus_stack_appends: int
    random_tape_draws: int
    random_tape_exhaustions: int


@dataclass(frozen=True, slots=True)
class VectorStepInput:
    """One batched source-action tick for rows in ``state``.

    ``step_many`` uses the batch shape from ``state["tick"]``. A single row is
    represented as ``B == 1`` rather than by a separate scalar API.
    """

    state: Mapping[str, np.ndarray]
    step_ms: Any
    source_moves: Any
    player_count: int
    print_manager_mode: Any | None = None
    event_mode: str = EVENT_MODE_DEBUG
    timer_advance_ms: Any | None = None
    death_mode: str = DEATH_MODE_NORMAL
    death_immunity_mask: Any | None = None
    disabled_player_mask: Any | None = None

    @classmethod
    def from_mapping(
        cls,
        state: Mapping[str, np.ndarray],
        prepared_batch: Mapping[str, Any],
        *,
        event_mode: str = EVENT_MODE_DEBUG,
    ) -> "VectorStepInput":
        """Normalize a fixture-style prepared batch into the runtime contract."""

        return prepare_step_input(
            state,
            prepared_batch,
            event_mode=event_mode,
        )


def step_many(step_input: VectorStepInput) -> VectorStepCounters:
    """Run one source-ordered in-place array tick for B stacked rows."""

    validate_step_input(step_input)
    return _step_many_kernel(step_input)


def _step_many_kernel(
    step_input: VectorStepInput,
    *,
    phase_timers: dict[str, float] | None = None,
) -> VectorStepCounters:
    """Run the extracted source-ordered batched transition kernel."""

    events_enabled = _events_enabled(step_input.event_mode)
    state = step_input.state
    player_count = step_input.player_count
    step_ms = np.asarray(step_input.step_ms, dtype=np.float64)
    moves = np.asarray(step_input.source_moves, dtype=np.int8)
    row_count = _state_array(state, "tick").shape[0]
    counters = empty_step_counters()
    random_tape_draw_count_before = snapshot_random_tape_counters(state)
    bonus_arrays = _optional_bonus_arrays(
        state,
        row_count=row_count,
        player_count=player_count,
    )
    death_enabled = step_input.death_mode == DEATH_MODE_NORMAL
    death_immunity_mask = _death_immunity_mask(
        step_input.death_immunity_mask,
        row_count=row_count,
        player_count=player_count,
    )
    disabled_player_mask = _disabled_player_mask(
        step_input.disabled_player_mask,
        row_count=row_count,
        player_count=player_count,
    )

    print_manager_mode = step_input.print_manager_mode
    if print_manager_mode is None:
        print_manager_mode = np.full(row_count, "none", dtype=object)
    else:
        print_manager_mode = np.asarray(print_manager_mode, dtype=object)
    no_toggle_mode = (print_manager_mode == "no_toggle_control") | (
        print_manager_mode == "delayed_start"
    )
    natural_toggle_mode = print_manager_mode == "natural_toggle"
    toggle_mode = (print_manager_mode == "toggle") | natural_toggle_mode
    death_stop_mode = (print_manager_mode == "death_stop") | natural_toggle_mode

    if events_enabled:
        started = _timer_start(phase_timers)
        _reset_event_arrays(state)
        _timer_add(phase_timers, "event_reset_sec", started)

    started = _timer_start(phase_timers)
    for name, value in _advance_pre_step_timers_batched(
        state,
        timer_advance_ms=step_input.timer_advance_ms,
        events_enabled=events_enabled,
    ).items():
        counters[name] += value
    _timer_add(phase_timers, "pre_step_timer_sec", started)

    frame_start_deaths = player_count - state["alive"][:, :player_count].sum(
        axis=1,
    ).astype(np.int32)
    death_rows = np.zeros(row_count, dtype=bool)
    for player in range(player_count - 1, -1, -1):
        live_mask = (
            state["alive"][:, player]
            & ~state["done"]
            & ~state["overflow"]
            & ~disabled_player_mask[:, player]
        )
        if not live_mask.any():
            continue

        started = _timer_start(phase_timers)
        counters["movement_updates"] += advance_player_movement(
            state,
            player=player,
            live_mask=live_mask,
            step_ms=step_ms,
            source_moves=moves,
        )
        _timer_add(phase_timers, "movement_sec", started)

        _append_visual_trail_points_batched(
            state,
            player=player,
            write_mask=live_mask & state["printing"][:, player],
        )

        if events_enabled:
            started = _timer_start(phase_timers)
            _emit_position_events_batched(state, player, live_mask)
            _timer_add(phase_timers, "event_emit_sec", started)

        started = _timer_start(phase_timers)
        cursor_dx = state["pos"][:, player, 0] - state["draw_cursor_pos"][:, player, 0]
        cursor_dy = state["pos"][:, player, 1] - state["draw_cursor_pos"][:, player, 1]
        cursor_dist_sq = cursor_dx * cursor_dx + cursor_dy * cursor_dy
        radius_sq = state["radius"][:, player] * state["radius"][:, player]
        should_draw = (
            live_mask
            & state["printing"][:, player]
            & (~state["has_draw_cursor"][:, player] | (cursor_dist_sq > radius_sq))
        )
        _timer_add(phase_timers, "normal_point_mask_sec", started)

        started = _timer_start(phase_timers)
        inserted, overflowed = _append_body_points_batched(
            state,
            player=player,
            write_mask=should_draw,
            insert_kind=BODY_KIND_NORMAL,
        )
        counters["normal_points_inserted"] += inserted
        counters["body_overflow_attempts"] += overflowed
        _timer_add(phase_timers, "normal_point_append_sec", started)

        if events_enabled:
            started = _timer_start(phase_timers)
            _emit_point_events_batched(state, player, should_draw, important=False)
            _timer_add(phase_timers, "event_emit_sec", started)

        started = _timer_start(phase_timers)
        wrap_count, wrapped_mask = apply_borderless_wrap(
            state,
            player=player,
            live_mask=live_mask,
        )
        counters["borderless_wraps"] += wrap_count
        _timer_add(phase_timers, "border_wrap_sec", started)

        _append_visual_trail_points_batched(
            state,
            player=player,
            write_mask=wrapped_mask & state["printing"][:, player],
        )

        if events_enabled:
            started = _timer_start(phase_timers)
            _emit_position_events_batched(state, player, wrapped_mask)
            _timer_add(phase_timers, "event_emit_sec", started)

        started = _timer_start(phase_timers)
        wall_hit_mask = normal_wall_hit_mask(
            state,
            player=player,
            live_mask=live_mask & ~wrapped_mask,
        )
        _timer_add(phase_timers, "wall_check_sec", started)
        player_death_enabled = death_enabled and not bool(death_immunity_mask[:, player].all())
        player_mortal_mask = ~death_immunity_mask[:, player]
        mortal_wall_hit_mask = wall_hit_mask & player_mortal_mask
        if mortal_wall_hit_mask.any() and player_death_enabled:
            started = _timer_start(phase_timers)
            state["alive"][mortal_wall_hit_mask, player] = False
            state["death_tick"][mortal_wall_hit_mask, player] = state["tick"][mortal_wall_hit_mask]
            state["round_score"][mortal_wall_hit_mask, player] += frame_start_deaths[
                mortal_wall_hit_mask
            ]
            _append_death_list_batched(
                state,
                player=player,
                death_mask=mortal_wall_hit_mask,
                row_count=row_count,
                death_cause=DEATH_CAUSE_WALL,
                death_hit_owner=-1,
            )
            death_rows |= mortal_wall_hit_mask
            counters["normal_wall_deaths"] += int(mortal_wall_hit_mask.sum())
            inserted, overflowed = _append_body_points_batched(
                state,
                player=player,
                write_mask=mortal_wall_hit_mask,
                insert_kind=BODY_KIND_DEATH,
            )
            counters["death_points_inserted"] += inserted
            counters["body_overflow_attempts"] += overflowed
            _timer_add(phase_timers, "wall_death_apply_sec", started)

            if events_enabled:
                started = _timer_start(phase_timers)
                _emit_point_events_batched(state, player, mortal_wall_hit_mask, important=False)
                _timer_add(phase_timers, "event_emit_sec", started)

            started = _timer_start(phase_timers)
            stop_count, stop_points, visual_clears = _stop_print_manager_on_death_batched(
                state,
                player=player,
                death_mask=mortal_wall_hit_mask & death_stop_mode,
                events_enabled=events_enabled,
            )
            counters["print_manager_death_stops"] += stop_count
            counters["print_manager_death_stop_points"] += stop_points
            counters["print_manager_death_stop_visual_clears"] += visual_clears
            _timer_add(phase_timers, "print_manager_death_stop_sec", started)

            if events_enabled:
                started = _timer_start(phase_timers)
                _emit_die_events_batched(state, player, mortal_wall_hit_mask)
                _emit_score_events_batched(
                    state,
                    player,
                    mortal_wall_hit_mask,
                    event_type=EVENT_SCORE_ROUND,
                )
                _timer_add(phase_timers, "event_emit_sec", started)

        started = _timer_start(phase_timers)
        wall_block_mask = (
            mortal_wall_hit_mask if death_enabled else np.zeros(row_count, dtype=bool)
        )
        collision_live_mask = live_mask & ~wrapped_mask & ~wall_block_mask
        hit_rows, candidate_count, scanned_slots = _body_collision_rows(
            state,
            player,
            collision_live_mask,
        )
        counters["body_candidates"] += candidate_count
        counters["body_scan_slots"] += scanned_slots
        _timer_add(phase_timers, "body_collision_sec", started)
        body_hit_mask = np.zeros(row_count, dtype=bool)
        if hit_rows.size:
            detected_body_hit_mask = _rows_to_mask(hit_rows, row_count)
            started = _timer_start(phase_timers)
            hit_owners = _first_hit_body_owner(state, player, hit_rows)
            _timer_add(phase_timers, "body_hit_owner_sec", started)
            counters["body_hits"] += int(hit_rows.size)

            if death_enabled:
                body_hit_mask = detected_body_hit_mask & player_mortal_mask
                if "invincible" in state:
                    body_hit_mask &= ~_bool_array_shape(
                        state,
                        "invincible",
                        shape=state["alive"].shape,
                    )[:, player]
                death_causes = _body_death_causes_from_hit_owners(
                    hit_owners,
                    player=player,
                )

                started = _timer_start(phase_timers)
                body_death_rows = np.flatnonzero(body_hit_mask)
                state["alive"][body_death_rows, player] = False
                state["death_tick"][body_death_rows, player] = state["tick"][body_death_rows]
                _append_death_list_batched(
                    state,
                    player=player,
                    death_mask=body_hit_mask,
                    row_count=row_count,
                    death_cause=death_causes,
                    death_hit_owner=hit_owners,
                )
                death_rows |= body_hit_mask
                inserted, overflowed = _append_body_points_batched(
                    state,
                    player=player,
                    write_mask=body_hit_mask,
                    insert_kind=BODY_KIND_DEATH,
                )
                counters["death_points_inserted"] += inserted
                counters["body_overflow_attempts"] += overflowed
                _timer_add(phase_timers, "body_death_apply_sec", started)

                if events_enabled:
                    started = _timer_start(phase_timers)
                    _emit_point_events_batched(state, player, body_hit_mask, important=False)
                    _timer_add(phase_timers, "event_emit_sec", started)

                started = _timer_start(phase_timers)
                stop_count, stop_points, visual_clears = _stop_print_manager_on_death_batched(
                    state,
                    player=player,
                    death_mask=body_hit_mask & death_stop_mode,
                    events_enabled=events_enabled,
                )
                counters["print_manager_death_stops"] += stop_count
                counters["print_manager_death_stop_points"] += stop_points
                counters["print_manager_death_stop_visual_clears"] += visual_clears
                _timer_add(phase_timers, "print_manager_death_stop_sec", started)

                if events_enabled:
                    started = _timer_start(phase_timers)
                    _emit_die_events_batched(
                        state,
                        player,
                        body_hit_mask,
                        other_player=hit_owners,
                        old=False,
                    )
                    _emit_score_events_batched(
                        state,
                        player,
                        body_hit_mask,
                        event_type=EVENT_SCORE_ROUND,
                    )
                    _timer_add(phase_timers, "event_emit_sec", started)

        body_block_mask = body_hit_mask if death_enabled else np.zeros(row_count, dtype=bool)
        print_manager_live_mask = live_mask & ~wall_block_mask & ~body_block_mask
        if print_manager_live_mask.any():
            started = _timer_start(phase_timers)
            no_toggle, unhandled_toggle = _update_print_manager_no_toggle_batched(
                state,
                player=player,
                live_mask=print_manager_live_mask & no_toggle_mode,
            )
            toggle, no_toggle_from_toggle, visual_clears = _update_print_manager_toggle_batched(
                state,
                player=player,
                live_mask=print_manager_live_mask & toggle_mode,
                events_enabled=events_enabled,
            )
            counters["print_manager_no_toggle_updates"] += no_toggle
            counters["print_manager_toggle_updates"] += toggle
            counters["print_manager_toggle_rows_unhandled"] += (
                unhandled_toggle + no_toggle_from_toggle
            )
            counters["print_manager_visual_clears"] += visual_clears
            _timer_add(phase_timers, "print_manager_update_sec", started)

        if bonus_arrays is not None:
            started = _timer_start(phase_timers)
            bonus_live_mask = print_manager_live_mask & ~state["overflow"]
            catch_counts, stack_appends = _catch_bonus_batched(
                state,
                bonus_arrays,
                player=player,
                live_mask=bonus_live_mask,
                events_enabled=events_enabled,
                row_count=row_count,
                player_count=player_count,
            )
            for name, value in catch_counts.items():
                counters[name] += value
            counters["bonus_stack_appends"] += stack_appends
            _timer_add(phase_timers, "bonus_catch_sec", started)

    counters["terminal_score_rows"] += _apply_terminal_score_for_rows_batched(
        state,
        death_rows,
        player_count=player_count,
        phase_timers=phase_timers,
        event_mode=step_input.event_mode,
    )
    finalized = finalize_step_counters(
        counters,
        state,
        random_tape_draw_count_before=random_tape_draw_count_before,
    )

    started = _timer_start(phase_timers)
    active_rows = ~state["done"]
    state["tick"][active_rows] += 1
    _timer_add(phase_timers, "tick_sec", started)
    return finalized


def prepare_step_input(
    state: Mapping[str, np.ndarray],
    prepared_batch: Mapping[str, Any],
    *,
    event_mode: str = EVENT_MODE_DEBUG,
) -> VectorStepInput:
    """Normalize and validate mapping-shaped step inputs for one batched tick."""

    if not isinstance(prepared_batch, Mapping):
        raise VectorRuntimeError("prepared_batch must be a mapping")

    tick = _state_array(state, "tick")
    if not np.issubdtype(tick.dtype, np.integer) or tick.ndim != 1:
        raise VectorRuntimeError("state['tick'] must be an integer array with shape [B]")
    batch_size = tick.shape[0]

    player_count = _positive_int(
        prepared_batch.get("player_count"),
        "prepared_batch.player_count",
    )
    step_ms = _array(
        prepared_batch.get("step_ms"),
        dtype=np.float64,
        field="prepared_batch.step_ms",
    )
    if step_ms.shape != (batch_size,):
        raise VectorRuntimeError("prepared_batch.step_ms must have shape [B]")

    source_moves = _array(
        prepared_batch.get("source_moves"),
        dtype=np.int8,
        field="prepared_batch.source_moves",
    )
    if source_moves.shape != (batch_size, player_count):
        raise VectorRuntimeError("prepared_batch.source_moves must have shape [B,P]")

    raw_modes = prepared_batch.get("print_manager_mode")
    if raw_modes is None:
        print_manager_mode = np.full(batch_size, "none", dtype=object)
    else:
        print_manager_mode = _array(
            raw_modes,
            dtype=object,
            field="prepared_batch.print_manager_mode",
        )
    if print_manager_mode.shape != (batch_size,):
        raise VectorRuntimeError("prepared_batch.print_manager_mode must have shape [B]")

    step_input = VectorStepInput(
        state=state,
        step_ms=step_ms,
        source_moves=source_moves,
        player_count=player_count,
        print_manager_mode=print_manager_mode,
        timer_advance_ms=_optional_timer_advance_ms(
            prepared_batch,
            batch_size=batch_size,
        ),
        event_mode=event_mode,
        disabled_player_mask=prepared_batch.get("disabled_player_mask"),
    )
    validate_step_input(step_input)
    return step_input


def _optional_timer_advance_ms(
    prepared_batch: Mapping[str, Any],
    *,
    batch_size: int,
) -> np.ndarray:
    raw = prepared_batch.get("timer_advance_ms")
    if raw is None:
        return np.zeros(batch_size, dtype=np.float64)
    return _row_float_input(raw, batch_size, "prepared_batch.timer_advance_ms")


def _events_enabled(event_mode: str) -> bool:
    if event_mode not in EVENT_MODES:
        raise VectorRuntimeError("event_mode must be 'debug-event' or 'no-event'")
    return event_mode == EVENT_MODE_DEBUG


def _timer_start(phase_timers: dict[str, float] | None) -> float:
    if phase_timers is None:
        return 0.0
    return time.perf_counter()


def _timer_add(
    phase_timers: dict[str, float] | None,
    key: str,
    started: float,
) -> None:
    if phase_timers is None:
        return
    phase_timers[key] = phase_timers.get(key, 0.0) + time.perf_counter() - started


def empty_step_counters(*, include_random_tape: bool = True) -> dict[str, int]:
    """Return the complete zeroed counter set for one vector step call."""

    counters = {name: 0 for name in STEP_COUNTER_NAMES}
    if not include_random_tape:
        counters.pop("random_tape_draws")
        counters.pop("random_tape_exhaustions")
    return counters


def snapshot_random_tape_counters(state: Mapping[str, np.ndarray]) -> np.ndarray:
    """Snapshot per-row random draw counts before a transition."""

    arrays = _random_tape_arrays(state)
    return arrays["draw_count"].copy()


def finalize_random_tape_counters(
    counters: dict[str, int],
    state: Mapping[str, np.ndarray],
    draw_count_before: np.ndarray,
) -> None:
    """Update random tape counters from a before/after draw-count snapshot."""

    arrays = _random_tape_arrays(state)
    before = np.asarray(draw_count_before)
    if before.shape != arrays["draw_count"].shape or not np.issubdtype(
        before.dtype,
        np.integer,
    ):
        raise VectorRuntimeError("draw_count_before must be an integer array with shape [B]")

    draw_delta = arrays["draw_count"].astype(np.int64) - before.astype(np.int64)
    if bool((draw_delta < 0).any()):
        raise VectorRuntimeError("random_tape_draw_count cannot decrease during a step")
    counters["random_tape_draws"] = int(draw_delta.sum())
    counters["random_tape_exhaustions"] = int(arrays["exhausted"].sum())


def finalize_step_counters(
    counters: Mapping[str, int],
    state: Mapping[str, np.ndarray],
    *,
    random_tape_draw_count_before: np.ndarray,
) -> dict[str, int]:
    """Return complete counters after state-derived accounting is applied."""

    finalized = empty_step_counters()
    for name, value in counters.items():
        if name not in finalized:
            raise VectorRuntimeError(f"unknown step counter {name!r}")
        finalized[name] = int(value)

    finalize_random_tape_counters(
        finalized,
        state,
        random_tape_draw_count_before,
    )
    finalized["events_emitted"] = _row_counter_sum(state, "event_count")
    finalized["event_overflow_attempts"] = _row_counter_sum(
        state,
        "event_overflow_attempts",
    )
    return finalized


def advance_warmup_no_bonus_timers(
    state: Mapping[str, np.ndarray],
    advance_ms: Any,
    *,
    player_count: int | None = None,
    max_timer_callbacks: int = 16,
) -> dict[str, Any]:
    """Advance the no-bonus warmup lifecycle timer slice for 2P/3P/4P rows.

    This helper continues the row state produced by
    ``vector_lifecycle.reset_spawn_warmup_no_bonus_rows``. It supports only the
    source warmup path: ``game:start`` activates the world and schedules
    reversed PrintManager start timers; those timers start PrintManagers,
    insert important body points, and consume row-local random tape.
    """

    return _advance_warmup_no_bonus_timers_impl(
        state,
        advance_ms,
        player_count=player_count,
        max_timer_callbacks=max_timer_callbacks,
        schema=WARMUP_TIMER_ADVANCE_NO_BONUS_INFO_SCHEMA_ID,
        surface=WARMUP_TIMER_ADVANCE_NO_BONUS_SURFACE,
    )


def advance_warmup_1v1_no_bonus_timers(
    state: Mapping[str, np.ndarray],
    advance_ms: Any,
    *,
    max_timer_callbacks: int = 16,
) -> dict[str, Any]:
    """Compatibility wrapper for strict 1v1/no-bonus warmup timer advance."""

    return _advance_warmup_no_bonus_timers_impl(
        state,
        advance_ms,
        player_count=SUPPORTED_WARMUP_PLAYER_COUNT,
        max_timer_callbacks=max_timer_callbacks,
        schema=WARMUP_TIMER_ADVANCE_INFO_SCHEMA_ID,
        surface=WARMUP_TIMER_ADVANCE_SURFACE,
    )


def advance_warmdown_no_bonus_timers(
    state: Mapping[str, np.ndarray],
    advance_ms: Any,
    *,
    player_count: int | None = None,
    next_round_warmup_ms: float = 0.0,
) -> dict[str, Any]:
    """Advance source-shaped no-bonus warmdown timers into the next round.

    This is a narrow lifecycle helper for rows that have already reached
    ``round:end`` without ending the public env episode. It handles only
    ``TIMER_KIND_WARMDOWN_END``: fire source-shaped ``game:stop``, clear
    round-local state, spawn the next natural round from row-local random tape,
    and schedule that round's ``game:start`` warmup timer.
    """

    tick = _state_array(state, "tick")
    if not np.issubdtype(tick.dtype, np.integer) or tick.ndim != 1:
        raise VectorRuntimeError("state['tick'] must be an integer array with shape [B]")
    row_count = tick.shape[0]
    player_arrays = _warmup_player_arrays(
        state,
        row_count=row_count,
        player_count=player_count,
    )
    player_count = int(player_arrays["alive"].shape[1])
    advance = _row_float_input(advance_ms, row_count, "advance_ms")
    if not np.isfinite(float(next_round_warmup_ms)) or float(next_round_warmup_ms) < 0.0:
        raise VectorRuntimeError("next_round_warmup_ms must be finite and nonnegative")

    timer_arrays = _warmup_timer_arrays(state, row_count=row_count)
    lifecycle_arrays = _warmup_lifecycle_arrays(state, row_count=row_count)
    _body_point_arrays(state, row_count=row_count, player_count=player_count)
    random_tape = _random_tape_arrays(state)
    random_draw_count_before = random_tape["draw_count"].copy()

    counters = {
        "pre_step_timer_advances": 0,
        "pre_step_timer_fires": 0,
        "warmdown_end_fires": 0,
        "game_stop_count": 0,
        "round_new_count": 0,
        "next_round_spawn_count": 0,
        "scheduled_game_start_count": 0,
        "body_overflow_attempts": 0,
        "timer_overflow_count": 0,
        "random_tape_draws": 0,
        "random_tape_exhaustions": 0,
    }
    warmdown_rows: list[int] = []
    game_stop_rows: list[int] = []
    round_new_rows: list[int] = []
    scheduled_rows: list[int] = []
    scheduled_slots: list[int] = []
    timer_overflow_rows: list[int] = []
    spawn_infos: list[dict[str, Any]] = []

    active_rows = np.flatnonzero(~lifecycle_arrays["done"] & ~lifecycle_arrays["overflow"])
    for row in active_rows:
        row_int = int(row)
        active_slots = np.flatnonzero(timer_arrays["active"][row_int])
        if active_slots.size == 0:
            continue

        counters["pre_step_timer_advances"] += 1
        remaining = timer_arrays["remaining_ms"][row_int, active_slots]
        due_after_ms = max(0.0, float(remaining.min()))
        if due_after_ms > float(advance[row_int]):
            timer_arrays["remaining_ms"][row_int, active_slots] -= float(advance[row_int])
            continue

        timer_arrays["remaining_ms"][row_int, active_slots] -= due_after_ms
        due_slots = [
            int(slot)
            for slot in active_slots
            if float(timer_arrays["remaining_ms"][row_int, int(slot)]) <= 0.0
        ]
        due_slots.sort(key=lambda slot: int(timer_arrays["seq"][row_int, slot]))

        for slot in due_slots:
            if not bool(timer_arrays["active"][row_int, slot]):
                continue
            kind = int(timer_arrays["kind"][row_int, slot])
            if kind != TIMER_KIND_WARMDOWN_END:
                raise VectorRuntimeError(f"unsupported warmdown timer kind {kind}")

            _clear_timer_slot(timer_arrays, row=row_int, slot=slot)
            counters["pre_step_timer_fires"] += 1
            counters["warmdown_end_fires"] += 1
            counters["game_stop_count"] += 1
            counters["round_new_count"] += 1
            warmdown_rows.append(row_int)
            game_stop_rows.append(row_int)
            round_new_rows.append(row_int)

            spawn_info, schedule_info = _fire_warmdown_end_timer(
                state,
                timer_arrays,
                lifecycle_arrays,
                row=row_int,
                row_count=row_count,
                player_count=player_count,
                next_round_warmup_ms=float(next_round_warmup_ms),
            )
            spawn_infos.append(spawn_info)
            counters["next_round_spawn_count"] += int(spawn_info["spawn_count"])
            counters["scheduled_game_start_count"] += int(schedule_info["scheduled"])
            counters["timer_overflow_count"] += int(schedule_info["overflowed"])
            if schedule_info["overflowed"]:
                timer_overflow_rows.append(row_int)
            else:
                scheduled_rows.append(row_int)
                scheduled_slots.append(int(schedule_info["slot"]))

    random_draw_delta = random_tape["draw_count"].astype(np.int64) - (
        random_draw_count_before.astype(np.int64)
    )
    if bool((random_draw_delta < 0).any()):
        raise VectorRuntimeError("random_tape_draw_count cannot decrease")
    counters["random_tape_draws"] = int(random_draw_delta.sum())
    counters["random_tape_exhaustions"] = int(random_tape["exhausted"].sum())

    return {
        "schema": WARMDOWN_TIMER_ADVANCE_NO_BONUS_INFO_SCHEMA_ID,
        "surface": WARMDOWN_TIMER_ADVANCE_NO_BONUS_SURFACE,
        "player_count": player_count,
        "supported_player_counts": list(SUPPORTED_NO_BONUS_WARMUP_PLAYER_COUNTS),
        **counters,
        "advance_ms": advance.copy(),
        "warmdown_end_rows": np.asarray(warmdown_rows, dtype=np.int32),
        "game_stop_rows": np.asarray(game_stop_rows, dtype=np.int32),
        "round_new_rows": np.asarray(round_new_rows, dtype=np.int32),
        "scheduled_game_start_rows": np.asarray(scheduled_rows, dtype=np.int32),
        "scheduled_game_start_slots": np.asarray(scheduled_slots, dtype=np.int16),
        "timer_overflow_rows": np.asarray(timer_overflow_rows, dtype=np.int32),
        "random_draw_count_delta": random_draw_delta.astype(np.int32),
        "spawn_infos": spawn_infos,
    }


def _advance_warmup_no_bonus_timers_impl(
    state: Mapping[str, np.ndarray],
    advance_ms: Any,
    *,
    player_count: int | None,
    max_timer_callbacks: int,
    schema: str,
    surface: str,
) -> dict[str, Any]:
    tick = _state_array(state, "tick")
    if not np.issubdtype(tick.dtype, np.integer) or tick.ndim != 1:
        raise VectorRuntimeError("state['tick'] must be an integer array with shape [B]")
    row_count = tick.shape[0]
    if (
        not isinstance(max_timer_callbacks, int)
        or isinstance(max_timer_callbacks, bool)
        or max_timer_callbacks < 1
    ):
        raise VectorRuntimeError("max_timer_callbacks must be a positive integer")

    advance = _row_float_input(advance_ms, row_count, "advance_ms")
    timer_arrays = _warmup_timer_arrays(state, row_count=row_count)
    lifecycle_arrays = _warmup_lifecycle_arrays(state, row_count=row_count)
    player_arrays = _warmup_player_arrays(
        state,
        row_count=row_count,
        player_count=player_count,
    )
    player_count = int(player_arrays["alive"].shape[1])
    body_arrays = _body_point_arrays(
        state,
        row_count=row_count,
        player_count=player_count,
    )
    visible_arrays = _optional_visible_trail_arrays(
        state,
        row_count=row_count,
        player_count=player_count,
    )
    random_tape = _random_tape_arrays(state)
    random_draw_count_before = random_tape["draw_count"].copy()

    counters = {
        "pre_step_timer_advances": 0,
        "pre_step_timer_fires": 0,
        "game_start_fires": 0,
        "scheduled_print_manager_start_count": 0,
        "print_manager_delayed_start_fires": 0,
        "print_manager_delayed_start_points": 0,
        "body_overflow_attempts": 0,
        "random_tape_draws": 0,
        "random_tape_exhaustions": 0,
    }
    game_start_rows: list[int] = []
    scheduled_rows: list[int] = []
    scheduled_slots: list[int] = []
    scheduled_players: list[int] = []
    print_start_rows: list[int] = []
    print_start_players: list[int] = []
    timer_overflow_rows: list[int] = []

    active_rows = np.flatnonzero(~lifecycle_arrays["done"] & ~lifecycle_arrays["overflow"])
    callback_count = 0
    for row in active_rows:
        row_int = int(row)
        if not bool(timer_arrays["active"][row_int].any()):
            continue

        counters["pre_step_timer_advances"] += 1
        budget_ms = float(advance[row_int])
        while bool(timer_arrays["active"][row_int].any()):
            active_slots = np.flatnonzero(timer_arrays["active"][row_int])
            active_remaining = timer_arrays["remaining_ms"][row_int, active_slots]
            due_after_ms = max(0.0, float(active_remaining.min()))
            if due_after_ms > budget_ms:
                timer_arrays["remaining_ms"][row_int, active_slots] -= budget_ms
                break

            timer_arrays["remaining_ms"][row_int, active_slots] -= due_after_ms
            budget_ms -= due_after_ms
            due_slots = [
                int(slot)
                for slot in active_slots
                if float(timer_arrays["remaining_ms"][row_int, int(slot)]) <= 0.0
            ]
            due_slots.sort(key=lambda slot: int(timer_arrays["seq"][row_int, slot]))
            if not due_slots:
                break

            for slot in due_slots:
                if not bool(timer_arrays["active"][row_int, slot]):
                    continue
                callback_count += 1
                if callback_count > max_timer_callbacks:
                    raise VectorRuntimeError(
                        f"timer advance exceeded {max_timer_callbacks} callbacks"
                    )

                kind = int(timer_arrays["kind"][row_int, slot])
                if kind == TIMER_KIND_GAME_START:
                    _clear_timer_slot(timer_arrays, row=row_int, slot=slot)
                    game_start_rows.append(row_int)
                    counters["pre_step_timer_fires"] += 1
                    counters["game_start_fires"] += 1
                    scheduled = _fire_game_start_timer(
                        timer_arrays,
                        lifecycle_arrays,
                        row=row_int,
                        player_count=player_count,
                        player_arrays=player_arrays,
                    )
                    if scheduled["overflowed"]:
                        timer_overflow_rows.append(row_int)
                    for scheduled_slot, player in scheduled["slots_and_players"]:
                        scheduled_rows.append(row_int)
                        scheduled_slots.append(scheduled_slot)
                        scheduled_players.append(player)
                    counters["scheduled_print_manager_start_count"] += len(
                        scheduled["slots_and_players"]
                    )
                elif kind == TIMER_KIND_PRINT_MANAGER_START:
                    player = int(timer_arrays["player"][row_int, slot])
                    fired, points, overflowed = _fire_print_manager_start_timer(
                        state,
                        timer_arrays,
                        lifecycle_arrays,
                        player_arrays,
                        body_arrays,
                        visible_arrays,
                        row=row_int,
                        player=player,
                    )
                    _clear_timer_slot(timer_arrays, row=row_int, slot=slot)
                    counters["pre_step_timer_fires"] += fired
                    counters["print_manager_delayed_start_fires"] += fired
                    counters["print_manager_delayed_start_points"] += points
                    counters["body_overflow_attempts"] += overflowed
                    if fired:
                        print_start_rows.append(row_int)
                        print_start_players.append(player)
                else:
                    raise VectorRuntimeError(f"unsupported warmup timer kind {kind}")

    random_draw_delta = random_tape["draw_count"].astype(np.int64) - (
        random_draw_count_before.astype(np.int64)
    )
    if bool((random_draw_delta < 0).any()):
        raise VectorRuntimeError("random_tape_draw_count cannot decrease")
    counters["random_tape_draws"] = int(random_draw_delta.sum())
    counters["random_tape_exhaustions"] = int(random_tape["exhausted"].sum())

    return {
        "schema": schema,
        "surface": surface,
        "player_count": player_count,
        "supported_player_counts": list(SUPPORTED_NO_BONUS_WARMUP_PLAYER_COUNTS),
        **counters,
        "advance_ms": advance.copy(),
        "game_start_rows": np.asarray(game_start_rows, dtype=np.int32),
        "scheduled_print_manager_start_rows": np.asarray(
            scheduled_rows,
            dtype=np.int32,
        ),
        "scheduled_print_manager_start_slots": np.asarray(
            scheduled_slots,
            dtype=np.int16,
        ),
        "scheduled_print_manager_start_players": np.asarray(
            scheduled_players,
            dtype=np.int16,
        ),
        "print_manager_start_rows": np.asarray(print_start_rows, dtype=np.int32),
        "print_manager_start_players": np.asarray(
            print_start_players,
            dtype=np.int16,
        ),
        "timer_overflow_rows": np.asarray(timer_overflow_rows, dtype=np.int32),
        "random_draw_count_delta": random_draw_delta.astype(np.int32),
    }


def assign_print_manager_random_distances(
    state: Mapping[str, np.ndarray],
    *,
    player: int,
    mask: np.ndarray,
) -> None:
    """Assign PrintManager random distances and advance row-local random tapes."""

    tick = _state_array(state, "tick")
    row_count = tick.shape[0]
    if not isinstance(player, int) or isinstance(player, bool):
        raise VectorRuntimeError("player must be an integer index")

    distance = _state_array(state, "print_manager_distance")
    printing = _state_array(state, "printing")
    if distance.ndim != 2 or printing.ndim != 2:
        raise VectorRuntimeError("print_manager_distance and printing must have shape [B,P]")
    if distance.shape[0] != row_count or printing.shape[0] != row_count:
        raise VectorRuntimeError("print_manager_distance and printing must have shape [B,P]")
    if player < 0 or player >= distance.shape[1] or player >= printing.shape[1]:
        raise VectorRuntimeError(f"unknown player index {player}")

    rows = _bool_mask(mask, row_count, "mask")
    for row in np.flatnonzero(rows):
        row_int = int(row)
        distance[row_int, player] = next_print_manager_random_distance(
            state,
            row=row_int,
            printing=bool(printing[row_int, player]),
        )


def advance_player_movement(
    state: Mapping[str, np.ndarray],
    *,
    player: int,
    live_mask: Any,
    step_ms: Any,
    source_moves: Any,
) -> int:
    """Apply the source-compatible movement update for one player across rows."""

    tick = _state_array(state, "tick")
    if not np.issubdtype(tick.dtype, np.integer) or tick.ndim != 1:
        raise VectorRuntimeError("state['tick'] must be an integer array with shape [B]")
    row_count = tick.shape[0]

    live_rows = _bool_mask(live_mask, row_count, "live_mask")
    step_ms_array = _array(step_ms, dtype=np.float64, field="step_ms")
    if step_ms_array.shape != (row_count,):
        raise VectorRuntimeError("step_ms must be a numeric array with shape [B]")
    if not bool(np.isfinite(step_ms_array).all()):
        raise VectorRuntimeError("step_ms values must be finite")

    moves = np.asarray(source_moves)
    if not np.issubdtype(moves.dtype, np.integer) or moves.ndim != 2:
        raise VectorRuntimeError("source_moves must be an integer array with shape [B,P]")
    if moves.shape[0] != row_count:
        raise VectorRuntimeError("source_moves must be an integer array with shape [B,P]")
    player_count = moves.shape[1]
    if not isinstance(player, int) or isinstance(player, bool) or player < 0:
        raise VectorRuntimeError("player must be an integer index")
    if player >= player_count:
        raise VectorRuntimeError(f"unknown player index {player}")

    pos = _player_points_array(state, "pos", row_count, player_count)
    prev_pos = _player_points_array(state, "prev_pos", row_count, player_count)
    heading = _player_numeric_array(state, "heading", row_count, player_count)
    angular_velocity = _player_numeric_array(
        state,
        "angular_velocity_per_ms",
        row_count,
        player_count,
    )
    speed = _player_numeric_array(state, "speed", row_count, player_count)
    inverse = None
    if "inverse" in state:
        inverse = _bool_array_shape(state, "inverse", shape=(row_count, player_count))
    live_body_num = _player_integral_array(state, "live_body_num", row_count, player_count)
    body_count = _player_integral_array(state, "body_count", row_count, player_count)

    live_body_num[live_rows, player] = body_count[live_rows, player]
    old_pos = pos[:, player].copy()
    old_heading = heading[:, player].copy()
    move_factor = moves[:, player].astype(np.float64, copy=False)
    if inverse is not None:
        move_factor = np.where(inverse[:, player], -move_factor, move_factor)
    angle_delta = angular_velocity[:, player] * step_ms_array * move_factor
    new_heading = old_heading + angle_delta
    heading[:, player] = np.where(live_rows, new_heading, old_heading)
    distance = speed[:, player] * step_ms_array / 1000.0
    prev_pos[live_rows, player] = old_pos[live_rows]
    pos[:, player, 0] = np.where(
        live_rows,
        old_pos[:, 0] + np.cos(heading[:, player]) * distance,
        pos[:, player, 0],
    )
    pos[:, player, 1] = np.where(
        live_rows,
        old_pos[:, 1] + np.sin(heading[:, player]) * distance,
        pos[:, player, 1],
    )
    return int(live_rows.sum())


def apply_borderless_wrap(
    state: Mapping[str, np.ndarray],
    *,
    player: int,
    live_mask: Any,
) -> tuple[int, np.ndarray]:
    """Apply source-compatible borderless wrapping for one player across rows."""

    tick = _state_array(state, "tick")
    if not np.issubdtype(tick.dtype, np.integer) or tick.ndim != 1:
        raise VectorRuntimeError("state['tick'] must be an integer array with shape [B]")
    row_count = tick.shape[0]

    live_rows = _bool_mask(live_mask, row_count, "live_mask")
    pos = _state_array(state, "pos")
    if pos.ndim != 3 or pos.shape[0] != row_count or pos.shape[2] != 2:
        raise VectorRuntimeError("pos must be a numeric array with shape [B,P,2]")
    if not np.issubdtype(pos.dtype, np.number):
        raise VectorRuntimeError("pos must be a numeric array with shape [B,P,2]")
    if not isinstance(player, int) or isinstance(player, bool) or player < 0:
        raise VectorRuntimeError("player must be an integer index")
    if player >= pos.shape[1]:
        raise VectorRuntimeError(f"unknown player index {player}")

    borderless = _state_array(state, "borderless")
    if borderless.shape != (row_count,) or borderless.dtype != np.bool_:
        raise VectorRuntimeError("borderless must be a bool array with shape [B]")
    map_size = _state_array(state, "map_size")
    if map_size.shape != (row_count,) or not np.issubdtype(map_size.dtype, np.number):
        raise VectorRuntimeError("map_size must be a numeric array with shape [B]")

    active = live_rows & borderless
    if not active.any():
        return 0, np.zeros(row_count, dtype=bool)

    x = pos[:, player, 0]
    y = pos[:, player, 1]
    x_low = active & (x < 0.0)
    x_high = active & (x > map_size)
    x_wrapped = x_low | x_high
    y_low = active & ~x_wrapped & (y < 0.0)
    y_high = active & ~x_wrapped & (y > map_size)
    wrapped = x_wrapped | y_low | y_high

    pos[x_low, player, 0] = map_size[x_low]
    pos[x_high, player, 0] = 0.0
    pos[y_low, player, 1] = map_size[y_low]
    pos[y_high, player, 1] = 0.0
    if "live_body_num" in state and "body_count" in state:
        live_body_num = _state_array(state, "live_body_num")
        body_count = _state_array(state, "body_count")
        if (
            live_body_num.ndim != 2
            or body_count.ndim != 2
            or live_body_num.shape[0] != row_count
            or body_count.shape[0] != row_count
            or player >= live_body_num.shape[1]
            or player >= body_count.shape[1]
        ):
            raise VectorRuntimeError("live_body_num and body_count must have shape [B,P]")
        live_body_num[wrapped, player] = body_count[wrapped, player]
    if wrapped.any() and "has_draw_cursor" in state:
        has_draw_cursor = _state_array(state, "has_draw_cursor")
        draw_cursor_pos = _state_array(state, "draw_cursor_pos")
        if (
            has_draw_cursor.ndim != 2
            or has_draw_cursor.shape[0] != row_count
            or has_draw_cursor.dtype != np.bool_
            or player >= has_draw_cursor.shape[1]
            or draw_cursor_pos.shape != (*has_draw_cursor.shape, 2)
            or not np.issubdtype(draw_cursor_pos.dtype, np.number)
        ):
            raise VectorRuntimeError(
                "has_draw_cursor and draw_cursor_pos must have shapes [B,P] and [B,P,2]"
            )
        has_draw_cursor[wrapped, player] = False
        draw_cursor_pos[wrapped, player] = 0.0
    if wrapped.any() and "has_visual_trail_last" in state:
        has_visual_trail_last = _state_array(state, "has_visual_trail_last")
        visual_trail_last_pos = _state_array(state, "visual_trail_last_pos")
        if (
            has_visual_trail_last.ndim != 2
            or has_visual_trail_last.shape[0] != row_count
            or has_visual_trail_last.dtype != np.bool_
            or player >= has_visual_trail_last.shape[1]
            or visual_trail_last_pos.shape != (*has_visual_trail_last.shape, 2)
            or not np.issubdtype(visual_trail_last_pos.dtype, np.number)
        ):
            raise VectorRuntimeError(
                "has_visual_trail_last and visual_trail_last_pos must have shapes [B,P] and [B,P,2]"
            )
        has_visual_trail_last[wrapped, player] = False
        visual_trail_last_pos[wrapped, player] = 0.0
    return int(wrapped.sum()), wrapped


def normal_wall_hit_mask(
    state: Mapping[str, np.ndarray],
    *,
    player: int,
    live_mask: Any,
) -> np.ndarray:
    """Return source-compatible normal-wall hits for one player across rows."""

    tick = _state_array(state, "tick")
    if not np.issubdtype(tick.dtype, np.integer) or tick.ndim != 1:
        raise VectorRuntimeError("state['tick'] must be an integer array with shape [B]")
    row_count = tick.shape[0]

    live_rows = _bool_mask(live_mask, row_count, "live_mask")
    pos = _state_array(state, "pos")
    if pos.ndim != 3 or pos.shape[0] != row_count or pos.shape[2] != 2:
        raise VectorRuntimeError("pos must be a numeric array with shape [B,P,2]")
    if not np.issubdtype(pos.dtype, np.number):
        raise VectorRuntimeError("pos must be a numeric array with shape [B,P,2]")
    player_count = pos.shape[1]
    if not isinstance(player, int) or isinstance(player, bool) or player < 0:
        raise VectorRuntimeError("player must be an integer index")
    if player >= player_count:
        raise VectorRuntimeError(f"unknown player index {player}")

    radius = _player_numeric_array(state, "radius", row_count, player_count)
    borderless = _state_array(state, "borderless")
    if borderless.shape != (row_count,) or borderless.dtype != np.bool_:
        raise VectorRuntimeError("borderless must be a bool array with shape [B]")
    map_size = _state_array(state, "map_size")
    if map_size.shape != (row_count,) or not np.issubdtype(map_size.dtype, np.number):
        raise VectorRuntimeError("map_size must be a numeric array with shape [B]")

    active = live_rows & ~borderless
    if not active.any():
        return np.zeros(row_count, dtype=bool)

    x = pos[:, player, 0]
    y = pos[:, player, 1]
    player_radius = radius[:, player]
    hit = (
        (x - player_radius < 0.0)
        | (x + player_radius > map_size)
        | (y - player_radius < 0.0)
        | (y + player_radius > map_size)
    )
    return active & hit


def _append_body_points_batched(
    state: Mapping[str, np.ndarray],
    *,
    player: int,
    write_mask: np.ndarray,
    insert_kind: int,
) -> tuple[int, int]:
    rows = np.flatnonzero(write_mask)
    if rows.size == 0:
        return 0, 0

    capacity = state["body_active"].shape[1]
    cursor = state["body_write_cursor"][rows].astype(np.int64, copy=False)
    can_insert = cursor < capacity
    overflow_rows = rows[~can_insert]
    if overflow_rows.size:
        state["body_overflow"][overflow_rows] = True
        state["overflow"][overflow_rows] = True

    insert_rows = rows[can_insert]
    insert_cursor = cursor[can_insert]
    if insert_rows.size == 0:
        return 0, int(overflow_rows.size)

    body_num = state["body_count"][insert_rows, player].copy()
    if "body_break_before" in state:
        body_break_before = _state_array(state, "body_break_before")
        if (
            body_break_before.shape != state["body_active"].shape
            or body_break_before.dtype != np.bool_
        ):
            raise VectorRuntimeError("body_break_before must be a bool array with shape [B,C]")
        if "has_draw_cursor" in state:
            has_draw_cursor = _state_array(state, "has_draw_cursor")
            if (
                has_draw_cursor.ndim != 2
                or has_draw_cursor.shape[0] != state["body_active"].shape[0]
                or has_draw_cursor.dtype != np.bool_
                or player >= has_draw_cursor.shape[1]
            ):
                raise VectorRuntimeError("has_draw_cursor must be a bool array with shape [B,P]")
            body_break_before[insert_rows, insert_cursor] = ~has_draw_cursor[
                insert_rows,
                player,
            ]
        else:
            body_break_before[insert_rows, insert_cursor] = False
    state["body_active"][insert_rows, insert_cursor] = True
    state["body_pos"][insert_rows, insert_cursor] = state["pos"][insert_rows, player]
    state["body_radius"][insert_rows, insert_cursor] = state["radius"][insert_rows, player]
    state["body_owner"][insert_rows, insert_cursor] = player
    state["body_num"][insert_rows, insert_cursor] = body_num
    state["body_insert_tick"][insert_rows, insert_cursor] = state["tick"][insert_rows]
    state["body_insert_kind"][insert_rows, insert_cursor] = insert_kind
    state["body_write_cursor"][insert_rows] += 1
    state["world_body_count"][insert_rows] += 1
    state["body_count"][insert_rows, player] += 1
    state["visible_trail_count"][insert_rows, player] += 1
    state["has_visible_trail_last"][insert_rows, player] = True
    state["visible_trail_last_pos"][insert_rows, player] = state["pos"][insert_rows, player]
    state["has_draw_cursor"][insert_rows, player] = True
    state["draw_cursor_pos"][insert_rows, player] = state["pos"][insert_rows, player]
    if insert_kind == BODY_KIND_IMPORTANT:
        _append_visual_trail_points_batched(
            state,
            player=player,
            write_mask=write_mask,
        )
    return int(insert_rows.size), int(overflow_rows.size)


def _append_visual_trail_points_batched(
    state: Mapping[str, np.ndarray],
    *,
    player: int,
    write_mask: np.ndarray,
) -> tuple[int, int]:
    visual_arrays = _optional_visual_trail_arrays(
        state,
        row_count=state["tick"].shape[0],
        player_count=state["pos"].shape[1],
    )
    if visual_arrays is None:
        return 0, 0

    rows = np.flatnonzero(write_mask)
    if rows.size == 0:
        return 0, 0

    capacity = visual_arrays["active"].shape[1]
    cursor = visual_arrays["write_cursor"][rows].astype(np.int64, copy=False)
    can_insert = cursor < capacity
    overflow_rows = rows[~can_insert]
    if overflow_rows.size:
        visual_arrays["overflow"][overflow_rows] = True

    insert_rows = rows[can_insert]
    insert_cursor = cursor[can_insert]
    if insert_rows.size == 0:
        return 0, int(overflow_rows.size)

    visual_arrays["active"][insert_rows, insert_cursor] = True
    visual_arrays["pos"][insert_rows, insert_cursor] = state["pos"][insert_rows, player]
    visual_arrays["radius"][insert_rows, insert_cursor] = state["radius"][insert_rows, player]
    visual_arrays["owner"][insert_rows, insert_cursor] = player
    visual_arrays["break_before"][insert_rows, insert_cursor] = ~visual_arrays["has_last"][
        insert_rows,
        player,
    ]
    visual_arrays["write_cursor"][insert_rows] += 1
    visual_arrays["has_last"][insert_rows, player] = True
    visual_arrays["last_pos"][insert_rows, player] = state["pos"][insert_rows, player]
    return int(insert_rows.size), int(overflow_rows.size)


def _update_print_manager_no_toggle_batched(
    state: Mapping[str, np.ndarray],
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


def _update_print_manager_toggle_batched(
    state: Mapping[str, np.ndarray],
    *,
    player: int,
    live_mask: np.ndarray,
    events_enabled: bool,
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
        _append_body_points_batched(
            state,
            player=player,
            write_mask=toggle,
            insert_kind=BODY_KIND_IMPORTANT,
        )
        if events_enabled:
            _emit_point_events_batched(state, player, toggle, important=True)
        if toggled_to_hole.any():
            _clear_visible_trail_batched(state, player, toggled_to_hole)
        if events_enabled:
            _emit_printing_property_events_batched(state, player, toggle)
        assign_print_manager_random_distances(
            state,
            player=player,
            mask=toggle,
        )

    return int(toggle.sum()), int(no_toggle.sum()), int(toggled_to_hole.sum())


def _stop_print_manager_on_death_batched(
    state: Mapping[str, np.ndarray],
    *,
    player: int,
    death_mask: np.ndarray,
    events_enabled: bool,
) -> tuple[int, int, int]:
    active = death_mask & state["print_manager_active"][:, player]
    if not active.any():
        return 0, 0, 0

    was_printing = state["printing"][:, player].copy()
    important_stop = active & was_printing

    if important_stop.any():
        _append_body_points_batched(
            state,
            player=player,
            write_mask=important_stop,
            insert_kind=BODY_KIND_IMPORTANT,
        )
        if events_enabled:
            _emit_point_events_batched(state, player, important_stop, important=True)

    state["printing"][active, player] = False
    if important_stop.any():
        _clear_visible_trail_batched(state, player, important_stop)
    if events_enabled:
        _emit_printing_property_events_batched(state, player, active)
    assign_print_manager_random_distances(state, player=player, mask=active)

    state["print_manager_active"][active, player] = False
    state["print_manager_distance"][active, player] = 0.0
    state["print_manager_last_pos"][active, player] = 0.0
    return int(active.sum()), int(important_stop.sum()), int(important_stop.sum())


def _optional_bonus_arrays(
    state: Mapping[str, np.ndarray],
    *,
    row_count: int,
    player_count: int,
) -> dict[str, np.ndarray | None] | None:
    names = (
        "bonus_active",
        "bonus_type",
        "bonus_id",
        "bonus_pos",
        "bonus_radius",
        "bonus_count",
        "bonus_world_body_count",
    )
    if not any(name in state for name in names):
        return None
    missing = [name for name in names if name not in state]
    if missing:
        missing_names = ", ".join(missing)
        raise VectorRuntimeError(f"bonus arrays must be supplied together; missing {missing_names}")

    active = _bool_array_shape(state, "bonus_active", ndim=2, row_count=row_count)
    bonus_shape = active.shape
    player_shape = (row_count, player_count)
    world_active = None
    if "bonus_world_active" in state:
        world_active = _bool_array_shape(state, "bonus_world_active", shape=(row_count,))
    radius_power = None
    if "radius_power" in state:
        radius_power = _integer_array_shape(state, "radius_power", shape=player_shape)
    base_radius = None
    if "base_radius" in state:
        base_radius = _numeric_array_shape(state, "base_radius", shape=player_shape)

    return {
        "active": active,
        "type": _integer_array_shape(state, "bonus_type", shape=bonus_shape),
        "id": _integer_array_shape(state, "bonus_id", shape=bonus_shape),
        "pos": _numeric_array_shape(state, "bonus_pos", shape=(*bonus_shape, 2)),
        "radius": _numeric_array_shape(state, "bonus_radius", shape=bonus_shape),
        "count": _integer_array_shape(state, "bonus_count", shape=(row_count,)),
        "world_body_count": _integer_array_shape(
            state,
            "bonus_world_body_count",
            shape=(row_count,),
        ),
        "world_active": world_active,
        "radius_power": radius_power,
        "base_radius": base_radius,
        "stack": _optional_bonus_stack_arrays(
            state,
            row_count=row_count,
            player_count=player_count,
        ),
        "game_stack": _optional_bonus_game_stack_arrays(
            state,
            row_count=row_count,
        ),
    }


def bonus_type_selection_metadata(
    state: Mapping[str, np.ndarray],
    type_draws: Any,
    *,
    player_count: int,
    eligible_rows: Any | None = None,
    enabled_type_codes: Any | None = None,
) -> dict[str, Any]:
    """Return source weighted bonus type selection metadata for supplied draws.

    This is a metadata-only bridge for rows whose natural pop timer, next-delay
    draw, cap check, and row-local RNG ownership were already handled by the
    caller. It does not schedule timers, draw positions, spawn bonuses, mutate
    bonus arrays, or claim public bonus-env support.
    """

    tick = _state_array(state, "tick")
    if tick.ndim != 1:
        raise VectorRuntimeError("state['tick'] must be an array with shape [B]")
    row_count = tick.shape[0]
    player_count = _positive_int(player_count, "player_count")
    player_shape = (row_count, player_count)

    alive = _bool_array_shape(state, "alive", shape=player_shape)
    present = _bool_array_shape(state, "present", shape=player_shape)
    draw_values = _row_unit_interval_input(type_draws, row_count, "type_draws")
    if eligible_rows is None:
        eligible_mask = np.ones(row_count, dtype=bool)
    else:
        eligible_mask = _bool_mask(eligible_rows, row_count, "eligible_rows")
    enabled_codes = _enabled_bonus_type_code_matrix(
        enabled_type_codes,
        row_count=row_count,
    )

    selected_type_code = np.zeros(row_count, dtype=np.int16)
    total_weight = np.zeros(row_count, dtype=np.float64)
    weighted_draw = np.zeros(row_count, dtype=np.float64)
    game_clear_probability = np.zeros(row_count, dtype=np.float64)

    for row in np.flatnonzero(eligible_mask):
        row_int = int(row)
        present_count = int(present[row_int].sum())
        alive_count = int(alive[row_int].sum())
        clear_probability = _source_bonus_game_clear_probability(
            alive_count=alive_count,
            present_count=present_count,
        )
        game_clear_probability[row_int] = clear_probability

        row_total_weight = 0.0
        for code in enabled_codes[row_int]:
            row_total_weight += _source_bonus_probability_for_code(
                int(code),
                game_clear_probability=clear_probability,
            )
        total_weight[row_int] = row_total_weight
        if row_total_weight <= 0.0:
            continue

        row_weighted_draw = float(draw_values[row_int]) * row_total_weight
        weighted_draw[row_int] = row_weighted_draw
        cumulative_weight = 0.0
        for code in enabled_codes[row_int]:
            code_int = int(code)
            probability = _source_bonus_probability_for_code(
                code_int,
                game_clear_probability=clear_probability,
            )
            if probability <= 0.0:
                continue
            cumulative_weight += probability
            if row_weighted_draw < cumulative_weight:
                selected_type_code[row_int] = code_int
                break

    return {
        "schema": BONUS_TYPE_SELECTION_METADATA_SCHEMA_ID,
        "surface": BONUS_TYPE_SELECTION_METADATA_SURFACE,
        "eligible_rows": eligible_mask.copy(),
        "type_draw": draw_values.copy(),
        "weighted_draw": weighted_draw,
        "total_weight": total_weight,
        "game_clear_probability": game_clear_probability,
        "selected_type_code": selected_type_code,
        "selected_type_name": np.asarray(
            [_bonus_type_name_from_code(int(code)) for code in selected_type_code],
            dtype=object,
        ),
    }


def bonus_spawn_cap_metadata(
    state: Mapping[str, np.ndarray],
    *,
    eligible_rows: Any | None = None,
) -> dict[str, Any]:
    """Return row-local source cap metadata for already-due bonus pop rows.

    This mirrors only the source ``popBonus()`` cap check after the caller has
    already handled natural timer eligibility and the next-delay draw.
    """

    tick = _state_array(state, "tick")
    if tick.ndim != 1:
        raise VectorRuntimeError("state['tick'] must be an array with shape [B]")
    row_count = tick.shape[0]
    bonus_count = _integer_array_shape(state, "bonus_count", shape=(row_count,))
    if np.any(bonus_count < 0):
        raise VectorRuntimeError("bonus_count values must be non-negative")

    if eligible_rows is None:
        eligible_mask = np.ones(row_count, dtype=bool)
    else:
        eligible_mask = _bool_mask(eligible_rows, row_count, "eligible_rows")

    capped_rows = eligible_mask & (bonus_count >= SOURCE_MAX_ACTIVE_BONUSES)
    selection_rows = eligible_mask & ~capped_rows
    return {
        "schema": BONUS_SPAWN_CAP_METADATA_SCHEMA_ID,
        "surface": BONUS_SPAWN_CAP_METADATA_SURFACE,
        "source_max_active_bonuses": SOURCE_MAX_ACTIVE_BONUSES,
        "eligible_rows": eligible_mask.copy(),
        "bonus_count": bonus_count.copy(),
        "capped_rows": capped_rows,
        "selection_rows": selection_rows,
    }


def bonus_spawn_due_rows(
    state: Mapping[str, np.ndarray],
    *,
    player_count: int,
    due_rows: Any | None = None,
    type_draws: Any | None = None,
    position_draws: Any | None = None,
    enabled_type_codes: Any | None = None,
    events_enabled: bool = False,
) -> dict[str, Any]:
    """Spawn natural bonuses for caller-owned due rows from supplied RNG draws.

    This intentionally stops short of owning the source timer system. The caller
    decides which rows are due, supplies the already-owned type and position
    draws, and owns scheduling the next natural pop.
    """

    tick = _state_array(state, "tick")
    if tick.ndim != 1:
        raise VectorRuntimeError("state['tick'] must be an array with shape [B]")
    row_count = tick.shape[0]
    player_count = _positive_int(player_count, "player_count")
    bonus_arrays = _optional_bonus_arrays(
        state,
        row_count=row_count,
        player_count=player_count,
    )
    if bonus_arrays is None:
        raise VectorRuntimeError("bonus_spawn_due_rows requires bonus arrays")

    if due_rows is None:
        due_mask = np.ones(row_count, dtype=bool)
    else:
        due_mask = _bool_mask(due_rows, row_count, "due_rows")

    cap_info = bonus_spawn_cap_metadata(state, eligible_rows=due_mask)
    selection_rows = cap_info["selection_rows"]
    selected_type_code = np.zeros(row_count, dtype=np.int16)
    type_selection_info: dict[str, Any] | None = None
    if bool(selection_rows.any()):
        if type_draws is None:
            raise VectorRuntimeError("type_draws are required for uncapped due rows")
        type_selection_info = bonus_type_selection_metadata(
            state,
            type_draws,
            player_count=player_count,
            eligible_rows=selection_rows,
            enabled_type_codes=enabled_type_codes,
        )
        selected_type_code = type_selection_info["selected_type_code"].copy()

    spawn_rows = selection_rows & (selected_type_code != BONUS_TYPE_NONE)
    spawned_slot = np.full(row_count, -1, dtype=np.int32)
    spawned_bonus_id = np.full(row_count, -1, dtype=np.int32)
    spawned_pos = np.zeros((row_count, 2), dtype=np.float64)
    accepted_position_attempt = np.full(row_count, -1, dtype=np.int32)
    position_attempt_count = np.zeros(row_count, dtype=np.int32)
    rejected_game_world_attempts = np.zeros(row_count, dtype=np.int32)
    rejected_bonus_world_attempts = np.zeros(row_count, dtype=np.int32)
    position_margin = np.zeros(row_count, dtype=np.float64)
    position_span = np.zeros(row_count, dtype=np.float64)
    spawn_events: list[dict[str, Any]] = []

    if bool(spawn_rows.any()):
        if position_draws is None:
            raise VectorRuntimeError("position_draws are required for selected spawn rows")
        position_draw_values = _bonus_position_draws_input(
            position_draws,
            row_count,
            "position_draws",
        )
        map_size = _numeric_array_shape(state, "map_size", shape=(row_count,))
        body_arrays = (
            _body_point_arrays(state, row_count=row_count, player_count=player_count)
            if "body_active" in state
            else None
        )
        next_bonus_id = None
        if "bonus_next_id" in state:
            next_bonus_id = _integer_array_shape(
                state,
                "bonus_next_id",
                shape=(row_count,),
            )

        for row in np.flatnonzero(spawn_rows):
            row_int = int(row)
            slot = _first_inactive_bonus_slot(bonus_arrays["active"], row=row_int)
            if slot < 0:
                raise VectorRuntimeError(
                    f"row {row_int} bonus array capacity is too small for spawn"
                )

            margin = SOURCE_BONUS_RADIUS + SOURCE_BONUS_POSITION_MARGIN_FRACTION * float(
                map_size[row_int],
            )
            span = float(map_size[row_int]) - margin * 2.0
            if not np.isfinite(margin) or not np.isfinite(span) or span < 0.0:
                raise VectorRuntimeError(
                    f"row {row_int} map_size is too small for source bonus spawn"
                )
            position_margin[row_int] = margin
            position_span[row_int] = span

            accepted_x = 0.0
            accepted_y = 0.0
            for attempt in range(position_draw_values.shape[1]):
                draw_x = float(position_draw_values[row_int, attempt, 0])
                draw_y = float(position_draw_values[row_int, attempt, 1])
                candidate_x = margin + draw_x * span
                candidate_y = margin + draw_y * span
                position_attempt_count[row_int] += 1
                game_collision = _bonus_spawn_position_collides_body_world(
                    body_arrays,
                    row=row_int,
                    x=candidate_x,
                    y=candidate_y,
                    radius=margin,
                )
                bonus_collision = _bonus_spawn_position_collides_bonus_world(
                    bonus_arrays,
                    row=row_int,
                    x=candidate_x,
                    y=candidate_y,
                    radius=margin,
                )
                rejected_game_world_attempts[row_int] += int(game_collision)
                rejected_bonus_world_attempts[row_int] += int(bonus_collision)
                if game_collision or bonus_collision:
                    continue
                accepted_position_attempt[row_int] = attempt
                accepted_x = candidate_x
                accepted_y = candidate_y
                break

            if accepted_position_attempt[row_int] < 0:
                raise VectorRuntimeError(
                    f"row {row_int} position_draws did not include an accepted candidate"
                )

            bonus_id = _next_spawn_bonus_id(
                bonus_arrays["id"],
                bonus_arrays["active"],
                row=row_int,
                next_bonus_id=next_bonus_id,
            )
            bonus_type_code = int(selected_type_code[row_int])
            bonus_arrays["active"][row_int, slot] = True
            bonus_arrays["type"][row_int, slot] = bonus_type_code
            bonus_arrays["id"][row_int, slot] = bonus_id
            bonus_arrays["pos"][row_int, slot] = (accepted_x, accepted_y)
            bonus_arrays["radius"][row_int, slot] = SOURCE_BONUS_RADIUS
            bonus_arrays["count"][row_int] = int(bonus_arrays["active"][row_int].sum())
            bonus_arrays["world_body_count"][row_int] = int(bonus_arrays["count"][row_int])
            if isinstance(bonus_arrays["world_active"], np.ndarray):
                bonus_arrays["world_active"][row_int] = True
            if next_bonus_id is not None:
                next_bonus_id[row_int] = bonus_id + 1

            spawned_slot[row_int] = slot
            spawned_bonus_id[row_int] = bonus_id
            spawned_pos[row_int] = (accepted_x, accepted_y)
            if events_enabled:
                _emit_bonus_pop_row(
                    state,
                    row=row_int,
                    bonus_id=bonus_id,
                    bonus_type_code=bonus_type_code,
                    x=accepted_x,
                    y=accepted_y,
                )
            spawn_events.append(
                {
                    "event": "bonus:pop",
                    "row": row_int,
                    "bonus": bonus_id,
                    "type": _bonus_type_name_from_code(bonus_type_code),
                    "x": accepted_x,
                    "y": accepted_y,
                }
            )

    return {
        "schema": BONUS_SPAWN_DUE_ROWS_SCHEMA_ID,
        "surface": BONUS_SPAWN_DUE_ROWS_SURFACE,
        "due_rows": due_mask.copy(),
        "capped_rows": cap_info["capped_rows"].copy(),
        "selection_rows": selection_rows.copy(),
        "spawn_rows": spawn_rows.copy(),
        "selected_type_code": selected_type_code.copy(),
        "selected_type_name": np.asarray(
            [_bonus_type_name_from_code(int(code)) for code in selected_type_code],
            dtype=object,
        ),
        "spawned_slot": spawned_slot,
        "spawned_bonus_id": spawned_bonus_id,
        "spawned_pos": spawned_pos,
        "accepted_position_attempt": accepted_position_attempt,
        "position_attempt_count": position_attempt_count,
        "rejected_game_world_attempts": rejected_game_world_attempts,
        "rejected_bonus_world_attempts": rejected_bonus_world_attempts,
        "position_margin": position_margin,
        "position_span": position_span,
        "source_bonus_radius": SOURCE_BONUS_RADIUS,
        "source_max_active_bonuses": SOURCE_MAX_ACTIVE_BONUSES,
        "cap_info": cap_info,
        "type_selection_info": type_selection_info,
        "spawn_events": spawn_events,
    }


def _source_bonus_game_clear_probability(
    *,
    alive_count: int,
    present_count: int,
) -> float:
    if present_count <= 0:
        return 0.0
    ratio = 1.0 - alive_count / present_count
    if ratio < 0.5:
        return 1.0
    return float(_js_round_positive((1.0 - ratio) * 10.0)) / 10.0


def _source_bonus_probability_for_code(
    bonus_type_code: int,
    *,
    game_clear_probability: float,
) -> float:
    if bonus_type_code == BONUS_TYPE_NONE:
        return 0.0
    if bonus_type_code == BONUS_TYPE_GAME_CLEAR:
        return game_clear_probability
    if bonus_type_code in (BONUS_TYPE_ENEMY_INVERSE, BONUS_TYPE_GAME_BORDERLESS):
        return 0.8
    if bonus_type_code == BONUS_TYPE_ENEMY_STRAIGHT_ANGLE:
        return 0.6
    if bonus_type_code in SOURCE_DEFAULT_BONUS_TYPE_CODES:
        return 1.0
    raise VectorRuntimeError(f"unsupported source bonus type code {bonus_type_code}")


def _enabled_bonus_type_code_matrix(
    enabled_type_codes: Any | None,
    *,
    row_count: int,
) -> np.ndarray:
    if enabled_type_codes is None:
        default_codes = np.asarray(SOURCE_DEFAULT_BONUS_TYPE_CODES, dtype=np.int16)
        return np.repeat(default_codes.reshape(1, -1), row_count, axis=0)

    codes = np.asarray(enabled_type_codes)
    if not np.issubdtype(codes.dtype, np.integer):
        raise VectorRuntimeError("enabled_type_codes must be an integer array")
    if codes.ndim == 1:
        codes = np.repeat(codes.reshape(1, -1), row_count, axis=0)
    elif codes.ndim != 2 or codes.shape[0] != row_count:
        raise VectorRuntimeError("enabled_type_codes must have shape [T] or [B,T]")

    allowed_codes = np.asarray(
        (BONUS_TYPE_NONE, *SOURCE_DEFAULT_BONUS_TYPE_CODES),
        dtype=np.int64,
    )
    if not bool(np.isin(codes, allowed_codes).all()):
        raise VectorRuntimeError(
            "enabled_type_codes must contain source default bonus type codes or BonusNone padding"
        )
    return codes.astype(np.int16, copy=False)


def _bonus_type_name_from_code(bonus_type_code: int) -> str:
    if bonus_type_code < 0 or bonus_type_code >= len(BONUS_TYPE_NAME_BY_CODE):
        raise VectorRuntimeError(f"unknown bonus type code {bonus_type_code}")
    return BONUS_TYPE_NAME_BY_CODE[bonus_type_code]


def _bonus_position_draws_input(value: Any, row_count: int, field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise VectorRuntimeError(f"{field} cannot be converted to float64") from exc

    if array.ndim == 1 and array.shape == (2,):
        array = np.repeat(array.reshape(1, 1, 2), row_count, axis=0)
    elif array.ndim == 2 and array.shape[1:] == (2,):
        array = np.repeat(array.reshape(1, array.shape[0], 2), row_count, axis=0)
    elif array.ndim != 3 or array.shape[0] != row_count or array.shape[2] != 2:
        raise VectorRuntimeError(f"{field} must have shape [2], [A,2], or [B,A,2]")

    if array.shape[1] < 1:
        raise VectorRuntimeError(f"{field} must include at least one position attempt")
    if not bool(np.isfinite(array).all()):
        raise VectorRuntimeError(f"{field} values must be finite")
    if bool((array < 0.0).any()) or bool((array >= 1.0).any()):
        raise VectorRuntimeError(f"{field} values must be in [0, 1)")
    return array


def _first_inactive_bonus_slot(active: np.ndarray, *, row: int) -> int:
    inactive_slots = np.flatnonzero(~active[row])
    if inactive_slots.size == 0:
        return -1
    return int(inactive_slots[0])


def _next_spawn_bonus_id(
    bonus_id: np.ndarray,
    active: np.ndarray,
    *,
    row: int,
    next_bonus_id: np.ndarray | None,
) -> int:
    if next_bonus_id is not None:
        value = int(next_bonus_id[row])
        if value < 1:
            raise VectorRuntimeError("bonus_next_id values must be positive")
        return value

    active_ids = bonus_id[row, active[row]]
    if active_ids.size == 0:
        return 1
    if bool((active_ids < 1).any()):
        raise VectorRuntimeError("active bonus_id values must be positive")
    return int(active_ids.max()) + 1


def _bonus_spawn_position_collides_body_world(
    body_arrays: Mapping[str, np.ndarray] | None,
    *,
    row: int,
    x: float,
    y: float,
    radius: float,
) -> bool:
    if body_arrays is None:
        return False
    active = body_arrays["active"][row]
    if not bool(active.any()):
        return False
    dx = body_arrays["pos"][row, :, 0] - x
    dy = body_arrays["pos"][row, :, 1] - y
    hit_radius = body_arrays["radius"][row] + radius
    return bool((active & (dx * dx + dy * dy < hit_radius * hit_radius)).any())


def _bonus_spawn_position_collides_bonus_world(
    bonus_arrays: Mapping[str, Any],
    *,
    row: int,
    x: float,
    y: float,
    radius: float,
) -> bool:
    active = bonus_arrays["active"][row]
    if not bool(active.any()):
        return False
    dx = bonus_arrays["pos"][row, :, 0] - x
    dy = bonus_arrays["pos"][row, :, 1] - y
    hit_radius = bonus_arrays["radius"][row] + radius
    return bool((active & (dx * dx + dy * dy < hit_radius * hit_radius)).any())


def _js_round_positive(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _optional_bonus_stack_arrays(
    state: Mapping[str, np.ndarray],
    *,
    row_count: int,
    player_count: int,
) -> dict[str, np.ndarray] | None:
    names = (
        "bonus_stack_count",
        "bonus_stack_id",
        "bonus_stack_type",
        "bonus_stack_duration_ms",
        "bonus_stack_radius_power",
    )
    if not any(name in state for name in names):
        return None
    missing = [name for name in names if name not in state]
    if missing:
        missing_names = ", ".join(missing)
        raise VectorRuntimeError(
            f"bonus stack arrays must be supplied together; missing {missing_names}"
        )

    player_shape = (row_count, player_count)
    count = _integer_array_shape(state, "bonus_stack_count", shape=player_shape)
    stack_id = _state_array(state, "bonus_stack_id")
    if (
        stack_id.ndim != 3
        or stack_id.shape[:2] != player_shape
        or not np.issubdtype(stack_id.dtype, np.integer)
    ):
        raise VectorRuntimeError("bonus_stack_id must be an integer array with shape [B,P,S]")
    stack_shape = stack_id.shape
    velocity_delta = None
    if "bonus_stack_velocity_delta" in state:
        velocity_delta = _numeric_array_shape(
            state,
            "bonus_stack_velocity_delta",
            shape=stack_shape,
        )
    inverse_delta = None
    if "bonus_stack_inverse_delta" in state:
        inverse_delta = _integer_array_shape(
            state,
            "bonus_stack_inverse_delta",
            shape=stack_shape,
        )
    angular_velocity_per_ms = None
    if "bonus_stack_angular_velocity_per_ms" in state:
        angular_velocity_per_ms = _numeric_array_shape(
            state,
            "bonus_stack_angular_velocity_per_ms",
            shape=stack_shape,
        )
    invincible_delta = None
    if "bonus_stack_invincible_delta" in state:
        invincible_delta = _integer_array_shape(
            state,
            "bonus_stack_invincible_delta",
            shape=stack_shape,
        )
    printing_delta = None
    if "bonus_stack_printing_delta" in state:
        printing_delta = _integer_array_shape(
            state,
            "bonus_stack_printing_delta",
            shape=stack_shape,
        )
    color = None
    if "bonus_stack_color" in state:
        color = _integer_array_shape(
            state,
            "bonus_stack_color",
            shape=stack_shape,
        )
    return {
        "count": count,
        "id": stack_id,
        "type": _integer_array_shape(state, "bonus_stack_type", shape=stack_shape),
        "duration_ms": _integer_array_shape(
            state,
            "bonus_stack_duration_ms",
            shape=stack_shape,
        ),
        "radius_power": _integer_array_shape(
            state,
            "bonus_stack_radius_power",
            shape=stack_shape,
        ),
        "velocity_delta": velocity_delta,
        "inverse_delta": inverse_delta,
        "angular_velocity_per_ms": angular_velocity_per_ms,
        "invincible_delta": invincible_delta,
        "printing_delta": printing_delta,
        "color": color,
    }


def _optional_bonus_game_stack_arrays(
    state: Mapping[str, np.ndarray],
    *,
    row_count: int,
) -> dict[str, np.ndarray] | None:
    names = (
        "bonus_game_stack_count",
        "bonus_game_stack_id",
        "bonus_game_stack_type",
        "bonus_game_stack_duration_ms",
        "bonus_game_stack_borderless",
    )
    if not any(name in state for name in names):
        return None
    missing = [name for name in names if name not in state]
    if missing:
        missing_names = ", ".join(missing)
        raise VectorRuntimeError(
            f"bonus game stack arrays must be supplied together; missing {missing_names}"
        )

    count = _integer_array_shape(state, "bonus_game_stack_count", shape=(row_count,))
    stack_id = _state_array(state, "bonus_game_stack_id")
    if (
        stack_id.ndim != 2
        or stack_id.shape[0] != row_count
        or not np.issubdtype(stack_id.dtype, np.integer)
    ):
        raise VectorRuntimeError("bonus_game_stack_id must be an integer array with shape [B,S]")
    stack_shape = stack_id.shape
    return {
        "count": count,
        "id": stack_id,
        "type": _integer_array_shape(state, "bonus_game_stack_type", shape=stack_shape),
        "duration_ms": _integer_array_shape(
            state,
            "bonus_game_stack_duration_ms",
            shape=stack_shape,
        ),
        "borderless": _integer_array_shape(
            state,
            "bonus_game_stack_borderless",
            shape=stack_shape,
        ),
    }


def _catch_bonus_batched(
    state: Mapping[str, np.ndarray],
    bonus_arrays: Mapping[str, np.ndarray | Mapping[str, np.ndarray] | None],
    *,
    player: int,
    live_mask: np.ndarray,
    events_enabled: bool,
    row_count: int,
    player_count: int,
) -> tuple[dict[str, int], int]:
    active = bonus_arrays["active"]
    bonus_type = bonus_arrays["type"]
    bonus_id = bonus_arrays["id"]
    bonus_pos = bonus_arrays["pos"]
    bonus_radius = bonus_arrays["radius"]
    bonus_count = bonus_arrays["count"]
    world_active = bonus_arrays["world_active"]
    stack_arrays = bonus_arrays["stack"]
    game_stack_arrays = bonus_arrays["game_stack"]

    if not isinstance(active, np.ndarray) or active.shape[1] == 0:
        return {}, 0

    live_rows = live_mask.copy()
    if isinstance(world_active, np.ndarray):
        live_rows &= world_active
    if not live_rows.any():
        return {}, 0

    catch_counts: dict[str, int] = {}
    stack_append_count = 0
    radius = state["radius"]
    pos = state["pos"]
    for row in np.flatnonzero(live_rows):
        row_int = int(row)
        hit_slot = _caught_bonus_slot(
            active,
            bonus_pos,
            bonus_radius,
            radius,
            pos,
            row=row_int,
            player=player,
        )
        if hit_slot < 0:
            continue

        caught_type = int(bonus_type[row_int, hit_slot])
        effect = BONUS_RUNTIME_EFFECT_BY_TYPE.get(caught_type)
        if effect is None:
            raise VectorRuntimeError(
                f"unsupported caught bonus type code {caught_type}; "
                "only table-backed runtime bonus effects are supported"
            )

        catch_counts[effect.catch_counter] = catch_counts.get(effect.catch_counter, 0) + 1
        caught_id = int(bonus_id[row_int, hit_slot])
        if "bonus_catch_count_step" in state:
            state["bonus_catch_count_step"][row_int, player] += 1
        active[row_int, hit_slot] = False
        bonus_count[row_int] = int(active[row_int].sum())
        if events_enabled:
            _emit_bonus_clear_row(state, row=row_int, bonus_id=caught_id)

        if effect.target_group == BONUS_TARGET_CLEAR:
            _apply_bonus_game_clear(
                state,
                row=row_int,
                row_count=row_count,
                player_count=player_count,
            )
            if events_enabled:
                _emit_clear_row(state, row=row_int)
            continue

        if effect.target_group == BONUS_TARGET_GAME:
            if caught_type != BONUS_TYPE_GAME_BORDERLESS:
                raise VectorRuntimeError(f"unsupported game bonus type {caught_type}")
            appended = _append_bonus_game_borderless_stack_entry(
                game_stack_arrays,
                row=row_int,
                bonus_id=caught_id,
            )
            stack_append_count += int(appended)
            borderless_changed = _apply_bonus_game_borderless(
                state,
                row=row_int,
                row_count=row_count,
            )
            if events_enabled and borderless_changed:
                _emit_borderless_row(state, row=row_int, value=True)
            continue

        target_players = _bonus_target_players(
            state,
            effect.target_group,
            row=row_int,
            catcher=player,
            player_count=player_count,
        )
        color_values = _bonus_all_color_values(
            state,
            row=row_int,
            target_players=target_players,
        )

        for target_player in reversed(target_players):
            appended = _append_bonus_stack_entry(
                stack_arrays,
                row=row_int,
                player=target_player,
                bonus_id=caught_id,
                bonus_type=caught_type,
                color_value=color_values.get(target_player),
            )
            stack_append_count += int(appended)
            old_radius = float(state["radius"][row_int, target_player])
            old_speed = _bonus_optional_player_float(
                state,
                "speed",
                row=row_int,
                player=target_player,
            )
            old_inverse = _bonus_optional_player_bool(
                state,
                "inverse",
                row=row_int,
                player=target_player,
            )
            old_invincible = _bonus_optional_player_bool(
                state,
                "invincible",
                row=row_int,
                player=target_player,
            )
            old_printing = _bonus_optional_player_bool(
                state,
                "printing",
                row=row_int,
                player=target_player,
            )
            old_color = _bonus_optional_player_int(
                state,
                "avatar_color",
                row=row_int,
                player=target_player,
            )
            _resolve_bonus_avatar_effects_from_stack(
                state,
                stack_arrays,
                changed_type=caught_type,
                row=row_int,
                player=target_player,
                row_count=row_count,
            )
            if events_enabled:
                if _bonus_has_radius_effect(caught_type):
                    next_radius = float(state["radius"][row_int, target_player])
                    if old_radius != next_radius:
                        _emit_radius_property_event_row(
                            state,
                            row=row_int,
                            player=target_player,
                            radius=next_radius,
                        )
                if _bonus_has_velocity_effect(caught_type) and old_speed is not None:
                    next_speed = float(state["speed"][row_int, target_player])
                    if old_speed != next_speed:
                        _emit_velocity_property_event_row(
                            state,
                            row=row_int,
                            player=target_player,
                            velocity=next_speed,
                        )
                if _bonus_has_inverse_effect(caught_type) and old_inverse is not None:
                    next_inverse = bool(state["inverse"][row_int, target_player])
                    if old_inverse != next_inverse:
                        _emit_inverse_property_event_row(
                            state,
                            row=row_int,
                            player=target_player,
                            value=next_inverse,
                        )
                if _bonus_has_invincible_effect(caught_type) and old_invincible is not None:
                    next_invincible = bool(state["invincible"][row_int, target_player])
                    if old_invincible != next_invincible:
                        _emit_invincible_property_event_row(
                            state,
                            row=row_int,
                            player=target_player,
                            value=next_invincible,
                        )
                if _bonus_has_printing_effect(caught_type) and old_printing is not None:
                    next_printing = bool(state["printing"][row_int, target_player])
                    if old_printing != next_printing:
                        _emit_printing_property_event_row(
                            state,
                            row=row_int,
                            player=target_player,
                            value=next_printing,
                        )
                if _bonus_has_color_effect(caught_type) and old_color is not None:
                    next_color = int(state["avatar_color"][row_int, target_player])
                    _emit_color_property_event_row(
                        state,
                        row=row_int,
                        player=target_player,
                        color=next_color,
                    )
                _emit_bonus_stack_add_row(
                    state,
                    row=row_int,
                    player=target_player,
                    bonus_id=caught_id,
                    bonus_type=caught_type,
                )

    return catch_counts, stack_append_count


def _bonus_target_players(
    state: Mapping[str, np.ndarray],
    target_group: str,
    *,
    row: int,
    catcher: int,
    player_count: int,
) -> tuple[int, ...]:
    if target_group == BONUS_TARGET_SELF:
        return (catcher,)
    alive = _state_array(state, "alive")
    if alive.ndim != 2 or alive.shape[1] < player_count:
        raise VectorRuntimeError("alive must be a bool array with shape [B,P]")
    if not np.issubdtype(alive.dtype, np.bool_):
        raise VectorRuntimeError("alive must be a bool array with shape [B,P]")
    if target_group == BONUS_TARGET_ENEMY:
        return tuple(
            target
            for target in range(player_count)
            if target != catcher and bool(alive[row, target])
        )
    if target_group == BONUS_TARGET_ALL:
        return tuple(target for target in range(player_count) if bool(alive[row, target]))
    raise VectorRuntimeError(f"unsupported avatar bonus target group {target_group!r}")


def _bonus_all_color_values(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    target_players: tuple[int, ...],
) -> dict[int, int]:
    if not target_players or "avatar_color" not in state:
        return {}
    colors = _state_array(state, "avatar_color")
    if colors.ndim != 2 or not np.issubdtype(colors.dtype, np.integer):
        raise VectorRuntimeError("avatar_color must be an integer array with shape [B,P]")
    snapshot = [int(colors[row, player]) for player in target_players]
    return {
        player: snapshot[(index + 1) % len(snapshot)] for index, player in enumerate(target_players)
    }


def _bonus_optional_player_float(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    row: int,
    player: int,
) -> float | None:
    if name not in state:
        return None
    values = _state_array(state, name)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        raise VectorRuntimeError(f"{name} must be a numeric array with shape [B,P]")
    return float(values[row, player])


def _bonus_optional_player_bool(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    row: int,
    player: int,
) -> bool | None:
    if name not in state:
        return None
    values = _state_array(state, name)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.bool_):
        raise VectorRuntimeError(f"{name} must be a bool array with shape [B,P]")
    return bool(values[row, player])


def _bonus_optional_player_int(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    row: int,
    player: int,
) -> int | None:
    if name not in state:
        return None
    values = _state_array(state, name)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise VectorRuntimeError(f"{name} must be an integer array with shape [B,P]")
    return int(values[row, player])


def _apply_bonus_game_clear(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    row_count: int,
    player_count: int,
) -> None:
    _set_optional_bool_row(
        state,
        "world_active",
        row=row,
        row_count=row_count,
        value=True,
    )
    if "world_body_count" in state:
        _integer_array_shape(state, "world_body_count", shape=(row_count,))[row] = 0
    if "body_active" in state:
        body_arrays = _body_point_arrays(
            state,
            row_count=row_count,
            player_count=player_count,
        )
        body_arrays["active"][row, :] = False
        body_arrays["pos"][row, :, :] = 0.0
        body_arrays["radius"][row, :] = 0.0
        body_arrays["owner"][row, :] = -1
        body_arrays["num"][row, :] = -1
        body_arrays["insert_tick"][row, :] = -1
        body_arrays["insert_kind"][row, :] = -1
        if "break_before" in body_arrays:
            body_arrays["break_before"][row, :] = False
        body_arrays["write_cursor"][row] = 0
        body_arrays["overflow"][row] = False
    _clear_visible_trail_row(
        state,
        row=row,
        row_count=row_count,
        player_count=player_count,
    )


def _apply_bonus_game_borderless(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    row_count: int,
) -> bool:
    borderless = _bool_array_shape(state, "borderless", shape=(row_count,))
    changed = not bool(borderless[row])
    borderless[row] = True
    return changed


def _clear_visible_trail_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    row_count: int,
    player_count: int,
) -> None:
    player_shape = (row_count, player_count)
    if "visible_trail_count" in state:
        _integer_array_shape(state, "visible_trail_count", shape=player_shape)[
            row,
            :player_count,
        ] = 0
    for name in ("has_visible_trail_last", "has_draw_cursor", "has_visual_trail_last"):
        if name in state:
            _bool_array_shape(state, name, shape=player_shape)[row, :player_count] = False
    for name in ("visible_trail_last_pos", "draw_cursor_pos", "visual_trail_last_pos"):
        if name in state:
            _numeric_array_shape(state, name, shape=(*player_shape, 2))[
                row,
                :player_count,
            ] = 0.0
    _clear_visual_trail_points_row(state, row=row, row_count=row_count)


def _clear_visual_trail_points_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    row_count: int,
) -> None:
    if "visual_trail_active" not in state:
        return
    active = _bool_array_shape(
        state,
        "visual_trail_active",
        ndim=2,
        row_count=row_count,
    )
    capacity = active.shape[1]
    trail_shape = (row_count, capacity)
    active[row, :] = False
    _numeric_array_shape(state, "visual_trail_pos", shape=(*trail_shape, 2))[row] = 0.0
    _numeric_array_shape(state, "visual_trail_radius", shape=trail_shape)[row] = 0.0
    _integer_array_shape(state, "visual_trail_owner", shape=trail_shape)[row] = -1
    _bool_array_shape(state, "visual_trail_break_before", shape=trail_shape)[row] = False
    _integer_array_shape(state, "visual_trail_write_cursor", shape=(row_count,))[row] = 0
    _bool_array_shape(state, "visual_trail_overflow", shape=(row_count,))[row] = False


def _caught_bonus_slot(
    active: np.ndarray,
    bonus_pos: np.ndarray,
    bonus_radius: np.ndarray,
    avatar_radius: np.ndarray,
    avatar_pos: np.ndarray,
    *,
    row: int,
    player: int,
) -> int:
    active_slots = np.flatnonzero(active[row])
    for slot in reversed(active_slots):
        slot_int = int(slot)
        dx = bonus_pos[row, slot_int, 0] - avatar_pos[row, player, 0]
        dy = bonus_pos[row, slot_int, 1] - avatar_pos[row, player, 1]
        radius_sum = avatar_radius[row, player] + bonus_radius[row, slot_int]
        if dx * dx + dy * dy < radius_sum * radius_sum:
            return slot_int
    return -1


def _append_bonus_stack_entry(
    stack_arrays: Mapping[str, np.ndarray] | None,
    *,
    row: int,
    player: int,
    bonus_id: int,
    bonus_type: int,
    color_value: int | None = None,
) -> bool:
    if stack_arrays is None:
        return False
    cursor = int(stack_arrays["count"][row, player])
    capacity = int(stack_arrays["id"].shape[2])
    if cursor < 0:
        raise VectorRuntimeError("bonus_stack_count values must be non-negative")
    if cursor >= capacity:
        raise VectorRuntimeError("bonus stack capacity is too small")

    stack_arrays["id"][row, player, cursor] = bonus_id
    stack_arrays["type"][row, player, cursor] = bonus_type
    stack_arrays["duration_ms"][row, player, cursor] = _bonus_avatar_stack_duration_ms(bonus_type)
    stack_arrays["radius_power"][row, player, cursor] = _bonus_radius_power(bonus_type)
    velocity_delta = stack_arrays.get("velocity_delta")
    if isinstance(velocity_delta, np.ndarray):
        velocity_delta[row, player, cursor] = _bonus_velocity_delta(bonus_type)
    inverse_delta = stack_arrays.get("inverse_delta")
    if isinstance(inverse_delta, np.ndarray):
        inverse_delta[row, player, cursor] = _bonus_inverse_delta(bonus_type)
    angular_velocity_per_ms = stack_arrays.get("angular_velocity_per_ms")
    if isinstance(angular_velocity_per_ms, np.ndarray):
        angular_velocity_per_ms[row, player, cursor] = _bonus_angular_velocity_per_ms(bonus_type)
    invincible_delta = stack_arrays.get("invincible_delta")
    if isinstance(invincible_delta, np.ndarray):
        invincible_delta[row, player, cursor] = _bonus_invincible_delta(bonus_type)
    printing_delta = stack_arrays.get("printing_delta")
    if isinstance(printing_delta, np.ndarray):
        printing_delta[row, player, cursor] = _bonus_printing_delta(bonus_type)
    color = stack_arrays.get("color")
    if isinstance(color, np.ndarray):
        color[row, player, cursor] = -1 if color_value is None else int(color_value)
    stack_arrays["count"][row, player] = cursor + 1
    return True


def _append_bonus_game_borderless_stack_entry(
    game_stack_arrays: Mapping[str, np.ndarray] | None,
    *,
    row: int,
    bonus_id: int,
) -> bool:
    if game_stack_arrays is None:
        return False
    cursor = int(game_stack_arrays["count"][row])
    capacity = int(game_stack_arrays["id"].shape[1])
    if cursor < 0:
        raise VectorRuntimeError("bonus_game_stack_count values must be non-negative")
    if cursor >= capacity:
        raise VectorRuntimeError("bonus game stack capacity is too small")

    game_stack_arrays["id"][row, cursor] = bonus_id
    game_stack_arrays["type"][row, cursor] = BONUS_TYPE_GAME_BORDERLESS
    game_stack_arrays["duration_ms"][row, cursor] = BONUS_GAME_BORDERLESS_DURATION_MS
    game_stack_arrays["borderless"][row, cursor] = BONUS_GAME_BORDERLESS_STACK_VALUE
    game_stack_arrays["count"][row] = cursor + 1
    return True


def _bonus_avatar_stack_duration_ms(bonus_type: int) -> int:
    effect = BONUS_RUNTIME_EFFECT_BY_TYPE.get(bonus_type)
    if effect is not None and effect.duration_ms > 0:
        return effect.duration_ms
    raise VectorRuntimeError(f"unsupported avatar bonus stack type {bonus_type}")


def _bonus_radius_power(bonus_type: int) -> int:
    effect = BONUS_RUNTIME_EFFECT_BY_TYPE.get(bonus_type)
    return 0 if effect is None else effect.radius_power


def _bonus_velocity_delta(bonus_type: int) -> float:
    effect = BONUS_RUNTIME_EFFECT_BY_TYPE.get(bonus_type)
    return 0.0 if effect is None else effect.velocity_delta


def _bonus_inverse_delta(bonus_type: int) -> int:
    effect = BONUS_RUNTIME_EFFECT_BY_TYPE.get(bonus_type)
    return 0 if effect is None else effect.inverse_delta


def _bonus_angular_velocity_per_ms(bonus_type: int) -> float:
    effect = BONUS_RUNTIME_EFFECT_BY_TYPE.get(bonus_type)
    return 0.0 if effect is None else effect.angular_velocity_per_ms


def _bonus_invincible_delta(bonus_type: int) -> int:
    effect = BONUS_RUNTIME_EFFECT_BY_TYPE.get(bonus_type)
    return 0 if effect is None else effect.invincible_delta


def _bonus_printing_delta(bonus_type: int) -> int:
    effect = BONUS_RUNTIME_EFFECT_BY_TYPE.get(bonus_type)
    return 0 if effect is None else effect.printing_delta


def _bonus_rotates_color(bonus_type: int) -> bool:
    effect = BONUS_RUNTIME_EFFECT_BY_TYPE.get(bonus_type)
    return bool(effect is not None and effect.rotates_color)


def _bonus_stack_event_value(bonus_type: int) -> float:
    effect = BONUS_RUNTIME_EFFECT_BY_TYPE.get(bonus_type)
    if effect is None:
        return 0.0
    if effect.radius_power:
        return float(effect.radius_power)
    if effect.velocity_delta:
        return float(effect.velocity_delta)
    if effect.inverse_delta:
        return float(effect.inverse_delta)
    if effect.angular_velocity_per_ms:
        return float(effect.angular_velocity_per_ms)
    if effect.invincible_delta:
        return float(effect.invincible_delta)
    if effect.printing_delta:
        return float(effect.printing_delta)
    return 0.0


def _bonus_has_radius_effect(bonus_type: int) -> bool:
    return _bonus_radius_power(bonus_type) != 0


def _bonus_has_velocity_effect(bonus_type: int) -> bool:
    return _bonus_velocity_delta(bonus_type) != 0.0


def _bonus_has_inverse_effect(bonus_type: int) -> bool:
    return _bonus_inverse_delta(bonus_type) != 0


def _bonus_has_angular_velocity_effect(bonus_type: int) -> bool:
    return _bonus_angular_velocity_per_ms(bonus_type) != 0.0


def _bonus_has_invincible_effect(bonus_type: int) -> bool:
    return _bonus_invincible_delta(bonus_type) != 0


def _bonus_has_printing_effect(bonus_type: int) -> bool:
    return _bonus_printing_delta(bonus_type) != 0


def _bonus_has_color_effect(bonus_type: int) -> bool:
    return _bonus_rotates_color(bonus_type)


def _resolve_bonus_avatar_effects_from_stack(
    state: Mapping[str, np.ndarray],
    stack_arrays: Mapping[str, np.ndarray] | None,
    *,
    changed_type: int,
    row: int,
    player: int,
    row_count: int,
) -> None:
    if stack_arrays is None:
        _apply_bonus_avatar_effect_without_stack(
            state,
            changed_type=changed_type,
            row=row,
            player=player,
            row_count=row_count,
        )
        return

    if _bonus_has_radius_effect(changed_type):
        _resolve_bonus_radius_from_stack(
            state,
            stack_arrays,
            row=row,
            player=player,
            row_count=row_count,
        )
    if _bonus_has_velocity_effect(changed_type):
        _resolve_bonus_speed_from_stack(
            state,
            stack_arrays,
            row=row,
            player=player,
            row_count=row_count,
        )
    if _bonus_has_inverse_effect(changed_type):
        _resolve_bonus_inverse_from_stack(
            state,
            stack_arrays,
            row=row,
            player=player,
            row_count=row_count,
        )
    if _bonus_has_angular_velocity_effect(changed_type):
        _resolve_bonus_angular_velocity_from_stack(
            state,
            stack_arrays,
            row=row,
            player=player,
            row_count=row_count,
        )
    if _bonus_has_invincible_effect(changed_type):
        _resolve_bonus_invincible_from_stack(
            state,
            stack_arrays,
            row=row,
            player=player,
            row_count=row_count,
        )
    if _bonus_has_printing_effect(changed_type):
        _resolve_bonus_printing_from_stack(
            state,
            stack_arrays,
            row=row,
            player=player,
            row_count=row_count,
        )
    if _bonus_has_color_effect(changed_type):
        _resolve_bonus_color_from_stack(
            state,
            stack_arrays,
            row=row,
            player=player,
            row_count=row_count,
        )


def _apply_bonus_avatar_effect_without_stack(
    state: Mapping[str, np.ndarray],
    *,
    changed_type: int,
    row: int,
    player: int,
    row_count: int,
) -> None:
    if _bonus_has_radius_effect(changed_type):
        radius = _numeric_array_shape(
            state,
            "radius",
            shape=(row_count, state["radius"].shape[1]),
        )
        radius_power = None
        if "radius_power" in state:
            radius_power = _integer_array_shape(
                state,
                "radius_power",
                shape=radius.shape,
            )
        base_radius = None
        if "base_radius" in state:
            base_radius = _numeric_array_shape(state, "base_radius", shape=radius.shape)
        _apply_bonus_radius_power(
            radius,
            radius_power,
            base_radius,
            row=row,
            player=player,
            radius_power_value=_bonus_radius_power(changed_type),
        )
    if _bonus_has_velocity_effect(changed_type) and "speed" in state:
        speed = _numeric_array_shape(
            state,
            "speed",
            shape=(row_count, state["speed"].shape[1]),
        )
        base_speed = _bonus_base_speed(state, row=row, player=player, shape=speed.shape)
        speed[row, player] = max(
            base_speed + _bonus_velocity_delta(changed_type),
            base_speed / 2.0,
        )
        _apply_speed_adjusted_angular_velocity(
            state,
            row=row,
            player=player,
            row_count=row_count,
        )
    if _bonus_has_inverse_effect(changed_type) and "inverse" in state:
        inverse = _bool_array_shape(
            state,
            "inverse",
            shape=(row_count, state["inverse"].shape[1]),
        )
        base_inverse = _bonus_base_inverse(
            state,
            row=row,
            player=player,
            shape=inverse.shape,
        )
        inverse[row, player] = (int(base_inverse) + _bonus_inverse_delta(changed_type)) % 2 != 0
    if _bonus_has_angular_velocity_effect(changed_type) and "angular_velocity_per_ms" in state:
        angular_velocity = _numeric_array_shape(
            state,
            "angular_velocity_per_ms",
            shape=(row_count, state["angular_velocity_per_ms"].shape[1]),
        )
        angular_velocity[row, player] = _bonus_angular_velocity_per_ms(changed_type)
    if _bonus_has_invincible_effect(changed_type) and "invincible" in state:
        invincible = _bool_array_shape(
            state,
            "invincible",
            shape=(row_count, state["invincible"].shape[1]),
        )
        invincible[row, player] = _bonus_invincible_delta(changed_type) != 0
    if _bonus_has_printing_effect(changed_type) and "printing" in state:
        printing = _bool_array_shape(
            state,
            "printing",
            shape=(row_count, state["printing"].shape[1]),
        )
        printing[row, player] = _bonus_printing_delta(changed_type) > 0
    if _bonus_has_color_effect(changed_type) and "avatar_color" in state:
        color = _integer_array_shape(
            state,
            "avatar_color",
            shape=(row_count, state["avatar_color"].shape[1]),
        )
        color[row, player] = _bonus_base_avatar_color(
            state,
            row=row,
            player=player,
            shape=color.shape,
        )


def _resolve_bonus_radius_from_stack(
    state: Mapping[str, np.ndarray],
    stack_arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    row_count: int,
) -> float:
    radius = _state_array(state, "radius")
    if radius.ndim != 2 or radius.shape[0] != row_count:
        raise VectorRuntimeError("radius must be a numeric array with shape [B,P]")
    if not np.issubdtype(radius.dtype, np.number):
        raise VectorRuntimeError("radius must be a numeric array with shape [B,P]")
    player_shape = radius.shape
    base_radius = None
    if "base_radius" in state:
        base_radius = _numeric_array_shape(state, "base_radius", shape=player_shape)
    radius_power = None
    if "radius_power" in state:
        radius_power = _integer_array_shape(state, "radius_power", shape=player_shape)

    total_power = 0
    cursor = int(stack_arrays["count"][row, player])
    for slot in range(cursor - 1, -1, -1):
        stack_type = int(stack_arrays["type"][row, player, slot])
        if stack_type == BONUS_TYPE_NONE:
            continue
        if not _is_runtime_avatar_bonus(stack_type):
            raise VectorRuntimeError(f"unsupported avatar bonus stack type {stack_type}")
        total_power += _bonus_radius_power(stack_type)

    return _apply_bonus_radius_power(
        radius,
        radius_power,
        base_radius,
        row=row,
        player=player,
        radius_power_value=total_power,
    )


def _apply_bonus_radius_power(
    radius: np.ndarray,
    radius_power: np.ndarray | None,
    base_radius: np.ndarray | None,
    *,
    row: int,
    player: int,
    radius_power_value: int,
) -> float:
    if radius_power is not None:
        radius_power[row, player] = radius_power_value
    base = SOURCE_AVATAR_RADIUS
    if base_radius is not None:
        base = float(base_radius[row, player])
    next_radius = max(base * (2.0 ** float(radius_power_value)), base / 8.0)
    radius[row, player] = next_radius
    return next_radius


def _resolve_bonus_speed_from_stack(
    state: Mapping[str, np.ndarray],
    stack_arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    row_count: int,
) -> float:
    if "speed" not in state:
        return SOURCE_AVATAR_SPEED
    speed = _state_array(state, "speed")
    if speed.ndim != 2 or speed.shape[0] != row_count:
        raise VectorRuntimeError("speed must be a numeric array with shape [B,P]")
    if not np.issubdtype(speed.dtype, np.number):
        raise VectorRuntimeError("speed must be a numeric array with shape [B,P]")

    base_speed = _bonus_base_speed(state, row=row, player=player, shape=speed.shape)
    total_delta = 0.0
    cursor = int(stack_arrays["count"][row, player])
    for slot in range(cursor - 1, -1, -1):
        stack_type = int(stack_arrays["type"][row, player, slot])
        if stack_type == BONUS_TYPE_NONE:
            continue
        if not _is_runtime_avatar_bonus(stack_type):
            raise VectorRuntimeError(f"unsupported avatar bonus stack type {stack_type}")
        total_delta += _bonus_velocity_delta(stack_type)

    next_speed = max(base_speed + total_delta, base_speed / 2.0)
    speed[row, player] = next_speed
    _resolve_bonus_angular_velocity_from_stack(
        state,
        stack_arrays,
        row=row,
        player=player,
        row_count=row_count,
    )
    return next_speed


def _resolve_bonus_inverse_from_stack(
    state: Mapping[str, np.ndarray],
    stack_arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    row_count: int,
) -> bool:
    if "inverse" not in state:
        return False
    inverse = _state_array(state, "inverse")
    if inverse.ndim != 2 or inverse.shape[0] != row_count:
        raise VectorRuntimeError("inverse must be a bool array with shape [B,P]")
    if not np.issubdtype(inverse.dtype, np.bool_):
        raise VectorRuntimeError("inverse must be a bool array with shape [B,P]")

    total_delta = int(
        _bonus_base_inverse(
            state,
            row=row,
            player=player,
            shape=inverse.shape,
        )
    )
    cursor = int(stack_arrays["count"][row, player])
    for slot in range(cursor):
        stack_type = int(stack_arrays["type"][row, player, slot])
        if stack_type == BONUS_TYPE_NONE:
            continue
        if not _is_runtime_avatar_bonus(stack_type):
            raise VectorRuntimeError(f"unsupported avatar bonus stack type {stack_type}")
        total_delta += _bonus_inverse_delta(stack_type)

    inverse[row, player] = total_delta % 2 != 0
    return bool(inverse[row, player])


def _resolve_bonus_angular_velocity_from_stack(
    state: Mapping[str, np.ndarray],
    stack_arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    row_count: int,
) -> float:
    if "angular_velocity_per_ms" not in state:
        return SOURCE_AVATAR_ANGULAR_VELOCITY_PER_MS
    angular_velocity = _state_array(state, "angular_velocity_per_ms")
    if angular_velocity.ndim != 2 or angular_velocity.shape[0] != row_count:
        raise VectorRuntimeError("angular_velocity_per_ms must be a numeric array with shape [B,P]")
    if not np.issubdtype(angular_velocity.dtype, np.number):
        raise VectorRuntimeError("angular_velocity_per_ms must be a numeric array with shape [B,P]")

    next_angular_velocity = _bonus_speed_adjusted_base_angular_velocity_per_ms(
        state,
        row=row,
        player=player,
        shape=angular_velocity.shape,
    )
    cursor = int(stack_arrays["count"][row, player])
    for slot in range(cursor - 1, -1, -1):
        stack_type = int(stack_arrays["type"][row, player, slot])
        if stack_type == BONUS_TYPE_NONE:
            continue
        if not _is_runtime_avatar_bonus(stack_type):
            raise VectorRuntimeError(f"unsupported avatar bonus stack type {stack_type}")
        stack_angular_velocity = _bonus_angular_velocity_per_ms(stack_type)
        if stack_angular_velocity:
            next_angular_velocity = stack_angular_velocity

    angular_velocity[row, player] = next_angular_velocity
    return next_angular_velocity


def _resolve_bonus_invincible_from_stack(
    state: Mapping[str, np.ndarray],
    stack_arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    row_count: int,
) -> bool:
    if "invincible" not in state:
        return False
    invincible = _bool_array_shape(
        state,
        "invincible",
        shape=(row_count, state["invincible"].shape[1]),
    )
    total_delta = int(
        _bonus_base_invincible(
            state,
            row=row,
            player=player,
            shape=invincible.shape,
        )
    )
    cursor = int(stack_arrays["count"][row, player])
    for slot in range(cursor):
        stack_type = int(stack_arrays["type"][row, player, slot])
        if stack_type == BONUS_TYPE_NONE:
            continue
        if not _is_runtime_avatar_bonus(stack_type):
            raise VectorRuntimeError(f"unsupported avatar bonus stack type {stack_type}")
        total_delta += _bonus_invincible_delta(stack_type)

    invincible[row, player] = total_delta > 0
    return bool(invincible[row, player])


def _resolve_bonus_printing_from_stack(
    state: Mapping[str, np.ndarray],
    stack_arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    row_count: int,
) -> bool:
    if "printing" not in state:
        return True
    printing = _bool_array_shape(
        state,
        "printing",
        shape=(row_count, state["printing"].shape[1]),
    )
    total_delta = 1
    cursor = int(stack_arrays["count"][row, player])
    for slot in range(cursor):
        stack_type = int(stack_arrays["type"][row, player, slot])
        if stack_type == BONUS_TYPE_NONE:
            continue
        if not _is_runtime_avatar_bonus(stack_type):
            raise VectorRuntimeError(f"unsupported avatar bonus stack type {stack_type}")
        total_delta += _bonus_printing_delta(stack_type)

    next_printing = total_delta > 0
    old_printing = bool(printing[row, player])
    printing[row, player] = next_printing
    if "print_manager_active" in state:
        _bool_array_shape(
            state,
            "print_manager_active",
            shape=printing.shape,
        )[row, player] = next_printing
    if old_printing != next_printing and "print_manager_distance" in state:
        distance = _numeric_array_shape(
            state,
            "print_manager_distance",
            shape=printing.shape,
        )
        if _has_random_tape_arrays(state):
            distance[row, player] = next_print_manager_random_distance(
                state,
                row=row,
                printing=next_printing,
            )
    if next_printing and "print_manager_last_pos" in state and "pos" in state:
        _numeric_array_shape(
            state,
            "print_manager_last_pos",
            shape=(row_count, printing.shape[1], 2),
        )[row, player] = _numeric_array_shape(
            state,
            "pos",
            shape=(row_count, printing.shape[1], 2),
        )[row, player]
    if old_printing and not next_printing and "visible_trail_count" in state:
        mask = np.zeros(row_count, dtype=bool)
        mask[row] = True
        _clear_visible_trail_batched(state, player, mask)
    return next_printing


def _resolve_bonus_color_from_stack(
    state: Mapping[str, np.ndarray],
    stack_arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    row_count: int,
) -> int:
    if "avatar_color" not in state:
        return player
    color = _integer_array_shape(
        state,
        "avatar_color",
        shape=(row_count, state["avatar_color"].shape[1]),
    )
    next_color = _bonus_base_avatar_color(
        state,
        row=row,
        player=player,
        shape=color.shape,
    )
    stack_color = stack_arrays.get("color")
    cursor = int(stack_arrays["count"][row, player])
    for slot in range(cursor - 1, -1, -1):
        stack_type = int(stack_arrays["type"][row, player, slot])
        if stack_type == BONUS_TYPE_NONE:
            continue
        if not _is_runtime_avatar_bonus(stack_type):
            raise VectorRuntimeError(f"unsupported avatar bonus stack type {stack_type}")
        if _bonus_has_color_effect(stack_type):
            if not isinstance(stack_color, np.ndarray):
                raise VectorRuntimeError("bonus_stack_color is required for color bonus stacks")
            slot_color = int(stack_color[row, player, slot])
            if slot_color >= 0:
                next_color = slot_color

    color[row, player] = next_color
    return int(color[row, player])


def _is_runtime_avatar_bonus(bonus_type: int) -> bool:
    effect = BONUS_RUNTIME_EFFECT_BY_TYPE.get(bonus_type)
    if effect is None:
        return False
    return effect.target_group in (
        BONUS_TARGET_SELF,
        BONUS_TARGET_ENEMY,
        BONUS_TARGET_ALL,
    )


def _bonus_base_speed(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    shape: tuple[int, int],
) -> float:
    if "base_speed" not in state:
        return SOURCE_AVATAR_SPEED
    base_speed = _numeric_array_shape(state, "base_speed", shape=shape)
    return float(base_speed[row, player])


def _bonus_base_inverse(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    shape: tuple[int, int],
) -> bool:
    if "base_inverse" not in state:
        return False
    base_inverse = _bool_array_shape(state, "base_inverse", shape=shape)
    return bool(base_inverse[row, player])


def _bonus_base_angular_velocity_per_ms(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    shape: tuple[int, int],
) -> float:
    if "base_angular_velocity_per_ms" not in state:
        return SOURCE_AVATAR_ANGULAR_VELOCITY_PER_MS
    base_angular_velocity = _numeric_array_shape(
        state,
        "base_angular_velocity_per_ms",
        shape=shape,
    )
    return float(base_angular_velocity[row, player])


def _source_speed_adjusted_angular_velocity_per_ms(
    *,
    speed: float,
    base_speed: float,
    base_angular_velocity_per_ms: float,
) -> float:
    if base_speed <= 0.0 or speed <= 0.0:
        return base_angular_velocity_per_ms
    ratio = speed / base_speed
    return ratio * base_angular_velocity_per_ms + math.log(1.0 / ratio) / 1000.0


def _bonus_speed_adjusted_base_angular_velocity_per_ms(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    shape: tuple[int, int],
) -> float:
    base_angular_velocity = _bonus_base_angular_velocity_per_ms(
        state,
        row=row,
        player=player,
        shape=shape,
    )
    if "speed" not in state:
        return base_angular_velocity
    speed = _numeric_array_shape(state, "speed", shape=shape)
    base_speed = _bonus_base_speed(state, row=row, player=player, shape=shape)
    return _source_speed_adjusted_angular_velocity_per_ms(
        speed=float(speed[row, player]),
        base_speed=base_speed,
        base_angular_velocity_per_ms=base_angular_velocity,
    )


def _apply_speed_adjusted_angular_velocity(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    row_count: int,
) -> None:
    if "angular_velocity_per_ms" not in state:
        return
    angular_velocity = _numeric_array_shape(
        state,
        "angular_velocity_per_ms",
        shape=(row_count, state["angular_velocity_per_ms"].shape[1]),
    )
    angular_velocity[row, player] = _bonus_speed_adjusted_base_angular_velocity_per_ms(
        state,
        row=row,
        player=player,
        shape=angular_velocity.shape,
    )


def _bonus_base_invincible(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    shape: tuple[int, int],
) -> bool:
    if "base_invincible" not in state:
        return False
    base_invincible = _bool_array_shape(state, "base_invincible", shape=shape)
    return bool(base_invincible[row, player])


def _bonus_base_avatar_color(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    shape: tuple[int, int],
) -> int:
    if "base_avatar_color" not in state:
        return player
    base_color = _integer_array_shape(state, "base_avatar_color", shape=shape)
    return int(base_color[row, player])


def _has_random_tape_arrays(state: Mapping[str, np.ndarray]) -> bool:
    return all(
        name in state
        for name in (
            "random_tape_values",
            "random_tape_length",
            "random_tape_cursor",
            "random_tape_draw_count",
            "random_tape_exhausted",
        )
    )


def _expire_bonus_avatar_stacks_batched(
    state: Mapping[str, np.ndarray],
    *,
    timer_advance_ms: np.ndarray,
    events_enabled: bool,
) -> dict[str, int]:
    stack_names = (
        "bonus_stack_count",
        "bonus_stack_id",
        "bonus_stack_type",
        "bonus_stack_duration_ms",
        "bonus_stack_radius_power",
    )
    if not any(name in state for name in stack_names):
        return {}

    tick = _state_array(state, "tick")
    row_count = tick.shape[0]
    radius = _state_array(state, "radius")
    if radius.ndim != 2 or radius.shape[0] != row_count:
        raise VectorRuntimeError("radius must be a numeric array with shape [B,P]")
    if not np.issubdtype(radius.dtype, np.number):
        raise VectorRuntimeError("radius must be a numeric array with shape [B,P]")

    player_count = int(radius.shape[1])
    stack_arrays = _optional_bonus_stack_arrays(
        state,
        row_count=row_count,
        player_count=player_count,
    )
    if stack_arrays is None:
        return {}

    active_rows = (
        ~_bool_array_shape(state, "done", shape=(row_count,))
        & ~_bool_array_shape(state, "overflow", shape=(row_count,))
        & (timer_advance_ms > 0.0)
    )
    expiry_counts: dict[str, int] = {}
    for row in np.flatnonzero(active_rows):
        row_int = int(row)
        advance_ms = float(timer_advance_ms[row_int])
        for player in range(player_count - 1, -1, -1):
            cursor = int(stack_arrays["count"][row_int, player])
            capacity = int(stack_arrays["id"].shape[2])
            if cursor < 0:
                raise VectorRuntimeError("bonus_stack_count values must be non-negative")
            if cursor > capacity:
                raise VectorRuntimeError("bonus_stack_count cannot exceed stack capacity")

            slot = 0
            while slot < cursor:
                stack_type = int(stack_arrays["type"][row_int, player, slot])
                if stack_type == BONUS_TYPE_NONE:
                    slot += 1
                    continue
                effect = BONUS_RUNTIME_EFFECT_BY_TYPE.get(stack_type)
                if (
                    effect is None
                    or effect.expiry_counter is None
                    or not _is_runtime_avatar_bonus(stack_type)
                ):
                    raise VectorRuntimeError(f"unsupported avatar bonus stack type {stack_type}")

                remaining_ms = float(stack_arrays["duration_ms"][row_int, player, slot])
                next_remaining_ms = remaining_ms - advance_ms
                if next_remaining_ms > 0.0:
                    stack_arrays["duration_ms"][row_int, player, slot] = int(
                        round(next_remaining_ms)
                    )
                    slot += 1
                    continue

                bonus_id = int(stack_arrays["id"][row_int, player, slot])
                expired_type = stack_type
                _remove_bonus_stack_slot(
                    stack_arrays,
                    row=row_int,
                    player=player,
                    slot=slot,
                    cursor=cursor,
                )
                cursor -= 1
                expiry_counts[effect.expiry_counter] = (
                    expiry_counts.get(effect.expiry_counter, 0) + 1
                )

                old_radius = float(radius[row_int, player])
                old_speed = _bonus_optional_player_float(
                    state,
                    "speed",
                    row=row_int,
                    player=player,
                )
                old_inverse = _bonus_optional_player_bool(
                    state,
                    "inverse",
                    row=row_int,
                    player=player,
                )
                old_invincible = _bonus_optional_player_bool(
                    state,
                    "invincible",
                    row=row_int,
                    player=player,
                )
                old_printing = _bonus_optional_player_bool(
                    state,
                    "printing",
                    row=row_int,
                    player=player,
                )
                old_color = _bonus_optional_player_int(
                    state,
                    "avatar_color",
                    row=row_int,
                    player=player,
                )
                _resolve_bonus_avatar_effects_from_stack(
                    state,
                    stack_arrays,
                    changed_type=expired_type,
                    row=row_int,
                    player=player,
                    row_count=row_count,
                )
                next_radius = float(radius[row_int, player])
                if events_enabled and old_radius != next_radius:
                    _emit_radius_property_event_row(
                        state,
                        row=row_int,
                        player=player,
                        radius=next_radius,
                    )
                if (
                    events_enabled
                    and _bonus_has_velocity_effect(expired_type)
                    and old_speed is not None
                ):
                    next_speed = float(state["speed"][row_int, player])
                    if old_speed != next_speed:
                        _emit_velocity_property_event_row(
                            state,
                            row=row_int,
                            player=player,
                            velocity=next_speed,
                        )
                if (
                    events_enabled
                    and _bonus_has_inverse_effect(expired_type)
                    and old_inverse is not None
                ):
                    next_inverse = bool(state["inverse"][row_int, player])
                    if old_inverse != next_inverse:
                        _emit_inverse_property_event_row(
                            state,
                            row=row_int,
                            player=player,
                            value=next_inverse,
                        )
                if (
                    events_enabled
                    and _bonus_has_invincible_effect(expired_type)
                    and old_invincible is not None
                ):
                    next_invincible = bool(state["invincible"][row_int, player])
                    if old_invincible != next_invincible:
                        _emit_invincible_property_event_row(
                            state,
                            row=row_int,
                            player=player,
                            value=next_invincible,
                        )
                if (
                    events_enabled
                    and _bonus_has_printing_effect(expired_type)
                    and old_printing is not None
                ):
                    next_printing = bool(state["printing"][row_int, player])
                    if old_printing != next_printing:
                        _emit_printing_property_event_row(
                            state,
                            row=row_int,
                            player=player,
                            value=next_printing,
                        )
                if (
                    events_enabled
                    and _bonus_has_color_effect(expired_type)
                    and old_color is not None
                ):
                    next_color = int(state["avatar_color"][row_int, player])
                    _emit_color_property_event_row(
                        state,
                        row=row_int,
                        player=player,
                        color=next_color,
                    )
                if events_enabled:
                    _emit_bonus_stack_remove_row(
                        state,
                        row=row_int,
                        player=player,
                        bonus_id=bonus_id,
                        bonus_type=expired_type,
                    )

    return expiry_counts


def _expire_bonus_game_borderless_stacks_batched(
    state: Mapping[str, np.ndarray],
    *,
    timer_advance_ms: np.ndarray,
    events_enabled: bool,
) -> int:
    stack_names = (
        "bonus_game_stack_count",
        "bonus_game_stack_id",
        "bonus_game_stack_type",
        "bonus_game_stack_duration_ms",
        "bonus_game_stack_borderless",
    )
    if not any(name in state for name in stack_names):
        return 0

    tick = _state_array(state, "tick")
    row_count = tick.shape[0]
    game_stack_arrays = _optional_bonus_game_stack_arrays(
        state,
        row_count=row_count,
    )
    if game_stack_arrays is None:
        return 0
    borderless = _bool_array_shape(state, "borderless", shape=(row_count,))

    active_rows = (
        ~_bool_array_shape(state, "done", shape=(row_count,))
        & ~_bool_array_shape(state, "overflow", shape=(row_count,))
        & (timer_advance_ms > 0.0)
    )
    expiry_count = 0
    for row in np.flatnonzero(active_rows):
        row_int = int(row)
        advance_ms = float(timer_advance_ms[row_int])
        cursor = int(game_stack_arrays["count"][row_int])
        capacity = int(game_stack_arrays["id"].shape[1])
        if cursor < 0:
            raise VectorRuntimeError("bonus_game_stack_count values must be non-negative")
        if cursor > capacity:
            raise VectorRuntimeError("bonus_game_stack_count cannot exceed stack capacity")

        slot = 0
        while slot < cursor:
            stack_type = int(game_stack_arrays["type"][row_int, slot])
            if stack_type == BONUS_TYPE_NONE:
                slot += 1
                continue
            if stack_type != BONUS_TYPE_GAME_BORDERLESS:
                raise VectorRuntimeError("only BonusGameBorderless game stack expiry is supported")

            remaining_ms = float(game_stack_arrays["duration_ms"][row_int, slot])
            next_remaining_ms = remaining_ms - advance_ms
            if next_remaining_ms > 0.0:
                game_stack_arrays["duration_ms"][row_int, slot] = int(round(next_remaining_ms))
                slot += 1
                continue

            _remove_bonus_game_stack_slot(
                game_stack_arrays,
                row=row_int,
                slot=slot,
                cursor=cursor,
            )
            cursor -= 1
            expiry_count += 1

            borderless_changed = _resolve_bonus_game_borderless_from_stack(
                borderless,
                game_stack_arrays,
                row=row_int,
            )
            if events_enabled and borderless_changed:
                _emit_borderless_row(
                    state,
                    row=row_int,
                    value=bool(borderless[row_int]),
                )

    return expiry_count


def _remove_bonus_stack_slot(
    stack_arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    slot: int,
    cursor: int,
) -> None:
    last = cursor - 1
    if slot < last:
        for name in (
            "id",
            "type",
            "duration_ms",
            "radius_power",
            "velocity_delta",
            "inverse_delta",
            "angular_velocity_per_ms",
            "invincible_delta",
            "printing_delta",
            "color",
        ):
            values = stack_arrays.get(name)
            if not isinstance(values, np.ndarray):
                continue
            values[row, player, slot:last] = values[
                row,
                player,
                slot + 1 : cursor,
            ]

    stack_arrays["id"][row, player, last] = -1
    stack_arrays["type"][row, player, last] = BONUS_TYPE_NONE
    stack_arrays["duration_ms"][row, player, last] = 0
    stack_arrays["radius_power"][row, player, last] = 0
    velocity_delta = stack_arrays.get("velocity_delta")
    if isinstance(velocity_delta, np.ndarray):
        velocity_delta[row, player, last] = 0.0
    inverse_delta = stack_arrays.get("inverse_delta")
    if isinstance(inverse_delta, np.ndarray):
        inverse_delta[row, player, last] = 0
    angular_velocity_per_ms = stack_arrays.get("angular_velocity_per_ms")
    if isinstance(angular_velocity_per_ms, np.ndarray):
        angular_velocity_per_ms[row, player, last] = 0.0
    invincible_delta = stack_arrays.get("invincible_delta")
    if isinstance(invincible_delta, np.ndarray):
        invincible_delta[row, player, last] = 0
    printing_delta = stack_arrays.get("printing_delta")
    if isinstance(printing_delta, np.ndarray):
        printing_delta[row, player, last] = 0
    color = stack_arrays.get("color")
    if isinstance(color, np.ndarray):
        color[row, player, last] = -1
    stack_arrays["count"][row, player] = last


def _remove_bonus_game_stack_slot(
    game_stack_arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    slot: int,
    cursor: int,
) -> None:
    last = cursor - 1
    if slot < last:
        for name in ("id", "type", "duration_ms", "borderless"):
            game_stack_arrays[name][row, slot:last] = game_stack_arrays[name][
                row,
                slot + 1 : cursor,
            ]

    game_stack_arrays["id"][row, last] = -1
    game_stack_arrays["type"][row, last] = BONUS_TYPE_NONE
    game_stack_arrays["duration_ms"][row, last] = 0
    game_stack_arrays["borderless"][row, last] = 0
    game_stack_arrays["count"][row] = last


def _resolve_bonus_game_borderless_from_stack(
    borderless: np.ndarray,
    game_stack_arrays: Mapping[str, np.ndarray],
    *,
    row: int,
) -> bool:
    next_borderless = False
    cursor = int(game_stack_arrays["count"][row])
    for slot in range(cursor):
        stack_type = int(game_stack_arrays["type"][row, slot])
        if stack_type == BONUS_TYPE_NONE:
            continue
        if stack_type != BONUS_TYPE_GAME_BORDERLESS:
            raise VectorRuntimeError(
                "only BonusGameBorderless game stack borderless resolution is supported"
            )
        next_borderless = next_borderless or bool(game_stack_arrays["borderless"][row, slot])

    old_borderless = bool(borderless[row])
    borderless[row] = next_borderless
    return old_borderless != next_borderless


def _advance_pre_step_timers_batched(
    state: Mapping[str, np.ndarray],
    *,
    timer_advance_ms: Any | None,
    events_enabled: bool,
) -> dict[str, int]:
    row_count = state["tick"].shape[0]
    advance = _row_float_input(
        np.zeros(row_count, dtype=np.float64) if timer_advance_ms is None else timer_advance_ms,
        row_count,
        "timer_advance_ms",
    )

    counters = {
        "pre_step_timer_advances": 0,
        "pre_step_timer_fires": 0,
        "print_manager_delayed_start_fires": 0,
        "print_manager_delayed_start_points": 0,
        "body_overflow_attempts": 0,
        "bonus_self_small_expiries": 0,
        "bonus_self_slow_expiries": 0,
        "bonus_self_fast_expiries": 0,
        "bonus_self_master_expiries": 0,
        "bonus_enemy_slow_expiries": 0,
        "bonus_enemy_fast_expiries": 0,
        "bonus_enemy_big_expiries": 0,
        "bonus_enemy_inverse_expiries": 0,
        "bonus_enemy_straight_angle_expiries": 0,
        "bonus_game_borderless_expiries": 0,
        "bonus_all_color_expiries": 0,
    }

    avatar_expiries = _expire_bonus_avatar_stacks_batched(
        state,
        timer_advance_ms=advance,
        events_enabled=events_enabled,
    )
    for name, value in avatar_expiries.items():
        counters[name] += value
    counters["bonus_game_borderless_expiries"] += _expire_bonus_game_borderless_stacks_batched(
        state,
        timer_advance_ms=advance,
        events_enabled=events_enabled,
    )
    if "timer_active" not in state or not state["timer_active"].any():
        return counters

    active_rows = np.flatnonzero(~state["done"] & ~state["overflow"])
    for row in active_rows:
        row_int = int(row)
        active_slots = np.flatnonzero(state["timer_active"][row_int])
        if active_slots.size == 0:
            continue

        counters["pre_step_timer_advances"] += 1
        state["timer_remaining_ms"][row_int, active_slots] -= advance[row_int]
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
                fired, points, overflowed = _fire_print_manager_start_timer_batched(
                    state,
                    row=row_int,
                    player=player,
                    events_enabled=events_enabled,
                )
                counters["pre_step_timer_fires"] += fired
                counters["print_manager_delayed_start_fires"] += fired
                counters["print_manager_delayed_start_points"] += points
                counters["body_overflow_attempts"] += overflowed
            _clear_state_timer_slot(state, row=row_int, slot=slot)

    return counters


def _fire_print_manager_start_timer_batched(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    events_enabled: bool,
) -> tuple[int, int, int]:
    if player < 0 or player >= state["print_manager_active"].shape[1]:
        raise VectorRuntimeError(f"timer references unknown player index {player}")
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
        points, overflowed = _append_body_points_batched(
            state,
            player=player,
            write_mask=row_mask,
            insert_kind=BODY_KIND_IMPORTANT,
        )
        if events_enabled:
            _emit_point_events_batched(state, player, row_mask, important=True)

    if events_enabled:
        _emit_printing_property_events_batched(state, player, row_mask)
    assign_print_manager_random_distances(state, player=player, mask=row_mask)
    return 1, points, overflowed


def _clear_visible_trail_batched(
    state: Mapping[str, np.ndarray],
    player: int,
    mask: np.ndarray,
) -> None:
    state["visible_trail_count"][mask, player] = 0
    state["has_visible_trail_last"][mask, player] = False
    state["visible_trail_last_pos"][mask, player] = 0.0
    state["has_draw_cursor"][mask, player] = False
    state["draw_cursor_pos"][mask, player] = 0.0
    if "has_visual_trail_last" in state:
        state["has_visual_trail_last"][mask, player] = False
        state["visual_trail_last_pos"][mask, player] = 0.0


def _append_death_list_batched(
    state: Mapping[str, np.ndarray],
    *,
    player: int,
    death_mask: np.ndarray,
    row_count: int,
    death_cause: int | np.ndarray = DEATH_CAUSE_BODY_UNKNOWN,
    death_hit_owner: int | np.ndarray = -1,
) -> None:
    if "death_count" not in state and "death_player" not in state:
        if "death_cause" in state or "death_hit_owner" in state:
            raise VectorRuntimeError(
                "death_cause and death_hit_owner require death_count and death_player"
            )
        return
    if "death_count" not in state or "death_player" not in state:
        raise VectorRuntimeError("death_count and death_player must be supplied together")

    death_count = _state_array(state, "death_count")
    death_player = _state_array(state, "death_player")
    if death_count.shape != (row_count,) or not np.issubdtype(
        death_count.dtype,
        np.integer,
    ):
        raise VectorRuntimeError("death_count must be an integer array with shape [B]")
    if (
        death_player.ndim != 2
        or death_player.shape[0] != row_count
        or not np.issubdtype(death_player.dtype, np.integer)
    ):
        raise VectorRuntimeError("death_player must be an integer array with shape [B,D]")

    capacity = int(death_player.shape[1])
    cause_array = _optional_death_detail_array(
        state,
        "death_cause",
        row_count=row_count,
        capacity=capacity,
    )
    hit_owner_array = _optional_death_detail_array(
        state,
        "death_hit_owner",
        row_count=row_count,
        capacity=capacity,
    )
    for row in np.flatnonzero(death_mask):
        row_int = int(row)
        cursor = int(death_count[row_int])
        if cursor < 0:
            raise VectorRuntimeError("death_count values must be non-negative")
        if cursor >= capacity:
            raise VectorRuntimeError("death_player capacity is too small")
        death_player[row_int, cursor] = player
        if cause_array is not None:
            cause_array[row_int, cursor] = _death_detail_value(
                death_cause,
                row=row_int,
                row_count=row_count,
                field="death_cause",
            )
        if hit_owner_array is not None:
            hit_owner_array[row_int, cursor] = _death_detail_value(
                death_hit_owner,
                row=row_int,
                row_count=row_count,
                field="death_hit_owner",
            )
        death_count[row_int] = cursor + 1


def _optional_death_detail_array(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    row_count: int,
    capacity: int,
) -> np.ndarray | None:
    if name not in state:
        return None
    array = _state_array(state, name)
    if array.shape != (row_count, capacity) or not np.issubdtype(array.dtype, np.integer):
        raise VectorRuntimeError(f"{name} must be an integer array with shape [B,D]")
    return array


def _death_detail_value(
    value: int | np.ndarray,
    *,
    row: int,
    row_count: int,
    field: str,
) -> int:
    array = np.asarray(value)
    if array.ndim == 0:
        return int(array.item())
    if array.shape != (row_count,):
        raise VectorRuntimeError(f"{field} detail values must be scalar or shape [B]")
    return int(array[row])


def _apply_terminal_score_for_rows_batched(
    state: Mapping[str, np.ndarray],
    death_rows: np.ndarray,
    *,
    player_count: int,
    phase_timers: dict[str, float] | None = None,
    event_mode: str = EVENT_MODE_DEBUG,
) -> int:
    events_enabled = _events_enabled(event_mode)
    alive_counts = state["alive"][:, :player_count].sum(axis=1)
    terminal_rows = death_rows & (alive_counts <= 1)
    if "round_done" in state:
        terminal_rows &= ~np.asarray(state["round_done"], dtype=bool)
    rows = np.flatnonzero(terminal_rows)
    if rows.size == 0:
        return 0

    for row in rows:
        row_int = int(row)
        started = _timer_start(phase_timers)
        live_players = np.flatnonzero(state["alive"][row_int, :player_count])
        winner_value = -1
        if live_players.size == 1:
            winner_value = int(live_players[0])
            state["round_score"][row_int, winner_value] += max(player_count - 1, 1)
        _timer_add(phase_timers, "terminal_score_state_sec", started)

        if events_enabled and winner_value >= 0:
            started = _timer_start(phase_timers)
            _emit_score_row(
                state,
                row=row_int,
                player=winner_value,
                event_type=EVENT_SCORE_ROUND,
            )
            _timer_add(phase_timers, "event_emit_sec", started)

        started = _timer_start(phase_timers)
        state["score"][row_int, :player_count] += state["round_score"][
            row_int,
            :player_count,
        ]
        _timer_add(phase_timers, "terminal_score_state_sec", started)

        if events_enabled:
            started = _timer_start(phase_timers)
            for player in range(player_count - 1, -1, -1):
                _emit_score_row(
                    state,
                    row=row_int,
                    player=player,
                    event_type=EVENT_SCORE,
                )
            _timer_add(phase_timers, "event_emit_sec", started)

        started = _timer_start(phase_timers)
        state["round_score"][row_int, :player_count] = 0
        _timer_add(phase_timers, "terminal_score_state_sec", started)

        if events_enabled:
            started = _timer_start(phase_timers)
            _emit_round_end_row(state, row=row_int, winner=winner_value)
            _timer_add(phase_timers, "event_emit_sec", started)

        started = _timer_start(phase_timers)
        _mark_terminal_lifecycle_row(
            state,
            row=row_int,
            row_count=terminal_rows.shape[0],
            player_count=player_count,
            winner=winner_value,
        )
        _timer_add(phase_timers, "terminal_score_state_sec", started)
    return int(rows.size)


def _mark_terminal_lifecycle_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    row_count: int,
    player_count: int,
    winner: int,
) -> None:
    reason = TERMINAL_REASON_SURVIVOR_WIN if winner >= 0 else TERMINAL_REASON_ALL_DEAD_DRAW
    if _uses_round_lifecycle(state):
        _mark_round_lifecycle_row(
            state,
            row=row,
            row_count=row_count,
            player_count=player_count,
            winner=winner,
            reason=reason,
        )
        return

    _set_optional_bool_row(state, "done", row=row, row_count=row_count, value=True)
    _set_optional_bool_row(
        state,
        "terminated",
        row=row,
        row_count=row_count,
        value=True,
    )
    _set_optional_bool_row(
        state,
        "reset_pending",
        row=row,
        row_count=row_count,
        value=True,
    )
    _set_optional_int_row(
        state,
        "terminal_reason",
        row=row,
        row_count=row_count,
        value=reason,
    )
    _set_optional_bool_row(
        state,
        "draw",
        row=row,
        row_count=row_count,
        value=winner < 0,
    )
    _set_optional_int_row(
        state,
        "winner",
        row=row,
        row_count=row_count,
        value=winner,
    )


def _uses_round_lifecycle(state: Mapping[str, np.ndarray]) -> bool:
    return any(
        name in state
        for name in (
            "round_done",
            "match_done",
            "warmdown_pending",
            "round_winner",
            "match_winner",
        )
    )


def _mark_round_lifecycle_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    row_count: int,
    player_count: int,
    winner: int,
    reason: int,
) -> None:
    match_done, match_winner = _match_done_for_row(
        state,
        row=row,
        row_count=row_count,
        player_count=player_count,
    )

    _set_optional_bool_row(state, "round_done", row=row, row_count=row_count, value=True)
    _set_optional_bool_row(
        state,
        "match_done",
        row=row,
        row_count=row_count,
        value=match_done,
    )
    _set_optional_bool_row(
        state,
        "warmdown_pending",
        row=row,
        row_count=row_count,
        value=not match_done,
    )
    _set_optional_int_row(
        state,
        "round_winner",
        row=row,
        row_count=row_count,
        value=winner,
    )
    _set_optional_int_row(
        state,
        "match_winner",
        row=row,
        row_count=row_count,
        value=match_winner,
    )
    _set_optional_int_row(
        state,
        "terminal_reason",
        row=row,
        row_count=row_count,
        value=reason,
    )
    _set_optional_bool_row(
        state,
        "draw",
        row=row,
        row_count=row_count,
        value=winner < 0,
    )
    _set_optional_int_row(
        state,
        "winner",
        row=row,
        row_count=row_count,
        value=winner,
    )

    if match_done:
        _set_optional_bool_row(state, "done", row=row, row_count=row_count, value=True)
        _set_optional_bool_row(
            state,
            "terminated",
            row=row,
            row_count=row_count,
            value=True,
        )
        _set_optional_bool_row(
            state,
            "reset_pending",
            row=row,
            row_count=row_count,
            value=True,
        )
        return

    _set_optional_bool_row(state, "done", row=row, row_count=row_count, value=False)
    _set_optional_bool_row(
        state,
        "terminated",
        row=row,
        row_count=row_count,
        value=False,
    )
    _set_optional_bool_row(
        state,
        "reset_pending",
        row=row,
        row_count=row_count,
        value=False,
    )
    _schedule_warmdown_end_timer(
        state,
        row=row,
        row_count=row_count,
        delay_ms=SOURCE_ROUND_WARMDOWN_MS,
    )


def _match_done_for_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    row_count: int,
    player_count: int,
) -> tuple[bool, int]:
    if "max_score" not in state:
        return False, -1

    max_score = _state_array(state, "max_score")
    if max_score.shape != (row_count,) or not np.issubdtype(max_score.dtype, np.number):
        raise VectorRuntimeError("max_score must be a numeric array with shape [B]")
    threshold = float(max_score[row])
    if not np.isfinite(threshold) or threshold <= 0.0:
        return False, -1

    scores = _state_array(state, "score")
    if scores.ndim != 2 or scores.shape[0] != row_count or scores.shape[1] < player_count:
        raise VectorRuntimeError("score must have shape [B,P]")
    leaders = np.flatnonzero(scores[row, :player_count].astype(np.float64) >= threshold)
    if leaders.size == 1:
        return True, int(leaders[0])
    return False, -1


def _schedule_warmdown_end_timer(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    row_count: int,
    delay_ms: float,
) -> None:
    timer_arrays = _optional_timer_arrays_for_row_lifecycle(
        state,
        row_count=row_count,
    )
    active = timer_arrays["active"]
    free_slots = np.flatnonzero(~active[row])
    if free_slots.size == 0:
        timer_arrays["overflow"][row] = True
        _set_optional_bool_row(state, "overflow", row=row, row_count=row_count, value=True)
        return

    slot = int(free_slots[0])
    active[row, slot] = True
    timer_arrays["remaining_ms"][row, slot] = delay_ms
    timer_arrays["kind"][row, slot] = TIMER_KIND_WARMDOWN_END
    timer_arrays["player"][row, slot] = TIMER_PLAYER_NONE
    timer_arrays["seq"][row, slot] = _next_timer_seq(timer_arrays, row=row)


def _optional_timer_arrays_for_row_lifecycle(
    state: Mapping[str, np.ndarray],
    *,
    row_count: int,
) -> dict[str, np.ndarray]:
    names = (
        "timer_active",
        "timer_remaining_ms",
        "timer_kind",
        "timer_player",
        "timer_seq",
        "timer_overflow",
    )
    missing = [name for name in names if name not in state]
    if missing:
        missing_names = ", ".join(missing)
        raise VectorRuntimeError(
            f"round lifecycle warmdown requires timer arrays; missing {missing_names}"
        )
    return _warmup_timer_arrays(state, row_count=row_count)


def _next_timer_seq(
    timer_arrays: Mapping[str, np.ndarray],
    *,
    row: int,
) -> int:
    if not bool(timer_arrays["active"][row].any()):
        return 0
    return int(timer_arrays["seq"][row, timer_arrays["active"][row]].max()) + 1


def _set_optional_bool_row(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    row: int,
    row_count: int,
    value: bool,
) -> None:
    if name not in state:
        return
    array = _state_array(state, name)
    if array.shape != (row_count,) or array.dtype != np.bool_:
        raise VectorRuntimeError(f"{name} must be a bool array with shape [B]")
    array[row] = value


def _set_optional_int_row(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    row: int,
    row_count: int,
    value: int,
) -> None:
    if name not in state:
        return
    array = _state_array(state, name)
    if array.shape != (row_count,) or not np.issubdtype(array.dtype, np.integer):
        raise VectorRuntimeError(f"{name} must be an integer array with shape [B]")
    array[row] = value


def _emit_position_events_batched(
    state: Mapping[str, np.ndarray],
    player: int,
    mask: np.ndarray,
) -> None:
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return
    _emit_event_rows_batched(
        state,
        rows,
        event_type=EVENT_POSITION,
        player=player,
        float_values=state["pos"][rows, player],
    )


def _emit_point_events_batched(
    state: Mapping[str, np.ndarray],
    player: int,
    mask: np.ndarray,
    *,
    important: bool,
) -> None:
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return
    _emit_event_rows_batched(
        state,
        rows,
        event_type=EVENT_POINT,
        player=player,
        bool_value=1 if important else 0,
        float_values=state["pos"][rows, player],
    )


def _emit_printing_property_events_batched(
    state: Mapping[str, np.ndarray],
    player: int,
    mask: np.ndarray,
) -> None:
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return
    _emit_event_rows_batched(
        state,
        rows,
        event_type=EVENT_PROPERTY,
        player=player,
        bool_value=state["printing"][rows, player].astype(np.int8),
        int_values=np.tile(
            np.asarray([PROPERTY_PRINTING, 0], dtype=np.int32),
            (rows.size, 1),
        ),
    )


def _emit_die_events_batched(
    state: Mapping[str, np.ndarray],
    player: int,
    mask: np.ndarray,
    *,
    other_player: np.ndarray | None = None,
    old: bool | None = None,
) -> None:
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return
    _emit_event_rows_batched(
        state,
        rows,
        event_type=EVENT_DIE,
        player=player,
        other_values=other_player,
        bool_value=-1 if old is None else 1 if old else 0,
    )


def _emit_score_events_batched(
    state: Mapping[str, np.ndarray],
    player: int,
    mask: np.ndarray,
    *,
    event_type: int,
) -> None:
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return
    _emit_event_rows_batched(
        state,
        rows,
        event_type=event_type,
        player=player,
        int_values=np.stack(
            (state["score"][rows, player], state["round_score"][rows, player]),
            axis=1,
        ),
    )


def _emit_score_row(
    state: Mapping[str, np.ndarray],
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
    state: Mapping[str, np.ndarray],
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


def _emit_bonus_clear_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    bonus_id: int,
) -> None:
    _emit_event_row(
        state,
        row=row,
        event_type=EVENT_BONUS_CLEAR,
        int_values=(bonus_id, 0),
    )


def _emit_bonus_pop_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    bonus_id: int,
    bonus_type_code: int,
    x: float,
    y: float,
) -> None:
    _emit_event_row(
        state,
        row=row,
        event_type=EVENT_BONUS_POP,
        int_values=(bonus_id, bonus_type_code),
        float_values=(x, y),
    )


def _emit_clear_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
) -> None:
    _emit_event_row(
        state,
        row=row,
        event_type=EVENT_CLEAR,
    )


def _emit_borderless_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    value: bool,
) -> None:
    _emit_event_row(
        state,
        row=row,
        event_type=EVENT_BORDERLESS,
        bool_value=1 if value else 0,
    )


def _emit_radius_property_event_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    radius: float,
) -> None:
    _emit_event_row(
        state,
        row=row,
        event_type=EVENT_PROPERTY,
        player=player,
        int_values=(PROPERTY_RADIUS, 0),
        float_values=(radius, 0.0),
    )


def _emit_velocity_property_event_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    velocity: float,
) -> None:
    _emit_event_row(
        state,
        row=row,
        event_type=EVENT_PROPERTY,
        player=player,
        int_values=(PROPERTY_VELOCITY, 0),
        float_values=(velocity, 0.0),
    )


def _emit_inverse_property_event_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    value: bool,
) -> None:
    _emit_event_row(
        state,
        row=row,
        event_type=EVENT_PROPERTY,
        player=player,
        bool_value=1 if value else 0,
        int_values=(PROPERTY_INVERSE, 0),
    )


def _emit_invincible_property_event_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    value: bool,
) -> None:
    _emit_event_row(
        state,
        row=row,
        event_type=EVENT_PROPERTY,
        player=player,
        bool_value=1 if value else 0,
        int_values=(PROPERTY_INVINCIBLE, 0),
    )


def _emit_printing_property_event_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    value: bool,
) -> None:
    _emit_event_row(
        state,
        row=row,
        event_type=EVENT_PROPERTY,
        player=player,
        bool_value=1 if value else 0,
        int_values=(PROPERTY_PRINTING, 0),
    )


def _emit_color_property_event_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    color: int,
) -> None:
    _emit_event_row(
        state,
        row=row,
        event_type=EVENT_PROPERTY,
        player=player,
        int_values=(PROPERTY_COLOR, int(color)),
    )


def _emit_bonus_stack_add_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    bonus_id: int,
    bonus_type: int = BONUS_TYPE_SELF_SMALL,
) -> None:
    _emit_event_row(
        state,
        row=row,
        event_type=EVENT_BONUS_STACK,
        player=player,
        bool_value=BONUS_STACK_METHOD_ADD,
        int_values=(bonus_id, bonus_type),
        float_values=(
            float(_bonus_avatar_stack_duration_ms(bonus_type)),
            _bonus_stack_event_value(bonus_type),
        ),
    )


def _emit_bonus_stack_remove_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    player: int,
    bonus_id: int,
    bonus_type: int = BONUS_TYPE_SELF_SMALL,
) -> None:
    _emit_event_row(
        state,
        row=row,
        event_type=EVENT_BONUS_STACK,
        player=player,
        bool_value=BONUS_STACK_METHOD_REMOVE,
        int_values=(bonus_id, bonus_type),
        float_values=(
            float(_bonus_avatar_stack_duration_ms(bonus_type)),
            _bonus_stack_event_value(bonus_type),
        ),
    )


def _emit_event_rows_batched(
    state: Mapping[str, np.ndarray],
    rows: np.ndarray,
    *,
    event_type: int,
    player: int = -1,
    other_values: np.ndarray | None = None,
    bool_value: int | np.ndarray = -1,
    int_values: np.ndarray | None = None,
    float_values: np.ndarray | None = None,
) -> None:
    cursors = state["event_count"][rows].astype(np.int64, copy=False)
    capacity = state["event_type"].shape[1]
    can_emit = cursors < capacity

    overflow_rows = rows[~can_emit]
    if overflow_rows.size:
        state["event_overflow"][overflow_rows] = True
        state["event_overflow_attempts"][overflow_rows] += 1

    emit_rows = rows[can_emit]
    if emit_rows.size == 0:
        return

    emit_cursors = cursors[can_emit]
    state["event_count"][emit_rows] += 1
    state["event_mask"][emit_rows, emit_cursors] = True
    state["event_type"][emit_rows, emit_cursors] = event_type
    state["event_player"][emit_rows, emit_cursors] = player
    state["event_other"][emit_rows, emit_cursors] = -1
    if other_values is not None:
        state["event_other"][emit_rows, emit_cursors] = other_values[emit_rows]
    if isinstance(bool_value, np.ndarray):
        state["event_bool"][emit_rows, emit_cursors] = bool_value[can_emit]
    else:
        state["event_bool"][emit_rows, emit_cursors] = bool_value
    if int_values is not None:
        state["event_value_i"][emit_rows, emit_cursors] = int_values[can_emit]
    if float_values is not None:
        state["event_value_f"][emit_rows, emit_cursors] = float_values[can_emit]


def _emit_event_row(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    event_type: int,
    player: int = -1,
    other: int = -1,
    bool_value: int = -1,
    int_values: tuple[int, int] | None = None,
    float_values: tuple[float, float] | np.ndarray | None = None,
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


def _body_death_causes_from_hit_owners(
    hit_owners: np.ndarray,
    *,
    player: int,
) -> np.ndarray:
    causes = np.full(hit_owners.shape, DEATH_CAUSE_BODY_UNKNOWN, dtype=np.int16)
    causes[hit_owners == player] = DEATH_CAUSE_OWN_TRAIL
    causes[(hit_owners >= 0) & (hit_owners != player)] = DEATH_CAUSE_OPPONENT_TRAIL
    return causes


def death_cause_name(cause: int) -> str:
    """Return the public name for a death-cause code."""

    cause_int = int(cause)
    if cause_int < 0 or cause_int >= len(DEATH_CAUSE_NAMES):
        raise VectorRuntimeError(f"unknown death cause code {cause_int}")
    return DEATH_CAUSE_NAMES[cause_int]


def death_cause_name_array(causes: np.ndarray) -> np.ndarray:
    """Return object names with the same shape as a death-cause array."""

    cause_array = np.asarray(causes)
    names = [death_cause_name(int(cause)) for cause in cause_array.reshape(-1)]
    return np.asarray(names, dtype=object).reshape(cause_array.shape)


def next_print_manager_random_distance(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    printing: bool,
) -> float:
    """Consume one row-local random value for the next PrintManager distance."""

    arrays = _random_tape_arrays(state)
    row_count = arrays["length"].shape[0]
    if not isinstance(row, int) or isinstance(row, bool) or row < 0 or row >= row_count:
        raise VectorRuntimeError(f"unknown row index {row}")

    length = int(arrays["length"][row])
    cursor = int(arrays["cursor"][row])
    draw_count = int(arrays["draw_count"][row])
    tape_capacity = int(arrays["values"].shape[1])
    if length < 0 or cursor < 0 or draw_count < 0:
        raise VectorRuntimeError("random tape cursor, length, and draw_count must be non-negative")
    if length > tape_capacity:
        raise VectorRuntimeError("random_tape_length cannot exceed random_tape_values capacity")
    if cursor > length:
        raise VectorRuntimeError("random_tape_cursor cannot exceed random_tape_length")

    if length <= 0:
        arrays["draw_count"][row] += 1
        return _default_print_manager_random_distance(printing=printing)

    if cursor >= length:
        arrays["exhausted"][row] = True
        raise VectorRuntimeError(f"row {row} Math.random tape exhausted after {cursor} calls")

    random_value = float(arrays["values"][row, cursor])
    if not np.isfinite(random_value) or random_value < 0.0 or random_value >= 1.0:
        raise VectorRuntimeError(f"row {row} Math.random tape value must be in [0, 1)")

    arrays["cursor"][row] = cursor + 1
    arrays["draw_count"][row] += 1
    if printing:
        return 60.0 * (0.3 + random_value * 0.7)
    return 5.0 * (0.8 + random_value * 0.5)


def validate_step_input(step_input: VectorStepInput) -> None:
    """Validate the production batch-step shape contract."""

    if not isinstance(step_input, VectorStepInput):
        raise VectorRuntimeError("step_many expects a VectorStepInput")
    if not isinstance(step_input.player_count, int) or isinstance(
        step_input.player_count,
        bool,
    ):
        raise VectorRuntimeError("player_count must be a positive integer")
    if step_input.player_count < 1:
        raise VectorRuntimeError("player_count must be a positive integer")
    if step_input.event_mode not in EVENT_MODES:
        raise VectorRuntimeError("event_mode must be 'debug-event' or 'no-event'")
    if step_input.death_mode not in DEATH_MODES:
        raise VectorRuntimeError("death_mode must be 'normal' or 'profile_no_death'")

    tick = _state_array(step_input.state, "tick")
    if not np.issubdtype(tick.dtype, np.integer) or tick.ndim != 1:
        raise VectorRuntimeError("state['tick'] must be an integer array with shape [B]")
    batch_size = tick.shape[0]

    step_ms = np.asarray(step_input.step_ms)
    if not np.issubdtype(step_ms.dtype, np.number) or step_ms.shape != (batch_size,):
        raise VectorRuntimeError("step_ms must be a numeric array with shape [B]")
    if not bool(np.isfinite(step_ms.astype(np.float64, copy=False)).all()):
        raise VectorRuntimeError("step_ms values must be finite")

    source_moves = np.asarray(step_input.source_moves)
    if not np.issubdtype(source_moves.dtype, np.integer) or source_moves.shape != (
        batch_size,
        step_input.player_count,
    ):
        raise VectorRuntimeError("source_moves must be an integer array with shape [B,P]")

    if step_input.print_manager_mode is not None:
        print_manager_mode = np.asarray(step_input.print_manager_mode, dtype=object)
        if print_manager_mode.shape != (batch_size,):
            raise VectorRuntimeError("print_manager_mode must have shape [B]")
        if not bool(np.isin(print_manager_mode, tuple(PRINT_MANAGER_MODES)).all()):
            raise VectorRuntimeError("print_manager_mode values must be known modes")

    if step_input.timer_advance_ms is not None:
        _row_float_input(step_input.timer_advance_ms, batch_size, "timer_advance_ms")
    if step_input.death_immunity_mask is not None:
        mask = np.asarray(step_input.death_immunity_mask)
        if mask.shape != (batch_size, step_input.player_count):
            raise VectorRuntimeError("death_immunity_mask must have shape [B,P]")
        if mask.dtype != np.bool_:
            raise VectorRuntimeError("death_immunity_mask must be a bool array")
    if step_input.disabled_player_mask is not None:
        mask = np.asarray(step_input.disabled_player_mask)
        if mask.shape != (batch_size, step_input.player_count):
            raise VectorRuntimeError("disabled_player_mask must have shape [B,P]")
        if mask.dtype != np.bool_:
            raise VectorRuntimeError("disabled_player_mask must be a bool array")


def _death_immunity_mask(
    value: Any | None,
    *,
    row_count: int,
    player_count: int,
) -> np.ndarray:
    if value is None:
        return np.zeros((row_count, player_count), dtype=bool)
    mask = np.asarray(value, dtype=bool)
    if mask.shape != (row_count, player_count):
        raise VectorRuntimeError("death_immunity_mask must have shape [B,P]")
    return mask


def _disabled_player_mask(
    value: Any | None,
    *,
    row_count: int,
    player_count: int,
) -> np.ndarray:
    if value is None:
        return np.zeros((row_count, player_count), dtype=bool)
    mask = np.asarray(value, dtype=bool)
    if mask.shape != (row_count, player_count):
        raise VectorRuntimeError("disabled_player_mask must have shape [B,P]")
    return mask


def _row_float_input(value: Any, row_count: int, field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise VectorRuntimeError(f"{field} cannot be converted to float64") from exc
    if array.ndim == 0:
        array = np.full(row_count, float(array), dtype=np.float64)
    if array.shape != (row_count,):
        raise VectorRuntimeError(f"{field} must be a numeric array with shape [B]")
    if not bool(np.isfinite(array).all()):
        raise VectorRuntimeError(f"{field} values must be finite")
    if bool((array < 0.0).any()):
        raise VectorRuntimeError(f"{field} values must be non-negative")
    return array


def _row_unit_interval_input(value: Any, row_count: int, field: str) -> np.ndarray:
    array = _row_float_input(value, row_count, field)
    if bool((array >= 1.0).any()):
        raise VectorRuntimeError(f"{field} values must be in [0, 1)")
    return array


def _validate_no_bonus_warmup_player_count(player_count: int) -> int:
    if not isinstance(player_count, int) or isinstance(player_count, bool):
        raise VectorRuntimeError("player_count must be 2, 3, or 4 for no-bonus warmup")
    if player_count not in SUPPORTED_NO_BONUS_WARMUP_PLAYER_COUNTS:
        raise VectorRuntimeError("player_count must be 2, 3, or 4 for no-bonus warmup")
    return player_count


def _normalize_event_mode(event_mode: str) -> str:
    if event_mode not in EVENT_MODES:
        choices = ", ".join(sorted(EVENT_MODES))
        raise VectorRuntimeError(f"event_mode must be one of {choices}; got {event_mode!r}")
    return event_mode


def _events_enabled(event_mode: str) -> bool:
    return _normalize_event_mode(event_mode) == EVENT_MODE_DEBUG


def _reset_event_arrays(state: Mapping[str, np.ndarray]) -> None:
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


def _rows_to_mask(rows: np.ndarray, size: int) -> np.ndarray:
    mask = np.zeros(size, dtype=bool)
    mask[rows] = True
    return mask


def _clear_state_timer_slot(
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


def _warmup_timer_arrays(
    state: Mapping[str, np.ndarray],
    *,
    row_count: int,
) -> dict[str, np.ndarray]:
    active = _bool_array_shape(state, "timer_active", ndim=2, row_count=row_count)
    timer_shape = active.shape
    remaining_ms = _numeric_array_shape(
        state,
        "timer_remaining_ms",
        shape=timer_shape,
    )
    if not bool(np.isfinite(remaining_ms).all()):
        raise VectorRuntimeError("timer_remaining_ms values must be finite")
    return {
        "active": active,
        "remaining_ms": remaining_ms,
        "kind": _integer_array_shape(state, "timer_kind", shape=timer_shape),
        "player": _integer_array_shape(state, "timer_player", shape=timer_shape),
        "seq": _integer_array_shape(state, "timer_seq", shape=timer_shape),
        "overflow": _bool_array_shape(
            state,
            "timer_overflow",
            shape=(row_count,),
        ),
    }


def _warmup_lifecycle_arrays(
    state: Mapping[str, np.ndarray],
    *,
    row_count: int,
) -> dict[str, np.ndarray]:
    return {
        "done": _bool_array_shape(state, "done", shape=(row_count,)),
        "overflow": _bool_array_shape(state, "overflow", shape=(row_count,)),
        "started": _bool_array_shape(state, "started", shape=(row_count,)),
        "in_round": _bool_array_shape(state, "in_round", shape=(row_count,)),
        "world_active": _bool_array_shape(
            state,
            "world_active",
            shape=(row_count,),
        ),
        "world_body_count": _integer_array_shape(
            state,
            "world_body_count",
            shape=(row_count,),
        ),
    }


def _warmup_player_arrays(
    state: Mapping[str, np.ndarray],
    *,
    row_count: int,
    player_count: int | None,
) -> dict[str, np.ndarray]:
    alive = _bool_array_shape(state, "alive", ndim=2, row_count=row_count)
    inferred_player_count = int(alive.shape[1])
    if player_count is None:
        player_count = inferred_player_count
    player_count = _validate_no_bonus_warmup_player_count(player_count)
    if inferred_player_count != player_count:
        raise VectorRuntimeError("alive must have shape [B,P] for player_count")
    player_shape = (row_count, player_count)
    return {
        "alive": alive,
        "present": _bool_array_shape(state, "present", shape=player_shape),
        "printing": _bool_array_shape(state, "printing", shape=player_shape),
        "print_manager_active": _bool_array_shape(
            state,
            "print_manager_active",
            shape=player_shape,
        ),
        "print_manager_distance": _numeric_array_shape(
            state,
            "print_manager_distance",
            shape=player_shape,
        ),
        "pos": _numeric_array_shape(state, "pos", shape=(*player_shape, 2)),
        "print_manager_last_pos": _numeric_array_shape(
            state,
            "print_manager_last_pos",
            shape=(*player_shape, 2),
        ),
    }


def _body_point_arrays(
    state: Mapping[str, np.ndarray],
    *,
    row_count: int,
    player_count: int,
) -> dict[str, np.ndarray]:
    body_active = _bool_array_shape(state, "body_active", ndim=2, row_count=row_count)
    body_shape = body_active.shape
    arrays = {
        "active": body_active,
        "pos": _numeric_array_shape(state, "body_pos", shape=(*body_shape, 2)),
        "radius": _numeric_array_shape(state, "body_radius", shape=body_shape),
        "owner": _integer_array_shape(state, "body_owner", shape=body_shape),
        "num": _integer_array_shape(state, "body_num", shape=body_shape),
        "insert_tick": _integer_array_shape(
            state,
            "body_insert_tick",
            shape=body_shape,
        ),
        "insert_kind": _integer_array_shape(
            state,
            "body_insert_kind",
            shape=body_shape,
        ),
        "write_cursor": _integer_array_shape(
            state,
            "body_write_cursor",
            shape=(row_count,),
        ),
        "count": _integer_array_shape(
            state,
            "body_count",
            shape=(row_count, player_count),
        ),
        "overflow": _bool_array_shape(state, "body_overflow", shape=(row_count,)),
        "radius_by_player": _numeric_array_shape(
            state,
            "radius",
            shape=(row_count, player_count),
        ),
    }
    if "body_break_before" in state:
        arrays["break_before"] = _bool_array_shape(
            state,
            "body_break_before",
            shape=body_shape,
        )
    return arrays


def _optional_visible_trail_arrays(
    state: Mapping[str, np.ndarray],
    *,
    row_count: int,
    player_count: int,
) -> dict[str, np.ndarray | None]:
    names = (
        "visible_trail_count",
        "has_visible_trail_last",
        "visible_trail_last_pos",
        "has_draw_cursor",
        "draw_cursor_pos",
    )
    if not any(name in state for name in names):
        return {name: None for name in names}
    missing = [name for name in names if name not in state]
    if missing:
        raise VectorRuntimeError("visible trail arrays must be supplied together")

    player_shape = (row_count, player_count)
    return {
        "visible_trail_count": _integer_array_shape(
            state,
            "visible_trail_count",
            shape=player_shape,
        ),
        "has_visible_trail_last": _bool_array_shape(
            state,
            "has_visible_trail_last",
            shape=player_shape,
        ),
        "visible_trail_last_pos": _numeric_array_shape(
            state,
            "visible_trail_last_pos",
            shape=(*player_shape, 2),
        ),
        "has_draw_cursor": _bool_array_shape(
            state,
            "has_draw_cursor",
            shape=player_shape,
        ),
        "draw_cursor_pos": _numeric_array_shape(
            state,
            "draw_cursor_pos",
            shape=(*player_shape, 2),
        ),
    }


def _optional_visual_trail_arrays(
    state: Mapping[str, np.ndarray],
    *,
    row_count: int,
    player_count: int,
) -> dict[str, np.ndarray] | None:
    names = (
        "visual_trail_active",
        "visual_trail_pos",
        "visual_trail_radius",
        "visual_trail_owner",
        "visual_trail_break_before",
        "visual_trail_write_cursor",
        "visual_trail_overflow",
        "has_visual_trail_last",
        "visual_trail_last_pos",
    )
    if not any(name in state for name in names):
        return None
    missing = [name for name in names if name not in state]
    if missing:
        raise VectorRuntimeError("visual trail arrays must be supplied together")

    active = _bool_array_shape(state, "visual_trail_active", ndim=2, row_count=row_count)
    trail_shape = active.shape
    player_shape = (row_count, player_count)
    write_cursor = _integer_array_shape(
        state,
        "visual_trail_write_cursor",
        shape=(row_count,),
    )
    if bool(((write_cursor < 0) | (write_cursor > trail_shape[1])).any()):
        raise VectorRuntimeError(
            "visual_trail_write_cursor values must be in [0, visual trail capacity]"
        )
    return {
        "active": active,
        "pos": _numeric_array_shape(
            state,
            "visual_trail_pos",
            shape=(*trail_shape, 2),
        ),
        "radius": _numeric_array_shape(
            state,
            "visual_trail_radius",
            shape=trail_shape,
        ),
        "owner": _integer_array_shape(
            state,
            "visual_trail_owner",
            shape=trail_shape,
        ),
        "break_before": _bool_array_shape(
            state,
            "visual_trail_break_before",
            shape=trail_shape,
        ),
        "write_cursor": write_cursor,
        "overflow": _bool_array_shape(
            state,
            "visual_trail_overflow",
            shape=(row_count,),
        ),
        "has_last": _bool_array_shape(
            state,
            "has_visual_trail_last",
            shape=player_shape,
        ),
        "last_pos": _numeric_array_shape(
            state,
            "visual_trail_last_pos",
            shape=(*player_shape, 2),
        ),
    }


def _fire_game_start_timer(
    timer_arrays: Mapping[str, np.ndarray],
    lifecycle_arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    player_count: int,
    player_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    lifecycle_arrays["world_active"][row] = True
    players = list(range(player_count - 1, -1, -1))
    free_slots = [
        int(slot)
        for slot in range(timer_arrays["active"].shape[1])
        if not bool(timer_arrays["active"][row, slot])
    ]
    if len(players) > len(free_slots):
        timer_arrays["overflow"][row] = True
        lifecycle_arrays["overflow"][row] = True
        return {"overflowed": True, "slots_and_players": []}

    scheduled: list[tuple[int, int]] = []
    for seq, (slot, player) in enumerate(zip(free_slots, players)):
        timer_arrays["active"][row, slot] = True
        timer_arrays["remaining_ms"][row, slot] = SOURCE_TRAIL_START_DELAY_MS
        timer_arrays["kind"][row, slot] = TIMER_KIND_PRINT_MANAGER_START
        timer_arrays["player"][row, slot] = player
        timer_arrays["seq"][row, slot] = seq
        scheduled.append((slot, player))
    return {"overflowed": False, "slots_and_players": scheduled}


def _fire_warmdown_end_timer(
    state: Mapping[str, np.ndarray],
    timer_arrays: Mapping[str, np.ndarray],
    lifecycle_arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    row_count: int,
    player_count: int,
    next_round_warmup_ms: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _clear_row_for_next_round(
        state,
        lifecycle_arrays,
        row=row,
        row_count=row_count,
        player_count=player_count,
    )
    row_mask = np.zeros(row_count, dtype=bool)
    row_mask[row] = True
    spawn_info = vector_spawn.spawn_round_rows(
        state,
        row_mask,
        player_count=player_count,
    )
    schedule_info = _schedule_next_round_game_start_timer(
        timer_arrays,
        lifecycle_arrays,
        row=row,
        delay_ms=next_round_warmup_ms,
    )
    return spawn_info, schedule_info


def _clear_row_for_next_round(
    state: Mapping[str, np.ndarray],
    lifecycle_arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    row_count: int,
    player_count: int,
) -> None:
    lifecycle_arrays["started"][row] = True
    lifecycle_arrays["in_round"][row] = True
    lifecycle_arrays["world_active"][row] = False
    lifecycle_arrays["world_body_count"][row] = 0

    _set_optional_bool_row(state, "done", row=row, row_count=row_count, value=False)
    _set_optional_bool_row(
        state,
        "terminated",
        row=row,
        row_count=row_count,
        value=False,
    )
    _set_optional_bool_row(
        state,
        "reset_pending",
        row=row,
        row_count=row_count,
        value=False,
    )
    _set_optional_int_row(
        state,
        "terminal_reason",
        row=row,
        row_count=row_count,
        value=TERMINAL_REASON_NONE,
    )
    _set_optional_bool_row(state, "draw", row=row, row_count=row_count, value=False)
    _set_optional_int_row(state, "winner", row=row, row_count=row_count, value=-1)
    _set_optional_bool_row(
        state,
        "round_done",
        row=row,
        row_count=row_count,
        value=False,
    )
    _set_optional_bool_row(
        state,
        "warmdown_pending",
        row=row,
        row_count=row_count,
        value=False,
    )
    _set_optional_int_row(
        state,
        "round_winner",
        row=row,
        row_count=row_count,
        value=-1,
    )
    _set_optional_int_row(
        state,
        "match_winner",
        row=row,
        row_count=row_count,
        value=-1,
    )
    if "round_id" in state:
        round_id = _state_array(state, "round_id")
        if round_id.shape != (row_count,) or not np.issubdtype(round_id.dtype, np.integer):
            raise VectorRuntimeError("round_id must be an integer array with shape [B]")
        round_id[row] += 1

    _clear_player_round_arrays(state, row=row, row_count=row_count, player_count=player_count)
    _clear_body_round_arrays(state, row=row, row_count=row_count, player_count=player_count)


def _clear_player_round_arrays(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    row_count: int,
    player_count: int,
) -> None:
    player_shape = (row_count, player_count)
    for name in ("alive", "printing", "print_manager_active"):
        if name in state:
            _bool_array_shape(state, name, shape=player_shape)[row, :player_count] = False
    if "death_tick" in state:
        _integer_array_shape(state, "death_tick", shape=player_shape)[
            row,
            :player_count,
        ] = -1
    if "round_score" in state:
        _integer_array_shape(state, "round_score", shape=player_shape)[
            row,
            :player_count,
        ] = 0
    if "death_count" in state:
        _integer_array_shape(state, "death_count", shape=(row_count,))[row] = 0
    if "death_player" in state:
        death_player = _state_array(state, "death_player")
        if death_player.ndim != 2 or death_player.shape[0] != row_count:
            raise VectorRuntimeError("death_player must be an integer array with leading shape [B]")
        if not np.issubdtype(death_player.dtype, np.integer):
            raise VectorRuntimeError("death_player must be an integer array")
        death_player[row, :] = -1
        death_capacity = int(death_player.shape[1])
    else:
        death_capacity = player_count
    if "death_cause" in state:
        cause = _optional_death_detail_array(
            state,
            "death_cause",
            row_count=row_count,
            capacity=death_capacity,
        )
        if cause is not None:
            cause[row, :] = DEATH_CAUSE_NONE
    if "death_hit_owner" in state:
        hit_owner = _optional_death_detail_array(
            state,
            "death_hit_owner",
            row_count=row_count,
            capacity=death_capacity,
        )
        if hit_owner is not None:
            hit_owner[row, :] = -1
    for name in ("print_manager_distance",):
        if name in state:
            _numeric_array_shape(state, name, shape=player_shape)[row, :player_count] = 0.0
    for name in (
        "print_manager_last_pos",
        "visible_trail_last_pos",
        "draw_cursor_pos",
        "visual_trail_last_pos",
    ):
        if name in state:
            _numeric_array_shape(state, name, shape=(*player_shape, 2))[row, :player_count] = 0.0
    for name in ("visible_trail_count", "body_count", "live_body_num"):
        if name in state:
            _integer_array_shape(state, name, shape=player_shape)[row, :player_count] = 0
    for name in ("has_visible_trail_last", "has_draw_cursor", "has_visual_trail_last"):
        if name in state:
            _bool_array_shape(state, name, shape=player_shape)[row, :player_count] = False


def _clear_body_round_arrays(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    row_count: int,
    player_count: int,
) -> None:
    if "body_active" not in state:
        return
    body_arrays = _body_point_arrays(
        state,
        row_count=row_count,
        player_count=player_count,
    )
    body_arrays["active"][row, :] = False
    body_arrays["pos"][row, :, :] = 0.0
    body_arrays["radius"][row, :] = 0.0
    body_arrays["owner"][row, :] = -1
    body_arrays["num"][row, :] = -1
    body_arrays["insert_tick"][row, :] = -1
    body_arrays["insert_kind"][row, :] = -1
    if "break_before" in body_arrays:
        body_arrays["break_before"][row, :] = False
    body_arrays["write_cursor"][row] = 0
    body_arrays["overflow"][row] = False
    _clear_visual_trail_points_row(state, row=row, row_count=row_count)


def _schedule_next_round_game_start_timer(
    timer_arrays: Mapping[str, np.ndarray],
    lifecycle_arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    delay_ms: float,
) -> dict[str, Any]:
    timer_arrays["active"][row, :] = False
    timer_arrays["remaining_ms"][row, :] = 0.0
    timer_arrays["kind"][row, :] = TIMER_KIND_NONE
    timer_arrays["player"][row, :] = TIMER_PLAYER_NONE
    timer_arrays["seq"][row, :] = 0
    timer_arrays["overflow"][row] = False

    if int(timer_arrays["active"].shape[1]) < 1:
        timer_arrays["overflow"][row] = True
        lifecycle_arrays["overflow"][row] = True
        return {"scheduled": False, "overflowed": True, "slot": -1}

    timer_arrays["active"][row, 0] = True
    timer_arrays["remaining_ms"][row, 0] = delay_ms
    timer_arrays["kind"][row, 0] = TIMER_KIND_GAME_START
    timer_arrays["player"][row, 0] = TIMER_PLAYER_NONE
    timer_arrays["seq"][row, 0] = 0
    return {"scheduled": True, "overflowed": False, "slot": 0}


def _fire_print_manager_start_timer(
    state: Mapping[str, np.ndarray],
    timer_arrays: Mapping[str, np.ndarray],
    lifecycle_arrays: Mapping[str, np.ndarray],
    player_arrays: Mapping[str, np.ndarray],
    body_arrays: Mapping[str, np.ndarray],
    visible_arrays: Mapping[str, np.ndarray | None],
    *,
    row: int,
    player: int,
) -> tuple[int, int, int]:
    player_count = int(player_arrays["alive"].shape[1])
    if player < 0 or player >= player_count:
        raise VectorRuntimeError(f"timer references unknown player index {player}")
    if bool(player_arrays["print_manager_active"][row, player]):
        return 0, 0, 0

    player_arrays["print_manager_active"][row, player] = True
    player_arrays["print_manager_last_pos"][row, player] = player_arrays["pos"][
        row,
        player,
    ]

    points = 0
    overflowed = 0
    if not bool(player_arrays["printing"][row, player]):
        player_arrays["printing"][row, player] = True
        if bool(lifecycle_arrays["world_active"][row]):
            points, overflowed = _append_important_body_point(
                state,
                lifecycle_arrays,
                body_arrays,
                visible_arrays,
                row=row,
                player=player,
            )

    row_mask = np.zeros(timer_arrays["active"].shape[0], dtype=bool)
    row_mask[row] = True
    assign_print_manager_random_distances(state, player=player, mask=row_mask)
    return 1, points, overflowed


def _append_important_body_point(
    state: Mapping[str, np.ndarray],
    lifecycle_arrays: Mapping[str, np.ndarray],
    body_arrays: Mapping[str, np.ndarray],
    visible_arrays: Mapping[str, np.ndarray | None],
    *,
    row: int,
    player: int,
) -> tuple[int, int]:
    cursor = int(body_arrays["write_cursor"][row])
    capacity = int(body_arrays["active"].shape[1])
    if cursor >= capacity:
        body_arrays["overflow"][row] = True
        lifecycle_arrays["overflow"][row] = True
        return 0, 1

    pos = _state_array(state, "pos")
    tick = _state_array(state, "tick")
    body_arrays["active"][row, cursor] = True
    body_arrays["pos"][row, cursor] = pos[row, player]
    body_arrays["radius"][row, cursor] = body_arrays["radius_by_player"][row, player]
    body_arrays["owner"][row, cursor] = player
    body_arrays["num"][row, cursor] = body_arrays["count"][row, player]
    body_arrays["insert_tick"][row, cursor] = tick[row]
    body_arrays["insert_kind"][row, cursor] = BODY_KIND_IMPORTANT
    if "break_before" in body_arrays:
        has_draw_cursor = visible_arrays.get("has_draw_cursor")
        body_arrays["break_before"][row, cursor] = (
            False if has_draw_cursor is None else not bool(has_draw_cursor[row, player])
        )
    body_arrays["write_cursor"][row] += 1
    lifecycle_arrays["world_body_count"][row] += 1
    body_arrays["count"][row, player] += 1

    if visible_arrays["visible_trail_count"] is not None:
        visible_arrays["visible_trail_count"][row, player] += 1
        visible_arrays["has_visible_trail_last"][row, player] = True
        visible_arrays["visible_trail_last_pos"][row, player] = pos[row, player]
        visible_arrays["has_draw_cursor"][row, player] = True
        visible_arrays["draw_cursor_pos"][row, player] = pos[row, player]
    row_mask = np.zeros(_state_array(state, "tick").shape[0], dtype=bool)
    row_mask[row] = True
    _append_visual_trail_points_batched(
        state,
        player=player,
        write_mask=row_mask,
    )
    return 1, 0


def _clear_timer_slot(
    timer_arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    slot: int,
) -> None:
    timer_arrays["active"][row, slot] = False
    timer_arrays["remaining_ms"][row, slot] = 0.0
    timer_arrays["kind"][row, slot] = TIMER_KIND_NONE
    timer_arrays["player"][row, slot] = TIMER_PLAYER_NONE
    timer_arrays["seq"][row, slot] = 0


def _bool_array_shape(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
    row_count: int | None = None,
) -> np.ndarray:
    array = _state_array(state, name)
    if array.dtype != np.bool_:
        raise VectorRuntimeError(f"{name} must be a bool array")
    _validate_shape(array, name, shape=shape, ndim=ndim, row_count=row_count)
    return array


def _numeric_array_shape(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = _state_array(state, name)
    if array.shape != shape or not np.issubdtype(array.dtype, np.number):
        raise VectorRuntimeError(f"{name} must be a numeric array with shape [B]")
    return array


def _integer_array_shape(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = _state_array(state, name)
    if array.shape != shape or not np.issubdtype(array.dtype, np.integer):
        raise VectorRuntimeError(f"{name} must be an integer array with shape [B]")
    return array


def _validate_shape(
    array: np.ndarray,
    name: str,
    *,
    shape: tuple[int, ...] | None,
    ndim: int | None,
    row_count: int | None,
) -> None:
    if shape is not None and array.shape != shape:
        raise VectorRuntimeError(f"{name} must have shape {_shape_phrase(shape)}")
    if ndim is not None and array.ndim != ndim:
        raise VectorRuntimeError(f"{name} must have {ndim} dimensions")
    if row_count is not None and (array.ndim < 1 or array.shape[0] != row_count):
        raise VectorRuntimeError(f"{name} must have leading shape [B]")


def _shape_phrase(shape: tuple[int, ...]) -> str:
    return "[" + ",".join(str(part) for part in shape) + "]"


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise VectorRuntimeError(f"{field} must be a positive integer")
    if value < 1:
        raise VectorRuntimeError(f"{field} must be a positive integer")
    return value


def _array(value: Any, *, dtype: Any, field: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise VectorRuntimeError(f"{field} cannot be converted to {np.dtype(dtype)}") from exc


def _bool_mask(value: Any, row_count: int, field: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.dtype != np.bool_ or mask.shape != (row_count,):
        raise VectorRuntimeError(f"{field} must be a bool array with shape [B]")
    return mask


def _state_array(state: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in state:
        raise VectorRuntimeError(f"state is missing required step array {name!r}")
    return np.asarray(state[name])


def _row_counter_sum(state: Mapping[str, np.ndarray], name: str) -> int:
    tick = _state_array(state, "tick")
    if tick.ndim != 1:
        raise VectorRuntimeError("state['tick'] must be an array with shape [B]")
    array = _state_array(state, name)
    if array.shape != tick.shape or not np.issubdtype(array.dtype, np.integer):
        raise VectorRuntimeError(f"{name} must be an integer array with shape [B]")
    return int(array.sum())


def _player_numeric_array(
    state: Mapping[str, np.ndarray],
    name: str,
    row_count: int,
    player_count: int,
) -> np.ndarray:
    array = _state_array(state, name)
    if array.shape != (row_count, player_count) or not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise VectorRuntimeError(f"{name} must be a numeric array with shape [B,P]")
    return array


def _player_integral_array(
    state: Mapping[str, np.ndarray],
    name: str,
    row_count: int,
    player_count: int,
) -> np.ndarray:
    array = _state_array(state, name)
    if array.shape != (row_count, player_count) or not np.issubdtype(
        array.dtype,
        np.integer,
    ):
        raise VectorRuntimeError(f"{name} must be an integer array with shape [B,P]")
    return array


def _player_points_array(
    state: Mapping[str, np.ndarray],
    name: str,
    row_count: int,
    player_count: int,
) -> np.ndarray:
    array = _state_array(state, name)
    if array.shape != (row_count, player_count, 2) or not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise VectorRuntimeError(f"{name} must be a numeric array with shape [B,P,2]")
    return array


def _random_tape_arrays(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    tick = _state_array(state, "tick")
    if tick.ndim != 1:
        raise VectorRuntimeError("state['tick'] must be an array with shape [B]")
    row_count = tick.shape[0]

    values = _state_array(state, "random_tape_values")
    length = _state_array(state, "random_tape_length")
    cursor = _state_array(state, "random_tape_cursor")
    draw_count = _state_array(state, "random_tape_draw_count")
    exhausted = _state_array(state, "random_tape_exhausted")
    if values.ndim != 2 or values.shape[0] != row_count:
        raise VectorRuntimeError("random_tape_values must have shape [B,N]")
    if not np.issubdtype(values.dtype, np.floating):
        raise VectorRuntimeError("random_tape_values must be a floating array")
    for name, array in (
        ("random_tape_length", length),
        ("random_tape_cursor", cursor),
        ("random_tape_draw_count", draw_count),
    ):
        if array.shape != (row_count,) or not np.issubdtype(array.dtype, np.integer):
            raise VectorRuntimeError(f"{name} must be an integer array with shape [B]")
    if exhausted.shape != (row_count,) or exhausted.dtype != np.bool_:
        raise VectorRuntimeError("random_tape_exhausted must be a bool array with shape [B]")
    return {
        "values": values,
        "length": length,
        "cursor": cursor,
        "draw_count": draw_count,
        "exhausted": exhausted,
    }


def _default_print_manager_random_distance(*, printing: bool) -> float:
    if printing:
        return PRINT_MANAGER_RANDOM_HALF_PRINT_DISTANCE
    return PRINT_MANAGER_RANDOM_HALF_HOLE_DISTANCE
