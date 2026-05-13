"""Metadata-only public vector environment for 2P/3P/4P rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from curvyzero.env import vector_lifecycle
from curvyzero.env import vector_reset
from curvyzero.env import vector_runtime
from curvyzero.env import vector_source_random
from curvyzero.env.config import CurvyTronReferenceDefaults
from curvyzero.env.trainer_contract import ACTION_ID_TO_SOURCE_MOVE
from curvyzero.env.trainer_contract import ACTION_NAMES
from curvyzero.env.trainer_contract import ACTION_SPACE_HASH
from curvyzero.env.trainer_contract import ACTION_SPACE_ID
from curvyzero.env.trainer_contract import (
    NATIVE_CONTROL_MODEL_ID as CONTRACT_NATIVE_CONTROL_MODEL_ID,
)
from curvyzero.env.trainer_contract import REWARD_SCHEMA_HASH
from curvyzero.env.trainer_contract import REWARD_SCHEMA_ID
from curvyzero.env.trainer_contract import (
    TRAINER_CONTROL_WRAPPER_ID as CONTRACT_TRAINER_CONTROL_WRAPPER_ID,
)
from curvyzero.env.trainer_contract import stable_contract_hash


PUBLIC_ENV_CONTRACT_ID = "curvyzero_public_multiplayer_env/v0"
# Bonus paths are feature/ruleset modes of the same public env, not new env versions.
PUBLIC_SEEDED_BONUS_ENV_CONTRACT_ID = PUBLIC_ENV_CONTRACT_ID
PUBLIC_NATURAL_BONUS_ENV_CONTRACT_ID = PUBLIC_ENV_CONTRACT_ID
ENV_IMPL_ID = "curvyzero_vector_multiplayer_env/v0"
SEEDED_BONUS_ENV_IMPL_ID = ENV_IMPL_ID
NATURAL_BONUS_ENV_IMPL_ID = ENV_IMPL_ID
ENV_VERSION_POLICY_ID = "single_vector_multiplayer_env_impl_feature_modes/v0"
RULESET_ID = "curvytron_no_bonus/v0"
SEEDED_BONUS_RULESET_ID = "curvytron_seeded_bonus_subset/v0"
NATURAL_BONUS_RULESET_ID = "curvytron_natural_bonus_source_default_spawn_subset/v0"
SEEDED_BONUS_SUPPORT_POLICY_ID = "curvyzero_public_seeded_bonus_fixture_support/v0"
SEEDED_BONUS_SUPPORT_CLAIM = "seeded public runtime-backed bonus effects; no natural bonus spawn"
NATURAL_BONUS_SUPPORT_POLICY_ID = "curvyzero_public_natural_bonus_source_default_spawn_support/v0"
NATURAL_BONUS_SUPPORT_CLAIM = (
    "opt-in natural source-default timer/random/type/position spawn accounting; "
    "catch/effects remain limited to runtime-supported bonus effect types"
)
NATURAL_BONUS_RANDOM_LABEL_POLICY_ID = "source_named_bonus_spawn_random_tape_labels/v0"
SOURCE_FRAME_DECISION_POLICY_ID = "source_frame_substeps_under_trainer_decision/v0"
SOURCE_PHYSICS_STEP_POLICY_ID = "curvytron_source_basegame_framerate_60hz/v0"
SOURCE_PHYSICS_STEP_MS = CurvyTronReferenceDefaults().tick_ms
DEFAULT_DECISION_SOURCE_FRAMES = 12
DEFAULT_SOURCE_FRAME_DECISION_MS = (
    SOURCE_PHYSICS_STEP_MS * DEFAULT_DECISION_SOURCE_FRAMES
)
LIFECYCLE_POLICY_ID = "curvyzero_public_explicit_reset_warmdown_bridge/v0"
RESET_EPISODE_ID_POLICY = "vector_reset_episode_id_increments_on_explicit_reset_only/v0"
SOURCE_ROUND_ID_POLICY = "one_based_source_round_increments_on_next_round_spawn/v0"
NATIVE_CONTROL_MODEL_ID = CONTRACT_NATIVE_CONTROL_MODEL_ID
TRAINER_CONTROL_WRAPPER_ID = CONTRACT_TRAINER_CONTROL_WRAPPER_ID
FINAL_OBSERVATION_POLICY = "terminal_public_observation_before_autoreset/v0"
DEATH_ORDER_POLICY = "curvytron_source_game_deaths_order/v0"
DEBUG_METADATA_OBSERVATION_SCHEMA_ID = "curvyzero_debug_metadata_only/v0"
JOINT_ACTION_SCHEMA_ID = "curvyzero_external_joint_action_player_major/v0"
PUBLIC_JOINT_ACTION_SIDECAR_SCHEMA_ID = "curvyzero_public_joint_action_sidecar/v0"
PUBLIC_RESET_POLICY_SCHEMA_ID = "curvyzero_public_multiplayer_masked_reset_policy/v0"
RESET_PROVENANCE_POLICY_ID = "reset_seed_source_random_tape_cursor_ref/v0"
PUBLIC_ACTIVE_ROUND_LEAVE_POLICY_ID = "curvyzero_public_active_round_leave_metadata_only/v0"
PUBLIC_WARMDOWN_LEAVE_POLICY_ID = "curvyzero_public_round_warmdown_leave_metadata_only/v0"
EPISODE_END_MODE_ROUND = "round"
EPISODE_END_MODE_MATCH = "match"
EPISODE_END_MODES = (EPISODE_END_MODE_ROUND, EPISODE_END_MODE_MATCH)
SUPPORTED_PLAYER_COUNTS = (2, 3, 4)
RANDOM_TAPE_SOURCE_UNINITIALIZED = "uninitialized"
RANDOM_TAPE_SOURCE_SEED_GENERATED = vector_source_random.RANDOM_TAPE_SOURCE_SEED_GENERATED
RANDOM_TAPE_SOURCE_SOURCE_FIXTURE = "source_fixture_random_tape_values"
RANDOM_TAPE_SOURCE_DIRECT_STATE = "direct_vector_runtime_state"
RNG_IMPL_ID_UNINITIALIZED = "uninitialized"
RNG_IMPL_ID_SEED_GENERATED = vector_source_random.SOURCE_RANDOM_HISTORY_IMPL_ID
RNG_IMPL_ID_SOURCE_FIXTURE = "source_fixture_random_tape_values/v0"
RNG_IMPL_ID_DIRECT_STATE = "external_vector_runtime_state/v0"
PUBLIC_LIFECYCLE_ARRAY_NAMES = (
    "round_done",
    "warmdown_pending",
    "match_done",
    "round_winner",
    "match_winner",
)

DEBUG_METADATA_OBSERVATION_FIELDS = (
    "present",
    "alive",
    "score",
    "round_score",
    "death_order_index",
    "done",
)
DEBUG_METADATA_OBSERVATION_SCHEMA = {
    "schema_id": DEBUG_METADATA_OBSERVATION_SCHEMA_ID,
    "dtype": "float32",
    "shape": ("batch", "player", len(DEBUG_METADATA_OBSERVATION_FIELDS)),
    "fields": DEBUG_METADATA_OBSERVATION_FIELDS,
    "purpose": "debug metadata only; not a learned trainer observation",
}
DEBUG_METADATA_OBSERVATION_SCHEMA_HASH = stable_contract_hash(DEBUG_METADATA_OBSERVATION_SCHEMA)
SOURCE_DEFAULT_NATURAL_BONUS_TYPE_CODES = {
    vector_runtime.BONUS_TYPE_NAME_BY_CODE[int(code)]: int(code)
    for code in vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES
}
SOURCE_DEFAULT_NATURAL_BONUS_TYPE_NAMES = tuple(SOURCE_DEFAULT_NATURAL_BONUS_TYPE_CODES)
SOURCE_REFERENCE_DEFAULT_BONUS_TYPE_NAMES = CurvyTronReferenceDefaults().default_bonus_types


def _public_runtime_bonus_effect_type_codes() -> dict[str, int]:
    codes: dict[str, int] = {}
    for code, effect in vector_runtime.BONUS_RUNTIME_EFFECT_BY_TYPE.items():
        codes[vector_runtime.BONUS_TYPE_NAME_BY_CODE[int(code)]] = int(code)
    return codes


PUBLIC_RUNTIME_BONUS_EFFECT_TYPE_CODES = _public_runtime_bonus_effect_type_codes()
PUBLIC_RUNTIME_BONUS_EFFECT_TYPE_NAMES = tuple(PUBLIC_RUNTIME_BONUS_EFFECT_TYPE_CODES)
SEEDED_BONUS_TYPE_CODES = {
    name: code
    for name, code in PUBLIC_RUNTIME_BONUS_EFFECT_TYPE_CODES.items()
    if code in vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES
}
SEEDED_BONUS_TYPE_NAMES = tuple(SEEDED_BONUS_TYPE_CODES)
NATURAL_BONUS_TYPE_CODES = SOURCE_DEFAULT_NATURAL_BONUS_TYPE_CODES
NATURAL_BONUS_TYPE_NAMES = SOURCE_DEFAULT_NATURAL_BONUS_TYPE_NAMES
NATURAL_BONUS_EFFECT_TYPE_CODES = {
    name: code
    for name, code in NATURAL_BONUS_TYPE_CODES.items()
    if code in PUBLIC_RUNTIME_BONUS_EFFECT_TYPE_CODES.values()
}
NATURAL_BONUS_EFFECT_TYPE_NAMES = tuple(NATURAL_BONUS_EFFECT_TYPE_CODES)
NATURAL_BONUS_UNSUPPORTED_TYPE_NAMES = tuple(
    name
    for name in SOURCE_REFERENCE_DEFAULT_BONUS_TYPE_NAMES
    if name not in NATURAL_BONUS_TYPE_CODES
)
NATURAL_BONUS_UNSUPPORTED_EFFECT_TYPE_NAMES = tuple(
    name for name in NATURAL_BONUS_TYPE_NAMES if name not in NATURAL_BONUS_EFFECT_TYPE_CODES
)


def _source_default_gap_claims(natural_bonus_enabled: bool) -> tuple[str, ...]:
    gaps: list[str] = []
    if not natural_bonus_enabled:
        gaps.append("source_default_bonus_spawn_not_enabled_for_no_bonus_ruleset/v0")
    if NATURAL_BONUS_UNSUPPORTED_EFFECT_TYPE_NAMES:
        gaps.append("source_default_bonus_effects_not_yet_runtime_supported/v0")
    gaps.append("public_observation_is_debug_metadata_only/v0")
    return tuple(gaps)


RULES_HASH = stable_contract_hash(
    {
        "ruleset_id": RULESET_ID,
        "bonuses": False,
        "player_counts": SUPPORTED_PLAYER_COUNTS,
        "runtime": "curvyzero.env.vector_runtime.step_many",
    }
)
SEEDED_BONUS_RULES_HASH = stable_contract_hash(
    {
        "ruleset_id": SEEDED_BONUS_RULESET_ID,
        "bonuses": "seeded_fixture_subset",
        "natural_bonus_spawn": False,
        "supported_seeded_bonus_types": SEEDED_BONUS_TYPE_NAMES,
        "player_counts": SUPPORTED_PLAYER_COUNTS,
        "runtime": "curvyzero.env.vector_runtime.step_many",
    }
)
NATURAL_BONUS_RULES_HASH = stable_contract_hash(
    {
        "ruleset_id": NATURAL_BONUS_RULESET_ID,
        "bonuses": "natural_spawn_subset",
        "natural_bonus_spawn": "source_default_type_selection",
        "supported_natural_bonus_types": NATURAL_BONUS_TYPE_NAMES,
        "supported_natural_bonus_effect_types": NATURAL_BONUS_EFFECT_TYPE_NAMES,
        "unsupported_natural_bonus_types": NATURAL_BONUS_UNSUPPORTED_TYPE_NAMES,
        "unsupported_natural_bonus_effects": (NATURAL_BONUS_UNSUPPORTED_EFFECT_TYPE_NAMES),
        "player_counts": SUPPORTED_PLAYER_COUNTS,
        "spawn_runtime": "curvyzero.env.vector_runtime.bonus_spawn_due_rows",
        "runtime": "curvyzero.env.vector_runtime.step_many",
    }
)

ACTION_COUNT = len(ACTION_NAMES)
DEFAULT_BODY_CAPACITY = 4096
DEFAULT_EVENT_CAPACITY = 64
DEFAULT_RANDOM_TAPE_CAPACITY = 4096
DEFAULT_SEEDED_BONUS_CAPACITY = 1
DEFAULT_SEEDED_BONUS_STACK_CAPACITY = vector_runtime.SOURCE_MAX_ACTIVE_BONUSES
DEFAULT_NATURAL_BONUS_CAPACITY = vector_runtime.SOURCE_MAX_ACTIVE_BONUSES
DEFAULT_NATURAL_BONUS_STACK_CAPACITY = vector_runtime.SOURCE_MAX_ACTIVE_BONUSES
# Source placement retries until it finds a free location. The vector runtime
# needs a finite draw slab; long no-death profiles can make 16 attempts too low.
DEFAULT_NATURAL_BONUS_POSITION_ATTEMPT_CAPACITY = 256
SOURCE_BONUS_POPING_TIME_MS = 3_000.0

_TERMINAL_REASON_NAMES = {
    vector_reset.TERMINAL_REASON_NONE: "none",
    vector_reset.TERMINAL_REASON_SURVIVOR_WIN: "round_survivor_win",
    vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW: "round_all_dead_draw",
    vector_reset.TERMINAL_REASON_TIMEOUT_TRUNCATED: "timeout",
    vector_reset.TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED: "event_overflow_truncated",
    vector_reset.TERMINAL_REASON_BODY_OVERFLOW_TRUNCATED: "capacity_truncated",
}


class VectorMultiplayerEnvError(ValueError):
    """Raised when the public multiplayer metadata env is used invalidly."""


@dataclass(frozen=True, slots=True)
class VectorMultiplayerBatch:
    """Public batch returned by reset/step for B multiplayer rows."""

    observation: np.ndarray
    action_mask: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    final_observation: np.ndarray | None
    final_reward: np.ndarray | None
    info: dict[str, Any]


class VectorMultiplayerEnv:
    """Small public 2P/3P/4P env with debug metadata observations.

    This is the fast public CurvyTron runtime under hardening. Default resets
    can still run without bonuses, and tests can seed supported bonus rows
    through ``seed_active_bonus``.
    """

    def __init__(
        self,
        batch_size: int = 1,
        *,
        player_count: int = 2,
        seed: int | None = None,
        decision_ms: float | None = None,
        decision_source_frames: int | None = None,
        source_physics_step_ms: float | None = None,
        max_ticks: int = 2_000,
        map_size: float | None = None,
        max_score: int | None = None,
        episode_end_mode: str = EPISODE_END_MODE_ROUND,
        body_capacity: int = DEFAULT_BODY_CAPACITY,
        event_capacity: int = DEFAULT_EVENT_CAPACITY,
        timer_capacity: int | None = None,
        random_tape_capacity: int = DEFAULT_RANDOM_TAPE_CAPACITY,
        event_mode: str = vector_runtime.EVENT_MODE_NONE,
        player_ids: tuple[str, ...] | None = None,
        max_warmup_timer_callbacks: int | None = None,
        natural_bonus_spawn: bool = False,
        natural_bonus_type_codes: tuple[str | int, ...] | np.ndarray | None = None,
        natural_bonus_capacity: int = DEFAULT_NATURAL_BONUS_CAPACITY,
        natural_bonus_position_attempt_capacity: int = (
            DEFAULT_NATURAL_BONUS_POSITION_ATTEMPT_CAPACITY
        ),
        death_mode: str = vector_runtime.DEATH_MODE_NORMAL,
        death_immunity_player_ids: tuple[int, ...] | np.ndarray | None = None,
    ) -> None:
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.player_count = _player_count(player_count)
        if episode_end_mode not in EPISODE_END_MODES:
            raise VectorMultiplayerEnvError("episode_end_mode must be 'round' or 'match'")
        self.episode_end_mode = episode_end_mode
        self.source_physics_step_ms = _positive_finite(
            SOURCE_PHYSICS_STEP_MS
            if source_physics_step_ms is None
            else source_physics_step_ms,
            "source_physics_step_ms",
        )
        self.decision_source_frames = _optional_positive_int(
            decision_source_frames,
            "decision_source_frames",
        )
        if self.decision_source_frames is None:
            self.decision_ms = _positive_finite(
                300.0 if decision_ms is None else decision_ms,
                "decision_ms",
            )
            self.source_frame_decision = False
        else:
            derived_decision_ms = self.decision_source_frames * self.source_physics_step_ms
            if decision_ms is not None and not np.isclose(
                float(decision_ms),
                derived_decision_ms,
                rtol=0.0,
                atol=1e-6,
            ):
                raise VectorMultiplayerEnvError(
                    "decision_ms must equal decision_source_frames * "
                    "source_physics_step_ms when decision_source_frames is set"
                )
            self.decision_ms = float(derived_decision_ms)
            self.source_frame_decision = True
        self.max_ticks = _nonnegative_int(max_ticks, "max_ticks")
        self.body_capacity = _positive_int(body_capacity, "body_capacity")
        self.event_capacity = _positive_int(event_capacity, "event_capacity")
        self.timer_capacity = _positive_int(
            max(4, self.player_count) if timer_capacity is None else timer_capacity,
            "timer_capacity",
        )
        self.random_tape_capacity = _positive_int(
            random_tape_capacity,
            "random_tape_capacity",
        )
        if event_mode not in vector_runtime.EVENT_MODES:
            raise VectorMultiplayerEnvError("event_mode must be 'debug-event' or 'no-event'")
        self.event_mode = event_mode
        self.player_ids = _player_ids(player_ids, player_count=self.player_count)
        self.max_warmup_timer_callbacks = _optional_positive_int(
            max_warmup_timer_callbacks,
            "max_warmup_timer_callbacks",
        )
        self.natural_bonus_spawn_default = _bool_flag(
            natural_bonus_spawn,
            "natural_bonus_spawn",
        )
        self.natural_bonus_type_codes = _natural_bonus_type_codes(
            natural_bonus_type_codes,
        )
        self.natural_bonus_capacity = _positive_int(
            natural_bonus_capacity,
            "natural_bonus_capacity",
        )
        self.natural_bonus_position_attempt_capacity = _positive_int(
            natural_bonus_position_attempt_capacity,
            "natural_bonus_position_attempt_capacity",
        )
        if death_mode not in vector_runtime.DEATH_MODES:
            raise VectorMultiplayerEnvError("death_mode must be 'normal' or 'profile_no_death'")
        self.death_mode = death_mode
        self.death_immunity_player_ids = _death_immunity_player_ids(
            death_immunity_player_ids,
            player_count=self.player_count,
        )

        reference = CurvyTronReferenceDefaults()
        self.map_size = (
            float(reference.arena_size_for_players(self.player_count))
            if map_size is None
            else _positive_finite(map_size, "map_size")
        )
        self.max_score = (
            reference.max_score_for_players(self.player_count)
            if max_score is None
            else _positive_int(max_score, "max_score")
        )
        self.speed = float(reference.avatar_velocity_units_per_s)
        self.angular_velocity_per_ms = float(reference.angular_velocity_radians_per_ms)
        self.radius = float(reference.avatar_radius)
        self.trail_latency = int(reference.trail_latency_points)
        self.first_warmup_ms = float(reference.round_warmup_ms)

        self.reset_template = _make_state_arrays(
            self.batch_size,
            player_count=self.player_count,
            body_capacity=self.body_capacity,
            event_capacity=self.event_capacity,
            timer_capacity=self.timer_capacity,
            random_tape_capacity=self.random_tape_capacity,
            map_size=self.map_size,
            max_score=self.max_score,
            speed=self.speed,
            angular_velocity_per_ms=self.angular_velocity_per_ms,
            radius=self.radius,
            trail_latency=self.trail_latency,
        )
        self.state = {name: array.copy() for name, array in self.reset_template.items()}
        self._seed_rng = np.random.default_rng(seed)
        self._has_reset = False
        self._needs_reset = np.zeros(self.batch_size, dtype=bool)
        self._random_tape_source = np.full(
            self.batch_size,
            RANDOM_TAPE_SOURCE_UNINITIALIZED,
            dtype=object,
        )
        self._rng_impl_id = np.full(
            self.batch_size,
            RNG_IMPL_ID_UNINITIALIZED,
            dtype=object,
        )
        self._source_fixture_ref = np.full(self.batch_size, None, dtype=object)
        self._natural_multiplayer_reset_claim = np.zeros(self.batch_size, dtype=bool)
        self._seeded_bonus_enabled = np.zeros(self.batch_size, dtype=bool)
        self._natural_bonus_spawn_enabled = np.full(
            self.batch_size,
            self.natural_bonus_spawn_default,
            dtype=bool,
        )
        self._natural_bonus_timer_active = np.zeros(self.batch_size, dtype=bool)
        self._natural_bonus_timer_remaining_ms = np.zeros(
            self.batch_size,
            dtype=np.float64,
        )
        self._natural_bonus_next_due_elapsed_ms = np.full(
            self.batch_size,
            np.nan,
            dtype=np.float64,
        )
        self._natural_bonus_pop_count = np.zeros(self.batch_size, dtype=np.int32)
        self.last_reset_info: dict[str, Any] | None = None
        self.last_step_info: dict[str, Any] | None = None

    def seed_active_bonus(
        self,
        *,
        row: int,
        bonus_type: str | int,
        x: float,
        y: float,
        radius: float = 3.0,
        bonus_id: int = 1,
        slot: int = 0,
        bonus_capacity: int = DEFAULT_SEEDED_BONUS_CAPACITY,
        stack_capacity: int = DEFAULT_SEEDED_BONUS_STACK_CAPACITY,
    ) -> dict[str, Any]:
        """Seed one active bonus for source fixture checks.

        This does not enable natural bonus spawning. It only installs the arrays
        that ``vector_runtime.step_many`` already consumes.
        """

        self._require_reset()
        row_int = self._row_index(row)
        bonus_code = _seeded_bonus_type_code(bonus_type)
        x_value = _finite_float(x, "x")
        y_value = _finite_float(y, "y")
        radius_value = _positive_finite(radius, "radius")
        bonus_id_value = _positive_int(bonus_id, "bonus_id")
        slot_value = _nonnegative_int(slot, "slot")
        capacity = max(_positive_int(bonus_capacity, "bonus_capacity"), slot_value + 1)
        stack_capacity_value = _positive_int(stack_capacity, "stack_capacity")

        if bool(self.state["done"][row_int]) or bool(self._needs_reset[row_int]):
            raise VectorMultiplayerEnvError("cannot seed a bonus into a terminal row")

        self._ensure_seeded_bonus_arrays(
            bonus_capacity=capacity,
            stack_capacity=stack_capacity_value,
        )
        row_mask = np.zeros(self.batch_size, dtype=bool)
        row_mask[row_int] = True
        self._clear_seeded_bonus_rows(row_mask)

        self.state["bonus_world_active"][row_int] = True
        self.state["bonus_active"][row_int, slot_value] = True
        self.state["bonus_type"][row_int, slot_value] = bonus_code
        self.state["bonus_id"][row_int, slot_value] = bonus_id_value
        self.state["bonus_pos"][row_int, slot_value] = np.asarray(
            [x_value, y_value],
            dtype=np.float64,
        )
        self.state["bonus_radius"][row_int, slot_value] = radius_value
        self.state["bonus_count"][row_int] = int(self.state["bonus_active"][row_int].sum())
        self.state["bonus_world_body_count"][row_int] = int(self.state["bonus_count"][row_int])
        self.state["base_radius"][row_int] = self.state["radius"][row_int]
        self.state["base_speed"][row_int] = self.state["speed"][row_int]
        self.state["base_inverse"][row_int] = self.state["inverse"][row_int]
        self.state["base_invincible"][row_int] = self.state["invincible"][row_int]
        self.state["base_avatar_color"][row_int] = self.state["avatar_color"][row_int]
        self.state["base_angular_velocity_per_ms"][row_int] = self.state["angular_velocity_per_ms"][
            row_int
        ]
        self.state["radius_power"][row_int] = 0
        self._seeded_bonus_enabled[row_int] = True
        return {
            "policy_id": SEEDED_BONUS_SUPPORT_POLICY_ID,
            "claim": SEEDED_BONUS_SUPPORT_CLAIM,
            "row": row_int,
            "slot": slot_value,
            "bonus_type": _seeded_bonus_type_name(bonus_code),
            "bonus_type_code": bonus_code,
            "bonus_id": bonus_id_value,
            "natural_bonus_spawn": False,
        }

    def reset(
        self,
        seed: int | np.ndarray | None = None,
        *,
        row_mask: np.ndarray | None = None,
        present: np.ndarray | None = None,
        source_fixture_random_tape_values: np.ndarray | None = None,
        source_fixture_ref: str | None = None,
        source_fixture_new_round_time_ms: float | None = None,
        source_fixture_warmup_advance_ms: float | np.ndarray | None = None,
    ) -> VectorMultiplayerBatch:
        """Reset all rows, or selected rows, through no-bonus spawn/warmup."""

        mask = self._row_mask(row_mask)
        return self._reset_selected_rows(
            seed=seed,
            mask=mask,
            scalar_seed_resets_rng=row_mask is None,
            present=present,
            source_fixture_random_tape_values=source_fixture_random_tape_values,
            source_fixture_ref=source_fixture_ref,
            source_fixture_new_round_time_ms=source_fixture_new_round_time_ms,
            source_fixture_warmup_advance_ms=source_fixture_warmup_advance_ms,
            reset_source=vector_reset.RESET_SOURCE_MANUAL,
            reset_api="reset",
        )

    def autoreset_done_rows(
        self,
        seed: int | np.ndarray | None = None,
        *,
        row_mask: np.ndarray | None = None,
        present: np.ndarray | None = None,
        source_fixture_random_tape_values: np.ndarray | None = None,
        source_fixture_ref: str | None = None,
        source_fixture_new_round_time_ms: float | None = None,
        source_fixture_warmup_advance_ms: float | np.ndarray | None = None,
    ) -> VectorMultiplayerBatch:
        """Explicitly reset rows that have already returned terminal public metadata."""

        self._require_reset()
        if row_mask is None:
            mask = self._needs_reset.copy()
        else:
            mask = self._row_mask(row_mask)
            invalid_rows = mask & ~self._needs_reset
            if bool(invalid_rows.any()):
                rows = np.flatnonzero(invalid_rows).astype(np.int32)
                raise VectorMultiplayerEnvError(
                    "autoreset_done_rows row_mask can only select rows with needs_reset; "
                    f"nonterminal rows={rows.tolist()}"
                )
        return self._reset_selected_rows(
            seed=seed,
            mask=mask,
            scalar_seed_resets_rng=row_mask is None,
            present=present,
            source_fixture_random_tape_values=source_fixture_random_tape_values,
            source_fixture_ref=source_fixture_ref,
            source_fixture_new_round_time_ms=source_fixture_new_round_time_ms,
            source_fixture_warmup_advance_ms=source_fixture_warmup_advance_ms,
            reset_source=vector_reset.RESET_SOURCE_AUTORESET,
            reset_api="autoreset_done_rows",
        )

    def _reset_selected_rows(
        self,
        *,
        seed: int | np.ndarray | None,
        mask: np.ndarray,
        scalar_seed_resets_rng: bool,
        present: np.ndarray | None,
        source_fixture_random_tape_values: np.ndarray | None,
        source_fixture_ref: str | None,
        source_fixture_new_round_time_ms: float | None,
        source_fixture_warmup_advance_ms: float | np.ndarray | None,
        reset_source: int,
        reset_api: str,
    ) -> VectorMultiplayerBatch:
        present_array = self._present_array(present, mask)
        fixture_random_tape_values = _source_fixture_random_tape_values(
            source_fixture_random_tape_values,
            batch_size=self.batch_size,
            random_tape_capacity=self.random_tape_capacity,
        )
        fixture_ref = _source_fixture_ref(
            source_fixture_ref,
            has_fixture_random_tape=fixture_random_tape_values is not None,
        )
        fixture_new_round_time_ms = (
            self.first_warmup_ms
            if source_fixture_new_round_time_ms is None
            else _nonnegative_finite(
                source_fixture_new_round_time_ms,
                "source_fixture_new_round_time_ms",
            )
        )
        fixture_warmup_advance_ms = _source_fixture_warmup_advance_ms(
            source_fixture_warmup_advance_ms,
            batch_size=self.batch_size,
        )
        seed_input = seed
        if scalar_seed_resets_rng and seed is not None and np.asarray(seed).ndim == 0:
            self._seed_rng = np.random.default_rng(
                _nonnegative_seed_scalar(np.asarray(seed).item(), "seed"),
            )
            seed_input = None
        reset_seed = self._reset_seed_array(seed_input, mask)

        self._prepare_reset_template_rows(
            mask,
            reset_seed,
            present=present_array,
            source_fixture_random_tape_values=fixture_random_tape_values,
        )
        reset_info = self._reset_spawn_warmup_rows(
            mask,
            reset_seed=reset_seed,
            reset_source=self._reset_source_array(
                mask,
                reset_source,
            ),
            first_warmup_ms=fixture_new_round_time_ms,
        )
        self.state["round_id"][mask] = 1
        if fixture_random_tape_values is None:
            random_tape_source = RANDOM_TAPE_SOURCE_SEED_GENERATED
            rng_impl_id = RNG_IMPL_ID_SEED_GENERATED
            natural_multiplayer_reset_claim = True
        else:
            random_tape_source = RANDOM_TAPE_SOURCE_SOURCE_FIXTURE
            rng_impl_id = RNG_IMPL_ID_SOURCE_FIXTURE
            natural_multiplayer_reset_claim = False
        self._random_tape_source[mask] = random_tape_source
        self._rng_impl_id[mask] = rng_impl_id
        self._source_fixture_ref[mask] = fixture_ref
        self._natural_multiplayer_reset_claim[mask] = natural_multiplayer_reset_claim
        reset_info = {
            **reset_info,
            "lifecycle_policy_id": LIFECYCLE_POLICY_ID,
            "reset_episode_id": self.state["episode_id"].copy(),
            "reset_episode_id_policy": RESET_EPISODE_ID_POLICY,
            "reset_round_id": self.state["round_id"].copy(),
            "source_round_id_policy": SOURCE_ROUND_ID_POLICY,
            "random_tape_source": random_tape_source,
            "random_tape_source_by_row": _selected_row_labels(
                mask,
                random_tape_source,
                batch_size=self.batch_size,
            ),
            "random_tape_length": self.state["random_tape_length"].copy(),
            "rng_impl_id": rng_impl_id,
            "rng_impl_id_by_row": _selected_row_labels(
                mask,
                rng_impl_id,
                batch_size=self.batch_size,
            ),
            "random_tape_history_ref": rng_impl_id,
            "random_tape_history_ref_by_row": _selected_row_labels(
                mask,
                rng_impl_id,
                batch_size=self.batch_size,
            ),
            "source_fixture_ref": fixture_ref,
            "source_fixture_ref_by_row": _selected_row_labels(
                mask,
                fixture_ref,
                batch_size=self.batch_size,
            ),
            "random_tape_seeded_by_reset_seed": fixture_random_tape_values is None,
            "natural_multiplayer_reset_claim": natural_multiplayer_reset_claim,
            "natural_multiplayer_reset_claim_by_row": _selected_row_labels(
                mask,
                natural_multiplayer_reset_claim,
                batch_size=self.batch_size,
            ),
            "natural_multiplayer_reset_claim_scope": (
                "seeded_source_history_reset_spawn_warmup_call_order/v0"
                if natural_multiplayer_reset_claim
                else None
            ),
        }
        self._clear_seeded_bonus_rows(mask)
        natural_bonus_reset_info = self._reset_natural_bonus_spawn_rows(
            mask,
            first_warmup_ms=fixture_new_round_time_ms,
        )
        selected_warmup_advance_ms = self._selected_warmup_advance_ms(
            mask,
            first_warmup_ms=fixture_new_round_time_ms,
            advance_ms=fixture_warmup_advance_ms,
        )
        warmup_info = self._advance_warmup_rows(
            mask,
            first_warmup_ms=fixture_new_round_time_ms,
            advance_ms=selected_warmup_advance_ms,
        )
        natural_bonus_warmup_info = self._advance_natural_bonus_spawn_timers(
            mask,
            advance_ms=selected_warmup_advance_ms,
            phase="reset_warmup",
        )
        self._needs_reset[mask] = False
        self._has_reset = True
        if self.death_mode == vector_runtime.DEATH_MODE_PROFILE_NO_DEATH:
            self.state["borderless"][mask] = True

        reward = np.zeros((self.batch_size, self.player_count), dtype=np.float32)
        info = {
            "reset_info": reset_info,
            "warmup_info": warmup_info,
            "natural_bonus_reset_info": natural_bonus_reset_info,
            "natural_bonus_warmup_info": natural_bonus_warmup_info,
            "reset_rows": np.flatnonzero(mask).astype(np.int32),
            "public_reset_policy": self._public_reset_policy(
                mask,
                reset_info,
                reset_api=reset_api,
                reset_source=reset_source,
            ),
            **self._public_info(),
        }
        batch = self._batch(
            reward=reward,
            done=self.state["done"].copy(),
            terminated=self.state["terminated"].copy(),
            truncated=self.state["truncated"].copy(),
            final_observation=None,
            final_reward=None,
            info=info,
        )
        self.last_reset_info = batch.info
        return batch

    def reset_from_state_arrays(
        self,
        state: dict[str, np.ndarray],
        *,
        reset_seed: int | np.ndarray | None = None,
        reset_source: int = vector_reset.RESET_SOURCE_FIXTURE,
    ) -> VectorMultiplayerBatch:
        """Reset from an existing vector-runtime fixture state without warmdown.

        This is the narrow fixture-oriented reset path for metadata validation.
        It does not claim natural multiplayer reset/warmup fidelity.
        """

        if not isinstance(state, dict) or not state:
            raise VectorMultiplayerEnvError("state must be a non-empty dict of arrays")
        for name, template_array in self.reset_template.items():
            np.copyto(self.state[name], template_array)

        copied_arrays: list[str] = []
        for name, value in state.items():
            if name not in self.state:
                raise VectorMultiplayerEnvError(f"state contains unknown array {name!r}")
            target = self.state[name]
            array = np.asarray(value)
            if array.shape != target.shape:
                raise VectorMultiplayerEnvError(
                    f"state array {name!r} must have shape {target.shape}"
                )
            try:
                np.copyto(target, array, casting="safe")
            except TypeError as exc:
                raise VectorMultiplayerEnvError(
                    f"state array {name!r} cannot be safely copied to {target.dtype}"
                ) from exc
            copied_arrays.append(name)

        self.state["episode_id"][:] = 1
        self.state["round_id"][:] = 1
        self.state["episode_step"][:] = 0
        self.state["env_active"][:] = True
        self.state["reset_pending"][:] = False
        self.state["terminated"][:] = False
        self.state["truncated"][:] = False
        self.state["terminal_reason"][:] = vector_reset.TERMINAL_REASON_NONE
        self.state["present"][:, : self.player_count] = True
        self.state["death_count"][:] = 0
        self.state["death_player"][:, :] = -1
        self.state["death_cause"][:, :] = vector_runtime.DEATH_CAUSE_NONE
        self.state["death_hit_owner"][:, :] = -1
        self.state["winner"][:] = -1
        self.state["draw"][:] = False
        self.state["reset_source"][:] = np.asarray(reset_source, dtype=np.int16)
        self.state["reset_seed"][:] = self._reset_seed_array(
            reset_seed,
            np.ones(self.batch_size, dtype=bool),
        )
        self._clear_public_lifecycle_rows(np.ones(self.batch_size, dtype=bool))
        self._random_tape_source[:] = RANDOM_TAPE_SOURCE_DIRECT_STATE
        self._rng_impl_id[:] = RNG_IMPL_ID_DIRECT_STATE
        self._source_fixture_ref[:] = None
        self._natural_multiplayer_reset_claim[:] = False
        self._natural_bonus_spawn_enabled[:] = False
        self._clear_seeded_bonus_rows(np.ones(self.batch_size, dtype=bool))
        self._clear_natural_bonus_spawn_rows(np.ones(self.batch_size, dtype=bool))
        self._needs_reset[:] = False
        self._has_reset = True

        reward = np.zeros((self.batch_size, self.player_count), dtype=np.float32)
        info = {
            "reset_info": {
                "schema": "curvyzero_vector_fixture_state_reset/v0",
                "source": "direct_vector_runtime_state/v0",
                "copied_arrays": tuple(sorted(copied_arrays)),
                "warmdown_waited": False,
                "lifecycle_policy_id": LIFECYCLE_POLICY_ID,
                "reset_episode_id": self.state["episode_id"].copy(),
                "reset_episode_id_policy": RESET_EPISODE_ID_POLICY,
                "reset_round_id": self.state["round_id"].copy(),
                "source_round_id_policy": SOURCE_ROUND_ID_POLICY,
                "random_tape_source": RANDOM_TAPE_SOURCE_DIRECT_STATE,
                "random_tape_source_by_row": np.full(
                    self.batch_size,
                    RANDOM_TAPE_SOURCE_DIRECT_STATE,
                    dtype=object,
                ),
                "random_tape_length": self.state["random_tape_length"].copy(),
                "rng_impl_id": RNG_IMPL_ID_DIRECT_STATE,
                "rng_impl_id_by_row": np.full(
                    self.batch_size,
                    RNG_IMPL_ID_DIRECT_STATE,
                    dtype=object,
                ),
                "random_tape_history_ref": RNG_IMPL_ID_DIRECT_STATE,
                "random_tape_history_ref_by_row": np.full(
                    self.batch_size,
                    RNG_IMPL_ID_DIRECT_STATE,
                    dtype=object,
                ),
                "source_fixture_ref": None,
                "source_fixture_ref_by_row": np.full(self.batch_size, None, dtype=object),
                "random_tape_seeded_by_reset_seed": False,
                "natural_multiplayer_reset_claim": False,
                "natural_multiplayer_reset_claim_by_row": np.zeros(
                    self.batch_size,
                    dtype=bool,
                ),
                "natural_multiplayer_reset_claim_scope": None,
            },
            **self._public_info(),
        }
        batch = self._batch(
            reward=reward,
            done=self.state["done"].copy(),
            terminated=self.state["terminated"].copy(),
            truncated=self.state["truncated"].copy(),
            final_observation=None,
            final_reward=None,
            info=info,
        )
        self.last_reset_info = batch.info
        return batch

    def remove_player(
        self,
        player_ids: int | np.ndarray,
        *,
        row_mask: np.ndarray | None = None,
    ) -> VectorMultiplayerBatch:
        """Remove zero-based public players from active rows as metadata only.

        Source avatar ids are one-based; selected source ids are reported in
        metadata. The supported paths are active-round continuation rows, the
        source-proven immediate round-end case when a removal leaves one or zero
        live present players, and staged match-mode warmdown rows.
        """

        if not self._has_reset:
            raise RuntimeError("reset must be called before remove_player")
        mask = self._row_mask(row_mask)
        leave_player_by_row = self._leave_player_id_array(player_ids, mask)

        warmdown_rows = mask & self._warmdown_pending_mask()
        round_done = self._metadata_state_array(
            "round_done",
            np.zeros(self.batch_size, dtype=bool),
            dtype=bool,
        )
        match_done = self._metadata_state_array(
            "match_done",
            np.zeros(self.batch_size, dtype=bool),
            dtype=bool,
        )
        invalid_warmdown_rows = warmdown_rows & (~round_done | self.state["in_round"])
        if bool(invalid_warmdown_rows.any()):
            rows = np.flatnonzero(invalid_warmdown_rows).astype(np.int32)
            raise RuntimeError(
                "remove_player requires source-shaped staged warmdown rows; "
                f"warmdown rows={rows.tolist()}"
            )

        terminal_rows = mask & (
            self.state["done"]
            | self.state["terminated"]
            | self.state["truncated"]
            | self._needs_reset
            | (round_done & ~warmdown_rows)
            | match_done
        )
        if bool(terminal_rows.any()):
            rows = np.flatnonzero(terminal_rows).astype(np.int32)
            raise RuntimeError(
                f"remove_player cannot run on terminal rows; terminal rows={rows.tolist()}"
            )

        inactive_round_rows = mask & ~warmdown_rows & ~self.state["in_round"]
        if bool(inactive_round_rows.any()):
            rows = np.flatnonzero(inactive_round_rows).astype(np.int32)
            raise RuntimeError(
                f"remove_player requires active in-round rows; inactive rows={rows.tolist()}"
            )

        rows = np.flatnonzero(mask).astype(np.int32)
        selected_players = leave_player_by_row[rows].astype(np.int16, copy=False)
        present = self.state["present"][:, : self.player_count]
        alive = self.state["alive"][:, : self.player_count]

        absent_rows = np.zeros(self.batch_size, dtype=bool)
        dead_rows = np.zeros(self.batch_size, dtype=bool)
        for row, player in zip(rows, selected_players, strict=True):
            row_int = int(row)
            player_int = int(player)
            if not bool(present[row_int, player_int]):
                absent_rows[row_int] = True
            elif not bool(alive[row_int, player_int]):
                dead_rows[row_int] = True
        if bool(absent_rows.any()):
            invalid = np.flatnonzero(absent_rows).astype(np.int32)
            raise VectorMultiplayerEnvError(
                f"remove_player selected players must be present; absent rows={invalid.tolist()}"
            )
        if bool(dead_rows.any()):
            invalid = np.flatnonzero(dead_rows).astype(np.int32)
            raise VectorMultiplayerEnvError(
                f"remove_player selected players must be alive; dead rows={invalid.tolist()}"
            )

        live_present_after = present & alive
        for row, player in zip(rows, selected_players, strict=True):
            live_present_after[int(row), int(player)] = False
        active_rows = mask & ~warmdown_rows
        immediate_terminal_rows = active_rows & (live_present_after.sum(axis=1) <= 1)

        for row, player in zip(rows, selected_players, strict=True):
            row_int = int(row)
            player_int = int(player)
            self._stop_print_manager_for_leave(row_int, player_int)
            self.state["present"][row_int, player_int] = False
            self.state["alive"][row_int, player_int] = False

        terminal_rows = np.flatnonzero(immediate_terminal_rows).astype(np.int32)
        if terminal_rows.size:
            self._resolve_immediate_leave_terminal_rows(immediate_terminal_rows)
            if self.episode_end_mode == EPISODE_END_MODE_MATCH:
                self._stage_match_mode_warmdown_rows(immediate_terminal_rows)
            else:
                self._mark_public_round_warmdown_rows(immediate_terminal_rows)
            public_terminal_mask = self.state["done"].copy()
            if bool(public_terminal_mask.any()):
                self.state["in_round"][public_terminal_mask] = False
                self._needs_reset |= public_terminal_mask

        reward = self._reward()
        reward[warmdown_rows] = 0.0
        final_row_mask = self.state["done"].copy()
        final_observation = self._observe_array() if bool(final_row_mask.any()) else None
        final_reward = reward.copy() if bool(final_row_mask.any()) else None
        leave_source_player_by_row = np.full(self.batch_size, -1, dtype=np.int16)
        if rows.size:
            leave_source_player_by_row[rows] = selected_players + 1
        info = {
            "leave_rows": rows.copy(),
            "leave_row_mask": mask.copy(),
            "leave_player_ids": selected_players.copy(),
            "leave_source_player_ids": (selected_players + 1).astype(np.int16),
            "leave_player_id_by_row": leave_player_by_row.copy(),
            "leave_source_player_id_by_row": leave_source_player_by_row.copy(),
            "leave_policy_id": PUBLIC_ACTIVE_ROUND_LEAVE_POLICY_ID,
            "active_round_leave_policy_id": PUBLIC_ACTIVE_ROUND_LEAVE_POLICY_ID,
            "leave_source_id_policy": "source_avatar_id_is_public_player_id_plus_one/v0",
            "leave_metadata_only": True,
            "leave_trainer_claim": False,
            "leave_immediate_terminal_rows": terminal_rows.copy(),
            "leave_warmdown_rows": np.flatnonzero(warmdown_rows).astype(np.int32),
            "leave_warmdown_row_mask": warmdown_rows.copy(),
            "warmdown_leave_policy_id": PUBLIC_WARMDOWN_LEAVE_POLICY_ID,
            "warmdown_leave_score_policy": (
                "source_warmdown_leave_does_not_rescore_or_emit_round_end/v0"
            ),
            "terminal_rows": np.flatnonzero(final_row_mask).astype(np.int32),
            "final_observation": final_observation,
            "final_reward_map": final_reward,
            **self._public_info(),
        }
        batch = self._batch(
            reward=reward,
            done=self.state["done"].copy(),
            terminated=self.state["terminated"].copy(),
            truncated=self.state["truncated"].copy(),
            final_observation=final_observation,
            final_reward=final_reward,
            final_row_mask=final_row_mask,
            info=info,
        )
        self.last_step_info = batch.info
        return batch

    def step(
        self,
        actions: np.ndarray,
        *,
        timer_advance_ms: float | np.ndarray | None = None,
        disabled_player_mask: np.ndarray | None = None,
    ) -> VectorMultiplayerBatch:
        """Step one source-shaped decision for every row with int[B,P] actions."""

        self._require_reset()
        if bool(self._needs_reset.any()):
            rows = np.flatnonzero(self._needs_reset).astype(np.int32)
            raise RuntimeError(
                "reset must be called before stepping rows that ended; "
                f"pending rows={rows.tolist()}"
            )
        warmdown_pending = self._warmdown_pending_mask()
        if bool(warmdown_pending.any()):
            rows = np.flatnonzero(warmdown_pending).astype(np.int32)
            raise RuntimeError(
                "advance_warmdown must be called before stepping rows between rounds; "
                f"pending rows={rows.tolist()}"
            )
        pre_alive = self.state["alive"][:, : self.player_count].copy()
        pre_death_count = self.state["death_count"].copy()
        pre_active = ~self.state["done"].copy()
        self.state["bonus_catch_count_step"][:, : self.player_count] = 0
        source_moves, action_sidecar = self._source_moves_and_action_sidecar(
            actions,
            pre_alive=pre_alive,
        )
        timer_advance = _step_timer_advance_ms(
            timer_advance_ms,
            batch_size=self.batch_size,
        )
        disabled_mask = self._disabled_player_mask(disabled_player_mask)
        action_sidecar["timer_advance_ms"] = timer_advance.copy()
        action_sidecar["disabled_player_mask"] = disabled_mask.copy()
        self._ensure_seed_generated_random_tape_headroom(
            pre_active,
            min_available=max(
                16,
                self.player_count * 4 * max(1, self.decision_source_frames or 1),
            ),
        )
        runtime_result = self._advance_runtime_for_public_step(
            pre_active=pre_active,
            source_moves=source_moves,
            timer_advance=timer_advance,
            disabled_player_mask=disabled_mask,
        )
        counters = runtime_result["step_counters"]
        natural_bonus_info = runtime_result["natural_bonus_info"]
        self._correct_leave_adjusted_death_scores(
            pre_alive=pre_alive,
            pre_death_count=pre_death_count,
        )
        self._append_new_deaths(pre_alive)
        self.state["episode_step"][pre_active] += 1
        self._mark_overflow_truncations(pre_active)
        self._mark_timeout_truncations(pre_active)

        transition_mask = self.state["done"].copy()
        round_transition_mask = (
            transition_mask & self.state["terminated"] & ~self.state["truncated"]
        )
        if bool(round_transition_mask.any()):
            if self.episode_end_mode == EPISODE_END_MODE_MATCH:
                self._stage_match_mode_warmdown_rows(round_transition_mask)
            else:
                self._mark_public_round_warmdown_rows(round_transition_mask)
        public_terminal_mask = self.state["done"].copy()
        if bool(public_terminal_mask.any()):
            self.state["in_round"][public_terminal_mask] = False
            self._needs_reset |= public_terminal_mask

        done = self.state["done"].copy()
        terminated = self.state["terminated"].copy()
        truncated = self.state["truncated"].copy()
        reward = self._reward()
        final_observation = self._observe_array() if bool(public_terminal_mask.any()) else None
        final_reward = reward.copy() if bool(public_terminal_mask.any()) else None
        terminal_rows = np.flatnonzero(public_terminal_mask).astype(np.int32)
        info = {
            "step_counters": counters,
            "joint_action": np.asarray(actions).copy(),
            "source_moves": source_moves.copy(),
            "disabled_player_mask": disabled_mask.copy(),
            "action_sidecar": action_sidecar,
            "natural_bonus_info": natural_bonus_info,
            "source_frame_decision": self.source_frame_decision,
            "decision_source_frames": self.decision_source_frames,
            "source_physics_step_ms": self.source_physics_step_ms,
            "source_physics_substeps_executed": runtime_result[
                "source_physics_substeps_executed"
            ],
            "source_physics_elapsed_ms": runtime_result["source_physics_elapsed_ms"],
            "terminal_rows": terminal_rows.copy(),
            "final_observation": final_observation,
            "final_reward_map": final_reward,
            **self._public_info(),
        }
        batch = self._batch(
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            final_observation=final_observation,
            final_reward=final_reward,
            final_row_mask=public_terminal_mask,
            info=info,
        )
        self.last_step_info = batch.info
        return batch

    def _advance_runtime_for_public_step(
        self,
        *,
        pre_active: np.ndarray,
        source_moves: np.ndarray,
        timer_advance: np.ndarray,
        disabled_player_mask: np.ndarray,
    ) -> dict[str, Any]:
        if not self.source_frame_decision:
            natural_bonus_info = self._advance_natural_bonus_spawn_timers(
                pre_active,
                advance_ms=timer_advance,
                phase="step",
            )
            counters = vector_runtime.step_many(
                vector_runtime.VectorStepInput(
                    state=self._runtime_step_state(),
                    step_ms=np.full(self.batch_size, self.decision_ms, dtype=np.float64),
                    source_moves=source_moves,
                    player_count=self.player_count,
                    print_manager_mode=np.full(
                        self.batch_size,
                        "natural_toggle",
                        dtype=object,
                    ),
                    event_mode=self.event_mode,
                    timer_advance_ms=timer_advance,
                    death_mode=self.death_mode,
                    death_immunity_mask=self._death_immunity_mask(),
                    disabled_player_mask=disabled_player_mask,
                )
            )
            elapsed = np.where(pre_active, self.decision_ms, 0.0).astype(np.float64)
            self.state["elapsed_ms"] += elapsed
            return {
                "step_counters": counters,
                "natural_bonus_info": natural_bonus_info,
                "source_physics_substeps_executed": pre_active.astype(np.int32),
                "source_physics_elapsed_ms": elapsed,
            }

        assert self.decision_source_frames is not None
        total_counters = vector_runtime.empty_step_counters()
        natural_bonus_infos: list[dict[str, Any]] = []
        substeps_executed = np.zeros(self.batch_size, dtype=np.int32)
        elapsed = np.zeros(self.batch_size, dtype=np.float64)
        remaining_timer_advance = timer_advance.astype(np.float64, copy=True)
        print_manager_mode = np.full(self.batch_size, "natural_toggle", dtype=object)

        for substep_index in range(self.decision_source_frames):
            active = pre_active & ~self.state["done"] & ~self.state["overflow"]
            if not bool(active.any()):
                break
            frame_ms = np.where(active, self.source_physics_step_ms, 0.0).astype(
                np.float64,
            )
            frame_timer_advance = np.minimum(remaining_timer_advance, frame_ms)
            frame_timer_advance = np.where(active, frame_timer_advance, 0.0).astype(
                np.float64,
            )
            natural_bonus_infos.append(
                self._advance_natural_bonus_spawn_timers(
                    active,
                    advance_ms=frame_timer_advance,
                    phase=f"step_source_frame_{substep_index}",
                )
            )
            counters = vector_runtime.step_many(
                vector_runtime.VectorStepInput(
                    state=self._runtime_step_state(),
                    step_ms=frame_ms,
                    source_moves=source_moves,
                    player_count=self.player_count,
                    print_manager_mode=print_manager_mode,
                    event_mode=self.event_mode,
                    timer_advance_ms=frame_timer_advance,
                    death_mode=self.death_mode,
                    death_immunity_mask=self._death_immunity_mask(),
                    disabled_player_mask=disabled_player_mask,
                )
            )
            for name in total_counters:
                total_counters[name] += int(counters.get(name, 0))
            substeps_executed[active] += 1
            elapsed[active] += frame_ms[active]
            self.state["elapsed_ms"] += frame_ms
            remaining_timer_advance = np.maximum(
                remaining_timer_advance - frame_timer_advance,
                0.0,
            )

        return {
            "step_counters": total_counters,
            "natural_bonus_info": self._natural_bonus_substep_info(
                natural_bonus_infos,
                timer_advance=timer_advance,
                remaining_timer_advance=remaining_timer_advance,
            ),
            "source_physics_substeps_executed": substeps_executed,
            "source_physics_elapsed_ms": elapsed,
        }

    def _natural_bonus_substep_info(
        self,
        infos: list[dict[str, Any]],
        *,
        timer_advance: np.ndarray,
        remaining_timer_advance: np.ndarray,
    ) -> dict[str, Any]:
        spawn_infos: list[dict[str, Any]] = []
        random_calls: list[dict[str, Any]] = []
        schedule_calls: list[dict[str, Any]] = []
        random_tape_draws = 0
        for info in infos:
            spawn_infos.extend(info.get("spawn_infos", ()))
            random_calls.extend(info.get("random_calls", ()))
            schedule_calls.extend(info.get("schedule_calls", ()))
            random_tape_draws += int(info.get("random_tape_draws", 0))
        return {
            "schema": "curvyzero_public_natural_bonus_spawn_source_frame_advance/v0",
            "phase": "step_source_frame_substeps",
            "enabled": bool(self._natural_bonus_spawn_enabled.any()),
            "substep_count": len(infos),
            "source_physics_step_ms": float(self.source_physics_step_ms),
            "decision_source_frames": self.decision_source_frames,
            "advance_ms": timer_advance.copy(),
            "unspent_advance_ms": remaining_timer_advance.copy(),
            "spawn_infos": spawn_infos,
            "schedule_calls": schedule_calls,
            "random_calls": random_calls,
            "random_tape_draws": random_tape_draws,
            "remaining_ms": self._natural_bonus_timer_remaining_ms.copy(),
            "next_due_elapsed_ms": self._natural_bonus_next_due_elapsed_ms.copy(),
            "pop_count": self._natural_bonus_pop_count.copy(),
            "substep_infos": infos,
            "source_bonus_poping_time_ms": SOURCE_BONUS_POPING_TIME_MS,
            "random_label_policy_id": NATURAL_BONUS_RANDOM_LABEL_POLICY_ID,
        }

    def advance_warmdown(
        self,
        advance_ms: float | np.ndarray = vector_lifecycle.SOURCE_ROUND_WARMDOWN_MS,
        *,
        next_warmup_ms: float = vector_lifecycle.SOURCE_ROUND_WARMUP_MS,
        max_timer_callbacks: int = 16,
    ) -> VectorMultiplayerBatch:
        """Advance terminal rows through source-shaped warmdown metadata.

        This is a narrow metadata-only bridge for source fixture coverage. It
        does not make the public env a trainer-ready natural lifecycle env.
        """

        self._require_reset()
        self._install_public_warmdown_adapter_rows()
        warmdown_info = vector_lifecycle.advance_warmdown_no_bonus_rows(
            self.state,
            advance_ms,
            player_count=self.player_count,
            next_warmup_ms=next_warmup_ms,
            max_timer_callbacks=max_timer_callbacks,
        )
        next_round_rows = np.asarray(warmdown_info["next_round_rows"], dtype=np.int32)
        if next_round_rows.size:
            self.state["round_id"][next_round_rows] += 1

        reward = np.zeros((self.batch_size, self.player_count), dtype=np.float32)
        final_row_mask = np.zeros(self.batch_size, dtype=bool)
        match_end_rows = np.asarray(warmdown_info["match_end_rows"], dtype=np.int32)
        final_row_mask[match_end_rows] = True
        final_observation = self._observe_array() if bool(final_row_mask.any()) else None
        final_reward = reward.copy() if bool(final_row_mask.any()) else None
        self._needs_reset[:] = self.state["done"] | self.state["reset_pending"]
        info = {
            "warmdown_info": warmdown_info,
            "warmdown_rows": np.asarray(warmdown_info["warmdown_end_rows"]).copy(),
            "next_round_rows": np.asarray(warmdown_info["next_round_rows"]).copy(),
            "terminal_rows": match_end_rows.copy(),
            "final_observation": final_observation,
            "final_reward_map": final_reward,
            **self._public_info(),
            "warmdown_waited": bool(warmdown_info["warmdown_end_fires"]),
        }
        batch = self._batch(
            reward=reward,
            done=self.state["done"].copy(),
            terminated=self.state["terminated"].copy(),
            truncated=self.state["truncated"].copy(),
            final_observation=final_observation,
            final_reward=final_reward,
            final_row_mask=final_row_mask,
            info=info,
        )
        self.last_step_info = batch.info
        return batch

    def advance_warmdown_frame(
        self,
        actions: np.ndarray,
        *,
        elapsed_ms: float | np.ndarray | None = None,
    ) -> VectorMultiplayerBatch:
        """Advance one explicit metadata-only frame for rows waiting in warmdown.

        Ordinary ``step()`` remains blocked while ``warmdown_pending`` is true.
        This bridge exists only to expose the source fact that live survivors can
        keep moving after ``round:end`` and before ``game:stop``.
        """

        self._require_reset()
        if bool(self._needs_reset.any()):
            rows = np.flatnonzero(self._needs_reset).astype(np.int32)
            raise RuntimeError(
                "advance_warmdown_frame cannot run after a terminal public episode; "
                "use episode_end_mode='match' for between-round warmdown rows or reset "
                f"pending rows={rows.tolist()}"
            )

        warmdown_pending = self._warmdown_pending_mask()
        if not bool(warmdown_pending.any()):
            raise RuntimeError(
                "advance_warmdown_frame requires warmdown_pending rows; "
                "call step() until round_done in episode_end_mode='match' first"
            )

        non_warmdown_live_rows = ~self.state["done"] & ~warmdown_pending
        if bool(non_warmdown_live_rows.any()):
            rows = np.flatnonzero(non_warmdown_live_rows).astype(np.int32)
            raise RuntimeError(
                "advance_warmdown_frame is a narrow all-active-rows-warmdown bridge; "
                f"non-warmdown live rows={rows.tolist()}"
            )

        frame_ms = _warmdown_frame_elapsed_ms(
            self.decision_ms if elapsed_ms is None else elapsed_ms,
            batch_size=self.batch_size,
        )
        self._validate_warmdown_frame_stays_before_game_stop(
            warmdown_pending,
            frame_ms,
        )
        pre_alive = self.state["alive"][:, : self.player_count].copy()
        pre_active = warmdown_pending & ~self.state["done"]
        source_moves, action_sidecar = self._source_moves_and_action_sidecar(
            actions,
            pre_alive=pre_alive,
        )
        action_sidecar["decision_ms"] = frame_ms.copy()
        counters = vector_runtime.step_many(
            vector_runtime.VectorStepInput(
                state=self.state,
                step_ms=np.where(warmdown_pending, frame_ms, 0.0).astype(np.float64),
                source_moves=source_moves,
                player_count=self.player_count,
                print_manager_mode=np.full(
                    self.batch_size,
                    "natural_toggle",
                    dtype=object,
                ),
                event_mode=self.event_mode,
                timer_advance_ms=np.where(warmdown_pending, frame_ms, 0.0).astype(
                    np.float64,
                ),
                death_immunity_mask=self._death_immunity_mask(),
            )
        )
        self._append_new_deaths(pre_alive)
        self.state["elapsed_ms"][pre_active] += frame_ms[pre_active]
        self._mark_overflow_truncations(pre_active)
        self._mark_timeout_truncations(pre_active)
        self._needs_reset[:] = self.state["done"] | self.state["reset_pending"]

        reward = np.zeros((self.batch_size, self.player_count), dtype=np.float32)
        public_terminal_mask = pre_active & self.state["done"]
        final_observation = self._observe_array() if bool(public_terminal_mask.any()) else None
        final_reward = reward.copy() if bool(public_terminal_mask.any()) else None
        info = {
            "warmdown_frame_counters": counters,
            "warmdown_frame_rows": np.flatnonzero(warmdown_pending).astype(np.int32),
            "warmdown_frame_policy": (
                "explicit_metadata_only_does_not_relax_public_step_barrier/v0"
            ),
            "warmdown_frame_elapsed_ms": frame_ms.copy(),
            "warmdown_frame_step_index_incremented": False,
            "joint_action": np.asarray(actions).copy(),
            "source_moves": source_moves.copy(),
            "action_sidecar": action_sidecar,
            "terminal_rows": np.flatnonzero(public_terminal_mask).astype(np.int32),
            "final_observation": final_observation,
            "final_reward_map": final_reward,
            **self._public_info(),
        }
        batch = self._batch(
            reward=reward,
            done=self.state["done"].copy(),
            terminated=self.state["terminated"].copy(),
            truncated=self.state["truncated"].copy(),
            final_observation=final_observation,
            final_reward=final_reward,
            final_row_mask=public_terminal_mask,
            info=info,
        )
        self.last_step_info = batch.info
        return batch

    def _validate_warmdown_frame_stays_before_game_stop(
        self,
        warmdown_pending: np.ndarray,
        frame_ms: np.ndarray,
    ) -> None:
        for row in np.flatnonzero(warmdown_pending):
            row_int = int(row)
            warmdown_slots = np.flatnonzero(
                self.state["timer_active"][row_int]
                & (self.state["timer_kind"][row_int] == vector_runtime.TIMER_KIND_WARMDOWN_END)
            )
            if warmdown_slots.size == 0:
                raise RuntimeError(
                    "advance_warmdown_frame requires a scheduled warmdown game:stop timer"
                )
            remaining_ms = float(self.state["timer_remaining_ms"][row_int, warmdown_slots].min())
            if float(frame_ms[row_int]) >= remaining_ms:
                raise RuntimeError(
                    "advance_warmdown_frame elapsed_ms must stay before game:stop; "
                    "use advance_warmdown for the warmdown timer boundary"
                )

    def advance_warmup(
        self,
        advance_ms: float | np.ndarray = vector_lifecycle.SOURCE_TRAIL_START_DELAY_MS,
        *,
        max_timer_callbacks: int | None = None,
    ) -> VectorMultiplayerBatch:
        """Advance source-shaped warmup timers for metadata-only fixture coverage."""

        self._require_reset()
        advance = _source_fixture_warmup_advance_ms(
            advance_ms,
            batch_size=self.batch_size,
        )
        assert advance is not None
        callback_cap = (
            self._warmup_timer_callback_cap(np.ones(self.batch_size, dtype=bool))
            if max_timer_callbacks is None
            else _positive_int(max_timer_callbacks, "max_timer_callbacks")
        )
        warmup_info = vector_runtime.advance_warmup_no_bonus_timers(
            self.state,
            advance,
            player_count=self.player_count,
            max_timer_callbacks=callback_cap,
        )
        self._needs_reset[:] = self.state["done"] | self.state["reset_pending"]

        reward = np.zeros((self.batch_size, self.player_count), dtype=np.float32)
        info = {
            "warmup_info": warmup_info,
            "warmup_rows": np.flatnonzero(advance > 0.0).astype(np.int32),
            **self._public_info(),
        }
        batch = self._batch(
            reward=reward,
            done=self.state["done"].copy(),
            terminated=self.state["terminated"].copy(),
            truncated=self.state["truncated"].copy(),
            final_observation=None,
            final_reward=None,
            info=info,
        )
        self.last_step_info = batch.info
        return batch

    def _reset_spawn_warmup_rows(
        self,
        mask: np.ndarray,
        *,
        reset_seed: np.ndarray,
        reset_source: np.ndarray,
        first_warmup_ms: float,
    ) -> dict[str, Any]:
        info = vector_lifecycle.reset_spawn_warmup_no_bonus_rows(
            self.state,
            self.reset_template,
            mask,
            player_count=self.player_count,
            reset_seed=reset_seed,
            reset_source=reset_source,
            first_warmup_ms=first_warmup_ms,
            snapshot_array_names=(
                "done",
                "terminated",
                "truncated",
                "terminal_reason",
                "episode_id",
                "round_id",
                "pos",
                "alive",
                "present",
                "score",
                "round_score",
                "death_count",
                "death_player",
                "death_cause",
                "death_hit_owner",
                "winner",
                "draw",
            ),
        )
        if not bool(info["can_compose"]):
            raise VectorMultiplayerEnvError("state and reset_template cannot compose")
        return info

    def _advance_warmup_rows(
        self,
        mask: np.ndarray,
        *,
        first_warmup_ms: float,
        advance_ms: np.ndarray | None = None,
    ) -> dict[str, Any]:
        selected_advance_ms = np.zeros(self.batch_size, dtype=np.float64)
        if advance_ms is None:
            selected_advance_ms[mask] = (
                first_warmup_ms + vector_lifecycle.SOURCE_TRAIL_START_DELAY_MS
            )
        else:
            selected_advance_ms[mask] = np.asarray(advance_ms, dtype=np.float64)[mask]
        return vector_runtime.advance_warmup_no_bonus_timers(
            self.state,
            selected_advance_ms,
            player_count=self.player_count,
            max_timer_callbacks=self._warmup_timer_callback_cap(mask),
        )

    def _warmup_timer_callback_cap(self, mask: np.ndarray) -> int:
        if self.max_warmup_timer_callbacks is not None:
            return self.max_warmup_timer_callbacks
        return max(16, int(mask.sum()) * (self.player_count + 1) * 4)

    def _selected_warmup_advance_ms(
        self,
        mask: np.ndarray,
        *,
        first_warmup_ms: float,
        advance_ms: np.ndarray | None,
    ) -> np.ndarray:
        selected_advance_ms = np.zeros(self.batch_size, dtype=np.float64)
        if advance_ms is None:
            selected_advance_ms[mask] = (
                first_warmup_ms + vector_lifecycle.SOURCE_TRAIL_START_DELAY_MS
            )
        else:
            selected_advance_ms[mask] = np.asarray(advance_ms, dtype=np.float64)[mask]
        return selected_advance_ms

    def _reset_natural_bonus_spawn_rows(
        self,
        mask: np.ndarray,
        *,
        first_warmup_ms: float,
    ) -> dict[str, Any]:
        row_mask = np.asarray(mask, dtype=bool)
        self._natural_bonus_spawn_enabled[row_mask] = self.natural_bonus_spawn_default
        self._clear_natural_bonus_spawn_rows(row_mask)
        enabled_rows = row_mask & self._natural_bonus_spawn_enabled
        if not bool(enabled_rows.any()):
            return self._natural_bonus_empty_info(
                phase="reset",
                candidate_rows=row_mask,
            )

        self._ensure_seeded_bonus_arrays(
            bonus_capacity=self.natural_bonus_capacity,
            stack_capacity=DEFAULT_NATURAL_BONUS_STACK_CAPACITY,
        )
        self._clear_seeded_bonus_rows(enabled_rows)
        schedule_info = self._schedule_natural_bonus_pop_rows(
            enabled_rows,
            label="bonus.start_delay",
            delay_origin_ms=first_warmup_ms,
        )
        return {
            **schedule_info,
            "phase": "reset",
            "enabled": True,
            "candidate_rows": row_mask.copy(),
            "enabled_rows": enabled_rows.copy(),
            "start_after_game_start_ms": first_warmup_ms,
            "source_bonus_poping_time_ms": SOURCE_BONUS_POPING_TIME_MS,
            "random_label_policy_id": NATURAL_BONUS_RANDOM_LABEL_POLICY_ID,
        }

    def _clear_natural_bonus_spawn_rows(self, rows: np.ndarray) -> None:
        row_mask = np.asarray(rows, dtype=bool)
        self._natural_bonus_timer_active[row_mask] = False
        self._natural_bonus_timer_remaining_ms[row_mask] = 0.0
        self._natural_bonus_next_due_elapsed_ms[row_mask] = np.nan
        self._natural_bonus_pop_count[row_mask] = 0

    def _schedule_natural_bonus_pop_rows(
        self,
        rows: np.ndarray,
        *,
        label: str,
        delay_origin_ms: float,
    ) -> dict[str, Any]:
        row_mask = np.asarray(rows, dtype=bool)
        random_calls: list[dict[str, Any]] = []
        delay_draw = np.zeros(self.batch_size, dtype=np.float64)
        delay_ms = np.zeros(self.batch_size, dtype=np.float64)
        remaining_ms = np.zeros(self.batch_size, dtype=np.float64)

        for row in np.flatnonzero(row_mask):
            row_int = int(row)
            draw_value = self._draw_natural_bonus_random(
                row_int,
                label,
                random_calls=random_calls,
            )
            row_delay_ms = SOURCE_BONUS_POPING_TIME_MS * (1.0 + draw_value)
            row_remaining_ms = delay_origin_ms + row_delay_ms
            delay_draw[row_int] = draw_value
            delay_ms[row_int] = row_delay_ms
            remaining_ms[row_int] = row_remaining_ms
            self._natural_bonus_timer_active[row_int] = True
            self._natural_bonus_timer_remaining_ms[row_int] = row_remaining_ms
            self._natural_bonus_next_due_elapsed_ms[row_int] = (
                float(self.state["elapsed_ms"][row_int]) + row_remaining_ms
            )

        return {
            "schema": "curvyzero_public_natural_bonus_spawn_schedule/v0",
            "scheduled_rows": np.flatnonzero(row_mask).astype(np.int32),
            "scheduled_row_mask": row_mask.copy(),
            "delay_label": label,
            "delay_origin_ms": float(delay_origin_ms),
            "delay_draw": delay_draw,
            "delay_ms": delay_ms,
            "remaining_ms": self._natural_bonus_timer_remaining_ms.copy(),
            "next_due_elapsed_ms": self._natural_bonus_next_due_elapsed_ms.copy(),
            "random_calls": random_calls,
            "random_tape_draws": len(random_calls),
        }

    def _advance_natural_bonus_spawn_timers(
        self,
        rows: np.ndarray,
        *,
        advance_ms: np.ndarray,
        phase: str,
    ) -> dict[str, Any]:
        candidate_rows = np.asarray(rows, dtype=bool)
        advance = np.asarray(advance_ms, dtype=np.float64)
        active_rows = (
            candidate_rows
            & self._natural_bonus_spawn_enabled
            & self._natural_bonus_timer_active
            & ~self.state["done"]
            & ~self.state["reset_pending"]
        )
        if not bool(active_rows.any()):
            return self._natural_bonus_empty_info(
                phase=phase,
                candidate_rows=candidate_rows,
            )

        due_rows = np.zeros(self.batch_size, dtype=bool)
        random_calls: list[dict[str, Any]] = []
        schedule_calls: list[dict[str, Any]] = []
        spawn_infos: list[dict[str, Any]] = []

        for row in np.flatnonzero(active_rows):
            row_int = int(row)
            budget_ms = float(advance[row_int])
            if budget_ms <= 0.0:
                continue

            remaining_ms = float(self._natural_bonus_timer_remaining_ms[row_int])
            if remaining_ms > budget_ms:
                self._natural_bonus_timer_remaining_ms[row_int] = remaining_ms - budget_ms
                self._natural_bonus_next_due_elapsed_ms[row_int] = (
                    float(self.state["elapsed_ms"][row_int])
                    + float(advance[row_int])
                    + self._natural_bonus_timer_remaining_ms[row_int]
                )
                continue

            while remaining_ms <= budget_ms:
                budget_ms -= remaining_ms
                due_rows[row_int] = True
                next_delay_draw = self._draw_natural_bonus_random(
                    row_int,
                    "bonus.next_delay_after_pop",
                    random_calls=random_calls,
                )
                next_delay_ms = SOURCE_BONUS_POPING_TIME_MS * (1.0 + next_delay_draw)
                schedule_calls.append(
                    {
                        "row": row_int,
                        "label": "bonus.next_delay_after_pop",
                        "delay_draw": next_delay_draw,
                        "delay_ms": next_delay_ms,
                    }
                )
                spawn_info = self._spawn_natural_bonus_due_row(
                    row_int,
                    random_calls=random_calls,
                )
                spawn_infos.append(spawn_info)
                self._natural_bonus_pop_count[row_int] += 1
                remaining_ms = next_delay_ms
                if remaining_ms > budget_ms:
                    break

            self._natural_bonus_timer_remaining_ms[row_int] = remaining_ms - budget_ms
            self._natural_bonus_next_due_elapsed_ms[row_int] = (
                float(self.state["elapsed_ms"][row_int])
                + float(advance[row_int])
                + self._natural_bonus_timer_remaining_ms[row_int]
            )

        return {
            "schema": "curvyzero_public_natural_bonus_spawn_advance/v0",
            "phase": phase,
            "enabled": bool(self._natural_bonus_spawn_enabled.any()),
            "candidate_rows": candidate_rows.copy(),
            "active_rows": active_rows.copy(),
            "advance_ms": advance.copy(),
            "due_rows": due_rows.copy(),
            "due_row_indices": np.flatnonzero(due_rows).astype(np.int32),
            "spawn_infos": spawn_infos,
            "schedule_calls": schedule_calls,
            "random_calls": random_calls,
            "random_tape_draws": len(random_calls),
            "remaining_ms": self._natural_bonus_timer_remaining_ms.copy(),
            "next_due_elapsed_ms": self._natural_bonus_next_due_elapsed_ms.copy(),
            "pop_count": self._natural_bonus_pop_count.copy(),
            "source_bonus_poping_time_ms": SOURCE_BONUS_POPING_TIME_MS,
            "random_label_policy_id": NATURAL_BONUS_RANDOM_LABEL_POLICY_ID,
        }

    def _spawn_natural_bonus_due_row(
        self,
        row: int,
        *,
        random_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self._ensure_seeded_bonus_arrays(
            bonus_capacity=self.natural_bonus_capacity,
            stack_capacity=DEFAULT_NATURAL_BONUS_STACK_CAPACITY,
        )
        due_rows = np.zeros(self.batch_size, dtype=bool)
        due_rows[row] = True
        if int(self.state["bonus_count"][row]) >= vector_runtime.SOURCE_MAX_ACTIVE_BONUSES:
            return vector_runtime.bonus_spawn_due_rows(
                self.state,
                player_count=self.player_count,
                due_rows=due_rows,
                enabled_type_codes=self.natural_bonus_type_codes,
                events_enabled=self.event_mode == vector_runtime.EVENT_MODE_DEBUG,
            )

        type_draws = np.zeros(self.batch_size, dtype=np.float64)
        type_call_index = len(random_calls)
        type_draws[row] = self._draw_natural_bonus_random(
            row,
            "bonus.type",
            random_calls=random_calls,
        )
        attempts: list[tuple[float, float]] = []
        last_position_error: vector_runtime.VectorRuntimeError | None = None
        attempt = 0
        while True:
            if (
                attempt >= self.natural_bonus_position_attempt_capacity
                and self._random_tape_source[row] != RANDOM_TAPE_SOURCE_SEED_GENERATED
            ):
                break
            x_label = "bonus.position.x" if attempt == 0 else f"bonus.position.retry_{attempt}.x"
            y_label = "bonus.position.y" if attempt == 0 else f"bonus.position.retry_{attempt}.y"
            attempts.append(
                (
                    self._draw_natural_bonus_random(
                        row,
                        x_label,
                        random_calls=random_calls,
                    ),
                    self._draw_natural_bonus_random(
                        row,
                        y_label,
                        random_calls=random_calls,
                    ),
                )
            )
            position_draws = np.zeros(
                (self.batch_size, len(attempts), 2),
                dtype=np.float64,
            )
            position_draws[row, :, :] = np.asarray(attempts, dtype=np.float64)
            try:
                spawn_info = vector_runtime.bonus_spawn_due_rows(
                    self.state,
                    player_count=self.player_count,
                    due_rows=due_rows,
                    type_draws=type_draws,
                    position_draws=position_draws,
                    enabled_type_codes=self.natural_bonus_type_codes,
                    events_enabled=self.event_mode == vector_runtime.EVENT_MODE_DEBUG,
                )
                selected_name = str(spawn_info["selected_type_name"][row])
                type_label = f"bonus.type.{selected_name}"
                random_calls[type_call_index]["label"] = type_label
                random_calls[type_call_index]["site"] = type_label
                return spawn_info
            except vector_runtime.VectorRuntimeError as exc:
                if "position_draws did not include an accepted candidate" not in str(exc):
                    raise
                last_position_error = exc
                attempt += 1

        raise VectorMultiplayerEnvError(
            "natural bonus position attempts were exhausted"
        ) from last_position_error

    def _draw_natural_bonus_random(
        self,
        row: int,
        label: str,
        *,
        random_calls: list[dict[str, Any]],
    ) -> float:
        length = int(self.state["random_tape_length"][row])
        cursor = int(self.state["random_tape_cursor"][row])
        draw_count = int(self.state["random_tape_draw_count"][row])
        capacity = int(self.state["random_tape_values"].shape[1])
        if length < 0 or cursor < 0 or draw_count < 0:
            raise VectorMultiplayerEnvError(
                "random tape cursor, length, and draw_count must be non-negative"
            )
        if length > capacity:
            raise VectorMultiplayerEnvError(
                "random_tape_length cannot exceed random_tape_values capacity"
            )
        if cursor > length:
            raise VectorMultiplayerEnvError("random_tape_cursor cannot exceed length")
        if cursor >= length:
            extended = self._extend_seed_generated_random_tape_row(
                row,
                min_length=cursor + 1,
            )
            if extended:
                length = int(self.state["random_tape_length"][row])
            else:
                self.state["random_tape_exhausted"][row] = True
                raise VectorMultiplayerEnvError(
                    f"row {row} Math.random tape exhausted after {cursor} calls"
                )

        value = float(self.state["random_tape_values"][row, cursor])
        if not np.isfinite(value) or value < 0.0 or value >= 1.0:
            raise VectorMultiplayerEnvError(
                "consumed random_tape_values must be finite values in [0, 1)"
            )
        self.state["random_tape_cursor"][row] = cursor + 1
        self.state["random_tape_draw_count"][row] = draw_count + 1
        random_calls.append(
            {
                "row": row,
                "label": label,
                "site": label,
                "tape_index": cursor,
                "draw_ordinal": draw_count,
                "value": value,
                "random_tape_source": self._random_tape_source[row],
                "rng_impl_id": self._rng_impl_id[row],
            }
        )
        return value

    def _ensure_seed_generated_random_tape_headroom(
        self,
        rows: np.ndarray,
        *,
        min_available: int,
    ) -> None:
        row_mask = np.asarray(rows, dtype=bool)
        available_required = _positive_int(min_available, "min_available")
        for row in np.flatnonzero(row_mask):
            row_int = int(row)
            cursor = int(self.state["random_tape_cursor"][row_int])
            length = int(self.state["random_tape_length"][row_int])
            if length - cursor < available_required:
                self._extend_seed_generated_random_tape_row(
                    row_int,
                    min_length=cursor + available_required,
                )

    def _extend_seed_generated_random_tape_row(self, row: int, *, min_length: int) -> bool:
        row_int = self._row_index(row)
        required_length = _positive_int(min_length, "min_length")
        if self._random_tape_source[row_int] != RANDOM_TAPE_SOURCE_SEED_GENERATED:
            return False
        current_capacity = int(self.state["random_tape_values"].shape[1])
        if required_length <= int(self.state["random_tape_length"][row_int]):
            return True

        next_capacity = max(current_capacity * 2, required_length)
        self._resize_random_tape_capacity(next_capacity)
        seed = np.asarray([self.state["reset_seed"][row_int]], dtype=np.uint64)
        self.state["random_tape_values"][row_int, :next_capacity] = (
            vector_source_random.seeded_source_math_random_history(
                seed,
                length=next_capacity,
            )[0]
        )
        self.state["random_tape_length"][row_int] = next_capacity
        self.state["random_tape_exhausted"][row_int] = False
        return True

    def _resize_random_tape_capacity(self, next_capacity: int) -> None:
        capacity = _positive_int(next_capacity, "next_capacity")
        current_capacity = int(self.state["random_tape_values"].shape[1])
        if capacity <= current_capacity:
            return
        for arrays in (self.state, self.reset_template):
            old_values = arrays["random_tape_values"]
            next_values = np.zeros((self.batch_size, capacity), dtype=old_values.dtype)
            next_values[:, :current_capacity] = old_values
            arrays["random_tape_values"] = next_values
        self.random_tape_capacity = capacity

    def _natural_bonus_empty_info(
        self,
        *,
        phase: str,
        candidate_rows: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "schema": "curvyzero_public_natural_bonus_spawn_advance/v0",
            "phase": phase,
            "enabled": bool(self._natural_bonus_spawn_enabled.any()),
            "candidate_rows": np.asarray(candidate_rows, dtype=bool).copy(),
            "active_rows": np.zeros(self.batch_size, dtype=bool),
            "due_rows": np.zeros(self.batch_size, dtype=bool),
            "due_row_indices": np.asarray([], dtype=np.int32),
            "spawn_infos": [],
            "schedule_calls": [],
            "random_calls": [],
            "random_tape_draws": 0,
            "remaining_ms": self._natural_bonus_timer_remaining_ms.copy(),
            "next_due_elapsed_ms": self._natural_bonus_next_due_elapsed_ms.copy(),
            "pop_count": self._natural_bonus_pop_count.copy(),
            "source_bonus_poping_time_ms": SOURCE_BONUS_POPING_TIME_MS,
            "random_label_policy_id": NATURAL_BONUS_RANDOM_LABEL_POLICY_ID,
        }

    def _runtime_step_state(self) -> dict[str, np.ndarray]:
        return {
            name: array
            for name, array in self.state.items()
            if name not in PUBLIC_LIFECYCLE_ARRAY_NAMES
        }

    def _death_immunity_mask(self) -> np.ndarray:
        mask = np.zeros((self.batch_size, self.player_count), dtype=bool)
        if self.death_immunity_player_ids.size:
            mask[:, self.death_immunity_player_ids] = True
        return mask

    def _disabled_player_mask(self, value: np.ndarray | None) -> np.ndarray:
        if value is None:
            return np.zeros((self.batch_size, self.player_count), dtype=bool)
        mask = np.asarray(value)
        if mask.shape != (self.batch_size, self.player_count):
            raise VectorMultiplayerEnvError("disabled_player_mask must have shape [B,P]")
        if mask.dtype != np.bool_:
            raise VectorMultiplayerEnvError("disabled_player_mask must be a bool array")
        return mask.copy()

    def _ensure_seeded_bonus_arrays(
        self,
        *,
        bonus_capacity: int,
        stack_capacity: int,
    ) -> None:
        current_capacity = (
            int(self.state["bonus_active"].shape[1]) if "bonus_active" in self.state else 0
        )
        current_stack_capacity = (
            int(self.state["bonus_stack_id"].shape[2]) if "bonus_stack_id" in self.state else 0
        )
        current_game_stack_capacity = (
            int(self.state["bonus_game_stack_id"].shape[1])
            if "bonus_game_stack_id" in self.state
            else 0
        )
        if (
            current_capacity >= bonus_capacity
            and current_stack_capacity >= stack_capacity
            and current_game_stack_capacity >= stack_capacity
            and "invincible" in self.state
            and "base_invincible" in self.state
            and "avatar_color" in self.state
            and "base_avatar_color" in self.state
            and "base_angular_velocity_per_ms" in self.state
            and "bonus_stack_angular_velocity_per_ms" in self.state
            and "bonus_stack_invincible_delta" in self.state
            and "bonus_stack_printing_delta" in self.state
            and "bonus_stack_color" in self.state
        ):
            self._ensure_reset_template_bonus_arrays(
                bonus_capacity=current_capacity,
                stack_capacity=max(current_stack_capacity, current_game_stack_capacity),
            )
            return

        next_bonus_capacity = max(current_capacity, bonus_capacity)
        next_stack_capacity = max(current_stack_capacity, stack_capacity)
        next_game_stack_capacity = max(current_game_stack_capacity, stack_capacity)
        old_state = self.state
        row_count = self.batch_size
        player_count = self.player_count

        bonus_active = np.zeros((row_count, next_bonus_capacity), dtype=bool)
        bonus_type = np.full(
            (row_count, next_bonus_capacity),
            vector_runtime.BONUS_TYPE_NONE,
            dtype=np.int16,
        )
        bonus_id = np.full((row_count, next_bonus_capacity), -1, dtype=np.int32)
        bonus_pos = np.zeros((row_count, next_bonus_capacity, 2), dtype=np.float64)
        bonus_radius = np.zeros((row_count, next_bonus_capacity), dtype=np.float64)
        bonus_next_id = np.ones(row_count, dtype=np.int32)
        if current_capacity:
            bonus_active[:, :current_capacity] = old_state["bonus_active"]
            bonus_type[:, :current_capacity] = old_state["bonus_type"]
            bonus_id[:, :current_capacity] = old_state["bonus_id"]
            bonus_pos[:, :current_capacity] = old_state["bonus_pos"]
            bonus_radius[:, :current_capacity] = old_state["bonus_radius"]
            if "bonus_next_id" in old_state:
                bonus_next_id[:] = old_state["bonus_next_id"]

        stack_count = np.zeros((row_count, player_count), dtype=np.int16)
        stack_id = np.full(
            (row_count, player_count, next_stack_capacity),
            -1,
            dtype=np.int32,
        )
        stack_type = np.full(
            (row_count, player_count, next_stack_capacity),
            vector_runtime.BONUS_TYPE_NONE,
            dtype=np.int16,
        )
        stack_duration_ms = np.zeros(
            (row_count, player_count, next_stack_capacity),
            dtype=np.int32,
        )
        stack_radius_power = np.zeros(
            (row_count, player_count, next_stack_capacity),
            dtype=np.int16,
        )
        stack_velocity_delta = np.zeros(
            (row_count, player_count, next_stack_capacity),
            dtype=np.float64,
        )
        stack_inverse_delta = np.zeros(
            (row_count, player_count, next_stack_capacity),
            dtype=np.int16,
        )
        stack_angular_velocity_per_ms = np.zeros(
            (row_count, player_count, next_stack_capacity),
            dtype=np.float64,
        )
        stack_invincible_delta = np.zeros(
            (row_count, player_count, next_stack_capacity),
            dtype=np.int16,
        )
        stack_printing_delta = np.zeros(
            (row_count, player_count, next_stack_capacity),
            dtype=np.int16,
        )
        stack_color = np.full(
            (row_count, player_count, next_stack_capacity),
            -1,
            dtype=np.int16,
        )
        if current_stack_capacity:
            stack_count[:, :] = old_state["bonus_stack_count"]
            stack_id[:, :, :current_stack_capacity] = old_state["bonus_stack_id"]
            stack_type[:, :, :current_stack_capacity] = old_state["bonus_stack_type"]
            stack_duration_ms[:, :, :current_stack_capacity] = old_state["bonus_stack_duration_ms"]
            stack_radius_power[:, :, :current_stack_capacity] = old_state[
                "bonus_stack_radius_power"
            ]
            if "bonus_stack_velocity_delta" in old_state:
                stack_velocity_delta[:, :, :current_stack_capacity] = old_state[
                    "bonus_stack_velocity_delta"
                ]
            if "bonus_stack_inverse_delta" in old_state:
                stack_inverse_delta[:, :, :current_stack_capacity] = old_state[
                    "bonus_stack_inverse_delta"
                ]
            if "bonus_stack_angular_velocity_per_ms" in old_state:
                stack_angular_velocity_per_ms[:, :, :current_stack_capacity] = old_state[
                    "bonus_stack_angular_velocity_per_ms"
                ]
            if "bonus_stack_invincible_delta" in old_state:
                stack_invincible_delta[:, :, :current_stack_capacity] = old_state[
                    "bonus_stack_invincible_delta"
                ]
            if "bonus_stack_printing_delta" in old_state:
                stack_printing_delta[:, :, :current_stack_capacity] = old_state[
                    "bonus_stack_printing_delta"
                ]
            if "bonus_stack_color" in old_state:
                stack_color[:, :, :current_stack_capacity] = old_state["bonus_stack_color"]

        game_stack_count = np.zeros(row_count, dtype=np.int16)
        game_stack_id = np.full(
            (row_count, next_game_stack_capacity),
            -1,
            dtype=np.int32,
        )
        game_stack_type = np.full(
            (row_count, next_game_stack_capacity),
            vector_runtime.BONUS_TYPE_NONE,
            dtype=np.int16,
        )
        game_stack_duration_ms = np.zeros(
            (row_count, next_game_stack_capacity),
            dtype=np.int32,
        )
        game_stack_borderless = np.zeros(
            (row_count, next_game_stack_capacity),
            dtype=np.int8,
        )
        if current_game_stack_capacity:
            game_stack_count[:] = old_state["bonus_game_stack_count"]
            game_stack_id[:, :current_game_stack_capacity] = old_state["bonus_game_stack_id"]
            game_stack_type[:, :current_game_stack_capacity] = old_state["bonus_game_stack_type"]
            game_stack_duration_ms[:, :current_game_stack_capacity] = old_state[
                "bonus_game_stack_duration_ms"
            ]
            game_stack_borderless[:, :current_game_stack_capacity] = old_state[
                "bonus_game_stack_borderless"
            ]

        self.state["bonus_world_active"] = old_state.get(
            "bonus_world_active",
            np.zeros(row_count, dtype=bool),
        ).copy()
        self.state["bonus_active"] = bonus_active
        self.state["bonus_type"] = bonus_type
        self.state["bonus_id"] = bonus_id
        self.state["bonus_next_id"] = bonus_next_id
        self.state["bonus_pos"] = bonus_pos
        self.state["bonus_radius"] = bonus_radius
        self.state["bonus_count"] = old_state.get(
            "bonus_count",
            np.zeros(row_count, dtype=np.int32),
        ).copy()
        self.state["bonus_world_body_count"] = old_state.get(
            "bonus_world_body_count",
            np.zeros(row_count, dtype=np.int32),
        ).copy()
        self.state["base_radius"] = old_state.get(
            "base_radius",
            self.state["radius"].copy(),
        ).copy()
        self.state["base_speed"] = old_state.get(
            "base_speed",
            self.state["speed"].copy(),
        ).copy()
        self.state["base_angular_velocity_per_ms"] = old_state.get(
            "base_angular_velocity_per_ms",
            self.state["angular_velocity_per_ms"].copy(),
        ).copy()
        self.state["base_inverse"] = old_state.get(
            "base_inverse",
            self.state["inverse"].copy(),
        ).copy()
        self.state["invincible"] = old_state.get(
            "invincible",
            np.zeros((row_count, player_count), dtype=bool),
        ).copy()
        self.state["base_invincible"] = old_state.get(
            "base_invincible",
            self.state["invincible"].copy(),
        ).copy()
        default_colors = np.tile(
            np.arange(player_count, dtype=np.int16),
            (row_count, 1),
        )
        self.state["avatar_color"] = old_state.get(
            "avatar_color",
            default_colors,
        ).copy()
        self.state["base_avatar_color"] = old_state.get(
            "base_avatar_color",
            self.state["avatar_color"].copy(),
        ).copy()
        self.state["radius_power"] = old_state.get(
            "radius_power",
            np.zeros((row_count, player_count), dtype=np.int16),
        ).copy()
        self.state["bonus_stack_count"] = stack_count
        self.state["bonus_stack_id"] = stack_id
        self.state["bonus_stack_type"] = stack_type
        self.state["bonus_stack_duration_ms"] = stack_duration_ms
        self.state["bonus_stack_radius_power"] = stack_radius_power
        self.state["bonus_stack_velocity_delta"] = stack_velocity_delta
        self.state["bonus_stack_inverse_delta"] = stack_inverse_delta
        self.state["bonus_stack_angular_velocity_per_ms"] = stack_angular_velocity_per_ms
        self.state["bonus_stack_invincible_delta"] = stack_invincible_delta
        self.state["bonus_stack_printing_delta"] = stack_printing_delta
        self.state["bonus_stack_color"] = stack_color
        self.state["bonus_game_stack_count"] = game_stack_count
        self.state["bonus_game_stack_id"] = game_stack_id
        self.state["bonus_game_stack_type"] = game_stack_type
        self.state["bonus_game_stack_duration_ms"] = game_stack_duration_ms
        self.state["bonus_game_stack_borderless"] = game_stack_borderless
        self._ensure_reset_template_bonus_arrays(
            bonus_capacity=next_bonus_capacity,
            stack_capacity=max(next_stack_capacity, next_game_stack_capacity),
        )

    def _ensure_reset_template_bonus_arrays(
        self,
        *,
        bonus_capacity: int,
        stack_capacity: int,
    ) -> None:
        template = self.reset_template
        current_capacity = (
            int(template["bonus_active"].shape[1]) if "bonus_active" in template else 0
        )
        current_stack_capacity = (
            int(template["bonus_stack_id"].shape[2]) if "bonus_stack_id" in template else 0
        )
        current_game_stack_capacity = (
            int(template["bonus_game_stack_id"].shape[1])
            if "bonus_game_stack_id" in template
            else 0
        )
        if (
            current_capacity >= bonus_capacity
            and current_stack_capacity >= stack_capacity
            and current_game_stack_capacity >= stack_capacity
            and "invincible" in template
            and "base_invincible" in template
            and "avatar_color" in template
            and "base_avatar_color" in template
            and "base_angular_velocity_per_ms" in template
            and "bonus_stack_angular_velocity_per_ms" in template
            and "bonus_stack_invincible_delta" in template
            and "bonus_stack_printing_delta" in template
            and "bonus_stack_color" in template
        ):
            return

        next_bonus_capacity = max(current_capacity, bonus_capacity)
        next_stack_capacity = max(current_stack_capacity, stack_capacity)
        next_game_stack_capacity = max(current_game_stack_capacity, stack_capacity)
        row_count = self.batch_size
        player_count = self.player_count

        bonus_active = np.zeros((row_count, next_bonus_capacity), dtype=bool)
        bonus_type = np.full(
            (row_count, next_bonus_capacity),
            vector_runtime.BONUS_TYPE_NONE,
            dtype=np.int16,
        )
        bonus_id = np.full((row_count, next_bonus_capacity), -1, dtype=np.int32)
        bonus_pos = np.zeros((row_count, next_bonus_capacity, 2), dtype=np.float64)
        bonus_radius = np.zeros((row_count, next_bonus_capacity), dtype=np.float64)
        if current_capacity:
            bonus_active[:, :current_capacity] = template["bonus_active"]
            bonus_type[:, :current_capacity] = template["bonus_type"]
            bonus_id[:, :current_capacity] = template["bonus_id"]
            bonus_pos[:, :current_capacity] = template["bonus_pos"]
            bonus_radius[:, :current_capacity] = template["bonus_radius"]

        stack_count = np.zeros((row_count, player_count), dtype=np.int16)
        stack_id = np.full(
            (row_count, player_count, next_stack_capacity),
            -1,
            dtype=np.int32,
        )
        stack_type = np.full(
            (row_count, player_count, next_stack_capacity),
            vector_runtime.BONUS_TYPE_NONE,
            dtype=np.int16,
        )
        stack_duration_ms = np.zeros(
            (row_count, player_count, next_stack_capacity),
            dtype=np.int32,
        )
        stack_radius_power = np.zeros(
            (row_count, player_count, next_stack_capacity),
            dtype=np.int16,
        )
        stack_velocity_delta = np.zeros(
            (row_count, player_count, next_stack_capacity),
            dtype=np.float64,
        )
        stack_inverse_delta = np.zeros(
            (row_count, player_count, next_stack_capacity),
            dtype=np.int16,
        )
        stack_angular_velocity_per_ms = np.zeros(
            (row_count, player_count, next_stack_capacity),
            dtype=np.float64,
        )
        stack_invincible_delta = np.zeros(
            (row_count, player_count, next_stack_capacity),
            dtype=np.int16,
        )
        stack_printing_delta = np.zeros(
            (row_count, player_count, next_stack_capacity),
            dtype=np.int16,
        )
        stack_color = np.full(
            (row_count, player_count, next_stack_capacity),
            -1,
            dtype=np.int16,
        )
        if current_stack_capacity:
            stack_count[:, :] = template["bonus_stack_count"]
            stack_id[:, :, :current_stack_capacity] = template["bonus_stack_id"]
            stack_type[:, :, :current_stack_capacity] = template["bonus_stack_type"]
            stack_duration_ms[:, :, :current_stack_capacity] = template["bonus_stack_duration_ms"]
            stack_radius_power[:, :, :current_stack_capacity] = template["bonus_stack_radius_power"]
            if "bonus_stack_velocity_delta" in template:
                stack_velocity_delta[:, :, :current_stack_capacity] = template[
                    "bonus_stack_velocity_delta"
                ]
            if "bonus_stack_inverse_delta" in template:
                stack_inverse_delta[:, :, :current_stack_capacity] = template[
                    "bonus_stack_inverse_delta"
                ]
            if "bonus_stack_angular_velocity_per_ms" in template:
                stack_angular_velocity_per_ms[:, :, :current_stack_capacity] = template[
                    "bonus_stack_angular_velocity_per_ms"
                ]
            if "bonus_stack_invincible_delta" in template:
                stack_invincible_delta[:, :, :current_stack_capacity] = template[
                    "bonus_stack_invincible_delta"
                ]
            if "bonus_stack_printing_delta" in template:
                stack_printing_delta[:, :, :current_stack_capacity] = template[
                    "bonus_stack_printing_delta"
                ]
            if "bonus_stack_color" in template:
                stack_color[:, :, :current_stack_capacity] = template["bonus_stack_color"]

        game_stack_count = np.zeros(row_count, dtype=np.int16)
        game_stack_id = np.full(
            (row_count, next_game_stack_capacity),
            -1,
            dtype=np.int32,
        )
        game_stack_type = np.full(
            (row_count, next_game_stack_capacity),
            vector_runtime.BONUS_TYPE_NONE,
            dtype=np.int16,
        )
        game_stack_duration_ms = np.zeros(
            (row_count, next_game_stack_capacity),
            dtype=np.int32,
        )
        game_stack_borderless = np.zeros(
            (row_count, next_game_stack_capacity),
            dtype=np.int8,
        )
        if current_game_stack_capacity:
            game_stack_count[:] = template["bonus_game_stack_count"]
            game_stack_id[:, :current_game_stack_capacity] = template["bonus_game_stack_id"]
            game_stack_type[:, :current_game_stack_capacity] = template["bonus_game_stack_type"]
            game_stack_duration_ms[:, :current_game_stack_capacity] = template[
                "bonus_game_stack_duration_ms"
            ]
            game_stack_borderless[:, :current_game_stack_capacity] = template[
                "bonus_game_stack_borderless"
            ]

        template["bonus_world_active"] = np.zeros(row_count, dtype=bool)
        template["bonus_active"] = bonus_active
        template["bonus_type"] = bonus_type
        template["bonus_id"] = bonus_id
        template["bonus_next_id"] = np.ones(row_count, dtype=np.int32)
        template["bonus_pos"] = bonus_pos
        template["bonus_radius"] = bonus_radius
        template["bonus_count"] = np.zeros(row_count, dtype=np.int32)
        template["bonus_world_body_count"] = np.zeros(row_count, dtype=np.int32)
        template["base_radius"] = template["radius"].copy()
        template["base_speed"] = template["speed"].copy()
        template["base_angular_velocity_per_ms"] = template["angular_velocity_per_ms"].copy()
        template["base_inverse"] = template["inverse"].copy()
        template["invincible"] = np.zeros((row_count, player_count), dtype=bool)
        template["base_invincible"] = template["invincible"].copy()
        default_colors = np.tile(
            np.arange(player_count, dtype=np.int16),
            (row_count, 1),
        )
        template["avatar_color"] = default_colors.copy()
        template["base_avatar_color"] = default_colors.copy()
        template["radius_power"] = np.zeros(
            (row_count, player_count),
            dtype=np.int16,
        )
        template["bonus_stack_count"] = stack_count
        template["bonus_stack_id"] = stack_id
        template["bonus_stack_type"] = stack_type
        template["bonus_stack_duration_ms"] = stack_duration_ms
        template["bonus_stack_radius_power"] = stack_radius_power
        template["bonus_stack_velocity_delta"] = stack_velocity_delta
        template["bonus_stack_inverse_delta"] = stack_inverse_delta
        template["bonus_stack_angular_velocity_per_ms"] = stack_angular_velocity_per_ms
        template["bonus_stack_invincible_delta"] = stack_invincible_delta
        template["bonus_stack_printing_delta"] = stack_printing_delta
        template["bonus_stack_color"] = stack_color
        template["bonus_game_stack_count"] = game_stack_count
        template["bonus_game_stack_id"] = game_stack_id
        template["bonus_game_stack_type"] = game_stack_type
        template["bonus_game_stack_duration_ms"] = game_stack_duration_ms
        template["bonus_game_stack_borderless"] = game_stack_borderless

    def _clear_seeded_bonus_rows(self, rows: np.ndarray) -> None:
        if "bonus_active" not in self.state:
            self._seeded_bonus_enabled[np.asarray(rows, dtype=bool)] = False
            return
        row_mask = np.asarray(rows, dtype=bool)
        self.state["bonus_world_active"][row_mask] = False
        self.state["bonus_active"][row_mask, :] = False
        self.state["bonus_type"][row_mask, :] = vector_runtime.BONUS_TYPE_NONE
        self.state["bonus_id"][row_mask, :] = -1
        if "bonus_next_id" in self.state:
            self.state["bonus_next_id"][row_mask] = 1
        self.state["bonus_pos"][row_mask, :, :] = 0.0
        self.state["bonus_radius"][row_mask, :] = 0.0
        self.state["bonus_count"][row_mask] = 0
        self.state["bonus_world_body_count"][row_mask] = 0
        self.state["base_radius"][row_mask] = self.state["radius"][row_mask]
        self.state["base_speed"][row_mask] = self.state["speed"][row_mask]
        self.state["angular_velocity_per_ms"][row_mask] = self.state[
            "base_angular_velocity_per_ms"
        ][row_mask]
        self.state["base_angular_velocity_per_ms"][row_mask] = self.state[
            "angular_velocity_per_ms"
        ][row_mask]
        self.state["inverse"][row_mask] = self.state["base_inverse"][row_mask]
        self.state["base_inverse"][row_mask] = self.state["inverse"][row_mask]
        self.state["invincible"][row_mask] = self.state["base_invincible"][row_mask]
        self.state["base_invincible"][row_mask] = self.state["invincible"][row_mask]
        self.state["avatar_color"][row_mask] = self.state["base_avatar_color"][row_mask]
        self.state["base_avatar_color"][row_mask] = self.state["avatar_color"][row_mask]
        self.state["radius_power"][row_mask, :] = 0
        self.state["bonus_stack_count"][row_mask, :] = 0
        self.state["bonus_stack_id"][row_mask, :, :] = -1
        self.state["bonus_stack_type"][row_mask, :, :] = vector_runtime.BONUS_TYPE_NONE
        self.state["bonus_stack_duration_ms"][row_mask, :, :] = 0
        self.state["bonus_stack_radius_power"][row_mask, :, :] = 0
        self.state["bonus_stack_velocity_delta"][row_mask, :, :] = 0.0
        self.state["bonus_stack_inverse_delta"][row_mask, :, :] = 0
        self.state["bonus_stack_angular_velocity_per_ms"][row_mask, :, :] = 0.0
        self.state["bonus_stack_invincible_delta"][row_mask, :, :] = 0
        self.state["bonus_stack_printing_delta"][row_mask, :, :] = 0
        self.state["bonus_stack_color"][row_mask, :, :] = -1
        self.state["bonus_game_stack_count"][row_mask] = 0
        self.state["bonus_game_stack_id"][row_mask, :] = -1
        self.state["bonus_game_stack_type"][
            row_mask,
            :,
        ] = vector_runtime.BONUS_TYPE_NONE
        self.state["bonus_game_stack_duration_ms"][row_mask, :] = 0
        self.state["bonus_game_stack_borderless"][row_mask, :] = 0
        self._seeded_bonus_enabled[row_mask] = False

    def _install_public_warmdown_adapter_rows(self) -> None:
        terminal_rows = (
            self._needs_reset
            & self.state["done"]
            & self.state["terminated"]
            & ~self.state["truncated"]
        )
        if not bool(terminal_rows.any()):
            return
        self._mark_public_round_warmdown_rows(terminal_rows)
        self.state["done"][terminal_rows] = False
        self.state["terminated"][terminal_rows] = False
        self.state["reset_pending"][terminal_rows] = False
        self._schedule_warmdown_timer_rows(terminal_rows)

    def _stage_match_mode_warmdown_rows(self, terminal_rows: np.ndarray) -> None:
        self._mark_public_round_warmdown_rows(terminal_rows)
        self.state["done"][terminal_rows] = False
        self.state["terminated"][terminal_rows] = False
        self.state["reset_pending"][terminal_rows] = False
        self.state["in_round"][terminal_rows] = False
        self._schedule_warmdown_timer_rows(terminal_rows)

    def _mark_public_round_warmdown_rows(self, rows: np.ndarray) -> None:
        row_mask = np.asarray(rows, dtype=bool)
        self.state["round_done"][row_mask] = True
        self.state["warmdown_pending"][row_mask] = True
        self.state["match_done"][row_mask] = False
        self.state["round_winner"][row_mask] = self.state["winner"][row_mask]
        self.state["match_winner"][row_mask] = -1

    def _clear_public_lifecycle_rows(self, rows: np.ndarray) -> None:
        row_mask = np.asarray(rows, dtype=bool)
        self.state["round_done"][row_mask] = False
        self.state["warmdown_pending"][row_mask] = False
        self.state["match_done"][row_mask] = False
        self.state["round_winner"][row_mask] = -1
        self.state["match_winner"][row_mask] = -1

    def _schedule_warmdown_timer_rows(self, rows: np.ndarray) -> None:
        for row in np.flatnonzero(np.asarray(rows, dtype=bool)):
            row_int = int(row)
            self.state["timer_active"][row_int, :] = False
            self.state["timer_remaining_ms"][row_int, :] = 0.0
            self.state["timer_kind"][row_int, :] = vector_runtime.TIMER_KIND_NONE
            self.state["timer_player"][row_int, :] = vector_runtime.TIMER_PLAYER_NONE
            self.state["timer_seq"][row_int, :] = 0
            self.state["timer_overflow"][row_int] = False
            self.state["timer_active"][row_int, 0] = True
            self.state["timer_remaining_ms"][row_int, 0] = vector_lifecycle.SOURCE_ROUND_WARMDOWN_MS
            self.state["timer_kind"][row_int, 0] = vector_runtime.TIMER_KIND_WARMDOWN_END
            self.state["timer_player"][row_int, 0] = vector_runtime.TIMER_PLAYER_NONE
            self.state["timer_seq"][row_int, 0] = 0

    def _warmdown_pending_mask(self) -> np.ndarray:
        if "warmdown_pending" not in self.state:
            return np.zeros(self.batch_size, dtype=bool)
        pending = np.asarray(self.state["warmdown_pending"], dtype=bool)
        if pending.shape != (self.batch_size,):
            return np.zeros(self.batch_size, dtype=bool)
        return pending.copy()

    def _append_new_deaths(self, pre_alive: np.ndarray) -> None:
        post_alive = self.state["alive"][:, : self.player_count]
        new_deaths = pre_alive & ~post_alive
        for row in np.flatnonzero(new_deaths.any(axis=1)):
            row_int = int(row)
            count = int(self.state["death_count"][row_int])
            existing = set(
                int(player)
                for player in self.state["death_player"][row_int, :count]
                if int(player) >= 0
            )
            for player in range(self.player_count - 1, -1, -1):
                if not bool(new_deaths[row_int, player]) or player in existing:
                    continue
                if count >= self.state["death_player"].shape[1]:
                    self.state["overflow"][row_int] = True
                    break
                self.state["death_player"][row_int, count] = player
                self.state["death_cause"][row_int, count] = vector_runtime.DEATH_CAUSE_BODY_UNKNOWN
                self.state["death_hit_owner"][row_int, count] = -1
                count += 1
                existing.add(player)
            self.state["death_count"][row_int] = count

    def _correct_leave_adjusted_death_scores(
        self,
        *,
        pre_alive: np.ndarray,
        pre_death_count: np.ndarray,
    ) -> None:
        post_alive = self.state["alive"][:, : self.player_count]
        new_deaths = pre_alive & ~post_alive
        if not bool(new_deaths.any()):
            return

        runtime_prior_deaths = self.player_count - pre_alive.sum(axis=1)
        score_overcount = runtime_prior_deaths - pre_death_count
        for row in np.flatnonzero(new_deaths.any(axis=1) & (score_overcount > 0)):
            row_int = int(row)
            players = new_deaths[row_int]
            correction = int(score_overcount[row_int])
            if bool(self.state["done"][row_int]):
                self.state["score"][row_int, players] -= correction
            else:
                self.state["round_score"][row_int, players] -= correction

    def _resolve_immediate_leave_terminal_rows(self, rows: np.ndarray) -> None:
        row_mask = np.asarray(rows, dtype=bool)
        present_alive = (
            self.state["present"][:, : self.player_count]
            & self.state["alive"][:, : self.player_count]
        )
        for row in np.flatnonzero(row_mask):
            row_int = int(row)
            live_players = np.flatnonzero(present_alive[row_int])
            winner = int(live_players[0]) if live_players.size == 1 else -1
            if winner >= 0:
                self.state["round_score"][row_int, winner] += max(
                    self.player_count - 1,
                    1,
                )
            self.state["score"][row_int, : self.player_count] += self.state["round_score"][
                row_int, : self.player_count
            ]
            self.state["round_score"][row_int, : self.player_count] = 0
            self.state["done"][row_int] = True
            self.state["terminated"][row_int] = True
            self.state["truncated"][row_int] = False
            self.state["reset_pending"][row_int] = True
            self.state["terminal_reason"][row_int] = (
                vector_reset.TERMINAL_REASON_SURVIVOR_WIN
                if winner >= 0
                else vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW
            )
            self.state["draw"][row_int] = winner < 0
            self.state["winner"][row_int] = winner

    def _mark_overflow_truncations(self, pre_active: np.ndarray) -> None:
        truncatable = pre_active & ~self.state["done"] & ~self.state["terminated"]
        body_mask = truncatable & (self.state["body_overflow"] | self.state["overflow"])
        event_mask = truncatable & self.state["event_overflow"]
        overflow_mask = body_mask | event_mask
        if not bool(overflow_mask.any()):
            return
        self.state["done"][overflow_mask] = True
        self.state["terminated"][overflow_mask] = False
        self.state["truncated"][overflow_mask] = True
        self.state["reset_pending"][overflow_mask] = True
        self.state["terminal_reason"][body_mask] = (
            vector_reset.TERMINAL_REASON_BODY_OVERFLOW_TRUNCATED
        )
        self.state["terminal_reason"][event_mask] = (
            vector_reset.TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED
        )

    def _mark_timeout_truncations(self, pre_active: np.ndarray) -> None:
        timeout_mask = pre_active & ~self.state["done"] & (self.state["tick"] >= self.max_ticks)
        if not bool(timeout_mask.any()):
            return
        self.state["done"][timeout_mask] = True
        self.state["terminated"][timeout_mask] = False
        self.state["truncated"][timeout_mask] = True
        self.state["reset_pending"][timeout_mask] = True
        self.state["terminal_reason"][timeout_mask] = vector_reset.TERMINAL_REASON_TIMEOUT_TRUNCATED

    def _reward(self) -> np.ndarray:
        reward = np.zeros((self.batch_size, self.player_count), dtype=np.float32)
        present = self.state["present"][:, : self.player_count]
        round_done = self._metadata_state_array(
            "round_done",
            np.zeros(self.batch_size, dtype=bool),
            dtype=bool,
        )
        reward_rows = self.state["terminated"] | round_done
        for row in range(self.batch_size):
            if not bool(reward_rows[row]):
                continue
            reason = int(self.state["terminal_reason"][row])
            if reason != vector_reset.TERMINAL_REASON_SURVIVOR_WIN:
                continue
            winner = int(self.state["winner"][row])
            if winner < 0:
                continue
            reward[row, present[row]] = -1.0
            reward[row, winner] = 1.0
        return reward

    def _batch(
        self,
        *,
        reward: np.ndarray,
        done: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        final_observation: np.ndarray | None,
        final_reward: np.ndarray | None,
        final_row_mask: np.ndarray | None = None,
        info: dict[str, Any],
    ) -> VectorMultiplayerBatch:
        if final_row_mask is None:
            terminal_mask = np.zeros(self.batch_size, dtype=bool)
        else:
            terminal_mask = np.asarray(final_row_mask, dtype=bool)
            if terminal_mask.shape != (self.batch_size,):
                raise VectorMultiplayerEnvError("final_row_mask must have shape [B]")
            terminal_mask = terminal_mask.copy()
        final_rows = np.flatnonzero(terminal_mask).astype(np.int32)
        final_observation_array = None
        if final_observation is not None:
            final_observation_array = np.asarray(final_observation, dtype=np.float32).copy()
            final_observation_array[~terminal_mask, ...] = 0.0
        final_reward_array = None
        if final_reward is not None:
            final_reward_array = np.asarray(final_reward, dtype=np.float32).copy()
            final_reward_array[~terminal_mask, ...] = 0.0
        batch_info = dict(info)
        if final_observation_array is not None:
            batch_info["final_observation"] = final_observation_array.copy()
        if final_reward_array is not None:
            batch_info["final_reward_map"] = final_reward_array.copy()
        batch_info.setdefault("final_observation_rows", final_rows.copy())
        batch_info.setdefault("final_observation_row_mask", terminal_mask.copy())
        batch_info.setdefault("final_reward_rows", final_rows.copy())
        batch_info.setdefault("final_reward_row_mask", terminal_mask.copy())
        batch_info.setdefault(
            "final_observation_row_policy",
            _final_row_policy(
                "final_observation",
                terminal_mask,
                present=final_observation is not None,
            ),
        )
        batch_info.setdefault(
            "final_reward_row_policy",
            _final_row_policy(
                "final_reward",
                terminal_mask,
                present=final_reward is not None,
            ),
        )
        return VectorMultiplayerBatch(
            observation=self._observe_array(),
            action_mask=self._action_mask(),
            reward=np.asarray(reward, dtype=np.float32).copy(),
            done=np.asarray(done, dtype=bool).copy(),
            terminated=np.asarray(terminated, dtype=bool).copy(),
            truncated=np.asarray(truncated, dtype=bool).copy(),
            final_observation=final_observation_array,
            final_reward=final_reward_array,
            info=batch_info,
        )

    def _observe_array(self) -> np.ndarray:
        observation = np.zeros(
            (
                self.batch_size,
                self.player_count,
                len(DEBUG_METADATA_OBSERVATION_FIELDS),
            ),
            dtype=np.float32,
        )
        present = self.state["present"][:, : self.player_count]
        alive = self.state["alive"][:, : self.player_count]
        observation[:, :, 0] = present.astype(np.float32)
        observation[:, :, 1] = alive.astype(np.float32)
        observation[:, :, 2] = self.state["score"][:, : self.player_count].astype(np.float32)
        observation[:, :, 3] = self.state["round_score"][:, : self.player_count].astype(np.float32)
        observation[:, :, 4] = -1.0
        for row in range(self.batch_size):
            for death_index, player in enumerate(
                self.state["death_player"][row, : int(self.state["death_count"][row])]
            ):
                player_int = int(player)
                if 0 <= player_int < self.player_count:
                    observation[row, player_int, 4] = float(death_index)
        observation[:, :, 5] = self.state["done"][:, None].astype(np.float32)
        return observation

    def _action_mask(self) -> np.ndarray:
        warmdown_pending = self._warmdown_pending_mask()
        active = (
            self.state["present"][:, : self.player_count]
            & self.state["alive"][:, : self.player_count]
            & ~self.state["done"][:, None]
            & ~warmdown_pending[:, None]
        )
        return np.repeat(active[:, :, None], ACTION_COUNT, axis=2)

    def _source_moves_and_action_sidecar(
        self,
        actions: np.ndarray,
        *,
        pre_alive: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        action_array = np.asarray(actions)
        if action_array.shape != (self.batch_size, self.player_count):
            raise VectorMultiplayerEnvError("actions must have shape [B,P]")
        if not np.issubdtype(action_array.dtype, np.integer):
            raise VectorMultiplayerEnvError("actions must be integer action ids")

        action_required = (
            self.state["present"][:, : self.player_count] & pre_alive & ~self.state["done"][:, None]
        )
        valid = (action_array >= 0) & (action_array < ACTION_COUNT)
        if bool((action_required & ~valid).any()):
            raise VectorMultiplayerEnvError(
                "live present players require left/straight/right action ids"
            )
        inactive_invalid = ~action_required & ~((action_array == -1) | valid)
        if bool(inactive_invalid.any()):
            raise VectorMultiplayerEnvError(
                "inactive action slots must be -1 or a left/straight/right action id"
            )

        move_lookup = np.asarray(ACTION_ID_TO_SOURCE_MOVE, dtype=np.int8)
        native = np.zeros((self.batch_size, self.player_count), dtype=np.int8)
        native[action_required] = move_lookup[
            action_array.astype(np.int64, copy=False)[action_required]
        ]
        action_source = np.full(
            (self.batch_size, self.player_count),
            "external_joint_action",
            dtype=object,
        )
        present = self.state["present"][:, : self.player_count]
        action_source[~present] = "absent_noop"
        action_source[present & ~pre_alive] = "dead_noop"
        terminal_padding = np.repeat(self.state["done"][:, None], self.player_count, axis=1)
        action_source[terminal_padding] = "terminal_padding"
        return native, {
            "schema_id": PUBLIC_JOINT_ACTION_SIDECAR_SCHEMA_ID,
            "action_space_id": ACTION_SPACE_ID,
            "decision_ms": self.decision_ms,
            "player_count": self.player_count,
            "player_action": action_array.astype(np.int16, copy=True),
            "player_action_mask": self._action_mask(),
            "action_required": action_required.copy(),
            "action_source": action_source,
            "native_control_value": native.copy(),
            "metadata_only": True,
            "joint_action_mcts_claim": False,
            "action_map_policy_id": "external_joint_action_player_major_no_mcts/v0",
            "ignored_action_policy": (
                "inactive_slots_accept_minus_one_or_valid_action_and_step_as_straight/v0"
            ),
            "joint_action_schema_id": JOINT_ACTION_SCHEMA_ID,
        }

    def _prepare_reset_template_rows(
        self,
        mask: np.ndarray,
        reset_seed: np.ndarray,
        *,
        present: np.ndarray,
        source_fixture_random_tape_values: np.ndarray | None = None,
    ) -> None:
        rows = np.flatnonzero(mask)
        template = self.reset_template
        template["episode_step"][mask] = 0
        template["env_active"][mask] = True
        template["reset_pending"][mask] = False
        template["done"][mask] = False
        template["terminated"][mask] = False
        template["truncated"][mask] = False
        template["terminal_reason"][mask] = vector_reset.TERMINAL_REASON_NONE
        template["round_done"][mask] = False
        template["warmdown_pending"][mask] = False
        template["match_done"][mask] = False
        template["round_winner"][mask] = -1
        template["match_winner"][mask] = -1
        template["reset_seed"][mask] = reset_seed[mask]
        template["reset_source"][mask] = vector_reset.RESET_SOURCE_MANUAL
        template["tick"][mask] = 0
        template["elapsed_ms"][mask] = 0.0
        template["pos"][mask, ...] = 0.0
        template["prev_pos"][mask, ...] = 0.0
        template["heading"][mask, ...] = 0.0
        template["alive"][mask, ...] = False
        template["present"][mask, ...] = present[mask]
        template["map_size"][mask] = self.map_size
        template["max_score"][mask] = self.max_score
        template["started"][mask] = False
        template["in_round"][mask] = False
        template["world_active"][mask] = False
        template["world_body_count"][mask] = 0
        template["timer_active"][mask, ...] = False
        template["timer_remaining_ms"][mask, ...] = 0.0
        template["timer_kind"][mask, ...] = vector_runtime.TIMER_KIND_NONE
        template["timer_player"][mask, ...] = vector_runtime.TIMER_PLAYER_NONE
        template["timer_seq"][mask, ...] = 0
        template["timer_overflow"][mask] = False
        template["score"][mask, ...] = 0
        template["round_score"][mask, ...] = 0
        template["printing"][mask, ...] = False
        template["print_manager_active"][mask, ...] = False
        template["print_manager_distance"][mask, ...] = 0.0
        template["print_manager_last_pos"][mask, ...] = 0.0
        template["death_count"][mask] = 0
        template["death_player"][mask, ...] = -1
        template["death_cause"][mask, ...] = vector_runtime.DEATH_CAUSE_NONE
        template["death_hit_owner"][mask, ...] = -1
        template["overflow"][mask] = False
        template["borderless"][mask] = False
        template["radius"][mask, ...] = self.radius
        template["speed"][mask, ...] = self.speed
        template["base_speed"][mask, ...] = self.speed
        template["inverse"][mask, ...] = False
        template["base_inverse"][mask, ...] = False
        if "invincible" in template:
            template["invincible"][mask, ...] = False
            template["base_invincible"][mask, ...] = False
        if "avatar_color" in template:
            default_colors = np.tile(
                np.arange(self.player_count, dtype=np.int16),
                (self.batch_size, 1),
            )
            template["avatar_color"][mask, ...] = default_colors[mask]
            template["base_avatar_color"][mask, ...] = default_colors[mask]
        template["angular_velocity_per_ms"][mask, ...] = self.angular_velocity_per_ms
        if "base_angular_velocity_per_ms" in template:
            template["base_angular_velocity_per_ms"][
                mask,
                ...,
            ] = self.angular_velocity_per_ms
        template["live_body_num"][mask, ...] = 0
        template["trail_latency"][mask, ...] = self.trail_latency
        template["death_tick"][mask, ...] = -1
        template["draw"][mask] = False
        template["winner"][mask] = -1
        template["body_overflow"][mask] = False
        template["visible_trail_count"][mask, ...] = 0
        template["has_visible_trail_last"][mask, ...] = False
        template["visible_trail_last_pos"][mask, ...] = 0.0
        template["has_draw_cursor"][mask, ...] = False
        template["draw_cursor_pos"][mask, ...] = 0.0
        template["visual_trail_active"][mask, ...] = False
        template["visual_trail_pos"][mask, ...] = 0.0
        template["visual_trail_radius"][mask, ...] = 0.0
        template["visual_trail_owner"][mask, ...] = -1
        template["visual_trail_break_before"][mask, ...] = False
        template["visual_trail_write_cursor"][mask] = 0
        template["visual_trail_overflow"][mask] = False
        template["has_visual_trail_last"][mask, ...] = False
        template["visual_trail_last_pos"][mask, ...] = 0.0
        template["body_active"][mask, ...] = False
        template["body_pos"][mask, ...] = 0.0
        template["body_radius"][mask, ...] = 0.0
        template["body_owner"][mask, ...] = -1
        template["body_num"][mask, ...] = -1
        template["body_insert_tick"][mask, ...] = -1
        template["body_insert_kind"][mask, ...] = -1
        template["body_break_before"][mask, ...] = False
        template["body_write_cursor"][mask] = 0
        template["body_count"][mask, ...] = 0
        template["event_count"][mask] = 0
        template["event_mask"][mask, ...] = False
        template["event_type"][mask, ...] = vector_runtime.EVENT_NONE
        template["event_player"][mask, ...] = -1
        template["event_other"][mask, ...] = -1
        template["event_bool"][mask, ...] = -1
        template["event_value_i"][mask, ...] = 0
        template["event_value_f"][mask, ...] = 0.0
        template["event_overflow"][mask] = False
        template["event_overflow_attempts"][mask] = 0
        template["bonus_catch_count_step"][mask, ...] = 0
        if "bonus_active" in template:
            template["bonus_world_active"][mask] = False
            template["bonus_active"][mask, ...] = False
            template["bonus_type"][mask, ...] = vector_runtime.BONUS_TYPE_NONE
            template["bonus_id"][mask, ...] = -1
            template["bonus_next_id"][mask] = 1
            template["bonus_pos"][mask, ...] = 0.0
            template["bonus_radius"][mask, ...] = 0.0
            template["bonus_count"][mask] = 0
            template["bonus_world_body_count"][mask] = 0
            template["base_radius"][mask] = template["radius"][mask]
            template["base_speed"][mask] = template["speed"][mask]
            template["base_angular_velocity_per_ms"][mask] = template["angular_velocity_per_ms"][
                mask
            ]
            template["inverse"][mask] = False
            template["base_inverse"][mask] = False
            template["invincible"][mask] = False
            template["base_invincible"][mask] = False
            default_colors = np.tile(
                np.arange(self.player_count, dtype=np.int16),
                (self.batch_size, 1),
            )
            template["avatar_color"][mask] = default_colors[mask]
            template["base_avatar_color"][mask] = default_colors[mask]
            template["radius_power"][mask, ...] = 0
            template["bonus_stack_count"][mask, ...] = 0
            template["bonus_stack_id"][mask, ...] = -1
            template["bonus_stack_type"][mask, ...] = vector_runtime.BONUS_TYPE_NONE
            template["bonus_stack_duration_ms"][mask, ...] = 0
            template["bonus_stack_radius_power"][mask, ...] = 0
            template["bonus_stack_velocity_delta"][mask, ...] = 0.0
            template["bonus_stack_inverse_delta"][mask, ...] = 0
            template["bonus_stack_angular_velocity_per_ms"][mask, ...] = 0.0
            template["bonus_stack_invincible_delta"][mask, ...] = 0
            template["bonus_stack_printing_delta"][mask, ...] = 0
            template["bonus_stack_color"][mask, ...] = -1
            template["bonus_game_stack_count"][mask] = 0
            template["bonus_game_stack_id"][mask, ...] = -1
            template["bonus_game_stack_type"][
                mask,
                ...,
            ] = vector_runtime.BONUS_TYPE_NONE
            template["bonus_game_stack_duration_ms"][mask, ...] = 0
            template["bonus_game_stack_borderless"][mask, ...] = 0
        template["random_tape_length"][mask] = self.random_tape_capacity
        template["random_tape_cursor"][mask] = 0
        template["random_tape_exhausted"][mask] = False
        template["random_tape_draw_count"][mask] = 0

        if source_fixture_random_tape_values is None and rows.size:
            template["random_tape_values"][rows, :] = (
                vector_source_random.seeded_source_math_random_history(
                    reset_seed[rows],
                    length=self.random_tape_capacity,
                )
            )
        elif source_fixture_random_tape_values is not None and rows.size:
            fixture_length = source_fixture_random_tape_values.shape[1]
            template["random_tape_values"][rows, :] = 0.0
            template["random_tape_values"][rows, :fixture_length] = (
                source_fixture_random_tape_values[rows, :]
            )
            template["random_tape_length"][rows] = fixture_length

    def _public_info(self) -> dict[str, Any]:
        terminal_reason = self.state["terminal_reason"].copy()
        terminal_reason_name = np.asarray(
            [_TERMINAL_REASON_NAMES[int(reason)] for reason in terminal_reason],
            dtype=object,
        )
        truncated = self.state["truncated"].copy()
        done = self.state["done"].copy()
        terminated = self.state["terminated"].copy()
        round_done = self._metadata_state_array("round_done", terminated.copy(), dtype=bool)
        warmdown_pending = self._metadata_state_array(
            "warmdown_pending",
            np.zeros(self.batch_size, dtype=bool),
            dtype=bool,
        )
        match_done = self._metadata_state_array(
            "match_done",
            np.zeros(self.batch_size, dtype=bool),
            dtype=bool,
        )
        round_winner = self._metadata_state_array(
            "round_winner",
            self.state["winner"].copy(),
            dtype=np.int16,
        )
        match_winner = self._metadata_state_array(
            "match_winner",
            np.full(self.batch_size, -1, dtype=np.int16),
            dtype=np.int16,
        )
        bonus_info = self._bonus_public_info()
        seeded_bonus_active = bool(self._seeded_bonus_enabled.any())
        natural_bonus_enabled = bool(self._natural_bonus_spawn_enabled.any())
        present = self.state["present"][:, : self.player_count].copy()
        alive = self.state["alive"][:, : self.player_count].copy()
        reset_provenance = self._reset_provenance_info()
        return {
            "public_env_contract_id": PUBLIC_ENV_CONTRACT_ID,
            "env_impl_id": ENV_IMPL_ID,
            "env_version_policy_id": ENV_VERSION_POLICY_ID,
            "env_feature_mode": bonus_info["mode"],
            "ruleset_id": (
                NATURAL_BONUS_RULESET_ID
                if natural_bonus_enabled
                else (SEEDED_BONUS_RULESET_ID if seeded_bonus_active else RULESET_ID)
            ),
            "rules_hash": (
                NATURAL_BONUS_RULES_HASH
                if natural_bonus_enabled
                else (SEEDED_BONUS_RULES_HASH if seeded_bonus_active else RULES_HASH)
            ),
            "base_public_env_contract_id": PUBLIC_ENV_CONTRACT_ID,
            "seeded_bonus_public_env_contract_id": PUBLIC_SEEDED_BONUS_ENV_CONTRACT_ID,
            "natural_bonus_public_env_contract_id": PUBLIC_NATURAL_BONUS_ENV_CONTRACT_ID,
            "bonus_support": bonus_info,
            "bonus_support_mode": bonus_info["mode"],
            "bonus_support_mode_by_row": bonus_info["mode_by_row"].copy(),
            "public_mechanics_gaps": bonus_info["source_default_gap_claims"],
            "death_mode": self.death_mode,
            "death_immunity_player_ids": self.death_immunity_player_ids.copy(),
            "death_immunity_mask": self._death_immunity_mask(),
            "death_immunity_diagnostic": bool(self.death_immunity_player_ids.size),
            "death_immunity_claim": (
                "diagnostic_not_source_faithful"
                if self.death_immunity_player_ids.size
                else "none"
            ),
            "death_suppression_for_profile": (
                self.death_mode == vector_runtime.DEATH_MODE_PROFILE_NO_DEATH
            ),
            "death_suppression_claim": (
                "profile_only_not_source_fidelity"
                if self.death_mode == vector_runtime.DEATH_MODE_PROFILE_NO_DEATH
                else "none"
            ),
            "native_control_model_id": NATIVE_CONTROL_MODEL_ID,
            "trainer_control_wrapper_id": TRAINER_CONTROL_WRAPPER_ID,
            "decision_ms": self.decision_ms,
            "source_frame_decision": self.source_frame_decision,
            "decision_source_frames": self.decision_source_frames,
            "source_physics_step_ms": self.source_physics_step_ms,
            "source_frame_decision_policy_id": SOURCE_FRAME_DECISION_POLICY_ID,
            "source_physics_step_policy_id": SOURCE_PHYSICS_STEP_POLICY_ID,
            "lifecycle_policy_id": LIFECYCLE_POLICY_ID,
            "reset_episode_id": self.state["episode_id"].copy(),
            "reset_episode_id_policy": RESET_EPISODE_ID_POLICY,
            "episode_id": self.state["episode_id"].copy(),
            "round_id": self.state["round_id"].copy(),
            "source_round_id": self.state["round_id"].copy(),
            "source_round_id_policy": SOURCE_ROUND_ID_POLICY,
            "episode_end_mode": self.episode_end_mode,
            "player_count": self.player_count,
            "supported_player_counts": SUPPORTED_PLAYER_COUNTS,
            "player_count_by_row": np.full(
                self.batch_size,
                self.player_count,
                dtype=np.int16,
            ),
            "player_ids": tuple(self.player_ids),
            "source_player_ids": np.arange(1, self.player_count + 1, dtype=np.int16),
            "map_size": self.state["map_size"].copy(),
            "present": present,
            "alive": alive,
            "present_player_count_by_row": present.sum(axis=1).astype(np.int16),
            "alive_player_count_by_row": alive.sum(axis=1).astype(np.int16),
            "score": self.state["score"][:, : self.player_count].copy(),
            "scores": self.state["score"][:, : self.player_count].copy(),
            "round_score": self.state["round_score"][:, : self.player_count].copy(),
            "round_scores": self.state["round_score"][:, : self.player_count].copy(),
            "borderless": self.state["borderless"].copy(),
            "death_player": self.state["death_player"].copy(),
            "death_count": self.state["death_count"].copy(),
            "death_cause": self.state["death_cause"].copy(),
            "death_cause_name": vector_runtime.death_cause_name_array(
                self.state["death_cause"],
            ),
            "death_hit_owner": self.state["death_hit_owner"].copy(),
            "death_order_policy": DEATH_ORDER_POLICY,
            "reset_seed": self.state["reset_seed"].copy(),
            "reset_source": self.state["reset_source"].copy(),
            "random_tape_cursor": self.state["random_tape_cursor"].copy(),
            "random_tape_draw_count": self.state["random_tape_draw_count"].copy(),
            "random_tape_source": self._random_tape_source.copy(),
            "random_tape_length": self.state["random_tape_length"].copy(),
            "natural_bonus_timer_active": self._natural_bonus_timer_active.copy(),
            "natural_bonus_timer_remaining_ms": (self._natural_bonus_timer_remaining_ms.copy()),
            "natural_bonus_next_due_elapsed_ms": (self._natural_bonus_next_due_elapsed_ms.copy()),
            "natural_bonus_pop_count": self._natural_bonus_pop_count.copy(),
            "bonus_catch_count_step": self.state["bonus_catch_count_step"][
                :,
                : self.player_count,
            ].copy(),
            "rng_impl_id": self._rng_impl_id.copy(),
            "rng_history_ref": self._rng_impl_id.copy(),
            "random_tape_history_ref": self._rng_impl_id.copy(),
            "source_fixture_ref": self._source_fixture_ref.copy(),
            "reset_provenance_policy_id": RESET_PROVENANCE_POLICY_ID,
            "reset_provenance": reset_provenance,
            "observation_schema_id": DEBUG_METADATA_OBSERVATION_SCHEMA_ID,
            "observation_schema_hash": DEBUG_METADATA_OBSERVATION_SCHEMA_HASH,
            "observation_schema": DEBUG_METADATA_OBSERVATION_SCHEMA,
            "action_space_id": ACTION_SPACE_ID,
            "action_space_hash": ACTION_SPACE_HASH,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "reward_schema_hash": REWARD_SCHEMA_HASH,
            "final_observation_policy": FINAL_OBSERVATION_POLICY,
            "metadata_only": True,
            "trainer_observation_claim": False,
            "trainer_replay_claim": False,
            "learned_observation_claim": False,
            "public_env_trainer_ready_claim": False,
            "trainer_observation_schema_id": None,
            "public_observation_claim": "debug_metadata_only_not_trainer_observation/v0",
            "warmdown_waited": False,
            "natural_multiplayer_reset_claim": bool(self._natural_multiplayer_reset_claim.all()),
            "natural_multiplayer_reset_claim_by_row": (
                self._natural_multiplayer_reset_claim.copy()
            ),
            "step_index": self.state["episode_step"].copy(),
            "tick_index": self.state["tick"].copy(),
            "elapsed_ms": self.state["elapsed_ms"].copy(),
            "round_done": round_done,
            "warmdown_pending": warmdown_pending,
            "match_done": match_done,
            "terminated": terminated,
            "truncated": truncated,
            "done": done,
            "needs_reset": self._needs_reset.copy(),
            "terminal_reason": terminal_reason,
            "terminal_reason_name": terminal_reason_name,
            "truncation_reason": self._truncation_reason_names(),
            "timeout": terminal_reason == vector_reset.TERMINAL_REASON_TIMEOUT_TRUNCATED,
            "draw": self.state["draw"].copy(),
            "winner": self.state["winner"].copy(),
            "round_winner": round_winner,
            "match_winner": match_winner,
            "winner_ids": self._winner_ids(),
            "round_winner_ids": self._ids_from_winner_array(
                round_winner,
                active=round_done,
            ),
            "match_winner_ids": self._ids_from_winner_array(
                match_winner,
                active=match_done,
            ),
            "loser_ids": self._loser_ids(),
            "max_score": self.max_score,
            "max_score_by_row": self.state["max_score"].copy(),
        }

    def _metadata_state_array(
        self,
        name: str,
        fallback: np.ndarray,
        *,
        dtype: Any,
    ) -> np.ndarray:
        if name not in self.state:
            return fallback.copy()
        value = np.asarray(self.state[name], dtype=dtype)
        if value.shape != (self.batch_size,):
            return fallback.copy()
        return value.copy()

    def _bonus_public_info(self) -> dict[str, Any]:
        mode_by_row = np.full(self.batch_size, "disabled", dtype=object)
        mode_by_row[self._seeded_bonus_enabled] = "seeded"
        mode_by_row[self._natural_bonus_spawn_enabled] = "natural_spawn"
        active_count = np.zeros(self.batch_size, dtype=np.int32)
        stack_count = np.zeros((self.batch_size, self.player_count), dtype=np.int16)
        game_stack_count = np.zeros(self.batch_size, dtype=np.int16)
        bonus_active = None
        bonus_type = None
        if "bonus_active" in self.state:
            active_count = self.state["bonus_count"].copy()
            stack_count = self.state["bonus_stack_count"][
                :,
                : self.player_count,
            ].copy()
            if "bonus_game_stack_count" in self.state:
                game_stack_count = self.state["bonus_game_stack_count"].copy()
            bonus_active = self.state["bonus_active"].copy()
            bonus_type = self.state["bonus_type"].copy()
        natural_enabled = bool(self._natural_bonus_spawn_enabled.any())
        seeded_enabled = bool(self._seeded_bonus_enabled.any())
        return {
            "policy_id": (
                NATURAL_BONUS_SUPPORT_POLICY_ID
                if natural_enabled
                else SEEDED_BONUS_SUPPORT_POLICY_ID
            ),
            "mode": (
                "natural_spawn" if natural_enabled else ("seeded" if seeded_enabled else "disabled")
            ),
            "public_env_contract_id": PUBLIC_ENV_CONTRACT_ID,
            "env_impl_id": ENV_IMPL_ID,
            "env_version_policy_id": ENV_VERSION_POLICY_ID,
            "mode_by_row": mode_by_row,
            "enabled_by_row": (
                self._natural_bonus_spawn_enabled | self._seeded_bonus_enabled
            ).copy(),
            "seeded_enabled_by_row": self._seeded_bonus_enabled.copy(),
            "natural_enabled_by_row": self._natural_bonus_spawn_enabled.copy(),
            "claim": (
                NATURAL_BONUS_SUPPORT_CLAIM if natural_enabled else SEEDED_BONUS_SUPPORT_CLAIM
            ),
            "natural_bonus_spawn": natural_enabled,
            "natural_bonus_spawn_default": self.natural_bonus_spawn_default,
            "natural_bonus_policy_id": NATURAL_BONUS_SUPPORT_POLICY_ID,
            "natural_bonus_random_label_policy_id": (NATURAL_BONUS_RANDOM_LABEL_POLICY_ID),
            "source_bonus_poping_time_ms": SOURCE_BONUS_POPING_TIME_MS,
            "supported_seeded_bonus_types": SEEDED_BONUS_TYPE_NAMES,
            "source_default_natural_bonus_types": SOURCE_DEFAULT_NATURAL_BONUS_TYPE_NAMES,
            "supported_natural_bonus_types": NATURAL_BONUS_TYPE_NAMES,
            "supported_natural_bonus_effect_types": NATURAL_BONUS_EFFECT_TYPE_NAMES,
            "enabled_natural_bonus_type_codes": self.natural_bonus_type_codes.copy(),
            "unsupported_natural_bonus_types": NATURAL_BONUS_UNSUPPORTED_TYPE_NAMES,
            "unsupported_natural_bonus_effects": (NATURAL_BONUS_UNSUPPORTED_EFFECT_TYPE_NAMES),
            "source_default_gap_claims": _source_default_gap_claims(natural_enabled),
            "active_count": active_count,
            "stack_count": stack_count,
            "game_stack_count": game_stack_count,
            "bonus_active": bonus_active,
            "bonus_type": bonus_type,
        }

    def _reset_provenance_info(self) -> dict[str, Any]:
        return {
            "schema_id": RESET_PROVENANCE_POLICY_ID,
            "reset_seed": self.state["reset_seed"].copy(),
            "reset_source": self.state["reset_source"].copy(),
            "random_tape_source": self._random_tape_source.copy(),
            "random_tape_cursor": self.state["random_tape_cursor"].copy(),
            "random_tape_draw_count": self.state["random_tape_draw_count"].copy(),
            "random_tape_length": self.state["random_tape_length"].copy(),
            "rng_impl_id": self._rng_impl_id.copy(),
            "rng_history_ref": self._rng_impl_id.copy(),
            "source_fixture_ref": self._source_fixture_ref.copy(),
            "natural_multiplayer_reset_claim_by_row": (
                self._natural_multiplayer_reset_claim.copy()
            ),
            "seed_alone_replay_complete": False,
        }

    def _row_index(self, row: int) -> int:
        row_int = _nonnegative_int(row, "row")
        if row_int >= self.batch_size:
            raise VectorMultiplayerEnvError("row must be within batch_size")
        return row_int

    def _public_reset_policy(
        self,
        mask: np.ndarray,
        reset_info: dict[str, Any],
        *,
        reset_api: str,
        reset_source: int,
    ) -> dict[str, Any]:
        snapshot = reset_info.get("terminal_transition_snapshot")
        if isinstance(snapshot, dict):
            snapshot_rows = np.asarray(snapshot.get("final_rows", []), dtype=np.int32)
            snapshot_mask = np.asarray(snapshot.get("final_mask", mask), dtype=bool)
            snapshot_schema = snapshot.get("schema")
        else:
            snapshot_rows = np.asarray([], dtype=np.int32)
            snapshot_mask = np.zeros(self.batch_size, dtype=bool)
            snapshot_schema = None

        return {
            "schema_id": PUBLIC_RESET_POLICY_SCHEMA_ID,
            "api": reset_api,
            "rows": np.flatnonzero(mask).astype(np.int32),
            "row_mask": mask.copy(),
            "selected_rows_only": not bool(mask.all()),
            "hidden_autoreset": False,
            "explicit_autoreset_done_rows": reset_api == "autoreset_done_rows",
            "reset_source": np.asarray(reset_source, dtype=np.int16).item(),
            "reset_source_policy": "manual_or_explicit_masked_public_reset/v0",
            "pre_reset_snapshot_schema": snapshot_schema,
            "pre_reset_snapshot_rows": snapshot_rows.copy(),
            "pre_reset_snapshot_row_mask": snapshot_mask.copy(),
            "previous_step_final_metadata": self._previous_step_final_metadata(mask),
        }

    def _previous_step_final_metadata(self, reset_mask: np.ndarray) -> dict[str, Any]:
        empty_mask = np.zeros(self.batch_size, dtype=bool)
        if self.last_step_info is None:
            return {
                "source": "last_step_info",
                "available": False,
                "rows": np.asarray([], dtype=np.int32),
                "row_mask": empty_mask,
                "overlaps_reset_rows": False,
                "not_mutated_by_reset": True,
            }

        final_mask_value = self.last_step_info.get("final_observation_row_mask")
        final_mask = np.asarray(final_mask_value, dtype=bool)
        if final_mask.shape != (self.batch_size,):
            final_mask = empty_mask.copy()
        overlap_mask = final_mask & reset_mask
        return {
            "source": "last_step_info.final_*_row_policy",
            "available": bool(final_mask.any()),
            "rows": np.flatnonzero(overlap_mask).astype(np.int32),
            "row_mask": overlap_mask.copy(),
            "overlaps_reset_rows": bool(overlap_mask.any()),
            "final_observation_present": self.last_step_info.get("final_observation") is not None,
            "final_reward_present": self.last_step_info.get("final_reward_map") is not None,
            "not_mutated_by_reset": True,
        }

    def _winner_ids(self) -> list[list[int]]:
        winners: list[list[int]] = []
        for row in range(self.batch_size):
            if (
                bool(self.state["terminated"][row])
                and int(self.state["terminal_reason"][row])
                == vector_reset.TERMINAL_REASON_SURVIVOR_WIN
            ):
                winners.append([int(self.state["winner"][row])])
            else:
                winners.append([])
        return winners

    def _loser_ids(self) -> list[list[int]]:
        losers: list[list[int]] = []
        present = self.state["present"][:, : self.player_count]
        for row in range(self.batch_size):
            if not bool(self.state["terminated"][row]):
                losers.append([])
                continue
            winner = int(self.state["winner"][row])
            if winner < 0:
                losers.append([])
                continue
            losers.append(
                [int(player) for player in np.flatnonzero(present[row]) if int(player) != winner]
            )
        return losers

    def _ids_from_winner_array(
        self,
        winner: np.ndarray,
        *,
        active: np.ndarray,
    ) -> list[list[int]]:
        ids: list[list[int]] = []
        for row in range(self.batch_size):
            winner_index = int(winner[row])
            if bool(active[row]) and 0 <= winner_index < self.player_count:
                ids.append([winner_index])
            else:
                ids.append([])
        return ids

    def _truncation_reason_names(self) -> np.ndarray:
        reasons = np.full(self.batch_size, None, dtype=object)
        terminal_reason = self.state["terminal_reason"]
        reasons[terminal_reason == vector_reset.TERMINAL_REASON_TIMEOUT_TRUNCATED] = "timeout"
        reasons[terminal_reason == vector_reset.TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED] = (
            "event_overflow_truncated"
        )
        reasons[terminal_reason == vector_reset.TERMINAL_REASON_BODY_OVERFLOW_TRUNCATED] = (
            "capacity_truncated"
        )
        reasons[~self.state["truncated"]] = None
        return reasons

    def _reset_seed_array(
        self,
        seed: int | np.ndarray | None,
        mask: np.ndarray,
    ) -> np.ndarray:
        reset_seed = self.state["reset_seed"].copy()
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            return reset_seed

        if seed is None:
            reset_seed[rows] = self._seed_rng.integers(
                0,
                np.iinfo(np.int64).max,
                size=rows.size,
                dtype=np.uint64,
            )
            return reset_seed

        array = np.asarray(seed)
        if array.ndim == 0:
            scalar = _nonnegative_seed_scalar(array.item(), "seed")
            rng = np.random.default_rng(scalar)
            generated = rng.integers(
                0,
                np.iinfo(np.int64).max,
                size=self.batch_size,
                dtype=np.uint64,
            )
            reset_seed[rows] = generated[rows]
            return reset_seed

        if array.shape != (self.batch_size,):
            raise VectorMultiplayerEnvError("seed array must have shape [B]")
        if not np.issubdtype(array.dtype, np.integer):
            raise VectorMultiplayerEnvError("seed array must be integer")
        if bool((array < 0).any()):
            raise VectorMultiplayerEnvError("seed values must be non-negative")
        reset_seed[rows] = array.astype(np.uint64, copy=False)[rows]
        return reset_seed

    def _reset_source_array(self, mask: np.ndarray, source: int) -> np.ndarray:
        reset_source = self.state["reset_source"].copy()
        reset_source[mask] = np.asarray(source, dtype=np.int16)
        return reset_source

    def _row_mask(self, row_mask: np.ndarray | None) -> np.ndarray:
        if row_mask is None:
            return np.ones(self.batch_size, dtype=bool)
        mask = np.asarray(row_mask)
        if mask.dtype != bool or mask.shape != (self.batch_size,):
            raise VectorMultiplayerEnvError("row_mask must be a bool array with shape [B]")
        return mask.copy()

    def _leave_player_id_array(
        self,
        player_ids: int | np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        supplied = np.asarray(player_ids)
        if supplied.ndim == 0:
            if not np.issubdtype(supplied.dtype, np.integer):
                raise VectorMultiplayerEnvError(
                    "player_ids must be an integer scalar or integer array with shape [B]"
                )
            leave_player_by_row = np.full(self.batch_size, -1, dtype=np.int64)
            leave_player_by_row[mask] = int(supplied.item())
        else:
            if supplied.shape != (self.batch_size,):
                raise VectorMultiplayerEnvError(
                    "player_ids must be an integer scalar or integer array with shape [B]"
                )
            if not np.issubdtype(supplied.dtype, np.integer):
                raise VectorMultiplayerEnvError("player_ids must be integer")
            leave_player_by_row = np.full(self.batch_size, -1, dtype=np.int64)
            leave_player_by_row[mask] = supplied.astype(np.int64, copy=False)[mask]

        selected_players = leave_player_by_row[mask]
        if bool(((selected_players < 0) | (selected_players >= self.player_count)).any()):
            raise VectorMultiplayerEnvError(
                "player_ids must be zero-based public player ids in [0, player_count)"
            )
        return leave_player_by_row.astype(np.int16)

    def _stop_print_manager_for_leave(self, row: int, player: int) -> None:
        if not bool(self.state["print_manager_active"][row, player]):
            return
        self.state["printing"][row, player] = False
        vector_runtime.next_print_manager_random_distance(
            self.state,
            row=row,
            printing=False,
        )
        self.state["print_manager_active"][row, player] = False
        self.state["print_manager_distance"][row, player] = 0.0
        self.state["print_manager_last_pos"][row, player] = 0.0
        self.state["visible_trail_count"][row, player] = 0
        self.state["has_visible_trail_last"][row, player] = False
        self.state["visible_trail_last_pos"][row, player] = 0.0
        self.state["has_draw_cursor"][row, player] = False
        self.state["draw_cursor_pos"][row, player] = 0.0
        self.state["has_visual_trail_last"][row, player] = False
        self.state["visual_trail_last_pos"][row, player] = 0.0

    def _present_array(self, present: np.ndarray | None, mask: np.ndarray) -> np.ndarray:
        present_array = self.state["present"][:, : self.player_count].copy()
        if present is None:
            present_array[mask] = True
        else:
            supplied = np.asarray(present)
            if supplied.dtype != bool or supplied.shape != (
                self.batch_size,
                self.player_count,
            ):
                raise VectorMultiplayerEnvError("present must be a bool array with shape [B,P]")
            present_array[mask] = supplied[mask]
        selected_counts = present_array[mask].sum(axis=1)
        if bool((selected_counts < 2).any()):
            raise VectorMultiplayerEnvError(
                "selected reset rows require at least two present players"
            )
        return present_array

    def _require_reset(self) -> None:
        if not self._has_reset:
            raise RuntimeError("reset must be called before step")


def _make_state_arrays(
    batch_size: int,
    *,
    player_count: int,
    body_capacity: int,
    event_capacity: int,
    timer_capacity: int,
    random_tape_capacity: int,
    map_size: float,
    max_score: int,
    speed: float,
    angular_velocity_per_ms: float,
    radius: float,
    trail_latency: int,
) -> dict[str, np.ndarray]:
    return {
        "episode_id": np.zeros(batch_size, dtype=np.int64),
        "round_id": np.zeros(batch_size, dtype=np.int64),
        "episode_step": np.zeros(batch_size, dtype=np.int32),
        "env_active": np.ones(batch_size, dtype=bool),
        "reset_pending": np.zeros(batch_size, dtype=bool),
        "done": np.zeros(batch_size, dtype=bool),
        "terminated": np.zeros(batch_size, dtype=bool),
        "truncated": np.zeros(batch_size, dtype=bool),
        "terminal_reason": np.full(
            batch_size,
            vector_reset.TERMINAL_REASON_NONE,
            dtype=np.int16,
        ),
        "round_done": np.zeros(batch_size, dtype=bool),
        "warmdown_pending": np.zeros(batch_size, dtype=bool),
        "match_done": np.zeros(batch_size, dtype=bool),
        "round_winner": np.full(batch_size, -1, dtype=np.int16),
        "match_winner": np.full(batch_size, -1, dtype=np.int16),
        "reset_seed": np.zeros(batch_size, dtype=np.uint64),
        "reset_source": np.full(
            batch_size,
            vector_reset.RESET_SOURCE_MANUAL,
            dtype=np.int16,
        ),
        "tick": np.zeros(batch_size, dtype=np.int32),
        "elapsed_ms": np.zeros(batch_size, dtype=np.float64),
        "pos": np.zeros((batch_size, player_count, 2), dtype=np.float64),
        "prev_pos": np.zeros((batch_size, player_count, 2), dtype=np.float64),
        "heading": np.zeros((batch_size, player_count), dtype=np.float64),
        "alive": np.zeros((batch_size, player_count), dtype=bool),
        "present": np.ones((batch_size, player_count), dtype=bool),
        "map_size": np.full(batch_size, map_size, dtype=np.float64),
        "max_score": np.full(batch_size, max_score, dtype=np.int32),
        "random_tape_values": np.zeros(
            (batch_size, random_tape_capacity),
            dtype=np.float64,
        ),
        "random_tape_length": np.full(
            batch_size,
            random_tape_capacity,
            dtype=np.int32,
        ),
        "random_tape_cursor": np.zeros(batch_size, dtype=np.int32),
        "random_tape_exhausted": np.zeros(batch_size, dtype=bool),
        "random_tape_draw_count": np.zeros(batch_size, dtype=np.int32),
        "started": np.zeros(batch_size, dtype=bool),
        "in_round": np.zeros(batch_size, dtype=bool),
        "world_active": np.zeros(batch_size, dtype=bool),
        "world_body_count": np.zeros(batch_size, dtype=np.int32),
        "timer_active": np.zeros((batch_size, timer_capacity), dtype=bool),
        "timer_remaining_ms": np.zeros((batch_size, timer_capacity), dtype=np.float64),
        "timer_kind": np.zeros((batch_size, timer_capacity), dtype=np.int16),
        "timer_player": np.full(
            (batch_size, timer_capacity),
            vector_runtime.TIMER_PLAYER_NONE,
            dtype=np.int16,
        ),
        "timer_seq": np.zeros((batch_size, timer_capacity), dtype=np.int32),
        "timer_overflow": np.zeros(batch_size, dtype=bool),
        "score": np.zeros((batch_size, player_count), dtype=np.int32),
        "round_score": np.zeros((batch_size, player_count), dtype=np.int32),
        "bonus_catch_count_step": np.zeros(
            (batch_size, player_count),
            dtype=np.int16,
        ),
        "printing": np.zeros((batch_size, player_count), dtype=bool),
        "print_manager_active": np.zeros((batch_size, player_count), dtype=bool),
        "print_manager_distance": np.zeros((batch_size, player_count), dtype=np.float64),
        "print_manager_last_pos": np.zeros(
            (batch_size, player_count, 2),
            dtype=np.float64,
        ),
        "death_count": np.zeros(batch_size, dtype=np.int32),
        "death_player": np.full((batch_size, player_count), -1, dtype=np.int16),
        "death_cause": np.full(
            (batch_size, player_count),
            vector_runtime.DEATH_CAUSE_NONE,
            dtype=np.int16,
        ),
        "death_hit_owner": np.full((batch_size, player_count), -1, dtype=np.int16),
        "overflow": np.zeros(batch_size, dtype=bool),
        "borderless": np.zeros(batch_size, dtype=bool),
        "radius": np.full((batch_size, player_count), radius, dtype=np.float64),
        "speed": np.full((batch_size, player_count), speed, dtype=np.float64),
        "base_speed": np.full((batch_size, player_count), speed, dtype=np.float64),
        "inverse": np.zeros((batch_size, player_count), dtype=bool),
        "base_inverse": np.zeros((batch_size, player_count), dtype=bool),
        "angular_velocity_per_ms": np.full(
            (batch_size, player_count),
            angular_velocity_per_ms,
            dtype=np.float64,
        ),
        "live_body_num": np.zeros((batch_size, player_count), dtype=np.int32),
        "trail_latency": np.full(
            (batch_size, player_count),
            trail_latency,
            dtype=np.int32,
        ),
        "death_tick": np.full((batch_size, player_count), -1, dtype=np.int32),
        "draw": np.zeros(batch_size, dtype=bool),
        "winner": np.full(batch_size, -1, dtype=np.int16),
        "body_overflow": np.zeros(batch_size, dtype=bool),
        "visible_trail_count": np.zeros((batch_size, player_count), dtype=np.int32),
        "has_visible_trail_last": np.zeros((batch_size, player_count), dtype=bool),
        "visible_trail_last_pos": np.zeros(
            (batch_size, player_count, 2),
            dtype=np.float64,
        ),
        "has_draw_cursor": np.zeros((batch_size, player_count), dtype=bool),
        "draw_cursor_pos": np.zeros((batch_size, player_count, 2), dtype=np.float64),
        "visual_trail_active": np.zeros((batch_size, body_capacity), dtype=bool),
        "visual_trail_pos": np.zeros((batch_size, body_capacity, 2), dtype=np.float64),
        "visual_trail_radius": np.zeros((batch_size, body_capacity), dtype=np.float64),
        "visual_trail_owner": np.full((batch_size, body_capacity), -1, dtype=np.int16),
        "visual_trail_break_before": np.zeros((batch_size, body_capacity), dtype=bool),
        "visual_trail_write_cursor": np.zeros(batch_size, dtype=np.int32),
        "visual_trail_overflow": np.zeros(batch_size, dtype=bool),
        "has_visual_trail_last": np.zeros((batch_size, player_count), dtype=bool),
        "visual_trail_last_pos": np.zeros(
            (batch_size, player_count, 2),
            dtype=np.float64,
        ),
        "body_active": np.zeros((batch_size, body_capacity), dtype=bool),
        "body_pos": np.zeros((batch_size, body_capacity, 2), dtype=np.float64),
        "body_radius": np.zeros((batch_size, body_capacity), dtype=np.float64),
        "body_owner": np.full((batch_size, body_capacity), -1, dtype=np.int16),
        "body_num": np.full((batch_size, body_capacity), -1, dtype=np.int32),
        "body_insert_tick": np.full((batch_size, body_capacity), -1, dtype=np.int32),
        "body_insert_kind": np.full((batch_size, body_capacity), -1, dtype=np.int16),
        "body_break_before": np.zeros((batch_size, body_capacity), dtype=bool),
        "body_write_cursor": np.zeros(batch_size, dtype=np.int32),
        "body_count": np.zeros((batch_size, player_count), dtype=np.int32),
        "event_count": np.zeros(batch_size, dtype=np.int16),
        "event_mask": np.zeros((batch_size, event_capacity), dtype=bool),
        "event_type": np.zeros((batch_size, event_capacity), dtype=np.int16),
        "event_player": np.full((batch_size, event_capacity), -1, dtype=np.int16),
        "event_other": np.full((batch_size, event_capacity), -1, dtype=np.int16),
        "event_bool": np.full((batch_size, event_capacity), -1, dtype=np.int8),
        "event_value_i": np.zeros((batch_size, event_capacity, 2), dtype=np.int32),
        "event_value_f": np.zeros((batch_size, event_capacity, 2), dtype=np.float64),
        "event_overflow": np.zeros(batch_size, dtype=bool),
        "event_overflow_attempts": np.zeros(batch_size, dtype=np.int32),
    }


def _final_row_policy(name: str, terminal_mask: np.ndarray, *, present: bool) -> dict[str, Any]:
    rows = np.flatnonzero(terminal_mask).astype(np.int32)
    observation_schema_id = (
        DEBUG_METADATA_OBSERVATION_SCHEMA_ID if name == "final_observation" else None
    )
    return {
        "array": name,
        "present": bool(present),
        "rows": rows,
        "row_mask": terminal_mask.copy(),
        "terminal_rows_only": True,
        "nonterminal_rows_zero_filled": bool(present),
        "observation_schema_id": observation_schema_id,
        "source_claim": "debug_metadata_only_public_terminal_rows/v0",
    }


def _selected_row_labels(mask: np.ndarray, label: object, *, batch_size: int) -> np.ndarray:
    labels = np.full(batch_size, "unchanged", dtype=object)
    labels[np.asarray(mask, dtype=bool)] = label
    return labels


def _source_fixture_ref(value: str | None, *, has_fixture_random_tape: bool) -> str | None:
    if value is None:
        return None
    if not has_fixture_random_tape:
        raise VectorMultiplayerEnvError(
            "source_fixture_ref requires source_fixture_random_tape_values"
        )
    if not isinstance(value, str) or not value:
        raise VectorMultiplayerEnvError("source_fixture_ref must be a non-empty string")
    return value


def _source_fixture_random_tape_values(
    value: np.ndarray | None,
    *,
    batch_size: int,
    random_tape_capacity: int,
) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise VectorMultiplayerEnvError(
            "source_fixture_random_tape_values must be numeric"
        ) from exc
    if array.ndim != 2 or array.shape[0] != batch_size:
        raise VectorMultiplayerEnvError("source_fixture_random_tape_values must have shape [B,N]")
    if array.shape[1] < 1:
        raise VectorMultiplayerEnvError(
            "source_fixture_random_tape_values must include at least one value per row"
        )
    if array.shape[1] > random_tape_capacity:
        raise VectorMultiplayerEnvError(
            "source_fixture_random_tape_values length exceeds random_tape_capacity"
        )
    if not bool(np.isfinite(array).all()):
        raise VectorMultiplayerEnvError("source_fixture_random_tape_values must be finite")
    if bool(((array < 0.0) | (array >= 1.0)).any()):
        raise VectorMultiplayerEnvError("source_fixture_random_tape_values must be in [0, 1)")
    return array.copy()


def _natural_bonus_type_codes(value: tuple[str | int, ...] | np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.asarray(
            vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES,
            dtype=np.int16,
        )
    if isinstance(value, np.ndarray):
        raw_values = tuple(value.reshape(-1).tolist())
    else:
        raw_values = tuple(value)
    if not raw_values:
        raise VectorMultiplayerEnvError(
            "natural_bonus_type_codes must include at least one source default bonus type"
        )

    codes: list[int] = []
    for item in raw_values:
        if isinstance(item, str):
            if item not in NATURAL_BONUS_TYPE_CODES:
                raise VectorMultiplayerEnvError(
                    "natural bonus spawn type selection supports only source default "
                    "bonus types known to vector_runtime"
                )
            codes.append(NATURAL_BONUS_TYPE_CODES[item])
            continue
        code = _nonnegative_int(item, "natural_bonus_type_codes")
        if code not in vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES:
            raise VectorMultiplayerEnvError(
                "natural bonus spawn type selection supports only source default "
                "bonus type codes known to vector_runtime"
            )
        codes.append(code)
    return np.asarray(codes, dtype=np.int16)


def _source_fixture_warmup_advance_ms(
    value: float | np.ndarray | None,
    *,
    batch_size: int,
) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise VectorMultiplayerEnvError("source_fixture_warmup_advance_ms must be numeric") from exc
    if array.ndim == 0:
        array = np.full(batch_size, float(array), dtype=np.float64)
    if array.shape != (batch_size,):
        raise VectorMultiplayerEnvError(
            "source_fixture_warmup_advance_ms must be a scalar or shape [B]"
        )
    if not bool(np.isfinite(array).all()) or bool((array < 0.0).any()):
        raise VectorMultiplayerEnvError(
            "source_fixture_warmup_advance_ms must be finite and nonnegative"
        )
    return array.copy()


def _step_timer_advance_ms(
    value: float | np.ndarray | None,
    *,
    batch_size: int,
) -> np.ndarray:
    if value is None:
        return np.zeros(batch_size, dtype=np.float64)
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise VectorMultiplayerEnvError("timer_advance_ms must be numeric") from exc
    if array.ndim == 0:
        array = np.full(batch_size, float(array), dtype=np.float64)
    if array.shape != (batch_size,):
        raise VectorMultiplayerEnvError("timer_advance_ms must be a scalar or shape [B]")
    if not bool(np.isfinite(array).all()) or bool((array < 0.0).any()):
        raise VectorMultiplayerEnvError("timer_advance_ms must be finite and nonnegative")
    return array.copy()


def _warmdown_frame_elapsed_ms(
    value: float | np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise VectorMultiplayerEnvError("warmdown frame elapsed_ms must be numeric") from exc
    if array.ndim == 0:
        array = np.full(batch_size, float(array), dtype=np.float64)
    if array.shape != (batch_size,):
        raise VectorMultiplayerEnvError("warmdown frame elapsed_ms must be a scalar or shape [B]")
    if not bool(np.isfinite(array).all()) or bool((array < 0.0).any()):
        raise VectorMultiplayerEnvError("warmdown frame elapsed_ms must be finite and nonnegative")
    return array.copy()


def _seeded_bonus_type_code(value: str | int) -> int:
    if isinstance(value, str):
        if value not in SEEDED_BONUS_TYPE_CODES:
            raise VectorMultiplayerEnvError(
                "seeded bonus_type must be a public runtime-backed bonus effect"
            )
        return int(SEEDED_BONUS_TYPE_CODES[value])
    code = _nonnegative_int(value, "bonus_type")
    if code not in SEEDED_BONUS_TYPE_CODES.values():
        raise VectorMultiplayerEnvError(
            "seeded bonus_type must be a public runtime-backed bonus effect"
        )
    return code


def _seeded_bonus_type_name(value: int) -> str:
    for name, code in SEEDED_BONUS_TYPE_CODES.items():
        if int(code) == int(value):
            return name
    raise VectorMultiplayerEnvError("unsupported seeded bonus type code")


def _player_count(value: int) -> int:
    if value not in SUPPORTED_PLAYER_COUNTS:
        raise VectorMultiplayerEnvError("player_count must be 2, 3, or 4")
    return int(value)


def _player_ids(value: tuple[str, ...] | None, *, player_count: int) -> tuple[str, ...]:
    if value is None:
        return tuple(f"player_{player}" for player in range(player_count))
    if not isinstance(value, tuple) or len(value) != player_count:
        raise VectorMultiplayerEnvError("player_ids must match player_count")
    if not all(isinstance(player_id, str) for player_id in value):
        raise VectorMultiplayerEnvError("player_ids must contain strings")
    if len(set(value)) != player_count:
        raise VectorMultiplayerEnvError("player_ids must be unique")
    return value


def _death_immunity_player_ids(
    value: tuple[int, ...] | np.ndarray | None,
    *,
    player_count: int,
) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.int16)
    ids = np.asarray(value, dtype=np.int16).reshape(-1)
    if ids.size and (
        bool((ids < 0).any()) or bool((ids >= int(player_count)).any())
    ):
        raise VectorMultiplayerEnvError(
            "death_immunity_player_ids must contain valid player ids"
        )
    if np.unique(ids).size != ids.size:
        raise VectorMultiplayerEnvError("death_immunity_player_ids must be unique")
    return ids


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise VectorMultiplayerEnvError(f"{name} must be a positive integer")
    return int(value)


def _bool_flag(value: bool, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise VectorMultiplayerEnvError(f"{name} must be a bool")
    return bool(value)


def _optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _nonnegative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise VectorMultiplayerEnvError(f"{name} must be a nonnegative integer")
    return int(value)


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise VectorMultiplayerEnvError(f"{name} must be positive and finite")
    return result


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise VectorMultiplayerEnvError(f"{name} must be finite")
    return result


def _nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise VectorMultiplayerEnvError(f"{name} must be finite and nonnegative")
    return result


def _nonnegative_seed_scalar(value: object, name: str) -> int:
    array = np.asarray(value)
    if array.dtype == np.dtype(bool) or array.dtype.kind not in ("i", "u"):
        raise VectorMultiplayerEnvError(f"{name} must be an integer")
    scalar = int(array.item())
    if scalar < 0:
        raise VectorMultiplayerEnvError(f"{name} must be non-negative")
    return scalar


__all__ = [
    "DEBUG_METADATA_OBSERVATION_SCHEMA_ID",
    "VectorMultiplayerBatch",
    "VectorMultiplayerEnvError",
    "VectorMultiplayerEnv",
]
