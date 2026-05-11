"""Narrow public vector trainer environment for source-shaped 1v1/no-bonus rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from curvyzero.env.config import CurvyTronReferenceDefaults
from curvyzero.env.trainer_contract import ACTION_ID_TO_SOURCE_MOVE
from curvyzero.env.trainer_contract import ACTION_NAMES
from curvyzero.env.trainer_contract import NATIVE_CONTROL_MODEL_ID
from curvyzero.env.trainer_contract import TRAINER_CONTROL_WRAPPER_ID
from curvyzero.env import vector_autoreset
from curvyzero.env import vector_lifecycle
from curvyzero.env import vector_reset
from curvyzero.env import vector_runtime
from curvyzero.env import vector_trainer_observation


PLAYER_COUNT = 2
ACTION_COUNT = len(ACTION_NAMES)
DEFAULT_BODY_CAPACITY = 4096
DEFAULT_EVENT_CAPACITY = 64
DEFAULT_TIMER_CAPACITY = 4
DEFAULT_RANDOM_TAPE_CAPACITY = 4096
DEFAULT_PLAYER_IDS = ("player_0", "player_1")


class VectorTrainerEnvError(ValueError):
    """Raised when the public vector trainer env API is used invalidly."""


@dataclass(frozen=True, slots=True)
class VectorTrainerBatch1v1NoBonus:
    """Public batch returned by reset/step for B source-shaped 1v1 rows."""

    observation: np.ndarray
    action_mask: np.ndarray
    lightzero_action_mask: np.ndarray
    to_play: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    final_observation: np.ndarray | None
    final_reward: np.ndarray | None
    info: dict[str, Any]


class VectorTrainerEnv1v1NoBonus:
    """Small public 1v1/no-bonus trainer env over the vector runtime arrays."""

    def __init__(
        self,
        batch_size: int = 1,
        *,
        autoreset: bool = True,
        seed: int | None = None,
        decision_ms: float = 300.0,
        max_ticks: int = 2_000,
        map_size: float | None = None,
        body_capacity: int = DEFAULT_BODY_CAPACITY,
        event_capacity: int = DEFAULT_EVENT_CAPACITY,
        timer_capacity: int = DEFAULT_TIMER_CAPACITY,
        random_tape_capacity: int = DEFAULT_RANDOM_TAPE_CAPACITY,
        event_mode: str = vector_runtime.EVENT_MODE_NONE,
        player_ids: tuple[str, str] = DEFAULT_PLAYER_IDS,
        max_warmup_timer_callbacks: int | None = None,
    ) -> None:
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.autoreset = bool(autoreset)
        self.decision_ms = _positive_finite(decision_ms, "decision_ms")
        self.max_ticks = _nonnegative_int(max_ticks, "max_ticks")
        self.body_capacity = _positive_int(body_capacity, "body_capacity")
        self.event_capacity = _positive_int(event_capacity, "event_capacity")
        self.timer_capacity = _positive_int(timer_capacity, "timer_capacity")
        self.random_tape_capacity = _positive_int(
            random_tape_capacity,
            "random_tape_capacity",
        )
        if event_mode not in vector_runtime.EVENT_MODES:
            raise VectorTrainerEnvError("event_mode must be 'debug-event' or 'no-event'")
        self.event_mode = event_mode
        self.player_ids = _player_ids(player_ids)
        self.max_warmup_timer_callbacks = _optional_positive_int(
            max_warmup_timer_callbacks,
            "max_warmup_timer_callbacks",
        )

        reference = CurvyTronReferenceDefaults()
        self.map_size = (
            float(reference.arena_size_for_players(PLAYER_COUNT))
            if map_size is None
            else _positive_finite(map_size, "map_size")
        )
        self.speed = float(reference.avatar_velocity_units_per_s)
        self.angular_velocity_per_ms = float(reference.angular_velocity_radians_per_ms)
        self.radius = float(reference.avatar_radius)
        self.trail_latency = int(reference.trail_latency_points)
        self.first_warmup_ms = float(reference.round_warmup_ms)

        self.reset_template = _make_state_arrays(
            self.batch_size,
            body_capacity=self.body_capacity,
            event_capacity=self.event_capacity,
            timer_capacity=self.timer_capacity,
            random_tape_capacity=self.random_tape_capacity,
            map_size=self.map_size,
            speed=self.speed,
            angular_velocity_per_ms=self.angular_velocity_per_ms,
            radius=self.radius,
            trail_latency=self.trail_latency,
        )
        self.state = {name: array.copy() for name, array in self.reset_template.items()}
        self._seed_rng = np.random.default_rng(seed)
        self._has_reset = False
        self._needs_reset = np.zeros(self.batch_size, dtype=bool)
        self.last_reset_info: dict[str, Any] | None = None
        self.last_step_info: dict[str, Any] | None = None

    def reset(
        self,
        seed: int | np.ndarray | None = None,
        *,
        row_mask: np.ndarray | None = None,
        source_fixture_random_tape_values: np.ndarray | None = None,
        source_fixture_new_round_time_ms: float | None = None,
        source_fixture_warmup_advance_ms: float | np.ndarray | None = None,
    ) -> VectorTrainerBatch1v1NoBonus:
        """Reset all rows, or only selected rows, through reset/spawn/warmup."""

        mask = self._row_mask(row_mask)
        fixture_policy_enabled = (
            source_fixture_random_tape_values is not None
            or source_fixture_new_round_time_ms is not None
            or source_fixture_warmup_advance_ms is not None
        )
        fixture_random_tape_values = _source_fixture_random_tape_values(
            source_fixture_random_tape_values,
            batch_size=self.batch_size,
            random_tape_capacity=self.random_tape_capacity,
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
        if row_mask is None and seed is not None and np.asarray(seed).ndim == 0:
            self._seed_rng = np.random.default_rng(
                _nonnegative_seed_scalar(np.asarray(seed).item(), "seed"),
            )
            seed_input = None
        reset_seed = self._reset_seed_array(seed_input, mask)
        self._prepare_reset_template_rows(
            mask,
            reset_seed,
            source_fixture_random_tape_values=fixture_random_tape_values,
        )
        reset_info = self._reset_spawn_warmup_rows(
            mask,
            reset_seed=reset_seed,
            reset_source=self._reset_source_array(
                mask,
                vector_reset.RESET_SOURCE_MANUAL,
            ),
            first_warmup_ms=fixture_new_round_time_ms,
        )
        warmup_info = self._advance_warmup_rows(
            mask,
            first_warmup_ms=fixture_new_round_time_ms,
            advance_ms=fixture_warmup_advance_ms,
        )
        self._needs_reset[mask] = False
        self._has_reset = True

        reward = np.zeros((self.batch_size, PLAYER_COUNT), dtype=np.float32)
        info: dict[str, Any] = {
            "reset_info": reset_info,
            "warmup_info": warmup_info,
            "reset_rows": np.flatnonzero(mask).astype(np.int32),
            "episode_id": self.state["episode_id"].copy(),
            "reset_seed": reset_seed.copy(),
            "reset_source": self.state["reset_source"].copy(),
            "returned_episode_id": self.state["episode_id"].copy(),
            "returned_reset_seed": self.state["reset_seed"].copy(),
            "returned_reset_source": self.state["reset_source"].copy(),
            "autoreset": self.autoreset,
            **self._control_wrapper_info(),
        }
        if fixture_policy_enabled:
            info["source_fixture_reset_policy"] = {
                "enabled": True,
                "scope": "source_fixture_reset_parity_only",
                "rows": np.flatnonzero(mask).astype(np.int32),
                "row_mask": mask.copy(),
                "random_tape_values_supplied": fixture_random_tape_values is not None,
                "random_tape_length": (
                    None
                    if fixture_random_tape_values is None
                    else np.full(
                        self.batch_size,
                        fixture_random_tape_values.shape[1],
                        dtype=np.int32,
                    )
                ),
                "new_round_time_ms": fixture_new_round_time_ms,
                "advance_timers_ms": np.asarray(
                    warmup_info["advance_ms"],
                    dtype=np.float64,
                ).copy(),
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

    def step(self, actions: np.ndarray) -> VectorTrainerBatch1v1NoBonus:
        """Step one source-shaped decision for every row with int[B,2] actions."""

        self._require_reset()
        if bool(self._needs_reset.any()):
            rows = np.flatnonzero(self._needs_reset).astype(np.int32)
            raise RuntimeError(
                "reset must be called before stepping rows that ended; "
                f"pending rows={rows.tolist()}"
            )

        source_moves = self._source_moves(actions)
        pre_active = ~self.state["done"].copy()
        step_input = vector_runtime.VectorStepInput(
            state=self.state,
            step_ms=np.full(self.batch_size, self.decision_ms, dtype=np.float64),
            source_moves=source_moves,
            player_count=PLAYER_COUNT,
            print_manager_mode=np.full(
                self.batch_size,
                "natural_toggle",
                dtype=object,
            ),
            event_mode=self.event_mode,
            timer_advance_ms=np.zeros(self.batch_size, dtype=np.float64),
        )
        counters = vector_runtime.step_many(step_input)
        self.state["episode_step"][pre_active] += 1
        self.state["elapsed_ms"][pre_active] += self.decision_ms
        self._mark_overflow_truncations(pre_active)
        self._mark_timeout_truncations(pre_active)

        transition_mask = self.state["done"].copy()
        if bool(transition_mask.any()):
            self.state["in_round"][transition_mask] = False
        done = self.state["done"].copy()
        terminated = self.state["terminated"].copy()
        truncated = self.state["truncated"].copy()
        terminal_reason = self.state["terminal_reason"].copy()
        winner = self.state["winner"].copy()
        draw = self.state["draw"].copy()
        death_count = self.state["death_count"].copy()
        death_player = self.state["death_player"].copy()
        death_cause = self.state["death_cause"].copy()
        death_hit_owner = self.state["death_hit_owner"].copy()
        episode_id = self.state["episode_id"].copy()
        transition_reset_seed = self.state["reset_seed"].copy()
        transition_reset_source = self.state["reset_source"].copy()
        terminal_rows = np.flatnonzero(transition_mask).astype(np.int32)
        reward = np.zeros((self.batch_size, PLAYER_COUNT), dtype=np.float32)
        final_observation: np.ndarray | None = None
        final_reward: np.ndarray | None = None
        autoreset_plan = None
        autoreset_reset_info = None
        autoreset_warmup_info = None

        if bool(transition_mask.any()):
            transition = (
                vector_trainer_observation.build_final_trainer_transition_1v1_no_bonus_rows(
                    self.state,
                    transition_mask,
                    player_ids=self.player_ids,
                    decision_ms=self.decision_ms,
                    max_ticks=self.max_ticks,
                )
            )
            final_observation = transition["final_observation"].copy()
            final_reward = transition["final_reward_map"].copy()
            reward[transition_mask] = final_reward[transition_mask]

            if self.autoreset:
                autoreset_reset_seed = self._reset_seed_array(None, transition_mask)
                autoreset_reset_source = self._reset_source_array(
                    transition_mask,
                    vector_reset.RESET_SOURCE_AUTORESET,
                )
                autoreset_plan = vector_autoreset.plan_autoreset_rows(
                    self.state,
                    final_observation=final_observation,
                    final_reward_map=final_reward,
                    reset_seed=autoreset_reset_seed,
                    reset_source=autoreset_reset_source,
                    autoreset_mask=transition_mask,
                )
                self._prepare_reset_template_rows(transition_mask, autoreset_reset_seed)
                autoreset_reset_info = self._reset_spawn_warmup_rows(
                    transition_mask,
                    reset_seed=autoreset_reset_seed,
                    reset_source=autoreset_reset_source,
                    first_warmup_ms=self.first_warmup_ms,
                )
                autoreset_warmup_info = self._advance_warmup_rows(
                    transition_mask,
                    first_warmup_ms=self.first_warmup_ms,
                )
            else:
                self._needs_reset |= transition_mask

        batch = self._batch(
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            final_observation=final_observation,
            final_reward=final_reward,
            info={
                "step_counters": counters,
                "joint_action": np.asarray(actions).copy(),
                "source_moves": source_moves.copy(),
                "print_manager_mode": "natural_toggle",
                "autoreset": self.autoreset,
                "autoreset_plan": autoreset_plan,
                "autoreset_reset_info": autoreset_reset_info,
                "autoreset_warmup_info": autoreset_warmup_info,
                "terminal_rows": terminal_rows.copy(),
                "terminal_reason": terminal_reason.copy(),
                "winner": winner.copy(),
                "draw": draw.copy(),
                "death_count": death_count.copy(),
                "death_player": death_player.copy(),
                "death_cause": death_cause.copy(),
                "death_cause_name": vector_runtime.death_cause_name_array(death_cause),
                "death_hit_owner": death_hit_owner.copy(),
                "terminated": terminated.copy(),
                "truncated": truncated.copy(),
                "episode_id": episode_id.copy(),
                "reset_seed": transition_reset_seed.copy(),
                "reset_source": transition_reset_source.copy(),
                "returned_episode_id": self.state["episode_id"].copy(),
                "returned_reset_seed": self.state["reset_seed"].copy(),
                "returned_reset_source": self.state["reset_source"].copy(),
                "returned_observation_source": _returned_observation_source(
                    transition_mask,
                    autoreset=self.autoreset,
                ),
                "final_observation_rows": terminal_rows.copy(),
                "final_observation_row_mask": transition_mask.copy(),
                "final_reward_rows": terminal_rows.copy(),
                "final_reward_row_mask": transition_mask.copy(),
                "final_observation_policy": _final_row_policy(
                    "final_observation",
                    transition_mask,
                    present=final_observation is not None,
                ),
                "final_reward_policy": _final_row_policy(
                    "final_reward",
                    transition_mask,
                    present=final_reward is not None,
                ),
                "needs_reset": self._needs_reset.copy(),
                **self._control_wrapper_info(),
            },
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
        info = vector_lifecycle.reset_spawn_warmup_1v1_no_bonus_rows(
            self.state,
            self.reset_template,
            mask,
            reset_seed=reset_seed,
            reset_source=reset_source,
            first_warmup_ms=first_warmup_ms,
            snapshot_array_names=(
                "done",
                "terminated",
                "truncated",
                "terminal_reason",
                "death_count",
                "death_player",
                "death_cause",
                "death_hit_owner",
                "pos",
                "alive",
                "winner",
                "draw",
            ),
        )
        if not bool(info["can_compose"]):
            raise VectorTrainerEnvError("state and reset_template cannot compose")
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
        return vector_runtime.advance_warmup_1v1_no_bonus_timers(
            self.state,
            selected_advance_ms,
            max_timer_callbacks=self._warmup_timer_callback_cap(mask),
        )

    def _warmup_timer_callback_cap(self, mask: np.ndarray) -> int:
        if self.max_warmup_timer_callbacks is not None:
            return self.max_warmup_timer_callbacks
        # Each reset row normally fires game:start plus one delayed PrintManager
        # start per player. Keep headroom for callback-order details while
        # preserving the runtime guard against infinite timer loops.
        return max(16, int(mask.sum()) * (PLAYER_COUNT + 1) * 4)

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
        self.state["terminal_reason"][
            body_mask
        ] = vector_reset.TERMINAL_REASON_BODY_OVERFLOW_TRUNCATED
        self.state["terminal_reason"][
            event_mask
        ] = vector_reset.TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED

    def _mark_timeout_truncations(self, pre_active: np.ndarray) -> None:
        timeout_mask = pre_active & ~self.state["done"] & (self.state["tick"] >= self.max_ticks)
        if not bool(timeout_mask.any()):
            return
        self.state["done"][timeout_mask] = True
        self.state["terminated"][timeout_mask] = False
        self.state["truncated"][timeout_mask] = True
        self.state["reset_pending"][timeout_mask] = True
        self.state["terminal_reason"][timeout_mask] = vector_reset.TERMINAL_REASON_TIMEOUT_TRUNCATED

    def _batch(
        self,
        *,
        reward: np.ndarray,
        done: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        final_observation: np.ndarray | None,
        final_reward: np.ndarray | None,
        info: dict[str, Any],
    ) -> VectorTrainerBatch1v1NoBonus:
        observation, action_mask, lightzero_action_mask, to_play = self._observe_arrays()
        return VectorTrainerBatch1v1NoBonus(
            observation=observation,
            action_mask=action_mask,
            lightzero_action_mask=lightzero_action_mask,
            to_play=to_play,
            reward=np.asarray(reward, dtype=np.float32).copy(),
            done=np.asarray(done, dtype=bool).copy(),
            terminated=np.asarray(terminated, dtype=bool).copy(),
            truncated=np.asarray(truncated, dtype=bool).copy(),
            final_observation=(
                None
                if final_observation is None
                else np.asarray(final_observation, dtype=np.float32).copy()
            ),
            final_reward=(
                None if final_reward is None else np.asarray(final_reward, dtype=np.float32).copy()
            ),
            info=dict(info),
        )

    def _observe_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return vector_trainer_observation.observe_vector_1v1_egocentric_rays_batch_arrays_v0(
            self.state,
            player_ids=self.player_ids,
            decision_ms=self.decision_ms,
            max_ticks=self.max_ticks,
        )

    def _prepare_reset_template_rows(
        self,
        mask: np.ndarray,
        reset_seed: np.ndarray,
        *,
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
        template["reset_seed"][mask] = reset_seed[mask]
        template["reset_source"][mask] = vector_reset.RESET_SOURCE_MANUAL
        template["tick"][mask] = 0
        template["elapsed_ms"][mask] = 0.0
        template["pos"][mask, ...] = 0.0
        template["prev_pos"][mask, ...] = 0.0
        template["heading"][mask, ...] = 0.0
        template["alive"][mask, ...] = False
        template["present"][mask, ...] = True
        template["map_size"][mask] = self.map_size
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
        template["angular_velocity_per_ms"][mask, ...] = self.angular_velocity_per_ms
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
        template["body_active"][mask, ...] = False
        template["body_pos"][mask, ...] = 0.0
        template["body_radius"][mask, ...] = 0.0
        template["body_owner"][mask, ...] = -1
        template["body_num"][mask, ...] = -1
        template["body_insert_tick"][mask, ...] = -1
        template["body_insert_kind"][mask, ...] = -1
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
        template["random_tape_length"][mask] = self.random_tape_capacity
        template["random_tape_cursor"][mask] = 0
        template["random_tape_exhausted"][mask] = False
        template["random_tape_draw_count"][mask] = 0

        for row in rows:
            rng = np.random.default_rng(int(reset_seed[int(row)]))
            template["random_tape_values"][int(row)] = rng.random(
                self.random_tape_capacity,
                dtype=np.float64,
            )
        if source_fixture_random_tape_values is not None and rows.size:
            fixture_length = source_fixture_random_tape_values.shape[1]
            template["random_tape_values"][rows, :] = 0.0
            template["random_tape_values"][rows, :fixture_length] = (
                source_fixture_random_tape_values[rows, :]
            )
            template["random_tape_length"][rows] = fixture_length

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
            raise VectorTrainerEnvError("seed array must have shape [B]")
        if not np.issubdtype(array.dtype, np.integer):
            raise VectorTrainerEnvError("seed array must be integer")
        if bool((array < 0).any()):
            raise VectorTrainerEnvError("seed values must be non-negative")
        reset_seed[rows] = array.astype(np.uint64, copy=False)[rows]
        return reset_seed

    def _reset_source_array(self, mask: np.ndarray, source: int) -> np.ndarray:
        reset_source = self.state["reset_source"].copy()
        reset_source[mask] = np.asarray(source, dtype=np.int16)
        return reset_source

    def _source_moves(self, actions: np.ndarray) -> np.ndarray:
        action_array = np.asarray(actions)
        if action_array.shape != (self.batch_size, PLAYER_COUNT):
            raise VectorTrainerEnvError("actions must have shape [B,2]")
        if not np.issubdtype(action_array.dtype, np.integer):
            raise VectorTrainerEnvError("actions must be integer action ids")
        if bool(((action_array < 0) | (action_array >= ACTION_COUNT)).any()):
            raise VectorTrainerEnvError("actions must be in left/straight/right id range")
        move_lookup = np.asarray(ACTION_ID_TO_SOURCE_MOVE, dtype=np.int8)
        return move_lookup[action_array.astype(np.int64, copy=False)]

    def _row_mask(self, row_mask: np.ndarray | None) -> np.ndarray:
        if row_mask is None:
            return np.ones(self.batch_size, dtype=bool)
        mask = np.asarray(row_mask)
        if mask.dtype != bool or mask.shape != (self.batch_size,):
            raise VectorTrainerEnvError("row_mask must be a bool array with shape [B]")
        return mask.copy()

    def _require_reset(self) -> None:
        if not self._has_reset:
            raise RuntimeError("reset must be called before step")

    def _control_wrapper_info(self) -> dict[str, Any]:
        return {
            "native_control_model_id": NATIVE_CONTROL_MODEL_ID,
            "trainer_control_wrapper_id": TRAINER_CONTROL_WRAPPER_ID,
            "decision_ms": self.decision_ms,
        }


def _returned_observation_source(terminal_mask: np.ndarray, *, autoreset: bool) -> np.ndarray:
    source = np.full(terminal_mask.shape, "live_state", dtype="<U16")
    if bool(terminal_mask.any()):
        source[terminal_mask] = "post_autoreset" if autoreset else "terminal_state"
    return source


def _final_row_policy(name: str, terminal_mask: np.ndarray, *, present: bool) -> dict[str, Any]:
    rows = np.flatnonzero(terminal_mask).astype(np.int32)
    return {
        "array": name,
        "present": bool(present),
        "rows": rows,
        "row_mask": terminal_mask.copy(),
        "terminal_rows_only": True,
        "nonterminal_rows_zero_filled": bool(present),
    }


def _make_state_arrays(
    batch_size: int,
    *,
    body_capacity: int,
    event_capacity: int,
    timer_capacity: int,
    random_tape_capacity: int,
    map_size: float,
    speed: float,
    angular_velocity_per_ms: float,
    radius: float,
    trail_latency: int,
) -> dict[str, np.ndarray]:
    return {
        "episode_id": np.zeros(batch_size, dtype=np.int64),
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
        "reset_seed": np.zeros(batch_size, dtype=np.uint64),
        "reset_source": np.full(
            batch_size,
            vector_reset.RESET_SOURCE_MANUAL,
            dtype=np.int16,
        ),
        "tick": np.zeros(batch_size, dtype=np.int32),
        "elapsed_ms": np.zeros(batch_size, dtype=np.float64),
        "pos": np.zeros((batch_size, PLAYER_COUNT, 2), dtype=np.float64),
        "prev_pos": np.zeros((batch_size, PLAYER_COUNT, 2), dtype=np.float64),
        "heading": np.zeros((batch_size, PLAYER_COUNT), dtype=np.float64),
        "alive": np.zeros((batch_size, PLAYER_COUNT), dtype=bool),
        "present": np.ones((batch_size, PLAYER_COUNT), dtype=bool),
        "map_size": np.full(batch_size, map_size, dtype=np.float64),
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
        "score": np.zeros((batch_size, PLAYER_COUNT), dtype=np.int32),
        "round_score": np.zeros((batch_size, PLAYER_COUNT), dtype=np.int32),
        "printing": np.zeros((batch_size, PLAYER_COUNT), dtype=bool),
        "print_manager_active": np.zeros((batch_size, PLAYER_COUNT), dtype=bool),
        "print_manager_distance": np.zeros((batch_size, PLAYER_COUNT), dtype=np.float64),
        "print_manager_last_pos": np.zeros(
            (batch_size, PLAYER_COUNT, 2),
            dtype=np.float64,
        ),
        "death_count": np.zeros(batch_size, dtype=np.int32),
        "death_player": np.full((batch_size, PLAYER_COUNT), -1, dtype=np.int16),
        "death_cause": np.full(
            (batch_size, PLAYER_COUNT),
            vector_runtime.DEATH_CAUSE_NONE,
            dtype=np.int16,
        ),
        "death_hit_owner": np.full((batch_size, PLAYER_COUNT), -1, dtype=np.int16),
        "overflow": np.zeros(batch_size, dtype=bool),
        "borderless": np.zeros(batch_size, dtype=bool),
        "radius": np.full((batch_size, PLAYER_COUNT), radius, dtype=np.float64),
        "speed": np.full((batch_size, PLAYER_COUNT), speed, dtype=np.float64),
        "angular_velocity_per_ms": np.full(
            (batch_size, PLAYER_COUNT),
            angular_velocity_per_ms,
            dtype=np.float64,
        ),
        "live_body_num": np.zeros((batch_size, PLAYER_COUNT), dtype=np.int32),
        "trail_latency": np.full((batch_size, PLAYER_COUNT), trail_latency, dtype=np.int32),
        "death_tick": np.full((batch_size, PLAYER_COUNT), -1, dtype=np.int32),
        "draw": np.zeros(batch_size, dtype=bool),
        "winner": np.full(batch_size, -1, dtype=np.int16),
        "body_overflow": np.zeros(batch_size, dtype=bool),
        "visible_trail_count": np.zeros((batch_size, PLAYER_COUNT), dtype=np.int32),
        "has_visible_trail_last": np.zeros((batch_size, PLAYER_COUNT), dtype=bool),
        "visible_trail_last_pos": np.zeros(
            (batch_size, PLAYER_COUNT, 2),
            dtype=np.float64,
        ),
        "has_draw_cursor": np.zeros((batch_size, PLAYER_COUNT), dtype=bool),
        "draw_cursor_pos": np.zeros((batch_size, PLAYER_COUNT, 2), dtype=np.float64),
        "body_active": np.zeros((batch_size, body_capacity), dtype=bool),
        "body_pos": np.zeros((batch_size, body_capacity, 2), dtype=np.float64),
        "body_radius": np.zeros((batch_size, body_capacity), dtype=np.float64),
        "body_owner": np.full((batch_size, body_capacity), -1, dtype=np.int16),
        "body_num": np.full((batch_size, body_capacity), -1, dtype=np.int32),
        "body_insert_tick": np.full((batch_size, body_capacity), -1, dtype=np.int32),
        "body_insert_kind": np.full((batch_size, body_capacity), -1, dtype=np.int16),
        "body_write_cursor": np.zeros(batch_size, dtype=np.int32),
        "body_count": np.zeros((batch_size, PLAYER_COUNT), dtype=np.int32),
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


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise VectorTrainerEnvError(f"{name} must be a positive integer")
    return int(value)


def _optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _nonnegative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise VectorTrainerEnvError(f"{name} must be a nonnegative integer")
    return int(value)


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise VectorTrainerEnvError(f"{name} must be positive and finite")
    return result


def _nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise VectorTrainerEnvError(f"{name} must be finite and nonnegative")
    return result


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
        raise VectorTrainerEnvError(
            "source_fixture_random_tape_values must be numeric"
        ) from exc
    if array.ndim != 2 or array.shape[0] != batch_size:
        raise VectorTrainerEnvError(
            "source_fixture_random_tape_values must have shape [B,N]"
        )
    if array.shape[1] < 1:
        raise VectorTrainerEnvError(
            "source_fixture_random_tape_values must include at least one value per row"
        )
    if array.shape[1] > random_tape_capacity:
        raise VectorTrainerEnvError(
            "source_fixture_random_tape_values length exceeds random_tape_capacity"
        )
    if not bool(np.isfinite(array).all()):
        raise VectorTrainerEnvError(
            "source_fixture_random_tape_values must be finite"
        )
    if bool(((array < 0.0) | (array >= 1.0)).any()):
        raise VectorTrainerEnvError(
            "source_fixture_random_tape_values must be in [0, 1)"
        )
    return array.copy()


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
        raise VectorTrainerEnvError(
            "source_fixture_warmup_advance_ms must be numeric"
        ) from exc
    if array.ndim == 0:
        array = np.full(batch_size, float(array), dtype=np.float64)
    if array.shape != (batch_size,):
        raise VectorTrainerEnvError(
            "source_fixture_warmup_advance_ms must be a scalar or shape [B]"
        )
    if not bool(np.isfinite(array).all()) or bool((array < 0.0).any()):
        raise VectorTrainerEnvError(
            "source_fixture_warmup_advance_ms must be finite and nonnegative"
        )
    return array.copy()


def _nonnegative_seed_scalar(value: object, name: str) -> int:
    array = np.asarray(value)
    if array.dtype == np.dtype(bool) or array.dtype.kind not in ("i", "u"):
        raise VectorTrainerEnvError(f"{name} must be an integer")
    scalar = int(array.item())
    if scalar < 0:
        raise VectorTrainerEnvError(f"{name} must be non-negative")
    return scalar


def _player_ids(value: tuple[str, str]) -> tuple[str, str]:
    if not isinstance(value, tuple) or len(value) != PLAYER_COUNT:
        raise VectorTrainerEnvError("player_ids must be a tuple of exactly two ids")
    if not all(isinstance(player_id, str) for player_id in value):
        raise VectorTrainerEnvError("player_ids must contain strings")
    if len(set(value)) != PLAYER_COUNT:
        raise VectorTrainerEnvError("player_ids must be unique")
    return value


__all__ = [
    "VectorTrainerBatch1v1NoBonus",
    "VectorTrainerEnv1v1NoBonus",
    "VectorTrainerEnvError",
]
