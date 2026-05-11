"""First trainer observation builder for ``curvyzero_egocentric_rays/v0``.

This is intentionally a narrow, pure Python/NumPy helper. It consumes the
current ``EnvState`` plus config geometry and returns the pinned trainer shapes.
The ray caster uses the toy/grid occupancy state available today; it is not a
source-fidelity claim for browser pixels, full trail gap semantics, bonuses, or
future vector body arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np

from curvyzero.env.config import CurvyTronConfig
from curvyzero.env.state import EnvState
from curvyzero.env.trainer_contract import ACTION_NAMES
from curvyzero.env.trainer_contract import LIGHTZERO_FLAT_OBSERVATION_SHAPE
from curvyzero.env.trainer_contract import OBSERVATION_SCHEMA_HASH
from curvyzero.env.trainer_contract import OBSERVATION_SCHEMA_ID
from curvyzero.env.trainer_contract import RAY_ANGLES_DEGREES
from curvyzero.env.trainer_contract import RAY_CHANNEL_NAMES
from curvyzero.env.trainer_contract import REWARD_SCHEMA_HASH
from curvyzero.env.trainer_contract import REWARD_SCHEMA_ID
from curvyzero.env.trainer_contract import SCALAR_NAMES
from curvyzero.env.trainer_contract import STRUCTURED_OBSERVATION_SHAPE
from curvyzero.env.trainer_contract import legal_action_mask


_EPSILON = 1e-9
_RAY_RADIANS = np.radians(np.asarray(RAY_ANGLES_DEGREES, dtype=np.float64))
_RAY_COS = np.cos(_RAY_RADIANS)
_RAY_SIN = np.sin(_RAY_RADIANS)


@dataclass(frozen=True, slots=True)
class TrainerObservation:
    """Trainer-facing observation bundle for one ego player."""

    rays: np.ndarray
    scalars: np.ndarray
    observation: np.ndarray
    action_mask: np.ndarray
    lightzero_action_mask: np.ndarray
    to_play: int
    reward: np.float32
    reward_info: dict[str, Any]

    def lightzero_payload(self) -> dict[str, Any]:
        """Return the LightZero-style observation dict with copied arrays."""

        return {
            "observation": self.observation.copy(),
            "action_mask": self.lightzero_action_mask.copy(),
            "to_play": self.to_play,
        }


@dataclass(frozen=True, slots=True)
class TrainerObservationBatch1v1:
    """Row-stacked trainer bundle for both ego players in one 1v1 state."""

    player_ids: tuple[str, str]
    rays: np.ndarray
    scalars: np.ndarray
    observation: np.ndarray
    action_mask: np.ndarray
    lightzero_action_mask: np.ndarray
    to_play: np.ndarray
    rewards: np.ndarray
    reward_info: dict[str, dict[str, Any]]
    reward_map: dict[str, float]
    final_reward_map: dict[str, float] | None
    done: bool
    terminated: bool
    truncated: bool

    def lightzero_payload(self, ego_player_id: str | int) -> dict[str, Any]:
        """Return a copied LightZero-style observation dict for one row."""

        row = _player_index(ego_player_id, self.player_ids)
        return {
            "observation": self.observation[row].copy(),
            "action_mask": self.lightzero_action_mask[row].copy(),
            "to_play": int(self.to_play[row]),
        }

    def lightzero_payloads(self) -> dict[str, dict[str, Any]]:
        """Return copied LightZero-style observation dicts keyed by player id."""

        return {player_id: self.lightzero_payload(player_id) for player_id in self.player_ids}


def observe_egocentric_rays_v0(
    state: EnvState,
    config: CurvyTronConfig,
    ego_player_id: str | int,
    *,
    player_ids: Sequence[str] | None = None,
    active: bool = True,
    needs_reset: bool = False,
    allow_straight: bool | None = None,
    borderless: bool = False,
    profile_timers: dict[str, float] | None = None,
) -> TrainerObservation:
    """Build the pinned v0 trainer observation for one 1v1 ego player.

    The helper is deterministic and pure: it reads the supplied state arrays but
    never mutates them. It uses current grid occupancy for trail channels and
    current player positions for opponent-head distance. Future source-faithful
    vector body/trail arrays should replace only the ray-hit internals while
    preserving this public shape contract.
    """

    started = _profile_started(profile_timers)
    validated = _validated_inputs(state, config, player_ids)
    _profile_add(profile_timers, "scalar_validation_sec", started)
    ids = validated["player_ids"]
    ego_idx = _player_index(ego_player_id, ids)
    opponent_idx = 1 - ego_idx
    arena_diagonal = _arena_diagonal(config)

    started = _profile_started(profile_timers)
    rays = _cast_rays(
        validated["positions"],
        validated["headings"],
        validated["occupancy"],
        config,
        ego_idx=ego_idx,
        opponent_idx=opponent_idx,
        arena_diagonal=arena_diagonal,
        borderless=borderless,
    )
    _profile_add(profile_timers, "ray_cast_sec", started)
    started = _profile_started(profile_timers)
    scalars = _scalars(
        state,
        config,
        validated["positions"],
        validated["headings"],
        validated["alive"],
        ego_idx=ego_idx,
        opponent_idx=opponent_idx,
        arena_diagonal=arena_diagonal,
    )
    _profile_add(profile_timers, "scalar_pack_sec", started)
    started = _profile_started(profile_timers)
    observation = np.concatenate((rays.reshape(-1), scalars)).astype(np.float32, copy=False)
    if observation.shape != LIGHTZERO_FLAT_OBSERVATION_SHAPE:
        raise RuntimeError(
            f"packed observation shape {observation.shape} does not match "
            f"{LIGHTZERO_FLAT_OBSERVATION_SHAPE}"
        )
    _profile_add(profile_timers, "flat_observation_pack_sec", started)

    started = _profile_started(profile_timers)
    reward, reward_info = sparse_round_outcome_reward_v0(
        state,
        config,
        ego_player_id,
        player_ids=ids,
    )
    _profile_add(profile_timers, "reward_sec", started)
    started = _profile_started(profile_timers)
    if allow_straight is None:
        allow_straight = bool(config.allow_straight_action)
    row_active = (
        bool(active)
        and not bool(needs_reset)
        and bool(validated["alive"][ego_idx])
        and not bool(reward_info["done"])
    )
    action_mask = np.asarray(
        legal_action_mask(active=row_active, allow_straight=bool(allow_straight)),
        dtype=np.bool_,
    )
    _profile_add(profile_timers, "action_mask_sec", started)
    started = _profile_started(profile_timers)
    result = TrainerObservation(
        rays=rays,
        scalars=scalars,
        observation=observation.copy(),
        action_mask=action_mask.copy(),
        lightzero_action_mask=action_mask.astype(np.int8, copy=True),
        to_play=-1,
        reward=reward,
        reward_info=reward_info,
    )
    _profile_add(profile_timers, "scalar_result_copy_sec", started)
    return result


def observe_1v1_egocentric_rays_v0(
    state: EnvState,
    config: CurvyTronConfig,
    *,
    player_ids: Sequence[str] | None = None,
    active: bool = True,
    needs_reset: bool = False,
    allow_straight: bool | None = None,
    borderless: bool = False,
    profile_timers: dict[str, float] | None = None,
) -> TrainerObservationBatch1v1:
    """Build row-stacked v0 observations/rewards for both players in one 1v1 state.

    This pins the actor-loop row order and terminal reward map while sharing
    validation and simple derived data across both ego rows. It deliberately
    does not claim source body/trail fidelity beyond the underlying scalar ray
    helper.
    """

    started = _profile_started(profile_timers)
    validated = _validated_inputs(state, config, player_ids)
    _profile_add(profile_timers, "batch_validation_sec", started)
    ids = validated["player_ids"]

    started = _profile_started(profile_timers)
    arena_diagonal = _arena_diagonal(config)
    ray_context = _ray_cast_context(validated["positions"], validated["occupancy"])
    rays = np.empty(
        (2, len(RAY_ANGLES_DEGREES), len(RAY_CHANNEL_NAMES)),
        dtype=np.float32,
    )
    for ego_idx in range(2):
        rays[ego_idx] = _cast_rays(
            validated["positions"],
            validated["headings"],
            validated["occupancy"],
            config,
            ego_idx=ego_idx,
            opponent_idx=1 - ego_idx,
            arena_diagonal=arena_diagonal,
            borderless=borderless,
            ray_context=ray_context,
        )
    _profile_add(profile_timers, "ray_cast_sec", started)

    started = _profile_started(profile_timers)
    scalars = np.empty((2, len(SCALAR_NAMES)), dtype=np.float32)
    for ego_idx in range(2):
        scalars[ego_idx] = _scalars(
            state,
            config,
            validated["positions"],
            validated["headings"],
            validated["alive"],
            ego_idx=ego_idx,
            opponent_idx=1 - ego_idx,
            arena_diagonal=arena_diagonal,
        )
    _profile_add(profile_timers, "scalar_pack_sec", started)

    started = _profile_started(profile_timers)
    observation = np.empty((2,) + LIGHTZERO_FLAT_OBSERVATION_SHAPE, dtype=np.float32)
    ray_value_count = len(RAY_ANGLES_DEGREES) * len(RAY_CHANNEL_NAMES)
    observation[:, :ray_value_count] = rays.reshape(2, ray_value_count)
    observation[:, ray_value_count:] = scalars
    expected_observation_shape = (2,) + LIGHTZERO_FLAT_OBSERVATION_SHAPE
    if observation.shape != expected_observation_shape:
        raise RuntimeError(
            f"batched observation shape {observation.shape} does not match "
            f"{expected_observation_shape}"
        )
    _profile_add(profile_timers, "flat_observation_pack_sec", started)

    started = _profile_started(profile_timers)
    rewards, reward_infos = _sparse_round_outcome_rewards_1v1_v0(
        state,
        config,
        validated,
    )
    done = _common_reward_info_flag(reward_infos, "done")
    terminated = _common_reward_info_flag(reward_infos, "terminated")
    truncated = _common_reward_info_flag(reward_infos, "truncated")
    reward_map = {player_id: float(rewards[row]) for row, player_id in enumerate(ids)}
    _profile_add(profile_timers, "reward_sec", started)

    started = _profile_started(profile_timers)
    if allow_straight is None:
        allow_straight = bool(config.allow_straight_action)
    active_action_mask = np.asarray(
        legal_action_mask(active=True, allow_straight=bool(allow_straight)),
        dtype=np.bool_,
    )
    inactive_action_mask = np.asarray(
        legal_action_mask(active=False, allow_straight=bool(allow_straight)),
        dtype=np.bool_,
    )
    row_active = bool(active) and not bool(needs_reset) and not done
    action_mask = np.empty((2, len(ACTION_NAMES)), dtype=np.bool_)
    for ego_idx in range(2):
        action_mask[ego_idx] = (
            active_action_mask
            if row_active and bool(validated["alive"][ego_idx])
            else inactive_action_mask
        )
    lightzero_action_mask = action_mask.astype(np.int8, copy=True)
    _profile_add(profile_timers, "action_mask_sec", started)

    to_play = np.full(2, -1, dtype=np.int64)

    started = _profile_started(profile_timers)
    result = TrainerObservationBatch1v1(
        player_ids=ids,
        rays=rays.copy(),
        scalars=scalars.copy(),
        observation=observation.copy(),
        action_mask=action_mask.copy(),
        lightzero_action_mask=lightzero_action_mask.copy(),
        to_play=to_play.copy(),
        rewards=rewards.copy(),
        reward_info={
            player_id: dict(reward_info)
            for player_id, reward_info in zip(ids, reward_infos, strict=True)
        },
        reward_map=dict(reward_map),
        final_reward_map=dict(reward_map) if done else None,
        done=done,
        terminated=terminated,
        truncated=truncated,
    )
    _profile_add(profile_timers, "batch_result_copy_sec", started)
    return result


def sparse_round_outcome_reward_v0(
    state: EnvState,
    config: CurvyTronConfig,
    ego_player_id: str | int,
    *,
    player_ids: Sequence[str] | None = None,
) -> tuple[np.float32, dict[str, Any]]:
    """Return the sparse round-outcome reward and compact terminal metadata."""

    validated = _validated_inputs(state, config, player_ids)
    ids = validated["player_ids"]
    ego_idx = _player_index(ego_player_id, ids)
    rewards, reward_infos = _sparse_round_outcome_rewards_1v1_v0(
        state,
        config,
        validated,
    )
    return np.float32(rewards[ego_idx]), reward_infos[ego_idx]


def _sparse_round_outcome_rewards_1v1_v0(
    state: EnvState,
    config: CurvyTronConfig,
    validated: dict[str, Any],
) -> tuple[np.ndarray, tuple[dict[str, Any], dict[str, Any]]]:
    ids = validated["player_ids"]
    alive = validated["alive"]
    alive_count = int(alive.sum())
    terminated = alive_count <= 1
    truncated = bool(config.max_ticks > 0 and int(state.tick) >= int(config.max_ticks))
    done = terminated or truncated

    terminal_reason = "none"
    winner_ids: tuple[str, ...] = ()
    loser_ids: tuple[str, ...] = ()
    draw = False
    rewards = np.zeros(2, dtype=np.float32)

    if terminated and alive_count == 1:
        winner_idx = int(np.flatnonzero(alive)[0])
        terminal_reason = "survivor_win"
        winner_ids = (ids[winner_idx],)
        loser_ids = tuple(
            player_id for index, player_id in enumerate(ids) if index != winner_idx
        )
        rewards[winner_idx] = np.float32(1.0)
        rewards[1 - winner_idx] = np.float32(-1.0)
    elif terminated:
        terminal_reason = "all_dead_draw"
        draw = True
    elif truncated:
        terminal_reason = "timeout"

    reward_infos = tuple(
        {
            "reward_schema_id": REWARD_SCHEMA_ID,
            "reward_schema_hash": REWARD_SCHEMA_HASH,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
            "ego_player_id": ids[ego_idx],
            "terminal_reason": terminal_reason,
            "winner_ids": winner_ids,
            "loser_ids": loser_ids,
            "draw": draw,
            "timeout": truncated,
            "truncation_reason": "max_ticks" if truncated else None,
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
        }
        for ego_idx in range(2)
    )
    return rewards, reward_infos


def _common_reward_info_flag(reward_infos: Sequence[dict[str, Any]], key: str) -> bool:
    first = bool(reward_infos[0][key])
    for reward_info in reward_infos[1:]:
        if bool(reward_info[key]) != first:
            raise RuntimeError(f"inconsistent reward_info[{key!r}] across 1v1 ego rows")
    return first


def _profile_started(profile_timers: dict[str, float] | None) -> float:
    return time.perf_counter() if profile_timers is not None else 0.0


def _profile_add(
    profile_timers: dict[str, float] | None,
    key: str,
    started: float,
) -> None:
    if profile_timers is None:
        return
    profile_timers[key] = profile_timers.get(key, 0.0) + time.perf_counter() - started


def flatten_structured_observation_v0(rays: np.ndarray, scalars: np.ndarray) -> np.ndarray:
    """Pack structured v0 rays/scalars into the canonical flat float32[106]."""

    rays_array = np.asarray(rays, dtype=np.float32)
    scalars_array = np.asarray(scalars, dtype=np.float32)
    if rays_array.shape != STRUCTURED_OBSERVATION_SHAPE["rays"]:
        raise ValueError(
            f"rays must have shape {STRUCTURED_OBSERVATION_SHAPE['rays']}, "
            f"got {rays_array.shape}"
        )
    if scalars_array.shape != STRUCTURED_OBSERVATION_SHAPE["scalars"]:
        raise ValueError(
            f"scalars must have shape {STRUCTURED_OBSERVATION_SHAPE['scalars']}, "
            f"got {scalars_array.shape}"
        )
    return np.concatenate((rays_array.reshape(-1), scalars_array)).astype(np.float32)


def _validated_inputs(
    state: EnvState,
    config: CurvyTronConfig,
    player_ids: Sequence[str] | None,
) -> dict[str, Any]:
    positions = np.asarray(state.positions, dtype=np.float64)
    headings = np.asarray(state.headings, dtype=np.float64)
    alive = np.asarray(state.alive, dtype=np.bool_)
    occupancy = np.asarray(state.occupancy)

    if positions.shape != (2, 2):
        raise ValueError(
            f"{OBSERVATION_SCHEMA_ID} currently supports exactly 2 players; "
            f"positions shape must be (2, 2), got {positions.shape}"
        )
    if headings.shape != (2,):
        raise ValueError(f"headings shape must be (2,), got {headings.shape}")
    if alive.shape != (2,):
        raise ValueError(f"alive shape must be (2,), got {alive.shape}")
    expected_occupancy_shape = (int(config.height), int(config.width))
    if occupancy.shape != expected_occupancy_shape:
        raise ValueError(
            f"occupancy shape must be {expected_occupancy_shape}, got {occupancy.shape}"
        )
    if int(config.players) != 2:
        raise ValueError(f"{OBSERVATION_SCHEMA_ID} currently supports exactly 2 players")

    ids = tuple(player_ids or ("player_0", "player_1"))
    if len(ids) != 2:
        raise ValueError(f"player_ids must contain exactly 2 ids, got {len(ids)}")
    return {
        "positions": positions,
        "headings": headings,
        "alive": alive,
        "occupancy": occupancy,
        "player_ids": ids,
    }


def _player_index(ego_player_id: str | int, player_ids: Sequence[str]) -> int:
    if isinstance(ego_player_id, int):
        if 0 <= ego_player_id < len(player_ids):
            return int(ego_player_id)
        raise ValueError(f"unknown player index {ego_player_id!r}")
    try:
        return tuple(player_ids).index(ego_player_id)
    except ValueError as exc:
        raise ValueError(f"unknown player {ego_player_id!r}") from exc


def _arena_diagonal(config: CurvyTronConfig) -> float:
    diagonal = math.hypot(float(config.width), float(config.height))
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("arena diagonal must be positive and finite")
    return diagonal


def _cast_rays(
    positions: np.ndarray,
    headings: np.ndarray,
    occupancy: np.ndarray,
    config: CurvyTronConfig,
    *,
    ego_idx: int,
    opponent_idx: int,
    arena_diagonal: float,
    borderless: bool,
    ray_context: dict[str, Any] | None = None,
) -> np.ndarray:
    rays = np.ones((len(RAY_ANGLES_DEGREES), len(RAY_CHANNEL_NAMES)), dtype=np.float32)
    ego_position = positions[ego_idx]
    ego_heading = float(headings[ego_idx])
    trail_radius = max(0.0, float(config.trail_radius))
    if ray_context is None:
        ray_context = _ray_cast_context(positions, occupancy)
    has_occupied_cells = bool(ray_context["has_occupied_cells"])
    if has_occupied_cells:
        occupancy_cells = ray_context["occupancy_cells"]
        head_cells = ray_context["head_cells"]
        own_centers = _cell_centers(
            occupancy_cells.get(ego_idx + 1, ()),
            excluded_cell=head_cells[ego_idx],
        )
        opponent_centers = _cell_centers(
            occupancy_cells.get(opponent_idx + 1, ()),
            excluded_cell=head_cells[opponent_idx],
        )
    else:
        own_centers = np.empty((0, 2), dtype=np.float64)
        opponent_centers = np.empty((0, 2), dtype=np.float64)

    ray_directions = _ray_directions(ego_heading)
    if has_occupied_cells:
        own_distances = _nearest_centers_hits(
            ego_position,
            ray_directions,
            own_centers,
            radius=trail_radius,
            arena_diagonal=arena_diagonal,
        )
        opponent_trail_distances = _nearest_centers_hits(
            ego_position,
            ray_directions,
            opponent_centers,
            radius=trail_radius,
            arena_diagonal=arena_diagonal,
        )
        rays[:, 1] = _normalized_hit_distances(own_distances, arena_diagonal)
        rays[:, 2] = _normalized_hit_distances(opponent_trail_distances, arena_diagonal)
    opponent_head_distances = _nearest_centers_hits(
        ego_position,
        ray_directions,
        positions[opponent_idx][None, :],
        radius=trail_radius,
        arena_diagonal=arena_diagonal,
    )
    rays[:, 3] = _normalized_hit_distances(opponent_head_distances, arena_diagonal)
    for ray_index, direction in enumerate(ray_directions):
        wall_hit = None if borderless else _wall_hit_distance(
            ego_position,
            direction,
            width=float(config.width),
            height=float(config.height),
        )
        rays[ray_index, 0] = _normalized_hit(wall_hit, arena_diagonal)

    return rays


def _ray_cast_context(positions: np.ndarray, occupancy: np.ndarray) -> dict[str, Any]:
    has_occupied_cells = bool(np.any(occupancy))
    return {
        "has_occupied_cells": has_occupied_cells,
        "occupancy_cells": (
            _occupancy_cells_by_owner(occupancy) if has_occupied_cells else {}
        ),
        "head_cells": tuple(_cell_for_position(position) for position in positions),
    }


def _scalars(
    state: EnvState,
    config: CurvyTronConfig,
    positions: np.ndarray,
    headings: np.ndarray,
    alive: np.ndarray,
    *,
    ego_idx: int,
    opponent_idx: int,
    arena_diagonal: float,
) -> np.ndarray:
    ego_heading = float(headings[ego_idx])
    opponent_heading = float(headings[opponent_idx])
    forward = np.array([math.cos(ego_heading), math.sin(ego_heading)], dtype=np.float64)
    left = np.array([math.sin(ego_heading), -math.cos(ego_heading)], dtype=np.float64)
    delta = positions[opponent_idx] - positions[ego_idx]
    tick_fraction = (
        float(np.clip(float(state.tick) / float(config.max_ticks), 0.0, 1.0))
        if config.max_ticks > 0
        else 0.0
    )
    relative_heading = opponent_heading - ego_heading
    values = np.array(
        [
            1.0 if bool(alive[ego_idx]) else 0.0,
            1.0 if bool(alive[opponent_idx]) else 0.0,
            tick_fraction,
            float(np.clip(float(np.dot(delta, forward)) / arena_diagonal, -1.0, 1.0)),
            float(np.clip(float(np.dot(delta, left)) / arena_diagonal, -1.0, 1.0)),
            math.sin(relative_heading),
            math.cos(relative_heading),
            abs(float(config.speed)) * max(1, int(config.action_repeat)) / arena_diagonal,
            abs(float(config.turn_rate_radians)) / math.pi,
            max(0.0, float(config.trail_radius)) / arena_diagonal,
        ],
        dtype=np.float32,
    )
    if values.shape != (len(SCALAR_NAMES),):
        raise RuntimeError("internal scalar shape mismatch")
    return values


def _ray_direction(ego_heading: float, ray_degrees: float) -> np.ndarray:
    ray_radians = math.radians(ray_degrees)
    global_angle = ego_heading - ray_radians
    return np.array([math.cos(global_angle), math.sin(global_angle)], dtype=np.float64)


def _ray_directions(ego_heading: float) -> np.ndarray:
    cos_heading = math.cos(ego_heading)
    sin_heading = math.sin(ego_heading)
    return np.column_stack(
        (
            cos_heading * _RAY_COS + sin_heading * _RAY_SIN,
            sin_heading * _RAY_COS - cos_heading * _RAY_SIN,
        )
    )


def _wall_hit_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    width: float,
    height: float,
) -> float | None:
    x, y = float(origin[0]), float(origin[1])
    if x < 0.0 or x >= width or y < 0.0 or y >= height:
        return 0.0

    candidates: list[float] = []
    dx, dy = float(direction[0]), float(direction[1])
    if abs(dx) > _EPSILON:
        candidates.append(((width if dx > 0.0 else 0.0) - x) / dx)
    if abs(dy) > _EPSILON:
        candidates.append(((height if dy > 0.0 else 0.0) - y) / dy)

    hits: list[float] = []
    for distance in candidates:
        if distance < 0.0:
            continue
        hit = origin + direction * distance
        if -_EPSILON <= hit[0] <= width + _EPSILON and -_EPSILON <= hit[1] <= height + _EPSILON:
            hits.append(float(distance))
    return min(hits) if hits else None


def _occupancy_cells_by_owner(occupancy: np.ndarray) -> dict[int, tuple[tuple[int, int], ...]]:
    cells: dict[int, list[tuple[int, int]]] = {}
    ys, xs = np.nonzero(occupancy)
    for y, x in zip(ys, xs, strict=True):
        owner = int(occupancy[y, x])
        if owner <= 0:
            continue
        cells.setdefault(owner, []).append((int(x), int(y)))
    return {owner: tuple(owner_cells) for owner, owner_cells in cells.items()}


def _cell_for_position(position: np.ndarray) -> tuple[int, int]:
    return (int(round(float(position[0]))), int(round(float(position[1]))))


def _cell_centers(
    cells: Sequence[tuple[int, int]],
    *,
    excluded_cell: tuple[int, int],
) -> np.ndarray:
    if not cells:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(
        [
            (float(cell[0]), float(cell[1]))
            for cell in cells
            if cell != excluded_cell
        ],
        dtype=np.float64,
    ).reshape((-1, 2))


def _nearest_centers_hits(
    origin: np.ndarray,
    directions: np.ndarray,
    centers: np.ndarray,
    *,
    radius: float,
    arena_diagonal: float,
) -> np.ndarray:
    no_hits = np.full(directions.shape[0], np.inf, dtype=np.float64)
    if centers.size == 0:
        return no_hits

    offsets = origin[None, :] - centers
    radius = max(0.0, float(radius))
    c_values = np.einsum("ij,ij->i", offsets, offsets) - radius * radius
    if np.any(c_values <= 0.0):
        return np.zeros(directions.shape[0], dtype=np.float64)

    b_values = 2.0 * (directions @ offsets.T)
    discriminants = b_values * b_values - 4.0 * c_values
    valid = discriminants >= 0.0
    if not bool(np.any(valid)):
        return no_hits

    sqrt_discriminants = np.sqrt(np.maximum(0.0, discriminants))
    first = (-b_values - sqrt_discriminants) / 2.0
    second = (-b_values + sqrt_discriminants) / 2.0
    distance = np.maximum(0.0, first)
    hit_mask = valid & (second >= 0.0) & (distance <= arena_diagonal)
    if not bool(np.any(hit_mask)):
        return no_hits
    masked_distance = np.where(hit_mask, distance, np.inf)
    return np.min(masked_distance, axis=1)


def _nearest_circle_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    *,
    radius: float,
    max_distance: float,
) -> float | None:
    offset = origin - center
    radius = max(0.0, float(radius))
    c_value = float(np.dot(offset, offset) - radius * radius)
    if c_value <= 0.0:
        return 0.0
    b_value = 2.0 * float(np.dot(direction, offset))
    discriminant = b_value * b_value - 4.0 * c_value
    if discriminant < 0.0:
        return None
    sqrt_discriminant = math.sqrt(max(0.0, discriminant))
    first = (-b_value - sqrt_discriminant) / 2.0
    second = (-b_value + sqrt_discriminant) / 2.0
    if second < 0.0:
        return None
    distance = max(0.0, first)
    if distance > max_distance:
        return None
    return float(distance)


def _normalized_hit(distance: float | None, arena_diagonal: float) -> np.float32:
    if distance is None or distance > arena_diagonal:
        return np.float32(1.0)
    return np.float32(np.clip(float(distance) / arena_diagonal, 0.0, 1.0))


def _normalized_hit_distances(distances: np.ndarray, arena_diagonal: float) -> np.ndarray:
    normalized = np.ones(distances.shape, dtype=np.float32)
    valid = np.isfinite(distances) & (distances <= arena_diagonal)
    normalized[valid] = np.clip(distances[valid] / arena_diagonal, 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )
    return normalized


__all__ = [
    "ACTION_NAMES",
    "TrainerObservationBatch1v1",
    "TrainerObservation",
    "flatten_structured_observation_v0",
    "observe_1v1_egocentric_rays_v0",
    "observe_egocentric_rays_v0",
    "sparse_round_outcome_reward_v0",
]
