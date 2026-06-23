"""Trainer observations from the vector runtime state.

This module is the narrow 1v1/no-bonus bridge from source-ordered vector
runtime arrays to the pinned trainer observation/reward contract. It raycasts
against vector body circles directly; it does not build a fake occupancy grid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import time
from typing import Any

import numpy as np

from curvyzero.env.config import CurvyTronReferenceDefaults
from curvyzero.env import vector_reset
from curvyzero.env import vector_runtime
from curvyzero.env.trainer_contract import ACTION_NAMES
from curvyzero.env.trainer_contract import LIGHTZERO_FLAT_OBSERVATION_SHAPE
from curvyzero.env.trainer_contract import OBSERVATION_SCHEMA_HASH
from curvyzero.env.trainer_contract import OBSERVATION_SCHEMA_ID
from curvyzero.env.trainer_contract import RAY_ANGLES_DEGREES
from curvyzero.env.trainer_contract import RAY_CHANNEL_NAMES
from curvyzero.env.trainer_contract import REWARD_SCHEMA_HASH
from curvyzero.env.trainer_contract import REWARD_SCHEMA_ID
from curvyzero.env.trainer_contract import SCALAR_NAMES
from curvyzero.env.trainer_contract import legal_action_mask
from curvyzero.env.trainer_observation import TrainerObservationBatch1v1


VECTOR_TRAINER_TRANSITION_SCHEMA_ID = (
    "curvyzero_vector_trainer_transition_1v1_no_bonus/v0"
)
VECTOR_TRAINER_OBSERVATION_SURFACE = "vector_1v1_egocentric_rays_v0"
PLAYER_COUNT = 2
ACTION_COUNT = len(ACTION_NAMES)
_EPSILON = 1e-9
_RAY_RADIANS = np.radians(np.asarray(RAY_ANGLES_DEGREES, dtype=np.float64))
_RAY_COS = np.cos(_RAY_RADIANS)
_RAY_SIN = np.sin(_RAY_RADIANS)
_PLAYER_IDS = ("player_0", "player_1")
_DEFAULT_TRAIL_LATENCY = CurvyTronReferenceDefaults().trail_latency_points
_TRUNCATION_INFO_BY_TERMINAL_REASON = {
    vector_reset.TERMINAL_REASON_TIMEOUT_TRUNCATED: (
        "timeout",
        True,
        "max_ticks",
    ),
    vector_reset.TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED: (
        "event_overflow_truncated",
        False,
        "event_overflow",
    ),
    vector_reset.TERMINAL_REASON_BODY_OVERFLOW_TRUNCATED: (
        "body_overflow_truncated",
        False,
        "body_overflow",
    ),
}


class VectorTrainerObservationError(ValueError):
    """Raised when vector runtime arrays cannot satisfy the trainer contract."""


def observe_vector_1v1_egocentric_rays_v0(
    state: Mapping[str, np.ndarray],
    row: int,
    *,
    player_ids: Sequence[str] = _PLAYER_IDS,
    decision_ms: float,
    max_ticks: int,
    allow_straight: bool = True,
    profile_timers: dict[str, float] | None = None,
) -> TrainerObservationBatch1v1:
    """Build the pinned trainer batch for one 1v1 vector runtime row."""

    started = _profile_started(profile_timers)
    arrays = _validated_arrays(state)
    ids = _player_ids(player_ids)
    row_index = _row_index(row, arrays["pos"].shape[0])
    decision_ms_value = _positive_finite(decision_ms, "decision_ms")
    max_ticks_value = int(max_ticks)
    if max_ticks_value < 0:
        raise VectorTrainerObservationError("max_ticks must be non-negative")
    arena_diagonal = _arena_diagonal(float(arrays["map_size"][row_index]))
    _profile_add(profile_timers, "batch_validation_sec", started)

    started = _profile_started(profile_timers)
    done, terminated, truncated = _row_done_flags(state, row_index, arrays["alive"])
    rewards, reward_map, final_reward_map, terminal_infos = _row_rewards(
        state,
        row_index,
        player_ids=ids,
        done=done,
        terminated=terminated,
        truncated=truncated,
    )
    _profile_add(profile_timers, "reward_sec", started)

    started = _profile_started(profile_timers)
    rays = np.empty((PLAYER_COUNT, len(RAY_ANGLES_DEGREES), len(RAY_CHANNEL_NAMES)), dtype=np.float32)
    for ego_idx in range(PLAYER_COUNT):
        rays[ego_idx] = _cast_rays_for_player(
            arrays,
            row_index,
            ego_idx=ego_idx,
            opponent_idx=1 - ego_idx,
            arena_diagonal=arena_diagonal,
        )
    _profile_add(profile_timers, "ray_cast_sec", started)

    started = _profile_started(profile_timers)
    scalars = np.empty((PLAYER_COUNT, len(SCALAR_NAMES)), dtype=np.float32)
    for ego_idx in range(PLAYER_COUNT):
        scalars[ego_idx] = _scalars_for_player(
            arrays,
            row_index,
            ego_idx=ego_idx,
            opponent_idx=1 - ego_idx,
            arena_diagonal=arena_diagonal,
            decision_ms=decision_ms_value,
            max_ticks=max_ticks_value,
        )
    _profile_add(profile_timers, "scalar_pack_sec", started)

    started = _profile_started(profile_timers)
    observations = np.empty((PLAYER_COUNT,) + LIGHTZERO_FLAT_OBSERVATION_SHAPE, dtype=np.float32)
    ray_value_count = len(RAY_ANGLES_DEGREES) * len(RAY_CHANNEL_NAMES)
    observations[:, :ray_value_count] = rays.reshape(PLAYER_COUNT, ray_value_count)
    observations[:, ray_value_count:] = scalars
    _profile_add(profile_timers, "flat_observation_pack_sec", started)

    started = _profile_started(profile_timers)
    masks = np.empty((PLAYER_COUNT, ACTION_COUNT), dtype=np.bool_)
    reward_infos: dict[str, dict[str, Any]] = {}
    for ego_idx, player_id in enumerate(ids):
        active = bool(arrays["alive"][row_index, ego_idx]) and not done
        masks[ego_idx] = legal_action_mask(active=active, allow_straight=allow_straight)
        reward_infos[player_id] = {
            "reward_schema_id": REWARD_SCHEMA_ID,
            "reward_schema_hash": REWARD_SCHEMA_HASH,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
            "ego_player_id": player_id,
            "terminal_reason": terminal_infos["terminal_reason"],
            "winner_ids": terminal_infos["winner_ids"],
            "loser_ids": terminal_infos["loser_ids"],
            "draw": terminal_infos["draw"],
            "timeout": terminal_infos["timeout"],
            "truncation_reason": terminal_infos["truncation_reason"],
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
        }
    _profile_add(profile_timers, "action_mask_sec", started)

    started = _profile_started(profile_timers)
    result = TrainerObservationBatch1v1(
        player_ids=ids,
        rays=rays,
        scalars=scalars,
        observation=observations,
        action_mask=masks,
        lightzero_action_mask=masks.astype(np.int8, copy=True),
        to_play=np.full(PLAYER_COUNT, -1, dtype=np.int64),
        rewards=rewards.astype(np.float32, copy=True),
        reward_info=reward_infos,
        reward_map=reward_map,
        final_reward_map=final_reward_map,
        done=done,
        terminated=terminated,
        truncated=truncated,
    )
    _profile_add(profile_timers, "batch_result_copy_sec", started)
    return result


def observe_vector_1v1_egocentric_rays_batch_arrays_v0(
    state: Mapping[str, np.ndarray],
    *,
    player_ids: Sequence[str] = _PLAYER_IDS,
    decision_ms: float,
    max_ticks: int,
    allow_straight: bool = True,
    profile_timers: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build batch observation/mask arrays for the public vector trainer env.

    This keeps the scalar row ray kernel for parity, but hoists state validation
    and output allocation out of the per-row loop used by the public env step.
    """

    started = _profile_started(profile_timers)
    arrays = _validated_arrays(state)
    _player_ids(player_ids)
    decision_ms_value = _positive_finite(decision_ms, "decision_ms")
    max_ticks_value = int(max_ticks)
    if max_ticks_value < 0:
        raise VectorTrainerObservationError("max_ticks must be non-negative")
    batch_size = int(arrays["pos"].shape[0])
    done, _terminated, _truncated = _done_flag_arrays(state, arrays["alive"])
    arena_diagonal = np.asarray(
        [_arena_diagonal(float(map_size)) for map_size in arrays["map_size"]],
        dtype=np.float64,
    )
    _profile_add(profile_timers, "batch_validation_sec", started)

    started = _profile_started(profile_timers)
    rays = _cast_rays_batch_1v1(arrays, arena_diagonal)
    _profile_add(profile_timers, "ray_cast_sec", started)

    started = _profile_started(profile_timers)
    scalars = np.empty((batch_size, PLAYER_COUNT, len(SCALAR_NAMES)), dtype=np.float32)
    for row in range(batch_size):
        for ego_idx in range(PLAYER_COUNT):
            scalars[row, ego_idx] = _scalars_for_player(
                arrays,
                row,
                ego_idx=ego_idx,
                opponent_idx=1 - ego_idx,
                arena_diagonal=float(arena_diagonal[row]),
                decision_ms=decision_ms_value,
                max_ticks=max_ticks_value,
            )
    _profile_add(profile_timers, "scalar_pack_sec", started)

    started = _profile_started(profile_timers)
    observation = np.empty(
        (batch_size, PLAYER_COUNT, *LIGHTZERO_FLAT_OBSERVATION_SHAPE),
        dtype=np.float32,
    )
    ray_value_count = len(RAY_ANGLES_DEGREES) * len(RAY_CHANNEL_NAMES)
    observation[:, :, :ray_value_count] = rays.reshape(batch_size, PLAYER_COUNT, ray_value_count)
    observation[:, :, ray_value_count:] = scalars
    _profile_add(profile_timers, "flat_observation_pack_sec", started)

    started = _profile_started(profile_timers)
    action_mask = np.zeros((batch_size, PLAYER_COUNT, ACTION_COUNT), dtype=np.bool_)
    active = arrays["alive"] & ~done[:, None]
    if allow_straight:
        action_mask[active] = True
    else:
        action_mask[active, 0] = True
        action_mask[active, 2] = True
    lightzero_action_mask = action_mask.astype(np.int8, copy=True)
    to_play = np.full((batch_size, PLAYER_COUNT), -1, dtype=np.int64)
    _profile_add(profile_timers, "action_mask_sec", started)
    return observation, action_mask, lightzero_action_mask, to_play


def build_final_trainer_transition_1v1_no_bonus_rows(
    state: Mapping[str, np.ndarray],
    row_mask: np.ndarray,
    *,
    player_ids: Sequence[str] = _PLAYER_IDS,
    decision_ms: float,
    max_ticks: int,
    allow_straight: bool = True,
) -> dict[str, Any]:
    """Build full-batch final observation/reward arrays for selected rows."""

    arrays = _validated_arrays(state)
    mask = _bool_row_mask(row_mask, "row_mask", batch_size=arrays["pos"].shape[0])
    rows = np.flatnonzero(mask).astype(np.int32)
    final_observation = np.zeros(
        (arrays["pos"].shape[0], PLAYER_COUNT, *LIGHTZERO_FLAT_OBSERVATION_SHAPE),
        dtype=np.float32,
    )
    final_reward_map = np.zeros((arrays["pos"].shape[0], PLAYER_COUNT), dtype=np.float32)

    trainer_batches: list[TrainerObservationBatch1v1] = []
    for row in rows:
        batch = observe_vector_1v1_egocentric_rays_v0(
            state,
            int(row),
            player_ids=player_ids,
            decision_ms=decision_ms,
            max_ticks=max_ticks,
            allow_straight=allow_straight,
        )
        final_observation[int(row)] = batch.observation
        if batch.final_reward_map is not None:
            final_reward_map[int(row)] = np.asarray(
                [batch.final_reward_map[player] for player in batch.player_ids],
                dtype=np.float32,
            )
        trainer_batches.append(batch)

    return {
        "schema": VECTOR_TRAINER_TRANSITION_SCHEMA_ID,
        "surface": VECTOR_TRAINER_OBSERVATION_SURFACE,
        "row_mask": mask.copy(),
        "rows": rows.copy(),
        "final_observation": final_observation,
        "final_reward_map": final_reward_map,
        "trainer_batch_rows": rows.copy(),
        "trainer_batches": tuple(trainer_batches),
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
        "reward_schema_id": REWARD_SCHEMA_ID,
        "reward_schema_hash": REWARD_SCHEMA_HASH,
    }


def _validated_arrays(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    pos = _numeric_array(state, "pos", ndim=3)
    if pos.shape[1] != PLAYER_COUNT or pos.shape[2] != 2:
        raise VectorTrainerObservationError("pos must have shape [B,2,2]")
    batch_size = pos.shape[0]
    player_shape = (batch_size, pos.shape[1])
    body_active = _bool_array(state, "body_active", ndim=2, batch_size=batch_size)
    body_shape = body_active.shape
    arrays = {
        "pos": pos,
        "heading": _numeric_array(state, "heading", shape=player_shape),
        "alive": _bool_array(state, "alive", shape=player_shape),
        "tick": _numeric_array(state, "tick", shape=(batch_size,)),
        "map_size": _numeric_array(state, "map_size", shape=(batch_size,)),
        "radius": _numeric_array(state, "radius", shape=player_shape),
        "speed": _numeric_array(state, "speed", shape=player_shape),
        "angular_velocity_per_ms": _numeric_array(
            state,
            "angular_velocity_per_ms",
            shape=player_shape,
        ),
        "body_active": body_active,
        "body_pos": _numeric_array(state, "body_pos", shape=(*body_shape, 2)),
        "body_radius": _numeric_array(state, "body_radius", shape=body_shape),
        "body_owner": _integer_array(state, "body_owner", shape=body_shape),
        "borderless": _optional_bool_row(state, "borderless", batch_size=batch_size),
    }
    if "body_num" in state:
        arrays["body_num"] = _integer_array(state, "body_num", shape=body_shape)
    if "body_write_cursor" in state:
        body_write_cursor = _integer_array(state, "body_write_cursor", shape=(batch_size,))
        if bool(((body_write_cursor < 0) | (body_write_cursor > body_shape[1])).any()):
            raise VectorTrainerObservationError(
                "body_write_cursor values must be within body capacity"
            )
        arrays["body_write_cursor"] = body_write_cursor
    if "live_body_num" in state:
        arrays["live_body_num"] = _integer_array(
            state,
            "live_body_num",
            shape=player_shape,
        )
    if "trail_latency" in state:
        arrays["trail_latency"] = _integer_array(
            state,
            "trail_latency",
            shape=player_shape,
        )
    if bool((arrays["map_size"] <= 0.0).any()):
        raise VectorTrainerObservationError("map_size values must be positive")
    return arrays


def _cast_rays_for_player(
    arrays: Mapping[str, np.ndarray],
    row: int,
    *,
    ego_idx: int,
    opponent_idx: int,
    arena_diagonal: float,
) -> np.ndarray:
    rays = np.ones((len(RAY_ANGLES_DEGREES), len(RAY_CHANNEL_NAMES)), dtype=np.float32)
    origin = arrays["pos"][row, ego_idx]
    directions = _ray_directions(float(arrays["heading"][row, ego_idx]))
    body_limit = _body_limit(arrays, row)
    active = arrays["body_active"][row, :body_limit]
    body_pos = arrays["body_pos"][row, :body_limit]
    body_radius = arrays["body_radius"][row, :body_limit]
    body_owner = arrays["body_owner"][row, :body_limit]
    body_num = arrays.get("body_num", None)
    body_num_row = None if body_num is None else body_num[row, :body_limit]

    own_mask = _trail_body_mask(
        active,
        body_pos,
        body_owner,
        owner=ego_idx,
        owner_head=arrays["pos"][row, ego_idx],
        body_num=body_num_row,
        live_body_num=arrays.get("live_body_num", None),
        trail_latency=arrays.get("trail_latency", None),
        row=row,
    )
    opponent_mask = _trail_body_mask(
        active,
        body_pos,
        body_owner,
        owner=opponent_idx,
        owner_head=arrays["pos"][row, opponent_idx],
    )
    rays[:, 1] = _normalized_hit_distances(
        _nearest_circle_hits(
            origin,
            directions,
            body_pos[own_mask],
            body_radius[own_mask],
        ),
        arena_diagonal,
    )
    rays[:, 2] = _normalized_hit_distances(
        _nearest_circle_hits(
            origin,
            directions,
            body_pos[opponent_mask],
            body_radius[opponent_mask],
        ),
        arena_diagonal,
    )
    rays[:, 3] = _normalized_hit_distances(
        _nearest_circle_hits(
            origin,
            directions,
            arrays["pos"][row, opponent_idx][None, :],
            arrays["radius"][row, opponent_idx : opponent_idx + 1],
        ),
        arena_diagonal,
    )

    if not bool(arrays["borderless"][row]):
        map_size = float(arrays["map_size"][row])
        rays[:, 0] = _normalized_hit_distances(
            _wall_hit_distances(origin, directions, map_size=map_size),
            arena_diagonal,
        )
    return rays


def _body_limit(arrays: Mapping[str, np.ndarray], row: int) -> int:
    body_capacity = int(arrays["body_active"].shape[1])
    if "body_write_cursor" not in arrays:
        return body_capacity
    return int(arrays["body_write_cursor"][row])


def _cast_rays_batch_1v1(
    arrays: Mapping[str, np.ndarray],
    arena_diagonal: np.ndarray,
) -> np.ndarray:
    batch_size = int(arrays["pos"].shape[0])
    ray_count = len(RAY_ANGLES_DEGREES)
    rays = np.ones(
        (batch_size, PLAYER_COUNT, ray_count, len(RAY_CHANNEL_NAMES)),
        dtype=np.float32,
    )
    origins = arrays["pos"].astype(np.float64, copy=False)
    directions = _ray_directions_batch(arrays["heading"].astype(np.float64, copy=False))
    body_arrays = _slice_body_arrays_to_batch_write_cursor(arrays)
    body_active = _active_body_slots(body_arrays)
    body_pos = body_arrays["body_pos"].astype(np.float64, copy=False)
    body_radius = body_arrays["body_radius"].astype(np.float64, copy=False)
    body_owner = body_arrays["body_owner"]

    fallback_owner_masks = _head_excluding_owner_body_masks(
        body_active,
        body_pos,
        body_owner,
        origins,
    )
    own_masks = _own_body_masks_batch(
        body_arrays,
        body_active,
        body_owner,
        fallback_owner_masks,
    )
    opponent_masks = np.empty_like(own_masks)
    opponent_masks[:, 0] = fallback_owner_masks[:, 1]
    opponent_masks[:, 1] = fallback_owner_masks[:, 0]

    rays[:, :, :, 1] = _normalized_hit_distances_batch(
        _nearest_circle_hits_batch(origins, directions, body_pos, body_radius, own_masks),
        arena_diagonal,
    )
    rays[:, :, :, 2] = _normalized_hit_distances_batch(
        _nearest_circle_hits_batch(
            origins,
            directions,
            body_pos,
            body_radius,
            opponent_masks,
        ),
        arena_diagonal,
    )

    opponent_centers = origins[:, [1, 0], None, :]
    opponent_radii = arrays["radius"][:, [1, 0], None].astype(np.float64, copy=False)
    head_masks = np.ones((batch_size, PLAYER_COUNT, 1), dtype=bool)
    rays[:, :, :, 3] = _normalized_hit_distances_batch(
        _nearest_circle_hits_batch(
            origins,
            directions,
            opponent_centers,
            opponent_radii,
            head_masks,
        ),
        arena_diagonal,
    )

    wall_rows = ~arrays["borderless"].astype(bool, copy=False)
    if bool(wall_rows.any()):
        wall_distances = _wall_hit_distances_batch(
            origins[wall_rows],
            directions[wall_rows],
            map_size=arrays["map_size"][wall_rows].astype(np.float64, copy=False),
        )
        rays[wall_rows, :, :, 0] = _normalized_hit_distances_batch(
            wall_distances,
            arena_diagonal[wall_rows],
        )
    return rays


def _slice_body_arrays_to_batch_write_cursor(
    arrays: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    if "body_write_cursor" not in arrays:
        return arrays

    body_capacity = int(arrays["body_active"].shape[1])
    body_write_cursor = arrays["body_write_cursor"]
    body_limit = int(np.max(body_write_cursor)) if body_write_cursor.size else 0
    body_limit = max(0, min(body_limit, body_capacity))
    if body_limit == body_capacity:
        return arrays

    trimmed = dict(arrays)
    for name in ("body_active", "body_pos", "body_radius", "body_owner", "body_num"):
        value = trimmed.get(name)
        if value is None or value.ndim < 2 or int(value.shape[1]) != body_capacity:
            continue
        trimmed[name] = value[:, :body_limit, ...]
    return trimmed


def _ray_directions_batch(headings: np.ndarray) -> np.ndarray:
    cos_heading = np.cos(headings)[:, :, None]
    sin_heading = np.sin(headings)[:, :, None]
    directions = np.empty((*headings.shape, len(RAY_ANGLES_DEGREES), 2), dtype=np.float64)
    directions[:, :, :, 0] = cos_heading * _RAY_COS + sin_heading * _RAY_SIN
    directions[:, :, :, 1] = sin_heading * _RAY_COS - cos_heading * _RAY_SIN
    return directions


def _active_body_slots(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    active = arrays["body_active"].astype(bool, copy=True)
    if "body_write_cursor" not in arrays:
        return active
    slots = np.arange(active.shape[1], dtype=np.int32)[None, :]
    return active & (slots < arrays["body_write_cursor"][:, None])


def _head_excluding_owner_body_masks(
    active: np.ndarray,
    body_pos: np.ndarray,
    body_owner: np.ndarray,
    player_pos: np.ndarray,
) -> np.ndarray:
    masks = np.empty((active.shape[0], PLAYER_COUNT, active.shape[1]), dtype=bool)
    for owner in range(PLAYER_COUNT):
        not_head = (
            np.linalg.norm(body_pos - player_pos[:, owner, None, :], axis=2)
            > _EPSILON
        )
        masks[:, owner] = active & (body_owner == owner) & not_head
    return masks


def _own_body_masks_batch(
    arrays: Mapping[str, np.ndarray],
    active: np.ndarray,
    body_owner: np.ndarray,
    fallback_owner_masks: np.ndarray,
) -> np.ndarray:
    body_num = arrays.get("body_num", None)
    live_body_num = arrays.get("live_body_num", None)
    if body_num is None or live_body_num is None:
        return fallback_owner_masks.copy()

    masks = np.empty_like(fallback_owner_masks)
    trail_latency = arrays.get("trail_latency", None)
    for owner in range(PLAYER_COUNT):
        latency = (
            trail_latency[:, owner]
            if trail_latency is not None
            else np.full(active.shape[0], _DEFAULT_TRAIL_LATENCY, dtype=np.int64)
        )
        own_delta = live_body_num[:, owner, None] - body_num
        masks[:, owner] = active & (body_owner == owner) & (own_delta > latency[:, None])
    return masks


def _scalars_for_player(
    arrays: Mapping[str, np.ndarray],
    row: int,
    *,
    ego_idx: int,
    opponent_idx: int,
    arena_diagonal: float,
    decision_ms: float,
    max_ticks: int,
) -> np.ndarray:
    ego_heading = float(arrays["heading"][row, ego_idx])
    opponent_heading = float(arrays["heading"][row, opponent_idx])
    forward = np.array([math.cos(ego_heading), math.sin(ego_heading)], dtype=np.float64)
    left = np.array([math.sin(ego_heading), -math.cos(ego_heading)], dtype=np.float64)
    delta = arrays["pos"][row, opponent_idx] - arrays["pos"][row, ego_idx]
    tick_fraction = (
        float(np.clip(float(arrays["tick"][row]) / float(max_ticks), 0.0, 1.0))
        if max_ticks > 0
        else 0.0
    )
    relative_heading = opponent_heading - ego_heading
    values = np.asarray(
        [
            1.0 if bool(arrays["alive"][row, ego_idx]) else 0.0,
            1.0 if bool(arrays["alive"][row, opponent_idx]) else 0.0,
            tick_fraction,
            float(np.clip(float(np.dot(delta, forward)) / arena_diagonal, -1.0, 1.0)),
            float(np.clip(float(np.dot(delta, left)) / arena_diagonal, -1.0, 1.0)),
            math.sin(relative_heading),
            math.cos(relative_heading),
            abs(float(arrays["speed"][row, ego_idx])) * decision_ms / 1000.0 / arena_diagonal,
            abs(float(arrays["angular_velocity_per_ms"][row, ego_idx]))
            * decision_ms
            / math.pi,
            max(0.0, float(arrays["radius"][row, ego_idx])) / arena_diagonal,
        ],
        dtype=np.float32,
    )
    if values.shape != (len(SCALAR_NAMES),):
        raise RuntimeError("internal scalar shape mismatch")
    return values


def _row_rewards(
    state: Mapping[str, np.ndarray],
    row: int,
    *,
    player_ids: tuple[str, str],
    done: bool,
    terminated: bool,
    truncated: bool,
) -> tuple[np.ndarray, dict[str, float], dict[str, float] | None, dict[str, Any]]:
    rewards = np.zeros(PLAYER_COUNT, dtype=np.float32)
    reason_name = "none"
    winner_ids: tuple[str, ...] = ()
    loser_ids: tuple[str, ...] = ()
    draw = False
    timeout = False
    truncation_reason = None
    if truncated:
        reason_name, timeout, truncation_reason = _truncation_info(state, row)

    if terminated:
        terminal_reason = _required_terminal_reason(state, row)
        if terminal_reason == vector_runtime.TERMINAL_REASON_SURVIVOR_WIN:
            winner = _required_winner(state, row)
            if winner not in (0, 1):
                raise VectorTrainerObservationError("survivor terminal winner must be 0 or 1")
            rewards[:] = -1.0
            rewards[winner] = 1.0
            reason_name = "survivor_win"
            winner_ids = (player_ids[winner],)
            loser_ids = (player_ids[1 - winner],)
        elif terminal_reason == vector_runtime.TERMINAL_REASON_ALL_DEAD_DRAW:
            _validate_draw_terminal(state, row)
            reason_name = "all_dead_draw"
            draw = True
        else:
            raise VectorTrainerObservationError(
                "unsupported terminal_reason for 1v1 no-bonus trainer reward"
            )
    elif timeout:
        reason_name = "timeout"

    reward_map = {
        player_ids[0]: float(rewards[0]),
        player_ids[1]: float(rewards[1]),
    }
    return (
        rewards,
        reward_map,
        dict(reward_map) if done else None,
        {
            "terminal_reason": reason_name,
            "winner_ids": winner_ids,
            "loser_ids": loser_ids,
            "draw": draw,
            "timeout": timeout,
            "truncation_reason": truncation_reason,
        },
    )


def _truncation_info(
    state: Mapping[str, np.ndarray],
    row: int,
) -> tuple[str, bool, str]:
    if "terminal_reason" not in state:
        return "timeout", True, "max_ticks"
    terminal_reason = _required_terminal_reason(state, row)
    return _TRUNCATION_INFO_BY_TERMINAL_REASON.get(
        terminal_reason,
        ("timeout", True, "max_ticks"),
    )


def _row_done_flags(
    state: Mapping[str, np.ndarray],
    row: int,
    alive: np.ndarray,
) -> tuple[bool, bool, bool]:
    batch_size = alive.shape[0]
    terminal_reason = _optional_terminal_reason(state, row, batch_size)
    terminal_by_reason = terminal_reason in (
        vector_runtime.TERMINAL_REASON_SURVIVOR_WIN,
        vector_runtime.TERMINAL_REASON_ALL_DEAD_DRAW,
    )
    terminated = (
        bool(_optional_bool_row(state, "terminated", batch_size=batch_size)[row])
        if "terminated" in state
        else terminal_by_reason or int(alive[row, :PLAYER_COUNT].sum()) <= 1
    )
    if terminal_by_reason and not terminated:
        raise VectorTrainerObservationError("terminated must match terminal_reason")
    truncated = (
        bool(_optional_bool_row(state, "truncated", batch_size=batch_size)[row])
        if "truncated" in state
        else False
    )
    done = (
        bool(_optional_bool_row(state, "done", batch_size=batch_size)[row])
        if "done" in state
        else terminated or truncated
    )
    if done != (terminated or truncated):
        raise VectorTrainerObservationError("done must equal terminated or truncated")
    if terminal_by_reason and not done:
        raise VectorTrainerObservationError("done must include terminal_reason")
    return done, terminated, truncated


def _done_flag_arrays(
    state: Mapping[str, np.ndarray],
    alive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_size = alive.shape[0]
    if "terminal_reason" in state:
        terminal_reason = _integer_array(state, "terminal_reason", shape=(batch_size,))
    else:
        terminal_reason = np.zeros(batch_size, dtype=np.int16)
    terminal_by_reason = np.isin(
        terminal_reason,
        (
            vector_runtime.TERMINAL_REASON_SURVIVOR_WIN,
            vector_runtime.TERMINAL_REASON_ALL_DEAD_DRAW,
        ),
    )
    terminated = (
        _optional_bool_row(state, "terminated", batch_size=batch_size).copy()
        if "terminated" in state
        else terminal_by_reason | (alive[:, :PLAYER_COUNT].sum(axis=1) <= 1)
    )
    if bool(np.any(terminal_by_reason & ~terminated)):
        raise VectorTrainerObservationError("terminated must match terminal_reason")
    truncated = (
        _optional_bool_row(state, "truncated", batch_size=batch_size).copy()
        if "truncated" in state
        else np.zeros(batch_size, dtype=bool)
    )
    done = (
        _optional_bool_row(state, "done", batch_size=batch_size).copy()
        if "done" in state
        else terminated | truncated
    )
    if bool(np.any(done != (terminated | truncated))):
        raise VectorTrainerObservationError("done must equal terminated or truncated")
    if bool(np.any(terminal_by_reason & ~done)):
        raise VectorTrainerObservationError("done must include terminal_reason")
    return done, terminated, truncated


def _optional_terminal_reason(
    state: Mapping[str, np.ndarray],
    row: int,
    batch_size: int,
) -> int:
    if "terminal_reason" not in state:
        return 0
    reason = _integer_array(state, "terminal_reason", shape=(batch_size,))
    return int(reason[row])


def _required_terminal_reason(state: Mapping[str, np.ndarray], row: int) -> int:
    if "terminal_reason" not in state:
        raise VectorTrainerObservationError("terminal_reason is required for terminal rewards")
    reason = _integer_array(state, "terminal_reason", ndim=1)
    if row >= reason.shape[0]:
        raise VectorTrainerObservationError("terminal_reason must have shape [B]")
    return int(reason[row])


def _required_winner(state: Mapping[str, np.ndarray], row: int) -> int:
    if "winner" not in state:
        raise VectorTrainerObservationError("winner is required for survivor terminal rewards")
    winner = _integer_array(state, "winner", ndim=1)
    if row >= winner.shape[0]:
        raise VectorTrainerObservationError("winner must have shape [B]")
    return int(winner[row])


def _validate_draw_terminal(state: Mapping[str, np.ndarray], row: int) -> None:
    if "draw" in state:
        draw = _optional_bool_row(state, "draw", batch_size=None)
        if row < draw.shape[0] and bool(draw[row]):
            return
    if "winner" in state and _required_winner(state, row) == -1:
        return
    raise VectorTrainerObservationError("draw terminal requires draw true or winner -1")


def _trail_body_mask(
    active: np.ndarray,
    body_pos: np.ndarray,
    body_owner: np.ndarray,
    *,
    owner: int,
    owner_head: np.ndarray,
    body_num: np.ndarray | None = None,
    live_body_num: np.ndarray | None = None,
    trail_latency: np.ndarray | None = None,
    row: int | None = None,
) -> np.ndarray:
    owner_mask = active & (body_owner == owner)
    if not bool(owner_mask.any()):
        return owner_mask
    if body_num is not None and live_body_num is not None and row is not None:
        latency = (
            int(trail_latency[row, owner])
            if trail_latency is not None
            else _DEFAULT_TRAIL_LATENCY
        )
        own_delta = int(live_body_num[row, owner]) - body_num
        return owner_mask & (own_delta > latency)
    not_head = np.linalg.norm(body_pos - owner_head[None, :], axis=1) > _EPSILON
    return owner_mask & not_head


def _nearest_circle_hits(
    origin: np.ndarray,
    directions: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    no_hits = np.full(directions.shape[0], np.inf, dtype=np.float64)
    if centers.size == 0:
        return no_hits
    if centers.shape[0] != radii.shape[0]:
        raise VectorTrainerObservationError("centers and radii must have matching length")
    offsets = origin[None, :] - centers
    clipped_radii = np.maximum(0.0, radii.astype(np.float64, copy=False))
    c_values = np.einsum("ij,ij->i", offsets, offsets) - clipped_radii * clipped_radii
    if bool(np.any(c_values <= 0.0)):
        return np.zeros(directions.shape[0], dtype=np.float64)

    b_values = directions @ offsets.T
    discriminants = b_values * b_values - c_values[None, :]
    valid = discriminants >= 0.0
    roots = -b_values - np.sqrt(np.maximum(discriminants, 0.0))
    roots = np.where(valid & (roots >= 0.0), roots, np.inf)
    return np.min(roots, axis=1)


def _nearest_circle_hits_batch(
    origins: np.ndarray,
    directions: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    batch_size, player_count, ray_count, _xy = directions.shape
    no_hits = np.full((batch_size, player_count, ray_count), np.inf, dtype=np.float64)
    circle_count = int(mask.shape[2])
    if circle_count == 0:
        return no_hits
    if mask.shape != (batch_size, player_count, circle_count):
        raise VectorTrainerObservationError("circle mask has unexpected shape")

    roots_min = no_hits
    overlapping_origin = np.zeros((batch_size, player_count), dtype=bool)
    chunk_size = _circle_hit_chunk_size(batch_size, player_count, ray_count, circle_count)
    for start in range(0, circle_count, chunk_size):
        stop = min(start + chunk_size, circle_count)
        centers_chunk = _circle_center_chunk(centers, start, stop)
        radii_chunk = _circle_radius_chunk(radii, start, stop)
        mask_chunk = mask[:, :, start:stop]
        offsets = origins[:, :, None, :] - centers_chunk
        clipped_radii = np.maximum(0.0, radii_chunk.astype(np.float64, copy=False))
        c_values = offsets[..., 0] * offsets[..., 0] + offsets[..., 1] * offsets[..., 1]
        c_values -= clipped_radii * clipped_radii
        overlapping_origin |= np.any(mask_chunk & (c_values <= 0.0), axis=2)

        b_values = np.einsum("berd,becd->berc", directions, offsets)
        discriminants = b_values * b_values - c_values[:, :, None, :]
        valid = mask_chunk[:, :, None, :] & (discriminants >= 0.0)
        roots = -b_values - np.sqrt(np.maximum(discriminants, 0.0))
        roots = np.where(valid & (roots >= 0.0), roots, np.inf)
        roots_min = np.minimum(roots_min, np.min(roots, axis=3))

    roots_min[overlapping_origin] = 0.0
    return roots_min


def _circle_center_chunk(centers: np.ndarray, start: int, stop: int) -> np.ndarray:
    if centers.ndim == 3:
        return centers[:, None, start:stop, :]
    if centers.ndim == 4:
        return centers[:, :, start:stop, :]
    raise VectorTrainerObservationError("circle array has unexpected rank")


def _circle_radius_chunk(radii: np.ndarray, start: int, stop: int) -> np.ndarray:
    if radii.ndim == 2:
        return radii[:, None, start:stop]
    if radii.ndim == 3:
        return radii[:, :, start:stop]
    raise VectorTrainerObservationError("circle array has unexpected rank")


def _circle_hit_chunk_size(
    batch_size: int,
    player_count: int,
    ray_count: int,
    circle_count: int,
) -> int:
    target_values = 2_000_000
    per_circle_values = max(1, batch_size * player_count * ray_count)
    return max(1, min(circle_count, target_values // per_circle_values))


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
    map_size: float,
) -> float | None:
    x, y = float(origin[0]), float(origin[1])
    if x < 0.0 or x >= map_size or y < 0.0 or y >= map_size:
        return 0.0
    candidates: list[float] = []
    dx, dy = float(direction[0]), float(direction[1])
    if abs(dx) > _EPSILON:
        candidates.append(((map_size if dx > 0.0 else 0.0) - x) / dx)
    if abs(dy) > _EPSILON:
        candidates.append(((map_size if dy > 0.0 else 0.0) - y) / dy)
    hits: list[float] = []
    for distance in candidates:
        if distance < 0.0:
            continue
        hit = origin + direction * distance
        if -_EPSILON <= hit[0] <= map_size + _EPSILON and -_EPSILON <= hit[1] <= map_size + _EPSILON:
            hits.append(float(distance))
    return min(hits) if hits else None


def _wall_hit_distances(
    origin: np.ndarray,
    directions: np.ndarray,
    *,
    map_size: float,
) -> np.ndarray:
    x, y = float(origin[0]), float(origin[1])
    if x < 0.0 or x >= map_size or y < 0.0 or y >= map_size:
        return np.zeros(directions.shape[0], dtype=np.float64)

    dx = directions[:, 0].astype(np.float64, copy=False)
    dy = directions[:, 1].astype(np.float64, copy=False)
    x_target = np.where(dx > 0.0, map_size, 0.0)
    y_target = np.where(dy > 0.0, map_size, 0.0)
    x_distance = np.divide(
        x_target - x,
        dx,
        out=np.full(dx.shape, np.inf, dtype=np.float64),
        where=np.abs(dx) > _EPSILON,
    )
    y_distance = np.divide(
        y_target - y,
        dy,
        out=np.full(dy.shape, np.inf, dtype=np.float64),
        where=np.abs(dy) > _EPSILON,
    )
    distances = np.minimum(
        np.where(x_distance >= 0.0, x_distance, np.inf),
        np.where(y_distance >= 0.0, y_distance, np.inf),
    )
    return distances


def _wall_hit_distances_batch(
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    map_size: np.ndarray,
) -> np.ndarray:
    x = origins[:, :, 0, None].astype(np.float64, copy=False)
    y = origins[:, :, 1, None].astype(np.float64, copy=False)
    size = map_size[:, None, None].astype(np.float64, copy=False)
    outside = (x < 0.0) | (x >= size) | (y < 0.0) | (y >= size)

    dx = directions[:, :, :, 0].astype(np.float64, copy=False)
    dy = directions[:, :, :, 1].astype(np.float64, copy=False)
    x_target = np.where(dx > 0.0, size, 0.0)
    y_target = np.where(dy > 0.0, size, 0.0)
    x_distance = np.divide(
        x_target - x,
        dx,
        out=np.full(dx.shape, np.inf, dtype=np.float64),
        where=np.abs(dx) > _EPSILON,
    )
    y_distance = np.divide(
        y_target - y,
        dy,
        out=np.full(dy.shape, np.inf, dtype=np.float64),
        where=np.abs(dy) > _EPSILON,
    )
    distances = np.minimum(
        np.where(x_distance >= 0.0, x_distance, np.inf),
        np.where(y_distance >= 0.0, y_distance, np.inf),
    )
    return np.where(outside, 0.0, distances)


def _normalized_hit_distances(distances: np.ndarray, arena_diagonal: float) -> np.ndarray:
    distance_array = np.asarray(distances, dtype=np.float64)
    normalized = np.ones(distance_array.shape, dtype=np.float32)
    finite = np.isfinite(distance_array)
    if bool(finite.any()):
        normalized[finite] = np.clip(
            np.maximum(0.0, distance_array[finite]) / float(arena_diagonal),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
    return normalized


def _normalized_hit_distances_batch(
    distances: np.ndarray,
    arena_diagonal: np.ndarray,
) -> np.ndarray:
    distance_array = np.asarray(distances, dtype=np.float64)
    normalized = np.ones(distance_array.shape, dtype=np.float32)
    finite = np.isfinite(distance_array)
    if bool(finite.any()):
        scale = arena_diagonal[:, None, None].astype(np.float64, copy=False)
        values = np.clip(np.maximum(0.0, distance_array) / scale, 0.0, 1.0)
        normalized[finite] = values[finite].astype(np.float32, copy=False)
    return normalized


def _normalized_hit_distances_slow(
    distances: np.ndarray,
    arena_diagonal: float,
) -> np.ndarray:
    return np.asarray(
        [_normalized_hit(float(distance), arena_diagonal) for distance in distances],
        dtype=np.float32,
    )


def _normalized_hit(distance: float | None, arena_diagonal: float) -> float:
    if distance is None or not math.isfinite(distance):
        return 1.0
    return float(np.clip(max(0.0, distance) / arena_diagonal, 0.0, 1.0))


def _arena_diagonal(map_size: float) -> float:
    diagonal = math.hypot(float(map_size), float(map_size))
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise VectorTrainerObservationError("arena diagonal must be positive and finite")
    return diagonal


def _profile_started(profile_timers: dict[str, float] | None) -> float:
    return time.perf_counter() if profile_timers is not None else 0.0


def _profile_add(
    profile_timers: dict[str, float] | None,
    key: str,
    started: float,
) -> None:
    if profile_timers is not None:
        profile_timers[key] = profile_timers.get(key, 0.0) + (time.perf_counter() - started)


def _player_ids(player_ids: Sequence[str]) -> tuple[str, str]:
    raw_ids = tuple(player_ids)
    if len(raw_ids) != PLAYER_COUNT:
        raise VectorTrainerObservationError("player_ids must contain exactly 2 ids")
    ids = (str(raw_ids[0]), str(raw_ids[1]))
    if len(set(ids)) != PLAYER_COUNT:
        raise VectorTrainerObservationError("player_ids must be unique")
    return ids


def _row_index(row: int, batch_size: int) -> int:
    row_index = int(row)
    if row_index < 0 or row_index >= batch_size:
        raise VectorTrainerObservationError("row is outside batch")
    return row_index


def _positive_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise VectorTrainerObservationError(f"{name} must be positive and finite")
    return number


def _bool_row_mask(value: np.ndarray, name: str, *, batch_size: int) -> np.ndarray:
    mask = np.asarray(value)
    if mask.dtype != bool or mask.shape != (batch_size,):
        raise VectorTrainerObservationError(f"{name} must be a bool array with shape [B]")
    return mask


def _numeric_array(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    if name not in state:
        raise VectorTrainerObservationError(f"state is missing {name!r}")
    array = np.asarray(state[name])
    if not np.issubdtype(array.dtype, np.number):
        raise VectorTrainerObservationError(f"{name} must be numeric")
    if shape is not None and array.shape != shape:
        raise VectorTrainerObservationError(f"{name} has unexpected shape")
    if ndim is not None and array.ndim != ndim:
        raise VectorTrainerObservationError(f"{name} has unexpected rank")
    if not bool(np.isfinite(array).all()):
        raise VectorTrainerObservationError(f"{name} values must be finite")
    return array


def _integer_array(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    array = _numeric_array(state, name, shape=shape, ndim=ndim)
    if not np.issubdtype(array.dtype, np.integer):
        raise VectorTrainerObservationError(f"{name} must be integer")
    return array


def _bool_array(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
    batch_size: int | None = None,
) -> np.ndarray:
    if name not in state:
        raise VectorTrainerObservationError(f"state is missing {name!r}")
    array = np.asarray(state[name])
    if array.dtype != bool:
        raise VectorTrainerObservationError(f"{name} must be bool")
    if shape is not None and array.shape != shape:
        raise VectorTrainerObservationError(f"{name} has unexpected shape")
    if ndim is not None and array.ndim != ndim:
        raise VectorTrainerObservationError(f"{name} has unexpected rank")
    if batch_size is not None and array.shape[0] != batch_size:
        raise VectorTrainerObservationError(f"{name} must have leading shape [B]")
    return array


def _optional_bool_row(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    batch_size: int | None,
) -> np.ndarray:
    if name not in state:
        if batch_size is None:
            raise VectorTrainerObservationError(f"state is missing {name!r}")
        return np.zeros(batch_size, dtype=bool)
    array = _bool_array(state, name, ndim=1)
    if batch_size is not None and array.shape != (batch_size,):
        raise VectorTrainerObservationError(f"{name} must have shape [B]")
    return array


__all__ = [
    "VECTOR_TRAINER_OBSERVATION_SURFACE",
    "VECTOR_TRAINER_TRANSITION_SCHEMA_ID",
    "VectorTrainerObservationError",
    "build_final_trainer_transition_1v1_no_bonus_rows",
    "observe_vector_1v1_egocentric_rays_batch_arrays_v0",
    "observe_vector_1v1_egocentric_rays_v0",
]
