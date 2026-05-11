"""Pure ego-row observations over the public multiplayer vector state.

This module is not a second environment.  It consumes the current
``VectorMultiplayerEnv`` state arrays and projects live ego-player rows
into a small fixed scalar observation for 3P/4P no-bonus trainer experiments.
Replay/env lifecycle ownership stays with the public multiplayer env.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
import math
from typing import Any

import numpy as np

from curvyzero.env.trainer_contract import ACTION_NAMES
from curvyzero.env.trainer_contract import legal_action_mask
from curvyzero.env.trainer_contract import stable_contract_hash


MULTIPLAYER_OBSERVATION_SCHEMA_ID = (
    "curvyzero_multiplayer_egocentric_scalars_3p4p_no_bonus/v0"
)
MULTIPLAYER_OBSERVATION_SURFACE_ID = (
    "curvyzero_vector_multiplayer_state_projection_egorows/v0"
)
SUPPORTED_PLAYER_COUNTS = (3, 4)
OPPONENT_SLOT_COUNT = 3
ACTION_COUNT = len(ACTION_NAMES)
PADDED_ROW_ID = -1
_EPSILON = 1e-9

EGO_FEATURE_NAMES = (
    "tick_fraction",
    "ego_radius_norm",
    "wall_forward_norm",
    "wall_left_norm",
    "wall_backward_norm",
    "wall_right_norm",
)
OPPONENT_SLOT_FEATURE_NAMES = (
    "slot_present_alive",
    "rel_forward_norm",
    "rel_left_norm",
    "distance_norm",
    "heading_sin_relative",
    "heading_cos_relative",
    "radius_norm",
)
FEATURE_NAMES = EGO_FEATURE_NAMES + tuple(
    f"opponent_{slot}_{name}"
    for slot in range(OPPONENT_SLOT_COUNT)
    for name in OPPONENT_SLOT_FEATURE_NAMES
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
MULTIPLAYER_OBSERVATION_SHAPE = (len(FEATURE_NAMES),)

HIDDEN_STATE_POLICY: dict[str, Any] = {
    "metadata_sidecar_only": (
        "schema_id",
        "schema_hash",
        "env_row_id",
        "ego_player_id",
        "row_mask",
        "source_shape",
    ),
    "excluded_from_float_observation": (
        "stable_player_index_as_feature",
        "seat_id",
        "color",
        "player_ids",
        "source_player_ids",
        "score",
        "round_score",
        "death_order",
        "reset_seed",
        "random_tape_cursor",
        "random_tape_draw_count",
        "event_rows",
        "trace_refs",
    ),
    "opponent_slot_policy": (
        "Only present+alive opponents are packed. Slots are canonicalized by "
        "ego-frame distance, then relative forward, then relative left; exact "
        "numeric ties use source slot order only as a deterministic final tie-break "
        "and that id is not packed into the float observation."
    ),
    "padding_policy": (
        "Absent players, dead players, missing opponent slots, and pad_to rows are "
        "zero-filled with row_mask/action_mask false and ids set to -1."
    ),
}

MULTIPLAYER_OBSERVATION_SCHEMA: dict[str, Any] = {
    "schema_id": MULTIPLAYER_OBSERVATION_SCHEMA_ID,
    "surface_id": MULTIPLAYER_OBSERVATION_SURFACE_ID,
    "source": "VectorMultiplayerEnv.state arrays",
    "separate_env_implementation": False,
    "supported_player_counts": SUPPORTED_PLAYER_COUNTS,
    "bonus_policy": "no_bonus_state_projection_only",
    "dtype": "float32",
    "row_observation_shape": MULTIPLAYER_OBSERVATION_SHAPE,
    "feature_names": FEATURE_NAMES,
    "action_mask_shape": (ACTION_COUNT,),
    "action_mask_dtype": "bool",
    "lightzero_action_mask_dtype": "int8",
    "emitted_rows": "present_alive_ego_players",
    "legal_mask_policy": (
        "present+alive ego rows have trainer turn3 actions only while the env row "
        "is not done; terminal rows keep the ego observation but expose no legal action"
    ),
    "normalization": {
        "arena_diagonal": "sqrt(2) * map_size",
        "tick_fraction": "tick / max_ticks clipped to [0, 1], or 0 when max_ticks is 0",
        "wall_distances": "ray to axis-aligned map boundary divided by arena diagonal",
        "relative_positions": "ego-frame opponent delta divided by arena diagonal",
        "distance": "euclidean opponent distance divided by arena diagonal",
        "radius": "collision radius divided by arena diagonal",
    },
    "hidden_state_policy": HIDDEN_STATE_POLICY,
    "claims": {
        "learned_observation_schema": True,
        "trainer_ready_env_claim": False,
        "replay_writer_claim": False,
        "visual_or_pixel_claim": False,
        "trail_ray_claim": False,
        "source_fidelity_completion_claim": False,
    },
}
MULTIPLAYER_OBSERVATION_SCHEMA_HASH = stable_contract_hash(MULTIPLAYER_OBSERVATION_SCHEMA)


class VectorMultiplayerObservationError(ValueError):
    """Raised when public multiplayer state cannot be projected safely."""


@dataclass(frozen=True, slots=True)
class VectorMultiplayerObservationRows:
    """Packed ego rows for present/alive multiplayer players."""

    observation: np.ndarray
    action_mask: np.ndarray
    lightzero_action_mask: np.ndarray
    to_play: np.ndarray
    env_row_id: np.ndarray
    ego_player_id: np.ndarray
    row_mask: np.ndarray
    source_shape: tuple[int, int]
    schema_id: str = MULTIPLAYER_OBSERVATION_SCHEMA_ID
    schema_hash: str = MULTIPLAYER_OBSERVATION_SCHEMA_HASH
    schema: Mapping[str, Any] = field(default_factory=lambda: MULTIPLAYER_OBSERVATION_SCHEMA)
    hidden_state_policy: Mapping[str, Any] = field(default_factory=lambda: HIDDEN_STATE_POLICY)

    @property
    def capacity(self) -> int:
        return int(self.row_mask.shape[0])

    @property
    def active_count(self) -> int:
        return int(self.row_mask.sum())


def pack_vector_multiplayer_observation_rows_v0(
    state: Mapping[str, np.ndarray],
    *,
    max_ticks: int,
    allow_straight: bool = True,
    pad_to: int | None = None,
) -> VectorMultiplayerObservationRows:
    """Pack ``VectorMultiplayerEnv.state`` into canonical ego rows.

    Rows are emitted for present+alive ego players in source row-major order.
    Dead/absent ego slots are filtered out; ``pad_to`` can reserve stable
    capacity for downstream collectors without changing the active rows.
    """

    arrays = _validated_arrays(state)
    max_ticks_value = _nonnegative_int(max_ticks, "max_ticks")
    batch_size, player_count = arrays["present"].shape
    active = arrays["present"] & arrays["alive"]
    active_indices = np.argwhere(active)
    active_count = int(active_indices.shape[0])
    capacity = _row_capacity(active_count, pad_to)

    observation = np.zeros((capacity, *MULTIPLAYER_OBSERVATION_SHAPE), dtype=np.float32)
    action_mask = np.zeros((capacity, ACTION_COUNT), dtype=np.bool_)
    env_row_id = np.full(capacity, PADDED_ROW_ID, dtype=np.int32)
    ego_player_id = np.full(capacity, PADDED_ROW_ID, dtype=np.int16)
    row_mask = np.zeros(capacity, dtype=np.bool_)

    for output_row, (env_row_raw, ego_raw) in enumerate(active_indices):
        env_row = int(env_row_raw)
        ego = int(ego_raw)
        observation[output_row] = _pack_ego_features(
            arrays,
            env_row=env_row,
            ego=ego,
            max_ticks=max_ticks_value,
        )
        legal = bool(active[env_row, ego]) and not bool(arrays["done"][env_row])
        action_mask[output_row] = np.asarray(
            legal_action_mask(active=legal, allow_straight=allow_straight),
            dtype=np.bool_,
        )
        env_row_id[output_row] = env_row
        ego_player_id[output_row] = ego
        row_mask[output_row] = True

    return VectorMultiplayerObservationRows(
        observation=observation,
        action_mask=action_mask,
        lightzero_action_mask=action_mask.astype(np.int8, copy=True),
        to_play=np.full(capacity, -1, dtype=np.int64),
        env_row_id=env_row_id,
        ego_player_id=ego_player_id,
        row_mask=row_mask,
        source_shape=(batch_size, player_count),
    )


def _pack_ego_features(
    arrays: Mapping[str, np.ndarray],
    *,
    env_row: int,
    ego: int,
    max_ticks: int,
) -> np.ndarray:
    values = np.zeros(MULTIPLAYER_OBSERVATION_SHAPE, dtype=np.float32)
    map_size = float(arrays["map_size"][env_row])
    arena_diagonal = _arena_diagonal(map_size)
    ego_pos = arrays["pos"][env_row, ego]
    ego_heading = float(arrays["heading"][env_row, ego])
    forward = np.asarray([math.cos(ego_heading), math.sin(ego_heading)], dtype=np.float64)
    left = np.asarray([math.sin(ego_heading), -math.cos(ego_heading)], dtype=np.float64)

    values[FEATURE_INDEX["tick_fraction"]] = (
        np.float32(np.clip(float(arrays["tick"][env_row]) / float(max_ticks), 0.0, 1.0))
        if max_ticks > 0
        else np.float32(0.0)
    )
    values[FEATURE_INDEX["ego_radius_norm"]] = np.float32(
        max(0.0, float(arrays["radius"][env_row, ego])) / arena_diagonal
    )

    wall_dirs = (
        ("wall_forward_norm", forward),
        ("wall_left_norm", left),
        ("wall_backward_norm", -forward),
        ("wall_right_norm", -left),
    )
    borderless = bool(arrays["borderless"][env_row])
    for name, direction in wall_dirs:
        values[FEATURE_INDEX[name]] = np.float32(
            1.0
            if borderless
            else _normalized_wall_distance(ego_pos, direction, map_size, arena_diagonal)
        )

    opponent_players = _canonical_opponents(arrays, env_row=env_row, ego=ego)
    for slot, opponent in enumerate(opponent_players[:OPPONENT_SLOT_COUNT]):
        base = len(EGO_FEATURE_NAMES) + slot * len(OPPONENT_SLOT_FEATURE_NAMES)
        delta = arrays["pos"][env_row, opponent] - ego_pos
        rel_forward = float(np.dot(delta, forward)) / arena_diagonal
        rel_left = float(np.dot(delta, left)) / arena_diagonal
        distance = float(np.linalg.norm(delta)) / arena_diagonal
        relative_heading = float(arrays["heading"][env_row, opponent]) - ego_heading
        values[base + 0] = 1.0
        values[base + 1] = np.float32(np.clip(rel_forward, -1.0, 1.0))
        values[base + 2] = np.float32(np.clip(rel_left, -1.0, 1.0))
        values[base + 3] = np.float32(np.clip(distance, 0.0, 1.0))
        values[base + 4] = np.float32(math.sin(relative_heading))
        values[base + 5] = np.float32(math.cos(relative_heading))
        values[base + 6] = np.float32(
            max(0.0, float(arrays["radius"][env_row, opponent])) / arena_diagonal
        )
    return values


def _canonical_opponents(
    arrays: Mapping[str, np.ndarray],
    *,
    env_row: int,
    ego: int,
) -> np.ndarray:
    present_alive = arrays["present"][env_row] & arrays["alive"][env_row]
    present_alive[ego] = False
    players = np.flatnonzero(present_alive).astype(np.int16)
    if players.size <= 1:
        return players

    ego_pos = arrays["pos"][env_row, ego]
    ego_heading = float(arrays["heading"][env_row, ego])
    forward = np.asarray([math.cos(ego_heading), math.sin(ego_heading)], dtype=np.float64)
    left = np.asarray([math.sin(ego_heading), -math.cos(ego_heading)], dtype=np.float64)
    deltas = arrays["pos"][env_row, players] - ego_pos[None, :]
    distances = np.linalg.norm(deltas, axis=1)
    rel_forward = deltas @ forward
    rel_left = deltas @ left
    order = np.lexsort((players, rel_left, rel_forward, distances))
    return players[order]


def _validated_arrays(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    pos = _numeric_array(state, "pos", ndim=3)
    if pos.shape[2] != 2:
        raise VectorMultiplayerObservationError("pos must have shape [B,P,2]")
    batch_size, player_count = int(pos.shape[0]), int(pos.shape[1])
    if player_count not in SUPPORTED_PLAYER_COUNTS:
        raise VectorMultiplayerObservationError("state must have 3 or 4 players")
    player_shape = (batch_size, player_count)

    arrays = {
        "pos": pos,
        "heading": _numeric_array(state, "heading", shape=player_shape),
        "present": _bool_array(state, "present", shape=player_shape),
        "alive": _bool_array(state, "alive", shape=player_shape),
        "tick": _numeric_array(state, "tick", shape=(batch_size,)),
        "map_size": _numeric_array(state, "map_size", shape=(batch_size,)),
        "radius": _numeric_array(state, "radius", shape=player_shape),
        "done": _optional_done(state, batch_size=batch_size),
        "borderless": _optional_borderless(state, batch_size=batch_size),
    }
    if bool((arrays["alive"] & ~arrays["present"]).any()):
        raise VectorMultiplayerObservationError("alive players must also be present")
    if bool((arrays["map_size"] <= 0.0).any()):
        raise VectorMultiplayerObservationError("map_size values must be positive")
    return arrays


def _numeric_array(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if name not in state:
        raise VectorMultiplayerObservationError(f"state is missing {name!r}")
    try:
        array = np.asarray(state[name], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise VectorMultiplayerObservationError(f"{name} must be numeric") from exc
    if ndim is not None and array.ndim != ndim:
        raise VectorMultiplayerObservationError(f"{name} must have rank {ndim}")
    if shape is not None and array.shape != shape:
        raise VectorMultiplayerObservationError(f"{name} must have shape {shape}")
    if not bool(np.isfinite(array).all()):
        raise VectorMultiplayerObservationError(f"{name} values must be finite")
    return array


def _bool_array(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int, ...],
) -> np.ndarray:
    if name not in state:
        raise VectorMultiplayerObservationError(f"state is missing {name!r}")
    array = np.asarray(state[name])
    if array.dtype != np.bool_ or array.shape != shape:
        raise VectorMultiplayerObservationError(f"{name} must be bool with shape {shape}")
    return array.astype(np.bool_, copy=False)


def _optional_done(state: Mapping[str, np.ndarray], *, batch_size: int) -> np.ndarray:
    if "done" not in state:
        return np.zeros(batch_size, dtype=np.bool_)
    return _bool_array(state, "done", shape=(batch_size,))


def _optional_borderless(state: Mapping[str, np.ndarray], *, batch_size: int) -> np.ndarray:
    if "borderless" not in state:
        return np.zeros(batch_size, dtype=np.bool_)
    return _bool_array(state, "borderless", shape=(batch_size,))


def _normalized_wall_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    map_size: float,
    arena_diagonal: float,
) -> float:
    if _outside_map(origin, map_size):
        return 0.0
    distance = _wall_distance(origin, direction, map_size=map_size)
    if distance is None:
        return 1.0
    return float(np.clip(max(0.0, distance) / arena_diagonal, 0.0, 1.0))


def _wall_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    map_size: float,
) -> float | None:
    x, y = float(origin[0]), float(origin[1])
    dx, dy = float(direction[0]), float(direction[1])
    candidates: list[float] = []
    if abs(dx) > _EPSILON:
        candidates.append(((map_size if dx > 0.0 else 0.0) - x) / dx)
    if abs(dy) > _EPSILON:
        candidates.append(((map_size if dy > 0.0 else 0.0) - y) / dy)

    hits: list[float] = []
    for distance in candidates:
        if distance < 0.0:
            continue
        hit = origin + direction * distance
        if -_EPSILON <= hit[0] <= map_size + _EPSILON and (
            -_EPSILON <= hit[1] <= map_size + _EPSILON
        ):
            hits.append(float(distance))
    return min(hits) if hits else None


def _outside_map(origin: np.ndarray, map_size: float) -> bool:
    x, y = float(origin[0]), float(origin[1])
    return x < 0.0 or x >= map_size or y < 0.0 or y >= map_size


def _arena_diagonal(map_size: float) -> float:
    return math.sqrt(2.0) * float(map_size)


def _row_capacity(active_count: int, pad_to: int | None) -> int:
    if pad_to is None:
        return active_count
    if not isinstance(pad_to, int):
        raise VectorMultiplayerObservationError("pad_to must be an integer when provided")
    if pad_to < active_count:
        raise VectorMultiplayerObservationError(
            f"pad_to={pad_to} is smaller than active row count {active_count}"
        )
    return pad_to


def _nonnegative_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise VectorMultiplayerObservationError(f"{name} must be a nonnegative integer")
    return int(value)


__all__ = [
    "FEATURE_INDEX",
    "FEATURE_NAMES",
    "HIDDEN_STATE_POLICY",
    "MULTIPLAYER_OBSERVATION_SCHEMA",
    "MULTIPLAYER_OBSERVATION_SCHEMA_HASH",
    "MULTIPLAYER_OBSERVATION_SCHEMA_ID",
    "MULTIPLAYER_OBSERVATION_SHAPE",
    "VectorMultiplayerObservationError",
    "VectorMultiplayerObservationRows",
    "pack_vector_multiplayer_observation_rows_v0",
]
