"""Vector-row natural spawn helper for source-derived CurvyTron facts.

This module only handles the first narrow spawn boundary: selected rows,
source-style spawn RNG consumption, positions, headings, alive/present masks,
and optional absent-player death lists. It does not start rounds, schedule
timers, insert bodies, or run a full environment lifecycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import math

import numpy as np


SPAWN_INFO_SCHEMA_ID = "curvyzero_vector_spawn_info/v1"

SOURCE_AVATAR_RADIUS = 0.6
SOURCE_SPAWN_MARGIN_FRACTION = 0.05
SOURCE_SPAWN_DIRECTION_TOLERANCE = 0.3

_FULL_TURN_RADIANS = math.pi * 2.0
_ANGLE_QUARTER_RADIANS = math.pi / 2.0


class VectorSpawnError(ValueError):
    """Raised when vector spawn inputs cannot satisfy the spawn contract."""


def spawn_round_rows(
    state: Mapping[str, np.ndarray],
    row_mask: np.ndarray,
    *,
    player_count: int,
) -> dict[str, Any]:
    """Spawn present players in selected rows using a deterministic random tape.

    Required state arrays:

    - ``pos``: float array with shape ``[B, P, 2]``
    - ``heading``: float array with shape ``[B, P]``
    - ``alive``: bool array with shape ``[B, P]``
    - ``present``: bool array with shape ``[B, P]``
    - ``map_size``: float array with shape ``[B]``
    - ``random_tape_values``: float array with shape ``[B, N]``
    - ``random_tape_length``: integer array with shape ``[B]``
    - ``random_tape_cursor``: integer array with shape ``[B]``
    - ``random_tape_draw_count``: integer array with shape ``[B]``

    Optional state arrays:

    - ``prev_pos``: float array with shape ``[B, P, 2]``; spawned players are
      stamped to their new position.
    - ``random_tape_exhausted``: bool array with shape ``[B]``; selected rows
      are cleared on successful spawn.
    - ``death_count`` and ``death_player``: integer arrays with shapes ``[B]``
      and ``[B, D]``; absent player indices are written in reverse player order.
    """

    mask = _bool_row_mask(row_mask, "row_mask")
    batch_size = mask.shape[0]
    if not isinstance(player_count, int) or player_count < 1:
        raise VectorSpawnError("player_count must be a positive integer")

    arrays = _required_arrays(state, batch_size=batch_size, player_count=player_count)
    optional = _optional_arrays(state, batch_size=batch_size, player_count=player_count)

    plans = [
        _plan_row_spawn(arrays, row=row, player_count=player_count)
        for row in np.flatnonzero(mask)
    ]
    _validate_death_capacity(optional, plans)

    spawned_player_mask = np.zeros((batch_size, player_count), dtype=bool)
    absent_player_mask = np.zeros((batch_size, player_count), dtype=bool)
    random_draw_delta = np.zeros(batch_size, dtype=np.int32)
    random_calls: list[dict[str, Any]] = []

    for plan in plans:
        row = int(plan["row"])
        random_draw_delta[row] = int(plan["draw_delta"])
        arrays["random_tape_cursor"][row] = int(plan["cursor"])
        arrays["random_tape_draw_count"][row] += int(plan["draw_delta"])
        if optional["random_tape_exhausted"] is not None:
            optional["random_tape_exhausted"][row] = False

        for player, position, heading in plan["spawned"]:
            arrays["pos"][row, player] = position
            arrays["heading"][row, player] = heading
            arrays["alive"][row, player] = True
            spawned_player_mask[row, player] = True
            if optional["prev_pos"] is not None:
                optional["prev_pos"][row, player] = position

        for player in plan["absent_players"]:
            arrays["alive"][row, player] = False
            absent_player_mask[row, player] = True

        if optional["death_count"] is not None and optional["death_player"] is not None:
            optional["death_player"][row, :] = -1
            absent_players = np.asarray(plan["absent_players"], dtype=optional["death_player"].dtype)
            optional["death_player"][row, : absent_players.shape[0]] = absent_players
            optional["death_count"][row] = absent_players.shape[0]

        random_calls.extend(plan["random_calls"])

    return {
        "schema": SPAWN_INFO_SCHEMA_ID,
        "spawn_count": int(mask.sum()),
        "spawn_mask": mask.copy(),
        "spawn_rows": np.flatnonzero(mask).astype(np.int32),
        "spawn_order": np.arange(player_count - 1, -1, -1, dtype=np.int16),
        "spawned_player_mask": spawned_player_mask,
        "absent_player_mask": absent_player_mask,
        "random_draw_count_delta": random_draw_delta,
        "random_tape_cursor": arrays["random_tape_cursor"].copy(),
        "random_tape_draw_count": arrays["random_tape_draw_count"].copy(),
        "random_calls": random_calls,
        "world_body_insert_count": 0,
    }


def _required_arrays(
    state: Mapping[str, np.ndarray],
    *,
    batch_size: int,
    player_count: int,
) -> dict[str, np.ndarray]:
    return {
        "pos": _floating_array(state, "pos", (batch_size, player_count, 2)),
        "heading": _floating_array(state, "heading", (batch_size, player_count)),
        "alive": _bool_array(state, "alive", (batch_size, player_count)),
        "present": _bool_array(state, "present", (batch_size, player_count)),
        "map_size": _floating_array(state, "map_size", (batch_size,)),
        "random_tape_values": _floating_array_with_ndim(
            state,
            "random_tape_values",
            ndim=2,
            batch_size=batch_size,
        ),
        "random_tape_length": _integer_array(state, "random_tape_length", (batch_size,)),
        "random_tape_cursor": _integer_array(state, "random_tape_cursor", (batch_size,)),
        "random_tape_draw_count": _integer_array(
            state,
            "random_tape_draw_count",
            (batch_size,),
        ),
    }


def _optional_arrays(
    state: Mapping[str, np.ndarray],
    *,
    batch_size: int,
    player_count: int,
) -> dict[str, np.ndarray | None]:
    death_count = None
    death_player = None
    if "death_count" in state or "death_player" in state:
        if "death_count" not in state or "death_player" not in state:
            raise VectorSpawnError("death_count and death_player must be supplied together")
        death_count = _integer_array(state, "death_count", (batch_size,))
        death_player = _integer_array_with_ndim(
            state,
            "death_player",
            ndim=2,
            batch_size=batch_size,
        )

    return {
        "prev_pos": (
            _floating_array(state, "prev_pos", (batch_size, player_count, 2))
            if "prev_pos" in state
            else None
        ),
        "random_tape_exhausted": (
            _bool_array(state, "random_tape_exhausted", (batch_size,))
            if "random_tape_exhausted" in state
            else None
        ),
        "death_count": death_count,
        "death_player": death_player,
    }


def _plan_row_spawn(
    arrays: Mapping[str, np.ndarray],
    *,
    row: int,
    player_count: int,
) -> dict[str, Any]:
    size = float(arrays["map_size"][row])
    if not math.isfinite(size) or size <= 0.0:
        raise VectorSpawnError("map_size values for selected rows must be finite and positive")

    margin = SOURCE_AVATAR_RADIUS + SOURCE_SPAWN_MARGIN_FRACTION * size
    span = size - margin * 2.0
    if span <= 0.0:
        raise VectorSpawnError("map_size is too small for the source spawn margin")

    cursor = int(arrays["random_tape_cursor"][row])
    draw_count = int(arrays["random_tape_draw_count"][row])
    tape_length = int(arrays["random_tape_length"][row])
    tape_capacity = int(arrays["random_tape_values"].shape[1])
    if cursor < 0 or draw_count < 0 or tape_length < 0:
        raise VectorSpawnError("random tape cursor, length, and draw_count must be non-negative")
    if tape_length > tape_capacity:
        raise VectorSpawnError("random_tape_length cannot exceed random_tape_values capacity")
    if cursor > tape_length:
        raise VectorSpawnError("random_tape_cursor cannot exceed random_tape_length")

    spawned: list[tuple[int, np.ndarray, float]] = []
    absent_players: list[int] = []
    random_calls: list[dict[str, Any]] = []

    def draw(site: str, player: int, attempt: int | None = None) -> float:
        nonlocal cursor, draw_count
        if cursor >= tape_length:
            raise VectorSpawnError(
                f"random tape exhausted while spawning row {row}, player {player}, site {site}"
            )
        value = float(arrays["random_tape_values"][row, cursor])
        if not math.isfinite(value) or value < 0.0 or value >= 1.0:
            raise VectorSpawnError("consumed random_tape_values must be finite values in [0, 1)")
        random_calls.append(
            {
                "row": row,
                "player": player,
                "site": site,
                "attempt": attempt,
                "tape_index": cursor,
                "draw_ordinal": draw_count,
                "value": value,
            }
        )
        cursor += 1
        draw_count += 1
        return value

    present = arrays["present"]
    for player in range(player_count - 1, -1, -1):
        if not bool(present[row, player]):
            absent_players.append(player)
            continue

        x = margin + draw("spawn.position_x", player) * span
        y = margin + draw("spawn.position_y", player) * span
        attempt = 0
        while True:
            heading = draw(f"spawn.angle_attempt_{attempt}", player, attempt) * _FULL_TURN_RADIANS
            if _direction_valid(heading, x, y, size=size):
                spawned.append((player, np.asarray([x, y], dtype=arrays["pos"].dtype), heading))
                break
            attempt += 1

    return {
        "row": row,
        "cursor": cursor,
        "draw_delta": len(random_calls),
        "spawned": spawned,
        "absent_players": absent_players,
        "random_calls": random_calls,
    }


def _validate_death_capacity(
    optional: Mapping[str, np.ndarray | None],
    plans: list[dict[str, Any]],
) -> None:
    death_player = optional["death_player"]
    if death_player is None:
        return
    death_capacity = int(death_player.shape[1])
    for plan in plans:
        absent_count = len(plan["absent_players"])
        if absent_count > death_capacity:
            raise VectorSpawnError("death_player capacity is too small for absent players")


def _direction_valid(angle: float, x: float, y: float, *, size: float) -> bool:
    margin = SOURCE_SPAWN_DIRECTION_TOLERANCE * size
    for border in range(4):
        start = _ANGLE_QUARTER_RADIANS * border
        end = _ANGLE_QUARTER_RADIANS * (border + 1)
        if angle >= start and angle < end:
            if _hypotenuse(angle - start, _distance_to_border(border, x, y, size)) < margin:
                return False
            next_border = border + 1 if border < 3 else 0
            if _hypotenuse(end - angle, _distance_to_border(next_border, x, y, size)) < margin:
                return False
            return True
    return False


def _hypotenuse(angle: float, adjacent: float) -> float:
    return adjacent / math.cos(angle)


def _distance_to_border(border: int, x: float, y: float, size: float) -> float:
    if border == 0:
        return size - x
    if border == 1:
        return size - y
    if border == 2:
        return x
    return y


def _bool_row_mask(value: np.ndarray, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.dtype != bool or mask.ndim != 1:
        raise VectorSpawnError(f"{name} must be a bool array with shape [B]")
    return mask


def _bool_array(
    state: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = _array(state, name)
    if array.dtype != bool or array.shape != shape:
        raise VectorSpawnError(f"{name} must be a bool array with shape {_shape_phrase(shape)}")
    return array


def _floating_array(
    state: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = _array(state, name)
    if not np.issubdtype(array.dtype, np.floating) or array.shape != shape:
        raise VectorSpawnError(f"{name} must be a float array with shape {_shape_phrase(shape)}")
    return array


def _floating_array_with_ndim(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    ndim: int,
    batch_size: int,
) -> np.ndarray:
    array = _array(state, name)
    if (
        not np.issubdtype(array.dtype, np.floating)
        or array.ndim != ndim
        or array.shape[0] != batch_size
    ):
        raise VectorSpawnError(f"{name} must be a float array with leading shape [B]")
    return array


def _integer_array(
    state: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = _array(state, name)
    if not np.issubdtype(array.dtype, np.integer) or array.shape != shape:
        raise VectorSpawnError(f"{name} must be an integer array with shape {_shape_phrase(shape)}")
    return array


def _integer_array_with_ndim(
    state: Mapping[str, np.ndarray],
    name: str,
    *,
    ndim: int,
    batch_size: int,
) -> np.ndarray:
    array = _array(state, name)
    if (
        not np.issubdtype(array.dtype, np.integer)
        or array.ndim != ndim
        or array.shape[0] != batch_size
    ):
        raise VectorSpawnError(f"{name} must be an integer array with leading shape [B]")
    return array


def _array(state: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in state:
        raise VectorSpawnError(f"state is missing required spawn array {name!r}")
    return np.asarray(state[name])


def _shape_phrase(shape: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(part) for part in shape) + "]"
