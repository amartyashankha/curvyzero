"""Adapters from source-env snapshots to trainer-shaped state.

The first adapter is deliberately narrow. It maps public `CurvyTronSourceEnv`
snapshots into the `EnvState` shape required by the current trainer observation
helper. It supports empty occupancy and a coarse center-cell source-body
occupancy policy. Positions, headings, and alive flags are source-derived;
trail/body ray occupancy is still approximate and not exact source rendering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from curvyzero.env.config import CurvyTronConfig
from curvyzero.env.state import EnvState
from curvyzero.env.vector_reset import TERMINAL_REASON_ALL_DEAD_DRAW
from curvyzero.env.vector_reset import TERMINAL_REASON_NONE
from curvyzero.env.vector_reset import TERMINAL_REASON_SURVIVOR_WIN


class SourceTrainerAdapterError(ValueError):
    """Raised when a source snapshot cannot be adapted safely."""


def config_from_source_snapshot(
    snapshot: Mapping[str, Any],
    *,
    base_config: CurvyTronConfig | None = None,
) -> CurvyTronConfig:
    """Return a trainer config sized to a public source-env snapshot."""

    game = _mapping(snapshot.get("game"), "snapshot.game")
    avatars = _avatars(snapshot)
    size = _positive_int(game.get("size"), "snapshot.game.size")
    base = base_config or CurvyTronConfig()
    return CurvyTronConfig(
        ruleset=base.ruleset,
        rule_provenance=base.rule_provenance,
        width=size,
        height=size,
        players=len(avatars),
        max_ticks=base.max_ticks,
        action_repeat=base.action_repeat,
        speed=base.speed,
        turn_rate_radians=base.turn_rate_radians,
        trail_radius=base.trail_radius,
        trail_gap_period=base.trail_gap_period,
        trail_gap_length=base.trail_gap_length,
        spawn_margin=base.spawn_margin,
        allow_straight_action=base.allow_straight_action,
        reference_defaults=base.reference_defaults,
    )


def source_snapshot_player_ids(snapshot: Mapping[str, Any]) -> tuple[str, ...]:
    """Return stable player ids from source snapshot avatar order."""

    ids = []
    for index, avatar in enumerate(_avatars(snapshot)):
        raw = avatar.get("name", avatar.get("id", f"p{index}"))
        ids.append(str(raw))
    return tuple(ids)


def source_snapshot_to_env_state(
    snapshot: Mapping[str, Any],
    config: CurvyTronConfig,
    *,
    occupancy_policy: str = "empty",
    world_bodies: Sequence[Mapping[str, Any]] | None = None,
    seed: int = 1,
) -> EnvState:
    """Adapt source snapshot positions/headings/alive into `EnvState`.

    Supported policies are `empty` and `source_world_bodies_center_cell_v0`.
    They are useful for actor-loop shape and timing probes, but they are not
    complete source-faithful trail/body observation evidence.
    """

    if occupancy_policy not in ("empty", "source_world_bodies_center_cell_v0"):
        raise SourceTrainerAdapterError(
            "unsupported occupancy_policy; expected 'empty' or "
            "'source_world_bodies_center_cell_v0'"
        )
    avatars = _avatars(snapshot)
    if len(avatars) != config.players:
        raise SourceTrainerAdapterError(
            f"snapshot has {len(avatars)} avatars but config.players={config.players}"
        )
    positions = np.asarray(
        [[_float(avatar.get("x"), "avatar.x"), _float(avatar.get("y"), "avatar.y")]
         for avatar in avatars],
        dtype=np.float32,
    )
    headings = np.asarray(
        [_float(avatar.get("angle"), "avatar.angle") for avatar in avatars],
        dtype=np.float32,
    )
    alive = np.asarray([_bool(avatar.get("alive"), "avatar.alive") for avatar in avatars], dtype=np.bool_)
    occupancy = np.zeros((config.height, config.width), dtype=np.int16)
    if occupancy_policy == "source_world_bodies_center_cell_v0":
        if world_bodies is None:
            raise SourceTrainerAdapterError(
                "world_bodies are required for occupancy_policy='source_world_bodies_center_cell_v0'"
            )
        occupancy = source_world_bodies_to_occupancy(snapshot, world_bodies, config)
    return EnvState(
        tick=_tick_from_snapshot(snapshot, config),
        positions=positions,
        headings=headings,
        alive=alive,
        death_tick=np.full(config.players, -1, dtype=np.int32),
        occupancy=occupancy,
        rng=np.random.default_rng(seed),
    )


def source_world_bodies_to_occupancy(
    snapshot: Mapping[str, Any],
    world_bodies: Sequence[Mapping[str, Any]],
    config: CurvyTronConfig,
) -> np.ndarray:
    """Rasterize source world bodies into trainer grid owner ids.

    This first policy marks only each body's rounded center cell. It is not exact
    source circle geometry and does not encode own-body latency.
    """

    avatar_owner = {
        int(_positive_int(avatar.get("id"), "avatar.id")): index + 1
        for index, avatar in enumerate(_avatars(snapshot))
    }
    occupancy = np.zeros((config.height, config.width), dtype=np.int16)
    for index, body in enumerate(world_bodies):
        if not isinstance(body, Mapping):
            raise SourceTrainerAdapterError(f"world_bodies[{index}] must be a mapping")
        avatar_id = body.get("avatarId")
        if avatar_id is None:
            continue
        owner = avatar_owner.get(int(avatar_id))
        if owner is None:
            continue
        x = int(round(_float(body.get("x"), "body.x")))
        y = int(round(_float(body.get("y"), "body.y")))
        if 0 <= x < config.width and 0 <= y < config.height:
            occupancy[y, x] = np.int16(owner)
    return occupancy


def source_snapshot_to_vector_trainer_state(
    snapshot: Mapping[str, Any],
    config: CurvyTronConfig,
    *,
    world_bodies: Sequence[Mapping[str, Any]],
    avatar_body_metadata: Sequence[Mapping[str, Any]] | None = None,
    decision_ms: float | None = None,
) -> dict[str, np.ndarray]:
    """Adapt source snapshot/body circles into the vector trainer surface.

    This keeps the public trainer observation schema unchanged. Unlike
    `source_world_bodies_center_cell_v0`, body/trail geometry is represented as
    source circle centers/radii plus body numbers for own-body latency masking.
    """

    avatars = _avatars(snapshot)
    if len(avatars) != 2 or int(config.players) != 2:
        raise SourceTrainerAdapterError("source vector trainer adapter currently supports 1v1")
    game = _mapping(snapshot.get("game"), "snapshot.game")
    map_size = _positive_int(game.get("size"), "snapshot.game.size")
    positions = np.asarray(
        [[_float(avatar.get("x"), "avatar.x"), _float(avatar.get("y"), "avatar.y")]
         for avatar in avatars],
        dtype=np.float64,
    )
    headings = np.asarray(
        [_float(avatar.get("angle"), "avatar.angle") for avatar in avatars],
        dtype=np.float64,
    )
    alive = np.asarray(
        [_bool(avatar.get("alive"), "avatar.alive") for avatar in avatars],
        dtype=np.bool_,
    )
    avatar_owner = {
        int(_positive_int(avatar.get("id"), "avatar.id")): index
        for index, avatar in enumerate(avatars)
    }
    body_metadata_by_id = _avatar_body_metadata_by_id(avatar_body_metadata)
    body_capacity = max(1, len(world_bodies))
    body_active = np.zeros((1, body_capacity), dtype=np.bool_)
    body_pos = np.zeros((1, body_capacity, 2), dtype=np.float64)
    body_radius = np.zeros((1, body_capacity), dtype=np.float64)
    body_owner = np.full((1, body_capacity), -1, dtype=np.int16)
    body_num = np.full((1, body_capacity), -1, dtype=np.int32)

    cursor = 0
    for index, body in enumerate(world_bodies):
        if not isinstance(body, Mapping):
            raise SourceTrainerAdapterError(f"world_bodies[{index}] must be a mapping")
        avatar_id = body.get("avatarId")
        if avatar_id is None:
            continue
        owner = avatar_owner.get(int(avatar_id))
        if owner is None:
            continue
        body_active[0, cursor] = True
        body_pos[0, cursor] = [
            _float(body.get("x"), "body.x"),
            _float(body.get("y"), "body.y"),
        ]
        body_radius[0, cursor] = _float(body.get("radius"), "body.radius")
        body_owner[0, cursor] = np.int16(owner)
        body_num[0, cursor] = np.int32(int(body.get("num", -1)))
        cursor += 1

    live_body_num = np.asarray(
        [
            [
                _int(
                    _avatar_runtime_value(avatar, body_metadata_by_id, "bodyNum"),
                    "avatar.bodyNum",
                )
                for avatar in avatars
            ]
        ],
        dtype=np.int32,
    )
    trail_latency = np.asarray(
        [
            [
                _int(
                    _avatar_runtime_value(
                        avatar,
                        body_metadata_by_id,
                        "trailLatency",
                        default=config.reference_defaults.trail_latency_points,
                    ),
                    "avatar.trailLatency",
                )
                for avatar in avatars
            ]
        ],
        dtype=np.int32,
    )
    alive_count = int(alive.sum())
    terminated = alive_count <= 1
    truncated = bool(config.max_ticks > 0 and _tick_from_snapshot(snapshot, config) >= config.max_ticks)
    terminal_reason = TERMINAL_REASON_NONE
    winner = -1
    draw = False
    if terminated and alive_count == 1:
        terminal_reason = TERMINAL_REASON_SURVIVOR_WIN
        winner = int(np.flatnonzero(alive)[0])
    elif terminated:
        terminal_reason = TERMINAL_REASON_ALL_DEAD_DRAW
        draw = True

    scalar_decision_ms = (
        config.reference_defaults.tick_ms * max(1, int(config.action_repeat))
        if decision_ms is None
        else float(decision_ms)
    )
    if scalar_decision_ms <= 0.0:
        raise SourceTrainerAdapterError("decision_ms must be positive")
    source_speed = (
        config.speed * max(1, int(config.action_repeat)) * 1000.0 / scalar_decision_ms
    )
    source_turn_per_ms = abs(float(config.turn_rate_radians)) / scalar_decision_ms
    return {
        "pos": positions[None, :, :],
        "heading": headings[None, :],
        "alive": alive[None, :],
        "tick": np.asarray([_tick_from_snapshot(snapshot, config)], dtype=np.int32),
        "map_size": np.asarray([float(map_size)], dtype=np.float64),
        "radius": np.asarray(
            [
                [
                    _float(
                        _avatar_runtime_value(
                            avatar,
                            body_metadata_by_id,
                            "radius",
                            default=config.reference_defaults.avatar_radius,
                        ),
                        "avatar.radius",
                    )
                    for avatar in avatars
                ]
            ],
            dtype=np.float64,
        ),
        "speed": np.full((1, 2), source_speed, dtype=np.float64),
        "angular_velocity_per_ms": np.full((1, 2), source_turn_per_ms, dtype=np.float64),
        "body_active": body_active,
        "body_pos": body_pos,
        "body_radius": body_radius,
        "body_owner": body_owner,
        "body_num": body_num,
        "body_write_cursor": np.asarray([cursor], dtype=np.int32),
        "live_body_num": live_body_num,
        "trail_latency": trail_latency,
        "borderless": np.asarray([bool(game.get("borderless", False))], dtype=np.bool_),
        "done": np.asarray([terminated or truncated], dtype=np.bool_),
        "terminated": np.asarray([terminated], dtype=np.bool_),
        "truncated": np.asarray([truncated], dtype=np.bool_),
        "terminal_reason": np.asarray([terminal_reason], dtype=np.int16),
        "winner": np.asarray([winner], dtype=np.int16),
        "draw": np.asarray([draw], dtype=np.bool_),
    }


def _avatars(snapshot: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    avatars = snapshot.get("avatars")
    if not isinstance(avatars, Sequence) or isinstance(avatars, (str, bytes)):
        raise SourceTrainerAdapterError("snapshot.avatars must be a sequence")
    normalized = []
    for index, avatar in enumerate(avatars):
        if not isinstance(avatar, Mapping):
            raise SourceTrainerAdapterError(f"snapshot.avatars[{index}] must be a mapping")
        normalized.append(avatar)
    return tuple(normalized)


def _avatar_body_metadata_by_id(
    avatar_body_metadata: Sequence[Mapping[str, Any]] | None,
) -> dict[int, Mapping[str, Any]]:
    if avatar_body_metadata is None:
        return {}
    metadata_by_id: dict[int, Mapping[str, Any]] = {}
    for index, metadata in enumerate(avatar_body_metadata):
        if not isinstance(metadata, Mapping):
            raise SourceTrainerAdapterError(
                f"avatar_body_metadata[{index}] must be a mapping"
            )
        metadata_by_id[int(_positive_int(metadata.get("id"), "avatar_body_metadata.id"))] = metadata
    return metadata_by_id


def _avatar_runtime_value(
    avatar: Mapping[str, Any],
    metadata_by_id: Mapping[int, Mapping[str, Any]],
    key: str,
    *,
    default: Any | None = None,
) -> Any:
    if key in avatar:
        return avatar[key]
    avatar_id = int(_positive_int(avatar.get("id"), "avatar.id"))
    metadata = metadata_by_id.get(avatar_id)
    if metadata is not None and key in metadata:
        return metadata[key]
    if default is not None:
        return default
    raise SourceTrainerAdapterError(f"avatar.{key} is required")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceTrainerAdapterError(f"{field} must be a mapping")
    return value


def _positive_int(value: Any, field: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise SourceTrainerAdapterError(f"{field} must be positive")
    return parsed


def _float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SourceTrainerAdapterError(f"{field} must be numeric") from exc


def _int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SourceTrainerAdapterError(f"{field} must be an integer") from exc


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise SourceTrainerAdapterError(f"{field} must be a boolean")
    return value


def _tick_from_snapshot(snapshot: Mapping[str, Any], config: CurvyTronConfig) -> int:
    at_ms = snapshot.get("atMs", 0.0)
    tick_ms = config.reference_defaults.tick_ms
    return int(round(_float(at_ms, "snapshot.atMs") / tick_ms))


__all__ = [
    "SourceTrainerAdapterError",
    "config_from_source_snapshot",
    "source_snapshot_player_ids",
    "source_snapshot_to_env_state",
    "source_snapshot_to_vector_trainer_state",
    "source_world_bodies_to_occupancy",
]
