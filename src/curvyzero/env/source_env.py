"""Scalar source-shaped CurvyTron oracle.

This module is a proof tool for source-project gameplay semantics, not the
product runtime path. It mirrors the lifecycle spine in ``fidelity.source_runners``
and the original JS model code so the fast vector env can absorb source rules
without guessing: round timers, reverse avatar iteration, delayed PrintManager
starts, source arena sizing, no-bonus wall-death scoring, and the narrow
source-backed bonus slices covered by tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

from curvyzero.env.config import CurvyTronReferenceDefaults

_ROUND_DIGITS = 6
_MAX_TIMER_CALLBACKS_PER_ADVANCE = 1000
_SOURCE_BONUS_DURATIONS_MS = {
    "BonusSelfSmall": 7_500,
    "BonusSelfSlow": 5_000,
    "BonusSelfFast": 4_000,
    "BonusSelfMaster": 7_500,
    "BonusEnemySlow": 5_000,
    "BonusEnemyFast": 6_000,
    "BonusEnemyBig": 7_500,
    "BonusEnemyInverse": 5_000,
    "BonusEnemyStraightAngle": 5_000,
    "BonusGameBorderless": 10_000,
    "BonusAllColor": 7_500,
    "BonusGameClear": 0,
}
_SOURCE_DEFAULT_BONUS_PROBABILITIES = {
    "BonusSelfSmall": 1.0,
    "BonusSelfSlow": 1.0,
    "BonusSelfFast": 1.0,
    "BonusSelfMaster": 1.0,
    "BonusEnemySlow": 1.0,
    "BonusEnemyFast": 1.0,
    "BonusEnemyBig": 1.0,
    "BonusEnemyInverse": 0.8,
    "BonusEnemyStraightAngle": 0.6,
    "BonusGameBorderless": 0.8,
    "BonusAllColor": 1.0,
    "BonusGameClear": 1.0,
}


class SourceEnvError(ValueError):
    """Raised when the source-shaped scalar env receives invalid input."""


@dataclass(slots=True)
class SourceRandomTape:
    """Deterministic stand-in for source ``Math.random`` with labeled calls."""

    sequence: tuple[float, ...] | None = None
    constant: float = 0.5
    calls: list[dict[str, object]] = field(default_factory=list)

    def random(self, *, at_ms: float, site: str, avatar_id: int | None) -> float:
        index = len(self.calls)
        if self.sequence is None:
            value = self.constant
        else:
            if index >= len(self.sequence):
                raise SourceEnvError(f"Math.random tape exhausted after {index} calls")
            value = self.sequence[index]
        _validate_random_value(value, f"random value {index}")
        self.calls.append(
            {
                "index": index,
                "value": value,
                "atMs": at_ms,
                "label": {"site": site, "avatar": avatar_id},
            }
        )
        return value


@dataclass(slots=True)
class SourcePrintManagerState:
    active: bool = False
    distance: float = 0.0
    last_x: float = 0.0
    last_y: float = 0.0

    def clear(self) -> None:
        self.active = False
        self.distance = 0.0
        self.last_x = 0.0
        self.last_y = 0.0

    def to_snapshot(self) -> dict[str, object]:
        return {
            "active": self.active,
            "distance": _source_number(self.distance),
            "lastX": _source_number(self.last_x),
            "lastY": _source_number(self.last_y),
        }


@dataclass(slots=True)
class SourceBodyState:
    x: float
    y: float
    radius: float
    avatar_id: int | None = None
    bonus_id: int | None = None
    num: int = 0
    birth_ms: float = 0.0
    trail_latency: int = 3
    break_before: bool = False
    id: int | None = None
    island_ids: set[str] = field(default_factory=set)

    def matches(self, current_body: SourceBodyState) -> bool:
        if self.avatar_id is not None and self.avatar_id == current_body.avatar_id:
            return current_body.num - self.num > current_body.trail_latency
        return True

    def is_old(self, now_ms: float) -> bool:
        return now_ms - self.birth_ms >= 2000


@dataclass(slots=True)
class SourceVisualTrailPoint:
    x: float
    y: float
    radius: float
    avatar_id: int
    break_before: bool = False

    def to_snapshot(self) -> dict[str, object]:
        return {
            "x": _source_number(self.x),
            "y": _source_number(self.y),
            "radius": _source_number(self.radius),
            "avatarId": self.avatar_id,
            "breakBefore": self.break_before,
        }


@dataclass(slots=True)
class SourceBonusState:
    id: int
    type: str
    x: float
    y: float
    radius: float = 3.0
    duration: int = 7500
    target_id: int | None = None
    target_ids: tuple[int, ...] = ()
    color_avatar_ids: tuple[int, ...] = ()
    color_values: tuple[str, ...] = ()
    body: SourceBodyState = field(init=False)

    def __post_init__(self) -> None:
        self.body = SourceBodyState(
            x=self.x,
            y=self.y,
            radius=self.radius,
            bonus_id=self.id,
        )

    def to_stack_snapshot(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type": self.type,
            "duration": self.duration,
            "effects": _bonus_snapshot_effects(self),
        }


@dataclass(slots=True)
class SourceIslandState:
    id: str
    size: float
    from_x: float
    from_y: float
    bodies: list[SourceBodyState] = field(default_factory=list)

    @property
    def to_x(self) -> float:
        return self.from_x + self.size

    @property
    def to_y(self) -> float:
        return self.from_y + self.size

    def add_body(self, body: SourceBodyState) -> None:
        if self.id in body.island_ids:
            return
        body.island_ids.add(self.id)
        self.bodies.append(body)

    def remove_body(self, body: SourceBodyState) -> None:
        if self.id not in body.island_ids:
            return
        body.island_ids.discard(self.id)
        try:
            self.bodies.remove(body)
        except ValueError:
            pass

    def get_body(self, body: SourceBodyState) -> SourceBodyState | None:
        if not self._body_in_bound(body):
            return None
        for stored_body in reversed(self.bodies):
            if _bodies_touch(stored_body, body):
                return stored_body
        return None

    def clear(self) -> None:
        self.bodies.clear()

    def _body_in_bound(self, body: SourceBodyState) -> bool:
        return (
            body.x + body.radius > self.from_x
            and body.x - body.radius < self.to_x
            and body.y + body.radius > self.from_y
            and body.y - body.radius < self.to_y
        )


@dataclass(slots=True)
class SourceWorldState:
    size: int
    island_count: int | None = None
    active: bool = False
    body_count: int = 0
    island_size: float = field(init=False)
    islands: dict[str, SourceIslandState] = field(init=False)

    def __post_init__(self) -> None:
        islands = self.island_count
        if islands is None:
            islands = max(1, _js_round(self.size / 40.0))
        if islands < 1:
            raise SourceEnvError("source world island count must be at least 1")
        self.island_count = islands
        self.island_size = self.size / islands
        self.islands = {}
        for y in range(islands - 1, -1, -1):
            for x in range(islands - 1, -1, -1):
                island_id = f"{x}:{y}"
                self.islands[island_id] = SourceIslandState(
                    id=island_id,
                    size=self.island_size,
                    from_x=x * self.island_size,
                    from_y=y * self.island_size,
                )

    def clear(self) -> None:
        self.active = False
        self.body_count = 0
        for island in self.islands.values():
            island.clear()

    def activate(self) -> None:
        self.active = True

    def add_body(self, body: SourceBodyState) -> None:
        if not self.active:
            return
        body.id = self.body_count
        body.island_ids.clear()
        self.body_count += 1
        self._add_body_by_point(body, body.x - body.radius, body.y - body.radius)
        self._add_body_by_point(body, body.x + body.radius, body.y - body.radius)
        self._add_body_by_point(body, body.x - body.radius, body.y + body.radius)
        self._add_body_by_point(body, body.x + body.radius, body.y + body.radius)

    def remove_body(self, body: SourceBodyState) -> None:
        if not self.active:
            return
        for island_id in list(body.island_ids):
            island = self.islands.get(island_id)
            if island is not None:
                island.remove_body(body)
        body.island_ids.clear()

    def get_body(self, body: SourceBodyState) -> SourceBodyState | None:
        return (
            self._get_body_by_point(body, body.x - body.radius, body.y - body.radius)
            or self._get_body_by_point(body, body.x + body.radius, body.y - body.radius)
            or self._get_body_by_point(body, body.x - body.radius, body.y + body.radius)
            or self._get_body_by_point(body, body.x + body.radius, body.y + body.radius)
        )

    def iter_unique_bodies(self) -> tuple[SourceBodyState, ...]:
        """Return stored bodies once, deduped across spatial islands."""

        bodies: dict[int, SourceBodyState] = {}
        fallback_index = self.body_count
        for island in self.islands.values():
            for body in island.bodies:
                body_id = body.id
                if body_id is None:
                    body_id = fallback_index
                    fallback_index += 1
                bodies.setdefault(int(body_id), body)
        return tuple(body for _body_id, body in sorted(bodies.items()))

    def _add_body_by_point(self, body: SourceBodyState, x: float, y: float) -> None:
        island = self._island_by_point(x, y)
        if island is not None:
            island.add_body(body)

    def _get_body_by_point(
        self,
        body: SourceBodyState,
        x: float,
        y: float,
    ) -> SourceBodyState | None:
        island = self._island_by_point(x, y)
        if island is None:
            return None
        return island.get_body(body)

    def _island_by_point(self, x: float, y: float) -> SourceIslandState | None:
        island_x = math.floor(x / self.island_size)
        island_y = math.floor(y / self.island_size)
        return self.islands.get(f"{island_x}:{island_y}")


@dataclass(slots=True)
class SourceAvatarState:
    id: int
    name: str
    color: str = "#ffffff"
    player_color: str = "#ffffff"
    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0
    alive: bool = True
    present: bool = True
    printing: bool = False
    score: int = 0
    round_score: int = 0
    trail_point_count: int = 0
    trail_last_x: float | None = None
    trail_last_y: float | None = None
    visual_trail_last_x: float | None = None
    visual_trail_last_y: float | None = None
    body_num: int = 0
    body_count: int = 0
    velocity: float = 16.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    angular_velocity: float = 0.0
    angular_velocity_base: float = 2.8 / 1000.0
    radius: float = 0.6
    trail_latency: int = 3
    inverse: bool = False
    invincible: bool = False
    direction_in_loop: bool = True
    print_manager: SourcePrintManagerState = field(default_factory=SourcePrintManagerState)
    active_bonuses: list[SourceBonusState] = field(default_factory=list)
    visual_trail_points: list[SourceVisualTrailPoint] = field(default_factory=list)

    def clear_for_round(self, reference: CurvyTronReferenceDefaults) -> None:
        self.x = reference.avatar_radius
        self.y = reference.avatar_radius
        self.angle = 0.0
        self.alive = True
        self.printing = False
        self.color = self.player_color
        self.round_score = 0
        self.body_num = 0
        self.body_count = 0
        self.velocity = reference.avatar_velocity_units_per_s
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.angular_velocity = 0.0
        self.angular_velocity_base = reference.angular_velocity_radians_per_ms
        self.radius = reference.avatar_radius
        self.trail_latency = reference.trail_latency_points
        self.visual_trail_last_x = None
        self.visual_trail_last_y = None
        self.inverse = False
        self.invincible = False
        self.direction_in_loop = True
        self.print_manager.clear()
        self.active_bonuses.clear()
        self.visual_trail_points.clear()

    def set_angle(self, angle: float) -> None:
        self.angle = angle
        self.update_velocities()

    def set_position(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.body_num = self.body_count

    def set_velocity(self, velocity: float, reference: CurvyTronReferenceDefaults) -> None:
        self.velocity = max(float(velocity), reference.avatar_velocity_units_per_s / 2.0)
        self.update_velocities(reference)

    def update_angular_velocity(self, factor: float | None = None) -> None:
        if factor is None:
            if self.angular_velocity == 0:
                return
            factor = 1.0 if self.angular_velocity > 0 else -1.0
            if self.inverse:
                factor *= -1.0
        self.angular_velocity = float(factor) * self.angular_velocity_base
        if self.inverse:
            self.angular_velocity *= -1.0

    def update_base_angular_velocity(self, reference: CurvyTronReferenceDefaults) -> None:
        if not self.direction_in_loop:
            return
        ratio = self.velocity / reference.avatar_velocity_units_per_s
        self.angular_velocity_base = (
            ratio * reference.angular_velocity_radians_per_ms + math.log(1.0 / ratio) / 1000.0
        )
        self.update_angular_velocity()

    def update_velocities(self, reference: CurvyTronReferenceDefaults | None = None) -> None:
        velocity = self.velocity / 1000.0
        self.velocity_x = math.cos(self.angle) * velocity
        self.velocity_y = math.sin(self.angle) * velocity
        if reference is not None:
            self.update_base_angular_velocity(reference)

    def update_angle(self, step_ms: float) -> bool:
        if not self.angular_velocity:
            return False
        previous_angle = self.angle
        if self.direction_in_loop:
            self.set_angle(self.angle + self.angular_velocity * step_ms)
        else:
            self.set_angle(self.angle + self.angular_velocity)
            self.update_angular_velocity(0)
        return self.angle != previous_angle

    def update_position(self, step_ms: float) -> None:
        self.set_position(
            self.x + self.velocity_x * step_ms,
            self.y + self.velocity_y * step_ms,
        )

    def live_body(self) -> SourceBodyState:
        return SourceBodyState(
            x=self.x,
            y=self.y,
            radius=self.radius,
            avatar_id=self.id,
            num=self.body_num,
            trail_latency=self.trail_latency,
        )

    def stored_body(self, birth_ms: float, *, break_before: bool = False) -> SourceBodyState:
        return SourceBodyState(
            x=self.x,
            y=self.y,
            radius=self.radius,
            avatar_id=self.id,
            num=self.body_count,
            birth_ms=birth_ms,
            trail_latency=self.trail_latency,
            break_before=break_before,
        )

    def to_snapshot(self, *, include_bonus: bool = False) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "id": self.id,
            "name": self.name,
            "x": _source_number(self.x),
            "y": _source_number(self.y),
            "angle": _source_number(self.angle),
            "alive": self.alive,
            "present": self.present,
            "printing": self.printing,
            "score": self.score,
            "roundScore": self.round_score,
            "trailPointCount": self.trail_point_count,
            "printManager": self.print_manager.to_snapshot(),
        }
        if include_bonus:
            snapshot["radius"] = _source_number(self.radius)
            snapshot["activeBonuses"] = [bonus.to_stack_snapshot() for bonus in self.active_bonuses]
        return snapshot


@dataclass(slots=True)
class SourceGameState:
    size: int
    max_score: float
    world: SourceWorldState | None = None
    bonus_world: SourceWorldState | None = None
    borderless: bool = False
    started: bool = False
    in_round: bool = False
    world_active: bool = False
    world_body_count: int = 0
    bonus_count: int = 0
    bonus_world_body_count: int = 0
    frame_scheduled: bool = False
    rendered: int | None = None
    frame_due_ms: float | None = None
    game_start_due_ms: float | None = None
    print_start_due_ms: float | None = None
    bonus_pop_due_ms: float | None = None
    game_stop_due_ms: float | None = None
    world_exists: bool = True
    death_ids: list[int] = field(default_factory=list)
    active_bonuses: list[SourceBonusState] = field(default_factory=list)

    def to_snapshot(
        self,
        *,
        include_deaths: bool = False,
        include_bonus: bool = False,
    ) -> dict[str, object]:
        world_active = self.world.active if self.world is not None else self.world_active
        world_body_count = (
            self.world.body_count if self.world is not None else self.world_body_count
        )
        bonus_world_body_count = (
            self.bonus_world.body_count
            if self.bonus_world is not None
            else self.bonus_world_body_count
        )
        snapshot: dict[str, object] = {
            "size": self.size,
            "started": self.started,
            "inRound": self.in_round,
            "worldActive": world_active if self.world_exists else None,
            "worldBodyCount": world_body_count if self.world_exists else None,
            "frameScheduled": self.frame_scheduled,
            "rendered": self.rendered,
        }
        if include_deaths:
            if self.world_exists:
                snapshot["deathCount"] = len(self.death_ids)
                snapshot["deaths"] = list(self.death_ids)
            else:
                snapshot["deathCount"] = None
                snapshot["deaths"] = None
        if include_bonus:
            snapshot["bonusCount"] = self.bonus_count
            snapshot["bonusWorldBodyCount"] = bonus_world_body_count
            snapshot["activeBonuses"] = [bonus.to_stack_snapshot() for bonus in self.active_bonuses]
        return snapshot


class CurvyTronSourceEnv:
    """Small public scalar env for source-shaped gameplay canaries."""

    def __init__(
        self,
        *,
        reference: CurvyTronReferenceDefaults | None = None,
        random_values: Sequence[float | Mapping[str, Any]] | None = None,
        random_constant: float = 0.5,
        max_score: float | None = None,
        include_deaths_snapshot: bool = False,
        include_bonus_snapshot: bool = False,
        emit_step_position_events: bool = False,
        emit_step_angle_events: bool = False,
        drain_frame_timers: bool = False,
        bonus_types: Sequence[str] | None = None,
        bonus_rate: float | None = None,
    ) -> None:
        self.reference = reference or CurvyTronReferenceDefaults()
        self.random = SourceRandomTape(
            sequence=(
                tuple(
                    _coerce_random_sequence_value(value, index)
                    for index, value in enumerate(random_values)
                )
                if random_values is not None
                else None
            ),
            constant=random_constant,
        )
        self.default_max_score = max_score
        self.include_deaths_snapshot = include_deaths_snapshot
        self.include_bonus_snapshot = include_bonus_snapshot
        self.emit_step_position_events = emit_step_position_events
        self.emit_step_angle_events = emit_step_angle_events
        self.drain_frame_timers = drain_frame_timers
        self.bonus_types = _normalize_bonus_types(bonus_types)
        self.bonus_rate = _coerce_bonus_rate(
            self.reference.default_bonus_rate if bonus_rate is None else bonus_rate
        )
        self.now_ms = 0.0
        self.events: list[dict[str, object]] = []
        self.avatars: list[SourceAvatarState] = []
        self.active_bonuses: list[SourceBonusState] = []
        self._next_bonus_id = 1
        self._bonus_expiry_timers: list[tuple[float, int, SourceBonusState]] = []
        self._next_bonus_expiry_timer_order = 1
        self.game: SourceGameState | None = None

    @property
    def random_calls(self) -> list[dict[str, object]]:
        return self.random.calls

    def reset(
        self,
        *,
        player_count: int = 2,
        players: Sequence[Mapping[str, Any]] | None = None,
        present: Sequence[bool] | None = None,
        warmup_ms: float | None = None,
        max_score: float | None = None,
        borderless: bool = False,
        bonus_types: Sequence[str] | None = None,
        bonus_rate: float | None = None,
    ) -> dict[str, object]:
        """Start a source-shaped round and return the post-new-round snapshot."""

        if player_count < 1:
            raise SourceEnvError("player_count must be at least 1")
        if bonus_types is not None or bonus_rate is not None:
            self._configure_bonus_manager(
                bonus_types=self.bonus_types if bonus_types is None else bonus_types,
                bonus_rate=self.bonus_rate if bonus_rate is None else bonus_rate,
            )
        self.now_ms = 0.0
        self.events.clear()
        self.random.calls.clear()
        self.avatars = self._build_avatars(
            player_count=player_count,
            players=players,
            present=present,
        )
        self.game = SourceGameState(
            size=self.reference.arena_size_for_players(player_count),
            max_score=(
                float(max_score)
                if max_score is not None
                else float(
                    self.default_max_score
                    if self.default_max_score is not None
                    else self.reference.max_score_for_players(player_count)
                )
            ),
            borderless=bool(borderless),
        )
        self.game.world = SourceWorldState(self.game.size)
        self.game.bonus_world = SourceWorldState(self.game.size, island_count=1)
        self.active_bonuses.clear()
        self._next_bonus_id = 1
        self._bonus_expiry_timers.clear()
        self._next_bonus_expiry_timer_order = 1
        delay_ms = self.reference.round_warmup_ms if warmup_ms is None else float(warmup_ms)
        self._new_round(delay_ms)
        return self.snapshot("after_new_round_call")

    def advance_timers(self, elapsed_ms: float) -> None:
        """Advance source setTimeout-style lifecycle timers."""

        self._require_game()
        if elapsed_ms < 0:
            raise SourceEnvError("elapsed_ms must be non-negative")
        target = self.now_ms + float(elapsed_ms)
        callbacks = 0
        while True:
            game = self._require_game()
            timers = [
                (game.game_start_due_ms, 0, "game:start", None),
                (game.print_start_due_ms, 1, "print_manager:start", None),
                (game.bonus_pop_due_ms, 2, "bonus:pop", None),
                (game.game_stop_due_ms, 3, "game:stop", None),
            ]
            timers.extend(
                (due_ms, 2.5 + order * 1e-9, "bonus:expiry", bonus)
                for due_ms, order, bonus in self._bonus_expiry_timers
            )
            if self.drain_frame_timers and game.frame_scheduled:
                timers.append((game.frame_due_ms, 4, "frame", None))
            due_timers = [timer for timer in timers if timer[0] is not None]
            if not due_timers:
                break

            due_ms, _order, timer_name, timer_payload = min(due_timers)
            if due_ms is None or due_ms > target:
                break
            if callbacks >= _MAX_TIMER_CALLBACKS_PER_ADVANCE:
                raise SourceEnvError(
                    f"timer advance exceeded {_MAX_TIMER_CALLBACKS_PER_ADVANCE} callbacks"
                )

            callbacks += 1
            self.now_ms = due_ms
            if timer_name == "game:start":
                game.game_start_due_ms = None
                self._start_game()
            elif timer_name == "print_manager:start":
                self._run_delayed_print_starts()
            elif timer_name == "bonus:pop":
                self._pop_bonus()
            elif timer_name == "bonus:expiry":
                assert isinstance(timer_payload, SourceBonusState)
                self._remove_bonus_expiry_timer(timer_payload)
                self._expire_bonus_from_stack(timer_payload)
            elif timer_name == "game:stop":
                self._stop_game()
            else:
                self._run_frame()
        self.now_ms = target

    def step(
        self,
        joint_actions: Mapping[int, float | int | str] | Sequence[float | int | str],
        elapsed_ms: float,
    ) -> dict[str, object]:
        """Apply source move/control state, then advance one elapsed-ms source frame.

        ``joint_actions`` is wrapper language for per-avatar source move factors
        applied before the frame; native source control is elapsed-ms state
        update, not a discrete trainer step.
        """

        if elapsed_ms < 0:
            raise SourceEnvError("elapsed_ms must be non-negative")
        self._apply_joint_actions(joint_actions)
        self._update_game(float(elapsed_ms))
        return self.snapshot("after_step")

    def snapshot(
        self,
        label: str = "snapshot",
        *,
        advance_ms: float | None = None,
        action: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        game = self._require_game()
        frame: dict[str, object] = {
            "label": label,
            "atMs": _source_number(self.now_ms),
            "game": game.to_snapshot(
                include_deaths=self.include_deaths_snapshot,
                include_bonus=self.include_bonus_snapshot,
            ),
            "avatars": [
                avatar.to_snapshot(include_bonus=self.include_bonus_snapshot)
                for avatar in self.avatars
            ],
        }
        if action is not None:
            frame["action"] = dict(action)
        if advance_ms is not None:
            frame["advanceMs"] = _source_number(advance_ms)
        return frame

    def world_bodies_snapshot(self) -> tuple[dict[str, object], ...]:
        """Return public source world body records for trainer adapters."""

        game = self._require_game()
        if game.world is None:
            return ()
        return tuple(
            {
                "id": body.id,
                "x": _source_number(body.x),
                "y": _source_number(body.y),
                "radius": _source_number(body.radius),
                "avatarId": body.avatar_id,
                "num": body.num,
                "birthMs": _source_number(body.birth_ms),
                "trailLatency": body.trail_latency,
                "breakBefore": body.break_before,
            }
            for body in game.world.iter_unique_bodies()
        )

    def visual_trail_snapshot(self) -> tuple[dict[str, object], ...]:
        """Return browser-like accumulated visual trail point records."""

        return tuple(
            point.to_snapshot() for avatar in self.avatars for point in avatar.visual_trail_points
        )

    def bonus_bodies_snapshot(self) -> tuple[dict[str, object], ...]:
        """Return active map bonus bodies for raw visual observation checks."""

        return tuple(
            {
                "id": bonus.id,
                "type": bonus.type,
                "x": _source_number(bonus.x),
                "y": _source_number(bonus.y),
                "radius": _source_number(bonus.radius),
                "duration": bonus.duration,
            }
            for bonus in self.active_bonuses
        )

    def avatar_body_metadata_snapshot(self) -> tuple[dict[str, object], ...]:
        """Return source body counters needed by trainer-side circle adapters."""

        return tuple(
            {
                "id": avatar.id,
                "bodyNum": avatar.body_num,
                "bodyCount": avatar.body_count,
                "radius": _source_number(avatar.radius),
                "trailLatency": avatar.trail_latency,
                "velocity": _source_number(avatar.velocity),
                "angularVelocity": _source_number(avatar.angular_velocity),
            }
            for avatar in self.avatars
        )

    def avatar_by_id(self, avatar_id: int | str) -> SourceAvatarState:
        for avatar in self.avatars:
            if str(avatar.id) == str(avatar_id):
                return avatar
        raise SourceEnvError(f"unknown avatar id {avatar_id}")

    def set_avatar_state(
        self,
        avatar_id: int | str,
        *,
        x: float | None = None,
        y: float | None = None,
        angle: float | None = None,
        velocity: float | None = None,
    ) -> SourceAvatarState:
        avatar = self.avatar_by_id(avatar_id)
        if x is not None or y is not None:
            avatar.set_position(
                avatar.x if x is None else float(x),
                avatar.y if y is None else float(y),
            )
        if angle is not None:
            avatar.set_angle(float(angle))
        if velocity is not None:
            avatar.set_velocity(float(velocity), self.reference)
        return avatar

    def seed_active_bonus(
        self,
        bonus_type: str,
        *,
        x: float,
        y: float,
    ) -> SourceBonusState:
        """Seed one active map bonus without timers or random selection."""

        game = self._require_game()
        if bonus_type not in _SOURCE_BONUS_DURATIONS_MS:
            raise SourceEnvError(
                f"source bonus seed supports only source default bonus types; got {bonus_type}"
            )
        if game.bonus_world is None:
            game.bonus_world = SourceWorldState(game.size, island_count=1)
        game.bonus_world.activate()
        bonus = SourceBonusState(
            id=self._next_bonus_id,
            type=bonus_type,
            x=float(x),
            y=float(y),
            radius=self.reference.bonus_radius,
            duration=_bonus_duration(bonus_type),
        )
        self._next_bonus_id += 1
        self.active_bonuses.append(bonus)
        game.bonus_world.add_body(bonus.body)
        self._sync_bonus_state()
        return bonus

    def seed_active_bonus_self_small(self, *, x: float, y: float) -> SourceBonusState:
        return self.seed_active_bonus("BonusSelfSmall", x=x, y=y)

    def set_random_sequence(self, random_values: Sequence[float | Mapping[str, Any]]) -> None:
        """Replace the deterministic Math.random tape and clear recorded calls."""

        self.random.sequence = tuple(
            _coerce_random_sequence_value(value, index) for index, value in enumerate(random_values)
        )
        self.random.calls.clear()

    def source_game_on_start(
        self,
        *,
        bonus_types: Sequence[str] | None = None,
        bonus_rate: float | None = None,
    ) -> None:
        """Invoke the source Game.onStart path for forced-state fixtures."""

        if bonus_types is not None or bonus_rate is not None:
            self._configure_bonus_manager(
                bonus_types=self.bonus_types if bonus_types is None else bonus_types,
                bonus_rate=self.bonus_rate if bonus_rate is None else bonus_rate,
            )
        game = self._require_game()
        game.game_start_due_ms = None
        self._start_game()

    def remove_avatar(self, avatar_id: int | str) -> SourceAvatarState:
        """Mirror source Game.removeAvatar for a player leaving an active game."""

        avatar = self.avatar_by_id(avatar_id)
        self._die_avatar_for_removal(avatar)
        self._destroy_avatar(avatar)
        self._record_event("player:leave", {"player": avatar.id})
        self._check_round_end()
        return avatar

    def _build_avatars(
        self,
        *,
        player_count: int,
        players: Sequence[Mapping[str, Any]] | None,
        present: Sequence[bool] | None,
    ) -> list[SourceAvatarState]:
        if players is not None and len(players) != player_count:
            raise SourceEnvError("players length must match player_count")
        if present is not None and len(present) != player_count:
            raise SourceEnvError("present length must match player_count")

        avatars: list[SourceAvatarState] = []
        for index in range(player_count):
            player = players[index] if players is not None else {}
            avatar_id = _coerce_avatar_id(player, index)
            name = str(player.get("name") or player.get("id") or f"p{index}")
            color = str(player.get("color", "#ffffff"))
            is_present = (
                bool(present[index])
                if present is not None
                else bool(player.get("present", player.get("avatarPresent", True)))
            )
            avatar = SourceAvatarState(
                id=avatar_id,
                name=name,
                color=color,
                player_color=color,
                present=is_present,
            )
            avatar.clear_for_round(self.reference)
            if not is_present:
                avatar.alive = False
            avatars.append(avatar)
        return avatars

    def _require_game(self) -> SourceGameState:
        if self.game is None:
            raise RuntimeError("reset must be called before using CurvyTronSourceEnv")
        return self.game

    def _record_event(self, name: str, data: Mapping[str, object] | None = None) -> None:
        self.events.append(
            {
                "order": len(self.events),
                "atMs": _source_number(self.now_ms),
                "event": name,
                "data": dict(data or {}),
            }
        )

    def _sync_bonus_state(self) -> None:
        game = self._require_game()
        game.bonus_count = len(self.active_bonuses)
        game.bonus_world_body_count = (
            game.bonus_world.body_count if game.bonus_world is not None else 0
        )

    def _next_random(self, site: str, avatar_id: int | None) -> float:
        value = self.random.random(at_ms=self.now_ms, site=site, avatar_id=avatar_id)
        self._record_event(
            "random",
            {
                "index": len(self.random.calls) - 1,
                "value": value,
                "site": site,
                "avatar": avatar_id,
            },
        )
        return value

    def _new_round(self, delay_ms: float) -> None:
        game = self._require_game()
        self._record_event("round:new")
        game.started = True
        game.in_round = True
        game.world_exists = True
        if game.world is None:
            game.world = SourceWorldState(game.size)
        if game.bonus_world is None:
            game.bonus_world = SourceWorldState(game.size, island_count=1)
        game.world.clear()
        game.bonus_world.clear()
        self.active_bonuses.clear()
        game.active_bonuses.clear()
        self._next_bonus_id = 1
        game.world_active = False
        game.world_body_count = 0
        game.bonus_count = 0
        game.bonus_world_body_count = 0
        game.frame_scheduled = False
        game.rendered = None
        game.frame_due_ms = None
        game.bonus_pop_due_ms = None
        game.death_ids.clear()
        for avatar in self.avatars:
            if avatar.present:
                self._stop_print_manager_for_round_clear(avatar)
                avatar.clear_for_round(self.reference)
        for avatar in reversed(self.avatars):
            if avatar.present:
                self._spawn_avatar(avatar)
            else:
                game.death_ids.append(avatar.id)
        game.game_start_due_ms = self.now_ms + delay_ms

    def _spawn_avatar(self, avatar: SourceAvatarState) -> None:
        game = self._require_game()
        margin = self.reference.avatar_radius + self.reference.spawn_margin * game.size
        span = game.size - margin * 2.0
        avatar.set_position(
            margin + self._next_random("spawn.position_x", avatar.id) * span,
            margin + self._next_random("spawn.position_y", avatar.id) * span,
        )
        self._record_event(
            "position",
            {
                "avatar": avatar.id,
                "x": _source_number(avatar.x),
                "y": _source_number(avatar.y),
            },
        )
        attempt = 0
        while True:
            angle = self._next_random(f"spawn.angle_attempt_{attempt}", avatar.id) * math.pi * 2
            if _direction_valid(
                angle,
                avatar.x,
                avatar.y,
                tolerance=self.reference.spawn_angle_margin,
                size=game.size,
            ):
                avatar.set_angle(angle)
                self._record_event(
                    "angle",
                    {"avatar": avatar.id, "angle": _source_number(avatar.angle)},
                )
                return
            attempt += 1

    def _start_game(self) -> None:
        game = self._require_game()
        self._record_event("game:start")
        if game.world is None:
            game.world = SourceWorldState(game.size)
        game.world.activate()
        game.world_active = True
        game.world_body_count = game.world.body_count
        game.frame_scheduled = True
        game.rendered = int(self.now_ms)
        game.frame_due_ms = self.now_ms + self.reference.tick_ms
        game.print_start_due_ms = self.now_ms + self.reference.trail_start_delay_ms
        self._start_bonus_manager()

    def _configure_bonus_manager(
        self,
        *,
        bonus_types: Sequence[str],
        bonus_rate: float,
    ) -> None:
        self.bonus_types = _normalize_bonus_types(bonus_types)
        self.bonus_rate = _coerce_bonus_rate(bonus_rate)

    def _start_bonus_manager(self) -> None:
        game = self._require_game()
        if game.bonus_world is None:
            game.bonus_world = SourceWorldState(game.size, island_count=1)
        game.bonus_world.clear()
        game.bonus_world.activate()
        self.active_bonuses.clear()
        self._next_bonus_id = 1
        self._sync_bonus_state()
        game.bonus_pop_due_ms = None
        if self.bonus_types:
            self._schedule_next_bonus_pop("bonus.start_delay")

    def _schedule_next_bonus_pop(self, site: str) -> None:
        game = self._require_game()
        random_value = self._next_random(site, None)
        base_time = self.reference.bonus_base_pop_time_ms
        base_time -= (base_time / 2.0) * self.bonus_rate
        game.bonus_pop_due_ms = self.now_ms + base_time * (1.0 + random_value)

    def _pop_bonus(self) -> None:
        game = self._require_game()
        game.bonus_pop_due_ms = None
        if not self.bonus_types:
            return
        self._schedule_next_bonus_pop("bonus.next_delay_after_pop")
        if len(self.active_bonuses) >= self.reference.bonus_spawn_cap:
            return
        bonus_type = self._select_bonus_type()
        if bonus_type is None:
            return
        x, y = self._natural_bonus_position()
        self._add_natural_bonus(bonus_type, x=x, y=y)

    def _select_bonus_type(self) -> str | None:
        game = self._require_game()
        cumulative_weights: list[float] = []
        weighted_types: list[str] = []
        total_weight = 0.0
        for bonus_type in self.bonus_types:
            probability = _bonus_probability(bonus_type, game, self.avatars)
            if probability > 0.0:
                total_weight += probability
                cumulative_weights.append(total_weight)
                weighted_types.append(bonus_type)

        value = self._next_random("bonus.type", None)
        if not cumulative_weights:
            self._retag_last_random("bonus.type.none")
            return None

        weighted_value = value * cumulative_weights[-1]
        for index, threshold in enumerate(cumulative_weights):
            if weighted_value < threshold:
                selected = weighted_types[index]
                self._retag_last_random(f"bonus.type.{selected}")
                return selected

        self._retag_last_random("bonus.type.none")
        return None

    def _retag_last_random(self, site: str) -> None:
        if not self.random.calls:
            return
        call = self.random.calls[-1]
        label = call["label"]
        if isinstance(label, dict):
            label["site"] = site
        for event in reversed(self.events):
            if event["event"] != "random":
                continue
            data = event["data"]
            if data.get("index") == call["index"]:
                data["site"] = site
                break

    def _natural_bonus_position(self) -> tuple[float, float]:
        game = self._require_game()
        if game.world is None or game.bonus_world is None:
            raise SourceEnvError("natural bonus spawn requires game and bonus worlds")
        margin = self.reference.bonus_radius + 0.01 * game.size
        span = game.size - margin * 2.0
        attempt = 0
        while True:
            x_site = "bonus.position.x" if attempt == 0 else f"bonus.position.retry_{attempt}.x"
            y_site = "bonus.position.y" if attempt == 0 else f"bonus.position.retry_{attempt}.y"
            body = SourceBodyState(
                x=margin + self._next_random(x_site, None) * span,
                y=margin + self._next_random(y_site, None) * span,
                radius=margin,
            )
            if game.world.get_body(body) is None and game.bonus_world.get_body(body) is None:
                return body.x, body.y
            attempt += 1

    def _add_natural_bonus(self, bonus_type: str, *, x: float, y: float) -> SourceBonusState:
        game = self._require_game()
        if game.bonus_world is None:
            game.bonus_world = SourceWorldState(game.size, island_count=1)
            game.bonus_world.activate()
        bonus = SourceBonusState(
            id=self._next_bonus_id,
            type=bonus_type,
            x=float(x),
            y=float(y),
            radius=self.reference.bonus_radius,
            duration=_bonus_duration(bonus_type),
        )
        self._next_bonus_id += 1
        self.active_bonuses.append(bonus)
        game.bonus_world.add_body(bonus.body)
        self._sync_bonus_state()
        self._record_event(
            "bonus:pop",
            {
                "bonus": bonus.id,
                "type": bonus.type,
                "x": _source_number(bonus.x),
                "y": _source_number(bonus.y),
            },
        )
        return bonus

    def _run_frame(self) -> None:
        game = self._require_game()
        if not game.frame_scheduled or game.rendered is None:
            return
        rendered_ms = int(self.now_ms)
        step_ms = rendered_ms - game.rendered
        game.frame_due_ms = self.now_ms + self.reference.tick_ms
        game.rendered = rendered_ms
        self._update_game(float(step_ms), require_in_round=False)

    def _advance_rendered_to(self, rendered_ms: int) -> None:
        game = self._require_game()
        if not game.frame_scheduled or game.rendered is None or rendered_ms <= game.rendered:
            return
        elapsed_ms = rendered_ms - game.rendered
        for avatar in self.avatars:
            if avatar.alive:
                avatar.update_position(elapsed_ms)
        game.rendered = rendered_ms

    def _run_delayed_print_starts(self) -> None:
        game = self._require_game()
        if game.print_start_due_ms is None:
            return
        self._advance_rendered_to(int(game.print_start_due_ms) - 1)
        for avatar in reversed(self.avatars):
            self._start_print_manager(avatar)
        game.print_start_due_ms = None

    def _start_print_manager(self, avatar: SourceAvatarState) -> None:
        manager = avatar.print_manager
        self._record_event("print_manager:start", {"avatar": avatar.id})
        if manager.active:
            return
        manager.active = True
        manager.last_x = avatar.x
        manager.last_y = avatar.y
        self._set_printing(avatar, True)
        manager.distance = self._print_random_distance(
            avatar.printing,
            "print_manager.start_distance",
            avatar.id,
        )

    def _stop_print_manager_after_death(self, avatar: SourceAvatarState) -> None:
        manager = avatar.print_manager
        if not manager.active:
            return
        manager.active = False
        self._set_printing(avatar, False)
        self._print_random_distance(False, "print_manager.stop_distance", avatar.id)
        manager.clear()

    def _stop_print_manager_for_round_clear(self, avatar: SourceAvatarState) -> None:
        manager = avatar.print_manager
        if not manager.active:
            return
        manager.active = False
        if avatar.printing:
            avatar.printing = False
            self._record_event(
                "property",
                {"avatar": avatar.id, "property": "printing", "value": False},
            )
        self._print_random_distance(False, "print_manager.stop_distance", avatar.id)
        manager.clear()

    def _print_random_distance(self, printing: bool, site: str, avatar_id: int) -> float:
        value = self._next_random(site, avatar_id)
        if printing:
            return self.reference.print_distance * (0.3 + value * 0.7)
        return self.reference.hole_distance * (0.8 + value * 0.5)

    def _set_printing(self, avatar: SourceAvatarState, printing: bool) -> None:
        new_printing = bool(printing)
        if avatar.printing != new_printing:
            avatar.printing = new_printing
            self._add_point(avatar, important=True)
            if not avatar.printing:
                avatar.trail_point_count = 0
                avatar.trail_last_x = None
                avatar.trail_last_y = None
                avatar.visual_trail_last_x = None
                avatar.visual_trail_last_y = None
        self._record_event(
            "property",
            {"avatar": avatar.id, "property": "printing", "value": avatar.printing},
        )

    def _add_visual_trail_point(self, avatar: SourceAvatarState) -> None:
        break_before = avatar.visual_trail_last_x is None or avatar.visual_trail_last_y is None
        avatar.visual_trail_points.append(
            SourceVisualTrailPoint(
                x=avatar.x,
                y=avatar.y,
                radius=avatar.radius,
                avatar_id=avatar.id,
                break_before=break_before,
            )
        )
        avatar.visual_trail_last_x = avatar.x
        avatar.visual_trail_last_y = avatar.y

    def _add_point(self, avatar: SourceAvatarState, *, important: bool) -> None:
        game = self._require_game()
        break_before = avatar.trail_last_x is None or avatar.trail_last_y is None
        avatar.trail_point_count += 1
        avatar.trail_last_x = avatar.x
        avatar.trail_last_y = avatar.y
        if important:
            self._add_visual_trail_point(avatar)
        if game.started and game.world is not None and game.world.active:
            game.world.add_body(avatar.stored_body(self.now_ms, break_before=break_before))
            game.world_body_count = game.world.body_count
            game.world_active = game.world.active
            avatar.body_count += 1
        if important:
            self._record_event(
                "point",
                {
                    "avatar": avatar.id,
                    "x": _source_number(avatar.x),
                    "y": _source_number(avatar.y),
                    "important": True,
                },
            )

    def _apply_joint_actions(
        self,
        joint_actions: Mapping[int, float | int | str] | Sequence[float | int | str],
    ) -> None:
        if isinstance(joint_actions, Mapping):
            for avatar_id, action in joint_actions.items():
                self.avatar_by_id(int(avatar_id)).update_angular_velocity(_move_factor(action))
            return
        if len(joint_actions) != len(self.avatars):
            raise SourceEnvError("joint_actions length must match avatar count")
        for avatar, action in zip(self.avatars, joint_actions, strict=True):
            avatar.update_angular_velocity(_move_factor(action))

    def _update_game(self, step_ms: float, *, require_in_round: bool = True) -> None:
        game = self._require_game()
        if not game.started or not game.world_active:
            return
        if require_in_round and not game.in_round:
            return
        frame_start_deaths = len(game.death_ids)
        death_in_frame = False
        for avatar in reversed(self.avatars):
            if not avatar.alive:
                continue
            angle_changed = avatar.update_angle(step_ms)
            if angle_changed and self.emit_step_angle_events:
                self._record_event(
                    "angle",
                    {"avatar": avatar.id, "angle": _source_number(avatar.angle)},
                )
            avatar.update_position(step_ms)
            if self.emit_step_position_events:
                self._record_event(
                    "position",
                    {
                        "avatar": avatar.id,
                        "x": _source_number(avatar.x),
                        "y": _source_number(avatar.y),
                    },
                )
            if avatar.printing:
                self._add_visual_trail_point(avatar)
            if avatar.printing and self._is_time_to_draw(avatar):
                self._add_point(avatar, important=False)
            border = _bound_intersect(
                avatar.x,
                avatar.y,
                margin=0.0 if game.borderless else avatar.radius,
                size=game.size,
            )
            if border is not None:
                if game.borderless:
                    avatar.set_position(*_opposite_position(border[0], border[1], game.size))
                    avatar.trail_last_x = None
                    avatar.trail_last_y = None
                    avatar.visual_trail_last_x = None
                    avatar.visual_trail_last_y = None
                    if avatar.printing:
                        self._add_visual_trail_point(avatar)
                else:
                    self._kill_avatar(avatar, None, frame_start_deaths)
                    death_in_frame = True
            elif not avatar.invincible and game.world is not None:
                killer = game.world.get_body(avatar.live_body())
                if killer is not None:
                    self._kill_avatar(avatar, killer, frame_start_deaths)
                    death_in_frame = True
            if avatar.alive:
                self._test_print_manager(avatar)
                self._test_bonus_catch(avatar)
        if death_in_frame:
            self._check_round_end()

    def _check_round_end(self) -> None:
        game = self._require_game()
        if not game.in_round:
            return
        alive_seen = False
        for avatar in reversed(self.avatars):
            if not avatar.alive:
                continue
            if alive_seen:
                return
            alive_seen = True
        self._end_round()

    def _kill_avatar(
        self,
        avatar: SourceAvatarState,
        killer: SourceBodyState | None,
        score: int,
    ) -> None:
        game = self._require_game()
        avatar.alive = False
        avatar.active_bonuses.clear()
        self._add_point(avatar, important=False)
        self._stop_print_manager_after_death(avatar)
        self._record_event(
            "die",
            {
                "avatar": avatar.id,
                "killer": killer.avatar_id if killer is not None else None,
                "old": killer.is_old(self.now_ms) if killer is not None else None,
            },
        )
        game.death_ids.append(avatar.id)
        avatar.round_score += score
        self._record_event(
            "score:round",
            {
                "avatar": avatar.id,
                "score": avatar.score,
                "roundScore": avatar.round_score,
            },
        )

    def _die_avatar_for_removal(self, avatar: SourceAvatarState) -> None:
        avatar.alive = False
        avatar.active_bonuses.clear()
        self._add_point(avatar, important=False)
        self._stop_print_manager_after_death(avatar)
        self._record_event(
            "die",
            {
                "avatar": avatar.id,
                "killer": None,
                "old": None,
            },
        )

    def _destroy_avatar(self, avatar: SourceAvatarState) -> None:
        avatar.clear_for_round(self.reference)
        avatar.present = False
        avatar.alive = False

    def _is_time_to_draw(self, avatar: SourceAvatarState) -> bool:
        if avatar.trail_last_x is None or avatar.trail_last_y is None:
            return True
        return (
            math.hypot(avatar.x - avatar.trail_last_x, avatar.y - avatar.trail_last_y)
            > avatar.radius
        )

    def _test_print_manager(self, avatar: SourceAvatarState) -> None:
        manager = avatar.print_manager
        if not manager.active:
            return
        distance = math.hypot(avatar.x - manager.last_x, avatar.y - manager.last_y)
        manager.distance -= distance
        manager.last_x = avatar.x
        manager.last_y = avatar.y
        if manager.distance <= 0:
            self._set_printing(avatar, not avatar.printing)
            manager.distance = self._print_random_distance(
                avatar.printing,
                "print_manager.toggle_distance",
                avatar.id,
            )

    def _test_bonus_catch(self, avatar: SourceAvatarState) -> None:
        game = self._require_game()
        if game.bonus_world is None or not game.bonus_world.active:
            return
        body = game.bonus_world.get_body(avatar.live_body())
        if body is None or body.bonus_id is None:
            return
        bonus = self._active_bonus_by_id(body.bonus_id)
        if bonus is None:
            return
        self._remove_active_bonus(bonus)
        if bonus.type.startswith("BonusSelf"):
            self._apply_bonus_to_avatars(bonus, [avatar] if avatar.alive else [])
        elif bonus.type.startswith("BonusEnemy"):
            self._apply_bonus_to_avatars(
                bonus,
                [target for target in self.avatars if target.alive and target.id != avatar.id],
            )
        elif bonus.type.startswith("BonusAll"):
            self._apply_bonus_to_avatars(
                bonus,
                [target for target in self.avatars if target.alive],
            )
        elif bonus.type == "BonusGameBorderless":
            self._apply_bonus_to_game_stack(bonus)
        elif bonus.type == "BonusGameClear":
            self._apply_bonus_game_clear()
        else:
            raise SourceEnvError(f"unsupported caught source bonus type {bonus.type}")

    def _active_bonus_by_id(self, bonus_id: int) -> SourceBonusState | None:
        for bonus in self.active_bonuses:
            if bonus.id == bonus_id:
                return bonus
        return None

    def _remove_active_bonus(self, bonus: SourceBonusState) -> None:
        game = self._require_game()
        if bonus in self.active_bonuses:
            self.active_bonuses.remove(bonus)
        if game.bonus_world is not None:
            game.bonus_world.remove_body(bonus.body)
        self._record_event("bonus:clear", {"bonus": bonus.id})
        self._sync_bonus_state()

    def _apply_bonus_to_avatars(
        self,
        bonus: SourceBonusState,
        targets: Sequence[SourceAvatarState],
    ) -> None:
        bonus.target_ids = tuple(target.id for target in targets)
        bonus.target_id = bonus.target_ids[0] if len(bonus.target_ids) == 1 else None
        if bonus.type == "BonusAllColor":
            bonus.color_avatar_ids = bonus.target_ids
            bonus.color_values = tuple(target.color for target in targets)
        self._schedule_bonus_expiry(bonus)
        for target in reversed(targets):
            target.active_bonuses.append(bonus)
            self._resolve_avatar_bonus_stack(target)
            self._record_event(
                "bonus:stack",
                {
                    "avatar": target.id,
                    "method": "add",
                    "bonus": bonus.to_stack_snapshot(),
                },
            )

    def _apply_bonus_game_clear(self) -> None:
        game = self._require_game()
        if game.world is None:
            return
        game.world.clear()
        game.world.activate()
        game.world_active = game.world.active
        game.world_body_count = game.world.body_count
        for avatar in self.avatars:
            avatar.visual_trail_points.clear()
            avatar.visual_trail_last_x = None
            avatar.visual_trail_last_y = None
        self._record_event("clear")

    def _apply_bonus_to_game_stack(self, bonus: SourceBonusState) -> None:
        game = self._require_game()
        bonus.target_id = 0
        self._schedule_bonus_expiry(bonus)
        game.active_bonuses.append(bonus)
        self._resolve_game_bonus_stack()
        self._record_event(
            "bonus:stack",
            {
                "target": "game",
                "method": "add",
                "bonus": bonus.to_stack_snapshot(),
            },
        )

    def _schedule_bonus_expiry(self, bonus: SourceBonusState) -> None:
        if not bonus.duration:
            return
        order = self._next_bonus_expiry_timer_order
        self._next_bonus_expiry_timer_order += 1
        self._bonus_expiry_timers.append((self.now_ms + float(bonus.duration), order, bonus))

    def _remove_bonus_expiry_timer(self, bonus: SourceBonusState) -> None:
        self._bonus_expiry_timers = [
            timer for timer in self._bonus_expiry_timers if timer[2] is not bonus
        ]

    def _expire_bonus_from_stack(self, bonus: SourceBonusState) -> None:
        if bonus.type == "BonusGameBorderless" and bonus.target_id == 0:
            game = self._require_game()
            if bonus in game.active_bonuses:
                game.active_bonuses.remove(bonus)
                self._resolve_game_bonus_stack(removed_bonus=bonus)
            self._record_event(
                "bonus:stack",
                {
                    "target": "game",
                    "method": "remove",
                    "bonus": bonus.to_stack_snapshot(),
                },
            )
            return
        if not bonus.target_ids:
            if bonus.target_id is not None:
                bonus.target_ids = (bonus.target_id,)
            else:
                return
        for avatar_id in reversed(bonus.target_ids):
            avatar = self.avatar_by_id(avatar_id)
            if bonus in avatar.active_bonuses:
                avatar.active_bonuses.remove(bonus)
                self._resolve_avatar_bonus_stack(avatar, removed_bonus=bonus)
            elif any(
                property_name == "printing"
                for property_name, _value in _bonus_effects_for_avatar(
                    bonus,
                    avatar,
                    self.reference,
                )
            ):
                continue
            self._record_event(
                "bonus:stack",
                {
                    "avatar": avatar.id,
                    "method": "remove",
                    "bonus": bonus.to_stack_snapshot(),
                },
            )

    def _resolve_avatar_bonus_stack(
        self,
        avatar: SourceAvatarState,
        *,
        removed_bonus: SourceBonusState | None = None,
    ) -> None:
        properties: dict[str, object] = {}
        if removed_bonus is not None:
            for property_name, _value in reversed(
                _bonus_effects_for_avatar(removed_bonus, avatar, self.reference)
            ):
                properties[property_name] = _avatar_bonus_default(
                    avatar,
                    property_name,
                    self.reference,
                )
        for active_bonus in reversed(avatar.active_bonuses):
            for property_name, value in reversed(
                _bonus_effects_for_avatar(active_bonus, avatar, self.reference)
            ):
                if property_name not in properties:
                    properties[property_name] = _avatar_bonus_default(
                        avatar,
                        property_name,
                        self.reference,
                    )
                properties[property_name] = _append_avatar_bonus_property(
                    properties[property_name],
                    property_name,
                    value,
                )
        for property_name, value in properties.items():
            self._apply_avatar_bonus_property(avatar, property_name, value)

    def _resolve_game_bonus_stack(
        self,
        *,
        removed_bonus: SourceBonusState | None = None,
    ) -> None:
        game = self._require_game()
        properties: dict[str, object] = {}
        if removed_bonus is not None:
            for property_name, _value in reversed(_bonus_effects_for_game(removed_bonus)):
                properties[property_name] = 0
        for active_bonus in reversed(game.active_bonuses):
            for property_name, value in reversed(_bonus_effects_for_game(active_bonus)):
                if property_name not in properties:
                    properties[property_name] = 0
                properties[property_name] = _append_game_bonus_property(
                    properties[property_name],
                    property_name,
                    value,
                )
        for property_name, value in properties.items():
            self._apply_game_bonus_property(property_name, value)

    def _apply_avatar_bonus_property(
        self,
        avatar: SourceAvatarState,
        property_name: str,
        value: object,
    ) -> None:
        if property_name == "radius":
            self._set_avatar_radius(
                avatar,
                self.reference.avatar_radius * math.pow(2.0, float(value)),
            )
            return
        if property_name == "velocity":
            self._set_avatar_velocity(avatar, float(value))
            return
        if property_name == "inverse":
            self._set_avatar_inverse(avatar, int(value) % 2 != 0)
            return
        if property_name == "invincible":
            self._set_avatar_invincible(avatar, bool(value))
            return
        if property_name == "printing":
            if float(value) > 0:
                self._start_print_manager_from_bonus_stack(avatar)
            else:
                self._stop_print_manager_from_bonus_stack(avatar)
            return
        if property_name == "color":
            self._set_avatar_color(avatar, str(value))
            return
        if property_name == "directionInLoop":
            avatar.direction_in_loop = bool(value)
            return
        if property_name == "angularVelocityBase":
            avatar.angular_velocity_base = float(value)
            return
        setattr(avatar, property_name, value)

    def _apply_game_bonus_property(self, property_name: str, value: object) -> None:
        if property_name == "borderless":
            self._set_game_borderless(bool(value))
            return
        setattr(self._require_game(), property_name, value)

    def _set_avatar_radius(self, avatar: SourceAvatarState, radius: float) -> None:
        new_radius = max(float(radius), self.reference.avatar_radius / 8.0)
        if avatar.radius == new_radius:
            return
        avatar.radius = new_radius
        self._record_event(
            "property",
            {
                "avatar": avatar.id,
                "property": "radius",
                "value": _source_number(avatar.radius),
            },
        )

    def _set_avatar_velocity(self, avatar: SourceAvatarState, velocity: float) -> None:
        raw_velocity = float(velocity)
        if avatar.velocity == raw_velocity:
            return
        new_velocity = max(
            raw_velocity,
            self.reference.avatar_velocity_units_per_s / 2.0,
        )
        if avatar.velocity != new_velocity:
            avatar.velocity = new_velocity
            avatar.update_velocities(self.reference)
        self._record_event(
            "property",
            {
                "avatar": avatar.id,
                "property": "velocity",
                "value": _source_number(avatar.velocity),
            },
        )

    def _set_avatar_inverse(self, avatar: SourceAvatarState, inverse: bool) -> None:
        if avatar.inverse != bool(inverse):
            avatar.inverse = bool(inverse)
            avatar.update_angular_velocity()
        self._record_event(
            "property",
            {"avatar": avatar.id, "property": "inverse", "value": avatar.inverse},
        )

    def _set_avatar_invincible(self, avatar: SourceAvatarState, invincible: bool) -> None:
        avatar.invincible = bool(invincible)
        self._record_event(
            "property",
            {
                "avatar": avatar.id,
                "property": "invincible",
                "value": avatar.invincible,
            },
        )

    def _set_avatar_color(self, avatar: SourceAvatarState, color: str) -> None:
        avatar.color = str(color)
        self._record_event(
            "property",
            {"avatar": avatar.id, "property": "color", "value": avatar.color},
        )

    def _set_game_borderless(self, borderless: bool) -> None:
        game = self._require_game()
        if game.borderless == bool(borderless):
            return
        game.borderless = bool(borderless)
        self._record_event("borderless", {"value": game.borderless})

    def _start_print_manager_from_bonus_stack(self, avatar: SourceAvatarState) -> None:
        manager = avatar.print_manager
        if manager.active:
            return
        manager.active = True
        manager.last_x = avatar.x
        manager.last_y = avatar.y
        self._set_printing(avatar, True)
        manager.distance = self._print_random_distance(
            avatar.printing,
            "print_manager.start_distance",
            avatar.id,
        )

    def _stop_print_manager_from_bonus_stack(self, avatar: SourceAvatarState) -> None:
        manager = avatar.print_manager
        if not manager.active:
            return
        manager.active = False
        self._set_printing(avatar, False)
        self._print_random_distance(False, "print_manager.stop_distance", avatar.id)
        manager.clear()

    def _end_round(self) -> None:
        game = self._require_game()
        if len(self.avatars) == 1:
            winner = self.avatars[0]
        else:
            winner = next((avatar for avatar in self.avatars if avatar.alive), None)
        if winner is not None:
            winner.round_score += max(len(self.avatars) - 1, 1)
            self._record_event(
                "score:round",
                {
                    "avatar": winner.id,
                    "score": winner.score,
                    "roundScore": winner.round_score,
                },
            )
        for avatar in reversed(self.avatars):
            self._record_event(
                "score",
                {
                    "avatar": avatar.id,
                    "score": avatar.score + avatar.round_score,
                    "roundScore": avatar.round_score,
                },
            )
            avatar.score += avatar.round_score
            avatar.round_score = 0
        game.in_round = False
        self._record_event("round:end", {"winner": winner.id if winner is not None else None})
        game.game_stop_due_ms = self.now_ms + self.reference.round_warmdown_ms

    def _stop_game(self) -> None:
        game = self._require_game()
        self._record_event("game:stop")
        game.frame_scheduled = False
        game.frame_due_ms = None
        game.rendered = None
        game.game_stop_due_ms = None
        game.bonus_pop_due_ms = None
        self._resize_world_to_present_avatars()
        if self._is_won():
            self._end_game()
        else:
            self._new_round(self.reference.round_warmup_ms)

    def _resize_world_to_present_avatars(self) -> None:
        game = self._require_game()
        present_count = sum(1 for avatar in self.avatars if avatar.present)
        if present_count < 1:
            return
        size = self.reference.arena_size_for_players(present_count)
        if game.size == size:
            return
        game.size = size
        game.world = SourceWorldState(game.size)
        game.bonus_world = SourceWorldState(game.size, island_count=1)
        game.world_active = False
        game.world_body_count = 0
        game.bonus_world_body_count = 0

    def _is_won(self) -> SourceAvatarState | bool | None:
        game = self._require_game()
        present = [avatar for avatar in self.avatars if avatar.present]
        if len(present) <= 0:
            return True
        if len(self.avatars) > 1 and len(present) <= 1:
            return True
        winners = [
            avatar for avatar in self.avatars if avatar.present and avatar.score >= game.max_score
        ]
        if not winners:
            return None
        if len(winners) == 1:
            return winners[0]
        winners.sort(key=lambda avatar: avatar.score, reverse=True)
        if winners[0].score == winners[1].score:
            return None
        return winners[0]

    def _end_game(self) -> None:
        game = self._require_game()
        if not game.started:
            return
        game.started = False
        self._record_event("end")
        self.avatars.clear()
        game.world_exists = False
        game.world_active = False
        game.world_body_count = 0
        if game.world is not None:
            game.world.clear()
        game.world = None
        self.active_bonuses.clear()
        game.active_bonuses.clear()
        game.bonus_count = 0
        if game.bonus_world is not None:
            game.bonus_world.clear()
        game.bonus_world_body_count = 0


def _coerce_avatar_id(player: Mapping[str, Any], index: int) -> int:
    raw = player.get("avatar_id", player.get("avatarId", player.get("id", index + 1)))
    if isinstance(raw, str) and raw.startswith("p") and raw[1:].isdigit():
        return int(raw[1:]) + 1
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise SourceEnvError(f"players[{index}] avatar id must be an integer")
    return raw


def _move_factor(action: float | int | str) -> float:
    if isinstance(action, str):
        normalized = action.lower()
        if normalized in {"left", "l"}:
            return -1.0
        if normalized in {"right", "r"}:
            return 1.0
        if normalized in {"straight", "none", "noop", "0"}:
            return 0.0
        raise SourceEnvError(f"unknown source move action {action!r}")
    if isinstance(action, bool):
        raise SourceEnvError("boolean actions are not valid source move factors")
    return float(action)


def _direction_valid(angle: float, x: float, y: float, *, tolerance: float, size: float) -> bool:
    quarter = math.pi / 2.0
    margin = tolerance * size
    for border in range(4):
        start = quarter * border
        end = quarter * (border + 1)
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


def _bound_intersect(
    x: float, y: float, *, margin: float, size: float
) -> tuple[float, float] | None:
    if x - margin < 0:
        return (0, y)
    if x + margin > size:
        return (size, y)
    if y - margin < 0:
        return (x, 0)
    if y + margin > size:
        return (x, size)
    return None


def _opposite_position(x: float, y: float, size: float) -> tuple[float, float]:
    if x == 0:
        return (size, y)
    if x == size:
        return (0, y)
    if y == 0:
        return (x, size)
    if y == size:
        return (x, 0)
    return (x, y)


def _bodies_touch(body_a: SourceBodyState, body_b: SourceBodyState) -> bool:
    radius = body_a.radius + body_b.radius
    return math.hypot(body_a.x - body_b.x, body_a.y - body_b.y) < radius and body_a.matches(body_b)


def _js_round(value: float) -> int:
    return math.floor(value + 0.5)


def _source_number(value: float) -> int | float:
    rounded = _source_round(float(value))
    nearest = round(rounded)
    if math.isclose(rounded, nearest, rel_tol=0.0, abs_tol=1e-9):
        return int(nearest)
    return rounded


def _source_round(value: float) -> float:
    scale = 10**_ROUND_DIGITS
    return math.floor(value * scale + 0.5) / scale


def _coerce_random_sequence_value(value: float | Mapping[str, Any], index: int) -> float:
    raw: object
    if isinstance(value, Mapping):
        raw = value.get("value")
    else:
        raw = value
    _validate_random_value(raw, f"random value {index}")
    return float(raw)


def _validate_random_value(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SourceEnvError(f"{field} must be a finite number")
    if not math.isfinite(float(value)) or float(value) < 0.0 or float(value) >= 1.0:
        raise SourceEnvError(f"{field} must be in [0, 1)")


def _bonus_duration(bonus_type: str) -> int:
    return _SOURCE_BONUS_DURATIONS_MS.get(
        bonus_type,
        CurvyTronReferenceDefaults().bonus_duration_ms,
    )


def _bonus_snapshot_effects(bonus: SourceBonusState) -> list[list[object]] | None:
    if bonus.type.startswith("BonusGame"):
        effects = _bonus_effects_for_game(bonus)
    else:
        effects = _bonus_effects_for_avatar(bonus, None, CurvyTronReferenceDefaults())
    if not effects:
        return None
    return [[property_name, value] for property_name, value in effects]


def _bonus_effects_for_avatar(
    bonus: SourceBonusState,
    avatar: SourceAvatarState | None,
    reference: CurvyTronReferenceDefaults,
) -> list[tuple[str, object]]:
    base_velocity = reference.avatar_velocity_units_per_s
    if bonus.type == "BonusSelfSmall":
        return [("radius", -1)]
    if bonus.type in {"BonusSelfSlow", "BonusEnemySlow"}:
        return [("velocity", -base_velocity / 2.0)]
    if bonus.type in {"BonusSelfFast", "BonusEnemyFast"}:
        return [("velocity", 0.75 * base_velocity)]
    if bonus.type == "BonusSelfMaster":
        return [("invincible", True), ("printing", -1)]
    if bonus.type == "BonusEnemyBig":
        return [("radius", 1)]
    if bonus.type == "BonusEnemyInverse":
        return [("inverse", 1)]
    if bonus.type == "BonusEnemyStraightAngle":
        return [("directionInLoop", False), ("angularVelocityBase", math.pi / 2.0)]
    if bonus.type == "BonusAllColor":
        return [("color", _bonus_all_color_value(bonus, avatar))]
    return []


def _bonus_effects_for_game(bonus: SourceBonusState) -> list[tuple[str, object]]:
    if bonus.type == "BonusGameBorderless":
        return [("borderless", True)]
    return []


def _bonus_all_color_value(
    bonus: SourceBonusState,
    avatar: SourceAvatarState | None,
) -> str:
    if not bonus.color_values:
        return ""
    avatar_id = avatar.id if avatar is not None else None
    try:
        index = bonus.color_avatar_ids.index(avatar_id)
    except ValueError:
        index = -1
    return bonus.color_values[(index + 1) % len(bonus.color_values)]


def _avatar_bonus_default(
    avatar: SourceAvatarState,
    property_name: str,
    reference: CurvyTronReferenceDefaults,
) -> object:
    if property_name == "printing":
        return 1
    if property_name == "radius":
        return 0
    if property_name == "color":
        return avatar.player_color
    if property_name == "velocity":
        return reference.avatar_velocity_units_per_s
    if property_name == "angularVelocityBase":
        return reference.angular_velocity_radians_per_ms
    if property_name == "inverse":
        return False
    if property_name == "invincible":
        return False
    if property_name == "directionInLoop":
        return True
    return 0


def _append_avatar_bonus_property(
    current: object,
    property_name: str,
    value: object,
) -> object:
    if property_name in {"directionInLoop", "angularVelocityBase", "color"}:
        return value
    return current + value  # type: ignore[operator]


def _append_game_bonus_property(
    current: object,
    property_name: str,
    value: object,
) -> object:
    _ = property_name
    return current + value  # type: ignore[operator]


def _normalize_bonus_types(bonus_types: Sequence[str] | None) -> tuple[str, ...]:
    if bonus_types is None:
        return ()
    normalized = tuple(str(bonus_type) for bonus_type in bonus_types)
    supported = set(CurvyTronReferenceDefaults().default_bonus_types)
    for bonus_type in normalized:
        if bonus_type not in supported:
            raise SourceEnvError(
                "natural source bonus spawn currently supports only default "
                f"source bonus types; got {bonus_type}"
            )
    return normalized


def _bonus_probability(
    bonus_type: str,
    game: SourceGameState,
    avatars: Sequence[SourceAvatarState],
) -> float:
    try:
        base_probability = _SOURCE_DEFAULT_BONUS_PROBABILITIES[bonus_type]
    except KeyError as exc:
        raise SourceEnvError(f"unsupported source bonus type {bonus_type}") from exc
    if bonus_type == "BonusGameClear":
        present_count = sum(1 for avatar in avatars if avatar.present)
        if present_count <= 0:
            return 0.0
        alive_count = sum(1 for avatar in avatars if avatar.alive)
        ratio = 1.0 - alive_count / present_count
        if ratio < 0.5:
            return base_probability
        return _js_round((base_probability - ratio) * 10.0) / 10.0
    return base_probability


def _coerce_bonus_rate(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SourceEnvError("bonus_rate must be a finite number")
    rate = float(value)
    if not math.isfinite(rate) or rate < -1.0 or rate > 1.0:
        raise SourceEnvError("bonus_rate must be in [-1, 1]")
    return rate


__all__ = [
    "SourceBodyState",
    "SourceBonusState",
    "SourceVisualTrailPoint",
    "CurvyTronSourceEnv",
    "SourceAvatarState",
    "SourceEnvError",
    "SourceGameState",
    "SourceWorldState",
    "SourcePrintManagerState",
    "SourceRandomTape",
]
