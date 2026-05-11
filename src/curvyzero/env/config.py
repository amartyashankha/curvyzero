"""Configuration for deterministic CurvyTron-like rulesets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from dataclasses import field
from math import sqrt
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class CurvyTronReferenceDefaults:
    """Source-derived constants from the inspected CurvyTron v1 codebase.

    These values are fidelity metadata for docs, tests, and future rulesets.
    The current ``curvyzero-v0`` simulator does not consume them directly.
    """

    tick_hz: float = 60.0
    round_warmup_ms: int = 3_000
    round_warmdown_ms: int = 5_000
    trail_start_delay_ms: int = 3_000
    arena_base_size: float = 80.0
    arena_player_area_fraction: float = 1.0 / 5.0
    avatar_velocity_units_per_s: float = 16.0
    angular_velocity_radians_per_ms: float = 2.8 / 1000.0
    avatar_radius: float = 0.6
    trail_latency_points: int = 3
    spawn_margin: float = 0.05
    spawn_angle_margin: float = 0.3
    print_distance: float = 60.0
    hole_distance: float = 5.0
    default_bonus_rate: float = 0.0
    bonus_radius: float = 3.0
    bonus_duration_ms: int = 5_000
    bonus_spawn_cap: int = 20
    bonus_base_pop_time_ms: int = 3_000
    default_bonus_types: tuple[str, ...] = (
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

    @property
    def tick_ms(self) -> float:
        return 1_000.0 / self.tick_hz

    def arena_size_for_players(self, players: int) -> int:
        if players < 1:
            raise ValueError("players must be at least 1")
        base_area = self.arena_base_size**2
        extra_area = (players - 1) * base_area * self.arena_player_area_fraction
        return int(sqrt(base_area + extra_area) + 0.5)

    def max_score_for_players(self, players: int) -> int:
        if players < 1:
            raise ValueError("players must be at least 1")
        return max(1, (players - 1) * 10)


@dataclass(frozen=True, slots=True)
class CurvyTronConfig:
    """Rules and geometry for the first 1v1 no-bonus simulator."""

    _RULE_HASH_FIELDS: ClassVar[tuple[str, ...]] = (
        "ruleset",
        "rule_provenance",
        "width",
        "height",
        "players",
        "max_ticks",
        "action_repeat",
        "speed",
        "turn_rate_radians",
        "trail_radius",
        "trail_gap_period",
        "trail_gap_length",
        "spawn_margin",
        "allow_straight_action",
    )

    ruleset: str = "curvyzero-v0"
    rule_provenance: str = "v0-choice"
    width: int = 64
    height: int = 64
    players: int = 2
    max_ticks: int = 2_000
    action_repeat: int = 4
    speed: float = 1.0
    turn_rate_radians: float = 0.08
    trail_radius: float = 1.5
    trail_gap_period: int | None = None
    trail_gap_length: int = 0
    spawn_margin: float = 12.0
    allow_straight_action: bool = True
    reference_defaults: CurvyTronReferenceDefaults = field(
        default_factory=CurvyTronReferenceDefaults
    )

    @property
    def action_count(self) -> int:
        return 3 if self.allow_straight_action else 2

    @property
    def rules_hash(self) -> str:
        """Stable short hash for behavior-affecting config fields."""

        behavior_payload = {name: getattr(self, name) for name in self._RULE_HASH_FIELDS}
        payload = json.dumps(behavior_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
