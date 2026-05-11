import pytest

from curvyzero.env.config import CurvyTronConfig, CurvyTronReferenceDefaults


def test_reference_defaults_expose_source_derived_constants():
    reference = CurvyTronConfig().reference_defaults

    assert reference.tick_ms == pytest.approx(1_000 / 60)
    assert reference.round_warmup_ms == 3_000
    assert reference.round_warmdown_ms == 5_000
    assert reference.trail_start_delay_ms == 3_000
    assert reference.avatar_velocity_units_per_s == 16.0
    assert reference.angular_velocity_radians_per_ms == pytest.approx(0.0028)
    assert reference.avatar_radius == 0.6
    assert reference.trail_latency_points == 3
    assert reference.spawn_margin == 0.05
    assert reference.spawn_angle_margin == 0.3
    assert reference.print_distance == 60.0
    assert reference.hole_distance == 5.0
    assert reference.bonus_radius == 3.0
    assert reference.bonus_duration_ms == 5_000
    assert reference.bonus_spawn_cap == 20
    assert reference.bonus_base_pop_time_ms == 3_000


def test_reference_defaults_expose_source_arena_and_score_formulas():
    reference = CurvyTronReferenceDefaults()

    assert [reference.arena_size_for_players(players) for players in (1, 2, 4, 8)] == [
        80,
        88,
        101,
        124,
    ]
    assert [reference.max_score_for_players(players) for players in (1, 2, 4)] == [1, 10, 30]


def test_reference_arena_formula_uses_javascript_rounding():
    reference = CurvyTronReferenceDefaults(
        arena_base_size=10.0,
        arena_player_area_fraction=0.5625,
    )

    assert reference.arena_size_for_players(2) == 13


def test_reference_defaults_expose_default_bonus_set():
    reference = CurvyTronReferenceDefaults()

    assert reference.default_bonus_types == (
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
    assert "BonusSelfGodzilla" not in reference.default_bonus_types


def test_reference_defaults_do_not_affect_curvyzero_v0_rules_hash():
    default = CurvyTronConfig()
    metadata_variant = CurvyTronConfig(
        reference_defaults=CurvyTronReferenceDefaults(tick_hz=30.0)
    )

    assert metadata_variant.rules_hash == default.rules_hash
