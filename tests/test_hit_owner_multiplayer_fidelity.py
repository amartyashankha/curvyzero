import math

import numpy as np

from curvyzero.env import vector_runtime
from curvyzero.env.source_env import CurvyTronSourceEnv, SourceBodyState
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv


def _source_island_size(map_size: float) -> float:
    island_count = max(1, math.floor(map_size / 40.0 + 0.5))
    return map_size / island_count


def _quiet_source_env(player_count: int) -> CurvyTronSourceEnv:
    env = CurvyTronSourceEnv()
    env.reset(player_count=player_count, warmup_ms=0)
    env.advance_timers(0)
    assert env.game is not None
    size = float(env.game.size)
    positions = (
        (20.0, 20.0),
        (size - 20.0, 20.0),
        (20.0, size - 20.0),
        (size - 20.0, size - 20.0),
    )
    for index, (x, y) in enumerate(positions[:player_count], start=1):
        env.set_avatar_state(index, x=x, y=y, angle=0.0)
    for avatar in env.avatars:
        avatar.printing = False
        avatar.trail_last_x = None
        avatar.trail_last_y = None
        avatar.visual_trail_last_x = None
        avatar.visual_trail_last_y = None
        avatar.visual_trail_points.clear()
        avatar.print_manager.clear()
    env.events.clear()
    env.random.calls.clear()
    return env


def _seed_source_body(
    env: CurvyTronSourceEnv,
    *,
    owner_id: int,
    x: float,
    y: float,
    num: int = 0,
) -> SourceBodyState:
    game = env.game
    assert game is not None
    assert game.world is not None
    owner = env.avatar_by_id(owner_id)
    body = SourceBodyState(
        x=x,
        y=y,
        radius=owner.radius,
        avatar_id=owner.id,
        num=num,
        birth_ms=env.now_ms,
        trail_latency=owner.trail_latency,
    )
    game.world.add_body(body)
    game.world_body_count = game.world.body_count
    game.world_active = game.world.active
    owner.body_count = max(owner.body_count, num + 1)
    return body


def _source_die_events(env: CurvyTronSourceEnv) -> list[dict[str, object]]:
    return [event for event in env.events if event["event"] == "die"]


def _runtime_state(
    *,
    player_count: int,
    body_capacity: int = 16,
    map_size: float | None = None,
) -> dict[str, np.ndarray]:
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=player_count,
        body_capacity=body_capacity,
        map_size=map_size,
    )
    state = {name: array.copy() for name, array in env.reset_template.items()}
    size = float(state["map_size"][0])
    positions = np.asarray(
        [
            [20.0, 20.0],
            [size - 20.0, 20.0],
            [20.0, size - 20.0],
            [size - 20.0, size - 20.0],
        ],
        dtype=np.float64,
    )
    state["started"][0] = True
    state["in_round"][0] = True
    state["world_active"][0] = True
    state["done"][0] = False
    state["terminated"][0] = False
    state["truncated"][0] = False
    state["reset_pending"][0] = False
    state["overflow"][0] = False
    state["alive"][0, :player_count] = True
    state["present"][0, :player_count] = True
    state["pos"][0, :player_count] = positions[:player_count]
    state["prev_pos"][0, :player_count] = positions[:player_count]
    state["heading"][0, :player_count] = 0.0
    state["angular_velocity_per_ms"][0, :player_count] = 0.0
    state["printing"][0, :player_count] = False
    state["print_manager_active"][0, :player_count] = False
    state["timer_active"][0] = False
    state["death_count"][0] = 0
    state["death_player"][0] = -1
    state["death_cause"][0] = vector_runtime.DEATH_CAUSE_NONE
    state["death_hit_owner"][0] = -1
    state["body_active"][0] = False
    state["body_owner"][0] = -1
    state["body_num"][0] = -1
    state["body_insert_tick"][0] = -1
    state["body_insert_kind"][0] = -1
    state["body_break_before"][0] = False
    state["body_write_cursor"][0] = 0
    state["world_body_count"][0] = 0
    state["body_count"][0, :player_count] = 0
    state["live_body_num"][0, :player_count] = 0
    return state


def _seed_vector_body(
    state: dict[str, np.ndarray],
    *,
    owner: int,
    x: float,
    y: float,
    num: int = 0,
) -> None:
    slot = int(state["body_write_cursor"][0])
    state["body_active"][0, slot] = True
    state["body_pos"][0, slot] = [x, y]
    state["body_radius"][0, slot] = state["radius"][0, owner]
    state["body_owner"][0, slot] = owner
    state["body_num"][0, slot] = num
    state["body_insert_tick"][0, slot] = int(state["tick"][0])
    state["body_insert_kind"][0, slot] = vector_runtime.BODY_KIND_NORMAL
    state["body_write_cursor"][0] = slot + 1
    state["world_body_count"][0] += 1
    state["body_count"][0, owner] = max(int(state["body_count"][0, owner]), num + 1)


def _step_vector(state: dict[str, np.ndarray]) -> dict[str, int]:
    player_count = state["alive"].shape[1]
    return vector_runtime.step_many(
        vector_runtime.VectorStepInput(
            state=state,
            step_ms=np.asarray([0.0], dtype=np.float64),
            source_moves=np.zeros((1, player_count), dtype=np.int8),
            player_count=player_count,
            print_manager_mode=np.asarray(["none"], dtype=object),
            event_mode=vector_runtime.EVENT_MODE_NONE,
        )
    )


def test_source_4p_same_frame_overlapping_bodies_report_newest_killer() -> None:
    env = _quiet_source_env(player_count=4)
    victim = env.avatar_by_id(1)
    for owner_id in (2, 3, 4):
        _seed_source_body(env, owner_id=owner_id, x=victim.x, y=victim.y)

    env.step({}, elapsed_ms=0)

    assert env.avatar_by_id(1).alive is False
    assert [event["data"] for event in _source_die_events(env)] == [
        {"avatar": 1, "killer": 4, "old": False},
    ]


def test_vector_4p_same_frame_overlapping_bodies_use_newest_hit_owner() -> None:
    state = _runtime_state(player_count=4)
    victim_x, victim_y = state["pos"][0, 0]
    for owner in (1, 2, 3):
        _seed_vector_body(state, owner=owner, x=float(victim_x), y=float(victim_y))

    counters = _step_vector(state)

    np.testing.assert_array_equal(state["alive"], np.asarray([[False, True, True, True]]))
    np.testing.assert_array_equal(
        state["death_player"], np.asarray([[0, -1, -1, -1]], dtype=np.int16)
    )
    np.testing.assert_array_equal(
        state["death_cause"],
        np.asarray(
            [
                [
                    vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
                    vector_runtime.DEATH_CAUSE_NONE,
                    vector_runtime.DEATH_CAUSE_NONE,
                    vector_runtime.DEATH_CAUSE_NONE,
                ]
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        state["death_hit_owner"], np.asarray([[3, -1, -1, -1]], dtype=np.int16)
    )
    assert counters["body_hits"] == 1
    assert counters["death_points_inserted"] == 1
    assert counters["normal_wall_deaths"] == 0


def test_source_4p_corner_island_order_beats_newer_later_island_body() -> None:
    env = _quiet_source_env(player_count=4)
    assert env.game is not None
    boundary = _source_island_size(float(env.game.size))
    offset = env.avatar_by_id(1).radius + 0.05
    env.set_avatar_state(1, x=boundary, y=boundary, angle=0.0)
    _seed_source_body(env, owner_id=2, x=boundary - offset, y=boundary - offset)
    _seed_source_body(env, owner_id=3, x=boundary + offset, y=boundary - offset)
    _seed_source_body(env, owner_id=4, x=boundary - offset, y=boundary + offset)

    env.step({}, elapsed_ms=0)

    assert env.avatar_by_id(1).alive is False
    assert [event["data"] for event in _source_die_events(env)] == [
        {"avatar": 1, "killer": 2, "old": False},
    ]


def test_vector_4p_corner_island_order_beats_newer_later_island_body() -> None:
    state = _runtime_state(player_count=4)
    boundary = _source_island_size(float(state["map_size"][0]))
    offset = float(state["radius"][0, 0]) + 0.05
    state["pos"][0, 0] = [boundary, boundary]
    state["prev_pos"][0, 0] = [boundary, boundary]
    _seed_vector_body(state, owner=1, x=boundary - offset, y=boundary - offset)
    _seed_vector_body(state, owner=2, x=boundary + offset, y=boundary - offset)
    _seed_vector_body(state, owner=3, x=boundary - offset, y=boundary + offset)

    counters = _step_vector(state)

    np.testing.assert_array_equal(state["alive"], np.asarray([[False, True, True, True]]))
    np.testing.assert_array_equal(
        state["death_player"], np.asarray([[0, -1, -1, -1]], dtype=np.int16)
    )
    np.testing.assert_array_equal(
        state["death_cause"],
        np.asarray(
            [
                [
                    vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
                    vector_runtime.DEATH_CAUSE_NONE,
                    vector_runtime.DEATH_CAUSE_NONE,
                    vector_runtime.DEATH_CAUSE_NONE,
                ]
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        state["death_hit_owner"], np.asarray([[1, -1, -1, -1]], dtype=np.int16)
    )
    assert counters["body_hits"] == 1
    assert counters["death_points_inserted"] == 1
    assert counters["normal_wall_deaths"] == 0


def test_source_3p_newest_own_body_still_respects_trail_latency() -> None:
    env = _quiet_source_env(player_count=3)
    victim = env.avatar_by_id(1)
    _seed_source_body(env, owner_id=2, x=victim.x, y=victim.y, num=0)
    _seed_source_body(env, owner_id=1, x=victim.x, y=victim.y, num=0)
    victim.body_count = victim.trail_latency

    env.step({}, elapsed_ms=0)

    assert env.avatar_by_id(1).alive is False
    assert [event["data"] for event in _source_die_events(env)] == [
        {"avatar": 1, "killer": 2, "old": False},
    ]


def test_vector_3p_newest_own_body_still_respects_trail_latency() -> None:
    state = _runtime_state(player_count=3)
    victim_x, victim_y = state["pos"][0, 0]
    _seed_vector_body(state, owner=1, x=float(victim_x), y=float(victim_y), num=0)
    _seed_vector_body(state, owner=0, x=float(victim_x), y=float(victim_y), num=0)
    state["body_count"][0, 0] = state["trail_latency"][0, 0]

    counters = _step_vector(state)

    np.testing.assert_array_equal(state["alive"], np.asarray([[False, True, True]]))
    np.testing.assert_array_equal(
        state["death_player"], np.asarray([[0, -1, -1]], dtype=np.int16)
    )
    np.testing.assert_array_equal(
        state["death_cause"],
        np.asarray(
            [
                [
                    vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
                    vector_runtime.DEATH_CAUSE_NONE,
                    vector_runtime.DEATH_CAUSE_NONE,
                ]
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        state["death_hit_owner"], np.asarray([[1, -1, -1]], dtype=np.int16)
    )
    assert counters["body_hits"] == 1
    assert counters["death_points_inserted"] == 1
    assert counters["normal_wall_deaths"] == 0


def test_vector_4p_two_victim_death_metadata_keeps_hit_owners_aligned() -> None:
    state = _runtime_state(player_count=4)
    victim_zero_x, victim_zero_y = state["pos"][0, 0]
    victim_two_x, victim_two_y = state["pos"][0, 2]
    _seed_vector_body(
        state,
        owner=1,
        x=float(victim_two_x),
        y=float(victim_two_y),
    )
    _seed_vector_body(
        state,
        owner=3,
        x=float(victim_zero_x),
        y=float(victim_zero_y),
    )

    counters = _step_vector(state)

    np.testing.assert_array_equal(state["alive"], np.asarray([[False, True, False, True]]))
    np.testing.assert_array_equal(
        state["death_player"], np.asarray([[2, 0, -1, -1]], dtype=np.int16)
    )
    np.testing.assert_array_equal(
        state["death_cause"],
        np.asarray(
            [
                [
                    vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
                    vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
                    vector_runtime.DEATH_CAUSE_NONE,
                    vector_runtime.DEATH_CAUSE_NONE,
                ]
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        state["death_hit_owner"], np.asarray([[1, 3, -1, -1]], dtype=np.int16)
    )
    assert counters["body_hits"] == 2
    assert counters["death_points_inserted"] == 2
    assert counters["normal_wall_deaths"] == 0
