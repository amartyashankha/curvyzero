import math

import numpy as np

from curvyzero.env import vector_runtime
from curvyzero.env.source_env import CurvyTronSourceEnv, SourceBodyState
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv


_TUNNEL_SINGLE_STEP_MS = 300.0
_SOURCE_SUBSTEP_MS = 1000.0 / 60.0
_TUNNEL_START_X = 20.0
_TUNNEL_BODY_X = 22.0
_TUNNEL_Y = 20.0
_TUNNEL_KILL_SUBSTEP = 4
_TUNNEL_SINGLE_STEP_X = (
    _TUNNEL_START_X
    + vector_runtime.SOURCE_AVATAR_SPEED * _TUNNEL_SINGLE_STEP_MS / 1000.0
)
_TUNNEL_SUBSTEP_KILL_X = (
    _TUNNEL_START_X
    + _TUNNEL_KILL_SUBSTEP
    * vector_runtime.SOURCE_AVATAR_SPEED
    * _SOURCE_SUBSTEP_MS
    / 1000.0
)


def _quiet_source_env(*, borderless: bool = False) -> CurvyTronSourceEnv:
    env = CurvyTronSourceEnv()
    env.reset(player_count=2, warmup_ms=0, borderless=borderless)
    env.advance_timers(0)
    env.set_avatar_state(1, x=20, y=20, angle=0)
    env.set_avatar_state(2, x=60, y=60, angle=math.pi)
    for avatar in env.avatars:
        avatar.printing = False
        avatar.trail_last_x = None
        avatar.trail_last_y = None
        avatar.visual_trail_last_x = None
        avatar.visual_trail_last_y = None
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


def _vector_state(*, borderless: bool = False, body_capacity: int = 8) -> dict[str, np.ndarray]:
    return {
        "tick": np.asarray([0], dtype=np.int32),
        "done": np.asarray([False], dtype=bool),
        "terminated": np.asarray([False], dtype=bool),
        "reset_pending": np.asarray([False], dtype=bool),
        "overflow": np.asarray([False], dtype=bool),
        "started": np.asarray([True], dtype=bool),
        "in_round": np.asarray([True], dtype=bool),
        "world_active": np.asarray([True], dtype=bool),
        "world_body_count": np.asarray([0], dtype=np.int32),
        "present": np.asarray([[True, True]], dtype=bool),
        "alive": np.asarray([[True, True]], dtype=bool),
        "pos": np.asarray([[[20.0, 20.0], [60.0, 60.0]]], dtype=np.float64),
        "prev_pos": np.asarray([[[20.0, 20.0], [60.0, 60.0]]], dtype=np.float64),
        "heading": np.asarray([[0.0, math.pi]], dtype=np.float64),
        "angular_velocity_per_ms": np.zeros((1, 2), dtype=np.float64),
        "speed": np.full((1, 2), vector_runtime.SOURCE_AVATAR_SPEED, dtype=np.float64),
        "inverse": np.asarray([[False, False]], dtype=bool),
        "radius": np.full((1, 2), vector_runtime.SOURCE_AVATAR_RADIUS, dtype=np.float64),
        "trail_latency": np.full((1, 2), 3, dtype=np.int32),
        "map_size": np.asarray([88.0], dtype=np.float64),
        "borderless": np.asarray([borderless], dtype=bool),
        "invincible": np.asarray([[False, False]], dtype=bool),
        "printing": np.asarray([[False, False]], dtype=bool),
        "print_manager_active": np.asarray([[False, False]], dtype=bool),
        "print_manager_distance": np.zeros((1, 2), dtype=np.float64),
        "print_manager_last_pos": np.zeros((1, 2, 2), dtype=np.float64),
        "body_active": np.zeros((1, body_capacity), dtype=bool),
        "body_pos": np.zeros((1, body_capacity, 2), dtype=np.float64),
        "body_radius": np.zeros((1, body_capacity), dtype=np.float64),
        "body_owner": np.full((1, body_capacity), -1, dtype=np.int16),
        "body_num": np.full((1, body_capacity), -1, dtype=np.int32),
        "body_insert_tick": np.full((1, body_capacity), -1, dtype=np.int32),
        "body_insert_kind": np.full((1, body_capacity), -1, dtype=np.int16),
        "body_write_cursor": np.asarray([0], dtype=np.int32),
        "body_count": np.zeros((1, 2), dtype=np.int32),
        "live_body_num": np.zeros((1, 2), dtype=np.int32),
        "body_overflow": np.asarray([False], dtype=bool),
        "body_break_before": np.zeros((1, body_capacity), dtype=bool),
        "visible_trail_count": np.zeros((1, 2), dtype=np.int32),
        "has_visible_trail_last": np.asarray([[False, False]], dtype=bool),
        "visible_trail_last_pos": np.zeros((1, 2, 2), dtype=np.float64),
        "has_draw_cursor": np.asarray([[False, False]], dtype=bool),
        "draw_cursor_pos": np.zeros((1, 2, 2), dtype=np.float64),
        "visual_trail_active": np.zeros((1, body_capacity), dtype=bool),
        "visual_trail_pos": np.zeros((1, body_capacity, 2), dtype=np.float64),
        "visual_trail_radius": np.zeros((1, body_capacity), dtype=np.float64),
        "visual_trail_owner": np.full((1, body_capacity), -1, dtype=np.int16),
        "visual_trail_break_before": np.zeros((1, body_capacity), dtype=bool),
        "visual_trail_write_cursor": np.asarray([0], dtype=np.int32),
        "visual_trail_overflow": np.asarray([False], dtype=bool),
        "has_visual_trail_last": np.asarray([[False, False]], dtype=bool),
        "visual_trail_last_pos": np.zeros((1, 2, 2), dtype=np.float64),
        "death_tick": np.full((1, 2), -1, dtype=np.int32),
        "death_count": np.asarray([0], dtype=np.int32),
        "death_player": np.full((1, 2), -1, dtype=np.int16),
        "death_cause": np.full((1, 2), vector_runtime.DEATH_CAUSE_NONE, dtype=np.int16),
        "death_hit_owner": np.full((1, 2), -1, dtype=np.int16),
        "score": np.zeros((1, 2), dtype=np.int32),
        "round_score": np.zeros((1, 2), dtype=np.int32),
        "terminal_reason": np.asarray([vector_runtime.TERMINAL_REASON_NONE], dtype=np.int16),
        "draw": np.asarray([False], dtype=bool),
        "winner": np.asarray([-1], dtype=np.int16),
        "random_tape_values": np.zeros((1, 1), dtype=np.float64),
        "random_tape_length": np.asarray([0], dtype=np.int32),
        "random_tape_cursor": np.asarray([0], dtype=np.int32),
        "random_tape_draw_count": np.asarray([0], dtype=np.int32),
        "random_tape_exhausted": np.asarray([False], dtype=bool),
        "event_count": np.asarray([0], dtype=np.int32),
        "event_overflow_attempts": np.asarray([0], dtype=np.int32),
    }


def _seed_vector_body(
    state: dict[str, np.ndarray],
    *,
    owner: int,
    x: float,
    y: float,
    num: int = 0,
    break_before: bool = False,
) -> None:
    slot = int(state["body_write_cursor"][0])
    state["body_active"][0, slot] = True
    state["body_pos"][0, slot] = [x, y]
    state["body_radius"][0, slot] = state["radius"][0, owner]
    state["body_owner"][0, slot] = owner
    state["body_num"][0, slot] = num
    state["body_insert_tick"][0, slot] = int(state["tick"][0])
    state["body_insert_kind"][0, slot] = vector_runtime.BODY_KIND_NORMAL
    state["body_break_before"][0, slot] = break_before
    state["body_write_cursor"][0] = slot + 1
    state["world_body_count"][0] += 1
    state["body_count"][0, owner] = max(int(state["body_count"][0, owner]), num + 1)


def _seed_vector_visual_point(
    state: dict[str, np.ndarray],
    *,
    owner: int,
    x: float,
    y: float,
    break_before: bool,
) -> None:
    slot = int(state["visual_trail_write_cursor"][0])
    state["visual_trail_active"][0, slot] = True
    state["visual_trail_pos"][0, slot] = [x, y]
    state["visual_trail_radius"][0, slot] = state["radius"][0, owner]
    state["visual_trail_owner"][0, slot] = owner
    state["visual_trail_break_before"][0, slot] = break_before
    state["visual_trail_write_cursor"][0] = slot + 1
    state["has_visual_trail_last"][0, owner] = True
    state["visual_trail_last_pos"][0, owner] = [x, y]


def _step_vector(state: dict[str, np.ndarray], *, step_ms: float = 0.0) -> dict[str, int]:
    return vector_runtime.step_many(
        vector_runtime.VectorStepInput(
            state=state,
            step_ms=np.asarray([step_ms], dtype=np.float64),
            source_moves=np.zeros((1, 2), dtype=np.int8),
            player_count=2,
            print_manager_mode=np.asarray(["none"], dtype=object),
            event_mode=vector_runtime.EVENT_MODE_NONE,
        )
    )


def test_source_2p_head_hits_opponent_stored_body_point() -> None:
    env = _quiet_source_env()
    _seed_source_body(env, owner_id=2, x=20, y=20)

    env.step({}, elapsed_ms=0)

    assert env.avatar_by_id(1).alive is False
    assert _source_die_events(env)[0]["data"] == {"avatar": 1, "killer": 2, "old": False}


def test_vector_2p_head_hits_opponent_stored_body_point() -> None:
    state = _vector_state()
    _seed_vector_body(state, owner=1, x=20, y=20)

    counters = _step_vector(state)

    np.testing.assert_array_equal(state["alive"], np.asarray([[False, True]]))
    np.testing.assert_array_equal(state["death_player"], np.asarray([[0, -1]], dtype=np.int16))
    np.testing.assert_array_equal(
        state["death_cause"],
        np.asarray(
            [[vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL, vector_runtime.DEATH_CAUSE_NONE]],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(state["death_hit_owner"], np.asarray([[1, -1]], dtype=np.int16))
    assert counters["body_hits"] == 1


def test_source_2p_tangent_stored_body_circle_is_safe() -> None:
    env = _quiet_source_env()
    _seed_source_body(env, owner_id=2, x=21.200000000000003, y=20)

    frame = env.step({}, elapsed_ms=0)

    assert env.avatar_by_id(1).alive is True
    assert frame["game"]["worldBodyCount"] == 1
    assert _source_die_events(env) == []


def test_vector_2p_tangent_stored_body_circle_is_safe() -> None:
    state = _vector_state()
    _seed_vector_body(state, owner=1, x=21.200000000000003, y=20)

    counters = _step_vector(state)

    np.testing.assert_array_equal(state["alive"], np.asarray([[True, True]]))
    np.testing.assert_array_equal(state["death_player"], np.asarray([[-1, -1]], dtype=np.int16))
    np.testing.assert_array_equal(state["world_body_count"], np.asarray([1], dtype=np.int32))
    assert counters["body_hits"] == 0


def test_source_2p_large_single_step_tunnels_past_opponent_stored_body() -> None:
    env = _quiet_source_env()
    env.set_avatar_state(1, x=_TUNNEL_START_X, y=_TUNNEL_Y, angle=0)
    _seed_source_body(env, owner_id=2, x=_TUNNEL_BODY_X, y=_TUNNEL_Y)

    env.step({}, elapsed_ms=_TUNNEL_SINGLE_STEP_MS)

    avatar = env.avatar_by_id(1)
    assert avatar.alive is True
    assert math.isclose(avatar.x, _TUNNEL_SINGLE_STEP_X)
    assert math.isclose(avatar.y, _TUNNEL_Y)
    assert _source_die_events(env) == []


def test_source_2p_source_sized_substeps_hit_opponent_stored_body() -> None:
    env = _quiet_source_env()
    env.set_avatar_state(1, x=_TUNNEL_START_X, y=_TUNNEL_Y, angle=0)
    _seed_source_body(env, owner_id=2, x=_TUNNEL_BODY_X, y=_TUNNEL_Y)

    death_substep = None
    for substep in range(1, 1 + int(_TUNNEL_SINGLE_STEP_MS / _SOURCE_SUBSTEP_MS)):
        env.step({}, elapsed_ms=_SOURCE_SUBSTEP_MS)
        if not env.avatar_by_id(1).alive:
            death_substep = substep
            break

    avatar = env.avatar_by_id(1)
    assert death_substep == _TUNNEL_KILL_SUBSTEP
    assert avatar.alive is False
    assert math.isclose(avatar.x, _TUNNEL_SUBSTEP_KILL_X)
    assert math.isclose(avatar.y, _TUNNEL_Y)
    assert _source_die_events(env)[0]["data"] == {"avatar": 1, "killer": 2, "old": False}


def test_vector_2p_large_single_step_tunnels_past_opponent_stored_body() -> None:
    state = _vector_state()
    state["pos"][0, 0] = [_TUNNEL_START_X, _TUNNEL_Y]
    state["prev_pos"][0, 0] = [_TUNNEL_START_X, _TUNNEL_Y]
    _seed_vector_body(state, owner=1, x=_TUNNEL_BODY_X, y=_TUNNEL_Y)

    counters = _step_vector(state, step_ms=_TUNNEL_SINGLE_STEP_MS)

    np.testing.assert_array_equal(state["alive"], np.asarray([[True, True]]))
    np.testing.assert_allclose(
        state["pos"][0, 0],
        np.asarray([_TUNNEL_SINGLE_STEP_X, _TUNNEL_Y]),
    )
    np.testing.assert_array_equal(state["death_player"], np.asarray([[-1, -1]], dtype=np.int16))
    assert counters["body_hits"] == 0


def test_vector_2p_source_sized_substeps_hit_opponent_stored_body() -> None:
    state = _vector_state()
    state["pos"][0, 0] = [_TUNNEL_START_X, _TUNNEL_Y]
    state["prev_pos"][0, 0] = [_TUNNEL_START_X, _TUNNEL_Y]
    _seed_vector_body(state, owner=1, x=_TUNNEL_BODY_X, y=_TUNNEL_Y)

    death_substep = None
    death_counters = None
    for substep in range(1, 1 + int(_TUNNEL_SINGLE_STEP_MS / _SOURCE_SUBSTEP_MS)):
        counters = _step_vector(state, step_ms=_SOURCE_SUBSTEP_MS)
        if not bool(state["alive"][0, 0]):
            death_substep = substep
            death_counters = counters
            break

    assert death_substep == _TUNNEL_KILL_SUBSTEP
    assert death_counters is not None
    np.testing.assert_array_equal(state["alive"], np.asarray([[False, True]]))
    np.testing.assert_allclose(
        state["pos"][0, 0],
        np.asarray([_TUNNEL_SUBSTEP_KILL_X, _TUNNEL_Y]),
    )
    np.testing.assert_array_equal(state["death_player"], np.asarray([[0, -1]], dtype=np.int16))
    np.testing.assert_array_equal(
        state["death_cause"],
        np.asarray(
            [[vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL, vector_runtime.DEATH_CAUSE_NONE]],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(state["death_hit_owner"], np.asarray([[1, -1]], dtype=np.int16))
    assert death_counters["body_hits"] == 1


def test_public_vector_2p_source_frame_decision_prevents_collision_tunneling() -> None:
    decision_source_frames = int(_TUNNEL_SINGLE_STEP_MS / _SOURCE_SUBSTEP_MS)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        decision_source_frames=decision_source_frames,
        body_capacity=8,
    )
    env.reset(
        seed=np.asarray([101], dtype=np.uint64),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    env.state["timer_active"][0] = False
    env.state["done"][0] = False
    env.state["terminated"][0] = False
    env.state["truncated"][0] = False
    env.state["reset_pending"][0] = False
    env.state["started"][0] = True
    env.state["in_round"][0] = True
    env.state["world_active"][0] = True
    env.state["alive"][0] = [True, True]
    env.state["present"][0] = [True, True]
    env.state["pos"][0] = [[_TUNNEL_START_X, _TUNNEL_Y], [60.0, 60.0]]
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["heading"][0] = [0.0, math.pi]
    env.state["angular_velocity_per_ms"][0] = [0.0, 0.0]
    env.state["printing"][0] = [False, False]
    env.state["print_manager_active"][0] = [False, False]
    env.state["body_active"][0] = False
    env.state["body_write_cursor"][0] = 0
    env.state["body_count"][0] = 0
    env.state["world_body_count"][0] = 0
    _seed_vector_body(env.state, owner=1, x=_TUNNEL_BODY_X, y=_TUNNEL_Y)

    batch = env.step(np.asarray([[1, 1]], dtype=np.int16))

    assert batch.info["source_frame_decision"] is True
    assert batch.info["decision_source_frames"] == decision_source_frames
    assert math.isclose(batch.info["decision_ms"], _TUNNEL_SINGLE_STEP_MS)
    np.testing.assert_array_equal(
        batch.info["source_physics_substeps_executed"],
        np.asarray([_TUNNEL_KILL_SUBSTEP], dtype=np.int32),
    )
    np.testing.assert_allclose(
        batch.info["source_physics_elapsed_ms"],
        np.asarray([_TUNNEL_SUBSTEP_KILL_X - _TUNNEL_START_X])
        / vector_runtime.SOURCE_AVATAR_SPEED
        * 1000.0,
    )
    np.testing.assert_array_equal(env.state["alive"], np.asarray([[False, True]]))
    np.testing.assert_allclose(
        env.state["pos"][0, 0],
        np.asarray([_TUNNEL_SUBSTEP_KILL_X, _TUNNEL_Y]),
    )
    np.testing.assert_array_equal(env.state["death_player"], np.asarray([[0, -1]], dtype=np.int16))
    np.testing.assert_array_equal(env.state["death_hit_owner"], np.asarray([[1, -1]], dtype=np.int16))
    assert batch.info["step_counters"]["body_hits"] == 1


def test_source_2p_own_stored_body_uses_trail_latency() -> None:
    safe_env = _quiet_source_env()
    safe_avatar = safe_env.avatar_by_id(1)
    _seed_source_body(safe_env, owner_id=1, x=20, y=20, num=0)
    safe_avatar.body_count = safe_avatar.trail_latency

    safe_env.step({}, elapsed_ms=0)

    assert safe_avatar.alive is True
    assert _source_die_events(safe_env) == []

    kill_env = _quiet_source_env()
    kill_avatar = kill_env.avatar_by_id(1)
    _seed_source_body(kill_env, owner_id=1, x=20, y=20, num=0)
    kill_avatar.body_count = kill_avatar.trail_latency + 1

    kill_env.step({}, elapsed_ms=0)

    assert kill_avatar.alive is False
    assert _source_die_events(kill_env)[0]["data"] == {
        "avatar": 1,
        "killer": 1,
        "old": False,
    }


def test_vector_2p_own_stored_body_uses_trail_latency_after_gap() -> None:
    safe_state = _vector_state()
    _seed_vector_body(safe_state, owner=0, x=20, y=20, num=0, break_before=True)
    safe_state["body_count"][0, 0] = 3

    safe_counters = _step_vector(safe_state)

    np.testing.assert_array_equal(safe_state["alive"], np.asarray([[True, True]]))
    np.testing.assert_array_equal(
        safe_state["death_player"], np.asarray([[-1, -1]], dtype=np.int16)
    )
    assert safe_counters["body_hits"] == 0

    kill_state = _vector_state()
    _seed_vector_body(kill_state, owner=0, x=20, y=20, num=0, break_before=True)
    kill_state["body_count"][0, 0] = 4

    kill_counters = _step_vector(kill_state)

    np.testing.assert_array_equal(kill_state["alive"], np.asarray([[False, True]]))
    np.testing.assert_array_equal(kill_state["death_player"], np.asarray([[0, -1]], dtype=np.int16))
    np.testing.assert_array_equal(
        kill_state["death_cause"],
        np.asarray(
            [[vector_runtime.DEATH_CAUSE_OWN_TRAIL, vector_runtime.DEATH_CAUSE_NONE]],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        kill_state["death_hit_owner"], np.asarray([[0, -1]], dtype=np.int16)
    )
    assert kill_counters["body_hits"] == 1


def test_source_2p_rendered_trail_line_crossing_without_stored_point_overlap_is_safe() -> None:
    env = _quiet_source_env()
    _seed_source_body(env, owner_id=2, x=10, y=20)
    _seed_source_body(env, owner_id=2, x=30, y=20)

    frame = env.step({}, elapsed_ms=0)

    assert env.avatar_by_id(1).alive is True
    assert frame["game"]["worldBodyCount"] == 2
    assert _source_die_events(env) == []


def test_vector_2p_rendered_trail_line_crossing_without_stored_point_overlap_is_safe() -> None:
    state = _vector_state()
    _seed_vector_body(state, owner=1, x=10, y=20)
    _seed_vector_body(state, owner=1, x=30, y=20)
    _seed_vector_visual_point(state, owner=1, x=10, y=20, break_before=True)
    _seed_vector_visual_point(state, owner=1, x=30, y=20, break_before=False)

    counters = _step_vector(state)

    np.testing.assert_array_equal(state["alive"], np.asarray([[True, True]]))
    np.testing.assert_array_equal(state["death_player"], np.asarray([[-1, -1]], dtype=np.int16))
    np.testing.assert_array_equal(state["world_body_count"], np.asarray([2], dtype=np.int32))
    assert counters["body_hits"] == 0


def test_source_2p_wall_death_takes_priority_over_stored_body_overlap() -> None:
    env = _quiet_source_env()
    env.set_avatar_state(1, x=0.3, y=20, angle=0)
    _seed_source_body(env, owner_id=2, x=0.3, y=20)
    env.events.clear()

    env.step({}, elapsed_ms=0)

    assert env.avatar_by_id(1).alive is False
    assert _source_die_events(env)[0]["data"] == {"avatar": 1, "killer": None, "old": None}


def test_vector_2p_wall_death_takes_priority_over_stored_body_overlap() -> None:
    state = _vector_state()
    state["pos"][0, 0] = [0.3, 20.0]
    state["prev_pos"][0, 0] = [0.3, 20.0]
    _seed_vector_body(state, owner=1, x=0.3, y=20)

    counters = _step_vector(state)

    np.testing.assert_array_equal(state["alive"], np.asarray([[False, True]]))
    np.testing.assert_array_equal(state["death_player"], np.asarray([[0, -1]], dtype=np.int16))
    np.testing.assert_array_equal(
        state["death_cause"],
        np.asarray(
            [[vector_runtime.DEATH_CAUSE_WALL, vector_runtime.DEATH_CAUSE_NONE]],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(state["death_hit_owner"], np.asarray([[-1, -1]], dtype=np.int16))
    assert counters["normal_wall_deaths"] == 1
    assert counters["body_hits"] == 0


def test_source_2p_borderless_wrap_skips_destination_body_until_next_frame() -> None:
    env = _quiet_source_env(borderless=True)
    env.set_avatar_state(1, x=87.8, y=44, angle=0)
    env.set_avatar_state(2, x=20, y=20, angle=math.pi)
    _seed_source_body(env, owner_id=2, x=0, y=44)
    env.events.clear()

    first = env.step({}, elapsed_ms=20)

    assert env.avatar_by_id(1).alive is True
    assert first["avatars"][0]["x"] == 0
    assert first["avatars"][0]["y"] == 44
    assert _source_die_events(env) == []

    second = env.step({}, elapsed_ms=0)

    assert env.avatar_by_id(1).alive is False
    assert second["game"]["worldBodyCount"] == 2
    assert _source_die_events(env)[0]["data"] == {"avatar": 1, "killer": 2, "old": False}


def test_vector_2p_borderless_wrap_skips_destination_body_until_next_frame() -> None:
    state = _vector_state(borderless=True)
    state["pos"][0, 0] = [87.8, 44.0]
    state["prev_pos"][0, 0] = [87.8, 44.0]
    state["pos"][0, 1] = [20.0, 20.0]
    state["prev_pos"][0, 1] = [20.0, 20.0]
    _seed_vector_body(state, owner=1, x=0, y=44)

    first_counters = _step_vector(state, step_ms=20)

    np.testing.assert_array_equal(state["alive"], np.asarray([[True, True]]))
    np.testing.assert_allclose(state["pos"][0, 0], np.asarray([0.0, 44.0]))
    np.testing.assert_array_equal(state["world_body_count"], np.asarray([1], dtype=np.int32))
    assert first_counters["borderless_wraps"] == 1
    assert first_counters["body_hits"] == 0

    second_counters = _step_vector(state, step_ms=0)

    np.testing.assert_array_equal(state["alive"], np.asarray([[False, True]]))
    np.testing.assert_array_equal(state["death_player"], np.asarray([[0, -1]], dtype=np.int16))
    np.testing.assert_array_equal(state["death_hit_owner"], np.asarray([[1, -1]], dtype=np.int16))
    np.testing.assert_array_equal(state["world_body_count"], np.asarray([2], dtype=np.int32))
    assert second_counters["body_hits"] == 1
