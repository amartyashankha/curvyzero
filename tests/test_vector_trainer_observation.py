import numpy as np
import pytest

from curvyzero.env import vector_autoreset
from curvyzero.env import vector_reset
from curvyzero.env import vector_runtime
from curvyzero.env import vector_trainer_observation
from curvyzero.env.trainer_contract import LIGHTZERO_FLAT_OBSERVATION_SHAPE


def _state(batch_size: int = 2, body_capacity: int = 6) -> dict[str, np.ndarray]:
    state = {
        "pos": np.zeros((batch_size, 2, 2), dtype=np.float64),
        "heading": np.zeros((batch_size, 2), dtype=np.float64),
        "alive": np.ones((batch_size, 2), dtype=bool),
        "tick": np.arange(batch_size, dtype=np.int32) + 5,
        "map_size": np.full(batch_size, 64.0, dtype=np.float64),
        "radius": np.ones((batch_size, 2), dtype=np.float64),
        "speed": np.full((batch_size, 2), 16.0, dtype=np.float64),
        "angular_velocity_per_ms": np.full((batch_size, 2), 2.8 / 1000.0, dtype=np.float64),
        "body_active": np.zeros((batch_size, body_capacity), dtype=bool),
        "body_pos": np.zeros((batch_size, body_capacity, 2), dtype=np.float64),
        "body_radius": np.ones((batch_size, body_capacity), dtype=np.float64),
        "body_owner": np.full((batch_size, body_capacity), -1, dtype=np.int16),
        "borderless": np.zeros(batch_size, dtype=bool),
        "done": np.zeros(batch_size, dtype=bool),
        "terminated": np.zeros(batch_size, dtype=bool),
        "truncated": np.zeros(batch_size, dtype=bool),
        "terminal_reason": np.full(
            batch_size,
            vector_reset.TERMINAL_REASON_NONE,
            dtype=np.int16,
        ),
        "winner": np.full(batch_size, -1, dtype=np.int16),
        "draw": np.zeros(batch_size, dtype=bool),
        "episode_id": np.asarray([10, 20], dtype=np.int64)[:batch_size],
        "episode_step": np.asarray([3, 7], dtype=np.int32)[:batch_size],
    }
    state["pos"][0] = [[10.0, 10.0], [20.0, 10.0]]
    state["heading"][0] = [0.0, np.pi]
    state["body_active"][0, :4] = True
    state["body_pos"][0, :4] = [
        [10.0, 10.0],
        [14.0, 10.0],
        [10.0, 5.0],
        [20.0, 10.0],
    ]
    state["body_owner"][0, :4] = [0, 0, 1, 1]

    if batch_size > 1:
        state["pos"][1] = [[12.0, 12.0], [40.0, 12.0]]
        state["heading"][1] = [0.0, np.pi]
        state["body_active"][1, :2] = True
        state["body_pos"][1, :2] = [[12.0, 12.0], [40.0, 12.0]]
        state["body_owner"][1, :2] = [0, 1]
    return state


def test_observe_vector_1v1_egocentric_rays_v0_returns_pinned_float32_106():
    batch = vector_trainer_observation.observe_vector_1v1_egocentric_rays_v0(
        _state(),
        0,
        decision_ms=300.0,
        max_ticks=100,
    )

    assert batch.player_ids == ("player_0", "player_1")
    assert batch.rays.shape == (2, 24, 4)
    assert batch.scalars.shape == (2, 10)
    assert batch.observation.shape == (2,) + LIGHTZERO_FLAT_OBSERVATION_SHAPE
    assert batch.observation.dtype == np.float32
    assert batch.lightzero_action_mask.dtype == np.int8
    np.testing.assert_array_equal(
        batch.action_mask,
        np.asarray([[True, True, True], [True, True, True]], dtype=bool),
    )
    assert float(batch.observation.min()) >= -1.0
    assert float(batch.rays.min()) >= 0.0
    assert float(batch.rays.max()) <= 1.0
    assert batch.done is False
    assert batch.final_reward_map is None


def test_observe_vector_1v1_egocentric_rays_batch_arrays_match_scalar_rows():
    state = _state(batch_size=2)
    state["borderless"][1] = True
    state["alive"][1] = [False, True]
    state["done"][1] = True
    state["terminated"][1] = True
    state["terminal_reason"][1] = vector_runtime.TERMINAL_REASON_SURVIVOR_WIN
    state["winner"][1] = 1

    observation, action_mask, lightzero_action_mask, to_play = (
        vector_trainer_observation.observe_vector_1v1_egocentric_rays_batch_arrays_v0(
            state,
            decision_ms=300.0,
            max_ticks=100,
        )
    )

    assert observation.shape == (2, 2, 106)
    assert observation.dtype == np.float32
    assert action_mask.shape == (2, 2, 3)
    assert action_mask.dtype == np.bool_
    assert lightzero_action_mask.dtype == np.int8
    assert to_play.dtype == np.int64
    for row in range(2):
        scalar = vector_trainer_observation.observe_vector_1v1_egocentric_rays_v0(
            state,
            row,
            decision_ms=300.0,
            max_ticks=100,
        )
        np.testing.assert_allclose(observation[row], scalar.observation)
        np.testing.assert_array_equal(action_mask[row], scalar.action_mask)
        np.testing.assert_array_equal(lightzero_action_mask[row], scalar.lightzero_action_mask)
        np.testing.assert_array_equal(to_play[row], scalar.to_play)


def test_observe_vector_1v1_egocentric_rays_v0_separates_body_channels():
    batch = vector_trainer_observation.observe_vector_1v1_egocentric_rays_v0(
        _state(),
        0,
        decision_ms=300.0,
        max_ticks=100,
    )

    player_0_rays = batch.rays[0]
    forward_ray = 0
    upward_ray = 6
    assert player_0_rays[forward_ray, 1] < 1.0
    assert player_0_rays[forward_ray, 3] < 1.0
    assert player_0_rays[upward_ray, 2] < 1.0
    assert player_0_rays[forward_ray, 2] == 1.0


def test_observe_vector_1v1_egocentric_rays_v0_survivor_reward_from_winner():
    state = _state()
    state["alive"][1] = [False, True]
    state["done"][1] = True
    state["terminated"][1] = True
    state["terminal_reason"][1] = vector_runtime.TERMINAL_REASON_SURVIVOR_WIN
    state["winner"][1] = 1

    batch = vector_trainer_observation.observe_vector_1v1_egocentric_rays_v0(
        state,
        1,
        decision_ms=300.0,
        max_ticks=100,
    )

    np.testing.assert_array_equal(batch.rewards, np.asarray([-1.0, 1.0], dtype=np.float32))
    assert batch.final_reward_map == {"player_0": -1.0, "player_1": 1.0}
    np.testing.assert_array_equal(batch.action_mask, np.zeros((2, 3), dtype=bool))
    assert batch.done is True
    assert batch.terminated is True
    assert batch.truncated is False


def test_observe_vector_1v1_egocentric_rays_v0_survivor_reward_preserves_truncation():
    state = _state()
    state["alive"][1] = [False, True]
    state["done"][1] = True
    state["terminated"][1] = True
    state["truncated"][1] = True
    state["terminal_reason"][1] = vector_runtime.TERMINAL_REASON_SURVIVOR_WIN
    state["winner"][1] = 1

    batch = vector_trainer_observation.observe_vector_1v1_egocentric_rays_v0(
        state,
        1,
        decision_ms=300.0,
        max_ticks=100,
    )

    np.testing.assert_array_equal(batch.rewards, np.asarray([-1.0, 1.0], dtype=np.float32))
    assert batch.reward_info["player_1"]["terminal_reason"] == "survivor_win"
    assert batch.reward_info["player_1"]["timeout"] is True
    assert batch.reward_info["player_1"]["truncation_reason"] == "max_ticks"
    assert batch.reward_info["player_1"]["terminated"] is True
    assert batch.reward_info["player_1"]["truncated"] is True
    assert batch.done is True
    assert batch.terminated is True
    assert batch.truncated is True


@pytest.mark.parametrize(
    (
        "terminal_reason",
        "expected_terminal_reason",
        "expected_timeout",
        "expected_truncation_reason",
    ),
    [
        (
            vector_reset.TERMINAL_REASON_NONE,
            "timeout",
            True,
            "max_ticks",
        ),
        (
            vector_reset.TERMINAL_REASON_TIMEOUT_TRUNCATED,
            "timeout",
            True,
            "max_ticks",
        ),
        (
            vector_reset.TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED,
            "event_overflow_truncated",
            False,
            "event_overflow",
        ),
        (
            vector_reset.TERMINAL_REASON_BODY_OVERFLOW_TRUNCATED,
            "body_overflow_truncated",
            False,
            "body_overflow",
        ),
    ],
)
def test_observe_vector_1v1_egocentric_rays_v0_labels_pure_truncation_reason(
    terminal_reason,
    expected_terminal_reason,
    expected_timeout,
    expected_truncation_reason,
):
    state = _state(batch_size=1)
    state["done"][0] = True
    state["terminated"][0] = False
    state["truncated"][0] = True
    state["terminal_reason"][0] = terminal_reason

    batch = vector_trainer_observation.observe_vector_1v1_egocentric_rays_v0(
        state,
        0,
        decision_ms=300.0,
        max_ticks=100,
    )

    np.testing.assert_array_equal(batch.rewards, np.zeros(2, dtype=np.float32))
    assert batch.final_reward_map == {"player_0": 0.0, "player_1": 0.0}
    for reward_info in batch.reward_info.values():
        assert reward_info["terminal_reason"] == expected_terminal_reason
        assert reward_info["timeout"] is expected_timeout
        assert reward_info["truncation_reason"] == expected_truncation_reason
        assert reward_info["terminated"] is False
        assert reward_info["truncated"] is True
    assert batch.done is True
    assert batch.terminated is False
    assert batch.truncated is True


def test_observe_vector_1v1_egocentric_rays_v0_draw_reward_from_terminal_reason():
    state = _state()
    state["alive"][0] = [False, False]
    state["done"][0] = True
    state["terminated"][0] = True
    state["terminal_reason"][0] = vector_runtime.TERMINAL_REASON_ALL_DEAD_DRAW
    state["winner"][0] = -1
    state["draw"][0] = True

    batch = vector_trainer_observation.observe_vector_1v1_egocentric_rays_v0(
        state,
        0,
        decision_ms=300.0,
        max_ticks=100,
    )

    np.testing.assert_array_equal(batch.rewards, np.asarray([0.0, 0.0], dtype=np.float32))
    assert batch.final_reward_map == {"player_0": 0.0, "player_1": 0.0}
    assert batch.reward_info["player_0"]["terminal_reason"] == "all_dead_draw"
    assert batch.reward_info["player_0"]["draw"] is True


def test_observe_vector_1v1_egocentric_rays_v0_own_trail_respects_body_latency():
    state = _state(batch_size=1, body_capacity=4)
    state["body_active"][0] = [True, True, True, False]
    state["body_pos"][0, :3] = [
        [10.0, 10.0],
        [14.0, 10.0],
        [18.0, 10.0],
    ]
    state["body_owner"][0] = [0, 0, 0, -1]
    state["body_num"] = np.asarray([[10, 9, 6, -1]], dtype=np.int32)
    state["live_body_num"] = np.asarray([[10, 0]], dtype=np.int32)

    batch = vector_trainer_observation.observe_vector_1v1_egocentric_rays_v0(
        state,
        0,
        decision_ms=300.0,
        max_ticks=100,
    )

    arena_diagonal = np.hypot(64.0, 64.0)
    fresh_body_hit = 3.0 / arena_diagonal
    older_body_hit = 7.0 / arena_diagonal
    assert batch.rays[0, 0, 1] > fresh_body_hit
    np.testing.assert_allclose(batch.rays[0, 0, 1], older_body_hit, rtol=1e-6)


def test_observe_vector_1v1_egocentric_rays_batch_arrays_preserve_trail_latency():
    state = _state(batch_size=1, body_capacity=4)
    state["body_active"][0] = [True, True, True, False]
    state["body_pos"][0, :3] = [
        [10.0, 10.0],
        [14.0, 10.0],
        [18.0, 10.0],
    ]
    state["body_owner"][0] = [0, 0, 0, -1]
    state["body_num"] = np.asarray([[10, 9, 6, -1]], dtype=np.int32)
    state["live_body_num"] = np.asarray([[10, 0]], dtype=np.int32)

    observation, _action_mask, _lightzero_action_mask, _to_play = (
        vector_trainer_observation.observe_vector_1v1_egocentric_rays_batch_arrays_v0(
            state,
            decision_ms=300.0,
            max_ticks=100,
        )
    )
    scalar = vector_trainer_observation.observe_vector_1v1_egocentric_rays_v0(
        state,
        0,
        decision_ms=300.0,
        max_ticks=100,
    )

    np.testing.assert_allclose(observation[0], scalar.observation)


def test_observe_vector_1v1_batch_arrays_trim_raycast_to_body_write_cursor(monkeypatch):
    state = _state(batch_size=2, body_capacity=32)
    state["body_active"][:] = True
    state["body_owner"][:] = 0
    state["body_write_cursor"] = np.asarray([2, 5], dtype=np.int32)
    captured_circle_counts = []

    def fake_nearest_circle_hits_batch(
        origins,
        directions,
        centers,
        radii,
        mask,
    ):
        del centers, radii
        captured_circle_counts.append(int(mask.shape[2]))
        return np.full(directions.shape[:3], np.inf, dtype=np.float64)

    monkeypatch.setattr(
        vector_trainer_observation,
        "_nearest_circle_hits_batch",
        fake_nearest_circle_hits_batch,
    )

    vector_trainer_observation.observe_vector_1v1_egocentric_rays_batch_arrays_v0(
        state,
        decision_ms=300.0,
        max_ticks=100,
    )

    assert captured_circle_counts == [5, 5, 1]


def test_observe_vector_1v1_egocentric_rays_v0_ignores_slots_after_body_write_cursor():
    state = _state(batch_size=1, body_capacity=6)
    state["body_active"][0, :6] = True
    state["body_pos"][0, :6] = [
        [10.0, 10.0],
        [14.0, 10.0],
        [10.0, 5.0],
        [20.0, 10.0],
        [11.0, 10.0],
        [12.0, 10.0],
    ]
    state["body_owner"][0, :6] = [0, 0, 1, 1, 1, 1]
    state["body_write_cursor"] = np.asarray([4], dtype=np.int32)
    state["body_num"] = np.asarray([[0, 1, 0, 1, 2, 3]], dtype=np.int32)
    state["live_body_num"] = np.asarray([[5, 5]], dtype=np.int32)

    with_tail_garbage = vector_trainer_observation.observe_vector_1v1_egocentric_rays_v0(
        state,
        0,
        decision_ms=300.0,
        max_ticks=100,
    )
    compact = _state(batch_size=1, body_capacity=4)
    for key in (
        "pos",
        "heading",
        "alive",
        "tick",
        "map_size",
        "radius",
        "speed",
        "angular_velocity_per_ms",
        "borderless",
        "done",
        "terminated",
        "truncated",
        "terminal_reason",
        "winner",
        "draw",
    ):
        compact[key][...] = state[key][...]
    compact["body_active"][0, :4] = state["body_active"][0, :4]
    compact["body_pos"][0, :4] = state["body_pos"][0, :4]
    compact["body_radius"][0, :4] = state["body_radius"][0, :4]
    compact["body_owner"][0, :4] = state["body_owner"][0, :4]
    compact["body_write_cursor"] = np.asarray([4], dtype=np.int32)
    compact["body_num"] = state["body_num"][:, :4].copy()
    compact["live_body_num"] = state["live_body_num"].copy()

    without_tail_garbage = vector_trainer_observation.observe_vector_1v1_egocentric_rays_v0(
        compact,
        0,
        decision_ms=300.0,
        max_ticks=100,
    )

    np.testing.assert_allclose(with_tail_garbage.observation, without_tail_garbage.observation)


def test_vectorized_wall_and_normalized_hits_match_scalar_helpers():
    origin = np.asarray([10.0, 12.0], dtype=np.float64)
    directions = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [np.sqrt(0.5), np.sqrt(0.5)],
            [-np.sqrt(0.5), np.sqrt(0.5)],
        ],
        dtype=np.float64,
    )
    map_size = 64.0
    arena_diagonal = np.hypot(map_size, map_size)

    vectorized_distances = vector_trainer_observation._wall_hit_distances(
        origin,
        directions,
        map_size=map_size,
    )
    scalar_distances = np.asarray(
        [
            vector_trainer_observation._wall_hit_distance(
                origin,
                direction,
                map_size=map_size,
            )
            for direction in directions
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(vectorized_distances, scalar_distances)
    np.testing.assert_allclose(
        vector_trainer_observation._normalized_hit_distances(
            vectorized_distances,
            arena_diagonal,
        ),
        vector_trainer_observation._normalized_hit_distances_slow(
            scalar_distances,
            arena_diagonal,
        ),
    )


def test_vectorized_wall_hits_match_scalar_helpers_for_ray_table():
    map_size = 64.0
    origins = np.asarray(
        [
            [1.0, 1.0],
            [10.0, 12.0],
            [31.5, 31.5],
            [63.0, 4.0],
            [-1.0, 8.0],
            [8.0, 64.0],
        ],
        dtype=np.float64,
    )
    headings = np.asarray([0.0, 0.23, np.pi / 2.0, np.pi, -1.7], dtype=np.float64)

    for origin in origins:
        for heading in headings:
            directions = vector_trainer_observation._ray_directions(float(heading))
            vectorized_distances = vector_trainer_observation._wall_hit_distances(
                origin,
                directions,
                map_size=map_size,
            )
            scalar_distances = np.asarray(
                [
                    vector_trainer_observation._wall_hit_distance(
                        origin,
                        direction,
                        map_size=map_size,
                    )
                    for direction in directions
                ],
                dtype=np.float64,
            )

            np.testing.assert_allclose(vectorized_distances, scalar_distances)


def test_observe_vector_1v1_egocentric_rays_v0_rejects_duplicate_player_ids():
    with pytest.raises(
        vector_trainer_observation.VectorTrainerObservationError,
        match="unique",
    ):
        vector_trainer_observation.observe_vector_1v1_egocentric_rays_v0(
            _state(),
            0,
            player_ids=("duplicate", "duplicate"),
            decision_ms=300.0,
            max_ticks=100,
        )


def test_build_final_trainer_transition_rows_feeds_autoreset_plan():
    state = _state()
    state["alive"][1] = [True, False]
    state["done"][1] = True
    state["terminated"][1] = True
    state["terminal_reason"][1] = vector_runtime.TERMINAL_REASON_SURVIVOR_WIN
    state["winner"][1] = 0

    transition = (
        vector_trainer_observation.build_final_trainer_transition_1v1_no_bonus_rows(
            state,
            np.asarray([False, True], dtype=bool),
            decision_ms=300.0,
            max_ticks=100,
        )
    )

    assert transition["schema"] == (
        vector_trainer_observation.VECTOR_TRAINER_TRANSITION_SCHEMA_ID
    )
    np.testing.assert_array_equal(transition["rows"], np.asarray([1], dtype=np.int32))
    assert transition["final_observation"].shape == (2, 2, 106)
    assert transition["final_reward_map"].shape == (2, 2)
    np.testing.assert_array_equal(
        transition["final_reward_map"][1],
        np.asarray([1.0, -1.0], dtype=np.float32),
    )

    plan = vector_autoreset.plan_autoreset_rows(
        state,
        final_observation=transition["final_observation"],
        final_reward_map=transition["final_reward_map"],
        reset_seed=np.asarray([111, 222], dtype=np.uint64),
        reset_source=np.asarray(
            [vector_reset.RESET_SOURCE_MANUAL, vector_reset.RESET_SOURCE_AUTORESET],
            dtype=np.int16,
        ),
    )

    assert plan["autoreset_count"] == 1
    np.testing.assert_array_equal(plan["row_ids"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(
        plan["final_transition_snapshot"]["final_observation"],
        transition["final_observation"][1:],
    )
    np.testing.assert_array_equal(
        plan["final_transition_snapshot"]["final_reward_map"],
        transition["final_reward_map"][1:],
    )
