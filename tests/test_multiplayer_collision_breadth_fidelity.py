import sys
from pathlib import Path

import numpy as np
import pytest

from curvyzero.env import vector_reset
from curvyzero.env import vector_runtime
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SourceStateMultiplayerTrainerReplayRecorder,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    MultiplayerTrainerStepV0,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    SourceStateMultiplayerTrainerSurface,
)


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import compare_vector_arrays_to_fidelity as vector_compare  # noqa: E402
import seed_vector_state_from_fixtures as seed_bridge  # noqa: E402


def test_head_head_reverse_order_fixture_reaches_public_surface_and_replay():
    public_env, public_steps = _run_public_fixture(
        "source_collision_head_head_reverse_order_single_death_step.json",
        body_capacity=8,
    )
    public = public_steps[0]

    _assert_death_metadata(
        public.info,
        alive=[False, True],
        death_count=1,
        death_player=[0, -1],
        death_cause=[
            vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
            vector_runtime.DEATH_CAUSE_NONE,
        ],
        death_hit_owner=[1, -1],
        score=[0, 1],
    )
    np.testing.assert_array_equal(public.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(public.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        public.reward,
        np.asarray([[-1.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        public.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    assert public.info["winner_ids"] == [[1]]
    assert public.info["loser_ids"] == [[0]]
    assert public.info["step_counters"]["normal_points_inserted"] == 2
    assert public.info["step_counters"]["body_hits"] == 1
    assert public.info["step_counters"]["death_points_inserted"] == 1
    assert public.info["step_counters"]["terminal_score_rows"] == 1
    np.testing.assert_array_equal(public_env.state["world_body_count"], np.asarray([3]))
    assert _debug_die_events(public_env) == [(0, 1, 0)]

    surface_steps = _run_surface_fixture(
        "source_collision_head_head_reverse_order_single_death_step.json",
        body_capacity=8,
    )
    surface = surface_steps[0]

    _assert_death_metadata(
        surface.info,
        alive=[False, True],
        death_count=1,
        death_player=[0, -1],
        death_cause=[
            vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
            vector_runtime.DEATH_CAUSE_NONE,
        ],
        death_hit_owner=[1, -1],
        score=[0, 1],
    )
    np.testing.assert_array_equal(surface.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(surface.final_observation_row_mask, np.asarray([True]))
    np.testing.assert_array_equal(surface.reward, np.asarray([[0.0, 1.0]], dtype=np.float32))
    np.testing.assert_array_equal(surface.final_reward_map, surface.reward)
    assert int(np.count_nonzero(surface.final_observation[0])) > 0

    chunk = _record_chunk(
        surface_steps,
        source_ref="source_collision_head_head_reverse_order_single_death_step",
    )
    assert chunk.metadata["closed_by_terminal"] is True
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"][:, 0],
        np.asarray([True], dtype=bool),
    )
    record = chunk.records[0]
    assert record["terminal_or_final"] is True
    assert record["final_observation_rows"] == [0]
    assert record["death_player"] == [[0, -1]]
    assert record["death_hit_owner"] == [[1, -1]]
    assert record["death_cause_name"] == [["opponent_trail", "none"]]
    assert record["winner_ids"] == [[1]]
    assert record["loser_ids"] == [[0]]


@pytest.mark.parametrize(
    (
        "scenario_name",
        "expected_alive",
        "expected_death_count",
        "expected_death_player",
        "expected_death_cause",
        "expected_death_hit_owner",
        "expected_world_body_count",
        "expected_public_body_hits",
        "expected_surface_reward",
    ),
    [
        (
            "source_body_own_delta3_safe_step.json",
            [True, True, True],
            0,
            [-1, -1, -1],
            [
                vector_runtime.DEATH_CAUSE_NONE,
                vector_runtime.DEATH_CAUSE_NONE,
                vector_runtime.DEATH_CAUSE_NONE,
            ],
            [-1, -1, -1],
            1,
            0,
            [1.0, 1.0, 1.0],
        ),
        (
            "source_body_own_delta4_kills_step.json",
            [False, True, True],
            1,
            [0, -1, -1],
            [
                vector_runtime.DEATH_CAUSE_OWN_TRAIL,
                vector_runtime.DEATH_CAUSE_NONE,
                vector_runtime.DEATH_CAUSE_NONE,
            ],
            [0, -1, -1],
            2,
            1,
            [0.0, 1.0, 1.0],
        ),
    ],
    ids=["delta3-safe", "delta4-kills"],
)
def test_own_trail_latency_fixtures_reach_public_surface_and_replay(
    scenario_name: str,
    expected_alive: list[bool],
    expected_death_count: int,
    expected_death_player: list[int],
    expected_death_cause: list[int],
    expected_death_hit_owner: list[int],
    expected_world_body_count: int,
    expected_public_body_hits: int,
    expected_surface_reward: list[float],
):
    public_env, public_steps = _run_public_fixture(scenario_name, body_capacity=8)
    public = public_steps[0]

    _assert_death_metadata(
        public.info,
        alive=expected_alive,
        death_count=expected_death_count,
        death_player=expected_death_player,
        death_cause=expected_death_cause,
        death_hit_owner=expected_death_hit_owner,
    )
    np.testing.assert_array_equal(public.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(public.reward, np.zeros((1, 3), dtype=np.float32))
    np.testing.assert_array_equal(
        public_env.state["world_body_count"],
        np.asarray([expected_world_body_count], dtype=np.int32),
    )
    assert public.info["step_counters"]["body_hits"] == expected_public_body_hits
    assert public.info["step_counters"]["death_points_inserted"] == expected_public_body_hits

    surface_steps = _run_surface_fixture(scenario_name, body_capacity=8)
    surface = surface_steps[0]

    _assert_death_metadata(
        surface.info,
        alive=expected_alive,
        death_count=expected_death_count,
        death_player=expected_death_player,
        death_cause=expected_death_cause,
        death_hit_owner=expected_death_hit_owner,
    )
    np.testing.assert_array_equal(surface.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(surface.final_observation_row_mask, np.asarray([False]))
    np.testing.assert_array_equal(
        surface.reward,
        np.asarray([expected_surface_reward], dtype=np.float32),
    )

    chunk = _record_chunk(surface_steps, source_ref=scenario_name.removesuffix(".json"))
    assert chunk.metadata["closed_by_terminal"] is False
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"][:, 0],
        np.asarray([False], dtype=bool),
    )
    record = chunk.records[0]
    assert record["terminal_or_final"] is False
    assert record["final_observation_rows"] == []
    assert record["death_player"] == [expected_death_player]
    assert record["death_hit_owner"] == [expected_death_hit_owner]
    assert record["death_cause"] == [expected_death_cause]
    assert record["alive"] == [expected_alive]


@pytest.mark.parametrize(
    (
        "scenario_name",
        "expected_alive",
        "expected_death_count",
        "expected_death_player",
        "expected_death_cause",
        "expected_death_hit_owner",
        "expected_printing",
        "expected_visible_trail_count",
        "expected_has_visible_trail_last",
        "expected_body_owner",
        "expected_body_insert_kind",
        "expected_counters",
        "expected_die_events",
        "expected_surface_reward",
    ),
    [
        (
            "source_body_opponent_tangent_safe_step.json",
            [True, True, True],
            0,
            [-1, -1, -1],
            [
                vector_runtime.DEATH_CAUSE_NONE,
                vector_runtime.DEATH_CAUSE_NONE,
                vector_runtime.DEATH_CAUSE_NONE,
            ],
            [-1, -1, -1],
            [False, False, False],
            [0, 0, 0],
            [False, False, False],
            [1],
            None,
            {
                "body_hits": 0,
                "normal_points_inserted": 0,
                "death_points_inserted": 0,
            },
            [],
            [1.0, 1.0, 1.0],
        ),
        (
            "source_body_opponent_overlap_kills_step.json",
            [False, True, True],
            1,
            [0, -1, -1],
            [
                vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
                vector_runtime.DEATH_CAUSE_NONE,
                vector_runtime.DEATH_CAUSE_NONE,
            ],
            [1, -1, -1],
            [False, False, False],
            [1, 0, 0],
            [True, False, False],
            [1, 0],
            None,
            {
                "body_hits": 1,
                "normal_points_inserted": 0,
                "death_points_inserted": 1,
            },
            [(0, 1, 0)],
            [0.0, 1.0, 1.0],
        ),
        (
            "source_body_same_frame_point_kills_step.json",
            [False, True, True],
            1,
            [0, -1, -1],
            [
                vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
                vector_runtime.DEATH_CAUSE_NONE,
                vector_runtime.DEATH_CAUSE_NONE,
            ],
            [1, -1, -1],
            [False, True, False],
            [1, 1, 0],
            [True, True, False],
            [1, 0],
            [
                vector_runtime.BODY_KIND_NORMAL,
                vector_runtime.BODY_KIND_DEATH,
            ],
            {
                "body_hits": 1,
                "normal_points_inserted": 1,
                "death_points_inserted": 1,
            },
            [(0, 1, 0)],
            [0.0, 1.0, 1.0],
        ),
        (
            "source_body_same_frame_point_control_safe_step.json",
            [True, True, True],
            0,
            [-1, -1, -1],
            [
                vector_runtime.DEATH_CAUSE_NONE,
                vector_runtime.DEATH_CAUSE_NONE,
                vector_runtime.DEATH_CAUSE_NONE,
            ],
            [-1, -1, -1],
            [False, False, False],
            [0, 0, 0],
            [False, False, False],
            [],
            [],
            {
                "body_hits": 0,
                "normal_points_inserted": 0,
                "death_points_inserted": 0,
            },
            [],
            [1.0, 1.0, 1.0],
        ),
    ],
    ids=[
        "opponent-tangent-safe",
        "opponent-overlap-kills",
        "same-frame-point-kills",
        "same-frame-control-safe",
    ],
)
def test_body_collision_canary_fixtures_reach_public_surface_and_replay(
    scenario_name: str,
    expected_alive: list[bool],
    expected_death_count: int,
    expected_death_player: list[int],
    expected_death_cause: list[int],
    expected_death_hit_owner: list[int],
    expected_printing: list[bool],
    expected_visible_trail_count: list[int],
    expected_has_visible_trail_last: list[bool],
    expected_body_owner: list[int],
    expected_body_insert_kind: list[int] | None,
    expected_counters: dict[str, int],
    expected_die_events: list[tuple[int, int, int]],
    expected_surface_reward: list[float],
) -> None:
    public_env, public_steps = _run_public_fixture(scenario_name, body_capacity=8)
    public = public_steps[0]

    _assert_death_metadata(
        public.info,
        alive=expected_alive,
        death_count=expected_death_count,
        death_player=expected_death_player,
        death_cause=expected_death_cause,
        death_hit_owner=expected_death_hit_owner,
        score=[0, 0, 0],
    )
    np.testing.assert_array_equal(public.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(public.reward, np.zeros((1, 3), dtype=np.float32))
    np.testing.assert_array_equal(
        public.info["final_observation_row_mask"],
        np.asarray([False], dtype=bool),
    )
    _assert_trail_gap_state(
        public_env,
        expected_printing=expected_printing,
        expected_visible_trail_count=expected_visible_trail_count,
        expected_has_visible_trail_last=expected_has_visible_trail_last,
        expected_body_owner=expected_body_owner,
        expected_body_insert_kind=expected_body_insert_kind,
    )
    _assert_selected_step_counters(public.info, expected_counters)
    assert _debug_die_events(public_env) == expected_die_events

    surface_env, surface_steps = _run_surface_fixture_with_env(
        scenario_name,
        body_capacity=8,
    )
    surface = surface_steps[0]

    _assert_death_metadata(
        surface.info,
        alive=expected_alive,
        death_count=expected_death_count,
        death_player=expected_death_player,
        death_cause=expected_death_cause,
        death_hit_owner=expected_death_hit_owner,
        score=[0, 0, 0],
    )
    np.testing.assert_array_equal(surface.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        surface.final_observation_row_mask,
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        surface.reward,
        np.asarray([expected_surface_reward], dtype=np.float32),
    )
    np.testing.assert_array_equal(surface.final_reward_map, np.zeros((1, 3)))
    np.testing.assert_array_equal(
        surface.live_mask,
        np.asarray([expected_alive], dtype=bool),
    )
    np.testing.assert_array_equal(
        surface.policy_env_row,
        np.zeros(sum(expected_alive), dtype=np.int32),
    )
    expected_policy_player = np.flatnonzero(np.asarray(expected_alive, dtype=bool))
    np.testing.assert_array_equal(
        surface.policy_player,
        expected_policy_player.astype(np.int16),
    )
    _assert_trail_gap_state(
        surface_env,
        expected_printing=expected_printing,
        expected_visible_trail_count=expected_visible_trail_count,
        expected_has_visible_trail_last=expected_has_visible_trail_last,
        expected_body_owner=expected_body_owner,
        expected_body_insert_kind=expected_body_insert_kind,
    )
    assert _debug_die_events(surface_env) == expected_die_events

    chunk = _record_chunk(surface_steps, source_ref=scenario_name.removesuffix(".json"))
    assert chunk.metadata["closed_by_terminal"] is False
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"][:, 0],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        chunk.arrays["reward"][0, 0],
        np.asarray(expected_surface_reward, dtype=np.float32),
    )
    assert int(np.count_nonzero(chunk.arrays["final_observation"])) == 0
    record = chunk.records[0]
    assert record["terminal_or_final"] is False
    assert record["final_observation_rows"] == []
    assert record["alive"] == [expected_alive]
    assert record["death_player"] == [expected_death_player]
    assert record["death_hit_owner"] == [expected_death_hit_owner]
    assert record["death_cause"] == [expected_death_cause]
    assert record["death_cause_name"] == [
        [vector_runtime.death_cause_name(cause) for cause in expected_death_cause]
    ]
    assert record["policy_player"] == [int(player) for player in expected_policy_player]
    for key, value in expected_counters.items():
        assert record["step_counters"][key] == value


def test_old_opponent_body_collision_needs_public_surface_and_replay_old_metadata():
    public_env, public_steps = _run_public_fixture(
        "source_body_old_opponent_overlap_kills_step.json",
        body_capacity=8,
    )
    public = public_steps[0]

    _assert_death_metadata(
        public.info,
        alive=[False, True, True],
        death_count=1,
        death_player=[0, -1, -1],
        death_cause=[
            vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
            vector_runtime.DEATH_CAUSE_NONE,
            vector_runtime.DEATH_CAUSE_NONE,
        ],
        death_hit_owner=[1, -1, -1],
        score=[0, 0, 0],
    )
    assert _debug_die_events(public_env) == [(0, 1, 1)]
    np.testing.assert_array_equal(
        public.info["death_hit_old"],
        np.asarray([[1, -1, -1]], dtype=np.int8),
    )

    surface_steps = _run_surface_fixture(
        "source_body_old_opponent_overlap_kills_step.json",
        body_capacity=8,
    )
    surface = surface_steps[0]
    np.testing.assert_array_equal(
        surface.info["death_hit_old"],
        np.asarray([[1, -1, -1]], dtype=np.int8),
    )

    chunk = _record_chunk(
        surface_steps,
        source_ref="source_body_old_opponent_overlap_kills_step",
    )
    assert chunk.records[0]["death_hit_old"] == [[1, -1, -1]]


@pytest.mark.parametrize(
    (
        "scenario_name",
        "expected_alive",
        "expected_death_count",
        "expected_death_player",
        "expected_death_cause",
        "expected_death_hit_owner",
        "expected_printing",
        "expected_visible_trail_count",
        "expected_has_visible_trail_last",
        "expected_body_owner",
        "expected_body_insert_kind",
        "expected_counters",
        "expected_surface_reward",
    ),
    [
        (
            "source_trail_gap_hole_space_safe_step.json",
            [True, True, True],
            0,
            [-1, -1, -1],
            [
                vector_runtime.DEATH_CAUSE_NONE,
                vector_runtime.DEATH_CAUSE_NONE,
                vector_runtime.DEATH_CAUSE_NONE,
            ],
            [-1, -1, -1],
            [False, False, False],
            [0, 0, 0],
            [False, False, False],
            [1],
            None,
            {
                "body_hits": 0,
                "death_points_inserted": 0,
                "print_manager_toggle_updates": 0,
                "print_manager_visual_clears": 0,
            },
            [1.0, 1.0, 1.0],
        ),
        (
            "source_trail_gap_stored_body_still_kills_step.json",
            [False, True, True],
            1,
            [0, -1, -1],
            [
                vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
                vector_runtime.DEATH_CAUSE_NONE,
                vector_runtime.DEATH_CAUSE_NONE,
            ],
            [1, -1, -1],
            [False, False, False],
            [1, 0, 0],
            [True, False, False],
            [1, 0],
            None,
            {
                "body_hits": 1,
                "death_points_inserted": 1,
                "print_manager_toggle_updates": 0,
                "print_manager_visual_clears": 0,
            },
            [0.0, 1.0, 1.0],
        ),
        (
            "source_trail_gap_print_to_hole_boundary_kills_step.json",
            [False, True, True],
            1,
            [0, -1, -1],
            [
                vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
                vector_runtime.DEATH_CAUSE_NONE,
                vector_runtime.DEATH_CAUSE_NONE,
            ],
            [1, -1, -1],
            [False, False, False],
            [1, 0, 0],
            [True, False, False],
            [1, 0],
            [
                vector_runtime.BODY_KIND_IMPORTANT,
                vector_runtime.BODY_KIND_DEATH,
            ],
            {
                "body_hits": 1,
                "death_points_inserted": 1,
                "print_manager_toggle_updates": 1,
                "print_manager_visual_clears": 1,
            },
            [0.0, 1.0, 1.0],
        ),
        (
            "source_trail_gap_hole_to_print_boundary_kills_step.json",
            [False, True, True],
            1,
            [0, -1, -1],
            [
                vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
                vector_runtime.DEATH_CAUSE_NONE,
                vector_runtime.DEATH_CAUSE_NONE,
            ],
            [1, -1, -1],
            [False, True, False],
            [1, 1, 0],
            [True, True, False],
            [1, 0],
            [
                vector_runtime.BODY_KIND_IMPORTANT,
                vector_runtime.BODY_KIND_DEATH,
            ],
            {
                "body_hits": 1,
                "death_points_inserted": 1,
                "print_manager_toggle_updates": 1,
                "print_manager_visual_clears": 0,
            },
            [0.0, 1.0, 1.0],
        ),
    ],
    ids=[
        "hole-space-safe",
        "stored-body-still-kills",
        "print-to-hole-boundary-kills",
        "hole-to-print-boundary-kills",
    ],
)
def test_trail_gap_body_semantics_reach_public_surface_and_replay(
    scenario_name: str,
    expected_alive: list[bool],
    expected_death_count: int,
    expected_death_player: list[int],
    expected_death_cause: list[int],
    expected_death_hit_owner: list[int],
    expected_printing: list[bool],
    expected_visible_trail_count: list[int],
    expected_has_visible_trail_last: list[bool],
    expected_body_owner: list[int],
    expected_body_insert_kind: list[int] | None,
    expected_counters: dict[str, int],
    expected_surface_reward: list[float],
) -> None:
    public_env, public_steps = _run_public_fixture(scenario_name, body_capacity=8)
    public = public_steps[0]

    _assert_death_metadata(
        public.info,
        alive=expected_alive,
        death_count=expected_death_count,
        death_player=expected_death_player,
        death_cause=expected_death_cause,
        death_hit_owner=expected_death_hit_owner,
        score=[0, 0, 0],
    )
    np.testing.assert_array_equal(public.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(public.reward, np.zeros((1, 3), dtype=np.float32))
    np.testing.assert_array_equal(
        public.info["final_observation_row_mask"],
        np.asarray([False], dtype=bool),
    )
    _assert_trail_gap_state(
        public_env,
        expected_printing=expected_printing,
        expected_visible_trail_count=expected_visible_trail_count,
        expected_has_visible_trail_last=expected_has_visible_trail_last,
        expected_body_owner=expected_body_owner,
        expected_body_insert_kind=expected_body_insert_kind,
    )
    _assert_selected_step_counters(public.info, expected_counters)
    expected_die_events = (
        []
        if expected_death_count == 0
        else [(0, expected_death_hit_owner[0], 0)]
    )
    assert _debug_die_events(public_env) == expected_die_events

    surface_env, surface_steps = _run_surface_fixture_with_env(
        scenario_name,
        body_capacity=8,
    )
    surface = surface_steps[0]

    _assert_death_metadata(
        surface.info,
        alive=expected_alive,
        death_count=expected_death_count,
        death_player=expected_death_player,
        death_cause=expected_death_cause,
        death_hit_owner=expected_death_hit_owner,
        score=[0, 0, 0],
    )
    np.testing.assert_array_equal(surface.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        surface.final_observation_row_mask,
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        surface.reward,
        np.asarray([expected_surface_reward], dtype=np.float32),
    )
    np.testing.assert_array_equal(surface.final_reward_map, np.zeros((1, 3)))
    np.testing.assert_array_equal(
        surface.live_mask,
        np.asarray([expected_alive], dtype=bool),
    )
    np.testing.assert_array_equal(
        surface.policy_env_row,
        np.zeros(sum(expected_alive), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        surface.policy_player,
        np.flatnonzero(np.asarray(expected_alive, dtype=bool)).astype(np.int16),
    )
    _assert_trail_gap_state(
        surface_env,
        expected_printing=expected_printing,
        expected_visible_trail_count=expected_visible_trail_count,
        expected_has_visible_trail_last=expected_has_visible_trail_last,
        expected_body_owner=expected_body_owner,
        expected_body_insert_kind=expected_body_insert_kind,
    )

    chunk = _record_chunk(surface_steps, source_ref=scenario_name.removesuffix(".json"))
    assert chunk.metadata["closed_by_terminal"] is False
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"][:, 0],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        chunk.arrays["reward"][0, 0],
        np.asarray(expected_surface_reward, dtype=np.float32),
    )
    assert int(np.count_nonzero(chunk.arrays["final_observation"])) == 0
    record = chunk.records[0]
    assert record["terminal_or_final"] is False
    assert record["final_observation_rows"] == []
    assert record["alive"] == [expected_alive]
    assert record["death_player"] == [expected_death_player]
    assert record["death_hit_owner"] == [expected_death_hit_owner]
    assert record["death_cause"] == [expected_death_cause]
    assert record["policy_player"] == [
        int(player) for player in np.flatnonzero(np.asarray(expected_alive, dtype=bool))
    ]
    for key, value in expected_counters.items():
        assert record["step_counters"][key] == value


def test_borderless_destination_body_fixture_skips_then_kills_through_replay():
    public_env, public_steps = _run_public_fixture(
        "source_borderless_wrap_skips_destination_body_then_next_frame_kills.json",
        body_capacity=8,
    )
    first, second = public_steps

    _assert_death_metadata(
        first.info,
        alive=[True, True, True],
        death_count=0,
        death_player=[-1, -1, -1],
        death_cause=[
            vector_runtime.DEATH_CAUSE_NONE,
            vector_runtime.DEATH_CAUSE_NONE,
            vector_runtime.DEATH_CAUSE_NONE,
        ],
        death_hit_owner=[-1, -1, -1],
    )
    np.testing.assert_allclose(public_env.state["pos"][0, 0], np.asarray([0.0, 44.0]))
    assert first.info["step_counters"]["borderless_wraps"] == 1
    assert first.info["step_counters"]["body_hits"] == 0

    _assert_death_metadata(
        second.info,
        alive=[False, True, True],
        death_count=1,
        death_player=[0, -1, -1],
        death_cause=[
            vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
            vector_runtime.DEATH_CAUSE_NONE,
            vector_runtime.DEATH_CAUSE_NONE,
        ],
        death_hit_owner=[1, -1, -1],
    )
    np.testing.assert_array_equal(second.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(public_env.state["world_body_count"], np.asarray([2]))
    assert second.info["step_counters"]["body_hits"] == 1
    assert second.info["step_counters"]["death_points_inserted"] == 1
    assert _debug_die_events(public_env) == [(0, 1, 0)]

    surface_steps = _run_surface_fixture(
        "source_borderless_wrap_skips_destination_body_then_next_frame_kills.json",
        body_capacity=8,
    )
    surface_first, surface_second = surface_steps
    np.testing.assert_array_equal(
        surface_first.reward,
        np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        surface_second.reward,
        np.asarray([[0.0, 1.0, 1.0]], dtype=np.float32),
    )
    _assert_death_metadata(
        surface_second.info,
        alive=[False, True, True],
        death_count=1,
        death_player=[0, -1, -1],
        death_cause=[
            vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
            vector_runtime.DEATH_CAUSE_NONE,
            vector_runtime.DEATH_CAUSE_NONE,
        ],
        death_hit_owner=[1, -1, -1],
    )

    chunk = _record_chunk(
        surface_steps,
        source_ref="source_borderless_wrap_skips_destination_body_then_next_frame_kills",
    )
    assert chunk.metadata["closed_by_terminal"] is False
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"][:, 0],
        np.asarray([False, False], dtype=bool),
    )
    assert chunk.records[0]["step_counters"]["borderless_wraps"] == 1
    assert chunk.records[0]["death_player"] == [[-1, -1, -1]]
    assert chunk.records[1]["step_counters"]["body_hits"] == 1
    assert chunk.records[1]["death_player"] == [[0, -1, -1]]
    assert chunk.records[1]["death_hit_owner"] == [[1, -1, -1]]


def test_4p_ordered_wall_fixture_preserves_death_order_in_surface_replay():
    public_env, public_steps = _run_public_fixture(
        "source_normal_wall_4p_ordered_deaths_survivor_score.json",
        body_capacity=8,
    )
    first, second, terminal = public_steps

    _assert_death_metadata(
        first.info,
        alive=[True, False, True, True],
        death_count=1,
        death_player=[1, -1, -1, -1],
        death_cause=[
            vector_runtime.DEATH_CAUSE_WALL,
            vector_runtime.DEATH_CAUSE_NONE,
            vector_runtime.DEATH_CAUSE_NONE,
            vector_runtime.DEATH_CAUSE_NONE,
        ],
        death_hit_owner=[-1, -1, -1, -1],
    )
    _assert_death_metadata(
        second.info,
        alive=[True, False, False, True],
        death_count=2,
        death_player=[1, 2, -1, -1],
        death_cause=[
            vector_runtime.DEATH_CAUSE_WALL,
            vector_runtime.DEATH_CAUSE_WALL,
            vector_runtime.DEATH_CAUSE_NONE,
            vector_runtime.DEATH_CAUSE_NONE,
        ],
        death_hit_owner=[-1, -1, -1, -1],
    )
    _assert_death_metadata(
        terminal.info,
        alive=[True, False, False, False],
        death_count=3,
        death_player=[1, 2, 3, -1],
        death_cause=[
            vector_runtime.DEATH_CAUSE_WALL,
            vector_runtime.DEATH_CAUSE_WALL,
            vector_runtime.DEATH_CAUSE_WALL,
            vector_runtime.DEATH_CAUSE_NONE,
        ],
        death_hit_owner=[-1, -1, -1, -1],
        score=[3, 0, 1, 2],
    )
    np.testing.assert_array_equal(terminal.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        terminal.reward,
        np.asarray([[1.0, -1.0, -1.0, -1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        terminal.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    assert terminal.info["winner_ids"] == [[0]]
    assert terminal.info["loser_ids"] == [[1, 2, 3]]
    assert [step.info["step_counters"]["normal_wall_deaths"] for step in public_steps] == [
        1,
        1,
        1,
    ]
    np.testing.assert_array_equal(public_env.state["world_body_count"], np.asarray([3]))

    surface_steps = _run_surface_fixture(
        "source_normal_wall_4p_ordered_deaths_survivor_score.json",
        body_capacity=8,
    )
    surface_terminal = surface_steps[-1]
    _assert_death_metadata(
        surface_terminal.info,
        alive=[True, False, False, False],
        death_count=3,
        death_player=[1, 2, 3, -1],
        death_cause=[
            vector_runtime.DEATH_CAUSE_WALL,
            vector_runtime.DEATH_CAUSE_WALL,
            vector_runtime.DEATH_CAUSE_WALL,
            vector_runtime.DEATH_CAUSE_NONE,
        ],
        death_hit_owner=[-1, -1, -1, -1],
        score=[3, 0, 1, 2],
    )
    np.testing.assert_array_equal(surface_terminal.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        surface_terminal.reward,
        np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(surface_terminal.final_reward_map, surface_terminal.reward)
    np.testing.assert_array_equal(
        surface_terminal.final_observation_row_mask,
        np.asarray([True], dtype=bool),
    )

    chunk = _record_chunk(
        surface_steps,
        source_ref="source_normal_wall_4p_ordered_deaths_survivor_score",
    )
    assert chunk.metadata["closed_by_terminal"] is True
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"][:, 0],
        np.asarray([False, False, True], dtype=bool),
    )
    assert [record["death_player"] for record in chunk.records] == [
        [[1, -1, -1, -1]],
        [[1, 2, -1, -1]],
        [[1, 2, 3, -1]],
    ]
    assert chunk.records[-1]["terminal_or_final"] is True
    assert chunk.records[-1]["winner_ids"] == [[0]]
    assert chunk.records[-1]["loser_ids"] == [[1, 2, 3]]
    assert chunk.records[-1]["score"] == [[3, 0, 1, 2]]


def _run_public_fixture(
    scenario_name: str,
    *,
    body_capacity: int,
):
    fixture, state = _fixture_state(scenario_name, body_capacity=body_capacity)
    env = _fixture_env(state, first_step_ms=_step_ms(fixture, step_index=0))
    steps = []
    for step_index in range(len(fixture["action_schedule"])):
        actions, step_ms, timer_advance_ms = _fixture_step(fixture, step_index=step_index)
        env.decision_ms = step_ms
        steps.append(env.step(actions, timer_advance_ms=timer_advance_ms))
    return env, steps


def _run_surface_fixture(
    scenario_name: str,
    *,
    body_capacity: int,
) -> list[MultiplayerTrainerStepV0]:
    _surface, steps = _run_surface_fixture_with_env(
        scenario_name,
        body_capacity=body_capacity,
    )
    return steps


def _run_surface_fixture_with_env(
    scenario_name: str,
    *,
    body_capacity: int,
) -> tuple[VectorMultiplayerEnv, list[MultiplayerTrainerStepV0]]:
    fixture, state = _fixture_state(scenario_name, body_capacity=body_capacity)
    env = _fixture_env(state, first_step_ms=_step_ms(fixture, step_index=0))
    surface = SourceStateMultiplayerTrainerSurface(env=env)
    surface.stack.reset_rows(env, np.asarray([True], dtype=bool))
    steps = []
    for step_index in range(len(fixture["action_schedule"])):
        actions, step_ms, timer_advance_ms = _fixture_step(fixture, step_index=step_index)
        surface.env.decision_ms = step_ms
        steps.append(surface.step(actions, timer_advance_ms=timer_advance_ms))
    return surface.env, steps


def _fixture_state(
    scenario_name: str,
    *,
    body_capacity: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    scenario_path = f"scenarios/environment/{scenario_name}"
    fixture = seed_bridge.seed_fixture(scenario_path, body_capacity=body_capacity)
    assert fixture["verification"]["source_fidelity_required"] is True
    state = vector_compare.array_state_from_seed(fixture)
    return fixture, state


def _fixture_env(
    state: dict[str, np.ndarray],
    *,
    first_step_ms: float,
) -> VectorMultiplayerEnv:
    player_count = int(state["pos"].shape[1])
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=player_count,
        decision_ms=first_step_ms if first_step_ms > 0.0 else 1.0,
        body_capacity=int(state["body_active"].shape[1]),
        event_capacity=16,
        timer_capacity=max(4, player_count),
        random_tape_capacity=int(state["random_tape_values"].shape[1]),
        event_mode="debug-event",
    )
    env.reset_from_state_arrays(state, reset_seed=np.asarray([101], dtype=np.uint64))
    return env


def _fixture_step(
    fixture: dict[str, object],
    *,
    step_index: int,
) -> tuple[np.ndarray, float, float]:
    prepared_step = vector_compare.prepare_fixture_array_step(
        fixture,
        step_index=step_index,
    )
    source_moves = np.asarray(prepared_step["source_moves"], dtype=np.int8)
    actions = source_moves.astype(np.int16).reshape(1, -1) + 1
    return (
        actions,
        float(prepared_step["step_ms"]),
        float(prepared_step.get("timer_advance_ms", 0.0)),
    )


def _step_ms(fixture: dict[str, object], *, step_index: int) -> float:
    _actions, step_ms, _timer_advance_ms = _fixture_step(fixture, step_index=step_index)
    return step_ms


def _assert_death_metadata(
    info: dict[str, object],
    *,
    alive: list[bool],
    death_count: int,
    death_player: list[int],
    death_cause: list[int],
    death_hit_owner: list[int],
    score: list[int] | None = None,
) -> None:
    np.testing.assert_array_equal(info["alive"], np.asarray([alive], dtype=bool))
    np.testing.assert_array_equal(
        info["death_count"],
        np.asarray([death_count], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        info["death_player"],
        np.asarray([death_player], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        info["death_cause"],
        np.asarray([death_cause], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        info["death_hit_owner"],
        np.asarray([death_hit_owner], dtype=np.int16),
    )
    if score is not None:
        np.testing.assert_array_equal(info["score"], np.asarray([score], dtype=np.int32))


def _assert_trail_gap_state(
    env: VectorMultiplayerEnv,
    *,
    expected_printing: list[bool],
    expected_visible_trail_count: list[int],
    expected_has_visible_trail_last: list[bool],
    expected_body_owner: list[int],
    expected_body_insert_kind: list[int] | None,
) -> None:
    body_count = len(expected_body_owner)
    np.testing.assert_array_equal(
        env.state["printing"],
        np.asarray([expected_printing], dtype=bool),
    )
    np.testing.assert_array_equal(
        env.state["visible_trail_count"],
        np.asarray([expected_visible_trail_count], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        env.state["has_visible_trail_last"],
        np.asarray([expected_has_visible_trail_last], dtype=bool),
    )
    np.testing.assert_array_equal(
        env.state["world_body_count"],
        np.asarray([body_count], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        env.state["body_owner"][:, :body_count],
        np.asarray([expected_body_owner], dtype=np.int16),
    )
    if expected_body_insert_kind is not None:
        np.testing.assert_array_equal(
            env.state["body_insert_kind"][:, :body_count],
            np.asarray([expected_body_insert_kind], dtype=np.int16),
        )


def _assert_selected_step_counters(
    info: dict[str, object],
    expected_counters: dict[str, int],
) -> None:
    for key, value in expected_counters.items():
        assert info["step_counters"][key] == value


def _debug_die_events(env: VectorMultiplayerEnv) -> list[tuple[int, int, int]]:
    count = int(env.state["event_count"][0])
    return [
        (
            int(env.state["event_player"][0, index]),
            int(env.state["event_other"][0, index]),
            int(env.state["event_bool"][0, index]),
        )
        for index in range(count)
        if int(env.state["event_type"][0, index]) == vector_runtime.EVENT_DIE
    ]


def _record_chunk(
    steps: list[MultiplayerTrainerStepV0],
    *,
    source_ref: str,
):
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    for index, step in enumerate(steps):
        recorder.record(step, source_ref=f"{source_ref}#{index}")
    return recorder.build_chunk()
