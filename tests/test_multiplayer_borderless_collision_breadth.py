import json
import sys
from pathlib import Path

import numpy as np
import pytest

from curvyzero.env import vector_reset
from curvyzero.env import vector_runtime
from curvyzero.env.scenarios import run_source_borderless_wrap_scenario
from curvyzero.env.trace_compare import project_common_trace
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


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ROOT = REPO_ROOT / "scenarios" / "environment"
SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import compare_vector_arrays_to_fidelity as vector_compare  # noqa: E402
import seed_vector_state_from_fixtures as seed_bridge  # noqa: E402


def test_death_point_fixture_kills_later_player_through_public_surface_and_replay():
    manifest = json.loads((SCENARIO_ROOT / "source_collision_order_batch.json").read_text())
    assert (
        "scenarios/environment/source_collision_death_point_kills_later_player_step.json"
        in manifest["scenarios"]
    )

    public_env, public_steps = _run_public_fixture(
        "source_collision_death_point_kills_later_player_step.json",
        body_capacity=8,
    )
    public = public_steps[0]

    _assert_death_metadata(
        public.info,
        alive=[False, False],
        death_count=2,
        death_player=[1, 0],
        death_cause=[
            vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
            vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
        ],
        death_hit_owner=[0, 1],
        score=[0, 0],
    )
    np.testing.assert_array_equal(public.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(public.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(public.truncated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(public.reward, np.zeros((1, 2), dtype=np.float32))
    np.testing.assert_array_equal(
        public.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW], dtype=np.int16),
    )
    np.testing.assert_array_equal(public.info["draw"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(public.info["winner"], np.asarray([-1], dtype=np.int16))
    np.testing.assert_array_equal(
        public.info["final_observation_row_mask"],
        np.asarray([True], dtype=bool),
    )
    assert public.info["winner_ids"] == [[]]
    assert public.info["loser_ids"] == [[]]
    assert public.final_observation is not None
    assert public.final_reward is not None
    np.testing.assert_array_equal(public.final_reward, public.reward)
    np.testing.assert_array_equal(public_env.state["world_body_count"], np.asarray([3]))
    assert public.info["step_counters"]["body_hits"] == 2
    assert public.info["step_counters"]["death_points_inserted"] == 2
    assert public.info["step_counters"]["terminal_score_rows"] == 1
    assert _debug_die_events(public_env) == [(1, 0, 0), (0, 1, 0)]

    surface_steps = _run_surface_fixture(
        "source_collision_death_point_kills_later_player_step.json",
        body_capacity=8,
    )
    surface = surface_steps[0]

    _assert_death_metadata(
        surface.info,
        alive=[False, False],
        death_count=2,
        death_player=[1, 0],
        death_cause=[
            vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
            vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
        ],
        death_hit_owner=[0, 1],
        score=[0, 0],
    )
    np.testing.assert_array_equal(surface.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(surface.reward, np.zeros((1, 2), dtype=np.float32))
    np.testing.assert_array_equal(surface.final_reward_map, surface.reward)
    np.testing.assert_array_equal(
        surface.final_observation_row_mask,
        np.asarray([True], dtype=bool),
    )
    assert int(np.count_nonzero(surface.final_observation[0])) > 0

    chunk = _record_chunk(
        surface_steps,
        source_ref="source_collision_death_point_kills_later_player_step",
    )
    assert chunk.metadata["closed_by_terminal"] is True
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"],
        np.asarray([[True]], dtype=bool),
    )
    record = chunk.records[0]
    assert record["terminal_or_final"] is True
    assert record["final_observation_rows"] == [0]
    assert record["death_player"] == [[1, 0]]
    assert record["death_hit_owner"] == [[0, 1]]
    assert record["death_cause_name"] == [["opponent_trail", "opponent_trail"]]
    assert record["winner_ids"] == [[]]
    assert record["loser_ids"] == [[]]
    assert record["step_counters"]["body_hits"] == 2
    assert record["step_counters"]["death_points_inserted"] == 2


@pytest.mark.parametrize(
    (
        "scenario_name",
        "expected_positions",
        "expected_position_events",
    ),
    [
        (
            "source_borderless_wrap_step.json",
            [[0.0, 44.0], [42.4, 44.0]],
            [
                (vector_runtime.EVENT_POSITION, 1, [42.4, 44.0]),
                (vector_runtime.EVENT_POSITION, 0, [88.95, 44.0]),
                (vector_runtime.EVENT_POSITION, 0, [0.0, 44.0]),
            ],
        ),
        (
            "source_borderless_exact_edge_corner_axis_step.json",
            [[0.0, 88.481371], [88.0, 20.0]],
            [
                (vector_runtime.EVENT_POSITION, 1, [88.0, 20.0]),
                (vector_runtime.EVENT_POSITION, 0, [88.481371, 88.481371]),
                (vector_runtime.EVENT_POSITION, 0, [0.0, 88.481371]),
            ],
        ),
    ],
    ids=["plain-wrap", "exact-edge-corner-axis"],
)
def test_borderless_wrap_fixtures_reach_public_surface_and_replay(
    scenario_name: str,
    expected_positions: list[list[float]],
    expected_position_events: list[tuple[int, int, list[float]]],
) -> None:
    public_env, public_steps = _run_public_fixture(scenario_name, body_capacity=8)
    public = public_steps[0]

    _assert_death_metadata(
        public.info,
        alive=[True, True],
        death_count=0,
        death_player=[-1, -1],
        death_cause=[
            vector_runtime.DEATH_CAUSE_NONE,
            vector_runtime.DEATH_CAUSE_NONE,
        ],
        death_hit_owner=[-1, -1],
        score=[0, 0],
    )
    np.testing.assert_array_equal(public.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(public.reward, np.zeros((1, 2), dtype=np.float32))
    np.testing.assert_array_equal(
        public.info["final_observation_row_mask"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(public_env.state["borderless"], np.asarray([True]))
    np.testing.assert_allclose(
        public_env.state["pos"][0],
        np.asarray(expected_positions, dtype=np.float64),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(public_env.state["world_body_count"], np.asarray([0]))
    assert public.info["step_counters"]["borderless_wraps"] == 1
    assert public.info["step_counters"]["body_hits"] == 0
    assert public.info["step_counters"]["normal_wall_deaths"] == 0
    assert _position_events(public_env) == expected_position_events

    surface_steps = _run_surface_fixture(scenario_name, body_capacity=8)
    surface = surface_steps[0]
    _assert_death_metadata(
        surface.info,
        alive=[True, True],
        death_count=0,
        death_player=[-1, -1],
        death_cause=[
            vector_runtime.DEATH_CAUSE_NONE,
            vector_runtime.DEATH_CAUSE_NONE,
        ],
        death_hit_owner=[-1, -1],
        score=[0, 0],
    )
    np.testing.assert_array_equal(surface.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        surface.reward,
        np.asarray([[1.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        surface.final_observation_row_mask,
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(surface.live_mask, np.asarray([[True, True]]))
    np.testing.assert_array_equal(surface.policy_env_row, np.asarray([0, 0]))
    np.testing.assert_array_equal(surface.policy_player, np.asarray([0, 1]))

    chunk = _record_chunk(surface_steps, source_ref=scenario_name.removesuffix(".json"))
    assert chunk.metadata["closed_by_terminal"] is False
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"],
        np.asarray([[False]], dtype=bool),
    )
    record = chunk.records[0]
    assert record["terminal_or_final"] is False
    assert record["final_observation_rows"] == []
    assert record["alive"] == [[True, True]]
    assert record["borderless"] == [True]
    assert record["death_player"] == [[-1, -1]]
    assert record["death_hit_owner"] == [[-1, -1]]
    assert record["death_cause_name"] == [["none", "none"]]
    assert record["step_counters"]["borderless_wraps"] == 1
    assert record["step_counters"]["body_hits"] == 0


def test_source_print_manager_borderless_wrap_toggle_preserves_source_events():
    payload = run_source_borderless_wrap_scenario(
        SCENARIO_ROOT / "source_borderless_print_manager_wrap_toggle_step.json"
    ).to_payload()
    frame = payload["trace"]["frames"][1]
    common_step = project_common_trace(payload)["steps"][0]

    assert payload["runner"] == "curvytron-v1-python-source-borderless-wrap-runner"
    assert payload["source_fidelity"] is True
    assert frame["positions"] == [[0.0, 40.0]]
    assert frame["alive"] == [True]
    assert frame["worldBodyCount"] == 2
    assert frame["trailPointCounts"] == [0]
    assert frame["lastTrailPoints"] == [None]
    assert frame["bodyNums"] == [1]
    assert frame["bodyCounts"] == [2]
    assert frame["printing"] == [False]
    assert frame["printManagers"] == [
        {"active": True, "distance": 5.25, "lastX": 0.0, "lastY": 40.0}
    ]
    expected_events = [
        {"event": "position", "player_id": "p0", "x": 80.12, "y": 40.0},
        {
            "event": "point",
            "player_id": "p0",
            "x": 80.12,
            "y": 40.0,
            "important": False,
        },
        {"event": "position", "player_id": "p0", "x": 0.0, "y": 40.0},
        {
            "event": "point",
            "player_id": "p0",
            "x": 0.0,
            "y": 40.0,
            "important": True,
        },
        {
            "event": "property",
            "player_id": "p0",
            "property": "printing",
            "value": False,
        },
    ]
    assert frame["events"] == expected_events
    assert common_step["events"] == expected_events


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
    fixture, state = _fixture_state(scenario_name, body_capacity=body_capacity)
    env = _fixture_env(state, first_step_ms=_step_ms(fixture, step_index=0))
    surface = SourceStateMultiplayerTrainerSurface(env=env)
    surface.stack.reset_rows(env, np.asarray([True], dtype=bool))
    steps = []
    for step_index in range(len(fixture["action_schedule"])):
        actions, step_ms, timer_advance_ms = _fixture_step(fixture, step_index=step_index)
        surface.env.decision_ms = step_ms
        steps.append(surface.step(actions, timer_advance_ms=timer_advance_ms))
    return steps


def _fixture_state(
    scenario_name: str,
    *,
    body_capacity: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    scenario_path = f"scenarios/environment/{scenario_name}"
    fixture = seed_bridge.seed_fixture(scenario_path, body_capacity=body_capacity)
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


def _position_events(env: VectorMultiplayerEnv) -> list[tuple[int, int, list[float]]]:
    count = int(env.state["event_count"][0])
    events: list[tuple[int, int, list[float]]] = []
    for index in range(count):
        event_type = int(env.state["event_type"][0, index])
        if event_type != vector_runtime.EVENT_POSITION:
            continue
        events.append(
            (
                event_type,
                int(env.state["event_player"][0, index]),
                np.round(env.state["event_value_f"][0, index], 6).tolist(),
            )
        )
    return events


def _record_chunk(
    steps: list[MultiplayerTrainerStepV0],
    *,
    source_ref: str,
):
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    for index, step in enumerate(steps):
        recorder.record(step, source_ref=f"{source_ref}#{index}")
    return recorder.build_chunk()
