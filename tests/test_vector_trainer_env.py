import json
from pathlib import Path

import numpy as np
import pytest

from curvyzero.env import VectorTrainerEnv1v1NoBonus
from curvyzero.env.source_env import CurvyTronSourceEnv
from curvyzero.env.vector_trainer_env import VectorTrainerEnvError
from curvyzero.env import vector_reset
from curvyzero.env import vector_runtime
from curvyzero.env.trainer_contract import NATIVE_CONTROL_MODEL_ID
from curvyzero.env.trainer_contract import TRAINER_CONTROL_WRAPPER_ID


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCENARIO_DIR = _REPO_ROOT / "scenarios" / "environment"
_LONG_NATURAL_ROLLOUT_SCENARIO = (
    _SCENARIO_DIR / "source_lifecycle_long_1v1_no_bonus_wall_round_done.json"
)
_SOURCE_NORMAL_WALL_SAME_FRAME_DRAW = (
    _SCENARIO_DIR / "source_normal_wall_same_frame_draw_step.json"
)
_SOURCE_BORDERLESS_PRINT_MANAGER_WRAP_TOGGLE = (
    _SCENARIO_DIR / "source_borderless_print_manager_wrap_toggle_step.json"
)
_SOURCE_COLLISION_DEATH_POINT_KILLS_LATER_PLAYER = (
    _SCENARIO_DIR / "source_collision_death_point_kills_later_player_step.json"
)
_SOURCE_COLLISION_HEAD_HEAD_REVERSE_ORDER_SINGLE_DEATH = (
    _SCENARIO_DIR / "source_collision_head_head_reverse_order_single_death_step.json"
)
_DECISION_MS_60HZ = 1000.0 / 60.0


def _load_scenario(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_source_long_1v1_no_bonus_wall_rollout() -> tuple[
    CurvyTronSourceEnv,
    dict[str, object],
    list[dict[str, object]],
]:
    scenario = _load_scenario(_LONG_NATURAL_ROLLOUT_SCENARIO)
    source_setup = scenario["source_setup"]
    assert isinstance(source_setup, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)
    room = source_setup["room"]
    assert isinstance(room, dict)
    lifecycle = scenario["lifecycle"]
    assert isinstance(lifecycle, dict)
    rollout = scenario["rollout"]
    assert isinstance(rollout, dict)

    env = CurvyTronSourceEnv(
        random_values=random_setup["math_random_sequence"],
        max_score=float(room["max_score"]),
        include_deaths_snapshot=True,
    )
    env.reset(
        player_count=int(scenario["player_count"]),
        players=scenario["players"],
        warmup_ms=float(lifecycle["new_round_time_ms"]),
    )

    records = []
    for _tick in range(int(rollout["step_count"])):
        env.advance_timers(float(rollout["advance_timers_ms"]))
        frame = env.step(rollout["moves"], elapsed_ms=float(rollout["step_ms"]))
        records.append(
            {
                "frame": frame,
                "world_bodies": env.world_bodies_snapshot(),
                "avatar_body_metadata": env.avatar_body_metadata_snapshot(),
            }
        )
    return env, scenario, records


def _source_avatars(frame: dict[str, object]) -> list[dict[str, object]]:
    avatars = frame["avatars"]
    assert isinstance(avatars, list)
    return avatars


def _source_positions(frame: dict[str, object]) -> list[list[float]]:
    return [[float(avatar["x"]), float(avatar["y"])] for avatar in _source_avatars(frame)]


def _source_alive(frame: dict[str, object]) -> list[bool]:
    return [bool(avatar["alive"]) for avatar in _source_avatars(frame)]


def _source_scores(frame: dict[str, object]) -> list[int]:
    return [int(avatar["score"]) for avatar in _source_avatars(frame)]


def _source_printing(frame: dict[str, object]) -> list[bool]:
    return [bool(avatar["printing"]) for avatar in _source_avatars(frame)]


def _source_print_manager_active(frame: dict[str, object]) -> list[bool]:
    active = []
    for avatar in _source_avatars(frame):
        print_manager = avatar["printManager"]
        assert isinstance(print_manager, dict)
        active.append(bool(print_manager["active"]))
    return active


def _assert_public_row_matches_source_record(
    env: VectorTrainerEnv1v1NoBonus,
    record: dict[str, object],
    *,
    tick: int,
) -> None:
    frame = record["frame"]
    assert isinstance(frame, dict)
    game = frame["game"]
    assert isinstance(game, dict)
    np.testing.assert_allclose(
        env.state["pos"][0],
        np.asarray(_source_positions(frame), dtype=np.float64),
        atol=1e-6,
        err_msg=f"tick {tick} positions",
    )
    np.testing.assert_array_equal(
        env.state["alive"][0],
        np.asarray(_source_alive(frame), dtype=bool),
        err_msg=f"tick {tick} alive",
    )
    np.testing.assert_array_equal(
        env.state["score"][0],
        np.asarray(_source_scores(frame), dtype=np.int32),
        err_msg=f"tick {tick} score",
    )
    np.testing.assert_array_equal(
        env.state["printing"][0],
        np.asarray(_source_printing(frame), dtype=bool),
        err_msg=f"tick {tick} printing",
    )
    np.testing.assert_array_equal(
        env.state["print_manager_active"][0],
        np.asarray(_source_print_manager_active(frame), dtype=bool),
        err_msg=f"tick {tick} print manager active",
    )
    assert bool(env.state["started"][0]) is bool(game["started"]), f"tick {tick} started"
    assert bool(env.state["in_round"][0]) is bool(game["inRound"]), f"tick {tick} in_round"
    assert bool(env.state["world_active"][0]) is bool(
        game["worldActive"]
    ), f"tick {tick} world_active"
    assert int(env.state["world_body_count"][0]) == int(
        game["worldBodyCount"]
    ), f"tick {tick} world_body_count"

    world_bodies = record["world_bodies"]
    assert isinstance(world_bodies, tuple)
    avatars = _source_avatars(frame)
    player_index_by_avatar_id = {
        int(avatar["id"]): index
        for index, avatar in enumerate(avatars)
    }
    active_slots = np.flatnonzero(env.state["body_active"][0])
    assert active_slots.size == len(world_bodies), f"tick {tick} body count"
    for slot, body in enumerate(world_bodies):
        assert isinstance(body, dict)
        assert int(active_slots[slot]) == slot, f"tick {tick} body slot {slot}"
        np.testing.assert_allclose(
            env.state["body_pos"][0, slot],
            np.asarray([float(body["x"]), float(body["y"])], dtype=np.float64),
            atol=1e-6,
            err_msg=f"tick {tick} body {slot} position",
        )
        assert float(env.state["body_radius"][0, slot]) == pytest.approx(
            float(body["radius"])
        ), f"tick {tick} body {slot} radius"
        avatar_id = body["avatarId"]
        expected_owner = (
            -1
            if avatar_id is None
            else player_index_by_avatar_id[int(avatar_id)]
        )
        assert int(env.state["body_owner"][0, slot]) == expected_owner, (
            f"tick {tick} body {slot} owner"
        )
        assert int(env.state["body_num"][0, slot]) == int(body["num"]), (
            f"tick {tick} body {slot} num"
        )


def _source_round_winner(events: list[dict[str, object]]) -> int | None:
    round_end_events = [event for event in events if event["event"] == "round:end"]
    assert len(round_end_events) == 1
    data = round_end_events[0]["data"]
    assert isinstance(data, dict)
    winner = data["winner"]
    assert winner is None or isinstance(winner, int)
    return winner


def _seed_public_env_row_from_initial_players(
    env: VectorTrainerEnv1v1NoBonus,
    scenario: dict[str, object],
    *,
    players: list[dict[str, object]] | None = None,
) -> None:
    env.reset(seed=np.asarray([0], dtype=np.uint64))
    row = 0
    for name, template_array in env.reset_template.items():
        env.state[name][row, ...] = template_array[row, ...]

    source_setup = scenario["source_setup"]
    assert isinstance(source_setup, dict)
    game = source_setup["game"]
    assert isinstance(game, dict)
    selected_players = scenario["players"] if players is None else players
    assert isinstance(selected_players, list)
    assert len(selected_players) == 2

    env.state["episode_id"][row] = 1
    env.state["episode_step"][row] = 0
    env.state["tick"][row] = 0
    env.state["elapsed_ms"][row] = 0.0
    env.state["done"][row] = False
    env.state["terminated"][row] = False
    env.state["truncated"][row] = False
    env.state["reset_pending"][row] = False
    env.state["terminal_reason"][row] = vector_reset.TERMINAL_REASON_NONE
    env.state["winner"][row] = -1
    env.state["draw"][row] = False
    env.state["started"][row] = bool(game["started"])
    env.state["in_round"][row] = bool(game["in_round"])
    env.state["world_active"][row] = bool(game["world_active"])
    env.state["borderless"][row] = bool(game["borderless"])
    env.state["map_size"][row] = float(source_setup["map_size"])
    env.state["random_tape_cursor"][row] = 0
    env.state["random_tape_draw_count"][row] = 0
    env.state["random_tape_exhausted"][row] = False

    random_setup = source_setup.get("random")
    if isinstance(random_setup, dict) and "math_random" in random_setup:
        env.state["random_tape_length"][row] = 1
        env.state["random_tape_values"][row, 0] = float(random_setup["math_random"])

    for index, player in enumerate(selected_players):
        assert isinstance(player, dict)
        initial = player["initial"]
        assert isinstance(initial, dict)
        env.state["pos"][row, index] = [float(initial["x"]), float(initial["y"])]
        env.state["prev_pos"][row, index] = env.state["pos"][row, index]
        env.state["heading"][row, index] = float(initial["angle_rad"])
        env.state["alive"][row, index] = bool(initial.get("alive", True))
        env.state["present"][row, index] = bool(initial.get("present", True))
        env.state["score"][row, index] = int(initial.get("score", 0))
        env.state["round_score"][row, index] = int(initial.get("round_score", 0))
        env.state["printing"][row, index] = bool(initial["printing"])
        env.state["live_body_num"][row, index] = int(initial.get("body_num", 0))
        env.state["body_count"][row, index] = int(initial.get("body_count", 0))
        env.state["visible_trail_count"][row, index] = 0
        env.state["has_visible_trail_last"][row, index] = False
        env.state["has_draw_cursor"][row, index] = False

        trail = initial.get("trail")
        if isinstance(trail, dict) and "last_x" in trail and "last_y" in trail:
            last_pos = [float(trail["last_x"]), float(trail["last_y"])]
            env.state["has_draw_cursor"][row, index] = True
            env.state["draw_cursor_pos"][row, index] = last_pos
            env.state["has_visible_trail_last"][row, index] = True
            env.state["visible_trail_last_pos"][row, index] = last_pos

        print_manager = initial.get("print_manager")
        if isinstance(print_manager, dict):
            env.state["print_manager_active"][row, index] = bool(
                print_manager["active"]
            )
            env.state["print_manager_distance"][row, index] = float(
                print_manager["distance"]
            )
            env.state["print_manager_last_pos"][row, index] = [
                float(print_manager["last_x"]),
                float(print_manager["last_y"]),
            ]
        else:
            env.state["print_manager_active"][row, index] = False
            env.state["print_manager_distance"][row, index] = 0.0
            env.state["print_manager_last_pos"][row, index] = 0.0

    player_index_by_id = {
        str(player["id"]): index
        for index, player in enumerate(selected_players)
    }
    initial_state = scenario.get("initial_state", {})
    assert isinstance(initial_state, dict)
    world_bodies = initial_state.get("world_bodies", [])
    assert isinstance(world_bodies, list)
    for slot, body in enumerate(world_bodies):
        assert isinstance(body, dict)
        owner = player_index_by_id[str(body["player_id"])]
        body_num = int(body["num"])
        env.state["body_active"][row, slot] = True
        env.state["body_pos"][row, slot] = [float(body["x"]), float(body["y"])]
        env.state["body_radius"][row, slot] = float(body["radius"])
        env.state["body_owner"][row, slot] = owner
        env.state["body_num"][row, slot] = body_num
        env.state["body_insert_tick"][row, slot] = 0
        env.state["body_insert_kind"][row, slot] = vector_runtime.BODY_KIND_NORMAL
        env.state["body_count"][row, owner] = max(
            int(env.state["body_count"][row, owner]),
            body_num + 1,
        )
        env.state["body_write_cursor"][row] = slot + 1
    env.state["world_body_count"][row] = len(world_bodies)


def _seed_vector_env_row_from_source_frame(
    env: VectorTrainerEnv1v1NoBonus,
    *,
    frame: dict[str, object],
    world_bodies: tuple[dict[str, object], ...],
    avatar_body_metadata: tuple[dict[str, object], ...],
    tick: int,
) -> None:
    env.reset(seed=np.asarray([0], dtype=np.uint64))
    row = 0
    for name, template_array in env.reset_template.items():
        env.state[name][row, ...] = template_array[row, ...]

    game = frame["game"]
    assert isinstance(game, dict)
    avatars = _source_avatars(frame)
    assert len(avatars) == 2
    metadata_by_id = {
        int(metadata["id"]): metadata
        for metadata in avatar_body_metadata
    }
    player_index_by_avatar_id = {
        int(avatar["id"]): index
        for index, avatar in enumerate(avatars)
    }

    env.state["episode_id"][row] = 1
    env.state["tick"][row] = tick
    env.state["elapsed_ms"][row] = tick * env.decision_ms
    env.state["done"][row] = False
    env.state["terminated"][row] = False
    env.state["truncated"][row] = False
    env.state["reset_pending"][row] = False
    env.state["terminal_reason"][row] = vector_reset.TERMINAL_REASON_NONE
    env.state["started"][row] = bool(game["started"])
    env.state["in_round"][row] = bool(game["inRound"])
    env.state["world_active"][row] = bool(game["worldActive"])
    env.state["map_size"][row] = float(game["size"])
    env.state["borderless"][row] = False

    for index, avatar in enumerate(avatars):
        avatar_id = int(avatar["id"])
        metadata = metadata_by_id[avatar_id]
        env.state["pos"][row, index] = [float(avatar["x"]), float(avatar["y"])]
        env.state["prev_pos"][row, index] = env.state["pos"][row, index]
        env.state["heading"][row, index] = float(avatar["angle"])
        env.state["alive"][row, index] = bool(avatar["alive"])
        env.state["present"][row, index] = bool(avatar["present"])
        env.state["score"][row, index] = int(avatar["score"])
        env.state["round_score"][row, index] = int(avatar["roundScore"])
        env.state["printing"][row, index] = bool(avatar["printing"])
        env.state["radius"][row, index] = float(metadata["radius"])
        env.state["speed"][row, index] = float(metadata["velocity"])
        env.state["live_body_num"][row, index] = int(metadata["bodyNum"])
        env.state["body_count"][row, index] = int(metadata["bodyCount"])
        env.state["trail_latency"][row, index] = int(metadata["trailLatency"])
        env.state["visible_trail_count"][row, index] = int(avatar["trailPointCount"])

        print_manager = avatar["printManager"]
        assert isinstance(print_manager, dict)
        env.state["print_manager_active"][row, index] = bool(print_manager["active"])
        env.state["print_manager_distance"][row, index] = float(print_manager["distance"])
        env.state["print_manager_last_pos"][row, index] = [
            float(print_manager["lastX"]),
            float(print_manager["lastY"]),
        ]

    latest_body_by_player: dict[int, dict[str, object]] = {}
    for slot, body in enumerate(world_bodies):
        avatar_id = body.get("avatarId")
        if avatar_id is None:
            continue
        player = player_index_by_avatar_id[int(avatar_id)]
        body_num = int(body["num"])
        current_latest = latest_body_by_player.get(player)
        if current_latest is None or body_num > int(current_latest["num"]):
            latest_body_by_player[player] = body

        env.state["body_active"][row, slot] = True
        env.state["body_pos"][row, slot] = [float(body["x"]), float(body["y"])]
        env.state["body_radius"][row, slot] = float(body["radius"])
        env.state["body_owner"][row, slot] = player
        env.state["body_num"][row, slot] = body_num
        env.state["body_insert_tick"][row, slot] = 0
        env.state["body_insert_kind"][row, slot] = vector_runtime.BODY_KIND_NORMAL
        env.state["body_write_cursor"][row] = slot + 1

    env.state["world_body_count"][row] = int(game["worldBodyCount"])
    assert int(env.state["body_write_cursor"][row]) == int(env.state["world_body_count"][row])

    for player, body in latest_body_by_player.items():
        latest_pos = [float(body["x"]), float(body["y"])]
        env.state["has_draw_cursor"][row, player] = True
        env.state["draw_cursor_pos"][row, player] = latest_pos
        env.state["has_visible_trail_last"][row, player] = True
        env.state["visible_trail_last_pos"][row, player] = latest_pos


def test_reset_returns_batch_observation_shape_dtype_and_masks():
    env = VectorTrainerEnv1v1NoBonus(batch_size=3, seed=1)

    batch = env.reset(seed=123)

    assert batch.observation.shape == (3, 2, 106)
    assert batch.observation.dtype == np.float32
    assert batch.action_mask.shape == (3, 2, 3)
    assert batch.action_mask.dtype == np.bool_
    assert batch.lightzero_action_mask.dtype == np.int8
    np.testing.assert_array_equal(batch.action_mask, np.ones((3, 2, 3), dtype=bool))
    np.testing.assert_array_equal(batch.reward, np.zeros((3, 2), dtype=np.float32))
    np.testing.assert_array_equal(batch.done, np.zeros(3, dtype=bool))
    np.testing.assert_array_equal(env.state["world_active"], np.ones(3, dtype=bool))
    assert batch.info["episode_id"].shape == (3,)
    assert batch.info["reset_seed"].shape == (3,)
    np.testing.assert_array_equal(
        batch.info["reset_source"],
        np.full(3, vector_reset.RESET_SOURCE_MANUAL, dtype=np.int16),
    )
    assert bool((env.state["random_tape_cursor"] >= 8).all())


def test_reset_scales_warmup_timer_callback_cap_for_larger_batches():
    env = VectorTrainerEnv1v1NoBonus(batch_size=128, seed=1)

    batch = env.reset(seed=np.arange(128, dtype=np.uint64))

    assert batch.observation.shape == (128, 2, 106)
    np.testing.assert_array_equal(batch.done, np.zeros(128, dtype=bool))
    np.testing.assert_array_equal(env.state["world_active"], np.ones(128, dtype=bool))


def test_reset_rejects_invalid_warmup_timer_callback_cap():
    with pytest.raises(VectorTrainerEnvError, match="positive integer"):
        VectorTrainerEnv1v1NoBonus(max_warmup_timer_callbacks=0)


def test_reset_rejects_invalid_source_fixture_reset_policy_inputs():
    env = VectorTrainerEnv1v1NoBonus(batch_size=1, random_tape_capacity=2)

    with pytest.raises(VectorTrainerEnvError, match=r"shape \[B,N\]"):
        env.reset(source_fixture_random_tape_values=np.asarray([0.5], dtype=np.float64))
    with pytest.raises(VectorTrainerEnvError, match=r"\[0, 1\)"):
        env.reset(source_fixture_random_tape_values=np.asarray([[1.0]], dtype=np.float64))
    with pytest.raises(VectorTrainerEnvError, match="finite and nonnegative"):
        env.reset(source_fixture_new_round_time_ms=-1.0)
    with pytest.raises(VectorTrainerEnvError, match="scalar or shape"):
        env.reset(source_fixture_warmup_advance_ms=np.asarray([[0.0]], dtype=np.float64))


def test_nonterminal_step_maps_actions_and_keeps_rows_live():
    env = VectorTrainerEnv1v1NoBonus(batch_size=1, seed=2)
    env.reset(seed=7)
    heading_before = env.state["heading"].copy()

    batch = env.step(np.asarray([[0, 2]], dtype=np.int64))

    assert batch.observation.shape == (1, 2, 106)
    np.testing.assert_array_equal(batch.reward, np.zeros((1, 2), dtype=np.float32))
    np.testing.assert_array_equal(batch.done, np.zeros(1, dtype=bool))
    np.testing.assert_array_equal(batch.info["source_moves"], np.asarray([[-1, 1]], dtype=np.int8))
    assert int(env.state["episode_step"][0]) == 1
    assert env.state["heading"][0, 0] < heading_before[0, 0]
    assert env.state["heading"][0, 1] > heading_before[0, 1]


def test_terminal_wall_death_autoreset_returns_final_reward_and_post_reset_observation():
    env = VectorTrainerEnv1v1NoBonus(batch_size=1, seed=3, decision_ms=300.0)
    env.reset(seed=11)
    env.state["pos"][0, 1] = [env.map_size - 1.0, env.map_size / 2.0]
    env.state["prev_pos"][0, 1] = env.state["pos"][0, 1]
    env.state["heading"][0, 1] = 0.0
    env.state["speed"][0, 1] = 200.0
    terminal_pos = env.state["pos"].copy()

    batch = env.step(np.asarray([[1, 1]], dtype=np.int8))

    np.testing.assert_array_equal(batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.reward, np.asarray([[1.0, -1.0]], dtype=np.float32))
    assert batch.final_observation is not None
    assert batch.final_reward is not None
    np.testing.assert_array_equal(batch.final_reward, np.asarray([[1.0, -1.0]], dtype=np.float32))
    np.testing.assert_allclose(
        batch.final_observation[0],
        batch.info["autoreset_plan"]["final_transition_snapshot"]["final_observation"][0],
    )
    assert bool(env.state["done"][0]) is False
    assert int(env.state["reset_source"][0]) == vector_reset.RESET_SOURCE_AUTORESET
    np.testing.assert_array_equal(batch.action_mask, np.ones((1, 2, 3), dtype=bool))
    assert not np.allclose(env.state["pos"][0], terminal_pos[0])


def test_public_vector_env_matches_source_long_1v1_wall_round_done_terminal_step():
    source_env, scenario, source_records = _run_source_long_1v1_no_bonus_wall_rollout()
    rollout = scenario["rollout"]
    assert isinstance(rollout, dict)
    expected = rollout["expected"]
    assert isinstance(expected, dict)
    assert int(rollout["terminal_tick"]) == len(source_records) - 1

    penultimate = source_records[-2]
    final = source_records[-1]
    penultimate_frame = penultimate["frame"]
    final_frame = final["frame"]
    assert isinstance(penultimate_frame, dict)
    assert isinstance(final_frame, dict)
    assert _source_alive(penultimate_frame) == [True, True]
    assert _source_positions(final_frame) == expected["final_positions"]
    assert _source_alive(final_frame) == expected["alive"]
    assert _source_scores(final_frame) == expected["scores"]

    source_winner_id = _source_round_winner(source_env.events)
    assert source_winner_id == expected["round_winner"]
    final_source_avatars = _source_avatars(final_frame)
    source_winner_index = next(
        index
        for index, avatar in enumerate(final_source_avatars)
        if int(avatar["id"]) == source_winner_id
    )

    public_env = VectorTrainerEnv1v1NoBonus(
        autoreset=False,
        decision_ms=_DECISION_MS_60HZ,
        player_ids=("p0", "p1"),
    )
    world_bodies = penultimate["world_bodies"]
    avatar_body_metadata = penultimate["avatar_body_metadata"]
    assert isinstance(world_bodies, tuple)
    assert isinstance(avatar_body_metadata, tuple)
    _seed_vector_env_row_from_source_frame(
        public_env,
        frame=penultimate_frame,
        world_bodies=world_bodies,
        avatar_body_metadata=avatar_body_metadata,
        tick=int(rollout["terminal_tick"]),
    )

    batch = public_env.step(np.asarray([[1, 1]], dtype=np.int8))

    np.testing.assert_array_equal(batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.truncated, np.asarray([False], dtype=bool))
    assert int(batch.info["terminal_reason"][0]) == vector_reset.TERMINAL_REASON_SURVIVOR_WIN
    assert int(batch.info["winner"][0]) == source_winner_index
    assert bool(batch.info["draw"][0]) is False
    np.testing.assert_allclose(
        public_env.state["pos"][0],
        np.asarray(_source_positions(final_frame), dtype=np.float64),
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        public_env.state["alive"][0],
        np.asarray(_source_alive(final_frame), dtype=bool),
    )
    np.testing.assert_array_equal(
        public_env.state["score"][0],
        np.asarray(_source_scores(final_frame), dtype=np.int32),
    )

    source_dead_avatar_ids = [
        int(avatar["id"])
        for avatar in final_source_avatars
        if not bool(avatar["alive"])
    ]
    vector_dead_avatar_ids = [
        int(final_source_avatars[index]["id"])
        for index, alive in enumerate(public_env.state["alive"][0])
        if not bool(alive)
    ]
    assert vector_dead_avatar_ids == source_dead_avatar_ids == expected["deaths"]

    expected_public_reward = np.full((1, 2), -1.0, dtype=np.float32)
    expected_public_reward[0, source_winner_index] = 1.0
    np.testing.assert_array_equal(batch.reward, expected_public_reward)
    assert batch.final_reward is not None
    np.testing.assert_array_equal(batch.final_reward, expected_public_reward)


def test_public_vector_env_reset_to_terminal_matches_source_long_1v1_fixture():
    source_env, scenario, source_records = _run_source_long_1v1_no_bonus_wall_rollout()
    source_setup = scenario["source_setup"]
    lifecycle = scenario["lifecycle"]
    rollout = scenario["rollout"]
    assert isinstance(source_setup, dict)
    assert isinstance(lifecycle, dict)
    assert isinstance(rollout, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)
    random_values = random_setup["math_random_sequence"]
    assert isinstance(random_values, list)
    assert float(lifecycle["new_round_time_ms"]) == 0.0
    assert float(rollout["advance_timers_ms"]) == 0.0

    public_env = VectorTrainerEnv1v1NoBonus(
        autoreset=False,
        decision_ms=float(rollout["step_ms"]),
        player_ids=("p0", "p1"),
        random_tape_capacity=len(random_values),
    )
    reset_batch = public_env.reset(
        seed=np.asarray([0], dtype=np.uint64),
        source_fixture_random_tape_values=np.asarray([random_values], dtype=np.float64),
        source_fixture_new_round_time_ms=float(lifecycle["new_round_time_ms"]),
        source_fixture_warmup_advance_ms=float(rollout["advance_timers_ms"]),
    )

    policy = reset_batch.info["source_fixture_reset_policy"]
    assert policy["enabled"] is True
    assert policy["scope"] == "source_fixture_reset_parity_only"
    assert policy["random_tape_values_supplied"] is True
    assert policy["new_round_time_ms"] == 0.0
    np.testing.assert_array_equal(
        policy["random_tape_length"],
        np.asarray([len(random_values)], dtype=np.int32),
    )
    np.testing.assert_allclose(
        policy["advance_timers_ms"],
        np.asarray([0.0], dtype=np.float64),
    )
    assert reset_batch.info["warmup_info"]["game_start_fires"] == 1
    assert reset_batch.info["warmup_info"]["print_manager_delayed_start_fires"] == 0
    assert int(public_env.state["random_tape_cursor"][0]) == len(random_values)
    assert int(public_env.state["random_tape_draw_count"][0]) == len(random_values)
    assert int(public_env.state["random_tape_length"][0]) == len(random_values)
    np.testing.assert_array_equal(
        public_env.state["print_manager_active"][0],
        np.zeros(2, dtype=bool),
    )

    expected_source_moves = np.asarray([rollout["moves"]], dtype=np.int8)
    # Fixture moves are native held controls; public actions are wrapper decisions.
    straight_actions = np.asarray([[1, 1]], dtype=np.int8)
    terminal_batch = None
    for tick, record in enumerate(source_records):
        batch = public_env.step(straight_actions)
        np.testing.assert_array_equal(
            batch.info["source_moves"],
            expected_source_moves,
            err_msg=f"tick {tick} source_moves",
        )
        frame = record["frame"]
        assert isinstance(frame, dict)
        game = frame["game"]
        assert isinstance(game, dict)
        source_terminal = not bool(game["inRound"])
        np.testing.assert_array_equal(
            batch.done,
            np.asarray([source_terminal], dtype=bool),
            err_msg=f"tick {tick} done",
        )
        np.testing.assert_array_equal(
            batch.terminated,
            np.asarray([source_terminal], dtype=bool),
            err_msg=f"tick {tick} terminated",
        )
        _assert_public_row_matches_source_record(public_env, record, tick=tick)
        if source_terminal:
            terminal_batch = batch
            assert tick == int(rollout["terminal_tick"])
            break

    assert terminal_batch is not None
    expected = rollout["expected"]
    assert isinstance(expected, dict)
    source_winner_id = _source_round_winner(source_env.events)
    assert source_winner_id == expected["round_winner"]
    final_source_avatars = _source_avatars(source_records[-1]["frame"])
    source_winner_index = next(
        index
        for index, avatar in enumerate(final_source_avatars)
        if int(avatar["id"]) == source_winner_id
    )
    assert int(terminal_batch.info["terminal_reason"][0]) == (
        vector_reset.TERMINAL_REASON_SURVIVOR_WIN
    )
    assert int(terminal_batch.info["winner"][0]) == source_winner_index
    expected_public_reward = np.full((1, 2), -1.0, dtype=np.float32)
    expected_public_reward[0, source_winner_index] = 1.0
    np.testing.assert_array_equal(terminal_batch.reward, expected_public_reward)
    np.testing.assert_array_equal(terminal_batch.final_reward, expected_public_reward)


def test_public_vector_env_labels_fixed_decision_wrapper_for_long_1v1_fixture():
    scenario = _load_scenario(_LONG_NATURAL_ROLLOUT_SCENARIO)
    lifecycle = scenario["lifecycle"]
    rollout = scenario["rollout"]
    assert isinstance(lifecycle, dict)
    assert isinstance(rollout, dict)
    assert float(lifecycle["new_round_time_ms"]) == 0.0
    assert float(rollout["advance_timers_ms"]) == 0.0

    public_env = VectorTrainerEnv1v1NoBonus(
        autoreset=False,
        decision_ms=float(rollout["step_ms"]),
        player_ids=("p0", "p1"),
    )
    reset_batch = public_env.reset(seed=np.asarray([123], dtype=np.uint64))

    assert reset_batch.info["native_control_model_id"] == NATIVE_CONTROL_MODEL_ID
    assert reset_batch.info["trainer_control_wrapper_id"] == TRAINER_CONTROL_WRAPPER_ID
    assert reset_batch.info["decision_ms"] == pytest.approx(float(rollout["step_ms"]))
    assert reset_batch.info["reset_info"]["scheduled_timer_kind"] == "game:start"
    assert reset_batch.info["warmup_info"]["game_start_fires"] == 1
    assert reset_batch.info["warmup_info"]["print_manager_delayed_start_fires"] == 2
    np.testing.assert_allclose(
        reset_batch.info["warmup_info"]["advance_ms"],
        np.asarray(
            [public_env.first_warmup_ms + vector_runtime.SOURCE_TRAIL_START_DELAY_MS],
            dtype=np.float64,
        ),
    )

    step_batch = public_env.step(np.asarray([[1, 1]], dtype=np.int8))

    assert step_batch.info["native_control_model_id"] == NATIVE_CONTROL_MODEL_ID
    assert step_batch.info["trainer_control_wrapper_id"] == TRAINER_CONTROL_WRAPPER_ID
    assert step_batch.info["decision_ms"] == pytest.approx(float(rollout["step_ms"]))
    np.testing.assert_array_equal(
        step_batch.info["source_moves"],
        np.asarray([rollout["moves"]], dtype=np.int8),
    )


def test_timeout_truncation_autoreset_stages_final_transition_and_resets_row():
    env = VectorTrainerEnv1v1NoBonus(
        batch_size=1,
        seed=6,
        decision_ms=300.0,
        max_ticks=1,
        autoreset=True,
    )
    env.reset(seed=19)

    batch = env.step(np.asarray([[1, 1]], dtype=np.int8))

    np.testing.assert_array_equal(batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(batch.truncated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.reward, np.zeros((1, 2), dtype=np.float32))
    assert batch.final_observation is not None
    assert batch.final_reward is not None
    np.testing.assert_array_equal(batch.final_reward, np.zeros((1, 2), dtype=np.float32))
    assert bool(env.state["done"][0]) is False
    assert int(env.state["reset_source"][0]) == vector_reset.RESET_SOURCE_AUTORESET
    assert int(env.state["terminal_reason"][0]) == vector_reset.TERMINAL_REASON_NONE
    np.testing.assert_array_equal(batch.action_mask, np.ones((1, 2, 3), dtype=bool))

    plan_snapshot = batch.info["autoreset_plan"]["final_transition_snapshot"]
    np.testing.assert_array_equal(plan_snapshot["done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(plan_snapshot["terminated"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(plan_snapshot["truncated"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        plan_snapshot["final_reward_map"],
        np.zeros((1, 2), dtype=np.float32),
    )

    reset_snapshot = batch.info["autoreset_reset_info"]["terminal_transition_snapshot"]
    np.testing.assert_array_equal(
        reset_snapshot["arrays"]["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_TIMEOUT_TRUNCATED], dtype=np.int16),
    )


def test_event_overflow_truncates_rows_without_overriding_source_termination():
    env = VectorTrainerEnv1v1NoBonus(
        batch_size=2,
        seed=8,
        event_capacity=1,
        event_mode=vector_runtime.EVENT_MODE_DEBUG,
        autoreset=False,
        decision_ms=300.0,
    )
    env.reset(seed=np.asarray([31, 41], dtype=np.uint64))
    env.state["pos"][1, 1] = [env.map_size - 1.0, env.map_size / 2.0]
    env.state["prev_pos"][1, 1] = env.state["pos"][1, 1]
    env.state["heading"][1, 1] = 0.0
    env.state["speed"][1, 1] = 200.0

    batch = env.step(np.asarray([[1, 1], [1, 1]], dtype=np.int8))

    np.testing.assert_array_equal(batch.done, np.asarray([True, True], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([False, True], dtype=bool))
    np.testing.assert_array_equal(batch.truncated, np.asarray([True, False], dtype=bool))
    np.testing.assert_array_equal(
        batch.reward,
        np.asarray([[0.0, 0.0], [1.0, -1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        batch.info["terminal_reason"],
        np.asarray(
            [
                vector_reset.TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED,
                vector_reset.TERMINAL_REASON_SURVIVOR_WIN,
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        batch.info["death_cause"][1],
        np.asarray(
            [
                vector_runtime.DEATH_CAUSE_WALL,
                vector_runtime.DEATH_CAUSE_NONE,
            ],
            dtype=np.int16,
        ),
    )
    assert batch.info["death_cause_name"][1].tolist() == ["wall", "none"]
    np.testing.assert_array_equal(
        batch.info["death_hit_owner"][1],
        np.asarray([-1, -1], dtype=np.int16),
    )
    assert bool(env.state["event_overflow"].all())
    assert int(env.state["winner"][1]) == 0


def test_body_overflow_truncates_row_with_zero_reward():
    env = VectorTrainerEnv1v1NoBonus(
        batch_size=1,
        seed=9,
        body_capacity=2,
        autoreset=False,
    )
    env.reset(seed=43)
    assert bool(env.state["body_overflow"][0]) is False
    assert bool(env.state["overflow"][0]) is False

    batch = env.step(np.asarray([[1, 1]], dtype=np.int8))

    np.testing.assert_array_equal(batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(batch.truncated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.reward, np.zeros((1, 2), dtype=np.float32))
    np.testing.assert_array_equal(batch.final_reward, np.zeros((1, 2), dtype=np.float32))
    np.testing.assert_array_equal(
        batch.info["terminal_reason"],
        np.asarray(
            [vector_reset.TERMINAL_REASON_BODY_OVERFLOW_TRUNCATED],
            dtype=np.int16,
        ),
    )
    assert bool(env.state["body_overflow"][0]) is True
    assert bool(env.state["overflow"][0]) is True


def test_step_info_exposes_terminal_and_episode_metadata_by_row():
    env = VectorTrainerEnv1v1NoBonus(
        batch_size=2,
        seed=10,
        autoreset=False,
        decision_ms=300.0,
    )
    env.reset(seed=np.asarray([53, 59], dtype=np.uint64))
    env.state["pos"][0, 1] = [env.map_size - 1.0, env.map_size / 2.0]
    env.state["prev_pos"][0, 1] = env.state["pos"][0, 1]
    env.state["heading"][0, 1] = 0.0
    env.state["speed"][0, 1] = 200.0

    batch = env.step(np.asarray([[1, 1], [1, 1]], dtype=np.int8))

    np.testing.assert_array_equal(batch.info["terminal_rows"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(
        batch.info["terminal_reason"],
        np.asarray(
            [
                vector_reset.TERMINAL_REASON_SURVIVOR_WIN,
                vector_reset.TERMINAL_REASON_NONE,
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(batch.info["winner"], np.asarray([0, -1], dtype=np.int16))
    np.testing.assert_array_equal(batch.info["draw"], np.asarray([False, False], dtype=bool))
    np.testing.assert_array_equal(
        batch.info["terminated"],
        np.asarray([True, False], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["truncated"],
        np.asarray([False, False], dtype=bool),
    )
    np.testing.assert_array_equal(batch.info["episode_id"], np.asarray([1, 1], dtype=np.int64))
    np.testing.assert_array_equal(batch.info["reset_seed"], np.asarray([53, 59], dtype=np.uint64))
    np.testing.assert_array_equal(
        batch.info["reset_source"],
        np.asarray(
            [vector_reset.RESET_SOURCE_MANUAL, vector_reset.RESET_SOURCE_MANUAL],
            dtype=np.int16,
        ),
    )


def test_autoreset_info_distinguishes_returned_observation_from_final_observation():
    env = VectorTrainerEnv1v1NoBonus(
        batch_size=1,
        seed=11,
        autoreset=True,
        decision_ms=300.0,
    )
    env.reset(seed=np.asarray([61], dtype=np.uint64))
    env.state["pos"][0, 1] = [env.map_size - 1.0, env.map_size / 2.0]
    env.state["prev_pos"][0, 1] = env.state["pos"][0, 1]
    env.state["heading"][0, 1] = 0.0
    env.state["speed"][0, 1] = 200.0

    batch = env.step(np.asarray([[1, 1]], dtype=np.int8))

    assert batch.final_observation is not None
    assert batch.final_reward is not None
    assert not np.allclose(batch.observation[0], batch.final_observation[0])
    np.testing.assert_array_equal(batch.info["episode_id"], np.asarray([1], dtype=np.int64))
    np.testing.assert_array_equal(batch.info["reset_seed"], np.asarray([61], dtype=np.uint64))
    np.testing.assert_array_equal(
        batch.info["reset_source"],
        np.asarray([vector_reset.RESET_SOURCE_MANUAL], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        batch.info["returned_episode_id"],
        np.asarray([2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        batch.info["returned_reset_source"],
        np.asarray([vector_reset.RESET_SOURCE_AUTORESET], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        batch.info["returned_observation_source"],
        np.asarray(["post_autoreset"]),
    )
    np.testing.assert_array_equal(
        batch.info["final_observation_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.info["final_observation_policy"]["row_mask"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["final_reward_policy"]["rows"],
        np.asarray([0], dtype=np.int32),
    )
    assert batch.info["final_observation_policy"]["terminal_rows_only"] is True


def test_autoreset_false_rejects_step_after_terminal_until_reset():
    env = VectorTrainerEnv1v1NoBonus(
        batch_size=1,
        seed=4,
        decision_ms=300.0,
        autoreset=False,
    )
    env.reset(seed=13)
    env.state["pos"][0, 1] = [env.map_size - 1.0, env.map_size / 2.0]
    env.state["prev_pos"][0, 1] = env.state["pos"][0, 1]
    env.state["heading"][0, 1] = 0.0
    env.state["speed"][0, 1] = 200.0

    batch = env.step(np.asarray([[1, 1]], dtype=np.int8))

    np.testing.assert_array_equal(batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.reward, np.asarray([[1.0, -1.0]], dtype=np.float32))
    assert bool(env.state["done"][0]) is True
    with pytest.raises(RuntimeError, match="reset must be called"):
        env.step(np.asarray([[1, 1]], dtype=np.int8))

    reset_batch = env.reset(seed=14)
    np.testing.assert_array_equal(reset_batch.done, np.asarray([False], dtype=bool))
    env.step(np.asarray([[1, 1]], dtype=np.int8))


def test_timeout_truncation_autoreset_false_rejects_step_until_reset():
    env = VectorTrainerEnv1v1NoBonus(
        batch_size=1,
        seed=7,
        decision_ms=300.0,
        max_ticks=1,
        autoreset=False,
    )
    env.reset(seed=23)

    batch = env.step(np.asarray([[1, 1]], dtype=np.int8))

    np.testing.assert_array_equal(batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(batch.truncated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.reward, np.zeros((1, 2), dtype=np.float32))
    assert batch.final_observation is not None
    assert batch.final_reward is not None
    np.testing.assert_array_equal(batch.final_reward, np.zeros((1, 2), dtype=np.float32))
    assert bool(env.state["done"][0]) is True
    assert bool(env.state["reset_pending"][0]) is True
    assert int(env.state["terminal_reason"][0]) == (vector_reset.TERMINAL_REASON_TIMEOUT_TRUNCATED)

    with pytest.raises(RuntimeError, match="reset must be called"):
        env.step(np.asarray([[1, 1]], dtype=np.int8))

    reset_batch = env.reset(seed=24)
    np.testing.assert_array_equal(reset_batch.done, np.asarray([False], dtype=bool))
    env.step(np.asarray([[1, 1]], dtype=np.int8))


def test_partial_batch_reset_mutates_only_selected_done_row():
    env = VectorTrainerEnv1v1NoBonus(batch_size=2, seed=5, autoreset=False)
    env.reset(seed=17)
    row_zero_before = {name: array[0].copy() for name, array in env.state.items()}
    env.state["done"][1] = True
    env.state["terminated"][1] = True
    env.state["terminal_reason"][1] = vector_reset.TERMINAL_REASON_SURVIVOR_WIN
    env.state["winner"][1] = 0
    env.state["reset_pending"][1] = True
    row_one_pos_before = env.state["pos"][1].copy()

    batch = env.reset(
        seed=np.asarray([100, 200], dtype=np.uint64),
        row_mask=np.asarray([False, True], dtype=bool),
    )

    for name, before in row_zero_before.items():
        np.testing.assert_array_equal(env.state[name][0], before, err_msg=name)
    assert bool(env.state["done"][1]) is False
    assert int(env.state["reset_source"][1]) == vector_reset.RESET_SOURCE_MANUAL
    assert int(env.state["reset_seed"][1]) == 200
    assert not np.allclose(env.state["pos"][1], row_one_pos_before)
    np.testing.assert_array_equal(batch.done, np.zeros(2, dtype=bool))
