import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from curvyzero.env import vector_reset
from curvyzero.env import vector_runtime
from curvyzero.env import vector_source_random
from curvyzero.env.source_env import CurvyTronSourceEnv
from curvyzero.env.vector_multiplayer_env import DEBUG_METADATA_OBSERVATION_FIELDS
from curvyzero.env.vector_multiplayer_env import DEBUG_METADATA_OBSERVATION_SCHEMA_ID
from curvyzero.env.vector_multiplayer_env import ENV_IMPL_ID
from curvyzero.env.vector_multiplayer_env import ENV_VERSION_POLICY_ID
from curvyzero.env.vector_multiplayer_env import LIFECYCLE_POLICY_ID
from curvyzero.env.vector_multiplayer_env import PUBLIC_LIFECYCLE_ARRAY_NAMES
from curvyzero.env.vector_multiplayer_env import PUBLIC_ENV_CONTRACT_ID
from curvyzero.env.vector_multiplayer_env import PUBLIC_JOINT_ACTION_SIDECAR_SCHEMA_ID
from curvyzero.env.vector_multiplayer_env import RANDOM_TAPE_SOURCE_SOURCE_FIXTURE
from curvyzero.env.vector_multiplayer_env import RESET_EPISODE_ID_POLICY
from curvyzero.env.vector_multiplayer_env import RESET_PROVENANCE_POLICY_ID
from curvyzero.env.vector_multiplayer_env import NATURAL_BONUS_EFFECT_TYPE_NAMES
from curvyzero.env.vector_multiplayer_env import SEEDED_BONUS_TYPE_NAMES
from curvyzero.env.vector_multiplayer_env import SOURCE_ROUND_ID_POLICY
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerBatch
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnvError
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import compare_vector_arrays_to_fidelity as vector_compare  # noqa: E402
import seed_vector_state_from_fixtures as seed_bridge  # noqa: E402

SCENARIO_ROOT = Path(__file__).resolve().parents[1] / "scenarios" / "environment"
_LONG_NATURAL_ROLLOUT_SCENARIO = (
    SCENARIO_ROOT / "source_lifecycle_long_1v1_no_bonus_wall_round_done.json"
)


def _fixture_state_and_actions(
    scenario_path: str,
    *,
    body_capacity: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    fixture = seed_bridge.seed_fixture(scenario_path, body_capacity=body_capacity)
    state = vector_compare.array_state_from_seed(fixture)
    actions, step_ms = _fixture_actions_for_step(fixture, step_index=0)
    return state, actions, step_ms


def _fixture_state(
    scenario_path: str,
    *,
    body_capacity: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    fixture = seed_bridge.seed_fixture(scenario_path, body_capacity=body_capacity)
    return vector_compare.array_state_from_seed(fixture), fixture


def _fixture_actions_for_step(
    fixture: dict[str, object],
    *,
    step_index: int,
) -> tuple[np.ndarray, float]:
    prepared_step = vector_compare.prepare_fixture_array_step(
        fixture,
        step_index=step_index,
    )
    source_moves = np.asarray(prepared_step["source_moves"], dtype=np.int8)
    actions = source_moves.astype(np.int16) + 1
    return actions.reshape(1, -1), float(prepared_step["step_ms"])


def _fixture_timer_advance_ms(
    fixture: dict[str, object],
    *,
    step_index: int,
) -> float:
    prepared_step = vector_compare.prepare_fixture_array_step(
        fixture,
        step_index=step_index,
    )
    return float(prepared_step.get("timer_advance_ms", 0.0))


def _first_active_bonus(scenario_name: str) -> dict[str, object]:
    payload = _lifecycle_payload(scenario_name)
    initial_state = payload["initial_state"]
    assert isinstance(initial_state, dict)
    active_bonuses = initial_state["active_bonuses"]
    assert isinstance(active_bonuses, list)
    assert len(active_bonuses) == 1
    bonus = active_bonuses[0]
    assert isinstance(bonus, dict)
    return bonus


def _lifecycle_random_tape(
    scenario_name: str,
    *,
    extra_values: int = 0,
) -> np.ndarray:
    payload = _lifecycle_payload(scenario_name)
    sequence = payload["source_setup"]["random"]["math_random_sequence"]
    if extra_values:
        sequence = [*sequence, *([0.5] * extra_values)]
    return np.asarray([sequence], dtype=np.float64)


def _bonus_spawn_random_values(scenario_name: str) -> list[float]:
    payload = _lifecycle_payload(scenario_name)
    sequence = payload["source_setup"]["random"]["math_random_sequence"]
    return [float(entry["value"]) for entry in sequence]


def _natural_bonus_spawn_tape(*, type_draw: float = 0.2) -> np.ndarray:
    spawn_prefix = _lifecycle_random_tape(
        "source_lifecycle_spawn_rng_2p_next_round.json",
    )[0, :6]
    bonus_values = (0.0, 0.5, type_draw, 0.25, 0.75)
    print_manager_start_values = (0.5, 0.5)
    return np.asarray(
        [[*spawn_prefix, *bonus_values, *print_manager_start_values]],
        dtype=np.float64,
    )


def _lifecycle_max_score(scenario_name: str) -> int:
    payload = _lifecycle_payload(scenario_name)
    source_setup = payload["source_setup"]
    assert isinstance(source_setup, dict)
    room = source_setup["room"]
    assert isinstance(room, dict)
    return int(room["max_score"])


def _lifecycle_payload(scenario_name: str) -> dict[str, object]:
    with (SCENARIO_ROOT / scenario_name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _source_round_winner(events: list[dict[str, object]]) -> int | None:
    round_end_events = [event for event in events if event["event"] == "round:end"]
    assert len(round_end_events) == 1
    data = round_end_events[0]["data"]
    assert isinstance(data, dict)
    winner = data["winner"]
    assert winner is None or isinstance(winner, int)
    return winner


def _assert_multiplayer_row_matches_source_record(
    env: VectorMultiplayerEnv,
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


def _reset_3p_public_lifecycle_env(
    scenario_name: str,
    *,
    episode_end_mode: str = "round",
) -> tuple[VectorMultiplayerEnv, VectorMultiplayerBatch]:
    tape = _lifecycle_random_tape(scenario_name)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=3,
        seed=555,
        decision_ms=100.0,
        max_score=_lifecycle_max_score(scenario_name),
        body_capacity=24,
        event_capacity=64,
        timer_capacity=3,
        random_tape_capacity=tape.shape[1],
        episode_end_mode=episode_end_mode,
    )
    batch = env.reset(
        seed=np.asarray([555], dtype=np.uint64),
        source_fixture_random_tape_values=tape,
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )
    return env, batch


def _reset_2p_public_lifecycle_env(
    scenario_name: str,
    *,
    episode_end_mode: str = "round",
) -> tuple[VectorMultiplayerEnv, VectorMultiplayerBatch]:
    tape = _lifecycle_random_tape(scenario_name)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        seed=555,
        decision_ms=100.0,
        max_score=_lifecycle_max_score(scenario_name),
        body_capacity=24,
        event_capacity=64,
        timer_capacity=3,
        random_tape_capacity=tape.shape[1],
        episode_end_mode=episode_end_mode,
    )
    batch = env.reset(
        seed=np.asarray([555], dtype=np.uint64),
        source_fixture_random_tape_values=tape,
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )
    return env, batch


def _reset_4p_public_lifecycle_env(
    scenario_name: str,
    *,
    episode_end_mode: str = "round",
    extra_random_tape_values: int = 0,
    source_fixture_ref: str | None = None,
) -> tuple[VectorMultiplayerEnv, VectorMultiplayerBatch]:
    tape = _lifecycle_random_tape(
        scenario_name,
        extra_values=extra_random_tape_values,
    )
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        seed=555,
        decision_ms=100.0,
        max_score=_lifecycle_max_score(scenario_name),
        body_capacity=24,
        event_capacity=64,
        timer_capacity=4,
        random_tape_capacity=tape.shape[1],
        episode_end_mode=episode_end_mode,
    )
    batch = env.reset(
        seed=np.asarray([555], dtype=np.uint64),
        source_fixture_random_tape_values=tape,
        source_fixture_ref=source_fixture_ref,
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )
    return env, batch


def _force_3p_player0_round_win(env: VectorMultiplayerEnv) -> None:
    env.state["pos"][0] = np.asarray(
        [[5.0, 5.0], [1.0, 47.5], [93.0, 47.5]],
        dtype=np.float64,
    )
    env.state["heading"][0] = np.asarray(
        [math.pi / 4.0, math.pi, 0.0],
        dtype=np.float64,
    )
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["speed"][0, 0] = 8.0
    env.state["print_manager_distance"][0, 0] = 999.0
    env.state["print_manager_last_pos"][0, 0] = env.state["pos"][0, 0]


def _force_4p_player0_round_win(env: VectorMultiplayerEnv) -> None:
    env.state["pos"][0] = np.asarray(
        [[10.0, 50.5], [50.5, 1.0], [1.0, 50.5], [99.0, 50.5]],
        dtype=np.float64,
    )
    env.state["heading"][0] = np.asarray(
        [0.0, math.tau * 0.75, math.pi, 0.0],
        dtype=np.float64,
    )
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["speed"][0] = np.asarray([8.0, 16.0, 16.0, 16.0], dtype=np.float64)
    env.state["print_manager_distance"][0, 0] = 999.0
    env.state["print_manager_last_pos"][0, 0] = env.state["pos"][0, 0]


def _force_2p_wall_draw(env: VectorMultiplayerEnv) -> None:
    env.state["pos"][0] = np.asarray(
        [[1.0, 44.0], [87.0, 44.0]],
        dtype=np.float64,
    )
    env.state["heading"][0] = np.asarray([math.pi, 0.0], dtype=np.float64)
    env.state["prev_pos"][0] = env.state["pos"][0]


def _force_2p_player0_round_win(env: VectorMultiplayerEnv) -> None:
    env.state["pos"][0] = np.asarray(
        [[5.0, 5.0], [87.0, 44.0]],
        dtype=np.float64,
    )
    env.state["heading"][0] = np.asarray([math.pi / 4.0, 0.0], dtype=np.float64)
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["speed"][0, 0] = 8.0
    env.state["print_manager_distance"][0, 0] = 999.0
    env.state["print_manager_last_pos"][0, 0] = env.state["pos"][0, 0]


def _force_safe_stationary_3p_positions(env: VectorMultiplayerEnv) -> None:
    env.state["pos"][0] = np.asarray(
        [[25.0, 25.0], [47.5, 47.5], [70.0, 70.0]],
        dtype=np.float64,
    )
    env.state["heading"][0] = np.asarray([0.0, math.pi / 2.0, math.pi], dtype=np.float64)
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["speed"][0] = np.zeros(3, dtype=np.float64)
    env.state["print_manager_distance"][0] = np.full(3, 999.0, dtype=np.float64)
    env.state["print_manager_last_pos"][0] = env.state["pos"][0]


def _assert_public_final_metadata(
    batch: VectorMultiplayerBatch,
    rows: list[int],
    *,
    expected_reward: np.ndarray | None,
) -> None:
    rows_array = np.asarray(rows, dtype=np.int32)
    row_mask = np.zeros(batch.done.shape, dtype=bool)
    row_mask[rows_array] = True
    np.testing.assert_array_equal(batch.info["final_observation_rows"], rows_array)
    np.testing.assert_array_equal(batch.info["final_observation_row_mask"], row_mask)
    np.testing.assert_array_equal(batch.info["final_reward_rows"], rows_array)
    np.testing.assert_array_equal(batch.info["final_reward_row_mask"], row_mask)

    observation_policy = batch.info["final_observation_row_policy"]
    assert observation_policy["present"] is bool(rows)
    assert observation_policy["terminal_rows_only"] is True
    assert observation_policy["nonterminal_rows_zero_filled"] is bool(rows)
    assert observation_policy["observation_schema_id"] == DEBUG_METADATA_OBSERVATION_SCHEMA_ID
    assert observation_policy["source_claim"] == "debug_metadata_only_public_terminal_rows/v0"
    np.testing.assert_array_equal(observation_policy["rows"], rows_array)
    np.testing.assert_array_equal(observation_policy["row_mask"], row_mask)

    reward_policy = batch.info["final_reward_row_policy"]
    assert reward_policy["present"] is (expected_reward is not None)
    assert reward_policy["terminal_rows_only"] is True
    assert reward_policy["source_claim"] == "debug_metadata_only_public_terminal_rows/v0"
    np.testing.assert_array_equal(reward_policy["rows"], rows_array)
    np.testing.assert_array_equal(reward_policy["row_mask"], row_mask)

    if rows:
        assert batch.final_observation is not None
        expected_final_observation = np.zeros_like(batch.observation)
        expected_final_observation[rows_array] = batch.observation[rows_array]
        np.testing.assert_array_equal(batch.final_observation, batch.info["final_observation"])
        np.testing.assert_array_equal(batch.final_observation, expected_final_observation)
        assert expected_reward is not None
        assert batch.final_reward is not None
        expected_final_reward = np.zeros_like(batch.reward)
        expected_final_reward[rows_array] = expected_reward[rows_array]
        np.testing.assert_array_equal(batch.final_reward, expected_final_reward)
        np.testing.assert_array_equal(batch.info["final_reward_map"], expected_final_reward)
    else:
        assert batch.final_observation is None
        assert batch.final_reward is None
        assert expected_reward is None


def _step_public_fixture_once(
    scenario_path: str,
    *,
    body_capacity: int,
) -> tuple[VectorMultiplayerEnv, VectorMultiplayerBatch]:
    state, actions, step_ms = _fixture_state_and_actions(
        scenario_path,
        body_capacity=body_capacity,
    )
    player_count = int(state["pos"].shape[1])
    decision_ms = step_ms if step_ms > 0.0 else 1.0
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=player_count,
        decision_ms=decision_ms,
        body_capacity=body_capacity,
        event_capacity=16,
        timer_capacity=max(4, player_count),
        random_tape_capacity=8,
        event_mode="debug-event",
    )
    env.reset_from_state_arrays(
        state,
        reset_seed=np.asarray([101], dtype=np.uint64),
    )
    if step_ms == 0.0:
        env.decision_ms = 0.0
    return env, env.step(actions)


def _step_public_seeded_bonus_fixture_once(
    scenario_name: str,
    *,
    body_capacity: int = 8,
    bonus_type: str | None = None,
) -> tuple[VectorMultiplayerEnv, VectorMultiplayerBatch]:
    scenario_path = f"scenarios/environment/{scenario_name}"
    state, actions, step_ms = _fixture_state_and_actions(
        scenario_path,
        body_capacity=body_capacity,
    )
    bonus = _first_active_bonus(scenario_name)
    player_count = int(state["pos"].shape[1])
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=player_count,
        decision_ms=step_ms,
        body_capacity=body_capacity,
        event_capacity=16,
        timer_capacity=max(4, player_count),
        random_tape_capacity=8,
        event_mode="debug-event",
    )
    env.reset_from_state_arrays(
        state,
        reset_seed=np.asarray([101], dtype=np.uint64),
    )
    env.seed_active_bonus(
        row=0,
        bonus_type=bonus_type or str(bonus["type"]),
        x=float(bonus["x"]),
        y=float(bonus["y"]),
    )
    return env, env.step(actions)


def _assert_seeded_bonus_public_claim(batch: VectorMultiplayerBatch) -> None:
    assert batch.info["public_env_contract_id"] == PUBLIC_ENV_CONTRACT_ID
    assert batch.info["env_impl_id"] == ENV_IMPL_ID
    assert batch.info["env_version_policy_id"] == ENV_VERSION_POLICY_ID
    assert batch.info["ruleset_id"] == "curvytron_seeded_bonus_subset/v0"
    assert batch.info["bonus_support_mode"] == "seeded"
    bonus_support = batch.info["bonus_support"]
    assert bonus_support["mode"] == "seeded"
    assert bonus_support["natural_bonus_spawn"] is False
    assert bonus_support["supported_seeded_bonus_types"] == SEEDED_BONUS_TYPE_NAMES
    assert "duration_ms" not in bonus_support
    assert "expiry_ms" not in bonus_support
    assert "expires" not in bonus_support
    np.testing.assert_array_equal(
        bonus_support["enabled_by_row"],
        np.asarray([True], dtype=bool),
    )


def _direct_public_seeded_bonus_env(
    *,
    player_count: int,
    positions: list[list[float]],
    bonus_type: str,
) -> VectorMultiplayerEnv:
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=player_count,
        decision_ms=1.0,
        body_capacity=8,
        event_capacity=16,
        timer_capacity=max(4, player_count),
        random_tape_capacity=32,
        event_mode="debug-event",
    )
    env.reset(
        seed=np.asarray([101], dtype=np.uint64),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    env.decision_ms = 0.0
    env.state["timer_active"][0] = False
    env.state["pos"][0] = np.asarray(positions, dtype=np.float64)
    env.state["alive"][0] = True
    env.state["present"][0] = True
    env.state["printing"][0] = True
    env.state["print_manager_active"][0] = True
    env.state["print_manager_distance"][0] = np.full(
        player_count,
        999.0,
        dtype=np.float64,
    )
    env.state["print_manager_last_pos"][0] = env.state["pos"][0]
    env.seed_active_bonus(
        row=0,
        bonus_type=bonus_type,
        x=float(positions[0][0]),
        y=float(positions[0][1]),
    )
    return env


def _direct_public_natural_bonus_env(
    *,
    bonus_type: str,
    player_count: int = 2,
) -> VectorMultiplayerEnv:
    tape = _natural_bonus_spawn_tape(type_draw=0.0)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=player_count,
        decision_ms=1.0,
        body_capacity=8,
        event_capacity=16,
        timer_capacity=max(4, player_count),
        random_tape_capacity=tape.shape[1],
        event_mode="debug-event",
        natural_bonus_spawn=True,
        natural_bonus_type_codes=(bonus_type,),
    )
    env.reset(
        seed=np.asarray([101], dtype=np.uint64),
        source_fixture_random_tape_values=tape,
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    env.decision_ms = 0.0
    env.state["timer_active"][0] = False
    env.state["pos"][0, 0] = np.asarray([23.94, 64.06], dtype=np.float64)
    if player_count > 1:
        env.state["pos"][0, 1:] = np.asarray(
            [[70.0 + index, 70.0] for index in range(player_count - 1)],
            dtype=np.float64,
        )
    env.state["alive"][0] = True
    env.state["present"][0] = True
    env.state["printing"][0] = True
    env.state["print_manager_active"][0] = True
    env.state["print_manager_distance"][0] = np.full(
        player_count,
        999.0,
        dtype=np.float64,
    )
    env.state["print_manager_last_pos"][0] = env.state["pos"][0]
    return env


def test_public_seeded_bonus_support_is_off_by_default():
    env, batch = _step_public_fixture_once(
        "scenarios/environment/source_bonus_self_small_catch_step.json",
        body_capacity=8,
    )

    assert batch.info["public_env_contract_id"].endswith("env/v0")
    assert batch.info["ruleset_id"] == "curvytron_no_bonus/v0"
    assert batch.info["bonus_support_mode"] == "disabled"
    assert batch.info["bonus_support"]["natural_bonus_spawn"] is False
    assert batch.info["bonus_support"]["public_env_contract_id"] == PUBLIC_ENV_CONTRACT_ID
    assert "BonusGameBorderless" in batch.info["bonus_support"][
        "supported_seeded_bonus_types"
    ]
    assert batch.info["step_counters"]["bonus_self_small_catches"] == 0
    assert batch.info["step_counters"]["bonus_stack_appends"] == 0
    np.testing.assert_array_equal(batch.info["borderless"], np.asarray([False]))
    assert "bonus_active" not in env.state
    np.testing.assert_allclose(env.state["radius"], np.asarray([[0.6, 0.6]]))


def test_public_seeded_bonus_self_small_catch_matches_source_fixture():
    env, batch = _step_public_seeded_bonus_fixture_once(
        "source_bonus_self_small_catch_step.json",
    )

    _assert_seeded_bonus_public_claim(batch)
    assert batch.info["step_counters"]["bonus_self_small_catches"] == 1
    assert batch.info["step_counters"]["bonus_stack_appends"] == 1
    np.testing.assert_allclose(env.state["radius"], np.asarray([[0.3, 0.6]]))
    np.testing.assert_array_equal(env.state["radius_power"], np.asarray([[-1, 0]]))
    np.testing.assert_array_equal(env.state["bonus_active"], np.asarray([[False]]))
    np.testing.assert_array_equal(env.state["bonus_count"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[1, 0]]))
    assert int(env.state["bonus_stack_type"][0, 0, 0]) == (
        vector_runtime.BONUS_TYPE_SELF_SMALL
    )
    assert int(env.state["bonus_stack_duration_ms"][0, 0, 0]) == 7500


def test_public_seeded_bonus_self_small_tangent_does_not_catch():
    env, batch = _step_public_seeded_bonus_fixture_once(
        "source_bonus_self_small_tangent_no_catch_step.json",
    )

    _assert_seeded_bonus_public_claim(batch)
    assert batch.info["step_counters"]["bonus_self_small_catches"] == 0
    assert batch.info["step_counters"]["bonus_stack_appends"] == 0
    np.testing.assert_allclose(env.state["radius"], np.asarray([[0.6, 0.6]]))
    np.testing.assert_array_equal(env.state["radius_power"], np.asarray([[0, 0]]))
    np.testing.assert_array_equal(env.state["bonus_active"], np.asarray([[True]]))
    np.testing.assert_array_equal(env.state["bonus_count"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[0, 0]]))


def test_public_seeded_bonus_self_small_expiry_restores_radius():
    scenario_name = "source_bonus_self_small_expiry_restore_step.json"
    scenario_path = f"scenarios/environment/{scenario_name}"
    state, fixture = _fixture_state(scenario_path, body_capacity=8)
    actions, step_ms = _fixture_actions_for_step(fixture, step_index=0)
    bonus = _first_active_bonus(scenario_name)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        decision_ms=step_ms,
        body_capacity=8,
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=8,
        event_mode="debug-event",
    )
    env.reset_from_state_arrays(
        state,
        reset_seed=np.asarray([101], dtype=np.uint64),
    )
    env.seed_active_bonus(
        row=0,
        bonus_type=str(bonus["type"]),
        x=float(bonus["x"]),
        y=float(bonus["y"]),
    )

    catch_batch = env.step(actions)
    second_actions, second_step_ms = _fixture_actions_for_step(fixture, step_index=1)
    env.decision_ms = second_step_ms
    expiry_batch = env.step(
        second_actions,
        timer_advance_ms=_fixture_timer_advance_ms(fixture, step_index=1),
    )

    _assert_seeded_bonus_public_claim(catch_batch)
    _assert_seeded_bonus_public_claim(expiry_batch)
    assert catch_batch.info["step_counters"]["bonus_self_small_catches"] == 1
    assert expiry_batch.info["step_counters"]["bonus_self_small_expiries"] == 1
    assert expiry_batch.info["step_counters"]["bonus_stack_appends"] == 0
    np.testing.assert_allclose(env.state["radius"], np.asarray([[0.6, 0.6]]))
    np.testing.assert_array_equal(env.state["radius_power"], np.asarray([[0, 0]]))
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[0, 0]]))
    assert int(env.state["bonus_stack_type"][0, 0, 0]) == vector_runtime.BONUS_TYPE_NONE


def test_public_seeded_bonus_game_clear_immediate_clear_matches_source_fixture():
    env, batch = _step_public_seeded_bonus_fixture_once(
        "source_bonus_game_clear_immediate_step.json",
        body_capacity=4,
    )

    _assert_seeded_bonus_public_claim(batch)
    assert batch.info["step_counters"]["bonus_game_clear_catches"] == 1
    assert batch.info["step_counters"]["bonus_stack_appends"] == 0
    np.testing.assert_array_equal(env.state["world_body_count"], np.asarray([0]))
    np.testing.assert_array_equal(env.state["body_active"], np.zeros((1, 4), dtype=bool))
    np.testing.assert_array_equal(env.state["bonus_active"], np.asarray([[False]]))
    np.testing.assert_array_equal(env.state["bonus_count"], np.asarray([0], dtype=np.int32))


def test_public_seeded_bonus_game_borderless_js_oracle_canary_flips_row_metadata():
    scenario_name = "source_bonus_game_borderless_catch_step.json"
    scenario_path = f"scenarios/environment/{scenario_name}"
    state, actions, step_ms = _fixture_state_and_actions(
        scenario_path,
        body_capacity=4,
    )
    bonus = _first_active_bonus(scenario_name)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        decision_ms=step_ms,
        body_capacity=4,
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=8,
        event_mode="debug-event",
    )
    env.reset_from_state_arrays(
        state,
        reset_seed=np.asarray([101], dtype=np.uint64),
    )

    seed_info = env.seed_active_bonus(
        row=0,
        bonus_type="BonusGameBorderless",
        x=float(bonus["x"]),
        y=float(bonus["y"]),
    )
    batch = env.step(actions)

    _assert_seeded_bonus_public_claim(batch)
    assert seed_info["bonus_type"] == "BonusGameBorderless"
    assert seed_info["natural_bonus_spawn"] is False
    assert batch.info["step_counters"]["bonus_self_small_catches"] == 0
    assert batch.info["step_counters"]["bonus_game_clear_catches"] == 0
    assert batch.info["step_counters"]["bonus_game_borderless_catches"] == 1
    assert batch.info["step_counters"]["bonus_stack_appends"] == 1
    np.testing.assert_array_equal(env.state["borderless"], np.asarray([True]))
    np.testing.assert_array_equal(batch.info["borderless"], np.asarray([True]))
    np.testing.assert_array_equal(env.state["bonus_active"], np.asarray([[False]]))
    np.testing.assert_array_equal(env.state["bonus_count"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[0, 0]]))


def test_public_seeded_bonus_game_borderless_expiry_restores_borderless_false():
    scenario_name = "source_bonus_game_borderless_expiry_restore_step.json"
    scenario_path = f"scenarios/environment/{scenario_name}"
    state, fixture = _fixture_state(scenario_path, body_capacity=4)
    actions, step_ms = _fixture_actions_for_step(fixture, step_index=0)
    bonus = _first_active_bonus(scenario_name)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        decision_ms=step_ms,
        body_capacity=4,
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=8,
        event_mode="debug-event",
    )
    env.reset_from_state_arrays(
        state,
        reset_seed=np.asarray([101], dtype=np.uint64),
    )
    env.seed_active_bonus(
        row=0,
        bonus_type=str(bonus["type"]),
        x=float(bonus["x"]),
        y=float(bonus["y"]),
    )

    catch_batch = env.step(actions)
    second_actions, second_step_ms = _fixture_actions_for_step(fixture, step_index=1)
    env.decision_ms = second_step_ms
    expiry_batch = env.step(
        second_actions,
        timer_advance_ms=_fixture_timer_advance_ms(fixture, step_index=1),
    )

    _assert_seeded_bonus_public_claim(catch_batch)
    _assert_seeded_bonus_public_claim(expiry_batch)
    assert catch_batch.info["step_counters"]["bonus_game_borderless_catches"] == 1
    assert catch_batch.info["step_counters"]["bonus_game_borderless_expiries"] == 0
    assert catch_batch.info["step_counters"]["bonus_stack_appends"] == 1
    np.testing.assert_array_equal(catch_batch.info["borderless"], np.asarray([True]))
    assert expiry_batch.info["action_sidecar"]["timer_advance_ms"][0] == pytest.approx(
        10_000.0
    )
    assert expiry_batch.info["step_counters"]["bonus_game_borderless_expiries"] == 1
    assert expiry_batch.info["step_counters"]["bonus_game_borderless_catches"] == 0
    assert expiry_batch.info["step_counters"]["bonus_stack_appends"] == 0
    np.testing.assert_array_equal(env.state["borderless"], np.asarray([False]))
    np.testing.assert_array_equal(expiry_batch.info["borderless"], np.asarray([False]))
    np.testing.assert_array_equal(env.state["bonus_game_stack_count"], np.asarray([0]))
    assert int(env.state["bonus_game_stack_type"][0, 0]) == vector_runtime.BONUS_TYPE_NONE
    assert int(env.state["bonus_game_stack_duration_ms"][0, 0]) == 0
    assert int(env.state["bonus_game_stack_borderless"][0, 0]) == 0


def test_public_seeded_bonus_enemy_straight_angle_expiry_restores_turn_rate():
    scenario_name = "source_bonus_self_small_expiry_restore_step.json"
    scenario_path = f"scenarios/environment/{scenario_name}"
    state, fixture = _fixture_state(scenario_path, body_capacity=8)
    actions, step_ms = _fixture_actions_for_step(fixture, step_index=0)
    bonus = _first_active_bonus(scenario_name)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        decision_ms=step_ms,
        body_capacity=8,
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=8,
        event_mode="debug-event",
    )
    env.reset_from_state_arrays(
        state,
        reset_seed=np.asarray([101], dtype=np.uint64),
    )
    base_angular_velocity = env.state["angular_velocity_per_ms"].copy()
    env.seed_active_bonus(
        row=0,
        bonus_type="BonusEnemyStraightAngle",
        x=float(bonus["x"]),
        y=float(bonus["y"]),
    )

    catch_batch = env.step(actions)

    _assert_seeded_bonus_public_claim(catch_batch)
    assert "BonusEnemyStraightAngle" in (
        catch_batch.info["bonus_support"]["supported_seeded_bonus_types"]
    )
    assert catch_batch.info["step_counters"]["bonus_enemy_straight_angle_catches"] == 1
    assert catch_batch.info["step_counters"]["bonus_stack_appends"] == 1
    np.testing.assert_allclose(
        env.state["angular_velocity_per_ms"],
        np.asarray(
            [
                [
                    base_angular_velocity[0, 0],
                    vector_runtime.SOURCE_STRAIGHT_ANGLE_RADIANS,
                ]
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[0, 1]]))
    assert int(env.state["bonus_stack_type"][0, 1, 0]) == (
        vector_runtime.BONUS_TYPE_ENEMY_STRAIGHT_ANGLE
    )
    assert env.state["bonus_stack_angular_velocity_per_ms"][0, 1, 0] == pytest.approx(
        vector_runtime.SOURCE_STRAIGHT_ANGLE_RADIANS
    )

    second_actions, second_step_ms = _fixture_actions_for_step(fixture, step_index=1)
    env.decision_ms = second_step_ms
    expiry_batch = env.step(
        second_actions,
        timer_advance_ms=vector_runtime.BONUS_ENEMY_STRAIGHT_ANGLE_DURATION_MS,
    )

    _assert_seeded_bonus_public_claim(expiry_batch)
    assert expiry_batch.info["step_counters"]["bonus_enemy_straight_angle_expiries"] == 1
    assert expiry_batch.info["step_counters"]["bonus_stack_appends"] == 0
    np.testing.assert_allclose(env.state["angular_velocity_per_ms"], base_angular_velocity)
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[0, 0]]))
    assert int(env.state["bonus_stack_type"][0, 1, 0]) == vector_runtime.BONUS_TYPE_NONE
    assert env.state["bonus_stack_angular_velocity_per_ms"][0, 1, 0] == pytest.approx(
        0.0
    )


def test_public_seeded_bonus_self_master_invincible_printing_and_expiry():
    env = _direct_public_seeded_bonus_env(
        player_count=2,
        positions=[[20.0, 20.0], [70.0, 70.0]],
        bonus_type="BonusSelfMaster",
    )

    catch_batch = env.step(np.asarray([[1, 1]], dtype=np.int16))

    _assert_seeded_bonus_public_claim(catch_batch)
    assert catch_batch.info["step_counters"]["bonus_self_master_catches"] == 1
    assert catch_batch.info["step_counters"]["bonus_stack_appends"] == 1
    np.testing.assert_array_equal(env.state["bonus_active"], np.asarray([[False]]))
    np.testing.assert_array_equal(env.state["bonus_count"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(env.state["invincible"], np.asarray([[True, False]]))
    np.testing.assert_array_equal(env.state["printing"], np.asarray([[False, True]]))
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[1, 0]]))
    assert int(env.state["bonus_stack_type"][0, 0, 0]) == (
        vector_runtime.BONUS_TYPE_SELF_MASTER
    )
    assert int(env.state["bonus_stack_duration_ms"][0, 0, 0]) == (
        vector_runtime.BONUS_SELF_MASTER_DURATION_MS
    )
    assert int(env.state["bonus_stack_invincible_delta"][0, 0, 0]) == 1
    assert int(env.state["bonus_stack_printing_delta"][0, 0, 0]) == -1

    expiry_batch = env.step(
        np.asarray([[1, 1]], dtype=np.int16),
        timer_advance_ms=vector_runtime.BONUS_SELF_MASTER_DURATION_MS,
    )

    _assert_seeded_bonus_public_claim(expiry_batch)
    assert expiry_batch.info["step_counters"]["bonus_self_master_expiries"] == 1
    assert expiry_batch.info["step_counters"]["bonus_stack_appends"] == 0
    np.testing.assert_array_equal(env.state["invincible"], np.asarray([[False, False]]))
    np.testing.assert_array_equal(env.state["printing"], np.asarray([[True, True]]))
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[0, 0]]))
    assert int(env.state["bonus_stack_type"][0, 0, 0]) == vector_runtime.BONUS_TYPE_NONE
    assert int(env.state["bonus_stack_invincible_delta"][0, 0, 0]) == 0
    assert int(env.state["bonus_stack_printing_delta"][0, 0, 0]) == 0


def test_public_seeded_bonus_self_master_blocks_body_death_but_not_wall_death():
    env = _direct_public_seeded_bonus_env(
        player_count=2,
        positions=[[20.0, 20.0], [70.0, 70.0]],
        bonus_type="BonusSelfMaster",
    )

    catch_batch = env.step(np.asarray([[1, 1]], dtype=np.int16))

    _assert_seeded_bonus_public_claim(catch_batch)
    np.testing.assert_array_equal(env.state["invincible"], np.asarray([[True, False]]))

    env.state["body_active"][0] = False
    env.state["body_pos"][0] = 0.0
    env.state["body_radius"][0] = 0.0
    env.state["body_owner"][0] = -1
    env.state["body_num"][0] = -1
    env.state["body_insert_tick"][0] = -1
    env.state["body_insert_kind"][0] = -1
    env.state["body_active"][0, 0] = True
    env.state["body_pos"][0, 0] = env.state["pos"][0, 0]
    env.state["body_radius"][0, 0] = 1.0
    env.state["body_owner"][0, 0] = 1
    env.state["body_num"][0, 0] = 0
    env.state["body_insert_tick"][0, 0] = int(env.state["tick"][0])
    env.state["body_insert_kind"][0, 0] = vector_runtime.BODY_KIND_NORMAL
    env.state["body_write_cursor"][0] = 1

    body_batch = env.step(np.asarray([[1, 1]], dtype=np.int16))

    _assert_seeded_bonus_public_claim(body_batch)
    assert body_batch.info["step_counters"]["body_hits"] == 1
    np.testing.assert_array_equal(env.state["alive"], np.asarray([[True, True]]))
    np.testing.assert_array_equal(
        body_batch.info["death_count"],
        np.asarray([0], dtype=np.int32),
    )

    env.state["pos"][0, 0] = np.asarray([0.3, 20.0], dtype=np.float64)
    env.state["prev_pos"][0, 0] = env.state["pos"][0, 0]
    env.state["heading"][0, 0] = math.pi

    wall_batch = env.step(np.asarray([[1, 1]], dtype=np.int16))

    _assert_seeded_bonus_public_claim(wall_batch)
    assert wall_batch.info["step_counters"]["normal_wall_deaths"] == 1
    np.testing.assert_array_equal(env.state["alive"], np.asarray([[False, True]]))
    np.testing.assert_array_equal(
        wall_batch.info["death_count"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        wall_batch.info["death_player"],
        np.asarray([[0, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        wall_batch.info["death_cause"],
        np.asarray(
            [[vector_runtime.DEATH_CAUSE_WALL, vector_runtime.DEATH_CAUSE_NONE]],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        wall_batch.info["death_hit_owner"],
        np.asarray([[-1, -1]], dtype=np.int16),
    )


def test_public_seeded_bonus_all_color_rotates_alive_colors_and_expires():
    env = _direct_public_seeded_bonus_env(
        player_count=3,
        positions=[[20.0, 20.0], [50.0, 20.0], [80.0, 20.0]],
        bonus_type="BonusAllColor",
    )
    np.testing.assert_array_equal(env.state["avatar_color"], np.asarray([[0, 1, 2]]))
    np.testing.assert_array_equal(
        env.state["base_avatar_color"],
        np.asarray([[0, 1, 2]]),
    )

    catch_batch = env.step(np.asarray([[1, 1, 1]], dtype=np.int16))

    _assert_seeded_bonus_public_claim(catch_batch)
    assert catch_batch.info["step_counters"]["bonus_all_color_catches"] == 1
    assert catch_batch.info["step_counters"]["bonus_stack_appends"] == 3
    event_count = int(env.state["event_count"][0])
    event_types = env.state["event_type"][0, :event_count]
    clear_index = int(np.flatnonzero(event_types == vector_runtime.EVENT_BONUS_CLEAR)[0])
    event_slice = slice(clear_index + 1, clear_index + 7)
    np.testing.assert_array_equal(
        env.state["event_type"][0, event_slice],
        np.asarray(
            [
                vector_runtime.EVENT_PROPERTY,
                vector_runtime.EVENT_BONUS_STACK,
                vector_runtime.EVENT_PROPERTY,
                vector_runtime.EVENT_BONUS_STACK,
                vector_runtime.EVENT_PROPERTY,
                vector_runtime.EVENT_BONUS_STACK,
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        env.state["event_player"][0, event_slice],
        np.asarray([2, 2, 1, 1, 0, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        env.state["event_value_i"][0, clear_index + 1, :],
        np.asarray([vector_runtime.PROPERTY_COLOR, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        env.state["event_value_i"][0, clear_index + 3, :],
        np.asarray([vector_runtime.PROPERTY_COLOR, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        env.state["event_value_i"][0, clear_index + 5, :],
        np.asarray([vector_runtime.PROPERTY_COLOR, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(env.state["bonus_active"], np.asarray([[False]]))
    np.testing.assert_array_equal(env.state["bonus_count"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(env.state["avatar_color"], np.asarray([[1, 2, 0]]))
    np.testing.assert_array_equal(
        env.state["base_avatar_color"],
        np.asarray([[0, 1, 2]]),
    )
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[1, 1, 1]]))
    np.testing.assert_array_equal(
        env.state["bonus_stack_type"][0, :, 0],
        np.asarray([vector_runtime.BONUS_TYPE_ALL_COLOR] * 3, dtype=np.int16),
    )
    np.testing.assert_array_equal(
        env.state["bonus_stack_duration_ms"][0, :, 0],
        np.asarray([vector_runtime.BONUS_ALL_COLOR_DURATION_MS] * 3, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        env.state["bonus_stack_color"][0, :, 0],
        np.asarray([1, 2, 0], dtype=np.int16),
    )

    expiry_batch = env.step(
        np.asarray([[1, 1, 1]], dtype=np.int16),
        timer_advance_ms=vector_runtime.BONUS_ALL_COLOR_DURATION_MS,
    )

    _assert_seeded_bonus_public_claim(expiry_batch)
    assert expiry_batch.info["step_counters"]["bonus_all_color_expiries"] == 3
    assert expiry_batch.info["step_counters"]["bonus_stack_appends"] == 0
    np.testing.assert_array_equal(env.state["avatar_color"], np.asarray([[0, 1, 2]]))
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[0, 0, 0]]))
    np.testing.assert_array_equal(
        env.state["bonus_stack_type"][0, :, 0],
        np.asarray([vector_runtime.BONUS_TYPE_NONE] * 3, dtype=np.int16),
    )
    np.testing.assert_array_equal(
        env.state["bonus_stack_color"][0, :, 0],
        np.asarray([-1, -1, -1], dtype=np.int16),
    )


def test_public_seeded_bonus_all_color_overlap_uses_older_stack_until_expiry():
    env = _direct_public_seeded_bonus_env(
        player_count=3,
        positions=[[20.0, 20.0], [50.0, 20.0], [80.0, 20.0]],
        bonus_type="BonusAllColor",
    )

    first_batch = env.step(np.asarray([[1, 1, 1]], dtype=np.int16))

    _assert_seeded_bonus_public_claim(first_batch)
    np.testing.assert_array_equal(env.state["avatar_color"], np.asarray([[1, 2, 0]]))
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[1, 1, 1]]))

    env.state["event_count"][0] = 0
    env.state["event_mask"][0] = False
    env.state["event_type"][0] = -1
    env.state["event_player"][0] = -1
    env.state["event_value_i"][0] = 0
    env.state["event_value_f"][0] = 0.0
    env.state["bonus_world_active"][0] = True
    env.state["bonus_active"][0, 0] = True
    env.state["bonus_type"][0, 0] = vector_runtime.BONUS_TYPE_ALL_COLOR
    env.state["bonus_id"][0, 0] = 2
    env.state["bonus_pos"][0, 0] = env.state["pos"][0, 0]
    env.state["bonus_radius"][0, 0] = 3.0
    env.state["bonus_count"][0] = 1
    env.state["bonus_world_body_count"][0] = 1

    second_batch = env.step(
        np.asarray([[1, 1, 1]], dtype=np.int16),
        timer_advance_ms=1.0,
    )

    _assert_seeded_bonus_public_claim(second_batch)
    assert second_batch.info["step_counters"]["bonus_all_color_catches"] == 1
    np.testing.assert_array_equal(env.state["avatar_color"], np.asarray([[1, 2, 0]]))
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[2, 2, 2]]))
    np.testing.assert_array_equal(
        env.state["bonus_stack_color"][0, :, 0],
        np.asarray([1, 2, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        env.state["bonus_stack_color"][0, :, 1],
        np.asarray([2, 0, 1], dtype=np.int16),
    )

    first_expiry_batch = env.step(
        np.asarray([[1, 1, 1]], dtype=np.int16),
        timer_advance_ms=vector_runtime.BONUS_ALL_COLOR_DURATION_MS - 1,
    )

    _assert_seeded_bonus_public_claim(first_expiry_batch)
    assert first_expiry_batch.info["step_counters"]["bonus_all_color_expiries"] == 3
    np.testing.assert_array_equal(env.state["avatar_color"], np.asarray([[2, 0, 1]]))
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[1, 1, 1]]))
    np.testing.assert_array_equal(
        env.state["bonus_stack_color"][0, :, 0],
        np.asarray([2, 0, 1], dtype=np.int16),
    )

    second_expiry_batch = env.step(
        np.asarray([[1, 1, 1]], dtype=np.int16),
        timer_advance_ms=1.0,
    )

    _assert_seeded_bonus_public_claim(second_expiry_batch)
    assert second_expiry_batch.info["step_counters"]["bonus_all_color_expiries"] == 3
    np.testing.assert_array_equal(env.state["avatar_color"], np.asarray([[0, 1, 2]]))
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[0, 0, 0]]))


def test_public_seeded_bonus_rejects_unsupported_bonus_types():
    env = VectorMultiplayerEnv(batch_size=1, player_count=2)
    env.reset(seed=np.asarray([101], dtype=np.uint64))

    with pytest.raises(VectorMultiplayerEnvError, match="public runtime-backed"):
        env.seed_active_bonus(
            row=0,
            bonus_type="BonusUnsupportedForPublicTest",
            x=20.0,
            y=20.0,
        )

    with pytest.raises(VectorMultiplayerEnvError, match="public runtime-backed"):
        env.seed_active_bonus(
            row=0,
            bonus_type=999,
            x=20.0,
            y=20.0,
        )


def test_public_natural_bonus_spawn_is_off_by_default():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        random_tape_capacity=16,
    )
    batch = env.reset(
        seed=np.asarray([101], dtype=np.uint64),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )

    assert batch.info["bonus_support"]["natural_bonus_spawn"] is False
    assert batch.info["bonus_support_mode"] == "disabled"
    np.testing.assert_array_equal(
        batch.info["natural_bonus_timer_active"],
        np.asarray([False], dtype=bool),
    )
    assert "bonus_active" not in env.state


def test_public_natural_bonus_spawn_uses_fixture_tape_and_schedules_next_pop():
    tape = _natural_bonus_spawn_tape()
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        random_tape_capacity=tape.shape[1],
        timer_capacity=4,
        event_mode="debug-event",
        natural_bonus_spawn=True,
        natural_bonus_type_codes=("BonusSelfSmall",),
    )
    reset_batch = env.reset(
        seed=np.asarray([101], dtype=np.uint64),
        source_fixture_random_tape_values=tape,
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    env.decision_ms = 0.0

    assert reset_batch.info["bonus_support_mode"] == "natural_spawn"
    assert reset_batch.info["bonus_support"]["natural_bonus_spawn"] is True
    assert int(reset_batch.info["random_tape_cursor"][0]) == 7
    assert reset_batch.info["natural_bonus_reset_info"]["random_calls"][-1]["label"] == (
        "bonus.start_delay"
    )
    np.testing.assert_allclose(
        reset_batch.info["natural_bonus_timer_remaining_ms"],
        np.asarray([3000.0]),
    )

    batch = env.step(
        np.asarray([[1, 1]], dtype=np.int16),
        timer_advance_ms=np.asarray([3000.0], dtype=np.float64),
    )

    natural_info = batch.info["natural_bonus_info"]
    assert [call["label"] for call in natural_info["random_calls"]] == [
        "bonus.next_delay_after_pop",
        "bonus.type.BonusSelfSmall",
        "bonus.position.x",
        "bonus.position.y",
    ]
    assert int(batch.info["random_tape_cursor"][0]) == tape.shape[1]
    assert natural_info["schedule_calls"][0]["delay_ms"] == pytest.approx(4500.0)
    np.testing.assert_array_equal(natural_info["due_rows"], np.asarray([True]))
    spawn_info = natural_info["spawn_infos"][0]
    np.testing.assert_array_equal(spawn_info["spawn_rows"], np.asarray([True]))
    np.testing.assert_allclose(spawn_info["spawned_pos"], [[23.94, 64.06]])
    np.testing.assert_array_equal(env.state["bonus_active"][:, :1], np.asarray([[True]]))
    assert int(env.state["bonus_type"][0, 0]) == vector_runtime.BONUS_TYPE_SELF_SMALL
    assert int(env.state["bonus_id"][0, 0]) == 1
    np.testing.assert_allclose(env.state["bonus_pos"][0, 0], [23.94, 64.06])
    np.testing.assert_allclose(
        batch.info["natural_bonus_timer_remaining_ms"],
        np.asarray([4500.0]),
    )


def test_public_natural_bonus_self_master_spawns_catches_and_expires():
    env = _direct_public_natural_bonus_env(bonus_type="BonusSelfMaster")

    catch_batch = env.step(
        np.asarray([[1, 1]], dtype=np.int16),
        timer_advance_ms=np.asarray([3000.0], dtype=np.float64),
    )

    assert catch_batch.info["bonus_support_mode"] == "natural_spawn"
    assert catch_batch.info["step_counters"]["bonus_self_master_catches"] == 1
    assert catch_batch.info["step_counters"]["bonus_stack_appends"] == 1
    assert catch_batch.info["step_counters"]["random_tape_draws"] == 1
    np.testing.assert_array_equal(env.state["bonus_active"][:, :1], np.asarray([[False]]))
    np.testing.assert_array_equal(env.state["invincible"], np.asarray([[True, False]]))
    np.testing.assert_array_equal(env.state["printing"], np.asarray([[False, True]]))
    assert int(env.state["bonus_stack_type"][0, 0, 0]) == (
        vector_runtime.BONUS_TYPE_SELF_MASTER
    )

    env._natural_bonus_timer_active[0] = False
    expiry_batch = env.step(
        np.asarray([[1, 1]], dtype=np.int16),
        timer_advance_ms=vector_runtime.BONUS_SELF_MASTER_DURATION_MS,
    )

    assert expiry_batch.info["step_counters"]["bonus_self_master_expiries"] == 1
    assert expiry_batch.info["step_counters"]["random_tape_draws"] == 1
    np.testing.assert_array_equal(env.state["invincible"], np.asarray([[False, False]]))
    np.testing.assert_array_equal(env.state["printing"], np.asarray([[True, True]]))
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[0, 0]]))


def test_public_natural_bonus_all_color_spawns_catches_and_expires():
    env = _direct_public_natural_bonus_env(bonus_type="BonusAllColor")

    catch_batch = env.step(
        np.asarray([[1, 1]], dtype=np.int16),
        timer_advance_ms=np.asarray([3000.0], dtype=np.float64),
    )

    assert catch_batch.info["bonus_support_mode"] == "natural_spawn"
    assert catch_batch.info["step_counters"]["bonus_all_color_catches"] == 1
    assert catch_batch.info["step_counters"]["bonus_stack_appends"] == 2
    np.testing.assert_array_equal(env.state["bonus_active"][:, :1], np.asarray([[False]]))
    np.testing.assert_array_equal(env.state["avatar_color"], np.asarray([[1, 0]]))
    np.testing.assert_array_equal(
        env.state["bonus_stack_color"][0, :, 0],
        np.asarray([1, 0], dtype=np.int16),
    )

    env._natural_bonus_timer_active[0] = False
    expiry_batch = env.step(
        np.asarray([[1, 1]], dtype=np.int16),
        timer_advance_ms=vector_runtime.BONUS_ALL_COLOR_DURATION_MS,
    )

    assert expiry_batch.info["step_counters"]["bonus_all_color_expiries"] == 2
    np.testing.assert_array_equal(env.state["avatar_color"], np.asarray([[0, 1]]))
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[0, 0]]))


@pytest.mark.parametrize(
    (
        "bonus_type",
        "type_code",
        "catch_counter",
        "expiry_counter",
        "duration_ms",
        "expected_radius",
        "expected_radius_power",
        "expected_speed",
    ),
    [
        (
            "BonusSelfSmall",
            vector_runtime.BONUS_TYPE_SELF_SMALL,
            "bonus_self_small_catches",
            "bonus_self_small_expiries",
            vector_runtime.BONUS_SELF_SMALL_DURATION_MS,
            0.3,
            -1,
            16.0,
        ),
        (
            "BonusSelfSlow",
            vector_runtime.BONUS_TYPE_SELF_SLOW,
            "bonus_self_slow_catches",
            "bonus_self_slow_expiries",
            vector_runtime.BONUS_SELF_SLOW_DURATION_MS,
            0.6,
            0,
            8.0,
        ),
        (
            "BonusSelfFast",
            vector_runtime.BONUS_TYPE_SELF_FAST,
            "bonus_self_fast_catches",
            "bonus_self_fast_expiries",
            vector_runtime.BONUS_SELF_FAST_DURATION_MS,
            0.6,
            0,
            28.0,
        ),
    ],
)
def test_public_natural_bonus_self_effects_spawn_catch_and_expire(
    bonus_type: str,
    type_code: int,
    catch_counter: str,
    expiry_counter: str,
    duration_ms: int,
    expected_radius: float,
    expected_radius_power: int,
    expected_speed: float,
):
    env = _direct_public_natural_bonus_env(bonus_type=bonus_type)
    env.state["print_manager_active"][0] = False
    base_angular_velocity = env.state["angular_velocity_per_ms"].copy()

    catch_batch = env.step(
        np.asarray([[1, 1]], dtype=np.int16),
        timer_advance_ms=np.asarray([3000.0], dtype=np.float64),
    )

    assert catch_batch.info["bonus_support_mode"] == "natural_spawn"
    assert catch_batch.info["step_counters"][catch_counter] == 1
    assert catch_batch.info["step_counters"]["bonus_stack_appends"] == 1
    np.testing.assert_array_equal(env.state["bonus_active"][:, :1], np.asarray([[False]]))
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[1, 0]]))
    assert int(env.state["bonus_stack_type"][0, 0, 0]) == type_code
    assert int(env.state["bonus_stack_duration_ms"][0, 0, 0]) == duration_ms
    assert env.state["radius"][0, 0] == pytest.approx(expected_radius)
    assert int(env.state["radius_power"][0, 0]) == expected_radius_power
    assert env.state["speed"][0, 0] == pytest.approx(expected_speed)
    np.testing.assert_array_equal(env.state["inverse"], np.asarray([[False, False]]))
    np.testing.assert_allclose(env.state["angular_velocity_per_ms"], base_angular_velocity)

    env._natural_bonus_timer_active[0] = False
    expiry_batch = env.step(
        np.asarray([[1, 1]], dtype=np.int16),
        timer_advance_ms=float(duration_ms),
    )

    assert expiry_batch.info["step_counters"][expiry_counter] == 1
    assert expiry_batch.info["step_counters"]["bonus_stack_appends"] == 0
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[0, 0]]))
    assert int(env.state["bonus_stack_type"][0, 0, 0]) == vector_runtime.BONUS_TYPE_NONE
    np.testing.assert_allclose(env.state["radius"], np.asarray([[0.6, 0.6]]))
    np.testing.assert_array_equal(env.state["radius_power"], np.asarray([[0, 0]]))
    np.testing.assert_allclose(env.state["speed"], np.asarray([[16.0, 16.0]]))
    np.testing.assert_array_equal(env.state["inverse"], np.asarray([[False, False]]))
    np.testing.assert_allclose(env.state["angular_velocity_per_ms"], base_angular_velocity)


@pytest.mark.parametrize(
    (
        "bonus_type",
        "type_code",
        "catch_counter",
        "expiry_counter",
        "duration_ms",
    ),
    [
        (
            "BonusEnemySlow",
            vector_runtime.BONUS_TYPE_ENEMY_SLOW,
            "bonus_enemy_slow_catches",
            "bonus_enemy_slow_expiries",
            vector_runtime.BONUS_ENEMY_SLOW_DURATION_MS,
        ),
        (
            "BonusEnemyFast",
            vector_runtime.BONUS_TYPE_ENEMY_FAST,
            "bonus_enemy_fast_catches",
            "bonus_enemy_fast_expiries",
            vector_runtime.BONUS_ENEMY_FAST_DURATION_MS,
        ),
        (
            "BonusEnemyBig",
            vector_runtime.BONUS_TYPE_ENEMY_BIG,
            "bonus_enemy_big_catches",
            "bonus_enemy_big_expiries",
            vector_runtime.BONUS_ENEMY_BIG_DURATION_MS,
        ),
        (
            "BonusEnemyInverse",
            vector_runtime.BONUS_TYPE_ENEMY_INVERSE,
            "bonus_enemy_inverse_catches",
            "bonus_enemy_inverse_expiries",
            vector_runtime.BONUS_ENEMY_INVERSE_DURATION_MS,
        ),
        (
            "BonusEnemyStraightAngle",
            vector_runtime.BONUS_TYPE_ENEMY_STRAIGHT_ANGLE,
            "bonus_enemy_straight_angle_catches",
            "bonus_enemy_straight_angle_expiries",
            vector_runtime.BONUS_ENEMY_STRAIGHT_ANGLE_DURATION_MS,
        ),
    ],
)
def test_public_natural_bonus_enemy_effects_spawn_catch_and_expire(
    bonus_type: str,
    type_code: int,
    catch_counter: str,
    expiry_counter: str,
    duration_ms: int,
):
    env = _direct_public_natural_bonus_env(bonus_type=bonus_type)
    env.state["print_manager_active"][0] = False
    base_angular_velocity = env.state["angular_velocity_per_ms"].copy()

    catch_batch = env.step(
        np.asarray([[1, 1]], dtype=np.int16),
        timer_advance_ms=np.asarray([3000.0], dtype=np.float64),
    )

    assert catch_batch.info["bonus_support_mode"] == "natural_spawn"
    assert catch_batch.info["step_counters"][catch_counter] == 1
    assert catch_batch.info["step_counters"]["bonus_stack_appends"] == 1
    np.testing.assert_array_equal(env.state["bonus_active"][:, :1], np.asarray([[False]]))
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[0, 1]]))
    assert int(env.state["bonus_stack_type"][0, 1, 0]) == type_code
    assert int(env.state["bonus_stack_duration_ms"][0, 1, 0]) == duration_ms
    if bonus_type == "BonusEnemySlow":
        np.testing.assert_allclose(env.state["speed"], np.asarray([[16.0, 8.0]]))
    elif bonus_type == "BonusEnemyFast":
        np.testing.assert_allclose(env.state["speed"], np.asarray([[16.0, 28.0]]))
    elif bonus_type == "BonusEnemyBig":
        np.testing.assert_allclose(env.state["radius"], np.asarray([[0.6, 1.2]]))
        np.testing.assert_array_equal(env.state["radius_power"], np.asarray([[0, 1]]))
    elif bonus_type == "BonusEnemyInverse":
        np.testing.assert_array_equal(env.state["inverse"], np.asarray([[False, True]]))
        assert int(env.state["bonus_stack_inverse_delta"][0, 1, 0]) == 1
    else:
        assert bonus_type == "BonusEnemyStraightAngle"
        assert env.state["angular_velocity_per_ms"][0, 1] == pytest.approx(
            vector_runtime.SOURCE_STRAIGHT_ANGLE_RADIANS
        )
        assert env.state["bonus_stack_angular_velocity_per_ms"][0, 1, 0] == (
            pytest.approx(vector_runtime.SOURCE_STRAIGHT_ANGLE_RADIANS)
        )

    env._natural_bonus_timer_active[0] = False
    expiry_batch = env.step(
        np.asarray([[1, 1]], dtype=np.int16),
        timer_advance_ms=float(duration_ms),
    )

    assert expiry_batch.info["step_counters"][expiry_counter] == 1
    assert expiry_batch.info["step_counters"]["bonus_stack_appends"] == 0
    np.testing.assert_array_equal(env.state["bonus_stack_count"], np.asarray([[0, 0]]))
    assert int(env.state["bonus_stack_type"][0, 1, 0]) == vector_runtime.BONUS_TYPE_NONE
    np.testing.assert_allclose(env.state["radius"], np.asarray([[0.6, 0.6]]))
    np.testing.assert_array_equal(env.state["radius_power"], np.asarray([[0, 0]]))
    np.testing.assert_allclose(env.state["speed"], np.asarray([[16.0, 16.0]]))
    np.testing.assert_array_equal(env.state["inverse"], np.asarray([[False, False]]))
    np.testing.assert_allclose(env.state["angular_velocity_per_ms"], base_angular_velocity)


def test_public_natural_bonus_game_borderless_spawns_catches_and_expires():
    env = _direct_public_natural_bonus_env(bonus_type="BonusGameBorderless")
    env.state["print_manager_active"][0] = False

    catch_batch = env.step(
        np.asarray([[1, 1]], dtype=np.int16),
        timer_advance_ms=np.asarray([3000.0], dtype=np.float64),
    )

    assert catch_batch.info["bonus_support_mode"] == "natural_spawn"
    assert catch_batch.info["step_counters"]["bonus_game_borderless_catches"] == 1
    assert catch_batch.info["step_counters"]["bonus_stack_appends"] == 1
    np.testing.assert_array_equal(env.state["bonus_active"][:, :1], np.asarray([[False]]))
    np.testing.assert_array_equal(env.state["borderless"], np.asarray([True]))
    np.testing.assert_array_equal(env.state["bonus_game_stack_count"], np.asarray([1]))
    assert int(env.state["bonus_game_stack_type"][0, 0]) == (
        vector_runtime.BONUS_TYPE_GAME_BORDERLESS
    )

    env._natural_bonus_timer_active[0] = False
    expiry_batch = env.step(
        np.asarray([[1, 1]], dtype=np.int16),
        timer_advance_ms=float(vector_runtime.BONUS_GAME_BORDERLESS_DURATION_MS),
    )

    assert expiry_batch.info["step_counters"]["bonus_game_borderless_expiries"] == 1
    assert expiry_batch.info["step_counters"]["bonus_stack_appends"] == 0
    np.testing.assert_array_equal(env.state["borderless"], np.asarray([False]))
    np.testing.assert_array_equal(env.state["bonus_game_stack_count"], np.asarray([0]))
    assert int(env.state["bonus_game_stack_type"][0, 0]) == vector_runtime.BONUS_TYPE_NONE


def test_public_natural_bonus_game_clear_spawns_catches_and_clears_world():
    env = _direct_public_natural_bonus_env(bonus_type="BonusGameClear")
    env.state["print_manager_active"][0] = False
    env.state["world_active"][0] = True
    env.state["world_body_count"][0] = 1
    env.state["body_active"][0, 0] = True
    env.state["body_pos"][0, 0] = np.asarray([10.0, 10.0], dtype=np.float64)
    env.state["body_radius"][0, 0] = 1.0
    env.state["body_owner"][0, 0] = 1
    env.state["body_num"][0, 0] = 0
    env.state["body_insert_tick"][0, 0] = int(env.state["tick"][0])
    env.state["body_insert_kind"][0, 0] = vector_runtime.BODY_KIND_NORMAL
    env.state["body_write_cursor"][0] = 1

    batch = env.step(
        np.asarray([[1, 1]], dtype=np.int16),
        timer_advance_ms=np.asarray([3000.0], dtype=np.float64),
    )

    assert batch.info["bonus_support_mode"] == "natural_spawn"
    assert batch.info["step_counters"]["bonus_game_clear_catches"] == 1
    assert batch.info["step_counters"]["bonus_stack_appends"] == 0
    np.testing.assert_array_equal(env.state["bonus_active"][:, :1], np.asarray([[False]]))
    np.testing.assert_array_equal(env.state["bonus_count"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(env.state["world_body_count"], np.asarray([0]))
    np.testing.assert_array_equal(env.state["body_active"], np.zeros((1, 8), dtype=bool))
    np.testing.assert_array_equal(env.state["body_write_cursor"], np.asarray([0]))


def test_public_natural_bonus_uses_source_bonus_cap_for_stack_capacity():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        random_tape_capacity=16,
        natural_bonus_spawn=True,
    )
    env.reset(
        seed=np.asarray([101], dtype=np.uint64),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )

    assert env.state["bonus_stack_id"].shape[2] == (
        vector_runtime.SOURCE_MAX_ACTIVE_BONUSES
    )
    assert env.state["bonus_game_stack_id"].shape[1] == (
        vector_runtime.SOURCE_MAX_ACTIVE_BONUSES
    )


@pytest.mark.parametrize(
    ("type_draw", "expected_name"),
    (
        (0.0, "BonusSelfSmall"),
        (0.945, "BonusGameClear"),
    ),
)
def test_public_natural_bonus_spawn_default_types_select_source_default_bonus(
    type_draw: float,
    expected_name: str,
):
    tape = _natural_bonus_spawn_tape(type_draw=type_draw)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        random_tape_capacity=tape.shape[1],
        timer_capacity=4,
        event_mode="debug-event",
        natural_bonus_spawn=True,
    )
    env.reset(
        seed=np.asarray([101], dtype=np.uint64),
        source_fixture_random_tape_values=tape,
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    env.decision_ms = 0.0

    batch = env.step(
        np.asarray([[1, 1]], dtype=np.int16),
        timer_advance_ms=np.asarray([3000.0], dtype=np.float64),
    )

    natural_info = batch.info["natural_bonus_info"]
    spawn_info = natural_info["spawn_infos"][0]
    expected_code = vector_runtime.BONUS_TYPE_NAME_BY_CODE.index(expected_name)
    assert str(spawn_info["selected_type_name"][0]) == expected_name
    assert int(spawn_info["selected_type_code"][0]) == expected_code
    assert int(env.state["bonus_type"][0, 0]) == expected_code
    assert expected_name in batch.info["bonus_support"]["supported_natural_bonus_types"]
    assert [call["label"] for call in natural_info["random_calls"]] == [
        "bonus.next_delay_after_pop",
        f"bonus.type.{expected_name}",
        "bonus.position.x",
        "bonus.position.y",
    ]
    if expected_name != "BonusSelfSmall":
        assert expected_name in batch.info["bonus_support"][
            "supported_natural_bonus_types"
        ]


def test_public_natural_bonus_same_frame_pop_precedes_print_manager_start_draws():
    tape = _natural_bonus_spawn_tape().copy()
    tape[0, -2:] = [0.1, 0.9]
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        random_tape_capacity=tape.shape[1],
        timer_capacity=4,
        event_mode="debug-event",
        natural_bonus_spawn=True,
    )
    reset_batch = env.reset(
        seed=np.asarray([101], dtype=np.uint64),
        source_fixture_random_tape_values=tape,
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    env.decision_ms = 0.0
    reset_cursor = int(reset_batch.info["random_tape_cursor"][0])
    assert reset_cursor == 7

    batch = env.step(
        np.asarray([[1, 1]], dtype=np.int16),
        timer_advance_ms=np.asarray([3000.0], dtype=np.float64),
    )

    natural_calls = batch.info["natural_bonus_info"]["random_calls"]
    natural_labels = [call["label"] for call in natural_calls]
    assert natural_labels[0] == "bonus.next_delay_after_pop"
    assert natural_labels[1].startswith("bonus.type.")
    assert natural_labels[2:] == ["bonus.position.x", "bonus.position.y"]
    assert [call["tape_index"] for call in natural_calls] == list(
        range(reset_cursor, reset_cursor + 4)
    )
    assert batch.info["step_counters"]["print_manager_delayed_start_fires"] == 2
    assert batch.info["step_counters"]["random_tape_draws"] == 2
    assert int(batch.info["random_tape_cursor"][0]) == reset_cursor + 6
    np.testing.assert_allclose(
        env.state["print_manager_distance"][0],
        [55.8, 22.2],
    )


def test_public_natural_bonus_spawn_cap_twenty_reschedules_without_spawn():
    spawn_prefix = _lifecycle_random_tape(
        "source_lifecycle_spawn_rng_2p_next_round.json",
    )[0, :6]
    tape = np.asarray([[*spawn_prefix, 0.0, 0.5, 0.5, 0.5]], dtype=np.float64)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        random_tape_capacity=tape.shape[1],
        timer_capacity=4,
        natural_bonus_spawn=True,
    )
    env.reset(
        seed=np.asarray([101], dtype=np.uint64),
        source_fixture_random_tape_values=tape,
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    for slot in range(vector_runtime.SOURCE_MAX_ACTIVE_BONUSES):
        env.state["bonus_active"][0, slot] = True
        env.state["bonus_type"][0, slot] = vector_runtime.BONUS_TYPE_SELF_SMALL
        env.state["bonus_id"][0, slot] = slot + 1
        env.state["bonus_pos"][0, slot] = (10.0 + slot, 20.0)
        env.state["bonus_radius"][0, slot] = vector_runtime.SOURCE_BONUS_RADIUS
    env.state["bonus_count"][0] = vector_runtime.SOURCE_MAX_ACTIVE_BONUSES
    env.state["bonus_world_body_count"][0] = vector_runtime.SOURCE_MAX_ACTIVE_BONUSES
    env.state["bonus_world_active"][0] = True
    env.state["bonus_next_id"][0] = vector_runtime.SOURCE_MAX_ACTIVE_BONUSES + 1
    env.decision_ms = 0.0

    batch = env.step(np.asarray([[1, 1]], dtype=np.int16), timer_advance_ms=3000.0)

    natural_info = batch.info["natural_bonus_info"]
    assert [call["label"] for call in natural_info["random_calls"]] == [
        "bonus.next_delay_after_pop",
    ]
    spawn_info = natural_info["spawn_infos"][0]
    np.testing.assert_array_equal(spawn_info["capped_rows"], np.asarray([True]))
    np.testing.assert_array_equal(spawn_info["spawn_rows"], np.asarray([False]))
    assert spawn_info["type_selection_info"] is None
    assert int(env.state["bonus_count"][0]) == vector_runtime.SOURCE_MAX_ACTIVE_BONUSES
    assert natural_info["schedule_calls"][0]["delay_ms"] == pytest.approx(4500.0)


def test_public_natural_bonus_metadata_keeps_source_defaults_and_exact_unsupported_effects():
    env = VectorMultiplayerEnv(batch_size=1, player_count=2, natural_bonus_spawn=True)
    batch = env.reset(
        seed=np.asarray([101], dtype=np.uint64),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )

    bonus_support = batch.info["bonus_support"]
    expected_default_names = tuple(
        vector_runtime.BONUS_TYPE_NAME_BY_CODE[int(code)]
        for code in vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES
    )
    assert bonus_support["policy_id"].endswith(
        "natural_bonus_source_default_spawn_support/v0"
    )
    assert bonus_support["source_default_natural_bonus_types"] == expected_default_names
    assert bonus_support["supported_natural_bonus_types"] == expected_default_names
    np.testing.assert_array_equal(
        bonus_support["enabled_natural_bonus_type_codes"],
        np.asarray(vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES, dtype=np.int16),
    )
    assert set(bonus_support["supported_natural_bonus_effect_types"]) == set(
        NATURAL_BONUS_EFFECT_TYPE_NAMES
    )
    assert set(bonus_support["supported_natural_bonus_effect_types"]) == set(
        expected_default_names
    )
    assert "BonusSelfMaster" in bonus_support["supported_natural_bonus_effect_types"]
    assert "BonusAllColor" in bonus_support["supported_natural_bonus_effect_types"]
    assert "BonusEnemySlow" in bonus_support["supported_natural_bonus_effect_types"]
    assert "BonusEnemyStraightAngle" in (
        bonus_support["supported_natural_bonus_effect_types"]
    )
    assert bonus_support["unsupported_natural_bonus_types"] == ()
    assert bonus_support["unsupported_natural_bonus_effects"] == ()
    assert set(bonus_support["supported_natural_bonus_effect_types"]).isdisjoint(
        bonus_support["unsupported_natural_bonus_effects"],
    )
    assert batch.info["public_env_contract_id"] == PUBLIC_ENV_CONTRACT_ID
    assert batch.info["env_impl_id"] == ENV_IMPL_ID
    assert batch.info["env_version_policy_id"] == ENV_VERSION_POLICY_ID


def test_reset_from_state_arrays_rejects_shape_mismatch():
    env = VectorMultiplayerEnv(batch_size=1, player_count=3)
    state = {"pos": np.zeros((1, 2, 2), dtype=np.float64)}

    with pytest.raises(VectorMultiplayerEnvError, match="state array 'pos'"):
        env.reset_from_state_arrays(state)


def test_step_rejects_bad_live_action_shape_and_values():
    env = VectorMultiplayerEnv(batch_size=1, player_count=3)
    env.reset(seed=np.asarray([1], dtype=np.uint64))

    with pytest.raises(VectorMultiplayerEnvError, match=r"shape \[B,P\]"):
        env.step(np.ones((1, 2), dtype=np.int16))
    with pytest.raises(VectorMultiplayerEnvError, match="left/straight/right"):
        env.step(np.asarray([[1, 99, 1]], dtype=np.int16))


def test_public_lifecycle_metadata_arrays_exist_from_reset():
    env = VectorMultiplayerEnv(
        batch_size=2,
        player_count=3,
        seed=555,
        timer_capacity=3,
        random_tape_capacity=16,
    )

    for name in PUBLIC_LIFECYCLE_ARRAY_NAMES:
        assert name in env.reset_template
        assert name in env.state

    batch = env.reset(
        seed=np.asarray([101, 202], dtype=np.uint64),
        source_fixture_new_round_time_ms=3000.0,
        source_fixture_warmup_advance_ms=0.0,
    )

    expected_false = np.zeros(2, dtype=bool)
    expected_winner = np.full(2, -1, dtype=np.int16)
    for arrays in (env.reset_template, env.state, batch.info):
        np.testing.assert_array_equal(arrays["round_done"], expected_false)
        np.testing.assert_array_equal(arrays["warmdown_pending"], expected_false)
        np.testing.assert_array_equal(arrays["match_done"], expected_false)
        np.testing.assert_array_equal(arrays["round_winner"], expected_winner)
        np.testing.assert_array_equal(arrays["match_winner"], expected_winner)


@pytest.mark.parametrize("player_count", [2, 3, 4])
def test_public_metadata_contract_is_single_env_version_for_2p_3p_4p(player_count: int):
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=player_count,
        seed=123,
        timer_capacity=max(4, player_count),
        random_tape_capacity=24,
    )
    reset_batch = env.reset(
        seed=np.asarray([100 + player_count], dtype=np.uint64),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    step_batch = env.step(np.ones((1, player_count), dtype=np.int16))

    for batch in (reset_batch, step_batch):
        assert batch.info["public_env_contract_id"] == PUBLIC_ENV_CONTRACT_ID
        assert batch.info["env_impl_id"] == ENV_IMPL_ID
        assert batch.info["env_version_policy_id"] == ENV_VERSION_POLICY_ID
        assert batch.info["supported_player_counts"] == (2, 3, 4)
        assert batch.info["player_count"] == player_count
        np.testing.assert_array_equal(
            batch.info["player_count_by_row"],
            np.asarray([player_count], dtype=np.int16),
        )
        np.testing.assert_array_equal(
            batch.info["present_player_count_by_row"],
            np.asarray([player_count], dtype=np.int16),
        )
        assert batch.observation.shape == (
            1,
            player_count,
            len(DEBUG_METADATA_OBSERVATION_FIELDS),
        )
        assert batch.action_mask.shape == (1, player_count, 3)
        assert batch.reward.shape == (1, player_count)
        assert batch.info["metadata_only"] is True
        assert batch.info["trainer_observation_claim"] is False
        assert batch.info["trainer_replay_claim"] is False
        assert batch.info["learned_observation_claim"] is False
        assert batch.info["public_env_trainer_ready_claim"] is False
        assert "source_default_bonus_spawn_not_enabled_for_no_bonus_ruleset/v0" in (
            batch.info["public_mechanics_gaps"]
        )
        assert "public_observation_is_debug_metadata_only/v0" in (
            batch.info["public_mechanics_gaps"]
        )

        provenance = batch.info["reset_provenance"]
        assert provenance["schema_id"] == RESET_PROVENANCE_POLICY_ID
        assert provenance["seed_alone_replay_complete"] is False
        np.testing.assert_array_equal(provenance["reset_seed"], batch.info["reset_seed"])
        np.testing.assert_array_equal(
            provenance["random_tape_cursor"],
            batch.info["random_tape_cursor"],
        )
        np.testing.assert_array_equal(
            provenance["random_tape_draw_count"],
            batch.info["random_tape_draw_count"],
        )

    action_sidecar = step_batch.info["action_sidecar"]
    assert action_sidecar["schema_id"] == PUBLIC_JOINT_ACTION_SIDECAR_SCHEMA_ID
    assert action_sidecar["player_count"] == player_count
    assert action_sidecar["metadata_only"] is True
    assert action_sidecar["joint_action_mcts_claim"] is False
    assert action_sidecar["action_map_policy_id"] == (
        "external_joint_action_player_major_no_mcts/v0"
    )


@pytest.mark.parametrize(
    (
        "scenario_name",
        "expected_alive",
        "expected_death_count",
        "expected_death_player",
        "expected_death_cause",
        "expected_death_hit_owner",
        "expected_world_body_count",
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
            1,
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
            2,
        ),
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
            0,
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
            2,
        ),
    ],
    ids=[
        "opponent-tangent-safe",
        "opponent-overlap-kills",
        "own-delta3-safe",
        "own-delta4-kills",
        "same-frame-control-safe",
        "same-frame-point-kills",
    ],
)
def test_public_body_canary_fixture_step_matches_expected_deaths(
    scenario_name: str,
    expected_alive: list[bool],
    expected_death_count: int,
    expected_death_player: list[int],
    expected_death_cause: list[int],
    expected_death_hit_owner: list[int],
    expected_world_body_count: int,
):
    env, batch = _step_public_fixture_once(
        f"scenarios/environment/{scenario_name}",
        body_capacity=8,
    )

    assert batch.info["metadata_only"] is True
    assert batch.info["trainer_observation_claim"] is False
    assert batch.info["player_count"] == 3
    np.testing.assert_array_equal(batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(batch.reward, np.zeros((1, 3), dtype=np.float32))
    np.testing.assert_array_equal(
        batch.info["alive"],
        np.asarray([expected_alive], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["death_count"],
        np.asarray([expected_death_count], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.info["death_player"],
        np.asarray([expected_death_player], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        batch.info["death_cause"],
        np.asarray([expected_death_cause], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        batch.info["death_hit_owner"],
        np.asarray([expected_death_hit_owner], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        env.state["world_body_count"],
        np.asarray([expected_world_body_count], dtype=np.int32),
    )
    _assert_public_final_metadata(batch, [], expected_reward=None)


@pytest.mark.parametrize(
    (
        "scenario_name",
        "expected_alive",
        "expected_death_count",
        "expected_death_player",
        "expected_death_hit_owner",
        "expected_world_body_count",
        "expected_printing",
    ),
    [
        (
            "source_trail_gap_hole_space_safe_step.json",
            [True, True, True],
            0,
            [-1, -1, -1],
            [-1, -1, -1],
            1,
            [False, False, False],
        ),
        (
            "source_trail_gap_print_to_hole_boundary_kills_step.json",
            [False, True, True],
            1,
            [0, -1, -1],
            [1, -1, -1],
            2,
            [False, False, False],
        ),
        (
            "source_trail_gap_hole_to_print_boundary_kills_step.json",
            [False, True, True],
            1,
            [0, -1, -1],
            [1, -1, -1],
            2,
            [False, True, False],
        ),
        (
            "source_trail_gap_stored_body_still_kills_step.json",
            [False, True, True],
            1,
            [0, -1, -1],
            [1, -1, -1],
            2,
            [False, False, False],
        ),
    ],
    ids=[
        "hole-space-safe",
        "print-to-hole-boundary-kills",
        "hole-to-print-boundary-kills",
        "stored-body-still-kills",
    ],
)
def test_public_trail_gap_canary_fixture_step_matches_expected_deaths(
    scenario_name: str,
    expected_alive: list[bool],
    expected_death_count: int,
    expected_death_player: list[int],
    expected_death_hit_owner: list[int],
    expected_world_body_count: int,
    expected_printing: list[bool],
):
    env, batch = _step_public_fixture_once(
        f"scenarios/environment/{scenario_name}",
        body_capacity=8,
    )

    assert batch.info["metadata_only"] is True
    assert batch.info["trainer_observation_claim"] is False
    assert batch.info["player_count"] == 3
    np.testing.assert_array_equal(batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(batch.reward, np.zeros((1, 3), dtype=np.float32))
    np.testing.assert_array_equal(
        batch.info["alive"],
        np.asarray([expected_alive], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["death_count"],
        np.asarray([expected_death_count], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.info["death_player"],
        np.asarray([expected_death_player], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        batch.info["death_cause"],
        np.asarray(
            [
                [
                    (
                        vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL
                        if player >= 0
                        else vector_runtime.DEATH_CAUSE_NONE
                    )
                    for player in expected_death_player
                ]
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        batch.info["death_hit_owner"],
        np.asarray([expected_death_hit_owner], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        env.state["world_body_count"],
        np.asarray([expected_world_body_count], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        env.state["printing"],
        np.asarray([expected_printing], dtype=bool),
    )
    _assert_public_final_metadata(batch, [], expected_reward=None)


@pytest.mark.parametrize(
    (
        "scenario_name",
        "expected_alive",
        "expected_score",
        "expected_death_count",
        "expected_death_player",
        "expected_death_hit_owner",
        "expected_winner",
        "expected_draw",
        "expected_terminal_reason",
        "expected_reward",
        "expected_world_body_count",
    ),
    [
        (
            "source_collision_death_point_kills_later_player_step.json",
            [False, False],
            [0, 0],
            2,
            [1, 0],
            [0, 1],
            -1,
            True,
            vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW,
            [0.0, 0.0],
            3,
        ),
        (
            "source_collision_head_head_reverse_order_single_death_step.json",
            [False, True],
            [0, 1],
            1,
            [0, -1],
            [1, -1],
            1,
            False,
            vector_reset.TERMINAL_REASON_SURVIVOR_WIN,
            [-1.0, 1.0],
            3,
        ),
    ],
    ids=["death-point-kills-later-player", "head-head-reverse-order-single-death"],
)
def test_2p_public_collision_order_canary_fixture_matches_source_terminal(
    scenario_name: str,
    expected_alive: list[bool],
    expected_score: list[int],
    expected_death_count: int,
    expected_death_player: list[int],
    expected_death_hit_owner: list[int],
    expected_winner: int,
    expected_draw: bool,
    expected_terminal_reason: int,
    expected_reward: list[float],
    expected_world_body_count: int,
):
    env, batch = _step_public_fixture_once(
        f"scenarios/environment/{scenario_name}",
        body_capacity=8,
    )

    assert batch.info["metadata_only"] is True
    assert batch.info["trainer_observation_claim"] is False
    assert batch.info["player_count"] == 2
    np.testing.assert_array_equal(batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.truncated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        batch.reward,
        np.asarray([expected_reward], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        batch.info["alive"],
        np.asarray([expected_alive], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["score"],
        np.asarray([expected_score], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.info["death_count"],
        np.asarray([expected_death_count], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.info["death_player"],
        np.asarray([expected_death_player], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        batch.info["death_cause"],
        np.asarray(
            [
                [
                    (
                        vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL
                        if player >= 0
                        else vector_runtime.DEATH_CAUSE_NONE
                    )
                    for player in expected_death_player
                ]
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        batch.info["death_hit_owner"],
        np.asarray([expected_death_hit_owner], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        batch.info["winner"],
        np.asarray([expected_winner], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        batch.info["draw"],
        np.asarray([expected_draw], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["terminal_reason"],
        np.asarray([expected_terminal_reason], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        env.state["world_body_count"],
        np.asarray([expected_world_body_count], dtype=np.int32),
    )
    _assert_public_final_metadata(batch, [0], expected_reward=batch.reward)


def test_2p_public_terminal_step_labels_metadata_only_final_rows():
    state, actions, step_ms = _fixture_state_and_actions(
        "scenarios/environment/source_normal_wall_death_step.json",
        body_capacity=8,
    )
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        decision_ms=step_ms,
        body_capacity=8,
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=8,
        event_mode="debug-event",
    )
    env.reset_from_state_arrays(
        state,
        reset_seed=np.asarray([101], dtype=np.uint64),
    )

    batch = env.step(actions)

    assert batch.info["metadata_only"] is True
    assert batch.info["trainer_observation_claim"] is False
    np.testing.assert_array_equal(batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.info["needs_reset"], np.asarray([True], dtype=bool))
    _assert_public_final_metadata(batch, [0], expected_reward=batch.reward)

    with pytest.raises(RuntimeError, match="reset must be called before stepping rows"):
        env.step(actions)


def test_2p_public_autoreset_done_rows_preserves_final_metadata_and_live_rows():
    state, actions, step_ms = _fixture_state_and_actions(
        "scenarios/environment/source_normal_wall_death_step.json",
        body_capacity=8,
    )
    batched_state = {
        name: np.repeat(array, repeats=2, axis=0)
        for name, array in state.items()
    }
    padded_random_tape = np.zeros((2, 16), dtype=np.float64)
    padded_random_tape[:, : batched_state["random_tape_values"].shape[1]] = (
        batched_state["random_tape_values"]
    )
    batched_state["random_tape_values"] = padded_random_tape
    batched_state["pos"][1] = np.asarray([[20.0, 20.0], [60.0, 60.0]], dtype=np.float64)
    batched_state["prev_pos"][1] = batched_state["pos"][1]
    batched_state["heading"][1] = np.asarray([0.0, math.pi], dtype=np.float64)
    batched_state["speed"][1] = np.asarray([0.0, 0.0], dtype=np.float64)

    env = VectorMultiplayerEnv(
        batch_size=2,
        player_count=2,
        decision_ms=step_ms,
        body_capacity=8,
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=16,
        event_mode="debug-event",
    )
    env.reset_from_state_arrays(
        batched_state,
        reset_seed=np.asarray([101, 202], dtype=np.uint64),
    )

    batch = env.step(np.repeat(actions, repeats=2, axis=0))

    np.testing.assert_array_equal(batch.done, np.asarray([True, False], dtype=bool))
    np.testing.assert_array_equal(batch.info["needs_reset"], np.asarray([True, False]))
    _assert_public_final_metadata(batch, [0], expected_reward=batch.reward)
    previous_final_observation = batch.final_observation.copy()
    previous_final_reward = batch.final_reward.copy()
    live_row_pos = env.state["pos"][1].copy()
    live_row_episode_id = int(env.state["episode_id"][1])
    live_row_episode_step = int(env.state["episode_step"][1])

    reset_batch = env.autoreset_done_rows(
        seed=np.asarray([303, 404], dtype=np.uint64),
    )

    np.testing.assert_array_equal(reset_batch.done, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(reset_batch.info["needs_reset"], np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(reset_batch.info["reset_rows"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(
        reset_batch.info["reset_source"],
        np.asarray([vector_reset.RESET_SOURCE_AUTORESET, vector_reset.RESET_SOURCE_FIXTURE]),
    )
    assert reset_batch.info["reset_info"]["random_tape_source"] == (
        vector_source_random.RANDOM_TAPE_SOURCE_SEED_GENERATED
    )
    np.testing.assert_array_equal(
        reset_batch.info["reset_info"]["random_tape_source_by_row"],
        np.asarray(
            [
                vector_source_random.RANDOM_TAPE_SOURCE_SEED_GENERATED,
                "unchanged",
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        reset_batch.info["random_tape_source"],
        np.asarray(
            [
                vector_source_random.RANDOM_TAPE_SOURCE_SEED_GENERATED,
                "direct_vector_runtime_state",
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        reset_batch.info["random_tape_length"],
        np.asarray([16, 0], dtype=np.int32),
    )
    assert reset_batch.info["reset_info"]["rng_impl_id"] == (
        vector_source_random.SOURCE_RANDOM_HISTORY_IMPL_ID
    )
    np.testing.assert_array_equal(
        reset_batch.info["reset_info"]["rng_impl_id_by_row"],
        np.asarray(
            [
                vector_source_random.SOURCE_RANDOM_HISTORY_IMPL_ID,
                "unchanged",
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        reset_batch.info["rng_impl_id"],
        np.asarray(
            [
                vector_source_random.SOURCE_RANDOM_HISTORY_IMPL_ID,
                "external_vector_runtime_state/v0",
            ],
            dtype=object,
        ),
    )
    assert reset_batch.info["reset_info"]["source_fixture_ref"] is None
    np.testing.assert_array_equal(env.state["pos"][1], live_row_pos)
    assert int(env.state["episode_id"][1]) == live_row_episode_id
    assert int(env.state["episode_step"][1]) == live_row_episode_step

    reset_policy = reset_batch.info["public_reset_policy"]
    assert reset_policy["schema_id"] == "curvyzero_public_multiplayer_masked_reset_policy/v0"
    assert reset_policy["api"] == "autoreset_done_rows"
    assert reset_policy["selected_rows_only"] is True
    assert reset_policy["hidden_autoreset"] is False
    assert reset_policy["explicit_autoreset_done_rows"] is True
    assert reset_policy["reset_source"] == vector_reset.RESET_SOURCE_AUTORESET
    np.testing.assert_array_equal(
        reset_policy["row_mask"],
        np.asarray([True, False], dtype=bool),
    )
    np.testing.assert_array_equal(reset_policy["rows"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(
        reset_policy["pre_reset_snapshot_rows"],
        np.asarray([0], dtype=np.int32),
    )

    previous_metadata = reset_policy["previous_step_final_metadata"]
    assert previous_metadata["available"] is True
    assert previous_metadata["overlaps_reset_rows"] is True
    assert previous_metadata["not_mutated_by_reset"] is True
    np.testing.assert_array_equal(previous_metadata["rows"], np.asarray([0], dtype=np.int32))

    snapshot = reset_batch.info["reset_info"]["terminal_transition_snapshot"]["arrays"]
    np.testing.assert_array_equal(snapshot["done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(snapshot["terminated"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        snapshot["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )

    assert env.last_step_info is not None
    np.testing.assert_array_equal(
        env.last_step_info["final_observation"],
        previous_final_observation,
    )
    np.testing.assert_array_equal(
        env.last_step_info["final_reward_map"],
        previous_final_reward,
    )

    with pytest.raises(VectorMultiplayerEnvError, match="only select rows with needs_reset"):
        env.autoreset_done_rows(row_mask=np.asarray([False, True], dtype=bool))


def test_2p_public_reset_to_terminal_matches_source_long_wall_fixture():
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
    room = source_setup["room"]
    assert isinstance(room, dict)
    assert float(lifecycle["new_round_time_ms"]) == 0.0
    assert float(rollout["advance_timers_ms"]) == 0.0

    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        decision_ms=float(rollout["step_ms"]),
        max_score=int(room["max_score"]),
        body_capacity=256,
        event_capacity=64,
        timer_capacity=3,
        random_tape_capacity=len(random_values),
        player_ids=("p0", "p1"),
    )
    reset_batch = env.reset(
        seed=np.asarray([0], dtype=np.uint64),
        source_fixture_random_tape_values=np.asarray([random_values], dtype=np.float64),
        source_fixture_ref=_LONG_NATURAL_ROLLOUT_SCENARIO.name,
        source_fixture_new_round_time_ms=float(lifecycle["new_round_time_ms"]),
        source_fixture_warmup_advance_ms=float(rollout["advance_timers_ms"]),
    )

    assert reset_batch.info["metadata_only"] is True
    assert reset_batch.info["trainer_observation_claim"] is False
    assert reset_batch.info["reset_info"]["source_fixture_ref"] == (
        _LONG_NATURAL_ROLLOUT_SCENARIO.name
    )
    assert reset_batch.info["warmup_info"]["game_start_fires"] == 1
    assert reset_batch.info["warmup_info"]["print_manager_delayed_start_fires"] == 0
    assert int(env.state["random_tape_cursor"][0]) == len(random_values)
    assert int(env.state["random_tape_draw_count"][0]) == len(random_values)
    assert int(env.state["random_tape_length"][0]) == len(random_values)
    np.testing.assert_array_equal(
        env.state["print_manager_active"][0],
        np.zeros(2, dtype=bool),
    )

    expected_source_moves = np.asarray([rollout["moves"]], dtype=np.int8)
    straight_actions = np.asarray([[1, 1]], dtype=np.int16)
    terminal_batch = None
    for tick, record in enumerate(source_records):
        batch = env.step(straight_actions)
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
        _assert_multiplayer_row_matches_source_record(env, record, tick=tick)
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
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["winner"],
        np.asarray([source_winner_index], dtype=np.int16),
    )
    np.testing.assert_array_equal(terminal_batch.info["draw"], np.asarray([False]))
    np.testing.assert_array_equal(
        terminal_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["score"],
        np.asarray([expected["scores"]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["alive"],
        np.asarray([expected["alive"]], dtype=bool),
    )
    np.testing.assert_allclose(
        env.state["pos"][0],
        np.asarray(expected["final_positions"], dtype=np.float64),
        atol=1e-6,
    )
    source_dead_avatar_ids = [
        int(avatar["id"])
        for avatar in final_source_avatars
        if not bool(avatar["alive"])
    ]
    vector_dead_avatar_ids = [
        int(final_source_avatars[index]["id"])
        for index, alive in enumerate(env.state["alive"][0])
        if not bool(alive)
    ]
    assert vector_dead_avatar_ids == source_dead_avatar_ids == expected["deaths"]
    expected_reward = np.full((1, 2), -1.0, dtype=np.float32)
    expected_reward[0, source_winner_index] = 1.0
    np.testing.assert_array_equal(terminal_batch.reward, expected_reward)
    _assert_public_final_metadata(terminal_batch, [0], expected_reward=expected_reward)


def test_3p_public_timeout_truncation_masks_actions_and_pays_zero():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=3,
        max_ticks=1,
        timer_capacity=3,
    )
    env.reset(source_fixture_new_round_time_ms=0.0, source_fixture_warmup_advance_ms=3000.0)
    _force_safe_stationary_3p_positions(env)

    batch = env.step(np.asarray([[1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(batch.truncated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_TIMEOUT_TRUNCATED], dtype=np.int16),
    )
    np.testing.assert_array_equal(batch.info["timeout"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        batch.info["truncation_reason"],
        np.asarray(["timeout"], dtype=object),
    )
    np.testing.assert_array_equal(batch.reward, np.zeros((1, 3), dtype=np.float32))
    np.testing.assert_array_equal(batch.action_mask, np.zeros((1, 3, 3), dtype=bool))
    _assert_public_final_metadata(batch, [0], expected_reward=batch.reward)


@pytest.mark.parametrize(
    ("overflow_name", "expected_reason", "expected_truncation_reason"),
    [
        (
            "body_overflow",
            vector_reset.TERMINAL_REASON_BODY_OVERFLOW_TRUNCATED,
            "capacity_truncated",
        ),
        (
            "event_overflow",
            vector_reset.TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED,
            "event_overflow_truncated",
        ),
    ],
)
def test_3p_public_overflow_truncation_masks_actions_and_pays_zero(
    overflow_name: str,
    expected_reason: int,
    expected_truncation_reason: str,
):
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=3,
        timer_capacity=3,
    )
    env.reset(source_fixture_new_round_time_ms=0.0, source_fixture_warmup_advance_ms=3000.0)
    _force_safe_stationary_3p_positions(env)
    env.state[overflow_name][0] = True

    batch = env.step(np.asarray([[1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(batch.truncated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        batch.info["terminal_reason"],
        np.asarray([expected_reason], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        batch.info["truncation_reason"],
        np.asarray([expected_truncation_reason], dtype=object),
    )
    np.testing.assert_array_equal(batch.reward, np.zeros((1, 3), dtype=np.float32))
    np.testing.assert_array_equal(batch.action_mask, np.zeros((1, 3, 3), dtype=bool))
    _assert_public_final_metadata(batch, [0], expected_reward=batch.reward)


def test_4p_public_reset_uses_source_fixture_tape_for_spawn_and_scheduled_warmup():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        seed=555,
        timer_capacity=4,
        random_tape_capacity=16,
    )
    batch = env.reset(
        seed=np.asarray([555], dtype=np.uint64),
        source_fixture_random_tape_values=_lifecycle_random_tape(
            "source_lifecycle_spawn_rng_order_4p.json"
        ),
        source_fixture_ref="scenarios/environment/source_lifecycle_spawn_rng_order_4p.json",
        source_fixture_new_round_time_ms=3000.0,
        source_fixture_warmup_advance_ms=0.0,
    )

    assert batch.info["metadata_only"] is True
    assert batch.info["trainer_observation_claim"] is False
    assert batch.info["natural_multiplayer_reset_claim"] is False
    assert batch.info["reset_info"]["random_tape_source"] == "source_fixture_random_tape_values"
    np.testing.assert_array_equal(
        batch.info["random_tape_source"],
        np.asarray(["source_fixture_random_tape_values"], dtype=object),
    )
    np.testing.assert_array_equal(
        batch.info["reset_info"]["random_tape_source_by_row"],
        np.asarray(["source_fixture_random_tape_values"], dtype=object),
    )
    np.testing.assert_array_equal(batch.info["random_tape_length"], np.asarray([12], dtype=np.int32))
    np.testing.assert_array_equal(
        batch.info["reset_info"]["random_tape_length"],
        np.asarray([12], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.info["rng_impl_id"],
        np.asarray(["source_fixture_random_tape_values/v0"], dtype=object),
    )
    assert batch.info["reset_info"]["rng_impl_id"] == "source_fixture_random_tape_values/v0"
    np.testing.assert_array_equal(
        batch.info["reset_info"]["rng_impl_id_by_row"],
        np.asarray(["source_fixture_random_tape_values/v0"], dtype=object),
    )
    np.testing.assert_array_equal(
        batch.info["source_fixture_ref"],
        np.asarray(["scenarios/environment/source_lifecycle_spawn_rng_order_4p.json"], dtype=object),
    )
    assert (
        batch.info["reset_info"]["source_fixture_ref"]
        == "scenarios/environment/source_lifecycle_spawn_rng_order_4p.json"
    )
    np.testing.assert_array_equal(
        batch.info["reset_info"]["source_fixture_ref_by_row"],
        np.asarray(["scenarios/environment/source_lifecycle_spawn_rng_order_4p.json"], dtype=object),
    )
    assert batch.info["reset_info"]["random_tape_seeded_by_reset_seed"] is False
    assert batch.info["reset_info"]["natural_multiplayer_reset_claim"] is False
    assert batch.info["player_count"] == 4
    np.testing.assert_array_equal(
        batch.info["present"],
        np.asarray([[True, True, True, True]]),
    )
    np.testing.assert_array_equal(
        batch.info["alive"],
        np.asarray([[True, True, True, True]]),
    )
    np.testing.assert_allclose(
        env.state["pos"][0],
        [[77.41, 50.5], [59.47, 50.5], [41.53, 50.5], [23.59, 50.5]],
    )
    np.testing.assert_allclose(
        env.state["heading"][0],
        [np.pi * 1.5, np.pi, 0.1, np.pi * 0.5],
    )
    np.testing.assert_array_equal(
        batch.action_mask,
        np.ones((1, 4, 3), dtype=bool),
    )
    assert int(batch.info["random_tape_cursor"][0]) == 12
    assert int(batch.info["random_tape_draw_count"][0]) == 12
    assert batch.info["reset_info"]["scheduled_timer_count"] == 1
    assert batch.info["reset_info"]["scheduled_timer_kind"] == "game:start"
    assert batch.info["warmup_info"]["game_start_fires"] == 0
    np.testing.assert_array_equal(
        env.state["timer_kind"],
        np.asarray([[vector_runtime.TIMER_KIND_GAME_START, 0, 0, 0]], dtype=np.int16),
    )
    np.testing.assert_allclose(env.state["timer_remaining_ms"][0], [3000.0, 0.0, 0.0, 0.0])


def test_public_seed_generated_reset_labels_rng_source_and_is_deterministic():
    seed = np.asarray([12345], dtype=np.uint64)
    other_seed = np.asarray([12346], dtype=np.uint64)

    env_a = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        seed=555,
        timer_capacity=4,
        random_tape_capacity=64,
    )
    batch_a = env_a.reset(
        seed=seed,
        source_fixture_new_round_time_ms=3000.0,
        source_fixture_warmup_advance_ms=0.0,
    )

    env_b = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        seed=999,
        timer_capacity=4,
        random_tape_capacity=64,
    )
    batch_b = env_b.reset(
        seed=seed,
        source_fixture_new_round_time_ms=3000.0,
        source_fixture_warmup_advance_ms=0.0,
    )

    env_c = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        seed=555,
        timer_capacity=4,
        random_tape_capacity=64,
    )
    batch_c = env_c.reset(
        seed=other_seed,
        source_fixture_new_round_time_ms=3000.0,
        source_fixture_warmup_advance_ms=0.0,
    )

    assert batch_a.info["natural_multiplayer_reset_claim"] is True
    assert batch_a.info["reset_info"]["natural_multiplayer_reset_claim"] is True
    assert batch_a.info["reset_info"]["natural_multiplayer_reset_claim_scope"] == (
        "seeded_source_history_reset_spawn_warmup_call_order/v0"
    )
    assert batch_a.info["reset_info"]["random_tape_source"] == (
        vector_source_random.RANDOM_TAPE_SOURCE_SEED_GENERATED
    )
    np.testing.assert_array_equal(
        batch_a.info["random_tape_source"],
        np.asarray(
            [vector_source_random.RANDOM_TAPE_SOURCE_SEED_GENERATED],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        batch_a.info["reset_info"]["random_tape_source_by_row"],
        np.asarray(
            [vector_source_random.RANDOM_TAPE_SOURCE_SEED_GENERATED],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(batch_a.info["random_tape_length"], np.asarray([64], dtype=np.int32))
    np.testing.assert_array_equal(
        batch_a.info["reset_info"]["random_tape_length"],
        np.asarray([64], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch_a.info["rng_impl_id"],
        np.asarray([vector_source_random.SOURCE_RANDOM_HISTORY_IMPL_ID], dtype=object),
    )
    assert batch_a.info["reset_info"]["rng_impl_id"] == (
        vector_source_random.SOURCE_RANDOM_HISTORY_IMPL_ID
    )
    np.testing.assert_array_equal(
        batch_a.info["reset_info"]["rng_impl_id_by_row"],
        np.asarray([vector_source_random.SOURCE_RANDOM_HISTORY_IMPL_ID], dtype=object),
    )
    assert batch_a.info["reset_info"]["random_tape_history_ref"] == (
        vector_source_random.SOURCE_RANDOM_HISTORY_IMPL_ID
    )
    np.testing.assert_array_equal(batch_a.info["source_fixture_ref"], np.asarray([None], dtype=object))
    assert batch_a.info["reset_info"]["source_fixture_ref"] is None
    np.testing.assert_array_equal(
        batch_a.info["reset_info"]["source_fixture_ref_by_row"],
        np.asarray([None], dtype=object),
    )
    assert batch_a.info["reset_info"]["random_tape_seeded_by_reset_seed"] is True

    np.testing.assert_array_equal(
        batch_b.info["reset_info"]["random_tape_source_by_row"],
        np.asarray(
            [vector_source_random.RANDOM_TAPE_SOURCE_SEED_GENERATED],
            dtype=object,
        ),
    )
    assert batch_c.info["reset_info"]["random_tape_source"] == (
        vector_source_random.RANDOM_TAPE_SOURCE_SEED_GENERATED
    )

    expected_history = vector_source_random.seeded_source_math_random_history(
        seed,
        length=64,
    )
    row_local_history = vector_source_random.seeded_source_math_random_history(
        np.asarray([12345, 12346], dtype=np.uint64),
        length=64,
    )
    assert row_local_history.dtype == np.float64
    assert bool(((row_local_history >= 0.0) & (row_local_history < 1.0)).all())
    np.testing.assert_array_equal(row_local_history[0:1], env_a.state["random_tape_values"])
    np.testing.assert_array_equal(row_local_history[1:2], env_c.state["random_tape_values"])
    np.testing.assert_array_equal(env_a.state["random_tape_values"], expected_history)
    np.testing.assert_array_equal(env_a.state["random_tape_values"], env_b.state["random_tape_values"])
    np.testing.assert_allclose(env_a.state["pos"], env_b.state["pos"])
    np.testing.assert_allclose(env_a.state["heading"], env_b.state["heading"])
    assert not np.array_equal(env_a.state["random_tape_values"], env_c.state["random_tape_values"])


def test_public_seed_generated_random_tape_extends_deterministically_on_demand():
    seed = np.asarray([12345], dtype=np.uint64)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        seed=555,
        timer_capacity=4,
        random_tape_capacity=16,
    )
    env.reset(
        seed=seed,
        source_fixture_new_round_time_ms=3000.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    old_capacity = int(env.state["random_tape_values"].shape[1])
    env.state["random_tape_cursor"][0] = old_capacity
    env.state["random_tape_draw_count"][0] = old_capacity
    random_calls: list[dict[str, object]] = []

    value = env._draw_natural_bonus_random(
        0,
        "test.auto_extend",
        random_calls=random_calls,
    )

    next_capacity = int(env.state["random_tape_values"].shape[1])
    expected = vector_source_random.seeded_source_math_random_history(
        seed,
        length=next_capacity,
    )
    assert next_capacity > old_capacity
    assert value == pytest.approx(float(expected[0, old_capacity]))
    np.testing.assert_array_equal(env.state["random_tape_values"], expected)
    np.testing.assert_array_equal(
        env.state["random_tape_length"],
        np.asarray([next_capacity], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        env.state["random_tape_cursor"],
        np.asarray([old_capacity + 1], dtype=np.int32),
    )
    assert random_calls[-1]["tape_index"] == old_capacity


def test_public_source_fixture_random_tape_stays_strict_on_exhaustion():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        seed=555,
        timer_capacity=4,
        random_tape_capacity=16,
    )
    env.reset(
        seed=np.asarray([12345], dtype=np.uint64),
        source_fixture_new_round_time_ms=3000.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    env._random_tape_source[0] = RANDOM_TAPE_SOURCE_SOURCE_FIXTURE
    env.state["random_tape_values"][0, :2] = np.asarray([0.25, 0.75], dtype=np.float64)
    env.state["random_tape_length"][0] = 2
    env.state["random_tape_cursor"][0] = 2
    env.state["random_tape_draw_count"][0] = 2
    random_calls: list[dict[str, object]] = []

    with pytest.raises(VectorMultiplayerEnvError, match="Math.random tape exhausted"):
        env._draw_natural_bonus_random(
            0,
            "test.fixture_exhaustion",
            random_calls=random_calls,
        )

    assert int(env.state["random_tape_values"].shape[1]) == 16
    assert bool(env.state["random_tape_exhausted"][0]) is True
    assert random_calls == []


def test_public_step_extends_seed_generated_random_tape_before_runtime_step():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        seed=555,
        timer_capacity=4,
        random_tape_capacity=16,
    )
    env.reset(
        seed=np.asarray([12345], dtype=np.uint64),
        source_fixture_new_round_time_ms=3000.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    old_capacity = int(env.state["random_tape_values"].shape[1])
    env.state["random_tape_cursor"][0] = old_capacity - 1
    env.state["random_tape_draw_count"][0] = old_capacity - 1

    batch = env.step(np.asarray([[1, 1]], dtype=np.int16))

    assert int(env.state["random_tape_values"].shape[1]) > old_capacity
    assert batch.info["random_tape_length"][0] == env.state["random_tape_values"].shape[1]


def test_seed_generated_natural_bonus_position_retries_past_attempt_capacity():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        seed=555,
        timer_capacity=4,
        random_tape_capacity=16,
        natural_bonus_spawn=True,
        natural_bonus_position_attempt_capacity=1,
    )
    env.reset(
        seed=np.asarray([12345], dtype=np.uint64),
        source_fixture_new_round_time_ms=3000.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    rejected = (0.20, 0.20)
    accepted = (0.80, 0.80)
    env.state["random_tape_values"][0, :5] = np.asarray(
        [0.0, rejected[0], rejected[1], accepted[0], accepted[1]],
        dtype=np.float64,
    )
    env.state["random_tape_length"][0] = 5
    env.state["random_tape_cursor"][0] = 0
    env.state["random_tape_draw_count"][0] = 0

    map_size = float(env.state["map_size"][0])
    margin = (
        vector_runtime.SOURCE_BONUS_RADIUS
        + vector_runtime.SOURCE_BONUS_POSITION_MARGIN_FRACTION * map_size
    )
    span = map_size - margin * 2.0
    rejected_pos = np.asarray(
        [margin + rejected[0] * span, margin + rejected[1] * span],
        dtype=np.float64,
    )
    env.state["body_active"][0, 0] = True
    env.state["body_pos"][0, 0] = rejected_pos
    env.state["body_radius"][0, 0] = 1.0
    env.state["body_owner"][0, 0] = 0
    env.state["body_num"][0, 0] = 0
    env.state["body_insert_kind"][0, 0] = vector_runtime.BODY_KIND_NORMAL
    env.state["body_write_cursor"][0] = 1
    random_calls: list[dict[str, object]] = []

    spawn_info = env._spawn_natural_bonus_due_row(0, random_calls=random_calls)

    assert [call["label"] for call in random_calls] == [
        "bonus.type.BonusSelfSmall",
        "bonus.position.x",
        "bonus.position.y",
        "bonus.position.retry_1.x",
        "bonus.position.retry_1.y",
    ]
    np.testing.assert_array_equal(spawn_info["accepted_position_attempt"], np.asarray([1]))
    np.testing.assert_array_equal(spawn_info["position_attempt_count"], np.asarray([2]))
    np.testing.assert_array_equal(spawn_info["rejected_game_world_attempts"], np.asarray([1]))
    assert int(env.state["bonus_count"][0]) == 1


def test_fixture_natural_bonus_position_does_not_autoextend_on_tape_exhaustion():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        seed=555,
        timer_capacity=4,
        random_tape_capacity=16,
        natural_bonus_spawn=True,
        natural_bonus_position_attempt_capacity=2,
    )
    env.reset(
        seed=np.asarray([12345], dtype=np.uint64),
        source_fixture_new_round_time_ms=3000.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    env._random_tape_source[0] = RANDOM_TAPE_SOURCE_SOURCE_FIXTURE
    rejected = (0.20, 0.20)
    env.state["random_tape_values"][0, :3] = np.asarray(
        [0.0, rejected[0], rejected[1]],
        dtype=np.float64,
    )
    env.state["random_tape_length"][0] = 3
    env.state["random_tape_cursor"][0] = 0
    env.state["random_tape_draw_count"][0] = 0

    map_size = float(env.state["map_size"][0])
    margin = (
        vector_runtime.SOURCE_BONUS_RADIUS
        + vector_runtime.SOURCE_BONUS_POSITION_MARGIN_FRACTION * map_size
    )
    span = map_size - margin * 2.0
    rejected_pos = np.asarray(
        [margin + rejected[0] * span, margin + rejected[1] * span],
        dtype=np.float64,
    )
    env.state["body_active"][0, 0] = True
    env.state["body_pos"][0, 0] = rejected_pos
    env.state["body_radius"][0, 0] = 1.0
    env.state["body_owner"][0, 0] = 0
    env.state["body_num"][0, 0] = 0
    env.state["body_insert_kind"][0, 0] = vector_runtime.BODY_KIND_NORMAL
    env.state["body_write_cursor"][0] = 1
    random_calls: list[dict[str, object]] = []

    with pytest.raises(VectorMultiplayerEnvError, match="Math.random tape exhausted"):
        env._spawn_natural_bonus_due_row(0, random_calls=random_calls)

    assert bool(env.state["random_tape_exhausted"][0]) is True
    assert int(env.state["bonus_count"][0]) == 0


def test_seed_generated_natural_bonus_timer_handles_many_due_callbacks():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        seed=555,
        timer_capacity=4,
        random_tape_capacity=16,
        natural_bonus_spawn=True,
    )
    env.reset(
        seed=np.asarray([12345], dtype=np.uint64),
        source_fixture_new_round_time_ms=3000.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    env.state["bonus_count"][0] = vector_runtime.SOURCE_MAX_ACTIVE_BONUSES
    env._natural_bonus_timer_remaining_ms[0] = 0.0
    env._natural_bonus_next_due_elapsed_ms[0] = float(env.state["elapsed_ms"][0])

    info = env._advance_natural_bonus_spawn_timers(
        np.asarray([True], dtype=bool),
        advance_ms=np.asarray([120_000.0], dtype=np.float64),
        phase="test",
    )

    assert len(info["schedule_calls"]) > 16
    assert int(env._natural_bonus_pop_count[0]) == len(info["schedule_calls"])
    assert int(env.state["bonus_count"][0]) == vector_runtime.SOURCE_MAX_ACTIVE_BONUSES
    assert int(env.state["random_tape_values"].shape[1]) > 16
    assert bool(env.state["random_tape_exhausted"][0]) is False


@pytest.mark.parametrize("player_count", [2, 3, 4])
def test_public_seed_generated_reset_history_matches_source_env_call_order(player_count):
    seed = np.asarray([424242], dtype=np.uint64)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=player_count,
        seed=555,
        timer_capacity=4,
        random_tape_capacity=64,
    )
    batch = env.reset(
        seed=seed,
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )

    tape = env.state["random_tape_values"][0].copy()
    source_env = CurvyTronSourceEnv(random_values=tape)
    source_spawn_snapshot = source_env.reset(player_count=player_count, warmup_ms=0.0)
    source_env.advance_timers(3000.0)
    source_warmup_snapshot = source_env.snapshot("after_warmup")

    public_spawn_calls = [
        (
            int(call["tape_index"]),
            str(call["site"]),
            int(call["player"]),
            float(call["value"]),
        )
        for call in batch.info["reset_info"]["spawn_info"]["random_calls"]
    ]
    source_spawn_calls = [
        (
            int(call["index"]),
            str(call["label"]["site"]),
            int(call["label"]["avatar"]) - 1,
            float(call["value"]),
        )
        for call in source_env.random_calls
        if str(call["label"]["site"]).startswith("spawn.")
    ]
    assert public_spawn_calls == source_spawn_calls

    spawn_draw_count = len(public_spawn_calls)
    warmup_players = [
        int(player)
        for player in batch.info["warmup_info"]["print_manager_start_players"]
    ]
    public_warmup_calls = [
        (
            spawn_draw_count + index,
            "print_manager.start_distance",
            player,
            float(tape[spawn_draw_count + index]),
        )
        for index, player in enumerate(warmup_players)
    ]
    source_warmup_calls = [
        (
            int(call["index"]),
            str(call["label"]["site"]),
            int(call["label"]["avatar"]) - 1,
            float(call["value"]),
        )
        for call in source_env.random_calls
        if str(call["label"]["site"]) == "print_manager.start_distance"
    ]
    assert source_warmup_calls == public_warmup_calls
    assert [
        (
            int(call["index"]),
            str(call["label"]["site"]),
            int(call["label"]["avatar"]) - 1,
            float(call["value"]),
        )
        for call in source_env.random_calls
    ] == public_spawn_calls + public_warmup_calls

    source_spawn_pos = np.asarray(
        [[avatar["x"], avatar["y"]] for avatar in source_spawn_snapshot["avatars"]],
        dtype=np.float64,
    )
    source_spawn_heading = np.asarray(
        [avatar["angle"] for avatar in source_spawn_snapshot["avatars"]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(env.state["pos"][0], source_spawn_pos, atol=1e-6)
    np.testing.assert_allclose(env.state["heading"][0], source_spawn_heading, atol=1e-6)
    np.testing.assert_allclose(
        env.state["print_manager_distance"][0],
        [
            avatar["printManager"]["distance"]
            for avatar in source_warmup_snapshot["avatars"]
        ],
        atol=1e-6,
    )
    assert int(batch.info["random_tape_cursor"][0]) == len(source_env.random_calls)
    assert batch.info["reset_info"]["natural_multiplayer_reset_claim"] is True


def test_3p_public_reset_skips_absent_player_with_source_fixture_tape():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=3,
        seed=555,
        timer_capacity=3,
        random_tape_capacity=8,
    )
    batch = env.reset(
        seed=np.asarray([555], dtype=np.uint64),
        present=np.asarray([[True, False, True]], dtype=bool),
        source_fixture_random_tape_values=_lifecycle_random_tape(
            "source_lifecycle_present_absent_3p_round_new.json"
        ),
        source_fixture_new_round_time_ms=3000.0,
        source_fixture_warmup_advance_ms=0.0,
    )

    assert batch.info["metadata_only"] is True
    assert batch.info["trainer_observation_claim"] is False
    assert batch.info["natural_multiplayer_reset_claim"] is False
    assert batch.info["player_count"] == 3
    np.testing.assert_array_equal(
        batch.info["present"],
        np.asarray([[True, False, True]]),
    )
    np.testing.assert_array_equal(
        batch.info["alive"],
        np.asarray([[True, False, True]]),
    )
    np.testing.assert_allclose(
        env.state["pos"][0, [0, 2]],
        [[68.575, 47.5], [26.425, 47.5]],
    )
    np.testing.assert_allclose(
        env.state["heading"][0, [0, 2]],
        [np.pi * 1.5, np.pi * 0.5],
    )
    np.testing.assert_array_equal(
        batch.action_mask,
        np.asarray(
            [
                [
                    [True, True, True],
                    [False, False, False],
                    [True, True, True],
                ]
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(batch.info["death_count"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(
        batch.info["death_player"],
        np.asarray([[1, -1, -1]], dtype=np.int16),
    )
    assert int(batch.info["random_tape_cursor"][0]) == 6
    assert int(batch.info["random_tape_draw_count"][0]) == 6
    assert batch.info["reset_info"]["spawn_info"]["spawn_count"] == 1
    np.testing.assert_array_equal(
        batch.info["reset_info"]["spawn_info"]["absent_player_mask"],
        np.asarray([[False, True, False]], dtype=bool),
    )
    assert batch.info["warmup_info"]["game_start_fires"] == 0


def test_4p_public_reset_skips_absent_player_with_source_fixture_tape():
    scenario_name = "source_lifecycle_present_absent_4p_round_new.json"
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        seed=555,
        timer_capacity=4,
        random_tape_capacity=9,
    )
    batch = env.reset(
        seed=np.asarray([555], dtype=np.uint64),
        present=np.asarray([[True, False, True, True]], dtype=bool),
        source_fixture_random_tape_values=_lifecycle_random_tape(scenario_name),
        source_fixture_ref=f"scenarios/environment/{scenario_name}",
        source_fixture_new_round_time_ms=3000.0,
        source_fixture_warmup_advance_ms=0.0,
    )

    assert batch.info["metadata_only"] is True
    assert batch.info["trainer_observation_claim"] is False
    assert batch.info["natural_multiplayer_reset_claim"] is False
    assert batch.info["player_count"] == 4
    np.testing.assert_array_equal(
        batch.info["present"],
        np.asarray([[True, False, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["alive"],
        np.asarray([[True, False, True, True]], dtype=bool),
    )
    np.testing.assert_allclose(
        env.state["pos"][0, [0, 2, 3]],
        [[77.41, 50.5], [41.53, 50.5], [23.59, 50.5]],
    )
    np.testing.assert_allclose(
        env.state["heading"][0, [0, 2, 3]],
        [np.pi * 1.5, 0.1, np.pi * 0.5],
    )
    np.testing.assert_array_equal(
        batch.action_mask,
        np.asarray(
            [
                [
                    [True, True, True],
                    [False, False, False],
                    [True, True, True],
                    [True, True, True],
                ]
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(batch.info["death_count"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(
        batch.info["death_player"],
        np.asarray([[1, -1, -1, -1]], dtype=np.int16),
    )
    assert int(batch.info["random_tape_cursor"][0]) == 9
    assert int(batch.info["random_tape_draw_count"][0]) == 9
    assert batch.info["reset_info"]["spawn_info"]["spawn_count"] == 1
    np.testing.assert_array_equal(
        batch.info["reset_info"]["spawn_info"]["absent_player_mask"],
        np.asarray([[False, True, False, False]], dtype=bool),
    )
    assert [
        (call["tape_index"], call["site"], call["player"])
        for call in batch.info["reset_info"]["spawn_info"]["random_calls"]
    ] == [
        (0, "spawn.position_x", 3),
        (1, "spawn.position_y", 3),
        (2, "spawn.angle_attempt_0", 3),
        (3, "spawn.position_x", 2),
        (4, "spawn.position_y", 2),
        (5, "spawn.angle_attempt_0", 2),
        (6, "spawn.position_x", 0),
        (7, "spawn.position_y", 0),
        (8, "spawn.angle_attempt_0", 0),
    ]
    assert batch.info["reset_info"]["scheduled_timer_count"] == 1
    assert batch.info["reset_info"]["scheduled_timer_kind"] == "game:start"
    assert batch.info["warmup_info"]["game_start_fires"] == 0
    np.testing.assert_array_equal(
        batch.info["source_fixture_ref"],
        np.asarray([f"scenarios/environment/{scenario_name}"], dtype=object),
    )


def test_3p_public_present_absent_survivor_scoring_matches_source_fixture_tape():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=3,
        seed=555,
        decision_ms=100.0,
        body_capacity=8,
        event_capacity=16,
        timer_capacity=3,
        random_tape_capacity=16,
    )
    reset_batch = env.reset(
        seed=np.asarray([555], dtype=np.uint64),
        present=np.asarray([[True, False, True]], dtype=bool),
        source_fixture_random_tape_values=_lifecycle_random_tape(
            "source_lifecycle_present_absent_3p_survivor_score_round_end.json"
        ),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )
    assert int(reset_batch.info["random_tape_cursor"][0]) == 9
    assert reset_batch.info["warmup_info"]["game_start_fires"] == 1

    env.state["pos"][0, 0] = np.asarray([20.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 0] = 0.0
    env.state["prev_pos"][0, 0] = env.state["pos"][0, 0]
    env.state["print_manager_last_pos"][0, 0] = env.state["pos"][0, 0]
    env.state["print_manager_distance"][0, 0] = 999.0
    env.state["pos"][0, 2] = np.asarray([93.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 2] = 0.0
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]

    batch = env.step(np.asarray([[1, -1, 1]], dtype=np.int16))

    assert batch.info["metadata_only"] is True
    assert batch.info["trainer_observation_claim"] is False
    np.testing.assert_array_equal(
        batch.info["present"],
        np.asarray([[True, False, True]]),
    )
    np.testing.assert_array_equal(
        batch.info["alive"],
        np.asarray([[True, False, False]]),
    )
    np.testing.assert_array_equal(
        batch.info["score"],
        np.asarray([[2, 0, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.info["death_count"],
        np.asarray([2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.info["death_player"],
        np.asarray([[1, 2, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(batch.info["winner"], np.asarray([0], dtype=np.int16))
    np.testing.assert_array_equal(batch.info["draw"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    assert batch.info["winner_ids"] == [[0]]
    assert batch.info["loser_ids"] == [[2]]
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["action_required"],
        np.asarray([[True, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["native_control_value"],
        np.asarray([[0, 0, 0]], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        batch.reward,
        np.asarray([[1.0, 0.0, -1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        batch.action_mask,
        np.zeros((1, 3, 3), dtype=bool),
    )
    np.testing.assert_array_equal(batch.info["needs_reset"], np.asarray([True], dtype=bool))
    assert int(batch.info["random_tape_cursor"][0]) == 10
    assert batch.final_observation is not None
    assert batch.final_reward is not None
    _assert_public_final_metadata(batch, [0], expected_reward=batch.reward)


def test_4p_public_present_absent_survivor_scoring_matches_source_fixture_tape():
    scenario_name = "source_lifecycle_present_absent_4p_survivor_score_round_end.json"
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        seed=555,
        decision_ms=100.0,
        body_capacity=24,
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=15,
    )
    reset_batch = env.reset(
        seed=np.asarray([555], dtype=np.uint64),
        present=np.asarray([[True, False, True, True]], dtype=bool),
        source_fixture_random_tape_values=_lifecycle_random_tape(scenario_name),
        source_fixture_ref=f"scenarios/environment/{scenario_name}",
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )
    assert reset_batch.info["metadata_only"] is True
    assert reset_batch.info["trainer_observation_claim"] is False
    assert int(reset_batch.info["random_tape_cursor"][0]) == 13
    assert reset_batch.info["warmup_info"]["game_start_fires"] == 1
    np.testing.assert_array_equal(
        reset_batch.info["warmup_info"]["print_manager_start_players"],
        np.asarray([3, 2, 1, 0], dtype=np.int16),
    )

    env.state["pos"][0, 0] = np.asarray([10.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 0] = 0.0
    env.state["prev_pos"][0, 0] = env.state["pos"][0, 0]
    env.state["speed"][0, 0] = 8.0
    env.state["print_manager_last_pos"][0, 0] = env.state["pos"][0, 0]
    env.state["print_manager_distance"][0, 0] = 999.0
    env.state["pos"][0, 3] = np.asarray([99.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 3] = 0.0
    env.state["prev_pos"][0, 3] = env.state["pos"][0, 3]

    first_death_batch = env.step(np.asarray([[1, -1, 1, 1]], dtype=np.int16))

    assert first_death_batch.info["metadata_only"] is True
    np.testing.assert_array_equal(first_death_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        first_death_batch.info["alive"],
        np.asarray([[True, False, True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["round_score"],
        np.asarray([[0, 0, 0, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["death_player"],
        np.asarray([[1, 3, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["action_sidecar"]["action_required"],
        np.asarray([[True, False, True, True]], dtype=bool),
    )
    assert int(first_death_batch.info["random_tape_cursor"][0]) == 14
    _assert_public_final_metadata(first_death_batch, [], expected_reward=None)

    env.state["pos"][0, 2] = np.asarray([1.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 2] = np.pi
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]

    terminal_batch = env.step(np.asarray([[1, -1, 1, -1]], dtype=np.int16))

    assert terminal_batch.info["metadata_only"] is True
    assert terminal_batch.info["trainer_observation_claim"] is False
    np.testing.assert_array_equal(
        terminal_batch.info["present"],
        np.asarray([[True, False, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["alive"],
        np.asarray([[True, False, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["score"],
        np.asarray([[3, 0, 2, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["death_player"],
        np.asarray([[1, 3, 2, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(terminal_batch.info["death_count"], np.asarray([3], dtype=np.int32))
    np.testing.assert_array_equal(terminal_batch.info["winner"], np.asarray([0], dtype=np.int16))
    np.testing.assert_array_equal(terminal_batch.info["draw"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(terminal_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal_batch.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    assert terminal_batch.info["winner_ids"] == [[0]]
    assert terminal_batch.info["loser_ids"] == [[2, 3]]
    np.testing.assert_array_equal(
        terminal_batch.reward,
        np.asarray([[1.0, 0.0, -1.0, -1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        terminal_batch.action_mask,
        np.zeros((1, 4, 3), dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["needs_reset"],
        np.asarray([True], dtype=bool),
    )
    assert int(terminal_batch.info["random_tape_cursor"][0]) == 15
    assert terminal_batch.final_observation is not None
    assert terminal_batch.final_reward is not None
    _assert_public_final_metadata(terminal_batch, [0], expected_reward=terminal_batch.reward)


def test_3p_public_active_round_leave_continues_then_scores_source_survivor():
    scenario_name = "source_lifecycle_mid_round_remove_avatar_3p_continue_round_end.json"
    env, reset_batch = _reset_3p_public_lifecycle_env(scenario_name)

    assert reset_batch.info["metadata_only"] is True
    assert reset_batch.info["trainer_observation_claim"] is False
    assert int(reset_batch.info["random_tape_cursor"][0]) == 12
    np.testing.assert_array_equal(
        reset_batch.info["present"],
        np.asarray([[True, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        reset_batch.info["alive"],
        np.asarray([[True, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        reset_batch.action_mask,
        np.ones((1, 3, 3), dtype=bool),
    )

    leave_batch = env.remove_player(1)

    assert leave_batch.info["metadata_only"] is True
    assert leave_batch.info["trainer_observation_claim"] is False
    assert leave_batch.info["leave_metadata_only"] is True
    assert leave_batch.info["leave_trainer_claim"] is False
    assert leave_batch.info["leave_source_id_policy"] == (
        "source_avatar_id_is_public_player_id_plus_one/v0"
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_player_ids"],
        np.asarray([1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_source_player_ids"],
        np.asarray([2], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_player_id_by_row"],
        np.asarray([1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_source_player_id_by_row"],
        np.asarray([2], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        leave_batch.info["present"],
        np.asarray([[True, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.info["alive"],
        np.asarray([[True, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.info["death_count"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        leave_batch.info["death_player"],
        np.asarray([[-1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        leave_batch.info["score"],
        np.asarray([[0, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(leave_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        leave_batch.info["round_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.info["warmdown_pending"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.action_mask,
        np.asarray(
            [
                [
                    [True, True, True],
                    [False, False, False],
                    [True, True, True],
                ]
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(
        leave_batch.reward,
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
    )
    _assert_public_final_metadata(leave_batch, [], expected_reward=None)
    assert int(leave_batch.info["random_tape_cursor"][0]) == 13

    env.state["pos"][0, 2] = np.asarray([93.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 2] = 0.0
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]
    terminal_batch = env.step(np.asarray([[1, -1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(
        terminal_batch.info["present"],
        np.asarray([[True, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["alive"],
        np.asarray([[True, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["death_count"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["death_player"],
        np.asarray([[2, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["score"],
        np.asarray([[2, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(terminal_batch.info["winner"], np.asarray([0], dtype=np.int16))
    np.testing.assert_array_equal(terminal_batch.info["draw"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(terminal_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal_batch.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    assert terminal_batch.info["winner_ids"] == [[0]]
    assert terminal_batch.info["loser_ids"] == [[2]]
    np.testing.assert_array_equal(
        terminal_batch.info["action_sidecar"]["action_required"],
        np.asarray([[True, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.reward,
        np.asarray([[1.0, 0.0, -1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        terminal_batch.action_mask,
        np.zeros((1, 3, 3), dtype=bool),
    )
    assert int(terminal_batch.info["random_tape_cursor"][0]) == 14
    _assert_public_final_metadata(
        terminal_batch,
        [0],
        expected_reward=terminal_batch.reward,
    )


def test_3p_public_active_round_leave_next_live_step_masks_leaver_and_pays_zero():
    scenario_name = "source_lifecycle_mid_round_remove_avatar_3p_continue_round_end.json"
    env, _ = _reset_3p_public_lifecycle_env(scenario_name)

    leave_batch = env.remove_player(1)
    np.testing.assert_array_equal(leave_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        leave_batch.action_mask,
        np.asarray(
            [
                [
                    [True, True, True],
                    [False, False, False],
                    [True, True, True],
                ]
            ],
            dtype=bool,
        ),
    )

    _force_safe_stationary_3p_positions(env)
    batch = env.step(np.asarray([[1, -1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(batch.truncated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        batch.info["present"],
        np.asarray([[True, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["alive"],
        np.asarray([[True, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.action_mask,
        np.asarray(
            [
                [
                    [True, True, True],
                    [False, False, False],
                    [True, True, True],
                ]
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(batch.reward, np.zeros((1, 3), dtype=np.float32))
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["action_required"],
        np.asarray([[True, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["action_source"],
        np.asarray([["external_joint_action", "absent_noop", "external_joint_action"]], dtype=object),
    )
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["native_control_value"],
        np.asarray([[0, 0, 0]], dtype=np.int8),
    )
    _assert_public_final_metadata(batch, [], expected_reward=None)


def test_3p_public_active_round_leave_to_single_present_warmdown_matches_source_fixture():
    scenario_name = "source_lifecycle_remove_avatar_to_single_present_3p.json"
    env, reset_batch = _reset_3p_public_lifecycle_env(scenario_name)

    assert reset_batch.info["metadata_only"] is True
    assert reset_batch.info["trainer_observation_claim"] is False
    assert int(reset_batch.info["random_tape_cursor"][0]) == 12
    np.testing.assert_array_equal(
        reset_batch.info["present"],
        np.asarray([[True, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        reset_batch.info["alive"],
        np.asarray([[True, True, True]], dtype=bool),
    )

    env.state["pos"][0, 2] = np.asarray([93.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 2] = 0.0
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]

    first_death_batch = env.step(np.asarray([[1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(first_death_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        first_death_batch.info["present"],
        np.asarray([[True, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["alive"],
        np.asarray([[True, True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["death_count"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["death_player"],
        np.asarray([[2, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["score"],
        np.zeros((1, 3), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["round_score"],
        np.zeros((1, 3), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        first_death_batch.action_mask,
        np.asarray(
            [
                [
                    [True, True, True],
                    [True, True, True],
                    [False, False, False],
                ]
            ],
            dtype=bool,
        ),
    )
    assert int(first_death_batch.info["random_tape_cursor"][0]) == 13
    _assert_public_final_metadata(first_death_batch, [], expected_reward=None)

    env.state["pos"][0, 0] = np.asarray([18.991, 60.0], dtype=np.float64)
    env.state["heading"][0, 0] = 0.0
    env.state["prev_pos"][0, 0] = env.state["pos"][0, 0]
    env.state["speed"][0, 0] = 8.0

    leave_batch = env.remove_player(1)

    assert leave_batch.info["metadata_only"] is True
    assert leave_batch.info["trainer_observation_claim"] is False
    assert leave_batch.info["leave_metadata_only"] is True
    assert leave_batch.info["leave_trainer_claim"] is False
    assert leave_batch.info["leave_source_id_policy"] == (
        "source_avatar_id_is_public_player_id_plus_one/v0"
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_player_ids"],
        np.asarray([1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_source_player_ids"],
        np.asarray([2], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_immediate_terminal_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        leave_batch.info["present"],
        np.asarray([[True, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.info["alive"],
        np.asarray([[True, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.info["death_count"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        leave_batch.info["death_player"],
        np.asarray([[2, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        leave_batch.info["score"],
        np.asarray([[2, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        leave_batch.info["round_score"],
        np.zeros((1, 3), dtype=np.int32),
    )
    np.testing.assert_array_equal(leave_batch.info["winner"], np.asarray([0], dtype=np.int16))
    np.testing.assert_array_equal(leave_batch.info["draw"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(leave_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(leave_batch.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        leave_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        leave_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.info["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.info["needs_reset"],
        np.asarray([True], dtype=bool),
    )
    assert leave_batch.info["winner_ids"] == [[0]]
    assert leave_batch.info["loser_ids"] == [[2]]
    np.testing.assert_array_equal(
        leave_batch.reward,
        np.asarray([[1.0, 0.0, -1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        leave_batch.action_mask,
        np.zeros((1, 3, 3), dtype=bool),
    )
    assert int(leave_batch.info["random_tape_cursor"][0]) == 14
    _assert_public_final_metadata(leave_batch, [0], expected_reward=leave_batch.reward)

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["metadata_only"] is True
    assert warmdown_batch.info["trainer_observation_claim"] is False
    assert warmdown_batch.info["warmdown_waited"] is True
    assert warmdown_batch.info["warmdown_info"]["warmdown_end_fires"] == 1
    assert warmdown_batch.info["warmdown_info"]["game_stop_fires"] == 1
    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 0
    assert warmdown_batch.info["warmdown_info"]["round_clear_print_manager_stops"] == 1
    assert warmdown_batch.info["warmdown_info"]["random_tape_draws"] == 7
    np.testing.assert_array_equal(
        warmdown_batch.info["warmdown_info"]["round_clear_print_manager_stop_players"],
        np.asarray([0], dtype=np.int16),
    )
    assert [
        (call["tape_index"], call["player"])
        for call in warmdown_batch.info["warmdown_info"]["spawn_infos"][0]["random_calls"]
    ] == [(15, 2), (16, 2), (17, 2), (18, 0), (19, 0), (20, 0)]
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        warmdown_batch.info["terminal_rows"],
        np.asarray([], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["round_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["warmdown_pending"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["present"],
        np.asarray([[True, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["alive"],
        np.asarray([[True, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["score"],
        np.asarray([[2, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["round_score"],
        np.zeros((1, 3), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["death_count"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["death_player"],
        np.asarray([[1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(warmdown_batch.info["round_id"], np.asarray([2]))
    np.testing.assert_array_equal(warmdown_batch.info["source_round_id"], np.asarray([2]))
    assert float(warmdown_batch.info["map_size"][0]) == 88.0
    assert int(warmdown_batch.info["random_tape_cursor"][0]) == 21
    assert int(warmdown_batch.info["random_tape_draw_count"][0]) == 21
    np.testing.assert_array_equal(
        warmdown_batch.action_mask,
        np.asarray(
            [
                [
                    [True, True, True],
                    [False, False, False],
                    [True, True, True],
                ]
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(
        warmdown_batch.reward,
        np.zeros((1, 3), dtype=np.float32),
    )
    _assert_public_final_metadata(warmdown_batch, [], expected_reward=None)


def test_4p_public_active_round_leave_continues_then_scores_source_order():
    scenario_name = "source_lifecycle_mid_round_remove_avatar_4p_continue_round_end.json"
    env, reset_batch = _reset_4p_public_lifecycle_env(scenario_name)

    assert reset_batch.info["metadata_only"] is True
    assert reset_batch.info["trainer_observation_claim"] is False
    assert int(reset_batch.info["random_tape_cursor"][0]) == 16
    np.testing.assert_array_equal(
        reset_batch.info["present"],
        np.asarray([[True, True, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        reset_batch.info["alive"],
        np.asarray([[True, True, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        reset_batch.action_mask,
        np.ones((1, 4, 3), dtype=bool),
    )

    leave_batch = env.remove_player(1)

    assert leave_batch.info["metadata_only"] is True
    assert leave_batch.info["trainer_observation_claim"] is False
    assert leave_batch.info["leave_metadata_only"] is True
    assert leave_batch.info["leave_trainer_claim"] is False
    assert leave_batch.info["leave_source_id_policy"] == (
        "source_avatar_id_is_public_player_id_plus_one/v0"
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_player_ids"],
        np.asarray([1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_source_player_ids"],
        np.asarray([2], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        leave_batch.info["present"],
        np.asarray([[True, False, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.info["alive"],
        np.asarray([[True, False, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.info["death_count"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        leave_batch.info["death_player"],
        np.asarray([[-1, -1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        leave_batch.info["score"],
        np.asarray([[0, 0, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(leave_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        leave_batch.info["round_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.info["warmdown_pending"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.action_mask,
        np.asarray(
            [
                [
                    [True, True, True],
                    [False, False, False],
                    [True, True, True],
                    [True, True, True],
                ]
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(
        leave_batch.reward,
        np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    _assert_public_final_metadata(leave_batch, [], expected_reward=None)
    assert int(leave_batch.info["random_tape_cursor"][0]) == 17

    env.state["pos"][0, 3] = np.asarray([99.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 3] = 0.0
    env.state["prev_pos"][0, 3] = env.state["pos"][0, 3]
    first_death_batch = env.step(np.asarray([[1, -1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(first_death_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        first_death_batch.info["present"],
        np.asarray([[True, False, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["alive"],
        np.asarray([[True, False, True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["death_count"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["death_player"],
        np.asarray([[3, -1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["score"],
        np.asarray([[0, 0, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        first_death_batch.action_mask,
        np.asarray(
            [
                [
                    [True, True, True],
                    [False, False, False],
                    [True, True, True],
                    [False, False, False],
                ]
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(
        first_death_batch.reward,
        np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    assert int(first_death_batch.info["random_tape_cursor"][0]) == 18
    _assert_public_final_metadata(first_death_batch, [], expected_reward=None)

    env.state["pos"][0, 2] = np.asarray([1.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 2] = math.pi
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]
    terminal_batch = env.step(np.asarray([[1, -1, 1, -1]], dtype=np.int16))

    np.testing.assert_array_equal(
        terminal_batch.info["present"],
        np.asarray([[True, False, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["alive"],
        np.asarray([[True, False, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["death_count"],
        np.asarray([2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["death_player"],
        np.asarray([[3, 2, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["score"],
        np.asarray([[3, 0, 1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(terminal_batch.info["winner"], np.asarray([0], dtype=np.int16))
    np.testing.assert_array_equal(terminal_batch.info["draw"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(terminal_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal_batch.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    assert terminal_batch.info["winner_ids"] == [[0]]
    assert terminal_batch.info["loser_ids"] == [[2, 3]]
    np.testing.assert_array_equal(
        terminal_batch.info["action_sidecar"]["action_required"],
        np.asarray([[True, False, True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.reward,
        np.asarray([[1.0, 0.0, -1.0, -1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        terminal_batch.action_mask,
        np.zeros((1, 4, 3), dtype=bool),
    )
    assert int(terminal_batch.info["random_tape_cursor"][0]) == 19
    _assert_public_final_metadata(
        terminal_batch,
        [0],
        expected_reward=terminal_batch.reward,
    )


def test_3p_public_match_mode_warmdown_leave_next_round_matches_source_fixture_tape():
    env, reset_batch = _reset_3p_public_lifecycle_env(
        "source_lifecycle_remove_avatar_during_warmdown_3p.json",
        episode_end_mode="match",
    )

    assert int(reset_batch.info["max_score_by_row"][0]) == 10
    assert int(reset_batch.info["random_tape_cursor"][0]) == 12

    env.state["pos"][0, 2] = np.asarray([93.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 2] = 0.0
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]
    env.state["pos"][0, 1] = np.asarray([1.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 1] = math.pi
    env.state["prev_pos"][0, 1] = env.state["pos"][0, 1]

    round_batch = env.step(np.asarray([[1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(round_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        round_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["score"],
        np.asarray([[2, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        round_batch.info["death_player"],
        np.asarray([[2, 1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        round_batch.reward,
        np.asarray([[1.0, -1.0, -1.0]], dtype=np.float32),
    )
    assert round_batch.info["round_winner_ids"] == [[0]]
    assert int(round_batch.info["random_tape_cursor"][0]) == 14
    _assert_public_final_metadata(round_batch, [], expected_reward=None)

    leave_batch = env.remove_player(0)

    assert leave_batch.info["leave_metadata_only"] is True
    assert leave_batch.info["leave_trainer_claim"] is False
    assert leave_batch.info["warmdown_leave_score_policy"] == (
        "source_warmdown_leave_does_not_rescore_or_emit_round_end/v0"
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_warmdown_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_player_ids"],
        np.asarray([0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_source_player_ids"],
        np.asarray([1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        leave_batch.info["present"],
        np.asarray([[False, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.info["alive"],
        np.asarray([[False, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.info["score"],
        np.asarray([[2, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        leave_batch.info["death_player"],
        np.asarray([[2, 1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(leave_batch.info["death_count"], np.asarray([2]))
    np.testing.assert_array_equal(
        leave_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.info["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(leave_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        leave_batch.reward,
        np.zeros((1, 3), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        leave_batch.action_mask,
        np.zeros((1, 3, 3), dtype=bool),
    )
    assert int(leave_batch.info["random_tape_cursor"][0]) == 15
    _assert_public_final_metadata(leave_batch, [], expected_reward=None)

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["warmdown_waited"] is True
    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 0
    assert warmdown_batch.info["warmdown_info"]["random_tape_draws"] == 6
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        warmdown_batch.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["round_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["warmdown_pending"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["present"],
        np.asarray([[False, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["alive"],
        np.asarray([[False, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["score"],
        np.asarray([[2, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["death_player"],
        np.asarray([[0, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(warmdown_batch.info["death_count"], np.asarray([1]))
    np.testing.assert_array_equal(warmdown_batch.info["round_id"], np.asarray([2]))
    np.testing.assert_array_equal(warmdown_batch.info["source_round_id"], np.asarray([2]))
    assert float(warmdown_batch.info["map_size"][0]) == 88.0
    assert int(warmdown_batch.info["random_tape_cursor"][0]) == 21
    np.testing.assert_array_equal(
        warmdown_batch.action_mask,
        np.asarray(
            [
                [
                    [False, False, False],
                    [True, True, True],
                    [True, True, True],
                ]
            ],
            dtype=bool,
        ),
    )
    _assert_public_final_metadata(warmdown_batch, [], expected_reward=None)


def test_remove_player_rejects_bad_row_and_player_shapes():
    env = VectorMultiplayerEnv(batch_size=2, player_count=3, timer_capacity=3)
    env.reset(seed=np.asarray([1, 2], dtype=np.uint64))

    with pytest.raises(VectorMultiplayerEnvError, match="row_mask"):
        env.remove_player(1, row_mask=np.asarray([True], dtype=bool))
    with pytest.raises(VectorMultiplayerEnvError, match="player_ids"):
        env.remove_player(np.asarray([[1, 1]], dtype=np.int16))
    with pytest.raises(VectorMultiplayerEnvError, match="zero-based public player ids"):
        env.remove_player(np.asarray([1, 99], dtype=np.int16))


def test_remove_player_rejects_before_reset_absent_dead_bad_warmdown_and_terminal():
    env = VectorMultiplayerEnv(batch_size=1, player_count=3, timer_capacity=3)
    with pytest.raises(RuntimeError, match="before remove_player"):
        env.remove_player(1)

    absent_env = VectorMultiplayerEnv(batch_size=1, player_count=3, timer_capacity=3)
    absent_env.reset(present=np.asarray([[True, False, True]], dtype=bool))
    with pytest.raises(VectorMultiplayerEnvError, match="must be present"):
        absent_env.remove_player(1)

    dead_env = VectorMultiplayerEnv(batch_size=1, player_count=3, timer_capacity=3)
    dead_env.reset()
    dead_env.state["alive"][0, 1] = False
    with pytest.raises(VectorMultiplayerEnvError, match="must be alive"):
        dead_env.remove_player(1)

    warmdown_env = VectorMultiplayerEnv(batch_size=1, player_count=3, timer_capacity=3)
    warmdown_env.reset()
    warmdown_env.state["warmdown_pending"][0] = True
    warmdown_env.state["round_done"][0] = True
    with pytest.raises(RuntimeError, match="warmdown rows"):
        warmdown_env.remove_player(1)

    terminal_env = VectorMultiplayerEnv(batch_size=1, player_count=3, timer_capacity=3)
    terminal_env.reset()
    terminal_env.state["done"][0] = True
    with pytest.raises(RuntimeError, match="terminal rows"):
        terminal_env.remove_player(1)


def test_2p_public_active_round_leave_immediately_scores_survivor_like_source():
    scenario_name = "source_lifecycle_mid_round_remove_avatar_2p.json"
    tape = _lifecycle_random_tape(scenario_name)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        seed=555,
        decision_ms=100.0,
        max_score=_lifecycle_max_score(scenario_name),
        event_capacity=32,
        timer_capacity=3,
        random_tape_capacity=tape.shape[1],
    )
    env.reset(
        seed=np.asarray([555], dtype=np.uint64),
        source_fixture_random_tape_values=tape,
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )

    batch = env.remove_player(1)

    assert batch.info["metadata_only"] is True
    assert batch.info["trainer_observation_claim"] is False
    assert batch.info["leave_metadata_only"] is True
    assert batch.info["leave_trainer_claim"] is False
    np.testing.assert_array_equal(
        batch.info["leave_immediate_terminal_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.info["present"],
        np.asarray([[True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["alive"],
        np.asarray([[True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["death_count"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.info["death_player"],
        np.asarray([[-1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        batch.info["score"],
        np.asarray([[1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(batch.info["winner"], np.asarray([0], dtype=np.int16))
    np.testing.assert_array_equal(batch.info["draw"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(batch.info["round_done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        batch.info["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        batch.reward,
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        batch.action_mask,
        np.zeros((1, 2, 3), dtype=bool),
    )
    _assert_public_final_metadata(batch, [0], expected_reward=batch.reward)


def test_2p_public_draw_warmdown_starts_next_round_like_source_fixture():
    env, reset_batch = _reset_2p_public_lifecycle_env(
        "source_lifecycle_spawn_rng_2p_next_round.json",
        episode_end_mode="match",
    )

    assert reset_batch.info["metadata_only"] is True
    assert int(reset_batch.info["max_score_by_row"][0]) == 10
    assert int(reset_batch.info["random_tape_cursor"][0]) == 8
    np.testing.assert_allclose(
        env.state["pos"][0],
        [[58.0, 44.0], [30.0, 44.0]],
    )
    np.testing.assert_allclose(
        env.state["heading"][0],
        [3.241592653589793, 0.1],
    )
    np.testing.assert_array_equal(
        reset_batch.info["warmup_info"]["print_manager_start_players"],
        np.asarray([1, 0], dtype=np.int16),
    )

    _force_2p_wall_draw(env)
    terminal_batch = env.step(np.asarray([[1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(terminal_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        terminal_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["score"],
        np.asarray([[0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(terminal_batch.info["winner"], np.asarray([-1], dtype=np.int16))
    np.testing.assert_array_equal(terminal_batch.info["draw"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        terminal_batch.info["death_player"],
        np.asarray([[1, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW], dtype=np.int16),
    )
    assert int(terminal_batch.info["random_tape_cursor"][0]) == 10
    _assert_public_final_metadata(terminal_batch, [], expected_reward=None)

    with pytest.raises(RuntimeError, match="advance_warmdown must be called"):
        env.step(np.asarray([[1, 1]], dtype=np.int16))

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["warmdown_waited"] is True
    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 0
    assert warmdown_batch.info["warmdown_info"]["scheduled_timer_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["round_clear_print_manager_stops"] == 0
    assert warmdown_batch.info["warmdown_info"]["random_tape_draws"] == 6
    assert [
        (call["tape_index"], call["player"])
        for call in warmdown_batch.info["warmdown_info"]["spawn_infos"][0]["random_calls"]
    ] == [(10, 1), (11, 1), (12, 1), (13, 0), (14, 0), (15, 0)]
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        warmdown_batch.info["round_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["warmdown_pending"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["alive"],
        np.asarray([[True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["score"],
        np.asarray([[0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["death_player"],
        np.asarray([[-1, -1]], dtype=np.int16),
    )
    assert int(warmdown_batch.info["round_id"][0]) == 2
    np.testing.assert_array_equal(warmdown_batch.info["source_round_id"], np.asarray([2]))
    assert int(warmdown_batch.info["random_tape_cursor"][0]) == 16
    np.testing.assert_array_equal(
        warmdown_batch.info["warmdown_info"]["next_round_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_allclose(
        env.state["pos"][0],
        [[51.8, 44.0], [36.2, 44.0]],
    )
    np.testing.assert_allclose(
        env.state["heading"][0],
        [math.tau * 0.75, math.pi * 0.5],
    )
    np.testing.assert_array_equal(warmdown_batch.action_mask, np.ones((1, 2, 3), dtype=bool))
    np.testing.assert_array_equal(warmdown_batch.reward, np.zeros((1, 2), dtype=np.float32))
    _assert_public_final_metadata(warmdown_batch, [], expected_reward=None)


def test_2p_public_unique_max_score_leader_warmdown_reports_match_end_metadata():
    env, reset_batch = _reset_2p_public_lifecycle_env(
        "source_lifecycle_match_end_at_max_score_2p.json",
        episode_end_mode="round",
    )

    assert reset_batch.info["metadata_only"] is True
    assert int(reset_batch.info["max_score_by_row"][0]) == 1
    assert int(reset_batch.info["random_tape_cursor"][0]) == 8

    _force_2p_player0_round_win(env)
    terminal_batch = env.step(np.asarray([[1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(terminal_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        terminal_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["score"],
        np.asarray([[1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(terminal_batch.info["winner"], np.asarray([0], dtype=np.int16))
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    assert terminal_batch.info["winner_ids"] == [[0]]
    assert int(terminal_batch.info["random_tape_cursor"][0]) == 9
    np.testing.assert_array_equal(
        terminal_batch.info["round_winner"],
        np.asarray([0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["match_winner"],
        np.asarray([-1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.reward,
        np.asarray([[1.0, -1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        terminal_batch.action_mask,
        np.zeros((1, 2, 3), dtype=bool),
    )
    _assert_public_final_metadata(terminal_batch, [0], expected_reward=terminal_batch.reward)

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["warmdown_waited"] is True
    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 0
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["scheduled_timer_count"] == 0
    assert warmdown_batch.info["warmdown_info"]["random_tape_draws"] == 0
    np.testing.assert_array_equal(
        warmdown_batch.info["next_round_rows"],
        np.asarray([], dtype=np.int32),
    )
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        warmdown_batch.info["terminal_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["needs_reset"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["warmdown_pending"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_winner"],
        np.asarray([0], dtype=np.int16),
    )
    assert warmdown_batch.info["round_winner_ids"] == [[0]]
    assert warmdown_batch.info["match_winner_ids"] == [[0]]
    np.testing.assert_array_equal(
        warmdown_batch.info["score"],
        np.asarray([[1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["alive"],
        np.asarray([[True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(warmdown_batch.info["round_id"], np.asarray([1]))
    np.testing.assert_array_equal(warmdown_batch.info["source_round_id"], np.asarray([1]))
    np.testing.assert_array_equal(
        warmdown_batch.action_mask,
        np.zeros((1, 2, 3), dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.reward,
        np.zeros((1, 2), dtype=np.float32),
    )
    assert int(warmdown_batch.info["random_tape_cursor"][0]) == 9
    _assert_public_final_metadata(
        warmdown_batch,
        [0],
        expected_reward=np.zeros((1, 2), dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match="reset must be called before stepping rows"):
        env.step(np.asarray([[1, 1]], dtype=np.int16))


def test_4p_public_active_round_leave_can_immediately_score_remaining_survivor():
    env = VectorMultiplayerEnv(batch_size=1, player_count=4, timer_capacity=3)
    env.reset(seed=np.asarray([1], dtype=np.uint64))
    env.state["alive"][0, 2:] = False
    env.state["death_count"][0] = 2
    env.state["death_player"][0, :2] = np.asarray([3, 2], dtype=np.int16)

    batch = env.remove_player(1)

    np.testing.assert_array_equal(
        batch.info["present"],
        np.asarray([[True, False, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["alive"],
        np.asarray([[True, False, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["death_player"],
        np.asarray([[3, 2, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        batch.info["score"],
        np.asarray([[3, 0, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(batch.info["winner"], np.asarray([0], dtype=np.int16))
    np.testing.assert_array_equal(batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        batch.reward,
        np.asarray([[1.0, 0.0, -1.0, -1.0]], dtype=np.float32),
    )
    _assert_public_final_metadata(batch, [0], expected_reward=batch.reward)


def test_3p_public_present_absent_draw_warmdown_next_round_matches_source_fixture_tape():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=3,
        seed=555,
        decision_ms=100.0,
        body_capacity=16,
        event_capacity=16,
        timer_capacity=3,
        random_tape_capacity=17,
    )
    reset_batch = env.reset(
        seed=np.asarray([555], dtype=np.uint64),
        present=np.asarray([[True, False, True]], dtype=bool),
        source_fixture_random_tape_values=_lifecycle_random_tape(
            "source_lifecycle_present_absent_3p_next_round.json"
        ),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )

    assert reset_batch.info["metadata_only"] is True
    assert reset_batch.info["trainer_observation_claim"] is False
    np.testing.assert_array_equal(
        reset_batch.info["present"],
        np.asarray([[True, False, True]], dtype=bool),
    )
    assert float(reset_batch.info["map_size"][0]) == 95.0
    assert int(reset_batch.info["random_tape_cursor"][0]) == 9
    assert int(reset_batch.info["random_tape_draw_count"][0]) == 9
    np.testing.assert_array_equal(
        reset_batch.info["warmup_info"]["print_manager_start_players"],
        np.asarray([2, 1, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        env.state["printing"],
        np.asarray([[True, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        env.state["print_manager_active"],
        np.asarray([[True, True, True]], dtype=bool),
    )

    env.state["pos"][0, 2] = np.asarray([93.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 2] = 0.0
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]
    env.state["pos"][0, 0] = np.asarray([1.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 0] = np.pi
    env.state["prev_pos"][0, 0] = env.state["pos"][0, 0]

    terminal_batch = env.step(np.asarray([[1, -1, 1]], dtype=np.int16))

    assert terminal_batch.info["metadata_only"] is True
    assert terminal_batch.info["trainer_observation_claim"] is False
    np.testing.assert_array_equal(
        terminal_batch.info["score"],
        np.asarray([[1, 0, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["death_player"],
        np.asarray([[1, 2, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["death_count"],
        np.asarray([3], dtype=np.int32),
    )
    np.testing.assert_array_equal(terminal_batch.info["winner"], np.asarray([-1], dtype=np.int16))
    np.testing.assert_array_equal(terminal_batch.info["draw"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW], dtype=np.int16),
    )
    assert int(terminal_batch.info["random_tape_cursor"][0]) == 11
    assert int(terminal_batch.info["random_tape_draw_count"][0]) == 11
    np.testing.assert_array_equal(
        terminal_batch.info["needs_reset"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.reward,
        np.zeros((1, 3), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        terminal_batch.action_mask,
        np.zeros((1, 3, 3), dtype=bool),
    )
    _assert_public_final_metadata(
        terminal_batch,
        [0],
        expected_reward=terminal_batch.reward,
    )

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["metadata_only"] is True
    assert warmdown_batch.info["trainer_observation_claim"] is False
    assert warmdown_batch.info["warmdown_waited"] is True
    assert warmdown_batch.info["warmdown_info"]["warmdown_end_fires"] == 1
    assert warmdown_batch.info["warmdown_info"]["game_stop_fires"] == 1
    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 0
    assert warmdown_batch.info["warmdown_info"]["round_clear_print_manager_stops"] == 0
    assert warmdown_batch.info["warmdown_info"]["random_tape_draws"] == 6
    np.testing.assert_array_equal(
        warmdown_batch.info["terminal_rows"],
        np.asarray([], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.reward,
        np.zeros((1, 3), dtype=np.float32),
    )
    _assert_public_final_metadata(warmdown_batch, [], expected_reward=None)
    assert [
        (call["tape_index"], call["player"])
        for call in warmdown_batch.info["warmdown_info"]["spawn_infos"][0]["random_calls"]
    ] == [(11, 2), (12, 2), (13, 2), (14, 0), (15, 0), (16, 0)]
    assert float(warmdown_batch.info["map_size"][0]) == 88.0
    np.testing.assert_array_equal(
        warmdown_batch.info["alive"],
        np.asarray([[True, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["score"],
        np.asarray([[1, 0, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["death_player"],
        np.asarray([[1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        warmdown_batch.action_mask,
        np.asarray(
            [
                [
                    [True, True, True],
                    [False, False, False],
                    [True, True, True],
                ]
            ],
            dtype=bool,
        ),
    )
    assert int(warmdown_batch.info["random_tape_cursor"][0]) == 17
    assert int(warmdown_batch.info["random_tape_draw_count"][0]) == 17
    np.testing.assert_array_equal(
        env.state["printing"],
        np.asarray([[False, True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        env.state["print_manager_active"],
        np.asarray([[False, True, False]], dtype=bool),
    )


def test_4p_public_present_absent_draw_warmdown_next_round_matches_source_fixture_tape():
    scenario_name = "source_lifecycle_present_absent_4p_next_round.json"
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        seed=555,
        decision_ms=100.0,
        body_capacity=24,
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=25,
    )
    reset_batch = env.reset(
        seed=np.asarray([555], dtype=np.uint64),
        present=np.asarray([[True, False, True, True]], dtype=bool),
        source_fixture_random_tape_values=_lifecycle_random_tape(scenario_name),
        source_fixture_ref=f"scenarios/environment/{scenario_name}",
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )

    assert reset_batch.info["metadata_only"] is True
    assert reset_batch.info["trainer_observation_claim"] is False
    np.testing.assert_array_equal(
        reset_batch.info["present"],
        np.asarray([[True, False, True, True]], dtype=bool),
    )
    assert float(reset_batch.info["map_size"][0]) == 101.0
    assert int(reset_batch.info["random_tape_cursor"][0]) == 13
    assert int(reset_batch.info["random_tape_draw_count"][0]) == 13
    np.testing.assert_array_equal(
        reset_batch.info["warmup_info"]["print_manager_start_players"],
        np.asarray([3, 2, 1, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        env.state["printing"],
        np.asarray([[True, True, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        env.state["print_manager_active"],
        np.asarray([[True, True, True, True]], dtype=bool),
    )

    env.state["pos"][0, 3] = np.asarray([99.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 3] = 0.0
    env.state["prev_pos"][0, 3] = env.state["pos"][0, 3]
    env.state["pos"][0, 2] = np.asarray([1.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 2] = np.pi
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]
    env.state["pos"][0, 0] = np.asarray([50.5, 1.0], dtype=np.float64)
    env.state["heading"][0, 0] = np.pi * 1.5
    env.state["prev_pos"][0, 0] = env.state["pos"][0, 0]

    terminal_batch = env.step(np.asarray([[1, -1, 1, 1]], dtype=np.int16))

    assert terminal_batch.info["metadata_only"] is True
    assert terminal_batch.info["trainer_observation_claim"] is False
    np.testing.assert_array_equal(
        terminal_batch.info["score"],
        np.asarray([[1, 0, 1, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["death_player"],
        np.asarray([[1, 3, 2, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["death_count"],
        np.asarray([4], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["winner"],
        np.asarray([-1], dtype=np.int16),
    )
    np.testing.assert_array_equal(terminal_batch.info["draw"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW], dtype=np.int16),
    )
    assert int(terminal_batch.info["random_tape_cursor"][0]) == 16
    assert int(terminal_batch.info["random_tape_draw_count"][0]) == 16
    np.testing.assert_array_equal(
        terminal_batch.info["needs_reset"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.reward,
        np.zeros((1, 4), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        terminal_batch.action_mask,
        np.zeros((1, 4, 3), dtype=bool),
    )
    _assert_public_final_metadata(
        terminal_batch,
        [0],
        expected_reward=terminal_batch.reward,
    )

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["metadata_only"] is True
    assert warmdown_batch.info["trainer_observation_claim"] is False
    assert warmdown_batch.info["warmdown_waited"] is True
    assert warmdown_batch.info["warmdown_info"]["warmdown_end_fires"] == 1
    assert warmdown_batch.info["warmdown_info"]["game_stop_fires"] == 1
    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 0
    assert warmdown_batch.info["warmdown_info"]["round_clear_print_manager_stops"] == 0
    assert warmdown_batch.info["warmdown_info"]["random_tape_draws"] == 9
    np.testing.assert_array_equal(
        warmdown_batch.info["terminal_rows"],
        np.asarray([], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.reward,
        np.zeros((1, 4), dtype=np.float32),
    )
    _assert_public_final_metadata(warmdown_batch, [], expected_reward=None)
    assert [
        (call["tape_index"], call["player"])
        for call in warmdown_batch.info["warmdown_info"]["spawn_infos"][0]["random_calls"]
    ] == [(16, 3), (17, 3), (18, 3), (19, 2), (20, 2), (21, 2), (22, 0), (23, 0), (24, 0)]
    assert float(warmdown_batch.info["map_size"][0]) == 95.0
    np.testing.assert_array_equal(
        warmdown_batch.info["alive"],
        np.asarray([[True, False, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["score"],
        np.asarray([[1, 0, 1, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["death_player"],
        np.asarray([[1, -1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        warmdown_batch.action_mask,
        np.asarray(
            [
                [
                    [True, True, True],
                    [False, False, False],
                    [True, True, True],
                    [True, True, True],
                ]
            ],
            dtype=bool,
        ),
    )
    assert int(warmdown_batch.info["random_tape_cursor"][0]) == 25
    assert int(warmdown_batch.info["random_tape_draw_count"][0]) == 25
    np.testing.assert_allclose(
        env.state["pos"][0, [0, 2, 3]],
        [[72.79, 47.5], [39.07, 47.5], [22.21, 47.5]],
    )
    np.testing.assert_allclose(
        env.state["heading"][0, [0, 2, 3]],
        [np.pi * 1.5, 0.1, np.pi * 0.5],
    )
    np.testing.assert_array_equal(
        env.state["printing"],
        np.asarray([[False, True, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        env.state["print_manager_active"],
        np.asarray([[False, True, False, False]], dtype=bool),
    )


def test_3p_public_unique_max_score_leader_warmdown_reports_match_end_metadata():
    env, reset_batch = _reset_3p_public_lifecycle_env(
        "source_lifecycle_match_end_at_max_score_3p.json"
    )

    assert reset_batch.info["metadata_only"] is True
    assert int(reset_batch.info["max_score_by_row"][0]) == 2
    assert int(reset_batch.info["random_tape_cursor"][0]) == 12

    env.state["pos"][0] = np.asarray(
        [[5.0, 5.0], [1.0, 47.5], [93.0, 47.5]],
        dtype=np.float64,
    )
    env.state["heading"][0] = np.asarray(
        [math.pi / 4.0, math.pi, 0.0],
        dtype=np.float64,
    )
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["speed"][0, 0] = 8.0
    env.state["print_manager_distance"][0, 0] = 999.0
    env.state["print_manager_last_pos"][0, 0] = env.state["pos"][0, 0]

    terminal_batch = env.step(np.asarray([[1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(terminal_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        terminal_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["score"],
        np.asarray([[2, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    assert terminal_batch.info["winner_ids"] == [[0]]
    assert int(terminal_batch.info["random_tape_cursor"][0]) == 14
    np.testing.assert_array_equal(
        terminal_batch.info["needs_reset"],
        np.asarray([True], dtype=bool),
    )
    _assert_public_final_metadata(
        terminal_batch,
        [0],
        expected_reward=terminal_batch.reward,
    )

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["warmdown_waited"] is True
    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 0
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 1
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        warmdown_batch.info["terminal_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["needs_reset"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_winner"],
        np.asarray([0], dtype=np.int16),
    )
    assert warmdown_batch.info["round_winner_ids"] == [[0]]
    assert warmdown_batch.info["match_winner_ids"] == [[0]]
    np.testing.assert_array_equal(
        warmdown_batch.action_mask,
        np.zeros((1, 3, 3), dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.reward,
        np.zeros((1, 3), dtype=np.float32),
    )
    _assert_public_final_metadata(
        warmdown_batch,
        [0],
        expected_reward=np.zeros((1, 3), dtype=np.float32),
    )
    assert int(warmdown_batch.info["random_tape_cursor"][0]) == 14

    with pytest.raises(RuntimeError, match="reset must be called before stepping rows"):
        env.step(np.asarray([[1, 1, 1]], dtype=np.int16))


def test_3p_public_episode_end_mode_round_vs_match_round_continuation_metadata():
    round_env, _ = _reset_3p_public_lifecycle_env(
        "source_lifecycle_multi_round_match_end_3p.json",
        episode_end_mode="round",
    )
    match_env, _ = _reset_3p_public_lifecycle_env(
        "source_lifecycle_multi_round_match_end_3p.json",
        episode_end_mode="match",
    )

    for env in (round_env, match_env):
        _force_3p_player0_round_win(env)

    round_batch = round_env.step(np.asarray([[1, 1, 1]], dtype=np.int16))
    match_round_batch = match_env.step(np.asarray([[1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(round_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        round_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["needs_reset"],
        np.asarray([True], dtype=bool),
    )
    _assert_public_final_metadata(round_batch, [0], expected_reward=round_batch.reward)

    np.testing.assert_array_equal(match_round_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        match_round_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        match_round_batch.info["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        match_round_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        match_round_batch.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(match_round_batch.action_mask, np.zeros((1, 3, 3), dtype=bool))
    np.testing.assert_array_equal(
        match_round_batch.reward,
        np.asarray([[1.0, -1.0, -1.0]], dtype=np.float32),
    )
    _assert_public_final_metadata(match_round_batch, [], expected_reward=None)

    with pytest.raises(RuntimeError, match="advance_warmdown must be called"):
        match_env.step(np.asarray([[1, 1, 1]], dtype=np.int16))

    warmdown_batch = match_env.advance_warmdown(5000.0)

    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 0
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        warmdown_batch.info["round_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["warmdown_pending"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(warmdown_batch.action_mask, np.ones((1, 3, 3), dtype=bool))
    np.testing.assert_array_equal(warmdown_batch.reward, np.zeros((1, 3), dtype=np.float32))
    _assert_public_final_metadata(warmdown_batch, [], expected_reward=None)


def test_3p_public_match_mode_round_done_masks_actions_but_pays_round_reward():
    env, _ = _reset_3p_public_lifecycle_env(
        "source_lifecycle_multi_round_match_end_3p.json",
        episode_end_mode="match",
    )
    _force_3p_player0_round_win(env)

    batch = env.step(np.asarray([[1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(batch.truncated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(batch.reward, np.asarray([[1.0, -1.0, -1.0]], dtype=np.float32))
    np.testing.assert_array_equal(batch.action_mask, np.zeros((1, 3, 3), dtype=bool))
    _assert_public_final_metadata(batch, [], expected_reward=None)


def test_3p_public_match_mode_explicit_warmdown_frame_moves_survivor_and_does_not_rescore():
    env, reset_batch = _reset_3p_public_lifecycle_env(
        "source_lifecycle_survivor_score_3p_next_round.json",
        episode_end_mode="match",
    )

    assert reset_batch.info["metadata_only"] is True
    assert int(reset_batch.info["random_tape_cursor"][0]) == 12

    env.state["pos"][0] = np.asarray(
        [[20.591, 47.5], [1.0, 47.5], [93.0, 47.5]],
        dtype=np.float64,
    )
    env.state["heading"][0] = np.asarray([math.pi, math.pi, 0.0], dtype=np.float64)
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["print_manager_last_pos"][0, 0] = env.state["pos"][0, 0]

    round_batch = env.step(np.asarray([[1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(round_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        round_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["alive"],
        np.asarray([[True, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["score"],
        np.asarray([[2, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        round_batch.info["round_score"],
        np.zeros((1, 3), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        round_batch.info["death_player"],
        np.asarray([[2, 1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        round_batch.info["death_count"],
        np.asarray([2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        round_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        round_batch.action_mask,
        np.zeros((1, 3, 3), dtype=bool),
    )
    np.testing.assert_allclose(env.state["pos"][0, 0], [18.991, 47.5])
    np.testing.assert_allclose(env.state["timer_remaining_ms"][0, 0], 5000.0)
    assert int(round_batch.info["random_tape_cursor"][0]) == 14

    with pytest.raises(RuntimeError, match="advance_warmdown must be called"):
        env.step(np.asarray([[1, -1, -1]], dtype=np.int16))

    warmdown_frame = env.advance_warmdown_frame(
        np.asarray([[1, -1, -1]], dtype=np.int16),
        elapsed_ms=1150.0,
    )

    assert warmdown_frame.info["metadata_only"] is True
    assert warmdown_frame.info["trainer_observation_claim"] is False
    assert warmdown_frame.info["warmdown_frame_policy"] == (
        "explicit_metadata_only_does_not_relax_public_step_barrier/v0"
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["warmdown_frame_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_allclose(warmdown_frame.info["warmdown_frame_elapsed_ms"], [1150.0])
    assert warmdown_frame.info["warmdown_frame_step_index_incremented"] is False
    assert warmdown_frame.info["warmdown_frame_counters"]["normal_wall_deaths"] == 1
    assert warmdown_frame.info["warmdown_frame_counters"]["terminal_score_rows"] == 0
    assert warmdown_frame.info["warmdown_frame_counters"]["print_manager_death_stops"] == 1
    np.testing.assert_array_equal(warmdown_frame.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        warmdown_frame.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["alive"],
        np.asarray([[False, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["score"],
        np.asarray([[2, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["round_score"],
        np.asarray([[2, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["death_player"],
        np.asarray([[2, 1, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["death_count"],
        np.asarray([3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_frame.reward,
        np.zeros((1, 3), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        warmdown_frame.action_mask,
        np.zeros((1, 3, 3), dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["action_sidecar"]["action_required"],
        np.asarray([[True, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["action_sidecar"]["native_control_value"],
        np.asarray([[0, 0, 0]], dtype=np.int8),
    )
    np.testing.assert_allclose(env.state["pos"][0, 0], [0.591, 47.5])
    np.testing.assert_allclose(env.state["timer_remaining_ms"][0, 0], 3850.0)
    assert int(warmdown_frame.info["random_tape_cursor"][0]) == 15

    next_round_batch = env.advance_warmdown(3850.0)

    assert next_round_batch.info["warmdown_waited"] is True
    assert next_round_batch.info["warmdown_info"]["next_round_count"] == 1
    assert next_round_batch.info["warmdown_info"]["match_end_count"] == 0
    assert next_round_batch.info["warmdown_info"]["random_tape_draws"] == 9
    np.testing.assert_array_equal(next_round_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        next_round_batch.info["alive"],
        np.asarray([[True, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        next_round_batch.info["score"],
        np.asarray([[2, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        next_round_batch.info["death_player"],
        np.asarray([[-1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        next_round_batch.info["round_id"],
        np.asarray([2]),
    )
    assert int(next_round_batch.info["random_tape_cursor"][0]) == 24


def test_4p_public_match_mode_explicit_warmdown_frame_moves_survivor_and_does_not_rescore():
    scenario_name = "source_lifecycle_survivor_score_4p_next_round.json"
    source_fixture_ref = f"scenarios/environment/{scenario_name}"
    env, reset_batch = _reset_4p_public_lifecycle_env(
        scenario_name,
        episode_end_mode="match",
        source_fixture_ref=source_fixture_ref,
    )

    assert reset_batch.info["metadata_only"] is True
    assert reset_batch.info["trainer_observation_claim"] is False
    np.testing.assert_array_equal(
        reset_batch.info["source_fixture_ref"],
        np.asarray([source_fixture_ref], dtype=object),
    )
    np.testing.assert_array_equal(reset_batch.info["round_id"], np.asarray([1]))
    assert int(reset_batch.info["random_tape_cursor"][0]) == 16

    env.state["pos"][0, 0] = np.asarray([87.6, 10.0], dtype=np.float64)
    env.state["heading"][0, 0] = 0.0
    env.state["prev_pos"][0, 0] = env.state["pos"][0, 0]
    env.state["speed"][0, 0] = 8.0
    env.state["print_manager_last_pos"][0, 0] = env.state["pos"][0, 0]
    env.state["print_manager_distance"][0, 0] = 999.0

    env.state["pos"][0, 3] = np.asarray([99.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 3] = 0.0
    env.state["prev_pos"][0, 3] = env.state["pos"][0, 3]

    first_death_batch = env.step(np.asarray([[1, 1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(first_death_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        first_death_batch.info["round_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["warmdown_pending"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["alive"],
        np.asarray([[True, True, True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["score"],
        np.zeros((1, 4), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["round_score"],
        np.zeros((1, 4), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["death_player"],
        np.asarray([[3, -1, -1, -1]], dtype=np.int16),
    )
    assert int(first_death_batch.info["random_tape_cursor"][0]) == 17

    env.state["pos"][0, 2] = np.asarray([1.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 2] = math.pi
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]

    second_death_batch = env.step(np.asarray([[1, 1, 1, -1]], dtype=np.int16))

    np.testing.assert_array_equal(second_death_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        second_death_batch.info["alive"],
        np.asarray([[True, True, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        second_death_batch.info["score"],
        np.zeros((1, 4), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        second_death_batch.info["round_score"],
        np.asarray([[0, 0, 1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        second_death_batch.info["death_player"],
        np.asarray([[3, 2, -1, -1]], dtype=np.int16),
    )
    assert int(second_death_batch.info["random_tape_cursor"][0]) == 18

    env.state["pos"][0, 1] = np.asarray([50.5, 1.0], dtype=np.float64)
    env.state["heading"][0, 1] = math.tau * 0.75
    env.state["prev_pos"][0, 1] = env.state["pos"][0, 1]

    round_batch = env.step(np.asarray([[1, 1, -1, -1]], dtype=np.int16))

    assert round_batch.info["metadata_only"] is True
    assert round_batch.info["trainer_observation_claim"] is False
    np.testing.assert_array_equal(round_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        round_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["alive"],
        np.asarray([[True, False, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["score"],
        np.asarray([[3, 2, 1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        round_batch.info["round_score"],
        np.zeros((1, 4), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        round_batch.info["death_player"],
        np.asarray([[3, 2, 1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        round_batch.info["death_count"],
        np.asarray([3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        round_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    assert round_batch.info["round_winner_ids"] == [[0]]
    assert round_batch.info["match_winner_ids"] == [[]]
    np.testing.assert_array_equal(
        round_batch.reward,
        np.asarray([[1.0, -1.0, -1.0, -1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        round_batch.action_mask,
        np.zeros((1, 4, 3), dtype=bool),
    )
    np.testing.assert_allclose(env.state["pos"][0, 0], [90.0, 10.0])
    np.testing.assert_allclose(env.state["timer_remaining_ms"][0, 0], 5000.0)
    assert int(round_batch.info["random_tape_cursor"][0]) == 19
    _assert_public_final_metadata(round_batch, [], expected_reward=None)

    with pytest.raises(RuntimeError, match="advance_warmdown must be called"):
        env.step(np.asarray([[1, -1, -1, -1]], dtype=np.int16))

    warmdown_frame = env.advance_warmdown_frame(
        np.asarray([[1, -1, -1, -1]], dtype=np.int16),
        elapsed_ms=1400.0,
    )

    assert warmdown_frame.info["metadata_only"] is True
    assert warmdown_frame.info["trainer_observation_claim"] is False
    assert warmdown_frame.info["warmdown_frame_policy"] == (
        "explicit_metadata_only_does_not_relax_public_step_barrier/v0"
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["warmdown_frame_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_allclose(warmdown_frame.info["warmdown_frame_elapsed_ms"], [1400.0])
    assert warmdown_frame.info["warmdown_frame_step_index_incremented"] is False
    assert warmdown_frame.info["warmdown_frame_counters"]["normal_wall_deaths"] == 1
    assert warmdown_frame.info["warmdown_frame_counters"]["body_hits"] == 0
    assert warmdown_frame.info["warmdown_frame_counters"]["terminal_score_rows"] == 0
    assert warmdown_frame.info["warmdown_frame_counters"]["print_manager_death_stops"] == 1
    np.testing.assert_array_equal(warmdown_frame.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        warmdown_frame.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["alive"],
        np.zeros((1, 4), dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["score"],
        np.asarray([[3, 2, 1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["round_score"],
        np.asarray([[3, 0, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["death_player"],
        np.asarray([[3, 2, 1, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["death_count"],
        np.asarray([4], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_frame.reward,
        np.zeros((1, 4), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        warmdown_frame.action_mask,
        np.zeros((1, 4, 3), dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["action_sidecar"]["action_required"],
        np.asarray([[True, False, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_frame.info["action_sidecar"]["native_control_value"],
        np.asarray([[0, 0, 0, 0]], dtype=np.int8),
    )
    np.testing.assert_allclose(env.state["pos"][0, 0], [101.2, 10.0])
    np.testing.assert_allclose(env.state["timer_remaining_ms"][0, 0], 3600.0)
    assert int(warmdown_frame.info["random_tape_cursor"][0]) == 20
    _assert_public_final_metadata(warmdown_frame, [], expected_reward=None)

    next_round_batch = env.advance_warmdown(3600.0)

    assert next_round_batch.info["metadata_only"] is True
    assert next_round_batch.info["trainer_observation_claim"] is False
    assert next_round_batch.info["warmdown_waited"] is True
    assert next_round_batch.info["warmdown_info"]["warmdown_end_fires"] == 1
    assert next_round_batch.info["warmdown_info"]["game_stop_fires"] == 1
    assert next_round_batch.info["warmdown_info"]["next_round_count"] == 1
    assert next_round_batch.info["warmdown_info"]["match_end_count"] == 0
    assert next_round_batch.info["warmdown_info"]["random_tape_draws"] == 12
    np.testing.assert_array_equal(
        next_round_batch.info["terminal_rows"],
        np.asarray([], dtype=np.int32),
    )
    np.testing.assert_array_equal(next_round_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        next_round_batch.info["round_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        next_round_batch.info["warmdown_pending"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        next_round_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        next_round_batch.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        next_round_batch.info["alive"],
        np.ones((1, 4), dtype=bool),
    )
    np.testing.assert_array_equal(
        next_round_batch.info["score"],
        np.asarray([[3, 2, 1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        next_round_batch.info["round_score"],
        np.zeros((1, 4), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        next_round_batch.info["death_player"],
        np.full((1, 4), -1, dtype=np.int16),
    )
    np.testing.assert_array_equal(
        next_round_batch.info["death_count"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        next_round_batch.action_mask,
        np.ones((1, 4, 3), dtype=bool),
    )
    np.testing.assert_array_equal(
        next_round_batch.reward,
        np.zeros((1, 4), dtype=np.float32),
    )
    _assert_public_final_metadata(next_round_batch, [], expected_reward=None)
    np.testing.assert_array_equal(next_round_batch.info["round_id"], np.asarray([2]))
    np.testing.assert_array_equal(next_round_batch.info["source_round_id"], np.asarray([2]))
    assert int(next_round_batch.info["random_tape_cursor"][0]) == 32
    assert int(next_round_batch.info["random_tape_draw_count"][0]) == 32
    assert [
        (call["tape_index"], call["player"])
        for call in next_round_batch.info["warmdown_info"]["spawn_infos"][0]["random_calls"]
    ] == [
        (20, 3),
        (21, 3),
        (22, 3),
        (23, 2),
        (24, 2),
        (25, 2),
        (26, 1),
        (27, 1),
        (28, 1),
        (29, 0),
        (30, 0),
        (31, 0),
    ]


def test_3p_public_match_mode_match_end_metadata_after_warmdown():
    env, _ = _reset_3p_public_lifecycle_env(
        "source_lifecycle_match_end_at_max_score_3p.json",
        episode_end_mode="match",
    )
    _force_3p_player0_round_win(env)

    round_batch = env.step(np.asarray([[1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(round_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        round_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(round_batch.action_mask, np.zeros((1, 3, 3), dtype=bool))
    np.testing.assert_array_equal(round_batch.reward, np.asarray([[1.0, -1.0, -1.0]], dtype=np.float32))
    _assert_public_final_metadata(round_batch, [], expected_reward=None)

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 0
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 1
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        warmdown_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["needs_reset"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(warmdown_batch.action_mask, np.zeros((1, 3, 3), dtype=bool))
    np.testing.assert_array_equal(warmdown_batch.reward, np.zeros((1, 3), dtype=np.float32))
    _assert_public_final_metadata(
        warmdown_batch,
        [0],
        expected_reward=np.zeros((1, 3), dtype=np.float32),
    )


def test_3p_public_tied_max_score_leaders_continue_to_next_round_metadata():
    env, reset_batch = _reset_3p_public_lifecycle_env(
        "source_lifecycle_tie_at_max_score_3p.json"
    )

    assert int(reset_batch.info["max_score_by_row"][0]) == 1
    assert int(reset_batch.info["random_tape_cursor"][0]) == 12

    env.state["pos"][0, 2] = np.asarray([93.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 2] = 0.0
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]

    first_death_batch = env.step(np.asarray([[1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(first_death_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        first_death_batch.info["death_player"],
        np.asarray([[2, -1, -1]], dtype=np.int16),
    )
    assert int(first_death_batch.info["random_tape_cursor"][0]) == 13

    env.state["pos"][0, 1] = np.asarray([1.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 1] = math.pi
    env.state["prev_pos"][0, 1] = env.state["pos"][0, 1]
    env.state["pos"][0, 0] = np.asarray([47.5, 1.0], dtype=np.float64)
    env.state["heading"][0, 0] = math.tau * 0.75
    env.state["prev_pos"][0, 0] = env.state["pos"][0, 0]

    terminal_batch = env.step(np.asarray([[1, 1, -1]], dtype=np.int16))

    np.testing.assert_array_equal(terminal_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        terminal_batch.info["score"],
        np.asarray([[1, 1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW], dtype=np.int16),
    )
    np.testing.assert_array_equal(terminal_batch.info["draw"], np.asarray([True], dtype=bool))
    assert terminal_batch.info["winner_ids"] == [[]]
    assert int(terminal_batch.info["random_tape_cursor"][0]) == 15

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 0
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        warmdown_batch.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    _assert_public_final_metadata(warmdown_batch, [], expected_reward=None)
    np.testing.assert_array_equal(
        warmdown_batch.info["round_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["alive"],
        np.asarray([[True, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["score"],
        np.asarray([[1, 1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.action_mask,
        np.ones((1, 3, 3), dtype=bool),
    )
    assert int(warmdown_batch.info["random_tape_cursor"][0]) == 24


def test_4p_public_tied_max_score_leaders_continue_to_next_round_metadata():
    env, reset_batch = _reset_4p_public_lifecycle_env(
        "source_lifecycle_tie_at_max_score_4p.json"
    )

    assert int(reset_batch.info["max_score_by_row"][0]) == 2
    assert int(reset_batch.info["random_tape_cursor"][0]) == 16

    env.state["pos"][0, 3] = np.asarray([99.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 3] = 0.0
    env.state["prev_pos"][0, 3] = env.state["pos"][0, 3]
    env.state["pos"][0, 2] = np.asarray([1.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 2] = math.pi
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]

    first_death_batch = env.step(np.asarray([[1, 1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(first_death_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        first_death_batch.info["alive"],
        np.asarray([[True, True, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["death_player"],
        np.asarray([[3, 2, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["score"],
        np.zeros((1, 4), dtype=np.int32),
    )
    assert int(first_death_batch.info["random_tape_cursor"][0]) == 18

    env.state["pos"][0, 1] = np.asarray([50.5, 1.0], dtype=np.float64)
    env.state["heading"][0, 1] = math.tau * 0.75
    env.state["prev_pos"][0, 1] = env.state["pos"][0, 1]
    env.state["pos"][0, 0] = np.asarray([50.5, 99.0], dtype=np.float64)
    env.state["heading"][0, 0] = math.pi / 2.0
    env.state["prev_pos"][0, 0] = env.state["pos"][0, 0]

    terminal_batch = env.step(np.asarray([[1, 1, -1, -1]], dtype=np.int16))

    np.testing.assert_array_equal(terminal_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        terminal_batch.info["score"],
        np.asarray([[2, 2, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["death_player"],
        np.asarray([[3, 2, 1, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW], dtype=np.int16),
    )
    np.testing.assert_array_equal(terminal_batch.info["draw"], np.asarray([True], dtype=bool))
    assert terminal_batch.info["winner_ids"] == [[]]
    assert int(terminal_batch.info["random_tape_cursor"][0]) == 20

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 0
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        warmdown_batch.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    _assert_public_final_metadata(warmdown_batch, [], expected_reward=None)
    np.testing.assert_array_equal(
        warmdown_batch.info["round_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["alive"],
        np.asarray([[True, True, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["score"],
        np.asarray([[2, 2, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.action_mask,
        np.ones((1, 4, 3), dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["round_id"],
        np.asarray([2]),
    )
    assert int(warmdown_batch.info["random_tape_cursor"][0]) == 32


def test_3p_public_present_absent_tied_max_score_leaders_continue_to_next_round_metadata():
    scenario_name = "source_lifecycle_present_absent_3p_tie_at_max_score.json"
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=3,
        seed=555,
        decision_ms=100.0,
        body_capacity=16,
        event_capacity=16,
        timer_capacity=3,
        random_tape_capacity=17,
        max_score=_lifecycle_max_score(scenario_name),
    )
    reset_batch = env.reset(
        seed=np.asarray([555], dtype=np.uint64),
        present=np.asarray([[True, False, True]], dtype=bool),
        source_fixture_random_tape_values=_lifecycle_random_tape(scenario_name),
        source_fixture_ref=f"scenarios/environment/{scenario_name}",
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )

    assert reset_batch.info["metadata_only"] is True
    assert int(reset_batch.info["max_score_by_row"][0]) == 1
    assert int(reset_batch.info["random_tape_cursor"][0]) == 9
    np.testing.assert_array_equal(
        reset_batch.info["source_fixture_ref"],
        np.asarray([f"scenarios/environment/{scenario_name}"], dtype=object),
    )
    np.testing.assert_array_equal(
        reset_batch.info["present"],
        np.asarray([[True, False, True]], dtype=bool),
    )

    env.state["pos"][0, 2] = np.asarray([93.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 2] = 0.0
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]
    env.state["pos"][0, 0] = np.asarray([1.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 0] = math.pi
    env.state["prev_pos"][0, 0] = env.state["pos"][0, 0]

    terminal_batch = env.step(np.asarray([[1, -1, 1]], dtype=np.int16))

    assert terminal_batch.info["metadata_only"] is True
    np.testing.assert_array_equal(terminal_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        terminal_batch.info["present"],
        np.asarray([[True, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["score"],
        np.asarray([[1, 0, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["death_player"],
        np.asarray([[1, 2, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["round_winner"],
        np.asarray([-1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["match_winner"],
        np.asarray([-1], dtype=np.int16),
    )
    np.testing.assert_array_equal(terminal_batch.info["draw"], np.asarray([True], dtype=bool))
    assert terminal_batch.info["winner_ids"] == [[]]
    assert terminal_batch.info["round_winner_ids"] == [[]]
    assert terminal_batch.info["match_winner_ids"] == [[]]
    assert int(terminal_batch.info["random_tape_cursor"][0]) == 11
    _assert_public_final_metadata(terminal_batch, [0], expected_reward=terminal_batch.reward)

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["metadata_only"] is True
    assert warmdown_batch.info["warmdown_info"]["game_stop_fires"] == 1
    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 0
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        warmdown_batch.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    _assert_public_final_metadata(warmdown_batch, [], expected_reward=None)
    np.testing.assert_array_equal(
        warmdown_batch.info["present"],
        np.asarray([[True, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["alive"],
        np.asarray([[True, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["score"],
        np.asarray([[1, 0, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["death_player"],
        np.asarray([[1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["round_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_winner"],
        np.asarray([-1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        warmdown_batch.action_mask,
        np.asarray(
            [
                [
                    [True, True, True],
                    [False, False, False],
                    [True, True, True],
                ]
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(warmdown_batch.info["round_id"], np.asarray([2]))
    assert int(warmdown_batch.info["random_tape_cursor"][0]) == 17


def test_4p_public_present_absent_tied_max_score_leaders_continue_to_next_round_metadata():
    scenario_name = "source_lifecycle_present_absent_4p_tie_at_max_score.json"
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        seed=555,
        decision_ms=100.0,
        body_capacity=24,
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=25,
        max_score=_lifecycle_max_score(scenario_name),
    )
    reset_batch = env.reset(
        seed=np.asarray([555], dtype=np.uint64),
        present=np.asarray([[True, False, True, True]], dtype=bool),
        source_fixture_random_tape_values=_lifecycle_random_tape(scenario_name),
        source_fixture_ref=f"scenarios/environment/{scenario_name}",
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )

    assert reset_batch.info["metadata_only"] is True
    assert int(reset_batch.info["max_score_by_row"][0]) == 1
    assert int(reset_batch.info["random_tape_cursor"][0]) == 13
    np.testing.assert_array_equal(
        reset_batch.info["source_fixture_ref"],
        np.asarray([f"scenarios/environment/{scenario_name}"], dtype=object),
    )
    np.testing.assert_array_equal(
        reset_batch.info["present"],
        np.asarray([[True, False, True, True]], dtype=bool),
    )

    env.state["pos"][0, 3] = np.asarray([99.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 3] = 0.0
    env.state["prev_pos"][0, 3] = env.state["pos"][0, 3]
    env.state["pos"][0, 2] = np.asarray([1.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 2] = math.pi
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]
    env.state["pos"][0, 0] = np.asarray([50.5, 1.0], dtype=np.float64)
    env.state["heading"][0, 0] = math.tau * 0.75
    env.state["prev_pos"][0, 0] = env.state["pos"][0, 0]

    terminal_batch = env.step(np.asarray([[1, -1, 1, 1]], dtype=np.int16))

    assert terminal_batch.info["metadata_only"] is True
    np.testing.assert_array_equal(terminal_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        terminal_batch.info["present"],
        np.asarray([[True, False, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["score"],
        np.asarray([[1, 0, 1, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["death_player"],
        np.asarray([[1, 3, 2, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["round_winner"],
        np.asarray([-1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["match_winner"],
        np.asarray([-1], dtype=np.int16),
    )
    np.testing.assert_array_equal(terminal_batch.info["draw"], np.asarray([True], dtype=bool))
    assert terminal_batch.info["winner_ids"] == [[]]
    assert terminal_batch.info["round_winner_ids"] == [[]]
    assert terminal_batch.info["match_winner_ids"] == [[]]
    assert int(terminal_batch.info["random_tape_cursor"][0]) == 16
    _assert_public_final_metadata(terminal_batch, [0], expected_reward=terminal_batch.reward)

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["metadata_only"] is True
    assert warmdown_batch.info["warmdown_info"]["game_stop_fires"] == 1
    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 0
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        warmdown_batch.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    _assert_public_final_metadata(warmdown_batch, [], expected_reward=None)
    np.testing.assert_array_equal(
        warmdown_batch.info["present"],
        np.asarray([[True, False, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["alive"],
        np.asarray([[True, False, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["score"],
        np.asarray([[1, 0, 1, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["death_player"],
        np.asarray([[1, -1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["round_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_winner"],
        np.asarray([-1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        warmdown_batch.action_mask,
        np.asarray(
            [
                [
                    [True, True, True],
                    [False, False, False],
                    [True, True, True],
                    [True, True, True],
                ]
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(warmdown_batch.info["round_id"], np.asarray([2]))
    assert int(warmdown_batch.info["random_tape_cursor"][0]) == 25


def test_4p_public_unique_max_score_leader_warmdown_reports_match_end_metadata():
    env, reset_batch = _reset_4p_public_lifecycle_env(
        "source_lifecycle_match_end_at_max_score_4p.json",
        extra_random_tape_values=8,
    )

    assert int(reset_batch.info["max_score_by_row"][0]) == 3
    assert int(reset_batch.info["random_tape_cursor"][0]) == 16
    np.testing.assert_array_equal(reset_batch.info["round_id"], np.asarray([1]))

    env.state["pos"][0, 0] = np.asarray([10.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 0] = 0.0
    env.state["prev_pos"][0, 0] = env.state["pos"][0, 0]
    env.state["speed"][0, 0] = 8.0
    env.state["pos"][0, 3] = np.asarray([99.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 3] = 0.0
    env.state["prev_pos"][0, 3] = env.state["pos"][0, 3]

    first_death_batch = env.step(np.asarray([[1, 1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(first_death_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        first_death_batch.info["alive"],
        np.asarray([[True, True, True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["death_player"],
        np.asarray([[3, -1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        first_death_batch.info["score"],
        np.zeros((1, 4), dtype=np.int32),
    )
    first_death_cursor = int(first_death_batch.info["random_tape_cursor"][0])
    assert first_death_cursor >= 16

    env.state["pos"][0, 2] = np.asarray([1.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 2] = math.pi
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]

    second_death_batch = env.step(np.asarray([[1, 1, 1, -1]], dtype=np.int16))

    np.testing.assert_array_equal(second_death_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        second_death_batch.info["alive"],
        np.asarray([[True, True, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        second_death_batch.info["death_player"],
        np.asarray([[3, 2, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second_death_batch.info["score"],
        np.zeros((1, 4), dtype=np.int32),
    )
    second_death_cursor = int(second_death_batch.info["random_tape_cursor"][0])
    assert second_death_cursor >= first_death_cursor

    env.state["pos"][0, 1] = np.asarray([50.5, 1.0], dtype=np.float64)
    env.state["heading"][0, 1] = math.tau * 0.75
    env.state["prev_pos"][0, 1] = env.state["pos"][0, 1]

    terminal_batch = env.step(np.asarray([[1, 1, -1, -1]], dtype=np.int16))

    np.testing.assert_array_equal(terminal_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        terminal_batch.info["score"],
        np.asarray([[3, 2, 1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["death_player"],
        np.asarray([[3, 2, 1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["round_winner"],
        np.asarray([0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["match_winner"],
        np.asarray([-1], dtype=np.int16),
    )
    assert terminal_batch.info["winner_ids"] == [[0]]
    assert terminal_batch.info["round_winner_ids"] == [[0]]
    assert terminal_batch.info["match_winner_ids"] == [[]]
    terminal_cursor = int(terminal_batch.info["random_tape_cursor"][0])
    assert terminal_cursor >= second_death_cursor

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["warmdown_waited"] is True
    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 0
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["scheduled_timer_count"] == 0
    assert warmdown_batch.info["warmdown_info"]["random_tape_draws"] == 0
    np.testing.assert_array_equal(
        warmdown_batch.info["next_round_rows"],
        np.asarray([], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["terminal_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        warmdown_batch.info["needs_reset"],
        np.asarray([True], dtype=bool),
    )
    _assert_public_final_metadata(
        warmdown_batch,
        [0],
        expected_reward=np.zeros((1, 4), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["round_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["warmdown_pending"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["round_winner"],
        np.asarray([0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_winner"],
        np.asarray([0], dtype=np.int16),
    )
    assert warmdown_batch.info["round_winner_ids"] == [[0]]
    assert warmdown_batch.info["match_winner_ids"] == [[0]]
    np.testing.assert_array_equal(
        warmdown_batch.info["score"],
        np.asarray([[3, 2, 1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["alive"],
        np.asarray([[True, False, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(warmdown_batch.info["round_id"], np.asarray([1]))
    np.testing.assert_array_equal(
        warmdown_batch.action_mask,
        np.zeros((1, 4, 3), dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.reward,
        np.zeros((1, 4), dtype=np.float32),
    )
    assert int(warmdown_batch.info["random_tape_cursor"][0]) == terminal_cursor

    with pytest.raises(RuntimeError, match="reset must be called before stepping rows"):
        env.step(np.asarray([[1, -1, -1, -1]], dtype=np.int16))


def test_3p_public_multi_round_match_ends_on_second_warmdown_metadata():
    env, reset_batch = _reset_3p_public_lifecycle_env(
        "source_lifecycle_multi_round_match_end_3p.json"
    )

    assert int(reset_batch.info["max_score_by_row"][0]) == 3
    assert int(reset_batch.info["random_tape_cursor"][0]) == 12
    np.testing.assert_array_equal(reset_batch.info["round_id"], np.asarray([1]))
    np.testing.assert_array_equal(reset_batch.info["source_round_id"], np.asarray([1]))
    np.testing.assert_array_equal(
        reset_batch.info["reset_episode_id"],
        reset_batch.info["episode_id"],
    )
    assert reset_batch.info["lifecycle_policy_id"] == LIFECYCLE_POLICY_ID
    assert reset_batch.info["reset_episode_id_policy"] == RESET_EPISODE_ID_POLICY
    assert reset_batch.info["source_round_id_policy"] == SOURCE_ROUND_ID_POLICY
    assert reset_batch.info["reset_info"]["lifecycle_policy_id"] == LIFECYCLE_POLICY_ID
    np.testing.assert_array_equal(
        reset_batch.info["reset_info"]["reset_episode_id"],
        reset_batch.info["episode_id"],
    )
    np.testing.assert_array_equal(
        reset_batch.info["reset_info"]["reset_round_id"],
        np.asarray([1]),
    )
    reset_episode_id = int(reset_batch.info["reset_episode_id"][0])

    env.state["pos"][0] = np.asarray(
        [[5.0, 5.0], [1.0, 47.5], [93.0, 47.5]],
        dtype=np.float64,
    )
    env.state["heading"][0] = np.asarray(
        [math.pi / 4.0, math.pi, 0.0],
        dtype=np.float64,
    )
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["speed"][0, 0] = 8.0
    env.state["print_manager_distance"][0, 0] = 999.0
    env.state["print_manager_last_pos"][0, 0] = env.state["pos"][0, 0]

    first_terminal_batch = env.step(np.asarray([[1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(
        first_terminal_batch.info["score"],
        np.asarray([[2, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        first_terminal_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    assert int(first_terminal_batch.info["random_tape_cursor"][0]) == 14
    np.testing.assert_array_equal(first_terminal_batch.info["round_id"], np.asarray([1]))
    np.testing.assert_array_equal(
        first_terminal_batch.info["source_round_id"],
        np.asarray([1]),
    )
    np.testing.assert_array_equal(
        first_terminal_batch.info["reset_episode_id"],
        np.asarray([reset_episode_id]),
    )

    first_warmdown_batch = env.advance_warmdown(5000.0)

    assert first_warmdown_batch.info["warmdown_info"]["next_round_count"] == 1
    assert first_warmdown_batch.info["warmdown_info"]["match_end_count"] == 0
    np.testing.assert_array_equal(first_warmdown_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        first_warmdown_batch.info["score"],
        np.asarray([[2, 0, 0]], dtype=np.int32),
    )
    assert int(first_warmdown_batch.info["random_tape_cursor"][0]) == 24
    np.testing.assert_array_equal(first_warmdown_batch.info["round_id"], np.asarray([2]))
    np.testing.assert_array_equal(
        first_warmdown_batch.info["source_round_id"],
        np.asarray([2]),
    )
    np.testing.assert_array_equal(
        first_warmdown_batch.info["reset_episode_id"],
        np.asarray([reset_episode_id]),
    )

    game_start_batch = env.advance_warmup(3000.0)

    assert game_start_batch.info["warmup_info"]["game_start_fires"] == 1
    assert game_start_batch.info["warmup_info"]["print_manager_delayed_start_fires"] == 0
    assert int(game_start_batch.info["random_tape_cursor"][0]) == 24
    np.testing.assert_array_equal(game_start_batch.info["round_id"], np.asarray([2]))
    np.testing.assert_array_equal(game_start_batch.info["source_round_id"], np.asarray([2]))
    np.testing.assert_array_equal(
        game_start_batch.info["reset_episode_id"],
        np.asarray([reset_episode_id]),
    )

    env.state["pos"][0] = np.asarray(
        [[5.0, 5.0], [70.0, 20.0], [70.0, 70.0]],
        dtype=np.float64,
    )
    env.state["heading"][0] = np.asarray(
        [math.pi / 4.0, math.pi / 4.0, math.tau * 0.875],
        dtype=np.float64,
    )
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["speed"][0] = np.asarray([8.0, 8.0, 8.0], dtype=np.float64)

    print_start_batch = env.advance_warmup(3000.0)

    assert print_start_batch.info["warmup_info"]["game_start_fires"] == 0
    assert print_start_batch.info["warmup_info"]["print_manager_delayed_start_fires"] == 3
    assert int(print_start_batch.info["random_tape_cursor"][0]) == 27
    np.testing.assert_array_equal(print_start_batch.info["round_id"], np.asarray([2]))
    np.testing.assert_array_equal(print_start_batch.info["source_round_id"], np.asarray([2]))
    np.testing.assert_array_equal(
        print_start_batch.info["reset_episode_id"],
        np.asarray([reset_episode_id]),
    )

    env.state["pos"][0, 2] = np.asarray([93.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 2] = 0.0
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]
    env.state["speed"][0, 2] = 16.0
    env.state["pos"][0, 1] = np.asarray([1.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 1] = math.pi
    env.state["prev_pos"][0, 1] = env.state["pos"][0, 1]
    env.state["speed"][0, 1] = 16.0
    env.state["print_manager_distance"][0, 0] = 999.0
    env.state["print_manager_last_pos"][0, 0] = env.state["pos"][0, 0]

    second_terminal_batch = env.step(np.asarray([[1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(
        second_terminal_batch.info["score"],
        np.asarray([[4, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        second_terminal_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    assert int(second_terminal_batch.info["random_tape_cursor"][0]) == 29
    np.testing.assert_array_equal(second_terminal_batch.info["round_id"], np.asarray([2]))
    np.testing.assert_array_equal(
        second_terminal_batch.info["source_round_id"],
        np.asarray([2]),
    )
    np.testing.assert_array_equal(
        second_terminal_batch.info["reset_episode_id"],
        np.asarray([reset_episode_id]),
    )

    second_warmdown_batch = env.advance_warmdown(5000.0)

    assert second_warmdown_batch.info["warmdown_info"]["next_round_count"] == 0
    assert second_warmdown_batch.info["warmdown_info"]["match_end_count"] == 1
    np.testing.assert_array_equal(second_warmdown_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        second_warmdown_batch.info["match_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        second_warmdown_batch.info["match_winner"],
        np.asarray([0], dtype=np.int16),
    )
    assert second_warmdown_batch.info["match_winner_ids"] == [[0]]
    np.testing.assert_array_equal(
        second_warmdown_batch.action_mask,
        np.zeros((1, 3, 3), dtype=bool),
    )
    np.testing.assert_array_equal(
        second_warmdown_batch.reward,
        np.zeros((1, 3), dtype=np.float32),
    )
    assert int(second_warmdown_batch.info["random_tape_cursor"][0]) == 29
    np.testing.assert_array_equal(second_warmdown_batch.info["round_id"], np.asarray([2]))
    np.testing.assert_array_equal(
        second_warmdown_batch.info["source_round_id"],
        np.asarray([2]),
    )
    np.testing.assert_array_equal(
        second_warmdown_batch.info["reset_episode_id"],
        np.asarray([reset_episode_id]),
    )


def test_4p_public_multi_round_match_ends_on_second_warmdown_metadata():
    env, reset_batch = _reset_4p_public_lifecycle_env(
        "source_lifecycle_multi_round_match_end_4p.json"
    )

    assert int(reset_batch.info["max_score_by_row"][0]) == 5
    assert int(reset_batch.info["random_tape_cursor"][0]) == 16
    np.testing.assert_array_equal(reset_batch.info["round_id"], np.asarray([1]))
    np.testing.assert_array_equal(reset_batch.info["source_round_id"], np.asarray([1]))
    assert reset_batch.info["lifecycle_policy_id"] == LIFECYCLE_POLICY_ID
    assert reset_batch.info["reset_episode_id_policy"] == RESET_EPISODE_ID_POLICY
    assert reset_batch.info["source_round_id_policy"] == SOURCE_ROUND_ID_POLICY
    reset_episode_id = int(reset_batch.info["reset_episode_id"][0])

    _force_4p_player0_round_win(env)

    first_terminal_batch = env.step(np.asarray([[1, 1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(
        first_terminal_batch.info["score"],
        np.asarray([[3, 0, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        first_terminal_batch.info["death_player"],
        np.asarray([[3, 2, 1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        first_terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        first_terminal_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    assert first_terminal_batch.info["round_winner_ids"] == [[0]]
    assert first_terminal_batch.info["match_winner_ids"] == [[]]
    assert int(first_terminal_batch.info["random_tape_cursor"][0]) == 19
    np.testing.assert_array_equal(first_terminal_batch.info["round_id"], np.asarray([1]))
    np.testing.assert_array_equal(
        first_terminal_batch.info["source_round_id"],
        np.asarray([1]),
    )
    np.testing.assert_array_equal(
        first_terminal_batch.info["reset_episode_id"],
        np.asarray([reset_episode_id]),
    )

    first_warmdown_batch = env.advance_warmdown(5000.0)

    assert first_warmdown_batch.info["warmdown_info"]["next_round_count"] == 1
    assert first_warmdown_batch.info["warmdown_info"]["match_end_count"] == 0
    assert first_warmdown_batch.info["warmdown_info"]["random_tape_draws"] == 13
    np.testing.assert_array_equal(first_warmdown_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        first_warmdown_batch.info["score"],
        np.asarray([[3, 0, 0, 0]], dtype=np.int32),
    )
    assert int(first_warmdown_batch.info["random_tape_cursor"][0]) == 32
    np.testing.assert_array_equal(first_warmdown_batch.info["round_id"], np.asarray([2]))
    np.testing.assert_array_equal(
        first_warmdown_batch.info["source_round_id"],
        np.asarray([2]),
    )
    np.testing.assert_array_equal(
        first_warmdown_batch.info["reset_episode_id"],
        np.asarray([reset_episode_id]),
    )

    game_start_batch = env.advance_warmup(3000.0)

    assert game_start_batch.info["warmup_info"]["game_start_fires"] == 1
    assert game_start_batch.info["warmup_info"]["print_manager_delayed_start_fires"] == 0
    assert int(game_start_batch.info["random_tape_cursor"][0]) == 32
    np.testing.assert_array_equal(game_start_batch.info["round_id"], np.asarray([2]))
    np.testing.assert_array_equal(game_start_batch.info["source_round_id"], np.asarray([2]))
    np.testing.assert_array_equal(
        game_start_batch.info["reset_episode_id"],
        np.asarray([reset_episode_id]),
    )

    _force_4p_player0_round_win(env)

    print_start_batch = env.advance_warmup(3000.0)

    assert print_start_batch.info["warmup_info"]["game_start_fires"] == 0
    assert print_start_batch.info["warmup_info"]["print_manager_delayed_start_fires"] == 4
    assert int(print_start_batch.info["random_tape_cursor"][0]) == 36
    np.testing.assert_array_equal(print_start_batch.info["round_id"], np.asarray([2]))
    np.testing.assert_array_equal(print_start_batch.info["source_round_id"], np.asarray([2]))
    np.testing.assert_array_equal(
        print_start_batch.info["reset_episode_id"],
        np.asarray([reset_episode_id]),
    )

    _force_4p_player0_round_win(env)

    second_terminal_batch = env.step(np.asarray([[1, 1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(
        second_terminal_batch.info["score"],
        np.asarray([[6, 0, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        second_terminal_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    assert int(second_terminal_batch.info["random_tape_cursor"][0]) == 39
    np.testing.assert_array_equal(second_terminal_batch.info["round_id"], np.asarray([2]))
    np.testing.assert_array_equal(
        second_terminal_batch.info["source_round_id"],
        np.asarray([2]),
    )
    np.testing.assert_array_equal(
        second_terminal_batch.info["reset_episode_id"],
        np.asarray([reset_episode_id]),
    )

    second_warmdown_batch = env.advance_warmdown(5000.0)

    assert second_warmdown_batch.info["warmdown_info"]["next_round_count"] == 0
    assert second_warmdown_batch.info["warmdown_info"]["match_end_count"] == 1
    assert second_warmdown_batch.info["warmdown_info"]["random_tape_draws"] == 0
    np.testing.assert_array_equal(second_warmdown_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        second_warmdown_batch.info["match_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        second_warmdown_batch.info["match_winner"],
        np.asarray([0], dtype=np.int16),
    )
    assert second_warmdown_batch.info["match_winner_ids"] == [[0]]
    np.testing.assert_array_equal(
        second_warmdown_batch.action_mask,
        np.zeros((1, 4, 3), dtype=bool),
    )
    np.testing.assert_array_equal(
        second_warmdown_batch.reward,
        np.zeros((1, 4), dtype=np.float32),
    )
    _assert_public_final_metadata(
        second_warmdown_batch,
        [0],
        expected_reward=np.zeros((1, 4), dtype=np.float32),
    )
    assert int(second_warmdown_batch.info["random_tape_cursor"][0]) == 39
    np.testing.assert_array_equal(second_warmdown_batch.info["round_id"], np.asarray([2]))
    np.testing.assert_array_equal(
        second_warmdown_batch.info["source_round_id"],
        np.asarray([2]),
    )
    np.testing.assert_array_equal(
        second_warmdown_batch.info["reset_episode_id"],
        np.asarray([reset_episode_id]),
    )


def test_3p_direct_wall_fixture_step_returns_metadata_only_terminal_info():
    state, actions, step_ms = _fixture_state_and_actions(
        "scenarios/environment/source_normal_wall_3p_two_die_one_survivor_step.json",
        body_capacity=8,
    )
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=3,
        decision_ms=step_ms,
        body_capacity=8,
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=8,
        event_mode="debug-event",
    )

    reset_batch = env.reset_from_state_arrays(
        state,
        reset_seed=np.asarray([123], dtype=np.uint64),
    )
    batch = env.step(actions)

    assert reset_batch.info["observation_schema_id"] == DEBUG_METADATA_OBSERVATION_SCHEMA_ID
    assert batch.info["observation_schema_id"] == DEBUG_METADATA_OBSERVATION_SCHEMA_ID
    assert batch.info["metadata_only"] is True
    assert batch.info["trainer_observation_claim"] is False
    assert batch.info["trainer_observation_schema_id"] is None
    assert batch.info["natural_multiplayer_reset_claim"] is False
    assert batch.info["warmdown_waited"] is False

    assert batch.info["player_count"] == 3
    np.testing.assert_array_equal(batch.info["present"], np.asarray([[True, True, True]]))
    np.testing.assert_array_equal(batch.info["alive"], np.asarray([[True, False, False]]))
    np.testing.assert_array_equal(batch.info["score"], np.asarray([[2, 0, 0]], dtype=np.int32))
    np.testing.assert_array_equal(
        batch.info["round_score"],
        np.asarray([[0, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(batch.info["death_count"], np.asarray([2], dtype=np.int32))
    np.testing.assert_array_equal(
        batch.info["death_player"],
        np.asarray([[2, 1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        batch.info["death_cause"],
        np.asarray(
            [
                [
                    vector_runtime.DEATH_CAUSE_WALL,
                    vector_runtime.DEATH_CAUSE_WALL,
                    vector_runtime.DEATH_CAUSE_NONE,
                ]
            ],
            dtype=np.int16,
        ),
    )
    assert batch.info["death_cause_name"].tolist() == [["wall", "wall", "none"]]
    np.testing.assert_array_equal(
        batch.info["death_hit_owner"],
        np.asarray([[-1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(batch.info["winner"], np.asarray([0], dtype=np.int16))
    np.testing.assert_array_equal(batch.info["draw"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.truncated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    assert batch.info["terminal_reason_name"].tolist() == ["round_survivor_win"]
    assert batch.info["winner_ids"] == [[0]]
    assert batch.info["loser_ids"] == [[1, 2]]
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["player_action"],
        actions.astype(np.int16),
    )
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["native_control_value"],
        actions.astype(np.int8) - 1,
    )
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["action_required"],
        np.asarray([[True, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.reward,
        np.asarray([[1.0, -1.0, -1.0]], dtype=np.float32),
    )
    assert batch.final_observation is not None
    assert batch.final_reward is not None


def test_4p_ordered_wall_fixture_public_env_metadata_matches_source_scores():
    state, fixture = _fixture_state(
        "scenarios/environment/source_normal_wall_4p_ordered_deaths_survivor_score.json",
        body_capacity=8,
    )
    first_actions, step_ms = _fixture_actions_for_step(fixture, step_index=0)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        decision_ms=step_ms,
        body_capacity=8,
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=8,
        event_mode="debug-event",
    )
    env.reset_from_state_arrays(state, reset_seed=np.asarray([456], dtype=np.uint64))

    first_batch = env.step(first_actions)
    second_actions, _ = _fixture_actions_for_step(fixture, step_index=1)
    second_batch = env.step(second_actions)
    third_actions, _ = _fixture_actions_for_step(fixture, step_index=2)
    batch = env.step(third_actions)

    np.testing.assert_array_equal(first_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(second_batch.done, np.asarray([False], dtype=bool))
    assert batch.info["metadata_only"] is True
    assert batch.info["trainer_observation_claim"] is False
    assert batch.info["natural_multiplayer_reset_claim"] is False
    assert batch.info["player_count"] == 4
    np.testing.assert_array_equal(
        batch.info["alive"],
        np.asarray([[True, False, False, False]]),
    )
    np.testing.assert_array_equal(
        batch.info["score"],
        np.asarray([[3, 0, 1, 2]], dtype=np.int32),
    )
    np.testing.assert_array_equal(batch.info["death_count"], np.asarray([3], dtype=np.int32))
    np.testing.assert_array_equal(
        batch.info["death_player"],
        np.asarray([[1, 2, 3, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(batch.info["winner"], np.asarray([0], dtype=np.int16))
    np.testing.assert_array_equal(batch.info["draw"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    assert batch.info["winner_ids"] == [[0]]
    assert batch.info["loser_ids"] == [[1, 2, 3]]
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["player_action"],
        third_actions.astype(np.int16),
    )
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["native_control_value"],
        third_actions.astype(np.int8) - 1,
    )
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["action_required"],
        np.asarray([[True, False, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.reward,
        np.asarray([[1.0, -1.0, -1.0, -1.0]], dtype=np.float32),
    )
    assert batch.final_observation is not None
    assert batch.final_reward is not None


def test_4p_public_nonterminal_dead_players_mask_out_and_reward_zero():
    state, fixture = _fixture_state(
        "scenarios/environment/source_normal_wall_4p_ordered_deaths_survivor_score.json",
        body_capacity=8,
    )
    first_actions, step_ms = _fixture_actions_for_step(fixture, step_index=0)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        decision_ms=step_ms,
        body_capacity=8,
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=8,
        event_mode="debug-event",
    )
    env.reset_from_state_arrays(state, reset_seed=np.asarray([456], dtype=np.uint64))

    first_batch = env.step(first_actions)

    np.testing.assert_array_equal(first_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(first_batch.reward, np.zeros((1, 4), dtype=np.float32))
    np.testing.assert_array_equal(
        first_batch.info["alive"],
        np.asarray([[True, False, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        first_batch.info["death_player"],
        np.asarray([[1, -1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        first_batch.action_mask,
        np.asarray(
            [
                [
                    [True, True, True],
                    [False, False, False],
                    [True, True, True],
                    [True, True, True],
                ]
            ],
            dtype=bool,
        ),
    )

    second_actions, _ = _fixture_actions_for_step(fixture, step_index=1)
    second_batch = env.step(second_actions)

    np.testing.assert_array_equal(second_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(second_batch.reward, np.zeros((1, 4), dtype=np.float32))
    np.testing.assert_array_equal(
        second_batch.info["alive"],
        np.asarray([[True, False, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        second_batch.info["death_player"],
        np.asarray([[1, 2, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second_batch.action_mask,
        np.asarray(
            [
                [
                    [True, True, True],
                    [False, False, False],
                    [False, False, False],
                    [True, True, True],
                ]
            ],
            dtype=bool,
        ),
    )
    _assert_public_final_metadata(first_batch, [], expected_reward=None)
    _assert_public_final_metadata(second_batch, [], expected_reward=None)


def test_4p_terminal_draw_public_env_metadata_matches_source_scores():
    state, fixture = _fixture_state(
        "scenarios/environment/source_normal_wall_4p_two_prior_then_same_frame_terminal_draw.json",
        body_capacity=8,
    )
    first_actions, step_ms = _fixture_actions_for_step(fixture, step_index=0)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        decision_ms=step_ms,
        body_capacity=8,
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=8,
        event_mode="debug-event",
    )
    env.reset_from_state_arrays(state, reset_seed=np.asarray([789], dtype=np.uint64))

    first_batch = env.step(first_actions)
    second_actions, _ = _fixture_actions_for_step(fixture, step_index=1)
    second_batch = env.step(second_actions)
    third_actions, _ = _fixture_actions_for_step(fixture, step_index=2)
    batch = env.step(third_actions)

    np.testing.assert_array_equal(first_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(second_batch.done, np.asarray([False], dtype=bool))
    assert batch.info["metadata_only"] is True
    assert batch.info["trainer_observation_claim"] is False
    assert batch.info["natural_multiplayer_reset_claim"] is False
    np.testing.assert_array_equal(
        batch.info["alive"],
        np.asarray([[False, False, False, False]]),
    )
    np.testing.assert_array_equal(
        batch.info["score"],
        np.asarray([[2, 2, 1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(batch.info["death_count"], np.asarray([4], dtype=np.int32))
    np.testing.assert_array_equal(
        batch.info["death_player"],
        np.asarray([[3, 2, 1, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(batch.info["winner"], np.asarray([-1], dtype=np.int16))
    np.testing.assert_array_equal(batch.info["draw"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(batch.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW], dtype=np.int16),
    )
    assert batch.info["winner_ids"] == [[]]
    assert batch.info["loser_ids"] == [[]]
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["player_action"],
        third_actions.astype(np.int16),
    )
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["native_control_value"],
        third_actions.astype(np.int8) - 1,
    )
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["action_required"],
        np.asarray([[True, True, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.reward,
        np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    assert batch.final_observation is not None
    assert batch.final_reward is not None
