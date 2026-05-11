import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from curvyzero.env import vector_runtime
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ROOT = REPO_ROOT / "scenarios" / "environment"
SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import compare_vector_arrays_to_fidelity as vector_compare  # noqa: E402
import seed_vector_state_from_fixtures as seed_bridge  # noqa: E402


TRAIL_GAP_PROBE_SCENARIOS = (
    "source_trail_gap_hole_space_safe_step.json",
    "source_trail_gap_stored_body_still_kills_step.json",
    "source_trail_gap_print_to_hole_boundary_kills_step.json",
    "source_trail_gap_hole_to_print_boundary_kills_step.json",
)


def _write_two_player_probe(tmp_path: Path, filename: str) -> Path:
    payload = json.loads((SCENARIO_ROOT / filename).read_text())
    payload["player_count"] = 2
    payload["source_setup"]["player_count"] = 2
    payload["source_setup"]["map_size"] = 88
    payload["source_setup"]["room"]["name"] = (
        f"{payload['source_setup']['room']['name']}-2p-probe"
    )
    payload["initial_state"]["map_size"] = 88
    payload["players"] = payload["players"][:2]
    for step in payload["steps"]:
        step["moves"] = [
            move for move in step["moves"] if move["player_id"] in {"p0", "p1"}
        ]
    payload["provenance"]["notes"].append(
        "Generated 2P probe variant used to prove the same trail/body rule "
        "without relying on the older 3P canary."
    )
    path = tmp_path / filename
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _run_js_scenario(path: Path) -> dict[str, object]:
    if shutil.which("node") is None:
        pytest.skip("node executable is not available")
    result = subprocess.run(
        [
            "node",
            str(REPO_ROOT / "tools" / "reference_oracle" / "scenario_runner.js"),
            str(path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _fixture_state_and_actions(
    path: Path,
    *,
    body_capacity: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    fixture = seed_bridge.seed_fixture(path, body_capacity=body_capacity)
    state = vector_compare.array_state_from_seed(fixture)
    prepared_step = vector_compare.prepare_fixture_array_step(
        fixture,
        step_index=0,
    )
    source_moves = np.asarray(prepared_step["source_moves"], dtype=np.int8)
    return (
        state,
        (source_moves.astype(np.int16) + 1).reshape(1, -1),
        float(prepared_step["step_ms"]),
    )


def _step_public_probe(path: Path) -> tuple[VectorMultiplayerEnv, object]:
    state, actions, step_ms = _fixture_state_and_actions(path, body_capacity=8)
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
    env.reset_from_state_arrays(state, reset_seed=np.asarray([101], dtype=np.uint64))
    return env, env.step(actions)


def _js_death_players(frame: dict[str, object]) -> list[int]:
    deaths = [int(avatar_id) - 1 for avatar_id in frame["game"]["deaths"]]
    return deaths + [-1] * (2 - len(deaths))


def _js_death_hit_owners(frame: dict[str, object]) -> list[int]:
    owners: list[int] = []
    for event in frame["events"]:
        if event["event"] != "die":
            continue
        killer = event["data"].get("killer")
        owners.append(-1 if killer is None else int(killer) - 1)
    return owners + [-1] * (2 - len(owners))


def _js_round_winner(frame: dict[str, object]) -> int:
    winner = frame["game"]["roundWinner"]
    return -1 if winner is None else int(winner) - 1


@pytest.mark.parametrize("filename", TRAIL_GAP_PROBE_SCENARIOS)
def test_2p_trail_gap_probe_matches_js_source_and_public_vector_env(
    tmp_path: Path,
    filename: str,
) -> None:
    probe_path = _write_two_player_probe(tmp_path, filename)
    js_payload = _run_js_scenario(probe_path)
    frame = js_payload["trace"][0]
    avatars = frame["avatars"]

    assert frame["game"]["size"] == 88
    assert len(avatars) == 2

    env, batch = _step_public_probe(probe_path)
    expected_alive = np.asarray([[avatar["alive"] for avatar in avatars]], dtype=bool)
    expected_printing = np.asarray(
        [[avatar["printing"] for avatar in avatars]],
        dtype=bool,
    )
    expected_score = np.asarray(
        [[avatar["score"] for avatar in avatars]],
        dtype=np.int32,
    )
    expected_death_player = np.asarray([_js_death_players(frame)], dtype=np.int16)
    expected_death_hit_owner = np.asarray([_js_death_hit_owners(frame)], dtype=np.int16)
    expected_death_cause = np.asarray(
        [
            [
                (
                    vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL
                    if player >= 0
                    else vector_runtime.DEATH_CAUSE_NONE
                )
                for player in expected_death_player[0]
            ]
        ],
        dtype=np.int16,
    )

    np.testing.assert_array_equal(batch.done, np.asarray([not frame["game"]["inRound"]]))
    np.testing.assert_array_equal(batch.info["alive"], expected_alive)
    np.testing.assert_array_equal(env.state["printing"], expected_printing)
    np.testing.assert_array_equal(env.state["score"], expected_score)
    np.testing.assert_array_equal(
        env.state["world_body_count"],
        np.asarray([frame["game"]["worldBodyCount"]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.info["death_count"],
        np.asarray([frame["game"]["deathCount"]], dtype=np.int32),
    )
    np.testing.assert_array_equal(batch.info["death_player"], expected_death_player)
    np.testing.assert_array_equal(batch.info["death_hit_owner"], expected_death_hit_owner)
    np.testing.assert_array_equal(batch.info["death_cause"], expected_death_cause)
    np.testing.assert_array_equal(
        batch.info["winner"],
        np.asarray([_js_round_winner(frame)], dtype=np.int16),
    )
