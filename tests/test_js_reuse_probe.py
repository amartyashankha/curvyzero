import json
import shutil
from pathlib import Path

import pytest

from curvyzero.fidelity.js_reuse_probe import CurvytronJsEnvWorker, run_js_reuse_env_probe


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCENARIO = (
    _REPO_ROOT / "scenarios" / "environment" / "source_kinematics_turn_multistep.json"
)
_LIFECYCLE_RESET_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_spawn_rng_warmup_print_start_2p.json"
)
_LONG_NATURAL_ROLLOUT_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_long_1v1_no_bonus_wall_round_done.json"
)


def _positions(frame: dict[str, object]) -> list[list[float]]:
    state = frame["state"]
    return [[player["x"], player["y"]] for player in state["players"]]


def _headings(frame: dict[str, object]) -> list[float]:
    state = frame["state"]
    return [player["angle"] for player in state["players"]]


def _scores(frame: dict[str, object]) -> list[int]:
    state = frame["state"]
    return [player["score"] for player in state["players"]]


def _alive(frame: dict[str, object]) -> list[bool]:
    state = frame["state"]
    return [player["alive"] for player in state["players"]]


def test_js_reuse_probe_runs_original_js_as_deterministic_reset_and_steps():
    if shutil.which("node") is None:
        pytest.skip("node executable is not available")

    payload = run_js_reuse_env_probe(_SCENARIO)
    repeated = run_js_reuse_env_probe(_SCENARIO)

    assert payload == repeated
    assert payload["schema"] == "curvytron-js-reuse-env-probe-v0"
    assert payload["capabilities"] == {
        "originalSourceLoaded": True,
        "deterministicMathRandom": True,
        "manualTimerAdvance": True,
        "resetCall": "forced_state",
        "stepCall": "avatar.updateAngularVelocity(move) + game.update(stepMs)",
    }
    assert payload["sourceRoot"] == "third_party/curvytron-reference"
    assert payload["loadedSourceCount"] == 25

    reset = payload["reset"]
    assert reset["mode"] == "forced_state"
    assert _positions(reset) == [[20, 40], [60, 40]]
    assert _headings(reset) == [0, 3.141593]

    assert len(payload["steps"]) == 4
    assert [frame["moves"] for frame in payload["steps"]] == [
        [{"player": 1, "move": -1}, {"player": 2, "move": 1}],
        [{"player": 1, "move": -1}, {"player": 2, "move": 0}],
        [{"player": 1, "move": 0}, {"player": 2, "move": 1}],
        [{"player": 1, "move": 1}, {"player": 2, "move": -1}],
    ]

    final = payload["steps"][-1]
    assert _positions(final) == [[21.063765, 39.925415], [58.935365, 39.937827]]
    assert _headings(final) == [-0.046667, 3.188259]
    assert final["reward"] == [0, 0]
    assert final["roundDone"] is False
    assert final["gameDone"] is False
    assert payload["randomCalls"] == []


def test_js_reuse_probe_can_reset_through_original_new_round_spawn():
    if shutil.which("node") is None:
        pytest.skip("node executable is not available")

    payload = run_js_reuse_env_probe(_LIFECYCLE_RESET_SCENARIO)

    assert payload["capabilities"]["resetCall"] == "source_new_round"
    assert payload["reset"]["mode"] == "source_new_round"
    assert payload["reset"]["state"]["game"]["worldActive"] is False
    assert _positions(payload["reset"]) == [[58, 44], [30, 44]]
    assert _headings(payload["reset"]) == [3.241593, 0.1]
    assert [call["value"] for call in payload["randomCalls"]] == [
        0.32051282051282054,
        0.5,
        0.015915494309189534,
        0.6794871794871795,
        0.5,
        0.5159154943091895,
    ]
    assert payload["steps"] == []


def test_js_reuse_worker_keeps_original_js_vm_loaded_across_step_commands():
    if shutil.which("node") is None:
        pytest.skip("node executable is not available")

    batch = run_js_reuse_env_probe(_SCENARIO)
    scenario = json.loads(_SCENARIO.read_text())

    with CurvytronJsEnvWorker() as worker:
        assert worker.ready["schema"] == "curvytron-js-reuse-env-worker-protocol-v0"
        assert worker.ready["loadedSourceCount"] == 25
        assert worker.source_load_count == 1

        reset = worker.reset(_SCENARIO)
        assert worker.pid
        assert worker.source_load_count == 1
        assert reset["schema"] == "curvytron-js-reuse-env-worker-v0"
        assert reset["runner"] == "original-curvytron-js-vm-ndjson-worker"
        assert reset["capabilities"]["persistentWorker"] is True
        assert reset["reset"] == batch["reset"]

        frames = [worker.step(step) for step in scenario["steps"]]
        assert worker.source_load_count == 1
        assert frames == [
            {**frame, "randomCalls": batch["randomCalls"]} for frame in batch["steps"]
        ]

        snapshot = worker.snapshot()
        assert worker.source_load_count == 1
        assert snapshot["state"] == batch["steps"][-1]["state"]
        assert snapshot["randomCalls"] == batch["randomCalls"]


def test_js_reuse_worker_runs_long_natural_1v1_rollout_to_round_done_without_reload():
    if shutil.which("node") is None:
        pytest.skip("node executable is not available")

    scenario = json.loads(_LONG_NATURAL_ROLLOUT_SCENARIO.read_text())
    rollout = scenario["rollout"]
    step_count = rollout["step_count"]

    with CurvytronJsEnvWorker() as worker:
        assert worker.source_load_count == 1

        reset = worker.reset(_LONG_NATURAL_ROLLOUT_SCENARIO)
        assert worker.source_load_count == 1
        assert reset["capabilities"]["resetCall"] == "source_new_round"
        assert reset["reset"]["mode"] == "source_new_round"
        assert reset["reset"]["state"]["game"]["worldActive"] is False
        assert _positions(reset["reset"]) == [[58, 44], [30, 44]]
        assert _headings(reset["reset"]) == [3.141593, 3.141593]
        assert [call["value"] for call in reset["randomCalls"]] == [
            0.32051282051282054,
            0.5,
            0.5,
            0.6794871794871795,
            0.5,
            0.5,
        ]

        frames = []
        for tick in range(step_count):
            frame = worker.step(
                {
                    "tick": tick,
                    "step_ms": rollout["step_ms"],
                    "advance_timers_ms": rollout["advance_timers_ms"],
                    "moves": rollout["moves"],
                }
            )
            frames.append(frame)
            assert worker.source_load_count == 1

        first = frames[0]
        assert first["events"][0]["event"] == "game:start"
        assert first["state"]["game"]["worldActive"] is True
        assert _positions(first) == [[57.733333, 44], [29.733333, 44]]

        penultimate = frames[-2]
        assert penultimate["tick"] == rollout["terminal_tick"] - 1
        assert penultimate["reward"] == [0, 0]
        assert penultimate["roundDone"] is False
        assert penultimate["gameDone"] is False
        assert _positions(penultimate) == [[28.666667, 44], [0.666667, 44]]
        assert _alive(penultimate) == [True, True]
        assert _scores(penultimate) == [0, 0]

        final = frames[-1]
        expected = rollout["expected"]
        assert final["tick"] == rollout["terminal_tick"]
        assert final["reward"] == expected["reward"]
        assert final["roundDone"] is expected["round_done"]
        assert final["gameDone"] is expected["game_done"]
        assert final["state"]["game"]["inRound"] is False
        assert final["state"]["game"]["deaths"] == expected["deaths"]
        assert final["state"]["game"]["roundWinner"] == expected["round_winner"]
        assert final["state"]["game"]["gameWinner"] is None
        assert _positions(final) == expected["final_positions"]
        assert _alive(final) == expected["alive"]
        assert _scores(final) == expected["scores"]
        assert final["randomCalls"] == reset["randomCalls"]
        assert {"event": "die", "data": {"player": 2, "killer": None, "old": None}} in final[
            "events"
        ]
        assert {"event": "round:end", "data": {"winner": 1}} in final["events"]

        # This is deliberately a round terminal proof, not a full source warmdown proof:
        # advancing the 5s warmdown would also run the source's free-running frame timer.
        snapshot = worker.snapshot()
        assert worker.source_load_count == 1
        assert snapshot["state"] == final["state"]
        assert snapshot["randomCalls"] == reset["randomCalls"]
