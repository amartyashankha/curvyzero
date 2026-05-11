"""Benchmark the source-shaped CurvyTron env on the long wall-death rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvyzero.env.source_env import CurvyTronSourceEnv  # noqa: E402
from curvyzero.fidelity.js_reuse_probe import CurvytronJsEnvWorker  # noqa: E402


DEFAULT_SCENARIO = (
    REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_long_1v1_no_bonus_wall_round_done.json"
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _load_scenario(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_source_env(scenario: dict[str, Any]) -> CurvyTronSourceEnv:
    source_setup = scenario["source_setup"]
    return CurvyTronSourceEnv(
        random_values=source_setup["random"]["math_random_sequence"],
        max_score=float(source_setup["room"]["max_score"]),
        include_deaths_snapshot=True,
    )


def _run_source_once(scenario: dict[str, Any]) -> dict[str, Any]:
    rollout = scenario["rollout"]
    env = _make_source_env(scenario)
    env.reset(
        player_count=int(scenario["player_count"]),
        players=scenario["players"],
        warmup_ms=float(scenario["lifecycle"]["new_round_time_ms"]),
    )
    frame = {}
    for _tick in range(int(rollout["step_count"])):
        env.advance_timers(float(rollout["advance_timers_ms"]))
        frame = env.step(rollout["moves"], elapsed_ms=float(rollout["step_ms"]))
    return frame


def _run_js_once(worker: CurvytronJsEnvWorker, path: Path, scenario: dict[str, Any]) -> dict[str, Any]:
    rollout = scenario["rollout"]
    worker.reset(path)
    frame = {}
    for tick in range(int(rollout["step_count"])):
        frame = worker.step(
            {
                "tick": tick,
                "step_ms": rollout["step_ms"],
                "advance_timers_ms": rollout["advance_timers_ms"],
                "moves": rollout["moves"],
            }
        )
    return frame


def _time_repeats(callback: Any, repeats: int) -> dict[str, Any]:
    durations = []
    final = None
    for _index in range(repeats):
        started = time.perf_counter()
        final = callback()
        durations.append(time.perf_counter() - started)
    total = sum(durations)
    return {
        "repeats": repeats,
        "total_sec": total,
        "mean_sec": total / repeats,
        "min_sec": min(durations),
        "max_sec": max(durations),
        "final": final,
    }


def _summary(frame: dict[str, Any], *, js: bool) -> dict[str, Any]:
    if js:
        state = frame["state"]
        players = state["players"]
        return {
            "positions": [[player["x"], player["y"]] for player in players],
            "alive": [player["alive"] for player in players],
            "scores": [player["score"] for player in players],
            "deaths": state["game"]["deaths"],
            "round_done": frame["roundDone"],
        }
    avatars = frame["avatars"]
    return {
        "positions": [[avatar["x"], avatar["y"]] for avatar in avatars],
        "alive": [avatar["alive"] for avatar in avatars],
        "scores": [avatar["score"] for avatar in avatars],
        "deaths": frame["game"]["deaths"],
        "round_done": not frame["game"]["inRound"],
    }


def _result(name: str, timing: dict[str, Any], step_count: int, *, js: bool) -> dict[str, Any]:
    final = timing.pop("final")
    steps = timing["repeats"] * step_count
    return {
        "name": name,
        **timing,
        "rollouts_per_sec": timing["repeats"] / timing["total_sec"],
        "steps_per_sec": steps / timing["total_sec"],
        "final": _summary(final, js=js),
    }


def _print_plain(results: list[dict[str, Any]]) -> None:
    for result in results:
        print(
            f"{result['name']}: repeats={result['repeats']} "
            f"mean={result['mean_sec']:.6f}s rollouts/s={result['rollouts_per_sec']:.2f} "
            f"steps/s={result['steps_per_sec']:.2f}"
        )
        print(f"  final={json.dumps(result['final'], sort_keys=True)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--repeats", type=_positive_int, default=10)
    parser.add_argument("--js-repeats", type=_positive_int, default=3)
    parser.add_argument("--js", action="store_true", help="also time the persistent JS worker")
    parser.add_argument("--json", action="store_true", help="print JSON instead of plain text")
    args = parser.parse_args()

    scenario_path = args.scenario.resolve()
    scenario = _load_scenario(scenario_path)
    step_count = int(scenario["rollout"]["step_count"])

    results = [
        _result(
            "python_source_env",
            _time_repeats(lambda: _run_source_once(scenario), args.repeats),
            step_count,
            js=False,
        )
    ]

    if args.js:
        if shutil.which("node") is None:
            raise SystemExit("node executable is not available")
        with CurvytronJsEnvWorker() as worker:
            results.append(
                _result(
                    "persistent_js_worker",
                    _time_repeats(lambda: _run_js_once(worker, scenario_path, scenario), args.js_repeats),
                    step_count,
                    js=True,
                )
            )

    payload = {
        "scenario": str(scenario_path.relative_to(REPO_ROOT)),
        "step_count": step_count,
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"scenario={payload['scenario']} step_count={step_count}")
        _print_plain(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
