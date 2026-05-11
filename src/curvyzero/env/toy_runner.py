"""Python toy-v0 scenario runner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from curvyzero.env.config import CurvyTronConfig
from curvyzero.env.core import CurvyTronEnv
from curvyzero.env.scenario_schema import (
    PYTHON_SCENARIO_TRACE_SCHEMA,
    LoadedScenario,
    ScenarioError,
    _coerce_scenario,
    _extract_alive,
    _extract_headings,
    _extract_positions,
)
from curvyzero.env.tracing import TRACE_SCHEMA_VERSION, TRACE_SCOPE, EnvTrace, TraceFrame

PYTHON_SCENARIO_RUNNER = "curvyzero-v0-python-toy-scenario-runner"
TOY_V0_MESSAGE = (
    "Python toy-v0 trace only; this does not prove CurvyTron source fidelity."
)


@dataclass(frozen=True, slots=True)
class ScenarioRun:
    """Result of running one shared scenario through the Python toy env."""

    scenario: LoadedScenario
    trace: EnvTrace

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": PYTHON_SCENARIO_TRACE_SCHEMA,
            "runner": PYTHON_SCENARIO_RUNNER,
            "scenario_id": self.scenario.scenario_id,
            "source_ruleset_id": self.scenario.ruleset_id,
            "toy_ruleset_id": self.trace.ruleset,
            "provenance": self.scenario.provenance,
            "toy_v0_behavior": True,
            "source_fidelity": False,
            "message": TOY_V0_MESSAGE,
            "scenario": self.scenario.to_payload(),
            "trace_scope": self.trace.scope,
            "trace_schema_version": self.trace.schema_version,
            "rules_hash": self.trace.rules_hash,
            "trace_fingerprint": self.trace.fingerprint,
            "trace": self.trace.to_payload(),
        }


def run_scenario(scenario: LoadedScenario | Mapping[str, Any] | str | Path) -> ScenarioRun:
    """Run one supported scenario in the current Python toy-v0 environment."""

    loaded = _coerce_scenario(scenario)
    if loaded.player_count != 2:
        raise ScenarioError("toy-v0 scenario runner currently supports exactly 2 players")

    env = CurvyTronEnv(_config_for_scenario(loaded))
    trace = _trace_loaded_scenario(env, loaded)
    return ScenarioRun(scenario=loaded, trace=trace)


def _trace_loaded_scenario(env: CurvyTronEnv, scenario: LoadedScenario) -> EnvTrace:
    env.reset(seed=scenario.seed)
    _force_initial_state(env, scenario.initial_state)

    frames = [TraceFrame.from_env(env)]
    for actions in scenario.toy_action_script:
        step_result = env.step(actions)
        frames.append(TraceFrame.from_env(env, step_result))
        if any(step_result.terminated.values()) or any(step_result.truncated.values()):
            break

    return EnvTrace(
        scope=TRACE_SCOPE,
        schema_version=TRACE_SCHEMA_VERSION,
        ruleset=env.config.ruleset,
        rules_hash=env.config.rules_hash,
        seed=scenario.seed,
        scripted_actions=scenario.toy_action_script,
        frames=tuple(frames),
    )


def _force_initial_state(env: CurvyTronEnv, initial_state: Mapping[str, Any]) -> None:
    if not initial_state:
        return
    if env.state is None:
        raise RuntimeError("reset must be called before forcing initial state")

    positions = _extract_positions(initial_state)
    headings = _extract_headings(initial_state)
    alive = _extract_alive(initial_state)

    if positions is not None:
        env.state.positions = np.asarray(positions, dtype=np.float32)
    if headings is not None:
        env.state.headings = np.asarray(headings, dtype=np.float32)
    if alive is not None:
        env.state.alive = np.asarray(alive, dtype=np.bool_)

    if env.state.positions.shape != (2, 2):
        raise ScenarioError("toy-v0 forced positions must have shape [2][2]")
    if env.state.headings.shape != (2,):
        raise ScenarioError("toy-v0 forced headings must have length 2")
    if env.state.alive.shape != (2,):
        raise ScenarioError("toy-v0 forced alive flags must have length 2")

    env.state.death_tick[:] = -1
    env.state.occupancy[:, :] = 0
    env._mark_players()


def _config_for_scenario(scenario: LoadedScenario) -> CurvyTronConfig:
    width, height = _extract_size(scenario.initial_state)
    max_ticks = max(64, len(scenario.toy_action_script) + 1)
    return CurvyTronConfig(width=width, height=height, action_repeat=1, max_ticks=max_ticks)


def _extract_size(initial_state: Mapping[str, Any]) -> tuple[int, int]:
    map_size = initial_state.get("map_size", initial_state.get("size"))
    if map_size is not None:
        size = int(map_size)
        return size, size

    width = int(initial_state.get("width", CurvyTronConfig().width))
    height = int(initial_state.get("height", CurvyTronConfig().height))
    return width, height
