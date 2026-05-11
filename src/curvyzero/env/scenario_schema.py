"""Shared scenario schema, parsing, and coercion helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from curvyzero.env.tracing import TRACE_SCHEMA_VERSION

PYTHON_SCENARIO_TRACE_SCHEMA = "curvyzero_python_scenario_trace/v1"

_SOURCE_MOVE_TO_TOY_ACTION = {-1: 0, 0: 1, 1: 2}


class ScenarioError(ValueError):
    """Raised when a scenario cannot be loaded or run by a scenario runner."""


@dataclass(frozen=True, slots=True)
class LoadedScenario:
    """Shared scenario fields plus normalized toy-env actions."""

    scenario_id: str
    ruleset_id: str
    player_count: int
    seed: int | None
    source_setup: Mapping[str, Any]
    initial_state: Mapping[str, Any]
    raw_action_script: tuple[dict[str, int], ...]
    toy_action_script: tuple[dict[str, int], ...]
    action_encoding: str
    time_policy: Mapping[str, Any]
    trace_schema_version: int | str
    tolerances: Mapping[str, Any]
    comparison: Mapping[str, Any]
    provenance: str

    def to_payload(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "ruleset_id": self.ruleset_id,
            "player_count": self.player_count,
            "seed": self.seed,
            "source_setup": dict(self.source_setup),
            "initial_state": dict(self.initial_state),
            "action_script": list(self.raw_action_script),
            "action_encoding": self.action_encoding,
            "toy_action_script": list(self.toy_action_script),
            "time_policy": dict(self.time_policy),
            "trace_schema_version": self.trace_schema_version,
            "tolerances": dict(self.tolerances),
            "comparison": dict(self.comparison),
            "provenance": self.provenance,
        }


def load_scenario(path: str | Path) -> LoadedScenario:
    """Load a JSON scenario file using the shared scenario shape."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ScenarioError("scenario JSON must be an object")
    return parse_scenario(payload)


def parse_scenario(payload: Mapping[str, Any]) -> LoadedScenario:
    """Parse the small shared scenario shape into a toy-runnable scenario."""

    scenario_id = _required_str(payload, "scenario_id", fallback_key="id")
    ruleset_id = _required_str(payload, "ruleset_id")
    source_setup = _optional_mapping(payload.get("source_setup"), "source_setup")
    comparison = _optional_mapping(payload.get("comparison"), "comparison")
    player_count = _required_int(payload, "player_count", fallback=source_setup.get("player_count"))
    seed = _optional_int(payload.get("seed"), "seed")
    initial_state = _scenario_initial_state(payload, source_setup)
    action_encoding = _normalize_action_encoding(
        str(payload.get("action_encoding", "source-move"))
    )
    raw_action_script = _parse_action_script(
        payload.get("action_script", payload.get("steps")),
        action_encoding,
        player_count,
    )
    toy_action_script = tuple(
        _to_toy_actions(step, action_encoding) for step in raw_action_script
    )
    time_policy = _scenario_time_policy(payload)
    trace_schema_version = payload.get(
        "trace_schema_version",
        comparison.get("trace_schema_version", TRACE_SCHEMA_VERSION),
    )
    tolerances = _optional_mapping(
        payload.get("tolerances", comparison.get("tolerances")),
        "tolerances",
    )
    provenance = _scenario_provenance(payload.get("provenance", "unresolved"))

    return LoadedScenario(
        scenario_id=scenario_id,
        ruleset_id=ruleset_id,
        player_count=player_count,
        seed=seed,
        source_setup=source_setup,
        initial_state=initial_state,
        raw_action_script=raw_action_script,
        toy_action_script=toy_action_script,
        action_encoding=action_encoding,
        time_policy=time_policy,
        trace_schema_version=trace_schema_version,
        tolerances=tolerances,
        comparison=comparison,
        provenance=provenance,
    )


def _scenario_initial_state(
    payload: Mapping[str, Any],
    source_setup: Mapping[str, Any],
) -> Mapping[str, Any]:
    initial_state = dict(_optional_mapping(payload.get("initial_state"), "initial_state"))
    if "map_size" not in initial_state and "map_size" in source_setup:
        initial_state["map_size"] = source_setup["map_size"]
    game_setup = source_setup.get("game")
    if "borderless" not in initial_state and isinstance(game_setup, Mapping):
        if "borderless" in game_setup:
            initial_state["borderless"] = bool(game_setup["borderless"])

    players = payload.get("players")
    if players is not None and "players" not in initial_state:
        initial_state["players"] = players
    if "players" in initial_state:
        initial_state["players"] = _scenario_players(initial_state["players"])
    world_bodies = payload.get("world_bodies")
    if world_bodies is not None and "world_bodies" not in initial_state:
        initial_state["world_bodies"] = world_bodies
    if "world_bodies" in initial_state:
        initial_state["world_bodies"] = _scenario_world_bodies(
            initial_state["world_bodies"]
        )
    return initial_state


def _scenario_players(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ScenarioError("initial_state.players must be a list")

    players: list[dict[str, Any]] = []
    for index, raw_player in enumerate(value):
        if not isinstance(raw_player, Mapping):
            raise ScenarioError("initial_state.players entries must be objects")
        player = dict(raw_player)
        _validate_player_body_counters(
            player, f"initial_state.players[{index}]"
        )
        for state_key in ("initial", "state"):
            if state_key not in player:
                continue
            state = player[state_key]
            if not isinstance(state, Mapping):
                raise ScenarioError(
                    f"initial_state.players[{index}].{state_key} must be an object"
                )
            state = dict(state)
            _validate_player_body_counters(
                state, f"initial_state.players[{index}].{state_key}"
            )
            player[state_key] = state
        players.append(player)
    return players


def _validate_player_body_counters(value: Mapping[str, Any], field: str) -> None:
    for key in ("body_count", "bodyCount", "body_num", "bodyNum"):
        if key in value:
            _required_body_num(value[key], f"{field}.{key}")


def _scenario_world_bodies(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ScenarioError("initial_state.world_bodies must be a list")

    bodies: list[dict[str, Any]] = []
    for index, raw_body in enumerate(value):
        if not isinstance(raw_body, Mapping):
            raise ScenarioError("initial_state.world_bodies entries must be objects")
        body = dict(raw_body)
        owner = _world_body_owner(body)
        if owner is None or owner == "":
            raise ScenarioError(
                f"initial_state.world_bodies[{index}].player_id is required"
            )
        _required_number(body.get("x"), f"initial_state.world_bodies[{index}].x")
        _required_number(body.get("y"), f"initial_state.world_bodies[{index}].y")
        _required_number(
            body.get("radius"), f"initial_state.world_bodies[{index}].radius"
        )
        _required_body_num(body.get("num"), f"initial_state.world_bodies[{index}].num")
        age_ms = _optional_world_body_age_ms(
            body, f"initial_state.world_bodies[{index}]"
        )
        if age_ms is not None:
            body["age_ms"] = age_ms
        bodies.append(body)
    return bodies


def _world_body_owner(body: Mapping[str, Any]) -> Any:
    for key in (
        "player_id",
        "playerId",
        "owner_id",
        "ownerId",
        "avatar_id",
        "avatarId",
        "avatar",
    ):
        if key in body:
            return body[key]
    return None


def _optional_world_body_age_ms(body: Mapping[str, Any], field: str) -> int | float | None:
    if "age_ms" in body:
        key = "age_ms"
    elif "ageMs" in body:
        key = "ageMs"
    else:
        return None
    value = body[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ScenarioError(f"{field}.{key} must be a non-negative finite number")
    if not math.isfinite(float(value)) or value < 0:
        raise ScenarioError(f"{field}.{key} must be a non-negative finite number")
    return value


def _scenario_time_policy(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    time_policy = _optional_mapping(payload.get("time_policy"), "time_policy")
    if time_policy:
        return time_policy

    steps = payload.get("steps")
    if isinstance(steps, list) and steps:
        first = steps[0]
        if isinstance(first, Mapping) and "step_ms" in first:
            return {"kind": "fixed", "step_ms": float(first["step_ms"])}
    return {}


def _scenario_provenance(value: Any) -> str:
    if isinstance(value, Mapping):
        labels = value.get("labels")
        if isinstance(labels, list) and labels:
            return ",".join(str(label) for label in labels)
    return str(value)


def _extract_positions(initial_state: Mapping[str, Any]) -> list[list[float]] | None:
    if "positions" in initial_state:
        return _two_float_pairs(initial_state["positions"], "initial_state.positions")

    players = initial_state.get("players")
    if players is None:
        return None
    if not isinstance(players, list) or len(players) < 2:
        raise ScenarioError("initial_state.players must contain at least 2 players")

    positions: list[list[float]] = [[0.0, 0.0] for _ in players]
    for index, player in enumerate(players):
        if not isinstance(player, Mapping):
            raise ScenarioError("initial_state.players entries must be objects")
        agent_index = _player_index(player.get("id"), index)
        initial = _optional_mapping(player.get("initial"), "player.initial")
        if initial and "x" in initial and "y" in initial:
            pair = [initial.get("x"), initial.get("y")]
        elif "position" in player:
            pair = player["position"]
        else:
            pair = [player.get("x"), player.get("y")]
        positions[agent_index] = _float_pair(pair, "player position")
    return positions


def _extract_headings(initial_state: Mapping[str, Any]) -> list[float] | None:
    if "headings" in initial_state:
        values = initial_state["headings"]
        if not isinstance(values, list) or len(values) < 2:
            raise ScenarioError("initial_state.headings must contain at least 2 values")
        return [float(value) for value in values]

    players = initial_state.get("players")
    if players is None:
        return None
    if not isinstance(players, list) or len(players) < 2:
        raise ScenarioError("initial_state.players must contain at least 2 players")

    headings: list[float] = [0.0 for _ in players]
    found = False
    for index, player in enumerate(players):
        if not isinstance(player, Mapping):
            raise ScenarioError("initial_state.players entries must be objects")
        initial = _optional_mapping(player.get("initial"), "player.initial")
        value = initial.get("angle_rad") if initial else None
        if value is None:
            value = player.get("heading", player.get("angle"))
        if value is None:
            continue
        found = True
        headings[_player_index(player.get("id"), index)] = float(value)
    return headings if found else None


def _extract_alive(initial_state: Mapping[str, Any]) -> list[bool] | None:
    if "alive" in initial_state:
        values = initial_state["alive"]
        if not isinstance(values, list) or len(values) < 2:
            raise ScenarioError("initial_state.alive must contain at least 2 values")
        return [bool(value) for value in values]

    players = initial_state.get("players")
    if players is None:
        return None
    if not isinstance(players, list) or len(players) < 2:
        raise ScenarioError("initial_state.players must contain at least 2 players")

    alive = [True for _ in players]
    found = False
    for index, player in enumerate(players):
        if not isinstance(player, Mapping):
            raise ScenarioError("initial_state.players entries must be objects")
        initial = _optional_mapping(player.get("initial"), "player.initial")
        if "alive" in initial:
            value = initial["alive"]
        elif "alive" in player:
            value = player["alive"]
        else:
            continue
        found = True
        alive[_player_index(player.get("id"), index)] = bool(value)
    return alive if found else None


def _scenario_player_ids(scenario: Mapping[str, Any]) -> tuple[str, ...]:
    initial_state = _optional_mapping(scenario.get("initial_state"), "scenario.initial_state")
    players = initial_state.get("players")
    if not isinstance(players, list):
        return ()
    ids = []
    for index, raw_player in enumerate(players):
        player = raw_player if isinstance(raw_player, Mapping) else {}
        ids.append(str(player.get("id") or player.get("player_id") or f"player_{index}"))
    return tuple(ids)


def _parse_action_script(
    value: Any,
    action_encoding: str,
    player_count: int,
) -> tuple[dict[str, int], ...]:
    if not isinstance(value, list) or not value:
        raise ScenarioError("action_script must be a non-empty list")

    steps: list[dict[str, int]] = []
    for step_index, raw_step in enumerate(value):
        step = _extract_step_actions(raw_step, step_index)
        normalized = _normalize_step_actions(step)
        expected = {f"player_{index}" for index in range(player_count)}
        if set(normalized) != expected:
            expected_text = ", ".join(sorted(expected))
            raise ScenarioError(f"action steps must include {expected_text}")
        _validate_actions(normalized, action_encoding)
        steps.append(dict(sorted(normalized.items())))
    return tuple(steps)


def _extract_step_actions(raw_step: Any, step_index: int) -> Mapping[str, Any] | list[Any]:
    if not isinstance(raw_step, Mapping):
        raise ScenarioError(f"action_script step {step_index} must be an object")
    if "actions" in raw_step:
        actions = raw_step["actions"]
    elif "moves" in raw_step:
        actions = raw_step["moves"]
    else:
        actions = raw_step
    if not isinstance(actions, Mapping) and not isinstance(actions, list):
        raise ScenarioError(f"action_script step {step_index} actions must be an object")
    return actions


def _normalize_step_actions(step: Mapping[str, Any] | list[Any]) -> dict[str, int]:
    if isinstance(step, list):
        moves: dict[str, int] = {}
        for index, move in enumerate(step):
            if not isinstance(move, Mapping):
                raise ScenarioError("list-form moves must contain objects")
            player_id = move.get("player_id", move.get("id"))
            value = move.get("move", move.get("action"))
            moves[_agent_id(player_id, index)] = int(value)
        return moves

    if "0" in step or "1" in step or "p0" in step or "player_0" in step:
        return {
            _agent_id(key, index): int(action)
            for index, (key, action) in enumerate(step.items())
        }

    moves: dict[str, int] = {}
    for index, move in enumerate(step.values()):
        if not isinstance(move, Mapping):
            continue
        player_id = move.get("player_id", move.get("id"))
        value = move.get("move", move.get("action"))
        moves[_agent_id(player_id, index)] = int(value)
    if moves:
        return moves

    return {
        _agent_id(key, index): int(action)
        for index, (key, action) in enumerate(step.items())
    }


def _validate_actions(actions: Mapping[str, int], action_encoding: str) -> None:
    valid = {-1, 0, 1} if action_encoding == "source-move" else {0, 1, 2}
    for action in actions.values():
        if action not in valid:
            raise ScenarioError(
                f"invalid {action_encoding} action {action!r}; expected one of {sorted(valid)}"
            )


def _to_toy_actions(actions: Mapping[str, int], action_encoding: str) -> dict[str, int]:
    if action_encoding == "toy-action":
        return {agent: int(action) for agent, action in sorted(actions.items())}
    return {
        agent: _SOURCE_MOVE_TO_TOY_ACTION[int(action)]
        for agent, action in sorted(actions.items())
    }


def _normalize_action_encoding(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "source": "source-move",
        "source-move": "source-move",
        "curvytron-move": "source-move",
        "js-move": "source-move",
        "toy": "toy-action",
        "toy-action": "toy-action",
        "curvyzero-v0-action": "toy-action",
    }
    if normalized not in aliases:
        raise ScenarioError(
            "action_encoding must be source-move or toy-action for the toy runner"
        )
    return aliases[normalized]


def _coerce_scenario(scenario: LoadedScenario | Mapping[str, Any] | str | Path) -> LoadedScenario:
    if isinstance(scenario, LoadedScenario):
        return scenario
    if isinstance(scenario, Mapping):
        return parse_scenario(scenario)
    return load_scenario(scenario)


def _required_str(payload: Mapping[str, Any], key: str, fallback_key: str | None = None) -> str:
    value = payload.get(key)
    if value is None and fallback_key is not None:
        value = payload.get(fallback_key)
    if not isinstance(value, str) or not value:
        raise ScenarioError(f"{key} must be a non-empty string")
    return value


def _required_int(payload: Mapping[str, Any], key: str, fallback: Any = None) -> int:
    value = payload.get(key, fallback)
    if not isinstance(value, int):
        raise ScenarioError(f"{key} must be an integer")
    return value


def _required_number(value: Any, key: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ScenarioError(f"{key} must be a number")
    return value


def _required_body_num(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScenarioError(f"{key} must be an integer")
    return value


def _optional_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ScenarioError(f"{key} must be an integer or null")
    return value


def _optional_mapping(value: Any, key: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ScenarioError(f"{key} must be an object")
    return value


def _two_float_pairs(value: Any, field: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) < 2:
        raise ScenarioError(f"{field} must contain at least 2 coordinate pairs")
    return [_float_pair(pair, field) for pair in value]


def _float_pair(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ScenarioError(f"{field} must be a coordinate pair")
    return [float(value[0]), float(value[1])]


def _agent_id(value: Any, fallback_index: int) -> str:
    return f"player_{_player_index(value, fallback_index)}"


def _player_index(value: Any, fallback_index: int) -> int:
    if value is None:
        return fallback_index
    text = str(value)
    aliases = {
        "player_0": 0,
        "p0": 0,
        "0": 0,
        "1": 0,
        "player_1": 1,
        "p1": 1,
        "2": 1,
        "player_2": 2,
        "p2": 2,
        "3": 2,
        "player_3": 3,
        "p3": 3,
        "4": 3,
    }
    if text in aliases:
        return aliases[text]
    raise ScenarioError(f"unsupported player id {value!r}")
