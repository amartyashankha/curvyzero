"""Seed vector-lane array state from existing environment scenario fixtures.

This is a bridge script only. It reads forced fixture setup and action schedules,
then emits JSON arrays that match the draft vector state shape. It does not step
the environment and it does not compare against JS/Python common traces.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvyzero.env.config import CurvyTronReferenceDefaults  # noqa: E402


SCHEMA_VERSION = "curvyzero_vector_fixture_seed/v1"
FIXTURE_SEED_SUPPORT_SCHEMA = "curvyzero_vector_fixture_seed_support/v1"
RANDOM_TAPE_CONTRACT_SCHEMA = "curvyzero_vector_random_tape_contract/v1"
LIFECYCLE_SEED_REJECTION_SCHEMA = "curvyzero_vector_lifecycle_seed_rejection/v1"
SOURCE_FIDELITY_CLAIM = (
    "initial fixture array seed only; no stepping, no source runner execution, "
    "and no common-trace equivalence claim"
)
BODY_KIND_SEEDED = 3
DEFAULT_RANDOM_TAPE_CAPACITY = 8
RANDOM_TAPE_SITE_CODES = {
    "spawn.position_x": 1,
    "spawn.position_y": 2,
    "spawn.angle_attempt_0": 3,
    "spawn.angle_attempt_1": 4,
    "print_manager.start_distance": 5,
    "print_manager.stop_distance": 6,
}
LIFECYCLE_SEED_REJECTION_REASON = (
    "natural Game.newRound() lifecycle fixtures are not ordinary initial_state "
    "one-step fixtures; vector seeding needs an explicit reset/spawn/timer path first"
)
LIFECYCLE_MISSING_VECTOR_CONTRACT = [
    "row reset lifecycle metadata",
    "spawn RNG draw order, site, avatar, and retry metadata",
    "spawned position and heading arrays from Game.newRound()",
    "heading rejection retry loop",
    "delayed PrintManager start timer setup",
    "terminal final-observation and replay handoff",
]


class SeedError(ValueError):
    """Raised when fixture setup cannot be converted into seed arrays."""

    def __init__(self, message: str, *, detail: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = detail


def seed_inputs(paths: Sequence[str | Path], *, body_capacity: int | None = None) -> dict[str, Any]:
    scenario_paths = _expand_input_paths(paths)
    fixtures = [
        seed_fixture(path, body_capacity=body_capacity)
        for path in scenario_paths
    ]
    return {
        "schema": SCHEMA_VERSION,
        "source_fidelity_claim": SOURCE_FIDELITY_CLAIM,
        "trust_level": (
            "Use this output as fixture-to-array plumbing only. The next gate is "
            "array stepping plus common-trace comparison."
        ),
        "input_count": len(scenario_paths),
        "fixture_count": len(fixtures),
        "batch_axis_status": (
            "Each fixture is emitted as B=1. Multi-fixture stacking is not done "
            "because current fixtures use different player counts and body caps."
        ),
        "fixtures": fixtures,
        "stackable_groups": _stackable_groups(fixtures),
        "global_limitations": [
            "does not run JS oracle traces",
            "does not run Python source-fidelity runners",
            "does not advance movement, trails, collisions, print-manager state, or scoring",
            "does not infer expected post-step common-trace rows",
            "serializes arrays as JSON lists, not NumPy .npy/.npz payloads",
        ],
        "batched_self_play_status": {
            "status": "not_supported_by_this_script",
            "missing_before_self_play": [
                "pure array transition step",
                "batched reset and autoreset",
                "production RNG streams beyond source Math.random tape arrays",
                "observation and reward arrays",
                "done/truncated/env-active masks",
                "policy/model batch-call boundary",
                "MuZero/MCTS search tree arrays",
                "common-trace comparison for array-step outputs",
            ],
        },
    }


def fixture_seed_support(path: str | Path) -> dict[str, Any]:
    """Return whether a fixture can be honestly converted into initial arrays."""

    fixture_path = _resolve_path(path)
    payload = _read_json_object(fixture_path)
    return _fixture_seed_support(payload, fixture_path)


def seed_fixture(path: str | Path, *, body_capacity: int | None = None) -> dict[str, Any]:
    fixture_path = _resolve_path(path)
    payload = _read_json_object(fixture_path)
    support = _fixture_seed_support(payload, fixture_path)
    if not bool(support["supported"]):
        raise SeedError(str(support["reason"]), detail=support)

    reference = CurvyTronReferenceDefaults()

    scenario_id = _scenario_id(payload)
    source_setup = _mapping(payload.get("source_setup"), "source_setup", default={})
    initial_state = _initial_state(payload, source_setup)
    players = _players(payload, initial_state)
    player_count = _player_count(payload, source_setup, players)
    player_ids, player_lookup = _player_ids_and_lookup(players, player_count)

    positions, headings, alive, death_tick, printing, score, round_score = _player_arrays(
        players,
        initial_state,
        player_count,
        player_lookup,
    )
    seeded_bodies = _seeded_bodies(initial_state, player_lookup)
    body_count, live_body_num = _body_counters(
        players,
        initial_state,
        player_count,
        player_lookup,
        seeded_bodies,
    )
    trail_state = _trail_state(players, player_count, player_lookup)
    print_manager_state = _print_manager_state(players, player_count, player_lookup)
    random_tape_state = _random_tape_state(source_setup)
    action_schedule = _action_schedule(payload, player_count, player_lookup, reference)

    body_capacity_value = _body_capacity(body_capacity, seeded_bodies)
    body_arrays = _body_arrays(seeded_bodies, body_capacity_value)
    map_size = _map_size(payload, source_setup, initial_state)
    borderless = _borderless(source_setup, initial_state)
    labels = _provenance_labels(payload)
    comparison = _mapping(payload.get("comparison"), "comparison", default={})

    arrays = {
        "tick": _array([0], "int32"),
        "done": _array([False], "bool"),
        "overflow": _array([False], "bool"),
        "borderless": _array([borderless], "bool"),
        "map_size": _array([map_size], "float64"),
        "pos": _array([positions], "float64"),
        "prev_pos": _array([positions], "float64"),
        "heading": _array([headings], "float64"),
        "alive": _array([alive], "bool"),
        "death_tick": _array([death_tick], "int32"),
        "score": _array([score], "int32"),
        "round_score": _array([round_score], "int32"),
        "printing": _array([printing], "bool"),
        "radius": _array([[reference.avatar_radius for _ in range(player_count)]], "float64"),
        "speed": _array(
            [[reference.avatar_velocity_units_per_s for _ in range(player_count)]],
            "float64",
        ),
        "angular_velocity_per_ms": _array(
            [[reference.angular_velocity_radians_per_ms for _ in range(player_count)]],
            "float64",
        ),
        "trail_latency": _array(
            [[reference.trail_latency_points for _ in range(player_count)]],
            "int16",
        ),
        "body_count": _array([body_count], "int32"),
        "live_body_num": _array([live_body_num], "int32"),
        "visible_trail_count": _array([trail_state["visible_trail_count"]], "int32"),
        "has_visible_trail_last": _array(
            [trail_state["has_visible_trail_last"]],
            "bool",
        ),
        "visible_trail_last_pos": _array(
            [trail_state["visible_trail_last_pos"]],
            "float64",
        ),
        "has_draw_cursor": _array([trail_state["has_draw_cursor"]], "bool"),
        "draw_cursor_pos": _array([trail_state["draw_cursor_pos"]], "float64"),
        "print_manager_active": _array([print_manager_state["active"]], "bool"),
        "print_manager_distance": _array([print_manager_state["distance"]], "float64"),
        "print_manager_last_pos": _array([print_manager_state["last_pos"]], "float64"),
        "random_tape_values": _array(
            [random_tape_state["values"]],
            "float64",
            shape=[1, DEFAULT_RANDOM_TAPE_CAPACITY],
        ),
        "random_tape_length": _array([random_tape_state["length"]], "int32"),
        "random_tape_cursor": _array([0], "int32"),
        "random_tape_exhausted": _array([False], "bool"),
        "random_tape_draw_count": _array([0], "int32"),
        **body_arrays,
    }

    return {
        "scenario_id": scenario_id,
        "path": _display_path(fixture_path),
        "schema_version": payload.get("schema_version"),
        "ruleset_id": payload.get("ruleset_id"),
        "provenance_labels": labels,
        "verification": {
            "js_oracle_pinned": "js-oracle-pinned" in labels,
            "python_runner_verified": "python-runner-verified" in labels,
            "source_fidelity_required": bool(comparison.get("source_fidelity_required", False)),
            "python_target": comparison.get("python_target"),
            "status_note": _verification_status_note(labels),
        },
        "profile": {
            "B": 1,
            "P": player_count,
            "K": body_capacity_value,
            "player_ids": list(player_ids),
            "update_order": list(range(player_count - 1, -1, -1)),
            "border_mode": "borderless-wrap" if borderless else "normal-wall",
            "map_size": map_size,
            "body_capacity_source": (
                "cli --body-capacity"
                if body_capacity is not None
                else "seeded body count only; production profiles must choose K"
            ),
        },
        "movement_scalars": {
            "speed_units_per_s": reference.avatar_velocity_units_per_s,
            "angular_velocity_radians_per_ms": reference.angular_velocity_radians_per_ms,
            "avatar_radius": reference.avatar_radius,
            "trail_latency_points": reference.trail_latency_points,
            "source_move_values": {"left": -1, "straight": 0, "right": 1},
            "per_step": action_schedule["per_step_scalars"],
        },
        "action_schedule": action_schedule["schedule"],
        "seeded_fixture_state": {
            "world_bodies": seeded_bodies,
            "visible_trail_points": trail_state["visible_trail_points"],
            "print_managers_present": print_manager_state["present"],
            "random_tape": {
                "capacity": DEFAULT_RANDOM_TAPE_CAPACITY,
                "length": random_tape_state["length"],
                "values": random_tape_state["source_values"],
            },
        },
        "arrays": arrays,
        "comparison_bridge": {
            "common_trace_status": "not_run",
            "next_step": (
                "Run an array step from these initial arrays and compare projected "
                "post-step x, y, angle, alive, trail/body counts, printing, "
                "worldBodyCount, and events against common traces."
            ),
            "fields_seeded_for_future_compare": [
                "x",
                "y",
                "angle",
                "alive",
                "printing",
                "trailPointCount",
                "lastTrailPoint",
                "bodyNum",
                "bodyCount",
                "worldBodyCount",
                "printManager",
                "random_tape_cursor",
                "random_tape_draw_count",
            ],
        },
        "limitations": _fixture_limitations(payload, seeded_bodies, trail_state, print_manager_state),
    }


def _fixture_seed_support(payload: Mapping[str, Any], fixture_path: Path) -> dict[str, Any]:
    scenario_id = _scenario_id(payload)
    source_setup = _mapping(payload.get("source_setup"), "source_setup", default={})
    labels = _provenance_labels(payload)
    random_contract = _random_tape_contract(payload, source_setup)

    if _is_natural_lifecycle_fixture(payload, scenario_id, labels):
        return {
            "schema": FIXTURE_SEED_SUPPORT_SCHEMA,
            "scenario_id": scenario_id,
            "path": _display_path(fixture_path),
            "supported": False,
            "reason": LIFECYCLE_SEED_REJECTION_REASON,
            "unsupported_kind": "natural_game_new_round_lifecycle",
            "random_tape_contract": random_contract,
            "lifecycle_rejection": _lifecycle_seed_rejection(payload),
        }

    return {
        "schema": FIXTURE_SEED_SUPPORT_SCHEMA,
        "scenario_id": scenario_id,
        "path": _display_path(fixture_path),
        "supported": True,
        "reason": None,
        "unsupported_kind": None,
        "random_tape_contract": random_contract,
    }


def _is_natural_lifecycle_fixture(
    payload: Mapping[str, Any],
    scenario_id: str,
    labels: Sequence[str],
) -> bool:
    if scenario_id.startswith("source_lifecycle_"):
        return True
    if "round-lifecycle" in labels and "natural-spawn" in labels:
        return True
    return isinstance(payload.get("lifecycle"), Mapping) and "natural-spawn" in labels


def _lifecycle_seed_rejection(payload: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = _mapping(payload.get("lifecycle"), "lifecycle", default={})
    return {
        "schema": LIFECYCLE_SEED_REJECTION_SCHEMA,
        "ordinary_seed_shape": "initial_state_one_step",
        "required_shape": "reset_many -> spawn_rng -> timer_setup -> final_obs_replay",
        "new_round_time_ms": lifecycle.get("new_round_time_ms", lifecycle.get("newRoundTimeMs")),
        "timer_advances_ms": lifecycle.get("timer_advances_ms", lifecycle.get("timerAdvancesMs")),
        "has_lifecycle_actions": isinstance(lifecycle.get("actions"), list),
        "missing_vector_contract": list(LIFECYCLE_MISSING_VECTOR_CONTRACT),
    }


def _random_tape_contract(
    payload: Mapping[str, Any],
    source_setup: Mapping[str, Any],
) -> dict[str, Any]:
    values = _random_tape_values(source_setup)
    calls = _expected_random_calls(payload)
    return {
        "schema": RANDOM_TAPE_CONTRACT_SCHEMA,
        "capacity": DEFAULT_RANDOM_TAPE_CAPACITY,
        "length": len(values),
        "minimum_capacity": max(DEFAULT_RANDOM_TAPE_CAPACITY, len(values)),
        "exceeds_seed_capacity": len(values) > DEFAULT_RANDOM_TAPE_CAPACITY,
        "values": values,
        "expected_call_count": len(calls),
        "expected_call_indices": [call["index"] for call in calls],
        "expected_call_sites": [call["site"] for call in calls],
        "expected_call_site_codes": [call["site_code"] for call in calls],
        "expected_call_avatars": [call["avatar"] for call in calls],
        "expected_call_values": [call["value"] for call in calls],
        "expected_call_at_ms": [call["at_ms"] for call in calls],
        "expected_calls": calls,
        "metadata_matches_tape": _random_call_metadata_matches_tape(calls, values),
        "site_codes": dict(RANDOM_TAPE_SITE_CODES),
    }


def _random_call_metadata_matches_tape(
    calls: Sequence[Mapping[str, Any]],
    values: Sequence[float],
) -> bool:
    if len(calls) != len(values):
        return False
    return all(
        int(call["index"]) == index and float(call["value"]) == values[index]
        for index, call in enumerate(calls)
    )


def _expected_random_calls(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    expectations = _mapping(payload.get("expectations"), "expectations", default={})
    event_order = expectations.get("event_order", [])
    if event_order is None:
        return []
    if not isinstance(event_order, list):
        raise SeedError("expectations.event_order must be a list")

    calls: list[dict[str, Any]] = []
    for event_order_index, raw_event in enumerate(event_order):
        if not isinstance(raw_event, Mapping):
            raise SeedError(f"expectations.event_order[{event_order_index}] must be an object")
        if raw_event.get("event") != "random":
            continue
        data = _mapping(
            raw_event.get("data"),
            f"expectations.event_order[{event_order_index}].data",
        )
        site = _non_empty_string(
            data.get("site"),
            f"expectations.event_order[{event_order_index}].data.site",
        )
        calls.append(
            {
                "event_order_index": event_order_index,
                "index": _int(
                    data.get("index"),
                    f"expectations.event_order[{event_order_index}].data.index",
                ),
                "value": _random_tape_value(
                    data.get("value"),
                    f"expectations.event_order[{event_order_index}].data.value",
                ),
                "site": site,
                "site_code": RANDOM_TAPE_SITE_CODES.get(site, -1),
                "avatar": _int(
                    data.get("avatar", data.get("avatar_id", data.get("avatarId"))),
                    f"expectations.event_order[{event_order_index}].data.avatar",
                ),
                "at_ms": _optional_at_ms(raw_event, event_order_index),
            }
        )
    return calls


def _optional_at_ms(event: Mapping[str, Any], event_order_index: int) -> float | None:
    if "atMs" not in event:
        return None
    value = event["atMs"]
    result = _finite_number(value, f"expectations.event_order[{event_order_index}].atMs")
    if result < 0.0:
        raise SeedError(f"expectations.event_order[{event_order_index}].atMs must be non-negative")
    return result


def _expand_input_paths(paths: Sequence[str | Path]) -> list[Path]:
    if not paths:
        raise SeedError("at least one scenario or batch path is required")
    scenario_paths: list[Path] = []
    for raw_path in paths:
        path = _resolve_path(raw_path)
        payload = _read_json_object(path)
        entries = payload.get("scenarios")
        if isinstance(entries, list):
            for index, entry in enumerate(entries):
                scenario_paths.append(_scenario_path_from_entry(entry, path.parent, index))
        else:
            scenario_paths.append(path)
    return scenario_paths


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SeedError(f"{path} must contain a JSON object")
    return payload


def _resolve_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    cwd_path = raw.resolve()
    if cwd_path.exists():
        return cwd_path
    repo_path = (REPO_ROOT / raw).resolve()
    if repo_path.exists():
        return repo_path
    raise SeedError(f"path does not exist: {path}")


def _scenario_path_from_entry(entry: Any, manifest_dir: Path, index: int) -> Path:
    if isinstance(entry, str):
        value = entry
    elif isinstance(entry, Mapping):
        value = entry.get("path", entry.get("scenario_path"))
    else:
        raise SeedError(f"batch scenario entry {index} must be a string or object")
    if not isinstance(value, str) or not value.strip():
        raise SeedError(f"batch scenario entry {index} must include a path")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    manifest_path = (manifest_dir / path).resolve()
    if manifest_path.exists():
        return manifest_path
    repo_path = (REPO_ROOT / path).resolve()
    if repo_path.exists():
        return repo_path
    raise SeedError(f"batch scenario entry {index} path does not exist: {value}")


def _scenario_id(payload: Mapping[str, Any]) -> str:
    value = payload.get("scenario_id", payload.get("id"))
    if not isinstance(value, str) or not value:
        raise SeedError("scenario_id or id must be a non-empty string")
    return value


def _initial_state(
    payload: Mapping[str, Any],
    source_setup: Mapping[str, Any],
) -> dict[str, Any]:
    initial_state = dict(_mapping(payload.get("initial_state"), "initial_state", default={}))
    if "players" not in initial_state and isinstance(payload.get("players"), list):
        initial_state["players"] = payload["players"]
    if "world_bodies" not in initial_state and isinstance(payload.get("world_bodies"), list):
        initial_state["world_bodies"] = payload["world_bodies"]
    if "map_size" not in initial_state and "map_size" in source_setup:
        initial_state["map_size"] = source_setup["map_size"]
    game = _mapping(source_setup.get("game"), "source_setup.game", default={})
    if "borderless" not in initial_state and "borderless" in game:
        initial_state["borderless"] = game["borderless"]
    return initial_state


def _players(payload: Mapping[str, Any], initial_state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = initial_state.get("players", payload.get("players"))
    if isinstance(value, list):
        players = []
        for index, raw_player in enumerate(value):
            if not isinstance(raw_player, Mapping):
                raise SeedError(f"players[{index}] must be an object")
            players.append(raw_player)
        return players
    return []


def _player_count(
    payload: Mapping[str, Any],
    source_setup: Mapping[str, Any],
    players: Sequence[Mapping[str, Any]],
) -> int:
    value = payload.get("player_count", source_setup.get("player_count"))
    if value is None:
        value = len(players)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SeedError("player_count must be a positive integer")
    if players and len(players) < value:
        raise SeedError(f"player_count is {value}, but only {len(players)} players are present")
    return value


def _player_ids_and_lookup(
    players: Sequence[Mapping[str, Any]],
    player_count: int,
) -> tuple[tuple[str, ...], dict[str, int]]:
    player_ids: list[str] = []
    lookup: dict[str, int] = {}
    for index in range(player_count):
        raw_player = players[index] if index < len(players) else {}
        player_id = str(
            raw_player.get("id")
            or raw_player.get("player_id")
            or raw_player.get("name")
            or f"player_{index}"
        )
        player_ids.append(player_id)
        aliases = {
            player_id,
            f"player_{index}",
            f"p{index}",
            str(index),
        }
        avatar_id = raw_player.get("avatar_id", raw_player.get("avatarId"))
        if avatar_id is not None:
            aliases.add(str(avatar_id))
        name = raw_player.get("name")
        if name is not None:
            aliases.add(str(name))
        for alias in aliases:
            lookup[alias] = index
    return tuple(player_ids), lookup


def _player_arrays(
    players: Sequence[Mapping[str, Any]],
    initial_state: Mapping[str, Any],
    player_count: int,
    player_lookup: Mapping[str, int],
) -> tuple[
    list[list[float]],
    list[float],
    list[bool],
    list[int],
    list[bool],
    list[int],
    list[int],
]:
    positions = _default_positions(initial_state, player_count)
    headings = _default_scalar_list(initial_state.get("headings"), player_count, 0.0)
    alive = _default_bool_list(initial_state.get("alive"), player_count, True)
    death_tick = _default_int_list(initial_state.get("death_tick"), player_count, -1)
    printing = [False for _ in range(player_count)]
    score = _default_int_list(initial_state.get("scores"), player_count, 0)
    round_score = _default_int_list(initial_state.get("roundScores"), player_count, 0)

    for fallback_index, raw_player in enumerate(players[:player_count]):
        state = _player_state(raw_player)
        index = _lookup_player_index(raw_player.get("id", raw_player.get("name")), fallback_index, player_lookup)
        x = state.get("x", raw_player.get("x"))
        y = state.get("y", raw_player.get("y"))
        if x is not None or y is not None:
            positions[index] = [
                _finite_number(x, f"players[{fallback_index}].initial.x"),
                _finite_number(y, f"players[{fallback_index}].initial.y"),
            ]
        heading = state.get("angle_rad", state.get("angle", raw_player.get("heading")))
        if heading is not None:
            headings[index] = _finite_number(
                heading,
                f"players[{fallback_index}].initial.angle_rad",
            )
        if "alive" in state:
            alive[index] = _bool(state["alive"], f"players[{fallback_index}].initial.alive")
        if "death_tick" in state or "deathTick" in state:
            death_tick[index] = _int(
                state.get("death_tick", state.get("deathTick")),
                f"players[{fallback_index}].initial.death_tick",
            )
        if "printing" in state:
            printing[index] = _bool(
                state["printing"],
                f"players[{fallback_index}].initial.printing",
            )
        if "score" in state:
            score[index] = _int(state["score"], f"players[{fallback_index}].initial.score")
        if "roundScore" in state or "round_score" in state:
            round_score[index] = _int(
                state.get("roundScore", state.get("round_score")),
                f"players[{fallback_index}].initial.roundScore",
            )
    return positions, headings, alive, death_tick, printing, score, round_score


def _default_positions(initial_state: Mapping[str, Any], player_count: int) -> list[list[float]]:
    raw_positions = initial_state.get("positions")
    if raw_positions is None:
        return [[0.0, 0.0] for _ in range(player_count)]
    if not isinstance(raw_positions, list) or len(raw_positions) < player_count:
        raise SeedError("initial_state.positions must include one pair per player")
    return [
        _point(raw_positions[index], f"initial_state.positions[{index}]")
        for index in range(player_count)
    ]


def _seeded_bodies(
    initial_state: Mapping[str, Any],
    player_lookup: Mapping[str, int],
) -> list[dict[str, Any]]:
    raw_bodies = initial_state.get("world_bodies", [])
    if raw_bodies is None:
        raw_bodies = []
    if not isinstance(raw_bodies, list):
        raise SeedError("initial_state.world_bodies must be a list")
    seeded = []
    for index, raw_body in enumerate(raw_bodies):
        if not isinstance(raw_body, Mapping):
            raise SeedError(f"initial_state.world_bodies[{index}] must be an object")
        owner_value = _first_present(
            raw_body,
            ("player_id", "playerId", "owner_id", "ownerId", "avatar_id", "avatarId", "avatar"),
        )
        owner_index = _lookup_player_index(
            owner_value,
            index,
            player_lookup,
            field=f"initial_state.world_bodies[{index}].player_id",
        )
        seeded.append(
            {
                "slot": index,
                "active": True,
                "owner_index": owner_index,
                "owner_ref": str(owner_value),
                "x": _finite_number(raw_body.get("x"), f"initial_state.world_bodies[{index}].x"),
                "y": _finite_number(raw_body.get("y"), f"initial_state.world_bodies[{index}].y"),
                "radius": _finite_number(
                    raw_body.get("radius"),
                    f"initial_state.world_bodies[{index}].radius",
                ),
                "num": _int(raw_body.get("num"), f"initial_state.world_bodies[{index}].num"),
                "age_ms": _optional_nonnegative_number(
                    _first_present(raw_body, ("age_ms", "ageMs")),
                    f"initial_state.world_bodies[{index}].age_ms",
                ),
                "insert_kind": "seeded",
            }
        )
    return seeded


def _body_counters(
    players: Sequence[Mapping[str, Any]],
    initial_state: Mapping[str, Any],
    player_count: int,
    player_lookup: Mapping[str, int],
    seeded_bodies: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[int]]:
    body_count = _default_int_list(initial_state.get("bodyCounts"), player_count, 0)
    live_body_num = _default_int_list(initial_state.get("bodyNums"), player_count, 0)
    for body in seeded_bodies:
        owner_index = _int(body["owner_index"], "seeded body owner_index")
        body_num = _int(body["num"], "seeded body num")
        body_count[owner_index] = max(body_count[owner_index], body_num + 1)

    for fallback_index, raw_player in enumerate(players[:player_count]):
        state = _player_state(raw_player)
        index = _lookup_player_index(raw_player.get("id", raw_player.get("name")), fallback_index, player_lookup)
        player_body_count = _optional_int_from_keys(state, ("body_count", "bodyCount"))
        player_body_num = _optional_int_from_keys(state, ("body_num", "bodyNum"))
        if player_body_count is not None:
            body_count[index] = player_body_count
            live_body_num[index] = player_body_count
        if player_body_num is not None:
            live_body_num[index] = player_body_num
    return body_count, live_body_num


def _trail_state(
    players: Sequence[Mapping[str, Any]],
    player_count: int,
    player_lookup: Mapping[str, int],
) -> dict[str, Any]:
    visible_count = [0 for _ in range(player_count)]
    has_visible_last = [False for _ in range(player_count)]
    visible_last_pos = [[0.0, 0.0] for _ in range(player_count)]
    has_draw_cursor = [False for _ in range(player_count)]
    draw_cursor_pos = [[0.0, 0.0] for _ in range(player_count)]
    visible_points: list[list[list[float]]] = [[] for _ in range(player_count)]

    for fallback_index, raw_player in enumerate(players[:player_count]):
        state = _player_state(raw_player)
        trail = state.get("trail", state.get("trail_state"))
        if trail is None:
            continue
        if not isinstance(trail, Mapping):
            raise SeedError(f"players[{fallback_index}].initial.trail must be an object")
        index = _lookup_player_index(raw_player.get("id", raw_player.get("name")), fallback_index, player_lookup)
        points = _trail_points(trail.get("points"), fallback_index)
        visible_points[index] = points
        visible_count[index] = len(points)
        if points:
            has_visible_last[index] = True
            visible_last_pos[index] = points[-1]
            has_draw_cursor[index] = True
            draw_cursor_pos[index] = points[-1]
        last_x = trail.get("last_x", trail.get("lastX"))
        last_y = trail.get("last_y", trail.get("lastY"))
        if last_x is not None or last_y is not None:
            draw_cursor_pos[index] = [
                _finite_number(last_x, f"players[{fallback_index}].initial.trail.last_x"),
                _finite_number(last_y, f"players[{fallback_index}].initial.trail.last_y"),
            ]
            has_draw_cursor[index] = True

    return {
        "visible_trail_count": visible_count,
        "has_visible_trail_last": has_visible_last,
        "visible_trail_last_pos": visible_last_pos,
        "has_draw_cursor": has_draw_cursor,
        "draw_cursor_pos": draw_cursor_pos,
        "visible_trail_points": visible_points,
    }


def _print_manager_state(
    players: Sequence[Mapping[str, Any]],
    player_count: int,
    player_lookup: Mapping[str, int],
) -> dict[str, Any]:
    active = [False for _ in range(player_count)]
    distance = [0.0 for _ in range(player_count)]
    last_pos = [[0.0, 0.0] for _ in range(player_count)]
    present = [False for _ in range(player_count)]

    for fallback_index, raw_player in enumerate(players[:player_count]):
        state = _player_state(raw_player)
        manager = state.get("print_manager", state.get("printManager"))
        if manager is None:
            continue
        if not isinstance(manager, Mapping):
            raise SeedError(f"players[{fallback_index}].initial.print_manager must be an object")
        index = _lookup_player_index(raw_player.get("id", raw_player.get("name")), fallback_index, player_lookup)
        present[index] = True
        active[index] = _bool(
            manager.get("active"),
            f"players[{fallback_index}].initial.print_manager.active",
        )
        distance[index] = _finite_number(
            manager.get("distance"),
            f"players[{fallback_index}].initial.print_manager.distance",
        )
        last_pos[index] = [
            _finite_number(
                manager.get("last_x", manager.get("lastX")),
                f"players[{fallback_index}].initial.print_manager.last_x",
            ),
            _finite_number(
                manager.get("last_y", manager.get("lastY")),
                f"players[{fallback_index}].initial.print_manager.last_y",
            ),
        ]
    return {"active": active, "distance": distance, "last_pos": last_pos, "present": present}


def _random_tape_values(source_setup: Mapping[str, Any]) -> list[float]:
    random_setup = _mapping(source_setup.get("random"), "source_setup.random", default={})
    raw_sequence = _first_present(
        random_setup,
        (
            "math_random_sequence",
            "mathRandomSequence",
            "math_random_tape",
            "mathRandomTape",
        ),
    )
    if raw_sequence is None:
        values: list[float] = []
    else:
        if not isinstance(raw_sequence, list):
            raise SeedError("source_setup.random.math_random_sequence must be a list")
        values = [
            _random_tape_value(
                value,
                f"source_setup.random.math_random_sequence[{index}]",
            )
            for index, value in enumerate(raw_sequence)
        ]
    return values


def _random_tape_state(source_setup: Mapping[str, Any]) -> dict[str, Any]:
    values = _random_tape_values(source_setup)
    if len(values) > DEFAULT_RANDOM_TAPE_CAPACITY:
        raise SeedError(
            "source_setup.random.math_random_sequence has "
            f"{len(values)} entries, but vector seed capacity is "
            f"{DEFAULT_RANDOM_TAPE_CAPACITY}"
        )
    padded = [*values, *([0.0] * (DEFAULT_RANDOM_TAPE_CAPACITY - len(values)))]
    return {"values": padded, "source_values": values, "length": len(values)}


def _random_tape_value(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SeedError(f"{field} must be a finite number in [0, 1)")
    result = float(value)
    if not 0.0 <= result < 1.0:
        raise SeedError(f"{field} must be a finite number in [0, 1)")
    return result


def _action_schedule(
    payload: Mapping[str, Any],
    player_count: int,
    player_lookup: Mapping[str, int],
    reference: CurvyTronReferenceDefaults,
) -> dict[str, Any]:
    raw_steps = payload.get("steps", payload.get("action_script"))
    if raw_steps is None:
        raw_steps = []
    if not isinstance(raw_steps, list):
        raise SeedError("steps/action_script must be a list")
    time_policy = _mapping(payload.get("time_policy"), "time_policy", default={})
    default_step_ms = time_policy.get("step_ms", reference.tick_ms)

    schedule = []
    per_step_scalars = []
    for step_index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            raise SeedError(f"steps[{step_index}] must be an object")
        step_ms = _finite_number(
            raw_step.get("step_ms", default_step_ms),
            f"steps[{step_index}].step_ms",
        )
        moves = _moves_for_step(raw_step, player_count, player_lookup, step_index)
        schedule_step = {
            "step_index": step_index,
            "tick": _int(raw_step.get("tick", step_index), f"steps[{step_index}].tick"),
            "step_ms": step_ms,
            "source_moves": moves,
        }
        timer_advance_ms = raw_step.get(
            "timer_advance_ms",
            raw_step.get("advance_timers_ms"),
        )
        if timer_advance_ms is not None:
            schedule_step["timer_advance_ms"] = _finite_number(
                timer_advance_ms,
                f"steps[{step_index}].advance_timers_ms",
            )
        schedule.append(schedule_step)
        per_step_scalars.append(
            {
                "step_index": step_index,
                "distance_units": reference.avatar_velocity_units_per_s * step_ms / 1000.0,
                "angle_delta_abs_rad": reference.angular_velocity_radians_per_ms * step_ms,
            }
        )
    return {"schedule": schedule, "per_step_scalars": per_step_scalars}


def _moves_for_step(
    raw_step: Mapping[str, Any],
    player_count: int,
    player_lookup: Mapping[str, int],
    step_index: int,
) -> list[int]:
    raw_moves = raw_step.get("moves", raw_step.get("actions"))
    moves = [0 for _ in range(player_count)]
    seen: set[int] = set()
    if isinstance(raw_moves, list):
        for move_index, raw_move in enumerate(raw_moves):
            if not isinstance(raw_move, Mapping):
                raise SeedError(f"steps[{step_index}].moves[{move_index}] must be an object")
            player_index = _lookup_player_index(
                raw_move.get("player_id", raw_move.get("id")),
                move_index,
                player_lookup,
                field=f"steps[{step_index}].moves[{move_index}].player_id",
            )
            moves[player_index] = _source_move(raw_move.get("move", raw_move.get("action")), step_index)
            seen.add(player_index)
    elif isinstance(raw_moves, Mapping):
        for fallback_index, (player_ref, value) in enumerate(raw_moves.items()):
            player_index = _lookup_player_index(
                player_ref,
                fallback_index,
                player_lookup,
                field=f"steps[{step_index}].moves.{player_ref}",
            )
            moves[player_index] = _source_move(value, step_index)
            seen.add(player_index)
    elif raw_moves is None:
        raise SeedError(f"steps[{step_index}] has no moves/actions")
    else:
        raise SeedError(f"steps[{step_index}].moves must be a list or object")
    if len(seen) != player_count:
        missing = sorted(set(range(player_count)) - seen)
        raise SeedError(f"steps[{step_index}] is missing source moves for player indices {missing}")
    return moves


def _body_capacity(body_capacity: int | None, seeded_bodies: Sequence[Mapping[str, Any]]) -> int:
    seeded_count = len(seeded_bodies)
    if body_capacity is None:
        return seeded_count
    if body_capacity < seeded_count:
        raise SeedError(
            f"--body-capacity {body_capacity} is smaller than seeded body count {seeded_count}"
        )
    return body_capacity


def _body_arrays(
    seeded_bodies: Sequence[Mapping[str, Any]],
    body_capacity: int,
) -> dict[str, Any]:
    active = [False for _ in range(body_capacity)]
    pos = [[0.0, 0.0] for _ in range(body_capacity)]
    radius = [0.0 for _ in range(body_capacity)]
    owner = [-1 for _ in range(body_capacity)]
    body_num = [-1 for _ in range(body_capacity)]
    insert_tick = [-1 for _ in range(body_capacity)]
    insert_kind = [-1 for _ in range(body_capacity)]
    birth_ms = [0.0 for _ in range(body_capacity)]

    for slot, body in enumerate(seeded_bodies):
        active[slot] = True
        pos[slot] = [float(body["x"]), float(body["y"])]
        radius[slot] = float(body["radius"])
        owner[slot] = int(body["owner_index"])
        body_num[slot] = int(body["num"])
        insert_kind[slot] = BODY_KIND_SEEDED
        age_ms = body.get("age_ms")
        birth_ms[slot] = 0.0 if age_ms is None else -float(age_ms)

    return {
        "body_active": _array([active], "bool"),
        "body_pos": _array([pos], "float64", shape=[1, body_capacity, 2]),
        "body_radius": _array([radius], "float64"),
        "body_owner": _array([owner], "int16"),
        "body_num": _array([body_num], "int32"),
        "body_insert_tick": _array([insert_tick], "int32"),
        "body_insert_kind": _array([insert_kind], "int8"),
        "body_birth_ms": _array([birth_ms], "float64"),
        "body_write_cursor": _array([len(seeded_bodies)], "int32"),
        "world_body_count": _array([len(seeded_bodies)], "int32"),
        "body_overflow": _array([False], "bool"),
    }


def _array(values: Any, dtype: str, *, shape: list[int] | None = None) -> dict[str, Any]:
    return {"shape": _shape(values) if shape is None else shape, "dtype": dtype, "values": values}


def _shape(value: Any) -> list[int]:
    shape = []
    current = value
    while isinstance(current, list):
        shape.append(len(current))
        current = current[0] if current else None
    return shape


def _stackable_groups(fixtures: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[str]] = {}
    for fixture in fixtures:
        profile = fixture.get("profile", {})
        if not isinstance(profile, Mapping):
            continue
        key = (int(profile["P"]), int(profile["K"]))
        groups.setdefault(key, []).append(str(fixture["scenario_id"]))
    return [
        {"P": key[0], "K": key[1], "scenario_ids": scenario_ids}
        for key, scenario_ids in sorted(groups.items())
    ]


def _fixture_limitations(
    payload: Mapping[str, Any],
    seeded_bodies: Sequence[Mapping[str, Any]],
    trail_state: Mapping[str, Any],
    print_manager_state: Mapping[str, Any],
) -> list[str]:
    limitations = [
        "initial arrays only; no post-step state has been computed",
        "common-trace comparison is the next step and is not included here",
        "death-point insertion, scoring, terminal events, bonuses, and wrappers are not seeded",
    ]
    if not seeded_bodies:
        limitations.append("no initial world bodies are present in this fixture")
    if not any(trail_state["visible_trail_points"]):
        limitations.append("no full visible trail point list is present beyond count/last/cursor state")
    if not any(print_manager_state["present"]):
        limitations.append("no initial PrintManager object is present")
    comparison = _mapping(payload.get("comparison"), "comparison", default={})
    if comparison.get("include_events"):
        limitations.append("event expectations are declared by the fixture but not materialized as arrays")
    return limitations


def _verification_status_note(labels: Sequence[str]) -> str:
    if "python-runner-verified" in labels:
        return "promoted through the narrow Python source-runner path"
    if "js-oracle-pinned" in labels:
        return "JS oracle pinned only; Python/vector equivalence still needs a gate"
    return "not marked as JS-pinned or Python-runner-verified"


def _provenance_labels(payload: Mapping[str, Any]) -> list[str]:
    provenance = _mapping(payload.get("provenance"), "provenance", default={})
    labels = provenance.get("labels", [])
    if not isinstance(labels, list):
        return []
    return [str(label) for label in labels]


def _map_size(
    payload: Mapping[str, Any],
    source_setup: Mapping[str, Any],
    initial_state: Mapping[str, Any],
) -> float:
    value = initial_state.get("map_size", source_setup.get("map_size", payload.get("map_size")))
    if value is None:
        raise SeedError("map_size is required in source_setup or initial_state")
    return _finite_number(value, "map_size")


def _borderless(source_setup: Mapping[str, Any], initial_state: Mapping[str, Any]) -> bool:
    game = _mapping(source_setup.get("game"), "source_setup.game", default={})
    value = initial_state.get("borderless", game.get("borderless", False))
    return _bool(value, "borderless")


def _mapping(value: Any, field: str, *, default: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    if value is None:
        return {} if default is None else default
    if not isinstance(value, Mapping):
        raise SeedError(f"{field} must be an object")
    return value


def _player_state(player: Mapping[str, Any]) -> Mapping[str, Any]:
    state = player.get("state", player.get("initial", player))
    if not isinstance(state, Mapping):
        raise SeedError("player state must be an object")
    return state


def _trail_points(raw_points: Any, player_index: int) -> list[list[float]]:
    if raw_points is None:
        return []
    if not isinstance(raw_points, list):
        raise SeedError(f"players[{player_index}].initial.trail.points must be a list")
    return [
        _point(point, f"players[{player_index}].initial.trail.points[{point_index}]")
        for point_index, point in enumerate(raw_points)
    ]


def _point(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise SeedError(f"{field} must be [x, y]")
    return [_finite_number(value[0], f"{field}[0]"), _finite_number(value[1], f"{field}[1]")]


def _default_scalar_list(raw_values: Any, count: int, default: float) -> list[float]:
    values = [default for _ in range(count)]
    if raw_values is None:
        return values
    if not isinstance(raw_values, list) or len(raw_values) < count:
        raise SeedError("scalar list must include one value per player")
    return [_finite_number(raw_values[index], f"scalar[{index}]") for index in range(count)]


def _default_bool_list(raw_values: Any, count: int, default: bool) -> list[bool]:
    values = [default for _ in range(count)]
    if raw_values is None:
        return values
    if not isinstance(raw_values, list) or len(raw_values) < count:
        raise SeedError("bool list must include one value per player")
    return [_bool(raw_values[index], f"bool[{index}]") for index in range(count)]


def _default_int_list(raw_values: Any, count: int, default: int) -> list[int]:
    values = [default for _ in range(count)]
    if raw_values is None:
        return values
    if not isinstance(raw_values, list) or len(raw_values) < count:
        raise SeedError("int list must include one value per player")
    return [_int(raw_values[index], f"int[{index}]") for index in range(count)]


def _optional_int_from_keys(source: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key in source:
            return _int(source[key], key)
    return None


def _source_move(value: Any, step_index: int) -> int:
    move = _int(value, f"steps[{step_index}].move")
    if move not in {-1, 0, 1}:
        raise SeedError(f"steps[{step_index}].move must be -1, 0, or 1")
    return move


def _lookup_player_index(
    value: Any,
    fallback_index: int,
    lookup: Mapping[str, int],
    *,
    field: str = "player reference",
) -> int:
    if value is None:
        return fallback_index
    text = str(value)
    if text in lookup:
        return lookup[text]
    raise SeedError(f"{field} references unknown player {value!r}")


def _first_present(source: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return None


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SeedError(f"{field} must be a finite number")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise SeedError(f"{field} must be a finite number")
    return result


def _optional_nonnegative_number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    result = _finite_number(value, field)
    if result < 0.0:
        raise SeedError(f"{field} must be non-negative")
    return result


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SeedError(f"{field} must be an integer")
    return int(value)


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SeedError(f"{field} must be a non-empty string")
    return value


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise SeedError(f"{field} must be a boolean")
    return bool(value)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed vector-lane JSON arrays from environment fixtures."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Scenario JSON files or batch manifests with a scenarios list.",
    )
    parser.add_argument(
        "--body-capacity",
        type=_nonnegative_int,
        default=None,
        help="Pad seeded body buffers to this K. Defaults to the seeded body count.",
    )
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON.")
    args = parser.parse_args()

    summary = seed_inputs(args.paths, body_capacity=args.body_capacity)
    indent = None if args.compact else 2
    print(json.dumps(summary, indent=indent, sort_keys=True))


if __name__ == "__main__":
    main()
