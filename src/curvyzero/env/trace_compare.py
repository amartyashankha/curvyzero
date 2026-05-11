"""Project JS and Python scenario traces into the first common comparison view."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

COMMON_TRACE_SCHEMA = "curvyzero_common_trace/v1"


class TraceCompareError(ValueError):
    """Raised when a trace payload cannot be projected."""


def project_common_trace(payload: Mapping[str, Any]) -> dict[str, object]:
    """Project current JS/Python runner output into common trace fields."""

    if not isinstance(payload, Mapping):
        raise TraceCompareError("payload must be an object")

    trace = payload.get("trace")
    if isinstance(trace, list):
        return _project_js_trace(payload, trace)
    if isinstance(trace, Mapping):
        return _project_python_trace(payload, trace)
    raise TraceCompareError("payload.trace must be a JS list or Python object")


def normalize_trace_payload(payload: Mapping[str, Any]) -> dict[str, object]:
    """Compatibility name for callers that want a normalized trace payload."""

    return project_common_trace(payload)


def _project_js_trace(payload: Mapping[str, Any], trace: list[Any]) -> dict[str, object]:
    scenario_id = _string(payload.get("scenario") or payload.get("scenario_id"), "scenario")
    include_events = _include_events(payload)
    include_body_state = _include_body_canary_state(payload)
    include_trail_state = _include_trail_cadence_state(payload)
    include_print_manager_state = _include_print_manager_state(payload)
    steps = []
    map_size: float | int | None = None
    for step_index, raw_frame in enumerate(trace):
        frame = _mapping(raw_frame, f"trace[{step_index}]")
        if step_index == 0:
            map_size = _optional_js_map_size(frame, step_index)
        raw_players = _list(frame.get("avatars"), f"trace[{step_index}].avatars")
        step: dict[str, object] = {
            "step_index": _int(frame.get("tick", step_index), f"trace[{step_index}].tick"),
            "step_ms": _number(
                frame.get("stepMs", frame.get("step_ms")),
                f"trace[{step_index}].stepMs",
            ),
            "players": [
                _project_js_player(
                    raw_player,
                    player_index,
                    step_index,
                    include_body_state=include_body_state,
                    include_trail_state=include_trail_state,
                    include_print_manager_state=include_print_manager_state,
                )
                for player_index, raw_player in enumerate(raw_players)
            ],
        }
        if include_body_state or include_trail_state or include_print_manager_state:
            step["worldBodyCount"] = _js_world_body_count(frame, step_index)
        if include_events:
            step["events"] = _project_js_events(
                frame.get("events"),
                _js_avatar_player_ids(raw_players, step_index),
                step_index,
            )
        steps.append(step)
    result: dict[str, object] = {
        "schema": COMMON_TRACE_SCHEMA,
        "scenario_id": scenario_id,
        "steps": steps,
    }
    if map_size is not None:
        result["map_size"] = map_size
    return result


def _project_python_trace(payload: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, object]:
    frames = _list(trace.get("frames"), "trace.frames")
    frames = _drop_reset_frame(payload, frames)
    scenario = _mapping(payload.get("scenario", {}), "scenario")
    player_ids = _player_ids(scenario)
    include_events = _include_events(payload, scenario)
    include_body_state = _include_body_canary_state(payload, scenario)
    include_trail_state = _include_trail_cadence_state(payload, scenario)
    include_print_manager_state = _include_print_manager_state(payload, scenario)

    steps = []
    for step_index, raw_frame in enumerate(frames):
        frame = _mapping(raw_frame, f"trace.frames[{step_index}]")
        step: dict[str, object] = {
            "step_index": step_index,
            "step_ms": _python_step_ms(scenario, payload, frame, step_index),
            "players": _project_python_players(
                frame,
                player_ids,
                step_index,
                include_body_state=include_body_state,
                include_trail_state=include_trail_state,
                include_print_manager_state=include_print_manager_state,
            ),
        }
        if include_body_state or include_trail_state or include_print_manager_state:
            step["worldBodyCount"] = _python_world_body_count(frame, step_index)
        if include_events:
            step["events"] = _project_python_events(frame.get("events"), step_index)
        steps.append(step)
    scenario_id = _string(payload.get("scenario_id") or scenario.get("scenario_id"), "scenario_id")
    result: dict[str, object] = {
        "schema": COMMON_TRACE_SCHEMA,
        "scenario_id": scenario_id,
        "steps": steps,
    }
    map_size = _optional_python_map_size(scenario)
    if map_size is not None:
        result["map_size"] = map_size
    return result


def _project_js_player(
    raw_player: Any,
    player_index: int,
    step_index: int,
    *,
    include_body_state: bool = False,
    include_trail_state: bool = False,
    include_print_manager_state: bool = False,
) -> dict[str, object]:
    avatar = _mapping(raw_player, f"trace[{step_index}].avatars[{player_index}]")
    player: dict[str, object] = {
        "player_id": str(avatar.get("name") or avatar.get("player_id") or avatar.get("id")),
    }
    _copy_number(player, avatar, "x")
    _copy_number(player, avatar, "y")
    _copy_number(player, avatar, "angle")
    _copy_bool(player, avatar, "alive")
    _copy_number(player, avatar, "score")
    _copy_number(player, avatar, "roundScore")
    if include_body_state or include_trail_state:
        _copy_number(player, avatar, "trailPointCount")
        _copy_last_trail_point(
            player,
            avatar,
            f"trace[{step_index}].avatars[{player_index}].lastTrailPoint",
        )
        _copy_number(player, avatar, "bodyNum")
        _copy_number(player, avatar, "bodyCount")
    if include_trail_state:
        _copy_bool(player, avatar, "printing")
    if include_print_manager_state:
        _copy_bool(player, avatar, "printing")
        _copy_number(player, avatar, "trailPointCount")
        _copy_last_trail_point(
            player,
            avatar,
            f"trace[{step_index}].avatars[{player_index}].lastTrailPoint",
        )
        _copy_number(player, avatar, "bodyNum")
        _copy_number(player, avatar, "bodyCount")
        _copy_print_manager(
            player,
            avatar.get("printManager", avatar.get("print_manager")),
            f"trace[{step_index}].avatars[{player_index}].printManager",
        )
    return player


def _project_python_players(
    frame: Mapping[str, Any],
    player_ids: tuple[str, ...],
    step_index: int,
    *,
    include_body_state: bool = False,
    include_trail_state: bool = False,
    include_print_manager_state: bool = False,
) -> list[dict[str, object]]:
    positions = _list(frame.get("positions"), f"trace.frames[{step_index}].positions")
    headings = _list(frame.get("headings"), f"trace.frames[{step_index}].headings")
    alive = _list(frame.get("alive"), f"trace.frames[{step_index}].alive")
    players = []
    for player_index, position in enumerate(positions):
        x, y = _xy(position, step_index, player_index)
        player: dict[str, object] = {
            "player_id": player_ids[player_index]
            if player_index < len(player_ids)
            else f"player_{player_index}",
            "x": x,
            "y": y,
        }
        if player_index < len(headings):
            player["angle"] = _number(
                headings[player_index],
                f"trace.frames[{step_index}].headings[{player_index}]",
            )
        if player_index < len(alive):
            player["alive"] = _bool(
                alive[player_index],
                f"trace.frames[{step_index}].alive[{player_index}]",
            )
        _copy_indexed_number(player, frame.get("scores"), player["player_id"], player_index, "score")
        _copy_indexed_number(
            player,
            frame.get("roundScores", frame.get("roundScore")),
            player["player_id"],
            player_index,
            "roundScore",
        )
        if include_body_state or include_trail_state:
            _copy_indexed_number(
                player,
                frame.get("trailPointCounts", frame.get("trailPointCount")),
                player["player_id"],
                player_index,
                "trailPointCount",
            )
            _copy_indexed_last_trail_point(
                player,
                frame.get("lastTrailPoints", frame.get("lastTrailPoint")),
                player["player_id"],
                player_index,
            )
            _copy_indexed_number(
                player,
                frame.get("bodyNums", frame.get("bodyNum")),
                player["player_id"],
                player_index,
                "bodyNum",
            )
            _copy_indexed_number(
                player,
                frame.get("bodyCounts", frame.get("bodyCount")),
                player["player_id"],
                player_index,
                "bodyCount",
            )
        if include_trail_state:
            _copy_indexed_bool(
                player,
                frame.get("printing", frame.get("printings")),
                player["player_id"],
                player_index,
                "printing",
            )
        if include_print_manager_state:
            _copy_indexed_bool(
                player,
                frame.get("printing", frame.get("printings")),
                player["player_id"],
                player_index,
                "printing",
            )
            _copy_indexed_number(
                player,
                frame.get("trailPointCounts", frame.get("trailPointCount")),
                player["player_id"],
                player_index,
                "trailPointCount",
            )
            _copy_indexed_last_trail_point(
                player,
                frame.get("lastTrailPoints", frame.get("lastTrailPoint")),
                player["player_id"],
                player_index,
            )
            _copy_indexed_number(
                player,
                frame.get("bodyNums", frame.get("bodyNum")),
                player["player_id"],
                player_index,
                "bodyNum",
            )
            _copy_indexed_number(
                player,
                frame.get("bodyCounts", frame.get("bodyCount")),
                player["player_id"],
                player_index,
                "bodyCount",
            )
            _copy_indexed_print_manager(
                player,
                frame.get("printManagers", frame.get("printManager")),
                player["player_id"],
                player_index,
            )
        players.append(player)
    return players


def _project_js_events(
    raw_events: Any,
    player_ids_by_avatar: Mapping[str, str],
    step_index: int,
) -> list[dict[str, object]]:
    events = []
    for event_index, raw_event in enumerate(
        _event_list(raw_events, f"trace[{step_index}].events")
    ):
        event = _mapping(raw_event, f"trace[{step_index}].events[{event_index}]")
        name = _string(event.get("event"), f"trace[{step_index}].events[{event_index}].event")
        data = _mapping(event.get("data", {}), f"trace[{step_index}].events[{event_index}].data")
        events.append(_project_source_event(name, data, player_ids_by_avatar, step_index, event_index))
    return events


def _project_python_events(raw_events: Any, step_index: int) -> list[dict[str, object]]:
    events = []
    for event_index, raw_event in enumerate(
        _event_list(raw_events, f"trace.frames[{step_index}].events")
    ):
        event = _mapping(raw_event, f"trace.frames[{step_index}].events[{event_index}]")
        name = _string(
            event.get("event"),
            f"trace.frames[{step_index}].events[{event_index}].event",
        )
        events.append(_project_python_event(name, event, step_index, event_index))
    return events


def _project_source_event(
    name: str,
    data: Mapping[str, Any],
    player_ids_by_avatar: Mapping[str, str],
    step_index: int,
    event_index: int,
) -> dict[str, object]:
    if name == "position":
        return {
            "event": name,
            "player_id": _js_player_id(
                data.get("avatar"),
                player_ids_by_avatar,
                f"trace[{step_index}].events[{event_index}].data.avatar",
            ),
            "x": _number(data.get("x"), f"trace[{step_index}].events[{event_index}].data.x"),
            "y": _number(data.get("y"), f"trace[{step_index}].events[{event_index}].data.y"),
        }
    if name == "angle":
        return {
            "event": name,
            "player_id": _js_player_id(
                data.get("avatar"),
                player_ids_by_avatar,
                f"trace[{step_index}].events[{event_index}].data.avatar",
            ),
            "angle": _number(
                data.get("angle"),
                f"trace[{step_index}].events[{event_index}].data.angle",
            ),
        }
    if name == "point":
        return {
            "event": name,
            "player_id": _js_player_id(
                data.get("avatar"),
                player_ids_by_avatar,
                f"trace[{step_index}].events[{event_index}].data.avatar",
            ),
            "x": _number(data.get("x"), f"trace[{step_index}].events[{event_index}].data.x"),
            "y": _number(data.get("y"), f"trace[{step_index}].events[{event_index}].data.y"),
            "important": _bool(
                data.get("important"),
                f"trace[{step_index}].events[{event_index}].data.important",
            ),
        }
    if name == "die":
        return {
            "event": name,
            "player_id": _js_player_id(
                data.get("avatar"),
                player_ids_by_avatar,
                f"trace[{step_index}].events[{event_index}].data.avatar",
            ),
            "killer_id": _js_optional_player_id(
                data.get("killer"),
                player_ids_by_avatar,
                f"trace[{step_index}].events[{event_index}].data.killer",
            ),
            "old": _optional_bool(
                data.get("old"),
                f"trace[{step_index}].events[{event_index}].data.old",
            ),
        }
    if name in {"score", "score:round"}:
        return {
            "event": name,
            "player_id": _js_player_id(
                data.get("avatar"),
                player_ids_by_avatar,
                f"trace[{step_index}].events[{event_index}].data.avatar",
            ),
            "score": _number(
                data.get("score"),
                f"trace[{step_index}].events[{event_index}].data.score",
            ),
            "roundScore": _number(
                data.get("roundScore"),
                f"trace[{step_index}].events[{event_index}].data.roundScore",
            ),
        }
    if name == "round:end":
        return {
            "event": name,
            "winner_id": _js_optional_player_id(
                data.get("winner"),
                player_ids_by_avatar,
                f"trace[{step_index}].events[{event_index}].data.winner",
            ),
        }
    if name == "property":
        return {
            "event": name,
            "player_id": _js_player_id(
                data.get("avatar"),
                player_ids_by_avatar,
                f"trace[{step_index}].events[{event_index}].data.avatar",
            ),
            "property": _string(
                data.get("property"),
                f"trace[{step_index}].events[{event_index}].data.property",
            ),
            "value": _property_value(
                data.get("value"),
                f"trace[{step_index}].events[{event_index}].data.value",
            ),
        }
    return {"event": name}


def _project_python_event(
    name: str,
    event: Mapping[str, Any],
    step_index: int,
    event_index: int,
) -> dict[str, object]:
    if name == "position":
        return {
            "event": name,
            "player_id": _string(
                event.get("player_id"),
                f"trace.frames[{step_index}].events[{event_index}].player_id",
            ),
            "x": _number(event.get("x"), f"trace.frames[{step_index}].events[{event_index}].x"),
            "y": _number(event.get("y"), f"trace.frames[{step_index}].events[{event_index}].y"),
        }
    if name == "angle":
        return {
            "event": name,
            "player_id": _string(
                event.get("player_id"),
                f"trace.frames[{step_index}].events[{event_index}].player_id",
            ),
            "angle": _number(
                event.get("angle"),
                f"trace.frames[{step_index}].events[{event_index}].angle",
            ),
        }
    if name == "point":
        return {
            "event": name,
            "player_id": _string(
                event.get("player_id"),
                f"trace.frames[{step_index}].events[{event_index}].player_id",
            ),
            "x": _number(event.get("x"), f"trace.frames[{step_index}].events[{event_index}].x"),
            "y": _number(event.get("y"), f"trace.frames[{step_index}].events[{event_index}].y"),
            "important": _bool(
                event.get("important"),
                f"trace.frames[{step_index}].events[{event_index}].important",
            ),
        }
    if name == "die":
        return {
            "event": name,
            "player_id": _string(
                event.get("player_id"),
                f"trace.frames[{step_index}].events[{event_index}].player_id",
            ),
            "killer_id": _optional_string(
                event.get("killer_id"),
                f"trace.frames[{step_index}].events[{event_index}].killer_id",
            ),
            "old": _optional_bool(
                event.get("old"),
                f"trace.frames[{step_index}].events[{event_index}].old",
            ),
        }
    if name in {"score", "score:round"}:
        return {
            "event": name,
            "player_id": _string(
                event.get("player_id"),
                f"trace.frames[{step_index}].events[{event_index}].player_id",
            ),
            "score": _number(
                event.get("score"),
                f"trace.frames[{step_index}].events[{event_index}].score",
            ),
            "roundScore": _number(
                event.get("roundScore"),
                f"trace.frames[{step_index}].events[{event_index}].roundScore",
            ),
        }
    if name == "round:end":
        return {
            "event": name,
            "winner_id": _optional_string(
                event.get("winner_id"),
                f"trace.frames[{step_index}].events[{event_index}].winner_id",
            ),
        }
    if name == "property":
        return {
            "event": name,
            "player_id": _string(
                event.get("player_id"),
                f"trace.frames[{step_index}].events[{event_index}].player_id",
            ),
            "property": _string(
                event.get("property"),
                f"trace.frames[{step_index}].events[{event_index}].property",
            ),
            "value": _property_value(
                event.get("value"),
                f"trace.frames[{step_index}].events[{event_index}].value",
            ),
        }
    return {"event": name}


def _drop_reset_frame(payload: Mapping[str, Any], frames: list[Any]) -> list[Any]:
    scenario = _mapping(payload.get("scenario", {}), "scenario")
    action_script = scenario.get("action_script", scenario.get("toy_action_script"))
    if isinstance(action_script, list) and len(frames) == len(action_script) + 1:
        return frames[1:]
    return frames


def _include_events(payload: Mapping[str, Any], *scenario_payloads: Mapping[str, Any]) -> bool:
    for source in (payload, *scenario_payloads):
        comparison = source.get("comparison")
        if isinstance(comparison, Mapping) and comparison.get("include_events") is True:
            return True
    return False


def _include_body_canary_state(
    payload: Mapping[str, Any],
    *scenario_payloads: Mapping[str, Any],
) -> bool:
    for source in (payload, *scenario_payloads):
        scenario_id = source.get("scenario_id", source.get("scenario"))
        if isinstance(scenario_id, str) and scenario_id.startswith("source_body_"):
            return True
        if isinstance(scenario_id, str) and scenario_id.startswith("source_collision_"):
            return True
        if scenario_id == "source_borderless_wrap_skips_destination_body_then_next_frame_kills":
            return True
        comparison = source.get("comparison")
        if isinstance(comparison, Mapping):
            python_target = comparison.get("python_target")
            if python_target in {
                "pending-source-body",
                "source-body-canary",
                "pending-source-collision-order",
                "source-collision-order-canary",
            }:
                return True
    return False


def _include_trail_cadence_state(
    payload: Mapping[str, Any],
    *scenario_payloads: Mapping[str, Any],
) -> bool:
    for source in (payload, *scenario_payloads):
        scenario_id = source.get("scenario_id", source.get("scenario"))
        if isinstance(scenario_id, str) and scenario_id.startswith("source_trail_"):
            return True
        comparison = source.get("comparison")
        if isinstance(comparison, Mapping):
            python_target = comparison.get("python_target")
            if python_target in {
                "pending-source-trail-cadence",
                "source-trail-cadence-canary",
            }:
                return True
    return False


def _include_print_manager_state(
    payload: Mapping[str, Any],
    *scenario_payloads: Mapping[str, Any],
) -> bool:
    for source in (payload, *scenario_payloads):
        scenario_id = source.get("scenario_id", source.get("scenario"))
        if scenario_id == "source_borderless_print_manager_wrap_toggle_step":
            return True
        if isinstance(scenario_id, str) and scenario_id.startswith("source_print_manager_"):
            return True
        comparison = source.get("comparison")
        if isinstance(comparison, Mapping):
            python_target = comparison.get("python_target")
            if python_target in {
                "pending-source-print-manager",
                "source-print-manager-canary",
                "pending-source-trail-gap",
                "source-trail-gap-canary",
            }:
                return True
    return False


def _js_avatar_player_ids(raw_players: list[Any], step_index: int) -> dict[str, str]:
    player_ids = {}
    for player_index, raw_player in enumerate(raw_players):
        avatar = _mapping(raw_player, f"trace[{step_index}].avatars[{player_index}]")
        avatar_id = avatar.get("id")
        if avatar_id is None:
            continue
        player_id = str(
            avatar.get("name")
            or avatar.get("player_id")
            or avatar.get("id")
        )
        player_ids[str(avatar_id)] = player_id
    return player_ids


def _js_player_id(value: Any, player_ids_by_avatar: Mapping[str, str], field: str) -> str:
    if value is None:
        raise TraceCompareError(f"{field} must be an avatar id")
    text = str(value)
    if text in player_ids_by_avatar:
        return player_ids_by_avatar[text]
    if isinstance(value, str) and value:
        return value
    raise TraceCompareError(f"{field} references unknown avatar {value!r}")


def _js_optional_player_id(
    value: Any,
    player_ids_by_avatar: Mapping[str, str],
    field: str,
) -> str | None:
    if value is None:
        return None
    return _js_player_id(value, player_ids_by_avatar, field)


def _python_step_ms(
    scenario: Mapping[str, Any],
    payload: Mapping[str, Any],
    frame: Mapping[str, Any] | None = None,
    step_index: int = 0,
) -> float | int:
    if frame is not None:
        frame_value = frame.get("stepMs", frame.get("step_ms"))
        if frame_value is not None:
            return _number(frame_value, f"trace.frames[{step_index}].stepMs")
    time_policy = _mapping(scenario.get("time_policy", {}), "scenario.time_policy")
    sequence = time_policy.get("step_ms_sequence")
    if isinstance(sequence, list) and step_index < len(sequence):
        return _number(
            sequence[step_index],
            f"scenario.time_policy.step_ms_sequence[{step_index}]",
        )
    value = time_policy.get("step_ms", scenario.get("step_ms", payload.get("step_ms")))
    return _number(value, "scenario.time_policy.step_ms")


def _optional_js_map_size(frame: Mapping[str, Any], step_index: int) -> float | int | None:
    game = frame.get("game")
    if not isinstance(game, Mapping):
        return None
    value = game.get("size")
    if value is None:
        return None
    return _number(value, f"trace[{step_index}].game.size")


def _optional_python_map_size(scenario: Mapping[str, Any]) -> float | int | None:
    initial_state = _mapping(scenario.get("initial_state", {}), "scenario.initial_state")
    value = initial_state.get("map_size", initial_state.get("size"))
    if value is None:
        return None
    return _number(value, "scenario.initial_state.map_size")


def _js_world_body_count(frame: Mapping[str, Any], step_index: int) -> float | int:
    game = _mapping(frame.get("game"), f"trace[{step_index}].game")
    return _number(game.get("worldBodyCount"), f"trace[{step_index}].game.worldBodyCount")


def _python_world_body_count(frame: Mapping[str, Any], step_index: int) -> float | int:
    value = frame.get("worldBodyCount", frame.get("world_body_count"))
    return _number(value, f"trace.frames[{step_index}].worldBodyCount")


def _player_ids(scenario: Mapping[str, Any]) -> tuple[str, ...]:
    initial_state = _mapping(scenario.get("initial_state", {}), "scenario.initial_state")
    players = initial_state.get("players")
    if not isinstance(players, list):
        return ()
    ids = []
    for index, raw_player in enumerate(players):
        player = raw_player if isinstance(raw_player, Mapping) else {}
        ids.append(str(player.get("id") or player.get("player_id") or f"player_{index}"))
    return tuple(ids)


def _copy_number(target: dict[str, object], source: Mapping[str, Any], key: str) -> None:
    if key in source and source[key] is not None:
        target[key] = _number(source[key], key)


def _copy_bool(target: dict[str, object], source: Mapping[str, Any], key: str) -> None:
    if key in source and source[key] is not None:
        target[key] = _bool(source[key], key)


def _copy_indexed_bool(
    target: dict[str, object],
    source: Any,
    player_id: object,
    player_index: int,
    key: str,
) -> None:
    value = None
    if isinstance(source, list) and player_index < len(source):
        value = source[player_index]
    elif isinstance(source, Mapping):
        value = source.get(str(player_id), source.get(f"player_{player_index}"))
    if value is not None:
        target[key] = _bool(value, key)


def _copy_indexed_number(
    target: dict[str, object],
    source: Any,
    player_id: object,
    player_index: int,
    key: str,
) -> None:
    value = None
    if isinstance(source, list) and player_index < len(source):
        value = source[player_index]
    elif isinstance(source, Mapping):
        value = source.get(str(player_id), source.get(f"player_{player_index}"))
    if value is not None:
        target[key] = _number(value, key)


def _copy_last_trail_point(
    target: dict[str, object],
    source: Mapping[str, Any],
    field: str,
) -> None:
    if "lastTrailPoint" in source:
        target["lastTrailPoint"] = _last_trail_point(source["lastTrailPoint"], field)


def _copy_print_manager(target: dict[str, object], source: Any, field: str) -> None:
    manager = _mapping(source, field)
    target["printManager"] = {
        "active": _bool(manager.get("active"), f"{field}.active"),
        "distance": _number(manager.get("distance"), f"{field}.distance"),
        "lastX": _number(manager.get("lastX", manager.get("last_x")), f"{field}.lastX"),
        "lastY": _number(manager.get("lastY", manager.get("last_y")), f"{field}.lastY"),
    }


def _copy_indexed_last_trail_point(
    target: dict[str, object],
    source: Any,
    player_id: object,
    player_index: int,
) -> None:
    value = None
    found = False
    if isinstance(source, list) and player_index < len(source):
        value = source[player_index]
        found = True
    elif isinstance(source, Mapping):
        keys = (str(player_id), f"player_{player_index}")
        for key in keys:
            if key in source:
                value = source[key]
                found = True
                break
    if found:
        target["lastTrailPoint"] = _last_trail_point(value, "lastTrailPoint")


def _copy_indexed_print_manager(
    target: dict[str, object],
    source: Any,
    player_id: object,
    player_index: int,
) -> None:
    value = None
    if isinstance(source, list) and player_index < len(source):
        value = source[player_index]
    elif isinstance(source, Mapping):
        value = source.get(str(player_id), source.get(f"player_{player_index}"))
    if value is not None:
        _copy_print_manager(target, value, "printManager")


def _last_trail_point(value: Any, field: str) -> list[float | int] | None:
    if value is None:
        return None
    pair = _list(value, field)
    if len(pair) != 2:
        raise TraceCompareError(f"{field} must be [x, y] or null")
    return [
        _number(pair[0], f"{field}[0]"),
        _number(pair[1], f"{field}[1]"),
    ]


def _xy(value: Any, step_index: int, player_index: int) -> tuple[float | int, float | int]:
    pair = _list(value, f"trace.frames[{step_index}].positions[{player_index}]")
    if len(pair) != 2:
        raise TraceCompareError("position must be [x, y]")
    return (
        _number(pair[0], f"trace.frames[{step_index}].positions[{player_index}][0]"),
        _number(pair[1], f"trace.frames[{step_index}].positions[{player_index}][1]"),
    )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TraceCompareError(f"{field} must be an object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TraceCompareError(f"{field} must be a list")
    return value


def _event_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TraceCompareError(
            f"{field} must be a list when comparison.include_events is true"
        )
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TraceCompareError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TraceCompareError(f"{field} must be an integer")
    return value


def _number(value: Any, field: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TraceCompareError(f"{field} must be a number")
    return value


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise TraceCompareError(f"{field} must be a boolean")
    return value


def _property_value(value: Any, field: str) -> object:
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    raise TraceCompareError(f"{field} must be a scalar value")


def _optional_bool(value: Any, field: str) -> bool | None:
    if value is None:
        return None
    return _bool(value, field)
