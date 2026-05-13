import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest

from curvyzero.env import source_env
from curvyzero.env.source_env import (
    CurvyTronSourceEnv,
    SourceBodyState,
    SourceEnvError,
    SourceWorldState,
)
from curvyzero.fidelity.js_reuse_probe import CurvytronJsEnvWorker


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCENARIO_DIR = _REPO_ROOT / "scenarios" / "environment"
_ORACLE = _REPO_ROOT / "tools" / "reference_oracle" / "lifecycle_oracle.js"
_SCENARIO_ORACLE = _REPO_ROOT / "tools" / "reference_oracle" / "scenario_runner.js"
_WARMUP_SCENARIO = _SCENARIO_DIR / "source_lifecycle_spawn_rng_warmup_print_start_2p.json"
_MATCH_END_SCENARIO = _SCENARIO_DIR / "source_lifecycle_match_end_at_max_score_2p.json"
_SPAWN_3P_SCENARIO = _SCENARIO_DIR / "source_lifecycle_spawn_rng_order_3p.json"
_SPAWN_4P_SCENARIO = _SCENARIO_DIR / "source_lifecycle_spawn_rng_order_4p.json"
_LONG_NATURAL_ROLLOUT_SCENARIO = (
    _SCENARIO_DIR / "source_lifecycle_long_1v1_no_bonus_wall_round_done.json"
)
_LIVE_MOVEMENT_EVENT_TRACE_SCENARIO = (
    _SCENARIO_DIR / "source_live_movement_event_trace_2p_no_bonus_multistep.json"
)
_MID_ROUND_REMOVE_AVATAR_SCENARIO = (
    _SCENARIO_DIR / "source_lifecycle_mid_round_remove_avatar_2p.json"
)
_MID_ROUND_REMOVE_AVATAR_3P_SCENARIO = (
    _SCENARIO_DIR / "source_lifecycle_mid_round_remove_avatar_3p_continue_round_end.json"
)
_MID_ROUND_REMOVE_AVATAR_4P_SCENARIO = (
    _SCENARIO_DIR / "source_lifecycle_mid_round_remove_avatar_4p_continue_round_end.json"
)


def _load_scenario(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_js_lifecycle_oracle(path: Path) -> dict[str, object]:
    if shutil.which("node") is None:
        pytest.skip("node executable is not available")

    result = subprocess.run(
        ["node", str(_ORACLE), str(path)],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _run_js_world_island_probe() -> dict[str, object]:
    if shutil.which("node") is None:
        pytest.skip("node executable is not available")

    probe = r"""
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const repoRoot = process.cwd();
const referenceRoot = path.join(repoRoot, 'third_party', 'curvytron-reference');
const context = vm.createContext({ Date, JSON, Math });

[
  'src/shared/Collection.js',
  'src/server/core/Body.js',
  'src/server/core/Island.js',
  'src/server/core/World.js',
  'src/server/core/AvatarBody.js',
].forEach(function (file) {
  vm.runInContext(
    fs.readFileSync(path.join(referenceRoot, file), 'utf8'),
    context,
    { filename: file }
  );
});

const result = vm.runInContext(`
(function () {
  function makeAvatar(id) {
    return {
      id: id,
      radius: 0.6,
      bodyCount: 0,
      trailLatency: 3,
      equal: function (other) { return Boolean(other) && other.id === id; }
    };
  }

  function makeBody(avatar, x, y, radius, num) {
    var body = new AvatarBody(x, y, avatar);
    body.radius = radius;
    body.num = num;
    avatar.bodyCount = Math.max(avatar.bodyCount, num + 1);
    return body;
  }

  function sortedIslandIds(body) {
    return body.islands.items.map(function (island) { return island.id; }).sort();
  }

  function islandBodyIds(world, islandId) {
    return world.islands.getById(islandId).bodies.items.map(function (body) {
      return body.id;
    });
  }

  function lookupId(world, body) {
    var hit = world.getBody(body);
    return hit ? hit.id : null;
  }

  var owner = makeAvatar(2);
  var queryOwner = makeAvatar(1);
  var world = new World(88);
  world.activate();

  var boundary = makeBody(owner, 44, 40, 0.6, 0);
  world.addBody(boundary);
  var edge = makeBody(owner, 0.2, 87.8, 0.6, 1);
  world.addBody(edge);
  var cornerOnly = makeBody(owner, 43, 20, 0.6, 2);
  world.addBody(cornerOnly);

  return {
    islandSize: world.islandSize,
    bodyCount: world.bodyCount,
    boundaryIslandIds: sortedIslandIds(boundary),
    edgeIslandIds: sortedIslandIds(edge),
    cornerOnlyIslandIds: sortedIslandIds(cornerOnly),
    islandBodies: {
      '0:0': islandBodyIds(world, '0:0'),
      '1:0': islandBodyIds(world, '1:0'),
      '0:1': islandBodyIds(world, '0:1'),
      '1:1': islandBodyIds(world, '1:1')
    },
    lookupHitIds: {
      boundary: lookupId(world, makeBody(queryOwner, 44.2, 40, 0.6, 0)),
      edge: lookupId(world, makeBody(queryOwner, 0.2, 87.8, 0.6, 1)),
      cornerOnlyFromAdjacentCenter: lookupId(
        world,
        makeBody(queryOwner, 44, 20, 0.6, 2)
      ),
      outside: lookupId(world, makeBody(queryOwner, 100, 100, 0.6, 3))
    }
  };
}())
`, context, { filename: 'world_island_probe.vm.js' });

process.stdout.write(JSON.stringify(result));
"""
    result = subprocess.run(
        ["node", "-e", probe],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _run_js_scenario_oracle(path: Path) -> dict[str, object] | None:
    if shutil.which("node") is None:
        return None

    result = subprocess.run(
        ["node", str(_SCENARIO_ORACLE), str(path)],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _env_from_scenario(path: Path) -> tuple[CurvyTronSourceEnv, dict[str, object]]:
    return _env_from_scenario_with_options(path)


def _env_from_scenario_with_options(
    path: Path, *, include_deaths_snapshot: bool = False
) -> tuple[CurvyTronSourceEnv, dict[str, object]]:
    scenario = _load_scenario(path)
    source_setup = scenario["source_setup"]
    assert isinstance(source_setup, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)
    room = source_setup["room"]
    assert isinstance(room, dict)
    env = CurvyTronSourceEnv(
        random_values=random_setup["math_random_sequence"],
        max_score=float(room["max_score"]),
        include_deaths_snapshot=include_deaths_snapshot,
    )
    return env, scenario


def _positions_from_source_snapshot(snapshot: dict[str, object]) -> list[list[float]]:
    avatars = snapshot["avatars"]
    assert isinstance(avatars, list)
    return [[avatar["x"], avatar["y"]] for avatar in avatars]


def _alive_from_source_snapshot(snapshot: dict[str, object]) -> list[bool]:
    avatars = snapshot["avatars"]
    assert isinstance(avatars, list)
    return [avatar["alive"] for avatar in avatars]


def _scores_from_source_snapshot(snapshot: dict[str, object]) -> list[int]:
    avatars = snapshot["avatars"]
    assert isinstance(avatars, list)
    return [avatar["score"] for avatar in avatars]


def _round_winner_from_source_events(events: list[dict[str, object]]) -> int | None:
    round_end_events = [event for event in events if event["event"] == "round:end"]
    assert len(round_end_events) == 1
    data = round_end_events[0]["data"]
    assert isinstance(data, dict)
    winner = data["winner"]
    assert winner is None or isinstance(winner, int)
    return winner


def _active_source_env(*, player_count: int = 2, borderless: bool = False) -> CurvyTronSourceEnv:
    env = CurvyTronSourceEnv()
    env.reset(player_count=player_count, warmup_ms=0, borderless=borderless)
    env.advance_timers(0)
    env.set_avatar_state(1, x=20, y=20, angle=0)
    env.set_avatar_state(2, x=60, y=60, angle=0)
    return env


def _seed_source_body(
    env: CurvyTronSourceEnv,
    *,
    owner_id: int,
    x: float,
    y: float,
    radius: float | None = None,
    num: int = 0,
    birth_ms: float | None = None,
) -> SourceBodyState:
    game = env.game
    assert game is not None
    assert game.world is not None
    owner = env.avatar_by_id(owner_id)
    body = SourceBodyState(
        x=x,
        y=y,
        radius=owner.radius if radius is None else float(radius),
        avatar_id=owner.id,
        num=num,
        birth_ms=env.now_ms if birth_ms is None else birth_ms,
        trail_latency=owner.trail_latency,
    )
    game.world.add_body(body)
    game.world_body_count = game.world.body_count
    game.world_active = game.world.active
    owner.body_count = max(owner.body_count, num + 1)
    return body


def test_source_world_body_snapshot_marks_visual_segment_starts():
    env = _active_source_env()
    avatar = env.avatar_by_id(1)
    avatar.printing = True
    avatar.trail_last_x = None
    avatar.trail_last_y = None

    env.step({}, elapsed_ms=0)
    env.set_avatar_state(1, x=21, y=20, angle=0)
    env.step({}, elapsed_ms=0)
    avatar.trail_last_x = None
    avatar.trail_last_y = None
    env.set_avatar_state(1, x=22, y=20, angle=0)
    env.step({}, elapsed_ms=0)

    bodies = env.world_bodies_snapshot()
    assert [body["breakBefore"] for body in bodies] == [True, False, True]


def test_source_visual_trail_snapshot_keeps_dense_position_points_separate_from_bodies():
    env = _active_source_env()
    avatar = env.avatar_by_id(1)
    avatar.printing = True
    avatar.trail_last_x = 20.0
    avatar.trail_last_y = 20.0
    avatar.visual_trail_last_x = 20.0
    avatar.visual_trail_last_y = 20.0

    env.set_avatar_state(1, x=20.25, y=20, angle=0)
    env.step({}, elapsed_ms=0)
    env.set_avatar_state(1, x=20.5, y=20, angle=0)
    env.step({}, elapsed_ms=0)

    assert [body for body in env.world_bodies_snapshot() if body["avatarId"] == 1] == []
    visual_points = env.visual_trail_snapshot()
    assert [(point["x"], point["y"]) for point in visual_points if point["avatarId"] == 1] == [
        (20.25, 20),
        (20.5, 20),
    ]
    assert [point["breakBefore"] for point in visual_points if point["avatarId"] == 1] == [
        False,
        False,
    ]


def _die_events(env: CurvyTronSourceEnv) -> list[dict[str, object]]:
    return [event for event in env.events if event["event"] == "die"]


def _event_names_data(env: CurvyTronSourceEnv) -> list[tuple[str, dict[str, object]]]:
    return [(event["event"], event["data"]) for event in env.events]


def _source_event_data(env: CurvyTronSourceEnv) -> list[dict[str, object]]:
    return [{"event": event["event"], "data": event["data"]} for event in env.events]


def _source_gameplay_event_data(env: CurvyTronSourceEnv) -> list[dict[str, object]]:
    return [
        {"event": event["event"], "data": event["data"]}
        for event in env.events
        if event["event"] != "random"
    ]


def _source_random_call_labels(env: CurvyTronSourceEnv) -> list[dict[str, object]]:
    calls = []
    for call in env.random_calls:
        label = call["label"]
        assert isinstance(label, dict)
        calls.append(
            {
                "index": call["index"],
                "value": call["value"],
                "label": label["site"],
            }
        )
    return calls


def _run_source_bonus_fixture(
    filename: str,
) -> tuple[CurvyTronSourceEnv, dict[str, object], dict[str, object]]:
    env, scenario, frames = _run_source_bonus_fixture_frames(filename)
    return env, scenario, frames[0]


def _run_source_bonus_fixture_frames(
    filename: str,
) -> tuple[CurvyTronSourceEnv, dict[str, object], list[dict[str, object]]]:
    scenario = _load_scenario(_SCENARIO_DIR / filename)
    source_setup = scenario["source_setup"]
    assert isinstance(source_setup, dict)
    room = source_setup["room"]
    assert isinstance(room, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)
    env = CurvyTronSourceEnv(
        random_constant=float(random_setup["math_random"]),
        max_score=float(room["max_score"]),
        include_bonus_snapshot=True,
        emit_step_position_events=True,
    )
    env.reset(
        player_count=int(scenario["player_count"]),
        players=scenario["players"],
        warmup_ms=0,
    )
    env.advance_timers(0)
    assert env.game is not None
    env.game.print_start_due_ms = None
    env.game.bonus_pop_due_ms = None
    for player in scenario["players"]:
        initial = player["initial"]
        avatar = env.set_avatar_state(
            player["avatar_id"],
            x=initial["x"],
            y=initial["y"],
            angle=initial["angle_rad"],
        )
        avatar.printing = bool(initial["printing"])
    initial_state = scenario["initial_state"]
    assert isinstance(initial_state, dict)
    avatar_ids_by_player = {}
    for player in scenario["players"]:
        avatar_id = int(player["avatar_id"])
        for key in ("id", "player_id", "playerId", "avatar_id", "avatarId", "name"):
            if key in player:
                avatar_ids_by_player[str(player[key])] = avatar_id
    for body in initial_state.get("world_bodies", []):
        owner_key = (
            body.get("player_id")
            or body.get("playerId")
            or body.get("avatar_id")
            or body.get("avatarId")
            or body.get("avatar")
        )
        assert owner_key is not None
        _seed_source_body(
            env,
            owner_id=avatar_ids_by_player[str(owner_key)],
            x=float(body["x"]),
            y=float(body["y"]),
            radius=float(body["radius"]),
            num=int(body["num"]),
        )
    for bonus in initial_state.get("active_bonuses", []):
        env.seed_active_bonus(str(bonus["type"]), x=float(bonus["x"]), y=float(bonus["y"]))
    env.events.clear()
    env.random.calls.clear()

    frames = []
    for step in scenario["steps"]:
        env.events.clear()
        env.advance_timers(float(step.get("advance_timers_ms", 0.0)))
        moves = [move["move"] for move in step["moves"]]
        frame = env.step(moves, elapsed_ms=float(step["step_ms"]))
        frame["game"]["borderless"] = env.game.borderless
        frame["events"] = _source_event_data(env)
        frames.append(frame)
    return env, scenario, frames


def _run_source_live_movement_event_trace(
    path: Path,
) -> tuple[CurvyTronSourceEnv, dict[str, object], list[dict[str, object]]]:
    scenario = _load_scenario(path)
    source_setup = scenario["source_setup"]
    assert isinstance(source_setup, dict)
    room = source_setup["room"]
    assert isinstance(room, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)

    env = CurvyTronSourceEnv(
        random_constant=float(random_setup["math_random"]),
        max_score=float(room["max_score"]),
        emit_step_position_events=True,
        emit_step_angle_events=True,
    )
    env.reset(
        player_count=int(scenario["player_count"]),
        players=scenario["players"],
        warmup_ms=0,
    )
    env.advance_timers(0)
    assert env.game is not None
    env.game.print_start_due_ms = None
    for player in scenario["players"]:
        initial = player["initial"]
        avatar = env.set_avatar_state(
            player["avatar_id"],
            x=initial["x"],
            y=initial["y"],
            angle=initial["angle_rad"],
        )
        avatar.printing = bool(initial["printing"])
    env.events.clear()
    env.random.calls.clear()

    frames = []
    for step in scenario["steps"]:
        env.events.clear()
        env.advance_timers(float(step.get("advance_timers_ms", 0.0)))
        frame = env.step(
            [move["move"] for move in step["moves"]],
            elapsed_ms=float(step["step_ms"]),
        )
        frame["events"] = _source_event_data(env)
        frames.append(frame)
    return env, scenario, frames


def _run_source_long_rollout(
    path: Path,
) -> tuple[CurvyTronSourceEnv, dict[str, object], list[dict[str, object]]]:
    env, scenario = _env_from_scenario_with_options(path, include_deaths_snapshot=True)
    lifecycle = scenario["lifecycle"]
    assert isinstance(lifecycle, dict)
    rollout = scenario["rollout"]
    assert isinstance(rollout, dict)

    env.reset(
        player_count=int(scenario["player_count"]),
        players=scenario["players"],
        warmup_ms=float(lifecycle["new_round_time_ms"]),
    )

    frames = []
    for _tick in range(int(rollout["step_count"])):
        env.advance_timers(float(rollout["advance_timers_ms"]))
        frames.append(env.step(rollout["moves"], elapsed_ms=float(rollout["step_ms"])))
    return env, scenario, frames


def _run_source_lifecycle_actions(
    path: Path,
) -> tuple[CurvyTronSourceEnv, dict[str, object], list[dict[str, object]]]:
    env, scenario = _env_from_scenario_with_options(
        path,
        include_deaths_snapshot=True,
    )
    lifecycle = scenario["lifecycle"]
    assert isinstance(lifecycle, dict)
    actions = lifecycle["actions"]
    assert isinstance(actions, list)

    snapshots = [
        env.reset(
            player_count=int(scenario["player_count"]),
            players=scenario["players"],
            warmup_ms=float(lifecycle["new_round_time_ms"]),
        )
    ]
    for index, action in enumerate(actions):
        assert isinstance(action, dict)
        action_type = str(action["type"])
        if action_type == "advance_timers":
            env.advance_timers(float(action["ms"]))
        elif action_type == "remove_avatar":
            env.remove_avatar(action["avatar"])
        elif action_type == "set_avatar_state":
            angle = action.get("angle_rad", action.get("angle"))
            env.set_avatar_state(
                action["avatar"],
                x=action.get("x"),
                y=action.get("y"),
                angle=None if angle is None else float(angle),
                velocity=action.get("velocity"),
            )
        elif action_type == "update":
            env.step({}, elapsed_ms=float(action["step_ms"]))
        else:
            raise AssertionError(f"unsupported test lifecycle action {action_type!r}")
        snapshots.append(
            env.snapshot(
                f"after_action_{index}_{action_type}",
                action=action,
            )
        )
    return env, scenario, snapshots


def test_source_env_catches_seeded_active_bonus_self_small_like_js_fixture():
    env, _scenario, frame = _run_source_bonus_fixture("source_bonus_self_small_catch_step.json")
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}
    bonus_data = {
        "id": 1,
        "type": "BonusSelfSmall",
        "duration": 7500,
        "effects": [["radius", -1]],
    }
    expected_events = [
        {"event": "position", "data": {"avatar": 2, "x": 71.6, "y": 70}},
        {"event": "position", "data": {"avatar": 1, "x": 21.6, "y": 20}},
        {"event": "bonus:clear", "data": {"bonus": 1}},
        {
            "event": "property",
            "data": {"avatar": 1, "property": "radius", "value": 0.3},
        },
        {
            "event": "bonus:stack",
            "data": {"avatar": 1, "method": "add", "bonus": bonus_data},
        },
    ]

    assert env.random_calls == []
    assert frame["game"]["size"] == 88
    assert frame["game"]["worldBodyCount"] == 0
    assert frame["game"]["bonusCount"] == 0
    assert frame["game"]["bonusWorldBodyCount"] == 1
    assert avatars["p0"]["x"] == 21.6
    assert avatars["p0"]["y"] == 20
    assert avatars["p0"]["radius"] == 0.3
    assert avatars["p0"]["activeBonuses"] == [bonus_data]
    assert avatars["p1"]["radius"] == 0.6
    assert avatars["p1"]["activeBonuses"] == []
    assert _source_event_data(env) == expected_events

    js_payload = _run_js_scenario_oracle(_SCENARIO_DIR / "source_bonus_self_small_catch_step.json")
    if js_payload is not None:
        js_frame = js_payload["trace"][0]
        js_avatars = {avatar["name"]: avatar for avatar in js_frame["avatars"]}
        assert _source_event_data(env) == js_frame["events"]
        assert frame["game"]["bonusCount"] == js_frame["game"]["bonusCount"]
        assert frame["game"]["bonusWorldBodyCount"] == js_frame["game"]["bonusWorldBodyCount"]
        assert avatars["p0"]["radius"] == js_avatars["p0"]["radius"]
        assert avatars["p0"]["activeBonuses"] == js_avatars["p0"]["activeBonuses"]


def test_source_env_bonus_game_clear_immediate_step_matches_js_fixture():
    env, _scenario, frame = _run_source_bonus_fixture("source_bonus_game_clear_immediate_step.json")
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}
    expected_events = [
        {"event": "position", "data": {"avatar": 2, "x": 71.6, "y": 70}},
        {"event": "position", "data": {"avatar": 1, "x": 21.6, "y": 20}},
        {"event": "bonus:clear", "data": {"bonus": 1}},
        {"event": "clear", "data": {}},
    ]

    assert env.random_calls == []
    assert frame["game"]["size"] == 88
    assert frame["game"]["worldActive"] is True
    assert frame["game"]["worldBodyCount"] == 0
    assert frame["game"]["bonusCount"] == 0
    assert frame["game"]["bonusWorldBodyCount"] == 1
    assert avatars["p0"]["x"] == 21.6
    assert avatars["p0"]["y"] == 20
    assert avatars["p0"]["alive"] is True
    assert avatars["p0"]["radius"] == 0.6
    assert avatars["p0"]["activeBonuses"] == []
    assert avatars["p1"]["alive"] is True
    assert avatars["p1"]["radius"] == 0.6
    assert avatars["p1"]["activeBonuses"] == []
    assert _source_event_data(env) == expected_events
    assert not any(event["event"] == "bonus:stack" for event in _source_event_data(env))
    assert not any(event["event"] == "property" for event in _source_event_data(env))

    js_payload = _run_js_scenario_oracle(
        _SCENARIO_DIR / "source_bonus_game_clear_immediate_step.json"
    )
    if js_payload is not None:
        js_frame = js_payload["trace"][0]
        js_avatars = {avatar["name"]: avatar for avatar in js_frame["avatars"]}
        assert _source_event_data(env) == js_frame["events"]
        assert frame["game"]["worldActive"] == js_frame["game"]["worldActive"]
        assert frame["game"]["worldBodyCount"] == js_frame["game"]["worldBodyCount"]
        assert frame["game"]["bonusCount"] == js_frame["game"]["bonusCount"]
        assert frame["game"]["bonusWorldBodyCount"] == js_frame["game"]["bonusWorldBodyCount"]
        assert avatars["p0"]["alive"] == js_avatars["p0"]["alive"]
        assert avatars["p0"]["radius"] == js_avatars["p0"]["radius"]
        assert avatars["p0"]["activeBonuses"] == js_avatars["p0"]["activeBonuses"]
        assert avatars["p1"]["activeBonuses"] == js_avatars["p1"]["activeBonuses"]


def test_source_env_bonus_self_small_tangent_does_not_catch_like_js_fixture():
    env, _scenario, frame = _run_source_bonus_fixture(
        "source_bonus_self_small_tangent_no_catch_step.json"
    )
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}
    expected_events = [
        {"event": "position", "data": {"avatar": 2, "x": 71.6, "y": 70}},
        {"event": "position", "data": {"avatar": 1, "x": 21.6, "y": 20}},
    ]

    assert env.random_calls == []
    assert frame["game"]["bonusCount"] == 1
    assert frame["game"]["bonusWorldBodyCount"] == 1
    assert avatars["p0"]["x"] == 21.6
    assert avatars["p0"]["radius"] == 0.6
    assert avatars["p0"]["activeBonuses"] == []
    assert avatars["p1"]["radius"] == 0.6
    assert avatars["p1"]["activeBonuses"] == []
    assert _source_event_data(env) == expected_events

    js_payload = _run_js_scenario_oracle(
        _SCENARIO_DIR / "source_bonus_self_small_tangent_no_catch_step.json"
    )
    if js_payload is not None:
        js_frame = js_payload["trace"][0]
        js_avatars = {avatar["name"]: avatar for avatar in js_frame["avatars"]}
        assert _source_event_data(env) == js_frame["events"]
        assert frame["game"]["bonusCount"] == js_frame["game"]["bonusCount"]
        assert frame["game"]["bonusWorldBodyCount"] == js_frame["game"]["bonusWorldBodyCount"]
        assert avatars["p0"]["radius"] == js_avatars["p0"]["radius"]
        assert avatars["p0"]["activeBonuses"] == js_avatars["p0"]["activeBonuses"]


def test_source_env_bonus_self_small_wall_death_does_not_catch_like_js_fixture():
    env, _scenario, frame = _run_source_bonus_fixture(
        "source_bonus_self_small_wall_death_no_catch_step.json"
    )
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}
    expected_events = [
        {"event": "position", "data": {"avatar": 2, "x": 71.6, "y": 70}},
        {"event": "position", "data": {"avatar": 1, "x": 87.9, "y": 20}},
        {"event": "die", "data": {"avatar": 1, "killer": None, "old": None}},
        {"event": "score:round", "data": {"avatar": 1, "score": 0, "roundScore": 0}},
        {"event": "score:round", "data": {"avatar": 2, "score": 0, "roundScore": 1}},
        {"event": "score", "data": {"avatar": 2, "score": 1, "roundScore": 1}},
        {"event": "score", "data": {"avatar": 1, "score": 0, "roundScore": 0}},
        {"event": "round:end", "data": {"winner": 2}},
    ]

    assert env.random_calls == []
    assert env.game is not None
    assert env.game.death_ids == [1]
    assert frame["game"]["inRound"] is False
    assert frame["game"]["worldBodyCount"] == 1
    assert frame["game"]["bonusCount"] == 1
    assert frame["game"]["bonusWorldBodyCount"] == 1
    assert avatars["p0"]["x"] == 87.9
    assert avatars["p0"]["alive"] is False
    assert avatars["p0"]["radius"] == 0.6
    assert avatars["p0"]["activeBonuses"] == []
    assert avatars["p1"]["alive"] is True
    assert avatars["p1"]["score"] == 1
    assert _source_event_data(env) == expected_events

    js_payload = _run_js_scenario_oracle(
        _SCENARIO_DIR / "source_bonus_self_small_wall_death_no_catch_step.json"
    )
    if js_payload is not None:
        js_frame = js_payload["trace"][0]
        js_avatars = {avatar["name"]: avatar for avatar in js_frame["avatars"]}
        js_events_without_non_important_points = [
            event for event in js_frame["events"] if event["event"] != "point"
        ]
        assert _source_event_data(env) == js_events_without_non_important_points
        assert frame["game"]["worldBodyCount"] == js_frame["game"]["worldBodyCount"]
        assert frame["game"]["bonusCount"] == js_frame["game"]["bonusCount"]
        assert frame["game"]["bonusWorldBodyCount"] == js_frame["game"]["bonusWorldBodyCount"]
        assert avatars["p0"]["alive"] == js_avatars["p0"]["alive"]
        assert avatars["p0"]["radius"] == js_avatars["p0"]["radius"]
        assert avatars["p0"]["activeBonuses"] == js_avatars["p0"]["activeBonuses"]
        assert avatars["p1"]["score"] == js_avatars["p1"]["score"]


def test_source_env_bonus_self_small_expiry_restores_radius_like_js_fixture():
    env, _scenario, frames = _run_source_bonus_fixture_frames(
        "source_bonus_self_small_expiry_restore_step.json"
    )
    catch_frame, expiry_frame = frames
    catch_avatars = {avatar["name"]: avatar for avatar in catch_frame["avatars"]}
    expiry_avatars = {avatar["name"]: avatar for avatar in expiry_frame["avatars"]}
    bonus_data = {
        "id": 1,
        "type": "BonusSelfSmall",
        "duration": 7500,
        "effects": [["radius", -1]],
    }

    assert env.random_calls == []
    assert catch_avatars["p0"]["radius"] == 0.3
    assert catch_avatars["p0"]["activeBonuses"] == [bonus_data]
    assert catch_frame["game"]["bonusCount"] == 0
    assert expiry_frame["game"]["bonusCount"] == 0
    assert expiry_frame["game"]["bonusWorldBodyCount"] == 1
    assert expiry_avatars["p0"]["radius"] == 0.6
    assert expiry_avatars["p0"]["activeBonuses"] == []
    assert expiry_avatars["p1"]["radius"] == 0.6
    assert expiry_avatars["p1"]["activeBonuses"] == []
    assert expiry_frame["events"] == [
        {
            "event": "property",
            "data": {"avatar": 1, "property": "radius", "value": 0.6},
        },
        {
            "event": "bonus:stack",
            "data": {"avatar": 1, "method": "remove", "bonus": bonus_data},
        },
        {"event": "position", "data": {"avatar": 2, "x": 71.6, "y": 70}},
        {"event": "position", "data": {"avatar": 1, "x": 21.6, "y": 20}},
    ]
    assert not any(event["event"] == "bonus:clear" for event in expiry_frame["events"])

    js_payload = _run_js_scenario_oracle(
        _SCENARIO_DIR / "source_bonus_self_small_expiry_restore_step.json"
    )
    if js_payload is not None:
        js_expiry_frame = js_payload["trace"][1]
        js_expiry_avatars = {avatar["name"]: avatar for avatar in js_expiry_frame["avatars"]}
        assert expiry_frame["events"] == js_expiry_frame["events"]
        assert expiry_frame["game"]["bonusCount"] == js_expiry_frame["game"]["bonusCount"]
        assert (
            expiry_frame["game"]["bonusWorldBodyCount"]
            == js_expiry_frame["game"]["bonusWorldBodyCount"]
        )
        assert expiry_avatars["p0"]["radius"] == js_expiry_avatars["p0"]["radius"]
        assert expiry_avatars["p0"]["activeBonuses"] == js_expiry_avatars["p0"]["activeBonuses"]


def test_source_env_bonus_self_speed_stack_expiry_restores_prior_velocity():
    env = _active_source_env()
    assert env.game is not None
    env.game.print_start_due_ms = None
    env.events.clear()
    env.random.calls.clear()

    avatar = env.avatar_by_id(1)
    env.seed_active_bonus("BonusSelfSlow", x=avatar.x, y=avatar.y)
    env.step({}, elapsed_ms=0)

    assert avatar.velocity == 8
    assert [bonus.type for bonus in avatar.active_bonuses] == ["BonusSelfSlow"]

    env.seed_active_bonus("BonusSelfFast", x=avatar.x, y=avatar.y)
    env.step({}, elapsed_ms=0)

    assert avatar.velocity == 20
    assert [bonus.type for bonus in avatar.active_bonuses] == [
        "BonusSelfSlow",
        "BonusSelfFast",
    ]

    env.events.clear()
    env.advance_timers(4000)

    assert avatar.velocity == 8
    assert [bonus.type for bonus in avatar.active_bonuses] == ["BonusSelfSlow"]
    assert [event["data"] for event in env.events if event["event"] == "property"] == [
        {"avatar": 1, "property": "velocity", "value": 8}
    ]

    env.events.clear()
    env.advance_timers(1000)

    assert avatar.velocity == 16
    assert avatar.active_bonuses == []
    assert [event["data"] for event in env.events if event["event"] == "property"] == [
        {"avatar": 1, "property": "velocity", "value": 16}
    ]


def test_source_env_bonus_stack_death_late_expiry_matches_js_inertness():
    env, _scenario, frames = _run_source_bonus_fixture_frames(
        "source_bonus_self_fast_stack_death_late_expiry_step.json"
    )
    first, second, third, death_frame, expiry_frame = frames

    first_p0 = next(avatar for avatar in first["avatars"] if avatar["name"] == "p0")
    second_p0 = next(avatar for avatar in second["avatars"] if avatar["name"] == "p0")
    third_p0 = next(avatar for avatar in third["avatars"] if avatar["name"] == "p0")
    assert [len(first_p0["activeBonuses"]), len(second_p0["activeBonuses"])] == [1, 2]
    assert len(third_p0["activeBonuses"]) == 3

    death_p0 = next(avatar for avatar in death_frame["avatars"] if avatar["name"] == "p0")
    expiry_p0 = next(avatar for avatar in expiry_frame["avatars"] if avatar["name"] == "p0")
    avatar = env.avatar_by_id(1)
    assert death_frame["game"]["inRound"] is False
    assert death_p0["alive"] is False
    assert death_p0["activeBonuses"] == []
    assert [
        event["data"]["avatar"] for event in death_frame["events"] if event["event"] == "die"
    ] == [1]
    assert expiry_p0["alive"] is False
    assert expiry_p0["activeBonuses"] == []
    assert avatar.velocity == 52
    assert avatar.active_bonuses == []
    assert [
        event["data"]["bonus"]["id"]
        for event in expiry_frame["events"]
        if event["event"] == "bonus:stack"
    ] == [3, 2, 1]
    assert not any(event["event"] == "property" for event in expiry_frame["events"])

    js_payload = _run_js_scenario_oracle(
        _SCENARIO_DIR / "source_bonus_self_fast_stack_death_late_expiry_step.json"
    )
    if js_payload is not None:
        js_expiry_frame = js_payload["trace"][-1]
        js_expiry_p0 = next(
            avatar for avatar in js_expiry_frame["avatars"] if avatar["name"] == "p0"
        )
        assert [
            event
            for event in expiry_frame["events"]
            if event["event"] == "bonus:stack"
        ] == [
            event
            for event in js_expiry_frame["events"]
            if event["event"] == "bonus:stack"
        ]
        assert not any(event["event"] == "property" for event in js_expiry_frame["events"])
        assert expiry_p0["activeBonuses"] == js_expiry_p0["activeBonuses"]
        assert avatar.velocity == js_expiry_p0["velocity"]


def test_source_env_bonus_self_fast_expiry_before_wall_death_matches_js():
    env, _scenario, frames = _run_source_bonus_fixture_frames(
        "source_bonus_self_fast_expiry_then_wall_death_same_tick_step.json"
    )
    catch_frame, death_frame = frames
    bonus_data = {
        "id": 1,
        "type": "BonusSelfFast",
        "duration": 4000,
        "effects": [["velocity", 12]],
    }

    catch_p0 = next(avatar for avatar in catch_frame["avatars"] if avatar["name"] == "p0")
    death_p0 = next(avatar for avatar in death_frame["avatars"] if avatar["name"] == "p0")
    death_p1 = next(avatar for avatar in death_frame["avatars"] if avatar["name"] == "p1")
    avatar = env.avatar_by_id(1)

    assert catch_p0["activeBonuses"] == [bonus_data]
    assert death_frame["game"]["inRound"] is False
    assert death_p0["alive"] is False
    assert death_p0["x"] == -1.4
    assert death_p0["activeBonuses"] == []
    assert death_p1["score"] == 1
    assert avatar.velocity == 16
    assert avatar.active_bonuses == []
    assert {
        "event": "property",
        "data": {"avatar": 1, "property": "velocity", "value": 28},
    } in catch_frame["events"]
    assert death_frame["events"][:2] == [
        {
            "event": "property",
            "data": {"avatar": 1, "property": "velocity", "value": 16},
        },
        {
            "event": "bonus:stack",
            "data": {"avatar": 1, "method": "remove", "bonus": bonus_data},
        },
    ]
    death_event_names = [event["event"] for event in death_frame["events"]]
    assert death_event_names.index("bonus:stack") < death_event_names.index("die")
    assert {"event": "die", "data": {"avatar": 1, "killer": None, "old": None}} in (
        death_frame["events"]
    )
    assert {"event": "round:end", "data": {"winner": 2}} in death_frame["events"]
    assert not any(event["event"] == "bonus:clear" for event in death_frame["events"])

    js_payload = _run_js_scenario_oracle(
        _SCENARIO_DIR / "source_bonus_self_fast_expiry_then_wall_death_same_tick_step.json"
    )
    if js_payload is not None:
        js_death_frame = js_payload["trace"][-1]
        js_death_p0 = next(
            avatar for avatar in js_death_frame["avatars"] if avatar["name"] == "p0"
        )
        js_event_names = [event["event"] for event in js_death_frame["events"]]
        assert js_event_names[:2] == ["property", "bonus:stack"]
        assert js_event_names.index("bonus:stack") < js_event_names.index("die")
        assert death_p0["x"] == js_death_p0["x"]
        assert avatar.velocity == js_death_p0["velocity"]
        assert death_p0["activeBonuses"] == js_death_p0["activeBonuses"]


def test_source_env_bonus_enemy_big_targets_other_alive_avatars_and_expires():
    env = _active_source_env(player_count=3)
    assert env.game is not None
    env.game.print_start_due_ms = None
    env.set_avatar_state(3, x=80, y=40, angle=0)
    env.events.clear()

    catcher = env.avatar_by_id(1)
    enemy_a = env.avatar_by_id(2)
    enemy_b = env.avatar_by_id(3)
    env.seed_active_bonus("BonusEnemyBig", x=catcher.x, y=catcher.y)
    env.step({}, elapsed_ms=0)

    assert catcher.radius == 0.6
    assert enemy_a.radius == 1.2
    assert enemy_b.radius == 1.2
    assert [bonus.type for bonus in enemy_a.active_bonuses] == ["BonusEnemyBig"]
    assert [bonus.type for bonus in enemy_b.active_bonuses] == ["BonusEnemyBig"]

    env.events.clear()
    env.advance_timers(7500)

    assert catcher.radius == 0.6
    assert enemy_a.radius == 0.6
    assert enemy_b.radius == 0.6
    assert enemy_a.active_bonuses == []
    assert enemy_b.active_bonuses == []


def test_source_env_4p_bonus_targets_skip_dead_and_absent_avatars():
    players = [
        {"avatar_id": 1, "name": "p0", "color": "#ff0000"},
        {"avatar_id": 2, "name": "p1", "color": "#00ff00"},
        {"avatar_id": 3, "name": "p2", "color": "#0000ff"},
        {"avatar_id": 4, "name": "p3", "color": "#ffff00"},
    ]
    env = CurvyTronSourceEnv()
    env.reset(
        player_count=4,
        players=players,
        present=[True, True, True, False],
        warmup_ms=0,
    )
    env.advance_timers(0)
    assert env.game is not None
    env.game.print_start_due_ms = None
    for avatar_id, x in ((1, 20), (2, 50), (3, 80), (4, 90)):
        env.set_avatar_state(avatar_id, x=x, y=20, angle=0)
    env.avatar_by_id(3).alive = False
    assert env.avatar_by_id(4).present is False
    assert env.avatar_by_id(4).alive is False
    env.events.clear()

    env.seed_active_bonus("BonusEnemySlow", x=20, y=20)
    env.step({}, elapsed_ms=0)

    assert [avatar.velocity for avatar in env.avatars] == [16.0, 8.0, 16.0, 16.0]
    assert [[bonus.type for bonus in avatar.active_bonuses] for avatar in env.avatars] == [
        [],
        ["BonusEnemySlow"],
        [],
        [],
    ]
    assert [
        event["data"]
        for event in env.events
        if event["event"] == "bonus:stack" and event["data"]["method"] == "add"
    ] == [
        {
            "avatar": 2,
            "method": "add",
            "bonus": {
                "id": 1,
                "type": "BonusEnemySlow",
                "duration": 5000,
                "effects": [["velocity", -8.0]],
            },
        },
    ]

    env.advance_timers(5000)

    assert [avatar.velocity for avatar in env.avatars] == [16.0, 16.0, 16.0, 16.0]
    assert all(not avatar.active_bonuses for avatar in env.avatars)


@pytest.mark.parametrize(
    "bonus_type",
    ["BonusEnemyInverse", "BonusEnemyStraightAngle"],
)
def test_source_env_bonus_enemy_turn_modifiers_target_enemy_and_expire(
    bonus_type: str,
):
    env = _active_source_env()
    assert env.game is not None
    env.game.print_start_due_ms = None
    env.events.clear()

    catcher = env.avatar_by_id(1)
    target = env.avatar_by_id(2)
    env.seed_active_bonus(bonus_type, x=catcher.x, y=catcher.y)
    env.step({}, elapsed_ms=0)

    if bonus_type == "BonusEnemyInverse":
        assert catcher.inverse is False
        assert target.inverse is True
    else:
        assert catcher.direction_in_loop is True
        assert target.direction_in_loop is False
        assert math.isclose(target.angular_velocity_base, math.pi / 2.0)

    env.advance_timers(5000)

    assert target.inverse is False
    assert target.direction_in_loop is True
    assert math.isclose(
        target.angular_velocity_base,
        env.reference.angular_velocity_radians_per_ms,
    )


def test_source_env_bonus_self_master_blocks_body_death_but_not_wall_death():
    env = _active_source_env()
    assert env.game is not None
    env.game.print_start_due_ms = None

    catcher = env.avatar_by_id(1)
    enemy = env.avatar_by_id(2)
    env.seed_active_bonus("BonusSelfMaster", x=catcher.x, y=catcher.y)
    env.step({}, elapsed_ms=0)

    assert catcher.invincible is True
    env.events.clear()
    _seed_source_body(
        env,
        owner_id=enemy.id,
        x=catcher.x,
        y=catcher.y,
        radius=1.0,
    )
    env.step({}, elapsed_ms=0)

    assert catcher.alive is True
    assert not [event for event in env.events if event["event"] == "die"]

    env.events.clear()
    env.set_avatar_state(catcher.id, x=0.3, y=20.0, angle=math.pi)
    env.step({}, elapsed_ms=0)

    assert catcher.alive is False
    assert [event["data"] for event in env.events if event["event"] == "die"] == [
        {"avatar": catcher.id, "killer": None, "old": None}
    ]


def test_source_env_bonus_all_color_rotates_alive_colors_and_expires():
    players = [
        {"avatar_id": 1, "name": "p0", "color": "#ff0000"},
        {"avatar_id": 2, "name": "p1", "color": "#00ff00"},
        {"avatar_id": 3, "name": "p2", "color": "#0000ff"},
    ]
    env = CurvyTronSourceEnv()
    env.reset(player_count=3, players=players, warmup_ms=0)
    env.advance_timers(0)
    assert env.game is not None
    env.game.print_start_due_ms = None
    env.set_avatar_state(1, x=20, y=20, angle=0)
    env.set_avatar_state(2, x=50, y=20, angle=0)
    env.set_avatar_state(3, x=80, y=20, angle=0)
    env.events.clear()

    env.seed_active_bonus("BonusAllColor", x=20, y=20)
    env.step({}, elapsed_ms=0)

    assert [avatar.color for avatar in env.avatars] == [
        "#00ff00",
        "#0000ff",
        "#ff0000",
    ]
    assert [
        event["data"]
        for event in env.events
        if event["event"] == "property" and event["data"]["property"] == "color"
    ] == [
        {"avatar": 3, "property": "color", "value": "#ff0000"},
        {"avatar": 2, "property": "color", "value": "#0000ff"},
        {"avatar": 1, "property": "color", "value": "#00ff00"},
    ]

    env.events.clear()
    env.advance_timers(7500)

    assert [avatar.color for avatar in env.avatars] == [
        "#ff0000",
        "#00ff00",
        "#0000ff",
    ]
    assert [
        event["data"]
        for event in env.events
        if event["event"] == "property" and event["data"]["property"] == "color"
    ] == [
        {"avatar": 3, "property": "color", "value": "#0000ff"},
        {"avatar": 2, "property": "color", "value": "#00ff00"},
        {"avatar": 1, "property": "color", "value": "#ff0000"},
    ]


def test_source_env_bonus_all_color_overlap_uses_older_stack_until_it_expires():
    players = [
        {"avatar_id": 1, "name": "p0", "color": "#ff0000"},
        {"avatar_id": 2, "name": "p1", "color": "#00ff00"},
        {"avatar_id": 3, "name": "p2", "color": "#0000ff"},
    ]
    env = CurvyTronSourceEnv()
    env.reset(player_count=3, players=players, warmup_ms=0)
    env.advance_timers(0)
    assert env.game is not None
    env.game.print_start_due_ms = None
    env.set_avatar_state(1, x=20, y=20, angle=0)
    env.set_avatar_state(2, x=50, y=20, angle=0)
    env.set_avatar_state(3, x=80, y=20, angle=0)

    catcher = env.avatar_by_id(1)
    env.seed_active_bonus("BonusAllColor", x=catcher.x, y=catcher.y)
    env.step({}, elapsed_ms=0)

    assert [avatar.color for avatar in env.avatars] == [
        "#00ff00",
        "#0000ff",
        "#ff0000",
    ]

    env.advance_timers(1)
    env.seed_active_bonus("BonusAllColor", x=catcher.x, y=catcher.y)
    env.step({}, elapsed_ms=0)

    assert [avatar.color for avatar in env.avatars] == [
        "#00ff00",
        "#0000ff",
        "#ff0000",
    ]
    assert [len(avatar.active_bonuses) for avatar in env.avatars] == [2, 2, 2]

    env.advance_timers(7499)

    assert [avatar.color for avatar in env.avatars] == [
        "#0000ff",
        "#ff0000",
        "#00ff00",
    ]
    assert [len(avatar.active_bonuses) for avatar in env.avatars] == [1, 1, 1]

    env.advance_timers(1)

    assert [avatar.color for avatar in env.avatars] == [
        "#ff0000",
        "#00ff00",
        "#0000ff",
    ]
    assert [avatar.active_bonuses for avatar in env.avatars] == [[], [], []]


def test_source_env_bonus_game_borderless_expiry_restores_like_js_fixture():
    scenario_path = _SCENARIO_DIR / "source_bonus_game_borderless_expiry_restore_step.json"
    js_payload = _run_js_scenario_oracle(scenario_path)
    if js_payload is not None:
        js_catch_frame, js_expiry_frame = js_payload["trace"]
        normalized_js_expiry_events = [
            (
                {"event": "borderless", "data": {"value": False}}
                if event["event"] == "borderless"
                else event
            )
            for event in js_expiry_frame["events"]
        ]
        assert js_catch_frame["game"]["borderless"] is True
        assert js_catch_frame["game"]["bonusCount"] == 0
        assert js_catch_frame["events"] == [
            {"event": "position", "data": {"avatar": 2, "x": 71.6, "y": 70}},
            {"event": "position", "data": {"avatar": 1, "x": 21.6, "y": 20}},
            {"event": "bonus:clear", "data": {"bonus": 1}},
            {"event": "borderless", "data": {"value": True}},
        ]
        assert js_expiry_frame["game"]["borderless"] is False
        assert js_expiry_frame["game"]["bonusCount"] == 0
        assert normalized_js_expiry_events == [
            {"event": "borderless", "data": {"value": False}},
            {"event": "position", "data": {"avatar": 2, "x": 71.6, "y": 70}},
            {"event": "position", "data": {"avatar": 1, "x": 21.6, "y": 20}},
        ]
        assert not any(event["event"] == "bonus:clear" for event in js_expiry_frame["events"])

    env, _scenario, frames = _run_source_bonus_fixture_frames(
        "source_bonus_game_borderless_expiry_restore_step.json"
    )

    catch_frame, expiry_frame = frames
    catch_events_without_stack = [
        event for event in catch_frame["events"] if event["event"] != "bonus:stack"
    ]
    expiry_events_without_stack = [
        event for event in expiry_frame["events"] if event["event"] != "bonus:stack"
    ]

    assert env.random_calls == []
    assert catch_frame["game"]["borderless"] is True
    assert catch_frame["game"]["bonusCount"] == 0
    assert expiry_frame["game"]["borderless"] is False
    assert expiry_frame["game"]["bonusCount"] == 0
    assert not any(event["event"] == "bonus:clear" for event in expiry_frame["events"])

    if js_payload is not None:
        js_catch_frame, js_expiry_frame = js_payload["trace"]
        normalized_js_expiry_events = [
            (
                {"event": "borderless", "data": {"value": False}}
                if event["event"] == "borderless"
                else event
            )
            for event in js_expiry_frame["events"]
        ]
        assert catch_events_without_stack == js_catch_frame["events"]
        assert expiry_events_without_stack == normalized_js_expiry_events
        assert catch_frame["game"]["borderless"] == js_catch_frame["game"]["borderless"]
        assert expiry_frame["game"]["borderless"] == js_expiry_frame["game"]["borderless"]


def test_source_env_matches_js_oracle_for_natural_bonus_spawn_type_position_rng_fixture():
    scenario_path = _SCENARIO_DIR / "source_bonus_spawn_type_position_rng_step.json"
    js_payload = _run_js_scenario_oracle(scenario_path)
    scenario = _load_scenario(scenario_path)
    source_setup = scenario["source_setup"]
    assert isinstance(source_setup, dict)
    room = source_setup["room"]
    assert isinstance(room, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)

    env = CurvyTronSourceEnv(
        random_constant=0.5,
        max_score=float(room["max_score"]),
        include_bonus_snapshot=True,
        emit_step_position_events=True,
    )
    env.reset(
        player_count=int(scenario["player_count"]),
        players=scenario["players"],
        warmup_ms=0,
    )
    for player in scenario["players"]:
        initial = player["initial"]
        avatar = env.set_avatar_state(
            player["avatar_id"],
            x=initial["x"],
            y=initial["y"],
            angle=initial["angle_rad"],
        )
        avatar.printing = bool(initial["printing"])

    env.set_random_sequence(random_setup["math_random_sequence"])
    env.source_game_on_start(
        bonus_types=room["bonuses"],
        bonus_rate=float(room["bonus_rate"]),
    )
    env.events.clear()

    step = scenario["steps"][0]
    env.advance_timers(float(step["advance_timers_ms"]))
    frame = env.step(
        [move["move"] for move in step["moves"]],
        elapsed_ms=float(step["step_ms"]),
    )

    js_frame = js_payload["trace"][0]
    js_avatars = {avatar["name"]: avatar for avatar in js_frame["avatars"]}
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert _source_random_call_labels(env) == js_payload["randomCalls"]
    assert _source_gameplay_event_data(env) == js_frame["events"]
    assert frame["game"]["size"] == js_frame["game"]["size"]
    assert frame["game"]["worldBodyCount"] == js_frame["game"]["worldBodyCount"]
    assert frame["game"]["bonusCount"] == js_frame["game"]["bonusCount"]
    assert frame["game"]["bonusWorldBodyCount"] == js_frame["game"]["bonusWorldBodyCount"]
    assert avatars["p0"]["x"] == js_avatars["p0"]["x"]
    assert avatars["p0"]["y"] == js_avatars["p0"]["y"]
    assert avatars["p0"]["activeBonuses"] == js_avatars["p0"]["activeBonuses"]
    assert avatars["p1"]["x"] == js_avatars["p1"]["x"]
    assert avatars["p1"]["y"] == js_avatars["p1"]["y"]
    assert avatars["p1"]["activeBonuses"] == js_avatars["p1"]["activeBonuses"]
    assert len(env.active_bonuses) == 1
    bonus = env.active_bonuses[0]
    assert bonus.type == "BonusSelfSmall"
    assert source_env._source_number(bonus.x) == 23.94
    assert source_env._source_number(bonus.y) == 64.06


def test_source_env_matches_js_oracle_for_natural_bonus_spawn_game_world_retry_fixture():
    scenario_path = _SCENARIO_DIR / "source_bonus_spawn_game_world_retry_step.json"
    js_payload = _run_js_scenario_oracle(scenario_path)
    scenario = _load_scenario(scenario_path)
    source_setup = scenario["source_setup"]
    assert isinstance(source_setup, dict)
    room = source_setup["room"]
    assert isinstance(room, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)

    env = CurvyTronSourceEnv(
        random_constant=0.5,
        max_score=float(room["max_score"]),
        include_bonus_snapshot=True,
        emit_step_position_events=True,
    )
    env.reset(
        player_count=int(scenario["player_count"]),
        players=scenario["players"],
        warmup_ms=0,
    )
    for player in scenario["players"]:
        initial = player["initial"]
        avatar = env.set_avatar_state(
            player["avatar_id"],
            x=initial["x"],
            y=initial["y"],
            angle=initial["angle_rad"],
        )
        avatar.printing = bool(initial["printing"])

    assert env.game is not None
    assert env.game.world is not None
    env.game.world.activate()
    initial_state = scenario["initial_state"]
    assert isinstance(initial_state, dict)
    world_body = initial_state["world_bodies"][0]
    _seed_source_body(
        env,
        owner_id=1,
        x=float(world_body["x"]),
        y=float(world_body["y"]),
        num=int(world_body["num"]),
    )

    env.set_random_sequence(random_setup["math_random_sequence"])
    env.source_game_on_start(
        bonus_types=room["bonuses"],
        bonus_rate=float(room["bonus_rate"]),
    )
    env.events.clear()

    step = scenario["steps"][0]
    env.advance_timers(float(step["advance_timers_ms"]))
    frame = env.step(
        [move["move"] for move in step["moves"]],
        elapsed_ms=float(step["step_ms"]),
    )

    js_frame = js_payload["trace"][0]
    js_avatars = {avatar["name"]: avatar for avatar in js_frame["avatars"]}
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert _source_random_call_labels(env) == js_payload["randomCalls"]
    assert len([call for call in env.random_calls if ".retry_" in call["label"]["site"]]) == 2
    assert _source_gameplay_event_data(env) == js_frame["events"]
    assert frame["game"]["size"] == js_frame["game"]["size"]
    assert frame["game"]["worldBodyCount"] == js_frame["game"]["worldBodyCount"]
    assert frame["game"]["bonusCount"] == js_frame["game"]["bonusCount"]
    assert frame["game"]["bonusWorldBodyCount"] == js_frame["game"]["bonusWorldBodyCount"]
    assert avatars["p0"]["x"] == js_avatars["p0"]["x"]
    assert avatars["p0"]["y"] == js_avatars["p0"]["y"]
    assert avatars["p0"]["activeBonuses"] == js_avatars["p0"]["activeBonuses"]
    assert avatars["p1"]["x"] == js_avatars["p1"]["x"]
    assert avatars["p1"]["y"] == js_avatars["p1"]["y"]
    assert avatars["p1"]["activeBonuses"] == js_avatars["p1"]["activeBonuses"]
    assert len(env.active_bonuses) == 1
    bonus = env.active_bonuses[0]
    assert bonus.type == "BonusSelfSmall"
    assert source_env._source_number(bonus.x) == 68.072
    assert source_env._source_number(bonus.y) == 19.928


def test_source_env_matches_js_oracle_for_natural_bonus_spawn_bonus_world_retry_fixture():
    scenario_path = _SCENARIO_DIR / "source_bonus_spawn_bonus_world_retry_step.json"
    js_payload = _run_js_scenario_oracle(scenario_path)
    scenario = _load_scenario(scenario_path)
    source_setup = scenario["source_setup"]
    assert isinstance(source_setup, dict)
    room = source_setup["room"]
    assert isinstance(room, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)

    env = CurvyTronSourceEnv(
        random_constant=0.5,
        max_score=float(room["max_score"]),
        include_bonus_snapshot=True,
        emit_step_position_events=True,
    )
    env.reset(
        player_count=int(scenario["player_count"]),
        players=scenario["players"],
        warmup_ms=0,
    )
    for player in scenario["players"]:
        initial = player["initial"]
        avatar = env.set_avatar_state(
            player["avatar_id"],
            x=initial["x"],
            y=initial["y"],
            angle=initial["angle_rad"],
        )
        avatar.printing = bool(initial["printing"])

    env.set_random_sequence(random_setup["math_random_sequence"])
    env.source_game_on_start(
        bonus_types=room["bonuses"],
        bonus_rate=float(room["bonus_rate"]),
    )
    initial_state = scenario["initial_state"]
    assert isinstance(initial_state, dict)
    seeded_bonus = initial_state["active_bonuses"][0]
    assert isinstance(seeded_bonus, dict)
    env.seed_active_bonus(
        str(seeded_bonus["type"]),
        x=float(seeded_bonus["x"]),
        y=float(seeded_bonus["y"]),
    )
    env.events.clear()

    step = scenario["steps"][0]
    env.advance_timers(float(step["advance_timers_ms"]))
    frame = env.step(
        [move["move"] for move in step["moves"]],
        elapsed_ms=float(step["step_ms"]),
    )

    js_frame = js_payload["trace"][0]
    js_avatars = {avatar["name"]: avatar for avatar in js_frame["avatars"]}
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert _source_random_call_labels(env) == js_payload["randomCalls"]
    assert len([call for call in env.random_calls if ".retry_" in call["label"]["site"]]) == 2
    assert _source_gameplay_event_data(env) == js_frame["events"]
    assert frame["game"]["size"] == js_frame["game"]["size"]
    assert frame["game"]["worldBodyCount"] == js_frame["game"]["worldBodyCount"] == 0
    assert frame["game"]["bonusCount"] == js_frame["game"]["bonusCount"] == 2
    assert frame["game"]["bonusWorldBodyCount"] == js_frame["game"]["bonusWorldBodyCount"] == 2
    assert avatars["p0"]["x"] == js_avatars["p0"]["x"]
    assert avatars["p0"]["y"] == js_avatars["p0"]["y"]
    assert avatars["p0"]["activeBonuses"] == js_avatars["p0"]["activeBonuses"]
    assert avatars["p1"]["x"] == js_avatars["p1"]["x"]
    assert avatars["p1"]["y"] == js_avatars["p1"]["y"]
    assert avatars["p1"]["activeBonuses"] == js_avatars["p1"]["activeBonuses"]
    assert len(env.active_bonuses) == 2
    bonus = env.active_bonuses[1]
    assert bonus.type == "BonusSelfSmall"
    assert source_env._source_number(bonus.x) == 68.072
    assert source_env._source_number(bonus.y) == 19.928


def test_source_env_matches_js_oracle_for_natural_bonus_spawn_cap_fixture():
    scenario_path = _SCENARIO_DIR / "source_bonus_spawn_cap_twenty_step.json"
    js_payload = _run_js_scenario_oracle(scenario_path)
    scenario = _load_scenario(scenario_path)
    source_setup = scenario["source_setup"]
    assert isinstance(source_setup, dict)
    room = source_setup["room"]
    assert isinstance(room, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)

    env = CurvyTronSourceEnv(
        random_constant=0.5,
        max_score=float(room["max_score"]),
        include_bonus_snapshot=True,
        emit_step_position_events=True,
    )
    env.reset(
        player_count=int(scenario["player_count"]),
        players=scenario["players"],
        warmup_ms=0,
    )
    for player in scenario["players"]:
        initial = player["initial"]
        avatar = env.set_avatar_state(
            player["avatar_id"],
            x=initial["x"],
            y=initial["y"],
            angle=initial["angle_rad"],
        )
        avatar.printing = bool(initial["printing"])

    env.set_random_sequence(random_setup["math_random_sequence"])
    env.source_game_on_start(
        bonus_types=room["bonuses"],
        bonus_rate=float(room["bonus_rate"]),
    )
    initial_state = scenario["initial_state"]
    assert isinstance(initial_state, dict)
    for bonus in initial_state["active_bonuses"]:
        assert isinstance(bonus, dict)
        env.seed_active_bonus(str(bonus["type"]), x=float(bonus["x"]), y=float(bonus["y"]))
    env.events.clear()

    step = scenario["steps"][0]
    env.advance_timers(float(step["advance_timers_ms"]))
    frame = env.step(
        [move["move"] for move in step["moves"]],
        elapsed_ms=float(step["step_ms"]),
    )

    js_frame = js_payload["trace"][0]
    js_avatars = {avatar["name"]: avatar for avatar in js_frame["avatars"]}
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert _source_random_call_labels(env) == js_payload["randomCalls"]
    assert _source_gameplay_event_data(env) == js_frame["events"]
    assert frame["game"]["size"] == js_frame["game"]["size"]
    assert frame["game"]["worldBodyCount"] == js_frame["game"]["worldBodyCount"]
    assert frame["game"]["bonusCount"] == js_frame["game"]["bonusCount"] == 20
    assert frame["game"]["bonusWorldBodyCount"] == js_frame["game"]["bonusWorldBodyCount"] == 20
    assert avatars["p0"]["activeBonuses"] == js_avatars["p0"]["activeBonuses"] == []
    assert avatars["p1"]["activeBonuses"] == js_avatars["p1"]["activeBonuses"] == []
    assert len(env.active_bonuses) == 20
    assert not any(event["event"] == "bonus:pop" for event in _source_gameplay_event_data(env))


def test_source_env_matches_js_oracle_for_default_bonus_weight_type_rng_fixture():
    scenario_path = _SCENARIO_DIR / "source_bonus_default_weights_type_rng_step.json"
    js_payload = _run_js_scenario_oracle(scenario_path)
    scenario = _load_scenario(scenario_path)
    source_setup = scenario["source_setup"]
    assert isinstance(source_setup, dict)
    room = source_setup["room"]
    assert isinstance(room, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)

    env = CurvyTronSourceEnv(
        random_constant=0.5,
        max_score=float(room["max_score"]),
        include_bonus_snapshot=True,
        emit_step_position_events=True,
    )
    env.reset(
        player_count=int(scenario["player_count"]),
        players=scenario["players"],
        warmup_ms=0,
    )
    for player in scenario["players"]:
        initial = player["initial"]
        avatar = env.set_avatar_state(
            player["avatar_id"],
            x=initial["x"],
            y=initial["y"],
            angle=initial["angle_rad"],
        )
        avatar.printing = bool(initial["printing"])
        avatar.alive = bool(initial.get("alive", True))

    env.set_random_sequence(random_setup["math_random_sequence"])
    env.source_game_on_start(
        bonus_types=room["bonuses"],
        bonus_rate=float(room["bonus_rate"]),
    )
    env.events.clear()

    step = scenario["steps"][0]
    env.advance_timers(float(step["advance_timers_ms"]))
    frame = env.step(
        [move["move"] for move in step["moves"]],
        elapsed_ms=float(step["step_ms"]),
    )

    js_frame = js_payload["trace"][0]
    js_bonus_event = next(event for event in js_frame["events"] if event["event"] == "bonus:pop")
    js_avatars = {avatar["name"]: avatar for avatar in js_frame["avatars"]}
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert _source_random_call_labels(env) == js_payload["randomCalls"]
    assert _source_gameplay_event_data(env) == js_frame["events"]
    assert frame["game"]["size"] == js_frame["game"]["size"]
    assert frame["game"]["worldBodyCount"] == js_frame["game"]["worldBodyCount"]
    assert frame["game"]["bonusCount"] == js_frame["game"]["bonusCount"]
    assert frame["game"]["bonusWorldBodyCount"] == js_frame["game"]["bonusWorldBodyCount"]
    assert avatars["p0"]["alive"] == js_avatars["p0"]["alive"]
    assert avatars["p1"]["alive"] == js_avatars["p1"]["alive"]
    assert avatars["p2"]["alive"] == js_avatars["p2"]["alive"]
    assert avatars["p3"]["alive"] == js_avatars["p3"]["alive"]
    assert avatars["p0"]["activeBonuses"] == js_avatars["p0"]["activeBonuses"]
    assert avatars["p1"]["activeBonuses"] == js_avatars["p1"]["activeBonuses"]
    assert len(env.active_bonuses) == 1
    bonus = env.active_bonuses[0]
    assert bonus.type == js_bonus_event["data"]["type"] == "BonusAllColor"
    assert source_env._source_number(bonus.x) == js_bonus_event["data"]["x"] == 27.255
    assert source_env._source_number(bonus.y) == js_bonus_event["data"]["y"] == 73.745


def test_source_env_default_non_clear_bonus_probabilities_are_base_one():
    game = source_env.SourceGameState(size=101, max_score=10)
    avatars = [
        source_env.SourceAvatarState(id=1, name="p0", alive=True),
        source_env.SourceAvatarState(id=2, name="p1", alive=True),
        source_env.SourceAvatarState(id=3, name="p2", alive=False),
        source_env.SourceAvatarState(id=4, name="p3", alive=False),
    ]

    for bonus_type in (
        "BonusEnemyInverse",
        "BonusEnemyStraightAngle",
        "BonusGameBorderless",
    ):
        assert source_env._bonus_probability(bonus_type, game, avatars) == 1.0
    assert source_env._bonus_probability("BonusGameClear", game, avatars) == 0.5


def test_source_env_matches_js_oracle_for_default_bonus_game_clear_type_rng_fixture():
    scenario_path = _SCENARIO_DIR / "source_bonus_default_weights_select_game_clear_step.json"
    js_payload = _run_js_scenario_oracle(scenario_path)
    scenario = _load_scenario(scenario_path)
    source_setup = scenario["source_setup"]
    assert isinstance(source_setup, dict)
    room = source_setup["room"]
    assert isinstance(room, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)

    env = CurvyTronSourceEnv(
        random_constant=0.5,
        max_score=float(room["max_score"]),
        include_bonus_snapshot=True,
        emit_step_position_events=True,
    )
    env.reset(
        player_count=int(scenario["player_count"]),
        players=scenario["players"],
        warmup_ms=0,
    )
    for player in scenario["players"]:
        initial = player["initial"]
        avatar = env.set_avatar_state(
            player["avatar_id"],
            x=initial["x"],
            y=initial["y"],
            angle=initial["angle_rad"],
        )
        avatar.printing = bool(initial["printing"])
        avatar.alive = bool(initial.get("alive", True))

    env.set_random_sequence(random_setup["math_random_sequence"])
    env.source_game_on_start(
        bonus_types=room["bonuses"],
        bonus_rate=float(room["bonus_rate"]),
    )
    env.events.clear()

    step = scenario["steps"][0]
    env.advance_timers(float(step["advance_timers_ms"]))
    frame = env.step(
        [move["move"] for move in step["moves"]],
        elapsed_ms=float(step["step_ms"]),
    )

    js_frame = js_payload["trace"][0]
    js_bonus_event = next(event for event in js_frame["events"] if event["event"] == "bonus:pop")
    js_avatars = {avatar["name"]: avatar for avatar in js_frame["avatars"]}
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert _source_random_call_labels(env) == js_payload["randomCalls"]
    assert _source_gameplay_event_data(env) == js_frame["events"]
    assert frame["game"]["size"] == js_frame["game"]["size"]
    assert frame["game"]["worldBodyCount"] == js_frame["game"]["worldBodyCount"]
    assert frame["game"]["bonusCount"] == js_frame["game"]["bonusCount"]
    assert frame["game"]["bonusWorldBodyCount"] == js_frame["game"]["bonusWorldBodyCount"]
    assert avatars["p0"]["alive"] == js_avatars["p0"]["alive"]
    assert avatars["p1"]["alive"] == js_avatars["p1"]["alive"]
    assert avatars["p2"]["alive"] == js_avatars["p2"]["alive"]
    assert avatars["p3"]["alive"] == js_avatars["p3"]["alive"]
    assert avatars["p0"]["activeBonuses"] == js_avatars["p0"]["activeBonuses"]
    assert avatars["p1"]["activeBonuses"] == js_avatars["p1"]["activeBonuses"]
    assert len(env.active_bonuses) == 1
    bonus = env.active_bonuses[0]
    assert bonus.type == js_bonus_event["data"]["type"] == "BonusGameClear"
    assert source_env._source_number(bonus.x) == js_bonus_event["data"]["x"] == 27.255
    assert source_env._source_number(bonus.y) == js_bonus_event["data"]["y"] == 73.745


def test_source_env_matches_js_oracle_for_default_bonus_game_clear_full_probability_fixture():
    scenario_path = (
        _SCENARIO_DIR / "source_bonus_default_weights_game_clear_full_probability_step.json"
    )
    js_payload = _run_js_scenario_oracle(scenario_path)
    scenario = _load_scenario(scenario_path)
    source_setup = scenario["source_setup"]
    assert isinstance(source_setup, dict)
    room = source_setup["room"]
    assert isinstance(room, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)

    env = CurvyTronSourceEnv(
        random_constant=0.5,
        max_score=float(room["max_score"]),
        include_bonus_snapshot=True,
        emit_step_position_events=True,
    )
    env.reset(
        player_count=int(scenario["player_count"]),
        players=scenario["players"],
        warmup_ms=0,
    )
    for player in scenario["players"]:
        initial = player["initial"]
        avatar = env.set_avatar_state(
            player["avatar_id"],
            x=initial["x"],
            y=initial["y"],
            angle=initial["angle_rad"],
        )
        avatar.printing = bool(initial["printing"])
        avatar.alive = bool(initial.get("alive", True))

    env.set_random_sequence(random_setup["math_random_sequence"])
    env.source_game_on_start(
        bonus_types=room["bonuses"],
        bonus_rate=float(room["bonus_rate"]),
    )
    env.events.clear()

    step = scenario["steps"][0]
    env.advance_timers(float(step["advance_timers_ms"]))
    frame = env.step(
        [move["move"] for move in step["moves"]],
        elapsed_ms=float(step["step_ms"]),
    )

    js_frame = js_payload["trace"][0]
    js_bonus_event = next(event for event in js_frame["events"] if event["event"] == "bonus:pop")
    js_avatars = {avatar["name"]: avatar for avatar in js_frame["avatars"]}
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert _source_random_call_labels(env) == js_payload["randomCalls"]
    assert _source_gameplay_event_data(env) == js_frame["events"]
    assert frame["game"]["size"] == js_frame["game"]["size"]
    assert frame["game"]["worldBodyCount"] == js_frame["game"]["worldBodyCount"]
    assert frame["game"]["bonusCount"] == js_frame["game"]["bonusCount"]
    assert frame["game"]["bonusWorldBodyCount"] == js_frame["game"]["bonusWorldBodyCount"]
    assert avatars["p0"]["alive"] == js_avatars["p0"]["alive"]
    assert avatars["p1"]["alive"] == js_avatars["p1"]["alive"]
    assert avatars["p2"]["alive"] == js_avatars["p2"]["alive"]
    assert avatars["p3"]["alive"] == js_avatars["p3"]["alive"]
    assert avatars["p0"]["activeBonuses"] == js_avatars["p0"]["activeBonuses"]
    assert avatars["p1"]["activeBonuses"] == js_avatars["p1"]["activeBonuses"]
    assert avatars["p3"]["activeBonuses"] == js_avatars["p3"]["activeBonuses"]
    assert len(env.active_bonuses) == 1
    bonus = env.active_bonuses[0]
    assert bonus.type == js_bonus_event["data"]["type"] == "BonusGameClear"
    assert source_env._source_number(bonus.x) == js_bonus_event["data"]["x"] == 27.255
    assert source_env._source_number(bonus.y) == js_bonus_event["data"]["y"] == 73.745


def test_source_env_matches_js_oracle_for_live_movement_angle_position_event_trace():
    js_payload = _run_js_scenario_oracle(_LIVE_MOVEMENT_EVENT_TRACE_SCENARIO)
    if js_payload is None:
        pytest.skip("node executable is not available")
    env, scenario, frames = _run_source_live_movement_event_trace(
        _LIVE_MOVEMENT_EVENT_TRACE_SCENARIO
    )
    expected = scenario["comparison"]["expected"]
    assert isinstance(expected, dict)

    assert env.random_calls == []
    assert len(frames) == len(js_payload["trace"]) == len(scenario["steps"])
    previous_angles = {
        int(player["avatar_id"]): source_env._source_number(player["initial"]["angle_rad"])
        for player in scenario["players"]
    }

    for frame, js_frame in zip(frames, js_payload["trace"], strict=True):
        assert frame["events"] == js_frame["events"]
        assert _positions_from_source_snapshot(frame) == [
            [avatar["x"], avatar["y"]] for avatar in js_frame["avatars"]
        ]
        assert [avatar["angle"] for avatar in frame["avatars"]] == [
            avatar["angle"] for avatar in js_frame["avatars"]
        ]

        position_order = [
            event["data"]["avatar"] for event in frame["events"] if event["event"] == "position"
        ]
        assert position_order == [2, 1]

        changed_in_reverse_order = []
        for avatar in reversed(js_frame["avatars"]):
            avatar_id = int(avatar["id"])
            if avatar["angle"] != previous_angles[avatar_id]:
                changed_in_reverse_order.append(avatar_id)
            previous_angles[avatar_id] = avatar["angle"]
        assert [
            event["data"]["avatar"] for event in frame["events"] if event["event"] == "angle"
        ] == changed_in_reverse_order

    assert [
        [event["data"]["avatar"] for event in frame["events"] if event["event"] == "position"]
        for frame in frames
    ] == expected["position_avatar_order_per_step"]
    assert [
        [event["data"]["avatar"] for event in frame["events"] if event["event"] == "angle"]
        for frame in frames
    ] == expected["angle_avatar_order_per_step"]


def test_source_env_matches_js_oracle_for_non_present_print_manager_start(tmp_path: Path):
    scenario_path = tmp_path / "source_lifecycle_non_present_print_start_2p.json"
    scenario_path.write_text(
        json.dumps(
            {
                "scenario_id": "source_lifecycle_non_present_print_start_2p",
                "player_count": 2,
                "players": [
                    {"id": "p0", "avatar_id": 1, "name": "p0", "present": True},
                    {"id": "p1", "avatar_id": 2, "name": "p1", "present": False},
                ],
                "source_setup": {
                    "random": {"math_random_sequence": [0.5, 0.5, 0.5, 0.5, 0.5]},
                    "room": {"max_score": 10},
                },
                "lifecycle": {
                    "new_round_time_ms": 0,
                    "include_deaths_snapshot": True,
                    "actions": [
                        {"type": "advance_timers", "ms": 0},
                        {
                            "type": "set_avatar_state",
                            "avatar": 1,
                            "x": 44,
                            "y": 44,
                            "angle": 0,
                            "velocity": 8,
                        },
                        {"type": "advance_timers", "ms": 3000},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    js_payload = _run_js_lifecycle_oracle(scenario_path)
    env, scenario = _env_from_scenario_with_options(
        scenario_path,
        include_deaths_snapshot=True,
    )
    lifecycle = scenario["lifecycle"]
    assert isinstance(lifecycle, dict)
    actions = lifecycle["actions"]
    assert isinstance(actions, list)

    snapshots = [
        env.reset(
            player_count=int(scenario["player_count"]),
            players=scenario["players"],
            warmup_ms=0,
        )
    ]
    env.advance_timers(0)
    snapshots.append(
        env.snapshot(
            "after_action_0_advance_timers",
            action=actions[0],
        )
    )
    env.set_avatar_state(1, x=44, y=44, angle=0, velocity=8)
    snapshots.append(
        env.snapshot(
            "after_action_1_set_avatar_state",
            action=actions[1],
        )
    )
    env.advance_timers(3000)
    snapshots.append(
        env.snapshot(
            "after_action_2_advance_timers",
            action=actions[2],
        )
    )

    assert env.events == js_payload["events"]
    assert env.random_calls == js_payload["randomCalls"]
    assert snapshots == js_payload["snapshots"]


def test_source_1p_wall_death_scores_dead_sole_avatar_as_round_winner():
    env = CurvyTronSourceEnv(include_deaths_snapshot=True)
    env.reset(player_count=1, warmup_ms=0)
    env.advance_timers(0)
    env.events.clear()
    env.random.calls.clear()

    env.set_avatar_state(1, x=0.3, y=20, angle=0)
    final = env.step({}, elapsed_ms=0)

    assert final["game"]["inRound"] is False
    assert final["game"]["deaths"] == [1]
    assert _alive_from_source_snapshot(final) == [False]
    assert _scores_from_source_snapshot(final) == [1]
    # Source Game.resolveScores special-cases avatars.count() === 1, so the
    # sole avatar is the winner even after the death that ended the round.
    assert _event_names_data(env) == [
        ("die", {"avatar": 1, "killer": None, "old": None}),
        ("score:round", {"avatar": 1, "score": 0, "roundScore": 0}),
        ("score:round", {"avatar": 1, "score": 0, "roundScore": 1}),
        ("score", {"avatar": 1, "score": 1, "roundScore": 1}),
        ("round:end", {"winner": 1}),
    ]


def test_source_env_matches_js_oracle_for_warmup_game_start_and_print_manager_start():
    js_payload = _run_js_lifecycle_oracle(_WARMUP_SCENARIO)
    env, scenario = _env_from_scenario(_WARMUP_SCENARIO)

    snapshots = [
        env.reset(
            player_count=int(scenario["player_count"]),
            players=scenario["players"],
            warmup_ms=0,
        )
    ]
    env.advance_timers(0)
    snapshots.append(env.snapshot("after_advance_0", advance_ms=0))
    env.advance_timers(3000)
    snapshots.append(env.snapshot("after_advance_1", advance_ms=3000))

    assert env.events == js_payload["events"]
    assert env.random_calls == js_payload["randomCalls"]
    assert snapshots == js_payload["snapshots"]


def test_source_env_is_player_count_generic_for_reset_spawn_order():
    env = CurvyTronSourceEnv(
        random_values=[
            0.5,
            0.5,
            0.2,
            0.55,
            0.5,
            0.3,
            0.45,
            0.5,
            0.4,
        ]
    )

    env.reset(player_count=3, warmup_ms=1000)

    assert len(env.avatars) == 3
    assert env.game is not None
    assert env.game.size == env.reference.arena_size_for_players(3)
    assert [
        event["data"]["avatar"]
        for event in env.events
        if event["event"] == "random" and event["data"]["site"] == "spawn.position_x"
    ] == [3, 2, 1]


@pytest.mark.parametrize("scenario_path", [_SPAWN_3P_SCENARIO, _SPAWN_4P_SCENARIO])
def test_source_env_matches_js_oracle_for_multiplayer_reset_spawn_order(
    scenario_path: Path,
):
    js_payload = _run_js_lifecycle_oracle(scenario_path)
    env, scenario = _env_from_scenario(scenario_path)

    snapshot = env.reset(
        player_count=int(scenario["player_count"]),
        players=scenario["players"],
        warmup_ms=0,
    )

    assert env.events == js_payload["events"]
    assert env.random_calls == js_payload["randomCalls"]
    assert [snapshot] == js_payload["snapshots"]


def test_source_env_matches_js_oracle_for_simple_wall_death_round_end_and_match_end():
    js_payload = _run_js_lifecycle_oracle(_MATCH_END_SCENARIO)
    env, scenario = _env_from_scenario(_MATCH_END_SCENARIO)

    env.reset(
        player_count=int(scenario["player_count"]),
        players=scenario["players"],
        warmup_ms=0,
    )
    env.advance_timers(0)

    avatar_1 = env.avatar_by_id(1)
    avatar_1.x = 5
    avatar_1.y = 5
    avatar_1.set_angle(0.7853981633974483)
    avatar_1.set_velocity(8, env.reference)

    env.advance_timers(3000)

    avatar_2 = env.avatar_by_id(2)
    avatar_2.x = 87
    avatar_2.y = 44
    avatar_2.set_angle(0)

    env.step({}, elapsed_ms=100)
    env.advance_timers(5000)

    assert env.events == js_payload["events"]
    assert env.random_calls == js_payload["randomCalls"]
    assert env.snapshot("final")["game"]["worldActive"] is None
    assert env.snapshot("final")["avatars"] == []


def test_advance_timers_runs_new_round_start_scheduled_by_game_stop():
    env = CurvyTronSourceEnv()
    env.reset(player_count=2, warmup_ms=0)
    env.advance_timers(0)
    env.events.clear()
    env.random.calls.clear()

    env.set_avatar_state(1, x=0.3, y=20)
    env.set_avatar_state(2, x=60, y=60)
    env.step({}, elapsed_ms=0)

    env.advance_timers(env.reference.round_warmdown_ms + env.reference.round_warmup_ms + 1)

    lifecycle_events = [
        (event["event"], event["atMs"])
        for event in env.events
        if event["event"] in {"round:end", "game:stop", "round:new", "game:start"}
    ]
    assert lifecycle_events == [
        ("round:end", 0),
        ("game:stop", 5000),
        ("round:new", 5000),
        ("game:start", 8000),
    ]
    game = env.game
    assert game is not None
    assert game.started is True
    assert game.in_round is True
    assert game.world_active is True
    assert game.rendered == 8000
    assert env.now_ms == 8001


def test_tie_at_max_score_starts_next_round():
    env = CurvyTronSourceEnv(max_score=2)
    env.reset(player_count=2, warmup_ms=0)
    env.advance_timers(0)
    env.avatar_by_id(1).score = 2
    env.avatar_by_id(2).score = 2
    env.events.clear()
    env.random.calls.clear()

    env.set_avatar_state(1, x=0.3, y=20)
    env.set_avatar_state(2, x=0.3, y=40)
    env.step({}, elapsed_ms=0)
    env.advance_timers(env.reference.round_warmdown_ms)

    lifecycle_events = [
        (event["event"], event["atMs"])
        for event in env.events
        if event["event"] in {"round:end", "game:stop", "round:new", "end"}
    ]
    assert lifecycle_events == [
        ("round:end", 0),
        ("game:stop", 5000),
        ("round:new", 5000),
    ]
    assert [avatar.score for avatar in env.avatars] == [2, 2]
    game = env.game
    assert game is not None
    assert game.started is True
    assert game.in_round is True
    assert game.game_start_due_ms == 8000


def test_advance_timers_stops_zero_delay_reschedule_loop(monkeypatch):
    monkeypatch.setattr(source_env, "_MAX_TIMER_CALLBACKS_PER_ADVANCE", 3)
    env = CurvyTronSourceEnv()
    env.reset(player_count=2, warmup_ms=0)

    def reschedule_start() -> None:
        game = env.game
        assert game is not None
        game.game_start_due_ms = env.now_ms

    monkeypatch.setattr(env, "_start_game", reschedule_start)

    with pytest.raises(SourceEnvError, match="timer advance exceeded 3 callbacks"):
        env.advance_timers(0)


def test_source_3p_ordered_deaths_resolve_round_scores():
    env = CurvyTronSourceEnv(include_deaths_snapshot=True)
    env.reset(player_count=3, warmup_ms=0)
    env.advance_timers(0)
    env.events.clear()
    env.random.calls.clear()

    env.set_avatar_state(1, x=0.3, y=20)
    env.set_avatar_state(2, x=40, y=40)
    env.set_avatar_state(3, x=60, y=60)
    first = env.step({}, elapsed_ms=0)

    assert first["game"]["inRound"] is True
    assert first["game"]["deaths"] == [1]
    assert _event_names_data(env) == [
        ("die", {"avatar": 1, "killer": None, "old": None}),
        ("score:round", {"avatar": 1, "score": 0, "roundScore": 0}),
    ]

    env.set_avatar_state(2, x=0.3, y=40)
    final = env.step({}, elapsed_ms=0)

    assert final["game"]["inRound"] is False
    assert final["game"]["deaths"] == [1, 2]
    assert [avatar.score for avatar in env.avatars] == [0, 1, 2]
    assert _event_names_data(env) == [
        ("die", {"avatar": 1, "killer": None, "old": None}),
        ("score:round", {"avatar": 1, "score": 0, "roundScore": 0}),
        ("die", {"avatar": 2, "killer": None, "old": None}),
        ("score:round", {"avatar": 2, "score": 0, "roundScore": 1}),
        ("score:round", {"avatar": 3, "score": 0, "roundScore": 2}),
        ("score", {"avatar": 3, "score": 2, "roundScore": 2}),
        ("score", {"avatar": 2, "score": 1, "roundScore": 1}),
        ("score", {"avatar": 1, "score": 0, "roundScore": 0}),
        ("round:end", {"winner": 3}),
    ]


def test_source_3p_same_frame_deaths_use_frame_start_death_count():
    env = CurvyTronSourceEnv(include_deaths_snapshot=True)
    env.reset(player_count=3, warmup_ms=0)
    env.advance_timers(0)
    env.events.clear()
    env.random.calls.clear()

    env.set_avatar_state(1, x=0.3, y=20)
    env.set_avatar_state(2, x=0.3, y=40)
    env.set_avatar_state(3, x=60, y=60)
    final = env.step({}, elapsed_ms=0)

    assert final["game"]["deaths"] == [2, 1]
    assert [avatar.score for avatar in env.avatars] == [0, 0, 2]
    assert _event_names_data(env) == [
        ("die", {"avatar": 2, "killer": None, "old": None}),
        ("score:round", {"avatar": 2, "score": 0, "roundScore": 0}),
        ("die", {"avatar": 1, "killer": None, "old": None}),
        ("score:round", {"avatar": 1, "score": 0, "roundScore": 0}),
        ("score:round", {"avatar": 3, "score": 0, "roundScore": 2}),
        ("score", {"avatar": 3, "score": 2, "roundScore": 2}),
        ("score", {"avatar": 2, "score": 0, "roundScore": 0}),
        ("score", {"avatar": 1, "score": 0, "roundScore": 0}),
        ("round:end", {"winner": 3}),
    ]


def test_source_3p_absent_avatar_counts_as_existing_death_for_scoring():
    env = CurvyTronSourceEnv(include_deaths_snapshot=True)
    env.reset(player_count=3, present=[True, False, True], warmup_ms=0)
    env.advance_timers(0)
    env.events.clear()
    env.random.calls.clear()

    env.set_avatar_state(1, x=0.3, y=20)
    env.set_avatar_state(3, x=60, y=60)
    final = env.step({}, elapsed_ms=0)

    assert final["game"]["deaths"] == [2, 1]
    assert [avatar.score for avatar in env.avatars] == [1, 0, 2]
    assert _event_names_data(env) == [
        ("die", {"avatar": 1, "killer": None, "old": None}),
        ("score:round", {"avatar": 1, "score": 0, "roundScore": 1}),
        ("score:round", {"avatar": 3, "score": 0, "roundScore": 2}),
        ("score", {"avatar": 3, "score": 2, "roundScore": 2}),
        ("score", {"avatar": 2, "score": 0, "roundScore": 0}),
        ("score", {"avatar": 1, "score": 1, "roundScore": 1}),
        ("round:end", {"winner": 3}),
    ]


def test_source_env_matches_js_oracle_for_mid_round_remove_avatar_leave():
    js_payload = _run_js_lifecycle_oracle(_MID_ROUND_REMOVE_AVATAR_SCENARIO)
    env, scenario = _env_from_scenario_with_options(
        _MID_ROUND_REMOVE_AVATAR_SCENARIO,
        include_deaths_snapshot=True,
    )
    lifecycle = scenario["lifecycle"]
    assert isinstance(lifecycle, dict)
    actions = lifecycle["actions"]
    assert isinstance(actions, list)

    snapshots = [
        env.reset(
            player_count=int(scenario["player_count"]),
            players=scenario["players"],
            warmup_ms=float(lifecycle["new_round_time_ms"]),
        )
    ]
    for index, action in enumerate(actions):
        assert isinstance(action, dict)
        action_type = action["type"]
        if action_type == "advance_timers":
            env.advance_timers(float(action["ms"]))
        elif action_type == "remove_avatar":
            env.remove_avatar(action["avatar"])
        else:
            raise AssertionError(f"unsupported test lifecycle action {action_type!r}")
        snapshots.append(
            env.snapshot(
                f"after_action_{index}_{action_type}",
                action=action,
            )
        )

    assert js_payload["expectations"]["eventOrder"]["status"] == "pass"
    assert env.events == js_payload["events"]
    assert env.random_calls == js_payload["randomCalls"]
    assert snapshots == js_payload["snapshots"]

    after_leave = snapshots[-1]
    assert after_leave["game"]["inRound"] is False
    assert after_leave["game"]["deaths"] == []
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in after_leave["avatars"]
    ] == [(1, True, True, 1), (2, False, False, 0)]


def test_source_env_matches_js_oracle_for_3p_mid_round_remove_avatar_continuation():
    js_payload = _run_js_lifecycle_oracle(_MID_ROUND_REMOVE_AVATAR_3P_SCENARIO)
    env, _scenario, snapshots = _run_source_lifecycle_actions(_MID_ROUND_REMOVE_AVATAR_3P_SCENARIO)

    assert js_payload["expectations"]["eventOrder"]["status"] == "pass"
    assert env.events == js_payload["events"]
    assert env.random_calls == js_payload["randomCalls"]
    assert snapshots == js_payload["snapshots"]

    after_leave = snapshots[3]
    assert after_leave["label"] == "after_action_2_remove_avatar"
    assert after_leave["game"]["inRound"] is True
    assert after_leave["game"]["deaths"] == []
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in after_leave["avatars"]
    ] == [
        (1, True, True, 0),
        (2, False, False, 0),
        (3, True, True, 0),
    ]

    after_update = snapshots[-1]
    assert after_update["label"] == "after_action_4_update"
    assert after_update["game"]["inRound"] is False
    assert after_update["game"]["deaths"] == [3]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in after_update["avatars"]
    ] == [
        (1, True, True, 2),
        (2, False, False, 0),
        (3, False, True, 0),
    ]


def test_source_env_matches_js_oracle_for_4p_mid_round_remove_avatar_continuation():
    js_payload = _run_js_lifecycle_oracle(_MID_ROUND_REMOVE_AVATAR_4P_SCENARIO)
    env, _scenario, snapshots = _run_source_lifecycle_actions(_MID_ROUND_REMOVE_AVATAR_4P_SCENARIO)

    assert js_payload["expectations"]["eventOrder"]["status"] == "pass"
    assert env.events == js_payload["events"]
    assert env.random_calls == js_payload["randomCalls"]
    assert snapshots == js_payload["snapshots"]

    after_leave = snapshots[4]
    assert after_leave["label"] == "after_action_3_remove_avatar"
    assert after_leave["game"]["inRound"] is True
    assert after_leave["game"]["deaths"] == []
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in after_leave["avatars"]
    ] == [
        (1, True, True, 0),
        (2, False, False, 0),
        (3, True, True, 0),
        (4, True, True, 0),
    ]

    after_first_death = snapshots[6]
    assert after_first_death["label"] == "after_action_5_update"
    assert after_first_death["game"]["inRound"] is True
    assert after_first_death["game"]["deaths"] == [4]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in after_first_death["avatars"]
    ] == [
        (1, True, True, 0),
        (2, False, False, 0),
        (3, True, True, 0),
        (4, False, True, 0),
    ]

    after_round_end = snapshots[-1]
    assert after_round_end["label"] == "after_action_7_update"
    assert after_round_end["game"]["inRound"] is False
    assert after_round_end["game"]["deaths"] == [4, 3]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in after_round_end["avatars"]
    ] == [
        (1, True, True, 3),
        (2, False, False, 0),
        (3, False, True, 1),
        (4, False, True, 0),
    ]


def test_source_env_runs_long_natural_1v1_wall_rollout_to_round_done():
    env, scenario, frames = _run_source_long_rollout(_LONG_NATURAL_ROLLOUT_SCENARIO)
    rollout = scenario["rollout"]
    assert isinstance(rollout, dict)
    expected = rollout["expected"]
    assert isinstance(expected, dict)

    assert len(frames) == rollout["step_count"]
    first = frames[0]
    assert first["game"]["inRound"] is True
    assert first["game"]["worldActive"] is True
    assert _positions_from_source_snapshot(first) == [[57.733333, 44], [29.733333, 44]]

    penultimate = frames[-2]
    assert rollout["terminal_tick"] == len(frames) - 1
    assert penultimate["game"]["inRound"] is True
    assert _positions_from_source_snapshot(penultimate) == [[28.666667, 44], [0.666667, 44]]
    assert _alive_from_source_snapshot(penultimate) == [True, True]
    assert _scores_from_source_snapshot(penultimate) == [0, 0]

    final = frames[-1]
    final_game = final["game"]
    assert final_game["inRound"] is not expected["round_done"]
    assert final_game["started"] is not expected["game_done"]
    assert final_game["deaths"] == expected["deaths"]
    assert _round_winner_from_source_events(env.events) == expected["round_winner"]
    assert _positions_from_source_snapshot(final) == expected["final_positions"]
    assert _alive_from_source_snapshot(final) == expected["alive"]
    assert _scores_from_source_snapshot(final) == expected["scores"]
    assert env.random_calls == [
        {
            "index": 0,
            "value": 0.32051282051282054,
            "atMs": 0,
            "label": {"site": "spawn.position_x", "avatar": 2},
        },
        {
            "index": 1,
            "value": 0.5,
            "atMs": 0,
            "label": {"site": "spawn.position_y", "avatar": 2},
        },
        {
            "index": 2,
            "value": 0.5,
            "atMs": 0,
            "label": {"site": "spawn.angle_attempt_0", "avatar": 2},
        },
        {
            "index": 3,
            "value": 0.6794871794871795,
            "atMs": 0,
            "label": {"site": "spawn.position_x", "avatar": 1},
        },
        {
            "index": 4,
            "value": 0.5,
            "atMs": 0,
            "label": {"site": "spawn.position_y", "avatar": 1},
        },
        {
            "index": 5,
            "value": 0.5,
            "atMs": 0,
            "label": {"site": "spawn.angle_attempt_0", "avatar": 1},
        },
    ]

    if shutil.which("node") is None:
        return

    with CurvytronJsEnvWorker() as worker:
        worker.reset(_LONG_NATURAL_ROLLOUT_SCENARIO)
        js_final = None
        for tick in range(int(rollout["step_count"])):
            js_final = worker.step(
                {
                    "tick": tick,
                    "step_ms": rollout["step_ms"],
                    "advance_timers_ms": rollout["advance_timers_ms"],
                    "moves": rollout["moves"],
                }
            )

    assert js_final is not None
    js_state = js_final["state"]
    assert js_final["roundDone"] is expected["round_done"]
    assert js_final["gameDone"] is expected["game_done"]
    assert js_final["reward"] == expected["reward"]
    assert js_state["game"]["deaths"] == final_game["deaths"]
    assert js_state["game"]["roundWinner"] == expected["round_winner"]
    assert [[player["x"], player["y"]] for player in js_state["players"]] == (
        _positions_from_source_snapshot(final)
    )
    assert [player["alive"] for player in js_state["players"]] == _alive_from_source_snapshot(final)
    assert [player["score"] for player in js_state["players"]] == _scores_from_source_snapshot(
        final
    )


def test_source_world_island_corner_lookup_matches_js_source():
    js_probe = _run_js_world_island_probe()
    world = SourceWorldState(size=88)
    world.activate()

    boundary = SourceBodyState(x=44, y=40, radius=0.6, avatar_id=2, num=0)
    world.add_body(boundary)
    edge = SourceBodyState(x=0.2, y=87.8, radius=0.6, avatar_id=2, num=1)
    world.add_body(edge)
    corner_only = SourceBodyState(x=43, y=20, radius=0.6, avatar_id=2, num=2)
    world.add_body(corner_only)

    def island_body_ids(island_id: str) -> list[int | None]:
        return [body.id for body in world.islands[island_id].bodies]

    def lookup_id(body: SourceBodyState) -> int | None:
        hit = world.get_body(body)
        return hit.id if hit is not None else None

    python_probe = {
        "islandSize": world.island_size,
        "bodyCount": world.body_count,
        "boundaryIslandIds": sorted(boundary.island_ids),
        "edgeIslandIds": sorted(edge.island_ids),
        "cornerOnlyIslandIds": sorted(corner_only.island_ids),
        "islandBodies": {
            "0:0": island_body_ids("0:0"),
            "1:0": island_body_ids("1:0"),
            "0:1": island_body_ids("0:1"),
            "1:1": island_body_ids("1:1"),
        },
        "lookupHitIds": {
            "boundary": lookup_id(SourceBodyState(x=44.2, y=40, radius=0.6, avatar_id=1, num=0)),
            "edge": lookup_id(SourceBodyState(x=0.2, y=87.8, radius=0.6, avatar_id=1, num=1)),
            "cornerOnlyFromAdjacentCenter": lookup_id(
                SourceBodyState(x=44, y=20, radius=0.6, avatar_id=1, num=2)
            ),
            "outside": lookup_id(SourceBodyState(x=100, y=100, radius=0.6, avatar_id=1, num=3)),
        },
    }
    expected = {
        "islandSize": 44,
        "bodyCount": 3,
        "boundaryIslandIds": ["0:0", "1:0"],
        "edgeIslandIds": ["0:1"],
        "cornerOnlyIslandIds": ["0:0"],
        "islandBodies": {
            "0:0": [0, 2],
            "1:0": [0],
            "0:1": [1],
            "1:1": [],
        },
        "lookupHitIds": {
            "boundary": 0,
            "edge": 1,
            "cornerOnlyFromAdjacentCenter": 2,
            "outside": None,
        },
    }

    assert js_probe == expected
    assert python_probe == js_probe


def test_source_body_opponent_tangent_is_safe_but_strict_overlap_kills():
    tangent_env = _active_source_env()
    _seed_source_body(
        tangent_env,
        owner_id=2,
        x=21.200000000000003,
        y=20,
    )

    tangent_env.step({}, elapsed_ms=0)

    assert tangent_env.avatar_by_id(1).alive is True
    assert _die_events(tangent_env) == []

    overlap_env = _active_source_env()
    _seed_source_body(overlap_env, owner_id=2, x=21.199999999999, y=20)

    overlap_env.step({}, elapsed_ms=0)

    assert overlap_env.avatar_by_id(1).alive is False
    assert _die_events(overlap_env)[0]["data"] == {"avatar": 1, "killer": 2, "old": False}


def test_source_body_collision_ignores_visual_line_between_stored_body_points():
    env = _active_source_env()
    _seed_source_body(env, owner_id=2, x=10, y=20)
    _seed_source_body(env, owner_id=2, x=30, y=20)

    frame = env.step({}, elapsed_ms=0)

    assert env.avatar_by_id(1).alive is True
    assert frame["game"]["worldBodyCount"] == 2
    assert _die_events(env) == []


def test_source_body_own_trail_latency_delta3_safe_delta4_kills():
    safe_env = _active_source_env()
    avatar = safe_env.avatar_by_id(1)
    _seed_source_body(safe_env, owner_id=1, x=20, y=20, num=0)
    avatar.body_count = 3

    safe_env.step({}, elapsed_ms=0)

    assert avatar.alive is True
    assert _die_events(safe_env) == []

    kill_env = _active_source_env()
    avatar = kill_env.avatar_by_id(1)
    _seed_source_body(kill_env, owner_id=1, x=20, y=20, num=0)
    avatar.body_count = 4

    kill_env.step({}, elapsed_ms=0)

    assert avatar.alive is False
    assert _die_events(kill_env)[0]["data"] == {"avatar": 1, "killer": 1, "old": False}


def test_source_body_old_flag_reports_true_at_2000ms_without_blocking_collision():
    env = _active_source_env()
    _seed_source_body(env, owner_id=2, x=20, y=20, birth_ms=0)
    env.advance_timers(2000)

    env.step({}, elapsed_ms=0)

    assert env.avatar_by_id(1).alive is False
    assert _die_events(env)[0]["data"] == {"avatar": 1, "killer": 2, "old": True}


def test_source_body_wall_death_takes_priority_over_body_collision():
    env = _active_source_env()
    env.set_avatar_state(1, x=0.3, y=20)
    _seed_source_body(env, owner_id=2, x=0.3, y=20)

    env.step({}, elapsed_ms=0)

    assert env.avatar_by_id(1).alive is False
    assert _die_events(env)[0]["data"] == {"avatar": 1, "killer": None, "old": None}


def test_source_print_manager_stop_emits_property_even_when_avatar_is_already_hole():
    env = _active_source_env()
    avatar = env.set_avatar_state(1, x=0.3, y=20)
    avatar.print_manager.active = True
    avatar.print_manager.distance = 1
    avatar.print_manager.last_x = avatar.x
    avatar.print_manager.last_y = avatar.y
    avatar.printing = False
    env.events.clear()
    env.random.calls.clear()

    env.step({}, elapsed_ms=0)

    step_events = [(event["event"], event["data"]) for event in env.events]
    assert step_events[:4] == [
        ("property", {"avatar": 1, "property": "printing", "value": False}),
        (
            "random",
            {
                "index": 0,
                "value": 0.5,
                "site": "print_manager.stop_distance",
                "avatar": 1,
            },
        ),
        ("die", {"avatar": 1, "killer": None, "old": None}),
        ("score:round", {"avatar": 1, "score": 0, "roundScore": 0}),
    ]


def test_source_trail_exact_radius_from_last_point_does_not_emit_body():
    env = _active_source_env()
    avatar = env.avatar_by_id(1)
    avatar.printing = True
    avatar.trail_point_count = 1
    avatar.trail_last_x = 0
    avatar.trail_last_y = 20
    env.set_avatar_state(1, x=avatar.radius, y=20)
    env.events.clear()

    frame = env.step({}, elapsed_ms=0)

    assert avatar.alive is True
    assert avatar.trail_point_count == 1
    assert avatar.trail_last_x == 0
    assert avatar.trail_last_y == 20
    assert frame["game"]["worldBodyCount"] == 0
    assert _die_events(env) == []


def test_source_trail_radius_plus_epsilon_from_last_point_emits_body():
    env = _active_source_env()
    avatar = env.avatar_by_id(1)
    avatar.printing = True
    avatar.trail_point_count = 1
    avatar.trail_last_x = 0
    avatar.trail_last_y = 20
    env.set_avatar_state(1, x=avatar.radius + 1e-12, y=20)
    env.events.clear()

    frame = env.step({}, elapsed_ms=0)

    assert avatar.alive is True
    assert avatar.trail_point_count == 2
    assert avatar.trail_last_x == avatar.radius + 1e-12
    assert avatar.trail_last_y == 20
    assert frame["game"]["worldBodyCount"] == 1
    assert _die_events(env) == []


def test_source_borderless_wrap_skips_body_until_next_frame():
    env = _active_source_env(player_count=3, borderless=True)
    env.set_avatar_state(1, x=94.8, y=44, angle=0)
    env.set_avatar_state(2, x=20, y=20, angle=0)
    env.set_avatar_state(3, x=80, y=20, angle=math.pi)
    _seed_source_body(env, owner_id=2, x=0, y=44)
    env.events.clear()

    first = env.step({}, elapsed_ms=20)

    assert env.avatar_by_id(1).alive is True
    assert _positions_from_source_snapshot(first)[0] == [0, 44]
    assert first["game"]["worldBodyCount"] == 1
    assert _die_events(env) == []

    second = env.step({}, elapsed_ms=0)

    assert env.avatar_by_id(1).alive is False
    assert _positions_from_source_snapshot(second)[0] == [0, 44]
    assert second["game"]["worldBodyCount"] == 2
    assert _die_events(env)[0]["data"] == {"avatar": 1, "killer": 2, "old": False}


def test_source_borderless_exact_edge_is_safe_and_corner_wraps_one_axis():
    env = _active_source_env(borderless=True)
    env.set_avatar_state(1, x=87.35, y=87.35, angle=math.pi / 4)
    env.set_avatar_state(2, x=86.4, y=20, angle=0)
    env.events.clear()

    frame = env.step({}, elapsed_ms=100)

    assert _positions_from_source_snapshot(frame) == [[0, 88.481371], [88, 20]]
    assert _alive_from_source_snapshot(frame) == [True, True]
    assert frame["game"]["worldBodyCount"] == 0
    assert _die_events(env) == []


def test_source_borderless_wrap_runs_print_manager_after_wrap():
    env = _active_source_env(borderless=True)
    avatar = env.set_avatar_state(1, x=87.8, y=44, angle=0)
    env.set_avatar_state(2, x=20, y=20, angle=math.pi)
    avatar.printing = True
    avatar.trail_last_x = 87.4
    avatar.trail_last_y = 44
    avatar.print_manager.active = True
    avatar.print_manager.distance = 1
    avatar.print_manager.last_x = avatar.x
    avatar.print_manager.last_y = avatar.y
    env.events.clear()
    env.random.calls.clear()

    frame = env.step({}, elapsed_ms=20)

    assert _positions_from_source_snapshot(frame)[0] == [0, 44]
    assert avatar.alive is True
    assert avatar.printing is False
    assert avatar.trail_point_count == 0
    assert avatar.trail_last_x is None
    assert avatar.trail_last_y is None
    assert avatar.print_manager.active is True
    assert avatar.print_manager.to_snapshot() == {
        "active": True,
        "distance": 5.25,
        "lastX": 0,
        "lastY": 44,
    }
    assert frame["game"]["worldBodyCount"] == 2
    assert env.random_calls == [
        {
            "index": 0,
            "value": 0.5,
            "atMs": 0,
            "label": {"site": "print_manager.toggle_distance", "avatar": 1},
        }
    ]


def test_source_body_same_frame_reverse_order_insertion_can_kill_lower_avatar():
    env = _active_source_env()
    env.set_avatar_state(1, x=20, y=20)
    avatar_2 = env.set_avatar_state(2, x=20, y=20)
    avatar_2.printing = True
    avatar_2.trail_last_x = None
    avatar_2.trail_last_y = None

    env.step({}, elapsed_ms=0)

    assert avatar_2.alive is True
    assert env.avatar_by_id(1).alive is False
    assert _die_events(env)[0]["data"] == {"avatar": 1, "killer": 2, "old": False}
