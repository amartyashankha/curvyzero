import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from curvyzero.env.scenarios import (
    ScenarioError,
    load_scenario,
    parse_scenario,
    run_source_body_canary_scenario,
    run_source_border_rules_scenario,
    run_source_borderless_wrap_scenario,
    run_scenario,
    run_source_kinematics_scenario,
    run_source_normal_wall_scenario,
    run_source_print_manager_scenario,
    run_source_trail_cadence_scenario,
    run_source_trail_gap_scenario,
)
from curvyzero.env.trace_compare import project_common_trace

_SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "environment"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DIFF_PATH = Path(__file__).resolve().parents[1] / "tools" / "fidelity_diff.py"
_DIFF_SPEC = importlib.util.spec_from_file_location("fidelity_diff", _DIFF_PATH)
assert _DIFF_SPEC is not None
assert _DIFF_SPEC.loader is not None
fidelity_diff = importlib.util.module_from_spec(_DIFF_SPEC)
sys.modules[_DIFF_SPEC.name] = fidelity_diff
_DIFF_SPEC.loader.exec_module(fidelity_diff)
diff_payload = fidelity_diff.diff_payload
first_mismatch = fidelity_diff.first_mismatch


def _run_js_scenario(filename: str | Path) -> dict[str, object]:
    if shutil.which("node") is None:
        pytest.skip("node executable is not available")

    scenario_path = filename if isinstance(filename, Path) else _SCENARIO_DIR / filename
    result = subprocess.run(
        [
            "node",
            str(_REPO_ROOT / "tools" / "reference_oracle" / "scenario_runner.js"),
            str(scenario_path),
        ],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _run_js_scenario_with_game_bonus_stack(filename: str | Path) -> dict[str, object]:
    if shutil.which("node") is None:
        pytest.skip("node executable is not available")

    scenario_path = filename if isinstance(filename, Path) else _SCENARIO_DIR / filename
    probe = r"""
const fs = require('fs');
const path = require('path');

const repoRoot = process.cwd();
const runnerPath = path.join(repoRoot, 'tools', 'reference_oracle', 'scenario_runner.js');
const scenarioPath = process.argv[1];

function replaceOnce(source, needle, replacement) {
  if (!source.includes(needle)) {
    throw new Error('scenario runner probe patch failed for: ' + needle);
  }
  return source.replace(needle, replacement);
}

let runner = fs.readFileSync(runnerPath, 'utf8');
runner = runner.replace(/^#!.*\n/, '');
runner = replaceOnce(
  runner,
  "      events.push({ event: alias || name, data: eventData(alias || name, data || {}) });",
  "      var eventPayload = typeof data === 'undefined' ? {} : data;\n      events.push({ event: alias || name, data: eventData(alias || name, eventPayload) });"
);
runner = replaceOnce(
  runner,
  "      worldBodyCount: game.world.bodyCount,\n      bonusCount: game.bonusManager.bonuses.count(),",
  "      worldBodyCount: game.world.bodyCount,\n      activeBonuses: game.bonusStack.bonuses.items.map(function (bonus) {\n        return bonusData(bonus);\n      }),\n      bonusCount: game.bonusManager.bonuses.count(),"
);
runner = replaceOnce(
  runner,
  "      case 'bonus:stack':\n        return {\n          avatar: data.avatar.id,\n          method: data.method,\n          bonus: bonusData(data.bonus)\n        };",
  "      case 'bonus:stack':\n        if (data.target === 'game') {\n          return {\n            target: 'game',\n            method: data.method,\n            bonus: bonusData(data.bonus)\n          };\n        }\n        return {\n          avatar: data.avatar.id,\n          method: data.method,\n          bonus: bonusData(data.bonus)\n        };"
);
runner = replaceOnce(
  runner,
  "  avatars.forEach(function (avatar) {\n",
  "  record(game.bonusStack, 'change', 'bonus:stack');\n\n  avatars.forEach(function (avatar) {\n"
);
runner = replaceOnce(
  runner,
  "  loadOriginalSources(context);\n\n  const result = runScenario(context, scenario);",
  "  loadOriginalSources(context);\n\n  vm.runInContext(`\n(function () {\n  var originalAdd = GameBonusStack.prototype.add;\n  var originalRemove = GameBonusStack.prototype.remove;\n\n  GameBonusStack.prototype.add = function (bonus) {\n    originalAdd.call(this, bonus);\n    this.emit('change', { target: 'game', method: 'add', bonus: bonus });\n  };\n\n  GameBonusStack.prototype.remove = function (bonus) {\n    originalRemove.call(this, bonus);\n    this.emit('change', { target: 'game', method: 'remove', bonus: bonus });\n  };\n}())\n`, context, { filename: 'game_bonus_stack_probe.vm.js' });\n\n  const result = runScenario(context, scenario);"
);

const previousArgv = process.argv;
process.argv = ['node', runnerPath, scenarioPath];
try {
  new Function('require', 'process', '__dirname', '__filename', runner)(
    require,
    process,
    path.dirname(runnerPath),
    runnerPath
  );
} finally {
  process.argv = previousArgv;
}
"""
    result = subprocess.run(
        ["node", "-e", probe, str(scenario_path)],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_loads_and_runs_first_forced_two_player_scenario(tmp_path):
    scenario_path = tmp_path / "forced_two_player_turn_step.json"
    scenario_path.write_text(
        json.dumps(
            {
                "scenario_id": "forced_two_player_turn_step",
                "ruleset_id": "curvytron-v1-reference",
                "player_count": 2,
                "seed": 123,
                "initial_state": {
                    "map_size": 88,
                    "positions": [[20, 40], [60, 40]],
                    "headings": [0, math.pi],
                },
                "action_script": [{"moves": {"1": -1, "2": 1}}],
                "action_encoding": "source-move",
                "time_policy": {"kind": "fixed", "step_ms": 1000 / 60},
                "trace_schema_version": 1,
                "tolerances": {},
                "provenance": "source-inspired",
            }
        ),
        encoding="utf-8",
    )

    scenario = load_scenario(scenario_path)
    run = run_scenario(scenario)
    payload = run.to_payload()

    assert payload["toy_v0_behavior"] is True
    assert payload["source_fidelity"] is False
    assert payload["scenario_id"] == "forced_two_player_turn_step"
    assert scenario.toy_action_script == ({"player_0": 0, "player_1": 2},)
    assert run.trace.frames[0].positions == ((20.0, 40.0), (60.0, 40.0))
    assert run.trace.frames[1].tick == 1


@pytest.mark.parametrize(
    ("filename", "expected_headings", "expected_positions"),
    [
        (
            "source_kinematics_straight_step.json",
            [0.0, 3.141593],
            [[20.266667, 40.0], [59.733333, 40.0]],
        ),
        (
            "source_kinematics_left_turn_step.json",
            [-0.046667, 3.141593],
            [[20.266376, 39.98756], [59.733333, 40.0]],
        ),
        (
            "source_kinematics_right_turn_step.json",
            [0.046667, 3.141593],
            [[20.266376, 40.01244], [59.733333, 40.0]],
        ),
        (
            "forced_two_player_turn_step.json",
            [-0.046667, 3.188259],
            [[20.266376, 39.98756], [59.733624, 39.98756]],
        ),
    ],
)
def test_source_kinematics_scenarios_match_js_angle_and_position(
    filename,
    expected_headings,
    expected_positions,
):
    run = run_source_kinematics_scenario(_SCENARIO_DIR / filename)
    payload = run.to_payload()
    frame = payload["trace"]["frames"][1]

    assert payload["runner"] == "curvytron-v1-python-source-kinematics-runner"
    assert payload["toy_v0_behavior"] is False
    assert payload["source_fidelity"] is True
    assert payload["source_fidelity_scope"] == "movement kinematics only"
    assert "first" not in payload["message"]
    assert frame["headings"] == expected_headings
    assert frame["positions"] == expected_positions


@pytest.mark.parametrize(
    "filename",
    [
        "source_kinematics_straight_multistep.json",
        "source_kinematics_turn_multistep.json",
    ],
)
def test_source_kinematics_multistep_matches_js_oracle_expectations(filename):
    scenario = load_scenario(_SCENARIO_DIR / filename)
    expected = scenario.comparison["expected"]

    js_payload = _run_js_scenario(filename)
    python_payload = run_source_kinematics_scenario(scenario).to_payload()
    python_frames = python_payload["trace"]["frames"][1:]

    assert len(python_frames) == len(expected["frames"])
    for frame, expected_frame in zip(python_frames, expected["frames"], strict=True):
        assert frame["positions"] == expected_frame["positions"]
        assert frame["headings"] == expected_frame["headings"]
        assert frame["alive"] == expected_frame["alive"]
        if "events" in expected_frame:
            assert [
                {"event": event["event"], "player_id": event["player_id"]}
                for event in frame["events"]
            ] == expected_frame["events"]

    final_frame = python_frames[-1]
    assert final_frame["positions"] == expected["final_state"]["positions"]
    assert final_frame["headings"] == expected["final_state"]["headings"]
    assert final_frame["alive"] == expected["final_state"]["alive"]
    assert final_frame["scores"] == expected["final_state"]["scores"]
    assert final_frame["roundScores"] == expected["final_state"]["roundScores"]
    assert diff_payload(js_payload, python_payload, common_trace=True)["status"] == "pass"


def test_source_kinematics_varied_elapsed_multistep_matches_js_oracle_expectations():
    scenario = load_scenario(
        _SCENARIO_DIR / "source_kinematics_varied_elapsed_multistep.json"
    )
    expected = scenario.comparison["expected"]

    js_payload = _run_js_scenario("source_kinematics_varied_elapsed_multistep.json")
    python_payload = run_source_kinematics_scenario(scenario).to_payload()
    python_frames = python_payload["trace"]["frames"][1:]

    assert len(python_frames) == len(expected["frames"])
    for frame, expected_frame in zip(python_frames, expected["frames"], strict=True):
        assert frame["positions"] == expected_frame["positions"]
        assert frame["headings"] == expected_frame["headings"]
        assert frame["alive"] == expected_frame["alive"]
        assert frame["stepMs"] == expected_frame["step_ms"]
        assert frame["events"] == expected_frame["events"]

    final_frame = python_frames[-1]
    assert final_frame["positions"] == expected["final_state"]["positions"]
    assert final_frame["headings"] == expected["final_state"]["headings"]
    assert final_frame["alive"] == expected["final_state"]["alive"]
    assert final_frame["scores"] == expected["final_state"]["scores"]
    assert final_frame["roundScores"] == expected["final_state"]["roundScores"]
    assert diff_payload(js_payload, python_payload, common_trace=True)["status"] == "pass"


def test_js_scenario_runner_pins_live_movement_angle_position_event_order():
    payload = _run_js_scenario("source_live_movement_event_trace_2p_no_bonus_multistep.json")
    frames = payload["trace"]

    assert payload["comparison"]["python_target"] == "curvyzero-source-env"
    assert payload["comparison"]["include_events"] is True
    assert [
        [event["data"]["avatar"] for event in frame["events"] if event["event"] == "position"]
        for frame in frames
    ] == [[2, 1], [2, 1], [2, 1], [2, 1]]
    assert [
        [event["data"]["avatar"] for event in frame["events"] if event["event"] == "angle"]
        for frame in frames
    ] == [[1], [2], [], [2, 1]]
    assert frames[0]["events"] == [
        {"event": "position", "data": {"avatar": 2, "x": 59.733333, "y": 40}},
        {"event": "angle", "data": {"avatar": 1, "angle": -0.046667}},
        {"event": "position", "data": {"avatar": 1, "x": 20.266376, "y": 39.98756}},
    ]
    assert frames[2]["events"] == [
        {"event": "position", "data": {"avatar": 2, "x": 59.200581, "y": 39.97512}},
        {"event": "position", "data": {"avatar": 1, "x": 20.799129, "y": 39.96268}},
    ]


def test_source_kinematics_runner_rejects_unscoped_scenario():
    scenario = {
        "scenario_id": "bonus_probe",
        "ruleset_id": "curvytron-v1-reference",
        "player_count": 2,
        "initial_state": {
            "positions": [[20, 40], [60, 40]],
            "headings": [0, math.pi],
        },
        "action_script": [{"moves": {"1": -1, "2": 1}}],
        "action_encoding": "source-move",
        "time_policy": {"kind": "fixed", "step_ms": 1000 / 60},
    }

    with pytest.raises(
        ScenarioError,
        match="source-kinematics runner supports source_kinematics_",
    ):
        run_source_kinematics_scenario(scenario)


@pytest.mark.parametrize(
    ("runner", "filename", "field", "value", "message"),
    [
        (
            run_source_kinematics_scenario,
            "source_kinematics_straight_step.json",
            "positions",
            [[20, 40], [60, 40], [40, 20]],
            "source-kinematics runner requires one forced position per player",
        ),
        (
            run_source_kinematics_scenario,
            "source_kinematics_straight_step.json",
            "headings",
            [0, math.pi, 0],
            "source-kinematics runner requires one forced heading per player",
        ),
        (
            run_source_kinematics_scenario,
            "source_kinematics_straight_step.json",
            "alive",
            [True, True, True],
            "source-kinematics runner requires one alive flag per player",
        ),
        (
            run_source_borderless_wrap_scenario,
            "source_borderless_wrap_step.json",
            "positions",
            [[87.35, 44], [44, 44], [20, 20]],
            "source-borderless-wrap runner requires one forced position per player",
        ),
        (
            run_source_borderless_wrap_scenario,
            "source_borderless_wrap_step.json",
            "headings",
            [0, math.pi, 0],
            "source-borderless-wrap runner requires one forced heading per player",
        ),
        (
            run_source_borderless_wrap_scenario,
            "source_borderless_wrap_step.json",
            "alive",
            [True, True, True],
            "source-borderless-wrap runner requires one alive flag per player",
        ),
    ],
)
def test_source_two_player_runners_reject_state_lengths_mismatched_to_player_count(
    runner,
    filename,
    field,
    value,
    message,
):
    scenario = load_scenario(_SCENARIO_DIR / filename).to_payload()
    scenario["initial_state"][field] = value

    with pytest.raises(ScenarioError, match=message):
        runner(scenario)


def test_source_alive_extraction_reads_player_initial_alive():
    scenario = {
        "scenario_id": "source_kinematics_initial_alive_shape",
        "ruleset_id": "curvytron-v1-reference",
        "player_count": 2,
        "seed": 123,
        "initial_state": {
            "map_size": 88,
            "players": [
                {"id": "p0", "initial": {"x": 20, "y": 40, "angle_rad": 0, "alive": False}},
                {"id": "p1", "initial": {"x": 60, "y": 40, "angle_rad": math.pi}},
            ],
        },
        "action_script": [{"moves": {"p0": 0, "p1": 0}}],
        "action_encoding": "source-move",
        "time_policy": {"kind": "fixed", "step_ms": 1000 / 60},
    }

    payload = run_source_kinematics_scenario(scenario).to_payload()

    assert payload["trace"]["frames"][0]["alive"] == [False, True]


def test_load_scenario_preserves_initial_world_bodies():
    scenario = load_scenario(
        _SCENARIO_DIR / "source_body_opponent_overlap_kills_step.json"
    ).to_payload()

    assert scenario["initial_state"]["map_size"] == 95
    assert scenario["initial_state"]["players"][0]["id"] == "p0"
    assert scenario["initial_state"]["world_bodies"] == [
        {"player_id": "p1", "x": 21.19, "y": 20, "radius": 0.6, "num": 0}
    ]


def test_load_scenario_preserves_initial_world_body_age_ms():
    scenario = load_scenario(
        _SCENARIO_DIR / "source_body_old_opponent_overlap_kills_step.json"
    ).to_payload()

    assert scenario["initial_state"]["world_bodies"] == [
        {
            "player_id": "p1",
            "x": 21.19,
            "y": 20,
            "radius": 0.6,
            "num": 0,
            "age_ms": 2000,
        }
    ]


def test_load_scenario_preserves_initial_player_body_counters():
    scenario = load_scenario(
        _SCENARIO_DIR / "source_body_own_delta3_safe_step.json"
    ).to_payload()

    p0_initial = scenario["initial_state"]["players"][0]["initial"]

    assert p0_initial["body_count"] == 3
    assert p0_initial["body_num"] == 3


def test_parse_scenario_promotes_top_level_world_bodies():
    scenario = load_scenario(
        _SCENARIO_DIR / "source_body_opponent_overlap_kills_step.json"
    ).to_payload()
    world_bodies = scenario["initial_state"].pop("world_bodies")
    scenario["world_bodies"] = world_bodies

    loaded = parse_scenario(scenario)

    assert loaded.initial_state["world_bodies"] == world_bodies


def test_scenario_schema_validates_initial_world_bodies():
    scenario = load_scenario(
        _SCENARIO_DIR / "source_body_opponent_overlap_kills_step.json"
    ).to_payload()
    del scenario["initial_state"]["world_bodies"][0]["num"]

    with pytest.raises(
        ScenarioError,
        match=r"initial_state\.world_bodies\[0\]\.num must be an integer",
    ):
        parse_scenario(scenario)


def test_scenario_schema_validates_initial_world_body_age_ms():
    scenario = load_scenario(
        _SCENARIO_DIR / "source_body_old_opponent_overlap_kills_step.json"
    ).to_payload()
    scenario["initial_state"]["world_bodies"][0]["age_ms"] = -1

    with pytest.raises(
        ScenarioError,
        match=r"initial_state\.world_bodies\[0\]\.age_ms must be a non-negative finite number",
    ):
        parse_scenario(scenario)


def test_scenario_schema_validates_initial_player_body_counters():
    scenario = load_scenario(
        _SCENARIO_DIR / "source_body_own_delta3_safe_step.json"
    ).to_payload()
    scenario["initial_state"]["players"][0]["initial"]["body_count"] = "bad"

    with pytest.raises(
        ScenarioError,
        match=r"initial_state\.players\[0\]\.initial\.body_count must be an integer",
    ):
        parse_scenario(scenario)


@pytest.mark.parametrize(
    ("filename", "expected_positions", "expected_alive", "expected_scores"),
    [
        (
            "source_normal_wall_death_step.json",
            [[88.95, 44.0], [42.4, 44.0]],
            [False, True],
            [0, 1],
        ),
        (
            "source_normal_wall_same_frame_draw_step.json",
            [[88.95, 44.0], [-0.95, 44.0]],
            [False, False],
            [0, 0],
        ),
    ],
)
def test_source_normal_wall_scenarios_match_js_state_fields(
    filename,
    expected_positions,
    expected_alive,
    expected_scores,
):
    run = run_source_normal_wall_scenario(_SCENARIO_DIR / filename)
    payload = run.to_payload()
    frame = payload["trace"]["frames"][1]

    assert payload["runner"] == "curvytron-v1-python-source-normal-wall-runner"
    assert payload["toy_v0_behavior"] is False
    assert payload["source_fidelity"] is True
    assert payload["source_fidelity_scope"] == "movement plus normal-wall death state/events only"
    assert frame["positions"] == expected_positions
    assert frame["headings"] == [0.0, 3.141593]
    assert frame["alive"] == expected_alive
    assert frame["scores"] == expected_scores
    assert frame["roundScores"] == [0, 0]


@pytest.mark.parametrize(
    (
        "filename",
        "frame_index",
        "expected_positions",
        "expected_headings",
        "expected_alive",
        "expected_scores",
    ),
    [
        (
            "source_normal_wall_3p_two_die_one_survivor_step.json",
            1,
            [[49.1, 47.5], [94.95, 47.5], [-0.55, 47.5]],
            [0.0, 0.0, 3.141593],
            [True, False, False],
            [2, 0, 0],
        ),
        (
            "source_normal_wall_4p_ordered_deaths_survivor_score.json",
            3,
            [[54.8, 50.0], [100.95, 50.0], [100.95, 50.0], [100.95, 50.0]],
            [0.0, 0.0, 0.0, 0.0],
            [True, False, False, False],
            [3, 0, 1, 2],
        ),
        (
            "source_normal_wall_4p_two_prior_then_same_frame_terminal_draw.json",
            3,
            [[100.95, 20.0], [100.95, 40.0], [100.95, 60.0], [100.95, 80.0]],
            [0.0, 0.0, 0.0, 0.0],
            [False, False, False, False],
            [2, 2, 1, 0],
        ),
    ],
)
def test_source_normal_wall_multiplayer_scenarios_match_js_state_fields(
    filename,
    frame_index,
    expected_positions,
    expected_headings,
    expected_alive,
    expected_scores,
):
    payload = run_source_normal_wall_scenario(_SCENARIO_DIR / filename).to_payload()
    frame = payload["trace"]["frames"][frame_index]

    assert payload["runner"] == "curvytron-v1-python-source-normal-wall-runner"
    assert payload["toy_v0_behavior"] is False
    assert payload["source_fidelity"] is True
    assert frame["positions"] == expected_positions
    assert frame["headings"] == expected_headings
    assert frame["alive"] == expected_alive
    assert frame["scores"] == expected_scores
    assert frame["roundScores"] == [0 for _ in expected_alive]


def test_source_borderless_wrap_scenario_matches_js_state_fields():
    run = run_source_borderless_wrap_scenario(_SCENARIO_DIR / "source_borderless_wrap_step.json")
    payload = run.to_payload()
    frame = payload["trace"]["frames"][1]

    assert payload["runner"] == "curvytron-v1-python-source-borderless-wrap-runner"
    assert payload["toy_v0_behavior"] is False
    assert payload["source_fidelity"] is True
    assert payload["source_fidelity_scope"] == (
        "movement plus source borderless wrap, exact-edge/corner-axis, "
        "and first destination-body skip state/events only"
    )
    assert frame["positions"] == [[0.0, 44.0], [42.4, 44.0]]
    assert frame["headings"] == [0.0, 3.141593]
    assert frame["alive"] == [True, True]
    assert frame["scores"] == [0, 0]
    assert frame["roundScores"] == [0, 0]


def test_js_scenario_runner_pins_borderless_print_manager_wrap_toggle():
    payload = _run_js_scenario("source_borderless_print_manager_wrap_toggle_step.json")
    frame = payload["trace"][0]
    avatar = frame["avatars"][0]

    assert payload["comparison"]["python_target"] == "source-borderless-wrap"
    assert payload["comparison"]["include_events"] is True
    assert frame["stepMs"] == 20
    assert frame["game"]["size"] == 80
    assert frame["game"]["deathCount"] == 0
    assert frame["game"]["worldBodyCount"] == 2
    assert avatar["name"] == "p0"
    assert avatar["x"] == 0
    assert avatar["y"] == 40
    assert avatar["alive"] is True
    assert avatar["printing"] is False
    assert avatar["trailPointCount"] == 0
    assert avatar["lastTrailPoint"] is None
    assert avatar["bodyNum"] == 1
    assert avatar["bodyCount"] == 2
    assert avatar["printManager"] == {
        "active": True,
        "distance": 5.25,
        "lastX": 0,
        "lastY": 40,
    }
    assert frame["events"] == [
        {"event": "position", "data": {"avatar": 1, "x": 80.12, "y": 40}},
        {
            "event": "point",
            "data": {"avatar": 1, "x": 80.12, "y": 40, "important": False},
        },
        {"event": "position", "data": {"avatar": 1, "x": 0, "y": 40}},
        {
            "event": "point",
            "data": {"avatar": 1, "x": 0, "y": 40, "important": True},
        },
        {
            "event": "property",
            "data": {"avatar": 1, "property": "printing", "value": False},
        },
    ]


def test_js_scenario_runner_pins_borderless_destination_body_skip_then_next_frame_kill():
    payload = _run_js_scenario(
        "source_borderless_wrap_skips_destination_body_then_next_frame_kills.json"
    )
    first_frame, second_frame = payload["trace"]
    first_avatars = {avatar["name"]: avatar for avatar in first_frame["avatars"]}
    second_avatars = {avatar["name"]: avatar for avatar in second_frame["avatars"]}

    assert payload["comparison"]["python_target"] == "source-borderless-wrap"
    assert payload["comparison"]["include_events"] is True
    assert first_frame["stepMs"] == 20
    assert first_frame["game"]["deathCount"] == 0
    assert first_frame["game"]["worldBodyCount"] == 1
    assert first_avatars["p0"]["x"] == 0
    assert first_avatars["p0"]["y"] == 44
    assert first_avatars["p0"]["alive"] is True
    assert first_frame["events"] == [
        {"event": "position", "data": {"avatar": 3, "x": 79.68, "y": 20}},
        {"event": "position", "data": {"avatar": 2, "x": 20.32, "y": 20}},
        {"event": "position", "data": {"avatar": 1, "x": 95.12, "y": 44}},
        {"event": "position", "data": {"avatar": 1, "x": 0, "y": 44}},
    ]

    assert second_frame["stepMs"] == 0
    assert second_frame["game"]["deathCount"] == 1
    assert second_frame["game"]["worldBodyCount"] == 2
    assert second_avatars["p0"]["alive"] is False
    assert second_avatars["p0"]["trailPointCount"] == 1
    assert second_avatars["p0"]["lastTrailPoint"] == [0, 44]
    assert second_frame["events"] == [
        {"event": "position", "data": {"avatar": 3, "x": 79.68, "y": 20}},
        {"event": "position", "data": {"avatar": 2, "x": 20.32, "y": 20}},
        {"event": "position", "data": {"avatar": 1, "x": 0, "y": 44}},
        {"event": "point", "data": {"avatar": 1, "x": 0, "y": 44, "important": False}},
        {"event": "die", "data": {"avatar": 1, "killer": 2, "old": False}},
        {"event": "score:round", "data": {"avatar": 1, "score": 0, "roundScore": 0}},
    ]


def test_js_scenario_runner_pins_borderless_exact_edge_corner_axis():
    payload = _run_js_scenario("source_borderless_exact_edge_corner_axis_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert payload["comparison"]["python_target"] == "source-borderless-wrap"
    assert payload["comparison"]["include_events"] is True
    assert frame["stepMs"] == 100
    assert frame["game"]["deathCount"] == 0
    assert frame["game"]["worldBodyCount"] == 0
    assert avatars["p1"]["x"] == 88
    assert avatars["p1"]["y"] == 20
    assert avatars["p1"]["alive"] is True
    assert avatars["p1"]["trailPointCount"] == 0
    assert avatars["p0"]["x"] == 0
    assert avatars["p0"]["y"] == 88.481371
    assert avatars["p0"]["alive"] is True
    assert avatars["p0"]["trailPointCount"] == 0
    assert frame["events"] == [
        {"event": "position", "data": {"avatar": 2, "x": 88, "y": 20}},
        {"event": "position", "data": {"avatar": 1, "x": 88.481371, "y": 88.481371}},
        {"event": "position", "data": {"avatar": 1, "x": 0, "y": 88.481371}},
    ]


def test_source_borderless_print_manager_wrap_matches_js_common_trace():
    filename = "source_borderless_print_manager_wrap_toggle_step.json"
    js_payload = _run_js_scenario(filename)
    python_payload = run_source_borderless_wrap_scenario(_SCENARIO_DIR / filename).to_payload()
    frame = python_payload["trace"]["frames"][1]
    common_step = project_common_trace(python_payload)["steps"][0]

    assert python_payload["runner"] == "curvytron-v1-python-source-borderless-wrap-runner"
    assert python_payload["source_fidelity_scope"] == (
        "movement plus source borderless wrap, exact-edge/corner-axis, "
        "and first destination-body skip state/events only"
    )
    assert frame["positions"] == [[0.0, 40.0]]
    assert frame["alive"] == [True]
    assert frame["worldBodyCount"] == 2
    assert frame["trailPointCounts"] == [0]
    assert frame["lastTrailPoints"] == [None]
    assert frame["bodyNums"] == [1]
    assert frame["bodyCounts"] == [2]
    assert frame["printing"] == [False]
    assert frame["printManagers"] == [
        {"active": True, "distance": 5.25, "lastX": 0.0, "lastY": 40.0}
    ]
    assert common_step["worldBodyCount"] == 2
    assert common_step["players"][0]["printManager"] == {
        "active": True,
        "distance": 5.25,
        "lastX": 0.0,
        "lastY": 40.0,
    }
    assert diff_payload(js_payload, python_payload, common_trace=True)["status"] == "pass"


def test_source_borderless_destination_body_skip_matches_js_common_trace():
    filename = "source_borderless_wrap_skips_destination_body_then_next_frame_kills.json"
    js_payload = _run_js_scenario(filename)
    python_payload = run_source_borderless_wrap_scenario(_SCENARIO_DIR / filename).to_payload()
    first_frame = python_payload["trace"]["frames"][1]
    second_frame = python_payload["trace"]["frames"][2]
    common_steps = project_common_trace(python_payload)["steps"]

    assert python_payload["runner"] == "curvytron-v1-python-source-borderless-wrap-runner"
    assert first_frame["stepMs"] == 20.0
    assert first_frame["positions"] == [[0.0, 44.0], [20.32, 20.0], [79.68, 20.0]]
    assert first_frame["alive"] == [True, True, True]
    assert first_frame["worldBodyCount"] == 1
    assert first_frame["bodyNums"] == [0, 1, 0]
    assert first_frame["bodyCounts"] == [0, 1, 0]
    assert first_frame["events"] == [
        {"event": "position", "player_id": "p2", "x": 79.68, "y": 20.0},
        {"event": "position", "player_id": "p1", "x": 20.32, "y": 20.0},
        {"event": "position", "player_id": "p0", "x": 95.12, "y": 44.0},
        {"event": "position", "player_id": "p0", "x": 0.0, "y": 44.0},
    ]

    assert second_frame["stepMs"] == 0.0
    assert second_frame["positions"] == [[0.0, 44.0], [20.32, 20.0], [79.68, 20.0]]
    assert second_frame["alive"] == [False, True, True]
    assert second_frame["worldBodyCount"] == 2
    assert second_frame["trailPointCounts"] == [1, 0, 0]
    assert second_frame["lastTrailPoints"] == [[0.0, 44.0], None, None]
    assert second_frame["bodyNums"] == [0, 1, 0]
    assert second_frame["bodyCounts"] == [1, 1, 0]
    assert second_frame["scores"] == [0, 0, 0]
    assert second_frame["roundScores"] == [0, 0, 0]
    assert second_frame["events"] == [
        {"event": "position", "player_id": "p2", "x": 79.68, "y": 20.0},
        {"event": "position", "player_id": "p1", "x": 20.32, "y": 20.0},
        {"event": "position", "player_id": "p0", "x": 0.0, "y": 44.0},
        {"event": "point", "player_id": "p0", "x": 0.0, "y": 44.0, "important": False},
        {"event": "die", "player_id": "p0", "killer_id": "p1", "old": False},
        {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 0},
    ]
    assert [step["step_ms"] for step in common_steps] == [20.0, 0.0]
    assert [step["worldBodyCount"] for step in common_steps] == [1, 2]
    assert diff_payload(js_payload, python_payload, common_trace=True)["status"] == "pass"


def test_source_borderless_exact_edge_corner_axis_matches_js_common_trace():
    filename = "source_borderless_exact_edge_corner_axis_step.json"
    js_payload = _run_js_scenario(filename)
    python_payload = run_source_borderless_wrap_scenario(_SCENARIO_DIR / filename).to_payload()
    frame = python_payload["trace"]["frames"][1]

    assert python_payload["runner"] == "curvytron-v1-python-source-borderless-wrap-runner"
    assert frame["positions"] == [[0.0, 88.481371], [88.0, 20.0]]
    assert frame["alive"] == [True, True]
    assert frame["events"] == [
        {"event": "position", "player_id": "p1", "x": 88.0, "y": 20.0},
        {"event": "position", "player_id": "p0", "x": 88.481371, "y": 88.481371},
        {"event": "position", "player_id": "p0", "x": 0.0, "y": 88.481371},
    ]
    assert diff_payload(js_payload, python_payload, common_trace=True)["status"] == "pass"


@pytest.mark.parametrize(
    ("filename", "expected_events"),
    [
        (
            "source_normal_wall_death_step.json",
            [
                {"event": "position", "player_id": "p1", "x": 42.4, "y": 44.0},
                {"event": "position", "player_id": "p0", "x": 88.95, "y": 44.0},
                {
                    "event": "point",
                    "player_id": "p0",
                    "x": 88.95,
                    "y": 44.0,
                    "important": False,
                },
                {"event": "die", "player_id": "p0", "killer_id": None, "old": None},
                {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 0},
                {"event": "score:round", "player_id": "p1", "score": 0, "roundScore": 1},
                {"event": "score", "player_id": "p1", "score": 1, "roundScore": 1},
                {"event": "score", "player_id": "p0", "score": 0, "roundScore": 0},
                {"event": "round:end", "winner_id": "p1"},
            ],
        ),
        (
            "source_normal_wall_same_frame_draw_step.json",
            [
                {"event": "position", "player_id": "p1", "x": -0.95, "y": 44.0},
                {
                    "event": "point",
                    "player_id": "p1",
                    "x": -0.95,
                    "y": 44.0,
                    "important": False,
                },
                {"event": "die", "player_id": "p1", "killer_id": None, "old": None},
                {"event": "score:round", "player_id": "p1", "score": 0, "roundScore": 0},
                {"event": "position", "player_id": "p0", "x": 88.95, "y": 44.0},
                {
                    "event": "point",
                    "player_id": "p0",
                    "x": 88.95,
                    "y": 44.0,
                    "important": False,
                },
                {"event": "die", "player_id": "p0", "killer_id": None, "old": None},
                {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 0},
                {"event": "score", "player_id": "p1", "score": 0, "roundScore": 0},
                {"event": "score", "player_id": "p0", "score": 0, "roundScore": 0},
                {"event": "round:end", "winner_id": None},
            ],
        ),
        (
            "source_borderless_wrap_step.json",
            [
                {"event": "position", "player_id": "p1", "x": 42.4, "y": 44.0},
                {"event": "position", "player_id": "p0", "x": 88.95, "y": 44.0},
                {"event": "position", "player_id": "p0", "x": 0.0, "y": 44.0},
            ],
        ),
        (
            "source_borderless_exact_edge_corner_axis_step.json",
            [
                {"event": "position", "player_id": "p1", "x": 88.0, "y": 20.0},
                {"event": "position", "player_id": "p0", "x": 88.481371, "y": 88.481371},
                {"event": "position", "player_id": "p0", "x": 0.0, "y": 88.481371},
            ],
        ),
    ],
)
def test_source_border_rule_runners_emit_opt_in_events(filename, expected_events):
    payload = run_source_border_rules_scenario(_SCENARIO_DIR / filename).to_payload()

    assert payload["scenario"]["comparison"]["include_events"] is True
    assert payload["trace"]["frames"][1]["events"] == expected_events


def test_source_normal_wall_3p_events_preserve_reverse_death_and_score_order():
    payload = run_source_border_rules_scenario(
        _SCENARIO_DIR / "source_normal_wall_3p_two_die_one_survivor_step.json"
    ).to_payload()

    assert payload["trace"]["frames"][1]["events"] == [
        {"event": "position", "player_id": "p2", "x": -0.55, "y": 47.5},
        {"event": "point", "player_id": "p2", "x": -0.55, "y": 47.5, "important": False},
        {"event": "die", "player_id": "p2", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p2", "score": 0, "roundScore": 0},
        {"event": "position", "player_id": "p1", "x": 94.95, "y": 47.5},
        {"event": "point", "player_id": "p1", "x": 94.95, "y": 47.5, "important": False},
        {"event": "die", "player_id": "p1", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p1", "score": 0, "roundScore": 0},
        {"event": "position", "player_id": "p0", "x": 49.1, "y": 47.5},
        {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 2},
        {"event": "score", "player_id": "p2", "score": 0, "roundScore": 0},
        {"event": "score", "player_id": "p1", "score": 0, "roundScore": 0},
        {"event": "score", "player_id": "p0", "score": 2, "roundScore": 2},
        {"event": "round:end", "winner_id": "p0"},
    ]


def test_source_normal_wall_4p_events_preserve_ordered_death_score_timing():
    payload = run_source_border_rules_scenario(
        _SCENARIO_DIR / "source_normal_wall_4p_ordered_deaths_survivor_score.json"
    ).to_payload()
    frames = payload["trace"]["frames"]

    assert frames[1]["events"] == [
        {"event": "position", "player_id": "p3", "x": 97.75, "y": 50.0},
        {"event": "position", "player_id": "p2", "x": 99.35, "y": 50.0},
        {"event": "position", "player_id": "p1", "x": 100.95, "y": 50.0},
        {"event": "point", "player_id": "p1", "x": 100.95, "y": 50.0, "important": False},
        {"event": "die", "player_id": "p1", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p1", "score": 0, "roundScore": 0},
        {"event": "position", "player_id": "p0", "x": 51.6, "y": 50.0},
    ]
    assert frames[2]["events"] == [
        {"event": "position", "player_id": "p3", "x": 99.35, "y": 50.0},
        {"event": "position", "player_id": "p2", "x": 100.95, "y": 50.0},
        {"event": "point", "player_id": "p2", "x": 100.95, "y": 50.0, "important": False},
        {"event": "die", "player_id": "p2", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p2", "score": 0, "roundScore": 1},
        {"event": "position", "player_id": "p0", "x": 53.2, "y": 50.0},
    ]
    assert frames[3]["events"] == [
        {"event": "position", "player_id": "p3", "x": 100.95, "y": 50.0},
        {"event": "point", "player_id": "p3", "x": 100.95, "y": 50.0, "important": False},
        {"event": "die", "player_id": "p3", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p3", "score": 0, "roundScore": 2},
        {"event": "position", "player_id": "p0", "x": 54.8, "y": 50.0},
        {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 3},
        {"event": "score", "player_id": "p3", "score": 2, "roundScore": 2},
        {"event": "score", "player_id": "p2", "score": 1, "roundScore": 1},
        {"event": "score", "player_id": "p1", "score": 0, "roundScore": 0},
        {"event": "score", "player_id": "p0", "score": 3, "roundScore": 3},
        {"event": "round:end", "winner_id": "p0"},
    ]


def test_source_normal_wall_4p_same_frame_terminal_draw_preserves_event_order():
    payload = run_source_border_rules_scenario(
        _SCENARIO_DIR / "source_normal_wall_4p_two_prior_then_same_frame_terminal_draw.json"
    ).to_payload()
    frames = payload["trace"]["frames"]

    assert frames[1]["events"] == [
        {"event": "position", "player_id": "p3", "x": 100.95, "y": 80.0},
        {"event": "point", "player_id": "p3", "x": 100.95, "y": 80.0, "important": False},
        {"event": "die", "player_id": "p3", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p3", "score": 0, "roundScore": 0},
        {"event": "position", "player_id": "p2", "x": 99.35, "y": 60.0},
        {"event": "position", "player_id": "p1", "x": 97.75, "y": 40.0},
        {"event": "position", "player_id": "p0", "x": 97.75, "y": 20.0},
    ]
    assert frames[2]["events"] == [
        {"event": "position", "player_id": "p2", "x": 100.95, "y": 60.0},
        {"event": "point", "player_id": "p2", "x": 100.95, "y": 60.0, "important": False},
        {"event": "die", "player_id": "p2", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p2", "score": 0, "roundScore": 1},
        {"event": "position", "player_id": "p1", "x": 99.35, "y": 40.0},
        {"event": "position", "player_id": "p0", "x": 99.35, "y": 20.0},
    ]
    assert frames[3]["events"] == [
        {"event": "position", "player_id": "p1", "x": 100.95, "y": 40.0},
        {"event": "point", "player_id": "p1", "x": 100.95, "y": 40.0, "important": False},
        {"event": "die", "player_id": "p1", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p1", "score": 0, "roundScore": 2},
        {"event": "position", "player_id": "p0", "x": 100.95, "y": 20.0},
        {"event": "point", "player_id": "p0", "x": 100.95, "y": 20.0, "important": False},
        {"event": "die", "player_id": "p0", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 2},
        {"event": "score", "player_id": "p3", "score": 0, "roundScore": 0},
        {"event": "score", "player_id": "p2", "score": 1, "roundScore": 1},
        {"event": "score", "player_id": "p1", "score": 2, "roundScore": 2},
        {"event": "score", "player_id": "p0", "score": 2, "roundScore": 2},
        {"event": "round:end", "winner_id": None},
    ]


def test_source_border_rule_runner_omits_events_without_opt_in():
    scenario = load_scenario(_SCENARIO_DIR / "source_normal_wall_death_step.json").to_payload()
    scenario["comparison"].pop("include_events", None)

    payload = run_source_border_rules_scenario(scenario).to_payload()

    assert "events" not in payload["trace"]["frames"][1]


@pytest.mark.parametrize(
    (
        "filename",
        "expected_p0_alive",
        "expected_raw_die_events",
        "expected_common_die_events",
        "expected_world_body_count",
    ),
    [
        (
            "source_body_opponent_tangent_safe_step.json",
            True,
            [],
            [],
            1,
        ),
        (
            "source_body_opponent_overlap_kills_step.json",
            False,
            [{"event": "die", "data": {"avatar": 1, "killer": 2, "old": False}}],
            [
                {
                    "event": "die",
                    "player_id": "p0",
                    "killer_id": "p1",
                    "old": False,
                }
            ],
            2,
        ),
    ],
)
def test_js_scenario_runner_seeds_initial_world_bodies(
    filename,
    expected_p0_alive,
    expected_raw_die_events,
    expected_common_die_events,
    expected_world_body_count,
):
    payload = _run_js_scenario(filename)
    frame = payload["trace"][0]
    p0 = next(avatar for avatar in frame["avatars"] if avatar["name"] == "p0")

    assert payload["comparison"]["python_target"] == "source-body-canary"
    assert payload["comparison"]["include_events"] is True
    assert frame["stepMs"] == 0
    assert frame["game"]["size"] == 95
    assert frame["game"]["worldBodyCount"] == expected_world_body_count
    assert p0["alive"] is expected_p0_alive
    raw_die_events = [event for event in frame["events"] if event["event"] == "die"]
    assert raw_die_events == expected_raw_die_events
    assert [
        event
        for event in project_common_trace(payload)["steps"][0]["events"]
        if event["event"] == "die"
    ] == expected_common_die_events


def test_js_scenario_runner_catches_seeded_active_bonus_self_small():
    payload = _run_js_scenario("source_bonus_self_small_catch_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}
    bonus_data = {
        "id": 1,
        "type": "BonusSelfSmall",
        "duration": 7500,
        "effects": [["radius", -1]],
    }

    assert payload["comparison"]["python_target"] == "source-bonus-js-oracle"
    assert payload["comparison"]["include_events"] is True
    assert payload["randomCalls"] == []
    assert "src/server/model/Bonus/BonusSelfSmall.js" in payload["loadedSources"]
    assert frame["stepMs"] == 100
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
    assert frame["events"] == [
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


def test_js_scenario_runner_bonus_game_clear_immediately_clears_world():
    payload = _run_js_scenario("source_bonus_game_clear_immediate_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert payload["comparison"]["python_target"] == "source-bonus-js-oracle"
    assert payload["comparison"]["include_events"] is True
    assert payload["randomCalls"] == []
    assert "src/server/model/Bonus/BonusGame.js" in payload["loadedSources"]
    assert "src/server/model/Bonus/BonusGameClear.js" in payload["loadedSources"]
    assert frame["stepMs"] == 100
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
    assert frame["events"] == [
        {"event": "position", "data": {"avatar": 2, "x": 71.6, "y": 70}},
        {"event": "position", "data": {"avatar": 1, "x": 21.6, "y": 20}},
        {"event": "bonus:clear", "data": {"bonus": 1}},
        {"event": "clear", "data": {}},
    ]
    assert not any(event["event"] == "bonus:stack" for event in frame["events"])
    assert not any(event["event"] == "property" for event in frame["events"])


def test_js_scenario_runner_bonus_self_small_expiry_restores_radius():
    payload = _run_js_scenario("source_bonus_self_small_expiry_restore_step.json")
    catch_frame, expiry_frame = payload["trace"]
    catch_avatars = {avatar["name"]: avatar for avatar in catch_frame["avatars"]}
    expiry_avatars = {avatar["name"]: avatar for avatar in expiry_frame["avatars"]}
    bonus_data = {
        "id": 1,
        "type": "BonusSelfSmall",
        "duration": 7500,
        "effects": [["radius", -1]],
    }

    assert payload["comparison"]["python_target"] == "source-bonus-js-oracle"
    assert payload["comparison"]["include_events"] is True
    assert payload["randomCalls"] == []
    assert catch_frame["stepMs"] == 100
    assert catch_avatars["p0"]["radius"] == 0.3
    assert catch_avatars["p0"]["activeBonuses"] == [bonus_data]
    assert expiry_frame["stepMs"] == 0
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


def test_js_scenario_runner_bonus_game_borderless_expiry_restores_borderless():
    payload = _run_js_scenario_with_game_bonus_stack(
        "source_bonus_game_borderless_expiry_restore_step.json"
    )
    catch_frame, expiry_frame = payload["trace"]
    catch_avatars = {avatar["name"]: avatar for avatar in catch_frame["avatars"]}
    expiry_avatars = {avatar["name"]: avatar for avatar in expiry_frame["avatars"]}
    bonus_data = {
        "id": 1,
        "type": "BonusGameBorderless",
        "duration": 10000,
        "effects": [["borderless", True]],
    }

    assert payload["comparison"]["python_target"] == "source-bonus-js-oracle"
    assert payload["comparison"]["include_events"] is True
    assert payload["randomCalls"] == []
    assert "src/server/model/Bonus/BonusGameBorderless.js" in payload["loadedSources"]
    assert catch_frame["stepMs"] == 100
    assert catch_frame["game"]["borderless"] is True
    assert catch_frame["game"]["activeBonuses"] == [bonus_data]
    assert catch_frame["game"]["bonusCount"] == 0
    assert catch_frame["game"]["bonusWorldBodyCount"] == 1
    assert catch_avatars["p0"]["activeBonuses"] == []
    assert catch_avatars["p1"]["activeBonuses"] == []
    assert catch_frame["events"] == [
        {"event": "position", "data": {"avatar": 2, "x": 71.6, "y": 70}},
        {"event": "position", "data": {"avatar": 1, "x": 21.6, "y": 20}},
        {"event": "bonus:clear", "data": {"bonus": 1}},
        {"event": "borderless", "data": {"value": True}},
        {
            "event": "bonus:stack",
            "data": {"target": "game", "method": "add", "bonus": bonus_data},
        },
    ]

    assert expiry_frame["stepMs"] == 0
    assert expiry_frame["game"]["borderless"] is False
    assert expiry_frame["game"]["activeBonuses"] == []
    assert expiry_frame["game"]["bonusCount"] == 0
    assert expiry_frame["game"]["bonusWorldBodyCount"] == 1
    assert expiry_avatars["p0"]["activeBonuses"] == []
    assert expiry_avatars["p1"]["activeBonuses"] == []
    assert expiry_frame["events"] == [
        {"event": "borderless", "data": {"value": False}},
        {
            "event": "bonus:stack",
            "data": {"target": "game", "method": "remove", "bonus": bonus_data},
        },
        {"event": "position", "data": {"avatar": 2, "x": 71.6, "y": 70}},
        {"event": "position", "data": {"avatar": 1, "x": 21.6, "y": 20}},
    ]
    assert not any(event["event"] == "bonus:clear" for event in expiry_frame["events"])


def test_js_scenario_runner_bonus_self_small_tangent_does_not_catch():
    payload = _run_js_scenario("source_bonus_self_small_tangent_no_catch_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert payload["comparison"]["python_target"] == "source-bonus-js-oracle"
    assert payload["randomCalls"] == []
    assert frame["stepMs"] == 100
    assert frame["game"]["bonusCount"] == 1
    assert frame["game"]["bonusWorldBodyCount"] == 1
    assert avatars["p0"]["x"] == 21.6
    assert avatars["p0"]["radius"] == 0.6
    assert avatars["p0"]["activeBonuses"] == []
    assert avatars["p1"]["radius"] == 0.6
    assert avatars["p1"]["activeBonuses"] == []
    assert frame["events"] == [
        {"event": "position", "data": {"avatar": 2, "x": 71.6, "y": 70}},
        {"event": "position", "data": {"avatar": 1, "x": 21.6, "y": 20}},
    ]


def test_js_scenario_runner_bonus_self_small_wall_death_does_not_catch():
    payload = _run_js_scenario("source_bonus_self_small_wall_death_no_catch_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert payload["comparison"]["python_target"] == "source-bonus-js-oracle"
    assert payload["randomCalls"] == []
    assert frame["stepMs"] == 100
    assert frame["game"]["inRound"] is False
    assert frame["game"]["deaths"] == [1]
    assert frame["game"]["worldBodyCount"] == 1
    assert frame["game"]["bonusCount"] == 1
    assert frame["game"]["bonusWorldBodyCount"] == 1
    assert avatars["p0"]["x"] == 87.9
    assert avatars["p0"]["alive"] is False
    assert avatars["p0"]["radius"] == 0.6
    assert avatars["p0"]["activeBonuses"] == []
    assert avatars["p1"]["alive"] is True
    assert avatars["p1"]["score"] == 1
    assert [event["event"] for event in frame["events"]] == [
        "position",
        "position",
        "point",
        "die",
        "score:round",
        "score:round",
        "score",
        "score",
        "round:end",
    ]
    assert not any(
        event["event"] in {"bonus:clear", "bonus:stack"} for event in frame["events"]
    )


def test_js_scenario_runner_natural_bonus_spawn_type_position_rng_order():
    payload = _run_js_scenario("source_bonus_spawn_type_position_rng_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert payload["comparison"]["python_target"] == "source-bonus-js-oracle"
    assert payload["randomCalls"] == [
        {"index": 0, "value": 0, "label": "bonus.start_delay"},
        {"index": 1, "value": 0.5, "label": "bonus.next_delay_after_pop"},
        {"index": 2, "value": 0.2, "label": "bonus.type.BonusSelfSmall"},
        {"index": 3, "value": 0.25, "label": "bonus.position.x"},
        {"index": 4, "value": 0.75, "label": "bonus.position.y"},
    ]
    assert frame["stepMs"] == 0
    assert frame["game"]["size"] == 88
    assert frame["game"]["worldBodyCount"] == 0
    assert frame["game"]["bonusCount"] == 1
    assert frame["game"]["bonusWorldBodyCount"] == 1
    assert avatars["p0"]["x"] == 20
    assert avatars["p0"]["y"] == 20
    assert avatars["p0"]["activeBonuses"] == []
    assert avatars["p1"]["activeBonuses"] == []
    assert frame["events"] == [
        {
            "event": "bonus:pop",
            "data": {
                "bonus": 1,
                "type": "BonusSelfSmall",
                "x": 23.94,
                "y": 64.06,
            },
        },
        {"event": "position", "data": {"avatar": 2, "x": 70, "y": 70}},
        {"event": "position", "data": {"avatar": 1, "x": 20, "y": 20}},
    ]


def test_js_scenario_runner_natural_bonus_spawn_retries_after_game_world_collision():
    payload = _run_js_scenario("source_bonus_spawn_game_world_retry_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert payload["comparison"]["python_target"] == "source-bonus-js-oracle"
    assert payload["randomCalls"] == [
        {"index": 0, "value": 0, "label": "bonus.start_delay"},
        {"index": 1, "value": 0.5, "label": "bonus.next_delay_after_pop"},
        {"index": 2, "value": 0.2, "label": "bonus.type.BonusSelfSmall"},
        {"index": 3, "value": 0.25, "label": "bonus.position.x"},
        {"index": 4, "value": 0.75, "label": "bonus.position.y"},
        {"index": 5, "value": 0.8, "label": "bonus.position.retry_1.x"},
        {"index": 6, "value": 0.2, "label": "bonus.position.retry_1.y"},
    ]
    retry_draws = [
        call for call in payload["randomCalls"] if ".retry_" in call["label"]
    ]
    assert len(retry_draws) == 2
    assert frame["stepMs"] == 0
    assert frame["game"]["size"] == 88
    assert frame["game"]["worldBodyCount"] == 1
    assert frame["game"]["bonusCount"] == 1
    assert frame["game"]["bonusWorldBodyCount"] == 1
    assert avatars["p0"]["activeBonuses"] == []
    assert avatars["p1"]["activeBonuses"] == []
    assert frame["events"] == [
        {
            "event": "bonus:pop",
            "data": {
                "bonus": 1,
                "type": "BonusSelfSmall",
                "x": 68.072,
                "y": 19.928,
            },
        },
        {"event": "position", "data": {"avatar": 2, "x": 70, "y": 70}},
        {"event": "position", "data": {"avatar": 1, "x": 20, "y": 20}},
    ]


def test_js_scenario_runner_natural_bonus_spawn_retries_after_bonus_world_collision():
    payload = _run_js_scenario("source_bonus_spawn_bonus_world_retry_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert payload["comparison"]["python_target"] == "source-bonus-js-oracle"
    assert payload["randomCalls"] == [
        {"index": 0, "value": 0, "label": "bonus.start_delay"},
        {"index": 1, "value": 0.5, "label": "bonus.next_delay_after_pop"},
        {"index": 2, "value": 0.2, "label": "bonus.type.BonusSelfSmall"},
        {"index": 3, "value": 0.25, "label": "bonus.position.x"},
        {"index": 4, "value": 0.75, "label": "bonus.position.y"},
        {"index": 5, "value": 0.8, "label": "bonus.position.retry_1.x"},
        {"index": 6, "value": 0.2, "label": "bonus.position.retry_1.y"},
    ]
    retry_draws = [
        call for call in payload["randomCalls"] if ".retry_" in call["label"]
    ]
    assert len(retry_draws) == 2
    assert frame["stepMs"] == 0
    assert frame["game"]["size"] == 88
    assert frame["game"]["worldBodyCount"] == 0
    assert frame["game"]["bonusCount"] == 2
    assert frame["game"]["bonusWorldBodyCount"] == 2
    assert avatars["p0"]["activeBonuses"] == []
    assert avatars["p1"]["activeBonuses"] == []
    assert frame["events"] == [
        {
            "event": "bonus:pop",
            "data": {
                "bonus": 2,
                "type": "BonusSelfSmall",
                "x": 68.072,
                "y": 19.928,
            },
        },
        {"event": "position", "data": {"avatar": 2, "x": 70, "y": 70}},
        {"event": "position", "data": {"avatar": 1, "x": 20, "y": 20}},
    ]


def test_js_scenario_runner_natural_bonus_pop_skips_spawn_at_cap():
    payload = _run_js_scenario("source_bonus_spawn_cap_twenty_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert payload["comparison"]["python_target"] == "source-bonus-js-oracle"
    assert payload["randomCalls"] == [
        {"index": 0, "value": 0, "label": "bonus.start_delay"},
        {"index": 1, "value": 0.5, "label": "bonus.next_delay_after_pop"},
    ]
    assert "src/shared/manager/BaseBonusManager.js" in payload["loadedSources"]
    assert "src/server/manager/BonusManager.js" in payload["loadedSources"]
    assert frame["stepMs"] == 0
    assert frame["game"]["size"] == 88
    assert frame["game"]["worldBodyCount"] == 0
    assert frame["game"]["bonusCount"] == 20
    assert frame["game"]["bonusWorldBodyCount"] == 20
    assert avatars["p0"]["activeBonuses"] == []
    assert avatars["p1"]["activeBonuses"] == []
    assert not any(event["event"] == "bonus:pop" for event in frame["events"])
    assert frame["events"] == [
        {"event": "position", "data": {"avatar": 2, "x": 70, "y": 70}},
        {"event": "position", "data": {"avatar": 1, "x": 20, "y": 20}},
    ]


def test_js_scenario_runner_default_bonus_weights_select_expected_type():
    payload = _run_js_scenario("source_bonus_default_weights_type_rng_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert payload["comparison"]["python_target"] == "source-bonus-js-oracle"
    assert payload["randomCalls"] == [
        {"index": 0, "value": 0, "label": "bonus.start_delay"},
        {"index": 1, "value": 0.5, "label": "bonus.next_delay_after_pop"},
        {"index": 2, "value": 0.945, "label": "bonus.type.BonusAllColor"},
        {"index": 3, "value": 0.25, "label": "bonus.position.x"},
        {"index": 4, "value": 0.75, "label": "bonus.position.y"},
    ]
    assert "src/server/model/Bonus/BonusAllColor.js" in payload["loadedSources"]
    assert "src/server/model/Bonus/BonusEnemyStraightAngle.js" in payload["loadedSources"]
    assert frame["stepMs"] == 0
    assert frame["game"]["size"] == 101
    assert frame["game"]["worldBodyCount"] == 0
    assert frame["game"]["bonusCount"] == 1
    assert frame["game"]["bonusWorldBodyCount"] == 1
    assert avatars["p0"]["alive"] is True
    assert avatars["p1"]["alive"] is True
    assert avatars["p2"]["alive"] is False
    assert avatars["p3"]["alive"] is False
    assert avatars["p0"]["activeBonuses"] == []
    assert avatars["p1"]["activeBonuses"] == []
    assert frame["events"] == [
        {
            "event": "bonus:pop",
            "data": {
                "bonus": 1,
                "type": "BonusAllColor",
                "x": 27.255,
                "y": 73.745,
            },
        },
        {"event": "position", "data": {"avatar": 2, "x": 80, "y": 80}},
        {"event": "position", "data": {"avatar": 1, "x": 20, "y": 20}},
    ]


def test_js_scenario_runner_default_bonus_weights_select_bonus_game_clear():
    payload = _run_js_scenario("source_bonus_default_weights_select_game_clear_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert payload["comparison"]["python_target"] == "source-bonus-js-oracle"
    assert payload["randomCalls"] == [
        {"index": 0, "value": 0, "label": "bonus.start_delay"},
        {"index": 1, "value": 0.5, "label": "bonus.next_delay_after_pop"},
        {"index": 2, "value": 0.965, "label": "bonus.type.BonusGameClear"},
        {"index": 3, "value": 0.25, "label": "bonus.position.x"},
        {"index": 4, "value": 0.75, "label": "bonus.position.y"},
    ]
    assert "src/server/model/Bonus/BonusGameClear.js" in payload["loadedSources"]
    assert frame["stepMs"] == 0
    assert frame["game"]["size"] == 101
    assert frame["game"]["worldBodyCount"] == 0
    assert frame["game"]["bonusCount"] == 1
    assert frame["game"]["bonusWorldBodyCount"] == 1
    assert avatars["p0"]["alive"] is True
    assert avatars["p1"]["alive"] is True
    assert avatars["p2"]["alive"] is False
    assert avatars["p3"]["alive"] is False
    assert avatars["p0"]["activeBonuses"] == []
    assert avatars["p1"]["activeBonuses"] == []
    assert frame["events"] == [
        {
            "event": "bonus:pop",
            "data": {
                "bonus": 1,
                "type": "BonusGameClear",
                "x": 27.255,
                "y": 73.745,
            },
        },
        {"event": "position", "data": {"avatar": 2, "x": 80, "y": 80}},
        {"event": "position", "data": {"avatar": 1, "x": 20, "y": 20}},
    ]


def test_js_scenario_runner_default_bonus_weights_keep_game_clear_full_probability():
    payload = _run_js_scenario(
        "source_bonus_default_weights_game_clear_full_probability_step.json"
    )
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert payload["comparison"]["python_target"] == "source-bonus-js-oracle"
    assert payload["randomCalls"] == [
        {"index": 0, "value": 0, "label": "bonus.start_delay"},
        {"index": 1, "value": 0.5, "label": "bonus.next_delay_after_pop"},
        {"index": 2, "value": 0.93, "label": "bonus.type.BonusGameClear"},
        {"index": 3, "value": 0.25, "label": "bonus.position.x"},
        {"index": 4, "value": 0.75, "label": "bonus.position.y"},
    ]
    assert "src/server/model/Bonus/BonusGameClear.js" in payload["loadedSources"]
    assert frame["stepMs"] == 0
    assert frame["game"]["size"] == 101
    assert frame["game"]["worldBodyCount"] == 0
    assert frame["game"]["bonusCount"] == 1
    assert frame["game"]["bonusWorldBodyCount"] == 1
    assert avatars["p0"]["alive"] is True
    assert avatars["p1"]["alive"] is True
    assert avatars["p2"]["alive"] is False
    assert avatars["p3"]["alive"] is True
    assert avatars["p0"]["activeBonuses"] == []
    assert avatars["p1"]["activeBonuses"] == []
    assert avatars["p3"]["activeBonuses"] == []
    assert frame["events"] == [
        {
            "event": "bonus:pop",
            "data": {
                "bonus": 1,
                "type": "BonusGameClear",
                "x": 27.255,
                "y": 73.745,
            },
        },
        {"event": "position", "data": {"avatar": 4, "x": 80, "y": 40}},
        {"event": "position", "data": {"avatar": 2, "x": 80, "y": 80}},
        {"event": "position", "data": {"avatar": 1, "x": 20, "y": 20}},
    ]


def test_source_body_old_metadata_reports_old_true_in_js_and_python_common_trace():
    if shutil.which("node") is None:
        pytest.skip("node executable is not available")

    filename = "source_body_old_opponent_overlap_kills_step.json"
    js_payload = _run_js_scenario(filename)
    python_payload = run_source_body_canary_scenario(_SCENARIO_DIR / filename).to_payload()

    js_frame = js_payload["trace"][0]
    js_die_events = [event for event in js_frame["events"] if event["event"] == "die"]
    assert js_die_events == [
        {"event": "die", "data": {"avatar": 1, "killer": 2, "old": True}}
    ]
    assert [
        event
        for event in project_common_trace(js_payload)["steps"][0]["events"]
        if event["event"] == "die"
    ] == [
        {"event": "die", "player_id": "p0", "killer_id": "p1", "old": True}
    ]

    python_frame = python_payload["trace"]["frames"][1]
    assert [
        event for event in python_frame["events"] if event["event"] == "die"
    ] == [
        {"event": "die", "player_id": "p0", "killer_id": "p1", "old": True}
    ]
    assert [
        event
        for event in project_common_trace(python_payload)["steps"][0]["events"]
        if event["event"] == "die"
    ] == [
        {"event": "die", "player_id": "p0", "killer_id": "p1", "old": True}
    ]
    assert diff_payload(js_payload, python_payload, common_trace=True)["status"] == "pass"


@pytest.mark.parametrize(
    (
        "filename",
        "expected_p0_alive",
        "expected_p0_body_num",
        "expected_p0_body_count",
        "expected_world_body_count",
        "expected_raw_die_events",
        "expected_common_die_events",
    ),
    [
        (
            "source_body_own_delta3_safe_step.json",
            True,
            3,
            3,
            1,
            [],
            [],
        ),
        (
            "source_body_own_delta4_kills_step.json",
            False,
            4,
            5,
            2,
            [{"event": "die", "data": {"avatar": 1, "killer": 1, "old": False}}],
            [
                {
                    "event": "die",
                    "player_id": "p0",
                    "killer_id": "p0",
                    "old": False,
                }
            ],
        ),
    ],
)
def test_js_scenario_runner_pins_own_body_latency(
    filename,
    expected_p0_alive,
    expected_p0_body_num,
    expected_p0_body_count,
    expected_world_body_count,
    expected_raw_die_events,
    expected_common_die_events,
):
    payload = _run_js_scenario(filename)
    frame = payload["trace"][0]
    p0 = next(avatar for avatar in frame["avatars"] if avatar["name"] == "p0")

    assert payload["comparison"]["python_target"] == "source-body-canary"
    assert payload["comparison"]["include_events"] is True
    assert frame["stepMs"] == 0
    assert frame["game"]["size"] == 95
    assert frame["game"]["worldBodyCount"] == expected_world_body_count
    assert frame["game"]["roundWinner"] is None
    assert all(avatar["printing"] is False for avatar in frame["avatars"])
    assert p0["alive"] is expected_p0_alive
    assert p0["bodyNum"] == expected_p0_body_num
    assert p0["bodyCount"] == expected_p0_body_count
    raw_die_events = [event for event in frame["events"] if event["event"] == "die"]
    assert raw_die_events == expected_raw_die_events
    assert [
        event
        for event in project_common_trace(payload)["steps"][0]["events"]
        if event["event"] == "die"
    ] == expected_common_die_events


@pytest.mark.parametrize(
    (
        "filename",
        "expected_alive",
        "expected_world_body_count",
        "expected_trail_counts",
        "expected_raw_events",
        "expected_common_events",
    ),
    [
        (
            "source_body_same_frame_point_kills_step.json",
            [False, True, True],
            2,
            [1, 1, 0],
            [
                {"event": "position", "data": {"avatar": 3, "x": 78.4, "y": 20}},
                {"event": "position", "data": {"avatar": 2, "x": 41.6, "y": 40}},
                {
                    "event": "point",
                    "data": {
                        "avatar": 2,
                        "x": 41.6,
                        "y": 40,
                        "important": False,
                    },
                },
                {"event": "position", "data": {"avatar": 1, "x": 41.6, "y": 40}},
                {
                    "event": "point",
                    "data": {
                        "avatar": 1,
                        "x": 41.6,
                        "y": 40,
                        "important": False,
                    },
                },
                {"event": "die", "data": {"avatar": 1, "killer": 2, "old": False}},
                {
                    "event": "score:round",
                    "data": {"avatar": 1, "score": 0, "roundScore": 0},
                },
            ],
            [
                {"event": "position", "player_id": "p2", "x": 78.4, "y": 20.0},
                {"event": "position", "player_id": "p1", "x": 41.6, "y": 40.0},
                {
                    "event": "point",
                    "player_id": "p1",
                    "x": 41.6,
                    "y": 40.0,
                    "important": False,
                },
                {"event": "position", "player_id": "p0", "x": 41.6, "y": 40.0},
                {
                    "event": "point",
                    "player_id": "p0",
                    "x": 41.6,
                    "y": 40.0,
                    "important": False,
                },
                {"event": "die", "player_id": "p0", "killer_id": "p1", "old": False},
                {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 0},
            ],
        ),
        (
            "source_body_same_frame_point_control_safe_step.json",
            [True, True, True],
            0,
            [0, 0, 0],
            [
                {"event": "position", "data": {"avatar": 3, "x": 78.4, "y": 20}},
                {"event": "position", "data": {"avatar": 2, "x": 41.6, "y": 40}},
                {"event": "position", "data": {"avatar": 1, "x": 41.6, "y": 40}},
            ],
            [
                {"event": "position", "player_id": "p2", "x": 78.4, "y": 20.0},
                {"event": "position", "player_id": "p1", "x": 41.6, "y": 40.0},
                {"event": "position", "player_id": "p0", "x": 41.6, "y": 40.0},
            ],
        ),
    ],
)
def test_js_scenario_runner_pins_same_frame_point_materialization(
    filename,
    expected_alive,
    expected_world_body_count,
    expected_trail_counts,
    expected_raw_events,
    expected_common_events,
):
    payload = _run_js_scenario(filename)
    frame = payload["trace"][0]
    avatars = sorted(frame["avatars"], key=lambda avatar: avatar["name"])
    common_step = project_common_trace(payload)["steps"][0]

    assert payload["comparison"]["python_target"] == "pending-source-body"
    assert payload["comparison"]["include_events"] is True
    assert frame["stepMs"] == 100
    assert frame["game"]["size"] == 95
    assert frame["game"]["worldBodyCount"] == expected_world_body_count
    assert frame["game"]["roundWinner"] is None
    assert [avatar["alive"] for avatar in avatars] == expected_alive
    assert [avatar["trailPointCount"] for avatar in avatars] == expected_trail_counts
    assert frame["events"] == expected_raw_events
    assert common_step["worldBodyCount"] == expected_world_body_count
    assert common_step["events"] == expected_common_events


@pytest.mark.parametrize(
    (
        "filename",
        "expected_positions",
        "expected_headings",
        "expected_step_ms",
        "expected_alive",
        "expected_world_body_count",
        "expected_body_state",
        "expected_events",
    ),
    [
        (
            "source_body_opponent_tangent_safe_step.json",
            [[20.0, 20.0], [70.0, 70.0], [80.0, 20.0]],
            [0.0, 0.0, 3.141593],
            0,
            [True, True, True],
            1,
            [
                {
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 0,
                    "bodyCount": 0,
                },
                {
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 1,
                    "bodyCount": 1,
                },
                {
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 0,
                    "bodyCount": 0,
                },
            ],
            [
                {"event": "position", "player_id": "p2", "x": 80.0, "y": 20.0},
                {"event": "position", "player_id": "p1", "x": 70.0, "y": 70.0},
                {"event": "position", "player_id": "p0", "x": 20.0, "y": 20.0},
            ],
        ),
        (
            "source_body_opponent_overlap_kills_step.json",
            [[20.0, 20.0], [70.0, 70.0], [80.0, 20.0]],
            [0.0, 0.0, 3.141593],
            0,
            [False, True, True],
            2,
            [
                {
                    "trailPointCount": 1,
                    "lastTrailPoint": [20.0, 20.0],
                    "bodyNum": 0,
                    "bodyCount": 1,
                },
                {
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 1,
                    "bodyCount": 1,
                },
                {
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 0,
                    "bodyCount": 0,
                },
            ],
            [
                {"event": "position", "player_id": "p2", "x": 80.0, "y": 20.0},
                {"event": "position", "player_id": "p1", "x": 70.0, "y": 70.0},
                {"event": "position", "player_id": "p0", "x": 20.0, "y": 20.0},
                {"event": "point", "player_id": "p0", "x": 20.0, "y": 20.0, "important": False},
                {"event": "die", "player_id": "p0", "killer_id": "p1", "old": False},
                {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 0},
            ],
        ),
        (
            "source_body_own_delta3_safe_step.json",
            [[20.0, 20.0], [70.0, 70.0], [80.0, 20.0]],
            [0.0, 0.0, 3.141593],
            0,
            [True, True, True],
            1,
            [
                {
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 3,
                    "bodyCount": 3,
                },
                {
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 0,
                    "bodyCount": 0,
                },
                {
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 0,
                    "bodyCount": 0,
                },
            ],
            [
                {"event": "position", "player_id": "p2", "x": 80.0, "y": 20.0},
                {"event": "position", "player_id": "p1", "x": 70.0, "y": 70.0},
                {"event": "position", "player_id": "p0", "x": 20.0, "y": 20.0},
            ],
        ),
        (
            "source_body_own_delta4_kills_step.json",
            [[20.0, 20.0], [70.0, 70.0], [80.0, 20.0]],
            [0.0, 0.0, 3.141593],
            0,
            [False, True, True],
            2,
            [
                {
                    "trailPointCount": 1,
                    "lastTrailPoint": [20.0, 20.0],
                    "bodyNum": 4,
                    "bodyCount": 5,
                },
                {
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 0,
                    "bodyCount": 0,
                },
                {
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 0,
                    "bodyCount": 0,
                },
            ],
            [
                {"event": "position", "player_id": "p2", "x": 80.0, "y": 20.0},
                {"event": "position", "player_id": "p1", "x": 70.0, "y": 70.0},
                {"event": "position", "player_id": "p0", "x": 20.0, "y": 20.0},
                {"event": "point", "player_id": "p0", "x": 20.0, "y": 20.0, "important": False},
                {"event": "die", "player_id": "p0", "killer_id": "p0", "old": False},
                {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 0},
            ],
        ),
        (
            "source_body_same_frame_point_kills_step.json",
            [[41.6, 40.0], [41.6, 40.0], [78.4, 20.0]],
            [0.0, 0.0, 3.141593],
            100,
            [False, True, True],
            2,
            [
                {
                    "trailPointCount": 1,
                    "lastTrailPoint": [41.6, 40.0],
                    "bodyNum": 0,
                    "bodyCount": 1,
                },
                {
                    "trailPointCount": 1,
                    "lastTrailPoint": [41.6, 40.0],
                    "bodyNum": 0,
                    "bodyCount": 1,
                },
                {
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 0,
                    "bodyCount": 0,
                },
            ],
            [
                {"event": "position", "player_id": "p2", "x": 78.4, "y": 20.0},
                {"event": "position", "player_id": "p1", "x": 41.6, "y": 40.0},
                {"event": "point", "player_id": "p1", "x": 41.6, "y": 40.0, "important": False},
                {"event": "position", "player_id": "p0", "x": 41.6, "y": 40.0},
                {"event": "point", "player_id": "p0", "x": 41.6, "y": 40.0, "important": False},
                {"event": "die", "player_id": "p0", "killer_id": "p1", "old": False},
                {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 0},
            ],
        ),
        (
            "source_body_same_frame_point_control_safe_step.json",
            [[41.6, 40.0], [41.6, 40.0], [78.4, 20.0]],
            [0.0, 0.0, 3.141593],
            100,
            [True, True, True],
            0,
            [
                {
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 0,
                    "bodyCount": 0,
                },
                {
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 0,
                    "bodyCount": 0,
                },
                {
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 0,
                    "bodyCount": 0,
                },
            ],
            [
                {"event": "position", "player_id": "p2", "x": 78.4, "y": 20.0},
                {"event": "position", "player_id": "p1", "x": 41.6, "y": 40.0},
                {"event": "position", "player_id": "p0", "x": 41.6, "y": 40.0},
            ],
        ),
        (
            "source_collision_death_point_kills_later_player_step.json",
            [[44.0, 44.0], [44.0, 44.0]],
            [3.141593, 0.0],
            100,
            [False, False],
            3,
            [
                {
                    "trailPointCount": 1,
                    "lastTrailPoint": [44.0, 44.0],
                    "bodyNum": 1,
                    "bodyCount": 2,
                },
                {
                    "trailPointCount": 1,
                    "lastTrailPoint": [44.0, 44.0],
                    "bodyNum": 0,
                    "bodyCount": 1,
                },
            ],
            [
                {"event": "position", "player_id": "p1", "x": 44.0, "y": 44.0},
                {"event": "point", "player_id": "p1", "x": 44.0, "y": 44.0, "important": False},
                {"event": "die", "player_id": "p1", "killer_id": "p0", "old": False},
                {"event": "score:round", "player_id": "p1", "score": 0, "roundScore": 0},
                {"event": "position", "player_id": "p0", "x": 44.0, "y": 44.0},
                {"event": "point", "player_id": "p0", "x": 44.0, "y": 44.0, "important": False},
                {"event": "die", "player_id": "p0", "killer_id": "p1", "old": False},
                {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 0},
                {"event": "score", "player_id": "p1", "score": 0, "roundScore": 0},
                {"event": "score", "player_id": "p0", "score": 0, "roundScore": 0},
                {"event": "round:end", "winner_id": None},
            ],
        ),
        (
            "source_collision_head_head_reverse_order_single_death_step.json",
            [[44.0, 44.0], [44.0, 44.0]],
            [0.0, 3.141593],
            100,
            [False, True],
            3,
            [
                {
                    "trailPointCount": 2,
                    "lastTrailPoint": [44.0, 44.0],
                    "bodyNum": 0,
                    "bodyCount": 2,
                },
                {
                    "trailPointCount": 1,
                    "lastTrailPoint": [44.0, 44.0],
                    "bodyNum": 0,
                    "bodyCount": 1,
                },
            ],
            [
                {"event": "position", "player_id": "p1", "x": 44.0, "y": 44.0},
                {"event": "point", "player_id": "p1", "x": 44.0, "y": 44.0, "important": False},
                {"event": "position", "player_id": "p0", "x": 44.0, "y": 44.0},
                {"event": "point", "player_id": "p0", "x": 44.0, "y": 44.0, "important": False},
                {"event": "point", "player_id": "p0", "x": 44.0, "y": 44.0, "important": False},
                {"event": "die", "player_id": "p0", "killer_id": "p1", "old": False},
                {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 0},
                {"event": "score:round", "player_id": "p1", "score": 0, "roundScore": 1},
                {"event": "score", "player_id": "p1", "score": 1, "roundScore": 1},
                {"event": "score", "player_id": "p0", "score": 0, "roundScore": 0},
                {"event": "round:end", "winner_id": "p1"},
            ],
        ),
    ],
)
def test_source_body_canary_scenarios_match_js_common_trace(
    filename,
    expected_positions,
    expected_headings,
    expected_step_ms,
    expected_alive,
    expected_world_body_count,
    expected_body_state,
    expected_events,
):
    if shutil.which("node") is None:
        pytest.skip("node executable is not available")

    js_payload = _run_js_scenario(filename)
    python_payload = run_source_body_canary_scenario(_SCENARIO_DIR / filename).to_payload()
    frame = python_payload["trace"]["frames"][1]
    common_step = project_common_trace(python_payload)["steps"][0]

    assert python_payload["runner"] == "curvytron-v1-python-source-body-canary-runner"
    assert python_payload["source_fidelity"] is True
    assert frame["positions"] == expected_positions
    assert frame["headings"] == expected_headings
    assert frame["alive"] == expected_alive
    assert frame["worldBodyCount"] == expected_world_body_count
    assert frame["events"] == expected_events
    assert common_step["step_ms"] == expected_step_ms
    assert common_step["worldBodyCount"] == expected_world_body_count
    assert [
        {
            "trailPointCount": player["trailPointCount"],
            "lastTrailPoint": player["lastTrailPoint"],
            "bodyNum": player["bodyNum"],
            "bodyCount": player["bodyCount"],
        }
        for player in common_step["players"]
    ] == expected_body_state
    assert diff_payload(js_payload, python_payload, common_trace=True)["status"] == "pass"


def test_js_scenario_runner_pins_same_frame_point_materialization_kill():
    payload = _run_js_scenario("source_body_same_frame_point_kills_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert payload["comparison"]["python_target"] == "pending-source-body"
    assert frame["stepMs"] == 100
    assert frame["game"]["worldBodyCount"] == 2
    assert frame["game"]["deathCount"] == 1
    assert [avatars[name]["alive"] for name in ("p0", "p1", "p2")] == [
        False,
        True,
        True,
    ]
    assert [avatars[name]["trailPointCount"] for name in ("p0", "p1", "p2")] == [
        1,
        1,
        0,
    ]
    assert frame["events"] == [
        {"event": "position", "data": {"avatar": 3, "x": 78.4, "y": 20}},
        {"event": "position", "data": {"avatar": 2, "x": 41.6, "y": 40}},
        {
            "event": "point",
            "data": {"avatar": 2, "x": 41.6, "y": 40, "important": False},
        },
        {"event": "position", "data": {"avatar": 1, "x": 41.6, "y": 40}},
        {
            "event": "point",
            "data": {"avatar": 1, "x": 41.6, "y": 40, "important": False},
        },
        {"event": "die", "data": {"avatar": 1, "killer": 2, "old": False}},
        {
            "event": "score:round",
            "data": {"avatar": 1, "score": 0, "roundScore": 0},
        },
    ]
    assert project_common_trace(payload)["steps"][0]["events"] == [
        {"event": "position", "player_id": "p2", "x": 78.4, "y": 20},
        {"event": "position", "player_id": "p1", "x": 41.6, "y": 40},
        {
            "event": "point",
            "player_id": "p1",
            "x": 41.6,
            "y": 40,
            "important": False,
        },
        {"event": "position", "player_id": "p0", "x": 41.6, "y": 40},
        {
            "event": "point",
            "player_id": "p0",
            "x": 41.6,
            "y": 40,
            "important": False,
        },
        {"event": "die", "player_id": "p0", "killer_id": "p1", "old": False},
        {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 0},
    ]


def test_js_scenario_runner_pins_same_frame_point_materialization_control():
    payload = _run_js_scenario("source_body_same_frame_point_control_safe_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert payload["comparison"]["python_target"] == "pending-source-body"
    assert frame["stepMs"] == 100
    assert frame["game"]["worldBodyCount"] == 0
    assert frame["game"]["deathCount"] == 0
    assert [avatars[name]["alive"] for name in ("p0", "p1", "p2")] == [
        True,
        True,
        True,
    ]
    assert all(avatar["trailPointCount"] == 0 for avatar in avatars.values())
    assert frame["events"] == [
        {"event": "position", "data": {"avatar": 3, "x": 78.4, "y": 20}},
        {"event": "position", "data": {"avatar": 2, "x": 41.6, "y": 40}},
        {"event": "position", "data": {"avatar": 1, "x": 41.6, "y": 40}},
    ]
    assert project_common_trace(payload)["steps"][0]["events"] == [
        {"event": "position", "player_id": "p2", "x": 78.4, "y": 20},
        {"event": "position", "player_id": "p1", "x": 41.6, "y": 40},
        {"event": "position", "player_id": "p0", "x": 41.6, "y": 40},
    ]


def test_js_scenario_runner_pins_collision_order_death_point_kills_later_player():
    payload = _run_js_scenario("source_collision_death_point_kills_later_player_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}
    common_step = project_common_trace(payload)["steps"][0]

    assert payload["comparison"]["python_target"] == "source-body-canary"
    assert frame["stepMs"] == 100
    assert frame["game"]["size"] == 88
    assert frame["game"]["inRound"] is False
    assert frame["game"]["deathCount"] == 2
    assert frame["game"]["deaths"] == [2, 1]
    assert frame["game"]["roundWinner"] is None
    assert frame["game"]["worldBodyCount"] == 3
    assert [avatars[name]["alive"] for name in ("p0", "p1")] == [False, False]
    assert [avatars[name]["score"] for name in ("p0", "p1")] == [0, 0]
    assert [avatars[name]["roundScore"] for name in ("p0", "p1")] == [0, 0]
    assert [
        event
        for event in common_step["events"]
        if event["event"] in {"die", "round:end"}
    ] == [
        {"event": "die", "player_id": "p1", "killer_id": "p0", "old": False},
        {"event": "die", "player_id": "p0", "killer_id": "p1", "old": False},
        {"event": "round:end", "winner_id": None},
    ]


def test_js_scenario_runner_pins_collision_order_head_head_single_death():
    payload = _run_js_scenario("source_collision_head_head_reverse_order_single_death_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}
    common_step = project_common_trace(payload)["steps"][0]

    assert payload["comparison"]["python_target"] == "source-body-canary"
    assert frame["stepMs"] == 100
    assert frame["game"]["size"] == 88
    assert frame["game"]["inRound"] is False
    assert frame["game"]["deathCount"] == 1
    assert frame["game"]["deaths"] == [1]
    assert frame["game"]["roundWinner"] == 2
    assert frame["game"]["worldBodyCount"] == 3
    assert [avatars[name]["alive"] for name in ("p0", "p1")] == [False, True]
    assert [avatars[name]["score"] for name in ("p0", "p1")] == [0, 1]
    assert [avatars[name]["roundScore"] for name in ("p0", "p1")] == [0, 0]
    assert [
        event
        for event in common_step["events"]
        if event["event"] in {"die", "score:round", "round:end"}
    ] == [
        {"event": "die", "player_id": "p0", "killer_id": "p1", "old": False},
        {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 0},
        {"event": "score:round", "player_id": "p1", "score": 0, "roundScore": 1},
        {"event": "round:end", "winner_id": "p1"},
    ]


def test_js_scenario_runner_forces_print_manager_and_trail_state_directly(tmp_path):
    scenario_path = tmp_path / "forced_print_manager_state.json"
    scenario_path.write_text(
        json.dumps(
            {
                "scenario_id": "forced_print_manager_state",
                "ruleset_id": "curvytron-v1-reference",
                "player_count": 1,
                "source_setup": {
                    "game": {"started": True, "in_round": True, "world_active": True}
                },
                "players": [
                    {
                        "id": "p0",
                        "avatar_id": 1,
                        "name": "p0",
                        "initial": {
                            "x": 40,
                            "y": 40,
                            "angle_rad": 0,
                            "printing": True,
                            "trail": {"points": [[40, 40]]},
                            "print_manager": {
                                "active": False,
                                "distance": 7,
                                "last_x": 1,
                                "last_y": 2,
                            },
                        },
                    }
                ],
                "steps": [{"tick": 0, "step_ms": 0, "moves": {"p0": 0}}],
                "comparison": {"include_events": True},
            }
        ),
        encoding="utf-8",
    )

    payload = _run_js_scenario(scenario_path)
    frame = payload["trace"][0]
    avatar = frame["avatars"][0]

    assert frame["game"]["worldBodyCount"] == 0
    assert avatar["printing"] is True
    assert avatar["trailPointCount"] == 1
    assert avatar["lastTrailPoint"] == [40, 40]
    assert avatar["printManager"] == {
        "active": False,
        "distance": 7,
        "lastX": 1,
        "lastY": 2,
    }


@pytest.mark.parametrize(
    (
        "filename",
        "expected_printing",
        "expected_step_ms",
        "expected_x",
        "expected_y",
        "expected_print_manager",
        "expected_world_body_count",
        "expected_trail_point_count",
        "expected_last_trail_point",
        "expected_body_count",
        "expected_events",
    ),
    [
        (
            "source_print_manager_print_to_hole_step.json",
            False,
            100,
            21.6,
            40,
            {"active": True, "distance": 5.25, "lastX": 21.6, "lastY": 40},
            1,
            0,
            None,
            1,
            [
                {"event": "position", "data": {"avatar": 1, "x": 21.6, "y": 40}},
                {
                    "event": "point",
                    "data": {"avatar": 1, "x": 21.6, "y": 40, "important": True},
                },
                {
                    "event": "property",
                    "data": {"avatar": 1, "property": "printing", "value": False},
                },
            ],
        ),
        (
            "source_print_manager_hole_to_print_step.json",
            True,
            100,
            21.6,
            40,
            {"active": True, "distance": 39, "lastX": 21.6, "lastY": 40},
            1,
            1,
            [21.6, 40],
            1,
            [
                {"event": "position", "data": {"avatar": 1, "x": 21.6, "y": 40}},
                {
                    "event": "point",
                    "data": {"avatar": 1, "x": 21.6, "y": 40, "important": True},
                },
                {
                    "event": "property",
                    "data": {"avatar": 1, "property": "printing", "value": True},
                },
            ],
        ),
        (
            "source_print_manager_exact_zero_toggle_step.json",
            True,
            0,
            20,
            40,
            {"active": True, "distance": 39, "lastX": 20, "lastY": 40},
            1,
            1,
            [20, 40],
            1,
            [
                {"event": "position", "data": {"avatar": 1, "x": 20, "y": 40}},
                {
                    "event": "point",
                    "data": {"avatar": 1, "x": 20, "y": 40, "important": True},
                },
                {
                    "event": "property",
                    "data": {"avatar": 1, "property": "printing", "value": True},
                },
            ],
        ),
        (
            "source_print_manager_no_toggle_control_step.json",
            False,
            100,
            21.6,
            40,
            {"active": True, "distance": 8.4, "lastX": 21.6, "lastY": 40},
            0,
            0,
            None,
            0,
            [
                {"event": "position", "data": {"avatar": 1, "x": 21.6, "y": 40}},
            ],
        ),
    ],
)
def test_js_scenario_runner_pins_print_manager_toggle_steps(
    filename,
    expected_printing,
    expected_step_ms,
    expected_x,
    expected_y,
    expected_print_manager,
    expected_world_body_count,
    expected_trail_point_count,
    expected_last_trail_point,
    expected_body_count,
    expected_events,
):
    payload = _run_js_scenario(filename)
    frame = payload["trace"][0]
    avatar = frame["avatars"][0]

    assert payload["comparison"]["python_target"] == "source-print-manager-canary"
    assert payload["comparison"]["include_events"] is True
    assert frame["stepMs"] == expected_step_ms
    assert frame["game"]["size"] == 80
    assert frame["game"]["deathCount"] == 0
    assert frame["game"]["worldBodyCount"] == expected_world_body_count
    assert avatar["name"] == "p0"
    assert avatar["x"] == expected_x
    assert avatar["y"] == expected_y
    assert avatar["alive"] is True
    assert avatar["printing"] is expected_printing
    assert avatar["printManager"] == expected_print_manager
    assert avatar["trailPointCount"] == expected_trail_point_count
    assert avatar["lastTrailPoint"] == expected_last_trail_point
    assert avatar["bodyNum"] == 0
    assert avatar["bodyCount"] == expected_body_count
    assert frame["events"] == expected_events


def test_js_scenario_runner_pins_print_manager_delayed_start_timer():
    payload = _run_js_scenario("source_print_manager_delayed_start_timer_step.json")
    before_start, after_start = payload["trace"]
    before_avatar = before_start["avatars"][0]
    after_avatar = after_start["avatars"][0]
    common_steps = project_common_trace(payload)["steps"]

    assert payload["comparison"]["python_target"] == "source-print-manager-canary"
    assert payload["comparison"]["include_events"] is True
    assert [frame["stepMs"] for frame in payload["trace"]] == [0, 0]
    assert before_start["game"]["worldBodyCount"] == 0
    assert before_avatar["printing"] is False
    assert before_avatar["trailPointCount"] == 0
    assert before_avatar["lastTrailPoint"] is None
    assert before_avatar["bodyNum"] == 0
    assert before_avatar["bodyCount"] == 0
    assert before_avatar["printManager"] == {
        "active": False,
        "distance": 0,
        "lastX": 0,
        "lastY": 0,
    }
    assert before_start["events"] == [
        {"event": "position", "data": {"avatar": 1, "x": 20, "y": 40}},
    ]

    assert after_start["game"]["worldBodyCount"] == 1
    assert after_avatar["printing"] is True
    assert after_avatar["trailPointCount"] == 1
    assert after_avatar["lastTrailPoint"] == [20, 40]
    assert after_avatar["bodyNum"] == 1
    assert after_avatar["bodyCount"] == 1
    assert after_avatar["printManager"] == {
        "active": True,
        "distance": 39,
        "lastX": 20,
        "lastY": 40,
    }
    assert after_start["events"] == [
        {
            "event": "point",
            "data": {"avatar": 1, "x": 20, "y": 40, "important": True},
        },
        {
            "event": "property",
            "data": {"avatar": 1, "property": "printing", "value": True},
        },
        {"event": "position", "data": {"avatar": 1, "x": 20, "y": 40}},
    ]
    assert common_steps[0]["players"][0]["printManager"] == {
        "active": False,
        "distance": 0,
        "lastX": 0,
        "lastY": 0,
    }
    assert common_steps[1]["events"] == [
        {
            "event": "point",
            "player_id": "p0",
            "x": 20,
            "y": 40,
            "important": True,
        },
        {
            "event": "property",
            "player_id": "p0",
            "property": "printing",
            "value": True,
        },
        {"event": "position", "player_id": "p0", "x": 20, "y": 40},
    ]


def test_js_scenario_runner_pins_print_manager_random_tape_call_order():
    payload = _run_js_scenario("source_print_manager_random_call_order_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}
    common_step = project_common_trace(payload)["steps"][0]

    assert payload["comparison"]["python_target"] == "source-print-manager-canary"
    assert payload["randomCalls"] == [
        {"index": 0, "value": 0.1},
        {"index": 1, "value": 0.9},
    ]
    assert frame["stepMs"] == 0
    assert frame["game"]["size"] == 88
    assert frame["game"]["worldBodyCount"] == 2
    assert avatars["p1"]["printManager"] == {
        "active": True,
        "distance": 22.2,
        "lastX": 60,
        "lastY": 40,
    }
    assert avatars["p0"]["printManager"] == {
        "active": True,
        "distance": 55.8,
        "lastX": 20,
        "lastY": 40,
    }
    assert frame["events"] == [
        {"event": "position", "data": {"avatar": 2, "x": 60, "y": 40}},
        {
            "event": "point",
            "data": {"avatar": 2, "x": 60, "y": 40, "important": True},
        },
        {
            "event": "property",
            "data": {"avatar": 2, "property": "printing", "value": True},
        },
        {"event": "position", "data": {"avatar": 1, "x": 20, "y": 40}},
        {
            "event": "point",
            "data": {"avatar": 1, "x": 20, "y": 40, "important": True},
        },
        {
            "event": "property",
            "data": {"avatar": 1, "property": "printing", "value": True},
        },
    ]
    assert common_step["events"] == [
        {"event": "position", "player_id": "p1", "x": 60, "y": 40},
        {
            "event": "point",
            "player_id": "p1",
            "x": 60,
            "y": 40,
            "important": True,
        },
        {
            "event": "property",
            "player_id": "p1",
            "property": "printing",
            "value": True,
        },
        {"event": "position", "player_id": "p0", "x": 20, "y": 40},
        {
            "event": "point",
            "player_id": "p0",
            "x": 20,
            "y": 40,
            "important": True,
        },
        {
            "event": "property",
            "player_id": "p0",
            "property": "printing",
            "value": True,
        },
    ]


def test_js_scenario_runner_pins_print_manager_random_cadence_multistep():
    payload = _run_js_scenario("source_print_manager_random_cadence_multistep.json")
    frames = payload["trace"]
    avatars = [frame["avatars"][0] for frame in frames]
    common_steps = project_common_trace(payload)["steps"]

    assert payload["comparison"]["python_target"] == "source-print-manager-canary"
    assert payload["randomCalls"] == [
        {"index": 0, "value": 0},
        {"index": 1, "value": 0.5},
        {"index": 2, "value": 0.25},
    ]
    assert [frame["stepMs"] for frame in frames] == [1000, 1000, 1000, 1000]
    assert [frame["game"]["worldBodyCount"] for frame in frames] == [1, 2, 4, 5]
    assert [avatar["x"] for avatar in avatars] == [26, 42, 58, 74]
    assert [avatar["printing"] for avatar in avatars] == [True, True, False, True]
    assert [avatar["trailPointCount"] for avatar in avatars] == [1, 2, 0, 1]
    assert [avatar["lastTrailPoint"] for avatar in avatars] == [
        [26, 40],
        [42, 40],
        None,
        [74, 40],
    ]
    assert [avatar["bodyCount"] for avatar in avatars] == [1, 2, 4, 5]
    assert [avatar["printManager"] for avatar in avatars] == [
        {"active": True, "distance": 18, "lastX": 26, "lastY": 40},
        {"active": True, "distance": 2, "lastX": 42, "lastY": 40},
        {"active": True, "distance": 5.25, "lastX": 58, "lastY": 40},
        {"active": True, "distance": 28.5, "lastX": 74, "lastY": 40},
    ]
    assert frames[2]["events"] == [
        {"event": "position", "data": {"avatar": 1, "x": 58, "y": 40}},
        {
            "event": "point",
            "data": {"avatar": 1, "x": 58, "y": 40, "important": False},
        },
        {
            "event": "point",
            "data": {"avatar": 1, "x": 58, "y": 40, "important": True},
        },
        {
            "event": "property",
            "data": {"avatar": 1, "property": "printing", "value": False},
        },
    ]
    assert common_steps[3]["events"] == [
        {"event": "position", "player_id": "p0", "x": 74, "y": 40},
        {
            "event": "point",
            "player_id": "p0",
            "x": 74,
            "y": 40,
            "important": True,
        },
        {
            "event": "property",
            "player_id": "p0",
            "property": "printing",
            "value": True,
        },
    ]


@pytest.mark.parametrize(
    ("filename", "expected_random_calls"),
    [
        (
            "source_print_manager_random_call_order_step.json",
            [{"index": 0, "value": 0.1}, {"index": 1, "value": 0.9}],
        ),
        (
            "source_print_manager_random_cadence_multistep.json",
            [
                {"index": 0, "value": 0},
                {"index": 1, "value": 0.5},
                {"index": 2, "value": 0.25},
            ],
        ),
    ],
)
def test_source_print_manager_random_runner_matches_js_common_trace(
    filename,
    expected_random_calls,
):
    js_payload = _run_js_scenario(filename)
    python_payload = run_source_print_manager_scenario(_SCENARIO_DIR / filename).to_payload()

    assert python_payload["randomCalls"] == expected_random_calls
    assert diff_payload(js_payload, python_payload, common_trace=True)["status"] == "pass"


def test_js_scenario_runner_pins_print_manager_active_stop_on_death():
    payload = _run_js_scenario("source_print_manager_active_stop_on_death_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}
    common_step = project_common_trace(payload)["steps"][0]

    assert payload["comparison"]["python_target"] == "source-print-manager-canary"
    assert payload["comparison"]["include_events"] is True
    assert frame["stepMs"] == 100
    assert frame["game"]["size"] == 95
    assert frame["game"]["inRound"] is True
    assert frame["game"]["deathCount"] == 1
    assert frame["game"]["deaths"] == [1]
    assert frame["game"]["worldBodyCount"] == 2
    assert avatars["p0"]["alive"] is False
    assert avatars["p0"]["printing"] is False
    assert avatars["p0"]["trailPointCount"] == 0
    assert avatars["p0"]["lastTrailPoint"] is None
    assert avatars["p0"]["bodyNum"] == 0
    assert avatars["p0"]["bodyCount"] == 2
    assert avatars["p0"]["printManager"] == {
        "active": False,
        "distance": 0,
        "lastX": 0,
        "lastY": 0,
    }
    assert [avatars[name]["alive"] for name in ("p1", "p2")] == [True, True]
    assert frame["events"] == [
        {"event": "position", "data": {"avatar": 3, "x": 49.1, "y": 30}},
        {"event": "position", "data": {"avatar": 2, "x": 49.1, "y": 47.5}},
        {"event": "position", "data": {"avatar": 1, "x": 95.5, "y": 47.5}},
        {
            "event": "point",
            "data": {"avatar": 1, "x": 95.5, "y": 47.5, "important": False},
        },
        {
            "event": "point",
            "data": {"avatar": 1, "x": 95.5, "y": 47.5, "important": True},
        },
        {
            "event": "property",
            "data": {"avatar": 1, "property": "printing", "value": False},
        },
        {"event": "die", "data": {"avatar": 1, "killer": None, "old": None}},
        {
            "event": "score:round",
            "data": {"avatar": 1, "score": 0, "roundScore": 0},
        },
    ]
    assert common_step["events"] == [
        {"event": "position", "player_id": "p2", "x": 49.1, "y": 30},
        {"event": "position", "player_id": "p1", "x": 49.1, "y": 47.5},
        {"event": "position", "player_id": "p0", "x": 95.5, "y": 47.5},
        {
            "event": "point",
            "player_id": "p0",
            "x": 95.5,
            "y": 47.5,
            "important": False,
        },
        {
            "event": "point",
            "player_id": "p0",
            "x": 95.5,
            "y": 47.5,
            "important": True,
        },
        {
            "event": "property",
            "player_id": "p0",
            "property": "printing",
            "value": False,
        },
        {"event": "die", "player_id": "p0", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 0},
    ]


def test_js_scenario_runner_pins_print_manager_active_hole_stop_on_death():
    payload = _run_js_scenario("source_print_manager_active_hole_stop_on_death_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}
    common_step = project_common_trace(payload)["steps"][0]

    assert payload["comparison"]["python_target"] == "source-print-manager-canary"
    assert payload["comparison"]["include_events"] is True
    assert frame["stepMs"] == 100
    assert frame["game"]["size"] == 95
    assert frame["game"]["inRound"] is True
    assert frame["game"]["deathCount"] == 1
    assert frame["game"]["deaths"] == [1]
    assert frame["game"]["worldBodyCount"] == 1
    assert avatars["p0"]["alive"] is False
    assert avatars["p0"]["printing"] is False
    assert avatars["p0"]["trailPointCount"] == 1
    assert avatars["p0"]["lastTrailPoint"] == [95.5, 47.5]
    assert avatars["p0"]["bodyNum"] == 0
    assert avatars["p0"]["bodyCount"] == 1
    assert avatars["p0"]["printManager"] == {
        "active": False,
        "distance": 0,
        "lastX": 0,
        "lastY": 0,
    }
    assert [avatars[name]["alive"] for name in ("p1", "p2")] == [True, True]
    assert frame["events"] == [
        {"event": "position", "data": {"avatar": 3, "x": 49.1, "y": 30}},
        {"event": "position", "data": {"avatar": 2, "x": 49.1, "y": 47.5}},
        {"event": "position", "data": {"avatar": 1, "x": 95.5, "y": 47.5}},
        {
            "event": "point",
            "data": {"avatar": 1, "x": 95.5, "y": 47.5, "important": False},
        },
        {
            "event": "property",
            "data": {"avatar": 1, "property": "printing", "value": False},
        },
        {"event": "die", "data": {"avatar": 1, "killer": None, "old": None}},
        {
            "event": "score:round",
            "data": {"avatar": 1, "score": 0, "roundScore": 0},
        },
    ]
    assert common_step["events"] == [
        {"event": "position", "player_id": "p2", "x": 49.1, "y": 30},
        {"event": "position", "player_id": "p1", "x": 49.1, "y": 47.5},
        {"event": "position", "player_id": "p0", "x": 95.5, "y": 47.5},
        {
            "event": "point",
            "player_id": "p0",
            "x": 95.5,
            "y": 47.5,
            "important": False,
        },
        {
            "event": "property",
            "player_id": "p0",
            "property": "printing",
            "value": False,
        },
        {"event": "die", "player_id": "p0", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 0},
    ]


def test_js_scenario_runner_pins_print_manager_body_collision_stop_on_death():
    payload = _run_js_scenario("source_print_manager_body_collision_stop_on_death_step.json")
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}
    common_step = project_common_trace(payload)["steps"][0]

    assert payload["comparison"]["python_target"] == "source-print-manager-canary"
    assert payload["comparison"]["include_events"] is True
    assert frame["stepMs"] == 0
    assert frame["game"]["size"] == 95
    assert frame["game"]["deathCount"] == 1
    assert frame["game"]["deaths"] == [1]
    assert frame["game"]["worldBodyCount"] == 3
    assert avatars["p0"]["alive"] is False
    assert avatars["p0"]["printing"] is False
    assert avatars["p0"]["trailPointCount"] == 0
    assert avatars["p0"]["lastTrailPoint"] is None
    assert avatars["p0"]["bodyNum"] == 0
    assert avatars["p0"]["bodyCount"] == 2
    assert avatars["p0"]["printManager"] == {
        "active": False,
        "distance": 0,
        "lastX": 0,
        "lastY": 0,
    }
    assert avatars["p1"]["bodyNum"] == 1
    assert avatars["p1"]["bodyCount"] == 1
    assert [avatars[name]["alive"] for name in ("p1", "p2")] == [True, True]
    assert frame["events"] == [
        {"event": "position", "data": {"avatar": 3, "x": 80, "y": 20}},
        {"event": "position", "data": {"avatar": 2, "x": 70, "y": 70}},
        {"event": "position", "data": {"avatar": 1, "x": 20, "y": 20}},
        {
            "event": "point",
            "data": {"avatar": 1, "x": 20, "y": 20, "important": False},
        },
        {
            "event": "point",
            "data": {"avatar": 1, "x": 20, "y": 20, "important": True},
        },
        {
            "event": "property",
            "data": {"avatar": 1, "property": "printing", "value": False},
        },
        {"event": "die", "data": {"avatar": 1, "killer": 2, "old": False}},
        {
            "event": "score:round",
            "data": {"avatar": 1, "score": 0, "roundScore": 0},
        },
    ]
    assert common_step["events"] == [
        {"event": "position", "player_id": "p2", "x": 80, "y": 20},
        {"event": "position", "player_id": "p1", "x": 70, "y": 70},
        {"event": "position", "player_id": "p0", "x": 20, "y": 20},
        {
            "event": "point",
            "player_id": "p0",
            "x": 20,
            "y": 20,
            "important": False,
        },
        {
            "event": "point",
            "player_id": "p0",
            "x": 20,
            "y": 20,
            "important": True,
        },
        {
            "event": "property",
            "player_id": "p0",
            "property": "printing",
            "value": False,
        },
        {"event": "die", "player_id": "p0", "killer_id": "p1", "old": False},
        {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 0},
    ]


@pytest.mark.parametrize(
    "filename",
    [
        "source_print_manager_print_to_hole_step.json",
        "source_print_manager_hole_to_print_step.json",
        "source_print_manager_exact_zero_toggle_step.json",
        "source_print_manager_no_toggle_control_step.json",
        "source_print_manager_delayed_start_timer_step.json",
        "source_print_manager_active_stop_on_death_step.json",
        "source_print_manager_active_hole_stop_on_death_step.json",
        "source_print_manager_body_collision_stop_on_death_step.json",
        "source_print_manager_random_cadence_multistep.json",
    ],
)
def test_source_print_manager_runner_matches_js_common_trace(filename):
    js_payload = _run_js_scenario(filename)
    python_payload = run_source_print_manager_scenario(_SCENARIO_DIR / filename).to_payload()

    assert python_payload["runner"] == "curvytron-v1-python-source-print-manager-runner"
    assert python_payload["source_fidelity_scope"] == (
        "movement plus deterministic PrintManager toggle/exact-zero, delayed start, "
        "random-tape call order/cadence, and wall/body death-stop state/events only"
    )
    assert diff_payload(js_payload, python_payload, common_trace=True)["status"] == "pass"


@pytest.mark.parametrize(
    (
        "filename",
        "expected_step_ms",
        "expected_x",
        "expected_world_body_count",
        "expected_trail_point_count",
        "expected_last_trail_point",
        "expected_body_count",
        "expected_events",
    ),
    [
        (
            "source_trail_normal_point_step.json",
            100,
            21.6,
            1,
            1,
            [21.6, 40],
            1,
            [
                {"event": "position", "data": {"avatar": 1, "x": 21.6, "y": 40}},
                {
                    "event": "point",
                    "data": {"avatar": 1, "x": 21.6, "y": 40, "important": False},
                },
            ],
        ),
        (
            "source_trail_no_point_below_radius_step.json",
            25,
            20.4,
            0,
            0,
            None,
            0,
            [
                {"event": "position", "data": {"avatar": 1, "x": 20.4, "y": 40}},
            ],
        ),
    ],
)
def test_js_scenario_runner_pins_trail_cadence_steps(
    filename,
    expected_step_ms,
    expected_x,
    expected_world_body_count,
    expected_trail_point_count,
    expected_last_trail_point,
    expected_body_count,
    expected_events,
):
    payload = _run_js_scenario(filename)
    frame = payload["trace"][0]
    avatar = frame["avatars"][0]

    assert payload["comparison"]["python_target"] == "pending-source-trail-cadence"
    assert payload["comparison"]["include_events"] is True
    assert frame["stepMs"] == expected_step_ms
    assert frame["game"]["size"] == 80
    assert frame["game"]["deathCount"] == 0
    assert frame["game"]["worldBodyCount"] == expected_world_body_count
    assert avatar["name"] == "p0"
    assert avatar["x"] == expected_x
    assert avatar["y"] == 40
    assert avatar["alive"] is True
    assert avatar["printing"] is True
    assert avatar["printManager"] == {
        "active": False,
        "distance": 0,
        "lastX": 20,
        "lastY": 40,
    }
    assert avatar["trailPointCount"] == expected_trail_point_count
    assert avatar["lastTrailPoint"] == expected_last_trail_point
    assert avatar["bodyNum"] == 0
    assert avatar["bodyCount"] == expected_body_count
    assert frame["events"] == expected_events


@pytest.mark.parametrize(
    (
        "filename",
        "expected_game",
        "expected_avatars",
        "expected_events",
    ),
    [
        (
            "source_trail_gap_hole_space_safe_step.json",
            {
                "deathCount": 0,
                "deaths": [],
                "worldBodyCount": 1,
            },
            {
                "p0": {
                    "x": 41.6,
                    "y": 40,
                    "alive": True,
                    "printing": False,
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 0,
                    "bodyCount": 0,
                    "printManager": {
                        "active": False,
                        "distance": 0,
                        "lastX": 0,
                        "lastY": 0,
                    },
                },
                "p1": {
                    "x": 41.6,
                    "y": 40,
                    "alive": True,
                    "printing": False,
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 1,
                    "bodyCount": 1,
                    "printManager": {
                        "active": True,
                        "distance": 8.4,
                        "lastX": 41.6,
                        "lastY": 40,
                    },
                },
                "p2": {
                    "x": 78.4,
                    "y": 20,
                    "alive": True,
                    "printing": False,
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 0,
                    "bodyCount": 0,
                    "printManager": {
                        "active": False,
                        "distance": 0,
                        "lastX": 0,
                        "lastY": 0,
                    },
                },
            },
            [
                {"event": "position", "data": {"avatar": 3, "x": 78.4, "y": 20}},
                {"event": "position", "data": {"avatar": 2, "x": 41.6, "y": 40}},
                {"event": "position", "data": {"avatar": 1, "x": 41.6, "y": 40}},
            ],
        ),
        (
            "source_trail_gap_stored_body_still_kills_step.json",
            {
                "deathCount": 1,
                "deaths": [1],
                "worldBodyCount": 2,
            },
            {
                "p0": {
                    "x": 41.6,
                    "y": 40,
                    "alive": False,
                    "printing": False,
                    "trailPointCount": 1,
                    "lastTrailPoint": [41.6, 40],
                    "bodyNum": 0,
                    "bodyCount": 1,
                    "printManager": {
                        "active": False,
                        "distance": 0,
                        "lastX": 0,
                        "lastY": 0,
                    },
                },
                "p1": {
                    "x": 41.6,
                    "y": 40,
                    "alive": True,
                    "printing": False,
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 1,
                    "bodyCount": 1,
                    "printManager": {
                        "active": True,
                        "distance": 8.4,
                        "lastX": 41.6,
                        "lastY": 40,
                    },
                },
                "p2": {
                    "x": 78.4,
                    "y": 20,
                    "alive": True,
                    "printing": False,
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 0,
                    "bodyCount": 0,
                    "printManager": {
                        "active": False,
                        "distance": 0,
                        "lastX": 0,
                        "lastY": 0,
                    },
                },
            },
            [
                {"event": "position", "data": {"avatar": 3, "x": 78.4, "y": 20}},
                {"event": "position", "data": {"avatar": 2, "x": 41.6, "y": 40}},
                {"event": "position", "data": {"avatar": 1, "x": 41.6, "y": 40}},
                {
                    "event": "point",
                    "data": {"avatar": 1, "x": 41.6, "y": 40, "important": False},
                },
                {"event": "die", "data": {"avatar": 1, "killer": 2, "old": False}},
                {
                    "event": "score:round",
                    "data": {"avatar": 1, "score": 0, "roundScore": 0},
                },
            ],
        ),
        (
            "source_trail_gap_print_to_hole_boundary_kills_step.json",
            {
                "deathCount": 1,
                "deaths": [1],
                "worldBodyCount": 2,
            },
            {
                "p0": {
                    "x": 41.6,
                    "y": 40,
                    "alive": False,
                    "printing": False,
                    "trailPointCount": 1,
                    "lastTrailPoint": [41.6, 40],
                    "bodyNum": 0,
                    "bodyCount": 1,
                    "printManager": {
                        "active": False,
                        "distance": 0,
                        "lastX": 0,
                        "lastY": 0,
                    },
                },
                "p1": {
                    "x": 41.6,
                    "y": 40,
                    "alive": True,
                    "printing": False,
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 0,
                    "bodyCount": 1,
                    "printManager": {
                        "active": True,
                        "distance": 5.25,
                        "lastX": 41.6,
                        "lastY": 40,
                    },
                },
                "p2": {
                    "x": 78.4,
                    "y": 20,
                    "alive": True,
                    "printing": False,
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 0,
                    "bodyCount": 0,
                    "printManager": {
                        "active": False,
                        "distance": 0,
                        "lastX": 0,
                        "lastY": 0,
                    },
                },
            },
            [
                {"event": "position", "data": {"avatar": 3, "x": 78.4, "y": 20}},
                {"event": "position", "data": {"avatar": 2, "x": 41.6, "y": 40}},
                {
                    "event": "point",
                    "data": {"avatar": 2, "x": 41.6, "y": 40, "important": True},
                },
                {
                    "event": "property",
                    "data": {"avatar": 2, "property": "printing", "value": False},
                },
                {"event": "position", "data": {"avatar": 1, "x": 41.6, "y": 40}},
                {
                    "event": "point",
                    "data": {"avatar": 1, "x": 41.6, "y": 40, "important": False},
                },
                {"event": "die", "data": {"avatar": 1, "killer": 2, "old": False}},
                {
                    "event": "score:round",
                    "data": {"avatar": 1, "score": 0, "roundScore": 0},
                },
            ],
        ),
        (
            "source_trail_gap_hole_to_print_boundary_kills_step.json",
            {
                "deathCount": 1,
                "deaths": [1],
                "worldBodyCount": 2,
            },
            {
                "p0": {
                    "x": 41.6,
                    "y": 40,
                    "alive": False,
                    "printing": False,
                    "trailPointCount": 1,
                    "lastTrailPoint": [41.6, 40],
                    "bodyNum": 0,
                    "bodyCount": 1,
                    "printManager": {
                        "active": False,
                        "distance": 0,
                        "lastX": 0,
                        "lastY": 0,
                    },
                },
                "p1": {
                    "x": 41.6,
                    "y": 40,
                    "alive": True,
                    "printing": True,
                    "trailPointCount": 1,
                    "lastTrailPoint": [41.6, 40],
                    "bodyNum": 0,
                    "bodyCount": 1,
                    "printManager": {
                        "active": True,
                        "distance": 39,
                        "lastX": 41.6,
                        "lastY": 40,
                    },
                },
                "p2": {
                    "x": 78.4,
                    "y": 20,
                    "alive": True,
                    "printing": False,
                    "trailPointCount": 0,
                    "lastTrailPoint": None,
                    "bodyNum": 0,
                    "bodyCount": 0,
                    "printManager": {
                        "active": False,
                        "distance": 0,
                        "lastX": 0,
                        "lastY": 0,
                    },
                },
            },
            [
                {"event": "position", "data": {"avatar": 3, "x": 78.4, "y": 20}},
                {"event": "position", "data": {"avatar": 2, "x": 41.6, "y": 40}},
                {
                    "event": "point",
                    "data": {"avatar": 2, "x": 41.6, "y": 40, "important": True},
                },
                {
                    "event": "property",
                    "data": {"avatar": 2, "property": "printing", "value": True},
                },
                {"event": "position", "data": {"avatar": 1, "x": 41.6, "y": 40}},
                {
                    "event": "point",
                    "data": {"avatar": 1, "x": 41.6, "y": 40, "important": False},
                },
                {"event": "die", "data": {"avatar": 1, "killer": 2, "old": False}},
                {
                    "event": "score:round",
                    "data": {"avatar": 1, "score": 0, "roundScore": 0},
                },
            ],
        ),
    ],
)
def test_js_scenario_runner_pins_trail_gap_steps(
    filename,
    expected_game,
    expected_avatars,
    expected_events,
):
    payload = _run_js_scenario(filename)
    frame = payload["trace"][0]
    avatars = {avatar["name"]: avatar for avatar in frame["avatars"]}

    assert payload["comparison"]["python_target"] == "pending-source-trail-gap"
    assert payload["comparison"]["include_events"] is True
    assert frame["stepMs"] == 100
    assert frame["game"]["size"] == 95
    assert frame["game"]["roundWinner"] is None
    assert frame["game"]["gameWinner"] is None
    assert {
        "deathCount": frame["game"]["deathCount"],
        "deaths": frame["game"]["deaths"],
        "worldBodyCount": frame["game"]["worldBodyCount"],
    } == expected_game
    assert set(avatars) == {"p0", "p1", "p2"}
    for name, expected_avatar in expected_avatars.items():
        avatar = avatars[name]
        assert {key: avatar[key] for key in expected_avatar} == expected_avatar
    assert frame["events"] == expected_events


@pytest.mark.parametrize(
    ("filename", "expected_world_body_count", "expected_alive", "expected_p1_print_manager"),
    [
        (
            "source_trail_gap_hole_space_safe_step.json",
            1,
            [True, True, True],
            {"active": True, "distance": 8.4, "lastX": 41.6, "lastY": 40.0},
        ),
        (
            "source_trail_gap_stored_body_still_kills_step.json",
            2,
            [False, True, True],
            {"active": True, "distance": 8.4, "lastX": 41.6, "lastY": 40.0},
        ),
        (
            "source_trail_gap_print_to_hole_boundary_kills_step.json",
            2,
            [False, True, True],
            {"active": True, "distance": 5.25, "lastX": 41.6, "lastY": 40.0},
        ),
        (
            "source_trail_gap_hole_to_print_boundary_kills_step.json",
            2,
            [False, True, True],
            {"active": True, "distance": 39.0, "lastX": 41.6, "lastY": 40.0},
        ),
        (
            "source_trail_gap_natural_multistep_hole_crossing.json",
            4,
            [True, True, True],
            {"active": True, "distance": 5.25, "lastX": 58.0, "lastY": 40.0},
        ),
    ],
)
def test_source_trail_gap_scenarios_match_js_common_trace(
    filename,
    expected_world_body_count,
    expected_alive,
    expected_p1_print_manager,
):
    js_payload = _run_js_scenario(filename)
    python_payload = run_source_trail_gap_scenario(_SCENARIO_DIR / filename).to_payload()

    assert python_payload["runner"] == "curvytron-v1-python-source-trail-gap-runner"
    assert python_payload["source_fidelity_scope"] == (
        "movement plus forced trail-gap body absence/collision and one natural "
        "multi-step hole-crossing state/events only"
    )
    assert diff_payload(js_payload, python_payload, common_trace=True)["status"] == "pass"

    common = project_common_trace(python_payload)
    step = common["steps"][-1]
    assert step["worldBodyCount"] == expected_world_body_count
    assert [player["alive"] for player in step["players"]] == expected_alive
    assert step["players"][1]["printManager"] == expected_p1_print_manager


def test_source_trail_gap_natural_multistep_crosses_hole_alive():
    filename = "source_trail_gap_natural_multistep_hole_crossing.json"
    js_payload = _run_js_scenario(filename)
    python_payload = run_source_trail_gap_scenario(_SCENARIO_DIR / filename).to_payload()

    assert js_payload["randomCalls"] == [
        {"index": 0, "value": 0},
        {"index": 1, "value": 0.5},
    ]
    assert python_payload["randomCalls"] == [
        {"index": 0, "value": 0.0},
        {"index": 1, "value": 0.5},
    ]
    assert diff_payload(js_payload, python_payload, common_trace=True)["status"] == "pass"

    crossing_step = project_common_trace(python_payload)["steps"][2]
    p0, p1, _p2 = crossing_step["players"]

    assert crossing_step["worldBodyCount"] == 4
    assert p0["x"] == 66
    assert p0["y"] == 40
    assert p0["alive"] is True
    assert p0["bodyCount"] == 0
    assert p1["x"] == 58
    assert p1["printing"] is False
    assert p1["trailPointCount"] == 0
    assert p1["printManager"] == {
        "active": True,
        "distance": 5.25,
        "lastX": 58,
        "lastY": 40,
    }
    assert all(event["event"] != "die" for event in crossing_step["events"])


@pytest.mark.parametrize(
    ("filename", "expected_world_body_count", "expected_body_count"),
    [
        ("source_trail_normal_point_step.json", 1, 1),
        ("source_trail_no_point_below_radius_step.json", 0, 0),
    ],
)
def test_source_trail_cadence_scenarios_match_js_common_trace(
    filename,
    expected_world_body_count,
    expected_body_count,
):
    js_payload = _run_js_scenario(filename)
    python_payload = run_source_trail_cadence_scenario(_SCENARIO_DIR / filename).to_payload()

    assert python_payload["runner"] == "curvytron-v1-python-source-trail-cadence-runner"
    assert python_payload["source_fidelity_scope"] == (
        "movement plus normal trail point cadence state/events only"
    )
    assert diff_payload(js_payload, python_payload, common_trace=True)["status"] == "pass"

    common = project_common_trace(python_payload)
    step = common["steps"][0]
    player = step["players"][0]
    assert step["worldBodyCount"] == expected_world_body_count
    assert player["printing"] is True
    assert player["bodyCount"] == expected_body_count


def test_source_trail_cadence_runner_rejects_unscoped_scenario():
    with pytest.raises(
        ScenarioError,
        match="source-trail-cadence-canary runner supports",
    ):
        run_source_trail_cadence_scenario(
            _SCENARIO_DIR / "source_print_manager_no_toggle_control_step.json"
        )


def test_source_trail_gap_runner_rejects_unscoped_scenario():
    with pytest.raises(
        ScenarioError,
        match="source-trail-gap-canary runner supports",
    ):
        run_source_trail_gap_scenario(_SCENARIO_DIR / "source_trail_normal_point_step.json")


def test_source_body_canary_runner_rejects_unscoped_scenario():
    with pytest.raises(
        ScenarioError,
        match="source-body-canary runner supports source_body_opponent",
    ):
        run_source_body_canary_scenario(_SCENARIO_DIR / "source_normal_wall_death_step.json")


def test_js_scenario_runner_rejects_seeded_world_bodies_when_world_inactive(tmp_path):
    if shutil.which("node") is None:
        pytest.skip("node executable is not available")

    scenario = json.loads(
        (_SCENARIO_DIR / "source_body_opponent_overlap_kills_step.json").read_text(
            encoding="utf-8"
        )
    )
    scenario["source_setup"]["game"]["world_active"] = False
    scenario_path = tmp_path / "inactive_world_body.json"
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")

    result = subprocess.run(
        [
            "node",
            str(_REPO_ROOT / "tools" / "reference_oracle" / "scenario_runner.js"),
            str(scenario_path),
        ],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    error = json.loads(result.stderr)
    assert (
        "initial_state.world_bodies requires source_setup.game.world_active not to be false"
        in error["error"]["message"]
    )


@pytest.mark.parametrize(
    ("body_patch", "expected_message"),
    [
        ({"player_id": "p9"}, "references unknown player p9"),
        ({"x": "bad"}, "initial_state.world_bodies[0].x must be a finite number"),
    ],
)
def test_js_scenario_runner_rejects_invalid_seeded_world_bodies(
    tmp_path,
    body_patch,
    expected_message,
):
    if shutil.which("node") is None:
        pytest.skip("node executable is not available")

    scenario = json.loads(
        (_SCENARIO_DIR / "source_body_opponent_overlap_kills_step.json").read_text(
            encoding="utf-8"
        )
    )
    scenario["initial_state"]["world_bodies"][0].update(body_patch)
    scenario_path = tmp_path / "invalid_world_body.json"
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")

    result = subprocess.run(
        [
            "node",
            str(_REPO_ROOT / "tools" / "reference_oracle" / "scenario_runner.js"),
            str(scenario_path),
        ],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    error = json.loads(result.stderr)
    assert expected_message in error["error"]["message"]


def test_js_scenario_runner_echoes_comparison_for_event_projection():
    if shutil.which("node") is None:
        pytest.skip("node executable is not available")

    result = subprocess.run(
        [
            "node",
            str(_REPO_ROOT / "tools" / "reference_oracle" / "scenario_runner.js"),
            str(_SCENARIO_DIR / "source_borderless_wrap_step.json"),
        ],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["comparison"]["include_events"] is True
    assert payload["trace"][0]["events"] == [
        {"event": "position", "data": {"avatar": 2, "x": 42.4, "y": 44}},
        {"event": "position", "data": {"avatar": 1, "x": 88.95, "y": 44}},
        {"event": "position", "data": {"avatar": 1, "x": 0, "y": 44}},
    ]


@pytest.mark.parametrize(
        ("filename", "expected_map_size"),
    [
        ("source_normal_wall_3p_two_die_one_survivor_step.json", 95),
        ("source_normal_wall_4p_ordered_deaths_survivor_score.json", 101),
        ("source_normal_wall_4p_two_prior_then_same_frame_terminal_draw.json", 101),
    ],
)
def test_source_normal_wall_multiplayer_common_trace_includes_map_size(
    filename,
    expected_map_size,
):
    if shutil.which("node") is None:
        pytest.skip("node executable is not available")

    result = subprocess.run(
        [
            "node",
            str(_REPO_ROOT / "tools" / "reference_oracle" / "scenario_runner.js"),
            str(_SCENARIO_DIR / filename),
        ],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    js_payload = json.loads(result.stdout)
    python_payload = run_source_border_rules_scenario(_SCENARIO_DIR / filename).to_payload()

    assert project_common_trace(js_payload)["map_size"] == expected_map_size
    assert project_common_trace(python_payload)["map_size"] == expected_map_size
    assert diff_payload(js_payload, python_payload, common_trace=True)["status"] == "pass"


@pytest.mark.parametrize(
    ("filename", "expected_runner"),
    [
        ("source_normal_wall_death_step.json", "curvytron-v1-python-source-normal-wall-runner"),
        (
            "source_borderless_wrap_step.json",
            "curvytron-v1-python-source-borderless-wrap-runner",
        ),
    ],
)
def test_source_border_rules_dispatches_to_matching_runner(filename, expected_runner):
    payload = run_source_border_rules_scenario(_SCENARIO_DIR / filename).to_payload()

    assert payload["runner"] == expected_runner
    assert payload["source_fidelity"] is True


def test_source_action_parser_rejects_player_count_mismatch():
    scenario = load_scenario(
        _SCENARIO_DIR / "source_normal_wall_3p_two_die_one_survivor_step.json"
    ).to_payload()
    scenario["action_script"] = [{"moves": {"p0": 0, "p1": 0}}]

    with pytest.raises(ScenarioError, match="player_2"):
        parse_scenario(scenario)


def test_toy_v0_runner_still_rejects_multiplayer_scenarios():
    with pytest.raises(ScenarioError, match="toy-v0 scenario runner currently supports exactly 2"):
        run_scenario(_SCENARIO_DIR / "source_normal_wall_3p_two_die_one_survivor_step.json")


def test_first_mismatch_reports_nested_value_path():
    left = {"trace": {"frames": [{"tick": 0}, {"tick": 1}]}}
    right = {"trace": {"frames": [{"tick": 0}, {"tick": 2}]}}

    mismatch = first_mismatch(left, right)

    assert mismatch is not None
    assert mismatch.path == "$.trace.frames[1].tick"
    assert mismatch.left == 1
    assert mismatch.right == 2
    assert mismatch.describe() == (
        "First mismatch at $.trace.frames[1].tick: "
        "left is 1, right is 2 (values differ)."
    )


def test_first_mismatch_reports_missing_key():
    result = diff_payload({"trace": {"scope": "toy"}}, {"trace": {"scope": "toy", "seed": 3}})

    assert result["match"] is False
    assert result["path"] == "$.trace.seed"
    assert result["reason"] == "key missing on left"
    assert result["message"] == (
        "First mismatch at $.trace.seed: "
        "left is null, right is 3 (key missing on left)."
    )


def test_diff_payload_reports_exact_match():
    assert diff_payload({"frames": [{"tick": 0}]}, {"frames": [{"tick": 0}]}) == {
        "match": True,
        "message": "JSON payloads match exactly.",
    }
