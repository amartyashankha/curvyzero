import importlib.util
import sys
from pathlib import Path

from curvyzero.env.trace_compare import project_common_trace

_DIFF_PATH = Path(__file__).resolve().parents[1] / "tools" / "fidelity_diff.py"
_DIFF_SPEC = importlib.util.spec_from_file_location("fidelity_diff", _DIFF_PATH)
assert _DIFF_SPEC is not None
assert _DIFF_SPEC.loader is not None
fidelity_diff = importlib.util.module_from_spec(_DIFF_SPEC)
sys.modules[_DIFF_SPEC.name] = fidelity_diff
_DIFF_SPEC.loader.exec_module(fidelity_diff)
diff_payload = fidelity_diff.diff_payload


def test_projects_js_runner_output_to_common_trace():
    payload = {
        "scenario": "forced_two_player_turn_step",
        "playerCount": 2,
        "loadedSources": ["src/shared/Collection.js"],
        "trace": [
            {
                "tick": 0,
                "stepMs": 16.666667,
                "avatars": [
                    {
                        "id": 1,
                        "name": "p0",
                        "x": 20.266376,
                        "y": 39.98756,
                        "angle": -0.046667,
                        "alive": True,
                        "score": 0,
                        "roundScore": 0,
                    }
                ],
            }
        ],
    }

    assert project_common_trace(payload) == {
        "schema": "curvyzero_common_trace/v1",
        "scenario_id": "forced_two_player_turn_step",
        "steps": [
            {
                "step_index": 0,
                "step_ms": 16.666667,
                "players": [
                    {
                        "player_id": "p0",
                        "x": 20.266376,
                        "y": 39.98756,
                        "angle": -0.046667,
                        "alive": True,
                        "score": 0,
                        "roundScore": 0,
                    }
                ],
            }
        ],
    }


def test_projects_python_runner_output_to_common_trace_and_drops_reset_frame():
    payload = {
        "schema": "curvyzero_python_scenario_trace/v1",
        "runner": "curvyzero-v0-python-toy-scenario-runner",
        "scenario_id": "forced_two_player_turn_step",
        "source_fidelity": False,
        "scenario": {
            "scenario_id": "forced_two_player_turn_step",
            "initial_state": {"players": [{"id": "p0"}]},
            "action_script": [{"moves": {"p0": -1}}],
            "time_policy": {"kind": "fixed", "step_ms": 16.666667},
        },
        "trace": {
            "frames": [
                {
                    "tick": 0,
                    "positions": [[20.0, 40.0]],
                    "headings": [0.0],
                    "alive": [True],
                },
                {
                    "tick": 1,
                    "positions": [[21.0, 40.0]],
                    "headings": [0.1],
                    "alive": [True],
                },
            ]
        },
    }

    assert project_common_trace(payload) == {
        "schema": "curvyzero_common_trace/v1",
        "scenario_id": "forced_two_player_turn_step",
        "steps": [
            {
                "step_index": 0,
                "step_ms": 16.666667,
                "players": [
                    {
                        "player_id": "p0",
                        "x": 21.0,
                        "y": 40.0,
                        "angle": 0.1,
                        "alive": True,
                    }
                ],
            }
        ],
    }


def test_projects_python_per_step_elapsed_ms_after_dropping_reset_frame():
    payload = {
        "schema": "curvyzero_python_scenario_trace/v1",
        "runner": "curvytron-v1-python-source-kinematics-runner",
        "scenario_id": "source_kinematics_varied_elapsed_multistep",
        "source_fidelity": True,
        "scenario": {
            "scenario_id": "source_kinematics_varied_elapsed_multistep",
            "initial_state": {"players": [{"id": "p0"}]},
            "action_script": [{"moves": {"p0": 0}}, {"moves": {"p0": 0}}],
            "time_policy": {
                "kind": "per-step",
                "step_ms_sequence": [10, 20],
            },
        },
        "trace": {
            "frames": [
                {
                    "tick": 0,
                    "positions": [[20.0, 40.0]],
                    "headings": [0.0],
                    "alive": [True],
                },
                {
                    "tick": 1,
                    "positions": [[20.16, 40.0]],
                    "headings": [0.0],
                    "alive": [True],
                },
                {
                    "tick": 2,
                    "positions": [[20.48, 40.0]],
                    "headings": [0.0],
                    "alive": [True],
                },
            ]
        },
    }

    assert [step["step_ms"] for step in project_common_trace(payload)["steps"]] == [10, 20]


def test_common_trace_diff_ignores_metadata_noise_and_reports_projected_mismatch():
    js_payload = {
        "scenario": "forced_two_player_turn_step",
        "loadedSources": ["js-only-metadata"],
        "trace": [
            {
                "tick": 0,
                "stepMs": 16.666667,
                "avatars": [
                    {"name": "p0", "x": 20.266376, "y": 39.98756, "angle": -0.046667, "alive": True}
                ],
            }
        ],
    }
    python_payload = {
        "scenario_id": "forced_two_player_turn_step",
        "runner": "python-only-metadata",
        "scenario": {
            "scenario_id": "forced_two_player_turn_step",
            "initial_state": {"players": [{"id": "p0"}]},
            "action_script": [{"moves": {"p0": -1}}],
            "time_policy": {"step_ms": 16.666667},
        },
        "trace": {
            "frames": [
                {"positions": [[20.0, 40.0]], "headings": [0.0], "alive": [True]},
                {"positions": [[21.0, 40.0]], "headings": [0.1], "alive": [True]},
            ]
        },
    }

    result = diff_payload(js_payload, python_payload, common_trace=True)

    assert result["match"] is False
    assert result["status"] == "fail"
    assert result["path"] == "$.steps[0].players[0].angle"
    assert result["reason"] == "values differ"


def test_common_trace_ignores_raw_events_without_opt_in():
    js_payload = {
        "scenario": "toy-v0-event-noise",
        "trace": [
            {
                "tick": 0,
                "stepMs": 16,
                "avatars": [
                    {
                        "id": 1,
                        "name": "p0",
                        "x": 21.0,
                        "y": 40.0,
                        "angle": 0.0,
                        "alive": True,
                    }
                ],
                "events": [
                    {"event": "position", "data": {"avatar": 1, "x": 21.0, "y": 40.0}}
                ],
            }
        ],
    }
    python_payload = {
        "scenario_id": "toy-v0-event-noise",
        "scenario": {
            "scenario_id": "toy-v0-event-noise",
            "initial_state": {"players": [{"id": "p0"}]},
            "action_script": [{"moves": {"p0": 0}}],
            "time_policy": {"step_ms": 16},
        },
        "trace": {
            "frames": [
                {"positions": [[20.0, 40.0]], "headings": [0.0], "alive": [True]},
                {"positions": [[21.0, 40.0]], "headings": [0.0], "alive": [True]},
            ]
        },
    }

    assert "events" not in project_common_trace(js_payload)["steps"][0]
    assert diff_payload(js_payload, python_payload, common_trace=True)["status"] == "pass"


def test_common_trace_projects_events_when_opted_in():
    expected_events = [
        {"event": "position", "player_id": "p1", "x": 42.4, "y": 44},
        {"event": "position", "player_id": "p0", "x": 88.95, "y": 44},
        {
            "event": "point",
            "player_id": "p0",
            "x": 88.95,
            "y": 44,
            "important": False,
        },
        {"event": "die", "player_id": "p0", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 0},
        {"event": "score:round", "player_id": "p1", "score": 0, "roundScore": 1},
        {"event": "score", "player_id": "p1", "score": 1, "roundScore": 1},
        {"event": "score", "player_id": "p0", "score": 0, "roundScore": 0},
        {"event": "round:end", "winner_id": "p1"},
    ]
    js_payload = {
        "scenario": "source_normal_wall_death_step",
        "comparison": {"include_events": True},
        "trace": [
            {
                "tick": 0,
                "stepMs": 100,
                "avatars": [
                    {
                        "id": 1,
                        "name": "p0",
                        "x": 88.95,
                        "y": 44,
                        "angle": 0,
                        "alive": False,
                    },
                    {
                        "id": 2,
                        "name": "p1",
                        "x": 42.4,
                        "y": 44,
                        "angle": 3.141593,
                        "alive": True,
                    },
                ],
                "events": [
                    {"event": "position", "data": {"avatar": 2, "x": 42.4, "y": 44}},
                    {"event": "position", "data": {"avatar": 1, "x": 88.95, "y": 44}},
                    {
                        "event": "point",
                        "data": {"avatar": 1, "x": 88.95, "y": 44, "important": False},
                    },
                    {"event": "die", "data": {"avatar": 1, "killer": None, "old": None}},
                    {
                        "event": "score:round",
                        "data": {"avatar": 1, "score": 0, "roundScore": 0},
                    },
                    {
                        "event": "score:round",
                        "data": {"avatar": 2, "score": 0, "roundScore": 1},
                    },
                    {"event": "score", "data": {"avatar": 2, "score": 1, "roundScore": 1}},
                    {"event": "score", "data": {"avatar": 1, "score": 0, "roundScore": 0}},
                    {"event": "round:end", "data": {"winner": 2}},
                ],
            }
        ],
    }
    python_payload = {
        "scenario_id": "source_normal_wall_death_step",
        "scenario": {
            "scenario_id": "source_normal_wall_death_step",
            "initial_state": {"players": [{"id": "p0"}, {"id": "p1"}]},
            "action_script": [{"moves": {"p0": 0, "p1": 0}}],
            "time_policy": {"step_ms": 100},
            "comparison": {"include_events": True},
        },
        "trace": {
            "frames": [
                {
                    "positions": [[87.35, 44], [44, 44]],
                    "headings": [0, 3.141593],
                    "alive": [True, True],
                },
                {
                    "positions": [[88.95, 44], [42.4, 44]],
                    "headings": [0, 3.141593],
                    "alive": [False, True],
                    "events": expected_events,
                },
            ]
        },
    }

    assert project_common_trace(js_payload)["steps"][0]["events"] == expected_events
    assert project_common_trace(python_payload)["steps"][0]["events"] == expected_events
    assert diff_payload(js_payload, python_payload, common_trace=True)["status"] == "pass"


def test_common_trace_projects_source_angle_events_when_opted_in():
    expected_events = [
        {"event": "angle", "player_id": "p1", "angle": 3.169593},
        {"event": "position", "player_id": "p1", "x": 59.840063, "y": 39.995521},
        {"event": "angle", "player_id": "p0", "angle": -0.028},
        {"event": "position", "player_id": "p0", "x": 20.159937, "y": 39.995521},
    ]
    js_payload = {
        "scenario": "source_kinematics_varied_elapsed_multistep",
        "comparison": {"include_events": True},
        "trace": [
            {
                "tick": 0,
                "stepMs": 10,
                "avatars": [
                    {
                        "id": 1,
                        "name": "p0",
                        "x": 20.159937,
                        "y": 39.995521,
                        "angle": -0.028,
                        "alive": True,
                    },
                    {
                        "id": 2,
                        "name": "p1",
                        "x": 59.840063,
                        "y": 39.995521,
                        "angle": 3.169593,
                        "alive": True,
                    },
                ],
                "events": [
                    {"event": "angle", "data": {"avatar": 2, "angle": 3.169593}},
                    {
                        "event": "position",
                        "data": {"avatar": 2, "x": 59.840063, "y": 39.995521},
                    },
                    {"event": "angle", "data": {"avatar": 1, "angle": -0.028}},
                    {
                        "event": "position",
                        "data": {"avatar": 1, "x": 20.159937, "y": 39.995521},
                    },
                ],
            }
        ],
    }
    python_payload = {
        "scenario_id": "source_kinematics_varied_elapsed_multistep",
        "scenario": {
            "scenario_id": "source_kinematics_varied_elapsed_multistep",
            "initial_state": {"players": [{"id": "p0"}, {"id": "p1"}]},
            "action_script": [{"moves": {"p0": -1, "p1": 1}}],
            "time_policy": {"kind": "per-step", "step_ms_sequence": [10]},
            "comparison": {"include_events": True},
        },
        "trace": {
            "frames": [
                {
                    "positions": [[20.0, 40.0], [60.0, 40.0]],
                    "headings": [0.0, 3.141593],
                    "alive": [True, True],
                },
                {
                    "positions": [[20.159937, 39.995521], [59.840063, 39.995521]],
                    "headings": [-0.028, 3.169593],
                    "alive": [True, True],
                    "events": expected_events,
                },
            ]
        },
    }

    assert project_common_trace(js_payload)["steps"][0]["events"] == expected_events
    assert project_common_trace(python_payload)["steps"][0]["events"] == expected_events
    assert diff_payload(js_payload, python_payload, common_trace=True)["status"] == "pass"


def test_common_trace_projects_source_map_size():
    js_payload = {
        "scenario": "source_normal_wall_3p_two_die_one_survivor_step",
        "trace": [
            {
                "tick": 0,
                "stepMs": 100,
                "game": {"size": 95},
                "avatars": [
                    {"id": 1, "name": "p0", "x": 49.1, "y": 47.5, "angle": 0, "alive": True},
                    {
                        "id": 2,
                        "name": "p1",
                        "x": 94.95,
                        "y": 47.5,
                        "angle": 0,
                        "alive": False,
                    },
                    {
                        "id": 3,
                        "name": "p2",
                        "x": -0.55,
                        "y": 47.5,
                        "angle": 3.141593,
                        "alive": False,
                    },
                ],
            }
        ],
    }
    python_payload = {
        "schema": "curvyzero_python_scenario_trace/v1",
        "scenario_id": "source_normal_wall_3p_two_die_one_survivor_step",
        "scenario": {
            "scenario_id": "source_normal_wall_3p_two_die_one_survivor_step",
            "initial_state": {
                "map_size": 95,
                "players": [{"id": "p0"}, {"id": "p1"}, {"id": "p2"}],
            },
            "action_script": [{"moves": {"p0": 0, "p1": 0, "p2": 0}}],
            "time_policy": {"step_ms": 100},
        },
        "trace": {
            "frames": [
                {
                    "positions": [[47.5, 47.5], [93.35, 47.5], [1.05, 47.5]],
                    "headings": [0, 0, 3.141593],
                    "alive": [True, True, True],
                },
                {
                    "positions": [[49.1, 47.5], [94.95, 47.5], [-0.55, 47.5]],
                    "headings": [0, 0, 3.141593],
                    "alive": [True, False, False],
                },
            ]
        },
    }

    assert project_common_trace(js_payload)["map_size"] == 95
    assert project_common_trace(python_payload)["map_size"] == 95
    assert diff_payload(js_payload, python_payload, common_trace=True)["status"] == "pass"


def test_include_events_requires_python_event_arrays():
    js_payload = {
        "scenario": "toy-v0-missing-events",
        "comparison": {"include_events": True},
        "trace": [
            {
                "tick": 0,
                "stepMs": 16,
                "avatars": [{"id": 1, "name": "p0", "x": 21.0, "y": 40.0}],
                "events": [],
            }
        ],
    }
    python_payload = {
        "scenario_id": "toy-v0-missing-events",
        "scenario": {
            "scenario_id": "toy-v0-missing-events",
            "initial_state": {"players": [{"id": "p0"}]},
            "action_script": [{"moves": {"p0": 0}}],
            "time_policy": {"step_ms": 16},
            "comparison": {"include_events": True},
        },
        "trace": {
            "frames": [
                {"positions": [[20.0, 40.0]], "headings": [0.0], "alive": [True]},
                {"positions": [[21.0, 40.0]], "headings": [0.0], "alive": [True]},
            ]
        },
    }

    result = diff_payload(js_payload, python_payload, common_trace=True)

    assert result["match"] is False
    assert result["status"] == "blocked"
    assert result["reason"] == "trace normalization error"
    assert (
        "trace.frames[0].events must be a list when comparison.include_events is true"
        in result["message"]
    )


def test_diff_payload_marks_exact_match_as_pass():
    result = diff_payload({"frames": [{"tick": 0}]}, {"frames": [{"tick": 0}]})

    assert result["match"] is True
    assert result["status"] == "pass"


def test_diff_payload_treats_equal_json_numbers_as_match():
    result = diff_payload(
        {"steps": [{"players": [{"angle": 0, "y": 40}]}]},
        {"steps": [{"players": [{"angle": 0.0, "y": 40.0}]}]},
    )

    assert result["match"] is True
    assert result["status"] == "pass"


def test_common_trace_normalization_error_is_blocked():
    result = diff_payload({"trace": "bad"}, {"trace": []}, common_trace=True)

    assert result["match"] is False
    assert result["status"] == "blocked"
    assert result["reason"] == "trace normalization error"
    assert result["message"].startswith("Trace normalization error:")
