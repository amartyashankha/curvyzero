import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_vector_arrays_to_fidelity.py"
_SPEC = importlib.util.spec_from_file_location("compare_vector_arrays_to_fidelity", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
vector_compare = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = vector_compare
_SPEC.loader.exec_module(vector_compare)


def test_source_body_opponent_tangent_safe_step_passes_common_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_body_opponent_tangent_safe_step.json",
        body_capacity=4,
    )

    players = result["array_projection"]["steps"][0]["players"]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["body_hits"] == 0
    assert result["array_counters"]["death_points_inserted"] == 0
    assert result["array_projection"]["steps"][0]["worldBodyCount"] == 1
    assert players[0]["alive"] is True
    assert players[1]["bodyNum"] == 1
    assert "$.steps[0].events[0].event" in result["compared_fields"]
    assert result["array_counters"]["events_emitted"] == 3
    assert result["array_event_arrays"]["event_count"] == [3]
    assert "strict tangent non-overlap remains safe" in result["covered_mechanics"]


def test_source_body_opponent_overlap_kills_step_passes_common_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_body_opponent_overlap_kills_step.json",
        body_capacity=4,
    )

    players = result["array_projection"]["steps"][0]["players"]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["body_hits"] == 1
    assert result["array_counters"]["death_points_inserted"] == 1
    assert result["array_projection"]["steps"][0]["worldBodyCount"] == 2
    assert players[0]["alive"] is False
    assert players[0]["lastTrailPoint"] == [20.0, 20.0]
    assert players[1]["alive"] is True
    assert result["array_counters"]["events_emitted"] == 6
    assert result["array_projection"]["steps"][0]["events"][-2] == {
        "event": "die",
        "player_id": "p0",
        "killer_id": "p1",
        "old": False,
    }
    assert "opponent strict-overlap body hit" in result["covered_mechanics"]


def test_verified_own_body_latency_kill_passes_common_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_body_own_delta4_kills_step.json",
        body_capacity=4,
    )

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["body_hits"] == 1
    assert result["array_counters"]["death_points_inserted"] == 1
    assert result["array_projection"]["steps"][0]["worldBodyCount"] == 2
    assert result["array_projection"]["steps"][0]["players"][0]["alive"] is False
    assert result["array_counters"]["events_emitted"] == 6
    assert result["array_event_arrays"]["event_type"][0][:6] == [
        vector_compare.EVENT_POSITION,
        vector_compare.EVENT_POSITION,
        vector_compare.EVENT_POSITION,
        vector_compare.EVENT_POINT,
        vector_compare.EVENT_DIE,
        vector_compare.EVENT_SCORE_ROUND,
    ]


def test_verified_same_frame_point_materializes_before_lower_player_collision():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_body_same_frame_point_kills_step.json",
        body_capacity=4,
    )

    players = result["array_projection"]["steps"][0]["players"]

    assert result["status"] == "pass"
    assert result["array_counters"]["normal_points_inserted"] == 1
    assert result["array_counters"]["death_points_inserted"] == 1
    assert result["array_projection"]["steps"][0]["worldBodyCount"] == 2
    assert players[0]["alive"] is False
    assert players[0]["lastTrailPoint"] == [41.6, 40.0]
    assert players[1]["trailPointCount"] == 1
    assert result["array_counters"]["events_emitted"] == 7


def test_source_borderless_wrap_step_passes_common_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_borderless_wrap_step.json",
        body_capacity=4,
    )

    players = result["array_projection"]["steps"][0]["players"]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["borderless_wraps"] == 1
    assert players[0]["x"] == 0.0
    assert players[0]["alive"] is True
    assert players[1]["x"] == 42.4
    assert result["array_counters"]["events_emitted"] == 3
    assert result["array_projection"]["steps"][0]["events"][-1] == {
        "event": "position",
        "player_id": "p0",
        "x": 0.0,
        "y": 44.0,
    }
    assert "simple source borderless wrap after movement" in result["covered_mechanics"]


def test_source_borderless_wrap_skips_destination_body_then_next_frame_kills_passes_full_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_borderless_wrap_skips_destination_body_then_next_frame_kills.json",
        body_capacity=4,
    )

    first_step, second_step = result["array_projection"]["steps"]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["borderless_wraps"] == 1
    assert result["array_counters"]["body_hits"] == 1
    assert result["array_counters"]["death_points_inserted"] == 1
    assert result["array_counters"]["terminal_score_rows"] == 0
    assert first_step["players"][0]["x"] == 0.0
    assert first_step["players"][0]["alive"] is True
    assert first_step["worldBodyCount"] == 1
    assert second_step["players"][0]["alive"] is False
    assert second_step["players"][0]["lastTrailPoint"] == [0.0, 44.0]
    assert second_step["worldBodyCount"] == 2
    assert second_step["events"][3:6] == [
        {
            "event": "point",
            "player_id": "p0",
            "x": 0.0,
            "y": 44.0,
            "important": False,
        },
        {
            "event": "die",
            "player_id": "p0",
            "killer_id": "p1",
            "old": False,
        },
        {
            "event": "score:round",
            "player_id": "p0",
            "score": 0,
            "roundScore": 0,
        },
    ]
    assert result["array_counters_by_step"][0]["body_hits"] == 0
    assert result["array_counters_by_step"][1]["body_hits"] == 1
    assert "wrap frame skips destination body collision" in result["covered_mechanics"]


def test_source_borderless_print_manager_wrap_toggle_step_passes_common_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_borderless_print_manager_wrap_toggle_step.json",
        body_capacity=4,
    )

    player = result["array_projection"]["steps"][0]["players"][0]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["borderless_wraps"] == 1
    assert result["array_counters"]["normal_points_inserted"] == 1
    assert result["array_counters"]["print_manager_toggle_updates"] == 1
    assert result["array_counters"]["print_manager_visual_clears"] == 1
    assert result["array_counters"]["random_tape_draws"] == 1
    assert result["array_projection"]["steps"][0]["worldBodyCount"] == 2
    assert player["x"] == 0.0
    assert player["printing"] is False
    assert player["trailPointCount"] == 0
    assert player["lastTrailPoint"] is None
    assert player["bodyNum"] == 1
    assert player["bodyCount"] == 2
    assert player["printManager"] == {
        "active": True,
        "distance": 5.25,
        "lastX": 0.0,
        "lastY": 40.0,
    }
    assert result["array_event_arrays"]["event_type"][0][:5] == [
        vector_compare.EVENT_POSITION,
        vector_compare.EVENT_POINT,
        vector_compare.EVENT_POSITION,
        vector_compare.EVENT_POINT,
        vector_compare.EVENT_PROPERTY,
    ]
    assert result["array_projection"]["steps"][0]["events"][-2:] == [
        {
            "event": "point",
            "player_id": "p0",
            "x": 0.0,
            "y": 40.0,
            "important": True,
        },
        {
            "event": "property",
            "player_id": "p0",
            "property": "printing",
            "value": False,
        },
    ]
    assert "post-wrap PrintManager important point insertion" in result[
        "covered_mechanics"
    ]


def test_source_normal_wall_death_step_passes_common_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_normal_wall_death_step.json",
        body_capacity=4,
    )

    players = result["array_projection"]["steps"][0]["players"]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["normal_wall_deaths"] == 1
    assert result["array_counters"]["death_points_inserted"] == 1
    assert result["array_counters"]["terminal_score_rows"] == 1
    assert result["array_projection"]["steps"][0]["worldBodyCount"] == 1
    assert players[0]["alive"] is False
    assert players[0]["score"] == 0
    assert players[0]["roundScore"] == 0
    assert players[1]["alive"] is True
    assert players[1]["score"] == 1
    assert players[1]["roundScore"] == 0
    assert result["array_counters"]["events_emitted"] == 9
    assert result["array_projection"]["steps"][0]["events"][-1] == {
        "event": "round:end",
        "winner_id": "p1",
    }


# The promoted multiplayer wall cases are direct fast-runtime/comparator
# canaries seeded from source fixtures. They do not claim public env parity.
def test_source_normal_wall_3p_two_die_one_survivor_passes_common_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_normal_wall_3p_two_die_one_survivor_step.json",
        body_capacity=4,
    )

    step = result["array_projection"]["steps"][0]
    players = step["players"]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["movement_updates"] == 3
    assert result["array_counters"]["normal_wall_deaths"] == 2
    assert result["array_counters"]["death_points_inserted"] == 2
    assert result["array_counters"]["terminal_score_rows"] == 1
    assert result["array_counters"]["events_emitted"] == 14
    assert step["worldBodyCount"] == 2
    assert [player["alive"] for player in players] == [True, False, False]
    assert [player["score"] for player in players] == [2, 0, 0]
    assert step["events"][:8] == [
        {"event": "position", "player_id": "p2", "x": pytest.approx(-0.55), "y": 47.5},
        {
            "event": "point",
            "player_id": "p2",
            "x": pytest.approx(-0.55),
            "y": 47.5,
            "important": False,
        },
        {"event": "die", "player_id": "p2", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p2", "score": 0, "roundScore": 0},
        {"event": "position", "player_id": "p1", "x": pytest.approx(94.95), "y": 47.5},
        {
            "event": "point",
            "player_id": "p1",
            "x": pytest.approx(94.95),
            "y": 47.5,
            "important": False,
        },
        {"event": "die", "player_id": "p1", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p1", "score": 0, "roundScore": 0},
    ]
    assert step["events"][-5:] == [
        {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 2},
        {"event": "score", "player_id": "p2", "score": 0, "roundScore": 0},
        {"event": "score", "player_id": "p1", "score": 0, "roundScore": 0},
        {"event": "score", "player_id": "p0", "score": 2, "roundScore": 2},
        {"event": "round:end", "winner_id": "p0"},
    ]
    assert "three-player same-frame wall deaths in reverse source order" in result[
        "covered_mechanics"
    ]


def test_source_normal_wall_4p_ordered_deaths_passes_full_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_normal_wall_4p_ordered_deaths_survivor_score.json",
        body_capacity=4,
    )

    first_step, second_step, third_step = result["array_projection"]["steps"]
    final_players = third_step["players"]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["normal_wall_deaths"] == 3
    assert result["array_counters"]["death_points_inserted"] == 3
    assert result["array_counters"]["terminal_score_rows"] == 1
    assert result["array_counters"]["events_emitted"] == 24
    assert [step["normal_wall_deaths"] for step in result["array_counters_by_step"]] == [
        1,
        1,
        1,
    ]
    assert [step["terminal_score_rows"] for step in result["array_counters_by_step"]] == [
        0,
        0,
        1,
    ]
    assert first_step["events"][3:6] == [
        {
            "event": "point",
            "player_id": "p1",
            "x": pytest.approx(100.95),
            "y": 50.0,
            "important": False,
        },
        {"event": "die", "player_id": "p1", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p1", "score": 0, "roundScore": 0},
    ]
    assert second_step["events"][2:5] == [
        {
            "event": "point",
            "player_id": "p2",
            "x": pytest.approx(100.95),
            "y": 50.0,
            "important": False,
        },
        {"event": "die", "player_id": "p2", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p2", "score": 0, "roundScore": 1},
    ]
    assert third_step["events"][-5:] == [
        {"event": "score", "player_id": "p3", "score": 2, "roundScore": 2},
        {"event": "score", "player_id": "p2", "score": 1, "roundScore": 1},
        {"event": "score", "player_id": "p1", "score": 0, "roundScore": 0},
        {"event": "score", "player_id": "p0", "score": 3, "roundScore": 3},
        {"event": "round:end", "winner_id": "p0"},
    ]
    assert [player["alive"] for player in final_players] == [True, False, False, False]
    assert [player["score"] for player in final_players] == [3, 0, 1, 2]
    assert "full multi-step four-player ordered wall death trace" in result[
        "covered_mechanics"
    ]


def test_source_normal_wall_4p_terminal_draw_passes_full_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_normal_wall_4p_two_prior_then_same_frame_terminal_draw.json",
        body_capacity=4,
    )

    first_step, second_step, third_step = result["array_projection"]["steps"]
    final_players = third_step["players"]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["normal_wall_deaths"] == 4
    assert result["array_counters"]["death_points_inserted"] == 4
    assert result["array_counters"]["terminal_score_rows"] == 1
    assert result["array_counters"]["events_emitted"] == 26
    assert [step["normal_wall_deaths"] for step in result["array_counters_by_step"]] == [
        1,
        1,
        2,
    ]
    assert [step["terminal_score_rows"] for step in result["array_counters_by_step"]] == [
        0,
        0,
        1,
    ]
    assert first_step["events"][1:4] == [
        {
            "event": "point",
            "player_id": "p3",
            "x": pytest.approx(100.95),
            "y": 80.0,
            "important": False,
        },
        {"event": "die", "player_id": "p3", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p3", "score": 0, "roundScore": 0},
    ]
    assert second_step["events"][1:4] == [
        {
            "event": "point",
            "player_id": "p2",
            "x": pytest.approx(100.95),
            "y": 60.0,
            "important": False,
        },
        {"event": "die", "player_id": "p2", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p2", "score": 0, "roundScore": 1},
    ]
    assert third_step["events"][2:8] == [
        {"event": "die", "player_id": "p1", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p1", "score": 0, "roundScore": 2},
        {"event": "position", "player_id": "p0", "x": pytest.approx(100.95), "y": 20.0},
        {
            "event": "point",
            "player_id": "p0",
            "x": pytest.approx(100.95),
            "y": 20.0,
            "important": False,
        },
        {"event": "die", "player_id": "p0", "killer_id": None, "old": None},
        {"event": "score:round", "player_id": "p0", "score": 0, "roundScore": 2},
    ]
    assert third_step["events"][-5:] == [
        {"event": "score", "player_id": "p3", "score": 0, "roundScore": 0},
        {"event": "score", "player_id": "p2", "score": 1, "roundScore": 1},
        {"event": "score", "player_id": "p1", "score": 2, "roundScore": 2},
        {"event": "score", "player_id": "p0", "score": 2, "roundScore": 2},
        {"event": "round:end", "winner_id": None},
    ]
    assert [player["alive"] for player in final_players] == [False, False, False, False]
    assert [player["score"] for player in final_players] == [2, 2, 1, 0]
    assert "same-frame terminal draw score order" in result["covered_mechanics"]


def test_source_print_manager_no_toggle_control_step_passes_common_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_print_manager_no_toggle_control_step.json",
        body_capacity=4,
    )

    player = result["array_projection"]["steps"][0]["players"][0]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["print_manager_no_toggle_updates"] == 1
    assert result["array_counters"]["print_manager_toggle_rows_unhandled"] == 0
    assert result["array_counters"]["events_emitted"] == 1
    assert result["array_projection"]["steps"][0]["worldBodyCount"] == 0
    assert player["printing"] is False
    assert player["printManager"] == {
        "active": True,
        "distance": 8.399999999999999,
        "lastX": 21.6,
        "lastY": 40.0,
    }
    assert result["array_projection"]["steps"][0]["events"] == [
        {
            "event": "position",
            "player_id": "p0",
            "x": 21.6,
            "y": 40.0,
        }
    ]
    assert "active PrintManager distance bookkeeping without toggle" in result["covered_mechanics"]


def test_source_print_manager_hole_to_print_step_passes_common_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_print_manager_hole_to_print_step.json",
        body_capacity=4,
    )

    player = result["array_projection"]["steps"][0]["players"][0]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["print_manager_toggle_updates"] == 1
    assert result["array_counters"]["print_manager_toggle_rows_unhandled"] == 0
    assert result["array_counters"]["print_manager_visual_clears"] == 0
    assert result["array_counters"]["events_emitted"] == 3
    assert result["array_projection"]["steps"][0]["worldBodyCount"] == 1
    assert player["printing"] is True
    assert player["trailPointCount"] == 1
    assert player["lastTrailPoint"] == [21.6, 40.0]
    assert player["bodyCount"] == 1
    assert player["printManager"] == {
        "active": True,
        "distance": 39.0,
        "lastX": 21.6,
        "lastY": 40.0,
    }
    assert result["array_projection"]["steps"][0]["events"] == [
        {
            "event": "position",
            "player_id": "p0",
            "x": 21.6,
            "y": 40.0,
        },
        {
            "event": "point",
            "player_id": "p0",
            "x": 21.6,
            "y": 40.0,
            "important": True,
        },
        {
            "event": "property",
            "player_id": "p0",
            "property": "printing",
            "value": True,
        },
    ]
    assert (
        "hole-to-print visible trail boundary retention"
        in result["covered_mechanics"]
    )


def test_source_print_manager_exact_zero_toggle_step_passes_common_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_print_manager_exact_zero_toggle_step.json",
        body_capacity=4,
    )

    player = result["array_projection"]["steps"][0]["players"][0]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["movement_updates"] == 1
    assert result["array_counters"]["print_manager_toggle_updates"] == 1
    assert result["array_counters"]["print_manager_toggle_rows_unhandled"] == 0
    assert result["array_counters"]["print_manager_visual_clears"] == 0
    assert result["array_counters"]["events_emitted"] == 3
    assert result["array_projection"]["steps"][0]["worldBodyCount"] == 1
    assert player["x"] == 20.0
    assert player["printing"] is True
    assert player["trailPointCount"] == 1
    assert player["lastTrailPoint"] == [20.0, 40.0]
    assert player["bodyCount"] == 1
    assert player["printManager"] == {
        "active": True,
        "distance": 39.0,
        "lastX": 20.0,
        "lastY": 40.0,
    }
    assert result["array_projection"]["steps"][0]["events"] == [
        {
            "event": "position",
            "player_id": "p0",
            "x": 20.0,
            "y": 40.0,
        },
        {
            "event": "point",
            "player_id": "p0",
            "x": 20.0,
            "y": 40.0,
            "important": True,
        },
        {
            "event": "property",
            "player_id": "p0",
            "property": "printing",
            "value": True,
        },
    ]
    assert (
        "exact-zero distance threshold toggles at distance <= 0"
        in result["covered_mechanics"]
    )


def test_source_print_manager_print_to_hole_step_clears_visual_trail():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_print_manager_print_to_hole_step.json",
        body_capacity=4,
    )

    player = result["array_projection"]["steps"][0]["players"][0]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["print_manager_toggle_updates"] == 1
    assert result["array_counters"]["print_manager_toggle_rows_unhandled"] == 0
    assert result["array_counters"]["print_manager_visual_clears"] == 1
    assert result["array_counters"]["events_emitted"] == 3
    assert result["array_projection"]["steps"][0]["worldBodyCount"] == 1
    assert player["printing"] is False
    assert player["trailPointCount"] == 0
    assert player["lastTrailPoint"] is None
    assert player["bodyCount"] == 1
    assert player["printManager"] == {
        "active": True,
        "distance": 5.25,
        "lastX": 21.6,
        "lastY": 40.0,
    }
    assert result["array_projection"]["steps"][0]["events"] == [
        {
            "event": "position",
            "player_id": "p0",
            "x": 21.6,
            "y": 40.0,
        },
        {
            "event": "point",
            "player_id": "p0",
            "x": 21.6,
            "y": 40.0,
            "important": True,
        },
        {
            "event": "property",
            "player_id": "p0",
            "property": "printing",
            "value": False,
        },
    ]

    fixture = vector_compare.seed_bridge.seed_fixture(
        "scenarios/environment/source_print_manager_print_to_hole_step.json",
        body_capacity=4,
    )
    state, _counters = vector_compare.step_seeded_arrays(fixture)
    assert bool(state["has_visible_trail_last"][0, 0]) is False
    assert bool(state["has_draw_cursor"][0, 0]) is False
    assert int(state["visible_trail_count"][0, 0]) == 0
    assert int(state["body_insert_kind"][0, 0]) == vector_compare.BODY_KIND_IMPORTANT
    assert "print-to-hole visible trail and draw cursor clear" in result["covered_mechanics"]


def test_source_trail_gap_hole_space_safe_step_passes_common_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_trail_gap_hole_space_safe_step.json",
        body_capacity=4,
    )

    players = result["array_projection"]["steps"][0]["players"]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["print_manager_no_toggle_updates"] == 1
    assert result["array_counters"]["normal_points_inserted"] == 0
    assert result["array_counters"]["body_hits"] == 0
    assert result["array_counters"]["events_emitted"] == 3
    assert result["array_projection"]["steps"][0]["worldBodyCount"] == 1
    assert players[0]["alive"] is True
    assert players[0]["trailPointCount"] == 0
    assert players[1]["printing"] is False
    assert players[1]["printManager"] == {
        "active": True,
        "distance": 8.399999999999999,
        "lastX": 41.6,
        "lastY": 40.0,
    }
    assert result["array_projection"]["steps"][0]["events"] == [
        {"event": "position", "player_id": "p2", "x": 78.4, "y": 20.0},
        {"event": "position", "player_id": "p1", "x": 41.6, "y": 40.0},
        {"event": "position", "player_id": "p0", "x": 41.6, "y": 40.0},
    ]
    assert "forced hole-space endpoint remains safe without stored body" in result["covered_mechanics"]


def test_source_trail_gap_stored_body_still_kills_step_passes_common_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_trail_gap_stored_body_still_kills_step.json",
        body_capacity=4,
    )

    players = result["array_projection"]["steps"][0]["players"]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["print_manager_no_toggle_updates"] == 1
    assert result["array_counters"]["body_hits"] == 1
    assert result["array_counters"]["death_points_inserted"] == 1
    assert result["array_counters"]["events_emitted"] == 6
    assert result["array_projection"]["steps"][0]["worldBodyCount"] == 2
    assert players[0]["alive"] is False
    assert players[0]["lastTrailPoint"] == [41.6, 40.0]
    assert players[1]["alive"] is True
    assert result["array_projection"]["steps"][0]["events"][-2] == {
        "event": "die",
        "player_id": "p0",
        "killer_id": "p1",
        "old": False,
    }
    assert result["array_event_arrays"]["event_type"][0][:6] == [
        vector_compare.EVENT_POSITION,
        vector_compare.EVENT_POSITION,
        vector_compare.EVENT_POSITION,
        vector_compare.EVENT_POINT,
        vector_compare.EVENT_DIE,
        vector_compare.EVENT_SCORE_ROUND,
    ]
    assert "stored body in visual hole still kills later player" in result["covered_mechanics"]


def test_source_trail_gap_print_to_hole_boundary_kills_step_passes_common_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_trail_gap_print_to_hole_boundary_kills_step.json",
        body_capacity=4,
    )

    players = result["array_projection"]["steps"][0]["players"]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["print_manager_toggle_updates"] == 1
    assert result["array_counters"]["print_manager_visual_clears"] == 1
    assert result["array_counters"]["body_hits"] == 1
    assert result["array_counters"]["death_points_inserted"] == 1
    assert result["array_counters"]["events_emitted"] == 8
    assert result["array_projection"]["steps"][0]["worldBodyCount"] == 2
    assert players[0]["alive"] is False
    assert players[0]["lastTrailPoint"] == [41.6, 40.0]
    assert players[1]["printing"] is False
    assert players[1]["trailPointCount"] == 0
    assert players[1]["lastTrailPoint"] is None
    assert players[1]["bodyCount"] == 1
    assert players[1]["printManager"] == {
        "active": True,
        "distance": 5.25,
        "lastX": 41.6,
        "lastY": 40.0,
    }
    assert result["array_event_arrays"]["event_type"][0][:8] == [
        vector_compare.EVENT_POSITION,
        vector_compare.EVENT_POSITION,
        vector_compare.EVENT_POINT,
        vector_compare.EVENT_PROPERTY,
        vector_compare.EVENT_POSITION,
        vector_compare.EVENT_POINT,
        vector_compare.EVENT_DIE,
        vector_compare.EVENT_SCORE_ROUND,
    ]
    assert result["array_projection"]["steps"][0]["events"][2:7] == [
        {
            "event": "point",
            "player_id": "p1",
            "x": 41.6,
            "y": 40.0,
            "important": True,
        },
        {
            "event": "property",
            "player_id": "p1",
            "property": "printing",
            "value": False,
        },
        {"event": "position", "player_id": "p0", "x": 41.6, "y": 40.0},
        {
            "event": "point",
            "player_id": "p0",
            "x": 41.6,
            "y": 40.0,
            "important": False,
        },
        {
            "event": "die",
            "player_id": "p0",
            "killer_id": "p1",
            "old": False,
        },
    ]
    assert "same-frame boundary body kills later player" in result["covered_mechanics"]


def test_source_trail_gap_hole_to_print_boundary_kills_step_passes_common_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_trail_gap_hole_to_print_boundary_kills_step.json",
        body_capacity=4,
    )

    players = result["array_projection"]["steps"][0]["players"]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["print_manager_toggle_updates"] == 1
    assert result["array_counters"]["print_manager_visual_clears"] == 0
    assert result["array_counters"]["body_hits"] == 1
    assert result["array_counters"]["death_points_inserted"] == 1
    assert result["array_counters"]["events_emitted"] == 8
    assert result["array_projection"]["steps"][0]["worldBodyCount"] == 2
    assert players[0]["alive"] is False
    assert players[0]["lastTrailPoint"] == [41.6, 40.0]
    assert players[1]["printing"] is True
    assert players[1]["trailPointCount"] == 1
    assert players[1]["lastTrailPoint"] == [41.6, 40.0]
    assert players[1]["bodyCount"] == 1
    assert players[1]["printManager"] == {
        "active": True,
        "distance": 39.0,
        "lastX": 41.6,
        "lastY": 40.0,
    }
    assert result["array_event_arrays"]["event_type"][0][:8] == [
        vector_compare.EVENT_POSITION,
        vector_compare.EVENT_POSITION,
        vector_compare.EVENT_POINT,
        vector_compare.EVENT_PROPERTY,
        vector_compare.EVENT_POSITION,
        vector_compare.EVENT_POINT,
        vector_compare.EVENT_DIE,
        vector_compare.EVENT_SCORE_ROUND,
    ]
    assert result["array_projection"]["steps"][0]["events"][2:7] == [
        {
            "event": "point",
            "player_id": "p1",
            "x": 41.6,
            "y": 40.0,
            "important": True,
        },
        {
            "event": "property",
            "player_id": "p1",
            "property": "printing",
            "value": True,
        },
        {"event": "position", "player_id": "p0", "x": 41.6, "y": 40.0},
        {
            "event": "point",
            "player_id": "p0",
            "x": 41.6,
            "y": 40.0,
            "important": False,
        },
        {
            "event": "die",
            "player_id": "p0",
            "killer_id": "p1",
            "old": False,
        },
    ]
    assert "hole-to-print boundary important point inserts collision body" in result[
        "covered_mechanics"
    ]
    assert "property printing=true event emission" in result["covered_mechanics"]


def test_source_trail_gap_batch_reports_all_supported():
    summary = vector_compare.compare_inputs(
        ["scenarios/environment/source_trail_gap_batch.json"],
        body_capacity=4,
    )

    assert summary["summary"] == {
        "passed": 4,
        "failed": 0,
        "unsupported": 0,
        "status": "pass",
    }


def test_source_collision_order_batch_reports_all_supported():
    summary = vector_compare.compare_inputs(
        ["scenarios/environment/source_collision_order_batch.json"],
        body_capacity=4,
    )

    death_point, head_head = summary["fixtures"]

    assert summary["summary"] == {
        "passed": 2,
        "failed": 0,
        "unsupported": 0,
        "status": "pass",
    }
    assert death_point["scenario_id"] == "source_collision_death_point_kills_later_player_step"
    assert death_point["array_counters"]["body_hits"] == 2
    assert death_point["array_counters"]["death_points_inserted"] == 2
    assert death_point["array_counters"]["terminal_score_rows"] == 1
    assert death_point["array_projection"]["steps"][0]["events"][-1] == {
        "event": "round:end",
        "winner_id": None,
    }
    assert death_point["array_projection"]["steps"][0]["players"][0]["alive"] is False
    assert death_point["array_projection"]["steps"][0]["players"][1]["alive"] is False

    assert head_head["scenario_id"] == "source_collision_head_head_reverse_order_single_death_step"
    assert head_head["array_counters"]["normal_points_inserted"] == 2
    assert head_head["array_counters"]["body_hits"] == 1
    assert head_head["array_counters"]["terminal_score_rows"] == 1
    assert head_head["array_projection"]["steps"][0]["events"][-1] == {
        "event": "round:end",
        "winner_id": "p1",
    }
    assert head_head["array_projection"]["steps"][0]["players"][0]["alive"] is False
    assert head_head["array_projection"]["steps"][0]["players"][1]["alive"] is True


def test_source_trail_gap_natural_multistep_hole_crossing_passes_full_trace():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_trail_gap_natural_multistep_hole_crossing.json",
        body_capacity=4,
    )

    first_step, second_step, third_step = result["array_projection"]["steps"]
    crossing_player = third_step["players"][0]
    gap_player = third_step["players"][1]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["random_tape_draws"] == 2
    assert result["array_counters_by_step"][0]["random_tape_draws"] == 1
    assert result["array_counters_by_step"][1]["random_tape_draws"] == 0
    assert result["array_counters_by_step"][2]["random_tape_draws"] == 1
    assert result["array_counters"]["print_manager_toggle_updates"] == 2
    assert result["array_counters"]["print_manager_no_toggle_updates"] == 1
    assert result["array_counters"]["print_manager_toggle_rows_unhandled"] == 0
    assert result["array_counters"]["normal_points_inserted"] == 2
    assert result["array_counters"]["body_hits"] == 0
    assert result["array_counters"]["events_emitted"] == 15
    assert first_step["players"][1]["printManager"]["distance"] == 18.0
    assert second_step["players"][1]["printManager"]["distance"] == 2.0
    assert third_step["worldBodyCount"] == 4
    assert crossing_player["alive"] is True
    assert crossing_player["x"] == 66.0
    assert crossing_player["y"] == 40.0
    assert gap_player["printing"] is False
    assert gap_player["trailPointCount"] == 0
    assert gap_player["lastTrailPoint"] is None
    assert gap_player["bodyNum"] == 2
    assert gap_player["bodyCount"] == 4
    assert gap_player["printManager"] == {
        "active": True,
        "distance": 5.25,
        "lastX": 58.0,
        "lastY": 40.0,
    }
    assert [event["event"] for event in third_step["events"]] == [
        "position",
        "position",
        "point",
        "point",
        "property",
        "position",
    ]
    assert third_step["events"][0]["player_id"] == "p2"
    assert third_step["events"][0]["x"] == 62.0
    assert third_step["events"][0]["y"] == pytest.approx(20.0)
    assert third_step["events"][2:5] == [
        {
            "event": "point",
            "player_id": "p1",
            "x": 58.0,
            "y": 40.0,
            "important": False,
        },
        {
            "event": "point",
            "player_id": "p1",
            "x": 58.0,
            "y": 40.0,
            "important": True,
        },
        {
            "event": "property",
            "player_id": "p1",
            "property": "printing",
            "value": False,
        },
    ]
    assert "row-local Math.random tape for PrintManager distances" in result[
        "covered_mechanics"
    ]


def test_natural_trail_gap_seed_exposes_row_local_random_tape_arrays():
    fixture = vector_compare.seed_bridge.seed_fixture(
        "scenarios/environment/source_trail_gap_natural_multistep_hole_crossing.json",
        body_capacity=4,
    )
    state = vector_compare.array_state_from_seed(fixture)

    np.testing.assert_allclose(state["random_tape_values"][0, :2], [0.0, 0.5])
    assert int(state["random_tape_length"][0]) == 2
    assert int(state["random_tape_cursor"][0]) == 0
    assert int(state["random_tape_draw_count"][0]) == 0

    vector_compare.step_prepared_arrays(
        state,
        vector_compare.prepare_fixture_array_step(fixture, step_index=0),
    )
    assert int(state["random_tape_cursor"][0]) == 1
    assert int(state["random_tape_draw_count"][0]) == 1
    assert state["print_manager_distance"][0, 1] == 18.0

    vector_compare.step_prepared_arrays(
        state,
        vector_compare.prepare_fixture_array_step(fixture, step_index=1),
    )
    assert int(state["random_tape_cursor"][0]) == 1
    assert int(state["random_tape_draw_count"][0]) == 1
    assert state["print_manager_distance"][0, 1] == 2.0

    vector_compare.step_prepared_arrays(
        state,
        vector_compare.prepare_fixture_array_step(fixture, step_index=2),
    )
    assert int(state["random_tape_cursor"][0]) == 2
    assert int(state["random_tape_draw_count"][0]) == 2
    assert bool(state["random_tape_exhausted"][0]) is False
    assert state["print_manager_distance"][0, 1] == 5.25


@pytest.mark.parametrize(
    ("path", "length", "exceeds_capacity", "sites", "avatars", "expected_call_at_ms"),
    (
        (
            "scenarios/environment/source_lifecycle_spawn_rng_warmup_print_start_2p.json",
            8,
            False,
            [
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "print_manager.start_distance",
                "print_manager.start_distance",
            ],
            [2, 2, 2, 1, 1, 1, 2, 1],
            [None] * 8,
        ),
        (
            "scenarios/environment/source_lifecycle_spawn_rng_2p_next_round.json",
            16,
            True,
            [
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "print_manager.start_distance",
                "print_manager.start_distance",
                "print_manager.stop_distance",
                "print_manager.stop_distance",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
            ],
            [2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1],
            [None] * 16,
        ),
        (
            "scenarios/environment/source_lifecycle_spawn_heading_rejection_retry_2p.json",
            7,
            False,
            [
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "spawn.angle_attempt_1",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
            ],
            [2, 2, 2, 2, 1, 1, 1],
            [None] * 7,
        ),
        (
            "scenarios/environment/source_lifecycle_spawn_rng_order_3p.json",
            9,
            True,
            [
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
            ],
            [3, 3, 3, 2, 2, 2, 1, 1, 1],
            [None] * 9,
        ),
        (
            "scenarios/environment/source_lifecycle_spawn_rng_warmup_print_start_3p.json",
            12,
            True,
            [
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "print_manager.start_distance",
                "print_manager.start_distance",
                "print_manager.start_distance",
            ],
            [3, 3, 3, 2, 2, 2, 1, 1, 1, 3, 2, 1],
            [None, None, None, None, None, None, None, None, None, 3000, 3000, 3000],
        ),
        (
            "scenarios/environment/source_lifecycle_spawn_rng_3p_next_round.json",
            24,
            True,
            [
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "print_manager.start_distance",
                "print_manager.start_distance",
                "print_manager.start_distance",
                "print_manager.stop_distance",
                "print_manager.stop_distance",
                "print_manager.stop_distance",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
            ],
            [3, 3, 3, 2, 2, 2, 1, 1, 1, 3, 2, 1, 3, 2, 1, 3, 3, 3, 2, 2, 2, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3000, 3000, 3000, 3000, 3000, 3000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000],
        ),
        (
            "scenarios/environment/source_lifecycle_spawn_rng_order_4p.json",
            12,
            True,
            [
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
            ],
            [4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1],
            [None] * 12,
        ),
        (
            "scenarios/environment/source_lifecycle_present_absent_3p_round_new.json",
            6,
            False,
            [
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
            ],
            [3, 3, 3, 1, 1, 1],
            [None] * 6,
        ),
        (
            "scenarios/environment/source_lifecycle_match_end_at_max_score_2p.json",
            9,
            True,
            [
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "spawn.position_x",
                "spawn.position_y",
                "spawn.angle_attempt_0",
                "print_manager.start_distance",
                "print_manager.start_distance",
                "print_manager.stop_distance",
            ],
            [2, 2, 2, 1, 1, 1, 2, 1, 2],
            [0, 0, 0, 0, 0, 0, 3000, 3000, 3000],
        ),
    ),
)
def test_natural_lifecycle_fixtures_are_rejected_with_rng_contract(
    path,
    length,
    exceeds_capacity,
    sites,
    avatars,
    expected_call_at_ms,
):
    support = vector_compare.seed_bridge.fixture_seed_support(path)

    assert support["schema"] == "curvyzero_vector_fixture_seed_support/v1"
    assert support["supported"] is False
    assert support["unsupported_kind"] == "natural_game_new_round_lifecycle"
    assert "natural Game.newRound()" in support["reason"]

    rejection = support["lifecycle_rejection"]
    assert rejection["schema"] == "curvyzero_vector_lifecycle_seed_rejection/v1"
    assert rejection["ordinary_seed_shape"] == "initial_state_one_step"
    assert (
        rejection["required_shape"]
        == "reset_many -> spawn_rng -> timer_setup -> final_obs_replay"
    )
    assert "spawn RNG draw order, site, avatar, and retry metadata" in rejection[
        "missing_vector_contract"
    ]

    random_contract = support["random_tape_contract"]
    assert random_contract["schema"] == "curvyzero_vector_random_tape_contract/v1"
    assert random_contract["capacity"] == 8
    assert random_contract["length"] == length
    assert random_contract["minimum_capacity"] == max(8, length)
    assert random_contract["exceeds_seed_capacity"] is exceeds_capacity
    assert random_contract["expected_call_count"] == length
    assert random_contract["expected_call_indices"] == list(range(length))
    assert random_contract["expected_call_sites"] == sites
    assert random_contract["expected_call_avatars"] == avatars
    assert random_contract["expected_call_at_ms"] == expected_call_at_ms
    assert random_contract["metadata_matches_tape"] is True

    with pytest.raises(
        vector_compare.seed_bridge.SeedError,
        match="natural Game.newRound",
    ) as exc:
        vector_compare.seed_bridge.seed_fixture(path, body_capacity=4)
    assert exc.value.detail["scenario_id"] == support["scenario_id"]
    assert exc.value.detail["random_tape_contract"]["expected_call_sites"] == sites


def test_empty_random_tape_default_distance_counts_draw_without_cursor_advance():
    fixture = vector_compare.seed_bridge.seed_fixture(
        "scenarios/environment/source_print_manager_print_to_hole_step.json",
        body_capacity=4,
    )
    state = vector_compare.array_state_from_seed(fixture)

    assert int(state["random_tape_length"][0]) == 0
    assert int(state["random_tape_cursor"][0]) == 0
    assert int(state["random_tape_draw_count"][0]) == 0

    counters = vector_compare.step_prepared_arrays(
        state,
        vector_compare.prepare_fixture_array_step(fixture),
    )

    assert counters["random_tape_draws"] == 1
    assert int(state["random_tape_cursor"][0]) == 0
    assert int(state["random_tape_draw_count"][0]) == 1
    assert bool(state["random_tape_exhausted"][0]) is False


def test_body_wrap_wall_print_manager_trail_gap_run_reports_expected_counts():
    summary = vector_compare.compare_inputs(
        [
            "scenarios/environment/source_body_canary_batch.json",
            "scenarios/environment/source_borderless_wrap_step.json",
            "scenarios/environment/source_normal_wall_death_step.json",
            "scenarios/environment/source_print_manager_print_to_hole_step.json",
            "scenarios/environment/source_print_manager_no_toggle_control_step.json",
            "scenarios/environment/source_print_manager_hole_to_print_step.json",
            "scenarios/environment/source_print_manager_exact_zero_toggle_step.json",
            "scenarios/environment/source_print_manager_active_stop_on_death_step.json",
            "scenarios/environment/source_print_manager_active_hole_stop_on_death_step.json",
            "scenarios/environment/source_print_manager_body_collision_stop_on_death_step.json",
            "scenarios/environment/source_trail_gap_batch.json",
        ],
        body_capacity=4,
    )

    assert summary["summary"] == {
        "passed": 19,
        "failed": 0,
        "unsupported": 0,
        "status": "pass",
    }


def test_prepared_array_step_matches_seeded_step_helper_and_can_reset_state():
    fixture = vector_compare.seed_bridge.seed_fixture(
        "scenarios/environment/source_body_opponent_overlap_kills_step.json",
        body_capacity=4,
    )
    initial_state = vector_compare.array_state_from_seed(fixture)
    working_state = vector_compare.copy_array_state(initial_state)
    prepared_step = vector_compare.prepare_fixture_array_step(fixture)

    prepared_counters = vector_compare.step_prepared_arrays(working_state, prepared_step)
    helper_state, helper_counters = vector_compare.step_seeded_arrays(fixture)

    assert prepared_counters == helper_counters
    assert vector_compare.project_array_state_to_common_trace(
        fixture,
        working_state,
        step_index=0,
    ) == vector_compare.project_array_state_to_common_trace(
        fixture,
        helper_state,
        step_index=0,
    )

    vector_compare.reset_array_state(working_state, initial_state)

    assert int(working_state["tick"][0]) == 0
    assert bool(working_state["alive"][0, 0]) is True
    assert int(working_state["world_body_count"][0]) == 1


def test_reset_array_rows_copies_only_masked_rows_from_fixture_seed_state():
    fixture = vector_compare.seed_bridge.seed_fixture(
        "scenarios/environment/source_body_opponent_overlap_kills_step.json",
        body_capacity=4,
    )
    seeded_state = vector_compare.array_state_from_seed(fixture)
    source = {
        name: np.concatenate([array, array.copy()], axis=0)
        for name, array in seeded_state.items()
    }
    target = vector_compare.copy_array_state(source)

    for array in target.values():
        if array.dtype == bool:
            array[0] = ~array[0]
            array[1] = ~array[1]
        else:
            array[0] += 7
            array[1] += 7
    mutated_row_zero = {name: array[0].copy() for name, array in target.items()}

    reset_count = vector_compare.reset_array_rows(
        target,
        source,
        np.asarray([False, True], dtype=bool),
    )

    assert reset_count == 1
    for name, source_array in source.items():
        np.testing.assert_array_equal(target[name][0], mutated_row_zero[name])
        np.testing.assert_array_equal(target[name][1], source_array[1])


def test_reset_array_rows_rejects_key_mismatch():
    fixture = vector_compare.seed_bridge.seed_fixture(
        "scenarios/environment/source_body_opponent_overlap_kills_step.json",
        body_capacity=4,
    )
    source = vector_compare.array_state_from_seed(fixture)
    target = vector_compare.copy_array_state(source)
    source_without_tick = {
        name: array for name, array in source.items() if name != "tick"
    }

    with pytest.raises(vector_compare.VectorCompareError, match="keys must match"):
        vector_compare.reset_array_rows(
            target,
            source_without_tick,
            np.asarray([True], dtype=bool),
        )


def test_reset_array_rows_rejects_shape_mismatch():
    fixture = vector_compare.seed_bridge.seed_fixture(
        "scenarios/environment/source_body_opponent_overlap_kills_step.json",
        body_capacity=4,
    )
    source = vector_compare.array_state_from_seed(fixture)
    target = vector_compare.copy_array_state(source)
    mismatched_source = dict(source)
    mismatched_source["tick"] = np.concatenate(
        [source["tick"], source["tick"]],
        axis=0,
    )

    with pytest.raises(vector_compare.VectorCompareError, match="shape differs"):
        vector_compare.reset_array_rows(
            target,
            mismatched_source,
            np.asarray([True], dtype=bool),
        )


@pytest.mark.parametrize(
    "reset_mask",
    (
        np.asarray([1], dtype=np.int8),
        np.asarray([[True]], dtype=bool),
    ),
)
def test_reset_array_rows_rejects_non_bool_or_non_row_mask(reset_mask):
    fixture = vector_compare.seed_bridge.seed_fixture(
        "scenarios/environment/source_body_opponent_overlap_kills_step.json",
        body_capacity=4,
    )
    source = vector_compare.array_state_from_seed(fixture)
    target = vector_compare.copy_array_state(source)

    with pytest.raises(vector_compare.VectorCompareError, match="reset_mask"):
        vector_compare.reset_array_rows(target, source, reset_mask)


def test_reset_array_rows_rejects_mask_batch_mismatch():
    fixture = vector_compare.seed_bridge.seed_fixture(
        "scenarios/environment/source_body_opponent_overlap_kills_step.json",
        body_capacity=4,
    )
    source = vector_compare.array_state_from_seed(fixture)
    target = vector_compare.copy_array_state(source)

    with pytest.raises(vector_compare.VectorCompareError, match="leading reset_mask"):
        vector_compare.reset_array_rows(
            target,
            source,
            np.asarray([True, False], dtype=bool),
        )


def test_reset_array_rows_with_info_snapshots_terminal_transition_before_row_reset():
    fixture = vector_compare.seed_bridge.seed_fixture(
        "scenarios/environment/source_body_opponent_overlap_kills_step.json",
        body_capacity=4,
    )
    seeded_state = vector_compare.array_state_from_seed(fixture)
    source = {
        name: np.concatenate([array, array.copy()], axis=0)
        for name, array in seeded_state.items()
    }
    source["episode_id"] = np.asarray([10, 21], dtype=np.int64)
    source["reset_seed"] = np.asarray([101, 202], dtype=np.uint64)
    source["reset_source"] = np.asarray(
        [vector_compare.RESET_SOURCE_FIXTURE, vector_compare.RESET_SOURCE_FIXTURE],
        dtype=np.int16,
    )
    target = vector_compare.copy_array_state(source)
    target["episode_id"] = np.asarray([9, 20], dtype=np.int64)
    target["reset_seed"] = np.asarray([100, 200], dtype=np.uint64)
    target["reset_source"] = np.asarray(
        [vector_compare.RESET_SOURCE_MANUAL, vector_compare.RESET_SOURCE_MANUAL],
        dtype=np.int16,
    )
    target["done"][1] = True
    target["tick"][1] = 42
    target["pos"][1, 0] = [123.0, 456.0]
    target["event_count"][1] = 2
    target["event_mask"][1, :2] = True
    target["event_type"][1, :2] = [
        vector_compare.EVENT_DIE,
        vector_compare.EVENT_ROUND_END,
    ]

    info = vector_compare.reset_array_rows_with_info(
        target,
        source,
        np.asarray([False, True], dtype=bool),
        reset_source=vector_compare.RESET_SOURCE_AUTORESET,
        snapshot_array_names=("done", "tick", "pos", "event_count", "event_type"),
    )

    assert info["schema"] == vector_compare.RESET_INFO_SCHEMA_ID
    assert info["reset_schema_id"] == vector_compare.RESET_INFO_SCHEMA_ID
    assert info["rules_schema_id"] == vector_compare.RULES_SCHEMA_ID
    assert info["state_schema_id"] == vector_compare.VECTOR_STATE_SCHEMA_ID
    assert info["reset_count"] == 1
    np.testing.assert_array_equal(
        info["reset_mask"],
        np.asarray([False, True], dtype=bool),
    )
    np.testing.assert_array_equal(info["reset_rows"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(
        info["reset_episode_id"],
        np.asarray([9, 21], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        info["reset_seed"],
        np.asarray([100, 202], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        info["reset_source"],
        np.asarray(
            [
                vector_compare.RESET_SOURCE_MANUAL,
                vector_compare.RESET_SOURCE_AUTORESET,
            ],
            dtype=np.int16,
        ),
    )

    snapshot = info["terminal_transition_snapshot"]
    assert snapshot["schema"] == vector_compare.TERMINAL_TRANSITION_SNAPSHOT_SCHEMA_ID
    np.testing.assert_array_equal(
        snapshot["final_mask"],
        np.asarray([False, True], dtype=bool),
    )
    np.testing.assert_array_equal(snapshot["final_rows"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(snapshot["arrays"]["done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(snapshot["arrays"]["tick"], np.asarray([42], dtype=np.int32))
    np.testing.assert_allclose(snapshot["arrays"]["pos"][0, 0], [123.0, 456.0])
    np.testing.assert_array_equal(
        snapshot["arrays"]["event_count"],
        np.asarray([2], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        snapshot["arrays"]["event_type"][0, :2],
        np.asarray([vector_compare.EVENT_DIE, vector_compare.EVENT_ROUND_END], dtype=np.int16),
    )

    assert bool(target["done"][1]) is False
    assert int(target["tick"][1]) == int(source["tick"][1])
    assert int(target["event_count"][1]) == 0
    np.testing.assert_array_equal(target["event_type"][1], source["event_type"][1])
    np.testing.assert_allclose(target["pos"][1], source["pos"][1])

    target["pos"][1, 0] = [0.0, 0.0]
    snapshot["arrays"]["event_count"][0] = 99
    np.testing.assert_allclose(snapshot["arrays"]["pos"][0, 0], [123.0, 456.0])
    assert int(target["event_count"][1]) == 0


def _two_row_reset_template_and_target():
    fixture = vector_compare.seed_bridge.seed_fixture(
        "scenarios/environment/source_body_opponent_overlap_kills_step.json",
        body_capacity=4,
    )
    seeded_state = vector_compare.array_state_from_seed(fixture)
    source = {
        name: np.concatenate([array, array.copy()], axis=0)
        for name, array in seeded_state.items()
    }
    target = vector_compare.copy_array_state(source)

    source.update(
        {
            "episode_id": np.asarray([100, 900], dtype=np.int64),
            "episode_step": np.asarray([0, 88], dtype=np.int32),
            "env_active": np.asarray([True, False], dtype=bool),
            "reset_pending": np.asarray([False, True], dtype=bool),
            "terminated": np.asarray([False, True], dtype=bool),
            "truncated": np.asarray([False, False], dtype=bool),
            "terminal_reason": np.asarray(
                [
                    vector_compare.TERMINAL_REASON_NONE,
                    vector_compare.TERMINAL_REASON_ALL_DEAD_DRAW,
                ],
                dtype=np.int16,
            ),
            "reset_seed": np.asarray([1001, 9001], dtype=np.uint64),
            "reset_source": np.asarray(
                [
                    vector_compare.RESET_SOURCE_FIXTURE,
                    vector_compare.RESET_SOURCE_REPLAY,
                ],
                dtype=np.int16,
            ),
        }
    )
    target.update(
        {
            "episode_id": np.asarray([10, 20], dtype=np.int64),
            "episode_step": np.asarray([3, 7], dtype=np.int32),
            "env_active": np.asarray([True, False], dtype=bool),
            "reset_pending": np.asarray([False, True], dtype=bool),
            "terminated": np.asarray([False, True], dtype=bool),
            "truncated": np.asarray([False, False], dtype=bool),
            "terminal_reason": np.asarray(
                [
                    vector_compare.TERMINAL_REASON_NONE,
                    vector_compare.TERMINAL_REASON_SURVIVOR_WIN,
                ],
                dtype=np.int16,
            ),
            "reset_seed": np.asarray([111, 222], dtype=np.uint64),
            "reset_source": np.asarray(
                [
                    vector_compare.RESET_SOURCE_REPLAY,
                    vector_compare.RESET_SOURCE_MANUAL,
                ],
                dtype=np.int16,
            ),
        }
    )
    return source, target


def test_reset_arrays_snapshots_then_stamps_next_episode_lifecycle_metadata():
    source, target = _two_row_reset_template_and_target()
    target["done"][1] = True
    target["tick"][1] = 41
    target["event_count"][1] = 2
    target["event_mask"][1, :2] = True
    target["event_type"][1, :2] = [
        vector_compare.EVENT_DIE,
        vector_compare.EVENT_ROUND_END,
    ]
    source["done"][1] = True
    source["tick"][1] = 99
    source["event_count"][1] = 3
    source["event_type"][1, :3] = [
        vector_compare.EVENT_POSITION,
        vector_compare.EVENT_POINT,
        vector_compare.EVENT_PROPERTY,
    ]

    info = vector_compare.reset_arrays(
        target,
        source,
        np.asarray([False, True], dtype=bool),
        reset_seed=555,
        reset_source=vector_compare.RESET_SOURCE_AUTORESET,
        snapshot_array_names=(
            "done",
            "tick",
            "episode_id",
            "episode_step",
            "event_count",
            "event_type",
        ),
    )

    assert info["schema"] == "curvyzero_vector_reset_info/v1"
    assert info["reset_schema_id"] == "curvyzero_vector_reset_info/v1"
    assert info["rules_schema_id"] == "curvyzero_source_fixture_rules/v1"
    assert info["state_schema_id"] == "curvyzero_vector_fixture_state/v1"
    assert (
        info["terminal_transition_snapshot"]["schema"]
        == "curvyzero_vector_terminal_transition_snapshot/v1"
    )
    assert info["reset_count"] == 1
    np.testing.assert_array_equal(
        info["reset_rows"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        info["reset_episode_id"],
        np.asarray([10, 21], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        info["reset_seed"],
        np.asarray([111, 555], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        info["reset_source"],
        np.asarray(
            [
                vector_compare.RESET_SOURCE_REPLAY,
                vector_compare.RESET_SOURCE_AUTORESET,
            ],
            dtype=np.int16,
        ),
    )

    snapshot = info["terminal_transition_snapshot"]
    np.testing.assert_array_equal(snapshot["arrays"]["done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(snapshot["arrays"]["tick"], np.asarray([41], dtype=np.int32))
    np.testing.assert_array_equal(
        snapshot["arrays"]["episode_id"],
        np.asarray([20], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        snapshot["arrays"]["episode_step"],
        np.asarray([7], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        snapshot["arrays"]["event_count"],
        np.asarray([2], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        snapshot["arrays"]["event_type"][0, :2],
        np.asarray([vector_compare.EVENT_DIE, vector_compare.EVENT_ROUND_END], dtype=np.int16),
    )

    assert int(target["episode_id"][1]) == 21
    assert int(target["episode_step"][1]) == 0
    assert bool(target["env_active"][1]) is True
    assert bool(target["reset_pending"][1]) is False
    assert bool(target["done"][1]) is False
    assert bool(target["terminated"][1]) is False
    assert bool(target["truncated"][1]) is False
    assert int(target["terminal_reason"][1]) == vector_compare.TERMINAL_REASON_NONE
    assert int(target["reset_seed"][1]) == 555
    assert int(target["reset_source"][1]) == vector_compare.RESET_SOURCE_AUTORESET
    assert int(target["tick"][1]) == 0
    assert int(target["event_count"][1]) == 0
    np.testing.assert_array_equal(target["event_type"][1], np.zeros(16, dtype=np.int16))

    assert int(target["episode_id"][0]) == 10
    assert int(target["reset_seed"][0]) == 111
    assert int(target["reset_source"][0]) == vector_compare.RESET_SOURCE_REPLAY

    target["event_count"][1] = 99
    np.testing.assert_array_equal(
        snapshot["arrays"]["event_count"],
        np.asarray([2], dtype=np.int16),
    )


def test_reset_many_snapshots_random_tape_then_copies_selected_row_template_metadata():
    source, target = _two_row_reset_template_and_target()
    source["random_tape_values"][1, :4] = [0.1, 0.2, 0.3, 0.4]
    source["random_tape_length"][1] = 4
    source["random_tape_cursor"][1] = 0
    source["random_tape_exhausted"][1] = False
    source["random_tape_draw_count"][1] = 0

    target["random_tape_values"][0, :3] = [0.11, 0.22, 0.33]
    target["random_tape_length"][0] = 3
    target["random_tape_cursor"][0] = 1
    target["random_tape_exhausted"][0] = False
    target["random_tape_draw_count"][0] = 4
    source["random_tape_values"][0, :3] = [0.44, 0.55, 0.66]
    source["random_tape_length"][0] = 6
    source["random_tape_cursor"][0] = 5
    source["random_tape_exhausted"][0] = True
    source["random_tape_draw_count"][0] = 9

    target["random_tape_values"][1, :4] = [0.8, 0.7, 0.6, 0.5]
    target["random_tape_length"][1] = 2
    target["random_tape_cursor"][1] = 2
    target["random_tape_exhausted"][1] = True
    target["random_tape_draw_count"][1] = 6

    skipped_before = {
        name: target[name][0].copy()
        for name in (
            "random_tape_values",
            "random_tape_length",
            "random_tape_cursor",
            "random_tape_exhausted",
            "random_tape_draw_count",
        )
    }

    info = vector_compare.reset_many(
        target,
        source,
        np.asarray([False, True], dtype=bool),
        reset_seed=555,
        reset_source=vector_compare.RESET_SOURCE_AUTORESET,
        snapshot_array_names=(
            "random_tape_values",
            "random_tape_length",
            "random_tape_cursor",
            "random_tape_exhausted",
            "random_tape_draw_count",
        ),
    )

    snapshot = info["terminal_transition_snapshot"]["arrays"]
    np.testing.assert_allclose(snapshot["random_tape_values"][0, :4], [0.8, 0.7, 0.6, 0.5])
    np.testing.assert_array_equal(
        snapshot["random_tape_length"],
        np.asarray([2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        snapshot["random_tape_cursor"],
        np.asarray([2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        snapshot["random_tape_exhausted"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        snapshot["random_tape_draw_count"],
        np.asarray([6], dtype=np.int32),
    )

    for name, before in skipped_before.items():
        np.testing.assert_array_equal(target[name][0], before)

    np.testing.assert_allclose(target["random_tape_values"][1, :4], [0.1, 0.2, 0.3, 0.4])
    assert int(target["random_tape_length"][1]) == 4
    assert int(target["random_tape_cursor"][1]) == int(source["random_tape_cursor"][1])
    assert bool(target["random_tape_exhausted"][1]) is bool(source["random_tape_exhausted"][1])
    assert int(target["random_tape_draw_count"][1]) == int(source["random_tape_draw_count"][1])


def test_reset_many_carries_selected_row_lifecycle_rng_template_metadata():
    source, target = _two_row_reset_template_and_target()
    source.update(
        {
            "lifecycle_rng_call_index": np.asarray(
                [[0, 1, 2, -1], [0, 1, 2, 3]],
                dtype=np.int32,
            ),
            "lifecycle_rng_call_site_code": np.asarray(
                [[1, 2, 3, 0], [1, 2, 3, 4]],
                dtype=np.int16,
            ),
            "lifecycle_rng_call_avatar": np.asarray(
                [[2, 2, 2, -1], [2, 2, 2, 2]],
                dtype=np.int16,
            ),
            "lifecycle_rng_call_value": np.asarray(
                [[0.32, 0.5, 0.01, 0.0], [0.0, 0.5, 0.5, 0.25]],
                dtype=np.float64,
            ),
            "lifecycle_rng_call_at_ms": np.asarray(
                [[0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
        }
    )
    target.update(
        {
            name: array.copy()
            for name, array in source.items()
            if name in vector_compare.LIFECYCLE_RNG_TEMPLATE_ARRAY_NAMES
        }
    )
    for name in vector_compare.LIFECYCLE_RNG_TEMPLATE_ARRAY_NAMES:
        target[name][0] += 10
        target[name][1] += 20
    skipped_before = {
        name: target[name][0].copy()
        for name in vector_compare.LIFECYCLE_RNG_TEMPLATE_ARRAY_NAMES
    }
    selected_before = {
        name: target[name][1].copy()
        for name in vector_compare.LIFECYCLE_RNG_TEMPLATE_ARRAY_NAMES
    }

    info = vector_compare.reset_many(
        target,
        source,
        np.asarray([False, True], dtype=bool),
        reset_seed=555,
        reset_source=vector_compare.RESET_SOURCE_AUTORESET,
        snapshot_array_names=vector_compare.LIFECYCLE_RNG_TEMPLATE_ARRAY_NAMES,
    )

    snapshot = info["terminal_transition_snapshot"]["arrays"]
    for name in vector_compare.LIFECYCLE_RNG_TEMPLATE_ARRAY_NAMES:
        np.testing.assert_array_equal(target[name][0], skipped_before[name])
        np.testing.assert_array_equal(target[name][1], source[name][1])
        np.testing.assert_array_equal(snapshot[name][0], selected_before[name])


@pytest.mark.parametrize(
    ("mutate", "kwargs", "match"),
    (
        (
            lambda source, target: target.pop("episode_id"),
            {},
            "missing 'episode_id'",
        ),
        (
            lambda source, target: target.__setitem__(
                "episode_step",
                np.asarray([0, 1], dtype=np.int64),
            ),
            {},
            "episode_step must be an int32 array",
        ),
        (
            lambda source, target: target.__setitem__(
                "terminal_reason",
                np.asarray([vector_compare.TERMINAL_REASON_NONE, 99], dtype=np.int16),
            ),
            {},
            "known terminal reason",
        ),
        (
            lambda source, target: None,
            {"reset_seed": -1},
            "reset_seed scalar must be non-negative",
        ),
        (
            lambda source, target: None,
            {"reset_source": np.asarray([vector_compare.RESET_SOURCE_MANUAL, 99], dtype=np.int16)},
            "known reset source",
        ),
    ),
)
def test_reset_arrays_rejects_invalid_lifecycle_or_reset_metadata(mutate, kwargs, match):
    source, target = _two_row_reset_template_and_target()
    mutate(source, target)

    reset_kwargs = {
        "reset_seed": 123,
        "reset_source": vector_compare.RESET_SOURCE_MANUAL,
        **kwargs,
    }
    with pytest.raises(vector_compare.VectorCompareError, match=match):
        vector_compare.reset_arrays(
            target,
            source,
            np.asarray([False, True], dtype=bool),
            snapshot_array_names=("tick",),
            **reset_kwargs,
        )

    if "tick" in target:
        assert int(target["tick"][1]) == 0


def test_reset_many_wraps_selected_row_reset_contract():
    source, target = _two_row_reset_template_and_target()
    target["done"][1] = True
    target["terminated"][1] = True
    target["terminal_reason"][1] = vector_compare.TERMINAL_REASON_SURVIVOR_WIN
    target["tick"][1] = 41
    target["pos"][1, 0] = [123.0, 456.0]
    target["event_count"][1] = 2
    target["event_mask"][1, :2] = True
    target["event_type"][1, :2] = [
        vector_compare.EVENT_DIE,
        vector_compare.EVENT_ROUND_END,
    ]
    target["event_player"][1, :2] = [0, -1]
    target["event_other"][1, :2] = [1, 1]
    target["event_bool"][1, :2] = [1, 0]
    target["event_value_i"][1, :2] = [[7, 8], [9, 10]]
    target["event_value_f"][1, :2] = [[1.25, 2.5], [3.75, 5.0]]
    target["event_overflow"][1] = True
    target["event_overflow_attempts"][1] = 4
    source["done"][1] = True
    source["event_count"][1] = 1
    source["event_mask"][1, 0] = True
    source["event_type"][1, 0] = vector_compare.EVENT_POSITION

    skipped_before = {name: array[0].copy() for name, array in target.items()}

    info = vector_compare.reset_many(
        target,
        source,
        np.asarray([False, True], dtype=bool),
        reset_seed=555,
        reset_source=vector_compare.RESET_SOURCE_AUTORESET,
        snapshot_array_names=(
            "done",
            "terminated",
            "truncated",
            "terminal_reason",
            "tick",
            "pos",
            "episode_id",
            "episode_step",
            "reset_seed",
            "reset_source",
            "event_count",
            "event_type",
        ),
    )

    assert info["reset_count"] == 1
    np.testing.assert_array_equal(
        info["reset_mask"],
        np.asarray([False, True], dtype=bool),
    )
    np.testing.assert_array_equal(info["reset_rows"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(
        info["reset_episode_id"],
        np.asarray([10, 21], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        info["reset_seed"],
        np.asarray([111, 555], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        info["reset_source"],
        np.asarray(
            [
                vector_compare.RESET_SOURCE_REPLAY,
                vector_compare.RESET_SOURCE_AUTORESET,
            ],
            dtype=np.int16,
        ),
    )

    snapshot = info["terminal_transition_snapshot"]
    np.testing.assert_array_equal(
        snapshot["final_mask"],
        np.asarray([False, True], dtype=bool),
    )
    np.testing.assert_array_equal(snapshot["final_rows"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(snapshot["arrays"]["done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        snapshot["arrays"]["terminated"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        snapshot["arrays"]["terminal_reason"],
        np.asarray([vector_compare.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(snapshot["arrays"]["tick"], np.asarray([41], dtype=np.int32))
    np.testing.assert_allclose(snapshot["arrays"]["pos"][0, 0], [123.0, 456.0])
    np.testing.assert_array_equal(
        snapshot["arrays"]["episode_id"],
        np.asarray([20], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        snapshot["arrays"]["episode_step"],
        np.asarray([7], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        snapshot["arrays"]["reset_seed"],
        np.asarray([222], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        snapshot["arrays"]["reset_source"],
        np.asarray([vector_compare.RESET_SOURCE_MANUAL], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        snapshot["arrays"]["event_count"],
        np.asarray([2], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        snapshot["arrays"]["event_type"][0, :2],
        np.asarray([vector_compare.EVENT_DIE, vector_compare.EVENT_ROUND_END], dtype=np.int16),
    )

    for name, before in skipped_before.items():
        np.testing.assert_array_equal(target[name][0], before)

    np.testing.assert_allclose(target["pos"][1], source["pos"][1])
    assert int(target["episode_id"][1]) == 21
    assert int(target["episode_step"][1]) == 0
    assert bool(target["env_active"][1]) is True
    assert bool(target["reset_pending"][1]) is False
    assert bool(target["done"][1]) is False
    assert bool(target["terminated"][1]) is False
    assert bool(target["truncated"][1]) is False
    assert int(target["terminal_reason"][1]) == vector_compare.TERMINAL_REASON_NONE
    assert int(target["reset_seed"][1]) == 555
    assert int(target["reset_source"][1]) == vector_compare.RESET_SOURCE_AUTORESET
    assert int(target["tick"][1]) == 0
    assert int(target["event_count"][1]) == 0
    np.testing.assert_array_equal(target["event_mask"][1], np.zeros(16, dtype=bool))
    np.testing.assert_array_equal(target["event_type"][1], np.zeros(16, dtype=np.int16))
    np.testing.assert_array_equal(target["event_player"][1], np.full(16, -1, dtype=np.int16))
    np.testing.assert_array_equal(target["event_other"][1], np.full(16, -1, dtype=np.int16))
    np.testing.assert_array_equal(target["event_bool"][1], np.full(16, -1, dtype=np.int8))
    np.testing.assert_array_equal(target["event_value_i"][1], np.zeros((16, 2), dtype=np.int32))
    np.testing.assert_array_equal(target["event_value_f"][1], np.zeros((16, 2), dtype=np.float64))
    assert bool(target["event_overflow"][1]) is False
    assert int(target["event_overflow_attempts"][1]) == 0


@pytest.mark.parametrize(
    ("reset_mask", "match"),
    (
        (np.asarray([1, 0], dtype=np.int8), "reset_mask must be a bool array"),
        (np.asarray([[False, True]], dtype=bool), "reset_mask must be a bool array"),
        (np.asarray([False, True, False], dtype=bool), "reset_mask shape"),
    ),
)
def test_reset_many_rejects_invalid_masks(reset_mask, match):
    source, target = _two_row_reset_template_and_target()

    with pytest.raises(vector_compare.VectorCompareError, match=match):
        vector_compare.reset_many(
            target,
            source,
            reset_mask,
            reset_seed=123,
            reset_source=vector_compare.RESET_SOURCE_AUTORESET,
            snapshot_array_names=("tick",),
        )


def test_reset_array_rows_with_info_accepts_explicit_metadata_arrays():
    fixture = vector_compare.seed_bridge.seed_fixture(
        "scenarios/environment/source_body_opponent_overlap_kills_step.json",
        body_capacity=4,
    )
    seeded_state = vector_compare.array_state_from_seed(fixture)
    source = {
        name: np.concatenate([array, array.copy()], axis=0)
        for name, array in seeded_state.items()
    }
    target = vector_compare.copy_array_state(source)

    info = vector_compare.reset_array_rows_with_info(
        target,
        source,
        np.asarray([True, False], dtype=bool),
        reset_episode_id=np.asarray([7, 8], dtype=np.int64),
        reset_seed=np.asarray([300, 301], dtype=np.uint64),
        reset_source=np.asarray(
            [vector_compare.RESET_SOURCE_REPLAY, vector_compare.RESET_SOURCE_MANUAL],
            dtype=np.int16,
        ),
        snapshot_array_names=("tick",),
    )

    np.testing.assert_array_equal(
        info["reset_episode_id"],
        np.asarray([7, 8], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        info["reset_seed"],
        np.asarray([300, 301], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        info["reset_source"],
        np.asarray(
            [vector_compare.RESET_SOURCE_REPLAY, vector_compare.RESET_SOURCE_MANUAL],
            dtype=np.int16,
        ),
    )
    assert tuple(info["terminal_transition_snapshot"]["arrays"]) == ("tick",)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        (
            {"reset_source": np.asarray([99], dtype=np.int16)},
            "known reset source",
        ),
        (
            {"reset_episode_id": np.asarray([1], dtype=np.int32)},
            "reset_episode_id must be",
        ),
        (
            {"reset_seed": -1},
            "reset_seed scalar must be non-negative",
        ),
    ),
)
def test_reset_array_rows_with_info_rejects_invalid_metadata_before_reset(kwargs, match):
    fixture = vector_compare.seed_bridge.seed_fixture(
        "scenarios/environment/source_body_opponent_overlap_kills_step.json",
        body_capacity=4,
    )
    source = vector_compare.array_state_from_seed(fixture)
    target = vector_compare.copy_array_state(source)
    target["tick"][0] = 77

    with pytest.raises(vector_compare.VectorCompareError, match=match):
        vector_compare.reset_array_rows_with_info(
            target,
            source,
            np.asarray([True], dtype=bool),
            snapshot_array_names=("tick",),
            **kwargs,
        )

    assert int(target["tick"][0]) == 77


def test_horizon_truncation_mask_marks_active_horizon_rows_for_final_mask_input():
    truncated = vector_compare.horizon_truncation_mask(
        np.asarray([0, 2, 4, 5, 6], dtype=np.int32),
        np.asarray([0, 2, 5, 5, 7], dtype=np.int32),
    )

    np.testing.assert_array_equal(
        truncated,
        np.asarray([False, True, False, True, False], dtype=bool),
    )

    final_mask = vector_compare.final_transition_mask(
        np.asarray([True, False, False, False, False], dtype=bool),
        truncated,
    )
    np.testing.assert_array_equal(
        final_mask,
        np.asarray([True, True, False, True, False], dtype=bool),
    )


@pytest.mark.parametrize(
    ("episode_step", "horizon_steps", "match"),
    (
        (
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([2, 2], dtype=np.int32),
            "episode_step must be an int32 array",
        ),
        (
            np.asarray([[0, 1]], dtype=np.int32),
            np.asarray([2, 2], dtype=np.int32),
            "episode_step must be an int32 array",
        ),
        (
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([2, 2], dtype=np.int16),
            "horizon_steps must be an int32 array",
        ),
        (
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([2], dtype=np.int32),
            "matching shape",
        ),
        (
            np.asarray([-1, 1], dtype=np.int32),
            np.asarray([2, 2], dtype=np.int32),
            "episode_step values must be non-negative",
        ),
        (
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([2, -2], dtype=np.int32),
            "horizon_steps values must be non-negative",
        ),
    ),
)
def test_horizon_truncation_mask_rejects_invalid_inputs(
    episode_step,
    horizon_steps,
    match,
):
    with pytest.raises(vector_compare.VectorCompareError, match=match):
        vector_compare.horizon_truncation_mask(episode_step, horizon_steps)


def test_final_transition_mask_combines_done_and_truncated_rows():
    mask = vector_compare.final_transition_mask(
        np.asarray([False, True, False, True], dtype=bool),
        np.asarray([False, False, True, True], dtype=bool),
    )

    np.testing.assert_array_equal(
        mask,
        np.asarray([False, True, True, True], dtype=bool),
    )


def test_final_transition_mask_uses_done_when_truncated_is_absent():
    done = np.asarray([False, True], dtype=bool)

    mask = vector_compare.final_transition_mask(done)
    mask[0] = True

    np.testing.assert_array_equal(done, np.asarray([False, True], dtype=bool))


@pytest.mark.parametrize(
    ("done", "truncated", "match"),
    (
        (
            np.asarray([0, 1], dtype=np.int8),
            None,
            "done must be a bool array",
        ),
        (
            np.asarray([[False, True]], dtype=bool),
            None,
            "done must be a bool array",
        ),
        (
            np.asarray([False, True], dtype=bool),
            np.asarray([0, 1], dtype=np.int8),
            "truncated must be a bool array",
        ),
        (
            np.asarray([False, True], dtype=bool),
            np.asarray([False], dtype=bool),
            "matching shape",
        ),
    ),
)
def test_final_transition_mask_rejects_invalid_inputs(done, truncated, match):
    with pytest.raises(vector_compare.VectorCompareError, match=match):
        vector_compare.final_transition_mask(done, truncated)


def test_row_lifecycle_arrays_builds_done_flags_and_terminal_reasons():
    source_terminated = np.asarray([False, True, False, False, False, True, False], dtype=bool)
    source_terminal_reason = np.asarray(
        [
            vector_compare.TERMINAL_REASON_NONE,
            vector_compare.TERMINAL_REASON_SURVIVOR_WIN,
            vector_compare.TERMINAL_REASON_NONE,
            vector_compare.TERMINAL_REASON_NONE,
            vector_compare.TERMINAL_REASON_NONE,
            vector_compare.TERMINAL_REASON_ALL_DEAD_DRAW,
            vector_compare.TERMINAL_REASON_NONE,
        ],
        dtype=np.int16,
    )

    lifecycle = vector_compare.row_lifecycle_arrays(
        source_terminated,
        source_terminal_reason,
        episode_step=np.asarray([0, 4, 3, 1, 4, 9, 7], dtype=np.int32),
        horizon_steps=np.asarray([0, 4, 3, 0, 4, 8, 7], dtype=np.int32),
        event_overflow=np.asarray([False, False, False, True, False, True, True], dtype=bool),
        body_overflow=np.asarray([False, False, False, False, True, True, True], dtype=bool),
    )

    np.testing.assert_array_equal(
        lifecycle["terminated"],
        np.asarray([False, True, False, False, False, True, False], dtype=bool),
    )
    np.testing.assert_array_equal(
        lifecycle["truncated"],
        np.asarray([False, False, True, True, True, False, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        lifecycle["done"],
        np.asarray([False, True, True, True, True, True, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        lifecycle["terminal_reason"],
        np.asarray(
            [
                vector_compare.TERMINAL_REASON_NONE,
                vector_compare.TERMINAL_REASON_SURVIVOR_WIN,
                vector_compare.TERMINAL_REASON_TIMEOUT_TRUNCATED,
                vector_compare.TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED,
                vector_compare.TERMINAL_REASON_BODY_OVERFLOW_TRUNCATED,
                vector_compare.TERMINAL_REASON_ALL_DEAD_DRAW,
                vector_compare.TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED,
            ],
            dtype=np.int16,
        ),
    )
    assert lifecycle["terminal_reason"].dtype == np.int16

    lifecycle["terminated"][0] = True
    np.testing.assert_array_equal(
        source_terminated,
        np.asarray([False, True, False, False, False, True, False], dtype=bool),
    )


@pytest.mark.parametrize(
    ("source_terminated", "source_terminal_reason", "event_overflow", "match"),
    (
        (
            np.asarray([True], dtype=bool),
            np.asarray([vector_compare.TERMINAL_REASON_NONE], dtype=np.int16),
            None,
            "survivor_win or all_dead_draw",
        ),
        (
            np.asarray([True], dtype=bool),
            np.asarray([vector_compare.TERMINAL_REASON_TIMEOUT_TRUNCATED], dtype=np.int16),
            None,
            "survivor_win or all_dead_draw",
        ),
        (
            np.asarray([False], dtype=bool),
            np.asarray([vector_compare.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
            None,
            "must be none",
        ),
        (
            np.asarray([False], dtype=bool),
            np.asarray([99], dtype=np.int16),
            None,
            "known terminal reason",
        ),
        (
            np.asarray([False], dtype=bool),
            np.asarray([vector_compare.TERMINAL_REASON_NONE], dtype=np.int16),
            np.asarray([False, True], dtype=bool),
            "event_overflow and source_terminated",
        ),
    ),
)
def test_row_lifecycle_arrays_rejects_invalid_source_reasons_and_shapes(
    source_terminated,
    source_terminal_reason,
    event_overflow,
    match,
):
    with pytest.raises(vector_compare.VectorCompareError, match=match):
        vector_compare.row_lifecycle_arrays(
            source_terminated,
            source_terminal_reason,
            episode_step=np.zeros(source_terminated.shape, dtype=np.int32),
            horizon_steps=np.zeros(source_terminated.shape, dtype=np.int32),
            event_overflow=event_overflow,
        )


def test_fixture_transition_support_reports_out_of_range_step_as_unsupported():
    fixture = vector_compare.seed_bridge.seed_fixture(
        "scenarios/environment/source_body_opponent_tangent_safe_step.json",
        body_capacity=4,
    )

    support = vector_compare.fixture_transition_support(fixture, step_index=1)

    assert support["supported"] is False
    assert support["unsupported_mechanics"][0]["mechanic"] == "multi-step rollout"


def test_source_print_manager_active_stop_on_death_step_passes_common_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_print_manager_active_stop_on_death_step.json",
        body_capacity=4,
    )

    player = result["array_projection"]["steps"][0]["players"][0]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["normal_wall_deaths"] == 1
    assert result["array_counters"]["print_manager_death_stops"] == 1
    assert result["array_counters"]["print_manager_death_stop_points"] == 1
    assert result["array_counters"]["print_manager_death_stop_visual_clears"] == 1
    assert result["array_projection"]["steps"][0]["worldBodyCount"] == 2
    assert player["alive"] is False
    assert player["printing"] is False
    assert player["trailPointCount"] == 0
    assert player["lastTrailPoint"] is None
    assert player["bodyCount"] == 2
    assert player["printManager"] == {
        "active": False,
        "distance": 0.0,
        "lastX": 0.0,
        "lastY": 0.0,
    }
    assert result["array_event_arrays"]["event_type"][0][3:8] == [
        vector_compare.EVENT_POINT,
        vector_compare.EVENT_POINT,
        vector_compare.EVENT_PROPERTY,
        vector_compare.EVENT_DIE,
        vector_compare.EVENT_SCORE_ROUND,
    ]
    assert result["array_projection"]["steps"][0]["events"][3:8] == [
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
    assert "active printing wall death stop important point before die" in result["covered_mechanics"]


def test_source_print_manager_active_hole_stop_on_death_step_passes_common_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_print_manager_active_hole_stop_on_death_step.json",
        body_capacity=4,
    )

    player = result["array_projection"]["steps"][0]["players"][0]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["normal_wall_deaths"] == 1
    assert result["array_counters"]["print_manager_death_stops"] == 1
    assert result["array_counters"]["print_manager_death_stop_points"] == 0
    assert result["array_counters"]["print_manager_death_stop_visual_clears"] == 0
    assert result["array_projection"]["steps"][0]["worldBodyCount"] == 1
    assert player["alive"] is False
    assert player["printing"] is False
    assert player["trailPointCount"] == 1
    assert player["lastTrailPoint"] == [95.5, 47.5]
    assert player["bodyCount"] == 1
    assert player["printManager"] == {
        "active": False,
        "distance": 0.0,
        "lastX": 0.0,
        "lastY": 0.0,
    }
    assert result["array_event_arrays"]["event_type"][0][3:7] == [
        vector_compare.EVENT_POINT,
        vector_compare.EVENT_PROPERTY,
        vector_compare.EVENT_DIE,
        vector_compare.EVENT_SCORE_ROUND,
    ]
    assert "already-hole death stop emits no important stop point" in result["covered_mechanics"]


def test_source_print_manager_body_collision_stop_on_death_step_passes_common_trace_compare():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_print_manager_body_collision_stop_on_death_step.json",
        body_capacity=4,
    )

    player = result["array_projection"]["steps"][0]["players"][0]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["body_hits"] == 1
    assert result["array_counters"]["print_manager_death_stops"] == 1
    assert result["array_counters"]["print_manager_death_stop_points"] == 1
    assert result["array_counters"]["print_manager_death_stop_visual_clears"] == 1
    assert result["array_projection"]["steps"][0]["worldBodyCount"] == 3
    assert player["alive"] is False
    assert player["printing"] is False
    assert player["trailPointCount"] == 0
    assert player["lastTrailPoint"] is None
    assert player["bodyCount"] == 2
    assert player["printManager"] == {
        "active": False,
        "distance": 0.0,
        "lastX": 0.0,
        "lastY": 0.0,
    }
    assert result["array_projection"]["steps"][0]["events"][6] == {
        "event": "die",
        "player_id": "p0",
        "killer_id": "p1",
        "old": False,
    }
    assert result["array_event_arrays"]["event_type"][0][3:8] == [
        vector_compare.EVENT_POINT,
        vector_compare.EVENT_POINT,
        vector_compare.EVENT_PROPERTY,
        vector_compare.EVENT_DIE,
        vector_compare.EVENT_SCORE_ROUND,
    ]
    assert "seeded body-collision death before PrintManager test" in result["covered_mechanics"]


def test_print_manager_batch_supports_death_stop_and_delayed_start():
    summary = vector_compare.compare_inputs(
        ["scenarios/environment/source_print_manager_batch.json"],
        body_capacity=4,
    )

    statuses = {fixture["scenario_id"]: fixture["status"] for fixture in summary["fixtures"]}

    assert summary["summary"] == {
        "passed": 8,
        "failed": 0,
        "unsupported": 0,
        "status": "pass",
    }
    assert statuses["source_print_manager_active_stop_on_death_step"] == "pass"
    assert statuses["source_print_manager_active_hole_stop_on_death_step"] == "pass"
    assert statuses["source_print_manager_body_collision_stop_on_death_step"] == "pass"
    assert statuses["source_print_manager_delayed_start_timer_step"] == "pass"


def test_source_print_manager_delayed_start_timer_step_passes_full_timer_trace():
    result = vector_compare.compare_fixture(
        "scenarios/environment/source_print_manager_delayed_start_timer_step.json",
        body_capacity=4,
    )

    first_step, second_step = result["array_projection"]["steps"]
    first_player = first_step["players"][0]
    second_player = second_step["players"][0]

    assert result["status"] == "pass"
    assert result["match"] is True
    assert result["array_counters"]["pre_step_timer_advances"] == 2
    assert result["array_counters"]["pre_step_timer_fires"] == 1
    assert result["array_counters"]["print_manager_delayed_start_fires"] == 1
    assert result["array_counters"]["print_manager_delayed_start_points"] == 1
    assert result["array_counters"]["events_emitted"] == 4
    assert result["array_counters_by_step"][0]["events_emitted"] == 1
    assert result["array_counters_by_step"][1]["events_emitted"] == 3
    assert first_step["events"] == [
        {
            "event": "position",
            "player_id": "p0",
            "x": 20.0,
            "y": 40.0,
        }
    ]
    assert first_player["printing"] is False
    assert first_player["printManager"] == {
        "active": False,
        "distance": 0.0,
        "lastX": 0.0,
        "lastY": 0.0,
    }
    assert second_step["events"] == [
        {
            "event": "point",
            "player_id": "p0",
            "x": 20.0,
            "y": 40.0,
            "important": True,
        },
        {
            "event": "property",
            "player_id": "p0",
            "property": "printing",
            "value": True,
        },
        {
            "event": "position",
            "player_id": "p0",
            "x": 20.0,
            "y": 40.0,
        },
    ]
    assert second_step["worldBodyCount"] == 1
    assert second_player["printing"] is True
    assert second_player["trailPointCount"] == 1
    assert second_player["lastTrailPoint"] == [20.0, 40.0]
    assert second_player["bodyNum"] == 1
    assert second_player["bodyCount"] == 1
    assert second_player["printManager"] == {
        "active": True,
        "distance": 39.0,
        "lastX": 20.0,
        "lastY": 40.0,
    }
    assert result["array_event_arrays"]["event_type"][0][:3] == [
        vector_compare.EVENT_POINT,
        vector_compare.EVENT_PROPERTY,
        vector_compare.EVENT_POSITION,
    ]
    assert "pre-step timer advancement before movement events" in result["covered_mechanics"]


def test_source_print_manager_delayed_start_does_not_support_mid_fixture_entrypoint():
    fixture = vector_compare.seed_bridge.seed_fixture(
        "scenarios/environment/source_print_manager_delayed_start_timer_step.json",
        body_capacity=4,
    )

    support = vector_compare.fixture_transition_support(fixture, step_index=1)

    assert support["supported"] is False
    assert support["unsupported_mechanics"][0]["mechanic"] == "delayed-start fixture trace"
