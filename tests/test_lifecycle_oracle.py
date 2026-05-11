import json
import shutil
import subprocess
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_spawn_rng_warmup_print_start_2p.json"
)
_NEXT_ROUND_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_spawn_rng_2p_next_round.json"
)
_SURVIVOR_NEXT_2P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_survivor_score_2p_next_round.json"
)
_MATCH_END_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_match_end_at_max_score_2p.json"
)
_MATCH_END_3P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_match_end_at_max_score_3p.json"
)
_MATCH_END_4P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_match_end_at_max_score_4p.json"
)
_TIE_AT_MAX_3P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_tie_at_max_score_3p.json"
)
_TIE_AT_MAX_4P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_tie_at_max_score_4p.json"
)
_MULTI_ROUND_MATCH_END_3P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_multi_round_match_end_3p.json"
)
_MULTI_ROUND_MATCH_END_4P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_multi_round_match_end_4p.json"
)
_HEADING_REJECTION_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_spawn_heading_rejection_retry_2p.json"
)
_SPAWN_ORDER_3P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_spawn_rng_order_3p.json"
)
_WARMUP_3P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_spawn_rng_warmup_print_start_3p.json"
)
_NEXT_ROUND_3P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_spawn_rng_3p_next_round.json"
)
_SURVIVOR_SCORE_3P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_survivor_score_3p_round_end.json"
)
_SURVIVOR_NEXT_3P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_survivor_score_3p_next_round.json"
)
_REMOVE_AVATAR_WARMDOWN_3P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_remove_avatar_during_warmdown_3p.json"
)
_REMOVE_AVATAR_TO_SINGLE_PRESENT_3P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_remove_avatar_to_single_present_3p.json"
)
_SPAWN_ORDER_4P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_spawn_rng_order_4p.json"
)
_NEXT_ROUND_4P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_spawn_rng_4p_next_round.json"
)
_SURVIVOR_NEXT_4P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_survivor_score_4p_next_round.json"
)
_PRESENT_ABSENT_3P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_present_absent_3p_round_new.json"
)
_PRESENT_ABSENT_SURVIVOR_3P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_present_absent_3p_survivor_score_round_end.json"
)
_PRESENT_ABSENT_NEXT_3P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_present_absent_3p_next_round.json"
)
_PRESENT_ABSENT_TIE_MAX_3P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_present_absent_3p_tie_at_max_score.json"
)
_PRESENT_ABSENT_4P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_present_absent_4p_round_new.json"
)
_PRESENT_ABSENT_SURVIVOR_4P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_present_absent_4p_survivor_score_round_end.json"
)
_PRESENT_ABSENT_NEXT_4P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_present_absent_4p_next_round.json"
)
_PRESENT_ABSENT_TIE_MAX_4P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_present_absent_4p_tie_at_max_score.json"
)
_MID_ROUND_REMOVE_AVATAR_3P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_mid_round_remove_avatar_3p_continue_round_end.json"
)
_MID_ROUND_REMOVE_AVATAR_4P_SCENARIO = (
    _REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_mid_round_remove_avatar_4p_continue_round_end.json"
)
_ORACLE = _REPO_ROOT / "tools" / "reference_oracle" / "lifecycle_oracle.js"


def _run_oracle(scenario):
    if shutil.which("node") is None:
        pytest.skip("node executable is not available")

    result = subprocess.run(
        ["node", str(_ORACLE), str(scenario)],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_lifecycle_oracle_pins_source_spawn_rng_and_print_start_body_claims():
    payload = _run_oracle(_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert [
        (call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
    ] == [
        ("spawn.position_x", 2, 0),
        ("spawn.position_y", 2, 0),
        ("spawn.angle_attempt_0", 2, 0),
        ("spawn.position_x", 1, 0),
        ("spawn.position_y", 1, 0),
        ("spawn.angle_attempt_0", 1, 0),
        ("print_manager.start_distance", 2, 3000),
        ("print_manager.start_distance", 1, 3000),
    ]

    snapshots = {snapshot["label"]: snapshot for snapshot in payload["snapshots"]}

    after_spawn = snapshots["after_new_round_call"]
    assert after_spawn["game"]["worldActive"] is False
    assert after_spawn["game"]["worldBodyCount"] == 0

    after_game_start_before_print_start = snapshots["after_advance_0"]
    assert after_game_start_before_print_start["game"]["worldActive"] is True
    assert after_game_start_before_print_start["game"]["worldBodyCount"] == 0

    after_delayed_print_start = snapshots["after_advance_1"]
    assert after_delayed_print_start["game"]["worldActive"] is True
    assert after_delayed_print_start["game"]["worldBodyCount"] == 2


def test_lifecycle_oracle_pins_source_next_round_stop_spawn_rng_and_final_snapshot():
    payload = _run_oracle(_NEXT_ROUND_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"

    events = payload["events"]
    round_end_index = next(
        index for index, event in enumerate(events) if event["event"] == "round:end"
    )
    assert events[round_end_index]["atMs"] == 3000
    assert events[round_end_index]["data"] == {"winner": None}

    game_stop_index = next(
        index for index, event in enumerate(events) if event["event"] == "game:stop"
    )
    assert events[game_stop_index]["atMs"] == 8000
    assert events[game_stop_index + 1]["event"] == "round:new"
    assert events[game_stop_index + 1]["atMs"] == 8000

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
        if 10 <= call["index"] <= 15
    ] == [
        (10, "spawn.position_x", 2, 8000),
        (11, "spawn.position_y", 2, 8000),
        (12, "spawn.angle_attempt_0", 2, 8000),
        (13, "spawn.position_x", 1, 8000),
        (14, "spawn.position_y", 1, 8000),
        (15, "spawn.angle_attempt_0", 1, 8000),
    ]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_5_advance_timers"
    assert final_snapshot["game"]["inRound"] is True
    assert final_snapshot["game"]["worldActive"] is False
    assert final_snapshot["game"]["worldBodyCount"] == 0
    assert [
        (avatar["id"], avatar["alive"], avatar["present"])
        for avatar in final_snapshot["avatars"]
    ] == [
        (1, True, True),
        (2, True, True),
    ]


def test_lifecycle_oracle_pins_source_2p_survivor_warmdown_next_round():
    payload = _run_oracle(_SURVIVOR_NEXT_2P_SCENARIO)

    assert payload["playerCount"] == 2

    events = payload["events"]
    event_names = [event["event"] for event in events]
    assert event_names.count("round:end") == 1
    round_end_index = event_names.index("round:end")
    assert events[round_end_index]["atMs"] == 3000
    assert events[round_end_index]["data"] == {"winner": 1}

    assert [
        event["event"] for event in events[round_end_index + 1 : round_end_index + 6]
    ] == [
        "point",
        "property",
        "random",
        "die",
        "score:round",
    ]
    assert [
        event["atMs"] for event in events[round_end_index + 1 : round_end_index + 6]
    ] == [
        4150,
        4150,
        4150,
        4150,
        4150,
    ]
    assert events[round_end_index + 4]["data"] == {
        "avatar": 1,
        "killer": None,
        "old": None,
    }
    assert events[round_end_index + 5]["data"] == {
        "avatar": 1,
        "score": 1,
        "roundScore": 1,
    }

    game_stop_index = event_names.index("game:stop")
    assert events[game_stop_index]["atMs"] == 8000
    assert events[game_stop_index + 1]["event"] == "round:new"
    assert events[game_stop_index + 1]["atMs"] == 8000

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], round(call["atMs"]))
        for call in payload["randomCalls"]
        if 9 <= call["index"] <= 15
    ] == [
        (9, "print_manager.stop_distance", 1, 4150),
        (10, "spawn.position_x", 2, 8000),
        (11, "spawn.position_y", 2, 8000),
        (12, "spawn.angle_attempt_0", 2, 8000),
        (13, "spawn.position_x", 1, 8000),
        (14, "spawn.position_y", 1, 8000),
        (15, "spawn.angle_attempt_0", 1, 8000),
    ]

    after_update = payload["snapshots"][-2]
    assert after_update["label"] == "after_action_4_update"
    assert after_update["game"]["inRound"] is False
    assert after_update["game"]["deaths"] == [2]
    assert [
        (avatar["id"], avatar["x"], avatar["alive"], avatar["printing"], avatar["score"])
        for avatar in after_update["avatars"]
    ] == [
        (1, 18.991, True, True, 1),
        (2, 88.6, False, False, 0),
    ]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_5_advance_timers"
    assert final_snapshot["game"]["inRound"] is True
    assert final_snapshot["game"]["worldActive"] is False
    assert final_snapshot["game"]["deaths"] == []
    assert [
        (avatar["id"], avatar["x"], avatar["y"], avatar["angle"], avatar["alive"], avatar["score"])
        for avatar in final_snapshot["avatars"]
    ] == [
        (1, 51.8, 44, 4.712389, True, 1),
        (2, 36.2, 44, 1.570796, True, 0),
    ]


def test_lifecycle_oracle_pins_source_match_end_at_max_score():
    payload = _run_oracle(_MATCH_END_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 0

    events = payload["events"]
    assert [event["event"] for event in events[-3:]] == ["round:end", "game:stop", "end"]
    assert events[-3]["atMs"] == 3000
    assert events[-3]["data"] == {"winner": 1}
    assert events[-2]["atMs"] == 8000
    assert events[-1]["atMs"] == 8000
    assert "round:new" not in [event["event"] for event in events[-2:]]

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
    ] == [
        (0, "spawn.position_x", 2, 0),
        (1, "spawn.position_y", 2, 0),
        (2, "spawn.angle_attempt_0", 2, 0),
        (3, "spawn.position_x", 1, 0),
        (4, "spawn.position_y", 1, 0),
        (5, "spawn.angle_attempt_0", 1, 0),
        (6, "print_manager.start_distance", 2, 3000),
        (7, "print_manager.start_distance", 1, 3000),
        (8, "print_manager.stop_distance", 2, 3000),
    ]

    after_update = payload["snapshots"][-2]
    assert after_update["label"] == "after_action_4_update"
    assert after_update["game"]["inRound"] is False
    assert [
        (avatar["id"], avatar["alive"], avatar["score"], avatar["roundScore"])
        for avatar in after_update["avatars"]
    ] == [
        (1, True, 1, 0),
        (2, False, 0, 0),
    ]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_5_advance_timers"
    assert final_snapshot["game"]["started"] is False
    assert final_snapshot["game"]["inRound"] is False
    assert final_snapshot["game"]["worldActive"] is None
    assert final_snapshot["game"]["worldBodyCount"] is None
    assert final_snapshot["avatars"] == []


def test_lifecycle_oracle_pins_source_3p_match_end_at_max_score():
    payload = _run_oracle(_MATCH_END_3P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 0

    events = payload["events"]
    assert [event["event"] for event in events[-3:]] == ["round:end", "game:stop", "end"]
    assert events[-3]["atMs"] == 3000
    assert events[-3]["data"] == {"winner": 1}
    assert events[-2]["atMs"] == 8000
    assert events[-1]["atMs"] == 8000
    assert "round:new" not in [event["event"] for event in events[-2:]]
    assert [
        event["data"]["avatar"]
        for event in events
        if event["event"] == "die"
    ] == [3, 2]

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
    ] == [
        (0, "spawn.position_x", 3, 0),
        (1, "spawn.position_y", 3, 0),
        (2, "spawn.angle_attempt_0", 3, 0),
        (3, "spawn.position_x", 2, 0),
        (4, "spawn.position_y", 2, 0),
        (5, "spawn.angle_attempt_0", 2, 0),
        (6, "spawn.position_x", 1, 0),
        (7, "spawn.position_y", 1, 0),
        (8, "spawn.angle_attempt_0", 1, 0),
        (9, "print_manager.start_distance", 3, 3000),
        (10, "print_manager.start_distance", 2, 3000),
        (11, "print_manager.start_distance", 1, 3000),
        (12, "print_manager.stop_distance", 3, 3000),
        (13, "print_manager.stop_distance", 2, 3000),
    ]

    after_update = payload["snapshots"][-2]
    assert after_update["label"] == "after_action_5_update"
    assert after_update["game"]["inRound"] is False
    assert [
        (avatar["id"], avatar["alive"], avatar["printing"], avatar["score"], avatar["roundScore"])
        for avatar in after_update["avatars"]
    ] == [
        (1, True, True, 2, 0),
        (2, False, False, 0, 0),
        (3, False, False, 0, 0),
    ]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_6_advance_timers"
    assert final_snapshot["game"]["started"] is False
    assert final_snapshot["game"]["inRound"] is False
    assert final_snapshot["game"]["worldActive"] is None
    assert final_snapshot["game"]["worldBodyCount"] is None
    assert final_snapshot["avatars"] == []


def test_lifecycle_oracle_pins_source_4p_match_end_at_max_score():
    payload = _run_oracle(_MATCH_END_4P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 0

    events = payload["events"]
    assert [event["event"] for event in events[-3:]] == ["round:end", "game:stop", "end"]
    assert events[-3]["atMs"] == 3000
    assert events[-3]["data"] == {"winner": 1}
    assert events[-2]["atMs"] == 8000
    assert events[-1]["atMs"] == 8000
    assert "round:new" not in [event["event"] for event in events[-2:]]
    assert [
        event["data"]["avatar"]
        for event in events
        if event["event"] == "die"
    ] == [4, 3, 2]

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
        if 12 <= call["index"] <= 18
    ] == [
        (12, "print_manager.start_distance", 4, 3000),
        (13, "print_manager.start_distance", 3, 3000),
        (14, "print_manager.start_distance", 2, 3000),
        (15, "print_manager.start_distance", 1, 3000),
        (16, "print_manager.stop_distance", 4, 3000),
        (17, "print_manager.stop_distance", 3, 3000),
        (18, "print_manager.stop_distance", 2, 3000),
    ]

    after_update = payload["snapshots"][-2]
    assert after_update["label"] == "after_action_8_update"
    assert after_update["game"]["inRound"] is False
    assert [
        (
            avatar["id"],
            avatar["alive"],
            avatar["printing"],
            avatar["score"],
            avatar["roundScore"],
        )
        for avatar in after_update["avatars"]
    ] == [
        (1, True, True, 3, 0),
        (2, False, False, 2, 0),
        (3, False, False, 1, 0),
        (4, False, False, 0, 0),
    ]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_9_advance_timers"
    assert final_snapshot["game"]["started"] is False
    assert final_snapshot["game"]["inRound"] is False
    assert final_snapshot["game"]["worldActive"] is None
    assert final_snapshot["game"]["worldBodyCount"] is None
    assert final_snapshot["avatars"] == []


def test_lifecycle_oracle_pins_source_3p_tie_at_max_score_continuation():
    payload = _run_oracle(_TIE_AT_MAX_3P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 3

    events = payload["events"]
    assert [event["event"] for event in events[-18:]] == [
        "round:end",
        "game:stop",
        "round:new",
        "random",
        "random",
        "position",
        "random",
        "angle",
        "random",
        "random",
        "position",
        "random",
        "angle",
        "random",
        "random",
        "position",
        "random",
        "angle",
    ]
    assert events[-19]["event"] == "score"
    assert events[-19]["data"] == {"avatar": 1, "score": 1, "roundScore": 1}
    assert events[-18]["atMs"] == 3000
    assert events[-18]["data"] == {"winner": None}
    assert events[-17]["atMs"] == 8000
    assert events[-16]["event"] == "round:new"
    assert events[-16]["atMs"] == 8000
    assert "end" not in [event["event"] for event in events[-18:]]
    assert [
        event["data"]["avatar"]
        for event in events
        if event["event"] == "die"
    ] == [3, 2, 1]

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
        if 12 <= call["index"] <= 23
    ] == [
        (12, "print_manager.stop_distance", 3, 3000),
        (13, "print_manager.stop_distance", 2, 3000),
        (14, "print_manager.stop_distance", 1, 3000),
        (15, "spawn.position_x", 3, 8000),
        (16, "spawn.position_y", 3, 8000),
        (17, "spawn.angle_attempt_0", 3, 8000),
        (18, "spawn.position_x", 2, 8000),
        (19, "spawn.position_y", 2, 8000),
        (20, "spawn.angle_attempt_0", 2, 8000),
        (21, "spawn.position_x", 1, 8000),
        (22, "spawn.position_y", 1, 8000),
        (23, "spawn.angle_attempt_0", 1, 8000),
    ]

    after_tie_round_end = payload["snapshots"][-2]
    assert after_tie_round_end["label"] == "after_action_6_update"
    assert after_tie_round_end["game"]["inRound"] is False
    assert after_tie_round_end["game"]["deaths"] == [3, 2, 1]
    assert [
        (avatar["id"], avatar["alive"], avatar["score"], avatar["roundScore"])
        for avatar in after_tie_round_end["avatars"]
    ] == [
        (1, False, 1, 0),
        (2, False, 1, 0),
        (3, False, 0, 0),
    ]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_7_advance_timers"
    assert final_snapshot["game"]["started"] is True
    assert final_snapshot["game"]["inRound"] is True
    assert final_snapshot["game"]["worldActive"] is False
    assert final_snapshot["game"]["worldBodyCount"] == 0
    assert final_snapshot["game"]["deaths"] == []
    assert [
        (avatar["id"], avatar["alive"], avatar["score"])
        for avatar in final_snapshot["avatars"]
    ] == [(1, True, 1), (2, True, 1), (3, True, 0)]


def test_lifecycle_oracle_pins_source_4p_tie_at_max_score_continuation():
    payload = _run_oracle(_TIE_AT_MAX_4P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 4

    events = payload["events"]
    assert "end" not in [event["event"] for event in events]
    round_end_index = next(
        index for index, event in enumerate(events) if event["event"] == "round:end"
    )
    assert events[round_end_index]["atMs"] == 3000
    assert events[round_end_index]["data"] == {"winner": None}
    assert events[round_end_index + 1]["event"] == "game:stop"
    assert events[round_end_index + 1]["atMs"] == 8000
    assert events[round_end_index + 2]["event"] == "round:new"
    assert events[round_end_index + 2]["atMs"] == 8000
    assert [
        event["data"]["avatar"]
        for event in events
        if event["event"] == "die"
    ] == [4, 3, 2, 1]
    assert [
        event["data"]
        for event in events[round_end_index - 4 : round_end_index]
    ] == [
        {"avatar": 4, "score": 0, "roundScore": 0},
        {"avatar": 3, "score": 0, "roundScore": 0},
        {"avatar": 2, "score": 2, "roundScore": 2},
        {"avatar": 1, "score": 2, "roundScore": 2},
    ]

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
        if 16 <= call["index"] <= 31
    ] == [
        (16, "print_manager.stop_distance", 4, 3000),
        (17, "print_manager.stop_distance", 3, 3000),
        (18, "print_manager.stop_distance", 2, 3000),
        (19, "print_manager.stop_distance", 1, 3000),
        (20, "spawn.position_x", 4, 8000),
        (21, "spawn.position_y", 4, 8000),
        (22, "spawn.angle_attempt_0", 4, 8000),
        (23, "spawn.position_x", 3, 8000),
        (24, "spawn.position_y", 3, 8000),
        (25, "spawn.angle_attempt_0", 3, 8000),
        (26, "spawn.position_x", 2, 8000),
        (27, "spawn.position_y", 2, 8000),
        (28, "spawn.angle_attempt_0", 2, 8000),
        (29, "spawn.position_x", 1, 8000),
        (30, "spawn.position_y", 1, 8000),
        (31, "spawn.angle_attempt_0", 1, 8000),
    ]

    after_first_pair = payload["snapshots"][5]
    assert after_first_pair["label"] == "after_action_4_update"
    assert after_first_pair["game"]["inRound"] is True
    assert after_first_pair["game"]["deaths"] == [4, 3]
    assert [
        (avatar["id"], avatar["alive"], avatar["score"])
        for avatar in after_first_pair["avatars"]
    ] == [
        (1, True, 0),
        (2, True, 0),
        (3, False, 0),
        (4, False, 0),
    ]

    after_tie_round_end = payload["snapshots"][-2]
    assert after_tie_round_end["label"] == "after_action_7_update"
    assert after_tie_round_end["game"]["inRound"] is False
    assert after_tie_round_end["game"]["deaths"] == [4, 3, 2, 1]
    assert [
        (avatar["id"], avatar["alive"], avatar["score"], avatar["roundScore"])
        for avatar in after_tie_round_end["avatars"]
    ] == [
        (1, False, 2, 0),
        (2, False, 2, 0),
        (3, False, 0, 0),
        (4, False, 0, 0),
    ]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_8_advance_timers"
    assert final_snapshot["game"]["started"] is True
    assert final_snapshot["game"]["inRound"] is True
    assert final_snapshot["game"]["worldActive"] is False
    assert final_snapshot["game"]["worldBodyCount"] == 0
    assert final_snapshot["game"]["deaths"] == []
    assert [
        (avatar["id"], avatar["alive"], avatar["score"])
        for avatar in final_snapshot["avatars"]
    ] == [(1, True, 2), (2, True, 2), (3, True, 0), (4, True, 0)]


def test_lifecycle_oracle_pins_source_3p_multi_round_match_end():
    payload = _run_oracle(_MULTI_ROUND_MATCH_END_3P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 0

    events = payload["events"]
    first_stop_index = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "game:stop"
    )
    assert events[first_stop_index]["atMs"] == 8000
    assert events[first_stop_index + 1]["event"] == "round:new"
    assert events[first_stop_index + 1]["atMs"] == 8000

    assert [event["event"] for event in events[-3:]] == ["round:end", "game:stop", "end"]
    assert events[-3]["atMs"] == 14000
    assert events[-3]["data"] == {"winner": 1}
    assert events[-2]["atMs"] == 19000
    assert events[-1]["atMs"] == 19000
    assert "round:new" not in [event["event"] for event in events[-2:]]

    assert [
        (event["event"], event["data"])
        for event in events
        if event["event"] in {"round:end", "game:stop", "end"}
    ] == [
        ("round:end", {"winner": 1}),
        ("game:stop", {}),
        ("round:end", {"winner": 1}),
        ("game:stop", {}),
        ("end", {}),
    ]
    assert [
        event["data"]["avatar"]
        for event in events
        if event["event"] == "die"
    ] == [3, 2, 3, 2]
    assert events[-4]["event"] == "score"
    assert events[-4]["data"] == {"avatar": 1, "score": 4, "roundScore": 2}

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
    ] == [
        (0, "spawn.position_x", 3, 0),
        (1, "spawn.position_y", 3, 0),
        (2, "spawn.angle_attempt_0", 3, 0),
        (3, "spawn.position_x", 2, 0),
        (4, "spawn.position_y", 2, 0),
        (5, "spawn.angle_attempt_0", 2, 0),
        (6, "spawn.position_x", 1, 0),
        (7, "spawn.position_y", 1, 0),
        (8, "spawn.angle_attempt_0", 1, 0),
        (9, "print_manager.start_distance", 3, 3000),
        (10, "print_manager.start_distance", 2, 3000),
        (11, "print_manager.start_distance", 1, 3000),
        (12, "print_manager.stop_distance", 3, 3000),
        (13, "print_manager.stop_distance", 2, 3000),
        (14, "print_manager.stop_distance", 1, 8000),
        (15, "spawn.position_x", 3, 8000),
        (16, "spawn.position_y", 3, 8000),
        (17, "spawn.angle_attempt_0", 3, 8000),
        (18, "spawn.position_x", 2, 8000),
        (19, "spawn.position_y", 2, 8000),
        (20, "spawn.angle_attempt_0", 2, 8000),
        (21, "spawn.position_x", 1, 8000),
        (22, "spawn.position_y", 1, 8000),
        (23, "spawn.angle_attempt_0", 1, 8000),
        (24, "print_manager.start_distance", 3, 14000),
        (25, "print_manager.start_distance", 2, 14000),
        (26, "print_manager.start_distance", 1, 14000),
        (27, "print_manager.stop_distance", 3, 14000),
        (28, "print_manager.stop_distance", 2, 14000),
    ]

    after_first_stop = payload["snapshots"][7]
    assert after_first_stop["label"] == "after_action_6_advance_timers"
    assert after_first_stop["game"]["inRound"] is True
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in after_first_stop["avatars"]
    ] == [(1, True, True, 2), (2, True, True, 0), (3, True, True, 0)]

    after_second_round_end = payload["snapshots"][-2]
    assert after_second_round_end["label"] == "after_action_14_update"
    assert after_second_round_end["game"]["inRound"] is False
    assert after_second_round_end["game"]["deaths"] == [3, 2]
    assert [
        (avatar["id"], avatar["alive"], avatar["score"], avatar["roundScore"])
        for avatar in after_second_round_end["avatars"]
    ] == [(1, True, 4, 0), (2, False, 0, 0), (3, False, 0, 0)]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_15_advance_timers"
    assert final_snapshot["game"]["started"] is False
    assert final_snapshot["game"]["inRound"] is False
    assert final_snapshot["game"]["worldActive"] is None
    assert final_snapshot["game"]["worldBodyCount"] is None
    assert final_snapshot["avatars"] == []


def test_lifecycle_oracle_pins_source_4p_multi_round_match_end():
    payload = _run_oracle(_MULTI_ROUND_MATCH_END_4P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 0

    events = payload["events"]
    first_stop_index = next(
        index
        for index, event in enumerate(events)
        if event["event"] == "game:stop"
    )
    assert events[first_stop_index]["atMs"] == 8000
    assert events[first_stop_index + 1]["event"] == "round:new"
    assert events[first_stop_index + 1]["atMs"] == 8000

    assert [event["event"] for event in events[-3:]] == ["round:end", "game:stop", "end"]
    assert events[-3]["atMs"] == 14000
    assert events[-3]["data"] == {"winner": 1}
    assert events[-2]["atMs"] == 19000
    assert events[-1]["atMs"] == 19000
    assert "round:new" not in [event["event"] for event in events[-2:]]

    assert [
        (event["event"], event["data"])
        for event in events
        if event["event"] in {"round:end", "game:stop", "end"}
    ] == [
        ("round:end", {"winner": 1}),
        ("game:stop", {}),
        ("round:end", {"winner": 1}),
        ("game:stop", {}),
        ("end", {}),
    ]
    assert [
        event["data"]["avatar"]
        for event in events
        if event["event"] == "die"
    ] == [4, 3, 2, 4, 3, 2]
    assert events[-4]["event"] == "score"
    assert events[-4]["data"] == {"avatar": 1, "score": 6, "roundScore": 3}

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
    ] == [
        (0, "spawn.position_x", 4, 0),
        (1, "spawn.position_y", 4, 0),
        (2, "spawn.angle_attempt_0", 4, 0),
        (3, "spawn.position_x", 3, 0),
        (4, "spawn.position_y", 3, 0),
        (5, "spawn.angle_attempt_0", 3, 0),
        (6, "spawn.position_x", 2, 0),
        (7, "spawn.position_y", 2, 0),
        (8, "spawn.angle_attempt_0", 2, 0),
        (9, "spawn.position_x", 1, 0),
        (10, "spawn.position_y", 1, 0),
        (11, "spawn.angle_attempt_0", 1, 0),
        (12, "print_manager.start_distance", 4, 3000),
        (13, "print_manager.start_distance", 3, 3000),
        (14, "print_manager.start_distance", 2, 3000),
        (15, "print_manager.start_distance", 1, 3000),
        (16, "print_manager.stop_distance", 4, 3000),
        (17, "print_manager.stop_distance", 3, 3000),
        (18, "print_manager.stop_distance", 2, 3000),
        (19, "print_manager.stop_distance", 1, 8000),
        (20, "spawn.position_x", 4, 8000),
        (21, "spawn.position_y", 4, 8000),
        (22, "spawn.angle_attempt_0", 4, 8000),
        (23, "spawn.position_x", 3, 8000),
        (24, "spawn.position_y", 3, 8000),
        (25, "spawn.angle_attempt_0", 3, 8000),
        (26, "spawn.position_x", 2, 8000),
        (27, "spawn.position_y", 2, 8000),
        (28, "spawn.angle_attempt_0", 2, 8000),
        (29, "spawn.position_x", 1, 8000),
        (30, "spawn.position_y", 1, 8000),
        (31, "spawn.angle_attempt_0", 1, 8000),
        (32, "print_manager.start_distance", 4, 14000),
        (33, "print_manager.start_distance", 3, 14000),
        (34, "print_manager.start_distance", 2, 14000),
        (35, "print_manager.start_distance", 1, 14000),
        (36, "print_manager.stop_distance", 4, 14000),
        (37, "print_manager.stop_distance", 3, 14000),
        (38, "print_manager.stop_distance", 2, 14000),
    ]

    after_first_stop = payload["snapshots"][8]
    assert after_first_stop["label"] == "after_action_7_advance_timers"
    assert after_first_stop["game"]["inRound"] is True
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in after_first_stop["avatars"]
    ] == [
        (1, True, True, 3),
        (2, True, True, 0),
        (3, True, True, 0),
        (4, True, True, 0),
    ]

    after_second_round_end = payload["snapshots"][-2]
    assert after_second_round_end["label"] == "after_action_17_update"
    assert after_second_round_end["game"]["inRound"] is False
    assert after_second_round_end["game"]["deaths"] == [4, 3, 2]
    assert [
        (avatar["id"], avatar["alive"], avatar["score"], avatar["roundScore"])
        for avatar in after_second_round_end["avatars"]
    ] == [
        (1, True, 6, 0),
        (2, False, 0, 0),
        (3, False, 0, 0),
        (4, False, 0, 0),
    ]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_18_advance_timers"
    assert final_snapshot["game"]["started"] is False
    assert final_snapshot["game"]["inRound"] is False
    assert final_snapshot["game"]["worldActive"] is None
    assert final_snapshot["game"]["worldBodyCount"] is None
    assert final_snapshot["avatars"] == []


def test_lifecycle_oracle_pins_source_spawn_heading_rejection_retry():
    payload = _run_oracle(_HEADING_REJECTION_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["value"])
        for call in payload["randomCalls"]
    ] == [
        (0, "spawn.position_x", 2, 0),
        (1, "spawn.position_y", 2, 0.5),
        (2, "spawn.angle_attempt_0", 2, 0.5),
        (3, "spawn.angle_attempt_1", 2, 0.25),
        (4, "spawn.position_x", 1, 0.6794871794871795),
        (5, "spawn.position_y", 1, 0.5),
        (6, "spawn.angle_attempt_0", 1, 0.5159154943091895),
    ]

    events = payload["events"]
    assert events[4]["event"] == "random"
    assert events[4]["data"]["site"] == "spawn.angle_attempt_0"
    assert events[5]["event"] == "random"
    assert events[5]["data"]["site"] == "spawn.angle_attempt_1"
    assert events[6]["event"] == "angle"
    assert events[6]["data"] == {"avatar": 2, "angle": 1.570796}

    snapshot = payload["snapshots"][0]
    assert snapshot["label"] == "after_new_round_call"
    assert snapshot["game"]["worldActive"] is False
    assert snapshot["game"]["worldBodyCount"] == 0
    assert [
        (avatar["id"], avatar["x"], avatar["y"], avatar["angle"])
        for avatar in snapshot["avatars"]
    ] == [
        (1, 58, 44, 3.241593),
        (2, 5, 44, 1.570796),
    ]


def test_lifecycle_oracle_pins_source_3p_spawn_rng_order():
    payload = _run_oracle(_SPAWN_ORDER_3P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 3
    assert payload["timerAdvancesMs"] == []
    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
    ] == [
        (0, "spawn.position_x", 3, 0),
        (1, "spawn.position_y", 3, 0),
        (2, "spawn.angle_attempt_0", 3, 0),
        (3, "spawn.position_x", 2, 0),
        (4, "spawn.position_y", 2, 0),
        (5, "spawn.angle_attempt_0", 2, 0),
        (6, "spawn.position_x", 1, 0),
        (7, "spawn.position_y", 1, 0),
        (8, "spawn.angle_attempt_0", 1, 0),
    ]

    snapshot = payload["snapshots"][0]
    assert snapshot["label"] == "after_new_round_call"
    assert snapshot["game"]["size"] == 95
    assert snapshot["game"]["worldActive"] is False
    assert snapshot["game"]["worldBodyCount"] == 0
    assert [
        (avatar["id"], avatar["x"], avatar["y"], avatar["angle"], avatar["present"])
        for avatar in snapshot["avatars"]
    ] == [
        (1, 68.575, 47.5, 4.712389, True),
        (2, 47.5, 47.5, 3.141593, True),
        (3, 26.425, 47.5, 1.570796, True),
    ]


def test_lifecycle_oracle_pins_source_4p_spawn_rng_order():
    payload = _run_oracle(_SPAWN_ORDER_4P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 4
    assert payload["timerAdvancesMs"] == []
    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
    ] == [
        (0, "spawn.position_x", 4, 0),
        (1, "spawn.position_y", 4, 0),
        (2, "spawn.angle_attempt_0", 4, 0),
        (3, "spawn.position_x", 3, 0),
        (4, "spawn.position_y", 3, 0),
        (5, "spawn.angle_attempt_0", 3, 0),
        (6, "spawn.position_x", 2, 0),
        (7, "spawn.position_y", 2, 0),
        (8, "spawn.angle_attempt_0", 2, 0),
        (9, "spawn.position_x", 1, 0),
        (10, "spawn.position_y", 1, 0),
        (11, "spawn.angle_attempt_0", 1, 0),
    ]

    snapshot = payload["snapshots"][0]
    assert snapshot["label"] == "after_new_round_call"
    assert snapshot["game"]["size"] == 101
    assert snapshot["game"]["worldActive"] is False
    assert snapshot["game"]["worldBodyCount"] == 0
    assert [
        (avatar["id"], avatar["x"], avatar["y"], avatar["angle"], avatar["present"])
        for avatar in snapshot["avatars"]
    ] == [
        (1, 77.41, 50.5, 4.712389, True),
        (2, 59.47, 50.5, 3.141593, True),
        (3, 41.53, 50.5, 0.1, True),
        (4, 23.59, 50.5, 1.570796, True),
    ]


def test_lifecycle_oracle_pins_source_4p_all_dead_warmdown_next_round():
    payload = _run_oracle(_NEXT_ROUND_4P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 4

    events = payload["events"]
    round_end_index = next(
        index for index, event in enumerate(events) if event["event"] == "round:end"
    )
    assert events[round_end_index]["atMs"] == 3000
    assert events[round_end_index]["data"] == {"winner": None}

    game_stop_index = next(
        index for index, event in enumerate(events) if event["event"] == "game:stop"
    )
    assert events[game_stop_index]["atMs"] == 8000
    assert events[game_stop_index + 1]["event"] == "round:new"
    assert events[game_stop_index + 1]["atMs"] == 8000
    assert [
        event["data"]["avatar"]
        for event in events
        if event["event"] == "die"
    ] == [4, 3, 2, 1]

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
        if 16 <= call["index"] <= 31
    ] == [
        (16, "print_manager.stop_distance", 4, 3000),
        (17, "print_manager.stop_distance", 3, 3000),
        (18, "print_manager.stop_distance", 2, 3000),
        (19, "print_manager.stop_distance", 1, 3000),
        (20, "spawn.position_x", 4, 8000),
        (21, "spawn.position_y", 4, 8000),
        (22, "spawn.angle_attempt_0", 4, 8000),
        (23, "spawn.position_x", 3, 8000),
        (24, "spawn.position_y", 3, 8000),
        (25, "spawn.angle_attempt_0", 3, 8000),
        (26, "spawn.position_x", 2, 8000),
        (27, "spawn.position_y", 2, 8000),
        (28, "spawn.angle_attempt_0", 2, 8000),
        (29, "spawn.position_x", 1, 8000),
        (30, "spawn.position_y", 1, 8000),
        (31, "spawn.angle_attempt_0", 1, 8000),
    ]

    after_update = payload["snapshots"][-2]
    assert after_update["label"] == "after_action_6_update"
    assert after_update["game"]["inRound"] is False
    assert after_update["game"]["deaths"] == [4, 3, 2, 1]
    assert [
        (avatar["id"], avatar["alive"], avatar["printing"], avatar["score"])
        for avatar in after_update["avatars"]
    ] == [
        (1, False, False, 0),
        (2, False, False, 0),
        (3, False, False, 0),
        (4, False, False, 0),
    ]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_7_advance_timers"
    assert final_snapshot["game"]["started"] is True
    assert final_snapshot["game"]["inRound"] is True
    assert final_snapshot["game"]["worldActive"] is False
    assert final_snapshot["game"]["worldBodyCount"] == 0
    assert final_snapshot["game"]["deaths"] == []
    assert [
        (avatar["id"], avatar["x"], avatar["y"], avatar["angle"], avatar["alive"])
        for avatar in final_snapshot["avatars"]
    ] == [
        (1, 77.41, 50.5, 4.712389, True),
        (2, 59.47, 50.5, 3.141593, True),
        (3, 41.53, 50.5, 0.1, True),
        (4, 23.59, 50.5, 1.570796, True),
    ]


def test_lifecycle_oracle_pins_source_4p_survivor_score_warmdown_next_round():
    payload = _run_oracle(_SURVIVOR_NEXT_4P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 4

    events = payload["events"]
    round_end_index = next(
        index for index, event in enumerate(events) if event["event"] == "round:end"
    )
    assert events[round_end_index]["atMs"] == 3000
    assert events[round_end_index]["data"] == {"winner": 1}
    assert [
        event["data"]["avatar"]
        for event in events
        if event["event"] == "die"
    ] == [4, 3, 2]
    assert [
        event["data"]
        for event in events[round_end_index - 5 : round_end_index]
    ] == [
        {"avatar": 1, "score": 0, "roundScore": 3},
        {"avatar": 4, "score": 0, "roundScore": 0},
        {"avatar": 3, "score": 1, "roundScore": 1},
        {"avatar": 2, "score": 2, "roundScore": 2},
        {"avatar": 1, "score": 3, "roundScore": 3},
    ]

    game_stop_index = next(
        index for index, event in enumerate(events) if event["event"] == "game:stop"
    )
    assert events[game_stop_index]["atMs"] == 8000
    assert events[game_stop_index + 1]["event"] == "round:new"
    assert events[game_stop_index + 1]["atMs"] == 8000

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
        if 16 <= call["index"] <= 31
    ] == [
        (16, "print_manager.stop_distance", 4, 3000),
        (17, "print_manager.stop_distance", 3, 3000),
        (18, "print_manager.stop_distance", 2, 3000),
        (19, "print_manager.stop_distance", 1, 8000),
        (20, "spawn.position_x", 4, 8000),
        (21, "spawn.position_y", 4, 8000),
        (22, "spawn.angle_attempt_0", 4, 8000),
        (23, "spawn.position_x", 3, 8000),
        (24, "spawn.position_y", 3, 8000),
        (25, "spawn.angle_attempt_0", 3, 8000),
        (26, "spawn.position_x", 2, 8000),
        (27, "spawn.position_y", 2, 8000),
        (28, "spawn.angle_attempt_0", 2, 8000),
        (29, "spawn.position_x", 1, 8000),
        (30, "spawn.position_y", 1, 8000),
        (31, "spawn.angle_attempt_0", 1, 8000),
    ]

    after_update = payload["snapshots"][-2]
    assert after_update["label"] == "after_action_8_update"
    assert after_update["game"]["inRound"] is False
    assert after_update["game"]["deaths"] == [4, 3, 2]
    assert [
        (avatar["id"], avatar["alive"], avatar["printing"], avatar["score"])
        for avatar in after_update["avatars"]
    ] == [
        (1, True, True, 3),
        (2, False, False, 2),
        (3, False, False, 1),
        (4, False, False, 0),
    ]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_9_advance_timers"
    assert final_snapshot["game"]["inRound"] is True
    assert final_snapshot["game"]["worldActive"] is False
    assert final_snapshot["game"]["deaths"] == []
    assert [
        (avatar["id"], avatar["x"], avatar["y"], avatar["angle"], avatar["alive"], avatar["score"])
        for avatar in final_snapshot["avatars"]
    ] == [
        (1, 77.41, 50.5, 4.712389, True, 3),
        (2, 59.47, 50.5, 3.141593, True, 2),
        (3, 41.53, 50.5, 0.1, True, 1),
        (4, 23.59, 50.5, 1.570796, True, 0),
    ]


def test_lifecycle_oracle_pins_source_3p_warmup_print_start():
    payload = _run_oracle(_WARMUP_3P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 3
    assert payload["timerAdvancesMs"] == [0, 3000]
    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
    ] == [
        (0, "spawn.position_x", 3, 0),
        (1, "spawn.position_y", 3, 0),
        (2, "spawn.angle_attempt_0", 3, 0),
        (3, "spawn.position_x", 2, 0),
        (4, "spawn.position_y", 2, 0),
        (5, "spawn.angle_attempt_0", 2, 0),
        (6, "spawn.position_x", 1, 0),
        (7, "spawn.position_y", 1, 0),
        (8, "spawn.angle_attempt_0", 1, 0),
        (9, "print_manager.start_distance", 3, 3000),
        (10, "print_manager.start_distance", 2, 3000),
        (11, "print_manager.start_distance", 1, 3000),
    ]

    after_game_start = payload["snapshots"][1]
    after_print_start = payload["snapshots"][2]
    assert after_game_start["label"] == "after_advance_0"
    assert after_game_start["game"]["worldActive"] is True
    assert after_game_start["game"]["worldBodyCount"] == 0
    assert after_print_start["label"] == "after_advance_1"
    assert after_print_start["game"]["inRound"] is True
    assert after_print_start["game"]["worldActive"] is True
    assert after_print_start["game"]["worldBodyCount"] == 3
    assert [
        (avatar["id"], avatar["x"], avatar["y"], avatar["alive"], avatar["printing"])
        for avatar in after_print_start["avatars"]
    ] == [
        (1, 20.591, 47.5, True, True),
        (2, 77.74428, 52.290407, True, True),
        (3, 74.16928, 52.290407, True, True),
    ]


def test_lifecycle_oracle_pins_source_3p_next_round_after_forced_wall_deaths():
    payload = _run_oracle(_NEXT_ROUND_3P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 3

    events = payload["events"]
    round_end_index = next(
        index for index, event in enumerate(events) if event["event"] == "round:end"
    )
    assert events[round_end_index]["atMs"] == 3000
    assert events[round_end_index]["data"] == {"winner": None}

    game_stop_index = next(
        index for index, event in enumerate(events) if event["event"] == "game:stop"
    )
    assert events[game_stop_index]["atMs"] == 8000
    assert events[game_stop_index + 1]["event"] == "round:new"
    assert events[game_stop_index + 1]["atMs"] == 8000

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
        if 12 <= call["index"] <= 23
    ] == [
        (12, "print_manager.stop_distance", 3, 3000),
        (13, "print_manager.stop_distance", 2, 3000),
        (14, "print_manager.stop_distance", 1, 3000),
        (15, "spawn.position_x", 3, 8000),
        (16, "spawn.position_y", 3, 8000),
        (17, "spawn.angle_attempt_0", 3, 8000),
        (18, "spawn.position_x", 2, 8000),
        (19, "spawn.position_y", 2, 8000),
        (20, "spawn.angle_attempt_0", 2, 8000),
        (21, "spawn.position_x", 1, 8000),
        (22, "spawn.position_y", 1, 8000),
        (23, "spawn.angle_attempt_0", 1, 8000),
    ]

    after_update = payload["snapshots"][-2]
    assert after_update["label"] == "after_action_5_update"
    assert after_update["game"]["inRound"] is False
    assert [
        (avatar["id"], avatar["alive"], avatar["printing"])
        for avatar in after_update["avatars"]
    ] == [(1, False, False), (2, False, False), (3, False, False)]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_6_advance_timers"
    assert final_snapshot["game"]["inRound"] is True
    assert final_snapshot["game"]["worldActive"] is False
    assert final_snapshot["game"]["worldBodyCount"] == 0
    assert [
        (avatar["id"], avatar["x"], avatar["y"], avatar["angle"], avatar["alive"])
        for avatar in final_snapshot["avatars"]
    ] == [
        (1, 68.575, 47.5, 4.712389, True),
        (2, 47.5, 47.5, 3.141593, True),
        (3, 26.425, 47.5, 1.570796, True),
    ]


def test_lifecycle_oracle_pins_source_3p_survivor_score_round_end():
    payload = _run_oracle(_SURVIVOR_SCORE_3P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 3

    events = payload["events"]
    assert [event["event"] for event in events[-5:]] == [
        "score:round",
        "score",
        "score",
        "score",
        "round:end",
    ]
    assert events[-5]["data"] == {"avatar": 1, "score": 0, "roundScore": 2}
    assert [event["data"] for event in events[-4:-1]] == [
        {"avatar": 3, "score": 0, "roundScore": 0},
        {"avatar": 2, "score": 0, "roundScore": 0},
        {"avatar": 1, "score": 2, "roundScore": 2},
    ]
    assert events[-1]["atMs"] == 3000
    assert events[-1]["data"] == {"winner": 1}

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
        if call["index"] >= 12
    ] == [
        (12, "print_manager.stop_distance", 3, 3000),
        (13, "print_manager.stop_distance", 2, 3000),
    ]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_4_update"
    assert final_snapshot["game"]["inRound"] is False
    assert final_snapshot["game"]["deathCount"] == 2
    assert final_snapshot["game"]["deaths"] == [3, 2]
    assert [
        (avatar["id"], avatar["alive"], avatar["printing"], avatar["score"], avatar["roundScore"])
        for avatar in final_snapshot["avatars"]
    ] == [
        (1, True, True, 2, 0),
        (2, False, False, 0, 0),
        (3, False, False, 0, 0),
    ]


def test_lifecycle_oracle_pins_source_3p_survivor_warmdown_next_round():
    payload = _run_oracle(_SURVIVOR_NEXT_3P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 3

    events = payload["events"]
    round_end_index = next(
        index for index, event in enumerate(events) if event["event"] == "round:end"
    )
    assert events[round_end_index]["atMs"] == 3000
    assert events[round_end_index]["data"] == {"winner": 1}

    assert [event["event"] for event in events[round_end_index + 1 : round_end_index + 6]] == [
        "point",
        "property",
        "random",
        "die",
        "score:round",
    ]
    assert [event["atMs"] for event in events[round_end_index + 1 : round_end_index + 6]] == [
        4150,
        4150,
        4150,
        4150,
        4150,
    ]
    assert events[round_end_index + 4]["data"] == {
        "avatar": 1,
        "killer": None,
        "old": None,
    }
    assert events[round_end_index + 5]["data"] == {
        "avatar": 1,
        "score": 2,
        "roundScore": 2,
    }

    game_stop_index = next(
        index for index, event in enumerate(events) if event["event"] == "game:stop"
    )
    assert events[game_stop_index]["atMs"] == 8000
    assert events[game_stop_index + 1]["event"] == "round:new"
    assert events[game_stop_index + 1]["atMs"] == 8000

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], round(call["atMs"]))
        for call in payload["randomCalls"]
        if 14 <= call["index"] <= 23
    ] == [
        (14, "print_manager.stop_distance", 1, 4150),
        (15, "spawn.position_x", 3, 8000),
        (16, "spawn.position_y", 3, 8000),
        (17, "spawn.angle_attempt_0", 3, 8000),
        (18, "spawn.position_x", 2, 8000),
        (19, "spawn.position_y", 2, 8000),
        (20, "spawn.angle_attempt_0", 2, 8000),
        (21, "spawn.position_x", 1, 8000),
        (22, "spawn.position_y", 1, 8000),
        (23, "spawn.angle_attempt_0", 1, 8000),
    ]

    after_update = payload["snapshots"][-2]
    assert after_update["label"] == "after_action_4_update"
    assert after_update["game"]["inRound"] is False
    assert after_update["game"]["deaths"] == [3, 2]
    assert [
        (avatar["id"], avatar["x"], avatar["alive"], avatar["printing"], avatar["score"])
        for avatar in after_update["avatars"]
    ] == [
        (1, 18.991, True, True, 2),
        (2, -0.6, False, False, 0),
        (3, 94.6, False, False, 0),
    ]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_5_advance_timers"
    assert final_snapshot["game"]["inRound"] is True
    assert final_snapshot["game"]["worldActive"] is False
    assert final_snapshot["game"]["deaths"] == []
    assert [
        (avatar["id"], avatar["x"], avatar["y"], avatar["angle"], avatar["alive"], avatar["score"])
        for avatar in final_snapshot["avatars"]
    ] == [
        (1, 68.575, 47.5, 4.712389, True, 2),
        (2, 47.5, 47.5, 3.141593, True, 0),
        (3, 26.425, 47.5, 1.570796, True, 0),
    ]


def test_lifecycle_oracle_pins_source_3p_remove_avatar_during_warmdown():
    payload = _run_oracle(_REMOVE_AVATAR_WARMDOWN_3P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 3

    events = payload["events"]
    assert "end" not in [event["event"] for event in events]
    assert [event["event"] for event in events].count("round:end") == 1

    round_end_index = next(
        index for index, event in enumerate(events) if event["event"] == "round:end"
    )
    leave_index = next(
        index for index, event in enumerate(events) if event["event"] == "player:leave"
    )
    game_stop_index = next(
        index for index, event in enumerate(events) if event["event"] == "game:stop"
    )

    assert events[round_end_index]["atMs"] == 3000
    assert events[round_end_index]["data"] == {"winner": 1}
    assert [event["event"] for event in events[round_end_index + 1 : leave_index + 1]] == [
        "point",
        "property",
        "random",
        "die",
        "player:leave",
    ]
    assert [
        event["event"]
        for event in events[round_end_index + 1 : game_stop_index]
        if event["event"] in {"score", "score:round", "round:end"}
    ] == []
    assert events[leave_index - 1]["data"] == {
        "avatar": 1,
        "killer": None,
        "old": None,
    }
    assert events[leave_index]["data"] == {"player": 1}
    assert events[game_stop_index]["atMs"] == 8000
    assert events[game_stop_index + 1]["event"] == "round:new"
    assert events[game_stop_index + 1]["atMs"] == 8000

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
        if 12 <= call["index"] <= 20
    ] == [
        (12, "print_manager.stop_distance", 3, 3000),
        (13, "print_manager.stop_distance", 2, 3000),
        (14, "print_manager.stop_distance", 1, 3000),
        (15, "spawn.position_x", 3, 8000),
        (16, "spawn.position_y", 3, 8000),
        (17, "spawn.angle_attempt_0", 3, 8000),
        (18, "spawn.position_x", 2, 8000),
        (19, "spawn.position_y", 2, 8000),
        (20, "spawn.angle_attempt_0", 2, 8000),
    ]

    after_round_end = payload["snapshots"][-3]
    assert after_round_end["label"] == "after_action_4_update"
    assert after_round_end["game"]["inRound"] is False
    assert after_round_end["game"]["deaths"] == [3, 2]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in after_round_end["avatars"]
    ] == [(1, True, True, 2), (2, False, True, 0), (3, False, True, 0)]

    after_leave = payload["snapshots"][-2]
    assert after_leave["label"] == "after_action_5_remove_avatar"
    assert after_leave["game"]["size"] == 95
    assert after_leave["game"]["inRound"] is False
    assert after_leave["game"]["deaths"] == [3, 2]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in after_leave["avatars"]
    ] == [(1, False, False, 2), (2, False, True, 0), (3, False, True, 0)]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_6_advance_timers"
    assert final_snapshot["game"]["size"] == 88
    assert final_snapshot["game"]["inRound"] is True
    assert final_snapshot["game"]["worldActive"] is False
    assert final_snapshot["game"]["deaths"] == [1]
    assert [
        (avatar["id"], avatar["x"], avatar["y"], avatar["angle"], avatar["alive"], avatar["present"])
        for avatar in final_snapshot["avatars"]
    ] == [
        (1, 0.6, 0.6, 0, False, False),
        (2, 63.5, 44, 4.712389, True, True),
        (3, 24.5, 44, 1.570796, True, True),
    ]


def test_lifecycle_oracle_pins_source_3p_remove_avatar_to_single_present():
    payload = _run_oracle(_REMOVE_AVATAR_TO_SINGLE_PRESENT_3P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 3

    events = payload["events"]
    event_names = [event["event"] for event in events]
    assert "end" not in event_names
    assert event_names.count("round:end") == 1
    assert [event["data"]["avatar"] for event in events if event["event"] == "die"] == [
        3,
        2,
    ]

    leave_index = next(
        index for index, event in enumerate(events) if event["event"] == "player:leave"
    )
    round_end_index = next(
        index for index, event in enumerate(events) if event["event"] == "round:end"
    )
    game_stop_index = next(
        index for index, event in enumerate(events) if event["event"] == "game:stop"
    )

    assert events[leave_index - 1]["data"] == {
        "avatar": 2,
        "killer": None,
        "old": None,
    }
    assert events[leave_index]["data"] == {"player": 2}
    assert [event["event"] for event in events[leave_index : round_end_index + 1]] == [
        "player:leave",
        "score:round",
        "score",
        "score",
        "score",
        "round:end",
    ]
    assert events[round_end_index]["data"] == {"winner": 1}
    assert events[game_stop_index]["atMs"] == 8000
    assert events[game_stop_index + 1]["event"] == "round:new"

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
        if call["index"] >= 12
    ] == [
        (12, "print_manager.stop_distance", 3, 3000),
        (13, "print_manager.stop_distance", 2, 3000),
        (14, "print_manager.stop_distance", 1, 8000),
        (15, "spawn.position_x", 3, 8000),
        (16, "spawn.position_y", 3, 8000),
        (17, "spawn.angle_attempt_0", 3, 8000),
        (18, "spawn.position_x", 1, 8000),
        (19, "spawn.position_y", 1, 8000),
        (20, "spawn.angle_attempt_0", 1, 8000),
    ]

    after_first_death = payload["snapshots"][4]
    assert after_first_death["label"] == "after_action_3_update"
    assert after_first_death["game"]["inRound"] is True
    assert after_first_death["game"]["deaths"] == [3]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in after_first_death["avatars"]
    ] == [(1, True, True, 0), (2, True, True, 0), (3, False, True, 0)]

    after_leave = payload["snapshots"][6]
    assert after_leave["label"] == "after_action_5_remove_avatar"
    assert after_leave["game"]["inRound"] is False
    assert after_leave["game"]["deaths"] == [3]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in after_leave["avatars"]
    ] == [(1, True, True, 2), (2, False, False, 0), (3, False, True, 0)]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_6_advance_timers"
    assert final_snapshot["game"]["size"] == 88
    assert final_snapshot["game"]["inRound"] is True
    assert final_snapshot["game"]["deaths"] == [2]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in final_snapshot["avatars"]
    ] == [(1, True, True, 2), (2, False, False, 0), (3, True, True, 0)]


def test_lifecycle_oracle_pins_source_3p_present_absent_round_new():
    payload = _run_oracle(_PRESENT_ABSENT_3P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 3
    assert payload["timerAdvancesMs"] == []
    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
    ] == [
        (0, "spawn.position_x", 3, 0),
        (1, "spawn.position_y", 3, 0),
        (2, "spawn.angle_attempt_0", 3, 0),
        (3, "spawn.position_x", 1, 0),
        (4, "spawn.position_y", 1, 0),
        (5, "spawn.angle_attempt_0", 1, 0),
    ]

    snapshot = payload["snapshots"][0]
    assert snapshot["label"] == "after_new_round_call"
    assert snapshot["game"]["size"] == 95
    assert snapshot["game"]["deathCount"] == 1
    assert snapshot["game"]["deaths"] == [2]
    assert snapshot["game"]["worldActive"] is False
    assert snapshot["game"]["worldBodyCount"] == 0
    assert [
        (avatar["id"], avatar["x"], avatar["y"], avatar["angle"], avatar["alive"], avatar["present"])
        for avatar in snapshot["avatars"]
    ] == [
        (1, 68.575, 47.5, 4.712389, True, True),
        (2, 0.6, 0.6, 0, False, False),
        (3, 26.425, 47.5, 1.570796, True, True),
    ]


def test_lifecycle_oracle_pins_source_3p_present_absent_survivor_score_round_end():
    payload = _run_oracle(_PRESENT_ABSENT_SURVIVOR_3P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 3

    events = payload["events"]
    assert [
        event["data"]["avatar"]
        for event in events
        if event["event"] == "die"
    ] == [3]

    assert [event["event"] for event in events[-6:]] == [
        "score:round",
        "score:round",
        "score",
        "score",
        "score",
        "round:end",
    ]
    assert events[-6]["data"] == {"avatar": 3, "score": 0, "roundScore": 1}
    assert events[-5]["data"] == {"avatar": 1, "score": 0, "roundScore": 2}
    assert [event["data"] for event in events[-4:-1]] == [
        {"avatar": 3, "score": 1, "roundScore": 1},
        {"avatar": 2, "score": 0, "roundScore": 0},
        {"avatar": 1, "score": 2, "roundScore": 2},
    ]
    assert events[-1]["atMs"] == 3000
    assert events[-1]["data"] == {"winner": 1}

    first_snapshot = payload["snapshots"][0]
    assert first_snapshot["game"]["deathCount"] == 1
    assert first_snapshot["game"]["deaths"] == [2]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_3_update"
    assert final_snapshot["game"]["inRound"] is False
    assert final_snapshot["game"]["deathCount"] == 2
    assert final_snapshot["game"]["deaths"] == [2, 3]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"], avatar["roundScore"])
        for avatar in final_snapshot["avatars"]
    ] == [
        (1, True, True, 2, 0),
        (2, False, False, 0, 0),
        (3, False, True, 1, 0),
    ]
    absent_avatar = final_snapshot["avatars"][1]
    assert absent_avatar["printing"] is True
    assert absent_avatar["printManager"]["active"] is True


def test_lifecycle_oracle_pins_source_3p_present_absent_next_round():
    payload = _run_oracle(_PRESENT_ABSENT_NEXT_3P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 3

    events = payload["events"]
    round_end_index = next(
        index for index, event in enumerate(events) if event["event"] == "round:end"
    )
    assert events[round_end_index]["atMs"] == 3000
    assert events[round_end_index]["data"] == {"winner": None}
    assert [event["event"] for event in events[round_end_index + 1 : round_end_index + 3]] == [
        "game:stop",
        "round:new",
    ]
    assert [event["atMs"] for event in events[round_end_index + 1 : round_end_index + 3]] == [
        8000,
        8000,
    ]

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
        if call["index"] >= 11
    ] == [
        (11, "spawn.position_x", 3, 8000),
        (12, "spawn.position_y", 3, 8000),
        (13, "spawn.angle_attempt_0", 3, 8000),
        (14, "spawn.position_x", 1, 8000),
        (15, "spawn.position_y", 1, 8000),
        (16, "spawn.angle_attempt_0", 1, 8000),
    ]

    after_update = payload["snapshots"][-2]
    assert after_update["label"] == "after_action_4_update"
    assert after_update["game"]["size"] == 95
    assert after_update["game"]["deaths"] == [2, 3, 1]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in after_update["avatars"]
    ] == [
        (1, False, True, 1),
        (2, False, False, 0),
        (3, False, True, 1),
    ]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_5_advance_timers"
    assert final_snapshot["game"]["size"] == 88
    assert final_snapshot["game"]["inRound"] is True
    assert final_snapshot["game"]["worldActive"] is False
    assert final_snapshot["game"]["deaths"] == [2]
    assert [
        (
            avatar["id"],
            avatar["x"],
            avatar["y"],
            avatar["angle"],
            avatar["alive"],
            avatar["present"],
            avatar["score"],
        )
        for avatar in final_snapshot["avatars"]
    ] == [
        (1, 63.5, 44, 4.712389, True, True, 1),
        (2, 0.6, 0.6, 0, False, False, 0),
        (3, 24.5, 44, 1.570796, True, True, 1),
    ]
    absent_avatar = final_snapshot["avatars"][1]
    assert absent_avatar["printing"] is True
    assert absent_avatar["printManager"]["active"] is True


def test_lifecycle_oracle_pins_source_3p_present_absent_tie_at_max_score():
    payload = _run_oracle(_PRESENT_ABSENT_TIE_MAX_3P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 3

    events = payload["events"]
    assert "end" not in [event["event"] for event in events]
    round_end_index = next(
        index for index, event in enumerate(events) if event["event"] == "round:end"
    )
    assert events[round_end_index]["atMs"] == 3000
    assert events[round_end_index]["data"] == {"winner": None}
    assert [event["event"] for event in events[round_end_index + 1 : round_end_index + 3]] == [
        "game:stop",
        "round:new",
    ]
    assert [event["atMs"] for event in events[round_end_index + 1 : round_end_index + 3]] == [
        8000,
        8000,
    ]
    assert [
        event["data"]["avatar"]
        for event in events
        if event["event"] == "die"
    ] == [3, 1]
    assert [event["data"] for event in events[round_end_index - 3 : round_end_index]] == [
        {"avatar": 3, "score": 1, "roundScore": 1},
        {"avatar": 2, "score": 0, "roundScore": 0},
        {"avatar": 1, "score": 1, "roundScore": 1},
    ]

    after_update = payload["snapshots"][-2]
    assert after_update["label"] == "after_action_4_update"
    assert after_update["game"]["deaths"] == [2, 3, 1]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in after_update["avatars"]
    ] == [
        (1, False, True, 1),
        (2, False, False, 0),
        (3, False, True, 1),
    ]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_5_advance_timers"
    assert final_snapshot["game"]["inRound"] is True
    assert final_snapshot["game"]["deaths"] == [2]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in final_snapshot["avatars"]
    ] == [
        (1, True, True, 1),
        (2, False, False, 0),
        (3, True, True, 1),
    ]


def test_lifecycle_oracle_pins_source_4p_present_absent_round_new():
    payload = _run_oracle(_PRESENT_ABSENT_4P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 4
    assert payload["timerAdvancesMs"] == []
    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
    ] == [
        (0, "spawn.position_x", 4, 0),
        (1, "spawn.position_y", 4, 0),
        (2, "spawn.angle_attempt_0", 4, 0),
        (3, "spawn.position_x", 3, 0),
        (4, "spawn.position_y", 3, 0),
        (5, "spawn.angle_attempt_0", 3, 0),
        (6, "spawn.position_x", 1, 0),
        (7, "spawn.position_y", 1, 0),
        (8, "spawn.angle_attempt_0", 1, 0),
    ]

    snapshot = payload["snapshots"][0]
    assert snapshot["label"] == "after_new_round_call"
    assert snapshot["game"]["size"] == 101
    assert snapshot["game"]["deathCount"] == 1
    assert snapshot["game"]["deaths"] == [2]
    assert snapshot["game"]["worldActive"] is False
    assert snapshot["game"]["worldBodyCount"] == 0
    assert [
        (
            avatar["id"],
            avatar["x"],
            avatar["y"],
            avatar["angle"],
            avatar["alive"],
            avatar["present"],
        )
        for avatar in snapshot["avatars"]
    ] == [
        (1, 77.41, 50.5, 4.712389, True, True),
        (2, 0.6, 0.6, 0, False, False),
        (3, 41.53, 50.5, 0.1, True, True),
        (4, 23.59, 50.5, 1.570796, True, True),
    ]


def test_lifecycle_oracle_pins_source_4p_present_absent_survivor_score_round_end():
    payload = _run_oracle(_PRESENT_ABSENT_SURVIVOR_4P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 4

    events = payload["events"]
    assert [
        event["data"]["avatar"]
        for event in events
        if event["event"] == "die"
    ] == [4, 3]

    assert [event["event"] for event in events[-7:]] == [
        "score:round",
        "score:round",
        "score",
        "score",
        "score",
        "score",
        "round:end",
    ]
    assert events[-7]["data"] == {"avatar": 3, "score": 0, "roundScore": 2}
    assert events[-6]["data"] == {"avatar": 1, "score": 0, "roundScore": 3}
    assert [event["data"] for event in events[-5:-1]] == [
        {"avatar": 4, "score": 1, "roundScore": 1},
        {"avatar": 3, "score": 2, "roundScore": 2},
        {"avatar": 2, "score": 0, "roundScore": 0},
        {"avatar": 1, "score": 3, "roundScore": 3},
    ]
    assert events[-1]["atMs"] == 3000
    assert events[-1]["data"] == {"winner": 1}

    first_snapshot = payload["snapshots"][0]
    assert first_snapshot["game"]["deathCount"] == 1
    assert first_snapshot["game"]["deaths"] == [2]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_6_update"
    assert final_snapshot["game"]["inRound"] is False
    assert final_snapshot["game"]["deathCount"] == 3
    assert final_snapshot["game"]["deaths"] == [2, 4, 3]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"], avatar["roundScore"])
        for avatar in final_snapshot["avatars"]
    ] == [
        (1, True, True, 3, 0),
        (2, False, False, 0, 0),
        (3, False, True, 2, 0),
        (4, False, True, 1, 0),
    ]
    absent_avatar = final_snapshot["avatars"][1]
    assert absent_avatar["printing"] is True
    assert absent_avatar["printManager"]["active"] is True


def test_lifecycle_oracle_pins_source_4p_present_absent_next_round():
    payload = _run_oracle(_PRESENT_ABSENT_NEXT_4P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 4

    events = payload["events"]
    round_end_index = next(
        index for index, event in enumerate(events) if event["event"] == "round:end"
    )
    assert events[round_end_index]["atMs"] == 3000
    assert events[round_end_index]["data"] == {"winner": None}
    assert [event["event"] for event in events[round_end_index + 1 : round_end_index + 3]] == [
        "game:stop",
        "round:new",
    ]
    assert [event["atMs"] for event in events[round_end_index + 1 : round_end_index + 3]] == [
        8000,
        8000,
    ]

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
        if call["index"] >= 16
    ] == [
        (16, "spawn.position_x", 4, 8000),
        (17, "spawn.position_y", 4, 8000),
        (18, "spawn.angle_attempt_0", 4, 8000),
        (19, "spawn.position_x", 3, 8000),
        (20, "spawn.position_y", 3, 8000),
        (21, "spawn.angle_attempt_0", 3, 8000),
        (22, "spawn.position_x", 1, 8000),
        (23, "spawn.position_y", 1, 8000),
        (24, "spawn.angle_attempt_0", 1, 8000),
    ]

    after_update = payload["snapshots"][-2]
    assert after_update["label"] == "after_action_5_update"
    assert after_update["game"]["size"] == 101
    assert after_update["game"]["deaths"] == [2, 4, 3, 1]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in after_update["avatars"]
    ] == [
        (1, False, True, 1),
        (2, False, False, 0),
        (3, False, True, 1),
        (4, False, True, 1),
    ]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_6_advance_timers"
    assert final_snapshot["game"]["size"] == 95
    assert final_snapshot["game"]["inRound"] is True
    assert final_snapshot["game"]["worldActive"] is False
    assert final_snapshot["game"]["deaths"] == [2]
    assert [
        (
            avatar["id"],
            avatar["x"],
            avatar["y"],
            avatar["angle"],
            avatar["alive"],
            avatar["present"],
            avatar["score"],
        )
        for avatar in final_snapshot["avatars"]
    ] == [
        (1, 72.79, 47.5, 4.712389, True, True, 1),
        (2, 0.6, 0.6, 0, False, False, 0),
        (3, 39.07, 47.5, 0.1, True, True, 1),
        (4, 22.21, 47.5, 1.570796, True, True, 1),
    ]
    absent_avatar = final_snapshot["avatars"][1]
    assert absent_avatar["printing"] is True
    assert absent_avatar["printManager"]["active"] is True


def test_lifecycle_oracle_pins_source_4p_present_absent_tie_at_max_score():
    payload = _run_oracle(_PRESENT_ABSENT_TIE_MAX_4P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 4

    events = payload["events"]
    assert "end" not in [event["event"] for event in events]
    round_end_index = next(
        index for index, event in enumerate(events) if event["event"] == "round:end"
    )
    assert events[round_end_index]["atMs"] == 3000
    assert events[round_end_index]["data"] == {"winner": None}
    assert [event["event"] for event in events[round_end_index + 1 : round_end_index + 3]] == [
        "game:stop",
        "round:new",
    ]
    assert [event["atMs"] for event in events[round_end_index + 1 : round_end_index + 3]] == [
        8000,
        8000,
    ]
    assert [
        event["data"]["avatar"]
        for event in events
        if event["event"] == "die"
    ] == [4, 3, 1]
    assert [event["data"] for event in events[round_end_index - 4 : round_end_index]] == [
        {"avatar": 4, "score": 1, "roundScore": 1},
        {"avatar": 3, "score": 1, "roundScore": 1},
        {"avatar": 2, "score": 0, "roundScore": 0},
        {"avatar": 1, "score": 1, "roundScore": 1},
    ]

    after_update = payload["snapshots"][-2]
    assert after_update["label"] == "after_action_5_update"
    assert after_update["game"]["deaths"] == [2, 4, 3, 1]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in after_update["avatars"]
    ] == [
        (1, False, True, 1),
        (2, False, False, 0),
        (3, False, True, 1),
        (4, False, True, 1),
    ]

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_6_advance_timers"
    assert final_snapshot["game"]["inRound"] is True
    assert final_snapshot["game"]["deaths"] == [2]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"])
        for avatar in final_snapshot["avatars"]
    ] == [
        (1, True, True, 1),
        (2, False, False, 0),
        (3, True, True, 1),
        (4, True, True, 1),
    ]


def test_lifecycle_oracle_pins_source_3p_mid_round_remove_avatar_continuation():
    payload = _run_oracle(_MID_ROUND_REMOVE_AVATAR_3P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 3

    events = payload["events"]
    leave_index = next(
        index for index, event in enumerate(events) if event["event"] == "player:leave"
    )
    assert events[leave_index - 1]["event"] == "die"
    assert events[leave_index - 1]["data"] == {
        "avatar": 2,
        "killer": None,
        "old": None,
    }
    assert events[leave_index]["data"] == {"player": 2}
    assert "round:end" not in [event["event"] for event in events[: leave_index + 1]]
    assert [
        event["data"]["avatar"]
        for event in events
        if event["event"] == "die"
    ] == [2, 3]

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"][-2:]
    ] == [
        (12, "print_manager.stop_distance", 2, 3000),
        (13, "print_manager.stop_distance", 3, 3000),
    ]

    after_leave = payload["snapshots"][3]
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

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_4_update"
    assert final_snapshot["game"]["inRound"] is False
    assert final_snapshot["game"]["deaths"] == [3]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"], avatar["roundScore"])
        for avatar in final_snapshot["avatars"]
    ] == [
        (1, True, True, 2, 0),
        (2, False, False, 0, 0),
        (3, False, True, 0, 0),
    ]
    assert [event["data"] for event in events[-6:]] == [
        {"avatar": 3, "score": 0, "roundScore": 0},
        {"avatar": 1, "score": 0, "roundScore": 2},
        {"avatar": 3, "score": 0, "roundScore": 0},
        {"avatar": 2, "score": 0, "roundScore": 0},
        {"avatar": 1, "score": 2, "roundScore": 2},
        {"winner": 1},
    ]


def test_lifecycle_oracle_pins_source_4p_mid_round_remove_avatar_continuation():
    payload = _run_oracle(_MID_ROUND_REMOVE_AVATAR_4P_SCENARIO)

    assert payload["expectations"]["eventOrder"]["status"] == "pass"
    assert payload["playerCount"] == 4

    events = payload["events"]
    leave_index = next(
        index for index, event in enumerate(events) if event["event"] == "player:leave"
    )
    assert events[leave_index - 1]["event"] == "die"
    assert events[leave_index - 1]["data"] == {
        "avatar": 2,
        "killer": None,
        "old": None,
    }
    assert events[leave_index]["data"] == {"player": 2}
    assert "round:end" not in [event["event"] for event in events[: leave_index + 1]]
    assert [event["data"]["avatar"] for event in events if event["event"] == "die"] == [
        2,
        4,
        3,
    ]

    assert [
        (call["index"], call["label"]["site"], call["label"]["avatar"], call["atMs"])
        for call in payload["randomCalls"]
        if call["index"] >= 16
    ] == [
        (16, "print_manager.stop_distance", 2, 3000),
        (17, "print_manager.stop_distance", 4, 3000),
        (18, "print_manager.stop_distance", 3, 3000),
    ]

    after_leave = payload["snapshots"][4]
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

    after_first_death = payload["snapshots"][6]
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

    final_snapshot = payload["snapshots"][-1]
    assert final_snapshot["label"] == "after_action_7_update"
    assert final_snapshot["game"]["inRound"] is False
    assert final_snapshot["game"]["deaths"] == [4, 3]
    assert [
        (avatar["id"], avatar["alive"], avatar["present"], avatar["score"], avatar["roundScore"])
        for avatar in final_snapshot["avatars"]
    ] == [
        (1, True, True, 3, 0),
        (2, False, False, 0, 0),
        (3, False, True, 1, 0),
        (4, False, True, 0, 0),
    ]
    assert [event["data"] for event in events[-6:]] == [
        {"avatar": 1, "score": 0, "roundScore": 3},
        {"avatar": 4, "score": 0, "roundScore": 0},
        {"avatar": 3, "score": 1, "roundScore": 1},
        {"avatar": 2, "score": 0, "roundScore": 0},
        {"avatar": 1, "score": 3, "roundScore": 3},
        {"winner": 1},
    ]
