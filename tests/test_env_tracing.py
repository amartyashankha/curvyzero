from curvyzero.env.config import CurvyTronConfig
from curvyzero.env.core import CurvyTronEnv
from curvyzero.env.tracing import TRACE_SCOPE, fingerprint_scripted_actions, trace_scripted_actions


def make_env() -> CurvyTronEnv:
    return CurvyTronEnv(CurvyTronConfig(action_repeat=1, max_ticks=64))


def test_trace_fingerprint_is_stable_for_same_seed_and_actions():
    actions = [
        {"player_0": 1, "player_1": 1},
        {"player_0": 2, "player_1": 0},
        {"player_0": 2, "player_1": 0},
        {"player_0": 1, "player_1": 1},
    ]

    first = trace_scripted_actions(make_env(), actions, seed=123)
    second = trace_scripted_actions(make_env(), actions, seed=123)

    assert first.scope == TRACE_SCOPE
    assert first.fingerprint == second.fingerprint
    assert first.to_payload() == second.to_payload()


def test_trace_records_toy_v0_state_and_step_flags():
    trace = trace_scripted_actions(
        make_env(),
        [{"player_0": 1, "player_1": 1}],
        seed=123,
    )

    reset_frame = trace.frames[0]
    step_frame = trace.frames[1]

    assert trace.scope == "curvyzero-v0-python-toy"
    assert reset_frame.tick == 0
    assert len(reset_frame.positions) == 2
    assert len(reset_frame.headings) == 2
    assert reset_frame.alive == (True, True)
    assert reset_frame.scores == (0, 0)
    assert reset_frame.round_scores == (0, 0)
    assert reset_frame.rewards is None
    assert reset_frame.terminated is None
    assert reset_frame.truncated is None

    assert step_frame.tick == 1
    assert step_frame.rewards == {"player_0": 0.0, "player_1": 0.0}
    assert step_frame.terminated == {"player_0": False, "player_1": False}
    assert step_frame.truncated == {"player_0": False, "player_1": False}


def test_trace_fingerprint_changes_when_actions_change():
    base_actions = [
        {"player_0": 1, "player_1": 1},
        {"player_0": 2, "player_1": 0},
        {"player_0": 2, "player_1": 0},
    ]
    changed_actions = [
        {"player_0": 0, "player_1": 2},
        {"player_0": 2, "player_1": 0},
        {"player_0": 2, "player_1": 0},
    ]

    assert fingerprint_scripted_actions(make_env(), base_actions, seed=123) != (
        fingerprint_scripted_actions(make_env(), changed_actions, seed=123)
    )
