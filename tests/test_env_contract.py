import numpy as np
import pytest

from curvyzero.env import CurvyTronConfig, CurvyTronEnv
from curvyzero.env.trainer_contract import ACTION_ID_TO_SOURCE_MOVE
from curvyzero.env.trainer_contract import ACTION_NAMES


def test_observe_returns_copied_current_debug_observation():
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1))
    observations = env.reset(seed=7)

    observed = env.observe("player_0")
    np.testing.assert_array_equal(observed, observations["player_0"])

    observed[0] = -999.0
    assert env.observe("player_0")[0] != -999.0


def test_legal_action_mask_uses_turn3_order_for_live_player():
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1))
    env.reset(seed=7)

    mask = env.legal_action_mask("player_0")

    assert mask.dtype == np.bool_
    assert mask.shape == (3,)
    np.testing.assert_array_equal(mask, np.array([True, True, True], dtype=np.bool_))


def test_action_one_is_the_explicit_no_turn_noop_action():
    assert ACTION_NAMES == ("left", "straight", "right")
    assert ACTION_ID_TO_SOURCE_MOVE == (-1, 0, 1)
    assert ACTION_NAMES[1] == "straight"
    assert ACTION_ID_TO_SOURCE_MOVE[1] == 0


def test_dead_player_does_not_need_action_and_has_empty_mask():
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1))
    env.reset(seed=7)
    assert env.state is not None
    env.state.alive[1] = False

    mask = env.legal_action_mask("player_1")
    result = env.step({"player_0": 1})

    np.testing.assert_array_equal(mask, np.array([False, False, False], dtype=np.bool_))
    np.testing.assert_array_equal(
        env.legal_action_mask("player_0"), np.array([False, False, False], dtype=np.bool_)
    )
    assert result.terminated["player_0"]
    assert result.terminated["player_1"]


def test_terminal_step_sets_final_observation_and_requires_reset():
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1))
    env.reset(seed=7)
    assert env.last_reset_info is not None
    first_episode_id = env.last_reset_info["episode_id"]
    assert env.state is not None
    env.state.alive[1] = False

    result = env.step({"player_0": 1})

    for agent, info in result.infos.items():
        assert info["done"] is True
        assert info["terminated"] is True
        assert info["truncated"] is False
        assert info["needs_reset"] is True
        np.testing.assert_array_equal(info["final_observation"], result.observations[agent])

    with pytest.raises(RuntimeError, match="terminal or truncated episode"):
        env.step({"player_0": 1})

    env.reset(seed=8)

    assert env.last_reset_info is not None
    assert env.last_reset_info["needs_reset"] is False
    assert env.last_reset_info["episode_id"] != first_episode_id
    result = env.step({"player_0": 1, "player_1": 1})
    assert result.infos["player_0"]["needs_reset"] is False


def test_truncation_sets_timeout_metadata_and_requires_reset():
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1, max_ticks=1))
    env.reset(seed=7)

    result = env.step({"player_0": 1, "player_1": 1})

    for agent, info in result.infos.items():
        assert result.terminated[agent] is False
        assert result.truncated[agent] is True
        assert info["done"] is True
        assert info["terminated"] is False
        assert info["truncated"] is True
        assert info["needs_reset"] is True
        assert info["terminal_reason"] == "timeout"
        assert info["truncation_reason"] == "max_ticks"
        np.testing.assert_array_equal(info["final_observation"], result.observations[agent])

    with pytest.raises(RuntimeError, match="terminal or truncated episode"):
        env.step({"player_0": 1, "player_1": 1})


def test_missing_live_player_action_is_an_error():
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1))
    env.reset(seed=7)

    with pytest.raises(ValueError, match="missing actions for live players: player_1"):
        env.step({"player_0": 1})


def test_step_infos_include_toy_v0_schema_ids():
    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1))
    env.reset(seed=7)

    result = env.step({"player_0": 1, "player_1": 1})

    for info in result.infos.values():
        assert info["ruleset_id"] == "curvyzero-v0"
        assert info["observation_schema_id"] == "curvyzero_debug_global_player_obs/v0"
        assert info["action_space_id"] == "curvyzero_turn3/v0"
        assert info["reward_schema_id"] == "curvyzero_sparse_round_outcome/v0"
        assert info["env_impl_id"] == "curvyzero_python_toy_v0_env/v0"
        assert info["terminal_reason"] == "none"
        assert info["done"] is False
        assert info["terminated"] is False
        assert info["truncated"] is False
        assert info["needs_reset"] is False
        assert info["final_observation"] is None
        assert info["step_index"] == 1
        assert info["tick_index"] == 1
        assert info["player_ids"] == ["player_0", "player_1"]
        assert info["observation_schema_hash"]
        assert info["action_space_hash"]
        assert info["reward_schema_hash"]
