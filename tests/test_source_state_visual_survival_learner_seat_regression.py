import numpy as np
import pytest

from curvyzero.env.observation_surface_contract import (
    POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID,
)
from curvyzero.training import curvyzero_source_state_visual_survival_lightzero_env as env_mod
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv,
    OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
    STACKED_SOURCE_STATE_GRAY64_SHAPE,
)
from curvyzero.training.multiplayer_opponent_policy import OpponentPolicySelection


LEFT_ACTION_ID = 0
STRAIGHT_ACTION_ID = 1
RIGHT_ACTION_ID = 2


def test_fixed_learner_seat_1_resets_and_legacy_ego_config_rejects():
    with pytest.raises(ValueError, match="ego_player_index config is removed"):
        CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
            {"ego_player_index": 2}
        )

    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 401,
            "source_max_steps": 16,
            "natural_bonus_spawn": False,
            "learner_seat_mode": "fixed_player_1",
        }
    )

    observation = env.reset(seed=401)

    assert env.ego_player_index == 1
    assert env.opponent_player_index == 0
    assert env.ego_player_id == "player_1"
    assert env.opponent_player_id == "player_0"
    assert observation["to_play"] == -1
    assert observation["observation"].shape == STACKED_SOURCE_STATE_GRAY64_SHAPE
    np.testing.assert_array_equal(
        observation["action_mask"],
        env._env._action_mask()[0, 1].astype(np.int8, copy=True),
    )
    assert env.last_reset_info["ego_player_index"] == 1
    assert env.last_reset_info["opponent_player_index"] == 0
    assert env.last_reset_info["learner_seat_assignment_schema_id"]
    assert (
        env.last_reset_info["player_perspective_schema_id"]
        == POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID
    )
    assert env.last_reset_info["source_state_player_perspective"] is True
    assert env.last_reset_info["observation_perspective_player_id"] == "player_1"


def test_fixed_learner_seat_1_routes_learner_action_to_player_1_and_reports_control():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 403,
            "source_max_steps": 16,
            "natural_bonus_spawn": False,
            "learner_seat_mode": "fixed_player_1",
            "policy_action_repeat_min": 1,
            "policy_action_repeat_max": 1,
        }
    )
    env.reset(seed=403)

    timestep = env.step(LEFT_ACTION_ID)

    assert timestep.info["joint_action"] == {
        "player_0": STRAIGHT_ACTION_ID,
        "player_1": LEFT_ACTION_ID,
    }
    assert timestep.info["requested_ego_action"] == LEFT_ACTION_ID
    assert timestep.info["executed_ego_action"] == LEFT_ACTION_ID
    assert timestep.info["opponent_action_id"] == STRAIGHT_ACTION_ID
    assert timestep.info["acting_player_id"] == "player_1"
    assert timestep.info["controlled_player_id"] == "player_1"
    assert timestep.info["ego_controlled_player_id"] == "player_1"
    assert timestep.info["opponent_controlled_player_id"] == "player_0"
    assert timestep.info["reward_player_id"] == "player_1"
    assert timestep.info["ego_player_index"] == 1
    assert timestep.info["opponent_player_index"] == 0
    assert timestep.info["source_state_player_perspective"] is True
    assert timestep.info["observation_perspective_player_id"] == "player_1"


def test_frozen_opponent_provider_for_fixed_learner_seat_1_controls_player_0(monkeypatch):
    calls = []

    class FakeFrozenOpponentPolicy:
        policy_id = "fake_snapshot_policy"
        policy_version = "v-test"
        seed = 499

        def select_actions(
            self,
            legal_action_mask,
            opponent_mask,
            *,
            decision_index=0,
            observation=None,
        ):
            calls.append(
                {
                    "legal_action_mask": legal_action_mask.copy(),
                    "opponent_mask": opponent_mask.copy(),
                    "decision_index": decision_index,
                    "observation": observation.copy(),
                }
            )
            actions = np.full((1, 2), -1, dtype=np.int16)
            actions[0, 0] = RIGHT_ACTION_ID
            action_seed = np.full((1, 2), -1, dtype=np.int64)
            action_seed[0, 0] = 4990
            action_logp = np.full((1, 2), np.nan, dtype=np.float32)
            action_logp[0, 0] = -0.125
            return OpponentPolicySelection(
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                seed=self.seed,
                actions=actions,
                action_seed=action_seed,
                action_logp=action_logp,
                opponent_mask=opponent_mask,
                decision_index=decision_index,
                policy_metadata={"provider_id": "fake_provider"},
            )

    def fake_build(cfg, *, seed):
        assert cfg["opponent_checkpoint_path"] == "/tmp/frozen-seat0.pth.tar"
        assert seed == 499
        return FakeFrozenOpponentPolicy()

    monkeypatch.setattr(
        env_mod,
        "_build_source_state_frozen_lightzero_opponent_policy",
        fake_build,
    )
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 405,
            "source_max_steps": 16,
            "natural_bonus_spawn": False,
            "learner_seat_mode": "fixed_player_1",
            "opponent_policy_kind": (
                env_mod.OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT
            ),
            "opponent_checkpoint_path": "/tmp/frozen-seat0.pth.tar",
            "opponent_policy_seed": 499,
            "policy_action_repeat_min": 1,
            "policy_action_repeat_max": 1,
        }
    )
    env.reset(seed=405)

    timestep = env.step(LEFT_ACTION_ID)

    assert timestep.info["joint_action"] == {
        "player_0": RIGHT_ACTION_ID,
        "player_1": LEFT_ACTION_ID,
    }
    assert timestep.info["opponent_action_id"] == RIGHT_ACTION_ID
    assert calls[0]["decision_index"] == 0
    np.testing.assert_array_equal(calls[0]["opponent_mask"], np.array([[True, False]]))
    assert calls[0]["legal_action_mask"].shape == (1, 2, 3)
    assert calls[0]["observation"].shape == (1, 2, *STACKED_SOURCE_STATE_GRAY64_SHAPE)
    assert float(calls[0]["observation"][0, 0].max()) > 0.0
    assert float(calls[0]["observation"][0, 1].max()) > 0.0


def test_blank_canvas_noop_with_fixed_learner_seat_1_scrubs_player_0_not_player_1():
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": 407,
            "source_max_steps": 64,
            "natural_bonus_spawn": False,
            "learner_seat_mode": "fixed_player_1",
            "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
            "policy_action_repeat_min": 1,
            "policy_action_repeat_max": 1,
        }
    )
    env.reset(seed=407)
    state = env._env.state
    state["body_active"][0, 0] = True
    state["body_owner"][0, 0] = 0
    state["body_pos"][0, 0] = state["pos"][0, 1]
    state["body_radius"][0, 0] = 30.0
    state["body_count"][0, 0] = 1
    state["world_body_count"][0] = int(state["body_active"][0].sum())

    timestep = env.step(STRAIGHT_ACTION_ID)
    render_state = env._render_state_view()

    assert not timestep.done
    assert bool(state["present"][0, 0])
    assert bool(state["alive"][0, 0])
    assert bool(state["present"][0, 1])
    assert bool(state["alive"][0, 1])
    assert not _owner_active(state, "body", 0)
    assert not _owner_active(state, "visual_trail", 0)
    assert not bool(render_state["present"][0, 0])
    assert bool(render_state["present"][0, 1])
    assert timestep.info["blank_canvas_noop"] is True
    assert timestep.info["opponent_controlled_player_id"] == "player_0"
    assert timestep.info["reward_player_id"] == "player_1"


def test_random_learner_seat_mode_uses_both_seats_deterministically_with_dynamic_seed():
    cfg = {
        "seed": 409,
        "source_max_steps": 16,
        "natural_bonus_spawn": False,
        "dynamic_seed": True,
        "learner_seat_mode": "random_per_episode",
    }

    first = _reset_seat_sequence(cfg, reset_count=16)
    second = _reset_seat_sequence(cfg, reset_count=16)

    assert first == second
    assert {seat for seat, _seed in first} == {0, 1}


def _reset_seat_sequence(
    cfg: dict[str, object],
    *,
    reset_count: int,
) -> list[tuple[int, int]]:
    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(cfg)
    seats = []
    for _ in range(reset_count):
        env.reset()
        info = env.last_reset_info
        assert "ego_player_index" in info
        assert "opponent_player_index" in info
        assert "learner_seat_assignment_schema_id" in info
        ego_player_index = int(info["ego_player_index"])
        seats.append((ego_player_index, int(info["episode_seed"])))
        assert int(info["opponent_player_index"]) == 1 - ego_player_index
        assert info["learner_seat_assignment_schema_id"]
    return seats


def _owner_active(state: dict[str, np.ndarray], prefix: str, owner: int) -> bool:
    active = state[f"{prefix}_active"]
    owners = state[f"{prefix}_owner"]
    return bool((active & (owners == owner)).any())
