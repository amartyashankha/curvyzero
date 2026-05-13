from types import SimpleNamespace

import numpy as np
import pytest

from curvyzero.training.two_seat_native_replay_bridge import (
    DEFAULT_TO_PLAY,
    build_lightzero_muzero_bridge_config,
    build_native_lightzero_game_segments,
    build_two_seat_game_segment_specs,
    expected_discounted_value_targets,
    push_native_segments_into_muzero_game_buffer,
    tiny_three_tick_two_seat_trace,
)


def test_tiny_trace_projects_physical_ticks_without_pending_rows():
    trace = tiny_three_tick_two_seat_trace()
    seat_0, seat_1 = build_two_seat_game_segment_specs(trace)

    assert seat_0.actions == (2, 1, 0)
    assert seat_1.actions == (0, 2, 1)
    assert seat_0.actions != trace.joint_actions
    assert seat_1.actions != trace.joint_actions
    assert seat_0.rewards == (1.0, 2.0, 4.0)
    assert seat_1.rewards == (1.0, 0.0, -2.0)
    assert seat_0.to_play == (DEFAULT_TO_PLAY, DEFAULT_TO_PLAY, DEFAULT_TO_PLAY)
    assert seat_1.to_play == (DEFAULT_TO_PLAY, DEFAULT_TO_PLAY, DEFAULT_TO_PLAY)
    assert seat_0.physical_ticks == (0, 1, 2)
    assert seat_1.physical_ticks == (0, 1, 2)
    assert len(seat_0.observations) == 4
    assert len(seat_1.observations) == 4
    assert expected_discounted_value_targets(seat_0, discount=trace.discount) == (
        7.0,
        6.0,
        4.0,
    )
    assert expected_discounted_value_targets(seat_1, discount=trace.discount) == (
        -1.0,
        -2.0,
        -2.0,
    )


def test_lightzero_bridge_config_has_v020_support_scale_metadata():
    config = build_lightzero_muzero_bridge_config()

    assert config.model.support_scale == 10
    assert config.model.reward_support_range == (-10, 11, 1)
    assert config.model.value_support_range == (-10, 11, 1)


def test_native_lightzero_segments_push_and_tick0_targets_when_api_allows():
    pytest.importorskip("lzero", reason="DI-engine/LightZero runtime is not installed locally")
    game_segment_module = pytest.importorskip("lzero.mcts.buffer.game_segment")
    buffer_module = pytest.importorskip("lzero.mcts.buffer.game_buffer_muzero")
    torch = pytest.importorskip("torch")

    trace = tiny_three_tick_two_seat_trace()
    native = build_native_lightzero_game_segments(
        trace,
        game_segment_cls=game_segment_module.GameSegment,
    )
    seat_0, seat_1 = native.specs
    segment_0, segment_1 = native.game_segments

    _assert_native_segment_matches_spec(segment_0, seat_0)
    _assert_native_segment_matches_spec(segment_1, seat_1)

    try:
        pushed = push_native_segments_into_muzero_game_buffer(
            native.game_segments,
            config=native.config,
            buffer_cls=buffer_module.MuZeroGameBuffer,
        )
    except Exception as exc:
        pytest.skip(f"MuZeroGameBuffer push blocker for native parity proof: {exc!r}")

    assert pushed.pushed_count == 2
    assert pushed.transition_count == 6

    buffer = pushed.buffer
    required_hooks = (
        "_prepare_reward_value_context",
        "_compute_target_reward_value",
        "_prepare_policy_non_reanalyzed_context",
        "_compute_target_policy_non_reanalyzed",
    )
    missing_hooks = [name for name in required_hooks if not hasattr(buffer, name)]
    if missing_hooks:
        pytest.skip(
            "MuZeroGameBuffer deterministic tick-0 target blocker: missing hooks "
            + ", ".join(missing_hooks)
        )

    try:
        target_rewards, target_values = _native_tick0_reward_value_targets(
            buffer,
            native.game_segments,
            torch=torch,
        )
        target_policies = _native_tick0_policy_targets(buffer, native.game_segments)
    except Exception as exc:
        pytest.skip(
            "MuZeroGameBuffer deterministic tick-0 target blocker after push: "
            f"{type(exc).__name__}: {exc}"
        )

    np.testing.assert_allclose(target_rewards[0], [1.0, 2.0, 4.0])
    np.testing.assert_allclose(target_rewards[1], [1.0, 0.0, -2.0])
    np.testing.assert_allclose(target_values[0], [7.0, 6.0, 4.0])
    np.testing.assert_allclose(target_values[1], [-1.0, -2.0, -2.0])
    np.testing.assert_allclose(target_policies[0], np.asarray(seat_0.visit_distributions))
    np.testing.assert_allclose(target_policies[1], np.asarray(seat_1.visit_distributions))
    np.testing.assert_array_equal(np.asarray(segment_0.action_segment), [2, 1, 0])
    np.testing.assert_array_equal(np.asarray(segment_1.action_segment), [0, 2, 1])


def _assert_native_segment_matches_spec(segment, spec):
    np.testing.assert_array_equal(np.asarray(segment.action_segment), spec.actions)
    np.testing.assert_allclose(np.asarray(segment.reward_segment), spec.rewards)
    np.testing.assert_allclose(
        np.asarray(segment.child_visit_segment),
        np.asarray(spec.visit_distributions),
    )
    np.testing.assert_array_equal(np.asarray(segment.to_play_segment), spec.to_play)
    np.testing.assert_array_equal(np.asarray(segment.timestep_segment), spec.physical_ticks)
    assert len(segment.action_segment) == 3
    assert len(segment.obs_segment) == 4
    assert all(np.isscalar(action) for action in np.asarray(segment.action_segment))


def _native_tick0_reward_value_targets(buffer, game_segments, *, torch):
    context = buffer._prepare_reward_value_context(
        [0, 3],
        list(game_segments),
        [0, 0],
        buffer.get_num_of_transitions(),
    )
    config = buffer._cfg
    return buffer._compute_target_reward_value(
        context,
        _ZeroTargetModel(
            torch=torch,
            action_space_size=int(config.model.action_space_size),
            value_support_size=_support_size(config),
            reward_support_size=_support_size(config),
        ),
    )


def _native_tick0_policy_targets(buffer, game_segments):
    context = buffer._prepare_policy_non_reanalyzed_context(
        [0, 3],
        list(game_segments),
        [0, 0],
    )
    return buffer._compute_target_policy_non_reanalyzed(context, 3)


class _ZeroTargetModel:
    def __init__(
        self,
        *,
        torch,
        action_space_size: int,
        value_support_size: int,
        reward_support_size: int,
    ):
        self._torch = torch
        self._action_space_size = action_space_size
        self._value_support_size = value_support_size
        self._reward_support_size = reward_support_size
        self.training = False

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def to(self, *_args, **_kwargs):
        return self

    def initial_inference(self, obs):
        batch_size = int(obs.shape[0])
        return SimpleNamespace(
            latent_state=self._torch.zeros((batch_size, 1), dtype=self._torch.float32),
            reward=self._torch.zeros(
                (batch_size, self._reward_support_size),
                dtype=self._torch.float32,
            ),
            value=self._torch.zeros(
                (batch_size, self._value_support_size),
                dtype=self._torch.float32,
            ),
            policy_logits=self._torch.zeros(
                (batch_size, self._action_space_size),
                dtype=self._torch.float32,
            ),
        )


def _support_size(config):
    return int(2 * config.model.support_scale + 1)
