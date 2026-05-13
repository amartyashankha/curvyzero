"""Tiny two-seat physical-tick to LightZero ``GameSegment`` bridge.

This module is intentionally small: it keeps the hand-authored trace and the
per-seat trajectory projection explicit, then only imports LightZero inside the
native bridge helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np


DEFAULT_TO_PLAY = -1
DEFAULT_ACTION_SPACE_SIZE = 3
DEFAULT_SUPPORT_SCALE = 10


class NativeReplayBridgeUnavailable(RuntimeError):
    """Raised when the installed LightZero bridge surface is unavailable."""


@dataclass(frozen=True)
class TwoSeatPhysicalTickTrace:
    """A simultaneous two-seat trace, indexed by physical tick."""

    observations_by_seat: tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]
    joint_actions: tuple[tuple[int, int], ...]
    rewards_by_seat: tuple[tuple[float, ...], tuple[float, ...]]
    visit_distributions_by_seat: tuple[
        tuple[tuple[float, ...], ...],
        tuple[tuple[float, ...], ...],
    ]
    terminal: bool = True
    to_play: int = DEFAULT_TO_PLAY
    discount: float = 1.0


@dataclass(frozen=True)
class SeatGameSegmentSpec:
    """Native-compatible data for one seat-perspective trajectory."""

    seat_id: int
    observations: tuple[np.ndarray, ...]
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    visit_distributions: tuple[tuple[float, ...], ...]
    action_masks: tuple[tuple[int, ...], ...]
    to_play: tuple[int, ...]
    physical_ticks: tuple[int, ...]
    terminal: bool


@dataclass(frozen=True)
class NativeBridgeSegments:
    """The two projected specs and their native LightZero objects."""

    specs: tuple[SeatGameSegmentSpec, SeatGameSegmentSpec]
    game_segments: tuple[Any, Any]
    config: Any


@dataclass(frozen=True)
class NativeBufferPushResult:
    """Result of pushing bridge segments into a native replay buffer."""

    buffer: Any
    pushed_count: int
    transition_count: int | None


def tiny_three_tick_two_seat_trace() -> TwoSeatPhysicalTickTrace:
    """Return Kuhn's minimal three-tick parity trace."""

    observations_by_seat = (
        tuple(np.asarray([0.0, float(tick)], dtype=np.float32) for tick in range(4)),
        tuple(np.asarray([1.0, float(tick)], dtype=np.float32) for tick in range(4)),
    )
    return TwoSeatPhysicalTickTrace(
        observations_by_seat=observations_by_seat,
        joint_actions=((2, 0), (1, 2), (0, 1)),
        rewards_by_seat=((1.0, 2.0, 4.0), (1.0, 0.0, -2.0)),
        visit_distributions_by_seat=(
            ((0.10, 0.20, 0.70), (0.15, 0.70, 0.15), (0.80, 0.10, 0.10)),
            ((0.75, 0.15, 0.10), (0.10, 0.20, 0.70), (0.20, 0.65, 0.15)),
        ),
        terminal=True,
        to_play=DEFAULT_TO_PLAY,
        discount=1.0,
    )


def build_two_seat_game_segment_specs(
    trace: TwoSeatPhysicalTickTrace,
    *,
    action_space_size: int = DEFAULT_ACTION_SPACE_SIZE,
) -> tuple[SeatGameSegmentSpec, SeatGameSegmentSpec]:
    """Project one simultaneous physical-tick trace into two seat trajectories."""

    _validate_trace(trace, action_space_size=action_space_size)
    action_masks = tuple(
        tuple(1 for _ in range(action_space_size)) for _ in trace.joint_actions
    )
    physical_ticks = tuple(range(len(trace.joint_actions)))
    specs = []
    for seat_id in range(2):
        specs.append(
            SeatGameSegmentSpec(
                seat_id=seat_id,
                observations=trace.observations_by_seat[seat_id],
                actions=tuple(
                    int(joint_action[seat_id]) for joint_action in trace.joint_actions
                ),
                rewards=tuple(float(reward) for reward in trace.rewards_by_seat[seat_id]),
                visit_distributions=trace.visit_distributions_by_seat[seat_id],
                action_masks=action_masks,
                to_play=tuple(trace.to_play for _ in trace.joint_actions),
                physical_ticks=physical_ticks,
                terminal=bool(trace.terminal),
            )
        )
    return (specs[0], specs[1])


def expected_discounted_value_targets(
    spec: SeatGameSegmentSpec,
    *,
    discount: float = 1.0,
) -> tuple[float, ...]:
    """Return no-bootstrap discounted returns for every physical tick."""

    rewards = spec.rewards
    targets: list[float] = []
    for start in range(len(rewards)):
        total = 0.0
        scale = 1.0
        for reward in rewards[start:]:
            total += scale * reward
            scale *= discount
        targets.append(float(total))
    return tuple(targets)


def build_lightzero_muzero_bridge_config(
    *,
    action_space_size: int = DEFAULT_ACTION_SPACE_SIZE,
    game_segment_length: int = 3,
    num_unroll_steps: int = 2,
    td_steps: int = 3,
    discount_factor: float = 1.0,
    support_scale: int = DEFAULT_SUPPORT_SCALE,
) -> Any:
    """Build the minimal config fields used by ``GameSegment`` and buffer tests."""

    support_scale = int(support_scale)
    support_range = (-support_scale, support_scale + 1, 1)
    config = {
        "env_type": "not_board_games",
        "action_type": "fixed_action_space",
        "replay_buffer_size": 64,
        "batch_size": 2,
        "priority_prob_alpha": 0.6,
        "priority_prob_beta": 0.4,
        "use_priority": False,
        "reanalyze_ratio": 0.0,
        "reanalyze_outdated": False,
        "device": "cpu",
        "num_unroll_steps": int(num_unroll_steps),
        "td_steps": int(td_steps),
        "game_segment_length": int(game_segment_length),
        "discount_factor": float(discount_factor),
        "use_root_value": False,
        "mini_infer_size": 16,
        "mcts_ctree": False,
        "reanalyze_noise": False,
        "root_dirichlet_alpha": 0.3,
        "root_noise_weight": 0.25,
        "sample_type": "transition",
        "gray_scale": False,
        "transform2string": False,
        "sampled_algo": False,
        "gumbel_algo": False,
        "use_ture_chance_label_in_chance_encoder": False,
        "model": {
            "action_space_size": int(action_space_size),
            "continuous_action_space": False,
            "frame_stack_num": 1,
            "observation_shape": 2,
            "model_type": "mlp",
            "support_scale": support_scale,
            "value_support_range": support_range,
            "reward_support_range": support_range,
        },
    }
    try:
        from easydict import EasyDict
    except Exception:
        return _namespace(config)
    return EasyDict(config)


def build_native_lightzero_game_segments(
    trace: TwoSeatPhysicalTickTrace,
    *,
    action_space_size: int = DEFAULT_ACTION_SPACE_SIZE,
    game_segment_cls: type[Any] | None = None,
    config: Any | None = None,
) -> NativeBridgeSegments:
    """Build two actual LightZero ``GameSegment`` objects if LightZero exists."""

    if game_segment_cls is None:
        try:
            from lzero.mcts.buffer.game_segment import GameSegment
        except Exception as exc:
            raise NativeReplayBridgeUnavailable(
                f"LightZero GameSegment import failed: {exc}"
            ) from exc
        game_segment_cls = GameSegment

    specs = build_two_seat_game_segment_specs(
        trace,
        action_space_size=action_space_size,
    )
    resolved_config = config or build_lightzero_muzero_bridge_config(
        action_space_size=action_space_size,
        game_segment_length=len(trace.joint_actions),
        num_unroll_steps=max(0, len(trace.joint_actions) - 1),
        td_steps=len(trace.joint_actions),
        discount_factor=trace.discount,
    )
    action_space = _DiscreteActionSpace(action_space_size)
    game_segments = tuple(
        _build_native_game_segment(
            spec,
            game_segment_cls=game_segment_cls,
            action_space=action_space,
            config=resolved_config,
        )
        for spec in specs
    )
    return NativeBridgeSegments(
        specs=specs,
        game_segments=(game_segments[0], game_segments[1]),
        config=resolved_config,
    )


def push_native_segments_into_muzero_game_buffer(
    segments: Sequence[Any],
    *,
    config: Any,
    buffer_cls: type[Any] | None = None,
    buffer: Any | None = None,
) -> NativeBufferPushResult:
    """Push native segments through ``MuZeroGameBuffer`` when available."""

    if buffer is None:
        if buffer_cls is None:
            try:
                from lzero.mcts.buffer.game_buffer_muzero import MuZeroGameBuffer
            except Exception as exc:
                raise NativeReplayBridgeUnavailable(
                    f"LightZero MuZeroGameBuffer import failed: {exc}"
                ) from exc
            buffer_cls = MuZeroGameBuffer
        buffer = buffer_cls(config)

    if not hasattr(buffer, "push_game_segments"):
        raise NativeReplayBridgeUnavailable(
            "MuZeroGameBuffer does not expose push_game_segments"
        )

    meta = [
        {
            "done": True,
            "priorities": None,
            "unroll_plus_td_steps": (
                int(config.num_unroll_steps) + int(config.td_steps)
            ),
        }
        for _ in segments
    ]
    buffer.push_game_segments((list(segments), meta))
    transition_count = (
        int(buffer.get_num_of_transitions())
        if hasattr(buffer, "get_num_of_transitions")
        else None
    )
    return NativeBufferPushResult(
        buffer=buffer,
        pushed_count=len(segments),
        transition_count=transition_count,
    )


def _build_native_game_segment(
    spec: SeatGameSegmentSpec,
    *,
    game_segment_cls: type[Any],
    action_space: Any,
    config: Any,
) -> Any:
    segment = game_segment_cls(
        action_space,
        game_segment_length=len(spec.actions),
        config=config,
    )
    segment.reset([np.asarray(spec.observations[0], dtype=np.float32)])
    for index, action in enumerate(spec.actions):
        segment.store_search_stats(
            list(spec.visit_distributions[index]),
            root_value=0.0,
        )
        segment.append(
            int(action),
            np.asarray(spec.observations[index + 1], dtype=np.float32),
            float(spec.rewards[index]),
            action_mask=np.asarray(spec.action_masks[index], dtype=np.int8),
            to_play=int(spec.to_play[index]),
            timestep=int(spec.physical_ticks[index]),
        )
    if hasattr(segment, "game_segment_to_array"):
        segment.game_segment_to_array()
    return segment


def _validate_trace(
    trace: TwoSeatPhysicalTickTrace,
    *,
    action_space_size: int,
) -> None:
    if action_space_size < 1:
        raise ValueError("action_space_size must be >= 1")
    tick_count = len(trace.joint_actions)
    if tick_count < 1:
        raise ValueError("trace must contain at least one physical tick")
    for seat_id in range(2):
        if len(trace.observations_by_seat[seat_id]) != tick_count + 1:
            raise ValueError("each seat needs one initial obs plus one next obs per tick")
        if len(trace.rewards_by_seat[seat_id]) != tick_count:
            raise ValueError("each seat needs one reward per physical tick")
        if len(trace.visit_distributions_by_seat[seat_id]) != tick_count:
            raise ValueError("each seat needs one visit distribution per physical tick")
    for tick, joint_action in enumerate(trace.joint_actions):
        if len(joint_action) != 2:
            raise ValueError(f"joint action at tick {tick} must contain two seats")
        for action in joint_action:
            if int(action) < 0 or int(action) >= action_space_size:
                raise ValueError(f"action {action!r} is outside the action space")
        for seat_id in range(2):
            visits = trace.visit_distributions_by_seat[seat_id][tick]
            if len(visits) != action_space_size:
                raise ValueError("visit distribution length must match action space")
            if not np.isclose(float(sum(visits)), 1.0):
                raise ValueError("visit distributions must sum to 1.0")


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    return value


@dataclass(frozen=True)
class _DiscreteActionSpace:
    n: int
