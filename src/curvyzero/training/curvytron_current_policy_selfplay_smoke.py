"""Bounded current-policy joint-action smoke for CurvyTron.

This is the first small artifact for the real two-seat lane. It does not call
LightZero's collector or trainer. It proves the important shape and control
boundary locally:

* observations are rendered as player-perspective ``float32[B,P,4,64,64]``;
* one shared policy object chooses actions for both players;
* the public two-player env is stepped with external ``joint_action[B,P]``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from dataclasses import field
import hashlib
import json
from typing import Any

import numpy as np

from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_GRAY64_NORMALIZED_DTYPE,
)
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_GRAY64_NORMALIZED_VALUE_RANGE,
)
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID
from curvyzero.env.vector_visual_observation import SOURCE_STATE_GRAY64_SHAPE
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
)
from curvyzero.env.vector_visual_observation import SourceStateBrowserLineTrailLayerCache
from curvyzero.env.vector_visual_observation import SourceStateCanvasGray64DirtyRenderCache
from curvyzero.env.vector_visual_observation import SourceStateGray64DownsampleScratch
from curvyzero.env.vector_visual_observation import TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
from curvyzero.env.vector_visual_observation import TRAIL_RENDER_MODE_BROWSER_LINES
from curvyzero.env.vector_visual_observation import TRAIL_RENDER_MODE_DEFAULT
from curvyzero.env.vector_visual_observation import TRAIL_RENDER_MODE_ORDER
from curvyzero.env.vector_visual_observation import normalize_source_state_gray64
from curvyzero.env.vector_visual_observation import render_source_state_canvas_gray64
from curvyzero.env.vector_visual_observation import (
    render_source_state_canvas_gray64_player_perspectives,
)
from curvyzero.env.vector_visual_observation import (
    render_source_state_gray64_fast_player_perspectives,
)
from curvyzero.training.policy_row_mapping import build_policy_row_mapping
from curvyzero.training.policy_row_mapping import policy_rows_to_joint_action


CURRENT_POLICY_SELFPLAY_SMOKE_SCHEMA_ID = (
    "curvyzero_current_policy_joint_action_smoke/v0"
)
CURRENT_POLICY_TWO_SEAT_TRAIN_SMOKE_SCHEMA_ID = (
    "curvyzero_current_policy_two_seat_train_smoke/v0"
)
SHARED_SEEDED_CURRENT_POLICY_ID = "curvyzero_shared_seeded_current_policy_smoke"
SHARED_SEEDED_CURRENT_POLICY_VERSION = "v0.2026-05-10"
SHARED_LINEAR_SURVIVAL_CURRENT_POLICY_ID = (
    "curvyzero_shared_linear_survival_current_policy_smoke"
)
SHARED_LINEAR_SURVIVAL_CURRENT_POLICY_VERSION = "v0.2026-05-10"
STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID = (
    "curvyzero_source_state_gray64_stack4_player_perspective/v1"
)
STACKED_SOURCE_STATE_GRAY64_SHAPE = (4, 64, 64)
PLAYER_PERSPECTIVE_SCHEMA_ID = "curvyzero_player_perspective_source_state_gray64/v0"
STACK_RENDER_MODE_FAST_GRAY64_DIRECT = "fast_gray64_direct"
STACK_RENDER_MODE_ORDER = (*TRAIL_RENDER_MODE_ORDER, STACK_RENDER_MODE_FAST_GRAY64_DIRECT)
STACK_RENDER_MODE_DEFAULT = TRAIL_RENDER_MODE_DEFAULT
SOURCE_BODY_VALUE_BASE = 96
SOURCE_BODY_VALUE_STEP = 32
SOURCE_BODY_VALUE_MAX = 192
SOURCE_HEAD_VALUE_BASE = 224
SOURCE_HEAD_VALUE_STEP = 8
SOURCE_HEAD_VALUE_MAX = 255
SELF_BODY_VALUE = 96
OTHER_BODY_VALUE = 128
SELF_HEAD_VALUE = 224
OTHER_HEAD_VALUE = 232
PERSPECTIVE_SELF_RGB = (SELF_BODY_VALUE, SELF_BODY_VALUE, SELF_BODY_VALUE)
PERSPECTIVE_OTHER_RGB = (OTHER_BODY_VALUE, OTHER_BODY_VALUE, OTHER_BODY_VALUE)
ACTION_COUNT = 3
NOOP_ACTION_ID = 1
LINEAR_FEATURE_DIM = 7


def _is_dirty_render_stat_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


def _merge_dirty_render_stats(target: dict[str, Any], stats: dict[str, Any]) -> None:
    for key, value in stats.items():
        name = str(key)
        if isinstance(value, dict):
            nested = target.setdefault(name, {})
            if not isinstance(nested, dict):
                continue
            for nested_key, nested_value in value.items():
                if _is_dirty_render_stat_number(nested_value):
                    nested_name = str(nested_key)
                    nested[nested_name] = nested.get(nested_name, 0) + nested_value
            continue
        if _is_dirty_render_stat_number(value):
            target[name] = target.get(name, 0) + value


@dataclass(frozen=True, slots=True)
class CurrentPolicyActionBatch:
    """Actions selected by one shared current-policy object."""

    action: np.ndarray
    action_weights: np.ndarray
    root_value: np.ndarray


@dataclass(slots=True)
class SharedSeededCurrentPolicy:
    """Tiny shared policy used only to prove both seats use one policy object.

    This policy is not learned and does not run MCTS. It intentionally produces
    deterministic legal actions with a policy-shaped output so the next worker
    can replace the body with a live LightZero eval/search call.
    """

    seed: int = 0
    policy_id: str = SHARED_SEEDED_CURRENT_POLICY_ID
    policy_version: str = SHARED_SEEDED_CURRENT_POLICY_VERSION

    def select_actions(
        self,
        observation: np.ndarray,
        legal_action_mask: np.ndarray,
        *,
        env_row_id: np.ndarray,
        player_id: np.ndarray,
        decision_index: int,
    ) -> CurrentPolicyActionBatch:
        obs = np.asarray(observation, dtype=np.float32)
        legal = np.asarray(legal_action_mask, dtype=bool)
        env_rows = np.asarray(env_row_id, dtype=np.int64)
        players = np.asarray(player_id, dtype=np.int64)
        if obs.ndim != 4 or obs.shape[1:] != STACKED_SOURCE_STATE_GRAY64_SHAPE:
            raise ValueError(
                "policy observation must have shape [R,4,64,64]; "
                f"got {obs.shape!r}"
            )
        if legal.shape != (obs.shape[0], ACTION_COUNT):
            raise ValueError(
                "legal_action_mask must have shape [R,3]; "
                f"got {legal.shape!r}"
            )
        if env_rows.shape != (obs.shape[0],) or players.shape != (obs.shape[0],):
            raise ValueError("env_row_id and player_id must have shape [R]")

        actions = np.zeros(obs.shape[0], dtype=np.int16)
        weights = np.zeros((obs.shape[0], ACTION_COUNT), dtype=np.float32)
        root_value = np.zeros(obs.shape[0], dtype=np.float32)
        for row in range(obs.shape[0]):
            legal_ids = np.flatnonzero(legal[row])
            if legal_ids.size == 0:
                raise ValueError(f"policy row {row} has no legal action")
            choice_offset = (
                int(self.seed)
                + int(decision_index)
                + int(env_rows[row]) * 2
                + int(players[row])
            ) % int(legal_ids.size)
            action = int(legal_ids[choice_offset])
            actions[row] = action
            weights[row, legal_ids] = np.float32(1.0 / legal_ids.size)
            root_value[row] = np.float32(0.0)
        return CurrentPolicyActionBatch(
            action=actions,
            action_weights=weights,
            root_value=root_value,
        )


@dataclass(slots=True)
class SharedLinearSurvivalCurrentPolicy:
    """Tiny mutable shared policy for bounded two-seat collect/replay/learn smoke.

    This is intentionally not a LightZero policy and not MCTS. It provides the
    smallest local current-policy contract: the same mutable object selects for
    both player seats, replay rows are sampled from those decisions, and a
    learner update changes the same object's weights.
    """

    seed: int = 0
    learning_rate: float = 0.05
    discount: float = 0.997
    policy_id: str = SHARED_LINEAR_SURVIVAL_CURRENT_POLICY_ID
    policy_version: str = SHARED_LINEAR_SURVIVAL_CURRENT_POLICY_VERSION
    action_weights: np.ndarray = field(init=False, repr=False)
    value_weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(int(self.seed))
        self.action_weights = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(ACTION_COUNT, LINEAR_FEATURE_DIM),
        ).astype(np.float32)
        self.value_weights = np.zeros(LINEAR_FEATURE_DIM, dtype=np.float32)

    def select_actions(
        self,
        observation: np.ndarray,
        legal_action_mask: np.ndarray,
        *,
        env_row_id: np.ndarray,
        player_id: np.ndarray,
        decision_index: int,
    ) -> CurrentPolicyActionBatch:
        del decision_index
        obs = np.asarray(observation, dtype=np.float32)
        legal = np.asarray(legal_action_mask, dtype=bool)
        env_rows = np.asarray(env_row_id, dtype=np.int64)
        players = np.asarray(player_id, dtype=np.int64)
        if obs.ndim != 4 or obs.shape[1:] != STACKED_SOURCE_STATE_GRAY64_SHAPE:
            raise ValueError(
                "policy observation must have shape [R,4,64,64]; "
                f"got {obs.shape!r}"
            )
        if legal.shape != (obs.shape[0], ACTION_COUNT):
            raise ValueError(
                "legal_action_mask must have shape [R,3]; "
                f"got {legal.shape!r}"
            )
        features = _linear_policy_features(obs, env_rows, players)
        logits = features @ self.action_weights.T
        probabilities = _masked_softmax(logits, legal)
        actions = np.zeros(obs.shape[0], dtype=np.int16)
        for row in range(obs.shape[0]):
            legal_ids = np.flatnonzero(legal[row])
            if legal_ids.size == 0:
                raise ValueError(f"policy row {row} has no legal action")
            # Deterministic current-policy action selection; no exploration
            # noise is hidden in the env.
            actions[row] = int(legal_ids[np.argmax(probabilities[row, legal_ids])])
        root_value = (features @ self.value_weights).astype(np.float32)
        return CurrentPolicyActionBatch(
            action=actions,
            action_weights=probabilities.astype(np.float32, copy=True),
            root_value=root_value,
        )

    def learn_from_replay(
        self,
        replay_rows: list[dict[str, Any]],
        *,
        updates: int = 1,
    ) -> dict[str, Any]:
        """Run a tiny policy/value learner update over collected two-seat rows."""

        if updates < 1:
            raise ValueError("updates must be >= 1")
        if not replay_rows:
            return {"status": "skipped", "reason": "empty replay"}

        observations = np.stack(
            [np.asarray(row["observation"], dtype=np.float32) for row in replay_rows],
            axis=0,
        )
        legal = np.stack(
            [np.asarray(row["legal_action_mask"], dtype=bool) for row in replay_rows],
            axis=0,
        )
        actions = np.asarray([int(row["action"]) for row in replay_rows], dtype=np.int64)
        env_rows = np.asarray([int(row["env_row_id"]) for row in replay_rows], dtype=np.int64)
        players = np.asarray([int(row["player_id"]) for row in replay_rows], dtype=np.int64)
        returns = _discounted_survival_returns(replay_rows, discount=self.discount)
        features = _linear_policy_features(observations, env_rows, players)

        before_hash = self.model_hash()
        before_loss = _linear_policy_loss(
            features,
            legal,
            actions,
            returns,
            self.action_weights,
            self.value_weights,
        )
        for _ in range(int(updates)):
            logits = features @ self.action_weights.T
            probabilities = _masked_softmax(logits, legal)
            values = features @ self.value_weights
            advantages = returns - values

            one_hot = np.zeros_like(probabilities, dtype=np.float32)
            one_hot[np.arange(actions.shape[0]), actions] = 1.0
            policy_grad = ((one_hot - probabilities) * advantages[:, None]).T @ features
            policy_grad /= np.float32(max(1, actions.shape[0]))
            value_grad = (advantages[:, None] * features).mean(axis=0)
            self.action_weights += np.float32(self.learning_rate) * policy_grad.astype(np.float32)
            self.value_weights += np.float32(self.learning_rate) * value_grad.astype(np.float32)

        after_hash = self.model_hash()
        after_loss = _linear_policy_loss(
            features,
            legal,
            actions,
            returns,
            self.action_weights,
            self.value_weights,
        )
        return {
            "status": "updated",
            "updates": int(updates),
            "api": "SharedLinearSurvivalCurrentPolicy.learn_from_replay",
            "search_kind": "masked_softmax_policy_head_no_mcts",
            "optimizer": "numpy_policy_gradient_value_fit",
            "reward_target": "discounted_steps_survived",
            "batch_size": int(actions.shape[0]),
            "model_hash_before": before_hash,
            "model_hash_after": after_hash,
            "model_parameters_changed": before_hash != after_hash,
            "loss_before": before_loss,
            "loss_after": after_loss,
        }

    def model_hash(self) -> str:
        digest = hashlib.sha256()
        for name, value in (
            ("action_weights", self.action_weights),
            ("value_weights", self.value_weights),
        ):
            array = np.asarray(value).astype(np.float32, copy=False)
            digest.update(name.encode("utf-8"))
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(array.tobytes())
        return digest.hexdigest()[:16]


class SourceStateGray64Stack4:
    """Render vector env rows and keep a player-perspective FIFO stack per slot."""

    def __init__(
        self,
        *,
        batch_size: int,
        player_count: int,
        trail_render_mode: str = STACK_RENDER_MODE_DEFAULT,
    ) -> None:
        self.batch_size = int(batch_size)
        self.player_count = int(player_count)
        self.trail_render_mode = validate_stack_trail_render_mode(trail_render_mode)
        self.stack = np.zeros(
            (self.batch_size, self.player_count, *STACKED_SOURCE_STATE_GRAY64_SHAPE),
            dtype=np.float32,
        )
        self._raw = np.zeros(SOURCE_STATE_GRAY64_SHAPE, dtype=np.uint8)
        self._rgb = np.zeros(
            (
                SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
                SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
                3,
            ),
            dtype=np.uint8,
        )
        self._rgb_base = np.zeros_like(self._rgb)
        self._raw_perspectives = np.zeros(
            (self.player_count, *SOURCE_STATE_GRAY64_SHAPE),
            dtype=np.uint8,
        )
        self._normalized = np.zeros(SOURCE_STATE_GRAY64_SHAPE, dtype=np.float32)
        self._downsample_scratch = SourceStateGray64DownsampleScratch(
            SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        )
        self._trail_layer_caches = [
            SourceStateBrowserLineTrailLayerCache(min_active_slots=1)
            for _ in range(self.batch_size)
        ]
        self._dirty_render_caches = [
            SourceStateCanvasGray64DirtyRenderCache(player_count=self.player_count)
            for _ in range(self.batch_size)
        ]

    def render_metadata(self) -> dict[str, Any]:
        return source_state_gray64_stack4_render_metadata(self.trail_render_mode)

    def dirty_render_stats(self) -> dict[str, Any]:
        totals: dict[str, Any] = {}
        for cache in self._dirty_render_caches:
            _merge_dirty_render_stats(totals, cache.stats.as_dict())
        hits = int(totals.get("hits", 0))
        attempts = int(totals.get("attempts", 0))
        return {
            "enabled": self.player_count == 2
            and self.trail_render_mode == TRAIL_RENDER_MODE_BROWSER_LINES,
            "rows": self.batch_size,
            "attempts": attempts,
            "hits": hits,
            "cold_starts": int(totals.get("cold_starts", 0)),
            "fallbacks": int(totals.get("fallbacks", 0)),
            "dirty_blocks_total": int(totals.get("dirty_blocks_total", 0)),
            "hit_rate": (float(hits) / float(attempts)) if attempts else None,
            "dirty_blocks_per_hit": (
                float(totals.get("dirty_blocks_total", 0)) / float(hits)
                if hits
                else None
            ),
        }

    def update(self, env: VectorMultiplayerEnv, *, copy: bool = True) -> np.ndarray:
        if env.batch_size != self.batch_size or env.player_count != self.player_count:
            raise ValueError("env shape changed after stack creation")
        for env_row in range(self.batch_size):
            self.stack[env_row, :, :-1] = self.stack[env_row, :, 1:]
            self._render_row_into_last_frame(env, env_row)
        return self.stack.copy() if copy else self.stack

    def reset_rows(
        self,
        env: VectorMultiplayerEnv,
        row_mask: np.ndarray,
        *,
        copy: bool = True,
    ) -> np.ndarray:
        if env.batch_size != self.batch_size or env.player_count != self.player_count:
            raise ValueError("env shape changed after stack creation")
        mask = np.asarray(row_mask, dtype=bool)
        if mask.shape != (self.batch_size,):
            raise ValueError("row_mask must have shape [B]")
        for env_row in np.flatnonzero(mask):
            row = int(env_row)
            self.stack[row] = 0.0
            self._trail_layer_caches[row] = SourceStateBrowserLineTrailLayerCache(
                min_active_slots=1
            )
            self._dirty_render_caches[row] = SourceStateCanvasGray64DirtyRenderCache(
                player_count=self.player_count
            )
            self._render_row_into_last_frame(env, row)
        return self.stack.copy() if copy else self.stack

    def _render_row_into_last_frame(
        self,
        env: VectorMultiplayerEnv,
        env_row: int,
    ) -> None:
        if self.trail_render_mode == STACK_RENDER_MODE_FAST_GRAY64_DIRECT:
            raw_frames = render_source_state_gray64_fast_player_perspectives(
                env.state,
                row=env_row,
                player_count=self.player_count,
                out=self._raw_perspectives,
            )
            for player in range(self.player_count):
                frame = normalize_source_state_gray64(
                    raw_frames[player],
                    out=self._normalized,
                )
                self.stack[env_row, player, -1] = frame[0]
            return
        if self.player_count == 2:
            raw_frames = render_source_state_canvas_gray64_player_perspectives(
                env.state,
                row=env_row,
                out=self._raw_perspectives,
                rgb_base_out=self._rgb_base,
                rgb_work_out=self._rgb,
                trail_cache=self._trail_layer_caches[env_row],
                dirty_render_cache=self._dirty_render_caches[env_row],
                downsample_scratch=self._downsample_scratch,
                player_rgbs=[
                    player_perspective_rgb_palette(
                        env.state,
                        row=env_row,
                        controlled_player=player,
                        player_count=self.player_count,
                    )
                    for player in range(self.player_count)
                ],
                trail_render_mode=self.trail_render_mode,
            )
            for player in range(self.player_count):
                frame = normalize_source_state_gray64(
                    raw_frames[player],
                    out=self._normalized,
                )
                self.stack[env_row, player, -1] = frame[0]
            return
        for player in range(self.player_count):
            raw = render_source_state_canvas_gray64(
                env.state,
                row=env_row,
                out=self._raw,
                rgb_out=self._rgb,
                player_rgb=player_perspective_rgb_palette(
                    env.state,
                    row=env_row,
                    controlled_player=player,
                    player_count=self.player_count,
                ),
                trail_render_mode=self.trail_render_mode,
                downsample_scratch=self._downsample_scratch,
            )
            frame = normalize_source_state_gray64(
                raw,
                out=self._normalized,
            )
            self.stack[env_row, player, -1] = frame[0]


def validate_stack_trail_render_mode(value: str) -> str:
    mode = str(value)
    if mode not in STACK_RENDER_MODE_ORDER:
        supported = ", ".join(STACK_RENDER_MODE_ORDER)
        raise ValueError(f"trail_render_mode must be one of [{supported}], got {value!r}")
    return mode


def source_state_gray64_stack4_render_metadata(
    trail_render_mode: str = STACK_RENDER_MODE_DEFAULT,
) -> dict[str, Any]:
    mode = validate_stack_trail_render_mode(trail_render_mode)
    if mode == STACK_RENDER_MODE_FAST_GRAY64_DIRECT:
        return {
            "trail_render_mode": mode,
            "default_trail_render_mode": STACK_RENDER_MODE_DEFAULT,
            "supported_trail_render_modes": list(STACK_RENDER_MODE_ORDER),
            "single_frame_render_api": (
                "render_source_state_gray64_fast_player_perspectives"
            ),
            "render_pipeline": "source_state_direct_gray64_visual_trail_circles_simple_bonus_symbols",
            "rgb_source_frame_size": None,
            "downsample_target_frame_size": 64,
            "rgb_to_gray64": False,
            "gray_conversion": "BT.601 luma values from the same RGB palette",
            "trail_renderer_kind": "circle_per_visual_trail_point",
            "trail_renderer_is_approximation": True,
            "bonus_render_mode": "simple_symbols",
            "bonus_renderer_kind": "simple_symbol_masks",
            "bonus_renderer_is_approximation": True,
            "player_perspective_palette": {
                "self_rgb": [SELF_BODY_VALUE] * 3,
                "other_rgb": [OTHER_BODY_VALUE] * 3,
                "semantics": (
                    "direct gray64 maps controlled-player trail/head luma to self "
                    "and all other visible players to other"
                ),
                "policy_reason": (
                    "single-channel model input still needs seat-relative "
                    "self/other contrast"
                ),
            },
        }
    return {
        "trail_render_mode": mode,
        "default_trail_render_mode": STACK_RENDER_MODE_DEFAULT,
        "supported_trail_render_modes": list(STACK_RENDER_MODE_ORDER),
        "single_frame_render_api": "render_source_state_canvas_gray64",
        "two_seat_optimized_render_api": (
            "render_source_state_canvas_gray64_player_perspectives"
        ),
        "two_seat_optimized_render_is_equivalence_cache": True,
        "render_pipeline": "source_state_rgb_canvas_like_raw_canvas_to_gray64",
        "rgb_source_frame_size": SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        "downsample_target_frame_size": 64,
        "rgb_to_gray64": True,
        "trail_renderer_kind": (
            "connected_rounded_lines"
            if mode == TRAIL_RENDER_MODE_BROWSER_LINES
            else "circle_per_body"
        ),
        "trail_renderer_is_approximation": (
            mode == TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
        ),
        "bonus_renderer_kind": "browser_sprites",
        "bonus_renderer_is_approximation": False,
        "player_perspective_palette": {
            "semantics": (
                "controlled player color index is rendered with self grayscale RGB; "
                "other visible player color indices are rendered with other grayscale RGB"
            ),
            "self_rgb": list(PERSPECTIVE_SELF_RGB),
            "other_rgb": list(PERSPECTIVE_OTHER_RGB),
            "policy_reason": (
                "single-channel model input still needs seat-relative self/other "
                "contrast after the RGB browser-style render path"
            ),
        },
    }


def player_perspective_rgb_palette(
    state: dict[str, np.ndarray] | Any,
    *,
    row: int,
    controlled_player: int,
    player_count: int,
) -> tuple[tuple[int, int, int], ...]:
    player = int(controlled_player)
    total_players = int(player_count)
    if player < 0 or player >= total_players:
        raise ValueError("controlled_player must be in [0, player_count)")

    color_indices = np.arange(total_players, dtype=np.int64)
    if "avatar_color" in state:
        avatar_color = np.asarray(state["avatar_color"])
        if avatar_color.ndim >= 2:
            color_indices = np.asarray(
                avatar_color[int(row), :total_players],
                dtype=np.int64,
            )
    if bool((color_indices < 0).any()):
        raise ValueError("avatar_color indices must be non-negative")
    max_color_index = int(color_indices.max()) if color_indices.size else total_players - 1
    palette = [
        PERSPECTIVE_OTHER_RGB
        for _ in range(max(total_players, max_color_index + 1))
    ]
    palette[int(color_indices[player])] = PERSPECTIVE_SELF_RGB
    return tuple(palette)


def normalize_source_state_player_perspective(
    frame: np.ndarray,
    *,
    controlled_player: int,
    player_count: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Remap owner-coded raw gray64 pixels into self/other perspective."""

    source = np.asarray(frame)
    if source.shape != SOURCE_STATE_GRAY64_SHAPE:
        raise ValueError(
            f"frame must have shape {SOURCE_STATE_GRAY64_SHAPE}, got {source.shape!r}"
        )
    if source.dtype != np.uint8:
        raise ValueError(f"frame dtype must be uint8, got {source.dtype}")
    player = int(controlled_player)
    total_players = int(player_count)
    if player < 0 or player >= total_players:
        raise ValueError("controlled_player must be in [0, player_count)")

    perspective = np.empty_like(source) if out is None else np.asarray(out)
    if perspective.shape != SOURCE_STATE_GRAY64_SHAPE:
        raise ValueError(
            f"out must have shape {SOURCE_STATE_GRAY64_SHAPE}, got {perspective.shape!r}"
        )
    if perspective.dtype != np.uint8:
        raise ValueError(f"out dtype must be uint8, got {perspective.dtype}")

    np.copyto(perspective, source)
    source_canvas = source[0]
    perspective_canvas = perspective[0]
    for source_player in range(total_players):
        body_value = _source_body_value(source_player)
        head_value = _source_head_value(source_player)
        target_body = SELF_BODY_VALUE if source_player == player else OTHER_BODY_VALUE
        target_head = SELF_HEAD_VALUE if source_player == player else OTHER_HEAD_VALUE
        perspective_canvas[source_canvas == body_value] = np.uint8(target_body)
        perspective_canvas[source_canvas == head_value] = np.uint8(target_head)
    return perspective


def player_perspective_diagnostics(observation: np.ndarray) -> dict[str, Any]:
    """Summarize whether player slots are visually distinguishable."""

    obs = np.asarray(observation, dtype=np.float32)
    result: dict[str, Any] = {
        "schema_id": PLAYER_PERSPECTIVE_SCHEMA_ID,
        "source_values": player_perspective_value_map(),
        "shape_preserved": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
    }
    if obs.ndim != 5 or obs.shape[2:] != STACKED_SOURCE_STATE_GRAY64_SHAPE:
        result.update(
            {
                "ok": False,
                "status": "invalid_shape",
                "observation_shape": list(obs.shape),
            }
        )
        return result

    adjacent_delta_min = None
    if obs.shape[1] >= 2 and obs.shape[0] > 0:
        deltas = [
            np.max(np.abs(obs[:, player] - obs[:, player + 1]))
            for player in range(obs.shape[1] - 1)
        ]
        adjacent_delta_min = float(np.min(np.asarray(deltas, dtype=np.float32)))
    result.update(
        {
            "ok": bool(adjacent_delta_min is None or adjacent_delta_min > 0.0),
            "status": "ok",
            "observation_shape": list(obs.shape),
            "player_count": int(obs.shape[1]),
            "adjacent_player_pair_min_max_abs_delta": adjacent_delta_min,
        }
    )
    return result


def player_perspective_value_map() -> dict[str, Any]:
    return {
        "verified_source": "vector_visual_observation._body_value/_head_value",
        "source_body": {
            "formula": "min(192, 96 + player_id * 32)",
            "player_0": _source_body_value(0),
            "player_1": _source_body_value(1),
        },
        "source_head": {
            "formula": "min(255, 224 + player_id * 8)",
            "player_0": _source_head_value(0),
            "player_1": _source_head_value(1),
        },
        "perspective_body": {"self": SELF_BODY_VALUE, "other": OTHER_BODY_VALUE},
        "perspective_head": {"self": SELF_HEAD_VALUE, "other": OTHER_HEAD_VALUE},
    }


def _source_body_value(player: int) -> int:
    return int(
        min(
            SOURCE_BODY_VALUE_MAX,
            SOURCE_BODY_VALUE_BASE + int(player) * SOURCE_BODY_VALUE_STEP,
        )
    )


def _source_head_value(player: int) -> int:
    return int(
        min(
            SOURCE_HEAD_VALUE_MAX,
            SOURCE_HEAD_VALUE_BASE + int(player) * SOURCE_HEAD_VALUE_STEP,
        )
    )


def probe_curvytron_two_seat_player_perspective(
    *,
    seed: int = 0,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Cheap local probe proving player slots no longer share identical frames."""

    env = VectorMultiplayerEnv(
        batch_size=batch_size,
        player_count=2,
        seed=seed,
        max_ticks=4,
    )
    visual_stack = SourceStateGray64Stack4(batch_size=batch_size, player_count=2)
    env.reset(seed=seed)
    reset_observation = visual_stack.update(env)
    joint_action = np.full((batch_size, 2), NOOP_ACTION_ID, dtype=np.int16)
    env.step(joint_action)
    step_observation = visual_stack.update(env)
    reset_delta = np.max(np.abs(reset_observation[:, 0] - reset_observation[:, 1]))
    step_delta = np.max(np.abs(step_observation[:, 0] - step_observation[:, 1]))
    return _to_plain(
        {
            "ok": bool(reset_delta > 0.0 and step_delta > 0.0),
            "schema_id": PLAYER_PERSPECTIVE_SCHEMA_ID,
            "seed": int(seed),
            "batch_size": int(batch_size),
            "shape": list(reset_observation.shape),
            "reset_player_pair_max_abs_delta": float(reset_delta),
            "step_player_pair_max_abs_delta": float(step_delta),
            "value_map": player_perspective_value_map(),
        }
    )


def run_curvytron_current_policy_selfplay_smoke(
    *,
    seed: int = 0,
    batch_size: int = 1,
    steps: int = 8,
    max_ticks: int | None = None,
    decision_ms: float = 300.0,
    learner_updates: int = 1,
) -> dict[str, Any]:
    """Run a bounded two-seat rollout through the public joint-action env."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    env = VectorMultiplayerEnv(
        batch_size=batch_size,
        player_count=2,
        seed=seed,
        decision_ms=decision_ms,
        max_ticks=steps if max_ticks is None else max_ticks,
    )
    policy = SharedLinearSurvivalCurrentPolicy(seed=seed)
    visual_stack = SourceStateGray64Stack4(batch_size=batch_size, player_count=2)

    reset_batch = env.reset(seed=seed)
    observation = visual_stack.update(env)
    problems = _validate_visual_batch(observation, reset_batch.action_mask, label="reset")
    records: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    all_action_counts: Counter[int] = Counter()
    per_player_action_counts: dict[str, Counter[int]] = {
        "player_0": Counter(),
        "player_1": Counter(),
    }

    batch = reset_batch
    for decision_index in range(steps):
        mapping = build_policy_row_mapping(
            observation,
            batch.action_mask.any(axis=2),
            batch.action_mask,
        )
        active = np.asarray(mapping.row_mask, dtype=bool)
        if not bool(active.any()):
            break
        active_observations = mapping.observations[active]
        active_legal = mapping.legal_action_mask[active]
        active_env_row_id = mapping.env_row_id[active]
        active_player_id = mapping.player_id[active]
        selected = policy.select_actions(
            active_observations,
            active_legal,
            env_row_id=active_env_row_id,
            player_id=active_player_id,
            decision_index=decision_index,
        )
        joint_action = policy_rows_to_joint_action(
            mapping,
            selected.action,
            noop_action_id=NOOP_ACTION_ID,
            validate_legal=True,
            dtype=np.int16,
        )
        before_alive = batch.info.get("alive")
        step_batch = env.step(joint_action)
        alive_after = step_batch.info.get("alive")
        for policy_row in range(selected.action.shape[0]):
            env_row = int(active_env_row_id[policy_row])
            player = int(active_player_id[policy_row])
            replay_rows.append(
                {
                    "schema_id": "curvyzero_current_policy_two_seat_replay_row/v0",
                    "decision_index": int(decision_index),
                    "env_row_id": env_row,
                    "player_id": player,
                    "to_play": player,
                    "observation": active_observations[policy_row].copy(),
                    "legal_action_mask": active_legal[policy_row].copy(),
                    "action": int(selected.action[policy_row]),
                    "action_weights": selected.action_weights[policy_row].copy(),
                    "root_value": float(selected.root_value[policy_row]),
                    "reward": _survival_reward(alive_after, env_row=env_row, player_id=player),
                    "done": bool(np.asarray(step_batch.done, dtype=bool)[env_row]),
                    "policy_id": policy.policy_id,
                    "policy_version": policy.policy_version,
                    "policy_model_hash": policy.model_hash(),
                }
            )
        for env_row, player in np.argwhere(batch.action_mask.any(axis=2)):
            action = int(joint_action[int(env_row), int(player)])
            all_action_counts[action] += 1
            per_player_action_counts[f"player_{int(player)}"][action] += 1

        records.append(
            {
                "decision_index": int(decision_index),
                "policy_input_shape": list(mapping.observations[active].shape),
                "policy_input_rows": int(mapping.active_count),
                "shared_policy_object_id": policy.policy_id,
                "shared_policy_object_version": policy.policy_version,
                "joint_action": joint_action.copy(),
                "action_counts": _counter_dict(Counter(int(item) for item in joint_action.reshape(-1))),
                "reward": step_batch.reward.copy(),
                "done": step_batch.done.copy(),
                "terminated": step_batch.terminated.copy(),
                "truncated": step_batch.truncated.copy(),
                "alive_before": None if before_alive is None else np.asarray(before_alive).copy(),
                "alive_after": step_batch.info.get("alive"),
                "terminal_reason_name": step_batch.info.get("terminal_reason_name"),
                "env_joint_action_echo": step_batch.info.get("joint_action"),
            }
        )
        batch = step_batch
        observation = visual_stack.update(env)
        problems.extend(
            _validate_visual_batch(
                observation,
                batch.action_mask,
                label=f"step_{decision_index}",
            )
        )
        if bool(batch.done.any()):
            break

    learner_result = policy.learn_from_replay(replay_rows, updates=learner_updates)
    steps_survived_by_row = np.full(batch_size, len(records), dtype=np.int32)
    if records:
        done = np.asarray(records[-1]["done"], dtype=bool)
        steps_survived_by_row[done] = len(records)

    result = {
        "ok": not problems,
        "schema_id": CURRENT_POLICY_TWO_SEAT_TRAIN_SMOKE_SCHEMA_ID,
        "legacy_schema_id": CURRENT_POLICY_SELFPLAY_SMOKE_SCHEMA_ID,
        "mode": "bounded_local_current_policy_two_seat_train_smoke",
        "called_train_muzero": False,
        "true_lightzero_current_policy_self_play_training": False,
        "true_local_current_policy_two_seat_training_smoke": True,
        "simple_label": (
            "one shared current-policy object controls both CurvyTron players in a "
            "bounded local joint-action smoke"
        ),
        "honest_limits": [
            "does not call LightZero train_muzero",
            "does not use the LightZero collector",
            "does not use LightZero MCTS",
            "does not refresh weights across distributed actors",
            "the included learner is a tiny local numpy policy, not MuZero",
            "visuals are source-state gray64, not browser pixel fidelity",
        ],
        "what_works": [
            "builds float32 [B,P,4,64,64] observations",
            "remaps source player pixels into self/other perspective per policy row",
            "uses the same mutable current-policy object for both player seats",
            "maps policy rows back to joint_action [B,P]",
            "steps VectorMultiplayerEnv with external joint actions",
            "records two-seat replay rows with survival rewards",
            "runs a bounded learner update on the same current-policy object",
            "records survival steps and action counts",
        ],
        "problems": problems,
        "inputs": {
            "seed": int(seed),
            "batch_size": int(batch_size),
            "player_count": 2,
            "steps_requested": int(steps),
            "max_ticks": int(env.max_ticks),
            "decision_ms": float(decision_ms),
            "learner_updates": int(learner_updates),
        },
        "current_policy": {
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
            "same_object_for_all_players": True,
            "learned_policy": True,
            "mcts_search": False,
            "search_kind": "masked_softmax_policy_head_no_mcts",
            "learner_weight_refresh": "single_process_same_object_update",
            "model_hash_after_learning": policy.model_hash(),
            "next_adapter_todo": (
                "replace SharedLinearSurvivalCurrentPolicy.select_actions and "
                "learn_from_replay with live LightZero eval/search and optimizer "
                "calls inside a custom two-seat collector"
            ),
        },
        "surface": {
            "observation_shape": list(observation.shape),
            "per_policy_row_shape": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
            "observation_dtype": SOURCE_STATE_GRAY64_NORMALIZED_DTYPE,
            "value_range": list(SOURCE_STATE_GRAY64_NORMALIZED_VALUE_RANGE),
            "single_frame_schema_id": SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID,
            "single_frame_schema_hash": SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH,
            "stack_schema_id": STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
            "player_perspective": player_perspective_diagnostics(observation),
            "action_space_size": ACTION_COUNT,
            "joint_action_shape": [int(batch_size), 2],
        },
        "env": {
            "class": "VectorMultiplayerEnv",
            "public_contract": batch.info.get("public_env_contract_id"),
            "joint_action_schema_id": (
                batch.info.get("action_sidecar", {}).get("joint_action_schema_id")
            ),
            "episode_end_mode": batch.info.get("episode_end_mode"),
        },
        "steps_survived": int(len(records)),
        "steps_survived_by_row": steps_survived_by_row,
        "action_counts": _counter_dict(all_action_counts),
        "action_counts_by_player": {
            player: _counter_dict(counts)
            for player, counts in sorted(per_player_action_counts.items())
        },
        "reset": {
            "metadata_observation_shape": list(reset_batch.observation.shape),
            "visual_observation_shape": [
                int(item) for item in (batch_size, 2, *STACKED_SOURCE_STATE_GRAY64_SHAPE)
            ],
            "action_mask_shape": list(reset_batch.action_mask.shape),
            "alive": reset_batch.info.get("alive"),
        },
        "records": records,
        "replay": {
            "status": "ok" if replay_rows else "empty",
            "row_count": int(len(replay_rows)),
            "row_schema": "curvyzero_current_policy_two_seat_replay_row/v0",
            "reward": "steps_survived_per_player; 1.0 while alive after step else 0.0",
            "sample": _summarize_replay_sample(replay_rows),
        },
        "learner": learner_result,
        "integration_status": {
            "true_lightzero_current_policy_self_play_training": False,
            "true_local_current_policy_two_seat_smoke": True,
            "two_seat_current_policy_collector": "bounded_local_smoke",
            "search": "bounded_local_policy_head_no_mcts",
            "replay_writer": "bounded_local_in_memory_rows",
            "learner": "bounded_local_numpy_update",
            "lightzero_collector": "missing_custom_two_seat_collector",
            "lightzero_trainer": "not_integrated",
            "artifact_kind": "current_policy_collect_search_replay_learner_smoke",
        },
    }
    return _to_plain(result)


def _linear_policy_features(
    observation: np.ndarray,
    env_row_id: np.ndarray,
    player_id: np.ndarray,
) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.ndim != 4 or obs.shape[1:] != STACKED_SOURCE_STATE_GRAY64_SHAPE:
        raise ValueError(f"observation must have shape [R,4,64,64], got {obs.shape!r}")
    env_rows = np.asarray(env_row_id, dtype=np.float32).reshape(-1)
    players = np.asarray(player_id, dtype=np.float32).reshape(-1)
    if env_rows.shape != (obs.shape[0],) or players.shape != (obs.shape[0],):
        raise ValueError("env_row_id and player_id must have one value per observation")
    channel_mean = obs.mean(axis=(2, 3), dtype=np.float32)
    env_parity = ((env_rows % 2.0) * 2.0 - 1.0).reshape(-1, 1)
    player_sign = (players * 2.0 - 1.0).reshape(-1, 1)
    return np.concatenate(
        [
            np.ones((obs.shape[0], 1), dtype=np.float32),
            channel_mean.astype(np.float32),
            env_parity.astype(np.float32),
            player_sign.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def _masked_softmax(logits: np.ndarray, legal_action_mask: np.ndarray) -> np.ndarray:
    raw = np.asarray(logits, dtype=np.float32)
    legal = np.asarray(legal_action_mask, dtype=bool)
    if raw.shape != legal.shape:
        raise ValueError(f"logits and legal mask shapes differ: {raw.shape!r} vs {legal.shape!r}")
    if bool((~legal.any(axis=1)).any()):
        raise ValueError("every policy row needs at least one legal action")
    masked = np.where(legal, raw, -np.inf)
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    exp = np.where(legal, np.exp(shifted), 0.0)
    denom = exp.sum(axis=1, keepdims=True)
    return (exp / denom).astype(np.float32)


def _survival_reward(alive_after: Any, *, env_row: int, player_id: int) -> float:
    alive = np.asarray(alive_after, dtype=bool)
    if alive.ndim != 2:
        raise ValueError("step info alive must have shape [B,P] for survival reward")
    return 1.0 if bool(alive[int(env_row), int(player_id)]) else 0.0


def _discounted_survival_returns(
    replay_rows: list[dict[str, Any]],
    *,
    discount: float,
) -> np.ndarray:
    returns = np.zeros(len(replay_rows), dtype=np.float32)
    running: dict[tuple[int, int], float] = {}
    for index in range(len(replay_rows) - 1, -1, -1):
        row = replay_rows[index]
        key = (int(row["env_row_id"]), int(row["player_id"]))
        if bool(row["done"]):
            running[key] = 0.0
        value = float(row["reward"]) + float(discount) * running.get(key, 0.0)
        returns[index] = np.float32(value)
        running[key] = value
    return returns


def _linear_policy_loss(
    features: np.ndarray,
    legal_action_mask: np.ndarray,
    actions: np.ndarray,
    returns: np.ndarray,
    action_weights: np.ndarray,
    value_weights: np.ndarray,
) -> dict[str, float]:
    probabilities = _masked_softmax(features @ action_weights.T, legal_action_mask)
    values = features @ value_weights
    chosen = probabilities[np.arange(actions.shape[0]), actions]
    advantages = returns - values
    policy_loss = -float(np.mean(np.log(np.maximum(chosen, 1e-8)) * advantages))
    value_loss = float(np.mean(np.square(advantages)))
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "total_loss": policy_loss + value_loss,
        "mean_return": float(np.mean(returns)),
        "mean_value": float(np.mean(values)),
    }


def _summarize_replay_sample(replay_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not replay_rows:
        return {"ok": False, "reason": "empty replay"}
    observations = np.stack(
        [np.asarray(row["observation"], dtype=np.float32) for row in replay_rows],
        axis=0,
    )
    action_weights = np.stack(
        [np.asarray(row["action_weights"], dtype=np.float32) for row in replay_rows],
        axis=0,
    )
    rewards = np.asarray([float(row["reward"]) for row in replay_rows], dtype=np.float32)
    return {
        "ok": True,
        "observation_batch_shape": list(observations.shape),
        "action_weights_shape": list(action_weights.shape),
        "reward_shape": list(rewards.shape),
        "reward_sum": float(rewards.sum()),
        "players": sorted({int(row["player_id"]) for row in replay_rows}),
    }


def _validate_visual_batch(
    observation: np.ndarray,
    action_mask: np.ndarray,
    *,
    label: str,
) -> list[str]:
    problems: list[str] = []
    obs = np.asarray(observation)
    mask = np.asarray(action_mask)
    if obs.ndim != 5 or obs.shape[1:] != (2, *STACKED_SOURCE_STATE_GRAY64_SHAPE):
        problems.append(f"{label}: observation shape {obs.shape!r}, expected [B,2,4,64,64]")
    if obs.dtype != np.float32:
        problems.append(f"{label}: observation dtype {obs.dtype}, expected float32")
    if obs.size and (float(obs.min()) < 0.0 or float(obs.max()) > 1.0):
        problems.append(f"{label}: observation values are outside [0,1]")
    if mask.ndim != 3 or mask.shape[:2] != obs.shape[:2] or mask.shape[2] != ACTION_COUNT:
        problems.append(f"{label}: action_mask shape {mask.shape!r}, expected [B,2,3]")
    if (
        obs.ndim == 5
        and obs.shape[1:] == (2, *STACKED_SOURCE_STATE_GRAY64_SHAPE)
        and mask.ndim == 3
        and mask.shape[:2] == obs.shape[:2]
    ):
        active_pair_rows = mask.any(axis=2).all(axis=1)
        if bool(active_pair_rows.any()):
            delta = np.max(
                np.abs(obs[active_pair_rows, 0] - obs[active_pair_rows, 1])
            )
            if float(delta) <= 0.0:
                problems.append(f"{label}: active player visual frames are identical")
    return problems


def _counter_dict(counter: Counter[int]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _to_plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Counter):
        return _counter_dict(value)
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--decision-ms", type=float, default=300.0)
    parser.add_argument("--learner-updates", type=int, default=1)
    args = parser.parse_args()
    result = run_curvytron_current_policy_selfplay_smoke(
        seed=args.seed,
        batch_size=args.batch_size,
        steps=args.steps,
        max_ticks=args.max_ticks,
        decision_ms=args.decision_ms,
        learner_updates=args.learner_updates,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "CURRENT_POLICY_SELFPLAY_SMOKE_SCHEMA_ID",
    "CURRENT_POLICY_TWO_SEAT_TRAIN_SMOKE_SCHEMA_ID",
    "SHARED_SEEDED_CURRENT_POLICY_ID",
    "SHARED_SEEDED_CURRENT_POLICY_VERSION",
    "SHARED_LINEAR_SURVIVAL_CURRENT_POLICY_ID",
    "SHARED_LINEAR_SURVIVAL_CURRENT_POLICY_VERSION",
    "STACK_RENDER_MODE_DEFAULT",
    "STACK_RENDER_MODE_FAST_GRAY64_DIRECT",
    "STACK_RENDER_MODE_ORDER",
    "STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID",
    "PLAYER_PERSPECTIVE_SCHEMA_ID",
    "SharedLinearSurvivalCurrentPolicy",
    "SharedSeededCurrentPolicy",
    "SourceStateGray64Stack4",
    "normalize_source_state_player_perspective",
    "player_perspective_diagnostics",
    "player_perspective_rgb_palette",
    "player_perspective_value_map",
    "probe_curvytron_two_seat_player_perspective",
    "run_curvytron_current_policy_selfplay_smoke",
    "source_state_gray64_stack4_render_metadata",
    "validate_stack_trail_render_mode",
]
