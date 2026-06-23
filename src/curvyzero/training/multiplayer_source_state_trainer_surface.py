"""Trainer-facing multiplayer source-state visual surface.

This module is intentionally a thin adapter over ``VectorMultiplayerEnv``. It
does not implement environment mechanics; it swaps the public metadata-only
observation for the existing source-state-backed visual stack used by trainer
smokes.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from curvyzero.env import vector_lifecycle
from curvyzero.env.observation_surface_contract import (
    POLICY_OBSERVATION_PERSPECTIVE,
    POLICY_OBSERVATION_PERSPECTIVE_PLAYER_AXIS,
    POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID,
    policy_observation_surface,
)
from curvyzero.env.trainer_contract import stable_contract_hash
from curvyzero.env.vector_multiplayer_env import JOINT_ACTION_SCHEMA_ID
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.env.vector_visual_observation import BONUS_RENDER_MODE_DEFAULT
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SHAPE
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_USES_ALE
from curvyzero.env.vector_visual_observation import SOURCE_STATE_GRAY64_NORMALIZED_DTYPE
from curvyzero.env.vector_visual_observation import SOURCE_STATE_GRAY64_NORMALIZED_VALUE_RANGE
from curvyzero.env.vector_visual_observation import TRAIL_RENDER_MODE_BROWSER_LINES
from curvyzero.training.curvytron_current_policy_selfplay_smoke import (
    STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
)
from curvyzero.training.curvytron_current_policy_selfplay_smoke import (
    SourceStateGray64Stack4,
)
from curvyzero.training.curvytron_current_policy_selfplay_smoke import (
    source_state_gray64_stack4_render_metadata,
)
from curvyzero.training.curvytron_current_policy_selfplay_smoke import (
    validate_stack_trail_render_mode,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_HASH,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_ID,
)
from curvyzero.training.source_state_batched_observation_profile import (
    SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND,
    SourceStateBatchedObservationRenderer,
    SourceStateBatchedRenderRequest,
)


MULTIPLAYER_TRAINER_SURFACE_SCHEMA_ID = (
    "curvyzero_source_state_multiplayer_trainer_surface/v0"
)
FINAL_VISUAL_OBSERVATION_POLICY_ID = (
    "curvyzero_source_state_visual_final_observation_rows/v0"
)
FINAL_REWARD_MAP_POLICY_ID = "curvyzero_survival_bonus_final_reward_rows/v0"
FRAME_STACK_OWNER = "SourceStateMultiplayerTrainerSurface"
NATIVE_SOURCE_CONTROL_MODEL = "real_time_control_state_plus_elapsed_ms_source_frames"
JOINT_ACTION_LABEL = "wrapper-facing player-major control action"
TRAINER_OBSERVATION_SOURCE = "SourceStateGray64Stack4"
TRAINER_OBSERVATION_SOURCE_RENDERER_BACKED = "RendererBackedSourceStateGray64Stack4"
TRAINER_STACK_BACKEND_CPU_DIRTY_CACHE = "cpu_dirty_cache"
TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE = "renderer_backed_profile"
TRAINER_STACK_BACKENDS = (
    TRAINER_STACK_BACKEND_CPU_DIRTY_CACHE,
    TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
)
TRAINER_ALLOWED_TRAIL_RENDER_MODES = (TRAIL_RENDER_MODE_BROWSER_LINES,)

MULTIPLAYER_TRAINER_SURFACE_SCHEMA = {
    "schema_id": MULTIPLAYER_TRAINER_SURFACE_SCHEMA_ID,
    "underlying_env": "curvyzero.env.vector_multiplayer_env.VectorMultiplayerEnv",
    "observation_schema_id": STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
    "single_frame_schema_id": SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID,
    "shape": ["batch", "player", 4, 64, 64],
    "dtype": SOURCE_STATE_GRAY64_NORMALIZED_DTYPE,
    "value_range": list(SOURCE_STATE_GRAY64_NORMALIZED_VALUE_RANGE),
    "perspective_schema_id": POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID,
    "perspective": POLICY_OBSERVATION_PERSPECTIVE,
    "perspective_player_axis": POLICY_OBSERVATION_PERSPECTIVE_PLAYER_AXIS,
    "reward_schema_id": SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_ID,
    "joint_action_schema_id": JOINT_ACTION_SCHEMA_ID,
}
MULTIPLAYER_TRAINER_SURFACE_SCHEMA_HASH = stable_contract_hash(
    MULTIPLAYER_TRAINER_SURFACE_SCHEMA
)


@dataclass(frozen=True, slots=True)
class MultiplayerTrainerStepV0:
    """Batch returned by the multiplayer trainer surface."""

    observation: np.ndarray
    legal_action_mask: np.ndarray
    lightzero_action_mask: np.ndarray
    live_mask: np.ndarray
    policy_observation: np.ndarray
    policy_action_mask: np.ndarray
    policy_env_row: np.ndarray
    policy_player: np.ndarray
    joint_action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    final_observation: np.ndarray
    final_observation_row_mask: np.ndarray
    final_reward_map: np.ndarray
    info: dict[str, Any]


class _RendererBackedSourceStateGray64Stack4:
    """Profile-only stack adapter that gets all player frames from one renderer call."""

    stack_class_name = TRAINER_OBSERVATION_SOURCE_RENDERER_BACKED

    def __init__(
        self,
        *,
        batch_size: int,
        player_count: int,
        trail_render_mode: str,
        renderer: SourceStateBatchedObservationRenderer,
    ) -> None:
        self.batch_size = int(batch_size)
        self.player_count = int(player_count)
        self.trail_render_mode = validate_stack_trail_render_mode(trail_render_mode)
        self.renderer = renderer
        self.renderer_backend_name = str(renderer.backend_name)
        self.stack = np.zeros(
            (self.batch_size, self.player_count, 4, 64, 64),
            dtype=np.float32,
        )
        self._raw_all = np.zeros(
            (self.batch_size * self.player_count, *SOURCE_STATE_CANVAS_GRAY64_SHAPE),
            dtype=np.uint8,
        )
        self._all_rows = np.arange(self.batch_size, dtype=np.int64)
        self._all_row_indices = np.repeat(self._all_rows, self.player_count)
        self._all_controlled_players = np.tile(
            np.arange(self.player_count, dtype=np.int64),
            self.batch_size,
        )
        self._render_calls = 0
        self.last_renderer_telemetry: dict[str, Any] = {}

    def render_metadata(self) -> dict[str, Any]:
        metadata = source_state_gray64_stack4_render_metadata(self.trail_render_mode)
        metadata.update(
            {
                "single_frame_render_api": "SourceStateBatchedObservationRenderer.render",
                "two_seat_optimized_render_api": "batched_renderer_all_player_views",
                "renderer_backed_profile": True,
                "renderer_backend_name": self.renderer_backend_name,
                "no_hidden_cpu_fallback": True,
                "profile_gpu_candidate_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND
                ),
            }
        )
        return metadata

    def dirty_render_stats(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "rows": self.batch_size,
            "attempts": self._render_calls,
            "hits": 0,
            "cold_starts": self._render_calls,
            "fallbacks": 0,
            "dirty_blocks_total": 0,
            "hit_rate": None,
            "dirty_blocks_per_hit": None,
        }

    def update(self, env: VectorMultiplayerEnv, *, copy: bool = True) -> np.ndarray:
        self._validate_env(env)
        self.stack[:, :, :-1] = self.stack[:, :, 1:]
        rows = np.arange(self.batch_size, dtype=np.int64)
        frames = self._render_rows(env, rows)
        self._write_latest(rows, frames)
        return self.stack.copy() if copy else self.stack

    def reset_rows(
        self,
        env: VectorMultiplayerEnv,
        row_mask: np.ndarray,
        *,
        copy: bool = True,
    ) -> np.ndarray:
        self._validate_env(env)
        mask = np.asarray(row_mask, dtype=bool)
        if mask.shape != (self.batch_size,):
            raise ValueError("row_mask must have shape [B]")
        rows = np.flatnonzero(mask).astype(np.int64)
        if rows.size:
            self.stack[rows] = 0.0
            frames = self._render_rows(env, rows)
            self._write_latest(rows, frames)
        return self.stack.copy() if copy else self.stack

    def _validate_env(self, env: VectorMultiplayerEnv) -> None:
        if env.batch_size != self.batch_size or env.player_count != self.player_count:
            raise ValueError("env shape changed after stack creation")

    def _render_rows(self, env: VectorMultiplayerEnv, rows: np.ndarray) -> np.ndarray:
        rows = rows.astype(np.int64, copy=False)
        if rows.shape == self._all_rows.shape and bool(np.array_equal(rows, self._all_rows)):
            row_indices = self._all_row_indices
            controlled_players = self._all_controlled_players
        else:
            row_indices = np.repeat(rows, self.player_count)
            controlled_players = np.tile(
                np.arange(self.player_count, dtype=np.int64),
                rows.size,
            )
        out = self._raw_all[: row_indices.size]
        request = SourceStateBatchedRenderRequest(
            state=env.state,
            row_indices=row_indices,
            controlled_players=controlled_players,
            out=out,
            trail_render_mode=self.trail_render_mode,
            bonus_render_mode=BONUS_RENDER_MODE_DEFAULT,
        )
        result = self.renderer.render(request)
        frames = np.asarray(result.frames)
        if frames.shape != out.shape:
            raise ValueError(
                "renderer returned frames with unexpected shape; "
                f"got {frames.shape}, expected {out.shape}"
            )
        if frames.dtype != np.uint8:
            raise ValueError(f"renderer frames must be uint8, got {frames.dtype}")
        self._render_calls += 1
        self.last_renderer_telemetry = dict(result.telemetry)
        return frames.reshape(rows.size, self.player_count, *SOURCE_STATE_CANVAS_GRAY64_SHAPE)

    def _write_latest(self, rows: np.ndarray, frames: np.ndarray) -> None:
        if rows.shape == self._all_rows.shape and bool(np.array_equal(rows, self._all_rows)):
            np.multiply(
                frames[:, :, 0],
                np.float32(1.0 / 255.0),
                out=self.stack[:, :, -1],
                casting="unsafe",
            )
        else:
            self.stack[rows, :, -1] = frames[:, :, 0].astype(
                np.float32,
                copy=False,
            ) * np.float32(1.0 / 255.0)


class SourceStateMultiplayerTrainerSurface:
    """Small source-state visual trainer surface backed by ``VectorMultiplayerEnv``."""

    def __init__(
        self,
        batch_size: int = 1,
        *,
        player_count: int = 2,
        seed: int | None = None,
        trail_render_mode: str = TRAIL_RENDER_MODE_BROWSER_LINES,
        observation_stack_backend: str = TRAINER_STACK_BACKEND_CPU_DIRTY_CACHE,
        observation_renderer: SourceStateBatchedObservationRenderer | None = None,
        required_observation_renderer_backend: str | None = None,
        env: VectorMultiplayerEnv | None = None,
        **env_kwargs: Any,
    ) -> None:
        mode = validate_stack_trail_render_mode(trail_render_mode)
        stack_backend = _validate_observation_stack_backend(observation_stack_backend)

        if env is not None and env_kwargs:
            raise ValueError("env_kwargs cannot be supplied when env is provided")
        self.env = (
            env
            if env is not None
            else VectorMultiplayerEnv(
                batch_size=batch_size,
                player_count=player_count,
                seed=seed,
                **env_kwargs,
            )
        )
        self.batch_size = int(self.env.batch_size)
        self.player_count = int(self.env.player_count)
        self.trail_render_mode = mode
        self.observation_stack_backend = stack_backend
        self.required_observation_renderer_backend = (
            None
            if required_observation_renderer_backend is None
            else str(required_observation_renderer_backend)
        )
        if stack_backend == TRAINER_STACK_BACKEND_CPU_DIRTY_CACHE:
            if observation_renderer is not None:
                raise ValueError(
                    "observation_renderer is only valid with "
                    f"{TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE!r}"
                )
            if self.required_observation_renderer_backend is not None:
                raise ValueError(
                    "required_observation_renderer_backend is only valid with "
                    f"{TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE!r}"
                )
            self.stack = SourceStateGray64Stack4(
                batch_size=self.batch_size,
                player_count=self.player_count,
                trail_render_mode=self.trail_render_mode,
            )
        else:
            if observation_renderer is None:
                raise ValueError(
                    f"{TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE!r} requires an "
                    "explicit observation_renderer; no hidden CPU fallback is allowed"
                )
            renderer_backend = str(observation_renderer.backend_name)
            if (
                self.required_observation_renderer_backend is not None
                and renderer_backend != self.required_observation_renderer_backend
            ):
                raise ValueError(
                    "observation_renderer backend mismatch; "
                    f"expected {self.required_observation_renderer_backend!r}, "
                    f"got {renderer_backend!r}"
                )
            self.stack = _RendererBackedSourceStateGray64Stack4(
                batch_size=self.batch_size,
                player_count=self.player_count,
                trail_render_mode=self.trail_render_mode,
                renderer=observation_renderer,
            )

    def reset(
        self,
        seed: int | np.ndarray | None = None,
        *,
        row_mask: np.ndarray | None = None,
        present: np.ndarray | None = None,
        source_fixture_random_tape_values: np.ndarray | None = None,
        source_fixture_ref: str | None = None,
        source_fixture_new_round_time_ms: float | None = None,
        source_fixture_warmup_advance_ms: float | np.ndarray | None = None,
    ) -> MultiplayerTrainerStepV0:
        batch = self.env.reset(
            seed=seed,
            row_mask=row_mask,
            present=present,
            source_fixture_random_tape_values=source_fixture_random_tape_values,
            source_fixture_ref=source_fixture_ref,
            source_fixture_new_round_time_ms=source_fixture_new_round_time_ms,
            source_fixture_warmup_advance_ms=source_fixture_warmup_advance_ms,
        )
        reset_mask = self._reset_row_mask(row_mask)
        observation = self.stack.reset_rows(self.env, reset_mask)
        reward = np.zeros((self.batch_size, self.player_count), dtype=np.float32)
        joint_action = np.full((self.batch_size, self.player_count), -1, dtype=np.int16)
        return self._surface_step(
            batch=batch,
            observation=observation,
            reward=reward,
            joint_action=joint_action,
            api="reset",
        )

    def step(
        self,
        joint_action: np.ndarray,
        *,
        timer_advance_ms: float | np.ndarray | None = None,
        disabled_player_mask: np.ndarray | None = None,
    ) -> MultiplayerTrainerStepV0:
        action = self._joint_action_array(joint_action)
        timing_enabled = self.observation_stack_backend == TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE
        started = time.perf_counter()
        batch = self.env.step(
            action,
            timer_advance_ms=timer_advance_ms,
            disabled_player_mask=disabled_player_mask,
        )
        env_step_sec = time.perf_counter() - started
        started = time.perf_counter()
        observation = self.stack.update(self.env, copy=not timing_enabled)
        stack_update_sec = time.perf_counter() - started
        started = time.perf_counter()
        reward = self._survival_plus_bonus_reward(batch.info, batch.done)
        reward_sec = time.perf_counter() - started
        started = time.perf_counter()
        surface_step = self._surface_step(
            batch=batch,
            observation=observation,
            reward=reward,
            joint_action=action,
            api="step",
        )
        package_sec = time.perf_counter() - started
        if timing_enabled:
            surface_timing = dict(surface_step.info.get("trainer_surface_profile_timing", {}))
            surface_timing.update(
                {
                "env_step_sec": env_step_sec,
                "stack_update_sec": stack_update_sec,
                "reward_sec": reward_sec,
                "package_sec": package_sec,
                }
            )
            surface_step.info["trainer_surface_profile_timing"] = surface_timing
        return surface_step

    def remove_player(
        self,
        player_ids: int | np.ndarray,
        *,
        row_mask: np.ndarray | None = None,
    ) -> MultiplayerTrainerStepV0:
        """Package a public leave event without implementing leave mechanics."""

        batch = self.env.remove_player(player_ids, row_mask=row_mask)
        observation = self.stack.update(self.env)
        reward = self._survival_plus_bonus_reward(batch.info, batch.done)
        joint_action = np.full(
            (self.batch_size, self.player_count),
            -1,
            dtype=np.int16,
        )
        return self._surface_step(
            batch=batch,
            observation=observation,
            reward=reward,
            joint_action=joint_action,
            api="remove_player",
        )

    def advance_warmdown(
        self,
        advance_ms: float | np.ndarray = vector_lifecycle.SOURCE_ROUND_WARMDOWN_MS,
        *,
        next_warmup_ms: float = vector_lifecycle.SOURCE_ROUND_WARMUP_MS,
        max_timer_callbacks: int = 16,
    ) -> MultiplayerTrainerStepV0:
        """Advance source-shaped warmdown rows and keep trainer arrays aligned."""

        batch = self.env.advance_warmdown(
            advance_ms,
            next_warmup_ms=next_warmup_ms,
            max_timer_callbacks=max_timer_callbacks,
        )
        self.stack.update(self.env, copy=False)
        next_round_rows = np.asarray(
            batch.info.get("next_round_rows", np.asarray([], dtype=np.int32)),
            dtype=np.int32,
        )
        if next_round_rows.size:
            reset_mask = np.zeros(self.batch_size, dtype=bool)
            reset_mask[next_round_rows] = True
            observation = self.stack.reset_rows(self.env, reset_mask)
        else:
            observation = self.stack.stack.copy()
        reward = np.asarray(batch.reward, dtype=np.float32).copy()
        joint_action = np.full(
            (self.batch_size, self.player_count),
            -1,
            dtype=np.int16,
        )
        return self._surface_step(
            batch=batch,
            observation=observation,
            reward=reward,
            joint_action=joint_action,
            api="advance_warmdown",
        )

    def advance_warmup(
        self,
        advance_ms: float | np.ndarray = vector_lifecycle.SOURCE_TRAIL_START_DELAY_MS,
        *,
        max_timer_callbacks: int | None = None,
    ) -> MultiplayerTrainerStepV0:
        """Advance source-shaped warmup rows and keep trainer arrays aligned."""

        batch = self.env.advance_warmup(
            advance_ms,
            max_timer_callbacks=max_timer_callbacks,
        )
        observation = self.stack.update(self.env)
        reward = np.asarray(batch.reward, dtype=np.float32).copy()
        joint_action = np.full(
            (self.batch_size, self.player_count),
            -1,
            dtype=np.int16,
        )
        return self._surface_step(
            batch=batch,
            observation=observation,
            reward=reward,
            joint_action=joint_action,
            api="advance_warmup",
        )

    def _surface_step(
        self,
        *,
        batch: Any,
        observation: np.ndarray,
        reward: np.ndarray,
        joint_action: np.ndarray,
        api: str,
    ) -> MultiplayerTrainerStepV0:
        timing_enabled = (
            self.observation_stack_backend == TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE
        )
        profile_timing: dict[str, float] | None = {} if timing_enabled else None
        started = time.perf_counter() if profile_timing is not None else 0.0
        legal_action_mask = np.asarray(batch.action_mask, dtype=bool).copy()
        lightzero_action_mask = legal_action_mask.copy()
        done = np.asarray(batch.done, dtype=bool).copy()
        terminated = np.asarray(batch.terminated, dtype=bool).copy()
        truncated = np.asarray(batch.truncated, dtype=bool).copy()
        if profile_timing is not None:
            profile_timing["package_mask_copy_sec"] = time.perf_counter() - started
            started = time.perf_counter()
        live_mask = self._live_mask(batch.info, done, legal_action_mask)
        if profile_timing is not None:
            profile_timing["package_live_mask_sec"] = time.perf_counter() - started
            started = time.perf_counter()
        policy_env_row, policy_player = self._policy_rows(live_mask)
        if profile_timing is not None:
            profile_timing["package_policy_rows_sec"] = time.perf_counter() - started
            started = time.perf_counter()
        observation_array = np.asarray(observation, dtype=np.float32)
        if (
            self.observation_stack_backend == TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE
            and self._policy_rows_are_full_row_major(policy_env_row, policy_player)
        ):
            policy_observation = observation_array.reshape(
                self.batch_size * self.player_count,
                *observation_array.shape[2:],
            )
        else:
            policy_observation = observation_array[policy_env_row, policy_player].astype(
                np.float32,
                copy=True,
            )
        if profile_timing is not None:
            profile_timing["package_policy_observation_sec"] = time.perf_counter() - started
            started = time.perf_counter()
        policy_action_mask = legal_action_mask[policy_env_row, policy_player].copy()
        if profile_timing is not None:
            profile_timing["package_policy_action_mask_sec"] = time.perf_counter() - started
            started = time.perf_counter()
        final_row_mask = np.asarray(
            batch.info.get("final_observation_row_mask", np.zeros(self.batch_size, dtype=bool)),
            dtype=bool,
        ).copy()
        if final_row_mask.shape != (self.batch_size,):
            final_row_mask = np.zeros(self.batch_size, dtype=bool)

        final_any = bool(final_row_mask.any())
        if final_any:
            final_observation = np.zeros_like(observation_array, dtype=np.float32)
            final_reward_map = np.zeros_like(reward, dtype=np.float32)
            final_observation[final_row_mask] = observation_array[final_row_mask]
            final_reward_map[final_row_mask] = reward[final_row_mask]
        else:
            final_observation = np.broadcast_to(np.float32(0.0), observation_array.shape)
            final_reward_map = np.broadcast_to(np.float32(0.0), reward.shape)
        if profile_timing is not None:
            profile_timing["package_final_observation_sec"] = time.perf_counter() - started
            started = time.perf_counter()

        info = self._info(
            batch_info=batch.info,
            legal_action_mask=legal_action_mask,
            joint_action=joint_action,
            reward=reward,
            live_mask=live_mask,
            policy_action_mask=policy_action_mask,
            policy_env_row=policy_env_row,
            policy_player=policy_player,
            final_observation=final_observation,
            final_observation_row_mask=final_row_mask,
            final_reward_map=final_reward_map,
            api=api,
            underlying_final_observation=batch.final_observation,
            underlying_observation=batch.observation,
        )
        if profile_timing is not None:
            profile_timing["package_info_sec"] = time.perf_counter() - started
            started = time.perf_counter()
            info["trainer_surface_profile_timing"] = profile_timing
        observation_output = (
            observation_array
            if self.observation_stack_backend == TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE
            else observation_array.copy()
        )
        if profile_timing is not None:
            profile_timing["package_output_copy_sec"] = time.perf_counter() - started
        return MultiplayerTrainerStepV0(
            observation=observation_output,
            legal_action_mask=legal_action_mask,
            lightzero_action_mask=lightzero_action_mask,
            live_mask=live_mask,
            policy_observation=policy_observation,
            policy_action_mask=policy_action_mask,
            policy_env_row=policy_env_row,
            policy_player=policy_player,
            joint_action=joint_action.copy(),
            reward=np.asarray(reward, dtype=np.float32).copy(),
            done=done,
            terminated=terminated,
            truncated=truncated,
            final_observation=final_observation.copy() if final_any else final_observation,
            final_observation_row_mask=final_row_mask,
            final_reward_map=final_reward_map.copy() if final_any else final_reward_map,
            info=info,
        )

    def _info(
        self,
        *,
        batch_info: dict[str, Any],
        legal_action_mask: np.ndarray,
        joint_action: np.ndarray,
        reward: np.ndarray,
        live_mask: np.ndarray,
        policy_action_mask: np.ndarray,
        policy_env_row: np.ndarray,
        policy_player: np.ndarray,
        final_observation: np.ndarray,
        final_observation_row_mask: np.ndarray,
        final_reward_map: np.ndarray,
        api: str,
        underlying_final_observation: np.ndarray | None,
        underlying_observation: np.ndarray,
    ) -> dict[str, Any]:
        rows = np.flatnonzero(final_observation_row_mask).astype(np.int32)
        render_metadata = self.stack.render_metadata()
        visual_stack_class = str(
            getattr(self.stack, "stack_class_name", TRAINER_OBSERVATION_SOURCE)
        )
        renderer_backend = getattr(self.stack, "renderer_backend_name", None)
        renderer_telemetry = getattr(self.stack, "last_renderer_telemetry", None)
        approximate = False
        info = dict(batch_info)
        info.update(
            {
                "trainer_surface_schema_id": MULTIPLAYER_TRAINER_SURFACE_SCHEMA_ID,
                "trainer_surface_schema_hash": MULTIPLAYER_TRAINER_SURFACE_SCHEMA_HASH,
                "trainer_surface_api": api,
                "visual_source_state_backed": True,
                "source_state_backed": True,
                "rgb_to_gray64": True,
                "debug_fidelity_only": False,
                "metadata_only": False,
                "underlying_env_metadata_only": bool(batch_info.get("metadata_only", False)),
                "underlying_env_observation_schema_id": batch_info.get(
                    "observation_schema_id"
                ),
                "underlying_env_observation_shape": tuple(
                    np.asarray(underlying_observation).shape
                ),
                "underlying_env_observation_is_metadata_only": bool(
                    batch_info.get("metadata_only", False)
                ),
                "underlying_env_observation_used_as_trainer_observation": False,
                "underlying_env_class": "VectorMultiplayerEnv",
                "visual_stack_class": visual_stack_class,
                "trainer_observation_source": visual_stack_class,
                "trainer_observation_stack_backend": self.observation_stack_backend,
                "trainer_observation_renderer_backend": renderer_backend,
                "trainer_observation_required_renderer_backend": (
                    self.required_observation_renderer_backend
                ),
                "renderer_backed_stack_profile": (
                    self.observation_stack_backend
                    == TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE
                ),
                "profile_gpu_candidate_backend": (
                    SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND
                ),
                "trainer_observation_no_hidden_fallback": (
                    self.observation_stack_backend
                    != TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE
                    or renderer_backend is not None
                ),
                "renderer_backed_stack_telemetry": (
                    None if renderer_telemetry is None else dict(renderer_telemetry)
                ),
                "trainer_observation_claim": True,
                "trainer_observation_claim_id": (
                    "source_state_visual_stack_per_live_seat/v0"
                ),
                "trainer_replay_claim": False,
                "trainer_replay_claim_id": None,
                "uses_ale": bool(SOURCE_STATE_CANVAS_GRAY64_USES_ALE),
                "browser_pixel_fidelity": False,
                "trail_render_mode": self.trail_render_mode,
                "default_trail_render_mode": TRAIL_RENDER_MODE_BROWSER_LINES,
                "trainer_supported_trail_render_modes": list(
                    TRAINER_ALLOWED_TRAIL_RENDER_MODES
                ),
                "bonus_render_mode": render_metadata.get("bonus_render_mode"),
                "default_bonus_render_mode": render_metadata.get(
                    "default_bonus_render_mode",
                    BONUS_RENDER_MODE_DEFAULT,
                ),
                "browser_sprites_bonus_render_claim": False,
                "single_frame_schema_id": SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID,
                "single_frame_schema_hash": SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH,
                "frame_stack_owner": FRAME_STACK_OWNER,
                "visual_stack_dirty_render_stats": self.stack.dirty_render_stats(),
                "observation_schema_id": STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
                "trainer_observation_schema_id": STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
                "policy_observation_contract": policy_observation_surface(
                    trail_render_mode=self.trail_render_mode,
                    bonus_render_mode=str(
                        render_metadata.get("bonus_render_mode", BONUS_RENDER_MODE_DEFAULT)
                    ),
                ),
                "policy_observation_perspective_schema_id": (
                    POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID
                ),
                "policy_observation_perspective": POLICY_OBSERVATION_PERSPECTIVE,
                "policy_observation_perspective_player_axis": (
                    POLICY_OBSERVATION_PERSPECTIVE_PLAYER_AXIS
                ),
                "source_state_player_perspective": True,
                "policy_observation_perspective_player": policy_player.copy(),
                "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
                "reward_schema_id": SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_ID,
                "reward_schema_hash": SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_HASH,
                "reward_formula": (
                    "alive_after_step + bonus_catch_count_step * "
                    "SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD"
                ),
                "bonus_pickup_reward_per_catch": (
                    SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD
                ),
                "decision_cadence_is_wrapper_abstraction": True,
                "native_source_control_model": NATIVE_SOURCE_CONTROL_MODEL,
                "native_source_control_model_id": NATIVE_SOURCE_CONTROL_MODEL,
                "joint_action": joint_action.copy(),
                "joint_action_label": JOINT_ACTION_LABEL,
                "joint_action_schema_id": JOINT_ACTION_SCHEMA_ID,
                "joint_action_player_major": True,
                "underlying_env_action_mask": legal_action_mask.copy(),
                "legal_action_mask": legal_action_mask.copy(),
                "lightzero_action_mask": legal_action_mask.copy(),
                "live_policy_row_mask": live_mask.copy(),
                "policy_row_mapping_schema_id": (
                    "curvyzero_source_state_multiplayer_live_policy_rows/v0"
                ),
                "policy_row_count": int(policy_env_row.size),
                "policy_env_row": policy_env_row.copy(),
                "policy_player": policy_player.copy(),
                "policy_action_mask": policy_action_mask.copy(),
                "final_observation": (
                    final_observation.copy()
                    if bool(final_observation_row_mask.any())
                    else None
                ),
                "final_observation_rows": rows.copy(),
                "final_observation_row_mask": final_observation_row_mask.copy(),
                "final_observation_policy": {
                    "schema_id": FINAL_VISUAL_OBSERVATION_POLICY_ID,
                    "rows": rows.copy(),
                    "row_mask": final_observation_row_mask.copy(),
                    "terminal_rows_only": True,
                    "nonterminal_rows_zero_filled": True,
                    "observation_schema_id": STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
                    "single_frame_schema_id": SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID,
                    "source_claim": "source_state_visual_terminal_rows/v0",
                    "metadata_only": False,
                },
                "final_reward_map": (
                    final_reward_map.copy()
                    if bool(final_observation_row_mask.any())
                    else None
                ),
                "final_reward_rows": rows.copy(),
                "final_reward_row_mask": final_observation_row_mask.copy(),
                "final_reward_policy": {
                    "schema_id": FINAL_REWARD_MAP_POLICY_ID,
                    "rows": rows.copy(),
                    "row_mask": final_observation_row_mask.copy(),
                    "terminal_rows_only": True,
                    "nonterminal_rows_zero_filled": True,
                    "reward_schema_id": SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_ID,
                    "source_claim": "surface_survival_plus_bonus_terminal_rows/v0",
                },
                "reward": reward.copy(),
                "underlying_final_observation_shape": (
                    None
                    if underlying_final_observation is None
                    else tuple(np.asarray(underlying_final_observation).shape)
                ),
                "underlying_final_observation_schema_id": batch_info.get(
                    "observation_schema_id"
                ),
                "render_metadata": render_metadata,
                "trail_renderer_is_approximation": bool(
                    render_metadata.get("trail_renderer_is_approximation", False)
                ),
                "visual_observation_is_approximation": bool(approximate),
                "approximate_trail_render_mode": bool(approximate),
            }
        )
        return info

    def _survival_plus_bonus_reward(
        self,
        batch_info: dict[str, Any],
        done: np.ndarray,
    ) -> np.ndarray:
        present = np.asarray(
            batch_info.get("present", self.env.state["present"][:, : self.player_count]),
            dtype=bool,
        )
        alive = np.asarray(
            batch_info.get("alive", self.env.state["alive"][:, : self.player_count]),
            dtype=bool,
        )
        if present.shape != (self.batch_size, self.player_count):
            present = self.env.state["present"][:, : self.player_count].astype(bool, copy=True)
        if alive.shape != (self.batch_size, self.player_count):
            alive = self.env.state["alive"][:, : self.player_count].astype(bool, copy=True)
        del done
        alive_after = present & alive
        bonus = np.asarray(
            batch_info.get(
                "bonus_catch_count_step",
                np.zeros((self.batch_size, self.player_count), dtype=np.int16),
            ),
            dtype=np.float32,
        )
        if bonus.shape != (self.batch_size, self.player_count):
            bonus = np.zeros((self.batch_size, self.player_count), dtype=np.float32)
        return (
            alive_after.astype(np.float32)
            + bonus * np.float32(SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD)
        ).astype(np.float32)

    def _live_mask(
        self,
        batch_info: dict[str, Any],
        done: np.ndarray,
        legal_action_mask: np.ndarray,
    ) -> np.ndarray:
        present = np.asarray(
            batch_info.get("present", self.env.state["present"][:, : self.player_count]),
            dtype=bool,
        )
        alive = np.asarray(
            batch_info.get("alive", self.env.state["alive"][:, : self.player_count]),
            dtype=bool,
        )
        if present.shape != (self.batch_size, self.player_count):
            present = self.env.state["present"][:, : self.player_count].astype(bool, copy=True)
        if alive.shape != (self.batch_size, self.player_count):
            alive = self.env.state["alive"][:, : self.player_count].astype(bool, copy=True)
        legal = np.asarray(legal_action_mask, dtype=bool)
        if legal.shape != (self.batch_size, self.player_count, 3):
            legal_player = np.zeros((self.batch_size, self.player_count), dtype=bool)
        else:
            legal_player = legal.any(axis=2)
        return (
            present
            & alive
            & ~np.asarray(done, dtype=bool)[:, None]
            & legal_player
        ).copy()

    def _policy_rows(self, live_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = np.asarray(live_mask, dtype=bool)
        if mask.shape != (self.batch_size, self.player_count):
            raise ValueError("live_mask must have shape [B,P]")
        env_row, player = np.nonzero(mask)
        return env_row.astype(np.int32, copy=True), player.astype(np.int16, copy=True)

    def _policy_rows_are_full_row_major(
        self,
        policy_env_row: np.ndarray,
        policy_player: np.ndarray,
    ) -> bool:
        expected_count = self.batch_size * self.player_count
        if policy_env_row.shape != (expected_count,) or policy_player.shape != (expected_count,):
            return False
        expected_rows = np.repeat(np.arange(self.batch_size, dtype=np.int32), self.player_count)
        expected_players = np.tile(
            np.arange(self.player_count, dtype=np.int16),
            self.batch_size,
        )
        return bool(
            np.array_equal(policy_env_row, expected_rows)
            and np.array_equal(policy_player, expected_players)
        )

    def _reset_row_mask(self, row_mask: np.ndarray | None) -> np.ndarray:
        if row_mask is None:
            return np.ones(self.batch_size, dtype=bool)
        mask = np.asarray(row_mask, dtype=bool)
        if mask.shape != (self.batch_size,):
            raise ValueError("row_mask must have shape [B]")
        return mask.copy()

    def _joint_action_array(self, joint_action: np.ndarray) -> np.ndarray:
        action = np.asarray(joint_action)
        if action.shape != (self.batch_size, self.player_count):
            raise ValueError("joint_action must have shape [B,P]")
        if not np.issubdtype(action.dtype, np.integer):
            raise ValueError("joint_action must contain integer action ids")
        return action.astype(np.int16, copy=True)


def _validate_observation_stack_backend(value: str) -> str:
    backend = str(value)
    if backend not in TRAINER_STACK_BACKENDS:
        supported = ", ".join(TRAINER_STACK_BACKENDS)
        raise ValueError(
            f"observation_stack_backend must be one of [{supported}], got {value!r}"
        )
    return backend


__all__ = [
    "FINAL_REWARD_MAP_POLICY_ID",
    "FINAL_VISUAL_OBSERVATION_POLICY_ID",
    "FRAME_STACK_OWNER",
    "JOINT_ACTION_LABEL",
    "MULTIPLAYER_TRAINER_SURFACE_SCHEMA_HASH",
    "MULTIPLAYER_TRAINER_SURFACE_SCHEMA_ID",
    "MultiplayerTrainerStepV0",
    "NATIVE_SOURCE_CONTROL_MODEL",
    "SourceStateMultiplayerTrainerSurface",
    "TRAINER_OBSERVATION_SOURCE_RENDERER_BACKED",
    "TRAINER_OBSERVATION_SOURCE",
    "TRAINER_STACK_BACKEND_CPU_DIRTY_CACHE",
    "TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE",
]
