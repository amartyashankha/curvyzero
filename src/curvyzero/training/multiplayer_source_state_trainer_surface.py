"""Trainer-facing multiplayer source-state visual surface.

This module is intentionally a thin adapter over ``VectorMultiplayerEnv``. It
does not implement environment mechanics; it swaps the public metadata-only
observation for the existing source-state-backed visual stack used by trainer
smokes.
"""

from __future__ import annotations

from dataclasses import dataclass
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


class SourceStateMultiplayerTrainerSurface:
    """Small source-state visual trainer surface backed by ``VectorMultiplayerEnv``."""

    def __init__(
        self,
        batch_size: int = 1,
        *,
        player_count: int = 2,
        seed: int | None = None,
        trail_render_mode: str = TRAIL_RENDER_MODE_BROWSER_LINES,
        env: VectorMultiplayerEnv | None = None,
        **env_kwargs: Any,
    ) -> None:
        mode = validate_stack_trail_render_mode(trail_render_mode)

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
        self.stack = SourceStateGray64Stack4(
            batch_size=self.batch_size,
            player_count=self.player_count,
            trail_render_mode=self.trail_render_mode,
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
        batch = self.env.step(
            action,
            timer_advance_ms=timer_advance_ms,
            disabled_player_mask=disabled_player_mask,
        )
        observation = self.stack.update(self.env)
        reward = self._survival_plus_bonus_reward(batch.info, batch.done)
        return self._surface_step(
            batch=batch,
            observation=observation,
            reward=reward,
            joint_action=action,
            api="step",
        )

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
        legal_action_mask = np.asarray(batch.action_mask, dtype=bool).copy()
        lightzero_action_mask = legal_action_mask.copy()
        done = np.asarray(batch.done, dtype=bool).copy()
        terminated = np.asarray(batch.terminated, dtype=bool).copy()
        truncated = np.asarray(batch.truncated, dtype=bool).copy()
        live_mask = self._live_mask(batch.info, done, legal_action_mask)
        policy_env_row, policy_player = self._policy_rows(live_mask)
        policy_observation = observation[policy_env_row, policy_player].astype(
            np.float32,
            copy=True,
        )
        policy_action_mask = legal_action_mask[policy_env_row, policy_player].copy()
        final_row_mask = np.asarray(
            batch.info.get("final_observation_row_mask", np.zeros(self.batch_size, dtype=bool)),
            dtype=bool,
        ).copy()
        if final_row_mask.shape != (self.batch_size,):
            final_row_mask = np.zeros(self.batch_size, dtype=bool)

        final_observation = np.zeros_like(observation, dtype=np.float32)
        final_reward_map = np.zeros_like(reward, dtype=np.float32)
        if bool(final_row_mask.any()):
            final_observation[final_row_mask] = observation[final_row_mask]
            final_reward_map[final_row_mask] = reward[final_row_mask]

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
        return MultiplayerTrainerStepV0(
            observation=np.asarray(observation, dtype=np.float32).copy(),
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
            final_observation=final_observation.copy(),
            final_observation_row_mask=final_row_mask,
            final_reward_map=final_reward_map.copy(),
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
                "visual_stack_class": TRAINER_OBSERVATION_SOURCE,
                "trainer_observation_source": TRAINER_OBSERVATION_SOURCE,
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
                "final_observation": final_observation.copy(),
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
                "final_reward_map": final_reward_map.copy(),
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
    "TRAINER_OBSERVATION_SOURCE",
]
