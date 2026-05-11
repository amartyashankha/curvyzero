"""No-train stacked debug-visual survival adapter for CurvyTron LightZero plumbing.

This adapter is deliberately narrow: it wraps the existing debug occupancy
visual env, keeps the observation debug-fidelity only, stacks four frames inside
the wrapper, and changes reward to survival time. It does not prove LightZero
env-manager frame stacking, MCTS/search, replay integration, or learner updates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from curvyzero.training.curvytron_visual_observation import (
    DEBUG_OCCUPANCY_GRAY64_LABEL,
    DEBUG_OCCUPANCY_GRAY64_OBSERVATION_DTYPE,
    DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_LABEL,
    DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_SCHEMA_HASH,
    DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID,
    DEBUG_OCCUPANCY_GRAY64_SHAPE,
    DEBUG_OCCUPANCY_GRAY64_SOURCE_FIDELITY_LEVEL,
    DEBUG_OCCUPANCY_GRAY64_SOURCE_VISUAL_FIDELITY,
    DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE,
    DEBUG_OCCUPANCY_GRAY64_TRUTH_LEVEL,
    DEBUG_OCCUPANCY_GRAY64_USES_ALE,
    DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE,
    DebugOccupancyGray64FrameStack,
)
from curvyzero.training.lightzero_checkpoint_opponent_provider import (
    snapshot_backed_lightzero_checkpoint_opponent_policy,
)
from curvyzero.training.multiplayer_opponent_policy import SnapshotBackedOpponentPolicy
from curvyzero.training.curvyzero_debug_visual_lightzero_smoke import (
    ACTION_ID_TO_SOURCE_MOVE,
    CurvyZeroDebugVisualLightZeroLocalSmokeEnv,
    _copy_lightzero_observation,
)
from curvyzero.training.curvyzero_debug_visual_lightzero_smoke import (
    LocalDebugVisualLightZeroTimestep,
)
from curvyzero.training.curvyzero_debug_visual_lightzero_smoke import (
    optional_base_env_timestep_cls,
)
from curvyzero.training.curvyzero_debug_visual_lightzero_smoke import (
    to_base_env_timestep,
)
from curvyzero.training.curvyzero_survival_time_lightzero_smoke import (
    SURVIVAL_TIME_REWARD_SCHEMA_HASH,
    SURVIVAL_TIME_REWARD_SCHEMA_ID,
)


LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_TYPE = (
    "curvyzero_stacked_debug_visual_survival_lightzero_local_smoke"
)
LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_ID = (
    "CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmoke-v0"
)
LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_TYPE = (
    "curvyzero_stacked_debug_visual_turn_commit_lightzero_local_smoke"
)
LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_ID = (
    "CurvyZeroStackedDebugVisualTurnCommitLightZeroLocalSmoke-v0"
)
STACKED_DEBUG_VISUAL_SURVIVAL_ADAPTER_IMPL_ID = (
    "curvyzero_stacked_debug_visual_survival_lightzero_local_smoke_adapter/v0"
)
STACKED_DEBUG_VISUAL_TURN_COMMIT_ADAPTER_IMPL_ID = (
    "curvyzero_stacked_debug_visual_turn_commit_lightzero_local_smoke_adapter/v0"
)
STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_ID = (
    "curvyzero_stacked_debug_occupancy_gray64_player_aware_survival_time/v1"
)
STACKED_DEBUG_VISUAL_SURVIVAL_FRAME_STACK_OWNER = (
    "curvyzero_wrapper_local_debug_frame_stack"
)
OPPONENT_POLICY_KIND_FIXED_STRAIGHT = "fixed_straight"
OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT = "frozen_lightzero_checkpoint"
OPPONENT_POLICY_KINDS = (
    OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
    OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
)
OPPONENT_TRAINING_RELATION_FIXED_STRAIGHT = "learner_vs_fixed_straight"
OPPONENT_TRAINING_RELATION_FROZEN_LIGHTZERO_CHECKPOINT = (
    "learner_vs_frozen_lightzero_checkpoint"
)
TURN_COMMIT_OPPONENT_POLICY_ID = "shared_lightzero_policy_turn_commit"
TURN_COMMIT_OPPONENT_POLICY_KIND = "shared_policy_turn_commit"
TURN_COMMIT_OPPONENT_TRAINING_RELATION = "shared_policy_turn_commit_smoke"
CURRENT_POLICY_SELF_PLAY_CLAIM = False
CURRENT_POLICY_SELF_PLAY_BLOCKER = (
    "LightZero train_muzero calls this env with only the ego action. The live "
    "collector policy and learner weights are outside env.step, so the env cannot "
    "ask the current policy for the opponent action without a larger collector or "
    "two-seat env change."
)
TURN_COMMIT_CURRENT_POLICY_SELF_PLAY_CLAIM = True
TURN_COMMIT_TRUSTED_SELF_PLAY_CLAIM = False
TURN_COMMIT_SIMULTANEOUS_GAME_THEORY_CLAIM = False
TURN_COMMIT_REWARD_CREDIT_CAVEAT = (
    "Native LightZero sees one scalar reward per env.step. The private turn-commit "
    "adapter advances the physical env only after both players commit, so the "
    "second commit receives the physical-step reward and the first commit receives "
    "a zero bookkeeping reward. Use this path first for native-stack profiling and "
    "smoke training, not as final simultaneous-game credit assignment."
)
_STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_PAYLOAD: dict[str, Any] = {
    "schema_id": STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_ID,
    "raw_observation_schema_id": DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_LABEL,
    "raw_observation_schema_hash": DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_SCHEMA_HASH,
    "legacy_anonymous_raw_observation_schema_id": DEBUG_OCCUPANCY_GRAY64_LABEL,
    "renderer_impl_id": DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID,
    "truth_level": DEBUG_OCCUPANCY_GRAY64_TRUTH_LEVEL,
    "source_fidelity_level": DEBUG_OCCUPANCY_GRAY64_SOURCE_FIDELITY_LEVEL,
    "source_fidelity_claim": "none",
    "player_aware": True,
    "controlled_player_id_field": "controlled_player_id",
    "shape": list(DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE),
    "raw_frame_shape": list(DEBUG_OCCUPANCY_GRAY64_SHAPE),
    "dtype": DEBUG_OCCUPANCY_GRAY64_OBSERVATION_DTYPE,
    "range": list(DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE),
    "frame_stack_owner": STACKED_DEBUG_VISUAL_SURVIVAL_FRAME_STACK_OWNER,
    "frame_stack_proof": "wrapper_owned_fifo_stack; not LightZero env-manager stacking",
    "uses_ale": DEBUG_OCCUPANCY_GRAY64_USES_ALE,
    "ale_usage": "none",
}
STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_HASH = hashlib.sha256(
    json.dumps(
        _STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_PAYLOAD,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()[:16]


class CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv(
    CurvyZeroDebugVisualLightZeroLocalSmokeEnv
):
    """Debug visual LightZero env with wrapper-owned [4,64,64] survival observations."""

    config = dict(CurvyZeroDebugVisualLightZeroLocalSmokeEnv.config)
    config.update(
        {
            "env_id": LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_ID,
            "lightzero_env_type": LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_TYPE,
            "reward_schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
            "observation_schema_id": STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_ID,
            "frame_stack_owner": STACKED_DEBUG_VISUAL_SURVIVAL_FRAME_STACK_OWNER,
        }
    )

    def __init__(self, cfg: Any | None = None):
        super().__init__(cfg)
        self.env_id = str(
            _cfg_get(cfg or {}, "env_id", LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_ID)
        )
        telemetry_path = _cfg_get(cfg or {}, "telemetry_path", None)
        self._telemetry_path = Path(str(telemetry_path)) if telemetry_path else None
        self._frame_stack = DebugOccupancyGray64FrameStack()
        self._opponent_frame_stack = DebugOccupancyGray64FrameStack()
        self._opponent_policy_kind = str(
            _cfg_get(cfg or {}, "opponent_policy_kind", OPPONENT_POLICY_KIND_FIXED_STRAIGHT)
        )
        if self._opponent_policy_kind not in OPPONENT_POLICY_KINDS:
            raise ValueError(
                "opponent_policy_kind must be one of "
                f"{OPPONENT_POLICY_KINDS!r}, got {self._opponent_policy_kind!r}"
            )
        self._last_opponent_policy_sidecar: dict[str, Any] | None = None
        if self._opponent_policy_kind == OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT:
            self.opponent_policy = _build_frozen_lightzero_opponent_policy(cfg or {})
        self._observation_space = {
            "type": "Box",
            "shape": DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE,
            "dtype": "float32",
            "low": 0.0,
            "high": 1.0,
        }
        self._reward_space = {"type": "Box", "shape": (), "dtype": "float32"}

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._frame_stack = DebugOccupancyGray64FrameStack()
        self._opponent_frame_stack = DebugOccupancyGray64FrameStack()
        self._last_opponent_policy_sidecar = None
        return super().reset(seed=seed)

    def step(self, action: Any) -> LocalDebugVisualLightZeroTimestep:
        timestep = super().step(action)
        self._write_telemetry_row(action=action, timestep=timestep)
        return timestep

    def _lightzero_observation(
        self,
        *,
        snapshot: Mapping[str, Any],
        needs_reset: bool,
    ) -> dict[str, Any]:
        frame = self._render_normalized(
            snapshot,
            controlled_player_id=self.ego_player_id,
        )
        stack = self._frame_stack.update(frame, copy=True)
        self._update_opponent_frame_stack(snapshot)
        ego_alive = self._player_alive(snapshot, self.ego_player_id)
        return {
            "observation": stack,
            "action_mask": self._action_mask(active=ego_alive and not needs_reset),
            "to_play": -1,
            "timestep": int(self._step_index),
        }

    def _base_info(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        info = super()._base_info(snapshot)
        info.update(
            {
                "env_id": self.env_id,
                "lightzero_env_type": LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_TYPE,
                "adapter_impl_id": STACKED_DEBUG_VISUAL_SURVIVAL_ADAPTER_IMPL_ID,
                "lightzero_adapter_kind": "stacked_debug_visual_survival_no_train_smoke",
                "schema_hash": STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_HASH,
                "observation_schema_id": STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_ID,
                "observation_schema_hash": STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_HASH,
                "raw_observation_schema_id": DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_LABEL,
                "raw_observation_schema_hash": DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_SCHEMA_HASH,
                "legacy_anonymous_raw_observation_schema_id": DEBUG_OCCUPANCY_GRAY64_LABEL,
                "player_aware": True,
                "ego_controlled_player_id": self.ego_player_id,
                "opponent_controlled_player_id": self.opponent_player_id,
                "renderer_impl_id": DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID,
                "truth_level": DEBUG_OCCUPANCY_GRAY64_TRUTH_LEVEL,
                "source_fidelity_level": DEBUG_OCCUPANCY_GRAY64_SOURCE_FIDELITY_LEVEL,
                "shape": list(DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE),
                "dtype": DEBUG_OCCUPANCY_GRAY64_OBSERVATION_DTYPE,
                "range": list(DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE),
                "value_range": list(DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE),
                "raw_frame_shape": list(DEBUG_OCCUPANCY_GRAY64_SHAPE),
                "lightzero_payload_shape": list(DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE),
                "model_observation_shape": list(DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE),
                "frame_stack_owner": STACKED_DEBUG_VISUAL_SURVIVAL_FRAME_STACK_OWNER,
                "frame_stack_proof": "wrapper_owned_fifo_stack; not LightZero env-manager stacking",
                "reward_schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
                "reward_schema_hash": SURVIVAL_TIME_REWARD_SCHEMA_HASH,
                "survival_terminal_step_counting_rule": "post_transition_alive",
                "terminal_outcome_bonus": 0.0,
                "loser_penalty": 0.0,
                "winner_bonus": 0.0,
                "opponent_training_relation": _opponent_training_relation(
                    self._opponent_policy_kind
                ),
                "current_policy_self_play": CURRENT_POLICY_SELF_PLAY_CLAIM,
                "current_policy_self_play_blocker": CURRENT_POLICY_SELF_PLAY_BLOCKER,
                "debug_fidelity_only": True,
                "source_fidelity_claim": "none",
                "source_visual_fidelity": DEBUG_OCCUPANCY_GRAY64_SOURCE_VISUAL_FIDELITY,
                "source_backed_observation_fidelity": DEBUG_OCCUPANCY_GRAY64_SOURCE_VISUAL_FIDELITY,
                "browser_pixel_fidelity": False,
                "uses_ale": DEBUG_OCCUPANCY_GRAY64_USES_ALE,
                "ale_usage": "none",
            }
        )
        return info

    def _reward(self, snapshot: Mapping[str, Any], *, terminated: bool) -> float:
        del terminated
        return 1.0 if self._player_alive(snapshot, self.ego_player_id) else 0.0

    def _reward_map(self, snapshot: Mapping[str, Any], *, terminated: bool) -> dict[str, float]:
        del terminated
        return {
            player_id: 1.0 if self._player_alive(snapshot, player_id) else 0.0
            for player_id in self.player_ids
        }

    def _opponent_action(
        self,
        *,
        snapshot: Mapping[str, Any],
        legal_action_mask: np.ndarray,
    ) -> int:
        """Select the source opponent action, optionally from a frozen checkpoint."""

        if self._opponent_policy_kind == OPPONENT_POLICY_KIND_FIXED_STRAIGHT:
            self._last_opponent_policy_sidecar = None
            return super()._opponent_action(
                snapshot=snapshot,
                legal_action_mask=legal_action_mask,
            )
        if not isinstance(self.opponent_policy, SnapshotBackedOpponentPolicy):
            raise TypeError("frozen LightZero opponent policy was not initialized")
        ego_stack = self._frame_stack.stack
        opponent_stack = self._opponent_frame_stack.stack
        if ego_stack is None:
            raise RuntimeError("ego frame stack is not initialized")
        if opponent_stack is None:
            raise RuntimeError("opponent frame stack is not initialized")
        legal = np.ones((1, 2, 3), dtype=bool)
        legal[0, 1] = np.asarray(legal_action_mask, dtype=bool)
        opponent_mask = np.array([[False, True]], dtype=bool)
        observation = np.zeros((1, 2, *DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE), dtype=np.float32)
        observation[0, 0] = ego_stack
        observation[0, 1] = opponent_stack
        selection = self.opponent_policy.select_actions(
            legal,
            opponent_mask,
            decision_index=int(self._step_index),
            observation=observation,
        )
        self._last_opponent_policy_sidecar = selection.sidecar()
        return int(selection.actions[0, 1])

    def _step_info(
        self,
        *,
        snapshot: Mapping[str, Any],
        joint_action: Mapping[str, int],
        joint_source_action: Mapping[int, float],
        reward: float,
        done: bool,
        terminated: bool,
        truncated: bool,
        next_obs: dict[str, Any],
    ) -> dict[str, Any]:
        info = super()._step_info(
            snapshot=snapshot,
            joint_action=joint_action,
            joint_source_action=joint_source_action,
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            next_obs=next_obs,
        )
        info["opponent_policy_kind"] = self._opponent_policy_kind
        info["opponent_training_relation"] = _opponent_training_relation(
            self._opponent_policy_kind
        )
        info["current_policy_self_play"] = CURRENT_POLICY_SELF_PLAY_CLAIM
        info["current_policy_self_play_blocker"] = CURRENT_POLICY_SELF_PLAY_BLOCKER
        info["player_aware"] = True
        info["ego_controlled_player_id"] = self.ego_player_id
        info["opponent_controlled_player_id"] = self.opponent_player_id
        if self._last_opponent_policy_sidecar is not None:
            info["opponent_policy_sidecar"] = self._last_opponent_policy_sidecar
        return info

    def _update_opponent_frame_stack(self, snapshot: Mapping[str, Any]) -> None:
        frame = self._render_normalized(
            snapshot,
            controlled_player_id=self.opponent_player_id,
        )
        self._opponent_frame_stack.update(frame, copy=False)

    def _write_telemetry_row(
        self,
        *,
        action: Any,
        timestep: LocalDebugVisualLightZeroTimestep,
    ) -> None:
        if self._telemetry_path is None:
            return
        info = timestep.info
        opponent_policy_sidecar = info.get("opponent_policy_sidecar")
        opponent_policy_metadata = _opponent_policy_metadata(opponent_policy_sidecar)
        row = {
            "schema_id": "curvyzero_stacked_debug_visual_survival_env_step/v0",
            "episode_index": int(info.get("episode_index", self._episode_index - 1)),
            "step_index": int(info.get("step_index", self._step_index - 1)),
            "ego_action": _to_int_or_repr(action),
            "scalar_action": _to_int_or_repr(action),
            "acting_player_id": info.get("acting_player_id"),
            "controlled_player_id": info.get("controlled_player_id"),
            "active_player_id": info.get("active_player_id"),
            "next_active_player_id": info.get("next_active_player_id"),
            "physical_env_advanced": bool(info.get("physical_env_advanced", True)),
            "pending_action_count": info.get("pending_action_count"),
            "joint_action": info.get("joint_action"),
            "opponent_action_id": info.get("opponent_action_id"),
            "opponent_policy_id": info.get("opponent_policy_id"),
            "opponent_policy_version": info.get("opponent_policy_version"),
            "opponent_policy_kind": info.get("opponent_policy_kind"),
            "opponent_training_relation": info.get("opponent_training_relation"),
            "shared_policy_turn_commit": bool(info.get("shared_policy_turn_commit", False)),
            "current_policy_self_play": bool(
                info.get("current_policy_self_play", CURRENT_POLICY_SELF_PLAY_CLAIM)
            ),
            "current_policy_self_play_blocker": info.get("current_policy_self_play_blocker"),
            "current_policy_self_play_caveat": info.get("current_policy_self_play_caveat"),
            "trusted_current_policy_self_play": bool(
                info.get("trusted_current_policy_self_play", False)
            ),
            "simultaneous_game_theory_claim": bool(
                info.get("simultaneous_game_theory_claim", False)
            ),
            "opponent_snapshot_ref": opponent_policy_metadata.get("snapshot_ref"),
            "opponent_checkpoint_ref": opponent_policy_metadata.get("checkpoint_ref"),
            "opponent_model_id": opponent_policy_metadata.get("model_id"),
            "opponent_provider_id": opponent_policy_metadata.get("provider_id"),
            "opponent_provider_version": opponent_policy_metadata.get("provider_version"),
            "reward": float(timestep.reward),
            "reward_player_id": info.get("reward_player_id"),
            "reward_perspective": info.get("reward_perspective"),
            "done": bool(timestep.done),
            "terminated": bool(info.get("terminated", False)),
            "truncated": bool(info.get("truncated", False)),
            "terminal_reason": info.get("terminal_reason"),
            "winner_ids": info.get("winner_ids"),
            "loser_ids": info.get("loser_ids"),
            "death_player_ids": info.get("death_player_ids"),
            "death_count": info.get("death_count"),
            "death_player": info.get("death_player"),
            "death_cause": info.get("death_cause"),
            "death_cause_name": info.get("death_cause_name"),
            "death_hit_owner": info.get("death_hit_owner"),
            "death_source_avatar": info.get("death_source_avatar"),
            "death_source_killer_avatar": info.get("death_source_killer_avatar"),
            "reward_schema_id": info.get("reward_schema_id"),
            "observation_schema_id": info.get("observation_schema_id"),
            "player_aware": bool(info.get("player_aware", True)),
            "ego_controlled_player_id": info.get("ego_controlled_player_id"),
            "opponent_controlled_player_id": info.get("opponent_controlled_player_id"),
            "frame_stack_owner": info.get("frame_stack_owner"),
            "trace_hash": info.get("trace_hash"),
        }
        self._telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with self._telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    def __repr__(self) -> str:
        return (
            "CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv("
            f"env_id={self.env_id!r}, "
            f"ego_player_id={self.ego_player_id!r}, "
            f"opponent_player_id={self.opponent_player_id!r}, "
            f"opponent_policy_id={self.opponent_policy.policy_id!r})"
        )


class CurvyZeroStackedDebugVisualTurnCommitLightZeroLocalSmokeEnv(
    CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv
):
    """Native LightZero turn-commit adapter for two-player CurvyTron.

    LightZero calls ``step`` with one scalar action. This wrapper alternates the
    controlled player, stores the first player's pending action privately, and
    advances the physical CurvyTron env only after player two commits.
    """

    config = dict(CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv.config)
    config.update(
        {
            "env_id": LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_ID,
            "lightzero_env_type": LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_TYPE,
            "adapter_impl_id": STACKED_DEBUG_VISUAL_TURN_COMMIT_ADAPTER_IMPL_ID,
            "turn_commit_adapter": True,
            "current_policy_self_play": TURN_COMMIT_CURRENT_POLICY_SELF_PLAY_CLAIM,
        }
    )

    def __init__(self, cfg: Any | None = None):
        super().__init__(cfg)
        self.env_id = str(
            _cfg_get(cfg or {}, "env_id", LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_ID)
        )
        self._turn_player_ids = ("player_0", "player_1")
        self._active_player_index = 0
        self._pending_actions: dict[str, int] = {}
        self._last_pending_sidecar: dict[str, Any] | None = None

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._active_player_index = 0
        self._pending_actions = {}
        self._last_pending_sidecar = None
        return super().reset(seed=seed)

    @property
    def active_player_id(self) -> str:
        return self._turn_player_ids[self._active_player_index]

    def step(self, action: Any) -> LocalDebugVisualLightZeroTimestep:
        if self._last_snapshot is None:
            raise RuntimeError("reset must be called before step")
        if self._needs_reset:
            raise RuntimeError(
                "reset must be called before stepping after a terminal or truncated episode"
            )

        acting_player = self.active_player_id
        action_id = self._validate_ego_action(action)
        if self._active_player_index == 0:
            timestep = self._record_pending_turn_action(
                player_id=acting_player,
                action_id=action_id,
            )
        else:
            timestep = self._commit_joint_turn_action(
                player_id=acting_player,
                action_id=action_id,
            )
        self._write_telemetry_row(action=action, timestep=timestep)
        return timestep

    def render(self, mode: str = "debug_visual_tensor") -> np.ndarray | None:
        if mode != "debug_visual_tensor":
            return None
        if self._last_snapshot is None:
            return None
        return self._render_normalized(
            self._last_snapshot,
            controlled_player_id=self.active_player_id,
        ).copy()

    def _record_pending_turn_action(
        self,
        *,
        player_id: str,
        action_id: int,
    ) -> LocalDebugVisualLightZeroTimestep:
        self._pending_actions[player_id] = int(action_id)
        self._active_player_index = 1
        next_obs = self._lightzero_observation(
            snapshot=self._last_snapshot,
            needs_reset=False,
        )
        info = self._turn_commit_info(
            snapshot=self._last_snapshot,
            action_id=action_id,
            acting_player_id=player_id,
            physical_env_advanced=False,
            reward=0.0,
            done=False,
            terminated=False,
            truncated=False,
            next_obs=next_obs,
        )
        self._last_pending_sidecar = dict(info)
        return LocalDebugVisualLightZeroTimestep(next_obs, 0.0, False, info)

    def _commit_joint_turn_action(
        self,
        *,
        player_id: str,
        action_id: int,
    ) -> LocalDebugVisualLightZeroTimestep:
        self._pending_actions[player_id] = int(action_id)
        joint_action = {
            player: int(self._pending_actions[player])
            for player in self._turn_player_ids
        }
        joint_source_action = {
            self._avatar_ids_by_player[player]: _turn_action_id_to_source_move(action)
            for player, action in joint_action.items()
        }

        self._env.advance_timers(0.0)
        snapshot = self._env.step(joint_source_action, elapsed_ms=self._source_step_ms)
        self._last_snapshot = dict(snapshot)
        terminated = self._terminated(snapshot)
        truncated = self._truncated()
        done = bool(terminated or truncated)
        self._needs_reset = done
        reward = self._survival_reward_for_player(snapshot, player_id=player_id)
        self._episode_return += reward
        self._step_index += 1
        self._active_player_index = 0
        self._pending_actions = {}

        next_obs = self._lightzero_observation(snapshot=snapshot, needs_reset=done)
        self._action_trace.append(
            {
                "step_index": self._step_index - 1,
                "source_at_ms": snapshot.get("atMs"),
                "joint_action": dict(joint_action),
                "joint_source_move": {
                    player: float(joint_source_action[self._avatar_ids_by_player[player]])
                    for player in joint_action
                },
                "reward": reward,
                "reward_player_id": player_id,
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
                "turn_commit_adapter": True,
            }
        )
        info = self._step_info(
            snapshot=snapshot,
            joint_action=joint_action,
            joint_source_action=joint_source_action,
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            next_obs=next_obs,
        )
        info.update(
            self._turn_commit_metadata(
                acting_player_id=player_id,
                physical_env_advanced=True,
                pending_action_count=0,
            )
        )
        info["opponent_action_id"] = None
        info["reward_player_id"] = player_id
        info["reward_credit_caveat"] = TURN_COMMIT_REWARD_CREDIT_CAVEAT
        return LocalDebugVisualLightZeroTimestep(next_obs, reward, done, info)

    def _lightzero_observation(
        self,
        *,
        snapshot: Mapping[str, Any],
        needs_reset: bool,
    ) -> dict[str, Any]:
        controlled_player_id = self.active_player_id
        frame = self._render_normalized(
            snapshot,
            controlled_player_id=controlled_player_id,
        )
        stack = self._frame_stack_for_player(controlled_player_id).update(
            frame,
            copy=True,
        )
        active_alive = self._player_alive(snapshot, controlled_player_id)
        return {
            "observation": stack,
            "action_mask": self._action_mask(active=active_alive and not needs_reset),
            "to_play": -1,
            "timestep": int(self._step_index),
        }

    def _base_info(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        info = super()._base_info(snapshot)
        info.update(
            {
                "env_id": self.env_id,
                "lightzero_env_type": LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_TYPE,
                "adapter_impl_id": STACKED_DEBUG_VISUAL_TURN_COMMIT_ADAPTER_IMPL_ID,
                "lightzero_adapter_kind": "stacked_debug_visual_turn_commit_native_train_muzero",
                "turn_commit_adapter": True,
                "turn_commit_rule": "physical_env_advances_only_after_all_players_commit",
                "current_policy_self_play": TURN_COMMIT_CURRENT_POLICY_SELF_PLAY_CLAIM,
                "current_policy_self_play_blocker": None,
                "current_policy_self_play_caveat": TURN_COMMIT_REWARD_CREDIT_CAVEAT,
                "trusted_current_policy_self_play": TURN_COMMIT_TRUSTED_SELF_PLAY_CLAIM,
                "simultaneous_game_theory_claim": (
                    TURN_COMMIT_SIMULTANEOUS_GAME_THEORY_CLAIM
                ),
                "active_player_id": self.active_player_id,
                "active_player_index": int(self._active_player_index),
                "pending_action_count": int(len(self._pending_actions)),
            }
        )
        return info

    def _turn_commit_info(
        self,
        *,
        snapshot: Mapping[str, Any],
        action_id: int,
        acting_player_id: str,
        physical_env_advanced: bool,
        reward: float,
        done: bool,
        terminated: bool,
        truncated: bool,
        next_obs: dict[str, Any],
    ) -> dict[str, Any]:
        info = self._base_info(snapshot)
        info.update(
            {
                "step_index": self._step_index,
                "tick_index": self._step_index,
                "adapter_timestep": self._step_index,
                "acting_player_id": acting_player_id,
                "committed_action_id": int(action_id),
                "joint_action": dict(self._pending_actions),
                "joint_source_move": {
                    player: _turn_action_id_to_source_move(action)
                    for player, action in self._pending_actions.items()
                },
                "opponent_action_id": None,
                "opponent_policy_id": TURN_COMMIT_OPPONENT_POLICY_ID,
                "opponent_policy_kind": TURN_COMMIT_OPPONENT_POLICY_KIND,
                "opponent_training_relation": TURN_COMMIT_OPPONENT_TRAINING_RELATION,
                "opponent_policy_version": "current",
                "physical_env_advanced": bool(physical_env_advanced),
                "terminal_reason": "none",
                "done": bool(done),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "needs_reset": self._needs_reset,
                "final_observation": _copy_lightzero_observation(next_obs) if done else None,
                "final_reward_map": None,
                "reward": float(reward),
                "reward_player_id": acting_player_id,
                "reward_credit_caveat": TURN_COMMIT_REWARD_CREDIT_CAVEAT,
                "trace_hash": self._trace_hash(),
            }
        )
        info.update(
            self._turn_commit_metadata(
                acting_player_id=acting_player_id,
                physical_env_advanced=physical_env_advanced,
                pending_action_count=len(self._pending_actions),
            )
        )
        return info

    def _turn_commit_metadata(
        self,
        *,
        acting_player_id: str,
        physical_env_advanced: bool,
        pending_action_count: int,
    ) -> dict[str, Any]:
        return {
            "turn_commit_adapter": True,
            "turn_commit_rule": "physical_env_advances_only_after_all_players_commit",
            "acting_player_id": acting_player_id,
            "controlled_player_id": acting_player_id,
            "next_active_player_id": self.active_player_id,
            "active_player_id": self.active_player_id,
            "active_player_index": int(self._active_player_index),
            "pending_action_count": int(pending_action_count),
            "physical_env_advanced": bool(physical_env_advanced),
            "pending_actions_private": True,
            "shared_policy_turn_commit": True,
            "ego_controlled_player_id": acting_player_id,
            "opponent_controlled_player_id": self.active_player_id,
            "opponent_policy_id": TURN_COMMIT_OPPONENT_POLICY_ID,
            "opponent_policy_kind": TURN_COMMIT_OPPONENT_POLICY_KIND,
            "opponent_training_relation": TURN_COMMIT_OPPONENT_TRAINING_RELATION,
            "opponent_policy_version": "current",
            "current_policy_self_play": TURN_COMMIT_CURRENT_POLICY_SELF_PLAY_CLAIM,
            "current_policy_self_play_blocker": None,
            "current_policy_self_play_caveat": TURN_COMMIT_REWARD_CREDIT_CAVEAT,
            "trusted_current_policy_self_play": TURN_COMMIT_TRUSTED_SELF_PLAY_CLAIM,
            "simultaneous_game_theory_claim": (
                TURN_COMMIT_SIMULTANEOUS_GAME_THEORY_CLAIM
            ),
            "source_ticks_per_commit": 1,
            "scalar_steps_per_source_tick": len(self._turn_player_ids),
            "reward_perspective": "just_controlled_player_after_commit",
        }

    def _frame_stack_for_player(self, player_id: str) -> DebugOccupancyGray64FrameStack:
        if player_id == self.ego_player_id:
            return self._frame_stack
        if player_id == self.opponent_player_id:
            return self._opponent_frame_stack
        raise ValueError(f"unknown player_id: {player_id!r}")

    def _survival_reward_for_player(
        self,
        snapshot: Mapping[str, Any],
        *,
        player_id: str,
    ) -> float:
        return 1.0 if self._player_alive(snapshot, player_id) else 0.0

    def __repr__(self) -> str:
        return (
            "CurvyZeroStackedDebugVisualTurnCommitLightZeroLocalSmokeEnv("
            f"env_id={self.env_id!r}, "
            f"active_player_id={self.active_player_id!r})"
        )


def run_stacked_debug_visual_survival_smoke(
    *,
    seed: int = 0,
    steps: int = 3,
) -> dict[str, Any]:
    """Run a tiny fixed-action collect/replay/sample smoke without MCTS or training."""

    started = time.perf_counter()
    env = CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv(
        {"seed": seed, "source_max_steps": max(steps + 1, 2)}
    )
    reset_obs = env.reset(seed=seed)
    rows: list[dict[str, Any]] = []
    problems = _validate_observation(reset_obs, label="reset")
    last_timestep: LocalDebugVisualLightZeroTimestep | None = None
    for step_index in range(int(steps)):
        timestep = env.step(1)
        last_timestep = timestep
        problems.extend(_validate_observation(timestep.obs, label=f"step_{step_index}.obs"))
        if timestep.info.get("reward_schema_id") != SURVIVAL_TIME_REWARD_SCHEMA_ID:
            problems.append(f"step_{step_index} did not report survival reward schema")
        rows.append(
            {
                "step_index": step_index,
                "observation_shape": list(np.asarray(timestep.obs["observation"]).shape),
                "action": 1,
                "reward": float(timestep.reward),
                "done": bool(timestep.done),
                "reward_schema_id": timestep.info.get("reward_schema_id"),
                "observation_schema_id": timestep.info.get("observation_schema_id"),
                "frame_stack_owner": timestep.info.get("frame_stack_owner"),
            }
        )
        if timestep.done:
            break

    if not rows:
        problems.append("no replay rows collected")
    sample_row = rows[0] if rows else None
    mcts_search = {
        "status": "not_run",
        "reason": (
            "requires installed LightZero MuZeroPolicy/eval-mode search wired to "
            "the stacked conv observation; this smoke only proves fixed-action "
            "visual collect, replay row packaging, and sampling shape"
        ),
    }
    learner_profile = {
        "status": "not_run",
        "reason": "no learner update or train_muzero call is allowed in this smoke",
    }
    return {
        "ok": not problems,
        "mode": "no_train_stacked_debug_visual_survival_collect_replay_sample_only",
        "call_policy": "does_not_train; does_not_call_lzero_entrypoints",
        "problems": problems,
        "env_type": LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_TYPE,
        "env_id": LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_ID,
        "observation_schema_id": STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_ID,
        "observation_schema_hash": STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_HASH,
        "raw_observation_schema_id": DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_LABEL,
        "raw_observation_schema_hash": DEBUG_OCCUPANCY_GRAY64_PLAYER_AWARE_SCHEMA_HASH,
        "legacy_anonymous_raw_observation_schema_id": DEBUG_OCCUPANCY_GRAY64_LABEL,
        "player_aware": True,
        "renderer_impl_id": DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID,
        "truth_level": DEBUG_OCCUPANCY_GRAY64_TRUTH_LEVEL,
        "source_fidelity_level": DEBUG_OCCUPANCY_GRAY64_SOURCE_FIDELITY_LEVEL,
        "observation_shape": list(DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE),
        "observation_dtype": DEBUG_OCCUPANCY_GRAY64_OBSERVATION_DTYPE,
        "value_range": list(DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE),
        "raw_frame_shape": list(DEBUG_OCCUPANCY_GRAY64_SHAPE),
        "frame_stack_owner": STACKED_DEBUG_VISUAL_SURVIVAL_FRAME_STACK_OWNER,
        "uses_ale": DEBUG_OCCUPANCY_GRAY64_USES_ALE,
        "ale_usage": "none",
        "reward_schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
        "reward_policy": {
            "survival_only": True,
            "terminal_outcome_bonus": 0.0,
            "loser_penalty": 0.0,
            "winner_bonus": 0.0,
        },
        "opponent_training_relation": _opponent_training_relation(
            OPPONENT_POLICY_KIND_FIXED_STRAIGHT
        ),
        "current_policy_self_play": CURRENT_POLICY_SELF_PLAY_CLAIM,
        "current_policy_self_play_blocker": CURRENT_POLICY_SELF_PLAY_BLOCKER,
        "reset": _summarize_observation(reset_obs),
        "collected_rows": rows,
        "sample_row": sample_row,
        "last_timestep": _summarize_timestep(last_timestep) if last_timestep else None,
        "mcts_search": mcts_search,
        "learner_profile": learner_profile,
        "elapsed_sec": round(time.perf_counter() - started, 6),
    }


def _validate_observation(obs: Mapping[str, Any], *, label: str) -> list[str]:
    problems: list[str] = []
    observation = np.asarray(obs.get("observation"))
    action_mask = np.asarray(obs.get("action_mask"))
    if observation.shape != DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE:
        problems.append(
            f"{label} observation shape {observation.shape!r}, "
            f"expected {DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE!r}"
        )
    if observation.dtype != np.float32:
        problems.append(f"{label} observation dtype {observation.dtype}, expected float32")
    if action_mask.shape != (3,):
        problems.append(f"{label} action_mask shape {action_mask.shape!r}, expected (3,)")
    if action_mask.dtype != np.int8:
        problems.append(f"{label} action_mask dtype {action_mask.dtype}, expected int8")
    if obs.get("to_play") != -1:
        problems.append(f"{label} to_play {obs.get('to_play')!r}, expected -1")
    return problems


def _summarize_observation(obs: Mapping[str, Any]) -> dict[str, Any]:
    observation = np.asarray(obs.get("observation"))
    action_mask = np.asarray(obs.get("action_mask"))
    return {
        "keys": sorted(str(key) for key in obs),
        "observation_shape": [int(item) for item in observation.shape],
        "observation_dtype": str(observation.dtype),
        "observation_min": float(observation.min()) if observation.size else None,
        "observation_max": float(observation.max()) if observation.size else None,
        "action_mask_shape": [int(item) for item in action_mask.shape],
        "action_mask_dtype": str(action_mask.dtype),
        "action_mask_values": action_mask.tolist(),
        "to_play": int(obs.get("to_play")),
        "timestep": int(obs.get("timestep")),
    }


def _summarize_timestep(
    timestep: LocalDebugVisualLightZeroTimestep | None,
) -> dict[str, Any] | None:
    if timestep is None:
        return None
    return {
        "reward": float(timestep.reward),
        "done": bool(timestep.done),
        "obs": _summarize_observation(timestep.obs),
        "info": {
            "reward_schema_id": timestep.info.get("reward_schema_id"),
            "observation_schema_id": timestep.info.get("observation_schema_id"),
            "frame_stack_owner": timestep.info.get("frame_stack_owner"),
            "terminal_reason": timestep.info.get("terminal_reason"),
            "eval_episode_return": timestep.info.get("eval_episode_return"),
        },
    }


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _turn_action_id_to_source_move(action_id: int) -> float:
    return float(ACTION_ID_TO_SOURCE_MOVE[int(action_id)])


def _build_frozen_lightzero_opponent_policy(cfg: Any) -> SnapshotBackedOpponentPolicy:
    checkpoint_path = _cfg_get(cfg, "opponent_checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError(
            "opponent_checkpoint_path is required for frozen_lightzero_checkpoint"
        )
    snapshot_ref = str(
        _cfg_get(cfg, "opponent_snapshot_ref", "curvytron_visual_survival_frozen_opponent")
    )
    checkpoint_ref = _cfg_get(cfg, "opponent_checkpoint_ref", None)
    state_key = _cfg_get(cfg, "opponent_checkpoint_state_key", None)
    return snapshot_backed_lightzero_checkpoint_opponent_policy(
        checkpoint_path=Path(str(checkpoint_path)),
        snapshot_ref=snapshot_ref,
        checkpoint_ref=None if checkpoint_ref is None else str(checkpoint_ref),
        seed=int(_cfg_get(cfg, "opponent_policy_seed", _cfg_get(cfg, "seed", 0))),
        num_simulations=int(_cfg_get(cfg, "opponent_num_simulations", 8)),
        batch_size=int(_cfg_get(cfg, "opponent_batch_size", 16)),
        use_cuda=bool(_cfg_get(cfg, "opponent_use_cuda", False)),
        state_key=None if state_key is None else str(state_key),
    )


def _opponent_training_relation(opponent_policy_kind: str) -> str:
    if opponent_policy_kind == OPPONENT_POLICY_KIND_FIXED_STRAIGHT:
        return OPPONENT_TRAINING_RELATION_FIXED_STRAIGHT
    if opponent_policy_kind == OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT:
        return OPPONENT_TRAINING_RELATION_FROZEN_LIGHTZERO_CHECKPOINT
    return f"unknown:{opponent_policy_kind}"


def _opponent_policy_metadata(sidecar: Any) -> dict[str, Any]:
    if not isinstance(sidecar, Mapping):
        return {}
    metadata = sidecar.get("policy_metadata")
    if not isinstance(metadata, Mapping):
        return {}
    return dict(metadata)


def _to_int_or_repr(value: Any) -> int | str:
    try:
        return int(np.asarray(value).item())
    except Exception:
        return repr(value)


def main() -> None:
    result = run_stacked_debug_visual_survival_smoke()
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv",
    "CurvyZeroStackedDebugVisualTurnCommitLightZeroLocalSmokeEnv",
    "LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_ID",
    "LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_TYPE",
    "LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_ID",
    "LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_TYPE",
    "LocalDebugVisualLightZeroTimestep",
    "STACKED_DEBUG_VISUAL_SURVIVAL_ADAPTER_IMPL_ID",
    "STACKED_DEBUG_VISUAL_SURVIVAL_FRAME_STACK_OWNER",
    "STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_HASH",
    "STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_ID",
    "STACKED_DEBUG_VISUAL_TURN_COMMIT_ADAPTER_IMPL_ID",
    "CURRENT_POLICY_SELF_PLAY_BLOCKER",
    "CURRENT_POLICY_SELF_PLAY_CLAIM",
    "TURN_COMMIT_CURRENT_POLICY_SELF_PLAY_CLAIM",
    "TURN_COMMIT_OPPONENT_POLICY_ID",
    "TURN_COMMIT_OPPONENT_POLICY_KIND",
    "TURN_COMMIT_OPPONENT_TRAINING_RELATION",
    "TURN_COMMIT_REWARD_CREDIT_CAVEAT",
    "TURN_COMMIT_SIMULTANEOUS_GAME_THEORY_CLAIM",
    "TURN_COMMIT_TRUSTED_SELF_PLAY_CLAIM",
    "OPPONENT_POLICY_KIND_FIXED_STRAIGHT",
    "OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT",
    "OPPONENT_TRAINING_RELATION_FIXED_STRAIGHT",
    "OPPONENT_TRAINING_RELATION_FROZEN_LIGHTZERO_CHECKPOINT",
    "optional_base_env_timestep_cls",
    "run_stacked_debug_visual_survival_smoke",
    "to_base_env_timestep",
]
