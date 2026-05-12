"""Source-state-backed CurvyTron turn-commit env for LightZero MuZero."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from curvyzero.env import vector_runtime
from curvyzero.env.vector_multiplayer_env import ACTION_COUNT
from curvyzero.env.vector_multiplayer_env import DEFAULT_DECISION_SOURCE_FRAMES
from curvyzero.env.vector_multiplayer_env import DEFAULT_SOURCE_FRAME_DECISION_MS
from curvyzero.env.vector_multiplayer_env import JOINT_ACTION_SCHEMA_ID
from curvyzero.env.vector_multiplayer_env import NATURAL_BONUS_ENV_IMPL_ID
from curvyzero.env.vector_multiplayer_env import NATURAL_BONUS_RULESET_ID
from curvyzero.env.vector_multiplayer_env import PUBLIC_NATURAL_BONUS_ENV_CONTRACT_ID
from curvyzero.env.vector_multiplayer_env import SOURCE_PHYSICS_STEP_MS
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.env.trainer_contract import stable_contract_hash
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY,
)
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_DTYPE,
)
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_VALUE_RANGE,
)
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_CANVAS_GRAY64_RENDERER_IMPL_ID,
)
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SHAPE
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_CANVAS_GRAY64_SOURCE_FIDELITY_LEVEL,
)
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_CANVAS_GRAY64_SOURCE_STATE_BACKED,
)
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SURFACE
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_TRUTH_LEVEL
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_USES_ALE
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
)
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB,
)
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_RGB_CANVAS_LIKE_RENDERER_IMPL_ID,
)
from curvyzero.env.vector_visual_observation import SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL,
)
from curvyzero.env.vector_visual_observation import TRAIL_RENDER_MODE_DEFAULT
from curvyzero.env.vector_visual_observation import TRAIL_RENDER_MODE_ORDER
from curvyzero.env.vector_visual_observation import render_source_state_canvas_gray64
from curvyzero.env.vector_visual_observation import render_source_state_rgb_canvas_like
from curvyzero.training.curvyzero_debug_visual_lightzero_smoke import (
    LocalDebugVisualLightZeroTimestep,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    SURVIVAL_TIME_REWARD_SCHEMA_ID,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    TURN_COMMIT_OPPONENT_POLICY_ID,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    TURN_COMMIT_OPPONENT_POLICY_KIND,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    TURN_COMMIT_OPPONENT_TRAINING_RELATION,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    TURN_COMMIT_REWARD_CREDIT_CAVEAT,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    TURN_COMMIT_SIMULTANEOUS_GAME_THEORY_CLAIM,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    TURN_COMMIT_TRUSTED_SELF_PLAY_CLAIM,
)
from curvyzero.training.curvyzero_survival_time_lightzero_smoke import (
    SURVIVAL_TIME_REWARD_SCHEMA_HASH,
)


try:  # Imported inside a LightZero/DI-engine runtime.
    import gym
    from ding.envs import BaseEnv
    from ding.envs import BaseEnvTimestep
    from ding.utils import ENV_REGISTRY
except ImportError as exc:  # pragma: no cover - local tree can compile without DI-engine.
    _LIGHTZERO_IMPORT_ERROR: ImportError | None = exc
    gym = None

    class BaseEnv:  # type: ignore[no-redef]
        pass

    class BaseEnvTimestep:  # type: ignore[no-redef]
        def __init__(self, obs: Any, reward: float, done: bool, info: dict[str, Any]):
            self.obs = obs
            self.reward = reward
            self.done = done
            self.info = info

    class _MissingEnvRegistry:
        def register(self, _name: str):
            def decorator(cls):
                return cls

            return decorator

    ENV_REGISTRY = _MissingEnvRegistry()
else:  # pragma: no cover - exercised only when LightZero/DI-engine is installed.
    _LIGHTZERO_IMPORT_ERROR = None


LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_TYPE = (
    "curvyzero_source_state_visual_turn_commit_lightzero"
)
LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_ID = (
    "CurvyZeroSourceStateVisualTurnCommitLightZero-v0"
)
LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_IMPORT_NAMES = (
    "curvyzero.training.curvyzero_source_state_visual_turn_commit_lightzero_env",
)
SOURCE_STATE_TURN_COMMIT_ADAPTER_IMPL_ID = (
    "curvyzero_source_state_visual_turn_commit_lightzero_adapter/v0"
)
STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID = (
    "curvyzero_source_state_rgb_canvas_like_gray64_stack4_player_perspective/v0"
)
STACKED_SOURCE_STATE_GRAY64_SHAPE = (4, 64, 64)
SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE = (
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    3,
)
SOURCE_STATE_CANVAS_LIKE_RAW_DTYPE = "uint8"
SOURCE_STATE_CANVAS_LIKE_RAW_VALUE_RANGE = (0, 255)
SOURCE_STATE_BROWSER_PIXEL_FIDELITY_CLAIM = "not_validated_against_browser_canvas"
SOURCE_STATE_CANVAS_LIKE_RAW_SCHEMA_HASH = stable_contract_hash(
    {
        "schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
        "renderer_impl_id": SOURCE_STATE_RGB_CANVAS_LIKE_RENDERER_IMPL_ID,
        "default_trail_render_mode": TRAIL_RENDER_MODE_DEFAULT,
        "supported_trail_render_modes": list(TRAIL_RENDER_MODE_ORDER),
        "shape": list(SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE),
        "dtype": SOURCE_STATE_CANVAS_LIKE_RAW_DTYPE,
        "range": list(SOURCE_STATE_CANVAS_LIKE_RAW_VALUE_RANGE),
        "frame_size": SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        "source": (
            "render_source_state_rgb_canvas_like("
            f"frame_size={SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE}, "
            f"trail_render_mode={TRAIL_RENDER_MODE_DEFAULT!r})"
        ),
        "truth_level": SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL,
        "browser_pixel_fidelity": False,
        "browser_pixel_fidelity_claim": SOURCE_STATE_BROWSER_PIXEL_FIDELITY_CLAIM,
    }
)
PLAYER_PERSPECTIVE_SCHEMA_ID = (
    "curvyzero_player_perspective_source_state_rgb_canvas_like_gray64/v0"
)
STACKED_SOURCE_STATE_GRAY64_SCHEMA_HASH = stable_contract_hash(
    {
        "schema_id": STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
        "single_frame_schema_id": SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID,
        "single_frame_schema_hash": SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH,
        "raw_observation_schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
        "raw_observation_schema_hash": SOURCE_STATE_CANVAS_LIKE_RAW_SCHEMA_HASH,
        "player_perspective_schema_id": PLAYER_PERSPECTIVE_SCHEMA_ID,
        "default_trail_render_mode": TRAIL_RENDER_MODE_DEFAULT,
        "supported_trail_render_modes": list(TRAIL_RENDER_MODE_ORDER),
        "shape": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
        "dtype": SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_DTYPE,
        "range": list(SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_VALUE_RANGE),
        "frame_stack_owner": "curvyzero_source_state_turn_commit_wrapper",
        "frame_stack_proof": (
            "wrapper_owned_full_canvas_rgb_to_player_perspective_gray64_fifo_stack; "
            "not LightZero env-manager stacking"
        ),
        "source_path": (
            "render_source_state_canvas_gray64 source-state raw RGB canvas -> "
            "optional active-player color perspective -> area-downsampled gray64 -> "
            "normalized FIFO stack"
        ),
        "controlled_player_semantics": (
            "active LightZero player is SELF; other player is OTHER in the model gray stack"
        ),
    }
)
SELF_BODY_VALUE = 96
OTHER_BODY_VALUE = 128
SELF_HEAD_VALUE = 224
OTHER_HEAD_VALUE = 232
SOURCE_BODY_VALUE_BASE = 96
SOURCE_BODY_VALUE_STEP = 32
SOURCE_HEAD_VALUE_BASE = 224
SOURCE_HEAD_VALUE_STEP = 8
DEFAULT_DECISION_MS = DEFAULT_SOURCE_FRAME_DECISION_MS
DEFAULT_MAX_TICKS = 2_000
ENV_VARIANT_SOURCE_STATE_TURN_COMMIT = "source_state_turn_commit"
SOURCE_STATE_TURN_COMMIT_RUNTIME_TOPOLOGY = (
    "stock_lightzero_scalar_turn_commit_over_simultaneous_source_state_env"
)
SOURCE_STATE_TURN_COMMIT_UNDERLYING_ENV_CLASS = "VectorMultiplayerEnv"
TURN_COMMIT_LEARNING_QUALITY_CLAIM = False
TURN_COMMIT_REWARD_CREDIT_STATUS = "untrusted_scalar_turn_commit_reward_credit"
TURN_COMMIT_TRAINING_STATUS = "plumbing_smoke_only"


class CurvyZeroSourceStateVisualTurnCommitLightZeroLocalEnv:
    """Private turn-commit adapter over the reconstructed vector CurvyTron env."""

    config = {
        "env_id": LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_ID,
        "lightzero_env_type": LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_TYPE,
        "observation_shape": STACKED_SOURCE_STATE_GRAY64_SHAPE,
        "action_space_size": ACTION_COUNT,
        "debug_fidelity_only": False,
        "source_fidelity_claim": "source_state_backed_non_browser_pixel",
        "env_variant": ENV_VARIANT_SOURCE_STATE_TURN_COMMIT,
        "runtime_topology": SOURCE_STATE_TURN_COMMIT_RUNTIME_TOPOLOGY,
        "two_seat_self_play": False,
        "current_policy_two_seat_action_collection": True,
        "two_seat_self_play_status": TURN_COMMIT_TRAINING_STATUS,
        "trusted_current_policy_self_play": False,
        "learning_quality_claim": TURN_COMMIT_LEARNING_QUALITY_CLAIM,
        "reward_credit_status": TURN_COMMIT_REWARD_CREDIT_STATUS,
        "underlying_env_class": SOURCE_STATE_TURN_COMMIT_UNDERLYING_ENV_CLASS,
        "runtime_env_impl_id": NATURAL_BONUS_ENV_IMPL_ID,
        "public_env_contract_id": PUBLIC_NATURAL_BONUS_ENV_CONTRACT_ID,
        "ruleset_id": NATURAL_BONUS_RULESET_ID,
        "death_mode": vector_runtime.DEATH_MODE_NORMAL,
        "uses_ale": False,
    }

    def __init__(self, cfg: Any | None = None):
        cfg = cfg or {}
        self.env_id = str(
            _cfg_get(cfg, "env_id", LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_ID)
        )
        self._seed = int(_cfg_get(cfg, "seed", 0))
        self._episode_seed = self._seed
        self._dynamic_seed = bool(_cfg_get(cfg, "dynamic_seed", False))
        (
            self._decision_source_frames,
            self._source_physics_step_ms,
            self._decision_ms,
        ) = _source_frame_decision_config(cfg)
        self._max_ticks = int(
            _cfg_get(cfg, "max_ticks", _cfg_get(cfg, "source_max_steps", DEFAULT_MAX_TICKS))
        )
        self._max_source_ticks = self._max_ticks * self._decision_source_frames
        disable_death_for_profile = bool(_cfg_get(cfg, "disable_death_for_profile", False))
        configured_death_mode = str(
            _cfg_get(cfg, "death_mode", vector_runtime.DEATH_MODE_NORMAL)
        )
        if disable_death_for_profile:
            configured_death_mode = vector_runtime.DEATH_MODE_PROFILE_NO_DEATH
        if configured_death_mode not in vector_runtime.DEATH_MODES:
            raise ValueError("death_mode must be 'normal' or 'profile_no_death'")
        self._death_mode = configured_death_mode
        self._disable_death_for_profile = (
            self._death_mode == vector_runtime.DEATH_MODE_PROFILE_NO_DEATH
        )
        telemetry_path = _cfg_get(cfg, "telemetry_path", None)
        self._telemetry_path = Path(str(telemetry_path)) if telemetry_path else None
        self._telemetry_stride = int(_cfg_get(cfg, "telemetry_stride", 1))
        if self._telemetry_stride < 1:
            raise ValueError("telemetry_stride must be at least 1")
        self._env = self._new_env(self._seed)
        self._raw_frame = np.zeros(SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE, dtype=np.uint8)
        self._perspective_frame = np.zeros(
            SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE,
            dtype=np.uint8,
        )
        self._gray64_frame = np.zeros(SOURCE_STATE_CANVAS_GRAY64_SHAPE, dtype=np.uint8)
        self._normalized_frame = np.zeros(
            SOURCE_STATE_CANVAS_GRAY64_SHAPE,
            dtype=np.float32,
        )
        self._perspective_rgb_palettes = tuple(
            _player_perspective_rgb_palette(player) for player in range(2)
        )
        self._stack = np.zeros((2, *STACKED_SOURCE_STATE_GRAY64_SHAPE), dtype=np.float32)
        self._turn_player_ids = ("player_0", "player_1")
        self._active_player_index = 0
        self._pending_actions: dict[int, int] = {}
        self._has_reset = False
        self._needs_reset = False
        self._last_batch = None
        self._episode_return = 0.0
        self._scalar_step_index = 0
        self._source_tick_index = 0
        self._observation_space = {
            "type": "Box",
            "shape": STACKED_SOURCE_STATE_GRAY64_SHAPE,
            "dtype": "float32",
            "low": 0.0,
            "high": 1.0,
        }
        self._action_space = {"type": "Discrete", "n": ACTION_COUNT}
        self._reward_space = {"type": "Box", "shape": (), "dtype": "float32"}

    @property
    def active_player_id(self) -> str:
        return self._turn_player_ids[self._active_player_index]

    @property
    def active_player_index(self) -> int:
        return self._active_player_index

    @property
    def last_reset_info(self) -> dict[str, Any] | None:
        if self._last_batch is None:
            return None
        return self._base_info()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        reset_seed = self._next_seed(seed)
        self._episode_seed = reset_seed
        self._env = self._new_env(reset_seed)
        self._stack.fill(0.0)
        self._active_player_index = 0
        self._pending_actions = {}
        self._needs_reset = False
        self._has_reset = True
        self._episode_return = 0.0
        self._scalar_step_index = 0
        self._source_tick_index = 0
        self._last_batch = self._env.reset(seed=reset_seed)
        return self._lightzero_observation(needs_reset=False)

    def step(self, action: Any) -> LocalDebugVisualLightZeroTimestep:
        if not self._has_reset:
            raise RuntimeError("reset must be called before step")
        if self._needs_reset:
            raise RuntimeError("reset must be called before stepping after done")
        action_id = _validate_action(action)
        acting_index = self._active_player_index
        if acting_index == 0:
            timestep = self._record_pending_turn_action(action_id)
        else:
            timestep = self._commit_joint_turn_action(action_id)
        self._scalar_step_index += 1
        self._write_telemetry_row(timestep=timestep)
        return timestep

    def close(self) -> None:
        return None

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = int(seed)
        self._episode_seed = self._seed
        self._dynamic_seed = bool(dynamic_seed)

    def random_action(self) -> int:
        return 1

    def enable_save_replay(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def render(self, mode: str = "source_state_visual_tensor") -> np.ndarray | None:
        if not self._has_reset:
            return None
        if mode == "source_state_visual_tensor":
            return self._stack[self._active_player_index].copy()
        if mode == "source_state_raw_visual_tensor":
            return self.raw_observation()
        if mode == "source_state_player_perspective_raw_visual_tensor":
            return self.raw_observation(player_perspective=True)
        if mode == "source_state_rgb_canvas_like":
            return self.raw_observation()
        if mode == "source_state_grayscale64_visual_tensor":
            return self._gray64_frame.copy()
        return None

    def raw_observation(self, *, player_perspective: bool = False) -> np.ndarray | None:
        """Return the latest full-size raw RGB canvas-like frame."""

        if not self._has_reset:
            return None
        frame = self._perspective_frame if player_perspective else self._raw_frame
        return frame.copy()

    def human_rgb_observation(
        self,
        *,
        frame_size: int = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    ) -> np.ndarray | None:
        if not self._has_reset:
            return None
        return render_source_state_rgb_canvas_like(self._env.state, row=0, frame_size=frame_size)

    def _new_env(self, seed: int) -> VectorMultiplayerEnv:
        return VectorMultiplayerEnv(
            batch_size=1,
            player_count=2,
            seed=seed,
            decision_ms=self._decision_ms,
            decision_source_frames=self._decision_source_frames,
            source_physics_step_ms=self._source_physics_step_ms,
            max_ticks=self._max_source_ticks,
            death_mode=self._death_mode,
            natural_bonus_spawn=True,
        )

    def _record_pending_turn_action(
        self,
        action_id: int,
    ) -> LocalDebugVisualLightZeroTimestep:
        acting_index = self._active_player_index
        self._pending_actions[acting_index] = int(action_id)
        self._active_player_index = 1
        next_obs = self._lightzero_observation(needs_reset=False)
        info = self._turn_info(
            acting_player_index=acting_index,
            next_active_player_index=1,
            committed_action_id=action_id,
            reward=0.0,
            done=False,
            terminated=False,
            truncated=False,
            physical_env_advanced=False,
            next_obs=next_obs,
        )
        return LocalDebugVisualLightZeroTimestep(next_obs, 0.0, False, info)

    def _commit_joint_turn_action(
        self,
        action_id: int,
    ) -> LocalDebugVisualLightZeroTimestep:
        acting_index = self._active_player_index
        self._pending_actions[acting_index] = int(action_id)
        actions = np.full((1, 2), -1, dtype=np.int16)
        for player, player_action in self._pending_actions.items():
            actions[0, player] = int(player_action)
        batch = self._env.step(actions, timer_advance_ms=self._decision_ms)
        self._last_batch = batch
        self._source_tick_index += 1
        done = bool(batch.done[0])
        terminated = bool(batch.terminated[0])
        truncated = bool(batch.truncated[0])
        reward = self._survival_reward_for_player(acting_index)
        self._episode_return += reward
        self._needs_reset = done
        self._pending_actions = {}
        next_obs = (
            self._lightzero_observation_for_player(acting_index, needs_reset=True)
            if done
            else self._lightzero_observation_for_player(0, needs_reset=False)
        )
        self._active_player_index = 0
        info = self._turn_info(
            acting_player_index=acting_index,
            next_active_player_index=0,
            committed_action_id=action_id,
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            physical_env_advanced=True,
            next_obs=next_obs,
            joint_action=actions[0].copy(),
            batch=batch,
        )
        return LocalDebugVisualLightZeroTimestep(next_obs, reward, done, info)

    def _lightzero_observation(self, *, needs_reset: bool) -> dict[str, Any]:
        return self._lightzero_observation_for_player(
            self._active_player_index,
            needs_reset=needs_reset,
        )

    def _lightzero_observation_for_player(
        self,
        player_index: int,
        *,
        needs_reset: bool,
    ) -> dict[str, Any]:
        stack = self._update_stack_for_player(player_index)
        mask = self._action_mask(player_index, active=not needs_reset)
        return {
            "observation": stack.copy(),
            "action_mask": mask,
            "to_play": -1,
            "timestep": int(self._scalar_step_index),
        }

    def _update_stack_for_player(self, player_index: int) -> np.ndarray:
        if player_index == 0:
            gray64 = render_source_state_canvas_gray64(
                self._env.state,
                row=0,
                out=self._gray64_frame,
                rgb_out=self._raw_frame,
                trail_render_mode=TRAIL_RENDER_MODE_DEFAULT,
            )
            np.copyto(self._perspective_frame, self._raw_frame)
        else:
            render_source_state_rgb_canvas_like(
                self._env.state,
                row=0,
                out=self._raw_frame,
                frame_size=SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
                trail_render_mode=TRAIL_RENDER_MODE_DEFAULT,
            )
            gray64 = render_source_state_canvas_gray64(
                self._env.state,
                row=0,
                out=self._gray64_frame,
                rgb_out=self._perspective_frame,
                player_rgb=self._perspective_rgb_palettes[player_index],
                trail_render_mode=TRAIL_RENDER_MODE_DEFAULT,
            )
        np.multiply(
            gray64,
            np.float32(1.0 / 255.0),
            out=self._normalized_frame,
            casting="unsafe",
        )
        self._stack[player_index, :-1] = self._stack[player_index, 1:]
        self._stack[player_index, -1] = self._normalized_frame[0]
        return self._stack[player_index]

    def _action_mask(self, player_index: int, *, active: bool) -> np.ndarray:
        if not active:
            return np.zeros(ACTION_COUNT, dtype=np.int8)
        mask = self._env._action_mask()[0, player_index].astype(np.int8, copy=True)
        return mask

    def _survival_reward_for_player(self, player_index: int) -> float:
        alive = bool(self._env.state["alive"][0, player_index])
        return 1.0 if alive else 0.0

    def _turn_info(
        self,
        *,
        acting_player_index: int,
        next_active_player_index: int,
        committed_action_id: int,
        reward: float,
        done: bool,
        terminated: bool,
        truncated: bool,
        physical_env_advanced: bool,
        next_obs: dict[str, Any],
        joint_action: np.ndarray | None = None,
        batch: Any | None = None,
    ) -> dict[str, Any]:
        info = self._base_info()
        acting_player_id = self._turn_player_ids[acting_player_index]
        next_active_player_id = self._turn_player_ids[next_active_player_index]
        joint_action_map = {
            self._turn_player_ids[player]: int(action)
            for player, action in sorted(self._pending_actions.items())
        }
        if joint_action is not None:
            joint_action_map = {
                self._turn_player_ids[player]: int(joint_action[player])
                for player in range(2)
            }
        reward_perspective = (
            "controlled_player_after_physical_commit"
            if physical_env_advanced
            else "bookkeeping_pending_action_no_physical_reward"
        )
        final_reward_map = None
        if done and batch is not None and batch.final_reward is not None:
            final_reward_map = {
                self._turn_player_ids[player]: float(batch.final_reward[0, player])
                for player in range(2)
            }
        info.update(
            {
                "step_index": int(self._source_tick_index - int(physical_env_advanced)),
                "adapter_timestep": int(self._scalar_step_index),
                "source_tick_index": int(self._source_tick_index),
                "acting_player_id": acting_player_id,
                "controlled_player_id": acting_player_id,
                "active_player_id": acting_player_id,
                "next_active_player_id": next_active_player_id,
                "committed_action_id": int(committed_action_id),
                "joint_action": joint_action_map,
                "joint_action_schema_id": JOINT_ACTION_SCHEMA_ID,
                "physical_env_advanced": bool(physical_env_advanced),
                "pending_action_count": int(len(self._pending_actions)),
                "pending_actions_private": True,
                "reward": float(reward),
                "reward_player_id": acting_player_id,
                "reward_perspective": reward_perspective,
                "reward_credit_caveat": TURN_COMMIT_REWARD_CREDIT_CAVEAT,
                "done": bool(done),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "needs_reset": bool(self._needs_reset),
                "terminal_reason": self._terminal_reason_name(),
                "final_observation": _copy_lightzero_observation(next_obs) if done else None,
                "final_reward_map": final_reward_map,
                "eval_episode_return": float(self._episode_return) if done else None,
            }
        )
        return info

    def _base_info(self) -> dict[str, Any]:
        public_info = self._env._public_info()
        runtime_env_impl_id = str(public_info["env_impl_id"])
        public_env_contract_id = str(public_info["public_env_contract_id"])
        ruleset_id = str(public_info["ruleset_id"])
        return {
            "env_id": self.env_id,
            "lightzero_env_type": LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_TYPE,
            "env_variant": ENV_VARIANT_SOURCE_STATE_TURN_COMMIT,
            "adapter_impl_id": SOURCE_STATE_TURN_COMMIT_ADAPTER_IMPL_ID,
            "lightzero_adapter_kind": "source_state_visual_turn_commit_native_train_muzero",
            "runtime_topology": SOURCE_STATE_TURN_COMMIT_RUNTIME_TOPOLOGY,
            "underlying_env_class": SOURCE_STATE_TURN_COMMIT_UNDERLYING_ENV_CLASS,
            "runtime_env_impl_id": runtime_env_impl_id,
            "turn_commit_adapter": True,
            "turn_commit_rule": "physical_env_advances_only_after_all_players_commit",
            "public_env_contract_id": public_env_contract_id,
            "env_impl_id": runtime_env_impl_id,
            "ruleset_id": ruleset_id,
            "rules_hash": str(public_info["rules_hash"]),
            "decision_ms": float(self._decision_ms),
            "decision_source_frames": int(self._decision_source_frames),
            "source_physics_step_ms": float(self._source_physics_step_ms),
            "source_frame_decision": True,
            "max_ticks": int(self._max_ticks),
            "max_source_ticks": int(self._max_source_ticks),
            "player_count": 2,
            "player_ids": self._turn_player_ids,
            "observation_schema_id": STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
            "observation_schema_hash": STACKED_SOURCE_STATE_GRAY64_SCHEMA_HASH,
            "schema_hash": STACKED_SOURCE_STATE_GRAY64_SCHEMA_HASH,
            "single_frame_schema_id": SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID,
            "single_frame_schema_hash": SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH,
            "raw_observation_schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
            "raw_observation_schema_hash": SOURCE_STATE_CANVAS_LIKE_RAW_SCHEMA_HASH,
            "raw_observation_available": True,
            "raw_observation_accessors": [
                "raw_observation()",
                "render('source_state_raw_visual_tensor')",
                "render('source_state_rgb_canvas_like')",
            ],
            "raw_observation_dtype": SOURCE_STATE_CANVAS_LIKE_RAW_DTYPE,
            "raw_observation_color_space": "RGB",
            "raw_observation_source": (
                "render_source_state_rgb_canvas_like("
                f"frame_size={SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE}, "
                f"trail_render_mode={TRAIL_RENDER_MODE_DEFAULT!r})"
            ),
            "grayscale_observation_source": (
                "render_source_state_canvas_gray64("
                "rgb_out=active_player_raw_observation_buffer, "
                f"trail_render_mode={TRAIL_RENDER_MODE_DEFAULT!r})"
            ),
            "player_perspective_schema_id": PLAYER_PERSPECTIVE_SCHEMA_ID,
            "renderer_impl_id": SOURCE_STATE_CANVAS_GRAY64_RENDERER_IMPL_ID,
            "raw_renderer_impl_id": SOURCE_STATE_RGB_CANVAS_LIKE_RENDERER_IMPL_ID,
            "human_rgb_renderer_impl_id": SOURCE_STATE_RGB_CANVAS_LIKE_RENDERER_IMPL_ID,
            "human_rgb_schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
            "human_rgb_truth_level": SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL,
            "human_rgb_default_frame_shape": [
                SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
                SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
                3,
            ],
            "truth_level": SOURCE_STATE_CANVAS_GRAY64_TRUTH_LEVEL,
            "source_fidelity_level": SOURCE_STATE_CANVAS_GRAY64_SOURCE_FIDELITY_LEVEL,
            "source_fidelity_claim": "source_state_backed_non_browser_pixel",
            "visual_surface": SOURCE_STATE_CANVAS_GRAY64_SURFACE,
            "visual_truth_level": SOURCE_STATE_CANVAS_GRAY64_TRUTH_LEVEL,
            "visual_source_state_backed": SOURCE_STATE_CANVAS_GRAY64_SOURCE_STATE_BACKED,
            "debug_fidelity_only": False,
            "browser_pixel_fidelity": SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY,
            "browser_pixel_fidelity_claim": SOURCE_STATE_BROWSER_PIXEL_FIDELITY_CLAIM,
            "uses_ale": SOURCE_STATE_CANVAS_GRAY64_USES_ALE,
            "ale_usage": "none",
            "default_trail_render_mode": TRAIL_RENDER_MODE_DEFAULT,
            "supported_trail_render_modes": list(TRAIL_RENDER_MODE_ORDER),
            "trail_render_mode": TRAIL_RENDER_MODE_DEFAULT,
            "shape": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
            "dtype": SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_DTYPE,
            "range": list(SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_VALUE_RANGE),
            "value_range": list(SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_VALUE_RANGE),
            "raw_frame_shape": list(SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE),
            "grayscale_frame_shape": list(SOURCE_STATE_CANVAS_GRAY64_SHAPE),
            "lightzero_payload_shape": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
            "model_observation_shape": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
            "frame_stack_owner": "curvyzero_source_state_turn_commit_wrapper",
            "frame_stack_proof": (
                "wrapper_owned_full_canvas_rgb_to_player_perspective_gray64_fifo_stack; "
                "not LightZero env-manager stacking"
            ),
            "reward_schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
            "reward_schema_hash": SURVIVAL_TIME_REWARD_SCHEMA_HASH,
            "bonus_support_mode": str(public_info["bonus_support_mode"]),
            "natural_bonus_spawn": bool(public_info["bonus_support"]["natural_bonus_spawn"]),
            "natural_bonus_pop_count": int(public_info["natural_bonus_pop_count"][0]),
            "natural_bonus_type_codes": public_info["bonus_support"][
                "enabled_natural_bonus_type_codes"
            ].astype(np.int16, copy=True).tolist(),
            "natural_bonus_type_names": [
                vector_runtime.BONUS_TYPE_NAME_BY_CODE[int(code)]
                for code in public_info["bonus_support"][
                    "enabled_natural_bonus_type_codes"
                ].tolist()
            ],
            "supported_natural_bonus_effect_types": list(
                public_info["bonus_support"]["supported_natural_bonus_effect_types"]
            ),
            "unsupported_natural_bonus_effects": list(
                public_info["bonus_support"]["unsupported_natural_bonus_effects"]
            ),
            "death_mode": self._death_mode,
            "disable_death_for_profile": self._disable_death_for_profile,
            "death_suppression_for_profile": self._disable_death_for_profile,
            "death_suppression_claim": (
                "profile_only_not_source_fidelity"
                if self._disable_death_for_profile
                else "none"
            ),
            "terminal_outcome_bonus": 0.0,
            "loser_penalty": 0.0,
            "winner_bonus": 0.0,
            "shared_policy_turn_commit": True,
            "opponent_policy_id": TURN_COMMIT_OPPONENT_POLICY_ID,
            "opponent_policy_kind": TURN_COMMIT_OPPONENT_POLICY_KIND,
            "opponent_training_relation": TURN_COMMIT_OPPONENT_TRAINING_RELATION,
            "opponent_policy_version": "current",
            "current_policy_self_play": True,
            "current_policy_self_play_blocker": None,
            "current_policy_self_play_caveat": TURN_COMMIT_REWARD_CREDIT_CAVEAT,
            "learning_quality_claim": TURN_COMMIT_LEARNING_QUALITY_CLAIM,
            "training_status": TURN_COMMIT_TRAINING_STATUS,
            "reward_credit_status": TURN_COMMIT_REWARD_CREDIT_STATUS,
            "trusted_current_policy_self_play": TURN_COMMIT_TRUSTED_SELF_PLAY_CLAIM,
            "simultaneous_game_theory_claim": TURN_COMMIT_SIMULTANEOUS_GAME_THEORY_CLAIM,
            "two_seat_self_play": False,
            "current_policy_two_seat_action_collection": True,
            "two_seat_self_play_status": TURN_COMMIT_TRAINING_STATUS,
            "fixed_opponent_is_two_seat_self_play": False,
            "episode_seed": self._episode_seed,
            "source_tick_index": int(self._source_tick_index),
            "adapter_timestep": int(self._scalar_step_index),
            "public_env_info": {
                "episode_id": int(public_info["episode_id"][0]),
                "tick_index": int(public_info["tick_index"][0]),
                "elapsed_ms": float(public_info["elapsed_ms"][0]),
                "terminal_reason_name": str(public_info["terminal_reason_name"][0]),
            },
        }

    def _write_telemetry_row(self, *, timestep: LocalDebugVisualLightZeroTimestep) -> None:
        if self._telemetry_path is None:
            return
        info = timestep.info
        adapter_timestep = int(info.get("adapter_timestep", self._scalar_step_index))
        physical_env_advanced = bool(info.get("physical_env_advanced", False))
        sampled_step = (
            adapter_timestep == 0
            or self._telemetry_stride == 1
            or adapter_timestep % self._telemetry_stride == 0
            or physical_env_advanced
            or bool(timestep.done)
        )
        if not sampled_step:
            return
        joint_action = info.get("joint_action")
        row = {
            "schema_id": "curvyzero_source_state_visual_turn_commit_env_step/v0",
            "telemetry_stride": int(self._telemetry_stride),
            "telemetry_sampled": self._telemetry_stride > 1,
            "env_variant": info.get("env_variant"),
            "runtime_topology": info.get("runtime_topology"),
            "underlying_env_class": info.get("underlying_env_class"),
            "runtime_env_impl_id": info.get("runtime_env_impl_id"),
            "env_impl_id": info.get("env_impl_id"),
            "public_env_contract_id": info.get("public_env_contract_id"),
            "ruleset_id": info.get("ruleset_id"),
            "rules_hash": info.get("rules_hash"),
            "bonus_support_mode": info.get("bonus_support_mode"),
            "natural_bonus_spawn": info.get("natural_bonus_spawn"),
            "natural_bonus_pop_count": info.get("natural_bonus_pop_count"),
            "death_mode": info.get("death_mode"),
            "disable_death_for_profile": info.get("disable_death_for_profile"),
            "step_index": info.get("step_index"),
            "adapter_timestep": adapter_timestep,
            "source_tick_index": info.get("source_tick_index"),
            "scalar_action": int(info.get("committed_action_id", -1)),
            "requested_ego_action": int(info.get("committed_action_id", -1)),
            "executed_ego_action": int(info.get("committed_action_id", -1)),
            "acting_player_id": info.get("acting_player_id"),
            "controlled_player_id": info.get("controlled_player_id"),
            "active_player_id": info.get("active_player_id"),
            "next_active_player_id": info.get("next_active_player_id"),
            "committed_action_id": info.get("committed_action_id"),
            "joint_action": joint_action,
            "current_policy_self_play": info.get("current_policy_self_play"),
            "trusted_current_policy_self_play": info.get("trusted_current_policy_self_play"),
            "learning_quality_claim": info.get("learning_quality_claim"),
            "training_status": info.get("training_status"),
            "reward_credit_status": info.get("reward_credit_status"),
            "reward_credit_caveat": info.get("reward_credit_caveat"),
            "simultaneous_game_theory_claim": info.get("simultaneous_game_theory_claim"),
            "two_seat_self_play": info.get("two_seat_self_play"),
            "current_policy_two_seat_action_collection": info.get(
                "current_policy_two_seat_action_collection"
            ),
            "two_seat_self_play_status": info.get("two_seat_self_play_status"),
            "physical_env_advanced": physical_env_advanced,
            "pending_action_count": info.get("pending_action_count"),
            "pending_actions_private": info.get("pending_actions_private"),
            "reward": float(timestep.reward),
            "reward_player_id": info.get("reward_player_id"),
            "reward_perspective": info.get("reward_perspective"),
            "done": bool(timestep.done),
            "terminated": bool(info.get("terminated", False)),
            "truncated": bool(info.get("truncated", False)),
            "terminal_reason": info.get("terminal_reason"),
            "reward_schema_id": info.get("reward_schema_id"),
            "observation_schema_id": info.get("observation_schema_id"),
            "frame_stack_owner": info.get("frame_stack_owner"),
            "visual_surface": info.get("visual_surface"),
            "visual_truth_level": info.get("visual_truth_level"),
            "visual_source_state_backed": info.get("visual_source_state_backed"),
            "source_fidelity_claim": info.get("source_fidelity_claim"),
            "debug_fidelity_only": info.get("debug_fidelity_only"),
            "uses_ale": info.get("uses_ale"),
        }
        self._telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with self._telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    def _terminal_reason_name(self) -> str:
        return str(self._env._public_info()["terminal_reason_name"][0])

    def _next_seed(self, seed: int | None) -> int:
        if seed is not None:
            return int(seed)
        if not self._dynamic_seed:
            return self._seed
        self._seed += 1
        return self._seed

    def __repr__(self) -> str:
        return (
            "CurvyZeroSourceStateVisualTurnCommitLightZeroLocalEnv("
            f"env_id={self.env_id!r}, active_player_id={self.active_player_id!r})"
        )


@ENV_REGISTRY.register(LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_TYPE)
class CurvyZeroSourceStateVisualTurnCommitLightZeroEnv(
    CurvyZeroSourceStateVisualTurnCommitLightZeroLocalEnv,
    BaseEnv,
):
    """Registered LightZero env using the source-state visual tensor."""

    config = dict(CurvyZeroSourceStateVisualTurnCommitLightZeroLocalEnv.config)

    def __init__(self, cfg: Any | None = None):
        super().__init__(cfg)
        self.lightzero_env_type = LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_TYPE
        if gym is not None:
            self._action_space = gym.spaces.Discrete(ACTION_COUNT)
            self._observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=STACKED_SOURCE_STATE_GRAY64_SHAPE,
                dtype=np.float32,
            )
            self._reward_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(),
                dtype=np.float32,
            )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_space(self):
        return self._reward_space

    def step(self, action: Any) -> BaseEnvTimestep:
        local_timestep = super().step(action)
        return local_timestep.to_base_env_timestep(BaseEnvTimestep)


def _normalize_player_perspective(
    frame: np.ndarray,
    *,
    controlled_player: int,
    out: np.ndarray,
    lut: np.ndarray | None = None,
) -> np.ndarray:
    """Legacy gray64 SELF/OTHER remap helper retained for focused compatibility tests."""

    mapping = _player_perspective_lut(controlled_player) if lut is None else lut
    np.take(mapping, frame, out=out)
    return out


def _player_perspective_lut(controlled_player: int) -> np.ndarray:
    mapping = np.arange(256, dtype=np.uint8)
    for source_player in range(2):
        body_value = SOURCE_BODY_VALUE_BASE + source_player * SOURCE_BODY_VALUE_STEP
        head_value = SOURCE_HEAD_VALUE_BASE + source_player * SOURCE_HEAD_VALUE_STEP
        mapping[body_value] = (
            SELF_BODY_VALUE if source_player == controlled_player else OTHER_BODY_VALUE
        )
        mapping[head_value] = (
            SELF_HEAD_VALUE if source_player == controlled_player else OTHER_HEAD_VALUE
        )
    return mapping


def _player_perspective_rgb_palette(controlled_player: int) -> tuple[tuple[int, int, int], ...]:
    """Map source player colors to stable SELF/OTHER colors for shared-policy input."""

    colors = np.asarray(SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB, dtype=np.uint8).copy()
    self_rgb = np.asarray(SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[0], dtype=np.uint8)
    other_rgb = np.asarray(SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[1], dtype=np.uint8)
    for source_player in range(2):
        colors[source_player] = self_rgb if source_player == controlled_player else other_rgb
    return tuple(tuple(int(channel) for channel in color) for color in colors)


def _validate_action(action: Any) -> int:
    try:
        action_id = int(np.asarray(action).item())
    except Exception as exc:
        raise ValueError(f"action must be scalar integer-like, got {action!r}") from exc
    if action_id < 0 or action_id >= ACTION_COUNT:
        raise ValueError(f"action must be in [0, {ACTION_COUNT}), got {action_id}")
    return action_id


def _copy_lightzero_observation(observation: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in observation.items():
        copied[key] = value.copy() if isinstance(value, np.ndarray) else value
    return copied


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _source_frame_decision_config(cfg: Any) -> tuple[int, float, float]:
    source_physics_step_ms = float(
        _cfg_get(cfg, "source_physics_step_ms", SOURCE_PHYSICS_STEP_MS)
    )
    if not np.isfinite(source_physics_step_ms) or source_physics_step_ms <= 0.0:
        raise ValueError("source_physics_step_ms must be positive and finite")

    raw_frames = _cfg_get(cfg, "decision_source_frames", None)
    if raw_frames is None:
        raw_decision_ms = _cfg_get(cfg, "decision_ms", None)
        if raw_decision_ms is None:
            frames = DEFAULT_DECISION_SOURCE_FRAMES
        else:
            ratio = float(raw_decision_ms) / source_physics_step_ms
            frames = int(round(ratio))
            if frames < 1 or not np.isclose(ratio, frames, rtol=0.0, atol=1e-6):
                raise ValueError(
                    "decision_ms must be a whole number of source physics frames; "
                    "prefer decision_source_frames"
                )
    else:
        frames = int(raw_frames)
        if frames < 1:
            raise ValueError("decision_source_frames must be positive")

    return frames, source_physics_step_ms, frames * source_physics_step_ms


__all__ = [
    "CurvyZeroSourceStateVisualTurnCommitLightZeroEnv",
    "CurvyZeroSourceStateVisualTurnCommitLightZeroLocalEnv",
    "LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_ID",
    "LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_TYPE",
    "LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_IMPORT_NAMES",
    "SOURCE_STATE_TURN_COMMIT_ADAPTER_IMPL_ID",
    "SOURCE_STATE_CANVAS_LIKE_RAW_SCHEMA_HASH",
    "SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE",
    "STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID",
    "STACKED_SOURCE_STATE_GRAY64_SCHEMA_HASH",
    "STACKED_SOURCE_STATE_GRAY64_SHAPE",
    "TURN_COMMIT_TRAINING_STATUS",
]
