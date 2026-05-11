"""Source-state-backed single-ego CurvyTron env for native LightZero MuZero."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from curvyzero.env import vector_runtime
from curvyzero.env.vector_multiplayer_env import ACTION_COUNT
from curvyzero.env.vector_multiplayer_env import JOINT_ACTION_SCHEMA_ID
from curvyzero.env.vector_multiplayer_env import NATURAL_BONUS_ENV_IMPL_ID
from curvyzero.env.vector_multiplayer_env import NATURAL_BONUS_RULESET_ID
from curvyzero.env.vector_multiplayer_env import PUBLIC_NATURAL_BONUS_ENV_CONTRACT_ID
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.env.trainer_contract import (
    REWARD_SCHEMA_HASH as SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_HASH,
)
from curvyzero.env.trainer_contract import (
    REWARD_SCHEMA_ID as SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_ID,
)
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
    SOURCE_STATE_RGB_CANVAS_LIKE_RENDERER_IMPL_ID,
)
from curvyzero.env.vector_visual_observation import SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL,
)
from curvyzero.env.vector_visual_observation import rgb_canvas_like_to_gray64
from curvyzero.env.vector_visual_observation import render_source_state_rgb_canvas_like
from curvyzero.training.curvyzero_debug_visual_lightzero_smoke import (
    LocalDebugVisualLightZeroTimestep,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    CURRENT_POLICY_SELF_PLAY_BLOCKER,
    CURRENT_POLICY_SELF_PLAY_CLAIM,
    OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
    OPPONENT_TRAINING_RELATION_FIXED_STRAIGHT,
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


LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_TYPE = (
    "curvyzero_source_state_visual_survival_lightzero"
)
LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_ID = (
    "CurvyZeroSourceStateVisualSurvivalLightZero-v0"
)
LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_IMPORT_NAMES = (
    "curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env",
)
SOURCE_STATE_VISUAL_SURVIVAL_ADAPTER_IMPL_ID = (
    "curvyzero_source_state_visual_survival_lightzero_adapter/v0"
)
SOURCE_STATE_FIXED_OPPONENT_ENV_VARIANT = "source_state_fixed_opponent"
SOURCE_STATE_FIXED_OPPONENT_RUNTIME_TOPOLOGY = (
    "single_ego_lightzero_action_vs_fixed_straight_opponent"
)
SOURCE_STATE_FIXED_OPPONENT_TWO_SEAT_STATUS = "not_two_seat_self_play"
SOURCE_STATE_FIXED_OPPONENT_UNDERLYING_ENV_CLASS = "VectorMultiplayerEnv"
LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_TYPE = (
    "curvyzero_source_state_visual_joint_action_lightzero"
)
LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_ID = (
    "CurvyZeroSourceStateVisualJointActionLightZero-v0"
)
LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_IMPORT_NAMES = (
    "curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env",
)
SOURCE_STATE_JOINT_ACTION_ADAPTER_IMPL_ID = (
    "curvyzero_source_state_visual_joint_action_lightzero_adapter/v0"
)
SOURCE_STATE_JOINT_ACTION_ENV_VARIANT = "source_state_joint_action"
SOURCE_STATE_JOINT_ACTION_RUNTIME_TOPOLOGY = (
    "stock_lightzero_centralized_9_action_joint_control_one_source_tick"
)
SOURCE_STATE_JOINT_ACTION_TRAINING_STATUS = (
    "centralized_joint_action_control_not_true_competitive_self_play"
)
JOINT_ACTION_COUNT = ACTION_COUNT * ACTION_COUNT
STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID = (
    "curvyzero_source_state_rgb_canvas_like_gray64_stack4/v0"
)
STACKED_SOURCE_STATE_GRAY64_SHAPE = (4, 64, 64)
SOURCE_STATE_CANVAS_LIKE_RAW64_SHAPE = (64, 64, 3)
SOURCE_STATE_CANVAS_LIKE_RAW64_DTYPE = "uint8"
SOURCE_STATE_CANVAS_LIKE_RAW64_VALUE_RANGE = (0, 255)
SOURCE_STATE_CANVAS_LIKE_RAW64_SCHEMA_HASH = stable_contract_hash(
    {
        "schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
        "renderer_impl_id": SOURCE_STATE_RGB_CANVAS_LIKE_RENDERER_IMPL_ID,
        "shape": list(SOURCE_STATE_CANVAS_LIKE_RAW64_SHAPE),
        "dtype": SOURCE_STATE_CANVAS_LIKE_RAW64_DTYPE,
        "range": list(SOURCE_STATE_CANVAS_LIKE_RAW64_VALUE_RANGE),
        "frame_size": 64,
        "source": "render_source_state_rgb_canvas_like(frame_size=64)",
        "truth_level": SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL,
        "browser_pixel_fidelity": False,
    }
)
SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_ID = SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID
SOURCE_STATE_CANVAS_LIKE_GRAY64_RENDERER_IMPL_ID = (
    SOURCE_STATE_CANVAS_GRAY64_RENDERER_IMPL_ID
)
SOURCE_STATE_CANVAS_LIKE_GRAY64_SURFACE = SOURCE_STATE_CANVAS_GRAY64_SURFACE
SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_HASH = SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH
PLAYER_PERSPECTIVE_SCHEMA_ID = "curvyzero_player_perspective_source_state_gray64/v0"
SELF_BODY_VALUE = 96
OTHER_BODY_VALUE = 128
SELF_HEAD_VALUE = 224
OTHER_HEAD_VALUE = 232
SOURCE_BODY_VALUE_BASE = 96
SOURCE_BODY_VALUE_STEP = 32
SOURCE_HEAD_VALUE_BASE = 224
SOURCE_HEAD_VALUE_STEP = 8
DEFAULT_DECISION_MS = 300.0
DEFAULT_MAX_TICKS = 2_000
DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY = 0.0
DEFAULT_POLICY_ACTION_REPEAT_MAX = 1
DEFAULT_POLICY_ACTION_REPEAT_MIN = 1
POLICY_ACTION_REPEAT_SEED_OFFSET = 2027
CONTROL_STOCHASTICITY_SCHEMA_ID = "curvyzero_policy_action_repeat_stochasticity/v0"
STRAIGHT_ACTION_ID = 1
REWARD_VARIANT_SPARSE_OUTCOME = "sparse_outcome"
REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME = "dense_survival_plus_outcome"
REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC = "all_players_alive_diagnostic"
SOURCE_STATE_FIXED_OPPONENT_REWARD_VARIANTS = (
    REWARD_VARIANT_SPARSE_OUTCOME,
    REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
)
SOURCE_STATE_JOINT_ACTION_REWARD_VARIANTS = (
    REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC,
)
DENSE_SURVIVAL_PLUS_OUTCOME_REWARD_SCHEMA_ID = (
    "curvyzero_dense_survival_plus_sparse_outcome/v0"
)
DENSE_SURVIVAL_PLUS_OUTCOME_REWARD_SCHEMA = {
    "schema_id": DENSE_SURVIVAL_PLUS_OUTCOME_REWARD_SCHEMA_ID,
    "dtype": "float32",
    "episode_unit": "one_round",
    "perspective": "ego_player",
    "alignment": "reward_t_plus_1_after_one_source_tick",
    "trainer_reward_terms": [
        "dense_alive_helper_for_ego_player",
        "sparse_round_outcome_for_ego_player",
    ],
    "dense_alive_helper": 1.0,
    "sparse_round_outcome_schema_id": SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_ID,
    "survival_length_metric_is_telemetry": True,
    "non_claims": [
        "not_zero_sum_after_dense_helper",
        "not_two_seat_self_play",
        "not_a_learning_claim",
    ],
}
DENSE_SURVIVAL_PLUS_OUTCOME_REWARD_SCHEMA_HASH = stable_contract_hash(
    DENSE_SURVIVAL_PLUS_OUTCOME_REWARD_SCHEMA
)
ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_ID = (
    "curvyzero_all_players_alive_diagnostic/v0"
)
ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA = {
    "schema_id": ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_ID,
    "dtype": "float32",
    "episode_unit": "one_round",
    "perspective": "centralized_joint_action_controller",
    "alignment": "reward_t_plus_1_after_one_source_tick",
    "reward_unit": "one_real_source_tick",
    "post_transition_all_players_alive_reward": 1.0,
    "post_transition_any_player_dead_reward": 0.0,
    "terminal_outcome_bonus": 0.0,
    "loser_penalty": 0.0,
    "winner_bonus": 0.0,
    "draw_bonus": 0.0,
    "truncation_bonus": 0.0,
    "episode_return": "sum of all-players-alive rewards for centralized control",
    "non_claims": [
        "not_per_player_reward",
        "not_zero_sum_reward",
        "not_true_competitive_self_play",
        "not_sparse_outcome_reward",
    ],
}
ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_HASH = stable_contract_hash(
    ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA
)
STACKED_SOURCE_STATE_GRAY64_SCHEMA_HASH = stable_contract_hash(
    {
        "schema_id": STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
        "single_frame_schema_id": SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_ID,
        "single_frame_schema_hash": SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_HASH,
        "raw_observation_schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
        "raw_observation_schema_hash": SOURCE_STATE_CANVAS_LIKE_RAW64_SCHEMA_HASH,
        "shape": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
        "dtype": SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_DTYPE,
        "range": list(SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_VALUE_RANGE),
        "frame_stack_owner": "curvyzero_source_state_survival_wrapper",
        "frame_stack_proof": "wrapper_owned_fifo_stack; not LightZero env-manager stacking",
        "source_path": (
            "source-state canvas-like RGB64 raw frame -> luminance gray64 -> "
            "normalized FIFO stack"
        ),
    }
)


class CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv:
    """Native single-ego LightZero env over the reconstructed vector CurvyTron env."""

    _default_reward_variant = REWARD_VARIANT_SPARSE_OUTCOME
    _allowed_reward_variants = SOURCE_STATE_FIXED_OPPONENT_REWARD_VARIANTS

    config = {
        "env_id": LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_ID,
        "lightzero_env_type": LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_TYPE,
        "lightzero_import_names": LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_IMPORT_NAMES,
        "observation_shape": STACKED_SOURCE_STATE_GRAY64_SHAPE,
        "action_space_size": ACTION_COUNT,
        "debug_fidelity_only": False,
        "source_fidelity_claim": "source_state_backed_non_browser_pixel",
        "env_variant": SOURCE_STATE_FIXED_OPPONENT_ENV_VARIANT,
        "runtime_topology": SOURCE_STATE_FIXED_OPPONENT_RUNTIME_TOPOLOGY,
        "two_seat_self_play": False,
        "two_seat_self_play_status": SOURCE_STATE_FIXED_OPPONENT_TWO_SEAT_STATUS,
        "fixed_opponent_is_two_seat_self_play": False,
        "underlying_env_class": SOURCE_STATE_FIXED_OPPONENT_UNDERLYING_ENV_CLASS,
        "runtime_env_impl_id": NATURAL_BONUS_ENV_IMPL_ID,
        "public_env_contract_id": PUBLIC_NATURAL_BONUS_ENV_CONTRACT_ID,
        "ruleset_id": NATURAL_BONUS_RULESET_ID,
        "death_mode": vector_runtime.DEATH_MODE_NORMAL,
        "disable_death_for_profile": False,
        "control_stochasticity_schema_id": CONTROL_STOCHASTICITY_SCHEMA_ID,
        "reward_variant": REWARD_VARIANT_SPARSE_OUTCOME,
        "policy_action_repeat_min": DEFAULT_POLICY_ACTION_REPEAT_MIN,
        "policy_action_repeat_max": DEFAULT_POLICY_ACTION_REPEAT_MAX,
        "policy_action_repeat_extra_probability": (
            DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
        ),
        "uses_ale": False,
    }

    def __init__(self, cfg: Any | None = None):
        cfg = cfg or {}
        self.env_id = str(
            _cfg_get(cfg, "env_id", LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_ID)
        )
        self.ego_player_index = int(_cfg_get(cfg, "ego_player_index", 0))
        if self.ego_player_index != 0:
            raise ValueError("source-state native survival env currently supports ego_player_index=0")
        self.opponent_player_index = 1
        self.ego_player_id = "player_0"
        self.opponent_player_id = "player_1"
        self._seed = int(_cfg_get(cfg, "seed", 0))
        self._episode_seed = self._seed
        self._dynamic_seed = bool(_cfg_get(cfg, "dynamic_seed", False))
        self._decision_ms = float(_cfg_get(cfg, "decision_ms", DEFAULT_DECISION_MS))
        self._max_ticks = int(
            _cfg_get(cfg, "max_ticks", _cfg_get(cfg, "source_max_steps", DEFAULT_MAX_TICKS))
        )
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
        self._override_probability = float(
            _cfg_get(cfg, "ego_action_straight_override_probability", 0.0)
        )
        if not 0.0 <= self._override_probability <= 1.0:
            raise ValueError("ego_action_straight_override_probability must be in [0, 1]")
        self._override_action_id = int(
            _cfg_get(cfg, "ego_action_straight_override_action_id", STRAIGHT_ACTION_ID)
        )
        if self._override_action_id != STRAIGHT_ACTION_ID:
            raise ValueError("only straight override action id 1 is supported")
        configured_override_seed = _cfg_get(cfg, "ego_action_straight_override_seed", None)
        self._configured_override_seed = (
            None if configured_override_seed is None else int(configured_override_seed)
        )
        self._override_seed = self._override_seed_for(self._seed)
        self._override_rng = np.random.default_rng(self._override_seed)
        self._policy_action_repeat_min = int(
            _cfg_get(cfg, "policy_action_repeat_min", DEFAULT_POLICY_ACTION_REPEAT_MIN)
        )
        self._policy_action_repeat_max = int(
            _cfg_get(cfg, "policy_action_repeat_max", DEFAULT_POLICY_ACTION_REPEAT_MAX)
        )
        self._policy_action_repeat_extra_probability = float(
            _cfg_get(
                cfg,
                "policy_action_repeat_extra_probability",
                DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY,
            )
        )
        self._validate_policy_action_repeat_config()
        configured_repeat_seed = _cfg_get(cfg, "policy_action_repeat_seed", None)
        self._configured_repeat_seed = (
            None if configured_repeat_seed is None else int(configured_repeat_seed)
        )
        self._policy_action_repeat_seed = self._policy_action_repeat_seed_for(self._seed)
        self._policy_action_repeat_rng = np.random.default_rng(
            self._policy_action_repeat_seed
        )
        configured_profile = _cfg_get(cfg, "control_noise_profile_id", None)
        self._control_noise_profile_id = (
            str(configured_profile)
            if configured_profile is not None
            else self._default_control_noise_profile_id()
        )
        default_reward_variant = str(
            getattr(self, "_default_reward_variant", REWARD_VARIANT_SPARSE_OUTCOME)
        )
        allowed_reward_variants = tuple(
            getattr(
                self,
                "_allowed_reward_variants",
                SOURCE_STATE_FIXED_OPPONENT_REWARD_VARIANTS,
            )
        )
        self._reward_variant = str(_cfg_get(cfg, "reward_variant", default_reward_variant))
        if self._reward_variant not in allowed_reward_variants:
            raise ValueError(
                f"{type(self).__name__} reward_variant must be one of "
                f"{allowed_reward_variants!r}; got {self._reward_variant!r}"
            )
        self._env = self._new_env(self._seed)
        self._raw_frame = np.zeros(SOURCE_STATE_CANVAS_LIKE_RAW64_SHAPE, dtype=np.uint8)
        self._gray64_frame = np.zeros(SOURCE_STATE_CANVAS_GRAY64_SHAPE, dtype=np.uint8)
        self._normalized_frame = np.zeros(SOURCE_STATE_CANVAS_GRAY64_SHAPE, dtype=np.float32)
        self._stack = np.zeros(STACKED_SOURCE_STATE_GRAY64_SHAPE, dtype=np.float32)
        self._has_reset = False
        self._needs_reset = False
        self._last_batch = None
        self._episode_return = 0.0
        self._step_index = 0
        self._physical_step_index = 0
        self._observation_space = {
            "type": "Box",
            "shape": STACKED_SOURCE_STATE_GRAY64_SHAPE,
            "dtype": "float32",
            "low": 0.0,
            "high": 1.0,
        }
        self._action_space = {"type": "Discrete", "n": ACTION_COUNT}
        self._reward_space = self._make_reward_space()

    @property
    def last_reset_info(self) -> dict[str, Any] | None:
        if not self._has_reset:
            return None
        return self._base_info()

    @property
    def legal_actions(self) -> np.ndarray:
        return np.arange(ACTION_COUNT, dtype=np.int64)

    def _validate_policy_action_repeat_config(self) -> None:
        if self._policy_action_repeat_min < 1:
            raise ValueError("policy_action_repeat_min must be at least 1")
        if self._policy_action_repeat_max < self._policy_action_repeat_min:
            raise ValueError(
                "policy_action_repeat_max must be greater than or equal to "
                "policy_action_repeat_min"
            )
        if not 0.0 <= self._policy_action_repeat_extra_probability <= 1.0:
            raise ValueError(
                "policy_action_repeat_extra_probability must be in [0, 1]"
            )

    def _policy_action_repeat_seed_for(self, reset_seed: int) -> int:
        if self._configured_repeat_seed is not None:
            return int(self._configured_repeat_seed)
        return int(reset_seed) + POLICY_ACTION_REPEAT_SEED_OFFSET

    def _default_control_noise_profile_id(self) -> str:
        if (
            self._policy_action_repeat_min == DEFAULT_POLICY_ACTION_REPEAT_MIN
            and self._policy_action_repeat_max == DEFAULT_POLICY_ACTION_REPEAT_MAX
            and self._policy_action_repeat_extra_probability
            == DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
        ):
            return "none"
        return (
            "policy_action_repeat:"
            f"min={self._policy_action_repeat_min},"
            f"max={self._policy_action_repeat_max},"
            f"extra={self._policy_action_repeat_extra_probability:g}"
        )

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        reset_seed = self._next_seed(seed)
        self._episode_seed = reset_seed
        self._override_seed = self._override_seed_for(reset_seed)
        self._override_rng = np.random.default_rng(self._override_seed)
        self._policy_action_repeat_seed = self._policy_action_repeat_seed_for(reset_seed)
        self._policy_action_repeat_rng = np.random.default_rng(
            self._policy_action_repeat_seed
        )
        self._env = self._new_env(reset_seed)
        self._stack.fill(0.0)
        self._needs_reset = False
        self._has_reset = True
        self._episode_return = 0.0
        self._step_index = 0
        self._physical_step_index = 0
        self._last_batch = self._env.reset(seed=reset_seed)
        return self._lightzero_observation(needs_reset=False)

    def step(self, action: Any) -> LocalDebugVisualLightZeroTimestep:
        if not self._has_reset:
            raise RuntimeError("reset must be called before step")
        if self._needs_reset:
            raise RuntimeError("reset must be called before stepping after done")
        requested_action = _validate_action(action)
        executed_action, override_applied = self._executed_ego_action(requested_action)
        opponent_action = STRAIGHT_ACTION_ID
        joint_action = np.array([[executed_action, opponent_action]], dtype=np.int16)
        action_repeat_requested = self._sample_policy_action_repeat()
        action_repeat_executed = 0
        reward = 0.0
        sparse_outcome_reward_sum = 0.0
        dense_survival_helper_sum = 0.0
        done = False
        terminated = False
        truncated = False
        batch = None
        for _ in range(action_repeat_requested):
            batch = self._env.step(joint_action, timer_advance_ms=self._decision_ms)
            self._last_batch = batch
            action_repeat_executed += 1
            self._physical_step_index += 1
            components = self._reward_components_for_player(
                batch=batch,
                player_index=self.ego_player_index,
            )
            sparse_outcome_reward_sum += components["sparse_outcome_reward"]
            dense_survival_helper_sum += components["dense_survival_helper"]
            reward += components["trainer_reward"]
            done = bool(batch.done[0])
            terminated = bool(batch.terminated[0])
            truncated = bool(batch.truncated[0])
            if done:
                break
        if batch is None:
            raise RuntimeError("policy action repeat produced no physical env step")
        self._needs_reset = done
        self._episode_return += reward
        self._step_index += 1
        next_obs = self._lightzero_observation(needs_reset=done)
        info = self._step_info(
            requested_action=requested_action,
            executed_action=executed_action,
            override_applied=override_applied,
            opponent_action=opponent_action,
            joint_action=joint_action[0],
            action_repeat_requested=action_repeat_requested,
            action_repeat_executed=action_repeat_executed,
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            next_obs=next_obs,
            batch=batch,
            sparse_outcome_reward_sum=sparse_outcome_reward_sum,
            dense_survival_helper_sum=dense_survival_helper_sum,
        )
        timestep = LocalDebugVisualLightZeroTimestep(next_obs, reward, done, info)
        self._write_telemetry_row(timestep=timestep)
        return timestep

    def close(self) -> None:
        return None

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = int(seed)
        self._episode_seed = self._seed
        self._dynamic_seed = bool(dynamic_seed)
        self._override_seed = self._override_seed_for(self._seed)
        self._override_rng = np.random.default_rng(self._override_seed)
        self._policy_action_repeat_seed = self._policy_action_repeat_seed_for(self._seed)
        self._policy_action_repeat_rng = np.random.default_rng(
            self._policy_action_repeat_seed
        )

    def random_action(self) -> int:
        rng = np.random.default_rng(self._seed + self._step_index)
        return int(rng.integers(ACTION_COUNT))

    def enable_save_replay(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def render(self, mode: str = "source_state_visual_tensor") -> np.ndarray | None:
        if not self._has_reset:
            return None
        if mode == "source_state_visual_tensor":
            return self._stack.copy()
        if mode == "source_state_raw_visual_tensor":
            return self.raw_observation()
        if mode == "source_state_rgb_canvas_like":
            return self.raw_observation()
        if mode == "source_state_grayscale64_visual_tensor":
            return self._gray64_frame.copy()
        if mode == "source_state_player_perspective_raw_visual_tensor":
            return self.raw_observation(player_perspective=True)
        return None

    def raw_observation(self, *, player_perspective: bool = False) -> np.ndarray | None:
        """Return the latest raw RGB canvas-like frame before grayscale stacking."""

        if not self._has_reset:
            return None
        _ = player_perspective
        return self._raw_frame.copy()

    def __repr__(self) -> str:
        return (
            "CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv("
            f"env_id={self.env_id!r}, "
            f"ego_player_id={self.ego_player_id!r}, "
            f"opponent_player_id={self.opponent_player_id!r}, "
            "opponent_policy_kind='fixed_straight')"
        )

    def _new_env(self, seed: int) -> VectorMultiplayerEnv:
        return VectorMultiplayerEnv(
            batch_size=1,
            player_count=2,
            seed=seed,
            decision_ms=self._decision_ms,
            max_ticks=self._max_ticks,
            death_mode=self._death_mode,
            natural_bonus_spawn=True,
        )

    def _lightzero_observation(self, *, needs_reset: bool) -> dict[str, Any]:
        stack = self._update_stack()
        return {
            "observation": stack.copy(),
            "action_mask": self._action_mask(active=not needs_reset),
            "to_play": -1,
            "timestep": int(self._step_index),
        }

    def _update_stack(self) -> np.ndarray:
        render_source_state_rgb_canvas_like(
            self._env.state,
            row=0,
            out=self._raw_frame,
            frame_size=64,
        )
        gray64 = rgb_canvas_like_to_gray64(
            self._raw_frame,
            out=self._gray64_frame,
        )
        np.multiply(
            gray64,
            np.float32(1.0 / 255.0),
            out=self._normalized_frame,
            casting="unsafe",
        )
        self._stack[:-1] = self._stack[1:]
        self._stack[-1] = self._normalized_frame[0]
        return self._stack

    def _action_mask(self, *, active: bool) -> np.ndarray:
        if not active:
            return np.zeros(ACTION_COUNT, dtype=np.int8)
        return self._env._action_mask()[0, self.ego_player_index].astype(np.int8, copy=True)

    def _executed_ego_action(self, requested_action: int) -> tuple[int, bool]:
        if self._override_probability <= 0.0:
            return int(requested_action), False
        if float(self._override_rng.random()) < self._override_probability:
            return STRAIGHT_ACTION_ID, True
        return int(requested_action), False

    def _validate_policy_action_repeat_config(self) -> None:
        if self._policy_action_repeat_min < 1:
            raise ValueError("policy_action_repeat_min must be at least 1")
        if self._policy_action_repeat_max < self._policy_action_repeat_min:
            raise ValueError(
                "policy_action_repeat_max must be greater than or equal to "
                "policy_action_repeat_min"
            )
        if not 0.0 <= self._policy_action_repeat_extra_probability <= 1.0:
            raise ValueError(
                "policy_action_repeat_extra_probability must be in [0, 1]"
            )

    def _sample_policy_action_repeat(self) -> int:
        repeat = int(self._policy_action_repeat_min)
        while repeat < self._policy_action_repeat_max:
            if (
                float(self._policy_action_repeat_rng.random())
                >= self._policy_action_repeat_extra_probability
            ):
                break
            repeat += 1
        return repeat

    def _default_control_noise_profile_id(self) -> str:
        parts = []
        if self._override_probability > 0.0:
            parts.append("straight_override")
        if self._policy_action_repeat_max > 1 or self._policy_action_repeat_min > 1:
            parts.append("policy_action_repeat")
        return "+".join(parts) if parts else "none"

    def _survival_reward_for_player(self, player_index: int) -> float:
        alive = bool(self._env.state["alive"][0, player_index])
        return 1.0 if alive else 0.0

    def _reward_components_for_player(
        self,
        *,
        batch: Any,
        player_index: int,
    ) -> dict[str, float]:
        sparse_outcome = float(batch.reward[0, player_index])
        dense_helper = 0.0
        if self._reward_variant == REWARD_VARIANT_SPARSE_OUTCOME:
            trainer_reward = sparse_outcome
        elif self._reward_variant == REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME:
            dense_helper = self._survival_reward_for_player(player_index)
            trainer_reward = sparse_outcome + dense_helper
        elif self._reward_variant == REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC:
            alive = self._env.state["alive"][0, :2].astype(bool)
            trainer_reward = 1.0 if bool(np.all(alive)) else 0.0
        else:
            raise RuntimeError(f"unsupported reward_variant {self._reward_variant!r}")
        return {
            "trainer_reward": float(trainer_reward),
            "sparse_outcome_reward": float(sparse_outcome),
            "dense_survival_helper": float(dense_helper),
        }

    def _scalar_reward_for_player(self, *, batch: Any, player_index: int) -> float:
        return self._reward_components_for_player(
            batch=batch,
            player_index=player_index,
        )["trainer_reward"]

    def _reward_schema_id(self) -> str:
        if self._reward_variant == REWARD_VARIANT_SPARSE_OUTCOME:
            return SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_ID
        if self._reward_variant == REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME:
            return DENSE_SURVIVAL_PLUS_OUTCOME_REWARD_SCHEMA_ID
        if self._reward_variant == REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC:
            return ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_ID
        raise RuntimeError(f"unsupported reward_variant {self._reward_variant!r}")

    def _reward_schema_hash(self) -> str:
        if self._reward_variant == REWARD_VARIANT_SPARSE_OUTCOME:
            return SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_HASH
        if self._reward_variant == REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME:
            return DENSE_SURVIVAL_PLUS_OUTCOME_REWARD_SCHEMA_HASH
        if self._reward_variant == REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC:
            return ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_HASH
        raise RuntimeError(f"unsupported reward_variant {self._reward_variant!r}")

    def _reward_perspective(self) -> str:
        if self._reward_variant == REWARD_VARIANT_SPARSE_OUTCOME:
            return "ego_player_sparse_round_outcome"
        if self._reward_variant == REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME:
            return "ego_player_dense_survival_helper_plus_sparse_outcome"
        if self._reward_variant == REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC:
            return "diagnostic_all_players_alive_after_one_source_tick"
        raise RuntimeError(f"unsupported reward_variant {self._reward_variant!r}")

    def _make_reward_space(self) -> dict[str, Any]:
        if self._reward_variant == REWARD_VARIANT_SPARSE_OUTCOME:
            low = -1.0
            high = 1.0
        elif self._reward_variant == REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME:
            low = -1.0
            high = float(self._policy_action_repeat_max + 1)
        elif self._reward_variant == REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC:
            low = 0.0
            high = 1.0
        else:
            raise RuntimeError(f"unsupported reward_variant {self._reward_variant!r}")
        return {
            "type": "Box",
            "shape": (),
            "dtype": "float32",
            "low": low,
            "high": high,
        }

    def _step_info(
        self,
        *,
        requested_action: int,
        executed_action: int,
        override_applied: bool,
        opponent_action: int,
        joint_action: np.ndarray,
        action_repeat_requested: int,
        action_repeat_executed: int,
        reward: float,
        done: bool,
        terminated: bool,
        truncated: bool,
        next_obs: dict[str, Any],
        batch: Any,
        sparse_outcome_reward_sum: float,
        dense_survival_helper_sum: float,
    ) -> dict[str, Any]:
        info = self._base_info()
        final_reward_map = None
        final_step_training_reward_map = None
        survival_reward_map = None
        source_terminal_reward_map = None
        if done:
            survival_reward_map = {
                "player_0": self._survival_reward_for_player(0),
                "player_1": self._survival_reward_for_player(1),
            }
            final_step_training_reward_map = {
                "player_0": self._scalar_reward_for_player(batch=batch, player_index=0),
                "player_1": self._scalar_reward_for_player(batch=batch, player_index=1),
            }
            if batch.final_reward is not None:
                source_terminal_reward_map = {
                    "player_0": float(batch.final_reward[0, 0]),
                    "player_1": float(batch.final_reward[0, 1]),
                }
                final_reward_map = source_terminal_reward_map
        info.update(
            {
                "step_index": int(self._step_index - 1),
                "tick_index": int(self._step_index),
                "adapter_timestep": int(self._step_index),
                "physical_step_index": int(self._physical_step_index),
                "source_tick_index": int(self._physical_step_index),
                "acting_player_id": self.ego_player_id,
                "active_player_id": self.ego_player_id,
                "next_active_player_id": self.ego_player_id,
                "controlled_player_id": self.ego_player_id,
                "ego_controlled_player_id": self.ego_player_id,
                "opponent_controlled_player_id": self.opponent_player_id,
                "requested_ego_action": int(requested_action),
                "executed_ego_action": int(executed_action),
                "ego_action_override_applied": bool(override_applied),
                "ego_action_straight_override_probability": self._override_probability,
                "ego_action_straight_override_action_id": self._override_action_id,
                "ego_action_straight_override_seed": self._override_seed,
                "control_stochasticity_schema_id": CONTROL_STOCHASTICITY_SCHEMA_ID,
                "policy_action_repeat_min": self._policy_action_repeat_min,
                "policy_action_repeat_max": self._policy_action_repeat_max,
                "policy_action_repeat_extra_probability": (
                    self._policy_action_repeat_extra_probability
                ),
                "policy_action_repeat_seed": self._policy_action_repeat_seed,
                "policy_action_repeat_requested": int(action_repeat_requested),
                "policy_action_repeat_executed": int(action_repeat_executed),
                "policy_action_repeat_extra_steps": int(action_repeat_executed - 1),
                "policy_observation_after_skipped_steps": int(action_repeat_executed - 1),
                "physical_decision_ms_total": float(
                    self._decision_ms * action_repeat_executed
                ),
                "control_noise_profile_id": self._control_noise_profile_id,
                "joint_action": {
                    "player_0": int(joint_action[0]),
                    "player_1": int(joint_action[1]),
                },
                "joint_action_schema_id": JOINT_ACTION_SCHEMA_ID,
                "opponent_action_id": int(opponent_action),
                "physical_env_advanced": True,
                "reward": float(reward),
                "trainer_reward": float(reward),
                "sparse_outcome_reward_for_ego": float(sparse_outcome_reward_sum),
                "dense_survival_helper_for_ego": float(dense_survival_helper_sum),
                "reward_player_id": self.ego_player_id,
                "reward_perspective": self._reward_perspective(),
                "scalar_training_reward_variant": self._reward_variant,
                "survival_reward_for_ego": self._survival_reward_for_player(
                    self.ego_player_index
                ),
                "done": bool(done),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "needs_reset": bool(self._needs_reset),
                "terminal_reason": self._terminal_reason_name(),
                "final_observation": _copy_lightzero_observation(next_obs) if done else None,
                "final_reward_map": final_reward_map,
                "final_step_training_reward_map": final_step_training_reward_map,
                "survival_reward_map": survival_reward_map,
                "source_terminal_reward_map": source_terminal_reward_map,
                "episode_training_return": float(self._episode_return) if done else None,
                "eval_episode_return": float(self._episode_return) if done else None,
                **self._public_outcome_info(),
                "trace_hash": self._trace_hash(joint_action=joint_action),
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
            "lightzero_env_type": LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_TYPE,
            "env_variant": SOURCE_STATE_FIXED_OPPONENT_ENV_VARIANT,
            "adapter_impl_id": SOURCE_STATE_VISUAL_SURVIVAL_ADAPTER_IMPL_ID,
            "lightzero_adapter_kind": "source_state_visual_survival_native_train_muzero",
            "runtime_topology": SOURCE_STATE_FIXED_OPPONENT_RUNTIME_TOPOLOGY,
            "underlying_env_class": SOURCE_STATE_FIXED_OPPONENT_UNDERLYING_ENV_CLASS,
            "runtime_env_impl_id": runtime_env_impl_id,
            "schema_hash": STACKED_SOURCE_STATE_GRAY64_SCHEMA_HASH,
            "public_env_contract_id": public_env_contract_id,
            "env_impl_id": runtime_env_impl_id,
            "ruleset_id": ruleset_id,
            "rules_hash": str(public_info["rules_hash"]),
            "decision_ms": float(self._decision_ms),
            "max_ticks": int(self._max_ticks),
            "player_count": 2,
            "player_ids": ("player_0", "player_1"),
            "observation_schema_id": STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
            "observation_schema_hash": STACKED_SOURCE_STATE_GRAY64_SCHEMA_HASH,
            "single_frame_schema_id": SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_ID,
            "single_frame_schema_hash": SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_HASH,
            "raw_observation_schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
            "raw_observation_schema_hash": SOURCE_STATE_CANVAS_LIKE_RAW64_SCHEMA_HASH,
            "raw_observation_available": True,
            "raw_observation_accessors": [
                "raw_observation()",
                "render('source_state_raw_visual_tensor')",
                "render('source_state_rgb_canvas_like')",
            ],
            "raw_observation_dtype": "uint8",
            "raw_observation_color_space": "RGB",
            "raw_observation_source": "render_source_state_rgb_canvas_like(frame_size=64)",
            "player_perspective_schema_id": None,
            "renderer_impl_id": SOURCE_STATE_CANVAS_LIKE_GRAY64_RENDERER_IMPL_ID,
            "raw_renderer_impl_id": SOURCE_STATE_RGB_CANVAS_LIKE_RENDERER_IMPL_ID,
            "truth_level": SOURCE_STATE_CANVAS_GRAY64_TRUTH_LEVEL,
            "source_fidelity_level": SOURCE_STATE_CANVAS_GRAY64_SOURCE_FIDELITY_LEVEL,
            "source_fidelity_claim": "source_state_backed_non_browser_pixel",
            "visual_surface": SOURCE_STATE_CANVAS_LIKE_GRAY64_SURFACE,
            "visual_truth_level": SOURCE_STATE_CANVAS_GRAY64_TRUTH_LEVEL,
            "visual_source_state_backed": SOURCE_STATE_CANVAS_GRAY64_SOURCE_STATE_BACKED,
            "debug_fidelity_only": False,
            "browser_pixel_fidelity": SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY,
            "uses_ale": SOURCE_STATE_CANVAS_GRAY64_USES_ALE,
            "ale_usage": "none",
            "shape": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
            "dtype": SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_DTYPE,
            "range": list(SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_VALUE_RANGE),
            "value_range": list(SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_VALUE_RANGE),
            "raw_frame_shape": list(SOURCE_STATE_CANVAS_LIKE_RAW64_SHAPE),
            "grayscale_frame_shape": list(SOURCE_STATE_CANVAS_GRAY64_SHAPE),
            "lightzero_payload_shape": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
            "model_observation_shape": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
            "frame_stack_owner": "curvyzero_source_state_survival_wrapper",
            "frame_stack_proof": (
                "wrapper_owned_canvas_like_rgb64_to_gray64_fifo_stack; "
                "not LightZero env-manager stacking"
            ),
            "reward_schema_id": self._reward_schema_id(),
            "reward_schema_hash": self._reward_schema_hash(),
            "scalar_training_reward_variant": self._reward_variant,
            "dense_survival_helper_enabled": (
                self._reward_variant == REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME
            ),
            "survival_length_is_eval_metric": True,
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
            "terminal_outcome_bonus": (
                1.0
                if self._reward_variant
                in (
                    REWARD_VARIANT_SPARSE_OUTCOME,
                    REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
                )
                else 0.0
            ),
            "loser_penalty": (
                -1.0
                if self._reward_variant
                in (
                    REWARD_VARIANT_SPARSE_OUTCOME,
                    REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
                )
                else 0.0
            ),
            "winner_bonus": (
                1.0
                if self._reward_variant
                in (
                    REWARD_VARIANT_SPARSE_OUTCOME,
                    REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
                )
                else 0.0
            ),
            "opponent_policy_id": "curvyzero_source_state_fixed_straight_opponent",
            "opponent_policy_kind": OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
            "opponent_training_relation": OPPONENT_TRAINING_RELATION_FIXED_STRAIGHT,
            "opponent_policy_version": "v0.2026-05-11",
            "opponent_policy_seed": self._seed,
            "episode_seed": self._episode_seed,
            "current_policy_self_play": CURRENT_POLICY_SELF_PLAY_CLAIM,
            "current_policy_self_play_blocker": CURRENT_POLICY_SELF_PLAY_BLOCKER,
            "trusted_current_policy_self_play": False,
            "simultaneous_game_theory_claim": False,
            "two_seat_self_play": False,
            "two_seat_self_play_status": SOURCE_STATE_FIXED_OPPONENT_TWO_SEAT_STATUS,
            "fixed_opponent_is_two_seat_self_play": False,
            "turn_commit_adapter": False,
            "control_stochasticity_schema_id": CONTROL_STOCHASTICITY_SCHEMA_ID,
            "control_noise_profile_id": self._control_noise_profile_id,
            "policy_action_repeat_min": self._policy_action_repeat_min,
            "policy_action_repeat_max": self._policy_action_repeat_max,
            "policy_action_repeat_extra_probability": (
                self._policy_action_repeat_extra_probability
            ),
            "policy_action_repeat_seed": self._policy_action_repeat_seed,
            "policy_action_repeat_semantics": (
                "one policy action is held for one or more physical source env "
                "steps before the next policy observation"
            ),
            "source_tick_index": int(self._physical_step_index),
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
        sampled_step = (
            self._step_index == 1
            or self._telemetry_stride == 1
            or self._step_index % self._telemetry_stride == 0
            or bool(timestep.done)
        )
        if not sampled_step:
            return
        info = timestep.info
        row = {
            "schema_id": "curvyzero_source_state_visual_survival_env_step/v0",
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
            "death_suppression_for_profile": info.get("death_suppression_for_profile"),
            "death_suppression_claim": info.get("death_suppression_claim"),
            "step_index": int(info.get("step_index", self._step_index - 1)),
            "physical_step_index": info.get("physical_step_index"),
            "source_tick_index": info.get("source_tick_index"),
            "scalar_action": info.get("scalar_action", info.get("requested_ego_action")),
            "joint_action_scalar": info.get("joint_action_scalar"),
            "joint_action_decode_rule": info.get("joint_action_decode_rule"),
            "centralized_joint_action_control": info.get(
                "centralized_joint_action_control"
            ),
            "true_competitive_self_play": info.get("true_competitive_self_play"),
            "ego_action": info.get("executed_ego_action"),
            "acting_player_id": info.get("acting_player_id"),
            "controlled_player_id": info.get("controlled_player_id"),
            "active_player_id": info.get("active_player_id"),
            "next_active_player_id": info.get("next_active_player_id"),
            "requested_ego_action": info.get("requested_ego_action"),
            "executed_ego_action": info.get("executed_ego_action"),
            "ego_action_override_applied": info.get("ego_action_override_applied"),
            "control_stochasticity_schema_id": info.get("control_stochasticity_schema_id"),
            "policy_action_repeat_min": info.get("policy_action_repeat_min"),
            "policy_action_repeat_max": info.get("policy_action_repeat_max"),
            "policy_action_repeat_extra_probability": info.get(
                "policy_action_repeat_extra_probability"
            ),
            "policy_action_repeat_seed": info.get("policy_action_repeat_seed"),
            "policy_action_repeat_requested": info.get("policy_action_repeat_requested"),
            "policy_action_repeat_executed": info.get("policy_action_repeat_executed"),
            "policy_action_repeat_extra_steps": info.get("policy_action_repeat_extra_steps"),
            "policy_observation_after_skipped_steps": info.get(
                "policy_observation_after_skipped_steps"
            ),
            "physical_decision_ms_total": info.get("physical_decision_ms_total"),
            "control_noise_profile_id": info.get("control_noise_profile_id"),
            "joint_action": info.get("joint_action"),
            "opponent_action_id": info.get("opponent_action_id"),
            "opponent_policy_id": info.get("opponent_policy_id"),
            "opponent_policy_version": info.get("opponent_policy_version"),
            "opponent_policy_kind": info.get("opponent_policy_kind"),
            "opponent_training_relation": info.get("opponent_training_relation"),
            "current_policy_self_play": info.get("current_policy_self_play"),
            "current_policy_self_play_blocker": info.get("current_policy_self_play_blocker"),
            "trusted_current_policy_self_play": info.get("trusted_current_policy_self_play"),
            "simultaneous_game_theory_claim": info.get("simultaneous_game_theory_claim"),
            "two_seat_self_play": info.get("two_seat_self_play"),
            "two_seat_self_play_status": info.get("two_seat_self_play_status"),
            "fixed_opponent_is_two_seat_self_play": info.get(
                "fixed_opponent_is_two_seat_self_play"
            ),
            "physical_env_advanced": True,
            "pending_action_count": 0,
            "reward": float(timestep.reward),
            "trainer_reward": info.get("trainer_reward"),
            "sparse_outcome_reward_for_ego": info.get("sparse_outcome_reward_for_ego"),
            "dense_survival_helper_for_ego": info.get("dense_survival_helper_for_ego"),
            "reward_player_id": info.get("reward_player_id"),
            "reward_perspective": info.get("reward_perspective"),
            "scalar_training_reward_variant": info.get("scalar_training_reward_variant"),
            "dense_survival_helper_enabled": info.get("dense_survival_helper_enabled"),
            "survival_length_is_eval_metric": info.get("survival_length_is_eval_metric"),
            "survival_reward_for_ego": info.get("survival_reward_for_ego"),
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
            "final_reward_map": info.get("final_reward_map"),
            "final_step_training_reward_map": info.get("final_step_training_reward_map"),
            "survival_reward_map": info.get("survival_reward_map"),
            "source_terminal_reward_map": info.get("source_terminal_reward_map"),
            "eval_episode_return": info.get("eval_episode_return"),
            "reward_schema_id": info.get("reward_schema_id"),
            "observation_schema_id": info.get("observation_schema_id"),
            "frame_stack_owner": info.get("frame_stack_owner"),
            "trace_hash": info.get("trace_hash"),
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

    def _public_outcome_info(self) -> dict[str, Any]:
        public_info = self._env._public_info()
        death_count = int(public_info["death_count"][0])
        death_player = [
            int(player)
            for player in public_info["death_player"][0, :death_count].tolist()
            if int(player) >= 0
        ]
        return {
            "winner_ids": tuple(f"player_{player}" for player in public_info["winner_ids"][0]),
            "loser_ids": tuple(f"player_{player}" for player in public_info["loser_ids"][0]),
            "death_player_ids": tuple(f"player_{player}" for player in death_player),
            "death_count": [death_count],
            "death_player": public_info["death_player"][:1].tolist(),
            "death_cause": public_info["death_cause"][:1].tolist(),
            "death_cause_name": public_info["death_cause_name"][:1].tolist(),
            "death_hit_owner": public_info["death_hit_owner"][:1].tolist(),
        }

    def _trace_hash(self, *, joint_action: np.ndarray) -> str:
        payload = {
            "seed": self._seed,
            "step_index": self._step_index,
            "physical_step_index": self._physical_step_index,
            "joint_action": [int(value) for value in joint_action.tolist()],
            "alive": self._env.state["alive"][0, :2].astype(bool).tolist(),
            "terminal_reason": self._terminal_reason_name(),
        }
        return stable_contract_hash(payload)

    def _next_seed(self, seed: int | None) -> int:
        if seed is not None:
            self._seed = int(seed)
            return self._seed
        if not self._dynamic_seed:
            return self._seed
        self._seed += 1
        return self._seed

    def _override_seed_for(self, reset_seed: int) -> int:
        if self._configured_override_seed is not None:
            return int(self._configured_override_seed)
        return int(reset_seed) + 1009

    def _policy_action_repeat_seed_for(self, reset_seed: int) -> int:
        if self._configured_repeat_seed is not None:
            return int(self._configured_repeat_seed)
        return int(reset_seed) + POLICY_ACTION_REPEAT_SEED_OFFSET


@ENV_REGISTRY.register(LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_TYPE)
class CurvyZeroSourceStateVisualSurvivalLightZeroEnv(
    CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv,
    BaseEnv,
):
    """Registered LightZero env using the source-state visual tensor."""

    config = dict(CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv.config)

    def __init__(self, cfg: Any | None = None):
        super().__init__(cfg)
        self.lightzero_env_type = LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_TYPE
        if gym is not None:
            reward_space = self._make_reward_space()
            self._action_space = gym.spaces.Discrete(ACTION_COUNT)
            self._observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=STACKED_SOURCE_STATE_GRAY64_SHAPE,
                dtype=np.float32,
            )
            self._reward_space = gym.spaces.Box(
                low=float(reward_space["low"]),
                high=float(reward_space["high"]),
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


class CurvyZeroSourceStateVisualJointActionLightZeroLocalEnv(
    CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv
):
    """Centralized 9-action wrapper: one scalar picks both player actions."""

    _default_reward_variant = REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC
    _allowed_reward_variants = SOURCE_STATE_JOINT_ACTION_REWARD_VARIANTS

    config = {
        **CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv.config,
        "env_id": LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_ID,
        "lightzero_env_type": LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_TYPE,
        "lightzero_import_names": LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_IMPORT_NAMES,
        "action_space_size": JOINT_ACTION_COUNT,
        "env_variant": SOURCE_STATE_JOINT_ACTION_ENV_VARIANT,
        "reward_variant": REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC,
        "reward_schema_id": ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_ID,
        "reward_schema_hash": ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_HASH,
        "runtime_topology": SOURCE_STATE_JOINT_ACTION_RUNTIME_TOPOLOGY,
        "two_seat_self_play": False,
        "two_seat_self_play_status": SOURCE_STATE_JOINT_ACTION_TRAINING_STATUS,
        "current_policy_self_play": False,
        "current_policy_self_play_blocker": (
            "centralized_joint_action_control_is_not_true_competitive_self_play"
        ),
        "trusted_current_policy_self_play": False,
        "simultaneous_game_theory_claim": False,
        "centralized_joint_action_control": True,
    }

    def __init__(self, cfg: Any | None = None):
        effective_cfg = _with_default_env_id(
            cfg,
            LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_ID,
        )
        super().__init__(effective_cfg)
        if self._override_probability != 0.0:
            raise ValueError("source_state_joint_action requires no ego action override")
        if (
            self._policy_action_repeat_min != 1
            or self._policy_action_repeat_max != 1
            or self._policy_action_repeat_extra_probability != 0.0
        ):
            raise ValueError("source_state_joint_action requires exactly one source tick per step")
        self._action_space = {"type": "Discrete", "n": JOINT_ACTION_COUNT}
        self._reward_space = {
            "type": "Box",
            "shape": (),
            "dtype": "float32",
            "low": 0.0,
            "high": 1.0,
        }

    @property
    def legal_actions(self) -> np.ndarray:
        return np.arange(JOINT_ACTION_COUNT, dtype=np.int64)

    def step(self, action: Any) -> LocalDebugVisualLightZeroTimestep:
        if not self._has_reset:
            raise RuntimeError("reset must be called before step")
        if self._needs_reset:
            raise RuntimeError("reset must be called before stepping after done")
        scalar_action = _validate_joint_action(action)
        player0_action, player1_action = _decode_joint_action(scalar_action)
        joint_action = np.array([[player0_action, player1_action]], dtype=np.int16)
        batch = self._env.step(joint_action, timer_advance_ms=self._decision_ms)
        self._last_batch = batch
        self._physical_step_index += 1
        reward = self._all_players_alive_reward()
        done = bool(batch.done[0])
        terminated = bool(batch.terminated[0])
        truncated = bool(batch.truncated[0])
        self._needs_reset = done
        self._episode_return += reward
        self._step_index += 1
        next_obs = self._lightzero_observation(needs_reset=done)
        info = self._step_info(
            requested_action=player0_action,
            executed_action=player0_action,
            override_applied=False,
            opponent_action=player1_action,
            joint_action=joint_action[0],
            action_repeat_requested=1,
            action_repeat_executed=1,
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            next_obs=next_obs,
            batch=batch,
            sparse_outcome_reward_sum=0.0,
            dense_survival_helper_sum=0.0,
        )
        info.update(
            {
                "scalar_action": int(scalar_action),
                "joint_action_scalar": int(scalar_action),
                "joint_action_decode_rule": "scalar // 3 -> player_0, scalar % 3 -> player_1",
                "centralized_joint_action_control": True,
                "true_competitive_self_play": False,
                "current_policy_self_play_blocker": (
                    "centralized_joint_action_control_is_not_true_competitive_self_play"
                ),
                "reward_perspective": "diagnostic_all_players_alive_after_one_source_tick",
                "source_ticks_advanced": 1,
                "pending_action_count": 0,
                "pending_actions_private": False,
            }
        )
        timestep = LocalDebugVisualLightZeroTimestep(next_obs, reward, done, info)
        self._write_telemetry_row(timestep=timestep)
        return timestep

    def random_action(self) -> int:
        rng = np.random.default_rng(self._seed + self._step_index)
        return int(rng.integers(JOINT_ACTION_COUNT))

    def _action_mask(self, *, active: bool) -> np.ndarray:
        if not active:
            return np.zeros(JOINT_ACTION_COUNT, dtype=np.int8)
        source_mask = self._env._action_mask()[0, :2].astype(np.int8, copy=False)
        joint_mask = np.zeros(JOINT_ACTION_COUNT, dtype=np.int8)
        for scalar_action in range(JOINT_ACTION_COUNT):
            player0_action, player1_action = _decode_joint_action(scalar_action)
            joint_mask[scalar_action] = np.int8(
                bool(source_mask[0, player0_action]) and bool(source_mask[1, player1_action])
            )
        return joint_mask

    def _all_players_alive_reward(self) -> float:
        alive = self._env.state["alive"][0, :2].astype(bool)
        return 1.0 if bool(np.all(alive)) else 0.0

    def _base_info(self) -> dict[str, Any]:
        info = super()._base_info()
        info.update(
            {
                "env_id": self.env_id,
                "lightzero_env_type": LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_TYPE,
                "env_variant": SOURCE_STATE_JOINT_ACTION_ENV_VARIANT,
                "adapter_impl_id": SOURCE_STATE_JOINT_ACTION_ADAPTER_IMPL_ID,
                "lightzero_adapter_kind": "source_state_visual_centralized_joint_action",
                "runtime_topology": SOURCE_STATE_JOINT_ACTION_RUNTIME_TOPOLOGY,
                "action_space_size": JOINT_ACTION_COUNT,
                "joint_action_scalar_count": JOINT_ACTION_COUNT,
                "joint_action_decode_rule": "scalar // 3 -> player_0, scalar % 3 -> player_1",
                "reward_schema_id": ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_ID,
                "reward_schema_hash": ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_HASH,
                "reward_contract": "diagnostic_all_players_alive_after_one_source_tick",
                "opponent_policy_id": "centralized_joint_action_controls_player_1",
                "opponent_policy_kind": "none_centralized_joint_action",
                "opponent_training_relation": "centralized_policy_controls_both_players",
                "opponent_policy_version": None,
                "current_policy_self_play": False,
                "current_policy_self_play_blocker": (
                    "centralized_joint_action_control_is_not_true_competitive_self_play"
                ),
                "trusted_current_policy_self_play": False,
                "simultaneous_game_theory_claim": False,
                "two_seat_self_play": False,
                "two_seat_self_play_status": SOURCE_STATE_JOINT_ACTION_TRAINING_STATUS,
                "fixed_opponent_is_two_seat_self_play": False,
                "turn_commit_adapter": False,
                "centralized_joint_action_control": True,
                "true_competitive_self_play": False,
                "policy_action_repeat_semantics": "exactly_one_source_tick_per_lightzero_step",
            }
        )
        return info

    def __repr__(self) -> str:
        return (
            "CurvyZeroSourceStateVisualJointActionLightZeroLocalEnv("
            f"env_id={self.env_id!r}, action_space_size={JOINT_ACTION_COUNT})"
        )


@ENV_REGISTRY.register(LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_TYPE)
class CurvyZeroSourceStateVisualJointActionLightZeroEnv(
    CurvyZeroSourceStateVisualJointActionLightZeroLocalEnv,
    BaseEnv,
):
    """Registered centralized joint-action LightZero env."""

    config = dict(CurvyZeroSourceStateVisualJointActionLightZeroLocalEnv.config)

    def __init__(self, cfg: Any | None = None):
        super().__init__(cfg)
        self.lightzero_env_type = LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_TYPE
        if gym is not None:
            self._action_space = gym.spaces.Discrete(JOINT_ACTION_COUNT)
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


def _validate_action(action: Any) -> int:
    try:
        action_id = int(np.asarray(action).item())
    except Exception as exc:
        raise ValueError(f"action must be scalar integer-like, got {action!r}") from exc
    if action_id < 0 or action_id >= ACTION_COUNT:
        raise ValueError(f"action must be in [0, {ACTION_COUNT}), got {action_id}")
    return action_id


def _validate_joint_action(action: Any) -> int:
    try:
        action_id = int(np.asarray(action).item())
    except Exception as exc:
        raise ValueError(f"joint action must be scalar integer-like, got {action!r}") from exc
    if action_id < 0 or action_id >= JOINT_ACTION_COUNT:
        raise ValueError(f"joint action must be in [0, {JOINT_ACTION_COUNT}), got {action_id}")
    return action_id


def _decode_joint_action(action_id: int) -> tuple[int, int]:
    return int(action_id // ACTION_COUNT), int(action_id % ACTION_COUNT)


def _copy_lightzero_observation(observation: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in observation.items():
        copied[key] = value.copy() if isinstance(value, np.ndarray) else value
    return copied


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _with_default_env_id(cfg: Any | None, env_id: str) -> Any:
    if cfg is None:
        return {"env_id": env_id}
    if isinstance(cfg, dict):
        copied = dict(cfg)
        copied.setdefault("env_id", env_id)
        return copied
    if getattr(cfg, "env_id", None) is None:
        try:
            setattr(cfg, "env_id", env_id)
        except Exception:
            pass
    return cfg


__all__ = [
    "CurvyZeroSourceStateVisualJointActionLightZeroEnv",
    "CurvyZeroSourceStateVisualJointActionLightZeroLocalEnv",
    "CurvyZeroSourceStateVisualSurvivalLightZeroEnv",
    "CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv",
    "ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_HASH",
    "ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_ID",
    "JOINT_ACTION_COUNT",
    "LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_ID",
    "LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_TYPE",
    "LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_IMPORT_NAMES",
    "LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_ID",
    "LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_TYPE",
    "LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_IMPORT_NAMES",
    "SOURCE_STATE_JOINT_ACTION_ADAPTER_IMPL_ID",
    "SOURCE_STATE_JOINT_ACTION_ENV_VARIANT",
    "SOURCE_STATE_JOINT_ACTION_RUNTIME_TOPOLOGY",
    "SOURCE_STATE_JOINT_ACTION_TRAINING_STATUS",
    "SOURCE_STATE_VISUAL_SURVIVAL_ADAPTER_IMPL_ID",
    "SOURCE_STATE_FIXED_OPPONENT_ENV_VARIANT",
    "SOURCE_STATE_FIXED_OPPONENT_RUNTIME_TOPOLOGY",
    "SOURCE_STATE_FIXED_OPPONENT_TWO_SEAT_STATUS",
    "SOURCE_STATE_FIXED_OPPONENT_UNDERLYING_ENV_CLASS",
    "SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_HASH",
    "SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_ID",
    "SOURCE_STATE_CANVAS_LIKE_GRAY64_SURFACE",
    "SOURCE_STATE_CANVAS_LIKE_RAW64_SCHEMA_HASH",
    "SOURCE_STATE_CANVAS_LIKE_RAW64_SHAPE",
    "STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID",
    "STACKED_SOURCE_STATE_GRAY64_SHAPE",
]
