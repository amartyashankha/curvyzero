"""Source-state-backed single-ego CurvyTron env for native LightZero MuZero."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from curvyzero.env import vector_runtime
from curvyzero.env.vector_multiplayer_env import ACTION_COUNT
from curvyzero.env.vector_multiplayer_env import DEFAULT_BODY_CAPACITY as VECTOR_DEFAULT_BODY_CAPACITY
from curvyzero.env.vector_multiplayer_env import JOINT_ACTION_SCHEMA_ID
from curvyzero.env.vector_multiplayer_env import NATURAL_BONUS_ENV_IMPL_ID
from curvyzero.env.vector_multiplayer_env import NATURAL_BONUS_RULESET_ID
from curvyzero.env.vector_multiplayer_env import PUBLIC_NATURAL_BONUS_ENV_CONTRACT_ID
from curvyzero.env.vector_multiplayer_env import SOURCE_PHYSICS_STEP_MS
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.env.trainer_contract import ACTION_ID_TO_SOURCE_MOVE
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
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
)
from curvyzero.env.vector_visual_observation import SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB
from curvyzero.env.vector_visual_observation import SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID
from curvyzero.env.vector_visual_observation import (
    BONUS_RENDER_MODE_CIRCLES_FAST,
    BONUS_RENDER_MODE_BROWSER_SPRITES,
    BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
    SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL,
    TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
    TRAIL_RENDER_MODE_BROWSER_LINES,
)
from curvyzero.env.vector_visual_observation import SourceStateBrowserLineTrailLayerCache
from curvyzero.env.vector_visual_observation import SourceStateCanvasGray64DirtyRenderCache
from curvyzero.env.vector_visual_observation import SourceStateGray64DownsampleScratch
from curvyzero.env.vector_visual_observation import render_source_state_canvas_gray64
from curvyzero.env.vector_visual_observation import (
    render_source_state_canvas_gray64_player_perspectives,
)
from curvyzero.env.vector_visual_observation import render_source_state_rgb_canvas_like
from curvyzero.env.observation_surface_contract import (
    DEFAULT_POLICY_OBSERVATION_BACKEND,
    POLICY_BONUS_RENDER_MODE,
    POLICY_OBSERVATION_BACKEND_CPU,
    POLICY_OBSERVATION_BACKEND_GPU,
    POLICY_OBSERVATION_BACKENDS,
    POLICY_OBSERVATION_CONTRACT_ID,
    POLICY_OBSERVATION_PERSPECTIVE,
    POLICY_OBSERVATION_PERSPECTIVE_OWNER,
    POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID,
    POLICY_STACK_SHAPE,
    POLICY_TRAIL_RENDER_MODE,
    policy_observation_surface,
)
from curvyzero.contracts.curvytron import (
    DEFAULT_LEARNER_SEAT_MODE,
    LEARNER_SEAT_MODE_CHOICES,
    LEARNER_SEAT_MODE_FIXED_PLAYER_0,
    LEARNER_SEAT_MODE_FIXED_PLAYER_1,
    LEARNER_SEAT_MODE_RANDOM_PER_EPISODE as _LEARNER_SEAT_MODE_RANDOM_PER_EPISODE,
    REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC,
    REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
    REWARD_VARIANT_SPARSE_OUTCOME,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
)

from curvyzero.training.curvyzero_debug_visual_lightzero_smoke import (
    LocalDebugVisualLightZeroTimestep,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    CURRENT_POLICY_SELF_PLAY_BLOCKER,
    CURRENT_POLICY_SELF_PLAY_CLAIM,
    OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
    OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
    OPPONENT_TRAINING_RELATION_FROZEN_LIGHTZERO_CHECKPOINT,
    OPPONENT_TRAINING_RELATION_FIXED_STRAIGHT,
)
from curvyzero.training.lightzero_checkpoint_opponent_provider import (
    snapshot_backed_lightzero_checkpoint_opponent_policy,
)
from curvyzero.training.opponent_mixture import (
    OPPONENT_MIXTURE_SCHEMA_ID,
    OPPONENT_MIXTURE_SELECTION_UNIT,
    parse_opponent_mixture_spec,
    select_opponent_mixture_entry,
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
STACKED_SOURCE_STATE_GRAY64_SHAPE = POLICY_STACK_SHAPE
SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SHAPE = (
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    3,
)
SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_DTYPE = "uint8"
SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_VALUE_RANGE = (0, 255)
SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES = TRAIL_RENDER_MODE_BROWSER_LINES
SOURCE_STATE_TRAIL_RENDER_MODE_BODY_CIRCLES_FAST = TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
SOURCE_STATE_DEFAULT_TRAIL_RENDER_MODE = POLICY_TRAIL_RENDER_MODE
SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES = (POLICY_TRAIL_RENDER_MODE,)
SOURCE_STATE_BONUS_RENDER_MODE_AUTO = "auto"
SOURCE_STATE_DEFAULT_BONUS_RENDER_MODE = POLICY_BONUS_RENDER_MODE
SOURCE_STATE_SUPPORTED_BONUS_RENDER_MODES = (POLICY_BONUS_RENDER_MODE,)
SOURCE_STATE_RAW_BONUS_RENDER_MODE = BONUS_RENDER_MODE_SIMPLE_SYMBOLS
SOURCE_STATE_POLICY_OBSERVATION_BACKENDS = POLICY_OBSERVATION_BACKENDS
SOURCE_STATE_BROWSER_TRAIL_SEMANTICS = "persistent_background_canvas_round_line_caps"
SOURCE_STATE_BROWSER_CLIENT_TRAIL_POINT_CAVEAT = (
    "vector/source snapshots expose persisted body points, not all client "
    "position-event trail points"
)
SOURCE_STATE_BROWSER_PIXEL_FIDELITY_CLAIM = "not_validated_against_browser_canvas"
SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SCHEMA_HASH = stable_contract_hash(
    {
        "schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
        "renderer_impl_id": SOURCE_STATE_RGB_CANVAS_LIKE_RENDERER_IMPL_ID,
        "default_trail_render_mode": SOURCE_STATE_DEFAULT_TRAIL_RENDER_MODE,
        "supported_trail_render_modes": list(SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES),
        "shape": list(SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SHAPE),
        "dtype": SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_DTYPE,
        "range": list(SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_VALUE_RANGE),
        "frame_size": SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        "source": (
            "render_source_state_rgb_canvas_like("
            f"frame_size={SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE}, "
            "trail_render_mode='browser_lines')"
        ),
        "role": "raw_full_frame_rgb_canvas_for_render_and_gray64_downsample_source",
        "truth_level": SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL,
        "browser_pixel_fidelity": False,
        "browser_pixel_fidelity_claim": SOURCE_STATE_BROWSER_PIXEL_FIDELITY_CLAIM,
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
DEFAULT_DECISION_SOURCE_FRAMES = 1
DEFAULT_DECISION_MS = SOURCE_PHYSICS_STEP_MS * DEFAULT_DECISION_SOURCE_FRAMES
DEFAULT_MAX_TICKS = 2_000
DEFAULT_POLICY_OBSERVATION_GPU_TRAIL_SLOT_SAFETY_FACTOR = 2
LEARNER_SEAT_ASSIGNMENT_SCHEMA_ID = "curvyzero_learner_seat_assignment/v0"
LEARNER_SEAT_MODE_RANDOM_PER_EPISODE = _LEARNER_SEAT_MODE_RANDOM_PER_EPISODE
LEARNER_SEAT_MODES = LEARNER_SEAT_MODE_CHOICES
OPPONENT_DEATH_MODE_NORMAL = "normal"
OPPONENT_DEATH_MODE_IMMORTAL = "immortal"
OPPONENT_DEATH_MODES = (
    OPPONENT_DEATH_MODE_NORMAL,
    OPPONENT_DEATH_MODE_IMMORTAL,
)
OPPONENT_RUNTIME_MODE_NORMAL = "normal"
OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP = "blank_canvas_noop"
OPPONENT_RUNTIME_MODES = (
    OPPONENT_RUNTIME_MODE_NORMAL,
    OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
)
DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY = 0.0
DEFAULT_POLICY_ACTION_REPEAT_MAX = 1
DEFAULT_POLICY_ACTION_REPEAT_MIN = 1
POLICY_ACTION_REPEAT_SEED_OFFSET = 2027
CONTROL_STOCHASTICITY_SCHEMA_ID = "curvyzero_policy_action_repeat_stochasticity/v0"
LEFT_ACTION_ID = 0
STRAIGHT_ACTION_ID = 1
RIGHT_ACTION_ID = 2
OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT = "proactive_wall_avoidant"
OPPONENT_TRAINING_RELATION_PROACTIVE_WALL_AVOIDANT = (
    "learner_vs_proactive_wall_avoidant"
)
PROACTIVE_WALL_AVOIDANT_OPPONENT_POLICY_ID = (
    "curvyzero_source_state_proactive_wall_avoidant_opponent"
)
PROACTIVE_WALL_AVOIDANT_OPPONENT_POLICY_VERSION = "v0.2026-05-13"
DEFAULT_OPPONENT_WALL_AVOIDANT_SAFE_MARGIN = 20.0
SOURCE_STATE_FIXED_OPPONENT_REWARD_VARIANTS = (
    REWARD_VARIANT_SPARSE_OUTCOME,
    REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
)
SOURCE_STATE_OPPONENT_POLICY_KINDS = (
    OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
    OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
    OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
)
OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION = (
    "none_centralized_joint_action"
)
SOURCE_STATE_JOINT_ACTION_OPPONENT_POLICY_KINDS = (
    OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION,
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
SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD = 1.0
SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_ID = (
    "curvyzero_survival_plus_bonus_no_outcome/v0"
)
SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA = {
    "schema_id": SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_ID,
    "dtype": "float32",
    "episode_unit": "one_round",
    "perspective": "ego_player",
    "alignment": "reward_t_plus_1_after_one_source_tick",
    "trainer_reward_terms": [
        "dense_alive_helper_for_ego_player",
        "same_step_bonus_pickup_helper_for_ego_player",
    ],
    "dense_alive_helper": 1.0,
    "bonus_pickup_reward_per_catch": SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD,
    "bonus_pickup_source": "bonus_catch_count_step[0, ego_player_index]",
    "sparse_round_outcome_schema_id": SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_ID,
    "sparse_round_outcome_is_telemetry_only": True,
    "terminal_outcome_bonus": 0.0,
    "loser_penalty": 0.0,
    "winner_bonus": 0.0,
    "draw_bonus": 0.0,
    "truncation_bonus": 0.0,
    "survival_length_metric_is_telemetry": True,
    "non_claims": [
        "not_zero_sum",
        "not_two_seat_self_play",
        "not_sparse_outcome_reward",
        "not_a_learning_claim",
    ],
}
SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_HASH = stable_contract_hash(
    SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA
)
SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA_ID = (
    "curvyzero_survival_plus_bonus_plus_outcome/v0"
)
SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA = {
    "schema_id": SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA_ID,
    "dtype": "float32",
    "episode_unit": "one_round",
    "perspective": "ego_player",
    "alignment": "reward_t_plus_1_after_one_source_tick",
    "trainer_reward_terms": [
        "dense_alive_helper_for_ego_player",
        "same_step_bonus_pickup_helper_for_ego_player",
        "terminal_sparse_outcome_scaled_by_episode_source_step_count",
    ],
    "dense_alive_helper": 1.0,
    "bonus_pickup_reward_per_catch": SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD,
    "bonus_pickup_source": "bonus_catch_count_step[0, ego_player_index]",
    "sparse_round_outcome_schema_id": SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_ID,
    "sparse_round_outcome_is_telemetry_only": False,
    "terminal_outcome_scale": "episode_source_step_count",
    "terminal_outcome_reward": "sparse_round_outcome * episode_source_step_count",
    "terminal_outcome_bonus": 1.0,
    "loser_penalty": -1.0,
    "winner_bonus": 1.0,
    "draw_bonus": 0.0,
    "truncation_bonus": 0.0,
    "survival_length_metric_is_telemetry": True,
    "non_claims": [
        "not_zero_sum_after_dense_helper_and_bonus",
        "not_two_seat_self_play",
        "not_a_learning_claim",
    ],
}
SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA_HASH = stable_contract_hash(
    SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA
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
        "raw_observation_schema_hash": SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SCHEMA_HASH,
        "default_trail_render_mode": SOURCE_STATE_DEFAULT_TRAIL_RENDER_MODE,
        "supported_trail_render_modes": list(SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES),
        "default_bonus_render_mode": SOURCE_STATE_DEFAULT_BONUS_RENDER_MODE,
        "supported_bonus_render_modes": list(SOURCE_STATE_SUPPORTED_BONUS_RENDER_MODES),
        "shape": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
        "dtype": SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_DTYPE,
        "range": list(SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_VALUE_RANGE),
        "frame_stack_owner": "curvyzero_source_state_survival_wrapper",
        "frame_stack_proof": (
            "wrapper_owned_raw_canvas_to_downsampled_gray64_fifo_stack; "
            "not LightZero env-manager stacking"
        ),
        "source_path": (
            "source-state canvas-like raw RGB canvas -> area-downsampled gray64 -> "
            "normalized FIFO stack"
        ),
    }
)


class CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv:
    """Native single-ego LightZero env over the reconstructed vector CurvyTron env."""

    _default_reward_variant = REWARD_VARIANT_SPARSE_OUTCOME
    _allowed_reward_variants = SOURCE_STATE_FIXED_OPPONENT_REWARD_VARIANTS
    _allowed_opponent_policy_kinds = SOURCE_STATE_OPPONENT_POLICY_KINDS

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
        "natural_bonus_spawn": True,
        "death_mode": vector_runtime.DEATH_MODE_NORMAL,
        "opponent_death_mode": OPPONENT_DEATH_MODE_NORMAL,
        "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_NORMAL,
        "opponent_policy_kind": OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        "supported_opponent_policy_kinds": SOURCE_STATE_OPPONENT_POLICY_KINDS,
        "disable_death_for_profile": False,
        "control_stochasticity_schema_id": CONTROL_STOCHASTICITY_SCHEMA_ID,
        "reward_variant": REWARD_VARIANT_SPARSE_OUTCOME,
        "source_state_trail_render_mode": SOURCE_STATE_DEFAULT_TRAIL_RENDER_MODE,
        "source_state_bonus_render_mode": SOURCE_STATE_DEFAULT_BONUS_RENDER_MODE,
        "policy_observation_backend": DEFAULT_POLICY_OBSERVATION_BACKEND,
        "supported_policy_observation_backends": SOURCE_STATE_POLICY_OBSERVATION_BACKENDS,
        "source_state_scalar_dirty_render_enabled": True,
        "default_trail_render_mode": SOURCE_STATE_DEFAULT_TRAIL_RENDER_MODE,
        "supported_trail_render_modes": SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES,
        "default_bonus_render_mode": SOURCE_STATE_DEFAULT_BONUS_RENDER_MODE,
        "supported_bonus_render_modes": SOURCE_STATE_SUPPORTED_BONUS_RENDER_MODES,
        "raw_observation_bonus_render_mode": SOURCE_STATE_RAW_BONUS_RENDER_MODE,
        "learner_seat_assignment_schema_id": LEARNER_SEAT_ASSIGNMENT_SCHEMA_ID,
        "learner_seat_mode": DEFAULT_LEARNER_SEAT_MODE,
        "supported_learner_seat_modes": LEARNER_SEAT_MODES,
        "policy_action_repeat_min": DEFAULT_POLICY_ACTION_REPEAT_MIN,
        "policy_action_repeat_max": DEFAULT_POLICY_ACTION_REPEAT_MAX,
        "policy_action_repeat_extra_probability": (
            DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
        ),
        "opponent_wall_avoidant_safe_margin": (
            DEFAULT_OPPONENT_WALL_AVOIDANT_SAFE_MARGIN
        ),
        "uses_ale": False,
    }

    def __init__(self, cfg: Any | None = None):
        cfg = cfg or {}
        self.env_id = str(
            _cfg_get(cfg, "env_id", LIGHTZERO_SOURCE_STATE_VISUAL_SURVIVAL_ENV_ID)
        )
        (
            self._learner_seat_mode,
            initial_ego_player_index,
        ) = _learner_seat_mode_from_cfg(cfg)
        self._set_learner_seat(initial_ego_player_index)
        self._opponent_policy_kind = str(
            _cfg_get(cfg, "opponent_policy_kind", OPPONENT_POLICY_KIND_FIXED_STRAIGHT)
        )
        self._configured_opponent_policy_kind = self._opponent_policy_kind
        allowed_opponent_policy_kinds = tuple(
            getattr(
                self,
                "_allowed_opponent_policy_kinds",
                SOURCE_STATE_OPPONENT_POLICY_KINDS,
            )
        )
        if self._opponent_policy_kind not in allowed_opponent_policy_kinds:
            raise ValueError(
                "opponent_policy_kind must be one of "
                f"{allowed_opponent_policy_kinds!r}; "
                f"got {self._opponent_policy_kind!r}"
            )
        self._seed = int(_cfg_get(cfg, "seed", 0))
        self._opponent_policy_seed = int(_cfg_get(cfg, "opponent_policy_seed", self._seed))
        self._configured_opponent_policy_seed = self._opponent_policy_seed
        self._opponent_wall_avoidant_safe_margin = float(
            _cfg_get(
                cfg,
                "opponent_wall_avoidant_safe_margin",
                DEFAULT_OPPONENT_WALL_AVOIDANT_SAFE_MARGIN,
            )
        )
        self._configured_opponent_wall_avoidant_safe_margin = (
            self._opponent_wall_avoidant_safe_margin
        )
        if self._opponent_wall_avoidant_safe_margin <= 0.0:
            raise ValueError("opponent_wall_avoidant_safe_margin must be positive")
        self._last_opponent_policy_sidecar: dict[str, Any] | None = None
        self._opponent_mixture = parse_opponent_mixture_spec(
            _cfg_get(cfg, "opponent_mixture", None)
        )
        if self._opponent_mixture is not None:
            for entry in self._opponent_mixture["entries"]:
                if entry["opponent_policy_kind"] not in allowed_opponent_policy_kinds:
                    raise ValueError(
                        "opponent mixture entry uses unsupported opponent_policy_kind "
                        f"{entry['opponent_policy_kind']!r}; expected one of "
                        f"{allowed_opponent_policy_kinds!r}"
                    )
        self._episode_opponent_mixture_entry: dict[str, Any] | None = None
        self._opponent_policy_cache: dict[str, Any] = {}
        self._opponent_assignment_context = _normalize_opponent_assignment_context(
            _cfg_get(cfg, "opponent_assignment_context", None)
        )
        self.opponent_policy = None
        if (
            self._opponent_mixture is None
            and self._opponent_policy_kind
            == OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT
        ):
            self.opponent_policy = _build_source_state_frozen_lightzero_opponent_policy(
                cfg,
                seed=self._opponent_policy_seed,
            )
        self._episode_seed = self._seed
        self._configured_dynamic_seed = bool(_cfg_get(cfg, "dynamic_seed", False))
        self._dynamic_seed = self._configured_dynamic_seed
        self._last_seed_call_dynamic_seed_arg: bool | None = None
        self._reset_index = 0
        (
            self._decision_source_frames,
            self._source_physics_step_ms,
            self._decision_ms,
        ) = _source_frame_decision_config(cfg)
        self._max_ticks = int(
            _cfg_get(cfg, "max_ticks", _cfg_get(cfg, "source_max_steps", DEFAULT_MAX_TICKS))
        )
        self._max_source_ticks = self._max_ticks * self._decision_source_frames
        self._source_state_trail_render_mode = _validate_trail_render_mode(
            _cfg_get(
                cfg,
                "source_state_trail_render_mode",
                SOURCE_STATE_DEFAULT_TRAIL_RENDER_MODE,
            )
        )
        self._source_state_bonus_render_mode = _validate_bonus_render_mode(
            _cfg_get(
                cfg,
                "source_state_bonus_render_mode",
                SOURCE_STATE_DEFAULT_BONUS_RENDER_MODE,
            )
        )
        self._model_bonus_render_mode = _resolve_model_bonus_render_mode(
            trail_render_mode=self._source_state_trail_render_mode,
            bonus_render_mode=self._source_state_bonus_render_mode,
        )
        self._policy_observation_backend = _validate_policy_observation_backend(
            _cfg_get(cfg, "policy_observation_backend", DEFAULT_POLICY_OBSERVATION_BACKEND)
        )
        self._jax_scalar_observation_renderer: _JaxScalarPolicyObservationRenderer | None = None
        if self._policy_observation_backend == POLICY_OBSERVATION_BACKEND_GPU:
            if (
                self._source_state_trail_render_mode
                != SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES
                or self._model_bonus_render_mode != BONUS_RENDER_MODE_SIMPLE_SYMBOLS
            ):
                raise ValueError(
                    "policy_observation_backend='jax_gpu' currently supports only "
                    "browser_lines + simple_symbols"
                )
            self._jax_scalar_observation_renderer = _JaxScalarPolicyObservationRenderer(
                player_count=2,
                min_trail_slots=self._policy_observation_gpu_min_trail_slots(cfg),
            )
        self._raw_observation_bonus_render_mode = SOURCE_STATE_RAW_BONUS_RENDER_MODE
        disable_death_for_profile = bool(_cfg_get(cfg, "disable_death_for_profile", False))
        configured_death_mode = str(
            _cfg_get(cfg, "death_mode", vector_runtime.DEATH_MODE_NORMAL)
        )
        if disable_death_for_profile:
            configured_death_mode = vector_runtime.DEATH_MODE_PROFILE_NO_DEATH
        if configured_death_mode not in vector_runtime.DEATH_MODES:
            raise ValueError("death_mode must be 'normal' or 'profile_no_death'")
        self._death_mode = configured_death_mode
        self._opponent_death_mode = str(
            _cfg_get(cfg, "opponent_death_mode", OPPONENT_DEATH_MODE_NORMAL)
        )
        self._configured_opponent_death_mode = self._opponent_death_mode
        if self._opponent_death_mode not in OPPONENT_DEATH_MODES:
            raise ValueError(
                f"opponent_death_mode must be one of {OPPONENT_DEATH_MODES!r}"
            )
        self._opponent_runtime_mode = str(
            _cfg_get(cfg, "opponent_runtime_mode", OPPONENT_RUNTIME_MODE_NORMAL)
        )
        self._configured_opponent_runtime_mode = self._opponent_runtime_mode
        if self._opponent_runtime_mode not in OPPONENT_RUNTIME_MODES:
            raise ValueError(
                f"opponent_runtime_mode must be one of {OPPONENT_RUNTIME_MODES!r}"
            )
        self._natural_bonus_spawn = bool(_cfg_get(cfg, "natural_bonus_spawn", True))
        self._disable_death_for_profile = (
            self._death_mode == vector_runtime.DEATH_MODE_PROFILE_NO_DEATH
        )
        telemetry_path = _cfg_get(cfg, "telemetry_path", None)
        self._telemetry_path = Path(str(telemetry_path)) if telemetry_path else None
        self._telemetry_stride = int(_cfg_get(cfg, "telemetry_stride", 1))
        if self._telemetry_stride < 1:
            raise ValueError("telemetry_stride must be at least 1")
        self._profile_env_timing_enabled = bool(
            _cfg_get(cfg, "profile_env_timing_enabled", False)
        )
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
        self._raw_frame = np.zeros(
            SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SHAPE,
            dtype=np.uint8,
        )
        self._raw_frame_dirty = True
        self._raw_frame_work = np.zeros_like(self._raw_frame)
        self._gray64_frame = np.zeros(SOURCE_STATE_CANVAS_GRAY64_SHAPE, dtype=np.uint8)
        self._gray64_perspective_frames = np.zeros(
            (2, *SOURCE_STATE_CANVAS_GRAY64_SHAPE),
            dtype=np.uint8,
        )
        self._opponent_gray64_frame = np.zeros(
            SOURCE_STATE_CANVAS_GRAY64_SHAPE,
            dtype=np.uint8,
        )
        self._normalized_frame = np.zeros(SOURCE_STATE_CANVAS_GRAY64_SHAPE, dtype=np.float32)
        self._opponent_normalized_frame = np.zeros(
            SOURCE_STATE_CANVAS_GRAY64_SHAPE,
            dtype=np.float32,
        )
        self._downsample_scratch = SourceStateGray64DownsampleScratch(
            SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        )
        self._scalar_dirty_render_enabled = bool(
            _cfg_get(cfg, "source_state_scalar_dirty_render_enabled", True)
        )
        self._scalar_trail_layer_cache = SourceStateBrowserLineTrailLayerCache(
            min_active_slots=1,
        )
        self._scalar_dirty_render_cache = SourceStateCanvasGray64DirtyRenderCache(
            player_count=2,
            profile_timing=self._profile_env_timing_enabled,
        )
        self._scalar_global_player_rgbs = (
            SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB,
            SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB,
        )
        self._stack = np.zeros(STACKED_SOURCE_STATE_GRAY64_SHAPE, dtype=np.float32)
        self._opponent_stack = np.zeros(STACKED_SOURCE_STATE_GRAY64_SHAPE, dtype=np.float32)
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

    def reset(
        self,
        seed: int | None = None,
        opponent_mixture: Any | None = None,
        opponent_assignment_context: Any | None = None,
    ) -> dict[str, Any]:
        if opponent_mixture is not None:
            self.set_opponent_mixture_for_next_reset(opponent_mixture)
        if opponent_assignment_context is not None:
            self._opponent_assignment_context = _normalize_opponent_assignment_context(
                opponent_assignment_context
            )
        reset_seed = self._next_seed(seed)
        self._episode_seed = reset_seed
        self._select_learner_seat_for_reset(reset_seed=reset_seed)
        self._select_episode_opponent(reset_seed=reset_seed)
        self._override_seed = self._override_seed_for(reset_seed)
        self._override_rng = np.random.default_rng(self._override_seed)
        self._policy_action_repeat_seed = self._policy_action_repeat_seed_for(reset_seed)
        self._policy_action_repeat_rng = np.random.default_rng(
            self._policy_action_repeat_seed
        )
        self._env = self._new_env(reset_seed)
        self._stack.fill(0.0)
        self._opponent_stack.fill(0.0)
        self._reset_scalar_render_cache()
        self._last_opponent_policy_sidecar = None
        self._needs_reset = False
        self._has_reset = True
        self._episode_return = 0.0
        self._step_index = 0
        self._physical_step_index = 0
        self._last_batch = self._env.reset(seed=reset_seed)
        self._scrub_blank_canvas_opponent()
        return self._lightzero_observation(needs_reset=False)

    def set_opponent_mixture_for_next_reset(self, mixture_spec: Any) -> None:
        mixture = parse_opponent_mixture_spec(mixture_spec)
        if mixture is None:
            raise ValueError("opponent mixture refresh requires a non-empty mixture")
        allowed_opponent_policy_kinds = tuple(
            getattr(
                self,
                "_allowed_opponent_policy_kinds",
                SOURCE_STATE_OPPONENT_POLICY_KINDS,
            )
        )
        for entry in mixture["entries"]:
            if entry["opponent_policy_kind"] not in allowed_opponent_policy_kinds:
                raise ValueError(
                    "opponent mixture entry uses unsupported opponent_policy_kind "
                    f"{entry['opponent_policy_kind']!r}; expected one of "
                    f"{allowed_opponent_policy_kinds!r}"
                )
            if (
                entry["opponent_policy_kind"]
                == OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT
                and not entry.get("opponent_checkpoint_path")
            ):
                raise ValueError(
                    "refreshed frozen opponent mixture entries require "
                    "resolved opponent_checkpoint_path"
                )

        self._opponent_mixture = mixture
        self._episode_opponent_mixture_entry = None
        self._opponent_policy_cache.clear()
        self.opponent_policy = None
        self._last_opponent_policy_sidecar = None

    def _set_learner_seat(self, ego_player_index: int) -> None:
        ego = int(ego_player_index)
        if ego not in (0, 1):
            raise ValueError("ego_player_index must be 0 or 1")
        self.ego_player_index = ego
        self.opponent_player_index = 1 - ego
        self.ego_player_id = f"player_{self.ego_player_index}"
        self.opponent_player_id = f"player_{self.opponent_player_index}"

    def _select_learner_seat_for_reset(self, *, reset_seed: int) -> None:
        if self._learner_seat_mode == LEARNER_SEAT_MODE_FIXED_PLAYER_1:
            self._set_learner_seat(1)
            return
        if self._learner_seat_mode == LEARNER_SEAT_MODE_FIXED_PLAYER_0:
            self._set_learner_seat(0)
            return
        reset_index = int(max(0, self._reset_index - 1))
        seed_sequence = np.random.SeedSequence(
            [
                int(reset_seed) & 0xFFFFFFFF,
                (int(reset_seed) >> 32) & 0xFFFFFFFF,
                reset_index & 0xFFFFFFFF,
                (reset_index >> 32) & 0xFFFFFFFF,
                0x51EA7,
            ]
        )
        rng = np.random.default_rng(seed_sequence)
        offset = int(rng.integers(0, 2, dtype=np.int64))
        self._set_learner_seat((offset + reset_index) % 2)

    def _select_episode_opponent(self, *, reset_seed: int) -> None:
        if self._opponent_mixture is None:
            self._episode_opponent_mixture_entry = None
            return
        reset_index = int(max(0, self._reset_index - 1))
        selected = select_opponent_mixture_entry(
            self._opponent_mixture,
            episode_seed=int(reset_seed),
            reset_index=reset_index,
        )
        self._episode_opponent_mixture_entry = selected
        self._opponent_policy_kind = str(selected["opponent_policy_kind"])
        self._opponent_runtime_mode = str(selected["opponent_runtime_mode"])
        self._opponent_death_mode = str(selected["opponent_death_mode"])
        self._opponent_policy_seed = int(
            selected.get("opponent_policy_seed", self._configured_opponent_policy_seed)
        )
        self._opponent_wall_avoidant_safe_margin = float(
            selected.get(
                "opponent_wall_avoidant_safe_margin",
                self._configured_opponent_wall_avoidant_safe_margin,
            )
        )
        self.opponent_policy = None
        if self._opponent_policy_kind == OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT:
            cache_key = str(selected["name"])
            if cache_key not in self._opponent_policy_cache:
                self._opponent_policy_cache[cache_key] = (
                    _build_source_state_frozen_lightzero_opponent_policy(
                        selected,
                        seed=self._opponent_policy_seed,
                    )
                )
            self.opponent_policy = self._opponent_policy_cache[cache_key]

    def step(self, action: Any) -> LocalDebugVisualLightZeroTimestep:
        if not self._has_reset:
            raise RuntimeError("reset must be called before step")
        if self._needs_reset:
            raise RuntimeError("reset must be called before stepping after done")
        timing_sec: dict[str, float] | None = (
            {} if self._profile_env_timing_enabled else None
        )
        step_started = time.perf_counter() if timing_sec is not None else 0.0
        requested_action = _validate_action(action)
        executed_action, override_applied = self._executed_ego_action(requested_action)
        opponent_started = time.perf_counter() if timing_sec is not None else 0.0
        opponent_action = self._opponent_action()
        if timing_sec is not None:
            timing_sec["opponent_action_sec"] = time.perf_counter() - opponent_started
        joint_action = np.full((1, 2), STRAIGHT_ACTION_ID, dtype=np.int16)
        joint_action[0, self.ego_player_index] = executed_action
        joint_action[0, self.opponent_player_index] = opponent_action
        action_repeat_requested = self._sample_policy_action_repeat()
        action_repeat_executed = 0
        reward = 0.0
        sparse_outcome_reward_sum = 0.0
        dense_survival_helper_sum = 0.0
        bonus_catch_count_step_sum = 0
        bonus_pickup_reward_sum = 0.0
        terminal_outcome_reward_sum = 0.0
        done = False
        terminated = False
        truncated = False
        batch = None
        physical_loop_started = time.perf_counter() if timing_sec is not None else 0.0
        vector_step_sec = 0.0
        reward_sec = 0.0
        for _ in range(action_repeat_requested):
            vector_started = time.perf_counter() if timing_sec is not None else 0.0
            self._scrub_blank_canvas_opponent()
            batch = self._env.step(
                joint_action,
                timer_advance_ms=self._decision_ms,
                disabled_player_mask=self._disabled_player_mask_for_step(),
            )
            self._scrub_blank_canvas_opponent()
            if timing_sec is not None:
                vector_step_sec += time.perf_counter() - vector_started
            self._last_batch = batch
            action_repeat_executed += 1
            self._physical_step_index += 1
            reward_started = time.perf_counter() if timing_sec is not None else 0.0
            components = self._reward_components_for_player(
                batch=batch,
                player_index=self.ego_player_index,
            )
            sparse_outcome_reward_sum += components["sparse_outcome_reward"]
            dense_survival_helper_sum += components["dense_survival_helper"]
            bonus_catch_count_step_sum += int(components["bonus_catch_count_step"])
            bonus_pickup_reward_sum += components["bonus_pickup_reward"]
            terminal_outcome_reward_sum += components["terminal_outcome_reward"]
            reward += components["trainer_reward"]
            done = bool(batch.done[0])
            terminated = bool(batch.terminated[0])
            truncated = bool(batch.truncated[0])
            if timing_sec is not None:
                reward_sec += time.perf_counter() - reward_started
            if done:
                break
        if timing_sec is not None:
            timing_sec["physical_loop_sec"] = time.perf_counter() - physical_loop_started
            timing_sec["vector_step_sec"] = vector_step_sec
            timing_sec["reward_sec"] = reward_sec
        if batch is None:
            raise RuntimeError("policy action repeat produced no physical env step")
        self._needs_reset = done
        self._episode_return += reward
        self._step_index += 1
        observation_started = time.perf_counter() if timing_sec is not None else 0.0
        next_obs = self._lightzero_observation(needs_reset=done)
        if timing_sec is not None:
            timing_sec["observation_sec"] = time.perf_counter() - observation_started
            timing_sec["step_total_before_info_sec"] = time.perf_counter() - step_started
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
            bonus_catch_count_step_sum=bonus_catch_count_step_sum,
            bonus_pickup_reward_sum=bonus_pickup_reward_sum,
            terminal_outcome_reward_sum=terminal_outcome_reward_sum,
            profile_env_timing_sec=timing_sec,
        )
        timestep = LocalDebugVisualLightZeroTimestep(next_obs, reward, done, info)
        self._write_telemetry_row(timestep=timestep)
        return timestep

    def close(self) -> None:
        return None

    def seed(self, seed: int, dynamic_seed: bool | None = None) -> None:
        previous_seed = self._seed
        self._seed = int(seed)
        self._episode_seed = self._seed
        self._last_seed_call_dynamic_seed_arg = (
            bool(dynamic_seed) if dynamic_seed is not None else None
        )
        self._dynamic_seed = self._configured_dynamic_seed
        if self._seed != previous_seed:
            self._reset_index = 0
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
        if self._raw_frame_dirty:
            render_source_state_rgb_canvas_like(
                self._render_state_view(),
                row=0,
                out=self._raw_frame,
                trail_render_mode=self._source_state_trail_render_mode,
                bonus_render_mode=self._raw_observation_bonus_render_mode,
            )
            self._raw_frame_dirty = False
        return self._raw_frame.copy()

    def human_rgb_observation(
        self,
        *,
        frame_size: int = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    ) -> np.ndarray | None:
        if not self._has_reset:
            return None
        return render_source_state_rgb_canvas_like(
            self._render_state_view(),
            row=0,
            frame_size=frame_size,
            trail_render_mode=self._source_state_trail_render_mode,
            bonus_render_mode=self._raw_observation_bonus_render_mode,
        )

    def __repr__(self) -> str:
        return (
            "CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv("
            f"env_id={self.env_id!r}, "
            f"ego_player_id={self.ego_player_id!r}, "
            f"opponent_player_id={self.opponent_player_id!r}, "
            f"opponent_policy_kind={self._opponent_policy_kind!r})"
        )

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
            death_immunity_player_ids=self._death_immunity_player_ids(),
            natural_bonus_spawn=self._natural_bonus_spawn,
        )

    def _policy_observation_gpu_min_trail_slots(self, cfg: Any) -> int:
        configured = _cfg_get(cfg, "policy_observation_gpu_min_trail_slots", None)
        if configured is not None:
            value = int(configured)
            if value < 1:
                raise ValueError("policy_observation_gpu_min_trail_slots must be positive")
            return value
        expected_visual_points = (
            DEFAULT_POLICY_OBSERVATION_GPU_TRAIL_SLOT_SAFETY_FACTOR
            * self._max_source_ticks
        )
        return _ceil_power_of_two(
            min(int(VECTOR_DEFAULT_BODY_CAPACITY), max(1, int(expected_visual_points)))
        )

    def _death_immunity_player_ids(self) -> tuple[int, ...]:
        if (
            self._opponent_death_mode == OPPONENT_DEATH_MODE_IMMORTAL
            or self._opponent_runtime_mode == OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP
        ):
            return (self.opponent_player_index,)
        return ()

    def _blank_canvas_noop_enabled(self) -> bool:
        return self._opponent_runtime_mode == OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP

    def _disabled_player_mask_for_step(self) -> np.ndarray | None:
        if not self._blank_canvas_noop_enabled():
            return None
        mask = np.zeros((1, 2), dtype=bool)
        mask[0, self.opponent_player_index] = True
        return mask

    def _scrub_blank_canvas_opponent(self) -> None:
        if not self._blank_canvas_noop_enabled():
            return
        state = self._env.state
        player = self.opponent_player_index
        state["present"][:, player] = True
        state["alive"][:, player] = True
        if "death_tick" in state:
            state["death_tick"][:, player] = -1
        for name in ("printing", "print_manager_active", "has_draw_cursor"):
            if name in state:
                state[name][:, player] = False
        for name in ("speed", "base_speed", "angular_velocity_per_ms"):
            if name in state:
                state[name][:, player] = 0.0
        if "base_angular_velocity_per_ms" in state:
            state["base_angular_velocity_per_ms"][:, player] = 0.0
        for name in ("radius", "base_radius"):
            if name in state:
                state[name][:, player] = 0.0
        for name in ("body_count", "live_body_num", "visible_trail_count"):
            if name in state:
                state[name][:, player] = 0
        for name in ("has_visible_trail_last", "has_visual_trail_last"):
            if name in state:
                state[name][:, player] = False
        if "bonus_catch_count_step" in state:
            state["bonus_catch_count_step"][:, player] = 0
        self._clear_owner_slots(state, player=player)
        self._remove_player_from_death_lists(state, player=player)

    def _clear_owner_slots(self, state: dict[str, np.ndarray], *, player: int) -> None:
        if "body_owner" in state and "body_active" in state:
            body_mask = state["body_active"] & (state["body_owner"] == player)
            state["body_active"][body_mask] = False
            state["body_owner"][body_mask] = -1
            if "body_radius" in state:
                state["body_radius"][body_mask] = 0.0
            if "body_pos" in state:
                state["body_pos"][body_mask] = 0.0
            if "body_num" in state:
                state["body_num"][body_mask] = -1
            if "body_insert_tick" in state:
                state["body_insert_tick"][body_mask] = -1
            if "body_insert_kind" in state:
                state["body_insert_kind"][body_mask] = -1
            if "body_break_before" in state:
                state["body_break_before"][body_mask] = False
            if "world_body_count" in state:
                state["world_body_count"][:] = state["body_active"].sum(axis=1).astype(
                    state["world_body_count"].dtype,
                    copy=False,
                )
        if "visual_trail_owner" in state and "visual_trail_active" in state:
            visual_mask = (
                state["visual_trail_active"] & (state["visual_trail_owner"] == player)
            )
            state["visual_trail_active"][visual_mask] = False
            state["visual_trail_owner"][visual_mask] = -1
            if "visual_trail_radius" in state:
                state["visual_trail_radius"][visual_mask] = 0.0
            if "visual_trail_pos" in state:
                state["visual_trail_pos"][visual_mask] = 0.0
            if "visual_trail_break_before" in state:
                state["visual_trail_break_before"][visual_mask] = False

    def _remove_player_from_death_lists(
        self,
        state: dict[str, np.ndarray],
        *,
        player: int,
    ) -> None:
        if "death_player" not in state or "death_count" not in state:
            return
        for row in range(state["death_player"].shape[0]):
            count = int(state["death_count"][row])
            if count <= 0:
                continue
            keep = [
                index
                for index in range(count)
                if int(state["death_player"][row, index]) != player
            ]
            if len(keep) == count:
                continue
            for target, source in enumerate(keep):
                state["death_player"][row, target] = state["death_player"][row, source]
                if "death_cause" in state:
                    state["death_cause"][row, target] = state["death_cause"][row, source]
                if "death_hit_owner" in state:
                    state["death_hit_owner"][row, target] = state["death_hit_owner"][
                        row,
                        source,
                    ]
            for index in range(len(keep), count):
                state["death_player"][row, index] = -1
                if "death_cause" in state:
                    state["death_cause"][row, index] = vector_runtime.DEATH_CAUSE_NONE
                if "death_hit_owner" in state:
                    state["death_hit_owner"][row, index] = -1
            state["death_count"][row] = len(keep)

    def _render_state_view(self) -> dict[str, np.ndarray]:
        if not self._blank_canvas_noop_enabled():
            return self._env.state
        view = dict(self._env.state)
        player = self.opponent_player_index
        present = self._env.state["present"].copy()
        alive = self._env.state["alive"].copy()
        present[:, player] = False
        alive[:, player] = False
        view["present"] = present
        view["alive"] = alive
        if "radius" in view:
            radius = self._env.state["radius"].copy()
            radius[:, player] = 0.0
            view["radius"] = radius
        self._mask_owner_slots_for_render(view, player=player)
        return view

    def _mask_owner_slots_for_render(
        self,
        view: dict[str, np.ndarray],
        *,
        player: int,
    ) -> None:
        if "body_owner" in view and "body_active" in view:
            owner = self._env.state["body_owner"]
            active = self._env.state["body_active"]
            body_mask = active & (owner == player)
            if bool(body_mask.any()):
                view["body_active"] = active.copy()
                view["body_active"][body_mask] = False
                view["body_owner"] = owner.copy()
                view["body_owner"][body_mask] = -1
        if "visual_trail_owner" in view and "visual_trail_active" in view:
            owner = self._env.state["visual_trail_owner"]
            active = self._env.state["visual_trail_active"]
            visual_mask = active & (owner == player)
            if bool(visual_mask.any()):
                view["visual_trail_active"] = active.copy()
                view["visual_trail_active"][visual_mask] = False
                view["visual_trail_owner"] = owner.copy()
                view["visual_trail_owner"][visual_mask] = -1

    def _lightzero_observation(self, *, needs_reset: bool) -> dict[str, Any]:
        stack = self._update_stack()
        return {
            "observation": stack.copy(),
            "action_mask": self._action_mask(active=not needs_reset),
            "to_play": -1,
            "timestep": int(self._step_index),
        }

    def _update_stack(self) -> np.ndarray:
        state_view = self._render_state_view()
        if self._policy_observation_backend == POLICY_OBSERVATION_BACKEND_GPU:
            if self._jax_scalar_observation_renderer is None:
                raise RuntimeError("jax_gpu observation backend was not initialized")
            raw_frames = self._jax_scalar_observation_renderer.render_player_perspectives(
                state_view,
                out=self._gray64_perspective_frames,
            )
            np.copyto(self._gray64_frame, raw_frames[self.ego_player_index])
            np.copyto(
                self._opponent_gray64_frame,
                raw_frames[self.opponent_player_index],
            )
            self._raw_frame_dirty = True
            gray64 = self._gray64_frame
        elif self._should_use_scalar_dirty_render_cache():
            raw_frames = render_source_state_canvas_gray64_player_perspectives(
                state_view,
                row=0,
                out=self._gray64_perspective_frames,
                rgb_base_out=self._raw_frame,
                rgb_work_out=self._raw_frame_work,
                trail_cache=self._scalar_trail_layer_cache,
                dirty_render_cache=self._scalar_dirty_render_cache,
                downsample_scratch=self._downsample_scratch,
                player_rgbs=(
                    _player_perspective_rgb_palette(
                        state_view,
                        row=0,
                        controlled_player=0,
                        player_count=2,
                    ),
                    _player_perspective_rgb_palette(
                        state_view,
                        row=0,
                        controlled_player=1,
                        player_count=2,
                    ),
                ),
                trail_render_mode=self._source_state_trail_render_mode,
                bonus_render_mode=self._model_bonus_render_mode,
            )
            np.copyto(self._gray64_frame, raw_frames[self.ego_player_index])
            np.copyto(
                self._opponent_gray64_frame,
                raw_frames[self.opponent_player_index],
            )
            if self._scalar_dirty_render_cache.initialized:
                self._raw_frame_dirty = True
            gray64 = self._gray64_frame
        else:
            gray64 = render_source_state_canvas_gray64(
                state_view,
                row=0,
                out=self._gray64_frame,
                rgb_out=self._raw_frame,
                downsample_scratch=self._downsample_scratch,
                player_rgb=_player_perspective_rgb_palette(
                    state_view,
                    row=0,
                    controlled_player=self.ego_player_index,
                    player_count=2,
                ),
                trail_render_mode=self._source_state_trail_render_mode,
                bonus_render_mode=self._model_bonus_render_mode,
            )
            render_source_state_canvas_gray64(
                state_view,
                row=0,
                out=self._opponent_gray64_frame,
                rgb_out=self._raw_frame_work,
                downsample_scratch=self._downsample_scratch,
                player_rgb=_player_perspective_rgb_palette(
                    state_view,
                    row=0,
                    controlled_player=self.opponent_player_index,
                    player_count=2,
                ),
                trail_render_mode=self._source_state_trail_render_mode,
                bonus_render_mode=self._model_bonus_render_mode,
            )
            self._raw_frame_dirty = (
                self._model_bonus_render_mode != self._raw_observation_bonus_render_mode
            )
        np.multiply(
            gray64,
            np.float32(1.0 / 255.0),
            out=self._normalized_frame,
            casting="unsafe",
        )
        self._stack[:-1] = self._stack[1:]
        self._stack[-1] = self._normalized_frame[0]
        np.multiply(
            self._opponent_gray64_frame,
            np.float32(1.0 / 255.0),
            out=self._opponent_normalized_frame,
            casting="unsafe",
        )
        self._opponent_stack[:-1] = self._opponent_stack[1:]
        self._opponent_stack[-1] = self._opponent_normalized_frame[0]
        return self._stack

    def _should_use_scalar_dirty_render_cache(self) -> bool:
        return (
            self._scalar_dirty_render_enabled
            and self._source_state_trail_render_mode
            == SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES
        )

    def _reset_scalar_render_cache(self) -> None:
        self._scalar_trail_layer_cache.reset()
        self._scalar_dirty_render_cache.reset()

    def _action_mask(self, *, active: bool) -> np.ndarray:
        if not active:
            return np.zeros(ACTION_COUNT, dtype=np.int8)
        return self._env._action_mask()[0, self.ego_player_index].astype(np.int8, copy=True)

    def _opponent_action(self) -> int:
        if self._blank_canvas_noop_enabled():
            self._last_opponent_policy_sidecar = {
                "opponent_runtime_mode": self._opponent_runtime_mode,
                "action_ignored": True,
            }
            return STRAIGHT_ACTION_ID
        if self._opponent_policy_kind == OPPONENT_POLICY_KIND_FIXED_STRAIGHT:
            self._last_opponent_policy_sidecar = None
            return STRAIGHT_ACTION_ID
        if (
            self._opponent_policy_kind
            == OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT
        ):
            return self._proactive_wall_avoidant_opponent_action()
        if self.opponent_policy is None:
            raise RuntimeError("frozen LightZero opponent policy was not initialized")
        legal = self._env._action_mask()[:, :2].astype(bool, copy=True)
        opponent_mask = np.zeros((1, 2), dtype=bool)
        opponent_mask[0, self.opponent_player_index] = True
        observation = np.zeros((1, 2, *STACKED_SOURCE_STATE_GRAY64_SHAPE), dtype=np.float32)
        observation[0, self.ego_player_index] = self._stack
        observation[0, self.opponent_player_index] = self._opponent_stack
        selection = self.opponent_policy.select_actions(
            legal,
            opponent_mask,
            decision_index=int(self._step_index),
            observation=observation,
        )
        self._last_opponent_policy_sidecar = selection.sidecar()
        return int(selection.actions[0, self.opponent_player_index])

    def _proactive_wall_avoidant_opponent_action(self) -> int:
        legal_mask = self._env._action_mask()[0, self.opponent_player_index].astype(
            bool,
            copy=True,
        )
        if not bool(legal_mask.any()):
            action = STRAIGHT_ACTION_ID
            fallback = "inactive_no_legal_action_slots"
            telemetry = self._wall_avoidant_geometry_metadata(action)
        else:
            preferred, telemetry = self._wall_avoidant_preferred_action()
            action, fallback = self._legal_opponent_action(preferred, legal_mask)
        telemetry.update(
            {
                "legal_action_mask": legal_mask.tolist(),
                "selected_action_id": int(action),
                "selected_from_legal_actions": bool(
                    0 <= int(action) < legal_mask.shape[0] and legal_mask[int(action)]
                ),
                "legal_fallback": fallback,
            }
        )
        self._last_opponent_policy_sidecar = {
            "policy_id": PROACTIVE_WALL_AVOIDANT_OPPONENT_POLICY_ID,
            "policy_version": PROACTIVE_WALL_AVOIDANT_OPPONENT_POLICY_VERSION,
            "seed": int(self._opponent_policy_seed),
            "metadata_only": True,
            "trainer_replay_claim": False,
            "learned_observation_claim": False,
            "policy_metadata": {
                "opponent_kind": OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
                "opponent_policy_variant": "proactive_force_field",
                "safe_margin": float(self._opponent_wall_avoidant_safe_margin),
                "uses_legal_left_straight_right_actions_only": True,
                "no_teleport_or_bounce": True,
                **telemetry,
            },
        }
        return int(action)

    def _legal_opponent_action(
        self,
        preferred_action: int,
        legal_mask: np.ndarray,
    ) -> tuple[int, str | None]:
        preferred = int(preferred_action)
        if 0 <= preferred < legal_mask.shape[0] and bool(legal_mask[preferred]):
            return preferred, None
        if (
            0 <= STRAIGHT_ACTION_ID < legal_mask.shape[0]
            and bool(legal_mask[STRAIGHT_ACTION_ID])
        ):
            return STRAIGHT_ACTION_ID, "preferred_illegal_used_straight"
        legal_ids = np.flatnonzero(legal_mask)
        if legal_ids.size:
            return int(legal_ids[0]), "preferred_illegal_used_first_legal"
        return STRAIGHT_ACTION_ID, "inactive_no_legal_action_slots"

    def _wall_avoidant_preferred_action(self) -> tuple[int, dict[str, Any]]:
        telemetry = self._wall_avoidant_geometry_metadata(STRAIGHT_ACTION_ID)
        if bool(telemetry["borderless"]):
            telemetry["selection_reason"] = "borderless_active"
            return STRAIGHT_ACTION_ID, telemetry
        if float(telemetry["nearest_wall_clearance"]) > float(
            self._opponent_wall_avoidant_safe_margin
        ):
            telemetry["selection_reason"] = "outside_safe_margin"
            return STRAIGHT_ACTION_ID, telemetry

        target_x = float(telemetry["away_vector"][0])
        target_y = float(telemetry["away_vector"][1])
        left_heading = self._wall_avoidant_post_action_heading(LEFT_ACTION_ID)
        right_heading = self._wall_avoidant_post_action_heading(RIGHT_ACTION_ID)
        left_score = float(np.cos(left_heading) * target_x + np.sin(left_heading) * target_y)
        right_score = float(
            np.cos(right_heading) * target_x + np.sin(right_heading) * target_y
        )
        telemetry.update(
            {
                "left_alignment_score": left_score,
                "right_alignment_score": right_score,
                "selection_reason": "inside_safe_margin_turn_toward_interior",
            }
        )
        if right_score > left_score:
            return RIGHT_ACTION_ID, telemetry
        return LEFT_ACTION_ID, telemetry

    def _wall_avoidant_geometry_metadata(self, action_id: int) -> dict[str, Any]:
        state = self._env.state
        player = self.opponent_player_index
        pos = state["pos"][0, player]
        radius = float(state["radius"][0, player])
        map_size = float(state["map_size"][0])
        clearances = np.asarray(
            [
                float(pos[0] - radius),
                float(map_size - (pos[0] + radius)),
                float(pos[1] - radius),
                float(map_size - (pos[1] + radius)),
            ],
            dtype=np.float64,
        )
        nearest_wall = int(clearances.argmin())
        safe_margin = float(self._opponent_wall_avoidant_safe_margin)
        danger_left = max(0.0, safe_margin - float(clearances[0]))
        danger_right = max(0.0, safe_margin - float(clearances[1]))
        danger_top = max(0.0, safe_margin - float(clearances[2]))
        danger_bottom = max(0.0, safe_margin - float(clearances[3]))
        away_x = danger_left - danger_right
        away_y = danger_top - danger_bottom
        away_norm = float(np.hypot(away_x, away_y))
        heading = float(state["heading"][0, player])
        if away_norm <= 1e-9:
            away = (float(np.cos(heading)), float(np.sin(heading)))
        else:
            away = (float(away_x / away_norm), float(away_y / away_norm))
        return {
            "safe_margin": safe_margin,
            "position": [float(pos[0]), float(pos[1])],
            "radius": radius,
            "map_size": map_size,
            "heading": heading,
            "post_action_heading": float(
                self._wall_avoidant_post_action_heading(action_id)
            ),
            "clearance_left": float(clearances[0]),
            "clearance_right": float(clearances[1]),
            "clearance_top": float(clearances[2]),
            "clearance_bottom": float(clearances[3]),
            "nearest_wall_index": nearest_wall,
            "nearest_wall_name": ("left", "right", "top", "bottom")[nearest_wall],
            "nearest_wall_clearance": float(clearances[nearest_wall]),
            "away_vector": [away[0], away[1]],
            "borderless": bool(state["borderless"][0]),
        }

    def _wall_avoidant_post_action_heading(self, action_id: int) -> float:
        state = self._env.state
        player = self.opponent_player_index
        source_move = float(ACTION_ID_TO_SOURCE_MOVE[int(action_id)])
        if "inverse" in state and bool(state["inverse"][0, player]):
            source_move = -source_move
        return float(
            state["heading"][0, player]
            + source_move
            * float(state["angular_velocity_per_ms"][0, player])
            * float(self._decision_ms)
        )

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

    def _bonus_catch_count_for_player(self, *, batch: Any, player_index: int) -> int:
        info = getattr(batch, "info", None)
        counts = info.get("bonus_catch_count_step") if isinstance(info, dict) else None
        if counts is None:
            raise RuntimeError("bonus_catch_count_step missing from source-state batch info")
        counts_array = np.asarray(counts)
        if counts_array.ndim != 2 or counts_array.shape[0] < 1:
            raise RuntimeError(
                "bonus_catch_count_step must have shape [batch, player]"
            )
        if not 0 <= int(player_index) < counts_array.shape[1]:
            raise RuntimeError(
                f"bonus_catch_count_step missing player index {player_index}"
            )
        return int(counts_array[0, int(player_index)])

    def _bonus_pickup_reward_for_player(
        self,
        *,
        batch: Any,
        player_index: int,
    ) -> float:
        catch_count = self._bonus_catch_count_for_player(
            batch=batch,
            player_index=player_index,
        )
        return float(catch_count) * SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD

    def _reward_components_for_player(
        self,
        *,
        batch: Any,
        player_index: int,
    ) -> dict[str, float]:
        sparse_outcome = float(batch.reward[0, player_index])
        dense_helper = 0.0
        bonus_catch_count = 0
        bonus_pickup_reward = 0.0
        terminal_outcome_reward = 0.0
        if self._reward_variant == REWARD_VARIANT_SPARSE_OUTCOME:
            trainer_reward = sparse_outcome
        elif self._reward_variant == REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME:
            dense_helper = self._survival_reward_for_player(player_index)
            trainer_reward = sparse_outcome + dense_helper
        elif (
            self._reward_variant
            == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME
        ):
            dense_helper = self._survival_reward_for_player(player_index)
            bonus_catch_count = self._bonus_catch_count_for_player(
                batch=batch,
                player_index=player_index,
            )
            bonus_pickup_reward = (
                float(bonus_catch_count)
                * SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD
            )
            trainer_reward = dense_helper + bonus_pickup_reward
        elif (
            self._reward_variant
            == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME
        ):
            dense_helper = self._survival_reward_for_player(player_index)
            bonus_catch_count = self._bonus_catch_count_for_player(
                batch=batch,
                player_index=player_index,
            )
            bonus_pickup_reward = (
                float(bonus_catch_count)
                * SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD
            )
            if bool(batch.done[0]) and sparse_outcome != 0.0:
                terminal_outcome_reward = sparse_outcome * float(
                    self._physical_step_index
                )
            trainer_reward = (
                dense_helper + bonus_pickup_reward + terminal_outcome_reward
            )
        elif self._reward_variant == REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC:
            alive = self._env.state["alive"][0, :2].astype(bool)
            trainer_reward = 1.0 if bool(np.all(alive)) else 0.0
        else:
            raise RuntimeError(f"unsupported reward_variant {self._reward_variant!r}")
        return {
            "trainer_reward": float(trainer_reward),
            "sparse_outcome_reward": float(sparse_outcome),
            "dense_survival_helper": float(dense_helper),
            "bonus_catch_count_step": float(bonus_catch_count),
            "bonus_pickup_reward": float(bonus_pickup_reward),
            "terminal_outcome_reward": float(terminal_outcome_reward),
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
        if self._reward_variant == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME:
            return SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_ID
        if self._reward_variant == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME:
            return SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA_ID
        if self._reward_variant == REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC:
            return ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_ID
        raise RuntimeError(f"unsupported reward_variant {self._reward_variant!r}")

    def _reward_schema_hash(self) -> str:
        if self._reward_variant == REWARD_VARIANT_SPARSE_OUTCOME:
            return SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_HASH
        if self._reward_variant == REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME:
            return DENSE_SURVIVAL_PLUS_OUTCOME_REWARD_SCHEMA_HASH
        if self._reward_variant == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME:
            return SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_HASH
        if self._reward_variant == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME:
            return SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA_HASH
        if self._reward_variant == REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC:
            return ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_HASH
        raise RuntimeError(f"unsupported reward_variant {self._reward_variant!r}")

    def _reward_perspective(self) -> str:
        if self._reward_variant == REWARD_VARIANT_SPARSE_OUTCOME:
            return "ego_player_sparse_round_outcome"
        if self._reward_variant == REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME:
            return "ego_player_dense_survival_helper_plus_sparse_outcome"
        if self._reward_variant == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME:
            return "ego_player_dense_survival_plus_same_step_bonus_no_outcome"
        if self._reward_variant == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME:
            return "ego_player_dense_survival_plus_same_step_bonus_plus_scaled_outcome"
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
        elif (
            self._reward_variant
            == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME
        ):
            low = 0.0
            high = float(
                self._policy_action_repeat_max
                * (1.0 + SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD)
            )
        elif (
            self._reward_variant
            == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME
        ):
            low = -float(self._max_source_ticks)
            high = float(
                self._max_source_ticks
                + self._policy_action_repeat_max
                * (1.0 + SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD)
            )
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
        bonus_catch_count_step_sum: int,
        bonus_pickup_reward_sum: float,
        terminal_outcome_reward_sum: float,
        profile_env_timing_sec: dict[str, float] | None = None,
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
                "learner_seat_assignment_schema_id": LEARNER_SEAT_ASSIGNMENT_SCHEMA_ID,
                "learner_seat_mode": self._learner_seat_mode,
                "ego_player_index": int(self.ego_player_index),
                "opponent_player_index": int(self.opponent_player_index),
                "learner_player_index": int(self.ego_player_index),
                "learner_player_id": self.ego_player_id,
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
                "bonus_catch_count_step_for_ego": int(bonus_catch_count_step_sum),
                "bonus_pickup_reward_for_ego": float(bonus_pickup_reward_sum),
                "terminal_outcome_reward_for_ego": float(terminal_outcome_reward_sum),
                "bonus_pickup_reward_per_catch": (
                    SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD
                    if self._reward_variant
                    in (
                        REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
                        REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
                    )
                    else 0.0
                ),
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
        if profile_env_timing_sec is not None:
            info["profile_env_timing_sec"] = {
                key: float(value) for key, value in profile_env_timing_sec.items()
            }
            info["source_state_scalar_dirty_render_cache_stats"] = (
                self._scalar_dirty_render_cache.stats.as_dict()
            )
        if self._last_opponent_policy_sidecar is not None:
            info["opponent_policy_sidecar"] = self._last_opponent_policy_sidecar
        return info

    def _base_info(self) -> dict[str, Any]:
        public_info = self._env._public_info()
        runtime_env_impl_id = str(public_info["env_impl_id"])
        public_env_contract_id = str(public_info["public_env_contract_id"])
        ruleset_id = str(public_info["ruleset_id"])
        info = {
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
            "decision_source_frames": int(self._decision_source_frames),
            "source_physics_step_ms": float(self._source_physics_step_ms),
            "source_frame_decision": True,
            "max_ticks": int(self._max_ticks),
            "max_source_ticks": int(self._max_source_ticks),
            "body_capacity": int(self._env.body_capacity),
            "visual_trail_capacity": int(self._env.state["visual_trail_active"].shape[1]),
            "visual_trail_write_cursor": int(
                self._env.state["visual_trail_write_cursor"][0]
            ),
            "visual_trail_active_count": int(
                self._env.state["visual_trail_active"].sum()
            ),
            "visual_trail_overflow": bool(self._env.state["visual_trail_overflow"][0]),
            "body_write_cursor": int(self._env.state["body_write_cursor"][0]),
            "body_active_count": int(self._env.state["body_active"].sum()),
            "body_overflow": bool(self._env.state["body_overflow"][0]),
            "player_count": 2,
            "player_ids": ("player_0", "player_1"),
            "learner_seat_assignment_schema_id": LEARNER_SEAT_ASSIGNMENT_SCHEMA_ID,
            "learner_seat_mode": self._learner_seat_mode,
            "supported_learner_seat_modes": list(LEARNER_SEAT_MODES),
            "ego_player_index": int(self.ego_player_index),
            "opponent_player_index": int(self.opponent_player_index),
            "learner_player_index": int(self.ego_player_index),
            "learner_player_id": self.ego_player_id,
            "opponent_player_id": self.opponent_player_id,
            "observation_schema_id": STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
            "observation_schema_hash": STACKED_SOURCE_STATE_GRAY64_SCHEMA_HASH,
            "single_frame_schema_id": SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_ID,
            "single_frame_schema_hash": SOURCE_STATE_CANVAS_LIKE_GRAY64_SCHEMA_HASH,
            "observation_contract_id": POLICY_OBSERVATION_CONTRACT_ID,
            "observation_contract": policy_observation_surface(
                trail_render_mode=self._source_state_trail_render_mode,
                bonus_render_mode=self._model_bonus_render_mode,
                backend=self._policy_observation_backend,
            ),
            "raw_observation_schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
            "raw_observation_schema_hash": SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SCHEMA_HASH,
            "raw_observation_available": True,
            "raw_observation_accessors": [
                "raw_observation()",
                "render('source_state_raw_visual_tensor')",
                "render('source_state_rgb_canvas_like')",
            ],
            "raw_observation_dtype": "uint8",
            "raw_observation_color_space": "RGB",
            "raw_observation_source": (
                "render_source_state_rgb_canvas_like("
                "frame_size="
                f"{SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE}, "
                f"trail_render_mode={self._source_state_trail_render_mode!r}, "
                f"bonus_render_mode={self._raw_observation_bonus_render_mode!r})"
            ),
            "grayscale_observation_source": (
                "jax_gpu_scalar_block_704_gray64_experimental("
                "browser_lines, simple_symbols)"
                if self._policy_observation_backend == POLICY_OBSERVATION_BACKEND_GPU
                else (
                    "render_source_state_canvas_gray64("
                    "rgb_out=raw_observation_buffer, "
                    f"trail_render_mode={self._source_state_trail_render_mode!r}, "
                    f"bonus_render_mode={self._model_bonus_render_mode!r})"
                )
            ),
            "policy_observation_backend": self._policy_observation_backend,
            "default_policy_observation_backend": DEFAULT_POLICY_OBSERVATION_BACKEND,
            "supported_policy_observation_backends": list(
                SOURCE_STATE_POLICY_OBSERVATION_BACKENDS
            ),
            "policy_observation_backend_kind": (
                "experimental_scalar_jax_gpu"
                if self._policy_observation_backend == POLICY_OBSERVATION_BACKEND_GPU
                else "cpu_oracle"
            ),
            "policy_observation_gpu_min_trail_slots": (
                int(self._jax_scalar_observation_renderer.min_trail_slots)
                if self._jax_scalar_observation_renderer is not None
                else None
            ),
            "policy_observation_gpu_last_profile": (
                dict(self._jax_scalar_observation_renderer.last_profile)
                if self._jax_scalar_observation_renderer is not None
                else None
            ),
            "player_perspective_schema_id": POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID,
            "observation_perspective": POLICY_OBSERVATION_PERSPECTIVE,
            "observation_perspective_owner": POLICY_OBSERVATION_PERSPECTIVE_OWNER,
            "source_state_player_perspective": True,
            "observation_perspective_player_index": int(self.ego_player_index),
            "observation_perspective_player_id": self.ego_player_id,
            "opponent_observation_perspective_player_index": int(
                self.opponent_player_index
            ),
            "opponent_observation_perspective_player_id": self.opponent_player_id,
            "renderer_impl_id": SOURCE_STATE_CANVAS_LIKE_GRAY64_RENDERER_IMPL_ID,
            "raw_renderer_impl_id": SOURCE_STATE_RGB_CANVAS_LIKE_RENDERER_IMPL_ID,
            "source_state_trail_render_mode": self._source_state_trail_render_mode,
            "source_state_bonus_render_mode": self._source_state_bonus_render_mode,
            "default_bonus_render_mode": SOURCE_STATE_DEFAULT_BONUS_RENDER_MODE,
            "supported_bonus_render_modes": list(SOURCE_STATE_SUPPORTED_BONUS_RENDER_MODES),
            "model_observation_bonus_render_mode": self._model_bonus_render_mode,
            "raw_observation_bonus_render_mode": self._raw_observation_bonus_render_mode,
            "bonus_render_mode": self._model_bonus_render_mode,
            "bonus_renderer_kind": (
                "simple_symbol_masks"
                if self._model_bonus_render_mode == BONUS_RENDER_MODE_SIMPLE_SYMBOLS
                else "typed_luma_circles"
                if self._model_bonus_render_mode == BONUS_RENDER_MODE_CIRCLES_FAST
                else "source_sprite_atlas_tiles"
            ),
            "bonus_renderer_is_approximation": bool(
                self._model_bonus_render_mode != BONUS_RENDER_MODE_BROWSER_SPRITES
            ),
            "source_state_scalar_dirty_render_enabled": bool(
                self._scalar_dirty_render_enabled
            ),
            "source_state_scalar_dirty_render_active": (
                self._should_use_scalar_dirty_render_cache()
            ),
            "source_state_scalar_dirty_render_kind": (
                "exact_browser_lines_dirty_cache"
                if self._should_use_scalar_dirty_render_cache()
                and self._policy_observation_backend == POLICY_OBSERVATION_BACKEND_CPU
                else "jax_gpu_scalar_experimental"
                if self._policy_observation_backend == POLICY_OBSERVATION_BACKEND_GPU
                else "scalar_full_render"
            ),
            "truth_level": SOURCE_STATE_CANVAS_GRAY64_TRUTH_LEVEL,
            "source_fidelity_level": SOURCE_STATE_CANVAS_GRAY64_SOURCE_FIDELITY_LEVEL,
            "source_fidelity_claim": "source_state_backed_non_browser_pixel",
            "visual_surface": SOURCE_STATE_CANVAS_LIKE_GRAY64_SURFACE,
            "visual_truth_level": SOURCE_STATE_CANVAS_GRAY64_TRUTH_LEVEL,
            "visual_source_state_backed": SOURCE_STATE_CANVAS_GRAY64_SOURCE_STATE_BACKED,
            "debug_fidelity_only": False,
            "browser_pixel_fidelity": SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY,
            "browser_pixel_fidelity_claim": SOURCE_STATE_BROWSER_PIXEL_FIDELITY_CLAIM,
            "uses_ale": SOURCE_STATE_CANVAS_GRAY64_USES_ALE,
            "ale_usage": "none",
            "default_trail_render_mode": SOURCE_STATE_DEFAULT_TRAIL_RENDER_MODE,
            "supported_trail_render_modes": list(SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES),
            "browser_trail_semantics": SOURCE_STATE_BROWSER_TRAIL_SEMANTICS,
            "browser_client_trail_point_caveat": (
                SOURCE_STATE_BROWSER_CLIENT_TRAIL_POINT_CAVEAT
            ),
            **_trail_render_metadata(self._source_state_trail_render_mode),
            "shape": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
            "dtype": SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_DTYPE,
            "range": list(SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_VALUE_RANGE),
            "value_range": list(SOURCE_STATE_CANVAS_GRAY64_NORMALIZED_VALUE_RANGE),
            "raw_frame_shape": list(SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SHAPE),
            "grayscale_frame_shape": list(SOURCE_STATE_CANVAS_GRAY64_SHAPE),
            "lightzero_payload_shape": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
            "model_observation_shape": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
            "frame_stack_owner": "curvyzero_source_state_survival_wrapper",
            "frame_stack_proof": (
                "wrapper_owned_raw_canvas_to_downsampled_gray64_fifo_stack; "
                "not LightZero env-manager stacking"
            ),
            "reward_schema_id": self._reward_schema_id(),
            "reward_schema_hash": self._reward_schema_hash(),
            "scalar_training_reward_variant": self._reward_variant,
            "dense_survival_helper_enabled": (
                self._reward_variant
                in (
                    REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
                    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
                    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
                )
            ),
            "bonus_pickup_reward_enabled": (
                self._reward_variant
                in (
                    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
                    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
                )
            ),
            "bonus_pickup_reward_per_catch": (
                SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD
                if self._reward_variant
                in (
                    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
                    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
                )
                else 0.0
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
            "opponent_death_mode": self._opponent_death_mode,
            "opponent_death_mode_diagnostic": (
                self._opponent_death_mode == OPPONENT_DEATH_MODE_IMMORTAL
            ),
            "opponent_death_mode_claim": (
                "diagnostic_opponent_immortal_not_source_faithful"
                if self._opponent_death_mode == OPPONENT_DEATH_MODE_IMMORTAL
                else "none"
            ),
            "opponent_runtime_mode": self._opponent_runtime_mode,
            "opponent_runtime_mode_claim": (
                (
                    "blank_canvas_noop_"
                    f"{self.opponent_player_id}_inert_hidden_no_trail_no_collision_no_bonus"
                )
                if self._blank_canvas_noop_enabled()
                else "normal_opponent_runtime"
            ),
            "opponent_visibility_mode": (
                "hidden_from_model_gif_raw_render_views_public_present_alive"
                if self._blank_canvas_noop_enabled()
                else "visible_if_present_alive"
            ),
            "opponent_collision_effect": (
                "disabled_no_player_1_movement_trail_collision_bonus_side_effects"
                if self._blank_canvas_noop_enabled()
                else "normal"
            ),
            "opponent_trail_mode": (
                "none_blank_canvas_scrubbed"
                if self._blank_canvas_noop_enabled()
                else "normal"
            ),
            "blank_canvas_noop": self._blank_canvas_noop_enabled(),
            "blank_canvas_noop_training_reward_expected": (
                "survival_plus_bonus_no_outcome"
                if self._blank_canvas_noop_enabled()
                else None
            ),
            "blank_canvas_noop_uses_remove_player": False,
            "blank_canvas_noop_public_opponent_present_alive": (
                bool(
                    self._env.state["present"][0, self.opponent_player_index]
                    and self._env.state["alive"][0, self.opponent_player_index]
                )
                if self._blank_canvas_noop_enabled()
                else None
            ),
            "blank_canvas_noop_public_player_1_present_alive": (
                bool(
                    self._env.state["present"][0, 1]
                    and self._env.state["alive"][0, 1]
                )
                if self._blank_canvas_noop_enabled()
                else None
            ),
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
                    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
                )
                else 0.0
            ),
            "terminal_outcome_reward_enabled": (
                self._reward_variant
                in (
                    REWARD_VARIANT_SPARSE_OUTCOME,
                    REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
                    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
                )
            ),
            "terminal_outcome_scale": (
                "episode_source_step_count"
                if self._reward_variant
                == REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME
                else "unit_sparse_outcome"
                if self._reward_variant
                in (
                    REWARD_VARIANT_SPARSE_OUTCOME,
                    REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
                )
                else "none"
            ),
            "loser_penalty": (
                -1.0
                if self._reward_variant
                in (
                    REWARD_VARIANT_SPARSE_OUTCOME,
                    REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
                    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
                )
                else 0.0
            ),
            "winner_bonus": (
                1.0
                if self._reward_variant
                in (
                    REWARD_VARIANT_SPARSE_OUTCOME,
                    REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
                    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
                )
                else 0.0
            ),
            "opponent_policy_id": self._opponent_policy_id(),
            "opponent_policy_kind": self._opponent_policy_kind,
            "supported_opponent_policy_kinds": list(SOURCE_STATE_OPPONENT_POLICY_KINDS),
            "opponent_training_relation": _source_state_opponent_training_relation(
                self._opponent_policy_kind
            ),
            "opponent_policy_version": self._opponent_policy_version(),
            "opponent_policy_seed": self._opponent_policy_seed,
            "opponent_mixture_enabled": self._opponent_mixture is not None,
            "opponent_mixture_schema_id": (
                OPPONENT_MIXTURE_SCHEMA_ID if self._opponent_mixture is not None else None
            ),
            "opponent_mixture_selection_unit": (
                OPPONENT_MIXTURE_SELECTION_UNIT
                if self._opponent_mixture is not None
                else None
            ),
            "opponent_mixture_entry_name": (
                self._episode_opponent_mixture_entry.get("name")
                if self._episode_opponent_mixture_entry is not None
                else None
            ),
            "opponent_mixture_entry_weight": (
                self._episode_opponent_mixture_entry.get("weight")
                if self._episode_opponent_mixture_entry is not None
                else None
            ),
            "opponent_mixture_entry_index": (
                self._episode_opponent_mixture_entry.get("selection_index")
                if self._episode_opponent_mixture_entry is not None
                else None
            ),
            "opponent_mixture_age_label": (
                self._episode_opponent_mixture_entry.get("age_label")
                if self._episode_opponent_mixture_entry is not None
                else None
            ),
            "opponent_assignment_id": (
                self._opponent_assignment_context.get("assignment_id")
                if self._opponent_assignment_context is not None
                else None
            ),
            "opponent_assignment_ref": (
                self._opponent_assignment_context.get("assignment_ref")
                if self._opponent_assignment_context is not None
                else None
            ),
            "opponent_assignment_sha256": (
                self._opponent_assignment_context.get("assignment_sha256")
                if self._opponent_assignment_context is not None
                else None
            ),
            "opponent_assignment_refresh_index": (
                self._opponent_assignment_context.get("refresh_index")
                if self._opponent_assignment_context is not None
                else None
            ),
            "opponent_wall_avoidant_safe_margin": (
                float(self._opponent_wall_avoidant_safe_margin)
                if self._opponent_policy_kind
                == OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT
                else None
            ),
            "episode_seed": self._episode_seed,
            "reset_seed_strategy": (
                "dynamic_seed_sequence_from_run_seed_and_reset_index/v0"
                if self._dynamic_seed
                else "fixed_seed"
            ),
            "configured_dynamic_seed": bool(self._configured_dynamic_seed),
            "effective_dynamic_seed": bool(self._dynamic_seed),
            "seed_call_dynamic_seed_arg": self._last_seed_call_dynamic_seed_arg,
            "run_seed": self._seed,
            "reset_index": int(max(0, self._reset_index - 1)),
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
        if self.opponent_policy is not None:
            info["opponent_policy_id"] = str(
                getattr(self.opponent_policy, "policy_id", info["opponent_policy_id"])
            )
            info["opponent_policy_version"] = str(
                getattr(self.opponent_policy, "policy_version", "unknown")
            )
        return info

    def _opponent_policy_id(self) -> str:
        if (
            self._opponent_policy_kind
            == OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT
        ):
            return PROACTIVE_WALL_AVOIDANT_OPPONENT_POLICY_ID
        if (
            self._opponent_policy_kind
            == OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT
            and self.opponent_policy is not None
        ):
            return str(getattr(self.opponent_policy, "policy_id", "frozen_lightzero_checkpoint"))
        return "curvyzero_source_state_fixed_straight_opponent"

    def _opponent_policy_version(self) -> str:
        if (
            self._opponent_policy_kind
            == OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT
        ):
            return PROACTIVE_WALL_AVOIDANT_OPPONENT_POLICY_VERSION
        if (
            self._opponent_policy_kind
            == OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT
            and self.opponent_policy is not None
        ):
            return str(getattr(self.opponent_policy, "policy_version", "unknown"))
        return "v0.2026-05-11"

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
        opponent_metadata = _opponent_policy_metadata(
            info.get("opponent_policy_sidecar")
        )
        opponent_load_summary = opponent_metadata.get("provider_load_summary")
        if not isinstance(opponent_load_summary, dict):
            opponent_load_summary = {}
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
            "decision_ms": info.get("decision_ms"),
            "decision_source_frames": info.get("decision_source_frames"),
            "source_physics_step_ms": info.get("source_physics_step_ms"),
            "source_frame_decision": info.get("source_frame_decision"),
            "max_ticks": info.get("max_ticks"),
            "max_source_ticks": info.get("max_source_ticks"),
            "bonus_support_mode": info.get("bonus_support_mode"),
            "natural_bonus_spawn": info.get("natural_bonus_spawn"),
            "natural_bonus_pop_count": info.get("natural_bonus_pop_count"),
            "death_mode": info.get("death_mode"),
            "opponent_death_mode": info.get("opponent_death_mode"),
            "opponent_death_mode_diagnostic": info.get(
                "opponent_death_mode_diagnostic"
            ),
            "opponent_death_mode_claim": info.get("opponent_death_mode_claim"),
            "opponent_runtime_mode": info.get("opponent_runtime_mode"),
            "opponent_runtime_mode_claim": info.get("opponent_runtime_mode_claim"),
            "opponent_visibility_mode": info.get("opponent_visibility_mode"),
            "opponent_collision_effect": info.get("opponent_collision_effect"),
            "opponent_trail_mode": info.get("opponent_trail_mode"),
            "blank_canvas_noop": info.get("blank_canvas_noop"),
            "blank_canvas_noop_training_reward_expected": info.get(
                "blank_canvas_noop_training_reward_expected"
            ),
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
            "opponent_mixture_enabled": info.get("opponent_mixture_enabled"),
            "opponent_mixture_schema_id": info.get("opponent_mixture_schema_id"),
            "opponent_mixture_selection_unit": info.get(
                "opponent_mixture_selection_unit"
            ),
            "opponent_mixture_entry_name": info.get("opponent_mixture_entry_name"),
            "opponent_mixture_entry_weight": info.get("opponent_mixture_entry_weight"),
            "opponent_mixture_entry_index": info.get("opponent_mixture_entry_index"),
            "opponent_mixture_age_label": info.get("opponent_mixture_age_label"),
            "opponent_assignment_id": info.get("opponent_assignment_id"),
            "opponent_assignment_ref": info.get("opponent_assignment_ref"),
            "opponent_assignment_sha256": info.get("opponent_assignment_sha256"),
            "opponent_assignment_refresh_index": info.get(
                "opponent_assignment_refresh_index"
            ),
            "opponent_checkpoint_ref": opponent_metadata.get("checkpoint_ref"),
            "opponent_snapshot_ref": opponent_metadata.get("snapshot_ref"),
            "opponent_provider_id": opponent_metadata.get("provider_id"),
            "opponent_provider_version": opponent_metadata.get("provider_version"),
            "opponent_provider_load_ok": opponent_load_summary.get("ok"),
            "opponent_provider_load_strict": opponent_load_summary.get("strict"),
            "opponent_provider_load_candidate": opponent_load_summary.get("candidate"),
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
            "bonus_catch_count_step_for_ego": info.get(
                "bonus_catch_count_step_for_ego"
            ),
            "bonus_pickup_reward_for_ego": info.get("bonus_pickup_reward_for_ego"),
            "bonus_pickup_reward_enabled": info.get("bonus_pickup_reward_enabled"),
            "bonus_pickup_reward_per_catch": info.get("bonus_pickup_reward_per_catch"),
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
            "default_trail_render_mode": info.get("default_trail_render_mode"),
            "supported_trail_render_modes": info.get("supported_trail_render_modes"),
            "trail_render_mode": info.get("trail_render_mode"),
            "trail_renderer_kind": info.get("trail_renderer_kind"),
            "trail_renderer_truth_level": info.get("trail_renderer_truth_level"),
            "trail_renderer_is_approximation": info.get(
                "trail_renderer_is_approximation"
            ),
            "browser_style_trail_renderer": info.get("browser_style_trail_renderer"),
            "browser_trail_semantics": info.get("browser_trail_semantics"),
            "browser_client_trail_point_caveat": info.get(
                "browser_client_trail_point_caveat"
            ),
            "source_state_bonus_render_mode": info.get("source_state_bonus_render_mode"),
            "default_bonus_render_mode": info.get("default_bonus_render_mode"),
            "supported_bonus_render_modes": info.get("supported_bonus_render_modes"),
            "model_observation_bonus_render_mode": info.get(
                "model_observation_bonus_render_mode"
            ),
            "raw_observation_bonus_render_mode": info.get("raw_observation_bonus_render_mode"),
            "bonus_render_mode": info.get("bonus_render_mode"),
            "bonus_renderer_kind": info.get("bonus_renderer_kind"),
            "bonus_renderer_is_approximation": info.get(
                "bonus_renderer_is_approximation"
            ),
            "browser_pixel_fidelity_claim": info.get("browser_pixel_fidelity_claim"),
            "source_fidelity_claim": info.get("source_fidelity_claim"),
            "debug_fidelity_only": info.get("debug_fidelity_only"),
            "uses_ale": info.get("uses_ale"),
        }
        profile_env_timing_sec = info.get("profile_env_timing_sec")
        if isinstance(profile_env_timing_sec, dict):
            row["profile_env_timing_sec"] = {
                str(key): float(value)
                for key, value in profile_env_timing_sec.items()
                if isinstance(value, (int, float))
            }
        dirty_render_stats = info.get("source_state_scalar_dirty_render_cache_stats")
        if isinstance(dirty_render_stats, dict):
            row["source_state_scalar_dirty_render_cache_stats"] = dirty_render_stats
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
            if not self._dynamic_seed:
                self._episode_seed = self._seed
                return self._seed
        if not self._dynamic_seed:
            self._episode_seed = self._seed
            return self._seed
        reset_seed = self._derived_reset_seed(self._seed, self._reset_index)
        self._reset_index += 1
        self._episode_seed = reset_seed
        return reset_seed

    @staticmethod
    def _derived_reset_seed(run_seed: int, reset_index: int) -> int:
        seed_sequence = np.random.SeedSequence(
            [
                int(run_seed) & 0xFFFFFFFF,
                (int(run_seed) >> 32) & 0xFFFFFFFF,
                int(reset_index) & 0xFFFFFFFF,
                (int(reset_index) >> 32) & 0xFFFFFFFF,
            ]
        )
        rng = np.random.default_rng(seed_sequence)
        return int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))

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
    _allowed_opponent_policy_kinds = SOURCE_STATE_JOINT_ACTION_OPPONENT_POLICY_KINDS

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
            bonus_catch_count_step_sum=0,
            bonus_pickup_reward_sum=0.0,
            terminal_outcome_reward_sum=0.0,
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
                "opponent_policy_kind": OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION,
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


def _build_source_state_frozen_lightzero_opponent_policy(
    cfg: Any,
    *,
    seed: int,
) -> Any:
    checkpoint_path = _cfg_get(cfg, "opponent_checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError(
            "opponent_checkpoint_path is required with opponent_policy_kind="
            f"{OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT!r}"
        )
    checkpoint_ref = _cfg_get(cfg, "opponent_checkpoint_ref", None)
    snapshot_ref = str(
        _cfg_get(
            cfg,
            "opponent_snapshot_ref",
            "curvytron_source_state_visual_survival_frozen_opponent",
        )
    )
    return snapshot_backed_lightzero_checkpoint_opponent_policy(
        checkpoint_path=checkpoint_path,
        snapshot_ref=snapshot_ref,
        checkpoint_ref=None if checkpoint_ref is None else str(checkpoint_ref),
        seed=int(seed),
        num_simulations=int(_cfg_get(cfg, "opponent_num_simulations", 8)),
        batch_size=int(_cfg_get(cfg, "opponent_batch_size", 16)),
        use_cuda=bool(_cfg_get(cfg, "opponent_use_cuda", False)),
        state_key=_cfg_get(cfg, "opponent_checkpoint_state_key", None),
    )


def _source_state_opponent_training_relation(opponent_policy_kind: str) -> str:
    if opponent_policy_kind == OPPONENT_POLICY_KIND_FIXED_STRAIGHT:
        return OPPONENT_TRAINING_RELATION_FIXED_STRAIGHT
    if opponent_policy_kind == OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT:
        return OPPONENT_TRAINING_RELATION_PROACTIVE_WALL_AVOIDANT
    if opponent_policy_kind == OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT:
        return OPPONENT_TRAINING_RELATION_FROZEN_LIGHTZERO_CHECKPOINT
    return f"unknown:{opponent_policy_kind}"


def _opponent_policy_metadata(sidecar: Any) -> dict[str, Any]:
    if not isinstance(sidecar, dict):
        return {}
    metadata = sidecar.get("policy_metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _learner_seat_mode_from_cfg(cfg: Any) -> tuple[str, int]:
    if _cfg_get(cfg, "ego_player_index", None) is not None:
        raise ValueError("ego_player_index config is removed; use learner_seat_mode")
    mode = str(_cfg_get(cfg, "learner_seat_mode", DEFAULT_LEARNER_SEAT_MODE))
    if mode not in LEARNER_SEAT_MODES:
        raise ValueError(
            "learner_seat_mode must be one of "
            f"{LEARNER_SEAT_MODES!r}; got {mode!r}"
        )
    if mode == LEARNER_SEAT_MODE_FIXED_PLAYER_1:
        return mode, 1
    return mode, 0


def _player_perspective_rgb_palette(
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
    self_rgb = (SELF_BODY_VALUE, SELF_BODY_VALUE, SELF_BODY_VALUE)
    other_rgb = (OTHER_BODY_VALUE, OTHER_BODY_VALUE, OTHER_BODY_VALUE)
    palette = [other_rgb for _ in range(max(total_players, max_color_index + 1))]
    palette[int(color_indices[player])] = self_rgb
    return tuple(palette)


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


class _JaxScalarPolicyObservationRenderer:
    """Experimental scalar JAX renderer for proving trainer integration.

    This is not the final fast design. It renders one env row at a time and
    copies results back to NumPy because the stock LightZero env API expects
    NumPy observations. The point is to make the GPU observation path real
    enough to profile.
    """

    def __init__(self, *, player_count: int, min_trail_slots: int):
        self.player_count = int(player_count)
        self.min_trail_slots = int(min_trail_slots)
        self._jax: Any | None = None
        self._jnp: Any | None = None
        self._bench: Any | None = None
        self._render_fns: dict[tuple[int, int, int, int, int, float, int], Any] = {}
        self.last_profile: dict[str, Any] = {}

    def render_player_perspectives(
        self,
        state: Mapping[str, np.ndarray],
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        self._ensure_loaded()
        if self._jax is None or self._bench is None:
            raise RuntimeError("JAX renderer imports were not initialized")
        player_count = int(np.asarray(state["pos"]).shape[1])
        if player_count != self.player_count:
            raise ValueError(
                f"jax_gpu scalar backend expected {self.player_count} players, got {player_count}"
            )
        expected_shape = (player_count, *SOURCE_STATE_CANVAS_GRAY64_SHAPE)
        if out is None:
            frames = np.empty(expected_shape, dtype=np.uint8)
        else:
            frames = np.asarray(out)
            if frames.shape != expected_shape or frames.dtype != np.uint8:
                raise ValueError(
                    f"jax_gpu scalar backend out must be uint8 {expected_shape}, "
                    f"got {frames.dtype} {frames.shape}"
                )
        render_started = time.perf_counter()
        slot_profile = self._trail_slot_profile(state)
        trail_slots = int(slot_profile["render_trail_slots"])
        bonus_count = int(np.asarray(state.get("bonus_active", np.zeros((1, 0)))).shape[1])
        map_size = float(np.asarray(state["map_size"])[0])
        config = {
            "batch_size": 1,
            "player_count": player_count,
            "trail_slots": trail_slots,
            "bonus_count": bonus_count,
            "frame_size": SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
            "target_size": SOURCE_STATE_CANVAS_GRAY64_SHAPE[1],
            "map_size": map_size,
            "render_surface": self._bench.RENDER_SURFACE_BLOCK_704_GRAY64,
            "render_mode": self._bench.RENDER_MODE_BROWSER_LINES,
            "bonus_render_mode": self._bench.BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
        }
        compact_started = time.perf_counter()
        compact_state = self._bench._production_to_benchmark_source_state(
            np=np,
            production_state=state,
            config=config,
        )
        compact_sec = time.perf_counter() - compact_started
        device_started = time.perf_counter()
        device_state = self._bench._copy_state_to_device(
            jax=self._jax,
            state=compact_state,
        )
        device_sec = time.perf_counter() - device_started
        render_secs: list[float] = []
        readback_secs: list[float] = []
        cache_misses = 0
        for controlled_player in range(player_count):
            render_fn, cache_miss = self._render_fn(
                config=config,
                controlled_player=controlled_player,
            )
            cache_misses += int(cache_miss)
            player_render_started = time.perf_counter()
            rendered = render_fn(device_state)
            rendered.block_until_ready()
            render_secs.append(time.perf_counter() - player_render_started)
            readback_started = time.perf_counter()
            frames[controlled_player] = np.asarray(rendered)[0]
            readback_secs.append(time.perf_counter() - readback_started)
        self.last_profile = {
            **slot_profile,
            "bonus_slots": int(bonus_count),
            "cache_misses": int(cache_misses),
            "cache_size": int(len(self._render_fns)),
            "compact_sec": float(compact_sec),
            "device_put_sec": float(device_sec),
            "render_sec_by_player": [float(value) for value in render_secs],
            "readback_sec_by_player": [float(value) for value in readback_secs],
            "render_total_sec": float(sum(render_secs)),
            "readback_total_sec": float(sum(readback_secs)),
            "total_sec": float(time.perf_counter() - render_started),
        }
        return frames

    def _trail_slot_profile(self, state: Mapping[str, np.ndarray]) -> dict[str, Any]:
        active = np.asarray(state["visual_trail_active"])
        if active.ndim != 2 or active.shape[0] != 1:
            raise ValueError(
                "jax_gpu scalar backend expects visual_trail_active with shape [1,C]"
            )
        capacity = int(active.shape[1])
        cursor = int(
            np.asarray(state.get("visual_trail_write_cursor", np.asarray([capacity])))[0]
        )
        if cursor < 0 or cursor > capacity:
            raise ValueError(
                "jax_gpu scalar backend received invalid visual_trail_write_cursor "
                f"{cursor}; expected [0, {capacity}]"
            )
        active_row = active[0].astype(bool, copy=False).copy()
        active_row[cursor:] = False
        active_indices = np.flatnonzero(active_row)
        last_active_exclusive = int(active_indices[-1] + 1) if active_indices.size else 1
        needed_slots = max(1, last_active_exclusive)
        min_slots = min(capacity, max(1, int(self.min_trail_slots)))
        render_slots = min(capacity, _ceil_power_of_two(max(min_slots, needed_slots)))
        if needed_slots > render_slots:
            raise RuntimeError(
                "jax_gpu scalar backend would truncate active visual trail slots: "
                f"needed {needed_slots}, render slots {render_slots}, capacity {capacity}"
            )
        return {
            "visual_trail_capacity": int(capacity),
            "visual_trail_write_cursor": int(cursor),
            "visual_trail_active_count": int(active_indices.size),
            "visual_trail_last_active_exclusive": int(last_active_exclusive),
            "min_render_trail_slots": int(self.min_trail_slots),
            "render_trail_slots": int(render_slots),
            "render_trail_slots_reduced_from_capacity": bool(render_slots < capacity),
        }

    def _ensure_loaded(self) -> None:
        if self._jax is not None:
            return
        import jax
        import jax.numpy as jnp

        from curvyzero.infra.modal import source_state_gpu_render_benchmark as bench

        backend = str(jax.default_backend())
        if backend != "gpu":
            raise RuntimeError(
                "policy_observation_backend='jax_gpu' requires a JAX GPU backend; "
                f"got {backend!r}"
            )
        self._jax = jax
        self._jnp = jnp
        self._bench = bench

    def _render_fn(
        self,
        *,
        config: dict[str, Any],
        controlled_player: int,
    ) -> tuple[Any, bool]:
        self._ensure_loaded()
        if self._jax is None or self._jnp is None or self._bench is None:
            raise RuntimeError("JAX renderer imports were not initialized")
        key = (
            int(config["player_count"]),
            int(config["trail_slots"]),
            int(config["bonus_count"]),
            int(config["frame_size"]),
            int(config["target_size"]),
            float(config["map_size"]),
            int(controlled_player),
        )
        cache_miss = key not in self._render_fns
        if key not in self._render_fns:
            render_config = {**config, "controlled_player": int(controlled_player)}
            self._render_fns[key] = self._bench._make_jax_render_fn(
                jax=self._jax,
                jnp=self._jnp,
                config=render_config,
                render_mode_id=self._bench.RENDER_MODE_IDS[
                    self._bench.RENDER_MODE_BROWSER_LINES
                ],
                bonus_render_mode_id=self._bench.BONUS_RENDER_MODE_IDS[
                    self._bench.BONUS_RENDER_MODE_SIMPLE_SYMBOLS
                ],
            )
        return self._render_fns[key], cache_miss


def _trail_render_metadata(trail_render_mode: str) -> dict[str, Any]:
    if trail_render_mode not in SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES:
        raise ValueError(
            "source_state_trail_render_mode must be one of "
            f"{SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES!r}; got {trail_render_mode!r}"
        )
    return {
        "trail_render_mode": trail_render_mode,
        "trail_renderer_kind": "connected_rounded_lines",
        "trail_renderer_truth_level": (
            "source_state_browser_style_lines_non_pixel_parity"
        ),
        "trail_renderer_is_approximation": False,
        "browser_style_trail_renderer": True,
    }


def _validate_trail_render_mode(value: Any) -> str:
    trail_render_mode = str(value)
    _trail_render_metadata(trail_render_mode)
    return trail_render_mode


def _validate_bonus_render_mode(value: Any) -> str:
    mode = str(value)
    if mode not in SOURCE_STATE_SUPPORTED_BONUS_RENDER_MODES:
        raise ValueError(
            "source_state_bonus_render_mode must be one of "
            f"{SOURCE_STATE_SUPPORTED_BONUS_RENDER_MODES!r}; got {mode!r}"
        )
    return mode


def _validate_policy_observation_backend(value: Any) -> str:
    backend = str(value)
    if backend not in SOURCE_STATE_POLICY_OBSERVATION_BACKENDS:
        raise ValueError(
            "policy_observation_backend must be one of "
            f"{SOURCE_STATE_POLICY_OBSERVATION_BACKENDS!r}; got {backend!r}"
        )
    return backend


def _resolve_model_bonus_render_mode(
    *,
    trail_render_mode: str,
    bonus_render_mode: str,
) -> str:
    if bonus_render_mode != SOURCE_STATE_BONUS_RENDER_MODE_AUTO:
        return bonus_render_mode
    return BONUS_RENDER_MODE_SIMPLE_SYMBOLS


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


def _ceil_power_of_two(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalize_opponent_assignment_context(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("opponent_assignment_context must be a JSON object")
    allowed = {
        "schema_id",
        "assignment_id",
        "assignment_ref",
        "assignment_sha256",
        "refresh_index",
        "source_epoch",
        "source_ref",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            "opponent_assignment_context has unknown keys "
            f"{sorted(unknown)!r}"
        )
    context = dict(value)
    assignment_id = context.get("assignment_id")
    if assignment_id is not None:
        context["assignment_id"] = str(assignment_id)
    assignment_ref = context.get("assignment_ref")
    if assignment_ref is not None:
        context["assignment_ref"] = str(assignment_ref)
    assignment_sha256 = context.get("assignment_sha256")
    if assignment_sha256 is not None:
        context["assignment_sha256"] = str(assignment_sha256)
    if context.get("refresh_index") is not None:
        context["refresh_index"] = int(context["refresh_index"])
    return context


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
    "OPPONENT_DEATH_MODE_IMMORTAL",
    "OPPONENT_DEATH_MODE_NORMAL",
    "OPPONENT_DEATH_MODES",
    "OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION",
    "OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT",
    "OPPONENT_TRAINING_RELATION_PROACTIVE_WALL_AVOIDANT",
    "REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME",
    "REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME",
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
    "SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_DTYPE",
    "SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SCHEMA_HASH",
    "SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_SHAPE",
    "SOURCE_STATE_CANVAS_LIKE_RAW_CANVAS_VALUE_RANGE",
    "SOURCE_STATE_BONUS_RENDER_MODE_AUTO",
    "SOURCE_STATE_DEFAULT_BONUS_RENDER_MODE",
    "SOURCE_STATE_DEFAULT_TRAIL_RENDER_MODE",
    "SOURCE_STATE_RAW_BONUS_RENDER_MODE",
    "SOURCE_STATE_SUPPORTED_BONUS_RENDER_MODES",
    "SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES",
    "SOURCE_STATE_TRAIL_RENDER_MODE_BODY_CIRCLES_FAST",
    "SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES",
    "STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID",
    "STACKED_SOURCE_STATE_GRAY64_SHAPE",
    "SURVIVAL_PLUS_BONUS_NO_OUTCOME_BONUS_REWARD",
    "SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_HASH",
    "SURVIVAL_PLUS_BONUS_NO_OUTCOME_REWARD_SCHEMA_ID",
    "SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA_HASH",
    "SURVIVAL_PLUS_BONUS_PLUS_OUTCOME_REWARD_SCHEMA_ID",
]
