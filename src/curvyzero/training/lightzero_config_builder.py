"""Pure LightZero config helpers shared by CurvyZero training surfaces."""

from __future__ import annotations

import copy
import importlib
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping

from curvyzero.contracts.curvytron import (
    CURVYTRON_DEFAULT_COLLECTOR_ENV_NUM,
    CURVYTRON_DEFAULT_MAX_ENV_STEP,
    CURVYTRON_DEFAULT_MAX_TRAIN_ITER,
    CURVYTRON_DEFAULT_N_EPISODE,
    CURVYTRON_DEFAULT_NUM_SIMULATIONS,
    CURVYTRON_DEFAULT_TRAIN_BATCH_SIZE,
    CURVYTRON_DECISION_MS,
    CURVYTRON_DECISION_SOURCE_FRAMES,
    CURVYTRON_SAVE_CKPT_AFTER_ITER,
    CURVYTRON_SOURCE_MAX_STEPS,
    DEFAULT_LEARNER_SEAT_MODE,
    LEARNER_SEAT_MODE_CHOICES,
)
from curvyzero.env.observation_surface_contract import (
    DEFAULT_POLICY_OBSERVATION_BACKEND,
    POLICY_BONUS_RENDER_MODE,
    POLICY_OBSERVATION_BACKENDS,
    POLICY_OBSERVATION_CONTRACT_ID,
    POLICY_TRAIL_RENDER_MODE,
    policy_observation_surface,
)
from curvyzero.env.trainer_contract import (
    REWARD_SCHEMA_ID as SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_ID,
)
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY,
    SOURCE_STATE_CANVAS_GRAY64_SHAPE,
    SOURCE_STATE_CANVAS_GRAY64_SOURCE_STATE_BACKED,
    SOURCE_STATE_CANVAS_GRAY64_SURFACE,
    SOURCE_STATE_CANVAS_GRAY64_TRUTH_LEVEL,
    SOURCE_STATE_CANVAS_GRAY64_USES_ALE,
    SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY,
    SOURCE_STATE_GRAY64_SOURCE_STATE_BACKED,
    SOURCE_STATE_GRAY64_SURFACE,
    SOURCE_STATE_GRAY64_TRUTH_LEVEL,
    SOURCE_STATE_GRAY64_USES_ALE,
    SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    DEFAULT_NATURAL_BONUS_SPAWN,
)
from curvyzero.training import exploration_bonus as xb
from curvyzero.training.curvytron_visual_observation import DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_ID,
    CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv,
    DEFAULT_DECISION_MS,
    DEFAULT_DECISION_SOURCE_FRAMES,
    DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY,
    DEFAULT_POLICY_ACTION_REPEAT_MAX,
    DEFAULT_POLICY_ACTION_REPEAT_MIN,
    JOINT_ACTION_COUNT as SOURCE_STATE_JOINT_ACTION_COUNT,
    LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_ID,
    LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_TYPE,
    LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_IMPORT_NAMES,
    OPPONENT_DEATH_MODE_IMMORTAL,
    OPPONENT_DEATH_MODE_NORMAL,
    OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
    OPPONENT_RUNTIME_MODE_NORMAL,
    OPPONENT_TRAINING_RELATION_PROACTIVE_WALL_AVOIDANT,
    SOURCE_STATE_JOINT_ACTION_ADAPTER_IMPL_ID,
    SOURCE_STATE_JOINT_ACTION_RUNTIME_TOPOLOGY,
    SOURCE_STATE_JOINT_ACTION_TRAINING_STATUS,
    SOURCE_STATE_SUPPORTED_BONUS_RENDER_MODES,
    SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES,
    STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
    STACKED_SOURCE_STATE_GRAY64_SHAPE,
)
from curvyzero.training.curvyzero_source_state_visual_turn_commit_lightzero_env import (
    LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_ID,
    LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_TYPE,
    LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_IMPORT_NAMES,
    SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE as TURN_COMMIT_SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE,
    SOURCE_STATE_TURN_COMMIT_ADAPTER_IMPL_ID,
    STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID as TURN_COMMIT_STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
    STACKED_SOURCE_STATE_GRAY64_SHAPE as TURN_COMMIT_STACKED_SOURCE_STATE_GRAY64_SHAPE,
    TURN_COMMIT_TRAINING_STATUS,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env import (
    LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_ID,
    LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_TYPE,
    LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_IMPORT_NAMES,
    LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_ID,
    LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_TYPE,
    LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_IMPORT_NAMES,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    CURRENT_POLICY_SELF_PLAY_BLOCKER,
    CURRENT_POLICY_SELF_PLAY_CLAIM,
    OPPONENT_TRAINING_RELATION_FIXED_STRAIGHT,
    OPPONENT_TRAINING_RELATION_FROZEN_LIGHTZERO_CHECKPOINT,
    STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_ID,
    TURN_COMMIT_CURRENT_POLICY_SELF_PLAY_CLAIM,
    TURN_COMMIT_OPPONENT_TRAINING_RELATION,
    TURN_COMMIT_REWARD_CREDIT_CAVEAT,
    TURN_COMMIT_SIMULTANEOUS_GAME_THEORY_CLAIM,
    TURN_COMMIT_TRUSTED_SELF_PLAY_CLAIM,
)
from curvyzero.training.curvyzero_survival_time_lightzero_smoke import (
    SURVIVAL_TIME_REWARD_SCHEMA_ID,
)
from curvyzero.training.opponent_mixture import (
    OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
    OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
    OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
)
from curvyzero.training.reward_contracts import (
    DEFAULT_MODEL_SUPPORT_CAP,
    DEFAULT_REWARD_OUTCOME_ALPHA,
    DEFAULT_TD_STEPS,
    REWARD_VARIANT_AUTO,
    REWARD_VARIANT_SPARSE_OUTCOME,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
    all_players_alive_diagnostic_reward_policy,
    lightzero_target_config_for_reward,
    normalize_reward_outcome_alpha,
    normalize_reward_variant_for_env,
    reward_policy_for_variant,
)

ENV_VARIANT_FIXED_OPPONENT = "fixed_opponent"
ENV_VARIANT_TURN_COMMIT = "turn_commit"
ENV_VARIANT_SOURCE_STATE_TURN_COMMIT = "source_state_turn_commit"
ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT = "source_state_fixed_opponent"
ENV_VARIANT_SOURCE_STATE_JOINT_ACTION = "source_state_joint_action"
ENV_VARIANT_CHOICES = (
    ENV_VARIANT_FIXED_OPPONENT,
    ENV_VARIANT_TURN_COMMIT,
    ENV_VARIANT_SOURCE_STATE_TURN_COMMIT,
    ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
    ENV_VARIANT_SOURCE_STATE_JOINT_ACTION,
)
DEFAULT_ENV_VARIANT = ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT

DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE = POLICY_BONUS_RENDER_MODE
SOURCE_STATE_BONUS_RENDER_MODE_CHOICES = tuple(SOURCE_STATE_SUPPORTED_BONUS_RENDER_MODES)
DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE = POLICY_TRAIL_RENDER_MODE
SOURCE_STATE_TRAIL_RENDER_MODE_CHOICES = tuple(SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES)
DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE = DEFAULT_POLICY_OBSERVATION_BACKEND
POLICY_OBSERVATION_BACKEND_CHOICES = tuple(POLICY_OBSERVATION_BACKENDS)
DEFAULT_SOURCE_PHYSICS_STEP_MS = DEFAULT_DECISION_MS / DEFAULT_DECISION_SOURCE_FRAMES
DEFAULT_OPPONENT_DEATH_MODE = OPPONENT_DEATH_MODE_NORMAL
DEFAULT_OPPONENT_RUNTIME_MODE = OPPONENT_RUNTIME_MODE_NORMAL
OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION = "none_centralized_joint_action"
OPPONENT_TRAINING_RELATION_WEIGHTED_EPISODE_MIXTURE = (
    "learner_vs_weighted_episode_opponent_mixture"
)
VISUAL_SURVIVAL_EXPERIMENT_SCALE_CURRENT_BROAD = "current_broad"
VISUAL_SURVIVAL_EXPERIMENT_SCALE_CHOICES = (
    VISUAL_SURVIVAL_EXPERIMENT_SCALE_CURRENT_BROAD,
)
DEFAULT_VISUAL_SURVIVAL_EXPERIMENT_SCALE = VISUAL_SURVIVAL_EXPERIMENT_SCALE_CURRENT_BROAD


@lru_cache(maxsize=1)
def source_state_fixed_opponent_wrapper_env_spec_fields() -> dict[str, Any]:
    """Mirror the fixed-opponent wrapper's visual contract in metadata."""

    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {"source_max_steps": 1, "max_ticks": 1}
    )
    try:
        info = env._base_info()
    finally:
        env.close()
    config = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv.config

    def int_list(value: Any) -> list[int]:
        return [int(item) for item in value]

    return {
        "env_type": str(info["lightzero_env_type"]),
        "env_id": str(info["env_id"]),
        "import_names": list(config["lightzero_import_names"]),
        "action_space_size": int(config["action_space_size"]),
        "observation_shape": int_list(info["model_observation_shape"]),
        "observation_schema_id": str(info["observation_schema_id"]),
        "single_frame_schema_id": str(info["single_frame_schema_id"]),
        "raw_observation_schema_id": str(info["raw_observation_schema_id"]),
        "raw_frame_shape": int_list(info["raw_frame_shape"]),
        "grayscale_frame_shape": int_list(info["grayscale_frame_shape"]),
        "frame_stack_proof": str(info["frame_stack_proof"]),
        "debug_fidelity_only": bool(info["debug_fidelity_only"]),
        "source_fidelity_claim": str(info["source_fidelity_claim"]),
        "underlying_env_class": str(info["underlying_env_class"]),
        "runtime_env_impl_id": str(info["runtime_env_impl_id"]),
        "runtime_topology": str(info["runtime_topology"]),
        "two_seat_self_play": bool(info["two_seat_self_play"]),
        "two_seat_self_play_status": str(info["two_seat_self_play_status"]),
        "fixed_opponent_is_two_seat_self_play": bool(info["fixed_opponent_is_two_seat_self_play"]),
        "browser_pixel_fidelity": bool(info["browser_pixel_fidelity"]),
        "uses_ale": bool(info["uses_ale"]),
        "visual_surface": str(info["visual_surface"]),
        "visual_truth_level": str(info["visual_truth_level"]),
        "visual_source_state_backed": bool(info["visual_source_state_backed"]),
    }


def survival_reward_policy() -> dict[str, Any]:
    return {
        "reward_schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
        "survival_only": True,
        "diagnostic_all_players_alive": False,
        "centralized_joint_action_control": False,
        "per_player_reward": False,
        "zero_sum_reward": False,
        "sparse_outcome_reward": False,
        "terminal_outcome_bonus": 0.0,
        "loser_penalty": 0.0,
        "winner_bonus": 0.0,
    }


def env_variant_spec(env_variant: str) -> dict[str, Any]:
    if env_variant == ENV_VARIANT_FIXED_OPPONENT:
        return {
            "env_variant": ENV_VARIANT_FIXED_OPPONENT,
            "env_type": LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_TYPE,
            "env_id": LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_ID,
            "import_names": list(LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_IMPORT_NAMES),
            "action_space_size": 3,
            "reward_schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
            "reward_policy": survival_reward_policy(),
            "current_policy_self_play": CURRENT_POLICY_SELF_PLAY_CLAIM,
            "current_policy_self_play_blocker": CURRENT_POLICY_SELF_PLAY_BLOCKER,
            "current_policy_self_play_caveat": None,
            "trusted_current_policy_self_play": False,
            "simultaneous_game_theory_claim": False,
            "opponent_training_relation": None,
            "turn_commit_adapter": False,
            "observation_shape": list(DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE),
            "observation_schema_id": STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_ID,
            "debug_fidelity_only": True,
            "source_fidelity_claim": "none",
            "single_product_runtime_path": False,
            "legacy_debug_variant": True,
            "underlying_env_class": "legacy_debug_visual_wrapper",
            "runtime_env_impl_id": "legacy_debug_visual_wrapper",
            "runtime_topology": "legacy_debug_visual_fixed_opponent",
            "two_seat_self_play": False,
            "two_seat_self_play_status": "not_two_seat_self_play",
            "fixed_opponent_is_two_seat_self_play": False,
            "browser_pixel_fidelity": False,
            "uses_ale": False,
            "visual_surface": "debug_occupancy_gray64_stack",
            "visual_truth_level": "debug_non_fidelity",
            "visual_source_state_backed": False,
        }
    if env_variant == ENV_VARIANT_TURN_COMMIT:
        return {
            "env_variant": ENV_VARIANT_TURN_COMMIT,
            "env_type": LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_TYPE,
            "env_id": LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_ID,
            "import_names": list(LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_IMPORT_NAMES),
            "action_space_size": 3,
            "reward_schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
            "reward_policy": survival_reward_policy(),
            "current_policy_self_play": TURN_COMMIT_CURRENT_POLICY_SELF_PLAY_CLAIM,
            "current_policy_self_play_blocker": None,
            "current_policy_self_play_caveat": TURN_COMMIT_REWARD_CREDIT_CAVEAT,
            "trusted_current_policy_self_play": TURN_COMMIT_TRUSTED_SELF_PLAY_CLAIM,
            "simultaneous_game_theory_claim": TURN_COMMIT_SIMULTANEOUS_GAME_THEORY_CLAIM,
            "opponent_training_relation": TURN_COMMIT_OPPONENT_TRAINING_RELATION,
            "turn_commit_adapter": True,
            "observation_shape": list(DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE),
            "observation_schema_id": STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_ID,
            "debug_fidelity_only": True,
            "source_fidelity_claim": "none",
            "single_product_runtime_path": False,
            "legacy_debug_variant": True,
            "underlying_env_class": "legacy_debug_visual_turn_commit_wrapper",
            "runtime_env_impl_id": "legacy_debug_visual_turn_commit_wrapper",
            "runtime_topology": "legacy_debug_visual_turn_commit_adapter",
            "two_seat_self_play": False,
            "two_seat_self_play_status": "turn_commit_adapter_not_two_seat_self_play",
            "fixed_opponent_is_two_seat_self_play": False,
            "browser_pixel_fidelity": False,
            "uses_ale": False,
            "visual_surface": "debug_occupancy_gray64_stack",
            "visual_truth_level": "debug_non_fidelity",
            "visual_source_state_backed": False,
        }
    if env_variant == ENV_VARIANT_SOURCE_STATE_TURN_COMMIT:
        return {
            "env_variant": ENV_VARIANT_SOURCE_STATE_TURN_COMMIT,
            "env_type": LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_TYPE,
            "env_id": LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_ID,
            "import_names": list(LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_IMPORT_NAMES),
            "action_space_size": 3,
            "reward_schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
            "reward_policy": survival_reward_policy(),
            "current_policy_self_play": TURN_COMMIT_CURRENT_POLICY_SELF_PLAY_CLAIM,
            "current_policy_self_play_blocker": None,
            "current_policy_self_play_caveat": TURN_COMMIT_REWARD_CREDIT_CAVEAT,
            "trusted_current_policy_self_play": TURN_COMMIT_TRUSTED_SELF_PLAY_CLAIM,
            "simultaneous_game_theory_claim": TURN_COMMIT_SIMULTANEOUS_GAME_THEORY_CLAIM,
            "opponent_training_relation": TURN_COMMIT_OPPONENT_TRAINING_RELATION,
            "turn_commit_adapter": True,
            "observation_shape": list(TURN_COMMIT_STACKED_SOURCE_STATE_GRAY64_SHAPE),
            "observation_schema_id": TURN_COMMIT_STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
            "raw_observation_schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
            "raw_frame_shape": list(TURN_COMMIT_SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE),
            "grayscale_frame_shape": list(SOURCE_STATE_CANVAS_GRAY64_SHAPE),
            "debug_fidelity_only": False,
            "source_fidelity_claim": "source_state_backed_non_browser_pixel",
            "single_product_runtime_path": True,
            "legacy_debug_variant": False,
            "underlying_env_class": "source_state_visual_turn_commit_wrapper",
            "runtime_env_impl_id": SOURCE_STATE_TURN_COMMIT_ADAPTER_IMPL_ID,
            "runtime_topology": (
                "stock_lightzero_scalar_turn_commit_over_simultaneous_source_state_env"
            ),
            "two_seat_self_play": False,
            "current_policy_two_seat_action_collection": True,
            "two_seat_self_play_status": TURN_COMMIT_TRAINING_STATUS,
            "fixed_opponent_is_two_seat_self_play": False,
            "browser_pixel_fidelity": SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY,
            "uses_ale": SOURCE_STATE_CANVAS_GRAY64_USES_ALE,
            "visual_surface": SOURCE_STATE_CANVAS_GRAY64_SURFACE,
            "visual_truth_level": SOURCE_STATE_CANVAS_GRAY64_TRUTH_LEVEL,
            "visual_source_state_backed": SOURCE_STATE_CANVAS_GRAY64_SOURCE_STATE_BACKED,
        }
    if env_variant == ENV_VARIANT_SOURCE_STATE_JOINT_ACTION:
        return {
            "env_variant": ENV_VARIANT_SOURCE_STATE_JOINT_ACTION,
            "env_type": LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_TYPE,
            "env_id": LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_ID,
            "import_names": list(LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_IMPORT_NAMES),
            "action_space_size": SOURCE_STATE_JOINT_ACTION_COUNT,
            "reward_schema_id": ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_ID,
            "reward_policy": all_players_alive_diagnostic_reward_policy(),
            "current_policy_self_play": False,
            "current_policy_self_play_blocker": (
                "centralized_joint_action_control_is_not_true_competitive_self_play"
            ),
            "current_policy_self_play_caveat": (
                "Centralized joint-action control: one policy chooses both players' actions."
            ),
            "trusted_current_policy_self_play": False,
            "simultaneous_game_theory_claim": False,
            "opponent_training_relation": "centralized_policy_controls_both_players",
            "turn_commit_adapter": False,
            "observation_shape": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
            "observation_schema_id": STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
            "debug_fidelity_only": False,
            "source_fidelity_claim": "source_state_backed_non_browser_pixel",
            "single_product_runtime_path": True,
            "legacy_debug_variant": False,
            "underlying_env_class": "source_state_visual_joint_action_wrapper",
            "runtime_env_impl_id": SOURCE_STATE_JOINT_ACTION_ADAPTER_IMPL_ID,
            "runtime_topology": SOURCE_STATE_JOINT_ACTION_RUNTIME_TOPOLOGY,
            "two_seat_self_play": False,
            "two_seat_self_play_status": SOURCE_STATE_JOINT_ACTION_TRAINING_STATUS,
            "fixed_opponent_is_two_seat_self_play": False,
            "browser_pixel_fidelity": SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY,
            "uses_ale": SOURCE_STATE_GRAY64_USES_ALE,
            "visual_surface": SOURCE_STATE_GRAY64_SURFACE,
            "visual_truth_level": SOURCE_STATE_GRAY64_TRUTH_LEVEL,
            "visual_source_state_backed": SOURCE_STATE_GRAY64_SOURCE_STATE_BACKED,
        }
    if env_variant == ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT:
        wrapper_spec = source_state_fixed_opponent_wrapper_env_spec_fields()
        return {
            "env_variant": ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
            **wrapper_spec,
            "reward_schema_id": SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_ID,
            "reward_policy": {
                "reward_variant": REWARD_VARIANT_SPARSE_OUTCOME,
                "reward_schema_id": SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_ID,
                "survival_length_is_eval_metric": True,
                "dense_survival_reward": False,
                "survival_only": False,
                "diagnostic_all_players_alive": False,
                "centralized_joint_action_control": False,
                "per_player_reward": True,
                "zero_sum_reward": True,
                "sparse_outcome_reward": True,
                "terminal_outcome_bonus": 1.0,
                "nonterminal_reward": 0.0,
                "loser_penalty": -1.0,
                "winner_bonus": 1.0,
                "draw_bonus": 0.0,
                "truncation_bonus": 0.0,
            },
            "current_policy_self_play": CURRENT_POLICY_SELF_PLAY_CLAIM,
            "current_policy_self_play_blocker": CURRENT_POLICY_SELF_PLAY_BLOCKER,
            "current_policy_self_play_caveat": None,
            "trusted_current_policy_self_play": False,
            "simultaneous_game_theory_claim": False,
            "opponent_training_relation": OPPONENT_TRAINING_RELATION_FIXED_STRAIGHT,
            "turn_commit_adapter": False,
            "single_product_runtime_path": True,
            "legacy_debug_variant": False,
        }
    raise ValueError(f"unknown env_variant: {env_variant!r}")


def validate_source_state_trail_render_mode(value: str) -> str:
    mode = str(value)
    if mode not in SOURCE_STATE_TRAIL_RENDER_MODE_CHOICES:
        raise ValueError(
            "source_state_trail_render_mode must be one of "
            f"{SOURCE_STATE_TRAIL_RENDER_MODE_CHOICES!r}; got {mode!r}"
        )
    return mode


def validate_source_state_bonus_render_mode(value: str) -> str:
    mode = str(value)
    if mode not in SOURCE_STATE_BONUS_RENDER_MODE_CHOICES:
        raise ValueError(
            "source_state_bonus_render_mode must be one of "
            f"{SOURCE_STATE_BONUS_RENDER_MODE_CHOICES!r}; got {mode!r}"
        )
    return mode


def validate_policy_observation_backend(value: str) -> str:
    backend = str(value)
    if backend not in POLICY_OBSERVATION_BACKEND_CHOICES:
        raise ValueError(
            "policy_observation_backend must be one of "
            f"{POLICY_OBSERVATION_BACKEND_CHOICES!r}; got {backend!r}"
        )
    return backend


def validate_learner_seat_mode(
    value: str | None,
    *,
    absent_default: str = DEFAULT_LEARNER_SEAT_MODE,
) -> str:
    mode = str(absent_default if value is None else value)
    if mode not in LEARNER_SEAT_MODE_CHOICES:
        raise ValueError(
            f"learner_seat_mode must be one of {LEARNER_SEAT_MODE_CHOICES!r}; got {mode!r}"
        )
    return mode


def validate_trusted_source_state_action_cadence(
    *,
    env_variant: str,
    decision_ms: float,
    decision_source_frames: int = DEFAULT_DECISION_SOURCE_FRAMES,
    source_physics_step_ms: float = DEFAULT_SOURCE_PHYSICS_STEP_MS,
    source_max_steps_semantics: str = "source_physics_steps",
    context: str,
) -> None:
    if env_variant != ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT:
        return
    if (
        int(decision_source_frames) == int(DEFAULT_DECISION_SOURCE_FRAMES)
        and math.isclose(
            float(source_physics_step_ms),
            float(DEFAULT_SOURCE_PHYSICS_STEP_MS),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and str(source_max_steps_semantics) == "source_physics_steps"
        and math.isclose(
            float(decision_ms),
            float(DEFAULT_DECISION_MS),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        return
    raise ValueError(
        f"{context} uses the trusted source_state_fixed_opponent lane, where "
        "one LightZero policy action must advance exactly one CurvyTron source "
        f"physics step. decision_ms must be {DEFAULT_DECISION_MS:g}; got "
        f"{float(decision_ms):g}; decision_source_frames must be "
        f"{DEFAULT_DECISION_SOURCE_FRAMES}; got {int(decision_source_frames)}; "
        f"source_physics_step_ms must be {DEFAULT_SOURCE_PHYSICS_STEP_MS:g}; got "
        f"{float(source_physics_step_ms):g}. Use policy_action_repeat_* for "
        "explicit action repeat instead of hiding repeat in cadence fields."
    )


def reward_policy_for_config(
    *,
    env_variant: str,
    reward_variant: str,
    reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
) -> dict[str, Any]:
    if env_variant in (
        ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        ENV_VARIANT_SOURCE_STATE_JOINT_ACTION,
    ):
        return reward_policy_for_variant(
            env_variant=env_variant,
            reward_variant=reward_variant,
            reward_outcome_alpha=reward_outcome_alpha,
        )
    env_spec = env_variant_spec(env_variant)
    if reward_variant != REWARD_VARIANT_AUTO:
        raise ValueError(
            f"{env_variant} does not support explicit reward_variant {reward_variant!r}"
        )
    return dict(env_spec["reward_policy"])


def opponent_training_relation(opponent_policy_kind: str) -> str:
    if opponent_policy_kind == OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION:
        return "centralized_policy_controls_both_players"
    if opponent_policy_kind == OPPONENT_POLICY_KIND_FIXED_STRAIGHT:
        return OPPONENT_TRAINING_RELATION_FIXED_STRAIGHT
    if opponent_policy_kind == OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT:
        return OPPONENT_TRAINING_RELATION_PROACTIVE_WALL_AVOIDANT
    if opponent_policy_kind == OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT:
        return OPPONENT_TRAINING_RELATION_FROZEN_LIGHTZERO_CHECKPOINT
    return f"unknown:{opponent_policy_kind}"


def opponent_training_relation_for_surface(
    *,
    env_variant: str,
    opponent_policy_kind: str,
    env_spec: dict[str, Any],
    opponent_mixture: dict[str, Any] | None,
) -> str:
    if opponent_mixture is not None:
        return OPPONENT_TRAINING_RELATION_WEIGHTED_EPISODE_MIXTURE
    if env_variant == ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT:
        return opponent_training_relation(opponent_policy_kind)
    return env_spec["opponent_training_relation"] or opponent_training_relation(
        opponent_policy_kind
    )


def easy_config(mapping: dict[str, Any]) -> Any:
    try:
        from easydict import EasyDict
    except ModuleNotFoundError:
        return mapping
    return EasyDict(mapping)


@dataclass(frozen=True)
class FrozenOpponentConfig:
    checkpoint: Mapping[str, Any] | None = None
    snapshot_ref: str | None = None
    checkpoint_state_key: str | None = None
    use_cuda: bool = False


@dataclass(frozen=True)
class VisualSurvivalRunRuntimeSpec:
    seed: int
    exp_name: Any
    telemetry_path: Any
    cuda: bool
    lightzero_multi_gpu: bool
    env_manager_type: str
    profile_env_timing_enabled: bool = False


@dataclass(frozen=True)
class VisualSurvivalTrainingScaleSpec:
    max_env_step: int
    source_max_steps: int
    collector_env_num: int
    evaluator_env_num: int
    n_evaluator_episode: int
    n_episode: int
    num_simulations: int
    batch_size: int
    lightzero_eval_freq: int
    max_train_iter: int
    save_ckpt_after_iter: int


@dataclass(frozen=True)
class VisualSurvivalTimingSpec:
    decision_ms: float
    decision_source_frames: int = DEFAULT_DECISION_SOURCE_FRAMES
    source_physics_step_ms: float = DEFAULT_SOURCE_PHYSICS_STEP_MS
    source_max_steps_semantics: str = "source_physics_steps"


@dataclass(frozen=True)
class VisualSurvivalObservationSpec:
    env_variant: str
    source_state_trail_render_mode: str = DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE
    source_state_bonus_render_mode: str = DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE
    policy_observation_backend: str = DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE
    learner_seat_mode: str = DEFAULT_LEARNER_SEAT_MODE


@dataclass(frozen=True)
class VisualSurvivalBehaviorSpec:
    ego_action_straight_override_probability: float
    control_noise_profile_id: str
    disable_death_for_profile: bool
    env_telemetry_stride: int
    policy_action_repeat_min: int = DEFAULT_POLICY_ACTION_REPEAT_MIN
    policy_action_repeat_max: int = DEFAULT_POLICY_ACTION_REPEAT_MAX
    policy_action_repeat_extra_probability: float = DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
    natural_bonus_spawn: bool = DEFAULT_NATURAL_BONUS_SPAWN


@dataclass(frozen=True)
class VisualSurvivalRewardTargetSpec:
    reward_variant: str
    reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA
    model_support_cap: int | None = DEFAULT_MODEL_SUPPORT_CAP
    td_steps: int | None = DEFAULT_TD_STEPS


@dataclass(frozen=True)
class VisualSurvivalOpponentSpec:
    opponent_policy_kind: str
    frozen_opponent: FrozenOpponentConfig = field(default_factory=FrozenOpponentConfig)
    opponent_mixture: Mapping[str, Any] | None = None
    opponent_assignment_context: Mapping[str, Any] | None = None
    opponent_death_mode: str = DEFAULT_OPPONENT_DEATH_MODE
    opponent_runtime_mode: str = DEFAULT_OPPONENT_RUNTIME_MODE


@dataclass(frozen=True)
class VisualSurvivalConfigSpec:
    run: VisualSurvivalRunRuntimeSpec
    training: VisualSurvivalTrainingScaleSpec
    timing: VisualSurvivalTimingSpec
    observation: VisualSurvivalObservationSpec
    behavior: VisualSurvivalBehaviorSpec
    reward: VisualSurvivalRewardTargetSpec
    opponent: VisualSurvivalOpponentSpec
    exploration_bonus: xb.ExplorationBonusSpec = field(
        default_factory=xb.normalize_exploration_bonus_spec
    )

    @classmethod
    def from_builder_kwargs(
        cls,
        kwargs: Mapping[str, Any],
    ) -> "VisualSurvivalConfigSpec":
        values = dict(kwargs)
        missing = object()

        def pop(name: str, default: Any = missing) -> Any:
            if name in values:
                return values.pop(name)
            if default is missing:
                raise TypeError(f"missing required visual survival builder kwarg: {name}")
            return default

        frozen_opponent = FrozenOpponentConfig(
            checkpoint=pop("opponent_checkpoint", None),
            snapshot_ref=pop("opponent_snapshot_ref", None),
            checkpoint_state_key=pop("opponent_checkpoint_state_key", None),
            use_cuda=pop("opponent_use_cuda", False),
        )
        spec = cls(
            run=VisualSurvivalRunRuntimeSpec(
                seed=pop("seed"),
                exp_name=pop("exp_name"),
                telemetry_path=pop("telemetry_path"),
                cuda=pop("cuda"),
                lightzero_multi_gpu=pop("lightzero_multi_gpu"),
                env_manager_type=pop("env_manager_type"),
                profile_env_timing_enabled=pop("profile_env_timing_enabled", False),
            ),
            training=VisualSurvivalTrainingScaleSpec(
                max_env_step=pop("max_env_step"),
                source_max_steps=pop("source_max_steps"),
                collector_env_num=pop("collector_env_num"),
                evaluator_env_num=pop("evaluator_env_num"),
                n_evaluator_episode=pop("n_evaluator_episode"),
                n_episode=pop("n_episode"),
                num_simulations=pop("num_simulations"),
                batch_size=pop("batch_size"),
                lightzero_eval_freq=pop("lightzero_eval_freq"),
                max_train_iter=pop("max_train_iter"),
                save_ckpt_after_iter=pop("save_ckpt_after_iter"),
            ),
            timing=VisualSurvivalTimingSpec(
                decision_ms=pop("decision_ms"),
                decision_source_frames=pop(
                    "decision_source_frames", DEFAULT_DECISION_SOURCE_FRAMES
                ),
                source_physics_step_ms=pop(
                    "source_physics_step_ms", DEFAULT_SOURCE_PHYSICS_STEP_MS
                ),
                source_max_steps_semantics=pop(
                    "source_max_steps_semantics", "source_physics_steps"
                ),
            ),
            observation=VisualSurvivalObservationSpec(
                env_variant=pop("env_variant"),
                source_state_trail_render_mode=pop(
                    "source_state_trail_render_mode",
                    DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE,
                ),
                source_state_bonus_render_mode=pop(
                    "source_state_bonus_render_mode",
                    DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE,
                ),
                policy_observation_backend=pop(
                    "policy_observation_backend",
                    DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE,
                ),
                learner_seat_mode=pop("learner_seat_mode", DEFAULT_LEARNER_SEAT_MODE),
            ),
            behavior=VisualSurvivalBehaviorSpec(
                ego_action_straight_override_probability=pop(
                    "ego_action_straight_override_probability"
                ),
                control_noise_profile_id=pop("control_noise_profile_id"),
                disable_death_for_profile=pop("disable_death_for_profile"),
                env_telemetry_stride=pop("env_telemetry_stride"),
                policy_action_repeat_min=pop(
                    "policy_action_repeat_min", DEFAULT_POLICY_ACTION_REPEAT_MIN
                ),
                policy_action_repeat_max=pop(
                    "policy_action_repeat_max", DEFAULT_POLICY_ACTION_REPEAT_MAX
                ),
                policy_action_repeat_extra_probability=pop(
                    "policy_action_repeat_extra_probability",
                    DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY,
                ),
                natural_bonus_spawn=pop("natural_bonus_spawn", DEFAULT_NATURAL_BONUS_SPAWN),
            ),
            reward=VisualSurvivalRewardTargetSpec(
                reward_variant=pop("reward_variant"),
                reward_outcome_alpha=pop(
                    "reward_outcome_alpha", DEFAULT_REWARD_OUTCOME_ALPHA
                ),
                model_support_cap=pop("model_support_cap", DEFAULT_MODEL_SUPPORT_CAP),
                td_steps=pop("td_steps", DEFAULT_TD_STEPS),
            ),
            opponent=VisualSurvivalOpponentSpec(
                opponent_policy_kind=pop("opponent_policy_kind"),
                frozen_opponent=frozen_opponent,
                opponent_mixture=pop("opponent_mixture", None),
                opponent_assignment_context=pop("opponent_assignment_context", None),
                opponent_death_mode=pop("opponent_death_mode", DEFAULT_OPPONENT_DEATH_MODE),
                opponent_runtime_mode=pop(
                    "opponent_runtime_mode", DEFAULT_OPPONENT_RUNTIME_MODE
                ),
            ),
            exploration_bonus=xb.normalize_exploration_bonus_config(
                pop("exploration_bonus", None)
            ),
        )
        if values:
            names = ", ".join(sorted(values))
            raise TypeError(f"unknown visual survival builder kwargs: {names}")
        return spec

    def to_builder_kwargs(self) -> dict[str, Any]:
        frozen_opponent = self.opponent.frozen_opponent
        return {
            "seed": self.run.seed,
            "exp_name": self.run.exp_name,
            "telemetry_path": self.run.telemetry_path,
            "cuda": self.run.cuda,
            "max_env_step": self.training.max_env_step,
            "source_max_steps": self.training.source_max_steps,
            "decision_ms": self.timing.decision_ms,
            "collector_env_num": self.training.collector_env_num,
            "evaluator_env_num": self.training.evaluator_env_num,
            "n_evaluator_episode": self.training.n_evaluator_episode,
            "n_episode": self.training.n_episode,
            "num_simulations": self.training.num_simulations,
            "batch_size": self.training.batch_size,
            "lightzero_eval_freq": self.training.lightzero_eval_freq,
            "lightzero_multi_gpu": self.run.lightzero_multi_gpu,
            "max_train_iter": self.training.max_train_iter,
            "save_ckpt_after_iter": self.training.save_ckpt_after_iter,
            "env_variant": self.observation.env_variant,
            "reward_variant": self.reward.reward_variant,
            "reward_outcome_alpha": self.reward.reward_outcome_alpha,
            "ego_action_straight_override_probability": (
                self.behavior.ego_action_straight_override_probability
            ),
            "control_noise_profile_id": self.behavior.control_noise_profile_id,
            "disable_death_for_profile": self.behavior.disable_death_for_profile,
            "env_telemetry_stride": self.behavior.env_telemetry_stride,
            "env_manager_type": self.run.env_manager_type,
            "opponent_policy_kind": self.opponent.opponent_policy_kind,
            "opponent_use_cuda": frozen_opponent.use_cuda,
            "opponent_checkpoint": frozen_opponent.checkpoint,
            "opponent_snapshot_ref": frozen_opponent.snapshot_ref,
            "opponent_checkpoint_state_key": frozen_opponent.checkpoint_state_key,
            "decision_source_frames": self.timing.decision_source_frames,
            "source_physics_step_ms": self.timing.source_physics_step_ms,
            "source_max_steps_semantics": self.timing.source_max_steps_semantics,
            "opponent_mixture": self.opponent.opponent_mixture,
            "opponent_assignment_context": self.opponent.opponent_assignment_context,
            "source_state_trail_render_mode": (
                self.observation.source_state_trail_render_mode
            ),
            "source_state_bonus_render_mode": (
                self.observation.source_state_bonus_render_mode
            ),
            "policy_observation_backend": self.observation.policy_observation_backend,
            "learner_seat_mode": self.observation.learner_seat_mode,
            "policy_action_repeat_min": self.behavior.policy_action_repeat_min,
            "policy_action_repeat_max": self.behavior.policy_action_repeat_max,
            "policy_action_repeat_extra_probability": (
                self.behavior.policy_action_repeat_extra_probability
            ),
            "natural_bonus_spawn": self.behavior.natural_bonus_spawn,
            "profile_env_timing_enabled": self.run.profile_env_timing_enabled,
            "opponent_death_mode": self.opponent.opponent_death_mode,
            "opponent_runtime_mode": self.opponent.opponent_runtime_mode,
            "model_support_cap": self.reward.model_support_cap,
            "td_steps": self.reward.td_steps,
            "exploration_bonus": self.exploration_bonus.as_dict(),
        }


@dataclass(frozen=True)
class VisualSurvivalExperimentSpec:
    """Small experiment-facing surface that expands to normalized builder config."""

    seed: int
    exp_name: Any
    telemetry_path: Any
    reward_variant: str = REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME
    reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA
    opponent_policy_kind: str = OPPONENT_POLICY_KIND_FIXED_STRAIGHT
    frozen_opponent: FrozenOpponentConfig = field(default_factory=FrozenOpponentConfig)
    opponent_mixture: Mapping[str, Any] | None = None
    opponent_assignment_context: Mapping[str, Any] | None = None
    action_noise_probability: float = 0.0
    scale_preset: str = DEFAULT_VISUAL_SURVIVAL_EXPERIMENT_SCALE


def _experiment_scale_training_spec(scale_preset: str) -> VisualSurvivalTrainingScaleSpec:
    if scale_preset != VISUAL_SURVIVAL_EXPERIMENT_SCALE_CURRENT_BROAD:
        raise ValueError(
            "unknown visual survival experiment scale preset "
            f"{scale_preset!r}; expected one of {VISUAL_SURVIVAL_EXPERIMENT_SCALE_CHOICES!r}"
        )
    return VisualSurvivalTrainingScaleSpec(
        max_env_step=CURVYTRON_DEFAULT_MAX_ENV_STEP,
        source_max_steps=CURVYTRON_SOURCE_MAX_STEPS,
        collector_env_num=CURVYTRON_DEFAULT_COLLECTOR_ENV_NUM,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        n_episode=CURVYTRON_DEFAULT_N_EPISODE,
        num_simulations=CURVYTRON_DEFAULT_NUM_SIMULATIONS,
        batch_size=CURVYTRON_DEFAULT_TRAIN_BATCH_SIZE,
        lightzero_eval_freq=0,
        max_train_iter=CURVYTRON_DEFAULT_MAX_TRAIN_ITER,
        save_ckpt_after_iter=CURVYTRON_SAVE_CKPT_AFTER_ITER,
    )


def _action_noise_profile_id(probability: float) -> str:
    probability = float(probability)
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"action_noise_probability must be in [0, 1], got {probability!r}")
    if probability == 0.0:
        return "none"
    percent = int(round(probability * 100))
    if not math.isclose(probability, percent / 100.0):
        raise ValueError(
            "action_noise_probability must be representable as a whole percent; "
            f"got {probability!r}"
        )
    return f"straight_override_p{percent}_repeat_p{percent}"


def visual_survival_config_spec_from_experiment(
    experiment: VisualSurvivalExperimentSpec,
) -> VisualSurvivalConfigSpec:
    action_noise_probability = float(experiment.action_noise_probability)
    return VisualSurvivalConfigSpec(
        run=VisualSurvivalRunRuntimeSpec(
            seed=experiment.seed,
            exp_name=experiment.exp_name,
            telemetry_path=experiment.telemetry_path,
            cuda=True,
            lightzero_multi_gpu=False,
            env_manager_type="subprocess",
            profile_env_timing_enabled=False,
        ),
        training=_experiment_scale_training_spec(experiment.scale_preset),
        timing=VisualSurvivalTimingSpec(
            decision_ms=CURVYTRON_DECISION_MS,
            decision_source_frames=CURVYTRON_DECISION_SOURCE_FRAMES,
            source_physics_step_ms=CURVYTRON_DECISION_MS / CURVYTRON_DECISION_SOURCE_FRAMES,
            source_max_steps_semantics="source_physics_steps",
        ),
        observation=VisualSurvivalObservationSpec(
            env_variant=ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
            source_state_trail_render_mode=DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE,
            source_state_bonus_render_mode=DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE,
            policy_observation_backend=DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE,
            learner_seat_mode=DEFAULT_LEARNER_SEAT_MODE,
        ),
        behavior=VisualSurvivalBehaviorSpec(
            ego_action_straight_override_probability=action_noise_probability,
            control_noise_profile_id=_action_noise_profile_id(action_noise_probability),
            disable_death_for_profile=False,
            env_telemetry_stride=1,
            policy_action_repeat_min=DEFAULT_POLICY_ACTION_REPEAT_MIN,
            policy_action_repeat_max=2
            if action_noise_probability > 0.0
            else DEFAULT_POLICY_ACTION_REPEAT_MAX,
            policy_action_repeat_extra_probability=action_noise_probability,
            natural_bonus_spawn=DEFAULT_NATURAL_BONUS_SPAWN,
        ),
        reward=VisualSurvivalRewardTargetSpec(
            reward_variant=experiment.reward_variant,
            reward_outcome_alpha=experiment.reward_outcome_alpha,
            model_support_cap=DEFAULT_MODEL_SUPPORT_CAP,
            td_steps=DEFAULT_TD_STEPS,
        ),
        opponent=VisualSurvivalOpponentSpec(
            opponent_policy_kind=experiment.opponent_policy_kind,
            frozen_opponent=experiment.frozen_opponent,
            opponent_mixture=experiment.opponent_mixture,
            opponent_assignment_context=experiment.opponent_assignment_context,
            opponent_death_mode=DEFAULT_OPPONENT_DEATH_MODE,
            opponent_runtime_mode=DEFAULT_OPPONENT_RUNTIME_MODE,
        ),
    )


def build_visual_survival_config_from_experiment(
    experiment: VisualSurvivalExperimentSpec,
) -> VisualSurvivalConfigResult:
    return build_visual_survival_config(
        visual_survival_config_spec_from_experiment(experiment)
    )


@dataclass(frozen=True)
class VisualSurvivalConfigResult:
    template_module: str
    main_config: Any
    create_config: Any
    surface: dict[str, Any]
    patches: list[dict[str, Any]]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VisualSurvivalConfigResult":
        return cls(
            template_module=str(value["template_module"]),
            main_config=value["main_config"],
            create_config=value["create_config"],
            surface=dict(value["surface"]),
            patches=list(value["patches"]),
        )

    @property
    def env_config(self) -> Any:
        return self.main_config["env"]

    @property
    def lightzero_target_config(self) -> Any:
        return self.env_config["lightzero_target_config"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "template_module": self.template_module,
            "main_config": self.main_config,
            "create_config": self.create_config,
            "surface": self.surface,
            "patches": self.patches,
        }


LightZeroConfigBuildResult = VisualSurvivalConfigResult


def build_visual_survival_configs_from_spec(
    spec: VisualSurvivalConfigSpec,
) -> VisualSurvivalConfigResult:
    return build_visual_survival_config(spec)


def build_visual_survival_config(
    spec: VisualSurvivalConfigSpec,
) -> VisualSurvivalConfigResult:
    return VisualSurvivalConfigResult.from_mapping(
        _build_visual_survival_configs_from_builder_kwargs(**spec.to_builder_kwargs())
    )


def to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return to_plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def set_or_add_path(mapping: Any, path: tuple[str, ...], value: Any) -> dict[str, Any]:
    current = mapping
    for part in path[:-1]:
        if part not in current or current[part] is None:
            current[part] = {}
        current = current[part]
    key = path[-1]
    old_value = current.get(key, "<missing>")
    current[key] = value
    return {"path": ".".join(path), "old": to_plain(old_value), "new": to_plain(value)}


def get_path(mapping: Any, path: tuple[str, ...], default: Any = None) -> Any:
    current = mapping
    try:
        for part in path:
            current = current[part]
    except KeyError:
        return default
    return current


def target_config_patches(
    main_config: Any,
    target_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not target_config:
        return []
    patches: list[dict[str, Any]] = []
    key_paths = {
        "discount_factor": ("policy", "discount_factor"),
        "td_steps": ("policy", "td_steps"),
        "model_support_scale": ("policy", "model", "support_scale"),
        "model_reward_support_size": ("policy", "model", "reward_support_size"),
        "model_value_support_size": ("policy", "model", "value_support_size"),
        "model_reward_support_range": ("policy", "model", "reward_support_range"),
        "model_value_support_range": ("policy", "model", "value_support_range"),
    }
    for key, path in key_paths.items():
        if key not in target_config:
            continue
        patch = set_or_add_path(main_config, path, target_config[key])
        patch["reason"] = "make LightZero value/reward target range match CurvyTron reward variant"
        patches.append(patch)
    return patches


def target_config_with_exploration_bonus_bound(
    target_config: dict[str, Any],
    exploration_bonus_spec: xb.ExplorationBonusSpec,
    *,
    source_max_steps: int,
) -> dict[str, Any]:
    adjusted = dict(target_config)
    if exploration_bonus_spec.mode != xb.EXPLORATION_BONUS_MODE_RND_REPLAY_TARGET_V0:
        return adjusted

    reward_bound = float(exploration_bonus_spec.weight)
    reward_support_extra = int(math.ceil(reward_bound))
    value_bound = reward_bound * float(source_max_steps)
    value_support_extra = int(math.ceil(value_bound))
    reward_requested = int(adjusted["model_reward_support_requested_scale"]) + (
        reward_support_extra
    )
    value_requested = int(adjusted["model_value_support_requested_scale"]) + (
        value_support_extra
    )
    support_cap_raw = adjusted.get("model_support_cap")
    support_cap = None if support_cap_raw is None else int(support_cap_raw)
    if support_cap is None:
        reward_effective = reward_requested
        value_effective = value_requested
        reward_capped = False
        value_capped = False
    else:
        reward_effective = min(reward_requested, support_cap)
        value_effective = min(value_requested, support_cap)
        reward_capped = reward_requested > support_cap
        value_capped = value_requested > support_cap
    model_support_scale = max(reward_effective, value_effective)
    model_support_size = int(2 * model_support_scale + 1)
    model_support_range = (
        -float(model_support_scale),
        float(model_support_scale + 1),
        1.0,
    )
    adjusted.update(
        {
            "model_support_scale": int(model_support_scale),
            "model_reward_support_size": model_support_size,
            "model_value_support_size": model_support_size,
            "model_reward_support_range": model_support_range,
            "model_value_support_range": model_support_range,
            "model_reward_support_capped": reward_capped,
            "model_value_support_capped": value_capped,
            "model_reward_support_requested_scale": int(reward_requested),
            "model_value_support_requested_scale": int(value_requested),
            "model_reward_support_effective_scale": int(model_support_scale),
            "model_value_support_effective_scale": int(model_support_scale),
            "uncapped_model_reward_support_scale": int(reward_requested),
            "uncapped_model_value_support_scale": int(value_requested),
            "exploration_bonus_support_adjusted": True,
            "exploration_bonus_intrinsic_reward_bound": reward_bound,
            "exploration_bonus_intrinsic_value_bound": value_bound,
            "exploration_bonus_intrinsic_reward_support_extra": int(
                reward_support_extra
            ),
            "exploration_bonus_intrinsic_value_support_extra": int(value_support_extra),
        }
    )
    return adjusted


def set_save_ckpt_after_iter(main_config: Any, value: int) -> dict[str, Any]:
    current = main_config["policy"]
    for part in ("learn", "learner", "hook"):
        if part not in current or current[part] is None:
            current[part] = {}
        current = current[part]
    old_value = current.get("save_ckpt_after_iter")
    current["save_ckpt_after_iter"] = int(value)
    return {
        "path": "policy.learn.learner.hook.save_ckpt_after_iter",
        "old": to_plain(old_value),
        "new": int(value),
        "reason": "first CurvyTron visual survival run checkpoints frequently for inspection",
    }


def set_load_ckpt_before_run(
    main_config: Any,
    checkpoint_path: str,
    *,
    reason: str = "automatic resume from the latest iteration checkpoint for this run",
) -> dict[str, Any]:
    current = main_config["policy"]
    for part in ("learn", "learner", "hook"):
        if part not in current or current[part] is None:
            current[part] = {}
        current = current[part]
    old_value = current.get("load_ckpt_before_run")
    current["load_ckpt_before_run"] = checkpoint_path
    return {
        "path": "policy.learn.learner.hook.load_ckpt_before_run",
        "old": to_plain(old_value),
        "new": checkpoint_path,
        "reason": reason,
    }


def _build_visual_survival_configs_from_builder_kwargs(
    *,
    seed: int,
    exp_name: Any,
    telemetry_path: Any,
    cuda: bool,
    max_env_step: int,
    source_max_steps: int,
    decision_ms: float,
    collector_env_num: int,
    evaluator_env_num: int,
    n_evaluator_episode: int,
    n_episode: int,
    num_simulations: int,
    batch_size: int,
    lightzero_eval_freq: int,
    lightzero_multi_gpu: bool,
    max_train_iter: int,
    save_ckpt_after_iter: int,
    env_variant: str,
    reward_variant: str,
    reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
    ego_action_straight_override_probability: float,
    control_noise_profile_id: str,
    disable_death_for_profile: bool,
    env_telemetry_stride: int,
    env_manager_type: str,
    opponent_policy_kind: str,
    opponent_use_cuda: bool,
    opponent_checkpoint: dict[str, Any] | None,
    opponent_snapshot_ref: str | None,
    opponent_checkpoint_state_key: str | None,
    decision_source_frames: int = DEFAULT_DECISION_SOURCE_FRAMES,
    source_physics_step_ms: float = DEFAULT_SOURCE_PHYSICS_STEP_MS,
    source_max_steps_semantics: str = "source_physics_steps",
    opponent_mixture: dict[str, Any] | None = None,
    opponent_assignment_context: dict[str, Any] | None = None,
    source_state_trail_render_mode: str = DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE,
    source_state_bonus_render_mode: str = DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE,
    policy_observation_backend: str = DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE,
    learner_seat_mode: str = DEFAULT_LEARNER_SEAT_MODE,
    policy_action_repeat_min: int = DEFAULT_POLICY_ACTION_REPEAT_MIN,
    policy_action_repeat_max: int = DEFAULT_POLICY_ACTION_REPEAT_MAX,
    policy_action_repeat_extra_probability: float = DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY,
    natural_bonus_spawn: bool = DEFAULT_NATURAL_BONUS_SPAWN,
    profile_env_timing_enabled: bool = False,
    opponent_death_mode: str = DEFAULT_OPPONENT_DEATH_MODE,
    opponent_runtime_mode: str = DEFAULT_OPPONENT_RUNTIME_MODE,
    model_support_cap: int | None = DEFAULT_MODEL_SUPPORT_CAP,
    td_steps: int | None = DEFAULT_TD_STEPS,
    exploration_bonus: Mapping[str, Any] | xb.ExplorationBonusSpec | str | None = None,
) -> dict[str, Any]:
    template_module = "zoo.atari.config.atari_muzero_config"
    module = importlib.import_module(template_module)
    env_spec = env_variant_spec(env_variant)
    reward_variant = normalize_reward_variant_for_env(
        env_variant=env_variant,
        reward_variant=reward_variant,
    )
    reward_outcome_alpha = normalize_reward_outcome_alpha(reward_outcome_alpha)
    reward_policy = reward_policy_for_config(
        env_variant=env_variant,
        reward_variant=reward_variant,
        reward_outcome_alpha=reward_outcome_alpha,
    )
    source_state_trail_render_mode = validate_source_state_trail_render_mode(
        source_state_trail_render_mode
    )
    source_state_bonus_render_mode = validate_source_state_bonus_render_mode(
        source_state_bonus_render_mode
    )
    policy_observation_backend = validate_policy_observation_backend(policy_observation_backend)
    learner_seat_mode = validate_learner_seat_mode(learner_seat_mode)
    if not profile_env_timing_enabled:
        validate_trusted_source_state_action_cadence(
            env_variant=env_variant,
            decision_ms=decision_ms,
            decision_source_frames=decision_source_frames,
            source_physics_step_ms=source_physics_step_ms,
            source_max_steps_semantics=source_max_steps_semantics,
            context="build_visual_survival_configs",
        )
    exploration_bonus_spec = xb.normalize_exploration_bonus_config(exploration_bonus)
    target_config = lightzero_target_config_for_reward(
        env_variant=env_variant,
        reward_variant=reward_variant,
        source_max_steps=source_max_steps,
        reward_outcome_alpha=reward_outcome_alpha,
        model_support_cap=model_support_cap,
        td_steps=td_steps,
    )
    target_config = target_config_with_exploration_bonus_bound(
        target_config,
        exploration_bonus_spec,
        source_max_steps=source_max_steps,
    )
    action_space_size = int(env_spec.get("action_space_size", 3))
    main_config = copy.deepcopy(module.main_config)
    create_config = easy_config(
        {
            "env": {
                "type": env_spec["env_type"],
                "import_names": env_spec["import_names"],
            },
            "env_manager": {"type": env_manager_type},
            "policy": {"type": "muzero", "import_names": ["lzero.policy.muzero"]},
        }
    )
    patches = [
        set_or_add_path(main_config, ("exp_name",), str(exp_name)),
        set_or_add_path(main_config, ("policy", "cuda"), bool(cuda)),
        set_or_add_path(main_config, ("policy", "multi_gpu"), bool(lightzero_multi_gpu)),
        set_or_add_path(main_config, ("policy", "env_type"), "not_board_games"),
        set_or_add_path(main_config, ("policy", "collector_env_num"), int(collector_env_num)),
        set_or_add_path(main_config, ("policy", "evaluator_env_num"), int(evaluator_env_num)),
        set_or_add_path(main_config, ("policy", "n_episode"), int(n_episode)),
        set_or_add_path(main_config, ("policy", "num_simulations"), int(num_simulations)),
        set_or_add_path(main_config, ("policy", "batch_size"), int(batch_size)),
        set_or_add_path(
            main_config,
            ("policy", "eval_freq"),
            int(lightzero_eval_freq) if int(lightzero_eval_freq) > 0 else int(max_train_iter) + 1,
        ),
        set_or_add_path(main_config, ("policy", "model", "model_type"), "conv"),
        set_or_add_path(main_config, ("policy", "model", "image_channel"), 4),
        set_or_add_path(main_config, ("policy", "model", "frame_stack_num"), 1),
        set_or_add_path(
            main_config,
            ("policy", "model", "self_supervised_learning_loss"),
            True,
        ),
        set_or_add_path(
            main_config,
            ("policy", "model", "observation_shape"),
            list(env_spec["observation_shape"]),
        ),
        set_or_add_path(
            main_config,
            ("policy", "model", "action_space_size"),
            action_space_size,
        ),
        set_save_ckpt_after_iter(main_config, int(save_ckpt_after_iter)),
    ]
    patches.extend(target_config_patches(main_config, target_config))
    env_cfg = easy_config(
        {
            **to_plain(main_config["env"]),
            "type": env_spec["env_type"],
            "import_names": env_spec["import_names"],
            "env_id": env_spec["env_id"],
            "env_variant": env_variant,
            "action_space_size": action_space_size,
            "collector_env_num": int(collector_env_num),
            "evaluator_env_num": int(evaluator_env_num),
            "n_evaluator_episode": int(n_evaluator_episode),
            "seed": int(seed),
            "dynamic_seed": True,
            "reset_seed_strategy": "dynamic_seed_sequence_from_run_seed_and_reset_index/v0",
            "source_max_steps": int(source_max_steps),
            "max_ticks": int(source_max_steps),
            "decision_ms": float(decision_ms),
            "decision_source_frames": int(decision_source_frames),
            "source_physics_step_ms": float(source_physics_step_ms),
            "source_max_steps_semantics": str(source_max_steps_semantics),
            "frame_stack_num": 1,
            "observation_shape": list(env_spec["observation_shape"]),
            "gray_scale": True,
            "image_channel": 4,
            "continuous": False,
            "manually_discretization": False,
            "telemetry_path": str(telemetry_path),
            "telemetry_stride": int(env_telemetry_stride),
            "profile_env_timing_enabled": bool(profile_env_timing_enabled),
            "reward_variant": reward_variant,
            "reward_outcome_alpha": float(reward_outcome_alpha),
            "reward_schema_id": reward_policy["reward_schema_id"],
            "reward_policy": reward_policy,
            "lightzero_target_config": target_config,
            "source_state_trail_render_mode": source_state_trail_render_mode,
            "source_state_bonus_render_mode": source_state_bonus_render_mode,
            "policy_observation_backend": policy_observation_backend,
            "policy_trail_render_mode": source_state_trail_render_mode,
            "policy_bonus_render_mode": source_state_bonus_render_mode,
            "policy_observation_contract_id": POLICY_OBSERVATION_CONTRACT_ID,
            "observation_contract": policy_observation_surface(
                trail_render_mode=source_state_trail_render_mode,
                bonus_render_mode=source_state_bonus_render_mode,
                backend=policy_observation_backend,
            ),
            "learner_seat_mode": learner_seat_mode,
            "default_trail_render_mode": DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE,
            "supported_trail_render_modes": list(SOURCE_STATE_TRAIL_RENDER_MODE_CHOICES),
            "default_bonus_render_mode": DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE,
            "supported_bonus_render_modes": list(SOURCE_STATE_BONUS_RENDER_MODE_CHOICES),
            "default_policy_observation_backend": DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE,
            "supported_policy_observation_backends": list(POLICY_OBSERVATION_BACKEND_CHOICES),
            "observation_schema_id": env_spec["observation_schema_id"],
            "debug_fidelity_only": env_spec["debug_fidelity_only"],
            "source_fidelity_claim": env_spec["source_fidelity_claim"],
            "single_product_runtime_path": env_spec["single_product_runtime_path"],
            "legacy_debug_variant": env_spec["legacy_debug_variant"],
            "underlying_env_class": env_spec["underlying_env_class"],
            "runtime_env_impl_id": env_spec["runtime_env_impl_id"],
            "runtime_topology": env_spec["runtime_topology"],
            "two_seat_self_play": env_spec["two_seat_self_play"],
            "current_policy_two_seat_action_collection": bool(
                env_spec.get("current_policy_two_seat_action_collection", False)
            ),
            "two_seat_self_play_status": env_spec["two_seat_self_play_status"],
            "fixed_opponent_is_two_seat_self_play": env_spec[
                "fixed_opponent_is_two_seat_self_play"
            ],
            "browser_pixel_fidelity": env_spec["browser_pixel_fidelity"],
            "uses_ale": env_spec["uses_ale"],
            "visual_surface": env_spec["visual_surface"],
            "visual_truth_level": env_spec["visual_truth_level"],
            "visual_source_state_backed": env_spec["visual_source_state_backed"],
            "ego_action_straight_override_probability": float(
                ego_action_straight_override_probability
            ),
            "policy_action_repeat_min": int(policy_action_repeat_min),
            "policy_action_repeat_max": int(policy_action_repeat_max),
            "policy_action_repeat_extra_probability": float(policy_action_repeat_extra_probability),
            "policy_action_repeat_semantics": (
                "repeat_selected_policy_action_inside_one_lightzero_env_step"
            ),
            "control_noise_profile_id": str(control_noise_profile_id),
            "disable_death_for_profile": bool(disable_death_for_profile),
            "opponent_death_mode": opponent_death_mode,
            "opponent_death_mode_diagnostic": (opponent_death_mode == OPPONENT_DEATH_MODE_IMMORTAL),
            "opponent_death_mode_claim": (
                "diagnostic_opponent_immortal_not_source_faithful"
                if opponent_death_mode == OPPONENT_DEATH_MODE_IMMORTAL
                else "none"
            ),
            "opponent_runtime_mode": opponent_runtime_mode,
            "opponent_runtime_mode_claim": (
                "blank_canvas_noop_player_1_inert_hidden_no_trail_no_collision_no_bonus"
                if opponent_runtime_mode == OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP
                else "normal_opponent_runtime"
            ),
            "opponent_visibility_mode": (
                "hidden_from_model_gif_raw_render_views_public_present_alive"
                if opponent_runtime_mode == OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP
                else "visible_if_present_alive"
            ),
            "opponent_collision_effect": (
                "disabled_no_player_1_movement_trail_collision_bonus_side_effects"
                if opponent_runtime_mode == OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP
                else "normal"
            ),
            "opponent_trail_mode": (
                "none_blank_canvas_scrubbed"
                if opponent_runtime_mode == OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP
                else "normal"
            ),
            "natural_bonus_spawn": bool(natural_bonus_spawn),
            "death_mode": "profile_no_death" if disable_death_for_profile else "normal",
            "turn_commit_adapter": bool(env_spec["turn_commit_adapter"]),
            "opponent_policy_kind": opponent_policy_kind,
            "opponent_training_relation": opponent_training_relation_for_surface(
                env_variant=env_variant,
                opponent_policy_kind=opponent_policy_kind,
                env_spec=env_spec,
                opponent_mixture=opponent_mixture,
            ),
            "current_policy_self_play": env_spec["current_policy_self_play"],
            "current_policy_self_play_blocker": env_spec["current_policy_self_play_blocker"],
            "current_policy_self_play_caveat": env_spec["current_policy_self_play_caveat"],
            "trusted_current_policy_self_play": env_spec["trusted_current_policy_self_play"],
            "simultaneous_game_theory_claim": env_spec["simultaneous_game_theory_claim"],
        }
    )
    if opponent_mixture is not None:
        env_cfg["opponent_mixture"] = opponent_mixture
    if opponent_assignment_context is not None:
        env_cfg["opponent_assignment_context"] = opponent_assignment_context
    if opponent_policy_kind == OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT:
        if opponent_checkpoint is None:
            raise ValueError("opponent_checkpoint is required for frozen opponent env config")
        env_cfg.update(
            {
                "opponent_checkpoint_path": opponent_checkpoint["resolved_checkpoint_path"],
                "opponent_checkpoint_ref": opponent_checkpoint["checkpoint_ref"],
                "opponent_snapshot_ref": (
                    opponent_snapshot_ref or "curvytron_visual_survival_frozen_opponent"
                ),
                "opponent_checkpoint_state_key": opponent_checkpoint_state_key,
                "opponent_policy_seed": int(seed),
                "opponent_num_simulations": int(num_simulations),
                "opponent_batch_size": int(batch_size),
                "opponent_use_cuda": bool(opponent_use_cuda),
            }
        )
    patches.append(set_or_add_path(main_config, ("env",), env_cfg))
    patches.extend(
        xb.apply_lightzero_exploration_bonus_config(
            main_config,
            create_config,
            exploration_bonus_spec,
            seed=int(seed),
        )
    )
    surface = extract_visual_survival_surface(
        main_config,
        create_config,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
    )
    return {
        "template_module": template_module,
        "main_config": main_config,
        "create_config": create_config,
        "surface": surface,
        "patches": patches,
    }


def build_visual_survival_configs(
    *,
    seed: int,
    exp_name: Any,
    telemetry_path: Any,
    cuda: bool,
    max_env_step: int,
    source_max_steps: int,
    decision_ms: float,
    collector_env_num: int,
    evaluator_env_num: int,
    n_evaluator_episode: int,
    n_episode: int,
    num_simulations: int,
    batch_size: int,
    lightzero_eval_freq: int,
    lightzero_multi_gpu: bool,
    max_train_iter: int,
    save_ckpt_after_iter: int,
    env_variant: str,
    reward_variant: str,
    reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
    ego_action_straight_override_probability: float,
    control_noise_profile_id: str,
    disable_death_for_profile: bool,
    env_telemetry_stride: int,
    env_manager_type: str,
    opponent_policy_kind: str,
    opponent_use_cuda: bool,
    opponent_checkpoint: dict[str, Any] | None,
    opponent_snapshot_ref: str | None,
    opponent_checkpoint_state_key: str | None,
    decision_source_frames: int = DEFAULT_DECISION_SOURCE_FRAMES,
    source_physics_step_ms: float = DEFAULT_SOURCE_PHYSICS_STEP_MS,
    source_max_steps_semantics: str = "source_physics_steps",
    opponent_mixture: dict[str, Any] | None = None,
    opponent_assignment_context: dict[str, Any] | None = None,
    source_state_trail_render_mode: str = DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE,
    source_state_bonus_render_mode: str = DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE,
    policy_observation_backend: str = DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE,
    learner_seat_mode: str = DEFAULT_LEARNER_SEAT_MODE,
    policy_action_repeat_min: int = DEFAULT_POLICY_ACTION_REPEAT_MIN,
    policy_action_repeat_max: int = DEFAULT_POLICY_ACTION_REPEAT_MAX,
    policy_action_repeat_extra_probability: float = DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY,
    natural_bonus_spawn: bool = DEFAULT_NATURAL_BONUS_SPAWN,
    profile_env_timing_enabled: bool = False,
    opponent_death_mode: str = DEFAULT_OPPONENT_DEATH_MODE,
    opponent_runtime_mode: str = DEFAULT_OPPONENT_RUNTIME_MODE,
    model_support_cap: int | None = DEFAULT_MODEL_SUPPORT_CAP,
    td_steps: int | None = DEFAULT_TD_STEPS,
    exploration_bonus: Mapping[str, Any] | xb.ExplorationBonusSpec | str | None = None,
) -> dict[str, Any]:
    spec = VisualSurvivalConfigSpec.from_builder_kwargs(locals())
    return build_visual_survival_config(spec).as_dict()


def extract_visual_survival_surface(
    main_config: Any,
    create_config: Any,
    *,
    max_env_step: int,
    max_train_iter: int,
) -> dict[str, Any]:
    policy = main_config["policy"]
    model = policy["model"]
    env = main_config["env"]
    exploration_bonus = env.get("exploration_bonus")
    exploration_bonus_spec = xb.normalize_exploration_bonus_config(exploration_bonus)
    reward_model = main_config.get("reward_model") or {}
    return {
        "env_type": create_config["env"]["type"],
        "env_import_names": to_plain(create_config["env"].get("import_names")),
        "env_manager_type": create_config["env_manager"]["type"],
        "env_id": env["env_id"],
        "model_type": model["model_type"],
        "observation_shape": to_plain(model["observation_shape"]),
        "env_observation_shape": to_plain(env.get("observation_shape")),
        "action_space_size": model["action_space_size"],
        "model_image_channel": model.get("image_channel"),
        "model_frame_stack_num": model.get("frame_stack_num"),
        "model_self_supervised_learning_loss": model.get("self_supervised_learning_loss"),
        "trainer_entrypoint": xb.lightzero_trainer_entrypoint_ref(exploration_bonus_spec),
        "exploration_bonus": exploration_bonus_spec.as_dict(),
        "policy_use_rnd_model": bool(policy.get("use_rnd_model", False)),
        "reward_model_type": reward_model.get("type"),
        "collector_env_num": env["collector_env_num"],
        "policy_collector_env_num": policy.get("collector_env_num"),
        "evaluator_env_num": env["evaluator_env_num"],
        "policy_evaluator_env_num": policy.get("evaluator_env_num"),
        "n_evaluator_episode": env.get("n_evaluator_episode"),
        "n_episode": policy.get("n_episode"),
        "num_simulations": policy.get("num_simulations"),
        "batch_size": policy.get("batch_size"),
        "cuda": policy.get("cuda"),
        "discount_factor": policy.get("discount_factor"),
        "td_steps": policy.get("td_steps"),
        "model_support_scale": model.get("support_scale"),
        "model_reward_support_size": model.get("reward_support_size"),
        "model_value_support_size": model.get("value_support_size"),
        "model_reward_support_range": to_plain(model.get("reward_support_range")),
        "model_value_support_range": to_plain(model.get("value_support_range")),
        "load_ckpt_before_run": get_path(
            policy,
            ("learn", "learner", "hook", "load_ckpt_before_run"),
        ),
        "frame_stack_num": env.get("frame_stack_num"),
        "source_max_steps": env.get("source_max_steps"),
        "decision_ms": env.get("decision_ms"),
        "decision_source_frames": env.get("decision_source_frames"),
        "source_physics_step_ms": env.get("source_physics_step_ms"),
        "source_max_steps_semantics": env.get("source_max_steps_semantics"),
        "dynamic_seed": env.get("dynamic_seed"),
        "reset_seed_strategy": env.get("reset_seed_strategy"),
        "telemetry_path": env.get("telemetry_path"),
        "telemetry_stride": env.get("telemetry_stride"),
        "profile_env_timing_enabled": env.get("profile_env_timing_enabled"),
        "reward_variant": env.get("reward_variant"),
        "reward_schema_id": env.get("reward_schema_id"),
        "reward_policy": to_plain(env.get("reward_policy")),
        "lightzero_target_config": to_plain(env.get("lightzero_target_config")),
        "source_state_trail_render_mode": env.get("source_state_trail_render_mode"),
        "source_state_bonus_render_mode": env.get("source_state_bonus_render_mode"),
        "policy_observation_backend": env.get("policy_observation_backend"),
        "policy_trail_render_mode": env.get("policy_trail_render_mode"),
        "policy_bonus_render_mode": env.get("policy_bonus_render_mode"),
        "policy_observation_contract_id": env.get("policy_observation_contract_id"),
        "observation_contract": to_plain(env.get("observation_contract")),
        "learner_seat_mode": env.get("learner_seat_mode"),
        "default_trail_render_mode": env.get("default_trail_render_mode"),
        "supported_trail_render_modes": to_plain(env.get("supported_trail_render_modes")),
        "default_bonus_render_mode": env.get("default_bonus_render_mode"),
        "supported_bonus_render_modes": to_plain(env.get("supported_bonus_render_modes")),
        "default_policy_observation_backend": env.get("default_policy_observation_backend"),
        "supported_policy_observation_backends": to_plain(
            env.get("supported_policy_observation_backends")
        ),
        "observation_schema_id": env.get("observation_schema_id"),
        "debug_fidelity_only": env.get("debug_fidelity_only"),
        "source_fidelity_claim": env.get("source_fidelity_claim"),
        "single_product_runtime_path": env.get("single_product_runtime_path"),
        "legacy_debug_variant": env.get("legacy_debug_variant"),
        "underlying_env_class": env.get("underlying_env_class"),
        "runtime_env_impl_id": env.get("runtime_env_impl_id"),
        "runtime_topology": env.get("runtime_topology"),
        "two_seat_self_play": env.get("two_seat_self_play"),
        "current_policy_two_seat_action_collection": env.get(
            "current_policy_two_seat_action_collection"
        ),
        "two_seat_self_play_status": env.get("two_seat_self_play_status"),
        "fixed_opponent_is_two_seat_self_play": env.get("fixed_opponent_is_two_seat_self_play"),
        "browser_pixel_fidelity": env.get("browser_pixel_fidelity"),
        "uses_ale": env.get("uses_ale"),
        "visual_surface": env.get("visual_surface"),
        "visual_truth_level": env.get("visual_truth_level"),
        "visual_source_state_backed": env.get("visual_source_state_backed"),
        "ego_action_straight_override_probability": env.get(
            "ego_action_straight_override_probability"
        ),
        "policy_action_repeat_min": env.get("policy_action_repeat_min"),
        "policy_action_repeat_max": env.get("policy_action_repeat_max"),
        "policy_action_repeat_extra_probability": env.get("policy_action_repeat_extra_probability"),
        "policy_action_repeat_semantics": env.get("policy_action_repeat_semantics"),
        "control_noise_profile_id": env.get("control_noise_profile_id"),
        "disable_death_for_profile": env.get("disable_death_for_profile"),
        "opponent_death_mode": env.get("opponent_death_mode"),
        "opponent_death_mode_diagnostic": env.get("opponent_death_mode_diagnostic"),
        "opponent_death_mode_claim": env.get("opponent_death_mode_claim"),
        "opponent_runtime_mode": env.get("opponent_runtime_mode"),
        "opponent_runtime_mode_claim": env.get("opponent_runtime_mode_claim"),
        "opponent_visibility_mode": env.get("opponent_visibility_mode"),
        "opponent_collision_effect": env.get("opponent_collision_effect"),
        "opponent_trail_mode": env.get("opponent_trail_mode"),
        "natural_bonus_spawn": env.get("natural_bonus_spawn"),
        "death_mode": env.get("death_mode"),
        "env_variant": env.get("env_variant"),
        "turn_commit_adapter": env.get("turn_commit_adapter"),
        "opponent_policy_kind": env.get("opponent_policy_kind"),
        "opponent_training_relation": env.get("opponent_training_relation"),
        "current_policy_self_play": env.get("current_policy_self_play"),
        "current_policy_self_play_blocker": env.get("current_policy_self_play_blocker"),
        "current_policy_self_play_caveat": env.get("current_policy_self_play_caveat"),
        "trusted_current_policy_self_play": env.get("trusted_current_policy_self_play"),
        "simultaneous_game_theory_claim": env.get("simultaneous_game_theory_claim"),
        "opponent_checkpoint_ref": env.get("opponent_checkpoint_ref"),
        "opponent_snapshot_ref": env.get("opponent_snapshot_ref"),
        "opponent_checkpoint_state_key": env.get("opponent_checkpoint_state_key"),
        "opponent_use_cuda": env.get("opponent_use_cuda"),
        "opponent_mixture": to_plain(env.get("opponent_mixture")),
        "opponent_assignment_context": to_plain(env.get("opponent_assignment_context")),
        "save_ckpt_after_iter": get_path(
            policy,
            ("learn", "learner", "hook", "save_ckpt_after_iter"),
        ),
        "max_env_step": int(max_env_step),
        "max_train_iter": int(max_train_iter),
    }
