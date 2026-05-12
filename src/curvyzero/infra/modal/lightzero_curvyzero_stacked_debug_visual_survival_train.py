"""Modal trainer for CurvyTron source-state visual survival LightZero MuZero.

This module still carries the original stacked-debug filename, but the live
source-state variants route through source-state visual wrappers over
``VectorMultiplayerEnv``. The default ``source_state_fixed_opponent`` lane is
an explicit fixed-opponent control. ``source_state_turn_commit`` is only a stock
LightZero plumbing smoke/control path; train mode is blocked for it because its
pending/commit scalar rewards have bad credit semantics.

Dry config check:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train --mode dry

Bounded train launch:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train --mode train

Frozen-checkpoint opponent wrapper smoke:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train --mode opponent-smoke --opponent-checkpoint-ref training/lightzero-curvytron-visual-survival/<run>/checkpoints/lightzero/iteration_293.pth.tar
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import importlib
import inspect
import json
import math
import os
import pickle
import random
import re
import shutil
import subprocess
import threading
import time
import traceback
from collections import Counter
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

from curvyzero.env.trainer_contract import (
    REWARD_SCHEMA_ID as SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_ID,
)
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY,
)
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SHAPE
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_CANVAS_GRAY64_SOURCE_STATE_BACKED,
)
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SURFACE
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_TRUTH_LEVEL
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_USES_ALE
from curvyzero.env.vector_visual_observation import SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY
from curvyzero.env.vector_visual_observation import SOURCE_STATE_GRAY64_SOURCE_STATE_BACKED
from curvyzero.env.vector_visual_observation import SOURCE_STATE_GRAY64_SURFACE
from curvyzero.env.vector_visual_observation import SOURCE_STATE_GRAY64_TRUTH_LEVEL
from curvyzero.env.vector_visual_observation import SOURCE_STATE_GRAY64_USES_ALE
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_SHEET_RELATIVE_PATH,
)
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
)
from curvyzero.env.vector_visual_observation import SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL,
)
from curvyzero.env.vector_visual_observation import (
    TRAIL_RENDER_MODE_BODY_CIRCLES_FAST as _TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
)
from curvyzero.env.vector_visual_observation import TRAIL_RENDER_MODE_DEFAULT
from curvyzero.infra.modal import run_management as runs
from curvyzero.training.curvytron_current_policy_selfplay_smoke import (
    STACK_RENDER_MODE_ORDER,
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
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    CURRENT_POLICY_SELF_PLAY_CLAIM,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    OPPONENT_TRAINING_RELATION_FIXED_STRAIGHT,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    OPPONENT_TRAINING_RELATION_FROZEN_LIGHTZERO_CHECKPOINT,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    STACKED_DEBUG_VISUAL_SURVIVAL_SCHEMA_ID,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    TURN_COMMIT_CURRENT_POLICY_SELF_PLAY_CLAIM,
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
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_ID,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    DENSE_SURVIVAL_PLUS_OUTCOME_REWARD_SCHEMA_ID,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    DEFAULT_DECISION_MS as SOURCE_STATE_DEFAULT_DECISION_MS,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    JOINT_ACTION_COUNT as SOURCE_STATE_JOINT_ACTION_COUNT,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_ID,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_ENV_TYPE,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    LIGHTZERO_SOURCE_STATE_VISUAL_JOINT_ACTION_IMPORT_NAMES,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    REWARD_VARIANT_SPARSE_OUTCOME,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    SOURCE_STATE_FIXED_OPPONENT_REWARD_VARIANTS,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    SOURCE_STATE_JOINT_ACTION_ADAPTER_IMPL_ID,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    SOURCE_STATE_JOINT_ACTION_RUNTIME_TOPOLOGY,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    SOURCE_STATE_JOINT_ACTION_TRAINING_STATUS,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    STACKED_SOURCE_STATE_GRAY64_SHAPE,
)
from curvyzero.training.curvyzero_source_state_visual_turn_commit_lightzero_env import (
    LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_ID,
)
from curvyzero.training.curvyzero_source_state_visual_turn_commit_lightzero_env import (
    LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_ENV_TYPE,
)
from curvyzero.training.curvyzero_source_state_visual_turn_commit_lightzero_env import (
    LIGHTZERO_SOURCE_STATE_VISUAL_TURN_COMMIT_IMPORT_NAMES,
)
from curvyzero.training.curvyzero_source_state_visual_turn_commit_lightzero_env import (
    SOURCE_STATE_TURN_COMMIT_ADAPTER_IMPL_ID,
)
from curvyzero.training.curvyzero_source_state_visual_turn_commit_lightzero_env import (
    SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE as TURN_COMMIT_SOURCE_STATE_CANVAS_LIKE_RAW_SHAPE,
)
from curvyzero.training.curvyzero_source_state_visual_turn_commit_lightzero_env import (
    STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID as TURN_COMMIT_STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
)
from curvyzero.training.curvyzero_source_state_visual_turn_commit_lightzero_env import (
    STACKED_SOURCE_STATE_GRAY64_SHAPE as TURN_COMMIT_STACKED_SOURCE_STATE_GRAY64_SHAPE,
)
from curvyzero.training.curvyzero_source_state_visual_turn_commit_lightzero_env import (
    TURN_COMMIT_TRAINING_STATUS,
)
from curvyzero.training.curvyzero_survival_time_lightzero_smoke import (
    SURVIVAL_TIME_REWARD_SCHEMA_ID,
)
from curvyzero.training.curvytron_visual_observation import (
    DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    DEFAULT_ACTION_NOOP_PROBABILITY as TWO_SEAT_DEFAULT_ACTION_NOOP_PROBABILITY,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    DEFAULT_ACTION_NOOP_WARMUP_ITERATIONS as TWO_SEAT_DEFAULT_ACTION_NOOP_WARMUP_ITERATIONS,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    DEFAULT_ALIVE_REWARD as TWO_SEAT_DEFAULT_ALIVE_REWARD,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    DEFAULT_BONUS_PICKUP_REWARD_PER_CATCH as TWO_SEAT_DEFAULT_BONUS_PICKUP_REWARD_PER_CATCH,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    DEFAULT_DEAD_REWARD as TWO_SEAT_DEFAULT_DEAD_REWARD,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    DEFAULT_DEATH_MODE as TWO_SEAT_DEFAULT_DEATH_MODE,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    DEFAULT_NATURAL_BONUS_SPAWN as TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    DEFAULT_OBSERVATION_NOISE_STD as TWO_SEAT_DEFAULT_OBSERVATION_NOISE_STD,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY as TWO_SEAT_DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    DEFAULT_POLICY_ACTION_REPEAT_MAX as TWO_SEAT_DEFAULT_POLICY_ACTION_REPEAT_MAX,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    DEFAULT_POLICY_ACTION_REPEAT_MIN as TWO_SEAT_DEFAULT_POLICY_ACTION_REPEAT_MIN,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    DEFAULT_POLICY_ACTION_REPEAT_WARMUP_ITERATIONS as TWO_SEAT_DEFAULT_POLICY_ACTION_REPEAT_WARMUP_ITERATIONS,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    DEFAULT_RETURN_TARGET_DISCOUNT as TWO_SEAT_DEFAULT_RETURN_TARGET_DISCOUNT,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    DEFAULT_TERMINAL_OUTCOME_REWARD_PER_STEP as TWO_SEAT_DEFAULT_TERMINAL_OUTCOME_REWARD_PER_STEP,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    compact_curvytron_two_seat_lightzero_train_smoke_summary,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    run_curvytron_two_seat_lightzero_train_smoke,
)


APP_NAME = "curvyzero-lightzero-curvytron-visual-survival-train"
TASK_ID = "lightzero-curvytron-visual-survival"
VOLUME_NAME = "curvyzero-runs"
TRAIL_RENDER_MODE_BODY_CIRCLES_FAST = _TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
LIGHTZERO_VERSION = "0.2.0"
REMOTE_ROOT = Path("/repo")
RUNS_MOUNT = Path("/runs")

DEFAULT_MODE = "two-seat-selfplay"
DEFAULT_COMPUTE = "gpu-l4-t4"
DEFAULT_SEED = 0
DEFAULT_MAX_ENV_STEP = 8192
DEFAULT_MAX_TRAIN_ITER = 64
DEFAULT_SOURCE_MAX_STEPS = 256
DEFAULT_COLLECTOR_ENV_NUM = 1
DEFAULT_EVALUATOR_ENV_NUM = 1
DEFAULT_N_EVALUATOR_EPISODE = 1
DEFAULT_N_EPISODE = 1
DEFAULT_NUM_SIMULATIONS = 8
DEFAULT_BATCH_SIZE = 16
DEFAULT_LIGHTZERO_EVAL_FREQ = 0
DEFAULT_SKIP_LIGHTZERO_EVAL_IN_PROFILE = True
DEFAULT_LIGHTZERO_MULTI_GPU = False
DEFAULT_PROFILE_CUDA_SYNC_ENABLED = False
DEFAULT_PROFILE_ALLOW_AUTO_RESUME = False
# Checkpoint cadence also controls automatic checkpoint eval/inspection/GIF work.
# Keep this frequent enough to observe progress but not every loop.
DEFAULT_SAVE_CKPT_AFTER_ITER = 100
DEFAULT_STOP_AFTER_LEARNER_TRAIN_CALLS = 0
DEFAULT_DECISION_MS = SOURCE_STATE_DEFAULT_DECISION_MS
DEFAULT_ENV_TELEMETRY_STRIDE = 1
DEFAULT_ENV_MANAGER_TYPE = "subprocess"
ENV_MANAGER_TYPE_CHOICES = ("base", "subprocess")
DEFAULT_RUN_ID = "curvytron-two-seat-selfplay-canonical-s0"
DEFAULT_ATTEMPT_ID = "two-seat-current-policy-selfplay"
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
REWARD_VARIANT_AUTO = "auto"
DEFAULT_REWARD_VARIANT = REWARD_VARIANT_AUTO
REWARD_VARIANT_CHOICES = (
    REWARD_VARIANT_AUTO,
    REWARD_VARIANT_SPARSE_OUTCOME,
    REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
    REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC,
)
DEFAULT_EGO_ACTION_STRAIGHT_OVERRIDE_PROBABILITY = 0.0
DEFAULT_CONTROL_NOISE_PROFILE_ID = "none"
DEFAULT_DISABLE_DEATH_FOR_PROFILE = False
OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION = (
    "none_centralized_joint_action"
)
DEFAULT_OPPONENT_POLICY_KIND = OPPONENT_POLICY_KIND_FIXED_STRAIGHT
OPPONENT_POLICY_KIND_CHOICES = (
    OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
    OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
    OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION,
)
DEFAULT_BACKGROUND_EVAL_ENABLED = True
DEFAULT_BACKGROUND_EVAL_COMPUTE = "cpu"
DEFAULT_BACKGROUND_EVAL_ID_PREFIX = "live_checkpoint"
DEFAULT_BACKGROUND_EVAL_SEED_COUNT = 4
DEFAULT_BACKGROUND_EVAL_SEED_RNG_SEED = 0
DEFAULT_BACKGROUND_EVAL_MAX_STEPS = 65_536
DEFAULT_BACKGROUND_EVAL_STEP_DETAIL_LIMIT = 4
DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS = DEFAULT_NUM_SIMULATIONS
DEFAULT_BACKGROUND_EVAL_BATCH_SIZE = 64
DEFAULT_BACKGROUND_GIF_ENABLED = True
DEFAULT_BACKGROUND_GIF_SEED_OFFSET = 10_000
DEFAULT_BACKGROUND_GIF_CHECKPOINT_SEED_MIXING_ENABLED = True
# 0 means no GIF-specific physical-step cap: capture stops when the env ends.
DEFAULT_BACKGROUND_GIF_MAX_STEPS = 0
DEFAULT_BACKGROUND_GIF_FRAME_STRIDE = 1
DEFAULT_BACKGROUND_GIF_FPS = 8.0
DEFAULT_BACKGROUND_GIF_SCALE = 4
DEFAULT_BACKGROUND_GIF_FRAME_SIZE = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
DEFAULT_BACKGROUND_CHECKPOINT_WAIT_TIMEOUT_SEC = 30 * 60
DEFAULT_BACKGROUND_CHECKPOINT_WAIT_POLL_SEC = 10.0
BACKGROUND_EVAL_LAUNCH_HOOK = "hook"
BACKGROUND_EVAL_LAUNCH_POLLER = "poller"
BACKGROUND_EVAL_LAUNCH_CHOICES = (
    BACKGROUND_EVAL_LAUNCH_HOOK,
    BACKGROUND_EVAL_LAUNCH_POLLER,
)
DEFAULT_BACKGROUND_EVAL_LAUNCH_KIND = BACKGROUND_EVAL_LAUNCH_POLLER
DEFAULT_BACKGROUND_EVAL_POLL_INTERVAL_SEC = 10.0
DEFAULT_BACKGROUND_EVAL_POLL_STABLE_POLLS = 1
DEFAULT_BACKGROUND_EVAL_POLLER_MAX_RUNTIME_SEC = 8 * 60 * 60
DEFAULT_BACKGROUND_EVAL_POLLER_IDLE_AFTER_DONE_SEC = 60.0
DEFAULT_TWO_SEAT_COLLECT_STEPS_PER_ITERATION = 64
DEFAULT_TWO_SEAT_UPDATES_PER_ITERATION = 4
DEFAULT_TWO_SEAT_MAX_TICKS = 65_536
DEFAULT_TWO_SEAT_MAX_REPLAY_ROWS = 65_536
DEFAULT_TWO_SEAT_SAVE_INITIAL_CHECKPOINT = True
DEFAULT_TWO_SEAT_PROGRESS_EVERY_ITERATIONS = 10
DEFAULT_TWO_SEAT_PROGRESS_COMMIT_EVERY_ITERATIONS = 10
DEFAULT_TWO_SEAT_LEARNER_SAMPLE_SIZE = 128


def _normalize_background_gif_max_steps(value: Any) -> int | None:
    if value is None:
        return None
    max_steps = int(value)
    if max_steps <= 0:
        return None
    return max_steps


def _background_gif_max_steps_arg(value: Any) -> int:
    max_steps = _normalize_background_gif_max_steps(value)
    return 0 if max_steps is None else int(max_steps)


def _background_gif_step_limit_kind(value: Any) -> str:
    return (
        "until_environment_done"
        if _normalize_background_gif_max_steps(value) is None
        else "physical_step_cap"
    )


def _normalize_opponent_policy_kind_for_env(
    *,
    env_variant: str,
    opponent_policy_kind: str,
    opponent_checkpoint_ref: str | None,
) -> str:
    if (
        env_variant == ENV_VARIANT_SOURCE_STATE_JOINT_ACTION
        and opponent_policy_kind == DEFAULT_OPPONENT_POLICY_KIND
        and opponent_checkpoint_ref is None
    ):
        return OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION
    return opponent_policy_kind


def _normalize_reward_variant_for_env(*, env_variant: str, reward_variant: str) -> str:
    if reward_variant == REWARD_VARIANT_AUTO:
        if env_variant == ENV_VARIANT_SOURCE_STATE_JOINT_ACTION:
            return REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC
        if env_variant == ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT:
            return REWARD_VARIANT_SPARSE_OUTCOME
        return REWARD_VARIANT_AUTO
    return reward_variant


def _reward_policy_for_variant(
    *, env_variant: str, reward_variant: str
) -> dict[str, Any]:
    reward_variant = _normalize_reward_variant_for_env(
        env_variant=env_variant,
        reward_variant=reward_variant,
    )
    if env_variant == ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT:
        if reward_variant == REWARD_VARIANT_SPARSE_OUTCOME:
            return {
                "reward_variant": reward_variant,
                "reward_schema_id": SPARSE_ROUND_OUTCOME_REWARD_SCHEMA_ID,
                "survival_length_is_eval_metric": True,
                "dense_survival_reward": False,
                "sparse_outcome_reward": True,
                "survival_only": False,
                "diagnostic_all_players_alive": False,
                "centralized_joint_action_control": False,
                "per_player_reward": True,
                "zero_sum_reward": True,
                "nonterminal_reward": 0.0,
                "winner_bonus": 1.0,
                "loser_penalty": -1.0,
                "draw_bonus": 0.0,
                "truncation_bonus": 0.0,
            }
        if reward_variant == REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME:
            return {
                "reward_variant": reward_variant,
                "reward_schema_id": DENSE_SURVIVAL_PLUS_OUTCOME_REWARD_SCHEMA_ID,
                "survival_length_is_eval_metric": True,
                "dense_survival_reward": True,
                "dense_alive_helper": 1.0,
                "sparse_outcome_reward": True,
                "survival_only": False,
                "diagnostic_all_players_alive": False,
                "centralized_joint_action_control": False,
                "per_player_reward": True,
                "zero_sum_reward": False,
                "post_transition_alive_reward": 1.0,
                "post_transition_dead_reward": 0.0,
                "winner_bonus": 1.0,
                "loser_penalty": -1.0,
                "draw_bonus": 0.0,
                "truncation_bonus": 0.0,
            }
        raise ValueError(
            "source_state_fixed_opponent reward_variant must be one of "
            f"{SOURCE_STATE_FIXED_OPPONENT_REWARD_VARIANTS!r}; got {reward_variant!r}"
        )
    if env_variant == ENV_VARIANT_SOURCE_STATE_JOINT_ACTION:
        if reward_variant != REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC:
            raise ValueError(
                "source_state_joint_action only supports reward_variant="
                f"{REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC!r}; got {reward_variant!r}"
            )
        return _all_players_alive_diagnostic_reward_policy()
    env_spec = _env_variant_spec(env_variant)
    if reward_variant != REWARD_VARIANT_AUTO:
        raise ValueError(
            f"{env_variant} does not support explicit reward_variant {reward_variant!r}"
        )
    return dict(env_spec["reward_policy"])


def _reward_schema_id_for_variant(*, env_variant: str, reward_variant: str) -> str:
    reward_policy = _reward_policy_for_variant(
        env_variant=env_variant,
        reward_variant=reward_variant,
    )
    return str(reward_policy["reward_schema_id"])


def _lightzero_target_config_for_reward(
    *,
    env_variant: str,
    reward_variant: str,
    source_max_steps: int,
) -> dict[str, Any]:
    reward_variant = _normalize_reward_variant_for_env(
        env_variant=env_variant,
        reward_variant=reward_variant,
    )
    if env_variant == ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT:
        support_scale = 1
        if reward_variant == REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME:
            support_scale = int(source_max_steps) + 1
        return {
            "discount_factor": 1.0,
            "td_steps": int(source_max_steps),
            "model_support_scale": int(support_scale),
            "model_reward_support_size": int(2 * support_scale + 1),
            "model_value_support_size": int(2 * support_scale + 1),
        }
    if env_variant == ENV_VARIANT_SOURCE_STATE_JOINT_ACTION:
        support_scale = max(1, int(source_max_steps))
        return {
            "discount_factor": 1.0,
            "td_steps": int(source_max_steps),
            "model_support_scale": int(support_scale),
            "model_reward_support_size": int(2 * support_scale + 1),
            "model_value_support_size": int(2 * support_scale + 1),
        }
    return {}


CHEAP_GPU_RESOURCE = ["L4", "T4"]
H100_GPU_RESOURCE = "H100"
H100X2_GPU_RESOURCE = "H100:2"
COMPUTE_CPU = "cpu"
COMPUTE_CPU64 = "cpu64"
COMPUTE_GPU_L4_T4 = "gpu-l4-t4"
COMPUTE_GPU_L4_T4_CPU40 = "gpu-l4-t4-cpu40"
COMPUTE_GPU_H100_CPU40 = "gpu-h100-cpu40"
COMPUTE_GPU_H100X2_CPU40 = "gpu-h100x2-cpu40"
COMPUTE_CHOICES = (
    COMPUTE_CPU,
    COMPUTE_CPU64,
    COMPUTE_GPU_L4_T4,
    COMPUTE_GPU_L4_T4_CPU40,
    COMPUTE_GPU_H100_CPU40,
    COMPUTE_GPU_H100X2_CPU40,
)
BACKGROUND_EVAL_COMPUTE_CHOICES = ("cpu",)
MODE_CHOICES = ("dry", "train", "profile")
TWO_SEAT_SELFPLAY_MODE = "two-seat-selfplay"
MAIN_MODE_CHOICES = (*MODE_CHOICES, TWO_SEAT_SELFPLAY_MODE)
OPPONENT_SMOKE_MODE = "opponent-smoke"
OUTPUT_DETAIL_COMPACT = "compact"
OUTPUT_DETAIL_FULL = "full"
OUTPUT_DETAIL_CHOICES = (OUTPUT_DETAIL_COMPACT, OUTPUT_DETAIL_FULL)
PROFILE_COUNT_KEYS = (
    "collector_collect_calls",
    "env_steps_collected",
    "game_segments_collected",
    "game_segments_pushed",
    "replay_push_calls",
    "replay_remove_oldest_calls",
    "replay_sample_calls",
    "replay_update_priority_calls",
    "mcts_search_calls",
    "learner_train_calls",
    "learner_train_iter_delta",
    "learner_save_checkpoint_calls",
    "evaluator_eval_calls",
    "env_reset_calls",
    "env_step_calls",
    "env_vector_step_calls",
    "env_runtime_step_many_calls",
    "env_obs_pack_calls",
    "env_stack_update_calls",
    "env_render_gray64_calls",
    "env_normalize_gray64_calls",
    "env_telemetry_write_calls",
    "policy_forward_collect_calls",
    "policy_forward_eval_calls",
    "policy_forward_learn_calls",
    "model_initial_inference_calls",
    "model_initial_inference_batch_sum",
    "model_recurrent_inference_calls",
    "model_recurrent_inference_batch_sum",
    "mcts_search_root_sum",
    "mcts_search_simulation_budget_sum",
    "mcts_search_node_budget_sum",
)


def _compute_uses_cuda(compute: str) -> bool:
    return compute.startswith("gpu-")
LIGHTZERO_RESUME_STATE_DIRNAME = "lightzero_resume_state"
TARGET_AUDIT_MAX_SEGMENTS = 4
TARGET_AUDIT_MAX_STEPS_PER_SEGMENT = 8
TARGET_AUDIT_MAX_REPLAY_SAMPLES = 3
CURVYTRON_BONUS_SPRITE_SHEET_LOCAL_PATH = (
    Path.cwd() / SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_SHEET_RELATIVE_PATH
)
CURVYTRON_BONUS_SPRITE_SHEET_REMOTE_PATH = (
    REMOTE_ROOT / SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_SHEET_RELATIVE_PATH
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        f"LightZero=={LIGHTZERO_VERSION}",
        "numpy>=1.26",
        "cloudpickle>=3",
        "pillow>=10",
    )
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_file(
        CURVYTRON_BONUS_SPRITE_SHEET_LOCAL_PATH,
        remote_path=str(CURVYTRON_BONUS_SPRITE_SHEET_REMOTE_PATH),
        copy=True,
    )
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


class _PhaseTimer:
    def __init__(
        self,
        profiler: "_LightZeroPhaseProfiler",
        name: str,
        *,
        sync_cuda: bool = False,
    ):
        self.profiler = profiler
        self.name = name
        self.sync_cuda = sync_cuda
        self.started = 0.0

    def __enter__(self) -> "_PhaseTimer":
        if self.sync_cuda:
            self.profiler.sync_cuda(f"{self.name}:before")
        self.profiler.context_stack.append(self.name)
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.sync_cuda:
            self.profiler.sync_cuda(f"{self.name}:after")
        self.profiler.add_time(self.name, time.perf_counter() - self.started)
        if self.profiler.context_stack and self.profiler.context_stack[-1] == self.name:
            self.profiler.context_stack.pop()
        elif self.name in self.profiler.context_stack:
            self.profiler.context_stack.remove(self.name)


class _LightZeroPhaseProfiler:
    """Tiny opt-in profiler for LightZero single-process phase timings."""

    def __init__(self, *, enabled: bool, cuda_sync_enabled: bool = False):
        self.enabled = enabled
        self.cuda_sync_enabled = bool(enabled and cuda_sync_enabled)
        self.gpu_sample_interval_sec = 1.0 if enabled else 0.0
        self.started = time.perf_counter()
        self.timers: dict[str, float] = {}
        self.counts: dict[str, int] = {key: 0 for key in PROFILE_COUNT_KEYS}
        self.installed_hooks: list[str] = []
        self.notes: list[str] = []
        self.gpu_samples: list[dict[str, Any]] = []
        self.gpu_sample_errors: list[str] = []
        self.samples: dict[str, list[Any]] = {}
        self.context_stack: list[str] = []

    def timer(self, name: str, *, sync_cuda: bool = False) -> _PhaseTimer:
        return _PhaseTimer(self, name, sync_cuda=sync_cuda)

    def add_time(self, name: str, elapsed_sec: float) -> None:
        self.timers[name] = self.timers.get(name, 0.0) + elapsed_sec

    def add_count(self, name: str, amount: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + int(amount)

    def add_installed_hook(self, hook: str) -> None:
        if hook not in self.installed_hooks:
            self.installed_hooks.append(hook)

    def add_note(self, note: str) -> None:
        if note not in self.notes and len(self.notes) < 30:
            self.notes.append(note)

    def add_sample(self, name: str, value: Any, *, limit: int = 80) -> None:
        samples = self.samples.setdefault(name, [])
        if len(samples) < limit:
            samples.append(value)

    def current_context(self) -> str | None:
        return self.context_stack[-1] if self.context_stack else None

    def sync_cuda(self, label: str) -> None:
        if not self.cuda_sync_enabled:
            return
        started = time.perf_counter()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            self.add_note(f"cuda sync failed at {label}: {type(exc).__name__}: {exc}")
            return
        self.add_time("cuda_sync_sec", time.perf_counter() - started)

    def sample_gpu(self) -> None:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,"
                    "memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            if len(self.gpu_sample_errors) < 10:
                self.gpu_sample_errors.append(f"{type(exc).__name__}: {exc}")
            return
        if proc.returncode != 0:
            if len(self.gpu_sample_errors) < 10:
                self.gpu_sample_errors.append(
                    proc.stderr.strip() or f"nvidia-smi rc={proc.returncode}"
                )
            return
        for line in proc.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 7:
                continue
            self.gpu_samples.append(
                {
                    "timestamp": parts[0],
                    "name": parts[1],
                    "gpu_util_percent": _parse_float_or_none(parts[2]),
                    "memory_util_percent": _parse_float_or_none(parts[3]),
                    "memory_used_mib": _parse_float_or_none(parts[4]),
                    "memory_total_mib": _parse_float_or_none(parts[5]),
                    "power_draw_w": _parse_float_or_none(parts[6]),
                }
            )

    def summary(self) -> dict[str, Any]:
        gpu_util = [
            float(sample["gpu_util_percent"])
            for sample in self.gpu_samples
            if sample.get("gpu_util_percent") is not None
        ]
        mem_used = [
            float(sample["memory_used_mib"])
            for sample in self.gpu_samples
            if sample.get("memory_used_mib") is not None
        ]
        sample_stats: dict[str, dict[str, Any]] = {}
        for key, values in self.samples.items():
            numeric_values = [
                float(value)
                for value in values
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if numeric_values:
                sample_stats[key] = {
                    "count": len(numeric_values),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "mean": sum(numeric_values) / len(numeric_values),
                }
        derived_stats: dict[str, float] = {}
        for key, value in self.counts.items():
            if not key.endswith("_batch_sum"):
                continue
            call_key = f"{key[:-len('_batch_sum')]}_calls"
            call_count = self.counts.get(call_key, 0)
            if call_count:
                derived_stats[f"{key[:-len('_batch_sum')]}_batch_mean"] = (
                    float(value) / float(call_count)
                )
        return {
            "enabled": self.enabled,
            "profile_label": "profile",
            "learning_proof": False,
            "elapsed_sec": round(time.perf_counter() - self.started, 6),
            "timers_sec": {key: round(value, 6) for key, value in sorted(self.timers.items())},
            "counts": dict(sorted(self.counts.items())),
            "gpu_sampling": {
                "interval_sec": self.gpu_sample_interval_sec,
                "sample_count": len(self.gpu_samples),
                "first_sample": self.gpu_samples[0] if self.gpu_samples else None,
                "last_sample": self.gpu_samples[-1] if self.gpu_samples else None,
                "max_gpu_util_percent": max(gpu_util) if gpu_util else None,
                "max_memory_used_mib": max(mem_used) if mem_used else None,
                "errors": self.gpu_sample_errors,
            },
            "samples": self.samples,
            "sample_stats": sample_stats,
            "derived_stats": dict(sorted(derived_stats.items())),
            "installed_hooks": self.installed_hooks,
            "notes": self.notes,
            "caveat": (
                "Optimizer/debug profiler only. The stop is raised after BaseLearner.train "
                "returns, so this may include a real optimizer step, but it is not a "
                "learning proof. Surface fidelity is reported in command/surface."
            ),
        }


class _LightZeroProfileStop(RuntimeError):
    """Intentional profiler-only stop after enough LightZero work was timed."""


def _parse_float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _start_gpu_sampler(
    *,
    profiler: _LightZeroPhaseProfiler,
    interval_sec: float,
) -> tuple[threading.Event, threading.Thread] | None:
    if interval_sec <= 0:
        return None

    stop_event = threading.Event()

    def _sample_loop() -> None:
        profiler.sample_gpu()
        while not stop_event.wait(interval_sec):
            profiler.sample_gpu()

    thread = threading.Thread(
        target=_sample_loop,
        name="curvytron-lightzero-gpu-sampler",
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def _install_lightzero_phase_profile(
    *,
    train_muzero: Any,
    profiler: _LightZeroPhaseProfiler,
    stop_after_learner_train_calls: int,
    skip_evaluator_eval: bool = False,
) -> Any:
    """Patch selected LightZero methods in place, returning a restore function."""

    originals: list[tuple[Any, str, Any]] = []
    patched_methods: set[tuple[int, str]] = set()
    installed_hooks: list[str] = []

    def hook_label(obj: Any, method_name: str) -> str:
        module = getattr(obj, "__module__", type(obj).__module__)
        qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", type(obj).__name__))
        return f"{module}.{qualname}.{method_name}"

    def patch_method(cls: Any, method_name: str, wrapped: Any) -> None:
        owner = next(
            (base for base in inspect.getmro(cls) if method_name in getattr(base, "__dict__", {})),
            None,
        )
        if owner is None:
            profiler.add_note(f"{hook_label(cls, method_name)} missing")
            return
        key = (id(owner), method_name)
        if key in patched_methods:
            return
        original = owner.__dict__[method_name]
        originals.append((owner, method_name, original))
        setattr(owner, method_name, wrapped(original))
        patched_methods.add(key)
        label = hook_label(owner, method_name)
        installed_hooks.append(label)
        profiler.add_installed_hook(label)

    def restore() -> None:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)

    def safe_int_attr(obj: Any, name: str, default: int) -> int:
        try:
            return int(getattr(obj, name, default) or default)
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            profiler.add_note(f"could not read integer attr {name}: {type(exc).__name__}: {exc}")
            return default

    def safe_len(value: Any, label: str) -> int | None:
        try:
            return len(value)
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            profiler.add_note(f"could not read length for {label}: {type(exc).__name__}: {exc}")
            return None

    def safe_batch_size(value: Any) -> int | None:
        shape = getattr(value, "shape", None)
        if shape:
            try:
                return int(shape[0])
            except Exception:
                pass
        size = getattr(value, "size", None)
        if callable(size):
            try:
                return int(size(0))
            except Exception:
                pass
        if isinstance(value, (list, tuple)) and value:
            first = value[0]
            shape = getattr(first, "shape", None)
            if shape:
                try:
                    return int(shape[0])
                except Exception:
                    pass
        return None

    def safe_key(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")

    def record_batch_size(prefix: str, value: Any, *, context: str | None = None) -> None:
        batch_size = safe_batch_size(value)
        if batch_size is None:
            return
        profiler.add_count(f"{prefix}_batch_sum", batch_size)
        profiler.add_sample(f"{prefix}_batch_size", batch_size, limit=120)
        if context:
            context_key = safe_key(context.removesuffix("_sec"))
            profiler.add_count(f"{prefix}_in_{context_key}_calls")
            profiler.add_count(f"{prefix}_in_{context_key}_batch_sum", batch_size)
            profiler.add_sample(
                f"{prefix}_in_{context_key}_batch_size",
                batch_size,
                limit=120,
            )

    def model_device_sample(model: Any) -> str | None:
        candidates = [
            model,
            getattr(model, "model", None),
            getattr(model, "_model", None),
            getattr(model, "module", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            parameters = getattr(candidate, "parameters", None)
            if not callable(parameters):
                continue
            try:
                parameter = next(parameters())
            except StopIteration:
                continue
            except Exception:
                continue
            device = getattr(parameter, "device", None)
            if device is not None:
                return str(device)
        return None

    def record_model_device(model: Any, *, label: str) -> None:
        device = model_device_sample(model)
        cls_name = f"{type(model).__module__}.{type(model).__qualname__}"
        profiler.add_sample("model_class", cls_name, limit=40)
        if device is not None:
            profiler.add_sample("model_parameter_device", f"{label}:{device}", limit=40)

    def patch_model_object(model: Any, *, label: str) -> None:
        if model is None:
            return
        record_model_device(model, label=label)
        model_cls = type(model)
        for method_name, timer_name, count_name, prefix in (
            (
                "initial_inference",
                "model_initial_inference_sec",
                "model_initial_inference_calls",
                "model_initial_inference",
            ),
            (
                "recurrent_inference",
                "model_recurrent_inference_sec",
                "model_recurrent_inference_calls",
                "model_recurrent_inference",
            ),
        ):
            if not callable(getattr(model, method_name, None)):
                continue

            def make_model_wrapped(
                original: Any,
                *,
                _timer_name: str = timer_name,
                _count_name: str = count_name,
                _prefix: str = prefix,
                _method_name: str = method_name,
            ) -> Any:
                def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    context = profiler.current_context()
                    record_model_device(self, label=_method_name)
                    if args:
                        record_batch_size(_prefix, args[0], context=context)
                    if context:
                        profiler.add_sample(
                            f"{_prefix}_context",
                            safe_key(context.removesuffix("_sec")),
                            limit=120,
                        )
                    with profiler.timer(_timer_name, sync_cuda=True):
                        result = original(self, *args, **kwargs)
                    profiler.add_count(_count_name)
                    return result

                return wrapped

            patch_method(model_cls, method_name, make_model_wrapped)

    def patch_policy_models(policy: Any) -> None:
        if policy is None:
            return
        profiler.add_sample(
            "policy_class",
            f"{type(policy).__module__}.{type(policy).__qualname__}",
            limit=20,
        )
        for attr_name in (
            "_model",
            "_learn_model",
            "_collect_model",
            "_eval_model",
            "_target_model",
        ):
            patch_model_object(
                getattr(policy, attr_name, None),
                label=f"policy.{attr_name}",
            )

    def record_mcts_search_inputs(searcher: Any, args: tuple[Any, ...]) -> None:
        cfg = getattr(searcher, "_cfg", None)
        num_simulations = getattr(cfg, "num_simulations", None)
        device = getattr(cfg, "device", None)
        roots = args[0] if args else None
        root_num = getattr(roots, "num", None)
        if root_num is not None:
            try:
                root_num_int = int(root_num)
            except Exception:
                root_num_int = None
            if root_num_int is not None:
                profiler.add_count("mcts_search_root_sum", root_num_int)
                profiler.add_sample("mcts_search_root_batch", root_num_int, limit=120)
                if num_simulations is not None:
                    try:
                        simulations_int = int(num_simulations)
                    except Exception:
                        simulations_int = None
                    if simulations_int is not None:
                        profiler.add_count(
                            "mcts_search_simulation_budget_sum",
                            simulations_int,
                        )
                        profiler.add_count(
                            "mcts_search_node_budget_sum",
                            root_num_int * simulations_int,
                        )
                        profiler.add_sample(
                            "mcts_search_simulations",
                            simulations_int,
                            limit=40,
                        )
        if device is not None:
            profiler.add_sample("mcts_cfg_device", str(device), limit=20)
        if len(args) > 1:
            patch_model_object(args[1], label="mcts.search_model_arg")

    def record_mcts_search_result(result: Any) -> None:
        if result is None:
            profiler.add_sample("mcts_search_return_type", "None", limit=40)
            return
        profiler.add_sample(
            "mcts_search_return_type",
            f"{type(result).__module__}.{type(result).__qualname__}",
            limit=40,
        )
        if isinstance(result, dict):
            profiler.add_sample(
                "mcts_search_return_keys",
                sorted(str(key) for key in result.keys())[:20],
                limit=20,
            )
            for key, value in list(result.items())[:8]:
                shape = getattr(value, "shape", None)
                if shape is not None:
                    profiler.add_sample(
                        f"mcts_search_return_shape.{key}",
                        [int(dim) for dim in shape],
                        limit=20,
                    )

    def patch_init(cls: Any, timer_name: str) -> None:
        def make_wrapped(original: Any) -> Any:
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                with profiler.timer(timer_name):
                    return original(self, *args, **kwargs)

            return wrapped

        patch_method(cls, "__init__", make_wrapped)

    def patch_module_function(module: Any, function_name: str, wrapped: Any) -> None:
        original = getattr(module, function_name, None)
        if not callable(original):
            profiler.add_note(f"{getattr(module, '__name__', module)}.{function_name} missing")
            return
        key = (id(module), function_name)
        if key in patched_methods:
            return
        originals.append((module, function_name, original))
        setattr(module, function_name, wrapped(original))
        patched_methods.add(key)
        installed_hooks.append(f"{getattr(module, '__name__', module)}.{function_name}")

    def patch_timed_method(
        cls: Any,
        method_name: str,
        timer_name: str,
        count_name: str,
        *,
        sync_cuda: bool = False,
    ) -> None:
        def make_wrapped(original: Any) -> Any:
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                with profiler.timer(timer_name, sync_cuda=sync_cuda):
                    result = original(self, *args, **kwargs)
                profiler.add_count(count_name)
                return result

            return wrapped

        patch_method(cls, method_name, make_wrapped)

    def patch_buffer_method(
        buffer_cls: Any,
        method_name: str,
        timer_name: str,
        count_name: str,
        *,
        count_segments: bool = False,
        sync_cuda: bool = False,
    ) -> None:
        def make_wrapped(original: Any) -> Any:
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                with profiler.timer(timer_name, sync_cuda=sync_cuda):
                    result = original(self, *args, **kwargs)
                try:
                    profiler.add_count(count_name)
                    if count_segments and args:
                        game_segments = safe_len(args[0], "pushed game segments")
                        if game_segments is not None:
                            profiler.add_count("game_segments_pushed", game_segments)
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    profiler.add_note(
                        f"replay {method_name} profile count failed: {type(exc).__name__}: {exc}"
                    )
                return result

            return wrapped

        patch_method(buffer_cls, method_name, make_wrapped)

    def patch_source_state_env_hooks() -> None:
        try:
            env_module = importlib.import_module(
                "curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env"
            )
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            profiler.add_note(f"could not import source-state env hooks: {type(exc).__name__}: {exc}")
            return
        env_cls = getattr(env_module, "CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv", None)
        if inspect.isclass(env_cls):
            patch_timed_method(env_cls, "reset", "env_reset_sec", "env_reset_calls")
            patch_timed_method(env_cls, "step", "env_step_sec", "env_step_calls")
            patch_timed_method(
                env_cls,
                "_lightzero_observation",
                "env_lightzero_obs_pack_sec",
                "env_obs_pack_calls",
            )
            patch_timed_method(
                env_cls,
                "_update_stack",
                "env_stack_update_sec",
                "env_stack_update_calls",
            )
            patch_timed_method(
                env_cls,
                "_write_telemetry_row",
                "env_telemetry_write_sec",
                "env_telemetry_write_calls",
            )
        else:
            profiler.add_note("source-state env class missing for env hooks")

        try:
            visual_module = importlib.import_module("curvyzero.env.vector_visual_observation")
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            profiler.add_note(f"could not import visual observation hooks: {type(exc).__name__}: {exc}")
        else:
            def make_rgb_canvas_render_wrapped(original: Any) -> Any:
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    with profiler.timer("env_render_rgb_canvas_sec"):
                        result = original(*args, **kwargs)
                    profiler.add_count("env_render_rgb_canvas_calls")
                    return result

                return wrapped

            def make_rgb_to_gray64_wrapped(original: Any) -> Any:
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    with profiler.timer("env_rgb_to_gray64_sec"):
                        result = original(*args, **kwargs)
                    profiler.add_count("env_rgb_to_gray64_calls")
                    return result

                return wrapped

            def make_gray64_render_wrapped(original: Any) -> Any:
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    with profiler.timer("env_render_gray64_sec"):
                        result = original(*args, **kwargs)
                    profiler.add_count("env_render_gray64_calls")
                    return result

                return wrapped

            patch_module_function(
                visual_module,
                "render_source_state_rgb_canvas_like",
                make_rgb_canvas_render_wrapped,
            )
            patch_module_function(
                visual_module,
                "rgb_canvas_like_to_gray64",
                make_rgb_to_gray64_wrapped,
            )
            patch_module_function(
                visual_module,
                "render_source_state_canvas_gray64",
                make_gray64_render_wrapped,
            )

        try:
            vector_env_module = importlib.import_module("curvyzero.env.vector_multiplayer_env")
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            profiler.add_note(f"could not import vector env hooks: {type(exc).__name__}: {exc}")
        else:
            vector_env_cls = getattr(vector_env_module, "VectorMultiplayerEnv", None)
            if inspect.isclass(vector_env_cls):
                patch_timed_method(
                    vector_env_cls,
                    "step",
                    "env_vector_step_sec",
                    "env_vector_step_calls",
                )
            else:
                profiler.add_note("VectorMultiplayerEnv missing for vector step hook")

        try:
            vector_runtime = importlib.import_module("curvyzero.env.vector_runtime")
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            profiler.add_note(f"could not import vector runtime hooks: {type(exc).__name__}: {exc}")
            return

        def make_step_many_wrapped(_original: Any) -> Any:
            def wrapped(step_input: Any) -> Any:
                phase_timers: dict[str, float] = {}
                started = time.perf_counter()
                vector_runtime.validate_step_input(step_input)
                result = vector_runtime._step_many_kernel(
                    step_input,
                    phase_timers=phase_timers,
                )
                profiler.add_time("env_runtime_step_many_sec", time.perf_counter() - started)
                profiler.add_count("env_runtime_step_many_calls")
                for key, value in phase_timers.items():
                    profiler.add_time(f"env_runtime_{key}", float(value))
                return result

            return wrapped

        patch_module_function(vector_runtime, "step_many", make_step_many_wrapped)

    def patch_policy_hooks() -> None:
        try:
            policy_module = importlib.import_module("lzero.policy.muzero")
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            profiler.add_note(f"could not import lzero.policy.muzero hooks: {type(exc).__name__}: {exc}")
            return
        policy_cls = getattr(policy_module, "MuZeroPolicy", None)
        if not inspect.isclass(policy_cls):
            profiler.add_note("MuZeroPolicy missing for policy hooks")
            return

        def make_policy_init(original: Any) -> Any:
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                with profiler.timer("policy_init_sec", sync_cuda=True):
                    result = original(self, *args, **kwargs)
                patch_policy_models(self)
                return result

            return wrapped

        def patch_policy_forward(
            method_name: str,
            timer_name: str,
            count_name: str,
        ) -> None:
            def make_wrapped(original: Any) -> Any:
                def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    patch_policy_models(self)
                    with profiler.timer(timer_name, sync_cuda=True):
                        result = original(self, *args, **kwargs)
                    profiler.add_count(count_name)
                    return result

                return wrapped

            patch_method(policy_cls, method_name, make_wrapped)

        patch_method(policy_cls, "__init__", make_policy_init)
        patch_policy_forward(
            "_forward_collect",
            "policy_forward_collect_sec",
            "policy_forward_collect_calls",
        )
        patch_policy_forward(
            "_forward_eval",
            "policy_forward_eval_sec",
            "policy_forward_eval_calls",
        )
        patch_policy_forward(
            "_forward_learn",
            "policy_forward_learn_sec",
            "policy_forward_learn_calls",
        )

    globals_map = getattr(train_muzero, "__globals__", {})

    try:
        if "Collector" in globals_map and inspect.isclass(globals_map["Collector"]):
            collector_cls = globals_map["Collector"]
            patch_init(collector_cls, "collector_init_sec")

            def make_collect(original: Any) -> Any:
                def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    before_envstep = safe_int_attr(self, "envstep", 0)
                    with profiler.timer("collector_collect_sec"):
                        result = original(self, *args, **kwargs)
                    try:
                        after_envstep = safe_int_attr(self, "envstep", before_envstep)
                        profiler.add_count("collector_collect_calls")
                        profiler.add_count(
                            "env_steps_collected", max(0, after_envstep - before_envstep)
                        )
                        game_segments = safe_len(result, "collector result")
                        if game_segments is not None:
                            profiler.add_count("game_segments_collected", game_segments)
                    except Exception as exc:  # pragma: no cover - remote diagnosis only.
                        profiler.add_note(
                            f"collector profile count failed: {type(exc).__name__}: {exc}"
                        )
                    return result

                return wrapped

            patch_method(collector_cls, "collect", make_collect)
        else:
            profiler.add_note("train_muzero globals did not expose Collector class")

        if "Evaluator" in globals_map and inspect.isclass(globals_map["Evaluator"]):
            evaluator_cls = globals_map["Evaluator"]
            patch_init(evaluator_cls, "evaluator_init_sec")

            def make_eval(original: Any) -> Any:
                def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    if skip_evaluator_eval:
                        profiler.add_count("evaluator_eval_skipped_calls")
                        n_episode = safe_int_attr(self, "_default_n_episode", 1)
                        skipped_returns = [0.0 for _ in range(max(1, n_episode))]
                        return False, {
                            "skipped": True,
                            "reason": "optimizer profile skipped stock LightZero evaluator",
                            "eval_episode_return": skipped_returns,
                            "eval_episode_return_mean": 0.0,
                            "reward_mean": 0.0,
                            "reward_std": 0.0,
                            "reward_max": 0.0,
                            "reward_min": 0.0,
                        }
                    with profiler.timer("evaluator_eval_sec"):
                        result = original(self, *args, **kwargs)
                    profiler.add_count("evaluator_eval_calls")
                    return result

                return wrapped

            patch_method(evaluator_cls, "eval", make_eval)
        else:
            profiler.add_note("train_muzero globals did not expose Evaluator class")

        if "BaseLearner" in globals_map and inspect.isclass(globals_map["BaseLearner"]):
            learner_cls = globals_map["BaseLearner"]
            patch_init(learner_cls, "learner_init_sec")

            def make_train(original: Any) -> Any:
                def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    before_iter = safe_int_attr(self, "train_iter", 0)
                    patch_policy_models(getattr(self, "policy", None))
                    patch_policy_models(getattr(self, "_policy", None))
                    with profiler.timer("learner_train_sec", sync_cuda=True):
                        result = original(self, *args, **kwargs)
                    patch_policy_models(getattr(self, "policy", None))
                    patch_policy_models(getattr(self, "_policy", None))
                    after_iter = safe_int_attr(self, "train_iter", before_iter)
                    profiler.add_count("learner_train_calls")
                    profiler.add_count("learner_train_iter_delta", max(0, after_iter - before_iter))
                    learner_train_calls = profiler.counts.get("learner_train_calls", 0)
                    if (
                        stop_after_learner_train_calls > 0
                        and learner_train_calls >= stop_after_learner_train_calls
                    ):
                        raise _LightZeroProfileStop(
                            "stopped after "
                            f"{learner_train_calls} BaseLearner.train calls "
                            "by optimizer profile cap"
                        )
                    return result

                return wrapped

            def make_save_checkpoint(original: Any) -> Any:
                def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    with profiler.timer("learner_save_checkpoint_sec"):
                        result = original(self, *args, **kwargs)
                    profiler.add_count("learner_save_checkpoint_calls")
                    return result

                return wrapped

            def make_call_hook(original: Any) -> Any:
                def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    hook_name = str(args[0]) if args else str(kwargs.get("name", "unknown"))
                    with profiler.timer(f"learner_hook_{hook_name}_sec"):
                        return original(self, *args, **kwargs)

                return wrapped

            patch_method(learner_cls, "train", make_train)
            patch_method(learner_cls, "save_checkpoint", make_save_checkpoint)
            patch_method(learner_cls, "call_hook", make_call_hook)
        else:
            profiler.add_note("train_muzero globals did not expose BaseLearner class")

        mcts_classes: list[Any] = []
        for class_name in ("MuZeroMCTSCtree", "MuZeroMCTSPtree"):
            candidate = globals_map.get(class_name)
            if inspect.isclass(candidate):
                mcts_classes.append(candidate)
        for module_name in ("lzero.mcts", "lzero.mcts.ctree_muzero", "lzero.mcts.ptree_muzero"):
            try:
                mcts_module = importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - remote diagnosis only.
                profiler.add_note(f"could not import {module_name}: {type(exc).__name__}: {exc}")
                continue
            for class_name in ("MuZeroMCTSCtree", "MuZeroMCTSPtree"):
                candidate = getattr(mcts_module, class_name, None)
                if inspect.isclass(candidate):
                    mcts_classes.append(candidate)

        seen_mcts_classes: set[int] = set()
        for mcts_cls in mcts_classes:
            if id(mcts_cls) in seen_mcts_classes:
                continue
            seen_mcts_classes.add(id(mcts_cls))

            def make_search(original: Any) -> Any:
                def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                    record_mcts_search_inputs(self, args)
                    with profiler.timer("mcts_search_sec", sync_cuda=True):
                        result = original(self, *args, **kwargs)
                    record_mcts_search_result(result)
                    profiler.add_count("mcts_search_calls")
                    return result

                return wrapped

            patch_method(mcts_cls, "search", make_search)
        if not seen_mcts_classes:
            profiler.add_note("no LightZero MuZero MCTS classes found for search hooks")

        buffer_classes: list[Any] = []
        buffer_names = (
            "MuZeroGameBuffer",
            "EfficientZeroGameBuffer",
            "SampledEfficientZeroGameBuffer",
            "SampledMuZeroGameBuffer",
            "GumbelMuZeroGameBuffer",
            "StochasticMuZeroGameBuffer",
        )
        for buffer_name in buffer_names:
            global_candidate = globals_map.get(buffer_name)
            if inspect.isclass(global_candidate):
                buffer_classes.append(global_candidate)
        try:
            mcts_module = importlib.import_module("lzero.mcts")
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            profiler.add_note(f"could not import lzero.mcts for GameBuffer patch: {type(exc).__name__}: {exc}")
        else:
            for buffer_name in buffer_names:
                candidate = getattr(mcts_module, buffer_name, None)
                if inspect.isclass(candidate):
                    buffer_classes.append(candidate)

        seen_buffer_classes: set[int] = set()
        for buffer_cls in buffer_classes:
            if id(buffer_cls) in seen_buffer_classes:
                continue
            seen_buffer_classes.add(id(buffer_cls))
            patch_init(buffer_cls, "replay_buffer_init_sec")
            patch_buffer_method(
                buffer_cls,
                "push_game_segments",
                "replay_push_game_segments_sec",
                "replay_push_calls",
                count_segments=True,
            )
            patch_buffer_method(
                buffer_cls,
                "remove_oldest_data_to_fit",
                "replay_remove_oldest_sec",
                "replay_remove_oldest_calls",
            )
            patch_buffer_method(
                buffer_cls,
                "sample",
                "replay_sample_sec",
                "replay_sample_calls",
                sync_cuda=True,
            )
            patch_buffer_method(
                buffer_cls,
                "update_priority",
                "replay_update_priority_sec",
                "replay_update_priority_calls",
            )
        if not seen_buffer_classes:
            profiler.add_note("no LightZero GameBuffer classes found for replay hooks")

        patch_source_state_env_hooks()
        patch_policy_hooks()
    except Exception:
        restore()
        raise

    for hook in installed_hooks:
        profiler.add_installed_hook(hook)
    return restore


def _install_live_checkpoint_publisher(
    *,
    train_muzero: Any,
    run_id: str,
    attempt_id: str,
    exp_name: Path,
    attempt_train_root: Path,
    background_eval_config: dict[str, Any] | None = None,
) -> Any:
    """Spawn eval workers from checkpoint saves without volume writes or commits."""

    if not background_eval_config or not background_eval_config.get("enabled"):
        return None
    globals_map = getattr(train_muzero, "__globals__", {})
    learner_cls = globals_map.get("BaseLearner")
    if not inspect.isclass(learner_cls):
        return None
    owner = next(
        (base for base in inspect.getmro(learner_cls) if "save_checkpoint" in getattr(base, "__dict__", {})),
        None,
    )
    if owner is None:
        return None
    original = owner.__dict__["save_checkpoint"]
    seen_checkpoint_refs: set[str] = set()

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        try:
            _spawn_checkpoint_eval_triggers(
                run_id=run_id,
                attempt_id=attempt_id,
                exp_name=exp_name,
                config=background_eval_config,
                seen_checkpoint_refs=seen_checkpoint_refs,
            )
        except Exception as exc:  # pragma: no cover - remote resilience only.
            print(
                "curvyzero checkpoint eval trigger failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        return result

    setattr(owner, "save_checkpoint", wrapped)

    def restore() -> None:
        setattr(owner, "save_checkpoint", original)

    return restore


def _install_lightzero_full_resume_state_hooks(
    *,
    train_muzero: Any,
    run_id: str,
    attempt_id: str,
    exp_name: Path,
    auto_resume: dict[str, Any],
) -> Any:
    """Save and restore LightZero state that is not inside the stock checkpoint."""

    holder: dict[str, Any] = {
        "collector": None,
        "evaluator": None,
        "replay_buffer": None,
        "resume_loaded": False,
        "skip_initial_eval": False,
    }
    restores: list[Any] = []

    def patch_method(owner: Any, name: str, make_wrapped: Any) -> None:
        original = getattr(owner, name)
        setattr(owner, name, make_wrapped(original))
        restores.append(lambda owner=owner, name=name, original=original: setattr(owner, name, original))

    def remember_instance(key: str) -> Any:
        def make_wrapped(original: Any) -> Any:
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = original(self, *args, **kwargs)
                holder[key] = self
                return result

            return wrapped

        return make_wrapped

    globals_map = getattr(train_muzero, "__globals__", {})
    restore_torch_load = _install_trusted_run_torch_load_retry(run_id=run_id)
    if restore_torch_load is not None:
        restores.append(restore_torch_load)

    original_random_collect = globals_map.get("random_collect")
    if callable(original_random_collect):

        def random_collect_wrapped(*args: Any, **kwargs: Any) -> Any:
            if holder.get("resume_loaded") and holder.get("replay_buffer_loaded"):
                return None
            return original_random_collect(*args, **kwargs)

        globals_map["random_collect"] = random_collect_wrapped
        restores.append(lambda: globals_map.__setitem__("random_collect", original_random_collect))

    collector_cls = globals_map.get("Collector")
    if inspect.isclass(collector_cls):
        patch_method(collector_cls, "__init__", remember_instance("collector"))

    evaluator_cls = globals_map.get("Evaluator")
    if inspect.isclass(evaluator_cls):
        patch_method(evaluator_cls, "__init__", remember_instance("evaluator"))

        def make_eval_wrapped(original: Any) -> Any:
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                if holder.get("skip_initial_eval"):
                    holder["skip_initial_eval"] = False
                    return False, {
                        "skipped": True,
                        "reason": "resumed run skips train_muzero initial eval to match uninterrupted flow",
                    }
                return original(self, *args, **kwargs)

            return wrapped

        patch_method(evaluator_cls, "eval", make_eval_wrapped)

    try:
        mcts_module = importlib.import_module("lzero.mcts")
    except Exception:
        mcts_module = None
    if mcts_module is not None:
        for buffer_name in (
            "MuZeroGameBuffer",
            "EfficientZeroGameBuffer",
            "SampledEfficientZeroGameBuffer",
            "SampledMuZeroGameBuffer",
            "GumbelMuZeroGameBuffer",
            "StochasticMuZeroGameBuffer",
        ):
            buffer_cls = getattr(mcts_module, buffer_name, None)
            if inspect.isclass(buffer_cls):
                patch_method(buffer_cls, "__init__", remember_instance("replay_buffer"))

    learner_cls = globals_map.get("BaseLearner")
    if inspect.isclass(learner_cls):

        def make_call_hook_wrapped(original: Any) -> Any:
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                hook_name = str(args[0]) if args else str(kwargs.get("name", ""))
                result = original(self, *args, **kwargs)
                if hook_name == "before_run":
                    loaded = _load_lightzero_resume_sidecar_state(
                        auto_resume=auto_resume,
                        holder=holder,
                        learner=self,
                    )
                    holder["resume_loaded"] = bool(loaded.get("loaded"))
                    holder["replay_buffer_loaded"] = bool(loaded.get("replay_buffer_loaded"))
                    holder["resume_load_result"] = loaded
                    holder["skip_initial_eval"] = bool(loaded.get("loaded"))
                return result

            return wrapped

        patch_method(learner_cls, "call_hook", make_call_hook_wrapped)

    try:
        learner_hook_module = importlib.import_module("ding.worker.learner.learner_hook")
        save_hook_cls = getattr(learner_hook_module, "SaveCkptHook", None)
    except Exception:
        save_hook_cls = None
    if inspect.isclass(save_hook_cls):

        def make_save_hook_wrapped(original: Any) -> Any:
            def wrapped(self: Any, engine: Any) -> Any:
                result = original(self, engine)
                _save_lightzero_resume_sidecar_state(
                    run_id=run_id,
                    attempt_id=attempt_id,
                    exp_name=exp_name,
                    holder=holder,
                    learner=engine,
                )
                return result

            return wrapped

        patch_method(save_hook_cls, "__call__", make_save_hook_wrapped)

    def restore() -> None:
        for undo in reversed(restores):
            undo()

    return restore


def _save_lightzero_resume_sidecar_state(
    *,
    run_id: str,
    attempt_id: str,
    exp_name: Path,
    holder: dict[str, Any],
    learner: Any,
) -> dict[str, Any]:
    iteration = int(getattr(learner, "train_iter", getattr(learner, "_last_iter", 0)))
    ckpt_path = Path(exp_name) / "ckpt" / _lightzero_iteration_checkpoint_name(iteration)
    if not ckpt_path.is_file():
        return {"saved": False, "reason": "matching_iteration_checkpoint_not_found"}

    sidecar_name = _lightzero_resume_state_name(iteration)
    sidecar_path = Path(exp_name) / LIGHTZERO_RESUME_STATE_DIRNAME / sidecar_name
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_lightzero_resume_sidecar_payload(
        run_id=run_id,
        attempt_id=attempt_id,
        iteration=iteration,
        checkpoint_path=ckpt_path,
        holder=holder,
        learner=learner,
    )
    _save_resume_state(sidecar_path, payload)

    mirror_path = (
        runs.volume_path(RUNS_MOUNT, runs.checkpoints_root_ref(TASK_ID, run_id))
        / LIGHTZERO_RESUME_STATE_DIRNAME
        / sidecar_name
    )
    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sidecar_path, mirror_path)
    return {
        "saved": True,
        "iteration": iteration,
        "path": str(sidecar_path),
        "ref": runs.file_ref(sidecar_path.resolve(), mount=RUNS_MOUNT.resolve()),
        "mirror_ref": runs.file_ref(mirror_path, mount=RUNS_MOUNT),
    }


def _build_lightzero_resume_sidecar_payload(
    *,
    run_id: str,
    attempt_id: str,
    iteration: int,
    checkpoint_path: Path,
    holder: dict[str, Any],
    learner: Any,
) -> dict[str, Any]:
    collector = holder.get("collector")
    evaluator = holder.get("evaluator")
    replay_buffer = holder.get("replay_buffer")
    policy = getattr(learner, "policy", None) or getattr(learner, "_policy", None)
    payload = {
        "schema_id": "curvyzero_lightzero_full_resume_sidecar/v0",
        "saved_at": runs.utc_timestamp(),
        "run_id": run_id,
        "attempt_id": attempt_id,
        "iteration": iteration,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "learner": {
            "train_iter": int(getattr(learner, "train_iter", iteration)),
            "collector_envstep": int(getattr(learner, "collector_envstep", 0)),
        },
        "collector": _lightzero_collector_state(collector),
        "evaluator": _lightzero_evaluator_state(evaluator),
        "replay_buffer": _lightzero_replay_buffer_state(replay_buffer),
        "policy_extras": _lightzero_policy_extra_state(policy),
        "rng": _lightzero_rng_state(),
        "state_scope": (
            "Extends the stock checkpoint with collector progress, "
            "evaluator progress, policy helper state when visible, and Python/NumPy/Torch "
            "random state. Replay GameSegments are recorded as metadata only for now because "
            "their raw LightZero objects are not portable across this save/load path. Live "
            "environment manager internals are not serialized."
        ),
    }
    return payload


def _load_lightzero_resume_sidecar_state(
    *,
    auto_resume: dict[str, Any],
    holder: dict[str, Any],
    learner: Any,
) -> dict[str, Any]:
    sidecar_path_text = auto_resume.get("resume_state_path")
    if not sidecar_path_text:
        return {"loaded": False, "reason": "resume_sidecar_not_found"}
    sidecar_path = Path(str(sidecar_path_text))
    if not sidecar_path.is_file():
        return {"loaded": False, "reason": "resume_sidecar_file_missing", "path": str(sidecar_path)}
    payload = _load_resume_state(sidecar_path)
    replay_buffer_loaded = _restore_lightzero_replay_buffer_state(
        holder.get("replay_buffer"),
        payload.get("replay_buffer"),
    )
    _restore_lightzero_collector_state(holder.get("collector"), payload.get("collector"))
    _restore_lightzero_evaluator_state(holder.get("evaluator"), payload.get("evaluator"))
    policy = getattr(learner, "policy", None) or getattr(learner, "_policy", None)
    _restore_lightzero_policy_extra_state(policy, payload.get("policy_extras"))
    learner_state = payload.get("learner") if isinstance(payload.get("learner"), dict) else {}
    if hasattr(learner, "collector_envstep") and learner_state.get("collector_envstep") is not None:
        learner.collector_envstep = int(learner_state["collector_envstep"])
    _restore_lightzero_rng_state(payload.get("rng"))
    return {
        "loaded": True,
        "path": str(sidecar_path),
        "iteration": payload.get("iteration"),
        "replay_buffer_loaded": replay_buffer_loaded,
        "state_scope": payload.get("state_scope"),
    }


def _lightzero_collector_state(collector: Any) -> dict[str, Any] | None:
    if collector is None:
        return None
    return {
        "total_envstep_count": int(getattr(collector, "_total_envstep_count", 0)),
        "total_episode_count": int(getattr(collector, "_total_episode_count", 0)),
        "total_duration": float(getattr(collector, "_total_duration", 0.0)),
        "last_train_iter": int(getattr(collector, "_last_train_iter", 0)),
    }


def _restore_lightzero_collector_state(collector: Any, state: Any) -> None:
    if collector is None or not isinstance(state, dict):
        return
    for attr, key, caster in (
        ("_total_envstep_count", "total_envstep_count", int),
        ("_total_episode_count", "total_episode_count", int),
        ("_total_duration", "total_duration", float),
        ("_last_train_iter", "last_train_iter", int),
    ):
        if key in state:
            setattr(collector, attr, caster(state[key]))


def _lightzero_evaluator_state(evaluator: Any) -> dict[str, Any] | None:
    if evaluator is None:
        return None
    return {
        "max_episode_return": float(getattr(evaluator, "_max_episode_return", float("-inf"))),
        "last_eval_iter": int(getattr(evaluator, "_last_eval_iter", 0)),
    }


def _restore_lightzero_evaluator_state(evaluator: Any, state: Any) -> None:
    if evaluator is None or not isinstance(state, dict):
        return
    if "max_episode_return" in state:
        evaluator._max_episode_return = float(state["max_episode_return"])
    if "last_eval_iter" in state:
        evaluator._last_eval_iter = int(state["last_eval_iter"])


def _lightzero_replay_buffer_state(replay_buffer: Any) -> dict[str, Any] | None:
    if replay_buffer is None:
        return None
    game_segment_buffer = getattr(replay_buffer, "game_segment_buffer", [])
    try:
        game_segment_count = len(game_segment_buffer)
    except Exception:
        game_segment_count = None
    return {
        "raw_game_segments_saved": False,
        "raw_game_segments_reason": (
            "LightZero GameSegment objects currently contain non-portable random generator "
            "state for this sidecar path."
        ),
        "game_segment_count": game_segment_count,
        "num_of_collected_episodes": int(getattr(replay_buffer, "num_of_collected_episodes", 0)),
        "base_idx": int(getattr(replay_buffer, "base_idx", 0)),
        "clear_time": int(getattr(replay_buffer, "clear_time", 0)),
        "keep_ratio": getattr(replay_buffer, "keep_ratio", 1),
        "compute_target_re_time": float(getattr(replay_buffer, "compute_target_re_time", 0.0)),
        "reuse_search_time": float(getattr(replay_buffer, "reuse_search_time", 0.0)),
        "origin_search_time": float(getattr(replay_buffer, "origin_search_time", 0.0)),
        "sample_times": int(getattr(replay_buffer, "sample_times", 0)),
        "active_root_num": int(getattr(replay_buffer, "active_root_num", 0)),
    }


def _restore_lightzero_replay_buffer_state(replay_buffer: Any, state: Any) -> bool:
    if replay_buffer is None or not isinstance(state, dict):
        return False
    if not state.get("raw_game_segments_saved"):
        return False
    for attr in (
        "game_segment_buffer",
        "game_pos_priorities",
        "game_segment_game_pos_look_up",
        "num_of_collected_episodes",
        "base_idx",
        "clear_time",
        "keep_ratio",
        "compute_target_re_time",
        "reuse_search_time",
        "origin_search_time",
        "sample_times",
        "active_root_num",
    ):
        if attr in state:
            setattr(replay_buffer, attr, state[attr])
    return True


def _lightzero_policy_extra_state(policy: Any) -> dict[str, Any] | None:
    if policy is None:
        return None
    state: dict[str, Any] = {}
    for attr_name in ("lr_scheduler", "_lr_scheduler"):
        scheduler = getattr(policy, attr_name, None)
        state_dict = getattr(scheduler, "state_dict", None)
        if callable(state_dict):
            state[f"{attr_name}_state_dict"] = state_dict()
    for attr_name in ("_target_model", "_learn_model", "_collect_model", "_eval_model"):
        wrapper_state = _lightzero_wrapper_extra_state(getattr(policy, attr_name, None))
        if wrapper_state:
            state[f"{attr_name}_wrapper"] = wrapper_state
    return state or None


def _restore_lightzero_policy_extra_state(policy: Any, state: Any) -> None:
    if policy is None or not isinstance(state, dict):
        return
    for attr_name in ("lr_scheduler", "_lr_scheduler"):
        scheduler = getattr(policy, attr_name, None)
        load_state_dict = getattr(scheduler, "load_state_dict", None)
        saved = state.get(f"{attr_name}_state_dict")
        if callable(load_state_dict) and saved is not None:
            load_state_dict(saved)
    for attr_name in ("_target_model", "_learn_model", "_collect_model", "_eval_model"):
        _restore_lightzero_wrapper_extra_state(
            getattr(policy, attr_name, None),
            state.get(f"{attr_name}_wrapper"),
        )


def _lightzero_wrapper_extra_state(wrapper: Any) -> dict[str, Any] | None:
    if wrapper is None:
        return None
    state: dict[str, Any] = {}
    for attr_name in (
        "_update_count",
        "_target_update_count",
        "update_count",
        "_step",
        "step",
    ):
        if not hasattr(wrapper, attr_name):
            continue
        value = getattr(wrapper, attr_name)
        if isinstance(value, (bool, int, float, str)):
            state[attr_name] = value
            continue
        item = getattr(value, "item", None)
        if callable(item):
            try:
                scalar = item()
            except Exception:
                continue
            if isinstance(scalar, (bool, int, float, str)):
                state[attr_name] = scalar
    return state or None


def _restore_lightzero_wrapper_extra_state(wrapper: Any, state: Any) -> None:
    if wrapper is None or not isinstance(state, dict):
        return
    for attr_name, value in state.items():
        if hasattr(wrapper, attr_name):
            setattr(wrapper, attr_name, value)


def _install_trusted_run_torch_load_retry(*, run_id: str) -> Any:
    """Let LightZero load trusted CurvyZero checkpoints under this run on PyTorch 2.6+."""

    try:
        import torch
    except Exception:
        return None

    original_load = torch.load
    trusted_root = runs.volume_path(RUNS_MOUNT, runs.run_root_ref(TASK_ID, run_id)).resolve()

    def wrapped_load(*args: Any, **kwargs: Any) -> Any:
        try:
            return original_load(*args, **kwargs)
        except pickle.UnpicklingError as exc:
            target = _torch_load_target_path(args, kwargs)
            if target is None or not _path_is_under(target, trusted_root):
                raise
            if "Weights only load failed" not in str(exc):
                raise
            retry_kwargs = dict(kwargs)
            retry_kwargs["weights_only"] = False
            return original_load(*args, **retry_kwargs)

    torch.load = wrapped_load

    def restore() -> None:
        torch.load = original_load

    return restore


def _torch_load_target_path(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Path | None:
    target = args[0] if args else kwargs.get("f")
    if isinstance(target, (str, os.PathLike)):
        return Path(target).resolve()
    name = getattr(target, "name", None)
    if isinstance(name, (str, os.PathLike)):
        return Path(name).resolve()
    return None


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _lightzero_rng_state() -> dict[str, Any]:
    import numpy as np
    import torch

    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def _restore_lightzero_rng_state(state: Any) -> None:
    if not isinstance(state, dict):
        return
    import numpy as np
    import torch

    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    if "torch_cpu" in state:
        torch.random.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda_all") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def _save_resume_state(path: Path, payload: dict[str, Any]) -> None:
    import cloudpickle

    with path.open("wb") as file:
        cloudpickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)


def _load_resume_state(path: Path) -> dict[str, Any]:
    import cloudpickle

    with path.open("rb") as file:
        loaded = cloudpickle.load(file)
    if not isinstance(loaded, dict):
        raise TypeError(f"resume sidecar is not a dict: {path}")
    return loaded


def _lightzero_iteration_checkpoint_name(iteration: int) -> str:
    return f"iteration_{int(iteration)}.pth.tar"


def _lightzero_resume_state_name(iteration: int) -> str:
    return f"iteration_{int(iteration)}.resume_state.pkl"


def _runtime_compute_summary(*, requested_compute: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "requested_compute": requested_compute,
        "cheap_gpu_resource": CHEAP_GPU_RESOURCE,
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        summary.update(
            {
                "torch_cuda_available": cuda_available,
                "torch_cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            }
        )
        if cuda_available:
            device = int(torch.cuda.current_device())
            summary.update(
                {
                    "torch_cuda_current_device": device,
                    "torch_cuda_device_name": torch.cuda.get_device_name(device),
                    "torch_cuda_capability": list(torch.cuda.get_device_capability(device)),
                }
            )
    except Exception as exc:  # pragma: no cover - remote runtime diagnosis only.
        summary["torch_cuda_probe_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def _audit_len(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


def _audit_get_item(value: Any, index: int, default: Any = None) -> Any:
    try:
        return value[index]
    except Exception:
        return default


def _audit_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except Exception:
        return None


def _audit_total_size(shape: list[int] | None) -> int | None:
    if not shape:
        return None
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _audit_compact_value(value: Any, *, depth: int = 3, max_items: int = 8) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth <= 0:
        shape = _audit_shape(value)
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "shape": shape,
        }
    shape = _audit_shape(value)
    if shape is not None:
        result: dict[str, Any] = {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "shape": shape,
        }
        total = _audit_total_size(shape)
        if total is not None and total <= 128 and hasattr(value, "tolist"):
            try:
                result["values"] = _audit_compact_value(
                    value.tolist(),
                    depth=depth - 1,
                    max_items=max_items,
                )
            except Exception as exc:
                result["values_error"] = f"{type(exc).__name__}: {exc}"
        elif hasattr(value, "__getitem__") and shape:
            try:
                result["sample"] = _audit_compact_value(
                    value[: min(int(shape[0]), max_items)],
                    depth=depth - 1,
                    max_items=max_items,
                )
            except Exception as exc:
                result["sample_error"] = f"{type(exc).__name__}: {exc}"
        return result
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        items = list(value.items())[:max_items]
        return {
            str(key): _audit_compact_value(item, depth=depth - 1, max_items=max_items)
            for key, item in items
        }
    if isinstance(value, (list, tuple)):
        return [
            _audit_compact_value(item, depth=depth - 1, max_items=max_items)
            for item in list(value)[:max_items]
        ]
    return repr(value)


def _audit_observation_summary(value: Any) -> dict[str, Any]:
    shape = _audit_shape(value)
    summary: dict[str, Any] = {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "shape": shape,
    }
    if isinstance(value, dict):
        summary["keys"] = sorted(str(key) for key in value.keys())
        for key in ("timestep", "to_play", "info"):
            if key in value:
                summary[key] = _audit_compact_value(value[key], depth=2)
    elif shape is None:
        summary["value"] = _audit_compact_value(value, depth=2)
    return summary


def _audit_segment_sequence_field(
    segment: Any,
    field: str,
    *,
    max_items: int = TARGET_AUDIT_MAX_STEPS_PER_SEGMENT,
) -> dict[str, Any]:
    values = getattr(segment, field, None)
    if values is None:
        return {"present": False}
    length = _audit_len(values)
    return {
        "present": True,
        "length": length,
        "first": [
            _audit_compact_value(_audit_get_item(values, index), depth=3)
            for index in range(min(length or 0, max_items))
        ],
    }


def _audit_segment_summary(
    segment: Any,
    *,
    collect_call_index: int,
    segment_index: int,
    metadata: Any,
) -> dict[str, Any]:
    fields = {
        field: _audit_segment_sequence_field(segment, field)
        for field in (
            "action_segment",
            "reward_segment",
            "to_play_segment",
            "child_visit_segment",
            "root_value_segment",
            "action_mask_segment",
            "obs_segment",
        )
    }
    lengths = {
        field: details.get("length")
        for field, details in fields.items()
        if details.get("present")
    }
    step_count = min(
        TARGET_AUDIT_MAX_STEPS_PER_SEGMENT,
        max([int(value) for value in lengths.values() if value is not None] or [0]),
    )
    steps: list[dict[str, Any]] = []
    for step_index in range(step_count):
        obs_value = _audit_get_item(getattr(segment, "obs_segment", None), step_index)
        row = {
            "step_index": step_index,
            "action": _audit_compact_value(
                _audit_get_item(getattr(segment, "action_segment", None), step_index),
                depth=3,
            ),
            "reward": _audit_compact_value(
                _audit_get_item(getattr(segment, "reward_segment", None), step_index),
                depth=3,
            ),
            "to_play": _audit_compact_value(
                _audit_get_item(getattr(segment, "to_play_segment", None), step_index),
                depth=3,
            ),
            "child_visit": _audit_compact_value(
                _audit_get_item(getattr(segment, "child_visit_segment", None), step_index),
                depth=3,
            ),
            "root_value": _audit_compact_value(
                _audit_get_item(getattr(segment, "root_value_segment", None), step_index),
                depth=3,
            ),
            "action_mask": _audit_compact_value(
                _audit_get_item(getattr(segment, "action_mask_segment", None), step_index),
                depth=3,
            ),
        }
        if obs_value is not None:
            row["obs"] = _audit_observation_summary(obs_value)
        steps.append(row)
    return {
        "collect_call_index": collect_call_index,
        "segment_index": segment_index,
        "segment_type": f"{type(segment).__module__}.{type(segment).__qualname__}",
        "len": _audit_len(segment),
        "action_space_size": _audit_compact_value(getattr(segment, "action_space_size", None)),
        "field_lengths": lengths,
        "fields": fields,
        "first_steps": steps,
        "metadata": _audit_compact_value(metadata, depth=3),
    }


def _audit_result_segments(result: Any) -> tuple[list[Any], list[Any], str]:
    if not isinstance(result, (list, tuple)):
        return [], [], f"collector result type {type(result).__name__} is not list/tuple"
    result_items = list(result)
    if result_items and isinstance(result_items[0], (list, tuple)):
        segments = list(result_items[0])
        metadata = list(result_items[1]) if len(result_items) > 1 and isinstance(result_items[1], (list, tuple)) else []
        return segments, metadata, "tuple/list first item"
    if any(hasattr(item, "action_segment") for item in result_items):
        return result_items, [], "direct list"
    return [], [], "collector result did not expose GameSegment-like items"


def _audit_sample_summary(result: Any, *, sample_call_index: int, args: tuple[Any, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_call_index": sample_call_index,
        "requested_batch_size": _audit_compact_value(args[0], depth=1) if args else None,
        "result_type": f"{type(result).__module__}.{type(result).__qualname__}",
    }
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        current_batch = result[0]
        target_batch = result[1]
        summary["current_batch"] = {
            "obs": _audit_compact_value(_audit_get_item(current_batch, 0), depth=2),
            "actions": _audit_compact_value(_audit_get_item(current_batch, 1), depth=3),
            "masks": _audit_compact_value(_audit_get_item(current_batch, 2), depth=3),
            "batch_indices": _audit_compact_value(_audit_get_item(current_batch, 3), depth=3),
            "weights": _audit_compact_value(_audit_get_item(current_batch, 4), depth=3),
        }
        summary["target_batch"] = {
            "rewards": _audit_compact_value(_audit_get_item(target_batch, 0), depth=3),
            "values": _audit_compact_value(_audit_get_item(target_batch, 1), depth=3),
            "policies": _audit_compact_value(_audit_get_item(target_batch, 2), depth=3),
        }
    else:
        summary["result"] = _audit_compact_value(result, depth=3)
    return summary


class _LightZeroTargetAudit:
    """Passive target/replay audit; records compact data and returns originals."""

    def __init__(self, *, mode: str, env_variant: str):
        self.mode = mode
        self.env_variant = env_variant
        self.installed_hooks: list[str] = []
        self.notes: list[str] = []
        self.errors: list[str] = []
        self.collect_calls = 0
        self.replay_sample_calls = 0
        self.replay_push_calls = 0
        self.segments_seen = 0
        self.segments_recorded: list[dict[str, Any]] = []
        self.replay_samples: list[dict[str, Any]] = []

    def add_installed_hook(self, hook: str) -> None:
        if hook not in self.installed_hooks:
            self.installed_hooks.append(hook)

    def add_note(self, note: str) -> None:
        if note not in self.notes and len(self.notes) < 30:
            self.notes.append(note)

    def add_error(self, error: str) -> None:
        if len(self.errors) < 20:
            self.errors.append(error)

    def record_collect_result(self, result: Any) -> None:
        collect_call_index = self.collect_calls
        self.collect_calls += 1
        try:
            segments, metadata, source = _audit_result_segments(result)
            self.add_note(f"collector result source: {source}")
            self.segments_seen += len(segments)
            remaining = TARGET_AUDIT_MAX_SEGMENTS - len(self.segments_recorded)
            for segment_index, segment in enumerate(segments[: max(0, remaining)]):
                segment_metadata = metadata[segment_index] if segment_index < len(metadata) else {}
                self.segments_recorded.append(
                    _audit_segment_summary(
                        segment,
                        collect_call_index=collect_call_index,
                        segment_index=segment_index,
                        metadata=segment_metadata,
                    )
                )
        except Exception as exc:  # pragma: no cover - remote audit only.
            self.add_error(f"collect audit failed: {type(exc).__name__}: {exc}")

    def record_replay_push(self, args: tuple[Any, ...]) -> None:
        self.replay_push_calls += 1
        if args:
            pushed = _audit_len(args[0])
            if pushed is not None:
                self.add_note(f"replay push saw {pushed} game segments")

    def record_replay_sample(self, result: Any, *, args: tuple[Any, ...]) -> None:
        sample_call_index = self.replay_sample_calls
        self.replay_sample_calls += 1
        if len(self.replay_samples) >= TARGET_AUDIT_MAX_REPLAY_SAMPLES:
            return
        try:
            self.replay_samples.append(
                _audit_sample_summary(
                    result,
                    sample_call_index=sample_call_index,
                    args=args,
                )
            )
        except Exception as exc:  # pragma: no cover - remote audit only.
            self.add_error(f"replay sample audit failed: {type(exc).__name__}: {exc}")

    def summary(self) -> dict[str, Any]:
        status = "collected" if self.segments_recorded or self.replay_samples else "missing"
        if self.mode == "dry":
            status = "not_installed"
        reason = None
        if status == "missing":
            reason = (
                "No GameSegment or replay sample summaries were captured. Check installed_hooks "
                "and notes; the intended hook point is Collector.collect after the original "
                "LightZero collect call returns, plus MuZeroGameBuffer.sample after the original "
                "sample call returns."
            )
        return {
            "schema_id": "curvyzero_lightzero_target_replay_audit/v0",
            "status": status,
            "missing_reason": reason,
            "mode": self.mode,
            "env_variant": self.env_variant,
            "training_behavior_changed": False,
            "hook_point": (
                "Passive wrappers around LightZero Collector.collect returns and "
                "GameBuffer.push_game_segments/sample returns."
            ),
            "limits": {
                "max_segments": TARGET_AUDIT_MAX_SEGMENTS,
                "max_steps_per_segment": TARGET_AUDIT_MAX_STEPS_PER_SEGMENT,
                "max_replay_samples": TARGET_AUDIT_MAX_REPLAY_SAMPLES,
            },
            "counts": {
                "collector_collect_calls": self.collect_calls,
                "game_segments_seen": self.segments_seen,
                "game_segments_recorded": len(self.segments_recorded),
                "replay_push_calls": self.replay_push_calls,
                "replay_sample_calls": self.replay_sample_calls,
                "replay_samples_recorded": len(self.replay_samples),
            },
            "installed_hooks": self.installed_hooks,
            "notes": self.notes,
            "errors": self.errors,
            "game_segments": self.segments_recorded,
            "replay_samples": self.replay_samples,
            "caveat": (
                "Audit artifact only. It mirrors compact fields from in-memory LightZero "
                "objects and sampled target batches when accessible; it does not prove reward "
                "credit correctness or change learner inputs."
            ),
        }


def _install_lightzero_target_audit(
    *,
    train_muzero: Any,
    audit: _LightZeroTargetAudit,
) -> Any:
    originals: list[tuple[Any, str, Any]] = []
    patched_methods: set[tuple[int, str]] = set()

    def hook_label(obj: Any, method_name: str) -> str:
        module = getattr(obj, "__module__", type(obj).__module__)
        qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", type(obj).__name__))
        return f"{module}.{qualname}.{method_name}"

    def patch_method(cls: Any, method_name: str, wrapped: Any) -> None:
        owner = next(
            (base for base in inspect.getmro(cls) if method_name in getattr(base, "__dict__", {})),
            None,
        )
        if owner is None:
            audit.add_note(f"{hook_label(cls, method_name)} missing for target audit")
            return
        key = (id(owner), method_name)
        if key in patched_methods:
            return
        original = owner.__dict__[method_name]
        originals.append((owner, method_name, original))
        setattr(owner, method_name, wrapped(original))
        patched_methods.add(key)
        audit.add_installed_hook(hook_label(owner, method_name))

    def restore() -> None:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)

    globals_map = getattr(train_muzero, "__globals__", {})

    collector_cls = globals_map.get("Collector")
    if not inspect.isclass(collector_cls):
        try:
            worker_module = importlib.import_module("lzero.worker")
            collector_cls = getattr(worker_module, "MuZeroCollector", None)
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            audit.add_note(f"could not import lzero.worker.MuZeroCollector: {type(exc).__name__}: {exc}")
            collector_cls = None
    if inspect.isclass(collector_cls):
        def make_collect_wrapped(original: Any) -> Any:
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = original(self, *args, **kwargs)
                audit.record_collect_result(result)
                return result

            return wrapped

        patch_method(collector_cls, "collect", make_collect_wrapped)
    else:
        audit.add_note("target audit missing Collector hook: no Collector class found")

    buffer_classes: list[Any] = []
    buffer_names = (
        "MuZeroGameBuffer",
        "EfficientZeroGameBuffer",
        "SampledEfficientZeroGameBuffer",
        "SampledMuZeroGameBuffer",
        "GumbelMuZeroGameBuffer",
        "StochasticMuZeroGameBuffer",
    )
    for buffer_name in buffer_names:
        candidate = globals_map.get(buffer_name)
        if inspect.isclass(candidate):
            buffer_classes.append(candidate)
    try:
        mcts_module = importlib.import_module("lzero.mcts")
    except Exception as exc:  # pragma: no cover - remote diagnosis only.
        audit.add_note(f"could not import lzero.mcts for target audit: {type(exc).__name__}: {exc}")
    else:
        for buffer_name in buffer_names:
            candidate = getattr(mcts_module, buffer_name, None)
            if inspect.isclass(candidate):
                buffer_classes.append(candidate)

    seen_buffer_classes: set[int] = set()
    for buffer_cls in buffer_classes:
        if id(buffer_cls) in seen_buffer_classes:
            continue
        seen_buffer_classes.add(id(buffer_cls))

        def make_push_wrapped(original: Any) -> Any:
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = original(self, *args, **kwargs)
                audit.record_replay_push(args)
                return result

            return wrapped

        def make_sample_wrapped(original: Any) -> Any:
            def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
                result = original(self, *args, **kwargs)
                audit.record_replay_sample(result, args=args)
                return result

            return wrapped

        patch_method(buffer_cls, "push_game_segments", make_push_wrapped)
        patch_method(buffer_cls, "sample", make_sample_wrapped)

    if not seen_buffer_classes:
        audit.add_note("target audit missing replay hook: no LightZero GameBuffer classes found")

    return restore


def _run_visual_survival_train(
    *,
    mode: str,
    compute: str,
    seed: int,
    run_id: str,
    attempt_id: str,
    max_env_step: int,
    max_train_iter: int,
    source_max_steps: int,
    decision_ms: float,
    collector_env_num: int,
    evaluator_env_num: int,
    n_evaluator_episode: int,
    n_episode: int,
    num_simulations: int,
    batch_size: int,
    lightzero_eval_freq: int,
    skip_lightzero_eval_in_profile: bool,
    profile_cuda_sync_enabled: bool,
    profile_allow_auto_resume: bool,
    save_ckpt_after_iter: int,
    stop_after_learner_train_calls: int,
    env_variant: str,
    reward_variant: str,
    ego_action_straight_override_probability: float,
    control_noise_profile_id: str,
    disable_death_for_profile: bool,
    env_telemetry_stride: int,
    env_manager_type: str,
    lightzero_multi_gpu: bool,
    opponent_policy_kind: str,
    opponent_checkpoint_ref: str | None,
    opponent_snapshot_ref: str | None,
    opponent_checkpoint_report_ref: str | None,
    opponent_checkpoint_state_key: str | None,
    background_eval_enabled: bool,
    background_eval_launch_kind: str,
    background_eval_compute: str,
    background_eval_id_prefix: str,
    background_eval_seed_count: int,
    background_eval_seed_rng_seed: int | None,
    background_eval_max_steps: int,
    background_eval_step_detail_limit: int | None,
    background_eval_num_simulations: int,
    background_eval_batch_size: int,
    background_gif_enabled: bool,
    background_gif_seed_offset: int,
    background_gif_max_steps: int,
    background_gif_frame_stride: int,
    background_gif_fps: float,
    background_gif_scale: int,
    background_gif_frame_size: int = DEFAULT_BACKGROUND_GIF_FRAME_SIZE,
    background_gif_checkpoint_seed_mixing_enabled: bool = (
        DEFAULT_BACKGROUND_GIF_CHECKPOINT_SEED_MIXING_ENABLED
    ),
) -> dict[str, Any]:
    started = time.perf_counter()
    if mode not in MODE_CHOICES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODE_CHOICES!r}")
    if compute not in COMPUTE_CHOICES:
        raise ValueError(f"unknown compute {compute!r}; expected one of {COMPUTE_CHOICES!r}")
    if env_variant not in ENV_VARIANT_CHOICES:
        raise ValueError(
            f"unknown env_variant {env_variant!r}; expected one of {ENV_VARIANT_CHOICES!r}"
        )
    if reward_variant not in REWARD_VARIANT_CHOICES:
        raise ValueError(
            f"unknown reward_variant {reward_variant!r}; expected one of {REWARD_VARIANT_CHOICES!r}"
        )
    opponent_policy_kind = _normalize_opponent_policy_kind_for_env(
        env_variant=env_variant,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
    )
    reward_variant = _normalize_reward_variant_for_env(
        env_variant=env_variant,
        reward_variant=reward_variant,
    )
    reward_policy = _reward_policy_for_variant(
        env_variant=env_variant,
        reward_variant=reward_variant,
    )
    lightzero_target_config = _lightzero_target_config_for_reward(
        env_variant=env_variant,
        reward_variant=reward_variant,
        source_max_steps=source_max_steps,
    )
    if opponent_policy_kind not in OPPONENT_POLICY_KIND_CHOICES:
        raise ValueError(
            f"unknown opponent_policy_kind {opponent_policy_kind!r}; "
            f"expected one of {OPPONENT_POLICY_KIND_CHOICES!r}"
        )
    if background_eval_compute not in BACKGROUND_EVAL_COMPUTE_CHOICES:
        raise ValueError(
            f"unknown background_eval_compute {background_eval_compute!r}; "
            f"expected one of {BACKGROUND_EVAL_COMPUTE_CHOICES!r}"
        )
    if background_eval_launch_kind not in BACKGROUND_EVAL_LAUNCH_CHOICES:
        raise ValueError(
            f"unknown background_eval_launch_kind {background_eval_launch_kind!r}; "
            f"expected one of {BACKGROUND_EVAL_LAUNCH_CHOICES!r}"
        )
    if env_manager_type not in ENV_MANAGER_TYPE_CHOICES:
        raise ValueError(
            f"unknown env_manager_type {env_manager_type!r}; "
            f"expected one of {ENV_MANAGER_TYPE_CHOICES!r}"
        )
    if env_variant == ENV_VARIANT_SOURCE_STATE_JOINT_ACTION:
        if opponent_policy_kind != OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION:
            raise ValueError(
                "source_state_joint_action controls both players with one "
                "centralized action; use opponent_policy_kind="
                f"{OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION!r}"
            )
    elif opponent_policy_kind == OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION:
        raise ValueError(
            f"{OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION!r} is only valid "
            "with env_variant='source_state_joint_action'"
        )
    if env_variant in (
        ENV_VARIANT_TURN_COMMIT,
        ENV_VARIANT_SOURCE_STATE_TURN_COMMIT,
        ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
    ) and opponent_policy_kind != OPPONENT_POLICY_KIND_FIXED_STRAIGHT:
        raise ValueError(f"{env_variant} env_variant does not use frozen opponent checkpoints")
    if float(decision_ms) <= 0.0:
        raise ValueError("decision_ms must be positive")
    if not 0.0 <= float(ego_action_straight_override_probability) <= 1.0:
        raise ValueError("ego_action_straight_override_probability must be in [0, 1]")
    if bool(disable_death_for_profile) and mode != "profile":
        raise ValueError("disable_death_for_profile is only allowed in mode='profile'")
    if env_variant == ENV_VARIANT_SOURCE_STATE_TURN_COMMIT and mode == "train":
        raise ValueError(
            "source_state_turn_commit is plumbing-smoke-only after target audit: "
            "stock LightZero stores pending and commit scalar steps as normal "
            "transitions, so reward credit is untrusted. Use mode='profile' for "
            "smokes or build a one-source-tick-per-transition path before training."
        )
    for name, value in (
        ("max_env_step", max_env_step),
        ("max_train_iter", max_train_iter),
        ("source_max_steps", source_max_steps),
        ("collector_env_num", collector_env_num),
        ("evaluator_env_num", evaluator_env_num),
        ("n_evaluator_episode", n_evaluator_episode),
        ("n_episode", n_episode),
        ("num_simulations", num_simulations),
        ("batch_size", batch_size),
        ("save_ckpt_after_iter", save_ckpt_after_iter),
        ("env_telemetry_stride", env_telemetry_stride),
        ("background_eval_seed_count", background_eval_seed_count),
        ("background_eval_max_steps", background_eval_max_steps),
        ("background_eval_num_simulations", background_eval_num_simulations),
        ("background_eval_batch_size", background_eval_batch_size),
        ("background_gif_frame_stride", background_gif_frame_stride),
        ("background_gif_scale", background_gif_scale),
        ("background_gif_frame_size", background_gif_frame_size),
    ):
        if int(value) < 1:
            raise ValueError(f"{name} must be at least 1")
    if int(background_gif_max_steps) < 0:
        raise ValueError(
            "background_gif_max_steps must be non-negative; 0 means no GIF step cap"
        )
    if int(stop_after_learner_train_calls) < 0:
        raise ValueError("stop_after_learner_train_calls must be non-negative")
    if int(lightzero_eval_freq) < 0:
        raise ValueError("lightzero_eval_freq must be non-negative")
    if float(background_gif_fps) <= 0.0:
        raise ValueError("background_gif_fps must be positive")
    packages = {
        "LightZero": _version_or_missing("LightZero", "lightzero"),
        "DI-engine": _version_or_missing("DI-engine", "ding"),
        "torch": _version_or_missing("torch"),
        "gym": _version_or_missing("gym"),
        "numpy": _version_or_missing("numpy"),
    }
    problems: list[str] = []
    if packages["LightZero"] != LIGHTZERO_VERSION:
        problems.append(
            f"installed LightZero version is {packages['LightZero']!r}, expected {LIGHTZERO_VERSION!r}"
        )

    started_at = runs.utc_timestamp()
    modal_task_id = os.environ.get("MODAL_TASK_ID")
    attempt_root_ref = runs.attempt_root_ref(TASK_ID, run_id, attempt_id)
    attempt_train_ref = runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
    attempt_root = runs.volume_path(RUNS_MOUNT, attempt_root_ref)
    attempt_train_root = runs.volume_path(RUNS_MOUNT, attempt_train_ref)
    telemetry_path = attempt_train_root / "env_steps.jsonl"
    exp_name_ref = attempt_train_ref / "lightzero_exp"
    exp_name = Path(exp_name_ref.as_posix())
    env_spec = _env_variant_spec(env_variant)
    opponent_checkpoint = _resolve_opponent_checkpoint_for_env(
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
        opponent_checkpoint_report_ref=opponent_checkpoint_report_ref,
    )

    gif_browser_run_marker_enabled = bool(background_gif_enabled)
    command = {
        "mode": mode,
        "compute": compute,
        "seed": int(seed),
        "reset_seed_strategy": "dynamic_seed_sequence_from_run_seed_and_reset_index/v0",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "max_env_step": int(max_env_step),
        "max_train_iter": int(max_train_iter),
        "source_max_steps": int(source_max_steps),
        "decision_ms": float(decision_ms),
        "collector_env_num": int(collector_env_num),
        "evaluator_env_num": int(evaluator_env_num),
        "n_evaluator_episode": int(n_evaluator_episode),
        "n_episode": int(n_episode),
        "num_simulations": int(num_simulations),
        "batch_size": int(batch_size),
        "lightzero_eval_freq": int(lightzero_eval_freq),
        "skip_lightzero_eval_in_profile": bool(skip_lightzero_eval_in_profile),
        "profile_cuda_sync_enabled": bool(profile_cuda_sync_enabled),
        "profile_allow_auto_resume": bool(profile_allow_auto_resume),
        "lightzero_multi_gpu": bool(lightzero_multi_gpu),
        "save_ckpt_after_iter": int(save_ckpt_after_iter),
        "stop_after_learner_train_calls": int(stop_after_learner_train_calls),
        "env_variant": env_variant,
        "env_type": env_spec["env_type"],
        "env_id": env_spec["env_id"],
        "action_space_size": int(env_spec.get("action_space_size", 3)),
        "reward_variant": reward_variant,
        "reward_schema_id": reward_policy["reward_schema_id"],
        "reward_policy": reward_policy,
        "lightzero_target_config": lightzero_target_config,
        "observation_schema_id": env_spec["observation_schema_id"],
        "debug_fidelity_only": env_spec["debug_fidelity_only"],
        "ego_action_straight_override_probability": float(
            ego_action_straight_override_probability
        ),
        "disable_death_for_profile": bool(disable_death_for_profile),
        "death_mode": (
            "profile_no_death" if disable_death_for_profile else "normal"
        ),
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
        "control_noise_profile_id": str(control_noise_profile_id),
        "env_telemetry_stride": int(env_telemetry_stride),
        "env_manager_type": env_manager_type,
        "profile_label": "profile" if mode == "profile" else None,
        "learning_proof": False,
        "source_fidelity_claim": env_spec["source_fidelity_claim"],
        "opponent_policy_kind": opponent_policy_kind,
        "opponent_training_relation": (
            env_spec["opponent_training_relation"]
            or _opponent_training_relation(opponent_policy_kind)
        ),
        "current_policy_self_play": env_spec["current_policy_self_play"],
        "current_policy_self_play_blocker": env_spec["current_policy_self_play_blocker"],
        "current_policy_self_play_caveat": env_spec["current_policy_self_play_caveat"],
        "trusted_current_policy_self_play": env_spec["trusted_current_policy_self_play"],
        "simultaneous_game_theory_claim": env_spec["simultaneous_game_theory_claim"],
        "opponent_checkpoint_ref": opponent_checkpoint_ref,
        "opponent_snapshot_ref": opponent_snapshot_ref,
        "opponent_checkpoint_report_ref": (
            opponent_checkpoint["checkpoint_ref"] if opponent_checkpoint else opponent_checkpoint_report_ref
        ),
        "opponent_checkpoint_state_key": opponent_checkpoint_state_key,
        "background_eval_enabled": bool(background_eval_enabled),
        "background_eval_launch_kind": background_eval_launch_kind,
        "background_eval_compute": background_eval_compute,
        "background_eval_id_prefix": background_eval_id_prefix,
        "background_eval_seed_count": int(background_eval_seed_count),
        "background_eval_seed_rng_seed": (
            int(background_eval_seed_rng_seed)
            if background_eval_seed_rng_seed is not None
            else None
        ),
        "background_eval_max_steps": int(background_eval_max_steps),
        "background_eval_step_detail_limit": (
            int(background_eval_step_detail_limit)
            if background_eval_step_detail_limit is not None
            else None
        ),
        "background_eval_num_simulations": int(background_eval_num_simulations),
        "background_eval_batch_size": int(background_eval_batch_size),
        "background_gif_enabled": bool(background_gif_enabled),
        "background_gif_seed_offset": int(background_gif_seed_offset),
        "background_gif_checkpoint_seed_mixing_enabled": bool(
            background_gif_checkpoint_seed_mixing_enabled
        ),
        "background_gif_max_steps": int(background_gif_max_steps),
        "background_gif_frame_stride": int(background_gif_frame_stride),
        "background_gif_fps": float(background_gif_fps),
        "background_gif_scale": int(background_gif_scale),
        "background_gif_frame_size": int(background_gif_frame_size),
        "gif_browser_run_marker_enabled": gif_browser_run_marker_enabled,
        "gif_browser_run_marker_ref": (
            runs.gif_browser_run_marker_ref(TASK_ID, run_id).as_posix()
            if gif_browser_run_marker_enabled
            else None
        ),
    }
    if gif_browser_run_marker_enabled:
        _write_gif_browser_run_marker(run_id=run_id, created_at=started_at)
    _write_run_manifest_once(run_id=run_id, config=command)
    _write_attempt_state(
        run_id=run_id,
        attempt_id=attempt_id,
        status="running",
        started_at=started_at,
        ended_at=None,
        summary_ref=None,
        config=command,
        modal_task_id=modal_task_id,
    )
    _write_latest_attempt(
        run_id=run_id,
        attempt_id=attempt_id,
        status="running",
        started_at=started_at,
        ended_at=None,
        summary_ref=None,
        modal_task_id=modal_task_id,
    )
    _write_train_status_heartbeat(
        run_id=run_id,
        attempt_id=attempt_id,
        status="running",
        stage="before_train_muzero",
        started_at=started_at,
        modal_task_id=modal_task_id,
        config=command,
        exp_name_ref=exp_name_ref.as_posix(),
    )
    entry_module = importlib.import_module("lzero.entry")
    train_muzero = entry_module.train_muzero
    patched = _build_visual_survival_configs(
        seed=seed,
        exp_name=exp_name,
        telemetry_path=telemetry_path,
        cuda=_compute_uses_cuda(compute),
        max_env_step=max_env_step,
        source_max_steps=source_max_steps,
        decision_ms=decision_ms,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=n_evaluator_episode,
        n_episode=n_episode,
        num_simulations=num_simulations,
        batch_size=batch_size,
        lightzero_eval_freq=lightzero_eval_freq,
        lightzero_multi_gpu=lightzero_multi_gpu,
        max_train_iter=max_train_iter,
        save_ckpt_after_iter=save_ckpt_after_iter,
        env_variant=env_variant,
        ego_action_straight_override_probability=ego_action_straight_override_probability,
        control_noise_profile_id=control_noise_profile_id,
        disable_death_for_profile=disable_death_for_profile,
        env_telemetry_stride=env_telemetry_stride,
        env_manager_type=env_manager_type,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint=opponent_checkpoint,
        opponent_snapshot_ref=opponent_snapshot_ref,
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
        reward_variant=reward_variant,
    )
    auto_resume = _prepare_lightzero_auto_resume(
        run_id=run_id,
        attempt_id=attempt_id,
        exp_name_ref=exp_name_ref,
    )
    command["auto_resume"] = {
        "enabled": True,
        "found": bool(auto_resume.get("found")),
        "checkpoint_ref": auto_resume.get("checkpoint_ref"),
        "checkpoint_iteration": auto_resume.get("iteration"),
        "source_kind": auto_resume.get("source_kind"),
        "resume_state_found": bool(auto_resume.get("resume_state_found")),
        "resume_state_ref": auto_resume.get("resume_state_ref"),
        "resume_state_source_kind": auto_resume.get("resume_state_source_kind"),
        "state_scope": auto_resume.get("state_scope"),
    }
    if mode == "profile" and auto_resume.get("found") and not profile_allow_auto_resume:
        problems.append(
            "profile auto-resume found an existing checkpoint; use a fresh run_id "
            "or set --profile-allow-auto-resume to profile resumed state intentionally"
        )
    _write_attempt_state(
        run_id=run_id,
        attempt_id=attempt_id,
        status="running",
        started_at=started_at,
        ended_at=None,
        summary_ref=None,
        config=command,
        modal_task_id=modal_task_id,
    )
    _write_train_status_heartbeat(
        run_id=run_id,
        attempt_id=attempt_id,
        status="running",
        stage="auto_resume_checked",
        started_at=started_at,
        modal_task_id=modal_task_id,
        config=command,
        exp_name_ref=exp_name_ref.as_posix(),
    )
    if auto_resume.get("found"):
        patched["patches"].append(
            _set_load_ckpt_before_run(
                patched["main_config"],
                str(auto_resume["checkpoint_path"]),
            )
        )
        patched["surface"] = _extract_surface(
            patched["main_config"],
            patched["create_config"],
            max_env_step=max_env_step,
            max_train_iter=max_train_iter,
        )
    surface = patched["surface"]
    problems.extend(_validate_visual_survival_surface(surface=surface, command=command))

    compile_summary = _compile_config_summary(
        patched["main_config"],
        patched["create_config"],
        seed=seed,
    )
    if not compile_summary.get("ok", False):
        problems.append(f"compile_config failed: {compile_summary.get('error')}")

    train_result: dict[str, Any] | None = None
    called_train_muzero = False
    stdout_text = ""
    stderr_text = ""
    profiler = _LightZeroPhaseProfiler(
        enabled=mode == "profile",
        cuda_sync_enabled=profile_cuda_sync_enabled,
    )
    target_audit = _LightZeroTargetAudit(mode=mode, env_variant=command["env_variant"])
    if mode in {"train", "profile"} and not problems:
        os.chdir(RUNS_MOUNT)
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        train_started = time.perf_counter()
        restore_profile = None
        restore_target_audit = None
        restore_live_publisher = None
        restore_resume_state = None
        gpu_sampler = None
        try:
            restore_resume_state = _install_lightzero_full_resume_state_hooks(
                train_muzero=train_muzero,
                run_id=run_id,
                attempt_id=attempt_id,
                exp_name=exp_name,
                auto_resume=auto_resume,
            )
            restore_live_publisher = _install_live_checkpoint_publisher(
                train_muzero=train_muzero,
                run_id=run_id,
                attempt_id=attempt_id,
                exp_name=exp_name,
                attempt_train_root=attempt_train_root,
                background_eval_config=(
                    _background_eval_config_from_command(command)
                    if command["background_eval_launch_kind"] == BACKGROUND_EVAL_LAUNCH_HOOK
                    else None
                ),
            )
            restore_target_audit = _install_lightzero_target_audit(
                train_muzero=train_muzero,
                audit=target_audit,
            )
            if mode == "profile":
                restore_profile = _install_lightzero_phase_profile(
                    train_muzero=train_muzero,
                    profiler=profiler,
                    stop_after_learner_train_calls=int(stop_after_learner_train_calls),
                    skip_evaluator_eval=bool(skip_lightzero_eval_in_profile),
                )
                gpu_sampler = _start_gpu_sampler(
                    profiler=profiler,
                    interval_sec=profiler.gpu_sample_interval_sec,
                )
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                called_train_muzero = True
                with (
                    profiler.timer("train_muzero_wall_sec")
                    if mode == "profile"
                    else contextlib.nullcontext()
                ):
                    output = train_muzero(
                        [patched["main_config"], patched["create_config"]],
                        seed=seed,
                        max_train_iter=max_train_iter,
                        max_env_step=max_env_step,
                    )
            stdout_text = stdout_buffer.getvalue()
            stderr_text = stderr_buffer.getvalue()
            train_result = {
                "ok": True,
                "return_type": type(output).__name__,
                "elapsed_sec": round(time.perf_counter() - train_started, 6),
                "log_signals": _parse_training_signals(stdout_text, stderr_text),
            }
        except _LightZeroProfileStop as exc:
            stdout_text = stdout_buffer.getvalue()
            stderr_text = stderr_buffer.getvalue()
            profiler.add_note(str(exc))
            train_result = {
                "ok": True,
                "stopped_by_optimizer_profile_cap": True,
                "stop_reason": str(exc),
                "elapsed_sec": round(time.perf_counter() - train_started, 6),
                "log_signals": _parse_training_signals(stdout_text, stderr_text),
            }
        except Exception as exc:  # pragma: no cover - remote trainer diagnosis.
            stdout_text = stdout_buffer.getvalue()
            stderr_text = stderr_buffer.getvalue()
            problems.append(f"LightZero train_muzero failed: {type(exc).__name__}: {exc}")
            train_result = {
                "ok": False,
                "elapsed_sec": round(time.perf_counter() - train_started, 6),
                "log_signals": _parse_training_signals(stdout_text, stderr_text),
                **_exception_result(exc),
            }
        finally:
            if restore_profile is not None:
                try:
                    restore_profile()
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    profiler.add_note(f"phase profile restore failed: {type(exc).__name__}: {exc}")
            if restore_target_audit is not None:
                try:
                    restore_target_audit()
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    target_audit.add_error(
                        f"target audit restore failed: {type(exc).__name__}: {exc}"
                    )
            if restore_live_publisher is not None:
                try:
                    restore_live_publisher()
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    profiler.add_note(
                        f"live checkpoint publisher restore failed: {type(exc).__name__}: {exc}"
                    )
            if restore_resume_state is not None:
                try:
                    restore_resume_state()
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    profiler.add_note(
                        f"full resume state restore hook cleanup failed: {type(exc).__name__}: {exc}"
                    )
            if gpu_sampler is not None:
                gpu_stop_event, gpu_thread = gpu_sampler
                gpu_stop_event.set()
                gpu_thread.join(timeout=5)
                profiler.sample_gpu()

    artifact_summary = _scan_lightzero_artifacts(str(exp_name))
    checkpoint_mirror = _mirror_lightzero_checkpoints(
        run_id=run_id,
        artifact_summary=artifact_summary,
    )
    action_summary = _summarize_env_step_telemetry(telemetry_path)
    training_readiness_gate = (
        _source_state_fixed_opponent_training_readiness_gate(
            command=command,
            surface=surface,
            action_observability=action_summary,
        )
        if command["env_variant"] == ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT
        else None
    )
    if mode == "train" and train_result and train_result.get("ok"):
        if not artifact_summary.get("checkpoint_files"):
            problems.append("no LightZero checkpoint artifacts were discovered")
        if not checkpoint_mirror.get("copied_checkpoints"):
            problems.append("no LightZero checkpoints were mirrored to curvyzero-runs")
        if int(action_summary.get("row_count", 0)) <= 0:
            problems.append("no env-step action telemetry rows were written")

    runtime_compute = _runtime_compute_summary(requested_compute=compute)
    phase_profile = profiler.summary()
    target_audit_summary = target_audit.summary()
    summary = {
        "schema_id": "curvyzero_lightzero_curvytron_visual_survival_train_summary/v0",
        "task_id": TASK_ID,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "algorithm": "LightZero MuZero",
        "mode": mode,
        "compute": compute,
        "ok": not problems and (mode == "dry" or bool(train_result and train_result.get("ok"))),
        "problems": problems,
        "called_train_muzero": called_train_muzero,
        "trainer_entrypoint": "lzero.entry.train_muzero",
        "packages": packages,
        "runtime_compute": runtime_compute,
        "command": command,
        "opponent_policy_kind": command["opponent_policy_kind"],
        "opponent_training_relation": command["opponent_training_relation"],
        "current_policy_self_play": command["current_policy_self_play"],
        "current_policy_self_play_blocker": command["current_policy_self_play_blocker"],
        "current_policy_self_play_caveat": command["current_policy_self_play_caveat"],
        "trusted_current_policy_self_play": command["trusted_current_policy_self_play"],
        "simultaneous_game_theory_claim": command["simultaneous_game_theory_claim"],
        "surface": surface,
        "opponent_checkpoint": opponent_checkpoint,
        "auto_resume": auto_resume,
        "patches": patched["patches"],
        "compile_config": compile_summary,
        "train_result": train_result,
        "phase_profile": phase_profile,
        "target_audit": target_audit_summary,
        "lightzero_artifacts": artifact_summary,
        "checkpoint_mirror": checkpoint_mirror,
        "action_observability": action_summary,
        "training_readiness_gate": training_readiness_gate,
        "background_checkpoint_eval": {
            "enabled": command["background_eval_enabled"],
            "launch_kind": command["background_eval_launch_kind"],
            "trigger": (
                "spawn_source_checkpoint_in_save_checkpoint_no_volume_commit"
                if command["background_eval_launch_kind"] == BACKGROUND_EVAL_LAUNCH_HOOK
                else "external_checkpoint_poller_no_training_loop_commit"
            ),
            "compute": command["background_eval_compute"],
            "eval_id_prefix": command["background_eval_id_prefix"],
            "seed_count": command["background_eval_seed_count"],
            "seed_rng_seed": command["background_eval_seed_rng_seed"],
            "max_eval_steps": command["background_eval_max_steps"],
            "step_detail_limit": command["background_eval_step_detail_limit"],
            "num_simulations": command["background_eval_num_simulations"],
            "batch_size": command["background_eval_batch_size"],
            "checkpoint_wait_timeout_sec": DEFAULT_BACKGROUND_CHECKPOINT_WAIT_TIMEOUT_SEC,
            "selfplay_gif": {
                "enabled": command["background_gif_enabled"],
                "capture_env_variant": ENV_VARIANT_SOURCE_STATE_TURN_COMMIT,
                "frame_source": "source_state_rgb_canvas_like",
                "frame_schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
                "browser_pixel_fidelity": False,
                "seed_offset": command["background_gif_seed_offset"],
                "checkpoint_seed_mixing_enabled": command.get(
                    "background_gif_checkpoint_seed_mixing_enabled",
                    DEFAULT_BACKGROUND_GIF_CHECKPOINT_SEED_MIXING_ENABLED,
                ),
                "max_steps": _normalize_background_gif_max_steps(
                    command["background_gif_max_steps"]
                ),
                "step_limit_kind": _background_gif_step_limit_kind(
                    command["background_gif_max_steps"]
                ),
                "frame_stride": command["background_gif_frame_stride"],
                "fps": command["background_gif_fps"],
                "frame_size": command["background_gif_frame_size"],
            },
        },
        "env_variant": command["env_variant"],
        "single_product_runtime_path": command["single_product_runtime_path"],
        "legacy_debug_variant": command["legacy_debug_variant"],
        "underlying_env_class": command["underlying_env_class"],
        "runtime_env_impl_id": command["runtime_env_impl_id"],
        "runtime_topology": command["runtime_topology"],
        "two_seat_self_play": command["two_seat_self_play"],
        "current_policy_two_seat_action_collection": command[
            "current_policy_two_seat_action_collection"
        ],
        "two_seat_self_play_status": command["two_seat_self_play_status"],
        "fixed_opponent_is_two_seat_self_play": command[
            "fixed_opponent_is_two_seat_self_play"
        ],
        "browser_pixel_fidelity": command["browser_pixel_fidelity"],
        "uses_ale": command["uses_ale"],
        "visual_surface": command["visual_surface"],
        "visual_truth_level": command["visual_truth_level"],
        "visual_source_state_backed": command["visual_source_state_backed"],
        "source_fidelity_claim": command["source_fidelity_claim"],
        "debug_fidelity_only": command["debug_fidelity_only"],
        "learning_proof": False,
        "reward_schema_id": command["reward_schema_id"],
        "reward_policy": command["reward_policy"],
        "elapsed_sec": round(time.perf_counter() - started, 6),
    }

    attempt_train_root.mkdir(parents=True, exist_ok=True)
    summary_path = attempt_train_root / "summary.json"
    config_path = attempt_root / "config.json"
    command_path = attempt_root / "command.json"
    stdout_path = attempt_train_root / "stdout_tail.txt"
    stderr_path = attempt_train_root / "stderr_tail.txt"
    artifacts_path = attempt_train_root / "lightzero_artifacts_manifest.json"
    actions_path = attempt_train_root / "action_observability.json"
    target_audit_path = attempt_train_root / "target_audit.json"
    runs.write_json(summary_path, summary)
    runs.write_json(
        config_path,
        {
            "command": command,
            "main_config": _to_plain(patched["main_config"]),
            "create_config": _to_plain(patched["create_config"]),
        },
    )
    runs.write_json(command_path, command)
    runs.write_json(artifacts_path, artifact_summary)
    runs.write_json(actions_path, action_summary)
    runs.write_json(target_audit_path, target_audit_summary)
    _write_text(stdout_path, "\n".join(_compact_log_tail(stdout_text)) + ("\n" if stdout_text else ""))
    _write_text(stderr_path, "\n".join(_compact_log_tail(stderr_text, limit=30)) + ("\n" if stderr_text else ""))

    ended_at = runs.utc_timestamp()
    status = "completed" if summary["ok"] else "failed"
    summary_ref = runs.file_ref(summary_path, mount=RUNS_MOUNT)
    attempt_manifest = _write_attempt_state(
        run_id=run_id,
        attempt_id=attempt_id,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        summary_ref=summary_ref,
        config=command,
        modal_task_id=modal_task_id,
    )
    latest_attempt = _write_latest_attempt(
        run_id=run_id,
        attempt_id=attempt_id,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        summary_ref=summary_ref,
        modal_task_id=modal_task_id,
    )
    result = {
        "ok": summary["ok"],
        "status": status,
        "mode": mode,
        "compute": compute,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "called_train_muzero": called_train_muzero,
        "summary_ref": summary_ref,
        "attempt_manifest": attempt_manifest,
        "latest_attempt": latest_attempt,
        "problems": problems,
        "command": command,
        "runtime_compute": summary.get("runtime_compute"),
        "action_observability": action_summary,
        "checkpoint_mirror": checkpoint_mirror,
        "auto_resume": auto_resume,
        "phase_profile": phase_profile,
        "target_audit": target_audit_summary,
        "artifact_refs": {
            "summary": runs.file_summary(summary_path, mount=RUNS_MOUNT),
            "config": runs.file_summary(config_path, mount=RUNS_MOUNT),
            "command": runs.file_summary(command_path, mount=RUNS_MOUNT),
            "actions": runs.file_summary(actions_path, mount=RUNS_MOUNT),
            "target_audit": runs.file_summary(target_audit_path, mount=RUNS_MOUNT),
            "lightzero_artifacts": runs.file_summary(artifacts_path, mount=RUNS_MOUNT),
        },
    }
    plain_result = _to_plain(result)
    print(json.dumps(_compact_train_result_for_output(plain_result), indent=2, sort_keys=True))
    return plain_result


def _run_lightzero_checkpoint_opponent_modal_smoke(
    *,
    run_id: str,
    attempt_id: str,
    opponent_checkpoint_ref: str | None,
    snapshot_ref: str,
    checkpoint_ref: str | None,
    seed: int,
    num_simulations: int,
    batch_size: int,
    state_key: str | None,
    ego_action_id: int,
    fake_action_id: int,
) -> dict[str, Any]:
    from curvyzero.training.lightzero_checkpoint_opponent_wrapper_smoke import (
        run_lightzero_checkpoint_opponent_wrapper_smoke,
    )

    checkpoint_path: str | None = None
    checkpoint_resolution: dict[str, Any] | None = None
    report_checkpoint_ref = checkpoint_ref
    if opponent_checkpoint_ref:
        path, resolution = runs.resolve_mounted_ref_or_path(
            opponent_checkpoint_ref,
            mount=RUNS_MOUNT,
            remote_root=REMOTE_ROOT,
        )
        if not path.is_file():
            raise FileNotFoundError(f"opponent checkpoint file not found: {path}")
        checkpoint_path = str(path)
        checkpoint_resolution = {
            **resolution,
            "input": opponent_checkpoint_ref,
            "resolved_checkpoint_path": checkpoint_path,
            "file": runs.file_summary_any_mount(path, mount=RUNS_MOUNT),
        }
        if report_checkpoint_ref is None:
            report_checkpoint_ref = (
                str(resolution.get("source_ref"))
                if resolution.get("source_ref")
                else opponent_checkpoint_ref
            )

    started_at = runs.utc_timestamp()
    report = run_lightzero_checkpoint_opponent_wrapper_smoke(
        checkpoint_path=checkpoint_path,
        checkpoint_ref=report_checkpoint_ref,
        snapshot_ref=snapshot_ref,
        seed=seed,
        ego_action_id=ego_action_id,
        fake_action_id=fake_action_id,
        num_simulations=num_simulations,
        batch_size=batch_size,
        state_key=state_key,
    )
    summary = {
        "schema_id": "curvyzero_lightzero_checkpoint_opponent_modal_smoke/v0",
        "task_id": TASK_ID,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "mode": OPPONENT_SMOKE_MODE,
        "ok": bool(report.get("ok")),
        "started_at": started_at,
        "ended_at": runs.utc_timestamp(),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "trainer_wired": False,
        "training_claim": "none",
        "purpose": (
            "No-train bridge smoke: prove a snapshot-backed opponent provider can "
            "fill opponent actions through MetadataOnlyMultiplayerEgoWrapper using "
            "visual rows shaped [B,P,4,64,64]."
        ),
        "command": {
            "opponent_checkpoint_ref": opponent_checkpoint_ref,
            "snapshot_ref": snapshot_ref,
            "checkpoint_ref": checkpoint_ref,
            "seed": int(seed),
            "num_simulations": int(num_simulations),
            "batch_size": int(batch_size),
            "state_key": state_key,
            "ego_action_id": int(ego_action_id),
            "fake_action_id": int(fake_action_id),
        },
        "checkpoint_resolution": checkpoint_resolution,
        "smoke_report": report,
    }
    output_root = runs.volume_path(
        RUNS_MOUNT,
        runs.attempt_eval_ref(TASK_ID, run_id, attempt_id, "snapshot_opponent_wrapper_smoke"),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    runs.write_json(summary_path, _to_plain(summary))
    result = {
        "ok": summary["ok"],
        "mode": OPPONENT_SMOKE_MODE,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "summary_ref": runs.file_ref(summary_path, mount=RUNS_MOUNT),
        "trainer_wired": False,
        "checkpoint_resolution": checkpoint_resolution,
        "smoke_report": report,
    }
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


def _resolve_opponent_checkpoint_for_env(
    *,
    opponent_policy_kind: str,
    opponent_checkpoint_ref: str | None,
    opponent_checkpoint_report_ref: str | None,
) -> dict[str, Any] | None:
    if opponent_policy_kind == OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION:
        if opponent_checkpoint_ref:
            raise ValueError(
                "opponent_checkpoint_ref is not valid for centralized joint-action control"
            )
        return None
    if opponent_policy_kind == OPPONENT_POLICY_KIND_FIXED_STRAIGHT:
        if opponent_checkpoint_ref:
            raise ValueError(
                "opponent_checkpoint_ref is only valid with frozen_lightzero_checkpoint"
            )
        return None
    if opponent_policy_kind != OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT:
        raise ValueError(f"unknown opponent_policy_kind {opponent_policy_kind!r}")
    if not opponent_checkpoint_ref:
        raise ValueError(
            "opponent_checkpoint_ref is required with frozen_lightzero_checkpoint"
        )
    path, resolution = runs.resolve_mounted_ref_or_path(
        opponent_checkpoint_ref,
        mount=RUNS_MOUNT,
        remote_root=REMOTE_ROOT,
    )
    if not path.is_file():
        raise FileNotFoundError(f"opponent checkpoint file not found: {path}")
    report_ref = opponent_checkpoint_report_ref
    if report_ref is None:
        report_ref = (
            str(resolution.get("source_ref"))
            if resolution.get("source_ref")
            else opponent_checkpoint_ref
        )
    return {
        **resolution,
        "input": opponent_checkpoint_ref,
        "resolved_checkpoint_path": str(path),
        "checkpoint_ref": report_ref,
        "file": runs.file_summary_any_mount(path, mount=RUNS_MOUNT),
    }


@lru_cache(maxsize=1)
def _source_state_fixed_opponent_wrapper_env_spec_fields() -> dict[str, Any]:
    """Mirror the fixed-opponent wrapper's visual contract in Modal metadata."""

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
        "fixed_opponent_is_two_seat_self_play": bool(
            info["fixed_opponent_is_two_seat_self_play"]
        ),
        "browser_pixel_fidelity": bool(info["browser_pixel_fidelity"]),
        "uses_ale": bool(info["uses_ale"]),
        "visual_surface": str(info["visual_surface"]),
        "visual_truth_level": str(info["visual_truth_level"]),
        "visual_source_state_backed": bool(info["visual_source_state_backed"]),
    }


def _env_variant_spec(env_variant: str) -> dict[str, Any]:
    if env_variant == ENV_VARIANT_FIXED_OPPONENT:
        return {
            "env_variant": ENV_VARIANT_FIXED_OPPONENT,
            "env_type": LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_TYPE,
            "env_id": LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_ID,
            "import_names": list(LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_IMPORT_NAMES),
            "action_space_size": 3,
            "reward_schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
            "reward_policy": _survival_reward_policy(),
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
            "reward_policy": _survival_reward_policy(),
            "current_policy_self_play": TURN_COMMIT_CURRENT_POLICY_SELF_PLAY_CLAIM,
            "current_policy_self_play_blocker": None,
            "current_policy_self_play_caveat": TURN_COMMIT_REWARD_CREDIT_CAVEAT,
            "trusted_current_policy_self_play": TURN_COMMIT_TRUSTED_SELF_PLAY_CLAIM,
            "simultaneous_game_theory_claim": (
                TURN_COMMIT_SIMULTANEOUS_GAME_THEORY_CLAIM
            ),
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
            "reward_policy": _survival_reward_policy(),
            "current_policy_self_play": TURN_COMMIT_CURRENT_POLICY_SELF_PLAY_CLAIM,
            "current_policy_self_play_blocker": None,
            "current_policy_self_play_caveat": TURN_COMMIT_REWARD_CREDIT_CAVEAT,
            "trusted_current_policy_self_play": TURN_COMMIT_TRUSTED_SELF_PLAY_CLAIM,
            "simultaneous_game_theory_claim": (
                TURN_COMMIT_SIMULTANEOUS_GAME_THEORY_CLAIM
            ),
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
            "reward_policy": _all_players_alive_diagnostic_reward_policy(),
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
        wrapper_spec = _source_state_fixed_opponent_wrapper_env_spec_fields()
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


def _survival_reward_policy() -> dict[str, Any]:
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


def _all_players_alive_diagnostic_reward_policy() -> dict[str, Any]:
    return {
        "reward_schema_id": ALL_PLAYERS_ALIVE_DIAGNOSTIC_REWARD_SCHEMA_ID,
        "survival_only": False,
        "diagnostic_all_players_alive": True,
        "centralized_joint_action_control": True,
        "per_player_reward": False,
        "zero_sum_reward": False,
        "sparse_outcome_reward": False,
        "terminal_outcome_bonus": 0.0,
        "loser_penalty": 0.0,
        "winner_bonus": 0.0,
    }


def _build_visual_survival_configs(
    *,
    seed: int,
    exp_name: Path,
    telemetry_path: Path,
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
    ego_action_straight_override_probability: float,
    control_noise_profile_id: str,
    disable_death_for_profile: bool,
    env_telemetry_stride: int,
    env_manager_type: str,
    opponent_policy_kind: str,
    opponent_checkpoint: dict[str, Any] | None,
    opponent_snapshot_ref: str | None,
    opponent_checkpoint_state_key: str | None,
    natural_bonus_spawn: bool = TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN,
) -> dict[str, Any]:
    from easydict import EasyDict

    template_module = "zoo.atari.config.atari_muzero_config"
    module = importlib.import_module(template_module)
    env_spec = _env_variant_spec(env_variant)
    reward_variant = _normalize_reward_variant_for_env(
        env_variant=env_variant,
        reward_variant=reward_variant,
    )
    reward_policy = _reward_policy_for_variant(
        env_variant=env_variant,
        reward_variant=reward_variant,
    )
    target_config = _lightzero_target_config_for_reward(
        env_variant=env_variant,
        reward_variant=reward_variant,
        source_max_steps=source_max_steps,
    )
    action_space_size = int(env_spec.get("action_space_size", 3))
    main_config = copy.deepcopy(module.main_config)
    create_config = EasyDict(
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
        _set_or_add_path(main_config, ("exp_name",), str(exp_name)),
        _set_or_add_path(main_config, ("policy", "cuda"), bool(cuda)),
        _set_or_add_path(main_config, ("policy", "multi_gpu"), bool(lightzero_multi_gpu)),
        _set_or_add_path(main_config, ("policy", "env_type"), "not_board_games"),
        _set_or_add_path(main_config, ("policy", "collector_env_num"), int(collector_env_num)),
        _set_or_add_path(main_config, ("policy", "evaluator_env_num"), int(evaluator_env_num)),
        _set_or_add_path(main_config, ("policy", "n_episode"), int(n_episode)),
        _set_or_add_path(main_config, ("policy", "num_simulations"), int(num_simulations)),
        _set_or_add_path(main_config, ("policy", "batch_size"), int(batch_size)),
        _set_or_add_path(
            main_config,
            ("policy", "eval_freq"),
            int(lightzero_eval_freq)
            if int(lightzero_eval_freq) > 0
            else int(max_train_iter) + 1,
        ),
        _set_or_add_path(main_config, ("policy", "model", "model_type"), "conv"),
        _set_or_add_path(main_config, ("policy", "model", "image_channel"), 4),
        _set_or_add_path(main_config, ("policy", "model", "frame_stack_num"), 1),
        _set_or_add_path(
            main_config,
            ("policy", "model", "self_supervised_learning_loss"),
            True,
        ),
        _set_or_add_path(
            main_config,
            ("policy", "model", "observation_shape"),
            list(env_spec["observation_shape"]),
        ),
        _set_or_add_path(
            main_config,
            ("policy", "model", "action_space_size"),
            action_space_size,
        ),
        _set_save_ckpt_after_iter(main_config, int(save_ckpt_after_iter)),
    ]
    for patch in _target_config_patches(main_config, target_config):
        patches.append(patch)
    env_cfg = EasyDict(
        {
            **_to_plain(main_config["env"]),
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
            "frame_stack_num": 1,
            "observation_shape": list(env_spec["observation_shape"]),
            "gray_scale": True,
            "image_channel": 4,
            "continuous": False,
            "manually_discretization": False,
            "telemetry_path": str(telemetry_path),
            "telemetry_stride": int(env_telemetry_stride),
            "reward_variant": reward_variant,
            "reward_schema_id": reward_policy["reward_schema_id"],
            "reward_policy": reward_policy,
            "lightzero_target_config": target_config,
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
            "control_noise_profile_id": str(control_noise_profile_id),
            "disable_death_for_profile": bool(disable_death_for_profile),
            "natural_bonus_spawn": bool(natural_bonus_spawn),
            "death_mode": (
                "profile_no_death" if disable_death_for_profile else "normal"
            ),
            "turn_commit_adapter": bool(env_spec["turn_commit_adapter"]),
            "opponent_policy_kind": opponent_policy_kind,
            "opponent_training_relation": (
                env_spec["opponent_training_relation"]
                or _opponent_training_relation(opponent_policy_kind)
            ),
            "current_policy_self_play": env_spec["current_policy_self_play"],
            "current_policy_self_play_blocker": env_spec["current_policy_self_play_blocker"],
            "current_policy_self_play_caveat": env_spec["current_policy_self_play_caveat"],
            "trusted_current_policy_self_play": env_spec["trusted_current_policy_self_play"],
            "simultaneous_game_theory_claim": env_spec["simultaneous_game_theory_claim"],
        }
    )
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
                "opponent_use_cuda": bool(cuda),
            }
        )
    patches.append(_set_or_add_path(main_config, ("env",), env_cfg))
    surface = _extract_surface(
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


def _extract_surface(
    main_config: Any,
    create_config: Any,
    *,
    max_env_step: int,
    max_train_iter: int,
) -> dict[str, Any]:
    policy = main_config["policy"]
    model = policy["model"]
    env = main_config["env"]
    return {
        "env_type": create_config["env"]["type"],
        "env_import_names": _to_plain(create_config["env"].get("import_names")),
        "env_manager_type": create_config["env_manager"]["type"],
        "env_id": env["env_id"],
        "model_type": model["model_type"],
        "observation_shape": _to_plain(model["observation_shape"]),
        "env_observation_shape": _to_plain(env.get("observation_shape")),
        "action_space_size": model["action_space_size"],
        "model_image_channel": model.get("image_channel"),
        "model_frame_stack_num": model.get("frame_stack_num"),
        "model_self_supervised_learning_loss": model.get("self_supervised_learning_loss"),
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
        "load_ckpt_before_run": _get_path(
            policy,
            ("learn", "learner", "hook", "load_ckpt_before_run"),
        ),
        "frame_stack_num": env.get("frame_stack_num"),
        "source_max_steps": env.get("source_max_steps"),
        "decision_ms": env.get("decision_ms"),
        "dynamic_seed": env.get("dynamic_seed"),
        "reset_seed_strategy": env.get("reset_seed_strategy"),
        "telemetry_path": env.get("telemetry_path"),
        "telemetry_stride": env.get("telemetry_stride"),
        "reward_variant": env.get("reward_variant"),
        "reward_schema_id": env.get("reward_schema_id"),
        "reward_policy": _to_plain(env.get("reward_policy")),
        "lightzero_target_config": _to_plain(env.get("lightzero_target_config")),
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
        "fixed_opponent_is_two_seat_self_play": env.get(
            "fixed_opponent_is_two_seat_self_play"
        ),
        "browser_pixel_fidelity": env.get("browser_pixel_fidelity"),
        "uses_ale": env.get("uses_ale"),
        "visual_surface": env.get("visual_surface"),
        "visual_truth_level": env.get("visual_truth_level"),
        "visual_source_state_backed": env.get("visual_source_state_backed"),
        "ego_action_straight_override_probability": env.get(
            "ego_action_straight_override_probability"
        ),
        "control_noise_profile_id": env.get("control_noise_profile_id"),
        "disable_death_for_profile": env.get("disable_death_for_profile"),
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
        "save_ckpt_after_iter": _get_path(
            policy,
            ("learn", "learner", "hook", "save_ckpt_after_iter"),
        ),
        "max_env_step": int(max_env_step),
        "max_train_iter": int(max_train_iter),
    }


def _validate_visual_survival_surface(
    *,
    surface: dict[str, Any],
    command: dict[str, Any],
) -> list[str]:
    env_spec = _env_variant_spec(command["env_variant"])
    expected = {
        "env_type": command["env_type"],
        "env_import_names": env_spec["import_names"],
        "env_manager_type": command["env_manager_type"],
        "env_id": command["env_id"],
        "model_type": "conv",
        "observation_shape": list(env_spec["observation_shape"]),
        "env_observation_shape": list(env_spec["observation_shape"]),
        "action_space_size": int(env_spec.get("action_space_size", 3)),
        "model_image_channel": 4,
        "model_frame_stack_num": 1,
        "model_self_supervised_learning_loss": True,
        "collector_env_num": command["collector_env_num"],
        "policy_collector_env_num": command["collector_env_num"],
        "evaluator_env_num": command["evaluator_env_num"],
        "policy_evaluator_env_num": command["evaluator_env_num"],
        "n_evaluator_episode": command["n_evaluator_episode"],
        "n_episode": command["n_episode"],
        "num_simulations": command["num_simulations"],
        "batch_size": command["batch_size"],
        "discount_factor": command["lightzero_target_config"].get("discount_factor"),
        "td_steps": command["lightzero_target_config"].get("td_steps"),
        "model_support_scale": command["lightzero_target_config"].get(
            "model_support_scale"
        ),
        "model_reward_support_size": command["lightzero_target_config"].get(
            "model_reward_support_size"
        ),
        "model_value_support_size": command["lightzero_target_config"].get(
            "model_value_support_size"
        ),
        "frame_stack_num": 1,
        "source_max_steps": command["source_max_steps"],
        "decision_ms": command["decision_ms"],
        "dynamic_seed": True,
        "reset_seed_strategy": command.get("reset_seed_strategy"),
        "telemetry_stride": command["env_telemetry_stride"],
        "reward_variant": command["reward_variant"],
        "reward_schema_id": command["reward_schema_id"],
        "reward_policy": command["reward_policy"],
        "lightzero_target_config": command["lightzero_target_config"],
        "observation_schema_id": command["observation_schema_id"],
        "debug_fidelity_only": command["debug_fidelity_only"],
        "source_fidelity_claim": command["source_fidelity_claim"],
        "single_product_runtime_path": command["single_product_runtime_path"],
        "legacy_debug_variant": command["legacy_debug_variant"],
        "underlying_env_class": command["underlying_env_class"],
        "runtime_env_impl_id": command["runtime_env_impl_id"],
        "runtime_topology": command["runtime_topology"],
        "two_seat_self_play": command["two_seat_self_play"],
        "current_policy_two_seat_action_collection": command[
            "current_policy_two_seat_action_collection"
        ],
        "two_seat_self_play_status": command["two_seat_self_play_status"],
        "fixed_opponent_is_two_seat_self_play": command[
            "fixed_opponent_is_two_seat_self_play"
        ],
        "browser_pixel_fidelity": command["browser_pixel_fidelity"],
        "uses_ale": command["uses_ale"],
        "visual_surface": command["visual_surface"],
        "visual_truth_level": command["visual_truth_level"],
        "visual_source_state_backed": command["visual_source_state_backed"],
        "ego_action_straight_override_probability": command[
            "ego_action_straight_override_probability"
        ],
        "control_noise_profile_id": command["control_noise_profile_id"],
        "disable_death_for_profile": command["disable_death_for_profile"],
        "death_mode": command["death_mode"],
        "env_variant": command["env_variant"],
        "turn_commit_adapter": env_spec["turn_commit_adapter"],
        "opponent_policy_kind": command["opponent_policy_kind"],
        "opponent_training_relation": command["opponent_training_relation"],
        "current_policy_self_play": command["current_policy_self_play"],
        "current_policy_self_play_blocker": command["current_policy_self_play_blocker"],
        "current_policy_self_play_caveat": command["current_policy_self_play_caveat"],
        "trusted_current_policy_self_play": command["trusted_current_policy_self_play"],
        "simultaneous_game_theory_claim": command["simultaneous_game_theory_claim"],
        "save_ckpt_after_iter": command["save_ckpt_after_iter"],
        "max_env_step": command["max_env_step"],
        "max_train_iter": command["max_train_iter"],
    }
    if command["opponent_policy_kind"] == OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT:
        expected["opponent_checkpoint_ref"] = command["opponent_checkpoint_report_ref"]
        expected["opponent_snapshot_ref"] = command["opponent_snapshot_ref"]
        expected["opponent_checkpoint_state_key"] = command["opponent_checkpoint_state_key"]
    problems = []
    for key, value in expected.items():
        if surface.get(key) != value:
            problems.append(f"{key}={surface.get(key)!r}, expected {value!r}")
    return problems


def _source_state_fixed_opponent_readiness_expected() -> dict[str, Any]:
    spec = _env_variant_spec(ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT)
    return {
        "env_variant": ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        "single_product_runtime_path": spec["single_product_runtime_path"],
        "legacy_debug_variant": spec["legacy_debug_variant"],
        "debug_fidelity_only": spec["debug_fidelity_only"],
        "source_fidelity_claim": spec["source_fidelity_claim"],
        "underlying_env_class": spec["underlying_env_class"],
        "runtime_env_impl_id": spec["runtime_env_impl_id"],
        "runtime_topology": spec["runtime_topology"],
        "opponent_policy_kind": OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        "opponent_training_relation": spec["opponent_training_relation"],
        "current_policy_self_play": spec["current_policy_self_play"],
        "trusted_current_policy_self_play": spec["trusted_current_policy_self_play"],
        "simultaneous_game_theory_claim": spec["simultaneous_game_theory_claim"],
        "two_seat_self_play": spec["two_seat_self_play"],
        "two_seat_self_play_status": spec["two_seat_self_play_status"],
        "fixed_opponent_is_two_seat_self_play": spec[
            "fixed_opponent_is_two_seat_self_play"
        ],
        "browser_pixel_fidelity": spec["browser_pixel_fidelity"],
        "uses_ale": spec["uses_ale"],
        "visual_surface": spec["visual_surface"],
        "visual_truth_level": spec["visual_truth_level"],
        "visual_source_state_backed": spec["visual_source_state_backed"],
    }


def _source_state_fixed_opponent_training_readiness_gate(
    *,
    command: dict[str, Any],
    surface: dict[str, Any] | None = None,
    action_observability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = _source_state_fixed_opponent_readiness_expected()
    problems: list[str] = []
    checks: dict[str, dict[str, Any]] = {}
    scopes = {"command": command}
    if surface is not None:
        scopes["surface"] = surface
    for scope_name, scope in scopes.items():
        for key, expected_value in expected.items():
            if key not in scope:
                problems.append(f"{scope_name}.{key}=<missing>, expected {expected_value!r}")
                checks[f"{scope_name}.{key}"] = {
                    "ok": False,
                    "actual": None,
                    "expected": expected_value,
                    "missing": True,
                }
                continue
            actual = scope.get(key)
            ok = actual == expected_value
            checks[f"{scope_name}.{key}"] = {
                "ok": ok,
                "actual": actual,
                "expected": expected_value,
                "missing": False,
            }
            if not ok:
                problems.append(f"{scope_name}.{key}={actual!r}, expected {expected_value!r}")

    observed_fields = _action_observability_observed_fields(action_observability)
    return {
        "schema_id": "curvyzero_source_state_fixed_opponent_training_readiness_gate/v0",
        "gate_id": "source_state_fixed_opponent_local_metadata_v0",
        "ok": not problems,
        "status": "ready" if not problems else "blocked",
        "problems": problems,
        "expected": expected,
        "checks": checks,
        "action_observability": {
            "status": (action_observability or {}).get("status"),
            "row_count": (action_observability or {}).get("row_count"),
            "done_count": (action_observability or {}).get("done_count"),
            "observed_fields": observed_fields,
        },
    }


def _action_observability_observed_fields(
    action_observability: dict[str, Any] | None,
) -> dict[str, bool]:
    if not action_observability:
        return {
            "requested_ego_action": False,
            "executed_ego_action": False,
            "fixed_opponent_action": False,
            "joint_action": False,
            "action_mask": False,
            "terminal_reason": False,
            "death_cause": False,
        }
    observed = action_observability.get("observed_fields")
    if isinstance(observed, dict):
        return {
            "requested_ego_action": bool(observed.get("requested_ego_action", False)),
            "executed_ego_action": bool(observed.get("executed_ego_action", False)),
            "fixed_opponent_action": bool(observed.get("fixed_opponent_action", False)),
            "joint_action": bool(observed.get("joint_action", False)),
            "action_mask": bool(observed.get("action_mask", False)),
            "terminal_reason": bool(observed.get("terminal_reason", False)),
            "death_cause": bool(observed.get("death_cause", False)),
        }
    rows = action_observability.get("first_rows")
    if not isinstance(rows, list):
        rows = []
    return _observed_fields_from_telemetry_rows(rows)


def _compile_config_summary(main_config: Any, create_config: Any, *, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        from ding.config import compile_config

        compiled = compile_config(
            copy.deepcopy(main_config),
            seed=seed,
            auto=True,
            create_cfg=copy.deepcopy(create_config),
            save_cfg=False,
        )
        env_cfg = getattr(compiled, "env", {})
        policy_cfg = getattr(compiled, "policy", {})
        model_cfg = _cfg_get(policy_cfg, "model", {})
        return {
            "ok": True,
            "env": {
                "type": _cfg_get(env_cfg, "type", None),
                "env_id": _cfg_get(env_cfg, "env_id", None),
                "collector_env_num": _cfg_get(env_cfg, "collector_env_num", None),
                "evaluator_env_num": _cfg_get(env_cfg, "evaluator_env_num", None),
                "frame_stack_num": _cfg_get(env_cfg, "frame_stack_num", None),
                "reward_variant": _cfg_get(env_cfg, "reward_variant", None),
                "reward_schema_id": _cfg_get(env_cfg, "reward_schema_id", None),
                "reward_policy": _to_plain(_cfg_get(env_cfg, "reward_policy", None)),
                "lightzero_target_config": _to_plain(
                    _cfg_get(env_cfg, "lightzero_target_config", None)
                ),
            },
            "policy_targets": {
                "discount_factor": _cfg_get(policy_cfg, "discount_factor", None),
                "td_steps": _cfg_get(policy_cfg, "td_steps", None),
            },
            "policy_model": {
                "model_type": _cfg_get(model_cfg, "model_type", None),
                "observation_shape": _to_plain(_cfg_get(model_cfg, "observation_shape", None)),
                "image_channel": _cfg_get(model_cfg, "image_channel", None),
                "frame_stack_num": _cfg_get(model_cfg, "frame_stack_num", None),
                "self_supervised_learning_loss": _cfg_get(
                    model_cfg,
                    "self_supervised_learning_loss",
                    None,
                ),
                "action_space_size": _cfg_get(model_cfg, "action_space_size", None),
                "support_scale": _cfg_get(model_cfg, "support_scale", None),
                "reward_support_size": _cfg_get(model_cfg, "reward_support_size", None),
                "value_support_size": _cfg_get(model_cfg, "value_support_size", None),
            },
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
    except Exception as exc:  # pragma: no cover - installed runtime diagnosis.
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }


def _target_config_patches(
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
    }
    for key, path in key_paths.items():
        if key not in target_config:
            continue
        patch = _set_or_add_path(main_config, path, target_config[key])
        patch["reason"] = "make LightZero value/reward target range match CurvyTron reward variant"
        patches.append(patch)
    return patches


def _set_save_ckpt_after_iter(main_config: Any, value: int) -> dict[str, Any]:
    current = main_config["policy"]
    for part in ("learn", "learner", "hook"):
        if part not in current or current[part] is None:
            current[part] = {}
        current = current[part]
    old_value = current.get("save_ckpt_after_iter")
    current["save_ckpt_after_iter"] = int(value)
    return {
        "path": "policy.learn.learner.hook.save_ckpt_after_iter",
        "old": _to_plain(old_value),
        "new": int(value),
        "reason": "first CurvyTron visual survival run checkpoints frequently for inspection",
    }


def _set_load_ckpt_before_run(main_config: Any, checkpoint_path: str) -> dict[str, Any]:
    current = main_config["policy"]
    for part in ("learn", "learner", "hook"):
        if part not in current or current[part] is None:
            current[part] = {}
        current = current[part]
    old_value = current.get("load_ckpt_before_run")
    current["load_ckpt_before_run"] = checkpoint_path
    return {
        "path": "policy.learn.learner.hook.load_ckpt_before_run",
        "old": _to_plain(old_value),
        "new": checkpoint_path,
        "reason": "automatic resume from the latest iteration checkpoint for this run",
    }


def _prepare_lightzero_auto_resume(
    *,
    run_id: str,
    attempt_id: str,
    exp_name_ref: Any,
) -> dict[str, Any]:
    """Find the newest LightZero iteration checkpoint already saved for this run."""

    if hasattr(runs_volume, "reload"):
        try:
            runs_volume.reload()
        except Exception:
            pass

    run_root = runs.volume_path(RUNS_MOUNT, runs.run_root_ref(TASK_ID, run_id))
    current_ckpt_dir = runs.volume_path(RUNS_MOUNT, exp_name_ref) / "ckpt"
    stable_ckpt_dir = (
        runs.volume_path(RUNS_MOUNT, runs.checkpoints_root_ref(TASK_ID, run_id))
        / "lightzero"
    )
    candidates: list[dict[str, Any]] = []
    source_roots: list[dict[str, Any]] = []

    def scan_dir(directory: Path, *, source_kind: str) -> None:
        source_roots.append(
            {
                "source_kind": source_kind,
                "path": str(directory),
                "ref": (
                    runs.file_ref(directory, mount=RUNS_MOUNT)
                    if _is_under_run_mount(directory)
                    else None
                ),
                "exists": directory.exists(),
            }
        )
        if not directory.is_dir():
            return
        for path in directory.iterdir():
            iteration = _lightzero_iteration_from_checkpoint_name(path.name)
            if iteration is None or not path.is_file():
                continue
            stat = path.stat()
            if stat.st_size <= 0:
                continue
            candidates.append(
                {
                    "found": True,
                    "source_kind": source_kind,
                    "checkpoint_path": str(path),
                    "checkpoint_ref": runs.file_ref(path, mount=RUNS_MOUNT),
                    "name": path.name,
                    "iteration": iteration,
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )

    scan_dir(current_ckpt_dir, source_kind="current_attempt_lightzero_exp")

    attempts_dir = run_root / "attempts"
    if attempts_dir.is_dir():
        for attempt_dir in sorted(attempts_dir.iterdir(), key=lambda path: path.name):
            if not attempt_dir.is_dir():
                continue
            ckpt_dir = attempt_dir / "train" / "lightzero_exp" / "ckpt"
            if ckpt_dir == current_ckpt_dir:
                continue
            source_kind = (
                "same_attempt_lightzero_exp"
                if attempt_dir.name == attempt_id
                else f"prior_attempt_lightzero_exp:{attempt_dir.name}"
            )
            scan_dir(ckpt_dir, source_kind=source_kind)

    scan_dir(stable_ckpt_dir, source_kind="run_checkpoint_mirror")

    base = {
        "schema_id": "curvyzero_lightzero_auto_resume/v0",
        "enabled": True,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "selection_policy": (
            "highest numbered iteration_*.pth.tar checkpoint found in this run; "
            "ckpt_best is ignored for resume"
        ),
        "source_roots": source_roots,
        "candidate_count": len(candidates),
        "state_scope": (
            "LightZero/DI-engine loads learner last_iter/last_step when present "
            "and the MuZero policy state, including model, target model, and optimizer. "
            "A matching CurvyZero sidecar extends this with collector progress, evaluator "
            "progress, policy helper state when visible, and random generator state. Replay "
            "GameSegments are not restored yet; live environment manager internals are also "
            "not serialized."
        ),
    }
    if not candidates:
        return {**base, "found": False}

    latest = max(
        candidates,
        key=lambda item: (
            int(item["iteration"]),
            int(item["mtime_ns"]),
            int(item["size_bytes"]),
            str(item["checkpoint_ref"]),
        ),
    )
    resume_state = _find_lightzero_resume_sidecar(
        run_id=run_id,
        attempt_id=attempt_id,
        exp_name_ref=exp_name_ref,
        iteration=int(latest["iteration"]),
    )
    return {
        **base,
        **latest,
        **resume_state,
        "found": True,
        "candidate_iterations": sorted({int(item["iteration"]) for item in candidates}),
    }


def _find_lightzero_resume_sidecar(
    *,
    run_id: str,
    attempt_id: str,
    exp_name_ref: Any,
    iteration: int,
) -> dict[str, Any]:
    """Find the sidecar state for one exact LightZero iteration checkpoint."""

    run_root = runs.volume_path(RUNS_MOUNT, runs.run_root_ref(TASK_ID, run_id))
    current_state_dir = runs.volume_path(RUNS_MOUNT, exp_name_ref) / LIGHTZERO_RESUME_STATE_DIRNAME
    stable_state_dir = (
        runs.volume_path(RUNS_MOUNT, runs.checkpoints_root_ref(TASK_ID, run_id))
        / LIGHTZERO_RESUME_STATE_DIRNAME
    )
    sidecar_name = _lightzero_resume_state_name(iteration)
    source_roots: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    def scan_dir(directory: Path, *, source_kind: str) -> None:
        source_roots.append(
            {
                "source_kind": source_kind,
                "path": str(directory),
                "ref": (
                    runs.file_ref(directory, mount=RUNS_MOUNT)
                    if _is_under_run_mount(directory)
                    else None
                ),
                "exists": directory.exists(),
            }
        )
        path = directory / sidecar_name
        if not path.is_file():
            return
        stat = path.stat()
        if stat.st_size <= 0:
            return
        candidates.append(
            {
                "resume_state_found": True,
                "resume_state_source_kind": source_kind,
                "resume_state_path": str(path),
                "resume_state_ref": runs.file_ref(path, mount=RUNS_MOUNT),
                "resume_state_name": path.name,
                "resume_state_iteration": int(iteration),
                "resume_state_size_bytes": int(stat.st_size),
                "resume_state_mtime_ns": int(stat.st_mtime_ns),
            }
        )

    scan_dir(current_state_dir, source_kind="current_attempt_lightzero_exp")

    attempts_dir = run_root / "attempts"
    if attempts_dir.is_dir():
        for attempt_dir in sorted(attempts_dir.iterdir(), key=lambda path: path.name):
            if not attempt_dir.is_dir():
                continue
            state_dir = attempt_dir / "train" / "lightzero_exp" / LIGHTZERO_RESUME_STATE_DIRNAME
            if state_dir == current_state_dir:
                continue
            source_kind = (
                "same_attempt_lightzero_exp"
                if attempt_dir.name == attempt_id
                else f"prior_attempt_lightzero_exp:{attempt_dir.name}"
            )
            scan_dir(state_dir, source_kind=source_kind)

    scan_dir(stable_state_dir, source_kind="run_resume_state_mirror")

    base = {
        "resume_state_lookup": {
            "iteration": int(iteration),
            "name": sidecar_name,
            "source_roots": source_roots,
            "candidate_count": len(candidates),
        },
        "resume_state_found": False,
    }
    if not candidates:
        return base
    latest = max(
        candidates,
        key=lambda item: (
            int(item["resume_state_mtime_ns"]),
            int(item["resume_state_size_bytes"]),
            str(item["resume_state_ref"]),
        ),
    )
    return {**base, **latest}


def _is_under_run_mount(path: Path) -> bool:
    try:
        path.relative_to(RUNS_MOUNT)
    except ValueError:
        return False
    return True


def _lightzero_iteration_from_checkpoint_name(name: str) -> int | None:
    prefix = "iteration_"
    suffix = ".pth.tar"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    middle = name[len(prefix) : -len(suffix)]
    if not middle.isdigit():
        return None
    return int(middle)


def _lightzero_iteration_from_resume_state_name(name: str) -> int | None:
    prefix = "iteration_"
    suffix = ".resume_state.pkl"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    middle = name[len(prefix) : -len(suffix)]
    if not middle.isdigit():
        return None
    return int(middle)


def _opponent_training_relation(opponent_policy_kind: str) -> str:
    if opponent_policy_kind == OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION:
        return "centralized_policy_controls_both_players"
    if opponent_policy_kind == OPPONENT_POLICY_KIND_FIXED_STRAIGHT:
        return OPPONENT_TRAINING_RELATION_FIXED_STRAIGHT
    if opponent_policy_kind == OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT:
        return OPPONENT_TRAINING_RELATION_FROZEN_LIGHTZERO_CHECKPOINT
    return f"unknown:{opponent_policy_kind}"


def _parse_training_signals(stdout_text: str, stderr_text: str) -> dict[str, Any]:
    text = stdout_text + "\n" + stderr_text
    checkpoint_saves = re.findall(r"learner save ckpt in\s+([^\n]+)", text)
    checkpoint_iterations = sorted(
        {int(value) for value in re.findall(r"iteration_(\d+)\.pth\.tar", "\n".join(checkpoint_saves))}
    )
    training_iterations = [int(value) for value in re.findall(r"Training Iteration\s+(\d+)", text)]
    final_rewards = [float(value) for value in re.findall(r"final reward:\s*([-+]?\d+(?:\.\d+)?)", text)]
    return {
        "training_iterations": training_iterations,
        "checkpoint_iterations": checkpoint_iterations,
        "max_checkpoint_iteration": max(checkpoint_iterations) if checkpoint_iterations else None,
        "checkpoint_saves": checkpoint_saves[-20:],
        "final_rewards": final_rewards,
        "stdout_line_count": len(stdout_text.splitlines()),
        "stderr_line_count": len(stderr_text.splitlines()),
        "stdout_tail": _compact_log_tail(stdout_text),
        "stderr_tail": _compact_log_tail(stderr_text, limit=30),
    }


def _scan_lightzero_artifacts(exp_name: str) -> dict[str, Any]:
    root = Path(exp_name)
    roots = [root]
    if exp_name.startswith("/"):
        roots.append(Path("." + exp_name))
    files: list[dict[str, Any]] = []
    for candidate in roots:
        if not candidate.exists():
            continue
        for path in sorted(candidate.rglob("*")):
            if path.is_file():
                stat = path.stat()
                files.append(
                    {
                        "root": str(candidate),
                        "path": str(path),
                        "relative_path": str(path.relative_to(candidate)),
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
    checkpoint_files = [
        item for item in files if item["relative_path"].endswith((".pth.tar", ".pt", ".pth"))
    ]
    log_files = [
        item
        for item in files
        if item["relative_path"].endswith((".txt", ".log", ".json", ".jsonl"))
        or "events.out.tfevents" in item["relative_path"]
    ]
    return {
        "exists": bool(files),
        "file_count": len(files),
        "files_sample": files[:40],
        "checkpoint_files": checkpoint_files,
        "log_files": log_files[:40],
    }


def _mirror_lightzero_checkpoints(
    *,
    run_id: str,
    artifact_summary: dict[str, Any],
    include_hashes: bool = True,
) -> dict[str, Any]:
    copied: list[dict[str, Any]] = []
    checkpoints = artifact_summary.get("checkpoint_files", [])
    for item in checkpoints:
        source = Path(str(item.get("path", "")))
        if not source.exists() or not source.is_file():
            continue
        if _lightzero_iteration_from_resume_state_name(source.name) is not None:
            continue
        dest = runs.volume_path(RUNS_MOUNT, runs.checkpoints_root_ref(TASK_ID, run_id))
        dest = dest / "lightzero" / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        source_stat = source.stat()
        should_copy = True
        if dest.exists():
            dest_stat = dest.stat()
            should_copy = (
                dest_stat.st_size != source_stat.st_size
                or dest_stat.st_mtime_ns < source_stat.st_mtime_ns
            )
        if should_copy:
            shutil.copy2(source, dest)
        if include_hashes:
            summary = runs.file_summary(dest, mount=RUNS_MOUNT)
        else:
            summary = {
                "ref": runs.file_ref(dest, mount=RUNS_MOUNT),
                "path": str(dest),
                "bytes": dest.stat().st_size,
            }
        summary["source_path"] = str(source)
        summary["copied_now"] = should_copy
        copied.append(summary)
    return {
        "copied_checkpoints": copied,
        "count": len(copied),
        "copied_now_count": sum(1 for item in copied if item.get("copied_now")),
    }


def _publish_live_lightzero_checkpoints(
    *,
    run_id: str,
    attempt_id: str,
    exp_name: Path,
    attempt_train_root: Path,
) -> dict[str, Any]:
    artifact_summary = _scan_lightzero_artifacts(str(exp_name))
    checkpoint_mirror = _mirror_lightzero_checkpoints(
        run_id=run_id,
        artifact_summary=artifact_summary,
        include_hashes=False,
    )
    payload = {
        "schema_id": "curvyzero_lightzero_curvytron_live_checkpoint_publish/v0",
        "task_id": TASK_ID,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "published_at": runs.utc_timestamp(),
        "exp_name": str(exp_name),
        "checkpoint_mirror": checkpoint_mirror,
        "latest_checkpoint": (
            checkpoint_mirror["copied_checkpoints"][-1]
            if checkpoint_mirror.get("copied_checkpoints")
            else None
        ),
    }
    live_path = attempt_train_root / "live_checkpoint_publish.json"
    runs.write_json(live_path, _to_plain(payload))
    return payload


def _background_eval_config_from_command(command: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(
            command.get("background_eval_enabled", DEFAULT_BACKGROUND_EVAL_ENABLED)
        ),
        "compute": str(command.get("background_eval_compute", DEFAULT_BACKGROUND_EVAL_COMPUTE)),
        "eval_id_prefix": str(
            command.get("background_eval_id_prefix", DEFAULT_BACKGROUND_EVAL_ID_PREFIX)
        ),
        "seed": int(command.get("seed", DEFAULT_SEED)),
        "eval_seed_count": int(
            command.get("background_eval_seed_count", DEFAULT_BACKGROUND_EVAL_SEED_COUNT)
        ),
        "eval_seed_rng_seed": command.get(
            "background_eval_seed_rng_seed",
            DEFAULT_BACKGROUND_EVAL_SEED_RNG_SEED,
        ),
        "max_eval_steps": int(
            command.get("background_eval_max_steps", DEFAULT_BACKGROUND_EVAL_MAX_STEPS)
        ),
        "step_detail_limit": command.get(
            "background_eval_step_detail_limit",
            DEFAULT_BACKGROUND_EVAL_STEP_DETAIL_LIMIT,
        ),
        "source_max_steps": int(command.get("source_max_steps", DEFAULT_SOURCE_MAX_STEPS)),
        "natural_bonus_spawn": bool(
            command.get("natural_bonus_spawn", TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN)
        ),
        "num_simulations": int(
            command.get(
                "background_eval_num_simulations",
                DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS,
            )
        ),
        "batch_size": int(
            command.get("background_eval_batch_size", DEFAULT_BACKGROUND_EVAL_BATCH_SIZE)
        ),
        "env_variant": str(command.get("env_variant", DEFAULT_ENV_VARIANT)),
        "reward_variant": str(command.get("eval_reward_variant", DEFAULT_REWARD_VARIANT)),
        "eval_reward_variant": str(
            command.get("eval_reward_variant", DEFAULT_REWARD_VARIANT)
        ),
        "training_reward_variant": str(
            command.get(
                "training_reward_variant",
                command.get("reward_variant", DEFAULT_REWARD_VARIANT),
            )
        ),
        "model_reward_variant": str(
            command.get(
                "model_reward_variant",
                command.get("reward_variant", DEFAULT_REWARD_VARIANT),
            )
        ),
        "opponent_policy_kind": str(
            command.get("opponent_policy_kind", DEFAULT_OPPONENT_POLICY_KIND)
        ),
        "opponent_checkpoint_ref": command.get("opponent_checkpoint_ref"),
        "opponent_snapshot_ref": command.get("opponent_snapshot_ref"),
        "opponent_checkpoint_state_key": command.get("opponent_checkpoint_state_key"),
        "selfplay_gif": _background_gif_config_from_command(command),
    }


def _background_gif_config_from_command(command: dict[str, Any]) -> dict[str, Any]:
    max_steps = _normalize_background_gif_max_steps(
        command.get("background_gif_max_steps", DEFAULT_BACKGROUND_GIF_MAX_STEPS)
    )
    return {
        "enabled": bool(command.get("background_gif_enabled", DEFAULT_BACKGROUND_GIF_ENABLED)),
        "seed": int(command.get("seed", DEFAULT_SEED))
        + int(command.get("background_gif_seed_offset", DEFAULT_BACKGROUND_GIF_SEED_OFFSET)),
        "checkpoint_seed_mixing_enabled": bool(
            command.get(
                "background_gif_checkpoint_seed_mixing_enabled",
                DEFAULT_BACKGROUND_GIF_CHECKPOINT_SEED_MIXING_ENABLED,
            )
        ),
        "max_steps": max_steps,
        "step_limit_kind": _background_gif_step_limit_kind(max_steps),
        "frame_stride": int(
            command.get("background_gif_frame_stride", DEFAULT_BACKGROUND_GIF_FRAME_STRIDE)
        ),
        "fps": float(command.get("background_gif_fps", DEFAULT_BACKGROUND_GIF_FPS)),
        "scale": int(command.get("background_gif_scale", DEFAULT_BACKGROUND_GIF_SCALE)),
        "frame_size": int(
            command.get("background_gif_frame_size", DEFAULT_BACKGROUND_GIF_FRAME_SIZE)
        ),
        "natural_bonus_spawn": bool(
            command.get("natural_bonus_spawn", TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN)
        ),
        "source_max_steps": int(command.get("source_max_steps", DEFAULT_SOURCE_MAX_STEPS)),
        "num_simulations": int(
            command.get(
                "background_eval_num_simulations",
                DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS,
            )
        ),
        "batch_size": int(
            command.get("background_eval_batch_size", DEFAULT_BACKGROUND_EVAL_BATCH_SIZE)
        ),
        "training_env_variant": str(command.get("env_variant", DEFAULT_ENV_VARIANT)),
        "training_reward_variant": str(
            command.get(
                "training_reward_variant",
                command.get("reward_variant", DEFAULT_REWARD_VARIANT),
            )
        ),
    }


def _stable_seed_mix(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _mix_seed(base_seed: int, seed_mix: int) -> int:
    return int((int(base_seed) + int(seed_mix)) % (2**31 - 1))


def _spawn_checkpoint_eval_triggers(
    *,
    run_id: str,
    attempt_id: str,
    exp_name: Path,
    config: dict[str, Any],
    seen_checkpoint_refs: set[str],
) -> list[dict[str, Any]]:
    artifact_summary = _scan_lightzero_artifacts(str(exp_name))
    triggered: list[dict[str, Any]] = []
    checkpoint_files = artifact_summary.get("checkpoint_files", [])
    if not isinstance(checkpoint_files, list):
        return triggered
    for index, item in enumerate(checkpoint_files):
        if not isinstance(item, dict):
            continue
        source = Path(str(item.get("path", "")))
        if not source.name:
            continue
        if not _live_eval_checkpoint_name(source.name):
            continue
        checkpoint_ref = _checkpoint_source_ref(source)
        if checkpoint_ref in seen_checkpoint_refs:
            continue
        checkpoint = {
            "ref": checkpoint_ref,
            "canonical_ref": (
                runs.checkpoints_root_ref(TASK_ID, run_id) / "lightzero" / source.name
            ).as_posix(),
            "source_path": str(source),
            "copied_now": True,
        }
        triggered.append(
            _schedule_one_checkpoint_background_eval(
                publish={"run_id": run_id, "attempt_id": attempt_id},
                checkpoint=checkpoint,
                checkpoint_index=index,
                config=config,
            )
        )
        if triggered[-1].get("scheduled"):
            seen_checkpoint_refs.add(checkpoint_ref)
    return triggered


def _checkpoint_source_ref(path: Path) -> str:
    """Return the Modal Volume ref for a LightZero checkpoint source path."""

    if path.is_absolute():
        return runs.file_ref(path.resolve(), mount=RUNS_MOUNT.resolve())
    return runs.require_relative_ref(path.as_posix()).as_posix()


def _schedule_live_checkpoint_background_eval(
    *,
    publish: dict[str, Any],
    attempt_train_root: Path,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    _ = attempt_train_root
    if not config or not config.get("enabled"):
        return {"scheduled": False, "reason": "background_eval_disabled"}

    copied = _copied_now_checkpoints(publish)
    if not copied:
        return {"scheduled": False, "reason": "no_new_checkpoints"}

    scheduled: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(copied):
        request = _schedule_one_checkpoint_background_eval(
            publish=publish,
            checkpoint=checkpoint,
            checkpoint_index=index,
            config=config,
        )
        scheduled.append(request)
    return {"scheduled": bool(scheduled), "requests": scheduled}


def _copied_now_checkpoints(publish: dict[str, Any]) -> list[dict[str, Any]]:
    mirror = publish.get("checkpoint_mirror")
    copied = mirror.get("copied_checkpoints") if isinstance(mirror, dict) else []
    if not isinstance(copied, list):
        return []
    checkpoints = [
        item for item in copied if isinstance(item, dict) and item.get("copied_now")
    ]
    return sorted(
        checkpoints,
        key=lambda item: _checkpoint_ref_sort_key(str(item.get("ref") or item.get("path") or "")),
    )


def _checkpoint_ref_sort_key(ref: str) -> tuple[int, int | str]:
    match = re.search(r"iteration_(\d+)", ref)
    if match is not None:
        return (0, int(match.group(1)))
    return (1, ref)


def _checkpoint_label_from_ref(ref: str, *, index: int) -> str:
    name = Path(ref).name
    for suffix in (".pth.tar", ".tar", ".pth", ".pt", ".bin"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if not name:
        name = f"checkpoint_{index:03d}"
    return _safe_generated_id(name, fallback=f"checkpoint_{index:03d}")


def _safe_generated_id(raw: str, *, fallback: str) -> str:
    cleaned = "".join(char if char in runs.SAFE_ID_CHARS else "_" for char in raw).strip("._-")
    if not cleaned:
        cleaned = fallback
    if not cleaned[0].isalnum():
        cleaned = f"{fallback}_{cleaned}"
    return runs.clean_id(cleaned[:80], label=fallback)


def _schedule_one_checkpoint_background_eval(
    *,
    publish: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_index: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    request, _jobs = _spawn_one_checkpoint_background_eval(
        publish=publish,
        checkpoint=checkpoint,
        checkpoint_index=checkpoint_index,
        config=config,
    )
    return request


def _spawn_one_checkpoint_background_eval(
    *,
    publish: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_index: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checkpoint_ref = str(checkpoint.get("ref") or "")
    if not checkpoint_ref:
        return {"scheduled": False, "reason": "checkpoint_ref_missing"}, []

    checkpoint_label = _checkpoint_label_from_ref(checkpoint_ref, index=checkpoint_index)
    eval_id = _safe_generated_id(
        f"{config['eval_id_prefix']}_{checkpoint_label}",
        fallback="live_checkpoint",
    )
    request: dict[str, Any] = {
        "schema_id": "curvyzero_lightzero_curvytron_visible_checkpoint_background_eval_spawn/v0",
        "scheduled": False,
        "status": "pending_spawn",
        "task_id": TASK_ID,
        "run_id": publish.get("run_id"),
        "attempt_id": publish.get("attempt_id"),
        "checkpoint": checkpoint,
        "checkpoint_ref": checkpoint_ref,
        "checkpoint_label": checkpoint_label,
        "eval_id": eval_id,
        "requested_at": runs.utc_timestamp(),
        "config": config,
    }
    jobs: list[dict[str, Any]] = []
    try:
        call = lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect.spawn(
            checkpoint_ref=checkpoint_ref,
            checkpoint_label=checkpoint_label,
            eval_id=eval_id,
            run_id=str(publish.get("run_id")),
            attempt_id=str(publish.get("attempt_id")),
            compute=str(config["compute"]),
            seed=int(config["seed"]),
            eval_seed_count=int(config["eval_seed_count"]),
            eval_seed_rng_seed=config.get("eval_seed_rng_seed"),
            max_eval_steps=int(config["max_eval_steps"]),
            step_detail_limit=config.get("step_detail_limit"),
            source_max_steps=int(config["source_max_steps"]),
            num_simulations=int(config["num_simulations"]),
            batch_size=int(config["batch_size"]),
            env_variant=str(config["env_variant"]),
            reward_variant=str(config.get("eval_reward_variant", DEFAULT_REWARD_VARIANT)),
            model_reward_variant=str(
                config.get(
                    "model_reward_variant",
                    config.get("training_reward_variant", DEFAULT_REWARD_VARIANT),
                )
            ),
            opponent_policy_kind=str(config["opponent_policy_kind"]),
            opponent_checkpoint_ref=config.get("opponent_checkpoint_ref"),
            opponent_snapshot_ref=config.get("opponent_snapshot_ref"),
            opponent_checkpoint_state_key=config.get("opponent_checkpoint_state_key"),
            natural_bonus_spawn=bool(
                config.get("natural_bonus_spawn", TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN)
            ),
        )
        request["scheduled"] = True
        request["eval_inspection_scheduled"] = True
        request["status"] = "spawned"
        request["spawned_at"] = runs.utc_timestamp()
        request["function_call_id"] = getattr(call, "object_id", None) or getattr(call, "id", None)
        jobs.append(
            {
                "kind": "eval_inspection",
                "eval_id": eval_id,
                "call": call,
            }
        )
    except Exception as exc:  # pragma: no cover - remote scheduling resilience only.
        request["scheduled"] = False
        request["eval_inspection_scheduled"] = False
        request["status"] = "spawn_failed"
        request["spawn_error"] = _exception_result(exc)
    gif_request, gif_call = _spawn_one_checkpoint_background_gif(
        publish=publish,
        checkpoint_ref=checkpoint_ref,
        checkpoint_label=checkpoint_label,
        eval_id=eval_id,
        config=config,
    )
    request["selfplay_gif"] = gif_request
    if gif_request.get("scheduled"):
        jobs.append(
            {
                "kind": "selfplay_gif",
                "eval_id": eval_id,
                "call": gif_call,
            }
        )
    if jobs:
        request["scheduled"] = True
        request["status"] = "spawned"
    else:
        request["scheduled"] = False
        request["status"] = "spawn_failed"

    return _to_plain(request), jobs


def _spawn_one_checkpoint_background_gif(
    *,
    publish: dict[str, Any],
    checkpoint_ref: str,
    checkpoint_label: str,
    eval_id: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Any | None]:
    gif_config = dict(config.get("selfplay_gif") or {})
    if not gif_config.get("enabled", DEFAULT_BACKGROUND_GIF_ENABLED):
        return {"scheduled": False, "reason": "background_gif_disabled"}, None
    training_env_variant = str(gif_config.get("training_env_variant", DEFAULT_ENV_VARIANT))
    if training_env_variant == ENV_VARIANT_SOURCE_STATE_JOINT_ACTION:
        return {
            "scheduled": False,
            "reason": "background_gif_unsupported_for_source_state_joint_action",
            "training_env_variant": training_env_variant,
        }, None
    natural_bonus_spawn = bool(
        gif_config.get(
            "natural_bonus_spawn",
            config.get("natural_bonus_spawn", TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN),
        )
    )
    gif_config["natural_bonus_spawn"] = natural_bonus_spawn
    base_seed = int(gif_config.get("seed", DEFAULT_SEED + DEFAULT_BACKGROUND_GIF_SEED_OFFSET))
    checkpoint_seed_mix = _stable_seed_mix(checkpoint_ref, checkpoint_label, eval_id)
    checkpoint_seed_mixing_enabled = bool(
        gif_config.get(
            "checkpoint_seed_mixing_enabled",
            DEFAULT_BACKGROUND_GIF_CHECKPOINT_SEED_MIXING_ENABLED,
        )
    )
    effective_seed = (
        _mix_seed(base_seed, checkpoint_seed_mix)
        if checkpoint_seed_mixing_enabled
        else base_seed
    )
    gif_config["base_seed"] = base_seed
    gif_config["checkpoint_seed_mix"] = checkpoint_seed_mix
    gif_config["checkpoint_seed_mixing_enabled"] = checkpoint_seed_mixing_enabled
    gif_config["effective_seed"] = effective_seed
    request: dict[str, Any] = {
        "schema_id": "curvyzero_lightzero_curvytron_selfplay_gif_spawn/v0",
        "scheduled": False,
        "status": "pending_spawn",
        "task_id": TASK_ID,
        "run_id": publish.get("run_id"),
        "attempt_id": publish.get("attempt_id"),
        "checkpoint_ref": checkpoint_ref,
        "checkpoint_label": checkpoint_label,
        "eval_id": eval_id,
        "requested_at": runs.utc_timestamp(),
        "config": gif_config,
    }
    try:
        call = lightzero_curvytron_visual_survival_checkpoint_selfplay_gif.spawn(
            checkpoint_ref=checkpoint_ref,
            checkpoint_label=checkpoint_label,
            eval_id=eval_id,
            run_id=str(publish.get("run_id")),
            attempt_id=str(publish.get("attempt_id")),
            seed=effective_seed,
            max_steps=_background_gif_max_steps_arg(
                gif_config.get("max_steps", DEFAULT_BACKGROUND_GIF_MAX_STEPS)
            ),
            source_max_steps=int(gif_config.get("source_max_steps", DEFAULT_SOURCE_MAX_STEPS)),
            num_simulations=int(
                gif_config.get(
                    "num_simulations",
                    DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS,
                )
            ),
            batch_size=int(gif_config.get("batch_size", DEFAULT_BACKGROUND_EVAL_BATCH_SIZE)),
            frame_stride=int(gif_config.get("frame_stride", DEFAULT_BACKGROUND_GIF_FRAME_STRIDE)),
            fps=float(gif_config.get("fps", DEFAULT_BACKGROUND_GIF_FPS)),
            scale=int(gif_config.get("scale", DEFAULT_BACKGROUND_GIF_SCALE)),
            frame_size=int(gif_config.get("frame_size", DEFAULT_BACKGROUND_GIF_FRAME_SIZE)),
            training_env_variant=training_env_variant,
            training_reward_variant=str(
                gif_config.get("training_reward_variant", DEFAULT_REWARD_VARIANT)
            ),
            natural_bonus_spawn=natural_bonus_spawn,
        )
        request["scheduled"] = True
        request["status"] = "spawned"
        request["spawned_at"] = runs.utc_timestamp()
        request["function_call_id"] = getattr(call, "object_id", None) or getattr(call, "id", None)
    except Exception as exc:  # pragma: no cover - remote scheduling resilience only.
        request["scheduled"] = False
        request["status"] = "spawn_failed"
        request["spawn_error"] = _exception_result(exc)
        call = None
    return _to_plain(request), call


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _live_eval_checkpoint_name(name: str) -> bool:
    return bool(re.fullmatch(r"iteration_\d+\.(?:pth\.tar|pth|pt)", name))


def _checkpoint_eval_poller_status_path(*, run_id: str, attempt_id: str) -> Path:
    return runs.volume_path(
        RUNS_MOUNT,
        runs.attempt_train_ref(TASK_ID, run_id, attempt_id) / "checkpoint_eval_poller.json",
    )


def _write_checkpoint_eval_poller_status(
    *,
    run_id: str,
    attempt_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    path = _checkpoint_eval_poller_status_path(run_id=run_id, attempt_id=attempt_id)
    runs.write_json(path, _to_plain(payload))
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _checkpoint_eval_poller_train_done(*, run_id: str, attempt_id: str) -> bool:
    summary_path = runs.volume_path(
        RUNS_MOUNT,
        runs.attempt_train_ref(TASK_ID, run_id, attempt_id) / "summary.json",
    )
    if summary_path.exists():
        return True
    attempt_path = runs.volume_path(
        RUNS_MOUNT,
        runs.attempt_manifest_ref(TASK_ID, run_id, attempt_id),
    )
    attempt = _read_json_if_exists(attempt_path)
    return bool(
        isinstance(attempt, dict)
        and attempt.get("status") in {"completed", "failed", "superseded"}
    )


def _checkpoint_eval_poller_command(
    *,
    seed: int,
    source_max_steps: int,
    env_variant: str,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    opponent_policy_kind: str,
    opponent_checkpoint_ref: str | None,
    opponent_snapshot_ref: str | None,
    opponent_checkpoint_state_key: str | None,
    background_eval_enabled: bool,
    background_eval_compute: str,
    background_eval_id_prefix: str,
    background_eval_seed_count: int,
    background_eval_seed_rng_seed: int | None,
    background_eval_max_steps: int,
    background_eval_step_detail_limit: int | None,
    background_eval_num_simulations: int,
    background_eval_batch_size: int,
    background_gif_enabled: bool,
    background_gif_seed_offset: int,
    background_gif_max_steps: int,
    background_gif_frame_stride: int,
    background_gif_fps: float,
    background_gif_scale: int,
    background_gif_frame_size: int = DEFAULT_BACKGROUND_GIF_FRAME_SIZE,
    background_gif_checkpoint_seed_mixing_enabled: bool = (
        DEFAULT_BACKGROUND_GIF_CHECKPOINT_SEED_MIXING_ENABLED
    ),
    natural_bonus_spawn: bool = TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "source_max_steps": source_max_steps,
        "env_variant": env_variant,
        "reward_variant": DEFAULT_REWARD_VARIANT,
        "eval_reward_variant": DEFAULT_REWARD_VARIANT,
        "training_reward_variant": reward_variant,
        "model_reward_variant": reward_variant,
        "opponent_policy_kind": opponent_policy_kind,
        "opponent_checkpoint_ref": opponent_checkpoint_ref,
        "opponent_snapshot_ref": opponent_snapshot_ref,
        "opponent_checkpoint_state_key": opponent_checkpoint_state_key,
        "natural_bonus_spawn": bool(natural_bonus_spawn),
        "background_eval_enabled": background_eval_enabled,
        "background_eval_compute": background_eval_compute,
        "background_eval_id_prefix": background_eval_id_prefix,
        "background_eval_seed_count": background_eval_seed_count,
        "background_eval_seed_rng_seed": background_eval_seed_rng_seed,
        "background_eval_max_steps": background_eval_max_steps,
        "background_eval_step_detail_limit": background_eval_step_detail_limit,
        "background_eval_num_simulations": background_eval_num_simulations,
        "background_eval_batch_size": background_eval_batch_size,
        "background_gif_enabled": background_gif_enabled,
        "background_gif_seed_offset": background_gif_seed_offset,
        "background_gif_checkpoint_seed_mixing_enabled": bool(
            background_gif_checkpoint_seed_mixing_enabled
        ),
        "background_gif_max_steps": background_gif_max_steps,
        "background_gif_frame_stride": background_gif_frame_stride,
        "background_gif_fps": background_gif_fps,
        "background_gif_scale": background_gif_scale,
        "background_gif_frame_size": background_gif_frame_size,
    }


def _run_checkpoint_eval_poller(
    *,
    run_id: str,
    attempt_id: str,
    exp_name_ref: str,
    poll_interval_sec: float,
    stable_polls: int,
    max_runtime_sec: float,
    idle_after_train_done_sec: float,
    command: dict[str, Any],
) -> dict[str, Any]:
    if poll_interval_sec <= 0.0:
        raise ValueError("poll_interval_sec must be positive")
    if stable_polls < 0:
        raise ValueError("stable_polls must be non-negative")
    if max_runtime_sec <= 0.0:
        raise ValueError("max_runtime_sec must be positive")
    if idle_after_train_done_sec < 0.0:
        raise ValueError("idle_after_train_done_sec must be non-negative")

    config = _background_eval_config_from_command(command)
    exp_name = runs.volume_path(RUNS_MOUNT, exp_name_ref)
    started_monotonic = time.monotonic()
    started_at = runs.utc_timestamp()
    observed: dict[str, dict[str, Any]] = {}
    seen_refs: set[str] = set()
    scheduled: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    outstanding: list[dict[str, Any]] = []
    train_done_at: float | None = None
    last_scan_count = 0

    def status_payload(status: str) -> dict[str, Any]:
        return {
            "schema_id": "curvyzero_lightzero_curvytron_checkpoint_eval_poller/v0",
            "status": status,
            "task_id": TASK_ID,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "started_at": started_at,
            "heartbeat_at": runs.utc_timestamp(),
            "exp_name_ref": exp_name_ref,
            "poll_interval_sec": poll_interval_sec,
            "stable_polls": stable_polls,
            "max_runtime_sec": max_runtime_sec,
            "idle_after_train_done_sec": idle_after_train_done_sec,
            "last_scan_count": last_scan_count,
            "seen_count": len(seen_refs),
            "scheduled_count": len(scheduled),
            "completed_count": len(completed),
            "eval_completed_count": sum(
                1 for item in completed if item.get("kind") == "eval_inspection"
            ),
            "outstanding_count": max(0, len(outstanding) - len(completed)),
            "gif_scheduled_count": sum(
                1
                for item in scheduled
                if isinstance(item.get("selfplay_gif"), dict)
                and item["selfplay_gif"].get("scheduled")
            ),
            "gif_completed_count": sum(
                1 for item in completed if item.get("kind") == "selfplay_gif"
            ),
            "train_done": train_done_at is not None,
            "scheduled": scheduled[-20:],
            "completed": completed[-20:],
        }

    _write_checkpoint_eval_poller_status(
        run_id=run_id,
        attempt_id=attempt_id,
        payload=status_payload("running"),
    )

    while time.monotonic() - started_monotonic < max_runtime_sec:
        if hasattr(runs_volume, "reload"):
            try:
                runs_volume.reload()
            except Exception:
                pass

        artifact_summary = _scan_lightzero_artifacts(str(exp_name))
        checkpoint_files = artifact_summary.get("checkpoint_files", [])
        candidates = [
            item
            for item in checkpoint_files
            if isinstance(item, dict)
            and _live_eval_checkpoint_name(Path(str(item.get("path", ""))).name)
        ]
        last_scan_count = len(candidates)

        for item in sorted(
            candidates,
            key=lambda value: _checkpoint_ref_sort_key(str(value.get("path") or "")),
        ):
            source = Path(str(item.get("path", "")))
            if not source.name or not source.exists() or not source.is_file():
                continue
            checkpoint_ref = _checkpoint_source_ref(source)
            size = int(item.get("size_bytes") or source.stat().st_size)
            mtime_ns = int(item.get("mtime_ns") or source.stat().st_mtime_ns)
            fingerprint = {"size_bytes": size, "mtime_ns": mtime_ns}
            old = observed.get(checkpoint_ref)
            stable_count = (
                int(old.get("stable_count", 0)) + 1
                if old and old.get("fingerprint") == fingerprint
                else 0
            )
            observed[checkpoint_ref] = {
                "fingerprint": fingerprint,
                "stable_count": stable_count,
            }
            if checkpoint_ref in seen_refs or stable_count < stable_polls:
                continue

            checkpoint = {
                "ref": checkpoint_ref,
                "canonical_ref": (
                    runs.checkpoints_root_ref(TASK_ID, run_id) / "lightzero" / source.name
                ).as_posix(),
                "source_path": str(source),
                "copied_now": True,
                **fingerprint,
            }
            request, jobs = _spawn_one_checkpoint_background_eval(
                publish={"run_id": run_id, "attempt_id": attempt_id},
                checkpoint=checkpoint,
                checkpoint_index=len(scheduled),
                config=config,
            )
            scheduled.append(request)
            if request.get("scheduled"):
                seen_refs.add(checkpoint_ref)
                for job in jobs:
                    outstanding.append(
                        {
                            "kind": str(job.get("kind")),
                            "eval_id": str(job.get("eval_id")),
                            "call": job.get("call"),
                        }
                    )
            _write_checkpoint_eval_poller_status(
                run_id=run_id,
                attempt_id=attempt_id,
                payload=status_payload("running"),
            )

        if _checkpoint_eval_poller_train_done(run_id=run_id, attempt_id=attempt_id):
            if train_done_at is None:
                train_done_at = time.monotonic()
            elif time.monotonic() - train_done_at >= idle_after_train_done_sec:
                break

        time.sleep(poll_interval_sec)

    for job in outstanding:
        call = job.get("call")
        if call is None:
            continue
        try:
            result = call.get()
            completed.append(
                {
                    "kind": job.get("kind"),
                    "eval_id": job.get("eval_id"),
                    "ok": bool(isinstance(result, dict) and result.get("ok")),
                    "result": _to_plain(result),
                }
            )
        except Exception as exc:  # pragma: no cover - remote resilience only.
            completed.append(
                {
                    "kind": job.get("kind"),
                    "eval_id": job.get("eval_id"),
                    "ok": False,
                    **_exception_result(exc),
                }
            )

    final_status = status_payload("completed")
    final_status["ended_at"] = runs.utc_timestamp()
    final_status["elapsed_sec"] = round(time.monotonic() - started_monotonic, 6)
    final_status["status_ref"] = _write_checkpoint_eval_poller_status(
        run_id=run_id,
        attempt_id=attempt_id,
        payload=final_status,
    )
    print(json.dumps(_to_plain(final_status), indent=2, sort_keys=True))
    return _to_plain(final_status)


def _live_train_summary_for_inspector(command: dict[str, Any]) -> dict[str, Any]:
    training_readiness_gate = (
        _source_state_fixed_opponent_training_readiness_gate(command=command)
        if command.get("env_variant") == ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT
        else None
    )
    return {
        "schema_id": "curvyzero_lightzero_curvytron_live_train_command_summary/v0",
        "run_id": command.get("run_id"),
        "attempt_id": command.get("attempt_id"),
        "mode": command.get("mode"),
        "ok": None,
        "command": command,
        "opponent_policy_kind": command.get("opponent_policy_kind"),
        "opponent_training_relation": command.get("opponent_training_relation"),
        "current_policy_self_play": command.get("current_policy_self_play"),
        "current_policy_self_play_blocker": command.get("current_policy_self_play_blocker"),
        "trusted_current_policy_self_play": command.get("trusted_current_policy_self_play"),
        "simultaneous_game_theory_claim": command.get("simultaneous_game_theory_claim"),
        "env_variant": command.get("env_variant"),
        "single_product_runtime_path": command.get("single_product_runtime_path"),
        "legacy_debug_variant": command.get("legacy_debug_variant"),
        "underlying_env_class": command.get("underlying_env_class"),
        "runtime_env_impl_id": command.get("runtime_env_impl_id"),
        "runtime_topology": command.get("runtime_topology"),
        "two_seat_self_play": command.get("two_seat_self_play"),
        "current_policy_two_seat_action_collection": command.get(
            "current_policy_two_seat_action_collection"
        ),
        "two_seat_self_play_status": command.get("two_seat_self_play_status"),
        "fixed_opponent_is_two_seat_self_play": command.get(
            "fixed_opponent_is_two_seat_self_play"
        ),
        "browser_pixel_fidelity": command.get("browser_pixel_fidelity"),
        "uses_ale": command.get("uses_ale"),
        "visual_surface": command.get("visual_surface"),
        "visual_truth_level": command.get("visual_truth_level"),
        "visual_source_state_backed": command.get("visual_source_state_backed"),
        "source_fidelity_claim": command.get("source_fidelity_claim"),
        "debug_fidelity_only": command.get("debug_fidelity_only"),
        "training_readiness_gate": training_readiness_gate,
        "learning_proof": command.get("learning_proof", False),
    }


def _load_live_train_summary_for_inspector(
    *,
    run_id: str,
    attempt_id: str,
) -> tuple[dict[str, Any] | None, Path | None]:
    summary_path = runs.volume_path(
        RUNS_MOUNT,
        runs.attempt_train_ref(TASK_ID, run_id, attempt_id) / "summary.json",
    )
    summary = _read_json_if_exists(summary_path)
    if summary is not None:
        return summary, summary_path
    command_path = runs.volume_path(
        RUNS_MOUNT,
        runs.attempt_root_ref(TASK_ID, run_id, attempt_id) / "command.json",
    )
    command = _read_json_if_exists(command_path)
    if command is None:
        return None, None
    return _live_train_summary_for_inspector(command), command_path


def _wait_for_visible_checkpoint(checkpoint_ref: str) -> Path:
    checkpoint_path = runs.volume_path(RUNS_MOUNT, checkpoint_ref)
    deadline = time.monotonic() + DEFAULT_BACKGROUND_CHECKPOINT_WAIT_TIMEOUT_SEC
    while not checkpoint_path.is_file() and time.monotonic() < deadline:
        if hasattr(runs_volume, "reload"):
            try:
                runs_volume.reload()
            except Exception:
                pass
        if checkpoint_path.is_file():
            break
        time.sleep(DEFAULT_BACKGROUND_CHECKPOINT_WAIT_POLL_SEC)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "checkpoint was not visible in the Modal volume before timeout: "
            f"{checkpoint_ref}"
        )
    return checkpoint_path


def _run_checkpoint_eval_and_inspect(
    *,
    checkpoint_ref: str,
    checkpoint_label: str | None,
    eval_id: str,
    run_id: str,
    attempt_id: str,
    compute: str,
    seed: int,
    eval_seed_count: int,
    eval_seed_rng_seed: int | None,
    max_eval_steps: int,
    step_detail_limit: int | None,
    source_max_steps: int,
    num_simulations: int,
    batch_size: int,
    env_variant: str,
    reward_variant: str,
    model_reward_variant: str | None,
    opponent_policy_kind: str,
    opponent_checkpoint_ref: str | None,
    opponent_snapshot_ref: str | None,
    opponent_checkpoint_state_key: str | None,
    natural_bonus_spawn: bool = TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN,
) -> dict[str, Any]:
    if compute not in BACKGROUND_EVAL_COMPUTE_CHOICES:
        raise ValueError(
            f"background eval currently supports {BACKGROUND_EVAL_COMPUTE_CHOICES!r}, got {compute!r}"
        )
    if env_variant not in ENV_VARIANT_CHOICES:
        raise ValueError(
            f"unknown env_variant {env_variant!r}; expected one of {ENV_VARIANT_CHOICES!r}"
        )

    if hasattr(runs_volume, "reload"):
        try:
            runs_volume.reload()
        except Exception:
            pass

    from curvyzero.infra.modal import lightzero_curvytron_visual_survival_eval as eval_mod
    from curvyzero.training.curvytron_inspector import build_inspector_report
    from curvyzero.training.curvytron_inspector import render_markdown_report

    started_at = runs.utc_timestamp()
    clean_eval_id = eval_mod._safe_generated_id(eval_id, fallback="live_checkpoint")
    clean_checkpoint_label = checkpoint_label or eval_mod._checkpoint_label(
        checkpoint_ref,
        index=0,
    )
    eval_seed_values, eval_seed_sampler_seed = eval_mod._parse_eval_seeds(
        seed=seed,
        eval_seeds=None,
        eval_seed_count=eval_seed_count,
        eval_seed_rng_seed=eval_seed_rng_seed,
    )
    _wait_for_visible_checkpoint(checkpoint_ref)
    stamp = runs.utc_stamp()
    jobs: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for job_index, eval_seed in enumerate(eval_seed_values):
        output_ref = eval_mod._output_ref(
            run_id=run_id,
            attempt_id=attempt_id,
            eval_id=clean_eval_id,
            checkpoint_label=clean_checkpoint_label,
            max_eval_steps=max_eval_steps,
            seed=eval_seed,
            stamp=stamp,
        )
        job = {
            "index": job_index,
            "checkpoint_ref": checkpoint_ref,
            "checkpoint_label": clean_checkpoint_label,
            "output_ref": output_ref,
            "seed": eval_seed,
        }
        result = eval_mod._run_eval(
            compute=compute,
            checkpoint_ref=checkpoint_ref,
            output_ref=output_ref,
            run_id=run_id,
            attempt_id=attempt_id,
            seed=eval_seed,
            max_eval_steps=max_eval_steps,
            step_detail_limit=step_detail_limit,
            source_max_steps=max(int(source_max_steps), int(max_eval_steps)),
            num_simulations=num_simulations,
            batch_size=batch_size,
            emit_result_json=False,
            quiet_framework_logs=True,
            env_variant=env_variant,
            reward_variant=reward_variant,
            model_reward_variant=model_reward_variant,
            opponent_policy_kind=opponent_policy_kind,
            opponent_checkpoint_ref=opponent_checkpoint_ref,
            opponent_snapshot_ref=opponent_snapshot_ref,
            opponent_checkpoint_state_key=opponent_checkpoint_state_key,
            natural_bonus_spawn=bool(natural_bonus_spawn),
        )
        jobs.append(job)
        results.append(result)

    table = [
        eval_mod._row_from_result(job, result)
        for job, result in zip(jobs, results, strict=True)
    ]
    survival_aggregate_table = eval_mod._survival_aggregate_table(table)
    survival_table = eval_mod._survival_table(table)
    manifest_ref = eval_mod._manifest_ref(
        run_id=run_id,
        attempt_id=attempt_id,
        eval_id=clean_eval_id,
        max_eval_steps=max_eval_steps,
        seeds=eval_seed_values,
        stamp=runs.utc_stamp(),
    )
    manifest = {
        "schema": "curvyzero_lightzero_curvytron_visual_survival_live_eval/v0",
        "ok": all(bool(result.get("ok")) for result in results),
        "created_at": runs.utc_timestamp(),
        "job_kind": "lightzero_curvytron_visual_survival_live_checkpoint_eval",
        "eval_id": clean_eval_id,
        "selection": {
            "checkpoint_ref": checkpoint_ref,
            "checkpoint_refs": checkpoint_ref,
            "selected_iterations": None,
            "eval_seeds": eval_seed_values,
            "eval_seed_count": eval_seed_count,
            "eval_seed_sampler_seed": eval_seed_sampler_seed,
            "eval_primary_metric": "steps_survived",
            "env_variant": env_variant,
            "eval_reward_variant": reward_variant,
            "env_reward_variant": reward_variant,
            "reward_variant": reward_variant,
            "reward_variant_role": "backward_compatible_alias_for_eval_reward_variant",
            "model_reward_variant": model_reward_variant,
            "effective_model_reward_variant": model_reward_variant or reward_variant,
            "model_reward_variant_role": "checkpoint_model_reconstruction_only_not_scoring",
            "opponent_policy_kind": opponent_policy_kind,
            "opponent_checkpoint_ref": opponent_checkpoint_ref,
            "opponent_snapshot_ref": opponent_snapshot_ref,
            "opponent_checkpoint_state_key": opponent_checkpoint_state_key,
            "natural_bonus_spawn": bool(natural_bonus_spawn),
            "jobs": jobs,
        },
        "config": {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "compute": compute,
            "env_variant": env_variant,
            "eval_primary_metric": "steps_survived",
            "eval_reward_variant": reward_variant,
            "env_reward_variant": reward_variant,
            "reward_variant": reward_variant,
            "reward_variant_role": "backward_compatible_alias_for_eval_reward_variant",
            "model_reward_variant": model_reward_variant,
            "effective_model_reward_variant": model_reward_variant or reward_variant,
            "model_reward_variant_role": "checkpoint_model_reconstruction_only_not_scoring",
            "training_reward_telemetry_field": "episode.total_reward",
            "max_eval_steps": max_eval_steps,
            "step_detail_limit": step_detail_limit,
            "source_max_steps": max(int(source_max_steps), int(max_eval_steps)),
            "num_simulations": num_simulations,
            "batch_size": batch_size,
            "slim_manifest": True,
            "LightZero": LIGHTZERO_VERSION,
            "remote_root": str(REMOTE_ROOT),
            "volume_name": VOLUME_NAME,
            "opponent_policy_kind": opponent_policy_kind,
            "opponent_checkpoint_ref": opponent_checkpoint_ref,
            "opponent_snapshot_ref": opponent_snapshot_ref,
            "opponent_checkpoint_state_key": opponent_checkpoint_state_key,
            "natural_bonus_spawn": bool(natural_bonus_spawn),
        },
        "table": table,
        "survival_aggregate_table": survival_aggregate_table,
        "survival_table": survival_table,
        "output_refs": [job["output_ref"] for job in jobs],
        "artifacts": [
            result.get("artifact")
            for result in results
            if isinstance(result, dict) and result.get("artifact")
        ],
        "results_omitted": True,
        "result_count": len(results),
    }
    manifest_path = runs.volume_path(RUNS_MOUNT, manifest_ref)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    runs.write_json(manifest_path, _to_plain(manifest))

    train_summary, train_summary_path = _load_live_train_summary_for_inspector(
        run_id=run_id,
        attempt_id=attempt_id,
    )
    report = build_inspector_report(
        manifest,
        eval_manifest_path=manifest_path,
        train_summary=train_summary,
        train_summary_path=train_summary_path,
        created_at=runs.utc_timestamp(),
    )
    report_root = runs.volume_path(
        RUNS_MOUNT,
        runs.attempt_eval_ref(TASK_ID, run_id, attempt_id, clean_eval_id) / "inspection",
    )
    report_root.mkdir(parents=True, exist_ok=True)
    report_json_path = report_root / "report.json"
    report_md_path = report_root / "report.md"
    runs.write_json(report_json_path, _to_plain(report))
    _write_text(report_md_path, render_markdown_report(report))

    summary = {
        "ok": bool(manifest["ok"]),
        "schema_id": "curvyzero_lightzero_curvytron_live_checkpoint_eval_inspect_summary/v0",
        "started_at": started_at,
        "ended_at": runs.utc_timestamp(),
        "run_id": run_id,
        "attempt_id": attempt_id,
        "checkpoint_ref": checkpoint_ref,
        "checkpoint_label": clean_checkpoint_label,
        "eval_id": clean_eval_id,
        "env_variant": env_variant,
        "eval_reward_variant": reward_variant,
        "model_reward_variant": model_reward_variant,
        "model_reward_variant_role": "checkpoint_model_reconstruction_only_not_scoring",
        "natural_bonus_spawn": bool(natural_bonus_spawn),
        "manifest_ref": manifest_ref,
        "manifest": runs.file_summary(manifest_path, mount=RUNS_MOUNT),
        "inspection_report_ref": runs.file_ref(report_json_path, mount=RUNS_MOUNT),
        "inspection_report_markdown_ref": runs.file_ref(report_md_path, mount=RUNS_MOUNT),
        "survival_aggregate_table": survival_aggregate_table,
        "survival_table": survival_table,
    }
    print(json.dumps(_to_plain(summary), indent=2, sort_keys=True))
    return _to_plain(summary)


def _selfplay_artifact_root(
    *,
    run_id: str,
    attempt_id: str,
    eval_id: str,
) -> Path:
    return runs.volume_path(
        RUNS_MOUNT,
        runs.attempt_eval_ref(TASK_ID, run_id, attempt_id, eval_id) / "selfplay",
    )


def _copy_source_state_raw_frame(env: Any) -> Any:
    import numpy as np

    raw = None
    if hasattr(env, "raw_observation"):
        raw = env.raw_observation()
    if raw is None and hasattr(env, "render"):
        raw = env.render("source_state_raw_visual_tensor")
    if raw is None:
        raise RuntimeError("self-play env did not expose source_state_raw_visual_tensor")
    frame = np.asarray(raw, dtype=np.uint8)
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return frame.copy()
    raise ValueError(f"raw source-state frame shape {frame.shape!r}; expected [H, W, 3] RGB")


def _resize_rgb_frame_for_gif(frame: Any, *, frame_size: int) -> tuple[Any, str]:
    import numpy as np

    rgb = np.asarray(frame, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"RGB frame shape {rgb.shape!r}; expected [H, W, 3]")
    target_size = int(frame_size)
    if target_size < 1:
        raise ValueError("background_gif_frame_size must be positive")
    if rgb.shape[:2] == (target_size, target_size):
        return rgb.copy(), "none"
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if height < 1 or width < 1:
        raise ValueError(f"RGB frame shape {rgb.shape!r}; expected non-empty height/width")
    if (
        target_size < height
        and target_size < width
        and height % target_size == 0
        and width % target_size == 0
    ):
        y_ratio = height // target_size
        x_ratio = width // target_size
        downsampled = rgb.reshape(target_size, y_ratio, target_size, x_ratio, 3).mean(
            axis=(1, 3),
            dtype=np.float32,
        )
        return np.rint(downsampled).astype(np.uint8), "area_average"
    y_index = np.minimum(
        np.arange(target_size, dtype=np.int64) * height // target_size,
        height - 1,
    )
    x_index = np.minimum(
        np.arange(target_size, dtype=np.int64) * width // target_size,
        width - 1,
    )
    return rgb[y_index[:, None], x_index[None, :]].copy(), "nearest"


def _source_state_rgb_frame_candidate(
    frame: Any,
    *,
    source: str,
    frame_size: int,
) -> tuple[Any, dict[str, Any]] | None:
    import numpy as np

    if frame is None:
        return None
    array = np.asarray(frame)
    input_shape = [int(item) for item in array.shape]
    if array.ndim != 3 or array.shape[-1] != 3:
        return None
    rgb, resize_method = _resize_rgb_frame_for_gif(array, frame_size=int(frame_size))
    return rgb, {
        "source": source,
        "input_shape": input_shape,
        "input_dtype": str(array.dtype),
        "output_shape": [int(item) for item in rgb.shape],
        "resized": input_shape != [int(frame_size), int(frame_size), 3],
        "resize_method": resize_method,
        "resized_nearest": resize_method == "nearest",
    }


def _copy_source_state_human_rgb_frame_with_source(
    env: Any,
    *,
    frame_size: int,
) -> tuple[Any, dict[str, Any]]:
    skipped: list[dict[str, Any]] = []

    def candidate(source: str, producer: Any) -> tuple[Any, dict[str, Any]] | None:
        try:
            frame = producer()
        except Exception as exc:
            skipped.append(
                {
                    "source": source,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            return None
        result = _source_state_rgb_frame_candidate(
            frame,
            source=source,
            frame_size=int(frame_size),
        )
        if result is None:
            shape = [int(item) for item in getattr(frame, "shape", [])]
            skipped.append(
                {
                    "source": source,
                    "status": "non_rgb",
                    "shape": shape,
                    "dtype": str(getattr(frame, "dtype", "missing")),
                }
            )
            return None
        result[1]["skipped_prior_sources"] = skipped.copy()
        return result

    if hasattr(env, "raw_observation"):
        raw_result = candidate("raw_observation", env.raw_observation)
        if raw_result is not None:
            if str(raw_result[1].get("resize_method")) != "nearest":
                return raw_result
            skipped.append(
                {
                    "source": "raw_observation",
                    "status": "rgb_requires_nearest_resize",
                    "shape": raw_result[1]["input_shape"],
                    "dtype": raw_result[1]["input_dtype"],
                    "preferred_frame_size": int(frame_size),
                }
            )
            deferred_raw_result = raw_result
        else:
            deferred_raw_result = None
    else:
        deferred_raw_result = None
    if hasattr(env, "human_rgb_observation"):
        human_result = candidate(
            "human_rgb_observation",
            lambda: env.human_rgb_observation(frame_size=int(frame_size)),
        )
        if human_result is not None:
            return human_result
    if hasattr(env, "render"):
        render_result = candidate(
            "render(source_state_rgb_canvas_like)",
            lambda: env.render("source_state_rgb_canvas_like"),
        )
        if render_result is not None:
            return render_result
    if deferred_raw_result is not None:
        deferred_raw_result[1]["skipped_prior_sources"] = skipped.copy()
        return deferred_raw_result
    raise RuntimeError(
        "self-play env did not expose an RGB source_state_rgb_canvas_like frame; "
        f"skipped_sources={skipped!r}"
    )


def _copy_source_state_human_rgb_frame(env: Any, *, frame_size: int) -> Any:
    frame, _source = _copy_source_state_human_rgb_frame_with_source(
        env,
        frame_size=int(frame_size),
    )
    return frame


SOURCE_STATE_COLLISION_MODEL_SUMMARY = (
    "server/source physics checks current head circle overlap against stored body "
    "circles; browser-style connected trail lines are visual inspection geometry"
)
SOURCE_STATE_COLLISION_VISUAL_WARNING = (
    "A rendered line crossing is not itself proof of a physics collision. Use "
    "death_player/death_cause/death_hit_owner and stored body state for collision truth."
)


def _first_death_from_info(info: dict[str, Any]) -> dict[str, Any]:
    def first_scalar(value: Any) -> Any:
        plain = _to_plain(value)
        while isinstance(plain, list) and plain:
            plain = plain[0]
        return plain

    def row_slot(value: Any, slot: int = 0) -> Any:
        plain = _to_plain(value)
        if plain is None:
            return None
        if isinstance(plain, list):
            if not plain:
                return None
            row = plain[0] if isinstance(plain[0], list) else plain
            if not isinstance(row, list) or len(row) <= slot:
                return None
            return row[slot]
        return plain if slot == 0 else None

    raw_count = first_scalar(info.get("death_count"))
    try:
        death_count = int(raw_count)
    except (TypeError, ValueError):
        death_count = 0

    player = row_slot(info.get("death_player"))
    hit_owner = row_slot(info.get("death_hit_owner"))
    return {
        "death_happened": death_count > 0,
        "death_count": death_count,
        "death_player": player,
        "death_player_id": f"player_{player}" if isinstance(player, int) and player >= 0 else None,
        "death_cause": row_slot(info.get("death_cause")),
        "death_cause_name": row_slot(info.get("death_cause_name")),
        "death_hit_owner": hit_owner,
        "death_hit_owner_id": (
            f"player_{hit_owner}" if isinstance(hit_owner, int) and hit_owner >= 0 else None
        ),
        "terminal_reason": info.get("terminal_reason"),
        "collision_model": SOURCE_STATE_COLLISION_MODEL_SUMMARY,
        "visual_warning": SOURCE_STATE_COLLISION_VISUAL_WARNING,
    }


def _save_raw_frames_gif(
    *,
    frames: Any,
    gif_path: Path,
    fps: float,
    scale: int,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    raw_frames = np.asarray(frames, dtype=np.uint8)
    if raw_frames.ndim == 3:
        frame_height = int(raw_frames.shape[1])
        frame_width = int(raw_frames.shape[2])
        is_rgb = False
    elif raw_frames.ndim == 4 and raw_frames.shape[-1] == 3:
        frame_height = int(raw_frames.shape[1])
        frame_width = int(raw_frames.shape[2])
        is_rgb = True
    else:
        raise ValueError("GIF frames must have shape [N, H, W] or [N, H, W, 3]")
    if raw_frames.shape[0] < 1:
        raise ValueError("GIF frames must include at least one frame")
    if fps <= 0.0:
        raise ValueError("background_gif_fps must be positive")
    if scale < 1:
        raise ValueError("background_gif_scale must be at least 1")
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, int(round(1000.0 / fps)))
    pil_frames = []
    for frame in raw_frames:
        image_frame = Image.fromarray(frame, mode="RGB") if is_rgb else Image.fromarray(frame)
        if scale != 1 and not is_rgb:
            image_frame = image_frame.resize(
                (frame_width * scale, frame_height * scale),
                Image.Resampling.NEAREST,
            )
        pil_frames.append(image_frame)
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return {
        "path": str(gif_path),
        "frame_count": int(raw_frames.shape[0]),
        "fps": float(fps),
        "duration_ms_per_frame": int(duration_ms),
        "scale": 1 if is_rgb else int(scale),
        "pixel_size": [
            frame_width if is_rgb else frame_width * int(scale),
            frame_height if is_rgb else frame_height * int(scale),
        ],
        "color_mode": "RGB" if is_rgb else "L",
    }


def _write_selfplay_raw_frames_npz(
    *,
    frames: Any,
    path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        raw_frames=np.asarray(frames, dtype=np.uint8),
        metadata=json.dumps(_to_plain(metadata), sort_keys=True).encode("utf-8"),
    )
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _run_checkpoint_selfplay_gif(
    *,
    checkpoint_ref: str,
    checkpoint_label: str | None,
    eval_id: str,
    run_id: str,
    attempt_id: str,
    seed: int,
    max_steps: int | None,
    source_max_steps: int,
    num_simulations: int,
    batch_size: int,
    frame_stride: int,
    fps: float,
    scale: int,
    frame_size: int,
    training_env_variant: str,
    training_reward_variant: str,
    natural_bonus_spawn: bool = TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN,
) -> dict[str, Any]:
    import numpy as np

    if max_steps is not None and int(max_steps) < 0:
        raise ValueError(
            "background_gif_max_steps must be non-negative; 0 means no GIF step cap"
        )
    max_step_limit = _normalize_background_gif_max_steps(max_steps)
    step_limit_kind = _background_gif_step_limit_kind(max_step_limit)
    if source_max_steps < 1:
        raise ValueError("source_max_steps must be at least 1")
    if num_simulations < 1:
        raise ValueError("background eval num_simulations must be at least 1")
    if batch_size < 1:
        raise ValueError("background eval batch_size must be at least 1")
    if frame_stride < 1:
        raise ValueError("background_gif_frame_stride must be at least 1")
    if frame_size < 64:
        raise ValueError("background_gif_frame_size must be at least 64")
    effective_source_max_steps = max(
        int(source_max_steps),
        int(max_step_limit)
        if max_step_limit is not None
        else int(DEFAULT_BACKGROUND_EVAL_MAX_STEPS),
    )

    started_at = runs.utc_timestamp()
    clean_eval_id = _safe_generated_id(eval_id, fallback="live_checkpoint")
    clean_checkpoint_label = _safe_generated_id(
        checkpoint_label or _checkpoint_label_from_ref(checkpoint_ref, index=0),
        fallback="checkpoint",
    )
    artifact_root = _selfplay_artifact_root(
        run_id=run_id,
        attempt_id=attempt_id,
        eval_id=clean_eval_id,
    )
    summary_path = artifact_root / "summary.json"
    gif_path = artifact_root / "raw.gif"
    frames_path = artifact_root / "raw_frames.npz"
    telemetry_path = artifact_root / "turn_commit_env_steps.jsonl"

    try:
        if hasattr(runs_volume, "reload"):
            try:
                runs_volume.reload()
            except Exception:
                pass
        from curvyzero.infra.modal import lightzero_curvytron_visual_survival_eval as eval_mod

        checkpoint_path = _wait_for_visible_checkpoint(checkpoint_ref)
        payload = eval_mod._torch_load(checkpoint_path)
        found = eval_mod._find_state_dict(payload)
        if found is None:
            raise ValueError("checkpoint payload did not contain a LightZero state dict")
        found_key, state_dict = found
        policy, env, surface = eval_mod._make_policy_and_env(
            state_dict=state_dict,
            seed=int(seed),
            use_cuda=False,
            source_max_steps=int(effective_source_max_steps),
            num_simulations=int(num_simulations),
            batch_size=int(batch_size),
            telemetry_path=telemetry_path,
            env_variant=ENV_VARIANT_SOURCE_STATE_TURN_COMMIT,
            reward_variant=DEFAULT_REWARD_VARIANT,
            model_env_variant=training_env_variant,
            model_reward_variant=training_reward_variant,
            opponent_policy_kind=OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
            opponent_checkpoint=None,
            opponent_snapshot_ref=None,
            opponent_checkpoint_state_key=None,
            natural_bonus_spawn=bool(natural_bonus_spawn),
        )

        observation = env.reset(seed=int(seed))
        first_frame, first_frame_capture = _copy_source_state_human_rgb_frame_with_source(
            env,
            frame_size=int(frame_size),
        )
        frames: list[Any] = [first_frame]
        frame_captures: list[dict[str, Any]] = [first_frame_capture]
        scalar_actions: list[dict[str, Any]] = []
        joint_actions: list[dict[str, Any]] = []
        physical_steps = 0
        scalar_steps = 0
        done = False
        last_info: dict[str, Any] = {}
        failure: dict[str, Any] | None = None

        while not done and (
            max_step_limit is None or physical_steps < int(max_step_limit)
        ):
            try:
                action_result = eval_mod._policy_eval_action(policy, observation)
                timestep = env.step(int(action_result["action"]))
                observation, reward, done, info = eval_mod._timestep_parts(timestep)
            except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
                failure = {"scalar_step_index": scalar_steps, **_exception_result(exc)}
                break

            last_info = _to_plain(info) if isinstance(info, dict) else {"raw_info": _to_plain(info)}
            scalar_actions.append(
                {
                    "scalar_step_index": int(scalar_steps),
                    "action": int(action_result["action"]),
                    "acting_player_id": last_info.get("acting_player_id"),
                    "physical_env_advanced": bool(last_info.get("physical_env_advanced", False)),
                    "reward": _to_plain(reward),
                    "done": bool(done),
                }
            )
            scalar_steps += 1
            if bool(last_info.get("physical_env_advanced", False)):
                physical_steps += 1
                joint_actions.append(
                    {
                        "physical_step_index": int(physical_steps - 1),
                        "joint_action": last_info.get("joint_action"),
                        "terminal_reason": last_info.get("terminal_reason"),
                        "done": bool(done),
                    }
                )
                if physical_steps % frame_stride == 0 or done:
                    frame, frame_capture = _copy_source_state_human_rgb_frame_with_source(
                        env,
                        frame_size=int(frame_size),
                    )
                    frames.append(frame)
                    frame_captures.append(frame_capture)

        raw_frames = np.stack(frames, axis=0).astype(np.uint8, copy=False)
        frame_capture_method_counts = dict(
            Counter(str(item.get("source", "unknown")) for item in frame_captures)
        )
        frame_capture_method = str(frame_captures[0].get("source", "unknown"))
        artifact_metadata = {
            "schema_id": "curvyzero_lightzero_curvytron_checkpoint_selfplay_rgb_frames/v0",
            "frame_source": "source_state_rgb_canvas_like",
            "frame_schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
            "frame_truth_level": SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL,
            "browser_pixel_fidelity": False,
            "frame_capture_method": frame_capture_method,
            "frame_capture_method_counts": frame_capture_method_counts,
            "frame_capture_details_sample": frame_captures[:4],
            "frame_shape": [int(frame_size), int(frame_size), 3],
            "saved_frame_shape": [int(frame_size), int(frame_size), 3],
            "gif_filename": "raw.gif",
            "gif_content_kind": "source_state_rgb_canvas_like_rgb_frames",
            "gif_filename_role": "legacy_selfplay_artifact_name",
            "gif_filename_note": (
                "raw.gif is a legacy artifact name; these frames are RGB "
                "source-state canvas-like visuals, not the old raw gray tensor."
            ),
            "frame_count": int(raw_frames.shape[0]),
            "frame_stride_physical_steps": int(frame_stride),
            "seed": int(seed),
            "natural_bonus_spawn": bool(natural_bonus_spawn),
            "checkpoint_ref": checkpoint_ref,
            "checkpoint_label": clean_checkpoint_label,
            "max_steps": max_step_limit,
            "step_limit_kind": step_limit_kind,
            "configured_source_max_steps": int(source_max_steps),
            "effective_source_max_steps": int(effective_source_max_steps),
        }
        frames_artifact = _write_selfplay_raw_frames_npz(
            frames=raw_frames,
            path=frames_path,
            metadata=artifact_metadata,
        )
        gif_artifact = _save_raw_frames_gif(
            frames=raw_frames,
            gif_path=gif_path,
            fps=float(fps),
            scale=int(scale),
        )
        if failure is not None:
            stop_reason = "capture_failure"
        elif bool(done):
            stop_reason = "environment_done"
        elif max_step_limit is not None and physical_steps >= int(max_step_limit):
            stop_reason = "gif_step_limit"
        else:
            stop_reason = "loop_exited"
        scalar_action_trace = (
            scalar_actions
            if max_step_limit is None
            else scalar_actions[: 2 * int(max_step_limit)]
        )
        joint_action_trace = (
            joint_actions
            if max_step_limit is None
            else joint_actions[: int(max_step_limit)]
        )
        action_space_size = None
        if isinstance(surface.get("compiled_policy"), dict):
            try:
                action_space_size = int(surface["compiled_policy"]["action_space_size"])
            except (KeyError, TypeError, ValueError):
                action_space_size = None
        greedy_action_summary = _action_trace_observability(
            scalar_actions,
            player_field="acting_player_id",
            action_field="action",
            action_space_size=action_space_size,
        )
        greedy_action_summary["source"] = "scalar_actions_fresh_policy_decisions"
        greedy_action_summary["greedy_action_collapse_warning"] = greedy_action_summary[
            "action_collapse_warning"
        ]
        greedy_action_summary["greedy_action_collapse_players"] = greedy_action_summary[
            "action_collapse_players"
        ]
        joint_action_summary = _joint_action_trace_observability(
            joint_actions,
            action_space_size=action_space_size,
        )
        joint_action_summary["source"] = "joint_actions_physical_commits"
        summary = {
            "ok": failure is None,
            "schema_id": "curvyzero_lightzero_curvytron_checkpoint_selfplay_gif_summary/v0",
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "run_id": run_id,
            "attempt_id": attempt_id,
            "eval_id": clean_eval_id,
            "checkpoint_ref": checkpoint_ref,
            "checkpoint_label": clean_checkpoint_label,
            "checkpoint_state_key": found_key,
            "training_env_variant": training_env_variant,
            "training_reward_variant": training_reward_variant,
            "natural_bonus_spawn": bool(natural_bonus_spawn),
            "capture_env_variant": ENV_VARIANT_SOURCE_STATE_TURN_COMMIT,
            "capture_env_reason": (
                "one checkpoint controls player_0 and player_1 through the source-state "
                "turn-commit adapter; only physical commits become GIF frames"
            ),
            "frame_source": "source_state_rgb_canvas_like",
            "frame_schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
            "frame_truth_level": SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL,
            "browser_pixel_fidelity": False,
            "frame_capture_method": frame_capture_method,
            "frame_capture_method_counts": frame_capture_method_counts,
            "frame_capture_details_sample": frame_captures[:4],
            "raw_frame_source": "source_state_rgb_canvas_like",
            "raw_frame_is_browser_pixel": False,
            "raw_frame_shape": [int(frame_size), int(frame_size), 3],
            "saved_frame_shape": [int(frame_size), int(frame_size), 3],
            "gif_filename": "raw.gif",
            "gif_content_kind": "source_state_rgb_canvas_like_rgb_frames",
            "gif_filename_role": "legacy_selfplay_artifact_name",
            "gif_filename_note": (
                "raw.gif is a legacy artifact name; these frames are RGB "
                "source-state canvas-like visuals, not the old raw gray tensor."
            ),
            "gif_ref": runs.file_ref(gif_path, mount=RUNS_MOUNT),
            "raw_frames_ref": runs.file_ref(frames_path, mount=RUNS_MOUNT),
            "telemetry_ref": (
                runs.file_ref(telemetry_path, mount=RUNS_MOUNT)
                if telemetry_path.exists()
                else None
            ),
            "frame_count": int(raw_frames.shape[0]),
            "physical_steps": int(physical_steps),
            "scalar_steps": int(scalar_steps),
            "greedy_action_collapse_warning": greedy_action_summary[
                "greedy_action_collapse_warning"
            ],
            "greedy_action_summary": greedy_action_summary,
            "joint_action_summary": joint_action_summary,
            "seed": int(seed),
            "seed_role": "effective_checkpoint_selfplay_seed",
            "done": bool(done),
            "terminal_reason": last_info.get("terminal_reason"),
            "stop_reason": stop_reason,
            "max_steps": max_step_limit,
            "step_limit_kind": step_limit_kind,
            "configured_source_max_steps": int(source_max_steps),
            "effective_source_max_steps": int(effective_source_max_steps),
            "frame_stride": int(frame_stride),
            "fps": float(fps),
            "scale": 1,
            "frame_size": int(frame_size),
            "num_simulations": int(num_simulations),
            "batch_size": int(batch_size),
            "surface": _to_plain(surface),
            "action_trace": {
                "scalar_actions": scalar_action_trace,
                "joint_actions": joint_action_trace,
            },
            "failure": failure,
            "artifacts": {
                "gif": {**runs.file_summary(gif_path, mount=RUNS_MOUNT), **gif_artifact},
                "raw_frames": frames_artifact,
            },
        }
    except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
        summary = {
            "ok": False,
            "schema_id": "curvyzero_lightzero_curvytron_checkpoint_selfplay_gif_summary/v0",
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "run_id": run_id,
            "attempt_id": attempt_id,
            "eval_id": clean_eval_id,
            "checkpoint_ref": checkpoint_ref,
            "checkpoint_label": clean_checkpoint_label,
            "training_env_variant": training_env_variant,
            "training_reward_variant": training_reward_variant,
            "natural_bonus_spawn": bool(natural_bonus_spawn),
            "capture_env_variant": ENV_VARIANT_SOURCE_STATE_TURN_COMMIT,
            "frame_source": "source_state_rgb_canvas_like",
            "frame_schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
            "frame_truth_level": SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL,
            "browser_pixel_fidelity": False,
            "raw_frame_source": "source_state_rgb_canvas_like",
            "seed": int(seed),
            "seed_role": "effective_checkpoint_selfplay_seed",
            "max_steps": max_step_limit,
            "step_limit_kind": step_limit_kind,
            "configured_source_max_steps": int(source_max_steps),
            "effective_source_max_steps": int(effective_source_max_steps),
            "gif_filename": "raw.gif",
            "gif_content_kind": "source_state_rgb_canvas_like_rgb_frames",
            "gif_filename_role": "legacy_selfplay_artifact_name",
            "gif_filename_note": (
                "raw.gif is a legacy artifact name; these frames are RGB "
                "source-state canvas-like visuals, not the old raw gray tensor."
            ),
            "error": _exception_result(exc),
        }

    artifact_root.mkdir(parents=True, exist_ok=True)
    summary["summary_ref"] = runs.file_ref(summary_path, mount=RUNS_MOUNT)
    runs.write_json(summary_path, _to_plain(summary))
    summary["summary"] = runs.file_summary(summary_path, mount=RUNS_MOUNT)
    if hasattr(runs_volume, "commit"):
        runs_volume.commit()
    print(json.dumps(_to_plain(summary), indent=2, sort_keys=True))
    return _to_plain(summary)


def _summarize_env_step_telemetry(path: Path) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    physical_action_counts: Counter[str] = Counter()
    opponent_counts: Counter[str] = Counter()
    acting_player_counts: Counter[str] = Counter()
    terminal_reasons: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    observed_fields = _observed_fields_from_telemetry_rows([])
    trainer_reward_sum = 0.0
    physical_trainer_reward_sum = 0.0
    done_count = 0
    physical_env_advanced_count = 0
    pending_scalar_count = 0
    telemetry_stride: int | None = None
    telemetry_sampled = False
    if not path.exists():
        return {
            "row_count": 0,
            "status": "missing",
            "path": str(path),
            "counts_scope": "missing",
            "telemetry_sampled": False,
            "telemetry_stride": None,
            "ego_action_histogram": {"0": 0, "1": 0, "2": 0},
            "physical_action_histogram": {"0": 0, "1": 0, "2": 0},
            "opponent_action_histogram": {"0": 0, "1": 0, "2": 0},
            "observed_fields": observed_fields,
        }
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        observed_fields = _merge_observed_fields(
            observed_fields,
            _observed_fields_from_telemetry_rows([row]),
        )
        if row.get("telemetry_stride") is not None:
            telemetry_stride = int(row["telemetry_stride"])
        telemetry_sampled = telemetry_sampled or bool(row.get("telemetry_sampled", False))
        if len(rows) < 20:
            rows.append(row)
        scalar_action = row.get("scalar_action", row.get("ego_action"))
        action_counts[str(scalar_action)] += 1
        if row.get("acting_player_id") is not None:
            acting_player_counts[str(row.get("acting_player_id"))] += 1
        opponent_counts[str(row.get("opponent_action_id"))] += 1
        physical_env_advanced = bool(row.get("physical_env_advanced", True))
        if physical_env_advanced:
            physical_env_advanced_count += 1
            physical_action_counts[str(scalar_action)] += 1
            physical_trainer_reward_sum += float(row.get("reward") or 0.0)
        else:
            pending_scalar_count += 1
        if row.get("terminal_reason"):
            terminal_reasons[str(row.get("terminal_reason"))] += 1
        trainer_reward_sum += float(row.get("reward") or 0.0)
        done_count += int(bool(row.get("done", False)))
    for action_id in ("0", "1", "2"):
        action_counts.setdefault(action_id, 0)
        physical_action_counts.setdefault(action_id, 0)
        opponent_counts.setdefault(action_id, 0)
    row_count = sum(action_counts.values())
    return {
        "status": "ok",
        "path": str(path),
        "row_count": int(row_count),
        "sampled_row_count": int(row_count),
        "counts_scope": "sampled_telemetry_rows" if telemetry_sampled else "all_telemetry_rows",
        "telemetry_sampled": bool(telemetry_sampled),
        "telemetry_stride": telemetry_stride,
        "scalar_step_count": int(row_count),
        "physical_env_advanced_count": int(physical_env_advanced_count),
        "pending_scalar_count": int(pending_scalar_count),
        "ego_action_histogram": dict(sorted(action_counts.items())),
        "physical_action_histogram": dict(sorted(physical_action_counts.items())),
        "opponent_action_histogram": dict(sorted(opponent_counts.items())),
        "acting_player_histogram": dict(sorted(acting_player_counts.items())),
        "done_count": int(done_count),
        "trainer_reward_sum": trainer_reward_sum,
        "trainer_reward_mean": trainer_reward_sum / row_count if row_count else None,
        "reward_sum": trainer_reward_sum,
        "reward_mean": trainer_reward_sum / row_count if row_count else None,
        "physical_trainer_reward_sum": physical_trainer_reward_sum,
        "physical_trainer_reward_mean": (
            physical_trainer_reward_sum / physical_env_advanced_count
            if physical_env_advanced_count
            else None
        ),
        "physical_reward_sum": physical_trainer_reward_sum,
        "physical_reward_mean": (
            physical_trainer_reward_sum / physical_env_advanced_count
            if physical_env_advanced_count
            else None
        ),
        "terminal_reasons": dict(sorted(terminal_reasons.items())),
        "observed_fields": observed_fields,
        "first_rows": rows,
        "collapse_warning": _action_collapse_warning(action_counts),
        "physical_action_collapse_warning": _action_collapse_warning(
            physical_action_counts
        ),
    }


def _observed_fields_from_telemetry_rows(rows: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "requested_ego_action": any(
            _telemetry_value_present(row.get("requested_ego_action")) for row in rows
        ),
        "executed_ego_action": any(
            _telemetry_value_present(row.get("executed_ego_action")) for row in rows
        ),
        "fixed_opponent_action": any(
            row.get("opponent_policy_kind") == OPPONENT_POLICY_KIND_FIXED_STRAIGHT
            and _telemetry_value_present(row.get("opponent_action_id"))
            for row in rows
        ),
        "joint_action": any(_telemetry_value_present(row.get("joint_action")) for row in rows),
        "action_mask": any(
            _telemetry_value_present(row.get("action_mask"))
            or _telemetry_value_present(row.get("final_observation_action_mask"))
            for row in rows
        ),
        "terminal_reason": any(
            _telemetry_value_present(row.get("terminal_reason")) for row in rows
        ),
        "death_cause": any(
            _telemetry_value_present(row.get("death_cause"))
            or _telemetry_value_present(row.get("death_cause_name"))
            or _telemetry_value_present(row.get("death_hit_owner"))
            for row in rows
        ),
    }


def _merge_observed_fields(
    left: dict[str, bool],
    right: dict[str, bool],
) -> dict[str, bool]:
    return {key: bool(left.get(key, False) or right.get(key, False)) for key in left}


def _telemetry_value_present(value: Any) -> bool:
    if value is None:
        return False
    if value == "":
        return False
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _action_collapse_warning(counts: Counter[str]) -> str | None:
    total = sum(counts.values())
    if total <= 0:
        return "no_actions_observed"
    most_common = counts.most_common(1)[0]
    if most_common[1] == total:
        return f"all_observed_ego_actions_are_{most_common[0]}"
    return None


def _write_run_manifest_once(*, run_id: str, config: dict[str, Any]) -> dict[str, Any]:
    path = runs.volume_path(RUNS_MOUNT, runs.run_manifest_ref(TASK_ID, run_id))
    if path.exists():
        return runs.file_summary(path, mount=RUNS_MOUNT)
    runs.write_json(path, runs.run_manifest(task_id=TASK_ID, run_id=run_id, config=config), exclusive=True)
    return runs.file_summary(path, mount=RUNS_MOUNT)


GREEDY_ACTION_COLLAPSE_WARNING_FRACTION = 0.95


def _action_trace_observability(
    rows: list[dict[str, Any]],
    *,
    player_field: str,
    action_field: str,
    action_space_size: int | None,
    collapse_threshold: float = GREEDY_ACTION_COLLAPSE_WARNING_FRACTION,
) -> dict[str, Any]:
    counts_by_player: dict[str, Counter[str]] = {}
    for row in rows:
        action = row.get(action_field)
        if action is None:
            continue
        try:
            action_key = str(int(action))
        except (TypeError, ValueError):
            continue
        player = row.get(player_field)
        if player is None:
            player_key = "player_unknown"
        elif str(player).startswith("player_"):
            player_key = str(player)
        else:
            player_key = f"player_{player}"
        counts_by_player.setdefault(player_key, Counter())[action_key] += 1

    action_keys = (
        [str(action_id) for action_id in range(int(action_space_size))]
        if action_space_size is not None and int(action_space_size) > 0
        else None
    )
    counts: dict[str, dict[str, int]] = {}
    top_fraction: dict[str, float | None] = {}
    entropy: dict[str, float | None] = {}
    collapse_players: list[str] = []
    for player_key, counter in sorted(counts_by_player.items()):
        if action_keys is None:
            keys = sorted(counter)
        else:
            keys = action_keys
        player_counts = {key: int(counter.get(key, 0)) for key in keys}
        total = sum(player_counts.values())
        max_count = max(player_counts.values(), default=0)
        fraction = max_count / total if total > 0 else None
        nonzero_counts = [count for count in player_counts.values() if count > 0]
        if total <= 0 or len(player_counts) <= 1:
            player_entropy = None
        elif len(nonzero_counts) <= 1:
            player_entropy = 0.0
        else:
            raw_entropy = -sum(
                (count / total) * math.log(count / total) for count in nonzero_counts
            )
            player_entropy = raw_entropy / math.log(len(player_counts))
        if fraction is not None and fraction >= float(collapse_threshold):
            collapse_players.append(player_key)
        counts[player_key] = player_counts
        top_fraction[player_key] = round(fraction, 6) if fraction is not None else None
        entropy[player_key] = (
            round(player_entropy, 6) if player_entropy is not None else None
        )

    return {
        "decision_count": sum(sum(player_counts.values()) for player_counts in counts.values()),
        "action_counts_by_player": counts,
        "top_action_fraction_by_player": top_fraction,
        "action_entropy_by_player": entropy,
        "collapse_threshold": float(collapse_threshold),
        "action_collapse_warning": bool(collapse_players),
        "action_collapse_players": collapse_players,
    }


def _joint_action_trace_observability(
    rows: list[dict[str, Any]],
    *,
    action_space_size: int | None,
) -> dict[str, Any]:
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        joint_action = row.get("joint_action")
        if not isinstance(joint_action, dict):
            continue
        for player_key, action in joint_action.items():
            flat_rows.append({"player": str(player_key), "action": action})
    return _action_trace_observability(
        flat_rows,
        player_field="player",
        action_field="action",
        action_space_size=action_space_size,
    )


def _write_gif_browser_run_marker(
    *,
    run_id: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    path = runs.volume_path(RUNS_MOUNT, runs.gif_browser_run_marker_ref(TASK_ID, run_id))
    if path.exists():
        return runs.file_summary(path, mount=RUNS_MOUNT)
    payload = runs.gif_browser_run_marker(
        task_id=TASK_ID,
        run_id=run_id,
        created_at=created_at,
    )
    try:
        runs.write_json(path, payload, exclusive=True)
    except FileExistsError:
        pass
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _write_attempt_state(
    *,
    run_id: str,
    attempt_id: str,
    status: str,
    started_at: str,
    ended_at: str | None,
    summary_ref: str | None,
    config: dict[str, Any],
    modal_task_id: str | None,
    exclusive: bool = False,
) -> dict[str, Any]:
    path = runs.volume_path(RUNS_MOUNT, runs.attempt_manifest_ref(TASK_ID, run_id, attempt_id))
    payload = runs.attempt_manifest(
        task_id=TASK_ID,
        run_id=run_id,
        attempt_id=attempt_id,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        modal_task_id=modal_task_id,
        summary_ref=summary_ref,
        config=config,
    )
    runs.write_json(path, payload, exclusive=exclusive)
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _write_latest_attempt(
    *,
    run_id: str,
    attempt_id: str,
    status: str,
    started_at: str,
    ended_at: str | None,
    summary_ref: str | None,
    modal_task_id: str | None,
) -> dict[str, Any]:
    path = runs.volume_path(RUNS_MOUNT, runs.latest_attempt_ref(TASK_ID, run_id))
    payload = runs.latest_attempt_pointer(
        task_id=TASK_ID,
        run_id=run_id,
        attempt_id=attempt_id,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        modal_task_id=modal_task_id,
        summary_ref=summary_ref,
    )
    runs.write_json(path, payload)
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _write_train_status_heartbeat(
    *,
    run_id: str,
    attempt_id: str,
    status: str,
    stage: str,
    started_at: str,
    modal_task_id: str | None,
    config: dict[str, Any],
    exp_name_ref: str,
) -> dict[str, Any]:
    path = runs.volume_path(RUNS_MOUNT, runs.attempt_train_ref(TASK_ID, run_id, attempt_id))
    path = path / "status_heartbeat.json"
    payload = {
        "schema_id": "curvyzero_lightzero_curvytron_train_status_heartbeat/v0",
        "task_id": TASK_ID,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "status": status,
        "stage": stage,
        "started_at": started_at,
        "heartbeat_at": runs.utc_timestamp(),
        "modal_task_id": modal_task_id,
        "exp_name_ref": exp_name_ref,
        "summary_ref": (runs.attempt_train_ref(TASK_ID, run_id, attempt_id) / "summary.json").as_posix(),
        "checkpoint_root_ref": (runs.checkpoints_root_ref(TASK_ID, run_id) / "lightzero").as_posix(),
        "command": config,
    }
    runs.write_json(path, payload)
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _write_text(path: Path, text: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _compact_log_tail(text: str, *, limit: int = 80) -> list[str]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def _version_or_missing(*packages: str) -> str:
    for package in packages:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return "missing"


def _exception_result(exc: BaseException) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _set_or_add_path(mapping: Any, path: tuple[str, ...], value: Any) -> dict[str, Any]:
    current = mapping
    for part in path[:-1]:
        if part not in current or current[part] is None:
            current[part] = {}
        current = current[part]
    key = path[-1]
    old_value = current.get(key, "<missing>")
    current[key] = value
    return {"path": ".".join(path), "old": _to_plain(old_value), "new": _to_plain(value)}


def _get_path(mapping: Any, path: tuple[str, ...], default: Any = None) -> Any:
    current = mapping
    try:
        for part in path:
            current = current[part]
    except KeyError:
        return default
    return current


def _cfg_get(mapping: Any, key: str, default: Any) -> Any:
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _runs_volume_commit_callback() -> None:
    if hasattr(runs_volume, "commit"):
        runs_volume.commit()


def _two_seat_checkpoint_ref(run_id: str) -> Path:
    return runs.checkpoints_root_ref(TASK_ID, run_id) / "lightzero"


def _two_seat_call_refs(run_id: str, attempt_id: str) -> dict[str, Any]:
    train_ref = runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
    checkpoint_ref = _two_seat_checkpoint_ref(run_id)
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "progress_ref": (train_ref / "progress.jsonl").as_posix(),
        "progress_latest_ref": (train_ref / "progress_latest.json").as_posix(),
        "summary_ref": (train_ref / "summary.json").as_posix(),
        "command_ref": (
            runs.attempt_root_ref(TASK_ID, run_id, attempt_id) / "command.json"
        ).as_posix(),
        "checkpoint_root_ref": checkpoint_ref.as_posix(),
    }


def _two_seat_background_eval_config(
    *,
    payload: dict[str, Any],
    background_eval_enabled: bool,
    background_eval_launch_kind: str,
    background_eval_compute: str,
    background_eval_id_prefix: str,
    background_eval_seed_count: int,
    background_eval_seed_rng_seed: int | None,
    background_eval_max_steps: int,
    background_eval_step_detail_limit: int | None,
    background_eval_num_simulations: int,
    background_eval_batch_size: int,
    background_eval_poll_interval_sec: float,
    background_eval_poll_stable_polls: int,
    background_eval_poller_max_runtime_sec: float,
    background_eval_poller_idle_after_done_sec: float,
    background_gif_enabled: bool,
    background_gif_seed_offset: int,
    background_gif_max_steps: int,
    background_gif_frame_stride: int,
    background_gif_fps: float,
    background_gif_scale: int,
    background_gif_frame_size: int,
) -> dict[str, Any]:
    return {
        "enabled": bool(background_eval_enabled),
        "launch_kind": background_eval_launch_kind,
        "compute": background_eval_compute,
        "eval_id_prefix": background_eval_id_prefix,
        "seed_count": int(background_eval_seed_count),
        "seed_rng_seed": background_eval_seed_rng_seed,
        "max_eval_steps": int(background_eval_max_steps),
        "step_detail_limit": background_eval_step_detail_limit,
        "num_simulations": int(background_eval_num_simulations),
        "batch_size": int(background_eval_batch_size),
        "natural_bonus_spawn": bool(
            payload.get("natural_bonus_spawn", TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN)
        ),
        "poll_interval_sec": float(background_eval_poll_interval_sec),
        "stable_polls": int(background_eval_poll_stable_polls),
        "max_runtime_sec": float(background_eval_poller_max_runtime_sec),
        "idle_after_train_done_sec": float(background_eval_poller_idle_after_done_sec),
        "selfplay_gif": {
            "enabled": bool(background_gif_enabled),
            "seed_offset": int(background_gif_seed_offset),
            "checkpoint_seed_mixing_enabled": DEFAULT_BACKGROUND_GIF_CHECKPOINT_SEED_MIXING_ENABLED,
            "max_steps": _normalize_background_gif_max_steps(background_gif_max_steps),
            "step_limit_kind": _background_gif_step_limit_kind(background_gif_max_steps),
            "frame_stride": int(background_gif_frame_stride),
            "fps": float(background_gif_fps),
            "scale": int(background_gif_scale),
            "frame_size": int(background_gif_frame_size),
            "natural_bonus_spawn": bool(
                payload.get("natural_bonus_spawn", TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN)
            ),
        },
        "eval_env_variant": ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        "eval_reward_variant": REWARD_VARIANT_SPARSE_OUTCOME,
        "eval_opponent_policy_kind": OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        "note": (
            "two-seat checkpoints are evaluated with the existing fixed-opponent "
            "survival reader until a two-seat eval surface replaces it"
        ),
        "checkpoint_root_ref": _two_seat_checkpoint_ref(str(payload["run_id"])).as_posix(),
    }


def _spawn_two_seat_checkpoint_poller(
    *,
    payload: dict[str, Any],
    background: dict[str, Any],
) -> tuple[Any | None, str | None]:
    if not background["enabled"]:
        return None, None
    if background["launch_kind"] != BACKGROUND_EVAL_LAUNCH_POLLER:
        raise ValueError("two-seat self-play currently supports poller background eval only")
    if not bool(payload["allow_optimizer_step"]):
        raise ValueError("two-seat background eval needs allow_optimizer_step=True")
    checkpoint_ref = _two_seat_checkpoint_ref(str(payload["run_id"]))
    call = lightzero_curvytron_visual_survival_checkpoint_eval_poller.spawn(
        run_id=str(payload["run_id"]),
        attempt_id=str(payload["attempt_id"]),
        exp_name_ref=checkpoint_ref.as_posix(),
        seed=int(payload["seed"]),
        source_max_steps=max(
            int(payload.get("max_ticks") or 0),
            int(background["max_eval_steps"]),
            DEFAULT_SOURCE_MAX_STEPS,
        ),
        env_variant=background["eval_env_variant"],
        reward_variant=background["eval_reward_variant"],
        opponent_policy_kind=background["eval_opponent_policy_kind"],
        opponent_checkpoint_ref=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        background_eval_compute=str(background["compute"]),
        background_eval_id_prefix=str(background["eval_id_prefix"]),
        background_eval_seed_count=int(background["seed_count"]),
        background_eval_seed_rng_seed=background["seed_rng_seed"],
        background_eval_max_steps=int(background["max_eval_steps"]),
        background_eval_step_detail_limit=background["step_detail_limit"],
        background_eval_num_simulations=int(background["num_simulations"]),
        background_eval_batch_size=int(background["batch_size"]),
        background_gif_enabled=bool(background["selfplay_gif"]["enabled"]),
        background_gif_seed_offset=int(background["selfplay_gif"]["seed_offset"]),
        background_gif_max_steps=_background_gif_max_steps_arg(
            background["selfplay_gif"].get("max_steps", DEFAULT_BACKGROUND_GIF_MAX_STEPS)
        ),
        background_gif_frame_stride=int(background["selfplay_gif"]["frame_stride"]),
        background_gif_fps=float(background["selfplay_gif"]["fps"]),
        background_gif_scale=int(background["selfplay_gif"]["scale"]),
        background_gif_frame_size=int(background["selfplay_gif"]["frame_size"]),
        background_gif_natural_bonus_spawn=bool(
            background["selfplay_gif"].get(
                "natural_bonus_spawn",
                background.get("natural_bonus_spawn", TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN),
            )
        ),
        poll_interval_sec=float(background["poll_interval_sec"]),
        stable_polls=int(background["stable_polls"]),
        max_runtime_sec=float(background["max_runtime_sec"]),
        idle_after_train_done_sec=float(background["idle_after_train_done_sec"]),
    )
    return call, getattr(call, "object_id", None) or getattr(call, "id", None)


def _run_two_seat_selfplay_payload(
    payload: dict[str, Any],
    *,
    compute_label: str,
    use_cuda: bool,
) -> dict[str, Any]:
    started_at = runs.utc_timestamp()
    run_id = str(payload["run_id"])
    attempt_id = str(payload["attempt_id"])
    checkpoint_ref = _two_seat_checkpoint_ref(run_id)
    checkpoint_dir = runs.volume_path(RUNS_MOUNT, checkpoint_ref)
    train_ref = runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
    attempt_root = runs.volume_path(RUNS_MOUNT, runs.attempt_root_ref(TASK_ID, run_id, attempt_id))
    train_root = runs.volume_path(RUNS_MOUNT, train_ref)
    progress_path = runs.volume_path(RUNS_MOUNT, train_ref / "progress.jsonl")
    summary_path = train_root / "summary.json"
    command_path = attempt_root / "command.json"
    attempt_path = runs.volume_path(
        RUNS_MOUNT,
        runs.attempt_manifest_ref(TASK_ID, run_id, attempt_id),
    )
    latest_attempt_path = runs.volume_path(
        RUNS_MOUNT,
        runs.latest_attempt_ref(TASK_ID, run_id),
    )
    gif_browser_run_marker_enabled = bool(
        payload.get("gif_browser_run_marker_enabled", DEFAULT_BACKGROUND_GIF_ENABLED)
    )
    natural_bonus_spawn = bool(
        payload.get("natural_bonus_spawn", TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN)
    )
    terminal_outcome_reward_per_step = float(
        payload.get(
            "terminal_outcome_reward_per_step",
            TWO_SEAT_DEFAULT_TERMINAL_OUTCOME_REWARD_PER_STEP,
        )
    )
    bonus_pickup_reward_per_catch = float(
        payload.get(
            "bonus_pickup_reward_per_catch",
            TWO_SEAT_DEFAULT_BONUS_PICKUP_REWARD_PER_CATCH,
        )
    )
    return_target_discount = float(
        payload.get("return_target_discount", TWO_SEAT_DEFAULT_RETURN_TARGET_DISCOUNT)
    )
    learning_rate = payload.get("learning_rate")
    command = {
        "schema_id": "curvyzero_canonical_two_seat_selfplay_command/v0",
        "mode": TWO_SEAT_SELFPLAY_MODE,
        "canonical_launcher": (
            "curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train"
        ),
        "launcher_status": "canonical_two_seat_selfplay",
        "compute": compute_label,
        "use_cuda": bool(use_cuda),
        **payload,
        "natural_bonus_spawn": natural_bonus_spawn,
        "terminal_outcome_reward_per_step": terminal_outcome_reward_per_step,
        "bonus_pickup_reward_per_catch": bonus_pickup_reward_per_catch,
        "return_target_discount": return_target_discount,
        "learning_rate": learning_rate,
        "gif_browser_run_marker_enabled": gif_browser_run_marker_enabled,
        "gif_browser_run_marker_ref": (
            runs.gif_browser_run_marker_ref(TASK_ID, run_id).as_posix()
            if gif_browser_run_marker_enabled
            else None
        ),
    }
    if gif_browser_run_marker_enabled:
        _write_gif_browser_run_marker(run_id=run_id, created_at=started_at)
    runs.write_json(command_path, _to_plain(command))
    runs.write_json(
        attempt_path,
        runs.attempt_manifest(
            task_id=TASK_ID,
            run_id=run_id,
            attempt_id=attempt_id,
            status="running",
            started_at=started_at,
            config=command,
        ),
    )
    runs.write_json(
        latest_attempt_path,
        runs.latest_attempt_pointer(
            task_id=TASK_ID,
            run_id=run_id,
            attempt_id=attempt_id,
            status="running",
            started_at=started_at,
        ),
    )
    if hasattr(runs_volume, "commit"):
        runs_volume.commit()

    result = run_curvytron_two_seat_lightzero_train_smoke(
        seed=int(payload["seed"]),
        batch_size=int(payload["batch_size"]),
        steps=int(payload["steps"]),
        outer_iterations=int(payload["outer_iterations"]),
        collect_steps_per_iteration=payload["collect_steps_per_iteration"],
        updates_per_iteration=payload["updates_per_iteration"],
        num_simulations=int(payload["num_simulations"]),
        learner_updates=int(payload["learner_updates"]),
        allow_optimizer_step=bool(payload["allow_optimizer_step"]),
        replay_scope=str(payload["replay_scope"]),
        learner_sample_size=payload["learner_sample_size"],
        max_replay_rows=payload["max_replay_rows"],
        record_log_limit=int(payload["record_log_limit"]),
        replay_row_log_limit=int(payload["replay_row_log_limit"]),
        max_ticks=payload["max_ticks"],
        death_mode=str(payload["death_mode"]),
        natural_bonus_spawn=natural_bonus_spawn,
        decision_ms=float(payload["decision_ms"]),
        alive_reward=float(payload["alive_reward"]),
        dead_reward=float(payload["dead_reward"]),
        terminal_outcome_reward_per_step=terminal_outcome_reward_per_step,
        bonus_pickup_reward_per_catch=bonus_pickup_reward_per_catch,
        return_target_discount=return_target_discount,
        action_selection_mode=str(payload["action_selection_mode"]),
        collect_temperature=float(payload["collect_temperature"]),
        collect_epsilon=float(payload["collect_epsilon"]),
        action_noop_probability=float(payload["action_noop_probability"]),
        action_noop_warmup_iterations=int(payload["action_noop_warmup_iterations"]),
        policy_action_repeat_min=int(payload["policy_action_repeat_min"]),
        policy_action_repeat_max=int(payload["policy_action_repeat_max"]),
        policy_action_repeat_extra_probability=float(
            payload["policy_action_repeat_extra_probability"]
        ),
        policy_action_repeat_warmup_iterations=int(
            payload["policy_action_repeat_warmup_iterations"]
        ),
        observation_noise_std=float(payload["observation_noise_std"]),
        trail_render_mode=str(payload["trail_render_mode"]),
        learning_rate=learning_rate,
        use_cuda=bool(use_cuda),
        checkpoint_every_iterations=int(payload["checkpoint_every_iterations"]),
        save_initial_checkpoint=bool(payload["save_initial_checkpoint"]),
        progress_path=progress_path,
        progress_every_iterations=int(payload["progress_every_iterations"]),
        progress_commit_every_iterations=int(payload["progress_commit_every_iterations"]),
        progress_commit_callback=_runs_volume_commit_callback,
        progress_print=True,
        checkpoint_dir=checkpoint_dir if bool(payload["allow_optimizer_step"]) else None,
        checkpoint_metadata={
            "task_id": TASK_ID,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "checkpoint_root_ref": checkpoint_ref.as_posix(),
            "modal_app": APP_NAME,
            "compute": compute_label,
            "canonical_launcher": command["canonical_launcher"],
            "two_seat_current_policy_selfplay": True,
            "trail_render_mode": payload["trail_render_mode"],
            "death_mode": payload["death_mode"],
            "natural_bonus_spawn": natural_bonus_spawn,
            "alive_reward": payload["alive_reward"],
            "dead_reward": payload["dead_reward"],
            "terminal_outcome_reward_per_step": terminal_outcome_reward_per_step,
            "bonus_pickup_reward_per_catch": bonus_pickup_reward_per_catch,
            "return_target_discount": return_target_discount,
            "learning_rate": learning_rate,
        },
        require_installed_lightzero=True,
    )
    result["summary_ref"] = runs.file_ref(summary_path, mount=RUNS_MOUNT)
    result["progress_ref"] = (train_ref / "progress.jsonl").as_posix()
    result["progress_latest_ref"] = (train_ref / "progress_latest.json").as_posix()
    result["checkpoint_root_ref"] = checkpoint_ref.as_posix()
    result["command_ref"] = runs.file_ref(command_path, mount=RUNS_MOUNT)
    result["volume_name"] = VOLUME_NAME
    result["canonical_launcher"] = command["canonical_launcher"]
    runs.write_json(summary_path, _to_plain(result))
    status = "completed" if bool(result.get("ok")) else "failed"
    ended_at = runs.utc_timestamp()
    summary_ref = runs.file_ref(summary_path, mount=RUNS_MOUNT)
    runs.write_json(
        attempt_path,
        runs.attempt_manifest(
            task_id=TASK_ID,
            run_id=run_id,
            attempt_id=attempt_id,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            summary_ref=summary_ref,
            config=command,
        ),
    )
    runs.write_json(
        latest_attempt_path,
        runs.latest_attempt_pointer(
            task_id=TASK_ID,
            run_id=run_id,
            attempt_id=attempt_id,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            summary_ref=summary_ref,
        ),
    )
    if hasattr(runs_volume, "commit"):
        runs_volume.commit()
    return _to_plain(result)


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=12 * 60 * 60,
    cpu=40.0,
    memory=65536,
)
def lightzero_curvytron_two_seat_selfplay_cpu(payload: dict[str, Any]) -> dict[str, Any]:
    return _run_two_seat_selfplay_payload(
        payload,
        compute_label=COMPUTE_CPU,
        use_cuda=False,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=12 * 60 * 60,
    cpu=40.0,
    memory=65536,
    gpu=CHEAP_GPU_RESOURCE,
)
def lightzero_curvytron_two_seat_selfplay_gpu(payload: dict[str, Any]) -> dict[str, Any]:
    return _run_two_seat_selfplay_payload(
        payload,
        compute_label=COMPUTE_GPU_L4_T4,
        use_cuda=True,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=12 * 60 * 60,
    cpu=40.0,
    memory=65536,
    gpu=H100_GPU_RESOURCE,
)
def lightzero_curvytron_two_seat_selfplay_h100(payload: dict[str, Any]) -> dict[str, Any]:
    return _run_two_seat_selfplay_payload(
        payload,
        compute_label=COMPUTE_GPU_H100_CPU40,
        use_cuda=True,
    )


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=40 * 60, cpu=2.0)
def lightzero_curvytron_visual_survival_checkpoint_eval_and_inspect(
    checkpoint_ref: str,
    checkpoint_label: str | None = None,
    eval_id: str = DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    compute: str = DEFAULT_BACKGROUND_EVAL_COMPUTE,
    seed: int = DEFAULT_SEED,
    eval_seed_count: int = DEFAULT_BACKGROUND_EVAL_SEED_COUNT,
    eval_seed_rng_seed: int | None = DEFAULT_BACKGROUND_EVAL_SEED_RNG_SEED,
    max_eval_steps: int = DEFAULT_BACKGROUND_EVAL_MAX_STEPS,
    step_detail_limit: int | None = DEFAULT_BACKGROUND_EVAL_STEP_DETAIL_LIMIT,
    source_max_steps: int = DEFAULT_SOURCE_MAX_STEPS,
    num_simulations: int = DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BACKGROUND_EVAL_BATCH_SIZE,
    env_variant: str = DEFAULT_ENV_VARIANT,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    model_reward_variant: str | None = None,
    opponent_policy_kind: str = DEFAULT_OPPONENT_POLICY_KIND,
    opponent_checkpoint_ref: str | None = None,
    opponent_snapshot_ref: str | None = None,
    opponent_checkpoint_state_key: str | None = None,
    natural_bonus_spawn: bool = TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN,
) -> dict[str, Any]:
    return _run_checkpoint_eval_and_inspect(
        checkpoint_ref=checkpoint_ref,
        checkpoint_label=checkpoint_label,
        eval_id=eval_id,
        run_id=run_id,
        attempt_id=attempt_id,
        compute=compute,
        seed=seed,
        eval_seed_count=eval_seed_count,
        eval_seed_rng_seed=eval_seed_rng_seed,
        max_eval_steps=max_eval_steps,
        step_detail_limit=step_detail_limit,
        source_max_steps=source_max_steps,
        num_simulations=num_simulations,
        batch_size=batch_size,
        env_variant=env_variant,
        reward_variant=reward_variant,
        model_reward_variant=model_reward_variant,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
        opponent_snapshot_ref=opponent_snapshot_ref,
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
        natural_bonus_spawn=bool(natural_bonus_spawn),
    )


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=40 * 60, cpu=2.0)
def lightzero_curvytron_visual_survival_checkpoint_selfplay_gif(
    checkpoint_ref: str,
    checkpoint_label: str | None = None,
    eval_id: str = DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    seed: int = DEFAULT_SEED + DEFAULT_BACKGROUND_GIF_SEED_OFFSET,
    max_steps: int = DEFAULT_BACKGROUND_GIF_MAX_STEPS,
    source_max_steps: int = DEFAULT_SOURCE_MAX_STEPS,
    num_simulations: int = DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BACKGROUND_EVAL_BATCH_SIZE,
    frame_stride: int = DEFAULT_BACKGROUND_GIF_FRAME_STRIDE,
    fps: float = DEFAULT_BACKGROUND_GIF_FPS,
    scale: int = DEFAULT_BACKGROUND_GIF_SCALE,
    frame_size: int = DEFAULT_BACKGROUND_GIF_FRAME_SIZE,
    training_env_variant: str = DEFAULT_ENV_VARIANT,
    training_reward_variant: str = DEFAULT_REWARD_VARIANT,
    natural_bonus_spawn: bool = TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN,
) -> dict[str, Any]:
    return _run_checkpoint_selfplay_gif(
        checkpoint_ref=checkpoint_ref,
        checkpoint_label=checkpoint_label,
        eval_id=eval_id,
        run_id=run_id,
        attempt_id=attempt_id,
        seed=seed,
        max_steps=max_steps,
        source_max_steps=source_max_steps,
        num_simulations=num_simulations,
        batch_size=batch_size,
        frame_stride=frame_stride,
        fps=fps,
        scale=scale,
        frame_size=frame_size,
        training_env_variant=training_env_variant,
        training_reward_variant=training_reward_variant,
        natural_bonus_spawn=bool(natural_bonus_spawn),
    )


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=8 * 60 * 60, cpu=1.0)
def lightzero_curvytron_visual_survival_checkpoint_eval_poller(
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    exp_name_ref: str | None = None,
    seed: int = DEFAULT_SEED,
    source_max_steps: int = DEFAULT_SOURCE_MAX_STEPS,
    env_variant: str = DEFAULT_ENV_VARIANT,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    opponent_policy_kind: str = DEFAULT_OPPONENT_POLICY_KIND,
    opponent_checkpoint_ref: str | None = None,
    opponent_snapshot_ref: str | None = None,
    opponent_checkpoint_state_key: str | None = None,
    background_eval_compute: str = DEFAULT_BACKGROUND_EVAL_COMPUTE,
    background_eval_id_prefix: str = DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
    background_eval_seed_count: int = DEFAULT_BACKGROUND_EVAL_SEED_COUNT,
    background_eval_seed_rng_seed: int | None = DEFAULT_BACKGROUND_EVAL_SEED_RNG_SEED,
    background_eval_max_steps: int = DEFAULT_BACKGROUND_EVAL_MAX_STEPS,
    background_eval_step_detail_limit: int | None = DEFAULT_BACKGROUND_EVAL_STEP_DETAIL_LIMIT,
    background_eval_num_simulations: int = DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS,
    background_eval_batch_size: int = DEFAULT_BACKGROUND_EVAL_BATCH_SIZE,
    background_gif_enabled: bool = DEFAULT_BACKGROUND_GIF_ENABLED,
    background_gif_seed_offset: int = DEFAULT_BACKGROUND_GIF_SEED_OFFSET,
    background_gif_max_steps: int = DEFAULT_BACKGROUND_GIF_MAX_STEPS,
    background_gif_frame_stride: int = DEFAULT_BACKGROUND_GIF_FRAME_STRIDE,
    background_gif_fps: float = DEFAULT_BACKGROUND_GIF_FPS,
    background_gif_scale: int = DEFAULT_BACKGROUND_GIF_SCALE,
    background_gif_frame_size: int = DEFAULT_BACKGROUND_GIF_FRAME_SIZE,
    background_gif_natural_bonus_spawn: bool = TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN,
    poll_interval_sec: float = DEFAULT_BACKGROUND_EVAL_POLL_INTERVAL_SEC,
    stable_polls: int = DEFAULT_BACKGROUND_EVAL_POLL_STABLE_POLLS,
    max_runtime_sec: float = DEFAULT_BACKGROUND_EVAL_POLLER_MAX_RUNTIME_SEC,
    idle_after_train_done_sec: float = DEFAULT_BACKGROUND_EVAL_POLLER_IDLE_AFTER_DONE_SEC,
) -> dict[str, Any]:
    exp_name_ref = exp_name_ref or (
        runs.attempt_train_ref(TASK_ID, run_id, attempt_id) / "lightzero_exp"
    ).as_posix()
    command = _checkpoint_eval_poller_command(
        seed=seed,
        source_max_steps=source_max_steps,
        env_variant=env_variant,
        reward_variant=reward_variant,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
        opponent_snapshot_ref=opponent_snapshot_ref,
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
        background_eval_enabled=True,
        background_eval_compute=background_eval_compute,
        background_eval_id_prefix=background_eval_id_prefix,
        background_eval_seed_count=background_eval_seed_count,
        background_eval_seed_rng_seed=background_eval_seed_rng_seed,
        background_eval_max_steps=background_eval_max_steps,
        background_eval_step_detail_limit=background_eval_step_detail_limit,
        background_eval_num_simulations=background_eval_num_simulations,
        background_eval_batch_size=background_eval_batch_size,
        background_gif_enabled=background_gif_enabled,
        background_gif_seed_offset=background_gif_seed_offset,
        background_gif_max_steps=background_gif_max_steps,
        background_gif_frame_stride=background_gif_frame_stride,
        background_gif_fps=background_gif_fps,
        background_gif_scale=background_gif_scale,
        background_gif_frame_size=background_gif_frame_size,
        natural_bonus_spawn=bool(background_gif_natural_bonus_spawn),
    )
    return _run_checkpoint_eval_poller(
        run_id=run_id,
        attempt_id=attempt_id,
        exp_name_ref=exp_name_ref,
        poll_interval_sec=poll_interval_sec,
        stable_polls=stable_polls,
        max_runtime_sec=max_runtime_sec,
        idle_after_train_done_sec=idle_after_train_done_sec,
        command=command,
    )


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=20 * 60, cpu=2.0)
def lightzero_curvytron_visual_survival_cpu(
    mode: str = DEFAULT_MODE,
    seed: int = DEFAULT_SEED,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    source_max_steps: int = DEFAULT_SOURCE_MAX_STEPS,
    decision_ms: float = DEFAULT_DECISION_MS,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    n_evaluator_episode: int = DEFAULT_N_EVALUATOR_EPISODE,
    n_episode: int = DEFAULT_N_EPISODE,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lightzero_eval_freq: int = DEFAULT_LIGHTZERO_EVAL_FREQ,
    skip_lightzero_eval_in_profile: bool = DEFAULT_SKIP_LIGHTZERO_EVAL_IN_PROFILE,
    profile_cuda_sync_enabled: bool = DEFAULT_PROFILE_CUDA_SYNC_ENABLED,
    profile_allow_auto_resume: bool = DEFAULT_PROFILE_ALLOW_AUTO_RESUME,
    lightzero_multi_gpu: bool = DEFAULT_LIGHTZERO_MULTI_GPU,
    save_ckpt_after_iter: int = DEFAULT_SAVE_CKPT_AFTER_ITER,
    stop_after_learner_train_calls: int = DEFAULT_STOP_AFTER_LEARNER_TRAIN_CALLS,
    env_variant: str = DEFAULT_ENV_VARIANT,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    ego_action_straight_override_probability: float = (
        DEFAULT_EGO_ACTION_STRAIGHT_OVERRIDE_PROBABILITY
    ),
    control_noise_profile_id: str = DEFAULT_CONTROL_NOISE_PROFILE_ID,
    disable_death_for_profile: bool = DEFAULT_DISABLE_DEATH_FOR_PROFILE,
    env_telemetry_stride: int = DEFAULT_ENV_TELEMETRY_STRIDE,
    env_manager_type: str = DEFAULT_ENV_MANAGER_TYPE,
    opponent_policy_kind: str = DEFAULT_OPPONENT_POLICY_KIND,
    opponent_checkpoint_ref: str | None = None,
    opponent_snapshot_ref: str | None = None,
    opponent_checkpoint_report_ref: str | None = None,
    opponent_checkpoint_state_key: str | None = None,
    background_eval_enabled: bool = DEFAULT_BACKGROUND_EVAL_ENABLED,
    background_eval_launch_kind: str = BACKGROUND_EVAL_LAUNCH_HOOK,
    background_eval_compute: str = DEFAULT_BACKGROUND_EVAL_COMPUTE,
    background_eval_id_prefix: str = DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
    background_eval_seed_count: int = DEFAULT_BACKGROUND_EVAL_SEED_COUNT,
    background_eval_seed_rng_seed: int | None = DEFAULT_BACKGROUND_EVAL_SEED_RNG_SEED,
    background_eval_max_steps: int = DEFAULT_BACKGROUND_EVAL_MAX_STEPS,
    background_eval_step_detail_limit: int | None = DEFAULT_BACKGROUND_EVAL_STEP_DETAIL_LIMIT,
    background_eval_num_simulations: int = DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS,
    background_eval_batch_size: int = DEFAULT_BACKGROUND_EVAL_BATCH_SIZE,
    background_gif_enabled: bool = DEFAULT_BACKGROUND_GIF_ENABLED,
    background_gif_seed_offset: int = DEFAULT_BACKGROUND_GIF_SEED_OFFSET,
    background_gif_max_steps: int = DEFAULT_BACKGROUND_GIF_MAX_STEPS,
    background_gif_frame_stride: int = DEFAULT_BACKGROUND_GIF_FRAME_STRIDE,
    background_gif_fps: float = DEFAULT_BACKGROUND_GIF_FPS,
    background_gif_scale: int = DEFAULT_BACKGROUND_GIF_SCALE,
    background_gif_frame_size: int = DEFAULT_BACKGROUND_GIF_FRAME_SIZE,
) -> dict[str, Any]:
    return _run_visual_survival_train(
        mode=mode,
        compute="cpu",
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        source_max_steps=source_max_steps,
        decision_ms=decision_ms,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=n_evaluator_episode,
        n_episode=n_episode,
        num_simulations=num_simulations,
        batch_size=batch_size,
        lightzero_eval_freq=lightzero_eval_freq,
        skip_lightzero_eval_in_profile=skip_lightzero_eval_in_profile,
        profile_cuda_sync_enabled=profile_cuda_sync_enabled,
        profile_allow_auto_resume=profile_allow_auto_resume,
        lightzero_multi_gpu=lightzero_multi_gpu,
        save_ckpt_after_iter=save_ckpt_after_iter,
        stop_after_learner_train_calls=stop_after_learner_train_calls,
        env_variant=env_variant,
        reward_variant=reward_variant,
        ego_action_straight_override_probability=ego_action_straight_override_probability,
        control_noise_profile_id=control_noise_profile_id,
        disable_death_for_profile=disable_death_for_profile,
        env_telemetry_stride=env_telemetry_stride,
        env_manager_type=env_manager_type,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
        opponent_snapshot_ref=opponent_snapshot_ref,
        opponent_checkpoint_report_ref=opponent_checkpoint_report_ref,
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
        background_eval_enabled=background_eval_enabled,
        background_eval_launch_kind=background_eval_launch_kind,
        background_eval_compute=background_eval_compute,
        background_eval_id_prefix=background_eval_id_prefix,
        background_eval_seed_count=background_eval_seed_count,
        background_eval_seed_rng_seed=background_eval_seed_rng_seed,
        background_eval_max_steps=background_eval_max_steps,
        background_eval_step_detail_limit=background_eval_step_detail_limit,
        background_eval_num_simulations=background_eval_num_simulations,
        background_eval_batch_size=background_eval_batch_size,
        background_gif_enabled=background_gif_enabled,
        background_gif_seed_offset=background_gif_seed_offset,
        background_gif_max_steps=background_gif_max_steps,
        background_gif_frame_stride=background_gif_frame_stride,
        background_gif_fps=background_gif_fps,
        background_gif_scale=background_gif_scale,
        background_gif_frame_size=background_gif_frame_size,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=8 * 60 * 60,
    cpu=64.0,
    memory=65536,
)
def lightzero_curvytron_visual_survival_cpu64(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("background_gif_frame_size", DEFAULT_BACKGROUND_GIF_FRAME_SIZE)
    kwargs.setdefault("reward_variant", DEFAULT_REWARD_VARIANT)
    kwargs.setdefault("lightzero_eval_freq", DEFAULT_LIGHTZERO_EVAL_FREQ)
    kwargs.setdefault("skip_lightzero_eval_in_profile", DEFAULT_SKIP_LIGHTZERO_EVAL_IN_PROFILE)
    return _run_visual_survival_train(compute=COMPUTE_CPU64, **kwargs)


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=8 * 60 * 60,
    cpu=8.0,
    memory=32768,
    gpu=CHEAP_GPU_RESOURCE,
)
def lightzero_curvytron_visual_survival_gpu(
    mode: str = DEFAULT_MODE,
    seed: int = DEFAULT_SEED,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    source_max_steps: int = DEFAULT_SOURCE_MAX_STEPS,
    decision_ms: float = DEFAULT_DECISION_MS,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    n_evaluator_episode: int = DEFAULT_N_EVALUATOR_EPISODE,
    n_episode: int = DEFAULT_N_EPISODE,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lightzero_eval_freq: int = DEFAULT_LIGHTZERO_EVAL_FREQ,
    skip_lightzero_eval_in_profile: bool = DEFAULT_SKIP_LIGHTZERO_EVAL_IN_PROFILE,
    profile_cuda_sync_enabled: bool = DEFAULT_PROFILE_CUDA_SYNC_ENABLED,
    profile_allow_auto_resume: bool = DEFAULT_PROFILE_ALLOW_AUTO_RESUME,
    lightzero_multi_gpu: bool = DEFAULT_LIGHTZERO_MULTI_GPU,
    save_ckpt_after_iter: int = DEFAULT_SAVE_CKPT_AFTER_ITER,
    stop_after_learner_train_calls: int = DEFAULT_STOP_AFTER_LEARNER_TRAIN_CALLS,
    env_variant: str = DEFAULT_ENV_VARIANT,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    ego_action_straight_override_probability: float = (
        DEFAULT_EGO_ACTION_STRAIGHT_OVERRIDE_PROBABILITY
    ),
    control_noise_profile_id: str = DEFAULT_CONTROL_NOISE_PROFILE_ID,
    disable_death_for_profile: bool = DEFAULT_DISABLE_DEATH_FOR_PROFILE,
    env_telemetry_stride: int = DEFAULT_ENV_TELEMETRY_STRIDE,
    env_manager_type: str = DEFAULT_ENV_MANAGER_TYPE,
    opponent_policy_kind: str = DEFAULT_OPPONENT_POLICY_KIND,
    opponent_checkpoint_ref: str | None = None,
    opponent_snapshot_ref: str | None = None,
    opponent_checkpoint_report_ref: str | None = None,
    opponent_checkpoint_state_key: str | None = None,
    background_eval_enabled: bool = DEFAULT_BACKGROUND_EVAL_ENABLED,
    background_eval_launch_kind: str = BACKGROUND_EVAL_LAUNCH_HOOK,
    background_eval_compute: str = DEFAULT_BACKGROUND_EVAL_COMPUTE,
    background_eval_id_prefix: str = DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
    background_eval_seed_count: int = DEFAULT_BACKGROUND_EVAL_SEED_COUNT,
    background_eval_seed_rng_seed: int | None = DEFAULT_BACKGROUND_EVAL_SEED_RNG_SEED,
    background_eval_max_steps: int = DEFAULT_BACKGROUND_EVAL_MAX_STEPS,
    background_eval_step_detail_limit: int | None = DEFAULT_BACKGROUND_EVAL_STEP_DETAIL_LIMIT,
    background_eval_num_simulations: int = DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS,
    background_eval_batch_size: int = DEFAULT_BACKGROUND_EVAL_BATCH_SIZE,
    background_gif_enabled: bool = DEFAULT_BACKGROUND_GIF_ENABLED,
    background_gif_seed_offset: int = DEFAULT_BACKGROUND_GIF_SEED_OFFSET,
    background_gif_max_steps: int = DEFAULT_BACKGROUND_GIF_MAX_STEPS,
    background_gif_frame_stride: int = DEFAULT_BACKGROUND_GIF_FRAME_STRIDE,
    background_gif_fps: float = DEFAULT_BACKGROUND_GIF_FPS,
    background_gif_scale: int = DEFAULT_BACKGROUND_GIF_SCALE,
    background_gif_frame_size: int = DEFAULT_BACKGROUND_GIF_FRAME_SIZE,
) -> dict[str, Any]:
    return _run_visual_survival_train(
        mode=mode,
        compute="gpu-l4-t4",
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        source_max_steps=source_max_steps,
        decision_ms=decision_ms,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=n_evaluator_episode,
        n_episode=n_episode,
        num_simulations=num_simulations,
        batch_size=batch_size,
        lightzero_eval_freq=lightzero_eval_freq,
        skip_lightzero_eval_in_profile=skip_lightzero_eval_in_profile,
        profile_cuda_sync_enabled=profile_cuda_sync_enabled,
        profile_allow_auto_resume=profile_allow_auto_resume,
        lightzero_multi_gpu=lightzero_multi_gpu,
        save_ckpt_after_iter=save_ckpt_after_iter,
        stop_after_learner_train_calls=stop_after_learner_train_calls,
        env_variant=env_variant,
        reward_variant=reward_variant,
        ego_action_straight_override_probability=ego_action_straight_override_probability,
        control_noise_profile_id=control_noise_profile_id,
        disable_death_for_profile=disable_death_for_profile,
        env_telemetry_stride=env_telemetry_stride,
        env_manager_type=env_manager_type,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
        opponent_snapshot_ref=opponent_snapshot_ref,
        opponent_checkpoint_report_ref=opponent_checkpoint_report_ref,
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
        background_eval_enabled=background_eval_enabled,
        background_eval_launch_kind=background_eval_launch_kind,
        background_eval_compute=background_eval_compute,
        background_eval_id_prefix=background_eval_id_prefix,
        background_eval_seed_count=background_eval_seed_count,
        background_eval_seed_rng_seed=background_eval_seed_rng_seed,
        background_eval_max_steps=background_eval_max_steps,
        background_eval_step_detail_limit=background_eval_step_detail_limit,
        background_eval_num_simulations=background_eval_num_simulations,
        background_eval_batch_size=background_eval_batch_size,
        background_gif_enabled=background_gif_enabled,
        background_gif_seed_offset=background_gif_seed_offset,
        background_gif_max_steps=background_gif_max_steps,
        background_gif_frame_stride=background_gif_frame_stride,
        background_gif_fps=background_gif_fps,
        background_gif_scale=background_gif_scale,
        background_gif_frame_size=background_gif_frame_size,
    )


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=8 * 60 * 60,
    cpu=40.0,
    memory=65536,
    gpu=CHEAP_GPU_RESOURCE,
)
def lightzero_curvytron_visual_survival_gpu_cpu40(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("background_gif_frame_size", DEFAULT_BACKGROUND_GIF_FRAME_SIZE)
    kwargs.setdefault("reward_variant", DEFAULT_REWARD_VARIANT)
    kwargs.setdefault("lightzero_eval_freq", DEFAULT_LIGHTZERO_EVAL_FREQ)
    kwargs.setdefault("skip_lightzero_eval_in_profile", DEFAULT_SKIP_LIGHTZERO_EVAL_IN_PROFILE)
    return _run_visual_survival_train(compute=COMPUTE_GPU_L4_T4_CPU40, **kwargs)


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=8 * 60 * 60,
    cpu=40.0,
    memory=65536,
    gpu=H100_GPU_RESOURCE,
)
def lightzero_curvytron_visual_survival_h100_cpu40(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("background_gif_frame_size", DEFAULT_BACKGROUND_GIF_FRAME_SIZE)
    kwargs.setdefault("reward_variant", DEFAULT_REWARD_VARIANT)
    kwargs.setdefault("lightzero_eval_freq", DEFAULT_LIGHTZERO_EVAL_FREQ)
    kwargs.setdefault("skip_lightzero_eval_in_profile", DEFAULT_SKIP_LIGHTZERO_EVAL_IN_PROFILE)
    return _run_visual_survival_train(compute=COMPUTE_GPU_H100_CPU40, **kwargs)


@app.function(
    image=image,
    volumes={str(RUNS_MOUNT): runs_volume},
    timeout=8 * 60 * 60,
    cpu=40.0,
    memory=65536,
    gpu=H100X2_GPU_RESOURCE,
)
def lightzero_curvytron_visual_survival_h100x2_cpu40(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("background_gif_frame_size", DEFAULT_BACKGROUND_GIF_FRAME_SIZE)
    kwargs.setdefault("reward_variant", DEFAULT_REWARD_VARIANT)
    kwargs.setdefault("lightzero_eval_freq", DEFAULT_LIGHTZERO_EVAL_FREQ)
    kwargs.setdefault("skip_lightzero_eval_in_profile", DEFAULT_SKIP_LIGHTZERO_EVAL_IN_PROFILE)
    return _run_visual_survival_train(compute=COMPUTE_GPU_H100X2_CPU40, **kwargs)


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=20 * 60, cpu=2.0)
def lightzero_curvytron_visual_survival_opponent_smoke(
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = "snapshot-opponent-wrapper-smoke",
    opponent_checkpoint_ref: str | None = None,
    snapshot_ref: str = "curvytron_visual_survival_snapshot_opponent_smoke",
    checkpoint_ref: str | None = None,
    seed: int = DEFAULT_SEED,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    state_key: str | None = None,
    ego_action_id: int = 0,
    fake_action_id: int = 1,
) -> dict[str, Any]:
    return _run_lightzero_checkpoint_opponent_modal_smoke(
        run_id=run_id,
        attempt_id=attempt_id,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
        snapshot_ref=snapshot_ref,
        checkpoint_ref=checkpoint_ref,
        seed=seed,
        num_simulations=num_simulations,
        batch_size=batch_size,
        state_key=state_key,
        ego_action_id=ego_action_id,
        fake_action_id=fake_action_id,
    )


def _compact_train_result_for_output(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    train_result = result.get("train") if isinstance(result.get("train"), dict) else result
    if not isinstance(train_result, dict):
        return result
    phase = train_result.get("phase_profile")
    command = train_result.get("command")
    action = train_result.get("action_observability")
    runtime = train_result.get("runtime_compute")
    phase = phase if isinstance(phase, dict) else {}
    command = command if isinstance(command, dict) else {}
    action = action if isinstance(action, dict) else {}
    runtime = runtime if isinstance(runtime, dict) else {}
    timers = phase.get("timers_sec") if isinstance(phase.get("timers_sec"), dict) else {}
    counts = phase.get("counts") if isinstance(phase.get("counts"), dict) else {}
    derived = phase.get("derived_stats") if isinstance(phase.get("derived_stats"), dict) else {}
    gpu = phase.get("gpu_sampling") if isinstance(phase.get("gpu_sampling"), dict) else {}
    mcts_search_calls = counts.get("mcts_search_calls")
    mcts_root_sum = counts.get("mcts_search_root_sum")
    train_wall = timers.get("train_muzero_wall_sec")
    env_steps = counts.get("env_steps_collected")
    steps_per_sec = None
    try:
        if train_wall and env_steps is not None:
            steps_per_sec = float(env_steps) / float(train_wall)
    except (TypeError, ValueError, ZeroDivisionError):
        steps_per_sec = None
    mcts_root_batch_mean = derived.get("mcts_search_root_batch_mean")
    try:
        if mcts_root_batch_mean is None and mcts_search_calls and mcts_root_sum is not None:
            mcts_root_batch_mean = float(mcts_root_sum) / float(mcts_search_calls)
    except (TypeError, ValueError, ZeroDivisionError):
        mcts_root_batch_mean = None
    compact: dict[str, Any] = {
        "schema_id": "curvyzero_lightzero_curvytron_visual_survival_compact_output/v0",
        "ok": train_result.get("ok"),
        "status": train_result.get("status"),
        "problems": train_result.get("problems", []),
        "run_id": train_result.get("run_id"),
        "attempt_id": train_result.get("attempt_id"),
        "summary_ref": train_result.get("summary_ref"),
        "mode": train_result.get("mode"),
        "compute": train_result.get("compute"),
        "called_train_muzero": train_result.get("called_train_muzero"),
        "command": {
            "env_variant": command.get("env_variant"),
            "reward_variant": command.get("reward_variant"),
            "env_manager_type": command.get("env_manager_type"),
            "collector_env_num": command.get("collector_env_num"),
            "n_episode": command.get("n_episode"),
            "num_simulations": command.get("num_simulations"),
            "batch_size": command.get("batch_size"),
            "lightzero_eval_freq": command.get("lightzero_eval_freq"),
            "skip_lightzero_eval_in_profile": command.get("skip_lightzero_eval_in_profile"),
            "profile_cuda_sync_enabled": command.get("profile_cuda_sync_enabled"),
            "profile_allow_auto_resume": command.get("profile_allow_auto_resume"),
            "lightzero_multi_gpu": command.get("lightzero_multi_gpu"),
            "source_max_steps": command.get("source_max_steps"),
            "disable_death_for_profile": command.get("disable_death_for_profile"),
            "env_telemetry_stride": command.get("env_telemetry_stride"),
            "save_ckpt_after_iter": command.get("save_ckpt_after_iter"),
        },
        "counts": {
            "env_steps_collected": env_steps,
            "mcts_search_calls": mcts_search_calls,
            "mcts_search_root_sum": mcts_root_sum,
            "mcts_search_simulation_budget_sum": counts.get(
                "mcts_search_simulation_budget_sum"
            ),
            "learner_train_calls": counts.get("learner_train_calls"),
            "replay_sample_calls": counts.get("replay_sample_calls"),
        },
        "timers_sec": {
            "train_muzero_wall": train_wall,
            "collector_collect": timers.get("collector_collect_sec"),
            "mcts_search": timers.get("mcts_search_sec"),
            "policy_forward_collect": timers.get("policy_forward_collect_sec"),
            "policy_forward_eval": timers.get("policy_forward_eval_sec"),
            "model_initial_inference": timers.get("model_initial_inference_sec"),
            "model_recurrent_inference": timers.get("model_recurrent_inference_sec"),
            "learner_train": timers.get("learner_train_sec"),
            "replay_sample": timers.get("replay_sample_sec"),
            "evaluator_eval": timers.get("evaluator_eval_sec"),
            "env_telemetry_write": timers.get("env_telemetry_write_sec"),
            "learner_save_checkpoint": timers.get("learner_save_checkpoint_sec"),
        },
        "derived": {
            "steps_per_sec": steps_per_sec,
            "mcts_root_batch_mean": mcts_root_batch_mean,
            "mcts_recurrent_batch_mean": derived.get(
                "model_recurrent_inference_in_mcts_search_batch_mean"
            ),
        },
        "telemetry": {
            "row_count": action.get("row_count"),
            "counts_scope": action.get("counts_scope"),
            "telemetry_sampled": action.get("telemetry_sampled"),
            "telemetry_stride": action.get("telemetry_stride"),
        },
        "gpu": {
            "requested_compute": runtime.get("requested_compute"),
            "available": runtime.get("torch_cuda_available"),
            "max_util_percent": gpu.get("max_gpu_util_percent"),
            "max_memory_used_mib": gpu.get("max_memory_used_mib"),
            "sample_count": gpu.get("sample_count"),
        },
    }
    if "background_eval" in result:
        compact["background_eval"] = result["background_eval"]
    return compact


@app.local_entrypoint()
def main(
    mode: str = DEFAULT_MODE,
    compute: str = DEFAULT_COMPUTE,
    seed: int = DEFAULT_SEED,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    source_max_steps: int = DEFAULT_SOURCE_MAX_STEPS,
    decision_ms: float = DEFAULT_DECISION_MS,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    n_evaluator_episode: int = DEFAULT_N_EVALUATOR_EPISODE,
    n_episode: int = DEFAULT_N_EPISODE,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lightzero_eval_freq: int = DEFAULT_LIGHTZERO_EVAL_FREQ,
    skip_lightzero_eval_in_profile: bool = DEFAULT_SKIP_LIGHTZERO_EVAL_IN_PROFILE,
    profile_cuda_sync_enabled: bool = DEFAULT_PROFILE_CUDA_SYNC_ENABLED,
    profile_allow_auto_resume: bool = DEFAULT_PROFILE_ALLOW_AUTO_RESUME,
    lightzero_multi_gpu: bool = DEFAULT_LIGHTZERO_MULTI_GPU,
    save_ckpt_after_iter: int = DEFAULT_SAVE_CKPT_AFTER_ITER,
    stop_after_learner_train_calls: int = DEFAULT_STOP_AFTER_LEARNER_TRAIN_CALLS,
    env_variant: str = DEFAULT_ENV_VARIANT,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    ego_action_straight_override_probability: float = (
        DEFAULT_EGO_ACTION_STRAIGHT_OVERRIDE_PROBABILITY
    ),
    control_noise_profile_id: str = DEFAULT_CONTROL_NOISE_PROFILE_ID,
    disable_death_for_profile: bool = DEFAULT_DISABLE_DEATH_FOR_PROFILE,
    env_telemetry_stride: int = DEFAULT_ENV_TELEMETRY_STRIDE,
    env_manager_type: str = DEFAULT_ENV_MANAGER_TYPE,
    wait_for_train: bool = False,
    opponent_policy_kind: str = DEFAULT_OPPONENT_POLICY_KIND,
    opponent_checkpoint_ref: str | None = None,
    snapshot_ref: str = "curvytron_visual_survival_snapshot_opponent_smoke",
    checkpoint_ref: str | None = None,
    state_key: str | None = None,
    ego_action_id: int = 0,
    fake_action_id: int = 1,
    two_seat_steps: int = DEFAULT_TWO_SEAT_COLLECT_STEPS_PER_ITERATION,
    two_seat_outer_iterations: int | None = None,
    two_seat_collect_steps_per_iteration: int | None = None,
    two_seat_updates_per_iteration: int | None = None,
    two_seat_learner_updates: int = 1,
    two_seat_allow_optimizer_step: bool = True,
    two_seat_replay_scope: str = "accumulated",
    two_seat_learner_sample_size: int | None = DEFAULT_TWO_SEAT_LEARNER_SAMPLE_SIZE,
    two_seat_max_replay_rows: int | None = DEFAULT_TWO_SEAT_MAX_REPLAY_ROWS,
    two_seat_record_log_limit: int = 512,
    two_seat_replay_row_log_limit: int = 256,
    two_seat_max_ticks: int | None = DEFAULT_TWO_SEAT_MAX_TICKS,
    two_seat_death_mode: str = TWO_SEAT_DEFAULT_DEATH_MODE,
    two_seat_natural_bonus_spawn: bool = TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN,
    two_seat_alive_reward: float = TWO_SEAT_DEFAULT_ALIVE_REWARD,
    two_seat_dead_reward: float = TWO_SEAT_DEFAULT_DEAD_REWARD,
    two_seat_terminal_outcome_reward_per_step: float = (
        TWO_SEAT_DEFAULT_TERMINAL_OUTCOME_REWARD_PER_STEP
    ),
    two_seat_bonus_pickup_reward_per_catch: float = (
        TWO_SEAT_DEFAULT_BONUS_PICKUP_REWARD_PER_CATCH
    ),
    two_seat_return_target_discount: float = TWO_SEAT_DEFAULT_RETURN_TARGET_DISCOUNT,
    two_seat_action_selection_mode: str = "collect",
    two_seat_collect_temperature: float = 1.0,
    two_seat_collect_epsilon: float = 0.25,
    two_seat_action_noop_probability: float = TWO_SEAT_DEFAULT_ACTION_NOOP_PROBABILITY,
    two_seat_action_noop_warmup_iterations: int = (
        TWO_SEAT_DEFAULT_ACTION_NOOP_WARMUP_ITERATIONS
    ),
    two_seat_policy_action_repeat_min: int = TWO_SEAT_DEFAULT_POLICY_ACTION_REPEAT_MIN,
    two_seat_policy_action_repeat_max: int = TWO_SEAT_DEFAULT_POLICY_ACTION_REPEAT_MAX,
    two_seat_policy_action_repeat_extra_probability: float = (
        TWO_SEAT_DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
    ),
    two_seat_policy_action_repeat_warmup_iterations: int = (
        TWO_SEAT_DEFAULT_POLICY_ACTION_REPEAT_WARMUP_ITERATIONS
    ),
    two_seat_observation_noise_std: float = TWO_SEAT_DEFAULT_OBSERVATION_NOISE_STD,
    two_seat_trail_render_mode: str = TRAIL_RENDER_MODE_DEFAULT,
    two_seat_learning_rate: float | None = None,
    two_seat_checkpoint_every_iterations: int | None = None,
    two_seat_save_initial_checkpoint: bool = DEFAULT_TWO_SEAT_SAVE_INITIAL_CHECKPOINT,
    two_seat_progress_every_iterations: int = DEFAULT_TWO_SEAT_PROGRESS_EVERY_ITERATIONS,
    two_seat_progress_commit_every_iterations: int = (
        DEFAULT_TWO_SEAT_PROGRESS_COMMIT_EVERY_ITERATIONS
    ),
    background_eval_enabled: bool = DEFAULT_BACKGROUND_EVAL_ENABLED,
    background_eval_launch_kind: str = DEFAULT_BACKGROUND_EVAL_LAUNCH_KIND,
    background_eval_compute: str = DEFAULT_BACKGROUND_EVAL_COMPUTE,
    background_eval_id_prefix: str = DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
    background_eval_seed_count: int = DEFAULT_BACKGROUND_EVAL_SEED_COUNT,
    background_eval_seed_rng_seed: int | None = DEFAULT_BACKGROUND_EVAL_SEED_RNG_SEED,
    background_eval_max_steps: int = DEFAULT_BACKGROUND_EVAL_MAX_STEPS,
    background_eval_step_detail_limit: int | None = DEFAULT_BACKGROUND_EVAL_STEP_DETAIL_LIMIT,
    background_eval_num_simulations: int = DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS,
    background_eval_batch_size: int = DEFAULT_BACKGROUND_EVAL_BATCH_SIZE,
    background_eval_poll_interval_sec: float = DEFAULT_BACKGROUND_EVAL_POLL_INTERVAL_SEC,
    background_eval_poll_stable_polls: int = DEFAULT_BACKGROUND_EVAL_POLL_STABLE_POLLS,
    background_eval_poller_max_runtime_sec: float = DEFAULT_BACKGROUND_EVAL_POLLER_MAX_RUNTIME_SEC,
    background_eval_poller_idle_after_done_sec: float = DEFAULT_BACKGROUND_EVAL_POLLER_IDLE_AFTER_DONE_SEC,
    background_gif_enabled: bool = DEFAULT_BACKGROUND_GIF_ENABLED,
    background_gif_seed_offset: int = DEFAULT_BACKGROUND_GIF_SEED_OFFSET,
    background_gif_max_steps: int = DEFAULT_BACKGROUND_GIF_MAX_STEPS,
    background_gif_frame_stride: int = DEFAULT_BACKGROUND_GIF_FRAME_STRIDE,
    background_gif_fps: float = DEFAULT_BACKGROUND_GIF_FPS,
    background_gif_scale: int = DEFAULT_BACKGROUND_GIF_SCALE,
    background_gif_frame_size: int = DEFAULT_BACKGROUND_GIF_FRAME_SIZE,
    output_detail: str = OUTPUT_DETAIL_COMPACT,
) -> None:
    if output_detail not in OUTPUT_DETAIL_CHOICES:
        raise ValueError(
            f"output_detail must be one of {OUTPUT_DETAIL_CHOICES!r}; got {output_detail!r}"
        )
    if mode == OPPONENT_SMOKE_MODE:
        result = lightzero_curvytron_visual_survival_opponent_smoke.remote(
            run_id=run_id,
            attempt_id=attempt_id,
            opponent_checkpoint_ref=opponent_checkpoint_ref,
            snapshot_ref=snapshot_ref,
            checkpoint_ref=checkpoint_ref,
            seed=seed,
            num_simulations=num_simulations,
            batch_size=batch_size,
            state_key=state_key,
            ego_action_id=ego_action_id,
            fake_action_id=fake_action_id,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if mode == TWO_SEAT_SELFPLAY_MODE:
        if compute == COMPUTE_CPU:
            two_seat_fn = lightzero_curvytron_two_seat_selfplay_cpu
        elif compute == COMPUTE_GPU_L4_T4:
            two_seat_fn = lightzero_curvytron_two_seat_selfplay_gpu
        elif compute == COMPUTE_GPU_H100_CPU40:
            two_seat_fn = lightzero_curvytron_two_seat_selfplay_h100
        else:
            raise ValueError(
                "two-seat self-play supports compute='cpu', 'gpu-l4-t4', "
                "or 'gpu-h100-cpu40'"
            )
        if two_seat_trail_render_mode not in STACK_RENDER_MODE_ORDER:
            raise ValueError(
                "two-seat self-play trail render mode must be one of "
                f"{STACK_RENDER_MODE_ORDER!r}; got {two_seat_trail_render_mode!r}"
            )
        two_seat_payload = {
            "seed": seed,
            "batch_size": batch_size,
            "steps": two_seat_steps,
            "outer_iterations": (
                max_train_iter
                if two_seat_outer_iterations is None
                else int(two_seat_outer_iterations)
            ),
            "collect_steps_per_iteration": (
                DEFAULT_TWO_SEAT_COLLECT_STEPS_PER_ITERATION
                if two_seat_collect_steps_per_iteration is None
                else int(two_seat_collect_steps_per_iteration)
            ),
            "updates_per_iteration": (
                DEFAULT_TWO_SEAT_UPDATES_PER_ITERATION
                if two_seat_updates_per_iteration is None
                else int(two_seat_updates_per_iteration)
            ),
            "num_simulations": num_simulations,
            "learner_updates": two_seat_learner_updates,
            "allow_optimizer_step": two_seat_allow_optimizer_step,
            "replay_scope": two_seat_replay_scope,
            "learner_sample_size": two_seat_learner_sample_size,
            "max_replay_rows": two_seat_max_replay_rows,
            "record_log_limit": two_seat_record_log_limit,
            "replay_row_log_limit": two_seat_replay_row_log_limit,
            "max_ticks": two_seat_max_ticks,
            "death_mode": two_seat_death_mode,
            "natural_bonus_spawn": two_seat_natural_bonus_spawn,
            "decision_ms": decision_ms,
            "alive_reward": two_seat_alive_reward,
            "dead_reward": two_seat_dead_reward,
            "terminal_outcome_reward_per_step": (
                two_seat_terminal_outcome_reward_per_step
            ),
            "bonus_pickup_reward_per_catch": two_seat_bonus_pickup_reward_per_catch,
            "return_target_discount": two_seat_return_target_discount,
            "action_selection_mode": two_seat_action_selection_mode,
            "collect_temperature": two_seat_collect_temperature,
            "collect_epsilon": two_seat_collect_epsilon,
            "action_noop_probability": two_seat_action_noop_probability,
            "action_noop_warmup_iterations": two_seat_action_noop_warmup_iterations,
            "policy_action_repeat_min": two_seat_policy_action_repeat_min,
            "policy_action_repeat_max": two_seat_policy_action_repeat_max,
            "policy_action_repeat_extra_probability": (
                two_seat_policy_action_repeat_extra_probability
            ),
            "policy_action_repeat_warmup_iterations": (
                two_seat_policy_action_repeat_warmup_iterations
            ),
            "observation_noise_std": two_seat_observation_noise_std,
            "trail_render_mode": two_seat_trail_render_mode,
            "learning_rate": two_seat_learning_rate,
            "checkpoint_every_iterations": (
                save_ckpt_after_iter
                if two_seat_checkpoint_every_iterations is None
                else int(two_seat_checkpoint_every_iterations)
            ),
            "save_initial_checkpoint": two_seat_save_initial_checkpoint,
            "progress_every_iterations": two_seat_progress_every_iterations,
            "progress_commit_every_iterations": two_seat_progress_commit_every_iterations,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "gif_browser_run_marker_enabled": bool(background_gif_enabled),
        }
        background = _two_seat_background_eval_config(
            payload=two_seat_payload,
            background_eval_enabled=background_eval_enabled,
            background_eval_launch_kind=background_eval_launch_kind,
            background_eval_compute=background_eval_compute,
            background_eval_id_prefix=background_eval_id_prefix,
            background_eval_seed_count=background_eval_seed_count,
            background_eval_seed_rng_seed=background_eval_seed_rng_seed,
            background_eval_max_steps=background_eval_max_steps,
            background_eval_step_detail_limit=background_eval_step_detail_limit,
            background_eval_num_simulations=background_eval_num_simulations,
            background_eval_batch_size=background_eval_batch_size,
            background_eval_poll_interval_sec=background_eval_poll_interval_sec,
            background_eval_poll_stable_polls=background_eval_poll_stable_polls,
            background_eval_poller_max_runtime_sec=background_eval_poller_max_runtime_sec,
            background_eval_poller_idle_after_done_sec=(
                background_eval_poller_idle_after_done_sec
            ),
            background_gif_enabled=background_gif_enabled,
            background_gif_seed_offset=background_gif_seed_offset,
            background_gif_max_steps=background_gif_max_steps,
            background_gif_frame_stride=background_gif_frame_stride,
            background_gif_fps=background_gif_fps,
            background_gif_scale=background_gif_scale,
            background_gif_frame_size=background_gif_frame_size,
        )
        poller_call = None
        poller_call_id = None
        if background["enabled"]:
            poller_call, poller_call_id = _spawn_two_seat_checkpoint_poller(
                payload=two_seat_payload,
                background=background,
            )
        if not wait_for_train:
            call = two_seat_fn.spawn(two_seat_payload)
            call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
            print(
                json.dumps(
                    {
                        "schema_id": "curvyzero_canonical_two_seat_selfplay_background_launch/v0",
                        "status": "spawned",
                        "mode": TWO_SEAT_SELFPLAY_MODE,
                        "compute": compute,
                        "seed": seed,
                        "refs": _two_seat_call_refs(run_id, attempt_id),
                        "function_call_id": call_id,
                        "background_eval": {
                            **background,
                            "poller_function_call_id": poller_call_id,
                            "status_ref": (
                                runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
                                / "checkpoint_eval_poller.json"
                            ).as_posix()
                            if poller_call_id
                            else None,
                        },
                        "command": two_seat_payload,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        result = two_seat_fn.remote(two_seat_payload)
        if poller_call is not None:
            result = {
                "train": result,
                "background_eval": {
                    **background,
                    "poller_function_call_id": poller_call_id,
                    "poller": poller_call.get(),
                },
            }
        if output_detail == OUTPUT_DETAIL_COMPACT and isinstance(result, dict):
            train_result = result.get("train") if "train" in result else result
            if isinstance(train_result, dict):
                compact = compact_curvytron_two_seat_lightzero_train_smoke_summary(
                    train_result
                )
                if "background_eval" in result:
                    compact["background_eval"] = result["background_eval"]
                result = compact
        print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
        return
    if compute == COMPUTE_CPU:
        train_fn = lightzero_curvytron_visual_survival_cpu
    elif compute == COMPUTE_CPU64:
        train_fn = lightzero_curvytron_visual_survival_cpu64
    elif compute == COMPUTE_GPU_L4_T4:
        train_fn = lightzero_curvytron_visual_survival_gpu
    elif compute == COMPUTE_GPU_L4_T4_CPU40:
        train_fn = lightzero_curvytron_visual_survival_gpu_cpu40
    elif compute == COMPUTE_GPU_H100_CPU40:
        train_fn = lightzero_curvytron_visual_survival_h100_cpu40
    elif compute == COMPUTE_GPU_H100X2_CPU40:
        train_fn = lightzero_curvytron_visual_survival_h100x2_cpu40
    else:
        raise ValueError(f"unknown compute {compute!r}; expected one of {COMPUTE_CHOICES!r}")
    opponent_policy_kind = _normalize_opponent_policy_kind_for_env(
        env_variant=env_variant,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
    )
    kwargs = {
        "mode": mode,
        "seed": seed,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "max_env_step": max_env_step,
        "max_train_iter": max_train_iter,
        "source_max_steps": source_max_steps,
        "decision_ms": decision_ms,
        "collector_env_num": collector_env_num,
        "evaluator_env_num": evaluator_env_num,
        "n_evaluator_episode": n_evaluator_episode,
        "n_episode": n_episode,
        "num_simulations": num_simulations,
        "batch_size": batch_size,
        "lightzero_eval_freq": lightzero_eval_freq,
        "skip_lightzero_eval_in_profile": skip_lightzero_eval_in_profile,
        "profile_cuda_sync_enabled": profile_cuda_sync_enabled,
        "profile_allow_auto_resume": profile_allow_auto_resume,
        "lightzero_multi_gpu": lightzero_multi_gpu,
        "save_ckpt_after_iter": save_ckpt_after_iter,
        "stop_after_learner_train_calls": stop_after_learner_train_calls,
        "env_variant": env_variant,
        "reward_variant": reward_variant,
        "ego_action_straight_override_probability": ego_action_straight_override_probability,
        "control_noise_profile_id": control_noise_profile_id,
        "disable_death_for_profile": disable_death_for_profile,
        "env_telemetry_stride": env_telemetry_stride,
        "env_manager_type": env_manager_type,
        "opponent_policy_kind": opponent_policy_kind,
        "opponent_checkpoint_ref": opponent_checkpoint_ref,
        "opponent_snapshot_ref": snapshot_ref,
        "opponent_checkpoint_report_ref": checkpoint_ref,
        "opponent_checkpoint_state_key": state_key,
        "background_eval_enabled": background_eval_enabled,
        "background_eval_launch_kind": background_eval_launch_kind,
        "background_eval_compute": background_eval_compute,
        "background_eval_id_prefix": background_eval_id_prefix,
        "background_eval_seed_count": background_eval_seed_count,
        "background_eval_seed_rng_seed": background_eval_seed_rng_seed,
        "background_eval_max_steps": background_eval_max_steps,
        "background_eval_step_detail_limit": background_eval_step_detail_limit,
        "background_eval_num_simulations": background_eval_num_simulations,
        "background_eval_batch_size": background_eval_batch_size,
        "background_gif_enabled": background_gif_enabled,
        "background_gif_seed_offset": background_gif_seed_offset,
        "background_gif_max_steps": background_gif_max_steps,
        "background_gif_frame_stride": background_gif_frame_stride,
        "background_gif_fps": background_gif_fps,
        "background_gif_scale": background_gif_scale,
        "background_gif_frame_size": background_gif_frame_size,
    }
    exp_name_ref = (runs.attempt_train_ref(TASK_ID, run_id, attempt_id) / "lightzero_exp").as_posix()
    poller_call = None
    poller_call_id = None
    train_kwargs = dict(kwargs)
    if mode == "train" and background_eval_enabled and background_eval_launch_kind == BACKGROUND_EVAL_LAUNCH_POLLER:
        train_kwargs["background_eval_launch_kind"] = BACKGROUND_EVAL_LAUNCH_POLLER
        poller_call = lightzero_curvytron_visual_survival_checkpoint_eval_poller.spawn(
            run_id=run_id,
            attempt_id=attempt_id,
            exp_name_ref=exp_name_ref,
            seed=seed,
            source_max_steps=source_max_steps,
            env_variant=env_variant,
            reward_variant=reward_variant,
            opponent_policy_kind=opponent_policy_kind,
            opponent_checkpoint_ref=opponent_checkpoint_ref,
            opponent_snapshot_ref=snapshot_ref,
            opponent_checkpoint_state_key=state_key,
            background_eval_compute=background_eval_compute,
            background_eval_id_prefix=background_eval_id_prefix,
            background_eval_seed_count=background_eval_seed_count,
            background_eval_seed_rng_seed=background_eval_seed_rng_seed,
            background_eval_max_steps=background_eval_max_steps,
            background_eval_step_detail_limit=background_eval_step_detail_limit,
            background_eval_num_simulations=background_eval_num_simulations,
            background_eval_batch_size=background_eval_batch_size,
            background_gif_enabled=background_gif_enabled,
            background_gif_seed_offset=background_gif_seed_offset,
            background_gif_max_steps=background_gif_max_steps,
            background_gif_frame_stride=background_gif_frame_stride,
            background_gif_fps=background_gif_fps,
            background_gif_scale=background_gif_scale,
            background_gif_frame_size=background_gif_frame_size,
            poll_interval_sec=background_eval_poll_interval_sec,
            stable_polls=background_eval_poll_stable_polls,
            max_runtime_sec=background_eval_poller_max_runtime_sec,
            idle_after_train_done_sec=background_eval_poller_idle_after_done_sec,
        )
        poller_call_id = (
            getattr(poller_call, "object_id", None) or getattr(poller_call, "id", None)
        )
    if mode == "train" and not wait_for_train:
        call = train_fn.spawn(**train_kwargs)
        call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
        print(
            json.dumps(
                {
                    "schema_id": "curvyzero_lightzero_curvytron_visual_survival_background_launch/v0",
                    "status": "spawned",
                    "mode": mode,
                    "compute": compute,
                    "seed": seed,
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "function_call_id": call_id,
                    "background_eval": {
                        "enabled": background_eval_enabled,
                        "launch_kind": background_eval_launch_kind,
                        "poller_function_call_id": poller_call_id,
                        "selfplay_gif_enabled": background_gif_enabled,
                        "status_ref": (
                            runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
                            / "checkpoint_eval_poller.json"
                        ).as_posix()
                        if poller_call_id
                        else None,
                    },
                    "summary_ref": (
                        runs.attempt_train_ref(TASK_ID, run_id, attempt_id) / "summary.json"
                    ).as_posix(),
                    "action_observability_ref": (
                        runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
                        / "action_observability.json"
                    ).as_posix(),
                    "command": kwargs,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = train_fn.remote(**train_kwargs)
    if poller_call is not None:
        result = {
            "train": result,
            "background_eval": {
                "enabled": True,
                "launch_kind": background_eval_launch_kind,
                "poller_function_call_id": poller_call_id,
                "selfplay_gif_enabled": background_gif_enabled,
                "poller": poller_call.get(),
            },
        }
    if output_detail == OUTPUT_DETAIL_COMPACT:
        result = _compact_train_result_for_output(result)
    print(json.dumps(result, indent=2, sort_keys=True))
