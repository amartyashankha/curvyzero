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
import tempfile
import threading
import time
import traceback
from collections import Counter
from functools import lru_cache
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

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
from curvyzero.contracts.curvytron import (
    CURVYTRON_DEFAULT_COLLECTOR_ENV_NUM,
    CURVYTRON_DEFAULT_MAX_ENV_STEP,
    CURVYTRON_DEFAULT_MAX_TRAIN_ITER,
    CURVYTRON_DEFAULT_N_EPISODE,
    CURVYTRON_DEFAULT_NUM_SIMULATIONS,
    CURVYTRON_DEFAULT_TRAIN_BATCH_SIZE,
    CURVYTRON_DEFAULT_TRAIN_COMPUTE,
    CURVYTRON_BACKGROUND_GIF_FPS,
    CURVYTRON_COMMIT_ON_CHECKPOINT,
    CURVYTRON_SAVE_CKPT_AFTER_ITER,
    CURVYTRON_SOURCE_MAX_STEPS,
    CURVYTRON_TRAINING_TASK_ID,
    curvytron_control_volume_name,
    curvytron_runs_volume_name,
    curvytron_train_app_name,
    modal_volume_kwargs_for_name,
)
from curvyzero.infra.modal import run_management as runs
from curvyzero.observability.feedback_loop_lineage import (
    append_lineage_event,
    lineage_events_path,
)
from curvyzero.training import lightzero_config_builder as lz_config
from curvyzero.training import lightzero_checkpoints as lz_checkpoints
from curvyzero.training import exploration_bonus as xb
from curvyzero.training.reward_contracts import (
    DEFAULT_MODEL_SUPPORT_CAP,
    DEFAULT_REWARD_OUTCOME_ALPHA,
    DEFAULT_REWARD_VARIANT,
    DEFAULT_TD_STEPS,
    REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC,
    REWARD_VARIANT_AUTO,
    REWARD_VARIANT_CHOICES,
    REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME,
    REWARD_VARIANT_SPARSE_OUTCOME,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
    all_players_alive_diagnostic_reward_policy,
    lightzero_target_config_for_reward,
    normalize_reward_outcome_alpha,
    normalize_reward_variant_for_env,
    reward_policy_for_variant,
)
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
    DEFAULT_DECISION_SOURCE_FRAMES as SOURCE_STATE_DEFAULT_DECISION_SOURCE_FRAMES,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    DEFAULT_DECISION_MS as SOURCE_STATE_DEFAULT_DECISION_MS,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY as SOURCE_STATE_DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    DEFAULT_POLICY_ACTION_REPEAT_MAX as SOURCE_STATE_DEFAULT_POLICY_ACTION_REPEAT_MAX,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    DEFAULT_POLICY_ACTION_REPEAT_MIN as SOURCE_STATE_DEFAULT_POLICY_ACTION_REPEAT_MIN,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    JOINT_ACTION_COUNT as SOURCE_STATE_JOINT_ACTION_COUNT,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    OPPONENT_DEATH_MODE_IMMORTAL,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    OPPONENT_DEATH_MODE_NORMAL,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    OPPONENT_DEATH_MODES,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    OPPONENT_RUNTIME_MODE_NORMAL,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    OPPONENT_RUNTIME_MODES,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    OPPONENT_TRAINING_RELATION_PROACTIVE_WALL_AVOIDANT,
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
from curvyzero.training.opponent_mixture import (
    OPPONENT_MIXTURE_SCHEMA_ID,
    deterministic_collector_env_mixture_plan,
    parse_opponent_mixture_spec,
    singleton_mixture_for_split_entry,
)
from curvyzero.training.opponent_leaderboard import validate_assignment_audit
from curvyzero.training.opponent_registry import (
    OPPONENT_ASSIGNMENT_SCHEMA_ID,
    canonical_assignment_json_sha256,
    parse_opponent_assignment_snapshot,
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
    SOURCE_STATE_SUPPORTED_BONUS_RENDER_MODES,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
)
from curvyzero.env.observation_surface_contract import (
    DEFAULT_POLICY_OBSERVATION_BACKEND,
    POLICY_BONUS_RENDER_MODE,
    POLICY_OBSERVATION_BACKENDS,
    POLICY_OBSERVATION_CONTRACT_ID,
    POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID,
    POLICY_TRAIL_RENDER_MODE,
    policy_observation_surface,
)
from curvyzero.infra.modal.mctx_dependency_smoke import JAX_VERSION
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    STACKED_SOURCE_STATE_GRAY64_SHAPE,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    LEARNER_SEAT_MODE_RANDOM_PER_EPISODE,
)
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
    LEARNER_SEAT_MODES,
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
    DEFAULT_FROZEN_OPPONENT_PLAYER_ID as TWO_SEAT_DEFAULT_FROZEN_OPPONENT_PLAYER_ID,
)
from curvyzero.training.curvytron_two_seat_lightzero_train_smoke import (
    DEFAULT_FROZEN_OPPONENT_PROBABILITY as TWO_SEAT_DEFAULT_FROZEN_OPPONENT_PROBABILITY,
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


APP_NAME = curvytron_train_app_name()
TASK_ID = CURVYTRON_TRAINING_TASK_ID
VOLUME_NAME = curvytron_runs_volume_name()
CONTROL_VOLUME_NAME = curvytron_control_volume_name()
TRAIL_RENDER_MODE_BODY_CIRCLES_FAST = _TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE = POLICY_BONUS_RENDER_MODE
SOURCE_STATE_BONUS_RENDER_MODE_CHOICES = tuple(SOURCE_STATE_SUPPORTED_BONUS_RENDER_MODES)
DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE = POLICY_TRAIL_RENDER_MODE
SOURCE_STATE_TRAIL_RENDER_MODE_CHOICES = tuple(SOURCE_STATE_SUPPORTED_TRAIL_RENDER_MODES)
DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE = DEFAULT_POLICY_OBSERVATION_BACKEND
POLICY_OBSERVATION_BACKEND_CHOICES = tuple(POLICY_OBSERVATION_BACKENDS)
DEFAULT_LEARNER_SEAT_MODE = LEARNER_SEAT_MODE_RANDOM_PER_EPISODE
LEARNER_SEAT_MODE_CHOICES = tuple(LEARNER_SEAT_MODES)
LIGHTZERO_VERSION = "0.2.0"
REMOTE_ROOT = Path("/repo")
RUNS_MOUNT = Path("/runs")
CONTROL_MOUNT = Path("/control")
_RUNS_VOLUME_RELOAD_LOCK = threading.Lock()
_CONTROL_VOLUME_RELOAD_LOCK = threading.Lock()
_RUNS_REF_PREFIX = "runs:"
_CONTROL_REF_PREFIX = "control:"

DEFAULT_MODE = "dry"
DEFAULT_COMPUTE = CURVYTRON_DEFAULT_TRAIN_COMPUTE
DEFAULT_SEED = 0
DEFAULT_MAX_ENV_STEP = CURVYTRON_DEFAULT_MAX_ENV_STEP
DEFAULT_MAX_TRAIN_ITER = CURVYTRON_DEFAULT_MAX_TRAIN_ITER
DEFAULT_SOURCE_MAX_STEPS = CURVYTRON_SOURCE_MAX_STEPS
DEFAULT_COLLECTOR_ENV_NUM = CURVYTRON_DEFAULT_COLLECTOR_ENV_NUM
DEFAULT_EVALUATOR_ENV_NUM = 1
DEFAULT_N_EVALUATOR_EPISODE = 1
DEFAULT_N_EPISODE = CURVYTRON_DEFAULT_N_EPISODE
DEFAULT_NUM_SIMULATIONS = CURVYTRON_DEFAULT_NUM_SIMULATIONS
DEFAULT_BATCH_SIZE = CURVYTRON_DEFAULT_TRAIN_BATCH_SIZE
DEFAULT_LIGHTZERO_EVAL_FREQ = 0
DEFAULT_SKIP_LIGHTZERO_EVAL_IN_PROFILE = True
DEFAULT_LIGHTZERO_MULTI_GPU = False
DEFAULT_PROFILE_CUDA_SYNC_ENABLED = False
DEFAULT_PROFILE_ALLOW_AUTO_RESUME = False
DEFAULT_PROFILE_VOLUME_COMMIT = False
DEFAULT_PROFILE_SPAWN = False
# Checkpoint cadence also controls automatic checkpoint eval/inspection/GIF work.
# Keep this frequent enough to observe progress but not every loop.
DEFAULT_SAVE_CKPT_AFTER_ITER = CURVYTRON_SAVE_CKPT_AFTER_ITER
DEFAULT_STOP_AFTER_LEARNER_TRAIN_CALLS = 0
DEFAULT_COMMIT_ON_CHECKPOINT = CURVYTRON_COMMIT_ON_CHECKPOINT
DEFAULT_OPPONENT_ASSIGNMENT_REFRESH_INTERVAL_TRAIN_ITER = 0
OPPONENT_ASSIGNMENT_REFRESH_POINTER_SCHEMA_ID = "curvyzero_opponent_assignment_refresh_pointer/v0"
DEFAULT_OWN_CHECKPOINT_OPPONENT_REFRESH_ENABLED = False
DEFAULT_DECISION_SOURCE_FRAMES = SOURCE_STATE_DEFAULT_DECISION_SOURCE_FRAMES
DEFAULT_DECISION_MS = SOURCE_STATE_DEFAULT_DECISION_MS
DEFAULT_SOURCE_PHYSICS_STEP_MS = DEFAULT_DECISION_MS / DEFAULT_DECISION_SOURCE_FRAMES
DEFAULT_ENV_TELEMETRY_STRIDE = 1
DEFAULT_ENV_MANAGER_TYPE = "subprocess"
CURVYZERO_BATCHED_PROFILE_ENV_MANAGER_TYPE = "curvyzero_batched_profile"
CURVYZERO_BATCHED_ZERO_OBS_PROFILE_ENV_MANAGER_TYPE = "curvyzero_batched_zero_obs_profile"
CURVYZERO_BATCHED_PROFILE_ENV_MANAGER_TYPES = (
    CURVYZERO_BATCHED_PROFILE_ENV_MANAGER_TYPE,
    CURVYZERO_BATCHED_ZERO_OBS_PROFILE_ENV_MANAGER_TYPE,
)
ENV_MANAGER_TYPE_CHOICES = (
    "base",
    "subprocess",
    *CURVYZERO_BATCHED_PROFILE_ENV_MANAGER_TYPES,
)
COLLECT_SEARCH_BACKEND_STOCK = "stock"
COLLECT_SEARCH_BACKEND_DIRECT_CTREE_GPU_LATENT = "direct_ctree_gpu_latent"
COLLECT_SEARCH_BACKEND_CHOICES = (
    COLLECT_SEARCH_BACKEND_STOCK,
    COLLECT_SEARCH_BACKEND_DIRECT_CTREE_GPU_LATENT,
)
DEFAULT_COLLECT_SEARCH_BACKEND = COLLECT_SEARCH_BACKEND_STOCK
COLLECT_SEARCH_CTREE_BACKEND_LIGHTZERO = "lightzero"
COLLECT_SEARCH_CTREE_BACKEND_FLAT_A3 = "flat_a3"
COLLECT_SEARCH_CTREE_BACKEND_CHOICES = (
    COLLECT_SEARCH_CTREE_BACKEND_LIGHTZERO,
    COLLECT_SEARCH_CTREE_BACKEND_FLAT_A3,
)
DEFAULT_COLLECT_SEARCH_CTREE_BACKEND = COLLECT_SEARCH_CTREE_BACKEND_LIGHTZERO
DEFAULT_RUN_ID = "curvytron-stock-loop-control-s0"
DEFAULT_ATTEMPT_ID = "stock-loop-control"
ENV_VARIANT_FIXED_OPPONENT = lz_config.ENV_VARIANT_FIXED_OPPONENT
ENV_VARIANT_TURN_COMMIT = lz_config.ENV_VARIANT_TURN_COMMIT
ENV_VARIANT_SOURCE_STATE_TURN_COMMIT = lz_config.ENV_VARIANT_SOURCE_STATE_TURN_COMMIT
ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT = lz_config.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT
ENV_VARIANT_SOURCE_STATE_JOINT_ACTION = lz_config.ENV_VARIANT_SOURCE_STATE_JOINT_ACTION
ENV_VARIANT_CHOICES = lz_config.ENV_VARIANT_CHOICES
DEFAULT_ENV_VARIANT = lz_config.DEFAULT_ENV_VARIANT
OPPONENT_TRAINING_RELATION_WEIGHTED_EPISODE_MIXTURE = (
    lz_config.OPPONENT_TRAINING_RELATION_WEIGHTED_EPISODE_MIXTURE
)
DEFAULT_EGO_ACTION_STRAIGHT_OVERRIDE_PROBABILITY = 0.0
DEFAULT_POLICY_ACTION_REPEAT_MIN = SOURCE_STATE_DEFAULT_POLICY_ACTION_REPEAT_MIN
DEFAULT_POLICY_ACTION_REPEAT_MAX = SOURCE_STATE_DEFAULT_POLICY_ACTION_REPEAT_MAX
DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY = (
    SOURCE_STATE_DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
)
DEFAULT_CONTROL_NOISE_PROFILE_ID = "none"
DEFAULT_DISABLE_DEATH_FOR_PROFILE = False
DEFAULT_OPPONENT_DEATH_MODE = OPPONENT_DEATH_MODE_NORMAL
DEFAULT_OPPONENT_RUNTIME_MODE = OPPONENT_RUNTIME_MODE_NORMAL
OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION = "none_centralized_joint_action"
DEFAULT_OPPONENT_POLICY_KIND = OPPONENT_POLICY_KIND_FIXED_STRAIGHT
DEFAULT_OPPONENT_USE_CUDA = False
INITIAL_POLICY_CHECKPOINT_LOAD_MODE_STRICT = "strict"
INITIAL_POLICY_CHECKPOINT_LOAD_MODE_MATCHING_SHAPE = "matching_shape"
DEFAULT_INITIAL_POLICY_CHECKPOINT_LOAD_MODE = INITIAL_POLICY_CHECKPOINT_LOAD_MODE_MATCHING_SHAPE
INITIAL_POLICY_CHECKPOINT_LOAD_MODE_CHOICES = (
    INITIAL_POLICY_CHECKPOINT_LOAD_MODE_STRICT,
    INITIAL_POLICY_CHECKPOINT_LOAD_MODE_MATCHING_SHAPE,
)
INITIAL_POLICY_MODEL_ONLY_OPTIMIZER_MARKER = "curvyzero_initial_policy_model_only_seed/v1"
OPPONENT_POLICY_KIND_CHOICES = (
    OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
    OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
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
DEFAULT_BACKGROUND_GIF_FPS = CURVYTRON_BACKGROUND_GIF_FPS
DEFAULT_BACKGROUND_GIF_MIN_FRAME_DURATION_MS = 10
DEFAULT_BACKGROUND_GIF_SCALE = 4
CHECKPOINT_SELFPLAY_GIF_FRAME_SIZE = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
DEFAULT_BACKGROUND_GIF_FRAME_SIZE = CHECKPOINT_SELFPLAY_GIF_FRAME_SIZE
BACKGROUND_GIF_POLICY_MODE_COLLECT = "collect"
BACKGROUND_GIF_POLICY_MODE_EVAL_GREEDY = "eval_greedy"
BACKGROUND_GIF_POLICY_MODE_CHOICES = (
    BACKGROUND_GIF_POLICY_MODE_COLLECT,
    BACKGROUND_GIF_POLICY_MODE_EVAL_GREEDY,
)
DEFAULT_BACKGROUND_GIF_POLICY_MODE = BACKGROUND_GIF_POLICY_MODE_EVAL_GREEDY
DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE = 1.0
DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON = 0.25
BACKGROUND_GIF_COLLECT_SETTINGS_SOURCE_TRAINING = "training_collect_settings"
BACKGROUND_GIF_COLLECT_SETTINGS_SOURCE_OVERRIDE = "background_gif_override"
BACKGROUND_GIF_VARIANT_EVAL_GREEDY = "eval_greedy"
BACKGROUND_GIF_VARIANT_COLLECT_T1 = "collect_t1"
BACKGROUND_GIF_VARIANT_FILENAMES = {
    BACKGROUND_GIF_VARIANT_EVAL_GREEDY: "raw.gif",
    BACKGROUND_GIF_VARIANT_COLLECT_T1: "collect_t1.gif",
}
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
DEFAULT_BACKGROUND_EVAL_POLLER_MAX_RUNTIME_SEC = 18 * 60 * 60
DEFAULT_BACKGROUND_EVAL_POLLER_IDLE_AFTER_DONE_SEC = 60.0
DEFAULT_STOCK_TRAIN_MODAL_TIMEOUT_SEC = 16 * 60 * 60
DEFAULT_BACKGROUND_EVAL_POLLER_MODAL_TIMEOUT_SEC = 20 * 60 * 60
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
    return normalize_reward_variant_for_env(
        env_variant=env_variant,
        reward_variant=reward_variant,
    )


def _validate_source_state_trail_render_mode(value: str) -> str:
    return lz_config.validate_source_state_trail_render_mode(value)


def _validate_source_state_bonus_render_mode(value: str) -> str:
    return lz_config.validate_source_state_bonus_render_mode(value)


def _validate_policy_observation_backend(value: str) -> str:
    return lz_config.validate_policy_observation_backend(value)


def _validate_collect_search_backend(value: str) -> str:
    if value not in COLLECT_SEARCH_BACKEND_CHOICES:
        raise ValueError(
            "collect_search_backend must be one of "
            f"{COLLECT_SEARCH_BACKEND_CHOICES!r}; got {value!r}"
        )
    return value


def _validate_collect_search_ctree_backend(value: str) -> str:
    if value not in COLLECT_SEARCH_CTREE_BACKEND_CHOICES:
        raise ValueError(
            "collect_search_ctree_backend must be one of "
            f"{COLLECT_SEARCH_CTREE_BACKEND_CHOICES!r}; got {value!r}"
        )
    return value


def _validate_learner_seat_mode(
    value: str | None,
    *,
    absent_default: str = DEFAULT_LEARNER_SEAT_MODE,
) -> str:
    return lz_config.validate_learner_seat_mode(value, absent_default=absent_default)


def _normalize_reward_outcome_alpha(value: float) -> float:
    return normalize_reward_outcome_alpha(value)


def _reward_policy_for_variant(
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
    reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
    model_support_cap: int | None = DEFAULT_MODEL_SUPPORT_CAP,
    td_steps: int | None = DEFAULT_TD_STEPS,
) -> dict[str, Any]:
    return lightzero_target_config_for_reward(
        env_variant=env_variant,
        reward_variant=reward_variant,
        source_max_steps=source_max_steps,
        reward_outcome_alpha=reward_outcome_alpha,
        model_support_cap=model_support_cap,
        td_steps=td_steps,
    )


def _validate_trusted_source_state_action_cadence(
    *,
    env_variant: str,
    decision_ms: float,
    decision_source_frames: int = DEFAULT_DECISION_SOURCE_FRAMES,
    source_physics_step_ms: float = DEFAULT_SOURCE_PHYSICS_STEP_MS,
    source_max_steps_semantics: str = "source_physics_steps",
    context: str,
) -> None:
    return lz_config.validate_trusted_source_state_action_cadence(
        env_variant=env_variant,
        decision_ms=decision_ms,
        decision_source_frames=decision_source_frames,
        source_physics_step_ms=source_physics_step_ms,
        source_max_steps_semantics=source_max_steps_semantics,
        context=context,
    )


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
    "evaluator_eval_skipped_calls",
    "env_reset_calls",
    "env_step_calls",
    "env_vector_step_calls",
    "env_runtime_step_many_calls",
    "env_registered_step_calls",
    "env_step_info_calls",
    "env_base_info_calls",
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
    "collect_search_backend_direct_ctree_gpu_latent_calls",
    "collect_search_backend_fallback_calls",
    "collect_search_backend_output_fast_path_calls",
    "collect_search_backend_output_rows",
    "collect_search_backend_ctree_traverse_calls",
    "collect_search_backend_ctree_backpropagate_calls",
    "collect_search_backend_recurrent_inference_calls",
    "collect_search_backend_model_output_d2h_bytes",
    "rnd_collect_data_calls",
    "rnd_train_with_data_calls",
    "rnd_estimate_calls",
    "rnd_metrics_snapshot_calls",
    "rnd_metrics_write_snapshot_calls",
    "rnd_state_hash_calls",
)


def _compute_uses_cuda(compute: str) -> bool:
    return compute.startswith("gpu-")


def _import_collect_search_tree_muzero(ctree_backend: str) -> Any:
    if ctree_backend == COLLECT_SEARCH_CTREE_BACKEND_LIGHTZERO:
        import lzero.mcts.tree_search.mcts_ctree as mcts_ctree

        return mcts_ctree.tree_muzero
    if ctree_backend != COLLECT_SEARCH_CTREE_BACKEND_FLAT_A3:
        raise ValueError(f"unsupported collect_search_ctree_backend {ctree_backend!r}")

    def prefer_repo_curvyzero_path() -> None:
        import importlib
        import sys

        repo_src = str(REMOTE_ROOT / "src")
        if repo_src in sys.path:
            sys.path.remove(repo_src)
        sys.path.insert(0, repo_src)

        package_paths = {
            "curvyzero": REMOTE_ROOT / "src" / "curvyzero",
            "curvyzero.vendor": REMOTE_ROOT / "src" / "curvyzero" / "vendor",
            "curvyzero.vendor.lightzero_ctree_a3": (
                REMOTE_ROOT / "src" / "curvyzero" / "vendor" / "lightzero_ctree_a3"
            ),
            "curvyzero.vendor.lightzero_ctree_a3.ctree_muzero": (
                REMOTE_ROOT
                / "src"
                / "curvyzero"
                / "vendor"
                / "lightzero_ctree_a3"
                / "ctree_muzero"
            ),
        }
        for package_name, package_path in package_paths.items():
            module = sys.modules.get(package_name)
            if module is None:
                continue
            module_path = getattr(module, "__path__", None)
            if module_path is None:
                continue
            repo_package_path = str(package_path)
            if repo_package_path in module_path:
                module_path.remove(repo_package_path)
            module_path.insert(0, repo_package_path)
        importlib.invalidate_caches()

    def import_flat_a3() -> Any:
        from curvyzero.vendor.lightzero_ctree_a3.ctree_muzero import mz_tree_a3

        return mz_tree_a3

    prefer_repo_curvyzero_path()
    try:
        return import_flat_a3()
    except ImportError:
        prefer_repo_curvyzero_path()
        return import_flat_a3()


def _should_install_lightzero_phase_profile(
    *, mode: str, stop_after_learner_train_calls: int
) -> bool:
    return mode == "profile" or int(stop_after_learner_train_calls) > 0


def _collect_search_backend_proof(
    *,
    command: Mapping[str, Any],
    phase_profile: Mapping[str, Any],
) -> dict[str, Any]:
    counts = phase_profile.get("counts") if isinstance(phase_profile, Mapping) else {}
    samples = phase_profile.get("samples") if isinstance(phase_profile, Mapping) else {}
    counts = counts if isinstance(counts, Mapping) else {}
    samples = samples if isinstance(samples, Mapping) else {}
    observed_backends = sorted(
        {
            str(value)
            for value in samples.get("collect_search_backend", [])
            if value is not None
        }
    )
    observed_ctree_backends = sorted(
        {
            str(value)
            for value in samples.get("collect_search_ctree_backend", [])
            if value is not None
        }
    )
    return {
        "schema_id": "curvyzero_collect_search_backend_proof/v0",
        "requested_backend": command.get("collect_search_backend"),
        "requested_ctree_backend": command.get("collect_search_ctree_backend"),
        "fallback_policy": command.get("collect_search_backend_fallback_policy"),
        "observed_collect_search_backends": observed_backends,
        "observed_collect_search_ctree_backends": observed_ctree_backends,
        "direct_ctree_gpu_latent_calls": counts.get(
            "collect_search_backend_direct_ctree_gpu_latent_calls"
        ),
        "fallback_calls": counts.get("collect_search_backend_fallback_calls"),
        "output_rows": counts.get("collect_search_backend_output_rows"),
        "recurrent_inference_calls": counts.get(
            "collect_search_backend_recurrent_inference_calls"
        ),
        "model_output_d2h_bytes": counts.get("collect_search_backend_model_output_d2h_bytes"),
    }


def _validate_train_collect_search_backend_proof(
    *,
    command: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> list[str]:
    if command.get("mode") != "train":
        return []
    requested = command.get("collect_search_backend")
    if requested == COLLECT_SEARCH_BACKEND_STOCK:
        return []
    problems: list[str] = []
    if proof.get("fallback_policy") != "fail_closed_when_non_stock":
        problems.append("non-stock train collect_search_backend must fail closed on fallback")
    if proof.get("fallback_calls") != 0:
        problems.append("non-stock train collect_search_backend fallback was used")
    direct_calls = _compact_int(proof.get("direct_ctree_gpu_latent_calls"))
    if direct_calls is None or direct_calls <= 0:
        problems.append("non-stock train collect_search_backend was not observed")
    output_rows = _compact_int(proof.get("output_rows"))
    if output_rows is None or output_rows <= 0:
        problems.append("non-stock train collect_search_backend produced no output rows")
    observed = proof.get("observed_collect_search_backends")
    if not isinstance(observed, list) or str(requested) not in observed:
        problems.append("non-stock train collect_search_backend sample was not observed")
    requested_ctree = command.get("collect_search_ctree_backend")
    observed_ctree = proof.get("observed_collect_search_ctree_backends")
    if not isinstance(observed_ctree, list) or str(requested_ctree) not in observed_ctree:
        problems.append("non-stock train collect_search_ctree_backend sample was not observed")
    num_simulations = _compact_int(command.get("num_simulations"))
    if num_simulations is not None and num_simulations > 0:
        recurrent_calls = _compact_int(proof.get("recurrent_inference_calls"))
        if recurrent_calls is None or recurrent_calls <= 0:
            problems.append(
                "non-stock train collect_search_backend did not run recurrent inference"
            )
        d2h_bytes = _compact_int(proof.get("model_output_d2h_bytes"))
        if d2h_bytes is None or d2h_bytes <= 0:
            problems.append(
                "non-stock train collect_search_backend did not materialize model outputs"
            )
    return problems


LIGHTZERO_RESUME_STATE_DIRNAME = "lightzero_resume_state"
TARGET_AUDIT_MAX_SEGMENTS = 4
TARGET_AUDIT_MAX_STEPS_PER_SEGMENT = 8
TARGET_AUDIT_MAX_REPLAY_SAMPLES = 3
TORCH_CUDA12_VERSION = "2.8.0"
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
        f"jax[cuda12]=={JAX_VERSION}",
        f"torch=={TORCH_CUDA12_VERSION}",
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
ctree_a3_image = (
    image.uv_pip_install("Cython>=3")
    .add_local_file(
        Path.cwd() / "scripts/build_lightzero_ctree_a3.py",
        remote_path=str(REMOTE_ROOT / "scripts/build_lightzero_ctree_a3.py"),
        copy=True,
    )
    .run_commands(
        f"cd {REMOTE_ROOT} && python scripts/build_lightzero_ctree_a3.py build_ext --inplace"
    )
)
runs_volume = modal.Volume.from_name(
    VOLUME_NAME,
    **modal_volume_kwargs_for_name(VOLUME_NAME),
)
control_volume = modal.Volume.from_name(
    CONTROL_VOLUME_NAME,
    **modal_volume_kwargs_for_name(CONTROL_VOLUME_NAME),
)
TRAINER_VOLUMES = {str(RUNS_MOUNT): runs_volume, str(CONTROL_MOUNT): control_volume}
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
            call_key = f"{key[: -len('_batch_sum')]}_calls"
            call_count = self.counts.get(call_key, 0)
            if call_count:
                derived_stats[f"{key[: -len('_batch_sum')]}_batch_mean"] = float(value) / float(
                    call_count
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
            profiler.add_note(
                f"could not import source-state env hooks: {type(exc).__name__}: {exc}"
            )
            return
        env_cls = getattr(env_module, "CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv", None)
        if inspect.isclass(env_cls):
            patch_timed_method(env_cls, "reset", "env_reset_sec", "env_reset_calls")
            patch_timed_method(env_cls, "step", "env_step_sec", "env_step_calls")
            patch_timed_method(
                env_cls,
                "_step_info",
                "env_step_info_sec",
                "env_step_info_calls",
            )
            patch_timed_method(
                env_cls,
                "_base_info",
                "env_base_info_sec",
                "env_base_info_calls",
            )
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
        registered_env_cls = getattr(
            env_module,
            "CurvyZeroSourceStateVisualSurvivalLightZeroEnv",
            None,
        )
        if inspect.isclass(registered_env_cls):
            patch_timed_method(
                registered_env_cls,
                "step",
                "env_registered_step_sec",
                "env_registered_step_calls",
            )
        else:
            profiler.add_note("registered source-state env class missing for env hooks")

        try:
            visual_module = importlib.import_module("curvyzero.env.vector_visual_observation")
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            profiler.add_note(
                f"could not import visual observation hooks: {type(exc).__name__}: {exc}"
            )
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

    def patch_rnd_reward_model_hooks() -> None:
        rnd_cls = getattr(xb, "CurvyRNDRewardModel", None)
        if not inspect.isclass(rnd_cls):
            profiler.add_note("CurvyRNDRewardModel missing for RND hooks")
            return
        for method_name, timer_name, count_name, sync_cuda in (
            ("collect_data", "rnd_collect_data_sec", "rnd_collect_data_calls", False),
            ("train_with_data", "rnd_train_with_data_sec", "rnd_train_with_data_calls", True),
            ("estimate", "rnd_estimate_sec", "rnd_estimate_calls", True),
            ("metrics_snapshot", "rnd_metrics_snapshot_sec", "rnd_metrics_snapshot_calls", False),
            (
                "_write_metrics_snapshot",
                "rnd_metrics_write_snapshot_sec",
                "rnd_metrics_write_snapshot_calls",
                False,
            ),
            ("_state_hash", "rnd_state_hash_sec", "rnd_state_hash_calls", False),
        ):
            patch_timed_method(
                rnd_cls,
                method_name,
                timer_name,
                count_name,
                sync_cuda=sync_cuda,
            )

    def patch_policy_hooks() -> None:
        try:
            policy_module = importlib.import_module("lzero.policy.muzero")
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            profiler.add_note(
                f"could not import lzero.policy.muzero hooks: {type(exc).__name__}: {exc}"
            )
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

    def resolve_collector_class() -> Any | None:
        collector_cls = globals_map.get("Collector")
        if inspect.isclass(collector_cls):
            return collector_cls
        try:
            worker_module = importlib.import_module("lzero.worker")
            collector_cls = getattr(worker_module, "MuZeroCollector", None)
        except Exception as exc:  # pragma: no cover - remote diagnosis only.
            profiler.add_note(
                f"could not import lzero.worker.MuZeroCollector: {type(exc).__name__}: {exc}"
            )
            return None
        if inspect.isclass(collector_cls):
            profiler.add_note(
                "train_muzero globals did not expose Collector class; "
                "using lzero.worker.MuZeroCollector"
            )
            return collector_cls
        return None

    try:
        collector_cls = resolve_collector_class()
        if inspect.isclass(collector_cls):
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
                            "by explicit learner train call cap"
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
            profiler.add_note(
                f"could not import lzero.mcts for GameBuffer patch: {type(exc).__name__}: {exc}"
            )
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
        patch_rnd_reward_model_hooks()
    except Exception:
        restore()
        raise

    for hook in installed_hooks:
        profiler.add_installed_hook(hook)
    return restore


def _direct_ctree_gpu_latent_search_for_collect(
    *,
    policy: Any,
    mcts: Any,
    model: Any,
    latent_state_roots: Any,
    roots: Any,
    to_play: list[int],
    ctree_backend: str,
    profiler: _LightZeroPhaseProfiler,
) -> None:
    """Run LightZero CTree search while keeping MuZero latent states on device."""

    import numpy as np
    import torch
    from lzero.policy import mz_network_output_unpack

    if not hasattr(latent_state_roots, "device"):
        raise ValueError("direct_ctree_gpu_latent collect hook requires tensor latents")
    device = latent_state_roots.device
    if getattr(device, "type", None) != "cuda":
        raise ValueError("direct_ctree_gpu_latent collect hook requires CUDA latents")

    tree_muzero = _import_collect_search_tree_muzero(ctree_backend)
    batch_size = int(roots.num)
    cfg = getattr(mcts, "_cfg", getattr(policy, "_cfg", None))
    pb_c_base = int(getattr(cfg, "pb_c_base"))
    pb_c_init = float(getattr(cfg, "pb_c_init"))
    discount_factor = float(getattr(cfg, "discount_factor"))
    num_simulations = int(getattr(cfg, "num_simulations"))
    value_delta_max = float(getattr(cfg, "value_delta_max", 0.01))
    inverse_scalar_transform = getattr(mcts, "inverse_scalar_transform_handle", None)
    if inverse_scalar_transform is None:
        inverse_scalar_transform = getattr(policy, "inverse_scalar_transform_handle", None)
    if inverse_scalar_transform is None:
        raise ValueError("direct_ctree_gpu_latent collect hook missing inverse transform")

    latent_pool = torch.empty(
        (num_simulations + 1, batch_size)
        + tuple(int(dim) for dim in latent_state_roots.shape[1:]),
        dtype=latent_state_roots.dtype,
        device=device,
    )
    latent_pool[0].copy_(latent_state_roots)
    min_max_stats_lst = tree_muzero.MinMaxStatsList(batch_size)
    min_max_stats_lst.set_delta(value_delta_max)
    to_play_batch = list(to_play)
    env_type = str(getattr(cfg, "env_type", "not_board_games"))

    with torch.no_grad():
        model.eval()
        for simulation_index in range(num_simulations):
            results = tree_muzero.ResultsWrapper(num=batch_size)
            traverse_to_play = (
                to_play_batch if env_type == "not_board_games" else copy.deepcopy(to_play_batch)
            )
            with profiler.timer("collect_search_backend_ctree_traverse_sec"):
                (
                    latent_path_indices,
                    latent_batch_indices,
                    last_actions,
                    virtual_to_play_batch,
                ) = tree_muzero.batch_traverse(
                    roots,
                    pb_c_base,
                    pb_c_init,
                    discount_factor,
                    min_max_stats_lst,
                    results,
                    traverse_to_play,
                )
            profiler.add_count("collect_search_backend_ctree_traverse_calls")

            if not latent_path_indices:
                raise ValueError("direct_ctree_gpu_latent collect hook got no latent indices")
            with profiler.timer("collect_search_backend_tensor_index_sec"):
                path_tensor = torch.as_tensor(
                    latent_path_indices,
                    dtype=torch.long,
                    device=device,
                )
                batch_tensor = torch.as_tensor(
                    latent_batch_indices,
                    dtype=torch.long,
                    device=device,
                )
                latent_states = latent_pool[path_tensor, batch_tensor]
                last_actions_tensor = torch.as_tensor(
                    np.asarray(last_actions),
                    dtype=torch.long,
                    device=device,
                )

            with profiler.timer("collect_search_backend_recurrent_inference_sec"):
                network_output = model.recurrent_inference(latent_states, last_actions_tensor)
            profiler.add_count("collect_search_backend_recurrent_inference_calls")

            with profiler.timer("collect_search_backend_model_output_d2h_sec"):
                next_latent_state, reward, value, policy_logits = mz_network_output_unpack(
                    network_output
                )
                reward_plain = inverse_scalar_transform(reward).reshape(batch_size, 1)
                value_plain = inverse_scalar_transform(value).reshape(batch_size, 1)
                policy_logits_plain = policy_logits.to(dtype=torch.float32).reshape(
                    batch_size,
                    -1,
                )
                model_output_np = (
                    torch.cat(
                        (reward_plain, value_plain, policy_logits_plain),
                        dim=1,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                reward_np = model_output_np[:, 0]
                value_np = model_output_np[:, 1]
                policy_logits_np = model_output_np[:, 2:]
            profiler.add_count(
                "collect_search_backend_model_output_d2h_bytes",
                int(model_output_np.nbytes),
            )

            latent_pool[simulation_index + 1].copy_(next_latent_state)
            if ctree_backend == COLLECT_SEARCH_CTREE_BACKEND_FLAT_A3:
                with profiler.timer("collect_search_backend_flat_payload_sec"):
                    reward_batch = np.ascontiguousarray(
                        reward_np.reshape(-1),
                        dtype=np.float32,
                    )
                    value_batch = np.ascontiguousarray(
                        value_np.reshape(-1),
                        dtype=np.float32,
                    )
                    policy_logits_batch = np.ascontiguousarray(
                        policy_logits_np,
                        dtype=np.float32,
                    )
                with profiler.timer("collect_search_backend_ctree_backpropagate_sec"):
                    tree_muzero.batch_backpropagate_flat_a3(
                        simulation_index + 1,
                        discount_factor,
                        reward_batch,
                        value_batch,
                        policy_logits_batch,
                        min_max_stats_lst,
                        results,
                    )
            else:
                with profiler.timer("collect_search_backend_model_output_listify_sec"):
                    reward_batch = reward_np.reshape(-1).tolist()
                    value_batch = value_np.reshape(-1).tolist()
                    policy_logits_batch = policy_logits_np.tolist()
                with profiler.timer("collect_search_backend_ctree_backpropagate_sec"):
                    tree_muzero.batch_backpropagate(
                        simulation_index + 1,
                        discount_factor,
                        reward_batch,
                        value_batch,
                        policy_logits_batch,
                        min_max_stats_lst,
                        results,
                        virtual_to_play_batch,
                    )
            profiler.add_count("collect_search_backend_ctree_backpropagate_calls")


def _install_lightzero_collect_search_backend_hook(
    *,
    train_muzero: Any,
    backend: str,
    ctree_backend: str = DEFAULT_COLLECT_SEARCH_CTREE_BACKEND,
    profiler: _LightZeroPhaseProfiler,
    allow_fallback: bool = True,
) -> Any:
    """Install an alternate LightZero collect/search backend hook."""

    ctree_backend = _validate_collect_search_ctree_backend(ctree_backend)
    if backend == COLLECT_SEARCH_BACKEND_STOCK:
        return None
    if backend != COLLECT_SEARCH_BACKEND_DIRECT_CTREE_GPU_LATENT:
        raise ValueError(f"unsupported collect_search_backend {backend!r}")

    import numpy as np
    import torch
    from lzero.policy import mz_network_output_unpack, select_action

    policy_module = importlib.import_module("lzero.policy.muzero")
    policy_cls = getattr(policy_module, "MuZeroPolicy", None)
    if not inspect.isclass(policy_cls):
        raise RuntimeError("lzero.policy.muzero.MuZeroPolicy is missing")

    original_forward_collect = policy_cls._forward_collect

    def wrapped_forward_collect(
        self: Any,
        data: Any,
        action_mask: list | None = None,
        temperature: float = 1,
        to_play: list[int] = [-1],
        epsilon: float = 0.25,
        ready_env_id: Any = None,
        **kwargs: Any,
    ) -> dict[Any, Any]:
        cfg = getattr(self, "_cfg", None)
        model_cfg = getattr(cfg, "model", None)
        model_type = getattr(model_cfg, "model_type", None)

        def fallback(reason: str) -> dict[Any, Any]:
            profiler.add_sample("collect_search_backend_fallback_reason", reason, limit=80)
            profiler.add_count("collect_search_backend_fallback_calls")
            if not allow_fallback:
                raise RuntimeError(
                    "direct_ctree_gpu_latent collect hook refused hidden fallback: "
                    f"{reason}"
                )
            return original_forward_collect(
                self,
                data,
                action_mask,
                temperature,
                to_play,
                epsilon,
                ready_env_id=ready_env_id,
                **kwargs,
            )

        if model_type not in {"conv", "mlp"}:
            return fallback(f"unsupported_model_type:{model_type}")
        if bool(getattr(cfg, "collect_with_pure_policy", False)):
            return fallback("collect_with_pure_policy")
        if not bool(getattr(cfg, "mcts_ctree", False)):
            return fallback("mcts_ptree")
        if bool(getattr(cfg, "sampled_algo", False)) or bool(getattr(cfg, "gumbel_algo", False)):
            return fallback("sampled_or_gumbel_algo")
        if not hasattr(data, "shape"):
            return fallback("data_without_shape")

        active_collect_env_num = int(data.shape[0])
        if ready_env_id is None:
            ready_env_id = np.arange(active_collect_env_num)
        if len(ready_env_id) != active_collect_env_num:
            raise ValueError(
                "direct_ctree_gpu_latent collect hook expected ready_env_id length "
                f"{active_collect_env_num}, got {len(ready_env_id)}"
            )
        if action_mask is None:
            raise ValueError("direct_ctree_gpu_latent collect hook requires action_mask")
        if len(action_mask) != active_collect_env_num:
            raise ValueError(
                "direct_ctree_gpu_latent collect hook expected action_mask length "
                f"{active_collect_env_num}, got {len(action_mask)}"
            )

        legal_actions: list[list[int]] = []
        legal_action_arrays: list[Any] = []
        all_actions_legal = True
        action_count: int | None = None
        try:
            action_mask_matrix = np.asarray(action_mask, dtype=np.float32)
        except (TypeError, ValueError):
            action_mask_matrix = None
        if (
            action_mask_matrix is not None
            and action_mask_matrix.ndim == 2
            and action_mask_matrix.shape[0] == active_collect_env_num
        ):
            if not np.all((action_mask_matrix == 0.0) | (action_mask_matrix == 1.0)):
                raise ValueError(
                    "direct_ctree_gpu_latent collect hook requires binary action masks"
                )
            legal_counts = action_mask_matrix.sum(axis=1)
            if bool((legal_counts <= 0.0).any()):
                row_index = int(np.flatnonzero(legal_counts <= 0.0)[0])
                raise ValueError(
                    "direct_ctree_gpu_latent collect hook got a zero-action mask at "
                    f"row {row_index}"
                )
            action_count = int(action_mask_matrix.shape[1])
            all_actions_legal = bool(np.all(action_mask_matrix == 1.0))
            if all_actions_legal:
                shared_legal_array = np.arange(action_count, dtype=np.int64)
                shared_legal = shared_legal_array.astype(int).tolist()
                legal_action_arrays = [shared_legal_array] * active_collect_env_num
                legal_actions = [list(shared_legal) for _ in range(active_collect_env_num)]
            else:
                for row_index, mask in enumerate(action_mask_matrix):
                    legal_array = np.flatnonzero(mask == 1.0).astype(np.int64, copy=False)
                    legal_action_arrays.append(legal_array)
                    legal_actions.append([int(index) for index in legal_array])
        else:
            for row_index, row_mask in enumerate(action_mask):
                mask = np.asarray(row_mask, dtype=np.float32).reshape(-1)
                if not np.all((mask == 0.0) | (mask == 1.0)):
                    raise ValueError(
                        "direct_ctree_gpu_latent collect hook requires binary action masks"
                    )
                legal_array = np.flatnonzero(mask == 1.0).astype(np.int64, copy=False)
                if legal_array.size == 0:
                    raise ValueError(
                        "direct_ctree_gpu_latent collect hook got a zero-action mask at "
                        f"row {row_index}"
                    )
                if action_count is None:
                    action_count = int(mask.size)
                elif action_count != int(mask.size):
                    all_actions_legal = False
                if legal_array.size != mask.size or not np.array_equal(
                    legal_array,
                    np.arange(mask.size, dtype=np.int64),
                ):
                    all_actions_legal = False
                legal_action_arrays.append(legal_array)
                legal_actions.append([int(index) for index in legal_array])

        if (
            ctree_backend == COLLECT_SEARCH_CTREE_BACKEND_FLAT_A3
            and action_count != 3
        ):
            raise ValueError(
                "collect_search_ctree_backend='flat_a3' requires action_count=3, "
                f"got {action_count!r}"
            )

        model = getattr(self, "_collect_model", None)
        mcts = getattr(self, "_mcts_collect", None)
        if model is None or not hasattr(model, "initial_inference"):
            return fallback("missing_collect_model")
        if mcts is None or not hasattr(type(mcts), "roots"):
            return fallback("missing_ctree_mcts")

        self._collect_model.eval()
        self._collect_mcts_temperature = temperature
        self.collect_epsilon = epsilon
        output = {env_id: None for env_id in ready_env_id}
        root_dirichlet_alpha = float(getattr(cfg, "root_dirichlet_alpha"))
        root_noise_weight = float(getattr(cfg, "root_noise_weight"))
        eps_cfg = getattr(cfg, "eps", None)
        eps_greedy = bool(getattr(eps_cfg, "eps_greedy_exploration_in_collect", False))

        def select_all_actions_legal_fast(
            distributions: Any,
        ) -> tuple[np.ndarray, np.ndarray] | None:
            if not all_actions_legal or action_count is None or action_count <= 0:
                return None
            try:
                visit_counts = np.asarray(distributions, dtype=np.float64)
            except (TypeError, ValueError):
                return None
            if visit_counts.shape != (active_collect_env_num, action_count):
                return None
            if not np.all(np.isfinite(visit_counts)):
                return None
            temperature_value = float(self._collect_mcts_temperature)
            if temperature_value <= 0.0:
                return None
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                action_weights = np.power(visit_counts, 1.0 / temperature_value)
            weight_sums = action_weights.sum(axis=1, keepdims=True)
            if (
                not np.all(np.isfinite(action_weights))
                or not np.all(np.isfinite(weight_sums))
                or np.any(weight_sums <= 0.0)
            ):
                return None
            action_probs = action_weights / weight_sums
            with np.errstate(divide="ignore", invalid="ignore"):
                entropy_terms = np.where(action_probs > 0.0, action_probs * np.log2(action_probs), 0.0)
            entropies = -entropy_terms.sum(axis=1)
            if eps_greedy:
                action_indices = np.argmax(visit_counts, axis=1).astype(np.int64, copy=False)
                if self.collect_epsilon > 0.0:
                    explore_mask = np.random.random(active_collect_env_num) < self.collect_epsilon
                    explore_count = int(explore_mask.sum())
                    if explore_count > 0:
                        action_indices[explore_mask] = np.random.randint(
                            0,
                            action_count,
                            size=explore_count,
                        )
            else:
                cumulative = np.cumsum(action_probs, axis=1)
                cumulative[:, -1] = 1.0
                thresholds = np.random.random(active_collect_env_num)
                action_indices = np.argmax(thresholds[:, None] <= cumulative, axis=1)
            return action_indices.astype(np.int64, copy=False), entropies

        with torch.no_grad():
            with profiler.timer("collect_search_backend_initial_inference_sec", sync_cuda=True):
                network_output = model.initial_inference(data)
            latent_state_roots, reward_roots, pred_values, policy_logits = (
                mz_network_output_unpack(network_output)
            )
            if not hasattr(latent_state_roots, "device") or latent_state_roots.device.type != "cuda":
                raise ValueError("direct_ctree_gpu_latent collect hook requires CUDA latents")

            with profiler.timer("collect_search_backend_root_prepare_sec"):
                pred_values_np = (
                    self.inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
                )
                policy_logits_list = policy_logits.detach().cpu().numpy().tolist()
                if all_actions_legal and action_count is not None:
                    noises = (
                        np.random.dirichlet(
                            [root_dirichlet_alpha] * action_count,
                            size=active_collect_env_num,
                        )
                        .astype(np.float32)
                        .tolist()
                    )
                else:
                    noises = [
                        np.random.dirichlet([root_dirichlet_alpha] * len(actions))
                        .astype(np.float32)
                        .tolist()
                        for actions in legal_actions
                    ]
                tree_muzero = _import_collect_search_tree_muzero(ctree_backend)
                roots = (
                    tree_muzero.Roots(active_collect_env_num, legal_actions)
                    if ctree_backend == COLLECT_SEARCH_CTREE_BACKEND_FLAT_A3
                    else type(mcts).roots(active_collect_env_num, legal_actions)
                )
                roots.prepare(root_noise_weight, noises, reward_roots, policy_logits_list, to_play)

            num_simulations = int(getattr(getattr(mcts, "_cfg", cfg), "num_simulations"))
            profiler.add_count("collect_search_backend_direct_ctree_gpu_latent_calls")
            profiler.add_count("mcts_search_calls")
            profiler.add_count("mcts_search_root_sum", active_collect_env_num)
            profiler.add_count("mcts_search_simulation_budget_sum", num_simulations)
            profiler.add_count("mcts_search_node_budget_sum", active_collect_env_num * num_simulations)
            profiler.add_sample("mcts_search_root_batch", active_collect_env_num, limit=120)
            profiler.add_sample("mcts_search_simulations", num_simulations, limit=40)
            profiler.add_sample(
                "collect_search_backend",
                COLLECT_SEARCH_BACKEND_DIRECT_CTREE_GPU_LATENT,
                limit=40,
            )
            profiler.add_sample("collect_search_ctree_backend", ctree_backend, limit=40)
            profiler.add_sample("mcts_cfg_device", str(latent_state_roots.device), limit=20)
            with profiler.timer("mcts_search_sec", sync_cuda=True):
                _direct_ctree_gpu_latent_search_for_collect(
                    policy=self,
                    mcts=mcts,
                    model=model,
                    latent_state_roots=latent_state_roots,
                    roots=roots,
                    to_play=to_play,
                    ctree_backend=ctree_backend,
                    profiler=profiler,
                )

            with profiler.timer("collect_search_backend_output_assembly_sec"):
                root_visit_count_distributions = roots.get_distributions()
                roots_values = roots.get_values()
                fast_selection = select_all_actions_legal_fast(root_visit_count_distributions)
                if fast_selection is not None:
                    profiler.add_count("collect_search_backend_output_fast_path_calls")
                    action_indices, visit_count_distribution_entropies = fast_selection
                    for i, env_id in enumerate(ready_env_id):
                        output[env_id] = {
                            "action": action_indices[i],
                            "visit_count_distributions": root_visit_count_distributions[i],
                            "visit_count_distribution_entropy": (
                                visit_count_distribution_entropies[i]
                            ),
                            "searched_value": roots_values[i],
                            "predicted_value": pred_values_np[i],
                            "predicted_policy_logits": policy_logits_list[i],
                        }
                else:
                    for i, env_id in enumerate(ready_env_id):
                        distributions = root_visit_count_distributions[i]
                        value = roots_values[i]
                        legal_array = legal_action_arrays[i]
                        if eps_greedy:
                            action_index, visit_count_distribution_entropy = select_action(
                                distributions,
                                temperature=self._collect_mcts_temperature,
                                deterministic=True,
                            )
                            action = legal_array[action_index]
                            if np.random.rand() < self.collect_epsilon:
                                action = np.random.choice(legal_array)
                        else:
                            action_index, visit_count_distribution_entropy = select_action(
                                distributions,
                                temperature=self._collect_mcts_temperature,
                                deterministic=False,
                            )
                            action = legal_array[action_index]
                        output[env_id] = {
                            "action": action,
                            "visit_count_distributions": distributions,
                            "visit_count_distribution_entropy": (
                                visit_count_distribution_entropy
                            ),
                            "searched_value": value,
                            "predicted_value": pred_values_np[i],
                            "predicted_policy_logits": policy_logits_list[i],
                        }
        profiler.add_count("collect_search_backend_output_rows", active_collect_env_num)
        return output

    setattr(policy_cls, "_forward_collect", wrapped_forward_collect)
    hook_label = "lzero.policy.muzero.MuZeroPolicy._forward_collect.direct_ctree_gpu_latent"
    profiler.add_installed_hook(hook_label)
    profiler.add_sample(
        "collect_search_backend_hook",
        COLLECT_SEARCH_BACKEND_DIRECT_CTREE_GPU_LATENT,
        limit=20,
    )

    def restore() -> None:
        setattr(policy_cls, "_forward_collect", original_forward_collect)

    return restore


def _install_batched_profile_env_manager_hook(
    *,
    train_muzero: Any,
    profiler: _LightZeroPhaseProfiler,
    env_manager_type: str,
    seed: int,
    source_max_steps: int,
    decision_ms: float,
    natural_bonus_spawn: bool,
    disable_death_for_profile: bool,
) -> Any:
    """Patch stock LightZero to create the profile-only batched CurvyTron manager."""

    globals_map = getattr(train_muzero, "__globals__", None)
    if not isinstance(globals_map, dict):
        raise RuntimeError("selected LightZero entrypoint does not expose patchable globals")
    original = globals_map.get("create_env_manager")
    if not callable(original):
        raise RuntimeError("selected LightZero entrypoint does not expose create_env_manager")

    def build_batched_manager(env_num: int, manager_type: str) -> Any:
        manager_type = str(manager_type)
        zero_observation_profile = (
            manager_type == CURVYZERO_BATCHED_ZERO_OBS_PROFILE_ENV_MANAGER_TYPE
        )
        if int(env_num) < 2 or int(env_num) % 2 != 0:
            raise RuntimeError(
                f"{manager_type} requires an even env count because "
                "one physical CurvyTron row exposes two scalar player views; "
                f"got env_num={env_num}"
            )
        import numpy as np
        from ding.envs import BaseEnvTimestep

        from curvyzero.env import vector_runtime
        from curvyzero.env.vector_multiplayer_env import DEFAULT_BODY_CAPACITY
        from curvyzero.training.multiplayer_source_state_trainer_surface import (
            SourceStateMultiplayerTrainerSurface,
        )
        from curvyzero.training.multiplayer_source_state_trainer_surface import (
            TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
        )
        from curvyzero.training.source_state_batched_observation_mock_collector import (
            BatchedLightZeroScalarActionBridge,
        )
        from curvyzero.training.source_state_batched_observation_mock_collector import (
            BatchedLightZeroStockEnvManagerAdapter,
        )
        from curvyzero.training.source_state_batched_observation_profile import (
            SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND,
        )
        from curvyzero.training.source_state_batched_observation_profile import (
            SourceStateBatchedRenderRequest,
        )
        from curvyzero.training.source_state_batched_observation_profile import (
            SourceStateBatchedRenderResult,
        )

        batch_size = int(env_num) // 2
        render_config: dict[str, Any] = {
            "prewarm_dynamic_render_functions": False,
        }
        render_fn_for_slots = None
        bp = None

        class ZeroObservationRenderer:
            backend_name = "zero_observation_profile"

            def render(
                self,
                request: SourceStateBatchedRenderRequest,
            ) -> SourceStateBatchedRenderResult:
                started = time.perf_counter()
                request.out[...] = 0
                elapsed = time.perf_counter() - started
                return SourceStateBatchedRenderResult(
                    frames=request.out,
                    telemetry={
                        "render_sec": elapsed,
                        "zero_fill_sec": elapsed,
                        "device_render_sec": 0.0,
                        "host_to_device_sec": 0.0,
                        "device_to_host_sec": 0.0,
                        "pack_sec": 0.0,
                    },
                )

        if zero_observation_profile:
            observation_renderer = ZeroObservationRenderer()
            required_observation_backend = ZeroObservationRenderer.backend_name
        else:
            import jax
            import jax.numpy as jnp

            from curvyzero.infra.modal import (
                source_state_batched_observation_boundary_profile as bp_mod,
            )
            from curvyzero.infra.modal.source_state_gpu_render_benchmark import (
                BONUS_RENDER_MODE_IDS,
            )
            from curvyzero.infra.modal.source_state_gpu_render_benchmark import (
                BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
            )
            from curvyzero.infra.modal.source_state_gpu_render_benchmark import (
                GEOMETRY_DTYPE_FLOAT32,
            )
            from curvyzero.infra.modal.source_state_gpu_render_benchmark import (
                RENDER_MODE_BROWSER_LINES,
            )
            from curvyzero.infra.modal.source_state_gpu_render_benchmark import (
                RENDER_MODE_IDS,
            )
            from curvyzero.infra.modal.source_state_gpu_render_benchmark import (
                RENDER_SURFACE_DIRECT_GRAY64,
            )

            bp = bp_mod
            checked = bp._validate_boundary_config(
                np=np,
                config={
                    "batch_size": batch_size,
                    "body_capacity": DEFAULT_BODY_CAPACITY,
                    "geometry_dtype": GEOMETRY_DTYPE_FLOAT32,
                    "render_surface": RENDER_SURFACE_DIRECT_GRAY64,
                    "trail_slots": DEFAULT_BODY_CAPACITY,
                    "dynamic_render_trail_slots": True,
                    "min_render_trail_slots": getattr(
                        bp,
                        "DEFAULT_DYNAMIC_MIN_RENDER_TRAIL_SLOTS",
                        32,
                    ),
                    # Full stock-loop rows should not prewarm every dynamic width:
                    # that compiles several large JAX render programs and can
                    # exhaust H100 memory before collection starts. Let the row pay
                    # first-use cost for widths it actually reaches, then report it.
                    "prewarm_dynamic_render_functions": False,
                    "max_ticks": int(source_max_steps),
                    "steps": 1,
                    "warmup_steps": 0,
                    "verify_steps": 0,
                    "cpu_reference_interval": 0,
                    "profile_env_manager_canary": True,
                    "surface_stack_backend": TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
                },
            )
            render_config = checked["render_config"]
            render_mode_id = RENDER_MODE_IDS[RENDER_MODE_BROWSER_LINES]
            bonus_render_mode_id = BONUS_RENDER_MODE_IDS[BONUS_RENDER_MODE_SIMPLE_SYMBOLS]
            render_fn_cache: dict[int, Any] = {}

            def render_fn_for_slots(render_trail_slots: int) -> Any:
                slots = int(render_trail_slots)
                cached = render_fn_cache.get(slots)
                if cached is not None:
                    return cached
                slot_config = dict(render_config)
                slot_config["trail_slots"] = slots
                cached = bp._make_jax_two_view_render_fn(
                    jax=jax,
                    jnp=jnp,
                    config=slot_config,
                    render_mode_id=render_mode_id,
                    bonus_render_mode_id=bonus_render_mode_id,
                )
                render_fn_cache[slots] = cached
                return cached

            observation_renderer = bp._DynamicJaxBatchedObservationRenderer(
                jax=jax,
                np=np,
                config=render_config,
                render_fn_for_slots=render_fn_for_slots,
            )
            required_observation_backend = SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND

        surface = SourceStateMultiplayerTrainerSurface(
            batch_size=batch_size,
            player_count=2,
            seed=int(seed),
            decision_source_frames=int(DEFAULT_DECISION_SOURCE_FRAMES),
            source_physics_step_ms=float(DEFAULT_SOURCE_PHYSICS_STEP_MS),
            body_capacity=DEFAULT_BODY_CAPACITY,
            natural_bonus_spawn=bool(natural_bonus_spawn),
            death_mode=(
                vector_runtime.DEATH_MODE_PROFILE_NO_DEATH
                if bool(disable_death_for_profile)
                else vector_runtime.DEATH_MODE_NORMAL
            ),
            max_ticks=int(source_max_steps),
            observation_stack_backend=TRAINER_STACK_BACKEND_RENDERER_BACKED_PROFILE,
            observation_renderer=observation_renderer,
            required_observation_renderer_backend=required_observation_backend,
        )
        manager = BatchedLightZeroStockEnvManagerAdapter(
            BatchedLightZeroScalarActionBridge(
                surface,
                timer_advance_ms=float(decision_ms),
            ),
            base_env_timestep_cls=BaseEnvTimestep,
        )
        original_manager_reset = manager.reset
        original_manager_step = manager.step
        prewarm_render_done = False

        def profiled_manager_reset(*args: Any, **kwargs: Any) -> Any:
            nonlocal prewarm_render_done
            started = time.perf_counter()
            try:
                result = original_manager_reset(*args, **kwargs)
                if (
                    not prewarm_render_done
                    and bp is not None
                    and render_fn_for_slots is not None
                    and bool(render_config.get("prewarm_dynamic_render_functions", False))
                ):
                    prewarm_started = time.perf_counter()
                    prewarm = bp._prewarm_dynamic_render_functions(
                        jax=jax,
                        np=np,
                        production_state=surface.env.state,
                        config=render_config,
                        render_fn_for_slots=render_fn_for_slots,
                    )
                    prewarm_render_done = True
                    profiler.add_time(
                        "batched_profile_renderer_prewarm_sec",
                        time.perf_counter() - prewarm_started,
                    )
                    for slot in prewarm.get("slots", []):
                        profiler.add_sample(
                            "batched_profile_renderer_prewarm_slot",
                            int(slot),
                            limit=40,
                        )
                elif not prewarm_render_done:
                    prewarm_render_done = True
                    profiler.add_count("batched_profile_renderer_prewarm_skipped_calls")
                return result
            finally:
                profiler.add_time(
                    "batched_profile_env_manager_reset_sec",
                    time.perf_counter() - started,
                )
                profiler.add_count("batched_profile_env_manager_reset_calls")
                profiler.add_sample(
                    "batched_profile_env_manager_ready_obs_after_reset",
                    len(manager.ready_obs),
                    limit=120,
                )

        def profiled_manager_step(actions: Mapping[int, Any]) -> Any:
            before_ready_ids = tuple(sorted(int(env_id) for env_id in manager.ready_obs))
            before_ready_rows = {int(env_id) // 2 for env_id in before_ready_ids}
            action_rows = {int(env_id) // 2 for env_id in actions}
            result = None
            started = time.perf_counter()
            try:
                result = original_manager_step(actions)
                return result
            finally:
                elapsed = time.perf_counter() - started
                profiler.add_time("batched_profile_env_manager_step_sec", elapsed)
                profiler.add_count("batched_profile_env_manager_step_calls")
                profiler.add_sample(
                    "batched_profile_env_manager_step_action_count",
                    len(actions),
                    limit=120,
                )
                profiler.add_sample(
                    "batched_profile_env_manager_ready_obs_before_step",
                    len(before_ready_ids),
                    limit=120,
                )
                profiler.add_sample(
                    "batched_profile_env_manager_ready_row_before_step",
                    len(before_ready_rows),
                    limit=120,
                )
                profiler.add_sample(
                    "batched_profile_env_manager_action_row_count",
                    len(action_rows),
                    limit=120,
                )
                profiler.add_sample(
                    "batched_profile_env_manager_complete_row_omission_count",
                    len(before_ready_rows.difference(action_rows)),
                    limit=120,
                )
                if isinstance(result, Mapping):
                    timestep_rows = {int(env_id) // 2 for env_id in result}
                    profiler.add_sample(
                        "batched_profile_env_manager_timestep_count",
                        len(result),
                        limit=120,
                    )
                    profiler.add_sample(
                        "batched_profile_env_manager_timestep_row_count",
                        len(timestep_rows),
                        limit=120,
                    )
                last_profile_step = getattr(manager, "last_profile_step", None)
                bridge_output = getattr(last_profile_step, "bridge_output", None)
                bridge_timing = getattr(bridge_output, "profile_timing_sec", None)
                if isinstance(bridge_timing, Mapping):
                    for key, value in bridge_timing.items():
                        try:
                            profiler.add_time(
                                f"batched_profile_bridge_{key}",
                                float(value),
                            )
                        except (TypeError, ValueError):
                            pass
                bridge_counts = getattr(bridge_output, "profile_counts", None)
                if isinstance(bridge_counts, Mapping):
                    for key, value in bridge_counts.items():
                        try:
                            profiler.add_sample(
                                f"batched_profile_bridge_{key}",
                                int(value),
                                limit=120,
                            )
                        except (TypeError, ValueError):
                            pass
                autoreset_row_mask = getattr(bridge_output, "autoreset_row_mask", None)
                if autoreset_row_mask is not None:
                    profiler.add_sample(
                        "batched_profile_env_manager_autoreset_rows",
                        int(np.asarray(autoreset_row_mask, dtype=bool).sum()),
                        limit=120,
                    )
                ready_after = getattr(bridge_output, "ready_obs", None)
                if isinstance(ready_after, Mapping):
                    profiler.add_sample(
                        "batched_profile_env_manager_ready_obs_after_step",
                        len(ready_after),
                        limit=120,
                    )
                    ready_after_rows = {int(env_id) // 2 for env_id in ready_after}
                    profiler.add_sample(
                        "batched_profile_env_manager_ready_row_after_step",
                        len(ready_after_rows),
                        limit=120,
                    )
                surface_step = getattr(bridge_output, "surface_step", None)
                surface_info = getattr(surface_step, "info", None)
                if isinstance(surface_info, Mapping):
                    surface_timing = surface_info.get("trainer_surface_profile_timing")
                    if isinstance(surface_timing, Mapping):
                        for key, value in surface_timing.items():
                            try:
                                profiler.add_time(
                                    f"batched_profile_surface_{key}",
                                    float(value),
                                )
                            except (TypeError, ValueError):
                                pass
                    renderer_telemetry = surface_info.get("renderer_backed_stack_telemetry")
                    if isinstance(renderer_telemetry, Mapping):
                        for key, value in renderer_telemetry.items():
                            if str(key).endswith("_sec"):
                                try:
                                    profiler.add_time(
                                        f"batched_profile_renderer_{key}",
                                        float(value),
                                    )
                                except (TypeError, ValueError):
                                    pass
                            elif str(key) in {
                                "render_trail_slots",
                                "active_trail_count_max",
                                "render_truncation_row_count",
                                "partial_render_request",
                                "render_output_count",
                            }:
                                try:
                                    profiler.add_sample(
                                        f"batched_profile_renderer_{key}",
                                        float(value),
                                        limit=120,
                                    )
                                except (TypeError, ValueError):
                                    pass

        manager.reset = profiled_manager_reset
        manager.step = profiled_manager_step
        profiler.add_count("batched_profile_env_manager_create_calls")
        profiler.add_sample("batched_profile_env_manager_env_num", int(env_num), limit=20)
        profiler.add_sample(
            "batched_profile_env_manager_batch_size",
            int(batch_size),
            limit=20,
        )
        profiler.add_sample(
            "batched_profile_env_manager_backend",
            str(required_observation_backend),
            limit=20,
        )
        profiler.add_sample(
            "batched_profile_env_manager_type",
            manager_type,
            limit=20,
        )
        profiler.add_sample(
            "batched_profile_env_manager_decision_ms",
            float(decision_ms),
            limit=20,
        )
        return manager

    registry_restore = None
    try:
        from ding.utils import ENV_MANAGER_REGISTRY

        previous_registrations = {
            name: ENV_MANAGER_REGISTRY.get(name)
            for name in CURVYZERO_BATCHED_PROFILE_ENV_MANAGER_TYPES
            if name in ENV_MANAGER_REGISTRY
        }

        def make_registry_class(registry_name: str) -> Any:
            """Registry-visible profile manager; construction delegates to the hook."""

            class CurvyZeroBatchedProfileEnvManagerForLightZero:
                config = {
                    "episode_num": float("inf"),
                    "max_retry": 1,
                    "retry_type": "reset",
                    "auto_reset": True,
                    "step_timeout": None,
                    "reset_timeout": None,
                    "retry_waiting_time": 0.1,
                }

                @classmethod
                def default_config(cls) -> Any:
                    from easydict import EasyDict

                    cfg = EasyDict(copy.deepcopy(cls.config))
                    cfg.cfg_type = cls.__name__ + "Dict"
                    return cfg

                def __init__(self, env_fn: list[Any], cfg: Any | None = None) -> None:
                    del cfg
                    self._delegate = build_batched_manager(len(env_fn), registry_name)

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._delegate, name)

            CurvyZeroBatchedProfileEnvManagerForLightZero.__name__ = (
                "CurvyZeroBatchedProfileEnvManagerForLightZero"
                if registry_name == CURVYZERO_BATCHED_PROFILE_ENV_MANAGER_TYPE
                else "CurvyZeroBatchedZeroObservationProfileEnvManagerForLightZero"
            )
            return CurvyZeroBatchedProfileEnvManagerForLightZero

        for registry_name in CURVYZERO_BATCHED_PROFILE_ENV_MANAGER_TYPES:
            ENV_MANAGER_REGISTRY.register(
                registry_name,
                module=make_registry_class(registry_name),
                force_overwrite=True,
            )

        def restore_registry() -> None:
            for registry_name in CURVYZERO_BATCHED_PROFILE_ENV_MANAGER_TYPES:
                if registry_name in previous_registrations:
                    ENV_MANAGER_REGISTRY[registry_name] = previous_registrations[
                        registry_name
                    ]
                else:
                    ENV_MANAGER_REGISTRY.pop(registry_name, None)

        registry_restore = restore_registry
    except Exception as exc:
        raise RuntimeError(
            "failed to register CurvyZero profile env managers with DI-engine "
            "ENV_MANAGER_REGISTRY"
        ) from exc

    def create_env_manager_wrapped(manager_cfg: Any, env_fns: Any, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            raise RuntimeError(
                "curvyzero_batched_profile create_env_manager hook expected only "
                "(manager_cfg, env_fns)"
            )
        try:
            env_num = len(env_fns)
        except Exception as exc:
            raise RuntimeError(
                "curvyzero_batched_profile could not read env_fns length"
            ) from exc
        manager_type = getattr(manager_cfg, "type", None)
        if manager_type is None and isinstance(manager_cfg, Mapping):
            manager_type = manager_cfg.get("type")
        manager_type = str(manager_type)
        if manager_type not in CURVYZERO_BATCHED_PROFILE_ENV_MANAGER_TYPES:
            raise RuntimeError(
                "CurvyZero profile env-manager hook was installed, but LightZero asked "
                f"for env manager type {manager_type!r}; refusing hidden fallback to "
                "the scalar env-manager path"
            )
        return build_batched_manager(int(env_num), manager_type)

    globals_map["create_env_manager"] = create_env_manager_wrapped

    def restore() -> None:
        globals_map["create_env_manager"] = original
        if registry_restore is not None:
            registry_restore()

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
        (
            base
            for base in inspect.getmro(learner_cls)
            if "save_checkpoint" in getattr(base, "__dict__", {})
        ),
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
                f"curvyzero checkpoint eval trigger failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
        return result

    setattr(owner, "save_checkpoint", wrapped)

    def restore() -> None:
        setattr(owner, "save_checkpoint", original)

    return restore


def _safe_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        if hasattr(value, "numel") and callable(value.numel):
            if int(value.numel()) != 1:
                return None
        if hasattr(value, "item") and callable(value.item):
            value = value.item()
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _flatten_numeric_scalars(
    value: Any,
    *,
    prefix: str = "",
    limit: int = 200,
) -> dict[str, float]:
    """Extract small scalar learner metrics without materializing large tensors."""

    metrics: dict[str, float] = {}

    def visit(item: Any, key_prefix: str) -> None:
        if len(metrics) >= limit:
            return
        scalar = _safe_float_or_none(item)
        if scalar is not None and key_prefix:
            metrics[key_prefix] = scalar
            return
        if isinstance(item, Mapping):
            for key in sorted(item.keys(), key=str):
                child_key = str(key)
                if child_key.startswith("_"):
                    continue
                next_prefix = f"{key_prefix}.{child_key}" if key_prefix else child_key
                visit(item[key], next_prefix)
                if len(metrics) >= limit:
                    return
            return
        if isinstance(item, (list, tuple)) and len(item) <= 8:
            for index, child in enumerate(item):
                next_prefix = f"{key_prefix}.{index}" if key_prefix else str(index)
                visit(child, next_prefix)
                if len(metrics) >= limit:
                    return

    visit(value, prefix)
    return metrics


class _LearnerMetricsRecorder:
    def __init__(
        self,
        *,
        run_id: str,
        attempt_id: str,
        attempt_train_root: Path,
        started_monotonic: float,
        sample_first_n: int = 5,
        sample_interval: int = 1000,
    ) -> None:
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.attempt_train_root = attempt_train_root
        self.started_monotonic = float(started_monotonic)
        self.sample_first_n = max(0, int(sample_first_n))
        self.sample_interval = max(0, int(sample_interval))
        self.call_index = 0
        self.latest_row: dict[str, Any] | None = None
        self.jsonl_path = attempt_train_root / "learner_metrics.jsonl"
        self.latest_path = attempt_train_root / "learner_metrics_latest.json"

    def latest(self) -> dict[str, Any] | None:
        return _to_plain(self.latest_row) if self.latest_row is not None else None

    def refs(self) -> dict[str, str | None]:
        result: dict[str, str | None] = {
            "learner_metrics_ref": None,
            "learner_metrics_latest_ref": None,
        }
        for key, path in (
            ("learner_metrics_ref", self.jsonl_path),
            ("learner_metrics_latest_ref", self.latest_path),
        ):
            try:
                result[key] = runs.file_ref(path.resolve(), mount=RUNS_MOUNT.resolve())
            except ValueError:
                result[key] = None
        return result

    def _should_persist(self, call_index: int) -> bool:
        if call_index <= self.sample_first_n:
            return True
        return self.sample_interval > 0 and call_index % self.sample_interval == 0

    def record(
        self,
        *,
        learner: Any,
        result: Any = None,
        exception: BaseException | None = None,
        train_iter_before: int | None,
        train_iter_after: int | None,
        elapsed_sec: float,
    ) -> None:
        self.call_index += 1
        now = runs.utc_timestamp()
        row: dict[str, Any] = {
            "schema_id": "curvyzero_lightzero_learner_metrics/v0",
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "created_at": now,
            "timestamp": now,
            "source": "BaseLearner.train",
            "learner_train_call_index": int(self.call_index),
            "train_iter_before": train_iter_before,
            "train_iter_after": train_iter_after,
            "train_iter_delta": (
                int(train_iter_after) - int(train_iter_before)
                if train_iter_before is not None and train_iter_after is not None
                else None
            ),
            "collector_envstep": _safe_int_or_none(getattr(learner, "collector_envstep", None)),
            "elapsed_sec": float(elapsed_sec),
            "train_wall_elapsed_sec": time.perf_counter() - self.started_monotonic,
            "result_type": type(result).__name__ if exception is None else None,
            "exception_type": type(exception).__name__ if exception is not None else None,
            "exception_message": str(exception) if exception is not None else None,
            "numeric_metrics": _flatten_numeric_scalars(result) if exception is None else {},
            "model_parameters_changed": None,
        }
        self.latest_row = _to_plain(row)
        if not self._should_persist(self.call_index):
            return
        try:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_to_plain(row), sort_keys=True) + "\n")
            runs.write_json(self.latest_path, _to_plain(row))
        except Exception as exc:  # pragma: no cover - remote resilience only.
            print(
                f"curvyzero learner metrics write failed: {type(exc).__name__}: {exc}",
                flush=True,
            )


def _lightzero_exp_sibling_roots(exp_name: Path) -> list[Path]:
    return lz_checkpoints.lightzero_exp_sibling_roots(exp_name)


def _lightzero_exp_checkpoint_dirs(exp_name: Path) -> list[Path]:
    return lz_checkpoints.lightzero_exp_checkpoint_dirs(exp_name)


def _lightzero_exp_resume_state_dirs(exp_name: Path) -> list[Path]:
    return lz_checkpoints.lightzero_exp_resume_state_dirs(exp_name)


def _latest_lightzero_iteration_checkpoint(exp_name: Path) -> dict[str, Any] | None:
    latest = lz_checkpoints.latest_lightzero_iteration_checkpoint_from_dirs(
        _lightzero_exp_checkpoint_dirs(exp_name)
    )
    if latest is None:
        return None
    path = latest.path
    checkpoint_ref = None
    try:
        checkpoint_ref = _checkpoint_source_ref(path)
    except ValueError:
        checkpoint_ref = None
    return {
        "iteration": int(latest.iteration),
        "checkpoint_name": latest.checkpoint_name,
        "checkpoint_path": str(path),
        "checkpoint_ref": checkpoint_ref,
    }


def _build_checkpoint_progress_latest_payload(
    *,
    run_id: str,
    attempt_id: str,
    checkpoint: dict[str, Any] | None,
    learner_iteration: int | None,
    elapsed_sec: float,
    timestamp: str,
    source: str,
) -> dict[str, Any]:
    iteration = (
        int(checkpoint["iteration"])
        if checkpoint is not None and checkpoint.get("iteration") is not None
        else learner_iteration
    )
    payload = {
        "schema_id": "curvyzero_lightzero_curvytron_train_progress_latest/v0",
        "task_id": TASK_ID,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "iteration": iteration,
        "learner_train_iter": learner_iteration,
        "event": "checkpoint",
        "elapsed_sec": round(max(0.0, float(elapsed_sec)), 6),
        "timestamp": timestamp,
        "updated_at": timestamp,
        "source": source,
    }
    if checkpoint is not None:
        payload["checkpoint"] = checkpoint
        payload["checkpoint_ref"] = checkpoint.get("checkpoint_ref")
        payload["checkpoint_name"] = checkpoint.get("checkpoint_name")
    return payload


def _own_checkpoint_opponent_root_ref(run_id: str, attempt_id: str) -> PurePosixPath:
    return runs.attempt_train_ref(TASK_ID, run_id, attempt_id) / "own_checkpoint_opponent"


def _own_checkpoint_opponent_refresh_pointer_ref(run_id: str, attempt_id: str) -> str:
    pointer_ref = _own_checkpoint_opponent_root_ref(run_id, attempt_id) / "latest.json"
    return f"{_RUNS_REF_PREFIX}{pointer_ref.as_posix()}"


def _write_own_checkpoint_opponent_refresh(
    *,
    run_id: str,
    attempt_id: str,
    checkpoint: Mapping[str, Any],
    timestamp: str,
    seed: int,
    num_simulations: int,
    batch_size: int,
    opponent_use_cuda: bool,
    assignment_refresh_ref: str | None = None,
) -> dict[str, Any]:
    iteration = _safe_int_or_none(checkpoint.get("iteration"))
    checkpoint_ref = str(checkpoint.get("checkpoint_ref") or "").strip()
    if iteration is None:
        return {"saved": False, "reason": "checkpoint_iteration_missing"}
    if iteration <= 0:
        return {
            "saved": False,
            "reason": "initial_checkpoint_not_used_as_own_opponent",
            "iteration": int(iteration),
        }
    if not checkpoint_ref:
        return {
            "saved": False,
            "reason": "checkpoint_ref_missing",
            "iteration": int(iteration),
        }
    if Path(checkpoint_ref).name != f"iteration_{int(iteration)}.pth.tar":
        return {
            "saved": False,
            "reason": "checkpoint_ref_not_exact_iteration_file",
            "iteration": int(iteration),
            "checkpoint_ref": checkpoint_ref,
        }

    root_ref = _own_checkpoint_opponent_root_ref(run_id, attempt_id)
    assignment_ref = root_ref / "assignments" / f"iteration_{int(iteration)}.json"
    if assignment_refresh_ref and str(assignment_refresh_ref).startswith(_CONTROL_REF_PREFIX):
        raise ValueError("own checkpoint opponent refresh pointer must live in the runs volume")
    if assignment_refresh_ref and str(assignment_refresh_ref).startswith(_RUNS_REF_PREFIX):
        pointer_ref = runs.require_relative_ref(
            str(assignment_refresh_ref)[len(_RUNS_REF_PREFIX) :]
        )
    elif assignment_refresh_ref:
        pointer_ref = runs.require_relative_ref(str(assignment_refresh_ref))
    else:
        pointer_ref = root_ref / "latest.json"

    assignment = {
        "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
        "assignment_id": f"own-checkpoint-iteration-{int(iteration)}",
        "source_epoch": int(iteration),
        "source_ref": checkpoint_ref,
        "created_at": timestamp,
        "seed": int(seed),
        "entries": [
            {
                "name": "own_previous_checkpoint",
                "age_label": f"own_iteration_{int(iteration)}",
                "weight": 100.0,
                "opponent_policy_kind": OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
                "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_NORMAL,
                "opponent_immortal": False,
                "opponent_checkpoint_ref": checkpoint_ref,
                "opponent_snapshot_ref": f"own_checkpoint_iteration_{int(iteration)}",
                "opponent_policy_seed": int(seed),
                "opponent_num_simulations": int(num_simulations),
                "opponent_batch_size": int(batch_size),
                "opponent_use_cuda": bool(opponent_use_cuda),
                "tags": {
                    "source": "same_run_previous_checkpoint",
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "iteration": int(iteration),
                },
            }
        ],
    }
    parse_opponent_assignment_snapshot(assignment)
    assignment_sha256 = canonical_assignment_json_sha256(assignment)
    pointer = {
        "schema_id": OPPONENT_ASSIGNMENT_REFRESH_POINTER_SCHEMA_ID,
        "created_at": timestamp,
        "updated_at": timestamp,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "source": "same_run_previous_checkpoint",
        "iteration": int(iteration),
        "checkpoint_ref": checkpoint_ref,
        "assignment_ref": assignment_ref.as_posix(),
        "assignment_sha256": assignment_sha256,
    }

    assignment_path = runs.volume_path(RUNS_MOUNT, assignment_ref)
    pointer_path = runs.volume_path(RUNS_MOUNT, pointer_ref)
    runs.write_json(assignment_path, _to_plain(assignment))
    runs.write_json(pointer_path, _to_plain(pointer))
    return {
        "saved": True,
        "iteration": int(iteration),
        "checkpoint_ref": checkpoint_ref,
        "assignment_ref": assignment_ref.as_posix(),
        "assignment_sha256": assignment_sha256,
        "pointer_ref": f"{_RUNS_REF_PREFIX}{pointer_ref.as_posix()}",
        "assignment_file": runs.file_summary_any_mount(assignment_path, mount=RUNS_MOUNT),
        "pointer_file": runs.file_summary_any_mount(pointer_path, mount=RUNS_MOUNT),
    }


def _write_checkpoint_progress_latest(
    *,
    run_id: str,
    attempt_id: str,
    attempt_train_root: Path,
    exp_name: Path,
    learner: Any,
    started_monotonic: float,
    source: str = "BaseLearner.save_checkpoint",
    checkpoint_metadata: Mapping[str, Any] | None = None,
    own_checkpoint_opponent_publisher: Any | None = None,
    learner_metrics_recorder: _LearnerMetricsRecorder | None = None,
) -> dict[str, Any]:
    checkpoint = _latest_lightzero_iteration_checkpoint(exp_name)
    learner_iteration = _safe_int_or_none(getattr(learner, "train_iter", None))
    now = runs.utc_timestamp()
    payload = _build_checkpoint_progress_latest_payload(
        run_id=run_id,
        attempt_id=attempt_id,
        checkpoint=checkpoint,
        learner_iteration=learner_iteration,
        elapsed_sec=time.perf_counter() - started_monotonic,
        timestamp=now,
        source=source,
    )
    if checkpoint is not None and checkpoint_metadata is not None:
        payload["checkpoint_policy_metadata_sidecar"] = _write_checkpoint_policy_metadata_sidecar(
            run_id=run_id,
            attempt_id=attempt_id,
            checkpoint=checkpoint,
            checkpoint_metadata=checkpoint_metadata,
            timestamp=now,
        )
    if checkpoint is not None and own_checkpoint_opponent_publisher is not None:
        try:
            payload["own_checkpoint_opponent_refresh"] = own_checkpoint_opponent_publisher(
                checkpoint=checkpoint,
                timestamp=now,
                learner_iteration=learner_iteration,
            )
        except Exception as exc:  # pragma: no cover - remote resilience only.
            payload["own_checkpoint_opponent_refresh"] = {
                "saved": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
    if learner_metrics_recorder is not None:
        latest_learner = learner_metrics_recorder.latest()
        if latest_learner is not None:
            payload["last_learner"] = latest_learner
            payload.update(
                {
                    key: value
                    for key, value in learner_metrics_recorder.refs().items()
                    if value is not None
                }
            )
    path = attempt_train_root / "progress_latest.json"
    write = runs.write_json(path, _to_plain(payload))
    if checkpoint is not None:
        sidecar = payload.get("checkpoint_policy_metadata_sidecar")
        append_lineage_event(
            lineage_events_path(attempt_train_root),
            stage="checkpoint_written",
            run_id=run_id,
            attempt_id=attempt_id,
            source=source,
            progress_ref=runs.file_ref(path, mount=RUNS_MOUNT),
            progress_sha256=write.get("sha256"),
            checkpoint_ref=checkpoint.get("checkpoint_ref"),
            checkpoint_name=checkpoint.get("checkpoint_name"),
            checkpoint_iteration=checkpoint.get("iteration"),
            learner_iteration=learner_iteration,
            checkpoint_bytes=checkpoint.get("bytes"),
            checkpoint_sha256=checkpoint.get("sha256"),
            checkpoint_policy_metadata_ref=(
                sidecar.get("ref") if isinstance(sidecar, Mapping) else None
            ),
            own_checkpoint_opponent_refresh=payload.get("own_checkpoint_opponent_refresh"),
        )
    return write


def _checkpoint_policy_metadata_sidecar_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.name}.metadata.json")


def _build_checkpoint_policy_metadata_sidecar_payload(
    *,
    run_id: str,
    attempt_id: str,
    checkpoint: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    metadata = dict(checkpoint_metadata)
    observation_contract = _to_plain(metadata.get("observation_contract"))
    if not isinstance(observation_contract, Mapping):
        observation_contract = policy_observation_surface(
            trail_render_mode=str(
                metadata.get("policy_trail_render_mode") or POLICY_TRAIL_RENDER_MODE
            ),
            bonus_render_mode=metadata.get("policy_bonus_render_mode"),
            backend=str(
                metadata.get("policy_observation_backend")
                or DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE
            ),
        )
    payload: dict[str, Any] = {
        "schema_id": "curvyzero_checkpoint_policy_metadata/v0",
        "created_at": timestamp,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "iteration": checkpoint.get("iteration"),
        "checkpoint_name": checkpoint.get("checkpoint_name"),
        "checkpoint_ref": checkpoint.get("checkpoint_ref"),
        "policy_observation_backend": metadata.get("policy_observation_backend"),
        "policy_trail_render_mode": metadata.get("policy_trail_render_mode"),
        "policy_bonus_render_mode": metadata.get("policy_bonus_render_mode"),
        "policy_observation_contract_id": metadata.get("policy_observation_contract_id"),
        "policy_observation_perspective_schema_id": (
            metadata.get("policy_observation_perspective_schema_id")
            or observation_contract.get("perspective_schema_id")
            or POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID
        ),
        "observation_contract": observation_contract,
        "source_state_trail_render_mode": metadata.get("source_state_trail_render_mode"),
        "source_state_bonus_render_mode": metadata.get("source_state_bonus_render_mode"),
        "model_env_variant": metadata.get("model_env_variant") or metadata.get("env_variant"),
        "model_reward_variant": (
            metadata.get("model_reward_variant") or metadata.get("reward_variant")
        ),
        "env_variant": metadata.get("env_variant"),
        "reward_variant": metadata.get("reward_variant"),
        "decision_ms": metadata.get("decision_ms"),
        "decision_source_frames": metadata.get("decision_source_frames"),
        "source_physics_step_ms": metadata.get("source_physics_step_ms"),
        "source_max_steps": metadata.get("source_max_steps"),
        "learner_seat_mode": metadata.get("learner_seat_mode"),
    }
    return _to_plain({key: value for key, value in payload.items() if value is not None})


def _write_checkpoint_policy_metadata_sidecar(
    *,
    run_id: str,
    attempt_id: str,
    checkpoint: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    checkpoint_path_value = checkpoint.get("checkpoint_path")
    if not checkpoint_path_value:
        return {"saved": False, "reason": "checkpoint_path_missing"}
    checkpoint_path = Path(str(checkpoint_path_value))
    sidecar_path = _checkpoint_policy_metadata_sidecar_path(checkpoint_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_checkpoint_policy_metadata_sidecar_payload(
        run_id=run_id,
        attempt_id=attempt_id,
        checkpoint=checkpoint,
        checkpoint_metadata=checkpoint_metadata,
        timestamp=timestamp,
    )
    sidecar_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    sidecar_ref = None
    try:
        sidecar_ref = runs.file_ref(sidecar_path.resolve(), mount=RUNS_MOUNT.resolve())
    except ValueError:
        sidecar_ref = None
    return {
        "saved": True,
        "path": str(sidecar_path),
        "ref": sidecar_ref,
        "schema_id": payload.get("schema_id"),
    }


def _install_checkpoint_progress_writer(
    *,
    train_muzero: Any,
    run_id: str,
    attempt_id: str,
    exp_name: Path,
    attempt_train_root: Path,
    started_monotonic: float,
    commit_on_checkpoint: bool = DEFAULT_COMMIT_ON_CHECKPOINT,
    checkpoint_metadata: Mapping[str, Any] | None = None,
    own_checkpoint_opponent_publisher: Any | None = None,
    learner_metrics_recorder: _LearnerMetricsRecorder | None = None,
) -> Any:
    """Write the progress file used by the GIF browser whenever LightZero checkpoints."""

    globals_map = getattr(train_muzero, "__globals__", {})
    learner_cls = globals_map.get("BaseLearner")
    if not inspect.isclass(learner_cls):
        return None
    owner = next(
        (
            base
            for base in inspect.getmro(learner_cls)
            if "save_checkpoint" in getattr(base, "__dict__", {})
        ),
        None,
    )
    if owner is None:
        return None
    original = owner.__dict__["save_checkpoint"]

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        try:
            _write_checkpoint_progress_latest(
                run_id=run_id,
                attempt_id=attempt_id,
                attempt_train_root=attempt_train_root,
                exp_name=exp_name,
                learner=self,
                started_monotonic=started_monotonic,
                checkpoint_metadata=checkpoint_metadata,
                own_checkpoint_opponent_publisher=own_checkpoint_opponent_publisher,
                learner_metrics_recorder=learner_metrics_recorder,
            )
            if commit_on_checkpoint:
                _commit_runs_volume_with_backoff(label="checkpoint_progress_commit")
        except Exception as exc:  # pragma: no cover - remote resilience only.
            print(
                f"curvyzero checkpoint progress write failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
        return result

    setattr(owner, "save_checkpoint", wrapped)

    def restore() -> None:
        setattr(owner, "save_checkpoint", original)

    return restore


def _install_lightzero_learner_metrics_recorder(
    *,
    train_muzero: Any,
    run_id: str,
    attempt_id: str,
    attempt_train_root: Path,
    started_monotonic: float,
    sample_first_n: int = 5,
    sample_interval: int = 1000,
) -> tuple[Any, _LearnerMetricsRecorder] | None:
    """Record lightweight BaseLearner.train metrics without affecting training."""

    globals_map = getattr(train_muzero, "__globals__", {})
    learner_cls = globals_map.get("BaseLearner")
    if not inspect.isclass(learner_cls):
        return None
    owner = next(
        (base for base in inspect.getmro(learner_cls) if "train" in getattr(base, "__dict__", {})),
        None,
    )
    if owner is None:
        return None
    original = owner.__dict__["train"]
    recorder = _LearnerMetricsRecorder(
        run_id=run_id,
        attempt_id=attempt_id,
        attempt_train_root=attempt_train_root,
        started_monotonic=started_monotonic,
        sample_first_n=sample_first_n,
        sample_interval=sample_interval,
    )

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        train_iter_before = _safe_int_or_none(getattr(self, "train_iter", None))
        started = time.perf_counter()
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            recorder.record(
                learner=self,
                exception=exc,
                train_iter_before=train_iter_before,
                train_iter_after=_safe_int_or_none(getattr(self, "train_iter", None)),
                elapsed_sec=time.perf_counter() - started,
            )
            raise
        recorder.record(
            learner=self,
            result=result,
            train_iter_before=train_iter_before,
            train_iter_after=_safe_int_or_none(getattr(self, "train_iter", None)),
            elapsed_sec=time.perf_counter() - started,
        )
        return result

    setattr(owner, "train", wrapped)

    def restore() -> None:
        setattr(owner, "train", original)

    return restore, recorder


def _install_lightzero_full_resume_state_hooks(
    *,
    train_muzero: Any,
    run_id: str,
    attempt_id: str,
    exp_name: Path,
    auto_resume: dict[str, Any],
    attempt_train_root: Path | None = None,
    started_monotonic: float | None = None,
    checkpoint_metadata: Mapping[str, Any] | None = None,
    own_checkpoint_opponent_publisher: Any | None = None,
    learner_metrics_recorder: _LearnerMetricsRecorder | None = None,
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
        restores.append(
            lambda owner=owner, name=name, original=original: setattr(owner, name, original)
        )

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
                try:
                    _save_lightzero_resume_sidecar_state(
                        run_id=run_id,
                        attempt_id=attempt_id,
                        exp_name=exp_name,
                        holder=holder,
                        learner=engine,
                    )
                except Exception as exc:  # pragma: no cover - remote resilience only.
                    print(
                        f"curvyzero resume sidecar save failed: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                if attempt_train_root is not None and started_monotonic is not None:
                    try:
                        _write_checkpoint_progress_latest(
                            run_id=run_id,
                            attempt_id=attempt_id,
                            attempt_train_root=attempt_train_root,
                            exp_name=exp_name,
                            learner=engine,
                            started_monotonic=started_monotonic,
                            source="SaveCkptHook.__call__",
                            checkpoint_metadata=checkpoint_metadata,
                            own_checkpoint_opponent_publisher=own_checkpoint_opponent_publisher,
                            learner_metrics_recorder=learner_metrics_recorder,
                        )
                    except Exception as exc:  # pragma: no cover - remote resilience only.
                        print(
                            "curvyzero checkpoint hook progress write failed: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
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
    checkpoint = lz_checkpoints.latest_lightzero_iteration_checkpoint(
        candidate
        for candidate in lz_checkpoints.collect_lightzero_iteration_checkpoints(
            _lightzero_exp_checkpoint_dirs(Path(exp_name)),
            require_non_empty=True,
        )
        if candidate.iteration == iteration
    )
    if checkpoint is None:
        return {"saved": False, "reason": "matching_iteration_checkpoint_not_found"}

    ckpt_path = checkpoint.path
    checkpoint_exp_name = ckpt_path.parent.parent
    sidecar_name = _lightzero_resume_state_name(iteration)
    sidecar_path = checkpoint_exp_name / LIGHTZERO_RESUME_STATE_DIRNAME / sidecar_name
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


def _checkpoint_payload_optimizer_keys(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    return [
        str(key)
        for key in payload
        if "optim" in str(key).lower() or "scheduler" in str(key).lower()
    ]


def _checkpoint_state_dict_candidate_score(state_dict: Any) -> int:
    if not isinstance(state_dict, dict):
        return -1
    keys = [str(key) for key in state_dict]
    tensor_count = sum(1 for value in state_dict.values() if hasattr(value, "shape"))
    if tensor_count <= 0:
        return -1
    score = tensor_count
    for marker in (
        "representation_network",
        "prediction_network",
        "dynamics_network",
        "encoder",
        "value_head",
        "policy_head",
    ):
        if any(marker in key for key in keys):
            score += 1000
    return score


def _nested_checkpoint_payload_value(payload: Any, state_key: str) -> Any:
    current = payload
    for part in state_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"checkpoint payload does not contain state key {state_key!r}")
        current = current[part]
    return current


def _select_checkpoint_model_state_dict(
    payload: Any,
    *,
    state_key: str | None,
) -> tuple[str, dict[str, Any]]:
    if state_key:
        value = _nested_checkpoint_payload_value(payload, state_key)
        if not isinstance(value, dict):
            raise TypeError(f"checkpoint state key {state_key!r} did not point to a state dict")
        if _checkpoint_state_dict_candidate_score(value) < 0:
            raise ValueError(f"checkpoint state key {state_key!r} did not contain tensors")
        return state_key, value

    candidates: list[tuple[str, Any]] = []
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict", "_model", "_learn_model"):
            if key in payload:
                candidates.append((key, payload[key]))
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            for nested_key in ("model", "state_dict", "model_state_dict", "_model", "_learn_model"):
                if nested_key in value:
                    candidates.append((f"{key}.{nested_key}", value[nested_key]))
        candidates.append(("payload", payload))

    best: tuple[str, dict[str, Any], int] | None = None
    for key, value in candidates:
        score = _checkpoint_state_dict_candidate_score(value)
        if score < 0 or not isinstance(value, dict):
            continue
        if best is None or score > best[2]:
            best = (key, value, score)
    if best is None:
        raise ValueError("checkpoint payload did not contain a model state dict")
    return best[0], best[1]


def _torch_load_checkpoint_payload(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def _prepare_initial_policy_checkpoint_load(
    *,
    initial_policy_checkpoint_ref: str | None = None,
    initial_policy_checkpoint_state_key: str | None = None,
    initial_policy_checkpoint_load_mode: str = DEFAULT_INITIAL_POLICY_CHECKPOINT_LOAD_MODE,
    attempt_train_root: Path,
) -> dict[str, Any] | None:
    checkpoint_ref = str(initial_policy_checkpoint_ref or "").strip()
    if not checkpoint_ref:
        if initial_policy_checkpoint_state_key:
            raise ValueError(
                "initial_policy_checkpoint_state_key requires initial_policy_checkpoint_ref"
            )
        return None
    if initial_policy_checkpoint_load_mode not in INITIAL_POLICY_CHECKPOINT_LOAD_MODE_CHOICES:
        raise ValueError(
            "initial_policy_checkpoint_load_mode must be one of "
            f"{INITIAL_POLICY_CHECKPOINT_LOAD_MODE_CHOICES!r}"
        )
    _reject_mutable_frozen_opponent_checkpoint_ref(checkpoint_ref)
    if not re.fullmatch(r"iteration_\d+\.pth\.tar", Path(checkpoint_ref).name):
        raise ValueError(
            "initial_policy_checkpoint_ref must be an immutable iteration_N.pth.tar ref: "
            f"{checkpoint_ref}"
        )

    source_path, resolution = runs.resolve_mounted_ref_or_path(
        checkpoint_ref,
        mount=RUNS_MOUNT,
        remote_root=REMOTE_ROOT,
    )
    if not source_path.exists():
        raise FileNotFoundError(f"initial policy checkpoint does not exist: {checkpoint_ref}")

    source_ref = resolution.get("source_ref") or (
        runs.file_ref(source_path, mount=RUNS_MOUNT)
        if _path_is_under(source_path.resolve(), RUNS_MOUNT.resolve())
        else None
    )
    report: dict[str, Any] = {
        "enabled": True,
        "input": checkpoint_ref,
        "checkpoint_ref": source_ref or checkpoint_ref,
        "source_path": str(source_path),
        "source_kind": resolution.get("source_kind"),
        "load_mode": initial_policy_checkpoint_load_mode,
        "state_key": initial_policy_checkpoint_state_key,
        "applied": False,
        "load_path": str(source_path),
        "prepared": {
            "kind": "original_checkpoint",
            "optimizer_keys_removed": [],
        },
    }

    if initial_policy_checkpoint_load_mode == INITIAL_POLICY_CHECKPOINT_LOAD_MODE_MATCHING_SHAPE:
        payload = _torch_load_checkpoint_payload(source_path)
        state_key, state_dict = _select_checkpoint_model_state_dict(
            payload,
            state_key=initial_policy_checkpoint_state_key,
        )
        output_dir = attempt_train_root / "initial_policy_checkpoint"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / Path(checkpoint_ref).name
        import torch

        torch.save(
            {
                "model": state_dict,
                "target_model": state_dict,
                "optimizer": {
                    "curvyzero_marker": INITIAL_POLICY_MODEL_ONLY_OPTIMIZER_MARKER,
                },
            },
            output_path,
        )
        report["load_path"] = str(output_path)
        report["prepared"] = {
            "kind": "model_only_checkpoint",
            "source_state_key": state_key,
            "source_optimizer_keys": _checkpoint_payload_optimizer_keys(payload),
            "optimizer_keys_removed": _checkpoint_payload_optimizer_keys(payload),
            "checkpoint_ref": runs.file_ref(output_path, mount=RUNS_MOUNT),
            "fresh_optimizer_intent": True,
            "note": (
                "Only model weights are handed to LightZero. The new run keeps its fresh "
                "optimizer state."
            ),
        }

    report["load_result"] = {
        "loaded": False,
        "status": "pending_train_muzero_load",
        "module_loads": [],
        "optimizer_load_calls": [],
    }
    return report


class _InitialPolicyCheckpointLoadAudit:
    def __init__(self, *, checkpoint: Mapping[str, Any]):
        self.checkpoint = checkpoint
        self.expected_load_path = Path(str(checkpoint.get("load_path") or "")).resolve()
        self.load_mode = str(checkpoint.get("load_mode") or "")
        self.torch_loads: list[dict[str, Any]] = []
        self.module_loads: list[dict[str, Any]] = []
        self.optimizer_load_calls: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.initial_load_seen = False
        self.module_load_slots_remaining = 0
        self.optimizer_load_slots_remaining = 0

    def record_torch_load(self, target: Path | None) -> None:
        if target is None or len(self.torch_loads) >= 20:
            return
        resolved = target.resolve()
        is_initial = bool(self.expected_load_path) and resolved == self.expected_load_path
        if is_initial:
            self.initial_load_seen = True
            self.module_load_slots_remaining = 4
            self.optimizer_load_slots_remaining = 1
        self.torch_loads.append({"path": str(resolved), "initial_policy_checkpoint": is_initial})

    def consume_initial_module_load_slot(self) -> bool:
        if not self.initial_load_seen or self.module_load_slots_remaining <= 0:
            return False
        self.module_load_slots_remaining -= 1
        return True

    def should_skip_optimizer_load(self, state_dict: Any) -> bool:
        if (
            isinstance(state_dict, dict)
            and state_dict.get("curvyzero_marker") == INITIAL_POLICY_MODEL_ONLY_OPTIMIZER_MARKER
        ):
            return True
        if self.initial_load_seen and self.optimizer_load_slots_remaining > 0:
            self.optimizer_load_slots_remaining -= 1
            return self.load_mode == INITIAL_POLICY_CHECKPOINT_LOAD_MODE_MATCHING_SHAPE
        return False

    def prepare_module_state_dict(self, module: Any, state_dict: Any) -> tuple[Any, dict[str, Any]]:
        if self.load_mode != INITIAL_POLICY_CHECKPOINT_LOAD_MODE_MATCHING_SHAPE:
            return state_dict, {"filtered": False}
        if not isinstance(state_dict, dict):
            return state_dict, {"filtered": False, "reason": "state_dict_not_dict"}
        current_state = getattr(module, "state_dict", None)
        if not callable(current_state):
            return state_dict, {"filtered": False, "reason": "module_has_no_state_dict"}
        target_state = current_state()
        matched: dict[str, Any] = {}
        skipped_shape: list[str] = []
        skipped_missing: list[str] = []
        for key, value in state_dict.items():
            key_text = str(key)
            target_value = target_state.get(key_text)
            if target_value is None:
                skipped_missing.append(key_text)
                continue
            if getattr(value, "shape", None) != getattr(target_value, "shape", None):
                skipped_shape.append(key_text)
                continue
            matched[key_text] = value
        return matched, {
            "filtered": True,
            "input_key_count": len(state_dict),
            "matched_key_count": len(matched),
            "skipped_missing_count": len(skipped_missing),
            "skipped_shape_count": len(skipped_shape),
            "skipped_missing_sample": skipped_missing[:20],
            "skipped_shape_sample": skipped_shape[:20],
        }

    def record_module_load(
        self,
        module: Any,
        state_dict: Any,
        *,
        strict: Any,
        filter_report: Mapping[str, Any] | None = None,
    ) -> None:
        if len(self.module_loads) >= 20:
            return
        score = _checkpoint_state_dict_candidate_score(state_dict)
        keys = [str(key) for key in state_dict] if isinstance(state_dict, dict) else []
        self.module_loads.append(
            {
                "module_type": f"{type(module).__module__}.{type(module).__qualname__}",
                "strict": strict,
                "state_dict_key_count": len(keys),
                "state_dict_key_sample": keys[:12],
                "meaningful_model_load": score >= 1000,
                "filter_report": dict(filter_report or {}),
            }
        )

    def record_optimizer_load(
        self,
        optimizer: Any,
        state_dict: Any,
        *,
        skipped: bool,
    ) -> None:
        if len(self.optimizer_load_calls) >= 20:
            return
        keys = sorted(str(key) for key in state_dict) if isinstance(state_dict, dict) else []
        self.optimizer_load_calls.append(
            {
                "optimizer_type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
                "state_dict_keys": keys[:20],
                "skipped_to_preserve_fresh_optimizer": skipped,
            }
        )

    def summary(self) -> dict[str, Any]:
        optimizer_loaded = any(
            not row.get("skipped_to_preserve_fresh_optimizer") for row in self.optimizer_load_calls
        )
        module_loads = []
        for row in self.module_loads:
            item = dict(row)
            item["fresh_optimizer_preserved"] = not optimizer_loaded
            module_loads.append(item)
        loaded = any(row.get("meaningful_model_load") for row in module_loads)
        return {
            "loaded": loaded,
            "fresh_optimizer_preserved": not optimizer_loaded,
            "module_loads": module_loads,
            "optimizer_load_calls": self.optimizer_load_calls,
            "torch_loads": self.torch_loads,
            "errors": self.errors,
        }


def _install_initial_policy_checkpoint_load_audit(
    audit: _InitialPolicyCheckpointLoadAudit,
) -> Any:
    import torch

    original_torch_load = torch.load
    original_module_load_state_dict = torch.nn.Module.load_state_dict
    original_optimizer_load_state_dict = torch.optim.Optimizer.load_state_dict

    def wrapped_torch_load(*args: Any, **kwargs: Any) -> Any:
        audit.record_torch_load(_torch_load_target_path(args, kwargs))
        return original_torch_load(*args, **kwargs)

    def wrapped_module_load_state_dict(
        self: Any, state_dict: Any, *args: Any, **kwargs: Any
    ) -> Any:
        strict = kwargs.get("strict", args[0] if args else True)
        filter_report: Mapping[str, Any] | None = None
        if audit.consume_initial_module_load_slot():
            state_dict, filter_report = audit.prepare_module_state_dict(self, state_dict)
            if (
                filter_report.get("filtered")
                and audit.load_mode == INITIAL_POLICY_CHECKPOINT_LOAD_MODE_MATCHING_SHAPE
            ):
                if args:
                    args = (False, *args[1:])
                else:
                    kwargs = dict(kwargs)
                    kwargs["strict"] = False
                strict = False
        try:
            result = original_module_load_state_dict(self, state_dict, *args, **kwargs)
        except Exception as exc:
            audit.errors.append(f"module load failed: {type(exc).__name__}: {exc}")
            raise
        audit.record_module_load(
            self,
            state_dict,
            strict=strict,
            filter_report=filter_report,
        )
        return result

    def wrapped_optimizer_load_state_dict(self: Any, state_dict: Any) -> Any:
        skip = audit.should_skip_optimizer_load(state_dict)
        audit.record_optimizer_load(self, state_dict, skipped=skip)
        if skip:
            return None
        return original_optimizer_load_state_dict(self, state_dict)

    torch.load = wrapped_torch_load
    torch.nn.Module.load_state_dict = wrapped_module_load_state_dict
    torch.optim.Optimizer.load_state_dict = wrapped_optimizer_load_state_dict

    def restore() -> None:
        torch.optim.Optimizer.load_state_dict = original_optimizer_load_state_dict
        torch.nn.Module.load_state_dict = original_module_load_state_dict
        torch.load = original_torch_load

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


def _seed_training_process(seed: int) -> dict[str, Any]:
    """Seed process-local RNGs before LightZero and Curvy sidecars are built."""
    random.seed(int(seed))
    seeded: dict[str, Any] = {
        "schema_id": "curvyzero_training_process_seed/v0",
        "seed": int(seed),
        "python_random": True,
        "numpy_random": False,
        "torch_cpu": False,
        "torch_cuda_all": False,
        "torch_deterministic_warn_only": False,
        "cudnn_deterministic": None,
        "cudnn_benchmark": None,
        "errors": [],
    }
    try:
        import numpy as np

        np.random.seed(int(seed))
        seeded["numpy_random"] = True
    except Exception as exc:  # pragma: no cover - diagnostic only.
        seeded["errors"].append(f"numpy: {type(exc).__name__}: {exc}")
    try:
        import torch

        torch.manual_seed(int(seed))
        seeded["torch_cpu"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
            seeded["torch_cuda_all"] = True
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
            seeded["torch_deterministic_warn_only"] = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            seeded["cudnn_deterministic"] = True
            seeded["cudnn_benchmark"] = False
    except Exception as exc:  # pragma: no cover - diagnostic only.
        seeded["errors"].append(f"torch: {type(exc).__name__}: {exc}")
    return seeded


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
        field: details.get("length") for field, details in fields.items() if details.get("present")
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
        metadata = (
            list(result_items[1])
            if len(result_items) > 1 and isinstance(result_items[1], (list, tuple))
            else []
        )
        return segments, metadata, "tuple/list first item"
    if any(hasattr(item, "action_segment") for item in result_items):
        return result_items, [], "direct list"
    return [], [], "collector result did not expose GameSegment-like items"


def _audit_sample_summary(
    result: Any, *, sample_call_index: int, args: tuple[Any, ...]
) -> dict[str, Any]:
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
            audit.add_note(
                f"could not import lzero.worker.MuZeroCollector: {type(exc).__name__}: {exc}"
            )
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


def _install_curvyzero_rnd_reward_model(
    *,
    train_muzero: Any,
    exploration_bonus_spec: xb.ExplorationBonusSpec,
) -> Any | None:
    if not exploration_bonus_spec.enabled:
        return None
    globals_map = getattr(train_muzero, "__globals__", None)
    if not isinstance(globals_map, dict):
        raise RuntimeError("selected LightZero entrypoint does not expose patchable globals")
    missing = object()
    restore_actions: list[Any] = []

    def patch_attr(holder: Any, name: str) -> None:
        previous = getattr(holder, name, missing)
        setattr(holder, name, xb.CurvyRNDRewardModel)

        def restore_attr() -> None:
            if previous is missing:
                try:
                    delattr(holder, name)
                except AttributeError:
                    pass
            else:
                setattr(holder, name, previous)

        restore_actions.append(restore_attr)

    previous_global = globals_map.get("RNDRewardModel", missing)
    globals_map["RNDRewardModel"] = xb.CurvyRNDRewardModel

    def restore_global() -> None:
        if previous_global is missing:
            globals_map.pop("RNDRewardModel", None)
        else:
            globals_map["RNDRewardModel"] = previous_global

    restore_actions.append(restore_global)
    try:
        rnd_module = importlib.import_module("lzero.reward_model.rnd_reward_model")
    except Exception:
        rnd_module = None
    if rnd_module is not None:
        patch_attr(rnd_module, "RNDRewardModel")

    def restore() -> None:
        for restore_action in reversed(restore_actions):
            restore_action()

    return restore


def _summarize_rnd_reward_model_metrics(
    *,
    enabled: bool,
    latest_path: Path,
    jsonl_path: Path,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_id": "curvyzero_rnd_reward_model_metrics_scan/v0",
        "enabled": bool(enabled),
        "latest_ref": runs.file_ref(latest_path, mount=RUNS_MOUNT),
        "jsonl_ref": runs.file_ref(jsonl_path, mount=RUNS_MOUNT),
        "latest_exists": latest_path.exists(),
        "jsonl_exists": jsonl_path.exists(),
        "latest": None,
        "event_count": 0,
        "last_event_reasons": [],
        "errors": [],
    }
    if latest_path.exists():
        try:
            summary["latest"] = json.loads(latest_path.read_text(encoding="utf-8"))
            summary["latest_file"] = runs.file_summary(latest_path, mount=RUNS_MOUNT)
        except Exception as exc:
            summary["errors"].append(
                f"latest metrics read failed: {type(exc).__name__}: {exc}"
            )
    if jsonl_path.exists():
        try:
            event_count = 0
            reasons: list[str | None] = []
            last_valid_event: dict[str, Any] | None = None
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    event_count += 1
                    if len(reasons) >= 20:
                        reasons.pop(0)
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict):
                            reasons.append(row.get("reason"))
                            last_valid_event = row
                        else:
                            reasons.append(None)
                    except json.JSONDecodeError:
                        reasons.append("json_decode_error")
            summary["event_count"] = event_count
            summary["last_event_reasons"] = reasons
            summary["jsonl_file"] = runs.file_summary(jsonl_path, mount=RUNS_MOUNT)
            if summary["latest"] is None and last_valid_event is not None:
                summary["latest"] = last_valid_event
                summary["latest_source"] = "jsonl_tail_fallback"
        except Exception as exc:
            summary["errors"].append(
                f"jsonl metrics read failed: {type(exc).__name__}: {exc}"
            )
    return _to_plain(summary)


def _validate_required_rnd_reward_model_metrics(
    metrics: Mapping[str, Any],
    *,
    weight: float,
) -> list[str]:
    problems: list[str] = []
    latest = metrics.get("latest") if isinstance(metrics.get("latest"), Mapping) else None
    if latest is None:
        return ["required RND reward-model metrics were not written"]
    if int(latest.get("collect_data_calls") or 0) < 1:
        problems.append("required RND metrics missing collect_data call proof")
    if int(latest.get("train_with_data_calls") or 0) < 1:
        problems.append("required RND metrics missing train_with_data call proof")
    if int(latest.get("estimate_calls") or 0) < 1:
        problems.append("required RND metrics missing estimate call proof")
    if int(latest.get("train_cnt_rnd") or 0) < 1:
        problems.append("required RND metrics missing predictor update proof")
    predictor_before = latest.get("last_predictor_hash_before_train")
    predictor_after = latest.get("last_predictor_hash_after_train")
    if not predictor_before or not predictor_after or predictor_before == predictor_after:
        problems.append("required RND metrics show predictor weights did not change")
    target_before = latest.get("last_target_hash_before_train")
    target_after = latest.get("last_target_hash_after_train")
    if target_before and target_after and target_before != target_after:
        problems.append("required RND metrics show frozen target weights changed")
    if float(weight) == 0.0 and latest.get("last_target_reward_changed") is not False:
        problems.append("required RND metrics did not prove weight=0 target rewards unchanged")
    if float(weight) > 0.0:
        if latest.get("last_target_reward_changed") is not True:
            problems.append("required RND metrics did not prove positive-weight target rewards changed")
        try:
            delta_abs_max = float(latest.get("last_target_reward_delta_abs_max") or 0.0)
        except (TypeError, ValueError):
            delta_abs_max = 0.0
        if delta_abs_max <= 0.0:
            problems.append("required RND metrics recorded zero positive-weight target delta")
    return problems


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
    profile_volume_commit: bool,
    save_ckpt_after_iter: int,
    commit_on_checkpoint: bool,
    stop_after_learner_train_calls: int,
    env_variant: str,
    reward_variant: str,
    reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
    source_state_trail_render_mode: str,
    source_state_bonus_render_mode: str,
    policy_observation_backend: str,
    collect_search_backend: str = DEFAULT_COLLECT_SEARCH_BACKEND,
    collect_search_ctree_backend: str = DEFAULT_COLLECT_SEARCH_CTREE_BACKEND,
    learner_seat_mode: str,
    ego_action_straight_override_probability: float,
    control_noise_profile_id: str,
    policy_action_repeat_min: int,
    policy_action_repeat_max: int,
    policy_action_repeat_extra_probability: float,
    disable_death_for_profile: bool,
    opponent_death_mode: str,
    opponent_runtime_mode: str,
    env_telemetry_stride: int,
    env_manager_type: str,
    lightzero_multi_gpu: bool,
    opponent_policy_kind: str,
    opponent_use_cuda: bool,
    opponent_checkpoint_ref: str | None,
    opponent_snapshot_ref: str | None,
    opponent_checkpoint_report_ref: str | None,
    opponent_checkpoint_state_key: str | None,
    opponent_mixture_spec: str | None,
    opponent_assignment_ref: str | None,
    initial_policy_checkpoint_ref: str | None = None,
    initial_policy_checkpoint_state_key: str | None = None,
    initial_policy_checkpoint_load_mode: str = DEFAULT_INITIAL_POLICY_CHECKPOINT_LOAD_MODE,
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
    background_gif_collect_temperature: float = DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE,
    background_gif_collect_epsilon: float = DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON,
    background_gif_checkpoint_seed_mixing_enabled: bool = (
        DEFAULT_BACKGROUND_GIF_CHECKPOINT_SEED_MIXING_ENABLED
    ),
    opponent_assignment_refresh_interval_train_iter: int = (
        DEFAULT_OPPONENT_ASSIGNMENT_REFRESH_INTERVAL_TRAIN_ITER
    ),
    opponent_assignment_refresh_ref: str | None = None,
    model_support_cap: int | None = DEFAULT_MODEL_SUPPORT_CAP,
    td_steps: int | None = DEFAULT_TD_STEPS,
    own_checkpoint_opponent_refresh_enabled: bool = (
        DEFAULT_OWN_CHECKPOINT_OPPONENT_REFRESH_ENABLED
    ),
    exploration_bonus_mode: str = xb.EXPLORATION_BONUS_MODE_NONE,
    exploration_bonus_weight: float = 0.0,
    exploration_bonus_feature_source: str = xb.RND_FEATURE_SOURCE_POLICY_GRAY64_LATEST_V0,
    exploration_bonus_rnd_batch_size: int = 64,
    exploration_bonus_rnd_update_per_collect: int = xb.RND_DEFAULT_UPDATE_PER_COLLECT,
    exploration_bonus_rnd_buffer_size: int = 100_000,
    exploration_bonus_rnd_learning_rate: float = 3e-4,
    exploration_bonus_rnd_weight_decay: float = 1e-4,
    exploration_bonus_rnd_input_norm: bool = False,
    require_rnd_metrics: bool = False,
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
    exploration_bonus_spec = xb.normalize_exploration_bonus_spec(
        mode=exploration_bonus_mode,
        weight=exploration_bonus_weight,
        feature_source=exploration_bonus_feature_source,
        rnd_batch_size=exploration_bonus_rnd_batch_size,
        rnd_update_per_collect=exploration_bonus_rnd_update_per_collect,
        rnd_buffer_size=exploration_bonus_rnd_buffer_size,
        rnd_learning_rate=exploration_bonus_rnd_learning_rate,
        rnd_weight_decay=exploration_bonus_rnd_weight_decay,
        rnd_input_norm=exploration_bonus_rnd_input_norm,
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
    reward_outcome_alpha = _normalize_reward_outcome_alpha(reward_outcome_alpha)
    if opponent_death_mode not in OPPONENT_DEATH_MODES:
        raise ValueError(f"opponent_death_mode must be one of {OPPONENT_DEATH_MODES!r}")
    if opponent_runtime_mode not in OPPONENT_RUNTIME_MODES:
        raise ValueError(f"opponent_runtime_mode must be one of {OPPONENT_RUNTIME_MODES!r}")
    reward_policy = _reward_policy_for_variant(
        env_variant=env_variant,
        reward_variant=reward_variant,
        reward_outcome_alpha=reward_outcome_alpha,
    )
    source_state_trail_render_mode = _validate_source_state_trail_render_mode(
        source_state_trail_render_mode
    )
    source_state_bonus_render_mode = _validate_source_state_bonus_render_mode(
        source_state_bonus_render_mode
    )
    policy_observation_backend = _validate_policy_observation_backend(policy_observation_backend)
    collect_search_backend = _validate_collect_search_backend(collect_search_backend)
    collect_search_ctree_backend = _validate_collect_search_ctree_backend(
        collect_search_ctree_backend
    )
    learner_seat_mode = _validate_learner_seat_mode(learner_seat_mode)
    lightzero_target_config = _lightzero_target_config_for_reward(
        env_variant=env_variant,
        reward_variant=reward_variant,
        source_max_steps=source_max_steps,
        reward_outcome_alpha=reward_outcome_alpha,
        model_support_cap=model_support_cap,
        td_steps=td_steps,
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
    if env_manager_type in CURVYZERO_BATCHED_PROFILE_ENV_MANAGER_TYPES:
        if mode != "profile":
            raise ValueError(f"{env_manager_type} env manager is profile-only")
        if int(collector_env_num) % 2 != 0 or int(evaluator_env_num) % 2 != 0:
            raise ValueError(
                f"{env_manager_type} requires even collector/evaluator env counts"
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
    if (
        env_variant
        in (
            ENV_VARIANT_TURN_COMMIT,
            ENV_VARIANT_SOURCE_STATE_TURN_COMMIT,
        )
        and opponent_policy_kind != OPPONENT_POLICY_KIND_FIXED_STRAIGHT
    ):
        raise ValueError(f"{env_variant} env_variant does not use frozen opponent checkpoints")
    if float(decision_ms) <= 0.0:
        raise ValueError("decision_ms must be positive")
    if mode in {"train", "dry"}:
        _validate_trusted_source_state_action_cadence(
            env_variant=env_variant,
            decision_ms=decision_ms,
            context=f"mode={mode!r}",
        )
    if not 0.0 <= float(ego_action_straight_override_probability) <= 1.0:
        raise ValueError("ego_action_straight_override_probability must be in [0, 1]")
    if int(policy_action_repeat_min) < 1:
        raise ValueError("policy_action_repeat_min must be at least 1")
    if int(policy_action_repeat_max) < int(policy_action_repeat_min):
        raise ValueError(
            "policy_action_repeat_max must be greater than or equal to policy_action_repeat_min"
        )
    if not 0.0 <= float(policy_action_repeat_extra_probability) <= 1.0:
        raise ValueError("policy_action_repeat_extra_probability must be in [0, 1]")
    if bool(disable_death_for_profile) and mode != "profile":
        raise ValueError("disable_death_for_profile is only allowed in mode='profile'")
    if collect_search_backend != COLLECT_SEARCH_BACKEND_STOCK:
        if mode not in {"profile", "train"}:
            raise ValueError(
                "non-stock collect_search_backend is only allowed in mode='profile' "
                "or mode='train'"
            )
        if not _compute_uses_cuda(compute):
            raise ValueError("direct_ctree_gpu_latent collect_search_backend requires GPU compute")
        if (
            mode == "train"
            and collect_search_ctree_backend != COLLECT_SEARCH_CTREE_BACKEND_LIGHTZERO
        ):
            raise ValueError(
                "mode='train' with collect_search_backend='direct_ctree_gpu_latent' "
                "uses LightZero CTree only; non-LightZero collect_search_ctree_backend "
                "is profile-only"
            )
    elif collect_search_ctree_backend != COLLECT_SEARCH_CTREE_BACKEND_LIGHTZERO:
        raise ValueError(
            "non-lightzero collect_search_ctree_backend requires "
            "collect_search_backend='direct_ctree_gpu_latent'"
        )
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
        raise ValueError("background_gif_max_steps must be non-negative; 0 means no GIF step cap")
    if int(stop_after_learner_train_calls) < 0:
        raise ValueError("stop_after_learner_train_calls must be non-negative")
    if int(lightzero_eval_freq) < 0:
        raise ValueError("lightzero_eval_freq must be non-negative")
    if int(opponent_assignment_refresh_interval_train_iter) < 0:
        raise ValueError("opponent_assignment_refresh_interval_train_iter must be non-negative")
    if bool(own_checkpoint_opponent_refresh_enabled):
        if env_variant != ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT:
            raise ValueError(
                "own_checkpoint_opponent_refresh_enabled requires "
                "env_variant='source_state_fixed_opponent'"
            )
        if int(opponent_assignment_refresh_interval_train_iter) <= 0:
            raise ValueError(
                "own_checkpoint_opponent_refresh_enabled requires a positive "
                "opponent_assignment_refresh_interval_train_iter"
            )
        if opponent_assignment_refresh_ref and str(opponent_assignment_refresh_ref).startswith(
            _CONTROL_REF_PREFIX
        ):
            raise ValueError(
                "own checkpoint opponent refresh must use a run-local runs: pointer, "
                "not a control: pointer"
            )
        if not opponent_assignment_refresh_ref:
            opponent_assignment_refresh_ref = _own_checkpoint_opponent_refresh_pointer_ref(
                run_id,
                attempt_id,
            )
    if (
        int(opponent_assignment_refresh_interval_train_iter) > 0
        and not opponent_assignment_refresh_ref
    ):
        raise ValueError(
            "opponent_assignment_refresh_ref is required when "
            "opponent_assignment_refresh_interval_train_iter is positive"
        )
    if (
        initial_policy_checkpoint_load_mode not in INITIAL_POLICY_CHECKPOINT_LOAD_MODE_CHOICES
        and initial_policy_checkpoint_ref
    ):
        raise ValueError(
            "initial_policy_checkpoint_load_mode must be one of "
            f"{INITIAL_POLICY_CHECKPOINT_LOAD_MODE_CHOICES!r}"
        )
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
    assignment_refresh_events_path = attempt_train_root / "opponent_assignment_refresh_events.jsonl"
    trainer_lineage_path = lineage_events_path(attempt_train_root)
    rnd_metrics_latest_path = attempt_train_root / "rnd_reward_model_metrics_latest.json"
    rnd_metrics_jsonl_path = attempt_train_root / "rnd_reward_model_metrics.jsonl"
    exp_name_ref = attempt_train_ref / "lightzero_exp"
    exp_name = Path(exp_name_ref.as_posix())
    env_spec = _env_variant_spec(env_variant)
    opponent_checkpoint = _resolve_opponent_checkpoint_for_env(
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
        opponent_checkpoint_report_ref=opponent_checkpoint_report_ref,
    )
    assignment_ref_uses_volume = bool(
        opponent_assignment_ref
        and str(opponent_assignment_ref).startswith((_CONTROL_REF_PREFIX, _RUNS_REF_PREFIX))
    )
    opponent_assignment = _resolve_opponent_assignment_for_env(
        opponent_assignment_ref=opponent_assignment_ref,
        reload_volume_before_read=assignment_ref_uses_volume,
        reload_checkpoint_volume_before_read=bool(opponent_assignment_ref),
    )
    if opponent_assignment is not None and opponent_mixture_spec is not None:
        raise ValueError("opponent_assignment_ref cannot be combined with opponent_mixture_spec")
    opponent_mixture = (
        opponent_assignment["opponent_mixture"]
        if opponent_assignment is not None
        else _resolve_opponent_mixture_for_env(opponent_mixture_spec=opponent_mixture_spec)
    )
    opponent_assignment_context = _opponent_assignment_context_for_env(opponent_assignment)
    if opponent_assignment is not None:
        _append_trainer_assignment_loaded_lineage(
            trainer_lineage_path,
            status="initial",
            run_id=run_id,
            attempt_id=attempt_id,
            opponent_assignment_ref=opponent_assignment_ref,
            opponent_assignment=opponent_assignment,
        )
    if opponent_mixture is not None:
        if env_variant != ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT:
            raise ValueError(
                "opponent_mixture_spec is only supported with "
                "env_variant='source_state_fixed_opponent'"
            )
        if opponent_checkpoint_ref:
            raise ValueError(
                "opponent_mixture_spec cannot be combined with top-level "
                "opponent_checkpoint_ref; put frozen refs inside mixture entries"
            )
    opponent_training_relation = _opponent_training_relation_for_surface(
        env_variant=env_variant,
        opponent_policy_kind=opponent_policy_kind,
        env_spec=env_spec,
        opponent_mixture=opponent_mixture,
    )
    natural_bonus_spawn = bool(TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN)
    initial_policy_checkpoint = _prepare_initial_policy_checkpoint_load(
        initial_policy_checkpoint_ref=initial_policy_checkpoint_ref,
        initial_policy_checkpoint_state_key=initial_policy_checkpoint_state_key,
        initial_policy_checkpoint_load_mode=initial_policy_checkpoint_load_mode,
        attempt_train_root=attempt_train_root,
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
        "decision_source_frames": int(DEFAULT_DECISION_SOURCE_FRAMES),
        "source_physics_step_ms": float(DEFAULT_SOURCE_PHYSICS_STEP_MS),
        "source_max_steps_semantics": "source_physics_steps",
        "collector_env_num": int(collector_env_num),
        "evaluator_env_num": int(evaluator_env_num),
        "n_evaluator_episode": int(n_evaluator_episode),
        "n_episode": int(n_episode),
        "num_simulations": int(num_simulations),
        "batch_size": int(batch_size),
        "model_support_cap": None if model_support_cap is None else int(model_support_cap),
        "td_steps": None if td_steps is None else int(td_steps),
        "lightzero_eval_freq": int(lightzero_eval_freq),
        "skip_lightzero_eval_in_profile": bool(skip_lightzero_eval_in_profile),
        "profile_cuda_sync_enabled": bool(profile_cuda_sync_enabled),
        "profile_allow_auto_resume": bool(profile_allow_auto_resume),
        "profile_volume_commit": bool(profile_volume_commit),
        "lightzero_multi_gpu": bool(lightzero_multi_gpu),
        "save_ckpt_after_iter": int(save_ckpt_after_iter),
        "commit_on_checkpoint": bool(commit_on_checkpoint),
        "stop_after_learner_train_calls": int(stop_after_learner_train_calls),
        "env_variant": env_variant,
        "env_type": env_spec["env_type"],
        "env_id": env_spec["env_id"],
        "action_space_size": int(env_spec.get("action_space_size", 3)),
        "reward_variant": reward_variant,
        "reward_outcome_alpha": float(reward_outcome_alpha),
        "reward_schema_id": reward_policy["reward_schema_id"],
        "reward_policy": reward_policy,
        "lightzero_target_config": lightzero_target_config,
        "exploration_bonus": exploration_bonus_spec.as_dict(),
        "exploration_bonus_mode": exploration_bonus_spec.mode,
        "exploration_bonus_weight": exploration_bonus_spec.weight,
        "exploration_bonus_feature_source": exploration_bonus_spec.feature_source,
        "exploration_bonus_rnd_batch_size": exploration_bonus_spec.rnd_batch_size,
        "exploration_bonus_rnd_update_per_collect": (
            exploration_bonus_spec.rnd_update_per_collect
        ),
        "exploration_bonus_rnd_buffer_size": exploration_bonus_spec.rnd_buffer_size,
        "exploration_bonus_rnd_learning_rate": exploration_bonus_spec.rnd_learning_rate,
        "exploration_bonus_rnd_weight_decay": exploration_bonus_spec.rnd_weight_decay,
        "exploration_bonus_rnd_input_norm": exploration_bonus_spec.rnd_input_norm,
        "require_rnd_metrics": bool(require_rnd_metrics),
        "rnd_reward_model_metrics": {
            "enabled": exploration_bonus_spec.enabled,
            "required": bool(require_rnd_metrics),
            "latest_ref": runs.file_ref(rnd_metrics_latest_path, mount=RUNS_MOUNT),
            "jsonl_ref": runs.file_ref(rnd_metrics_jsonl_path, mount=RUNS_MOUNT),
        },
        "trainer_entrypoint": xb.lightzero_trainer_entrypoint_ref(exploration_bonus_spec),
        "source_state_trail_render_mode": source_state_trail_render_mode,
        "source_state_bonus_render_mode": source_state_bonus_render_mode,
        "policy_observation_backend": policy_observation_backend,
        "collect_search_backend": collect_search_backend,
        "collect_search_ctree_backend": collect_search_ctree_backend,
        "collect_search_backend_fallback_policy": (
            "allow_profile_fallback"
            if mode == "profile" and collect_search_backend != COLLECT_SEARCH_BACKEND_STOCK
            else "fail_closed_when_non_stock"
            if collect_search_backend != COLLECT_SEARCH_BACKEND_STOCK
            else "stock_backend_no_hook"
        ),
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
        "ego_action_straight_override_probability": float(ego_action_straight_override_probability),
        "policy_action_repeat_min": int(policy_action_repeat_min),
        "policy_action_repeat_max": int(policy_action_repeat_max),
        "policy_action_repeat_extra_probability": float(policy_action_repeat_extra_probability),
        "policy_action_repeat_semantics": (
            "repeat_selected_policy_action_inside_one_lightzero_env_step"
        ),
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
        "death_mode": ("profile_no_death" if disable_death_for_profile else "normal"),
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
        "fixed_opponent_is_two_seat_self_play": env_spec["fixed_opponent_is_two_seat_self_play"],
        "browser_pixel_fidelity": env_spec["browser_pixel_fidelity"],
        "uses_ale": env_spec["uses_ale"],
        "visual_surface": env_spec["visual_surface"],
        "visual_truth_level": env_spec["visual_truth_level"],
        "visual_source_state_backed": env_spec["visual_source_state_backed"],
        "control_noise_profile_id": str(control_noise_profile_id),
        "env_telemetry_stride": int(env_telemetry_stride),
        "env_manager_type": env_manager_type,
        "profile_label": "profile" if mode == "profile" else None,
        "profile_env_timing_enabled": bool(mode == "profile"),
        "learning_proof": False,
        "source_fidelity_claim": env_spec["source_fidelity_claim"],
        "opponent_policy_kind": opponent_policy_kind,
        "opponent_use_cuda": bool(opponent_use_cuda),
        "opponent_training_relation": opponent_training_relation,
        "current_policy_self_play": env_spec["current_policy_self_play"],
        "current_policy_self_play_blocker": env_spec["current_policy_self_play_blocker"],
        "current_policy_self_play_caveat": env_spec["current_policy_self_play_caveat"],
        "trusted_current_policy_self_play": env_spec["trusted_current_policy_self_play"],
        "simultaneous_game_theory_claim": env_spec["simultaneous_game_theory_claim"],
        "opponent_checkpoint_ref": opponent_checkpoint_ref,
        "opponent_snapshot_ref": opponent_snapshot_ref,
        "opponent_checkpoint_report_ref": (
            opponent_checkpoint["checkpoint_ref"]
            if opponent_checkpoint
            else opponent_checkpoint_report_ref
        ),
        "opponent_checkpoint_state_key": opponent_checkpoint_state_key,
        "opponent_mixture_enabled": opponent_mixture is not None,
        "opponent_mixture_spec": opponent_mixture_spec,
        "opponent_mixture": opponent_mixture,
        "opponent_assignment_ref": opponent_assignment_ref,
        "opponent_assignment_context": opponent_assignment_context,
        "opponent_assignment_refresh": {
            "enabled": int(opponent_assignment_refresh_interval_train_iter) > 0
            and bool(opponent_assignment_refresh_ref),
            "mode": "assignment_or_pointer_ref_refresh",
            "interval_train_iter": int(opponent_assignment_refresh_interval_train_iter),
            "pending_assignment_ref": opponent_assignment_refresh_ref,
            "own_checkpoint_opponent_refresh_enabled": bool(
                own_checkpoint_opponent_refresh_enabled
            ),
            "events_ref": runs.file_ref(assignment_refresh_events_path, mount=RUNS_MOUNT),
            "control_plane_caveat": (
                "Refresh may read either an immutable assignment JSON or a "
                "curvyzero_opponent_assignment_refresh_pointer/v0 JSON that "
                "names the immutable assignment and expected sha256. The pointer "
                "write must be atomic enough for Modal Volume readers."
            ),
        },
        "opponent_assignment": (
            {
                key: opponent_assignment.get(key)
                for key in (
                    "assignment_id",
                    "source_epoch",
                    "source_ref",
                    "assignment_ref",
                    "assignment_sha256",
                    "assignment_file",
                )
            }
            if opponent_assignment is not None
            else None
        ),
        "initial_policy_checkpoint": (
            {
                key: initial_policy_checkpoint.get(key)
                for key in (
                    "enabled",
                    "input",
                    "checkpoint_ref",
                    "source_kind",
                    "load_mode",
                    "state_key",
                    "applied",
                    "load_path",
                    "prepared",
                    "load_result",
                )
            }
            if initial_policy_checkpoint is not None
            else None
        ),
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
        "background_gif_collect_temperature": float(background_gif_collect_temperature),
        "background_gif_collect_epsilon": float(background_gif_collect_epsilon),
        "gif_browser_run_marker_enabled": gif_browser_run_marker_enabled,
        "gif_browser_run_marker_ref": (
            runs.gif_browser_run_marker_ref(TASK_ID, run_id).as_posix()
            if gif_browser_run_marker_enabled
            else None
        ),
    }
    command["process_seed"] = _seed_training_process(int(seed))
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
    entrypoint_name = xb.lightzero_entrypoint_name(exploration_bonus_spec)
    train_muzero = getattr(entry_module, entrypoint_name, None)
    trainer_entrypoint_ref = xb.lightzero_trainer_entrypoint_ref(exploration_bonus_spec)
    if train_muzero is None:
        problems.append(f"LightZero entrypoint is missing: {trainer_entrypoint_ref}")
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
        model_support_cap=model_support_cap,
        td_steps=td_steps,
        lightzero_eval_freq=lightzero_eval_freq,
        lightzero_multi_gpu=lightzero_multi_gpu,
        max_train_iter=max_train_iter,
        save_ckpt_after_iter=save_ckpt_after_iter,
        env_variant=env_variant,
        ego_action_straight_override_probability=ego_action_straight_override_probability,
        control_noise_profile_id=control_noise_profile_id,
        policy_action_repeat_min=policy_action_repeat_min,
        policy_action_repeat_max=policy_action_repeat_max,
        policy_action_repeat_extra_probability=policy_action_repeat_extra_probability,
        disable_death_for_profile=disable_death_for_profile,
        opponent_death_mode=opponent_death_mode,
        opponent_runtime_mode=opponent_runtime_mode,
        env_telemetry_stride=env_telemetry_stride,
        profile_env_timing_enabled=(mode == "profile"),
        env_manager_type=env_manager_type,
        opponent_policy_kind=opponent_policy_kind,
        opponent_use_cuda=opponent_use_cuda,
        opponent_checkpoint=opponent_checkpoint,
        opponent_snapshot_ref=opponent_snapshot_ref,
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
        opponent_mixture=opponent_mixture,
        opponent_assignment_context=opponent_assignment_context,
        reward_variant=reward_variant,
        reward_outcome_alpha=reward_outcome_alpha,
        exploration_bonus=exploration_bonus_spec.as_dict(),
        source_state_trail_render_mode=source_state_trail_render_mode,
        source_state_bonus_render_mode=source_state_bonus_render_mode,
        policy_observation_backend=policy_observation_backend,
        learner_seat_mode=learner_seat_mode,
        natural_bonus_spawn=natural_bonus_spawn,
    )
    if exploration_bonus_spec.enabled:
        patched["patches"].append(
            _set_or_add_path(
                patched["main_config"],
                ("reward_model", "curvyzero_metrics_latest_path"),
                str(rnd_metrics_latest_path),
            )
        )
        patched["patches"].append(
            _set_or_add_path(
                patched["main_config"],
                ("reward_model", "curvyzero_metrics_jsonl_path"),
                str(rnd_metrics_jsonl_path),
            )
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
        if initial_policy_checkpoint is not None:
            initial_policy_checkpoint["applied"] = False
            initial_policy_checkpoint["skip_reason"] = (
                "auto_resume_found_existing_run_checkpoint; resume wins over initial seed"
            )
            command["initial_policy_checkpoint"] = _to_plain(initial_policy_checkpoint)
    elif initial_policy_checkpoint is not None:
        initial_policy_checkpoint["applied"] = True
        patched["patches"].append(
            _set_load_ckpt_before_run(
                patched["main_config"],
                str(initial_policy_checkpoint["load_path"]),
                reason=(
                    "seed fresh training run from immutable tournament winner checkpoint "
                    "while preserving fresh optimizer state"
                ),
            )
        )
        patched["surface"] = _extract_surface(
            patched["main_config"],
            patched["create_config"],
            max_env_step=max_env_step,
            max_train_iter=max_train_iter,
        )
        command["initial_policy_checkpoint"] = _to_plain(initial_policy_checkpoint)
    surface = patched["surface"]
    problems.extend(_validate_visual_survival_surface(surface=surface, command=command))

    compile_summary = _compile_config_summary(
        patched["main_config"],
        patched["create_config"],
        seed=seed,
        allow_profile_custom_env_manager_skip=(
            env_manager_type in CURVYZERO_BATCHED_PROFILE_ENV_MANAGER_TYPES
        ),
    )
    if not compile_summary.get("ok", False):
        problems.append(f"compile_config failed: {compile_summary.get('error')}")

    train_result: dict[str, Any] | None = None
    called_train_muzero = False
    stdout_text = ""
    stderr_text = ""
    install_phase_profile = _should_install_lightzero_phase_profile(
        mode=mode,
        stop_after_learner_train_calls=int(stop_after_learner_train_calls),
    )
    profiler = _LightZeroPhaseProfiler(
        enabled=install_phase_profile,
        cuda_sync_enabled=profile_cuda_sync_enabled,
    )
    target_audit = _LightZeroTargetAudit(mode=mode, env_variant=command["env_variant"])
    initial_policy_load_audit = (
        _InitialPolicyCheckpointLoadAudit(checkpoint=initial_policy_checkpoint)
        if initial_policy_checkpoint is not None and initial_policy_checkpoint.get("applied")
        else None
    )
    assignment_refresh_events: list[dict[str, Any]] = []

    def record_assignment_refresh_event(event: Mapping[str, Any]) -> None:
        row = _to_plain(
            {
                "schema_id": "curvyzero_opponent_assignment_refresh_event/v0",
                "run_id": run_id,
                "attempt_id": attempt_id,
                "created_at": runs.utc_timestamp(),
                **dict(event),
            }
        )
        assignment_refresh_events.append(row)
        assignment_refresh_events_path.parent.mkdir(parents=True, exist_ok=True)
        with assignment_refresh_events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        if row.get("decision") == "applied":
            _append_trainer_assignment_applied_lineage(
                trainer_lineage_path,
                run_id=run_id,
                attempt_id=attempt_id,
                event=row,
            )

    if mode in {"train", "profile"} and not problems:
        train_original_cwd = Path.cwd()
        train_restore_cwd = (
            _preferred_cwd_outside_runs_mount()
            if _path_is_inside_or_equal(train_original_cwd, RUNS_MOUNT)
            else train_original_cwd
        )
        os.chdir(RUNS_MOUNT)
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        train_started = time.perf_counter()
        restore_profile = None
        restore_target_audit = None
        restore_assignment_refresh = None
        restore_live_publisher = None
        restore_progress_writer = None
        restore_resume_state = None
        restore_initial_policy_load_audit = None
        restore_learner_metrics = None
        restore_rnd_reward_model = None
        restore_collect_search_backend = None
        restore_batched_profile_env_manager = None
        learner_metrics_recorder = None
        gpu_sampler = None
        own_checkpoint_opponent_publisher = None
        if command["opponent_assignment_refresh"].get("own_checkpoint_opponent_refresh_enabled"):
            published_own_checkpoint_iterations: set[int] = set()

            def publish_own_checkpoint_opponent(**kwargs: Any) -> dict[str, Any]:
                checkpoint = kwargs["checkpoint"]
                iteration = _safe_int_or_none(checkpoint.get("iteration"))
                if iteration is not None and iteration in published_own_checkpoint_iterations:
                    return {
                        "saved": False,
                        "reason": "iteration_already_published_in_this_process",
                        "iteration": int(iteration),
                    }
                result = _write_own_checkpoint_opponent_refresh(
                    run_id=run_id,
                    attempt_id=attempt_id,
                    checkpoint=checkpoint,
                    timestamp=str(kwargs["timestamp"]),
                    seed=seed,
                    num_simulations=num_simulations,
                    batch_size=batch_size,
                    opponent_use_cuda=opponent_use_cuda,
                    assignment_refresh_ref=opponent_assignment_refresh_ref,
                )
                if result.get("saved") and iteration is not None:
                    published_own_checkpoint_iterations.add(int(iteration))
                return result

            own_checkpoint_opponent_publisher = publish_own_checkpoint_opponent
        try:
            restore_collect_search_backend = _install_lightzero_collect_search_backend_hook(
                train_muzero=train_muzero,
                backend=collect_search_backend,
                ctree_backend=collect_search_ctree_backend,
                profiler=profiler,
                allow_fallback=(mode == "profile"),
            )
            restore_rnd_reward_model = _install_curvyzero_rnd_reward_model(
                train_muzero=train_muzero,
                exploration_bonus_spec=exploration_bonus_spec,
            )
            if initial_policy_load_audit is not None:
                restore_initial_policy_load_audit = _install_initial_policy_checkpoint_load_audit(
                    initial_policy_load_audit
                )
            learner_metrics_install = _install_lightzero_learner_metrics_recorder(
                train_muzero=train_muzero,
                run_id=run_id,
                attempt_id=attempt_id,
                attempt_train_root=attempt_train_root,
                started_monotonic=train_started,
            )
            if learner_metrics_install is not None:
                restore_learner_metrics, learner_metrics_recorder = learner_metrics_install
            restore_resume_state = _install_lightzero_full_resume_state_hooks(
                train_muzero=train_muzero,
                run_id=run_id,
                attempt_id=attempt_id,
                exp_name=exp_name,
                auto_resume=auto_resume,
                attempt_train_root=attempt_train_root,
                started_monotonic=train_started,
                checkpoint_metadata=command,
                own_checkpoint_opponent_publisher=own_checkpoint_opponent_publisher,
                learner_metrics_recorder=learner_metrics_recorder,
            )
            restore_progress_writer = _install_checkpoint_progress_writer(
                train_muzero=train_muzero,
                run_id=run_id,
                attempt_id=attempt_id,
                exp_name=exp_name,
                attempt_train_root=attempt_train_root,
                started_monotonic=train_started,
                commit_on_checkpoint=command["commit_on_checkpoint"],
                checkpoint_metadata=command,
                own_checkpoint_opponent_publisher=own_checkpoint_opponent_publisher,
                learner_metrics_recorder=learner_metrics_recorder,
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
            if command["opponent_assignment_refresh"]["enabled"]:
                refresh_reads_local_own_checkpoint = bool(
                    command["opponent_assignment_refresh"].get(
                        "own_checkpoint_opponent_refresh_enabled"
                    )
                )

                def load_pending_opponent_assignment_for_refresh() -> dict[str, Any] | None:
                    pending_assignment = _resolve_opponent_assignment_for_env(
                        opponent_assignment_ref=opponent_assignment_refresh_ref,
                        reload_volume_before_read=not refresh_reads_local_own_checkpoint,
                        reload_checkpoint_volume_before_read=(
                            not refresh_reads_local_own_checkpoint
                        ),
                        checkpoint_volume_reload_errors_fatal=False,
                    )
                    if pending_assignment is not None:
                        _append_trainer_assignment_loaded_lineage(
                            trainer_lineage_path,
                            status="refresh_candidate",
                            run_id=run_id,
                            attempt_id=attempt_id,
                            opponent_assignment_ref=opponent_assignment_refresh_ref,
                            opponent_assignment=pending_assignment,
                        )
                    return pending_assignment

                restore_assignment_refresh = _install_lightzero_opponent_assignment_refresh_hook(
                    train_muzero=train_muzero,
                    interval_train_iter=int(
                        command["opponent_assignment_refresh"]["interval_train_iter"]
                    ),
                    load_pending_assignment=load_pending_opponent_assignment_for_refresh,
                    initial_assignment=opponent_assignment,
                    event_sink=record_assignment_refresh_event,
                )
                if restore_assignment_refresh is None:
                    raise RuntimeError("opponent assignment refresh hook was not installed")
            restore_target_audit = _install_lightzero_target_audit(
                train_muzero=train_muzero,
                audit=target_audit,
            )
            if install_phase_profile:
                restore_profile = _install_lightzero_phase_profile(
                    train_muzero=train_muzero,
                    profiler=profiler,
                    stop_after_learner_train_calls=int(stop_after_learner_train_calls),
                    skip_evaluator_eval=(
                        bool(skip_lightzero_eval_in_profile) if mode == "profile" else False
                    ),
                )
            if env_manager_type in CURVYZERO_BATCHED_PROFILE_ENV_MANAGER_TYPES:
                restore_batched_profile_env_manager = _install_batched_profile_env_manager_hook(
                    train_muzero=train_muzero,
                    profiler=profiler,
                    env_manager_type=env_manager_type,
                    seed=seed,
                    source_max_steps=source_max_steps,
                    decision_ms=decision_ms,
                    natural_bonus_spawn=natural_bonus_spawn,
                    disable_death_for_profile=disable_death_for_profile,
                )
            if mode == "profile":
                gpu_sampler = _start_gpu_sampler(
                    profiler=profiler,
                    interval_sec=profiler.gpu_sample_interval_sec,
                )
            with (
                contextlib.redirect_stdout(stdout_buffer),
                contextlib.redirect_stderr(stderr_buffer),
            ):
                called_train_muzero = True
                with profiler.timer("train_muzero_wall_sec"):
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
                "stopped_by_learner_train_call_cap": True,
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
            if restore_collect_search_backend is not None:
                try:
                    restore_collect_search_backend()
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    profiler.add_note(
                        "collect search backend restore failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
            if restore_learner_metrics is not None:
                try:
                    restore_learner_metrics()
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    profiler.add_note(
                        f"learner metrics restore failed: {type(exc).__name__}: {exc}"
                    )
            if restore_target_audit is not None:
                try:
                    restore_target_audit()
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    target_audit.add_error(
                        f"target audit restore failed: {type(exc).__name__}: {exc}"
                    )
            if restore_assignment_refresh is not None:
                try:
                    restore_assignment_refresh()
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    record_assignment_refresh_event(
                        {
                            "decision": "restore_failed",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
            if restore_live_publisher is not None:
                try:
                    restore_live_publisher()
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    profiler.add_note(
                        f"live checkpoint publisher restore failed: {type(exc).__name__}: {exc}"
                    )
            if restore_progress_writer is not None:
                try:
                    restore_progress_writer()
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    profiler.add_note(
                        f"checkpoint progress writer restore failed: {type(exc).__name__}: {exc}"
                    )
            if restore_resume_state is not None:
                try:
                    restore_resume_state()
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    profiler.add_note(
                        f"full resume state restore hook cleanup failed: {type(exc).__name__}: {exc}"
                    )
            if restore_initial_policy_load_audit is not None:
                try:
                    restore_initial_policy_load_audit()
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    if initial_policy_load_audit is not None:
                        initial_policy_load_audit.errors.append(
                            f"initial policy load audit restore failed: {type(exc).__name__}: {exc}"
                        )
            if restore_rnd_reward_model is not None:
                try:
                    restore_rnd_reward_model()
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    profiler.add_note(
                        f"RND reward-model restore failed: {type(exc).__name__}: {exc}"
                    )
            if restore_batched_profile_env_manager is not None:
                try:
                    restore_batched_profile_env_manager()
                except Exception as exc:  # pragma: no cover - remote diagnosis only.
                    profiler.add_note(
                        "batched profile env manager restore failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
            if gpu_sampler is not None:
                gpu_stop_event, gpu_thread = gpu_sampler
                gpu_stop_event.set()
                gpu_thread.join(timeout=5)
                profiler.sample_gpu()
            try:
                os.chdir(train_restore_cwd)
            except Exception as exc:  # pragma: no cover - remote diagnosis only.
                profiler.add_note(f"cwd restore failed: {type(exc).__name__}: {exc}")

    if initial_policy_checkpoint is not None and initial_policy_load_audit is not None:
        initial_policy_checkpoint["load_result"] = initial_policy_load_audit.summary()
        command["initial_policy_checkpoint"] = _to_plain(initial_policy_checkpoint)

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
    collect_search_backend_proof = _collect_search_backend_proof(
        command=command,
        phase_profile=phase_profile,
    )
    problems.extend(
        _validate_train_collect_search_backend_proof(
            command=command,
            proof=collect_search_backend_proof,
        )
    )
    target_audit_summary = target_audit.summary()
    rnd_reward_model_metrics = _summarize_rnd_reward_model_metrics(
        enabled=exploration_bonus_spec.enabled,
        latest_path=rnd_metrics_latest_path,
        jsonl_path=rnd_metrics_jsonl_path,
    )
    if bool(require_rnd_metrics) and exploration_bonus_spec.enabled:
        problems.extend(
            _validate_required_rnd_reward_model_metrics(
                rnd_reward_model_metrics,
                weight=exploration_bonus_spec.weight,
            )
        )
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
        "trainer_entrypoint": trainer_entrypoint_ref,
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
        "initial_policy_checkpoint": (
            _to_plain(initial_policy_checkpoint) if initial_policy_checkpoint is not None else None
        ),
        "patches": patched["patches"],
        "compile_config": compile_summary,
        "train_result": train_result,
        "phase_profile": phase_profile,
        "collect_search_backend_proof": collect_search_backend_proof,
        "target_audit": target_audit_summary,
        "rnd_reward_model_metrics": rnd_reward_model_metrics,
        "opponent_assignment_refresh": {
            "enabled": command["opponent_assignment_refresh"]["enabled"],
            "mode": command["opponent_assignment_refresh"]["mode"],
            "interval_train_iter": command["opponent_assignment_refresh"]["interval_train_iter"],
            "pending_assignment_ref": command["opponent_assignment_refresh"][
                "pending_assignment_ref"
            ],
            "events_ref": command["opponent_assignment_refresh"]["events_ref"],
            "event_count": len(assignment_refresh_events),
            "events": assignment_refresh_events[-20:],
        },
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
                "requested_frame_size": command["background_gif_frame_size"],
                "frame_size": CHECKPOINT_SELFPLAY_GIF_FRAME_SIZE,
                "frame_size_policy": (
                    "checkpoint_selfplay_gif_always_uses_full_source_state_rgb_canvas"
                ),
                "collect_temperature": command["background_gif_collect_temperature"],
                "collect_epsilon": command["background_gif_collect_epsilon"],
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
        "fixed_opponent_is_two_seat_self_play": command["fixed_opponent_is_two_seat_self_play"],
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
    rnd_metrics_scan_path = attempt_train_root / "rnd_reward_model_metrics_scan.json"
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
    runs.write_json(rnd_metrics_scan_path, rnd_reward_model_metrics)
    _write_text(
        stdout_path, "\n".join(_compact_log_tail(stdout_text)) + ("\n" if stdout_text else "")
    )
    _write_text(
        stderr_path,
        "\n".join(_compact_log_tail(stderr_text, limit=30)) + ("\n" if stderr_text else ""),
    )

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
    final_status_heartbeat = _write_train_status_heartbeat(
        run_id=run_id,
        attempt_id=attempt_id,
        status=status,
        stage=status,
        started_at=started_at,
        modal_task_id=modal_task_id,
        config=command,
        exp_name_ref=exp_name_ref.as_posix(),
    )
    final_volume_commit = {"attempted": False}
    should_commit_final_artifacts = mode == "train" or (mode == "profile" and profile_volume_commit)
    if should_commit_final_artifacts and hasattr(runs_volume, "commit"):
        final_volume_commit = {"attempted": True, "ok": False}
        commit_started = time.perf_counter()
        try:
            _commit_runs_volume_with_backoff(label=f"{mode}_final_commit")
            final_volume_commit["ok"] = True
        except Exception as exc:  # pragma: no cover - remote artifact durability only.
            final_volume_commit["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            final_volume_commit["elapsed_sec"] = round(
                time.perf_counter() - commit_started,
                6,
            )
    result = {
        "ok": summary["ok"],
        "status": status,
        "mode": mode,
        "compute": compute,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "called_train_muzero": called_train_muzero,
        "trainer_entrypoint": trainer_entrypoint_ref,
        "summary_ref": summary_ref,
        "attempt_manifest": attempt_manifest,
        "latest_attempt": latest_attempt,
        "final_status_heartbeat": final_status_heartbeat,
        "problems": problems,
        "command": command,
        "runtime_compute": summary.get("runtime_compute"),
        "action_observability": action_summary,
        "checkpoint_mirror": checkpoint_mirror,
        "auto_resume": auto_resume,
        "initial_policy_checkpoint": summary.get("initial_policy_checkpoint"),
        "final_volume_commit": final_volume_commit,
        "phase_profile": phase_profile,
        "collect_search_backend_proof": collect_search_backend_proof,
        "target_audit": target_audit_summary,
        "rnd_reward_model_metrics": rnd_reward_model_metrics,
        "artifact_refs": {
            "summary": runs.file_summary(summary_path, mount=RUNS_MOUNT),
            "config": runs.file_summary(config_path, mount=RUNS_MOUNT),
            "command": runs.file_summary(command_path, mount=RUNS_MOUNT),
            "actions": runs.file_summary(actions_path, mount=RUNS_MOUNT),
            "target_audit": runs.file_summary(target_audit_path, mount=RUNS_MOUNT),
            "rnd_reward_model_metrics_scan": runs.file_summary(
                rnd_metrics_scan_path,
                mount=RUNS_MOUNT,
            ),
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
    reload_checkpoint_volume_before_read: bool = False,
) -> dict[str, Any] | None:
    if opponent_policy_kind == OPPONENT_POLICY_KIND_NONE_CENTRALIZED_JOINT_ACTION:
        if opponent_checkpoint_ref:
            raise ValueError(
                "opponent_checkpoint_ref is not valid for centralized joint-action control"
            )
        return None
    if opponent_policy_kind in (
        OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
    ):
        if opponent_checkpoint_ref:
            raise ValueError(
                "opponent_checkpoint_ref is only valid with frozen_lightzero_checkpoint"
            )
        return None
    if opponent_policy_kind != OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT:
        raise ValueError(f"unknown opponent_policy_kind {opponent_policy_kind!r}")
    if not opponent_checkpoint_ref:
        raise ValueError("opponent_checkpoint_ref is required with frozen_lightzero_checkpoint")
    _reject_mutable_frozen_opponent_checkpoint_ref(str(opponent_checkpoint_ref))
    if reload_checkpoint_volume_before_read:
        _safe_reload_volume_for_ref(
            str(opponent_checkpoint_ref),
            reason="opponent checkpoint resolution",
        )
    path, resolution = _resolve_trainer_ref_or_path(
        opponent_checkpoint_ref,
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
        "file": _file_summary_for_resolution(path, resolution),
    }


def _resolve_opponent_mixture_for_env(
    *,
    opponent_mixture_spec: Any,
    reload_checkpoint_volume_before_read: bool = False,
    checkpoint_volume_reload_errors_fatal: bool = True,
) -> dict[str, Any] | None:
    mixture, _warnings = _resolve_opponent_mixture_for_env_with_warnings(
        opponent_mixture_spec=opponent_mixture_spec,
        reload_checkpoint_volume_before_read=reload_checkpoint_volume_before_read,
        checkpoint_volume_reload_errors_fatal=checkpoint_volume_reload_errors_fatal,
    )
    return mixture


def _resolve_opponent_mixture_for_env_with_warnings(
    *,
    opponent_mixture_spec: Any,
    reload_checkpoint_volume_before_read: bool = False,
    checkpoint_volume_reload_errors_fatal: bool = True,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    mixture = parse_opponent_mixture_spec(opponent_mixture_spec)
    if mixture is None:
        return None, []
    entries: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    for entry in mixture["entries"]:
        resolved_entry = dict(entry)
        if entry["opponent_policy_kind"] == OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT:
            checkpoint_ref = entry.get("opponent_checkpoint_ref")
            if not checkpoint_ref:
                raise ValueError(
                    "frozen opponent mixture entries in the Modal trainer must use "
                    "opponent_checkpoint_ref so the ref is captured in run metadata"
                )
            _reject_mutable_frozen_opponent_checkpoint_ref(str(checkpoint_ref))
            if not re.fullmatch(r"iteration_\d+\.pth\.tar", Path(str(checkpoint_ref)).name):
                raise ValueError(
                    "frozen opponent mixture entries must use exact immutable "
                    "iteration_N.pth.tar checkpoint refs"
                )
            if reload_checkpoint_volume_before_read:
                try:
                    _safe_reload_volume_for_ref(
                        str(checkpoint_ref),
                        reason="opponent mixture checkpoint resolution",
                    )
                except RuntimeError as exc:
                    if checkpoint_volume_reload_errors_fatal:
                        raise
                    warnings.append(
                        {
                            "entry_name": str(entry.get("name", "")),
                            "opponent_checkpoint_ref": str(checkpoint_ref),
                            "warning": str(exc),
                        }
                    )
            path, resolution = _resolve_trainer_ref_or_path(
                str(checkpoint_ref),
            )
            if not path.is_file():
                raise FileNotFoundError(f"opponent mixture checkpoint file not found: {path}")
            report_ref = entry.get("opponent_checkpoint_report_ref")
            if report_ref is None:
                report_ref = (
                    str(resolution.get("source_ref"))
                    if resolution.get("source_ref")
                    else str(checkpoint_ref)
                )
            resolved_entry.update(
                {
                    "opponent_checkpoint_path": str(path),
                    "opponent_checkpoint_ref": str(report_ref),
                    "opponent_checkpoint_resolution": resolution,
                    "opponent_checkpoint_file": _file_summary_for_resolution(
                        path,
                        resolution,
                    ),
                }
            )
        entries.append(resolved_entry)
    return (
        {
            **mixture,
            "schema_id": OPPONENT_MIXTURE_SCHEMA_ID,
            "entries": entries,
        },
        warnings,
    )


def _path_is_inside_or_equal(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False


def _preferred_cwd_outside_runs_mount() -> Path:
    for candidate in (Path(tempfile.gettempdir()), REMOTE_ROOT, Path("/")):
        if candidate.exists() and not _path_is_inside_or_equal(candidate, RUNS_MOUNT):
            return candidate
    return Path("/")


def _safe_reload_volume(
    volume: Any,
    *,
    mount: Path,
    lock: threading.Lock,
    reason: str,
) -> None:
    """Reload a Volume without keeping cwd inside that mounted Volume."""

    if not hasattr(volume, "reload"):
        return
    with lock:
        try:
            original_cwd = Path.cwd()
        except OSError:
            original_cwd = None
        moved = False
        if original_cwd is None or _path_is_inside_or_equal(original_cwd, mount):
            os.chdir(_preferred_cwd_outside_runs_mount())
            moved = True
        try:
            volume.reload()
        except Exception as exc:
            raise RuntimeError(f"volume reload failed during {reason}: {exc}") from exc
        finally:
            if moved and original_cwd is not None:
                os.chdir(original_cwd)


def _safe_reload_runs_volume(*, reason: str) -> None:
    _safe_reload_volume(
        runs_volume,
        mount=RUNS_MOUNT,
        lock=_RUNS_VOLUME_RELOAD_LOCK,
        reason=reason,
    )


def _safe_reload_control_volume(*, reason: str) -> None:
    _safe_reload_volume(
        control_volume,
        mount=CONTROL_MOUNT,
        lock=_CONTROL_VOLUME_RELOAD_LOCK,
        reason=reason,
    )


def _volume_name_for_ref(path_text: str, *, default_mount_name: str = "runs") -> str | None:
    if path_text.startswith(_CONTROL_REF_PREFIX):
        return "control"
    if path_text.startswith(_RUNS_REF_PREFIX):
        return "runs"
    path = Path(path_text)
    if not path.is_absolute():
        return default_mount_name
    path_posix = path.as_posix()
    control_posix = CONTROL_MOUNT.as_posix()
    runs_posix = RUNS_MOUNT.as_posix()
    if path_posix == control_posix or path_posix.startswith(f"{control_posix}/"):
        return "control"
    if path_posix == runs_posix or path_posix.startswith(f"{runs_posix}/"):
        return "runs"
    return None


def _safe_reload_volume_for_ref(
    path_text: str,
    *,
    reason: str,
    default_mount_name: str = "runs",
) -> None:
    volume_name = _volume_name_for_ref(path_text, default_mount_name=default_mount_name)
    if volume_name == "control":
        _safe_reload_control_volume(reason=reason)
    elif volume_name == "runs":
        _safe_reload_runs_volume(reason=reason)


def _prefixed_volume_ref(path_text: str) -> tuple[Path, str, str] | None:
    for prefix, mount_name, mount in (
        (_CONTROL_REF_PREFIX, "control", CONTROL_MOUNT),
        (_RUNS_REF_PREFIX, "runs", RUNS_MOUNT),
    ):
        if path_text.startswith(prefix):
            raw_ref = path_text[len(prefix) :]
            ref = runs.require_relative_ref(raw_ref)
            return mount / Path(*ref.parts), mount_name, ref.as_posix()
    return None


def _resolve_trainer_ref_or_path(
    path_text: str,
    *,
    default_mount: Path | None = None,
    default_mount_name: str = "runs",
) -> tuple[Path, dict[str, Any]]:
    mount = RUNS_MOUNT if default_mount is None else default_mount
    prefixed = _prefixed_volume_ref(str(path_text))
    if prefixed is not None:
        path, mount_name, source_ref = prefixed
        return path, {
            "source_kind": f"{mount_name}_volume_ref",
            "source_ref": source_ref,
            "mount": mount_name,
        }
    path, resolution = runs.resolve_mounted_ref_or_path(
        str(path_text),
        mount=mount,
        remote_root=REMOTE_ROOT,
    )
    return path, {"mount": default_mount_name, **resolution}


def _file_summary_for_resolution(path: Path, resolution: Mapping[str, Any]) -> dict[str, Any]:
    mount = CONTROL_MOUNT if resolution.get("mount") == "control" else RUNS_MOUNT
    return runs.file_summary_any_mount(path, mount=mount)


def _resolve_opponent_assignment_for_env(
    *,
    opponent_assignment_ref: str | None,
    reload_volume_before_read: bool = False,
    reload_checkpoint_volume_before_read: bool = False,
    checkpoint_volume_reload_errors_fatal: bool = True,
    _pointer_depth: int = 0,
) -> dict[str, Any] | None:
    """Read an assignment, or a small mutable pointer to one, for the env."""

    if not opponent_assignment_ref:
        return None
    if reload_volume_before_read:
        if str(opponent_assignment_ref).startswith(_CONTROL_REF_PREFIX):
            _safe_reload_control_volume(reason="opponent assignment refresh")
        else:
            _safe_reload_runs_volume(reason="opponent assignment refresh")
    path, resolution = _resolve_trainer_ref_or_path(
        str(opponent_assignment_ref),
    )
    if not path.is_file():
        raise FileNotFoundError(f"opponent assignment file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_id") == OPPONENT_ASSIGNMENT_REFRESH_POINTER_SCHEMA_ID:
        if int(_pointer_depth) >= 3:
            raise ValueError("opponent assignment refresh pointer chain is too deep")
        pointed_ref = payload.get("assignment_ref")
        if not isinstance(pointed_ref, str) or not pointed_ref:
            raise ValueError("opponent assignment refresh pointer requires assignment_ref")
        resolved = _resolve_opponent_assignment_for_env(
            opponent_assignment_ref=pointed_ref,
            reload_volume_before_read=False,
            reload_checkpoint_volume_before_read=reload_checkpoint_volume_before_read,
            checkpoint_volume_reload_errors_fatal=checkpoint_volume_reload_errors_fatal,
            _pointer_depth=int(_pointer_depth) + 1,
        )
        if resolved is None:
            return None
        expected_sha = payload.get("assignment_sha256")
        if expected_sha and str(expected_sha) != str(resolved.get("assignment_sha256")):
            raise ValueError(
                "opponent assignment refresh pointer sha mismatch: "
                f"expected {expected_sha}, got {resolved.get('assignment_sha256')}"
            )
        pointer_ref = (
            str(resolution.get("source_ref"))
            if resolution.get("source_ref")
            else str(opponent_assignment_ref)
        )
        resolved["assignment_pointer"] = {
            "schema_id": OPPONENT_ASSIGNMENT_REFRESH_POINTER_SCHEMA_ID,
            "pointer_ref": pointer_ref,
            "pointer_path": str(path),
            "pointed_assignment_ref": pointed_ref,
            "expected_assignment_sha256": expected_sha,
            "pointer_resolution": resolution,
            "pointer_file": _file_summary_for_resolution(path, resolution),
        }
        return resolved
    parsed = parse_opponent_assignment_snapshot(payload)
    if parsed is None:
        return None
    assignment_hash = canonical_assignment_json_sha256(payload)
    opponent_mixture, checkpoint_reload_warnings = _resolve_opponent_mixture_for_env_with_warnings(
        opponent_mixture_spec=parsed["opponent_mixture"],
        reload_checkpoint_volume_before_read=reload_checkpoint_volume_before_read,
        checkpoint_volume_reload_errors_fatal=checkpoint_volume_reload_errors_fatal,
    )
    result = {
        "assignment_id": parsed["assignment_id"],
        "source_epoch": parsed.get("source_epoch"),
        "source_ref": parsed.get("source_ref"),
        "assignment_ref": (
            str(resolution.get("source_ref"))
            if resolution.get("source_ref")
            else str(opponent_assignment_ref)
        ),
        "assignment_path": str(path),
        "assignment_sha256": assignment_hash,
        "assignment_resolution": resolution,
        "assignment_file": _file_summary_for_resolution(path, resolution),
        "opponent_mixture": opponent_mixture,
    }
    if checkpoint_reload_warnings:
        result["opponent_checkpoint_volume_reload_warnings"] = checkpoint_reload_warnings
    return result


def _opponent_assignment_context_for_env(
    opponent_assignment: Mapping[str, Any] | None,
    *,
    refresh_index: int | None = None,
) -> dict[str, Any] | None:
    if opponent_assignment is None:
        return None
    context = {
        "assignment_id": opponent_assignment.get("assignment_id"),
        "assignment_ref": opponent_assignment.get("assignment_ref"),
        "assignment_sha256": opponent_assignment.get("assignment_sha256"),
        "source_epoch": opponent_assignment.get("source_epoch"),
        "source_ref": opponent_assignment.get("source_ref"),
    }
    if refresh_index is not None:
        context["refresh_index"] = int(refresh_index)
    return {key: value for key, value in context.items() if value is not None}


def _append_trainer_assignment_loaded_lineage(
    lineage_path: Path,
    *,
    status: str,
    run_id: str,
    attempt_id: str,
    opponent_assignment_ref: str | None,
    opponent_assignment: Mapping[str, Any],
) -> dict[str, Any]:
    return append_lineage_event(
        lineage_path,
        stage="trainer_assignment_loaded",
        status=status,
        run_id=run_id,
        attempt_id=attempt_id,
        opponent_assignment_ref=opponent_assignment_ref,
        assignment_id=opponent_assignment.get("assignment_id"),
        assignment_ref=opponent_assignment.get("assignment_ref"),
        assignment_sha256=opponent_assignment.get("assignment_sha256"),
        assignment_pointer=opponent_assignment.get("assignment_pointer"),
        source_epoch=opponent_assignment.get("source_epoch"),
    )


def _append_trainer_assignment_applied_lineage(
    lineage_path: Path,
    *,
    run_id: str,
    attempt_id: str,
    event: Mapping[str, Any],
) -> dict[str, Any]:
    return append_lineage_event(
        lineage_path,
        stage="trainer_assignment_applied",
        run_id=run_id,
        attempt_id=attempt_id,
        train_iter=event.get("train_iter"),
        bucket=event.get("bucket"),
        refresh_index=event.get("refresh_index"),
        assignment_id=event.get("assignment_id"),
        assignment_ref=event.get("assignment_ref"),
        assignment_sha256=event.get("assignment_sha256"),
        env_ready_report=event.get("env_ready_report"),
    )


def _lightzero_collect_train_iter_from_call(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> int:
    """Extract MuZeroCollector.collect's train_iter argument.

    LightZero 0.2.0 uses collect(n_episode=None, train_iter=0, ...), so a
    single positional argument is n_episode, not train_iter.
    """

    if "train_iter" in kwargs:
        return int(kwargs["train_iter"])
    if len(args) >= 2:
        return int(args[1])
    return 0


def _opponent_assignment_refresh_bucket(
    *,
    train_iter: int,
    interval_train_iter: int,
) -> int:
    interval = int(interval_train_iter)
    if interval < 1:
        raise ValueError("opponent assignment refresh interval must be at least 1")
    train_iter = int(train_iter)
    if train_iter <= 0:
        return 0
    return train_iter // interval


def _opponent_assignment_refresh_due(
    *,
    train_iter: int,
    interval_train_iter: int,
    last_checked_bucket: int,
) -> bool:
    bucket = _opponent_assignment_refresh_bucket(
        train_iter=train_iter,
        interval_train_iter=interval_train_iter,
    )
    return bucket > int(last_checked_bucket)


def _opponent_assignment_refresh_reset_param(
    *,
    env_num: int,
    opponent_assignment: Mapping[str, Any],
    refresh_index: int,
) -> dict[int, dict[str, Any]]:
    env_num = int(env_num)
    if env_num < 1:
        raise ValueError("opponent assignment refresh requires at least one env")
    opponent_mixture = opponent_assignment.get("opponent_mixture")
    if not isinstance(opponent_mixture, Mapping):
        raise ValueError("opponent assignment refresh requires a resolved opponent_mixture")
    context = _opponent_assignment_context_for_env(
        opponent_assignment,
        refresh_index=int(refresh_index),
    )
    if context is None:
        raise ValueError("opponent assignment refresh requires assignment context")
    for key in ("assignment_id", "assignment_ref", "assignment_sha256", "refresh_index"):
        if key not in context:
            raise ValueError(f"opponent assignment refresh context missing {key}")
    split_plan = deterministic_collector_env_mixture_plan(
        dict(opponent_mixture),
        env_num=env_num,
        seed_context={
            "assignment_id": context.get("assignment_id"),
            "assignment_sha256": context.get("assignment_sha256"),
            "refresh_index": int(refresh_index),
        },
    )
    count_by_entry_name = {
        str(row["entry_name"]): int(row["count"]) for row in split_plan["slot_counts"]
    }
    return {
        env_id: {
            "opponent_mixture": singleton_mixture_for_split_entry(
                dict(opponent_mixture),
                entry_index=int(split_plan["assignments"][env_id]["entry_index"]),
            ),
            "opponent_assignment_context": {
                **copy.deepcopy(context),
                "opponent_split_unit": split_plan["unit"],
                "opponent_split_mode": split_plan["mode"],
                "opponent_split_plan_sha256": split_plan["plan_sha256"],
                "opponent_split_env_index": int(env_id),
                "opponent_split_env_num": env_num,
                "opponent_split_entry_name": split_plan["assignments"][env_id]["entry_name"],
                "opponent_split_entry_count": count_by_entry_name[
                    str(split_plan["assignments"][env_id]["entry_name"])
                ],
            },
        }
        for env_id in range(env_num)
    }


def _opponent_assignment_refresh_ready_report(
    *,
    env_manager: Any,
    opponent_assignment: Mapping[str, Any],
    refresh_index: int,
) -> dict[str, Any]:
    env_num = int(getattr(env_manager, "env_num"))
    expected_context = _opponent_assignment_context_for_env(
        opponent_assignment,
        refresh_index=int(refresh_index),
    )
    if expected_context is None:
        return {"ok": False, "reason": "missing expected assignment context"}

    ready_obs = getattr(env_manager, "ready_obs", None)
    if isinstance(ready_obs, Mapping):
        ready_ids = set(int(key) for key in ready_obs.keys())
    elif isinstance(ready_obs, (list, tuple)):
        ready_ids = set(range(len(ready_obs)))
    else:
        return {
            "ok": False,
            "reason": f"ready_obs has unsupported type {type(ready_obs).__name__}",
        }
    expected_ids = set(range(env_num))
    if ready_ids != expected_ids:
        return {
            "ok": False,
            "reason": "not all envs are ready after assignment refresh",
            "ready_env_ids": sorted(ready_ids),
            "expected_env_ids": sorted(expected_ids),
        }

    infos = getattr(env_manager, "last_reset_info", None)
    if isinstance(infos, Mapping):
        info_by_env = {int(key): value for key, value in infos.items()}
    elif isinstance(infos, (list, tuple)):
        info_by_env = {env_id: value for env_id, value in enumerate(infos)}
    else:
        return {
            "ok": False,
            "reason": f"last_reset_info has unsupported type {type(infos).__name__}",
        }
    if set(info_by_env) != expected_ids:
        return {
            "ok": False,
            "reason": "last_reset_info missing env ids after assignment refresh",
            "info_env_ids": sorted(info_by_env),
            "expected_env_ids": sorted(expected_ids),
        }

    mismatches: list[dict[str, Any]] = []
    expected_pairs = {
        "opponent_assignment_id": expected_context.get("assignment_id"),
        "opponent_assignment_ref": expected_context.get("assignment_ref"),
        "opponent_assignment_sha256": expected_context.get("assignment_sha256"),
        "opponent_assignment_refresh_index": expected_context.get("refresh_index"),
    }
    reset_param = _opponent_assignment_refresh_reset_param(
        env_num=env_num,
        opponent_assignment=opponent_assignment,
        refresh_index=refresh_index,
    )
    expected_split_fields_by_env = {
        env_id: {
            key: value
            for key, value in reset_param[env_id]["opponent_assignment_context"].items()
            if key.startswith("opponent_split_")
        }
        for env_id in expected_ids
    }
    for env_id, info in sorted(info_by_env.items()):
        if not isinstance(info, Mapping):
            mismatches.append({"env_id": env_id, "reason": f"info type {type(info).__name__}"})
            continue
        expected_fields = {
            **expected_pairs,
            **expected_split_fields_by_env[env_id],
        }
        for key, expected in expected_fields.items():
            if info.get(key) != expected:
                mismatches.append(
                    {
                        "env_id": env_id,
                        "field": key,
                        "actual": info.get(key),
                        "expected": expected,
                    }
                )
    if mismatches:
        return {
            "ok": False,
            "reason": "env assignment info mismatch after refresh",
            "mismatches": mismatches,
        }
    slot_counts: dict[str, int] = {}
    split_plan_sha256_values: set[str] = set()
    for info in info_by_env.values():
        if not isinstance(info, Mapping):
            continue
        entry_name = info.get("opponent_split_entry_name", info.get("opponent_mixture_entry_name"))
        if entry_name is not None:
            slot_counts[str(entry_name)] = slot_counts.get(str(entry_name), 0) + 1
        split_plan_sha256 = info.get("opponent_split_plan_sha256")
        if split_plan_sha256 is not None:
            split_plan_sha256_values.add(str(split_plan_sha256))
    return {
        "ok": True,
        "reason": "all envs ready with refreshed opponent assignment",
        "env_num": env_num,
        "assignment_id": expected_context.get("assignment_id"),
        "assignment_ref": expected_context.get("assignment_ref"),
        "assignment_sha256": expected_context.get("assignment_sha256"),
        "refresh_index": int(refresh_index),
        "opponent_split_slot_counts": dict(sorted(slot_counts.items())),
        "opponent_split_plan_sha256": (
            next(iter(split_plan_sha256_values)) if len(split_plan_sha256_values) == 1 else None
        ),
    }


def _apply_opponent_assignment_refresh_to_collector_env(
    *,
    collector: Any,
    opponent_assignment: Mapping[str, Any],
    refresh_index: int,
) -> dict[str, Any]:
    env_manager = getattr(collector, "_env")
    env_num = int(getattr(env_manager, "env_num"))
    reset_param = _opponent_assignment_refresh_reset_param(
        env_num=env_num,
        opponent_assignment=opponent_assignment,
        refresh_index=int(refresh_index),
    )
    env_manager.reset(reset_param)
    report = _opponent_assignment_refresh_ready_report(
        env_manager=env_manager,
        opponent_assignment=opponent_assignment,
        refresh_index=int(refresh_index),
    )
    if not report.get("ok", False):
        raise RuntimeError(
            f"opponent assignment refresh reset was not proven: {report.get('reason')}"
        )

    policy = getattr(collector, "_policy", None)
    policy_reset = getattr(policy, "reset", None)
    if callable(policy_reset):
        policy_reset(list(reset_param.keys()))
    reset_stat = getattr(collector, "_reset_stat", None)
    if callable(reset_stat):
        for env_id in reset_param:
            reset_stat(env_id)
    return report


def _install_lightzero_opponent_assignment_refresh_hook(
    *,
    train_muzero: Any,
    interval_train_iter: int,
    load_pending_assignment: Any,
    initial_assignment: Mapping[str, Any] | None = None,
    event_sink: Any | None = None,
) -> Any:
    interval_train_iter = int(interval_train_iter)
    if interval_train_iter < 1:
        raise ValueError("opponent assignment refresh interval must be at least 1")
    if not callable(load_pending_assignment):
        raise TypeError("load_pending_assignment must be callable")

    globals_map = getattr(train_muzero, "__globals__", {})
    collector_cls = globals_map.get("Collector")
    if not inspect.isclass(collector_cls):
        try:
            worker_module = importlib.import_module("lzero.worker")
            collector_cls = getattr(worker_module, "MuZeroCollector", None)
        except Exception:
            collector_cls = None
    if not inspect.isclass(collector_cls):
        emit_missing = _to_plain(
            {
                "decision": "hook_not_installed",
                "reason": "no LightZero Collector class found",
            }
        )
        if callable(event_sink):
            event_sink(emit_missing)
        return None

    owner = next(
        (
            base
            for base in inspect.getmro(collector_cls)
            if "collect" in getattr(base, "__dict__", {})
        ),
        None,
    )
    if owner is None:
        return None

    original = owner.__dict__["collect"]
    initial_context = _opponent_assignment_context_for_env(initial_assignment)
    state = {
        "initial_collect_checked": False,
        "last_checked_bucket": 0,
        "refresh_index": int((initial_context or {}).get("refresh_index", 0) or 0),
        "last_applied_assignment_sha256": (
            str((initial_context or {}).get("assignment_sha256"))
            if (initial_context or {}).get("assignment_sha256") is not None
            else None
        ),
        "force_initial_deterministic_split": initial_assignment is not None,
    }

    def emit(event: Mapping[str, Any]) -> None:
        if callable(event_sink):
            event_sink(_to_plain(dict(event)))

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        train_iter = _lightzero_collect_train_iter_from_call(args, kwargs)
        bucket = _opponent_assignment_refresh_bucket(
            train_iter=train_iter,
            interval_train_iter=interval_train_iter,
        )
        check_initial_collect = int(train_iter) <= 0 and not bool(state["initial_collect_checked"])
        if check_initial_collect or bucket > int(state["last_checked_bucket"]):
            try:
                pending_assignment = load_pending_assignment()
            except Exception as exc:
                emit(
                    {
                        "decision": "kept_previous",
                        "reason": f"pending assignment load failed: {type(exc).__name__}: {exc}",
                        "train_iter": int(train_iter),
                        "bucket": int(bucket),
                    }
                )
            else:
                if pending_assignment is None:
                    state["initial_collect_checked"] = True
                    state["last_checked_bucket"] = bucket
                    emit(
                        {
                            "decision": "kept_previous",
                            "reason": "no pending assignment",
                            "train_iter": int(train_iter),
                            "bucket": int(bucket),
                        }
                    )
                else:
                    pending_sha = pending_assignment.get("assignment_sha256")
                    if not pending_sha:
                        emit(
                            {
                                "decision": "kept_previous",
                                "reason": "pending assignment missing assignment_sha256",
                                "train_iter": int(train_iter),
                                "bucket": int(bucket),
                            }
                        )
                    elif pending_sha == state["last_applied_assignment_sha256"] and not (
                        check_initial_collect and bool(state["force_initial_deterministic_split"])
                    ):
                        state["initial_collect_checked"] = True
                        state["last_checked_bucket"] = bucket
                        emit(
                            {
                                "decision": "unchanged",
                                "assignment_sha256": str(pending_sha),
                                "train_iter": int(train_iter),
                                "bucket": int(bucket),
                            }
                        )
                    else:
                        refresh_index = int(state["refresh_index"]) + 1
                        same_assignment_as_previous = (
                            pending_sha == state["last_applied_assignment_sha256"]
                        )
                        try:
                            report = _apply_opponent_assignment_refresh_to_collector_env(
                                collector=self,
                                opponent_assignment=pending_assignment,
                                refresh_index=refresh_index,
                            )
                        except Exception as exc:
                            emit(
                                {
                                    "decision": "failed_after_reset_attempt",
                                    "reason": f"{type(exc).__name__}: {exc}",
                                    "train_iter": int(train_iter),
                                    "bucket": int(bucket),
                                    "assignment_sha256": str(pending_sha),
                                    "refresh_index": int(refresh_index),
                                }
                            )
                            raise
                        state["refresh_index"] = refresh_index
                        state["initial_collect_checked"] = True
                        state["last_checked_bucket"] = bucket
                        state["last_applied_assignment_sha256"] = str(pending_sha)
                        state["force_initial_deterministic_split"] = False
                        emit(
                            {
                                "decision": "applied",
                                "reason": (
                                    "initial_collect_deterministic_split"
                                    if same_assignment_as_previous
                                    else "assignment_changed"
                                ),
                                "train_iter": int(train_iter),
                                "bucket": int(bucket),
                                "refresh_index": int(refresh_index),
                                "assignment_id": pending_assignment.get("assignment_id"),
                                "assignment_ref": pending_assignment.get("assignment_ref"),
                                "assignment_sha256": str(pending_sha),
                                "env_ready_report": report,
                            }
                        )
        return original(self, *args, **kwargs)

    setattr(owner, "collect", wrapped)

    def restore() -> None:
        setattr(owner, "collect", original)

    return restore


def _opponent_assignment_artifact_refs(
    *,
    run_id: str,
    attempt_id: str,
    assignment_id: str,
) -> dict[str, Any]:
    root = (
        runs.attempt_root_ref(TASK_ID, run_id, attempt_id)
        / "opponents"
        / "assignments"
        / runs.clean_id(assignment_id, label="assignment_id")
    )
    return {
        "root": root,
        "assignment": root / "assignment.json",
        "audit": root / "audit.json",
    }


def _write_opponent_assignment_artifacts(
    *,
    run_id: str,
    attempt_id: str,
    assignment: Mapping[str, Any],
    audit: Mapping[str, Any] | None = None,
    target_volume: str = "runs",
    mirror_checkpoints_to_control: bool = False,
) -> dict[str, Any]:
    if mirror_checkpoints_to_control:
        raise NotImplementedError(
            "mirror_checkpoints_to_control is not wired in this trainer writer yet"
        )
    target_volume = str(target_volume or "runs")
    if target_volume not in {"runs", "control"}:
        raise ValueError("target_volume must be 'runs' or 'control'")
    target_mount = CONTROL_MOUNT if target_volume == "control" else RUNS_MOUNT
    parsed = parse_opponent_assignment_snapshot(assignment)
    if parsed is None:
        raise ValueError("opponent assignment is required")
    assignment_id = str(parsed["assignment_id"])
    refs = _opponent_assignment_artifact_refs(
        run_id=run_id,
        attempt_id=attempt_id,
        assignment_id=assignment_id,
    )
    assignment_path = runs.volume_path(target_mount, refs["assignment"])
    assignment_sha256 = canonical_assignment_json_sha256(assignment)
    runs.write_json(assignment_path, _to_plain(dict(assignment)))
    audit_summary = None
    if audit is not None:
        validate_assignment_audit(audit, assignment=assignment)
        audit_path = runs.volume_path(target_mount, refs["audit"])
        runs.write_json(audit_path, _to_plain(dict(audit)))
        audit_summary = runs.file_summary_any_mount(audit_path, mount=target_mount)
    if target_volume == "control":
        _commit_control_volume_with_backoff(label="opponent_assignment_artifact_commit")
        ref_prefix = _CONTROL_REF_PREFIX
    else:
        _commit_runs_volume_with_backoff(label="opponent_assignment_artifact_commit")
        ref_prefix = ""
    assignment_ref = f"{ref_prefix}{refs['assignment'].as_posix()}"
    audit_ref = f"{ref_prefix}{refs['audit'].as_posix()}" if audit is not None else None
    return {
        "schema_id": "curvyzero_opponent_assignment_artifact_write/v0",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "assignment_id": assignment_id,
        "target_volume": target_volume,
        "assignment_ref": assignment_ref,
        "assignment_sha256": assignment_sha256,
        "assignment_file": runs.file_summary_any_mount(assignment_path, mount=target_mount),
        "audit_ref": audit_ref,
        "audit_file": audit_summary,
    }


@lru_cache(maxsize=1)
def _source_state_fixed_opponent_wrapper_env_spec_fields() -> dict[str, Any]:
    return lz_config.source_state_fixed_opponent_wrapper_env_spec_fields()


def _env_variant_spec(env_variant: str) -> dict[str, Any]:
    return lz_config.env_variant_spec(env_variant)


def _survival_reward_policy() -> dict[str, Any]:
    return lz_config.survival_reward_policy()


def _all_players_alive_diagnostic_reward_policy() -> dict[str, Any]:
    return lz_config.all_players_alive_diagnostic_reward_policy()


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
    policy_action_repeat_extra_probability: float = (
        DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
    ),
    natural_bonus_spawn: bool = TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN,
    profile_env_timing_enabled: bool = False,
    opponent_death_mode: str = DEFAULT_OPPONENT_DEATH_MODE,
    opponent_runtime_mode: str = DEFAULT_OPPONENT_RUNTIME_MODE,
    model_support_cap: int | None = DEFAULT_MODEL_SUPPORT_CAP,
    td_steps: int | None = DEFAULT_TD_STEPS,
    exploration_bonus: Mapping[str, Any] | xb.ExplorationBonusSpec | str | None = None,
) -> dict[str, Any]:
    return lz_config.build_visual_survival_configs(**locals())


def _extract_surface(
    main_config: Any,
    create_config: Any,
    *,
    max_env_step: int,
    max_train_iter: int,
) -> dict[str, Any]:
    return lz_config.extract_visual_survival_surface(
        main_config,
        create_config,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
    )


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
        "trainer_entrypoint": command["trainer_entrypoint"],
        "exploration_bonus": _to_plain(command["exploration_bonus"]),
        "policy_use_rnd_model": command["exploration_bonus"]["mode"] != xb.EXPLORATION_BONUS_MODE_NONE,
        "reward_model_type": (
            "rnd_muzero"
            if command["exploration_bonus"]["mode"] != xb.EXPLORATION_BONUS_MODE_NONE
            else None
        ),
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
        "model_support_scale": command["lightzero_target_config"].get("model_support_scale"),
        "model_reward_support_size": command["lightzero_target_config"].get(
            "model_reward_support_size"
        ),
        "model_value_support_size": command["lightzero_target_config"].get(
            "model_value_support_size"
        ),
        "frame_stack_num": 1,
        "source_max_steps": command["source_max_steps"],
        "decision_ms": command["decision_ms"],
        "decision_source_frames": command["decision_source_frames"],
        "source_physics_step_ms": command["source_physics_step_ms"],
        "source_max_steps_semantics": command["source_max_steps_semantics"],
        "dynamic_seed": True,
        "reset_seed_strategy": command.get("reset_seed_strategy"),
        "telemetry_stride": command["env_telemetry_stride"],
        "profile_env_timing_enabled": command["profile_env_timing_enabled"],
        "reward_variant": command["reward_variant"],
        "reward_schema_id": command["reward_schema_id"],
        "reward_policy": command["reward_policy"],
        "lightzero_target_config": _to_plain(command["lightzero_target_config"]),
        "source_state_trail_render_mode": command["source_state_trail_render_mode"],
        "source_state_bonus_render_mode": command["source_state_bonus_render_mode"],
        "policy_observation_backend": command["policy_observation_backend"],
        "policy_trail_render_mode": command["policy_trail_render_mode"],
        "policy_bonus_render_mode": command["policy_bonus_render_mode"],
        "policy_observation_contract_id": command["policy_observation_contract_id"],
        "observation_contract": _to_plain(command["observation_contract"]),
        "learner_seat_mode": command["learner_seat_mode"],
        "default_trail_render_mode": command["default_trail_render_mode"],
        "supported_trail_render_modes": command["supported_trail_render_modes"],
        "default_bonus_render_mode": command["default_bonus_render_mode"],
        "supported_bonus_render_modes": command["supported_bonus_render_modes"],
        "default_policy_observation_backend": command["default_policy_observation_backend"],
        "supported_policy_observation_backends": command["supported_policy_observation_backends"],
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
        "fixed_opponent_is_two_seat_self_play": command["fixed_opponent_is_two_seat_self_play"],
        "browser_pixel_fidelity": command["browser_pixel_fidelity"],
        "uses_ale": command["uses_ale"],
        "visual_surface": command["visual_surface"],
        "visual_truth_level": command["visual_truth_level"],
        "visual_source_state_backed": command["visual_source_state_backed"],
        "ego_action_straight_override_probability": command[
            "ego_action_straight_override_probability"
        ],
        "policy_action_repeat_min": command["policy_action_repeat_min"],
        "policy_action_repeat_max": command["policy_action_repeat_max"],
        "policy_action_repeat_extra_probability": command["policy_action_repeat_extra_probability"],
        "policy_action_repeat_semantics": command["policy_action_repeat_semantics"],
        "control_noise_profile_id": command["control_noise_profile_id"],
        "disable_death_for_profile": command["disable_death_for_profile"],
        "opponent_death_mode": command["opponent_death_mode"],
        "opponent_death_mode_diagnostic": command["opponent_death_mode_diagnostic"],
        "opponent_death_mode_claim": command["opponent_death_mode_claim"],
        "opponent_runtime_mode": command["opponent_runtime_mode"],
        "opponent_runtime_mode_claim": command["opponent_runtime_mode_claim"],
        "opponent_visibility_mode": command["opponent_visibility_mode"],
        "opponent_collision_effect": command["opponent_collision_effect"],
        "opponent_trail_mode": command["opponent_trail_mode"],
        "natural_bonus_spawn": command["natural_bonus_spawn"],
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
        "opponent_mixture": _to_plain(command["opponent_mixture"]),
        "opponent_assignment_context": _to_plain(command["opponent_assignment_context"]),
        "save_ckpt_after_iter": command["save_ckpt_after_iter"],
        "max_env_step": command["max_env_step"],
        "max_train_iter": command["max_train_iter"],
    }
    if command["opponent_policy_kind"] == OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT:
        expected["opponent_checkpoint_ref"] = command["opponent_checkpoint_report_ref"]
        expected["opponent_snapshot_ref"] = command["opponent_snapshot_ref"]
        expected["opponent_checkpoint_state_key"] = command["opponent_checkpoint_state_key"]
        expected["opponent_use_cuda"] = command["opponent_use_cuda"]
    problems = []
    for key, value in expected.items():
        if surface.get(key) != value:
            problems.append(f"{key}={surface.get(key)!r}, expected {value!r}")
    return problems


def _source_state_fixed_opponent_readiness_expected(
    *,
    opponent_policy_kind: str = OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
    opponent_runtime_mode: str = DEFAULT_OPPONENT_RUNTIME_MODE,
    opponent_mixture_enabled: bool = False,
) -> dict[str, Any]:
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
        "opponent_policy_kind": opponent_policy_kind,
        "opponent_runtime_mode": opponent_runtime_mode,
        "opponent_training_relation": (
            OPPONENT_TRAINING_RELATION_WEIGHTED_EPISODE_MIXTURE
            if opponent_mixture_enabled
            else _opponent_training_relation(opponent_policy_kind)
        ),
        "current_policy_self_play": spec["current_policy_self_play"],
        "trusted_current_policy_self_play": spec["trusted_current_policy_self_play"],
        "simultaneous_game_theory_claim": spec["simultaneous_game_theory_claim"],
        "two_seat_self_play": spec["two_seat_self_play"],
        "two_seat_self_play_status": spec["two_seat_self_play_status"],
        "fixed_opponent_is_two_seat_self_play": spec["fixed_opponent_is_two_seat_self_play"],
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
    expected = _source_state_fixed_opponent_readiness_expected(
        opponent_policy_kind=str(
            command.get("opponent_policy_kind", OPPONENT_POLICY_KIND_FIXED_STRAIGHT)
        ),
        opponent_runtime_mode=str(
            command.get("opponent_runtime_mode", DEFAULT_OPPONENT_RUNTIME_MODE)
        ),
        opponent_mixture_enabled=bool(
            command.get("opponent_mixture_enabled") or command.get("opponent_mixture")
        ),
    )
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


def _compile_config_summary(
    main_config: Any,
    create_config: Any,
    *,
    seed: int,
    allow_profile_custom_env_manager_skip: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    env_manager_cfg = _cfg_get(create_config, "env_manager", {})
    env_manager_type = _cfg_get(env_manager_cfg, "type", None)
    if (
        allow_profile_custom_env_manager_skip
        and str(env_manager_type) in CURVYZERO_BATCHED_PROFILE_ENV_MANAGER_TYPES
    ):
        env_cfg = _cfg_get(main_config, "env", {})
        policy_cfg = _cfg_get(main_config, "policy", {})
        model_cfg = _cfg_get(policy_cfg, "model", {})
        return {
            "ok": True,
            "skipped": True,
            "skip_reason": (
                f"{env_manager_type} is a profile-only custom env manager "
                "installed by a train_muzero hook; stock compile_config does not "
                "know this manager type before the hook is installed"
            ),
            "env_manager_type": env_manager_type,
            "env": {
                "type": _cfg_get(env_cfg, "type", None),
                "env_id": _cfg_get(env_cfg, "env_id", None),
                "collector_env_num": _cfg_get(env_cfg, "collector_env_num", None),
                "evaluator_env_num": _cfg_get(env_cfg, "evaluator_env_num", None),
                "frame_stack_num": _cfg_get(env_cfg, "frame_stack_num", None),
                "reward_variant": _cfg_get(env_cfg, "reward_variant", None),
                "reward_schema_id": _cfg_get(env_cfg, "reward_schema_id", None),
                "source_state_trail_render_mode": _cfg_get(
                    env_cfg,
                    "source_state_trail_render_mode",
                    None,
                ),
                "source_state_bonus_render_mode": _cfg_get(
                    env_cfg,
                    "source_state_bonus_render_mode",
                    None,
                ),
            },
            "policy_model": {
                "model_type": _cfg_get(model_cfg, "model_type", None),
                "observation_shape": _to_plain(_cfg_get(model_cfg, "observation_shape", None)),
                "image_channel": _cfg_get(model_cfg, "image_channel", None),
                "frame_stack_num": _cfg_get(model_cfg, "frame_stack_num", None),
                "action_space_size": _cfg_get(model_cfg, "action_space_size", None),
            },
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
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
                "source_state_trail_render_mode": _cfg_get(
                    env_cfg,
                    "source_state_trail_render_mode",
                    None,
                ),
                "source_state_bonus_render_mode": _cfg_get(
                    env_cfg,
                    "source_state_bonus_render_mode",
                    None,
                ),
                "default_trail_render_mode": _cfg_get(
                    env_cfg,
                    "default_trail_render_mode",
                    None,
                ),
                "supported_trail_render_modes": _to_plain(
                    _cfg_get(env_cfg, "supported_trail_render_modes", None)
                ),
                "default_bonus_render_mode": _cfg_get(
                    env_cfg,
                    "default_bonus_render_mode",
                    None,
                ),
                "supported_bonus_render_modes": _to_plain(
                    _cfg_get(env_cfg, "supported_bonus_render_modes", None)
                ),
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
                "reward_support_range": _to_plain(
                    _cfg_get(model_cfg, "reward_support_range", None)
                ),
                "value_support_range": _to_plain(_cfg_get(model_cfg, "value_support_range", None)),
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
    return lz_config.target_config_patches(main_config, target_config)


def _set_save_ckpt_after_iter(main_config: Any, value: int) -> dict[str, Any]:
    return lz_config.set_save_ckpt_after_iter(main_config, value)


def _set_load_ckpt_before_run(
    main_config: Any,
    checkpoint_path: str,
    *,
    reason: str = "automatic resume from the latest iteration checkpoint for this run",
) -> dict[str, Any]:
    return lz_config.set_load_ckpt_before_run(
        main_config,
        checkpoint_path,
        reason=reason,
    )


def _prepare_lightzero_auto_resume(
    *,
    run_id: str,
    attempt_id: str,
    exp_name_ref: Any,
) -> dict[str, Any]:
    """Find the newest LightZero iteration checkpoint already saved for this run."""

    try:
        _safe_reload_runs_volume(reason="auto resume checkpoint scan")
    except Exception:
        pass

    run_root = runs.volume_path(RUNS_MOUNT, runs.run_root_ref(TASK_ID, run_id))
    current_exp_name = runs.volume_path(RUNS_MOUNT, exp_name_ref)
    current_ckpt_dir = current_exp_name / "ckpt"
    stable_ckpt_dir = (
        runs.volume_path(RUNS_MOUNT, runs.checkpoints_root_ref(TASK_ID, run_id)) / "lightzero"
    )
    candidates: list[dict[str, Any]] = []
    source_roots: list[dict[str, Any]] = []
    scanned_dirs: set[str] = set()

    def scan_dir(directory: Path, *, source_kind: str) -> None:
        directory_key = str(directory)
        if directory_key in scanned_dirs:
            return
        scanned_dirs.add(directory_key)
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
        for candidate in lz_checkpoints.collect_lightzero_iteration_checkpoints(
            [directory],
            require_non_empty=True,
        ):
            candidates.append(
                {
                    "found": True,
                    "source_kind": source_kind,
                    "checkpoint_path": str(candidate.path),
                    "checkpoint_ref": runs.file_ref(candidate.path, mount=RUNS_MOUNT),
                    "name": candidate.checkpoint_name,
                    "iteration": int(candidate.iteration),
                    "size_bytes": int(candidate.size_bytes),
                    "mtime_ns": int(candidate.mtime_ns),
                }
            )

    for ckpt_dir in _lightzero_exp_checkpoint_dirs(current_exp_name):
        scan_dir(ckpt_dir, source_kind=f"current_attempt_lightzero_exp:{ckpt_dir.parent.name}")

    attempts_dir = run_root / "attempts"
    if attempts_dir.is_dir():
        for attempt_dir in sorted(attempts_dir.iterdir(), key=lambda path: path.name):
            if not attempt_dir.is_dir():
                continue
            source_kind = (
                "same_attempt_lightzero_exp"
                if attempt_dir.name == attempt_id
                else f"prior_attempt_lightzero_exp:{attempt_dir.name}"
            )
            exp_name = attempt_dir / "train" / "lightzero_exp"
            for ckpt_dir in _lightzero_exp_checkpoint_dirs(exp_name):
                if ckpt_dir == current_ckpt_dir:
                    continue
                scan_dir(ckpt_dir, source_kind=f"{source_kind}:{ckpt_dir.parent.name}")

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
    current_exp_name = runs.volume_path(RUNS_MOUNT, exp_name_ref)
    current_state_dir = current_exp_name / LIGHTZERO_RESUME_STATE_DIRNAME
    stable_state_dir = (
        runs.volume_path(RUNS_MOUNT, runs.checkpoints_root_ref(TASK_ID, run_id))
        / LIGHTZERO_RESUME_STATE_DIRNAME
    )
    sidecar_name = _lightzero_resume_state_name(iteration)
    source_roots: list[dict[str, Any]] = []
    source_kind_by_dir: dict[str, str] = {}
    scanned_dirs: set[str] = set()

    def record_source_dir(directory: Path, *, source_kind: str) -> None:
        directory_key = str(directory)
        if directory_key in scanned_dirs:
            return
        scanned_dirs.add(directory_key)
        source_kind_by_dir[directory_key] = source_kind
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

    for state_dir in _lightzero_exp_resume_state_dirs(current_exp_name):
        record_source_dir(
            state_dir,
            source_kind=f"current_attempt_lightzero_exp:{state_dir.parent.name}",
        )

    attempts_dir = run_root / "attempts"
    if attempts_dir.is_dir():
        for attempt_dir in sorted(attempts_dir.iterdir(), key=lambda path: path.name):
            if not attempt_dir.is_dir():
                continue
            source_kind = (
                "same_attempt_lightzero_exp"
                if attempt_dir.name == attempt_id
                else f"prior_attempt_lightzero_exp:{attempt_dir.name}"
            )
            exp_name = attempt_dir / "train" / "lightzero_exp"
            for state_dir in _lightzero_exp_resume_state_dirs(exp_name):
                if state_dir == current_state_dir:
                    continue
                record_source_dir(state_dir, source_kind=f"{source_kind}:{state_dir.parent.name}")

    record_source_dir(stable_state_dir, source_kind="run_resume_state_mirror")
    candidates = lz_checkpoints.collect_lightzero_resume_state_candidates(
        [Path(item["path"]) for item in source_roots],
        iteration=iteration,
        require_non_empty=True,
    )

    base = {
        "resume_state_lookup": {
            "iteration": int(iteration),
            "name": sidecar_name,
            "source_roots": source_roots,
            "candidate_count": len(candidates),
        },
        "resume_state_found": False,
    }
    latest = lz_checkpoints.latest_lightzero_resume_state_candidate(candidates)
    if latest is None:
        return base
    state_dir = latest.path.parent
    return {
        **base,
        "resume_state_found": True,
        "resume_state_source_kind": source_kind_by_dir.get(str(state_dir), "unknown"),
        "resume_state_path": str(latest.path),
        "resume_state_ref": runs.file_ref(latest.path, mount=RUNS_MOUNT),
        "resume_state_name": latest.state_name,
        "resume_state_iteration": int(latest.iteration),
        "resume_state_size_bytes": int(latest.size_bytes),
        "resume_state_mtime_ns": int(latest.mtime_ns),
    }


def _is_under_run_mount(path: Path) -> bool:
    try:
        path.relative_to(RUNS_MOUNT)
    except ValueError:
        return False
    return True


def _lightzero_iteration_from_checkpoint_name(name: str) -> int | None:
    return lz_checkpoints.lightzero_iteration_from_checkpoint_name(name)


def _lightzero_iteration_from_resume_state_name(name: str) -> int | None:
    return lz_checkpoints.lightzero_iteration_from_resume_state_name(name)


def _opponent_training_relation(opponent_policy_kind: str) -> str:
    return lz_config.opponent_training_relation(opponent_policy_kind)


def _opponent_training_relation_for_surface(
    *,
    env_variant: str,
    opponent_policy_kind: str,
    env_spec: dict[str, Any],
    opponent_mixture: dict[str, Any] | None,
) -> str:
    return lz_config.opponent_training_relation_for_surface(
        env_variant=env_variant,
        opponent_policy_kind=opponent_policy_kind,
        env_spec=env_spec,
        opponent_mixture=opponent_mixture,
    )


def _parse_training_signals(stdout_text: str, stderr_text: str) -> dict[str, Any]:
    text = stdout_text + "\n" + stderr_text
    checkpoint_saves = re.findall(r"learner save ckpt in\s+([^\n]+)", text)
    checkpoint_iterations = sorted(
        {
            int(value)
            for value in re.findall(r"iteration_(\d+)\.pth\.tar", "\n".join(checkpoint_saves))
        }
    )
    training_iterations = [int(value) for value in re.findall(r"Training Iteration\s+(\d+)", text)]
    final_rewards = [
        float(value) for value in re.findall(r"final reward:\s*([-+]?\d+(?:\.\d+)?)", text)
    ]
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
    roots: list[Path] = []
    seen_roots: set[str] = set()

    def add_roots(paths: list[Path]) -> None:
        for path in paths:
            key = str(path)
            if key in seen_roots:
                continue
            seen_roots.add(key)
            roots.append(path)

    add_roots(_lightzero_exp_sibling_roots(root))
    if not root.is_absolute():
        add_roots(_lightzero_exp_sibling_roots(RUNS_MOUNT / root))
    if exp_name.startswith("/"):
        add_roots(_lightzero_exp_sibling_roots(Path("." + exp_name)))
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
    training_reward_variant = str(
        command.get(
            "training_reward_variant",
            command.get("reward_variant", DEFAULT_REWARD_VARIANT),
        )
    )
    eval_reward_variant = str(command.get("eval_reward_variant", training_reward_variant))
    model_reward_variant = str(
        command.get(
            "model_reward_variant",
            command.get("reward_variant", DEFAULT_REWARD_VARIANT),
        )
    )
    reward_outcome_alpha = float(
        command.get("reward_outcome_alpha", DEFAULT_REWARD_OUTCOME_ALPHA)
    )
    return {
        "enabled": bool(command.get("background_eval_enabled", DEFAULT_BACKGROUND_EVAL_ENABLED)),
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
        "decision_ms": float(command.get("decision_ms", DEFAULT_DECISION_MS)),
        "decision_source_frames": int(
            command.get("decision_source_frames", DEFAULT_DECISION_SOURCE_FRAMES)
        ),
        "source_physics_step_ms": float(
            command.get("source_physics_step_ms", DEFAULT_SOURCE_PHYSICS_STEP_MS)
        ),
        "source_max_steps_semantics": str(
            command.get("source_max_steps_semantics", "source_physics_steps")
        ),
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
        "reward_variant": eval_reward_variant,
        "reward_outcome_alpha": reward_outcome_alpha,
        "eval_reward_variant": eval_reward_variant,
        "training_reward_variant": training_reward_variant,
        "model_reward_variant": model_reward_variant,
        "opponent_policy_kind": str(
            command.get("opponent_policy_kind", DEFAULT_OPPONENT_POLICY_KIND)
        ),
        "opponent_checkpoint_ref": command.get("opponent_checkpoint_ref"),
        "opponent_snapshot_ref": command.get("opponent_snapshot_ref"),
        "opponent_checkpoint_state_key": command.get("opponent_checkpoint_state_key"),
        "opponent_death_mode": str(command.get("opponent_death_mode", DEFAULT_OPPONENT_DEATH_MODE)),
        "opponent_runtime_mode": str(
            command.get("opponent_runtime_mode", DEFAULT_OPPONENT_RUNTIME_MODE)
        ),
        "opponent_mixture": _to_plain(command.get("opponent_mixture")),
        "selfplay_gif": _background_gif_config_from_command(command),
    }


def _background_gif_config_from_command(command: dict[str, Any]) -> dict[str, Any]:
    max_steps = _normalize_background_gif_max_steps(
        command.get("background_gif_max_steps", DEFAULT_BACKGROUND_GIF_MAX_STEPS)
    )
    requested_frame_size = int(
        command.get("background_gif_frame_size", DEFAULT_BACKGROUND_GIF_FRAME_SIZE)
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
        "requested_frame_size": requested_frame_size,
        "frame_size": CHECKPOINT_SELFPLAY_GIF_FRAME_SIZE,
        "frame_size_policy": ("checkpoint_selfplay_gif_always_uses_full_source_state_rgb_canvas"),
        "collect_temperature": float(
            command.get(
                "background_gif_collect_temperature",
                DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE,
            )
        ),
        "collect_epsilon": float(
            command.get(
                "background_gif_collect_epsilon",
                DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON,
            )
        ),
        "natural_bonus_spawn": bool(
            command.get("natural_bonus_spawn", TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN)
        ),
        "source_max_steps": int(command.get("source_max_steps", DEFAULT_SOURCE_MAX_STEPS)),
        "decision_ms": float(command.get("decision_ms", DEFAULT_DECISION_MS)),
        "decision_source_frames": int(
            command.get("decision_source_frames", DEFAULT_DECISION_SOURCE_FRAMES)
        ),
        "source_physics_step_ms": float(
            command.get("source_physics_step_ms", DEFAULT_SOURCE_PHYSICS_STEP_MS)
        ),
        "source_max_steps_semantics": str(
            command.get("source_max_steps_semantics", "source_physics_steps")
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
        "training_env_variant": str(command.get("env_variant", DEFAULT_ENV_VARIANT)),
        "training_reward_variant": str(
            command.get(
                "training_reward_variant",
                command.get("reward_variant", DEFAULT_REWARD_VARIANT),
            )
        ),
        "reward_outcome_alpha": float(
            command.get("reward_outcome_alpha", DEFAULT_REWARD_OUTCOME_ALPHA)
        ),
        "opponent_death_mode": str(command.get("opponent_death_mode", DEFAULT_OPPONENT_DEATH_MODE)),
        "opponent_runtime_mode": str(
            command.get("opponent_runtime_mode", DEFAULT_OPPONENT_RUNTIME_MODE)
        ),
        "opponent_mixture": _to_plain(command.get("opponent_mixture")),
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
    checkpoints = [item for item in copied if isinstance(item, dict) and item.get("copied_now")]
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
            decision_ms=float(config.get("decision_ms", DEFAULT_DECISION_MS)),
            decision_source_frames=int(
                config.get("decision_source_frames", DEFAULT_DECISION_SOURCE_FRAMES)
            ),
            source_physics_step_ms=float(
                config.get("source_physics_step_ms", DEFAULT_SOURCE_PHYSICS_STEP_MS)
            ),
            source_max_steps_semantics=str(
                config.get("source_max_steps_semantics", "source_physics_steps")
            ),
            num_simulations=int(config["num_simulations"]),
            batch_size=int(config["batch_size"]),
            env_variant=str(config["env_variant"]),
            reward_variant=str(
                config.get(
                    "eval_reward_variant",
                    config.get("reward_variant", DEFAULT_REWARD_VARIANT),
                )
            ),
            model_reward_variant=str(
                config.get(
                    "model_reward_variant",
                    config.get("training_reward_variant", DEFAULT_REWARD_VARIANT),
                )
            ),
            reward_outcome_alpha=float(
                config.get("reward_outcome_alpha", DEFAULT_REWARD_OUTCOME_ALPHA)
            ),
            opponent_policy_kind=str(config["opponent_policy_kind"]),
            opponent_checkpoint_ref=config.get("opponent_checkpoint_ref"),
            opponent_snapshot_ref=config.get("opponent_snapshot_ref"),
            opponent_checkpoint_state_key=config.get("opponent_checkpoint_state_key"),
            opponent_mixture_spec=config.get("opponent_mixture"),
            opponent_death_mode=str(config.get("opponent_death_mode", DEFAULT_OPPONENT_DEATH_MODE)),
            opponent_runtime_mode=str(
                config.get("opponent_runtime_mode", DEFAULT_OPPONENT_RUNTIME_MODE)
            ),
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
    requested_frame_size = int(
        gif_config.get(
            "requested_frame_size",
            gif_config.get("frame_size", DEFAULT_BACKGROUND_GIF_FRAME_SIZE),
        )
    )
    gif_config["requested_frame_size"] = requested_frame_size
    gif_config["frame_size"] = CHECKPOINT_SELFPLAY_GIF_FRAME_SIZE
    gif_config["frame_size_policy"] = (
        "checkpoint_selfplay_gif_always_uses_full_source_state_rgb_canvas"
    )
    gif_config["decision_ms"] = float(
        gif_config.get("decision_ms", config.get("decision_ms", DEFAULT_DECISION_MS))
    )
    gif_config["decision_source_frames"] = int(
        gif_config.get(
            "decision_source_frames",
            config.get("decision_source_frames", DEFAULT_DECISION_SOURCE_FRAMES),
        )
    )
    gif_config["source_physics_step_ms"] = float(
        gif_config.get(
            "source_physics_step_ms",
            config.get("source_physics_step_ms", DEFAULT_SOURCE_PHYSICS_STEP_MS),
        )
    )
    gif_config["source_max_steps_semantics"] = str(
        gif_config.get(
            "source_max_steps_semantics",
            config.get("source_max_steps_semantics", "source_physics_steps"),
        )
    )
    base_seed = int(gif_config.get("seed", DEFAULT_SEED + DEFAULT_BACKGROUND_GIF_SEED_OFFSET))
    checkpoint_seed_mix = _stable_seed_mix(checkpoint_ref, checkpoint_label, eval_id)
    checkpoint_seed_mixing_enabled = bool(
        gif_config.get(
            "checkpoint_seed_mixing_enabled",
            DEFAULT_BACKGROUND_GIF_CHECKPOINT_SEED_MIXING_ENABLED,
        )
    )
    effective_seed = (
        _mix_seed(base_seed, checkpoint_seed_mix) if checkpoint_seed_mixing_enabled else base_seed
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
            decision_ms=float(gif_config["decision_ms"]),
            decision_source_frames=int(gif_config["decision_source_frames"]),
            source_physics_step_ms=float(gif_config["source_physics_step_ms"]),
            source_max_steps_semantics=str(gif_config["source_max_steps_semantics"]),
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
            collect_temperature=float(
                gif_config.get(
                    "collect_temperature",
                    DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE,
                )
            ),
            collect_epsilon=float(
                gif_config.get(
                    "collect_epsilon",
                    DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON,
                )
            ),
            training_env_variant=training_env_variant,
            training_reward_variant=str(
                gif_config.get("training_reward_variant", DEFAULT_REWARD_VARIANT)
            ),
            training_reward_outcome_alpha=float(
                gif_config.get("reward_outcome_alpha", DEFAULT_REWARD_OUTCOME_ALPHA)
            ),
            opponent_death_mode=str(
                gif_config.get("opponent_death_mode", DEFAULT_OPPONENT_DEATH_MODE)
            ),
            opponent_runtime_mode=str(
                gif_config.get("opponent_runtime_mode", DEFAULT_OPPONENT_RUNTIME_MODE)
            ),
            opponent_mixture_spec=gif_config.get("opponent_mixture"),
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
        isinstance(attempt, dict) and attempt.get("status") in {"completed", "failed", "superseded"}
    )


def _checkpoint_eval_poller_command(
    *,
    seed: int,
    source_max_steps: int,
    decision_ms: float = DEFAULT_DECISION_MS,
    decision_source_frames: int = DEFAULT_DECISION_SOURCE_FRAMES,
    source_physics_step_ms: float = DEFAULT_SOURCE_PHYSICS_STEP_MS,
    source_max_steps_semantics: str = "source_physics_steps",
    env_variant: str,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
    opponent_policy_kind: str,
    opponent_checkpoint_ref: str | None,
    opponent_snapshot_ref: str | None,
    opponent_checkpoint_state_key: str | None,
    opponent_mixture_spec: Any | None = None,
    opponent_assignment_ref: str | None = None,
    opponent_death_mode: str = DEFAULT_OPPONENT_DEATH_MODE,
    opponent_runtime_mode: str = DEFAULT_OPPONENT_RUNTIME_MODE,
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
    background_gif_collect_temperature: float = DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE,
    background_gif_collect_epsilon: float = DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON,
    background_gif_checkpoint_seed_mixing_enabled: bool = (
        DEFAULT_BACKGROUND_GIF_CHECKPOINT_SEED_MIXING_ENABLED
    ),
    natural_bonus_spawn: bool = TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN,
) -> dict[str, Any]:
    reload_assignment_volume = bool(
        opponent_assignment_ref
        and str(opponent_assignment_ref).startswith((_CONTROL_REF_PREFIX, _RUNS_REF_PREFIX))
    )
    opponent_assignment = _resolve_opponent_assignment_for_env(
        opponent_assignment_ref=opponent_assignment_ref,
        reload_volume_before_read=reload_assignment_volume,
        reload_checkpoint_volume_before_read=reload_assignment_volume,
    )
    if opponent_assignment is not None and opponent_mixture_spec is not None:
        raise ValueError("opponent_assignment_ref cannot be combined with opponent_mixture_spec")
    effective_opponent_mixture = (
        opponent_assignment["opponent_mixture"]
        if opponent_assignment is not None
        else opponent_mixture_spec
    )
    return {
        "seed": seed,
        "source_max_steps": source_max_steps,
        "decision_ms": float(decision_ms),
        "decision_source_frames": int(decision_source_frames),
        "source_physics_step_ms": float(source_physics_step_ms),
        "source_max_steps_semantics": str(source_max_steps_semantics),
        "env_variant": env_variant,
        "reward_variant": reward_variant,
        "reward_outcome_alpha": float(_normalize_reward_outcome_alpha(reward_outcome_alpha)),
        "eval_reward_variant": reward_variant,
        "training_reward_variant": reward_variant,
        "model_reward_variant": reward_variant,
        "opponent_policy_kind": opponent_policy_kind,
        "opponent_checkpoint_ref": opponent_checkpoint_ref,
        "opponent_snapshot_ref": opponent_snapshot_ref,
        "opponent_checkpoint_state_key": opponent_checkpoint_state_key,
        "opponent_mixture": _to_plain(effective_opponent_mixture),
        "opponent_assignment_ref": opponent_assignment_ref,
        "opponent_assignment": (
            {
                key: opponent_assignment.get(key)
                for key in (
                    "assignment_id",
                    "source_epoch",
                    "source_ref",
                    "assignment_ref",
                    "assignment_sha256",
                    "assignment_file",
                )
            }
            if opponent_assignment is not None
            else None
        ),
        "opponent_death_mode": opponent_death_mode,
        "opponent_runtime_mode": opponent_runtime_mode,
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
        "background_gif_collect_temperature": background_gif_collect_temperature,
        "background_gif_collect_epsilon": background_gif_collect_epsilon,
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
        try:
            _safe_reload_runs_volume(reason="checkpoint eval poller scan")
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
        "fixed_opponent_is_two_seat_self_play": command.get("fixed_opponent_is_two_seat_self_play"),
        "browser_pixel_fidelity": command.get("browser_pixel_fidelity"),
        "uses_ale": command.get("uses_ale"),
        "visual_surface": command.get("visual_surface"),
        "visual_truth_level": command.get("visual_truth_level"),
        "visual_source_state_backed": command.get("visual_source_state_backed"),
        "source_fidelity_claim": command.get("source_fidelity_claim"),
        "source_state_trail_render_mode": command.get("source_state_trail_render_mode"),
        "source_state_bonus_render_mode": command.get("source_state_bonus_render_mode"),
        "opponent_death_mode": command.get("opponent_death_mode"),
        "opponent_runtime_mode": command.get("opponent_runtime_mode"),
        "opponent_runtime_mode_claim": command.get("opponent_runtime_mode_claim"),
        "opponent_visibility_mode": command.get("opponent_visibility_mode"),
        "opponent_collision_effect": command.get("opponent_collision_effect"),
        "opponent_trail_mode": command.get("opponent_trail_mode"),
        "default_trail_render_mode": command.get("default_trail_render_mode"),
        "supported_trail_render_modes": command.get("supported_trail_render_modes"),
        "default_bonus_render_mode": command.get("default_bonus_render_mode"),
        "supported_bonus_render_modes": command.get("supported_bonus_render_modes"),
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
        try:
            _safe_reload_runs_volume(reason="wait for visible checkpoint")
        except Exception:
            pass
        if checkpoint_path.is_file():
            break
        time.sleep(DEFAULT_BACKGROUND_CHECKPOINT_WAIT_POLL_SEC)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint was not visible in the Modal volume before timeout: {checkpoint_ref}"
        )
    return checkpoint_path


def _is_transient_volume_commit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        exc.__class__.__name__ in {"DataLossError", "GRPCError", "RetryError"}
        or "failed to publish commit" in text
        or "transport is closing" in text
        or "deadline" in text
    )


def _commit_runs_volume_with_backoff(
    *,
    label: str,
    attempts: int = 6,
    initial_jitter_sec: float = 0.0,
    max_sleep_sec: float = 20.0,
) -> None:
    _commit_volume_with_backoff(
        runs_volume,
        label=label,
        attempts=attempts,
        initial_jitter_sec=initial_jitter_sec,
        max_sleep_sec=max_sleep_sec,
    )


def _commit_control_volume_with_backoff(
    *,
    label: str,
    attempts: int = 6,
    initial_jitter_sec: float = 0.0,
    max_sleep_sec: float = 20.0,
) -> None:
    _commit_volume_with_backoff(
        control_volume,
        label=label,
        attempts=attempts,
        initial_jitter_sec=initial_jitter_sec,
        max_sleep_sec=max_sleep_sec,
    )


def _commit_volume_with_backoff(
    volume: Any,
    *,
    label: str,
    attempts: int = 6,
    initial_jitter_sec: float = 0.0,
    max_sleep_sec: float = 20.0,
) -> None:
    if not hasattr(volume, "commit"):
        return
    if initial_jitter_sec > 0:
        time.sleep(random.uniform(0.0, float(initial_jitter_sec)))
    for attempt_index in range(int(attempts)):
        try:
            volume.commit()
            return
        except Exception as exc:
            is_last = attempt_index >= int(attempts) - 1
            if is_last or not _is_transient_volume_commit_error(exc):
                raise
            delay_sec = min(float(max_sleep_sec), 2.0**attempt_index) + random.uniform(
                0.0,
                min(5.0, 1.0 + attempt_index),
            )
            print(
                json.dumps(
                    {
                        "event": "modal_volume_commit_retry",
                        "label": label,
                        "attempt": attempt_index + 1,
                        "next_delay_sec": round(delay_sec, 3),
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(delay_sec)


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
    decision_ms: float = DEFAULT_DECISION_MS,
    decision_source_frames: int = DEFAULT_DECISION_SOURCE_FRAMES,
    source_physics_step_ms: float = DEFAULT_SOURCE_PHYSICS_STEP_MS,
    source_max_steps_semantics: str = "source_physics_steps",
    num_simulations: int,
    batch_size: int,
    env_variant: str,
    reward_variant: str,
    reward_outcome_alpha: float,
    model_reward_variant: str | None,
    opponent_policy_kind: str,
    opponent_checkpoint_ref: str | None,
    opponent_snapshot_ref: str | None,
    opponent_checkpoint_state_key: str | None,
    opponent_mixture_spec: Any | None = None,
    opponent_assignment_ref: str | None = None,
    opponent_death_mode: str = DEFAULT_OPPONENT_DEATH_MODE,
    opponent_runtime_mode: str = DEFAULT_OPPONENT_RUNTIME_MODE,
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

    try:
        _safe_reload_runs_volume(reason="background checkpoint eval startup")
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
    opponent_assignment = _resolve_opponent_assignment_for_env(
        opponent_assignment_ref=opponent_assignment_ref
    )
    if opponent_assignment is not None and opponent_mixture_spec is not None:
        raise ValueError("opponent_assignment_ref cannot be combined with opponent_mixture_spec")
    effective_opponent_mixture = (
        opponent_assignment["opponent_mixture"]
        if opponent_assignment is not None
        else opponent_mixture_spec
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
            decision_ms=decision_ms,
            decision_source_frames=decision_source_frames,
            source_physics_step_ms=source_physics_step_ms,
            source_max_steps_semantics=source_max_steps_semantics,
            num_simulations=num_simulations,
            batch_size=batch_size,
            emit_result_json=False,
            quiet_framework_logs=True,
            env_variant=env_variant,
            reward_variant=reward_variant,
            reward_outcome_alpha=float(reward_outcome_alpha),
            model_reward_variant=model_reward_variant,
            model_reward_outcome_alpha=float(reward_outcome_alpha),
            opponent_policy_kind=opponent_policy_kind,
            opponent_checkpoint_ref=opponent_checkpoint_ref,
            opponent_snapshot_ref=opponent_snapshot_ref,
            opponent_checkpoint_state_key=opponent_checkpoint_state_key,
            opponent_mixture_spec=effective_opponent_mixture,
            opponent_death_mode=opponent_death_mode,
            opponent_runtime_mode=opponent_runtime_mode,
            natural_bonus_spawn=bool(natural_bonus_spawn),
            commit=False,
        )
        jobs.append(job)
        results.append(result)

    table = [
        eval_mod._row_from_result(job, result) for job, result in zip(jobs, results, strict=True)
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
            "decision_ms": float(decision_ms),
            "decision_source_frames": int(decision_source_frames),
            "source_physics_step_ms": float(source_physics_step_ms),
            "source_max_steps_semantics": str(source_max_steps_semantics),
            "eval_reward_variant": reward_variant,
            "env_reward_variant": reward_variant,
            "reward_variant": reward_variant,
            "reward_variant_role": "backward_compatible_alias_for_eval_reward_variant",
            "model_reward_variant": model_reward_variant,
            "effective_model_reward_variant": model_reward_variant or reward_variant,
            "model_reward_variant_role": "checkpoint_model_reconstruction_only_not_scoring",
            "opponent_policy_kind": opponent_policy_kind,
            "opponent_checkpoint_ref": opponent_checkpoint_ref,
            "opponent_mixture_enabled": effective_opponent_mixture is not None,
            "opponent_mixture": _to_plain(effective_opponent_mixture),
            "opponent_assignment_ref": opponent_assignment_ref,
            "opponent_assignment": (
                {
                    key: opponent_assignment.get(key)
                    for key in (
                        "assignment_id",
                        "source_epoch",
                        "source_ref",
                        "assignment_ref",
                        "assignment_sha256",
                        "assignment_file",
                    )
                }
                if opponent_assignment is not None
                else None
            ),
            "opponent_snapshot_ref": opponent_snapshot_ref,
            "opponent_checkpoint_state_key": opponent_checkpoint_state_key,
            "opponent_runtime_mode": opponent_runtime_mode,
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
            "decision_ms": float(decision_ms),
            "decision_source_frames": int(decision_source_frames),
            "source_physics_step_ms": float(source_physics_step_ms),
            "source_max_steps_semantics": str(source_max_steps_semantics),
            "num_simulations": num_simulations,
            "batch_size": batch_size,
            "slim_manifest": True,
            "LightZero": LIGHTZERO_VERSION,
            "remote_root": str(REMOTE_ROOT),
            "volume_name": VOLUME_NAME,
            "opponent_policy_kind": opponent_policy_kind,
            "opponent_checkpoint_ref": opponent_checkpoint_ref,
            "opponent_mixture_enabled": effective_opponent_mixture is not None,
            "opponent_mixture": _to_plain(effective_opponent_mixture),
            "opponent_assignment_ref": opponent_assignment_ref,
            "opponent_assignment": (
                {
                    key: opponent_assignment.get(key)
                    for key in (
                        "assignment_id",
                        "source_epoch",
                        "source_ref",
                        "assignment_ref",
                        "assignment_sha256",
                        "assignment_file",
                    )
                }
                if opponent_assignment is not None
                else None
            ),
            "opponent_snapshot_ref": opponent_snapshot_ref,
            "opponent_checkpoint_state_key": opponent_checkpoint_state_key,
            "opponent_runtime_mode": opponent_runtime_mode,
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
    _commit_runs_volume_with_backoff(
        label="checkpoint_eval_and_inspect",
        initial_jitter_sec=8.0,
    )

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
    duration_ms = max(
        int(DEFAULT_BACKGROUND_GIF_MIN_FRAME_DURATION_MS),
        int(round(1000.0 / fps)),
    )
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


def _checkpoint_gif_policy_action(
    *,
    eval_mod: Any,
    policy: Any,
    observation: dict[str, Any],
    policy_mode: str,
    collect_temperature: float,
    collect_epsilon: float,
) -> dict[str, Any]:
    if policy_mode == BACKGROUND_GIF_POLICY_MODE_EVAL_GREEDY:
        result = dict(eval_mod._policy_eval_action(policy, observation))
        result["policy_mode"] = BACKGROUND_GIF_POLICY_MODE_EVAL_GREEDY
        result["policy_mode_label"] = "Greedy eval"
        result["temperature"] = 0.0
        result["epsilon"] = 0.0
        return result
    if policy_mode != BACKGROUND_GIF_POLICY_MODE_COLLECT:
        raise ValueError(
            "background GIF policy mode must be one of "
            f"{BACKGROUND_GIF_POLICY_MODE_CHOICES!r}; got {policy_mode!r}"
        )
    if collect_temperature <= 0.0:
        raise ValueError("collect temperature must be positive")
    if not 0.0 <= collect_epsilon <= 1.0:
        raise ValueError("collect epsilon must be in [0, 1]")

    import numpy as np
    import torch

    obs_tensor = torch.as_tensor(
        np.asarray([observation["observation"]]),
        dtype=torch.float32,
        device=eval_mod._policy_model_device(policy),
    )
    action_mask = np.asarray([observation["action_mask"]], dtype=np.float32)
    to_play = [int(np.asarray(observation.get("to_play", -1)).reshape(-1)[0])]
    ready_env_id = np.asarray([0])
    with torch.no_grad():
        output = policy.collect_mode.forward(
            obs_tensor,
            action_mask=action_mask,
            temperature=float(collect_temperature),
            to_play=to_play,
            epsilon=float(collect_epsilon),
            ready_env_id=ready_env_id,
        )
    return {
        "ok": True,
        "source": "policy_collect_mode",
        "policy_mode": BACKGROUND_GIF_POLICY_MODE_COLLECT,
        "policy_mode_label": "Collect sample",
        "action": eval_mod._extract_eval_action(output),
        "temperature": float(collect_temperature),
        "epsilon": float(collect_epsilon),
        "compact_output": eval_mod._compact_mcts_output(output),
    }


def _checkpoint_gif_collect_label(*, collect_temperature: float, collect_epsilon: float) -> str:
    return f"Train collect T={collect_temperature:g} eps={collect_epsilon:g}"


def _checkpoint_gif_variant_specs(
    *,
    collect_temperature: float,
    collect_epsilon: float,
) -> list[dict[str, Any]]:
    return [
        {
            "variant_id": BACKGROUND_GIF_VARIANT_EVAL_GREEDY,
            "label": "Greedy eval",
            "policy_mode": BACKGROUND_GIF_POLICY_MODE_EVAL_GREEDY,
            "temperature": 0.0,
            "epsilon": 0.0,
            "gif_filename": BACKGROUND_GIF_VARIANT_FILENAMES[BACKGROUND_GIF_VARIANT_EVAL_GREEDY],
            "raw_frames_filename": "raw_frames.npz",
            "telemetry_filename": "turn_commit_env_steps.jsonl",
            "compatibility_role": "legacy_raw_gif",
        },
        {
            "variant_id": BACKGROUND_GIF_VARIANT_COLLECT_T1,
            "label": _checkpoint_gif_collect_label(
                collect_temperature=float(collect_temperature),
                collect_epsilon=float(collect_epsilon),
            ),
            "policy_mode": BACKGROUND_GIF_POLICY_MODE_COLLECT,
            "temperature": float(collect_temperature),
            "epsilon": float(collect_epsilon),
            "gif_filename": BACKGROUND_GIF_VARIANT_FILENAMES[BACKGROUND_GIF_VARIANT_COLLECT_T1],
            "raw_frames_filename": "collect_t1_frames.npz",
            "telemetry_filename": "turn_commit_env_steps_collect_t1.jsonl",
            "compatibility_role": "training_like_collect_sample",
        },
    ]


def _checkpoint_gif_capture_surface(
    *,
    training_env_variant: str,
    training_reward_variant: str,
    opponent_runtime_mode: str,
    opponent_mixture: dict[str, Any] | None,
) -> dict[str, str]:
    if opponent_mixture is not None:
        return {
            "env_variant": training_env_variant,
            "reward_variant": training_reward_variant,
            "reason": (
                "one checkpoint controls player_0 while the source-state fixed-opponent "
                "env samples one opponent-mixture component at reset"
            ),
        }
    if opponent_runtime_mode == OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP:
        return {
            "env_variant": training_env_variant,
            "reward_variant": training_reward_variant,
            "reason": (
                "one checkpoint controls player_0 on the same blank-canvas training "
                "surface; opponent runtime is disabled"
            ),
        }
    return {
        "env_variant": ENV_VARIANT_SOURCE_STATE_TURN_COMMIT,
        "reward_variant": DEFAULT_REWARD_VARIANT,
        "reason": (
            "one checkpoint controls player_0 and player_1 through the source-state "
            "turn-commit adapter; only physical commits become GIF frames"
        ),
    }


def _capture_checkpoint_selfplay_gif_variant(
    *,
    eval_mod: Any,
    state_dict: dict[str, Any],
    seed: int,
    checkpoint_ref: str,
    clean_checkpoint_label: str,
    artifact_root: Path,
    variant: dict[str, Any],
    source_max_steps: int,
    effective_source_max_steps: int,
    decision_ms: float,
    decision_source_frames: int,
    source_physics_step_ms: float,
    source_max_steps_semantics: str,
    max_step_limit: int | None,
    step_limit_kind: str,
    num_simulations: int,
    batch_size: int,
    frame_stride: int,
    fps: float,
    scale: int,
    frame_size: int,
    training_env_variant: str,
    training_reward_variant: str,
    training_reward_outcome_alpha: float,
    opponent_death_mode: str,
    opponent_runtime_mode: str,
    opponent_mixture: dict[str, Any] | None,
    natural_bonus_spawn: bool,
) -> dict[str, Any]:
    import numpy as np

    gif_path = artifact_root / str(variant["gif_filename"])
    frames_path = artifact_root / str(variant["raw_frames_filename"])
    telemetry_path = artifact_root / str(variant["telemetry_filename"])
    capture_surface = _checkpoint_gif_capture_surface(
        training_env_variant=training_env_variant,
        training_reward_variant=training_reward_variant,
        opponent_runtime_mode=opponent_runtime_mode,
        opponent_mixture=opponent_mixture,
    )
    policy, env, surface = eval_mod._make_policy_and_env(
        state_dict=state_dict,
        seed=int(seed),
        use_cuda=False,
        source_max_steps=int(effective_source_max_steps),
        decision_ms=float(decision_ms),
        decision_source_frames=int(decision_source_frames),
        source_physics_step_ms=float(source_physics_step_ms),
        source_max_steps_semantics=str(source_max_steps_semantics),
        num_simulations=int(num_simulations),
        batch_size=int(batch_size),
        telemetry_path=telemetry_path,
        env_variant=capture_surface["env_variant"],
        reward_variant=capture_surface["reward_variant"],
        reward_outcome_alpha=float(training_reward_outcome_alpha),
        model_env_variant=training_env_variant,
        model_reward_variant=training_reward_variant,
        model_reward_outcome_alpha=float(training_reward_outcome_alpha),
        opponent_policy_kind=OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_checkpoint=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        opponent_mixture=opponent_mixture,
        opponent_death_mode=opponent_death_mode,
        opponent_runtime_mode=opponent_runtime_mode,
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

    while not done and (max_step_limit is None or physical_steps < int(max_step_limit)):
        try:
            action_result = _checkpoint_gif_policy_action(
                eval_mod=eval_mod,
                policy=policy,
                observation=observation,
                policy_mode=str(variant["policy_mode"]),
                collect_temperature=float(variant["temperature"]),
                collect_epsilon=float(variant["epsilon"]),
            )
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
                "policy_mode": variant["policy_mode"],
                "temperature": variant["temperature"],
                "epsilon": variant["epsilon"],
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
        "variant_id": variant["variant_id"],
        "variant_label": variant["label"],
        "policy_mode": variant["policy_mode"],
        "temperature": variant["temperature"],
        "epsilon": variant["epsilon"],
        "frame_source": "source_state_rgb_canvas_like",
        "frame_schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
        "frame_truth_level": SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL,
        "browser_pixel_fidelity": False,
        "frame_capture_method": frame_capture_method,
        "frame_capture_method_counts": frame_capture_method_counts,
        "frame_capture_details_sample": frame_captures[:4],
        "frame_shape": [int(frame_size), int(frame_size), 3],
        "saved_frame_shape": [int(frame_size), int(frame_size), 3],
        "gif_filename": variant["gif_filename"],
        "gif_content_kind": "source_state_rgb_canvas_like_rgb_frames",
        "frame_count": int(raw_frames.shape[0]),
        "frame_stride_physical_steps": int(frame_stride),
        "seed": int(seed),
        "natural_bonus_spawn": bool(natural_bonus_spawn),
        "opponent_death_mode": opponent_death_mode,
        "opponent_mixture_enabled": opponent_mixture is not None,
        "opponent_mixture": _to_plain(opponent_mixture),
        "checkpoint_ref": checkpoint_ref,
        "checkpoint_label": clean_checkpoint_label,
        "max_steps": max_step_limit,
        "step_limit_kind": step_limit_kind,
        "configured_source_max_steps": int(source_max_steps),
        "effective_source_max_steps": int(effective_source_max_steps),
        "decision_ms": float(decision_ms),
        "decision_source_frames": int(decision_source_frames),
        "source_physics_step_ms": float(source_physics_step_ms),
        "source_max_steps_semantics": str(source_max_steps_semantics),
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
        scalar_actions if max_step_limit is None else scalar_actions[: 2 * int(max_step_limit)]
    )
    joint_action_trace = (
        joint_actions if max_step_limit is None else joint_actions[: int(max_step_limit)]
    )
    action_space_size = None
    if isinstance(surface.get("compiled_policy"), dict):
        try:
            action_space_size = int(surface["compiled_policy"]["action_space_size"])
        except (KeyError, TypeError, ValueError):
            action_space_size = None
    action_summary = _action_trace_observability(
        scalar_actions,
        player_field="acting_player_id",
        action_field="action",
        action_space_size=action_space_size,
    )
    action_summary["source"] = f"{variant['variant_id']}_scalar_policy_decisions"
    joint_action_summary = _joint_action_trace_observability(
        joint_actions,
        action_space_size=action_space_size,
    )
    joint_action_summary["source"] = f"{variant['variant_id']}_joint_actions_physical_commits"
    return {
        "variant_id": variant["variant_id"],
        "label": variant["label"],
        "policy_mode": variant["policy_mode"],
        "temperature": variant["temperature"],
        "epsilon": variant["epsilon"],
        "compatibility_role": variant["compatibility_role"],
        "ok": failure is None,
        "gif_filename": variant["gif_filename"],
        "gif_ref": runs.file_ref(gif_path, mount=RUNS_MOUNT),
        "raw_frames_ref": runs.file_ref(frames_path, mount=RUNS_MOUNT),
        "telemetry_ref": (
            runs.file_ref(telemetry_path, mount=RUNS_MOUNT) if telemetry_path.exists() else None
        ),
        "frame_count": int(raw_frames.shape[0]),
        "physical_steps": int(physical_steps),
        "scalar_steps": int(scalar_steps),
        "done": bool(done),
        "terminal_reason": last_info.get("terminal_reason"),
        "opponent_mixture_enabled": bool(last_info.get("opponent_mixture_enabled")),
        "opponent_mixture_entry_name": last_info.get("opponent_mixture_entry_name"),
        "opponent_mixture_entry_weight": last_info.get("opponent_mixture_entry_weight"),
        "opponent_mixture_entry_index": last_info.get("opponent_mixture_entry_index"),
        "opponent_mixture_age_label": last_info.get("opponent_mixture_age_label"),
        "stop_reason": stop_reason,
        "max_steps": max_step_limit,
        "step_limit_kind": step_limit_kind,
        "configured_source_max_steps": int(source_max_steps),
        "effective_source_max_steps": int(effective_source_max_steps),
        "decision_ms": float(decision_ms),
        "decision_source_frames": int(decision_source_frames),
        "source_physics_step_ms": float(source_physics_step_ms),
        "source_max_steps_semantics": str(source_max_steps_semantics),
        "frame_stride": int(frame_stride),
        "fps": float(fps),
        "scale": 1,
        "frame_size": int(frame_size),
        "frame_source": "source_state_rgb_canvas_like",
        "frame_schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
        "frame_truth_level": SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL,
        "browser_pixel_fidelity": False,
        "frame_capture_method": frame_capture_method,
        "frame_capture_method_counts": frame_capture_method_counts,
        "frame_capture_details_sample": frame_captures[:4],
        "action_summary": action_summary,
        "greedy_action_collapse_warning": action_summary["action_collapse_warning"],
        "greedy_action_collapse_players": action_summary["action_collapse_players"],
        "joint_action_summary": joint_action_summary,
        "action_trace": {
            "scalar_actions": scalar_action_trace,
            "joint_actions": joint_action_trace,
        },
        "failure": failure,
        "surface": _to_plain(surface),
        "training_opponent_runtime_mode": opponent_runtime_mode,
        "artifacts": {
            "gif": {**runs.file_summary(gif_path, mount=RUNS_MOUNT), **gif_artifact},
            "raw_frames": frames_artifact,
        },
    }


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
    decision_ms: float = DEFAULT_DECISION_MS,
    decision_source_frames: int = DEFAULT_DECISION_SOURCE_FRAMES,
    source_physics_step_ms: float = DEFAULT_SOURCE_PHYSICS_STEP_MS,
    source_max_steps_semantics: str = "source_physics_steps",
    num_simulations: int,
    batch_size: int,
    frame_stride: int,
    fps: float,
    scale: int,
    frame_size: int,
    training_env_variant: str,
    training_reward_variant: str,
    training_reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
    opponent_death_mode: str = DEFAULT_OPPONENT_DEATH_MODE,
    opponent_runtime_mode: str = DEFAULT_OPPONENT_RUNTIME_MODE,
    opponent_mixture_spec: Any | None = None,
    natural_bonus_spawn: bool = TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN,
    collect_temperature: float = DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE,
    collect_epsilon: float = DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON,
) -> dict[str, Any]:
    if max_steps is not None and int(max_steps) < 0:
        raise ValueError("background_gif_max_steps must be non-negative; 0 means no GIF step cap")
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
    requested_frame_size = int(frame_size)
    frame_size = int(CHECKPOINT_SELFPLAY_GIF_FRAME_SIZE)
    effective_source_max_steps = max(
        int(source_max_steps),
        int(max_step_limit)
        if max_step_limit is not None
        else int(DEFAULT_BACKGROUND_EVAL_MAX_STEPS),
    )
    if collect_temperature <= 0.0:
        raise ValueError("background GIF collect temperature must be positive")
    if not 0.0 <= collect_epsilon <= 1.0:
        raise ValueError("background GIF collect epsilon must be in [0, 1]")
    opponent_mixture = _resolve_opponent_mixture_for_env(
        opponent_mixture_spec=opponent_mixture_spec
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

    try:
        try:
            _safe_reload_runs_volume(reason="checkpoint selfplay gif startup")
        except Exception:
            pass
        from curvyzero.infra.modal import lightzero_curvytron_visual_survival_eval as eval_mod

        checkpoint_path = _wait_for_visible_checkpoint(checkpoint_ref)
        payload = eval_mod._torch_load(checkpoint_path)
        found = eval_mod._find_state_dict(payload)
        if found is None:
            raise ValueError("checkpoint payload did not contain a LightZero state dict")
        found_key, state_dict = found
        variants: dict[str, dict[str, Any]] = {}
        capture_surface = _checkpoint_gif_capture_surface(
            training_env_variant=training_env_variant,
            training_reward_variant=training_reward_variant,
            opponent_runtime_mode=opponent_runtime_mode,
            opponent_mixture=opponent_mixture,
        )
        for variant in _checkpoint_gif_variant_specs(
            collect_temperature=float(collect_temperature),
            collect_epsilon=float(collect_epsilon),
        ):
            result = _capture_checkpoint_selfplay_gif_variant(
                eval_mod=eval_mod,
                state_dict=state_dict,
                seed=int(seed),
                checkpoint_ref=checkpoint_ref,
                clean_checkpoint_label=clean_checkpoint_label,
                artifact_root=artifact_root,
                variant=variant,
                source_max_steps=int(source_max_steps),
                effective_source_max_steps=int(effective_source_max_steps),
                decision_ms=float(decision_ms),
                decision_source_frames=int(decision_source_frames),
                source_physics_step_ms=float(source_physics_step_ms),
                source_max_steps_semantics=str(source_max_steps_semantics),
                max_step_limit=max_step_limit,
                step_limit_kind=step_limit_kind,
                num_simulations=int(num_simulations),
                batch_size=int(batch_size),
                frame_stride=int(frame_stride),
                fps=float(fps),
                scale=int(scale),
                frame_size=int(frame_size),
                training_env_variant=training_env_variant,
                training_reward_variant=training_reward_variant,
                training_reward_outcome_alpha=float(training_reward_outcome_alpha),
                opponent_death_mode=opponent_death_mode,
                opponent_runtime_mode=opponent_runtime_mode,
                opponent_mixture=opponent_mixture,
                natural_bonus_spawn=bool(natural_bonus_spawn),
            )
            variants[str(result["variant_id"])] = result

        legacy = variants[BACKGROUND_GIF_VARIANT_EVAL_GREEDY]
        collect = variants[BACKGROUND_GIF_VARIANT_COLLECT_T1]
        legacy_action_summary = dict(legacy["action_summary"])
        legacy_action_summary["greedy_action_collapse_warning"] = legacy_action_summary[
            "action_collapse_warning"
        ]
        legacy_action_summary["greedy_action_collapse_players"] = legacy_action_summary[
            "action_collapse_players"
        ]
        gif_variants = {
            key: {
                item_key: value
                for item_key, value in item.items()
                if item_key not in {"action_trace", "surface"}
            }
            for key, item in variants.items()
        }
        summary = {
            "ok": all(bool(item.get("ok")) for item in variants.values()),
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
            "training_reward_outcome_alpha": float(training_reward_outcome_alpha),
            "opponent_death_mode": opponent_death_mode,
            "opponent_runtime_mode": opponent_runtime_mode,
            "opponent_mixture_enabled": opponent_mixture is not None,
            "opponent_mixture": _to_plain(opponent_mixture),
            "natural_bonus_spawn": bool(natural_bonus_spawn),
            "capture_env_variant": capture_surface["env_variant"],
            "capture_env_reason": capture_surface["reason"],
            "frame_source": "source_state_rgb_canvas_like",
            "frame_schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
            "frame_truth_level": SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL,
            "browser_pixel_fidelity": False,
            "frame_capture_method": legacy.get("frame_capture_method"),
            "frame_capture_method_counts": legacy.get("frame_capture_method_counts"),
            "frame_capture_details_sample": legacy.get("frame_capture_details_sample"),
            "raw_frame_source": "source_state_rgb_canvas_like",
            "raw_frame_is_browser_pixel": False,
            "raw_frame_shape": [int(frame_size), int(frame_size), 3],
            "saved_frame_shape": [int(frame_size), int(frame_size), 3],
            "requested_frame_size": int(requested_frame_size),
            "effective_frame_size": int(frame_size),
            "frame_size_policy": (
                "checkpoint_selfplay_gif_always_uses_full_source_state_rgb_canvas"
            ),
            "gif_filename": "raw.gif",
            "gif_content_kind": "source_state_rgb_canvas_like_rgb_frames",
            "gif_filename_role": "legacy_selfplay_artifact_name",
            "gif_filename_note": (
                "raw.gif is the backward-compatible greedy/eval artifact. "
                "collect_t1.gif is the training-like collect-mode sample."
            ),
            "gif_ref": legacy["gif_ref"],
            "raw_frames_ref": legacy["raw_frames_ref"],
            "telemetry_ref": legacy["telemetry_ref"],
            "frame_count": legacy["frame_count"],
            "physical_steps": legacy["physical_steps"],
            "scalar_steps": legacy["scalar_steps"],
            "greedy_action_collapse_warning": legacy_action_summary[
                "greedy_action_collapse_warning"
            ],
            "greedy_action_summary": legacy_action_summary,
            "joint_action_summary": legacy["joint_action_summary"],
            "seed": int(seed),
            "seed_role": "effective_checkpoint_selfplay_seed",
            "done": legacy["done"],
            "terminal_reason": legacy["terminal_reason"],
            "opponent_mixture_entry_name": legacy.get("opponent_mixture_entry_name"),
            "opponent_mixture_entry_weight": legacy.get("opponent_mixture_entry_weight"),
            "opponent_mixture_entry_index": legacy.get("opponent_mixture_entry_index"),
            "opponent_mixture_age_label": legacy.get("opponent_mixture_age_label"),
            "stop_reason": legacy["stop_reason"],
            "max_steps": max_step_limit,
            "step_limit_kind": step_limit_kind,
            "configured_source_max_steps": int(source_max_steps),
            "effective_source_max_steps": int(effective_source_max_steps),
            "decision_ms": float(decision_ms),
            "decision_source_frames": int(decision_source_frames),
            "source_physics_step_ms": float(source_physics_step_ms),
            "source_max_steps_semantics": str(source_max_steps_semantics),
            "frame_stride": int(frame_stride),
            "fps": float(fps),
            "scale": 1,
            "frame_size": int(frame_size),
            "num_simulations": int(num_simulations),
            "batch_size": int(batch_size),
            "surface": legacy["surface"],
            "action_trace": legacy["action_trace"],
            "failure": legacy["failure"] or collect["failure"],
            "gif_variants_schema_id": "curvyzero_lightzero_curvytron_selfplay_gif_variants/v0",
            "default_gif_variant": BACKGROUND_GIF_VARIANT_EVAL_GREEDY,
            "compatibility_gif_variant": BACKGROUND_GIF_VARIANT_EVAL_GREEDY,
            "gif_variants": gif_variants,
            "variant_action_traces": {key: item["action_trace"] for key, item in variants.items()},
            "variant_surfaces": {key: item["surface"] for key, item in variants.items()},
            "artifacts": {
                "gif": legacy["artifacts"]["gif"],
                "raw_frames": legacy["artifacts"]["raw_frames"],
                "gif_variants": {key: item["artifacts"] for key, item in variants.items()},
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
            "opponent_mixture_enabled": opponent_mixture_spec is not None,
            "opponent_mixture": _to_plain(opponent_mixture_spec),
            "natural_bonus_spawn": bool(natural_bonus_spawn),
            "capture_env_variant": ENV_VARIANT_SOURCE_STATE_TURN_COMMIT,
            "frame_source": "source_state_rgb_canvas_like",
            "frame_schema_id": SOURCE_STATE_RGB_CANVAS_LIKE_SCHEMA_ID,
            "frame_truth_level": SOURCE_STATE_RGB_CANVAS_LIKE_TRUTH_LEVEL,
            "browser_pixel_fidelity": False,
            "raw_frame_source": "source_state_rgb_canvas_like",
            "seed": int(seed),
            "seed_role": "effective_checkpoint_selfplay_seed",
            "requested_frame_size": int(requested_frame_size),
            "effective_frame_size": int(frame_size),
            "frame_size": int(frame_size),
            "frame_size_policy": (
                "checkpoint_selfplay_gif_always_uses_full_source_state_rgb_canvas"
            ),
            "max_steps": max_step_limit,
            "step_limit_kind": step_limit_kind,
            "configured_source_max_steps": int(source_max_steps),
            "effective_source_max_steps": int(effective_source_max_steps),
            "decision_ms": float(decision_ms),
            "decision_source_frames": int(decision_source_frames),
            "source_physics_step_ms": float(source_physics_step_ms),
            "source_max_steps_semantics": str(source_max_steps_semantics),
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
    gif_browser_run_marker = _write_gif_browser_run_marker(
        run_id=run_id,
        created_at=started_at,
    )
    summary["summary_ref"] = runs.file_ref(summary_path, mount=RUNS_MOUNT)
    summary["gif_browser_run_marker"] = gif_browser_run_marker
    runs.write_json(summary_path, _to_plain(summary))
    summary["summary"] = runs.file_summary(summary_path, mount=RUNS_MOUNT)
    _commit_runs_volume_with_backoff(
        label="checkpoint_selfplay_gif",
        initial_jitter_sec=8.0,
    )
    print(json.dumps(_to_plain(summary), indent=2, sort_keys=True))
    return _to_plain(summary)


_POLICY_OBSERVATION_GPU_PROFILE_INT_FIELDS = frozenset(
    {
        "visual_trail_capacity",
        "visual_trail_active_count",
        "visual_trail_last_active_exclusive",
        "visual_trail_inactive_prefix_slots",
        "min_render_trail_slots",
        "render_trail_slots",
        "render_trail_prefix_headroom_slots",
        "cache_misses",
        "cache_size",
    }
)
_POLICY_OBSERVATION_GPU_PROFILE_SUM_FIELDS = (
    "compact_sec",
    "device_put_sec",
    "render_total_sec",
    "readback_total_sec",
    "total_sec",
    "cache_misses",
)
_POLICY_OBSERVATION_GPU_PROFILE_MAX_FIELDS = (
    "visual_trail_capacity",
    "visual_trail_active_count",
    "visual_trail_last_active_exclusive",
    "visual_trail_inactive_prefix_slots",
    "visual_trail_active_prefix_fill_ratio",
    "min_render_trail_slots",
    "render_trail_slots",
    "render_trail_prefix_headroom_slots",
    "render_trail_slot_utilization",
    "render_trail_capacity_ratio",
    "compact_sec",
    "device_put_sec",
    "render_total_sec",
    "readback_total_sec",
    "total_sec",
    "cache_misses",
    "cache_size",
)
_POLICY_OBSERVATION_GPU_PROFILE_MIN_FIELDS = (
    "visual_trail_active_prefix_fill_ratio",
    "render_trail_slot_utilization",
    "render_trail_capacity_ratio",
)
_POLICY_OBSERVATION_GPU_PROFILE_FIELDS = tuple(
    dict.fromkeys(
        (
            *_POLICY_OBSERVATION_GPU_PROFILE_MAX_FIELDS,
            *_POLICY_OBSERVATION_GPU_PROFILE_SUM_FIELDS,
        )
    )
)


def _telemetry_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _policy_observation_gpu_profile_output_value(key: str, value: float) -> int | float:
    if key in _POLICY_OBSERVATION_GPU_PROFILE_INT_FIELDS:
        return int(value)
    return float(value)


def _compact_policy_observation_gpu_profile(profile: Any) -> dict[str, int | float]:
    if not isinstance(profile, dict):
        return {}
    compact: dict[str, int | float] = {}
    for key in _POLICY_OBSERVATION_GPU_PROFILE_FIELDS:
        value = _telemetry_float(profile.get(key))
        if value is None:
            continue
        compact[key] = _policy_observation_gpu_profile_output_value(key, value)
    active_count = _telemetry_float(compact.get("visual_trail_active_count"))
    active_prefix = _telemetry_float(compact.get("visual_trail_last_active_exclusive"))
    render_slots = _telemetry_float(compact.get("render_trail_slots"))
    capacity = _telemetry_float(compact.get("visual_trail_capacity"))
    if active_count is not None and active_prefix is not None:
        compact["visual_trail_inactive_prefix_slots"] = int(max(0.0, active_prefix - active_count))
        if active_prefix > 0.0:
            compact["visual_trail_active_prefix_fill_ratio"] = float(active_count / active_prefix)
    if active_prefix is not None and render_slots is not None:
        compact["render_trail_prefix_headroom_slots"] = int(max(0.0, render_slots - active_prefix))
    if active_count is not None and render_slots is not None and render_slots > 0.0:
        compact["render_trail_slot_utilization"] = float(active_count / render_slots)
    if capacity is not None and render_slots is not None and capacity > 0.0:
        compact["render_trail_capacity_ratio"] = float(render_slots / capacity)
    return compact


def _summarize_policy_observation_gpu_profiles(
    *,
    profile_count: int,
    profile_sums: Counter[str],
    profile_counts: Counter[str],
    profile_max: dict[str, float],
    profile_min: dict[str, float],
    last_profile: dict[str, int | float],
    scope: str,
    inactive_prefix_row_count: int,
    capacity_reduction_row_count: int,
    cache_miss_row_count: int,
) -> dict[str, Any] | None:
    if profile_count <= 0:
        return None
    summary: dict[str, Any] = {
        "scope": scope,
        "sampled_profile_count": int(profile_count),
        "sampled_inactive_prefix_row_count": int(inactive_prefix_row_count),
        "sampled_capacity_reduction_row_count": int(capacity_reduction_row_count),
        "sampled_cache_miss_row_count": int(cache_miss_row_count),
        **last_profile,
    }
    sampled_sum = {
        key: _policy_observation_gpu_profile_output_value(key, profile_sums[key])
        for key in _POLICY_OBSERVATION_GPU_PROFILE_SUM_FIELDS
        if profile_counts[key]
    }
    if sampled_sum:
        summary["sampled_sum"] = sampled_sum
    sampled_mean = {
        key: float(profile_sums[key]) / float(profile_counts[key])
        for key in _POLICY_OBSERVATION_GPU_PROFILE_SUM_FIELDS
        if profile_counts[key]
    }
    if sampled_mean:
        summary["sampled_mean"] = sampled_mean
    sampled_max = {
        key: _policy_observation_gpu_profile_output_value(key, profile_max[key])
        for key in _POLICY_OBSERVATION_GPU_PROFILE_MAX_FIELDS
        if key in profile_max
    }
    if sampled_max:
        summary["sampled_max"] = sampled_max
    sampled_min = {
        key: _policy_observation_gpu_profile_output_value(key, profile_min[key])
        for key in _POLICY_OBSERVATION_GPU_PROFILE_MIN_FIELDS
        if key in profile_min
    }
    if sampled_min:
        summary["sampled_min"] = sampled_min
    return summary


def _summarize_env_step_telemetry(path: Path) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    physical_action_counts: Counter[str] = Counter()
    opponent_counts: Counter[str] = Counter()
    acting_player_counts: Counter[str] = Counter()
    terminal_reasons: Counter[str] = Counter()
    profile_env_timing_sums: Counter[str] = Counter()
    profile_env_timing_counts: Counter[str] = Counter()
    policy_observation_gpu_profile_sums: Counter[str] = Counter()
    policy_observation_gpu_profile_counts: Counter[str] = Counter()
    policy_observation_gpu_profile_max: dict[str, float] = {}
    policy_observation_gpu_profile_min: dict[str, float] = {}
    policy_observation_gpu_profile_count = 0
    policy_observation_gpu_inactive_prefix_row_count = 0
    policy_observation_gpu_capacity_reduction_row_count = 0
    policy_observation_gpu_cache_miss_row_count = 0
    policy_observation_gpu_last_profile: dict[str, int | float] = {}
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
        profile_env_timing = row.get("profile_env_timing_sec")
        if isinstance(profile_env_timing, dict):
            for key, value in profile_env_timing.items():
                try:
                    seconds = float(value)
                except (TypeError, ValueError):
                    continue
                profile_env_timing_sums[str(key)] += seconds
                profile_env_timing_counts[str(key)] += 1
        raw_policy_observation_gpu_profile = row.get("policy_observation_gpu_last_profile")
        policy_observation_gpu_profile = _compact_policy_observation_gpu_profile(
            raw_policy_observation_gpu_profile
        )
        if policy_observation_gpu_profile:
            policy_observation_gpu_profile_count += 1
            policy_observation_gpu_last_profile = policy_observation_gpu_profile
            if (
                _telemetry_float(
                    policy_observation_gpu_profile.get("visual_trail_inactive_prefix_slots")
                )
                or 0.0
            ) > 0.0:
                policy_observation_gpu_inactive_prefix_row_count += 1
            render_trail_capacity_ratio = _telemetry_float(
                policy_observation_gpu_profile.get("render_trail_capacity_ratio")
            )
            render_slots_reduced = (
                isinstance(raw_policy_observation_gpu_profile, dict)
                and bool(
                    raw_policy_observation_gpu_profile.get(
                        "render_trail_slots_reduced_from_capacity"
                    )
                )
            ) or (render_trail_capacity_ratio is not None and render_trail_capacity_ratio < 1.0)
            if render_slots_reduced:
                policy_observation_gpu_capacity_reduction_row_count += 1
            if (_telemetry_float(policy_observation_gpu_profile.get("cache_misses")) or 0.0) > 0.0:
                policy_observation_gpu_cache_miss_row_count += 1
            for key, value in policy_observation_gpu_profile.items():
                numeric_value = float(value)
                policy_observation_gpu_profile_sums[key] += numeric_value
                policy_observation_gpu_profile_counts[key] += 1
                if (
                    key not in policy_observation_gpu_profile_max
                    or numeric_value > policy_observation_gpu_profile_max[key]
                ):
                    policy_observation_gpu_profile_max[key] = numeric_value
                if (
                    key not in policy_observation_gpu_profile_min
                    or numeric_value < policy_observation_gpu_profile_min[key]
                ):
                    policy_observation_gpu_profile_min[key] = numeric_value
        trainer_reward_sum += float(row.get("reward") or 0.0)
        done_count += int(bool(row.get("done", False)))
    for action_id in ("0", "1", "2"):
        action_counts.setdefault(action_id, 0)
        physical_action_counts.setdefault(action_id, 0)
        opponent_counts.setdefault(action_id, 0)
    row_count = sum(action_counts.values())
    telemetry_scope = "sampled_telemetry_rows" if telemetry_sampled else "all_telemetry_rows"
    summary = {
        "status": "ok",
        "path": str(path),
        "row_count": int(row_count),
        "sampled_row_count": int(row_count),
        "counts_scope": telemetry_scope,
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
        "profile_env_timing_sec": {
            "scope": telemetry_scope,
            "sampled_sum": dict(sorted(profile_env_timing_sums.items())),
            "sampled_count": dict(sorted(profile_env_timing_counts.items())),
            "sampled_mean": {
                key: (float(profile_env_timing_sums[key]) / float(profile_env_timing_counts[key]))
                for key in sorted(profile_env_timing_sums)
                if profile_env_timing_counts[key]
            },
        },
        "terminal_reasons": dict(sorted(terminal_reasons.items())),
        "observed_fields": observed_fields,
        "first_rows": rows,
        "collapse_warning": _action_collapse_warning(action_counts),
        "physical_action_collapse_warning": _action_collapse_warning(physical_action_counts),
    }
    policy_observation_gpu_summary = _summarize_policy_observation_gpu_profiles(
        profile_count=policy_observation_gpu_profile_count,
        profile_sums=policy_observation_gpu_profile_sums,
        profile_counts=policy_observation_gpu_profile_counts,
        profile_max=policy_observation_gpu_profile_max,
        profile_min=policy_observation_gpu_profile_min,
        last_profile=policy_observation_gpu_last_profile,
        scope=telemetry_scope,
        inactive_prefix_row_count=policy_observation_gpu_inactive_prefix_row_count,
        capacity_reduction_row_count=policy_observation_gpu_capacity_reduction_row_count,
        cache_miss_row_count=policy_observation_gpu_cache_miss_row_count,
    )
    if policy_observation_gpu_summary is not None:
        summary["policy_observation_gpu_last_profile"] = policy_observation_gpu_summary
    return summary


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
    runs.write_json(
        path, runs.run_manifest(task_id=TASK_ID, run_id=run_id, config=config), exclusive=True
    )
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
        entropy[player_key] = round(player_entropy, 6) if player_entropy is not None else None

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
        "summary_ref": (
            runs.attempt_train_ref(TASK_ID, run_id, attempt_id) / "summary.json"
        ).as_posix(),
        "checkpoint_root_ref": (
            runs.checkpoints_root_ref(TASK_ID, run_id) / "lightzero"
        ).as_posix(),
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
    return lz_config.set_or_add_path(mapping, path, value)


def _get_path(mapping: Any, path: tuple[str, ...], default: Any = None) -> Any:
    return lz_config.get_path(mapping, path, default)


def _cfg_get(mapping: Any, key: str, default: Any) -> Any:
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _runs_volume_commit_callback() -> None:
    _commit_runs_volume_with_backoff(label="training_progress_commit")


def _two_seat_checkpoint_ref(run_id: str) -> Path:
    return runs.checkpoints_root_ref(TASK_ID, run_id) / "lightzero"


def _reject_mutable_frozen_opponent_checkpoint_ref(ref: str | None) -> None:
    if ref is None:
        return
    name = Path(str(ref)).name
    if name in {"latest.pth.tar", "ckpt_best.pth.tar"}:
        raise ValueError(
            "frozen opponent checkpoint refs must be immutable exact "
            f"iteration_*.pth.tar files; got mutable ref {ref!r}"
        )
    if lz_checkpoints.lightzero_iteration_from_checkpoint_name(name) is None:
        raise ValueError(
            "frozen opponent checkpoint refs must be immutable exact "
            f"iteration_N.pth.tar files; got {ref!r}"
        )


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
    background_gif_collect_temperature: float | None = None,
    background_gif_collect_epsilon: float | None = None,
) -> dict[str, Any]:
    training_collect_temperature = float(
        payload.get("collect_temperature", DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE)
    )
    training_collect_epsilon = float(
        payload.get("collect_epsilon", DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON)
    )
    gif_collect_temperature_source = (
        BACKGROUND_GIF_COLLECT_SETTINGS_SOURCE_TRAINING
        if background_gif_collect_temperature is None
        else BACKGROUND_GIF_COLLECT_SETTINGS_SOURCE_OVERRIDE
    )
    gif_collect_epsilon_source = (
        BACKGROUND_GIF_COLLECT_SETTINGS_SOURCE_TRAINING
        if background_gif_collect_epsilon is None
        else BACKGROUND_GIF_COLLECT_SETTINGS_SOURCE_OVERRIDE
    )
    gif_collect_temperature = (
        training_collect_temperature
        if background_gif_collect_temperature is None
        else float(background_gif_collect_temperature)
    )
    gif_collect_epsilon = (
        training_collect_epsilon
        if background_gif_collect_epsilon is None
        else float(background_gif_collect_epsilon)
    )
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
            "requested_frame_size": int(background_gif_frame_size),
            "frame_size": CHECKPOINT_SELFPLAY_GIF_FRAME_SIZE,
            "frame_size_policy": (
                "checkpoint_selfplay_gif_always_uses_full_source_state_rgb_canvas"
            ),
            "collect_temperature": float(gif_collect_temperature),
            "collect_epsilon": float(gif_collect_epsilon),
            "collect_temperature_source": gif_collect_temperature_source,
            "collect_epsilon_source": gif_collect_epsilon_source,
            "training_collect_temperature": float(training_collect_temperature),
            "training_collect_epsilon": float(training_collect_epsilon),
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
        background_gif_collect_temperature=float(
            background["selfplay_gif"].get(
                "collect_temperature",
                DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE,
            )
        ),
        background_gif_collect_epsilon=float(
            background["selfplay_gif"].get(
                "collect_epsilon",
                DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON,
            )
        ),
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
    payload = dict(payload)
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
    frozen_opponent_probability = float(
        payload.get(
            "frozen_opponent_probability",
            TWO_SEAT_DEFAULT_FROZEN_OPPONENT_PROBABILITY,
        )
    )
    if frozen_opponent_probability > 0.0:
        frozen_checkpoint_path_text = payload.get("frozen_opponent_checkpoint_path")
        frozen_checkpoint_ref = payload.get("frozen_opponent_checkpoint_ref")
        _reject_mutable_frozen_opponent_checkpoint_ref(
            str(frozen_checkpoint_ref or frozen_checkpoint_path_text)
            if frozen_checkpoint_ref or frozen_checkpoint_path_text
            else None
        )
        if frozen_checkpoint_path_text:
            frozen_checkpoint_path = Path(str(frozen_checkpoint_path_text))
            frozen_checkpoint_resolution = {
                "source_kind": "path_from_payload",
                "input": str(frozen_checkpoint_path_text),
                "resolved_checkpoint_path": str(frozen_checkpoint_path),
            }
        elif frozen_checkpoint_ref:
            frozen_checkpoint_path, frozen_checkpoint_resolution = runs.resolve_mounted_ref_or_path(
                str(frozen_checkpoint_ref),
                mount=RUNS_MOUNT,
                remote_root=REMOTE_ROOT,
            )
        else:
            raise ValueError(
                "frozen_opponent_checkpoint_ref is required when frozen_opponent_probability > 0"
            )
        if not frozen_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"two-seat frozen opponent checkpoint file not found: {frozen_checkpoint_path}"
            )
        frozen_checkpoint_report_ref = str(
            frozen_checkpoint_resolution.get("source_ref")
            or frozen_checkpoint_ref
            or frozen_checkpoint_path_text
        )
        payload["frozen_opponent_checkpoint_path"] = str(frozen_checkpoint_path)
        payload["frozen_opponent_checkpoint_ref"] = frozen_checkpoint_report_ref
        payload["frozen_opponent_checkpoint_resolution"] = {
            **frozen_checkpoint_resolution,
            "input": str(frozen_checkpoint_ref or frozen_checkpoint_path_text),
            "resolved_checkpoint_path": str(frozen_checkpoint_path),
            "file": runs.file_summary_any_mount(
                frozen_checkpoint_path,
                mount=RUNS_MOUNT,
            ),
        }
    command = {
        "schema_id": "curvyzero_experimental_two_seat_adapter_command/v0",
        "mode": TWO_SEAT_SELFPLAY_MODE,
        "canonical_launcher": (
            "curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train"
        ),
        "launcher_status": "experimental_two_seat_adapter",
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
    _commit_runs_volume_with_backoff(label="two_seat_attempt_start_commit")

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
        verify_model_update_hash=bool(payload.get("verify_model_update_hash", False)),
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
        frozen_opponent_probability=frozen_opponent_probability,
        frozen_opponent_checkpoint_path=payload.get("frozen_opponent_checkpoint_path"),
        frozen_opponent_checkpoint_ref=payload.get("frozen_opponent_checkpoint_ref"),
        frozen_opponent_snapshot_ref=payload.get("frozen_opponent_snapshot_ref"),
        frozen_opponent_checkpoint_state_key=payload.get("frozen_opponent_checkpoint_state_key"),
        frozen_opponent_player_id=int(
            payload.get(
                "frozen_opponent_player_id",
                TWO_SEAT_DEFAULT_FROZEN_OPPONENT_PLAYER_ID,
            )
        ),
        frozen_opponent_num_simulations=payload.get("frozen_opponent_num_simulations"),
        frozen_opponent_use_cuda=payload.get("frozen_opponent_use_cuda"),
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
            "verify_model_update_hash": bool(payload.get("verify_model_update_hash", False)),
            "trail_render_mode": payload["trail_render_mode"],
            "death_mode": payload["death_mode"],
            "natural_bonus_spawn": natural_bonus_spawn,
            "alive_reward": payload["alive_reward"],
            "dead_reward": payload["dead_reward"],
            "terminal_outcome_reward_per_step": terminal_outcome_reward_per_step,
            "bonus_pickup_reward_per_catch": bonus_pickup_reward_per_catch,
            "return_target_discount": return_target_discount,
            "learning_rate": learning_rate,
            "frozen_opponent_probability": frozen_opponent_probability,
            "frozen_opponent_checkpoint_ref": payload.get("frozen_opponent_checkpoint_ref"),
            "frozen_opponent_snapshot_ref": payload.get("frozen_opponent_snapshot_ref"),
            "frozen_opponent_player_id": payload.get(
                "frozen_opponent_player_id",
                TWO_SEAT_DEFAULT_FROZEN_OPPONENT_PLAYER_ID,
            ),
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
    _commit_runs_volume_with_backoff(label="two_seat_attempt_final_commit")
    return _to_plain(result)


@app.function(
    image=image,
    volumes=TRAINER_VOLUMES,
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
    volumes=TRAINER_VOLUMES,
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
    volumes=TRAINER_VOLUMES,
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
    return lz_config.to_plain(value)


@app.function(image=image, volumes=TRAINER_VOLUMES, timeout=40 * 60, cpu=2.0)
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
    decision_ms: float = DEFAULT_DECISION_MS,
    decision_source_frames: int = DEFAULT_DECISION_SOURCE_FRAMES,
    source_physics_step_ms: float = DEFAULT_SOURCE_PHYSICS_STEP_MS,
    source_max_steps_semantics: str = "source_physics_steps",
    num_simulations: int = DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BACKGROUND_EVAL_BATCH_SIZE,
    env_variant: str = DEFAULT_ENV_VARIANT,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
    model_reward_variant: str | None = None,
    opponent_policy_kind: str = DEFAULT_OPPONENT_POLICY_KIND,
    opponent_checkpoint_ref: str | None = None,
    opponent_snapshot_ref: str | None = None,
    opponent_checkpoint_state_key: str | None = None,
    opponent_mixture_spec: Any | None = None,
    opponent_assignment_ref: str | None = None,
    opponent_death_mode: str = DEFAULT_OPPONENT_DEATH_MODE,
    opponent_runtime_mode: str = DEFAULT_OPPONENT_RUNTIME_MODE,
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
        decision_ms=decision_ms,
        decision_source_frames=decision_source_frames,
        source_physics_step_ms=source_physics_step_ms,
        source_max_steps_semantics=source_max_steps_semantics,
        num_simulations=num_simulations,
        batch_size=batch_size,
        env_variant=env_variant,
        reward_variant=reward_variant,
        reward_outcome_alpha=reward_outcome_alpha,
        model_reward_variant=model_reward_variant,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
        opponent_snapshot_ref=opponent_snapshot_ref,
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
        opponent_mixture_spec=opponent_mixture_spec,
        opponent_assignment_ref=opponent_assignment_ref,
        opponent_death_mode=opponent_death_mode,
        opponent_runtime_mode=opponent_runtime_mode,
        natural_bonus_spawn=bool(natural_bonus_spawn),
    )


@app.function(image=image, volumes=TRAINER_VOLUMES, timeout=40 * 60, cpu=2.0)
def lightzero_curvytron_visual_survival_checkpoint_selfplay_gif(
    checkpoint_ref: str,
    checkpoint_label: str | None = None,
    eval_id: str = DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    seed: int = DEFAULT_SEED + DEFAULT_BACKGROUND_GIF_SEED_OFFSET,
    max_steps: int = DEFAULT_BACKGROUND_GIF_MAX_STEPS,
    source_max_steps: int = DEFAULT_SOURCE_MAX_STEPS,
    decision_ms: float = DEFAULT_DECISION_MS,
    decision_source_frames: int = DEFAULT_DECISION_SOURCE_FRAMES,
    source_physics_step_ms: float = DEFAULT_SOURCE_PHYSICS_STEP_MS,
    source_max_steps_semantics: str = "source_physics_steps",
    num_simulations: int = DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BACKGROUND_EVAL_BATCH_SIZE,
    frame_stride: int = DEFAULT_BACKGROUND_GIF_FRAME_STRIDE,
    fps: float = DEFAULT_BACKGROUND_GIF_FPS,
    scale: int = DEFAULT_BACKGROUND_GIF_SCALE,
    frame_size: int = DEFAULT_BACKGROUND_GIF_FRAME_SIZE,
    training_env_variant: str = DEFAULT_ENV_VARIANT,
    training_reward_variant: str = DEFAULT_REWARD_VARIANT,
    training_reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
    opponent_death_mode: str = DEFAULT_OPPONENT_DEATH_MODE,
    opponent_runtime_mode: str = DEFAULT_OPPONENT_RUNTIME_MODE,
    opponent_mixture_spec: Any | None = None,
    natural_bonus_spawn: bool = TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN,
    collect_temperature: float = DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE,
    collect_epsilon: float = DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON,
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
        decision_ms=decision_ms,
        decision_source_frames=decision_source_frames,
        source_physics_step_ms=source_physics_step_ms,
        source_max_steps_semantics=source_max_steps_semantics,
        num_simulations=num_simulations,
        batch_size=batch_size,
        frame_stride=frame_stride,
        fps=fps,
        scale=scale,
        frame_size=frame_size,
        training_env_variant=training_env_variant,
        training_reward_variant=training_reward_variant,
        training_reward_outcome_alpha=float(training_reward_outcome_alpha),
        opponent_death_mode=opponent_death_mode,
        opponent_runtime_mode=opponent_runtime_mode,
        opponent_mixture_spec=opponent_mixture_spec,
        natural_bonus_spawn=bool(natural_bonus_spawn),
        collect_temperature=float(collect_temperature),
        collect_epsilon=float(collect_epsilon),
    )


@app.function(
    image=image,
    volumes=TRAINER_VOLUMES,
    timeout=DEFAULT_BACKGROUND_EVAL_POLLER_MODAL_TIMEOUT_SEC,
    cpu=1.0,
)
def lightzero_curvytron_visual_survival_checkpoint_eval_poller(
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    exp_name_ref: str | None = None,
    seed: int = DEFAULT_SEED,
    source_max_steps: int = DEFAULT_SOURCE_MAX_STEPS,
    decision_ms: float = DEFAULT_DECISION_MS,
    decision_source_frames: int = DEFAULT_DECISION_SOURCE_FRAMES,
    source_physics_step_ms: float = DEFAULT_SOURCE_PHYSICS_STEP_MS,
    source_max_steps_semantics: str = "source_physics_steps",
    env_variant: str = DEFAULT_ENV_VARIANT,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
    opponent_policy_kind: str = DEFAULT_OPPONENT_POLICY_KIND,
    opponent_checkpoint_ref: str | None = None,
    opponent_snapshot_ref: str | None = None,
    opponent_checkpoint_state_key: str | None = None,
    opponent_mixture_spec: Any | None = None,
    opponent_assignment_ref: str | None = None,
    opponent_death_mode: str = DEFAULT_OPPONENT_DEATH_MODE,
    opponent_runtime_mode: str = DEFAULT_OPPONENT_RUNTIME_MODE,
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
    background_gif_collect_temperature: float = DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE,
    background_gif_collect_epsilon: float = DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON,
    background_gif_natural_bonus_spawn: bool = TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN,
    poll_interval_sec: float = DEFAULT_BACKGROUND_EVAL_POLL_INTERVAL_SEC,
    stable_polls: int = DEFAULT_BACKGROUND_EVAL_POLL_STABLE_POLLS,
    max_runtime_sec: float = DEFAULT_BACKGROUND_EVAL_POLLER_MAX_RUNTIME_SEC,
    idle_after_train_done_sec: float = DEFAULT_BACKGROUND_EVAL_POLLER_IDLE_AFTER_DONE_SEC,
) -> dict[str, Any]:
    exp_name_ref = (
        exp_name_ref
        or (runs.attempt_train_ref(TASK_ID, run_id, attempt_id) / "lightzero_exp").as_posix()
    )
    command = _checkpoint_eval_poller_command(
        seed=seed,
        source_max_steps=source_max_steps,
        decision_ms=decision_ms,
        decision_source_frames=decision_source_frames,
        source_physics_step_ms=source_physics_step_ms,
        source_max_steps_semantics=source_max_steps_semantics,
        env_variant=env_variant,
        reward_variant=reward_variant,
        reward_outcome_alpha=reward_outcome_alpha,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
        opponent_snapshot_ref=opponent_snapshot_ref,
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
        opponent_mixture_spec=opponent_mixture_spec,
        opponent_assignment_ref=opponent_assignment_ref,
        opponent_death_mode=opponent_death_mode,
        opponent_runtime_mode=opponent_runtime_mode,
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
        background_gif_collect_temperature=background_gif_collect_temperature,
        background_gif_collect_epsilon=background_gif_collect_epsilon,
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


@app.function(image=image, volumes=TRAINER_VOLUMES, timeout=20 * 60, cpu=2.0)
def lightzero_curvytron_write_opponent_assignment_artifacts(
    payload: dict[str, Any],
) -> dict[str, Any]:
    return _write_opponent_assignment_artifacts(
        run_id=str(payload["run_id"]),
        attempt_id=str(payload["attempt_id"]),
        assignment=payload["assignment"],
        audit=payload.get("audit"),
        target_volume=str(payload.get("target_volume") or "runs"),
        mirror_checkpoints_to_control=bool(payload.get("mirror_checkpoints_to_control", False)),
    )


@app.function(image=image, volumes=TRAINER_VOLUMES, timeout=20 * 60, cpu=2.0)
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
    model_support_cap: int | None = DEFAULT_MODEL_SUPPORT_CAP,
    td_steps: int | None = DEFAULT_TD_STEPS,
    lightzero_eval_freq: int = DEFAULT_LIGHTZERO_EVAL_FREQ,
    skip_lightzero_eval_in_profile: bool = DEFAULT_SKIP_LIGHTZERO_EVAL_IN_PROFILE,
    profile_cuda_sync_enabled: bool = DEFAULT_PROFILE_CUDA_SYNC_ENABLED,
    profile_allow_auto_resume: bool = DEFAULT_PROFILE_ALLOW_AUTO_RESUME,
    profile_volume_commit: bool = DEFAULT_PROFILE_VOLUME_COMMIT,
    lightzero_multi_gpu: bool = DEFAULT_LIGHTZERO_MULTI_GPU,
    save_ckpt_after_iter: int = DEFAULT_SAVE_CKPT_AFTER_ITER,
    commit_on_checkpoint: bool = DEFAULT_COMMIT_ON_CHECKPOINT,
    stop_after_learner_train_calls: int = DEFAULT_STOP_AFTER_LEARNER_TRAIN_CALLS,
    env_variant: str = DEFAULT_ENV_VARIANT,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
    source_state_trail_render_mode: str = DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE,
    source_state_bonus_render_mode: str = DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE,
    policy_observation_backend: str = DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE,
    collect_search_backend: str = DEFAULT_COLLECT_SEARCH_BACKEND,
    collect_search_ctree_backend: str = DEFAULT_COLLECT_SEARCH_CTREE_BACKEND,
    learner_seat_mode: str = DEFAULT_LEARNER_SEAT_MODE,
    ego_action_straight_override_probability: float = (
        DEFAULT_EGO_ACTION_STRAIGHT_OVERRIDE_PROBABILITY
    ),
    policy_action_repeat_min: int = DEFAULT_POLICY_ACTION_REPEAT_MIN,
    policy_action_repeat_max: int = DEFAULT_POLICY_ACTION_REPEAT_MAX,
    policy_action_repeat_extra_probability: float = (
        DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
    ),
    control_noise_profile_id: str = DEFAULT_CONTROL_NOISE_PROFILE_ID,
    disable_death_for_profile: bool = DEFAULT_DISABLE_DEATH_FOR_PROFILE,
    opponent_death_mode: str = DEFAULT_OPPONENT_DEATH_MODE,
    opponent_runtime_mode: str = DEFAULT_OPPONENT_RUNTIME_MODE,
    env_telemetry_stride: int = DEFAULT_ENV_TELEMETRY_STRIDE,
    env_manager_type: str = DEFAULT_ENV_MANAGER_TYPE,
    opponent_policy_kind: str = DEFAULT_OPPONENT_POLICY_KIND,
    opponent_use_cuda: bool = DEFAULT_OPPONENT_USE_CUDA,
    opponent_checkpoint_ref: str | None = None,
    opponent_snapshot_ref: str | None = None,
    opponent_checkpoint_report_ref: str | None = None,
    opponent_checkpoint_state_key: str | None = None,
    opponent_mixture_spec: str | None = None,
    opponent_assignment_ref: str | None = None,
    initial_policy_checkpoint_ref: str | None = None,
    initial_policy_checkpoint_state_key: str | None = None,
    initial_policy_checkpoint_load_mode: str = DEFAULT_INITIAL_POLICY_CHECKPOINT_LOAD_MODE,
    opponent_assignment_refresh_interval_train_iter: int = (
        DEFAULT_OPPONENT_ASSIGNMENT_REFRESH_INTERVAL_TRAIN_ITER
    ),
    opponent_assignment_refresh_ref: str | None = None,
    own_checkpoint_opponent_refresh_enabled: bool = (
        DEFAULT_OWN_CHECKPOINT_OPPONENT_REFRESH_ENABLED
    ),
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
    background_gif_collect_temperature: float = DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE,
    background_gif_collect_epsilon: float = DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON,
    exploration_bonus_mode: str = xb.EXPLORATION_BONUS_MODE_NONE,
    exploration_bonus_weight: float = 0.0,
    exploration_bonus_feature_source: str = xb.RND_FEATURE_SOURCE_POLICY_GRAY64_LATEST_V0,
    exploration_bonus_rnd_batch_size: int = 64,
    exploration_bonus_rnd_update_per_collect: int = xb.RND_DEFAULT_UPDATE_PER_COLLECT,
    exploration_bonus_rnd_buffer_size: int = 100_000,
    exploration_bonus_rnd_learning_rate: float = 3e-4,
    exploration_bonus_rnd_weight_decay: float = 1e-4,
    exploration_bonus_rnd_input_norm: bool = False,
    require_rnd_metrics: bool = False,
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
        model_support_cap=model_support_cap,
        td_steps=td_steps,
        lightzero_eval_freq=lightzero_eval_freq,
        skip_lightzero_eval_in_profile=skip_lightzero_eval_in_profile,
        profile_cuda_sync_enabled=profile_cuda_sync_enabled,
        profile_allow_auto_resume=profile_allow_auto_resume,
        profile_volume_commit=profile_volume_commit,
        lightzero_multi_gpu=lightzero_multi_gpu,
        save_ckpt_after_iter=save_ckpt_after_iter,
        commit_on_checkpoint=commit_on_checkpoint,
        stop_after_learner_train_calls=stop_after_learner_train_calls,
        env_variant=env_variant,
        reward_variant=reward_variant,
        reward_outcome_alpha=reward_outcome_alpha,
        source_state_trail_render_mode=source_state_trail_render_mode,
        source_state_bonus_render_mode=source_state_bonus_render_mode,
        policy_observation_backend=policy_observation_backend,
        collect_search_backend=collect_search_backend,
        collect_search_ctree_backend=collect_search_ctree_backend,
        learner_seat_mode=learner_seat_mode,
        ego_action_straight_override_probability=ego_action_straight_override_probability,
        policy_action_repeat_min=policy_action_repeat_min,
        policy_action_repeat_max=policy_action_repeat_max,
        policy_action_repeat_extra_probability=policy_action_repeat_extra_probability,
        control_noise_profile_id=control_noise_profile_id,
        disable_death_for_profile=disable_death_for_profile,
        opponent_death_mode=opponent_death_mode,
        opponent_runtime_mode=opponent_runtime_mode,
        env_telemetry_stride=env_telemetry_stride,
        env_manager_type=env_manager_type,
        opponent_policy_kind=opponent_policy_kind,
        opponent_use_cuda=opponent_use_cuda,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
        opponent_snapshot_ref=opponent_snapshot_ref,
        opponent_checkpoint_report_ref=opponent_checkpoint_report_ref,
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
        opponent_mixture_spec=opponent_mixture_spec,
        opponent_assignment_ref=opponent_assignment_ref,
        initial_policy_checkpoint_ref=initial_policy_checkpoint_ref,
        initial_policy_checkpoint_state_key=initial_policy_checkpoint_state_key,
        initial_policy_checkpoint_load_mode=initial_policy_checkpoint_load_mode,
        opponent_assignment_refresh_interval_train_iter=(
            opponent_assignment_refresh_interval_train_iter
        ),
        opponent_assignment_refresh_ref=opponent_assignment_refresh_ref,
        own_checkpoint_opponent_refresh_enabled=own_checkpoint_opponent_refresh_enabled,
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
        background_gif_collect_temperature=background_gif_collect_temperature,
        background_gif_collect_epsilon=background_gif_collect_epsilon,
        exploration_bonus_mode=exploration_bonus_mode,
        exploration_bonus_weight=exploration_bonus_weight,
        exploration_bonus_feature_source=exploration_bonus_feature_source,
        exploration_bonus_rnd_batch_size=exploration_bonus_rnd_batch_size,
        exploration_bonus_rnd_update_per_collect=exploration_bonus_rnd_update_per_collect,
        exploration_bonus_rnd_buffer_size=exploration_bonus_rnd_buffer_size,
        exploration_bonus_rnd_learning_rate=exploration_bonus_rnd_learning_rate,
        exploration_bonus_rnd_weight_decay=exploration_bonus_rnd_weight_decay,
        exploration_bonus_rnd_input_norm=exploration_bonus_rnd_input_norm,
        require_rnd_metrics=require_rnd_metrics,
    )


def _apply_visual_survival_train_default_kwargs(kwargs: dict[str, Any]) -> None:
    defaults: dict[str, Any] = {
        "mode": DEFAULT_MODE,
        "seed": DEFAULT_SEED,
        "run_id": DEFAULT_RUN_ID,
        "attempt_id": DEFAULT_ATTEMPT_ID,
        "max_env_step": DEFAULT_MAX_ENV_STEP,
        "max_train_iter": DEFAULT_MAX_TRAIN_ITER,
        "source_max_steps": DEFAULT_SOURCE_MAX_STEPS,
        "decision_ms": DEFAULT_DECISION_MS,
        "collector_env_num": DEFAULT_COLLECTOR_ENV_NUM,
        "evaluator_env_num": DEFAULT_EVALUATOR_ENV_NUM,
        "n_evaluator_episode": DEFAULT_N_EVALUATOR_EPISODE,
        "n_episode": DEFAULT_N_EPISODE,
        "num_simulations": DEFAULT_NUM_SIMULATIONS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "model_support_cap": DEFAULT_MODEL_SUPPORT_CAP,
        "td_steps": DEFAULT_TD_STEPS,
        "lightzero_eval_freq": DEFAULT_LIGHTZERO_EVAL_FREQ,
        "skip_lightzero_eval_in_profile": DEFAULT_SKIP_LIGHTZERO_EVAL_IN_PROFILE,
        "profile_cuda_sync_enabled": DEFAULT_PROFILE_CUDA_SYNC_ENABLED,
        "profile_allow_auto_resume": DEFAULT_PROFILE_ALLOW_AUTO_RESUME,
        "profile_volume_commit": DEFAULT_PROFILE_VOLUME_COMMIT,
        "lightzero_multi_gpu": DEFAULT_LIGHTZERO_MULTI_GPU,
        "save_ckpt_after_iter": DEFAULT_SAVE_CKPT_AFTER_ITER,
        "commit_on_checkpoint": DEFAULT_COMMIT_ON_CHECKPOINT,
        "stop_after_learner_train_calls": DEFAULT_STOP_AFTER_LEARNER_TRAIN_CALLS,
        "env_variant": DEFAULT_ENV_VARIANT,
        "reward_variant": DEFAULT_REWARD_VARIANT,
        "reward_outcome_alpha": DEFAULT_REWARD_OUTCOME_ALPHA,
        "source_state_trail_render_mode": DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE,
        "source_state_bonus_render_mode": DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE,
        "policy_observation_backend": DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE,
        "collect_search_backend": DEFAULT_COLLECT_SEARCH_BACKEND,
        "learner_seat_mode": DEFAULT_LEARNER_SEAT_MODE,
        "ego_action_straight_override_probability": (
            DEFAULT_EGO_ACTION_STRAIGHT_OVERRIDE_PROBABILITY
        ),
        "policy_action_repeat_min": DEFAULT_POLICY_ACTION_REPEAT_MIN,
        "policy_action_repeat_max": DEFAULT_POLICY_ACTION_REPEAT_MAX,
        "policy_action_repeat_extra_probability": (DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY),
        "control_noise_profile_id": DEFAULT_CONTROL_NOISE_PROFILE_ID,
        "disable_death_for_profile": DEFAULT_DISABLE_DEATH_FOR_PROFILE,
        "opponent_death_mode": DEFAULT_OPPONENT_DEATH_MODE,
        "opponent_runtime_mode": DEFAULT_OPPONENT_RUNTIME_MODE,
        "env_telemetry_stride": DEFAULT_ENV_TELEMETRY_STRIDE,
        "env_manager_type": DEFAULT_ENV_MANAGER_TYPE,
        "opponent_policy_kind": DEFAULT_OPPONENT_POLICY_KIND,
        "opponent_use_cuda": DEFAULT_OPPONENT_USE_CUDA,
        "opponent_checkpoint_ref": None,
        "opponent_snapshot_ref": None,
        "opponent_checkpoint_report_ref": None,
        "opponent_checkpoint_state_key": None,
        "opponent_mixture_spec": None,
        "opponent_assignment_ref": None,
        "initial_policy_checkpoint_ref": None,
        "initial_policy_checkpoint_state_key": None,
        "initial_policy_checkpoint_load_mode": DEFAULT_INITIAL_POLICY_CHECKPOINT_LOAD_MODE,
        "opponent_assignment_refresh_interval_train_iter": (
            DEFAULT_OPPONENT_ASSIGNMENT_REFRESH_INTERVAL_TRAIN_ITER
        ),
        "opponent_assignment_refresh_ref": None,
        "own_checkpoint_opponent_refresh_enabled": (
            DEFAULT_OWN_CHECKPOINT_OPPONENT_REFRESH_ENABLED
        ),
        "background_eval_enabled": DEFAULT_BACKGROUND_EVAL_ENABLED,
        "background_eval_launch_kind": DEFAULT_BACKGROUND_EVAL_LAUNCH_KIND,
        "background_eval_compute": DEFAULT_BACKGROUND_EVAL_COMPUTE,
        "background_eval_id_prefix": DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
        "background_eval_seed_count": DEFAULT_BACKGROUND_EVAL_SEED_COUNT,
        "background_eval_seed_rng_seed": DEFAULT_BACKGROUND_EVAL_SEED_RNG_SEED,
        "background_eval_max_steps": DEFAULT_BACKGROUND_EVAL_MAX_STEPS,
        "background_eval_step_detail_limit": DEFAULT_BACKGROUND_EVAL_STEP_DETAIL_LIMIT,
        "background_eval_num_simulations": DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS,
        "background_eval_batch_size": DEFAULT_BACKGROUND_EVAL_BATCH_SIZE,
        "background_gif_enabled": DEFAULT_BACKGROUND_GIF_ENABLED,
        "background_gif_seed_offset": DEFAULT_BACKGROUND_GIF_SEED_OFFSET,
        "background_gif_max_steps": DEFAULT_BACKGROUND_GIF_MAX_STEPS,
        "background_gif_frame_stride": DEFAULT_BACKGROUND_GIF_FRAME_STRIDE,
        "background_gif_fps": DEFAULT_BACKGROUND_GIF_FPS,
        "background_gif_scale": DEFAULT_BACKGROUND_GIF_SCALE,
        "background_gif_frame_size": DEFAULT_BACKGROUND_GIF_FRAME_SIZE,
        "background_gif_collect_temperature": DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE,
        "background_gif_collect_epsilon": DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON,
        "exploration_bonus_mode": xb.EXPLORATION_BONUS_MODE_NONE,
        "exploration_bonus_weight": 0.0,
        "exploration_bonus_feature_source": xb.RND_FEATURE_SOURCE_POLICY_GRAY64_LATEST_V0,
        "exploration_bonus_rnd_batch_size": 64,
        "exploration_bonus_rnd_update_per_collect": xb.RND_DEFAULT_UPDATE_PER_COLLECT,
        "exploration_bonus_rnd_buffer_size": 100_000,
        "exploration_bonus_rnd_learning_rate": 3e-4,
        "exploration_bonus_rnd_weight_decay": 1e-4,
        "exploration_bonus_rnd_input_norm": False,
        "require_rnd_metrics": False,
    }
    for key, value in defaults.items():
        kwargs.setdefault(key, value)


@app.function(
    image=image,
    volumes=TRAINER_VOLUMES,
    timeout=DEFAULT_STOCK_TRAIN_MODAL_TIMEOUT_SEC,
    cpu=64.0,
    memory=65536,
)
def lightzero_curvytron_visual_survival_cpu64(**kwargs: Any) -> dict[str, Any]:
    _apply_visual_survival_train_default_kwargs(kwargs)
    return _run_visual_survival_train(compute=COMPUTE_CPU64, **kwargs)


@app.function(
    image=image,
    volumes=TRAINER_VOLUMES,
    timeout=DEFAULT_STOCK_TRAIN_MODAL_TIMEOUT_SEC,
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
    model_support_cap: int | None = DEFAULT_MODEL_SUPPORT_CAP,
    td_steps: int | None = DEFAULT_TD_STEPS,
    lightzero_eval_freq: int = DEFAULT_LIGHTZERO_EVAL_FREQ,
    skip_lightzero_eval_in_profile: bool = DEFAULT_SKIP_LIGHTZERO_EVAL_IN_PROFILE,
    profile_cuda_sync_enabled: bool = DEFAULT_PROFILE_CUDA_SYNC_ENABLED,
    profile_allow_auto_resume: bool = DEFAULT_PROFILE_ALLOW_AUTO_RESUME,
    profile_volume_commit: bool = DEFAULT_PROFILE_VOLUME_COMMIT,
    lightzero_multi_gpu: bool = DEFAULT_LIGHTZERO_MULTI_GPU,
    save_ckpt_after_iter: int = DEFAULT_SAVE_CKPT_AFTER_ITER,
    commit_on_checkpoint: bool = DEFAULT_COMMIT_ON_CHECKPOINT,
    stop_after_learner_train_calls: int = DEFAULT_STOP_AFTER_LEARNER_TRAIN_CALLS,
    env_variant: str = DEFAULT_ENV_VARIANT,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
    source_state_trail_render_mode: str = DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE,
    source_state_bonus_render_mode: str = DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE,
    policy_observation_backend: str = DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE,
    collect_search_backend: str = DEFAULT_COLLECT_SEARCH_BACKEND,
    collect_search_ctree_backend: str = DEFAULT_COLLECT_SEARCH_CTREE_BACKEND,
    learner_seat_mode: str = DEFAULT_LEARNER_SEAT_MODE,
    ego_action_straight_override_probability: float = (
        DEFAULT_EGO_ACTION_STRAIGHT_OVERRIDE_PROBABILITY
    ),
    policy_action_repeat_min: int = DEFAULT_POLICY_ACTION_REPEAT_MIN,
    policy_action_repeat_max: int = DEFAULT_POLICY_ACTION_REPEAT_MAX,
    policy_action_repeat_extra_probability: float = (
        DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
    ),
    control_noise_profile_id: str = DEFAULT_CONTROL_NOISE_PROFILE_ID,
    disable_death_for_profile: bool = DEFAULT_DISABLE_DEATH_FOR_PROFILE,
    opponent_death_mode: str = DEFAULT_OPPONENT_DEATH_MODE,
    opponent_runtime_mode: str = DEFAULT_OPPONENT_RUNTIME_MODE,
    env_telemetry_stride: int = DEFAULT_ENV_TELEMETRY_STRIDE,
    env_manager_type: str = DEFAULT_ENV_MANAGER_TYPE,
    opponent_policy_kind: str = DEFAULT_OPPONENT_POLICY_KIND,
    opponent_use_cuda: bool = DEFAULT_OPPONENT_USE_CUDA,
    opponent_checkpoint_ref: str | None = None,
    opponent_snapshot_ref: str | None = None,
    opponent_checkpoint_report_ref: str | None = None,
    opponent_checkpoint_state_key: str | None = None,
    opponent_mixture_spec: str | None = None,
    opponent_assignment_ref: str | None = None,
    initial_policy_checkpoint_ref: str | None = None,
    initial_policy_checkpoint_state_key: str | None = None,
    initial_policy_checkpoint_load_mode: str = DEFAULT_INITIAL_POLICY_CHECKPOINT_LOAD_MODE,
    opponent_assignment_refresh_interval_train_iter: int = (
        DEFAULT_OPPONENT_ASSIGNMENT_REFRESH_INTERVAL_TRAIN_ITER
    ),
    opponent_assignment_refresh_ref: str | None = None,
    own_checkpoint_opponent_refresh_enabled: bool = (
        DEFAULT_OWN_CHECKPOINT_OPPONENT_REFRESH_ENABLED
    ),
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
    background_gif_collect_temperature: float = DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE,
    background_gif_collect_epsilon: float = DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON,
    exploration_bonus_mode: str = xb.EXPLORATION_BONUS_MODE_NONE,
    exploration_bonus_weight: float = 0.0,
    exploration_bonus_feature_source: str = xb.RND_FEATURE_SOURCE_POLICY_GRAY64_LATEST_V0,
    exploration_bonus_rnd_batch_size: int = 64,
    exploration_bonus_rnd_update_per_collect: int = xb.RND_DEFAULT_UPDATE_PER_COLLECT,
    exploration_bonus_rnd_buffer_size: int = 100_000,
    exploration_bonus_rnd_learning_rate: float = 3e-4,
    exploration_bonus_rnd_weight_decay: float = 1e-4,
    exploration_bonus_rnd_input_norm: bool = False,
    require_rnd_metrics: bool = False,
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
        model_support_cap=model_support_cap,
        td_steps=td_steps,
        lightzero_eval_freq=lightzero_eval_freq,
        skip_lightzero_eval_in_profile=skip_lightzero_eval_in_profile,
        profile_cuda_sync_enabled=profile_cuda_sync_enabled,
        profile_allow_auto_resume=profile_allow_auto_resume,
        profile_volume_commit=profile_volume_commit,
        lightzero_multi_gpu=lightzero_multi_gpu,
        save_ckpt_after_iter=save_ckpt_after_iter,
        commit_on_checkpoint=commit_on_checkpoint,
        stop_after_learner_train_calls=stop_after_learner_train_calls,
        env_variant=env_variant,
        reward_variant=reward_variant,
        reward_outcome_alpha=reward_outcome_alpha,
        source_state_trail_render_mode=source_state_trail_render_mode,
        source_state_bonus_render_mode=source_state_bonus_render_mode,
        policy_observation_backend=policy_observation_backend,
        collect_search_backend=collect_search_backend,
        collect_search_ctree_backend=collect_search_ctree_backend,
        learner_seat_mode=learner_seat_mode,
        ego_action_straight_override_probability=ego_action_straight_override_probability,
        policy_action_repeat_min=policy_action_repeat_min,
        policy_action_repeat_max=policy_action_repeat_max,
        policy_action_repeat_extra_probability=policy_action_repeat_extra_probability,
        control_noise_profile_id=control_noise_profile_id,
        disable_death_for_profile=disable_death_for_profile,
        opponent_death_mode=opponent_death_mode,
        opponent_runtime_mode=opponent_runtime_mode,
        env_telemetry_stride=env_telemetry_stride,
        env_manager_type=env_manager_type,
        opponent_policy_kind=opponent_policy_kind,
        opponent_use_cuda=opponent_use_cuda,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
        opponent_snapshot_ref=opponent_snapshot_ref,
        opponent_checkpoint_report_ref=opponent_checkpoint_report_ref,
        opponent_checkpoint_state_key=opponent_checkpoint_state_key,
        opponent_mixture_spec=opponent_mixture_spec,
        opponent_assignment_ref=opponent_assignment_ref,
        initial_policy_checkpoint_ref=initial_policy_checkpoint_ref,
        initial_policy_checkpoint_state_key=initial_policy_checkpoint_state_key,
        initial_policy_checkpoint_load_mode=initial_policy_checkpoint_load_mode,
        opponent_assignment_refresh_interval_train_iter=(
            opponent_assignment_refresh_interval_train_iter
        ),
        opponent_assignment_refresh_ref=opponent_assignment_refresh_ref,
        own_checkpoint_opponent_refresh_enabled=own_checkpoint_opponent_refresh_enabled,
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
        background_gif_collect_temperature=background_gif_collect_temperature,
        background_gif_collect_epsilon=background_gif_collect_epsilon,
        exploration_bonus_mode=exploration_bonus_mode,
        exploration_bonus_weight=exploration_bonus_weight,
        exploration_bonus_feature_source=exploration_bonus_feature_source,
        exploration_bonus_rnd_batch_size=exploration_bonus_rnd_batch_size,
        exploration_bonus_rnd_update_per_collect=exploration_bonus_rnd_update_per_collect,
        exploration_bonus_rnd_buffer_size=exploration_bonus_rnd_buffer_size,
        exploration_bonus_rnd_learning_rate=exploration_bonus_rnd_learning_rate,
        exploration_bonus_rnd_weight_decay=exploration_bonus_rnd_weight_decay,
        exploration_bonus_rnd_input_norm=exploration_bonus_rnd_input_norm,
        require_rnd_metrics=require_rnd_metrics,
    )


@app.function(
    image=image,
    volumes=TRAINER_VOLUMES,
    timeout=DEFAULT_STOCK_TRAIN_MODAL_TIMEOUT_SEC,
    cpu=40.0,
    memory=65536,
    gpu=CHEAP_GPU_RESOURCE,
)
def lightzero_curvytron_visual_survival_gpu_cpu40(**kwargs: Any) -> dict[str, Any]:
    _apply_visual_survival_train_default_kwargs(kwargs)
    return _run_visual_survival_train(compute=COMPUTE_GPU_L4_T4_CPU40, **kwargs)


@app.function(
    image=ctree_a3_image,
    volumes=TRAINER_VOLUMES,
    timeout=DEFAULT_STOCK_TRAIN_MODAL_TIMEOUT_SEC,
    cpu=40.0,
    memory=65536,
    gpu=CHEAP_GPU_RESOURCE,
)
def lightzero_curvytron_visual_survival_gpu_cpu40_ctree_a3(
    **kwargs: Any,
) -> dict[str, Any]:
    _apply_visual_survival_train_default_kwargs(kwargs)
    return _run_visual_survival_train(compute=COMPUTE_GPU_L4_T4_CPU40, **kwargs)


@app.function(
    image=image,
    volumes=TRAINER_VOLUMES,
    timeout=DEFAULT_STOCK_TRAIN_MODAL_TIMEOUT_SEC,
    cpu=40.0,
    memory=65536,
    gpu=H100_GPU_RESOURCE,
)
def lightzero_curvytron_visual_survival_h100_cpu40(**kwargs: Any) -> dict[str, Any]:
    _apply_visual_survival_train_default_kwargs(kwargs)
    return _run_visual_survival_train(compute=COMPUTE_GPU_H100_CPU40, **kwargs)


@app.function(
    image=ctree_a3_image,
    volumes=TRAINER_VOLUMES,
    timeout=DEFAULT_STOCK_TRAIN_MODAL_TIMEOUT_SEC,
    cpu=40.0,
    memory=65536,
    gpu=H100_GPU_RESOURCE,
)
def lightzero_curvytron_visual_survival_h100_cpu40_ctree_a3(
    **kwargs: Any,
) -> dict[str, Any]:
    _apply_visual_survival_train_default_kwargs(kwargs)
    return _run_visual_survival_train(compute=COMPUTE_GPU_H100_CPU40, **kwargs)


@app.function(
    image=image,
    volumes=TRAINER_VOLUMES,
    timeout=DEFAULT_STOCK_TRAIN_MODAL_TIMEOUT_SEC,
    cpu=40.0,
    memory=65536,
    gpu=H100X2_GPU_RESOURCE,
)
def lightzero_curvytron_visual_survival_h100x2_cpu40(**kwargs: Any) -> dict[str, Any]:
    _apply_visual_survival_train_default_kwargs(kwargs)
    return _run_visual_survival_train(compute=COMPUTE_GPU_H100X2_CPU40, **kwargs)


@app.function(image=image, volumes=TRAINER_VOLUMES, timeout=20 * 60, cpu=2.0)
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


def _compact_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compact_effective_env_steps(
    *,
    counts: dict[str, Any],
    command: dict[str, Any],
) -> tuple[int | None, int | None, str | None]:
    raw_env_steps = _compact_int(counts.get("env_steps_collected"))
    if raw_env_steps is not None and raw_env_steps > 0:
        return raw_env_steps, raw_env_steps, "collector_envstep_delta"

    mcts_root_sum = _compact_int(counts.get("mcts_search_root_sum"))
    evaluator_eval_calls = _compact_int(counts.get("evaluator_eval_calls"))
    evaluator_eval_skipped_calls = _compact_int(counts.get("evaluator_eval_skipped_calls"))
    lightzero_eval_freq = _compact_int(command.get("lightzero_eval_freq"))
    skip_eval = bool(command.get("skip_lightzero_eval_in_profile"))
    if (
        (raw_env_steps is None or raw_env_steps == 0)
        and mcts_root_sum is not None
        and mcts_root_sum > 0
        and skip_eval
        and lightzero_eval_freq == 0
        and evaluator_eval_calls == 0
        and evaluator_eval_skipped_calls == 0
    ):
        return mcts_root_sum, raw_env_steps, "mcts_search_root_sum_profile_fallback"
    return raw_env_steps, raw_env_steps, "collector_envstep_delta"


def _compact_steps_per_sec_currency(env_steps_source: str | None) -> str:
    if env_steps_source == "collector_envstep_delta":
        return "stock_train_muzero_profile_env_steps_per_sec"
    if env_steps_source == "mcts_search_root_sum_profile_fallback":
        return "stock_train_muzero_profile_mcts_roots_per_sec_fallback"
    return "unknown_profile_steps_per_sec"


def _compact_train_result_for_output(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    train_result = result.get("train") if isinstance(result.get("train"), dict) else result
    if not isinstance(train_result, dict):
        return result
    phase = train_result.get("phase_profile")
    collect_search_backend_proof = train_result.get("collect_search_backend_proof")
    command = train_result.get("command")
    action = train_result.get("action_observability")
    runtime = train_result.get("runtime_compute")
    rnd_metrics = train_result.get("rnd_reward_model_metrics")
    phase = phase if isinstance(phase, dict) else {}
    collect_search_backend_proof = (
        collect_search_backend_proof
        if isinstance(collect_search_backend_proof, dict)
        else {}
    )
    command = command if isinstance(command, dict) else {}
    action = action if isinstance(action, dict) else {}
    runtime = runtime if isinstance(runtime, dict) else {}
    rnd_metrics = rnd_metrics if isinstance(rnd_metrics, dict) else {}
    rnd_latest = (
        rnd_metrics.get("latest")
        if isinstance(rnd_metrics.get("latest"), dict)
        else {}
    )
    timers = phase.get("timers_sec") if isinstance(phase.get("timers_sec"), dict) else {}
    counts = phase.get("counts") if isinstance(phase.get("counts"), dict) else {}
    derived = phase.get("derived_stats") if isinstance(phase.get("derived_stats"), dict) else {}
    sample_stats = phase.get("sample_stats") if isinstance(phase.get("sample_stats"), dict) else {}
    phase_samples = phase.get("samples") if isinstance(phase.get("samples"), dict) else {}
    observed_collect_search_ctree_backends = sorted(
        {
            str(value)
            for value in phase_samples.get("collect_search_ctree_backend", [])
            if value is not None
        }
    )
    observed_collect_search_backends = sorted(
        {
            str(value)
            for value in phase_samples.get("collect_search_backend", [])
            if value is not None
        }
    )
    gpu = phase.get("gpu_sampling") if isinstance(phase.get("gpu_sampling"), dict) else {}
    mcts_search_calls = counts.get("mcts_search_calls")
    mcts_root_sum = counts.get("mcts_search_root_sum")
    train_wall = timers.get("train_muzero_wall_sec")
    env_steps, env_steps_raw, env_steps_source = _compact_effective_env_steps(
        counts=counts,
        command=command,
    )
    steps_per_sec_currency = _compact_steps_per_sec_currency(env_steps_source)
    steps_per_sec_uses_fallback = env_steps_source != "collector_envstep_delta"
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
    direct_recurrent_batch_mean = None
    try:
        direct_recurrent_calls = counts.get("collect_search_backend_recurrent_inference_calls")
        direct_node_budget = counts.get("mcts_search_node_budget_sum")
        if direct_recurrent_calls and direct_node_budget is not None:
            direct_recurrent_batch_mean = float(direct_node_budget) / float(direct_recurrent_calls)
    except (TypeError, ValueError, ZeroDivisionError):
        direct_recurrent_batch_mean = None
    observation_contract = _to_plain(command.get("observation_contract"))
    if not isinstance(observation_contract, dict):
        observation_contract = {}
    env_manager_type = command.get("env_manager_type")
    scalar_materialization_semantics = (
        "batched_profile_manager_materializes_lightzero_scalar_timesteps"
        if env_manager_type == "curvyzero_batched_profile"
        else "stock_env_manager_materializes_lightzero_scalar_timesteps"
    )
    lightzero_to_play_mode = (
        "fixed_opponent_minus_one"
        if command.get("env_variant") == "source_state_fixed_opponent"
        else "env_observation_to_play"
    )
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
        "trainer_entrypoint": train_result.get("trainer_entrypoint"),
        "initial_policy_checkpoint": train_result.get("initial_policy_checkpoint"),
        "command": {
            "env_variant": command.get("env_variant"),
            "reward_variant": command.get("reward_variant"),
            "reward_outcome_alpha": command.get("reward_outcome_alpha"),
            "exploration_bonus": command.get("exploration_bonus"),
            "trainer_entrypoint": command.get("trainer_entrypoint"),
            "opponent_policy_kind": command.get("opponent_policy_kind"),
            "opponent_use_cuda": command.get("opponent_use_cuda"),
            "env_manager_type": command.get("env_manager_type"),
            "collector_env_num": command.get("collector_env_num"),
            "n_episode": command.get("n_episode"),
            "num_simulations": command.get("num_simulations"),
            "batch_size": command.get("batch_size"),
            "source_state_trail_render_mode": command.get("source_state_trail_render_mode"),
            "source_state_bonus_render_mode": command.get("source_state_bonus_render_mode"),
            "policy_observation_backend": command.get("policy_observation_backend"),
            "collect_search_backend": command.get("collect_search_backend"),
            "collect_search_ctree_backend": command.get("collect_search_ctree_backend"),
            "collect_search_backend_fallback_policy": command.get(
                "collect_search_backend_fallback_policy"
            ),
            "policy_trail_render_mode": command.get("policy_trail_render_mode"),
            "policy_bonus_render_mode": command.get("policy_bonus_render_mode"),
            "policy_observation_contract_id": command.get("policy_observation_contract_id"),
            "observation_contract": observation_contract,
            "lightzero_eval_freq": command.get("lightzero_eval_freq"),
            "skip_lightzero_eval_in_profile": command.get("skip_lightzero_eval_in_profile"),
            "profile_cuda_sync_enabled": command.get("profile_cuda_sync_enabled"),
            "profile_allow_auto_resume": command.get("profile_allow_auto_resume"),
            "profile_volume_commit": command.get("profile_volume_commit"),
            "profile_env_timing_enabled": command.get("profile_env_timing_enabled"),
            "lightzero_multi_gpu": command.get("lightzero_multi_gpu"),
            "source_max_steps": command.get("source_max_steps"),
            "ego_action_straight_override_probability": command.get(
                "ego_action_straight_override_probability"
            ),
            "policy_action_repeat_min": command.get("policy_action_repeat_min"),
            "policy_action_repeat_max": command.get("policy_action_repeat_max"),
            "policy_action_repeat_extra_probability": command.get(
                "policy_action_repeat_extra_probability"
            ),
            "control_noise_profile_id": command.get("control_noise_profile_id"),
            "disable_death_for_profile": command.get("disable_death_for_profile"),
            "opponent_death_mode": command.get("opponent_death_mode"),
            "opponent_runtime_mode": command.get("opponent_runtime_mode"),
            "env_telemetry_stride": command.get("env_telemetry_stride"),
            "save_ckpt_after_iter": command.get("save_ckpt_after_iter"),
            "commit_on_checkpoint": command.get("commit_on_checkpoint"),
            "opponent_assignment_refresh": command.get("opponent_assignment_refresh"),
        },
        "counts": {
            "env_steps_collected": env_steps,
            "env_steps_collected_effective": env_steps,
            "env_steps_collected_raw": env_steps_raw,
            "env_steps_collected_source": env_steps_source,
            "env_steps_collected_uses_fallback": steps_per_sec_uses_fallback,
            "mcts_search_calls": mcts_search_calls,
            "mcts_search_root_sum": mcts_root_sum,
            "mcts_search_simulation_budget_sum": counts.get("mcts_search_simulation_budget_sum"),
            "mcts_search_node_budget_sum": counts.get("mcts_search_node_budget_sum"),
            "collect_search_backend_direct_ctree_gpu_latent_calls": counts.get(
                "collect_search_backend_direct_ctree_gpu_latent_calls"
            ),
            "collect_search_backend_fallback_calls": counts.get(
                "collect_search_backend_fallback_calls"
            ),
            "collect_search_backend_output_fast_path_calls": counts.get(
                "collect_search_backend_output_fast_path_calls"
            ),
            "collect_search_backend_output_rows": counts.get(
                "collect_search_backend_output_rows"
            ),
            "collect_search_backend_ctree_traverse_calls": counts.get(
                "collect_search_backend_ctree_traverse_calls"
            ),
            "collect_search_backend_ctree_backpropagate_calls": counts.get(
                "collect_search_backend_ctree_backpropagate_calls"
            ),
            "collect_search_backend_recurrent_inference_calls": counts.get(
                "collect_search_backend_recurrent_inference_calls"
            ),
            "collect_search_backend_model_output_d2h_bytes": counts.get(
                "collect_search_backend_model_output_d2h_bytes"
            ),
            "learner_train_calls": counts.get("learner_train_calls"),
            "replay_sample_calls": counts.get("replay_sample_calls"),
            "evaluator_eval_calls": counts.get("evaluator_eval_calls"),
            "evaluator_eval_skipped_calls": counts.get("evaluator_eval_skipped_calls"),
            "env_registered_step_calls": counts.get("env_registered_step_calls"),
            "batched_profile_env_manager_create_calls": counts.get(
                "batched_profile_env_manager_create_calls"
            ),
            "batched_profile_env_manager_reset_calls": counts.get(
                "batched_profile_env_manager_reset_calls"
            ),
            "batched_profile_env_manager_step_calls": counts.get(
                "batched_profile_env_manager_step_calls"
            ),
            "env_step_info_calls": counts.get("env_step_info_calls"),
            "env_base_info_calls": counts.get("env_base_info_calls"),
            "rnd_collect_data_calls": counts.get("rnd_collect_data_calls"),
            "rnd_train_with_data_calls": counts.get("rnd_train_with_data_calls"),
            "rnd_estimate_calls": counts.get("rnd_estimate_calls"),
            "rnd_metrics_snapshot_calls": counts.get("rnd_metrics_snapshot_calls"),
            "rnd_metrics_write_snapshot_calls": counts.get("rnd_metrics_write_snapshot_calls"),
            "rnd_state_hash_calls": counts.get("rnd_state_hash_calls"),
        },
        "timers_sec": {
            "train_muzero_wall": train_wall,
            "collector_collect": timers.get("collector_collect_sec"),
            "mcts_search": timers.get("mcts_search_sec"),
            "policy_forward_collect": timers.get("policy_forward_collect_sec"),
            "policy_forward_eval": timers.get("policy_forward_eval_sec"),
            "model_initial_inference": timers.get("model_initial_inference_sec"),
            "model_recurrent_inference": timers.get("model_recurrent_inference_sec"),
            "collect_search_backend_initial_inference": timers.get(
                "collect_search_backend_initial_inference_sec"
            ),
            "collect_search_backend_root_prepare": timers.get(
                "collect_search_backend_root_prepare_sec"
            ),
            "collect_search_backend_ctree_traverse": timers.get(
                "collect_search_backend_ctree_traverse_sec"
            ),
            "collect_search_backend_tensor_index": timers.get(
                "collect_search_backend_tensor_index_sec"
            ),
            "collect_search_backend_recurrent_inference": timers.get(
                "collect_search_backend_recurrent_inference_sec"
            ),
            "collect_search_backend_model_output_d2h": timers.get(
                "collect_search_backend_model_output_d2h_sec"
            ),
            "collect_search_backend_model_output_listify": timers.get(
                "collect_search_backend_model_output_listify_sec"
            ),
            "collect_search_backend_ctree_backpropagate": timers.get(
                "collect_search_backend_ctree_backpropagate_sec"
            ),
            "collect_search_backend_flat_payload": timers.get(
                "collect_search_backend_flat_payload_sec"
            ),
            "collect_search_backend_output_assembly": timers.get(
                "collect_search_backend_output_assembly_sec"
            ),
            "learner_train": timers.get("learner_train_sec"),
            "replay_sample": timers.get("replay_sample_sec"),
            "evaluator_eval": timers.get("evaluator_eval_sec"),
            "env_telemetry_write": timers.get("env_telemetry_write_sec"),
            "env_registered_step": timers.get("env_registered_step_sec"),
            "batched_profile_env_manager_reset": timers.get(
                "batched_profile_env_manager_reset_sec"
            ),
            "batched_profile_env_manager_step": timers.get(
                "batched_profile_env_manager_step_sec"
            ),
            "batched_profile_surface_env_step": timers.get(
                "batched_profile_surface_env_step_sec"
            ),
            "batched_profile_surface_stack_update": timers.get(
                "batched_profile_surface_stack_update_sec"
            ),
            "batched_profile_surface_reward": timers.get(
                "batched_profile_surface_reward_sec"
            ),
            "batched_profile_surface_package": timers.get(
                "batched_profile_surface_package_sec"
            ),
            "batched_profile_surface_package_mask_copy": timers.get(
                "batched_profile_surface_package_mask_copy_sec"
            ),
            "batched_profile_surface_package_live_mask": timers.get(
                "batched_profile_surface_package_live_mask_sec"
            ),
            "batched_profile_surface_package_policy_rows": timers.get(
                "batched_profile_surface_package_policy_rows_sec"
            ),
            "batched_profile_surface_package_policy_observation": timers.get(
                "batched_profile_surface_package_policy_observation_sec"
            ),
            "batched_profile_surface_package_policy_action_mask": timers.get(
                "batched_profile_surface_package_policy_action_mask_sec"
            ),
            "batched_profile_surface_package_final_observation": timers.get(
                "batched_profile_surface_package_final_observation_sec"
            ),
            "batched_profile_surface_package_info": timers.get(
                "batched_profile_surface_package_info_sec"
            ),
            "batched_profile_surface_package_output_copy": timers.get(
                "batched_profile_surface_package_output_copy_sec"
            ),
            "batched_profile_bridge_action_env_id_sort": timers.get(
                "batched_profile_bridge_action_env_id_sort_sec"
            ),
            "batched_profile_bridge_joint_action_from_scalar": timers.get(
                "batched_profile_bridge_joint_action_from_scalar_sec"
            ),
            "batched_profile_bridge_loop_step": timers.get(
                "batched_profile_bridge_loop_step_sec"
            ),
            "batched_profile_bridge_output_from_loop_step": timers.get(
                "batched_profile_bridge_output_from_loop_step_sec"
            ),
            "batched_profile_bridge_policy_env_id_build": timers.get(
                "batched_profile_bridge_policy_env_id_build_sec"
            ),
            "batched_profile_bridge_ready_obs_by_env_id": timers.get(
                "batched_profile_bridge_ready_obs_by_env_id_sec"
            ),
            "batched_profile_bridge_timestep_env_id_prepare": timers.get(
                "batched_profile_bridge_timestep_env_id_prepare_sec"
            ),
            "batched_profile_bridge_env_id_timestep_materialize": timers.get(
                "batched_profile_bridge_env_id_timestep_materialize_sec"
            ),
            "batched_profile_bridge_split_timestep_by_env_id": timers.get(
                "batched_profile_bridge_split_timestep_by_env_id_sec"
            ),
            "batched_profile_bridge_stock_base_env_timestep_conversion": timers.get(
                "batched_profile_bridge_stock_base_env_timestep_conversion_sec"
            ),
            "batched_profile_bridge_autoreset_reset_materialize": timers.get(
                "batched_profile_bridge_autoreset_reset_materialize_sec"
            ),
            "batched_profile_renderer_render": timers.get(
                "batched_profile_renderer_render_sec"
            ),
            "batched_profile_renderer_device_render": timers.get(
                "batched_profile_renderer_device_render_sec"
            ),
            "batched_profile_renderer_host_to_device": timers.get(
                "batched_profile_renderer_host_to_device_sec"
            ),
            "batched_profile_renderer_device_to_host": timers.get(
                "batched_profile_renderer_device_to_host_sec"
            ),
            "batched_profile_renderer_prewarm": timers.get(
                "batched_profile_renderer_prewarm_sec"
            ),
            "batched_profile_renderer_pack": timers.get(
                "batched_profile_renderer_owner_ordered_pack_sec"
            ),
            "env_step_info": timers.get("env_step_info_sec"),
            "env_base_info": timers.get("env_base_info_sec"),
            "learner_save_checkpoint": timers.get("learner_save_checkpoint_sec"),
            "rnd_collect_data": timers.get("rnd_collect_data_sec"),
            "rnd_train_with_data": timers.get("rnd_train_with_data_sec"),
            "rnd_estimate": timers.get("rnd_estimate_sec"),
            "rnd_metrics_snapshot": timers.get("rnd_metrics_snapshot_sec"),
            "rnd_metrics_write_snapshot": timers.get("rnd_metrics_write_snapshot_sec"),
            "rnd_state_hash": timers.get("rnd_state_hash_sec"),
        },
        "derived": {
            "steps_per_sec": steps_per_sec,
            "steps_per_sec_currency": steps_per_sec_currency,
            "steps_per_sec_source": env_steps_source,
            "steps_per_sec_uses_fallback_denominator": steps_per_sec_uses_fallback,
            "mcts_root_batch_mean": mcts_root_batch_mean,
            "mcts_recurrent_batch_mean": derived.get(
                "model_recurrent_inference_in_mcts_search_batch_mean"
            ),
            "collect_search_backend_recurrent_batch_mean": direct_recurrent_batch_mean,
            "batched_profile_ready_obs_before_step_mean": (
                sample_stats.get("batched_profile_env_manager_ready_obs_before_step", {})
                .get("mean")
            ),
            "batched_profile_ready_obs_after_step_mean": (
                sample_stats.get("batched_profile_env_manager_ready_obs_after_step", {})
                .get("mean")
            ),
            "batched_profile_timestep_count_mean": (
                sample_stats.get("batched_profile_env_manager_timestep_count", {}).get("mean")
            ),
            "batched_profile_action_row_count_mean": (
                sample_stats.get("batched_profile_env_manager_action_row_count", {}).get("mean")
            ),
            "batched_profile_complete_row_omission_count_mean": (
                sample_stats.get(
                    "batched_profile_env_manager_complete_row_omission_count",
                    {},
                ).get("mean")
            ),
            "batched_profile_partial_render_request_mean": (
                sample_stats.get("batched_profile_renderer_partial_render_request", {}).get(
                    "mean"
                )
            ),
            "batched_profile_render_output_count_mean": (
                sample_stats.get("batched_profile_renderer_render_output_count", {}).get("mean")
            ),
        },
        "semantic_identity": {
            "schema_id": "curvyzero_optimizer_profile_semantic_identity/v0",
            "profile_mode": train_result.get("mode"),
            "called_train_muzero": train_result.get("called_train_muzero"),
            "trainer_entrypoint": train_result.get("trainer_entrypoint"),
            "env_variant": command.get("env_variant"),
            "env_manager_type": env_manager_type,
            "policy_observation_backend": command.get("policy_observation_backend"),
            "collect_search_backend": command.get("collect_search_backend"),
            "collect_search_ctree_backend": command.get("collect_search_ctree_backend"),
            "collect_search_backend_fallback_policy": command.get(
                "collect_search_backend_fallback_policy"
            ),
            "policy_observation_contract_id": command.get("policy_observation_contract_id"),
            "observation_contract_id": observation_contract.get("contract_id"),
            "observation_surface_label": observation_contract.get("surface_label"),
            "observation_stack_shape": observation_contract.get("stack_shape"),
            "observation_single_frame_shape": observation_contract.get("single_frame_shape"),
            "observation_raw_dtype": observation_contract.get("raw_dtype"),
            "observation_model_dtype": observation_contract.get("model_dtype"),
            "source_state_trail_render_mode": command.get("source_state_trail_render_mode"),
            "source_state_bonus_render_mode": command.get("source_state_bonus_render_mode"),
            "policy_trail_render_mode": command.get("policy_trail_render_mode"),
            "policy_bonus_render_mode": command.get("policy_bonus_render_mode"),
            "death_mode": (
                "profile_no_death"
                if command.get("disable_death_for_profile")
                else "normal"
            ),
            "opponent_death_mode": command.get("opponent_death_mode"),
            "rnd_mode": (
                command.get("exploration_bonus", {}).get("mode")
                if isinstance(command.get("exploration_bonus"), dict)
                else None
            ),
            "env_steps_collected_source": env_steps_source,
            "speed_currency": steps_per_sec_currency,
            "scalar_materialization_semantics": scalar_materialization_semantics,
            "materialized_timestep_count_mean": (
                sample_stats.get("batched_profile_env_manager_timestep_count", {}).get("mean")
            ),
            "lightzero_to_play_mode": lightzero_to_play_mode,
            "zero_mask_filtering_semantics": "environment_action_mask_consumed_by_lightzero",
            "consumer_semantics": "stock_lightzero_train_muzero_collect_search_replay_learner",
            "cpu_tree_included": True,
        },
        "telemetry": {
            "row_count": action.get("row_count"),
            "counts_scope": action.get("counts_scope"),
            "telemetry_sampled": action.get("telemetry_sampled"),
            "telemetry_stride": action.get("telemetry_stride"),
            "profile_env_timing_sec": action.get("profile_env_timing_sec"),
        },
        "gpu": {
            "requested_compute": runtime.get("requested_compute"),
            "available": runtime.get("torch_cuda_available"),
            "max_util_percent": gpu.get("max_gpu_util_percent"),
            "max_memory_used_mib": gpu.get("max_memory_used_mib"),
            "sample_count": gpu.get("sample_count"),
        },
        "search_backend_proof": {
            "schema_id": collect_search_backend_proof.get("schema_id"),
            "requested_backend": collect_search_backend_proof.get("requested_backend"),
            "requested_ctree_backend": collect_search_backend_proof.get(
                "requested_ctree_backend"
            ),
            "fallback_policy": collect_search_backend_proof.get("fallback_policy"),
            "observed_collect_search_backends": observed_collect_search_backends,
            "observed_collect_search_ctree_backends": observed_collect_search_ctree_backends,
            "direct_ctree_gpu_latent_calls": collect_search_backend_proof.get(
                "direct_ctree_gpu_latent_calls"
            ),
            "fallback_calls": collect_search_backend_proof.get("fallback_calls"),
            "output_rows": collect_search_backend_proof.get("output_rows"),
            "flat_payload_timer_present": timers.get("collect_search_backend_flat_payload_sec")
            is not None,
        },
        "final_volume_commit": train_result.get("final_volume_commit"),
        "rnd_reward_model_metrics": {
            "enabled": rnd_metrics.get("enabled"),
            "required": command.get("require_rnd_metrics"),
            "latest_exists": rnd_metrics.get("latest_exists"),
            "jsonl_exists": rnd_metrics.get("jsonl_exists"),
            "event_count": rnd_metrics.get("event_count"),
            "latest_ref": rnd_metrics.get("latest_ref"),
            "jsonl_ref": rnd_metrics.get("jsonl_ref"),
            "constructed": rnd_latest.get("constructed"),
            "buffer_count": rnd_latest.get("buffer_count"),
            "collect_data_calls": rnd_latest.get("collect_data_calls"),
            "train_with_data_calls": rnd_latest.get("train_with_data_calls"),
            "train_with_data_skipped_small_buffer_count": rnd_latest.get(
                "train_with_data_skipped_small_buffer_count"
            ),
            "train_cnt_rnd": rnd_latest.get("train_cnt_rnd"),
            "estimate_calls": rnd_latest.get("estimate_calls"),
            "estimate_cnt_rnd": rnd_latest.get("estimate_cnt_rnd"),
            "train_cnt_per_estimate": rnd_latest.get("train_cnt_per_estimate"),
            "train_with_data_calls_per_collect": rnd_latest.get(
                "train_with_data_calls_per_collect"
            ),
            "last_train_loss": rnd_latest.get("last_train_loss"),
            "last_raw_mse_mean": rnd_latest.get("last_raw_mse_mean"),
            "last_raw_mse_min": rnd_latest.get("last_raw_mse_min"),
            "last_raw_mse_max": rnd_latest.get("last_raw_mse_max"),
            "last_raw_mse_std": rnd_latest.get("last_raw_mse_std"),
            "last_raw_mse_p50": rnd_latest.get("last_raw_mse_p50"),
            "last_raw_mse_p95": rnd_latest.get("last_raw_mse_p95"),
            "last_intrinsic_mean": rnd_latest.get("last_intrinsic_mean"),
            "last_intrinsic_min": rnd_latest.get("last_intrinsic_min"),
            "last_intrinsic_max": rnd_latest.get("last_intrinsic_max"),
            "last_target_reward_changed": rnd_latest.get("last_target_reward_changed"),
            "last_target_reward_delta_abs_mean": rnd_latest.get(
                "last_target_reward_delta_abs_mean"
            ),
            "last_target_reward_delta_abs_max": rnd_latest.get(
                "last_target_reward_delta_abs_max"
            ),
            "predictor_changed": bool(
                rnd_latest.get("last_predictor_hash_before_train")
                and rnd_latest.get("last_predictor_hash_after_train")
                and rnd_latest.get("last_predictor_hash_before_train")
                != rnd_latest.get("last_predictor_hash_after_train")
            ),
            "target_changed": (
                None
                if not rnd_latest.get("last_target_hash_before_train")
                or not rnd_latest.get("last_target_hash_after_train")
                else rnd_latest.get("last_target_hash_before_train")
                != rnd_latest.get("last_target_hash_after_train")
            ),
        },
    }
    policy_observation_gpu_summary = action.get("policy_observation_gpu_last_profile")
    if policy_observation_gpu_summary is not None:
        compact["telemetry"]["policy_observation_gpu_last_profile"] = policy_observation_gpu_summary
    elif str(command.get("policy_observation_backend") or "") == "jax_gpu":
        compact["telemetry"]["policy_observation_gpu_last_profile"] = {
            "status": "missing",
            "scope": action.get("counts_scope"),
            "sampled_profile_count": 0,
            "reason": "telemetry rows lacked policy_observation_gpu_last_profile",
        }
    if "background_eval" in result:
        compact["background_eval"] = result["background_eval"]
    return compact


@app.local_entrypoint()
def checkpoint_selfplay_gif_main(
    checkpoint_ref: str,
    checkpoint_label: str = "",
    eval_id: str = DEFAULT_BACKGROUND_EVAL_ID_PREFIX,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    seed: int = DEFAULT_SEED + DEFAULT_BACKGROUND_GIF_SEED_OFFSET,
    max_steps: int = DEFAULT_BACKGROUND_GIF_MAX_STEPS,
    source_max_steps: int = DEFAULT_SOURCE_MAX_STEPS,
    decision_ms: float = DEFAULT_DECISION_MS,
    decision_source_frames: int = DEFAULT_DECISION_SOURCE_FRAMES,
    source_physics_step_ms: float = DEFAULT_SOURCE_PHYSICS_STEP_MS,
    source_max_steps_semantics: str = "source_physics_steps",
    num_simulations: int = DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BACKGROUND_EVAL_BATCH_SIZE,
    frame_stride: int = DEFAULT_BACKGROUND_GIF_FRAME_STRIDE,
    fps: float = DEFAULT_BACKGROUND_GIF_FPS,
    scale: int = DEFAULT_BACKGROUND_GIF_SCALE,
    frame_size: int = DEFAULT_BACKGROUND_GIF_FRAME_SIZE,
    training_env_variant: str = DEFAULT_ENV_VARIANT,
    training_reward_variant: str = DEFAULT_REWARD_VARIANT,
    training_reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
    opponent_death_mode: str = DEFAULT_OPPONENT_DEATH_MODE,
    opponent_runtime_mode: str = DEFAULT_OPPONENT_RUNTIME_MODE,
    opponent_mixture_json: str = "",
    natural_bonus_spawn: bool = TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN,
    collect_temperature: float = DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE,
    collect_epsilon: float = DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON,
) -> None:
    opponent_mixture_spec = None
    if opponent_mixture_json:
        opponent_mixture_spec = json.loads(opponent_mixture_json)
    result = lightzero_curvytron_visual_survival_checkpoint_selfplay_gif.remote(
        checkpoint_ref=checkpoint_ref,
        checkpoint_label=checkpoint_label or None,
        eval_id=eval_id,
        run_id=run_id,
        attempt_id=attempt_id,
        seed=seed,
        max_steps=max_steps,
        source_max_steps=source_max_steps,
        decision_ms=decision_ms,
        decision_source_frames=decision_source_frames,
        source_physics_step_ms=source_physics_step_ms,
        source_max_steps_semantics=source_max_steps_semantics,
        num_simulations=num_simulations,
        batch_size=batch_size,
        frame_stride=frame_stride,
        fps=fps,
        scale=scale,
        frame_size=frame_size,
        training_env_variant=training_env_variant,
        training_reward_variant=training_reward_variant,
        training_reward_outcome_alpha=training_reward_outcome_alpha,
        opponent_death_mode=opponent_death_mode,
        opponent_runtime_mode=opponent_runtime_mode,
        opponent_mixture_spec=opponent_mixture_spec,
        natural_bonus_spawn=natural_bonus_spawn,
        collect_temperature=collect_temperature,
        collect_epsilon=collect_epsilon,
    )
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))


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
    model_support_cap: int | None = DEFAULT_MODEL_SUPPORT_CAP,
    td_steps: int | None = DEFAULT_TD_STEPS,
    lightzero_eval_freq: int = DEFAULT_LIGHTZERO_EVAL_FREQ,
    skip_lightzero_eval_in_profile: bool = DEFAULT_SKIP_LIGHTZERO_EVAL_IN_PROFILE,
    profile_cuda_sync_enabled: bool = DEFAULT_PROFILE_CUDA_SYNC_ENABLED,
    profile_allow_auto_resume: bool = DEFAULT_PROFILE_ALLOW_AUTO_RESUME,
    profile_volume_commit: bool = DEFAULT_PROFILE_VOLUME_COMMIT,
    profile_spawn: bool = DEFAULT_PROFILE_SPAWN,
    lightzero_multi_gpu: bool = DEFAULT_LIGHTZERO_MULTI_GPU,
    save_ckpt_after_iter: int = DEFAULT_SAVE_CKPT_AFTER_ITER,
    commit_on_checkpoint: bool = DEFAULT_COMMIT_ON_CHECKPOINT,
    stop_after_learner_train_calls: int = DEFAULT_STOP_AFTER_LEARNER_TRAIN_CALLS,
    env_variant: str = DEFAULT_ENV_VARIANT,
    reward_variant: str = DEFAULT_REWARD_VARIANT,
    reward_outcome_alpha: float = DEFAULT_REWARD_OUTCOME_ALPHA,
    source_state_trail_render_mode: str = DEFAULT_SOURCE_STATE_TRAIL_RENDER_MODE,
    source_state_bonus_render_mode: str = DEFAULT_SOURCE_STATE_BONUS_RENDER_MODE,
    policy_observation_backend: str = DEFAULT_POLICY_OBSERVATION_BACKEND_CHOICE,
    collect_search_backend: str = DEFAULT_COLLECT_SEARCH_BACKEND,
    collect_search_ctree_backend: str = DEFAULT_COLLECT_SEARCH_CTREE_BACKEND,
    learner_seat_mode: str = DEFAULT_LEARNER_SEAT_MODE,
    ego_action_straight_override_probability: float = (
        DEFAULT_EGO_ACTION_STRAIGHT_OVERRIDE_PROBABILITY
    ),
    policy_action_repeat_min: int = DEFAULT_POLICY_ACTION_REPEAT_MIN,
    policy_action_repeat_max: int = DEFAULT_POLICY_ACTION_REPEAT_MAX,
    policy_action_repeat_extra_probability: float = (
        DEFAULT_POLICY_ACTION_REPEAT_EXTRA_PROBABILITY
    ),
    control_noise_profile_id: str = DEFAULT_CONTROL_NOISE_PROFILE_ID,
    disable_death_for_profile: bool = DEFAULT_DISABLE_DEATH_FOR_PROFILE,
    opponent_death_mode: str = DEFAULT_OPPONENT_DEATH_MODE,
    opponent_runtime_mode: str = DEFAULT_OPPONENT_RUNTIME_MODE,
    env_telemetry_stride: int = DEFAULT_ENV_TELEMETRY_STRIDE,
    env_manager_type: str = DEFAULT_ENV_MANAGER_TYPE,
    wait_for_train: bool = False,
    opponent_policy_kind: str = DEFAULT_OPPONENT_POLICY_KIND,
    opponent_use_cuda: bool = DEFAULT_OPPONENT_USE_CUDA,
    opponent_checkpoint_ref: str | None = None,
    opponent_mixture_spec: str | None = None,
    opponent_assignment_ref: str | None = None,
    initial_policy_checkpoint_ref: str | None = None,
    initial_policy_checkpoint_state_key: str | None = None,
    initial_policy_checkpoint_load_mode: str = DEFAULT_INITIAL_POLICY_CHECKPOINT_LOAD_MODE,
    opponent_assignment_refresh_interval_train_iter: int = (
        DEFAULT_OPPONENT_ASSIGNMENT_REFRESH_INTERVAL_TRAIN_ITER
    ),
    opponent_assignment_refresh_ref: str | None = None,
    own_checkpoint_opponent_refresh_enabled: bool = (
        DEFAULT_OWN_CHECKPOINT_OPPONENT_REFRESH_ENABLED
    ),
    opponent_assignment_json_path: str = "",
    opponent_assignment_audit_json_path: str = "",
    opponent_assignment_target_volume: str = "runs",
    mirror_assignment_checkpoints_to_control: bool = False,
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
    two_seat_verify_model_update_hash: bool = False,
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
    two_seat_action_noop_warmup_iterations: int = (TWO_SEAT_DEFAULT_ACTION_NOOP_WARMUP_ITERATIONS),
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
    two_seat_frozen_opponent_probability: float = (TWO_SEAT_DEFAULT_FROZEN_OPPONENT_PROBABILITY),
    two_seat_frozen_opponent_checkpoint_ref: str | None = None,
    two_seat_frozen_opponent_snapshot_ref: str | None = None,
    two_seat_frozen_opponent_checkpoint_state_key: str | None = None,
    two_seat_frozen_opponent_player_id: int = (TWO_SEAT_DEFAULT_FROZEN_OPPONENT_PLAYER_ID),
    two_seat_frozen_opponent_num_simulations: int | None = None,
    two_seat_frozen_opponent_use_cuda: bool | None = None,
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
    background_gif_collect_temperature: float | None = None,
    background_gif_collect_epsilon: float | None = None,
    exploration_bonus_mode: str = xb.EXPLORATION_BONUS_MODE_NONE,
    exploration_bonus_weight: float = 0.0,
    exploration_bonus_feature_source: str = xb.RND_FEATURE_SOURCE_POLICY_GRAY64_LATEST_V0,
    exploration_bonus_rnd_batch_size: int = 64,
    exploration_bonus_rnd_update_per_collect: int = xb.RND_DEFAULT_UPDATE_PER_COLLECT,
    exploration_bonus_rnd_buffer_size: int = 100_000,
    exploration_bonus_rnd_learning_rate: float = 3e-4,
    exploration_bonus_rnd_weight_decay: float = 1e-4,
    exploration_bonus_rnd_input_norm: bool = False,
    require_rnd_metrics: bool = False,
    output_detail: str = OUTPUT_DETAIL_COMPACT,
) -> None:
    if output_detail not in OUTPUT_DETAIL_CHOICES:
        raise ValueError(
            f"output_detail must be one of {OUTPUT_DETAIL_CHOICES!r}; got {output_detail!r}"
        )
    xb.normalize_exploration_bonus_spec(
        mode=exploration_bonus_mode,
        weight=exploration_bonus_weight,
        feature_source=exploration_bonus_feature_source,
        rnd_batch_size=exploration_bonus_rnd_batch_size,
        rnd_update_per_collect=exploration_bonus_rnd_update_per_collect,
        rnd_buffer_size=exploration_bonus_rnd_buffer_size,
        rnd_learning_rate=exploration_bonus_rnd_learning_rate,
        rnd_weight_decay=exploration_bonus_rnd_weight_decay,
        rnd_input_norm=exploration_bonus_rnd_input_norm,
    )
    resolved_background_gif_collect_temperature = (
        DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE
        if background_gif_collect_temperature is None
        else float(background_gif_collect_temperature)
    )
    resolved_background_gif_collect_epsilon = (
        DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON
        if background_gif_collect_epsilon is None
        else float(background_gif_collect_epsilon)
    )
    if profile_spawn and mode != "profile":
        raise ValueError("profile_spawn is only valid with mode='profile'")
    if mode == "write-assignment":
        if not opponent_assignment_json_path:
            raise ValueError("mode='write-assignment' requires opponent_assignment_json_path")
        assignment = json.loads(Path(opponent_assignment_json_path).read_text(encoding="utf-8"))
        audit = (
            json.loads(Path(opponent_assignment_audit_json_path).read_text(encoding="utf-8"))
            if opponent_assignment_audit_json_path
            else None
        )
        result = lightzero_curvytron_write_opponent_assignment_artifacts.remote(
            {
                "run_id": run_id,
                "attempt_id": attempt_id,
                "assignment": assignment,
                "audit": audit,
                "target_volume": opponent_assignment_target_volume,
                "mirror_checkpoints_to_control": mirror_assignment_checkpoints_to_control,
            }
        )
        print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
        return
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
                "two-seat self-play supports compute='cpu', 'gpu-l4-t4', or 'gpu-h100-cpu40'"
            )
        if two_seat_trail_render_mode not in STACK_RENDER_MODE_ORDER:
            raise ValueError(
                "two-seat self-play trail render mode must be one of "
                f"{STACK_RENDER_MODE_ORDER!r}; got {two_seat_trail_render_mode!r}"
            )
        two_seat_frozen_opponent_probability = float(two_seat_frozen_opponent_probability)
        if not 0.0 <= two_seat_frozen_opponent_probability <= 1.0:
            raise ValueError("two_seat_frozen_opponent_probability must be in [0, 1]")
        two_seat_frozen_checkpoint_input = (
            two_seat_frozen_opponent_checkpoint_ref or opponent_checkpoint_ref
        )
        if two_seat_frozen_opponent_probability > 0.0:
            if not two_seat_frozen_checkpoint_input:
                raise ValueError(
                    "two_seat_frozen_opponent_checkpoint_ref is required when "
                    "two_seat_frozen_opponent_probability > 0"
                )
            _reject_mutable_frozen_opponent_checkpoint_ref(two_seat_frozen_checkpoint_input)
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
            "verify_model_update_hash": two_seat_verify_model_update_hash,
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
            "terminal_outcome_reward_per_step": (two_seat_terminal_outcome_reward_per_step),
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
            "frozen_opponent_probability": two_seat_frozen_opponent_probability,
            "frozen_opponent_checkpoint_path": None,
            "frozen_opponent_checkpoint_ref": two_seat_frozen_checkpoint_input,
            "frozen_opponent_checkpoint_resolution": None,
            "frozen_opponent_snapshot_ref": two_seat_frozen_opponent_snapshot_ref,
            "frozen_opponent_checkpoint_state_key": (two_seat_frozen_opponent_checkpoint_state_key),
            "frozen_opponent_player_id": int(two_seat_frozen_opponent_player_id),
            "frozen_opponent_num_simulations": (two_seat_frozen_opponent_num_simulations),
            "frozen_opponent_use_cuda": two_seat_frozen_opponent_use_cuda,
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
            background_eval_poller_idle_after_done_sec=(background_eval_poller_idle_after_done_sec),
            background_gif_enabled=background_gif_enabled,
            background_gif_seed_offset=background_gif_seed_offset,
            background_gif_max_steps=background_gif_max_steps,
            background_gif_frame_stride=background_gif_frame_stride,
            background_gif_fps=background_gif_fps,
            background_gif_scale=background_gif_scale,
            background_gif_frame_size=background_gif_frame_size,
            background_gif_collect_temperature=background_gif_collect_temperature,
            background_gif_collect_epsilon=background_gif_collect_epsilon,
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
                        "schema_id": "curvyzero_experimental_two_seat_adapter_background_launch/v0",
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
                compact = compact_curvytron_two_seat_lightzero_train_smoke_summary(train_result)
                if "background_eval" in result:
                    compact["background_eval"] = result["background_eval"]
                result = compact
        print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
        return
    flat_a3_ctree_requested = (
        _validate_collect_search_ctree_backend(collect_search_ctree_backend)
        == COLLECT_SEARCH_CTREE_BACKEND_FLAT_A3
    )
    if compute == COMPUTE_CPU:
        if flat_a3_ctree_requested:
            raise ValueError(
                "collect_search_ctree_backend='flat_a3' is isolated to "
                "gpu-l4-t4-cpu40 and gpu-h100-cpu40 optimizer profile images"
            )
        train_fn = lightzero_curvytron_visual_survival_cpu
    elif compute == COMPUTE_CPU64:
        if flat_a3_ctree_requested:
            raise ValueError(
                "collect_search_ctree_backend='flat_a3' is isolated to "
                "gpu-l4-t4-cpu40 and gpu-h100-cpu40 optimizer profile images"
            )
        train_fn = lightzero_curvytron_visual_survival_cpu64
    elif compute == COMPUTE_GPU_L4_T4:
        if flat_a3_ctree_requested:
            raise ValueError(
                "collect_search_ctree_backend='flat_a3' uses the isolated CPU40 "
                "optimizer image; choose compute='gpu-l4-t4-cpu40' or "
                "'gpu-h100-cpu40'"
            )
        train_fn = lightzero_curvytron_visual_survival_gpu
    elif compute == COMPUTE_GPU_L4_T4_CPU40:
        train_fn = (
            lightzero_curvytron_visual_survival_gpu_cpu40_ctree_a3
            if flat_a3_ctree_requested
            else lightzero_curvytron_visual_survival_gpu_cpu40
        )
    elif compute == COMPUTE_GPU_H100_CPU40:
        train_fn = (
            lightzero_curvytron_visual_survival_h100_cpu40_ctree_a3
            if flat_a3_ctree_requested
            else lightzero_curvytron_visual_survival_h100_cpu40
        )
    elif compute == COMPUTE_GPU_H100X2_CPU40:
        if flat_a3_ctree_requested:
            raise ValueError(
                "collect_search_ctree_backend='flat_a3' is isolated to the "
                "single-GPU optimizer images for now"
            )
        train_fn = lightzero_curvytron_visual_survival_h100x2_cpu40
    else:
        raise ValueError(f"unknown compute {compute!r}; expected one of {COMPUTE_CHOICES!r}")
    opponent_policy_kind = _normalize_opponent_policy_kind_for_env(
        env_variant=env_variant,
        opponent_policy_kind=opponent_policy_kind,
        opponent_checkpoint_ref=opponent_checkpoint_ref,
    )
    source_state_trail_render_mode = _validate_source_state_trail_render_mode(
        source_state_trail_render_mode
    )
    source_state_bonus_render_mode = _validate_source_state_bonus_render_mode(
        source_state_bonus_render_mode
    )
    policy_observation_backend = _validate_policy_observation_backend(policy_observation_backend)
    collect_search_backend = _validate_collect_search_backend(collect_search_backend)
    collect_search_ctree_backend = _validate_collect_search_ctree_backend(
        collect_search_ctree_backend
    )
    learner_seat_mode = _validate_learner_seat_mode(learner_seat_mode)
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
        "model_support_cap": model_support_cap,
        "td_steps": td_steps,
        "lightzero_eval_freq": lightzero_eval_freq,
        "skip_lightzero_eval_in_profile": skip_lightzero_eval_in_profile,
        "profile_cuda_sync_enabled": profile_cuda_sync_enabled,
        "profile_allow_auto_resume": profile_allow_auto_resume,
        "profile_volume_commit": profile_volume_commit,
        "lightzero_multi_gpu": lightzero_multi_gpu,
        "save_ckpt_after_iter": save_ckpt_after_iter,
        "commit_on_checkpoint": commit_on_checkpoint,
        "stop_after_learner_train_calls": stop_after_learner_train_calls,
        "env_variant": env_variant,
        "reward_variant": reward_variant,
        "reward_outcome_alpha": reward_outcome_alpha,
        "source_state_trail_render_mode": source_state_trail_render_mode,
        "source_state_bonus_render_mode": source_state_bonus_render_mode,
        "policy_observation_backend": policy_observation_backend,
        "collect_search_backend": collect_search_backend,
        "collect_search_ctree_backend": collect_search_ctree_backend,
        "learner_seat_mode": learner_seat_mode,
        "ego_action_straight_override_probability": ego_action_straight_override_probability,
        "policy_action_repeat_min": policy_action_repeat_min,
        "policy_action_repeat_max": policy_action_repeat_max,
        "policy_action_repeat_extra_probability": policy_action_repeat_extra_probability,
        "control_noise_profile_id": control_noise_profile_id,
        "disable_death_for_profile": disable_death_for_profile,
        "opponent_death_mode": opponent_death_mode,
        "opponent_runtime_mode": opponent_runtime_mode,
        "env_telemetry_stride": env_telemetry_stride,
        "env_manager_type": env_manager_type,
        "opponent_policy_kind": opponent_policy_kind,
        "opponent_use_cuda": opponent_use_cuda,
        "opponent_checkpoint_ref": opponent_checkpoint_ref,
        "opponent_mixture_spec": opponent_mixture_spec,
        "opponent_assignment_ref": opponent_assignment_ref,
        "initial_policy_checkpoint_ref": initial_policy_checkpoint_ref,
        "initial_policy_checkpoint_state_key": initial_policy_checkpoint_state_key,
        "initial_policy_checkpoint_load_mode": initial_policy_checkpoint_load_mode,
        "opponent_snapshot_ref": snapshot_ref,
        "opponent_checkpoint_report_ref": checkpoint_ref,
        "opponent_checkpoint_state_key": state_key,
        "opponent_assignment_refresh_interval_train_iter": (
            opponent_assignment_refresh_interval_train_iter
        ),
        "opponent_assignment_refresh_ref": opponent_assignment_refresh_ref,
        "own_checkpoint_opponent_refresh_enabled": own_checkpoint_opponent_refresh_enabled,
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
        "background_gif_collect_temperature": resolved_background_gif_collect_temperature,
        "background_gif_collect_epsilon": resolved_background_gif_collect_epsilon,
        "exploration_bonus_mode": exploration_bonus_mode,
        "exploration_bonus_weight": exploration_bonus_weight,
        "exploration_bonus_feature_source": exploration_bonus_feature_source,
        "exploration_bonus_rnd_batch_size": exploration_bonus_rnd_batch_size,
        "exploration_bonus_rnd_update_per_collect": exploration_bonus_rnd_update_per_collect,
        "exploration_bonus_rnd_buffer_size": exploration_bonus_rnd_buffer_size,
        "exploration_bonus_rnd_learning_rate": exploration_bonus_rnd_learning_rate,
        "exploration_bonus_rnd_weight_decay": exploration_bonus_rnd_weight_decay,
        "exploration_bonus_rnd_input_norm": exploration_bonus_rnd_input_norm,
        "require_rnd_metrics": require_rnd_metrics,
    }
    exp_name_ref = (
        runs.attempt_train_ref(TASK_ID, run_id, attempt_id) / "lightzero_exp"
    ).as_posix()
    poller_call = None
    poller_call_id = None
    train_kwargs = dict(kwargs)
    if (
        mode == "train"
        and background_eval_enabled
        and background_eval_launch_kind == BACKGROUND_EVAL_LAUNCH_POLLER
    ):
        train_kwargs["background_eval_launch_kind"] = BACKGROUND_EVAL_LAUNCH_POLLER
        poller_call = lightzero_curvytron_visual_survival_checkpoint_eval_poller.spawn(
            run_id=run_id,
            attempt_id=attempt_id,
            exp_name_ref=exp_name_ref,
            seed=seed,
            source_max_steps=source_max_steps,
            decision_ms=decision_ms,
            decision_source_frames=DEFAULT_DECISION_SOURCE_FRAMES,
            source_physics_step_ms=DEFAULT_SOURCE_PHYSICS_STEP_MS,
            source_max_steps_semantics="source_physics_steps",
            env_variant=env_variant,
            reward_variant=reward_variant,
            opponent_policy_kind=opponent_policy_kind,
            opponent_checkpoint_ref=opponent_checkpoint_ref,
            opponent_snapshot_ref=snapshot_ref,
            opponent_checkpoint_state_key=state_key,
            opponent_assignment_ref=opponent_assignment_ref,
            opponent_death_mode=opponent_death_mode,
            opponent_runtime_mode=opponent_runtime_mode,
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
            background_gif_collect_temperature=resolved_background_gif_collect_temperature,
            background_gif_collect_epsilon=resolved_background_gif_collect_epsilon,
            poll_interval_sec=background_eval_poll_interval_sec,
            stable_polls=background_eval_poll_stable_polls,
            max_runtime_sec=background_eval_poller_max_runtime_sec,
            idle_after_train_done_sec=background_eval_poller_idle_after_done_sec,
        )
        poller_call_id = getattr(poller_call, "object_id", None) or getattr(poller_call, "id", None)
    if mode == "profile" and profile_spawn:
        call = train_fn.spawn(**train_kwargs)
        call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
        print(
            json.dumps(
                {
                    "schema_id": "curvyzero_lightzero_curvytron_profile_spawn/v0",
                    "status": "spawned",
                    "mode": mode,
                    "compute": compute,
                    "seed": seed,
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "function_call_id": call_id,
                    "result_readback": "modal.FunctionCall.from_id(function_call_id).get()",
                    "summary_ref": (
                        runs.attempt_train_ref(TASK_ID, run_id, attempt_id) / "summary.json"
                    ).as_posix(),
                    "summary_ref_status": (
                        "best_effort_volume_artifact_not_profile_source_of_truth"
                    ),
                    "command": kwargs,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
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
