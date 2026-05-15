"""One source of truth for the current CurvyTron training/tournament lane."""

from __future__ import annotations

import os
from typing import Any

from curvyzero.env.observation_surface_contract import (
    POLICY_BONUS_RENDER_MODE,
    POLICY_TRAIL_RENDER_MODE,
)
from curvyzero.env.vector_multiplayer_env import SOURCE_PHYSICS_STEP_MS


CURVYTRON_TRAINING_TASK_ID = "lightzero-curvytron-visual-survival"
CURVYTRON_TOURNAMENT_TASK_ID = "curvytron-checkpoint-tournament"

CURVYTRON_VOLUME_FS_VERSION = 2
DEFAULT_CURVYTRON_RUNS_VOLUME_NAME = "curvyzero-runs-v2"
DEFAULT_CURVYTRON_TOURNAMENT_VOLUME_NAME = "curvyzero-curvytron-tournaments-v2"
DEFAULT_CURVYTRON_CONTROL_VOLUME_NAME = "curvyzero-curvytron-control-v2"
CURVYTRON_MODAL_VOLUME_FS_VERSION_BY_NAME: dict[str, int] = {
    DEFAULT_CURVYTRON_RUNS_VOLUME_NAME: CURVYTRON_VOLUME_FS_VERSION,
    DEFAULT_CURVYTRON_TOURNAMENT_VOLUME_NAME: CURVYTRON_VOLUME_FS_VERSION,
    DEFAULT_CURVYTRON_CONTROL_VOLUME_NAME: CURVYTRON_VOLUME_FS_VERSION,
}

DEFAULT_CURVYTRON_TRAIN_APP_NAME = (
    "curvyzero-lightzero-curvytron-visual-survival-train-v2"
)
DEFAULT_CURVYTRON_TOURNAMENT_APP_NAME = "curvyzero-checkpoint-tournament-v2"
DEFAULT_CURVYTRON_GIF_BROWSER_APP_NAME = "curvyzero-curvytron-gif-browser-v2"
DEFAULT_CURVYTRON_CURRENT_TOURNAMENT_ID = "curvy-e2e-allv2-canary-live-20260515a"
DEFAULT_CURVYTRON_CURRENT_RATING_RUN_ID = "elo-e2e-allv2-canary-live-20260515a"

DEFAULT_CURVYTRON_CHECKPOINT_INTAKE_DICT_NAME = (
    "curvyzero-curvytron-checkpoint-intake-v2"
)
DEFAULT_CURVYTRON_CHECKPOINT_INTAKE_QUEUE_NAME = (
    "curvyzero-curvytron-checkpoint-events-v2"
)
DEFAULT_CURVYTRON_OPPONENT_LEADERBOARD_DICT_NAME = (
    "curvyzero-curvytron-opponent-leaderboard-live-v2"
)

CURVYTRON_DECISION_SOURCE_FRAMES = 1
CURVYTRON_DECISION_MS = float(SOURCE_PHYSICS_STEP_MS * CURVYTRON_DECISION_SOURCE_FRAMES)
CURVYTRON_SOURCE_MAX_STEPS = 1_048_576
CURVYTRON_SAVE_CKPT_AFTER_ITER = 10_000
CURVYTRON_COMMIT_ON_CHECKPOINT = True
CURVYTRON_ASSIGNMENT_REFRESH_INTERVAL_TRAIN_ITER = 2_000
CURVYTRON_BACKGROUND_GIF_FPS = 80.0
CURVYTRON_DEFAULT_NUM_SIMULATIONS = 8
CURVYTRON_DEFAULT_POLICY_BATCH_SIZE = 8

CURVYTRON_POLICY_TRAIL_RENDER_MODE = POLICY_TRAIL_RENDER_MODE
CURVYTRON_POLICY_BONUS_RENDER_MODE = POLICY_BONUS_RENDER_MODE

COMPUTE_H100_CPU40 = "gpu-h100-cpu40"

LEARNER_SEAT_MODE_FIXED_PLAYER_0 = "fixed_player_0"
LEARNER_SEAT_MODE_FIXED_PLAYER_1 = "fixed_player_1"
LEARNER_SEAT_MODE_RANDOM_PER_EPISODE = "random_per_episode"
LEARNER_SEAT_MODE_CHOICES = (
    LEARNER_SEAT_MODE_FIXED_PLAYER_0,
    LEARNER_SEAT_MODE_FIXED_PLAYER_1,
    LEARNER_SEAT_MODE_RANDOM_PER_EPISODE,
)
DEFAULT_LEARNER_SEAT_MODE = LEARNER_SEAT_MODE_RANDOM_PER_EPISODE

REWARD_VARIANT_SPARSE_OUTCOME = "sparse_outcome"
REWARD_VARIANT_DENSE_SURVIVAL_PLUS_OUTCOME = "dense_survival_plus_outcome"
REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME = "survival_plus_bonus_no_outcome"
REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME = "survival_plus_bonus_plus_outcome"
REWARD_VARIANT_ALL_PLAYERS_ALIVE_DIAGNOSTIC = "all_players_alive_diagnostic"
CURVYTRON_MAIN_REWARD_VARIANTS = (
    REWARD_VARIANT_SPARSE_OUTCOME,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_PLUS_OUTCOME,
)

TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT: tuple[str, ...] = (
    "mode",
    "seed",
    "run_id",
    "attempt_id",
    "max_env_step",
    "max_train_iter",
    "source_max_steps",
    "decision_ms",
    "collector_env_num",
    "evaluator_env_num",
    "n_evaluator_episode",
    "n_episode",
    "num_simulations",
    "batch_size",
    "lightzero_eval_freq",
    "skip_lightzero_eval_in_profile",
    "profile_cuda_sync_enabled",
    "profile_allow_auto_resume",
    "profile_volume_commit",
    "lightzero_multi_gpu",
    "save_ckpt_after_iter",
    "commit_on_checkpoint",
    "stop_after_learner_train_calls",
    "env_variant",
    "reward_variant",
    "source_state_trail_render_mode",
    "source_state_bonus_render_mode",
    "learner_seat_mode",
    "ego_action_straight_override_probability",
    "policy_action_repeat_min",
    "policy_action_repeat_max",
    "policy_action_repeat_extra_probability",
    "control_noise_profile_id",
    "disable_death_for_profile",
    "opponent_death_mode",
    "opponent_runtime_mode",
    "env_telemetry_stride",
    "env_manager_type",
    "opponent_policy_kind",
    "opponent_use_cuda",
    "opponent_checkpoint_ref",
    "opponent_snapshot_ref",
    "opponent_checkpoint_report_ref",
    "opponent_checkpoint_state_key",
    "initial_policy_checkpoint_ref",
    "initial_policy_checkpoint_state_key",
    "initial_policy_checkpoint_load_mode",
    "background_eval_enabled",
    "background_eval_launch_kind",
    "background_eval_compute",
    "background_eval_id_prefix",
    "background_eval_seed_count",
    "background_eval_seed_rng_seed",
    "background_eval_max_steps",
    "background_eval_step_detail_limit",
    "background_eval_num_simulations",
    "background_eval_batch_size",
    "background_gif_enabled",
    "background_gif_seed_offset",
    "background_gif_max_steps",
    "background_gif_frame_stride",
    "background_gif_fps",
    "background_gif_scale",
    "background_gif_frame_size",
    "background_gif_collect_temperature",
    "background_gif_collect_epsilon",
)

POLLER_KWARGS_ALLOWED_FOR_GROUPED_SUBMIT: frozenset[str] = frozenset(
    {
        "run_id",
        "attempt_id",
        "exp_name_ref",
        "seed",
        "source_max_steps",
        "env_variant",
        "reward_variant",
        "opponent_policy_kind",
        "opponent_checkpoint_ref",
        "opponent_snapshot_ref",
        "opponent_checkpoint_state_key",
        "opponent_mixture_spec",
        "opponent_assignment_ref",
        "opponent_death_mode",
        "opponent_runtime_mode",
        "background_eval_compute",
        "background_eval_id_prefix",
        "background_eval_seed_count",
        "background_eval_seed_rng_seed",
        "background_eval_max_steps",
        "background_eval_step_detail_limit",
        "background_eval_num_simulations",
        "background_eval_batch_size",
        "background_gif_enabled",
        "background_gif_seed_offset",
        "background_gif_max_steps",
        "background_gif_frame_stride",
        "background_gif_fps",
        "background_gif_scale",
        "background_gif_frame_size",
        "background_gif_collect_temperature",
        "background_gif_collect_epsilon",
        "poll_interval_sec",
        "stable_polls",
        "max_runtime_sec",
        "idle_after_train_done_sec",
    }
)


def env_name(env_var: str, default: str) -> str:
    value = os.environ.get(env_var, "").strip()
    return value or default


def current_v2_object_name(env_var: str, default: str) -> str:
    value = env_name(env_var, default)
    if not value.endswith("-v2"):
        raise ValueError(
            f"{env_var} must point at the current all-v2 CurvyTron object; got {value!r}"
        )
    return value


def modal_volume_kwargs_for_name(
    volume_name: str,
    *,
    create_if_missing: bool = True,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"create_if_missing": create_if_missing}
    volume_fs_version = CURVYTRON_MODAL_VOLUME_FS_VERSION_BY_NAME.get(volume_name)
    if volume_fs_version is not None:
        kwargs["version"] = volume_fs_version
    return kwargs


def curvytron_runs_volume_name() -> str:
    return current_v2_object_name(
        "CURVYZERO_RUNS_VOLUME_NAME",
        DEFAULT_CURVYTRON_RUNS_VOLUME_NAME,
    )


def curvytron_tournament_volume_name() -> str:
    return current_v2_object_name(
        "CURVYZERO_TOURNAMENT_VOLUME_NAME",
        DEFAULT_CURVYTRON_TOURNAMENT_VOLUME_NAME,
    )


def curvytron_control_volume_name() -> str:
    return current_v2_object_name(
        "CURVYZERO_CONTROL_VOLUME_NAME",
        DEFAULT_CURVYTRON_CONTROL_VOLUME_NAME,
    )


def curvytron_train_app_name() -> str:
    return current_v2_object_name(
        "CURVYZERO_TRAIN_APP_NAME",
        DEFAULT_CURVYTRON_TRAIN_APP_NAME,
    )


def curvytron_tournament_app_name() -> str:
    return current_v2_object_name(
        "CURVYZERO_TOURNAMENT_APP_NAME",
        DEFAULT_CURVYTRON_TOURNAMENT_APP_NAME,
    )


def curvytron_gif_browser_app_name() -> str:
    return current_v2_object_name(
        "CURVYZERO_GIF_BROWSER_APP_NAME",
        DEFAULT_CURVYTRON_GIF_BROWSER_APP_NAME,
    )


def curvytron_current_tournament_id() -> str:
    return env_name(
        "CURVYZERO_CURRENT_TOURNAMENT_ID",
        DEFAULT_CURVYTRON_CURRENT_TOURNAMENT_ID,
    )


def curvytron_current_rating_run_id() -> str:
    return env_name(
        "CURVYZERO_CURRENT_RATING_RUN_ID",
        DEFAULT_CURVYTRON_CURRENT_RATING_RUN_ID,
    )


def curvytron_checkpoint_intake_dict_name() -> str:
    return current_v2_object_name(
        "CURVYZERO_CHECKPOINT_INTAKE_DICT_NAME",
        DEFAULT_CURVYTRON_CHECKPOINT_INTAKE_DICT_NAME,
    )


def curvytron_checkpoint_intake_queue_name() -> str:
    return current_v2_object_name(
        "CURVYZERO_CHECKPOINT_INTAKE_QUEUE_NAME",
        DEFAULT_CURVYTRON_CHECKPOINT_INTAKE_QUEUE_NAME,
    )


def curvytron_opponent_leaderboard_dict_name() -> str:
    return current_v2_object_name(
        "CURVYZERO_OPPONENT_LEADERBOARD_DICT_NAME",
        DEFAULT_CURVYTRON_OPPONENT_LEADERBOARD_DICT_NAME,
    )
