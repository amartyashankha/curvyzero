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

DEFAULT_CURVYTRON_TRAIN_APP_NAME = "curvyzero-lightzero-curvytron-visual-survival-train-v2"
DEFAULT_CURVYTRON_TOURNAMENT_APP_NAME = "curvyzero-checkpoint-tournament-v2"
DEFAULT_CURVYTRON_GIF_BROWSER_APP_NAME = "curvyzero-curvytron-gif-browser-v2"
DEFAULT_CURVYTRON_CURRENT_TOURNAMENT_ID = "cz26-live-20260517a"
DEFAULT_CURVYTRON_CURRENT_RATING_RUN_ID = "elo-cz26-live-20260517a"
DEFAULT_CURVYTRON_CURRENT_GIF_RUN_PREFIXES = (
    "rnd-blank-current-",
    "rnd-blank-sweep-fastckpt-20260519a-",
)
DEFAULT_CURVYTRON_TRAINING_CANDIDATE_ASSIGNMENT_BANK_RUN_ID = (
    "cz26-training-candidates"
)
DEFAULT_CURVYTRON_TRAINING_CANDIDATE_ASSIGNMENT_BANK_ATTEMPT_ID = (
    "try-cz26-training-candidates"
)
DEFAULT_CURVYTRON_TRAINING_CANDIDATE_REFRESH_CONFIG_REF = (
    "control:training/lightzero-curvytron-visual-survival/cz26-control/"
    "attempts/try-cz26-control/opponents/"
    "training_candidate_refresh_config.json"
)
_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT = (
    "control:training/lightzero-curvytron-visual-survival/cz26-control/"
    "attempts/try-cz26-control/opponents/refresh_pointers"
)
DEFAULT_CURVYTRON_TRAINING_CANDIDATE_REFRESH_POINTERS = (
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b100/imm0/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b100/imm10/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b10w05r1/imm0/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b10w05r1/imm10/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b20w05lad4/imm0/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b20w05lad4/imm10/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b20w05r1/imm0/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b20w05r1/imm10/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b20w05top2/imm0/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b20w05top2/imm10/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b20w10r1/imm0/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b20w10r1/imm10/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b20w20lad4s/imm0/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b20w20lad4s/imm10/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b25w25r1/imm0/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b25w25r1/imm10/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b30w05r1/imm0/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b30w05r1/imm10/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b50r1/imm0/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/b50r1/imm10/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/r1/imm0/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/r1/imm10/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/w100/imm0/refresh_pointer.json",
    f"{_CZ26_TRAINING_CANDIDATE_REFRESH_POINTER_ROOT}/w100/imm10/refresh_pointer.json",
)
DEFAULT_CURVYTRON_TRAINING_CANDIDATE_ASSIGNMENT_SEED = 20260516
DEFAULT_CURVYTRON_TRAINING_CANDIDATE_MIN_ACTIVE_COUNT = 100
DEFAULT_CURVYTRON_TRAINING_CANDIDATE_MAX_ACTIVE_RANK = 100
DEFAULT_CURVYTRON_TRAINING_CANDIDATE_ACTIVE_MIN_VALID_GAMES = 21
DEFAULT_CURVYTRON_TRAINING_CANDIDATE_ACTIVE_MIN_DISTINCT_OPPONENTS = 1
DEFAULT_CURVYTRON_TRAINING_CANDIDATE_REFRESH_SECONDS = 30 * 60
DEFAULT_CURVYTRON_CHECKPOINT_INTAKE_SCAN_SECONDS = 10

DEFAULT_CURVYTRON_CHECKPOINT_INTAKE_DICT_NAME = "curvyzero-curvytron-checkpoint-intake-v2"
DEFAULT_CURVYTRON_CHECKPOINT_INTAKE_QUEUE_NAME = "curvyzero-curvytron-checkpoint-events-v2"
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
CURVYTRON_TOURNAMENT_GIF_FPS = 800.0
CURVYTRON_TOURNAMENT_GIF_MIN_FRAME_DURATION_MS = 1
CURVYTRON_DEFAULT_MAX_ENV_STEP = 30_000_000
CURVYTRON_DEFAULT_MAX_TRAIN_ITER = 300_000
CURVYTRON_DEFAULT_NUM_SIMULATIONS = 8
CURVYTRON_DEFAULT_COLLECTOR_ENV_NUM = 256
CURVYTRON_DEFAULT_N_EPISODE = 256
CURVYTRON_DEFAULT_TRAIN_BATCH_SIZE = 64
CURVYTRON_DEFAULT_POLICY_BATCH_SIZE = 8

CURVYTRON_POLICY_TRAIL_RENDER_MODE = POLICY_TRAIL_RENDER_MODE
CURVYTRON_POLICY_BONUS_RENDER_MODE = POLICY_BONUS_RENDER_MODE

COMPUTE_L4_T4_CPU40 = "gpu-l4-t4-cpu40"
COMPUTE_H100_CPU40 = "gpu-h100-cpu40"
CURVYTRON_DEFAULT_TRAIN_COMPUTE = COMPUTE_L4_T4_CPU40

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
        "reward_outcome_alpha",
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
    if volume_name.startswith("curvyzero-") and not volume_name.endswith("-v2"):
        raise ValueError(
            f"CurvyTron launch code must use all-v2 Modal Volumes; got {volume_name!r}"
        )
    kwargs: dict[str, Any] = {"create_if_missing": create_if_missing}
    volume_fs_version = CURVYTRON_MODAL_VOLUME_FS_VERSION_BY_NAME.get(volume_name)
    if volume_name.endswith("-v2"):
        kwargs["version"] = volume_fs_version or CURVYTRON_VOLUME_FS_VERSION
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


def curvytron_current_gif_run_prefixes() -> tuple[str, ...]:
    value = os.environ.get("CURVYZERO_CURRENT_GIF_RUN_PREFIXES", "").strip()
    if not value:
        return DEFAULT_CURVYTRON_CURRENT_GIF_RUN_PREFIXES
    prefixes = tuple(item.strip() for item in value.split(",") if item.strip())
    if not prefixes:
        return DEFAULT_CURVYTRON_CURRENT_GIF_RUN_PREFIXES
    return prefixes


def curvytron_training_candidate_assignment_bank_run_id() -> str:
    return env_name(
        "CURVYZERO_TRAINING_CANDIDATE_ASSIGNMENT_BANK_RUN_ID",
        DEFAULT_CURVYTRON_TRAINING_CANDIDATE_ASSIGNMENT_BANK_RUN_ID,
    )


def curvytron_training_candidate_assignment_bank_attempt_id() -> str:
    return env_name(
        "CURVYZERO_TRAINING_CANDIDATE_ASSIGNMENT_BANK_ATTEMPT_ID",
        DEFAULT_CURVYTRON_TRAINING_CANDIDATE_ASSIGNMENT_BANK_ATTEMPT_ID,
    )


def curvytron_training_candidate_refresh_pointers() -> tuple[str, ...]:
    value = os.environ.get("CURVYZERO_TRAINING_CANDIDATE_REFRESH_POINTERS", "").strip()
    if not value:
        return DEFAULT_CURVYTRON_TRAINING_CANDIDATE_REFRESH_POINTERS
    pointers = tuple(item.strip() for item in value.split(",") if item.strip())
    if not pointers:
        return DEFAULT_CURVYTRON_TRAINING_CANDIDATE_REFRESH_POINTERS
    return pointers


def curvytron_training_candidate_refresh_config_ref() -> str:
    return env_name(
        "CURVYZERO_TRAINING_CANDIDATE_REFRESH_CONFIG_REF",
        DEFAULT_CURVYTRON_TRAINING_CANDIDATE_REFRESH_CONFIG_REF,
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
