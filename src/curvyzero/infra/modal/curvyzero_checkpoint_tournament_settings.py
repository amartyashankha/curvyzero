"""Shared settings for the CurvyTron checkpoint tournament Modal app."""

from __future__ import annotations

from pathlib import Path

from curvyzero.contracts.curvytron import (
    CURVYTRON_TRAINING_TASK_ID,
    curvytron_checkpoint_intake_dict_name,
    curvytron_checkpoint_intake_queue_name,
    curvytron_current_rating_run_id,
    curvytron_current_tournament_id,
    curvytron_opponent_leaderboard_dict_name,
    curvytron_runs_volume_name,
    curvytron_tournament_app_name,
    curvytron_tournament_volume_name,
)


APP_NAME = curvytron_tournament_app_name()
CHECKPOINT_VOLUME_NAME = curvytron_runs_volume_name()
TOURNAMENT_VOLUME_NAME = curvytron_tournament_volume_name()
LIGHTZERO_VERSION = "0.2.0"
REMOTE_ROOT = Path("/repo")
RUNS_MOUNT = Path("/runs")
TOURNAMENT_MOUNT = Path("/tournament-runs")
CURVYTRON_BONUS_SPRITE_SHEET_RELATIVE_PATH = (
    "third_party/curvytron-reference/web/images/bonus.png"
)
DEFAULT_LIMIT = 50
MAX_LIMIT = 500
DEFAULT_BATTLE_GAME_LIMIT = 50
GIF_CACHE_MAX_AGE_SECONDS = 86_400
DYNAMIC_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}
WEB_PAGE_RELOAD_MIN_INTERVAL_SECONDS = 30.0
WEB_PROGRESS_RELOAD_MIN_INTERVAL_SECONDS = 60.0
WEB_PROGRESS_CACHE_TTL_SECONDS = 5.0
WEB_BATTLE_DETAIL_CACHE_TTL_SECONDS = 30.0
WEB_BATTLE_DETAIL_CACHE_VERSION = "games-page-v2"
WEB_PROVISIONAL_RATING_CACHE_TTL_SECONDS = 30.0
WEB_GIF_BYTES_CACHE_TTL_SECONDS = 300.0
WEB_GIF_BYTES_CACHE_MAX_ITEM_BYTES = 24 * 1024 * 1024
DEFAULT_PROVISIONAL_RATING_INTERVAL_SECONDS = 60.0
DEFAULT_PROVISIONAL_RATING_MAX_SECONDS = 23 * 60 * 60
TRAINING_TASK_ID = CURVYTRON_TRAINING_TASK_ID
CURRENT_TOURNAMENT_ID = curvytron_current_tournament_id()
CURRENT_RATING_RUN_ID = curvytron_current_rating_run_id()
CHECKPOINT_INTAKE_DICT_NAME = curvytron_checkpoint_intake_dict_name()
CHECKPOINT_INTAKE_QUEUE_NAME = curvytron_checkpoint_intake_queue_name()
CHECKPOINT_INTAKE_ACTIVE_KEYS = "active_manifest_keys"
OPPONENT_LEADERBOARD_DICT_NAME = curvytron_opponent_leaderboard_dict_name()
DEFAULT_CHECKPOINT_INTAKE_SCAN_SECONDS = 10
DEFAULT_CHECKPOINT_INTAKE_QUEUE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_CHECKPOINT_INTAKE_CLAIM_STALE_SECONDS = 24 * 60 * 60
