"""Named contracts for CurvyTron checkpoint tournaments."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from curvyzero.env.vector_multiplayer_env import (
    DEFAULT_DECISION_SOURCE_FRAMES,
    SOURCE_PHYSICS_STEP_MS,
)
from curvyzero.infra.modal import run_management as runs


TOURNAMENT_TASK_ID = "curvytron-checkpoint-tournament"
TOURNAMENT_BASE_REF = PurePosixPath("tournaments") / "curvytron"
TOURNAMENT_RUN_MARKER_FILENAME = "show_in_tournament_browser.flag"
CHECKPOINT_SELECTION_LATEST = "latest"
CHECKPOINT_SELECTION_ALL = "all"
CHECKPOINT_SELECTION_ITERATION = "iteration"
CHECKPOINT_SELECTION_CHOICES = (
    CHECKPOINT_SELECTION_LATEST,
    CHECKPOINT_SELECTION_ALL,
    CHECKPOINT_SELECTION_ITERATION,
)
CHECKPOINT_SCAN_GLOB = "train/lightzero_exp*/ckpt/iteration_*.pth.tar"
CHECKPOINT_EXP_CKPT_DIR_GLOB = "lightzero_exp*/ckpt"
CHECKPOINT_WEIGHT_FILENAME_GLOB = "iteration_*.pth.tar"

TOURNAMENT_SCHEMA_ID = "curvyzero_curvytron_checkpoint_tournament/v0"
BATTLE_SCHEMA_ID = "curvyzero_curvytron_checkpoint_tournament_battle/v0"
GAME_SCHEMA_ID = "curvyzero_curvytron_checkpoint_tournament_game/v0"
GAME_SHARD_SCHEMA_ID = "curvyzero_curvytron_checkpoint_tournament_game_shard/v0"
RATING_CONFIG_SCHEMA_ID = "curvyzero_curvytron_checkpoint_rating_config/v0"
RATING_ROUND_SCHEMA_ID = "curvyzero_curvytron_checkpoint_rating_round/v0"
RATING_SNAPSHOT_SCHEMA_ID = "curvyzero_curvytron_checkpoint_rating_snapshot/v0"
RATING_PROGRESS_SCHEMA_ID = "curvyzero_curvytron_checkpoint_rating_progress/v0"
RATING_SCHEDULER_STATE_SCHEMA_ID = "curvyzero_curvytron_checkpoint_rating_scheduler_state/v0"
PAIR_HISTORY_SCHEMA_ID = "curvyzero_curvytron_checkpoint_rating_pair_history/v0"
RATING_FORMULA_VERSION = "batch_elo_v1"

POLICY_MODE_EVAL = "eval"
POLICY_MODE_COLLECT = "collect"
POLICY_MODE_CHOICES = (POLICY_MODE_EVAL, POLICY_MODE_COLLECT)

DEFAULT_GAMES_PER_PAIR = 21
DEFAULT_GAMES_PER_SHARD = 1
DEFAULT_REUSE_POLICIES_PER_SHARD = True
DEFAULT_MAX_STEPS = 8_000
DEFAULT_SOURCE_PHYSICS_STEP_MS = float(SOURCE_PHYSICS_STEP_MS)
DEFAULT_DECISION_MS = float(DEFAULT_DECISION_SOURCE_FRAMES * DEFAULT_SOURCE_PHYSICS_STEP_MS)
DEFAULT_NUM_SIMULATIONS = 8
DEFAULT_POLICY_BATCH_SIZE = 8
DEFAULT_COLLECT_TEMPERATURE = 1.0
DEFAULT_COLLECT_EPSILON = 0.25
DEFAULT_FRAME_STRIDE = 1
DEFAULT_GIF_FPS = 8.0
DEFAULT_FRAME_SIZE = 704
DEFAULT_GIF_TRAIL_RENDER_MODE = "browser_lines"
DEFAULT_SAVE_GIF = False
DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR = 1
DEFAULT_GIF_SAMPLE_STRATEGY = "evenly_spaced"
GIF_SAMPLE_STRATEGY_CHOICES = ("first_n", "evenly_spaced")
DEFAULT_SAVE_FRAMES_NPZ = False
DEFAULT_ORDERED_PAIRS = False
DEFAULT_INCLUDE_SELF_PAIRS = False
DEFAULT_RATING_RUN_ID = "elo"
DEFAULT_RATING_ROUND_COUNT = 1
RATING_PAIR_SELECTION_ALL_PAIRS = "all_pairs"
RATING_PAIR_SELECTION_RANDOM = "random"
RATING_PAIR_SELECTION_ADAPTIVE_V0 = "adaptive_v0"
DEFAULT_RATING_PAIR_SELECTION = RATING_PAIR_SELECTION_ALL_PAIRS
DEFAULT_RATING_INITIAL_RATING = 1500.0
DEFAULT_RATING_BASE_K = 32.0
DEFAULT_RATING_K_REFERENCE_GAMES = 50.0
DEFAULT_RATING_K_MIN = 16.0
DEFAULT_RATING_K_MAX = 64.0
DEFAULT_RATING_DELTA_CLAMP = 80.0
DEFAULT_RATING_DRAW_SCORE = 0.5
DEFAULT_RATING_MIN_VALID_FRACTION = 0.8
DEFAULT_RATING_STOP_MAX_DELTA = 10.0
DEFAULT_RATING_ACTIVE_POOL_LIMIT = 100
RATING_PAIR_SELECTION_CHOICES = (
    RATING_PAIR_SELECTION_ALL_PAIRS,
    RATING_PAIR_SELECTION_RANDOM,
    RATING_PAIR_SELECTION_ADAPTIVE_V0,
)

SCHEDULE_REASON_PLACEMENT = "placement"
SCHEDULE_REASON_NEAR_RATING = "near_rating"
SCHEDULE_REASON_UNCERTAIN = "uncertain"
SCHEDULE_REASON_RANDOM_BRIDGE = "random_bridge"
SCHEDULE_REASON_FILL = "fill"
SCHEDULE_REASON_CHOICES = (
    SCHEDULE_REASON_PLACEMENT,
    SCHEDULE_REASON_NEAR_RATING,
    SCHEDULE_REASON_UNCERTAIN,
    SCHEDULE_REASON_RANDOM_BRIDGE,
    SCHEDULE_REASON_FILL,
)

ARTIFACT_GAME_GIF_FILENAME = "game.gif"
ARTIFACT_FRAMES_NPZ_FILENAME = "frames.npz"
ARTIFACT_SUMMARY_FILENAME = "summary.json"
ARTIFACT_BATTLE_SUMMARY_FILENAME = "battle.json"
ARTIFACT_PAIR_SPEC_FILENAME = "pair_spec.json"
ARTIFACT_TOURNAMENT_MANIFEST_FILENAME = "tournament.json"
ARTIFACT_TOURNAMENT_STANDINGS_FILENAME = "standings.json"
ARTIFACT_TOURNAMENT_COMPLETE_FILENAME = "complete.json"
ARTIFACT_BATTLE_INDEX_FILENAME = "battle_index.json"
ARTIFACT_CONFIG_FILENAME = "config.json"
ARTIFACT_INPUT_FILENAME = "input.json"
ARTIFACT_RESULTS_FILENAME = "results.json"
ARTIFACT_RATINGS_FILENAME = "ratings.json"
ARTIFACT_LATEST_FILENAME = "latest.json"
ARTIFACT_PROVISIONAL_LATEST_FILENAME = "provisional_latest.json"
ARTIFACT_PROGRESS_FILENAME = "progress.json"
ARTIFACT_PAIR_HISTORY_FILENAME = "pair_history.json"
ARTIFACT_SCHEDULER_STATE_FILENAME = "scheduler_state.json"

ALLOWED_TOURNAMENT_ARTIFACT_FILENAMES = frozenset(
    {
        ARTIFACT_GAME_GIF_FILENAME,
        ARTIFACT_FRAMES_NPZ_FILENAME,
        ARTIFACT_SUMMARY_FILENAME,
        ARTIFACT_BATTLE_SUMMARY_FILENAME,
        ARTIFACT_PAIR_SPEC_FILENAME,
        ARTIFACT_TOURNAMENT_MANIFEST_FILENAME,
        ARTIFACT_TOURNAMENT_STANDINGS_FILENAME,
        ARTIFACT_TOURNAMENT_COMPLETE_FILENAME,
        ARTIFACT_BATTLE_INDEX_FILENAME,
        ARTIFACT_CONFIG_FILENAME,
        ARTIFACT_INPUT_FILENAME,
        ARTIFACT_RESULTS_FILENAME,
        ARTIFACT_RATINGS_FILENAME,
        ARTIFACT_LATEST_FILENAME,
        ARTIFACT_PROVISIONAL_LATEST_FILENAME,
        ARTIFACT_PROGRESS_FILENAME,
        ARTIFACT_PAIR_HISTORY_FILENAME,
        ARTIFACT_SCHEDULER_STATE_FILENAME,
    }
)


class TournamentRefError(ValueError):
    """Raised when a tournament Volume ref is not safe."""


def _safe_id(raw: str, *, label: str) -> str:
    return runs.clean_id(raw, label=label)


def _slug(text: str, *, max_len: int = 36) -> str:
    chars = []
    for char in str(text):
        if char.isalnum() or char in {"-", "_", "."}:
            chars.append(char)
        elif char in {"/", " ", ":"}:
            chars.append("-")
    slug = "".join(chars).strip("-._")
    if not slug:
        slug = "item"
    return slug[:max_len].strip("-._") or "item"


def _short_hash(text: str, *, length: int = 10) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:length]


def checkpoint_id_from_ref(ref: str, *, index: int = 0) -> str:
    path = PurePosixPath(runs.require_relative_ref(ref))
    pieces = [part for part in path.parts if part not in {"training", "checkpoints", "lightzero"}]
    useful = "-".join(pieces[-4:]) if pieces else f"checkpoint-{index:03d}"
    return _safe_id(
        f"ckpt-{index:03d}-{_slug(useful, max_len=42)}-{_short_hash(path.as_posix(), length=8)}",
        label="checkpoint_id",
    )


def rating_pool_hash(checkpoints: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for checkpoint in checkpoints:
        rows.append(
            {
                "checkpoint_id": str(checkpoint.get("checkpoint_id") or ""),
                "checkpoint_ref": str(checkpoint.get("checkpoint_ref") or ""),
                "model_env_variant": checkpoint.get("model_env_variant"),
                "model_reward_variant": checkpoint.get("model_reward_variant"),
                "policy_trail_render_mode": checkpoint.get("policy_trail_render_mode"),
            }
        )
    rows.sort(key=lambda row: (row["checkpoint_id"], row["checkpoint_ref"]))
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return _short_hash(payload, length=16)


def rating_roster_by_checkpoint(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    roster = {}
    for checkpoint in checkpoints:
        checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
        if not checkpoint_id:
            continue
        roster[checkpoint_id] = {
            "checkpoint_ref": str(checkpoint.get("checkpoint_ref") or ""),
            "run_id": checkpoint.get("run_id"),
            "attempt_id": checkpoint.get("attempt_id"),
            "iteration": checkpoint.get("iteration"),
            "latest_for_run": bool(checkpoint.get("latest_for_run", False)),
            "checkpoint_mtime_ns": checkpoint.get("checkpoint_mtime_ns"),
            "model_env_variant": checkpoint.get("model_env_variant"),
            "model_reward_variant": checkpoint.get("model_reward_variant"),
            "policy_trail_render_mode": checkpoint.get("policy_trail_render_mode"),
        }
    return dict(sorted(roster.items()))


def rating_pair_key(checkpoint_id_a: str, checkpoint_id_b: str) -> str:
    ids = sorted([str(checkpoint_id_a), str(checkpoint_id_b)])
    digest = _short_hash("\n".join(ids), length=12)
    return _safe_id(
        f"pairkey-{_slug(ids[0], max_len=24)}-vs-{_slug(ids[1], max_len=24)}-{digest}",
        label="pair_key",
    )


def tournament_root_ref(tournament_id: str) -> PurePosixPath:
    return TOURNAMENT_BASE_REF / _safe_id(tournament_id, label="tournament_id")


def tournament_manifest_ref(tournament_id: str) -> PurePosixPath:
    return tournament_root_ref(tournament_id) / ARTIFACT_TOURNAMENT_MANIFEST_FILENAME


def tournament_marker_ref(tournament_id: str) -> PurePosixPath:
    return tournament_root_ref(tournament_id) / TOURNAMENT_RUN_MARKER_FILENAME


def tournament_standings_ref(tournament_id: str) -> PurePosixPath:
    return tournament_root_ref(tournament_id) / ARTIFACT_TOURNAMENT_STANDINGS_FILENAME


def tournament_complete_ref(tournament_id: str) -> PurePosixPath:
    return tournament_root_ref(tournament_id) / ARTIFACT_TOURNAMENT_COMPLETE_FILENAME


def tournament_battle_index_ref(tournament_id: str) -> PurePosixPath:
    return tournament_root_ref(tournament_id) / ARTIFACT_BATTLE_INDEX_FILENAME


def tournament_checkpoint_battle_index_ref(
    tournament_id: str,
    checkpoint_id: str,
) -> PurePosixPath:
    return (
        tournament_root_ref(tournament_id)
        / "checkpoints"
        / _safe_id(checkpoint_id, label="checkpoint_id")
        / ARTIFACT_BATTLE_INDEX_FILENAME
    )


def tournament_intake_root_ref(tournament_id: str, rating_run_id: str) -> PurePosixPath:
    return (
        tournament_root_ref(tournament_id)
        / "intake"
        / _safe_id(rating_run_id, label="rating_run_id")
    )


def tournament_intake_manifest_ref(tournament_id: str, rating_run_id: str) -> PurePosixPath:
    return tournament_intake_root_ref(tournament_id, rating_run_id) / ARTIFACT_CONFIG_FILENAME


def tournament_intake_latest_tick_ref(tournament_id: str, rating_run_id: str) -> PurePosixPath:
    return tournament_intake_root_ref(tournament_id, rating_run_id) / ARTIFACT_PROGRESS_FILENAME


def rating_root_ref(tournament_id: str, rating_run_id: str) -> PurePosixPath:
    return (
        tournament_root_ref(tournament_id)
        / "ratings"
        / _safe_id(rating_run_id, label="rating_run_id")
    )


def rating_config_ref(tournament_id: str, rating_run_id: str) -> PurePosixPath:
    return rating_root_ref(tournament_id, rating_run_id) / ARTIFACT_CONFIG_FILENAME


def rating_latest_ref(tournament_id: str, rating_run_id: str) -> PurePosixPath:
    return rating_root_ref(tournament_id, rating_run_id) / ARTIFACT_LATEST_FILENAME


def rating_provisional_latest_ref(tournament_id: str, rating_run_id: str) -> PurePosixPath:
    return rating_root_ref(tournament_id, rating_run_id) / ARTIFACT_PROVISIONAL_LATEST_FILENAME


def rating_progress_ref(tournament_id: str, rating_run_id: str) -> PurePosixPath:
    return rating_root_ref(tournament_id, rating_run_id) / ARTIFACT_PROGRESS_FILENAME


def rating_scheduler_state_ref(tournament_id: str, rating_run_id: str) -> PurePosixPath:
    return rating_root_ref(tournament_id, rating_run_id) / ARTIFACT_SCHEDULER_STATE_FILENAME


def rating_pair_history_ref(tournament_id: str, rating_run_id: str) -> PurePosixPath:
    return rating_root_ref(tournament_id, rating_run_id) / ARTIFACT_PAIR_HISTORY_FILENAME


def rating_run_results_ref(tournament_id: str, rating_run_id: str) -> PurePosixPath:
    return rating_root_ref(tournament_id, rating_run_id) / ARTIFACT_RESULTS_FILENAME


def rating_round_root_ref(
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
) -> PurePosixPath:
    return rating_root_ref(tournament_id, rating_run_id) / "rounds" / _safe_id(
        round_id,
        label="round_id",
    )


def rating_round_input_ref(
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
) -> PurePosixPath:
    return rating_round_root_ref(tournament_id, rating_run_id, round_id) / ARTIFACT_INPUT_FILENAME


def rating_round_results_ref(
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
) -> PurePosixPath:
    return rating_round_root_ref(tournament_id, rating_run_id, round_id) / ARTIFACT_RESULTS_FILENAME


def rating_round_ratings_ref(
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
) -> PurePosixPath:
    return rating_round_root_ref(tournament_id, rating_run_id, round_id) / ARTIFACT_RATINGS_FILENAME


def rating_round_progress_ref(
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
) -> PurePosixPath:
    return rating_round_root_ref(tournament_id, rating_run_id, round_id) / ARTIFACT_PROGRESS_FILENAME


def battle_root_ref(tournament_id: str, battle_id: str) -> PurePosixPath:
    return tournament_root_ref(tournament_id) / "battles" / _safe_id(battle_id, label="battle_id")


def battle_summary_ref(tournament_id: str, battle_id: str) -> PurePosixPath:
    return battle_root_ref(tournament_id, battle_id) / ARTIFACT_BATTLE_SUMMARY_FILENAME


def battle_pair_spec_ref(tournament_id: str, battle_id: str) -> PurePosixPath:
    return battle_root_ref(tournament_id, battle_id) / ARTIFACT_PAIR_SPEC_FILENAME


def game_root_ref(tournament_id: str, battle_id: str, game_id: str) -> PurePosixPath:
    return battle_root_ref(tournament_id, battle_id) / "games" / _safe_id(game_id, label="game_id")


def game_summary_ref(tournament_id: str, battle_id: str, game_id: str) -> PurePosixPath:
    return game_root_ref(tournament_id, battle_id, game_id) / ARTIFACT_SUMMARY_FILENAME


def game_shard_root_ref(tournament_id: str, battle_id: str, shard_id: str) -> PurePosixPath:
    return battle_root_ref(tournament_id, battle_id) / "shards" / _safe_id(
        shard_id,
        label="shard_id",
    )


def game_shard_summary_ref(tournament_id: str, battle_id: str, shard_id: str) -> PurePosixPath:
    return game_shard_root_ref(tournament_id, battle_id, shard_id) / ARTIFACT_SUMMARY_FILENAME


def game_gif_ref(tournament_id: str, battle_id: str, game_id: str) -> PurePosixPath:
    return game_root_ref(tournament_id, battle_id, game_id) / ARTIFACT_GAME_GIF_FILENAME


def game_frames_ref(tournament_id: str, battle_id: str, game_id: str) -> PurePosixPath:
    return game_root_ref(tournament_id, battle_id, game_id) / ARTIFACT_FRAMES_NPZ_FILENAME


def validate_tournament_artifact_ref(ref: str | PurePosixPath) -> PurePosixPath:
    path = runs.require_relative_ref(ref)
    if path.parts[:2] != TOURNAMENT_BASE_REF.parts:
        raise TournamentRefError("ref must be under tournaments/curvytron")
    if any("\\" in part or "\x00" in part for part in path.parts):
        raise TournamentRefError("ref contains an unsafe segment")
    if path.name not in ALLOWED_TOURNAMENT_ARTIFACT_FILENAMES:
        raise TournamentRefError("ref is not an allowed tournament artifact")
    return path


def parse_checkpoint_refs(text: str) -> list[str]:
    refs = []
    for item in str(text).replace("\n", ",").split(","):
        stripped = item.strip()
        if stripped:
            refs.append(runs.require_relative_ref(stripped).as_posix())
    return refs


__all__ = [
    "ALLOWED_TOURNAMENT_ARTIFACT_FILENAMES",
    "ARTIFACT_BATTLE_INDEX_FILENAME",
    "ARTIFACT_BATTLE_SUMMARY_FILENAME",
    "ARTIFACT_CONFIG_FILENAME",
    "ARTIFACT_FRAMES_NPZ_FILENAME",
    "ARTIFACT_GAME_GIF_FILENAME",
    "ARTIFACT_INPUT_FILENAME",
    "ARTIFACT_LATEST_FILENAME",
    "ARTIFACT_PAIR_HISTORY_FILENAME",
    "ARTIFACT_PAIR_SPEC_FILENAME",
    "ARTIFACT_PROGRESS_FILENAME",
    "ARTIFACT_PROVISIONAL_LATEST_FILENAME",
    "ARTIFACT_RATINGS_FILENAME",
    "ARTIFACT_RESULTS_FILENAME",
    "ARTIFACT_SCHEDULER_STATE_FILENAME",
    "ARTIFACT_SUMMARY_FILENAME",
    "ARTIFACT_TOURNAMENT_COMPLETE_FILENAME",
    "ARTIFACT_TOURNAMENT_MANIFEST_FILENAME",
    "ARTIFACT_TOURNAMENT_STANDINGS_FILENAME",
    "BATTLE_SCHEMA_ID",
    "CHECKPOINT_EXP_CKPT_DIR_GLOB",
    "CHECKPOINT_SCAN_GLOB",
    "CHECKPOINT_SELECTION_ALL",
    "CHECKPOINT_SELECTION_CHOICES",
    "CHECKPOINT_SELECTION_ITERATION",
    "CHECKPOINT_SELECTION_LATEST",
    "CHECKPOINT_WEIGHT_FILENAME_GLOB",
    "DEFAULT_COLLECT_EPSILON",
    "DEFAULT_COLLECT_TEMPERATURE",
    "DEFAULT_DECISION_MS",
    "DEFAULT_FRAME_SIZE",
    "DEFAULT_FRAME_STRIDE",
    "DEFAULT_GAMES_PER_PAIR",
    "DEFAULT_GAMES_PER_SHARD",
    "DEFAULT_GIF_FPS",
    "DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR",
    "DEFAULT_GIF_SAMPLE_STRATEGY",
    "DEFAULT_GIF_TRAIL_RENDER_MODE",
    "DEFAULT_INCLUDE_SELF_PAIRS",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_NUM_SIMULATIONS",
    "DEFAULT_ORDERED_PAIRS",
    "DEFAULT_POLICY_BATCH_SIZE",
    "DEFAULT_RATING_BASE_K",
    "DEFAULT_RATING_DELTA_CLAMP",
    "DEFAULT_RATING_DRAW_SCORE",
    "DEFAULT_RATING_INITIAL_RATING",
    "DEFAULT_RATING_ACTIVE_POOL_LIMIT",
    "DEFAULT_RATING_K_MAX",
    "DEFAULT_RATING_K_MIN",
    "DEFAULT_RATING_K_REFERENCE_GAMES",
    "DEFAULT_RATING_MIN_VALID_FRACTION",
    "DEFAULT_RATING_PAIR_SELECTION",
    "DEFAULT_RATING_ROUND_COUNT",
    "DEFAULT_RATING_RUN_ID",
    "DEFAULT_RATING_STOP_MAX_DELTA",
    "DEFAULT_REUSE_POLICIES_PER_SHARD",
    "DEFAULT_SAVE_FRAMES_NPZ",
    "DEFAULT_SAVE_GIF",
    "DEFAULT_SOURCE_PHYSICS_STEP_MS",
    "GAME_SCHEMA_ID",
    "GAME_SHARD_SCHEMA_ID",
    "GIF_SAMPLE_STRATEGY_CHOICES",
    "PAIR_HISTORY_SCHEMA_ID",
    "POLICY_MODE_CHOICES",
    "POLICY_MODE_COLLECT",
    "POLICY_MODE_EVAL",
    "RATING_CONFIG_SCHEMA_ID",
    "RATING_FORMULA_VERSION",
    "RATING_PAIR_SELECTION_ADAPTIVE_V0",
    "RATING_PAIR_SELECTION_ALL_PAIRS",
    "RATING_PAIR_SELECTION_CHOICES",
    "RATING_PAIR_SELECTION_RANDOM",
    "RATING_PROGRESS_SCHEMA_ID",
    "RATING_ROUND_SCHEMA_ID",
    "RATING_SCHEDULER_STATE_SCHEMA_ID",
    "RATING_SNAPSHOT_SCHEMA_ID",
    "SCHEDULE_REASON_CHOICES",
    "SCHEDULE_REASON_FILL",
    "SCHEDULE_REASON_NEAR_RATING",
    "SCHEDULE_REASON_PLACEMENT",
    "SCHEDULE_REASON_RANDOM_BRIDGE",
    "SCHEDULE_REASON_UNCERTAIN",
    "TOURNAMENT_BASE_REF",
    "TOURNAMENT_RUN_MARKER_FILENAME",
    "TOURNAMENT_SCHEMA_ID",
    "TOURNAMENT_TASK_ID",
    "TournamentRefError",
    "_safe_id",
    "_short_hash",
    "_slug",
    "battle_pair_spec_ref",
    "battle_root_ref",
    "battle_summary_ref",
    "checkpoint_id_from_ref",
    "game_frames_ref",
    "game_gif_ref",
    "game_root_ref",
    "game_shard_root_ref",
    "game_shard_summary_ref",
    "game_summary_ref",
    "parse_checkpoint_refs",
    "rating_config_ref",
    "rating_latest_ref",
    "rating_pair_history_ref",
    "rating_pair_key",
    "rating_pool_hash",
    "rating_roster_by_checkpoint",
    "rating_progress_ref",
    "rating_provisional_latest_ref",
    "rating_root_ref",
    "rating_round_input_ref",
    "rating_round_progress_ref",
    "rating_round_ratings_ref",
    "rating_round_results_ref",
    "rating_round_root_ref",
    "rating_run_results_ref",
    "rating_scheduler_state_ref",
    "tournament_battle_index_ref",
    "tournament_checkpoint_battle_index_ref",
    "tournament_complete_ref",
    "tournament_manifest_ref",
    "tournament_marker_ref",
    "tournament_intake_latest_tick_ref",
    "tournament_intake_manifest_ref",
    "tournament_intake_root_ref",
    "tournament_root_ref",
    "tournament_standings_ref",
    "validate_tournament_artifact_ref",
]
