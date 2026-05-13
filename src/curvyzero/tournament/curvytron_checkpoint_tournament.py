"""CurvyTron checkpoint tournament helpers.

The tournament lane is intentionally separate from training. A game is one
checkpoint in seat 0 against one checkpoint in seat 1. The score is simple:
the first dead player loses; simultaneous death or timeout is a draw.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import traceback
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from curvyzero.env.vector_multiplayer_env import (
    DEFAULT_DECISION_SOURCE_FRAMES,
    SOURCE_PHYSICS_STEP_MS,
)
from curvyzero.infra.modal import run_management as runs


TOURNAMENT_TASK_ID = "curvytron-checkpoint-tournament"
TOURNAMENT_BASE_REF = PurePosixPath("tournaments") / "curvytron"
TOURNAMENT_RUN_MARKER_FILENAME = "show_in_tournament_browser.flag"

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
DEFAULT_RATING_PAIR_SELECTION = "all_pairs"
DEFAULT_RATING_INITIAL_RATING = 1500.0
DEFAULT_RATING_BASE_K = 32.0
DEFAULT_RATING_K_REFERENCE_GAMES = 50.0
DEFAULT_RATING_K_MIN = 16.0
DEFAULT_RATING_K_MAX = 64.0
DEFAULT_RATING_DELTA_CLAMP = 80.0
DEFAULT_RATING_DRAW_SCORE = 0.5
DEFAULT_RATING_MIN_VALID_FRACTION = 0.8
DEFAULT_RATING_STOP_MAX_DELTA = 10.0
RATING_PAIR_SELECTION_CHOICES = ("all_pairs", "random", "adaptive_v0")

ALLOWED_TOURNAMENT_ARTIFACT_FILENAMES = frozenset(
    {
        "game.gif",
        "frames.npz",
        "summary.json",
        "battle.json",
        "tournament.json",
        "standings.json",
        "complete.json",
        "battle_index.json",
        "config.json",
        "input.json",
        "results.json",
        "ratings.json",
        "latest.json",
        "provisional_latest.json",
        "progress.json",
    }
)


class TournamentRefError(ValueError):
    """Raised when a tournament Volume ref is not safe."""


def _validate_games_per_pair(value: int) -> int:
    games_per_pair = int(value)
    if games_per_pair < 1:
        raise ValueError("games_per_pair must be at least 1")
    if games_per_pair % 2 == 0:
        raise ValueError("games_per_pair must be odd")
    return games_per_pair


def _to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _to_plain(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def exception_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-16:],
    }


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


def normalize_checkpoint_spec(raw: str | Mapping[str, Any], *, index: int = 0) -> dict[str, Any]:
    if isinstance(raw, str):
        ref = runs.require_relative_ref(raw).as_posix()
        return {
            "checkpoint_id": checkpoint_id_from_ref(ref, index=index),
            "label": PurePosixPath(ref).name,
            "checkpoint_ref": ref,
            "checkpoint_state_key": None,
            "model_env_variant": None,
            "model_reward_variant": None,
            "policy_trail_render_mode": None,
        }
    ref_value = raw.get("checkpoint_ref") or raw.get("ref")
    if not isinstance(ref_value, str):
        raise ValueError("checkpoint spec needs checkpoint_ref")
    ref = runs.require_relative_ref(ref_value).as_posix()
    checkpoint_id = str(raw.get("checkpoint_id") or raw.get("id") or checkpoint_id_from_ref(ref, index=index))
    label = str(raw.get("label") or checkpoint_id)
    observation_contract = raw.get("observation_contract")
    contract_trail_render_mode = (
        observation_contract.get("trail_render_mode")
        if isinstance(observation_contract, Mapping)
        else None
    )
    return {
        "checkpoint_id": _safe_id(checkpoint_id, label="checkpoint_id"),
        "label": label,
        "checkpoint_ref": ref,
        "checkpoint_state_key": raw.get("checkpoint_state_key"),
        "model_env_variant": raw.get("model_env_variant"),
        "model_reward_variant": raw.get("model_reward_variant"),
        "policy_trail_render_mode": (
            raw.get("policy_trail_render_mode")
            or raw.get("observation_trail_render_mode")
            or contract_trail_render_mode
            or raw.get("trail_render_mode")
        ),
    }


def normalize_checkpoint_specs(checkpoints: Sequence[str | Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        normalize_checkpoint_spec(checkpoint, index=index)
        for index, checkpoint in enumerate(checkpoints)
    ]


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
    return tournament_root_ref(tournament_id) / "tournament.json"


def tournament_marker_ref(tournament_id: str) -> PurePosixPath:
    return tournament_root_ref(tournament_id) / TOURNAMENT_RUN_MARKER_FILENAME


def tournament_standings_ref(tournament_id: str) -> PurePosixPath:
    return tournament_root_ref(tournament_id) / "standings.json"


def tournament_complete_ref(tournament_id: str) -> PurePosixPath:
    return tournament_root_ref(tournament_id) / "complete.json"


def tournament_battle_index_ref(tournament_id: str) -> PurePosixPath:
    return tournament_root_ref(tournament_id) / "battle_index.json"


def rating_root_ref(tournament_id: str, rating_run_id: str) -> PurePosixPath:
    return (
        tournament_root_ref(tournament_id)
        / "ratings"
        / _safe_id(rating_run_id, label="rating_run_id")
    )


def rating_config_ref(tournament_id: str, rating_run_id: str) -> PurePosixPath:
    return rating_root_ref(tournament_id, rating_run_id) / "config.json"


def rating_latest_ref(tournament_id: str, rating_run_id: str) -> PurePosixPath:
    return rating_root_ref(tournament_id, rating_run_id) / "latest.json"


def rating_progress_ref(tournament_id: str, rating_run_id: str) -> PurePosixPath:
    return rating_root_ref(tournament_id, rating_run_id) / "progress.json"


def rating_scheduler_state_ref(tournament_id: str, rating_run_id: str) -> PurePosixPath:
    return rating_root_ref(tournament_id, rating_run_id) / "scheduler_state.json"


def rating_pair_history_ref(tournament_id: str, rating_run_id: str) -> PurePosixPath:
    return rating_root_ref(tournament_id, rating_run_id) / "pair_history.json"


def rating_round_id(round_index: int) -> str:
    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    return f"round-{int(round_index):06d}"


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
    return rating_round_root_ref(tournament_id, rating_run_id, round_id) / "input.json"


def rating_round_results_ref(
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
) -> PurePosixPath:
    return rating_round_root_ref(tournament_id, rating_run_id, round_id) / "results.json"


def rating_round_ratings_ref(
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
) -> PurePosixPath:
    return rating_round_root_ref(tournament_id, rating_run_id, round_id) / "ratings.json"


def rating_round_progress_ref(
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
) -> PurePosixPath:
    return rating_round_root_ref(tournament_id, rating_run_id, round_id) / "progress.json"


def battle_root_ref(tournament_id: str, battle_id: str) -> PurePosixPath:
    return tournament_root_ref(tournament_id) / "battles" / _safe_id(battle_id, label="battle_id")


def battle_summary_ref(tournament_id: str, battle_id: str) -> PurePosixPath:
    return battle_root_ref(tournament_id, battle_id) / "battle.json"


def game_root_ref(tournament_id: str, battle_id: str, game_id: str) -> PurePosixPath:
    return battle_root_ref(tournament_id, battle_id) / "games" / _safe_id(game_id, label="game_id")


def game_summary_ref(tournament_id: str, battle_id: str, game_id: str) -> PurePosixPath:
    return game_root_ref(tournament_id, battle_id, game_id) / "summary.json"


def game_shard_root_ref(tournament_id: str, battle_id: str, shard_id: str) -> PurePosixPath:
    return battle_root_ref(tournament_id, battle_id) / "shards" / _safe_id(
        shard_id,
        label="shard_id",
    )


def game_shard_summary_ref(tournament_id: str, battle_id: str, shard_id: str) -> PurePosixPath:
    return game_shard_root_ref(tournament_id, battle_id, shard_id) / "summary.json"


def game_gif_ref(tournament_id: str, battle_id: str, game_id: str) -> PurePosixPath:
    return game_root_ref(tournament_id, battle_id, game_id) / "game.gif"


def game_frames_ref(tournament_id: str, battle_id: str, game_id: str) -> PurePosixPath:
    return game_root_ref(tournament_id, battle_id, game_id) / "frames.npz"


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


def battle_id_for_pair(pair_index: int, player_a: Mapping[str, Any], player_b: Mapping[str, Any]) -> str:
    raw = f"pair-{pair_index:06d}-{player_a['checkpoint_id']}-vs-{player_b['checkpoint_id']}"
    if len(raw) <= 96:
        return _safe_id(raw, label="battle_id")
    digest = _short_hash(raw, length=12)
    return _safe_id(
        f"pair-{pair_index:06d}-{_slug(player_a['checkpoint_id'], max_len=24)}-vs-{_slug(player_b['checkpoint_id'], max_len=24)}-{digest}",
        label="battle_id",
    )


def build_pair_specs(
    *,
    tournament_id: str,
    checkpoints: Sequence[str | Mapping[str, Any]],
    games_per_pair: int = DEFAULT_GAMES_PER_PAIR,
    ordered_pairs: bool = DEFAULT_ORDERED_PAIRS,
    include_self_pairs: bool = DEFAULT_INCLUDE_SELF_PAIRS,
    seed: int = 0,
    **settings: Any,
) -> list[dict[str, Any]]:
    players = normalize_checkpoint_specs(checkpoints)
    if len(players) < 2 and not include_self_pairs:
        raise ValueError("at least two checkpoints are needed for a tournament")
    games_per_pair = _validate_games_per_pair(int(games_per_pair))
    pair_specs: list[dict[str, Any]] = []
    pair_index = 0
    for i, player_a in enumerate(players):
        for j, player_b in enumerate(players):
            if not include_self_pairs and i == j:
                continue
            if not ordered_pairs and (
                j < i or (not include_self_pairs and j == i)
            ):
                continue
            battle_id = battle_id_for_pair(pair_index, player_a, player_b)
            pair_specs.append(
                normalize_pair_spec(
                    {
                        "tournament_id": tournament_id,
                        "battle_id": battle_id,
                        "pair_index": pair_index,
                        "players": [
                            {"seat": 0, **player_a},
                            {"seat": 1, **player_b},
                        ],
                        "games_per_pair": games_per_pair,
                        "seed": int(seed) + pair_index * 10_000,
                        **settings,
                    }
                )
            )
            pair_index += 1
    return pair_specs


def normalize_pair_spec(raw: Mapping[str, Any]) -> dict[str, Any]:
    tournament_id = _safe_id(str(raw.get("tournament_id") or "tournament"), label="tournament_id")
    players = raw.get("players")
    if not isinstance(players, Sequence) or len(players) != 2:
        raise ValueError("pair spec needs exactly two players")
    normalized_players = []
    for seat, player in enumerate(players):
        checkpoint = normalize_checkpoint_spec(player, index=seat)
        normalized_players.append({"seat": seat, **checkpoint})
    battle_id = _safe_id(
        str(raw.get("battle_id") or battle_id_for_pair(int(raw.get("pair_index") or 0), normalized_players[0], normalized_players[1])),
        label="battle_id",
    )
    policy_mode = str(raw.get("policy_mode", POLICY_MODE_EVAL))
    if policy_mode not in POLICY_MODE_CHOICES:
        raise ValueError(f"policy_mode must be one of {POLICY_MODE_CHOICES!r}")
    gif_sample_games_per_pair = int(
        raw.get("gif_sample_games_per_pair", DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR)
    )
    if gif_sample_games_per_pair < -1:
        raise ValueError("gif_sample_games_per_pair must be -1, 0, or positive")
    gif_sample_strategy = str(
        raw.get("gif_sample_strategy", DEFAULT_GIF_SAMPLE_STRATEGY)
    )
    if gif_sample_strategy not in GIF_SAMPLE_STRATEGY_CHOICES:
        raise ValueError(
            f"gif_sample_strategy must be one of {GIF_SAMPLE_STRATEGY_CHOICES!r}"
        )
    games_per_pair = _validate_games_per_pair(
        int(raw.get("games_per_pair", DEFAULT_GAMES_PER_PAIR))
    )
    games_per_shard = int(raw.get("games_per_shard", DEFAULT_GAMES_PER_SHARD))
    if games_per_shard < 1:
        raise ValueError("games_per_shard must be at least 1")
    reuse_policies_per_shard = bool(
        raw.get("reuse_policies_per_shard", DEFAULT_REUSE_POLICIES_PER_SHARD)
    )
    policy_trail_render_mode = (
        raw.get("policy_trail_render_mode")
        or raw.get("observation_trail_render_mode")
        or raw.get("trail_render_mode")
    )
    gif_trail_render_mode = DEFAULT_GIF_TRAIL_RENDER_MODE
    normalized = {
        "schema_id": BATTLE_SCHEMA_ID,
        "tournament_id": tournament_id,
        "battle_id": battle_id,
        "pair_index": int(raw.get("pair_index") or 0),
        "players": normalized_players,
        "games_per_pair": games_per_pair,
        "games_per_shard": games_per_shard,
        "reuse_policies_per_shard": reuse_policies_per_shard,
        "seed": int(raw.get("seed", 0)),
        "max_steps": int(raw.get("max_steps", DEFAULT_MAX_STEPS)),
        "decision_ms": float(raw.get("decision_ms", DEFAULT_DECISION_MS)),
        "decision_source_frames": raw.get("decision_source_frames"),
        "source_physics_step_ms": float(
            raw.get("source_physics_step_ms", DEFAULT_SOURCE_PHYSICS_STEP_MS)
        ),
        "num_simulations": int(raw.get("num_simulations", DEFAULT_NUM_SIMULATIONS)),
        "policy_batch_size": int(raw.get("policy_batch_size", DEFAULT_POLICY_BATCH_SIZE)),
        "policy_mode": policy_mode,
        "collect_temperature": float(raw.get("collect_temperature", DEFAULT_COLLECT_TEMPERATURE)),
        "collect_epsilon": float(raw.get("collect_epsilon", DEFAULT_COLLECT_EPSILON)),
        "natural_bonus_spawn": bool(raw.get("natural_bonus_spawn", True)),
        "policy_trail_render_mode": policy_trail_render_mode,
        "gif_trail_render_mode": gif_trail_render_mode,
        "trail_render_mode": policy_trail_render_mode,
        "frame_stride": int(raw.get("frame_stride", DEFAULT_FRAME_STRIDE)),
        "frame_size": DEFAULT_FRAME_SIZE,
        "gif_fps": float(raw.get("gif_fps", DEFAULT_GIF_FPS)),
        "save_gif": bool(raw.get("save_gif", DEFAULT_SAVE_GIF)),
        "gif_sample_games_per_pair": gif_sample_games_per_pair,
        "gif_sample_strategy": gif_sample_strategy,
        "save_frames_npz": bool(raw.get("save_frames_npz", DEFAULT_SAVE_FRAMES_NPZ)),
        "action_trace_limit": int(raw.get("action_trace_limit", 128)),
    }
    for key in (
        "pair_key",
        "schedule_reason",
        "schedule_priority",
        "scheduled_round_index",
    ):
        if key in raw and raw.get(key) is not None:
            normalized[key] = raw[key]
    if isinstance(raw.get("schedule"), Mapping):
        normalized["schedule"] = _to_plain(dict(raw["schedule"]))
    return normalized


def gif_sample_count_for_pair(pair_spec: Mapping[str, Any]) -> int:
    pair = normalize_pair_spec(pair_spec)
    if not pair["save_gif"]:
        return 0
    games_per_pair = int(pair["games_per_pair"])
    sample = int(pair["gif_sample_games_per_pair"])
    if sample < 0:
        return games_per_pair
    return min(games_per_pair, sample)


def gif_sample_indices_for_pair(pair_spec: Mapping[str, Any]) -> set[int]:
    pair = normalize_pair_spec(pair_spec)
    count = gif_sample_count_for_pair(pair)
    if count <= 0:
        return set()
    games_per_pair = int(pair["games_per_pair"])
    if count >= games_per_pair:
        return set(range(games_per_pair))
    if pair["gif_sample_strategy"] == "first_n":
        return set(range(count))
    if count == 1:
        return {0}
    last = games_per_pair - 1
    return {
        int(round(index * last / float(count - 1)))
        for index in range(count)
    }


def build_game_specs_for_pair(pair_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    pair = normalize_pair_spec(pair_spec)
    count = int(pair["games_per_pair"])
    if count < 1:
        raise ValueError("games_per_pair must be at least 1")
    gif_sample_indices = gif_sample_indices_for_pair(pair)
    specs = []
    for game_index in range(count):
        game_id = f"game-{game_index:06d}"
        specs.append(
            {
                "schema_id": GAME_SCHEMA_ID,
                "tournament_id": pair["tournament_id"],
                "battle_id": pair["battle_id"],
                "pair_index": pair["pair_index"],
                "game_index": game_index,
                "game_id": game_id,
                "players": pair["players"],
                "seed": int(pair["seed"]) + game_index,
                "max_steps": pair["max_steps"],
                "decision_ms": pair["decision_ms"],
                "decision_source_frames": pair["decision_source_frames"],
                "source_physics_step_ms": pair["source_physics_step_ms"],
                "num_simulations": pair["num_simulations"],
                "policy_batch_size": pair["policy_batch_size"],
                "policy_mode": pair["policy_mode"],
                "collect_temperature": pair["collect_temperature"],
                "collect_epsilon": pair["collect_epsilon"],
                "natural_bonus_spawn": pair["natural_bonus_spawn"],
                "policy_trail_render_mode": pair["policy_trail_render_mode"],
                "gif_trail_render_mode": pair["gif_trail_render_mode"],
                "trail_render_mode": pair["trail_render_mode"],
                "frame_stride": pair["frame_stride"],
                "frame_size": pair["frame_size"],
                "gif_fps": pair["gif_fps"],
                "save_gif": bool(pair["save_gif"]) and game_index in gif_sample_indices,
                "gif_sample_games_per_pair": pair["gif_sample_games_per_pair"],
                "gif_sample_strategy": pair["gif_sample_strategy"],
                "save_frames_npz": pair["save_frames_npz"],
                "action_trace_limit": pair["action_trace_limit"],
            }
        )
    return specs


def build_game_shard_specs_for_pair(
    pair_spec: Mapping[str, Any],
    *,
    games_per_shard: int | None = None,
) -> list[dict[str, Any]]:
    pair = normalize_pair_spec(pair_spec)
    if games_per_shard is None:
        shard_size = int(pair.get("games_per_shard") or DEFAULT_GAMES_PER_SHARD)
    else:
        shard_size = int(games_per_shard)
    if shard_size < 1:
        raise ValueError("games_per_shard must be at least 1")
    games = build_game_specs_for_pair(pair)
    shards = []
    for shard_index, start in enumerate(range(0, len(games), shard_size)):
        shard_games = games[start : start + shard_size]
        shard_id = f"shard-{shard_index:06d}-games-{start:06d}-{start + len(shard_games) - 1:06d}"
        shards.append(
            {
                "schema_id": GAME_SHARD_SCHEMA_ID,
                "tournament_id": pair["tournament_id"],
                "battle_id": pair["battle_id"],
                "pair_index": pair["pair_index"],
                "shard_index": shard_index,
                "shard_id": shard_id,
                "games_per_shard": shard_size,
                "reuse_policies": bool(pair["reuse_policies_per_shard"]),
                "game_count": len(shard_games),
                "game_index_start": int(shard_games[0]["game_index"]),
                "game_index_end": int(shard_games[-1]["game_index"]),
                "game_specs": shard_games,
            }
        )
    return shards


def count_pair_candidates(
    checkpoint_count: int,
    *,
    ordered_pairs: bool = DEFAULT_ORDERED_PAIRS,
    include_self_pairs: bool = DEFAULT_INCLUDE_SELF_PAIRS,
) -> int:
    count = 0
    for i in range(int(checkpoint_count)):
        for j in range(int(checkpoint_count)):
            if not include_self_pairs and i == j:
                continue
            if not ordered_pairs and (
                j < i or (not include_self_pairs and j == i)
            ):
                continue
            count += 1
    return count


def estimate_tournament_plan(
    *,
    checkpoint_count: int,
    games_per_pair: int = DEFAULT_GAMES_PER_PAIR,
    ordered_pairs: bool = DEFAULT_ORDERED_PAIRS,
    include_self_pairs: bool = DEFAULT_INCLUDE_SELF_PAIRS,
    pairs_per_round: int | None = None,
    games_per_shard: int = DEFAULT_GAMES_PER_SHARD,
    reuse_policies_per_shard: bool = DEFAULT_REUSE_POLICIES_PER_SHARD,
    save_gif: bool = DEFAULT_SAVE_GIF,
    gif_sample_games_per_pair: int = DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR,
    gif_sample_strategy: str = DEFAULT_GIF_SAMPLE_STRATEGY,
    save_frames_npz: bool = DEFAULT_SAVE_FRAMES_NPZ,
) -> dict[str, Any]:
    checkpoint_count = int(checkpoint_count)
    games_per_pair = _validate_games_per_pair(int(games_per_pair))
    games_per_shard = int(games_per_shard)
    gif_sample_games_per_pair = int(gif_sample_games_per_pair)
    gif_sample_strategy = str(gif_sample_strategy)
    if checkpoint_count < 0:
        raise ValueError("checkpoint_count must be non-negative")
    if games_per_shard < 1:
        raise ValueError("games_per_shard must be at least 1")
    if gif_sample_games_per_pair < -1:
        raise ValueError("gif_sample_games_per_pair must be -1, 0, or positive")
    if gif_sample_strategy not in GIF_SAMPLE_STRATEGY_CHOICES:
        raise ValueError(
            f"gif_sample_strategy must be one of {GIF_SAMPLE_STRATEGY_CHOICES!r}"
        )
    pair_candidate_count = count_pair_candidates(
        checkpoint_count,
        ordered_pairs=ordered_pairs,
        include_self_pairs=include_self_pairs,
    )
    if pairs_per_round in (None, "", 0, "0"):
        pair_count = pair_candidate_count
        normalized_pairs_per_round = None
    else:
        normalized_pairs_per_round = int(pairs_per_round)
        if normalized_pairs_per_round < 1:
            raise ValueError("pairs_per_round must be positive or empty")
        pair_count = min(pair_candidate_count, normalized_pairs_per_round)
    game_count = pair_count * games_per_pair
    game_call_count = pair_count * math.ceil(games_per_pair / games_per_shard)
    gif_per_pair = 0
    if save_gif:
        gif_per_pair = (
            games_per_pair
            if gif_sample_games_per_pair < 0
            else min(games_per_pair, gif_sample_games_per_pair)
        )
    gif_count = pair_count * gif_per_pair
    frames_npz_count = game_count if save_frames_npz else 0
    return {
        "schema_id": "curvyzero_curvytron_checkpoint_tournament_plan_estimate/v0",
        "checkpoint_count": checkpoint_count,
        "ordered_pairs": bool(ordered_pairs),
        "include_self_pairs": bool(include_self_pairs),
        "pair_candidate_count": pair_candidate_count,
        "pairs_per_round": normalized_pairs_per_round,
        "pair_count": pair_count,
        "games_per_pair": games_per_pair,
        "games_per_shard": games_per_shard,
        "reuse_policies_per_shard": bool(reuse_policies_per_shard),
        "game_count": game_count,
        "game_call_count": game_call_count,
        "save_gif": bool(save_gif),
        "gif_sample_games_per_pair": gif_sample_games_per_pair,
        "gif_sample_strategy": gif_sample_strategy,
        "gif_per_pair": gif_per_pair,
        "gif_count": gif_count,
        "frames_npz_count": frames_npz_count,
        "approx_json_file_count": game_count + pair_count + 4,
        "approx_artifact_file_count": game_count + pair_count + gif_count + frames_npz_count + 4,
        "approx_game_worker_commit_count": game_call_count,
    }


def _row0(value: Any) -> Any:
    plain = _to_plain(value)
    if isinstance(plain, list) and plain:
        return plain[0]
    return plain


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_value(value: Any) -> bool:
    plain = _row0(value)
    if isinstance(plain, bool):
        return plain
    if isinstance(plain, (int, float)):
        return bool(plain)
    return False


def _death_players(info: Mapping[str, Any]) -> list[int]:
    death_count = _int_or_none(_row0(info.get("death_count"))) or 0
    if death_count <= 0:
        return []
    raw_players = _row0(info.get("death_player"))
    if not isinstance(raw_players, list):
        raw_players = [raw_players]
    players = []
    for value in raw_players[:death_count]:
        player = _int_or_none(value)
        if player is not None and player >= 0 and player not in players:
            players.append(player)
    return players


def score_from_info(
    info: Mapping[str, Any],
    *,
    done: bool,
    truncated: bool,
    physical_steps: int,
    max_steps: int,
) -> dict[str, Any]:
    plain_info = _to_plain(info)
    deaths = _death_players(plain_info)
    terminal_reason = _row0(plain_info.get("terminal_reason_name")) or _row0(
        plain_info.get("terminal_reason")
    )
    if len(deaths) == 1:
        loser = int(deaths[0])
        winner = 1 - loser if loser in (0, 1) else None
        return {
            "outcome": "seat_0_win" if winner == 0 else "seat_1_win",
            "winner_seat": winner,
            "loser_seat": loser,
            "draw": False,
            "score_reason": "single_player_death",
            "death_players": deaths,
            "terminal_reason": terminal_reason,
            "physical_steps": int(physical_steps),
            "max_steps": int(max_steps),
        }
    if len(deaths) > 1:
        return {
            "outcome": "draw",
            "winner_seat": None,
            "loser_seat": None,
            "draw": True,
            "score_reason": "simultaneous_death_same_public_step",
            "death_players": deaths,
            "terminal_reason": terminal_reason,
            "physical_steps": int(physical_steps),
            "max_steps": int(max_steps),
        }
    for winner_key in ("winner", "round_winner", "match_winner"):
        winner = _int_or_none(_row0(plain_info.get(winner_key)))
        if winner in (0, 1):
            return {
                "outcome": "seat_0_win" if winner == 0 else "seat_1_win",
                "winner_seat": winner,
                "loser_seat": 1 - winner,
                "draw": False,
                "score_reason": winner_key,
                "death_players": [],
                "terminal_reason": terminal_reason,
                "physical_steps": int(physical_steps),
                "max_steps": int(max_steps),
            }
    draw = _bool_value(plain_info.get("draw"))
    if draw or bool(truncated) or (done and not deaths):
        return {
            "outcome": "draw",
            "winner_seat": None,
            "loser_seat": None,
            "draw": True,
            "score_reason": "draw_or_timeout",
            "death_players": [],
            "terminal_reason": terminal_reason,
            "physical_steps": int(physical_steps),
            "max_steps": int(max_steps),
        }
    return {
        "outcome": "unfinished",
        "winner_seat": None,
        "loser_seat": None,
        "draw": False,
        "score_reason": "step_limit_without_terminal",
        "death_players": [],
        "terminal_reason": terminal_reason,
        "physical_steps": int(physical_steps),
        "max_steps": int(max_steps),
    }


def tally_game_results(game_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    wins_by_seat = Counter()
    wins_by_checkpoint = Counter()
    failures = 0
    total_steps = 0
    completed = 0
    for result in game_results:
        if not result.get("ok"):
            failures += 1
            continue
        completed += 1
        score = result.get("score") if isinstance(result.get("score"), Mapping) else {}
        outcome = str(score.get("outcome") or "unknown")
        counts[outcome] += 1
        total_steps += int(score.get("physical_steps") or result.get("physical_steps") or 0)
        winner = score.get("winner_seat")
        if winner in (0, 1):
            wins_by_seat[f"seat_{winner}"] += 1
            players = result.get("players") if isinstance(result.get("players"), Sequence) else []
            try:
                checkpoint_id = str(players[int(winner)]["checkpoint_id"])
                wins_by_checkpoint[checkpoint_id] += 1
            except Exception:
                pass
        elif score.get("draw"):
            counts["draw"] += 0
    return {
        "game_count": int(len(game_results)),
        "completed_count": int(completed),
        "failure_count": int(failures),
        "outcomes": dict(sorted(counts.items())),
        "wins_by_seat": dict(sorted(wins_by_seat.items())),
        "wins_by_checkpoint": dict(sorted(wins_by_checkpoint.items())),
        "draw_count": int(counts.get("draw", 0)),
        "average_physical_steps": (
            float(total_steps) / float(completed) if completed else None
        ),
    }


def merge_game_tallies(tallies: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    wins_by_seat = Counter()
    wins_by_checkpoint = Counter()
    game_count = 0
    completed = 0
    failures = 0
    physical_step_total = 0.0
    for tally in tallies:
        if not isinstance(tally, Mapping):
            continue
        game_count += int(tally.get("game_count") or 0)
        completed_count = int(tally.get("completed_count") or 0)
        completed += completed_count
        failures += int(tally.get("failure_count") or 0)
        for key, value in (tally.get("outcomes") or {}).items():
            counts[str(key)] += int(value or 0)
        for key, value in (tally.get("wins_by_seat") or {}).items():
            wins_by_seat[str(key)] += int(value or 0)
        for key, value in (tally.get("wins_by_checkpoint") or {}).items():
            wins_by_checkpoint[str(key)] += int(value or 0)
        average_steps = tally.get("average_physical_steps")
        if average_steps is not None and completed_count:
            physical_step_total += float(average_steps) * float(completed_count)
    return {
        "game_count": int(game_count),
        "completed_count": int(completed),
        "failure_count": int(failures),
        "outcomes": dict(sorted(counts.items())),
        "wins_by_seat": dict(sorted(wins_by_seat.items())),
        "wins_by_checkpoint": dict(sorted(wins_by_checkpoint.items())),
        "draw_count": int(counts.get("draw", 0)),
        "average_physical_steps": (
            float(physical_step_total) / float(completed) if completed else None
        ),
    }


def _pair_summary_settings(pair: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: pair[key]
        for key in (
            "games_per_pair",
            "max_steps",
            "decision_ms",
            "decision_source_frames",
            "source_physics_step_ms",
            "num_simulations",
            "policy_mode",
            "collect_temperature",
            "collect_epsilon",
            "natural_bonus_spawn",
            "policy_trail_render_mode",
            "gif_trail_render_mode",
            "trail_render_mode",
            "frame_stride",
            "frame_size",
            "save_gif",
            "gif_sample_games_per_pair",
            "gif_sample_strategy",
        )
    }


def summarize_pair_from_tally(
    pair_spec: Mapping[str, Any],
    *,
    tally: Mapping[str, Any],
    first_gif_ref: str | None = None,
    game_summary_refs: Sequence[str] | None = None,
    games: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    pair = normalize_pair_spec(pair_spec)
    summary = {
        "schema_id": BATTLE_SCHEMA_ID,
        "ok": int(tally.get("failure_count") or 0) == 0,
        "tournament_id": pair["tournament_id"],
        "battle_id": pair["battle_id"],
        "pair_index": pair["pair_index"],
        "players": pair["players"],
        "settings": _pair_summary_settings(pair),
        "tally": _to_plain(dict(tally)),
        "first_gif_ref": first_gif_ref,
        "game_summary_ref_count": len(game_summary_refs or []),
    }
    for key in (
        "pair_key",
        "schedule_reason",
        "schedule_priority",
        "scheduled_round_index",
        "schedule",
    ):
        if key in pair:
            summary[key] = pair[key]
    if game_summary_refs is not None:
        summary["game_summary_refs"] = list(game_summary_refs)
    if games is not None:
        summary["games"] = [_compact_game_result(result) for result in games]
    return summary


def summarize_pair_results(pair_spec: Mapping[str, Any], game_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pair = normalize_pair_spec(pair_spec)
    ordered_results = sorted(
        game_results,
        key=lambda result: (
            int(result.get("game_index", 0) or 0),
            str(result.get("game_id") or ""),
        ),
    )
    tally = tally_game_results(ordered_results)
    first_gif_ref = None
    for result in ordered_results:
        if result.get("gif_ref"):
            first_gif_ref = result["gif_ref"]
            break
    return summarize_pair_from_tally(
        pair,
        tally=tally,
        first_gif_ref=first_gif_ref,
        game_summary_refs=[
            str(result.get("summary_ref"))
            for result in ordered_results
            if result.get("summary_ref")
        ],
        games=ordered_results,
    )


def _compact_game_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(result.get("ok")),
        "tournament_id": result.get("tournament_id"),
        "battle_id": result.get("battle_id"),
        "pair_index": result.get("pair_index"),
        "game_id": result.get("game_id"),
        "game_index": result.get("game_index"),
        "seed": result.get("seed"),
        "players": result.get("players"),
        "score": result.get("score"),
        "physical_steps": result.get("physical_steps"),
        "gif_ref": result.get("gif_ref"),
        "summary_ref": result.get("summary_ref"),
        "worker_timing": result.get("worker_timing"),
        "error": result.get("error"),
        "error_type": result.get("error_type"),
    }


def standings_from_pair_results(pair_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for pair in pair_results:
        players = pair.get("players") if isinstance(pair.get("players"), Sequence) else []
        for player in players:
            checkpoint_id = str(player.get("checkpoint_id"))
            rows.setdefault(
                checkpoint_id,
                {
                    "checkpoint_id": checkpoint_id,
                    "label": player.get("label"),
                    "checkpoint_ref": player.get("checkpoint_ref"),
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "games": 0,
                    "failures": 0,
                },
            )
        for game in pair.get("games", []):
            if not game.get("ok"):
                for player in players:
                    rows[str(player.get("checkpoint_id"))]["failures"] += 1
                continue
            score = game.get("score") if isinstance(game.get("score"), Mapping) else {}
            winner = score.get("winner_seat")
            loser = score.get("loser_seat")
            if winner in (0, 1):
                winner_id = str(players[int(winner)]["checkpoint_id"])
                loser_id = str(players[int(loser)]["checkpoint_id"])
                rows[winner_id]["wins"] += 1
                rows[loser_id]["losses"] += 1
                rows[winner_id]["games"] += 1
                rows[loser_id]["games"] += 1
            elif score.get("draw"):
                for player in players:
                    row = rows[str(player.get("checkpoint_id"))]
                    row["draws"] += 1
                    row["games"] += 1
    standings = sorted(
        rows.values(),
        key=lambda row: (
            -int(row["wins"]),
            int(row["losses"]),
            -int(row["draws"]),
            str(row["checkpoint_id"]),
        ),
    )
    for rank, row in enumerate(standings, start=1):
        row["rank"] = rank
        games = int(row["games"])
        row["win_rate"] = float(row["wins"]) / float(games) if games else None
    return {
        "schema_id": "curvyzero_curvytron_checkpoint_tournament_standings/v0",
        "checkpoint_count": len(standings),
        "standings": standings,
    }


def normalize_rating_spec(raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
    spec = dict(raw or {})
    checkpoints = spec.get("checkpoints") or spec.get("checkpoint_refs") or []
    if isinstance(checkpoints, str):
        checkpoints = parse_checkpoint_refs(checkpoints)
    if checkpoints and not isinstance(checkpoints, Sequence):
        raise ValueError("rating spec checkpoints must be a list or comma-separated refs")
    pair_selection = str(spec.get("pair_selection", DEFAULT_RATING_PAIR_SELECTION))
    if pair_selection not in RATING_PAIR_SELECTION_CHOICES:
        raise ValueError(
            f"pair_selection must be one of {RATING_PAIR_SELECTION_CHOICES!r}"
        )
    pairs_per_round_raw = spec.get("pairs_per_round")
    pairs_per_round = None
    if pairs_per_round_raw not in (None, "", 0, "0"):
        pairs_per_round = int(pairs_per_round_raw)
        if pairs_per_round < 1:
            raise ValueError("pairs_per_round must be positive or empty")
    if pair_selection == "adaptive_v0" and not pairs_per_round:
        raise ValueError("adaptive_v0 pair selection requires pairs_per_round")
    draw_score = float(spec.get("draw_score", DEFAULT_RATING_DRAW_SCORE))
    if draw_score != 0.5:
        raise ValueError("rating v0 requires draw_score=0.5")
    min_valid_fraction = float(
        spec.get("min_valid_fraction", DEFAULT_RATING_MIN_VALID_FRACTION)
    )
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in [0, 1]")
    policy_mode = str(spec.get("policy_mode", POLICY_MODE_EVAL))
    if policy_mode not in POLICY_MODE_CHOICES:
        raise ValueError(f"policy_mode must be one of {POLICY_MODE_CHOICES!r}")
    k_reference_games = float(
        spec.get("k_reference_games", DEFAULT_RATING_K_REFERENCE_GAMES)
    )
    if k_reference_games <= 0.0:
        raise ValueError("k_reference_games must be positive")
    normalized_checkpoints = normalize_checkpoint_specs(list(checkpoints)) if checkpoints else []
    round_count = int(spec.get("round_count", DEFAULT_RATING_ROUND_COUNT))
    if round_count < 1:
        raise ValueError("round_count must be at least 1")
    games_per_pair = _validate_games_per_pair(
        int(spec.get("games_per_pair", DEFAULT_GAMES_PER_PAIR))
    )
    games_per_shard = int(spec.get("games_per_shard", DEFAULT_GAMES_PER_SHARD))
    if games_per_shard < 1:
        raise ValueError("games_per_shard must be at least 1")
    reuse_policies_per_shard = bool(
        spec.get("reuse_policies_per_shard", DEFAULT_REUSE_POLICIES_PER_SHARD)
    )
    gif_sample_games_per_pair = int(
        spec.get("gif_sample_games_per_pair", DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR)
    )
    if gif_sample_games_per_pair < -1:
        raise ValueError("gif_sample_games_per_pair must be -1, 0, or positive")
    gif_sample_strategy = str(
        spec.get("gif_sample_strategy", DEFAULT_GIF_SAMPLE_STRATEGY)
    )
    if gif_sample_strategy not in GIF_SAMPLE_STRATEGY_CHOICES:
        raise ValueError(
            f"gif_sample_strategy must be one of {GIF_SAMPLE_STRATEGY_CHOICES!r}"
        )
    return {
        "schema_id": RATING_CONFIG_SCHEMA_ID,
        "formula_version": RATING_FORMULA_VERSION,
        "tournament_id": _safe_id(
            str(spec.get("tournament_id") or "tournament"),
            label="tournament_id",
        ),
        "rating_run_id": _safe_id(
            str(spec.get("rating_run_id") or DEFAULT_RATING_RUN_ID),
            label="rating_run_id",
        ),
        "checkpoints": normalized_checkpoints,
        "round_count": round_count,
        "pairs_per_round": pairs_per_round,
        "pair_selection": pair_selection,
        "games_per_pair": games_per_pair,
        "ordered_pairs": bool(spec.get("ordered_pairs", DEFAULT_ORDERED_PAIRS)),
        "include_self_pairs": bool(
            spec.get("include_self_pairs", DEFAULT_INCLUDE_SELF_PAIRS)
        ),
        "games_per_shard": games_per_shard,
        "reuse_policies_per_shard": reuse_policies_per_shard,
        "seed": int(spec.get("seed", 0)),
        "max_steps": int(spec.get("max_steps", DEFAULT_MAX_STEPS)),
        "decision_ms": float(spec.get("decision_ms", DEFAULT_DECISION_MS)),
        "decision_source_frames": spec.get("decision_source_frames"),
        "source_physics_step_ms": float(
            spec.get("source_physics_step_ms", DEFAULT_SOURCE_PHYSICS_STEP_MS)
        ),
        "num_simulations": int(spec.get("num_simulations", DEFAULT_NUM_SIMULATIONS)),
        "policy_batch_size": int(
            spec.get("policy_batch_size", DEFAULT_POLICY_BATCH_SIZE)
        ),
        "policy_mode": policy_mode,
        "collect_temperature": float(
            spec.get("collect_temperature", DEFAULT_COLLECT_TEMPERATURE)
        ),
        "collect_epsilon": float(
            spec.get("collect_epsilon", DEFAULT_COLLECT_EPSILON)
        ),
        "natural_bonus_spawn": bool(spec.get("natural_bonus_spawn", True)),
        "policy_trail_render_mode": (
            spec.get("policy_trail_render_mode")
            or spec.get("observation_trail_render_mode")
            or spec.get("trail_render_mode")
        ),
        "gif_trail_render_mode": DEFAULT_GIF_TRAIL_RENDER_MODE,
        "trail_render_mode": (
            spec.get("policy_trail_render_mode")
            or spec.get("observation_trail_render_mode")
            or spec.get("trail_render_mode")
        ),
        "frame_stride": int(spec.get("frame_stride", DEFAULT_FRAME_STRIDE)),
        "frame_size": DEFAULT_FRAME_SIZE,
        "gif_fps": float(spec.get("gif_fps", DEFAULT_GIF_FPS)),
        "save_gif": bool(spec.get("save_gif", DEFAULT_SAVE_GIF)),
        "gif_sample_games_per_pair": gif_sample_games_per_pair,
        "gif_sample_strategy": gif_sample_strategy,
        "save_frames_npz": bool(
            spec.get("save_frames_npz", DEFAULT_SAVE_FRAMES_NPZ)
        ),
        "action_trace_limit": int(spec.get("action_trace_limit", 128)),
        "initial_rating": float(
            spec.get("initial_rating", DEFAULT_RATING_INITIAL_RATING)
        ),
        "base_k": float(spec.get("base_k", DEFAULT_RATING_BASE_K)),
        "k_reference_games": k_reference_games,
        "k_min": float(spec.get("k_min", DEFAULT_RATING_K_MIN)),
        "k_max": float(spec.get("k_max", DEFAULT_RATING_K_MAX)),
        "delta_clamp": float(spec.get("delta_clamp", DEFAULT_RATING_DELTA_CLAMP)),
        "draw_score": draw_score,
        "min_valid_fraction": min_valid_fraction,
        "stop_max_delta": float(
            spec.get("stop_max_delta", DEFAULT_RATING_STOP_MAX_DELTA)
        ),
    }


def elo_expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (float(rating_b) - float(rating_a)) / 400.0))


def elo_k_for_games(valid_games: int, rating_spec: Mapping[str, Any]) -> float:
    if valid_games < 1:
        return 0.0
    base = float(rating_spec.get("base_k", DEFAULT_RATING_BASE_K))
    reference = float(
        rating_spec.get("k_reference_games", DEFAULT_RATING_K_REFERENCE_GAMES)
    )
    k_min = float(rating_spec.get("k_min", DEFAULT_RATING_K_MIN))
    k_max = float(rating_spec.get("k_max", DEFAULT_RATING_K_MAX))
    value = base * math.sqrt(float(valid_games) / reference)
    return max(k_min, min(k_max, value))


def clamp_delta(delta: float, rating_spec: Mapping[str, Any]) -> float:
    limit = abs(float(rating_spec.get("delta_clamp", DEFAULT_RATING_DELTA_CLAMP)))
    return max(-limit, min(limit, float(delta)))


def _rating_battle_id(
    *,
    rating_run_id: str,
    round_id: str,
    pair_slot: int,
    player_a: Mapping[str, Any],
    player_b: Mapping[str, Any],
) -> str:
    raw = (
        f"{rating_run_id}:{round_id}:{pair_slot}:"
        f"{player_a['checkpoint_id']}:{player_b['checkpoint_id']}"
    )
    digest = _short_hash(raw, length=10)
    short_round = str(round_id).replace("round-", "r")
    return _safe_id(
        "rate-"
        f"{_slug(str(rating_run_id), max_len=14)}-"
        f"{_slug(short_round, max_len=9)}-"
        f"pair-{int(pair_slot):06d}-"
        f"{_slug(str(player_a['checkpoint_id']), max_len=16)}-vs-"
        f"{_slug(str(player_b['checkpoint_id']), max_len=16)}-"
        f"{digest}",
        label="battle_id",
    )


def _pair_history_rows_by_key(
    pair_history: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not pair_history:
        return {}
    rows = pair_history.get("rows") or []
    if not isinstance(rows, Sequence):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping) and row.get("pair_key"):
            result[str(row["pair_key"])] = dict(row)
    return result


def _rating_row_value(
    rows: Mapping[str, Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    row = rows.get(str(checkpoint["checkpoint_id"]), {})
    return row.get(key, default)


def select_adaptive_v0_pair_slots(
    rating_spec: Mapping[str, Any],
    *,
    previous_snapshot: Mapping[str, Any] | None = None,
    scheduler_state: Mapping[str, Any] | None = None,
    pair_history: Mapping[str, Any] | None = None,
    round_index: int = 0,
) -> list[dict[str, Any]]:
    spec = normalize_rating_spec(rating_spec)
    budget = int(spec["pairs_per_round"] or 0)
    if budget < 1:
        raise ValueError("adaptive_v0 pair selection requires pairs_per_round")
    checkpoints = spec["checkpoints"]
    if len(checkpoints) < 2 and not spec["include_self_pairs"]:
        raise ValueError("at least two checkpoints are needed for adaptive_v0")

    expected_pool_hash = rating_pool_hash(checkpoints)
    for state, label in (
        (scheduler_state, "scheduler_state"),
        (pair_history, "pair_history"),
    ):
        if isinstance(state, Mapping) and state.get("pool_hash"):
            if str(state["pool_hash"]) != expected_pool_hash:
                raise ValueError(f"{label} pool_hash does not match rating spec")

    current_rows = _rating_rows_by_checkpoint(previous_snapshot)
    history_by_key = _pair_history_rows_by_key(pair_history)
    rng = random.Random(int(spec["seed"]) + int(round_index) * 1_000_003 + 17)
    rows = []
    for index, checkpoint in enumerate(checkpoints):
        checkpoint_id = str(checkpoint["checkpoint_id"])
        games = int(_rating_row_value(current_rows, checkpoint, "games", 0) or 0)
        distinct = int(
            _rating_row_value(current_rows, checkpoint, "distinct_opponents", 0)
            or len(_rating_row_value(current_rows, checkpoint, "opponent_ids", []) or [])
            or 0
        )
        rows.append(
            {
                "index": index,
                "checkpoint": checkpoint,
                "checkpoint_id": checkpoint_id,
                "rating": float(
                    _rating_row_value(
                        current_rows,
                        checkpoint,
                        "rating",
                        spec["initial_rating"],
                    )
                ),
                "games": games,
                "distinct_opponents": distinct,
                "rated_battles": int(
                    _rating_row_value(current_rows, checkpoint, "rated_battles", 0)
                    or 0
                ),
                "last_round_delta": abs(
                    float(
                        _rating_row_value(
                            current_rows,
                            checkpoint,
                            "last_round_delta",
                            0.0,
                        )
                        or 0.0
                    )
                ),
                "failure_count": int(
                    _rating_row_value(current_rows, checkpoint, "failure_count", 0)
                    or 0
                ),
            }
        )

    by_rating = sorted(rows, key=lambda row: (float(row["rating"]), str(row["checkpoint_id"])))
    rating_position = {int(row["index"]): pos for pos, row in enumerate(by_rating)}
    anchor_indices = []
    if by_rating:
        for pos in {0, len(by_rating) // 4, len(by_rating) // 2, len(by_rating) - 1}:
            if 0 <= pos < len(by_rating):
                anchor_indices.append(int(by_rating[pos]["index"]))
    anchor_indices = sorted(set(anchor_indices))

    slots: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    considered = 0

    def add_pair(i: int, j: int, reason: str, priority: float) -> bool:
        nonlocal considered
        considered += 1
        if len(slots) >= budget:
            return False
        if not spec["include_self_pairs"] and i == j:
            return False
        if i < 0 or j < 0 or i >= len(checkpoints) or j >= len(checkpoints):
            return False
        player_a = checkpoints[i]
        player_b = checkpoints[j]
        pair_key = rating_pair_key(
            str(player_a["checkpoint_id"]),
            str(player_b["checkpoint_id"]),
        )
        if pair_key in used_keys:
            return False
        used_keys.add(pair_key)
        history = history_by_key.get(pair_key, {})
        priority_value = float(priority)
        if history.get("battle_count"):
            priority_value *= 0.5
        slots.append(
            {
                "player_a_index": i,
                "player_b_index": j,
                "pair_key": pair_key,
                "schedule_reason": reason,
                "schedule_priority": priority_value,
                "scheduled_round_index": int(round_index),
                "prior_battle_count": int(history.get("battle_count") or 0),
            }
        )
        return True

    placement_budget = max(1, int(round(budget * 0.2)))
    low_coverage = sorted(
        rows,
        key=lambda row: (
            int(row["distinct_opponents"]),
            int(row["games"]),
            str(row["checkpoint_id"]),
        ),
    )
    for row in low_coverage:
        if len([slot for slot in slots if slot["schedule_reason"] == "placement"]) >= placement_budget:
            break
        target = int(row["index"])
        candidates = [anchor for anchor in anchor_indices if anchor != target]
        if not candidates:
            candidates = [int(other["index"]) for other in rows if int(other["index"]) != target]
        rng.shuffle(candidates)
        for opponent in candidates:
            if add_pair(target, opponent, "placement", 1.0):
                break

    near_budget = max(1, int(round(budget * 0.6)))
    target_order = list(range(len(by_rating)))
    rng.shuffle(target_order)
    for pos in target_order:
        if len([slot for slot in slots if slot["schedule_reason"] == "near_rating"]) >= near_budget:
            break
        row = by_rating[pos]
        index = int(row["index"])
        window = []
        for offset in range(1, 9):
            for neighbor_pos in (pos - offset, pos + offset):
                if 0 <= neighbor_pos < len(by_rating):
                    window.append(int(by_rating[neighbor_pos]["index"]))
        for opponent in window:
            if add_pair(index, opponent, "near_rating", 0.9):
                break

    uncertain_budget = max(1, int(round(budget * 0.2)))
    uncertain_rows = sorted(
        rows,
        key=lambda row: (
            -float(row["last_round_delta"]),
            int(row["distinct_opponents"]),
            int(row["games"]),
            -int(row["failure_count"]),
            str(row["checkpoint_id"]),
        ),
    )
    for row in uncertain_rows:
        if len([slot for slot in slots if slot["schedule_reason"] == "uncertain"]) >= uncertain_budget:
            break
        index = int(row["index"])
        pos = rating_position[index]
        nearby = [
            int(by_rating[neighbor_pos]["index"])
            for neighbor_pos in range(max(0, pos - 4), min(len(by_rating), pos + 5))
            if neighbor_pos != pos
        ]
        rng.shuffle(nearby)
        for opponent in nearby + anchor_indices:
            if add_pair(index, opponent, "uncertain", 0.8):
                break

    bridge_attempts = max(budget * 20, 20)
    for _attempt in range(bridge_attempts):
        if len(slots) >= budget:
            break
        i = rng.randrange(len(checkpoints))
        j = rng.randrange(len(checkpoints))
        if add_pair(i, j, "random_bridge", 0.4):
            continue

    for pos, row in enumerate(by_rating):
        if len(slots) >= budget:
            break
        index = int(row["index"])
        for neighbor_pos in range(pos + 1, min(len(by_rating), pos + 17)):
            if add_pair(index, int(by_rating[neighbor_pos]["index"]), "fill", 0.1):
                break

    for slot_index, slot in enumerate(slots):
        slot["schedule_slot"] = slot_index
    return slots


def build_rating_round_pair_specs(
    rating_spec: Mapping[str, Any],
    *,
    previous_snapshot: Mapping[str, Any] | None = None,
    scheduler_state: Mapping[str, Any] | None = None,
    pair_history: Mapping[str, Any] | None = None,
    round_index: int = 0,
) -> list[dict[str, Any]]:
    spec = normalize_rating_spec(rating_spec)
    checkpoints = spec["checkpoints"]
    if len(checkpoints) < 2 and not spec["include_self_pairs"]:
        raise ValueError("at least two checkpoints are needed for rating")
    current_ratings = _rating_rows_by_checkpoint(previous_snapshot)
    candidates: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    for i, player_a in enumerate(checkpoints):
        for j, player_b in enumerate(checkpoints):
            if not spec["include_self_pairs"] and i == j:
                continue
            if not spec["ordered_pairs"] and (
                j < i or (not spec["include_self_pairs"] and j == i)
            ):
                continue
            candidates.append((i, j, player_a, player_b))

    if spec["pair_selection"] == "random" and spec["pairs_per_round"]:
        count = min(int(spec["pairs_per_round"]), len(candidates))
        rng = random.Random(int(spec["seed"]) + int(round_index) * 1_000_003)
        candidates = rng.sample(candidates, count)
    elif spec["pairs_per_round"]:
        candidates = candidates[: int(spec["pairs_per_round"])]

    def sort_key(item: tuple[int, int, dict[str, Any], dict[str, Any]]) -> tuple[Any, ...]:
        i, j, player_a, player_b = item
        if spec["pair_selection"] == "random":
            return (i, j)
        rating_a = float(
            current_ratings.get(str(player_a["checkpoint_id"]), {}).get(
                "rating",
                spec["initial_rating"],
            )
        )
        rating_b = float(
            current_ratings.get(str(player_b["checkpoint_id"]), {}).get(
                "rating",
                spec["initial_rating"],
            )
        )
        return (abs(rating_a - rating_b), i, j)

    if spec["pair_selection"] != "random":
        candidates.sort(key=sort_key)
    round_id = rating_round_id(round_index)
    pair_specs = []
    for pair_slot, (_i, _j, player_a, player_b) in enumerate(candidates):
        battle_id = _rating_battle_id(
            rating_run_id=spec["rating_run_id"],
            round_id=round_id,
            pair_slot=pair_slot,
            player_a=player_a,
            player_b=player_b,
        )
        pair_specs.append(
            normalize_pair_spec(
                {
                    "tournament_id": spec["tournament_id"],
                    "battle_id": battle_id,
                    "pair_index": pair_slot,
                    "players": [
                        {"seat": 0, **player_a},
                        {"seat": 1, **player_b},
                    ],
                    "games_per_pair": spec["games_per_pair"],
                    "games_per_shard": spec["games_per_shard"],
                    "reuse_policies_per_shard": spec["reuse_policies_per_shard"],
                    "seed": int(spec["seed"])
                    + int(round_index) * 1_000_000
                    + pair_slot * 10_000,
                    "max_steps": spec["max_steps"],
                    "decision_ms": spec["decision_ms"],
                    "decision_source_frames": spec["decision_source_frames"],
                    "source_physics_step_ms": spec["source_physics_step_ms"],
                    "num_simulations": spec["num_simulations"],
                    "policy_batch_size": spec["policy_batch_size"],
                    "policy_mode": spec["policy_mode"],
                    "collect_temperature": spec["collect_temperature"],
                    "collect_epsilon": spec["collect_epsilon"],
                    "natural_bonus_spawn": spec["natural_bonus_spawn"],
                    "policy_trail_render_mode": spec["policy_trail_render_mode"],
                    "gif_trail_render_mode": spec["gif_trail_render_mode"],
                    "trail_render_mode": spec["trail_render_mode"],
                    "frame_stride": spec["frame_stride"],
                    "frame_size": spec["frame_size"],
                    "gif_fps": spec["gif_fps"],
                    "save_gif": spec["save_gif"],
                    "gif_sample_games_per_pair": spec["gif_sample_games_per_pair"],
                    "gif_sample_strategy": spec["gif_sample_strategy"],
                    "save_frames_npz": spec["save_frames_npz"],
                    "action_trace_limit": spec["action_trace_limit"],
                }
            )
        )
    return pair_specs


def _rating_rows_by_checkpoint(
    snapshot: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not snapshot:
        return {}
    rows = snapshot.get("ratings") or snapshot.get("rows") or []
    if not isinstance(rows, Sequence):
        return {}
    result = {}
    for row in rows:
        if isinstance(row, Mapping) and row.get("checkpoint_id"):
            result[str(row["checkpoint_id"])] = dict(row)
    return result


def _base_rating_rows(
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    previous_snapshot: Mapping[str, Any] | None,
    rating_spec: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    previous = _rating_rows_by_checkpoint(previous_snapshot)
    rows = {}
    for checkpoint in checkpoints:
        checkpoint_id = str(checkpoint["checkpoint_id"])
        prior = previous.get(checkpoint_id, {})
        rows[checkpoint_id] = {
            "checkpoint_id": checkpoint_id,
            "label": checkpoint.get("label") or prior.get("label") or checkpoint_id,
            "checkpoint_ref": checkpoint.get("checkpoint_ref")
            or prior.get("checkpoint_ref"),
            "rating": float(
                prior.get("rating", rating_spec.get("initial_rating", DEFAULT_RATING_INITIAL_RATING))
            ),
            "games": int(prior.get("games", 0) or 0),
            "wins": int(prior.get("wins", 0) or 0),
            "losses": int(prior.get("losses", 0) or 0),
            "draws": int(prior.get("draws", 0) or 0),
            "failure_count": int(prior.get("failure_count", 0) or prior.get("failures", 0) or 0),
            "battles": int(prior.get("battles", 0) or 0),
            "rated_battles": int(prior.get("rated_battles", 0) or 0),
            "opponent_ids": sorted(str(item) for item in prior.get("opponent_ids", []) or []),
            "last_battle_ref": prior.get("last_battle_ref"),
        }
    return rows


def rating_result_from_pair_summary(
    pair_summary: Mapping[str, Any],
    rating_spec: Mapping[str, Any],
) -> dict[str, Any]:
    spec = normalize_rating_spec(rating_spec)
    players = pair_summary.get("players")
    if not isinstance(players, Sequence) or len(players) != 2:
        raise ValueError("pair summary needs exactly two players for rating")
    player_a = dict(players[0])
    player_b = dict(players[1])
    checkpoint_a = str(player_a["checkpoint_id"])
    checkpoint_b = str(player_b["checkpoint_id"])
    games = pair_summary.get("games")
    if not isinstance(games, Sequence):
        games = []
    requested_games = int(
        (pair_summary.get("settings") or {}).get(
            "games_per_pair",
            (pair_summary.get("tally") or {}).get("game_count", len(games)),
        )
        or len(games)
    )
    wins_a = 0
    wins_b = 0
    draws = 0
    failure_count = 0
    invalid_count = 0
    if games:
        for game in games:
            if not isinstance(game, Mapping):
                invalid_count += 1
                continue
            if not game.get("ok"):
                failure_count += 1
                continue
            score = game.get("score") if isinstance(game.get("score"), Mapping) else {}
            outcome = str(score.get("outcome") or "")
            winner = score.get("winner_seat")
            if score.get("draw") or outcome == "draw":
                draws += 1
            elif winner == 0 or outcome == "seat_0_win":
                wins_a += 1
            elif winner == 1 or outcome == "seat_1_win":
                wins_b += 1
            else:
                invalid_count += 1
    else:
        tally = pair_summary.get("tally") if isinstance(pair_summary.get("tally"), Mapping) else {}
        wins_by_seat = tally.get("wins_by_seat") if isinstance(tally.get("wins_by_seat"), Mapping) else {}
        wins_a = int(wins_by_seat.get("seat_0") or 0)
        wins_b = int(wins_by_seat.get("seat_1") or 0)
        draws = int(tally.get("draw_count") or 0)
        failure_count = int(tally.get("failure_count") or 0)
        tally_game_count = int(tally.get("game_count") or requested_games)
        invalid_count = max(0, tally_game_count - wins_a - wins_b - draws - failure_count)
    valid_games = wins_a + wins_b + draws
    min_valid_games = math.ceil(float(requested_games) * float(spec["min_valid_fraction"]))
    rated = valid_games > 0 and valid_games >= min_valid_games
    reason = "rated" if rated else "not_enough_valid_games"
    score_a = None
    score_b = None
    if valid_games:
        score_a = (float(wins_a) + float(spec["draw_score"]) * float(draws)) / float(valid_games)
        score_b = (float(wins_b) + float(spec["draw_score"]) * float(draws)) / float(valid_games)
    return {
        "battle_id": pair_summary.get("battle_id"),
        "pair_index": int(pair_summary.get("pair_index", 0) or 0),
        "summary_ref": pair_summary.get("summary_ref"),
        "checkpoint_a": checkpoint_a,
        "checkpoint_b": checkpoint_b,
        "label_a": player_a.get("label"),
        "label_b": player_b.get("label"),
        "requested_games": requested_games,
        "valid_games": int(valid_games),
        "wins_a": int(wins_a),
        "wins_b": int(wins_b),
        "draws": int(draws),
        "failure_count": int(failure_count),
        "invalid_count": int(invalid_count),
        "score_a": score_a,
        "score_b": score_b,
        "rated": bool(rated),
        "rating_skip_reason": None if rated else reason,
    }


def rating_snapshot_from_pair_results(
    *,
    pair_results: Sequence[Mapping[str, Any]],
    rating_spec: Mapping[str, Any],
    previous_snapshot: Mapping[str, Any] | None = None,
    round_index: int = 0,
    created_at: str | None = None,
) -> dict[str, Any]:
    spec = normalize_rating_spec(rating_spec)
    checkpoints = spec["checkpoints"]
    rows = _base_rating_rows(
        checkpoints,
        previous_snapshot=previous_snapshot,
        rating_spec=spec,
    )
    pair_rating_results = [
        rating_result_from_pair_summary(pair, spec)
        for pair in pair_results
    ]
    pair_rating_results.sort(
        key=lambda item: (
            int(item.get("pair_index", 0) or 0),
            str(item.get("battle_id") or ""),
        )
    )
    start_ratings = {
        checkpoint_id: float(row["rating"])
        for checkpoint_id, row in rows.items()
    }
    deltas = Counter()
    rated_pair_count = 0
    for result in pair_rating_results:
        checkpoint_a = str(result["checkpoint_a"])
        checkpoint_b = str(result["checkpoint_b"])
        for checkpoint_id, label_key in (
            (checkpoint_a, "label_a"),
            (checkpoint_b, "label_b"),
        ):
            if checkpoint_id not in rows:
                rows[checkpoint_id] = {
                    "checkpoint_id": checkpoint_id,
                    "label": result.get(label_key) or checkpoint_id,
                    "checkpoint_ref": None,
                    "rating": float(spec["initial_rating"]),
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "failure_count": 0,
                    "battles": 0,
                    "rated_battles": 0,
                    "opponent_ids": [],
                    "last_battle_ref": None,
                }
                start_ratings[checkpoint_id] = float(spec["initial_rating"])
        row_a = rows[checkpoint_a]
        row_b = rows[checkpoint_b]
        opponents_a = set(row_a.get("opponent_ids") or [])
        opponents_b = set(row_b.get("opponent_ids") or [])
        if checkpoint_a != checkpoint_b:
            opponents_a.add(checkpoint_b)
            opponents_b.add(checkpoint_a)
        row_a["opponent_ids"] = sorted(opponents_a)
        row_b["opponent_ids"] = sorted(opponents_b)
        row_a["battles"] += 1
        row_b["battles"] += 1
        row_a["failure_count"] += int(result["failure_count"])
        row_b["failure_count"] += int(result["failure_count"])
        valid_games = int(result["valid_games"])
        row_a["games"] += valid_games
        row_b["games"] += valid_games
        row_a["wins"] += int(result["wins_a"])
        row_a["losses"] += int(result["wins_b"])
        row_b["wins"] += int(result["wins_b"])
        row_b["losses"] += int(result["wins_a"])
        row_a["draws"] += int(result["draws"])
        row_b["draws"] += int(result["draws"])
        row_a["last_battle_ref"] = result.get("summary_ref") or row_a.get("last_battle_ref")
        row_b["last_battle_ref"] = result.get("summary_ref") or row_b.get("last_battle_ref")
        if not result["rated"]:
            continue
        rated_pair_count += 1
        row_a["rated_battles"] += 1
        row_b["rated_battles"] += 1
        expected_a = elo_expected_score(
            start_ratings[checkpoint_a],
            start_ratings[checkpoint_b],
        )
        observed_a = float(result["score_a"])
        k_pair = elo_k_for_games(valid_games, spec)
        delta_a = clamp_delta(k_pair * (observed_a - expected_a), spec)
        deltas[checkpoint_a] += delta_a
        deltas[checkpoint_b] -= delta_a
        result["rating"] = {
            "expected_a": expected_a,
            "observed_a": observed_a,
            "k_pair": k_pair,
            "delta_a": delta_a,
            "rating_a_before": start_ratings[checkpoint_a],
            "rating_b_before": start_ratings[checkpoint_b],
        }

    standings = []
    max_abs_delta = 0.0
    for checkpoint_id, row in rows.items():
        delta = float(deltas[checkpoint_id])
        max_abs_delta = max(max_abs_delta, abs(delta))
        row["previous_rating"] = float(row["rating"])
        row["last_round_delta"] = delta
        row["rating"] = float(row["rating"]) + delta
        row["distinct_opponents"] = len(row.get("opponent_ids") or [])
        games = int(row["games"])
        row["win_rate"] = float(row["wins"]) / float(games) if games else None
        row["status"] = (
            "active"
            if games >= 300 and int(row["distinct_opponents"]) >= 5
            else "provisional"
        )
        standings.append(row)
    standings.sort(
        key=lambda row: (
            -float(row["rating"]),
            -int(row["games"]),
            str(row["checkpoint_id"]),
        )
    )
    for rank, row in enumerate(standings, start=1):
        row["rank"] = rank
    round_id = rating_round_id(round_index)
    return {
        "schema_id": RATING_SNAPSHOT_SCHEMA_ID,
        "formula_version": RATING_FORMULA_VERSION,
        "tournament_id": spec["tournament_id"],
        "rating_run_id": spec["rating_run_id"],
        "round_id": round_id,
        "round_index": int(round_index),
        "created_at": created_at,
        "rating_spec": {
            key: value
            for key, value in spec.items()
            if key != "checkpoints"
        },
        "checkpoint_count": len(standings),
        "pair_count": len(pair_rating_results),
        "rated_pair_count": int(rated_pair_count),
        "invalid_pair_count": int(len(pair_rating_results) - rated_pair_count),
        "max_abs_delta": float(max_abs_delta),
        "stable": bool(max_abs_delta <= float(spec["stop_max_delta"])),
        "ratings": _to_plain(standings),
        "pair_rating_results": _to_plain(pair_rating_results),
    }


def write_json_artifact(mount: Path, ref: PurePosixPath, payload: Any) -> dict[str, Any]:
    path = runs.volume_path(mount, ref)
    summary = runs.write_json(path, _to_plain(payload))
    summary["ref"] = ref.as_posix()
    return summary


def _save_gif(frames: Sequence[Any], path: Path, *, fps: float) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    raw_frames = np.asarray(frames, dtype=np.uint8)
    if raw_frames.ndim != 4 or raw_frames.shape[-1] != 3:
        raise ValueError("GIF frames must have shape [N, H, W, 3]")
    if raw_frames.shape[0] < 1:
        raise ValueError("GIF needs at least one frame")
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, int(round(1000.0 / float(fps))))
    pil_frames = [Image.fromarray(frame, mode="RGB") for frame in raw_frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return {
        "path": str(path),
        "frame_count": int(raw_frames.shape[0]),
        "duration_ms_per_frame": int(duration_ms),
        "pixel_size": [int(raw_frames.shape[2]), int(raw_frames.shape[1])],
        "color_mode": "RGB",
    }


def _write_frames_npz(frames: Sequence[Any], path: Path, *, metadata: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=np.asarray(frames, dtype=np.uint8),
        metadata=json.dumps(_to_plain(metadata), sort_keys=True).encode("utf-8"),
    )
    return {"path": str(path), "bytes": path.stat().st_size}


def _lookup_state_dict_by_key(payload: Any, key: str) -> Any:
    if key == "<root>":
        return payload
    current = payload
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"checkpoint payload has no state key {key!r}")
        current = current[part]
    return current


def _checkpoint_policy_trail_render_mode_from_payload(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    candidates: list[Any] = []
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        candidates.extend(
            [
                metadata.get("policy_trail_render_mode"),
                metadata.get("trail_render_mode"),
                metadata.get("source_state_trail_render_mode"),
            ]
        )
        observation_contract = metadata.get("observation_contract")
        if isinstance(observation_contract, Mapping):
            candidates.append(observation_contract.get("trail_render_mode"))
    config = payload.get("config")
    if isinstance(config, Mapping):
        candidates.extend(
            [
                config.get("policy_trail_render_mode"),
                config.get("source_state_trail_render_mode"),
                config.get("trail_render_mode"),
            ]
        )
    candidates.extend(
        [
            payload.get("policy_trail_render_mode"),
            payload.get("source_state_trail_render_mode"),
            payload.get("trail_render_mode"),
        ]
    )
    for value in candidates:
        if value:
            return str(value)
    return None


def _checkpoint_runtime_settings_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    candidates: list[Mapping[str, Any]] = []
    for key in ("config", "metadata", "runtime_settings"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    candidates.append(payload)
    settings: dict[str, Any] = {}
    for candidate in candidates:
        for key in ("decision_source_frames", "source_physics_step_ms", "decision_ms"):
            if key in settings:
                continue
            value = candidate.get(key)
            if value is not None:
                settings[key] = value
    return settings


def _checkpoint_model_contract_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    candidates: list[Mapping[str, Any]] = []
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        candidates.append(metadata)
        observation_contract = metadata.get("observation_contract")
        if isinstance(observation_contract, Mapping):
            candidates.append(observation_contract)
    config = payload.get("config")
    if isinstance(config, Mapping):
        candidates.append(config)
    candidates.append(payload)

    model_env_variant = None
    model_reward_variant = None
    for candidate in candidates:
        if model_env_variant is None:
            model_env_variant = (
                candidate.get("model_env_variant")
                or candidate.get("training_env_variant")
                or candidate.get("env_variant")
            )
        if model_reward_variant is None:
            model_reward_variant = (
                candidate.get("model_reward_variant")
                or candidate.get("training_reward_variant")
                or candidate.get("reward_variant")
            )
    contract: dict[str, Any] = {}
    if model_env_variant:
        contract["model_env_variant"] = str(model_env_variant)
    if model_reward_variant:
        contract["model_reward_variant"] = str(model_reward_variant)
    return contract


def _checkpoint_model_contract_from_ref(
    checkpoint_ref: str,
    *,
    mount: Path,
) -> dict[str, Any]:
    path = runs.require_relative_ref(checkpoint_ref)
    parts = path.parts
    metadata_refs: list[PurePosixPath] = []
    if len(parts) >= 5 and parts[0] == "training" and parts[3] == "attempts":
        attempt_root = PurePosixPath(*parts[:5])
        metadata_refs.append(attempt_root / "command.json")
        metadata_refs.append(attempt_root / "attempt.json")
    if len(parts) >= 3 and parts[0] == "training":
        run_root = PurePosixPath(*parts[:3])
        metadata_refs.append(run_root / "run.json")

    contract: dict[str, Any] = {}
    for ref in metadata_refs:
        metadata_path = runs.volume_path(mount, ref)
        if not metadata_path.exists():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        payload_contract = _checkpoint_model_contract_from_payload(payload)
        for key, value in payload_contract.items():
            contract.setdefault(key, value)
    return contract


def _checkpoint_policy_trail_render_mode_from_ref(
    checkpoint_ref: str,
    *,
    mount: Path,
) -> str | None:
    path = runs.require_relative_ref(checkpoint_ref)
    parts = path.parts
    metadata_refs: list[PurePosixPath] = []
    if len(parts) >= 3 and parts[0] == "training":
        run_root = PurePosixPath(*parts[:3])
        metadata_refs.append(run_root / "run.json")
    if len(parts) >= 5 and parts[0] == "training" and parts[3] == "attempts":
        attempt_root = PurePosixPath(*parts[:5])
        metadata_refs.append(attempt_root / "attempt.json")
        metadata_refs.append(attempt_root / "command.json")
    for ref in metadata_refs:
        metadata_path = runs.volume_path(mount, ref)
        if not metadata_path.exists():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        mode = _checkpoint_policy_trail_render_mode_from_payload(payload)
        if mode:
            return mode
    return None


def _checkpoint_runtime_settings_from_ref(
    checkpoint_ref: str,
    *,
    mount: Path,
) -> dict[str, Any]:
    path = runs.require_relative_ref(checkpoint_ref)
    parts = path.parts
    metadata_refs: list[PurePosixPath] = []
    if len(parts) >= 3 and parts[0] == "training":
        run_root = PurePosixPath(*parts[:3])
        metadata_refs.append(run_root / "run.json")
    if len(parts) >= 5 and parts[0] == "training" and parts[3] == "attempts":
        attempt_root = PurePosixPath(*parts[:5])
        metadata_refs.append(attempt_root / "attempt.json")
        metadata_refs.append(attempt_root / "command.json")
    for ref in metadata_refs:
        metadata_path = runs.volume_path(mount, ref)
        if not metadata_path.exists():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        settings = _checkpoint_runtime_settings_from_payload(payload)
        if settings:
            return settings
    return {}


def _load_policy_from_checkpoint(
    *,
    checkpoint_ref: str,
    checkpoint_state_key: str | None,
    seed: int,
    source_max_steps: int,
    num_simulations: int,
    batch_size: int,
    telemetry_path: Path,
    mount: Path,
    remote_root: Path | None,
    model_env_variant: str | None,
    model_reward_variant: str | None,
) -> dict[str, Any]:
    from curvyzero.infra.modal import lightzero_curvytron_visual_survival_eval as eval_mod
    from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
        DEFAULT_OPPONENT_DEATH_MODE,
        DEFAULT_REWARD_VARIANT,
        ENV_VARIANT_SOURCE_STATE_TURN_COMMIT,
        OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
    )

    checkpoint_path, resolution = runs.resolve_mounted_ref_or_path(
        checkpoint_ref,
        mount=mount,
        remote_root=remote_root,
    )
    payload = eval_mod._torch_load(checkpoint_path)
    policy_trail_render_mode = (
        _checkpoint_policy_trail_render_mode_from_payload(payload)
        or _checkpoint_policy_trail_render_mode_from_ref(checkpoint_ref, mount=mount)
    )
    payload_model_contract = _checkpoint_model_contract_from_payload(payload)
    ref_model_contract = _checkpoint_model_contract_from_ref(checkpoint_ref, mount=mount)
    effective_model_env_variant = (
        model_env_variant
        or payload_model_contract.get("model_env_variant")
        or ref_model_contract.get("model_env_variant")
    )
    effective_model_reward_variant = (
        model_reward_variant
        or payload_model_contract.get("model_reward_variant")
        or ref_model_contract.get("model_reward_variant")
    )
    runtime_settings = {
        **_checkpoint_runtime_settings_from_ref(checkpoint_ref, mount=mount),
        **_checkpoint_runtime_settings_from_payload(payload),
    }
    if checkpoint_state_key:
        state_dict = _lookup_state_dict_by_key(payload, checkpoint_state_key)
        found_key = checkpoint_state_key
    else:
        found = eval_mod._find_state_dict(payload)
        if found is None:
            raise ValueError("checkpoint payload did not contain a LightZero state dict")
        found_key, state_dict = found
    if not isinstance(state_dict, dict):
        raise ValueError("selected checkpoint state is not a dict")
    policy, unused_env, surface = eval_mod._make_policy_and_env(
        state_dict=state_dict,
        seed=int(seed),
        use_cuda=False,
        source_max_steps=int(source_max_steps),
        num_simulations=int(num_simulations),
        batch_size=int(batch_size),
        telemetry_path=telemetry_path,
        env_variant=ENV_VARIANT_SOURCE_STATE_TURN_COMMIT,
        reward_variant=DEFAULT_REWARD_VARIANT,
        model_env_variant=(
            str(effective_model_env_variant) if effective_model_env_variant else None
        ),
        model_reward_variant=(
            str(effective_model_reward_variant)
            if effective_model_reward_variant
            else None
        ),
        opponent_policy_kind=OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_checkpoint=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        opponent_death_mode=DEFAULT_OPPONENT_DEATH_MODE,
        natural_bonus_spawn=True,
    )
    if hasattr(unused_env, "close"):
        try:
            unused_env.close()
        except Exception:
            pass
    return {
        "policy": policy,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_resolution": resolution,
        "checkpoint_state_key": found_key,
        "policy_trail_render_mode": policy_trail_render_mode,
        "runtime_settings": runtime_settings,
        "model_env_variant": (
            str(effective_model_env_variant) if effective_model_env_variant else None
        ),
        "model_reward_variant": (
            str(effective_model_reward_variant)
            if effective_model_reward_variant
            else None
        ),
        "model_contract_source": {
            "explicit_model_env_variant": model_env_variant,
            "explicit_model_reward_variant": model_reward_variant,
            "payload": payload_model_contract,
            "metadata": ref_model_contract,
        },
        "surface": surface,
    }


def load_policy_entries_for_game(
    spec: Mapping[str, Any],
    *,
    checkpoint_mount: Path | None = None,
    artifact_mount: Path | None = None,
    mount: Path | None = None,
    remote_root: Path | None = None,
) -> list[dict[str, Any]]:
    if mount is not None:
        checkpoint_mount = checkpoint_mount or mount
        artifact_mount = artifact_mount or mount
    if checkpoint_mount is None or artifact_mount is None:
        raise ValueError("load_policy_entries_for_game needs checkpoint_mount and artifact_mount")

    game = dict(spec)
    pair = normalize_pair_spec({**game, "games_per_pair": 1})
    game_id = _safe_id(str(game.get("game_id") or "game-000000"), label="game_id")
    seed = int(game.get("seed", pair["seed"]))
    max_steps = int(game.get("max_steps", pair["max_steps"]))
    root_path = runs.volume_path(
        artifact_mount,
        game_root_ref(pair["tournament_id"], pair["battle_id"], game_id),
    )
    root_path.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for player in pair["players"]:
        entries.append(
            _load_policy_from_checkpoint(
                checkpoint_ref=str(player["checkpoint_ref"]),
                checkpoint_state_key=(
                    str(player["checkpoint_state_key"])
                    if player.get("checkpoint_state_key")
                    else None
                ),
                seed=seed + int(player["seat"]),
                source_max_steps=max_steps,
                num_simulations=int(game.get("num_simulations", pair["num_simulations"])),
                batch_size=int(game.get("policy_batch_size", pair["policy_batch_size"])),
                telemetry_path=root_path / f"policy_seat_{player['seat']}_loader_telemetry.jsonl",
                mount=checkpoint_mount,
                remote_root=remote_root,
                model_env_variant=(
                    str(player["model_env_variant"])
                    if player.get("model_env_variant")
                    else None
                ),
                model_reward_variant=(
                    str(player["model_reward_variant"])
                    if player.get("model_reward_variant")
                    else None
                ),
            )
        )
    return entries


def _policy_action(
    *,
    policy: Any,
    observation: Mapping[str, Any],
    policy_mode: str,
    collect_temperature: float,
    collect_epsilon: float,
) -> dict[str, Any]:
    from curvyzero.infra.modal import lightzero_curvytron_visual_survival_eval as eval_mod

    if policy_mode == POLICY_MODE_EVAL:
        result = dict(eval_mod._policy_eval_action(policy, dict(observation)))
        result["policy_mode"] = POLICY_MODE_EVAL
        return result
    if policy_mode != POLICY_MODE_COLLECT:
        raise ValueError(f"policy_mode must be one of {POLICY_MODE_CHOICES!r}")
    if collect_temperature <= 0.0:
        raise ValueError("collect_temperature must be positive")
    if not 0.0 <= collect_epsilon <= 1.0:
        raise ValueError("collect_epsilon must be in [0, 1]")

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
        "policy_mode": POLICY_MODE_COLLECT,
        "action": eval_mod._extract_eval_action(output),
        "temperature": float(collect_temperature),
        "epsilon": float(collect_epsilon),
        "compact_output": eval_mod._compact_mcts_output(output),
    }


def _consistent_runtime_value(
    policy_entries: Sequence[Mapping[str, Any]],
    key: str,
) -> Any | None:
    values = []
    for entry in policy_entries:
        runtime_settings = entry.get("runtime_settings")
        if not isinstance(runtime_settings, Mapping):
            continue
        value = runtime_settings.get(key)
        if value is not None:
            values.append(value)
    if not values:
        return None
    first = values[0]
    for value in values[1:]:
        try:
            if not math.isclose(float(value), float(first), rtol=0.0, abs_tol=1e-6):
                raise ValueError
        except (TypeError, ValueError):
            if str(value) != str(first):
                raise ValueError(
                    f"mixed checkpoint runtime setting {key}: {values!r}"
                ) from None
    return first


def _source_frame_runtime_settings(
    game: Mapping[str, Any],
    pair: Mapping[str, Any],
    policy_entries: Sequence[Mapping[str, Any]],
    *,
    max_steps: int,
) -> dict[str, Any]:
    runtime_source_physics = _consistent_runtime_value(
        policy_entries,
        "source_physics_step_ms",
    )
    source_physics_step_ms = float(
        game.get("source_physics_step_ms")
        or pair.get("source_physics_step_ms")
        or runtime_source_physics
        or DEFAULT_SOURCE_PHYSICS_STEP_MS
    )
    if not math.isfinite(source_physics_step_ms) or source_physics_step_ms <= 0.0:
        raise ValueError("source_physics_step_ms must be positive and finite")

    runtime_frames = _consistent_runtime_value(policy_entries, "decision_source_frames")
    raw_frames = (
        game.get("decision_source_frames")
        or pair.get("decision_source_frames")
        or runtime_frames
    )
    if raw_frames is None:
        runtime_decision_ms = _consistent_runtime_value(policy_entries, "decision_ms")
        spec_decision_ms = float(game.get("decision_ms", pair["decision_ms"]))
        decision_ms = (
            float(runtime_decision_ms)
            if runtime_decision_ms is not None
            and math.isclose(
                spec_decision_ms,
                DEFAULT_DECISION_MS,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            else spec_decision_ms
        )
        ratio = decision_ms / source_physics_step_ms
        decision_source_frames = int(round(ratio))
        if decision_source_frames < 1 or not math.isclose(
            ratio,
            float(decision_source_frames),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "decision_ms must be a whole number of source physics frames; "
                "set decision_source_frames explicitly"
            )
    else:
        decision_source_frames = int(raw_frames)
        if decision_source_frames < 1:
            raise ValueError("decision_source_frames must be positive")
    decision_ms = float(decision_source_frames) * source_physics_step_ms
    return {
        "decision_ms": decision_ms,
        "decision_source_frames": int(decision_source_frames),
        "source_physics_step_ms": source_physics_step_ms,
        "source_max_ticks": int(max_steps) * int(decision_source_frames),
    }


def run_checkpoint_game(
    spec: Mapping[str, Any],
    *,
    checkpoint_mount: Path | None = None,
    artifact_mount: Path | None = None,
    mount: Path | None = None,
    remote_root: Path | None = None,
    preloaded_policy_entries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    import numpy as np

    from curvyzero.env import vector_runtime
    from curvyzero.env.vector_multiplayer_env import ACTION_COUNT, VectorMultiplayerEnv
    from curvyzero.env.vector_visual_observation import (
        SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        TRAIL_RENDER_MODE_BROWSER_LINES,
        TRAIL_RENDER_MODE_DEFAULT,
        render_source_state_rgb_canvas_like,
    )
    from curvyzero.training.curvytron_current_policy_selfplay_smoke import (
        SourceStateGray64Stack4,
        validate_stack_trail_render_mode,
    )

    if mount is not None:
        checkpoint_mount = checkpoint_mount or mount
        artifact_mount = artifact_mount or mount
    if artifact_mount is None:
        raise ValueError("run_checkpoint_game needs artifact_mount")
    if checkpoint_mount is None and preloaded_policy_entries is None:
        raise ValueError("run_checkpoint_game needs checkpoint_mount when policies are not preloaded")

    game = dict(spec)
    pair = normalize_pair_spec({**game, "games_per_pair": 1})
    game_id = _safe_id(str(game.get("game_id") or "game-000000"), label="game_id")
    tournament_id = pair["tournament_id"]
    battle_id = pair["battle_id"]
    seed = int(game.get("seed", pair["seed"]))
    max_steps = int(game.get("max_steps", pair["max_steps"]))
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    policy_mode = str(game.get("policy_mode", pair["policy_mode"]))
    if policy_mode not in POLICY_MODE_CHOICES:
        raise ValueError(f"policy_mode must be one of {POLICY_MODE_CHOICES!r}")
    frame_size = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    frame_stride = max(1, int(game.get("frame_stride", pair["frame_stride"])))
    capture_frames = bool(
        game.get("save_gif", pair["save_gif"])
        or game.get("save_frames_npz", pair["save_frames_npz"])
    )
    root_ref = game_root_ref(tournament_id, battle_id, game_id)
    root_path = runs.volume_path(artifact_mount, root_ref)
    summary_path = runs.volume_path(artifact_mount, game_summary_ref(tournament_id, battle_id, game_id))
    gif_path = runs.volume_path(artifact_mount, game_gif_ref(tournament_id, battle_id, game_id))
    frames_path = runs.volume_path(artifact_mount, game_frames_ref(tournament_id, battle_id, game_id))
    root_path.mkdir(parents=True, exist_ok=True)

    if preloaded_policy_entries is None:
        policy_entries = load_policy_entries_for_game(
            game,
            checkpoint_mount=checkpoint_mount,
            artifact_mount=artifact_mount,
            remote_root=remote_root,
        )
        preloaded = False
    else:
        policy_entries = [dict(entry) for entry in preloaded_policy_entries]
        preloaded = True
    if len(policy_entries) != 2:
        raise ValueError("run_checkpoint_game needs exactly two policy entries")

    policy_loads = []
    policies = []
    for load in policy_entries:
        if "policy" not in load:
            raise ValueError("policy entry missing policy")
        policies.append(load["policy"])
        policy_loads.append(
            {
                **{key: value for key, value in load.items() if key != "policy"},
                "preloaded": preloaded,
            }
        )
    default_policy_trail_render_mode = validate_stack_trail_render_mode(
        str(
            game.get("policy_trail_render_mode")
            or game.get("observation_trail_render_mode")
            or game.get("trail_render_mode")
            or pair.get("policy_trail_render_mode")
            or pair.get("trail_render_mode")
            or TRAIL_RENDER_MODE_DEFAULT
        )
    )
    policy_trail_render_modes = []
    for player, load in zip(pair["players"], policy_entries, strict=True):
        mode = validate_stack_trail_render_mode(
            str(
                player.get("policy_trail_render_mode")
                or player.get("observation_trail_render_mode")
                or load.get("policy_trail_render_mode")
                or default_policy_trail_render_mode
            )
        )
        policy_trail_render_modes.append(mode)
    gif_trail_render_mode = TRAIL_RENDER_MODE_BROWSER_LINES
    runtime_settings = _source_frame_runtime_settings(
        game,
        pair,
        policy_entries,
        max_steps=max_steps,
    )

    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        seed=seed,
        decision_ms=float(runtime_settings["decision_ms"]),
        decision_source_frames=int(runtime_settings["decision_source_frames"]),
        source_physics_step_ms=float(runtime_settings["source_physics_step_ms"]),
        max_ticks=int(runtime_settings["source_max_ticks"]),
        death_mode=vector_runtime.DEATH_MODE_NORMAL,
        natural_bonus_spawn=bool(game.get("natural_bonus_spawn", pair["natural_bonus_spawn"])),
    )
    visual_stacks = {
        mode: SourceStateGray64Stack4(
            batch_size=1,
            player_count=2,
            trail_render_mode=mode,
        )
        for mode in sorted(set(policy_trail_render_modes))
    }
    batch = env.reset(seed=seed)
    observations_by_mode = {
        mode: stack.update(env, copy=False)
        for mode, stack in visual_stacks.items()
    }
    frames = []
    if capture_frames:
        frames.append(
            render_source_state_rgb_canvas_like(
                env.state,
                row=0,
                frame_size=frame_size,
                trail_render_mode=gif_trail_render_mode,
            ).copy()
        )
    action_trace: list[dict[str, Any]] = []
    action_counts: dict[str, Counter[int]] = {
        "seat_0": Counter(),
        "seat_1": Counter(),
    }
    physical_steps = 0
    done = False
    truncated = False
    last_info: dict[str, Any] = _to_plain(batch.info)
    failure: dict[str, Any] | None = None

    while not done and physical_steps < max_steps:
        actions = np.zeros((1, 2), dtype=np.int16)
        step_policy: list[dict[str, Any]] = []
        try:
            for seat in (0, 1):
                observation = observations_by_mode[policy_trail_render_modes[seat]]
                obs = {
                    "observation": np.asarray(observation[0, seat], dtype=np.float32),
                    "action_mask": np.asarray(batch.action_mask[0, seat], dtype=np.float32),
                    "to_play": seat,
                }
                result = _policy_action(
                    policy=policies[seat],
                    observation=obs,
                    policy_mode=policy_mode,
                    collect_temperature=float(game.get("collect_temperature", pair["collect_temperature"])),
                    collect_epsilon=float(game.get("collect_epsilon", pair["collect_epsilon"])),
                )
                action = int(result["action"])
                if action < 0 or action >= ACTION_COUNT or not bool(obs["action_mask"][action]):
                    raise ValueError(f"seat {seat} produced illegal action {action}")
                actions[0, seat] = action
                action_counts[f"seat_{seat}"][action] += 1
                step_policy.append(
                    {
                        "seat": seat,
                        "action": action,
                        "policy_mode": policy_mode,
                        "compact_output": result.get("compact_output"),
                    }
                )
            batch = env.step(actions, timer_advance_ms=float(runtime_settings["decision_ms"]))
            physical_steps += 1
            done = bool(batch.done[0])
            truncated = bool(batch.truncated[0])
            last_info = _to_plain(batch.info)
            observations_by_mode = {
                mode: stack.update(env, copy=False)
                for mode, stack in visual_stacks.items()
            }
            if capture_frames and (physical_steps % frame_stride == 0 or done):
                frames.append(
                    render_source_state_rgb_canvas_like(
                        env.state,
                        row=0,
                        frame_size=frame_size,
                        trail_render_mode=gif_trail_render_mode,
                    ).copy()
                )
            if len(action_trace) < int(game.get("action_trace_limit", pair["action_trace_limit"])):
                action_trace.append(
                    {
                        "physical_step": physical_steps,
                        "joint_action": [int(actions[0, 0]), int(actions[0, 1])],
                        "done": done,
                        "truncated": truncated,
                        "policy": step_policy,
                    }
                )
        except Exception as exc:
            failure = exception_payload(exc)
            break

    score = score_from_info(
        last_info,
        done=done,
        truncated=truncated,
        physical_steps=physical_steps,
        max_steps=max_steps,
    )
    artifacts: dict[str, Any] = {}
    gif_ref = None
    if bool(game.get("save_gif", pair["save_gif"])):
        gif_info = _save_gif(frames, gif_path, fps=float(game.get("gif_fps", pair["gif_fps"])))
        gif_info["ref"] = runs.file_ref(gif_path, mount=artifact_mount)
        artifacts["gif"] = gif_info
        gif_ref = gif_info["ref"]
    frames_ref = None
    if bool(game.get("save_frames_npz", pair["save_frames_npz"])):
        frames_info = _write_frames_npz(
            frames,
            frames_path,
            metadata={
                "schema_id": "curvyzero_curvytron_tournament_rgb_frames/v0",
                "tournament_id": tournament_id,
                "battle_id": battle_id,
                "game_id": game_id,
                "seed": seed,
            },
        )
        frames_info["ref"] = runs.file_ref(frames_path, mount=artifact_mount)
        artifacts["frames_npz"] = frames_info
        frames_ref = frames_info["ref"]

    summary = {
        "schema_id": GAME_SCHEMA_ID,
        "ok": failure is None,
        "tournament_id": tournament_id,
        "battle_id": battle_id,
        "pair_index": pair["pair_index"],
        "game_id": game_id,
        "game_index": int(game.get("game_index", 0)),
        "seed": seed,
        "players": pair["players"],
        "policy_mode": policy_mode,
        "collect_temperature": float(game.get("collect_temperature", pair["collect_temperature"])),
        "collect_epsilon": float(game.get("collect_epsilon", pair["collect_epsilon"])),
        "decision_ms": float(runtime_settings["decision_ms"]),
        "decision_source_frames": int(runtime_settings["decision_source_frames"]),
        "source_physics_step_ms": float(runtime_settings["source_physics_step_ms"]),
        "source_max_ticks": int(runtime_settings["source_max_ticks"]),
        "score": score,
        "done": done,
        "truncated": truncated,
        "physical_steps": int(physical_steps),
        "max_steps": int(max_steps),
        "terminal_info": last_info,
        "action_counts": {
            seat: {str(action): int(count) for action, count in sorted(counts.items())}
            for seat, counts in action_counts.items()
        },
        "action_trace": action_trace,
        "frame_count": len(frames),
        "frame_size": frame_size,
        "frame_stride": frame_stride,
        "policy_trail_render_modes": {
            f"seat_{seat}": mode for seat, mode in enumerate(policy_trail_render_modes)
        },
        "policy_trail_render_mode": (
            policy_trail_render_modes[0]
            if policy_trail_render_modes[0] == policy_trail_render_modes[1]
            else "mixed"
        ),
        "gif_trail_render_mode": gif_trail_render_mode,
        "trail_render_mode": (
            policy_trail_render_modes[0]
            if policy_trail_render_modes[0] == policy_trail_render_modes[1]
            else "mixed"
        ),
        "render_contract": {
            "policy_observation": (
                "per-seat SourceStateGray64Stack4, using the checkpoint's "
                "policy_trail_render_mode when supplied"
            ),
            "gif": "full_704_rgb_canvas_like_browser_lines",
            "gif_frame_size": frame_size,
            "gif_trail_render_mode": gif_trail_render_mode,
        },
        "gif_ref": gif_ref,
        "frames_ref": frames_ref,
        "summary_ref": runs.file_ref(summary_path, mount=artifact_mount),
        "policy_loads": policy_loads,
        "artifacts": artifacts,
        "failure": failure,
    }
    runs.write_json(summary_path, _to_plain(summary))
    return summary


def failure_game_summary(
    spec: Mapping[str, Any],
    exc: BaseException,
    *,
    artifact_mount: Path | None = None,
    mount: Path | None = None,
) -> dict[str, Any]:
    if mount is not None:
        artifact_mount = artifact_mount or mount
    if artifact_mount is None:
        raise ValueError("failure_game_summary needs artifact_mount")
    game = dict(spec)
    pair = normalize_pair_spec({**game, "games_per_pair": 1})
    game_id = _safe_id(str(game.get("game_id") or "game-000000"), label="game_id")
    summary_ref = game_summary_ref(pair["tournament_id"], pair["battle_id"], game_id)
    summary_path = runs.volume_path(artifact_mount, summary_ref)
    summary = {
        "schema_id": GAME_SCHEMA_ID,
        "ok": False,
        "tournament_id": pair["tournament_id"],
        "battle_id": pair["battle_id"],
        "game_id": game_id,
        "game_index": int(game.get("game_index", 0)),
        "players": pair["players"],
        "summary_ref": summary_ref.as_posix(),
        **exception_payload(exc),
    }
    runs.write_json(summary_path, _to_plain(summary))
    return summary
