"""CurvyTron checkpoint tournament helpers.

The tournament lane is intentionally separate from training. A game is one
checkpoint in seat 0 against one checkpoint in seat 1. The score is simple:
the first dead player loses; simultaneous death or timeout is a draw.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from curvyzero.infra.modal import run_management as runs


TOURNAMENT_TASK_ID = "curvytron-checkpoint-tournament"
TOURNAMENT_BASE_REF = PurePosixPath("tournaments") / "curvytron"
TOURNAMENT_RUN_MARKER_FILENAME = "show_in_tournament_browser.flag"

TOURNAMENT_SCHEMA_ID = "curvyzero_curvytron_checkpoint_tournament/v0"
BATTLE_SCHEMA_ID = "curvyzero_curvytron_checkpoint_tournament_battle/v0"
GAME_SCHEMA_ID = "curvyzero_curvytron_checkpoint_tournament_game/v0"

POLICY_MODE_EVAL = "eval"
POLICY_MODE_COLLECT = "collect"
POLICY_MODE_CHOICES = (POLICY_MODE_EVAL, POLICY_MODE_COLLECT)

DEFAULT_GAMES_PER_PAIR = 10
DEFAULT_MAX_STEPS = 512
DEFAULT_DECISION_MS = 300.0
DEFAULT_NUM_SIMULATIONS = 8
DEFAULT_POLICY_BATCH_SIZE = 8
DEFAULT_COLLECT_TEMPERATURE = 1.0
DEFAULT_COLLECT_EPSILON = 0.25
DEFAULT_FRAME_STRIDE = 1
DEFAULT_GIF_FPS = 8.0
DEFAULT_FRAME_SIZE = 704
DEFAULT_SAVE_GIF = True
DEFAULT_SAVE_FRAMES_NPZ = False
DEFAULT_ORDERED_PAIRS = False
DEFAULT_INCLUDE_SELF_PAIRS = False

ALLOWED_TOURNAMENT_ARTIFACT_FILENAMES = frozenset(
    {
        "game.gif",
        "frames.npz",
        "summary.json",
        "battle.json",
        "tournament.json",
        "standings.json",
        "complete.json",
    }
)


class TournamentRefError(ValueError):
    """Raised when a tournament Volume ref is not safe."""


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
        }
    ref_value = raw.get("checkpoint_ref") or raw.get("ref")
    if not isinstance(ref_value, str):
        raise ValueError("checkpoint spec needs checkpoint_ref")
    ref = runs.require_relative_ref(ref_value).as_posix()
    checkpoint_id = str(raw.get("checkpoint_id") or raw.get("id") or checkpoint_id_from_ref(ref, index=index))
    label = str(raw.get("label") or checkpoint_id)
    return {
        "checkpoint_id": _safe_id(checkpoint_id, label="checkpoint_id"),
        "label": label,
        "checkpoint_ref": ref,
        "checkpoint_state_key": raw.get("checkpoint_state_key"),
        "model_env_variant": raw.get("model_env_variant"),
        "model_reward_variant": raw.get("model_reward_variant"),
    }


def normalize_checkpoint_specs(checkpoints: Sequence[str | Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        normalize_checkpoint_spec(checkpoint, index=index)
        for index, checkpoint in enumerate(checkpoints)
    ]


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


def battle_root_ref(tournament_id: str, battle_id: str) -> PurePosixPath:
    return tournament_root_ref(tournament_id) / "battles" / _safe_id(battle_id, label="battle_id")


def battle_summary_ref(tournament_id: str, battle_id: str) -> PurePosixPath:
    return battle_root_ref(tournament_id, battle_id) / "battle.json"


def game_root_ref(tournament_id: str, battle_id: str, game_id: str) -> PurePosixPath:
    return battle_root_ref(tournament_id, battle_id) / "games" / _safe_id(game_id, label="game_id")


def game_summary_ref(tournament_id: str, battle_id: str, game_id: str) -> PurePosixPath:
    return game_root_ref(tournament_id, battle_id, game_id) / "summary.json"


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
    pair_specs: list[dict[str, Any]] = []
    pair_index = 0
    for i, player_a in enumerate(players):
        for j, player_b in enumerate(players):
            if not include_self_pairs and i == j:
                continue
            if not ordered_pairs and j <= i:
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
                        "games_per_pair": int(games_per_pair),
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
    return {
        "schema_id": BATTLE_SCHEMA_ID,
        "tournament_id": tournament_id,
        "battle_id": battle_id,
        "pair_index": int(raw.get("pair_index") or 0),
        "players": normalized_players,
        "games_per_pair": int(raw.get("games_per_pair", DEFAULT_GAMES_PER_PAIR)),
        "seed": int(raw.get("seed", 0)),
        "max_steps": int(raw.get("max_steps", DEFAULT_MAX_STEPS)),
        "decision_ms": float(raw.get("decision_ms", DEFAULT_DECISION_MS)),
        "num_simulations": int(raw.get("num_simulations", DEFAULT_NUM_SIMULATIONS)),
        "policy_batch_size": int(raw.get("policy_batch_size", DEFAULT_POLICY_BATCH_SIZE)),
        "policy_mode": policy_mode,
        "collect_temperature": float(raw.get("collect_temperature", DEFAULT_COLLECT_TEMPERATURE)),
        "collect_epsilon": float(raw.get("collect_epsilon", DEFAULT_COLLECT_EPSILON)),
        "natural_bonus_spawn": bool(raw.get("natural_bonus_spawn", True)),
        "trail_render_mode": raw.get("trail_render_mode"),
        "frame_stride": int(raw.get("frame_stride", DEFAULT_FRAME_STRIDE)),
        "frame_size": int(raw.get("frame_size", DEFAULT_FRAME_SIZE)),
        "gif_fps": float(raw.get("gif_fps", DEFAULT_GIF_FPS)),
        "save_gif": bool(raw.get("save_gif", DEFAULT_SAVE_GIF)),
        "save_frames_npz": bool(raw.get("save_frames_npz", DEFAULT_SAVE_FRAMES_NPZ)),
        "action_trace_limit": int(raw.get("action_trace_limit", 128)),
    }


def build_game_specs_for_pair(pair_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    pair = normalize_pair_spec(pair_spec)
    count = int(pair["games_per_pair"])
    if count < 1:
        raise ValueError("games_per_pair must be at least 1")
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
                "num_simulations": pair["num_simulations"],
                "policy_batch_size": pair["policy_batch_size"],
                "policy_mode": pair["policy_mode"],
                "collect_temperature": pair["collect_temperature"],
                "collect_epsilon": pair["collect_epsilon"],
                "natural_bonus_spawn": pair["natural_bonus_spawn"],
                "trail_render_mode": pair["trail_render_mode"],
                "frame_stride": pair["frame_stride"],
                "frame_size": pair["frame_size"],
                "gif_fps": pair["gif_fps"],
                "save_gif": pair["save_gif"],
                "save_frames_npz": pair["save_frames_npz"],
                "action_trace_limit": pair["action_trace_limit"],
            }
        )
    return specs


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


def summarize_pair_results(pair_spec: Mapping[str, Any], game_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pair = normalize_pair_spec(pair_spec)
    tally = tally_game_results(game_results)
    first_gif_ref = None
    for result in game_results:
        if result.get("gif_ref"):
            first_gif_ref = result["gif_ref"]
            break
    return {
        "schema_id": BATTLE_SCHEMA_ID,
        "ok": tally["failure_count"] == 0,
        "tournament_id": pair["tournament_id"],
        "battle_id": pair["battle_id"],
        "pair_index": pair["pair_index"],
        "players": pair["players"],
        "settings": {
            key: pair[key]
            for key in (
                "games_per_pair",
                "max_steps",
                "decision_ms",
                "num_simulations",
                "policy_mode",
                "collect_temperature",
                "collect_epsilon",
                "natural_bonus_spawn",
                "trail_render_mode",
                "frame_stride",
                "frame_size",
                "save_gif",
            )
        },
        "tally": tally,
        "first_gif_ref": first_gif_ref,
        "game_summary_refs": [
            result.get("summary_ref") for result in game_results if result.get("summary_ref")
        ],
        "games": [_compact_game_result(result) for result in game_results],
    }


def _compact_game_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(result.get("ok")),
        "game_id": result.get("game_id"),
        "game_index": result.get("game_index"),
        "seed": result.get("seed"),
        "score": result.get("score"),
        "physical_steps": result.get("physical_steps"),
        "gif_ref": result.get("gif_ref"),
        "summary_ref": result.get("summary_ref"),
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
        model_env_variant=model_env_variant,
        model_reward_variant=model_reward_variant,
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
        "surface": surface,
    }


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


def run_checkpoint_game(
    spec: Mapping[str, Any],
    *,
    mount: Path,
    remote_root: Path | None = None,
) -> dict[str, Any]:
    import numpy as np

    from curvyzero.env import vector_runtime
    from curvyzero.env.vector_multiplayer_env import ACTION_COUNT, VectorMultiplayerEnv
    from curvyzero.env.vector_visual_observation import (
        SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        TRAIL_RENDER_MODE_DEFAULT,
        render_source_state_rgb_canvas_like,
    )
    from curvyzero.training.curvytron_current_policy_selfplay_smoke import (
        SourceStateGray64Stack4,
        validate_stack_trail_render_mode,
    )

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
    frame_size = int(game.get("frame_size", pair["frame_size"]) or DEFAULT_FRAME_SIZE)
    if frame_size < 64:
        frame_size = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    frame_stride = max(1, int(game.get("frame_stride", pair["frame_stride"])))
    trail_render_mode = validate_stack_trail_render_mode(
        str(game.get("trail_render_mode") or pair.get("trail_render_mode") or TRAIL_RENDER_MODE_DEFAULT)
    )
    root_ref = game_root_ref(tournament_id, battle_id, game_id)
    root_path = runs.volume_path(mount, root_ref)
    summary_path = runs.volume_path(mount, game_summary_ref(tournament_id, battle_id, game_id))
    gif_path = runs.volume_path(mount, game_gif_ref(tournament_id, battle_id, game_id))
    frames_path = runs.volume_path(mount, game_frames_ref(tournament_id, battle_id, game_id))
    root_path.mkdir(parents=True, exist_ok=True)

    policy_loads = []
    policies = []
    for player in pair["players"]:
        load = _load_policy_from_checkpoint(
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
            mount=mount,
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
        policies.append(load["policy"])
        policy_loads.append(
            {
                key: value
                for key, value in load.items()
                if key != "policy"
            }
        )

    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        seed=seed,
        decision_ms=float(game.get("decision_ms", pair["decision_ms"])),
        max_ticks=max_steps,
        death_mode=vector_runtime.DEATH_MODE_NORMAL,
        natural_bonus_spawn=bool(game.get("natural_bonus_spawn", pair["natural_bonus_spawn"])),
    )
    visual_stack = SourceStateGray64Stack4(
        batch_size=1,
        player_count=2,
        trail_render_mode=trail_render_mode,
    )
    batch = env.reset(seed=seed)
    observation = visual_stack.update(env, copy=False)
    frames = [
        render_source_state_rgb_canvas_like(
            env.state,
            row=0,
            frame_size=frame_size,
            trail_render_mode=trail_render_mode,
        ).copy()
    ]
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
            batch = env.step(actions, timer_advance_ms=float(game.get("decision_ms", pair["decision_ms"])))
            physical_steps += 1
            done = bool(batch.done[0])
            truncated = bool(batch.truncated[0])
            last_info = _to_plain(batch.info)
            observation = visual_stack.update(env, copy=False)
            if physical_steps % frame_stride == 0 or done:
                frames.append(
                    render_source_state_rgb_canvas_like(
                        env.state,
                        row=0,
                        frame_size=frame_size,
                        trail_render_mode=trail_render_mode,
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
        gif_info["ref"] = runs.file_ref(gif_path, mount=mount)
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
        frames_info["ref"] = runs.file_ref(frames_path, mount=mount)
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
        "trail_render_mode": trail_render_mode,
        "gif_ref": gif_ref,
        "frames_ref": frames_ref,
        "summary_ref": runs.file_ref(summary_path, mount=mount),
        "policy_loads": policy_loads,
        "artifacts": artifacts,
        "failure": failure,
    }
    runs.write_json(summary_path, _to_plain(summary))
    return summary


def failure_game_summary(spec: Mapping[str, Any], exc: BaseException, *, mount: Path) -> dict[str, Any]:
    game = dict(spec)
    pair = normalize_pair_spec({**game, "games_per_pair": 1})
    game_id = _safe_id(str(game.get("game_id") or "game-000000"), label="game_id")
    summary_ref = game_summary_ref(pair["tournament_id"], pair["battle_id"], game_id)
    summary_path = runs.volume_path(mount, summary_ref)
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
