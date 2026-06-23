"""Modal app for CurvyTron checkpoint tournaments.

One app owns the whole tournament lane. The lowest-level function runs one game.
Higher functions fan out over games and checkpoint pairs.

Checkpoint files are read from the training Volume. Tournament summaries and
GIFs are written to a separate tournament Volume.
"""

from __future__ import annotations

import json
import copy
import hashlib
import re
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.observability.feedback_loop_lineage import (
    append_lineage_event,
    lineage_events_path,
)
from curvyzero.infra.modal.curvyzero_checkpoint_tournament_browser_render import (
    _battle_href as _battle_href,
    _battle_opponent_for_checkpoint as _battle_opponent_for_checkpoint,
    _battle_player_for_checkpoint as _battle_player_for_checkpoint,
    _checkpoint_battle_sort_key as _checkpoint_battle_sort_key,
    _friendly_progress_label as _friendly_progress_label,
    _href as _href,
    _page_href as _page_href,
    _rating_rank_by_checkpoint as _rating_rank_by_checkpoint,
    _rating_row_by_checkpoint as _rating_row_by_checkpoint,
    _rating_rows as _rating_rows,
    _render_battle_detail_section as _render_battle_detail_section,
    _render_battle_page as _render_battle_page,
    _render_page as _render_page,
    _review_battle_row as _review_battle_row,
    _short_battle_label as _short_battle_label,
    _sort_checkpoint_battle_rows as _sort_checkpoint_battle_rows,
    _wins_for_checkpoint as _wins_for_checkpoint,
)
from curvyzero.infra.modal.curvyzero_checkpoint_tournament_runtime import (
    LOW_LEVEL_WORKER_RETRIES,
    app,
    checkpoint_intake_queue,
    checkpoint_intake_state,
    checkpoint_volume,
    checkpoint_volumes as _checkpoint_volumes,
    control_volume,
    controller_volumes as _controller_volumes,
    game_volumes as _game_volumes,
    image,
    opponent_leaderboard_state,
    tournament_volume,
    tournament_volumes as _tournament_volumes,
)
from curvyzero.infra.modal.curvyzero_checkpoint_tournament_settings import (
    APP_NAME,
    CHECKPOINT_INTAKE_ACTIVE_KEYS,
    CHECKPOINT_INTAKE_DICT_NAME,
    CHECKPOINT_INTAKE_QUEUE_NAME,
    CHECKPOINT_VOLUME_NAME,
    CONTROL_MOUNT,
    CONTROL_VOLUME_NAME,
    CURRENT_RATING_RUN_ID,
    CURRENT_TOURNAMENT_ID,
    DEFAULT_BATTLE_GAME_LIMIT,
    DEFAULT_CHECKPOINT_INTAKE_CLAIM_STALE_SECONDS,
    DEFAULT_CHECKPOINT_INTAKE_SCAN_SECONDS,
    DEFAULT_CHECKPOINT_INTAKE_QUEUE_TTL_SECONDS,
    DEFAULT_LIMIT,
    DEFAULT_PROVISIONAL_RATING_INTERVAL_SECONDS,
    DEFAULT_PROVISIONAL_RATING_MAX_SECONDS,
    DEFAULT_RATING_ROUND_PARTIAL_REDUCE_AFTER_SECONDS,
    DEFAULT_RATING_ROUND_PARTIAL_REDUCE_MIN_COMPLETED_GAMES,
    DEFAULT_RATING_ROUND_PROGRESS_FULL_SCAN_GAME_LIMIT,
    DEFAULT_RATING_ROUND_STALE_SECONDS,
    DYNAMIC_HEADERS,
    GIF_BROWSER_APP_NAME,
    GIF_CACHE_MAX_AGE_SECONDS,
    MAX_LIMIT,
    OPPONENT_LEADERBOARD_DICT_NAME,
    REMOTE_ROOT,
    RUNS_MOUNT,
    TOURNAMENT_MOUNT,
    TOURNAMENT_VOLUME_NAME,
    TOURNAMENT_GAME_SHARD_WORKER_BUFFER_CONTAINERS,
    TOURNAMENT_GAME_SHARD_WORKER_MIN_CONTAINERS,
    TOURNAMENT_GAME_SHARD_WORKER_SCALEDOWN_WINDOW_SECONDS,
    TOURNAMENT_GAME_WORKER_BUFFER_CONTAINERS,
    TOURNAMENT_GAME_WORKER_MIN_CONTAINERS,
    TOURNAMENT_GAME_WORKER_SCALEDOWN_WINDOW_SECONDS,
    TRAIN_APP_NAME,
    TRAINING_CANDIDATE_ACTIVE_MIN_DISTINCT_OPPONENTS,
    TRAINING_CANDIDATE_ACTIVE_MIN_VALID_GAMES,
    TRAINING_CANDIDATE_ASSIGNMENT_BANK_ATTEMPT_ID,
    TRAINING_CANDIDATE_ASSIGNMENT_BANK_RUN_ID,
    TRAINING_CANDIDATE_ASSIGNMENT_SEED,
    TRAINING_CANDIDATE_REFRESH_CONFIG_REF,
    TRAINING_CANDIDATE_MAX_ACTIVE_RANK,
    TRAINING_CANDIDATE_MIN_ACTIVE_COUNT,
    TRAINING_CANDIDATE_REFRESH_POINTERS,
    TRAINING_CANDIDATE_REFRESH_SECONDS,
    TRAINING_TASK_ID,
    WEB_BATTLE_DETAIL_CACHE_TTL_SECONDS,
    WEB_BATTLE_DETAIL_CACHE_VERSION,
    WEB_GIF_BYTES_CACHE_MAX_ITEM_BYTES,
    WEB_GIF_BYTES_CACHE_TTL_SECONDS,
    WEB_PAGE_RELOAD_MIN_INTERVAL_SECONDS,
    WEB_PROGRESS_CACHE_TTL_SECONDS,
    WEB_PROGRESS_RELOAD_MIN_INTERVAL_SECONDS,
    WEB_PROVISIONAL_RATING_CACHE_TTL_SECONDS,
)
from curvyzero.training.opponent_leaderboard import (
    DEFAULT_ACTIVE_MIN_DISTINCT_OPPONENTS,
    DEFAULT_ACTIVE_MIN_VALID_GAMES,
    DEFAULT_MAX_FAILURE_RATE,
    build_leaderboard_pointer,
    build_leaderboard_snapshot_from_rating_snapshot,
    canonical_assignment_json_sha256,
    validate_leaderboard_snapshot,
    validate_assignment_audit,
    validate_rating_snapshot_source,
)
from curvyzero.training.opponent_mixture import (
    OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
    OPPONENT_RUNTIME_MODE_NORMAL,
)
from curvyzero.training.opponent_registry import parse_opponent_assignment_snapshot
from curvyzero.tournament import checkpoint_intake_service as intake_service
from curvyzero.tournament import curvytron_checkpoint_tournament as arena


_LAST_WEB_VOLUME_RELOAD_TS = 0.0
_WEB_CACHE: dict[str, tuple[float, Any]] = {}


def _commit_volume(volume: Any = tournament_volume) -> str | None:
    if not hasattr(volume, "commit"):
        return None
    try:
        volume.commit()
    except Exception as exc:  # pragma: no cover - remote Volume resilience.
        return f"{type(exc).__name__}: {exc}"
    return None


def _reload_volume(volume: Any, *, force: bool = False) -> str | None:
    if volume is None or not hasattr(volume, "reload"):
        return None
    try:
        volume.reload()
    except Exception as exc:  # pragma: no cover - remote Volume resilience.
        return f"{type(exc).__name__}: {exc}"
    return None


def _web_reload_volume(
    volume: Any,
    *,
    force: bool = False,
    min_interval_sec: float = WEB_PROGRESS_RELOAD_MIN_INTERVAL_SECONDS,
) -> str | None:
    """Refresh the web container's Volume view without reloading on every poll."""

    global _LAST_WEB_VOLUME_RELOAD_TS
    now = time.monotonic()
    if not force and now - _LAST_WEB_VOLUME_RELOAD_TS < float(min_interval_sec):
        return None
    error = _reload_volume(volume, force=force)
    _LAST_WEB_VOLUME_RELOAD_TS = now
    if error is None:
        _WEB_CACHE.clear()
    return error


def _web_cache_get(key: str, *, ttl_seconds: float) -> Any | None:
    item = _WEB_CACHE.get(key)
    if item is None:
        return None
    created_at, value = item
    if time.monotonic() - created_at > float(ttl_seconds):
        _WEB_CACHE.pop(key, None)
        return None
    return value


def _web_cache_set(key: str, value: Any) -> Any:
    _WEB_CACHE[key] = (time.monotonic(), value)
    return value


def _read_cached_file_bytes(
    path: Path,
    *,
    cache_prefix: str,
    ttl_seconds: float,
    max_item_bytes: int,
) -> bytes:
    stat = path.stat()
    key = f"{cache_prefix}:{path.as_posix()}:{stat.st_mtime_ns}:{stat.st_size}"
    if stat.st_size <= int(max_item_bytes):
        cached = _web_cache_get(key, ttl_seconds=ttl_seconds)
        if isinstance(cached, bytes):
            return cached
    data = path.read_bytes()
    if len(data) <= int(max_item_bytes):
        _web_cache_set(key, data)
    return data


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _checkpoint_iteration_from_path(path: Path) -> int | None:
    name = path.name
    prefix = "iteration_"
    suffix = ".pth.tar"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    try:
        return int(name[len(prefix) : -len(suffix)])
    except ValueError:
        return None


def _run_ids_from_prefix(mount: Path, *, run_id_prefix: str, max_runs: int = 0) -> list[str]:
    if not run_id_prefix:
        return []
    base = runs.volume_path(mount, Path("training") / TRAINING_TASK_ID)
    if not base.exists():
        return []
    rows = [
        path.name
        for path in base.iterdir()
        if path.is_dir() and path.name.startswith(run_id_prefix)
    ]
    rows.sort()
    if max_runs > 0:
        return rows[:max_runs]
    return rows


def _sort_discovery_rows_by_latest_checkpoint(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        [dict(row) for row in rows],
        key=lambda row: (
            -int(row.get("checkpoint_mtime_ns") or 0),
            -int(row.get("iteration") or -1),
            str(row.get("run_id") or ""),
        ),
    )


def _attempt_roots_for_run(run_root: Path) -> list[Path]:
    latest = _read_json(run_root / "latest_attempt.json")
    latest_attempt_id = latest.get("attempt_id")
    roots: list[Path] = []
    if isinstance(latest_attempt_id, str):
        latest_root = run_root / "attempts" / latest_attempt_id
        if latest_root.exists():
            roots.append(latest_root)
    attempts_root = run_root / "attempts"
    if attempts_root.exists():
        for path in sorted(attempts_root.iterdir(), key=lambda item: item.name):
            if path.is_dir() and path not in roots:
                roots.append(path)
    return roots


def _checkpoint_candidate_rows_for_run(
    mount: Path,
    *,
    run_id: str,
    checkpoint_iteration: int | None = None,
) -> list[dict[str, Any]]:
    run_root = runs.volume_path(mount, runs.run_root_ref(TRAINING_TASK_ID, run_id))
    candidates: list[dict[str, Any]] = []
    if not run_root.exists():
        return candidates
    for attempt_root in _attempt_roots_for_run(run_root):
        train_root = attempt_root / "train"
        for ckpt_dir in sorted(train_root.glob(arena.CHECKPOINT_EXP_CKPT_DIR_GLOB)):
            for checkpoint_path in sorted(ckpt_dir.glob(arena.CHECKPOINT_WEIGHT_FILENAME_GLOB)):
                if not checkpoint_path.is_file():
                    continue
                stat = checkpoint_path.stat()
                if stat.st_size <= 0:
                    continue
                iteration = _checkpoint_iteration_from_path(checkpoint_path)
                if iteration is None:
                    continue
                if checkpoint_iteration is not None and iteration != checkpoint_iteration:
                    continue
                checkpoint_ref = runs.file_ref(checkpoint_path, mount=mount)
                metadata_row = _checkpoint_discovery_row_from_ref(
                    checkpoint_ref,
                    mount=mount,
                    found=True,
                )
                candidates.append(
                    {
                        **metadata_row,
                        "run_id": run_id,
                        "found": True,
                        "attempt_id": attempt_root.name,
                        "exp_dir_name": ckpt_dir.parent.name,
                        "checkpoint_name": checkpoint_path.name,
                        "iteration": int(iteration),
                        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
                        "checkpoint_size_bytes": int(stat.st_size),
                        "checkpoint_ref": checkpoint_ref,
                        "checkpoint_path": str(checkpoint_path),
                    }
                )
    return candidates


def _discover_checkpoint_refs(
    mount: Path,
    *,
    run_ids: Sequence[str] | None = None,
    run_id_prefix: str = "",
    max_runs: int = 0,
    checkpoint_iteration: int | None = None,
    checkpoint_selection: str = arena.CHECKPOINT_SELECTION_LATEST,
) -> dict[str, Any]:
    checkpoint_selection = str(checkpoint_selection or arena.CHECKPOINT_SELECTION_LATEST)
    if checkpoint_selection not in arena.CHECKPOINT_SELECTION_CHOICES:
        raise ValueError(
            f"checkpoint_selection must be one of {arena.CHECKPOINT_SELECTION_CHOICES!r}"
        )
    if (
        checkpoint_selection == arena.CHECKPOINT_SELECTION_ITERATION
        and checkpoint_iteration is None
    ):
        raise ValueError("checkpoint_selection=iteration requires checkpoint_iteration")
    ids = [
        runs.clean_id(str(run_id), label="run_id")
        for run_id in (run_ids or [])
        if str(run_id).strip()
    ]
    if not ids and run_id_prefix:
        ids = _run_ids_from_prefix(
            mount,
            run_id_prefix=runs.clean_id(run_id_prefix, label="run_id_prefix"),
            max_runs=0,
        )
    candidates_by_run: dict[str, list[dict[str, Any]]] = {}
    for run_id in ids:
        candidates_by_run[run_id] = _checkpoint_candidate_rows_for_run(
            mount,
            run_id=run_id,
            checkpoint_iteration=checkpoint_iteration,
        )

    missing_rows = [
        {
            "run_id": run_id,
            "found": False,
            "iteration": checkpoint_iteration,
            "checkpoint_ref": None,
            "reason": "no_matching_iteration_pth_tar",
        }
        for run_id in ids
        if not candidates_by_run.get(run_id)
    ]
    latest_rows = []
    for run_id, candidates in candidates_by_run.items():
        if not candidates:
            continue
        latest_rows.append(
            max(
                candidates,
                key=lambda row: (
                    int(row.get("iteration") or -1),
                    int(row.get("checkpoint_mtime_ns") or 0),
                    str(row.get("checkpoint_ref") or ""),
                ),
            )
        )

    selected_run_ids = list(ids)
    selection = "all_requested"
    if run_id_prefix and max_runs > 0:
        found_rows = _sort_discovery_rows_by_latest_checkpoint(latest_rows)
        sorted_missing = sorted(
            missing_rows,
            key=lambda row: str(row.get("run_id") or ""),
        )
        selected_seed_rows = [*found_rows[:max_runs]]
        if len(selected_seed_rows) < max_runs:
            selected_seed_rows.extend(sorted_missing[: max_runs - len(selected_seed_rows)])
        selected_run_ids = [str(row.get("run_id")) for row in selected_seed_rows]
        selection = "latest_checkpoint_mtime"

    rows: list[dict[str, Any]] = []
    for run_id in selected_run_ids:
        candidates = candidates_by_run.get(run_id, [])
        if not candidates:
            rows.append(
                {
                    "run_id": run_id,
                    "found": False,
                    "iteration": checkpoint_iteration,
                    "checkpoint_ref": None,
                    "reason": "no_matching_iteration_pth_tar",
                }
            )
            continue
        if checkpoint_selection == arena.CHECKPOINT_SELECTION_ALL:
            rows.extend(
                sorted(
                    candidates,
                    key=lambda row: (
                        int(row.get("iteration") or -1),
                        str(row.get("attempt_id") or ""),
                        str(row.get("exp_dir_name") or ""),
                        str(row.get("checkpoint_ref") or ""),
                    ),
                )
            )
            continue
        rows.append(
            max(
                candidates,
                key=lambda row: (
                    int(row.get("iteration") or -1),
                    int(row.get("checkpoint_mtime_ns") or 0),
                    str(row.get("checkpoint_ref") or ""),
                ),
            )
        )
    found_rows = [row for row in rows if row.get("found")]
    missing_rows = [row for row in rows if not row.get("found")]
    return {
        "schema_id": "curvyzero_curvytron_checkpoint_discovery/v0",
        "checkpoint_volume_name": CHECKPOINT_VOLUME_NAME,
        "run_id_prefix": run_id_prefix,
        "checkpoint_selection": checkpoint_selection,
        "checkpoint_scan_glob": arena.CHECKPOINT_SCAN_GLOB,
        "selection": selection,
        "requested_run_count": len(ids),
        "selected_run_count": len(selected_run_ids),
        "found_count": len(found_rows),
        "found_checkpoint_count": len(found_rows),
        "found_run_count": len({str(row["run_id"]) for row in found_rows}),
        "missing_count": len(missing_rows),
        "missing_run_count": len(missing_rows),
        "checkpoint_iteration": checkpoint_iteration,
        "rows": rows,
        "checkpoint_refs": [
            str(row["checkpoint_ref"]) for row in rows if row.get("checkpoint_ref")
        ],
    }


def _discover_latest_checkpoint_refs(
    mount: Path,
    *,
    run_ids: Sequence[str] | None = None,
    run_id_prefix: str = "",
    max_runs: int = 0,
    checkpoint_iteration: int | None = None,
) -> dict[str, Any]:
    return _discover_checkpoint_refs(
        mount,
        run_ids=run_ids,
        run_id_prefix=run_id_prefix,
        max_runs=max_runs,
        checkpoint_iteration=checkpoint_iteration,
        checkpoint_selection=arena.CHECKPOINT_SELECTION_LATEST,
    )


def _intake_manifest_key(tournament_id: str, rating_run_id: str) -> str:
    return (
        "manifest:"
        f"{runs.clean_id(tournament_id, label='tournament_id')}:"
        f"{runs.clean_id(rating_run_id, label='rating_run_id')}"
    )


def _intake_queue_partition(tournament_id: str, rating_run_id: str) -> str:
    clean_tournament_id = runs.clean_id(tournament_id, label="tournament_id")
    clean_rating_run_id = runs.clean_id(rating_run_id, label="rating_run_id")
    digest = arena._short_hash(f"{clean_tournament_id}:{clean_rating_run_id}", length=10)
    return (
        f"q:{arena._slug(clean_tournament_id, max_len=24)}:"
        f"{arena._slug(clean_rating_run_id, max_len=16)}:{digest}"
    )


def _leaderboard_root_ref(leaderboard_id: str) -> PurePosixPath:
    return (
        arena.TOURNAMENT_BASE_REF
        / "leaderboards"
        / runs.clean_id(leaderboard_id, label="leaderboard_id")
    )


def _leaderboard_snapshot_ref(leaderboard_id: str, snapshot_id: str) -> PurePosixPath:
    clean_snapshot_id = runs.clean_id(snapshot_id, label="snapshot_id")
    return _leaderboard_root_ref(leaderboard_id) / "snapshots" / f"{clean_snapshot_id}.json"


def _leaderboard_latest_ref(leaderboard_id: str) -> PurePosixPath:
    return _leaderboard_root_ref(leaderboard_id) / arena.ARTIFACT_LATEST_FILENAME


def _leaderboard_pointer_key(leaderboard_id: str) -> str:
    return f"current:{runs.clean_id(leaderboard_id, label='leaderboard_id')}"


def _tournament_rating_lineage_path(
    tournament_id: str,
    rating_run_id: str,
    *,
    mount: Path | None = None,
) -> Path:
    root = TOURNAMENT_MOUNT if mount is None else mount
    return lineage_events_path(
        runs.volume_path(
            root,
            arena.rating_root_ref(tournament_id, rating_run_id),
        )
    )


def _append_tournament_lineage_event(
    *,
    stage: str,
    tournament_id: str,
    rating_run_id: str,
    status: str = "ok",
    mount: Path | None = None,
    **fields: Any,
) -> dict[str, Any]:
    return append_lineage_event(
        _tournament_rating_lineage_path(tournament_id, rating_run_id, mount=mount),
        stage=stage,
        status=status,
        tournament_id=runs.clean_id(tournament_id, label="tournament_id"),
        rating_run_id=runs.clean_id(rating_run_id, label="rating_run_id"),
        **fields,
    )


def _repaired_leaderboard_pointer_from_latest(
    mount: Path,
    *,
    leaderboard_id: str,
) -> tuple[dict[str, Any], PurePosixPath, PurePosixPath]:
    clean_leaderboard_id = runs.clean_id(leaderboard_id, label="leaderboard_id")
    latest_ref = _leaderboard_latest_ref(clean_leaderboard_id)
    latest_path = runs.volume_path(mount, latest_ref)
    if not latest_path.is_file():
        raise ValueError(f"leaderboard latest snapshot not found: {latest_ref.as_posix()}")
    latest = validate_leaderboard_snapshot(_read_json(latest_path))
    if latest["leaderboard_id"] != clean_leaderboard_id:
        raise ValueError(
            "leaderboard latest snapshot id mismatch: "
            f"expected {clean_leaderboard_id!r}, got {latest['leaderboard_id']!r}"
        )

    snapshot_ref = _leaderboard_snapshot_ref(clean_leaderboard_id, latest["snapshot_id"])
    snapshot_path = runs.volume_path(mount, snapshot_ref)
    if not snapshot_path.is_file():
        raise ValueError(f"leaderboard immutable snapshot not found: {snapshot_ref.as_posix()}")
    immutable = validate_leaderboard_snapshot(_read_json(snapshot_path))
    for key in ("leaderboard_id", "snapshot_id", "snapshot_sha256"):
        if immutable.get(key) != latest.get(key):
            raise ValueError(f"leaderboard immutable snapshot {key} mismatch")

    pointer = build_leaderboard_pointer(
        immutable,
        snapshot_ref=snapshot_ref.as_posix(),
        published_at=runs.utc_timestamp(),
        writer={
            "kind": "curvytron_opponent_leaderboard_pointer_repair",
            "app_name": APP_NAME,
            "latest_ref": latest_ref.as_posix(),
        },
    )
    return pointer, latest_ref, snapshot_ref


def _clean_ref_set(refs: Any) -> set[str]:
    if isinstance(refs, str):
        values = refs.replace("\n", ",").split(",")
    elif isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)):
        values = refs
    else:
        values = []
    return {str(ref).strip() for ref in values if str(ref).strip()}


def _checkpoint_ref_pool_hash(refs: Any) -> str:
    cleaned_refs = sorted(_clean_ref_set(refs))
    encoded = json.dumps(cleaned_refs, separators=(",", ":"), sort_keys=True)
    return arena._short_hash(encoded, length=16)


def _parse_run_ids(run_ids: Any) -> list[str]:
    return intake_service.parse_run_ids_value(run_ids)


def _parse_checkpoint_refs_value(checkpoint_refs: Any) -> list[str]:
    if isinstance(checkpoint_refs, str):
        return arena.parse_checkpoint_refs(checkpoint_refs)
    if isinstance(checkpoint_refs, Sequence) and not isinstance(checkpoint_refs, (str, bytes)):
        refs = []
        for item in checkpoint_refs:
            stripped = str(item).strip()
            if stripped:
                refs.append(runs.require_relative_ref(stripped).as_posix())
        return refs
    return []


def _intake_scan_spec_is_live_watch(scan_spec: Mapping[str, Any]) -> bool:
    return bool(
        _parse_run_ids(scan_spec.get("run_ids"))
        or str(scan_spec.get("run_id_prefix") or "").strip()
    )


def _is_unset_int(value: Any) -> bool:
    return value in (None, "", 0, "0")


def _live_intake_scheduler_values(
    *,
    scan_spec: Mapping[str, Any],
    rating_defaults: Mapping[str, Any],
    extra: Mapping[str, Any],
    continue_from_latest: bool,
) -> tuple[str, int | None, int]:
    live_continuation = bool(continue_from_latest) or _intake_scan_spec_is_live_watch(scan_spec)
    pair_selection_raw = extra.get(
        "pair_selection",
        rating_defaults.get("pair_selection"),
    )
    pair_selection = str(
        pair_selection_raw
        or (
            arena.DEFAULT_LIVE_INTAKE_PAIR_SELECTION
            if live_continuation
            else arena.DEFAULT_RATING_PAIR_SELECTION
        )
    )
    pairs_per_round_raw = extra.get(
        "pairs_per_round",
        rating_defaults.get("pairs_per_round"),
    )
    pairs_per_round = int(pairs_per_round_raw) if not _is_unset_int(pairs_per_round_raw) else None
    if live_continuation and pairs_per_round is None:
        pairs_per_round = int(arena.DEFAULT_LIVE_INTAKE_PAIRS_PER_ROUND)
        if pair_selection == arena.RATING_PAIR_SELECTION_ALL_PAIRS:
            pair_selection = arena.DEFAULT_LIVE_INTAKE_PAIR_SELECTION
    active_pool_limit = int(
        extra.get(
            "active_pool_limit",
            rating_defaults.get(
                "active_pool_limit",
                arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT,
            ),
        )
        or arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT
    )
    if live_continuation:
        active_pool_limit = min(active_pool_limit, arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT)
        if pair_selection == arena.RATING_PAIR_SELECTION_ALL_PAIRS:
            pair_selection = arena.DEFAULT_LIVE_INTAKE_PAIR_SELECTION
            if (
                pairs_per_round is None
                or pairs_per_round > int(arena.DEFAULT_LIVE_INTAKE_PAIRS_PER_ROUND)
            ):
                pairs_per_round = int(arena.DEFAULT_LIVE_INTAKE_PAIRS_PER_ROUND)
    return pair_selection, pairs_per_round, active_pool_limit


def _live_intake_gif_values(
    *,
    scan_spec: Mapping[str, Any],
    rating_defaults: Mapping[str, Any],
    extra: Mapping[str, Any],
    continue_from_latest: bool,
) -> tuple[bool, int, str]:
    live_continuation = bool(continue_from_latest) or _intake_scan_spec_is_live_watch(scan_spec)
    gif_sample_games_per_pair = int(
        extra.get(
            "gif_sample_games_per_pair",
            rating_defaults.get(
                "gif_sample_games_per_pair",
                arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR,
            ),
        )
    )
    if live_continuation:
        if gif_sample_games_per_pair >= 0:
            gif_sample_games_per_pair = max(
                int(arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR),
                gif_sample_games_per_pair,
            )
        return (
            True,
            gif_sample_games_per_pair,
            str(
                extra.get(
                    "gif_sample_strategy",
                    rating_defaults.get(
                        "gif_sample_strategy",
                        arena.DEFAULT_GIF_SAMPLE_STRATEGY,
                    ),
                )
            ),
        )
    return (
        bool(extra.get("save_gif", rating_defaults.get("save_gif", arena.DEFAULT_SAVE_GIF))),
        gif_sample_games_per_pair,
        str(
            extra.get(
                "gif_sample_strategy",
                rating_defaults.get(
                    "gif_sample_strategy",
                    arena.DEFAULT_GIF_SAMPLE_STRATEGY,
                ),
            )
        ),
    )


def _repair_live_intake_rating_defaults(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    updated = dict(manifest)
    scan_spec = updated.get("scan_spec") if isinstance(updated.get("scan_spec"), Mapping) else {}
    rating_defaults = (
        dict(updated.get("rating_defaults"))
        if isinstance(updated.get("rating_defaults"), Mapping)
        else {}
    )
    live_continuation = bool(rating_defaults.get("continue_from_latest", False)) or (
        _intake_scan_spec_is_live_watch(scan_spec)
    )
    if not live_continuation:
        return arena._to_plain(updated)
    original_pair_selection = str(
        rating_defaults.get("pair_selection") or arena.DEFAULT_RATING_PAIR_SELECTION
    )
    pair_selection, pairs_per_round, active_pool_limit = _live_intake_scheduler_values(
        scan_spec=scan_spec,
        rating_defaults=rating_defaults,
        extra={},
        continue_from_latest=True,
    )
    checkpoint_count = len(updated.get("checkpoint_refs") or [])
    force_bounded = original_pair_selection == arena.RATING_PAIR_SELECTION_ALL_PAIRS
    if force_bounded:
        pair_selection = arena.DEFAULT_LIVE_INTAKE_PAIR_SELECTION
        if (
            pairs_per_round is None
            or pairs_per_round > int(arena.DEFAULT_LIVE_INTAKE_PAIRS_PER_ROUND)
        ):
            pairs_per_round = int(arena.DEFAULT_LIVE_INTAKE_PAIRS_PER_ROUND)
    original_save_gif = bool(rating_defaults.get("save_gif", arena.DEFAULT_SAVE_GIF))
    original_gif_sample_games = int(
        rating_defaults.get(
            "gif_sample_games_per_pair",
            arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR,
        )
    )
    save_gif, gif_sample_games_per_pair, gif_sample_strategy = _live_intake_gif_values(
        scan_spec=scan_spec,
        rating_defaults=rating_defaults,
        extra={},
        continue_from_latest=True,
    )
    rating_defaults["continue_from_latest"] = True
    rating_defaults["pair_selection"] = pair_selection
    rating_defaults["pairs_per_round"] = pairs_per_round
    rating_defaults["active_pool_limit"] = active_pool_limit
    rating_defaults["save_gif"] = save_gif
    rating_defaults["gif_sample_games_per_pair"] = gif_sample_games_per_pair
    rating_defaults["gif_sample_strategy"] = gif_sample_strategy
    if force_bounded:
        rating_defaults["live_all_pairs_repaired_to_bounded"] = True
        rating_defaults["live_all_pairs_repair_checkpoint_count"] = checkpoint_count
        rating_defaults["live_all_pairs_repair_reason"] = "live_pool_exceeds_active_pool_limit"
    if (
        not original_save_gif
        or (
            original_gif_sample_games >= 0
            and original_gif_sample_games < int(arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR)
        )
    ):
        rating_defaults["live_gif_repaired_to_enabled"] = True
        rating_defaults["live_gif_repair_checkpoint_count"] = checkpoint_count
        rating_defaults["live_gif_repair_reason"] = "live_tournament_gifs_are_required"
    updated["rating_defaults"] = arena._to_plain(rating_defaults)
    return arena._to_plain(updated)


def _explicit_checkpoint_refs_scan_spec(checkpoint_refs: Sequence[str]) -> dict[str, Any]:
    return {
        "checkpoint_refs": sorted(_clean_ref_set(checkpoint_refs)),
        "run_ids": "",
        "run_id_prefix": "",
        "max_runs": 0,
        "checkpoint_iteration": None,
        "checkpoint_selection": arena.CHECKPOINT_SELECTION_LATEST,
    }


def _scan_spec_with_checkpoint_refs(
    checkpoint_refs: Sequence[str],
    *scan_specs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    refs = sorted(_clean_ref_set(checkpoint_refs))
    for scan_spec in scan_specs:
        if isinstance(scan_spec, Mapping) and _intake_scan_spec_is_live_watch(scan_spec):
            updated = dict(scan_spec)
            updated["checkpoint_refs"] = refs
            return arena._to_plain(updated)
    return _explicit_checkpoint_refs_scan_spec(refs)


def _checkpoint_iteration_from_raw(raw: Any) -> int | None:
    if raw in (None, "", -1, "-1"):
        return None
    return int(raw)


def _discover_checkpoint_refs_from_scan_spec(
    scan_spec: Mapping[str, Any],
    *,
    mount: Path,
) -> dict[str, Any]:
    explicit_refs = _parse_checkpoint_refs_value(scan_spec.get("checkpoint_refs"))
    if explicit_refs:
        if _intake_scan_spec_is_live_watch(scan_spec):
            live = _discover_checkpoint_refs(
                mount,
                run_ids=_parse_run_ids(scan_spec.get("run_ids")),
                run_id_prefix=str(scan_spec.get("run_id_prefix") or ""),
                max_runs=int(scan_spec.get("max_runs") or 0),
                checkpoint_iteration=_checkpoint_iteration_from_raw(
                    scan_spec.get("checkpoint_iteration")
                ),
                checkpoint_selection=str(
                    scan_spec.get("checkpoint_selection") or arena.CHECKPOINT_SELECTION_LATEST
                ),
            )
            checkpoint_refs = []
            seen_refs: set[str] = set()
            for ref in [
                *explicit_refs,
                *[str(ref) for ref in live.get("checkpoint_refs", []) if str(ref).strip()],
            ]:
                clean_ref = runs.require_relative_ref(str(ref)).as_posix()
                if clean_ref in seen_refs:
                    continue
                seen_refs.add(clean_ref)
                checkpoint_refs.append(clean_ref)
            rows = [
                _checkpoint_discovery_row_from_ref(ref, mount=mount, found=True)
                for ref in explicit_refs
            ]
            rows.extend(dict(row) for row in live.get("rows", []) if isinstance(row, Mapping))
            deduped_rows = []
            seen_row_refs: set[str] = set()
            for row in rows:
                ref = str(row.get("checkpoint_ref") or "")
                if ref and ref in seen_row_refs:
                    continue
                if ref:
                    seen_row_refs.add(ref)
                deduped_rows.append(row)
            return {
                **live,
                "checkpoint_selection": "explicit_refs_plus_run_watch",
                "selection": "explicit_refs_plus_run_watch",
                "checkpoint_refs": checkpoint_refs,
                "rows": deduped_rows,
                "found_checkpoint_count": len(checkpoint_refs),
                "found_count": len(checkpoint_refs),
            }
        return {
            "schema_id": "curvyzero_curvytron_checkpoint_discovery/v0",
            "checkpoint_volume_name": CHECKPOINT_VOLUME_NAME,
            "checkpoint_scan_glob": arena.CHECKPOINT_SCAN_GLOB,
            "checkpoint_selection": "explicit_refs",
            "selection": "explicit_refs",
            "run_ids": [],
            "run_id_prefix": "",
            "requested_run_count": 0,
            "selected_run_count": 0,
            "found_run_count": 0,
            "missing_run_count": 0,
            "found_checkpoint_count": len(explicit_refs),
            "found_count": len(explicit_refs),
            "missing_count": 0,
            "checkpoint_refs": explicit_refs,
            "rows": [
                _checkpoint_discovery_row_from_ref(ref, mount=mount, found=True)
                for ref in explicit_refs
            ],
        }
    return _discover_checkpoint_refs(
        mount,
        run_ids=_parse_run_ids(scan_spec.get("run_ids")),
        run_id_prefix=str(scan_spec.get("run_id_prefix") or ""),
        max_runs=int(scan_spec.get("max_runs") or 0),
        checkpoint_iteration=_checkpoint_iteration_from_raw(scan_spec.get("checkpoint_iteration")),
        checkpoint_selection=str(
            scan_spec.get("checkpoint_selection") or arena.CHECKPOINT_SELECTION_LATEST
        ),
    )


def _validate_submitted_checkpoint_refs_exist(
    checkpoint_refs: Sequence[str],
    *,
    mount: Path,
) -> list[str]:
    clean_refs = []
    for raw_ref in checkpoint_refs:
        ref = runs.require_relative_ref(str(raw_ref)).as_posix()
        if _checkpoint_iteration_from_path(Path(ref)) is None:
            raise ValueError(
                "submitted checkpoint refs must be immutable exact "
                f"iteration_N.pth.tar files; got {ref!r}"
            )
        path = runs.volume_path(mount, ref)
        if not path.is_file():
            raise ValueError(f"submitted checkpoint ref does not exist: {ref}")
        if path.stat().st_size <= 0:
            raise ValueError(f"submitted checkpoint ref is empty: {ref}")
        clean_refs.append(ref)
    return clean_refs


def _checkpoint_discovery_row_from_ref(
    ref: str,
    *,
    mount: Path,
    found: bool = True,
) -> dict[str, Any]:
    clean_ref = runs.require_relative_ref(ref).as_posix()
    metadata = arena.checkpoint_metadata_from_ref(clean_ref)
    row: dict[str, Any] = {
        "checkpoint_ref": clean_ref,
        "run_id": metadata.get("run_id"),
        "attempt_id": metadata.get("attempt_id"),
        "iteration": (
            metadata.get("iteration")
            if metadata.get("iteration") is not None
            else _checkpoint_iteration_from_path(Path(clean_ref))
        ),
        "found": bool(found),
    }
    path = runs.volume_path(mount, clean_ref)
    if path.is_file():
        stat = path.stat()
        row["checkpoint_mtime_ns"] = int(stat.st_mtime_ns)
        row["checkpoint_size_bytes"] = int(stat.st_size)
    sidecar = arena.checkpoint_policy_metadata_from_ref(clean_ref, mount=mount)
    model_contract = arena._checkpoint_model_contract_from_ref(clean_ref, mount=mount)
    runtime_settings = arena._checkpoint_runtime_settings_from_ref(clean_ref, mount=mount)
    for key in (
        "policy_observation_contract_id",
        "policy_observation_perspective_schema_id",
        "observation_contract",
        "policy_trail_render_mode",
        "policy_bonus_render_mode",
        "policy_observation_backend",
        "source_state_trail_render_mode",
        "source_state_bonus_render_mode",
        "model_env_variant",
        "model_reward_variant",
        "env_variant",
        "reward_variant",
        "learner_seat_mode",
    ):
        value = sidecar.get(key)
        if value is not None:
            row[key] = value
    row.setdefault(
        "policy_trail_render_mode",
        arena._checkpoint_policy_trail_render_mode_from_ref(clean_ref, mount=mount),
    )
    row.setdefault(
        "policy_bonus_render_mode",
        arena._checkpoint_policy_bonus_render_mode_from_ref(clean_ref, mount=mount),
    )
    row.setdefault(
        "policy_observation_backend",
        arena._checkpoint_policy_observation_backend_from_ref(clean_ref, mount=mount),
    )
    row.setdefault(
        "policy_observation_contract_id",
        arena._checkpoint_policy_observation_contract_id_from_ref(clean_ref, mount=mount),
    )
    row.setdefault(
        "policy_observation_perspective_schema_id",
        arena._checkpoint_policy_observation_perspective_schema_id_from_ref(
            clean_ref,
            mount=mount,
        ),
    )
    for key, value in model_contract.items():
        row.setdefault(key, value)
    for key, value in runtime_settings.items():
        row.setdefault(key, value)
    sidecar_ref = arena.checkpoint_policy_metadata_sidecar_ref(clean_ref)
    if runs.volume_path(mount, sidecar_ref).is_file():
        row["checkpoint_metadata_ref"] = sidecar_ref.as_posix()
    return row


def _intake_rating_spec_from_manifest(
    manifest: Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rating_defaults = manifest.get("rating_defaults")
    if not isinstance(rating_defaults, Mapping):
        rating_defaults = {}
    extra = dict(overrides or {})
    continue_from_latest = bool(
        extra.get(
            "continue_from_latest",
            rating_defaults.get("continue_from_latest", False),
        )
    )
    scan_spec = manifest.get("scan_spec") if isinstance(manifest.get("scan_spec"), Mapping) else {}
    pair_selection, pairs_per_round, active_pool_limit = _live_intake_scheduler_values(
        scan_spec=scan_spec,
        rating_defaults=rating_defaults,
        extra=extra,
        continue_from_latest=continue_from_latest,
    )
    save_gif, gif_sample_games_per_pair, gif_sample_strategy = _live_intake_gif_values(
        scan_spec=scan_spec,
        rating_defaults=rating_defaults,
        extra=extra,
        continue_from_latest=continue_from_latest,
    )
    checkpoints = _intake_manifest_rating_checkpoints(
        manifest,
        continue_from_latest=continue_from_latest,
    )
    return arena.normalize_rating_spec(
        {
            "tournament_id": manifest["tournament_id"],
            "rating_run_id": manifest["rating_run_id"],
            "checkpoints": checkpoints,
            "round_count": extra.get(
                "round_count",
                rating_defaults.get("round_count", arena.DEFAULT_RATING_ROUND_COUNT),
            ),
            "continue_from_latest": continue_from_latest,
            "pairs_per_round": pairs_per_round,
            "placement_min_games": extra.get(
                "placement_min_games",
                rating_defaults.get("placement_min_games"),
            ),
            "placement_min_opponents": extra.get(
                "placement_min_opponents",
                rating_defaults.get("placement_min_opponents", 20),
            ),
            "pair_selection": pair_selection,
            "games_per_pair": extra.get(
                "games_per_pair",
                rating_defaults.get("games_per_pair", arena.DEFAULT_GAMES_PER_PAIR),
            ),
            "games_per_shard": extra.get(
                "games_per_shard",
                rating_defaults.get("games_per_shard", arena.DEFAULT_GAMES_PER_SHARD),
            ),
            "reuse_policies_per_shard": extra.get(
                "reuse_policies_per_shard",
                rating_defaults.get(
                    "reuse_policies_per_shard",
                    arena.DEFAULT_REUSE_POLICIES_PER_SHARD,
                ),
            ),
            "seat_order_mode": extra.get(
                "seat_order_mode",
                rating_defaults.get("seat_order_mode", arena.DEFAULT_SEAT_ORDER_MODE),
            ),
            "seed": extra.get("seed", rating_defaults.get("seed", 0)),
            "max_steps": extra.get(
                "max_steps",
                rating_defaults.get("max_steps", arena.DEFAULT_MAX_STEPS),
            ),
            "decision_ms": extra.get(
                "decision_ms",
                rating_defaults.get("decision_ms", arena.DEFAULT_DECISION_MS),
            ),
            "decision_source_frames": extra.get(
                "decision_source_frames",
                rating_defaults.get("decision_source_frames"),
            ),
            "source_physics_step_ms": extra.get(
                "source_physics_step_ms",
                rating_defaults.get(
                    "source_physics_step_ms",
                    arena.DEFAULT_SOURCE_PHYSICS_STEP_MS,
                ),
            ),
            "policy_mode": extra.get(
                "policy_mode",
                rating_defaults.get("policy_mode", arena.POLICY_MODE_EVAL),
            ),
            "collect_temperature": extra.get(
                "collect_temperature",
                rating_defaults.get(
                    "collect_temperature",
                    arena.DEFAULT_COLLECT_TEMPERATURE,
                ),
            ),
            "collect_epsilon": extra.get(
                "collect_epsilon",
                rating_defaults.get("collect_epsilon", arena.DEFAULT_COLLECT_EPSILON),
            ),
            "policy_trail_render_mode": extra.get(
                "policy_trail_render_mode",
                rating_defaults.get("policy_trail_render_mode"),
            ),
            "policy_bonus_render_mode": extra.get(
                "policy_bonus_render_mode",
                rating_defaults.get("policy_bonus_render_mode"),
            ),
            "num_simulations": extra.get(
                "num_simulations",
                rating_defaults.get("num_simulations", arena.DEFAULT_NUM_SIMULATIONS),
            ),
            "save_gif": save_gif,
            "gif_sample_games_per_pair": gif_sample_games_per_pair,
            "gif_sample_strategy": gif_sample_strategy,
            "initial_rating": extra.get(
                "initial_rating",
                rating_defaults.get(
                    "initial_rating",
                    arena.DEFAULT_RATING_INITIAL_RATING,
                ),
            ),
            "active_pool_limit": active_pool_limit,
            "stop_when_stable": extra.get(
                "stop_when_stable",
                rating_defaults.get("stop_when_stable", False),
            ),
        }
    )


def _validate_intake_rating_defaults(
    rating_defaults: Mapping[str, Any],
    *,
    tournament_id: str,
    rating_run_id: str,
) -> dict[str, Any]:
    return arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [],
            "round_count": rating_defaults.get(
                "round_count",
                arena.DEFAULT_RATING_ROUND_COUNT,
            ),
            "continue_from_latest": rating_defaults.get("continue_from_latest", False),
            "pairs_per_round": rating_defaults.get("pairs_per_round"),
            "placement_min_games": rating_defaults.get("placement_min_games"),
            "placement_min_opponents": rating_defaults.get("placement_min_opponents", 20),
            "pair_selection": rating_defaults.get(
                "pair_selection",
                arena.DEFAULT_RATING_PAIR_SELECTION,
            ),
            "games_per_pair": rating_defaults.get(
                "games_per_pair",
                arena.DEFAULT_GAMES_PER_PAIR,
            ),
            "games_per_shard": rating_defaults.get(
                "games_per_shard",
                arena.DEFAULT_GAMES_PER_SHARD,
            ),
            "reuse_policies_per_shard": rating_defaults.get(
                "reuse_policies_per_shard",
                arena.DEFAULT_REUSE_POLICIES_PER_SHARD,
            ),
            "seed": rating_defaults.get("seed", 0),
            "max_steps": rating_defaults.get("max_steps", arena.DEFAULT_MAX_STEPS),
            "decision_ms": rating_defaults.get("decision_ms", arena.DEFAULT_DECISION_MS),
            "decision_source_frames": rating_defaults.get("decision_source_frames"),
            "source_physics_step_ms": rating_defaults.get(
                "source_physics_step_ms",
                arena.DEFAULT_SOURCE_PHYSICS_STEP_MS,
            ),
            "policy_mode": rating_defaults.get("policy_mode", arena.POLICY_MODE_EVAL),
            "collect_temperature": rating_defaults.get(
                "collect_temperature",
                arena.DEFAULT_COLLECT_TEMPERATURE,
            ),
            "collect_epsilon": rating_defaults.get(
                "collect_epsilon",
                arena.DEFAULT_COLLECT_EPSILON,
            ),
            "policy_trail_render_mode": rating_defaults.get("policy_trail_render_mode"),
            "policy_bonus_render_mode": rating_defaults.get("policy_bonus_render_mode"),
            "num_simulations": rating_defaults.get(
                "num_simulations",
                arena.DEFAULT_NUM_SIMULATIONS,
            ),
            "save_gif": rating_defaults.get("save_gif", arena.DEFAULT_SAVE_GIF),
            "gif_sample_games_per_pair": rating_defaults.get(
                "gif_sample_games_per_pair",
                arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR,
            ),
            "gif_sample_strategy": rating_defaults.get(
                "gif_sample_strategy",
                arena.DEFAULT_GIF_SAMPLE_STRATEGY,
            ),
            "initial_rating": rating_defaults.get(
                "initial_rating",
                arena.DEFAULT_RATING_INITIAL_RATING,
            ),
            "active_pool_limit": rating_defaults.get(
                "active_pool_limit",
                arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT,
            ),
            "stop_when_stable": rating_defaults.get("stop_when_stable", False),
        }
    )


def _intake_manifest_rating_checkpoints(
    manifest: Mapping[str, Any],
    *,
    continue_from_latest: bool,
) -> list[str | dict[str, Any]]:
    checkpoint_refs = _intake_manifest_rating_checkpoint_refs(
        manifest,
        continue_from_latest=continue_from_latest,
    )
    discovery = manifest.get("discovery")
    rows = discovery.get("rows") if isinstance(discovery, Mapping) else []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        rows = []
    rows_by_ref = {
        str(row["checkpoint_ref"]): row
        for row in rows
        if isinstance(row, Mapping) and row.get("checkpoint_ref")
    }
    checkpoints: list[str | dict[str, Any]] = []
    for ref in checkpoint_refs:
        row = rows_by_ref.get(ref)
        if not isinstance(row, Mapping):
            row = _checkpoint_discovery_row_from_ref(
                ref,
                mount=RUNS_MOUNT,
                found=True,
            )
        checkpoints.append(_rating_checkpoint_spec_from_discovery_row(ref, row))
    return checkpoints


_RATING_CHECKPOINT_DISCOVERY_KEYS = (
    "run_id",
    "attempt_id",
    "iteration",
    "checkpoint_mtime_ns",
    "checkpoint_size_bytes",
    "exp_dir_name",
    "policy_observation_contract_id",
    "policy_observation_perspective_schema_id",
    "observation_contract",
    "policy_trail_render_mode",
    "policy_bonus_render_mode",
    "policy_observation_backend",
    "source_state_trail_render_mode",
    "source_state_bonus_render_mode",
    "model_env_variant",
    "model_reward_variant",
    "env_variant",
    "reward_variant",
    "decision_ms",
    "decision_source_frames",
    "source_physics_step_ms",
    "source_max_steps",
    "learner_seat_mode",
    "checkpoint_metadata_ref",
)


def _rating_checkpoint_spec_from_discovery_row(
    ref: str,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {"checkpoint_ref": ref}
    for key in _RATING_CHECKPOINT_DISCOVERY_KEYS:
        if row.get(key) is not None:
            checkpoint[key] = row[key]
    return checkpoint


def _rating_checkpoint_specs_from_refs(
    refs: Sequence[str],
    *,
    discovery: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows = discovery.get("rows") if isinstance(discovery, Mapping) else []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        rows = []
    rows_by_ref = {
        str(row["checkpoint_ref"]): row
        for row in rows
        if isinstance(row, Mapping) and row.get("checkpoint_ref")
    }
    checkpoints = []
    for ref in refs:
        row = rows_by_ref.get(ref)
        if not isinstance(row, Mapping):
            row = _checkpoint_discovery_row_from_ref(ref, mount=RUNS_MOUNT, found=True)
        checkpoints.append(_rating_checkpoint_spec_from_discovery_row(ref, row))
    return checkpoints


def _intake_manifest_rating_checkpoint_refs(
    manifest: Mapping[str, Any],
    *,
    continue_from_latest: bool,
) -> list[str]:
    primary_refs = [
        str(ref) for ref in manifest.get("checkpoint_refs", []) if str(ref).strip()
    ]
    if not continue_from_latest:
        return primary_refs
    refs: list[str] = []
    seen: set[str] = set()
    for ref in [
        *primary_refs,
        *[str(ref) for ref in manifest.get("seen_checkpoint_refs", []) if str(ref).strip()],
    ]:
        if ref in seen:
            continue
        refs.append(ref)
        seen.add(ref)
    return refs


def _intake_manifest_from_discovery(
    *,
    tournament_id: str,
    rating_run_id: str,
    scan_spec: Mapping[str, Any],
    rating_defaults: Mapping[str, Any],
    discovery: Mapping[str, Any],
    existing: Mapping[str, Any] | None = None,
    active: bool = True,
) -> dict[str, Any]:
    clean_tournament_id = runs.clean_id(tournament_id, label="tournament_id")
    clean_rating_run_id = runs.clean_id(rating_run_id, label="rating_run_id")
    existing_refs = []
    queued_refs: set[str] = set()
    if isinstance(existing, Mapping):
        existing_refs = [
            str(ref) for ref in existing.get("seen_checkpoint_refs", []) if str(ref).strip()
        ]
        queued_refs = _clean_ref_set(existing.get("queued_checkpoint_refs", []))
    checkpoint_refs = [str(ref) for ref in discovery.get("checkpoint_refs", []) if str(ref).strip()]
    seen_refs = sorted(set(existing_refs).union(checkpoint_refs))
    checkpoint_pool_hash = _checkpoint_ref_pool_hash(checkpoint_refs)
    seen_checkpoint_pool_hash = _checkpoint_ref_pool_hash(seen_refs)
    merged_rows_by_ref: dict[str, dict[str, Any]] = {}
    existing_discovery = existing.get("discovery") if isinstance(existing, Mapping) else {}
    for source in (existing_discovery, discovery):
        rows = source.get("rows") if isinstance(source, Mapping) else []
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for row in rows:
            if not isinstance(row, Mapping) or not row.get("checkpoint_ref"):
                continue
            merged_rows_by_ref[str(row["checkpoint_ref"])] = dict(row)
    merged_discovery = dict(discovery)
    if merged_rows_by_ref:
        merged_discovery["rows"] = [
            merged_rows_by_ref[ref] for ref in seen_refs if ref in merged_rows_by_ref
        ]
    return {
        "schema_id": "curvyzero_curvytron_checkpoint_intake_manifest/v0",
        "app_name": APP_NAME,
        "artifact_volume_name": TOURNAMENT_VOLUME_NAME,
        "checkpoint_volume_name": CHECKPOINT_VOLUME_NAME,
        "dict_name": CHECKPOINT_INTAKE_DICT_NAME,
        "queue_name": CHECKPOINT_INTAKE_QUEUE_NAME,
        "manifest_key": _intake_manifest_key(clean_tournament_id, clean_rating_run_id),
        "queue_partition": _intake_queue_partition(clean_tournament_id, clean_rating_run_id),
        "tournament_id": clean_tournament_id,
        "rating_run_id": clean_rating_run_id,
        "active": bool(active),
        "updated_at": runs.utc_timestamp(),
        "scan_spec": arena._to_plain(dict(scan_spec)),
        "rating_defaults": arena._to_plain(dict(rating_defaults)),
        "checkpoint_refs": checkpoint_refs,
        "seen_checkpoint_refs": seen_refs,
        "queued_checkpoint_refs": sorted(queued_refs.intersection(seen_refs)),
        "checkpoint_count": len(checkpoint_refs),
        "seen_checkpoint_count": len(seen_refs),
        "queued_checkpoint_count": len(queued_refs.intersection(seen_refs)),
        "checkpoint_pool_hash": checkpoint_pool_hash,
        "seen_checkpoint_pool_hash": seen_checkpoint_pool_hash,
        "discovery": arena._to_plain(merged_discovery),
    }


def _mark_intake_manifest_queued(
    manifest: Mapping[str, Any],
    checkpoint_refs: Sequence[str],
) -> dict[str, Any]:
    updated = dict(manifest)
    queued_refs = _clean_ref_set(updated.get("queued_checkpoint_refs", []))
    queued_refs.update(_clean_ref_set(checkpoint_refs))
    seen_refs = _clean_ref_set(updated.get("seen_checkpoint_refs", []))
    queued_refs = queued_refs.intersection(seen_refs) if seen_refs else queued_refs
    updated["queued_checkpoint_refs"] = sorted(queued_refs)
    updated["queued_checkpoint_count"] = len(queued_refs)
    updated["updated_at"] = runs.utc_timestamp()
    return updated


def _discovery_rows_by_ref(discovery: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(discovery, Mapping):
        return {}
    rows = discovery.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return {}
    rows_by_ref: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("checkpoint_ref"):
            continue
        rows_by_ref[str(row["checkpoint_ref"])] = dict(row)
    return rows_by_ref


def _intake_manifest_with_merged_pool(
    candidate: Mapping[str, Any],
    current: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge a stale intake write without shrinking the durable checkpoint pool."""

    updated = dict(candidate)
    if not isinstance(current, Mapping):
        return updated
    if current.get("manifest_key") != candidate.get("manifest_key"):
        return updated

    candidate_refs = _clean_ref_set(candidate.get("checkpoint_refs", []))
    current_refs = _clean_ref_set(current.get("checkpoint_refs", []))
    merged_refs = sorted(candidate_refs.union(current_refs))

    candidate_seen_refs = _clean_ref_set(candidate.get("seen_checkpoint_refs", []))
    current_seen_refs = _clean_ref_set(current.get("seen_checkpoint_refs", []))
    merged_seen_refs = sorted(candidate_seen_refs.union(current_seen_refs).union(merged_refs))

    candidate_queued_refs = _clean_ref_set(candidate.get("queued_checkpoint_refs", []))
    current_queued_refs = _clean_ref_set(current.get("queued_checkpoint_refs", []))
    merged_queued_refs = sorted(
        candidate_queued_refs.union(current_queued_refs).intersection(set(merged_seen_refs))
    )

    candidate_scan_spec = (
        candidate.get("scan_spec") if isinstance(candidate.get("scan_spec"), Mapping) else {}
    )
    current_scan_spec = (
        current.get("scan_spec") if isinstance(current.get("scan_spec"), Mapping) else {}
    )
    current_has_wider_pool = bool(current_refs - candidate_refs)
    scan_spec_needs_merge = (
        current_has_wider_pool
        or _intake_scan_spec_is_live_watch(candidate_scan_spec)
        or _intake_scan_spec_is_live_watch(current_scan_spec)
    )
    if scan_spec_needs_merge:
        updated["scan_spec"] = _scan_spec_with_checkpoint_refs(
            merged_refs,
            candidate_scan_spec,
            current_scan_spec,
        )
    if current_has_wider_pool:
        if isinstance(current.get("rating_defaults"), Mapping):
            updated["rating_defaults"] = arena._to_plain(dict(current["rating_defaults"]))

    rows_by_ref: dict[str, dict[str, Any]] = {}
    rows_by_ref.update(_discovery_rows_by_ref(current.get("discovery")))
    rows_by_ref.update(_discovery_rows_by_ref(candidate.get("discovery")))
    discovery = (
        dict(candidate.get("discovery")) if isinstance(candidate.get("discovery"), Mapping) else {}
    )
    discovery["checkpoint_refs"] = merged_refs
    discovery["found_count"] = len(merged_refs)
    discovery["found_checkpoint_count"] = len(merged_refs)
    discovery["rows"] = [
        rows_by_ref.get(ref)
        or _checkpoint_discovery_row_from_ref(ref, mount=RUNS_MOUNT, found=True)
        for ref in merged_refs
    ]

    updated["checkpoint_refs"] = merged_refs
    updated["seen_checkpoint_refs"] = merged_seen_refs
    updated["queued_checkpoint_refs"] = merged_queued_refs
    updated["checkpoint_count"] = len(merged_refs)
    updated["seen_checkpoint_count"] = len(merged_seen_refs)
    updated["queued_checkpoint_count"] = len(merged_queued_refs)
    updated["checkpoint_pool_hash"] = _checkpoint_ref_pool_hash(merged_refs)
    updated["seen_checkpoint_pool_hash"] = _checkpoint_ref_pool_hash(merged_seen_refs)
    updated["discovery"] = arena._to_plain(discovery)
    return updated


def _put_intake_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    key = str(manifest["manifest_key"])
    current = checkpoint_intake_state.get(key, None)
    merged_manifest = _repair_live_intake_rating_defaults(
        _intake_manifest_with_merged_pool(manifest, current)
    )
    checkpoint_intake_state.put(key, arena._to_plain(merged_manifest))
    active_keys = checkpoint_intake_state.get(CHECKPOINT_INTAKE_ACTIVE_KEYS, []) or []
    if not isinstance(active_keys, list):
        active_keys = []
    active_set = {str(item) for item in active_keys}
    if merged_manifest.get("active"):
        active_set.add(key)
    else:
        active_set.discard(key)
    checkpoint_intake_state.put(CHECKPOINT_INTAKE_ACTIVE_KEYS, sorted(active_set))
    return {
        "key": key,
        "active_manifest_count": len(active_set),
        "manifest": arena._to_plain(merged_manifest),
    }


def _write_intake_manifest_artifact(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return arena.write_json_artifact(
        TOURNAMENT_MOUNT,
        arena.tournament_intake_manifest_ref(
            str(manifest["tournament_id"]),
            str(manifest["rating_run_id"]),
        ),
        manifest,
    )


def _load_intake_manifest(
    tournament_id: str,
    rating_run_id: str,
    *,
    repair_state: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    manifest_key = _intake_manifest_key(tournament_id, rating_run_id)
    manifest = checkpoint_intake_state.get(manifest_key, None)
    if isinstance(manifest, Mapping) and (
        manifest.get("tournament_id")
        and manifest.get("rating_run_id")
        and manifest.get("queue_partition")
    ):
        loaded = dict(manifest)
        repaired = False
        if repair_state:
            repaired_loaded = _repair_live_intake_rating_defaults(loaded)
            repaired = repaired_loaded != loaded
            if repaired:
                state_write = _put_intake_manifest(repaired_loaded)
                repaired_loaded = dict(state_write.get("manifest") or repaired_loaded)
            loaded = repaired_loaded
        return loaded, {
            "manifest_key": manifest_key,
            "manifest_source": "dict",
            "manifest_state_repaired": repaired,
        }
    volume_manifest = _read_json(
        runs.volume_path(
            TOURNAMENT_MOUNT,
            arena.tournament_intake_manifest_ref(tournament_id, rating_run_id),
        )
    )
    if not isinstance(volume_manifest, Mapping):
        return None, {
            "manifest_key": manifest_key,
            "manifest_source": "missing",
            "manifest_state_repaired": False,
        }
    loaded = dict(volume_manifest)
    if not (
        loaded.get("tournament_id")
        and loaded.get("rating_run_id")
        and loaded.get("queue_partition")
    ):
        return None, {
            "manifest_key": manifest_key,
            "manifest_source": "missing",
            "manifest_state_repaired": False,
        }
    loaded["manifest_key"] = str(loaded.get("manifest_key") or manifest_key)
    if str(loaded["manifest_key"]) != manifest_key:
        raise ValueError("intake manifest key does not match requested tournament/rating")
    repaired = False
    if repair_state:
        loaded = _repair_live_intake_rating_defaults(loaded)
        state_write = _put_intake_manifest(loaded)
        loaded = dict(state_write.get("manifest") or loaded)
        repaired = True
    return loaded, {
        "manifest_key": manifest_key,
        "manifest_source": "volume",
        "manifest_state_repaired": repaired,
    }


def _write_intake_tick_artifact(tick: Mapping[str, Any]) -> dict[str, Any]:
    return arena.write_json_artifact(
        TOURNAMENT_MOUNT,
        arena.tournament_intake_latest_tick_ref(
            str(tick["tournament_id"]),
            str(tick["rating_run_id"]),
        ),
        tick,
    )


def _rating_run_has_existing_output(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
) -> bool:
    refs = (
        arena.rating_config_ref(tournament_id, rating_run_id),
        arena.rating_progress_ref(tournament_id, rating_run_id),
        arena.rating_latest_ref(tournament_id, rating_run_id),
    )
    return any(runs.volume_path(mount, ref).exists() for ref in refs)


def _round_index_from_round_id(round_id: Any) -> int | None:
    text = str(round_id or "")
    prefix = "round-"
    if not text.startswith(prefix):
        return None
    try:
        return int(text[len(prefix) :])
    except ValueError:
        return None


def _round_index_from_payload(payload: Mapping[str, Any]) -> int | None:
    round_index = _round_index_from_round_id(payload.get("round_id"))
    if round_index is None and payload.get("round_index") not in (None, ""):
        try:
            round_index = int(payload["round_index"])
        except (TypeError, ValueError):
            return None
    return round_index


def _latest_rating_round_index(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
) -> int:
    latest = _read_json(
        runs.volume_path(mount, arena.rating_latest_ref(tournament_id, rating_run_id))
    )
    round_index = _round_index_from_payload(latest) if latest else None
    return -1 if round_index is None else round_index


def _rating_latest_publish_decision(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    latest = _read_json(
        runs.volume_path(mount, arena.rating_latest_ref(tournament_id, rating_run_id))
    )
    latest_round_index = _latest_rating_round_index(
        mount,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    snapshot_round_index = _round_index_from_payload(snapshot)
    if snapshot_round_index is None:
        return {
            "publish": False,
            "reason": "snapshot_missing_round_index",
            "latest_round_index": latest_round_index,
            "snapshot_round_index": None,
        }
    if not bool(snapshot.get("global_outputs_published", True)):
        return {
            "publish": False,
            "reason": str(snapshot.get("latest_write_skipped_reason") or "snapshot_not_global"),
            "latest_round_index": latest_round_index,
            "snapshot_round_index": int(snapshot_round_index),
        }
    if int(snapshot_round_index) < int(latest_round_index):
        return {
            "publish": False,
            "reason": "newer_round_already_latest",
            "latest_round_index": latest_round_index,
            "snapshot_round_index": int(snapshot_round_index),
        }
    latest_checkpoint_count = int(
        latest.get("checkpoint_count") or len(latest.get("ratings") or []) or 0
    ) if isinstance(latest, Mapping) else 0
    snapshot_checkpoint_count = int(
        snapshot.get("checkpoint_count") or len(snapshot.get("ratings") or []) or 0
    )
    if (
        latest_checkpoint_count > 0
        and snapshot_checkpoint_count > 0
        and snapshot_checkpoint_count < latest_checkpoint_count
    ):
        return {
            "publish": False,
            "reason": "higher_checkpoint_count_already_latest",
            "latest_round_index": latest_round_index,
            "snapshot_round_index": int(snapshot_round_index),
            "latest_checkpoint_count": latest_checkpoint_count,
            "snapshot_checkpoint_count": snapshot_checkpoint_count,
        }
    return {
        "publish": True,
        "reason": "",
        "latest_round_index": latest_round_index,
        "snapshot_round_index": int(snapshot_round_index),
        "latest_checkpoint_count": latest_checkpoint_count,
        "snapshot_checkpoint_count": snapshot_checkpoint_count,
    }


def _publish_rating_latest_snapshot_if_current(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    decision = _rating_latest_publish_decision(
        mount,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        snapshot=snapshot,
    )
    if decision.get("publish"):
        arena.write_json_artifact(
            mount,
            arena.rating_latest_ref(tournament_id, rating_run_id),
            _slim_rating_snapshot(snapshot),
        )
    return arena._to_plain(decision)


def _root_rating_progress_round_index(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
) -> int:
    progress = _read_json(
        runs.volume_path(mount, arena.rating_progress_ref(tournament_id, rating_run_id))
    )
    round_index = _round_index_from_payload(progress) if progress else None
    return -1 if round_index is None else round_index


def _rating_writer_has_finished(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
) -> bool:
    clean_tournament_id = runs.clean_id(tournament_id, label="tournament_id")
    clean_rating_run_id = runs.clean_id(rating_run_id, label="rating_run_id")
    latest = _read_json(
        runs.volume_path(
            mount,
            arena.rating_latest_ref(clean_tournament_id, clean_rating_run_id),
        )
    )
    latest_round_index = _round_index_from_payload(latest) if latest else None
    latest_round_index = -1 if latest_round_index is None else latest_round_index
    rating_root = runs.volume_path(
        mount,
        arena.rating_root_ref(clean_tournament_id, clean_rating_run_id),
    )
    rounds_root = rating_root / "rounds"
    if rounds_root.exists():
        for input_path in rounds_root.glob("round-*/input.json"):
            round_id = input_path.parent.name
            round_index = _round_index_from_round_id(round_id)
            if round_index is None or round_index <= latest_round_index:
                continue
            round_progress = _read_json(
                runs.volume_path(
                    mount,
                    arena.rating_round_progress_ref(
                        clean_tournament_id,
                        clean_rating_run_id,
                        round_id,
                    ),
                )
            )
            if str(round_progress.get("status") or "") == "skipped":
                continue
            ratings_path = runs.volume_path(
                mount,
                arena.rating_round_ratings_ref(
                    clean_tournament_id,
                    clean_rating_run_id,
                    round_id,
                ),
            )
            if not ratings_path.is_file():
                return False
    progress = _read_json(
        runs.volume_path(
            mount,
            arena.rating_progress_ref(clean_tournament_id, clean_rating_run_id),
        )
    )
    if progress:
        progress_round_index = _round_index_from_payload(progress)
        latest_is_finished = bool(
            latest
            and (
                latest.get("ended_at")
                or latest.get("ratings_ref")
                or latest.get("global_outputs_published")
            )
        )
        if (
            latest_is_finished
            and progress_round_index is not None
            and progress_round_index <= latest_round_index
        ):
            return True
        if str(progress.get("status") or "") == "skipped":
            return True
        if (
            progress_round_index is not None
            and progress_round_index > latest_round_index
            and str(progress.get("phase") or "") == "waiting_for_round_input"
            and int(progress.get("pair_count") or 0) == 0
            and int(progress.get("game_count") or 0) == 0
            and int(progress.get("completed_pair_count") or 0) == 0
            and int(progress.get("completed_game_count") or 0) == 0
        ):
            return True
        return bool(progress.get("ratings_written") or progress.get("status") == "complete")
    return bool(latest)


def _oldest_unrated_rating_round(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
) -> dict[str, Any] | None:
    clean_tournament_id = runs.clean_id(tournament_id, label="tournament_id")
    clean_rating_run_id = runs.clean_id(rating_run_id, label="rating_run_id")
    indices = _unrated_rating_round_indices(
        mount,
        tournament_id=clean_tournament_id,
        rating_run_id=clean_rating_run_id,
    )
    if not indices:
        return None
    round_index = indices[0]
    round_id = arena.rating_round_id(round_index)
    return arena._to_plain(
        {
            "round_id": round_id,
            "round_index": round_index,
            "input_ref": arena.rating_round_input_ref(
                clean_tournament_id,
                clean_rating_run_id,
                round_id,
            ).as_posix(),
        }
    )


def _unrated_rating_round_indices(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
    after_round_index: int | None = None,
) -> list[int]:
    clean_tournament_id = runs.clean_id(tournament_id, label="tournament_id")
    clean_rating_run_id = runs.clean_id(rating_run_id, label="rating_run_id")
    latest = _read_json(
        runs.volume_path(
            mount,
            arena.rating_latest_ref(clean_tournament_id, clean_rating_run_id),
        )
    )
    latest_round_index = _round_index_from_round_id(latest.get("round_id"))
    if latest_round_index is None and latest.get("round_index") not in (None, ""):
        latest_round_index = int(latest["round_index"])
    latest_round_index = -1 if latest_round_index is None else latest_round_index
    if after_round_index is not None:
        latest_round_index = max(latest_round_index, int(after_round_index))
    rounds_root = (
        runs.volume_path(
            mount,
            arena.rating_root_ref(clean_tournament_id, clean_rating_run_id),
        )
        / "rounds"
    )
    if not rounds_root.exists():
        return []
    candidates = []
    for input_path in rounds_root.glob("round-*/input.json"):
        round_id = input_path.parent.name
        round_index = _round_index_from_round_id(round_id)
        if round_index is None or round_index <= latest_round_index:
            continue
        round_progress = _read_json(
            runs.volume_path(
                mount,
                arena.rating_round_progress_ref(
                    clean_tournament_id,
                    clean_rating_run_id,
                    round_id,
                ),
            )
        )
        if str(round_progress.get("status") or "") == "skipped":
            continue
        ratings_path = runs.volume_path(
            mount,
            arena.rating_round_ratings_ref(
                clean_tournament_id,
                clean_rating_run_id,
                round_id,
            ),
        )
        if ratings_path.is_file():
            continue
        candidates.append(round_index)
    return sorted(set(candidates))


def _highest_skipped_rating_round_index_after(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
    after_round_index: int,
) -> int:
    clean_tournament_id = runs.clean_id(tournament_id, label="tournament_id")
    clean_rating_run_id = runs.clean_id(rating_run_id, label="rating_run_id")
    rounds_root = (
        runs.volume_path(
            mount,
            arena.rating_root_ref(clean_tournament_id, clean_rating_run_id),
        )
        / "rounds"
    )
    highest = int(after_round_index)
    if not rounds_root.exists():
        return highest
    for progress_path in rounds_root.glob("round-*/progress.json"):
        round_id = progress_path.parent.name
        round_index = _round_index_from_round_id(round_id)
        if round_index is None or round_index <= highest:
            continue
        progress = _read_json(progress_path)
        if str(progress.get("status") or "") == "skipped":
            highest = round_index
    return highest


def _intake_rating_reduce_claim_key(
    *,
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
) -> str:
    return (
        "rating_reduce_claim:"
        f"{runs.clean_id(tournament_id, label='tournament_id')}:"
        f"{runs.clean_id(rating_run_id, label='rating_run_id')}:"
        f"{runs.clean_id(round_id, label='round_id')}"
    )


def _rating_round_checkpoint_count(input_payload: Mapping[str, Any]) -> int:
    roster = input_payload.get("checkpoint_roster")
    if isinstance(roster, Sequence) and not isinstance(roster, (str, bytes)):
        return len(roster)
    rating_spec = input_payload.get("rating_spec")
    if isinstance(rating_spec, Mapping):
        checkpoints = rating_spec.get("checkpoints")
        if isinstance(checkpoints, Sequence) and not isinstance(checkpoints, (str, bytes)):
            return len(checkpoints)
    return 0


def _path_mtime_seconds(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _rating_round_skip_decision(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
    desired_checkpoint_count: int,
    latest_checkpoint_count: int = 0,
    desired_rating_spec: Mapping[str, Any] | None = None,
    stale_after_seconds: int = DEFAULT_RATING_ROUND_STALE_SECONDS,
    scan_output_progress: bool = False,
) -> dict[str, Any]:
    if scan_output_progress and Path(mount) == Path(TOURNAMENT_MOUNT):
        _reload_volume(tournament_volume)
    input_ref = arena.rating_round_input_ref(tournament_id, rating_run_id, round_id)
    input_path = runs.volume_path(mount, input_ref)
    input_payload = _read_json(input_path)
    if not input_payload:
        return {
            "skip": False,
            "reason": "missing_round_input",
            "input_ref": input_ref.as_posix(),
        }
    progress_ref = arena.rating_round_progress_ref(tournament_id, rating_run_id, round_id)
    progress_path = runs.volume_path(mount, progress_ref)
    progress = _read_json(progress_path)
    now_ts = time.time()
    input_updated_ts = _path_mtime_seconds(input_path)
    progress_updated_ts = _path_mtime_seconds(progress_path)
    input_checkpoint_count = _rating_round_checkpoint_count(input_payload)
    pair_count = int(input_payload.get("pair_count") or 0)
    game_count = int(input_payload.get("game_count") or 0)
    completed_game_count = int(progress.get("completed_game_count") or 0)
    started_pair_count = int(progress.get("started_pair_count") or 0)
    desired_pool_is_newer = (
        int(desired_checkpoint_count) > 0
        and input_checkpoint_count > 0
        and int(desired_checkpoint_count) > input_checkpoint_count
    )
    input_is_no_newer_than_latest = (
        int(latest_checkpoint_count) > 0
        and input_checkpoint_count > 0
        and input_checkpoint_count <= int(latest_checkpoint_count)
    )
    different_spec_error = ""
    if isinstance(desired_rating_spec, Mapping) and desired_rating_spec:
        round_index = _round_index_from_round_id(round_id)
        if round_index is not None:
            try:
                _validate_existing_rating_round_input_matches_spec(
                    input_payload,
                    spec=arena.normalize_rating_spec(desired_rating_spec),
                    round_id=round_id,
                    round_index=round_index,
                )
            except FileExistsError as exc:
                different_spec_error = str(exc)
    different_spec_already_rated_pool = bool(
        different_spec_error and desired_pool_is_newer and input_is_no_newer_than_latest
    )
    progress_scan_error = None
    latest_result_ts: float | None = None
    progress_scan_mode = ""
    progress_count_semantics = ""
    if (
        scan_output_progress
        and game_count > 0
        and completed_game_count < game_count
        and not different_spec_already_rated_pool
    ):
        try:
            if game_count > DEFAULT_RATING_ROUND_PROGRESS_FULL_SCAN_GAME_LIMIT:
                progress_scan_mode = "bounded_activity_probe"
                progress_count_semantics = "bounded_activity_sample"
                live_progress = _rating_round_activity_probe(
                    mount,
                    tournament_id=tournament_id,
                    rating_run_id=rating_run_id,
                    round_id=round_id,
                    max_pairs=256,
                    stop_after_first_output=False,
                    stale_after_seconds=stale_after_seconds,
                )
            else:
                progress_scan_mode = "full_progress_payload"
                progress_count_semantics = "full_progress_count"
                live_progress, _games_by_battle = _rating_round_progress_payload(
                    mount,
                    tournament_id=tournament_id,
                    rating_run_id=rating_run_id,
                    round_id=round_id,
                    load_summaries=False,
                    pair_only=True,
                    count_game_summaries=True,
                )
            raw_latest_result_ts = live_progress.get("latest_result_ts")
            if raw_latest_result_ts not in (None, ""):
                latest_result_ts = float(raw_latest_result_ts)
            completed_game_count = max(
                completed_game_count,
                int(live_progress.get("completed_game_count") or 0),
            )
            started_pair_count = max(
                started_pair_count,
                int(live_progress.get("started_pair_count") or 0),
            )
            pair_count = pair_count or int(live_progress.get("pair_count") or 0)
            game_count = game_count or int(live_progress.get("game_count") or 0)
        except Exception as exc:  # pragma: no cover - remote recovery telemetry.
            if not progress_scan_mode:
                progress_scan_mode = "failed"
            progress_scan_error = f"{type(exc).__name__}: {exc}"
    newest_real_activity_ts = max(
        value
        for value in (
            input_updated_ts,
            latest_result_ts,
        )
        if value is not None
    ) if (input_updated_ts is not None or latest_result_ts is not None) else now_ts
    effective_stale_after_seconds = max(0, int(stale_after_seconds))
    input_age_seconds = (
        max(0.0, now_ts - input_updated_ts) if input_updated_ts is not None else None
    )
    stale_age_seconds = max(0.0, now_ts - newest_real_activity_ts)
    is_stale = stale_age_seconds >= effective_stale_after_seconds
    no_completed_games = completed_game_count <= 0
    incomplete_games = (
        completed_game_count > 0 and game_count > 0 and completed_game_count < game_count
    )
    scan_count_is_exhaustive = progress_scan_mode not in {"bounded_activity_probe"}
    different_spec_zero_output = bool(
        different_spec_error
        and no_completed_games
        and (input_is_no_newer_than_latest or is_stale)
    )
    different_spec_stale_incomplete = bool(
        different_spec_error and incomplete_games and is_stale and scan_count_is_exhaustive
    )
    zero_progress = no_completed_games and started_pair_count <= 0
    stale_started_without_completed_games = (
        no_completed_games
        and started_pair_count > 0
        and is_stale
        and game_count > 0
        and scan_count_is_exhaustive
    )
    stale_incomplete_smaller_pool = (
        incomplete_games and desired_pool_is_newer and is_stale and scan_count_is_exhaustive
    )
    stale_incomplete_round = incomplete_games and is_stale and scan_count_is_exhaustive
    scan_failed_blocks_skip = bool(scan_output_progress and progress_scan_error)
    partial_reduce_after_seconds = max(
        0,
        int(DEFAULT_RATING_ROUND_PARTIAL_REDUCE_AFTER_SECONDS),
    )
    partial_reduce_min_completed_games = max(
        1,
        int(DEFAULT_RATING_ROUND_PARTIAL_REDUCE_MIN_COMPLETED_GAMES),
    )
    partial_reduce_recommended = bool(
        game_count > 0
        and not different_spec_error
        and not scan_failed_blocks_skip
        and input_age_seconds is not None
        and input_age_seconds >= partial_reduce_after_seconds
        and completed_game_count >= partial_reduce_min_completed_games
        and completed_game_count < game_count
    )
    skip = bool(
        game_count > 0
        and not scan_failed_blocks_skip
        and not partial_reduce_recommended
        and (
            different_spec_zero_output
            or different_spec_already_rated_pool
            or different_spec_stale_incomplete
            or (
                is_stale
                and (
            (
                no_completed_games
                and (desired_pool_is_newer or zero_progress or started_pair_count > 0)
            )
            or stale_incomplete_round
                )
            )
        )
    )
    reason = ""
    if skip:
        if different_spec_zero_output:
            reason = "different_spec_zero_output"
        elif different_spec_already_rated_pool:
            reason = "different_spec_already_rated_pool"
        elif different_spec_stale_incomplete:
            reason = "different_spec_stale_incomplete"
        elif stale_incomplete_smaller_pool:
            reason = "stale_incomplete_smaller_pool"
        elif stale_incomplete_round:
            reason = "stale_incomplete_round"
        elif stale_started_without_completed_games:
            reason = "stale_started_without_completed_games"
        elif desired_pool_is_newer:
            reason = "zero_progress_smaller_pool"
        else:
            reason = "zero_progress_orphan_round"
    return {
        "skip": skip,
        "reason": reason or "not_skippable",
        "input_ref": input_ref.as_posix(),
        "progress_ref": progress_ref.as_posix(),
        "input_checkpoint_count": input_checkpoint_count,
        "desired_checkpoint_count": int(desired_checkpoint_count),
        "latest_checkpoint_count": int(latest_checkpoint_count),
        "pair_count": pair_count,
        "game_count": game_count,
        "completed_game_count": completed_game_count,
        "started_pair_count": started_pair_count,
        "stale_after_seconds": effective_stale_after_seconds,
        "input_age_seconds": input_age_seconds,
        "stale_age_seconds": stale_age_seconds,
        "stale_basis": "round_input_or_game_output",
        "input_updated_ts": input_updated_ts,
        "progress_updated_ts_ignored_for_stale": progress_updated_ts,
        "latest_result_ts": latest_result_ts,
        "newest_real_activity_ts": newest_real_activity_ts,
        "is_stale": is_stale,
        "scan_output_progress": bool(scan_output_progress),
        "progress_scan_mode": progress_scan_mode,
        "progress_count_semantics": progress_count_semantics,
        "progress_scan_error": progress_scan_error,
        "progress_scan_error_blocks_skip": scan_failed_blocks_skip,
        "partial_reduce_recommended": partial_reduce_recommended,
        "partial_reduce_after_seconds": partial_reduce_after_seconds,
        "partial_reduce_min_completed_games": partial_reduce_min_completed_games,
        "different_spec_error": different_spec_error,
        "different_spec": bool(different_spec_error),
        "input_is_no_newer_than_latest": input_is_no_newer_than_latest,
    }


def _write_rating_round_skipped_progress(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
    round_index: int,
    skip_decision: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
        "app_name": APP_NAME,
        "artifact_volume_name": TOURNAMENT_VOLUME_NAME,
        "checkpoint_volume_name": CHECKPOINT_VOLUME_NAME,
        "tournament_id": tournament_id,
        "rating_run_id": rating_run_id,
        "round_id": round_id,
        "round_index": int(round_index),
        "status": "skipped",
        "phase": "stale_orphan_round_skipped",
        "skip_reason": str(skip_decision.get("reason") or "stale_orphan_round"),
        "pair_count": int(skip_decision.get("pair_count") or 0),
        "game_count": int(skip_decision.get("game_count") or 0),
        "completed_pair_count": 0,
        "completed_game_count": int(skip_decision.get("completed_game_count") or 0),
        "started_pair_count": int(skip_decision.get("started_pair_count") or 0),
        "completion_fraction": 0.0,
        "estimated_completion_fraction": 0.0,
        "skipped_at": runs.utc_timestamp(),
        "updated_at": runs.utc_timestamp(),
        "updated_ts": time.time(),
        "input_ref": skip_decision.get("input_ref")
        or arena.rating_round_input_ref(tournament_id, rating_run_id, round_id).as_posix(),
        "progress_ref": arena.rating_progress_ref(tournament_id, rating_run_id).as_posix(),
        "round_progress_ref": arena.rating_round_progress_ref(
            tournament_id,
            rating_run_id,
            round_id,
        ).as_posix(),
        "latest_ref": arena.rating_latest_ref(tournament_id, rating_run_id).as_posix(),
        "skip_decision": arena._to_plain(dict(skip_decision)),
    }
    _write_rating_progress(mount, payload)
    return arena._to_plain(payload)


def _rating_latest_checkpoint_refs(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
) -> list[str]:
    latest = _read_json(
        runs.volume_path(
            mount,
            arena.rating_latest_ref(tournament_id, rating_run_id),
        )
    )
    ratings = latest.get("ratings")
    if not isinstance(ratings, Sequence) or isinstance(ratings, (str, bytes)):
        return []
    refs: list[str] = []
    for row in ratings:
        if not isinstance(row, Mapping):
            continue
        ref = str(row.get("checkpoint_ref") or "").strip()
        if ref:
            refs.append(ref)
    return refs


def _assert_rating_round_input_still_matches(
    mount: Path,
    input_payload: Mapping[str, Any],
) -> None:
    ref = arena.rating_round_input_ref(
        str(input_payload["tournament_id"]),
        str(input_payload["rating_run_id"]),
        str(input_payload["round_id"]),
    )
    path = runs.volume_path(mount, ref)
    if not path.exists():
        return
    current = _read_json(path)
    watched_keys = (
        "pool_hash",
        "roster_hash",
        "context_hash",
        "pair_count",
        "game_count",
    )
    changed = [key for key in watched_keys if current.get(key) != input_payload.get(key)]
    if changed:
        raise RuntimeError(
            f"rating round input was replaced while work was running: {', '.join(changed)}"
        )


def _validate_existing_rating_round_input_matches_spec(
    input_payload: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    round_id: str,
    round_index: int,
) -> None:
    expected = {
        "tournament_id": spec["tournament_id"],
        "rating_run_id": spec["rating_run_id"],
        "round_id": round_id,
        "round_index": int(round_index),
        "pool_hash": arena.rating_pool_hash(spec["checkpoints"]),
        "roster_hash": arena.rating_pool_hash(spec["checkpoints"]),
        "context_hash": arena.rating_context_hash(spec),
    }
    changed = []
    for key, value in expected.items():
        current = input_payload.get(key)
        if key == "round_index" and current not in (None, ""):
            current = int(current)
        if current != value:
            changed.append(key)
    if changed:
        raise FileExistsError(
            f"existing rating round input belongs to a different spec: {', '.join(changed)}"
        )


def _completed_rating_round_result_from_snapshot(
    mount: Path,
    *,
    spec: Mapping[str, Any],
    round_id: str,
    round_index: int,
    snapshot: Mapping[str, Any],
    work_summary: Mapping[str, Any] | None = None,
    resumed_existing_round: bool = False,
) -> dict[str, Any]:
    pair_history = _read_json(
        runs.volume_path(
            mount,
            arena.rating_pair_history_ref(spec["tournament_id"], spec["rating_run_id"]),
        )
    )
    scheduler_state = _read_json(
        runs.volume_path(
            mount,
            arena.rating_scheduler_state_ref(
                spec["tournament_id"],
                spec["rating_run_id"],
            ),
        )
    )
    rated_pair_count = int(snapshot.get("rated_pair_count") or 0)
    return arena._to_plain(
        {
            "round_id": round_id,
            "round_index": int(round_index),
            "snapshot": _slim_rating_snapshot(dict(snapshot)),
            "pair_history": pair_history if isinstance(pair_history, Mapping) else None,
            "scheduler_state": (scheduler_state if isinstance(scheduler_state, Mapping) else None),
            "pair_count": rated_pair_count,
            "game_count": int(snapshot.get("game_count") or 0),
            "work_summary": dict(work_summary or {}),
            "rated_pair_count": rated_pair_count,
            "resumed_existing_round": bool(resumed_existing_round),
        }
    )


def _drop_default_intake_drain_rating_overrides(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "continue_from_latest": False,
        "games_per_pair": arena.DEFAULT_GAMES_PER_PAIR,
        "games_per_shard": arena.DEFAULT_GAMES_PER_SHARD,
        "reuse_policies_per_shard": arena.DEFAULT_REUSE_POLICIES_PER_SHARD,
        "round_count": arena.DEFAULT_RATING_ROUND_COUNT,
        "pairs_per_round": None,
        "placement_min_games": None,
        "placement_min_opponents": 20,
        "pair_selection": arena.DEFAULT_RATING_PAIR_SELECTION,
        "initial_rating": arena.DEFAULT_RATING_INITIAL_RATING,
        "active_pool_limit": arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT,
        "stop_when_stable": False,
        "seed": 0,
        "max_steps": arena.DEFAULT_MAX_STEPS,
        "decision_ms": arena.DEFAULT_DECISION_MS,
        "decision_source_frames": arena.DEFAULT_DECISION_SOURCE_FRAMES,
        "source_physics_step_ms": arena.DEFAULT_SOURCE_PHYSICS_STEP_MS,
        "policy_mode": arena.POLICY_MODE_EVAL,
        "collect_temperature": arena.DEFAULT_COLLECT_TEMPERATURE,
        "collect_epsilon": arena.DEFAULT_COLLECT_EPSILON,
        "policy_trail_render_mode": None,
        "policy_bonus_render_mode": None,
        "num_simulations": arena.DEFAULT_NUM_SIMULATIONS,
        "save_gif": arena.DEFAULT_SAVE_GIF,
        "gif_sample_games_per_pair": arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR,
        "gif_sample_strategy": arena.DEFAULT_GIF_SAMPLE_STRATEGY,
    }
    cleaned = dict(payload)
    for key, default in defaults.items():
        if key not in cleaned:
            continue
        value = cleaned[key]
        if key == "decision_source_frames" and value in (None, "", 0, "0", default):
            cleaned.pop(key, None)
            continue
        if value in (None, "", 0, "0") and default is None:
            cleaned.pop(key, None)
        elif value == default:
            cleaned.pop(key, None)
    return cleaned


def _enqueue_checkpoint_events(
    *,
    manifest: Mapping[str, Any],
    checkpoint_refs: Sequence[str],
    reason: str,
) -> dict[str, Any]:
    partition = str(manifest["queue_partition"])
    enqueued = []
    for checkpoint_ref in checkpoint_refs:
        event_id = arena._short_hash(
            f"{manifest['tournament_id']}:{manifest['rating_run_id']}:{checkpoint_ref}",
            length=16,
        )
        event = {
            "schema_id": "curvyzero_curvytron_checkpoint_intake_event/v0",
            "event_id": event_id,
            "event_type": "checkpoint_seen",
            "reason": reason,
            "created_at": runs.utc_timestamp(),
            "tournament_id": manifest["tournament_id"],
            "rating_run_id": manifest["rating_run_id"],
            "checkpoint_ref": checkpoint_ref,
        }
        checkpoint_intake_queue.put(
            arena._to_plain(event),
            block=False,
            partition=partition,
            partition_ttl=DEFAULT_CHECKPOINT_INTAKE_QUEUE_TTL_SECONDS,
        )
        _append_tournament_lineage_event(
            stage="checkpoint_intake_enqueued",
            tournament_id=str(manifest["tournament_id"]),
            rating_run_id=str(manifest["rating_run_id"]),
            reason=reason,
            event_id=event_id,
            queue_partition=partition,
            checkpoint_ref=checkpoint_ref,
        )
        enqueued.append(event)
    return {"partition": partition, "enqueued_count": len(enqueued), "events": enqueued}


def _intake_rating_claim_key(
    manifest: Mapping[str, Any],
    *,
    continue_from_latest: bool = False,
) -> str:
    manifest_key = str(manifest["manifest_key"])
    claim_mode = "continue" if continue_from_latest else "fresh"
    if continue_from_latest:
        return f"rating_claim:{manifest_key}:mode-{claim_mode}:active"
    rating_refs = _intake_manifest_rating_checkpoint_refs(
        manifest,
        continue_from_latest=continue_from_latest,
    )
    return (
        f"rating_claim:{manifest_key}:mode-{claim_mode}:"
        f"pool-{_checkpoint_ref_pool_hash(rating_refs)}"
    )


def _parse_intake_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _intake_rating_claim_is_stale(
    claim: Any,
    *,
    stale_after_seconds: int = DEFAULT_CHECKPOINT_INTAKE_CLAIM_STALE_SECONDS,
    now: datetime | None = None,
) -> bool:
    if not isinstance(claim, Mapping):
        return False
    created_at = _parse_intake_timestamp(claim.get("created_at"))
    if created_at is None:
        return False
    if now is None:
        now = datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    age_seconds = (now.astimezone(UTC) - created_at).total_seconds()
    return age_seconds >= max(0, int(stale_after_seconds))


def _intake_rating_claim_needs_pool_repair(
    claim: Any,
    *,
    pool_hash: str,
    checkpoint_count: int,
) -> bool:
    if not isinstance(claim, Mapping):
        return False
    claim_pool_hash = str(claim.get("pool_hash") or "").strip()
    if claim_pool_hash and claim_pool_hash != pool_hash:
        return True
    claim_checkpoint_count = claim.get("checkpoint_count")
    if claim_checkpoint_count in (None, ""):
        return True
    try:
        return int(claim_checkpoint_count) != int(checkpoint_count)
    except (TypeError, ValueError):
        return True


def _intake_queue_repair_refs(manifest: Mapping[str, Any]) -> list[str]:
    queued_refs = _clean_ref_set(manifest.get("queued_checkpoint_refs", []))
    if not queued_refs:
        return []
    rating_refs = _clean_ref_set(manifest.get("seen_checkpoint_refs", []))
    if not rating_refs:
        rating_refs = _clean_ref_set(manifest.get("checkpoint_refs", []))
    return sorted(queued_refs.intersection(rating_refs)) if rating_refs else sorted(queued_refs)


def _path_for_ref(ref: str | Path) -> Path:
    return runs.volume_path(TOURNAMENT_MOUNT, arena.validate_tournament_artifact_ref(ref))


def _write_tournament_marker_at(mount: Path, tournament_id: str) -> dict[str, Any]:
    ref = arena.tournament_marker_ref(tournament_id)
    payload = {
        "schema_id": "curvyzero_curvytron_tournament_browser_marker/v0",
        "artifact_volume_name": TOURNAMENT_VOLUME_NAME,
        "tournament_id": runs.clean_id(tournament_id, label="tournament_id"),
        "created_at": runs.utc_timestamp(),
    }
    return arena.write_json_artifact(mount, ref, payload)


def _write_tournament_marker(tournament_id: str) -> dict[str, Any]:
    return _write_tournament_marker_at(TOURNAMENT_MOUNT, tournament_id)


def _game_count_from_shard_id(shard_id: str, *, default: int) -> int:
    parts = str(shard_id or "").split("-games-", 1)
    if len(parts) != 2:
        return max(0, int(default))
    bounds = parts[1].split("-", 1)
    if len(bounds) != 2:
        return max(0, int(default))
    try:
        start = int(bounds[0])
        end = int(bounds[1])
    except ValueError:
        return max(0, int(default))
    if end < start:
        return max(0, int(default))
    return end - start + 1


def _list_tournament_visibility_rows(mount: Path) -> list[dict[str, Any]]:
    base = runs.volume_path(mount, arena.TOURNAMENT_BASE_REF)
    if not base.exists():
        return []
    rows = []
    for root in sorted(base.iterdir(), key=lambda path: path.name):
        if not root.is_dir():
            continue
        tournament_id = root.name
        marker = root / arena.TOURNAMENT_RUN_MARKER_FILENAME
        manifest = _read_json(root / "tournament.json")
        complete = _read_json(root / "complete.json")
        rating_root = root / "ratings"
        rating_run_count = (
            sum(1 for path in rating_root.iterdir() if path.is_dir()) if rating_root.exists() else 0
        )
        candidates = [
            path
            for path in (
                marker,
                root / "complete.json",
                root / "tournament.json",
                root / "battle_index.json",
            )
            if path.exists()
        ]
        updated_ts = max(
            (path.stat().st_mtime for path in candidates), default=root.stat().st_mtime
        )
        rows.append(
            {
                "tournament_id": tournament_id,
                "visible": marker.exists(),
                "marker_ref": arena.tournament_marker_ref(tournament_id).as_posix(),
                "status": complete.get("status") or manifest.get("status"),
                "pair_count": complete.get("pair_count") or manifest.get("pair_count"),
                "checkpoint_count": complete.get("checkpoint_count")
                or manifest.get("checkpoint_count"),
                "rating_run_count": rating_run_count,
                "updated_ts": updated_ts,
                "updated_at": complete.get("ended_at") or manifest.get("updated_at"),
            }
        )
    rows.sort(key=lambda row: (-float(row.get("updated_ts") or 0), row["tournament_id"]))
    return rows


def _update_tournament_visibility(
    mount: Path,
    *,
    action: str = "list",
    tournament_ids: Sequence[str] | str | None = None,
    keep_tournament_ids: Sequence[str] | str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    def parse_ids(value: Sequence[str] | str | None) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            raw_values = [item.strip() for item in value.split(",")]
        else:
            raw_values = [str(item).strip() for item in value]
        return {runs.clean_id(item, label="tournament_id") for item in raw_values if item}

    clean_action = str(action or "list")
    if clean_action not in {"list", "hide", "show", "hide_except"}:
        raise ValueError("visibility action must be list, hide, show, or hide_except")
    before = _list_tournament_visibility_rows(mount)
    visible_before = {row["tournament_id"] for row in before if row.get("visible")}
    requested = parse_ids(tournament_ids)
    keep = parse_ids(keep_tournament_ids)
    if clean_action == "hide_except":
        targets = visible_before - keep
    else:
        targets = requested
    if clean_action in {"hide", "show"} and not targets:
        raise ValueError(f"visibility action {clean_action!r} needs tournament_ids")

    changes = []
    for tournament_id in sorted(targets):
        marker_ref = arena.tournament_marker_ref(tournament_id)
        marker_path = runs.volume_path(mount, marker_ref)
        existed = marker_path.exists()
        changed = False
        if clean_action in {"hide", "hide_except"}:
            changed = existed
            if changed and not dry_run:
                marker_path.unlink()
        elif clean_action == "show":
            changed = not existed
            if changed and not dry_run:
                _write_tournament_marker_at(mount, tournament_id)
        changes.append(
            {
                "tournament_id": tournament_id,
                "action": clean_action,
                "marker_ref": marker_ref.as_posix(),
                "was_visible": existed,
                "changed": changed,
            }
        )
    after = _list_tournament_visibility_rows(mount) if not dry_run else before
    return {
        "schema_id": "curvyzero_curvytron_tournament_visibility/v0",
        "action": clean_action,
        "dry_run": bool(dry_run),
        "visible_before_count": len(visible_before),
        "changed_count": sum(1 for row in changes if row.get("changed")),
        "changes": changes,
        "rows": after,
    }


def _write_tournament_manifest(spec: Mapping[str, Any], *, status: str) -> dict[str, Any]:
    tournament_id = runs.clean_id(str(spec["tournament_id"]), label="tournament_id")
    ref = arena.tournament_manifest_ref(tournament_id)
    payload = {
        "schema_id": arena.TOURNAMENT_SCHEMA_ID,
        "app_name": APP_NAME,
        "artifact_volume_name": TOURNAMENT_VOLUME_NAME,
        "checkpoint_volume_name": CHECKPOINT_VOLUME_NAME,
        "tournament_id": tournament_id,
        "status": status,
        "updated_at": runs.utc_timestamp(),
        "spec": arena._to_plain(dict(spec)),
    }
    return arena.write_json_artifact(TOURNAMENT_MOUNT, ref, payload)


def _write_rating_config(spec: Mapping[str, Any]) -> dict[str, Any]:
    rating_spec = arena.normalize_rating_spec(spec)
    pool_hash = arena.rating_pool_hash(rating_spec["checkpoints"])
    return arena.write_json_artifact(
        TOURNAMENT_MOUNT,
        arena.rating_config_ref(
            rating_spec["tournament_id"],
            rating_spec["rating_run_id"],
        ),
        {
            "schema_id": arena.RATING_CONFIG_SCHEMA_ID,
            "app_name": APP_NAME,
            "artifact_volume_name": TOURNAMENT_VOLUME_NAME,
            "checkpoint_volume_name": CHECKPOINT_VOLUME_NAME,
            "created_at": runs.utc_timestamp(),
            "pool_hash": pool_hash,
            "roster_hash": pool_hash,
            "context_hash": arena.rating_context_hash(rating_spec),
            "checkpoint_roster": arena.rating_roster_by_checkpoint(rating_spec["checkpoints"]),
            "rating_spec": arena._to_plain(rating_spec),
        },
    )


def _rating_loop_start_state(
    mount: Path,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    rating_spec = arena.normalize_rating_spec(spec)
    latest_snapshot: dict[str, Any] | None = None
    start_round_index = 0
    latest_round_index = -1
    if bool(rating_spec.get("continue_from_latest", False)):
        latest = _read_json(
            runs.volume_path(
                mount,
                arena.rating_latest_ref(
                    rating_spec["tournament_id"],
                    rating_spec["rating_run_id"],
                ),
            )
        )
        if latest:
            arena._validate_rating_state_compatibility(
                latest,
                expected_pool_hash=arena.rating_pool_hash(rating_spec["checkpoints"]),
                expected_context_hash=arena.rating_context_hash(rating_spec),
                expected_roster=arena.rating_roster_by_checkpoint(rating_spec["checkpoints"]),
                label="latest snapshot",
            )
            latest_snapshot = dict(latest)
            raw_latest_round_index = latest_snapshot.get("round_index")
            latest_round_index = (
                int(raw_latest_round_index) if raw_latest_round_index not in (None, "") else -1
            )
            start_round_index = latest_round_index + 1
        skipped_round_index = _highest_skipped_rating_round_index_after(
            mount,
            tournament_id=rating_spec["tournament_id"],
            rating_run_id=rating_spec["rating_run_id"],
            after_round_index=latest_round_index,
        )
        start_round_index = max(start_round_index, skipped_round_index + 1)

    previous_pair_history = (
        _read_json(
            runs.volume_path(
                mount,
                arena.rating_pair_history_ref(
                    rating_spec["tournament_id"],
                    rating_spec["rating_run_id"],
                ),
            )
        )
        or None
    )
    scheduler_state = (
        _read_json(
            runs.volume_path(
                mount,
                arena.rating_scheduler_state_ref(
                    rating_spec["tournament_id"],
                    rating_spec["rating_run_id"],
                ),
            )
        )
        or None
    )
    return {
        "previous_snapshot": latest_snapshot,
        "previous_pair_history": previous_pair_history,
        "scheduler_state": scheduler_state,
        "start_round_index": start_round_index,
        "continued_from_latest": latest_snapshot is not None,
        "latest_ref": arena.rating_latest_ref(
            rating_spec["tournament_id"],
            rating_spec["rating_run_id"],
        ).as_posix(),
    }


def _compact_battle_index_row(
    summary: Mapping[str, Any],
    *,
    updated_ts: float | None = None,
) -> dict[str, Any]:
    players = summary.get("players")
    checkpoint_ids = []
    if isinstance(players, Sequence) and not isinstance(players, (str, bytes)):
        checkpoint_ids = [
            str(player["checkpoint_id"])
            for player in players
            if isinstance(player, Mapping) and player.get("checkpoint_id")
        ]
    row = {
        "tournament_id": summary.get("tournament_id"),
        "rating_run_id": summary.get("rating_run_id"),
        "round_id": summary.get("round_id"),
        "round_index": summary.get("round_index"),
        "battle_id": summary.get("battle_id"),
        "pair_index": summary.get("pair_index"),
        "players": players,
        "checkpoint_ids": checkpoint_ids,
        "tally": summary.get("tally"),
        "ok": summary.get("ok"),
        "summary_ref": summary.get("summary_ref"),
        "first_gif_ref": summary.get("first_gif_ref"),
        "sample_gif_refs": summary.get("sample_gif_refs"),
        "shard_summary_refs": summary.get("shard_summary_refs"),
        "shard_summary_ref_count": summary.get("shard_summary_ref_count"),
        "updated_at": summary.get("ended_at") or summary.get("started_at"),
        "updated_ts": float(updated_ts if updated_ts is not None else time.time()),
    }
    for key in (
        "pair_key",
        "schedule_reason",
        "schedule_priority",
        "scheduled_round_index",
        "schedule",
    ):
        if key in summary:
            row[key] = arena._to_plain(summary[key])
    return row


def _write_battle_index(
    tournament_id: str,
    pair_results: Sequence[Mapping[str, Any]],
    *,
    mount: Path = TOURNAMENT_MOUNT,
) -> dict[str, Any]:
    clean_id = runs.clean_id(tournament_id, label="tournament_id")
    ref = arena.tournament_battle_index_ref(clean_id)
    path = runs.volume_path(mount, ref)
    existing = _read_json(path)
    rows_by_battle: dict[str, dict[str, Any]] = {}
    existing_rows = existing.get("rows")
    if isinstance(existing_rows, list):
        for row in existing_rows:
            if isinstance(row, Mapping) and row.get("battle_id"):
                rows_by_battle[str(row["battle_id"])] = dict(row)
    now = time.time()
    for offset, summary in enumerate(pair_results):
        battle_id = summary.get("battle_id")
        if not battle_id:
            continue
        rows_by_battle[str(battle_id)] = _compact_battle_index_row(
            summary,
            updated_ts=now + float(offset) / 1_000_000.0,
        )
    rows = sorted(
        rows_by_battle.values(),
        key=lambda row: (
            -float(row.get("updated_ts") or 0.0),
            str(row.get("battle_id") or ""),
        ),
    )
    payload = {
        "schema_id": "curvyzero_curvytron_checkpoint_tournament_battle_index/v0",
        "tournament_id": clean_id,
        "updated_at": runs.utc_timestamp(),
        "total": len(rows),
        "rows": arena._to_plain(rows),
    }
    write_summary = arena.write_json_artifact(mount, ref, payload)
    payload["ref"] = write_summary.get("ref")
    checkpoint_index_count = 0
    rows_by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for checkpoint_id in row.get("checkpoint_ids") or []:
            if checkpoint_id:
                rows_by_checkpoint[str(checkpoint_id)].append(row)
    for checkpoint_id, checkpoint_rows in rows_by_checkpoint.items():
        checkpoint_ref = arena.tournament_checkpoint_battle_index_ref(
            clean_id,
            checkpoint_id,
        )
        checkpoint_payload = {
            **payload,
            "ref": checkpoint_ref.as_posix(),
            "checkpoint_id": checkpoint_id,
            "total": len(checkpoint_rows),
            "rows": arena._to_plain(checkpoint_rows),
        }
        checkpoint_write = arena.write_json_artifact(
            mount,
            checkpoint_ref,
            checkpoint_payload,
        )
        checkpoint_payload["ref"] = checkpoint_write.get("ref")
        checkpoint_index_count += 1
    payload["checkpoint_index_count"] = checkpoint_index_count
    return arena._to_plain(payload)


def _slim_rating_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    slim = dict(snapshot)
    slim.pop("pair_rating_results", None)
    return arena._to_plain(slim)


def _slim_provisional_rating_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    slim = _slim_rating_snapshot(snapshot)
    slim.pop("live_pair_results", None)
    return arena._to_plain(slim)


def _write_provisional_rating_artifacts(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    live_pair_results = snapshot.get("live_pair_results")
    battle_index: dict[str, Any] = {}
    if isinstance(live_pair_results, Sequence) and not isinstance(
        live_pair_results,
        (str, bytes),
    ):
        battle_index = _write_battle_index(
            tournament_id,
            [dict(row) for row in live_pair_results if isinstance(row, Mapping)],
            mount=mount,
        )
    slim = dict(_slim_provisional_rating_snapshot(snapshot))
    slim["battle_index_ref"] = arena.tournament_battle_index_ref(tournament_id).as_posix()
    arena.write_json_artifact(
        mount,
        _rating_provisional_latest_ref(tournament_id, rating_run_id),
        slim,
    )
    return arena._to_plain({"snapshot": slim, "battle_index": battle_index})


def _read_rating_round_input(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
) -> dict[str, Any]:
    ref = arena.rating_round_input_ref(tournament_id, rating_run_id, round_id)
    payload = _read_json(runs.volume_path(mount, ref))
    if not payload:
        raise FileNotFoundError(f"rating round input not found: {ref.as_posix()}")
    pair_specs = payload.get("pair_specs")
    if not isinstance(pair_specs, list):
        raise ValueError(f"rating round input has no pair_specs list: {ref.as_posix()}")
    return payload


def _iter_rating_game_summaries(
    mount: Path,
    *,
    tournament_id: str,
    expected_battle_ids: set[str],
    load_payloads: bool = True,
) -> list[tuple[Path, dict[str, Any]]]:
    root = runs.volume_path(mount, arena.tournament_root_ref(tournament_id)) / "battles"
    rows: list[tuple[Path, dict[str, Any]]] = []
    if not root.exists():
        return rows
    if not load_payloads:
        for battle_dir in root.iterdir():
            if not battle_dir.is_dir() or battle_dir.name not in expected_battle_ids:
                continue
            games_root = battle_dir / "games"
            if not games_root.exists():
                continue
            for game_dir in games_root.iterdir():
                if not game_dir.is_dir():
                    continue
                path = game_dir / "summary.json"
                if not path.is_file():
                    continue
                rows.append(
                    (
                        path,
                        {
                            "battle_id": battle_dir.name,
                            "game_id": game_dir.name,
                            "summary_ref": runs.file_ref(path, mount=mount),
                        },
                    )
                )
        return rows
    for battle_id in sorted(expected_battle_ids):
        battle_root = root / battle_id / "games"
        for path in battle_root.glob("*/summary.json") if battle_root.exists() else []:
            if load_payloads:
                payload = _read_json(path)
                if not payload:
                    continue
            else:
                payload = {
                    "battle_id": battle_id,
                    "game_id": path.parent.name,
                    "summary_ref": runs.file_ref(path, mount=mount),
                }
            rows.append((path, payload))
    return rows


def _extract_top_level_json_field(
    text: str,
    *,
    key: str,
    next_key: str,
) -> Any | None:
    marker = f'\n  "{key}": '
    start = text.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end_marker = f',\n  "{next_key}":'
    end = text.find(end_marker, start)
    if end < 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def _extract_top_level_scalar(text: str, key: str) -> Any | None:
    match = re.search(rf'\n  "{re.escape(key)}": ([^,\n]+)', text)
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _players_for_game_seats_from_pair(
    pair: Mapping[str, Any],
    seat_order: Mapping[str, Any],
) -> list[dict[str, Any]]:
    logical_players = pair.get("players")
    if not isinstance(logical_players, Sequence) or isinstance(logical_players, (str, bytes)):
        return []
    seat_to_logical = seat_order.get("seat_to_logical_index")
    if not isinstance(seat_to_logical, Sequence) or isinstance(seat_to_logical, (str, bytes)):
        return [dict(player) for player in logical_players if isinstance(player, Mapping)]
    players: list[dict[str, Any]] = []
    for seat, logical_index in enumerate(seat_to_logical):
        try:
            player = dict(logical_players[int(logical_index)])
        except Exception:
            return [dict(item) for item in logical_players if isinstance(item, Mapping)]
        player["seat"] = int(seat)
        players.append(player)
    return players


def _read_compact_game_summary(
    path: Path,
    *,
    pair: Mapping[str, Any],
    mount: Path,
) -> dict[str, Any]:
    text = path.read_text()
    score = _extract_top_level_json_field(text, key="score", next_key="seat_order")
    seat_order = _extract_top_level_json_field(
        text,
        key="seat_order",
        next_key="seat_order_mode",
    )
    if not isinstance(score, Mapping) or not isinstance(seat_order, Mapping):
        payload = _read_json(path)
        return arena._compact_game_result(payload) if payload else {}
    battle_id = str(pair.get("battle_id") or path.parents[2].name)
    game_id = str(_extract_top_level_scalar(text, "game_id") or path.parent.name)
    game_index_raw = _extract_top_level_scalar(text, "game_index")
    seed_raw = _extract_top_level_scalar(text, "seed")
    ok = _extract_top_level_scalar(text, "ok")
    return {
        "ok": bool(ok),
        "tournament_id": pair.get("tournament_id"),
        "battle_id": battle_id,
        "pair_index": pair.get("pair_index"),
        "game_id": game_id,
        "game_index": int(game_index_raw or 0),
        "seed": int(seed_raw or 0),
        "players": _players_for_game_seats_from_pair(pair, seat_order),
        "battle_players": pair.get("players"),
        "seat_order": dict(seat_order),
        "seat_order_mode": seat_order.get("mode") or pair.get("seat_order_mode"),
        "score": dict(score),
        "physical_steps": score.get("physical_steps"),
        "gif_ref": None,
        "summary_ref": runs.file_ref(path, mount=mount),
    }


def _compact_rating_game_summaries_for_pairs(
    mount: Path,
    *,
    tournament_id: str,
    pair_specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    root = runs.volume_path(mount, arena.tournament_root_ref(tournament_id)) / "battles"
    rows: list[dict[str, Any]] = []
    for pair in pair_specs:
        battle_id = str(pair.get("battle_id") or "")
        if not battle_id:
            continue
        games_root = root / battle_id / "games"
        if not games_root.exists():
            continue
        for path in games_root.glob("*/summary.json"):
            row = _read_compact_game_summary(path, pair=pair, mount=mount)
            if row:
                rows.append(row)
    return rows


def _rating_round_progress_payload(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
    game_results: Sequence[Mapping[str, Any]] | None = None,
    load_summaries: bool = True,
    pair_only: bool = False,
    count_game_summaries: bool = False,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    clean_tournament_id = runs.clean_id(tournament_id, label="tournament_id")
    clean_rating_run_id = runs.clean_id(rating_run_id, label="rating_run_id")
    clean_round_id = runs.clean_id(round_id, label="round_id")
    input_payload = _read_rating_round_input(
        mount,
        tournament_id=clean_tournament_id,
        rating_run_id=clean_rating_run_id,
        round_id=clean_round_id,
    )
    pair_specs = [
        dict(pair)
        for pair in input_payload.get("pair_specs", [])
        if isinstance(pair, Mapping) and pair.get("battle_id")
    ]
    expected_by_battle = {
        str(pair["battle_id"]): int(pair.get("games_per_pair") or 0) for pair in pair_specs
    }
    game_count = int(input_payload.get("game_count") or sum(expected_by_battle.values()))
    latest_snapshot = _read_json(
        runs.volume_path(
            mount,
            arena.rating_latest_ref(clean_tournament_id, clean_rating_run_id),
        )
    )
    if (
        pair_only
        and latest_snapshot
        and str(latest_snapshot.get("round_id") or clean_round_id) == clean_round_id
    ):
        payload = {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "app_name": APP_NAME,
            "artifact_volume_name": TOURNAMENT_VOLUME_NAME,
            "checkpoint_volume_name": CHECKPOINT_VOLUME_NAME,
            "tournament_id": clean_tournament_id,
            "rating_run_id": clean_rating_run_id,
            "round_id": clean_round_id,
            "round_index": int(input_payload.get("round_index", 0) or 0),
            "updated_at": runs.utc_timestamp(),
            "updated_ts": time.time(),
            "input_ref": arena.rating_round_input_ref(
                clean_tournament_id,
                clean_rating_run_id,
                clean_round_id,
            ).as_posix(),
            "progress_ref": arena.rating_progress_ref(
                clean_tournament_id,
                clean_rating_run_id,
            ).as_posix(),
            "round_progress_ref": arena.rating_round_progress_ref(
                clean_tournament_id,
                clean_rating_run_id,
                clean_round_id,
            ).as_posix(),
            "latest_ref": arena.rating_latest_ref(
                clean_tournament_id,
                clean_rating_run_id,
            ).as_posix(),
            "pair_count": len(pair_specs),
            "game_count": game_count,
            "completed_game_count": game_count,
            "estimated_seen_game_count": game_count,
            "ok_game_count": 0,
            "failed_game_count": 0,
            "unknown_result_count": 0,
            "result_counts_known": False,
            "count_basis": "latest_snapshot",
            "counted_game_summary_files": False,
            "started_pair_count": len(pair_specs),
            "partial_pair_count": 0,
            "completed_pair_count": len(pair_specs),
            "completion_fraction": 1.0 if game_count else 0.0,
            "estimated_completion_fraction": 1.0 if game_count else 0.0,
            "max_started_pair_index": len(pair_specs) - 1 if pair_specs else None,
            "max_completed_pair_index": len(pair_specs) - 1 if pair_specs else None,
            "recent_started_pairs": [],
            "ratings_written": True,
            "status": "complete",
            "phase": "ratings_written",
            "rated_pair_count": latest_snapshot.get("rated_pair_count"),
            "max_abs_delta": latest_snapshot.get("max_abs_delta"),
            "stable": latest_snapshot.get("stable"),
        }
        return arena._to_plain(payload), {}
    pair_index_by_battle = {
        str(pair["battle_id"]): int(pair.get("pair_index", 0) or 0) for pair in pair_specs
    }
    shard_size_by_battle = {
        str(pair["battle_id"]): max(
            1,
            int(pair.get("games_per_shard") or arena.DEFAULT_GAMES_PER_SHARD),
        )
        for pair in pair_specs
    }
    should_count_game_summaries = bool(
        count_game_summaries or any(size <= 1 for size in shard_size_by_battle.values())
    )
    seen_pair_dir_ids: set[str] = set()
    seen_shard_game_counts: dict[str, int] = {}
    latest_result_ts: float | None = None
    counted_game_summary_files = False
    scan_errors: list[dict[str, str]] = []
    if pair_only and game_results is None:
        root = runs.volume_path(mount, arena.tournament_root_ref(clean_tournament_id)) / "battles"
        try:
            root_exists = root.exists()
        except OSError as exc:
            root_exists = False
            scan_errors.append(
                {
                    "path": runs.file_ref(root, mount=mount),
                    "operation": "exists",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        if root_exists:
            expected_battle_ids = set(expected_by_battle)
            for battle_id in expected_battle_ids:
                battle_root = root / battle_id
                try:
                    battle_root_is_dir = battle_root.is_dir()
                except OSError as exc:
                    battle_root_is_dir = False
                    scan_errors.append(
                        {
                            "path": runs.file_ref(battle_root, mount=mount),
                            "operation": "is_dir",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                if not battle_root_is_dir:
                    continue
                seen_pair_dir_ids.add(battle_id)
                try:
                    shard_paths = list(battle_root.glob("shards/*/summary.json"))
                except OSError as exc:
                    shard_paths = []
                    scan_errors.append(
                        {
                            "path": runs.file_ref(battle_root / "shards", mount=mount),
                            "operation": "glob_shard_summaries",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                for path in shard_paths:
                    path_mtime = _path_mtime_seconds(path)
                    if path_mtime is not None:
                        latest_result_ts = (
                            path_mtime
                            if latest_result_ts is None
                            else max(latest_result_ts, path_mtime)
                        )
                    expected = int(expected_by_battle.get(battle_id) or 0)
                    shard_size = max(1, int(shard_size_by_battle.get(battle_id) or 1))
                    seen_shard_game_counts[battle_id] = min(
                        expected,
                        seen_shard_game_counts.get(battle_id, 0)
                        + _game_count_from_shard_id(
                            path.parent.name,
                            default=shard_size,
                        ),
                    )
            if should_count_game_summaries:
                for battle_id in expected_battle_ids:
                    battle_root = root / battle_id
                    try:
                        battle_root_is_dir = battle_root.is_dir()
                    except OSError as exc:
                        battle_root_is_dir = False
                        scan_errors.append(
                            {
                                "path": runs.file_ref(battle_root, mount=mount),
                                "operation": "is_dir",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                    if not battle_root_is_dir:
                        continue
                    try:
                        game_summary_paths = list(battle_root.glob("games/*/summary.json"))
                    except OSError as exc:
                        game_summary_paths = []
                        scan_errors.append(
                            {
                                "path": runs.file_ref(battle_root / "games", mount=mount),
                                "operation": "glob_game_summaries",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                    )
                    game_summary_count = 0
                    for path in game_summary_paths:
                        path_mtime = _path_mtime_seconds(path)
                        if path_mtime is not None:
                            latest_result_ts = (
                                path_mtime
                                if latest_result_ts is None
                                else max(latest_result_ts, path_mtime)
                            )
                        game_summary_count += 1
                    if game_summary_count <= 0:
                        continue
                    expected = int(expected_by_battle.get(battle_id) or 0)
                    seen_shard_game_counts[battle_id] = min(
                        expected,
                        max(
                            seen_shard_game_counts.get(battle_id, 0),
                            int(game_summary_count),
                        ),
                    )
                counted_game_summary_files = True
        summaries = []
    elif game_results is None:
        summaries: Sequence[tuple[Path | None, dict[str, Any]]] = _iter_rating_game_summaries(
            mount,
            tournament_id=clean_tournament_id,
            expected_battle_ids=set(expected_by_battle),
            load_payloads=load_summaries,
        )
    else:
        summaries = [(None, dict(result)) for result in game_results if isinstance(result, Mapping)]
    games_by_battle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_games: set[tuple[str, str]] = set()
    ok_game_count = 0
    failed_game_count = 0
    unknown_result_count = 0
    for path, payload in summaries:
        battle_id = str(
            payload.get("battle_id") or (path.parents[2].name if path is not None else "")
        )
        game_id = str(payload.get("game_id") or (path.parent.name if path is not None else ""))
        if not battle_id or not game_id:
            continue
        key = (battle_id, game_id)
        if key in seen_games:
            continue
        seen_games.add(key)
        if path is not None:
            payload.setdefault("summary_ref", runs.file_ref(path, mount=mount))
            path_mtime = _path_mtime_seconds(path)
            if path_mtime is not None:
                latest_result_ts = (
                    path_mtime if latest_result_ts is None else max(latest_result_ts, path_mtime)
                )
        games_by_battle[battle_id].append(payload)
        if "ok" not in payload:
            unknown_result_count += 1
        elif payload.get("ok"):
            ok_game_count += 1
        else:
            failed_game_count += 1

    started_pairs = 0
    partial_pairs = 0
    completed_pairs = 0
    max_started_pair_index: int | None = None
    max_completed_pair_index: int | None = None
    pair_rows = []
    for pair in pair_specs:
        battle_id = str(pair["battle_id"])
        expected = int(expected_by_battle.get(battle_id) or 0)
        if pair_only:
            seen = min(expected, int(seen_shard_game_counts.get(battle_id, 0) or 0))
        else:
            seen = len(games_by_battle.get(battle_id, []))
        pair_index = int(pair_index_by_battle.get(battle_id, 0))
        pair_dir_seen = battle_id in seen_pair_dir_ids
        if seen > 0 or pair_dir_seen:
            started_pairs += 1
            max_started_pair_index = (
                pair_index
                if max_started_pair_index is None
                else max(max_started_pair_index, pair_index)
            )
        if expected > 0 and seen >= expected:
            completed_pairs += 1
            max_completed_pair_index = (
                pair_index
                if max_completed_pair_index is None
                else max(max_completed_pair_index, pair_index)
            )
        elif seen > 0:
            partial_pairs += 1
        if seen or pair_dir_seen:
            pair_rows.append(
                {
                    "battle_id": battle_id,
                    "pair_index": pair_index,
                    "seen_game_count": seen if seen else (None if pair_only else 0),
                    "expected_game_count": expected,
                    "complete": bool(expected > 0 and seen >= expected),
                }
            )

    completed_game_count = (
        sum(
            min(
                int(expected_by_battle.get(battle_id) or 0),
                int(seen_shard_game_counts.get(battle_id, 0) or 0),
            )
            for battle_id in expected_by_battle
        )
        if pair_only
        else len(seen_games)
    )
    estimated_seen_game_count = completed_game_count if pair_only else None
    progress_ref = arena.rating_progress_ref(
        clean_tournament_id,
        clean_rating_run_id,
    ).as_posix()
    round_progress_ref = arena.rating_round_progress_ref(
        clean_tournament_id,
        clean_rating_run_id,
        clean_round_id,
    ).as_posix()
    payload = {
        "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
        "app_name": APP_NAME,
        "artifact_volume_name": TOURNAMENT_VOLUME_NAME,
        "checkpoint_volume_name": CHECKPOINT_VOLUME_NAME,
        "tournament_id": clean_tournament_id,
        "rating_run_id": clean_rating_run_id,
        "round_id": clean_round_id,
        "round_index": int(input_payload.get("round_index", 0) or 0),
        "updated_at": runs.utc_timestamp(),
        "updated_ts": time.time(),
        "latest_result_ts": latest_result_ts,
        "input_ref": arena.rating_round_input_ref(
            clean_tournament_id,
            clean_rating_run_id,
            clean_round_id,
        ).as_posix(),
        "progress_ref": progress_ref,
        "round_progress_ref": round_progress_ref,
        "latest_ref": arena.rating_latest_ref(
            clean_tournament_id,
            clean_rating_run_id,
        ).as_posix(),
        "pair_count": len(pair_specs),
        "game_count": game_count,
        "completed_game_count": completed_game_count,
        "estimated_seen_game_count": estimated_seen_game_count,
        "ok_game_count": ok_game_count,
        "failed_game_count": failed_game_count,
        "unknown_result_count": unknown_result_count,
        "result_counts_known": bool((not pair_only) and unknown_result_count == 0),
        "count_basis": (
            "shard_and_game_summary_files"
            if pair_only and counted_game_summary_files
            else ("shard_summary_files" if pair_only else "summary_files")
        ),
        "counted_game_summary_files": counted_game_summary_files,
        "started_pair_count": started_pairs,
        "partial_pair_count": partial_pairs,
        "completed_pair_count": completed_pairs,
        "completion_fraction": (
            float(completed_game_count) / float(game_count) if game_count else 0.0
        ),
        "estimated_completion_fraction": (
            float(estimated_seen_game_count) / float(game_count)
            if pair_only and game_count
            else None
        ),
        "max_started_pair_index": max_started_pair_index,
        "max_completed_pair_index": max_completed_pair_index,
        "recent_started_pairs": sorted(
            pair_rows,
            key=lambda row: (-int(row["pair_index"]), str(row["battle_id"])),
        )[:25],
    }
    if scan_errors:
        payload["scan_errors"] = scan_errors
    payload["status"] = (
        "complete" if completed_game_count >= game_count and game_count else "running"
    )
    payload["phase"] = "all_games_seen" if payload["status"] == "complete" else "games_running"
    if latest_snapshot and str(latest_snapshot.get("round_id") or clean_round_id) == clean_round_id:
        payload["ratings_written"] = True
        payload["status"] = "complete"
        payload["phase"] = "ratings_written"
        payload["rated_pair_count"] = latest_snapshot.get("rated_pair_count")
        payload["max_abs_delta"] = latest_snapshot.get("max_abs_delta")
        payload["stable"] = latest_snapshot.get("stable")
        if pair_only:
            payload["completed_game_count"] = game_count
            payload["completed_pair_count"] = len(pair_specs)
            payload["partial_pair_count"] = 0
            payload["completion_fraction"] = 1.0 if game_count else 0.0
            payload["estimated_completion_fraction"] = 1.0 if game_count else 0.0
            payload["max_completed_pair_index"] = max_started_pair_index
            for row in payload["recent_started_pairs"]:
                row["complete"] = True
                row["seen_game_count"] = row["expected_game_count"]
    return arena._to_plain(payload), games_by_battle


def _sample_pair_specs_for_status(
    pair_specs: Sequence[Mapping[str, Any]],
    *,
    max_pairs: int,
) -> list[Mapping[str, Any]]:
    limit = max(0, int(max_pairs))
    if limit <= 0 or len(pair_specs) <= limit:
        return list(pair_specs)
    head_count = max(1, min(len(pair_specs), limit // 2))
    sampled_indices: list[int] = list(range(head_count))
    remaining = max(0, limit - len(sampled_indices))
    if remaining == 1:
        sampled_indices.append(len(pair_specs) - 1)
    elif remaining > 1:
        last_index = len(pair_specs) - 1
        sampled_indices.extend(
            int(round(index * last_index / float(remaining - 1)))
            for index in range(remaining)
        )
    seen: set[int] = set()
    sampled: list[Mapping[str, Any]] = []
    for index in sampled_indices:
        bounded = min(max(0, int(index)), len(pair_specs) - 1)
        if bounded in seen:
            continue
        seen.add(bounded)
        sampled.append(pair_specs[bounded])
    return sampled


def _rating_round_activity_probe(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
    max_pairs: int = 16,
    stop_after_first_output: bool = True,
    stale_after_seconds: int = DEFAULT_RATING_ROUND_STALE_SECONDS,
) -> dict[str, Any]:
    clean_tournament_id = runs.clean_id(tournament_id, label="tournament_id")
    clean_rating_run_id = runs.clean_id(rating_run_id, label="rating_run_id")
    clean_round_id = runs.clean_id(round_id, label="round_id")
    input_payload = _read_rating_round_input(
        mount,
        tournament_id=clean_tournament_id,
        rating_run_id=clean_rating_run_id,
        round_id=clean_round_id,
    )
    pair_specs = [
        dict(pair)
        for pair in input_payload.get("pair_specs", [])
        if isinstance(pair, Mapping) and pair.get("battle_id")
    ]
    sampled_pairs = _sample_pair_specs_for_status(pair_specs, max_pairs=max_pairs)
    root = runs.volume_path(mount, arena.tournament_root_ref(clean_tournament_id)) / "battles"
    latest_result_ts: float | None = None
    seen_summary_count = 0
    seen_pair_count = 0
    scan_errors: list[dict[str, str]] = []
    try:
        root_exists = root.exists()
    except OSError as exc:
        root_exists = False
        scan_errors.append(
            {
                "path": runs.file_ref(root, mount=mount),
                "operation": "exists",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    if root_exists:
        for pair in sampled_pairs:
            battle_id = str(pair["battle_id"])
            battle_root = root / battle_id
            try:
                battle_root_is_dir = battle_root.is_dir()
            except OSError as exc:
                battle_root_is_dir = False
                scan_errors.append(
                    {
                        "path": runs.file_ref(battle_root, mount=mount),
                        "operation": "is_dir",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            if not battle_root_is_dir:
                continue
            pair_has_output = False
            for glob_label, pattern in (
                ("glob_shard_summaries", "shards/*/summary.json"),
                ("glob_game_summaries", "games/*/summary.json"),
            ):
                try:
                    paths = list(battle_root.glob(pattern))
                except OSError as exc:
                    paths = []
                    scan_errors.append(
                        {
                            "path": runs.file_ref(battle_root, mount=mount),
                            "operation": glob_label,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                if paths:
                    pair_has_output = True
                seen_summary_count += len(paths)
                for path in paths:
                    path_mtime = _path_mtime_seconds(path)
                    if path_mtime is not None:
                        latest_result_ts = (
                            path_mtime
                            if latest_result_ts is None
                            else max(latest_result_ts, path_mtime)
                        )
            if pair_has_output:
                seen_pair_count += 1
                if stop_after_first_output:
                    break
    now_ts = time.time()
    latest_result_age_seconds = (
        max(0.0, now_ts - latest_result_ts) if latest_result_ts is not None else None
    )
    effective_stale_after_seconds = max(0, int(stale_after_seconds))
    return arena._to_plain(
        {
            "schema_id": "curvyzero_curvytron_rating_round_activity_probe/v0",
            "scan_output_progress": True,
            "skip": False,
            "reason": "activity_probe",
            "round_id": clean_round_id,
            "pair_count": len(pair_specs),
            "sampled_pair_count": len(sampled_pairs),
            "stopped_after_first_output": bool(
                stop_after_first_output and seen_summary_count > 0
            ),
            "seen_pair_count": seen_pair_count,
            "completed_game_count": seen_summary_count,
            "started_pair_count": seen_pair_count,
            "latest_result_ts": latest_result_ts,
            "latest_result_age_seconds": latest_result_age_seconds,
            "has_output": bool(seen_summary_count > 0 or latest_result_ts is not None),
            "progress_scan_error": scan_errors[0]["error"] if scan_errors else None,
            "scan_error_count": len(scan_errors),
            "progress_scan_error_blocks_skip": False,
            "stale_after_seconds": effective_stale_after_seconds,
            "is_stale": bool(
                latest_result_age_seconds is not None
                and latest_result_age_seconds >= effective_stale_after_seconds
            ),
        }
    )


def _pending_rating_progress(
    *,
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
    round_index: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
        "app_name": APP_NAME,
        "artifact_volume_name": TOURNAMENT_VOLUME_NAME,
        "checkpoint_volume_name": CHECKPOINT_VOLUME_NAME,
        "tournament_id": tournament_id,
        "rating_run_id": rating_run_id,
        "round_id": round_id,
        "round_index": int(round_index),
        "status": "pending",
        "phase": reason,
        "pair_count": 0,
        "game_count": 0,
        "completed_pair_count": 0,
        "completed_game_count": 0,
        "estimated_seen_game_count": 0,
        "completion_fraction": 0.0,
        "estimated_completion_fraction": 0.0,
        "updated_at": runs.utc_timestamp(),
        "updated_ts": time.time(),
        "input_ref": arena.rating_round_input_ref(
            tournament_id,
            rating_run_id,
            round_id,
        ).as_posix(),
        "progress_ref": arena.rating_progress_ref(
            tournament_id,
            rating_run_id,
        ).as_posix(),
        "round_progress_ref": arena.rating_round_progress_ref(
            tournament_id,
            rating_run_id,
            round_id,
        ).as_posix(),
        "latest_ref": arena.rating_latest_ref(
            tournament_id,
            rating_run_id,
        ).as_posix(),
    }


def _write_rating_progress(
    mount: Path,
    progress: Mapping[str, Any],
) -> dict[str, Any]:
    tournament_id = str(progress["tournament_id"])
    rating_run_id = str(progress["rating_run_id"])
    round_id = str(progress["round_id"])
    progress_round_index = _round_index_from_payload(progress)
    arena.write_json_artifact(
        mount,
        arena.rating_round_progress_ref(tournament_id, rating_run_id, round_id),
        progress,
    )
    latest_round_index = _latest_rating_round_index(
        mount,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    root_progress_round_index = _root_rating_progress_round_index(
        mount,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    if progress_round_index is not None and progress_round_index < max(
        latest_round_index, root_progress_round_index
    ):
        return {
            "ref": arena.rating_round_progress_ref(
                tournament_id,
                rating_run_id,
                round_id,
            ).as_posix(),
            "root_progress_write_skipped": True,
            "root_progress_write_skipped_reason": "newer_round_already_visible",
            "round_index": progress_round_index,
            "latest_round_index": latest_round_index,
            "root_progress_round_index": root_progress_round_index,
        }
    return arena.write_json_artifact(
        mount,
        arena.rating_progress_ref(tournament_id, rating_run_id),
        progress,
    )


def _is_empty_waiting_rating_progress(progress: Mapping[str, Any]) -> bool:
    return (
        str(progress.get("status") or "") == "pending"
        and str(progress.get("phase") or "") == "waiting_for_round_input"
        and int(progress.get("pair_count") or 0) == 0
        and int(progress.get("game_count") or 0) == 0
        and int(progress.get("completed_pair_count") or 0) == 0
        and int(progress.get("completed_game_count") or 0) == 0
    )


def _rating_round_existing_blocking_artifacts(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
) -> list[Path]:
    artifacts: list[Path] = []
    input_ref = arena.rating_round_input_ref(tournament_id, rating_run_id, round_id)
    input_path = runs.volume_path(mount, input_ref)
    if input_path.exists():
        artifacts.append(input_ref)
    progress_ref = arena.rating_round_progress_ref(tournament_id, rating_run_id, round_id)
    progress_path = runs.volume_path(mount, progress_ref)
    if progress_path.exists():
        progress = _read_json(progress_path)
        if not _is_empty_waiting_rating_progress(progress):
            artifacts.append(progress_ref)
    return artifacts


def _rating_scheduler_state_payload(
    *,
    spec: Mapping[str, Any],
    round_id: str,
    round_index: int,
    pair_specs: Sequence[Mapping[str, Any]],
    pair_history: Mapping[str, Any],
) -> dict[str, Any]:
    reason_counts: dict[str, int] = defaultdict(int)
    prior_battle_count = 0
    pair_keys = []
    for pair in pair_specs:
        reason = str(pair.get("schedule_reason") or "unspecified")
        reason_counts[reason] += 1
        if pair.get("pair_key"):
            pair_keys.append(str(pair["pair_key"]))
        schedule = pair.get("schedule") if isinstance(pair.get("schedule"), Mapping) else {}
        prior_battle_count += int(schedule.get("prior_battle_count") or 0)
    pool_hash = arena.rating_pool_hash(spec.get("checkpoints") or [])
    context_hash = arena.rating_context_hash(spec)
    checkpoint_roster = arena.rating_roster_by_checkpoint(spec.get("checkpoints") or [])
    return {
        "schema_id": arena.RATING_SCHEDULER_STATE_SCHEMA_ID,
        "app_name": APP_NAME,
        "artifact_volume_name": TOURNAMENT_VOLUME_NAME,
        "checkpoint_volume_name": CHECKPOINT_VOLUME_NAME,
        "tournament_id": spec["tournament_id"],
        "rating_run_id": spec["rating_run_id"],
        "pool_hash": pool_hash,
        "roster_hash": pool_hash,
        "context_hash": context_hash,
        "checkpoint_roster": checkpoint_roster,
        "round_id": round_id,
        "round_index": int(round_index),
        "updated_at": runs.utc_timestamp(),
        "pair_selection": spec.get("pair_selection"),
        "pairs_per_round": spec.get("pairs_per_round"),
        "pair_count": len(pair_specs),
        "game_count": sum(int(pair.get("games_per_pair") or 0) for pair in pair_specs),
        "schedule_reason_counts": dict(sorted(reason_counts.items())),
        "prior_battle_count": int(prior_battle_count),
        "scheduled_pair_key_count": len(set(pair_keys)),
        "pair_history_row_count": len(pair_history.get("rows") or []),
        "pair_history_ref": arena.rating_pair_history_ref(
            spec["tournament_id"],
            spec["rating_run_id"],
        ).as_posix(),
        "latest_ref": arena.rating_latest_ref(
            spec["tournament_id"],
            spec["rating_run_id"],
        ).as_posix(),
    }


def _write_rating_scheduler_state(
    mount: Path,
    *,
    spec: Mapping[str, Any],
    round_id: str,
    round_index: int,
    pair_specs: Sequence[Mapping[str, Any]],
    pair_history: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _rating_scheduler_state_payload(
        spec=spec,
        round_id=round_id,
        round_index=round_index,
        pair_specs=pair_specs,
        pair_history=pair_history,
    )
    arena.write_json_artifact(
        mount,
        arena.rating_scheduler_state_ref(spec["tournament_id"], spec["rating_run_id"]),
        payload,
    )
    return arena._to_plain(payload)


def _previous_rating_snapshot(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
    round_index: int,
) -> dict[str, Any] | None:
    for previous_index in range(int(round_index) - 1, -1, -1):
        previous_round_id = arena.rating_round_id(previous_index)
        previous = _read_json(
            runs.volume_path(
                mount,
                arena.rating_round_ratings_ref(
                    tournament_id,
                    rating_run_id,
                    previous_round_id,
                ),
            )
        )
        if previous:
            return previous
    return None


def _write_rating_round_outputs(
    mount: Path,
    *,
    spec: Mapping[str, Any],
    round_id: str,
    round_index: int,
    pair_results: Sequence[Mapping[str, Any]],
    pair_specs: Sequence[Mapping[str, Any]] | None = None,
    game_count: int,
    started_at: str | None,
    previous_snapshot: Mapping[str, Any] | None,
    previous_pair_history: Mapping[str, Any] | None = None,
    include_pair_results: bool = True,
    result_detail_mode: str = "games",
) -> dict[str, Any]:
    _write_battle_index(spec["tournament_id"], pair_results, mount=mount)
    pair_history = arena.pair_history_from_pair_results(
        pair_results,
        previous_pair_history=previous_pair_history,
        rating_spec=spec,
        round_index=round_index,
    )
    scheduler_state = _rating_scheduler_state_payload(
        spec=spec,
        round_id=round_id,
        round_index=round_index,
        pair_specs=pair_specs or pair_results,
        pair_history=pair_history,
    )
    snapshot = arena.rating_snapshot_from_pair_results(
        pair_results=pair_results,
        rating_spec=spec,
        previous_snapshot=previous_snapshot,
        round_index=round_index,
        created_at=runs.utc_timestamp(),
    )
    snapshot["started_at"] = started_at
    snapshot["ended_at"] = runs.utc_timestamp()
    snapshot["input_ref"] = arena.rating_round_input_ref(
        spec["tournament_id"],
        spec["rating_run_id"],
        round_id,
    ).as_posix()
    snapshot["ratings_ref"] = arena.rating_round_ratings_ref(
        spec["tournament_id"],
        spec["rating_run_id"],
        round_id,
    ).as_posix()
    snapshot["latest_ref"] = arena.rating_latest_ref(
        spec["tournament_id"],
        spec["rating_run_id"],
    ).as_posix()
    results_ref = arena.rating_round_results_ref(
        spec["tournament_id"],
        spec["rating_run_id"],
        round_id,
    ).as_posix()
    snapshot["pair_rating_results_ref"] = results_ref
    snapshot["pair_history_ref"] = arena.rating_pair_history_ref(
        spec["tournament_id"],
        spec["rating_run_id"],
    ).as_posix()
    snapshot["scheduler_state_ref"] = arena.rating_scheduler_state_ref(
        spec["tournament_id"],
        spec["rating_run_id"],
    ).as_posix()
    snapshot["game_count"] = int(game_count)
    publish_decision = _rating_latest_publish_decision(
        mount,
        tournament_id=str(spec["tournament_id"]),
        rating_run_id=str(spec["rating_run_id"]),
        snapshot=snapshot,
    )
    publish_global_outputs = bool(publish_decision.get("publish"))
    snapshot["global_outputs_published"] = publish_global_outputs
    snapshot["latest_round_index_before_write"] = publish_decision.get("latest_round_index")
    snapshot["latest_checkpoint_count_before_write"] = publish_decision.get(
        "latest_checkpoint_count"
    )
    if not publish_global_outputs:
        snapshot["latest_write_skipped"] = True
        snapshot["latest_write_skipped_reason"] = str(
            publish_decision.get("reason") or "latest_publish_rejected"
        )
    results_payload = {
        "schema_id": arena.RATING_ROUND_SCHEMA_ID,
        "tournament_id": spec["tournament_id"],
        "rating_run_id": spec["rating_run_id"],
        "round_id": round_id,
        "round_index": round_index,
        "result_detail_mode": result_detail_mode,
        "pair_count": len(pair_results),
        "game_count": int(game_count),
        "pair_summary_refs": [
            pair.get("summary_ref") for pair in pair_results if pair.get("summary_ref")
        ],
        "pair_history_ref": snapshot["pair_history_ref"],
        "scheduler_state_ref": snapshot["scheduler_state_ref"],
        "pair_rating_results": snapshot.get("pair_rating_results", []),
    }
    if include_pair_results:
        results_payload["pair_results"] = arena._to_plain(pair_results)
    else:
        results_payload["pair_result_count"] = len(pair_results)
    arena.write_json_artifact(
        mount,
        arena.rating_round_results_ref(
            spec["tournament_id"],
            spec["rating_run_id"],
            round_id,
        ),
        results_payload,
    )
    slim_snapshot = _slim_rating_snapshot(snapshot)
    arena.write_json_artifact(
        mount,
        arena.rating_round_ratings_ref(
            spec["tournament_id"],
            spec["rating_run_id"],
            round_id,
        ),
        slim_snapshot,
    )
    if publish_global_outputs:
        arena.write_json_artifact(
            mount,
            arena.rating_pair_history_ref(spec["tournament_id"], spec["rating_run_id"]),
            pair_history,
        )
        arena.write_json_artifact(
            mount,
            arena.rating_scheduler_state_ref(
                spec["tournament_id"],
                spec["rating_run_id"],
            ),
            scheduler_state,
        )
        arena.write_json_artifact(
            mount,
            arena.rating_latest_ref(spec["tournament_id"], spec["rating_run_id"]),
            slim_snapshot,
        )
    _append_tournament_lineage_event(
        stage="rating_latest_written",
        tournament_id=str(spec["tournament_id"]),
        rating_run_id=str(spec["rating_run_id"]),
        status="ok" if publish_global_outputs else "skipped",
        reason=None if publish_global_outputs else snapshot.get("latest_write_skipped_reason"),
        mount=mount,
        round_id=round_id,
        round_index=round_index,
        ratings_ref=slim_snapshot.get("ratings_ref"),
        latest_ref=slim_snapshot.get("latest_ref"),
        rating_count=len(slim_snapshot.get("ratings") or []),
        rated_pair_count=slim_snapshot.get("rated_pair_count"),
        stable=slim_snapshot.get("stable"),
        max_abs_delta=slim_snapshot.get("max_abs_delta"),
        global_outputs_published=publish_global_outputs,
        latest_round_index_before_write=publish_decision.get("latest_round_index"),
        latest_checkpoint_count_before_write=publish_decision.get("latest_checkpoint_count"),
    )
    return arena._to_plain(
        {
            "snapshot": slim_snapshot,
            "pair_history": pair_history,
            "scheduler_state": scheduler_state,
            "global_outputs_published": publish_global_outputs,
        }
    )


def _reduce_rating_round_from_summaries(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
    allow_partial: bool = False,
) -> dict[str, Any]:
    input_payload = _read_rating_round_input(
        mount,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
    )
    pair_specs = [
        dict(pair)
        for pair in input_payload.get("pair_specs", [])
        if isinstance(pair, Mapping) and pair.get("battle_id")
    ]
    expected_game_count = int(
        input_payload.get("game_count")
        or sum(int(pair.get("games_per_pair") or 0) for pair in pair_specs)
    )
    shard_progress, _shard_games_by_battle = _rating_round_progress_payload(
        mount,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        load_summaries=False,
        pair_only=True,
    )
    shard_completed_game_count = int(shard_progress.get("completed_game_count") or 0)
    use_shard_tallies = bool(
        shard_completed_game_count
        and not bool(shard_progress.get("counted_game_summary_files"))
        and shard_completed_game_count >= expected_game_count
    )
    games_by_battle: dict[str, list[dict[str, Any]]] = {}
    if use_shard_tallies:
        effective_progress = shard_progress
    else:
        compact_games = _compact_rating_game_summaries_for_pairs(
            mount,
            tournament_id=tournament_id,
            pair_specs=pair_specs,
        )
        progress, games_by_battle = _rating_round_progress_payload(
            mount,
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
            round_id=round_id,
            game_results=compact_games,
            load_summaries=False,
        )
        effective_progress = progress
    effective_completed_game_count = int(effective_progress.get("completed_game_count") or 0)
    if not allow_partial and effective_completed_game_count < expected_game_count:
        _write_rating_progress(mount, effective_progress)
        raise ValueError(
            "rating round incomplete: "
            f"{effective_completed_game_count}/{expected_game_count} games, "
            f"{effective_progress.get('completed_pair_count')}/"
            f"{effective_progress.get('pair_count')} pairs"
        )

    spec = arena.normalize_rating_spec(input_payload.get("rating_spec") or {})
    round_index = int(input_payload.get("round_index", 0) or 0)
    previous_snapshot = _previous_rating_snapshot(
        mount,
        tournament_id=spec["tournament_id"],
        rating_run_id=spec["rating_run_id"],
        round_index=round_index,
    )
    previous_pair_history = (
        _read_json(
            runs.volume_path(
                mount,
                arena.rating_pair_history_ref(spec["tournament_id"], spec["rating_run_id"]),
            )
        )
        or None
    )
    if use_shard_tallies:
        shard_results = []
        for pair in pair_specs:
            shards = _read_battle_shard_summaries(
                mount,
                tournament_id=str(pair["tournament_id"]),
                battle_id=str(pair["battle_id"]),
            )
            if allow_partial and not shards:
                continue
            shard_results.extend(shards)
        work_summary = {
            "work_kind": "shard",
            "parent_result_mode": "volume_shard_tallies",
            "reduced_from_volume": True,
            "games_per_shard": int(spec.get("games_per_shard", arena.DEFAULT_GAMES_PER_SHARD)),
        }
        pair_results, game_count = _summarize_pair_results_from_shard_tallies(
            mount=mount,
            pair_specs=pair_specs,
            shard_results=shard_results,
            started_at=str(input_payload.get("started_at") or ""),
            work_summary=work_summary,
            rating_run_id=spec["rating_run_id"],
            round_id=round_id,
            round_index=round_index,
        )
        progress_for_output = _rating_progress_from_pair_results(
            input_payload=input_payload,
            pair_results=pair_results,
            work_summary=work_summary,
        )
    else:
        pair_results = []
        for pair in pair_specs:
            games = games_by_battle.get(str(pair["battle_id"]), [])
            if allow_partial and not games:
                continue
            spec_ref = arena.battle_pair_spec_ref(
                pair["tournament_id"],
                pair["battle_id"],
            )
            arena.write_json_artifact(mount, spec_ref, pair)
            summary = arena.summarize_pair_results(pair, games)
            summary["started_at"] = input_payload.get("started_at")
            summary["ended_at"] = runs.utc_timestamp()
            summary["summary_ref"] = arena.battle_summary_ref(
                pair["tournament_id"],
                pair["battle_id"],
            ).as_posix()
            arena.write_json_artifact(
                mount,
                arena.battle_summary_ref(pair["tournament_id"], pair["battle_id"]),
                summary,
            )
            pair_results.append(summary)
        game_count = sum(len(games) for games in games_by_battle.values())
        progress_for_output = progress

    round_outputs = _write_rating_round_outputs(
        mount,
        spec=spec,
        round_id=round_id,
        round_index=round_index,
        pair_results=pair_results,
        pair_specs=pair_specs,
        game_count=game_count,
        started_at=str(input_payload.get("started_at") or ""),
        previous_snapshot=previous_snapshot,
        previous_pair_history=(
            previous_pair_history if isinstance(previous_pair_history, Mapping) else None
        ),
    )
    snapshot = round_outputs["snapshot"]
    progress_for_output["reduced_at"] = runs.utc_timestamp()
    progress_for_output["allow_partial_reduce"] = bool(allow_partial)
    progress_for_output["rated_pair_count"] = snapshot.get("rated_pair_count")
    progress_for_output["status"] = "complete"
    progress_for_output["phase"] = "reduced"
    progress_for_output["work_summary"] = (
        work_summary if use_shard_tallies else {"reduced_from_volume": True}
    )
    _write_rating_progress(mount, progress_for_output)
    return arena._to_plain(
        {
            "progress": progress_for_output,
            "snapshot": snapshot,
            "pair_history": round_outputs.get("pair_history"),
            "scheduler_state": round_outputs.get("scheduler_state"),
            "pair_count": len(pair_results),
            "game_count": game_count,
            "reduced_from": "shard_tallies" if use_shard_tallies else "game_summaries",
        }
    )


def _flatten_game_results_from_shards(
    shard_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    games_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for shard in shard_results:
        rows = shard.get("games") if isinstance(shard, Mapping) else None
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping):
                game = dict(row)
                key = (
                    str(game.get("battle_id") or ""),
                    str(game.get("game_id") or game.get("summary_ref") or ""),
                )
                games_by_key[key] = game
    games = list(games_by_key.values())
    games.sort(
        key=lambda game: (
            int(game.get("pair_index", 0) or 0),
            str(game.get("battle_id") or ""),
            int(game.get("game_index", 0) or 0),
            str(game.get("game_id") or ""),
        )
    )
    return games


def _games_from_shards_for_tally_fallback(
    mount: Path,
    shards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    games = _flatten_game_results_from_shards(shards)
    if games:
        return games
    hydrated_shards = []
    for ref in _refs_from_shards(shards, "summary_ref"):
        shard = _read_tournament_json_ref(mount, ref)
        if shard:
            hydrated_shards.append(shard)
    return _flatten_game_results_from_shards(hydrated_shards)


def _dedupe_shard_results(
    shard_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    shards_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for shard in shard_results:
        if not isinstance(shard, Mapping):
            continue
        key = (
            str(shard.get("battle_id") or ""),
            str(shard.get("shard_id") or shard.get("shard_index") or ""),
        )
        if not key[0] or not key[1]:
            continue
        shards_by_key[key] = dict(shard)
    shards = list(shards_by_key.values())
    shards.sort(
        key=lambda shard: (
            int(shard.get("pair_index", 0) or 0),
            str(shard.get("battle_id") or ""),
            int(shard.get("shard_index", 0) or 0),
            str(shard.get("shard_id") or ""),
        )
    )
    return shards


def _first_ref_from_shards(shards: Sequence[Mapping[str, Any]], key: str) -> str | None:
    for shard in shards:
        value = shard.get(key)
        if value:
            return str(value)
    return None


def _refs_from_shards(shards: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    refs = []
    for shard in shards:
        value = shard.get(key)
        if isinstance(value, str) and value:
            refs.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            refs.extend(str(item) for item in value if item)
    return refs


def _read_battle_shard_summaries(
    mount: Path,
    *,
    tournament_id: str,
    battle_id: str,
) -> list[dict[str, Any]]:
    root = runs.volume_path(mount, arena.battle_root_ref(tournament_id, battle_id)) / "shards"
    if not root.exists():
        return []
    shards = []
    for path in root.glob("*/summary.json"):
        shard = _read_json(path)
        if not shard:
            continue
        shard.setdefault("summary_ref", runs.file_ref(path, mount=mount))
        shards.append(shard)
    shards.sort(
        key=lambda shard: (
            int(shard.get("shard_index", 0) or 0),
            str(shard.get("shard_id") or ""),
        )
    )
    return arena._to_plain(shards)


def _summarize_live_pair_from_shards(
    pair: Mapping[str, Any],
    *,
    shards: Sequence[Mapping[str, Any]],
    rating_run_id: str | None = None,
    round_id: str | None = None,
    round_index: int | None = None,
) -> dict[str, Any]:
    tally = arena.merge_game_tallies(
        [dict(shard.get("tally")) for shard in shards if isinstance(shard.get("tally"), Mapping)]
    )
    summary = arena.summarize_pair_from_tally(
        pair,
        tally=tally,
        first_gif_ref=_first_ref_from_shards(shards, "first_gif_ref"),
    )
    summary["summary_ref"] = arena.battle_summary_ref(
        pair["tournament_id"],
        pair["battle_id"],
    ).as_posix()
    summary["shard_summary_refs"] = _refs_from_shards(shards, "summary_ref")
    summary["shard_summary_ref_count"] = len(summary["shard_summary_refs"])
    summary["sample_gif_refs"] = _refs_from_shards(shards, "sample_gif_refs")
    summary["shard_count"] = len(shards)
    summary["result_detail_mode"] = "live_shard_tally"
    if rating_run_id:
        summary["rating_run_id"] = rating_run_id
    if round_id:
        summary["round_id"] = round_id
    if round_index is not None:
        summary["round_index"] = int(round_index)
    return arena._to_plain(summary)


def _summarize_pair_results_from_shard_tallies(
    *,
    mount: Path,
    pair_specs: Sequence[Mapping[str, Any]],
    shard_results: Sequence[Mapping[str, Any]],
    started_at: str | None,
    work_summary: Mapping[str, Any],
    rating_run_id: str | None = None,
    round_id: str | None = None,
    round_index: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    shards_by_battle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shard in _dedupe_shard_results(shard_results):
        shards_by_battle[str(shard.get("battle_id") or "")].append(shard)

    pair_results: list[dict[str, Any]] = []
    total_games = 0
    for pair in pair_specs:
        battle_id = str(pair["battle_id"])
        shards = shards_by_battle.get(battle_id, [])
        tally = arena.merge_game_tallies(
            [
                dict(shard.get("tally"))
                for shard in shards
                if isinstance(shard.get("tally"), Mapping)
            ]
        )
        games_for_tally = []
        if not tally.get("wins_by_checkpoint"):
            games_for_tally = _games_from_shards_for_tally_fallback(mount, shards)
            if games_for_tally:
                tally = arena.tally_game_results(games_for_tally)
        total_games += int(tally.get("game_count") or 0)
        spec_ref = arena.battle_pair_spec_ref(
            pair["tournament_id"],
            pair["battle_id"],
        )
        arena.write_json_artifact(mount, spec_ref, pair)
        summary = arena.summarize_pair_from_tally(
            pair,
            tally=tally,
            first_gif_ref=_first_ref_from_shards(shards, "first_gif_ref"),
            games=games_for_tally or None,
        )
        summary["game_summary_ref_count"] = sum(
            int(shard.get("game_summary_ref_count") or 0) for shard in shards
        )
        summary["shard_summary_refs"] = _refs_from_shards(shards, "summary_ref")
        summary["shard_summary_ref_count"] = len(summary["shard_summary_refs"])
        summary["sample_gif_refs"] = _refs_from_shards(shards, "sample_gif_refs")
        summary["shard_count"] = len(shards)
        summary["started_at"] = started_at
        summary["ended_at"] = runs.utc_timestamp()
        summary["work_summary"] = dict(work_summary)
        summary["result_detail_mode"] = "shard_tally"
        if rating_run_id:
            summary["rating_run_id"] = rating_run_id
        if round_id:
            summary["round_id"] = round_id
        if round_index is not None:
            summary["round_index"] = int(round_index)
        summary["summary_ref"] = arena.battle_summary_ref(
            pair["tournament_id"],
            pair["battle_id"],
        ).as_posix()
        arena.write_json_artifact(
            mount,
            arena.battle_summary_ref(pair["tournament_id"], pair["battle_id"]),
            summary,
        )
        pair_results.append(summary)
    return arena._to_plain(pair_results), int(total_games)


def _rating_progress_from_pair_results(
    *,
    input_payload: Mapping[str, Any],
    pair_results: Sequence[Mapping[str, Any]],
    work_summary: Mapping[str, Any],
    snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tournament_id = str(input_payload["tournament_id"])
    rating_run_id = str(input_payload["rating_run_id"])
    round_id = str(input_payload["round_id"])
    pair_specs = [
        dict(pair)
        for pair in input_payload.get("pair_specs", [])
        if isinstance(pair, Mapping) and pair.get("battle_id")
    ]
    summaries_by_battle = {
        str(summary.get("battle_id")): summary
        for summary in pair_results
        if isinstance(summary, Mapping) and summary.get("battle_id")
    }
    completed_game_count = 0
    ok_game_count = 0
    failed_game_count = 0
    unknown_result_count = 0
    completed_pair_count = 0
    partial_pair_count = 0
    started_pair_count = 0
    max_started_pair_index: int | None = None
    max_completed_pair_index: int | None = None
    recent_rows = []
    for pair in pair_specs:
        battle_id = str(pair["battle_id"])
        pair_index = int(pair.get("pair_index", 0) or 0)
        expected = int(pair.get("games_per_pair") or 0)
        summary = summaries_by_battle.get(battle_id, {})
        tally = summary.get("tally") if isinstance(summary.get("tally"), Mapping) else {}
        seen = int(tally.get("game_count") or 0)
        ok = int(tally.get("completed_count") or 0)
        failed = int(tally.get("failure_count") or 0)
        unknown = max(0, seen - ok - failed)
        completed_game_count += seen
        ok_game_count += ok
        failed_game_count += failed
        unknown_result_count += unknown
        if seen:
            started_pair_count += 1
            max_started_pair_index = (
                pair_index
                if max_started_pair_index is None
                else max(max_started_pair_index, pair_index)
            )
        if expected > 0 and seen >= expected:
            completed_pair_count += 1
            max_completed_pair_index = (
                pair_index
                if max_completed_pair_index is None
                else max(max_completed_pair_index, pair_index)
            )
        elif seen:
            partial_pair_count += 1
        if seen:
            recent_rows.append(
                {
                    "battle_id": battle_id,
                    "pair_index": pair_index,
                    "seen_game_count": seen,
                    "expected_game_count": expected,
                    "complete": bool(expected > 0 and seen >= expected),
                }
            )
    game_count = int(
        input_payload.get("game_count")
        or sum(int(pair.get("games_per_pair") or 0) for pair in pair_specs)
    )
    payload = {
        "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
        "app_name": APP_NAME,
        "artifact_volume_name": TOURNAMENT_VOLUME_NAME,
        "checkpoint_volume_name": CHECKPOINT_VOLUME_NAME,
        "tournament_id": tournament_id,
        "rating_run_id": rating_run_id,
        "round_id": round_id,
        "round_index": int(input_payload.get("round_index", 0) or 0),
        "updated_at": runs.utc_timestamp(),
        "updated_ts": time.time(),
        "input_ref": arena.rating_round_input_ref(
            tournament_id,
            rating_run_id,
            round_id,
        ).as_posix(),
        "progress_ref": arena.rating_progress_ref(tournament_id, rating_run_id).as_posix(),
        "round_progress_ref": arena.rating_round_progress_ref(
            tournament_id,
            rating_run_id,
            round_id,
        ).as_posix(),
        "latest_ref": arena.rating_latest_ref(tournament_id, rating_run_id).as_posix(),
        "pair_count": len(pair_specs),
        "game_count": game_count,
        "completed_game_count": completed_game_count,
        "estimated_seen_game_count": completed_game_count,
        "ok_game_count": ok_game_count,
        "failed_game_count": failed_game_count,
        "unknown_result_count": unknown_result_count,
        "result_counts_known": bool(unknown_result_count == 0),
        "count_basis": "shard_tallies",
        "started_pair_count": started_pair_count,
        "partial_pair_count": partial_pair_count,
        "completed_pair_count": completed_pair_count,
        "completion_fraction": (
            float(completed_game_count) / float(game_count) if game_count else 0.0
        ),
        "estimated_completion_fraction": (
            float(completed_game_count) / float(game_count) if game_count else 0.0
        ),
        "max_started_pair_index": max_started_pair_index,
        "max_completed_pair_index": max_completed_pair_index,
        "recent_started_pairs": sorted(
            recent_rows,
            key=lambda row: (-int(row["pair_index"]), str(row["battle_id"])),
        )[:25],
        "phase": "reduced",
        "status": "complete" if completed_game_count >= game_count and game_count else "running",
        "work_summary": dict(work_summary),
    }
    if snapshot:
        payload["ratings_written"] = True
        payload["rated_pair_count"] = snapshot.get("rated_pair_count")
        payload["max_abs_delta"] = snapshot.get("max_abs_delta")
        payload["stable"] = snapshot.get("stable")
    return arena._to_plain(payload)


def _build_game_work_specs(
    pair_specs: Sequence[Mapping[str, Any]],
    *,
    games_per_shard: int = arena.DEFAULT_GAMES_PER_SHARD,
    return_games: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    shard_size = int(games_per_shard)
    if shard_size < 1:
        raise ValueError("games_per_shard must be at least 1")
    if shard_size == 1:
        return (
            "game",
            [game for pair in pair_specs for game in arena.build_game_specs_for_pair(pair)],
        )
    shards = [
        shard
        for pair in pair_specs
        for shard in arena.build_game_shard_specs_for_pair(
            pair,
            games_per_shard=shard_size,
        )
    ]
    for shard in shards:
        shard["return_games"] = bool(return_games)
    return ("shard", shards)


def _run_game_work_map(
    pair_specs: Sequence[Mapping[str, Any]],
    *,
    games_per_shard: int = arena.DEFAULT_GAMES_PER_SHARD,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    work_kind, work_specs = _build_game_work_specs(
        pair_specs,
        games_per_shard=games_per_shard,
    )
    if work_kind == "game":
        game_results = list(
            curvytron_tournament_game.map(
                work_specs,
                order_outputs=False,
            )
        )
    else:
        shard_results = list(
            curvytron_tournament_game_shard.map(
                work_specs,
                order_outputs=False,
            )
        )
        game_results = _flatten_game_results_from_shards(shard_results)
    return arena._to_plain(game_results), {
        "work_kind": work_kind,
        "work_count": len(work_specs),
        "games_per_shard": int(games_per_shard),
        "reuse_policies_per_shard": bool(
            work_kind == "shard" and work_specs and work_specs[0].get("reuse_policies", True)
        ),
    }


def _rating_spec_with_latest_roster(
    mount: Path,
    raw_spec: Mapping[str, Any],
) -> dict[str, Any]:
    raw = dict(raw_spec)
    checkpoints = raw.get("checkpoints") or raw.get("checkpoint_refs") or []
    if not bool(raw.get("continue_from_latest", False)):
        return arena.normalize_rating_spec(raw)

    tournament_id = runs.clean_id(
        str(raw.get("tournament_id") or "tournament"),
        label="tournament_id",
    )
    rating_run_id = runs.clean_id(
        str(raw.get("rating_run_id") or arena.DEFAULT_RATING_RUN_ID),
        label="rating_run_id",
    )
    latest = _read_json(
        runs.volume_path(
            mount,
            arena.rating_latest_ref(tournament_id, rating_run_id),
        )
    )
    if not latest:
        return arena.normalize_rating_spec(raw)

    roster = latest.get("checkpoint_roster")
    if not isinstance(roster, Mapping):
        roster = {}
    rows = arena._rating_rows_by_checkpoint(latest)
    ordered_ids = [
        str(row["checkpoint_id"])
        for row in latest.get("ratings", [])
        if isinstance(row, Mapping) and row.get("checkpoint_id")
    ]
    seen_ordered_ids = set(ordered_ids)
    ordered_ids.extend(
        checkpoint_id
        for checkpoint_id in sorted(str(item) for item in roster)
        if checkpoint_id not in seen_ordered_ids
    )

    restored_checkpoints: list[dict[str, Any]] = []
    for checkpoint_id in ordered_ids:
        roster_row = roster.get(checkpoint_id, {})
        if not isinstance(roster_row, Mapping):
            roster_row = {}
        rating_row = rows.get(checkpoint_id, {})
        checkpoint_ref = roster_row.get("checkpoint_ref") or rating_row.get("checkpoint_ref")
        if not checkpoint_ref:
            continue
        restored_checkpoints.append(
            {
                "checkpoint_id": checkpoint_id,
                "label": rating_row.get("label") or checkpoint_id,
                "checkpoint_ref": checkpoint_ref,
                "run_id": roster_row.get("run_id") or rating_row.get("run_id"),
                "attempt_id": roster_row.get("attempt_id") or rating_row.get("attempt_id"),
                "iteration": (
                    roster_row.get("iteration")
                    if roster_row.get("iteration") is not None
                    else rating_row.get("iteration")
                ),
                "latest_for_run": bool(
                    roster_row.get(
                        "latest_for_run",
                        rating_row.get("latest_for_run", False),
                    )
                ),
                "checkpoint_mtime_ns": (
                    roster_row.get("checkpoint_mtime_ns")
                    if roster_row.get("checkpoint_mtime_ns") is not None
                    else rating_row.get("checkpoint_mtime_ns")
                ),
                "model_env_variant": roster_row.get("model_env_variant"),
                "model_reward_variant": roster_row.get("model_reward_variant"),
                "policy_trail_render_mode": roster_row.get("policy_trail_render_mode"),
                "policy_bonus_render_mode": roster_row.get("policy_bonus_render_mode"),
            }
        )
    if checkpoints:
        explicit_checkpoints = checkpoints
        if isinstance(explicit_checkpoints, str):
            explicit_checkpoints = arena.parse_checkpoint_refs(explicit_checkpoints)
        normalized_checkpoints = arena.normalize_checkpoint_specs(list(explicit_checkpoints))
        restored_by_ref = {
            str(checkpoint["checkpoint_ref"]): checkpoint
            for checkpoint in restored_checkpoints
            if checkpoint.get("checkpoint_ref")
        }
        merged_checkpoints = []
        for checkpoint in normalized_checkpoints:
            restored = restored_by_ref.get(str(checkpoint.get("checkpoint_ref") or ""))
            if not restored:
                merged_checkpoints.append(checkpoint)
                continue
            merged = dict(checkpoint)
            merged["checkpoint_id"] = restored["checkpoint_id"]
            if restored.get("label"):
                merged["label"] = restored["label"]
            for key in (
                "run_id",
                "attempt_id",
                "iteration",
                "checkpoint_mtime_ns",
                "model_env_variant",
                "model_reward_variant",
                "policy_trail_render_mode",
                "policy_bonus_render_mode",
            ):
                if merged.get(key) in (None, ""):
                    merged[key] = restored.get(key)
            merged_checkpoints.append(merged)
        raw["checkpoints"] = merged_checkpoints
        raw.pop("checkpoint_refs", None)
        return arena.normalize_rating_spec(raw)
    raw["checkpoints"] = restored_checkpoints
    return arena.normalize_rating_spec(raw)


@app.function(
    image=image,
    volumes=_checkpoint_volumes(),
    cpu=1.0,
    memory=1024,
    timeout=10 * 60,
)
def curvytron_discover_checkpoints(discovery_spec: dict[str, Any]) -> dict[str, Any]:
    _reload_volume(checkpoint_volume)
    run_ids = discovery_spec.get("run_ids") or []
    if isinstance(run_ids, str):
        run_ids = [item.strip() for item in run_ids.replace("\n", ",").split(",") if item.strip()]
    checkpoint_iteration_raw = discovery_spec.get("checkpoint_iteration")
    checkpoint_iteration = (
        int(checkpoint_iteration_raw)
        if checkpoint_iteration_raw not in (None, "", -1, "-1")
        else None
    )
    result = _discover_checkpoint_refs(
        RUNS_MOUNT,
        run_ids=run_ids if isinstance(run_ids, Sequence) else [],
        run_id_prefix=str(discovery_spec.get("run_id_prefix") or ""),
        max_runs=int(discovery_spec.get("max_runs") or 0),
        checkpoint_iteration=checkpoint_iteration,
        checkpoint_selection=str(
            discovery_spec.get("checkpoint_selection") or arena.CHECKPOINT_SELECTION_LATEST
        ),
    )
    print(
        json.dumps(
            arena._to_plain(
                {
                    "found_count": result["found_count"],
                    "missing_count": result["missing_count"],
                    "checkpoint_iteration": result["checkpoint_iteration"],
                }
            ),
            sort_keys=True,
        )
    )
    return arena._to_plain(result)


def _failure_game_summary_or_inline(
    game_spec: Mapping[str, Any],
    exc: BaseException,
    *,
    artifact_mount: Path,
) -> dict[str, Any]:
    try:
        return arena.failure_game_summary(
            dict(game_spec),
            exc,
            artifact_mount=artifact_mount,
        )
    except Exception as summary_exc:  # pragma: no cover - remote Volume diagnosis.
        result = arena.failure_game_summary_payload(dict(game_spec), exc)
        intended_summary_ref = result.pop("summary_ref", None)
        result["summary_ref"] = None
        if intended_summary_ref:
            result["intended_summary_ref"] = intended_summary_ref
        result["failure_summary_write_error_type"] = type(summary_exc).__name__
        result["failure_summary_write_error"] = str(summary_exc)
        result["failure_summary_write_error_context"] = (
            "arena.failure_game_summary failed; returning an in-memory failure "
            "so the rating round can finish and report this game as failed"
        )
        return arena._to_plain(result)


@app.function(
    image=image,
    volumes=_game_volumes(),
    cpu=1.0,
    memory=4096,
    timeout=24 * 60 * 60,
    min_containers=TOURNAMENT_GAME_WORKER_MIN_CONTAINERS,
    buffer_containers=TOURNAMENT_GAME_WORKER_BUFFER_CONTAINERS,
    scaledown_window=TOURNAMENT_GAME_WORKER_SCALEDOWN_WINDOW_SECONDS,
    retries=LOW_LEVEL_WORKER_RETRIES,
)
def curvytron_tournament_game(game_spec: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    reload_started = time.perf_counter()
    checkpoint_reload_error = _reload_volume(checkpoint_volume)
    reload_seconds = time.perf_counter() - reload_started
    run_started = time.perf_counter()
    try:
        result = arena.run_checkpoint_game(
            game_spec,
            checkpoint_mount=RUNS_MOUNT,
            artifact_mount=TOURNAMENT_MOUNT,
            remote_root=REMOTE_ROOT,
        )
    except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
        result = _failure_game_summary_or_inline(
            game_spec,
            exc,
            artifact_mount=TOURNAMENT_MOUNT,
        )
    run_seconds = time.perf_counter() - run_started
    result["worker_timing"] = {
        "checkpoint_reload_seconds": reload_seconds,
        "run_seconds": run_seconds,
        "precommit_total_seconds": time.perf_counter() - started,
    }
    summary_ref = result.get("summary_ref")
    if isinstance(summary_ref, str) and summary_ref:
        try:
            runs.write_json(
                runs.volume_path(TOURNAMENT_MOUNT, summary_ref),
                arena._to_plain(result),
            )
        except Exception as exc:  # pragma: no cover - best-effort telemetry.
            result["summary_timing_rewrite_error"] = f"{type(exc).__name__}: {exc}"
    commit_started = time.perf_counter()
    commit_error = _commit_volume(tournament_volume)
    commit_seconds = time.perf_counter() - commit_started
    result["worker_timing"]["commit_seconds"] = commit_seconds
    result["worker_timing"]["total_seconds"] = time.perf_counter() - started
    if checkpoint_reload_error:
        result["checkpoint_reload_error"] = checkpoint_reload_error
    if commit_error:
        result["commit_error"] = commit_error
    compact = arena._compact_game_result(result)
    print(
        json.dumps(
            arena._to_plain(
                {
                    "ok": compact.get("ok"),
                    "battle_id": compact.get("battle_id"),
                    "game_id": compact.get("game_id"),
                    "pair_index": compact.get("pair_index"),
                    "seat_order": compact.get("seat_order"),
                    "score": compact.get("score"),
                    "physical_steps": compact.get("physical_steps"),
                    "worker_timing": compact.get("worker_timing"),
                    "error_type": compact.get("error_type"),
                }
            ),
            sort_keys=True,
        )
    )
    return arena._to_plain(compact)


@app.function(
    image=image,
    volumes=_game_volumes(),
    cpu=1.0,
    memory=4096,
    timeout=24 * 60 * 60,
    min_containers=TOURNAMENT_GAME_SHARD_WORKER_MIN_CONTAINERS,
    buffer_containers=TOURNAMENT_GAME_SHARD_WORKER_BUFFER_CONTAINERS,
    scaledown_window=TOURNAMENT_GAME_SHARD_WORKER_SCALEDOWN_WINDOW_SECONDS,
    retries=LOW_LEVEL_WORKER_RETRIES,
)
def curvytron_tournament_game_shard(shard_spec: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    reload_started = time.perf_counter()
    checkpoint_reload_error = _reload_volume(checkpoint_volume)
    reload_seconds = time.perf_counter() - reload_started
    game_specs = shard_spec.get("game_specs")
    if not isinstance(game_specs, list) or not game_specs:
        raise ValueError("game shard needs a non-empty game_specs list")

    requested_policy_reuse = bool(shard_spec.get("reuse_policies", True))
    shard_policy_mode = str(dict(game_specs[0]).get("policy_mode", arena.POLICY_MODE_EVAL))
    reuse_disabled_reason = None
    reuse_policies = requested_policy_reuse
    if shard_policy_mode != arena.POLICY_MODE_EVAL:
        reuse_policies = False
        if requested_policy_reuse:
            reuse_disabled_reason = "policy_reuse_only_enabled_for_eval_mode"
    policy_entries: list[dict[str, Any]] | None = None
    policy_load_failure: dict[str, Any] | None = None
    policy_load_started = time.perf_counter()
    if reuse_policies:
        try:
            policy_entries = arena.load_policy_entries_for_game(
                dict(game_specs[0]),
                checkpoint_mount=RUNS_MOUNT,
                artifact_mount=TOURNAMENT_MOUNT,
                remote_root=REMOTE_ROOT,
            )
        except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
            policy_load_failure = arena.exception_payload(exc)
    policy_load_seconds = time.perf_counter() - policy_load_started

    games = []
    run_seconds_total = 0.0
    for game_spec in game_specs:
        run_started = time.perf_counter()
        try:
            if policy_load_failure is not None:
                raise RuntimeError(
                    "shard policy preload failed: "
                    f"{policy_load_failure.get('error_type')}: {policy_load_failure.get('error')}"
                )
            result = arena.run_checkpoint_game(
                dict(game_spec),
                checkpoint_mount=RUNS_MOUNT,
                artifact_mount=TOURNAMENT_MOUNT,
                remote_root=REMOTE_ROOT,
                preloaded_policy_entries=policy_entries,
            )
        except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
            result = _failure_game_summary_or_inline(
                dict(game_spec),
                exc,
                artifact_mount=TOURNAMENT_MOUNT,
            )
            if policy_load_failure is not None:
                result["policy_load_failure"] = policy_load_failure
        run_seconds = time.perf_counter() - run_started
        run_seconds_total += run_seconds
        result["worker_timing"] = {
            "checkpoint_reload_seconds": reload_seconds,
            "requested_policy_reuse": requested_policy_reuse,
            "policy_reuse": reuse_policies,
            "policy_reuse_disabled_reason": reuse_disabled_reason,
            "policy_load_seconds": policy_load_seconds,
            "run_seconds": run_seconds,
            "precommit_total_seconds": time.perf_counter() - started,
            "shard_id": shard_spec.get("shard_id"),
            "shard_index": shard_spec.get("shard_index"),
        }
        summary_ref = result.get("summary_ref")
        if isinstance(summary_ref, str) and summary_ref:
            try:
                runs.write_json(
                    runs.volume_path(TOURNAMENT_MOUNT, summary_ref),
                    arena._to_plain(result),
                )
            except Exception as exc:  # pragma: no cover - best-effort telemetry.
                result["summary_timing_rewrite_error"] = f"{type(exc).__name__}: {exc}"
        if checkpoint_reload_error:
            result["checkpoint_reload_error"] = checkpoint_reload_error
        games.append(arena._compact_game_result(result))

    timing = {
        "checkpoint_reload_seconds": reload_seconds,
        "requested_policy_reuse": requested_policy_reuse,
        "policy_reuse": reuse_policies,
        "policy_reuse_disabled_reason": reuse_disabled_reason,
        "policy_load_seconds": policy_load_seconds,
        "run_seconds_total": run_seconds_total,
        "commit_seconds": None,
        "total_seconds": time.perf_counter() - started,
    }
    payload = {
        "schema_id": arena.GAME_SHARD_SCHEMA_ID,
        "ok": all(bool(game.get("ok")) for game in games),
        "tournament_id": shard_spec.get("tournament_id"),
        "battle_id": shard_spec.get("battle_id"),
        "pair_index": shard_spec.get("pair_index"),
        "shard_id": shard_spec.get("shard_id"),
        "shard_index": shard_spec.get("shard_index"),
        "game_count": len(games),
        "games": arena._to_plain(games),
        "worker_timing": timing,
    }
    if policy_load_failure is not None:
        payload["policy_load_failure"] = policy_load_failure
    if checkpoint_reload_error:
        payload["checkpoint_reload_error"] = checkpoint_reload_error
    tally = arena.tally_game_results(games)
    payload["tally"] = tally
    payload["game_summary_ref_count"] = sum(1 for game in games if game.get("summary_ref"))
    sample_gif_refs = [str(game["gif_ref"]) for game in games if game.get("gif_ref")]
    payload["sample_gif_refs"] = sample_gif_refs
    for game in games:
        if game.get("gif_ref"):
            payload["first_gif_ref"] = game["gif_ref"]
            break
    shard_summary_ref = arena.game_shard_summary_ref(
        str(shard_spec.get("tournament_id")),
        str(shard_spec.get("battle_id")),
        str(shard_spec.get("shard_id")),
    )
    payload["summary_ref"] = shard_summary_ref.as_posix()
    try:
        runs.write_json(
            runs.volume_path(TOURNAMENT_MOUNT, shard_summary_ref),
            arena._to_plain(payload),
        )
    except Exception as exc:  # pragma: no cover - best-effort telemetry.
        payload["shard_summary_write_error"] = f"{type(exc).__name__}: {exc}"
    commit_started = time.perf_counter()
    commit_error = _commit_volume(tournament_volume)
    commit_seconds = time.perf_counter() - commit_started
    timing["commit_seconds"] = commit_seconds
    timing["total_seconds"] = time.perf_counter() - started
    if commit_error:
        payload["commit_error"] = commit_error
    if not bool(shard_spec.get("return_games", True)):
        payload.pop("games", None)
    print(
        json.dumps(
            arena._to_plain(
                {
                    "ok": payload["ok"],
                    "battle_id": payload.get("battle_id"),
                    "shard_id": payload.get("shard_id"),
                    "pair_index": payload.get("pair_index"),
                    "game_count": payload.get("game_count"),
                    "tally": tally,
                    "worker_timing": timing,
                }
            ),
            sort_keys=True,
        )
    )
    return arena._to_plain(payload)


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    cpu=0.5,
    memory=1024,
    timeout=24 * 60 * 60,
    max_containers=100,
)
def curvytron_tournament_pair(pair_spec: dict[str, Any]) -> dict[str, Any]:
    _reload_volume(tournament_volume)
    pair = arena.normalize_pair_spec(pair_spec)
    started_at = runs.utc_timestamp()
    spec_ref = arena.battle_pair_spec_ref(pair["tournament_id"], pair["battle_id"])
    arena.write_json_artifact(TOURNAMENT_MOUNT, spec_ref, pair)
    games_per_shard = int(pair.get("games_per_shard", arena.DEFAULT_GAMES_PER_SHARD))
    game_results, work_summary = _run_game_work_map(
        [pair],
        games_per_shard=games_per_shard,
    )
    summary = arena.summarize_pair_results(pair, game_results)
    summary["started_at"] = started_at
    summary["ended_at"] = runs.utc_timestamp()
    summary["work_summary"] = work_summary
    summary["summary_ref"] = arena.battle_summary_ref(
        pair["tournament_id"],
        pair["battle_id"],
    ).as_posix()
    arena.write_json_artifact(
        TOURNAMENT_MOUNT,
        arena.battle_summary_ref(pair["tournament_id"], pair["battle_id"]),
        summary,
    )
    commit_error = _commit_volume(tournament_volume)
    if commit_error:
        summary["commit_error"] = commit_error
    print(
        json.dumps(
            arena._to_plain({"battle_id": pair["battle_id"], "tally": summary["tally"]}),
            sort_keys=True,
        )
    )
    return arena._to_plain(summary)


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    cpu=1.0,
    memory=2048,
    timeout=24 * 60 * 60,
    max_containers=10,
)
def curvytron_tournament_run(tournament_spec: dict[str, Any]) -> dict[str, Any]:
    _reload_volume(tournament_volume)
    spec = dict(tournament_spec)
    tournament_id = runs.clean_id(
        str(spec.get("tournament_id") or runs.new_run_id("arena")), label="tournament_id"
    )
    spec["tournament_id"] = tournament_id
    started_at = runs.utc_timestamp()
    _write_tournament_marker(tournament_id)
    _write_tournament_manifest(spec, status="running")
    startup_commit_error = _commit_volume(tournament_volume)

    checkpoints = spec.get("checkpoints") or spec.get("checkpoint_refs") or []
    if isinstance(checkpoints, str):
        checkpoints = arena.parse_checkpoint_refs(checkpoints)
    if not isinstance(checkpoints, list):
        raise ValueError(
            "tournament spec needs a checkpoints list or comma-separated checkpoint_refs"
        )
    pair_specs = arena.build_pair_specs(
        tournament_id=tournament_id,
        checkpoints=checkpoints,
        games_per_pair=int(spec.get("games_per_pair", arena.DEFAULT_GAMES_PER_PAIR)),
        ordered_pairs=bool(spec.get("ordered_pairs", arena.DEFAULT_ORDERED_PAIRS)),
        include_self_pairs=bool(spec.get("include_self_pairs", arena.DEFAULT_INCLUDE_SELF_PAIRS)),
        seed=int(spec.get("seed", 0)),
        max_steps=int(spec.get("max_steps", arena.DEFAULT_MAX_STEPS)),
        decision_ms=float(spec.get("decision_ms", arena.DEFAULT_DECISION_MS)),
        decision_source_frames=spec.get("decision_source_frames"),
        source_physics_step_ms=float(
            spec.get("source_physics_step_ms", arena.DEFAULT_SOURCE_PHYSICS_STEP_MS)
        ),
        num_simulations=int(spec.get("num_simulations", arena.DEFAULT_NUM_SIMULATIONS)),
        policy_batch_size=int(spec.get("policy_batch_size", arena.DEFAULT_POLICY_BATCH_SIZE)),
        policy_mode=str(spec.get("policy_mode", arena.POLICY_MODE_EVAL)),
        collect_temperature=float(
            spec.get("collect_temperature", arena.DEFAULT_COLLECT_TEMPERATURE)
        ),
        collect_epsilon=float(spec.get("collect_epsilon", arena.DEFAULT_COLLECT_EPSILON)),
        natural_bonus_spawn=bool(spec.get("natural_bonus_spawn", True)),
        policy_trail_render_mode=spec.get("policy_trail_render_mode"),
        policy_bonus_render_mode=spec.get("policy_bonus_render_mode"),
        trail_render_mode=spec.get("trail_render_mode"),
        frame_stride=int(spec.get("frame_stride", arena.DEFAULT_FRAME_STRIDE)),
        frame_size=int(spec.get("frame_size", arena.DEFAULT_FRAME_SIZE)),
        gif_fps=float(spec.get("gif_fps", arena.DEFAULT_GIF_FPS)),
        save_gif=bool(spec.get("save_gif", arena.DEFAULT_SAVE_GIF)),
        gif_sample_games_per_pair=int(
            spec.get(
                "gif_sample_games_per_pair",
                arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR,
            )
        ),
        gif_sample_strategy=str(spec.get("gif_sample_strategy", arena.DEFAULT_GIF_SAMPLE_STRATEGY)),
        games_per_shard=int(spec.get("games_per_shard", arena.DEFAULT_GAMES_PER_SHARD)),
        reuse_policies_per_shard=bool(
            spec.get(
                "reuse_policies_per_shard",
                arena.DEFAULT_REUSE_POLICIES_PER_SHARD,
            )
        ),
        seat_order_mode=str(spec.get("seat_order_mode", arena.DEFAULT_SEAT_ORDER_MODE)),
        save_frames_npz=bool(spec.get("save_frames_npz", arena.DEFAULT_SAVE_FRAMES_NPZ)),
        action_trace_limit=int(spec.get("action_trace_limit", 128)),
    )
    game_results, work_summary = _run_game_work_map(
        pair_specs,
        games_per_shard=int(spec.get("games_per_shard", arena.DEFAULT_GAMES_PER_SHARD)),
    )
    games_by_battle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for game in game_results:
        games_by_battle[str(game.get("battle_id") or "")].append(game)
    pair_results = []
    for pair in pair_specs:
        spec_ref = arena.battle_pair_spec_ref(
            pair["tournament_id"],
            pair["battle_id"],
        )
        arena.write_json_artifact(TOURNAMENT_MOUNT, spec_ref, pair)
        summary = arena.summarize_pair_results(
            pair,
            games_by_battle.get(pair["battle_id"], []),
        )
        summary["started_at"] = started_at
        summary["ended_at"] = runs.utc_timestamp()
        summary["work_summary"] = work_summary
        summary["summary_ref"] = arena.battle_summary_ref(
            pair["tournament_id"],
            pair["battle_id"],
        ).as_posix()
        arena.write_json_artifact(
            TOURNAMENT_MOUNT,
            arena.battle_summary_ref(pair["tournament_id"], pair["battle_id"]),
            summary,
        )
        pair_results.append(summary)
    _write_battle_index(tournament_id, pair_results)
    standings = arena.standings_from_pair_results(pair_results)
    standings["tournament_id"] = tournament_id
    standings["created_at"] = runs.utc_timestamp()
    arena.write_json_artifact(
        TOURNAMENT_MOUNT,
        arena.tournament_standings_ref(tournament_id),
        standings,
    )
    complete = {
        "schema_id": arena.TOURNAMENT_SCHEMA_ID,
        "ok": all(bool(pair.get("ok")) for pair in pair_results),
        "app_name": APP_NAME,
        "artifact_volume_name": TOURNAMENT_VOLUME_NAME,
        "checkpoint_volume_name": CHECKPOINT_VOLUME_NAME,
        "tournament_id": tournament_id,
        "started_at": started_at,
        "ended_at": runs.utc_timestamp(),
        "checkpoint_count": len(checkpoints),
        "pair_count": len(pair_results),
        "game_count": len(game_results),
        "games_per_pair": int(spec.get("games_per_pair", arena.DEFAULT_GAMES_PER_PAIR)),
        "games_per_shard": int(spec.get("games_per_shard", arena.DEFAULT_GAMES_PER_SHARD)),
        "reuse_policies_per_shard": bool(
            spec.get(
                "reuse_policies_per_shard",
                arena.DEFAULT_REUSE_POLICIES_PER_SHARD,
            )
        ),
        "work_summary": work_summary,
        "pair_summary_refs": [
            pair.get("summary_ref") for pair in pair_results if pair.get("summary_ref")
        ],
        "standings_ref": arena.tournament_standings_ref(tournament_id).as_posix(),
    }
    arena.write_json_artifact(
        TOURNAMENT_MOUNT,
        arena.tournament_complete_ref(tournament_id),
        complete,
    )
    _write_tournament_manifest({**spec, "pair_count": len(pair_specs)}, status="completed")
    commit_error = _commit_volume(tournament_volume)
    if commit_error:
        complete["commit_error"] = commit_error
    if startup_commit_error:
        complete["startup_commit_error"] = startup_commit_error
    print(json.dumps(arena._to_plain(complete), sort_keys=True))
    return arena._to_plain({"complete": complete, "standings": standings})


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    cpu=1.0,
    memory=2048,
    timeout=12 * 60 * 60,
    max_containers=20,
)
def curvytron_rating_round(round_spec: dict[str, Any]) -> dict[str, Any]:
    _reload_volume(tournament_volume)
    spec = arena.normalize_rating_spec(round_spec)
    round_index = int(round_spec.get("round_index", 0))
    round_id = arena.rating_round_id(round_index)
    round_ratings_ref = arena.rating_round_ratings_ref(
        spec["tournament_id"],
        spec["rating_run_id"],
        round_id,
    )
    round_ratings_path = runs.volume_path(TOURNAMENT_MOUNT, round_ratings_ref)
    if round_ratings_path.exists():
        snapshot = _read_json(round_ratings_path)
        return _completed_rating_round_result_from_snapshot(
            TOURNAMENT_MOUNT,
            spec=spec,
            round_id=round_id,
            round_index=round_index,
            snapshot=snapshot,
            work_summary={
                "resumed_existing_round": True,
                "source": "ratings_json",
            },
            resumed_existing_round=True,
        )
    round_input_ref = arena.rating_round_input_ref(
        spec["tournament_id"],
        spec["rating_run_id"],
        round_id,
    )
    round_input_path = runs.volume_path(TOURNAMENT_MOUNT, round_input_ref)
    if round_input_path.exists():
        existing_input = _read_json(round_input_path)
        round_progress_ref = arena.rating_round_progress_ref(
            spec["tournament_id"],
            spec["rating_run_id"],
            round_id,
        )
        existing_progress = _read_json(runs.volume_path(TOURNAMENT_MOUNT, round_progress_ref))
        if str(existing_progress.get("status") or "") == "skipped":
            return arena._to_plain(
                {
                    "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
                    "status": "skipped",
                    "phase": str(existing_progress.get("phase") or "existing_round_skipped"),
                    "round_id": round_id,
                    "round_index": round_index,
                    "input_ref": round_input_ref.as_posix(),
                    "progress_ref": round_progress_ref.as_posix(),
                    "pair_count": int(
                        existing_input.get("pair_count") or existing_progress.get("pair_count") or 0
                    ),
                    "game_count": int(
                        existing_input.get("game_count") or existing_progress.get("game_count") or 0
                    ),
                    "completed_pair_count": int(existing_progress.get("completed_pair_count") or 0),
                    "completed_game_count": int(existing_progress.get("completed_game_count") or 0),
                    "work_summary": {
                        "resumed_existing_round": True,
                        "source": "existing_input_skipped",
                    },
                    "rated_pair_count": 0,
                }
            )
        try:
            _validate_existing_rating_round_input_matches_spec(
                existing_input,
                spec=spec,
                round_id=round_id,
                round_index=round_index,
            )
        except FileExistsError as exc:
            return arena._to_plain(
                {
                    "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
                    "status": "running_existing_round",
                    "phase": "existing_input_different_spec",
                    "round_id": round_id,
                    "round_index": round_index,
                    "input_ref": round_input_ref.as_posix(),
                    "progress_ref": round_progress_ref.as_posix(),
                    "pair_count": int(
                        existing_input.get("pair_count") or existing_progress.get("pair_count") or 0
                    ),
                    "game_count": int(
                        existing_input.get("game_count") or existing_progress.get("game_count") or 0
                    ),
                    "completed_pair_count": int(existing_progress.get("completed_pair_count") or 0),
                    "completed_game_count": int(existing_progress.get("completed_game_count") or 0),
                    "work_summary": {
                        "resumed_existing_round": True,
                        "source": "existing_input_different_spec",
                        "error": str(exc),
                    },
                    "rated_pair_count": 0,
                }
            )
        try:
            recovered = _reduce_rating_round_from_summaries(
                TOURNAMENT_MOUNT,
                tournament_id=spec["tournament_id"],
                rating_run_id=spec["rating_run_id"],
                round_id=round_id,
            )
        except ValueError as exc:
            if "rating round incomplete:" not in str(exc):
                raise
            progress = _read_json(
                runs.volume_path(
                    TOURNAMENT_MOUNT,
                    arena.rating_round_progress_ref(
                        spec["tournament_id"],
                        spec["rating_run_id"],
                        round_id,
                    ),
                )
            )
            return arena._to_plain(
                {
                    "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
                    "status": "running_existing_round",
                    "phase": "existing_input_incomplete",
                    "round_id": round_id,
                    "round_index": round_index,
                    "input_ref": round_input_ref.as_posix(),
                    "progress_ref": arena.rating_round_progress_ref(
                        spec["tournament_id"],
                        spec["rating_run_id"],
                        round_id,
                    ).as_posix(),
                    "pair_count": int(
                        existing_input.get("pair_count") or progress.get("pair_count") or 0
                    ),
                    "game_count": int(
                        existing_input.get("game_count") or progress.get("game_count") or 0
                    ),
                    "completed_pair_count": int(progress.get("completed_pair_count") or 0),
                    "completed_game_count": int(progress.get("completed_game_count") or 0),
                    "work_summary": {
                        "resumed_existing_round": True,
                        "source": "existing_input_running",
                        "error": str(exc),
                    },
                    "rated_pair_count": 0,
                }
            )
        commit_error = _commit_volume(tournament_volume)
        result = _completed_rating_round_result_from_snapshot(
            TOURNAMENT_MOUNT,
            spec=spec,
            round_id=round_id,
            round_index=round_index,
            snapshot=recovered["snapshot"],
            work_summary={
                "resumed_existing_round": True,
                "source": "existing_input_reduce",
                "reduced_from": recovered.get("reduced_from"),
                "commit_error": commit_error,
            },
            resumed_existing_round=True,
        )
        result["pair_history"] = recovered.get("pair_history")
        result["scheduler_state"] = recovered.get("scheduler_state")
        result["pair_count"] = int(recovered.get("pair_count") or 0)
        result["game_count"] = int(recovered.get("game_count") or 0)
        result["rated_pair_count"] = int(result["snapshot"].get("rated_pair_count") or 0)
        return arena._to_plain(result)
    existing_round_artifacts = _rating_round_existing_blocking_artifacts(
        TOURNAMENT_MOUNT,
        tournament_id=spec["tournament_id"],
        rating_run_id=spec["rating_run_id"],
        round_id=round_id,
    )
    if existing_round_artifacts:
        refs = ", ".join(ref.as_posix() for ref in existing_round_artifacts)
        raise FileExistsError(f"rating round already has artifacts: {refs}")
    previous_snapshot = round_spec.get("previous_snapshot")
    previous_pair_history = round_spec.get("pair_history")
    scheduler_state = round_spec.get("scheduler_state")
    started_at = runs.utc_timestamp()
    pair_specs = arena.build_rating_round_pair_specs(
        spec,
        previous_snapshot=previous_snapshot if isinstance(previous_snapshot, Mapping) else None,
        scheduler_state=scheduler_state if isinstance(scheduler_state, Mapping) else None,
        pair_history=previous_pair_history if isinstance(previous_pair_history, Mapping) else None,
        round_index=round_index,
    )
    input_payload = {
        "schema_id": arena.RATING_ROUND_SCHEMA_ID,
        "app_name": APP_NAME,
        "tournament_id": spec["tournament_id"],
        "rating_run_id": spec["rating_run_id"],
        "checkpoint_count": len(spec["checkpoints"]),
        "rating_spec_checkpoint_count": len(spec["checkpoints"]),
        "pool_hash": arena.rating_pool_hash(spec["checkpoints"]),
        "roster_hash": arena.rating_pool_hash(spec["checkpoints"]),
        "context_hash": arena.rating_context_hash(spec),
        "checkpoint_roster": arena.rating_roster_by_checkpoint(spec["checkpoints"]),
        "round_id": round_id,
        "round_index": round_index,
        "started_at": started_at,
        "rating_spec": arena._to_plain(spec),
        "previous_round_id": (
            previous_snapshot.get("round_id") if isinstance(previous_snapshot, Mapping) else None
        ),
        "pair_count": len(pair_specs),
        "game_count": sum(int(pair["games_per_pair"]) for pair in pair_specs),
        "pair_specs": arena._to_plain(pair_specs),
    }
    arena.write_json_artifact(
        TOURNAMENT_MOUNT,
        arena.rating_round_input_ref(
            spec["tournament_id"],
            spec["rating_run_id"],
            round_id,
        ),
        input_payload,
    )
    initial_progress, _initial_games = _rating_round_progress_payload(
        TOURNAMENT_MOUNT,
        tournament_id=spec["tournament_id"],
        rating_run_id=spec["rating_run_id"],
        round_id=round_id,
        game_results=[],
    )
    initial_progress["phase"] = "game_map_started"
    _write_rating_progress(TOURNAMENT_MOUNT, initial_progress)
    start_commit_error = _commit_volume(tournament_volume)
    _append_tournament_lineage_event(
        stage="rating_round_started",
        tournament_id=str(spec["tournament_id"]),
        rating_run_id=str(spec["rating_run_id"]),
        status="commit_error" if start_commit_error else "ok",
        round_id=round_id,
        round_index=round_index,
        input_ref=arena.rating_round_input_ref(
            spec["tournament_id"],
            spec["rating_run_id"],
            round_id,
        ).as_posix(),
        progress_ref=arena.rating_round_progress_ref(
            spec["tournament_id"],
            spec["rating_run_id"],
            round_id,
        ).as_posix(),
        pair_count=len(pair_specs),
        game_count=sum(int(pair["games_per_pair"]) for pair in pair_specs),
        commit_error=start_commit_error,
    )
    _commit_volume(tournament_volume)
    games_per_shard = int(spec.get("games_per_shard", arena.DEFAULT_GAMES_PER_SHARD))
    work_kind, work_specs = _build_game_work_specs(
        pair_specs,
        games_per_shard=games_per_shard,
        return_games=False,
    )
    result_detail_mode = "games"
    include_pair_results = True
    progress_game_results: list[dict[str, Any]] | None = None
    if work_kind == "game":
        game_results = list(
            curvytron_tournament_game.map(
                work_specs,
                order_outputs=False,
            )
        )
        game_results = arena._to_plain(game_results)
        progress_game_results = game_results
        work_summary = {
            "work_kind": work_kind,
            "work_count": len(work_specs),
            "games_per_shard": games_per_shard,
            "reuse_policies_per_shard": False,
            "parent_result_mode": "games",
        }
        games_by_battle: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for game in game_results:
            games_by_battle[str(game.get("battle_id") or "")].append(game)
        pair_results = []
        for pair in pair_specs:
            spec_ref = arena.battle_pair_spec_ref(
                pair["tournament_id"],
                pair["battle_id"],
            )
            arena.write_json_artifact(TOURNAMENT_MOUNT, spec_ref, pair)
            summary = arena.summarize_pair_results(
                pair,
                games_by_battle.get(pair["battle_id"], []),
            )
            summary["started_at"] = started_at
            summary["ended_at"] = runs.utc_timestamp()
            summary["work_summary"] = work_summary
            summary["rating_run_id"] = spec["rating_run_id"]
            summary["round_id"] = round_id
            summary["round_index"] = round_index
            summary["result_detail_mode"] = "games"
            summary["summary_ref"] = arena.battle_summary_ref(
                pair["tournament_id"],
                pair["battle_id"],
            ).as_posix()
            arena.write_json_artifact(
                TOURNAMENT_MOUNT,
                arena.battle_summary_ref(pair["tournament_id"], pair["battle_id"]),
                summary,
            )
            pair_results.append(summary)
        game_count = len(game_results)
    else:
        shard_results = list(
            curvytron_tournament_game_shard.map(
                work_specs,
                order_outputs=False,
            )
        )
        shard_results = arena._to_plain(shard_results)
        work_summary = {
            "work_kind": work_kind,
            "work_count": len(work_specs),
            "games_per_shard": games_per_shard,
            "reuse_policies_per_shard": bool(
                work_specs and work_specs[0].get("reuse_policies", True)
            ),
            "parent_result_mode": "shard_tallies",
        }
        pair_results, game_count = _summarize_pair_results_from_shard_tallies(
            mount=TOURNAMENT_MOUNT,
            pair_specs=pair_specs,
            shard_results=shard_results,
            started_at=started_at,
            work_summary=work_summary,
            rating_run_id=spec["rating_run_id"],
            round_id=round_id,
            round_index=round_index,
        )
        result_detail_mode = "shard_tallies"
        include_pair_results = False
    round_outputs = _write_rating_round_outputs(
        TOURNAMENT_MOUNT,
        spec=spec,
        round_id=round_id,
        round_index=round_index,
        pair_results=pair_results,
        pair_specs=pair_specs,
        game_count=game_count,
        started_at=started_at,
        previous_snapshot=(previous_snapshot if isinstance(previous_snapshot, Mapping) else None),
        previous_pair_history=(
            previous_pair_history if isinstance(previous_pair_history, Mapping) else None
        ),
        include_pair_results=include_pair_results,
        result_detail_mode=result_detail_mode,
    )
    slim_snapshot = round_outputs["snapshot"]
    if progress_game_results is None:
        progress = _rating_progress_from_pair_results(
            input_payload=input_payload,
            pair_results=pair_results,
            work_summary=work_summary,
            snapshot=slim_snapshot,
        )
    else:
        progress, _games_by_battle_from_volume = _rating_round_progress_payload(
            TOURNAMENT_MOUNT,
            tournament_id=spec["tournament_id"],
            rating_run_id=spec["rating_run_id"],
            round_id=round_id,
            game_results=progress_game_results,
        )
    progress["phase"] = "reduced"
    progress["status"] = "complete"
    progress["reduced_at"] = slim_snapshot.get("ended_at")
    progress["rated_pair_count"] = slim_snapshot.get("rated_pair_count")
    progress["work_summary"] = work_summary
    progress["pair_history_ref"] = slim_snapshot.get("pair_history_ref")
    progress["scheduler_state_ref"] = slim_snapshot.get("scheduler_state_ref")
    _write_rating_progress(TOURNAMENT_MOUNT, progress)
    commit_error = _commit_volume(tournament_volume)
    if commit_error:
        slim_snapshot["commit_error"] = commit_error
    _append_tournament_lineage_event(
        stage="rating_round_reduced",
        tournament_id=str(spec["tournament_id"]),
        rating_run_id=str(spec["rating_run_id"]),
        status="commit_error" if commit_error else "ok",
        round_id=round_id,
        round_index=round_index,
        ratings_ref=slim_snapshot.get("ratings_ref"),
        latest_ref=slim_snapshot.get("latest_ref"),
        pair_history_ref=slim_snapshot.get("pair_history_ref"),
        scheduler_state_ref=slim_snapshot.get("scheduler_state_ref"),
        pair_count=len(pair_results),
        game_count=game_count,
        rated_pair_count=slim_snapshot.get("rated_pair_count"),
        stable=slim_snapshot.get("stable"),
        max_abs_delta=slim_snapshot.get("max_abs_delta"),
        global_outputs_published=round_outputs.get("global_outputs_published"),
        commit_error=commit_error,
    )
    _commit_volume(tournament_volume)
    print(
        json.dumps(
            arena._to_plain(
                {
                    "rating_run_id": spec["rating_run_id"],
                    "round_id": round_id,
                    "pair_count": len(pair_results),
                    "game_count": game_count,
                    "rated_pair_count": slim_snapshot["rated_pair_count"],
                    "max_abs_delta": slim_snapshot["max_abs_delta"],
                    "stable": slim_snapshot["stable"],
                }
            ),
            sort_keys=True,
        )
    )
    return arena._to_plain(
        {
            "round_id": round_id,
            "round_index": round_index,
            "snapshot": slim_snapshot,
            "pair_history": round_outputs.get("pair_history"),
            "scheduler_state": round_outputs.get("scheduler_state"),
            "pair_count": len(pair_results),
            "game_count": game_count,
            "work_summary": work_summary,
            "rated_pair_count": slim_snapshot["rated_pair_count"],
        }
    )


def _spawn_rating_round_and_get(round_spec: Mapping[str, Any]) -> dict[str, Any]:
    call = curvytron_rating_round.spawn(dict(round_spec))
    return call.get()


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    cpu=1.0,
    memory=2048,
    timeout=24 * 60 * 60,
    max_containers=5,
)
def curvytron_rating_loop(rating_spec: dict[str, Any]) -> dict[str, Any]:
    _reload_volume(tournament_volume)
    spec = _rating_spec_with_latest_roster(TOURNAMENT_MOUNT, rating_spec)
    _write_tournament_marker(spec["tournament_id"])
    _write_tournament_manifest(
        {
            "tournament_id": spec["tournament_id"],
            "rating_run_id": spec["rating_run_id"],
            "mode": "rating",
            "checkpoint_count": len(spec["checkpoints"]),
            "plan_estimate": arena.estimate_tournament_plan(
                checkpoint_count=len(spec["checkpoints"]),
                games_per_pair=int(spec["games_per_pair"]),
                games_per_shard=int(spec["games_per_shard"]),
                reuse_policies_per_shard=bool(spec["reuse_policies_per_shard"]),
                ordered_pairs=bool(spec["ordered_pairs"]),
                include_self_pairs=bool(spec["include_self_pairs"]),
                pairs_per_round=spec.get("pairs_per_round"),
                save_gif=bool(spec["save_gif"]),
                gif_sample_games_per_pair=int(spec["gif_sample_games_per_pair"]),
                save_frames_npz=bool(spec["save_frames_npz"]),
            ),
        },
        status="rating",
    )
    _write_rating_config(spec)
    startup_commit_error = _commit_volume(tournament_volume)
    provisional_call = curvytron_rating_provisional_loop.spawn(
        {
            "tournament_id": spec["tournament_id"],
            "rating_run_id": spec["rating_run_id"],
        }
    )
    provisional_call_id = getattr(provisional_call, "object_id", None) or getattr(
        provisional_call, "id", None
    )
    start_state = _rating_loop_start_state(TOURNAMENT_MOUNT, spec)
    previous_snapshot = start_state["previous_snapshot"]
    previous_pair_history = start_state["previous_pair_history"]
    scheduler_state = start_state["scheduler_state"]
    start_round_index = int(start_state["start_round_index"])
    rounds = []
    started_at = runs.utc_timestamp()
    stop_when_stable = bool(rating_spec.get("stop_when_stable", False))
    for round_offset in range(int(spec["round_count"])):
        round_index = start_round_index + round_offset
        result = _spawn_rating_round_and_get(
            {
                **spec,
                "round_index": round_index,
                "previous_snapshot": previous_snapshot,
                "pair_history": previous_pair_history,
                "scheduler_state": scheduler_state,
            }
        )
        if str(result.get("status") or "") == "running_existing_round":
            rounds.append(
                {
                    "round_id": result["round_id"],
                    "round_index": result["round_index"],
                    "status": result["status"],
                    "pair_count": result.get("pair_count"),
                    "game_count": result.get("game_count"),
                    "completed_pair_count": result.get("completed_pair_count"),
                    "completed_game_count": result.get("completed_game_count"),
                    "work_summary": result.get("work_summary"),
                    "rated_pair_count": 0,
                }
            )
            break
        previous_snapshot = result["snapshot"]
        previous_pair_history = result.get("pair_history")
        scheduler_state = result.get("scheduler_state")
        latest_publish_decision: dict[str, Any] = {}
        if isinstance(previous_snapshot, Mapping):
            latest_publish_decision = _publish_rating_latest_snapshot_if_current(
                TOURNAMENT_MOUNT,
                tournament_id=str(spec["tournament_id"]),
                rating_run_id=str(spec["rating_run_id"]),
                snapshot=previous_snapshot,
            )
            if latest_publish_decision.get("publish"):
                _commit_volume(tournament_volume)
                try:
                    refresh_call = curvytron_training_candidate_refresh_tick.spawn(
                        {
                            "tournament_id": str(spec["tournament_id"]),
                            "rating_run_id": str(spec["rating_run_id"]),
                            "leaderboard_id": (
                                f"{spec['tournament_id']}-{spec['rating_run_id']}-training"
                            ),
                        }
                    )
                    latest_publish_decision["training_candidate_refresh_call_id"] = (
                        getattr(refresh_call, "object_id", None)
                        or getattr(refresh_call, "id", None)
                        or ""
                    )
                except Exception as exc:  # pragma: no cover - remote resilience.
                    latest_publish_decision["training_candidate_refresh_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
        rounds.append(
            {
                "round_id": result["round_id"],
                "round_index": result["round_index"],
                "pair_count": result["pair_count"],
                "game_count": result.get("game_count"),
                "work_summary": result.get("work_summary"),
                "rated_pair_count": result["rated_pair_count"],
                "ratings_ref": previous_snapshot.get("ratings_ref"),
                "pair_history_ref": previous_snapshot.get("pair_history_ref"),
                "scheduler_state_ref": previous_snapshot.get("scheduler_state_ref"),
                "max_abs_delta": previous_snapshot.get("max_abs_delta"),
                "stable": previous_snapshot.get("stable"),
                "latest_publish_decision": latest_publish_decision,
            }
        )
        if stop_when_stable and previous_snapshot.get("stable"):
            break
    complete = {
        "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
        "app_name": APP_NAME,
        "artifact_volume_name": TOURNAMENT_VOLUME_NAME,
        "checkpoint_volume_name": CHECKPOINT_VOLUME_NAME,
        "tournament_id": spec["tournament_id"],
        "rating_run_id": spec["rating_run_id"],
        "started_at": started_at,
        "ended_at": runs.utc_timestamp(),
        "start_round_index": start_round_index,
        "round_count_requested": int(spec["round_count"]),
        "round_count_completed": len(rounds),
        "continued_from_latest": bool(start_state["continued_from_latest"]),
        "provisional_rating_call_id": provisional_call_id,
        "rounds": rounds,
        "latest_ref": arena.rating_latest_ref(
            spec["tournament_id"],
            spec["rating_run_id"],
        ).as_posix(),
        "stable": previous_snapshot.get("stable") if previous_snapshot else False,
    }
    arena.write_json_artifact(
        TOURNAMENT_MOUNT,
        arena.rating_run_results_ref(spec["tournament_id"], spec["rating_run_id"]),
        complete,
    )
    commit_error = _commit_volume(tournament_volume)
    if commit_error:
        complete["commit_error"] = commit_error
    if startup_commit_error:
        complete["startup_commit_error"] = startup_commit_error
    print(json.dumps(arena._to_plain(complete), sort_keys=True))
    return arena._to_plain({"complete": complete, "latest": previous_snapshot})


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    cpu=1.0,
    memory=2048,
    timeout=10 * 60,
)
def curvytron_rating_startup_probe(rating_spec: dict[str, Any]) -> dict[str, Any]:
    _reload_volume(tournament_volume)
    spec = _rating_spec_with_latest_roster(TOURNAMENT_MOUNT, rating_spec)
    state = _rating_loop_start_state(TOURNAMENT_MOUNT, spec)
    pair_specs = arena.build_rating_round_pair_specs(
        spec,
        previous_snapshot=state["previous_snapshot"],
        pair_history=state["previous_pair_history"],
        scheduler_state=state["scheduler_state"],
        round_index=int(state["start_round_index"]),
    )
    prior_counts = [
        int(row.get("distinct_opponents") or 0)
        for row in (
            state.get("previous_snapshot", {}).get("ratings", [])
            if isinstance(state.get("previous_snapshot"), Mapping)
            else []
        )
        if isinstance(row, Mapping)
    ]
    return arena._to_plain(
        {
            "schema_id": "curvyzero_curvytron_rating_startup_probe/v0",
            "tournament_id": spec["tournament_id"],
            "rating_run_id": spec["rating_run_id"],
            "checkpoint_count": len(spec["checkpoints"]),
            "pool_hash": arena.rating_pool_hash(spec["checkpoints"]),
            "context_hash": arena.rating_context_hash(spec),
            "start_round_index": int(state["start_round_index"]),
            "continued_from_latest": bool(state["continued_from_latest"]),
            "pair_count": len(pair_specs),
            "schedule_reason_counts": dict(
                Counter(str(pair["schedule_reason"]) for pair in pair_specs)
            ),
            "prior_min_distinct_opponents": min(prior_counts) if prior_counts else None,
            "prior_max_distinct_opponents": max(prior_counts) if prior_counts else None,
        }
    )


_TRAINING_CANDIDATE_AUTO_REFRESH_STATE_PREFIX = "training_candidate_auto_refresh:"


def _training_candidate_auto_refresh_state_key(
    tournament_id: str,
    rating_run_id: str,
    leaderboard_id: str,
) -> str:
    return (
        f"{_TRAINING_CANDIDATE_AUTO_REFRESH_STATE_PREFIX}"
        f"{tournament_id}:{rating_run_id}:{leaderboard_id}"
    )


def _training_candidate_pointer_state(pointer_refs: Sequence[str]) -> list[dict[str, Any]]:
    states = []
    for pointer_ref in pointer_refs:
        pointer = _read_json_by_volume_ref(str(pointer_ref))
        source = pointer.get("source_leaderboard")
        if not isinstance(source, Mapping):
            source = {}
        states.append(
            {
                "pointer_ref": str(pointer_ref),
                "assignment_ref": pointer.get("assignment_ref"),
                "assignment_sha256": pointer.get("assignment_sha256"),
                "generation": _refresh_pointer_generation(pointer),
                "source_snapshot_sha256": source.get("snapshot_sha256"),
                "source_snapshot_id": source.get("snapshot_id"),
                "source_generation": source.get("generation"),
            }
        )
    return states


def _training_candidate_refresh_payload_from_config(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Merge the scheduled refresh payload with the mutable control config.

    Explicit caller payload wins. The scheduled tick normally passes no payload,
    so this lets the current training lane move to a new arena/pointer set by
    writing one control-volume config instead of redeploying the app.
    """

    merged = dict(payload)
    if merged.get("disable_control_config"):
        return merged, None
    config_ref = str(merged.get("config_ref") or TRAINING_CANDIDATE_REFRESH_CONFIG_REF)
    if not config_ref:
        return merged, None
    try:
        config = _read_json_by_volume_ref(config_ref)
    except FileNotFoundError:
        return merged, None
    if not isinstance(config, Mapping):
        raise ValueError(f"training candidate refresh config is not an object: {config_ref}")
    if config.get("schema_id") != "curvyzero_training_candidate_refresh_config/v0":
        raise ValueError(
            "training candidate refresh config has wrong schema_id: "
            f"{config.get('schema_id')!r}"
        )
    if bool(config.get("active", True)):
        for key, value in config.items():
            if key in {"schema_id", "active", "config_ref", "written_at"}:
                continue
            merged.setdefault(key, value)
    return merged, {"config_ref": config_ref, "active": bool(config.get("active", True))}


def _training_candidate_auto_refresh_cli_payload(
    *,
    tournament_id: str,
    rating_run_id: str,
    leaderboard_id: str,
    config_ref: str = "",
    refresh_pointer_refs: Sequence[str] = (),
    assignment_bank_run_id: str = "",
    assignment_bank_attempt_id: str = "",
    assignment_id_prefix: str = "",
    assignment_seed: int = 0,
    generation: int = -1,
    min_active_count: int = 1,
    allow_partial_assignment: bool = False,
    active_min_valid_games: int = 300,
    active_min_distinct_opponents: int = 20,
    max_active_rank: int = arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT,
) -> dict[str, Any]:
    """Build the manual auto-refresh payload without smuggling stale defaults.

    The scheduled tick is meant to be controlled by the mutable config JSON.
    The CLI path should only override that config when the caller explicitly
    passes a value.
    """

    payload: dict[str, Any] = {
        "tournament_id": tournament_id,
        "rating_run_id": rating_run_id,
        "leaderboard_id": leaderboard_id,
    }
    if config_ref:
        payload["config_ref"] = config_ref
    if refresh_pointer_refs:
        payload["refresh_pointers"] = [str(item) for item in refresh_pointer_refs]
    if assignment_bank_run_id:
        payload["assignment_bank_run_id"] = assignment_bank_run_id
    if assignment_bank_attempt_id:
        payload["assignment_bank_attempt_id"] = assignment_bank_attempt_id
    if assignment_id_prefix:
        payload["assignment_id_prefix"] = assignment_id_prefix
    if int(assignment_seed):
        payload["assignment_seed"] = int(assignment_seed)
    if int(generation) >= 0:
        payload["generation"] = int(generation)
    if int(min_active_count) != 1:
        payload["min_active_count"] = int(min_active_count)
    if allow_partial_assignment:
        payload["allow_partial_assignment"] = True
    if int(active_min_valid_games) != 300:
        payload["active_min_valid_games"] = int(active_min_valid_games)
    if int(active_min_distinct_opponents) != 20:
        payload["active_min_distinct_opponents"] = int(active_min_distinct_opponents)
    if int(max_active_rank) != arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT:
        payload["max_active_rank"] = int(max_active_rank)
    return payload


@app.function(
    image=image,
    volumes=_controller_volumes(),
    schedule=modal.Period(seconds=TRAINING_CANDIDATE_REFRESH_SECONDS),
    cpu=1.0,
    memory=2048,
    timeout=20 * 60,
    max_containers=1,
)
def curvytron_training_candidate_refresh_tick(
    spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(spec or {})
    tournament_id = CURRENT_TOURNAMENT_ID
    rating_run_id = CURRENT_RATING_RUN_ID
    leaderboard_id = f"{tournament_id}-{rating_run_id}-training"
    try:
        reload_errors = {
            "tournament": _reload_volume(tournament_volume),
            "control": _reload_volume(control_volume),
            "checkpoint": _reload_volume(checkpoint_volume),
        }
        reload_errors = {key: value for key, value in reload_errors.items() if value}
        if reload_errors:
            raise RuntimeError(f"training candidate auto-refresh reload failed: {reload_errors}")
        payload, config_state = _training_candidate_refresh_payload_from_config(payload)
        tournament_id = runs.clean_id(
            str(payload.get("tournament_id") or CURRENT_TOURNAMENT_ID),
            label="tournament_id",
        )
        rating_run_id = runs.clean_id(
            str(payload.get("rating_run_id") or CURRENT_RATING_RUN_ID),
            label="rating_run_id",
        )
        leaderboard_id = runs.clean_id(
            str(payload.get("leaderboard_id") or f"{tournament_id}-{rating_run_id}-training"),
            label="leaderboard_id",
        )
        refresh_pointers = payload.get("refresh_pointers") or list(
            TRAINING_CANDIDATE_REFRESH_POINTERS
        )
        if isinstance(refresh_pointers, (str, bytes)) or not isinstance(
            refresh_pointers,
            Sequence,
        ):
            raise ValueError("refresh_pointers must be a sequence of refs")
        refresh_pointer_refs = [str(item) for item in refresh_pointers if str(item).strip()]
        if not refresh_pointer_refs:
            raise ValueError("no refresh pointers configured")
        rating_snapshot = _read_best_rating_snapshot_for_run(
            TOURNAMENT_MOUNT,
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
            allow_live_provisional=bool(payload.get("allow_live_provisional", False)),
        )
        if not rating_snapshot:
            result = {
                "schema_id": "curvyzero_training_candidate_refresh_tick/v0",
                "status": "no_rating_snapshot",
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "leaderboard_id": leaderboard_id,
                "config": config_state,
                "created_at": runs.utc_timestamp(),
            }
            checkpoint_intake_state.put(
                _training_candidate_auto_refresh_state_key(
                    tournament_id,
                    rating_run_id,
                    leaderboard_id,
                ),
                arena._to_plain(result),
            )
            return arena._to_plain(result)
        if rating_snapshot.get("provisional") and not bool(
            payload.get("allow_live_provisional", False)
        ):
            raise ValueError("refusing provisional rating snapshot for auto-refresh")
        rating_spec = rating_snapshot.get("rating_spec")
        raw_decision_source_frames = (
            rating_spec.get("decision_source_frames") if isinstance(rating_spec, Mapping) else None
        )
        if _safe_int_or_none(raw_decision_source_frames) != 1:
            raise ValueError(
                "training candidate auto-refresh requires "
                "rating_spec.decision_source_frames=1; "
                f"got {raw_decision_source_frames!r}"
            )
        source_info = validate_rating_snapshot_source(rating_snapshot)
        source_snapshot_sha256 = str(source_info["rating_snapshot_sha256"])
        pointer_states = _training_candidate_pointer_state(refresh_pointer_refs)
        if pointer_states and all(
            str(state.get("source_snapshot_sha256") or "") == source_snapshot_sha256
            for state in pointer_states
        ):
            result = {
                "schema_id": "curvyzero_training_candidate_refresh_tick/v0",
                "status": "already_current",
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "leaderboard_id": leaderboard_id,
                "rating_source": source_info,
                "pointer_states": pointer_states,
                "config": config_state,
                "created_at": runs.utc_timestamp(),
            }
            checkpoint_intake_state.put(
                _training_candidate_auto_refresh_state_key(
                    tournament_id,
                    rating_run_id,
                    leaderboard_id,
                ),
                arena._to_plain(result),
            )
            return arena._to_plain(result)
        current_generations = [
            int(state["generation"])
            for state in pointer_states
            if state.get("generation") is not None
        ]
        generation = int(payload.get("generation") or (max(current_generations or [0]) + 1))
        round_index = int(rating_snapshot.get("round_index", 0) or 0)
        source_short_hash = arena._short_hash(source_snapshot_sha256, length=8)
        snapshot_id = runs.clean_id(
            str(
                payload.get("snapshot_id")
                or f"auto-r{round_index:06d}-g{generation}-{source_short_hash}"
            ),
            label="snapshot_id",
        )
        assignment_id_prefix = runs.clean_id(
            str(
                payload.get("assignment_id_prefix")
                or f"auto-r{round_index:06d}g{generation}-{source_short_hash}"
            ),
            label="assignment_id_prefix",
        )
        refresh = curvytron_training_candidate_refresh.local(
            {
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "leaderboard_id": leaderboard_id,
                "snapshot_id": snapshot_id,
                "active_min_valid_games": int(
                    payload.get(
                        "active_min_valid_games",
                        TRAINING_CANDIDATE_ACTIVE_MIN_VALID_GAMES,
                    )
                ),
                "active_min_distinct_opponents": int(
                    payload.get(
                        "active_min_distinct_opponents",
                        TRAINING_CANDIDATE_ACTIVE_MIN_DISTINCT_OPPONENTS,
                    )
                ),
                "max_active_rank": int(
                    payload.get("max_active_rank", TRAINING_CANDIDATE_MAX_ACTIVE_RANK)
                ),
                "generation": generation,
                "min_active_count": int(
                    payload.get("min_active_count", TRAINING_CANDIDATE_MIN_ACTIVE_COUNT)
                ),
                "assignment_bank_run_id": str(
                    payload.get("assignment_bank_run_id")
                    or TRAINING_CANDIDATE_ASSIGNMENT_BANK_RUN_ID
                ),
                "assignment_bank_attempt_id": str(
                    payload.get("assignment_bank_attempt_id")
                    or TRAINING_CANDIDATE_ASSIGNMENT_BANK_ATTEMPT_ID
                ),
                "assignment_id_prefix": assignment_id_prefix,
                "assignment_seed": int(
                    payload.get("assignment_seed", TRAINING_CANDIDATE_ASSIGNMENT_SEED)
                ),
                "refresh_pointers": refresh_pointer_refs,
                "allow_partial_assignment": bool(payload.get("allow_partial_assignment", False)),
            }
        )
        rewritten_pointers = [
            {
                "recipe_id": row.get("recipe_id"),
                "pointer_ref": row.get("pointer_ref"),
                "assignment_ref": row.get("assignment_ref"),
                "assignment_sha256": row.get("assignment_sha256"),
            }
            for row in refresh.get("rewritten_pointers", [])
            if isinstance(row, Mapping)
        ]
        result = {
            "schema_id": "curvyzero_training_candidate_refresh_tick/v0",
            "status": "refreshed",
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "leaderboard_id": leaderboard_id,
            "snapshot_id": refresh.get("snapshot_id"),
            "generation": generation,
            "rating_source": refresh.get("rating_source"),
            "row_count": refresh.get("row_count"),
            "active_count": refresh.get("active_count"),
            "rating_stable": refresh.get("rating_stable"),
            "rating_max_abs_delta": refresh.get("rating_max_abs_delta"),
            "rewritten_pointer_count": refresh.get("rewritten_pointer_count"),
            "rewritten_pointers": rewritten_pointers,
            "previous_pointer_states": pointer_states,
            "config": config_state,
            "created_at": runs.utc_timestamp(),
        }
    except Exception as exc:  # pragma: no cover - remote scheduler resilience.
        result = {
            "schema_id": "curvyzero_training_candidate_refresh_tick/v0",
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "leaderboard_id": leaderboard_id,
            "created_at": runs.utc_timestamp(),
        }
    checkpoint_intake_state.put(
        _training_candidate_auto_refresh_state_key(tournament_id, rating_run_id, leaderboard_id),
        arena._to_plain(result),
    )
    return arena._to_plain(result)


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    cpu=1.0,
    memory=2048,
    timeout=60 * 60,
    max_containers=10,
)
def curvytron_rating_progress(progress_spec: dict[str, Any]) -> dict[str, Any]:
    _reload_volume(tournament_volume)
    tournament_id = runs.clean_id(
        str(progress_spec["tournament_id"]),
        label="tournament_id",
    )
    rating_run_id = runs.clean_id(
        str(progress_spec.get("rating_run_id") or arena.DEFAULT_RATING_RUN_ID),
        label="rating_run_id",
    )
    round_index = int(progress_spec.get("round_index", 0) or 0)
    round_id = str(progress_spec.get("round_id") or arena.rating_round_id(round_index))
    try:
        progress, _games_by_battle = _rating_round_progress_payload(
            TOURNAMENT_MOUNT,
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
            round_id=round_id,
            load_summaries=bool(progress_spec.get("load_summaries", False)),
            pair_only=not bool(progress_spec.get("load_summaries", False)),
            count_game_summaries=bool(progress_spec.get("count_game_summaries", False)),
        )
    except FileNotFoundError:
        progress = _pending_rating_progress(
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
            round_id=round_id,
            round_index=round_index,
            reason="waiting_for_round_input",
        )
    _write_rating_progress(TOURNAMENT_MOUNT, progress)
    commit_error = _commit_volume(tournament_volume)
    if commit_error:
        progress["commit_error"] = commit_error
    print(
        json.dumps(
            arena._to_plain(
                {
                    "tournament_id": tournament_id,
                    "rating_run_id": rating_run_id,
                    "round_id": round_id,
                    "completed_game_count": progress.get("completed_game_count"),
                    "game_count": progress.get("game_count"),
                    "completed_pair_count": progress.get("completed_pair_count"),
                    "pair_count": progress.get("pair_count"),
                    "completion_fraction": progress.get("completion_fraction"),
                }
            ),
            sort_keys=True,
        )
    )
    return arena._to_plain(progress)


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    cpu=1.0,
    memory=2048,
    timeout=60 * 60,
    max_containers=3,
)
def curvytron_rating_provisional(provisional_spec: dict[str, Any]) -> dict[str, Any]:
    _reload_volume(tournament_volume)
    tournament_id = runs.clean_id(
        str(provisional_spec["tournament_id"]),
        label="tournament_id",
    )
    rating_run_id = runs.clean_id(
        str(provisional_spec.get("rating_run_id") or arena.DEFAULT_RATING_RUN_ID),
        label="rating_run_id",
    )
    snapshot = _build_provisional_rating_snapshot_for_run(
        TOURNAMENT_MOUNT,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    if not snapshot:
        progress = _read_rating_progress(
            TOURNAMENT_MOUNT,
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
        )
        payload = {
            "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "provisional": True,
            "status": "pending",
            "rating_count": 0,
            "progress": progress,
            "updated_at": runs.utc_timestamp(),
            "provisional_ref": _rating_provisional_latest_ref(
                tournament_id,
                rating_run_id,
            ).as_posix(),
        }
        arena.write_json_artifact(
            TOURNAMENT_MOUNT,
            _rating_provisional_latest_ref(tournament_id, rating_run_id),
            payload,
        )
        commit_error = _commit_volume(tournament_volume)
        if commit_error:
            payload["commit_error"] = commit_error
        return arena._to_plain(payload)
    artifacts = _write_provisional_rating_artifacts(
        TOURNAMENT_MOUNT,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        snapshot=snapshot,
    )
    slim = artifacts["snapshot"]
    commit_error = _commit_volume(tournament_volume)
    if commit_error:
        slim["commit_error"] = commit_error
    payload = {
        "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
        "tournament_id": tournament_id,
        "rating_run_id": rating_run_id,
        "provisional": True,
        "status": "written",
        "rating_count": len(slim.get("ratings") or []),
        "completed_pair_count": slim.get("completed_pair_count"),
        "completed_game_count": slim.get("completed_game_count"),
        "provisional_ref": slim.get("provisional_ref"),
        "battle_index_total": (artifacts.get("battle_index") or {}).get("total"),
        "updated_at": runs.utc_timestamp(),
    }
    if commit_error:
        payload["commit_error"] = commit_error
    print(json.dumps(arena._to_plain(payload), sort_keys=True))
    return arena._to_plain(payload)


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    cpu=2.0,
    memory=8192,
    timeout=24 * 60 * 60,
    max_containers=2,
)
def curvytron_rating_provisional_loop(loop_spec: dict[str, Any]) -> dict[str, Any]:
    tournament_id = runs.clean_id(
        str(loop_spec["tournament_id"]),
        label="tournament_id",
    )
    rating_run_id = runs.clean_id(
        str(loop_spec.get("rating_run_id") or arena.DEFAULT_RATING_RUN_ID),
        label="rating_run_id",
    )
    interval_seconds = max(
        15.0,
        float(loop_spec.get("interval_seconds", DEFAULT_PROVISIONAL_RATING_INTERVAL_SECONDS)),
    )
    max_seconds = max(
        interval_seconds,
        float(loop_spec.get("max_seconds", DEFAULT_PROVISIONAL_RATING_MAX_SECONDS)),
    )
    started = time.monotonic()
    writes = 0
    last_payload: dict[str, Any] = {}
    while True:
        _reload_volume(tournament_volume)
        snapshot = _build_provisional_rating_snapshot_for_run(
            TOURNAMENT_MOUNT,
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
        )
        latest = _read_rating_snapshot_for_run(
            TOURNAMENT_MOUNT,
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
        )
        if latest and latest.get("ratings"):
            last_payload = {
                "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "status": "final_snapshot_seen",
                "rating_count": len(latest.get("ratings") or []),
                "writes": writes,
                "updated_at": runs.utc_timestamp(),
            }
            print(json.dumps(arena._to_plain(last_payload), sort_keys=True))
            return arena._to_plain(last_payload)
        if snapshot and snapshot.get("ratings"):
            artifacts = _write_provisional_rating_artifacts(
                TOURNAMENT_MOUNT,
                tournament_id=tournament_id,
                rating_run_id=rating_run_id,
                snapshot=snapshot,
            )
            slim = artifacts["snapshot"]
            commit_error = _commit_volume(tournament_volume)
            writes += 1
            last_payload = {
                "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "status": "written",
                "rating_count": len(slim.get("ratings") or []),
                "completed_pair_count": slim.get("completed_pair_count"),
                "completed_game_count": slim.get("completed_game_count"),
                "battle_index_total": (artifacts.get("battle_index") or {}).get("total"),
                "writes": writes,
                "updated_at": runs.utc_timestamp(),
            }
            if commit_error:
                last_payload["commit_error"] = commit_error
            print(json.dumps(arena._to_plain(last_payload), sort_keys=True))
        elif not last_payload:
            last_payload = {
                "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "status": "waiting_for_shard_summaries",
                "rating_count": 0,
                "writes": writes,
                "updated_at": runs.utc_timestamp(),
            }
            arena.write_json_artifact(
                TOURNAMENT_MOUNT,
                _rating_provisional_latest_ref(tournament_id, rating_run_id),
                last_payload,
            )
            _commit_volume(tournament_volume)
        if time.monotonic() - started >= max_seconds:
            last_payload["status"] = "stopped_after_max_seconds"
            last_payload["updated_at"] = runs.utc_timestamp()
            print(json.dumps(arena._to_plain(last_payload), sort_keys=True))
            return arena._to_plain(last_payload)
        time.sleep(interval_seconds)


def _curvytron_rating_reduce_body(
    reduce_spec: Mapping[str, Any],
    *,
    reducer: str | None = None,
) -> dict[str, Any]:
    _reload_volume(tournament_volume)
    tournament_id = runs.clean_id(
        str(reduce_spec["tournament_id"]),
        label="tournament_id",
    )
    rating_run_id = runs.clean_id(
        str(reduce_spec.get("rating_run_id") or arena.DEFAULT_RATING_RUN_ID),
        label="rating_run_id",
    )
    round_index = int(reduce_spec.get("round_index", 0) or 0)
    round_id = str(reduce_spec.get("round_id") or arena.rating_round_id(round_index))
    result = _reduce_rating_round_from_summaries(
        TOURNAMENT_MOUNT,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        allow_partial=bool(reduce_spec.get("allow_partial", False)),
    )
    commit_error = _commit_volume(tournament_volume)
    if commit_error:
        result["commit_error"] = commit_error
    snapshot = result.get("snapshot") if isinstance(result.get("snapshot"), Mapping) else {}
    event_fields = {
        "round_id": round_id,
        "round_index": round_index,
        "ratings_ref": snapshot.get("ratings_ref"),
        "latest_ref": snapshot.get("latest_ref"),
        "pair_history_ref": snapshot.get("pair_history_ref"),
        "scheduler_state_ref": snapshot.get("scheduler_state_ref"),
        "pair_count": result.get("pair_count"),
        "game_count": result.get("game_count"),
        "rated_pair_count": snapshot.get("rated_pair_count"),
        "stable": snapshot.get("stable"),
        "max_abs_delta": snapshot.get("max_abs_delta"),
        "global_outputs_published": result.get("global_outputs_published"),
        "commit_error": commit_error,
        "reduced_from": result.get("reduced_from"),
    }
    if reducer:
        event_fields["reducer"] = reducer
    _append_tournament_lineage_event(
        stage="rating_round_reduced",
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        status="commit_error" if commit_error else "ok",
        **event_fields,
    )
    _commit_volume(tournament_volume)
    print_payload = {
        "tournament_id": tournament_id,
        "rating_run_id": rating_run_id,
        "round_id": round_id,
        "pair_count": result.get("pair_count"),
        "game_count": result.get("game_count"),
        "rated_pair_count": (
            result.get("snapshot", {}).get("rated_pair_count")
            if isinstance(result.get("snapshot"), Mapping)
            else None
        ),
    }
    if reducer:
        print_payload["reducer"] = reducer
    print(
        json.dumps(
            arena._to_plain(print_payload),
            sort_keys=True,
        )
    )
    return arena._to_plain(result)


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    cpu=0.25,
    memory=1024,
    timeout=4 * 60 * 60,
    max_containers=20,
)
def curvytron_rating_reduce(reduce_spec: dict[str, Any]) -> dict[str, Any]:
    return _curvytron_rating_reduce_body(reduce_spec)


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    cpu=0.25,
    memory=1024,
    timeout=4 * 60 * 60,
    max_containers=20,
)
def curvytron_rating_reduce_rescue(reduce_spec: dict[str, Any]) -> dict[str, Any]:
    """Use a separate Modal function queue when the normal reducer is wedged."""

    return _curvytron_rating_reduce_body(reduce_spec, reducer="rescue")


@app.function(
    image=image,
    volumes=_controller_volumes(),
    timeout=30 * 60,
    cpu=1.0,
    memory=1024,
)
def curvytron_feedback_loop_reduce_rescue(
    reduce_spec: dict[str, Any],
) -> dict[str, Any]:
    """Reduce through the control-plane lane when rating reducers cannot start."""

    _reload_volume(control_volume)
    return _curvytron_rating_reduce_body(reduce_spec, reducer="control_rescue")


def _list_tournaments(mount: Path) -> list[dict[str, Any]]:
    base = runs.volume_path(mount, arena.TOURNAMENT_BASE_REF)
    rows = []
    if not base.exists():
        return []
    for marker in base.glob(f"*/{arena.TOURNAMENT_RUN_MARKER_FILENAME}"):
        root = marker.parent
        tournament_id = root.name
        manifest = _read_json(root / "tournament.json")
        complete = _read_json(root / "complete.json")
        updated_path = root / "complete.json" if (root / "complete.json").exists() else marker
        rows.append(
            {
                "tournament_id": tournament_id,
                "is_current": tournament_id == CURRENT_TOURNAMENT_ID,
                "status": complete.get("status") or manifest.get("status"),
                "updated_ts": updated_path.stat().st_mtime,
                "updated_at": complete.get("ended_at") or manifest.get("updated_at"),
                "pair_count": complete.get("pair_count"),
                "checkpoint_count": complete.get("checkpoint_count"),
            }
        )
    rows.sort(key=lambda row: (-float(row.get("updated_ts") or 0), row["tournament_id"]))
    return rows


def _battle_checkpoint_ids(row: Mapping[str, Any]) -> list[str]:
    players = row.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return []
    ids = []
    for player in players:
        if isinstance(player, Mapping) and player.get("checkpoint_id"):
            ids.append(str(player["checkpoint_id"]))
    return ids


def _battle_matches_checkpoint(row: Mapping[str, Any], checkpoint_id: str) -> bool:
    if not checkpoint_id:
        return True
    return checkpoint_id in _battle_checkpoint_ids(row)


def _list_battle_index(
    mount: Path,
    *,
    tournament_id: str,
    limit: int,
    offset: int,
    checkpoint_id: str = "",
) -> dict[str, Any]:
    clean_id = runs.clean_id(tournament_id, label="tournament_id")
    selected_checkpoint_id = str(checkpoint_id or "").strip()
    source = "battle_index"
    if selected_checkpoint_id:
        index_path = runs.volume_path(
            mount,
            arena.tournament_checkpoint_battle_index_ref(
                clean_id,
                selected_checkpoint_id,
            ),
        )
        if index_path.exists():
            source = "checkpoint_battle_index"
        else:
            index_path = runs.volume_path(mount, arena.tournament_battle_index_ref(clean_id))
    else:
        index_path = runs.volume_path(mount, arena.tournament_battle_index_ref(clean_id))
    index = _read_json(index_path)
    index_rows = index.get("rows")
    if not isinstance(index_rows, list):
        return {
            "rows": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "has_older": False,
            "has_newer": offset > 0,
            "source": "battle_index_missing",
            "checkpoint_id": selected_checkpoint_id,
        }
    rows = [dict(row) for row in index_rows if isinstance(row, Mapping)]
    if selected_checkpoint_id:
        rows = [row for row in rows if _battle_matches_checkpoint(row, selected_checkpoint_id)]
    rows.sort(
        key=lambda row: (
            -float(row.get("updated_ts") or 0.0),
            str(row.get("battle_id") or ""),
        )
    )
    total = len(rows)
    page = rows[offset : offset + limit]
    return {
        "rows": page,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_older": offset + limit < total,
        "has_newer": offset > 0,
        "source": source,
        "checkpoint_id": selected_checkpoint_id,
    }


def _read_battle_index_row(
    mount: Path,
    *,
    tournament_id: str,
    battle_id: str,
) -> tuple[dict[str, Any], str]:
    clean_id = runs.clean_id(tournament_id, label="tournament_id")
    selected_battle_id = str(battle_id or "").strip()
    if not selected_battle_id:
        return {}, "battle_index_missing"
    index_path = runs.volume_path(mount, arena.tournament_battle_index_ref(clean_id))
    index = _read_json(index_path)
    index_rows = index.get("rows")
    if not isinstance(index_rows, list):
        return {}, "battle_index_missing"
    for row in index_rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("battle_id") or "") == selected_battle_id:
            return dict(row), "battle_index"
    return {}, "battle_index"


def _list_battles(
    mount: Path,
    *,
    tournament_id: str,
    limit: int,
    offset: int,
    checkpoint_id: str = "",
) -> dict[str, Any]:
    clean_id = runs.clean_id(tournament_id, label="tournament_id")
    selected_checkpoint_id = str(checkpoint_id or "").strip()
    index_page = _list_battle_index(
        mount,
        tournament_id=clean_id,
        limit=limit,
        offset=offset,
        checkpoint_id=selected_checkpoint_id,
    )
    if index_page["source"] in {"battle_index", "checkpoint_battle_index"}:
        return index_page
    root = runs.volume_path(mount, arena.tournament_root_ref(clean_id)) / "battles"
    summaries = sorted(
        root.glob("*/battle.json") if root.exists() else [],
        key=lambda path: (-path.stat().st_mtime, path.as_posix()),
    )
    rows = []
    paths = summaries if selected_checkpoint_id else summaries[offset : offset + limit]
    for path in paths:
        row = _read_json(path)
        if not row:
            continue
        if selected_checkpoint_id and not _battle_matches_checkpoint(
            row,
            selected_checkpoint_id,
        ):
            continue
        first_gif_ref = row.get("first_gif_ref")
        rows.append(
            {
                "tournament_id": row.get("tournament_id"),
                "battle_id": row.get("battle_id"),
                "players": row.get("players"),
                "tally": row.get("tally"),
                "ok": row.get("ok"),
                "summary_ref": runs.file_ref(path, mount=mount),
                "first_gif_ref": first_gif_ref,
                "updated_ts": path.stat().st_mtime,
            }
        )
    total = len(rows) if selected_checkpoint_id else len(summaries)
    page = rows[offset : offset + limit] if selected_checkpoint_id else rows
    return {
        "rows": page,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_older": offset + limit < total,
        "has_newer": offset > 0,
        "source": "battle_scan",
        "checkpoint_id": selected_checkpoint_id,
    }


def _list_rating_runs(mount: Path, *, tournament_id: str) -> list[dict[str, Any]]:
    clean_id = runs.clean_id(tournament_id, label="tournament_id")
    root = runs.volume_path(mount, arena.tournament_root_ref(clean_id)) / "ratings"
    rows = []
    for rating_root in sorted(root.iterdir(), key=lambda path: path.name) if root.exists() else []:
        if not rating_root.is_dir():
            continue
        rating_run_id = rating_root.name
        latest_path = rating_root / "latest.json"
        progress_path = rating_root / "progress.json"
        config_path = rating_root / "config.json"
        latest = _read_json(latest_path)
        progress = _read_json(progress_path)
        config = _read_json(config_path)
        if not latest and not progress and not config:
            continue
        source = progress or latest
        config_spec = (
            config.get("rating_spec") if isinstance(config.get("rating_spec"), Mapping) else {}
        )
        checkpoint_count = (
            latest.get("checkpoint_count")
            or progress.get("checkpoint_count")
            or (
                len(config_spec.get("checkpoints", []))
                if isinstance(config_spec.get("checkpoints"), list)
                else None
            )
        )
        updated_paths = [
            path for path in (latest_path, progress_path, config_path) if path.exists()
        ]
        updated_path = max(updated_paths, key=lambda path: path.stat().st_mtime)
        rows.append(
            {
                "tournament_id": clean_id,
                "rating_run_id": rating_run_id,
                "is_current": rating_run_id == CURRENT_RATING_RUN_ID,
                "updated_ts": updated_path.stat().st_mtime,
                "updated_at": (
                    progress.get("updated_at")
                    or latest.get("created_at")
                    or latest.get("ended_at")
                    or config.get("created_at")
                ),
                "status": progress.get("status") or ("complete" if latest else "configured"),
                "phase": progress.get("phase"),
                "round_id": source.get("round_id"),
                "round_index": source.get("round_index"),
                "checkpoint_count": checkpoint_count,
                "rated_pair_count": latest.get("rated_pair_count"),
                "max_abs_delta": latest.get("max_abs_delta"),
                "stable": latest.get("stable"),
                "pair_count": progress.get("pair_count"),
                "game_count": progress.get("game_count"),
                "completed_game_count": progress.get("completed_game_count"),
                "estimated_seen_game_count": progress.get("estimated_seen_game_count"),
                "completed_pair_count": progress.get("completed_pair_count"),
                "started_pair_count": progress.get("started_pair_count"),
                "completion_fraction": progress.get("completion_fraction"),
                "estimated_completion_fraction": progress.get("estimated_completion_fraction"),
                "latest_ref": (
                    runs.file_ref(latest_path, mount=mount) if latest_path.exists() else None
                ),
                "progress_ref": (
                    runs.file_ref(progress_path, mount=mount) if progress_path.exists() else None
                ),
                "config_ref": (
                    runs.file_ref(config_path, mount=mount) if config_path.exists() else None
                ),
                "formula_version": latest.get("formula_version")
                or progress.get("formula_version")
                or config.get("formula_version"),
            }
        )
    rows.sort(key=lambda row: (-float(row.get("updated_ts") or 0), row["rating_run_id"]))
    return rows


def _list_rating_latest_runs(mount: Path, *, tournament_id: str) -> list[dict[str, Any]]:
    clean_id = runs.clean_id(tournament_id, label="tournament_id")
    root = runs.volume_path(mount, arena.tournament_root_ref(clean_id)) / "ratings"
    rows = []
    for rating_root in sorted(root.iterdir(), key=lambda path: path.name) if root.exists() else []:
        if not rating_root.is_dir():
            continue
        rating_run_id = rating_root.name
        latest_path = rating_root / "latest.json"
        latest = _read_json(latest_path)
        if not latest:
            continue
        rows.append(
            {
                "tournament_id": clean_id,
                "rating_run_id": rating_run_id,
                "updated_ts": latest_path.stat().st_mtime,
                "updated_at": latest.get("created_at") or latest.get("ended_at"),
                "status": "complete",
                "round_id": latest.get("round_id"),
                "round_index": latest.get("round_index"),
                "checkpoint_count": latest.get("checkpoint_count"),
                "rated_pair_count": latest.get("rated_pair_count"),
                "max_abs_delta": latest.get("max_abs_delta"),
                "stable": latest.get("stable"),
                "latest_ref": runs.file_ref(latest_path, mount=mount),
                "formula_version": latest.get("formula_version"),
            }
        )
    rows.sort(key=lambda row: (-float(row.get("updated_ts") or 0), row["rating_run_id"]))
    return rows


def _default_rating_run_id(rows: list[dict[str, Any]], requested: str) -> str:
    if requested and requested != "latest":
        clean = runs.clean_id(requested, label="rating_run_id")
        if any(row["rating_run_id"] == clean for row in rows):
            return clean
    for row in rows:
        if row.get("is_current") and row.get("rating_run_id") == CURRENT_RATING_RUN_ID:
            return CURRENT_RATING_RUN_ID
    if any(row.get("rating_run_id") == CURRENT_RATING_RUN_ID for row in rows):
        return CURRENT_RATING_RUN_ID
    return str(rows[0]["rating_run_id"]) if rows else ""


def _read_rating_snapshot(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str = "latest",
) -> dict[str, Any]:
    rating_runs = _list_rating_runs(mount, tournament_id=tournament_id)
    selected = _default_rating_run_id(rating_runs, rating_run_id)
    if not selected:
        return {}
    ref = arena.rating_latest_ref(tournament_id, selected)
    return _read_json(runs.volume_path(mount, ref))


def _read_rating_snapshot_for_run(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
) -> dict[str, Any]:
    if not rating_run_id:
        return {}
    ref = arena.rating_latest_ref(tournament_id, rating_run_id)
    return _read_json(runs.volume_path(mount, ref))


def _read_rating_progress(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str = "latest",
) -> dict[str, Any]:
    rating_runs = _list_rating_runs(mount, tournament_id=tournament_id)
    selected = _default_rating_run_id(rating_runs, rating_run_id)
    if not selected:
        return {}
    ref = arena.rating_progress_ref(tournament_id, selected)
    return _read_json(runs.volume_path(mount, ref))


def _read_live_rating_progress(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str = "latest",
) -> dict[str, Any]:
    """Read cheap live progress without writing back to the Volume."""

    rating_runs = _list_rating_runs(mount, tournament_id=tournament_id)
    selected = _default_rating_run_id(rating_runs, rating_run_id)
    if not selected:
        return {}
    stored = _read_json(runs.volume_path(mount, arena.rating_progress_ref(tournament_id, selected)))
    if not stored:
        return {}
    if stored.get("status") == "complete":
        return stored
    round_id = str(
        stored.get("round_id") or arena.rating_round_id(int(stored.get("round_index", 0) or 0))
    )
    try:
        progress, _games_by_battle = _rating_round_progress_payload(
            mount,
            tournament_id=tournament_id,
            rating_run_id=selected,
            round_id=round_id,
            load_summaries=False,
            pair_only=True,
        )
    except FileNotFoundError:
        return stored
    return progress


def _read_cached_live_rating_progress(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str = "latest",
    force_live: bool = False,
) -> dict[str, Any]:
    cache_key = f"live-progress:{mount}:{tournament_id}:{rating_run_id}"
    cached = (
        None
        if force_live
        else _web_cache_get(
            cache_key,
            ttl_seconds=WEB_PROGRESS_CACHE_TTL_SECONDS,
        )
    )
    if isinstance(cached, dict):
        return cached
    if force_live:
        progress = _read_live_rating_progress(
            mount,
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
        )
    else:
        progress = _read_rating_progress(
            mount,
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
        )
    progress = _merge_progress_with_provisional_snapshot(
        mount,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        progress=progress,
    )
    return _web_cache_set(
        cache_key,
        progress,
    )


def _maybe_spawn_rating_progress_refresh(
    *,
    tournament_id: str,
    rating_run_id: str,
    progress: Mapping[str, Any],
    min_interval_seconds: float = 30.0,
) -> str:
    if not tournament_id or not rating_run_id:
        return ""
    if progress.get("status") == "complete":
        return ""
    cache_key = f"progress-refresh-spawn:{tournament_id}:{rating_run_id}"
    cached = _web_cache_get(cache_key, ttl_seconds=min_interval_seconds)
    if cached:
        return str(cached)
    round_index = int(progress.get("round_index", 0) or 0)
    round_id = str(progress.get("round_id") or arena.rating_round_id(round_index))
    try:
        call = curvytron_rating_progress.spawn(
            {
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "round_id": round_id,
                "round_index": round_index,
                "load_summaries": False,
            }
        )
        call_id = getattr(call, "object_id", "") or str(call)
    except Exception as exc:  # pragma: no cover - web best-effort refresh.
        call_id = f"{type(exc).__name__}: {exc}"
    _web_cache_set(cache_key, call_id)
    return str(call_id)


def _read_rating_config_for_run(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
) -> dict[str, Any]:
    if not tournament_id or not rating_run_id:
        return {}
    return _read_json(
        runs.volume_path(mount, arena.rating_config_ref(tournament_id, rating_run_id))
    )


def _rating_provisional_latest_ref(tournament_id: str, rating_run_id: str) -> Path:
    return arena.rating_provisional_latest_ref(tournament_id, rating_run_id)


def _read_provisional_rating_snapshot_for_run(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
) -> dict[str, Any]:
    if not tournament_id or not rating_run_id:
        return {}
    return _read_json(
        runs.volume_path(
            mount,
            _rating_provisional_latest_ref(tournament_id, rating_run_id),
        )
    )


def _merge_progress_with_provisional_snapshot(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
    progress: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(progress or {})
    if merged.get("status") == "complete":
        return arena._to_plain(merged)
    snapshot = _read_provisional_rating_snapshot_for_run(
        mount,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    if not snapshot or not snapshot.get("ratings"):
        return arena._to_plain(merged)
    completed_pairs = int(snapshot.get("completed_pair_count") or 0)
    completed_games = int(snapshot.get("completed_game_count") or 0)
    pair_count = int(
        merged.get("pair_count") or snapshot.get("total_pair_count") or completed_pairs or 0
    )
    game_count = int(
        merged.get("game_count") or snapshot.get("total_game_count") or completed_games or 0
    )
    merged_completed_pairs = max(
        int(merged.get("completed_pair_count") or 0),
        completed_pairs,
    )
    merged_completed_games = max(
        int(merged.get("completed_game_count") or 0),
        completed_games,
    )
    estimated_seen_games = max(
        int(merged.get("estimated_seen_game_count") or 0),
        merged_completed_games,
    )
    merged.update(
        {
            "status": merged.get("status") or "running",
            "phase": "games_running_with_provisional_ratings",
            "provisional_ratings_written": True,
            "provisional_rating_count": len(snapshot.get("ratings") or []),
            "provisional_ref": snapshot.get("provisional_ref"),
            "completed_pair_count": merged_completed_pairs,
            "started_pair_count": max(
                int(merged.get("started_pair_count") or 0),
                merged_completed_pairs,
            ),
            "completed_game_count": merged_completed_games,
            "estimated_seen_game_count": estimated_seen_games,
            "pair_count": pair_count,
            "game_count": game_count,
            "completion_fraction": (
                float(merged_completed_games) / float(game_count) if game_count else 0.0
            ),
            "estimated_completion_fraction": (
                float(estimated_seen_games) / float(game_count) if game_count else 0.0
            ),
            "updated_at": snapshot.get("updated_at") or merged.get("updated_at"),
        }
    )
    return arena._to_plain(merged)


def _live_pair_results_from_shard_summaries(
    mount: Path,
    *,
    input_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    pair_specs = [
        dict(pair)
        for pair in input_payload.get("pair_specs", [])
        if isinstance(pair, Mapping) and pair.get("battle_id")
    ]
    pair_by_battle = {str(pair["battle_id"]): pair for pair in pair_specs}
    shards_by_battle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tournament_id = str(input_payload["tournament_id"])
    battles_root = runs.volume_path(mount, arena.tournament_root_ref(tournament_id)) / "battles"
    if battles_root.exists():
        for path in battles_root.glob("*/shards/*/summary.json"):
            battle_id = path.parents[2].name
            if battle_id not in pair_by_battle:
                continue
            shard = _read_json(path)
            if not shard:
                continue
            shard.setdefault("summary_ref", runs.file_ref(path, mount=mount))
            shards_by_battle[battle_id].append(shard)

    pair_results: list[dict[str, Any]] = []
    total_games = 0
    for pair in pair_specs:
        battle_id = str(pair["battle_id"])
        shards = sorted(
            shards_by_battle.get(battle_id, []),
            key=lambda shard: (
                int(shard.get("shard_index", 0) or 0),
                str(shard.get("shard_id") or ""),
            ),
        )
        if not shards:
            continue
        summary = _summarize_live_pair_from_shards(
            pair,
            shards=shards,
            rating_run_id=str(input_payload.get("rating_run_id") or ""),
            round_id=str(input_payload.get("round_id") or ""),
            round_index=int(input_payload.get("round_index", 0) or 0),
        )
        tally = summary.get("tally") if isinstance(summary.get("tally"), Mapping) else {}
        total_games += int(tally.get("game_count") or 0)
        pair_results.append(summary)
    return arena._to_plain(pair_results), int(total_games)


def _build_provisional_rating_snapshot_for_run(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
) -> dict[str, Any]:
    if not tournament_id or not rating_run_id:
        return {}
    cache_key = f"provisional-rating:{mount}:{tournament_id}:{rating_run_id}"
    cached = _web_cache_get(
        cache_key,
        ttl_seconds=WEB_PROVISIONAL_RATING_CACHE_TTL_SECONDS,
    )
    if isinstance(cached, dict):
        return cached

    config = _read_rating_config_for_run(
        mount,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    spec = config.get("rating_spec") if isinstance(config.get("rating_spec"), Mapping) else {}
    if not spec:
        return {}
    progress = _read_rating_progress(
        mount,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    round_index = int(progress.get("round_index", 0) or 0)
    round_id = str(progress.get("round_id") or arena.rating_round_id(round_index))
    try:
        input_payload = _read_rating_round_input(
            mount,
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
            round_id=round_id,
        )
    except FileNotFoundError:
        return {}
    pair_results, game_count = _live_pair_results_from_shard_summaries(
        mount,
        input_payload=input_payload,
    )
    if not pair_results:
        return {}
    previous_snapshot = _previous_rating_snapshot(
        mount,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_index=round_index,
    )
    snapshot = arena.rating_snapshot_from_pair_results(
        pair_results=pair_results,
        rating_spec=spec,
        previous_snapshot=previous_snapshot,
        round_index=round_index,
        created_at=runs.utc_timestamp(),
    )
    snapshot["provisional"] = True
    snapshot["source"] = "live_shard_summaries"
    snapshot["status"] = "running"
    snapshot["game_count"] = int(game_count)
    snapshot["total_pair_count"] = int(input_payload.get("pair_count") or 0)
    snapshot["total_game_count"] = int(input_payload.get("game_count") or 0)
    snapshot["completed_pair_count"] = len(pair_results)
    snapshot["completed_game_count"] = int(game_count)
    snapshot["latest_ref"] = arena.rating_latest_ref(tournament_id, rating_run_id).as_posix()
    snapshot["provisional_ref"] = _rating_provisional_latest_ref(
        tournament_id,
        rating_run_id,
    ).as_posix()
    snapshot["progress_ref"] = arena.rating_progress_ref(tournament_id, rating_run_id).as_posix()
    snapshot["live_pair_results"] = pair_results
    return _web_cache_set(cache_key, arena._to_plain(snapshot))


def _read_best_rating_snapshot_for_run(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
    allow_live_provisional: bool = False,
) -> dict[str, Any]:
    snapshot = _read_rating_snapshot_for_run(
        mount,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    if snapshot:
        return snapshot
    provisional = _read_provisional_rating_snapshot_for_run(
        mount,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    if not allow_live_provisional:
        return provisional
    live = _build_provisional_rating_snapshot_for_run(
        mount,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    if not live:
        return provisional
    if not provisional:
        return live
    live_games = int(live.get("completed_game_count") or live.get("game_count") or 0)
    provisional_games = int(
        provisional.get("completed_game_count") or provisional.get("game_count") or 0
    )
    return live if live_games >= provisional_games else provisional


def _battle_index_from_pair_results(
    pair_results: Sequence[Mapping[str, Any]],
    *,
    checkpoint_id: str = "",
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    selected_checkpoint_id = str(checkpoint_id or "").strip()
    rows = []
    for summary in pair_results:
        if not isinstance(summary, Mapping):
            continue
        row = _compact_battle_index_row(summary)
        if selected_checkpoint_id and not _battle_matches_checkpoint(
            row,
            selected_checkpoint_id,
        ):
            continue
        rows.append(row)
    rows.sort(
        key=lambda row: (
            int(row.get("pair_index", 0) or 0),
            str(row.get("battle_id") or ""),
        )
    )
    total = len(rows)
    page = rows[offset : offset + limit]
    return {
        "rows": arena._to_plain(page),
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_older": offset + limit < total,
        "has_newer": offset > 0,
        "source": "live_shard_tallies",
        "checkpoint_id": selected_checkpoint_id,
    }


def _safe_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _checkpoint_live_shard_battles(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
    checkpoint_id: str,
    limit: int = MAX_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    if not tournament_id or not rating_run_id or not checkpoint_id:
        return {
            "rows": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "has_older": False,
            "has_newer": offset > 0,
            "source": "checkpoint_live_shards_unavailable",
            "checkpoint_id": checkpoint_id,
        }
    progress = _read_rating_progress(
        mount,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    round_index = int(progress.get("round_index", 0) or 0)
    round_id = str(progress.get("round_id") or arena.rating_round_id(round_index))
    try:
        input_payload = _read_rating_round_input(
            mount,
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
            round_id=round_id,
        )
    except FileNotFoundError:
        return {
            "rows": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "has_older": False,
            "has_newer": offset > 0,
            "source": "checkpoint_live_shards_missing_round_input",
            "checkpoint_id": checkpoint_id,
        }
    pair_results = []
    for pair in input_payload.get("pair_specs", []):
        if not isinstance(pair, Mapping) or not pair.get("battle_id"):
            continue
        pair_row = dict(pair)
        if not _battle_matches_checkpoint(pair_row, checkpoint_id):
            continue
        pair_row["summary_ref"] = arena.battle_summary_ref(
            tournament_id,
            str(pair_row["battle_id"]),
        ).as_posix()
        pair_row["rating_run_id"] = rating_run_id
        pair_row["round_id"] = round_id
        pair_row["round_index"] = round_index
        pair_results.append(pair_row)
    page = _battle_index_from_pair_results(
        pair_results,
        checkpoint_id=checkpoint_id,
        limit=limit,
        offset=offset,
    )
    page["source"] = "checkpoint_round_input"
    page["round_id"] = round_id
    page["round_index"] = round_index
    return arena._to_plain(page)


def _enrich_battle_row_from_live_shards(
    mount: Path,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    tournament_id = str(row.get("tournament_id") or "")
    battle_id = str(row.get("battle_id") or "")
    if not tournament_id or not battle_id:
        return dict(row)
    if row.get("tally") and (row.get("sample_gif_refs") or row.get("first_gif_ref")):
        return dict(row)
    shards = _read_battle_shard_summaries(
        mount,
        tournament_id=tournament_id,
        battle_id=battle_id,
    )
    if not shards:
        return dict(row)
    summary = _summarize_live_pair_from_shards(
        row,
        shards=shards,
        rating_run_id=str(row.get("rating_run_id") or ""),
        round_id=str(row.get("round_id") or ""),
        round_index=_safe_int_or_none(row.get("round_index")),
    )
    enriched = dict(row)
    for key in (
        "tally",
        "ok",
        "summary_ref",
        "first_gif_ref",
        "sample_gif_refs",
        "shard_summary_refs",
        "shard_summary_ref_count",
        "shard_count",
        "result_detail_mode",
    ):
        if key in summary:
            enriched[key] = summary[key]
    return arena._to_plain(enriched)


def _assert_checkpoint_count(
    *,
    refs: Sequence[str],
    discovery: Mapping[str, Any] | None,
    expected_checkpoint_count: int = 0,
    max_runs: int = 0,
    allow_missing_checkpoints: bool = False,
) -> None:
    if allow_missing_checkpoints:
        return
    expected = int(expected_checkpoint_count or 0)
    discovery_selection = str(
        (discovery or {}).get("checkpoint_selection") or arena.CHECKPOINT_SELECTION_LATEST
    )
    if (
        expected <= 0
        and int(max_runs or 0) > 0
        and discovery is not None
        and discovery_selection != arena.CHECKPOINT_SELECTION_ALL
    ):
        expected = int(max_runs)
    missing = int(discovery.get("missing_count", 0) or 0) if discovery else 0
    found = len(refs)
    if missing or (expected > 0 and found != expected):
        raise ValueError(
            "checkpoint discovery incomplete: "
            f"found {found}, expected {expected or 'all requested'}, missing {missing}; "
            "pass allow_missing_checkpoints=true only for an intentional partial run"
        )


def _default_tournament_id(rows: list[dict[str, Any]], requested: str) -> str:
    if requested:
        clean = runs.clean_id(requested, label="tournament_id")
        if any(row["tournament_id"] == clean for row in rows):
            return clean
    for row in rows:
        if row.get("is_current") and row.get("tournament_id") == CURRENT_TOURNAMENT_ID:
            return CURRENT_TOURNAMENT_ID
    if any(row.get("tournament_id") == CURRENT_TOURNAMENT_ID for row in rows):
        return CURRENT_TOURNAMENT_ID
    return str(rows[0]["tournament_id"]) if rows else ""


def _review_rankings_payload(
    mount: Path,
    *,
    tournament_id: str = "",
    rating_run_id: str = "latest",
    limit: int = 100,
    offset: int = 0,
    allow_live_provisional: bool = False,
) -> dict[str, Any]:
    tournaments = _list_tournaments(mount)
    selected_tournament = _default_tournament_id(tournaments, tournament_id)
    rating_runs = (
        _list_rating_runs(mount, tournament_id=selected_tournament) if selected_tournament else []
    )
    selected_rating_run = _default_rating_run_id(rating_runs, rating_run_id)
    snapshot = (
        _read_best_rating_snapshot_for_run(
            mount,
            tournament_id=selected_tournament,
            rating_run_id=selected_rating_run,
            allow_live_provisional=allow_live_provisional,
        )
        if selected_tournament and selected_rating_run
        else {}
    )
    rows = _rating_rows(snapshot)
    page = rows[offset : offset + limit]
    return {
        "selected_tournament_id": selected_tournament,
        "rating_run_id": selected_rating_run,
        "round_id": snapshot.get("round_id"),
        "provisional": bool(snapshot.get("provisional")),
        "source": snapshot.get("source"),
        "rows": arena._to_plain(page),
        "total": len(rows),
        "limit": limit,
        "offset": offset,
        "has_older": offset + limit < len(rows),
        "has_newer": offset > 0,
        "ratings_ref": snapshot.get("ratings_ref") or snapshot.get("latest_ref"),
    }


def _review_checkpoint_payload(
    mount: Path,
    *,
    tournament_id: str = "",
    rating_run_id: str = "latest",
    checkpoint_id: str,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    safe_limit = max(1, min(MAX_LIMIT, int(limit)))
    safe_offset = max(0, int(offset))
    rankings = _review_rankings_payload(
        mount,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        limit=MAX_LIMIT,
        offset=0,
    )
    selected_tournament = str(rankings.get("selected_tournament_id") or "")
    selected_rating_run = str(rankings.get("rating_run_id") or "")
    snapshot = (
        _read_best_rating_snapshot_for_run(
            mount,
            tournament_id=selected_tournament,
            rating_run_id=selected_rating_run,
        )
        if selected_tournament and selected_rating_run
        else {}
    )
    rating_row = _rating_row_by_checkpoint(snapshot, checkpoint_id)
    rank_by_checkpoint = _rating_rank_by_checkpoint(snapshot)
    battles = (
        _list_battle_index(
            mount,
            tournament_id=selected_tournament,
            limit=MAX_LIMIT,
            offset=0,
            checkpoint_id=checkpoint_id,
        )
        if selected_tournament
        else {"rows": [], "total": 0, "source": "none"}
    )
    if (
        battles.get("source") == "battle_index_missing"
        and selected_tournament
        and selected_rating_run
    ):
        battles = _checkpoint_live_shard_battles(
            mount,
            tournament_id=selected_tournament,
            rating_run_id=selected_rating_run,
            checkpoint_id=checkpoint_id,
            limit=MAX_LIMIT,
            offset=0,
        )
    if battles.get("source") in {
        "battle_index_missing",
        "checkpoint_live_shards_missing_round_input",
        "checkpoint_live_shards_unavailable",
        "none",
    }:
        live_pair_results = snapshot.get("live_pair_results")
        if isinstance(live_pair_results, Sequence) and not isinstance(
            live_pair_results, (str, bytes)
        ):
            battles = _battle_index_from_pair_results(
                live_pair_results,
                checkpoint_id=checkpoint_id,
                limit=MAX_LIMIT,
                offset=0,
            )
    source = str(battles.get("source") or "")
    battle_rows = battles.get("rows", [])
    try:
        total_rows = int(battles.get("total") or 0)
    except (TypeError, ValueError):
        total_rows = 0
    raw_rows = _sort_checkpoint_battle_rows(
        battle_rows if isinstance(battle_rows, Sequence) else [],
        checkpoint_id=checkpoint_id,
        rank_by_checkpoint=rank_by_checkpoint,
    )
    if total_rows <= 0:
        total_rows = len(raw_rows)
    page_rows = raw_rows[safe_offset : safe_offset + safe_limit]
    if source in {"battle_index", "checkpoint_round_input", "live_shard_tallies"}:
        page_rows = [_enrich_battle_row_from_live_shards(mount, row) for row in page_rows]
    page = [
        _review_battle_row(
            row,
            checkpoint_id,
            rank_by_checkpoint=rank_by_checkpoint,
        )
        for row in page_rows
    ]
    return {
        "selected_tournament_id": selected_tournament,
        "rating_run_id": selected_rating_run,
        "round_id": snapshot.get("round_id"),
        "checkpoint_id": checkpoint_id,
        "ranking": arena._to_plain(rating_row),
        "rows": arena._to_plain(page),
        "total": total_rows,
        "limit": safe_limit,
        "offset": safe_offset,
        "has_older": safe_offset + safe_limit < total_rows,
        "has_newer": safe_offset > 0,
        "source": source,
    }


def _review_battle_payload(
    mount: Path,
    *,
    tournament_id: str = "",
    battle_id: str,
    gif_sample_limit: int = 10,
    game_limit: int = DEFAULT_BATTLE_GAME_LIMIT,
    game_offset: int = 0,
) -> dict[str, Any]:
    tournaments = _list_tournaments(mount)
    selected_tournament = _default_tournament_id(tournaments, tournament_id)
    safe_game_limit = max(1, min(MAX_LIMIT, int(game_limit)))
    safe_game_offset = max(0, int(game_offset))
    cache_key = (
        f"battle-detail:{mount}:{selected_tournament}:{battle_id}:"
        f"{WEB_BATTLE_DETAIL_CACHE_VERSION}:{gif_sample_limit}:"
        f"{safe_game_limit}:{safe_game_offset}"
    )
    cached = _web_cache_get(
        cache_key,
        ttl_seconds=WEB_BATTLE_DETAIL_CACHE_TTL_SECONDS,
    )
    if isinstance(cached, dict):
        return cached
    source = "none"
    row: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    if selected_tournament:
        battle_ref = arena.battle_summary_ref(selected_tournament, battle_id)
        summary = _read_json(runs.volume_path(mount, battle_ref))
        if summary:
            source = "battle_summary"
            row = {
                "tournament_id": selected_tournament,
                "battle_id": battle_id,
                "players": summary.get("players"),
                "tally": summary.get("tally"),
                "ok": summary.get("ok"),
                "summary_ref": summary.get("summary_ref") or battle_ref.as_posix(),
                "first_gif_ref": summary.get("first_gif_ref"),
            }
        else:
            battle_root = runs.volume_path(
                mount,
                arena.battle_root_ref(selected_tournament, battle_id),
            )
            if battle_root.exists():
                shards = _read_battle_shard_summaries(
                    mount,
                    tournament_id=selected_tournament,
                    battle_id=battle_id,
                )
                if shards:
                    summary = {
                        "tournament_id": selected_tournament,
                        "battle_id": battle_id,
                        "tally": arena.merge_game_tallies(
                            [
                                dict(shard.get("tally"))
                                for shard in shards
                                if isinstance(shard.get("tally"), Mapping)
                            ]
                        ),
                        "shard_summary_refs": _refs_from_shards(
                            shards,
                            "summary_ref",
                        ),
                        "shard_summary_ref_count": len(shards),
                        "sample_gif_refs": _refs_from_shards(
                            shards,
                            "sample_gif_refs",
                        ),
                        "first_gif_ref": _first_ref_from_shards(
                            shards,
                            "first_gif_ref",
                        ),
                        "shard_count": len(shards),
                        "result_detail_mode": "live_shard_tally",
                    }
                source = "battle_dir"
                row = {
                    "tournament_id": selected_tournament,
                    "battle_id": battle_id,
                    "tally": summary.get("tally") if summary else None,
                    "ok": None,
                    "summary_ref": None,
                    "first_gif_ref": summary.get("first_gif_ref") if summary else None,
                }
            else:
                row, source = _read_battle_index_row(
                    mount,
                    tournament_id=selected_tournament,
                    battle_id=battle_id,
                )
                summary = _read_battle_summary(
                    mount,
                    tournament_id=selected_tournament,
                    battle_id=battle_id,
                    battle_index_row=row,
                )
    games, game_sources, total_game_count = (
        _read_game_summary_refs(
            mount,
            tournament_id=selected_tournament,
            battle_id=battle_id,
            battle_summary=summary,
            game_limit=safe_game_limit,
            game_offset=safe_game_offset,
        )
        if selected_tournament and (summary or row)
        else ([], [], 0)
    )
    summary_without_games = dict(summary)
    summary_without_games.pop("games", None)
    samples = _sample_gif_refs(
        battle_summary=summary,
        games=games,
        mount=mount,
        limit=gif_sample_limit,
    )
    payload = {
        "selected_tournament_id": selected_tournament,
        "battle_id": battle_id,
        "battle": arena._to_plain(row),
        "summary": arena._to_plain(summary_without_games),
        "games": arena._to_plain(games),
        "game_count": int(total_game_count),
        "game_rows_returned": len(games),
        "game_limit": safe_game_limit,
        "game_offset": safe_game_offset,
        "has_older_games": safe_game_offset + safe_game_limit < int(total_game_count),
        "has_newer_games": safe_game_offset > 0,
        "game_sources": game_sources,
        "sample_gifs": samples,
        "sample_gif_count": len(samples),
        "source": source,
    }
    if payload["game_count"] or payload["sample_gif_count"] or source == "battle_summary":
        return _web_cache_set(cache_key, arena._to_plain(payload))
    return arena._to_plain(payload)


def _read_tournament_json_ref(mount: Path, ref: str) -> dict[str, Any]:
    try:
        safe_ref = arena.validate_tournament_artifact_ref(ref)
    except ValueError:
        return {}
    if safe_ref.suffix != ".json":
        return {}
    return _read_json(runs.volume_path(mount, safe_ref))


def _compact_review_game(game: Mapping[str, Any]) -> dict[str, Any]:
    score = game.get("score") if isinstance(game.get("score"), Mapping) else {}
    outcome = score.get("outcome")
    if not outcome:
        if score.get("draw"):
            outcome = "draw"
        elif score.get("winner_seat") == 0:
            outcome = "seat_0_win"
        elif score.get("winner_seat") == 1:
            outcome = "seat_1_win"
    return arena._to_plain(
        {
            "ok": bool(game.get("ok")),
            "game_id": game.get("game_id"),
            "game_index": game.get("game_index"),
            "seed": game.get("seed"),
            "outcome": outcome,
            "winner_seat": score.get("winner_seat"),
            "loser_seat": score.get("loser_seat"),
            "draw": bool(score.get("draw")),
            "physical_steps": game.get("physical_steps") or score.get("physical_steps"),
            "max_steps": game.get("max_steps") or score.get("max_steps"),
            "gif_ref": game.get("gif_ref"),
            "summary_ref": game.get("summary_ref"),
            "error_type": game.get("error_type"),
            "error": game.get("error"),
        }
    )


def _read_battle_summary(
    mount: Path,
    *,
    tournament_id: str,
    battle_id: str,
    battle_index_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary_ref = (
        str(battle_index_row.get("summary_ref") or "")
        if isinstance(battle_index_row, Mapping)
        else ""
    )
    summary = _read_tournament_json_ref(mount, summary_ref) if summary_ref else {}
    if summary:
        return summary
    return _read_json(runs.volume_path(mount, arena.battle_summary_ref(tournament_id, battle_id)))


def _read_game_summary_refs(
    mount: Path,
    *,
    tournament_id: str,
    battle_id: str,
    battle_summary: Mapping[str, Any],
    game_limit: int = DEFAULT_BATTLE_GAME_LIMIT,
    game_offset: int = 0,
) -> tuple[list[dict[str, Any]], list[str], int]:
    safe_limit = max(1, min(MAX_LIMIT, int(game_limit)))
    safe_offset = max(0, int(game_offset))
    end = safe_offset + safe_limit
    games_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    sources: set[str] = set()
    expected_count = int(
        (
            battle_summary.get("tally") if isinstance(battle_summary.get("tally"), Mapping) else {}
        ).get("game_count")
        or battle_summary.get("game_summary_ref_count")
        or 0
    )
    embedded_total = 0
    summary_ref_total = 0
    shard_game_total = 0

    def add_game(raw: Mapping[str, Any], source: str) -> None:
        game = _compact_review_game(raw)
        key = (
            str(game.get("game_id") or ""),
            str(game.get("summary_ref") or ""),
        )
        if not key[0] and not key[1]:
            return
        games_by_key[key] = game
        sources.add(source)

    embedded_games = battle_summary.get("games")
    if isinstance(embedded_games, Sequence) and not isinstance(embedded_games, (str, bytes)):
        embedded_total = len(embedded_games)
        for game in embedded_games[safe_offset:end]:
            if isinstance(game, Mapping):
                add_game(game, "battle_json")

    summary_refs = battle_summary.get("game_summary_refs")
    if isinstance(summary_refs, Sequence) and not isinstance(summary_refs, (str, bytes)):
        valid_refs = [ref for ref in summary_refs if isinstance(ref, str)]
        summary_ref_total = len(valid_refs)
        for ref in valid_refs[safe_offset:end]:
            if not isinstance(ref, str):
                continue
            payload = _read_tournament_json_ref(mount, ref)
            if payload:
                add_game(payload, "game_summary_refs")

    shard_refs = battle_summary.get("shard_summary_refs")
    if isinstance(shard_refs, Sequence) and not isinstance(shard_refs, (str, bytes)):
        cursor = 0
        for ref in shard_refs:
            if not isinstance(ref, str):
                continue
            shard = _read_tournament_json_ref(mount, ref)
            shard_games = shard.get("games") if isinstance(shard, Mapping) else None
            if not isinstance(shard_games, Sequence) or isinstance(shard_games, (str, bytes)):
                continue
            shard_game_total += len(shard_games)
            for game in shard_games:
                if not isinstance(game, Mapping):
                    cursor += 1
                    continue
                game_index = _safe_int_or_none(game.get("game_index"))
                logical_index = game_index if game_index is not None else cursor
                cursor += 1
                if safe_offset <= logical_index < end:
                    add_game(game, "shard_summary_refs")

    known_total = max(
        expected_count,
        embedded_total,
        summary_ref_total,
        shard_game_total,
        len(games_by_key),
    )
    has_indexed_game_source = bool(embedded_total or summary_ref_total or shard_game_total)
    if games_by_key or has_indexed_game_source:
        games = list(games_by_key.values())
        games.sort(
            key=lambda game: (
                int(game.get("game_index", 0) or 0),
                str(game.get("game_id") or ""),
            )
        )
        total = max(known_total, len(games))
        return games, sorted(sources), total

    games_root = (
        runs.volume_path(mount, arena.tournament_root_ref(tournament_id))
        / "battles"
        / runs.clean_id(battle_id, label="battle_id")
        / "games"
    )
    scanned_total = 0
    if games_root.exists():
        paths = sorted(games_root.glob("*/summary.json"), key=lambda item: item.as_posix())
        scanned_total = len(paths)
        for path in paths[safe_offset:end]:
            payload = _read_json(path)
            if not payload:
                continue
            payload.setdefault("summary_ref", runs.file_ref(path, mount=mount))
            add_game(payload, "game_summary_scan")

    games = list(games_by_key.values())
    games.sort(
        key=lambda game: (
            int(game.get("game_index", 0) or 0),
            str(game.get("game_id") or ""),
        )
    )
    total = max(expected_count, scanned_total, len(games))
    return games, sorted(sources), total


def _sample_gif_refs(
    *,
    battle_summary: Mapping[str, Any],
    games: Sequence[Mapping[str, Any]],
    mount: Path | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    sample_limit = max(1, int(limit))
    samples: list[dict[str, Any]] = []
    summary_refs = battle_summary.get("sample_gif_refs")
    if isinstance(summary_refs, Sequence) and not isinstance(summary_refs, (str, bytes)):
        for index, ref in enumerate(summary_refs):
            if not ref:
                continue
            samples.append(
                {
                    "game_id": None,
                    "game_index": index,
                    "outcome": None,
                    "gif_ref": str(ref),
                }
            )
            if len(samples) >= sample_limit:
                break
    if samples:
        return _dedupe_gif_samples(samples)

    game_summary_refs = battle_summary.get("game_summary_refs")
    if (
        mount is not None
        and isinstance(game_summary_refs, Sequence)
        and not isinstance(game_summary_refs, (str, bytes))
    ):
        valid_refs = [ref for ref in game_summary_refs if isinstance(ref, str)]
        if valid_refs:
            selected_indices: list[int] = []
            if len(valid_refs) <= sample_limit:
                selected_indices.extend(range(len(valid_refs)))
            elif sample_limit == 1:
                selected_indices.append(0)
            else:
                last = len(valid_refs) - 1
                selected_indices.extend(
                    round(index * last / float(sample_limit - 1)) for index in range(sample_limit)
                )
            scan_cap = min(len(valid_refs), max(50, sample_limit * 10))
            selected_indices.extend(range(scan_cap))
            seen_indices: set[int] = set()
            for ref_index in selected_indices:
                if ref_index in seen_indices:
                    continue
                seen_indices.add(ref_index)
                payload = _read_tournament_json_ref(mount, valid_refs[ref_index])
                gif_ref = payload.get("gif_ref") if isinstance(payload, Mapping) else None
                if not gif_ref:
                    continue
                compact = _compact_review_game(payload)
                samples.append(
                    {
                        "game_id": compact.get("game_id"),
                        "game_index": compact.get("game_index"),
                        "outcome": compact.get("outcome"),
                        "gif_ref": gif_ref,
                    }
                )
                if len(samples) >= sample_limit:
                    break
        if samples:
            return _dedupe_gif_samples(samples)

    gif_games = [game for game in games if isinstance(game, Mapping) and game.get("gif_ref")]
    selected = gif_games
    if len(gif_games) > sample_limit:
        if sample_limit == 1:
            selected = [gif_games[0]]
        else:
            selected = []
            last = len(gif_games) - 1
            for index in range(sample_limit):
                selected.append(gif_games[round(index * last / float(sample_limit - 1))])
    samples = [
        {
            "game_id": game.get("game_id"),
            "game_index": game.get("game_index"),
            "outcome": game.get("outcome"),
            "gif_ref": game.get("gif_ref"),
        }
        for game in selected
    ]
    if not samples and battle_summary.get("first_gif_ref"):
        samples.append(
            {
                "game_id": None,
                "game_index": None,
                "outcome": None,
                "gif_ref": battle_summary.get("first_gif_ref"),
            }
        )
    return _dedupe_gif_samples(samples)


def _dedupe_gif_samples(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen_refs: set[str] = set()
    deduped = []
    for sample in samples:
        ref = str(sample.get("gif_ref") or "")
        if not ref or ref in seen_refs:
            continue
        seen_refs.add(ref)
        deduped.append(sample)
    return arena._to_plain(deduped)


def _build_fastapi_app(volume: Any):
    from fastapi import FastAPI, Header, Query
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    web_app = FastAPI(title="CurvyTron checkpoint tournament")

    @web_app.get("/")
    def index(
        tournament_id: str = "",
        rating_run_id: str = "latest",
        checkpoint_id: str = "",
        battle_id: str = "",
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
        offset: int = Query(0, ge=0),
        game_limit: int = Query(DEFAULT_BATTLE_GAME_LIMIT, ge=1, le=MAX_LIMIT),
        game_offset: int = Query(0, ge=0),
        fresh: bool = False,
        live_scan: bool = False,
    ) -> HTMLResponse:
        reload_error = _web_reload_volume(
            volume,
            force=bool(fresh),
            min_interval_sec=WEB_PAGE_RELOAD_MIN_INTERVAL_SECONDS,
        )
        tournaments = _list_tournaments(TOURNAMENT_MOUNT)
        selected = _default_tournament_id(tournaments, tournament_id)
        rating_runs = (
            _list_rating_runs(TOURNAMENT_MOUNT, tournament_id=selected) if selected else []
        )
        selected_rating_run = _default_rating_run_id(rating_runs, rating_run_id)
        rating_snapshot = (
            _read_best_rating_snapshot_for_run(
                TOURNAMENT_MOUNT,
                tournament_id=selected,
                rating_run_id=selected_rating_run,
                allow_live_provisional=bool(fresh),
            )
            if selected and selected_rating_run
            else {}
        )
        rating_progress = (
            _read_cached_live_rating_progress(
                TOURNAMENT_MOUNT,
                tournament_id=selected,
                rating_run_id=selected_rating_run,
                force_live=bool(live_scan),
            )
            if selected and selected_rating_run
            else {}
        )
        if selected and checkpoint_id:
            checkpoint_payload = _review_checkpoint_payload(
                TOURNAMENT_MOUNT,
                tournament_id=selected,
                rating_run_id=selected_rating_run,
                checkpoint_id=checkpoint_id,
                limit=limit,
                offset=offset,
            )
            battles = {
                "rows": checkpoint_payload.get("rows", []),
                "total": checkpoint_payload.get("total", 0),
                "limit": checkpoint_payload.get("limit", limit),
                "offset": checkpoint_payload.get("offset", offset),
                "has_older": checkpoint_payload.get("has_older", False),
                "has_newer": checkpoint_payload.get("has_newer", False),
                "source": checkpoint_payload.get("source"),
                "checkpoint_id": checkpoint_id,
            }
        else:
            battles = {"rows": [], "total": 0, "limit": limit, "offset": offset}
        battle_detail = (
            _review_battle_payload(
                TOURNAMENT_MOUNT,
                tournament_id=selected,
                battle_id=battle_id,
                gif_sample_limit=10,
                game_limit=game_limit,
                game_offset=game_offset,
            )
            if selected and battle_id
            else {}
        )
        return HTMLResponse(
            _render_page(
                tournaments=tournaments,
                selected_tournament_id=selected,
                selected_rating_run_id=selected_rating_run,
                selected_checkpoint_id=checkpoint_id,
                rating_runs=rating_runs,
                rating_snapshot=rating_snapshot,
                rating_progress=rating_progress,
                battles=battles,
                selected_battle_id=battle_id,
                battle_detail=battle_detail,
                volume_reload_error=reload_error or "",
            ),
            headers=DYNAMIC_HEADERS,
        )

    @web_app.get("/battle")
    def battle_page(
        battle_id: str,
        tournament_id: str = "",
        rating_run_id: str = "latest",
        checkpoint_id: str = "",
        game_limit: int = Query(DEFAULT_BATTLE_GAME_LIMIT, ge=1, le=MAX_LIMIT),
        game_offset: int = Query(0, ge=0),
        fresh: bool = False,
    ) -> HTMLResponse:
        _web_reload_volume(
            volume,
            force=bool(fresh),
            min_interval_sec=WEB_PAGE_RELOAD_MIN_INTERVAL_SECONDS,
        )
        payload = _review_battle_payload(
            TOURNAMENT_MOUNT,
            tournament_id=tournament_id,
            battle_id=battle_id,
            gif_sample_limit=10,
            game_limit=game_limit,
            game_offset=game_offset,
        )
        return HTMLResponse(
            _render_battle_page(
                payload=payload,
                rating_run_id=rating_run_id,
                checkpoint_id=checkpoint_id,
            ),
            headers=DYNAMIC_HEADERS,
        )

    @web_app.get("/api/tournaments")
    def tournaments(fresh: bool = False) -> JSONResponse:
        reload_error = _web_reload_volume(volume, force=True) if fresh else None
        return JSONResponse(
            {
                "tournaments": _list_tournaments(TOURNAMENT_MOUNT),
                "volume_reload_error": reload_error,
            },
            headers=DYNAMIC_HEADERS,
        )

    @web_app.get("/api/ratings")
    def ratings(
        tournament_id: str = "",
        fresh: bool = False,
    ) -> JSONResponse:
        reload_error = _web_reload_volume(volume, force=True) if fresh else None
        tournaments = _list_tournaments(TOURNAMENT_MOUNT)
        selected = _default_tournament_id(tournaments, tournament_id)
        rows = _list_rating_runs(TOURNAMENT_MOUNT, tournament_id=selected) if selected else []
        return JSONResponse(
            {
                "selected_tournament_id": selected,
                "rating_runs": rows,
                "volume_reload_error": reload_error,
            },
            headers=DYNAMIC_HEADERS,
        )

    @web_app.get("/api/rating-progress")
    def rating_progress(
        tournament_id: str = "",
        rating_run_id: str = "latest",
        fresh: bool = False,
        live_scan: bool = False,
    ) -> JSONResponse:
        reload_error = _web_reload_volume(
            volume,
            force=bool(fresh),
            min_interval_sec=WEB_PROGRESS_RELOAD_MIN_INTERVAL_SECONDS,
        )
        tournaments = _list_tournaments(TOURNAMENT_MOUNT)
        selected_tournament = _default_tournament_id(tournaments, tournament_id)
        rating_runs = (
            _list_rating_runs(TOURNAMENT_MOUNT, tournament_id=selected_tournament)
            if selected_tournament
            else []
        )
        selected_rating_run = _default_rating_run_id(rating_runs, rating_run_id)
        progress = (
            _read_cached_live_rating_progress(
                TOURNAMENT_MOUNT,
                tournament_id=selected_tournament,
                rating_run_id=selected_rating_run,
                force_live=bool(live_scan),
            )
            if selected_tournament and selected_rating_run
            else {}
        )
        refresh_call_id = (
            _maybe_spawn_rating_progress_refresh(
                tournament_id=selected_tournament,
                rating_run_id=selected_rating_run,
                progress=progress,
            )
            if selected_tournament and selected_rating_run and progress
            else ""
        )
        return JSONResponse(
            {
                "selected_tournament_id": selected_tournament,
                "rating_run_id": selected_rating_run,
                "progress": progress,
                "progress_refresh_call_id": refresh_call_id,
                "volume_reload_error": reload_error,
            },
            headers=DYNAMIC_HEADERS,
        )

    @web_app.get("/api/rating-standings")
    def rating_standings(
        tournament_id: str = "",
        rating_run_id: str = "latest",
        limit: int = Query(100, ge=1, le=MAX_LIMIT),
        offset: int = Query(0, ge=0),
        fresh: bool = False,
        live_scan: bool = False,
        live_provisional: bool = False,
    ) -> JSONResponse:
        reload_error = _web_reload_volume(volume, force=True) if fresh else None
        tournaments = _list_tournaments(TOURNAMENT_MOUNT)
        selected_tournament = _default_tournament_id(tournaments, tournament_id)
        rating_runs = (
            _list_rating_runs(TOURNAMENT_MOUNT, tournament_id=selected_tournament)
            if selected_tournament
            else []
        )
        selected_rating_run = _default_rating_run_id(rating_runs, rating_run_id)
        snapshot = (
            _read_best_rating_snapshot_for_run(
                TOURNAMENT_MOUNT,
                tournament_id=selected_tournament,
                rating_run_id=selected_rating_run,
                allow_live_provisional=bool(fresh or live_provisional),
            )
            if selected_tournament and selected_rating_run
            else {}
        )
        all_rows = snapshot.get("ratings") if isinstance(snapshot.get("ratings"), list) else []
        progress = (
            _read_cached_live_rating_progress(
                TOURNAMENT_MOUNT,
                tournament_id=selected_tournament,
                rating_run_id=selected_rating_run,
                force_live=bool(live_scan),
            )
            if selected_tournament and selected_rating_run
            else {}
        )
        rows = all_rows[offset : offset + limit]
        return JSONResponse(
            {
                "selected_tournament_id": selected_tournament,
                "rating_run_id": snapshot.get("rating_run_id") or selected_rating_run,
                "round_id": snapshot.get("round_id"),
                "provisional": bool(snapshot.get("provisional")),
                "source": snapshot.get("source"),
                "progress": progress,
                "rows": rows,
                "total": len(all_rows),
                "limit": limit,
                "offset": offset,
                "has_older": offset + limit < len(all_rows),
                "has_newer": offset > 0,
                "ratings_ref": snapshot.get("ratings_ref") or snapshot.get("latest_ref"),
                "volume_reload_error": reload_error,
            },
            headers=DYNAMIC_HEADERS,
        )

    @web_app.get("/api/review/rankings")
    def review_rankings(
        tournament_id: str = "",
        rating_run_id: str = "latest",
        limit: int = Query(100, ge=1, le=MAX_LIMIT),
        offset: int = Query(0, ge=0),
        fresh: bool = False,
        live_provisional: bool = False,
    ) -> JSONResponse:
        reload_error = None
        if fresh:
            reload_error = _web_reload_volume(volume, force=True)
        payload = _review_rankings_payload(
            TOURNAMENT_MOUNT,
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
            limit=limit,
            offset=offset,
            allow_live_provisional=bool(fresh or live_provisional),
        )
        payload["volume_reload_error"] = reload_error
        return JSONResponse(payload, headers=DYNAMIC_HEADERS)

    @web_app.get("/api/review/checkpoint")
    def review_checkpoint(
        checkpoint_id: str,
        tournament_id: str = "",
        rating_run_id: str = "latest",
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
        offset: int = Query(0, ge=0),
        fresh: bool = False,
    ) -> JSONResponse:
        if fresh:
            _web_reload_volume(volume, force=True)
        return JSONResponse(
            _review_checkpoint_payload(
                TOURNAMENT_MOUNT,
                tournament_id=tournament_id,
                rating_run_id=rating_run_id,
                checkpoint_id=checkpoint_id,
                limit=limit,
                offset=offset,
            ),
            headers=DYNAMIC_HEADERS,
        )

    @web_app.get("/api/review/battle")
    def review_battle(
        battle_id: str,
        tournament_id: str = "",
        gif_sample_limit: int = Query(10, ge=1, le=10),
        game_limit: int = Query(DEFAULT_BATTLE_GAME_LIMIT, ge=1, le=MAX_LIMIT),
        game_offset: int = Query(0, ge=0),
        fresh: bool = False,
    ) -> JSONResponse:
        if fresh:
            _web_reload_volume(volume, force=True)
        return JSONResponse(
            _review_battle_payload(
                TOURNAMENT_MOUNT,
                tournament_id=tournament_id,
                battle_id=battle_id,
                gif_sample_limit=gif_sample_limit,
                game_limit=game_limit,
                game_offset=game_offset,
            ),
            headers=DYNAMIC_HEADERS,
        )

    @web_app.get("/api/battles")
    def battles(
        tournament_id: str = "",
        checkpoint_id: str = "",
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
        offset: int = Query(0, ge=0),
        fresh: bool = False,
    ) -> JSONResponse:
        if fresh:
            _web_reload_volume(volume, force=True)
        tournaments = _list_tournaments(TOURNAMENT_MOUNT)
        selected = _default_tournament_id(tournaments, tournament_id)
        page = (
            _list_battles(
                TOURNAMENT_MOUNT,
                tournament_id=selected,
                limit=limit,
                offset=offset,
                checkpoint_id=checkpoint_id,
            )
            if selected
            else {"rows": [], "total": 0, "limit": limit, "offset": offset}
        )
        return JSONResponse({"selected_tournament_id": selected, **page}, headers=DYNAMIC_HEADERS)

    @web_app.get("/gif")
    def gif(ref: str, if_none_match: str = Header(default="")) -> Response:
        try:
            safe_ref = arena.validate_tournament_artifact_ref(ref)
        except ValueError as exc:
            return Response(str(exc), status_code=400)
        if safe_ref.name != "game.gif":
            return Response("not a tournament GIF ref", status_code=400)
        path = runs.volume_path(TOURNAMENT_MOUNT, safe_ref)
        if not path.is_file():
            _web_reload_volume(volume, force=True)
            if not path.is_file():
                return Response("GIF not found", status_code=404)
        stat = path.stat()
        etag = f'W/"{stat.st_mtime_ns}-{stat.st_size}"'
        headers = {
            "Cache-Control": f"public, max-age={GIF_CACHE_MAX_AGE_SECONDS}, immutable",
            "ETag": etag,
            "Content-Length": str(stat.st_size),
        }
        if if_none_match == etag:
            return Response(status_code=304, headers=headers)
        return Response(
            _read_cached_file_bytes(
                path,
                cache_prefix="gif-bytes",
                ttl_seconds=WEB_GIF_BYTES_CACHE_TTL_SECONDS,
                max_item_bytes=WEB_GIF_BYTES_CACHE_MAX_ITEM_BYTES,
            ),
            media_type="image/gif",
            headers=headers,
        )

    @web_app.get("/meta")
    def meta(ref: str) -> Response:
        try:
            safe_ref = arena.validate_tournament_artifact_ref(ref)
        except ValueError as exc:
            return Response(str(exc), status_code=400)
        if safe_ref.suffix != ".json":
            return Response("not a JSON ref", status_code=400)
        path = runs.volume_path(TOURNAMENT_MOUNT, safe_ref)
        if not path.is_file():
            _web_reload_volume(volume, force=True)
            if not path.is_file():
                return Response("JSON not found", status_code=404)
        return Response(
            _read_cached_file_bytes(
                path,
                cache_prefix="json-bytes",
                ttl_seconds=30.0,
                max_item_bytes=1024 * 1024,
            ),
            media_type="application/json",
            headers={"Cache-Control": "no-cache"},
        )

    return web_app


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    timeout=300,
    cpu=4,
    memory=4096,
    max_containers=20,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def curvytron_tournament_browser():
    return _build_fastapi_app(tournament_volume)


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    timeout=300,
    cpu=1,
    memory=1024,
)
def curvytron_tournament_visibility(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(spec or {})
    result = _update_tournament_visibility(
        TOURNAMENT_MOUNT,
        action=str(payload.get("action") or "list"),
        tournament_ids=payload.get("tournament_ids"),
        keep_tournament_ids=payload.get("keep_tournament_ids"),
        dry_run=bool(payload.get("dry_run", True)),
    )
    if not result["dry_run"] and result["changed_count"]:
        result["commit_error"] = _commit_volume(tournament_volume)
    return arena._to_plain(result)


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    timeout=10 * 60,
    cpu=1.0,
    memory=1024,
)
def curvytron_opponent_leaderboard_publish(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(spec or {})
    _reload_volume(tournament_volume)
    tournament_id = runs.clean_id(str(payload["tournament_id"]), label="tournament_id")
    rating_run_id = runs.clean_id(
        str(payload.get("rating_run_id") or arena.DEFAULT_RATING_RUN_ID),
        label="rating_run_id",
    )
    leaderboard_id = runs.clean_id(
        str(payload.get("leaderboard_id") or f"{tournament_id}-{rating_run_id}"),
        label="leaderboard_id",
    )
    default_snapshot_id = f"snapshot-{arena._short_hash(runs.utc_timestamp(), length=12)}"
    snapshot_id = runs.clean_id(
        str(payload.get("snapshot_id") or default_snapshot_id),
        label="snapshot_id",
    )
    allow_live_provisional = bool(payload.get("allow_live_provisional", False))
    rating_snapshot = _read_best_rating_snapshot_for_run(
        TOURNAMENT_MOUNT,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        allow_live_provisional=allow_live_provisional,
    )
    if not rating_snapshot:
        raise ValueError("rating snapshot not found")
    if rating_snapshot.get("provisional") and not allow_live_provisional:
        raise ValueError(
            "refusing to publish provisional rating snapshot without allow_live_provisional=True"
        )
    diagnostic_only = bool(payload.get("diagnostic_only", False))
    rating_spec = rating_snapshot.get("rating_spec")
    raw_decision_source_frames = (
        rating_spec.get("decision_source_frames") if isinstance(rating_spec, Mapping) else None
    )
    decision_source_frames = _safe_int_or_none(raw_decision_source_frames)
    allow_legacy_rating_snapshot = bool(payload.get("allow_legacy_rating_snapshot", False))
    if decision_source_frames != 1:
        if not diagnostic_only and not allow_legacy_rating_snapshot:
            got = (
                "missing"
                if raw_decision_source_frames is None
                else repr(raw_decision_source_frames)
            )
            raise ValueError(
                "refusing to publish leaderboard from rating snapshot without "
                "rating_spec.decision_source_frames=1; got "
                f"{got}. Pass diagnostic_only=True or "
                "allow_legacy_rating_snapshot=True for a diagnostic legacy publish."
            )
        diagnostic_only = True
    source_info = validate_rating_snapshot_source(
        rating_snapshot,
        expected_round_id=payload.get("expected_round_id"),
        expected_round_index=(
            int(payload["expected_round_index"])
            if payload.get("expected_round_index") is not None
            else None
        ),
        expected_rating_context_hash=payload.get("expected_rating_context_hash"),
        expected_roster_hash=payload.get("expected_roster_hash"),
        expected_rating_snapshot_sha256=payload.get("expected_rating_snapshot_sha256"),
    )
    active_min_distinct_opponents = (
        20
        if payload.get("active_min_distinct_opponents") is None
        else int(payload["active_min_distinct_opponents"])
    )
    active_min_valid_games = (
        300
        if payload.get("active_min_valid_games") is None
        else int(payload["active_min_valid_games"])
    )
    max_failure_rate = (
        0.02 if payload.get("max_failure_rate") is None else float(payload["max_failure_rate"])
    )
    max_active_rank = (
        arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT
        if payload.get("max_active_rank") is None
        else int(payload["max_active_rank"])
    )
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        rating_snapshot,
        leaderboard_id=leaderboard_id,
        snapshot_id=snapshot_id,
        generation=int(
            payload.get("generation") or int(rating_snapshot.get("round_index", 0) or 0)
        ),
        created_at=runs.utc_timestamp(),
        active_min_distinct_opponents=active_min_distinct_opponents,
        active_min_valid_games=active_min_valid_games,
        max_failure_rate=max_failure_rate,
        max_active_rank=max_active_rank,
    )
    active_count = sum(1 for row in snapshot["rows"] if row.get("status") == "active")
    if active_count < 1 and bool(payload.get("allow_no_active_rows", False)):
        diagnostic_only = True
    if active_count < 1 and not diagnostic_only:
        raise ValueError(
            "refusing to publish leaderboard with no active rows without allow_no_active_rows=True"
        )
    snapshot_ref = _leaderboard_snapshot_ref(leaderboard_id, snapshot_id)
    latest_ref = _leaderboard_latest_ref(leaderboard_id)
    snapshot_write = arena.write_json_artifact(TOURNAMENT_MOUNT, snapshot_ref, snapshot)
    latest_write = (
        {} if diagnostic_only else arena.write_json_artifact(TOURNAMENT_MOUNT, latest_ref, snapshot)
    )
    pointer = build_leaderboard_pointer(
        snapshot,
        snapshot_ref=snapshot_ref.as_posix(),
        published_at=runs.utc_timestamp(),
        writer={
            "kind": "curvytron_opponent_leaderboard_publish",
            "app_name": APP_NAME,
        },
    )
    pointer_key = _leaderboard_pointer_key(leaderboard_id)
    commit_error = _commit_volume(tournament_volume)
    pointer_published = False
    if not commit_error and not diagnostic_only:
        opponent_leaderboard_state.put(pointer_key, arena._to_plain(pointer))
        pointer_published = True
    _append_tournament_lineage_event(
        stage="leaderboard_published",
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        status="ok" if pointer_published else "not_published",
        leaderboard_id=leaderboard_id,
        snapshot_id=snapshot_id,
        snapshot_ref=snapshot_write.get("ref"),
        latest_ref=latest_write.get("ref"),
        pointer_key=pointer_key,
        row_count=len(snapshot["rows"]),
        active_count=active_count,
        rating_snapshot_sha256=source_info["rating_snapshot_sha256"],
        rating_stable=bool(rating_snapshot.get("stable", False)),
        rating_max_abs_delta=rating_snapshot.get("max_abs_delta"),
        diagnostic_only=diagnostic_only,
        commit_error=commit_error,
        pointer_published=pointer_published,
    )
    _commit_volume(tournament_volume)
    return arena._to_plain(
        {
            "schema_id": "curvyzero_opponent_leaderboard_publish/v0",
            "leaderboard_id": leaderboard_id,
            "snapshot_id": snapshot_id,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "snapshot_ref": snapshot_write.get("ref"),
            "latest_ref": latest_write.get("ref"),
            "pointer_key": pointer_key,
            "dict_name": OPPONENT_LEADERBOARD_DICT_NAME,
            "row_count": len(snapshot["rows"]),
            "active_count": active_count,
            "rating_snapshot_sha256": source_info["rating_snapshot_sha256"],
            "rating_stable": bool(rating_snapshot.get("stable", False)),
            "rating_max_abs_delta": rating_snapshot.get("max_abs_delta"),
            "rating_source": source_info,
            "provisional_count": pointer["compact_summary"]["provisional_count"],
            "diagnostic_only": diagnostic_only,
            "commit_error": commit_error,
            "pointer_published": pointer_published,
            "pointer": pointer,
        }
    )


def _volume_ref_to_mount_path(ref: str) -> tuple[Path, PurePosixPath]:
    text = str(ref or "").strip()
    if not text:
        raise ValueError("volume ref is required")
    if text.startswith("control:"):
        return CONTROL_MOUNT, runs.require_relative_ref(text[len("control:") :])
    if text.startswith("runs:"):
        return RUNS_MOUNT, runs.require_relative_ref(text[len("runs:") :])
    return RUNS_MOUNT, runs.require_relative_ref(text)


def _read_json_by_volume_ref(ref: str) -> dict[str, Any]:
    mount, relative_ref = _volume_ref_to_mount_path(ref)
    payload = _read_json(runs.volume_path(mount, relative_ref))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {ref!r}")
    return payload


def _write_json_by_volume_ref(
    ref: str,
    payload: Mapping[str, Any],
    *,
    commit: bool = True,
) -> dict[str, Any]:
    mount, relative_ref = _volume_ref_to_mount_path(ref)
    if mount != CONTROL_MOUNT:
        raise ValueError(f"training candidate controller only writes control: refs, got {ref!r}")
    path = runs.volume_path(mount, relative_ref)
    body = runs.json_bytes(arena._to_plain(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f".{path.name}.{arena._short_hash(runs.utc_timestamp(), length=12)}.tmp"
    )
    tmp_path.write_bytes(body)
    tmp_path.replace(path)
    write = {
        "path": str(path),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "ref": relative_ref.as_posix(),
    }
    commit_error = _commit_volume(control_volume) if commit else None
    return {
        **write,
        "volume_name": CONTROL_VOLUME_NAME,
        "commit_error": commit_error,
        "path": str(path),
    }


def _write_bytes_by_volume_ref(
    ref: str,
    body: bytes,
    *,
    commit: bool = True,
) -> dict[str, Any]:
    mount, relative_ref = _volume_ref_to_mount_path(ref)
    if mount != CONTROL_MOUNT:
        raise ValueError(f"training candidate controller only writes control: refs, got {ref!r}")
    path = runs.volume_path(mount, relative_ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f".{path.name}.{arena._short_hash(runs.utc_timestamp(), length=12)}.tmp"
    )
    tmp_path.write_bytes(body)
    tmp_path.replace(path)
    commit_error = _commit_volume(control_volume) if commit else None
    return {
        "path": str(path),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "ref": relative_ref.as_posix(),
        "volume_name": CONTROL_VOLUME_NAME,
        "commit_error": commit_error,
    }


def _materialize_training_candidate_checkpoint_ref(
    checkpoint_ref: str,
    *,
    assignment_run_id: str,
    assignment_attempt_id: str,
    assignment_id: str,
    entry_name: str,
    rank: int,
) -> tuple[str, dict[str, Any]]:
    source_mount, source_relative_ref = _volume_ref_to_mount_path(checkpoint_ref)
    source_path = runs.volume_path(source_mount, source_relative_ref)
    if not source_path.is_file():
        raise FileNotFoundError(
            "training candidate checkpoint source not found while materializing "
            f"control copy: {checkpoint_ref}"
        )
    checkpoint_name = source_path.name
    if not re.fullmatch(r"iteration_\d+\.pth\.tar", checkpoint_name):
        raise ValueError(
            "training candidate checkpoint refs must be immutable iteration_N.pth.tar "
            f"files; got {checkpoint_ref!r}"
        )
    source_metadata = _checkpoint_discovery_row_from_ref(
        source_relative_ref.as_posix(),
        mount=source_mount,
    )
    arena._require_loaded_policy_observation_metadata(
        checkpoint_ref=source_relative_ref.as_posix(),
        policy_trail_render_mode=source_metadata.get("policy_trail_render_mode"),
        policy_bonus_render_mode=source_metadata.get("policy_bonus_render_mode"),
        policy_observation_backend=source_metadata.get("policy_observation_backend"),
        policy_observation_contract_id=source_metadata.get("policy_observation_contract_id"),
        policy_observation_perspective_schema_id=source_metadata.get(
            "policy_observation_perspective_schema_id"
        ),
    )
    entry_id = runs.clean_id(str(entry_name or f"rank{rank}"), label="entry_name")
    source_hash = arena._short_hash(str(checkpoint_ref), length=12)
    target_ref = (
        "control:"
        + (
            runs.attempt_root_ref(TRAINING_TASK_ID, assignment_run_id, assignment_attempt_id)
            / "opponents"
            / "frozen_checkpoints"
            / runs.clean_id(assignment_id, label="assignment_id")
            / f"{entry_id}-rank{int(rank)}-{source_hash}"
            / checkpoint_name
        ).as_posix()
    )
    checkpoint_write = _write_bytes_by_volume_ref(
        target_ref,
        source_path.read_bytes(),
        commit=False,
    )
    sidecar_source_path = source_path.with_name(f"{source_path.name}.metadata.json")
    source_sidecar: dict[str, Any] = {}
    if sidecar_source_path.is_file():
        try:
            loaded_sidecar = json.loads(sidecar_source_path.read_text(encoding="utf-8"))
        except Exception:
            loaded_sidecar = {}
        if isinstance(loaded_sidecar, Mapping):
            source_sidecar = dict(loaded_sidecar)
    target_sidecar = {
        **source_sidecar,
        "schema_id": arena.CHECKPOINT_POLICY_METADATA_SIDECAR_SCHEMA_ID,
        "copied_at": runs.utc_timestamp(),
        "checkpoint_ref": target_ref,
        "source_checkpoint_ref": str(checkpoint_ref),
        "policy_trail_render_mode": source_metadata.get("policy_trail_render_mode"),
        "policy_bonus_render_mode": source_metadata.get("policy_bonus_render_mode"),
        "policy_observation_backend": source_metadata.get("policy_observation_backend"),
        "policy_observation_contract_id": source_metadata.get("policy_observation_contract_id"),
        "policy_observation_perspective_schema_id": source_metadata.get(
            "policy_observation_perspective_schema_id"
        ),
        "observation_contract": source_metadata.get("observation_contract"),
        "source_state_trail_render_mode": source_metadata.get("source_state_trail_render_mode"),
        "source_state_bonus_render_mode": source_metadata.get("source_state_bonus_render_mode"),
        "model_env_variant": source_metadata.get("model_env_variant"),
        "model_reward_variant": source_metadata.get("model_reward_variant"),
        "env_variant": source_metadata.get("env_variant"),
        "reward_variant": source_metadata.get("reward_variant"),
        "learner_seat_mode": source_metadata.get("learner_seat_mode"),
    }
    sidecar_write = _write_json_by_volume_ref(
        f"{target_ref}.metadata.json",
        target_sidecar,
        commit=False,
    )
    return target_ref, {
        "source_ref": str(checkpoint_ref),
        "source_volume": (
            CONTROL_VOLUME_NAME if source_mount == CONTROL_MOUNT else CHECKPOINT_VOLUME_NAME
        ),
        "target_ref": target_ref,
        "checkpoint_write": checkpoint_write,
        "metadata_write": sidecar_write,
    }


_RANK_TOKEN_PATTERN = re.compile(r"rank(?P<rank>[1-9][0-9]*)(?:_immortal)?")


def _rank_from_assignment_entry_tags(entry: Mapping[str, Any]) -> int | None:
    tags = entry.get("tags")
    if not isinstance(tags, Mapping):
        return None
    for key in ("rank", "source_slot"):
        value = tags.get(key)
        if value is None:
            continue
        if key == "rank":
            try:
                rank = int(value)
            except (TypeError, ValueError):
                rank = 0
            if rank > 0:
                return rank
        match = _RANK_TOKEN_PATTERN.search(str(value))
        if match:
            return int(match.group("rank"))
    return None


def _rank_from_assignment_entry(entry: Mapping[str, Any]) -> int | None:
    rank = _rank_from_assignment_entry_tags(entry)
    if rank is not None:
        return rank
    for key in ("name", "age_label", "opponent_snapshot_ref"):
        value = entry.get(key)
        if value is None:
            continue
        match = _RANK_TOKEN_PATTERN.search(str(value))
        if match:
            return int(match.group("rank"))
    return None


def _leaderboard_rows_by_rank(snapshot: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows_by_rank: dict[int, Mapping[str, Any]] = {}
    for row in snapshot.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        if str(row.get("status") or "") != "active":
            continue
        try:
            rank = int(row.get("rank") or 0)
        except (TypeError, ValueError):
            continue
        if rank > 0 and row.get("checkpoint_ref"):
            rows_by_rank.setdefault(rank, row)
    return rows_by_rank


def _assignment_entries_signature(assignment: Mapping[str, Any]) -> list[dict[str, Any]]:
    parsed = parse_opponent_assignment_snapshot(dict(assignment))
    signature = []
    for entry in parsed["opponent_mixture"]["entries"]:
        signature.append(
            {
                "name": entry.get("name"),
                "weight": float(entry.get("weight", 0.0)),
                "opponent_policy_kind": entry.get("opponent_policy_kind"),
                "opponent_runtime_mode": entry.get("opponent_runtime_mode"),
                "opponent_immortal": bool(entry.get("opponent_immortal", False)),
                "rank": _rank_from_assignment_entry(entry),
            }
        )
    return signature


def _refresh_pointer_generation(pointer: Mapping[str, Any]) -> int | None:
    for value in (
        pointer.get("generation"),
        (
            pointer.get("source_leaderboard", {}).get("generation")
            if isinstance(pointer.get("source_leaderboard"), Mapping)
            else None
        ),
    ):
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _validate_assignment_refresh_pointer_for_controller(
    pointer_ref: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    pointer = _read_json_by_volume_ref(pointer_ref)
    if pointer.get("schema_id") != "curvyzero_opponent_assignment_refresh_pointer/v0":
        raise ValueError(f"refresh pointer {pointer_ref!r} has invalid schema_id")
    assignment_ref = pointer.get("assignment_ref")
    if not isinstance(assignment_ref, str) or not assignment_ref:
        raise ValueError(f"refresh pointer {pointer_ref!r} requires assignment_ref")
    assignment = _read_json_by_volume_ref(assignment_ref)
    assignment_sha256 = canonical_assignment_json_sha256(assignment)
    expected_sha256 = pointer.get("assignment_sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError(f"refresh pointer {pointer_ref!r} requires assignment_sha256")
    if expected_sha256 != assignment_sha256:
        raise ValueError(
            f"refresh pointer {pointer_ref!r} assignment_sha256 mismatch: "
            f"expected {expected_sha256}, got {assignment_sha256}"
        )
    parse_opponent_assignment_snapshot(assignment)
    return pointer, assignment, assignment_sha256


def _assert_refresh_pointer_unchanged(
    pointer_ref: str,
    *,
    expected_assignment_ref: str,
    expected_assignment_sha256: str,
) -> None:
    pointer = _read_json_by_volume_ref(pointer_ref)
    actual_ref = str(pointer.get("assignment_ref") or "")
    actual_sha256 = str(pointer.get("assignment_sha256") or "")
    if actual_ref != str(expected_assignment_ref) or actual_sha256 != str(
        expected_assignment_sha256
    ):
        raise ValueError(
            f"refresh pointer {pointer_ref!r} changed during controller refresh; "
            "refusing stale overwrite"
        )


def _assignment_with_leaderboard_ranks(
    assignment: Mapping[str, Any],
    *,
    leaderboard_snapshot: Mapping[str, Any],
    assignment_id: str,
    source_ref: str,
    seed: int,
    allow_partial: bool,
    checkpoint_ref_rewriter: Callable[
        [Mapping[str, Any], Mapping[str, Any], int],
        tuple[str, Mapping[str, Any] | None],
    ]
    | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = parse_opponent_assignment_snapshot(dict(assignment))
    source_entries = parsed["opponent_mixture"]["entries"]
    rows_by_rank = _leaderboard_rows_by_rank(leaderboard_snapshot)
    entries: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    missing_ranks: list[int] = []
    for raw_entry in source_entries:
        entry = copy.deepcopy(dict(raw_entry))
        is_frozen = (
            entry.get("opponent_policy_kind") == OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT
        )
        tagged_rank = _rank_from_assignment_entry_tags(entry)
        if not is_frozen and tagged_rank is None:
            entries.append(entry)
            continue
        rank = _rank_from_assignment_entry(entry) if is_frozen else tagged_rank
        if rank is None:
            raise ValueError(
                f"cannot infer leaderboard rank for frozen entry {entry.get('name')!r}"
            )
        row = rows_by_rank.get(rank)
        if row is None:
            missing_ranks.append(rank)
            if allow_partial:
                entries.append(entry)
                continue
            continue
        assigned_checkpoint_ref = str(row["checkpoint_ref"])
        checkpoint_copy: Mapping[str, Any] | None = None
        if checkpoint_ref_rewriter is not None:
            assigned_checkpoint_ref, checkpoint_copy = checkpoint_ref_rewriter(
                entry,
                row,
                rank,
            )
        entry["opponent_checkpoint_ref"] = assigned_checkpoint_ref
        entry.pop("opponent_checkpoint_path", None)
        entry.pop("opponent_checkpoint_resolution", None)
        entry.pop("opponent_checkpoint_file", None)
        entry.pop("opponent_wall_avoidant_safe_margin", None)
        entry["opponent_policy_kind"] = OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT
        entry["opponent_runtime_mode"] = OPPONENT_RUNTIME_MODE_NORMAL
        entry["opponent_snapshot_ref"] = (
            f"leaderboard-rank{rank}-"
            f"iteration_{row.get('iteration')}-"
            f"{str(row.get('checkpoint_id') or '')[:12]}"
        )
        tags = entry.get("tags")
        if not isinstance(tags, Mapping):
            tags = {}
        tags = dict(tags)
        tags.update(
            {
                "rank": rank,
                "source_slot": f"rank{rank}",
                "checkpoint_id": row.get("checkpoint_id"),
                "rating": row.get("rating"),
                "leaderboard_status": row.get("status"),
                "source_leaderboard_snapshot_id": leaderboard_snapshot.get("snapshot_id"),
            }
        )
        entry["tags"] = tags
        entries.append(entry)
        selected_rows.append(
            {
                "entry_name": entry.get("name"),
                "rank": rank,
                "checkpoint_id": row.get("checkpoint_id"),
                "source_checkpoint_ref": row.get("checkpoint_ref"),
                "checkpoint_ref": assigned_checkpoint_ref,
                "checkpoint_copy": checkpoint_copy,
                "rating": row.get("rating"),
                "status": row.get("status"),
            }
        )
    if missing_ranks and not allow_partial:
        raise ValueError(
            f"leaderboard missing active ranks required by assignment: {missing_ranks}"
        )
    next_assignment = {
        "schema_id": parsed["schema_id"],
        "assignment_id": assignment_id,
        "source_epoch": leaderboard_snapshot.get("generation"),
        "source_ref": source_ref,
        "seed": int(seed),
        "entries": entries,
    }
    parse_opponent_assignment_snapshot(next_assignment)
    assignment_sha256 = canonical_assignment_json_sha256(next_assignment)
    audit = {
        "schema_id": "curvyzero_opponent_assignment_audit/v0",
        "assignment_id": assignment_id,
        "assignment_sha256": assignment_sha256,
        "source_leaderboard": {
            "leaderboard_id": leaderboard_snapshot["leaderboard_id"],
            "snapshot_id": leaderboard_snapshot["snapshot_id"],
            "generation": leaderboard_snapshot.get("generation"),
            "snapshot_ref": source_ref,
            "snapshot_sha256": leaderboard_snapshot["snapshot_sha256"],
        },
        "selection": {
            "strategy_id": "preserve_recipe_rank_slots_v1",
            "seed": int(seed),
            "allow_partial": bool(allow_partial),
        },
        "selected_rows": selected_rows,
        "missing_ranks": missing_ranks,
    }
    validate_assignment_audit(audit, assignment=next_assignment)
    return next_assignment, audit


def _assignment_audit_ref_for_assignment_ref(assignment_ref: str) -> str:
    text = str(assignment_ref)
    if not text.endswith("/assignment.json"):
        raise ValueError(f"assignment ref must end with /assignment.json: {assignment_ref}")
    return f"{text[: -len('/assignment.json')]}/audit.json"


@app.function(
    image=image,
    volumes=_controller_volumes(),
    cpu=1.0,
    memory=2048,
    timeout=20 * 60,
)
def curvytron_training_candidate_refresh(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(spec or {})
    reload_errors = {
        "tournament": _reload_volume(tournament_volume),
        "control": _reload_volume(control_volume),
        "checkpoint": _reload_volume(checkpoint_volume),
    }
    reload_errors = {key: value for key, value in reload_errors.items() if value}
    if reload_errors:
        raise RuntimeError(f"training candidate refresh reload failed: {reload_errors}")
    tournament_id = runs.clean_id(str(payload["tournament_id"]), label="tournament_id")
    rating_run_id = runs.clean_id(
        str(payload.get("rating_run_id") or arena.DEFAULT_RATING_RUN_ID),
        label="rating_run_id",
    )
    leaderboard_id = runs.clean_id(
        str(payload.get("leaderboard_id") or f"{tournament_id}-{rating_run_id}-training"),
        label="leaderboard_id",
    )
    rating_snapshot = _read_best_rating_snapshot_for_run(
        TOURNAMENT_MOUNT,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        allow_live_provisional=bool(payload.get("allow_live_provisional", False)),
    )
    if not rating_snapshot:
        raise ValueError("rating snapshot not found")
    if rating_snapshot.get("provisional") and not bool(
        payload.get("allow_live_provisional", False)
    ):
        raise ValueError(
            "refusing live provisional training candidate without allow_live_provisional=True"
        )
    rating_spec = rating_snapshot.get("rating_spec")
    raw_decision_source_frames = (
        rating_spec.get("decision_source_frames") if isinstance(rating_spec, Mapping) else None
    )
    if _safe_int_or_none(raw_decision_source_frames) != 1:
        raise ValueError(
            "training candidate requires rating_spec.decision_source_frames=1; "
            f"got {raw_decision_source_frames!r}"
        )
    source_info = validate_rating_snapshot_source(
        rating_snapshot,
        expected_round_id=payload.get("expected_round_id"),
        expected_round_index=(
            int(payload["expected_round_index"])
            if payload.get("expected_round_index") is not None
            else None
        ),
        expected_rating_context_hash=payload.get("expected_rating_context_hash"),
        expected_roster_hash=payload.get("expected_roster_hash"),
        expected_rating_snapshot_sha256=payload.get("expected_rating_snapshot_sha256"),
    )
    source_short_hash = arena._short_hash(source_info["rating_snapshot_sha256"], length=12)
    snapshot_id = runs.clean_id(
        str(payload.get("snapshot_id") or f"training-candidate-{source_short_hash}"),
        label="snapshot_id",
    )
    generation = int(
        payload.get("generation")
        if payload.get("generation") is not None
        else int(rating_snapshot.get("round_index", 0) or 0)
    )
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        rating_snapshot,
        leaderboard_id=leaderboard_id,
        snapshot_id=snapshot_id,
        generation=generation,
        created_at=runs.utc_timestamp(),
        active_min_distinct_opponents=int(
            payload.get("active_min_distinct_opponents", DEFAULT_ACTIVE_MIN_DISTINCT_OPPONENTS)
        ),
        active_min_valid_games=int(
            payload.get("active_min_valid_games", DEFAULT_ACTIVE_MIN_VALID_GAMES)
        ),
        max_failure_rate=float(payload.get("max_failure_rate", DEFAULT_MAX_FAILURE_RATE)),
        max_active_rank=int(payload.get("max_active_rank", arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT)),
    )
    snapshot_context = dict(snapshot.get("context", {}))
    snapshot_context.update(
        {
            "snapshot_kind": "training_candidate",
            "rating_stable": bool(rating_snapshot.get("stable", False)),
            "rating_max_abs_delta": rating_snapshot.get("max_abs_delta"),
            "source_round_id": rating_snapshot.get("round_id"),
            "source_round_index": rating_snapshot.get("round_index"),
        }
    )
    snapshot["context"] = snapshot_context
    snapshot.pop("snapshot_sha256", None)
    snapshot = validate_leaderboard_snapshot(snapshot)
    active_count = sum(1 for row in snapshot["rows"] if row.get("status") == "active")
    if active_count < int(payload.get("min_active_count", 1)):
        raise ValueError(
            f"training candidate has too few active rows: {active_count} < "
            f"{int(payload.get('min_active_count', 1))}"
        )
    snapshot_ref = _leaderboard_snapshot_ref(leaderboard_id, snapshot_id)
    latest_ref = _leaderboard_latest_ref(leaderboard_id)
    assignment_run_id = runs.clean_id(
        str(payload.get("assignment_bank_run_id") or f"{leaderboard_id}-assignments"),
        label="assignment_bank_run_id",
    )
    assignment_attempt_id = runs.clean_id(
        str(payload.get("assignment_bank_attempt_id") or f"try-{leaderboard_id}-assignments"),
        label="assignment_bank_attempt_id",
    )
    refresh_pointers = payload.get("refresh_pointers")
    if not isinstance(refresh_pointers, Sequence) or isinstance(refresh_pointers, (str, bytes)):
        raise ValueError("refresh_pointers must be a list of pointer refs")
    if not refresh_pointers:
        raise ValueError("refresh_pointers must not be empty")
    prepared_rewrites = []
    allow_partial = bool(payload.get("allow_partial_assignment", False))
    for index, pointer_ref in enumerate(refresh_pointers):
        pointer_ref = str(pointer_ref)
        current_pointer, current_assignment, current_assignment_sha256 = (
            _validate_assignment_refresh_pointer_for_controller(pointer_ref)
        )
        current_generation = _refresh_pointer_generation(current_pointer)
        current_source = current_pointer.get("source_leaderboard")
        current_snapshot_sha256 = (
            current_source.get("snapshot_sha256") if isinstance(current_source, Mapping) else None
        )
        if current_generation is not None and current_generation > generation:
            raise ValueError(
                f"refresh pointer {pointer_ref!r} is already newer than source generation "
                f"{generation}: {current_generation}"
            )
        if (
            current_generation is not None
            and current_generation == generation
            and current_snapshot_sha256
            and str(current_snapshot_sha256) != str(snapshot["snapshot_sha256"])
        ):
            raise ValueError(
                f"refresh pointer {pointer_ref!r} already points at a different "
                f"snapshot for generation {generation}"
            )
        recipe_id = str(current_pointer.get("recipe_id") or f"recipe-{index:02d}")
        assignment_id = runs.clean_id(
            str(payload.get("assignment_id_prefix") or snapshot_id) + f"-{recipe_id}",
            label="assignment_id",
        )
        checkpoint_copies: list[Mapping[str, Any]] = []

        def rewrite_checkpoint_ref(
            entry: Mapping[str, Any],
            row: Mapping[str, Any],
            rank: int,
        ) -> tuple[str, Mapping[str, Any] | None]:
            target_ref, checkpoint_copy = _materialize_training_candidate_checkpoint_ref(
                str(row["checkpoint_ref"]),
                assignment_run_id=assignment_run_id,
                assignment_attempt_id=assignment_attempt_id,
                assignment_id=assignment_id,
                entry_name=str(entry.get("name") or f"rank{rank}"),
                rank=int(rank),
            )
            checkpoint_copies.append(checkpoint_copy)
            return target_ref, checkpoint_copy

        assignment, audit = _assignment_with_leaderboard_ranks(
            current_assignment,
            leaderboard_snapshot=snapshot,
            assignment_id=assignment_id,
            source_ref=snapshot_ref.as_posix(),
            seed=int(payload.get("assignment_seed", 0)) + index,
            allow_partial=allow_partial,
            checkpoint_ref_rewriter=rewrite_checkpoint_ref,
        )
        assignment_ref = (
            "control:"
            + (
                runs.attempt_root_ref(TRAINING_TASK_ID, assignment_run_id, assignment_attempt_id)
                / "opponents"
                / "assignments"
                / runs.clean_id(assignment_id, label="assignment_id")
                / "assignment.json"
            ).as_posix()
        )
        audit_ref = _assignment_audit_ref_for_assignment_ref(assignment_ref)
        next_pointer = {
            "schema_id": "curvyzero_opponent_assignment_refresh_pointer/v0",
            "assignment_ref": assignment_ref,
            "assignment_sha256": canonical_assignment_json_sha256(assignment),
            "recipe_id": recipe_id,
            "generation": generation,
            "previous_assignment_ref": str(current_pointer["assignment_ref"]),
            "previous_assignment_sha256": current_assignment_sha256,
            "source_leaderboard": {
                "leaderboard_id": leaderboard_id,
                "snapshot_id": snapshot_id,
                "generation": generation,
                "snapshot_ref": snapshot_ref.as_posix(),
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "rating_stable": bool(rating_snapshot.get("stable", False)),
                "rating_max_abs_delta": rating_snapshot.get("max_abs_delta"),
            },
            "recipe_signature": _assignment_entries_signature(assignment),
        }
        prepared_rewrites.append(
            {
                "recipe_id": recipe_id,
                "pointer_ref": pointer_ref,
                "previous_assignment_ref": current_pointer["assignment_ref"],
                "previous_assignment_sha256": current_assignment_sha256,
                "assignment_ref": assignment_ref,
                "assignment_sha256": next_pointer["assignment_sha256"],
                "assignment": assignment,
                "audit": audit,
                "audit_ref": audit_ref,
                "pointer": next_pointer,
                "checkpoint_copies": checkpoint_copies,
            }
        )
    snapshot_write = arena.write_json_artifact(TOURNAMENT_MOUNT, snapshot_ref, snapshot)
    latest_write = arena.write_json_artifact(TOURNAMENT_MOUNT, latest_ref, snapshot)
    tournament_commit_error = _commit_volume(tournament_volume)
    if tournament_commit_error:
        raise RuntimeError(
            f"failed to commit training candidate leaderboard: {tournament_commit_error}"
        )
    assignment_writes = []
    audit_writes = []
    for rewrite in prepared_rewrites:
        assignment_writes.append(
            _write_json_by_volume_ref(
                rewrite["assignment_ref"], rewrite["assignment"], commit=False
            )
        )
        audit_writes.append(
            _write_json_by_volume_ref(rewrite["audit_ref"], rewrite["audit"], commit=False)
        )
    assignment_commit_error = _commit_volume(control_volume)
    if assignment_commit_error:
        raise RuntimeError(
            f"failed to commit training candidate assignments: {assignment_commit_error}"
        )
    for rewrite in prepared_rewrites:
        _assert_refresh_pointer_unchanged(
            rewrite["pointer_ref"],
            expected_assignment_ref=rewrite["previous_assignment_ref"],
            expected_assignment_sha256=rewrite["previous_assignment_sha256"],
        )
    pointer_writes = []
    for rewrite in prepared_rewrites:
        pointer_writes.append(
            _write_json_by_volume_ref(rewrite["pointer_ref"], rewrite["pointer"], commit=False)
        )
    pointer_commit_error = _commit_volume(control_volume)
    if pointer_commit_error:
        raise RuntimeError(
            f"failed to commit training candidate refresh pointers: {pointer_commit_error}"
        )
    for rewrite, assignment_write, audit_write, pointer_write in zip(
        prepared_rewrites,
        assignment_writes,
        audit_writes,
        pointer_writes,
        strict=True,
    ):
        _append_tournament_lineage_event(
            stage="training_candidate_assignment_written",
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
            leaderboard_id=leaderboard_id,
            snapshot_id=snapshot_id,
            generation=generation,
            recipe_id=rewrite["recipe_id"],
            assignment_ref=rewrite["assignment_ref"],
            assignment_sha256=rewrite["assignment_sha256"],
            assignment_write=assignment_write,
            audit_ref=rewrite["audit_ref"],
            audit_write=audit_write,
            snapshot_ref=snapshot_write.get("ref"),
            snapshot_sha256=snapshot["snapshot_sha256"],
        )
        _append_tournament_lineage_event(
            stage="assignment_pointer_rewritten",
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
            leaderboard_id=leaderboard_id,
            snapshot_id=snapshot_id,
            generation=generation,
            recipe_id=rewrite["recipe_id"],
            pointer_ref=rewrite["pointer_ref"],
            previous_assignment_ref=rewrite["previous_assignment_ref"],
            assignment_ref=rewrite["assignment_ref"],
            assignment_sha256=rewrite["assignment_sha256"],
            pointer_write=pointer_write,
        )
    _commit_volume(tournament_volume)
    pointer = build_leaderboard_pointer(
        snapshot,
        snapshot_ref=snapshot_ref.as_posix(),
        published_at=runs.utc_timestamp(),
        writer={
            "kind": "curvytron_training_candidate_refresh",
            "app_name": APP_NAME,
            "rating_stable": bool(rating_snapshot.get("stable", False)),
            "rating_max_abs_delta": rating_snapshot.get("max_abs_delta"),
            "assignment_pointer_count": len(prepared_rewrites),
        },
    )
    pointer_key = _leaderboard_pointer_key(leaderboard_id)
    opponent_leaderboard_state.put(pointer_key, arena._to_plain(pointer))
    pointer_published = True
    rewritten = []
    for rewrite, assignment_write, audit_write, pointer_write in zip(
        prepared_rewrites,
        assignment_writes,
        audit_writes,
        pointer_writes,
        strict=True,
    ):
        rewritten.append(
            {
                "recipe_id": rewrite["recipe_id"],
                "pointer_ref": rewrite["pointer_ref"],
                "previous_assignment_ref": rewrite["previous_assignment_ref"],
                "assignment_ref": rewrite["assignment_ref"],
                "assignment_sha256": rewrite["assignment_sha256"],
                "assignment_write": assignment_write,
                "audit_write": audit_write,
                "pointer_write": pointer_write,
                "checkpoint_copies": rewrite["checkpoint_copies"],
            }
        )
    return arena._to_plain(
        {
            "schema_id": "curvyzero_training_candidate_refresh/v0",
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "leaderboard_id": leaderboard_id,
            "snapshot_id": snapshot_id,
            "snapshot_ref": snapshot_write.get("ref"),
            "latest_ref": latest_write.get("ref"),
            "pointer_key": pointer_key,
            "pointer_published": pointer_published,
            "rating_stable": bool(rating_snapshot.get("stable", False)),
            "rating_max_abs_delta": rating_snapshot.get("max_abs_delta"),
            "rating_source": source_info,
            "row_count": len(snapshot["rows"]),
            "active_count": active_count,
            "tournament_commit_error": tournament_commit_error,
            "assignment_commit_error": assignment_commit_error,
            "pointer_commit_error": pointer_commit_error,
            "rewritten_pointer_count": len(rewritten),
            "rewritten_pointers": rewritten,
        }
    )


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    timeout=300,
    cpu=1.0,
    memory=1024,
)
def curvytron_opponent_leaderboard_pointer_repair(
    spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(spec or {})
    _reload_volume(tournament_volume)
    leaderboard_id = runs.clean_id(str(payload["leaderboard_id"]), label="leaderboard_id")
    pointer, latest_ref, snapshot_ref = _repaired_leaderboard_pointer_from_latest(
        TOURNAMENT_MOUNT,
        leaderboard_id=leaderboard_id,
    )
    pointer_key = _leaderboard_pointer_key(leaderboard_id)
    existing_pointer = opponent_leaderboard_state.get(pointer_key, None)
    previous_pointer_status = "missing"
    if existing_pointer is not None:
        previous_pointer_status = "current"
        if not isinstance(existing_pointer, Mapping) or any(
            existing_pointer.get(key) != pointer.get(key)
            for key in ("leaderboard_id", "snapshot_id", "snapshot_ref", "snapshot_sha256")
        ):
            previous_pointer_status = "stale"
    opponent_leaderboard_state.put(pointer_key, arena._to_plain(pointer))
    return arena._to_plain(
        {
            "schema_id": "curvyzero_opponent_leaderboard_pointer_repair/v0",
            "leaderboard_id": leaderboard_id,
            "snapshot_id": pointer["snapshot_id"],
            "snapshot_sha256": pointer["snapshot_sha256"],
            "latest_ref": latest_ref.as_posix(),
            "snapshot_ref": snapshot_ref.as_posix(),
            "pointer_key": pointer_key,
            "dict_name": OPPONENT_LEADERBOARD_DICT_NAME,
            "previous_pointer_status": previous_pointer_status,
            "pointer_published": True,
            "pointer": pointer,
        }
    )


@app.function(
    image=image,
    volumes=_game_volumes(),
    timeout=10 * 60,
    cpu=1.0,
    memory=1024,
)
def curvytron_checkpoint_intake_seed(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(spec or {})
    _reload_volume(checkpoint_volume)
    _reload_volume(tournament_volume)
    tournament_id = runs.clean_id(
        str(payload.get("tournament_id") or runs.new_run_id("arena-intake")),
        label="tournament_id",
    )
    rating_run_id = runs.clean_id(
        str(payload.get("rating_run_id") or arena.DEFAULT_RATING_RUN_ID),
        label="rating_run_id",
    )
    scan_spec = {
        "checkpoint_refs": payload.get("checkpoint_refs") or "",
        "run_ids": payload.get("run_ids") or "",
        "run_id_prefix": str(payload.get("run_id_prefix") or ""),
        "max_runs": int(payload.get("max_runs") or 0),
        "checkpoint_iteration": payload.get("checkpoint_iteration"),
        "checkpoint_selection": str(
            payload.get("checkpoint_selection") or arena.CHECKPOINT_SELECTION_LATEST
        ),
    }
    live_watch = _intake_scan_spec_is_live_watch(scan_spec)
    pairs_per_round = (
        int(payload["pairs_per_round"])
        if payload.get("pairs_per_round") not in (None, "", 0, "0")
        else None
    )
    pair_selection = str(
        payload.get("pair_selection")
        or (
            arena.DEFAULT_LIVE_INTAKE_PAIR_SELECTION
            if live_watch
            else arena.DEFAULT_RATING_PAIR_SELECTION
        )
    )
    if live_watch and pairs_per_round is None:
        pairs_per_round = int(arena.DEFAULT_LIVE_INTAKE_PAIRS_PER_ROUND)
        if pair_selection == arena.RATING_PAIR_SELECTION_ALL_PAIRS:
            pair_selection = arena.DEFAULT_LIVE_INTAKE_PAIR_SELECTION
    active_pool_limit = int(
        payload.get("active_pool_limit") or arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT
    )
    if live_watch:
        active_pool_limit = min(active_pool_limit, arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT)
    rating_defaults = {
        "round_count": int(payload.get("round_count") or arena.DEFAULT_RATING_ROUND_COUNT),
        "continue_from_latest": bool(payload.get("continue_from_latest", False)) or live_watch,
        "pairs_per_round": pairs_per_round,
        "placement_min_games": (
            int(payload["placement_min_games"])
            if payload.get("placement_min_games") not in (None, "", 0, "0")
            else None
        ),
        "placement_min_opponents": int(payload.get("placement_min_opponents") or 20),
        "pair_selection": pair_selection,
        "games_per_pair": int(payload.get("games_per_pair") or arena.DEFAULT_GAMES_PER_PAIR),
        "games_per_shard": int(payload.get("games_per_shard") or arena.DEFAULT_GAMES_PER_SHARD),
        "reuse_policies_per_shard": bool(
            payload.get("reuse_policies_per_shard", arena.DEFAULT_REUSE_POLICIES_PER_SHARD)
        ),
        "seed": int(payload.get("seed") or 0),
        "max_steps": int(payload.get("max_steps") or arena.DEFAULT_MAX_STEPS),
        "decision_ms": float(payload.get("decision_ms") or arena.DEFAULT_DECISION_MS),
        "decision_source_frames": (
            int(payload["decision_source_frames"])
            if payload.get("decision_source_frames") not in (None, "", 0, "0")
            else arena.DEFAULT_DECISION_SOURCE_FRAMES
        ),
        "source_physics_step_ms": float(
            payload.get("source_physics_step_ms") or arena.DEFAULT_SOURCE_PHYSICS_STEP_MS
        ),
        "policy_mode": str(payload.get("policy_mode") or arena.POLICY_MODE_EVAL),
        "collect_temperature": float(
            payload.get("collect_temperature") or arena.DEFAULT_COLLECT_TEMPERATURE
        ),
        "collect_epsilon": float(payload.get("collect_epsilon") or arena.DEFAULT_COLLECT_EPSILON),
        "policy_trail_render_mode": payload.get("policy_trail_render_mode") or None,
        "policy_bonus_render_mode": payload.get("policy_bonus_render_mode") or None,
        "num_simulations": int(payload.get("num_simulations") or arena.DEFAULT_NUM_SIMULATIONS),
        "save_gif": bool(payload.get("save_gif", arena.DEFAULT_SAVE_GIF)),
        "gif_sample_games_per_pair": int(
            payload.get(
                "gif_sample_games_per_pair",
                arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR,
            )
        ),
        "gif_sample_strategy": str(
            payload.get("gif_sample_strategy") or arena.DEFAULT_GIF_SAMPLE_STRATEGY
        ),
        "initial_rating": float(
            payload.get("initial_rating") or arena.DEFAULT_RATING_INITIAL_RATING
        ),
        "active_pool_limit": active_pool_limit,
        "stop_when_stable": bool(payload.get("stop_when_stable", False)),
    }
    _validate_intake_rating_defaults(
        rating_defaults,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    discovery = _discover_checkpoint_refs_from_scan_spec(scan_spec, mount=RUNS_MOUNT)
    existing, _existing_manifest_load = _load_intake_manifest(
        tournament_id,
        rating_run_id,
        repair_state=True,
    )
    manifest = _intake_manifest_from_discovery(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        scan_spec=scan_spec,
        rating_defaults=rating_defaults,
        discovery=discovery,
        existing=existing if isinstance(existing, Mapping) else None,
        active=bool(payload.get("active", True)),
    )
    queue_write = {"enqueued_count": 0, "events": []}
    if bool(payload.get("enqueue_existing", False)):
        queue_write = _enqueue_checkpoint_events(
            manifest=manifest,
            checkpoint_refs=manifest["checkpoint_refs"],
            reason="seed",
        )
        manifest = _mark_intake_manifest_queued(
            manifest,
            [str(event["checkpoint_ref"]) for event in queue_write.get("events", [])],
        )
    manifest = _repair_live_intake_rating_defaults(manifest)
    state_write = _put_intake_manifest(manifest)
    manifest = dict(state_write.get("manifest") or manifest)
    manifest_write = _write_intake_manifest_artifact(manifest)
    state_write_summary = {k: v for k, v in state_write.items() if k != "manifest"}
    _write_tournament_marker(tournament_id)
    commit_error = _commit_volume(tournament_volume)
    rating_call_id = ""
    if bool(payload.get("spawn_rating", False)) and len(manifest["checkpoint_refs"]) >= 2:
        rating_spec = _intake_rating_spec_from_manifest(manifest)
        call = curvytron_rating_loop.spawn(rating_spec)
        rating_call_id = getattr(call, "object_id", None) or getattr(call, "id", None) or ""
    return arena._to_plain(
        {
            "schema_id": "curvyzero_curvytron_checkpoint_intake_seed/v0",
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "manifest_key": manifest["manifest_key"],
            "queue_partition": manifest["queue_partition"],
            "checkpoint_count": manifest["checkpoint_count"],
            "found_count": discovery.get("found_count"),
            "missing_count": discovery.get("missing_count"),
            "manifest_ref": manifest_write.get("ref"),
            "state_write": state_write_summary,
            "queue_write": queue_write,
            "rating_call_id": rating_call_id,
            "commit_error": commit_error,
            "manifest": manifest,
        }
    )


@app.function(
    image=image,
    volumes=_game_volumes(),
    timeout=10 * 60,
    cpu=1.0,
    memory=1024,
)
def curvytron_checkpoint_intake_submit(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(spec or {})
    intake_service.validate_submit_payload(payload)
    if not intake_service.submit_has_candidate_input(payload):
        raise ValueError("submit needs checkpoint_refs, run_ids, or run_id_prefix")
    _reload_volume(checkpoint_volume)
    _reload_volume(tournament_volume)
    tournament_id = runs.clean_id(str(payload["tournament_id"]), label="tournament_id")
    rating_run_id = runs.clean_id(
        str(payload.get("rating_run_id") or arena.DEFAULT_RATING_RUN_ID),
        label="rating_run_id",
    )
    manifest_key = _intake_manifest_key(tournament_id, rating_run_id)
    manifest, manifest_load = _load_intake_manifest(tournament_id, rating_run_id)
    if not isinstance(manifest, Mapping):
        raise ValueError("intake service is not configured; run mode=intake-seed first")
    if not bool(manifest.get("active", True)):
        raise ValueError("intake service is inactive")

    existing_scan_spec = (
        manifest.get("scan_spec") if isinstance(manifest.get("scan_spec"), Mapping) else {}
    )
    scan_spec = dict(existing_scan_spec)
    submitted_refs = _parse_checkpoint_refs_value(payload.get("checkpoint_refs"))
    if submitted_refs:
        submitted_refs = _validate_submitted_checkpoint_refs_exist(
            submitted_refs,
            mount=RUNS_MOUNT,
        )
        cumulative_refs = sorted(
            _clean_ref_set(manifest.get("checkpoint_refs", [])).union(submitted_refs)
        )
        # Exact ref submissions are append-only control-plane input. Make them
        # the durable scan source too, otherwise the scheduled subscriber can
        # rebuild the manifest from an older run_id scan and silently drop them.
        # If this intake is already watching run ids/prefixes, keep that watch
        # alive and pin the exact refs as additional durable seeds.
        scan_spec = _scan_spec_with_checkpoint_refs(cumulative_refs, existing_scan_spec)
        discovery = {
            "schema_id": "curvyzero_curvytron_checkpoint_discovery/v0",
            "checkpoint_volume_name": CHECKPOINT_VOLUME_NAME,
            "checkpoint_scan_glob": arena.CHECKPOINT_SCAN_GLOB,
            "checkpoint_selection": "submit",
            "selection": "submit",
            "found_count": len(cumulative_refs),
            "missing_count": 0,
            "checkpoint_refs": cumulative_refs,
            "rows": [
                _checkpoint_discovery_row_from_ref(
                    ref,
                    mount=RUNS_MOUNT,
                    found=True,
                )
                for ref in cumulative_refs
            ],
        }
    else:
        scan_spec = intake_service.merge_submit_scan_spec(
            scan_spec,
            payload,
            default_checkpoint_selection=arena.CHECKPOINT_SELECTION_LATEST,
        )
        run_discovery = _discover_checkpoint_refs_from_scan_spec(scan_spec, mount=RUNS_MOUNT)
        submitted_refs = [
            str(ref) for ref in run_discovery.get("checkpoint_refs", []) if str(ref).strip()
        ]
        cumulative_refs = sorted(
            _clean_ref_set(manifest.get("checkpoint_refs", [])).union(submitted_refs)
        )
        discovery = dict(run_discovery)
        discovery["checkpoint_refs"] = cumulative_refs
        discovery["found_count"] = len(cumulative_refs)
    if not submitted_refs:
        raise ValueError("submit did not resolve any checkpoint refs")

    previous_seen_refs = _clean_ref_set(manifest.get("seen_checkpoint_refs", []))
    new_refs = [ref for ref in submitted_refs if ref not in previous_seen_refs]
    already_seen_refs = [ref for ref in submitted_refs if ref in previous_seen_refs]
    for checkpoint_ref in new_refs:
        _append_tournament_lineage_event(
            stage="checkpoint_intake_seen",
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
            reason="submit",
            checkpoint_ref=checkpoint_ref,
            manifest_key=manifest_key,
        )
    updated_manifest = _intake_manifest_from_discovery(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        scan_spec=scan_spec,
        rating_defaults=(
            manifest.get("rating_defaults")
            if isinstance(manifest.get("rating_defaults"), Mapping)
            else {}
        ),
        discovery=discovery,
        existing=manifest,
        active=True,
    )
    queue_write = _enqueue_checkpoint_events(
        manifest=updated_manifest,
        checkpoint_refs=new_refs,
        reason="submit",
    )
    updated_manifest = _mark_intake_manifest_queued(
        updated_manifest,
        [str(event["checkpoint_ref"]) for event in queue_write.get("events", [])],
    )
    updated_manifest = _repair_live_intake_rating_defaults(updated_manifest)
    state_write = _put_intake_manifest(updated_manifest)
    updated_manifest = dict(state_write.get("manifest") or updated_manifest)
    manifest_write = _write_intake_manifest_artifact(updated_manifest)
    state_write_summary = {k: v for k, v in state_write.items() if k != "manifest"}
    commit_error = _commit_volume(tournament_volume)
    return arena._to_plain(
        {
            "schema_id": "curvyzero_curvytron_checkpoint_submission_ack/v0",
            "status": "accepted",
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "manifest_key": manifest_key,
            "queue_partition": updated_manifest["queue_partition"],
            "submitted_checkpoint_count": len(submitted_refs),
            "accepted_checkpoint_refs": new_refs,
            "already_seen_checkpoint_refs": already_seen_refs,
            "enqueued_count": int(queue_write.get("enqueued_count") or 0),
            "event_ids": [
                str(event.get("event_id") or "")
                for event in queue_write.get("events", [])
                if event.get("event_id")
            ],
            "manifest_source": manifest_load["manifest_source"],
            "manifest_state_repaired": manifest_load["manifest_state_repaired"],
            "manifest_ref": manifest_write.get("ref"),
            "state_write": state_write_summary,
            "commit_error": commit_error,
        }
    )


@app.function(
    image=image,
    volumes=_game_volumes(),
    timeout=10 * 60,
    cpu=1.0,
    memory=1024,
)
def curvytron_checkpoint_intake_tick(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(spec or {})
    _reload_volume(checkpoint_volume)
    _reload_volume(tournament_volume)
    requested_keys: list[str] = []
    if payload.get("tournament_id"):
        requested_keys.append(
            _intake_manifest_key(
                str(payload["tournament_id"]),
                str(payload.get("rating_run_id") or arena.DEFAULT_RATING_RUN_ID),
            )
        )
    else:
        active_keys = checkpoint_intake_state.get(CHECKPOINT_INTAKE_ACTIVE_KEYS, []) or []
        if isinstance(active_keys, list):
            requested_keys = [str(key) for key in active_keys]
    ticks = []
    for key in requested_keys:
        key_parts = str(key).split(":", 2)
        if len(key_parts) != 3:
            continue
        manifest, manifest_load = _load_intake_manifest(
            key_parts[1],
            key_parts[2],
        )
        if not isinstance(manifest, Mapping):
            continue
        scan_spec = manifest.get("scan_spec")
        if not isinstance(scan_spec, Mapping):
            continue
        discovery = _discover_checkpoint_refs_from_scan_spec(scan_spec, mount=RUNS_MOUNT)
        current_refs = [
            str(ref) for ref in discovery.get("checkpoint_refs", []) if str(ref).strip()
        ]
        previous_seen_refs = _clean_ref_set(manifest.get("seen_checkpoint_refs", []))
        new_refs = [ref for ref in current_refs if ref not in previous_seen_refs]
        for checkpoint_ref in new_refs:
            _append_tournament_lineage_event(
                stage="checkpoint_intake_seen",
                tournament_id=str(manifest["tournament_id"]),
                rating_run_id=str(manifest["rating_run_id"]),
                reason="tick",
                checkpoint_ref=checkpoint_ref,
                manifest_key=key,
            )
        updated_manifest = _intake_manifest_from_discovery(
            tournament_id=str(manifest["tournament_id"]),
            rating_run_id=str(manifest["rating_run_id"]),
            scan_spec=scan_spec,
            rating_defaults=(
                manifest.get("rating_defaults")
                if isinstance(manifest.get("rating_defaults"), Mapping)
                else {}
            ),
            discovery=discovery,
            existing=manifest,
            active=bool(manifest.get("active", True)),
        )
        queue_write = _enqueue_checkpoint_events(
            manifest=updated_manifest,
            checkpoint_refs=new_refs,
            reason="tick",
        )
        updated_manifest = _mark_intake_manifest_queued(
            updated_manifest,
            [str(event["checkpoint_ref"]) for event in queue_write.get("events", [])],
        )
        updated_manifest = _repair_live_intake_rating_defaults(updated_manifest)
        state_write = _put_intake_manifest(updated_manifest)
        updated_manifest = dict(state_write.get("manifest") or updated_manifest)
        _write_intake_manifest_artifact(updated_manifest)
        tick = {
            "schema_id": "curvyzero_curvytron_checkpoint_intake_tick/v0",
            "tournament_id": updated_manifest["tournament_id"],
            "rating_run_id": updated_manifest["rating_run_id"],
            "manifest_key": key,
            "updated_at": runs.utc_timestamp(),
            "checkpoint_count": len(current_refs),
            "seen_checkpoint_count": updated_manifest["seen_checkpoint_count"],
            "queued_checkpoint_count": updated_manifest["queued_checkpoint_count"],
            "new_checkpoint_count": len(new_refs),
            "new_checkpoint_refs": new_refs,
            "manifest_source": manifest_load["manifest_source"],
            "manifest_state_repaired": manifest_load["manifest_state_repaired"],
            "queue_write": queue_write,
            "discovery": {
                "found_count": discovery.get("found_count"),
                "missing_count": discovery.get("missing_count"),
                "checkpoint_selection": discovery.get("checkpoint_selection"),
            },
        }
        tick_write = _write_intake_tick_artifact(tick)
        tick["tick_ref"] = tick_write.get("ref")
        ticks.append(tick)
    commit_error = _commit_volume(tournament_volume)
    return arena._to_plain(
        {
            "schema_id": "curvyzero_curvytron_checkpoint_intake_tick_batch/v0",
            "dict_name": CHECKPOINT_INTAKE_DICT_NAME,
            "queue_name": CHECKPOINT_INTAKE_QUEUE_NAME,
            "requested_key_count": len(requested_keys),
            "tick_count": len(ticks),
            "new_checkpoint_count": sum(int(tick["new_checkpoint_count"]) for tick in ticks),
            "commit_error": commit_error,
            "ticks": ticks,
        }
    )


@app.function(
    image=image,
    volumes=_game_volumes(),
    schedule=modal.Period(seconds=DEFAULT_CHECKPOINT_INTAKE_SCAN_SECONDS),
    timeout=10 * 60,
    cpu=1.0,
    memory=1024,
    max_containers=1,
)
def curvytron_checkpoint_intake_subscriber_tick() -> dict[str, Any]:
    return curvytron_checkpoint_intake_tick.local({})


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    timeout=10 * 60,
    cpu=1.0,
    memory=1024,
)
def curvytron_checkpoint_intake_status(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(spec or {})
    key = ""
    manifest = None
    manifest_load = {
        "manifest_key": "",
        "manifest_source": "unspecified",
        "manifest_state_repaired": False,
    }
    if payload.get("tournament_id"):
        key = _intake_manifest_key(
            str(payload["tournament_id"]),
            str(payload.get("rating_run_id") or arena.DEFAULT_RATING_RUN_ID),
        )
        manifest, manifest_load = _load_intake_manifest(
            str(payload["tournament_id"]),
            str(payload.get("rating_run_id") or arena.DEFAULT_RATING_RUN_ID),
        )
    active_keys = checkpoint_intake_state.get(CHECKPOINT_INTAKE_ACTIVE_KEYS, []) or []
    if not isinstance(active_keys, list):
        active_keys = []
    queue_len = None
    if manifest and isinstance(manifest, Mapping) and manifest.get("queue_partition"):
        queue_len = checkpoint_intake_queue.len(
            partition=str(manifest.get("queue_partition") or "")
        )
    return arena._to_plain(
        {
            "schema_id": "curvyzero_curvytron_checkpoint_intake_status/v0",
            "dict_name": CHECKPOINT_INTAKE_DICT_NAME,
            "queue_name": CHECKPOINT_INTAKE_QUEUE_NAME,
            "active_manifest_keys": active_keys,
            "manifest_key": key,
            "manifest_source": manifest_load["manifest_source"],
            "manifest_state_repaired": manifest_load["manifest_state_repaired"],
            "queue_len": queue_len,
            "manifest": manifest if isinstance(manifest, Mapping) else None,
        }
    )


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    timeout=10 * 60,
    cpu=1.0,
    memory=1024,
)
def curvytron_checkpoint_intake_drain(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(spec or {})
    _reload_volume(tournament_volume)
    tournament_id = runs.clean_id(str(payload["tournament_id"]), label="tournament_id")
    rating_run_id = runs.clean_id(
        str(payload.get("rating_run_id") or arena.DEFAULT_RATING_RUN_ID),
        label="rating_run_id",
    )
    manifest_key = _intake_manifest_key(tournament_id, rating_run_id)
    manifest, manifest_load = _load_intake_manifest(tournament_id, rating_run_id)
    if not isinstance(manifest, Mapping):
        raise ValueError("intake manifest not found; run mode=intake-seed first")
    max_events = max(0, int(payload.get("max_events") or 100))
    partition = str(manifest["queue_partition"])
    rating_call_id = ""
    existing_rating_run = _rating_run_has_existing_output(
        TOURNAMENT_MOUNT,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    rating_writer_finished = (
        _rating_writer_has_finished(
            TOURNAMENT_MOUNT,
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
        )
        if existing_rating_run
        else True
    )
    rating_defaults = manifest.get("rating_defaults")
    if not isinstance(rating_defaults, Mapping):
        rating_defaults = {}
    scan_spec = manifest.get("scan_spec")
    scan_spec_is_live_watch = (
        _intake_scan_spec_is_live_watch(scan_spec) if isinstance(scan_spec, Mapping) else False
    )
    requested_continue_from_latest = bool(payload.get("continue_from_latest", False))
    manifest_continue_from_latest = bool(rating_defaults.get("continue_from_latest", False))
    continue_from_latest = (
        requested_continue_from_latest
        or manifest_continue_from_latest
        or scan_spec_is_live_watch
    )
    continuation_reason = ""
    if (
        continue_from_latest
        and not requested_continue_from_latest
        and scan_spec_is_live_watch
    ):
        continuation_reason = "live_watch"
    allow_existing_rating = continue_from_latest
    rating_checkpoint_refs = _intake_manifest_rating_checkpoint_refs(
        manifest,
        continue_from_latest=continue_from_latest,
    )
    latest_rating_checkpoint_refs = (
        _rating_latest_checkpoint_refs(
            TOURNAMENT_MOUNT,
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
        )
        if continue_from_latest
        else []
    )
    desired_pool_new_checkpoint_refs = sorted(
        set(rating_checkpoint_refs) - set(latest_rating_checkpoint_refs)
    )
    desired_rating_overrides = intake_service.rating_overrides_from_payload(
        payload,
        allow_rating_overrides=bool(payload.get("allow_rating_overrides", True)),
    )
    desired_rating_overrides["continue_from_latest"] = continue_from_latest
    desired_rating_spec = _intake_rating_spec_from_manifest(
        manifest,
        overrides=desired_rating_overrides,
    )
    desired_rating_spec_checkpoints = desired_rating_spec.get("checkpoints") or []
    desired_rating_spec_checkpoint_count = (
        len(desired_rating_spec_checkpoints)
        if isinstance(desired_rating_spec_checkpoints, Sequence)
        and not isinstance(desired_rating_spec_checkpoints, (str, bytes))
        else 0
    )
    desired_rating_spec_pool_hash = arena.rating_pool_hash(
        desired_rating_spec_checkpoints
    )
    queue_len_before = checkpoint_intake_queue.len(partition=partition)
    queue_len_after_repair = queue_len_before
    queue_repair = {"partition": partition, "enqueued_count": 0, "events": []}
    spawn_requested = bool(payload.get("spawn_rating", False))
    spawn_if_empty = bool(payload.get("spawn_if_empty", False))
    claim_key = _intake_rating_claim_key(
        manifest,
        continue_from_latest=continue_from_latest,
    )
    claim_pool_hash = _checkpoint_ref_pool_hash(rating_checkpoint_refs)
    rating_claimed = False
    rating_claim_stale = False
    rating_claim_repaired = False
    spawn_skipped_reason = ""
    rating_result = None
    rating_recovery_round = None
    rating_recovery_claim_key = ""
    rating_recovery_claimed = False
    rating_recovery_claim_stale = False
    rating_recovery_reduce_ready = False
    rating_recovery_partial_reduce_recommended = False
    rating_recovery_skip_decision = None
    rating_recovery_skipped_progress = None
    rating_recovery_skipped_progresses = []
    raw_claim_stale_after_seconds = payload.get(
        "claim_stale_after_seconds",
        DEFAULT_CHECKPOINT_INTAKE_CLAIM_STALE_SECONDS,
    )
    raw_rating_round_stale_after_seconds = payload.get(
        "rating_round_stale_after_seconds",
        DEFAULT_RATING_ROUND_STALE_SECONDS,
    )
    claim_stale_after_seconds = max(
        0,
        int(
            DEFAULT_CHECKPOINT_INTAKE_CLAIM_STALE_SECONDS
            if raw_claim_stale_after_seconds in (None, "")
            else raw_claim_stale_after_seconds
        ),
    )
    rating_round_stale_after_seconds = max(
        0,
        int(
            DEFAULT_RATING_ROUND_STALE_SECONDS
            if raw_rating_round_stale_after_seconds in (None, "")
            else raw_rating_round_stale_after_seconds
        ),
    )
    if (
        spawn_requested
        and existing_rating_run
        and continue_from_latest
        and not rating_writer_finished
    ):
        seen_recovery_rounds: set[str] = set()
        while not rating_writer_finished:
            rating_recovery_round = _oldest_unrated_rating_round(
                TOURNAMENT_MOUNT,
                tournament_id=tournament_id,
                rating_run_id=rating_run_id,
            )
            if not rating_recovery_round:
                break
            recovery_round_id = str(rating_recovery_round["round_id"])
            if recovery_round_id in seen_recovery_rounds:
                break
            seen_recovery_rounds.add(recovery_round_id)
            rating_recovery_skip_decision = _rating_round_skip_decision(
                TOURNAMENT_MOUNT,
                tournament_id=tournament_id,
                rating_run_id=rating_run_id,
                round_id=recovery_round_id,
                desired_checkpoint_count=len(rating_checkpoint_refs),
                latest_checkpoint_count=len(latest_rating_checkpoint_refs),
                desired_rating_spec=desired_rating_spec,
                stale_after_seconds=rating_round_stale_after_seconds,
                scan_output_progress=True,
            )
            recovery_game_count = int(rating_recovery_skip_decision.get("game_count") or 0)
            recovery_completed_game_count = int(
                rating_recovery_skip_decision.get("completed_game_count") or 0
            )
            rating_recovery_reduce_ready = bool(
                recovery_game_count > 0 and recovery_completed_game_count >= recovery_game_count
            )
            rating_recovery_partial_reduce_recommended = bool(
                rating_recovery_skip_decision.get("partial_reduce_recommended")
            )
            if bool(rating_recovery_skip_decision.get("skip")):
                rating_recovery_skipped_progress = _write_rating_round_skipped_progress(
                    TOURNAMENT_MOUNT,
                    tournament_id=tournament_id,
                    rating_run_id=rating_run_id,
                    round_id=recovery_round_id,
                    round_index=int(rating_recovery_round["round_index"]),
                    skip_decision=rating_recovery_skip_decision,
                )
                rating_recovery_skipped_progresses.append(rating_recovery_skipped_progress)
                _commit_volume(tournament_volume)
                rating_writer_finished = _rating_writer_has_finished(
                    TOURNAMENT_MOUNT,
                    tournament_id=tournament_id,
                    rating_run_id=rating_run_id,
                )
                continue
            break
    if spawn_requested:
        if len(rating_checkpoint_refs) < 2:
            spawn_skipped_reason = "needs_at_least_two_checkpoints"
        elif existing_rating_run and not allow_existing_rating:
            spawn_skipped_reason = "rating_run_already_exists"
        elif existing_rating_run and continue_from_latest and not rating_writer_finished:
            if rating_recovery_round and (
                rating_recovery_reduce_ready or rating_recovery_partial_reduce_recommended
            ):
                allow_partial_reduce = bool(
                    rating_recovery_partial_reduce_recommended
                    and not rating_recovery_reduce_ready
                )
                rating_recovery_claim_key = _intake_rating_reduce_claim_key(
                    tournament_id=tournament_id,
                    rating_run_id=rating_run_id,
                    round_id=str(rating_recovery_round["round_id"]),
                )
                existing_recovery_claim = checkpoint_intake_state.get(
                    rating_recovery_claim_key,
                    None,
                )
                rating_recovery_claim_stale = _intake_rating_claim_is_stale(
                    existing_recovery_claim,
                    stale_after_seconds=claim_stale_after_seconds,
                )
                claim_payload = {
                    "schema_id": ("curvyzero_curvytron_checkpoint_intake_rating_reduce_claim/v0"),
                    "manifest_key": manifest_key,
                    "tournament_id": tournament_id,
                    "rating_run_id": rating_run_id,
                    "round_id": rating_recovery_round["round_id"],
                    "round_index": rating_recovery_round["round_index"],
                    "queue_len_before": queue_len_before,
                    "allow_partial_reduce": allow_partial_reduce,
                    "stale_after_seconds": claim_stale_after_seconds,
                    "created_at": runs.utc_timestamp(),
                }
                if rating_recovery_claim_stale and isinstance(
                    existing_recovery_claim,
                    Mapping,
                ):
                    claim_payload["replaces_stale_claim_created_at"] = str(
                        existing_recovery_claim.get("created_at") or ""
                    )
                if rating_recovery_claim_stale:
                    rating_recovery_claimed = checkpoint_intake_state.put(
                        rating_recovery_claim_key,
                        claim_payload,
                    )
                else:
                    rating_recovery_claimed = checkpoint_intake_state.put(
                        rating_recovery_claim_key,
                        claim_payload,
                        skip_if_exists=True,
                    )
                if rating_recovery_claimed:
                    call = curvytron_rating_reduce.spawn(
                        {
                            "tournament_id": tournament_id,
                            "rating_run_id": rating_run_id,
                            "round_index": int(rating_recovery_round["round_index"]),
                            "allow_partial": allow_partial_reduce,
                        }
                    )
                    rating_call_id = (
                        getattr(call, "object_id", None) or getattr(call, "id", None) or ""
                    )
                    spawn_skipped_reason = (
                        "spawned_unfinished_round_partial_reduce"
                        if allow_partial_reduce
                        else "spawned_unfinished_round_reduce"
                    )
                else:
                    spawn_skipped_reason = "rating_writer_not_finished_reduce_claim_exists"
            elif rating_recovery_round:
                spawn_skipped_reason = "rating_writer_not_finished_round_running"
            else:
                spawn_skipped_reason = "rating_writer_not_finished"
        else:
            existing_claim = checkpoint_intake_state.get(claim_key, None)
            repair_refs = _intake_queue_repair_refs(manifest)
            if continue_from_latest and latest_rating_checkpoint_refs:
                desired_pool_new_ref_set = set(desired_pool_new_checkpoint_refs)
                repair_refs = [ref for ref in repair_refs if ref in desired_pool_new_ref_set]
            rating_claim_stale = _intake_rating_claim_is_stale(
                existing_claim,
                stale_after_seconds=claim_stale_after_seconds,
            )
            if continue_from_latest and _intake_rating_claim_needs_pool_repair(
                existing_claim,
                pool_hash=claim_pool_hash,
                checkpoint_count=len(rating_checkpoint_refs),
            ):
                rating_claim_stale = True
            if (
                existing_claim is not None
                and not rating_claim_stale
                and continue_from_latest
                and rating_writer_finished
                and (spawn_if_empty or queue_len_before or repair_refs)
            ):
                rating_claim_stale = True
            if existing_claim is not None and not rating_claim_stale:
                spawn_skipped_reason = "rating_run_claim_exists"
            elif not queue_len_before and not spawn_if_empty and not repair_refs:
                spawn_skipped_reason = "no_queued_events"
            else:
                claim_payload = {
                    "schema_id": "curvyzero_curvytron_checkpoint_intake_rating_claim/v0",
                    "manifest_key": manifest_key,
                    "tournament_id": tournament_id,
                    "rating_run_id": rating_run_id,
                    "queue_len_before": queue_len_before,
                    "checkpoint_count": len(rating_checkpoint_refs),
                    "pool_hash": claim_pool_hash,
                    "stale_after_seconds": claim_stale_after_seconds,
                    "created_at": runs.utc_timestamp(),
                }
                if rating_claim_stale and isinstance(existing_claim, Mapping):
                    claim_payload["replaces_stale_claim_created_at"] = str(
                        existing_claim.get("created_at") or ""
                    )
                if rating_claim_stale:
                    rating_claimed = checkpoint_intake_state.put(claim_key, claim_payload)
                    rating_claim_repaired = bool(rating_claimed)
                else:
                    rating_claimed = checkpoint_intake_state.put(
                        claim_key,
                        claim_payload,
                        skip_if_exists=True,
                    )
                if not rating_claimed:
                    spawn_skipped_reason = "rating_run_claim_exists"
    events = []
    if max_events and (not spawn_requested or rating_claimed):
        if spawn_requested and rating_claimed and not queue_len_before and not spawn_if_empty:
            repair_refs = _intake_queue_repair_refs(manifest)
            if continue_from_latest and latest_rating_checkpoint_refs:
                desired_pool_new_ref_set = set(desired_pool_new_checkpoint_refs)
                repair_refs = [ref for ref in repair_refs if ref in desired_pool_new_ref_set]
            if repair_refs:
                queue_repair = _enqueue_checkpoint_events(
                    manifest=manifest,
                    checkpoint_refs=repair_refs,
                    reason="repair_missing_queue_events",
                )
                queue_len_after_repair = queue_len_before + int(
                    queue_repair.get("enqueued_count") or 0
                )
        events = (
            checkpoint_intake_queue.get_many(
                max_events,
                block=False,
                partition=partition,
            )
            or []
        )
    should_spawn = spawn_requested and rating_claimed and (bool(events) or spawn_if_empty)
    if spawn_requested and rating_claimed and not should_spawn and not spawn_skipped_reason:
        spawn_skipped_reason = "no_drained_events"
    if rating_claimed:
        final_claim_payload = {
            "schema_id": "curvyzero_curvytron_checkpoint_intake_rating_claim/v0",
            "manifest_key": manifest_key,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "queue_len_before": queue_len_before,
            "queue_len_after_repair": queue_len_after_repair,
            "event_count": len(events),
            "checkpoint_count": len(rating_checkpoint_refs),
            "pool_hash": claim_pool_hash,
            "stale_after_seconds": claim_stale_after_seconds,
            "repaired_stale_claim": rating_claim_repaired,
            "created_at": runs.utc_timestamp(),
        }
        if rating_claim_repaired and isinstance(existing_claim, Mapping):
            final_claim_payload["replaces_stale_claim_created_at"] = str(
                existing_claim.get("created_at") or ""
            )
        checkpoint_intake_state.put(claim_key, final_claim_payload)
    if should_spawn:
        call = curvytron_rating_loop.spawn(desired_rating_spec)
        rating_call_id = getattr(call, "object_id", None) or getattr(call, "id", None) or ""
        if bool(payload.get("wait_for_rating", False)):
            rating_result = call.get()
    lineage_commit_error = None
    if rating_recovery_claimed:
        _append_tournament_lineage_event(
            stage="rating_spawn_claimed",
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
            status="ok",
            reason="unfinished_round_reduce",
            manifest_key=manifest_key,
            claim_key=rating_recovery_claim_key,
            claim_kind="rating_reduce_recovery",
            round_id=(
                str(rating_recovery_round.get("round_id"))
                if isinstance(rating_recovery_round, Mapping)
                else None
            ),
            round_index=(
                rating_recovery_round.get("round_index")
                if isinstance(rating_recovery_round, Mapping)
                else None
            ),
            rating_call_id=rating_call_id,
            queue_partition=partition,
            queue_len_before=queue_len_before,
            event_count=len(events),
            checkpoint_count=len(rating_checkpoint_refs),
            continue_from_latest=continue_from_latest,
        )
        lineage_commit_error = _commit_volume(tournament_volume)
    if rating_claimed:
        _append_tournament_lineage_event(
            stage="rating_spawn_claimed",
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
            status="ok" if should_spawn else "claimed_no_spawn",
            reason=None if should_spawn else spawn_skipped_reason,
            manifest_key=manifest_key,
            claim_key=claim_key,
            claim_kind="rating_loop",
            rating_call_id=rating_call_id,
            queue_partition=partition,
            queue_len_before=queue_len_before,
            queue_len_after_repair=queue_len_after_repair,
            event_count=len(events),
            checkpoint_count=len(rating_checkpoint_refs),
            pool_hash=claim_pool_hash,
            repaired_stale_claim=rating_claim_repaired,
            continue_from_latest=continue_from_latest,
        )
        lineage_commit_error = _commit_volume(tournament_volume)
    return arena._to_plain(
        {
            "schema_id": "curvyzero_curvytron_checkpoint_intake_drain/v0",
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "manifest_key": manifest_key,
            "manifest_source": manifest_load["manifest_source"],
            "manifest_state_repaired": manifest_load["manifest_state_repaired"],
            "queue_partition": manifest["queue_partition"],
            "queue_len_before": queue_len_before,
            "queue_len_after_repair": queue_len_after_repair,
            "queue_repair": queue_repair,
            "event_count": len(events),
            "events": events,
            "checkpoint_count": len(manifest.get("checkpoint_refs") or []),
            "rating_checkpoint_count": len(rating_checkpoint_refs),
            "desired_rating_spec_checkpoint_count": desired_rating_spec_checkpoint_count,
            "desired_rating_spec_pool_hash": desired_rating_spec_pool_hash,
            "latest_rating_checkpoint_count": len(latest_rating_checkpoint_refs),
            "desired_pool_new_checkpoint_count": len(desired_pool_new_checkpoint_refs),
            "existing_rating_run": existing_rating_run,
            "rating_writer_finished": rating_writer_finished,
            "continue_from_latest": continue_from_latest,
            "requested_continue_from_latest": requested_continue_from_latest,
            "continuation_reason": continuation_reason,
            "rating_claim_key": claim_key,
            "claim_stale_after_seconds": claim_stale_after_seconds,
            "rating_round_stale_after_seconds": rating_round_stale_after_seconds,
            "rating_claimed": rating_claimed,
            "rating_claim_stale": rating_claim_stale,
            "rating_claim_repaired": rating_claim_repaired,
            "rating_recovery_round": rating_recovery_round,
            "rating_recovery_claim_key": rating_recovery_claim_key,
            "rating_recovery_claimed": rating_recovery_claimed,
            "rating_recovery_claim_stale": rating_recovery_claim_stale,
            "rating_recovery_reduce_ready": rating_recovery_reduce_ready,
            "rating_recovery_partial_reduce_recommended": (
                rating_recovery_partial_reduce_recommended
            ),
            "rating_recovery_skip_decision": rating_recovery_skip_decision,
            "rating_recovery_skipped_progress": rating_recovery_skipped_progress,
            "rating_recovery_skipped_progresses": rating_recovery_skipped_progresses,
            "spawn_skipped_reason": spawn_skipped_reason,
            "rating_call_id": rating_call_id,
            "rating_result": rating_result,
            "lineage_commit_error": lineage_commit_error,
        }
    )


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    schedule=modal.Period(seconds=DEFAULT_CHECKPOINT_INTAKE_SCAN_SECONDS),
    timeout=10 * 60,
    cpu=1.0,
    memory=1024,
    max_containers=1,
)
def curvytron_checkpoint_intake_drain_tick() -> dict[str, Any]:
    active_keys = checkpoint_intake_state.get(CHECKPOINT_INTAKE_ACTIVE_KEYS, []) or []
    if not isinstance(active_keys, list):
        active_keys = []
    results = []
    for key in active_keys:
        manifest = checkpoint_intake_state.get(str(key), None)
        if not isinstance(manifest, Mapping) or not bool(manifest.get("active", True)):
            continue
        rating_defaults = manifest.get("rating_defaults")
        if not isinstance(rating_defaults, Mapping):
            rating_defaults = {}
        scan_spec = manifest.get("scan_spec")
        scan_spec_is_live_watch = (
            _intake_scan_spec_is_live_watch(scan_spec) if isinstance(scan_spec, Mapping) else False
        )
        continue_from_latest = (
            bool(rating_defaults.get("continue_from_latest", False)) or scan_spec_is_live_watch
        )
        tournament_id = str(manifest["tournament_id"])
        rating_run_id = str(manifest["rating_run_id"])
        leaderboard_id = f"{tournament_id}-{rating_run_id}-training"
        status = _feedback_loop_status_payload(
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
            leaderboard_id=leaderboard_id,
            lookahead_batches=64,
            status_activity_probe_pairs=8,
        )
        decision = _feedback_loop_control_decision(status, action="drain-if-ready")
        if not bool(decision.get("spawn_drain")):
            results.append(
                {
                    "manifest_key": str(key),
                    "tournament_id": tournament_id,
                    "rating_run_id": rating_run_id,
                    "status": status.get("status"),
                    "decision": decision,
                    "drain_call_id": "",
                    "spawned": False,
                }
            )
            continue
        drain_spec = {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "max_events": max(100, len(manifest.get("checkpoint_refs") or [])),
            "spawn_rating": True,
            "continue_from_latest": continue_from_latest,
            "spawn_if_existing": continue_from_latest,
            "rating_round_stale_after_seconds": DEFAULT_RATING_ROUND_STALE_SECONDS,
        }
        call = curvytron_checkpoint_intake_drain.spawn(drain_spec)
        results.append(
            {
                "manifest_key": str(key),
                "tournament_id": drain_spec["tournament_id"],
                "rating_run_id": drain_spec["rating_run_id"],
                "drain_call_id": getattr(call, "object_id", None) or getattr(call, "id", None) or "",
                "max_events": drain_spec["max_events"],
                "continue_from_latest": continue_from_latest,
                "status": status.get("status"),
                "decision": decision,
                "spawned": True,
            }
        )
    return arena._to_plain(
        {
            "schema_id": "curvyzero_curvytron_checkpoint_intake_drain_tick/v0",
            "active_manifest_count": len(active_keys),
            "spawned_drain_count": sum(1 for row in results if row.get("spawned")),
            "drain_call_ids": [
                row.get("drain_call_id") for row in results if row.get("drain_call_id")
            ],
            "results": results,
        }
    )


def _reject_submit_cli_scheduler_overrides(**values: Any) -> None:
    defaults = {
        "checkpoint_iteration": -1,
        "checkpoint_selection": arena.CHECKPOINT_SELECTION_LATEST,
        "games_per_pair": arena.DEFAULT_GAMES_PER_PAIR,
        "games_per_shard": arena.DEFAULT_GAMES_PER_SHARD,
        "reuse_policies_per_shard": arena.DEFAULT_REUSE_POLICIES_PER_SHARD,
        "round_count": arena.DEFAULT_RATING_ROUND_COUNT,
        "continue_from_latest": False,
        "pairs_per_round": 0,
        "placement_min_games": 0,
        "placement_min_opponents": 20,
        "pair_selection": arena.DEFAULT_RATING_PAIR_SELECTION,
        "initial_rating": arena.DEFAULT_RATING_INITIAL_RATING,
        "active_pool_limit": arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT,
        "stop_when_stable": False,
        "seed": 0,
        "max_steps": arena.DEFAULT_MAX_STEPS,
        "decision_ms": arena.DEFAULT_DECISION_MS,
        "decision_source_frames": arena.DEFAULT_DECISION_SOURCE_FRAMES,
        "source_physics_step_ms": arena.DEFAULT_SOURCE_PHYSICS_STEP_MS,
        "policy_mode": arena.POLICY_MODE_EVAL,
        "collect_temperature": arena.DEFAULT_COLLECT_TEMPERATURE,
        "collect_epsilon": arena.DEFAULT_COLLECT_EPSILON,
        "policy_trail_render_mode": arena.DEFAULT_POLICY_TRAIL_RENDER_MODE,
        "policy_bonus_render_mode": arena.DEFAULT_POLICY_BONUS_RENDER_MODE,
        "num_simulations": arena.DEFAULT_NUM_SIMULATIONS,
        "save_gif": arena.DEFAULT_SAVE_GIF,
        "gif_sample_games_per_pair": arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR,
        "gif_sample_strategy": arena.DEFAULT_GIF_SAMPLE_STRATEGY,
        "intake_enqueue_existing": False,
        "intake_spawn_rating": False,
        "intake_spawn_if_existing": False,
        "intake_allow_rating_overrides": False,
        "intake_max_events": 100,
        "intake_claim_stale_after_seconds": DEFAULT_CHECKPOINT_INTAKE_CLAIM_STALE_SECONDS,
        "intake_rating_round_stale_after_seconds": DEFAULT_RATING_ROUND_STALE_SECONDS,
        "intake_active": True,
        "max_runs": 0,
    }
    for key, default in defaults.items():
        value = values.get(key, default)
        if value != default:
            raise ValueError(
                "tournament-submit only accepts candidate refs/run ids; "
                f"configure scheduler policy with intake-seed, not {key}"
            )


CURRENT_LANE_CLI_MODES = {
    "intake-seed",
    "intake-submit",
    "tournament-submit",
    "intake-status",
    "intake-drain",
    "progress",
    "provisional",
    "provisional-loop",
    "reduce",
    "leaderboard-publish",
    "leaderboard-pointer-repair",
    "training-candidate-refresh",
    "training-candidate-auto-refresh",
    "current",
    "status-current",
    "loop-status",
    "loop-control",
}


def _cli_tournament_id_for_mode(*, mode: str, tournament_id: str) -> str:
    clean_tournament_id = str(tournament_id or "").strip()
    if clean_tournament_id:
        return clean_tournament_id
    if mode in CURRENT_LANE_CLI_MODES:
        return CURRENT_TOURNAMENT_ID
    return runs.new_run_id("arena")


def _cli_rating_run_id_for_mode(
    *,
    mode: str,
    tournament_id: str,
    rating_run_id: str,
) -> str:
    clean_tournament_id = str(tournament_id or "").strip()
    clean_rating_run_id = str(rating_run_id or "").strip()
    if (
        mode in CURRENT_LANE_CLI_MODES
        and clean_rating_run_id in {"", arena.DEFAULT_RATING_RUN_ID}
        and (not clean_tournament_id or clean_tournament_id == CURRENT_TOURNAMENT_ID)
    ):
        return CURRENT_RATING_RUN_ID
    return clean_rating_run_id or arena.DEFAULT_RATING_RUN_ID


UNSAFE_LOCAL_ENTRYPOINT_CONTROL_MODES = {"loop-status", "loop-control", "intake-drain"}


def _assert_safe_local_entrypoint_mode(mode: str) -> None:
    if mode not in UNSAFE_LOCAL_ENTRYPOINT_CONTROL_MODES:
        return
    raise ValueError(
        f"mode={mode!r} is unsafe through modal run for live control because it "
        "creates a temporary scheduled Modal app. Use: uv run --extra modal "
        "python scripts/curvytron_live_loop_control.py --action status "
        "--activity-probe-pairs 4 --lookahead-batches 12"
    )


def _current_lane_static_status() -> dict[str, Any]:
    return arena._to_plain(
        {
            "schema_id": "curvyzero_curvytron_current_lane_status/v0",
            "app": {
                "tournament": APP_NAME,
                "trainer": TRAIN_APP_NAME,
                "gif_browser": GIF_BROWSER_APP_NAME,
            },
            "volumes": {
                "runs": CHECKPOINT_VOLUME_NAME,
                "tournaments": TOURNAMENT_VOLUME_NAME,
                "control": CONTROL_VOLUME_NAME,
            },
            "coordination": {
                "checkpoint_intake_dict": CHECKPOINT_INTAKE_DICT_NAME,
                "checkpoint_intake_queue": CHECKPOINT_INTAKE_QUEUE_NAME,
                "opponent_leaderboard_dict": OPPONENT_LEADERBOARD_DICT_NAME,
            },
            "current": {
                "tournament_id": CURRENT_TOURNAMENT_ID,
                "rating_run_id": CURRENT_RATING_RUN_ID,
                "training_candidate_refresh_config_ref": TRAINING_CANDIDATE_REFRESH_CONFIG_REF,
                "training_candidate_refresh_pointers": list(TRAINING_CANDIDATE_REFRESH_POINTERS),
                "assignment_bank_run_id": TRAINING_CANDIDATE_ASSIGNMENT_BANK_RUN_ID,
                "assignment_bank_attempt_id": (TRAINING_CANDIDATE_ASSIGNMENT_BANK_ATTEMPT_ID),
            },
            "automation": {
                "checkpoint_intake_subscriber_tick_seconds": (
                    DEFAULT_CHECKPOINT_INTAKE_SCAN_SECONDS
                ),
                "checkpoint_intake_drain_tick_seconds": (DEFAULT_CHECKPOINT_INTAKE_SCAN_SECONDS),
                "training_candidate_refresh_tick_seconds": (TRAINING_CANDIDATE_REFRESH_SECONDS),
                "rating_continuation": "spawned_by_intake_drain",
                "trainer_assignment_refresh": "per_launched_trainer_with_refresh_pointer",
            },
            "manual": [
                "trainer batch launch",
                "initial intake seed/config changes",
                "app deploy/redeploy",
                "visibility cleanup",
                "storage/app purge",
                "debug/repair modes",
            ],
            "tournament_defaults": {
                "games_per_pair": arena.DEFAULT_GAMES_PER_PAIR,
                "active_pool_limit": arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT,
                "live_intake_pair_selection": arena.DEFAULT_LIVE_INTAKE_PAIR_SELECTION,
                "live_intake_pairs_per_round": (arena.DEFAULT_LIVE_INTAKE_PAIRS_PER_ROUND),
                "decision_source_frames": arena.DEFAULT_DECISION_SOURCE_FRAMES,
                "max_steps": arena.DEFAULT_MAX_STEPS,
                "save_gif": arena.DEFAULT_SAVE_GIF,
                "gif_sample_games_per_pair": (arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR),
                "gif_fps": arena.DEFAULT_GIF_FPS,
            },
        }
    )


def _status_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rating_snapshot_status_summary(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(snapshot, Mapping) or not snapshot:
        return {
            "exists": False,
            "round_id": "",
            "round_index": -1,
            "checkpoint_count": 0,
            "rating_count": 0,
            "active_count": 0,
            "max_checkpoint_iteration": None,
            "stable": False,
            "max_abs_delta": None,
        }
    rows = snapshot.get("ratings")
    rating_rows = []
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        rating_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    active_count = sum(1 for row in rating_rows if str(row.get("status") or "") == "active")
    iterations = [
        int(row["iteration"])
        for row in rating_rows
        if _safe_int_or_none(row.get("iteration")) is not None
    ]
    return {
        "exists": True,
        "round_id": str(snapshot.get("round_id") or ""),
        "round_index": _status_int(snapshot.get("round_index"), -1),
        "checkpoint_count": _status_int(
            snapshot.get("checkpoint_count"),
            len(rating_rows),
        ),
        "rating_count": len(rating_rows),
        "active_count": active_count,
        "max_checkpoint_iteration": max(iterations) if iterations else None,
        "stable": bool(snapshot.get("stable", False)),
        "max_abs_delta": snapshot.get("max_abs_delta"),
        "created_at": snapshot.get("created_at"),
        "updated_at": snapshot.get("updated_at"),
    }


def _rating_game_batch_status_summary(
    mount: Path,
    *,
    tournament_id: str,
    rating_run_id: str,
    round_index: int,
) -> dict[str, Any]:
    round_id = arena.rating_round_id(int(round_index))
    input_ref = arena.rating_round_input_ref(tournament_id, rating_run_id, round_id)
    progress_ref = arena.rating_round_progress_ref(tournament_id, rating_run_id, round_id)
    ratings_ref = arena.rating_round_ratings_ref(tournament_id, rating_run_id, round_id)
    input_path = runs.volume_path(mount, input_ref)
    progress_path = runs.volume_path(mount, progress_ref)
    ratings_path = runs.volume_path(mount, ratings_ref)
    input_payload = _read_json(input_path)
    progress = _read_json(progress_path)
    ratings = _read_json(ratings_path)
    input_exists = input_path.exists()
    progress_exists = progress_path.exists()
    ratings_written = ratings_path.exists() and bool(ratings)
    if ratings_written:
        status = "complete"
        phase = "ratings_written"
    elif str(progress.get("status") or ""):
        status = str(progress.get("status") or "")
        phase = str(progress.get("phase") or "")
    elif input_exists:
        status = "running"
        phase = "input_written"
    else:
        status = "missing"
        phase = "missing"
    checkpoint_count = _status_int(input_payload.get("checkpoint_count"), 0)
    rating_spec_checkpoint_count = 0
    checkpoint_roster_count = 0
    rating_spec = input_payload.get("rating_spec")
    if not isinstance(rating_spec, Mapping):
        rating_spec = {}
    rating_spec_checkpoints = rating_spec.get("checkpoints")
    if isinstance(rating_spec_checkpoints, Sequence) and not isinstance(
        rating_spec_checkpoints,
        (str, bytes),
    ):
        rating_spec_checkpoint_count = len(rating_spec_checkpoints)
    roster = input_payload.get("checkpoint_roster")
    if isinstance(roster, Mapping):
        checkpoint_roster_count = len(roster)
    if not checkpoint_count:
        checkpoints = input_payload.get("checkpoints")
        if isinstance(checkpoints, Sequence) and not isinstance(checkpoints, (str, bytes)):
            checkpoint_count = len(checkpoints)
        elif rating_spec_checkpoint_count:
            checkpoint_count = rating_spec_checkpoint_count
        elif checkpoint_roster_count:
            checkpoint_count = checkpoint_roster_count
    pair_count = _status_int(input_payload.get("pair_count"), _status_int(progress.get("pair_count")))
    game_count = _status_int(input_payload.get("game_count"), _status_int(progress.get("game_count")))
    completed_game_count = _status_int(progress.get("completed_game_count"))
    completed_pair_count = _status_int(progress.get("completed_pair_count"))
    skip_decision = progress.get("skip_decision")
    skip_summary: dict[str, Any] | None = None
    if isinstance(skip_decision, Mapping):
        skip_summary = {
            "reason": skip_decision.get("reason") or progress.get("skip_reason"),
            "input_checkpoint_count": _status_int(skip_decision.get("input_checkpoint_count")),
            "desired_checkpoint_count": _status_int(skip_decision.get("desired_checkpoint_count")),
            "pair_count": _status_int(skip_decision.get("pair_count")),
            "game_count": _status_int(skip_decision.get("game_count")),
            "completed_game_count": _status_int(skip_decision.get("completed_game_count")),
            "started_pair_count": _status_int(skip_decision.get("started_pair_count")),
            "stale_after_seconds": _status_int(skip_decision.get("stale_after_seconds")),
            "stale_age_seconds": skip_decision.get("stale_age_seconds"),
            "latest_result_ts": skip_decision.get("latest_result_ts"),
            "newest_real_activity_ts": skip_decision.get("newest_real_activity_ts"),
            "is_stale": bool(skip_decision.get("is_stale")),
            "scan_output_progress": bool(skip_decision.get("scan_output_progress")),
            "progress_scan_error": skip_decision.get("progress_scan_error"),
            "progress_scan_error_blocks_skip": bool(
                skip_decision.get("progress_scan_error_blocks_skip")
            ),
            "different_spec": bool(skip_decision.get("different_spec")),
            "different_spec_error": skip_decision.get("different_spec_error"),
        }
    if ratings_written:
        completed_pair_count = pair_count
        completed_game_count = game_count
    summary = {
        "round_id": round_id,
        "round_index": int(round_index),
        "exists": bool(input_exists or progress_exists or ratings_written),
        "input_exists": input_exists,
        "progress_exists": progress_exists,
        "ratings_written": ratings_written,
        "status": status,
        "phase": phase,
        "checkpoint_count": checkpoint_count,
        "rating_spec_checkpoint_count": rating_spec_checkpoint_count,
        "checkpoint_roster_count": checkpoint_roster_count,
        "pair_count": pair_count,
        "game_count": game_count,
        "started_pair_count": _status_int(progress.get("started_pair_count")),
        "completed_pair_count": completed_pair_count,
        "completed_game_count": completed_game_count,
        "completion_fraction": (
            float(completed_game_count) / float(game_count) if game_count else None
        ),
        "updated_at": progress.get("updated_at") or input_payload.get("started_at"),
        "input_ref": input_ref.as_posix() if input_exists else "",
        "progress_ref": progress_ref.as_posix() if progress_exists else "",
        "ratings_ref": ratings_ref.as_posix() if ratings_written else "",
        "config": {
            "pair_selection": rating_spec.get("pair_selection"),
            "pairs_per_round": rating_spec.get("pairs_per_round"),
            "active_pool_limit": rating_spec.get("active_pool_limit"),
            "games_per_pair": rating_spec.get("games_per_pair"),
            "save_gif": rating_spec.get("save_gif"),
            "max_steps": rating_spec.get("max_steps"),
            "seat_order_mode": rating_spec.get("seat_order_mode"),
        },
    }
    if str(status) == "skipped":
        summary["skip_reason"] = progress.get("skip_reason") or (
            skip_summary.get("reason") if isinstance(skip_summary, Mapping) else ""
        )
        if skip_summary is not None:
            summary["skip_decision"] = skip_summary
    return summary


def _feedback_loop_status_from_state(
    *,
    tournament_id: str,
    rating_run_id: str,
    manifest: Mapping[str, Any] | None,
    manifest_load: Mapping[str, Any],
    active_manifest_keys: Sequence[str],
    queue_len: int | None,
    latest_snapshot: Mapping[str, Any] | None,
    batch_window: Sequence[Mapping[str, Any]],
    trainer_refresh_state: Mapping[str, Any] | None,
    current_batch_recovery_probe: Mapping[str, Any] | None = None,
    current_batch_progress_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_summary = {
        "exists": isinstance(manifest, Mapping),
        "source": manifest_load.get("manifest_source"),
        "state_repaired": bool(manifest_load.get("manifest_state_repaired", False)),
        "key": manifest_load.get("manifest_key", ""),
        "active": bool(manifest.get("active", False)) if isinstance(manifest, Mapping) else False,
        "checkpoint_count": (
            _status_int(manifest.get("checkpoint_count")) if isinstance(manifest, Mapping) else 0
        ),
        "queued_checkpoint_count": (
            _status_int(manifest.get("queued_checkpoint_count"))
            if isinstance(manifest, Mapping)
            else 0
        ),
        "queue_partition": str(manifest.get("queue_partition") or "") if isinstance(manifest, Mapping) else "",
    }
    rating_defaults = manifest.get("rating_defaults") if isinstance(manifest, Mapping) else {}
    if not isinstance(rating_defaults, Mapping):
        rating_defaults = {}
    manifest_summary["rating_defaults"] = {
        "pair_selection": rating_defaults.get("pair_selection"),
        "pairs_per_round": rating_defaults.get("pairs_per_round"),
        "active_pool_limit": rating_defaults.get("active_pool_limit"),
        "games_per_pair": rating_defaults.get("games_per_pair"),
        "save_gif": rating_defaults.get("save_gif"),
        "continue_from_latest": bool(rating_defaults.get("continue_from_latest", False)),
    }
    latest_summary = _rating_snapshot_status_summary(latest_snapshot)
    active_batches = [
        dict(batch)
        for batch in batch_window
        if batch.get("exists")
        and not batch.get("ratings_written")
        and str(batch.get("status") or "") != "skipped"
    ]
    active_batch_count = len(active_batches)
    current_batch = active_batches[-1] if active_batches else {}
    recovery_probe = (
        dict(current_batch_recovery_probe)
        if isinstance(current_batch_recovery_probe, Mapping)
        else {}
    )
    if current_batch and recovery_probe:
        current_batch["recovery_probe"] = {
            "scan_output_progress": bool(recovery_probe.get("scan_output_progress")),
            "skip": bool(recovery_probe.get("skip")),
            "reason": recovery_probe.get("reason"),
            "completed_game_count": _status_int(recovery_probe.get("completed_game_count")),
            "sampled_pair_count": _status_int(recovery_probe.get("sampled_pair_count")),
            "seen_pair_count": _status_int(recovery_probe.get("seen_pair_count")),
            "stopped_after_first_output": bool(
                recovery_probe.get("stopped_after_first_output")
            ),
            "has_output": bool(recovery_probe.get("has_output")),
            "started_pair_count": _status_int(recovery_probe.get("started_pair_count")),
            "latest_result_ts": recovery_probe.get("latest_result_ts"),
            "latest_result_age_seconds": recovery_probe.get("latest_result_age_seconds"),
            "progress_scan_error": recovery_probe.get("progress_scan_error"),
            "progress_scan_error_blocks_skip": bool(
                recovery_probe.get("progress_scan_error_blocks_skip")
            ),
            "stale_after_seconds": recovery_probe.get("stale_after_seconds"),
            "stale_age_seconds": recovery_probe.get("stale_age_seconds"),
            "is_stale": bool(recovery_probe.get("is_stale")),
            "count_semantics": (
                "liveness_sample_not_total"
                if bool(recovery_probe.get("stopped_after_first_output"))
                else "bounded_sample_not_total"
            ),
        }
    progress_probe = (
        dict(current_batch_progress_probe)
        if isinstance(current_batch_progress_probe, Mapping)
        else {}
    )
    if current_batch and progress_probe:
        latest_result_ts = progress_probe.get("latest_result_ts")
        latest_result_age_seconds = None
        try:
            if latest_result_ts is not None:
                latest_result_age_seconds = max(0.0, time.time() - float(latest_result_ts))
        except (TypeError, ValueError):
            latest_result_age_seconds = None
        current_batch["progress_probe"] = {
            "scan_output_progress": bool(progress_probe.get("scan_output_progress", True)),
            "count_basis": progress_probe.get("count_basis"),
            "counted_game_summary_files": bool(
                progress_probe.get("counted_game_summary_files")
            ),
            "pair_count": _status_int(progress_probe.get("pair_count")),
            "game_count": _status_int(progress_probe.get("game_count")),
            "completed_game_count": _status_int(progress_probe.get("completed_game_count")),
            "estimated_seen_game_count": _status_int(
                progress_probe.get("estimated_seen_game_count")
            ),
            "started_pair_count": _status_int(progress_probe.get("started_pair_count")),
            "partial_pair_count": _status_int(progress_probe.get("partial_pair_count")),
            "completed_pair_count": _status_int(progress_probe.get("completed_pair_count")),
            "completion_fraction": progress_probe.get("completion_fraction"),
            "estimated_completion_fraction": progress_probe.get(
                "estimated_completion_fraction"
            ),
            "max_started_pair_index": progress_probe.get("max_started_pair_index"),
            "max_completed_pair_index": progress_probe.get("max_completed_pair_index"),
            "latest_result_ts": latest_result_ts,
            "latest_result_age_seconds": latest_result_age_seconds,
            "scan_error_count": len(progress_probe.get("scan_errors") or []),
            "error_type": progress_probe.get("error_type"),
            "error": progress_probe.get("error"),
            "count_semantics": (
                "exhaustive_summary_file_count"
                if progress_probe.get("counted_game_summary_files")
                else "summary_file_count"
            ),
        }
    desired_new = max(
        0,
        int(manifest_summary["checkpoint_count"]) - int(latest_summary["checkpoint_count"]),
    )
    active_batch_checkpoint_count = (
        _status_int(current_batch.get("checkpoint_count")) if current_batch else 0
    )
    pool_status = {
        "intake_checkpoint_count": int(manifest_summary["checkpoint_count"]),
        "latest_rating_checkpoint_count": int(latest_summary["checkpoint_count"]),
        "active_game_batch_checkpoint_count": active_batch_checkpoint_count,
        "new_checkpoints_not_in_latest_rating": desired_new,
        "active_batch_newer_than_latest": bool(
            current_batch
            and active_batch_checkpoint_count > int(latest_summary["checkpoint_count"])
        ),
        "active_batch_not_covering_new_checkpoints": bool(
            current_batch
            and desired_new > 0
            and int(latest_summary["checkpoint_count"]) > 0
            and active_batch_checkpoint_count <= int(latest_summary["checkpoint_count"])
        ),
        "active_batch_missing_from_intake_count": (
            max(0, int(manifest_summary["checkpoint_count"]) - active_batch_checkpoint_count)
            if current_batch
            else None
        ),
    }
    if current_batch:
        current_batch["pool_status"] = pool_status
    flags: list[str] = []
    current_batch_age_seconds = None
    if current_batch:
        current_batch_updated_at = _parse_intake_timestamp(current_batch.get("updated_at"))
        if current_batch_updated_at is not None:
            current_batch_age_seconds = max(
                0.0,
                (datetime.now(UTC) - current_batch_updated_at).total_seconds(),
            )
    if len(active_manifest_keys) != 1:
        flags.append("active_manifest_key_count_not_one")
    if desired_new:
        flags.append("latest_rating_behind_intake")
    if queue_len:
        flags.append("checkpoint_queue_nonempty")
    if current_batch:
        flags.append("rating_game_batch_active")
        if active_batch_count > 1:
            flags.append("multiple_active_game_batches")
        if (
            _status_int(current_batch.get("started_pair_count")) == 0
            and _status_int(current_batch.get("completed_game_count")) == 0
        ):
            flags.append("active_game_batch_zero_started")
        if pool_status["active_batch_not_covering_new_checkpoints"]:
            flags.append("active_game_batch_not_covering_new_checkpoints")
        if recovery_probe.get("progress_scan_error"):
            flags.append("active_game_batch_output_scan_error")
        progress_probe_has_output = (
            _status_int(progress_probe.get("completed_game_count")) > 0
            or progress_probe.get("latest_result_ts") not in (None, "")
        )
        liveness_probe_has_output = (
            _status_int(recovery_probe.get("completed_game_count")) > 0
            or recovery_probe.get("latest_result_ts") not in (None, "")
        )
        if progress_probe_has_output or liveness_probe_has_output:
            flags.append("active_game_batch_has_game_output")
            if current_batch_age_seconds is not None:
                current_batch["age_seconds"] = current_batch_age_seconds
            progress_age = (
                current_batch.get("progress_probe", {}).get("latest_result_age_seconds")
                if isinstance(current_batch.get("progress_probe"), Mapping)
                else None
            )
            progress_is_stale = (
                isinstance(progress_age, (int, float))
                and progress_age >= DEFAULT_RATING_ROUND_STALE_SECONDS
            )
            if progress_is_stale or bool(recovery_probe.get("is_stale")):
                flags.append("active_game_batch_output_stale")
            if (
                current_batch_age_seconds is not None
                and current_batch_age_seconds
                >= DEFAULT_RATING_ROUND_PARTIAL_REDUCE_AFTER_SECONDS
                and _status_int(current_batch.get("completed_game_count")) < _status_int(
                    current_batch.get("game_count")
                )
            ):
                current_batch["partial_reduce_due"] = True
                current_batch["partial_reduce_after_seconds"] = (
                    DEFAULT_RATING_ROUND_PARTIAL_REDUCE_AFTER_SECONDS
                )
                flags.append("active_game_batch_partial_reduce_due")
    if (
        manifest_summary["rating_defaults"].get("pair_selection")
        == arena.RATING_PAIR_SELECTION_ALL_PAIRS
        and int(manifest_summary["checkpoint_count"]) > int(arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT)
    ):
        flags.append("bad_live_all_pairs_config")
    if current_batch:
        overall_status = "rating_game_batch_active"
    elif desired_new and queue_len:
        overall_status = "queued_waiting_for_drain"
    elif desired_new:
        overall_status = "ready_for_next_rating_batch"
    else:
        overall_status = "caught_up"
    refresh = dict(trainer_refresh_state) if isinstance(trainer_refresh_state, Mapping) else {}
    rewritten_pointers = refresh.get("rewritten_pointers")
    rewritten_assignment_sha256s: list[str] = []
    if isinstance(rewritten_pointers, Sequence) and not isinstance(
        rewritten_pointers, (str, bytes)
    ):
        for pointer in rewritten_pointers:
            if not isinstance(pointer, Mapping):
                continue
            assignment_sha256 = str(pointer.get("assignment_sha256") or "").strip()
            if assignment_sha256:
                rewritten_assignment_sha256s.append(assignment_sha256)
    refresh_summary = {
        "exists": bool(refresh),
        "status": refresh.get("status", ""),
        "generation": refresh.get("generation"),
        "snapshot_id": refresh.get("snapshot_id"),
        "row_count": refresh.get("row_count"),
        "active_count": refresh.get("active_count"),
        "rewritten_pointer_count": refresh.get("rewritten_pointer_count"),
        "rating_source": refresh.get("rating_source"),
        "created_at": refresh.get("created_at"),
        "assignment_sha256s": rewritten_assignment_sha256s,
        "assignment_sha256_prefixes": [
            sha[:8] for sha in rewritten_assignment_sha256s
        ],
    }
    return arena._to_plain(
        {
            "schema_id": "curvyzero_curvytron_feedback_loop_status/v0",
            "status": overall_status,
            "flags": flags,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "active_manifest_keys": list(active_manifest_keys),
            "intake": {
                **manifest_summary,
                "queue_len": queue_len,
                "new_checkpoints_not_in_latest_rating": desired_new,
            },
            "latest_rating": latest_summary,
            "pool_status": pool_status,
            "current_game_batch": current_batch or None,
            "active_game_batch_count": active_batch_count,
            "recent_game_batches": [dict(batch) for batch in batch_window if batch.get("exists")],
            "trainer_refresh": refresh_summary,
            "operator_next_action": _feedback_loop_next_action(
                overall_status=overall_status,
                flags=flags,
                current_batch=current_batch,
                desired_new_checkpoint_count=desired_new,
            ),
            "checked_at": runs.utc_timestamp(),
        }
    )


def _feedback_loop_next_action(
    *,
    overall_status: str,
    flags: Sequence[str],
    current_batch: Mapping[str, Any],
    desired_new_checkpoint_count: int,
) -> str:
    if "bad_live_all_pairs_config" in flags:
        return "repair live manifest before spawning more rating work"
    if "active_game_batch_output_scan_error" in flags:
        return "fix the game-output scan before skipping this active game batch"
    if "active_game_batch_not_covering_new_checkpoints" in flags:
        return (
            "active game batch is running but appears to cover only the already-rated "
            "pool; do not call catch-up validated, let it finish/recover, then start "
            "a full-pool drain with spec-count proof"
        )
    if "active_game_batch_output_stale" in flags:
        return "game output exists but is stale; let drain recovery scan the full output set before deciding whether to skip"
    if "active_game_batch_partial_reduce_due" in flags:
        return "active game batch is old enough to partially reduce; spawn recovery so completed games can publish and the loop can continue"
    if "active_game_batch_has_game_output" in flags:
        if "multiple_active_game_batches" in flags:
            return "game outputs are landing, but multiple active game-batch artifacts exist; do not spawn more, wait for completion/reduction, then repair stale leftovers"
        return "game outputs are landing for the active batch; wait for completion or reduce"
    if current_batch and "active_game_batch_zero_started" in flags:
        return "no game output seen in status yet; let stale recovery scan output before any skip"
    if overall_status == "rating_game_batch_active":
        return "wait for current bounded game batch to finish, then refresh trainer candidates"
    if overall_status == "queued_waiting_for_drain":
        return (
            "let scheduled drain run, or use scripts/curvytron_live_loop_control.py "
            "--action drain-if-ready to start the next bounded game batch"
        )
    if overall_status == "ready_for_next_rating_batch" and desired_new_checkpoint_count:
        return (
            "use scripts/curvytron_live_loop_control.py --action drain-if-ready "
            "to rate the new checkpoint pool"
        )
    return "no manual action required"


def _feedback_loop_control_decision(
    status: Mapping[str, Any],
    *,
    action: str,
) -> dict[str, Any]:
    normalized_action = str(action or "status").strip().lower()
    if normalized_action not in {"status", "drain-if-ready", "drain"}:
        raise ValueError("loop_control_action must be status, drain-if-ready, or drain")
    flags = [str(flag) for flag in status.get("flags") or []]
    current_batch = status.get("current_game_batch")
    intake = status.get("intake") if isinstance(status.get("intake"), Mapping) else {}
    desired_new = _status_int(intake.get("new_checkpoints_not_in_latest_rating"))
    queue_len = intake.get("queue_len")
    if normalized_action == "status":
        return {
            "action": normalized_action,
            "spawn_drain": False,
            "reason": "status_only",
            "new_checkpoints_not_in_latest_rating": desired_new,
            "queue_len": queue_len,
        }
    if "bad_live_all_pairs_config" in flags:
        return {
            "action": normalized_action,
            "spawn_drain": False,
            "reason": "blocked_bad_live_all_pairs_config",
            "new_checkpoints_not_in_latest_rating": desired_new,
            "queue_len": queue_len,
        }
    if current_batch and (
        "active_game_batch_output_stale" in flags
        or "active_game_batch_not_covering_new_checkpoints" in flags
        or "active_game_batch_partial_reduce_due" in flags
    ):
        return {
            "action": normalized_action,
            "spawn_drain": True,
            "reason": "spawn_active_game_batch_recovery_scan",
            "status": status.get("status"),
            "new_checkpoints_not_in_latest_rating": desired_new,
            "queue_len": queue_len,
        }
    if current_batch:
        return {
            "action": normalized_action,
            "spawn_drain": False,
            "reason": "blocked_active_game_batch",
            "new_checkpoints_not_in_latest_rating": desired_new,
            "queue_len": queue_len,
        }
    if desired_new <= 0 and not queue_len:
        return {
            "action": normalized_action,
            "spawn_drain": False,
            "reason": "nothing_new_to_rate",
            "new_checkpoints_not_in_latest_rating": desired_new,
            "queue_len": queue_len,
        }
    if normalized_action == "drain-if-ready" and str(status.get("status") or "") not in {
        "queued_waiting_for_drain",
        "ready_for_next_rating_batch",
    }:
        return {
            "action": normalized_action,
            "spawn_drain": False,
            "reason": "status_not_ready_for_drain",
            "status": status.get("status"),
            "new_checkpoints_not_in_latest_rating": desired_new,
            "queue_len": queue_len,
        }
    return {
        "action": normalized_action,
        "spawn_drain": True,
        "reason": "spawn_next_bounded_rating_batch",
        "new_checkpoints_not_in_latest_rating": desired_new,
        "queue_len": queue_len,
    }


def _feedback_loop_status_payload(
    *,
    tournament_id: str,
    rating_run_id: str,
    leaderboard_id: str,
    lookahead_batches: int = 64,
    status_activity_probe_pairs: int = 4,
    status_progress_probe: bool = False,
) -> dict[str, Any]:
    manifest, manifest_load = _load_intake_manifest(tournament_id, rating_run_id)
    active_keys = checkpoint_intake_state.get(CHECKPOINT_INTAKE_ACTIVE_KEYS, []) or []
    if not isinstance(active_keys, list):
        active_keys = []
    queue_len = None
    if isinstance(manifest, Mapping) and manifest.get("queue_partition"):
        queue_len = checkpoint_intake_queue.len(partition=str(manifest["queue_partition"]))
    latest_snapshot = _read_rating_snapshot_for_run(
        TOURNAMENT_MOUNT,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    latest_index = _status_int(latest_snapshot.get("round_index"), -1)
    start_index = max(0, latest_index - 1)
    end_index = latest_index + max(0, int(lookahead_batches))
    window_indices = set(range(start_index, max(start_index, end_index) + 1))
    window_indices.update(
        _unrated_rating_round_indices(
            TOURNAMENT_MOUNT,
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
            after_round_index=latest_index,
        )
    )
    batch_window = [
        _rating_game_batch_status_summary(
            TOURNAMENT_MOUNT,
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
            round_index=index,
        )
        for index in sorted(window_indices)
    ]
    active_batches = [
        dict(batch)
        for batch in batch_window
        if batch.get("exists")
        and not batch.get("ratings_written")
        and str(batch.get("status") or "") != "skipped"
    ]
    current_batch = active_batches[-1] if active_batches else {}
    current_batch_recovery_probe = None
    current_batch_progress_probe = None
    if current_batch and int(status_activity_probe_pairs) > 0:
        current_batch_recovery_probe = _rating_round_activity_probe(
            TOURNAMENT_MOUNT,
            tournament_id=tournament_id,
            rating_run_id=rating_run_id,
            round_id=str(current_batch.get("round_id") or ""),
            max_pairs=int(status_activity_probe_pairs),
        )
    if current_batch and bool(status_progress_probe):
        try:
            current_batch_progress_probe, _games_by_battle = _rating_round_progress_payload(
                TOURNAMENT_MOUNT,
                tournament_id=tournament_id,
                rating_run_id=rating_run_id,
                round_id=str(current_batch.get("round_id") or ""),
                load_summaries=False,
                pair_only=True,
                count_game_summaries=True,
            )
        except Exception as exc:
            current_batch_progress_probe = {
                "schema_id": "curvyzero_curvytron_rating_round_progress_probe_error/v0",
                "scan_output_progress": True,
                "round_id": str(current_batch.get("round_id") or ""),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    trainer_refresh_state = checkpoint_intake_state.get(
        _training_candidate_auto_refresh_state_key(
            tournament_id,
            rating_run_id,
            leaderboard_id,
        ),
        {},
    )
    return _feedback_loop_status_from_state(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        manifest=manifest,
        manifest_load=manifest_load,
        active_manifest_keys=[str(key) for key in active_keys],
        queue_len=queue_len,
        latest_snapshot=latest_snapshot,
        batch_window=batch_window,
        trainer_refresh_state=(
            trainer_refresh_state if isinstance(trainer_refresh_state, Mapping) else {}
        ),
        current_batch_recovery_probe=current_batch_recovery_probe,
        current_batch_progress_probe=current_batch_progress_probe,
    )


@app.function(
    image=image,
    volumes=_controller_volumes(),
    timeout=30 * 60,
    cpu=1.0,
    memory=1024,
)
def curvytron_feedback_loop_status(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(spec or {})
    _reload_volume(tournament_volume)
    _reload_volume(control_volume)
    if str(payload.get("mode") or "") == "reduce_rescue":
        return _curvytron_rating_reduce_body(payload, reducer="status_function_rescue")
    tournament_id = runs.clean_id(
        str(payload.get("tournament_id") or CURRENT_TOURNAMENT_ID),
        label="tournament_id",
    )
    rating_run_id = runs.clean_id(
        str(payload.get("rating_run_id") or CURRENT_RATING_RUN_ID),
        label="rating_run_id",
    )
    leaderboard_id = runs.clean_id(
        str(payload.get("leaderboard_id") or f"{tournament_id}-{rating_run_id}-training"),
        label="leaderboard_id",
    )
    lookahead_raw = payload.get("lookahead_batches")
    lookahead_batches = int(lookahead_raw) if lookahead_raw not in (None, "") else 64
    probe_pairs_raw = payload.get("status_activity_probe_pairs")
    status_activity_probe_pairs = (
        int(probe_pairs_raw) if probe_pairs_raw not in (None, "") else 4
    )
    progress_probe_raw = payload.get("status_progress_probe")
    status_progress_probe = (
        bool(progress_probe_raw) if progress_probe_raw not in (None, "") else False
    )
    return _feedback_loop_status_payload(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        leaderboard_id=leaderboard_id,
        lookahead_batches=lookahead_batches,
        status_activity_probe_pairs=status_activity_probe_pairs,
        status_progress_probe=status_progress_probe,
    )


@app.local_entrypoint()
def main(
    mode: str = "pair",
    tournament_id: str = "",
    checkpoint_refs: str = "",
    run_ids: str = "",
    run_id_prefix: str = "",
    max_runs: int = 0,
    expected_checkpoint_count: int = 0,
    allow_missing_checkpoints: bool = False,
    checkpoint_iteration: int = -1,
    checkpoint_selection: str = arena.CHECKPOINT_SELECTION_LATEST,
    games_per_pair: int = arena.DEFAULT_GAMES_PER_PAIR,
    games_per_shard: int = arena.DEFAULT_GAMES_PER_SHARD,
    reuse_policies_per_shard: bool = arena.DEFAULT_REUSE_POLICIES_PER_SHARD,
    rating_run_id: str = arena.DEFAULT_RATING_RUN_ID,
    round_count: int = arena.DEFAULT_RATING_ROUND_COUNT,
    continue_from_latest: bool = False,
    pairs_per_round: int = 0,
    placement_min_games: int = 0,
    placement_min_opponents: int = 20,
    pair_selection: str = arena.DEFAULT_RATING_PAIR_SELECTION,
    initial_rating: float = arena.DEFAULT_RATING_INITIAL_RATING,
    active_pool_limit: int = arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT,
    stop_when_stable: bool = False,
    seed: int = 0,
    max_steps: int = arena.DEFAULT_MAX_STEPS,
    decision_ms: float = arena.DEFAULT_DECISION_MS,
    decision_source_frames: int = arena.DEFAULT_DECISION_SOURCE_FRAMES,
    source_physics_step_ms: float = arena.DEFAULT_SOURCE_PHYSICS_STEP_MS,
    policy_mode: str = arena.POLICY_MODE_EVAL,
    collect_temperature: float = arena.DEFAULT_COLLECT_TEMPERATURE,
    collect_epsilon: float = arena.DEFAULT_COLLECT_EPSILON,
    policy_trail_render_mode: str = arena.DEFAULT_POLICY_TRAIL_RENDER_MODE,
    policy_bonus_render_mode: str = arena.DEFAULT_POLICY_BONUS_RENDER_MODE,
    num_simulations: int = arena.DEFAULT_NUM_SIMULATIONS,
    save_gif: bool = arena.DEFAULT_SAVE_GIF,
    gif_sample_games_per_pair: int = arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR,
    gif_sample_strategy: str = arena.DEFAULT_GIF_SAMPLE_STRATEGY,
    round_index: int = 0,
    allow_partial_reduce: bool = False,
    progress_read_summaries: bool = False,
    visibility_action: str = "list",
    visibility_tournament_ids: str = "",
    visibility_keep_tournament_ids: str = "",
    visibility_dry_run: bool = True,
    leaderboard_id: str = "",
    leaderboard_snapshot_id: str = "",
    leaderboard_allow_live_provisional: bool = False,
    leaderboard_diagnostic_only: bool = False,
    leaderboard_allow_no_active_rows: bool = False,
    leaderboard_allow_legacy_rating_snapshot: bool = False,
    leaderboard_active_min_distinct_opponents: int = 20,
    leaderboard_active_min_valid_games: int = 300,
    leaderboard_max_failure_rate: float = 0.02,
    leaderboard_max_active_rank: int = arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT,
    leaderboard_expected_round_id: str = "",
    leaderboard_expected_round_index: int = -1,
    leaderboard_expected_rating_context_hash: str = "",
    leaderboard_expected_roster_hash: str = "",
    leaderboard_expected_rating_snapshot_sha256: str = "",
    training_candidate_refresh_pointers: str = "",
    training_candidate_refresh_config_ref: str = "",
    training_candidate_assignment_bank_run_id: str = "",
    training_candidate_assignment_bank_attempt_id: str = "",
    training_candidate_assignment_id_prefix: str = "",
    training_candidate_assignment_seed: int = 0,
    training_candidate_generation: int = -1,
    training_candidate_min_active_count: int = 1,
    training_candidate_allow_partial_assignment: bool = False,
    intake_enqueue_existing: bool = False,
    intake_spawn_rating: bool = False,
    intake_spawn_if_existing: bool = False,
    intake_allow_rating_overrides: bool = False,
    intake_max_events: int = 100,
    intake_claim_stale_after_seconds: int = DEFAULT_CHECKPOINT_INTAKE_CLAIM_STALE_SECONDS,
    intake_rating_round_stale_after_seconds: int = DEFAULT_RATING_ROUND_STALE_SECONDS,
    status_lookahead_batches: int = 64,
    status_activity_probe_pairs: int = 4,
    loop_control_action: str = "status",
    intake_active: bool = True,
    wait: bool = False,
) -> None:
    mode = str(mode)
    _assert_safe_local_entrypoint_mode(mode)
    resolved_tournament_id = _cli_tournament_id_for_mode(
        mode=mode,
        tournament_id=tournament_id,
    )
    resolved_rating_run_id = _cli_rating_run_id_for_mode(
        mode=mode,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
    )
    refs = arena.parse_checkpoint_refs(checkpoint_refs)
    discovery = None
    intake_modes = {
        "intake-seed",
        "intake-submit",
        "tournament-submit",
        "intake-tick",
        "intake-drain",
        "intake-status",
        "leaderboard-pointer-repair",
        "training-candidate-refresh",
        "training-candidate-auto-refresh",
        "current",
        "status-current",
        "loop-status",
        "loop-control",
    }
    if not refs and mode not in intake_modes and (run_ids or run_id_prefix):
        discovery = curvytron_discover_checkpoints.remote(
            {
                "run_ids": run_ids,
                "run_id_prefix": run_id_prefix,
                "max_runs": int(max_runs),
                "checkpoint_iteration": int(checkpoint_iteration),
                "checkpoint_selection": checkpoint_selection,
            }
        )
        refs = [str(ref) for ref in discovery.get("checkpoint_refs", [])]
    if mode not in {
        "discover",
        "estimate",
        "game",
        "pair",
        "tournament",
        "rating",
        "progress",
        "provisional",
        "provisional-loop",
        "reduce",
        "visibility",
        "leaderboard-publish",
        "leaderboard-pointer-repair",
        "training-candidate-refresh",
        "training-candidate-auto-refresh",
        "current",
        "status-current",
        "loop-status",
        "loop-control",
        "intake-seed",
        "intake-submit",
        "tournament-submit",
        "intake-tick",
        "intake-drain",
        "intake-status",
    }:
        raise ValueError(
            "mode must be one of: discover, estimate, game, pair, tournament, rating, "
            "progress, provisional, provisional-loop, reduce, visibility, "
            "leaderboard-publish, leaderboard-pointer-repair, "
            "training-candidate-refresh, training-candidate-auto-refresh, "
            "current, status-current, loop-status, loop-control, intake-seed, "
            "intake-submit, tournament-submit, intake-tick, intake-drain, intake-status"
        )
    if mode in {"current", "status-current"}:
        print(json.dumps(_current_lane_static_status(), indent=2, sort_keys=True))
        return
    if mode == "loop-status":
        result = curvytron_feedback_loop_status.remote(
            {
                "tournament_id": resolved_tournament_id,
                "rating_run_id": resolved_rating_run_id,
                "leaderboard_id": leaderboard_id
                or f"{resolved_tournament_id}-{resolved_rating_run_id}-training",
                "lookahead_batches": int(status_lookahead_batches),
                "status_activity_probe_pairs": int(status_activity_probe_pairs),
            }
        )
        print(json.dumps(arena._to_plain(result), indent=2, sort_keys=True))
        return
    if mode == "loop-control":
        status = curvytron_feedback_loop_status.remote(
            {
                "tournament_id": resolved_tournament_id,
                "rating_run_id": resolved_rating_run_id,
                "leaderboard_id": leaderboard_id
                or f"{resolved_tournament_id}-{resolved_rating_run_id}-training",
                "lookahead_batches": int(status_lookahead_batches),
                "status_activity_probe_pairs": int(status_activity_probe_pairs),
            }
        )
        decision = _feedback_loop_control_decision(status, action=loop_control_action)
        result: dict[str, Any] = {
            "schema_id": "curvyzero_curvytron_feedback_loop_control/v0",
            "mode": "loop-control",
            "tournament_id": resolved_tournament_id,
            "rating_run_id": resolved_rating_run_id,
            "decision": decision,
            "status": status,
        }
        if bool(decision.get("spawn_drain")):
            drain_spec = {
                "tournament_id": resolved_tournament_id,
                "rating_run_id": resolved_rating_run_id,
                "spawn_rating": True,
                "spawn_if_existing": True,
                "max_events": int(intake_max_events),
                "claim_stale_after_seconds": int(intake_claim_stale_after_seconds),
                "rating_round_stale_after_seconds": int(intake_rating_round_stale_after_seconds),
                "wait_for_rating": False,
            }
            call = curvytron_checkpoint_intake_drain.spawn(drain_spec)
            result["drain_spawn"] = {
                "status": "spawned",
                "function_call_id": getattr(call, "object_id", None)
                or getattr(call, "id", None)
                or "",
                "spawn_if_existing": True,
                "max_events": int(intake_max_events),
            }
        print(json.dumps(arena._to_plain(result), indent=2, sort_keys=True))
        return
    if mode == "training-candidate-auto-refresh":
        refresh_pointer_refs = [
            item.strip()
            for item in str(training_candidate_refresh_pointers).split(",")
            if item.strip()
        ]
        result = curvytron_training_candidate_refresh_tick.remote(
            _training_candidate_auto_refresh_cli_payload(
                tournament_id=resolved_tournament_id,
                rating_run_id=resolved_rating_run_id,
                leaderboard_id=leaderboard_id
                or f"{resolved_tournament_id}-{resolved_rating_run_id}-training",
                config_ref=training_candidate_refresh_config_ref,
                refresh_pointer_refs=refresh_pointer_refs,
                assignment_bank_run_id=training_candidate_assignment_bank_run_id,
                assignment_bank_attempt_id=training_candidate_assignment_bank_attempt_id,
                assignment_id_prefix=training_candidate_assignment_id_prefix,
                assignment_seed=int(training_candidate_assignment_seed),
                generation=int(training_candidate_generation),
                min_active_count=int(training_candidate_min_active_count),
                allow_partial_assignment=bool(training_candidate_allow_partial_assignment),
                active_min_valid_games=int(leaderboard_active_min_valid_games),
                active_min_distinct_opponents=int(leaderboard_active_min_distinct_opponents),
                max_active_rank=int(leaderboard_max_active_rank),
            )
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if mode == "training-candidate-refresh":
        refresh_pointer_refs = [
            item.strip()
            for item in str(training_candidate_refresh_pointers).split(",")
            if item.strip()
        ]
        result = curvytron_training_candidate_refresh.remote(
            {
                "tournament_id": resolved_tournament_id,
                "rating_run_id": resolved_rating_run_id,
                "leaderboard_id": leaderboard_id
                or f"{resolved_tournament_id}-{resolved_rating_run_id}-training",
                "snapshot_id": leaderboard_snapshot_id,
                "allow_live_provisional": bool(leaderboard_allow_live_provisional),
                "active_min_distinct_opponents": int(leaderboard_active_min_distinct_opponents),
                "active_min_valid_games": int(leaderboard_active_min_valid_games),
                "max_failure_rate": float(leaderboard_max_failure_rate),
                "max_active_rank": int(leaderboard_max_active_rank),
                "expected_round_id": leaderboard_expected_round_id or None,
                "expected_round_index": (
                    int(leaderboard_expected_round_index)
                    if int(leaderboard_expected_round_index) >= 0
                    else None
                ),
                "expected_rating_context_hash": (leaderboard_expected_rating_context_hash or None),
                "expected_roster_hash": leaderboard_expected_roster_hash or None,
                "expected_rating_snapshot_sha256": (
                    leaderboard_expected_rating_snapshot_sha256 or None
                ),
                "refresh_pointers": refresh_pointer_refs,
                "assignment_bank_run_id": training_candidate_assignment_bank_run_id,
                "assignment_bank_attempt_id": training_candidate_assignment_bank_attempt_id,
                "assignment_id_prefix": training_candidate_assignment_id_prefix,
                "assignment_seed": int(training_candidate_assignment_seed),
                "generation": (
                    int(training_candidate_generation)
                    if int(training_candidate_generation) >= 0
                    else None
                ),
                "min_active_count": int(training_candidate_min_active_count),
                "allow_partial_assignment": bool(training_candidate_allow_partial_assignment),
            }
        )
        print(json.dumps(arena._to_plain(result), indent=2, sort_keys=True))
        return
    if mode == "leaderboard-publish":
        result = curvytron_opponent_leaderboard_publish.remote(
            {
                "tournament_id": resolved_tournament_id,
                "rating_run_id": resolved_rating_run_id,
                "leaderboard_id": leaderboard_id
                or f"{resolved_tournament_id}-{resolved_rating_run_id}",
                "snapshot_id": leaderboard_snapshot_id,
                "allow_live_provisional": bool(leaderboard_allow_live_provisional),
                "diagnostic_only": bool(leaderboard_diagnostic_only),
                "allow_no_active_rows": bool(leaderboard_allow_no_active_rows),
                "allow_legacy_rating_snapshot": bool(leaderboard_allow_legacy_rating_snapshot),
                "active_min_distinct_opponents": int(leaderboard_active_min_distinct_opponents),
                "active_min_valid_games": int(leaderboard_active_min_valid_games),
                "max_failure_rate": float(leaderboard_max_failure_rate),
                "max_active_rank": int(leaderboard_max_active_rank),
                "expected_round_id": leaderboard_expected_round_id or None,
                "expected_round_index": (
                    int(leaderboard_expected_round_index)
                    if int(leaderboard_expected_round_index) >= 0
                    else None
                ),
                "expected_rating_context_hash": (leaderboard_expected_rating_context_hash or None),
                "expected_roster_hash": leaderboard_expected_roster_hash or None,
                "expected_rating_snapshot_sha256": (
                    leaderboard_expected_rating_snapshot_sha256 or None
                ),
            }
        )
        print(json.dumps(arena._to_plain(result), indent=2, sort_keys=True))
        return
    if mode == "leaderboard-pointer-repair":
        result = curvytron_opponent_leaderboard_pointer_repair.remote(
            {
                "leaderboard_id": leaderboard_id
                or f"{resolved_tournament_id}-{resolved_rating_run_id}",
            }
        )
        print(json.dumps(arena._to_plain(result), indent=2, sort_keys=True))
        return
    if mode == "visibility":
        result = curvytron_tournament_visibility.remote(
            {
                "action": visibility_action,
                "tournament_ids": visibility_tournament_ids,
                "keep_tournament_ids": visibility_keep_tournament_ids,
                "dry_run": bool(visibility_dry_run),
            }
        )
        print(json.dumps(arena._to_plain(result), indent=2, sort_keys=True))
        return
    if mode == "discover":
        if discovery is None:
            discovery = curvytron_discover_checkpoints.remote(
                {
                    "run_ids": run_ids,
                    "run_id_prefix": run_id_prefix,
                    "max_runs": int(max_runs),
                    "checkpoint_iteration": int(checkpoint_iteration),
                    "checkpoint_selection": checkpoint_selection,
                }
            )
        print(json.dumps(arena._to_plain(discovery), indent=2, sort_keys=True))
        return
    if mode in {
        "intake-seed",
        "intake-submit",
        "tournament-submit",
        "intake-tick",
        "intake-drain",
        "intake-status",
    }:
        submit_spec = {
            "tournament_id": resolved_tournament_id,
            "rating_run_id": resolved_rating_run_id,
            "checkpoint_refs": refs,
            "run_ids": run_ids,
            "run_id_prefix": run_id_prefix,
        }
        if mode in {"intake-submit", "tournament-submit"}:
            _reject_submit_cli_scheduler_overrides(
                checkpoint_iteration=checkpoint_iteration,
                checkpoint_selection=checkpoint_selection,
                games_per_pair=games_per_pair,
                games_per_shard=games_per_shard,
                reuse_policies_per_shard=reuse_policies_per_shard,
                round_count=round_count,
                continue_from_latest=continue_from_latest,
                pairs_per_round=pairs_per_round,
                placement_min_games=placement_min_games,
                placement_min_opponents=placement_min_opponents,
                pair_selection=pair_selection,
                initial_rating=initial_rating,
                active_pool_limit=active_pool_limit,
                stop_when_stable=stop_when_stable,
                seed=seed,
                max_steps=max_steps,
                decision_ms=decision_ms,
                decision_source_frames=decision_source_frames,
                source_physics_step_ms=source_physics_step_ms,
                policy_mode=policy_mode,
                collect_temperature=collect_temperature,
                collect_epsilon=collect_epsilon,
                policy_trail_render_mode=policy_trail_render_mode,
                policy_bonus_render_mode=policy_bonus_render_mode,
                num_simulations=num_simulations,
                save_gif=save_gif,
                gif_sample_games_per_pair=gif_sample_games_per_pair,
                gif_sample_strategy=gif_sample_strategy,
                intake_enqueue_existing=intake_enqueue_existing,
                intake_spawn_rating=intake_spawn_rating,
                intake_spawn_if_existing=intake_spawn_if_existing,
                intake_allow_rating_overrides=intake_allow_rating_overrides,
                intake_max_events=intake_max_events,
                intake_claim_stale_after_seconds=intake_claim_stale_after_seconds,
                intake_rating_round_stale_after_seconds=intake_rating_round_stale_after_seconds,
                intake_active=intake_active,
                max_runs=max_runs,
            )
            result = curvytron_checkpoint_intake_submit.remote(submit_spec)
            print(json.dumps(arena._to_plain(result), indent=2, sort_keys=True))
            return
        intake_spec = {
            "tournament_id": resolved_tournament_id,
            "rating_run_id": resolved_rating_run_id,
            "checkpoint_refs": refs,
            "run_ids": run_ids,
            "run_id_prefix": run_id_prefix,
            "max_runs": int(max_runs),
            "checkpoint_iteration": int(checkpoint_iteration),
            "checkpoint_selection": checkpoint_selection,
            "round_count": int(round_count),
            "continue_from_latest": bool(continue_from_latest),
            "pairs_per_round": int(pairs_per_round) if int(pairs_per_round) > 0 else None,
            "placement_min_games": (
                int(placement_min_games) if int(placement_min_games) > 0 else None
            ),
            "placement_min_opponents": int(placement_min_opponents),
            "pair_selection": pair_selection,
            "initial_rating": float(initial_rating),
            "active_pool_limit": int(active_pool_limit),
            "stop_when_stable": bool(stop_when_stable),
            "games_per_pair": int(games_per_pair),
            "games_per_shard": int(games_per_shard),
            "reuse_policies_per_shard": bool(reuse_policies_per_shard),
            "seed": int(seed),
            "max_steps": int(max_steps),
            "decision_ms": float(decision_ms),
            "decision_source_frames": int(decision_source_frames),
            "source_physics_step_ms": float(source_physics_step_ms),
            "policy_mode": policy_mode,
            "collect_temperature": float(collect_temperature),
            "collect_epsilon": float(collect_epsilon),
            "policy_trail_render_mode": policy_trail_render_mode or None,
            "policy_bonus_render_mode": policy_bonus_render_mode or None,
            "num_simulations": int(num_simulations),
            "save_gif": bool(save_gif),
            "gif_sample_games_per_pair": int(gif_sample_games_per_pair),
            "gif_sample_strategy": str(gif_sample_strategy),
            "enqueue_existing": bool(intake_enqueue_existing),
            "spawn_rating": bool(intake_spawn_rating),
            "spawn_if_existing": bool(intake_spawn_if_existing),
            "allow_rating_overrides": bool(intake_allow_rating_overrides),
            "max_events": int(intake_max_events),
            "claim_stale_after_seconds": int(intake_claim_stale_after_seconds),
            "rating_round_stale_after_seconds": int(intake_rating_round_stale_after_seconds),
            "wait_for_rating": bool(wait),
            "active": bool(intake_active),
        }
        if mode == "intake-seed":
            result = curvytron_checkpoint_intake_seed.remote(intake_spec)
        elif mode == "intake-tick":
            result = curvytron_checkpoint_intake_tick.remote(
                intake_spec if mode != "intake-tick" or tournament_id else {}
            )
        elif mode == "intake-drain":
            if bool(intake_spawn_rating) and not bool(wait):
                call = curvytron_checkpoint_intake_drain.spawn(
                    _drop_default_intake_drain_rating_overrides(intake_spec)
                )
                print(
                    json.dumps(
                        {
                            "status": "spawned",
                            "mode": "intake-drain",
                            "function_call_id": getattr(call, "object_id", ""),
                            "tournament_id": resolved_tournament_id,
                            "rating_run_id": resolved_rating_run_id,
                            "spawn_rating": True,
                            "spawn_if_existing": bool(intake_spawn_if_existing),
                            "max_events": int(intake_max_events),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return
            result = curvytron_checkpoint_intake_drain.remote(
                _drop_default_intake_drain_rating_overrides(intake_spec)
            )
        else:
            result = curvytron_checkpoint_intake_status.remote(intake_spec)
        print(json.dumps(arena._to_plain(result), indent=2, sort_keys=True))
        return
    if mode in {"progress", "provisional", "provisional-loop", "reduce"}:
        progress_spec = {
            "tournament_id": resolved_tournament_id,
            "rating_run_id": resolved_rating_run_id,
            "round_index": int(round_index),
            "round_id": arena.rating_round_id(int(round_index)),
            "load_summaries": bool(progress_read_summaries),
        }
        if mode == "progress":
            if wait:
                result = curvytron_rating_progress.remote(progress_spec)
                print(json.dumps(arena._to_plain(result), indent=2, sort_keys=True))
                return
            call = curvytron_rating_progress.spawn(progress_spec)
            call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
            payload = {
                "status": "spawned",
                "app_name": APP_NAME,
                "mode": mode,
                "tournament_id": resolved_tournament_id,
                "rating_run_id": resolved_rating_run_id,
                "round_index": int(round_index),
                "function_call_id": call_id,
            }
            print(json.dumps(arena._to_plain(payload), indent=2, sort_keys=True))
            return
        if mode == "provisional":
            call = curvytron_rating_provisional.spawn(progress_spec)
            call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
            payload: dict[str, Any] = {
                "status": "spawned",
                "app_name": APP_NAME,
                "mode": mode,
                "tournament_id": resolved_tournament_id,
                "rating_run_id": resolved_rating_run_id,
                "round_index": int(round_index),
                "function_call_id": call_id,
            }
            if wait:
                payload["result"] = call.get()
            print(json.dumps(arena._to_plain(payload), indent=2, sort_keys=True))
            return
        if mode == "provisional-loop":
            call = curvytron_rating_provisional_loop.spawn(progress_spec)
            call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
            payload = {
                "status": "spawned",
                "app_name": APP_NAME,
                "mode": mode,
                "tournament_id": resolved_tournament_id,
                "rating_run_id": resolved_rating_run_id,
                "round_index": int(round_index),
                "function_call_id": call_id,
            }
            if wait:
                payload["result"] = call.get()
            print(json.dumps(arena._to_plain(payload), indent=2, sort_keys=True))
            return
        call = curvytron_rating_reduce.spawn(
            {
                **progress_spec,
                "allow_partial": bool(allow_partial_reduce),
            }
        )
        call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
        payload: dict[str, Any] = {
            "status": "spawned",
            "app_name": APP_NAME,
            "mode": mode,
            "tournament_id": resolved_tournament_id,
            "rating_run_id": resolved_rating_run_id,
            "round_index": int(round_index),
            "allow_partial_reduce": bool(allow_partial_reduce),
            "function_call_id": call_id,
        }
        if wait:
            payload["result"] = call.get()
        print(json.dumps(arena._to_plain(payload), indent=2, sort_keys=True))
        return
    _assert_checkpoint_count(
        refs=refs,
        discovery=discovery if isinstance(discovery, Mapping) else None,
        expected_checkpoint_count=int(expected_checkpoint_count),
        max_runs=int(max_runs),
        allow_missing_checkpoints=bool(allow_missing_checkpoints),
    )
    if mode in {"game", "pair"} and len(refs) != 2:
        raise ValueError("game/pair mode needs exactly two checkpoint refs")
    if mode == "tournament" and len(refs) < 2:
        raise ValueError("tournament mode needs at least two checkpoint refs")
    if mode == "rating" and len(refs) < 2 and not bool(continue_from_latest):
        raise ValueError("rating mode needs at least two checkpoint refs")

    estimate_games_per_pair = 1 if mode == "game" else int(games_per_pair)
    plan_estimate = arena.estimate_tournament_plan(
        checkpoint_count=len(refs),
        games_per_pair=estimate_games_per_pair,
        games_per_shard=int(games_per_shard),
        reuse_policies_per_shard=bool(reuse_policies_per_shard),
        pairs_per_round=int(pairs_per_round) if int(pairs_per_round) > 0 else None,
        save_gif=bool(save_gif),
        gif_sample_games_per_pair=int(gif_sample_games_per_pair),
        gif_sample_strategy=str(gif_sample_strategy),
    )
    if mode == "estimate":
        print(
            json.dumps(
                arena._to_plain(
                    {
                        "mode": mode,
                        "checkpoint_count": len(refs),
                        "checkpoint_discovery": discovery,
                        "plan_estimate": plan_estimate,
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    checkpoint_specs = _rating_checkpoint_specs_from_refs(
        refs,
        discovery=discovery if isinstance(discovery, Mapping) else None,
    )

    common = {
        "games_per_pair": int(games_per_pair),
        "games_per_shard": int(games_per_shard),
        "reuse_policies_per_shard": bool(reuse_policies_per_shard),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "decision_ms": float(decision_ms),
        "decision_source_frames": int(decision_source_frames),
        "source_physics_step_ms": float(source_physics_step_ms),
        "policy_mode": policy_mode,
        "collect_temperature": float(collect_temperature),
        "collect_epsilon": float(collect_epsilon),
        "policy_trail_render_mode": policy_trail_render_mode or None,
        "policy_bonus_render_mode": policy_bonus_render_mode or None,
        "num_simulations": int(num_simulations),
        "save_gif": bool(save_gif),
        "gif_sample_games_per_pair": int(gif_sample_games_per_pair),
        "gif_sample_strategy": str(gif_sample_strategy),
    }
    if mode == "rating":
        spec = {
            "tournament_id": resolved_tournament_id,
            "rating_run_id": resolved_rating_run_id,
            "checkpoints": checkpoint_specs,
            "round_count": int(round_count),
            "continue_from_latest": bool(continue_from_latest),
            "pairs_per_round": int(pairs_per_round) if int(pairs_per_round) > 0 else None,
            "placement_min_games": (
                int(placement_min_games) if int(placement_min_games) > 0 else None
            ),
            "placement_min_opponents": int(placement_min_opponents),
            "pair_selection": pair_selection,
            "initial_rating": float(initial_rating),
            "active_pool_limit": int(active_pool_limit),
            "stop_when_stable": bool(stop_when_stable),
            **common,
        }
        call = curvytron_rating_loop.spawn(spec)
    elif mode == "tournament":
        spec = {
            "tournament_id": resolved_tournament_id,
            "checkpoints": checkpoint_specs,
            **common,
        }
        call = curvytron_tournament_run.spawn(spec)
    else:
        pair = arena.build_pair_specs(
            checkpoints=checkpoint_specs,
            tournament_id=resolved_tournament_id,
            **common,
        )[0]
        if mode == "pair":
            call = curvytron_tournament_pair.spawn(pair)
        else:
            game = arena.build_game_specs_for_pair(pair)[0]
            call = curvytron_tournament_game.spawn(game)
    call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
    payload: dict[str, Any] = {
        "status": "spawned",
        "app_name": APP_NAME,
        "mode": mode,
        "tournament_id": resolved_tournament_id,
        "function_call_id": call_id,
        "checkpoint_discovery": (
            {
                "found_count": discovery.get("found_count"),
                "missing_count": discovery.get("missing_count"),
                "checkpoint_iteration": discovery.get("checkpoint_iteration"),
            }
            if isinstance(discovery, Mapping)
            else None
        ),
        "plan_estimate": plan_estimate,
        "browser_url_hint": (
            "deploy this module, then open the curvytron_tournament_browser web endpoint"
        ),
    }
    if wait:
        payload["result"] = call.get()
    print(json.dumps(arena._to_plain(payload), indent=2, sort_keys=True))
