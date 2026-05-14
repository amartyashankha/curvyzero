"""Modal app for CurvyTron checkpoint tournaments.

One app owns the whole tournament lane. The lowest-level function runs one game.
Higher functions fan out over games and checkpoint pairs.

Checkpoint files are read from the training Volume. Tournament summaries and
GIFs are written to a separate v2 Volume.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.tournament import curvytron_checkpoint_tournament as arena


APP_NAME = "curvyzero-checkpoint-tournament"
CHECKPOINT_VOLUME_NAME = "curvyzero-runs"
TOURNAMENT_VOLUME_NAME = "curvyzero-curvytron-tournaments"
LIGHTZERO_VERSION = "0.2.0"
REMOTE_ROOT = Path("/repo")
RUNS_MOUNT = Path("/runs")
TOURNAMENT_MOUNT = Path("/tournament-runs")
CURVYTRON_BONUS_SPRITE_SHEET_RELATIVE_PATH = (
    "third_party/curvytron-reference/web/images/bonus.png"
)
DEFAULT_LIMIT = 50
MAX_LIMIT = 500
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
WEB_PROVISIONAL_RATING_CACHE_TTL_SECONDS = 30.0
WEB_GIF_BYTES_CACHE_TTL_SECONDS = 300.0
WEB_GIF_BYTES_CACHE_MAX_ITEM_BYTES = 24 * 1024 * 1024
DEFAULT_PROVISIONAL_RATING_INTERVAL_SECONDS = 60.0
DEFAULT_PROVISIONAL_RATING_MAX_SECONDS = 23 * 60 * 60
TRAINING_TASK_ID = "lightzero-curvytron-visual-survival"
CHECKPOINT_INTAKE_DICT_NAME = "curvyzero-curvytron-checkpoint-intake-v0"
CHECKPOINT_INTAKE_QUEUE_NAME = "curvyzero-curvytron-checkpoint-events-v0"
CHECKPOINT_INTAKE_ACTIVE_KEYS = "active_manifest_keys"
DEFAULT_CHECKPOINT_INTAKE_SCAN_SECONDS = 60
DEFAULT_CHECKPOINT_INTAKE_QUEUE_TTL_SECONDS = 24 * 60 * 60
LOW_LEVEL_WORKER_RETRIES = modal.Retries(
    max_retries=2,
    initial_delay=5.0,
    backoff_coefficient=2.0,
    max_delay=60.0,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        f"LightZero=={LIGHTZERO_VERSION}",
        "numpy>=1.26",
        "cloudpickle>=3",
        "pillow>=10",
        "fastapi>=0.110",
    )
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_file(
        Path.cwd() / CURVYTRON_BONUS_SPRITE_SHEET_RELATIVE_PATH,
        remote_path=str(REMOTE_ROOT / CURVYTRON_BONUS_SPRITE_SHEET_RELATIVE_PATH),
        copy=True,
    )
)
checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME,
    create_if_missing=True,
).read_only()
tournament_volume = modal.Volume.from_name(
    TOURNAMENT_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
checkpoint_intake_state = modal.Dict.from_name(
    CHECKPOINT_INTAKE_DICT_NAME,
    create_if_missing=True,
)
checkpoint_intake_queue = modal.Queue.from_name(
    CHECKPOINT_INTAKE_QUEUE_NAME,
    create_if_missing=True,
)
app = modal.App(APP_NAME)
_LAST_WEB_VOLUME_RELOAD_TS = 0.0
_WEB_CACHE: dict[str, tuple[float, Any]] = {}


def _checkpoint_volumes() -> dict[str, Any]:
    return {RUNS_MOUNT.as_posix(): checkpoint_volume}


def _tournament_volumes() -> dict[str, Any]:
    return {TOURNAMENT_MOUNT.as_posix(): tournament_volume}


def _game_volumes() -> dict[str, Any]:
    return {**_checkpoint_volumes(), **_tournament_volumes()}


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
    _LAST_WEB_VOLUME_RELOAD_TS = now
    error = _reload_volume(volume, force=force)
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


def _sort_discovery_rows_by_latest_checkpoint(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
                candidates.append(
                    {
                        "run_id": run_id,
                        "found": True,
                        "attempt_id": attempt_root.name,
                        "exp_dir_name": ckpt_dir.parent.name,
                        "checkpoint_name": checkpoint_path.name,
                        "iteration": int(iteration),
                        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
                        "checkpoint_size_bytes": int(stat.st_size),
                        "checkpoint_ref": runs.file_ref(checkpoint_path, mount=mount),
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
    if checkpoint_selection == arena.CHECKPOINT_SELECTION_ITERATION and checkpoint_iteration is None:
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


def _clean_ref_set(refs: Any) -> set[str]:
    if isinstance(refs, str):
        values = refs.replace("\n", ",").split(",")
    elif isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)):
        values = refs
    else:
        values = []
    return {str(ref).strip() for ref in values if str(ref).strip()}


def _parse_run_ids(run_ids: Any) -> list[str]:
    if isinstance(run_ids, str):
        return [
            item.strip()
            for item in run_ids.replace("\n", ",").split(",")
            if item.strip()
        ]
    if isinstance(run_ids, Sequence) and not isinstance(run_ids, (str, bytes)):
        return [str(item).strip() for item in run_ids if str(item).strip()]
    return []


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
                {
                    "checkpoint_ref": ref,
                    "iteration": _checkpoint_iteration_from_path(Path(ref)),
                    "found": True,
                }
                for ref in explicit_refs
            ],
        }
    return _discover_checkpoint_refs(
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


def _intake_rating_spec_from_manifest(
    manifest: Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rating_defaults = manifest.get("rating_defaults")
    if not isinstance(rating_defaults, Mapping):
        rating_defaults = {}
    extra = dict(overrides or {})
    checkpoint_refs = [
        str(ref)
        for ref in manifest.get("checkpoint_refs", [])
        if str(ref).strip()
    ]
    return arena.normalize_rating_spec(
        {
            "tournament_id": manifest["tournament_id"],
            "rating_run_id": manifest["rating_run_id"],
            "checkpoints": checkpoint_refs,
            "round_count": extra.get(
                "round_count",
                rating_defaults.get("round_count", arena.DEFAULT_RATING_ROUND_COUNT),
            ),
            "continue_from_latest": extra.get(
                "continue_from_latest",
                rating_defaults.get("continue_from_latest", False),
            ),
            "pairs_per_round": extra.get(
                "pairs_per_round",
                rating_defaults.get("pairs_per_round"),
            ),
            "placement_min_games": extra.get(
                "placement_min_games",
                rating_defaults.get("placement_min_games"),
            ),
            "placement_min_opponents": extra.get(
                "placement_min_opponents",
                rating_defaults.get("placement_min_opponents", 20),
            ),
            "pair_selection": extra.get(
                "pair_selection",
                rating_defaults.get(
                    "pair_selection",
                    arena.RATING_PAIR_SELECTION_ADAPTIVE_V0,
                ),
            ),
            "games_per_pair": extra.get(
                "games_per_pair",
                rating_defaults.get("games_per_pair", arena.DEFAULT_GAMES_PER_PAIR),
            ),
            "games_per_shard": extra.get(
                "games_per_shard",
                rating_defaults.get("games_per_shard", arena.DEFAULT_GAMES_PER_PAIR),
            ),
            "reuse_policies_per_shard": extra.get(
                "reuse_policies_per_shard",
                rating_defaults.get(
                    "reuse_policies_per_shard",
                    arena.DEFAULT_REUSE_POLICIES_PER_SHARD,
                ),
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
            "num_simulations": extra.get(
                "num_simulations",
                rating_defaults.get("num_simulations", arena.DEFAULT_NUM_SIMULATIONS),
            ),
            "save_gif": extra.get(
                "save_gif",
                rating_defaults.get("save_gif", False),
            ),
            "gif_sample_games_per_pair": extra.get(
                "gif_sample_games_per_pair",
                rating_defaults.get(
                    "gif_sample_games_per_pair",
                    arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR,
                ),
            ),
            "gif_sample_strategy": extra.get(
                "gif_sample_strategy",
                rating_defaults.get(
                    "gif_sample_strategy",
                    arena.DEFAULT_GIF_SAMPLE_STRATEGY,
                ),
            ),
            "initial_rating": extra.get(
                "initial_rating",
                rating_defaults.get(
                    "initial_rating",
                    arena.DEFAULT_RATING_INITIAL_RATING,
                ),
            ),
            "stop_when_stable": extra.get(
                "stop_when_stable",
                rating_defaults.get("stop_when_stable", False),
            ),
        }
    )


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
            str(ref)
            for ref in existing.get("seen_checkpoint_refs", [])
            if str(ref).strip()
        ]
        queued_refs = _clean_ref_set(existing.get("queued_checkpoint_refs", []))
    checkpoint_refs = [
        str(ref)
        for ref in discovery.get("checkpoint_refs", [])
        if str(ref).strip()
    ]
    seen_refs = sorted(set(existing_refs).union(checkpoint_refs))
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
        "discovery": arena._to_plain(dict(discovery)),
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


def _put_intake_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    key = str(manifest["manifest_key"])
    checkpoint_intake_state.put(key, arena._to_plain(dict(manifest)))
    active_keys = checkpoint_intake_state.get(CHECKPOINT_INTAKE_ACTIVE_KEYS, []) or []
    if not isinstance(active_keys, list):
        active_keys = []
    active_set = {str(item) for item in active_keys}
    if manifest.get("active"):
        active_set.add(key)
    else:
        active_set.discard(key)
    checkpoint_intake_state.put(CHECKPOINT_INTAKE_ACTIVE_KEYS, sorted(active_set))
    return {"key": key, "active_manifest_count": len(active_set)}


def _write_intake_manifest_artifact(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return arena.write_json_artifact(
        TOURNAMENT_MOUNT,
        arena.tournament_intake_manifest_ref(
            str(manifest["tournament_id"]),
            str(manifest["rating_run_id"]),
        ),
        manifest,
    )


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
        enqueued.append(event)
    return {"partition": partition, "enqueued_count": len(enqueued), "events": enqueued}


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
            sum(1 for path in rating_root.iterdir() if path.is_dir())
            if rating_root.exists()
            else 0
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
        updated_ts = max((path.stat().st_mtime for path in candidates), default=root.stat().st_mtime)
        rows.append(
            {
                "tournament_id": tournament_id,
                "visible": marker.exists(),
                "marker_ref": arena.tournament_marker_ref(tournament_id).as_posix(),
                "status": complete.get("status") or manifest.get("status"),
                "pair_count": complete.get("pair_count") or manifest.get("pair_count"),
                "checkpoint_count": complete.get("checkpoint_count") or manifest.get("checkpoint_count"),
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
        return {
            runs.clean_id(item, label="tournament_id")
            for item in raw_values
            if item
        }

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
            "checkpoint_roster": arena.rating_roster_by_checkpoint(
                rating_spec["checkpoints"]
            ),
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
                expected_roster=arena.rating_roster_by_checkpoint(
                    rating_spec["checkpoints"]
                ),
                label="latest snapshot",
            )
            latest_snapshot = dict(latest)
            start_round_index = int(latest_snapshot.get("round_index", -1) or -1) + 1

    previous_pair_history = _read_json(
        runs.volume_path(
            mount,
            arena.rating_pair_history_ref(
                rating_spec["tournament_id"],
                rating_spec["rating_run_id"],
            ),
        )
    ) or None
    scheduler_state = _read_json(
        runs.volume_path(
            mount,
            arena.rating_scheduler_state_ref(
                rating_spec["tournament_id"],
                rating_spec["rating_run_id"],
            ),
        )
    ) or None
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
        str(pair["battle_id"]): int(pair.get("games_per_pair") or 0)
        for pair in pair_specs
    }
    pair_index_by_battle = {
        str(pair["battle_id"]): int(pair.get("pair_index", 0) or 0)
        for pair in pair_specs
    }
    shard_size_by_battle = {
        str(pair["battle_id"]): max(
            1,
            int(pair.get("games_per_shard") or arena.DEFAULT_GAMES_PER_SHARD),
        )
        for pair in pair_specs
    }
    seen_pair_dir_ids: set[str] = set()
    seen_shard_game_counts: dict[str, int] = {}
    if pair_only and game_results is None:
        root = runs.volume_path(mount, arena.tournament_root_ref(clean_tournament_id)) / "battles"
        if root.exists():
            expected_battle_ids = set(expected_by_battle)
            for battle_root in root.iterdir():
                if battle_root.is_dir() and battle_root.name in expected_battle_ids:
                    seen_pair_dir_ids.add(battle_root.name)
            for path in root.glob("*/shards/*/summary.json"):
                battle_id = path.parents[2].name
                if battle_id not in expected_battle_ids:
                    continue
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
            if count_game_summaries:
                game_summary_counts: dict[str, int] = defaultdict(int)
                for path in root.glob("*/games/*/summary.json"):
                    battle_id = path.parents[2].name
                    if battle_id in expected_battle_ids:
                        game_summary_counts[battle_id] += 1
                for battle_id, game_summary_count in game_summary_counts.items():
                    expected = int(expected_by_battle.get(battle_id) or 0)
                    seen_shard_game_counts[battle_id] = min(
                        expected,
                        max(
                            seen_shard_game_counts.get(battle_id, 0),
                            int(game_summary_count),
                        ),
                    )
        summaries = []
    elif game_results is None:
        summaries: Sequence[tuple[Path | None, dict[str, Any]]] = _iter_rating_game_summaries(
            mount,
            tournament_id=clean_tournament_id,
            expected_battle_ids=set(expected_by_battle),
            load_payloads=load_summaries,
        )
    else:
        summaries = [
            (None, dict(result))
            for result in game_results
            if isinstance(result, Mapping)
        ]
    games_by_battle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_games: set[tuple[str, str]] = set()
    ok_game_count = 0
    failed_game_count = 0
    unknown_result_count = 0
    for path, payload in summaries:
        battle_id = str(
            payload.get("battle_id")
            or (path.parents[2].name if path is not None else "")
        )
        game_id = str(
            payload.get("game_id")
            or (path.parent.name if path is not None else "")
        )
        if not battle_id or not game_id:
            continue
        key = (battle_id, game_id)
        if key in seen_games:
            continue
        seen_games.add(key)
        if path is not None:
            payload.setdefault("summary_ref", runs.file_ref(path, mount=mount))
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

    game_count = int(input_payload.get("game_count") or sum(expected_by_battle.values()))
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
    estimated_seen_game_count = (
        max(
            completed_game_count,
            sum(int(expected_by_battle.get(battle_id) or 0) for battle_id in seen_pair_dir_ids),
        )
        if pair_only
        else None
    )
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
        "count_basis": "shard_summary_files" if pair_only else "summary_files",
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
    payload["status"] = "complete" if completed_game_count >= game_count and game_count else "running"
    payload["phase"] = "all_games_seen" if payload["status"] == "complete" else "games_running"
    latest_snapshot = _read_json(
        runs.volume_path(
            mount,
            arena.rating_latest_ref(clean_tournament_id, clean_rating_run_id),
        )
    )
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
    arena.write_json_artifact(
        mount,
        arena.rating_round_progress_ref(tournament_id, rating_run_id, round_id),
        progress,
    )
    return arena.write_json_artifact(
        mount,
        arena.rating_progress_ref(tournament_id, rating_run_id),
        progress,
    )


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
    checkpoint_roster = arena.rating_roster_by_checkpoint(
        spec.get("checkpoints") or []
    )
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
    if round_index <= 0:
        return None
    previous_round_id = arena.rating_round_id(round_index - 1)
    previous = _read_json(
        runs.volume_path(
            mount,
            arena.rating_round_ratings_ref(tournament_id, rating_run_id, previous_round_id),
        )
    )
    return previous or None


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
    arena.write_json_artifact(
        mount,
        arena.rating_pair_history_ref(spec["tournament_id"], spec["rating_run_id"]),
        pair_history,
    )
    scheduler_state = _write_rating_scheduler_state(
        mount,
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
    arena.write_json_artifact(
        mount,
        arena.rating_latest_ref(spec["tournament_id"], spec["rating_run_id"]),
        slim_snapshot,
    )
    return arena._to_plain(
        {
            "snapshot": slim_snapshot,
            "pair_history": pair_history,
            "scheduler_state": scheduler_state,
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
    progress, games_by_battle = _rating_round_progress_payload(
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
    expected_game_count = int(progress.get("game_count") or 0)
    completed_game_count = int(progress.get("completed_game_count") or 0)
    if not allow_partial and completed_game_count < expected_game_count:
        _write_rating_progress(mount, progress)
        raise ValueError(
            "rating round incomplete: "
            f"{completed_game_count}/{expected_game_count} games, "
            f"{progress.get('completed_pair_count')}/{progress.get('pair_count')} pairs"
        )

    spec = arena.normalize_rating_spec(input_payload.get("rating_spec") or {})
    round_index = int(input_payload.get("round_index", 0) or 0)
    previous_snapshot = _previous_rating_snapshot(
        mount,
        tournament_id=spec["tournament_id"],
        rating_run_id=spec["rating_run_id"],
        round_index=round_index,
    )
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

    round_outputs = _write_rating_round_outputs(
        mount,
        spec=spec,
        round_id=round_id,
        round_index=round_index,
        pair_results=pair_results,
        pair_specs=pair_specs,
        game_count=sum(len(games) for games in games_by_battle.values()),
        started_at=str(input_payload.get("started_at") or ""),
        previous_snapshot=previous_snapshot,
    )
    snapshot = round_outputs["snapshot"]
    progress["reduced_at"] = runs.utc_timestamp()
    progress["allow_partial_reduce"] = bool(allow_partial)
    progress["rated_pair_count"] = snapshot.get("rated_pair_count")
    _write_rating_progress(mount, progress)
    return arena._to_plain(
        {
            "progress": progress,
            "snapshot": snapshot,
            "pair_history": round_outputs.get("pair_history"),
            "scheduler_state": round_outputs.get("scheduler_state"),
            "pair_count": len(pair_results),
            "game_count": sum(len(games) for games in games_by_battle.values()),
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
    root = (
        runs.volume_path(mount, arena.battle_root_ref(tournament_id, battle_id))
        / "shards"
    )
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
        [
            dict(shard.get("tally"))
            for shard in shards
            if isinstance(shard.get("tally"), Mapping)
        ]
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
    game_count = int(input_payload.get("game_count") or sum(int(pair.get("games_per_pair") or 0) for pair in pair_specs))
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
            [
                game
                for pair in pair_specs
                for game in arena.build_game_specs_for_pair(pair)
            ],
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
            work_kind == "shard"
            and work_specs
            and work_specs[0].get("reuse_policies", True)
        ),
    }


def _rating_spec_with_latest_roster(
    mount: Path,
    raw_spec: Mapping[str, Any],
) -> dict[str, Any]:
    raw = dict(raw_spec)
    checkpoints = raw.get("checkpoints") or raw.get("checkpoint_refs") or []
    if checkpoints or not bool(raw.get("continue_from_latest", False)):
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
    ordered_ids.extend(
        checkpoint_id
        for checkpoint_id in sorted(str(item) for item in roster)
        if checkpoint_id not in set(ordered_ids)
    )

    restored_checkpoints = []
    for checkpoint_id in ordered_ids:
        roster_row = roster.get(checkpoint_id, {})
        if not isinstance(roster_row, Mapping):
            roster_row = {}
        rating_row = rows.get(checkpoint_id, {})
        checkpoint_ref = roster_row.get("checkpoint_ref") or rating_row.get(
            "checkpoint_ref"
        )
        if not checkpoint_ref:
            continue
        restored_checkpoints.append(
            {
                "checkpoint_id": checkpoint_id,
                "label": rating_row.get("label") or checkpoint_id,
                "checkpoint_ref": checkpoint_ref,
                "model_env_variant": roster_row.get("model_env_variant"),
                "model_reward_variant": roster_row.get("model_reward_variant"),
                "policy_trail_render_mode": roster_row.get(
                    "policy_trail_render_mode"
                ),
            }
        )
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
    print(json.dumps(arena._to_plain({
        "found_count": result["found_count"],
        "missing_count": result["missing_count"],
        "checkpoint_iteration": result["checkpoint_iteration"],
    }), sort_keys=True))
    return arena._to_plain(result)


@app.function(
    image=image,
    volumes=_game_volumes(),
    cpu=1.0,
    memory=4096,
    timeout=30 * 60,
    max_containers=500,
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
        result = arena.failure_game_summary(
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
    timeout=2 * 60 * 60,
    max_containers=500,
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
    shard_policy_mode = str(
        dict(game_specs[0]).get("policy_mode", arena.POLICY_MODE_EVAL)
    )
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
            result = arena.failure_game_summary(
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
    payload["game_summary_ref_count"] = sum(
        1 for game in games if game.get("summary_ref")
    )
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
    timeout=4 * 60 * 60,
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
    print(json.dumps(arena._to_plain({"battle_id": pair["battle_id"], "tally": summary["tally"]}), sort_keys=True))
    return arena._to_plain(summary)


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    cpu=1.0,
    memory=2048,
    timeout=12 * 60 * 60,
    max_containers=10,
)
def curvytron_tournament_run(tournament_spec: dict[str, Any]) -> dict[str, Any]:
    _reload_volume(tournament_volume)
    spec = dict(tournament_spec)
    tournament_id = runs.clean_id(str(spec.get("tournament_id") or runs.new_run_id("arena")), label="tournament_id")
    spec["tournament_id"] = tournament_id
    started_at = runs.utc_timestamp()
    _write_tournament_marker(tournament_id)
    _write_tournament_manifest(spec, status="running")
    startup_commit_error = _commit_volume(tournament_volume)

    checkpoints = spec.get("checkpoints") or spec.get("checkpoint_refs") or []
    if isinstance(checkpoints, str):
        checkpoints = arena.parse_checkpoint_refs(checkpoints)
    if not isinstance(checkpoints, list):
        raise ValueError("tournament spec needs a checkpoints list or comma-separated checkpoint_refs")
    pair_specs = arena.build_pair_specs(
        tournament_id=tournament_id,
        checkpoints=checkpoints,
        games_per_pair=int(spec.get("games_per_pair", arena.DEFAULT_GAMES_PER_PAIR)),
        ordered_pairs=bool(spec.get("ordered_pairs", arena.DEFAULT_ORDERED_PAIRS)),
        include_self_pairs=bool(
            spec.get("include_self_pairs", arena.DEFAULT_INCLUDE_SELF_PAIRS)
        ),
        seed=int(spec.get("seed", 0)),
        max_steps=int(spec.get("max_steps", arena.DEFAULT_MAX_STEPS)),
        decision_ms=float(spec.get("decision_ms", arena.DEFAULT_DECISION_MS)),
        decision_source_frames=spec.get("decision_source_frames"),
        source_physics_step_ms=float(
            spec.get("source_physics_step_ms", arena.DEFAULT_SOURCE_PHYSICS_STEP_MS)
        ),
        num_simulations=int(spec.get("num_simulations", arena.DEFAULT_NUM_SIMULATIONS)),
        policy_batch_size=int(
            spec.get("policy_batch_size", arena.DEFAULT_POLICY_BATCH_SIZE)
        ),
        policy_mode=str(spec.get("policy_mode", arena.POLICY_MODE_EVAL)),
        collect_temperature=float(
            spec.get("collect_temperature", arena.DEFAULT_COLLECT_TEMPERATURE)
        ),
        collect_epsilon=float(spec.get("collect_epsilon", arena.DEFAULT_COLLECT_EPSILON)),
        natural_bonus_spawn=bool(spec.get("natural_bonus_spawn", True)),
        policy_trail_render_mode=(
            spec.get("policy_trail_render_mode")
            or spec.get("observation_trail_render_mode")
            or spec.get("trail_render_mode")
        ),
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
        gif_sample_strategy=str(
            spec.get("gif_sample_strategy", arena.DEFAULT_GIF_SAMPLE_STRATEGY)
        ),
        games_per_shard=int(
            spec.get("games_per_shard", arena.DEFAULT_GAMES_PER_SHARD)
        ),
        reuse_policies_per_shard=bool(
            spec.get(
                "reuse_policies_per_shard",
                arena.DEFAULT_REUSE_POLICIES_PER_SHARD,
            )
        ),
        save_frames_npz=bool(
            spec.get("save_frames_npz", arena.DEFAULT_SAVE_FRAMES_NPZ)
        ),
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
        "pool_hash": arena.rating_pool_hash(spec["checkpoints"]),
        "roster_hash": arena.rating_pool_hash(spec["checkpoints"]),
        "context_hash": arena.rating_context_hash(spec),
        "checkpoint_roster": arena.rating_roster_by_checkpoint(spec["checkpoints"]),
        "round_id": round_id,
        "round_index": round_index,
        "started_at": started_at,
        "rating_spec": arena._to_plain(spec),
        "previous_round_id": (
            previous_snapshot.get("round_id")
            if isinstance(previous_snapshot, Mapping)
            else None
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
        previous_snapshot=(
            previous_snapshot if isinstance(previous_snapshot, Mapping) else None
        ),
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
    print(json.dumps(arena._to_plain({
        "rating_run_id": spec["rating_run_id"],
        "round_id": round_id,
        "pair_count": len(pair_results),
        "game_count": game_count,
        "rated_pair_count": slim_snapshot["rated_pair_count"],
        "max_abs_delta": slim_snapshot["max_abs_delta"],
        "stable": slim_snapshot["stable"],
    }), sort_keys=True))
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
    provisional_call_id = (
        getattr(provisional_call, "object_id", None)
        or getattr(provisional_call, "id", None)
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
        result = curvytron_rating_round.remote(
            {
                **spec,
                "round_index": round_index,
                "previous_snapshot": previous_snapshot,
                "pair_history": previous_pair_history,
                "scheduler_state": scheduler_state,
            }
        )
        previous_snapshot = result["snapshot"]
        previous_pair_history = result.get("pair_history")
        scheduler_state = result.get("scheduler_state")
        if isinstance(previous_snapshot, Mapping):
            arena.write_json_artifact(
                TOURNAMENT_MOUNT,
                arena.rating_latest_ref(spec["tournament_id"], spec["rating_run_id"]),
                _slim_rating_snapshot(previous_snapshot),
            )
            _commit_volume(tournament_volume)
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


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    cpu=2.0,
    memory=8192,
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
    cpu=2.0,
    memory=8192,
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


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    cpu=2.0,
    memory=8192,
    timeout=4 * 60 * 60,
    max_containers=3,
)
def curvytron_rating_reduce(reduce_spec: dict[str, Any]) -> dict[str, Any]:
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
    print(
        json.dumps(
            arena._to_plain(
                {
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
            ),
            sort_keys=True,
        )
    )
    return arena._to_plain(result)


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
        rows = [
            row
            for row in rows
            if _battle_matches_checkpoint(row, selected_checkpoint_id)
        ]
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
        config_spec = config.get("rating_spec") if isinstance(config.get("rating_spec"), Mapping) else {}
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
            path
            for path in (latest_path, progress_path, config_path)
            if path.exists()
        ]
        updated_path = max(updated_paths, key=lambda path: path.stat().st_mtime)
        rows.append(
            {
                "tournament_id": clean_id,
                "rating_run_id": rating_run_id,
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
                    runs.file_ref(latest_path, mount=mount)
                    if latest_path.exists()
                    else None
                ),
                "progress_ref": (
                    runs.file_ref(progress_path, mount=mount)
                    if progress_path.exists()
                    else None
                ),
                "config_ref": (
                    runs.file_ref(config_path, mount=mount)
                    if config_path.exists()
                    else None
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
    stored = _read_json(
        runs.volume_path(mount, arena.rating_progress_ref(tournament_id, selected))
    )
    if not stored:
        return {}
    if stored.get("status") == "complete":
        return stored
    round_id = str(
        stored.get("round_id")
        or arena.rating_round_id(int(stored.get("round_index", 0) or 0))
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
) -> dict[str, Any]:
    cache_key = f"live-progress:{mount}:{tournament_id}:{rating_run_id}"
    cached = _web_cache_get(cache_key, ttl_seconds=WEB_PROGRESS_CACHE_TTL_SECONDS)
    if isinstance(cached, dict):
        return cached
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
        merged.get("pair_count")
        or snapshot.get("total_pair_count")
        or completed_pairs
        or 0
    )
    game_count = int(
        merged.get("game_count")
        or snapshot.get("total_game_count")
        or completed_games
        or 0
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
    battles_root = (
        runs.volume_path(mount, arena.tournament_root_ref(tournament_id)) / "battles"
    )
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
    return str(rows[0]["tournament_id"]) if rows else ""


def _rating_row_by_checkpoint(
    rating_snapshot: Mapping[str, Any],
    checkpoint_id: str,
) -> dict[str, Any]:
    rows = rating_snapshot.get("ratings")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return {}
    for row in rows:
        if isinstance(row, Mapping) and str(row.get("checkpoint_id") or "") == checkpoint_id:
            return dict(row)
    return {}


def _rating_rows(rating_snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = rating_snapshot.get("ratings")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _rating_rank_by_checkpoint(rating_snapshot: Mapping[str, Any]) -> dict[str, int]:
    ranks = {}
    for row in _rating_rows(rating_snapshot):
        checkpoint_id = str(row.get("checkpoint_id") or "")
        if not checkpoint_id:
            continue
        try:
            ranks[checkpoint_id] = int(row.get("rank") or 0)
        except (TypeError, ValueError):
            continue
    return ranks


def _battle_player_for_checkpoint(
    row: Mapping[str, Any],
    checkpoint_id: str,
) -> dict[str, Any]:
    players = row.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return {}
    for player in players:
        if (
            isinstance(player, Mapping)
            and str(player.get("checkpoint_id") or "") == checkpoint_id
        ):
            return dict(player)
    return {}


def _battle_opponent_for_checkpoint(
    row: Mapping[str, Any],
    checkpoint_id: str,
) -> dict[str, Any]:
    players = row.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return {}
    for player in players:
        if (
            isinstance(player, Mapping)
            and str(player.get("checkpoint_id") or "") != checkpoint_id
        ):
            return dict(player)
    return {}


def _checkpoint_battle_sort_key(
    row: Mapping[str, Any],
    checkpoint_id: str,
    rank_by_checkpoint: Mapping[str, int],
) -> tuple[int, int, str]:
    opponent = _battle_opponent_for_checkpoint(row, checkpoint_id)
    opponent_id = str(opponent.get("checkpoint_id") or "")
    try:
        opponent_rank = int(rank_by_checkpoint.get(opponent_id) or 1_000_000)
    except (TypeError, ValueError):
        opponent_rank = 1_000_000
    try:
        pair_index = int(row.get("pair_index", 0) or 0)
    except (TypeError, ValueError):
        pair_index = 0
    return opponent_rank, pair_index, str(row.get("battle_id") or "")


def _sort_checkpoint_battle_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_id: str,
    rank_by_checkpoint: Mapping[str, int],
) -> list[dict[str, Any]]:
    return sorted(
        [dict(row) for row in rows if isinstance(row, Mapping)],
        key=lambda row: _checkpoint_battle_sort_key(
            row,
            checkpoint_id,
            rank_by_checkpoint,
        ),
    )


def _wins_for_checkpoint(row: Mapping[str, Any], checkpoint_id: str) -> int:
    tally = row.get("tally") if isinstance(row.get("tally"), Mapping) else {}
    wins_by_checkpoint = (
        tally.get("wins_by_checkpoint")
        if isinstance(tally.get("wins_by_checkpoint"), Mapping)
        else {}
    )
    if checkpoint_id in wins_by_checkpoint:
        return int(wins_by_checkpoint.get(checkpoint_id) or 0)
    player = _battle_player_for_checkpoint(row, checkpoint_id)
    seat = player.get("seat")
    wins_by_seat = (
        tally.get("wins_by_seat") if isinstance(tally.get("wins_by_seat"), Mapping) else {}
    )
    if seat in (0, 1):
        return int(wins_by_seat.get(f"seat_{seat}") or 0)
    return 0


def _review_battle_row(
    row: Mapping[str, Any],
    checkpoint_id: str,
    *,
    rank_by_checkpoint: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    battle = dict(row)
    tally = battle.get("tally") if isinstance(battle.get("tally"), Mapping) else {}
    opponent = _battle_opponent_for_checkpoint(battle, checkpoint_id)
    opponent_id = str(opponent.get("checkpoint_id") or "")
    rank_by_checkpoint = rank_by_checkpoint or {}
    checkpoint_wins = _wins_for_checkpoint(battle, checkpoint_id)
    opponent_wins = _wins_for_checkpoint(battle, opponent_id) if opponent_id else 0
    draws = int(tally.get("draw_count") or 0)
    battle.update(
        {
            "checkpoint_id": checkpoint_id,
            "opponent": opponent,
            "opponent_rank": rank_by_checkpoint.get(opponent_id),
            "checkpoint_wins": checkpoint_wins,
            "opponent_wins": opponent_wins,
            "draws": draws,
            "completed_count": int(tally.get("completed_count") or 0),
            "failure_count": int(tally.get("failure_count") or 0),
            "average_physical_steps": tally.get("average_physical_steps"),
        }
    )
    return arena._to_plain(battle)


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
        _list_rating_runs(mount, tournament_id=selected_tournament)
        if selected_tournament
        else []
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
    limit: int = MAX_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
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
            limit=1_000_000,
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
        if (
            isinstance(live_pair_results, Sequence)
            and not isinstance(live_pair_results, (str, bytes))
        ):
            battles = _battle_index_from_pair_results(
                live_pair_results,
                checkpoint_id=checkpoint_id,
                limit=MAX_LIMIT,
                offset=0,
            )
    source = str(battles.get("source") or "")
    battle_rows = battles.get("rows", [])
    raw_rows = _sort_checkpoint_battle_rows(
        battle_rows if isinstance(battle_rows, Sequence) else [],
        checkpoint_id=checkpoint_id,
        rank_by_checkpoint=rank_by_checkpoint,
    )
    page_rows = raw_rows[offset : offset + limit]
    if source in {"battle_index", "checkpoint_round_input", "live_shard_tallies"}:
        page_rows = [
            _enrich_battle_row_from_live_shards(mount, row)
            for row in page_rows
        ]
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
        "total": len(raw_rows),
        "limit": limit,
        "offset": offset,
        "has_older": offset + limit < len(raw_rows),
        "has_newer": offset > 0,
        "source": source,
    }


def _review_battle_payload(
    mount: Path,
    *,
    tournament_id: str = "",
    battle_id: str,
    gif_sample_limit: int = 10,
) -> dict[str, Any]:
    tournaments = _list_tournaments(mount)
    selected_tournament = _default_tournament_id(tournaments, tournament_id)
    cache_key = f"battle-detail:{mount}:{selected_tournament}:{battle_id}:{gif_sample_limit}"
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
                page = _list_battle_index(
                    mount,
                    tournament_id=selected_tournament,
                    limit=1_000_000,
                    offset=0,
                )
                source = str(page.get("source") or "battle_index")
                row = next(
                    (
                        dict(item)
                        for item in page.get("rows", [])
                        if isinstance(item, Mapping) and str(item.get("battle_id") or "") == battle_id
                    ),
                    {},
                )
                summary = _read_battle_summary(
                    mount,
                    tournament_id=selected_tournament,
                    battle_id=battle_id,
                    battle_index_row=row,
                )
    games, game_sources = (
        _read_game_summary_refs(
            mount,
            tournament_id=selected_tournament,
            battle_id=battle_id,
            battle_summary=summary,
        )
        if selected_tournament and (summary or row)
        else ([], [])
    )
    summary_without_games = dict(summary)
    summary_without_games.pop("games", None)
    samples = _sample_gif_refs(
        battle_summary=summary,
        games=games,
        limit=gif_sample_limit,
    )
    payload = {
        "selected_tournament_id": selected_tournament,
        "battle_id": battle_id,
        "battle": arena._to_plain(row),
        "summary": arena._to_plain(summary_without_games),
        "games": arena._to_plain(games),
        "game_count": len(games),
        "game_sources": game_sources,
        "sample_gifs": samples,
        "sample_gif_count": len(samples),
        "source": source,
    }
    if payload["game_count"] or payload["sample_gif_count"] or source == "battle_summary":
        return _web_cache_set(cache_key, arena._to_plain(payload))
    return arena._to_plain(payload)


def _href(path: str, **params: Any) -> str:
    clean = {
        key: str(value)
        for key, value in params.items()
        if value not in (None, "")
    }
    query = urlencode(clean)
    return f"{path}?{query}" if query else path


def _page_href(**params: Any) -> str:
    return _href("/", **params)


def _battle_href(**params: Any) -> str:
    return _href("/battle", **params)


def _friendly_progress_label(progress: Mapping[str, Any]) -> str:
    status = str(progress.get("status") or "")
    phase = str(progress.get("phase") or "")
    if status == "complete":
        return "rankings ready"
    if phase in {"game_map_started", "games_running", "all_games_seen"}:
        return "running games"
    if phase in {"reduced", "ratings_written"}:
        return "finalizing rankings"
    if status == "pending":
        return "starting"
    return (status or phase or "starting").replace("_", " ")


def _short_battle_label(battle_id: str, pair_index: Any = None) -> str:
    if pair_index is not None:
        return f"pair {pair_index}"
    marker = "-pair-"
    if marker in battle_id:
        tail = battle_id.split(marker, 1)[1]
        return f"pair {tail.split('-', 1)[0]}"
    return battle_id


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
    return _read_json(
        runs.volume_path(mount, arena.battle_summary_ref(tournament_id, battle_id))
    )


def _read_game_summary_refs(
    mount: Path,
    *,
    tournament_id: str,
    battle_id: str,
    battle_summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    games_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    sources: set[str] = set()

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
        for game in embedded_games:
            if isinstance(game, Mapping):
                add_game(game, "battle_json")

    summary_refs = battle_summary.get("game_summary_refs")
    if isinstance(summary_refs, Sequence) and not isinstance(summary_refs, (str, bytes)):
        for ref in summary_refs:
            if not isinstance(ref, str):
                continue
            payload = _read_tournament_json_ref(mount, ref)
            if payload:
                add_game(payload, "game_summary_refs")

    shard_refs = battle_summary.get("shard_summary_refs")
    if isinstance(shard_refs, Sequence) and not isinstance(shard_refs, (str, bytes)):
        for ref in shard_refs:
            if not isinstance(ref, str):
                continue
            shard = _read_tournament_json_ref(mount, ref)
            shard_games = shard.get("games") if isinstance(shard, Mapping) else None
            if not isinstance(shard_games, Sequence) or isinstance(shard_games, (str, bytes)):
                continue
            for game in shard_games:
                if isinstance(game, Mapping):
                    add_game(game, "shard_summary_refs")

    expected_count = int(
        (
            battle_summary.get("tally")
            if isinstance(battle_summary.get("tally"), Mapping)
            else {}
        ).get("game_count")
        or battle_summary.get("game_summary_ref_count")
        or 0
    )
    if games_by_key and (expected_count <= 0 or len(games_by_key) >= expected_count):
        games = list(games_by_key.values())
        games.sort(
            key=lambda game: (
                int(game.get("game_index", 0) or 0),
                str(game.get("game_id") or ""),
            )
        )
        return games, sorted(sources)

    games_root = (
        runs.volume_path(mount, arena.tournament_root_ref(tournament_id))
        / "battles"
        / runs.clean_id(battle_id, label="battle_id")
        / "games"
    )
    if games_root.exists():
        for path in sorted(games_root.glob("*/summary.json"), key=lambda item: item.as_posix()):
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
    return games, sorted(sources)


def _sample_gif_refs(
    *,
    battle_summary: Mapping[str, Any],
    games: Sequence[Mapping[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    sample_limit = max(1, int(limit))
    gif_games = [
        game
        for game in games
        if isinstance(game, Mapping) and game.get("gif_ref")
    ]
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
    if samples:
        return arena._to_plain(samples)

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
    if not samples and battle_summary.get("first_gif_ref"):
        samples.append(
            {
                "game_id": None,
                "game_index": None,
                "outcome": None,
                "gif_ref": battle_summary.get("first_gif_ref"),
            }
        )
    seen_refs: set[str] = set()
    deduped = []
    for sample in samples:
        ref = str(sample.get("gif_ref") or "")
        if not ref or ref in seen_refs:
            continue
        seen_refs.add(ref)
        deduped.append(sample)
    return arena._to_plain(deduped)


def _render_battle_detail_section(*, payload: Mapping[str, Any]) -> str:
    import html

    def fmt(value: Any, *, digits: int = 2) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return ""

    battle_id = str(payload.get("battle_id") or "")
    if not payload:
        return '<div class="empty">No battle detail found.</div>'
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    battle = payload.get("battle") if isinstance(payload.get("battle"), Mapping) else {}
    players = summary.get("players") or battle.get("players") or []
    if isinstance(players, Sequence) and not isinstance(players, (str, bytes)):
        matchup = " vs ".join(
            html.escape(str(player.get("label") or player.get("checkpoint_id") or ""))
            for player in players
            if isinstance(player, Mapping)
        )
    else:
        matchup = ""
    tally = summary.get("tally") if isinstance(summary.get("tally"), Mapping) else {}
    summary_ref = html.escape(str(summary.get("summary_ref") or battle.get("summary_ref") or ""))
    summary_link = f'<a href="/meta?ref={summary_ref}">JSON</a>' if summary_ref else ""
    samples = payload.get("sample_gifs") if isinstance(payload.get("sample_gifs"), list) else []
    games = payload.get("games") if isinstance(payload.get("games"), list) else []

    sample_cards = []
    for sample in samples:
        if not isinstance(sample, Mapping) or not sample.get("gif_ref"):
            continue
        gif_ref = html.escape(str(sample["gif_ref"]))
        caption = " ".join(
            item
            for item in (
                str(sample.get("game_id") or ""),
                str(sample.get("outcome") or ""),
            )
            if item
        )
        sample_cards.append(
            f"""
            <a class="gif-card" href="/gif?ref={gif_ref}">
              <img src="/gif?ref={gif_ref}" alt="{html.escape(caption)}" loading="lazy" decoding="async">
              <span>{html.escape(caption or "Sample")}</span>
            </a>
            """
        )
    sample_html = (
        f"""
        <section class="panel">
          <div class="panel-head"><h2>GIF Samples</h2><span>{len(sample_cards)} shown</span></div>
          <div class="gif-grid">{"".join(sample_cards)}</div>
        </section>
        """
        if sample_cards
        else '<p class="summary">No GIF samples were captured for this battle.</p>'
    )

    game_rows = []
    for game in games:
        if not isinstance(game, Mapping):
            continue
        game_summary_ref = html.escape(str(game.get("summary_ref") or ""))
        gif_ref = html.escape(str(game.get("gif_ref") or ""))
        gif_link = f'<a href="/gif?ref={gif_ref}">GIF</a>' if gif_ref else ""
        json_link = f'<a href="/meta?ref={game_summary_ref}">JSON</a>' if game_summary_ref else ""
        game_rows.append(
            "<tr>"
            f"<td>{html.escape(str(game.get('game_index') if game.get('game_index') is not None else ''))}</td>"
            f"<td>{html.escape(str(game.get('game_id') or ''))}</td>"
            f"<td>{html.escape(str(game.get('outcome') or ''))}</td>"
            f"<td>{html.escape(str(game.get('seed') or ''))}</td>"
            f"<td>{html.escape(str(game.get('physical_steps') or ''))}</td>"
            f"<td>{html.escape(str(game.get('ok')))}</td>"
            f"<td>{gif_link}</td>"
            f"<td>{json_link}</td>"
            "</tr>"
        )
    games_html = (
        f"""
        <section class="panel">
          <div class="panel-head"><h2>Games</h2><span>{html.escape(str(payload.get("game_count", 0)))} found</span></div>
          <table>
            <thead><tr><th>#</th><th>Game</th><th>Outcome</th><th>Seed</th><th>Steps</th><th>OK</th><th>GIF</th><th>JSON</th></tr></thead>
            <tbody>{"".join(game_rows)}</tbody>
          </table>
        </section>
        """
        if game_rows
        else '<div class="empty">No game summaries found for this battle.</div>'
    )

    return f"""
    <section class="panel selected-battle" id="battle-detail">
      <div class="panel-head">
        <h2>{html.escape(battle_id)}</h2>
        <span>{matchup}</span>
      </div>
      <div class="progress-body">
        <div><strong>{html.escape(str(tally.get("completed_count", 0)))}</strong><span>games</span></div>
        <div><strong>{html.escape(str(tally.get("failure_count", 0)))}</strong><span>failures</span></div>
        <div><strong>{html.escape(str(tally.get("draw_count", 0)))}</strong><span>draws</span></div>
        <div><strong>{fmt(tally.get("average_physical_steps"))}</strong><span>avg steps</span></div>
        <div><strong>{summary_link}</strong><span>battle JSON</span></div>
      </div>
    </section>
    {sample_html}
    {games_html}
    """


def _render_page(
    *,
    tournaments: list[dict[str, Any]],
    selected_tournament_id: str,
    selected_rating_run_id: str,
    selected_checkpoint_id: str,
    rating_runs: list[dict[str, Any]],
    rating_snapshot: dict[str, Any],
    rating_progress: dict[str, Any],
    battles: dict[str, Any],
    selected_battle_id: str = "",
    battle_detail: Mapping[str, Any] | None = None,
    volume_reload_error: str = "",
) -> str:
    import html
    import math

    def fmt_number(value: Any, *, digits: int = 1) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return ""

    def sort_number_attr(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        if not math.isfinite(number):
            return ""
        if number.is_integer():
            return str(int(number))
        return f"{number:.6f}".rstrip("0").rstrip(".")

    options = "\n".join(
        f'<option value="{html.escape(row["tournament_id"])}" '
        f'{"selected" if row["tournament_id"] == selected_tournament_id else ""}>'
        f'{html.escape(row["tournament_id"])}</option>'
        for row in tournaments
    )
    rating_options = "\n".join(
        f'<option value="{html.escape(row["rating_run_id"])}" '
        f'{"selected" if row["rating_run_id"] == selected_rating_run_id else ""}>'
        f'{html.escape(row["rating_run_id"])}'
        f'{html.escape(" (" + str(row.get("status")) + ")" if row.get("status") else "")}</option>'
        for row in rating_runs
    )
    rating_rows = _rating_rows(rating_snapshot)
    selected_rating_row = (
        _rating_row_by_checkpoint(rating_snapshot, selected_checkpoint_id)
        if selected_checkpoint_id
        else {}
    )
    rating_html = ""
    if rating_rows:
        provisional = bool(rating_snapshot.get("provisional"))
        ranking_title = "Live Rankings" if provisional else "Rankings"
        ranking_status = (
            "updating from finished games"
            if provisional
            else str(rating_snapshot.get("round_id", ""))
        )
        body = []
        for row in rating_rows:
            record = row if isinstance(row, Mapping) else {}
            checkpoint_id = str(record.get("checkpoint_id", ""))
            checkpoint_label = str(record.get("label") or checkpoint_id)
            href = _page_href(
                tournament_id=selected_tournament_id,
                rating_run_id=selected_rating_run_id,
                checkpoint_id=checkpoint_id,
            )
            selected_class = " selected-row" if checkpoint_id == selected_checkpoint_id else ""
            body.append(
                f"<tr class=\"{selected_class.strip()}\">"
                f"<td>{html.escape(str(record.get('rank', '')))}</td>"
                f"<td title=\"{html.escape(checkpoint_id)}\">"
                f"<a href=\"{html.escape(href)}\">{html.escape(checkpoint_label)}</a></td>"
                f"<td>{float(record.get('rating', 0.0)):.1f}</td>"
                f"<td>{html.escape(str(record.get('games', 0)))}</td>"
                f"<td>{html.escape(str(record.get('wins', 0)))}-"
                f"{html.escape(str(record.get('losses', 0)))}-"
                f"{html.escape(str(record.get('draws', 0)))}</td>"
                f"<td>{fmt_number(record.get('win_rate'), digits=3)}</td>"
                f"<td>{html.escape(str(record.get('distinct_opponents') or record.get('battles') or 0))}</td>"
                f"<td>{html.escape(str(record.get('failure_count', 0)))}</td>"
                "</tr>"
            )
        rating_html = f"""
        <section class="panel">
          <div class="panel-head">
            <h2>{html.escape(ranking_title)}</h2>
            <span>{html.escape(str(rating_snapshot.get("rating_run_id", "")))} / {html.escape(ranking_status)}</span>
          </div>
          <div class="scroll-panel rankings-scroll">
            <table>
              <thead><tr><th>Rank</th><th>Checkpoint</th><th>Rating</th><th>Games</th><th>W-L-D</th><th>Win rate</th><th>Opp.</th><th>Failures</th></tr></thead>
              <tbody>{"".join(body)}</tbody>
            </table>
          </div>
        </section>
        """
    elif rating_runs:
        status = str(rating_progress.get("status") or "")
        if status and status != "complete":
            friendly_status = _friendly_progress_label(rating_progress)
            rating_html = f"""
            <section class="panel">
              <div class="panel-head">
                <h2>Rankings</h2>
                <span>{html.escape(friendly_status)}</span>
              </div>
              <p class="in-panel">Rankings will appear as soon as finished games are visible. This page keeps updating while the tournament runs.</p>
            </section>
            """
        else:
            rating_html = """
            <section class="panel">
              <div class="panel-head"><h2>Rankings</h2><span>empty</span></div>
              <p class="in-panel">This rating run exists, but no rating rows were found.</p>
            </section>
            """
    progress_html = ""
    if rating_progress:
        try:
            pct = 100.0 * float(rating_progress.get("estimated_completion_fraction") or rating_progress.get("completion_fraction") or 0.0)
        except (TypeError, ValueError):
            pct = 0.0
        progress_html = f"""
        <section class="panel" id="progress-panel" data-tournament-id="{html.escape(selected_tournament_id)}" data-rating-run-id="{html.escape(selected_rating_run_id)}" data-has-ratings="{html.escape('true' if rating_rows else 'false')}">
          <div class="panel-head">
            <h2>Progress</h2>
            <span data-progress-field="updated">{html.escape(str(rating_progress.get("updated_at") or ""))}</span>
          </div>
          <div class="progress-body">
            <div><strong data-progress-field="status">{html.escape(_friendly_progress_label(rating_progress))}</strong><span>state</span></div>
            <div><strong data-progress-field="phase">{html.escape(str(rating_progress.get("phase") or ""))}</strong><span>detail</span></div>
            <div><strong data-progress-field="pairs">{html.escape(str(rating_progress.get("started_pair_count") or 0))}/{html.escape(str(rating_progress.get("pair_count") or 0))}</strong><span>pairs started</span></div>
            <div><strong data-progress-field="games">{html.escape(str(rating_progress.get("estimated_seen_game_count") or rating_progress.get("completed_game_count") or 0))}/{html.escape(str(rating_progress.get("game_count") or 0))}</strong><span>games seen</span></div>
            <div><strong data-progress-field="percent">{pct:.1f}%</strong><span>estimated progress</span></div>
          </div>
        </section>
        """
    elif rating_runs:
        progress_html = f"""
        <section class="panel" id="progress-panel" data-tournament-id="{html.escape(selected_tournament_id)}" data-rating-run-id="{html.escape(selected_rating_run_id)}" data-has-ratings="{html.escape('true' if rating_rows else 'false')}">
          <div class="panel-head">
            <h2>Progress</h2>
            <span data-progress-field="updated"></span>
          </div>
          <p class="in-panel">Tournament state is loading. This page will check again automatically.</p>
        </section>
        """
    checkpoint_html = ""
    if selected_checkpoint_id:
        detail_name = str(
            selected_rating_row.get("label")
            or selected_rating_row.get("checkpoint_id")
            or selected_checkpoint_id
        )
        if selected_rating_row:
            checkpoint_html = f"""
            <section class="panel selected-checkpoint">
              <div class="panel-head">
                <h2>{html.escape(detail_name)}</h2>
                <a href="{html.escape(_page_href(tournament_id=selected_tournament_id, rating_run_id=selected_rating_run_id))}">Clear</a>
              </div>
              <div class="progress-body">
                <div><strong>{float(selected_rating_row.get("rating", 0.0)):.1f}</strong><span>rating</span></div>
                <div><strong>{html.escape(str(selected_rating_row.get("rank", "")))}</strong><span>rank</span></div>
                <div><strong>{html.escape(str(selected_rating_row.get("games", 0)))}</strong><span>games</span></div>
                <div><strong>{html.escape(str(selected_rating_row.get("wins", 0)))}-{html.escape(str(selected_rating_row.get("losses", 0)))}-{html.escape(str(selected_rating_row.get("draws", 0)))}</strong><span>W-L-D</span></div>
                <div><strong>{html.escape(str(battles.get("total", 0)))}</strong><span>battles shown below</span></div>
              </div>
            </section>
            """
        else:
            checkpoint_html = f"""
            <section class="panel selected-checkpoint">
              <div class="panel-head">
                <h2>{html.escape(selected_checkpoint_id)}</h2>
                <a href="{html.escape(_page_href(tournament_id=selected_tournament_id, rating_run_id=selected_rating_run_id))}">Clear</a>
              </div>
              <p class="in-panel">No rating row was found for this checkpoint in the selected rating run.</p>
            </section>
            """
    battle_html = '<p class="summary">Select a checkpoint row to inspect its battles.</p>'
    if selected_checkpoint_id:
        rank_by_checkpoint = _rating_rank_by_checkpoint(rating_snapshot)
        body = []
        battle_rows = battles.get("rows", [])
        sorted_battle_rows = _sort_checkpoint_battle_rows(
            battle_rows if isinstance(battle_rows, Sequence) else [],
            checkpoint_id=selected_checkpoint_id,
            rank_by_checkpoint=rank_by_checkpoint,
        )
        for sort_index, raw_row in enumerate(sorted_battle_rows):
            row = _review_battle_row(
                raw_row,
                selected_checkpoint_id,
                rank_by_checkpoint=rank_by_checkpoint,
            )
            opponent = row.get("opponent") if isinstance(row.get("opponent"), Mapping) else {}
            opponent_id = str(opponent.get("checkpoint_id") or "")
            opponent_label = str(opponent.get("label") or opponent_id or "unknown")
            summary_ref = html.escape(str(row.get("summary_ref") or ""))
            summary_link = f'<a href="/meta?ref={summary_ref}">JSON</a>' if summary_ref else ""
            gif_ref = html.escape(str(row.get("first_gif_ref") or ""))
            gif_link = f'<a href="/gif?ref={gif_ref}">GIF</a>' if gif_ref else "No GIF"
            battle_id = str(row.get("battle_id") or "")
            battle_href = _page_href(
                tournament_id=selected_tournament_id,
                rating_run_id=selected_rating_run_id,
                checkpoint_id=selected_checkpoint_id,
                battle_id=battle_id,
            )
            if battle_id:
                battle_href += "#battle-detail"
            selected_battle_class = (
                ' class="selected-row"' if battle_id and battle_id == selected_battle_id else ""
            )
            opponent_rank_sort = sort_number_attr(row.get("opponent_rank"))
            avg_steps_sort = sort_number_attr(row.get("average_physical_steps"))
            failure_count_sort = sort_number_attr(row.get("failure_count"))
            body.append(
                f"<tr{selected_battle_class} data-battle-row "
                f"data-sort-index=\"{sort_index}\" "
                f"data-sort-rank=\"{html.escape(opponent_rank_sort)}\" "
                f"data-sort-avg-steps=\"{html.escape(avg_steps_sort)}\" "
                f"data-sort-failures=\"{html.escape(failure_count_sort)}\">"
                f"<td>{html.escape(str(row.get('opponent_rank') or ''))}</td>"
                f"<td title=\"{html.escape(opponent_id)}\"><a href=\"{html.escape(battle_href)}\">{html.escape(opponent_label)}</a></td>"
                f"<td>{html.escape(str(row.get('checkpoint_wins', 0)))}-"
                f"{html.escape(str(row.get('opponent_wins', 0)))}-"
                f"{html.escape(str(row.get('draws', 0)))}</td>"
                f"<td>{html.escape(str(row.get('completed_count', 0)))}</td>"
                f"<td>{fmt_number(row.get('average_physical_steps'), digits=2)}</td>"
                f"<td>{html.escape(str(row.get('failure_count', 0)))}</td>"
                f"<td>{gif_link}</td>"
                f"<td>{summary_link}</td>"
                f"<td><a href=\"{html.escape(battle_href)}\">Games</a></td>"
                "</tr>"
            )
        if body:
            battle_html = f"""
            <section class="panel">
              <div class="panel-head">
                <h2>Battles</h2>
                <span>{html.escape(str(battles.get("total", 0)))} total</span>
              </div>
              <div class="scroll-panel battles-scroll">
                <table data-battle-table data-sort-key="rank" data-sort-direction="asc">
                  <thead><tr><th aria-sort="ascending"><button type="button" class="sort-button" data-battle-sort="rank">Opp. rank <span class="sort-indicator" data-sort-indicator="rank">asc</span></button></th><th>Opponent</th><th>W-L-D</th><th>Games</th><th><button type="button" class="sort-button" data-battle-sort="avgSteps">Avg steps <span class="sort-indicator" data-sort-indicator="avgSteps"></span></button></th><th><button type="button" class="sort-button" data-battle-sort="failures">Failures <span class="sort-indicator" data-sort-indicator="failures"></span></button></th><th>GIF</th><th>JSON</th><th>Battle</th></tr></thead>
                  <tbody>{"".join(body)}</tbody>
                </table>
              </div>
            </section>
            """
        else:
            battle_html = '<div class="empty">No battles found for this checkpoint.</div>'
    elif rating_progress and isinstance(rating_progress.get("recent_started_pairs"), Sequence):
        recent_rows = [
            row
            for row in rating_progress.get("recent_started_pairs", [])
            if isinstance(row, Mapping) and row.get("battle_id")
        ]
        if recent_rows:
            body = []
            for row in recent_rows[:50]:
                battle_id = str(row.get("battle_id") or "")
                battle_label = _short_battle_label(battle_id, row.get("pair_index"))
                battle_href = _page_href(
                    tournament_id=selected_tournament_id,
                    rating_run_id=selected_rating_run_id,
                    battle_id=battle_id,
                )
                body.append(
                    "<tr>"
                    f"<td>{html.escape(str(row.get('pair_index') if row.get('pair_index') is not None else ''))}</td>"
                    f"<td title=\"{html.escape(battle_id)}\"><a href=\"{html.escape(battle_href)}#battle-detail\">{html.escape(battle_label)}</a></td>"
                    f"<td>{html.escape(str(row.get('expected_game_count') or ''))}</td>"
                    f"<td>{html.escape('yes' if row.get('complete') else 'running')}</td>"
                    f"<td><a href=\"{html.escape(battle_href)}#battle-detail\">Games</a></td>"
                    "</tr>"
                )
            battle_html = f"""
            <section class="panel">
              <div class="panel-head">
                <h2>Recent Battles</h2>
                <span>live sample</span>
              </div>
              <div class="scroll-panel battles-scroll">
                <table>
                  <thead><tr><th>Pair</th><th>Battle</th><th>Games</th><th>State</th><th>Open</th></tr></thead>
                  <tbody>{"".join(body)}</tbody>
                </table>
              </div>
            </section>
            """
    battle_detail_html = (
        _render_battle_detail_section(payload=battle_detail or {"battle_id": selected_battle_id})
        if selected_battle_id
        else ""
    )
    reload_html = (
        f"""
        <section class="panel">
          <div class="panel-head"><h2>Volume Refresh</h2><span>using last visible data</span></div>
          <p class="in-panel">{html.escape(volume_reload_error)}</p>
        </section>
        """
        if volume_reload_error
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CurvyTron Tournament</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #202124; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 18px; }}
    header {{ display: flex; align-items: end; justify-content: space-between; gap: 12px; margin-bottom: 14px; }}
    h1 {{ margin: 0; font-size: 22px; }}
    form {{ display: flex; gap: 8px; align-items: end; }}
    select, button {{ height: 36px; border: 1px solid #dadce0; border-radius: 6px; padding: 0 8px; background: white; }}
    button {{ background: #1a73e8; border-color: #1a73e8; color: white; }}
    .summary {{ margin: 0 0 12px; color: #5f6368; font-size: 13px; }}
    .in-panel {{ margin: 0; padding: 12px; color: #5f6368; font-size: 13px; }}
    .panel {{ background: white; border: 1px solid #dadce0; border-radius: 8px; margin-bottom: 12px; overflow: hidden; }}
    .panel-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; padding: 10px 12px; border-bottom: 1px solid #eef0f3; }}
    .panel h2 {{ margin: 0; font-size: 15px; }}
    .panel span {{ color: #5f6368; font-size: 12px; }}
    a {{ color: #1557b0; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .progress-body {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; background: #eef0f3; }}
    .progress-body div {{ display: grid; gap: 3px; padding: 12px; background: white; }}
    .progress-body strong {{ font-size: 18px; }}
    .progress-body span {{ color: #5f6368; font-size: 12px; }}
    .gif-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; padding: 10px; }}
    .gif-card {{ display: grid; gap: 6px; color: #202124; }}
    .gif-card img {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: #111827; }}
    .gif-card span {{ color: #5f6368; font-size: 12px; }}
    .scroll-panel {{ overflow: auto; }}
    .rankings-scroll {{ max-height: min(48vh, 520px); }}
    .battles-scroll {{ max-height: min(36vh, 380px); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 7px 10px; border-bottom: 1px solid #eef0f3; text-align: left; }}
    th {{ color: #5f6368; font-weight: 600; position: sticky; top: 0; z-index: 1; background: white; }}
    .sort-button {{ all: unset; display: inline-flex; align-items: center; gap: 4px; cursor: pointer; color: inherit; font: inherit; }}
    .sort-button:focus-visible {{ outline: 2px solid #1a73e8; outline-offset: 2px; border-radius: 4px; }}
    .panel .sort-indicator {{ min-width: 28px; color: #80868b; font-size: 11px; }}
    .selected-row td {{ background: #eef4ff; }}
    td:nth-child(n+3) {{ white-space: nowrap; }}
    .empty {{ padding: 60px; text-align: center; background: white; border: 1px dashed #dadce0; border-radius: 8px; color: #80868b; }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>CurvyTron Tournament</h1>
      <p class="summary">Checkpoint battles. Score is who dies first.</p>
    </div>
    <form method="get" id="tournament-picker">
      <label>Tournament<br><select name="tournament_id" data-picker="tournament">{options}</select></label>
      <label>Rating<br><select name="rating_run_id" data-picker="rating">{rating_options}</select></label>
      <button type="submit">Open</button>
    </form>
  </header>
  {reload_html}
  {progress_html}
  {rating_html}
  {checkpoint_html}
  {battle_html}
  {battle_detail_html}
</main>
<script>
(() => {{
  const picker = document.getElementById("tournament-picker");
  const tournamentSelect = picker ? picker.querySelector("[name='tournament_id']") : null;
  const ratingSelect = picker ? picker.querySelector("[name='rating_run_id']") : null;
  const disablePicker = () => {{
    if (!picker) return;
    picker.querySelectorAll("select, button").forEach((node) => {{
      node.disabled = true;
    }});
  }};
  const navigatePicker = (changed) => {{
    if (!picker) return;
    const url = new URL(window.location.href);
    const tournamentId = tournamentSelect ? tournamentSelect.value : "";
    const ratingRunId = ratingSelect ? ratingSelect.value : "";
    if (tournamentId) {{
      url.searchParams.set("tournament_id", tournamentId);
    }} else {{
      url.searchParams.delete("tournament_id");
    }}
    if (changed === "tournament") {{
      url.searchParams.delete("rating_run_id");
    }} else if (ratingRunId) {{
      url.searchParams.set("rating_run_id", ratingRunId);
    }} else {{
      url.searchParams.delete("rating_run_id");
    }}
    url.searchParams.delete("checkpoint_id");
    url.searchParams.delete("battle_id");
    url.searchParams.delete("fresh");
    url.hash = "";
    disablePicker();
    window.location.assign(url.toString());
  }};
  if (tournamentSelect) {{
    tournamentSelect.addEventListener("change", () => navigatePicker("tournament"));
  }}
  if (ratingSelect) {{
    ratingSelect.addEventListener("change", () => navigatePicker("rating"));
  }}
  if (picker) {{
    picker.addEventListener("submit", (event) => {{
      event.preventDefault();
      navigatePicker("rating");
    }});
  }}
  const battleTable = document.querySelector("[data-battle-table]");
  if (battleTable) {{
    const tbody = battleTable.querySelector("tbody");
    const sortButtons = battleTable.querySelectorAll("[data-battle-sort]");
    const sortFields = {{
      rank: "sortRank",
      avgSteps: "sortAvgSteps",
      failures: "sortFailures",
    }};
    const sortValue = (row, key) => {{
      const field = sortFields[key];
      const raw = field ? row.dataset[field] : "";
      if (raw === undefined || raw === "") return null;
      const parsed = Number(raw);
      return Number.isFinite(parsed) ? parsed : null;
    }};
    const originalIndex = (row) => {{
      const parsed = Number(row.dataset.sortIndex || "0");
      return Number.isFinite(parsed) ? parsed : 0;
    }};
    const updateSortIndicators = (key, direction) => {{
      battleTable.dataset.sortKey = key;
      battleTable.dataset.sortDirection = direction;
      battleTable.querySelectorAll("th[aria-sort]").forEach((cell) => {{
        cell.removeAttribute("aria-sort");
      }});
      battleTable.querySelectorAll("[data-sort-indicator]").forEach((node) => {{
        node.textContent = node.dataset.sortIndicator === key ? direction : "";
      }});
      const activeButton = battleTable.querySelector(`[data-battle-sort="${{key}}"]`);
      if (activeButton && activeButton.closest("th")) {{
        activeButton.closest("th").setAttribute(
          "aria-sort",
          direction === "asc" ? "ascending" : "descending",
        );
      }}
    }};
    const applyBattleSort = (key, direction) => {{
      if (!tbody || !sortFields[key]) return;
      const multiplier = direction === "desc" ? -1 : 1;
      const rows = Array.from(tbody.querySelectorAll("[data-battle-row]"));
      rows.sort((a, b) => {{
        const left = sortValue(a, key);
        const right = sortValue(b, key);
        if (left === null && right === null) return originalIndex(a) - originalIndex(b);
        if (left === null) return 1;
        if (right === null) return -1;
        const diff = (left - right) * multiplier;
        if (diff !== 0) return diff;
        return originalIndex(a) - originalIndex(b);
      }});
      rows.forEach((row) => tbody.appendChild(row));
      updateSortIndicators(key, direction);
    }};
    sortButtons.forEach((button) => {{
      button.addEventListener("click", () => {{
        const key = button.dataset.battleSort || "rank";
        const currentKey = battleTable.dataset.sortKey || "rank";
        const currentDirection = battleTable.dataset.sortDirection || "asc";
        const nextDirection = key === currentKey && currentDirection === "asc" ? "desc" : "asc";
        applyBattleSort(key, nextDirection);
      }});
    }});
  }}
  const panel = document.getElementById("progress-panel");
  if (!panel) return;
  const fields = {{}};
  document.querySelectorAll("[data-progress-field]").forEach((node) => {{
    fields[node.dataset.progressField] = node;
  }});
  const text = (value) => value === null || value === undefined ? "" : String(value);
  const number = (value) => {{
    const parsed = Number(value || 0);
    return Number.isFinite(parsed) ? parsed : 0;
  }};
  const set = (name, value) => {{
    if (fields[name]) fields[name].textContent = value;
  }};
  const stateLabel = (progress) => {{
    if (progress.status === "complete") return "rankings ready";
    if (["game_map_started", "games_running", "all_games_seen"].includes(progress.phase)) return "running games";
    if (["reduced", "ratings_written"].includes(progress.phase)) return "finalizing rankings";
    if (progress.status === "pending") return "starting";
    return text(progress.status || progress.phase || "starting").replaceAll("_", " ");
  }};
  const reloadKey = `curvyzero:tournament-ratings-reloaded:${{panel.dataset.tournamentId}}:${{panel.dataset.ratingRunId}}`;
  let pollTimer = null;
  let inFlight = false;
  let failureCount = 0;
  const scheduleNext = (delayMs) => {{
    window.clearTimeout(pollTimer);
    pollTimer = window.setTimeout(() => refreshProgress().catch(() => {{}}), delayMs);
  }};
  async function refreshProgress() {{
    if (inFlight) {{
      scheduleNext(10000);
      return;
    }}
    inFlight = true;
    const params = new URLSearchParams({{
      tournament_id: panel.dataset.tournamentId || "",
      rating_run_id: panel.dataset.ratingRunId || "",
    }});
    const pollCount = Number(panel.dataset.pollCount || "0") + 1;
    panel.dataset.pollCount = String(pollCount);
    try {{
      const response = await fetch("/api/rating-progress?" + params.toString(), {{
        cache: "no-store",
      }});
      if (!response.ok) throw new Error(`progress ${{response.status}}`);
      const payload = await response.json();
      const progress = payload.progress || {{}};
      const seen = progress.estimated_seen_game_count ?? progress.completed_game_count ?? 0;
      const fraction = progress.estimated_completion_fraction ?? progress.completion_fraction ?? 0;
      set("status", stateLabel(progress));
      set("phase", text(progress.phase));
      set("pairs", `${{text(progress.started_pair_count ?? 0)}}/${{text(progress.pair_count ?? 0)}}`);
      set("games", `${{text(seen)}}/${{text(progress.game_count ?? 0)}}`);
      set("percent", `${{(100 * number(fraction)).toFixed(1)}}%`);
      set("updated", text(progress.updated_at));
      failureCount = 0;
      if (progress.status === "complete" && panel.dataset.hasRatings !== "true" && !sessionStorage.getItem(reloadKey)) {{
        sessionStorage.setItem(reloadKey, "1");
        const url = new URL(window.location.href);
        url.searchParams.set("fresh", "true");
        window.location.href = url.toString();
        return;
      }}
    }} catch (error) {{
      failureCount += 1;
    }} finally {{
      inFlight = false;
      const hiddenDelay = document.hidden ? 60000 : 10000;
      const retryDelay = Math.min(60000, hiddenDelay * Math.max(1, failureCount));
      scheduleNext(retryDelay);
    }}
  }}
  document.addEventListener("visibilitychange", () => {{
    if (!document.hidden) {{
      scheduleNext(250);
    }}
  }});
  scheduleNext(250);
}})();
</script>
</body>
</html>"""


def _render_battle_page(
    *,
    payload: Mapping[str, Any],
    rating_run_id: str = "latest",
    checkpoint_id: str = "",
) -> str:
    import html

    tournament_id = str(payload.get("selected_tournament_id") or "")
    battle_id = str(payload.get("battle_id") or "")
    back_href = _page_href(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        checkpoint_id=checkpoint_id,
    )
    detail_html = _render_battle_detail_section(payload=payload)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CurvyTron Battle</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #202124; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 18px; }}
    header {{ display: grid; gap: 8px; margin-bottom: 14px; }}
    h1 {{ margin: 0; font-size: 20px; }}
    a {{ color: #1557b0; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .summary {{ margin: 0 0 12px; color: #5f6368; font-size: 13px; }}
    .panel {{ background: white; border: 1px solid #dadce0; border-radius: 8px; margin-bottom: 12px; overflow: hidden; }}
    .panel-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; padding: 10px 12px; border-bottom: 1px solid #eef0f3; }}
    .panel h2 {{ margin: 0; font-size: 15px; }}
    .panel span {{ color: #5f6368; font-size: 12px; }}
    .progress-body {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; background: #eef0f3; }}
    .progress-body div {{ display: grid; gap: 3px; padding: 12px; background: white; }}
    .progress-body strong {{ font-size: 18px; }}
    .progress-body span {{ color: #5f6368; font-size: 12px; }}
    .gif-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; padding: 10px; }}
    .gif-card {{ display: grid; gap: 6px; color: #202124; }}
    .gif-card img {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: #111827; }}
    .gif-card span {{ font-size: 12px; color: #5f6368; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 7px 10px; border-bottom: 1px solid #eef0f3; text-align: left; }}
    th {{ color: #5f6368; font-weight: 600; }}
    td:nth-child(n+3) {{ white-space: nowrap; }}
    .empty {{ padding: 60px; text-align: center; background: white; border: 1px dashed #dadce0; border-radius: 8px; color: #80868b; }}
  </style>
</head>
<body>
<main>
  <header>
    <a href="{html.escape(back_href)}">Back to checkpoint</a>
    <h1>{html.escape(battle_id)}</h1>
  </header>
  {detail_html}
</main>
</body>
</html>"""


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
        limit: int = Query(MAX_LIMIT, ge=1, le=MAX_LIMIT),
        offset: int = Query(0, ge=0),
        fresh: bool = False,
    ) -> HTMLResponse:
        reload_error = _web_reload_volume(
            volume,
            force=bool(fresh),
            min_interval_sec=WEB_PAGE_RELOAD_MIN_INTERVAL_SECONDS,
        )
        tournaments = _list_tournaments(TOURNAMENT_MOUNT)
        selected = _default_tournament_id(tournaments, tournament_id)
        rating_runs = _list_rating_runs(TOURNAMENT_MOUNT, tournament_id=selected) if selected else []
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
        limit: int = Query(MAX_LIMIT, ge=1, le=MAX_LIMIT),
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
    rating_defaults = {
        "round_count": int(payload.get("round_count") or arena.DEFAULT_RATING_ROUND_COUNT),
        "continue_from_latest": bool(payload.get("continue_from_latest", False)),
        "pairs_per_round": (
            int(payload["pairs_per_round"])
            if payload.get("pairs_per_round") not in (None, "", 0, "0")
            else None
        ),
        "placement_min_games": (
            int(payload["placement_min_games"])
            if payload.get("placement_min_games") not in (None, "", 0, "0")
            else None
        ),
        "placement_min_opponents": int(payload.get("placement_min_opponents") or 20),
        "pair_selection": str(
            payload.get("pair_selection") or arena.RATING_PAIR_SELECTION_ADAPTIVE_V0
        ),
        "games_per_pair": int(payload.get("games_per_pair") or arena.DEFAULT_GAMES_PER_PAIR),
        "games_per_shard": int(payload.get("games_per_shard") or arena.DEFAULT_GAMES_PER_PAIR),
        "reuse_policies_per_shard": bool(
            payload.get("reuse_policies_per_shard", arena.DEFAULT_REUSE_POLICIES_PER_SHARD)
        ),
        "seed": int(payload.get("seed") or 0),
        "max_steps": int(payload.get("max_steps") or arena.DEFAULT_MAX_STEPS),
        "decision_ms": float(payload.get("decision_ms") or arena.DEFAULT_DECISION_MS),
        "decision_source_frames": (
            int(payload["decision_source_frames"])
            if payload.get("decision_source_frames") not in (None, "", 0, "0")
            else None
        ),
        "source_physics_step_ms": float(
            payload.get("source_physics_step_ms") or arena.DEFAULT_SOURCE_PHYSICS_STEP_MS
        ),
        "policy_mode": str(payload.get("policy_mode") or arena.POLICY_MODE_EVAL),
        "collect_temperature": float(
            payload.get("collect_temperature") or arena.DEFAULT_COLLECT_TEMPERATURE
        ),
        "collect_epsilon": float(
            payload.get("collect_epsilon") or arena.DEFAULT_COLLECT_EPSILON
        ),
        "policy_trail_render_mode": payload.get("policy_trail_render_mode") or None,
        "num_simulations": int(
            payload.get("num_simulations") or arena.DEFAULT_NUM_SIMULATIONS
        ),
        "save_gif": bool(payload.get("save_gif", False)),
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
        "stop_when_stable": bool(payload.get("stop_when_stable", False)),
    }
    discovery = _discover_checkpoint_refs_from_scan_spec(scan_spec, mount=RUNS_MOUNT)
    existing = checkpoint_intake_state.get(
        _intake_manifest_key(tournament_id, rating_run_id),
        None,
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
    state_write = _put_intake_manifest(manifest)
    manifest_write = _write_intake_manifest_artifact(manifest)
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
            "state_write": state_write,
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
        manifest = checkpoint_intake_state.get(key, None)
        if not isinstance(manifest, Mapping):
            continue
        scan_spec = manifest.get("scan_spec")
        if not isinstance(scan_spec, Mapping):
            continue
        discovery = _discover_checkpoint_refs_from_scan_spec(scan_spec, mount=RUNS_MOUNT)
        current_refs = [
            str(ref)
            for ref in discovery.get("checkpoint_refs", [])
            if str(ref).strip()
        ]
        queued_refs = _clean_ref_set(manifest.get("queued_checkpoint_refs", []))
        new_refs = [ref for ref in current_refs if ref not in queued_refs]
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
        _put_intake_manifest(updated_manifest)
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
    if payload.get("tournament_id"):
        key = _intake_manifest_key(
            str(payload["tournament_id"]),
            str(payload.get("rating_run_id") or arena.DEFAULT_RATING_RUN_ID),
        )
        manifest = checkpoint_intake_state.get(key, None)
    active_keys = checkpoint_intake_state.get(CHECKPOINT_INTAKE_ACTIVE_KEYS, []) or []
    if not isinstance(active_keys, list):
        active_keys = []
    queue_len = None
    if manifest and isinstance(manifest, Mapping):
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
    manifest = checkpoint_intake_state.get(manifest_key, None)
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
    allow_existing_rating = bool(payload.get("spawn_if_existing", False))
    queue_len_before = checkpoint_intake_queue.len(partition=partition)
    spawn_requested = bool(payload.get("spawn_rating", False))
    claim_key = f"rating_claim:{manifest_key}"
    rating_claimed = False
    spawn_skipped_reason = ""
    if spawn_requested:
        if len(manifest.get("checkpoint_refs") or []) < 2:
            spawn_skipped_reason = "needs_at_least_two_checkpoints"
        elif existing_rating_run and not allow_existing_rating:
            spawn_skipped_reason = "rating_run_already_exists"
        elif not queue_len_before and not bool(payload.get("spawn_if_empty", False)):
            spawn_skipped_reason = "no_queued_events"
        else:
            rating_claimed = checkpoint_intake_state.put(
                claim_key,
                {
                    "schema_id": "curvyzero_curvytron_checkpoint_intake_rating_claim/v0",
                    "manifest_key": manifest_key,
                    "tournament_id": tournament_id,
                    "rating_run_id": rating_run_id,
                    "queue_len_before": queue_len_before,
                    "created_at": runs.utc_timestamp(),
                },
                skip_if_exists=True,
            )
            if not rating_claimed:
                spawn_skipped_reason = "rating_run_claim_exists"
    events = []
    if max_events and (not spawn_requested or rating_claimed):
        events = checkpoint_intake_queue.get_many(
            max_events,
            block=False,
            partition=partition,
        ) or []
    should_spawn = (
        spawn_requested
        and rating_claimed
        and (bool(events) or bool(payload.get("spawn_if_empty", False)))
    )
    if spawn_requested and rating_claimed and not should_spawn and not spawn_skipped_reason:
        spawn_skipped_reason = "no_drained_events"
    if rating_claimed:
        checkpoint_intake_state.put(
            claim_key,
            {
                "schema_id": "curvyzero_curvytron_checkpoint_intake_rating_claim/v0",
                "manifest_key": manifest_key,
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "queue_len_before": queue_len_before,
                "event_count": len(events),
                "created_at": runs.utc_timestamp(),
            },
        )
    if should_spawn:
        rating_spec = _intake_rating_spec_from_manifest(
            manifest,
            overrides={
                key: value
                for key, value in payload.items()
                if key
                in {
                    "round_count",
                    "pairs_per_round",
                    "placement_min_games",
                    "placement_min_opponents",
                    "pair_selection",
                    "games_per_pair",
                    "games_per_shard",
                    "reuse_policies_per_shard",
                    "seed",
                    "max_steps",
                    "decision_ms",
                    "decision_source_frames",
                    "source_physics_step_ms",
                    "policy_mode",
                    "collect_temperature",
                    "collect_epsilon",
                    "policy_trail_render_mode",
                    "num_simulations",
                    "save_gif",
                    "gif_sample_games_per_pair",
                    "gif_sample_strategy",
                    "initial_rating",
                    "stop_when_stable",
                }
            },
        )
        call = curvytron_rating_loop.spawn(rating_spec)
        rating_call_id = getattr(call, "object_id", None) or getattr(call, "id", None) or ""
    return arena._to_plain(
        {
            "schema_id": "curvyzero_curvytron_checkpoint_intake_drain/v0",
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "manifest_key": manifest_key,
            "queue_partition": manifest["queue_partition"],
            "queue_len_before": queue_len_before,
            "event_count": len(events),
            "events": events,
            "checkpoint_count": len(manifest.get("checkpoint_refs") or []),
            "existing_rating_run": existing_rating_run,
            "rating_claim_key": claim_key,
            "rating_claimed": rating_claimed,
            "spawn_skipped_reason": spawn_skipped_reason,
            "rating_call_id": rating_call_id,
        }
    )


@app.function(
    image=image,
    volumes=_tournament_volumes(),
    schedule=modal.Period(seconds=DEFAULT_CHECKPOINT_INTAKE_SCAN_SECONDS),
    timeout=10 * 60,
    cpu=1.0,
    memory=1024,
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
        result = curvytron_checkpoint_intake_drain.local(
            {
                "tournament_id": str(manifest["tournament_id"]),
                "rating_run_id": str(manifest["rating_run_id"]),
                "max_events": max(100, len(manifest.get("checkpoint_refs") or [])),
                "spawn_rating": True,
            }
        )
        if int(result.get("event_count") or 0) or result.get("rating_call_id"):
            results.append(result)
    return arena._to_plain(
        {
            "schema_id": "curvyzero_curvytron_checkpoint_intake_drain_tick/v0",
            "active_manifest_count": len(active_keys),
            "drained_manifest_count": len(results),
            "event_count": sum(int(row.get("event_count") or 0) for row in results),
            "rating_call_ids": [
                row.get("rating_call_id") for row in results if row.get("rating_call_id")
            ],
            "results": results,
        }
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
    stop_when_stable: bool = False,
    seed: int = 0,
    max_steps: int = arena.DEFAULT_MAX_STEPS,
    decision_ms: float = arena.DEFAULT_DECISION_MS,
    decision_source_frames: int = 0,
    source_physics_step_ms: float = arena.DEFAULT_SOURCE_PHYSICS_STEP_MS,
    policy_mode: str = arena.POLICY_MODE_EVAL,
    collect_temperature: float = arena.DEFAULT_COLLECT_TEMPERATURE,
    collect_epsilon: float = arena.DEFAULT_COLLECT_EPSILON,
    policy_trail_render_mode: str = "",
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
    intake_enqueue_existing: bool = False,
    intake_spawn_rating: bool = False,
    intake_max_events: int = 100,
    intake_active: bool = True,
    wait: bool = False,
) -> None:
    mode = str(mode)
    refs = arena.parse_checkpoint_refs(checkpoint_refs)
    discovery = None
    intake_modes = {"intake-seed", "intake-tick", "intake-drain", "intake-status"}
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
    resolved_tournament_id = tournament_id or runs.new_run_id("arena")
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
        "intake-seed",
        "intake-tick",
        "intake-drain",
        "intake-status",
    }:
        raise ValueError(
            "mode must be one of: discover, estimate, game, pair, tournament, rating, "
            "progress, provisional, provisional-loop, reduce, visibility, "
            "intake-seed, intake-tick, intake-drain, intake-status"
        )
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
    if mode in {"intake-seed", "intake-tick", "intake-drain", "intake-status"}:
        intake_spec = {
            "tournament_id": resolved_tournament_id,
            "rating_run_id": rating_run_id,
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
            "stop_when_stable": bool(stop_when_stable),
            "games_per_pair": int(games_per_pair),
            "games_per_shard": int(games_per_shard),
            "reuse_policies_per_shard": bool(reuse_policies_per_shard),
            "seed": int(seed),
            "max_steps": int(max_steps),
            "decision_ms": float(decision_ms),
            "decision_source_frames": (
                int(decision_source_frames) if int(decision_source_frames) > 0 else None
            ),
            "source_physics_step_ms": float(source_physics_step_ms),
            "policy_mode": policy_mode,
            "collect_temperature": float(collect_temperature),
            "collect_epsilon": float(collect_epsilon),
            "policy_trail_render_mode": policy_trail_render_mode or None,
            "num_simulations": int(num_simulations),
            "save_gif": bool(save_gif),
            "gif_sample_games_per_pair": int(gif_sample_games_per_pair),
            "gif_sample_strategy": str(gif_sample_strategy),
            "enqueue_existing": bool(intake_enqueue_existing),
            "spawn_rating": bool(intake_spawn_rating),
            "max_events": int(intake_max_events),
            "active": bool(intake_active),
        }
        if mode == "intake-seed":
            result = curvytron_checkpoint_intake_seed.remote(intake_spec)
        elif mode == "intake-tick":
            result = curvytron_checkpoint_intake_tick.remote(
                intake_spec if tournament_id else {}
            )
        elif mode == "intake-drain":
            result = curvytron_checkpoint_intake_drain.remote(intake_spec)
        else:
            result = curvytron_checkpoint_intake_status.remote(
                intake_spec if tournament_id else {}
            )
        print(json.dumps(arena._to_plain(result), indent=2, sort_keys=True))
        return
    if mode in {"progress", "provisional", "provisional-loop", "reduce"}:
        if not tournament_id:
            raise ValueError(f"{mode} mode needs --tournament-id")
        progress_spec = {
            "tournament_id": resolved_tournament_id,
            "rating_run_id": rating_run_id,
            "round_index": int(round_index),
            "round_id": arena.rating_round_id(int(round_index)),
            "load_summaries": bool(progress_read_summaries),
        }
        if mode == "progress":
            result = curvytron_rating_progress.remote(progress_spec)
            print(json.dumps(arena._to_plain(result), indent=2, sort_keys=True))
            return
        if mode == "provisional":
            call = curvytron_rating_provisional.spawn(progress_spec)
            call_id = getattr(call, "object_id", None) or getattr(call, "id", None)
            payload: dict[str, Any] = {
                "status": "spawned",
                "app_name": APP_NAME,
                "mode": mode,
                "tournament_id": resolved_tournament_id,
                "rating_run_id": rating_run_id,
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
                "rating_run_id": rating_run_id,
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
            "rating_run_id": rating_run_id,
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

    common = {
        "games_per_pair": int(games_per_pair),
        "games_per_shard": int(games_per_shard),
        "reuse_policies_per_shard": bool(reuse_policies_per_shard),
        "seed": int(seed),
        "max_steps": int(max_steps),
        "decision_ms": float(decision_ms),
        "decision_source_frames": (
            int(decision_source_frames) if int(decision_source_frames) > 0 else None
        ),
        "source_physics_step_ms": float(source_physics_step_ms),
        "policy_mode": policy_mode,
        "collect_temperature": float(collect_temperature),
        "collect_epsilon": float(collect_epsilon),
        "policy_trail_render_mode": policy_trail_render_mode or None,
        "num_simulations": int(num_simulations),
        "save_gif": bool(save_gif),
        "gif_sample_games_per_pair": int(gif_sample_games_per_pair),
        "gif_sample_strategy": str(gif_sample_strategy),
    }
    if mode == "rating":
        spec = {
            "tournament_id": resolved_tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": refs,
            "round_count": int(round_count),
            "continue_from_latest": bool(continue_from_latest),
            "pairs_per_round": int(pairs_per_round) if int(pairs_per_round) > 0 else None,
            "placement_min_games": (
                int(placement_min_games) if int(placement_min_games) > 0 else None
            ),
            "placement_min_opponents": int(placement_min_opponents),
            "pair_selection": pair_selection,
            "initial_rating": float(initial_rating),
            "stop_when_stable": bool(stop_when_stable),
            **common,
        }
        call = curvytron_rating_loop.spawn(spec)
    elif mode == "tournament":
        spec = {"tournament_id": resolved_tournament_id, "checkpoints": refs, **common}
        call = curvytron_tournament_run.spawn(spec)
    else:
        pair = arena.build_pair_specs(
            checkpoints=refs,
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
