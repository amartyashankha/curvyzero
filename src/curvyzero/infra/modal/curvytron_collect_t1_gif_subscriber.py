"""Scheduled sidecar that backfills CurvyTron collect-mode GIFs.

This is intentionally separate from the training jobs. It watches the fixed
overnight40 run set, notices checkpoints that do not have ``collect_t1.gif``,
and spawns the current GIF worker code for those checkpoints.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import io
import json
from pathlib import Path, PurePosixPath
import time
from typing import Any

import modal

from curvyzero.infra.modal import (
    lightzero_curvyzero_stacked_debug_visual_survival_train as train,
)
from curvyzero.infra.modal import run_management as runs


APP_NAME = "curvyzero-curvytron-collect-t1-gif-subscriber"
SUBSCRIBER_ID = "overnight40a-20260512"
SCHEDULE_MINUTES = 5
DEFAULT_SPAWN_LIMIT = 80
DEFAULT_RETRY_AFTER_SEC = 45 * 60
DEFAULT_MAX_CHECKPOINTS_PER_RUN = 0

SIDECAR_ROOT_REF = (
    PurePosixPath("sidecars")
    / train.TASK_ID
    / "collect-t1-gif-subscriber"
    / SUBSCRIBER_ID
)
MANIFEST_REF = SIDECAR_ROOT_REF / "manifest.json"
LATEST_TICK_REF = SIDECAR_ROOT_REF / "latest_tick.json"
TICKS_REF = SIDECAR_ROOT_REF / "ticks"

app = modal.App(APP_NAME)


@dataclass(frozen=True)
class RunSpec:
    row: int
    run_id: str
    attempt_id: str
    seed: int


OVERNIGHT40_RUNS: tuple[RunSpec, ...] = (
    RunSpec(1, "curvy2seat-selfplay-overnight40a-01-main-fast-gray64-direct-b64-sim8", "curvy2seat-selfplay-overnight40a-01-main-fast-gray64-direct-b64-sim8-20260512", 1201),
    RunSpec(2, "curvy2seat-selfplay-overnight40a-02-main-seed-fast-gray64-direct-b64-sim8", "curvy2seat-selfplay-overnight40a-02-main-seed-fast-gray64-direct-b64-sim8-20260512", 1202),
    RunSpec(3, "curvy2seat-selfplay-overnight40a-03-main-seed-fast-gray64-direct-b64-sim8", "curvy2seat-selfplay-overnight40a-03-main-seed-fast-gray64-direct-b64-sim8-20260512", 1203),
    RunSpec(4, "curvy2seat-selfplay-overnight40a-04-main-seed-fast-gray64-direct-b64-sim8", "curvy2seat-selfplay-overnight40a-04-main-seed-fast-gray64-direct-b64-sim8-20260512", 1204),
    RunSpec(5, "curvy2seat-selfplay-overnight40a-05-main-seed-fast-gray64-direct-b64-sim8", "curvy2seat-selfplay-overnight40a-05-main-seed-fast-gray64-direct-b64-sim8-20260512", 1205),
    RunSpec(6, "curvy2seat-selfplay-overnight40a-06-search16-fast-gray64-direct-b64-sim16", "curvy2seat-selfplay-overnight40a-06-search16-fast-gray64-direct-b64-sim16-20260512", 1206),
    RunSpec(7, "curvy2seat-selfplay-overnight40a-07-search32-fast-gray64-direct-b64-sim32", "curvy2seat-selfplay-overnight40a-07-search32-fast-gray64-direct-b64-sim32-20260512", 1207),
    RunSpec(8, "curvy2seat-selfplay-overnight40a-08-small-batch-fast-gray64-direct-b32-sim8", "curvy2seat-selfplay-overnight40a-08-small-batch-fast-gray64-direct-b32-sim8-20260512", 1208),
    RunSpec(9, "curvy2seat-selfplay-overnight40a-09-large-batch-l4-fast-gray64-direct-b128-sim8", "curvy2seat-selfplay-overnight40a-09-large-batch-l4-fast-gray64-direct-b128-sim8-20260512", 1209),
    RunSpec(10, "curvy2seat-selfplay-overnight40a-10-large-search-l4-fast-gray64-direct-b128-sim16", "curvy2seat-selfplay-overnight40a-10-large-search-l4-fast-gray64-direct-b128-sim16-20260512", 1210),
    RunSpec(11, "curvy2seat-selfplay-overnight40a-11-collect128-fast-gray64-direct-b64-sim8", "curvy2seat-selfplay-overnight40a-11-collect128-fast-gray64-direct-b64-sim8-20260512", 1211),
    RunSpec(12, "curvy2seat-selfplay-overnight40a-12-collect256-fast-gray64-direct-b64-sim8", "curvy2seat-selfplay-overnight40a-12-collect256-fast-gray64-direct-b64-sim8-20260512", 1212),
    RunSpec(13, "curvy2seat-selfplay-overnight40a-13-updates8-fast-gray64-direct-b64-sim8", "curvy2seat-selfplay-overnight40a-13-updates8-fast-gray64-direct-b64-sim8-20260512", 1213),
    RunSpec(14, "curvy2seat-selfplay-overnight40a-14-updates16-fast-gray64-direct-b64-sim8", "curvy2seat-selfplay-overnight40a-14-updates16-fast-gray64-direct-b64-sim8-20260512", 1214),
    RunSpec(15, "curvy2seat-selfplay-overnight40a-15-learner512-fast-gray64-direct-b64-sim8", "curvy2seat-selfplay-overnight40a-15-learner512-fast-gray64-direct-b64-sim8-20260512", 1215),
    RunSpec(16, "curvy2seat-selfplay-overnight40a-16-learner1024-fast-gray64-direct-b128-sim8", "curvy2seat-selfplay-overnight40a-16-learner1024-fast-gray64-direct-b128-sim8-20260512", 1216),
    RunSpec(17, "curvy2seat-selfplay-overnight40a-17-lr-3e-5-fast-gray64-direct-b64-sim8-lr3e-5", "curvy2seat-selfplay-overnight40a-17-lr-3e-5-fast-gray64-direct-b64-sim8-lr3e-5-20260512", 1217),
    RunSpec(18, "curvy2seat-selfplay-overnight40a-18-lr-1e-4-fast-gray64-direct-b64-sim8-lr1e-4", "curvy2seat-selfplay-overnight40a-18-lr-1e-4-fast-gray64-direct-b64-sim8-lr1e-4-20260512", 1218),
    RunSpec(19, "curvy2seat-selfplay-overnight40a-19-lr-3e-4-fast-gray64-direct-b64-sim8-lr3e-4", "curvy2seat-selfplay-overnight40a-19-lr-3e-4-fast-gray64-direct-b64-sim8-lr3e-4-20260512", 1219),
    RunSpec(20, "curvy2seat-selfplay-overnight40a-20-lr-1e-3-fast-gray64-direct-b64-sim8-lr1e-3", "curvy2seat-selfplay-overnight40a-20-lr-1e-3-fast-gray64-direct-b64-sim8-lr1e-3-20260512", 1220),
    RunSpec(21, "curvy2seat-selfplay-overnight40a-21-lr-3e-3-fast-gray64-direct-b64-sim8-lr3e-3", "curvy2seat-selfplay-overnight40a-21-lr-3e-3-fast-gray64-direct-b64-sim8-lr3e-3-20260512", 1221),
    RunSpec(22, "curvy2seat-selfplay-overnight40a-22-no-bonus-fast-gray64-direct-b64-sim8-no-bonus", "curvy2seat-selfplay-overnight40a-22-no-bonus-fast-gray64-direct-b64-sim8-no-bonus-20260512", 1222),
    RunSpec(23, "curvy2seat-overnight40a-23-terminal-only-fast-b64-sim8-terminal-only", "overnight40a-23-terminal-only-20260512", 1223),
    RunSpec(24, "curvy2seat-overnight40a-24-stronger-terminal-fast-b64-sim8-terminal-x2", "overnight40a-24-stronger-terminal-20260512", 1224),
    RunSpec(25, "curvy2seat-overnight40a-25-survival-only-ctrl-fast-b64-sim8-survival-only", "overnight40a-25-survival-only-ctrl-20260512", 1225),
    RunSpec(26, "curvy2seat-overnight40a-26-bonus-heavy-fast-b64-sim8-bonus-x2", "overnight40a-26-bonus-heavy-20260512", 1226),
    RunSpec(27, "curvy2seat-overnight40a-27-no-obs-noise-fast-b64-sim8-obs-noise-0", "overnight40a-27-no-obs-noise-20260512", 1227),
    RunSpec(28, "curvy2seat-overnight40a-28-obs-noise-05-fast-b64-sim8-obs-noise-05", "overnight40a-28-obs-noise-05-20260512", 1228),
    RunSpec(29, "curvy2seat-overnight40a-29-obs-noise-20-fast-b64-sim8-obs-noise-20", "overnight40a-29-obs-noise-20-20260512", 1229),
    RunSpec(30, "curvy2seat-overnight40a-30-action-repeat-fast-b64-sim8-repeat-20pct", "overnight40a-30-action-repeat-20260512", 1230),
    RunSpec(31, "curvy2seat-overnight40a-31-action-noop-05-fast-b64-sim8-action-noop-5pct", "overnight40a-31-action-noop-20260512", 1231),
    RunSpec(32, "curvy2seat-overnight40a-32-no-stochasticity-fast-b64-sim8-none", "overnight40a-32-no-stochasticity-20260512", 1232),
    RunSpec(33, "curvy2seat-overnight40a-33-large-batch-h100-fast-b128-sim8", "overnight40a-33-large-batch-h100-20260512", 1233),
    RunSpec(34, "curvy2seat-overnight40a-34-large-search-h100-fast-b128-sim16", "overnight40a-34-large-search-h100-20260512", 1234),
    RunSpec(35, "curvy2seat-overnight40a-35-h100-search32-fast-b128-sim32", "overnight40a-35-h100-search32-20260512", 1235),
    RunSpec(36, "curvy2seat-overnight40a-36-h100-collect128-fast-b128-sim16", "overnight40a-36-h100-collect128-20260512", 1236),
    RunSpec(37, "curvy2seat-overnight40a-37-h100-b256-fast-b256-sim8", "overnight40a-37-h100-b256-20260512", 1237),
    RunSpec(38, "curvy2seat-overnight40a-38-h100-lr-3e-4-fast-b128-sim8-lr3e-4", "overnight40a-38-h100-lr-3e-4-20260512", 1238),
    RunSpec(39, "curvy2seat-overnight40a-39-browser-sentinel-browser-b16-sim8", "overnight40a-39-browser-sentinel-20260512", 1239),
    RunSpec(40, "curvy2seat-overnight40a-40-browser-sentinel-lr-browser-b16-sim8-lr3e-4", "overnight40a-40-browser-sentinel-lr-20260512", 1240),
)


def _to_plain(value: Any) -> Any:
    return train._to_plain(value)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_path() -> Path:
    return runs.volume_path(train.RUNS_MOUNT, MANIFEST_REF)


def _load_manifest() -> dict[str, Any]:
    manifest = _read_json(_manifest_path())
    if manifest is None:
        return {
            "schema_id": "curvyzero_collect_t1_gif_subscriber_manifest/v0",
            "subscriber_id": SUBSCRIBER_ID,
            "app_name": APP_NAME,
            "created_at": runs.utc_timestamp(),
            "items": {},
        }
    manifest.setdefault("items", {})
    if not isinstance(manifest["items"], dict):
        manifest["items"] = {}
    return manifest


def _write_json_ref(ref: PurePosixPath, payload: dict[str, Any]) -> dict[str, Any]:
    path = runs.volume_path(train.RUNS_MOUNT, ref)
    runs.write_json(path, _to_plain(payload))
    return runs.file_summary(path, mount=train.RUNS_MOUNT)


def _checkpoint_root(spec: RunSpec) -> Path:
    ref = runs.checkpoints_root_ref(train.TASK_ID, spec.run_id) / "lightzero"
    return runs.volume_path(train.RUNS_MOUNT, ref)


def _selfplay_root_ref(spec: RunSpec, eval_id: str) -> PurePosixPath:
    return runs.attempt_eval_ref(
        train.TASK_ID,
        spec.run_id,
        spec.attempt_id,
        eval_id,
    ) / "selfplay"


def _candidate_key(spec: RunSpec, checkpoint_ref: str, eval_id: str) -> str:
    return f"{spec.run_id}|{spec.attempt_id}|{checkpoint_ref}|{eval_id}"


def _existing_summary_settings(summary_path: Path) -> dict[str, Any]:
    summary = _read_json(summary_path) or {}
    max_steps = summary.get("max_steps")
    return {
        "seed": summary.get("seed"),
        "max_steps": 0 if max_steps is None else max_steps,
        "source_max_steps": summary.get("configured_source_max_steps"),
        "num_simulations": summary.get("num_simulations"),
        "batch_size": summary.get("batch_size"),
        "frame_stride": summary.get("frame_stride"),
        "fps": summary.get("fps"),
        "frame_size": summary.get("frame_size"),
        "training_env_variant": summary.get("training_env_variant"),
        "training_reward_variant": summary.get("training_reward_variant"),
        "natural_bonus_spawn": summary.get("natural_bonus_spawn"),
    }


def _effective_seed(spec: RunSpec, *, checkpoint_ref: str, checkpoint_label: str, eval_id: str) -> int:
    base_seed = int(spec.seed) + int(train.DEFAULT_BACKGROUND_GIF_SEED_OFFSET)
    seed_mix = train._stable_seed_mix(checkpoint_ref, checkpoint_label, eval_id)
    return train._mix_seed(base_seed, seed_mix)


def _worker_args_for_candidate(
    spec: RunSpec,
    *,
    checkpoint_ref: str,
    checkpoint_label: str,
    eval_id: str,
    summary_path: Path,
) -> dict[str, Any]:
    settings = _existing_summary_settings(summary_path)

    def setting(name: str, default: Any) -> Any:
        value = settings.get(name)
        return default if value is None else value

    return {
        "checkpoint_ref": checkpoint_ref,
        "checkpoint_label": checkpoint_label,
        "eval_id": eval_id,
        "run_id": spec.run_id,
        "attempt_id": spec.attempt_id,
        "seed": int(
            setting(
                "seed",
                _effective_seed(
                    spec,
                    checkpoint_ref=checkpoint_ref,
                    checkpoint_label=checkpoint_label,
                    eval_id=eval_id,
                ),
            )
        ),
        "max_steps": int(setting("max_steps", train.DEFAULT_BACKGROUND_GIF_MAX_STEPS)),
        "source_max_steps": int(
            setting("source_max_steps", train.DEFAULT_BACKGROUND_EVAL_MAX_STEPS)
        ),
        "num_simulations": int(
            setting("num_simulations", train.DEFAULT_BACKGROUND_EVAL_NUM_SIMULATIONS)
        ),
        "batch_size": int(setting("batch_size", train.DEFAULT_BACKGROUND_EVAL_BATCH_SIZE)),
        "frame_stride": int(
            setting("frame_stride", train.DEFAULT_BACKGROUND_GIF_FRAME_STRIDE)
        ),
        "fps": float(setting("fps", train.DEFAULT_BACKGROUND_GIF_FPS)),
        "scale": int(train.DEFAULT_BACKGROUND_GIF_SCALE),
        "frame_size": int(setting("frame_size", train.DEFAULT_BACKGROUND_GIF_FRAME_SIZE)),
        "training_env_variant": str(
            setting("training_env_variant", train.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT)
        ),
        "training_reward_variant": str(
            setting("training_reward_variant", train.REWARD_VARIANT_SPARSE_OUTCOME)
        ),
        "natural_bonus_spawn": bool(
            setting("natural_bonus_spawn", train.TWO_SEAT_DEFAULT_NATURAL_BONUS_SPAWN)
        ),
        "collect_temperature": float(train.DEFAULT_BACKGROUND_GIF_COLLECT_TEMPERATURE),
        "collect_epsilon": float(train.DEFAULT_BACKGROUND_GIF_COLLECT_EPSILON),
    }


def _checkpoint_candidates(
    spec: RunSpec,
    *,
    max_checkpoints_per_run: int,
) -> list[dict[str, Any]]:
    root = _checkpoint_root(spec)
    if not root.exists():
        return [
            {
                "row": spec.row,
                "run_id": spec.run_id,
                "attempt_id": spec.attempt_id,
                "status": "checkpoint_root_missing",
                "checkpoint_root": str(root),
            }
        ]
    checkpoints = [
        path
        for path in root.iterdir()
        if path.is_file() and train._live_eval_checkpoint_name(path.name)
    ]
    checkpoints.sort(key=lambda path: train._checkpoint_ref_sort_key(path.name))
    if max_checkpoints_per_run > 0:
        checkpoints = checkpoints[-int(max_checkpoints_per_run) :]

    candidates: list[dict[str, Any]] = []
    for index, path in enumerate(checkpoints):
        checkpoint_ref = runs.file_ref(path.resolve(), mount=train.RUNS_MOUNT.resolve())
        checkpoint_label = train._checkpoint_label_from_ref(checkpoint_ref, index=index)
        eval_id = train._safe_generated_id(
            f"{train.DEFAULT_BACKGROUND_EVAL_ID_PREFIX}_{checkpoint_label}",
            fallback="live_checkpoint",
        )
        selfplay_ref = _selfplay_root_ref(spec, eval_id)
        collect_ref = (
            selfplay_ref
            / train.BACKGROUND_GIF_VARIANT_FILENAMES[
                train.BACKGROUND_GIF_VARIANT_COLLECT_T1
            ]
        )
        raw_ref = (
            selfplay_ref
            / train.BACKGROUND_GIF_VARIANT_FILENAMES[
                train.BACKGROUND_GIF_VARIANT_EVAL_GREEDY
            ]
        )
        summary_ref = selfplay_ref / "summary.json"
        collect_path = runs.volume_path(train.RUNS_MOUNT, collect_ref)
        summary_path = runs.volume_path(train.RUNS_MOUNT, summary_ref)
        stat = path.stat()
        candidates.append(
            {
                "row": spec.row,
                "run_id": spec.run_id,
                "attempt_id": spec.attempt_id,
                "seed": spec.seed,
                "checkpoint_ref": checkpoint_ref,
                "checkpoint_label": checkpoint_label,
                "checkpoint_size_bytes": stat.st_size,
                "checkpoint_mtime_ns": stat.st_mtime_ns,
                "eval_id": eval_id,
                "selfplay_ref": selfplay_ref.as_posix(),
                "summary_ref": summary_ref.as_posix(),
                "raw_gif_ref": raw_ref.as_posix(),
                "collect_gif_ref": collect_ref.as_posix(),
                "collect_gif_exists": collect_path.exists(),
                "summary_exists": summary_path.exists(),
                "key": _candidate_key(spec, checkpoint_ref, eval_id),
                "worker_args": _worker_args_for_candidate(
                    spec,
                    checkpoint_ref=checkpoint_ref,
                    checkpoint_label=checkpoint_label,
                    eval_id=eval_id,
                    summary_path=summary_path,
                ),
            }
        )
    return candidates


def _should_retry(record: dict[str, Any], *, now: float, retry_after_sec: float) -> bool:
    if record.get("status") not in {"spawned", "pending"}:
        return True
    scheduled_at_epoch = record.get("scheduled_at_epoch")
    if not isinstance(scheduled_at_epoch, (int, float)):
        return True
    return now - float(scheduled_at_epoch) >= float(retry_after_sec)


def _run_tick(
    *,
    dry_run: bool,
    spawn_limit: int,
    retry_after_sec: float,
    max_checkpoints_per_run: int,
) -> dict[str, Any]:
    if spawn_limit < 0:
        raise ValueError("spawn_limit must be non-negative")
    if retry_after_sec < 0:
        raise ValueError("retry_after_sec must be non-negative")
    if max_checkpoints_per_run < 0:
        raise ValueError("max_checkpoints_per_run must be non-negative")

    if hasattr(train.runs_volume, "reload"):
        train.runs_volume.reload()

    started_at = runs.utc_timestamp()
    now = time.time()
    manifest = _load_manifest()
    items = manifest["items"]
    scanned = 0
    already_done = 0
    missing_roots: list[dict[str, Any]] = []
    skipped_pending: list[dict[str, Any]] = []
    would_spawn: list[dict[str, Any]] = []
    spawned: list[dict[str, Any]] = []
    spawn_failed: list[dict[str, Any]] = []

    for spec in OVERNIGHT40_RUNS:
        for candidate in _checkpoint_candidates(
            spec,
            max_checkpoints_per_run=max_checkpoints_per_run,
        ):
            if candidate.get("status") == "checkpoint_root_missing":
                missing_roots.append(candidate)
                continue
            scanned += 1
            key = str(candidate["key"])
            existing = items.get(key)
            existing = existing if isinstance(existing, dict) else {}
            if candidate["collect_gif_exists"]:
                already_done += 1
                items[key] = {
                    **existing,
                    **{
                        "status": "done",
                        "done_at": existing.get("done_at") or runs.utc_timestamp(),
                        "run_id": candidate["run_id"],
                        "attempt_id": candidate["attempt_id"],
                        "checkpoint_ref": candidate["checkpoint_ref"],
                        "checkpoint_label": candidate["checkpoint_label"],
                        "eval_id": candidate["eval_id"],
                        "collect_gif_ref": candidate["collect_gif_ref"],
                    },
                }
                continue
            if existing and not _should_retry(existing, now=now, retry_after_sec=retry_after_sec):
                skipped_pending.append(
                    {
                        "run_id": candidate["run_id"],
                        "attempt_id": candidate["attempt_id"],
                        "checkpoint_label": candidate["checkpoint_label"],
                        "eval_id": candidate["eval_id"],
                        "status": existing.get("status"),
                    }
                )
                continue
            if len(spawned) + len(would_spawn) >= spawn_limit:
                continue

            record = {
                "status": "would_spawn" if dry_run else "pending_spawn",
                "run_id": candidate["run_id"],
                "attempt_id": candidate["attempt_id"],
                "checkpoint_ref": candidate["checkpoint_ref"],
                "checkpoint_label": candidate["checkpoint_label"],
                "checkpoint_size_bytes": candidate["checkpoint_size_bytes"],
                "checkpoint_mtime_ns": candidate["checkpoint_mtime_ns"],
                "eval_id": candidate["eval_id"],
                "summary_ref": candidate["summary_ref"],
                "raw_gif_ref": candidate["raw_gif_ref"],
                "collect_gif_ref": candidate["collect_gif_ref"],
                "summary_exists_before_spawn": candidate["summary_exists"],
                "attempts": int(existing.get("attempts", 0)) + int(not dry_run),
                "last_seen_at": started_at,
            }
            if dry_run:
                would_spawn.append(record)
                continue

            try:
                call = collect_t1_gif_subscriber_worker.spawn(**candidate["worker_args"])
                record.update(
                    {
                        "status": "spawned",
                        "scheduled_at": runs.utc_timestamp(),
                        "scheduled_at_epoch": now,
                        "function_call_id": getattr(call, "object_id", None)
                        or getattr(call, "id", None),
                    }
                )
                spawned.append(record)
            except Exception as exc:  # pragma: no cover - Modal scheduling resilience.
                record.update(
                    {
                        "status": "spawn_failed",
                        "scheduled_at": runs.utc_timestamp(),
                        "error": train._exception_result(exc),
                    }
                )
                spawn_failed.append(record)
            items[key] = record

    manifest.update(
        {
            "schema_id": "curvyzero_collect_t1_gif_subscriber_manifest/v0",
            "subscriber_id": SUBSCRIBER_ID,
            "app_name": APP_NAME,
            "updated_at": runs.utc_timestamp(),
            "schedule_minutes": SCHEDULE_MINUTES,
            "run_count": len(OVERNIGHT40_RUNS),
            "item_count": len(items),
        }
    )
    tick = {
        "schema_id": "curvyzero_collect_t1_gif_subscriber_tick/v0",
        "subscriber_id": SUBSCRIBER_ID,
        "app_name": APP_NAME,
        "dry_run": bool(dry_run),
        "started_at": started_at,
        "ended_at": runs.utc_timestamp(),
        "spawn_limit": int(spawn_limit),
        "retry_after_sec": float(retry_after_sec),
        "max_checkpoints_per_run": int(max_checkpoints_per_run),
        "run_count": len(OVERNIGHT40_RUNS),
        "checkpoint_count": scanned,
        "already_done_count": already_done,
        "missing_root_count": len(missing_roots),
        "skipped_pending_count": len(skipped_pending),
        "would_spawn_count": len(would_spawn),
        "spawned_count": len(spawned),
        "spawn_failed_count": len(spawn_failed),
        "manifest_ref": MANIFEST_REF.as_posix(),
        "missing_roots": missing_roots[:20],
        "skipped_pending": skipped_pending[:20],
        "would_spawn": would_spawn[:20],
        "spawned": spawned[:40],
        "spawn_failed": spawn_failed[:20],
    }
    if not dry_run:
        _write_json_ref(MANIFEST_REF, manifest)
    tick_ref = TICKS_REF / f"{runs.utc_stamp()}.json"
    tick["tick_ref"] = tick_ref.as_posix()
    if not dry_run:
        _write_json_ref(tick_ref, tick)
        _write_json_ref(LATEST_TICK_REF, tick)
        if hasattr(train.runs_volume, "commit"):
            train.runs_volume.commit()
    print(json.dumps(_to_plain(tick), indent=2, sort_keys=True))
    return _to_plain(tick)


@app.function(
    image=train.image,
    volumes={str(train.RUNS_MOUNT): train.runs_volume},
    timeout=40 * 60,
    cpu=2.0,
)
def collect_t1_gif_subscriber_worker(**kwargs: Any) -> dict[str, Any]:
    # The shared GIF helper prints a full summary, including action traces.
    # In a fan-out sidecar that makes Modal logs unreadable; the same summary is
    # written to the Volume, so keep the worker log quiet.
    with contextlib.redirect_stdout(io.StringIO()):
        return train._run_checkpoint_selfplay_gif(**kwargs)


@app.function(
    image=train.image,
    volumes={str(train.RUNS_MOUNT): train.runs_volume},
    schedule=modal.Period(minutes=SCHEDULE_MINUTES),
    timeout=20 * 60,
    cpu=1.0,
)
def collect_t1_gif_subscriber_tick(
    dry_run: bool = False,
    spawn_limit: int = DEFAULT_SPAWN_LIMIT,
    retry_after_sec: float = DEFAULT_RETRY_AFTER_SEC,
    max_checkpoints_per_run: int = DEFAULT_MAX_CHECKPOINTS_PER_RUN,
) -> dict[str, Any]:
    return _run_tick(
        dry_run=bool(dry_run),
        spawn_limit=int(spawn_limit),
        retry_after_sec=float(retry_after_sec),
        max_checkpoints_per_run=int(max_checkpoints_per_run),
    )


@app.local_entrypoint()
def main(
    dry_run: bool = True,
    spawn_limit: int = DEFAULT_SPAWN_LIMIT,
    retry_after_sec: float = DEFAULT_RETRY_AFTER_SEC,
    max_checkpoints_per_run: int = DEFAULT_MAX_CHECKPOINTS_PER_RUN,
) -> None:
    result = collect_t1_gif_subscriber_tick.remote(
        dry_run=bool(dry_run),
        spawn_limit=int(spawn_limit),
        retry_after_sec=float(retry_after_sec),
        max_checkpoints_per_run=int(max_checkpoints_per_run),
    )
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
