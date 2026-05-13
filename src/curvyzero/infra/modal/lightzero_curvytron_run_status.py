"""Compact status reader for CurvyTron LightZero training runs.

This reads Modal Volume progress files from inside one Modal function. It is
meant to replace noisy manual ``modal volume get`` polling.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
    RUNS_MOUNT,
    TASK_ID,
    image,
    runs_volume,
)


APP_NAME = "curvyzero-lightzero-curvytron-run-status"
DEFAULT_COLLAPSE_THRESHOLD = 0.95
EVAL_JOB_KINDS = {
    "lightzero_curvytron_visual_survival_checkpoint_curve_eval",
    "lightzero_curvytron_visual_survival_live_checkpoint_eval",
}

LIVE6_RUN_IDS = (
    "curvytron-two-seat-selfplay-live6-clean-survival001-b32-open-u8-sim4-20260510",
    "curvytron-two-seat-selfplay-live6-deadpenalty-survival001-b32-open-u8-sim4-20260510",
    "curvytron-two-seat-selfplay-live6-deadpenalty-survival002-b32-open-u8-sim4-20260510",
    "curvytron-two-seat-selfplay-live6-deadpenalty-survival005-b32-open-u8-sim4-20260510",
    "curvytron-two-seat-selfplay-live6-noopwarm005-survival001-b32-open-u8-sim4-20260510",
    "curvytron-two-seat-selfplay-live6-noopwarm010-survival001-b32-open-u8-sim4-20260510",
    "curvytron-two-seat-selfplay-live6-noopwarm015-survival001-b32-open-u8-sim4-20260510",
    "curvytron-two-seat-selfplay-live6-imgnoise001-survival001-b32-open-u8-sim4-20260510",
    "curvytron-two-seat-selfplay-live6-imgnoise003-survival001-b32-open-u8-sim4-20260510",
    "curvytron-two-seat-selfplay-live6-imgnoise010-survival001-b32-open-u8-sim4-20260510",
    "curvytron-two-seat-selfplay-live6-mixed-img003-noop010-survival001-b32-open-u8-sim4-20260510",
    "curvytron-two-seat-selfplay-live6-mixed-img010-noop005-survival002-b32-open-u8-sim4-20260510",
)

LIVE11_RUN_IDS = (
    "curvytron-two-seat-selfplay-live11-base-u1-temp10-eps025-sim5-20260511",
    "curvytron-two-seat-selfplay-live11-base-u2-temp10-eps025-sim5-20260511",
    "curvytron-two-seat-selfplay-live11-explore-u1-temp15-eps025-sim5-20260511",
    "curvytron-two-seat-selfplay-live11-explore-u1-temp20-eps025-sim5-20260511",
    "curvytron-two-seat-selfplay-live11-explore-u1-temp30-eps025-sim5-20260511",
    "curvytron-two-seat-selfplay-live11-explore-u1-temp20-eps050-sim5-20260511",
    "curvytron-two-seat-selfplay-live11-sim8-u1-temp20-eps025-20260511",
    "curvytron-two-seat-selfplay-live11-sim8-u1-temp15-eps050-20260511",
    "curvytron-two-seat-selfplay-live11-survival005-u1-temp20-eps025-20260511",
    "curvytron-two-seat-selfplay-live11-nodead-u1-temp20-eps025-20260511",
    "curvytron-two-seat-selfplay-live11-noop05warm-u1-temp20-eps025-20260511",
    "curvytron-two-seat-selfplay-live11-noop10warm-u1-temp20-eps025-20260511",
    "curvytron-two-seat-selfplay-live11-imgnoise005-u1-temp20-eps025-20260511",
    "curvytron-two-seat-selfplay-live11-imgnoise010-u1-temp20-eps025-20260511",
    "curvytron-two-seat-selfplay-live11-mixed-noop05-img005-u1-temp20-eps025-20260511",
    "curvytron-two-seat-selfplay-live11-wide-explore-noop10-img005-u1-temp30-eps050-20260511",
)

STOCK_HIGH_SIGNAL_V1_RUN_IDS = (
    "curvytron-stock-stock-high-signal-v1-01-fixed-straight-sparse-b32-sim8",
    "curvytron-stock-stock-high-signal-v1-02-fixed-straight-dense-b32-sim8",
    "curvytron-stock-stock-high-signal-v1-03-frozen-recent-dense-b32-sim8",
    "curvytron-stock-stock-high-signal-v1-04-frozen-recent-sparse-b32-sim8",
    "curvytron-stock-stock-high-signal-v1-05-frozen-mid-dense-b32-sim8",
    "curvytron-stock-stock-high-signal-v1-06-frozen-old-dense-b32-sim8",
    "curvytron-stock-stock-high-signal-v1-07-frozen-recent-dense-b32-sim16",
    "curvytron-stock-stock-high-signal-v1-08-frozen-recent-dense-b64-sim8",
    "curvytron-stock-stock-high-signal-v1-09-joint-diagnostic-b32-sim8",
    "curvytron-stock-stock-high-signal-v1-10-joint-diagnostic-b32-sim16",
)

STOCK_HIGH_SIGNAL_V1_ATTEMPT_IDS = (
    "stock-high-signal-v1-attempt-01-fixed-straight-sparse-b32-sim8",
    "stock-high-signal-v1-attempt-02-fixed-straight-dense-b32-sim8",
    "stock-high-signal-v1-attempt-03-frozen-recent-dense-b32-sim8",
    "stock-high-signal-v1-attempt-04-frozen-recent-sparse-b32-sim8",
    "stock-high-signal-v1-attempt-05-frozen-mid-dense-b32-sim8",
    "stock-high-signal-v1-attempt-06-frozen-old-dense-b32-sim8",
    "stock-high-signal-v1-attempt-07-frozen-recent-dense-b32-sim16",
    "stock-high-signal-v1-attempt-08-frozen-recent-dense-b64-sim8",
    "stock-high-signal-v1-attempt-09-joint-diagnostic-b32-sim8",
    "stock-high-signal-v1-attempt-10-joint-diagnostic-b32-sim16",
)

app = modal.App(APP_NAME)


def _attempt_id_for_run(run_id: str) -> str:
    prefix = "curvytron-two-seat-selfplay-"
    if run_id.startswith(prefix):
        return run_id.removeprefix(prefix)
    return "train"


def _latest_attempt_id_for_run(run_id: str) -> str | None:
    latest_path = runs.volume_path(
        RUNS_MOUNT,
        runs.latest_attempt_ref(TASK_ID, run_id),
    )
    latest = _load_json(latest_path)
    if not latest:
        return None
    attempt_id = latest.get("attempt_id")
    return str(attempt_id) if attempt_id else None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        return {"error": f"expected JSON object in {path}"}
    return value


def _load_progress_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("event") == "iteration":
            rows.append(value)
    return rows


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _action_summary(
    action_counts: dict[str, Any] | None,
    *,
    collapse_threshold: float,
) -> dict[str, Any]:
    if not action_counts:
        return {
            "action_total": 0,
            "top_action": None,
            "top_action_count": 0,
            "top_action_fraction": None,
            "collapsed": None,
        }
    counts: dict[str, int] = {}
    for key, value in action_counts.items():
        try:
            counts[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    total = sum(counts.values())
    if total <= 0:
        return {
            "action_total": 0,
            "top_action": None,
            "top_action_count": 0,
            "top_action_fraction": None,
            "collapsed": None,
        }
    top_action, top_count = max(counts.items(), key=lambda item: item[1])
    top_fraction = top_count / total
    return {
        "action_total": total,
        "top_action": top_action,
        "top_action_count": top_count,
        "top_action_fraction": top_fraction,
        "collapsed": top_fraction >= collapse_threshold,
    }


def _manifest_timestamp(path: Path, manifest: dict[str, Any]) -> str | None:
    created_at = manifest.get("created_at")
    if isinstance(created_at, str) and created_at:
        return created_at
    try:
        return path.stat().st_mtime_ns.__str__()
    except FileNotFoundError:
        return None


def _manifest_sort_key(path: Path, manifest: dict[str, Any]) -> tuple[str, str]:
    return (_manifest_timestamp(path, manifest) or "", path.as_posix())


def _load_eval_manifests(run_id: str, attempt_id: str) -> list[tuple[Path, dict[str, Any]]]:
    eval_root = runs.volume_path(
        RUNS_MOUNT,
        runs.attempt_root_ref(TASK_ID, run_id, attempt_id) / "eval",
    )
    if not eval_root.is_dir():
        return []
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for path in eval_root.rglob("manifest_*.json"):
        try:
            manifest = _load_json(path)
        except json.JSONDecodeError:
            continue
        if not manifest:
            continue
        if manifest.get("job_kind") not in EVAL_JOB_KINDS:
            continue
        manifests.append((path, manifest))
    return sorted(manifests, key=lambda item: _manifest_sort_key(*item))


def _load_gif_summaries(run_id: str, attempt_id: str) -> list[tuple[Path, dict[str, Any]]]:
    eval_root = runs.volume_path(
        RUNS_MOUNT,
        runs.attempt_root_ref(TASK_ID, run_id, attempt_id) / "eval",
    )
    if not eval_root.is_dir():
        return []
    summaries: list[tuple[Path, dict[str, Any]]] = []
    for path in eval_root.rglob("selfplay/summary.json"):
        try:
            summary = _load_json(path)
        except json.JSONDecodeError:
            continue
        if not summary:
            continue
        if (
            summary.get("schema_id")
            != "curvyzero_lightzero_curvytron_checkpoint_selfplay_gif_summary/v0"
        ):
            continue
        summaries.append((path, summary))
    return sorted(summaries, key=lambda item: _manifest_sort_key(*item))


def _normalise_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, raw_count in value.items():
        count = _safe_int(raw_count)
        if count is None:
            continue
        counts[str(key)] = count
    return counts


def _sum_action_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(_normalise_counts(row.get("action_histogram")))
    return dict(sorted(counts.items()))


def _outcome_histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        outcome = (
            row.get("terminal_reason")
            or row.get("death_cause_name")
            or row.get("death_cause")
            or row.get("error_type")
            or ("ok" if row.get("ok") else "failed")
        )
        counts[str(outcome)] += 1
    return dict(sorted(counts.items()))


def _rows_for_checkpoint(manifest: dict[str, Any], checkpoint: str) -> list[dict[str, Any]]:
    rows = manifest.get("table")
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("checkpoint_label") or "") == checkpoint
    ]


def _aggregate_by_checkpoint(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    aggregate_table = manifest.get("survival_aggregate_table")
    if isinstance(aggregate_table, list):
        for row in aggregate_table:
            if not isinstance(row, dict):
                continue
            checkpoint = str(row.get("checkpoint") or "")
            if checkpoint:
                aggregates[checkpoint] = row
    return aggregates


def _checkpoint_iteration(checkpoint: Any) -> int | None:
    text = str(checkpoint or "")
    if text.startswith("iteration_"):
        return _safe_int(text.removeprefix("iteration_"))
    return None


def _checkpoint_sort_key(checkpoint: Any) -> tuple[int, int, str]:
    text = str(checkpoint or "")
    iteration = _checkpoint_iteration(text)
    if iteration is None:
        return (1, 0, text)
    return (0, iteration, text)


def _eval_manifest_rollup(
    run_id: str,
    attempt_id: str,
    *,
    collapse_threshold: float,
) -> dict[str, Any]:
    manifests = _load_eval_manifests(run_id, attempt_id)
    if not manifests:
        return {
            "eval_manifest_count": 0,
            "latest_eval_manifest_ref": None,
            "eval_checkpoints": [],
        }

    latest_by_checkpoint: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    for path, manifest in manifests:
        for checkpoint, aggregate in _aggregate_by_checkpoint(manifest).items():
            latest_by_checkpoint[checkpoint] = (path, manifest, aggregate)

    checkpoint_rows: list[dict[str, Any]] = []
    for checkpoint, (path, manifest, aggregate) in sorted(
        latest_by_checkpoint.items(),
        key=lambda item: _checkpoint_sort_key(item[0]),
    ):
        rows = _rows_for_checkpoint(manifest, checkpoint)
        action_summary = _action_summary(
            _sum_action_counts(rows),
            collapse_threshold=collapse_threshold,
        )
        checkpoint_rows.append(
            {
                "checkpoint": checkpoint,
                "manifest_ref": path.relative_to(RUNS_MOUNT).as_posix(),
                "eval_id": manifest.get("eval_id"),
                "created_at": manifest.get("created_at"),
                "seeds": aggregate.get("seeds"),
                "mean_steps": _safe_float(aggregate.get("mean_steps")),
                "median_steps": _safe_float(aggregate.get("median_steps")),
                "min_steps": _safe_float(aggregate.get("min_steps")),
                "max_steps": _safe_float(aggregate.get("max_steps")),
                "ok_count": aggregate.get("ok_count"),
                "capped_count": aggregate.get("capped_count"),
                "failure_count": aggregate.get("failure_count"),
                "outcome_histogram": _outcome_histogram(rows),
                "action_summary": action_summary,
                "row_action_collapsed_count": sum(
                    1
                    for row in rows
                    if _action_summary(
                        _normalise_counts(row.get("action_histogram")),
                        collapse_threshold=collapse_threshold,
                    ).get("collapsed")
                    is True
                ),
            }
        )

    latest_path, latest_manifest = manifests[-1]
    return {
        "eval_manifest_count": len(manifests),
        "latest_eval_manifest_ref": latest_path.relative_to(RUNS_MOUNT).as_posix(),
        "latest_eval_created_at": latest_manifest.get("created_at"),
        "eval_checkpoints": checkpoint_rows,
    }


def _action_observability_rollup(run_id: str, attempt_id: str) -> dict[str, Any]:
    train_ref = runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
    action_ref = train_ref / "action_observability.json"
    action_observability = _load_json(runs.volume_path(RUNS_MOUNT, action_ref))
    if action_observability is None:
        return {
            "action_observability_exists": False,
            "action_observability_ref": action_ref.as_posix(),
        }
    ego_counts = _normalise_counts(action_observability.get("ego_action_histogram"))
    physical_counts = _normalise_counts(action_observability.get("physical_action_histogram"))
    return {
        "action_observability_exists": True,
        "action_observability_ref": action_ref.as_posix(),
        "action_observability_status": action_observability.get("status"),
        "action_observability_rows": action_observability.get("row_count"),
        "train_action_summary": _action_summary(
            ego_counts,
            collapse_threshold=DEFAULT_COLLAPSE_THRESHOLD,
        ),
        "train_physical_action_summary": _action_summary(
            physical_counts,
            collapse_threshold=DEFAULT_COLLAPSE_THRESHOLD,
        ),
        "train_terminal_reasons": _normalise_counts(action_observability.get("terminal_reasons")),
    }


def _poller_rollup(run_id: str, attempt_id: str) -> dict[str, Any]:
    train_ref = runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
    poller_ref = train_ref / "checkpoint_eval_poller.json"
    status = _load_json(runs.volume_path(RUNS_MOUNT, poller_ref))
    if status is None:
        return {
            "background_poller_exists": False,
            "background_poller_ref": poller_ref.as_posix(),
        }
    return {
        "background_poller_exists": True,
        "background_poller_ref": poller_ref.as_posix(),
        "background_poller_status": status.get("status"),
        "background_poller_seen_count": status.get("seen_count"),
        "background_poller_scheduled_count": status.get("scheduled_count"),
        "background_poller_completed_count": status.get("completed_count"),
        "background_poller_eval_completed_count": status.get("eval_completed_count"),
        "background_poller_gif_scheduled_count": status.get("gif_scheduled_count"),
        "background_poller_gif_completed_count": status.get("gif_completed_count"),
    }


def _gif_rollup(
    run_id: str,
    attempt_id: str,
    *,
    collapse_threshold: float,
) -> dict[str, Any]:
    summaries = _load_gif_summaries(run_id, attempt_id)
    if not summaries:
        return {
            "gif_artifact_count": 0,
            "latest_gif_summary_ref": None,
            "gif_artifacts": [],
        }

    artifacts: list[dict[str, Any]] = []
    for path, summary in summaries:
        greedy_summary = (
            summary.get("greedy_action_summary")
            if isinstance(summary.get("greedy_action_summary"), dict)
            else {}
        )
        variants = (
            summary.get("gif_variants") if isinstance(summary.get("gif_variants"), dict) else {}
        )
        collect = variants.get("collect_t1") if isinstance(variants.get("collect_t1"), dict) else {}
        artifacts.append(
            {
                "summary_ref": path.relative_to(RUNS_MOUNT).as_posix(),
                "eval_id": summary.get("eval_id"),
                "checkpoint_label": summary.get("checkpoint_label"),
                "ok": summary.get("ok"),
                "gif_ref": summary.get("gif_ref"),
                "collect_t1_gif_ref": collect.get("gif_ref"),
                "frame_count": summary.get("frame_count"),
                "physical_steps": summary.get("physical_steps"),
                "terminal_reason": summary.get("terminal_reason"),
                "stop_reason": summary.get("stop_reason"),
                "greedy_action_collapse_warning": summary.get("greedy_action_collapse_warning"),
                "greedy_decision_count": greedy_summary.get("decision_count"),
                "greedy_action_summary": greedy_summary,
            }
        )

    latest_path, latest_summary = summaries[-1]
    latest_action_counts: dict[str, int] = {}
    latest_greedy = latest_summary.get("greedy_action_summary")
    if isinstance(latest_greedy, dict):
        counts_by_player = latest_greedy.get("action_counts_by_player")
        if isinstance(counts_by_player, dict):
            for counts in counts_by_player.values():
                for action, count in _normalise_counts(counts).items():
                    latest_action_counts[action] = latest_action_counts.get(action, 0) + count
    return {
        "gif_artifact_count": len(summaries),
        "latest_gif_summary_ref": latest_path.relative_to(RUNS_MOUNT).as_posix(),
        "latest_gif_ref": latest_summary.get("gif_ref"),
        "latest_gif_checkpoint": latest_summary.get("checkpoint_label"),
        "latest_gif_terminal_reason": latest_summary.get("terminal_reason"),
        "latest_gif_action_summary": _action_summary(
            latest_action_counts,
            collapse_threshold=collapse_threshold,
        ),
        "gif_artifacts": artifacts,
    }


def _train_artifact_rollup(
    run_id: str,
    attempt_id: str,
    *,
    progress_exists: bool,
    collapse_threshold: float,
) -> dict[str, Any]:
    train_ref = runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
    train_root = runs.volume_path(RUNS_MOUNT, train_ref)
    heartbeat_ref = train_ref / "status_heartbeat.json"
    heartbeat = _load_json(runs.volume_path(RUNS_MOUNT, heartbeat_ref))
    if progress_exists:
        missing_reason = None
    elif not train_root.exists():
        missing_reason = "train_root_absent"
    elif heartbeat is not None:
        missing_reason = "progress_latest_absent_after_train_heartbeat"
    else:
        missing_reason = "train_root_exists_progress_latest_absent"
    return {
        "train_root_exists": train_root.exists(),
        "status_heartbeat_exists": heartbeat is not None,
        "status_heartbeat_ref": heartbeat_ref.as_posix(),
        "train_status": heartbeat.get("status") if isinstance(heartbeat, dict) else None,
        "train_stage": heartbeat.get("stage") if isinstance(heartbeat, dict) else None,
        "progress_missing_reason": missing_reason,
        **_action_observability_rollup(run_id, attempt_id),
        **_poller_rollup(run_id, attempt_id),
        **_gif_rollup(
            run_id,
            attempt_id,
            collapse_threshold=collapse_threshold,
        ),
    }


def _checkpoint_summary(run_id: str, attempt_id: str | None = None) -> dict[str, Any]:
    checkpoint_dirs = [
        runs.volume_path(
            RUNS_MOUNT,
            runs.checkpoints_root_ref(TASK_ID, run_id) / "lightzero",
        )
    ]
    if attempt_id:
        checkpoint_dirs.append(
            runs.volume_path(
                RUNS_MOUNT,
                runs.attempt_train_ref(TASK_ID, run_id, attempt_id) / "lightzero_exp" / "ckpt",
            )
        )
    checkpoint_dir = next((path for path in checkpoint_dirs if path.is_dir()), None)
    if checkpoint_dir is None:
        return {"checkpoint_count": 0, "latest_checkpoint": None}
    iterations: list[int] = []
    for child in checkpoint_dir.iterdir():
        name = child.name
        if not name.startswith("iteration_") or not name.endswith(".pth.tar"):
            continue
        text = name.removeprefix("iteration_").removesuffix(".pth.tar")
        try:
            iterations.append(int(text))
        except ValueError:
            pass
    if not iterations:
        return {"checkpoint_count": 0, "latest_checkpoint": None}
    return {
        "checkpoint_count": len(iterations),
        "latest_checkpoint": f"iteration_{max(iterations)}",
    }


def _run_status(
    run_id: str,
    *,
    attempt_id: str | None,
    collapse_threshold: float,
) -> dict[str, Any]:
    resolved_attempt_id = (
        attempt_id or _latest_attempt_id_for_run(run_id) or _attempt_id_for_run(run_id)
    )
    train_ref = runs.attempt_train_ref(TASK_ID, run_id, resolved_attempt_id)
    latest_path = runs.volume_path(RUNS_MOUNT, train_ref / "progress_latest.json")
    progress = _load_json(latest_path)
    checkpoints = _checkpoint_summary(run_id, resolved_attempt_id)
    eval_rollup = _eval_manifest_rollup(
        run_id,
        resolved_attempt_id,
        collapse_threshold=collapse_threshold,
    )
    train_artifacts = _train_artifact_rollup(
        run_id,
        resolved_attempt_id,
        progress_exists=progress is not None,
        collapse_threshold=collapse_threshold,
    )
    row: dict[str, Any] = {
        "run_id": run_id,
        "short_name": run_id.removeprefix("curvytron-two-seat-selfplay-"),
        "attempt_id": resolved_attempt_id,
        "progress_exists": progress is not None,
        "progress_ref": (train_ref / "progress_latest.json").as_posix(),
        **checkpoints,
        **eval_rollup,
        **train_artifacts,
    }
    if progress is None:
        row.update(
            {
                "event": "missing",
                "iteration": None,
                "mean_steps": None,
                "max_steps": None,
                "completed_episodes": None,
                "model_changed": None,
                "problems": None,
                "top_action": None,
                "top_action_fraction": None,
                "collapsed": None,
            }
        )
        return row
    learner = progress.get("last_learner")
    if not isinstance(learner, dict):
        learner = {}
    row.update(
        {
            "event": progress.get("event"),
            "iteration": progress.get("iteration"),
            "mean_steps": _safe_float(progress.get("mean_completed_episode_steps")),
            "max_steps": progress.get("max_completed_episode_steps"),
            "completed_episodes": progress.get("completed_episode_count"),
            "model_changed": learner.get("model_parameters_changed"),
            "problems": progress.get("problem_count"),
            "effective_noop": _safe_float(progress.get("effective_action_noop_probability")),
            "elapsed_sec": _safe_float(progress.get("elapsed_sec")),
            "timestamp": progress.get("timestamp"),
        }
    )
    row.update(
        _action_summary(
            progress.get("action_counts")
            if isinstance(progress.get("action_counts"), dict)
            else None,
            collapse_threshold=collapse_threshold,
        )
    )
    return row


def _progress_curve(
    run_id: str,
    *,
    attempt_id: str | None,
    collapse_threshold: float,
) -> dict[str, Any]:
    resolved_attempt_id = (
        attempt_id or _latest_attempt_id_for_run(run_id) or _attempt_id_for_run(run_id)
    )
    train_ref = runs.attempt_train_ref(TASK_ID, run_id, resolved_attempt_id)
    progress_path = runs.volume_path(RUNS_MOUNT, train_ref / "progress.jsonl")
    rows = _load_progress_rows(progress_path)
    curve: list[dict[str, Any]] = []
    for progress in rows:
        learner = progress.get("last_learner")
        if not isinstance(learner, dict):
            learner = {}
        point = {
            "iteration": progress.get("iteration"),
            "timestamp": progress.get("timestamp"),
            "mean_steps": _safe_float(progress.get("mean_completed_episode_steps")),
            "max_steps": progress.get("max_completed_episode_steps"),
            "completed_episodes": progress.get("completed_episode_count"),
            "survival_reward_sum": _safe_float(progress.get("survival_reward_sum")),
            "checkpoint_saved": bool(progress.get("checkpoint_saved")),
            "model_changed": learner.get("model_parameters_changed"),
            "problems": progress.get("problem_count"),
            "effective_noop": _safe_float(progress.get("effective_action_noop_probability")),
            "elapsed_sec": _safe_float(progress.get("elapsed_sec")),
        }
        point.update(
            _action_summary(
                progress.get("action_counts")
                if isinstance(progress.get("action_counts"), dict)
                else None,
                collapse_threshold=collapse_threshold,
            )
        )
        curve.append(point)
    return {
        "run_id": run_id,
        "short_name": run_id.removeprefix("curvytron-two-seat-selfplay-"),
        "attempt_id": resolved_attempt_id,
        "progress_ref": (train_ref / "progress.jsonl").as_posix(),
        "point_count": len(curve),
        "curve": curve,
        **_checkpoint_summary(run_id, resolved_attempt_id),
        **_eval_manifest_rollup(
            run_id,
            resolved_attempt_id,
            collapse_threshold=collapse_threshold,
        ),
        **_train_artifact_rollup(
            run_id,
            resolved_attempt_id,
            progress_exists=(progress_path.parent / "progress_latest.json").exists(),
            collapse_threshold=collapse_threshold,
        ),
    }


def _eval_curve_status(
    run_id: str,
    *,
    attempt_id: str | None,
    collapse_threshold: float,
) -> dict[str, Any]:
    resolved_attempt_id = (
        attempt_id or _latest_attempt_id_for_run(run_id) or _attempt_id_for_run(run_id)
    )
    train_ref = runs.attempt_train_ref(TASK_ID, run_id, resolved_attempt_id)
    heartbeat = _load_json(
        runs.volume_path(RUNS_MOUNT, train_ref / "status_heartbeat.json")
    )
    return {
        "run_id": run_id,
        "short_name": run_id.removeprefix("curvytron-two-seat-selfplay-"),
        "attempt_id": resolved_attempt_id,
        "status_heartbeat_exists": heartbeat is not None,
        "train_status": heartbeat.get("status") if isinstance(heartbeat, dict) else None,
        "train_stage": heartbeat.get("stage") if isinstance(heartbeat, dict) else None,
        **_checkpoint_summary(run_id, resolved_attempt_id),
        **_eval_manifest_rollup(
            run_id,
            resolved_attempt_id,
            collapse_threshold=collapse_threshold,
        ),
        **_poller_rollup(run_id, resolved_attempt_id),
    }


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _preset_run_ids(preset: str) -> tuple[list[str], list[str]]:
    if preset == "live6":
        return list(LIVE6_RUN_IDS), []
    if preset == "live11":
        return list(LIVE11_RUN_IDS), []
    if preset == "stock-high-signal-v1":
        return list(STOCK_HIGH_SIGNAL_V1_RUN_IDS), list(STOCK_HIGH_SIGNAL_V1_ATTEMPT_IDS)
    raise ValueError(
        "use --preset live6, --preset live11, --preset stock-high-signal-v1, or pass --run-ids"
    )


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_counts(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    parts = []
    for key in sorted(value):
        count = value.get(key)
        if count in (None, 0, "0"):
            continue
        parts.append(f"{key}:{count}")
    return ",".join(parts)


def _print_table(rows: list[dict[str, Any]]) -> None:
    fields = [
        "short_name",
        "iteration",
        "mean_steps",
        "max_steps",
        "top_action",
        "top_action_fraction",
        "collapsed",
        "model_changed",
        "checkpoint_count",
        "latest_checkpoint",
        "progress_missing_reason",
        "status_heartbeat_exists",
        "action_observability_exists",
        "train_top_action",
        "train_top_action_fraction",
        "eval_manifest_count",
        "latest_eval_checkpoint",
        "latest_eval_mean_steps",
        "latest_eval_top_action",
        "latest_eval_action_fraction",
        "latest_eval_collapsed",
        "background_poller_status",
        "background_poller_seen_count",
        "background_poller_completed_count",
        "gif_artifact_count",
        "latest_gif_checkpoint",
        "latest_gif_top_action",
        "latest_gif_action_fraction",
        "problems",
        "elapsed_sec",
    ]
    print("\t".join(fields))
    for row in rows:
        eval_checkpoints = row.get("eval_checkpoints")
        latest_eval = (
            eval_checkpoints[-1] if isinstance(eval_checkpoints, list) and eval_checkpoints else {}
        )
        latest_action = (
            latest_eval.get("action_summary")
            if isinstance(latest_eval.get("action_summary"), dict)
            else {}
        )
        rendered_row = {
            **row,
            "train_top_action": (
                row.get("train_action_summary", {}).get("top_action")
                if isinstance(row.get("train_action_summary"), dict)
                else None
            ),
            "train_top_action_fraction": (
                row.get("train_action_summary", {}).get("top_action_fraction")
                if isinstance(row.get("train_action_summary"), dict)
                else None
            ),
            "latest_eval_checkpoint": latest_eval.get("checkpoint"),
            "latest_eval_mean_steps": latest_eval.get("mean_steps"),
            "latest_eval_top_action": latest_action.get("top_action"),
            "latest_eval_action_fraction": latest_action.get("top_action_fraction"),
            "latest_eval_collapsed": latest_action.get("collapsed"),
            "latest_gif_top_action": (
                row.get("latest_gif_action_summary", {}).get("top_action")
                if isinstance(row.get("latest_gif_action_summary"), dict)
                else None
            ),
            "latest_gif_action_fraction": (
                row.get("latest_gif_action_summary", {}).get("top_action_fraction")
                if isinstance(row.get("latest_gif_action_summary"), dict)
                else None
            ),
        }
        values = []
        for field in fields:
            value = rendered_row.get(field)
            if field in {
                "mean_steps",
                "top_action_fraction",
                "elapsed_sec",
                "latest_eval_mean_steps",
                "latest_eval_action_fraction",
                "train_top_action_fraction",
                "latest_gif_action_fraction",
            }:
                value = _format_float(value)
            elif value is None:
                value = ""
            values.append(str(value))
        print("\t".join(values))


def _print_curve_table(rows: list[dict[str, Any]]) -> None:
    fields = [
        "short_name",
        "iteration",
        "mean_steps",
        "max_steps",
        "top_action",
        "top_action_fraction",
        "collapsed",
        "checkpoint_saved",
        "problems",
        "elapsed_sec",
    ]
    print("\t".join(fields))
    for row in rows:
        short_name = row.get("short_name")
        for point in row.get("curve", []):
            values = []
            merged = {"short_name": short_name, **point}
            for field in fields:
                value = merged.get(field)
                if field in {"mean_steps", "top_action_fraction", "elapsed_sec"}:
                    value = _format_float(value)
                elif value is None:
                    value = ""
                values.append(str(value))
            print("\t".join(values))


def _print_curve_summary(rows: list[dict[str, Any]]) -> None:
    fields = [
        "short_name",
        "curve_source",
        "points",
        "first_iter",
        "first_mean",
        "best_iter",
        "best_mean",
        "latest_iter",
        "latest_mean",
        "latest_max",
        "latest_outcomes",
        "latest_top_action",
        "latest_top_fraction",
        "any_collapsed",
        "checkpoints",
        "latest_checkpoint",
        "progress_missing_reason",
        "eval_manifests",
        "latest_eval_checkpoint",
        "latest_eval_mean",
        "latest_eval_top_action",
        "latest_eval_collapsed",
        "poller_status",
        "gif_artifacts",
        "latest_gif_checkpoint",
    ]
    print("\t".join(fields))
    for row in rows:
        eval_checkpoints = row.get("eval_checkpoints")
        latest_eval = (
            eval_checkpoints[-1] if isinstance(eval_checkpoints, list) and eval_checkpoints else {}
        )
        latest_action = (
            latest_eval.get("action_summary")
            if isinstance(latest_eval.get("action_summary"), dict)
            else {}
        )
        raw_curve = row.get("curve", [])
        curve_source = "progress"
        curve = raw_curve if isinstance(raw_curve, list) else []
        if not curve and isinstance(eval_checkpoints, list):
            curve_source = "eval"
            curve = []
            for checkpoint in eval_checkpoints:
                if not isinstance(checkpoint, dict):
                    continue
                action_summary = checkpoint.get("action_summary")
                if not isinstance(action_summary, dict):
                    action_summary = {}
                curve.append(
                    {
                        "iteration": _checkpoint_iteration(checkpoint.get("checkpoint")),
                        "mean_steps": checkpoint.get("mean_steps"),
                        "max_steps": checkpoint.get("max_steps"),
                        "top_action": action_summary.get("top_action"),
                        "top_action_fraction": action_summary.get("top_action_fraction"),
                        "collapsed": action_summary.get("collapsed"),
                    }
                )
        elif not curve:
            curve_source = "none"
        if not curve:
            print(
                "\t".join(
                    [
                        str(row.get("short_name")),
                        curve_source,
                        "0",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        str(row.get("checkpoint_count")),
                        str(row.get("latest_checkpoint") or ""),
                        str(row.get("progress_missing_reason") or ""),
                        str(row.get("eval_manifest_count") or 0),
                        str(latest_eval.get("checkpoint") or ""),
                        _format_float(latest_eval.get("mean_steps")),
                        str(latest_action.get("top_action") or ""),
                        str(latest_action.get("collapsed") or ""),
                        str(row.get("background_poller_status") or ""),
                        str(row.get("gif_artifact_count") or 0),
                        str(row.get("latest_gif_checkpoint") or ""),
                    ]
                )
            )
            continue
        first = curve[0]
        latest = curve[-1]
        best = max(
            curve,
            key=lambda point: (
                float(point.get("mean_steps") or float("-inf")),
                int(point.get("iteration") or -1),
            ),
        )
        values = [
            str(row.get("short_name")),
            curve_source,
            str(len(curve)),
            str(first.get("iteration") if first.get("iteration") is not None else ""),
            _format_float(first.get("mean_steps")),
            str(best.get("iteration") if best.get("iteration") is not None else ""),
            _format_float(best.get("mean_steps")),
            str(latest.get("iteration") if latest.get("iteration") is not None else ""),
            _format_float(latest.get("mean_steps")),
            str(latest.get("max_steps") or ""),
            str(latest.get("top_action") or ""),
            _format_float(latest.get("top_action_fraction")),
            str(any(point.get("collapsed") for point in curve)),
            str(row.get("checkpoint_count")),
            str(row.get("latest_checkpoint") or ""),
            str(row.get("progress_missing_reason") or ""),
            str(row.get("eval_manifest_count") or 0),
            str(latest_eval.get("checkpoint") or ""),
            _format_float(latest_eval.get("mean_steps")),
            str(latest_action.get("top_action") or ""),
            str(latest_action.get("collapsed") or ""),
            str(row.get("background_poller_status") or ""),
            str(row.get("gif_artifact_count") or 0),
            str(row.get("latest_gif_checkpoint") or ""),
        ]
        print("\t".join(values))


def _print_eval_summary(rows: list[dict[str, Any]]) -> None:
    fields = [
        "short_name",
        "points",
        "first_iter",
        "first_mean",
        "best_iter",
        "best_mean",
        "latest_iter",
        "latest_mean",
        "latest_max",
        "latest_outcomes",
        "latest_top_action",
        "latest_top_fraction",
        "any_collapsed",
        "checkpoints",
        "latest_checkpoint",
        "eval_manifests",
        "poller_status",
        "poller_seen",
        "poller_done",
        "train_status",
        "train_stage",
    ]
    print("\t".join(fields))
    for row in rows:
        raw_checkpoints = row.get("eval_checkpoints")
        checkpoints = [
            checkpoint
            for checkpoint in raw_checkpoints
            if isinstance(checkpoint, dict)
        ] if isinstance(raw_checkpoints, list) else []
        points: list[dict[str, Any]] = []
        for checkpoint in checkpoints:
            action_summary = checkpoint.get("action_summary")
            if not isinstance(action_summary, dict):
                action_summary = {}
            points.append(
                {
                    "iteration": _checkpoint_iteration(checkpoint.get("checkpoint")),
                    "mean_steps": checkpoint.get("mean_steps"),
                    "max_steps": checkpoint.get("max_steps"),
                    "outcome_histogram": checkpoint.get("outcome_histogram"),
                    "top_action": action_summary.get("top_action"),
                    "top_action_fraction": action_summary.get("top_action_fraction"),
                    "collapsed": action_summary.get("collapsed"),
                }
            )
        if not points:
            print(
                "\t".join(
                    [
                        str(row.get("short_name")),
                        "0",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        str(row.get("checkpoint_count") or 0),
                        str(row.get("latest_checkpoint") or ""),
                        str(row.get("eval_manifest_count") or 0),
                        str(row.get("background_poller_status") or ""),
                        str(row.get("background_poller_seen_count") or ""),
                        str(row.get("background_poller_completed_count") or ""),
                        str(row.get("train_status") or ""),
                        str(row.get("train_stage") or ""),
                    ]
                )
            )
            continue
        first = points[0]
        latest = points[-1]
        best = max(
            points,
            key=lambda point: (
                float(point.get("mean_steps") or float("-inf")),
                int(point.get("iteration") or -1),
            ),
        )
        values = [
            str(row.get("short_name")),
            str(len(points)),
            str(first.get("iteration") if first.get("iteration") is not None else ""),
            _format_float(first.get("mean_steps")),
            str(best.get("iteration") if best.get("iteration") is not None else ""),
            _format_float(best.get("mean_steps")),
            str(latest.get("iteration") if latest.get("iteration") is not None else ""),
            _format_float(latest.get("mean_steps")),
            str(latest.get("max_steps") or ""),
            _format_counts(latest.get("outcome_histogram")),
            str(latest.get("top_action") or ""),
            _format_float(latest.get("top_action_fraction")),
            str(any(point.get("collapsed") for point in points)),
            str(row.get("checkpoint_count") or 0),
            str(row.get("latest_checkpoint") or ""),
            str(row.get("eval_manifest_count") or 0),
            str(row.get("background_poller_status") or ""),
            str(row.get("background_poller_seen_count") or ""),
            str(row.get("background_poller_completed_count") or ""),
            str(row.get("train_status") or ""),
            str(row.get("train_stage") or ""),
        ]
        print("\t".join(values))


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=5 * 60, cpu=1.0)
def curvytron_run_status(
    run_ids: list[str],
    attempt_ids: list[str] | None = None,
    collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
) -> list[dict[str, Any]]:
    resolved_attempt_ids = attempt_ids or []
    rows: list[dict[str, Any]] = []
    for index, run_id in enumerate(run_ids):
        attempt_id = (
            resolved_attempt_ids[index]
            if index < len(resolved_attempt_ids) and resolved_attempt_ids[index]
            else None
        )
        rows.append(
            _run_status(
                run_id,
                attempt_id=attempt_id,
                collapse_threshold=float(collapse_threshold),
            )
        )
    return rows


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=5 * 60, cpu=1.0)
def curvytron_run_curves(
    run_ids: list[str],
    attempt_ids: list[str] | None = None,
    collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
) -> list[dict[str, Any]]:
    resolved_attempt_ids = attempt_ids or []
    rows: list[dict[str, Any]] = []
    for index, run_id in enumerate(run_ids):
        attempt_id = (
            resolved_attempt_ids[index]
            if index < len(resolved_attempt_ids) and resolved_attempt_ids[index]
            else None
        )
        rows.append(
            _progress_curve(
                run_id,
                attempt_id=attempt_id,
                collapse_threshold=float(collapse_threshold),
            )
        )
    return rows


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=5 * 60, cpu=1.0)
def curvytron_run_eval_curves(
    run_ids: list[str],
    attempt_ids: list[str] | None = None,
    collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
) -> list[dict[str, Any]]:
    resolved_attempt_ids = attempt_ids or []
    rows: list[dict[str, Any]] = []
    for index, run_id in enumerate(run_ids):
        attempt_id = (
            resolved_attempt_ids[index]
            if index < len(resolved_attempt_ids) and resolved_attempt_ids[index]
            else None
        )
        rows.append(
            _eval_curve_status(
                run_id,
                attempt_id=attempt_id,
                collapse_threshold=float(collapse_threshold),
            )
        )
    return rows


@app.local_entrypoint()
def main(
    preset: str = "live6",
    run_ids: str | None = None,
    attempt_ids: str | None = None,
    collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
    output: str = "table",
) -> None:
    if run_ids:
        selected_run_ids = _split_csv(run_ids)
        preset_attempt_ids: list[str] = []
    else:
        selected_run_ids, preset_attempt_ids = _preset_run_ids(preset)
    selected_attempt_ids = _split_csv(attempt_ids) or preset_attempt_ids
    if output == "eval-summary":
        rows = curvytron_run_eval_curves.remote(
            selected_run_ids,
            selected_attempt_ids,
            collapse_threshold,
        )
    elif output in {"curve-json", "curve-table", "curve-summary"}:
        rows = curvytron_run_curves.remote(
            selected_run_ids,
            selected_attempt_ids,
            collapse_threshold,
        )
    else:
        rows = curvytron_run_status.remote(
            selected_run_ids,
            selected_attempt_ids,
            collapse_threshold,
        )
    if output == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif output == "table":
        _print_table(rows)
    elif output == "curve-json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif output == "curve-table":
        _print_curve_table(rows)
    elif output == "curve-summary":
        _print_curve_summary(rows)
    elif output == "eval-summary":
        _print_eval_summary(rows)
    else:
        raise ValueError(
            "output must be table, json, curve-table, curve-summary, "
            "curve-json, or eval-summary"
        )
