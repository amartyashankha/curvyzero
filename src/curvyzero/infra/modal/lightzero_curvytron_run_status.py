"""Compact status reader for CurvyTron LightZero training runs.

This reads Modal Volume progress files from inside one Modal function. It is
meant to replace noisy manual ``modal volume get`` polling.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from curvyzero.training import lightzero_checkpoints as lz_checkpoints


APP_NAME = "curvyzero-lightzero-curvytron-run-status"
DEFAULT_COLLAPSE_THRESHOLD = 0.95
DEFAULT_ASSIGNMENT_PROOF_TAIL_BYTES = 128 * 1024 * 1024
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
    except json.JSONDecodeError as exc:
        return {"error": f"invalid JSON in {path}: {exc}"}
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


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
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
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _iter_jsonl_rows(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    yield value
    except FileNotFoundError:
        return


def _iter_jsonl_tail_rows(path: Path, *, max_tail_bytes: int) -> Any:
    max_tail_bytes = int(max_tail_bytes)
    if max_tail_bytes <= 0:
        yield from _iter_jsonl_rows(path)
        return
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            offset = max(0, size - max_tail_bytes)
            handle.seek(offset)
            if offset > 0:
                handle.readline()
            for raw_line in handle:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    yield value
    except FileNotFoundError:
        return


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


def _load_learner_metrics_latest(train_ref: Path) -> tuple[dict[str, Any] | None, str | None]:
    latest_path = runs.volume_path(RUNS_MOUNT, train_ref / "learner_metrics_latest.json")
    latest = _load_json(latest_path)
    if latest is None:
        return None, None
    error = latest.get("error") if isinstance(latest, dict) else None
    if error:
        return None, str(error)
    return latest, None


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


def _first_mapping(value: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, dict):
            return candidate
    return {}


def _mean_numeric(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for row in rows:
        for key in keys:
            value = _safe_float(row.get(key))
            if value is not None:
                values.append(value)
                break
    if not values:
        return None
    return sum(values) / len(values)


def _sum_count_mappings(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for key in keys:
            mapping = _normalise_counts(row.get(key))
            if mapping:
                counts.update(mapping)
                break
    return dict(sorted(counts.items()))


def _sum_action_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(_normalise_counts(row.get("action_histogram")))
    return dict(sorted(counts.items()))


def _outcome_histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        outcome = (
            row.get("outcome")
            or row.get("terminal_reason")
            or row.get("death_cause_name")
            or row.get("death_cause")
            or row.get("error_type")
            or ("ok" if row.get("ok") else "failed")
        )
        counts[str(outcome)] += 1
    return dict(sorted(counts.items()))


def _terminal_histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    mapped = _sum_count_mappings(
        rows,
        (
            "terminal_cause_histogram",
            "terminal_reason_histogram",
            "terminal_histogram",
            "terminal_reasons",
        ),
    )
    if mapped:
        return mapped
    counts: Counter[str] = Counter()
    for row in rows:
        cause = (
            row.get("terminal_cause")
            or row.get("terminal_reason")
            or row.get("death_cause_name")
            or row.get("death_cause")
            or row.get("error_type")
        )
        if cause:
            counts[str(cause)] += 1
    return dict(sorted(counts.items()))


def _normalised_histogram_entropy(counts: dict[str, int]) -> float | None:
    total = sum(count for count in counts.values() if count > 0)
    positive = [count for count in counts.values() if count > 0]
    if total <= 0 or len(positive) <= 1:
        return 0.0 if positive else None
    entropy = -sum((count / total) * math.log(count / total) for count in positive)
    return entropy / math.log(len(positive))


def _mean_numeric_mapping(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> dict[str, float]:
    sums: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for row in rows:
        mapping = _first_mapping(row, keys)
        for key, raw_value in mapping.items():
            value = _safe_float(raw_value)
            if value is None:
                continue
            sums[str(key)] += value
            counts[str(key)] += 1
    return {key: sums[key] / counts[key] for key in sorted(sums) if counts[key] > 0}


def _eval_checkpoint_extra_fields(
    aggregate: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    for key in (
        "mean_reward",
        "reward_mean",
        "mean_training_reward",
        "training_reward",
        "mean_bonus_count",
        "bonus_count",
        "mean_bonus_pickup_count",
        "bonus_pickup_count",
        "mean_bonus_reward",
        "bonus_reward",
        "bonus_pickup_reward",
        "action_entropy",
        "failure_rate",
    ):
        value = _safe_float(aggregate.get(key))
        if value is not None:
            extras[key] = value
    for key in ("eval_health", "health"):
        if aggregate.get(key) is not None:
            extras[key] = aggregate.get(key)
    for key in ("reward_components", "mean_reward_components"):
        value = aggregate.get(key)
        if isinstance(value, dict):
            extras[key] = value

    mean_training_reward = _mean_numeric(
        rows,
        ("training_reward", "mean_training_reward", "mean_reward", "reward_mean"),
    )
    if "mean_training_reward" not in extras and mean_training_reward is not None:
        extras["mean_training_reward"] = mean_training_reward
    mean_reward = _mean_numeric(rows, ("mean_reward", "reward_mean"))
    if "mean_reward" not in extras and mean_reward is not None:
        extras["mean_reward"] = mean_reward
    mean_bonus_count = _mean_numeric(rows, ("bonus_count", "mean_bonus_count"))
    if "mean_bonus_count" not in extras and mean_bonus_count is not None:
        extras["mean_bonus_count"] = mean_bonus_count
    mean_bonus_pickup_count = _mean_numeric(
        rows,
        ("bonus_pickup_count", "mean_bonus_pickup_count"),
    )
    if "mean_bonus_pickup_count" not in extras and mean_bonus_pickup_count is not None:
        extras["mean_bonus_pickup_count"] = mean_bonus_pickup_count
    mean_bonus_reward = _mean_numeric(
        rows,
        ("bonus_reward", "mean_bonus_reward", "bonus_pickup_reward"),
    )
    if "mean_bonus_reward" not in extras and mean_bonus_reward is not None:
        extras["mean_bonus_reward"] = mean_bonus_reward

    reward_components = _first_mapping(
        aggregate,
        ("mean_reward_components", "reward_components"),
    )
    if not reward_components:
        reward_components = _mean_numeric_mapping(
            rows,
            ("reward_components", "mean_reward_components"),
        )
    if reward_components and "mean_reward_components" not in extras:
        extras["mean_reward_components"] = reward_components

    action_histogram = _first_mapping(
        aggregate,
        ("action_histogram", "ego_action_histogram", "selected_action_histogram"),
    )
    for key in ("ego_action_histogram", "selected_action_histogram"):
        value = aggregate.get(key)
        if isinstance(value, dict):
            extras[key] = {
                str(item_key): item_value
                for item_key, item_value in sorted(
                    value.items(),
                    key=lambda item: str(item[0]),
                )
            }
    if not action_histogram:
        action_histogram = _sum_count_mappings(
            rows,
            ("action_histogram", "ego_action_histogram", "selected_action_histogram"),
        )
    if action_histogram:
        extras["action_histogram"] = {
            str(key): value
            for key, value in sorted(action_histogram.items(), key=lambda item: str(item[0]))
        }
        if "action_entropy" not in extras:
            extras["action_entropy"] = _normalised_histogram_entropy(
                _normalise_counts(action_histogram)
            )

    terminal_histogram = _first_mapping(
        aggregate,
        (
            "terminal_cause_histogram",
            "terminal_reason_histogram",
            "terminal_histogram",
            "terminal_reasons",
        ),
    )
    for key in ("terminal_reason_histogram", "terminal_histogram", "terminal_reasons"):
        value = aggregate.get(key)
        if isinstance(value, dict):
            extras[key] = {
                str(item_key): item_value
                for item_key, item_value in sorted(
                    value.items(),
                    key=lambda item: str(item[0]),
                )
            }
    if not terminal_histogram:
        terminal_histogram = _terminal_histogram(rows)
    if terminal_histogram:
        extras["terminal_cause_histogram"] = {
            str(key): value
            for key, value in sorted(
                terminal_histogram.items(),
                key=lambda item: str(item[0]),
            )
        }

    ok_count = _safe_float(aggregate.get("ok_count"))
    failure_count = _safe_float(aggregate.get("failure_count"))
    if "failure_rate" not in extras and failure_count is not None:
        denominator = (ok_count or 0.0) + failure_count
        if denominator > 0:
            extras["failure_rate"] = failure_count / denominator
    if "eval_health" not in extras and failure_count is not None:
        extras["eval_health"] = "has_failures" if failure_count > 0 else "ok"

    return extras


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
            "latest_eval_checkpoint": None,
            "latest_eval_mean_steps": None,
            "latest_eval_top_action": None,
            "latest_eval_action_fraction": None,
            "latest_eval_collapsed": None,
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
        action_counts = _sum_action_counts(rows)
        action_summary = _action_summary(
            action_counts,
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
                "outcome_histogram": aggregate.get("outcome_histogram")
                if isinstance(aggregate.get("outcome_histogram"), dict)
                else _outcome_histogram(rows),
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
                **_eval_checkpoint_extra_fields(aggregate, rows),
            }
        )

    latest_path, latest_manifest = manifests[-1]
    latest_checkpoint = checkpoint_rows[-1] if checkpoint_rows else {}
    latest_action = (
        latest_checkpoint.get("action_summary")
        if isinstance(latest_checkpoint.get("action_summary"), dict)
        else {}
    )
    return {
        "eval_manifest_count": len(manifests),
        "latest_eval_manifest_ref": latest_path.relative_to(RUNS_MOUNT).as_posix(),
        "latest_eval_created_at": latest_manifest.get("created_at"),
        "latest_eval_checkpoint": latest_checkpoint.get("checkpoint"),
        "latest_eval_mean_steps": latest_checkpoint.get("mean_steps"),
        "latest_eval_top_action": latest_action.get("top_action"),
        "latest_eval_action_fraction": latest_action.get("top_action_fraction"),
        "latest_eval_collapsed": latest_action.get("collapsed"),
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


def _assignment_refresh_rollup(run_id: str, attempt_id: str) -> dict[str, Any]:
    train_ref = runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
    refresh_ref = train_ref / "opponent_assignment_refresh_events.jsonl"
    events = _load_jsonl_rows(runs.volume_path(RUNS_MOUNT, refresh_ref))
    if not events:
        return {
            "assignment_refresh_event_count": 0,
            "assignment_refresh_events_ref": refresh_ref.as_posix(),
            "assignment_refresh_latest_decision": None,
            "assignment_refresh_latest_sha256": None,
            "assignment_refresh_applied_count": 0,
            "assignment_refresh_latest_applied_sha256": None,
        }
    applied = [event for event in events if event.get("decision") == "applied"]
    latest = events[-1]
    latest_applied = applied[-1] if applied else {}
    return {
        "assignment_refresh_event_count": len(events),
        "assignment_refresh_events_ref": refresh_ref.as_posix(),
        "assignment_refresh_latest_decision": latest.get("decision"),
        "assignment_refresh_latest_reason": latest.get("reason"),
        "assignment_refresh_latest_train_iter": latest.get("train_iter"),
        "assignment_refresh_latest_sha256": (
            latest.get("assignment_sha256") or latest.get("pending_assignment_sha256")
        ),
        "assignment_refresh_applied_count": len(applied),
        "assignment_refresh_latest_applied_train_iter": latest_applied.get("train_iter"),
        "assignment_refresh_latest_applied_sha256": latest_applied.get("assignment_sha256"),
        "assignment_refresh_latest_applied_ref": latest_applied.get("assignment_ref"),
    }


def _assignment_env_proof_rollup(
    run_id: str,
    attempt_id: str,
    *,
    target_assignment_shas: list[str],
    max_tail_bytes: int,
) -> dict[str, Any]:
    train_ref = runs.attempt_train_ref(TASK_ID, run_id, attempt_id)
    env_ref = train_ref / "env_steps.jsonl"
    env_path = runs.volume_path(RUNS_MOUNT, env_ref)
    target_shas = {sha for sha in target_assignment_shas if sha}
    target_prefixes = {sha[:8] for sha in target_shas}

    try:
        env_file_size = env_path.stat().st_size
    except FileNotFoundError:
        env_file_size = None

    scanned_row_count = 0
    assignment_row_count = 0
    target_row_count = 0
    target_provider_ok_count = 0
    target_provider_false_count = 0
    target_provider_null_count = 0
    target_checkpoint_ref_count = 0
    target_done_count = 0
    assignment_counts: Counter[str] = Counter()
    target_policy_kinds: Counter[str] = Counter()
    target_entry_names: Counter[str] = Counter()
    target_death_modes: Counter[str] = Counter()
    target_refresh_indices: Counter[str] = Counter()
    sample_rows: list[dict[str, Any]] = []

    for row in _iter_jsonl_tail_rows(env_path, max_tail_bytes=max_tail_bytes):
        scanned_row_count += 1
        sha = row.get("opponent_assignment_sha256")
        if sha:
            sha_text = str(sha)
            assignment_row_count += 1
            assignment_counts[sha_text[:8]] += 1
        else:
            sha_text = ""
        if target_shas and sha_text not in target_shas:
            continue
        if not target_shas and not sha_text:
            continue

        target_row_count += 1
        provider_ok = row.get("opponent_provider_load_ok")
        if provider_ok is True:
            target_provider_ok_count += 1
        elif provider_ok is False:
            target_provider_false_count += 1
        else:
            target_provider_null_count += 1
        if row.get("opponent_checkpoint_ref"):
            target_checkpoint_ref_count += 1
        if row.get("done") is True:
            target_done_count += 1
        for counter, key in (
            (target_policy_kinds, "opponent_policy_kind"),
            (target_entry_names, "opponent_mixture_entry_name"),
            (target_death_modes, "opponent_death_mode"),
            (target_refresh_indices, "opponent_assignment_refresh_index"),
        ):
            value = row.get(key)
            if value is not None:
                counter[str(value)] += 1
        if len(sample_rows) < 3:
            sample_rows.append(
                {
                    "opponent_assignment_sha256": sha_text[:8] if sha_text else None,
                    "opponent_assignment_refresh_index": row.get(
                        "opponent_assignment_refresh_index"
                    ),
                    "opponent_mixture_entry_name": row.get("opponent_mixture_entry_name"),
                    "opponent_policy_kind": row.get("opponent_policy_kind"),
                    "opponent_checkpoint_ref": row.get("opponent_checkpoint_ref"),
                    "opponent_provider_load_ok": row.get("opponent_provider_load_ok"),
                    "opponent_provider_load_candidate": row.get(
                        "opponent_provider_load_candidate"
                    ),
                    "decision_source_frames": row.get("decision_source_frames"),
                    "done": row.get("done"),
                    "terminal_reason": row.get("terminal_reason"),
                }
            )

    return {
        "assignment_env_proof_ref": env_ref.as_posix(),
        "assignment_env_proof_target_sha_prefixes": sorted(target_prefixes),
        "assignment_env_proof_env_file_exists": env_path.exists(),
        "assignment_env_proof_env_file_size_bytes": env_file_size,
        "assignment_env_proof_tail_bytes": int(max_tail_bytes),
        "assignment_env_proof_scanned_row_count": scanned_row_count,
        "assignment_env_proof_assignment_row_count": assignment_row_count,
        "assignment_env_proof_target_row_count": target_row_count,
        "assignment_env_proof_target_provider_ok_count": target_provider_ok_count,
        "assignment_env_proof_target_provider_false_count": target_provider_false_count,
        "assignment_env_proof_target_provider_null_count": target_provider_null_count,
        "assignment_env_proof_target_checkpoint_ref_count": target_checkpoint_ref_count,
        "assignment_env_proof_target_done_count": target_done_count,
        "assignment_env_proof_assignment_sha_counts": dict(sorted(assignment_counts.items())),
        "assignment_env_proof_target_policy_kinds": dict(sorted(target_policy_kinds.items())),
        "assignment_env_proof_target_entry_names": dict(sorted(target_entry_names.items())),
        "assignment_env_proof_target_death_modes": dict(sorted(target_death_modes.items())),
        "assignment_env_proof_target_refresh_indices": dict(
            sorted(target_refresh_indices.items())
        ),
        "assignment_env_proof_sample_rows": sample_rows,
    }


def _assignment_proof_status(
    run_id: str,
    *,
    attempt_id: str | None,
    target_assignment_shas: list[str],
    max_tail_bytes: int,
) -> dict[str, Any]:
    resolved_attempt_id = (
        attempt_id or _latest_attempt_id_for_run(run_id) or _attempt_id_for_run(run_id)
    )
    return {
        "run_id": run_id,
        "attempt_id": resolved_attempt_id,
        **_assignment_refresh_rollup(run_id, resolved_attempt_id),
        **_assignment_env_proof_rollup(
            run_id,
            resolved_attempt_id,
            target_assignment_shas=target_assignment_shas,
            max_tail_bytes=max_tail_bytes,
        ),
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
                "opponent_mixture_enabled": summary.get("opponent_mixture_enabled"),
                "opponent_mixture_entry_name": summary.get("opponent_mixture_entry_name"),
                "opponent_mixture_age_label": summary.get("opponent_mixture_age_label"),
                "opponent_mixture_entry_weight": summary.get("opponent_mixture_entry_weight"),
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
        "latest_gif_opponent_mixture_enabled": latest_summary.get("opponent_mixture_enabled"),
        "latest_gif_opponent_mixture_entry_name": latest_summary.get("opponent_mixture_entry_name"),
        "latest_gif_opponent_mixture_age_label": latest_summary.get("opponent_mixture_age_label"),
        "latest_gif_opponent_mixture_entry_weight": latest_summary.get(
            "opponent_mixture_entry_weight"
        ),
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
        "heartbeat_error": heartbeat.get("error") if isinstance(heartbeat, dict) else None,
        "heartbeat_exception": (
            heartbeat.get("exception") if isinstance(heartbeat, dict) else None
        ),
        "heartbeat_message": (
            heartbeat.get("message")
            or heartbeat.get("status_message")
            or heartbeat.get("failure_reason")
            if isinstance(heartbeat, dict)
            else None
        ),
        "heartbeat_updated_at": (
            heartbeat.get("updated_at")
            or heartbeat.get("timestamp")
            if isinstance(heartbeat, dict)
            else None
        ),
        "progress_missing_reason": missing_reason,
        **_assignment_refresh_rollup(run_id, attempt_id),
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
        attempt_train_root = runs.volume_path(
            RUNS_MOUNT,
            runs.attempt_train_ref(TASK_ID, run_id, attempt_id),
        )
        checkpoint_dirs.extend(
            lz_checkpoints.lightzero_exp_checkpoint_dirs(
                attempt_train_root / "lightzero_exp"
            )
        )
    artifacts: list[dict[str, Any]] = []
    for candidate in lz_checkpoints.collect_lightzero_iteration_checkpoints(
        checkpoint_dirs
    ):
        artifacts.append(
            {
                "checkpoint": f"iteration_{candidate.iteration}",
                "iteration": candidate.iteration,
                "mtime": candidate.mtime,
                "size_bytes": candidate.size_bytes,
                "path": str(candidate.path),
                "exp_dir_name": candidate.exp_dir_name,
            }
        )
    if not artifacts:
        return {"checkpoint_count": 0, "latest_checkpoint": None, "checkpoints": []}
    artifacts.sort(
        key=lambda item: (
            item["iteration"],
            item["mtime"] if item["mtime"] is not None else -1.0,
            item["size_bytes"] if item["size_bytes"] is not None else -1,
            item["path"],
        )
    )
    latest = artifacts[-1]
    return {
        "checkpoint_count": len(artifacts),
        "latest_checkpoint": latest["checkpoint"],
        "latest_checkpoint_mtime": latest["mtime"],
        "checkpoints": artifacts,
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
    progress_error = progress.get("error") if isinstance(progress, dict) else None
    progress_readable = progress is not None and progress_error is None
    learner_metrics_latest, learner_metrics_error = _load_learner_metrics_latest(train_ref)
    learner_metrics_path = runs.volume_path(RUNS_MOUNT, train_ref / "learner_metrics.jsonl")
    learner_metrics_rows = _load_jsonl_rows(learner_metrics_path)
    checkpoints = _checkpoint_summary(run_id, resolved_attempt_id)
    eval_rollup = _eval_manifest_rollup(
        run_id,
        resolved_attempt_id,
        collapse_threshold=collapse_threshold,
    )
    train_artifacts = _train_artifact_rollup(
        run_id,
        resolved_attempt_id,
        progress_exists=progress_readable,
        collapse_threshold=collapse_threshold,
    )
    row: dict[str, Any] = {
        "run_id": run_id,
        "short_name": run_id.removeprefix("curvytron-two-seat-selfplay-"),
        "attempt_id": resolved_attempt_id,
        "progress_exists": progress_readable,
        "progress_ref": (train_ref / "progress_latest.json").as_posix(),
        "progress_error": progress_error,
        "learner_metrics_latest_exists": learner_metrics_latest is not None,
        "learner_metrics_latest_ref": (train_ref / "learner_metrics_latest.json").as_posix(),
        "learner_metrics_ref": (train_ref / "learner_metrics.jsonl").as_posix(),
        "learner_metrics_error": learner_metrics_error,
        "learner_metrics_point_count": len(learner_metrics_rows),
        **checkpoints,
        **eval_rollup,
        **train_artifacts,
    }
    if progress is None or progress_error is not None:
        learner = learner_metrics_latest if isinstance(learner_metrics_latest, dict) else {}
        numeric_metrics = learner.get("numeric_metrics")
        if not isinstance(numeric_metrics, dict):
            numeric_metrics = {}
        if progress_error is not None:
            row["progress_missing_reason"] = "progress_latest_unreadable"
        row.update(
            {
                "event": "missing" if progress is None else "unreadable",
                "iteration": None,
                "mean_steps": None,
                "max_steps": None,
                "completed_episodes": None,
                "model_changed": learner.get("model_parameters_changed"),
                "learner_train_call_index": learner.get("learner_train_call_index"),
                "learner_train_iter_before": learner.get("train_iter_before"),
                "learner_train_iter_after": learner.get("train_iter_after"),
                "learner_train_iter_delta": learner.get("train_iter_delta"),
                "learner_collector_envstep": learner.get("collector_envstep"),
                "learner_elapsed_sec": _safe_float(learner.get("elapsed_sec")),
                "learner_numeric_metric_count": len(numeric_metrics),
                "learner_numeric_metrics": numeric_metrics,
                "problems": None,
                "top_action": None,
                "top_action_fraction": None,
                "collapsed": None,
            }
        )
        return row
    learner = progress.get("last_learner")
    if not isinstance(learner, dict):
        learner = learner_metrics_latest if isinstance(learner_metrics_latest, dict) else {}
    numeric_metrics = learner.get("numeric_metrics")
    if not isinstance(numeric_metrics, dict):
        numeric_metrics = {}
    row.update(
        {
            "event": progress.get("event"),
            "iteration": progress.get("iteration"),
            "mean_steps": _safe_float(progress.get("mean_completed_episode_steps")),
            "max_steps": progress.get("max_completed_episode_steps"),
            "completed_episodes": progress.get("completed_episode_count"),
            "model_changed": learner.get("model_parameters_changed"),
            "learner_train_call_index": learner.get("learner_train_call_index"),
            "learner_train_iter_before": learner.get("train_iter_before"),
            "learner_train_iter_after": learner.get("train_iter_after"),
            "learner_train_iter_delta": learner.get("train_iter_delta"),
            "learner_collector_envstep": learner.get("collector_envstep"),
            "learner_elapsed_sec": _safe_float(learner.get("elapsed_sec")),
            "learner_numeric_metric_count": len(numeric_metrics),
            "learner_numeric_metrics": numeric_metrics,
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


def _run_fast_status(run_id: str, *, attempt_id: str | None) -> dict[str, Any]:
    resolved_attempt_id = (
        attempt_id or _latest_attempt_id_for_run(run_id) or _attempt_id_for_run(run_id)
    )
    train_ref = runs.attempt_train_ref(TASK_ID, run_id, resolved_attempt_id)
    train_root = runs.volume_path(RUNS_MOUNT, train_ref)
    heartbeat_ref = train_ref / "status_heartbeat.json"
    heartbeat = _load_json(runs.volume_path(RUNS_MOUNT, heartbeat_ref))
    progress_ref = train_ref / "progress_latest.json"
    progress = _load_json(runs.volume_path(RUNS_MOUNT, progress_ref))
    progress_error = progress.get("error") if isinstance(progress, dict) else None
    progress_readable = progress is not None and progress_error is None
    learner_metrics_latest, learner_metrics_error = _load_learner_metrics_latest(train_ref)
    checkpoints = _checkpoint_summary(run_id, resolved_attempt_id)
    learner = progress.get("last_learner") if isinstance(progress, dict) else None
    if not isinstance(learner, dict):
        learner = learner_metrics_latest if isinstance(learner_metrics_latest, dict) else {}
    numeric_metrics = learner.get("numeric_metrics")
    if not isinstance(numeric_metrics, dict):
        numeric_metrics = {}
    if progress_readable:
        progress_missing_reason = None
    elif not train_root.exists():
        progress_missing_reason = "train_root_absent"
    elif heartbeat is not None:
        progress_missing_reason = "progress_latest_absent_after_train_heartbeat"
    else:
        progress_missing_reason = "train_root_exists_progress_latest_absent"
    return {
        "run_id": run_id,
        "short_name": run_id.removeprefix("curvytron-two-seat-selfplay-"),
        "attempt_id": resolved_attempt_id,
        "train_root_exists": train_root.exists(),
        "status_heartbeat_exists": heartbeat is not None,
        "status_heartbeat_ref": heartbeat_ref.as_posix(),
        "train_status": heartbeat.get("status") if isinstance(heartbeat, dict) else None,
        "train_stage": heartbeat.get("stage") if isinstance(heartbeat, dict) else None,
        "heartbeat_error": heartbeat.get("error") if isinstance(heartbeat, dict) else None,
        "heartbeat_exception": (
            heartbeat.get("exception") if isinstance(heartbeat, dict) else None
        ),
        "heartbeat_message": (
            heartbeat.get("message")
            or heartbeat.get("status_message")
            or heartbeat.get("failure_reason")
            if isinstance(heartbeat, dict)
            else None
        ),
        "heartbeat_updated_at": (
            heartbeat.get("updated_at")
            or heartbeat.get("timestamp")
            if isinstance(heartbeat, dict)
            else None
        ),
        "progress_exists": progress_readable,
        "progress_ref": progress_ref.as_posix(),
        "progress_error": progress_error,
        "progress_missing_reason": progress_missing_reason,
        "event": progress.get("event") if isinstance(progress, dict) else None,
        "iteration": progress.get("iteration") if isinstance(progress, dict) else None,
        "mean_steps": (
            progress.get("mean_completed_episode_steps") if isinstance(progress, dict) else None
        ),
        "max_steps": (
            progress.get("max_completed_episode_steps") if isinstance(progress, dict) else None
        ),
        "completed_episodes": (
            progress.get("completed_episode_count") if isinstance(progress, dict) else None
        ),
        "problem_count": progress.get("problem_count") if isinstance(progress, dict) else None,
        "model_changed": learner.get("model_parameters_changed"),
        "learner_train_call_index": learner.get("learner_train_call_index"),
        "learner_train_iter_before": learner.get("train_iter_before"),
        "learner_train_iter_after": learner.get("train_iter_after"),
        "learner_train_iter_delta": learner.get("train_iter_delta"),
        "learner_collector_envstep": learner.get("collector_envstep"),
        "learner_elapsed_sec": _safe_float(learner.get("elapsed_sec")),
        "learner_numeric_metric_count": len(numeric_metrics),
        "learner_metrics_latest_exists": learner_metrics_latest is not None,
        "learner_metrics_error": learner_metrics_error,
        "checkpoint_count": checkpoints.get("checkpoint_count"),
        "latest_checkpoint": checkpoints.get("latest_checkpoint"),
        "latest_checkpoint_mtime": checkpoints.get("latest_checkpoint_mtime"),
    }


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
    heartbeat = _load_json(runs.volume_path(RUNS_MOUNT, train_ref / "status_heartbeat.json"))
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


def _chunks(items: list[str], size: int) -> list[tuple[int, list[str]]]:
    chunk_size = max(1, int(size))
    return [
        (start, items[start : start + chunk_size])
        for start in range(0, len(items), chunk_size)
    ]


def _remote_rows_in_chunks(
    fn: Any,
    selected_run_ids: list[str],
    selected_attempt_ids: list[str],
    *remote_args: Any,
    chunk_size: int,
    chunk_workers: int,
) -> list[dict[str, Any]]:
    if not selected_run_ids:
        return []
    if chunk_size <= 0 or len(selected_run_ids) <= chunk_size:
        return fn.remote(selected_run_ids, selected_attempt_ids, *remote_args)

    chunks = _chunks(selected_run_ids, chunk_size)
    attempt_chunks = {
        start: selected_attempt_ids[start : start + len(run_chunk)]
        for start, run_chunk in chunks
    }
    max_workers = max(1, min(int(chunk_workers), len(chunks)))
    rows_by_start: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                fn.remote,
                run_chunk,
                attempt_chunks[start],
                *remote_args,
            ): start
            for start, run_chunk in chunks
        }
        for future in as_completed(futures):
            start = futures[future]
            rows_by_start[start] = future.result()

    merged: list[dict[str, Any]] = []
    for start, _run_chunk in chunks:
        merged.extend(rows_by_start.get(start, []))
    return merged


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


def _print_fast_table(rows: list[dict[str, Any]]) -> None:
    fields = [
        "short_name",
        "train_status",
        "train_stage",
        "iteration",
        "learner_train_iter_after",
        "learner_collector_envstep",
        "mean_steps",
        "max_steps",
        "checkpoint_count",
        "latest_checkpoint",
        "progress_exists",
        "status_heartbeat_exists",
        "model_changed",
        "problem_count",
        "progress_missing_reason",
    ]
    print("\t".join(fields))
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            if field == "mean_steps":
                value = _format_float(value)
            elif value is None:
                value = ""
            values.append(str(value))
        print("\t".join(values))


def _print_fast_summary(rows: list[dict[str, Any]]) -> None:
    checkpoint_counts = [
        int(row["checkpoint_count"])
        for row in rows
        if isinstance(row.get("checkpoint_count"), int)
    ]
    iterations = [
        int(row["iteration"])
        for row in rows
        if isinstance(row.get("iteration"), int)
    ]
    grid_counts: Counter[str] = Counter()
    for row in rows:
        run_id = str(row.get("run_id") or "")
        grid_counts[run_id.split("-", 1)[0] or "unknown"] += 1
    summary = {
        "row_count": len(rows),
        "grid_counts": dict(sorted(grid_counts.items())),
        "train_status_counts": dict(
            sorted(Counter(str(row.get("train_status") or "missing") for row in rows).items())
        ),
        "train_stage_counts": dict(
            sorted(Counter(str(row.get("train_stage") or "missing") for row in rows).items())
        ),
        "heartbeat_count": sum(1 for row in rows if row.get("status_heartbeat_exists")),
        "progress_latest_count": sum(1 for row in rows if row.get("progress_exists")),
        "model_changed_true_count": sum(1 for row in rows if row.get("model_changed") is True),
        "checkpoint_count_min": min(checkpoint_counts) if checkpoint_counts else None,
        "checkpoint_count_max": max(checkpoint_counts) if checkpoint_counts else None,
        "checkpoint_count_sum": sum(checkpoint_counts),
        "iteration_min": min(iterations) if iterations else None,
        "iteration_max": max(iterations) if iterations else None,
        "failed_run_ids": [
            str(row.get("run_id"))
            for row in rows
            if str(row.get("train_status") or "") == "failed"
        ],
        "lowest_checkpoint_runs": [
            {
                "run_id": row.get("run_id"),
                "checkpoint_count": row.get("checkpoint_count"),
                "latest_checkpoint": row.get("latest_checkpoint"),
                "train_status": row.get("train_status"),
            }
            for row in sorted(
                rows,
                key=lambda item: (
                    int(item.get("checkpoint_count") or 0),
                    str(item.get("run_id") or ""),
                ),
            )[:8]
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


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
        "first_outcomes",
        "best_iter",
        "best_mean",
        "best_outcomes",
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
        checkpoints = (
            [checkpoint for checkpoint in raw_checkpoints if isinstance(checkpoint, dict)]
            if isinstance(raw_checkpoints, list)
            else []
        )
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
            _format_counts(first.get("outcome_histogram")),
            str(best.get("iteration") if best.get("iteration") is not None else ""),
            _format_float(best.get("mean_steps")),
            _format_counts(best.get("outcome_histogram")),
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
def curvytron_run_fast_status(
    run_ids: list[str],
    attempt_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    resolved_attempt_ids = attempt_ids or []
    rows: list[dict[str, Any]] = []
    for index, run_id in enumerate(run_ids):
        attempt_id = (
            resolved_attempt_ids[index]
            if index < len(resolved_attempt_ids) and resolved_attempt_ids[index]
            else None
        )
        rows.append(_run_fast_status(run_id, attempt_id=attempt_id))
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


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=5 * 60, cpu=1.0)
def curvytron_assignment_proof(
    run_ids: list[str],
    attempt_ids: list[str] | None = None,
    target_assignment_shas: list[str] | None = None,
    max_tail_bytes: int = DEFAULT_ASSIGNMENT_PROOF_TAIL_BYTES,
) -> list[dict[str, Any]]:
    resolved_attempt_ids = attempt_ids or []
    target_shas = target_assignment_shas or []
    rows: list[dict[str, Any]] = []
    for index, run_id in enumerate(run_ids):
        attempt_id = (
            resolved_attempt_ids[index]
            if index < len(resolved_attempt_ids) and resolved_attempt_ids[index]
            else None
        )
        rows.append(
            _assignment_proof_status(
                run_id,
                attempt_id=attempt_id,
                target_assignment_shas=target_shas,
                max_tail_bytes=int(max_tail_bytes),
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
    target_assignment_shas: str | None = None,
    assignment_proof_tail_bytes: int = DEFAULT_ASSIGNMENT_PROOF_TAIL_BYTES,
    chunk_size: int = 16,
    chunk_workers: int = 8,
) -> None:
    if run_ids:
        selected_run_ids = _split_csv(run_ids)
        preset_attempt_ids: list[str] = []
    else:
        selected_run_ids, preset_attempt_ids = _preset_run_ids(preset)
    selected_attempt_ids = _split_csv(attempt_ids) or preset_attempt_ids
    if output in {"fast-table", "fast-json", "fast-summary"}:
        rows = _remote_rows_in_chunks(
            curvytron_run_fast_status,
            selected_run_ids,
            selected_attempt_ids,
            chunk_size=chunk_size,
            chunk_workers=chunk_workers,
        )
    elif output in {"eval-summary", "eval-json"}:
        rows = _remote_rows_in_chunks(
            curvytron_run_eval_curves,
            selected_run_ids,
            selected_attempt_ids,
            collapse_threshold,
            chunk_size=chunk_size,
            chunk_workers=chunk_workers,
        )
    elif output in {"assignment-proof-json"}:
        rows = _remote_rows_in_chunks(
            curvytron_assignment_proof,
            selected_run_ids,
            selected_attempt_ids,
            _split_csv(target_assignment_shas),
            int(assignment_proof_tail_bytes),
            chunk_size=chunk_size,
            chunk_workers=chunk_workers,
        )
    elif output in {"curve-json", "curve-table", "curve-summary"}:
        rows = _remote_rows_in_chunks(
            curvytron_run_curves,
            selected_run_ids,
            selected_attempt_ids,
            collapse_threshold,
            chunk_size=chunk_size,
            chunk_workers=chunk_workers,
        )
    else:
        rows = _remote_rows_in_chunks(
            curvytron_run_status,
            selected_run_ids,
            selected_attempt_ids,
            collapse_threshold,
            chunk_size=chunk_size,
            chunk_workers=chunk_workers,
        )
    if output == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif output == "fast-json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif output == "fast-table":
        _print_fast_table(rows)
    elif output == "fast-summary":
        _print_fast_summary(rows)
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
    elif output == "eval-json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif output == "assignment-proof-json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        raise ValueError(
            "output must be table, json, fast-table, fast-summary, fast-json, "
            "curve-table, curve-summary, curve-json, eval-summary, eval-json, "
            "or assignment-proof-json"
        )
