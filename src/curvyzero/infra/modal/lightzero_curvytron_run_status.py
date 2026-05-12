"""Compact status reader for CurvyTron LightZero training runs.

This reads Modal Volume progress files from inside one Modal function. It is
meant to replace noisy manual ``modal volume get`` polling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
    RUNS_MOUNT,
    TASK_ID,
    VOLUME_NAME,
    image,
    runs_volume,
)


APP_NAME = "curvyzero-lightzero-curvytron-run-status"
DEFAULT_COLLAPSE_THRESHOLD = 0.95

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

app = modal.App(APP_NAME)


def _attempt_id_for_run(run_id: str) -> str:
    prefix = "curvytron-two-seat-selfplay-"
    if run_id.startswith(prefix):
        return run_id.removeprefix(prefix)
    return "train"


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


def _checkpoint_summary(run_id: str) -> dict[str, Any]:
    checkpoint_dir = runs.volume_path(
        RUNS_MOUNT,
        runs.checkpoints_root_ref(TASK_ID, run_id) / "lightzero",
    )
    if not checkpoint_dir.is_dir():
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
    resolved_attempt_id = attempt_id or _attempt_id_for_run(run_id)
    train_ref = runs.attempt_train_ref(TASK_ID, run_id, resolved_attempt_id)
    latest_path = runs.volume_path(RUNS_MOUNT, train_ref / "progress_latest.json")
    progress = _load_json(latest_path)
    checkpoints = _checkpoint_summary(run_id)
    row: dict[str, Any] = {
        "run_id": run_id,
        "short_name": run_id.removeprefix("curvytron-two-seat-selfplay-"),
        "attempt_id": resolved_attempt_id,
        "progress_exists": progress is not None,
        "progress_ref": (train_ref / "progress_latest.json").as_posix(),
        **checkpoints,
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
            "effective_noop": _safe_float(
                progress.get("effective_action_noop_probability")
            ),
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
    resolved_attempt_id = attempt_id or _attempt_id_for_run(run_id)
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
            "effective_noop": _safe_float(
                progress.get("effective_action_noop_probability")
            ),
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
        **_checkpoint_summary(run_id),
    }


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


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
        "problems",
        "elapsed_sec",
    ]
    print("\t".join(fields))
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            if field in {"mean_steps", "top_action_fraction", "elapsed_sec"}:
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
        "points",
        "first_iter",
        "first_mean",
        "best_iter",
        "best_mean",
        "latest_iter",
        "latest_mean",
        "latest_max",
        "latest_top_action",
        "latest_top_fraction",
        "any_collapsed",
        "checkpoints",
        "latest_checkpoint",
    ]
    print("\t".join(fields))
    for row in rows:
        curve = row.get("curve", [])
        if not curve:
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
                        str(row.get("checkpoint_count")),
                        str(row.get("latest_checkpoint") or ""),
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
            str(len(curve)),
            str(first.get("iteration") or ""),
            _format_float(first.get("mean_steps")),
            str(best.get("iteration") or ""),
            _format_float(best.get("mean_steps")),
            str(latest.get("iteration") or ""),
            _format_float(latest.get("mean_steps")),
            str(latest.get("max_steps") or ""),
            str(latest.get("top_action") or ""),
            _format_float(latest.get("top_action_fraction")),
            str(any(point.get("collapsed") for point in curve)),
            str(row.get("checkpoint_count")),
            str(row.get("latest_checkpoint") or ""),
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
    elif preset == "live6":
        selected_run_ids = list(LIVE6_RUN_IDS)
    elif preset == "live11":
        selected_run_ids = list(LIVE11_RUN_IDS)
    else:
        raise ValueError("use --preset live6, --preset live11, or pass --run-ids")
    selected_attempt_ids = _split_csv(attempt_ids)
    if output in {"curve-json", "curve-table", "curve-summary"}:
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
    else:
        raise ValueError(
            "output must be table, json, curve-table, curve-summary, or curve-json"
        )
