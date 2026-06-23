#!/usr/bin/env python3
"""Summarize Modal app-list capacity context before a CurvyTron Wave A launch."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCHEMA_ID = "curvyzero_curvytron_wave_a_capacity_audit/v0"
CURVY_TRAIN_APP_DESCRIPTION = "curvyzero-lightzero-curvytron-visual-survival-train-v2"
CURVY_STATUS_APP_DESCRIPTION = "curvyzero-lightzero-curvytron-run-status"
DETACHED_STATE = "ephemeral (detached)"


def _parse_task_count(value: Any) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return int(text)


def _load_app_list_from_input(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain the JSON list from modal app list --json")
    apps = [app for app in payload if isinstance(app, dict)]
    if len(apps) != len(payload):
        raise ValueError(f"{path} contains non-object app entries")
    return apps


def _load_app_list_from_modal(modal_bin: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [modal_bin, "app", "list", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "modal app list --json failed: "
            f"returncode={result.returncode}, stderr={result.stderr.strip()!r}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"modal app list returned invalid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise RuntimeError("modal app list JSON payload was not a list")
    apps = [app for app in payload if isinstance(app, dict)]
    if len(apps) != len(payload):
        raise RuntimeError("modal app list JSON contained non-object app entries")
    return apps


def _app_summary(app: dict[str, Any]) -> dict[str, Any]:
    return {
        "app_id": app.get("App ID"),
        "description": app.get("Description"),
        "state": app.get("State"),
        "tasks": _parse_task_count(app.get("Tasks")),
        "created_at": app.get("Created at"),
        "stopped_at": app.get("Stopped at"),
    }


def _matching_apps(apps: Sequence[dict[str, Any]], description: str) -> list[dict[str, Any]]:
    return [
        _app_summary(app)
        for app in apps
        if str(app.get("Description") or "") == description
    ]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if args.input is None:
        apps = _load_app_list_from_modal(args.modal_bin)
        input_path = None
        source = "modal app list --json"
    else:
        apps = _load_app_list_from_input(args.input)
        input_path = str(args.input)
        source = "input_json"

    task_counts = [_parse_task_count(app.get("Tasks")) for app in apps]
    state_counts = Counter(str(app.get("State") or "") for app in apps)
    description_counts = Counter(str(app.get("Description") or "") for app in apps)
    detached_apps = [
        _app_summary(app)
        for app in apps
        if str(app.get("State") or "") == DETACHED_STATE
    ]
    curvy_train_apps = _matching_apps(apps, CURVY_TRAIN_APP_DESCRIPTION)
    curvy_status_apps = _matching_apps(apps, CURVY_STATUS_APP_DESCRIPTION)

    total_tasks = sum(task_counts)
    detached_tasks = sum(app["tasks"] for app in detached_apps)
    curvy_train_tasks = sum(app["tasks"] for app in curvy_train_apps)
    curvy_status_tasks = sum(app["tasks"] for app in curvy_status_apps)
    projected_total_tasks = total_tasks + args.requested_h100_rows
    target_envelope = args.target_h100_envelope
    requested_plus_buffer = args.requested_h100_rows + args.reserved_h100_buffer
    max_additional_rows_under_task_proxy = max(0, target_envelope - total_tasks)
    full_launch_within_task_proxy = args.requested_h100_rows <= max_additional_rows_under_task_proxy

    if not curvy_train_apps:
        errors.append(
            {
                "message": "CurvyTron train app is absent from Modal app list",
                "description": CURVY_TRAIN_APP_DESCRIPTION,
            }
        )
    if not curvy_status_apps:
        errors.append(
            {
                "message": "CurvyTron status app is absent from Modal app list",
                "description": CURVY_STATUS_APP_DESCRIPTION,
            }
        )
    if curvy_train_tasks:
        warnings.append(
            {
                "message": "CurvyTron train app already has active tasks",
                "tasks": curvy_train_tasks,
            }
        )
    if curvy_status_tasks:
        warnings.append(
            {
                "message": "CurvyTron status app already has active tasks",
                "tasks": curvy_status_tasks,
            }
        )
    if requested_plus_buffer > target_envelope:
        errors.append(
            {
                "message": "requested rows plus reserved buffer exceeds target envelope",
                "requested_h100_rows": args.requested_h100_rows,
                "reserved_h100_buffer": args.reserved_h100_buffer,
                "target_h100_envelope": target_envelope,
            }
        )
    if projected_total_tasks > target_envelope:
        warnings.append(
            {
                "message": "current Modal task count plus requested rows exceeds the target H100 envelope proxy",
                "total_tasks": total_tasks,
                "requested_h100_rows": args.requested_h100_rows,
                "projected_total_tasks": projected_total_tasks,
                "target_h100_envelope": target_envelope,
                "note": "Modal task count is a coarse proxy and may include non-H100 work.",
            }
        )

    approval_recommendation = "capacity_proxy_clear"
    if warnings:
        approval_recommendation = "operator_capacity_review_required"
    if errors:
        approval_recommendation = "do_not_launch_until_fixed"

    snapshot_time = args.snapshot_time or datetime.now(timezone.utc).isoformat()
    return {
        "schema_id": SCHEMA_ID,
        "ok": not errors,
        "source": source,
        "input": input_path,
        "snapshot_time": snapshot_time,
        "requested_h100_rows": args.requested_h100_rows,
        "reserved_h100_buffer": args.reserved_h100_buffer,
        "target_h100_envelope": target_envelope,
        "requested_plus_buffer": requested_plus_buffer,
        "app_count": len(apps),
        "state_counts": dict(sorted(state_counts.items())),
        "top_descriptions": [
            {"description": description, "count": count}
            for description, count in description_counts.most_common(12)
        ],
        "total_tasks": total_tasks,
        "detached_running": len(detached_apps),
        "detached_tasks": detached_tasks,
        "projected_total_tasks": projected_total_tasks,
        "projected_task_proxy_excess": max(0, projected_total_tasks - target_envelope),
        "max_additional_rows_under_task_proxy": max_additional_rows_under_task_proxy,
        "full_launch_within_task_proxy": full_launch_within_task_proxy,
        "curvy_train_apps": curvy_train_apps,
        "curvy_status_apps": curvy_status_apps,
        "curvy_train_tasks": curvy_train_tasks,
        "curvy_status_tasks": curvy_status_tasks,
        "approval_recommendation": approval_recommendation,
        "warning_count": len(warnings),
        "error_count": len(errors),
        "warnings": warnings,
        "errors": errors,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="JSON output from modal app list --json; if omitted, run modal app list --json",
    )
    parser.add_argument("--modal-bin", default="modal")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--requested-h100-rows", type=int, default=90)
    parser.add_argument("--reserved-h100-buffer", type=int, default=10)
    parser.add_argument("--target-h100-envelope", type=int, default=100)
    parser.add_argument(
        "--snapshot-time",
        default=None,
        help="override snapshot timestamp for reproducible tests",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
