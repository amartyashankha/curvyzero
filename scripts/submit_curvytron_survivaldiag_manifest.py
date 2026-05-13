#!/usr/bin/env python3
"""Submit CurvyTron survivaldiag manifest rows into one deployed Modal app."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _call_id(call: Any) -> str | None:
    value = getattr(call, "object_id", None) or getattr(call, "id", None)
    return None if value is None else str(value)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must contain non-empty rows")
    return payload


def _selected_rows(manifest: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = list(manifest["rows"])
    if args.row_id:
        wanted = set(args.row_id)
        rows = [row for row in rows if str(row.get("row_id")) in wanted]
    if args.row_kind:
        wanted_kinds = set(args.row_kind)
        rows = [row for row in rows if str(row.get("row_kind")) in wanted_kinds]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("row filters selected no rows")
    seen: set[str] = set()
    for row in rows:
        run_id = str(row.get("run_id") or "")
        if not run_id:
            raise ValueError(f"row {row.get('row_id')} has empty run_id")
        if run_id in seen:
            raise ValueError(f"duplicate run_id in selected rows: {run_id}")
        seen.add(run_id)
    return rows


def _launch_row(row: dict[str, Any], *, app_name: str, dry_run: bool) -> dict[str, Any]:
    submission = row.get("deployed_app_submission")
    if not isinstance(submission, dict):
        raise ValueError(f"row {row.get('row_id')} lacks deployed_app_submission")
    train_function = str(submission.get("train_function") or "")
    poller_function = str(submission.get("poller_function") or "")
    train_kwargs = row.get("train_kwargs")
    poller_kwargs = row.get("poller_kwargs")
    if not train_function or not poller_function:
        raise ValueError(f"row {row.get('row_id')} lacks deployed function names")
    if not isinstance(train_kwargs, dict) or not isinstance(poller_kwargs, dict):
        raise ValueError(f"row {row.get('row_id')} lacks train/poller kwargs")

    record = {
        "row_id": row.get("row_id"),
        "label": row.get("label"),
        "run_id": row.get("run_id"),
        "attempt_id": row.get("attempt_id"),
        "app_name": app_name,
        "train_function": train_function,
        "poller_function": poller_function,
        "artifact_refs": row.get("artifact_refs"),
    }
    if dry_run:
        return {"status": "dry_run", **record}

    import modal

    poller_fn = modal.Function.from_name(app_name, poller_function)
    train_fn = modal.Function.from_name(app_name, train_function)
    poller_call = poller_fn.spawn(**poller_kwargs)
    train_call = train_fn.spawn(**train_kwargs)
    return {
        "status": "spawned",
        **record,
        "poller_function_call_id": _call_id(poller_call),
        "train_function_call_id": _call_id(train_call),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit rows from a CurvyTron survivaldiag manifest into the deployed "
            "trainer app. This avoids one ephemeral Modal app per row."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--app-name", default=None)
    parser.add_argument("--row-id", action="append", default=[])
    parser.add_argument("--row-kind", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = _load_manifest(args.manifest)
    rows = _selected_rows(manifest, args)
    app_name = args.app_name or str(manifest["guards"]["deployed_app_name"])
    dry_run = not args.allow_launch
    records = [_launch_row(row, app_name=app_name, dry_run=dry_run) for row in rows]
    payload = {
        "schema_id": "curvyzero_curvytron_survivaldiag_grouped_modal_submission/v0",
        "status": "dry_run" if dry_run else "submitted",
        "dry_run": dry_run,
        "app_name": app_name,
        "matrix_name": manifest.get("matrix_name"),
        "matrix_profile": manifest.get("matrix_profile"),
        "row_count": len(records),
        "records": records,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
