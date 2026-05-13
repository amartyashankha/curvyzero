#!/usr/bin/env python3
"""Launch and collect CurvyTron optimizer profile manifest rows.

This script keeps profile readback local and explicit:

1. Launch rows whose commands print a Modal `function_call_id`.
2. Store those launch records under `artifacts/local/...`.
3. Collect returned Modal function results by call id.

It does not rely on `summary.json` being committed to the main run volume.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_MANIFEST = Path(
    "artifacts/local/curvytron_optimizer_profile_manifests/"
    "opt-stock-frozen-profile-first-wave-20260512e.json"
)
DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_optimizer_profile_results")


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _selected_rows(manifest: dict[str, Any], row_ids: set[str] | None) -> list[dict[str, Any]]:
    rows = list(manifest.get("rows") or [])
    if row_ids is None:
        return rows
    selected = [row for row in rows if str(row.get("row_id")) in row_ids]
    missing = sorted(row_ids.difference(str(row.get("row_id")) for row in selected))
    if missing:
        raise SystemExit(f"unknown row id(s): {', '.join(missing)}")
    return selected


def _parse_rows(raw: list[str]) -> set[str] | None:
    if not raw:
        return None
    rows: set[str] = set()
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if part:
                rows.add(part.zfill(2) if part.isdigit() else part)
    return rows


def _extract_last_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append((index, end, value))
    if not candidates:
        raise ValueError("no JSON object found in command output")
    for _index, _end, value in reversed(candidates):
        if value.get("schema_id") or value.get("function_call_id"):
            return value
    return max(candidates, key=lambda item: item[1])[2]


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    from curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train import (
        _compact_train_result_for_output,
        _to_plain,
    )

    compact = _compact_train_result_for_output(_to_plain(result))
    if isinstance(compact, dict):
        return compact
    return {"result": compact}


def _launch_rows(
    *,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    launches_path: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        command = list(row.get("command") or [])
        command_text = str(row.get("command_text") or " ".join(command))
        if "--profile-spawn" not in command:
            raise SystemExit(
                f"row {row.get('row_id')} does not use --profile-spawn; "
                "refusing to launch through function-call readback"
            )
        record: dict[str, Any] = {
            "schema_id": "curvyzero_optimizer_profile_launch_record/v0",
            "created_at": _utc_timestamp(),
            "experiment_id": manifest.get("experiment_id"),
            "row_id": row.get("row_id"),
            "family": row.get("family"),
            "run_id": row.get("run_id"),
            "attempt_id": row.get("attempt_id"),
            "command_text": command_text,
            "dry_run": dry_run,
        }
        if dry_run:
            print(command_text)
            records.append(record)
            continue
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        record["returncode"] = completed.returncode
        record["stdout_tail"] = completed.stdout[-4000:]
        if completed.returncode != 0:
            record["status"] = "launch_failed"
            _append_jsonl(launches_path, record)
            records.append(record)
            continue
        try:
            payload = _extract_last_json_object(completed.stdout)
        except ValueError as exc:
            record["status"] = "launch_parse_failed"
            record["problem"] = str(exc)
            _append_jsonl(launches_path, record)
            records.append(record)
            continue
        record["status"] = payload.get("status")
        record["function_call_id"] = payload.get("function_call_id")
        record["launch_payload"] = payload
        _append_jsonl(launches_path, record)
        records.append(record)
        print(
            json.dumps(
                {
                    "row_id": row.get("row_id"),
                    "status": record["status"],
                    "function_call_id": record.get("function_call_id"),
                },
                sort_keys=True,
            )
        )
    return records


def _load_launch_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"launch record file not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _collect_rows(
    *,
    records: list[dict[str, Any]],
    row_ids: set[str] | None,
    output_dir: Path,
    timeout_sec: float | None,
) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if row_ids is None or str(record.get("row_id")) in row_ids
    ]
    results: list[dict[str, Any]] = []
    for record in selected:
        call_id = record.get("function_call_id")
        if not call_id:
            results.append(
                {
                    "row_id": record.get("row_id"),
                    "status": "missing_function_call_id",
                    "run_id": record.get("run_id"),
                    "attempt_id": record.get("attempt_id"),
                }
            )
            continue
        call = modal.FunctionCall.from_id(str(call_id))
        try:
            result = call.get(timeout=timeout_sec)
            compact = _compact_result(result if isinstance(result, dict) else {"result": result})
            status = "complete"
            problem = None
        except Exception as exc:  # pragma: no cover - remote failures are runtime data.
            result = None
            compact = None
            status = "collect_failed"
            problem = f"{type(exc).__name__}: {exc}"
        row_id = str(record.get("row_id"))
        payload = {
            "schema_id": "curvyzero_optimizer_profile_collected_result/v0",
            "collected_at": _utc_timestamp(),
            "row_id": row_id,
            "family": record.get("family"),
            "run_id": record.get("run_id"),
            "attempt_id": record.get("attempt_id"),
            "function_call_id": call_id,
            "status": status,
            "problem": problem,
            "compact": compact,
            "result": result,
        }
        _write_json(output_dir / f"row_{row_id}_result.json", payload)
        print(
            json.dumps(
                {
                    "row_id": row_id,
                    "status": status,
                    "steps": (compact or {}).get("counts", {}).get("env_steps_collected")
                    if isinstance(compact, dict)
                    else None,
                    "wall": (compact or {}).get("timers_sec", {}).get("train_muzero_wall")
                    if isinstance(compact, dict)
                    else None,
                    "problem": problem,
                },
                sort_keys=True,
            )
        )
        results.append(payload)
    _write_json(output_dir / "collected_results.json", results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rows", action="append", default=[])
    parser.add_argument(
        "--action",
        choices=("launch", "collect", "launch-and-collect"),
        default="launch-and-collect",
    )
    parser.add_argument("--launches-path", type=Path)
    parser.add_argument("--collect-timeout-sec", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = _load_json(args.manifest)
    row_ids = _parse_rows(args.rows)
    output_dir = args.output_root / str(manifest["experiment_id"])
    launches_path = args.launches_path or output_dir / "launches.jsonl"
    rows = _selected_rows(manifest, row_ids)

    records: list[dict[str, Any]] = []
    if args.action in {"launch", "launch-and-collect"}:
        records = _launch_rows(
            manifest=manifest,
            rows=rows,
            launches_path=launches_path,
            dry_run=args.dry_run,
        )
    if args.action in {"collect", "launch-and-collect"} and not args.dry_run:
        if not records:
            records = _load_launch_records(launches_path)
        _collect_rows(
            records=records,
            row_ids=row_ids,
            output_dir=output_dir,
            timeout_sec=args.collect_timeout_sec,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
