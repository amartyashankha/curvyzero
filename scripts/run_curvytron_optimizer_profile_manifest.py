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

DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_optimizer_profile_results")
CURRENT_MODAL_MODULE = (
    "curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train::main"
)
TOP_LEVEL_JSON_SCHEMA_IDS = {
    "curvyzero_lightzero_curvytron_visual_survival_compact_output/v0",
    "curvyzero_lightzero_curvytron_profile_spawn/v0",
    "curvyzero_lightzero_curvytron_visual_survival_background_launch/v0",
}
MATCHED_DENOMINATOR_ROW_PURPOSE = "matched_denominator_speed"
MATCHED_STOCK_SPEED_CURRENCY = "stock_train_muzero_profile_env_steps_per_sec"
MATCHED_STOCK_EXPECTED_ROW = {
    "batch_size": 64,
    "collect_search_backend": "stock",
    "collect_search_ctree_backend": "lightzero",
    "collector_env_num": 512,
    "compute": "gpu-h100-cpu40",
    "disable_death_for_profile": True,
    "env_manager_type": "subprocess",
    "exploration_bonus_mode": "none",
    "exploration_bonus_weight": 0.0,
    "num_simulations": 8,
    "source_max_steps": 512,
}


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
    selected: list[dict[str, Any]] = []
    matched: set[str] = set()
    for row in rows:
        aliases = _row_id_aliases(str(row.get("row_id")))
        hits = row_ids.intersection(aliases)
        if hits:
            selected.append(row)
            matched.update(hits)
    missing = sorted(row_ids.difference(matched))
    if missing:
        raise SystemExit(f"unknown row id(s): {', '.join(missing)}")
    return selected


def _row_id_aliases(row_id: str) -> set[str]:
    aliases = {row_id}
    if row_id.isdigit():
        number = str(int(row_id))
        aliases.update({number, number.zfill(2), number.zfill(3)})
    return aliases


def _parse_rows(raw: list[str]) -> set[str] | None:
    if not raw:
        return None
    rows: set[str] = set()
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if part:
                rows.add(str(int(part)) if part.isdigit() else part)
    return rows


def _validate_manifest(manifest: dict[str, Any], *, action: str) -> None:
    rows = list(manifest.get("rows") or [])
    if not rows:
        raise SystemExit("manifest has no rows")
    problems: list[str] = []
    for row in rows:
        row_id = row.get("row_id")
        command = list(row.get("command") or [])
        if not command:
            problems.append(f"row {row_id}: missing command")
            continue
        joined = " ".join(str(part) for part in command)
        if CURRENT_MODAL_MODULE not in command:
            problems.append(
                f"row {row_id}: command must use current Modal entrypoint "
                f"{CURRENT_MODAL_MODULE}"
            )
        if "--mode" not in command or "profile" not in command:
            problems.append(f"row {row_id}: command must run --mode profile")
        if "--output-detail" not in command or "compact" not in command:
            problems.append(f"row {row_id}: command must request --output-detail compact")
        if "--skip-lightzero-eval-in-profile" not in command:
            problems.append(f"row {row_id}: missing --skip-lightzero-eval-in-profile")
        if "--no-background-eval-enabled" not in command:
            problems.append(f"row {row_id}: missing --no-background-eval-enabled")
        if "--no-background-gif-enabled" not in command:
            problems.append(f"row {row_id}: missing --no-background-gif-enabled")
        is_spawn = "--profile-spawn" in command
        is_detached = "--detach" in command
        if is_spawn != is_detached:
            problems.append(
                f"row {row_id}: --profile-spawn and --detach must be used together"
            )
        if action in {"collect", "launch-and-collect"} and "modal run" not in joined:
            problems.append(f"row {row_id}: command does not look like a Modal run")
        problems.extend(_matched_stock_preflight_problems(row, command))
    if problems:
        raise SystemExit("manifest preflight failed:\n- " + "\n- ".join(problems))


def _command_flag_value(command: list[Any], flag: str) -> str | None:
    try:
        index = [str(part) for part in command].index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return str(command[index + 1])


def _matched_stock_preflight_problems(
    row: dict[str, Any],
    command: list[Any],
) -> list[str]:
    if not str(row.get("matched_denominator_id") or "").strip():
        return []
    row_id = row.get("row_id")
    problems: list[str] = []
    if row.get("matched_pair_role") != "stock_reference":
        problems.append(f"row {row_id}: matched denominator role must be stock_reference")
    if row.get("speed_currency") != MATCHED_STOCK_SPEED_CURRENCY:
        problems.append(
            f"row {row_id}: matched denominator speed_currency must be "
            f"{MATCHED_STOCK_SPEED_CURRENCY}"
        )
    if row.get("row_purpose") != MATCHED_DENOMINATOR_ROW_PURPOSE:
        problems.append(
            f"row {row_id}: matched denominator row_purpose must be "
            f"{MATCHED_DENOMINATOR_ROW_PURPOSE}"
        )
    if row.get("promotion_claim") is not False:
        problems.append(f"row {row_id}: matched denominator promotion_claim must be false")
    if not str(row.get("counterpart_manifest_ref") or "").strip():
        problems.append(f"row {row_id}: matched denominator counterpart manifest missing")
    if not str(row.get("counterpart_row_id") or "").strip():
        problems.append(f"row {row_id}: matched denominator counterpart row missing")
    for key, expected in MATCHED_STOCK_EXPECTED_ROW.items():
        actual = row.get(key)
        if actual != expected:
            problems.append(
                f"row {row_id}: matched denominator {key} "
                f"{actual!r} != expected {expected!r}"
            )
    fixed = row.get("fixed_denominator")
    if not isinstance(fixed, dict):
        problems.append(f"row {row_id}: matched denominator fixed_denominator missing")
    elif fixed.get("speed_currency") != row.get("speed_currency"):
        problems.append(f"row {row_id}: fixed_denominator.speed_currency must match row")
    expected_flag_values = {
        "--batch-size": 64,
        "--collect-search-backend": "stock",
        "--collect-search-ctree-backend": "lightzero",
        "--collector-env-num": 512,
        "--compute": "gpu-h100-cpu40",
        "--exploration-bonus-mode": "none",
        "--num-simulations": 8,
        "--source-max-steps": 512,
        "--stop-after-learner-train-calls": 12,
    }
    for flag, expected in expected_flag_values.items():
        actual = _command_flag_value(command, flag)
        if actual != str(expected):
            problems.append(
                f"row {row_id}: matched denominator command {flag} "
                f"{actual!r} != expected {str(expected)!r}"
            )
    required_flags = (
        "--detach",
        "--profile-spawn",
        "--skip-lightzero-eval-in-profile",
        "--no-background-eval-enabled",
        "--no-background-gif-enabled",
        "--disable-death-for-profile",
    )
    for flag in required_flags:
        if flag not in command:
            problems.append(f"row {row_id}: matched denominator missing {flag}")
    if "--require-rnd-metrics" in command:
        problems.append(f"row {row_id}: matched denominator stock row is no-RND")
    return problems


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
        if value.get("schema_id") in TOP_LEVEL_JSON_SCHEMA_IDS:
            return value
    for _index, _end, value in reversed(candidates):
        if value.get("function_call_id"):
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


def _profile_result_payload(
    *,
    row_id: str,
    record: dict[str, Any],
    compact: dict[str, Any] | None,
    result: Any,
    status: str,
    problem: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_id": "curvyzero_optimizer_profile_collected_result/v0",
        "collected_at": _utc_timestamp(),
        "row_id": row_id,
        "family": record.get("family"),
        "run_id": record.get("run_id"),
        "attempt_id": record.get("attempt_id"),
        "function_call_id": record.get("function_call_id"),
        "matched_denominator_id": record.get("matched_denominator_id"),
        "status": status,
        "problem": problem,
        "compact": compact,
        "result": result,
    }


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _matched_stock_result_problems(
    row: dict[str, Any] | None,
    compact: dict[str, Any] | None,
) -> list[str]:
    if row is None or not str(row.get("matched_denominator_id") or "").strip():
        return []
    if not isinstance(compact, dict):
        return ["matched denominator compact payload missing"]
    problems: list[str] = []
    counts = compact.get("counts") if isinstance(compact.get("counts"), dict) else {}
    derived = compact.get("derived") if isinstance(compact.get("derived"), dict) else {}
    if compact.get("ok") is not True:
        problems.append("compact.ok=true")
    if compact.get("called_train_muzero") is not True:
        problems.append("compact.called_train_muzero=true")
    if compact.get("mode") != "profile":
        problems.append("compact.mode=profile")
    if compact.get("trainer_entrypoint") != "lzero.entry.train_muzero":
        problems.append("trainer_entrypoint=lzero.entry.train_muzero")
    if counts.get("env_steps_collected_source") != "collector_envstep_delta":
        problems.append("counts.env_steps_collected_source=collector_envstep_delta")
    if counts.get("env_steps_collected_uses_fallback") is not False:
        problems.append("counts.env_steps_collected_uses_fallback=false")
    if derived.get("steps_per_sec_currency") != MATCHED_STOCK_SPEED_CURRENCY:
        problems.append(f"derived.steps_per_sec_currency={MATCHED_STOCK_SPEED_CURRENCY}")
    if derived.get("steps_per_sec_uses_fallback_denominator") is not False:
        problems.append("derived.steps_per_sec_uses_fallback_denominator=false")
    for key in ("env_steps_collected", "env_steps_collected_raw", "learner_train_calls", "replay_sample_calls"):
        value = _as_float(counts.get(key))
        if value is None or value <= 0.0:
            problems.append(f"counts.{key}>0")
    if counts.get("evaluator_eval_calls") != 0:
        problems.append("counts.evaluator_eval_calls=0")
    return problems


def _profile_status_from_compact(
    compact: dict[str, Any] | None,
    *,
    row: dict[str, Any] | None = None,
) -> tuple[str, str | None]:
    if isinstance(compact, dict) and compact.get("ok") is False:
        problem = compact.get("error") or compact.get("problem")
        return "profile_failed", str(problem or "compact output returned ok=false")
    matched_problems = _matched_stock_result_problems(row, compact)
    if matched_problems:
        return "matched_denominator_invariant_failed", "; ".join(matched_problems)
    return "complete", None


def _print_result_line(payload: dict[str, Any]) -> None:
    compact = payload.get("compact") if isinstance(payload.get("compact"), dict) else {}
    counts = compact.get("counts") if isinstance(compact.get("counts"), dict) else {}
    timers = compact.get("timers_sec") if isinstance(compact.get("timers_sec"), dict) else {}
    derived = compact.get("derived") if isinstance(compact.get("derived"), dict) else {}
    print(
        json.dumps(
            {
                "row_id": payload.get("row_id"),
                "status": payload.get("status"),
                "steps": counts.get("env_steps_collected"),
                "steps_raw": counts.get("env_steps_collected_raw"),
                "steps_source": counts.get("env_steps_collected_source"),
                "wall": timers.get("train_muzero_wall"),
                "steps_per_sec": derived.get("steps_per_sec"),
                "steps_per_sec_currency": derived.get("steps_per_sec_currency"),
                "steps_per_sec_uses_fallback_denominator": derived.get(
                    "steps_per_sec_uses_fallback_denominator"
                ),
                "problem": payload.get("problem"),
            },
            sort_keys=True,
        )
    )


def _write_result(output_dir: Path, payload: dict[str, Any]) -> None:
    _write_json(output_dir / f"row_{payload['row_id']}_result.json", payload)


def _launch_rows(
    *,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    launches_path: Path,
    output_dir: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        command = list(row.get("command") or [])
        command_text = str(row.get("command_text") or " ".join(command))
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
            "readback": row.get("readback"),
            "matched_denominator_id": row.get("matched_denominator_id"),
            "manifest_row": row if row.get("matched_denominator_id") else None,
        }
        if dry_run:
            print(command_text)
            records.append(record)
            continue
        is_spawn = "--profile-spawn" in command
        is_detached = "--detach" in command
        if is_spawn and not is_detached:
            raise SystemExit(
                f"row {row.get('row_id')} uses --profile-spawn without --detach. "
                "That can print a call id and then kill the child when the "
                "ephemeral parent exits. Rebuild the manifest without "
                "--no-detach, or run direct blocking rows without --profile-spawn."
            )
        if is_detached and not is_spawn:
            raise SystemExit(
                f"row {row.get('row_id')} uses --detach without --profile-spawn; "
                "there is no structured function-call result to collect."
            )
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
        if is_spawn:
            record["status"] = payload.get("status")
            record["function_call_id"] = payload.get("function_call_id")
            record["launch_payload"] = payload
        else:
            record["status"] = "blocking_complete"
            record["blocking_result_path"] = str(
                output_dir / f"row_{row.get('row_id')}_result.json"
            )
            compact = (
                payload
                if isinstance(payload, dict)
                and payload.get("schema_id")
                == "curvyzero_lightzero_curvytron_visual_survival_compact_output/v0"
                else _compact_result(payload)
                if isinstance(payload, dict)
                else {"result": payload}
            )
            status, problem = _profile_status_from_compact(compact, row=row)
            result_payload = _profile_result_payload(
                row_id=str(row.get("row_id")),
                record=record,
                compact=compact,
                result=payload,
                status=status,
                problem=problem,
            )
            _write_result(output_dir, result_payload)
            _print_result_line(result_payload)
        _append_jsonl(launches_path, record)
        records.append(record)
        if is_spawn:
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
        if record.get("status") == "blocking_complete" and record.get("blocking_result_path"):
            payload = _load_json(Path(str(record["blocking_result_path"])))
            results.append(payload)
            continue
        call_id = record.get("function_call_id")
        if not call_id:
            row_id = str(record.get("row_id"))
            payload = _profile_result_payload(
                row_id=row_id,
                record=record,
                compact=None,
                result=None,
                status="missing_function_call_id",
                problem="launch record has no function_call_id",
            )
            _write_result(output_dir, payload)
            _print_result_line(payload)
            results.append(payload)
            continue
        call = modal.FunctionCall.from_id(str(call_id))
        try:
            result = call.get(timeout=timeout_sec)
            compact = _compact_result(result if isinstance(result, dict) else {"result": result})
            status, problem = _profile_status_from_compact(
                compact,
                row=record.get("manifest_row") if isinstance(record.get("manifest_row"), dict) else None,
            )
        except Exception as exc:  # pragma: no cover - remote failures are runtime data.
            result = None
            compact = None
            status = "collect_failed"
            problem = f"{type(exc).__name__}: {exc}"
        row_id = str(record.get("row_id"))
        payload = _profile_result_payload(
            row_id=row_id,
            record=record,
            compact=compact,
            result=result,
            status=status,
            problem=problem,
        )
        _write_result(output_dir, payload)
        _print_result_line(payload)
        results.append(payload)
    _write_json(output_dir / "collected_results.json", results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
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
    _validate_manifest(manifest, action=args.action)
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
            output_dir=output_dir,
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
