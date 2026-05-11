#!/usr/bin/env python3
"""Print a compact LightZero Pong replication status table."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VOLUME_NAME = "curvyzero-runs"
LOCAL_ARTIFACT_ROOT = Path("artifacts/local")
DEFAULT_DOC_STATUS = Path("docs/working/lightzero_pong_replication_monitor_2026-05-11.md")
ITERATION_RE = re.compile(r"\biteration_(\d+)\.pth\.tar\b")
CHECKPOINT_RE = re.compile(r"`(?:iteration_)?(\d+)`")
NUMBER_RE = re.compile(r"`?(-?\d+(?:\.\d+)?)`?")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
WARN_MISSING = False


@dataclass(frozen=True)
class RunRow:
    lane: str
    run: str
    run_id: str
    attempt_id: str
    task_id: str
    exp_dir: str

    @property
    def train_root(self) -> str:
        return f"training/{self.task_id}/{self.run_id}/attempts/{self.attempt_id}/train"

    @property
    def checkpoint_ref(self) -> str:
        return f"{self.train_root}/{self.exp_dir}/ckpt"

    @property
    def progress_ref(self) -> str:
        return f"{self.train_root}/progress/latest.json"

    @property
    def eval_ref(self) -> str:
        return f"training/{self.task_id}/{self.run_id}/attempts/{self.attempt_id}/eval"


RUNS = [
    RunRow(
        lane="stock64",
        run="s113",
        run_id="lz-visual-pong-exact-repro-20260511-s113-detached",
        attempt_id="train-stock-exact-200k-l4cpu40-detached-wait",
        task_id="lightzero-official-visual-pong",
        exp_dir="lightzero_exp",
    ),
    RunRow(
        lane="stock64",
        run="s114",
        run_id="lz-visual-pong-faithful-short-20260511-s114-detached",
        attempt_id="train-stock-surface-50k-ckpt1000-l4cpu40-detached-wait",
        task_id="lightzero-official-visual-pong",
        exp_dir="lightzero_exp",
    ),
    RunRow(
        lane="stock64",
        run="s120",
        run_id="lz-visual-pong-replication-matrix-20260511-s120",
        attempt_id="train-stock-surface-50k-ckpt1000-l4cpu40-detached-wait",
        task_id="lightzero-official-visual-pong",
        exp_dir="lightzero_exp",
    ),
    RunRow(
        lane="stock64",
        run="s121",
        run_id="lz-visual-pong-replication-matrix-20260511-s121",
        attempt_id="train-stock-surface-65k-ckpt1000-l4cpu40-detached-wait",
        task_id="lightzero-official-visual-pong",
        exp_dir="lightzero_exp",
    ),
    RunRow(
        lane="stock64",
        run="s122",
        run_id="lz-visual-pong-replication-matrix-20260511-s122-h100",
        attempt_id="train-stock-surface-100k-ckpt1000-h100cpu40-detached-wait",
        task_id="lightzero-official-visual-pong",
        exp_dir="lightzero_exp",
    ),
    RunRow(
        lane="stock64",
        run="s123",
        run_id="lz-visual-pong-replication-matrix-20260511-s123-h100-exact",
        attempt_id="train-stock-exact-200k-h100cpu40-detached-wait",
        task_id="lightzero-official-visual-pong",
        exp_dir="lightzero_exp",
    ),
    RunRow(
        lane="stock64",
        run="s130",
        run_id="lz-visual-pong-replication-matrix-20260511-s130-l4-exact",
        attempt_id="train-stock-exact-200k-l4cpu40-detached",
        task_id="lightzero-official-visual-pong",
        exp_dir="lightzero_exp",
    ),
    RunRow(
        lane="stock64",
        run="s131",
        run_id="lz-visual-pong-replication-matrix-20260511-s131-l4-50k",
        attempt_id="train-stock-surface-50k-ckpt1000-l4cpu40-detached",
        task_id="lightzero-official-visual-pong",
        exp_dir="lightzero_exp",
    ),
    RunRow(
        lane="stock64",
        run="s132",
        run_id="lz-visual-pong-replication-matrix-20260511-s132-l4-65k",
        attempt_id="train-stock-surface-65k-ckpt1000-l4cpu40-detached",
        task_id="lightzero-official-visual-pong",
        exp_dir="lightzero_exp",
    ),
    RunRow(
        lane="stock64",
        run="s133",
        run_id="lz-visual-pong-replication-matrix-20260511-s133-l4-100k",
        attempt_id="train-stock-surface-100k-ckpt1000-l4cpu40-detached",
        task_id="lightzero-official-visual-pong",
        exp_dir="lightzero_exp",
    ),
    RunRow(
        lane="agent96",
        run="s125",
        run_id="lz-visual-pong-muzero-agent96-20260511-s125-modelcard-detached",
        attempt_id="train-agent96-modelcard-500k-h100cpu40-detached",
        task_id="lightzero-official-visual-pong-muzero-agent96",
        exp_dir="agent_exp",
    ),
    RunRow(
        lane="agent96",
        run="s127",
        run_id="lz-visual-pong-muzero-agent96-20260511-s127-short-h100-detached",
        attempt_id="train-agent96-50k-h100cpu40-ckpt1000-detached",
        task_id="lightzero-official-visual-pong-muzero-agent96",
        exp_dir="agent_exp",
    ),
]

KNOWN_EVAL_SURVIVAL = {
    "lz-visual-pong-faithful-short-20260511-s114-detached": "0=761.25,1000=761.25,2000=761.25",
    "lz-visual-pong-replication-matrix-20260511-s120": "0=761.25,1000=768.25",
    "lz-visual-pong-replication-matrix-20260511-s121": "0=835.875,1000=761.25",
    "lz-visual-pong-replication-matrix-20260511-s122-h100": "0=768.75,1000=768.375",
}

RUN_ID_BY_SHORT = {row.run: row.run_id for row in RUNS}


def _run_command(args: list[str], *, timeout_sec: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args, text=True, capture_output=True, check=False, timeout=timeout_sec
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args,
            124,
            stdout=exc.stdout or "",
            stderr=f"timed out after {timeout_sec:.0f}s",
        )


def _modal_ls(
    volume: str, ref: str, *, offline: bool, modal_bin: str, timeout_sec: float
) -> str:
    if offline:
        return ""
    result = _run_command(
        [
            modal_bin,
            "volume",
            "ls",
            "--json",
            volume,
            ref,
        ],
        timeout_sec=timeout_sec,
    )
    if result.returncode == 0:
        return result.stdout
    _note_missing(ref, result.stderr)
    return ""


def _modal_get_json(
    volume: str, ref: str, *, offline: bool, modal_bin: str, timeout_sec: float
) -> dict[str, Any] | None:
    if offline:
        return None
    result = _run_command(
        [modal_bin, "volume", "get", volume, ref, "-"], timeout_sec=timeout_sec
    )
    if result.returncode != 0:
        _note_missing(ref, result.stderr)
        return None
    value = _parse_first_json(result.stdout)
    if value is None:
        exc = "no JSON object found"
        print(f"# could not parse JSON at {ref}: {exc}", file=sys.stderr)
        return None
    return value if isinstance(value, dict) else None


def _parse_first_json(text: str) -> Any:
    stripped = text.lstrip()
    if not stripped:
        return None
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    return value


def _note_missing(ref: str, stderr: str) -> None:
    if not WARN_MISSING:
        return
    message = " ".join(stderr.strip().split())
    if message:
        print(f"# missing/unreadable: {ref} ({message})", file=sys.stderr)


def _checkpoint_iterations(listing: str) -> str:
    iterations = sorted(
        {int(match.group(1)) for match in ITERATION_RE.finditer(_clean(listing))}
    )
    return ",".join(str(iteration) for iteration in iterations) if iterations else "-"


def _checkpoint_status(listing: str, progress: dict[str, Any] | None) -> str:
    scan = progress.get("scan") if isinstance(progress, dict) else None
    if isinstance(scan, dict):
        count = scan.get("checkpoint_count")
        latest = _latest_checkpoint_from_scan(scan)
        if isinstance(count, int) and latest:
            return f"{count}/{latest}"
    iterations = _checkpoint_iterations_list(listing)
    if not iterations:
        return "-"
    return f"{len(iterations)}/iteration_{iterations[-1]}"


def _checkpoint_iterations_list(listing: str) -> list[int]:
    return sorted(
        {int(match.group(1)) for match in ITERATION_RE.finditer(_clean(listing))}
    )


def _latest_checkpoint_from_scan(scan: dict[str, Any]) -> str | None:
    newest = scan.get("newest_checkpoints")
    if not isinstance(newest, list):
        return None
    paths = []
    for item in newest:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if isinstance(path, str):
            paths.append(path)
    iterations = _checkpoint_iterations_list("\n".join(paths))
    if iterations:
        return f"iteration_{iterations[-1]}"
    if paths:
        return paths[0].rsplit("/", 1)[-1].removesuffix(".pth.tar")
    return None


def _entry_names(listing: str) -> list[str]:
    json_names = _json_listing_names(listing)
    if json_names:
        return json_names
    names: list[str] = []
    seen: set[str] = set()
    for raw_line in _clean(listing).splitlines():
        line = raw_line.strip().rstrip("/")
        if not line:
            continue
        name = line.split()[-1].rstrip("/").rsplit("/", 1)[-1]
        if name and name not in {".", ".."} and name not in seen:
            names.append(name)
            seen.add(name)
    return sorted(names)


def _json_listing_names(listing: str) -> list[str]:
    try:
        value = json.loads(listing)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for item in value:
        name: str | None = None
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            for key in ("name", "path", "filename", "Filename"):
                raw = item.get(key)
                if isinstance(raw, str):
                    name = raw
                    break
        if name:
            names.append(name.rstrip("/").rsplit("/", 1)[-1])
    return sorted({name for name in names if name and name not in {".", ".."}})


def _clean(text: str) -> str:
    return ANSI_RE.sub("", text)


def _progress_summary(progress: dict[str, Any] | None) -> str:
    if progress is None:
        return "-"
    phase = _first_text(progress, "phase", "status", "state") or "?"
    elapsed = _first_number(
        progress,
        "train_elapsed_sec",
        "elapsed_sec",
        "elapsed_seconds",
        "wall_time_sec",
    )
    timestamp = _first_text(progress, "timestamp", "updated_at", "checked_at")
    parts = [phase]
    if elapsed is not None:
        parts.append(_format_seconds(elapsed))
    if timestamp:
        parts.append(timestamp.replace("T", " ").removesuffix("Z"))
    return " ".join(parts)


def _train_status(progress: dict[str, Any] | None) -> str:
    if progress is None:
        return "-"
    phase = _first_text(progress, "phase", "status", "state") or "?"
    elapsed = _first_number(
        progress,
        "train_elapsed_sec",
        "elapsed_sec",
        "elapsed_seconds",
        "wall_time_sec",
    )
    if elapsed is None:
        return phase
    return f"{phase} {_format_seconds(elapsed)}"


def _notes(eval_roots: list[str], local_summaries: str, *, local_only: bool) -> str:
    notes: list[str] = []
    if eval_roots:
        notes.append("eval:" + ",".join(eval_roots[:2]))
        if len(eval_roots) > 2:
            notes.append(f"+{len(eval_roots) - 2}")
    if local_summaries != "-":
        notes.append("local:" + local_summaries)
    if local_only:
        notes.append("local-only")
    return "; ".join(notes) if notes else "-"


def _first_text(data: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _first_number(data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _format_seconds(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, sec = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def _local_progress_by_run(root: Path) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return found
    for path in root.rglob("*.json"):
        if (
            "latest" not in path.name
            and "progress" not in path.name
            and "summary" not in path.name
        ):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        run_id = data.get("run_id")
        if isinstance(run_id, str) and run_id not in found:
            found[run_id] = data
    return found


def _local_summary_roots(root: Path, row: RunRow, *, limit: int) -> str:
    if not root.exists():
        return "-"
    matches: list[str] = []
    needles = {row.run, row.run_id}
    for path in root.rglob("summary.json"):
        text = path.as_posix()
        if any(needle in text for needle in needles):
            matches.append(path.parent.as_posix())
    matches = sorted(set(matches))
    if not matches:
        return "-"
    names = [Path(match).name for match in matches[:limit]]
    if len(matches) > limit:
        names.append(f"+{len(matches) - limit}")
    return ",".join(names)


def _live_row(
    row: RunRow,
    *,
    volume: str,
    local_only: bool,
    modal_bin: str,
    timeout_sec: float,
) -> tuple[RunRow, str, dict[str, Any] | None, list[str]]:
    checkpoint_listing = _modal_ls(
        volume,
        row.checkpoint_ref,
        offline=local_only,
        modal_bin=modal_bin,
        timeout_sec=timeout_sec,
    )
    progress = _modal_get_json(
        volume,
        row.progress_ref,
        offline=local_only,
        modal_bin=modal_bin,
        timeout_sec=timeout_sec,
    )
    eval_listing = _modal_ls(
        volume,
        row.eval_ref,
        offline=local_only,
        modal_bin=modal_bin,
        timeout_sec=timeout_sec,
    )
    return row, checkpoint_listing, progress, _entry_names(eval_listing)


def _live_rows(
    rows: list[RunRow],
    *,
    volume: str,
    local_only: bool,
    modal_bin: str,
    timeout_sec: float,
    workers: int,
) -> dict[str, tuple[str, dict[str, Any] | None, list[str]]]:
    if local_only:
        return {row.run_id: ("", None, []) for row in rows}
    max_workers = max(1, workers)
    results: dict[str, tuple[str, dict[str, Any] | None, list[str]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _live_row,
                row,
                volume=volume,
                local_only=local_only,
                modal_bin=modal_bin,
                timeout_sec=timeout_sec,
            ): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                _, checkpoint_listing, progress, eval_roots = future.result()
            except Exception as exc:  # pragma: no cover - defensive CLI boundary.
                if WARN_MISSING:
                    print(f"# live read failed for {row.run_id}: {exc}", file=sys.stderr)
                checkpoint_listing, progress, eval_roots = "", None, []
            results[row.run_id] = (checkpoint_listing, progress, eval_roots)
    return results


def _local_eval_survival_by_run(root: Path) -> dict[str, str]:
    by_run: dict[str, dict[int, list[float]]] = {}
    for base in (root / "lightzero-eval-manifests", root / "eval-manifest-files"):
        if not base.exists():
            continue
        for path in base.rglob("*.json"):
            if path.name.startswith("lightzero_visual_pong_eval_"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            _add_manifest_survival(data, by_run)
    return {
        run_id: _format_survival_curve(points)
        for run_id, points in by_run.items()
        if points
    }


def _add_manifest_survival(
    data: Any, by_run: dict[str, dict[int, list[float]]]
) -> None:
    if not isinstance(data, dict):
        return
    default_run_id = _nested(data, "config", "run_id")
    results = data.get("results")
    if not isinstance(results, list):
        return
    for result in results:
        if not isinstance(result, dict):
            continue
        run_id = _first_string(
            _nested(result, "config", "run_id"),
            default_run_id,
        )
        iteration = _iteration_from_value(
            _first_string(
                result.get("checkpoint"),
                _nested(result, "checkpoint", "path"),
                _nested(result, "config", "checkpoint_ref"),
            )
        )
        steps = _first_float(
            result.get("stock_steps_survived"),
            _nested(result, "stock_rollout", "steps_run"),
            _nested(result, "stock_rollout", "episode_length"),
        )
        if run_id and iteration is not None and steps is not None:
            by_run.setdefault(run_id, {}).setdefault(iteration, []).append(steps)


def _doc_eval_survival_by_run(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    found: dict[str, str] = {}
    for line in lines:
        if not line.startswith("| `s"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        run_id = RUN_ID_BY_SHORT.get(cells[0].strip("`"))
        if not run_id:
            continue
        checkpoints = [int(value) for value in CHECKPOINT_RE.findall(cells[2])]
        values = [float(value) for value in NUMBER_RE.findall(cells[3])]
        if checkpoints and len(checkpoints) == len(values):
            found[run_id] = ",".join(
                f"{checkpoint}={_format_number(value)}"
                for checkpoint, value in zip(checkpoints, values, strict=True)
            )
    return found


def _eval_survival_by_run(local_root: Path, doc_status: Path) -> dict[str, str]:
    survival = dict(KNOWN_EVAL_SURVIVAL)
    survival.update(_doc_eval_survival_by_run(doc_status))
    survival.update(_local_eval_survival_by_run(local_root))
    return survival


def _format_survival_curve(points: dict[int, list[float]]) -> str:
    parts = []
    for iteration in sorted(points):
        values = points[iteration]
        parts.append(f"{iteration}={_format_number(sum(values) / len(values))}")
    return ",".join(parts) if parts else "-"


def _iteration_from_value(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\biteration_(\d+)\b", value)
    return int(match.group(1)) if match else None


def _first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _nested(data: Any, *keys: str) -> Any:
    value = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.6g}"


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    lines = []
    for row_index, row in enumerate(rows):
        pieces = [cell.ljust(widths[index]) for index, cell in enumerate(row)]
        lines.append("  ".join(pieces).rstrip())
        if row_index == 0:
            lines.append("  ".join("-" * width for width in widths).rstrip())
    return "\n".join(lines)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Poll known 2026-05-11 LightZero Pong replication paths and print "
            "a compact checkpoint/progress/eval status table."
        )
    )
    parser.add_argument("--volume", default=VOLUME_NAME)
    parser.add_argument("--local-root", type=Path, default=LOCAL_ARTIFACT_ROOT)
    parser.add_argument(
        "--live-modal",
        action="store_true",
        help="Poll Modal Volume paths. Default is local fetched JSON only.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Alias for the default local-only mode.",
    )
    parser.add_argument(
        "--verbose-missing",
        action="store_true",
        help="Print Modal missing-path errors to stderr.",
    )
    parser.add_argument(
        "--modal-bin",
        default="modal",
        help=(
            "Modal executable. Default uses the local modal CLI directly; set to "
            "a wrapper only if needed."
        ),
    )
    parser.add_argument("--modal-timeout-sec", type=float, default=8.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel live Modal row reads when --live-modal is set.",
    )
    parser.add_argument("--max-local-roots", type=int, default=2)
    parser.add_argument(
        "--doc-status",
        type=Path,
        default=DEFAULT_DOC_STATUS,
        help="Local monitor note to mine for known eval survival curves.",
    )
    return parser.parse_args()


def main() -> int:
    global WARN_MISSING
    args = parse_args()
    WARN_MISSING = args.verbose_missing
    local_only = args.offline or not args.live_modal
    local_progress = _local_progress_by_run(args.local_root)
    eval_survival = _eval_survival_by_run(args.local_root, args.doc_status)
    live_rows = _live_rows(
        RUNS,
        volume=args.volume,
        local_only=local_only,
        modal_bin=args.modal_bin,
        timeout_sec=args.modal_timeout_sec,
        workers=args.workers,
    )
    output_rows = [[
        "run id",
        "attempt",
        "family",
        "train status",
        "ckpt count/latest",
        "eval survival mean",
        "notes",
    ]]

    for row in RUNS:
        checkpoint_listing, progress, eval_roots = live_rows.get(
            row.run_id, ("", None, [])
        )
        if progress is None:
            progress = local_progress.get(row.run_id)
        local_summaries = _local_summary_roots(
            args.local_root,
            row,
            limit=args.max_local_roots,
        )
        output_rows.append(
            [
                row.run_id,
                row.attempt_id,
                row.lane,
                _train_status(progress),
                _checkpoint_status(checkpoint_listing, progress),
                eval_survival.get(row.run_id, "-"),
                _notes(eval_roots, local_summaries, local_only=local_only),
            ]
        )

    print(_table(output_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
