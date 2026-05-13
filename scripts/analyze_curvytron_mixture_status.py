#!/usr/bin/env python3
"""Summarize CurvyTron opponent-mixture status snapshots.

Input is a local opponent-mixture manifest plus JSON output from
``lightzero_curvytron_run_status.py --output json``. The main readout is the
matched fast/browser checkpoint comparison, so launch order does not masquerade
as render speed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


CHECKPOINT_RE = re.compile(r"iteration_(\d+)$")


def _load_json_with_modal_noise(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = None
    for match in re.finditer(r"(?m)^[ \t]*(?:\[(?=[\s{\]])|\{(?=\s*\"))", text):
        start = match.start()
        break
    if start is None:
        raise ValueError(f"{path} does not contain JSON")
    decoder = json.JSONDecoder()
    value, _end = decoder.raw_decode(text[start:])
    return value


def _checkpoint_iter(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    match = CHECKPOINT_RE.fullmatch(value)
    return int(match.group(1)) if match else None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _repeat_token(row: dict[str, Any]) -> str:
    base_settings = row.get("base_settings")
    if isinstance(base_settings, dict):
        token = base_settings.get("token")
        if isinstance(token, str):
            parts = token.split("-")
            for part in parts:
                if part.startswith("rep"):
                    return part
    base_token = row.get("base_token")
    if isinstance(base_token, str):
        for part in base_token.split("-"):
            if part.startswith("rep"):
                return part
    return "unknown"


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    settings = row.get("base_settings")
    if not isinstance(settings, dict):
        settings = {}
    return (
        row.get("recipe_id"),
        row.get("copy_index"),
        row.get("training_seed"),
        settings.get("num_simulations"),
        settings.get("collector_env_num"),
        settings.get("n_episode"),
        settings.get("batch_size"),
        _repeat_token(row),
        settings.get("save_ckpt_after_iter"),
    )


def summarize(
    manifest: dict[str, Any],
    status_rows: list[dict[str, Any]],
    *,
    target_iteration: int,
) -> dict[str, Any]:
    manifest_rows = manifest.get("rows", [])
    if not isinstance(manifest_rows, list):
        raise ValueError("manifest rows must be a list")
    by_run_id = {
        str(row.get("run_id")): row
        for row in manifest_rows
        if isinstance(row, dict) and row.get("run_id")
    }
    status_by_run_id = {
        str(row.get("short_name")): row
        for row in status_rows
        if isinstance(row, dict) and row.get("short_name")
    }

    enriched: list[dict[str, Any]] = []
    for run_id, manifest_row in by_run_id.items():
        status = status_by_run_id.get(run_id, {})
        latest_iter = _checkpoint_iter(status.get("latest_checkpoint"))
        item = {
            "run_id": run_id,
            "recipe_id": manifest_row.get("recipe_id"),
            "render": manifest_row.get("source_state_trail_render_mode"),
            "render_role": manifest_row.get("render_role"),
            "repeat": _repeat_token(manifest_row),
            "copy_index": manifest_row.get("copy_index"),
            "training_seed": manifest_row.get("training_seed"),
            "latest_iter": latest_iter,
            "latest_checkpoint": status.get("latest_checkpoint"),
            "checkpoint_count": status.get("checkpoint_count"),
            "elapsed_sec": _safe_float(status.get("elapsed_sec")),
            "checkpoint_mtime": _checkpoint_mtime(status, target_iteration),
            "iteration0_mtime": _checkpoint_mtime(status, 0),
            "heartbeat": status.get("status_heartbeat_exists"),
            "progress_missing_reason": status.get("progress_missing_reason"),
            "eval_checkpoint": status.get("latest_eval_checkpoint"),
            "gif_checkpoint": status.get("latest_gif_checkpoint"),
            "pair_key": _pair_key(manifest_row),
        }
        enriched.append(item)

    by_render: dict[str, Counter[str]] = defaultdict(Counter)
    by_recipe: dict[str, Counter[str]] = defaultdict(Counter)
    for item in enriched:
        bucket = (
            "target"
            if item["latest_iter"] is not None and item["latest_iter"] >= target_iteration
            else "iteration0"
            if item["latest_iter"] == 0
            else "none"
        )
        render = str(item.get("render") or "unknown")
        recipe = str(item.get("recipe_id") or "unknown")
        by_render[render][bucket] += 1
        by_render[render]["rows"] += 1
        by_recipe[recipe][bucket] += 1
        by_recipe[recipe]["rows"] += 1

    pairs_by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for item in enriched:
        role = item.get("render_role")
        if role in {"fast", "browser"}:
            pairs_by_key[item["pair_key"]][str(role)] = item

    matched_pairs: list[dict[str, Any]] = []
    ready_pairs = 0
    for key, pair in pairs_by_key.items():
        fast = pair.get("fast")
        browser = pair.get("browser")
        if not fast or not browser:
            continue
        fast_ready = fast["latest_iter"] is not None and fast["latest_iter"] >= target_iteration
        browser_ready = (
            browser["latest_iter"] is not None and browser["latest_iter"] >= target_iteration
        )
        if fast_ready and browser_ready:
            ready_pairs += 1
            delta = None
            if fast["elapsed_sec"] is not None and browser["elapsed_sec"] is not None:
                delta = browser["elapsed_sec"] - fast["elapsed_sec"]
            checkpoint_delta = None
            if fast["checkpoint_mtime"] is not None and browser["checkpoint_mtime"] is not None:
                checkpoint_delta = browser["checkpoint_mtime"] - fast["checkpoint_mtime"]
            fast_checkpoint_gap = None
            if fast["checkpoint_mtime"] is not None and fast["iteration0_mtime"] is not None:
                fast_checkpoint_gap = fast["checkpoint_mtime"] - fast["iteration0_mtime"]
            browser_checkpoint_gap = None
            if browser["checkpoint_mtime"] is not None and browser["iteration0_mtime"] is not None:
                browser_checkpoint_gap = browser["checkpoint_mtime"] - browser["iteration0_mtime"]
            gap_delta = None
            if fast_checkpoint_gap is not None and browser_checkpoint_gap is not None:
                gap_delta = browser_checkpoint_gap - fast_checkpoint_gap
            matched_pairs.append(
                {
                    "recipe_id": key[0],
                    "copy_index": key[1],
                    "training_seed": key[2],
                    "repeat": key[7],
                    "fast_elapsed_sec": fast["elapsed_sec"],
                    "browser_elapsed_sec": browser["elapsed_sec"],
                    "browser_minus_fast_sec": delta,
                    "fast_checkpoint_gap_sec": fast_checkpoint_gap,
                    "browser_checkpoint_gap_sec": browser_checkpoint_gap,
                    "browser_minus_fast_checkpoint_gap_sec": gap_delta,
                    "browser_minus_fast_checkpoint_mtime_sec": checkpoint_delta,
                    "fast_run_id": fast["run_id"],
                    "browser_run_id": browser["run_id"],
                }
            )

    deltas = [
        pair["browser_minus_fast_sec"]
        for pair in matched_pairs
        if pair["browser_minus_fast_sec"] is not None
    ]
    gap_deltas = [
        pair["browser_minus_fast_checkpoint_gap_sec"]
        for pair in matched_pairs
        if pair["browser_minus_fast_checkpoint_gap_sec"] is not None
    ]
    return {
        "matrix_name": manifest.get("matrix_name"),
        "row_count": len(enriched),
        "target_iteration": target_iteration,
        "rows_at_target": sum(
            item["latest_iter"] is not None and item["latest_iter"] >= target_iteration
            for item in enriched
        ),
        "rows_at_iteration0": sum(item["latest_iter"] == 0 for item in enriched),
        "rows_without_checkpoint": sum(item["latest_iter"] is None for item in enriched),
        "by_render": {key: dict(value) for key, value in sorted(by_render.items())},
        "by_recipe": {key: dict(value) for key, value in sorted(by_recipe.items())},
        "matched_pair_count": len(pairs_by_key),
        "matched_pairs_at_target": ready_pairs,
        "matched_pairs": matched_pairs,
        "delta_summary": {
            "count": len(deltas),
            "mean_browser_minus_fast_sec": mean(deltas) if deltas else None,
            "median_browser_minus_fast_sec": median(deltas) if deltas else None,
            "min_browser_minus_fast_sec": min(deltas) if deltas else None,
            "max_browser_minus_fast_sec": max(deltas) if deltas else None,
        },
        "checkpoint_gap_delta_summary": {
            "count": len(gap_deltas),
            "mean_browser_minus_fast_sec": mean(gap_deltas) if gap_deltas else None,
            "median_browser_minus_fast_sec": median(gap_deltas) if gap_deltas else None,
            "min_browser_minus_fast_sec": min(gap_deltas) if gap_deltas else None,
            "max_browser_minus_fast_sec": max(gap_deltas) if gap_deltas else None,
        },
    }


def _checkpoint_mtime(status: dict[str, Any], iteration: int) -> float | None:
    checkpoints = status.get("checkpoints")
    if not isinstance(checkpoints, list):
        return None
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            continue
        if checkpoint.get("iteration") != iteration:
            continue
        return _safe_float(checkpoint.get("mtime"))
    return None


def _print_markdown(summary: dict[str, Any]) -> None:
    print(f"# Mixture Status Summary: {summary['matrix_name']}")
    print()
    print(f"Target checkpoint: iteration_{summary['target_iteration']}")
    print()
    print("| Field | Value |")
    print("| --- | ---: |")
    for key in (
        "row_count",
        "rows_at_target",
        "rows_at_iteration0",
        "rows_without_checkpoint",
        "matched_pair_count",
        "matched_pairs_at_target",
    ):
        print(f"| {key} | {summary[key]} |")
    print()
    print("## By Render")
    print()
    print("| Render | Rows | Target | Iteration 0 | None |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for render, counts in summary["by_render"].items():
        print(
            f"| {render} | {counts.get('rows', 0)} | {counts.get('target', 0)} | "
            f"{counts.get('iteration0', 0)} | {counts.get('none', 0)} |"
        )
    print()
    print("## Matched Render Pairs At Target")
    print()
    if not summary["matched_pairs"]:
        print("No matched fast/browser pairs have both reached the target checkpoint.")
        return
    print(
        "| Recipe | Repeat | Copy | Fast k0->target sec | Browser k0->target sec | "
        "Browser - fast gap sec | Latest elapsed delta sec |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for pair in summary["matched_pairs"]:
        fast_gap = _format_optional_float(pair["fast_checkpoint_gap_sec"])
        browser_gap = _format_optional_float(pair["browser_checkpoint_gap_sec"])
        gap_delta = _format_optional_float(pair["browser_minus_fast_checkpoint_gap_sec"])
        elapsed_delta = _format_optional_float(pair["browser_minus_fast_sec"])
        print(
            f"| {pair['recipe_id']} | {pair['repeat']} | {pair['copy_index']} | "
            f"{fast_gap} | {browser_gap} | {gap_delta} | {elapsed_delta} |"
        )
    delta = summary["checkpoint_gap_delta_summary"]
    print()
    if delta["count"]:
        print(
            "Checkpoint gap delta summary: "
            f"count={delta['count']}, "
            f"median={delta['median_browser_minus_fast_sec']:.3f}s, "
            f"mean={delta['mean_browser_minus_fast_sec']:.3f}s."
        )
    else:
        print("Checkpoint gap delta summary: count=0; checkpoint mtimes are not available yet.")


def _format_optional_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("status_json", type=Path)
    parser.add_argument("--target-iteration", type=int, default=10_000)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()

    manifest = _load_json_with_modal_noise(args.manifest)
    status_rows = _load_json_with_modal_noise(args.status_json)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    if not isinstance(status_rows, list):
        raise ValueError("status_json must be a JSON list")
    summary = summarize(manifest, status_rows, target_iteration=args.target_iteration)
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_markdown(summary)


if __name__ == "__main__":
    main()
