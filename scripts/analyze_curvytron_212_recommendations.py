#!/usr/bin/env python3
"""Build a compact CurvyTron 212-run recommendation report.

This is intentionally local-only. It consumes already-exported status snapshots
and analysis CSVs so we can iterate on interpretation without touching Modal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median, quantiles
from typing import Any


DEFAULT_ANALYSIS_DIR = Path("artifacts/local/curvytron_status_snapshots/analysis_20260513e")
DEFAULT_STATUS_JSON = Path(
    "artifacts/local/curvytron_pruning/status_chunks_20260513e/combined_status.json"
)


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _int(value: Any) -> int | None:
    parsed = _float(value)
    return None if parsed is None else int(parsed)


def _bool(value: Any) -> bool:
    return str(value).lower() == "true"


def _quartiles(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 4:
        return None, None
    qs = quantiles(values, n=4)
    return qs[0], qs[2]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [row["delta"] for row in rows if row.get("delta") is not None]
    rels = [row["relative_delta_pct"] for row in rows if row.get("relative_delta_pct") is not None]
    q1, q3 = _quartiles(deltas)
    rq1, rq3 = _quartiles(rels)
    improved = sum(1 for row in rows if (row.get("delta") or 0.0) > 0)
    return {
        "n": len(rows),
        "improved_n": improved,
        "improved_pct": improved / len(rows) if rows else None,
        "median_delta_steps": median(deltas) if deltas else None,
        "p25_delta_steps": q1,
        "p75_delta_steps": q3,
        "median_relative_delta_pct": median(rels) if rels else None,
        "p25_relative_delta_pct": rq1,
        "p75_relative_delta_pct": rq3,
    }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            first = _float(row.get("eval_first_mean_steps"))
            latest = _float(row.get("eval_latest_mean_steps"))
            delta = _float(row.get("eval_delta_latest_minus_first"))
            if delta is None and first is not None and latest is not None:
                delta = latest - first
            relative_delta_pct = None
            if first is not None and first > 0 and delta is not None:
                relative_delta_pct = 100.0 * delta / first
            rows.append(
                {
                    **row,
                    "latest_iteration": _int(row.get("latest_iteration")),
                    "eval_points": _int(row.get("eval_points")) or 0,
                    "first": first,
                    "latest": latest,
                    "delta": delta,
                    "relative_delta_pct": relative_delta_pct,
                    "stale_gt_180m": _bool(row.get("stale_gt_180m")),
                    "stale_gt_360m": _bool(row.get("stale_gt_360m")),
                    "sim": _int(row.get("sim")),
                    "collector_env_num": _int(row.get("collector_env_num")),
                    "batch_size": _int(row.get("batch_size")),
                }
            )
    return rows


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = str(row.get(key) or "unknown")
        grouped[value].append(row)
    return {name: _summary(group) for name, group in sorted(grouped.items())}


def _checkpoint_map(eval_checkpoints: list[dict[str, Any]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for checkpoint in eval_checkpoints:
        label = checkpoint.get("checkpoint")
        if not isinstance(label, str) or not label.startswith("iteration_"):
            continue
        iteration = _int(label.removeprefix("iteration_"))
        mean_steps = _float(checkpoint.get("mean_steps"))
        if iteration is None or mean_steps is None:
            continue
        out[iteration] = mean_steps
    return out


def _checkpoint_matched_render(
    status_path: Path,
    fair_pairs_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status_by_run_id = {
        row["run_id"]: row
        for row in status.get("rows", [])
        if isinstance(row, dict) and row.get("run_id")
    }
    pair_rows: list[dict[str, Any]] = []
    with fair_pairs_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("pair_type") != "mix_render_pair":
                continue
            fast = status_by_run_id.get(str(row.get("fast_run_id")))
            browser = status_by_run_id.get(str(row.get("browser_run_id")))
            if not fast or not browser:
                continue
            fast_curve = _checkpoint_map(fast.get("eval_checkpoints") or [])
            browser_curve = _checkpoint_map(browser.get("eval_checkpoints") or [])
            common = sorted(set(fast_curve) & set(browser_curve))
            if not common:
                continue
            deltas = [browser_curve[iteration] - fast_curve[iteration] for iteration in common]
            pair_rows.append(
                {
                    "pair_key": row.get("pair_key"),
                    "common_checkpoint_count": len(common),
                    "latest_common_iteration": common[-1],
                    "median_common_delta_browser_minus_fast": median(deltas),
                    "latest_common_delta_browser_minus_fast": deltas[-1],
                }
            )
    medians = [row["median_common_delta_browser_minus_fast"] for row in pair_rows]
    result: dict[str, Any] = {
        "pair_count": len(pair_rows),
        "browser_better": sum(1 for value in medians if value > 0),
        "browser_worse": sum(1 for value in medians if value < 0),
        "tie": sum(1 for value in medians if value == 0),
        "median_of_pair_medians": median(medians) if medians else None,
    }
    for threshold in (10000, 50000, 100000, 150000):
        subset = [
            row
            for row in pair_rows
            if row["latest_common_iteration"] is not None
            and row["latest_common_iteration"] >= threshold
        ]
        values = [row["median_common_delta_browser_minus_fast"] for row in subset]
        result[f"common_ge_{threshold}"] = {
            "n": len(subset),
            "browser_better": sum(1 for value in values if value > 0),
            "browser_worse": sum(1 for value in values if value < 0),
            "median_of_pair_medians": median(values) if values else None,
        }
    return result, pair_rows


def _matched_knobs(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            knob = str(row.get("knob") or "unknown").lstrip("_")
            metric = str(row.get("metric") or "unknown")
            out.setdefault(knob, {})[metric] = {
                "contrast_count": _int(row.get("contrast_count")),
                "median_delta_high_minus_low": _float(row.get("median_delta_high_minus_low")),
                "positive_count": _int(row.get("positive_count")),
                "negative_count": _int(row.get("negative_count")),
            }
    return out


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines)


def _render_markdown(report: dict[str, Any]) -> str:
    families = report["families"]
    family_rows = [
        [
            family,
            stats["n"],
            100.0 * stats["improved_pct"] if stats["improved_pct"] is not None else None,
            stats["median_delta_steps"],
            stats["p25_delta_steps"],
            stats["p75_delta_steps"],
            stats["median_relative_delta_pct"],
        ]
        for family, stats in families.items()
    ]
    survival_rows = [
        [
            name,
            stats["n"],
            stats["median_delta_steps"],
            stats["median_relative_delta_pct"],
            100.0 * stats["improved_pct"] if stats["improved_pct"] is not None else None,
        ]
        for name, stats in report["survival_stochasticity"].items()
    ]
    repeat_rows = [
        [
            name,
            stats["n"],
            stats["median_delta_steps"],
            stats["median_relative_delta_pct"],
            100.0 * stats["improved_pct"] if stats["improved_pct"] is not None else None,
        ]
        for name, stats in report["mix_repeat"].items()
    ]
    knob_rows: list[list[Any]] = []
    for knob, metrics in report["matched_knobs"].items():
        for metric, stats in metrics.items():
            knob_rows.append(
                [
                    knob,
                    metric,
                    stats["contrast_count"],
                    stats["median_delta_high_minus_low"],
                    stats["positive_count"],
                    stats["negative_count"],
                ]
            )
    render = report["checkpoint_matched_render"]
    render_rows = [
        [
            "all common checkpoints",
            render["pair_count"],
            render["browser_better"],
            render["browser_worse"],
            render["median_of_pair_medians"],
        ]
    ]
    for threshold in (10000, 50000, 100000, 150000):
        stats = render[f"common_ge_{threshold}"]
        render_rows.append(
            [
                f"common >= {threshold}",
                stats["n"],
                stats["browser_better"],
                stats["browser_worse"],
                stats["median_of_pair_medians"],
            ]
        )

    return "\n\n".join(
        [
            "# CurvyTron 212-run clean report",
            "## Survival Improvement Magnitude",
            _markdown_table(
                [
                    "Family",
                    "N",
                    "% improved",
                    "Median gain",
                    "P25 gain",
                    "P75 gain",
                    "Median relative gain %",
                ],
                family_rows,
            ),
            "## Checkpoint-Matched Render Comparison",
            "Delta is browser minus fast, using only checkpoints both paired runs evaluated.",
            _markdown_table(
                ["Slice", "Pairs", "Browser better", "Browser worse", "Median pair delta"],
                render_rows,
            ),
            "## Survival Stochasticity",
            _markdown_table(
                ["Level", "N", "Median gain", "Median relative gain %", "% improved"],
                survival_rows,
            ),
            "## Mix Repeat Proxy",
            _markdown_table(
                ["Repeat", "N", "Median gain", "Median relative gain %", "% improved"],
                repeat_rows,
            ),
            "## Matched Compute Knobs",
            _markdown_table(
                [
                    "Knob",
                    "Metric",
                    "Contrasts",
                    "Median high-low delta",
                    "Positive",
                    "Negative",
                ],
                knob_rows,
            ),
        ]
    ) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(summary: dict[str, dict[str, Any]], *, label_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, stats in summary.items():
        rows.append({label_name: label, **stats})
    return rows


def _knob_rows(matched_knobs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for knob, metrics in matched_knobs.items():
        for metric, stats in metrics.items():
            rows.append({"knob": knob, "metric": metric, **stats})
    return rows


def build_report(analysis_dir: Path, status_json: Path) -> dict[str, Any]:
    row_path = analysis_dir / "row_projections.csv"
    rows = _load_rows(row_path)
    usable = [row for row in rows if row.get("delta") is not None]
    robust = [
        row
        for row in usable
        if row.get("eval_points", 0) >= 2 and (row.get("latest_iteration") or 0) >= 100000
    ]
    survival = [row for row in usable if row.get("family") == "survival"]
    mix = [row for row in usable if row.get("family") in {"mix2", "mix3"}]
    checkpoint_matched_render, checkpoint_matched_render_rows = _checkpoint_matched_render(
        status_json,
        analysis_dir / "fair_pair_comparisons.csv",
    )
    return {
        "source": {
            "analysis_dir": analysis_dir.as_posix(),
            "status_json": status_json.as_posix(),
            "row_count": len(rows),
        },
        "overall": _summary(usable),
        "robust_eval_points_ge_2_iter_ge_100k": _summary(robust),
        "families": _group_summary(usable, "family"),
        "survival_stochasticity": _group_summary(survival, "stochasticity"),
        "mix_repeat": _group_summary(mix, "repeat"),
        "checkpoint_matched_render": checkpoint_matched_render,
        "checkpoint_matched_render_rows": checkpoint_matched_render_rows,
        "matched_knobs": _matched_knobs(analysis_dir / "matched_knob_summary_mix3.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--status-json", type=Path, default=DEFAULT_STATUS_JSON)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR / "clean_report",
    )
    args = parser.parse_args()

    report = build_report(args.analysis_dir, args.status_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "clean_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "clean_report.md").write_text(
        _render_markdown(report),
        encoding="utf-8",
    )
    _write_csv(
        args.output_dir / "family_summary.csv",
        _summary_rows(report["families"], label_name="family"),
    )
    _write_csv(
        args.output_dir / "survival_stochasticity_summary.csv",
        _summary_rows(report["survival_stochasticity"], label_name="stochasticity"),
    )
    _write_csv(
        args.output_dir / "mix_repeat_summary.csv",
        _summary_rows(report["mix_repeat"], label_name="repeat"),
    )
    _write_csv(
        args.output_dir / "checkpoint_matched_render_pairs.csv",
        report["checkpoint_matched_render_rows"],
    )
    _write_csv(
        args.output_dir / "matched_compute_knobs.csv",
        _knob_rows(report["matched_knobs"]),
    )
    print((args.output_dir / "clean_report.md").as_posix())


if __name__ == "__main__":
    main()
