#!/usr/bin/env python3
"""Analyze local CZ26 CurvyTron grid snapshots.

This is intentionally local-only. It joins a CZ26 manifest with already-pulled
eval/status and optional tournament rating snapshots, then emits:

- one row per manifest run;
- per-axis projection tables for Grid A and Grid B;
- a compact Markdown summary.

The script does not pull from Modal. Use the deployed status tools to create
snapshots, then run this on those local files.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from curvyzero.analysis.eval_curves import EvalCurve, build_curves


DEFAULT_METRICS = ("mean_survival", "mean_training_reward")
ITERATION_RE = re.compile(r"iteration_(\d+)")


def load_json_with_modal_noise(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"(?m)^[ \t]*(?:\[(?=[\s{\]])|\{(?=\s*\"))", text):
        decoder = json.JSONDecoder()
        value, _end = decoder.raw_decode(text[match.start() :])
        return value
    raise ValueError(f"{path} does not contain a JSON object or array")


def payload_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in (
            "rows",
            "runs",
            "scores",
            "eval_summary",
            "standings",
            "ratings",
            "rating_rows",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def manifest_rows(path: Path) -> list[dict[str, Any]]:
    payload = load_json_with_modal_noise(path)
    rows = payload_rows(payload)
    if not rows:
        raise ValueError(f"{path} has no manifest rows")
    return rows


def run_id(row: Mapping[str, Any]) -> str | None:
    for key in ("run_id", "short_name", "name"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def row_attempt_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("attempt_id")
    if isinstance(value, str) and value:
        return value
    train_kwargs = row.get("train_kwargs")
    if isinstance(train_kwargs, Mapping):
        value = train_kwargs.get("attempt_id")
        if isinstance(value, str) and value:
            return value
    return None


def metric_values(curve: EvalCurve, metric: str) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for point in curve.sorted_points():
        value = point.metrics.get(metric)
        if value is None:
            continue
        try:
            values.append((point.iteration, float(value)))
        except (TypeError, ValueError):
            continue
    return values


def common_endpoint(curves: Iterable[EvalCurve], metric: str) -> int | None:
    common: set[int] | None = None
    for curve in curves:
        iterations = {iteration for iteration, _value in metric_values(curve, metric)}
        if not iterations:
            continue
        common = iterations if common is None else common & iterations
    return max(common) if common else None


def value_at_or_before(values: Sequence[tuple[int, float]], endpoint: int | None) -> float | None:
    if endpoint is None:
        return None
    eligible = [value for iteration, value in values if iteration <= endpoint]
    return eligible[-1] if eligible else None


def auc_until(values: Sequence[tuple[int, float]], endpoint: int | None) -> float | None:
    eligible = [value for iteration, value in values if endpoint is None or iteration <= endpoint]
    return mean(eligible) if eligible else None


def score_curve_for_metric(curve: EvalCurve | None, metric: str, endpoint: int | None) -> dict[str, Any]:
    if curve is None:
        return {
            "points": 0,
            "first_iter": None,
            "first": None,
            "matched_endpoint": endpoint,
            "matched": None,
            "latest_iter": None,
            "latest": None,
            "best_iter": None,
            "best": None,
            "auc": None,
            "retention_latest_over_best": None,
        }
    values = metric_values(curve, metric)
    if not values:
        return {
            "points": 0,
            "first_iter": None,
            "first": None,
            "matched_endpoint": endpoint,
            "matched": None,
            "latest_iter": None,
            "latest": None,
            "best_iter": None,
            "best": None,
            "auc": None,
            "retention_latest_over_best": None,
        }
    best_iter, best_value = max(values, key=lambda item: item[1])
    latest_iter, latest_value = values[-1]
    return {
        "points": len(values),
        "first_iter": values[0][0],
        "first": values[0][1],
        "matched_endpoint": endpoint,
        "matched": value_at_or_before(values, endpoint),
        "latest_iter": latest_iter,
        "latest": latest_value,
        "best_iter": best_iter,
        "best": best_value,
        "auc": auc_until(values, endpoint),
        "retention_latest_over_best": (latest_value / best_value) if best_value else None,
    }


def checkpoint_iteration(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    matches = list(ITERATION_RE.finditer(value))
    if not matches:
        return None
    return int(matches[-1].group(1))


def tournament_row_run_id(row: Mapping[str, Any]) -> str | None:
    for key in ("run_id", "short_name", "training_run_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    for container_key in ("checkpoint", "metadata", "source", "training"):
        container = row.get(container_key)
        if isinstance(container, Mapping):
            value = tournament_row_run_id(container)
            if value:
                return value
    checkpoint_ref = row.get("checkpoint_ref")
    if isinstance(checkpoint_ref, str):
        parts = checkpoint_ref.split("/")
        if "lightzero-curvytron-visual-survival" in parts:
            index = parts.index("lightzero-curvytron-visual-survival")
            if index + 1 < len(parts):
                return parts[index + 1]
    return None


def tournament_row_iteration(row: Mapping[str, Any]) -> int | None:
    for key in ("iteration", "checkpoint_iteration", "train_iter"):
        value = row.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    for key in ("checkpoint_ref", "checkpoint_id", "checkpoint"):
        value = row.get(key)
        if isinstance(value, Mapping):
            nested = tournament_row_iteration(value)
            if nested is not None:
                return nested
        iteration = checkpoint_iteration(value)
        if iteration is not None:
            return iteration
    return None


def numeric(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def summarize_tournament_rows(run_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ranked: list[tuple[float, dict[str, Any]]] = []
    top_counts = Counter()
    games = 0.0
    distinct_opponents = 0.0
    latest_row: dict[str, Any] | None = None
    latest_iter = -1
    for row in run_rows:
        rank = numeric(row, ("rank", "current_rank", "placement"))
        if rank is not None:
            ranked.append((rank, row))
            if rank <= 10:
                top_counts["top10"] += 1
            if rank <= 30:
                top_counts["top30"] += 1
            if rank <= 100:
                top_counts["top100"] += 1
        games += numeric(row, ("games", "valid_games", "game_count", "n_games")) or 0.0
        distinct_opponents += (
            numeric(row, ("distinct_opponents", "opponent_count", "distinct_opponent_count"))
            or 0.0
        )
        iteration = tournament_row_iteration(row)
        if iteration is not None and iteration >= latest_iter:
            latest_iter = iteration
            latest_row = row
        elif row.get("latest_for_run") is True:
            latest_row = row
    best_rank, best_row = min(ranked, key=lambda item: item[0]) if ranked else (None, {})
    latest_rank = numeric(latest_row or {}, ("rank", "current_rank", "placement"))
    return {
        "rating_rows": len(run_rows),
        "best_rank": best_rank,
        "best_rating": numeric(best_row, ("rating", "elo", "score")),
        "latest_tournament_iter": latest_iter if latest_iter >= 0 else None,
        "latest_rank": latest_rank,
        "top10_rows": top_counts["top10"],
        "top30_rows": top_counts["top30"],
        "top100_rows": top_counts["top100"],
        "hit_top10": top_counts["top10"] > 0,
        "hit_top30": top_counts["top30"] > 0,
        "hit_top100": top_counts["top100"] > 0,
        "games": games or None,
        "distinct_opponents_sum": distinct_opponents or None,
    }


def prefixed(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def tournament_summary_by_run(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = payload_rows(load_json_with_modal_noise(path))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rid = tournament_row_run_id(row)
        if rid:
            grouped[rid].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for rid, run_rows in grouped.items():
        learned_rows = [
            row
            for row in run_rows
            if (tournament_row_iteration(row) is not None and tournament_row_iteration(row) > 0)
        ]
        summaries[rid] = {
            **summarize_tournament_rows(run_rows),
            **prefixed("learned", summarize_tournament_rows(learned_rows)),
            "seed_rating_rows": len(run_rows) - len(learned_rows),
        }
    return summaries


def manifest_axis(row: Mapping[str, Any]) -> dict[str, Any]:
    train_kwargs = row.get("train_kwargs") if isinstance(row.get("train_kwargs"), Mapping) else {}
    return {
        "grid": row.get("grid"),
        "row_kind": row.get("row_kind"),
        "reward_tag": row.get("reward_tag"),
        "reward_outcome_alpha": row.get("reward_outcome_alpha"),
        "noise_tag": row.get("noise_tag"),
        "leaderboard_immortal_tag": row.get("leaderboard_immortal_tag"),
        "leaderboard_immortal_probability": row.get("leaderboard_immortal_probability"),
        "recipe_code": row.get("recipe_code"),
        "recipe_slot_counts": row.get("recipe_slot_counts"),
        "max_train_iter": train_kwargs.get("max_train_iter"),
        "save_ckpt_after_iter": train_kwargs.get("save_ckpt_after_iter"),
        "opponent_assignment_refresh_interval_train_iter": train_kwargs.get(
            "opponent_assignment_refresh_interval_train_iter"
        ),
    }


def build_row_summaries(
    rows: Sequence[dict[str, Any]],
    curves_by_run: Mapping[str, EvalCurve],
    tournament_by_run: Mapping[str, dict[str, Any]],
    *,
    metrics: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    endpoints: dict[str, dict[str, int | None]] = {}
    for metric in metrics:
        endpoints[metric] = {
            "all": common_endpoint(curves_by_run.values(), metric),
            "grid_a": common_endpoint(
                [
                    curves_by_run[rid]
                    for row in rows
                    if row.get("grid") == "grid_a"
                    for rid in [run_id(row)]
                    if rid in curves_by_run
                ],
                metric,
            ),
            "grid_b": common_endpoint(
                [
                    curves_by_run[rid]
                    for row in rows
                    if row.get("grid") == "grid_b"
                    for rid in [run_id(row)]
                    if rid in curves_by_run
                ],
                metric,
            ),
        }

    summaries: list[dict[str, Any]] = []
    for row in rows:
        rid = run_id(row)
        if not rid:
            continue
        grid = str(row.get("grid") or "")
        curve = curves_by_run.get(rid)
        metric_scores = {
            metric: score_curve_for_metric(
                curve,
                metric,
                endpoints.get(metric, {}).get(grid) or endpoints.get(metric, {}).get("all"),
            )
            for metric in metrics
        }
        summaries.append(
            {
                "run_id": rid,
                "attempt_id": row_attempt_id(row),
                **manifest_axis(row),
                "metrics": metric_scores,
                "tournament": tournament_by_run.get(rid, {}),
                "missing_eval_curve": curve is None,
                "missing_tournament_rows": rid not in tournament_by_run,
            }
        )
    return summaries, {"common_endpoints": endpoints}


def mean_or_none(values: Iterable[Any]) -> float | None:
    numeric_values: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    return mean(numeric_values) if numeric_values else None


def project(
    row_summaries: Sequence[Mapping[str, Any]],
    *,
    grid: str,
    axis: str,
    metric: str,
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in row_summaries:
        if row.get("grid") != grid:
            continue
        grouped[row.get(axis)].append(row)

    output: list[dict[str, Any]] = []
    for value, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        scores = [
            row.get("metrics", {}).get(metric, {})
            for row in rows
            if isinstance(row.get("metrics"), Mapping)
        ]
        tournaments = [
            row.get("tournament", {})
            for row in rows
            if isinstance(row.get("tournament"), Mapping)
        ]
        output.append(
            {
                "grid": grid,
                "axis": axis,
                "value": value,
                "rows": len(rows),
                "rows_with_metric": sum(bool(score.get("points")) for score in scores),
                "mean_first": mean_or_none(score.get("first") for score in scores),
                "mean_matched": mean_or_none(score.get("matched") for score in scores),
                "mean_latest": mean_or_none(score.get("latest") for score in scores),
                "mean_best": mean_or_none(score.get("best") for score in scores),
                "mean_auc": mean_or_none(score.get("auc") for score in scores),
                "mean_retention_latest_over_best": mean_or_none(
                    score.get("retention_latest_over_best") for score in scores
                ),
                "best_tournament_rank": min(
                    (
                        float(value)
                        for tournament in tournaments
                        for value in [tournament.get("best_rank")]
                        if value is not None
                    ),
                    default=None,
                ),
                "mean_best_tournament_rank": mean_or_none(
                    tournament.get("best_rank") for tournament in tournaments
                ),
                "mean_latest_tournament_rank": mean_or_none(
                    tournament.get("latest_rank") for tournament in tournaments
                ),
                "mean_rating_rows": mean_or_none(
                    tournament.get("rating_rows") for tournament in tournaments
                ),
                "learned_best_tournament_rank": min(
                    (
                        float(value)
                        for tournament in tournaments
                        for value in [tournament.get("learned_best_rank")]
                        if value is not None
                    ),
                    default=None,
                ),
                "learned_mean_best_tournament_rank": mean_or_none(
                    tournament.get("learned_best_rank") for tournament in tournaments
                ),
                "learned_mean_latest_tournament_rank": mean_or_none(
                    tournament.get("learned_latest_rank") for tournament in tournaments
                ),
                "learned_mean_rating_rows": mean_or_none(
                    tournament.get("learned_rating_rows") for tournament in tournaments
                ),
                "top10_rows": sum(int(tournament.get("top10_rows") or 0) for tournament in tournaments),
                "top30_rows": sum(int(tournament.get("top30_rows") or 0) for tournament in tournaments),
                "top100_rows": sum(int(tournament.get("top100_rows") or 0) for tournament in tournaments),
                "top10_runs": sum(bool(tournament.get("hit_top10")) for tournament in tournaments),
                "top30_runs": sum(bool(tournament.get("hit_top30")) for tournament in tournaments),
                "top100_runs": sum(bool(tournament.get("hit_top100")) for tournament in tournaments),
                "learned_top10_rows": sum(
                    int(tournament.get("learned_top10_rows") or 0) for tournament in tournaments
                ),
                "learned_top30_rows": sum(
                    int(tournament.get("learned_top30_rows") or 0) for tournament in tournaments
                ),
                "learned_top100_rows": sum(
                    int(tournament.get("learned_top100_rows") or 0) for tournament in tournaments
                ),
                "learned_top10_runs": sum(
                    bool(tournament.get("learned_hit_top10")) for tournament in tournaments
                ),
                "learned_top30_runs": sum(
                    bool(tournament.get("learned_hit_top30")) for tournament in tournaments
                ),
                "learned_top100_runs": sum(
                    bool(tournament.get("learned_hit_top100")) for tournament in tournaments
                ),
                "missing_eval_rows": sum(bool(row.get("missing_eval_curve")) for row in rows),
                "missing_tournament_rows": sum(
                    bool(row.get("missing_tournament_rows")) for row in rows
                ),
            }
        )
    return output


def build_projections(row_summaries: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> dict[str, Any]:
    grids = {
        "grid_a": ("reward_tag", "noise_tag", "leaderboard_immortal_tag", "recipe_code"),
        "grid_b": ("recipe_code", "noise_tag", "leaderboard_immortal_tag"),
    }
    projections: dict[str, Any] = {}
    for metric in metrics:
        metric_tables: dict[str, Any] = {}
        for grid, axes in grids.items():
            metric_tables[grid] = {
                axis: project(row_summaries, grid=grid, axis=axis, metric=metric)
                for axis in axes
            }
        projections[metric] = metric_tables
    return projections


def render_projection_table(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "value",
        "rows",
        "rows_with_metric",
        "mean_matched",
        "mean_latest",
        "mean_best",
        "mean_auc",
        "mean_retention_latest_over_best",
        "learned_best_tournament_rank",
        "learned_mean_best_tournament_rank",
        "learned_mean_latest_tournament_rank",
        "learned_mean_rating_rows",
        "learned_top10_runs",
        "learned_top30_runs",
        "learned_top100_runs",
        "learned_top10_rows",
        "learned_top30_rows",
        "learned_top100_rows",
        "missing_eval_rows",
        "missing_tournament_rows",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(display(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(payload: Mapping[str, Any], metrics: Sequence[str]) -> str:
    lines = [
        "# CZ26 Grid Analysis",
        "",
        "## Completeness",
        "",
        f"- Manifest rows: `{payload['manifest_row_count']}`",
        f"- Eval curves: `{payload['eval_curve_count']}`",
        f"- Rows missing eval curves: `{payload['missing_eval_curve_count']}`",
        f"- Rows missing tournament rows: `{payload['missing_tournament_row_count']}`",
        "",
        "## Common Endpoints",
        "",
        "```json",
        json.dumps(payload["metadata"].get("common_endpoints", {}), indent=2, sort_keys=True),
        "```",
        "",
    ]
    projections = payload.get("projections", {})
    for metric in metrics:
        metric_tables = projections.get(metric, {})
        lines.extend([f"## Metric: `{metric}`", ""])
        for grid, axes in metric_tables.items():
            for axis, table in axes.items():
                lines.extend([f"### {grid} by `{axis}`", "", render_projection_table(table), ""])
    return "\n".join(lines).rstrip() + "\n"


def parse_metrics(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_METRICS
    metrics: list[str] = []
    for value in values:
        metrics.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(metrics) or DEFAULT_METRICS


def analyze(
    *,
    manifest_path: Path,
    eval_status_path: Path | None,
    tournament_rating_path: Path | None,
    metrics: Sequence[str],
) -> dict[str, Any]:
    rows = manifest_rows(manifest_path)
    curves_by_run: dict[str, EvalCurve] = {}
    if eval_status_path is not None:
        for curve in build_curves(load_json_with_modal_noise(eval_status_path)):
            curves_by_run[curve.run_id] = curve
    tournament_by_run = tournament_summary_by_run(tournament_rating_path)
    row_summaries, metadata = build_row_summaries(
        rows,
        curves_by_run,
        tournament_by_run,
        metrics=metrics,
    )
    return {
        "schema_id": "curvyzero_cz26_grid_analysis/v0",
        "manifest": manifest_path.as_posix(),
        "eval_status": eval_status_path.as_posix() if eval_status_path else None,
        "tournament_rating": tournament_rating_path.as_posix() if tournament_rating_path else None,
        "manifest_row_count": len(rows),
        "eval_curve_count": len(curves_by_run),
        "missing_eval_curve_count": sum(bool(row["missing_eval_curve"]) for row in row_summaries),
        "missing_tournament_row_count": sum(
            bool(row["missing_tournament_rows"]) for row in row_summaries
        ),
        "metadata": metadata,
        "rows": row_summaries,
        "projections": build_projections(row_summaries, metrics),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze local CZ26 grid snapshots.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval-status", type=Path)
    parser.add_argument("--tournament-rating", type=Path)
    parser.add_argument(
        "--metric",
        action="append",
        help="Metric to analyze. May be repeated or comma-separated.",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)
    metrics = parse_metrics(args.metric)
    payload = analyze(
        manifest_path=args.manifest,
        eval_status_path=args.eval_status,
        tournament_rating_path=args.tournament_rating,
        metrics=metrics,
    )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(payload, metrics), encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(render_markdown(payload, metrics), end="")


if __name__ == "__main__":
    main()
