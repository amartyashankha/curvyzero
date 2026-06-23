#!/usr/bin/env python3
"""Build second-pass CZ26 analysis tables from local snapshots.

This script is deliberately offline. It reads already-pulled local artifacts and
writes plain JSON/Markdown tables for the analysis docs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_curvytron_cz26_grid import (
    checkpoint_iteration,
    load_json_with_modal_noise,
    numeric,
    payload_rows,
    tournament_row_iteration,
    tournament_row_run_id,
)


DEFAULT_HORIZONS = (30_000, 170_000, 300_000)


def mean_or_none(values: Iterable[Any]) -> float | None:
    nums: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            continue
    return mean(nums) if nums else None


def median_or_none(values: Iterable[Any]) -> float | None:
    nums: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            continue
    return median(nums) if nums else None


def display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(display(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def eval_rows_by_run(eval_status_path: Path) -> dict[str, dict[str, Any]]:
    rows = payload_rows(load_json_with_modal_noise(eval_status_path))
    return {
        str(row["run_id"]): row
        for row in rows
        if isinstance(row.get("run_id"), str) and isinstance(row.get("eval_checkpoints"), list)
    }


def joined_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain joined rows")
    return [row for row in rows if isinstance(row, dict)]


def point_iteration(point: Mapping[str, Any]) -> int | None:
    value = point.get("iteration")
    if isinstance(value, int):
        return value
    return checkpoint_iteration(point.get("checkpoint"))


def eval_points(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    points = row.get("eval_checkpoints")
    if not isinstance(points, list):
        return []
    output: list[dict[str, Any]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        iteration = point_iteration(point)
        if iteration is None:
            continue
        with_iteration = dict(point)
        with_iteration["_iteration"] = iteration
        output.append(with_iteration)
    return sorted(output, key=lambda point: point["_iteration"])


def point_by_iteration(row: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {point["_iteration"]: point for point in eval_points(row)}


def reward_parts(point: Mapping[str, Any]) -> dict[str, float | None]:
    components = point.get("mean_reward_components")
    if not isinstance(components, Mapping):
        components = {}
    reward = numeric(point, ("mean_training_reward", "mean_reward"))
    survival_component = numeric(components, ("survival",))
    bonus_component = numeric(components, ("bonus",))
    if bonus_component is None:
        bonus_component = numeric(point, ("mean_bonus_reward",))
    residual = None
    if reward is not None:
        residual = reward - (survival_component or 0.0) - (bonus_component or 0.0)
    return {
        "training_reward": reward,
        "survival_component": survival_component,
        "bonus_component": bonus_component,
        "outcome_residual": residual,
    }


def point_metrics(point: Mapping[str, Any]) -> dict[str, Any]:
    action_summary = point.get("action_summary")
    if not isinstance(action_summary, Mapping):
        action_summary = {}
    collapsed = bool(action_summary.get("collapsed"))
    return {
        "iteration": point["_iteration"],
        "survival": numeric(point, ("mean_steps", "mean_survival")),
        "median_steps": numeric(point, ("median_steps",)),
        "min_steps": numeric(point, ("min_steps",)),
        "max_steps": numeric(point, ("max_steps",)),
        "action_entropy": numeric(point, ("action_entropy",)),
        "top_action_fraction": numeric(action_summary, ("top_action_fraction",)),
        "collapsed": collapsed,
        "row_action_collapsed_count": numeric(point, ("row_action_collapsed_count",)),
        "bonus_pickups": numeric(point, ("mean_bonus_pickup_count",)),
        **reward_parts(point),
    }


def exact_point_metric(eval_by_run: Mapping[str, Mapping[str, Any]], run_id: str, horizon: int) -> dict[str, Any] | None:
    row = eval_by_run.get(run_id)
    if row is None:
        return None
    point = point_by_iteration(row).get(horizon)
    if point is None:
        return None
    return point_metrics(point)


def all_point_metrics(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [point_metrics(point) for point in eval_points(row)]


def best_point(points: Sequence[Mapping[str, Any]], metric: str) -> Mapping[str, Any] | None:
    candidates = [point for point in points if point.get(metric) is not None]
    return max(candidates, key=lambda point: float(point[metric])) if candidates else None


def tournament_rows_by_run(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows = payload_rows(load_json_with_modal_noise(path))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rid = tournament_row_run_id(row)
        if rid:
            grouped[rid].append(row)
    return grouped


def best_learned_tournament_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    learned: list[dict[str, Any]] = []
    for row in rows:
        iteration = tournament_row_iteration(row)
        rank = numeric(row, ("rank", "current_rank", "placement"))
        if iteration is not None and iteration > 0 and rank is not None:
            learned.append(row)
    if not learned:
        return None
    return min(learned, key=lambda row: float(numeric(row, ("rank", "current_rank", "placement"))))


def summarize_run(
    joined: Mapping[str, Any],
    eval_by_run: Mapping[str, Mapping[str, Any]],
    tournament_by_run: Mapping[str, Sequence[dict[str, Any]]],
    horizons: Sequence[int],
) -> dict[str, Any]:
    rid = str(joined["run_id"])
    eval_row = eval_by_run.get(rid)
    points = all_point_metrics(eval_row or {})
    first = points[0] if points else None
    latest = points[-1] if points else None
    best_survival = best_point(points, "survival")
    best_reward = best_point(points, "training_reward")
    best_tournament = best_learned_tournament_row(tournament_by_run.get(rid, ()))
    tournament = joined.get("tournament") if isinstance(joined.get("tournament"), Mapping) else {}
    output = {
        "run_id": rid,
        "grid": joined.get("grid"),
        "reward": joined.get("reward_tag"),
        "noise": joined.get("noise_tag"),
        "leaderboard_immortality": joined.get("leaderboard_immortal_tag"),
        "recipe": joined.get("recipe_code"),
        "train_status": (eval_row or {}).get("train_status"),
        "eval_points": len(points),
        "latest_eval_iteration": latest.get("iteration") if latest else None,
        "first_survival": first.get("survival") if first else None,
        "latest_survival": latest.get("survival") if latest else None,
        "best_survival": best_survival.get("survival") if best_survival else None,
        "best_survival_iteration": best_survival.get("iteration") if best_survival else None,
        "survival_retention": (
            latest.get("survival") / best_survival.get("survival")
            if latest
            and best_survival
            and latest.get("survival") is not None
            and best_survival.get("survival")
            else None
        ),
        "first_reward": first.get("training_reward") if first else None,
        "latest_reward": latest.get("training_reward") if latest else None,
        "best_reward": best_reward.get("training_reward") if best_reward else None,
        "best_reward_iteration": best_reward.get("iteration") if best_reward else None,
        "latest_outcome_residual": latest.get("outcome_residual") if latest else None,
        "latest_bonus_component": latest.get("bonus_component") if latest else None,
        "latest_top_action_fraction": latest.get("top_action_fraction") if latest else None,
        "latest_action_collapsed": latest.get("collapsed") if latest else None,
        "latest_row_action_collapsed_count": latest.get("row_action_collapsed_count") if latest else None,
        "collapsed_checkpoint_count": sum(bool(point.get("collapsed")) for point in points),
        "row_action_collapsed_sum": sum(point.get("row_action_collapsed_count") or 0 for point in points),
        "learned_best_rank": tournament.get("learned_best_rank"),
        "learned_latest_rank": tournament.get("learned_latest_rank"),
        "learned_rating_rows": tournament.get("learned_rating_rows"),
        "learned_games": tournament.get("learned_games"),
        "learned_top100_rows": tournament.get("learned_top100_rows"),
        "best_tournament_iteration": tournament_row_iteration(best_tournament or {}),
        "best_tournament_games": numeric(best_tournament or {}, ("games", "valid_games", "game_count")),
        "best_tournament_battles": numeric(best_tournament or {}, ("battles",)),
    }
    for horizon in horizons:
        metrics = exact_point_metric(eval_by_run, rid, horizon)
        output[f"survival_at_{horizon}"] = metrics.get("survival") if metrics else None
        output[f"reward_at_{horizon}"] = metrics.get("training_reward") if metrics else None
        output[f"outcome_residual_at_{horizon}"] = (
            metrics.get("outcome_residual") if metrics else None
        )
        output[f"action_collapsed_at_{horizon}"] = metrics.get("collapsed") if metrics else None
    return output


def axis_for_grid(grid: str) -> tuple[str, ...]:
    if grid == "grid_a":
        return ("reward", "noise", "leaderboard_immortality", "recipe")
    if grid == "grid_b":
        return ("recipe", "noise", "leaderboard_immortality")
    return ()


def summarize_axis_at_horizon(rows: Sequence[Mapping[str, Any]], grid: str, axis: str, horizon: int) -> list[dict[str, Any]]:
    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("grid") == grid:
            grouped[row.get(axis)].append(row)
    output: list[dict[str, Any]] = []
    for value, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        survival_key = f"survival_at_{horizon}"
        reward_key = f"reward_at_{horizon}"
        residual_key = f"outcome_residual_at_{horizon}"
        collapsed_key = f"action_collapsed_at_{horizon}"
        output.append(
            {
                "grid": grid,
                "axis": axis,
                "value": value,
                "expected_rows": len(group),
                "rows_at_horizon": sum(row.get(survival_key) is not None for row in group),
                "horizon": horizon,
                "mean_survival": mean_or_none(row.get(survival_key) for row in group),
                "median_survival": median_or_none(row.get(survival_key) for row in group),
                "mean_training_reward": mean_or_none(row.get(reward_key) for row in group),
                "mean_outcome_residual": mean_or_none(row.get(residual_key) for row in group),
                "collapsed_rows": sum(row.get(collapsed_key) is True for row in group),
                "mean_learned_best_rank": mean_or_none(row.get("learned_best_rank") for row in group),
                "best_learned_rank": min(
                    (float(row["learned_best_rank"]) for row in group if row.get("learned_best_rank") is not None),
                    default=None,
                ),
                "learned_top100_runs": sum(
                    (row.get("learned_best_rank") is not None and float(row["learned_best_rank"]) <= 100)
                    for row in group
                ),
            }
        )
    return output


def lifecycle_axis_summary(rows: Sequence[Mapping[str, Any]], grid: str, axis: str) -> list[dict[str, Any]]:
    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("grid") == grid:
            grouped[row.get(axis)].append(row)
    output: list[dict[str, Any]] = []
    for value, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        output.append(
            {
                "grid": grid,
                "axis": axis,
                "value": value,
                "rows": len(group),
                "mean_first_survival": mean_or_none(row.get("first_survival") for row in group),
                "mean_best_survival": mean_or_none(row.get("best_survival") for row in group),
                "mean_latest_survival": mean_or_none(row.get("latest_survival") for row in group),
                "mean_survival_retention": mean_or_none(row.get("survival_retention") for row in group),
                "mean_best_reward": mean_or_none(row.get("best_reward") for row in group),
                "mean_latest_reward": mean_or_none(row.get("latest_reward") for row in group),
                "collapsed_latest_rows": sum(row.get("latest_action_collapsed") is True for row in group),
                "collapsed_ever_rows": sum((row.get("collapsed_checkpoint_count") or 0) > 0 for row in group),
                "row_action_collapsed_sum": sum(row.get("row_action_collapsed_sum") or 0 for row in group),
                "best_learned_rank": min(
                    (float(row["learned_best_rank"]) for row in group if row.get("learned_best_rank") is not None),
                    default=None,
                ),
                "mean_learned_best_rank": mean_or_none(row.get("learned_best_rank") for row in group),
                "learned_top100_runs": sum(
                    (row.get("learned_best_rank") is not None and float(row["learned_best_rank"]) <= 100)
                    for row in group
                ),
            }
        )
    return output


def coverage_rows(run_rows: Sequence[Mapping[str, Any]], horizons: Sequence[int]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for grid in ("grid_a", "grid_b"):
        group = [row for row in run_rows if row.get("grid") == grid]
        base = {
            "grid": grid,
            "rows": len(group),
            "completed": sum(row.get("train_status") == "completed" for row in group),
            "running": sum(row.get("train_status") == "running" for row in group),
            "min_latest_eval_iteration": min(
                (int(row["latest_eval_iteration"]) for row in group if row.get("latest_eval_iteration") is not None),
                default=None,
            ),
            "max_latest_eval_iteration": max(
                (int(row["latest_eval_iteration"]) for row in group if row.get("latest_eval_iteration") is not None),
                default=None,
            ),
            "mean_eval_points": mean_or_none(row.get("eval_points") for row in group),
        }
        for horizon in horizons:
            base[f"rows_at_{horizon}"] = sum(row.get(f"survival_at_{horizon}") is not None for row in group)
        output.append(base)
    return output


def summarize_deltas(deltas: Sequence[float], *, higher_is_better: bool) -> dict[str, Any]:
    if not deltas:
        return {
            "pairs": 0,
            "mean_delta": None,
            "median_delta": None,
            "better_count": 0,
            "worse_count": 0,
        }
    if higher_is_better:
        better = sum(delta > 0 for delta in deltas)
        worse = sum(delta < 0 for delta in deltas)
    else:
        better = sum(delta < 0 for delta in deltas)
        worse = sum(delta > 0 for delta in deltas)
    return {
        "pairs": len(deltas),
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        "better_count": better,
        "worse_count": worse,
    }


def exact_horizon_contrasts(run_rows: Sequence[Mapping[str, Any]], horizons: Sequence[int]) -> list[dict[str, Any]]:
    specs = [
        (
            "grid_a",
            "reward",
            ("recipe", "noise", "leaderboard_immortality"),
            (("out0", "out33"), ("out0", "out67"), ("out0", "out100"), ("out33", "out67"), ("out67", "out100")),
        ),
        (
            "grid_a",
            "noise",
            ("recipe", "reward", "leaderboard_immortality"),
            (("n0", "n10"), ("n0", "n20"), ("n10", "n20")),
        ),
        (
            "grid_a",
            "leaderboard_immortality",
            ("recipe", "reward", "noise"),
            (("imm0", "imm10"),),
        ),
        (
            "grid_a",
            "recipe",
            ("reward", "noise", "leaderboard_immortality"),
            (("b20w05r1", "b10w05r1"), ("b20w05r1", "b20w10r1"), ("b20w05r1", "b20w05top2")),
        ),
        (
            "grid_b",
            "recipe",
            ("noise", "leaderboard_immortality"),
            (
                ("r1", "b100"),
                ("r1", "w100"),
                ("r1", "b50r1"),
                ("r1", "b25w25r1"),
                ("r1", "b20w05lad4"),
                ("r1", "b20w05r1"),
                ("r1", "b30w05r1"),
                ("r1", "b20w20lad4s"),
                ("r1", "b20w05top2"),
            ),
        ),
        (
            "grid_b",
            "noise",
            ("recipe", "leaderboard_immortality"),
            (("n0", "n10"),),
        ),
        (
            "grid_b",
            "leaderboard_immortality",
            ("recipe", "noise"),
            (("imm0", "imm10"),),
        ),
    ]
    output: list[dict[str, Any]] = []
    for grid, axis, match_axes, pairs in specs:
        blocks: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in run_rows:
            if row.get("grid") != grid:
                continue
            blocks[tuple(row.get(match_axis) for match_axis in match_axes)][str(row.get(axis))] = row
        for base, variant in pairs:
            for metric_name, higher_is_better in (
                ("survival", True),
                ("training_reward", True),
                ("outcome_residual", True),
            ):
                for horizon in horizons:
                    key = f"{metric_name if metric_name != 'training_reward' else 'reward'}_at_{horizon}"
                    deltas: list[float] = []
                    missing = 0
                    for block in blocks.values():
                        base_row = block.get(base)
                        variant_row = block.get(variant)
                        if base_row is None or variant_row is None:
                            missing += 1
                            continue
                        base_value = base_row.get(key)
                        variant_value = variant_row.get(key)
                        if base_value is None or variant_value is None:
                            missing += 1
                            continue
                        deltas.append(float(variant_value) - float(base_value))
                    output.append(
                        {
                            "grid": grid,
                            "axis": axis,
                            "metric": metric_name,
                            "horizon": horizon,
                            "delta": f"{variant} - {base}",
                            "higher_is_better": higher_is_better,
                            "missing_blocks": missing,
                            **summarize_deltas(deltas, higher_is_better=higher_is_better),
                        }
                    )
            deltas = []
            missing = 0
            for block in blocks.values():
                base_row = block.get(base)
                variant_row = block.get(variant)
                if base_row is None or variant_row is None:
                    missing += 1
                    continue
                base_value = base_row.get("learned_best_rank")
                variant_value = variant_row.get("learned_best_rank")
                if base_value is None or variant_value is None:
                    missing += 1
                    continue
                deltas.append(float(variant_value) - float(base_value))
            output.append(
                {
                    "grid": grid,
                    "axis": axis,
                    "metric": "learned_best_rank",
                    "horizon": None,
                    "delta": f"{variant} - {base}",
                    "higher_is_better": False,
                    "missing_blocks": missing,
                    **summarize_deltas(deltas, higher_is_better=False),
                }
            )
    return output


def recipe_rows(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, Mapping[str, Any]] = {}
    for row in run_rows:
        recipe = row.get("recipe")
        if recipe is None:
            continue
        grouped[str(recipe)] = row
    output: list[dict[str, Any]] = []
    for recipe, row in sorted(grouped.items()):
        slot_counts = row.get("recipe_slot_counts")
        if not isinstance(slot_counts, Mapping):
            # The joined row summary keeps this under original field names.
            slot_counts = {}
        output.append({"recipe": recipe, **{str(k): v for k, v in sorted(slot_counts.items())}})
    return output


def top_tournament_rows(run_rows: Sequence[Mapping[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    ranked = [
        row
        for row in run_rows
        if row.get("learned_best_rank") is not None
    ]
    ranked.sort(key=lambda row: float(row["learned_best_rank"]))
    columns = (
        "run_id",
        "grid",
        "reward",
        "noise",
        "leaderboard_immortality",
        "recipe",
        "learned_best_rank",
        "best_tournament_iteration",
        "best_tournament_games",
        "best_tournament_battles",
        "best_survival",
        "best_survival_iteration",
        "latest_survival",
        "survival_retention",
    )
    return [{column: row.get(column) for column in columns} for row in ranked[:limit]]


def collapse_rows(run_rows: Sequence[Mapping[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    ranked = sorted(
        run_rows,
        key=lambda row: (
            bool(row.get("latest_action_collapsed")),
            int(row.get("collapsed_checkpoint_count") or 0),
            float(row.get("latest_top_action_fraction") or 0.0),
        ),
        reverse=True,
    )
    columns = (
        "run_id",
        "grid",
        "reward",
        "noise",
        "leaderboard_immortality",
        "recipe",
        "latest_action_collapsed",
        "collapsed_checkpoint_count",
        "row_action_collapsed_sum",
        "latest_top_action_fraction",
        "latest_row_action_collapsed_count",
        "latest_survival",
        "learned_best_rank",
    )
    return [{column: row.get(column) for column in columns} for row in ranked[:limit]]


def build_report(
    *,
    joined_path: Path,
    eval_status_path: Path,
    tournament_rating_path: Path,
    horizons: Sequence[int],
) -> dict[str, Any]:
    joined = joined_rows(joined_path)
    eval_by_run = eval_rows_by_run(eval_status_path)
    tournament_by_run = tournament_rows_by_run(tournament_rating_path)
    run_rows = [
        summarize_run(row, eval_by_run, tournament_by_run, horizons)
        for row in joined
    ]
    for row, source in zip(run_rows, joined):
        row["recipe_slot_counts"] = source.get("recipe_slot_counts")
        row["reward_outcome_alpha"] = source.get("reward_outcome_alpha")
        row["leaderboard_immortal_probability"] = source.get("leaderboard_immortal_probability")
    horizon_tables = []
    lifecycle_tables = []
    for grid in ("grid_a", "grid_b"):
        for axis in axis_for_grid(grid):
            lifecycle_tables.extend(lifecycle_axis_summary(run_rows, grid, axis))
            for horizon in horizons:
                horizon_tables.extend(summarize_axis_at_horizon(run_rows, grid, axis, horizon))
    return {
        "schema_id": "curvyzero_cz26_deep_report/v0",
        "sources": {
            "joined": joined_path.as_posix(),
            "eval_status": eval_status_path.as_posix(),
            "tournament_rating": tournament_rating_path.as_posix(),
        },
        "horizons": list(horizons),
        "coverage": coverage_rows(run_rows, horizons),
        "recipes": recipe_rows(run_rows),
        "run_rows": run_rows,
        "axis_lifecycle": lifecycle_tables,
        "axis_horizons": horizon_tables,
        "exact_horizon_contrasts": exact_horizon_contrasts(run_rows, horizons),
        "top_learned_tournament_runs": top_tournament_rows(run_rows),
        "action_collapse_rows": collapse_rows(run_rows),
    }


def rows_for(payload: Mapping[str, Any], key: str, **filters: Any) -> list[dict[str, Any]]:
    rows = payload.get(key)
    if not isinstance(rows, list):
        return []
    output = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if all(row.get(filter_key) == filter_value for filter_key, filter_value in filters.items()):
            output.append(row)
    return output


def render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# CZ26 Deep Analysis Tables",
        "",
        "This is a generated offline report from local snapshots. It does not pull",
        "or mutate Modal state.",
        "",
        "## Sources",
        "",
    ]
    for key, value in (payload.get("sources") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Plain Terms",
            "",
            "- `survival`: average eval game length. Higher is better.",
            "- `training_reward`: scalar reward reported by the eval/training code. Only compare it within the same reward definition.",
            "- `outcome_residual`: inferred terminal win/loss contribution: training reward minus survival component minus bonus component.",
            "- `learned_best_rank`: best tournament rank after ignoring iteration-0 seed checkpoints. Lower is better.",
            "- `horizon`: exact checkpoint iteration used for a table, such as 30k, 170k, or 300k.",
            "",
            "## Coverage",
            "",
            markdown_table(
                payload.get("coverage", []),
                (
                    "grid",
                    "rows",
                    "completed",
                    "running",
                    "min_latest_eval_iteration",
                    "max_latest_eval_iteration",
                    "mean_eval_points",
                    "rows_at_30000",
                    "rows_at_170000",
                    "rows_at_300000",
                ),
            ),
            "",
            "Read: 30k covers both grids. Grid A has a usable 170k all-row horizon and a nearly complete 300k sensitivity read. Grid B has a true all-row horizon of 30k, so its 300k table is useful but excludes low-coverage rows.",
            "",
            "## Recipes",
            "",
            markdown_table(
                payload.get("recipes", []),
                ("recipe", "blank", "rank1", "rank2", "rank3", "rank4", "wall"),
            ),
            "",
            "Read: blank and wall are hard-coded immortal opponents. Rank slots come from the tournament leaderboard, with optional leaderboard immortality controlled separately by imm0/imm10.",
            "",
        ]
    )

    for grid in ("grid_a", "grid_b"):
        lines.extend([f"## {grid.replace('_', ' ').title()} Lifecycle By Axis", ""])
        for axis in axis_for_grid(grid):
            table = rows_for(payload, "axis_lifecycle", grid=grid, axis=axis)
            lines.extend(
                [
                    f"### By {axis}",
                    "",
                    markdown_table(
                        table,
                        (
                            "value",
                            "rows",
                            "mean_first_survival",
                            "mean_best_survival",
                            "mean_latest_survival",
                            "mean_survival_retention",
                            "best_learned_rank",
                            "mean_learned_best_rank",
                            "learned_top100_runs",
                            "collapsed_latest_rows",
                            "collapsed_ever_rows",
                            "row_action_collapsed_sum",
                        ),
                    ),
                    "",
                    "Read: this table shows the full training shape: where runs started, how high they ever got, where they ended, and whether the tournament saw any learned top-100 checkpoint.",
                    "",
                ]
            )

    for grid, horizons in (("grid_a", (30_000, 170_000, 300_000)), ("grid_b", (30_000, 300_000))):
        lines.extend([f"## {grid.replace('_', ' ').title()} Exact-Horizon Tables", ""])
        for horizon in horizons:
            lines.extend([f"### Horizon {horizon}", ""])
            for axis in axis_for_grid(grid):
                table = rows_for(payload, "axis_horizons", grid=grid, axis=axis, horizon=horizon)
                lines.extend(
                    [
                        f"#### By {axis}",
                        "",
                        markdown_table(
                            table,
                            (
                                "value",
                                "expected_rows",
                                "rows_at_horizon",
                                "mean_survival",
                                "median_survival",
                                "mean_training_reward",
                                "mean_outcome_residual",
                                "collapsed_rows",
                                "best_learned_rank",
                                "mean_learned_best_rank",
                                "learned_top100_runs",
                            ),
                        ),
                        "",
                        "Read: this is an exact-checkpoint comparison. A row only counts if that exact eval checkpoint exists.",
                        "",
                    ]
                )

    lines.extend(["## Matched Pairwise Contrasts", ""])
    contrast_columns = (
        "grid",
        "axis",
        "metric",
        "horizon",
        "delta",
        "pairs",
        "mean_delta",
        "median_delta",
        "better_count",
        "worse_count",
        "missing_blocks",
    )
    for grid in ("grid_a", "grid_b"):
        for axis in axis_for_grid(grid):
            table = rows_for(payload, "exact_horizon_contrasts", grid=grid, axis=axis)
            lines.extend(
                [
                    f"### {grid.replace('_', ' ').title()} By {axis}",
                    "",
                    markdown_table(table, contrast_columns),
                    "",
                    "Read: each row compares two settings while holding the other listed settings fixed. Positive survival/reward deltas favor the variant. For learned tournament rank, negative deltas favor the variant because lower rank is better.",
                    "",
                ]
            )

    lines.extend(
        [
            "## Top Learned Tournament Rows",
            "",
            markdown_table(
                payload.get("top_learned_tournament_runs", []),
                (
                    "run_id",
                    "grid",
                    "reward",
                    "noise",
                    "leaderboard_immortality",
                    "recipe",
                    "learned_best_rank",
                    "best_tournament_iteration",
                    "best_tournament_games",
                    "best_tournament_battles",
                    "best_survival",
                    "best_survival_iteration",
                    "latest_survival",
                    "survival_retention",
                ),
            ),
            "",
            "Read: this is sorted by best learned tournament rank. Many best-ranked checkpoints have very few tournament battles, so fine ordering is not trustworthy yet.",
            "",
            "## Action Collapse Watchlist",
            "",
            markdown_table(
                payload.get("action_collapse_rows", []),
                (
                    "run_id",
                    "grid",
                    "reward",
                    "noise",
                    "leaderboard_immortality",
                    "recipe",
                    "latest_action_collapsed",
                    "collapsed_checkpoint_count",
                    "row_action_collapsed_sum",
                    "latest_top_action_fraction",
                    "latest_row_action_collapsed_count",
                    "latest_survival",
                    "learned_best_rank",
                ),
            ),
            "",
            "Read: action collapse means the policy mostly chose one action in eval. It can be a failure mode even when one survival or tournament number looks good.",
            "",
            "## Per-Run Rows",
            "",
            markdown_table(
                payload.get("run_rows", []),
                (
                    "run_id",
                    "grid",
                    "reward",
                    "noise",
                    "leaderboard_immortality",
                    "recipe",
                    "train_status",
                    "eval_points",
                    "latest_eval_iteration",
                    "first_survival",
                    "survival_at_30000",
                    "survival_at_170000",
                    "survival_at_300000",
                    "best_survival",
                    "best_survival_iteration",
                    "latest_survival",
                    "survival_retention",
                    "latest_reward",
                    "latest_outcome_residual",
                    "collapsed_checkpoint_count",
                    "row_action_collapsed_sum",
                    "learned_best_rank",
                    "best_tournament_iteration",
                    "best_tournament_games",
                ),
            ),
            "",
            "Read: this table is intentionally wide. Use it to inspect exceptions before trusting an aggregate.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_horizons(values: Sequence[str] | None) -> tuple[int, ...]:
    if not values:
        return DEFAULT_HORIZONS
    horizons: list[int] = []
    for value in values:
        horizons.extend(int(part.strip()) for part in value.split(",") if part.strip())
    return tuple(dict.fromkeys(horizons))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CZ26 deep analysis report.")
    parser.add_argument("--joined-analysis", type=Path, required=True)
    parser.add_argument("--eval-status", type=Path, required=True)
    parser.add_argument("--tournament-rating", type=Path, required=True)
    parser.add_argument("--horizon", action="append")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_report(
        joined_path=args.joined_analysis,
        eval_status_path=args.eval_status,
        tournament_rating_path=args.tournament_rating,
        horizons=parse_horizons(args.horizon),
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_report(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
