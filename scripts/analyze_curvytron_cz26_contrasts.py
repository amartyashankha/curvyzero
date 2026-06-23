#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def metric_value(row: Mapping[str, Any], metric: str, score: str) -> float | None:
    if metric == "learned_tournament_rank":
        value = (row.get("tournament") or {}).get("learned_best_rank")
    else:
        value = ((row.get("metrics") or {}).get(metric) or {}).get(score)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_deltas(deltas: Sequence[float], *, higher_is_better: bool) -> dict[str, Any]:
    if not deltas:
        return {
            "n_pairs": 0,
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
        "n_pairs": len(deltas),
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        "better_count": better,
        "worse_count": worse,
    }


def contrast_axis(
    rows: Sequence[Mapping[str, Any]],
    *,
    grid: str,
    axis: str,
    match_axes: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    metric: str,
    score: str,
    higher_is_better: bool,
) -> list[dict[str, Any]]:
    blocks: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("grid") != grid:
            continue
        axis_value = row.get(axis)
        if axis_value is None:
            continue
        key = tuple(row.get(match_axis) for match_axis in match_axes)
        blocks[key][str(axis_value)] = row

    output: list[dict[str, Any]] = []
    for base, variant in pairs:
        deltas: list[float] = []
        missing_blocks = 0
        for block in blocks.values():
            base_row = block.get(base)
            variant_row = block.get(variant)
            if base_row is None or variant_row is None:
                missing_blocks += 1
                continue
            base_value = metric_value(base_row, metric, score)
            variant_value = metric_value(variant_row, metric, score)
            if base_value is None or variant_value is None:
                missing_blocks += 1
                continue
            deltas.append(variant_value - base_value)
        output.append(
            {
                "grid": grid,
                "axis": axis,
                "metric": metric,
                "score": score,
                "base": base,
                "variant": variant,
                "delta": f"{variant} - {base}",
                "higher_is_better": higher_is_better,
                "missing_blocks": missing_blocks,
                **summarize_deltas(deltas, higher_is_better=higher_is_better),
            }
        )
    return output


def build_contrasts(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("joined analysis must contain a rows list")

    metric_specs = [
        ("mean_survival", "matched", True),
        ("mean_survival", "latest", True),
        ("mean_survival", "best", True),
        ("mean_survival", "auc", True),
        ("mean_training_reward", "matched", True),
        ("mean_training_reward", "latest", True),
        ("mean_training_reward", "best", True),
        ("mean_training_reward", "auc", True),
        ("learned_tournament_rank", "best", False),
    ]
    contrast_specs = [
        (
            "grid_a",
            "reward_tag",
            ("recipe_code", "noise_tag", "leaderboard_immortal_tag"),
            (
                ("out0", "out33"),
                ("out0", "out67"),
                ("out0", "out100"),
                ("out33", "out67"),
                ("out67", "out100"),
            ),
        ),
        (
            "grid_a",
            "noise_tag",
            ("recipe_code", "reward_tag", "leaderboard_immortal_tag"),
            (("n0", "n10"), ("n0", "n20"), ("n10", "n20")),
        ),
        (
            "grid_a",
            "leaderboard_immortal_tag",
            ("recipe_code", "reward_tag", "noise_tag"),
            (("imm0", "imm10"),),
        ),
        (
            "grid_a",
            "recipe_code",
            ("reward_tag", "noise_tag", "leaderboard_immortal_tag"),
            (
                ("b20w05r1", "b10w05r1"),
                ("b20w05r1", "b20w10r1"),
                ("b20w05r1", "b20w05top2"),
            ),
        ),
        (
            "grid_b",
            "noise_tag",
            ("recipe_code", "leaderboard_immortal_tag"),
            (("n0", "n10"),),
        ),
        (
            "grid_b",
            "leaderboard_immortal_tag",
            ("recipe_code", "noise_tag"),
            (("imm0", "imm10"),),
        ),
        (
            "grid_b",
            "recipe_code",
            ("noise_tag", "leaderboard_immortal_tag"),
            (
                ("r1", "b100"),
                ("r1", "w100"),
                ("r1", "b50r1"),
                ("r1", "b25w25r1"),
                ("r1", "b20w05lad4"),
                ("r1", "b20w05r1"),
                ("r1", "b30w05r1"),
            ),
        ),
    ]

    output: list[dict[str, Any]] = []
    for metric, score, higher_is_better in metric_specs:
        for grid, axis, match_axes, pairs in contrast_specs:
            output.extend(
                contrast_axis(
                    rows,
                    grid=grid,
                    axis=axis,
                    match_axes=match_axes,
                    pairs=pairs,
                    metric=metric,
                    score=score,
                    higher_is_better=higher_is_better,
                )
            )
    return output


def display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "grid",
        "axis",
        "metric",
        "score",
        "delta",
        "n_pairs",
        "mean_delta",
        "median_delta",
        "better_count",
        "worse_count",
        "missing_blocks",
    )
    lines = [
        "# CZ26 Matched Contrasts",
        "",
        "Positive deltas mean the variant has a larger metric value. For",
        "`learned_tournament_rank`, smaller is better, so `better_count` counts",
        "negative deltas.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(display(row.get(column)) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build matched contrasts from CZ26 joined analysis.")
    parser.add_argument("--joined-analysis", required=True)
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args()

    rows = build_contrasts(load_json(Path(args.joined_analysis)))
    output = {"schema_id": "curvyzero_cz26_matched_contrasts/v0", "contrasts": rows}
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(output, indent=2, sort_keys=True))
    if args.markdown_output:
        Path(args.markdown_output).write_text(render_markdown(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
