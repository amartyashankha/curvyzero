"""Summarize official LightZero Atari Pong parallel eval manifests.

This is a local readout helper for JSON artifacts produced by
``curvyzero.infra.modal.lightzero_pong_eval_smoke``. It does not run evals or
training; it only turns manifests into a compact table for checkpoint triage.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_COLUMNS = (
    "checkpoint",
    "eval_seed",
    "stock_steps_survived",
    "stock_survival_fraction",
    "eval_cap_steps",
    "stock_return",
    "stock_episode_length",
    "stock_positive_reward_count",
    "stock_negative_reward_count",
    "stock_nonzero_reward_count",
    "stock_done",
    "stock_truncated",
    "stock_terminal_reason",
    "steps_survived",
    "survival_fraction",
    "survived_to_cap",
    "return",
    "nonzero_reward_count",
    "positive_reward_count",
    "checkpoint_ref",
    "strict_load",
    "fallback_used",
    "stock_manual_match",
    "stock_action_histogram",
    "dominant_action",
    "dominant_action_share",
    "action_entropy",
    "verdict",
)

BASELINE_DELTA_COLUMNS = (
    "delta_stock_steps_survived",
    "delta_stock_survival_fraction",
    "delta_steps_survived",
    "delta_survival_fraction",
    "delta_stock_return",
    "delta_return",
    "delta_positive_rewards",
)

SURVIVAL_CURVE_COLUMNS = (
    "checkpoint",
    "eval_seed_count",
    "stock_steps_survived",
    "stock_steps_survived_mean",
    "stock_steps_survived_min",
    "stock_steps_survived_max",
    "stock_steps_survived_latest",
    "eval_cap_steps",
    "stock_survival_fraction",
    "run_best",
    "latest",
    "eval_seed_latest",
    "eval_seeds",
    "delta_stock_steps_survived",
    "delta_previous_stock_steps_survived",
    "best_so_far_stock_steps_survived",
    "delta_best_so_far_stock_steps_survived",
    "stock_return",
    "stock_return_mean",
    "delta_stock_return",
    "stock_positive_reward_count",
    "stock_positive_reward_count_mean",
    "stock_positive_reward_count_sum",
    "duplicate_eval_count",
    "duplicate_stock_steps_survived_values",
    "duplicate_stock_steps_disagree",
    "steps_survived",
    "delta_steps_survived",
    "verdict",
)

SURVIVAL_AGGREGATE_COLUMNS = (
    "checkpoint",
    "eval_seed_count",
    "stock_steps_mean",
    "stock_steps_min",
    "stock_steps_max",
    "delta_stock_steps_mean",
    "stock_return_mean",
    "stock_positive_reward_count_sum",
)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    rows: list[dict[str, Any]] = []
    for source in args.manifests:
        for manifest_path in _resolve_sources(source, volume_root=args.volume_root):
            manifest = _load_json(manifest_path)
            rows.extend(
                _manifest_rows(
                    manifest,
                    source_label=_source_label(source, manifest_path),
                )
            )
    _add_eval_seed(rows)
    rows = _dedupe_rows(rows)
    if args.survival_curve:
        rows = _sort_by_checkpoint_iteration(rows)
        rows = _collapse_duplicate_checkpoints(rows)
        _add_baseline_deltas(rows)
        _add_survival_curve_fields(rows)
        if args.survival_aggregate:
            rows = _survival_aggregate_rows(rows) + rows
    elif args.sort_checkpoints:
        rows = _sort_by_checkpoint_iteration(rows)
        if args.baseline_deltas:
            _add_baseline_deltas(rows)
    elif args.baseline_deltas:
        _add_baseline_deltas(rows)
    rendered = _render_rows(
        rows,
        fmt=args.format,
        include_source=args.include_source,
        baseline_deltas=args.baseline_deltas,
        survival_curve=args.survival_curve,
        survival_aggregate=args.survival_aggregate,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifests",
        nargs="+",
        help=(
            "Local manifest JSON path, ref:<volume/path>, volume:<volume/path>, "
            "a directory containing manifest_*.json files and/or per-episode "
            "JSON artifacts, a glob such as '<root>/**/*.json', or a bare "
            "relative Modal Volume ref."
        ),
    )
    parser.add_argument(
        "--volume-root",
        type=Path,
        default=Path(os.environ.get("CURVYZERO_RUNS_MOUNT", "/runs")),
        help="Local mount root used to resolve Modal Volume refs.",
    )
    parser.add_argument(
        "--format",
        choices=("md", "tsv", "json"),
        default="md",
        help="Output table format.",
    )
    parser.add_argument(
        "--include-source",
        action="store_true",
        help="Include the manifest source in each row.",
    )
    parser.add_argument(
        "--baseline-deltas",
        action="store_true",
        help=(
            "Include delta_* columns versus the loaded iteration_0 row, or the "
            "first loaded row if iteration_0 is absent."
        ),
    )
    parser.add_argument(
        "--sort-checkpoints",
        action="store_true",
        help="Sort rows by numeric iteration_<N> checkpoint before rendering.",
    )
    parser.add_argument(
        "--survival-curve",
        action="store_true",
        help=(
            "Print a compact survival-first curve sorted by checkpoint. This "
            "implies baseline deltas and adds previous/best/latest stock-step "
            "columns."
        ),
    )
    parser.add_argument(
        "--survival-aggregate",
        action="store_true",
        help=(
            "With --survival-curve, append a compact checkpoint-level aggregate "
            "over eval seeds."
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Also save the rendered table.")
    return parser.parse_args(argv)


def _resolve_sources(source: str, *, volume_root: Path) -> list[Path]:
    raw = source
    if raw.startswith("ref:"):
        raw = raw.removeprefix("ref:")
        path = volume_root / raw
    elif raw.startswith("volume:"):
        raw = raw.removeprefix("volume:")
        path = volume_root / raw
    else:
        path = Path(raw)
        if _has_glob_magic(raw):
            candidates = _glob_candidates(raw)
            if not candidates and not path.is_absolute():
                candidates = _glob_candidates(str(volume_root / raw))
            if not candidates:
                raise FileNotFoundError(f"no JSON files matched: {source}")
            return candidates
        if not path.exists() and not path.is_absolute():
            volume_path = volume_root / raw
            if volume_path.exists():
                path = volume_path
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {source} (resolved to {path})")
    if path.is_dir():
        candidates = _directory_candidates(path)
        if not candidates:
            raise FileNotFoundError(
                f"no manifest_*.json files or per-episode eval JSON files under {path}"
            )
        return candidates
    return [path]


def _has_glob_magic(raw: str) -> bool:
    return any(character in raw for character in "*?[")


def _glob_candidates(pattern: str) -> list[Path]:
    return _known_eval_json_files(
        Path(match)
        for match in glob.glob(pattern, recursive=True)
        if Path(match).is_file()
    )


def _directory_candidates(path: Path) -> list[Path]:
    return _known_eval_json_files(path.glob("**/*.json"))


def _known_eval_json_files(paths: Any) -> list[Path]:
    candidates = sorted(Path(path) for path in paths if Path(path).is_file())
    known = [
        path
        for path in candidates
        if path.name.startswith("manifest_")
        or path.name.startswith("lightzero_visual_pong_eval_")
    ]
    return known if known else candidates


def _source_label(source: str, manifest_path: Path) -> str:
    stripped_source = source.removeprefix("ref:").removeprefix("volume:")
    source_path = Path(stripped_source)
    if _has_glob_magic(source):
        return str(manifest_path)
    if source_path.name == manifest_path.name:
        return source
    if manifest_path.parent != source_path.parent:
        return str(manifest_path)
    return f"{source.rstrip('/')}/{manifest_path.name}"


def _is_raw_eval_artifact(source_label: str) -> bool:
    return Path(source_label).name.startswith("lightzero_visual_pong_eval_")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return loaded


def _manifest_rows(
    manifest: dict[str, Any],
    *,
    source_label: str,
) -> list[dict[str, Any]]:
    table = manifest.get("table")
    results = manifest.get("results")
    if isinstance(table, list):
        table_rows = [row for row in table if isinstance(row, dict)]
    else:
        table_rows = []
    if isinstance(results, list):
        result_rows = [row for row in results if isinstance(row, dict)]
    elif manifest.get("schema") == "curvyzero_lightzero_visual_pong_eval_smoke/v0":
        result_rows = [manifest]
    else:
        result_rows = []

    if not table_rows and not result_rows:
        raise ValueError("manifest has neither a table nor result rows")

    count = max(len(table_rows), len(result_rows))
    rows = []
    for index in range(count):
        table_row = table_rows[index] if index < len(table_rows) else {}
        result = result_rows[index] if index < len(result_rows) else {}
        row = _summary_row(table_row, result, index=index)
        row["source"] = source_label
        if row.get("artifact_ref") is None and _is_raw_eval_artifact(source_label):
            row["artifact_ref"] = source_label
        rows.append(row)
    return rows


def _summary_row(table_row: dict[str, Any], result: dict[str, Any], *, index: int) -> dict[str, Any]:
    episode = _dict_or_empty(result.get("episode"))
    status = _dict_or_empty(result.get("status"))
    manual_vs_stock = _dict_or_empty(result.get("manual_vs_stock"))
    config = _dict_or_empty(result.get("config"))
    stock_evaluator = _dict_or_empty(result.get("stock_evaluator"))
    stock_rollout = _dict_or_empty(stock_evaluator.get("stock_rollout"))

    checkpoint_ref = _first_present(
        table_row.get("checkpoint_ref"),
        config.get("checkpoint_ref"),
        _nested(result, "checkpoint", "path"),
        _nested(result, "artifact", "ref"),
    )
    checkpoint = _first_present(
        table_row.get("checkpoint"),
        _checkpoint_label(checkpoint_ref),
        f"checkpoint_{index:03d}",
    )
    strict_load = _first_present(
        table_row.get("strict_load"),
        table_row.get("strict_policy_model_load_ok"),
        status.get("strict_policy_model_load_ok"),
        _nested(result, "surface", "load_state_dict", "strict"),
    )
    fallback_used = _first_present(
        table_row.get("fallback_used"),
        status.get("model_fallback_used"),
        _fallback_from_episode(episode),
    )
    stock_manual_match = _first_present(
        table_row.get("stock_manual_match"),
        manual_vs_stock.get("actions_match_for_recorded_prefix"),
    )
    histogram = _first_present(table_row.get("action_histogram"), episode.get("action_histogram"))
    dominant_action = table_row.get("dominant_action")
    dominant_share = table_row.get("dominant_action_share")
    action_entropy = table_row.get("action_entropy")
    if dominant_action is None or dominant_share is None or action_entropy is None:
        computed_action, computed_share, computed_entropy = _histogram_stats(histogram)
        dominant_action = _first_present(dominant_action, computed_action)
        dominant_share = _first_present(dominant_share, computed_share)
        action_entropy = _first_present(action_entropy, computed_entropy)

    nonzero_count, positive_count = _reward_counts(table_row, episode)
    total_return = _first_present(table_row.get("return"), episode.get("total_reward"))
    stock_return = _first_present(
        table_row.get("stock_return"),
        _stock_evaluator_return(stock_evaluator),
    )
    eval_cap_steps = _first_present(
        table_row.get("eval_cap_steps"),
        episode.get("max_eval_steps"),
        config.get("max_eval_steps"),
    )
    stock_steps_survived = _first_present(
        table_row.get("stock_steps_survived"),
        stock_rollout.get("steps_run"),
    )
    stock_episode_length = _first_present(
        table_row.get("stock_episode_length"),
        stock_rollout.get("episode_length"),
        stock_steps_survived,
    )
    stock_survival_fraction = _first_present(
        table_row.get("stock_survival_fraction"),
        _survival_fraction(
            _first_present(stock_steps_survived, stock_episode_length),
            eval_cap_steps,
        ),
    )
    stock_nonzero_reward_count = _first_present(
        table_row.get("stock_nonzero_reward_count"),
        stock_rollout.get("nonzero_reward_count"),
    )
    stock_positive_reward_count = _first_present(
        table_row.get("stock_positive_reward_count"),
        stock_rollout.get("positive_reward_count"),
    )
    stock_negative_reward_count = _first_present(
        table_row.get("stock_negative_reward_count"),
        stock_rollout.get("negative_reward_count"),
    )
    stock_done = _first_present(table_row.get("stock_done"), stock_rollout.get("done"))
    stock_truncated = _first_present(
        table_row.get("stock_truncated"),
        stock_rollout.get("truncated"),
    )
    stock_terminal_reason = _first_present(
        table_row.get("stock_terminal_reason"),
        stock_rollout.get("terminal_reason"),
    )
    stock_action_histogram = _first_present(
        table_row.get("stock_action_histogram"),
        stock_rollout.get("action_histogram"),
    )
    steps_survived = _first_present(
        table_row.get("steps_survived"),
        status.get("steps_run"),
        episode.get("steps_run"),
    )
    survival_fraction = _first_present(
        table_row.get("survival_fraction"),
        table_row.get("survival_rate"),
        _survival_fraction(steps_survived, eval_cap_steps),
    )
    survived_to_cap = _first_present(
        table_row.get("survived_to_cap"),
        _survived_to_cap(steps_survived, eval_cap_steps),
    )
    verdict = _first_present(
        table_row.get("verdict"),
        _verdict(
            ok=_first_present(table_row.get("ok"), result.get("ok")),
            fallback_used=fallback_used,
            stock_manual_match=stock_manual_match,
            dominant_action_share=dominant_share,
            positive_reward_count=positive_count,
            total_return=total_return,
        ),
    )

    return {
        "checkpoint": checkpoint,
        "artifact_ref": _first_present(table_row.get("artifact_ref"), _nested(result, "artifact", "ref")),
        "checkpoint_ref": checkpoint_ref,
        "eval_seed": _first_present(table_row.get("eval_seed"), table_row.get("seed"), config.get("seed")),
        "strict_load": strict_load,
        "fallback_used": fallback_used,
        "stock_manual_match": stock_manual_match,
        "eval_cap_steps": eval_cap_steps,
        "steps_survived": steps_survived,
        "survival_fraction": survival_fraction,
        "survival_rate": survival_fraction,
        "survived_to_cap": survived_to_cap,
        "return": total_return,
        "stock_return": stock_return,
        "stock_steps_survived": stock_steps_survived,
        "stock_survival_fraction": stock_survival_fraction,
        "stock_episode_length": stock_episode_length,
        "stock_nonzero_reward_count": stock_nonzero_reward_count,
        "stock_positive_reward_count": stock_positive_reward_count,
        "stock_negative_reward_count": stock_negative_reward_count,
        "stock_done": stock_done,
        "stock_truncated": stock_truncated,
        "stock_terminal_reason": stock_terminal_reason,
        "stock_action_histogram": stock_action_histogram,
        "nonzero_reward_count": nonzero_count,
        "positive_reward_count": positive_count,
        "dominant_action": dominant_action,
        "dominant_action_share": dominant_share,
        "action_entropy": action_entropy,
        "verdict": verdict,
    }


def _add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    baseline = next((row for row in rows if row.get("checkpoint") == "iteration_0"), rows[0])
    comparisons = (
        ("delta_stock_steps_survived", "stock_steps_survived"),
        ("delta_stock_survival_fraction", "stock_survival_fraction"),
        ("delta_steps_survived", "steps_survived"),
        ("delta_survival_fraction", "survival_fraction"),
        ("delta_stock_return", "stock_return"),
        ("delta_return", "return"),
        ("delta_positive_rewards", "positive_reward_count"),
    )
    for row in rows:
        for delta_column, value_column in comparisons:
            row[delta_column] = _numeric_delta(row.get(value_column), baseline.get(value_column))


def _add_grouped_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    for _key, group in _group_rows_by_eval_seed(rows).items():
        _add_baseline_deltas(group)


def _sort_by_checkpoint_iteration(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=_checkpoint_sort_key)


def _collapse_duplicate_checkpoints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int | None, str], list[dict[str, Any]]] = {}
    order: list[tuple[int | None, str]] = []
    for row in rows:
        key = (
            _checkpoint_iteration(row),
            str(row.get("checkpoint") or ""),
        )
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)

    collapsed: list[dict[str, Any]] = []
    for key in order:
        group = sorted(grouped[key], key=_row_freshness_key)
        latest_rows = _latest_rows_by_eval_seed(group)
        chosen = dict(sorted(latest_rows, key=_row_freshness_key)[-1])
        stock_values = [
            value
            for value in (_format_optional_number(row.get("stock_steps_survived")) for row in group)
            if value is not None
        ]
        unique_stock_values = sorted(set(stock_values), key=lambda value: float(value))
        latest_stock_numbers = [
            value
            for value in (_numeric_value(row.get("stock_steps_survived")) for row in latest_rows)
            if value is not None
        ]
        latest_stock_returns = [
            value
            for value in (_numeric_value(row.get("stock_return")) for row in latest_rows)
            if value is not None
        ]
        latest_stock_positive_rewards = [
            value
            for value in (
                _numeric_value(row.get("stock_positive_reward_count"))
                for row in latest_rows
            )
            if value is not None
        ]
        eval_seeds = sorted(
            {
                str(seed)
                for seed in (row.get("eval_seed") for row in latest_rows)
                if seed is not None
            },
            key=lambda value: int(value) if value.lstrip("-").isdigit() else value,
        )
        stock_mean = (
            round(sum(latest_stock_numbers) / len(latest_stock_numbers), 6)
            if latest_stock_numbers
            else None
        )
        chosen["duplicate_eval_count"] = len(group)
        chosen["duplicate_stock_steps_survived_values"] = (
            ",".join(unique_stock_values) if len(group) > 1 else ""
        )
        chosen["duplicate_stock_steps_disagree"] = len(unique_stock_values) > 1
        chosen["eval_seed_count"] = len(eval_seeds)
        chosen["eval_seeds"] = ",".join(eval_seeds)
        chosen["eval_seed_latest"] = chosen.get("eval_seed")
        chosen["stock_steps_survived_latest"] = chosen.get("stock_steps_survived")
        chosen["stock_steps_survived_min"] = (
            _format_numeric_delta(min(latest_stock_numbers)) if latest_stock_numbers else None
        )
        chosen["stock_steps_survived_mean"] = stock_mean
        chosen["stock_steps_survived_max"] = (
            _format_numeric_delta(max(latest_stock_numbers)) if latest_stock_numbers else None
        )
        if stock_mean is not None:
            chosen["stock_steps_survived"] = _format_numeric_delta(stock_mean)
        stock_return_mean = _mean(latest_stock_returns)
        if stock_return_mean is not None:
            chosen["stock_return_mean"] = stock_return_mean
            chosen["stock_return"] = stock_return_mean
        stock_positive_reward_mean = _mean(latest_stock_positive_rewards)
        if stock_positive_reward_mean is not None:
            chosen["stock_positive_reward_count_mean"] = stock_positive_reward_mean
            chosen["stock_positive_reward_count"] = stock_positive_reward_mean
            chosen["stock_positive_reward_count_sum"] = _format_numeric_delta(
                sum(latest_stock_positive_rewards)
            )
        collapsed.append(chosen)
    return _sort_by_checkpoint_iteration(collapsed)


def _checkpoint_sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    iteration = _checkpoint_iteration(row)
    eval_seed = row.get("eval_seed")
    seed_sort = int(eval_seed) if isinstance(eval_seed, int) else -1
    if iteration is None:
        return (1, 0, seed_sort, str(row.get("checkpoint") or row.get("checkpoint_ref") or ""))
    return (0, iteration, seed_sort, str(row.get("checkpoint") or ""))


def _checkpoint_iteration(row: dict[str, Any]) -> int | None:
    for value in (row.get("checkpoint"), row.get("checkpoint_ref")):
        if value is None:
            continue
        match = re.search(r"\biteration_(\d+)\b", str(value))
        if match is not None:
            return int(match.group(1))
    return None


def _add_eval_seed(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["eval_seed"] = _first_present(row.get("eval_seed"), _eval_seed(row))


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str] | tuple[str, int], dict[str, Any]] = {}
    order: list[tuple[str, str] | tuple[str, int]] = []
    for index, row in enumerate(rows):
        key = _row_identity(row, index=index)
        if key not in deduped:
            order.append(key)
            deduped[key] = row
            continue
        deduped[key] = _preferred_duplicate_row(deduped[key], row)
    return [deduped[key] for key in order]


def _row_identity(
    row: dict[str, Any],
    *,
    index: int,
) -> tuple[str, str] | tuple[str, int]:
    artifact_ref = row.get("artifact_ref")
    if artifact_ref:
        artifact_suffix = _artifact_path_suffix(artifact_ref)
        if artifact_suffix is not None:
            return ("artifact_path", artifact_suffix)
        return ("artifact_ref", str(artifact_ref))
    source = row.get("source")
    if source:
        return ("source", str(source))
    return ("row", index)


def _artifact_path_suffix(value: Any) -> str | None:
    path = Path(str(value))
    if not path.name.startswith("lightzero_visual_pong_eval_"):
        return None
    return f"{path.parent.name}/{path.name}"


def _preferred_duplicate_row(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    left_source = str(left.get("source") or "")
    right_source = str(right.get("source") or "")
    left_is_raw = "lightzero_visual_pong_eval_" in Path(left_source).name
    right_is_raw = "lightzero_visual_pong_eval_" in Path(right_source).name
    if right_is_raw and not left_is_raw:
        return right
    if left_is_raw and not right_is_raw:
        return left
    return sorted((left, right), key=_row_freshness_key)[-1]


def _eval_seed(row: dict[str, Any]) -> int | None:
    for key in ("artifact_ref", "source", "checkpoint_ref"):
        value = row.get(key)
        if value is None:
            continue
        match = re.search(r"(?:^|[_/-])seed(-?\d+)(?:[_./-]|$)", str(value))
        if match is not None:
            return int(match.group(1))
    return None


def _latest_rows_by_eval_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        seed = row.get("eval_seed")
        key = str(seed) if seed is not None else f"unknown-{index}"
        grouped.setdefault(key, []).append(row)
    return [sorted(group, key=_row_freshness_key)[-1] for group in grouped.values()]


def _row_freshness_key(row: dict[str, Any]) -> tuple[int, str]:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("artifact_ref", "source", "checkpoint_ref")
    )
    matches = re.findall(r"20\d{6}T\d{6}Z", text)
    if not matches:
        return (0, text)
    return (1, max(matches))


def _add_survival_curve_fields(rows: list[dict[str, Any]]) -> None:
    best_steps: float | None = None
    latest_iteration = max(
        (iteration for row in rows if (iteration := _checkpoint_iteration(row)) is not None),
        default=None,
    )
    run_best_steps = max(
        (
            steps
            for row in rows
            if (steps := _numeric_value(row.get("stock_steps_survived"))) is not None
        ),
        default=None,
    )
    previous_steps: float | None = None
    for row in rows:
        steps = _numeric_value(row.get("stock_steps_survived"))
        if steps is None:
            row["delta_previous_stock_steps_survived"] = None
            row["best_so_far_stock_steps_survived"] = best_steps
            row["delta_best_so_far_stock_steps_survived"] = None
        else:
            row["delta_previous_stock_steps_survived"] = _format_numeric_delta(
                None if previous_steps is None else steps - previous_steps
            )
            if best_steps is None or steps > best_steps:
                best_steps = steps
            row["best_so_far_stock_steps_survived"] = _format_numeric_delta(best_steps)
            row["delta_best_so_far_stock_steps_survived"] = _format_numeric_delta(
                steps - best_steps
            )
            previous_steps = steps
        iteration = _checkpoint_iteration(row)
        row["latest"] = latest_iteration is not None and iteration == latest_iteration
        row["run_best"] = (
            run_best_steps is not None
            and steps is not None
            and steps == run_best_steps
        )


def _add_grouped_survival_curve_fields(rows: list[dict[str, Any]]) -> None:
    for _key, group in _group_rows_by_eval_seed(rows).items():
        _add_survival_curve_fields(group)


def _group_rows_by_eval_seed(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("eval_seed") if row.get("eval_seed") is not None else "")
        grouped.setdefault(key, []).append(row)
    return grouped


def _survival_aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        iteration = _checkpoint_iteration(row)
        if iteration is None:
            continue
        grouped.setdefault(iteration, []).append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for iteration in sorted(grouped):
        group = grouped[iteration]
        stock_steps = [
            value
            for row in group
            if (value := _numeric_value(row.get("stock_steps_survived"))) is not None
        ]
        stock_returns = [
            value
            for row in group
            if (value := _numeric_value(row.get("stock_return"))) is not None
        ]
        positive_rewards = [
            value
            for row in group
            if (value := _numeric_value(row.get("stock_positive_reward_count"))) is not None
        ]
        deltas = [
            value
            for row in group
            if (value := _numeric_value(row.get("delta_stock_steps_survived"))) is not None
        ]
        aggregate_rows.append(
            {
                "_aggregate": True,
                "checkpoint": f"iteration_{iteration}",
                "eval_seed_count": len({row.get("eval_seed") for row in group}),
                "stock_steps_mean": _mean(stock_steps),
                "stock_steps_min": _format_numeric_delta(min(stock_steps)) if stock_steps else None,
                "stock_steps_max": _format_numeric_delta(max(stock_steps)) if stock_steps else None,
                "delta_stock_steps_mean": _mean(deltas),
                "stock_return_mean": _mean(stock_returns),
                "stock_positive_reward_count_sum": (
                    _format_numeric_delta(sum(positive_rewards)) if positive_rewards else None
                ),
            }
        )
    return aggregate_rows


def _mean(values: list[float]) -> int | float | None:
    if not values:
        return None
    return _format_numeric_delta(sum(values) / len(values))


def _numeric_delta(value: Any, baseline: Any) -> int | float | None:
    try:
        numeric_value = float(value)
        numeric_baseline = float(baseline)
    except (TypeError, ValueError):
        return None
    return _format_numeric_delta(numeric_value - numeric_baseline)


def _numeric_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_numeric_delta(delta: float | None) -> int | float | None:
    if delta is None:
        return None
    return int(delta) if delta.is_integer() else round(delta, 6)


def _format_optional_number(value: Any) -> str | None:
    numeric = _numeric_value(value)
    if numeric is None:
        return None
    formatted = _format_numeric_delta(numeric)
    return str(formatted)


def _survival_fraction(steps_survived: Any, eval_cap_steps: Any) -> float | None:
    try:
        steps = float(steps_survived)
        cap = float(eval_cap_steps)
    except (TypeError, ValueError):
        return None
    if cap <= 0.0:
        return None
    return round(steps / cap, 6)


def _survival_rate(steps_survived: Any, eval_cap_steps: Any) -> float | None:
    return _survival_fraction(steps_survived, eval_cap_steps)


def _survived_to_cap(steps_survived: Any, eval_cap_steps: Any) -> bool | None:
    try:
        steps = float(steps_survived)
        cap = float(eval_cap_steps)
    except (TypeError, ValueError):
        return None
    if cap <= 0.0:
        return None
    return steps >= cap


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _checkpoint_label(checkpoint_ref: Any) -> str | None:
    if not checkpoint_ref:
        return None
    name = Path(str(checkpoint_ref)).name
    for suffix in (".pth.tar", ".tar", ".pth", ".pt", ".bin"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name or None


def _fallback_from_episode(episode: dict[str, Any]) -> bool | None:
    fallback_count = episode.get("fallback_step_count")
    if fallback_count is None:
        return None
    try:
        return int(fallback_count) > 0
    except (TypeError, ValueError):
        return None


def _reward_counts(table_row: dict[str, Any], episode: dict[str, Any]) -> tuple[Any, Any]:
    nonzero_count = table_row.get("nonzero_reward_count")
    positive_count = table_row.get("positive_reward_count")
    if nonzero_count is not None and positive_count is not None:
        return nonzero_count, positive_count
    steps = episode.get("nonzero_reward_steps")
    if not isinstance(steps, list):
        return nonzero_count, positive_count
    rewards = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        try:
            rewards.append(float(step.get("reward", 0.0)))
        except (TypeError, ValueError):
            continue
    return (
        nonzero_count if nonzero_count is not None else len(rewards),
        positive_count if positive_count is not None else sum(1 for reward in rewards if reward > 0.0),
    )


def _stock_evaluator_return(stock_evaluator: dict[str, Any]) -> float | None:
    eval_output = stock_evaluator.get("eval_output")
    if (
        isinstance(eval_output, (list, tuple))
        and len(eval_output) >= 2
        and isinstance(eval_output[1], dict)
    ):
        raw = eval_output[1].get("eval_episode_return_mean")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    return None


def _histogram_stats(histogram: Any) -> tuple[str | None, float | None, float | None]:
    if not isinstance(histogram, dict) or not histogram:
        return None, None, None
    counts: list[tuple[str, int]] = []
    for key, value in histogram.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts.append((str(key), count))
    total = sum(count for _key, count in counts)
    if total <= 0:
        return None, None, None
    dominant_action, dominant_count = max(counts, key=lambda item: item[1])
    if len(counts) <= 1:
        entropy = 0.0
    else:
        import math

        raw_entropy = -sum((count / total) * math.log(count / total) for _key, count in counts)
        entropy = raw_entropy / math.log(len(counts))
    return dominant_action, round(dominant_count / total, 6), round(entropy, 6)


def _verdict(
    *,
    ok: Any,
    fallback_used: Any,
    stock_manual_match: Any,
    dominant_action_share: Any,
    positive_reward_count: Any,
    total_return: Any,
) -> str:
    if ok is False:
        return "eval_failed"
    if fallback_used is True:
        return "invalid_fallback_used"
    if stock_manual_match is False:
        return "manual_stock_mismatch"
    try:
        if dominant_action_share is not None and float(dominant_action_share) >= 0.98:
            return "collapsed_action"
    except (TypeError, ValueError):
        pass
    try:
        if positive_reward_count is not None and int(positive_reward_count) > 0:
            return "has_positive_reward"
    except (TypeError, ValueError):
        pass
    try:
        if total_return is not None and float(total_return) < 0.0:
            return "negative_return"
    except (TypeError, ValueError):
        pass
    return "no_clear_signal"


def _render_rows(
    rows: list[dict[str, Any]],
    *,
    fmt: str,
    include_source: bool,
    baseline_deltas: bool,
    survival_curve: bool,
    survival_aggregate: bool,
) -> str:
    if survival_curve:
        columns = (("source",) if include_source else ()) + SURVIVAL_CURVE_COLUMNS
        if survival_aggregate and fmt == "tsv" and any(row.get("_aggregate") for row in rows):
            return _render_survival_curve_with_aggregate(rows, columns)
    else:
        columns = (
            (("source",) if include_source else ())
            + DEFAULT_COLUMNS
            + (BASELINE_DELTA_COLUMNS if baseline_deltas else ())
        )
    projected = [{column: row.get(column) for column in columns} for row in rows]
    if fmt == "json":
        return json.dumps(projected, indent=2, sort_keys=True)
    if fmt == "tsv":
        return _render_tsv(projected, columns)
    return _render_markdown(projected, columns)


def _render_survival_curve_with_aggregate(
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
) -> str:
    detail_rows = [row for row in rows if not row.get("_aggregate")]
    aggregate_rows = [row for row in rows if row.get("_aggregate")]
    if not aggregate_rows:
        projected = [{column: row.get(column) for column in columns} for row in detail_rows]
        return _render_tsv(projected, columns)
    detail = _render_tsv(
        [{column: row.get(column) for column in columns} for row in detail_rows],
        columns,
    )
    aggregate = _render_tsv(
        [
            {column: row.get(column) for column in SURVIVAL_AGGREGATE_COLUMNS}
            for row in aggregate_rows
        ],
        SURVIVAL_AGGREGATE_COLUMNS,
    )
    return "# aggregate_by_checkpoint\n" + aggregate + "\n\n# checkpoint_curve\n" + detail


def _render_tsv(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(_format_value(row.get(column), markdown=False) for column in columns))
    return "\n".join(lines)


def _render_markdown(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    headers = [_header(column) for column in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _column in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format_value(row.get(column), markdown=True) for column in columns)
            + " |"
        )
    return "\n".join(lines)


def _header(column: str) -> str:
    return {
        "checkpoint_ref": "Checkpoint Ref",
        "eval_seed": "Eval Seed",
        "strict_load": "Strict Load",
        "fallback_used": "Fallback",
        "stock_manual_match": "Manual/Stock Match",
        "eval_cap_steps": "Eval Cap",
        "steps_survived": "Steps",
        "survival_fraction": "Fraction",
        "survival_rate": "Fraction (Old Rate)",
        "survived_to_cap": "Survived To Cap",
        "stock_return": "Stock Return",
        "stock_steps_survived": "Stock Steps",
        "stock_steps_survived_min": "Stock Steps Min",
        "stock_steps_survived_mean": "Stock Steps Mean",
        "stock_steps_survived_max": "Stock Steps Max",
        "stock_steps_survived_latest": "Stock Steps Latest",
        "stock_survival_fraction": "Stock Fraction",
        "stock_episode_length": "Stock Episode Length",
        "stock_nonzero_reward_count": "Stock Nonzero Rewards",
        "stock_positive_reward_count": "Stock Positive Rewards",
        "stock_negative_reward_count": "Stock Negative Rewards",
        "stock_done": "Stock Done",
        "stock_truncated": "Stock Truncated",
        "stock_terminal_reason": "Stock Terminal",
        "stock_action_histogram": "Stock Action Histogram",
        "nonzero_reward_count": "Nonzero Rewards",
        "positive_reward_count": "Positive Rewards",
        "dominant_action": "Dominant Action",
        "dominant_action_share": "Dominant Share",
        "action_entropy": "Entropy",
        "delta_stock_steps_survived": "Delta Stock Steps",
        "delta_stock_survival_fraction": "Delta Stock Fraction",
        "delta_steps_survived": "Delta Steps",
        "delta_survival_fraction": "Delta Fraction",
        "delta_survival_rate": "Delta Fraction (Old Rate)",
        "delta_stock_return": "Delta Stock Return",
        "delta_return": "Delta Return",
        "delta_positive_rewards": "Delta Positive Rewards",
        "delta_previous_stock_steps_survived": "Delta Previous Stock Steps",
        "best_so_far_stock_steps_survived": "Best So Far Stock Steps",
        "delta_best_so_far_stock_steps_survived": "Delta Best So Far Stock Steps",
        "run_best": "Run Best",
        "latest": "Latest",
        "eval_seed_count": "Eval Seed Count",
        "eval_seeds": "Eval Seeds",
        "eval_seed_latest": "Latest Eval Seed",
    }.get(column, column.replace("_", " ").title())


def _format_value(value: Any, *, markdown: bool) -> str:
    if value is None:
        text = ""
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, float):
        text = str(int(value)) if value.is_integer() else f"{value:.6g}"
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    else:
        text = str(value)
    if markdown:
        text = text.replace("|", "\\|")
    return text


if __name__ == "__main__":
    main()
