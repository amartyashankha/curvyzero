"""Local CurvyTron eval-curve loading and scoring helpers.

This module is intentionally Modal-free. It consumes local JSON/JSONL snapshots:

- canonical curve objects with ``points``;
- eval-summary style rows from the CurvyTron run-status tool;
- optional matrix manifests for run axes.

The scores are triage filters, not final evidence of learning.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_METRICS = ("mean_survival",)
DEFAULT_METRIC = DEFAULT_METRICS[0]
DEFAULT_COLLAPSE_THRESHOLD = 0.95
DEFAULT_SIGNAL_DELTA = 1.0


@dataclass(frozen=True)
class MetricSpec:
    """Named scalar that may appear in CurvyTron eval curves."""

    name: str
    family: str
    description: str


METRIC_SCHEMA: tuple[MetricSpec, ...] = (
    MetricSpec("win_rate", "outcome", "Fraction of eval episodes won by the ego policy."),
    MetricSpec("loss_rate", "outcome", "Fraction of eval episodes lost by the ego policy."),
    MetricSpec("draw_rate", "outcome", "Fraction of eval episodes ending in a draw."),
    MetricSpec("cap_rate", "outcome", "Fraction of eval episodes ending at the eval cap."),
    MetricSpec("mean_survival", "survival", "Mean eval survival steps."),
    MetricSpec("median_survival", "survival", "Median eval survival steps."),
    MetricSpec("min_survival", "survival", "Minimum eval survival steps."),
    MetricSpec("max_survival", "survival", "Maximum eval survival steps."),
    MetricSpec("mean_reward", "reward", "Mean eval/trainer reward when exported."),
    MetricSpec("mean_training_reward", "reward", "Mean trainer reward reported by eval rows."),
    MetricSpec("bonus_count", "bonus", "Mean or total bonus pickups/counts when exported."),
    MetricSpec("bonus_pickup_count", "bonus", "Mean or total bonus pickups when exported."),
    MetricSpec("bonus_reward", "bonus", "Mean or total bonus reward when exported."),
    MetricSpec("top_action_fraction", "action", "Share of actions assigned to the most common action."),
    MetricSpec("action_entropy", "action", "Normalized action entropy when exported or derivable."),
    MetricSpec("left_rate", "action", "Fraction of eval actions mapped to action 0."),
    MetricSpec("straight_rate", "action", "Fraction of eval actions mapped to action 1."),
    MetricSpec("right_rate", "action", "Fraction of eval actions mapped to action 2."),
    MetricSpec("wall_rate", "terminal", "Fraction of terminal causes reported as wall."),
    MetricSpec("own_trail_rate", "terminal", "Fraction of terminal causes reported as own trail."),
    MetricSpec("opponent_trail_rate", "terminal", "Fraction of terminal causes reported as opponent trail."),
    MetricSpec("timeout_rate", "terminal", "Fraction of terminal causes reported as timeout/cap."),
    MetricSpec("failure_rate", "health", "Fraction of eval rows that failed when counts are present."),
)
METRIC_SCHEMA_BY_NAME = {spec.name: spec for spec in METRIC_SCHEMA}


@dataclass(frozen=True)
class CurvePoint:
    """One checkpoint on an eval curve."""

    iteration: int
    metrics: dict[str, float | int | bool | str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalCurve:
    """A per-run checkpoint curve with optional matrix axes."""

    run_id: str
    attempt_id: str | None = None
    axes: dict[str, Any] = field(default_factory=dict)
    points: tuple[CurvePoint, ...] = ()

    def sorted_points(self) -> tuple[CurvePoint, ...]:
        return tuple(sorted(self.points, key=lambda point: point.iteration))


def load_json_or_jsonl(path: Path) -> Any:
    """Load a JSON value or a JSONL list from ``path``."""

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def load_manifest_axes(path: Path | None) -> dict[str, dict[str, Any]]:
    """Return ``run_id -> axes`` from a local matrix manifest, if provided."""

    if path is None:
        return {}
    payload = load_json_or_jsonl(path)
    rows = _payload_rows(payload)
    axes_by_run: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        run_id = _run_id(row)
        if not run_id:
            continue
        axes = dict(row.get("axes") or {})
        label = row.get("label")
        if label is not None:
            axes.setdefault("label", label)
        command = row.get("command") or []
        if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
            axes.update(_command_axes(command))
        for key in (
            "row",
            "row_id",
            "reward_variant",
            "env_variant",
            "opponent_policy_kind",
            "source_state_trail_render_mode",
            "batch_size",
            "num_simulations",
            "collector_env_num",
        ):
            if row.get(key) is not None:
                axes.setdefault(key, row[key])
        axes_by_run[run_id] = axes
    return axes_by_run


def build_curves(
    payload: Any,
    *,
    manifest_axes: Mapping[str, Mapping[str, Any]] | None = None,
    metric_name: str = DEFAULT_METRIC,
) -> list[EvalCurve]:
    """Build curves from canonical curve JSON or eval-summary-like rows."""

    axes = manifest_axes or {}
    if isinstance(payload, Mapping) and "curves" in payload:
        rows = payload.get("curves") or []
        return [_curve_from_canonical(row, axes) for row in rows if isinstance(row, Mapping)]
    rows = _payload_rows(payload)
    curves: list[EvalCurve] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if "points" in row:
            curves.append(_curve_from_canonical(row, axes))
        elif "eval_checkpoints" in row:
            curve = _curve_from_eval_checkpoints_row(row, axes)
            if curve is not None:
                curves.append(curve)
        else:
            curve = _curve_from_eval_summary_row(row, axes, metric_name=metric_name)
            if curve is not None:
                curves.append(curve)
    return curves


def load_curves(
    curve_path: Path,
    *,
    manifest_path: Path | None = None,
    metric_name: str = DEFAULT_METRIC,
) -> list[EvalCurve]:
    """Load local curve or eval-summary data and attach manifest axes."""

    return build_curves(
        load_json_or_jsonl(curve_path),
        manifest_axes=load_manifest_axes(manifest_path),
        metric_name=metric_name,
    )


def latest(curve: EvalCurve, metric: str) -> float | None:
    values = _metric_values(curve, metric)
    return values[-1][1] if values else None


def best(curve: EvalCurve, metric: str) -> float | None:
    values = _metric_values(curve, metric)
    return max(value for _iteration, value in values) if values else None


def delta(curve: EvalCurve, metric: str) -> float | None:
    values = _metric_values(curve, metric)
    if len(values) < 2:
        return None
    return values[-1][1] - values[0][1]


def best_delta(curve: EvalCurve, metric: str) -> float | None:
    values = _metric_values(curve, metric)
    if not values:
        return None
    return max(value for _iteration, value in values) - values[0][1]


def early_slope(curve: EvalCurve, metric: str) -> float | None:
    values = _metric_values(curve, metric)
    if len(values) < 2:
        return None
    end = max(2, (len(values) + 1) // 2)
    return _slope(values[:end])


def late_slope(curve: EvalCurve, metric: str) -> float | None:
    values = _metric_values(curve, metric)
    if len(values) < 2:
        return None
    start = len(values) // 2
    return _slope(values[start:])


def peak_then_crash(
    curve: EvalCurve,
    metric: str,
    *,
    min_drop: float = 3.0,
) -> bool:
    values = _metric_values(curve, metric)
    if len(values) < 3:
        return False
    best_index, (_best_iter, best_value) = max(enumerate(values), key=lambda item: item[1][1])
    if best_index == len(values) - 1:
        return False
    return best_value - values[-1][1] >= min_drop


def is_flat(
    curve: EvalCurve,
    metric: str,
    *,
    epsilon: float = 1.0,
) -> bool:
    values = _metric_values(curve, metric)
    if not values:
        return False
    metric_values = [value for _iteration, value in values]
    return max(metric_values) - min(metric_values) <= epsilon


def collapse_flag(
    curve: EvalCurve,
    *,
    collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
) -> bool:
    """Return true when any point records action collapse evidence."""

    for point in curve.points:
        metrics = point.metrics
        for key in ("collapsed", "any_collapsed", "action_collapsed", "collapse_flag"):
            if bool(metrics.get(key)):
                return True
        fraction = _numeric(metrics.get("top_action_fraction"))
        if fraction is not None and fraction >= collapse_threshold:
            return True
    return False


def score_curve(
    curve: EvalCurve,
    *,
    metric: str = DEFAULT_METRIC,
    flat_epsilon: float | None = None,
    crash_drop: float | None = None,
    signal_delta: float | None = None,
    collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
) -> dict[str, Any]:
    """Score one curve for one metric with simple triage filters."""

    effective_flat_epsilon = _default_flat_epsilon(metric) if flat_epsilon is None else flat_epsilon
    effective_crash_drop = _default_crash_drop(metric) if crash_drop is None else crash_drop
    effective_signal_delta = _default_signal_delta(metric) if signal_delta is None else signal_delta
    values = _metric_values(curve, metric)
    first_iteration = values[0][0] if values else None
    latest_iteration = values[-1][0] if values else None
    best_pair = max(values, key=lambda item: item[1]) if values else (None, None)
    late = late_slope(curve, metric)
    best_lift = best_delta(curve, metric)
    return {
        "run_id": curve.run_id,
        "attempt_id": curve.attempt_id,
        "axes": curve.axes,
        "points": len(values),
        "metric": metric,
        "first_iter": first_iteration,
        "first": values[0][1] if values else None,
        "latest_iter": latest_iteration,
        "latest": values[-1][1] if values else None,
        "best_iter": best_pair[0],
        "best": best_pair[1],
        "delta": delta(curve, metric),
        "best_delta": best_lift,
        "early_slope": early_slope(curve, metric),
        "late_slope": late,
        "peak_signal": _peak_signal(values, min_lift=effective_signal_delta),
        "late_bloom": _late_bloom(values, min_lift=effective_signal_delta),
        "peak_then_crash": peak_then_crash(curve, metric, min_drop=effective_crash_drop),
        "flat": is_flat(curve, metric, epsilon=effective_flat_epsilon),
        "collapsed": collapse_flag(curve, collapse_threshold=collapse_threshold),
        "eval_health": _eval_health(curve),
    }


def score_curves(
    curves: Iterable[EvalCurve],
    *,
    metric: str = DEFAULT_METRIC,
    flat_epsilon: float | None = None,
    crash_drop: float | None = None,
    signal_delta: float | None = None,
    collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
) -> list[dict[str, Any]]:
    return [
        score_curve(
            curve,
            metric=metric,
            flat_epsilon=flat_epsilon,
            crash_drop=crash_drop,
            signal_delta=signal_delta,
            collapse_threshold=collapse_threshold,
        )
        for curve in curves
    ]


def score_curve_metrics(
    curve: EvalCurve,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    flat_epsilon: float | None = None,
    crash_drop: float | None = None,
    signal_delta: float | None = None,
    collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Score one run for every requested metric that has curve data."""

    return [
        score_curve(
            curve,
            metric=metric,
            flat_epsilon=flat_epsilon,
            crash_drop=crash_drop,
            signal_delta=signal_delta,
            collapse_threshold=collapse_threshold,
        )
        for metric in metrics
        if _metric_values(curve, metric)
    ]


def score_curves_multi(
    curves: Iterable[EvalCurve],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    flat_epsilon: float | None = None,
    crash_drop: float | None = None,
    signal_delta: float | None = None,
    collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Score many curves for many metrics without choosing a single truth metric."""

    scores: list[dict[str, Any]] = []
    for curve in curves:
        scores.extend(
            score_curve_metrics(
                curve,
                metrics=metrics,
                flat_epsilon=flat_epsilon,
                crash_drop=crash_drop,
                signal_delta=signal_delta,
                collapse_threshold=collapse_threshold,
            )
        )
    return scores


def summarize_curve_metrics(
    curve: EvalCurve,
    *,
    metrics: Sequence[str] | None = None,
    collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
) -> dict[str, Any]:
    """Summarize all available eval dimensions without choosing one truth metric."""

    requested = tuple(metrics) if metrics is not None else tuple(
        spec.name for spec in METRIC_SCHEMA if _metric_values(curve, spec.name)
    )
    families: dict[str, list[dict[str, Any]]] = {}
    for metric in requested:
        if not _metric_values(curve, metric):
            continue
        spec = METRIC_SCHEMA_BY_NAME.get(metric, MetricSpec(metric, "other", ""))
        families.setdefault(spec.family, []).append(
            score_curve(curve, metric=metric, collapse_threshold=collapse_threshold)
        )

    latest_point = curve.sorted_points()[-1] if curve.points else None
    latest_metrics = latest_point.metrics if latest_point is not None else {}
    return {
        "run_id": curve.run_id,
        "attempt_id": curve.attempt_id,
        "axes": curve.axes,
        "points": len(curve.points),
        "families": families,
        "latest_iteration": latest_point.iteration if latest_point is not None else None,
        "latest_terminal_cause": latest_metrics.get("top_terminal_cause"),
        "latest_top_action": latest_metrics.get("top_action"),
        "collapsed": collapse_flag(curve, collapse_threshold=collapse_threshold),
        "eval_health": _eval_health(curve),
    }


def render_scores(scores: Sequence[Mapping[str, Any]], *, fmt: str = "json") -> str:
    """Render scores as JSON or a compact Markdown table."""

    if fmt == "json":
        return json.dumps(list(scores), indent=2, sort_keys=True)
    if fmt != "markdown":
        raise ValueError(f"unknown format: {fmt}")
    columns = (
        "run_id",
        "metric",
        "points",
        "first",
        "latest",
        "best",
        "delta",
        "best_delta",
        "early_slope",
        "late_slope",
        "peak_signal",
        "late_bloom",
        "peak_then_crash",
        "flat",
        "collapsed",
        "eval_health",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for score in scores:
        cells = [_display(score.get(column)) for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score local CurvyTron eval curves.")
    parser.add_argument("curves", type=Path, help="Local curve JSON, eval-summary JSON, or JSONL")
    parser.add_argument("--manifest", type=Path, help="Optional matrix manifest JSON")
    parser.add_argument(
        "--metric",
        action="append",
        help=(
            "Metric to score. May be repeated or comma-separated. Defaults to "
            "mean_survival. Examples: --metric win_rate --metric mean_survival,mean_reward"
        ),
    )
    parser.add_argument(
        "--flat-epsilon",
        type=float,
        default=None,
        help="Override the metric-aware flat threshold.",
    )
    parser.add_argument(
        "--crash-drop",
        type=float,
        default=None,
        help="Override the metric-aware peak-then-crash threshold.",
    )
    parser.add_argument(
        "--signal-delta",
        type=float,
        default=None,
        help="Override the metric-aware peak/late-bloom signal threshold.",
    )
    parser.add_argument("--collapse-threshold", type=float, default=DEFAULT_COLLAPSE_THRESHOLD)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    metrics = _parse_metrics(args.metric)

    scores = score_curves_multi(
        load_curves(args.curves, manifest_path=args.manifest, metric_name=metrics[0]),
        metrics=metrics,
        flat_epsilon=args.flat_epsilon,
        crash_drop=args.crash_drop,
        signal_delta=args.signal_delta,
        collapse_threshold=args.collapse_threshold,
    )
    text = render_scores(scores, fmt=args.format) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def _payload_rows(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in ("rows", "runs", "scores", "eval_summary"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _curve_from_canonical(
    row: Mapping[str, Any],
    manifest_axes: Mapping[str, Mapping[str, Any]],
) -> EvalCurve:
    run_id = _run_id(row) or "unknown"
    axes = dict(manifest_axes.get(run_id, {}))
    axes.update(row.get("axes") or {})
    points = []
    for point in row.get("points") or []:
        if not isinstance(point, Mapping):
            continue
        iteration = _iteration(point)
        if iteration is None:
            continue
        metrics = dict(point.get("metrics") or {})
        for key, value in point.items():
            if key not in {"iteration", "checkpoint", "checkpoint_label", "metrics"}:
                metrics.setdefault(key, value)
        points.append(CurvePoint(iteration=iteration, metrics=metrics))
    return EvalCurve(
        run_id=run_id,
        attempt_id=_string_or_none(row.get("attempt_id")),
        axes=axes,
        points=_merge_points(points),
    )


def _curve_from_eval_summary_row(
    row: Mapping[str, Any],
    manifest_axes: Mapping[str, Mapping[str, Any]],
    *,
    metric_name: str,
) -> EvalCurve | None:
    run_id = _run_id(row)
    if not run_id:
        return None
    points: list[CurvePoint] = []
    for prefix in ("first", "best", "latest"):
        iteration = _int_or_none(row.get(f"{prefix}_iter"))
        metrics = _eval_summary_metrics(row, prefix=prefix, fallback_metric=metric_name)
        if iteration is None or not metrics:
            continue
        if prefix == "latest":
            for source, target in (
                ("latest_max", "max_survival"),
                ("latest_top_action", "top_action"),
                ("latest_top_fraction", "top_action_fraction"),
            ):
                if row.get(source) is not None:
                    metrics[target] = row[source]
        if row.get("any_collapsed") is not None:
            metrics["any_collapsed"] = bool(row["any_collapsed"])
        if row.get("eval_manifests") is not None:
            metrics["eval_manifest_count"] = _int_or_none(row["eval_manifests"])
        if row.get(f"{prefix}_action_entropy") is not None:
            metrics["action_entropy"] = _numeric(row.get(f"{prefix}_action_entropy"))
        if row.get(f"{prefix}_terminal_cause") is not None:
            metrics["top_terminal_cause"] = _string_or_none(row.get(f"{prefix}_terminal_cause"))
        points.append(CurvePoint(iteration=iteration, metrics=metrics))
    if not points:
        return None
    axes = dict(manifest_axes.get(run_id, {}))
    for key in ("short_name", "poller_status", "train_status", "train_stage", "latest_checkpoint"):
        if row.get(key) is not None:
            axes.setdefault(key, row[key])
    return EvalCurve(
        run_id=run_id,
        attempt_id=_string_or_none(row.get("attempt_id")),
        axes=axes,
        points=_merge_points(points),
    )


def _curve_from_eval_checkpoints_row(
    row: Mapping[str, Any],
    manifest_axes: Mapping[str, Mapping[str, Any]],
) -> EvalCurve | None:
    run_id = _run_id(row)
    if not run_id:
        return None
    raw_checkpoints = row.get("eval_checkpoints")
    if not isinstance(raw_checkpoints, list):
        return None
    points: list[CurvePoint] = []
    for checkpoint in raw_checkpoints:
        if not isinstance(checkpoint, Mapping):
            continue
        iteration = _iteration(checkpoint)
        if iteration is None:
            continue
        action_summary = checkpoint.get("action_summary")
        if not isinstance(action_summary, Mapping):
            action_summary = {}
        outcome_histogram = checkpoint.get("outcome_histogram")
        if not isinstance(outcome_histogram, Mapping):
            outcome_histogram = {}
        terminal_histogram = _first_mapping(
            checkpoint,
            (
                "terminal_cause_histogram",
                "terminal_reason_histogram",
                "terminal_histogram",
                "terminal_reasons",
            ),
        )
        action_histogram = _first_mapping(
            checkpoint,
            ("action_histogram", "ego_action_histogram", "selected_action_histogram"),
        )
        if not action_histogram:
            action_histogram = {
                key.removeprefix("action_"): value
                for key, value in action_summary.items()
                if str(key).startswith("action_") and str(key).removeprefix("action_").isdigit()
            }
        action_entropy = _first_numeric_in_mappings(
            (action_summary, checkpoint),
            ("action_entropy",),
        )
        if action_entropy is None:
            action_entropy = _normalized_histogram_entropy(action_histogram)
        metrics: dict[str, float | int | bool | str | None] = {
            "mean_survival": _numeric(checkpoint.get("mean_steps")),
            "mean_steps": _numeric(checkpoint.get("mean_steps")),
            "median_survival": _numeric(checkpoint.get("median_steps")),
            "min_survival": _numeric(checkpoint.get("min_steps")),
            "max_survival": _numeric(checkpoint.get("max_steps")),
            "mean_reward": _first_numeric(
                checkpoint,
                ("mean_reward", "reward_mean", "mean_training_reward", "training_reward"),
            ),
            "mean_training_reward": _first_numeric(
                checkpoint,
                ("mean_training_reward", "training_reward", "mean_reward", "reward_mean"),
            ),
            "bonus_count": _first_numeric(
                checkpoint,
                (
                    "bonus_count",
                    "mean_bonus_count",
                    "bonus_pickup_count",
                    "mean_bonus_pickup_count",
                ),
            ),
            "bonus_pickup_count": _first_numeric(
                checkpoint,
                (
                    "bonus_pickup_count",
                    "mean_bonus_pickup_count",
                    "bonus_count",
                    "mean_bonus_count",
                ),
            ),
            "bonus_reward": _first_numeric(
                checkpoint,
                ("bonus_reward", "mean_bonus_reward", "bonus_pickup_reward"),
            ),
            "ok_count": _int_or_none(checkpoint.get("ok_count")),
            "failure_count": _int_or_none(checkpoint.get("failure_count")),
            "top_action": _string_or_none(action_summary.get("top_action")),
            "top_action_fraction": _numeric(action_summary.get("top_action_fraction")),
            "action_entropy": action_entropy,
            "collapsed": bool(action_summary.get("collapsed")),
        }
        metrics.update(_outcome_rate_metrics(outcome_histogram, seeds=checkpoint.get("seeds")))
        metrics.update(_action_rate_metrics(action_histogram))
        metrics.update(_terminal_rate_metrics(terminal_histogram, seeds=checkpoint.get("seeds")))
        if terminal_histogram:
            top_terminal_cause, top_terminal_count = max(
                terminal_histogram.items(),
                key=lambda item: _numeric(item[1]) or 0.0,
            )
            metrics["top_terminal_cause"] = str(top_terminal_cause)
            denominator = _numeric(checkpoint.get("seeds")) or sum(
                _numeric(value) or 0.0 for value in terminal_histogram.values()
            )
            if denominator:
                metrics["top_terminal_cause_fraction"] = (
                    (_numeric(top_terminal_count) or 0.0) / denominator
                )
        ok_count = _numeric(metrics.get("ok_count"))
        failure_count = _numeric(metrics.get("failure_count"))
        if ok_count is not None and failure_count is not None and ok_count + failure_count > 0:
            metrics["failure_rate"] = failure_count / (ok_count + failure_count)
        points.append(CurvePoint(iteration=iteration, metrics=metrics))
    if not points:
        return None
    axes = dict(manifest_axes.get(run_id, {}))
    for key in (
        "short_name",
        "poller_status",
        "background_poller_status",
        "train_status",
        "train_stage",
        "latest_checkpoint",
    ):
        if row.get(key) is not None:
            axes.setdefault(key, row[key])
    return EvalCurve(
        run_id=run_id,
        attempt_id=_string_or_none(row.get("attempt_id")),
        axes=axes,
        points=_merge_points(points),
    )


def _merge_points(points: Iterable[CurvePoint]) -> tuple[CurvePoint, ...]:
    merged: dict[int, dict[str, Any]] = {}
    for point in points:
        merged.setdefault(point.iteration, {}).update(point.metrics)
    return tuple(CurvePoint(iteration=iteration, metrics=merged[iteration]) for iteration in sorted(merged))


def _metric_values(curve: EvalCurve, metric: str) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for point in curve.sorted_points():
        value = _numeric(point.metrics.get(metric))
        if value is not None:
            values.append((point.iteration, value))
    return values


def _slope(values: Sequence[tuple[int, float]]) -> float | None:
    if len(values) < 2:
        return None
    first_iter, first_value = values[0]
    latest_iter, latest_value = values[-1]
    if latest_iter == first_iter:
        return 0.0
    return (latest_value - first_value) / (latest_iter - first_iter)


def _peak_signal(values: Sequence[tuple[int, float]], *, min_lift: float) -> bool:
    """Flag temporary or non-latest signal so manual review does not miss it."""

    if len(values) < 2:
        return False
    first_value = values[0][1]
    best_index, (_best_iter, best_value) = max(enumerate(values), key=lambda item: item[1][1])
    return best_index != 0 and best_value - first_value >= min_lift


def _late_bloom(values: Sequence[tuple[int, float]], *, min_lift: float) -> bool:
    """Flag curves where the main lift arrives in the second half."""

    if len(values) < 3:
        return False
    first_value = values[0][1]
    latest_value = values[-1][1]
    if latest_value - first_value < min_lift:
        return False
    midpoint = max(1, len(values) // 2)
    first_half_best = max(value for _iteration, value in values[:midpoint])
    second_half_best = max(value for _iteration, value in values[midpoint:])
    return second_half_best - first_half_best >= min_lift


def _eval_health(curve: EvalCurve) -> str:
    if not curve.points:
        return "no_points"
    failed = 0
    total = 0
    for point in curve.points:
        ok = point.metrics.get("ok")
        if ok is not None:
            total += 1
            if not bool(ok):
                failed += 1
        failure_count = _numeric(point.metrics.get("failure_count"))
        if failure_count:
            failed += int(failure_count)
    if failed:
        return "has_failures"
    if total or curve.points:
        return "ok"
    return "unknown"


def _default_flat_epsilon(metric: str) -> float:
    if _rate_like_metric(metric):
        return 0.02
    if "reward" in metric:
        return 0.05
    return 1.0


def _default_crash_drop(metric: str) -> float:
    if _rate_like_metric(metric):
        return 0.1
    if "reward" in metric:
        return 0.2
    return 3.0


def _default_signal_delta(metric: str) -> float:
    if _rate_like_metric(metric):
        return 0.05
    if "reward" in metric:
        return 0.1
    return DEFAULT_SIGNAL_DELTA


def _rate_like_metric(metric: str) -> bool:
    return metric.endswith("_rate") or metric.endswith("_fraction") or metric in {
        "win_rate",
        "draw_rate",
        "survival_fraction",
    }


def _eval_summary_metrics(
    row: Mapping[str, Any],
    *,
    prefix: str,
    fallback_metric: str,
) -> dict[str, float | int | bool | str | None]:
    metric_columns = {
        "mean": fallback_metric,
        "mean_survival": "mean_survival",
        "survival_mean": "mean_survival",
        "median_survival": "median_survival",
        "max_survival": "max_survival",
        "win_rate": "win_rate",
        "outcome_win_rate": "win_rate",
        "mean_reward": "mean_reward",
        "reward_mean": "mean_reward",
        "training_reward": "mean_reward",
        "mean_training_reward": "mean_reward",
        "bonus_pickup_count": "bonus_pickup_count",
        "bonus_reward": "bonus_reward",
    }
    metrics: dict[str, float | int | bool | str | None] = {}
    for suffix, metric in metric_columns.items():
        value = _numeric(row.get(f"{prefix}_{suffix}"))
        if value is not None:
            metrics[metric] = value
    return metrics


def _outcome_rate_metrics(
    outcome_histogram: Mapping[str, Any],
    *,
    seeds: Any,
) -> dict[str, float]:
    counts = {str(key): _numeric(value) or 0.0 for key, value in outcome_histogram.items()}
    denominator = _numeric(seeds) or sum(counts.values())
    if not denominator:
        return {}
    return {
        "win_rate": counts.get("win", 0.0) / denominator,
        "loss_rate": counts.get("loss", 0.0) / denominator,
        "draw_rate": counts.get("draw", 0.0) / denominator,
        "cap_rate": counts.get("cap", 0.0) / denominator,
    }


def _action_rate_metrics(action_histogram: Mapping[str, Any]) -> dict[str, float]:
    counts = {str(key): _numeric(value) or 0.0 for key, value in action_histogram.items()}
    total = sum(counts.values())
    if not total:
        return {}
    metrics = {
        f"action_{key}_rate": value / total
        for key, value in counts.items()
    }
    for key, label in (("0", "left_rate"), ("1", "straight_rate"), ("2", "right_rate")):
        if key in counts:
            metrics[label] = counts[key] / total
    return metrics


def _terminal_rate_metrics(
    terminal_histogram: Mapping[str, Any],
    *,
    seeds: Any,
) -> dict[str, float]:
    counts = {str(key): _numeric(value) or 0.0 for key, value in terminal_histogram.items()}
    denominator = _numeric(seeds) or sum(counts.values())
    if not denominator:
        return {}
    metrics = {
        f"terminal_{_metric_slug(key)}_rate": value / denominator
        for key, value in counts.items()
    }
    groups = {
        "wall_rate": ("wall", "normal_wall", "round_wall"),
        "own_trail_rate": ("own_trail", "self_trail"),
        "opponent_trail_rate": ("opponent_trail", "enemy_trail", "other_trail"),
        "timeout_rate": ("timeout", "cap", "max_steps", "timeout_truncated"),
    }
    for metric, names in groups.items():
        metrics[metric] = sum(counts.get(name, 0.0) for name in names) / denominator
    return metrics


def _normalized_histogram_entropy(histogram: Mapping[str, Any]) -> float | None:
    counts = [_numeric(value) or 0.0 for value in histogram.values()]
    counts = [count for count in counts if count > 0]
    total = sum(counts)
    if total <= 0:
        return None
    if len(counts) <= 1:
        return 0.0
    entropy = -sum((count / total) * math.log(count / total) for count in counts)
    return entropy / math.log(len(counts))


def _first_mapping(row: Mapping[str, Any], keys: Sequence[str]) -> Mapping[str, Any]:
    for key in keys:
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _first_numeric(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _numeric(row.get(key))
        if value is not None:
            return value
    return None


def _first_numeric_in_mappings(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> float | None:
    for row in rows:
        value = _first_numeric(row, keys)
        if value is not None:
            return value
    return None


def _metric_slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_")


def _command_axes(command: Sequence[Any]) -> dict[str, Any]:
    flags = {
        "--reward-variant": "reward_variant",
        "--env-variant": "env_variant",
        "--opponent-policy-kind": "opponent_policy_kind",
        "--source-state-trail-render-mode": "source_state_trail_render_mode",
        "--batch-size": "batch_size",
        "--num-simulations": "num_simulations",
        "--collector-env-num": "collector_env_num",
        "--source-max-steps": "source_max_steps",
    }
    axes: dict[str, Any] = {}
    for index, token in enumerate(command):
        key = flags.get(str(token))
        if key and index + 1 < len(command):
            axes[key] = _coerce_scalar(command[index + 1])
    return axes


def _run_id(row: Mapping[str, Any]) -> str | None:
    return _string_or_none(row.get("run_id") or row.get("short_name") or row.get("name"))


def _iteration(row: Mapping[str, Any]) -> int | None:
    for key in ("iteration", "checkpoint", "checkpoint_label"):
        value = row.get(key)
        if value is None:
            continue
        parsed = _int_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _coerce_scalar(value: Any) -> Any:
    parsed_int = _int_or_none(value)
    if parsed_int is not None:
        return parsed_int
    parsed_float = _numeric(value)
    if parsed_float is not None and not isinstance(value, str):
        return parsed_float
    return value


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("iteration_"):
            raw = raw.removeprefix("iteration_")
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _parse_metrics(raw_metrics: Sequence[str] | None) -> tuple[str, ...]:
    if not raw_metrics:
        return DEFAULT_METRICS
    metrics: list[str] = []
    for raw in raw_metrics:
        for metric in raw.split(","):
            cleaned = metric.strip()
            if cleaned and cleaned not in metrics:
                metrics.append(cleaned)
    return tuple(metrics or DEFAULT_METRICS)


if __name__ == "__main__":
    main()
