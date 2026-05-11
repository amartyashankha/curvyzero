"""Summarize LightZero dummy Pong scorecard artifacts.

The default mode reads one or more local ``summary.json`` files or Modal
Volume refs and prints a compact Markdown/TSV/JSON table. A second, optional
``debug-mcts`` mode emits the first N decisions from a LightZero MCTS
checkpoint, including the adapter's exposed forward-output fields when present.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


BASELINE_POLICIES = {"random_uniform", "lagged_track_ball_1", "track_ball"}
DEFAULT_ACTION_LABELS = ("up", "stay", "down")
RECENT_DUMMY_PONG_PRESET = (
    {
        "alias": "sparse-rung1",
        "required": "ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-2x/attempts/train-2048x32-sparse-h120/eval/mcts-scoreboard-rung1-s10-2x-iter0-iter32-best-s1701/summary.json",
        "sources": (
            "ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-2x/attempts/train-2048x32-sparse-h120/eval/mcts-scoreboard-rung1-s10-2x-iter0-iter32-best-s1701/summary.json",
        ),
    },
    {
        "alias": "upc25",
        "required": "ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-sim8/attempts/train-2048x32-sim8-upc25-sparse-h120/eval/mcts-scoreboard-upc25-sim8-iter0-iter50-best-e8/summary.json",
        "sources": (
            "ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-sim8/attempts/train-2048x32-sim8-upc25-sparse-h120/eval/mcts-scoreboard-upc25-sim8-iter0-iter50-best-e8/summary.json",
        ),
    },
    {
        "alias": "epscollect",
        "required": "ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-epscollect/attempts/train-2048x32-sim8-upc25-epscollect-sparse-h120/eval/mcts-scoreboard-upc25-epscollect-sim8-iter0-iter50-best-e8/summary.json",
        "sources": (
            "ref:training/lightzero-dummy-pong/lz-dpong-sparse-h120-lag1-s10-upc25-epscollect/attempts/train-2048x32-sim8-upc25-epscollect-sparse-h120/eval/mcts-scoreboard-upc25-epscollect-sim8-iter0-iter50-best-e8/summary.json",
        ),
    },
    {
        "alias": "contact-pressure-modest",
        "required": "ref:training/lightzero-dummy-pong/lz-dpong-20260509T175407Z-77159cc3a6b4/attempts/attempt-20260509T175407Z-8105d62c1e00/eval/mcts-scoreboard-contact-pressure-modest-rung/summary.json",
        "sources": (
            "artifacts/local/contact-pressure-modest-rung-2026-05-09/contact-pressure-summary.json",
            "ref:training/lightzero-dummy-pong/lz-dpong-20260509T175407Z-77159cc3a6b4/attempts/attempt-20260509T175407Z-8105d62c1e00/eval/mcts-scoreboard-contact-pressure-modest-rung/summary.json",
        ),
    },
    {
        "alias": "raster-smoke",
        "required": "ref:training/lightzero-dummy-pong/lz-dpong-raster-flat-h120-lag1-s10/attempts/train-512x8-raster-h120/eval/mcts-scoreboard-iter8-raster-h120-s1701-small/summary.json",
        "sources": (
            "ref:training/lightzero-dummy-pong/lz-dpong-raster-flat-h120-lag1-s10/attempts/train-512x8-raster-h120/eval/mcts-scoreboard-iter8-raster-h120-s1701-small/summary.json",
        ),
    },
)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "debug-mcts":
        _run_debug_mcts(_parse_debug_args(argv[1:]))
        return
    if argv and argv[0] == "table":
        argv = argv[1:]
    _run_table(_parse_table_args(argv))


def _parse_table_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summaries",
        nargs="*",
        help=(
            "Local summary.json path, summary directory, ref:<volume/path>, "
            "volume:<volume/path>, or a bare relative Volume ref."
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
        "--preset",
        choices=("recent-dummy-pong",),
        default=None,
        help="Add the recent dummy Pong sparse/raster/contact scorecard refs.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing preset/local refs and report the refs needed to complete the table.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Render the compact decision table columns.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Also save the rendered table.")
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Include baseline sanity rows that do not contain a checkpoint policy.",
    )
    parser.add_argument(
        "--baseline-opponents-only",
        action="store_true",
        help="Only include checkpoint rows against baseline opponents.",
    )
    args = parser.parse_args(argv)
    if not args.summaries and args.preset is None:
        parser.error("at least one summary or --preset is required")
    return args


def _parse_debug_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit first-N LightZero MCTS eval-mode decisions for one checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "LightZero checkpoint as lightzero:LABEL=PATH_OR_REF, LABEL=PATH_OR_REF, "
            "or PATH_OR_REF."
        ),
    )
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent-policy", default="random_uniform")
    parser.add_argument("--lightzero-env", default="dummy_pong_lag1")
    parser.add_argument("--feature-mode", default="tabular_ego")
    parser.add_argument("--max-env-step", type=int, default=64)
    parser.add_argument("--num-simulations", type=int, default=2)
    parser.add_argument(
        "--pong-reset-profile",
        choices=("default", "contact_pressure"),
        default="default",
        help="Reset profile for first-N debug states.",
    )
    parser.add_argument(
        "--pong-reset-pressure-agent",
        choices=("random", "player_0", "player_1"),
        default="random",
        help="Pressure agent for contact-pressure debug states.",
    )
    parser.add_argument(
        "--scoreability-summary",
        type=Path,
        default=None,
        help=(
            "Optional contact-pressure scoreability summary.json. When set, "
            "debug rows are sampled from matching reset groups instead of a rollout walk."
        ),
    )
    parser.add_argument(
        "--scoreability-best-action",
        default=None,
        help="Filter scoreability groups by best action: up/stay/down or 0/1/2.",
    )
    parser.add_argument(
        "--scoreability-require-unique-best",
        action="store_true",
        help="Keep only scoreability groups with exactly one best action.",
    )
    parser.add_argument(
        "--decision-mode",
        choices=("eval", "collect", "both"),
        default="eval",
        help="Forward through eval_mode, collect_mode, or both on each observation.",
    )
    parser.add_argument("--collect-repeats", type=int, default=0)
    parser.add_argument("--collect-temperature", type=float, default=0.25)
    parser.add_argument("--collect-epsilon", type=float, default=0.0)
    parser.add_argument("--to-play", type=int, default=-1)
    parser.add_argument(
        "--volume-root",
        type=Path,
        default=Path(os.environ.get("CURVYZERO_RUNS_MOUNT", "/runs")),
    )
    parser.add_argument("--format", choices=("md", "json", "jsonl"), default="jsonl")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def _run_table(args: argparse.Namespace) -> None:
    table_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for entry in _table_source_entries(args):
        loaded = None
        last_error = None
        for source in entry["sources"]:
            try:
                loaded = (*_load_summary_source(source, volume_root=args.volume_root), source)
                break
            except FileNotFoundError as exc:
                last_error = exc
        if loaded is None:
            if args.allow_missing:
                missing.append(entry["required"])
                continue
            if last_error is not None:
                raise last_error
            raise FileNotFoundError(entry["required"])
        summary, resolved, loaded_source = loaded
        source_label = entry.get("alias") or _source_label(loaded_source, resolved)
        table_rows.extend(
            _summary_table_rows(
                summary,
                source_label=source_label,
                include_baselines=args.include_baselines,
                baseline_opponents_only=args.baseline_opponents_only,
            )
        )
    table_rows.sort(key=lambda row: (str(row["source"]), str(row["checkpoint"]), str(row["opponent"])))
    rendered = _render_rows(table_rows, output_format=args.format, compact=args.compact)
    print(rendered)
    if missing:
        print(
            "\nMissing refs needed for the full preset:\n"
            + "\n".join(f"- {source}" for source in missing),
            file=sys.stderr,
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")


def _table_source_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if args.preset == "recent-dummy-pong":
        entries.extend(
            {
                "alias": str(entry["alias"]),
                "required": str(entry["required"]),
                "sources": [str(source) for source in entry["sources"]],
            }
            for entry in RECENT_DUMMY_PONG_PRESET
        )
    entries.extend(
        {"alias": "", "required": source, "sources": [source]} for source in args.summaries
    )
    return entries


def _load_summary_source(source: str, *, volume_root: Path) -> tuple[dict[str, Any], Path]:
    path = _resolve_source_path(source, volume_root=volume_root)
    payload = _read_json(path)
    if "scoreboard_rows" in payload or "pong_scorecard" in payload:
        return payload, path
    nested_scoreboard = payload.get("scoreboard")
    if isinstance(nested_scoreboard, dict) and "scoreboard_rows" in nested_scoreboard:
        merged = dict(payload)
        merged.setdefault("scoreboard_rows", nested_scoreboard.get("scoreboard_rows", []))
        merged.setdefault("checkpoint_specs", nested_scoreboard.get("checkpoint_specs", []))
        merged.setdefault("episodes_per_match", nested_scoreboard.get("episodes_per_match"))
        merged.setdefault("total_episodes", nested_scoreboard.get("total_episodes"))
        return merged, path
    output_refs = payload.get("output_refs")
    if isinstance(output_refs, dict) and output_refs.get("summary_json"):
        next_path = _resolve_source_path(str(output_refs["summary_json"]), volume_root=volume_root)
        return _load_summary_source(str(next_path), volume_root=volume_root)
    raise ValueError(f"does not look like a dummy Pong scorecard summary: {path}")


def _resolve_source_path(source: str, *, volume_root: Path) -> Path:
    ref_text = _strip_ref_prefix(source)
    if ref_text is not None:
        path = volume_root / Path(ref_text)
    else:
        candidate = Path(source)
        if candidate.exists() or candidate.is_absolute():
            path = candidate
        else:
            volume_candidate = volume_root / candidate
            path = volume_candidate if volume_candidate.exists() else candidate
    if path.is_dir():
        path = path / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"summary file not found: {path}")
    return path


def _strip_ref_prefix(value: str) -> str | None:
    for prefix in ("ref:", "volume:"):
        if value.startswith(prefix):
            ref = value[len(prefix) :]
            if ref.startswith("/") or ".." in Path(ref).parts:
                raise ValueError(f"unsafe Volume ref: {value!r}")
            return ref
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _summary_table_rows(
    summary: dict[str, Any],
    *,
    source_label: str,
    include_baselines: bool,
    baseline_opponents_only: bool,
) -> list[dict[str, Any]]:
    if isinstance(summary.get("pong_scorecard"), dict):
        return [_trainer_side_scorecard_row(summary, source_label=source_label)]

    rows = _scoreboard_rows(summary)
    checkpoint_labels = _checkpoint_labels(summary)
    action_labels = tuple(summary.get("action_labels") or DEFAULT_ACTION_LABELS)
    histograms_by_pair_group = _action_histograms_by_pair_group(summary)
    comparison_context = _comparison_context(rows)
    seating = _seating_label(summary)
    profile = _profile_label(summary)
    table_rows = []
    for row in rows:
        policies = [str(policy) for policy in row.get("policies", [])]
        if not policies and row.get("pair_group_id"):
            policies = _policies_from_pair_group(str(row["pair_group_id"]))
        if baseline_opponents_only and not any(policy in BASELINE_POLICIES for policy in policies):
            continue
        subject_policies = [
            policy
            for policy in policies
            if policy in checkpoint_labels or policy not in BASELINE_POLICIES
        ]
        if not subject_policies and include_baselines:
            subject_policies = policies[:1]
        for subject_policy in subject_policies:
            table_rows.append(
                _checkpoint_row(
                    row,
                    subject_policy=subject_policy,
                    policies=policies,
                    checkpoint_labels=checkpoint_labels,
                    action_labels=action_labels,
                    fallback_histograms=histograms_by_pair_group,
                    comparison_context=comparison_context,
                    seating=seating,
                    profile=profile,
                    source_label=source_label,
                )
            )
    return table_rows


def _scoreboard_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("scoreboard_rows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    rows = summary.get("pair_groups")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _checkpoint_labels(summary: dict[str, Any]) -> dict[str, str]:
    labels = {}
    specs = summary.get("checkpoint_specs", [])
    if isinstance(specs, list):
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            policy_id = spec.get("policy_id")
            if not policy_id:
                continue
            label = spec.get("checkpoint_label") or spec.get("checkpoint_arg") or policy_id
            labels[str(policy_id)] = str(label)
    policies = summary.get("checkpoint_policies", [])
    if isinstance(policies, list):
        for policy in policies:
            if not isinstance(policy, dict):
                continue
            policy_id = policy.get("policy_id")
            if policy_id and str(policy_id) not in labels:
                labels[str(policy_id)] = str(policy_id)
    return labels


def _checkpoint_row(
    row: dict[str, Any],
    *,
    subject_policy: str,
    policies: list[str],
    checkpoint_labels: dict[str, str],
    action_labels: tuple[str, ...],
    fallback_histograms: dict[str, dict[str, list[int]]],
    comparison_context: dict[str, Any],
    seating: str,
    profile: str,
    source_label: str,
) -> dict[str, Any]:
    pair_group_id = str(row.get("pair_group_id", ""))
    wins_by_policy = _dict(row.get("wins_by_policy"))
    episodes = _int_or_none(row.get("episodes"))
    truncations = _int_or_zero(row.get("truncations"))
    wins = _int_or_zero(wins_by_policy.get(subject_policy))
    losses = sum(
        _int_or_zero(count)
        for policy, count in wins_by_policy.items()
        if str(policy) != subject_policy
    )
    survival = _stats_from_row(row, "survival_steps", "mean_steps", "median_steps", "p90_steps")
    shaped = _policy_stats(row, subject_policy, "shaped_loss_delay_return_stats_by_policy")
    if shaped is None:
        shaped = _mean_only(row, subject_policy, "mean_shaped_loss_delay_return_by_policy")
    score = _policy_stats(row, subject_policy, "score_return_stats_by_policy")
    if score is None:
        score = _mean_only(row, subject_policy, "mean_reward_by_policy")
    histograms = _dict(row.get("action_histogram_by_policy"))
    histogram = histograms.get(subject_policy)
    if histogram is None:
        histogram = fallback_histograms.get(pair_group_id, {}).get(subject_policy)
    histogram_list = _int_list(histogram)
    warnings = _collapse_warnings(
        histogram_list,
        action_labels=action_labels,
        wins=wins,
        episodes=episodes,
        truncations=truncations,
    )
    warnings.extend(
        _comparison_warnings(
            row,
            subject_policy=subject_policy,
            policies=policies,
            survival_mean=_stat(survival, "mean"),
            score_mean=_stat(score, "mean"),
            comparison_context=comparison_context,
        )
    )
    return {
        "source": source_label,
        "checkpoint": checkpoint_labels.get(subject_policy, subject_policy),
        "policy_id": subject_policy,
        "opponent": _opponent_label(policies, subject_policy),
        "seating": seating,
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "truncations": truncations,
        "survival_mean": _stat(survival, "mean"),
        "survival_median": _stat(survival, "median"),
        "survival_p90": _stat(survival, "p90"),
        "shaped_loss_delay_return": _stat(shaped, "mean"),
        "score_return": _stat(score, "mean"),
        "action_histogram": _format_histogram(histogram_list, action_labels=action_labels),
        "action_entropy": _normalized_entropy(histogram_list),
        "feature_reset_profile": profile,
        "warnings": ",".join(dict.fromkeys(warnings)) if warnings else "",
    }


def _trainer_side_scorecard_row(summary: dict[str, Any], *, source_label: str) -> dict[str, Any]:
    scorecard = _dict(summary.get("pong_scorecard"))
    config = _dict(summary.get("config_surface"))
    actions_by_agent = _dict(scorecard.get("action_counts_by_agent"))
    player0_hist = _hist_from_label_counts(_dict(actions_by_agent.get("player_0")))
    episodes = _int_or_none(scorecard.get("episodes"))
    wins = _int_or_zero(scorecard.get("wins"))
    losses = _int_or_zero(scorecard.get("losses"))
    truncations = _int_or_zero(scorecard.get("timeouts"))
    action_labels = tuple(scorecard.get("action_labels") or DEFAULT_ACTION_LABELS)
    warnings = _collapse_warnings(
        player0_hist,
        action_labels=action_labels,
        wins=wins,
        episodes=episodes,
        truncations=truncations,
    )
    label = str(summary.get("run_id") or summary.get("attempt_id") or "trainer_side")
    return {
        "source": source_label,
        "checkpoint": label,
        "policy_id": "lightzero_trainer_side",
        "opponent": str(config.get("opponent_policy") or "unknown"),
        "seating": "trainer-side",
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "truncations": truncations,
        "survival_mean": _stat(_dict(scorecard.get("survival_steps")), "mean"),
        "survival_median": _stat(_dict(scorecard.get("survival_steps")), "median"),
        "survival_p90": _stat(_dict(scorecard.get("survival_steps")), "p90"),
        "shaped_loss_delay_return": _stat(_dict(scorecard.get("shaped_loss_delay_return")), "mean"),
        "score_return": _stat(_dict(scorecard.get("score_return")), "mean"),
        "action_histogram": _format_histogram(player0_hist, action_labels=action_labels),
        "action_entropy": _normalized_entropy(player0_hist),
        "feature_reset_profile": _profile_label(summary),
        "warnings": ",".join(warnings) if warnings else "",
    }


def _action_histograms_by_pair_group(
    summary: dict[str, Any],
) -> dict[str, dict[str, list[int]]]:
    histograms_by_pair_group: dict[str, dict[str, list[int]]] = {}
    matchups = summary.get("matchups", [])
    if not isinstance(matchups, list):
        return histograms_by_pair_group
    for matchup in matchups:
        if not isinstance(matchup, dict):
            continue
        pair_group_id = matchup.get("pair_group_id")
        if pair_group_id is None:
            continue
        pair_histograms = histograms_by_pair_group.setdefault(str(pair_group_id), {})
        action_histograms = matchup.get("action_histogram_by_policy", {})
        if not isinstance(action_histograms, dict):
            continue
        for policy_id, counts in action_histograms.items():
            counts_list = _int_list(counts)
            if not counts_list:
                continue
            dest = pair_histograms.setdefault(str(policy_id), [0] * len(counts_list))
            if len(dest) < len(counts_list):
                dest.extend([0] * (len(counts_list) - len(dest)))
            for index, count in enumerate(counts_list):
                dest[index] += count
    return histograms_by_pair_group


def _comparison_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    random_survival_by_opponent: dict[str, float] = {}
    for row in rows:
        policies = [str(policy) for policy in row.get("policies", [])]
        if not policies and row.get("pair_group_id"):
            policies = _policies_from_pair_group(str(row["pair_group_id"]))
        if "random_uniform" not in policies:
            continue
        opponents = [policy for policy in policies if policy != "random_uniform"]
        if not opponents:
            continue
        survival = _stats_from_row(
            row,
            "survival_steps",
            "mean_steps",
            "median_steps",
            "p90_steps",
        )
        mean = _float_or_none(_stat(survival, "mean"))
        if mean is None:
            continue
        for opponent in opponents:
            random_survival_by_opponent[opponent] = mean
    return {"random_survival_by_opponent": random_survival_by_opponent}


def _comparison_warnings(
    row: dict[str, Any],
    *,
    subject_policy: str,
    policies: list[str],
    survival_mean: Any,
    score_mean: Any,
    comparison_context: dict[str, Any],
) -> list[str]:
    warnings = []
    score_value = _float_or_none(score_mean)
    score_by_policy = _dict(row.get("score_return_stats_by_policy"))
    if not score_by_policy:
        score_by_policy = {
            policy: {"mean": value}
            for policy, value in _dict(row.get("mean_reward_by_policy")).items()
        }
    random_score = _stat(_dict(score_by_policy.get("random_uniform")), "mean")
    random_score_value = _float_or_none(random_score)
    if (
        subject_policy != "random_uniform"
        and "random_uniform" in policies
        and score_value is not None
        and random_score_value is not None
        and score_value < random_score_value
    ):
        warnings.append("worse_than_random")

    survival_value = _float_or_none(survival_mean)
    if subject_policy != "random_uniform" and survival_value is not None:
        for opponent in policies:
            if opponent == subject_policy:
                continue
            baseline = _dict(comparison_context.get("random_survival_by_opponent")).get(opponent)
            baseline_value = _float_or_none(baseline)
            if baseline_value is not None and survival_value <= baseline_value:
                warnings.append("no_survival_gain_if_comparable")
                break
    return warnings


def _stats_from_row(
    row: dict[str, Any],
    stats_key: str,
    mean_key: str,
    median_key: str,
    p90_key: str,
) -> dict[str, Any]:
    stats = _dict(row.get(stats_key))
    if stats:
        return stats
    return {
        "mean": row.get(mean_key),
        "median": row.get(median_key),
        "p90": row.get(p90_key),
    }


def _policy_stats(row: dict[str, Any], policy: str, key: str) -> dict[str, Any] | None:
    stats_by_policy = _dict(row.get(key))
    stats = stats_by_policy.get(policy)
    return _dict(stats) or None


def _mean_only(row: dict[str, Any], policy: str, key: str) -> dict[str, Any] | None:
    by_policy = _dict(row.get(key))
    if policy not in by_policy:
        return None
    return {"mean": by_policy.get(policy)}


def _collapse_warnings(
    histogram: list[int],
    *,
    action_labels: tuple[str, ...],
    wins: int,
    episodes: int | None,
    truncations: int,
) -> list[str]:
    warnings = []
    total = sum(histogram)
    if not histogram:
        warnings.append("missing_action_hist")
    elif total <= 0:
        warnings.append("empty_action_hist")
    else:
        if len(histogram) >= 3 and histogram[2] == 0:
            warnings.append("zero_down")
        if histogram and histogram[0] == 0:
            warnings.append("zero_up")
        max_index, max_count = max(enumerate(histogram), key=lambda item: item[1])
        del max_index
        share = max_count / total
        if share > 0.95:
            warnings.append("single_action_gt_95pct")
    if episodes is not None and episodes > 0:
        if wins == 0:
            warnings.append("no_wins")
        if truncations == episodes:
            warnings.append("all_truncated")
    return warnings


def _render_rows(rows: list[dict[str, Any]], *, output_format: str, compact: bool) -> str:
    if compact:
        columns = [
            "source",
            "opponent",
            "checkpoint",
            "score",
            "shaped",
            "survival mean/p90",
            "actions",
            "entropy",
            "feature/reset",
            "warnings",
        ]
    else:
        columns = [
            "source",
            "checkpoint",
            "opponent",
            "seating",
            "episodes",
            "w/l/t",
            "survival mean/med/p90",
            "shaped",
            "score",
            "actions",
            "entropy",
            "feature/reset",
            "warnings",
        ]
    display_rows = [_display_row(row) for row in rows]
    if output_format == "json":
        return json.dumps(rows, indent=2, sort_keys=True)
    if output_format == "tsv":
        lines = ["\t".join(columns)]
        for row in display_rows:
            lines.append("\t".join(row[column] for column in columns))
        return "\n".join(lines)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in display_rows:
        lines.append("| " + " | ".join(_escape_md(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _display_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "source": str(row.get("source", "")),
        "checkpoint": str(row.get("checkpoint", "")),
        "opponent": str(row.get("opponent", "")),
        "seating": str(row.get("seating", "")),
        "episodes": _fmt_value(row.get("episodes")),
        "w/l/t": (
            f"{_fmt_value(row.get('wins'))}/"
            f"{_fmt_value(row.get('losses'))}/"
            f"{_fmt_value(row.get('truncations'))}"
        ),
        "survival mean/med/p90": (
            f"{_fmt_value(row.get('survival_mean'))}/"
            f"{_fmt_value(row.get('survival_median'))}/"
            f"{_fmt_value(row.get('survival_p90'))}"
        ),
        "survival mean/p90": (
            f"{_fmt_value(row.get('survival_mean'))}/"
            f"{_fmt_value(row.get('survival_p90'))}"
        ),
        "shaped": _fmt_value(row.get("shaped_loss_delay_return")),
        "score": _fmt_value(row.get("score_return")),
        "actions": str(row.get("action_histogram", "")),
        "entropy": _fmt_value(row.get("action_entropy")),
        "feature/reset": str(row.get("feature_reset_profile", "")),
        "warnings": str(row.get("warnings", "")),
    }


def _run_debug_mcts(args: argparse.Namespace) -> None:
    if args.rows < 1:
        raise ValueError("--rows must be at least 1")
    if args.collect_repeats < 0:
        raise ValueError("--collect-repeats must be non-negative")
    label, checkpoint_ref = _parse_checkpoint_ref(args.checkpoint)
    checkpoint_path = _resolve_source_path(checkpoint_ref, volume_root=args.volume_root)

    from curvyzero.training.dummy_pong import ACTION_LABELS, PongConfig, PongEnv
    from curvyzero.training.lightzero_dummy_pong_policy import (
        load_lightzero_mcts_eval_mode_checkpoint,
    )

    spec = load_lightzero_mcts_eval_mode_checkpoint(
        policy_id=f"lightzero_{_clean_label(label or checkpoint_path.stem)}",
        checkpoint_path=checkpoint_path,
        env=args.lightzero_env,
        feature_mode=args.feature_mode,
        opponent_policy=args.opponent_policy,
        seed=args.seed,
        max_env_step=args.max_env_step,
        num_simulations=args.num_simulations,
    )
    rows = []
    selected_states = _scoreability_debug_states(args, limit=args.rows)
    if selected_states:
        for selected in selected_states:
            if len(rows) >= args.rows:
                break
            episode_seed = int(selected["reset_seed"])
            pressure_agent = str(selected["pressure_agent"])
            env = PongEnv(
                PongConfig(
                    max_steps=args.max_env_step,
                    reset_profile="contact_pressure",
                    reset_pressure_agent=pressure_agent,
                )
            )
            observations = env.reset(seed=episode_seed)
            observation = observations["player_0"]
            opponent = _make_debug_opponent(
                args.opponent_policy,
                seed=args.seed + 1000 + len(rows),
            )
            opponent.reset(episode_seed, "player_1")
            opponent_action = int(
                opponent.action(observations["player_1"], env.raster_observation(), "player_1")
            )
            rows.append(
                _debug_decision_row(
                    args=args,
                    spec_policy=spec.policy,
                    checkpoint_label=label or checkpoint_path.name,
                    row_index=len(rows),
                    episode_seed=episode_seed,
                    episode=len(rows),
                    observation=observation,
                    raster=env.raster_observation(),
                    opponent_action=opponent_action,
                    scoreability_group=selected,
                )
            )
    else:
        episode = 0
        while len(rows) < args.rows:
            episode_seed = args.seed + episode
            reset_pressure_agent = (
                args.pong_reset_pressure_agent
                if args.pong_reset_profile == "contact_pressure"
                and args.pong_reset_pressure_agent != "random"
                else "random"
            )
            env = PongEnv(
                PongConfig(
                    max_steps=args.max_env_step,
                    reset_profile=args.pong_reset_profile,
                    reset_pressure_agent=reset_pressure_agent,
                )
            )
            observations = env.reset(seed=episode_seed)
            policy = spec.policy
            opponent = _make_debug_opponent(args.opponent_policy, seed=args.seed + 1000 + episode)
            policy.reset(episode_seed, "player_0")
            opponent.reset(episode_seed, "player_1")
            while len(rows) < args.rows:
                raster = env.raster_observation()
                observation = observations["player_0"]
                opponent_action = int(opponent.action(observations["player_1"], raster, "player_1"))
                row = _debug_decision_row(
                    args=args,
                    spec_policy=spec.policy,
                    checkpoint_label=label or checkpoint_path.name,
                    row_index=len(rows),
                    episode_seed=episode_seed,
                    episode=episode,
                    observation=observation,
                    raster=raster,
                    opponent_action=opponent_action,
                    scoreability_group=None,
                )
                rows.append(row)
                step = env.step({"player_0": int(row["action"]), "player_1": opponent_action})
                observations = step.observations
                if step.terminated or step.truncated:
                    break
            episode += 1
    rendered = _render_debug_rows(rows, output_format=args.format)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")


def _parse_checkpoint_ref(value: str) -> tuple[str | None, str]:
    if value.startswith("lightzero:"):
        value = value[len("lightzero:") :]
    if "=" not in value:
        return None, value
    label, ref = value.split("=", 1)
    label = label.strip()
    ref = ref.strip()
    if not label or not ref:
        raise ValueError("checkpoint label and path/ref must be non-empty")
    return label, ref


def _make_debug_opponent(name: str, *, seed: int) -> Any:
    from curvyzero.training.dummy_pong_eval import (
        LaggedTrackBallPolicy,
        RandomUniformPolicy,
        TrackBallPolicy,
    )

    if name == "random_uniform":
        return RandomUniformPolicy(seed=seed)
    if name == "lagged_track_ball_1":
        return LaggedTrackBallPolicy(delay=1)
    if name == "track_ball":
        return TrackBallPolicy()
    raise ValueError(f"unsupported debug opponent policy: {name!r}")


def _debug_decision_row(
    *,
    args: argparse.Namespace,
    spec_policy: Any,
    checkpoint_label: str,
    row_index: int,
    episode_seed: int,
    episode: int,
    observation: Any,
    raster: Any,
    opponent_action: int,
    scoreability_group: dict[str, Any] | None,
) -> dict[str, Any]:
    from curvyzero.training.dummy_pong import ACTION_LABELS

    eval_debug = None
    collect_debug_rows: list[dict[str, Any]] = []
    if args.decision_mode in {"eval", "both"}:
        eval_debug = _run_policy_decision(
            spec_policy,
            observation=observation,
            raster=raster,
            feature_mode=args.feature_mode,
            max_env_step=args.max_env_step,
            mode="eval",
            to_play=args.to_play,
            collect_temperature=args.collect_temperature,
            collect_epsilon=args.collect_epsilon,
        )
    repeat_count = args.collect_repeats
    if args.decision_mode == "collect" and repeat_count == 0:
        repeat_count = 1
    elif args.decision_mode == "both" and repeat_count == 0:
        repeat_count = 4
    if args.decision_mode in {"collect", "both"}:
        import numpy as np

        for repeat in range(repeat_count):
            np.random.seed(args.seed + row_index * 1000 + repeat)
            collect_debug_rows.append(
                _run_policy_decision(
                    spec_policy,
                    observation=observation,
                    raster=raster,
                    feature_mode=args.feature_mode,
                    max_env_step=args.max_env_step,
                    mode="collect",
                    to_play=args.to_play,
                    collect_temperature=args.collect_temperature,
                    collect_epsilon=args.collect_epsilon,
                )
            )

    primary_debug = eval_debug or collect_debug_rows[0]
    action = int(primary_debug["action"])
    row: dict[str, Any] = {
        "row": row_index,
        "checkpoint": checkpoint_label,
        "seed": episode_seed,
        "episode": episode,
        "seat": "player_0",
        "observation": asdict(observation),
        "action": action,
        "action_label": ACTION_LABELS[action],
        "opponent_policy": args.opponent_policy,
        "opponent_action": opponent_action,
        "decision_mode": args.decision_mode,
        "feature_mode": args.feature_mode,
        "num_simulations": args.num_simulations,
        "action_mask": [1, 1, 1],
        "legal_actions": [0, 1, 2],
        "to_play": [args.to_play],
    }
    if eval_debug is not None:
        row.update(
            {
                "eval_action": eval_debug["action"],
                "eval_action_label": eval_debug["action_label"],
                "policy_logits": eval_debug.get("policy_logits"),
                "visit_counts": eval_debug.get("visit_counts"),
                "searched_value": eval_debug.get("searched_value"),
                "predicted_value": eval_debug.get("predicted_value"),
            }
        )
    if collect_debug_rows:
        collect_actions = [int(item["action"]) for item in collect_debug_rows]
        row["collect_actions"] = collect_actions
        row["collect_action_labels"] = [ACTION_LABELS[action] for action in collect_actions]
        row["collect_action_histogram"] = [
            sum(1 for action in collect_actions if action == action_id)
            for action_id in range(len(ACTION_LABELS))
        ]
        row["collect_decisions"] = collect_debug_rows
    if scoreability_group is not None:
        row["scoreability"] = scoreability_group
    return row


def _run_policy_decision(
    spec_policy: Any,
    *,
    observation: Any,
    raster: Any,
    feature_mode: str,
    max_env_step: int,
    mode: str,
    to_play: int,
    collect_temperature: float,
    collect_epsilon: float,
) -> dict[str, Any]:
    import numpy as np

    from curvyzero.training.dummy_pong import ACTION_LABELS, PongConfig
    from curvyzero.training.lightzero_dummy_pong_features import encode_lightzero_observation

    torch_module = spec_policy._torch
    encoded = encode_lightzero_observation(
        feature_mode=feature_mode,
        observation=observation,
        config=PongConfig(max_steps=max_env_step),
        raster_grid=raster,
    )
    obs_tensor = torch_module.as_tensor(encoded[None, :], dtype=torch_module.float32)
    action_mask = np.ones((1, len(ACTION_LABELS)), dtype=np.float32)
    with torch_module.no_grad():
        if mode == "eval":
            output = spec_policy._policy.eval_mode.forward(
                obs_tensor,
                action_mask=action_mask,
                to_play=[to_play],
                ready_env_id=np.asarray([0]),
            )
        elif mode == "collect":
            output = spec_policy._policy.collect_mode.forward(
                obs_tensor,
                action_mask=action_mask,
                temperature=collect_temperature,
                to_play=[to_play],
                epsilon=collect_epsilon,
                ready_env_id=np.asarray([0]),
            )
        else:
            raise ValueError(f"unknown decision mode {mode!r}")
    debug = _extract_forward_debug(output)
    action = _extract_forward_action(output)
    return {
        "mode": mode,
        "action": action,
        "action_label": ACTION_LABELS[action],
        "feature_vector": _to_jsonable(encoded),
        "action_mask": action_mask.tolist(),
        "to_play": [to_play],
        **debug,
    }


def _extract_forward_action(output: Any) -> int:
    payload = _unwrap_single_env_output(output)
    value = _lookup_output_key(payload, "action")
    if value is None:
        value = _lookup_output_key(payload, "selected_action")
    if value is None:
        raise ValueError(f"could not extract action from policy output: {output!r}")
    return int(_to_jsonable(value))


def _scoreability_debug_states(args: argparse.Namespace, *, limit: int) -> list[dict[str, Any]]:
    if args.scoreability_summary is None:
        return []
    summary = _read_json(args.scoreability_summary)
    groups = summary.get("group_summaries", [])
    if not isinstance(groups, list):
        raise ValueError(f"scoreability summary has no group_summaries: {args.scoreability_summary}")
    best_action = _parse_action_filter(args.scoreability_best_action)
    states = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        if str(group.get("opponent_policy")) != args.opponent_policy:
            continue
        pressure_agent = str(group.get("pressure_agent"))
        if (
            args.pong_reset_pressure_agent != "random"
            and pressure_agent != args.pong_reset_pressure_agent
        ):
            continue
        best_actions = [int(action) for action in group.get("best_ego_actions_by_score_then_survival", [])]
        if best_action is not None and best_action not in best_actions:
            continue
        if args.scoreability_require_unique_best and len(best_actions) != 1:
            continue
        states.append(
            {
                "reset_state_id": group.get("reset_state_id"),
                "reset_seed": group.get("reset_seed"),
                "pressure_agent": pressure_agent,
                "opponent_policy": group.get("opponent_policy"),
                "best_ego_actions_by_score_then_survival": best_actions,
                "candidate_results": group.get("candidate_results"),
            }
        )
        if len(states) >= limit:
            break
    if not states:
        raise ValueError(
            "scoreability summary filter matched no groups; check opponent, pressure agent, "
            "and best-action filters"
        )
    return states


def _parse_action_filter(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    labels = {label: index for index, label in enumerate(DEFAULT_ACTION_LABELS)}
    if normalized in labels:
        return labels[normalized]
    try:
        action_id = int(normalized)
    except ValueError as exc:
        raise ValueError(f"unknown action filter {value!r}; expected up/stay/down or 0/1/2") from exc
    if action_id < 0 or action_id >= len(DEFAULT_ACTION_LABELS):
        raise ValueError(f"action filter out of range: {value!r}")
    return action_id


def _extract_forward_debug(output: Any) -> dict[str, Any]:
    payload = _unwrap_single_env_output(output)
    result: dict[str, Any] = {}
    for output_key, result_key in (
        ("action", "adapter_action"),
        ("selected_action", "adapter_action"),
        ("policy_logits", "policy_logits"),
        ("predicted_policy_logits", "policy_logits"),
        ("logits", "policy_logits"),
        ("visit_count_distributions", "visit_counts"),
        ("visit_counts", "visit_counts"),
        ("visit_count_distribution", "visit_counts"),
        ("searched_value", "searched_value"),
        ("predicted_value", "predicted_value"),
        ("pred_value", "predicted_value"),
        ("value", "value"),
    ):
        value = _lookup_output_key(payload, output_key)
        if value is not None and result_key not in result:
            result[result_key] = _to_jsonable(value)
    if not result:
        result["adapter_forward_output_type"] = type(output).__name__
    return result


def _unwrap_single_env_output(output: Any) -> Any:
    if isinstance(output, dict):
        for key in (0, "0"):
            if key in output:
                return _unwrap_single_env_output(output[key])
    if isinstance(output, (list, tuple)) and len(output) == 1:
        return _unwrap_single_env_output(output[0])
    return output


def _lookup_output_key(output: Any, key: str) -> Any:
    if isinstance(output, dict) and key in output:
        return output[key]
    if hasattr(output, key):
        return getattr(output, key)
    return None


def _render_debug_rows(rows: list[dict[str, Any]], *, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(rows, indent=2, sort_keys=True)
    if output_format == "jsonl":
        return "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    columns = ["row", "seed", "seat", "obs", "logits", "visits", "action"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        obs = row["observation"]
        compact_obs = (
            f"step={obs['step']} dx={obs['ball_dx_forward']} "
            f"dy={obs['ball_dy_from_ego_center']} vx={obs['ball_vx_forward']} "
            f"vy={obs['ball_vy']}"
        )
        display = {
            "row": str(row["row"]),
            "seed": str(row["seed"]),
            "seat": str(row["seat"]),
            "obs": compact_obs,
            "logits": _fmt_debug_list(row.get("policy_logits")),
            "visits": _fmt_debug_list(row.get("visit_counts")),
            "action": f"{row['action']}:{row['action_label']}",
        }
        lines.append("| " + " | ".join(_escape_md(display[column]) for column in columns) + " |")
    return "\n".join(lines)


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _source_label(source: str, resolved: Path) -> str:
    if source.startswith(("ref:", "volume:")):
        return source
    try:
        return str(resolved)
    except ValueError:
        return source


def _seating_label(summary: dict[str, Any]) -> str:
    eval_split = _dict(summary.get("eval_split"))
    if "paired_seat" in eval_split:
        return "paired" if bool(eval_split["paired_seat"]) else "player0-only"
    if "paired_seats" in summary:
        return "paired" if bool(summary["paired_seats"]) else "player0-only"
    eval_seating = _dict(summary.get("eval_seating"))
    mode = str(eval_seating.get("mode", ""))
    if "player0" in mode:
        return "player0-only"
    if "paired" in mode:
        return "paired"
    return "unknown"


def _profile_label(summary: dict[str, Any]) -> str:
    eval_split = _dict(summary.get("eval_split"))
    config = _dict(summary.get("config_surface"))
    command = _dict(summary.get("command"))
    lightzero_eval_config = _dict(summary.get("lightzero_eval_config"))
    feature = (
        eval_split.get("feature_mode")
        or summary.get("feature_mode")
        or lightzero_eval_config.get("feature_mode")
        or config.get("feature_mode")
        or command.get("feature_mode")
        or _first_checkpoint_field(summary, "feature_mode")
    )
    reset = (
        eval_split.get("pong_reset_profile")
        or summary.get("pong_reset_profile")
        or lightzero_eval_config.get("pong_reset_profile")
        or config.get("pong_reset_profile")
        or command.get("pong_reset_profile")
        or "default"
    )
    pressure_agent = (
        eval_split.get("pong_reset_pressure_agent")
        or summary.get("pong_reset_pressure_agent")
        or lightzero_eval_config.get("pong_reset_pressure_agent")
        or config.get("pong_reset_pressure_agent")
        or command.get("pong_reset_pressure_agent")
    )
    parts = [str(feature or "unknown"), str(reset or "default")]
    if reset and reset != "default" and pressure_agent:
        parts[-1] = f"{parts[-1]}:{pressure_agent}"
    return "/".join(parts)


def _first_checkpoint_field(summary: dict[str, Any], key: str) -> Any:
    for container_key in ("checkpoint_specs", "checkpoint_policies"):
        items = summary.get(container_key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and item.get(key) is not None:
                return item[key]
    return None


def _opponent_label(policies: list[str], subject_policy: str) -> str:
    opponents = [policy for policy in policies if policy != subject_policy]
    if not opponents:
        return "self"
    return ",".join(opponents)


def _policies_from_pair_group(pair_group_id: str) -> list[str]:
    if "_vs_" not in pair_group_id:
        return [pair_group_id]
    left, right = pair_group_id.rsplit("_vs_", 1)
    return [left] if left == right else [left, right]


def _hist_from_label_counts(counts: dict[str, Any]) -> list[int]:
    if not counts:
        return []
    return [_int_or_zero(counts.get(label)) for label in DEFAULT_ACTION_LABELS]


def _format_histogram(histogram: list[int], *, action_labels: tuple[str, ...]) -> str:
    if not histogram:
        return ""
    parts = []
    for index, count in enumerate(histogram):
        label = action_labels[index] if index < len(action_labels) else f"a{index}"
        parts.append(f"{label}={count}")
    return " ".join(parts)


def _normalized_entropy(histogram: list[int]) -> float | None:
    total = sum(histogram)
    if total <= 0 or len(histogram) <= 1:
        return None
    entropy = 0.0
    for count in histogram:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(len(histogram))


def _fmt_debug_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        flat = value[0] if len(value) == 1 and isinstance(value[0], list) else value
        return "[" + ",".join(_fmt_value(item) for item in flat) + "]"
    return str(value)


def _fmt_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _stat(stats: dict[str, Any] | None, key: str) -> Any:
    if not stats:
        return None
    return stats.get(key)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    parsed = _int_or_none(value)
    return 0 if parsed is None else parsed


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [_int_or_zero(item) for item in value]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_label(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


if __name__ == "__main__":
    main()
