"""Local CurvyTron run inspector.

This module is intentionally read-only. It turns an eval manifest and an
optional train summary into a compact report that is easy to critique.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPORT_SCHEMA_ID = "curvyzero_curvytron_inspector_report/v0"
DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron-inspector")
DEFAULT_COLLAPSE_THRESHOLD = 0.95
DEFAULT_MIN_MEANINGFUL_DELTA_STEPS = 5.0
DEFAULT_MIN_MEANINGFUL_DELTA_FRACTION = 0.05
DEFAULT_HEAVY_CAPPED_FRACTION = 0.5

_ROUND_OUTCOME_REASONS = {
    "none",
    "missing",
    "round_all_dead_draw",
    "round_survivor_win",
    "survivor_win",
    "unknown",
}
_DEATH_CAUSE_KEYS = (
    "death_cause_name",
    "death_cause",
    "death_reason",
    "collision_kind",
    "hit_kind",
    "death_hit_owner",
    "terminal_detail",
)


def load_json_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {resolved}")
    return value


def build_inspector_report(
    manifest: dict[str, Any],
    *,
    eval_manifest_path: str | Path | None = None,
    train_summary: dict[str, Any] | None = None,
    train_summary_path: str | Path | None = None,
    collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
    min_meaningful_delta_steps: float = DEFAULT_MIN_MEANINGFUL_DELTA_STEPS,
    min_meaningful_delta_fraction: float = DEFAULT_MIN_MEANINGFUL_DELTA_FRACTION,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a plain evidence report from one eval manifest."""

    config = _dict_or_empty(manifest.get("config"))
    selection = _dict_or_empty(manifest.get("selection"))
    table = _list_of_dicts(manifest.get("table"))
    aggregate_table = _list_of_dicts(manifest.get("survival_aggregate_table"))

    checkpoint_rows = _checkpoint_summaries(
        table,
        aggregate_table=aggregate_table,
        collapse_threshold=collapse_threshold,
    )
    replay_inspection = _local_replay_inspection(
        table=table,
        eval_manifest_path=eval_manifest_path,
    )
    comparability_blockers = _comparability_blockers(table, checkpoint_rows)
    paired_first_latest = _paired_first_latest_survival_delta(table)
    warnings = _warnings(
        manifest=manifest,
        train_summary=train_summary,
        checkpoint_rows=checkpoint_rows,
        table=table,
        config=config,
        collapse_threshold=collapse_threshold,
        replay_inspection=replay_inspection,
    )
    verdict = _survival_verdict(
        manifest=manifest,
        config=config,
        train_summary=train_summary,
        checkpoint_rows=checkpoint_rows,
        replay_inspection=replay_inspection,
        warnings=warnings,
        comparability_blockers=comparability_blockers,
        paired_first_latest_survival_delta=paired_first_latest,
        min_meaningful_delta_steps=min_meaningful_delta_steps,
        min_meaningful_delta_fraction=min_meaningful_delta_fraction,
    )

    run_id = _first_nonempty(config.get("run_id"), manifest.get("run_id"))
    attempt_id = _first_nonempty(config.get("attempt_id"), manifest.get("attempt_id"))
    eval_id = _first_nonempty(manifest.get("eval_id"), config.get("eval_id"), "unknown_eval")
    train_observability = _train_action_observability(train_summary)

    return {
        "schema_id": REPORT_SCHEMA_ID,
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "inputs": {
            "eval_manifest_path": str(eval_manifest_path) if eval_manifest_path is not None else None,
            "train_summary_path": str(train_summary_path) if train_summary_path is not None else None,
        },
        "run": {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "eval_id": eval_id,
            "manifest_schema": manifest.get("schema"),
            "manifest_ok": manifest.get("ok"),
            "job_kind": manifest.get("job_kind"),
            "compute": config.get("compute"),
            "opponent_policy_kind": _first_nonempty(
                config.get("opponent_policy_kind"),
                selection.get("opponent_policy_kind"),
            ),
            "opponent_comparison": _opponent_comparison_label(config, train_summary),
            "opponent_checkpoint_ref": _first_nonempty(
                config.get("opponent_checkpoint_ref"),
                selection.get("opponent_checkpoint_ref"),
            ),
            "opponent_snapshot_ref": _first_nonempty(
                config.get("opponent_snapshot_ref"),
                selection.get("opponent_snapshot_ref"),
            ),
            "max_eval_steps": config.get("max_eval_steps"),
            "source_max_steps": config.get("source_max_steps"),
            "num_simulations": config.get("num_simulations"),
            "batch_size": config.get("batch_size"),
            "step_detail_limit": config.get("step_detail_limit"),
            "selected_iterations": _selected_iterations(selection.get("selected_iterations")),
            "eval_seed_count": selection.get("eval_seed_count"),
            "eval_seed_sampler_seed": selection.get("eval_seed_sampler_seed"),
        },
        "train_summary": _train_summary_block(train_summary),
        "train_action_observability": train_observability,
        "eval": {
            "checkpoint_count": len(checkpoint_rows),
            "row_count": len(table),
            "unique_seed_count": _unique_seed_count(table),
            "seed_sets_match": _seed_sets_match(table),
            "strict_comparable": not comparability_blockers,
            "comparability_blockers": comparability_blockers,
            "paired_first_latest_survival_delta": paired_first_latest,
            "checkpoints": checkpoint_rows,
        },
        "replay_inspection": replay_inspection,
        "verdict": verdict,
        "warnings": warnings,
        "questions_next": _questions_next(warnings=warnings, verdict=verdict),
    }


def write_report_files(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    resolved = Path(output_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    json_path = resolved / "report.json"
    md_path = resolved / "report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def output_dir_for_report(report: dict[str, Any], output_root: str | Path) -> Path:
    run = _dict_or_empty(report.get("run"))
    run_id = _clean_path_part(_first_nonempty(run.get("run_id"), "unknown_run"))
    eval_id = _clean_path_part(_first_nonempty(run.get("eval_id"), "unknown_eval"))
    return Path(output_root) / run_id / eval_id


def render_markdown_report(report: dict[str, Any]) -> str:
    run = _dict_or_empty(report.get("run"))
    verdict = _dict_or_empty(report.get("verdict"))
    warnings = [str(item) for item in report.get("warnings", []) if item]
    checkpoints = _list_of_dicts(_dict_or_empty(report.get("eval")).get("checkpoints"))
    train_summary = _dict_or_empty(report.get("train_summary"))
    replay = _dict_or_empty(report.get("replay_inspection"))

    lines = [
        f"# CurvyTron Inspector Report - {run.get('eval_id') or 'unknown_eval'}",
        "",
        "## Verdict",
        "",
        f"- Survival read: `{verdict.get('survival_read')}`",
        f"- Learning claim: `{verdict.get('learning_claim')}`",
        f"- Claim blockers: {_compact_list(verdict.get('claim_blockers'))}",
        f"- Coach next move: {verdict.get('coach_next_move')}",
        f"- Plain read: {verdict.get('plain_read')}",
        f"- Action collapse seen: `{verdict.get('action_collapse_seen')}`",
        "",
        "## Run",
        "",
        f"- Run id: `{run.get('run_id')}`",
        f"- Attempt id: `{run.get('attempt_id')}`",
        f"- Opponent: `{run.get('opponent_policy_kind')}`",
        f"- Opponent comparison: `{run.get('opponent_comparison')}`",
        f"- Max eval steps: `{run.get('max_eval_steps')}`",
        f"- Source max steps: `{run.get('source_max_steps')}`",
        f"- Simulations: `{run.get('num_simulations')}`",
        f"- Batch size: `{run.get('batch_size')}`",
    ]
    if train_summary:
        lines.extend(
            [
                f"- Current-policy self-play: `{train_summary.get('current_policy_self_play')}`",
                f"- Opponent training relation: `{train_summary.get('opponent_training_relation')}`",
                f"- Debug fidelity only: `{train_summary.get('debug_fidelity_only')}`",
                f"- Learning proof: `{train_summary.get('learning_proof')}`",
            ]
        )

    if replay:
        lines.extend(
            [
                "",
                "## Replay Death Read",
                "",
                f"- Status: `{replay.get('status')}`",
                f"- Local artifacts checked: `{replay.get('local_artifact_count')}`",
                f"- Replayed: `{replay.get('replayed_count')}`",
                f"- Trace hash matches: `{replay.get('trace_hash_match_count')}`",
                f"- Death causes: {_compact_counts(replay.get('death_cause_counts'))}",
                f"- First death players: {_compact_counts(replay.get('first_death_player_counts'))}",
                f"- Replay failures: {_compact_counts(replay.get('failure_reasons'))}",
            ]
        )
        shortest = _list_of_dicts(replay.get("shortest_deaths"))[:5]
        if shortest:
            lines.extend(
                [
                    "",
                    "| steps | player | cause | actions | artifact |",
                    "| ---: | --- | --- | --- | --- |",
                ]
            )
            for row in shortest:
                lines.append(
                    "| {steps} | {player} | {cause} | {actions} | `{artifact}` |".format(
                        steps=_format_md(row.get("steps")),
                        player=_format_md(row.get("player")),
                        cause=_format_md(row.get("cause")),
                        actions=_format_md(row.get("actions")),
                        artifact=_format_md(row.get("artifact")),
                    )
                )

    lines.extend(
        [
            "",
            "## Survival Curve",
            "",
            "| checkpoint | rows | unique seeds | trusted rows | mean | median | min | max | top action frac | terminal reasons | death causes |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in checkpoints:
        lines.append(
            "| {checkpoint} | {rows} | {seeds} | {trusted} | {mean} | {median} | {min_} | {max_} | {top_frac} | {terminal} | {death} |".format(
                checkpoint=row.get("checkpoint"),
                rows=_format_md(row.get("row_count")),
                seeds=_format_md(row.get("unique_seed_count")),
                trusted=_format_md(row.get("trusted_row_count")),
                mean=_format_md(row.get("mean_steps")),
                median=_format_md(row.get("median_steps")),
                min_=_format_md(row.get("min_steps")),
                max_=_format_md(row.get("max_steps")),
                top_frac=_format_md(row.get("top_action_fraction")),
                terminal=_compact_counts(row.get("terminal_counts")),
                death=_compact_counts(row.get("death_cause_counts")),
            )
        )

    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None.")

    lines.extend(["", "## Next Questions", ""])
    questions = [str(item) for item in report.get("questions_next", []) if item]
    if questions:
        lines.extend(f"- {question}" for question in questions)
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def _checkpoint_summaries(
    table: list[dict[str, Any]],
    *,
    aggregate_table: list[dict[str, Any]],
    collapse_threshold: float,
) -> list[dict[str, Any]]:
    if not table:
        return _checkpoint_summaries_from_aggregate(
            aggregate_table,
            collapse_threshold=collapse_threshold,
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in table:
        checkpoint = str(_first_nonempty(row.get("checkpoint_label"), row.get("checkpoint"), "missing"))
        grouped.setdefault(checkpoint, []).append(row)

    summaries = []
    for checkpoint, rows in sorted(grouped.items(), key=lambda item: _checkpoint_sort_key(item[0])):
        primary_rows = [row for row in rows if _row_can_drive_primary_survival(row)]
        steps = [
            _safe_float(_first_nonempty(row.get("steps_survived"), row.get("steps")))
            for row in primary_rows
        ]
        steps = [value for value in steps if value is not None]
        seeds = sorted(
            {
                int(seed)
                for seed in (_safe_int(row.get("seed")) for row in rows)
                if seed is not None
            }
        )
        caps = sorted(
            {
                value
                for value in (_safe_float(row.get("cap")) for row in rows)
                if value is not None
            }
        )
        ok_values = [
            parsed
            for row in rows
            if (parsed := _safe_bool(row.get("ok"))) is not None
        ]
        strict_values = [
            _safe_bool(_first_nonempty(row.get("strict_load"), row.get("strict")))
            for row in rows
        ]
        strict_known_values = [value for value in strict_values if value is not None]
        action_counts = _sum_action_counts(rows)
        action_summary = _action_summary(action_counts, collapse_threshold=collapse_threshold)
        row_action_summaries = [
            _action_summary(
                _normalise_counts(_first_nonempty(row.get("action_histogram"), row.get("actions"))),
                collapse_threshold=collapse_threshold,
            )
            for row in rows
        ]
        row_top_fractions = [
            summary["top_action_fraction"]
            for summary in row_action_summaries
            if summary["top_action_fraction"] is not None
        ]
        terminal_counts = Counter(
            str(_first_nonempty(row.get("terminal_reason"), row.get("terminal"), "missing"))
            for row in rows
        )
        death_cause_counts = _death_cause_counts(rows)
        capped_count = sum(1 for row in rows if _row_is_capped(row))
        capped_fraction = capped_count / len(rows) if rows else None
        artifact_refs = [
            str(ref)
            for ref in (
                _first_nonempty(row.get("artifact_ref"), row.get("artifact"), row.get("output_ref"))
                for row in rows
            )
            if ref
        ]
        checkpoint_refs = sorted(
            {
                str(ref)
                for ref in (_first_nonempty(row.get("checkpoint_ref"), row.get("checkpoint_path")) for row in rows)
                if ref
            }
        )
        summaries.append(
            {
                "checkpoint": checkpoint,
                "row_count": len(rows),
                "seed_count": len(rows),
                "unique_seed_count": len(seeds),
                "trusted_row_count": len(primary_rows),
                "seeds": seeds,
                "mean_steps": _mean(steps),
                "median_steps": _median(steps),
                "min_steps": min(steps) if steps else None,
                "max_steps": max(steps) if steps else None,
                "caps": caps,
                "ok_count": sum(ok_values) if ok_values else None,
                "failure_count": len(ok_values) - sum(ok_values) if ok_values else None,
                "capped_count": capped_count,
                "capped_fraction": capped_fraction,
                "strict_load_all": (
                    all(value is True for value in strict_values) if strict_known_values else None
                ),
                "strict_load_missing_count": sum(1 for value in strict_values if value is None),
                "action_counts": action_counts,
                **action_summary,
                "max_row_top_action_fraction": max(row_top_fractions) if row_top_fractions else None,
                "row_action_collapsed_count": sum(
                    1 for summary in row_action_summaries if summary.get("action_collapsed") is True
                ),
                "terminal_counts": dict(sorted(terminal_counts.items())),
                "death_cause_counts": dict(sorted(death_cause_counts.items())),
                "death_cause_available": bool(death_cause_counts),
                "checkpoint_refs": checkpoint_refs[:3],
                "artifact_refs": artifact_refs[:3],
            }
        )
    return summaries


def _checkpoint_summaries_from_aggregate(
    aggregate_table: list[dict[str, Any]],
    *,
    collapse_threshold: float,
) -> list[dict[str, Any]]:
    summaries = []
    for row in sorted(
        aggregate_table,
        key=lambda item: _checkpoint_sort_key(str(_first_nonempty(item.get("checkpoint"), "missing"))),
    ):
        ok_count = _first_nonempty(row.get("ok_count"), row.get("ok"))
        seeds = _safe_int(row.get("seeds"))
        ok_count_int = _safe_int(ok_count)
        summaries.append(
            {
                "checkpoint": str(_first_nonempty(row.get("checkpoint"), "missing")),
                "row_count": seeds,
                "seed_count": seeds,
                "unique_seed_count": seeds,
                "trusted_row_count": None,
                "seeds": [],
                "mean_steps": _safe_float(row.get("mean_steps")),
                "median_steps": _safe_float(row.get("median_steps")),
                "min_steps": _safe_float(row.get("min_steps")),
                "max_steps": _safe_float(row.get("max_steps")),
                "caps": [],
                "ok_count": ok_count_int,
                "failure_count": (
                    seeds - ok_count_int
                    if seeds is not None and ok_count_int is not None
                    else _safe_int(row.get("failure_count"))
                ),
                "capped_count": _safe_int(row.get("capped_count")),
                "capped_fraction": None,
                "strict_load_all": None,
                "strict_load_missing_count": None,
                "action_counts": {},
                **_action_summary({}, collapse_threshold=collapse_threshold),
                "max_row_top_action_fraction": None,
                "row_action_collapsed_count": 0,
                "terminal_counts": {},
                "death_cause_counts": {},
                "death_cause_available": False,
                "checkpoint_refs": [],
                "artifact_refs": [],
            }
        )
    return summaries


def _local_replay_inspection(
    *,
    table: list[dict[str, Any]],
    eval_manifest_path: str | Path | None,
) -> dict[str, Any]:
    base = {
        "status": "not_run",
        "local_artifact_count": 0,
        "replayed_count": 0,
        "failed_count": 0,
        "trace_hash_match_count": 0,
        "trace_hash_mismatch_count": 0,
        "death_cause_counts": {},
        "first_death_player_counts": {},
        "failure_reasons": {},
        "shortest_deaths": [],
    }
    if eval_manifest_path is None or not table:
        return base

    artifact_paths = _local_episode_artifact_paths(
        table=table,
        eval_manifest_path=Path(eval_manifest_path),
    )
    base["local_artifact_count"] = len(artifact_paths)
    if not artifact_paths:
        base["status"] = "no_local_artifacts"
        return base

    from curvyzero.training.curvytron_visual_survival_replay_inspector import (
        inspect_episode_artifact,
    )

    death_cause_counts: Counter[str] = Counter()
    death_player_counts: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    shortest: list[dict[str, Any]] = []
    replayed_count = 0
    hash_match_count = 0
    hash_mismatch_count = 0

    for artifact_path in artifact_paths:
        result = inspect_episode_artifact(artifact_path)
        if not result.get("ok"):
            failure_reasons[str(_first_nonempty(result.get("reason"), "unknown"))] += 1
            continue
        replayed_count += 1
        if result.get("trace_hash_match") is True:
            hash_match_count += 1
        else:
            hash_mismatch_count += 1
        first_death = _dict_or_empty(result.get("first_death"))
        cause = str(_first_nonempty(first_death.get("cause_name"), "unknown"))
        player = str(_first_nonempty(first_death.get("player_id"), "unknown"))
        death_cause_counts[cause] += 1
        death_player_counts[player] += 1
        shortest.append(
            {
                "steps": result.get("actions_replayed"),
                "player": player,
                "cause": cause,
                "actions": result.get("action_read"),
                "trace_hash_match": result.get("trace_hash_match"),
                "artifact": Path(str(result.get("artifact_path"))).parent.name,
            }
        )

    base.update(
        {
            "status": "ok" if replayed_count else "no_replayable_artifacts",
            "replayed_count": replayed_count,
            "failed_count": len(artifact_paths) - replayed_count,
            "trace_hash_match_count": hash_match_count,
            "trace_hash_mismatch_count": hash_mismatch_count,
            "death_cause_counts": dict(sorted(death_cause_counts.items())),
            "first_death_player_counts": dict(sorted(death_player_counts.items())),
            "failure_reasons": dict(sorted(failure_reasons.items())),
            "shortest_deaths": sorted(
                shortest,
                key=lambda row: (
                    _safe_float(row.get("steps")) is None,
                    _safe_float(row.get("steps")) or math.inf,
                ),
            )[:10],
        }
    )
    return base


def _local_episode_artifact_paths(
    *,
    table: list[dict[str, Any]],
    eval_manifest_path: Path,
) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for row in table:
        ref = _first_nonempty(row.get("artifact_ref"), row.get("artifact"), row.get("output_ref"))
        if not ref:
            continue
        path = _local_episode_artifact_path(str(ref), eval_manifest_path=eval_manifest_path)
        if path is None or not path.exists():
            continue
        resolved_key = path.as_posix()
        if resolved_key in seen:
            continue
        seen.add(resolved_key)
        paths.append(path)
    return paths


def _local_episode_artifact_path(
    artifact_ref: str,
    *,
    eval_manifest_path: Path,
) -> Path | None:
    direct = Path(artifact_ref)
    if direct.exists():
        return direct
    marker = "/eval/"
    if marker not in artifact_ref:
        return None
    tail = artifact_ref.split(marker, 1)[1].lstrip("/")
    direct_tail = eval_manifest_path.parent / tail
    if direct_tail.exists():
        return direct_tail
    eval_dir = eval_manifest_path.stem
    if eval_dir.endswith("_manifest"):
        eval_dir = eval_dir[: -len("_manifest")]
    nested_tail = eval_manifest_path.parent / eval_dir / tail
    if nested_tail.exists():
        return nested_tail
    return direct_tail


def _replay_has_death_cause(replay_inspection: dict[str, Any]) -> bool:
    counts = _dict_or_empty(replay_inspection.get("death_cause_counts"))
    return any(str(cause) not in {"", "none", "unknown"} for cause in counts)


def _comparability_blockers(
    table: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if table and _seed_sets_match(table) is False:
        blockers.append("seed_counter_mismatch")

    caps = sorted({cap for row in checkpoint_rows for cap in row.get("caps", [])})
    if len(caps) > 1:
        blockers.append("mixed_caps")

    if any(_safe_int(row.get("failure_count")) for row in checkpoint_rows):
        blockers.append("failed_eval_rows")

    if table and any(row.get("strict_load_all") is not True for row in checkpoint_rows):
        blockers.append("non_strict_or_missing_checkpoint_load")

    if any(
        (fraction := _safe_float(row.get("capped_fraction"))) is not None
        and fraction >= DEFAULT_HEAVY_CAPPED_FRACTION
        for row in checkpoint_rows
    ):
        blockers.append("heavy_capped_rows")

    return blockers


def _paired_first_latest_survival_delta(table: list[dict[str, Any]]) -> dict[str, Any] | None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in table:
        checkpoint = str(_first_nonempty(row.get("checkpoint_label"), row.get("checkpoint"), "missing"))
        grouped.setdefault(checkpoint, []).append(row)
    if len(grouped) < 2:
        return None

    labels = sorted(grouped, key=_checkpoint_sort_key)
    first_label = labels[0]
    latest_label = labels[-1]
    first_rows = [row for row in grouped[first_label] if _row_can_drive_primary_survival(row)]
    latest_rows = [row for row in grouped[latest_label] if _row_can_drive_primary_survival(row)]
    first_steps = _mean_steps_by_seed(first_rows)
    latest_steps = _mean_steps_by_seed(latest_rows)
    seed_counters = _seed_counters_by_checkpoint(first_rows + latest_rows)
    same_seed_counter = seed_counters.get(first_label) == seed_counters.get(latest_label)
    common_seeds = sorted(set(first_steps).intersection(latest_steps))
    seed_deltas = [
        {
            "seed": seed,
            "first_steps": first_steps[seed],
            "latest_steps": latest_steps[seed],
            "delta_steps": latest_steps[seed] - first_steps[seed],
        }
        for seed in common_seeds
    ]
    deltas = [row["delta_steps"] for row in seed_deltas]

    return {
        "first_checkpoint": first_label,
        "latest_checkpoint": latest_label,
        "first_seed_count": len(first_steps),
        "latest_seed_count": len(latest_steps),
        "common_seed_count": len(common_seeds),
        "same_seed_set": same_seed_counter and bool(common_seeds),
        "mean_delta_steps": _mean(deltas),
        "seed_deltas": seed_deltas,
    }


def _mean_steps_by_seed(rows: list[dict[str, Any]]) -> dict[int, float]:
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        seed = _safe_int(row.get("seed"))
        steps = _safe_float(_first_nonempty(row.get("steps_survived"), row.get("steps")))
        if seed is None or steps is None:
            continue
        by_seed.setdefault(seed, []).append(steps)
    return {
        seed: mean
        for seed, values in sorted(by_seed.items())
        if (mean := _mean(values)) is not None
    }


def _warnings(
    *,
    manifest: dict[str, Any],
    train_summary: dict[str, Any] | None,
    checkpoint_rows: list[dict[str, Any]],
    table: list[dict[str, Any]],
    config: dict[str, Any],
    collapse_threshold: float,
    replay_inspection: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if not checkpoint_rows:
        warnings.append("No checkpoint rows were found, so the inspector cannot read survival.")
    if not table:
        warnings.append("The manifest has no detailed table rows; the report may only use aggregates.")
    if train_summary is None:
        warnings.append("No train summary was provided, so training path and debug-fidelity claims are unknown.")
    else:
        manifest_run_id = _first_nonempty(config.get("run_id"), manifest.get("run_id"))
        train_run_id = train_summary.get("run_id")
        if manifest_run_id and train_run_id and manifest_run_id != train_run_id:
            warnings.append("Eval manifest run_id does not match train summary run_id.")
        if train_summary.get("debug_fidelity_only") is True:
            warnings.append("Train summary says debug_fidelity_only=true; do not treat this as proof of learning.")
        if train_summary.get("learning_proof") is False:
            warnings.append("Train summary says learning_proof=false.")
        if train_summary.get("current_policy_self_play") is False:
            warnings.append("Train summary says current_policy_self_play=false.")

    if manifest.get("ok") is False:
        warnings.append("Manifest ok=false.")

    collapsed = [
        row
        for row in checkpoint_rows
        if row.get("top_action_fraction") is not None
        and float(row["top_action_fraction"]) >= collapse_threshold
    ]
    if collapsed:
        labels = ", ".join(str(row.get("checkpoint")) for row in collapsed[:5])
        warnings.append(f"Action collapse at threshold {collapse_threshold:g} for: {labels}.")
    row_collapsed = [
        row
        for row in checkpoint_rows
        if (_safe_int(row.get("row_action_collapsed_count")) or 0) > 0
    ]
    if row_collapsed:
        labels = ", ".join(str(row.get("checkpoint")) for row in row_collapsed[:5])
        warnings.append(f"Per-row action collapse at threshold {collapse_threshold:g} for: {labels}.")

    if any(row.get("strict_load_all") is False for row in checkpoint_rows):
        warnings.append("At least one eval row did not strict-load its checkpoint.")
    if any(row.get("strict_load_all") is None for row in checkpoint_rows) or any(
        (_safe_int(row.get("strict_load_missing_count")) or 0) > 0 for row in checkpoint_rows
    ):
        warnings.append("Strict checkpoint load status is missing for at least one checkpoint.")
    if any((_safe_int(row.get("failure_count")) or 0) > 0 for row in checkpoint_rows):
        warnings.append("Failed eval rows are present and are excluded from primary survival.")

    caps = sorted({cap for row in checkpoint_rows for cap in row.get("caps", [])})
    if len(caps) > 1:
        warnings.append("Eval rows use mixed caps, so survival comparisons may be unfair.")
    heavy_capped = [
        row
        for row in checkpoint_rows
        if (fraction := _safe_float(row.get("capped_fraction"))) is not None
        and fraction >= DEFAULT_HEAVY_CAPPED_FRACTION
    ]
    if heavy_capped:
        labels = ", ".join(str(row.get("checkpoint")) for row in heavy_capped[:5])
        warnings.append(
            "At least half of rows are capped/right-censored for: "
            f"{labels}; survival lift is not trustworthy."
        )

    if table and _seed_sets_match(table) is False:
        warnings.append("Checkpoint rows do not share the same eval seed counter.")

    terminal_values = {
        reason
        for row in checkpoint_rows
        for reason in _dict_or_empty(row.get("terminal_counts")).keys()
    }
    if terminal_values and terminal_values.issubset(_ROUND_OUTCOME_REASONS):
        if _replay_has_death_cause(replay_inspection):
            warnings.append(
                "Manifest terminal reasons are only round outcomes, but local action replay "
                "recovered death causes."
            )
        else:
            warnings.append(
                "Terminal reasons are only round outcomes or unknown values; real death cause is missing."
            )
    if not any(row.get("death_cause_available") for row in checkpoint_rows):
        if _replay_has_death_cause(replay_inspection):
            warnings.append(
                "No death-cause fields were found in eval rows; local replay recovered them."
            )
        else:
            warnings.append("No death-cause fields were found in eval rows.")

    opponent_label = _opponent_comparison_label(config, train_summary)
    if opponent_label == "fixed_opponent":
        warnings.append("This is a fixed-opponent eval; it is not self-play learning proof.")
    if opponent_label == "frozen_opponent":
        warnings.append("This is a frozen-opponent eval; it is useful but still a narrow comparison.")

    return _dedupe(warnings)


def _survival_verdict(
    manifest: dict[str, Any],
    config: dict[str, Any],
    train_summary: dict[str, Any] | None,
    checkpoint_rows: list[dict[str, Any]],
    replay_inspection: dict[str, Any],
    *,
    warnings: list[str],
    comparability_blockers: list[str],
    paired_first_latest_survival_delta: dict[str, Any] | None,
    min_meaningful_delta_steps: float,
    min_meaningful_delta_fraction: float,
) -> dict[str, Any]:
    rows_with_mean = [row for row in checkpoint_rows if row.get("mean_steps") is not None]
    action_collapse_seen = any("action collapse" in warning.lower() for warning in warnings)
    claim_blockers = _claim_blockers(
        manifest=manifest,
        config=config,
        train_summary=train_summary,
        checkpoint_rows=checkpoint_rows,
        replay_inspection=replay_inspection,
        comparability_blockers=comparability_blockers,
        action_collapse_seen=action_collapse_seen,
    )
    if len(rows_with_mean) < 2:
        plain_read = "There are not enough checkpoints with mean survival to compare."
        if _replay_has_death_cause(replay_inspection):
            plain_read += (
                " Local replay found death causes: "
                f"{_compact_counts(replay_inspection.get('death_cause_counts'))}."
            )
        coach_next_move = _coach_next_move(
            survival_read="unknown",
            learning_claim="unknown",
            claim_blockers=claim_blockers,
            comparability_blockers=comparability_blockers,
        )
        return {
            "survival_read": "unknown",
            "learning_claim": "unknown",
            "claim_blockers": claim_blockers,
            "coach_next_move": coach_next_move,
            "action_collapse_seen": action_collapse_seen,
            "strict_comparable": not comparability_blockers,
            "comparability_blockers": comparability_blockers,
            "paired_first_latest_survival_delta": paired_first_latest_survival_delta,
            "plain_read": plain_read,
        }

    first = rows_with_mean[0]
    latest = rows_with_mean[-1]
    best = max(rows_with_mean, key=lambda row: float(row.get("mean_steps") or -math.inf))
    first_mean = float(first["mean_steps"])
    latest_mean = float(latest["mean_steps"])
    paired_delta = _safe_float(
        _dict_or_empty(paired_first_latest_survival_delta).get("mean_delta_steps")
    )
    use_paired_delta = (
        paired_delta is not None
        and _dict_or_empty(paired_first_latest_survival_delta).get("same_seed_set") is True
    )
    delta = paired_delta if use_paired_delta else latest_mean - first_mean
    threshold = max(min_meaningful_delta_steps, abs(first_mean) * min_meaningful_delta_fraction)
    if delta >= threshold:
        raw_survival_read = "improved"
    elif delta <= -threshold:
        raw_survival_read = "worse"
    else:
        raw_survival_read = "flat"

    survival_read = "unknown" if comparability_blockers else raw_survival_read

    if comparability_blockers:
        learning_claim = "blocked_by_non_comparable_eval"
    elif survival_read == "improved" and not claim_blockers:
        learning_claim = "survival_improved_in_this_eval_only"
    elif survival_read == "improved":
        learning_claim = "incomplete_due_to_claim_blockers"
    elif survival_read == "flat":
        learning_claim = "no_clear_survival_lift"
    else:
        learning_claim = "survival_got_worse_in_this_eval"

    percent = (delta / first_mean * 100.0) if first_mean else None
    plain_delta = f"{delta:.2f} steps"
    if percent is not None:
        plain_delta += f" ({percent:.1f}%)"
    delta_basis = "paired same-seed" if use_paired_delta else "checkpoint-mean"
    if comparability_blockers:
        blocker_text = ", ".join(comparability_blockers)
        plain_read = (
            f"Latest survival is {plain_delta} versus the first evaluated checkpoint "
            f"by {delta_basis} comparison, but this eval is not strictly comparable "
            f"({blocker_text}), so no learning claim is made."
        )
    else:
        plain_read = (
            f"Latest survival is {plain_delta} versus the first evaluated checkpoint "
            f"by {delta_basis} comparison. The read is {survival_read} for this eval only."
        )
        if raw_survival_read == "improved" and claim_blockers:
            plain_read += (
                " Do not promote this to a learning claim until blockers are cleared: "
                f"{_human_claim_blockers(claim_blockers)}."
            )
    if any("death cause is missing" in warning.lower() for warning in warnings):
        plain_read += " The report cannot yet explain the crashes because death cause is missing."
    elif _replay_has_death_cause(replay_inspection):
        plain_read += f" Local replay found death causes: {_compact_counts(replay_inspection.get('death_cause_counts'))}."

    coach_next_move = _coach_next_move(
        survival_read=survival_read,
        learning_claim=learning_claim,
        claim_blockers=claim_blockers,
        comparability_blockers=comparability_blockers,
    )

    return {
        "survival_read": survival_read,
        "raw_survival_read": raw_survival_read,
        "learning_claim": learning_claim,
        "claim_blockers": claim_blockers,
        "coach_next_move": coach_next_move,
        "action_collapse_seen": action_collapse_seen,
        "strict_comparable": not comparability_blockers,
        "comparability_blockers": comparability_blockers,
        "first_checkpoint": first.get("checkpoint"),
        "latest_checkpoint": latest.get("checkpoint"),
        "best_checkpoint": best.get("checkpoint"),
        "first_mean_steps": first_mean,
        "latest_mean_steps": latest_mean,
        "best_mean_steps": best.get("mean_steps"),
        "latest_minus_first_steps": delta,
        "latest_over_first": latest_mean / first_mean if first_mean else None,
        "delta_basis": delta_basis,
        "paired_first_latest_survival_delta": paired_first_latest_survival_delta,
        "meaningful_delta_threshold_steps": threshold,
        "plain_read": plain_read,
    }


def _claim_blockers(
    *,
    manifest: dict[str, Any],
    config: dict[str, Any],
    train_summary: dict[str, Any] | None,
    checkpoint_rows: list[dict[str, Any]],
    replay_inspection: dict[str, Any],
    comparability_blockers: list[str],
    action_collapse_seen: bool,
) -> list[str]:
    blockers = list(comparability_blockers)
    if action_collapse_seen:
        blockers.append("action_collapse")
    if train_summary is None:
        blockers.append("missing_train_summary")
    else:
        if _safe_bool(train_summary.get("current_policy_self_play")) is False:
            blockers.append("train_summary_current_policy_self_play_false")
        if _safe_bool(train_summary.get("debug_fidelity_only")) is True:
            blockers.append("debug_fidelity_only_true")
        if _safe_bool(train_summary.get("learning_proof")) is False:
            blockers.append("learning_proof_false")

    opponent_label = _opponent_comparison_label(config, train_summary)
    if opponent_label == "fixed_opponent":
        blockers.append("fixed_opponent")
    elif opponent_label == "frozen_opponent":
        blockers.append("frozen_opponent")

    if not any(row.get("death_cause_available") for row in checkpoint_rows) and not _replay_has_death_cause(
        replay_inspection
    ):
        blockers.append("missing_death_cause")

    if not _has_baseline_panel(manifest):
        blockers.append("missing_baseline_panel")

    return _dedupe(blockers)


def _has_baseline_panel(manifest: dict[str, Any]) -> bool:
    # There is no full baseline-panel support in this inspector yet, but accept
    # explicit future fields without changing the claim-blocker contract.
    for key in ("baseline_panel", "baseline_eval", "baseline_report", "baseline_survival_panel"):
        value = manifest.get(key)
        if isinstance(value, dict) and value:
            return True
        if isinstance(value, list) and value:
            return True
    return False


def _coach_next_move(
    *,
    survival_read: str,
    learning_claim: str,
    claim_blockers: list[str],
    comparability_blockers: list[str],
) -> str:
    if comparability_blockers:
        return (
            "Rerun a comparable panel first; this eval has comparability blockers: "
            f"{_human_claim_blockers(comparability_blockers)}."
        )
    if survival_read == "worse":
        return "Stop or avoid scaling this run; inspect death causes before more training."
    if survival_read == "flat":
        return "Do not scale from this run; inspect the shortest deaths and compare baselines."
    if survival_read == "improved" and claim_blockers:
        return (
            "Keep this as narrow survival evidence and clear blockers: "
            f"{_human_claim_blockers(claim_blockers)}."
        )
    if survival_read == "improved" and learning_claim == "survival_improved_in_this_eval_only":
        return "Treat this as tentative and verify it on another seed bundle."
    if claim_blockers:
        return (
            "Do not promote this as learning yet; keep survival as a narrow eval read "
            f"and clear: {_human_claim_blockers(claim_blockers)}."
        )
    return "Collect at least two comparable checkpoints before making a training call."


def _human_claim_blockers(blockers: list[str]) -> str:
    labels = {
        "action_collapse": "action collapse",
        "debug_fidelity_only_true": "debug-fidelity-only training",
        "fixed_opponent": "fixed opponent",
        "frozen_opponent": "frozen opponent",
        "heavy_capped_rows": "too many capped eval rows",
        "learning_proof_false": "train summary says learning_proof=false",
        "missing_baseline_panel": "missing baseline panel",
        "missing_death_cause": "missing death cause",
        "missing_train_summary": "missing train summary",
        "mixed_caps": "mixed eval caps",
        "non_strict_or_missing_checkpoint_load": "non-strict or missing checkpoint loads",
        "seed_counter_mismatch": "seed counter mismatch",
        "failed_eval_rows": "failed eval rows",
        "train_summary_current_policy_self_play_false": (
            "train summary says current_policy_self_play=false"
        ),
    }
    return ", ".join(labels.get(blocker, blocker.replace("_", " ")) for blocker in blockers)


def _questions_next(*, warnings: list[str], verdict: dict[str, Any]) -> list[str]:
    questions: list[str] = []
    warning_text = " ".join(warnings).lower()
    claim_blockers = set(str(item) for item in verdict.get("claim_blockers", []) if item)
    if "missing_death_cause" in claim_blockers:
        questions.append("Add or fetch death cause so short episodes can be explained.")
    if "fixed-opponent" in warning_text or "frozen-opponent" in warning_text:
        questions.append("Compare against simple baselines and self-play evals before claiming learning.")
    if verdict.get("survival_read") in {"flat", "worse"}:
        questions.append("Inspect the shortest seeds near death and ask why the agent dies.")
    if verdict.get("survival_read") == "improved":
        questions.append("Check whether the same improvement holds across another seed bundle.")
    if verdict.get("action_collapse_seen"):
        questions.append("Inspect policy action mix by player and by episode, not only aggregate counts.")
    if "missing_baseline_panel" in claim_blockers:
        questions.append("Add a simple baseline panel before upgrading survival lift to learning.")
    if "missing_train_summary" in claim_blockers:
        questions.append("Attach the train summary so training path and fidelity claims can be checked.")
    if not questions:
        questions.append("Run the same inspector on the next eval bundle and compare reports.")
    return _dedupe(questions)


def _opponent_comparison_label(
    config: dict[str, Any],
    train_summary: dict[str, Any] | None,
) -> str:
    opponent_kind = str(config.get("opponent_policy_kind") or "")
    relation = ""
    if train_summary is not None:
        relation = str(train_summary.get("opponent_training_relation") or "")
    text = f"{opponent_kind} {relation}".lower()
    if "frozen" in text:
        return "frozen_opponent"
    if opponent_kind.startswith("fixed_") or "fixed" in text:
        return "fixed_opponent"
    if "self" in text or "current" in text:
        return "self_play_or_current_policy"
    if opponent_kind:
        return "opponent_policy"
    return "unknown"


def _train_summary_block(train_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if train_summary is None:
        return None
    return {
        "schema_id": train_summary.get("schema_id"),
        "run_id": train_summary.get("run_id"),
        "attempt_id": train_summary.get("attempt_id"),
        "algorithm": train_summary.get("algorithm"),
        "surface": train_summary.get("surface"),
        "compute": train_summary.get("compute"),
        "called_train_muzero": train_summary.get("called_train_muzero"),
        "current_policy_self_play": train_summary.get("current_policy_self_play"),
        "opponent_training_relation": train_summary.get("opponent_training_relation"),
        "source_fidelity_claim": train_summary.get("source_fidelity_claim"),
        "debug_fidelity_only": train_summary.get("debug_fidelity_only"),
        "learning_proof": train_summary.get("learning_proof"),
        "reward_policy": train_summary.get("reward_policy"),
        "trainer_entrypoint": train_summary.get("trainer_entrypoint"),
    }


def _train_action_observability(train_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if train_summary is None:
        return None
    action_observability = _dict_or_empty(train_summary.get("action_observability"))
    ego_counts = _normalise_counts(action_observability.get("ego_action_histogram"))
    opponent_counts = _normalise_counts(action_observability.get("opponent_action_histogram"))
    return {
        "status": action_observability.get("status"),
        "row_count": action_observability.get("row_count"),
        "done_count": action_observability.get("done_count"),
        "ego_action_histogram": ego_counts,
        "ego_action_summary": _action_summary(
            ego_counts,
            collapse_threshold=DEFAULT_COLLAPSE_THRESHOLD,
        ),
        "opponent_action_histogram": opponent_counts,
        "opponent_action_summary": _action_summary(
            opponent_counts,
            collapse_threshold=DEFAULT_COLLAPSE_THRESHOLD,
        ),
        "terminal_reasons": _normalise_counts(action_observability.get("terminal_reasons")),
        "collapse_warning": action_observability.get("collapse_warning"),
        "path": action_observability.get("path"),
    }


def _sum_action_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        histogram = _first_nonempty(row.get("action_histogram"), row.get("actions"))
        for action, count in _normalise_counts(histogram).items():
            counts[action] += count
    return dict(sorted(counts.items()))


def _action_summary(
    counts: dict[str, int],
    *,
    collapse_threshold: float,
) -> dict[str, Any]:
    total = sum(counts.values())
    if total <= 0:
        return {
            "action_total": 0,
            "top_action": None,
            "top_action_count": 0,
            "top_action_fraction": None,
            "action_collapsed": None,
        }
    top_action, top_count = max(counts.items(), key=lambda item: item[1])
    fraction = top_count / total
    return {
        "action_total": total,
        "top_action": top_action,
        "top_action_count": top_count,
        "top_action_fraction": fraction,
        "action_collapsed": fraction >= collapse_threshold,
    }


def _death_cause_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        cause = _first_nonempty(
            row.get("death_cause_name"),
            row.get("death_cause"),
            row.get("death_reason"),
            row.get("collision_kind"),
            row.get("hit_kind"),
            row.get("terminal_detail"),
        )
        if cause in (None, "", "none", 0):
            continue
        hit_owner = row.get("death_hit_owner")
        label = str(cause)
        if hit_owner not in (None, "", -1):
            label = f"{label}:hit_owner={hit_owner}"
        counts[label] += 1
    return counts


def _row_is_capped(row: dict[str, Any]) -> bool:
    terminal = str(_first_nonempty(row.get("terminal_reason"), row.get("terminal"), ""))
    if terminal == "cap":
        return True
    steps = _safe_float(_first_nonempty(row.get("steps_survived"), row.get("steps")))
    cap = _safe_float(row.get("cap"))
    return steps is not None and cap is not None and steps >= cap


def _row_can_drive_primary_survival(row: dict[str, Any]) -> bool:
    if _safe_bool(row.get("ok")) is False:
        return False
    return _safe_bool(_first_nonempty(row.get("strict_load"), row.get("strict"))) is True


def _unique_seed_count(table: list[dict[str, Any]]) -> int:
    return len(
        {
            seed
            for seed in (_safe_int(row.get("seed")) for row in table)
            if seed is not None
        }
    )


def _seed_sets_match(table: list[dict[str, Any]]) -> bool | None:
    if not table:
        return None
    grouped = _seed_counters_by_checkpoint(table)
    if len(grouped) <= 1:
        return None
    seed_counters = {tuple(sorted(counter.items())) for counter in grouped.values()}
    return len(seed_counters) == 1


def _seed_counters_by_checkpoint(table: list[dict[str, Any]]) -> dict[str, Counter[int]]:
    grouped: dict[str, Counter[int]] = {}
    for row in table:
        checkpoint = str(_first_nonempty(row.get("checkpoint_label"), row.get("checkpoint"), "missing"))
        seed = _safe_int(row.get("seed"))
        if seed is None:
            continue
        grouped.setdefault(checkpoint, Counter())[seed] += 1
    return grouped



def _selected_iterations(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _checkpoint_sort_key(label: str) -> tuple[int, int | str]:
    match = re.search(r"iteration_(\d+)", str(label))
    if match is not None:
        return (0, int(match.group(1)))
    return (1, str(label))


def _normalise_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, raw_count in value.items():
        count = _safe_int(raw_count)
        if count is None:
            continue
        counts[str(key)] = count
    return dict(sorted(counts.items()))


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _clean_path_part(value: Any) -> str:
    text = str(value or "unknown")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return cleaned or "unknown"


def _format_md(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _compact_counts(value: Any) -> str:
    counts = _dict_or_empty(value)
    if not counts:
        return ""
    return ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _compact_list(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "None"
    return ", ".join(str(item) for item in value if item)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a local CurvyTron eval manifest.")
    parser.add_argument("--eval-manifest", required=True, help="Path to eval manifest JSON.")
    parser.add_argument("--train-summary", help="Optional path to train summary JSON.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Base output directory when --output-dir is not set.",
    )
    parser.add_argument("--output-dir", help="Exact output directory for report files.")
    parser.add_argument(
        "--collapse-threshold",
        type=float,
        default=DEFAULT_COLLAPSE_THRESHOLD,
        help="Warn when one action reaches this fraction.",
    )
    parser.add_argument(
        "--min-meaningful-delta-steps",
        type=float,
        default=DEFAULT_MIN_MEANINGFUL_DELTA_STEPS,
        help="Minimum absolute mean-survival change counted as meaningful.",
    )
    parser.add_argument(
        "--min-meaningful-delta-fraction",
        type=float,
        default=DEFAULT_MIN_MEANINGFUL_DELTA_FRACTION,
        help="Minimum fractional mean-survival change counted as meaningful.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = Path(args.eval_manifest)
    train_summary_path = Path(args.train_summary) if args.train_summary else None
    manifest = load_json_object(manifest_path)
    train_summary = load_json_object(train_summary_path) if train_summary_path is not None else None
    report = build_inspector_report(
        manifest,
        eval_manifest_path=manifest_path,
        train_summary=train_summary,
        train_summary_path=train_summary_path,
        collapse_threshold=args.collapse_threshold,
        min_meaningful_delta_steps=args.min_meaningful_delta_steps,
        min_meaningful_delta_fraction=args.min_meaningful_delta_fraction,
    )
    output_dir = Path(args.output_dir) if args.output_dir else output_dir_for_report(report, args.output_root)
    paths = write_report_files(report, output_dir)
    print(f"wrote {paths['json']}")
    print(f"wrote {paths['markdown']}")
    print(render_markdown_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
