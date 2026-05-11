"""Audit LightZero dummy Pong policy targets against scoreable actions.

This is a small diagnostic, not training. It samples concrete dummy Pong
contact-pressure states, sweeps the ego actions in the real env to get a
scoreability oracle, then asks a LightZero checkpoint for:

* initial policy logits
* eval-mode MCTS root visit counts
* eval-mode selected action
* collect-mode executed action, when requested

LightZero trains policy logits from the MCTS root visit distribution, not from
the exploratory action that collect mode happens to execute. This audit keeps
those two things separate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


ACTION_LABELS = ("up", "stay", "down")
FEATURE_NAMES = (
    "ego_paddle_y",
    "opponent_paddle_y",
    "ego_paddle_x",
    "opponent_paddle_x",
    "ball_dx_forward",
    "ball_dy_from_ego_center",
    "ball_vx_forward",
    "ball_vy",
    "ball_y",
    "step",
)


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in {"inspect", "inspect-stored"}:
        args = _parse_inspect_args(argv[1:])
        result = inspect_stored_targets(args)
        rendered = _render_inspect(result, output_format=args.format)
        print(rendered)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                rendered + ("" if rendered.endswith("\n") else "\n"),
                encoding="utf-8",
            )
        return

    args = _parse_args(argv)
    result = audit_target_policy(args)
    rendered = _render(result, output_format=args.format)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            rendered + ("" if rendered.endswith("\n") else "\n"),
            encoding="utf-8",
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="lightzero:LABEL=PATH_OR_REF",
        help=(
            "LightZero checkpoint. Repeat for multiple checkpoints. Ref prefixes "
            "ref: and volume: are resolved under --volume-root."
        ),
    )
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--state-seed",
        action="append",
        type=int,
        default=None,
        help="Explicit contact-pressure reset seed. Repeat to choose exact states.",
    )
    parser.add_argument(
        "--pressure-agent",
        choices=("player_0", "player_1"),
        default="player_0",
        help="Which paddle is the audited ego/pressure agent.",
    )
    parser.add_argument(
        "--opponent-policy",
        choices=("track_ball", "lagged_track_ball_1", "random_uniform", "stay"),
        default="lagged_track_ball_1",
        help="Opponent used by the scoreability oracle rollout.",
    )
    parser.add_argument(
        "--ego-action-mode",
        choices=("first_step", "until_contact"),
        default="until_contact",
        help="How long to hold each candidate ego action before TrackBall takes over.",
    )
    parser.add_argument("--max-env-step", type=int, default=64)
    parser.add_argument("--lightzero-env", default="dummy_pong_lag1")
    parser.add_argument("--feature-mode", choices=("tabular_ego", "raster_flat"), default="tabular_ego")
    parser.add_argument(
        "--config-opponent-policy",
        default="random_uniform",
        help="Opponent policy used only to reconstruct the LightZero checkpoint config.",
    )
    parser.add_argument(
        "--num-simulations",
        action="append",
        type=int,
        default=None,
        help="MCTS simulation count. Repeat for a small ladder.",
    )
    parser.add_argument("--collect-repeats", type=int, default=1)
    parser.add_argument("--collect-temperature", type=float, default=0.25)
    parser.add_argument("--collect-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--replay-rows",
        type=Path,
        default=None,
        help=(
            "Optional JSONL replay rows. When an observation matches a sampled "
            "state, stored target/collector action fields are included."
        ),
    )
    parser.add_argument(
        "--scoreability-only",
        action="store_true",
        help="Only sample states and run the env oracle; no LightZero import needed.",
    )
    parser.add_argument(
        "--volume-root",
        type=Path,
        default=Path(os.environ.get("CURVYZERO_RUNS_MOUNT", "/runs")),
    )
    parser.add_argument("--format", choices=("md", "json", "jsonl"), default="md")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.rows < 1:
        parser.error("--rows must be at least 1")
    if args.max_env_step < 1:
        parser.error("--max-env-step must be at least 1")
    if args.collect_repeats < 0:
        parser.error("--collect-repeats must be non-negative")
    if not args.scoreability_only and not args.checkpoint:
        parser.error("--checkpoint is required unless --scoreability-only is set")
    return args


def _parse_inspect_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect stored LightZero collect targets in local artifacts. "
            "This looks for root visit targets already persisted to disk; it "
            "does not run MCTS."
        )
    )
    parser.add_argument(
        "--path",
        action="append",
        type=Path,
        default=[],
        help="Artifact file or directory to inspect. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--with-pickle",
        action="store_true",
        help=(
            "Also load .pkl/.pickle files. Only use for trusted local artifacts; "
            "pickle loading can execute code."
        ),
    )
    parser.add_argument("--max-files", type=int, default=400)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument(
        "--oracle-action",
        choices=("up", "stay", "down", "0", "1", "2"),
        default="down",
        help="Useful action to measure target mass for. Defaults to down/action 2.",
    )
    parser.add_argument(
        "--state-seed",
        action="append",
        type=int,
        default=None,
        help="Optional contact-pressure state seed to try matching against stored observations.",
    )
    parser.add_argument("--pressure-agent", choices=("player_0", "player_1"), default="player_0")
    parser.add_argument("--max-env-step", type=int, default=64)
    parser.add_argument("--feature-mode", choices=("tabular_ego", "raster_flat"), default="tabular_ego")
    parser.add_argument("--format", choices=("md", "json"), default="md")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.path:
        parser.error("inspect-stored requires at least one --path")
    if args.max_files < 1:
        parser.error("--max-files must be at least 1")
    return args


def inspect_stored_targets(args: argparse.Namespace) -> dict[str, Any]:
    oracle_action = _parse_action_id(args.oracle_action)
    match_vectors = _sample_match_vectors(args)
    files = _collect_inspect_files(
        args.path,
        with_pickle=args.with_pickle,
        max_files=args.max_files,
    )
    file_reports = []
    records = []
    for path in files:
        report = _inspect_target_file(
            path,
            with_pickle=args.with_pickle,
            oracle_action=oracle_action,
            match_vectors=match_vectors,
            max_examples=args.max_examples,
        )
        file_reports.append(report)
        records.extend(report.get("records", []))
    return {
        "schema": "curvyzero_lightzero_dummy_pong_stored_target_inspection/v0",
        "ok": True,
        "question": (
            "Do persisted collect/replay artifacts contain LightZero "
            "visit_count_distributions, and do those stored policy targets put "
            f"mass on {ACTION_LABELS[oracle_action]}?"
        ),
        "storage_read": {
            "lightzero_collect_target_in_memory": "GameSegment.child_visit_segment",
            "lightzero_collect_executed_action_in_memory": "GameSegment.action_segment",
            "training_sample_target_source": (
                "MuZeroGameBuffer samples GameSegment.child_visit_segment for "
                "non-reanalyzed policy targets and may overwrite it during reanalyze."
            ),
            "curvyzero_current_modal_mirror": (
                "summary.json, episodes.jsonl, lightzero_training_signals.json, "
                "lightzero_artifacts_manifest.json, config/command/stdout/stderr, "
                "and mirrored checkpoints; no replay buffer/game segment artifact "
                "is mirrored by the current wrapper."
            ),
        },
        "config": {
            "paths": [str(path) for path in args.path],
            "files_inspected": len(files),
            "with_pickle": bool(args.with_pickle),
            "oracle_action": oracle_action,
            "oracle_action_label": ACTION_LABELS[oracle_action],
            "state_seeds": args.state_seed or [],
            "pressure_agent": args.pressure_agent,
            "max_env_step": args.max_env_step,
            "feature_mode": args.feature_mode,
        },
        "matched_state_vectors": match_vectors,
        "file_reports": file_reports,
        "summary": _summarize_stored_records(records, oracle_action=oracle_action),
    }


def _collect_inspect_files(
    paths: list[Path],
    *,
    with_pickle: bool,
    max_files: int,
) -> list[Path]:
    suffixes = {".json", ".jsonl", ".npz"}
    if with_pickle:
        suffixes.update({".pkl", ".pickle"})
    files = []
    for root in paths:
        if root.is_file() and root.suffix.lower() in suffixes:
            files.append(root)
        elif root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix.lower() in suffixes:
                    files.append(path)
                    if len(files) >= max_files:
                        return files
    return files[:max_files]


def _inspect_target_file(
    path: Path,
    *,
    with_pickle: bool,
    oracle_action: int,
    match_vectors: list[dict[str, Any]],
    max_examples: int,
) -> dict[str, Any]:
    try:
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".jsonl":
            payload = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                payload = {key: data[key] for key in data.files}
        elif suffix in {".pkl", ".pickle"} and with_pickle:
            import pickle

            with path.open("rb") as handle:
                payload = pickle.load(handle)
        else:
            return {
                "path": str(path),
                "status": "skipped",
                "reason": f"unsupported suffix {suffix!r}",
                "records": [],
            }
        records = list(
            _extract_target_records(
                payload,
                source_path=path,
                object_path=(),
                oracle_action=oracle_action,
                match_vectors=match_vectors,
                record_limit=2000,
            )
        )
        return {
            "path": str(path),
            "status": "ok",
            "records_found": len(records),
            "examples": records[:max_examples],
            "records": records,
        }
    except Exception as exc:
        return {
            "path": str(path),
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "records": [],
        }


def _extract_target_records(
    value: Any,
    *,
    source_path: Path,
    object_path: tuple[Any, ...],
    oracle_action: int,
    match_vectors: list[dict[str, Any]],
    record_limit: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if record_limit <= 0:
        return records

    if hasattr(value, "child_visit_segment"):
        records.extend(
            _records_from_visit_action_arrays(
                visits=getattr(value, "child_visit_segment", None),
                actions=getattr(value, "action_segment", None),
                observations=getattr(value, "obs_segment", None),
                source_path=source_path,
                object_path=object_path + ("GameSegment",),
                oracle_action=oracle_action,
                match_vectors=match_vectors,
                record_limit=record_limit,
            )
        )
        return records

    if isinstance(value, dict):
        visit_value = _first_present(
            value,
            (
                "child_visit_segment",
                "visit_count_distributions",
                "visit_count_distribution",
                "visit_counts",
                "target_policy",
                "target_policies",
                "policy_target",
            ),
        )
        if visit_value is not None:
            records.extend(
                _records_from_visit_action_arrays(
                    visits=visit_value,
                    actions=_first_present(
                        value,
                        ("action_segment", "actions", "action", "collector_action_id"),
                    ),
                    observations=_first_present(
                        value,
                        ("obs_segment", "observation", "observations", "feature_vector"),
                    ),
                    source_path=source_path,
                    object_path=object_path,
                    oracle_action=oracle_action,
                    match_vectors=match_vectors,
                    record_limit=record_limit,
                )
            )
        for key, item in value.items():
            if len(records) >= record_limit:
                break
            if key in {
                "child_visit_segment",
                "visit_count_distributions",
                "visit_count_distribution",
                "visit_counts",
                "target_policy",
                "target_policies",
                "policy_target",
            }:
                continue
            if _may_contain_target(item):
                records.extend(
                    _extract_target_records(
                        item,
                        source_path=source_path,
                        object_path=object_path + (str(key),),
                        oracle_action=oracle_action,
                        match_vectors=match_vectors,
                        record_limit=record_limit - len(records),
                    )
                )
        return records

    if isinstance(value, list):
        for index, item in enumerate(value[:500]):
            if len(records) >= record_limit:
                break
            if _may_contain_target(item):
                records.extend(
                    _extract_target_records(
                        item,
                        source_path=source_path,
                        object_path=object_path + (index,),
                        oracle_action=oracle_action,
                        match_vectors=match_vectors,
                        record_limit=record_limit - len(records),
                    )
                )
        return records

    return records


def _records_from_visit_action_arrays(
    *,
    visits: Any,
    actions: Any,
    observations: Any,
    source_path: Path,
    object_path: tuple[Any, ...],
    oracle_action: int,
    match_vectors: list[dict[str, Any]],
    record_limit: int,
) -> list[dict[str, Any]]:
    visit_rows = _as_action_rows(visits)
    if visit_rows is None:
        return []
    action_values = _as_flat_actions(actions)
    obs_rows = _as_observation_rows(observations)
    records = []
    for index, vector in enumerate(visit_rows[:record_limit]):
        summary = _target_summary(vector, oracle_best_actions=[oracle_action])
        action = action_values[index] if action_values is not None and index < len(action_values) else None
        obs_match = _match_observation(
            None if obs_rows is None or index >= len(obs_rows) else obs_rows[index],
            match_vectors,
        )
        records.append(
            {
                "source_path": str(source_path),
                "object_path": "/".join(str(part) for part in object_path),
                "index": index,
                "stored_target_distribution": summary["target_distribution"],
                "stored_target_top_action": summary["target_top_action"],
                "stored_target_top_action_label": summary["target_top_action_label"],
                "stored_target_top_minus_second": summary["target_top_minus_second"],
                "oracle_action_mass": summary["target_mass_on_oracle_best"],
                "oracle_action_gets_positive_mass": summary["oracle_best_gets_positive_mass"],
                "executed_action": action,
                "executed_action_label": None if action is None else ACTION_LABELS[int(action)],
                "executed_oracle_action_but_target_zero": (
                    None
                    if action is None
                    else int(action) == oracle_action
                    and not bool(summary["oracle_best_gets_positive_mass"])
                ),
                "matched_state": obs_match,
            }
        )
    return records


def _as_action_rows(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    try:
        arr = np.asarray(_to_jsonable(value), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        if arr.shape[0] < len(ACTION_LABELS):
            return None
        return [[float(item) for item in arr[: len(ACTION_LABELS)].tolist()]]
    if arr.shape[-1] < len(ACTION_LABELS):
        return None
    return [
        [float(item) for item in row[: len(ACTION_LABELS)].tolist()]
        for row in arr.reshape(-1, arr.shape[-1])
    ]


def _as_flat_actions(value: Any) -> list[int] | None:
    if value is None:
        return None
    try:
        arr = np.asarray(_to_jsonable(value))
    except (TypeError, ValueError):
        return None
    if arr.ndim == 0:
        try:
            return [int(arr.item())]
        except (TypeError, ValueError):
            return None
    try:
        return [int(item) for item in arr.reshape(-1).tolist()]
    except (TypeError, ValueError):
        return None


def _as_observation_rows(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        key = _observation_key(value)
        if all(item is not None for item in key):
            return [[float(item) for item in key]]
        feature_vector = value.get("feature_vector") or value.get("values")
        if feature_vector is not None:
            return _as_observation_rows(feature_vector)
        return None
    try:
        arr = np.asarray(_to_jsonable(value), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        if arr.shape[0] < len(FEATURE_NAMES):
            return None
        return [[float(item) for item in arr[: len(FEATURE_NAMES)].tolist()]]
    if arr.shape[-1] < len(FEATURE_NAMES):
        return None
    return [
        [float(item) for item in row[: len(FEATURE_NAMES)].tolist()]
        for row in arr.reshape(-1, arr.shape[-1])
    ]


def _match_observation(
    observation: list[float] | None,
    match_vectors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if observation is None or not match_vectors:
        return None
    obs = np.asarray(observation, dtype=np.float64)
    for candidate in match_vectors:
        vector = np.asarray(candidate["feature_vector"], dtype=np.float64)
        if obs.shape == vector.shape and np.allclose(obs, vector, atol=1e-6, rtol=1e-6):
            return {
                "state_id": candidate["state_id"],
                "seed": candidate["seed"],
            }
    return None


def _sample_match_vectors(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.state_seed:
        return []
    rows = []
    for seed in args.state_seed:
        state = _state_oracle_row(
            seed=seed,
            pressure_agent=args.pressure_agent,
            max_env_step=args.max_env_step,
            opponent_policy="lagged_track_ball_1",
            ego_action_mode="until_contact",
            feature_mode=args.feature_mode,
            replay_index={},
        )
        rows.append(
            {
                "state_id": state["state_id"],
                "seed": seed,
                "feature_vector": state["feature_vector"],
                "oracle_best_actions": state["scoreability"]["oracle_best_actions"],
                "oracle_best_action_labels": state["scoreability"]["oracle_best_action_labels"],
            }
        )
    return rows


def _summarize_stored_records(
    records: list[dict[str, Any]],
    *,
    oracle_action: int,
) -> dict[str, Any]:
    top_counts = Counter(
        int(row["stored_target_top_action"])
        for row in records
        if row.get("stored_target_top_action") is not None
    )
    action_counts = Counter(
        int(row["executed_action"])
        for row in records
        if row.get("executed_action") is not None
    )
    zero_mass = [
        row for row in records
        if row.get("oracle_action_gets_positive_mass") is False
    ]
    positive_mass = [
        row for row in records
        if row.get("oracle_action_gets_positive_mass") is True
    ]
    matched = [row for row in records if row.get("matched_state")]
    return {
        "records": len(records),
        "stored_target_top_counts_up_stay_down": [int(top_counts[i]) for i in range(3)],
        "executed_action_counts_up_stay_down": [int(action_counts[i]) for i in range(3)],
        "oracle_action": oracle_action,
        "oracle_action_label": ACTION_LABELS[oracle_action],
        "oracle_action_zero_mass_count": len(zero_mass),
        "oracle_action_positive_mass_count": len(positive_mass),
        "oracle_action_positive_mass_rate": (
            None if not records else len(positive_mass) / len(records)
        ),
        "executed_oracle_action_but_target_zero_count": sum(
            bool(row.get("executed_oracle_action_but_target_zero")) for row in records
        ),
        "matched_state_record_count": len(matched),
        "matched_state_examples": matched[:8],
    }


def _render_inspect(result: dict[str, Any], *, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(result, indent=2, sort_keys=True)
    summary = result["summary"]
    lines = [
        "Stored target inspection",
        "",
        f"- Files inspected: {result['config']['files_inspected']}",
        f"- Records with stored policy targets: {summary.get('records', 0)}",
        f"- Stored location in LightZero: {result['storage_read']['lightzero_collect_target_in_memory']}",
        f"- Executed action location in LightZero: {result['storage_read']['lightzero_collect_executed_action_in_memory']}",
        f"- Current CurvyZero mirror: {result['storage_read']['curvyzero_current_modal_mirror']}",
    ]
    if summary.get("records", 0):
        lines.extend(
            [
                "",
                "| metric | value |",
                "| --- | --- |",
                f"| target top counts up/stay/down | {summary['stored_target_top_counts_up_stay_down']} |",
                f"| executed action counts up/stay/down | {summary['executed_action_counts_up_stay_down']} |",
                f"| {summary['oracle_action_label']} positive target mass rate | {_fmt(summary['oracle_action_positive_mass_rate'])} |",
                f"| {summary['oracle_action_label']} zero target mass count | {summary['oracle_action_zero_mass_count']} |",
                f"| executed {summary['oracle_action_label']} but target zero count | {summary['executed_oracle_action_but_target_zero_count']} |",
                f"| matched sampled-state records | {summary['matched_state_record_count']} |",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "No persisted visit target records were found in the inspected files.",
                "For current Modal train outputs this means the answer is: the targets are stored in memory during LightZero collection/training, but the wrapper does not yet save the replay buffer/game segments to disk.",
            ]
        )
    target_files = [
        report for report in result["file_reports"] if report.get("records_found", 0)
    ]
    if target_files:
        lines.extend(["", "| file | records | first examples |", "| --- | ---: | --- |"])
        for report in target_files[:20]:
            examples = report.get("examples", [])
            compact = "; ".join(
                (
                    f"i={item['index']} target={_fmt_vector(item['stored_target_distribution'])} "
                    f"exec={item.get('executed_action_label')}"
                )
                for item in examples[:3]
            )
            lines.append(
                f"| {_escape_md(report['path'])} | {report.get('records_found', 0)} | {_escape_md(compact)} |"
            )
    return "\n".join(lines)


def _may_contain_target(value: Any) -> bool:
    if hasattr(value, "child_visit_segment"):
        return True
    if isinstance(value, dict):
        keys = set(value)
        if keys & {
            "child_visit_segment",
            "visit_count_distributions",
            "visit_count_distribution",
            "visit_counts",
            "target_policy",
            "target_policies",
            "policy_target",
        }:
            return True
        return any(_may_contain_target(item) for item in list(value.values())[:50])
    if isinstance(value, list):
        return any(_may_contain_target(item) for item in value[:50])
    return False


def _parse_action_id(value: str) -> int:
    labels = {label: index for index, label in enumerate(ACTION_LABELS)}
    if value in labels:
        return labels[value]
    action = int(value)
    if action < 0 or action >= len(ACTION_LABELS):
        raise ValueError(f"action out of range: {value!r}")
    return action


def audit_target_policy(args: argparse.Namespace) -> dict[str, Any]:
    state_seeds = list(args.state_seed or [])
    while len(state_seeds) < args.rows:
        state_seeds.append(args.seed + len(state_seeds))
    state_seeds = state_seeds[: args.rows]

    replay_index = _load_replay_index(args.replay_rows) if args.replay_rows else {}
    states = [
        _state_oracle_row(
            seed=state_seed,
            pressure_agent=args.pressure_agent,
            max_env_step=args.max_env_step,
            opponent_policy=args.opponent_policy,
            ego_action_mode=args.ego_action_mode,
            feature_mode=args.feature_mode,
            replay_index=replay_index,
        )
        for state_seed in state_seeds
    ]
    if args.scoreability_only:
        audit_rows: list[dict[str, Any]] = []
        checkpoint_summaries: list[dict[str, Any]] = []
    else:
        checkpoint_summaries = []
        audit_rows = []
        for checkpoint_arg in args.checkpoint:
            label, checkpoint_path = _resolve_checkpoint_arg(
                checkpoint_arg,
                volume_root=args.volume_root,
            )
            checkpoint_summaries.append(
                {
                    "label": label,
                    "path": str(checkpoint_path),
                    "bytes": checkpoint_path.stat().st_size,
                }
            )
            for num_simulations in args.num_simulations or [2]:
                audit_rows.extend(
                    _checkpoint_audit_rows(
                        checkpoint_label=label,
                        checkpoint_path=checkpoint_path,
                        states=states,
                        lightzero_env=args.lightzero_env,
                        feature_mode=args.feature_mode,
                        config_opponent_policy=args.config_opponent_policy,
                        seed=args.seed,
                        max_env_step=args.max_env_step,
                        num_simulations=num_simulations,
                        collect_repeats=args.collect_repeats,
                        collect_temperature=args.collect_temperature,
                        collect_epsilon=args.collect_epsilon,
                    )
                )

    return {
        "schema": "curvyzero_lightzero_dummy_pong_target_policy_audit/v0",
        "ok": True,
        "note": (
            "LightZero policy logits are trained from MCTS root visit targets. "
            "Collect-mode executed actions are reported separately."
        ),
        "config": {
            "rows": args.rows,
            "seed": args.seed,
            "state_seeds": state_seeds,
            "pressure_agent": args.pressure_agent,
            "opponent_policy_for_scoreability": args.opponent_policy,
            "ego_action_mode": args.ego_action_mode,
            "max_env_step": args.max_env_step,
            "lightzero_env": args.lightzero_env,
            "feature_mode": args.feature_mode,
            "config_opponent_policy": args.config_opponent_policy,
            "num_simulations": args.num_simulations or [2],
            "collect_repeats": args.collect_repeats,
            "collect_temperature": args.collect_temperature,
            "collect_epsilon": args.collect_epsilon,
            "replay_rows": None if args.replay_rows is None else str(args.replay_rows),
            "scoreability_only": bool(args.scoreability_only),
        },
        "action_labels": list(ACTION_LABELS),
        "feature_names": list(FEATURE_NAMES),
        "checkpoint_summaries": checkpoint_summaries,
        "states": states,
        "audit_rows": audit_rows,
        "summary": _summarize_audit_rows(audit_rows),
    }


def _state_oracle_row(
    *,
    seed: int,
    pressure_agent: str,
    max_env_step: int,
    opponent_policy: str,
    ego_action_mode: str,
    feature_mode: str,
    replay_index: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    from curvyzero.training.dummy_pong import PongConfig
    from curvyzero.training.dummy_pong import PongEnv
    from curvyzero.training.lightzero_dummy_pong_features import encode_lightzero_observation

    config = PongConfig(
        max_steps=max_env_step,
        reset_profile="contact_pressure",
        reset_pressure_agent=pressure_agent,
    )
    env = PongEnv(config)
    observations = env.reset(seed=seed)
    observation = observations[pressure_agent]
    raster = env.raster_observation()
    encoded = encode_lightzero_observation(
        feature_mode=feature_mode,
        observation=observation,
        config=config,
        raster_grid=raster,
    )
    candidates = [
        _candidate_rollout(
            config=config,
            seed=seed,
            ego_agent=pressure_agent,
            action=action,
            opponent_policy_name=opponent_policy,
            ego_action_mode=ego_action_mode,
        )
        for action in range(len(ACTION_LABELS))
    ]
    best_score = max(float(row["score_return"]) for row in candidates)
    best_survival = max(
        int(row["survival_steps"])
        for row in candidates
        if float(row["score_return"]) == best_score
    )
    best_actions = [
        int(row["action"])
        for row in candidates
        if float(row["score_return"]) == best_score
        and int(row["survival_steps"]) == best_survival
    ]
    sorted_scores = sorted(
        {float(row["score_return"]) for row in candidates},
        reverse=True,
    )
    score_margin = (
        float(sorted_scores[0] - sorted_scores[1])
        if len(sorted_scores) > 1
        else 0.0
    )
    replay_match = replay_index.get(_observation_key(asdict(observation)))
    return {
        "state_id": f"{pressure_agent}-seed-{seed}",
        "seed": int(seed),
        "pressure_agent": pressure_agent,
        "ego_agent": pressure_agent,
        "opponent_agent": _opponent(pressure_agent),
        "initial_snapshot": _snapshot(env),
        "observation": asdict(observation),
        "feature_vector": [float(value) for value in encoded.reshape(-1).tolist()],
        "scoreability": {
            "oracle_best_actions": best_actions,
            "oracle_best_action_labels": [ACTION_LABELS[action] for action in best_actions],
            "oracle_best_is_tie": len(best_actions) > 1,
            "oracle_best_score_return": best_score,
            "oracle_score_margin": score_margin,
            "candidate_results": candidates,
        },
        "replay_match": replay_match,
    }


def _candidate_rollout(
    *,
    config: Any,
    seed: int,
    ego_agent: str,
    action: int,
    opponent_policy_name: str,
    ego_action_mode: str,
) -> dict[str, Any]:
    from curvyzero.training.dummy_pong import PongEnv
    from curvyzero.training.dummy_pong_eval import TrackBallPolicy

    opponent_agent = _opponent(ego_agent)
    env = PongEnv(config)
    env.reset(seed=seed)
    ego_policy = TrackBallPolicy()
    opponent_policy = _make_policy(opponent_policy_name, seed=seed + 10_000)
    ego_policy.reset(seed, ego_agent)
    opponent_policy.reset(seed, opponent_agent)

    released_to_baseline = False
    score_return = 0.0
    first_contact = None
    trace_prefix = []
    final_step = None
    for rollout_step in range(config.max_steps):
        observations = env.observations()
        raster = env.raster_observation()
        ego_action = (
            int(ego_policy.action(observations[ego_agent], raster, ego_agent))
            if released_to_baseline
            else int(action)
        )
        opponent_action = int(
            opponent_policy.action(observations[opponent_agent], raster, opponent_agent)
        )
        joint_action = {ego_agent: ego_action, opponent_agent: opponent_action}
        if len(trace_prefix) < 8:
            trace_prefix.append(dict(joint_action))
        final_step = env.step(joint_action)
        score_return += float(final_step.rewards[ego_agent])
        impact = final_step.infos["last_hit_impact"]
        if first_contact is None and impact is not None:
            first_contact = {
                "step": int(next(iter(final_step.observations.values())).step),
                **dict(impact),
            }
            if ego_action_mode == "until_contact":
                released_to_baseline = True
        if ego_action_mode == "first_step" and rollout_step == 0:
            released_to_baseline = True
        if final_step.terminated or final_step.truncated:
            break
    if final_step is None:
        raise RuntimeError("candidate rollout did not run")
    terminal_step = int(next(iter(final_step.observations.values())).step)
    return {
        "action": int(action),
        "action_label": ACTION_LABELS[int(action)],
        "score_return": float(score_return),
        "ego_reward": float(final_step.rewards[ego_agent]),
        "winner": final_step.infos["winner"],
        "terminated": bool(final_step.terminated),
        "truncated": bool(final_step.truncated),
        "survival_steps": terminal_step,
        "first_contact": first_contact,
        "action_trace_prefix": trace_prefix,
    }


def _checkpoint_audit_rows(
    *,
    checkpoint_label: str,
    checkpoint_path: Path,
    states: list[dict[str, Any]],
    lightzero_env: str,
    feature_mode: str,
    config_opponent_policy: str,
    seed: int,
    max_env_step: int,
    num_simulations: int,
    collect_repeats: int,
    collect_temperature: float,
    collect_epsilon: float,
) -> list[dict[str, Any]]:
    import torch

    from curvyzero.training.lightzero_dummy_pong_policy import (
        load_lightzero_mcts_eval_mode_checkpoint,
    )

    spec = load_lightzero_mcts_eval_mode_checkpoint(
        policy_id=f"lightzero_{_clean_label(checkpoint_label)}_mcts{num_simulations}",
        checkpoint_path=checkpoint_path,
        env=lightzero_env,
        feature_mode=feature_mode,
        opponent_policy=config_opponent_policy,
        seed=seed,
        max_env_step=max_env_step,
        num_simulations=num_simulations,
    )
    policy = spec.policy._policy
    torch_module = spec.policy._torch
    rows = []
    for row_index, state in enumerate(states):
        obs_tensor = torch.as_tensor([state["feature_vector"]], dtype=torch.float32)
        with torch.no_grad():
            network_output = policy._model.initial_inference(obs_tensor)
        logits = [
            float(value)
            for value in network_output.policy_logits.detach().cpu().reshape(-1).tolist()
        ]
        eval_output = _forward_policy(
            policy,
            torch_module=torch_module,
            obs_tensor=obs_tensor,
            mode="eval",
            temperature=collect_temperature,
            epsilon=collect_epsilon,
        )
        eval_compact = _compact_forward_output(eval_output)
        mcts_action = _extract_action(eval_output)
        visit_field, visit_vector = _visit_vector(eval_compact)
        target = _target_summary(
            visit_vector,
            oracle_best_actions=state["scoreability"]["oracle_best_actions"],
        )
        collect_decisions = []
        for repeat in range(collect_repeats):
            np.random.seed(seed + row_index * 1000 + repeat)
            collect_output = _forward_policy(
                policy,
                torch_module=torch_module,
                obs_tensor=obs_tensor,
                mode="collect",
                temperature=collect_temperature,
                epsilon=collect_epsilon,
            )
            collect_action = _extract_action(collect_output)
            collect_decisions.append(
                {
                    "repeat": repeat,
                    "action": int(collect_action),
                    "action_label": ACTION_LABELS[int(collect_action)],
                    "matches_root_target_top": (
                        None
                        if target["target_top_action"] is None
                        else int(collect_action) == int(target["target_top_action"])
                    ),
                    "matches_oracle_best": int(collect_action)
                    in set(state["scoreability"]["oracle_best_actions"]),
                    "output": _compact_forward_output(collect_output),
                }
            )
        logit_summary = _vector_summary(logits)
        replay_match = state.get("replay_match")
        rows.append(
            {
                "state_id": state["state_id"],
                "checkpoint": checkpoint_label,
                "checkpoint_path": str(checkpoint_path),
                "num_simulations": int(num_simulations),
                "oracle_best_actions": state["scoreability"]["oracle_best_actions"],
                "oracle_best_action_labels": state["scoreability"]["oracle_best_action_labels"],
                "oracle_best_is_tie": state["scoreability"]["oracle_best_is_tie"],
                "oracle_score_margin": state["scoreability"]["oracle_score_margin"],
                "policy_logits": logits,
                "policy_logit_summary": logit_summary,
                "mcts_root_visit_field": visit_field,
                "mcts_root_visits": visit_vector,
                "mcts_root_target": target,
                "mcts_action": int(mcts_action),
                "mcts_action_label": ACTION_LABELS[int(mcts_action)],
                "mcts_action_matches_oracle_best": int(mcts_action)
                in set(state["scoreability"]["oracle_best_actions"]),
                "collect_decisions": collect_decisions,
                "stored_replay_target": _stored_replay_target(replay_match),
                "eval_output": eval_compact,
            }
        )
    return rows


def _forward_policy(
    policy: Any,
    *,
    torch_module: Any,
    obs_tensor: Any,
    mode: str,
    temperature: float,
    epsilon: float,
) -> Any:
    action_mask = np.ones((1, len(ACTION_LABELS)), dtype=np.float32)
    with torch_module.no_grad():
        if mode == "eval":
            return policy.eval_mode.forward(
                obs_tensor,
                action_mask=action_mask,
                to_play=[-1],
                ready_env_id=np.asarray([0]),
            )
        if mode == "collect":
            return policy.collect_mode.forward(
                obs_tensor,
                action_mask=action_mask,
                temperature=temperature,
                epsilon=epsilon,
                to_play=[-1],
                ready_env_id=np.asarray([0]),
            )
    raise ValueError(f"unknown policy mode: {mode!r}")


def _compact_forward_output(output: Any) -> dict[str, Any]:
    root = _unwrap_single_env_output(_to_jsonable(output))
    if not isinstance(root, dict):
        return {"raw": root, "output_type": type(output).__name__}
    wanted = (
        "action",
        "selected_action",
        "visit_count_distribution",
        "visit_count_distributions",
        "visit_count_distribution_entropy",
        "visit_counts",
        "policy_logits",
        "predicted_policy_logits",
        "predicted_value",
        "searched_value",
        "value",
    )
    compact = {key: root[key] for key in wanted if key in root}
    compact["output_keys"] = sorted(str(key) for key in root.keys())
    return compact


def _target_summary(
    visit_vector: list[float] | None,
    *,
    oracle_best_actions: list[int],
) -> dict[str, Any]:
    if visit_vector is None:
        return {
            "target_source": None,
            "target_distribution": None,
            "target_top_action": None,
            "target_top_action_label": None,
            "target_top_minus_second": None,
            "target_top_is_tie": None,
            "target_mass_on_oracle_best": None,
            "oracle_best_gets_positive_mass": None,
            "target_top_matches_oracle_best": None,
        }
    total = float(sum(visit_vector))
    if total > 0.0:
        distribution = [float(value) / total for value in visit_vector]
    else:
        distribution = [0.0 for _ in visit_vector]
    summary = _vector_summary(visit_vector)
    oracle_set = set(int(action) for action in oracle_best_actions)
    target_top = summary["top_action"]
    oracle_mass = sum(distribution[action] for action in oracle_set if action < len(distribution))
    return {
        "target_source": "mcts_root_visits_normalized",
        "target_distribution": distribution,
        "target_top_action": target_top,
        "target_top_action_label": summary["top_action_label"],
        "target_top_minus_second": summary["top_minus_second"],
        "target_top_is_tie": summary["is_top_tie_at_1e_12"],
        "target_mass_on_oracle_best": float(oracle_mass),
        "oracle_best_gets_positive_mass": bool(oracle_mass > 0.0),
        "target_top_matches_oracle_best": int(target_top) in oracle_set,
    }


def _vector_summary(vector: list[float] | None) -> dict[str, Any] | None:
    if vector is None:
        return None
    order = sorted(range(len(vector)), key=vector.__getitem__, reverse=True)
    top = int(order[0])
    second = int(order[1]) if len(order) > 1 else top
    return {
        "up_stay_down": [float(value) for value in vector[:3]],
        "top_action": top,
        "top_action_label": ACTION_LABELS[top],
        "top_minus_second": float(vector[top] - vector[second]),
        "is_top_tie_at_1e_12": bool(abs(vector[top] - vector[second]) <= 1e-12),
        "softmax_up_stay_down": _softmax(vector[:3]),
    }


def _visit_vector(compact: dict[str, Any]) -> tuple[str | None, list[float] | None]:
    for field in ("visit_count_distributions", "visit_count_distribution", "visit_counts"):
        vector = _flat_action_vector(compact.get(field))
        if vector is not None:
            return field, vector
    return None, None


def _flat_action_vector(value: Any) -> list[float] | None:
    plain = _to_jsonable(value)
    if plain is None:
        return None
    current = plain
    while isinstance(current, list) and len(current) == 1 and isinstance(current[0], list):
        current = current[0]
    if not isinstance(current, list) or len(current) < len(ACTION_LABELS):
        return None
    try:
        return [float(item) for item in current[: len(ACTION_LABELS)]]
    except (TypeError, ValueError):
        return None


def _extract_action(output: Any) -> int:
    payload = _unwrap_single_env_output(_to_jsonable(output))
    if isinstance(payload, dict):
        for key in ("action", "selected_action", "selected_actions", "actions"):
            if key in payload:
                value = payload[key]
                while isinstance(value, list) and len(value) == 1:
                    value = value[0]
                return int(value)
    raise ValueError(f"could not extract action from policy output: {payload!r}")


def _make_policy(policy_name: str, *, seed: int) -> Any:
    from curvyzero.training.dummy_pong_eval import LaggedTrackBallPolicy
    from curvyzero.training.dummy_pong_eval import RandomUniformPolicy
    from curvyzero.training.dummy_pong_eval import TrackBallPolicy

    if policy_name == "track_ball":
        return TrackBallPolicy()
    if policy_name == "lagged_track_ball_1":
        return LaggedTrackBallPolicy(delay=1)
    if policy_name == "random_uniform":
        return RandomUniformPolicy(seed=seed)
    if policy_name == "stay":
        return StayPolicy()
    raise ValueError(f"unknown opponent policy: {policy_name!r}")


class StayPolicy:
    name = "stay"

    def reset(self, episode_seed: int, agent: str) -> None:
        del episode_seed, agent

    def action(self, observation: object, raster_grid: np.ndarray, agent: str) -> int:
        del observation, raster_grid, agent
        return 1


def _load_replay_index(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    replay_path = path / "replay_rows.jsonl" if path.is_dir() else path
    if not replay_path.is_file():
        raise FileNotFoundError(f"replay rows file not found: {replay_path}")
    index = {}
    for line in replay_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        observation = row.get("observation")
        if isinstance(observation, dict):
            index.setdefault(_observation_key(observation), row)
    return index


def _stored_replay_target(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    target_action = _first_present(row, ("target_action_id", "policy_target_action_id"))
    collector_action = _first_present(row, ("collector_action_id", "behavior_action_id", "action_id"))
    policy_target = _first_present(
        row,
        (
            "policy_target",
            "target_policy",
            "target_policy_distribution",
            "visit_count_distribution",
            "visit_counts",
        ),
    )
    return {
        "schema_id": row.get("schema_id"),
        "target_policy_id": row.get("target_policy_id"),
        "target_action_id": None if target_action is None else int(target_action),
        "target_action_label": None if target_action is None else ACTION_LABELS[int(target_action)],
        "collector_action_id": None if collector_action is None else int(collector_action),
        "collector_action_label": (
            None if collector_action is None else ACTION_LABELS[int(collector_action)]
        ),
        "policy_target": _to_jsonable(policy_target),
        "replay_semantics_id": row.get("replay_semantics_id"),
    }


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _observation_key(observation: dict[str, Any]) -> tuple[Any, ...]:
    fields = (
        "ego_paddle_y",
        "opponent_paddle_y",
        "ego_paddle_x",
        "opponent_paddle_x",
        "ball_dx_forward",
        "ball_dy_from_ego_center",
        "ball_vx_forward",
        "ball_vy",
        "ball_y",
        "step",
    )
    return tuple(observation.get(field) for field in fields)


def _summarize_audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    root_matches = [
        bool(row["mcts_root_target"]["target_top_matches_oracle_best"])
        for row in rows
        if row["mcts_root_target"]["target_top_matches_oracle_best"] is not None
    ]
    mcts_matches = [bool(row["mcts_action_matches_oracle_best"]) for row in rows]
    positive_mass = [
        bool(row["mcts_root_target"]["oracle_best_gets_positive_mass"])
        for row in rows
        if row["mcts_root_target"]["oracle_best_gets_positive_mass"] is not None
    ]
    action_counts = Counter(int(row["mcts_action"]) for row in rows)
    return {
        "rows": len(rows),
        "mcts_action_counts_up_stay_down": [int(action_counts[i]) for i in range(3)],
        "root_target_top_matches_oracle_rate": _mean_bool(root_matches),
        "mcts_action_matches_oracle_rate": _mean_bool(mcts_matches),
        "oracle_best_positive_target_mass_rate": _mean_bool(positive_mass),
    }


def _render(result: dict[str, Any], *, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(result, indent=2, sort_keys=True)
    if output_format == "jsonl":
        return "\n".join(json.dumps(row, sort_keys=True) for row in result["audit_rows"])
    if not result["audit_rows"]:
        return json.dumps(
            {
                "states": [
                    {
                        "state_id": state["state_id"],
                        "oracle_best": state["scoreability"]["oracle_best_action_labels"],
                        "oracle_margin": state["scoreability"]["oracle_score_margin"],
                    }
                    for state in result["states"]
                ],
                "summary": result["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    columns = [
        "row",
        "ckpt",
        "sims",
        "state",
        "oracle",
        "logits top/margin",
        "visits",
        "target mass",
        "mcts",
        "collect",
        "stored",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for index, row in enumerate(result["audit_rows"]):
        target = row["mcts_root_target"]
        logit_summary = row["policy_logit_summary"] or {}
        collect_parts = []
        for item in row["collect_decisions"]:
            field, vector = _visit_vector(item.get("output", {}))
            del field
            collect_parts.append(
                f"{item['action_label']}:{_fmt_vector(vector)}" if vector is not None else item["action_label"]
            )
        collect = ",".join(collect_parts)
        stored = row.get("stored_replay_target")
        stored_text = ""
        if stored:
            stored_text = (
                f"target={stored.get('target_action_label')} "
                f"collector={stored.get('collector_action_label')}"
            )
        display = {
            "row": str(index),
            "ckpt": str(row["checkpoint"]),
            "sims": str(row["num_simulations"]),
            "state": str(row["state_id"]),
            "oracle": _action_list(row["oracle_best_actions"]),
            "logits top/margin": (
                f"{logit_summary.get('top_action_label')}/"
                f"{_fmt(logit_summary.get('top_minus_second'))}"
            ),
            "visits": _fmt_vector(row.get("mcts_root_visits")),
            "target mass": _fmt(target.get("target_mass_on_oracle_best")),
            "mcts": f"{row['mcts_action_label']}:{'Y' if row['mcts_action_matches_oracle_best'] else 'N'}",
            "collect": collect,
            "stored": stored_text,
        }
        lines.append("| " + " | ".join(_escape_md(display[column]) for column in columns) + " |")
    summary = result["summary"]
    lines.append("")
    lines.append(
        "Summary: "
        f"rows={summary.get('rows')}, "
        f"target_top_matches_oracle={_fmt(summary.get('root_target_top_matches_oracle_rate'))}, "
        f"oracle_positive_target_mass={_fmt(summary.get('oracle_best_positive_target_mass_rate'))}, "
        f"mcts_action_matches_oracle={_fmt(summary.get('mcts_action_matches_oracle_rate'))}"
    )
    return "\n".join(lines)


def _resolve_checkpoint_arg(value: str, *, volume_root: Path) -> tuple[str, Path]:
    text = value.removeprefix("lightzero:")
    if "=" in text:
        label, ref = text.split("=", 1)
        label = label.strip()
        ref = ref.strip()
    else:
        ref = text.strip()
        label = Path(ref).stem
    if not label or not ref:
        raise ValueError(f"bad checkpoint arg: {value!r}")
    path = _resolve_path_or_ref(ref, volume_root=volume_root)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {path}")
    return label, path


def _resolve_path_or_ref(value: str, *, volume_root: Path) -> Path:
    for prefix in ("ref:", "volume:"):
        if value.startswith(prefix):
            ref = value[len(prefix) :]
            if ref.startswith("/") or ".." in Path(ref).parts:
                raise ValueError(f"unsafe volume ref: {value!r}")
            return volume_root / ref
    return Path(value)


def _snapshot(env: Any) -> dict[str, Any]:
    return {
        "player_0_y": int(env._paddle_y["player_0"]),
        "player_1_y": int(env._paddle_y["player_1"]),
        "ball_x": int(env._ball_x),
        "ball_y": int(env._ball_y),
        "ball_vx": int(env._ball_vx),
        "ball_vy": int(env._ball_vy),
        "step": int(env._step),
        "reset_info": dict(env._reset_info),
    }


def _opponent(agent: str) -> str:
    if agent == "player_0":
        return "player_1"
    if agent == "player_1":
        return "player_0"
    raise ValueError(f"unknown agent: {agent!r}")


def _unwrap_single_env_output(value: Any) -> Any:
    if isinstance(value, dict):
        for key in (0, "0"):
            if key in value:
                return _unwrap_single_env_output(value[key])
    if isinstance(value, list) and len(value) == 1:
        return _unwrap_single_env_output(value[0])
    return value


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


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _mean_bool(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _action_list(actions: list[int]) -> str:
    return ",".join(ACTION_LABELS[int(action)] for action in actions)


def _fmt_vector(value: Any) -> str:
    vector = _flat_action_vector(value)
    if vector is None:
        return ""
    return "[" + ",".join(_fmt(item) for item in vector) + "]"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _escape_md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _clean_label(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
