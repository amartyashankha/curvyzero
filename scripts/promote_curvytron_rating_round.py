#!/usr/bin/env python3
"""Promote one verified CurvyTron rating round into a trainer assignment.

This is deliberately a small operator/controller transaction. It does not run
training policy logic and it does not trust mutable pointers as durable truth.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from curvyzero.infra.modal import run_management as runs
from curvyzero.contracts.curvytron import (
    CURVYTRON_TRAINING_TASK_ID,
    curvytron_control_volume_name,
    curvytron_runs_volume_name,
    curvytron_tournament_volume_name,
)
from curvyzero.tournament import curvytron_checkpoint_tournament as arena
from curvyzero.training.opponent_leaderboard import (
    DEFAULT_ACTIVE_MIN_DISTINCT_OPPONENTS,
    DEFAULT_ACTIVE_MIN_VALID_GAMES,
    DEFAULT_MAX_ACTIVE_RANK,
    DEFAULT_MAX_FAILURE_RATE,
    OPPONENT_DEATH_MODE_NORMAL,
    STABLE_SENTINEL_BLANK_CANVAS,
    STABLE_SLOT_PROFILE_3,
    canonical_json_sha256,
    validate_assignment_audit,
    validate_leaderboard_snapshot,
    validate_rating_snapshot_source,
)
from curvyzero.training.opponent_registry import (
    canonical_assignment_json_sha256,
    parse_opponent_assignment_snapshot,
)


TRAINING_VOLUME_NAME = curvytron_runs_volume_name()
CONTROL_VOLUME_NAME = curvytron_control_volume_name()
TOURNAMENT_VOLUME_NAME = curvytron_tournament_volume_name()
TOURNAMENT_MODULE = "curvyzero.infra.modal.curvyzero_checkpoint_tournament"
TRAINING_MODULE = "curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train"
TRAINING_TASK_ID = CURVYTRON_TRAINING_TASK_ID


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def leaderboard_snapshot_ref(leaderboard_id: str, snapshot_id: str) -> PurePosixPath:
    return (
        arena.TOURNAMENT_BASE_REF
        / "leaderboards"
        / runs.clean_id(leaderboard_id, label="leaderboard_id")
        / "snapshots"
        / f"{runs.clean_id(snapshot_id, label='snapshot_id')}.json"
    )


def load_json_with_modal_noise(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        candidates = json_values_with_modal_noise(text)
        if candidates:
            return candidates[0]
    raise ValueError(f"{path} does not contain JSON")


def json_values_with_modal_noise(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    values = []
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        values.append(value)
    return values


def best_json_object_from_command_output(text: str) -> dict[str, Any]:
    candidates = [
        value for value in json_values_with_modal_noise(text) if isinstance(value, dict)
    ]
    if not candidates:
        raise ValueError("command output did not contain a JSON object")
    for value in reversed(candidates):
        if value.get("schema_id"):
            return value
    return candidates[0]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_command(command: Sequence[str], *, dry_run: bool = False) -> dict[str, Any]:
    if dry_run:
        return {
            "command": list(command),
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "dry_run": True,
        }
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )
    result = {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed with exit code "
            f"{completed.returncode}: {' '.join(command)}\n{completed.stderr}"
        )
    return result


def run_json_command(command: Sequence[str], *, dry_run: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    result = run_command(command, dry_run=dry_run)
    if dry_run:
        return result, {}
    scratch = Path("/private/tmp") / "curvy-promotion-json-output.json"
    scratch.write_text(
        str(result.get("stdout") or "") + "\n" + str(result.get("stderr") or ""),
        encoding="utf-8",
    )
    payload = best_json_object_from_command_output(scratch.read_text(encoding="utf-8"))
    return result, payload


def fetch_volume_json(
    *,
    volume_name: str,
    ref: PurePosixPath | str,
    output_path: Path,
    dry_run: bool = False,
    skip_fetch: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    clean_ref = runs.require_relative_ref(ref).as_posix()
    if not skip_fetch:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "modal",
            "volume",
            "get",
            "--force",
            volume_name,
            clean_ref,
            output_path.as_posix(),
        ]
        command_result = run_command(command, dry_run=dry_run)
    else:
        command_result = {
            "command": [
                "modal",
                "volume",
                "get",
                "--force",
                volume_name,
                clean_ref,
                output_path.as_posix(),
            ],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "skip_fetch": True,
        }
    if dry_run:
        return command_result, {}
    payload = load_json_with_modal_noise(output_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{output_path} must contain a JSON object")
    return command_result, payload


def fetch_volume_file(
    *,
    volume_name: str,
    ref: PurePosixPath | str,
    output_path: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    clean_ref = runs.require_relative_ref(ref).as_posix()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "modal",
        "volume",
        "get",
        "--force",
        volume_name,
        clean_ref,
        output_path.as_posix(),
    ]
    return run_command(command, dry_run=dry_run)


def rating_round_refs(
    *,
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
) -> dict[str, PurePosixPath]:
    return {
        "input": arena.rating_round_input_ref(tournament_id, rating_run_id, round_id),
        "progress": arena.rating_round_progress_ref(tournament_id, rating_run_id, round_id),
        "results": arena.rating_round_results_ref(tournament_id, rating_run_id, round_id),
        "ratings": arena.rating_round_ratings_ref(tournament_id, rating_run_id, round_id),
        "latest": arena.rating_latest_ref(tournament_id, rating_run_id),
    }


def _round_id_if_present(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("round_id")
    return str(value) if value is not None else None


def _round_index_if_present(payload: Mapping[str, Any]) -> int | None:
    value = payload.get("round_index")
    return int(value) if value is not None else None


def _nested_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else []


def _int_if_present(payload: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key not in payload or payload[key] is None:
            continue
        try:
            return int(payload[key])
        except (TypeError, ValueError):
            return None
    return None


def _failure_count(*payloads: Mapping[str, Any]) -> int:
    total = 0
    for payload in payloads:
        for key in ("failed_game_count", "failure_count", "invalid_game_count"):
            value = _int_if_present(payload, key)
            if value is not None:
                total += value
        for row in _sequence(payload.get("ratings")):
            if isinstance(row, Mapping):
                total += int(row.get("failure_count") or 0)
    return total


def validate_round_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    tournament_id: str,
    rating_run_id: str,
    round_id: str,
    allow_failed_games: bool = False,
) -> dict[str, Any]:
    latest = dict(artifacts["latest"])
    ratings = dict(artifacts["ratings"])
    for name, payload in artifacts.items():
        artifact_round_id = _round_id_if_present(payload)
        if artifact_round_id is not None and artifact_round_id != round_id:
            raise ValueError(
                f"{name}.json round_id mismatch: expected {round_id!r}, got {artifact_round_id!r}"
            )
        if payload.get("tournament_id") and payload.get("tournament_id") != tournament_id:
            raise ValueError(f"{name}.json tournament_id mismatch")
        if payload.get("rating_run_id") and payload.get("rating_run_id") != rating_run_id:
            raise ValueError(f"{name}.json rating_run_id mismatch")
    latest_sha = canonical_json_sha256(latest)
    ratings_sha = canonical_json_sha256(ratings)
    if ratings_sha != latest_sha:
        raise ValueError("round ratings.json and latest.json are not the same payload")
    rating_spec = _nested_mapping(latest.get("rating_spec")) or _nested_mapping(
        artifacts["input"].get("rating_spec")
    )
    if int(rating_spec.get("decision_source_frames") or 0) != 1:
        raise ValueError("rating round is not one-frame decision_source_frames=1")
    failed_games = _failure_count(latest, artifacts["results"], artifacts["progress"])
    if failed_games and not allow_failed_games:
        raise ValueError(f"rating round has failed games: {failed_games}")
    rows = _sequence(latest.get("ratings"))
    if not rows:
        raise ValueError("rating latest.json has no rating rows")
    pair_count = _int_if_present(latest, "pair_count")
    result_pair_count = _int_if_present(artifacts["results"], "pair_count", "pair_result_count")
    game_count = _int_if_present(latest, "game_count")
    result_game_count = _int_if_present(artifacts["results"], "game_count")
    if pair_count is not None and result_pair_count is not None and pair_count != result_pair_count:
        raise ValueError(
            f"pair count mismatch: latest={pair_count}, results={result_pair_count}"
        )
    if game_count is not None and result_game_count is not None and game_count != result_game_count:
        raise ValueError(
            f"game count mismatch: latest={game_count}, results={result_game_count}"
        )
    source = validate_rating_snapshot_source(
        latest,
        expected_round_id=round_id,
        expected_round_index=_round_index_if_present(latest),
    )
    return {
        "round_artifacts_agree": True,
        "one_frame": True,
        "zero_failed_games": failed_games == 0,
        "row_count": len(rows),
        "pair_count": pair_count,
        "game_count": game_count,
        "rating_snapshot_sha256": source["rating_snapshot_sha256"],
        "rating_context_hash": source["rating_context_hash"],
        "roster_hash": source["roster_hash"],
        "round_index": source["round_index"],
    }


def build_publish_command(args: argparse.Namespace, source: Mapping[str, Any]) -> list[str]:
    command = [
        "uv",
        "run",
        "--extra",
        "modal",
        "modal",
        "run",
        "-m",
        TOURNAMENT_MODULE,
        "--mode",
        "leaderboard-publish",
        "--tournament-id",
        args.tournament_id,
        "--rating-run-id",
        args.rating_run_id,
        "--leaderboard-id",
        args.leaderboard_id,
        "--leaderboard-snapshot-id",
        args.snapshot_id,
        "--leaderboard-active-min-distinct-opponents",
        str(args.active_min_distinct_opponents),
        "--leaderboard-active-min-valid-games",
        str(args.active_min_valid_games),
        "--leaderboard-max-failure-rate",
        str(args.max_failure_rate),
        "--leaderboard-max-active-rank",
        str(args.max_active_rank),
        "--leaderboard-expected-round-id",
        args.round_id,
        "--leaderboard-expected-rating-context-hash",
        str(source["rating_context_hash"]),
        "--leaderboard-expected-roster-hash",
        str(source["roster_hash"]),
        "--leaderboard-expected-rating-snapshot-sha256",
        str(source["rating_snapshot_sha256"]),
    ]
    if source.get("round_index") is not None:
        command.extend(["--leaderboard-expected-round-index", str(source["round_index"])])
    if args.allow_no_active_rows:
        command.append("--leaderboard-allow-no-active-rows")
    if args.diagnostic_only:
        command.append("--leaderboard-diagnostic-only")
    return command


def build_materialize_command(
    args: argparse.Namespace,
    *,
    snapshot_path: Path,
    snapshot_ref: str,
    output_dir: Path,
    generation: int,
    rating_context_hash: str,
) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "scripts/materialize_curvytron_leaderboard_assignment.py",
        snapshot_path.as_posix(),
        "--output-dir",
        output_dir.as_posix(),
        "--leaderboard-id",
        args.leaderboard_id,
        "--snapshot-id",
        args.snapshot_id,
        "--snapshot-ref",
        snapshot_ref,
        "--assignment-id",
        args.assignment_id,
        "--assignment-source-ref",
        snapshot_ref,
        "--generation",
        str(generation),
        "--seed",
        str(args.assignment_seed),
        "--materializer",
        "stable_slots_v1",
        "--profile",
        args.assignment_profile,
        "--sentinel",
        args.assignment_sentinel,
        "--checkpoint-death-mode",
        args.checkpoint_death_mode,
        "--expected-rating-context-hash",
        rating_context_hash,
    ] + (["--allow-recent-provisional"] if args.allow_recent_provisional else [])


def build_write_assignment_command(args: argparse.Namespace, *, assignment_dir: Path) -> list[str]:
    command = [
        "uv",
        "run",
        "--extra",
        "modal",
        "modal",
        "run",
        "-m",
        TRAINING_MODULE,
        "--mode",
        "write-assignment",
        "--run-id",
        args.assignment_bank_run_id,
        "--attempt-id",
        args.assignment_bank_attempt_id,
        "--opponent-assignment-json-path",
        (assignment_dir / "assignment.json").as_posix(),
        "--opponent-assignment-audit-json-path",
        (assignment_dir / "audit.json").as_posix(),
    ]
    if getattr(args, "assignment_target_volume", "runs") != "runs":
        command.extend(
            [
                "--opponent-assignment-target-volume",
                str(args.assignment_target_volume),
            ]
        )
    if getattr(args, "mirror_assignment_checkpoints_to_control", False):
        command.append("--mirror-assignment-checkpoints-to-control")
    return command


def volume_name_for_assignment_pointer(value: str) -> str:
    if value == "control":
        return CONTROL_VOLUME_NAME
    if value == "runs":
        return TRAINING_VOLUME_NAME
    raise ValueError(f"unknown pointer volume {value!r}")


def pointer_volume_relative_ref(pointer_ref: str) -> str:
    """Return the path Modal's volume CLI should write inside the selected volume."""

    text = str(pointer_ref or "").strip()
    for prefix in ("control:", "runs:"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return runs.require_relative_ref(text).as_posix()


def write_refresh_pointer_command(
    *,
    pointer_path: Path,
    pointer_ref: str,
    pointer_volume: str,
) -> list[str]:
    return [
        "uv",
        "run",
        "--extra",
        "modal",
        "modal",
        "volume",
        "put",
        "--force",
        volume_name_for_assignment_pointer(pointer_volume),
        pointer_path.as_posix(),
        pointer_volume_relative_ref(pointer_ref),
    ]


def build_smoke_command(
    args: argparse.Namespace,
    *,
    assignment_ref: str,
    champion_checkpoint_ref: str,
) -> list[str]:
    return [
        "uv",
        "run",
        "--extra",
        "modal",
        "modal",
        "run",
        "-m",
        TRAINING_MODULE,
        "--mode",
        "train",
        "--compute",
        args.smoke_compute,
        "--run-id",
        args.smoke_run_id,
        "--attempt-id",
        args.smoke_attempt_id,
        "--max-train-iter",
        "1",
        "--max-env-step",
        "128",
        "--source-max-steps",
        "128",
        "--collector-env-num",
        "2",
        "--n-episode",
        "2",
        "--evaluator-env-num",
        "1",
        "--n-evaluator-episode",
        "1",
        "--num-simulations",
        "2",
        "--batch-size",
        "4",
        "--lightzero-eval-freq",
        "0",
        "--save-ckpt-after-iter",
        "1",
        "--stop-after-learner-train-calls",
        "1",
        "--env-variant",
        "source_state_fixed_opponent",
        "--reward-variant",
        args.smoke_reward_variant,
        "--source-state-trail-render-mode",
        "browser_lines",
        "--source-state-bonus-render-mode",
        "simple_symbols",
        "--env-manager-type",
        "base",
        "--env-telemetry-stride",
        "1",
        "--opponent-assignment-ref",
        assignment_ref,
        "--initial-policy-checkpoint-ref",
        champion_checkpoint_ref,
        "--initial-policy-checkpoint-load-mode",
        "matching_shape",
        "--no-background-eval-enabled",
        "--no-background-gif-enabled",
        "--wait-for-train",
        "--output-detail",
        "compact",
    ]


def smoke_artifact_refs(*, run_id: str, attempt_id: str) -> dict[str, PurePosixPath]:
    train_root = runs.attempt_train_ref(TRAINING_TASK_ID, run_id, attempt_id)
    return {
        "summary": train_root / "summary.json",
        "env_steps": train_root / "env_steps.jsonl",
    }


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def verify_smoke_artifacts(
    *,
    summary: Mapping[str, Any],
    env_steps_path: Path,
    assignment_ref: str,
    assignment_sha256: str,
) -> dict[str, Any]:
    if summary.get("ok") is not True:
        raise ValueError("smoke summary did not report ok=true")
    if summary.get("called_train_muzero") is not True:
        raise ValueError("smoke summary did not call train_muzero")
    initial_policy = _nested_mapping(summary.get("initial_policy_checkpoint"))
    load_result = _nested_mapping(initial_policy.get("load_result"))
    module_loads = _sequence(load_result.get("module_loads"))
    meaningful_load = any(
        isinstance(row, Mapping) and row.get("meaningful_model_load") is True
        for row in module_loads
    )
    fresh_optimizer_preserved = any(
        isinstance(row, Mapping) and row.get("fresh_optimizer_preserved") is True
        for row in module_loads
    )
    auto_resume = _nested_mapping(summary.get("auto_resume"))
    if load_result.get("loaded") is not True or not meaningful_load:
        raise ValueError("smoke did not prove meaningful champion checkpoint load")
    if not fresh_optimizer_preserved:
        raise ValueError("smoke did not prove fresh optimizer preservation")
    if auto_resume.get("found") is not False:
        raise ValueError("smoke did not prove fresh-run auto-resume was absent")
    rows = _iter_jsonl(env_steps_path)
    provider_rows = [
        row
        for row in rows
        if row.get("opponent_assignment_ref") == assignment_ref
        and row.get("opponent_assignment_sha256") == assignment_sha256
        and row.get("opponent_provider_load_ok") is True
    ]
    if not provider_rows:
        raise ValueError("env telemetry did not prove assignment/provider load")
    return {
        "summary_ok": True,
        "called_train_muzero": True,
        "initial_checkpoint_loaded": True,
        "meaningful_model_load": True,
        "fresh_optimizer_preserved": True,
        "auto_resume_found": False,
        "env_step_row_count": len(rows),
        "provider_ok_row_count": len(provider_rows),
        "assignment_ref": assignment_ref,
        "assignment_sha256": assignment_sha256,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tournament-id", required=True)
    parser.add_argument("--rating-run-id", default=arena.DEFAULT_RATING_RUN_ID)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--leaderboard-id", required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument("--assignment-bank-run-id", required=True)
    parser.add_argument("--assignment-bank-attempt-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/curvy-promotions"),
    )
    parser.add_argument("--assignment-seed", type=int, default=0)
    parser.add_argument("--assignment-profile", default=STABLE_SLOT_PROFILE_3)
    parser.add_argument("--assignment-sentinel", default=STABLE_SENTINEL_BLANK_CANVAS)
    parser.add_argument("--checkpoint-death-mode", default=OPPONENT_DEATH_MODE_NORMAL)
    parser.add_argument(
        "--assignment-target-volume",
        choices=("runs", "control"),
        default="runs",
        help="Where write-assignment stores the trainer assignment.",
    )
    parser.add_argument(
        "--mirror-assignment-checkpoints-to-control",
        action="store_true",
        help="Rewrite checkpoint refs to control-volume mirrors while writing assignment.",
    )
    parser.add_argument(
        "--refresh-pointer-ref",
        default="",
        help="Optional mutable pointer ref to update after the assignment write succeeds.",
    )
    parser.add_argument(
        "--refresh-pointer-volume",
        choices=("runs", "control"),
        default="control",
        help="Volume containing --refresh-pointer-ref.",
    )
    parser.add_argument("--allow-recent-provisional", action="store_true")
    parser.add_argument(
        "--active-min-distinct-opponents",
        type=int,
        default=DEFAULT_ACTIVE_MIN_DISTINCT_OPPONENTS,
    )
    parser.add_argument(
        "--active-min-valid-games",
        type=int,
        default=DEFAULT_ACTIVE_MIN_VALID_GAMES,
    )
    parser.add_argument("--max-failure-rate", type=float, default=DEFAULT_MAX_FAILURE_RATE)
    parser.add_argument("--max-active-rank", type=int, default=DEFAULT_MAX_ACTIVE_RANK)
    parser.add_argument("--allow-no-active-rows", action="store_true")
    parser.add_argument("--allow-failed-games", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Read already-fetched JSON files from the output directory.",
    )
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--smoke-run-id", default="")
    parser.add_argument("--smoke-attempt-id", default="")
    parser.add_argument("--smoke-compute", default="cpu")
    parser.add_argument(
        "--smoke-reward-variant",
        default="survival_plus_bonus_plus_outcome",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    promotion_id = f"{args.leaderboard_id}-{args.snapshot_id}-{args.assignment_id}"
    output_dir = args.output_dir / runs.clean_id(promotion_id[:96], label="promotion_id")
    fetched_dir = output_dir / "fetched"
    local_assignment_dir = output_dir / "assignment"
    command_log: list[dict[str, Any]] = []

    refs = rating_round_refs(
        tournament_id=args.tournament_id,
        rating_run_id=args.rating_run_id,
        round_id=args.round_id,
    )
    artifacts: dict[str, dict[str, Any]] = {}
    for name, ref in refs.items():
        result, payload = fetch_volume_json(
            volume_name=TOURNAMENT_VOLUME_NAME,
            ref=ref,
            output_path=fetched_dir / f"{name}.json",
            dry_run=args.dry_run,
            skip_fetch=args.skip_fetch,
        )
        command_log.append(result)
        artifacts[name] = payload

    if args.dry_run:
        source = {
            "round_index": None,
            "rating_context_hash": "<dry-run>",
            "roster_hash": "<dry-run>",
            "rating_snapshot_sha256": "<dry-run>",
        }
    else:
        source = validate_round_artifacts(
            artifacts,
            tournament_id=args.tournament_id,
            rating_run_id=args.rating_run_id,
            round_id=args.round_id,
            allow_failed_games=args.allow_failed_games,
        )

    publish_command = build_publish_command(args, source)
    publish_result, publish_payload = run_json_command(publish_command, dry_run=args.dry_run)
    command_log.append(publish_result)
    if not args.dry_run and publish_payload.get("commit_error"):
        raise RuntimeError(f"leaderboard publish commit failed: {publish_payload['commit_error']}")

    snapshot_ref = leaderboard_snapshot_ref(args.leaderboard_id, args.snapshot_id).as_posix()
    snapshot_path = fetched_dir / "leaderboard_snapshot.json"
    fetch_snapshot_result, snapshot_payload = fetch_volume_json(
        volume_name=TOURNAMENT_VOLUME_NAME,
        ref=snapshot_ref,
        output_path=snapshot_path,
        dry_run=args.dry_run,
        skip_fetch=False,
    )
    command_log.append(fetch_snapshot_result)
    if args.dry_run:
        snapshot = {}
        champion_checkpoint_ref = "<dry-run>"
        leaderboard_snapshot_sha256 = "<dry-run>"
    else:
        snapshot = validate_leaderboard_snapshot(snapshot_payload)
        leaderboard_snapshot_sha256 = snapshot["snapshot_sha256"]
        if snapshot["source"]["rating_snapshot_sha256"] != source["rating_snapshot_sha256"]:
            raise ValueError("published leaderboard source hash does not match preflight rating")
        champion_row = snapshot["rows"][0]
        champion_checkpoint_ref = str(champion_row["checkpoint_ref"])

    materialize_command = build_materialize_command(
        args,
        snapshot_path=snapshot_path,
        snapshot_ref=snapshot_ref,
        output_dir=local_assignment_dir,
        generation=int(source.get("round_index") or 0),
        rating_context_hash=str(source["rating_context_hash"]),
    )
    materialize_result, materialize_payload = run_json_command(
        materialize_command,
        dry_run=args.dry_run,
    )
    command_log.append(materialize_result)
    if args.dry_run:
        assignment = {}
        audit = {}
        assignment_sha256 = "<dry-run>"
    else:
        assignment = load_json_with_modal_noise(local_assignment_dir / "assignment.json")
        audit = load_json_with_modal_noise(local_assignment_dir / "audit.json")
        parsed = parse_opponent_assignment_snapshot(assignment)
        if parsed is None:
            raise ValueError("materialized assignment did not parse")
        validate_assignment_audit(audit, assignment=assignment)
        assignment_sha256 = canonical_assignment_json_sha256(assignment)

    write_command = build_write_assignment_command(args, assignment_dir=local_assignment_dir)
    write_result, write_payload = run_json_command(write_command, dry_run=args.dry_run)
    command_log.append(write_result)
    if args.dry_run:
        assignment_ref = "<dry-run>"
        assignment_audit_ref = "<dry-run>"
    else:
        assignment_ref = str(write_payload["assignment_ref"])
        assignment_audit_ref = str(write_payload.get("audit_ref") or "")
        written_assignment_sha256 = str(write_payload.get("assignment_sha256") or "")
        if written_assignment_sha256 != assignment_sha256 and not (
            args.assignment_target_volume == "control"
            and args.mirror_assignment_checkpoints_to_control
        ):
            raise ValueError("remote assignment write returned a different sha256")
        if written_assignment_sha256:
            assignment_sha256 = written_assignment_sha256

    refresh_pointer_ref = str(args.refresh_pointer_ref or "")
    refresh_pointer_payload: dict[str, Any] | None = None
    if refresh_pointer_ref:
        refresh_pointer_payload = {
            "schema_id": "curvyzero_opponent_assignment_refresh_pointer/v0",
            "assignment_ref": assignment_ref,
            "assignment_sha256": assignment_sha256,
        }
        pointer_path = output_dir / "refresh_pointer.json"
        write_json(pointer_path, refresh_pointer_payload)
        pointer_command = write_refresh_pointer_command(
            pointer_path=pointer_path,
            pointer_ref=refresh_pointer_ref,
            pointer_volume=args.refresh_pointer_volume,
        )
        pointer_result = run_command(pointer_command, dry_run=args.dry_run)
        command_log.append(pointer_result)

    smoke_payload: dict[str, Any] | None = None
    smoke_artifact_verification: dict[str, Any] | None = None
    if args.run_smoke:
        if not args.smoke_run_id or not args.smoke_attempt_id:
            raise ValueError("--run-smoke requires --smoke-run-id and --smoke-attempt-id")
        smoke_command = build_smoke_command(
            args,
            assignment_ref=assignment_ref,
            champion_checkpoint_ref=champion_checkpoint_ref,
        )
        smoke_result, smoke_payload = run_json_command(smoke_command, dry_run=args.dry_run)
        command_log.append(smoke_result)
        if not args.dry_run and (
            smoke_payload.get("ok") is not True
            or smoke_payload.get("called_train_muzero") is not True
        ):
            raise RuntimeError(
                "trainer smoke did not prove a train call: "
                f"ok={smoke_payload.get('ok')!r}, "
                f"called_train_muzero={smoke_payload.get('called_train_muzero')!r}"
            )
        if not args.dry_run:
            smoke_refs = smoke_artifact_refs(
                run_id=args.smoke_run_id,
                attempt_id=args.smoke_attempt_id,
            )
            smoke_dir = output_dir / "smoke"
            smoke_summary_result, smoke_summary = fetch_volume_json(
                volume_name=TRAINING_VOLUME_NAME,
                ref=smoke_refs["summary"],
                output_path=smoke_dir / "summary.json",
                dry_run=False,
                skip_fetch=False,
            )
            command_log.append(smoke_summary_result)
            smoke_env_result = fetch_volume_file(
                volume_name=TRAINING_VOLUME_NAME,
                ref=smoke_refs["env_steps"],
                output_path=smoke_dir / "env_steps.jsonl",
                dry_run=False,
            )
            command_log.append(smoke_env_result)
            smoke_artifact_verification = verify_smoke_artifacts(
                summary=smoke_summary,
                env_steps_path=smoke_dir / "env_steps.jsonl",
                assignment_ref=assignment_ref,
                assignment_sha256=assignment_sha256,
            )

    decision = {
        "schema_id": "curvyzero_promotion_decision/v0",
        "promotion_id": promotion_id,
        "created_at": utc_timestamp(),
        "source": {
            "tournament_id": args.tournament_id,
            "rating_run_id": args.rating_run_id,
            "round_id": args.round_id,
            "rating_snapshot_ref": refs["latest"].as_posix(),
            **dict(source),
        },
        "leaderboard": {
            "leaderboard_id": args.leaderboard_id,
            "snapshot_id": args.snapshot_id,
            "snapshot_ref": snapshot_ref,
            "snapshot_sha256": leaderboard_snapshot_sha256,
            "publish_result": publish_payload,
        },
        "training_bundle": {
            "assignment_id": args.assignment_id,
            "assignment_ref": assignment_ref,
            "assignment_sha256": assignment_sha256,
            "assignment_audit_ref": assignment_audit_ref,
            "assignment_target_volume": args.assignment_target_volume,
            "refresh_pointer_ref": refresh_pointer_ref,
            "refresh_pointer_volume": args.refresh_pointer_volume if refresh_pointer_ref else "",
            "refresh_pointer_payload": refresh_pointer_payload,
            "champion_checkpoint_ref": champion_checkpoint_ref,
            "smoke_result": smoke_payload,
            "smoke_artifact_verification": smoke_artifact_verification,
        },
        "gates": {
            "round_artifacts_agree": bool(source.get("round_artifacts_agree", False)),
            "one_frame": bool(source.get("one_frame", False)),
            "zero_failed_games": bool(source.get("zero_failed_games", False)),
            "leaderboard_snapshot_fetched": bool(snapshot_ref),
            "assignment_written": bool(assignment_ref),
            "refresh_pointer_written": bool(refresh_pointer_ref),
            "smoke_requested": bool(args.run_smoke),
            "smoke_passed": (
                bool(
                    smoke_payload
                    and smoke_payload.get("ok") is True
                    and smoke_payload.get("called_train_muzero") is True
                )
                if args.run_smoke
                else None
            ),
            "smoke_artifacts_verified": (
                smoke_artifact_verification is not None if args.run_smoke else None
            ),
        },
        "materialize_result": materialize_payload,
        "commands": command_log,
    }
    write_json(output_dir / "promotion_decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
