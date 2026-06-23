#!/usr/bin/env python3
"""Prepare old CurvyTron champion refs for all-v2 rematerialization/rerate."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from curvyzero.contracts.curvytron import (
    CURVYTRON_DECISION_MS,
    CURVYTRON_DEFAULT_NUM_SIMULATIONS,
    CURVYTRON_POLICY_BONUS_RENDER_MODE,
    CURVYTRON_POLICY_TRAIL_RENDER_MODE,
    CURVYTRON_SOURCE_MAX_STEPS,
    DEFAULT_CURVYTRON_RUNS_VOLUME_NAME,
    DEFAULT_CURVYTRON_TOURNAMENT_VOLUME_NAME,
)


SCHEMA_ID = "curvyzero_curvytron_restart_source_ref_plan/v0"
ITERATION_CKPT_RE = re.compile(r"iteration_\d+\.pth\.tar\Z")
DEFAULT_SOURCE_VOLUME = "curvyzero-runs"
DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_restart_source_refs")
DEFAULT_PLAN_ID = "restart18-source-loop18-top100-20260515a"
DEFAULT_TOURNAMENT_ID = "curvy-restart18-source-rerate-20260515a"
DEFAULT_RATING_RUN_ID = "elo-restart18-source-rerate-20260515a"


def _load_snapshot(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    rows = payload.get("rows") or payload.get("ratings")
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain rows or ratings")
    return payload


def _row_rank(row: dict[str, Any], fallback: int) -> int:
    value = row.get("rank", fallback)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _select_rows(
    snapshot: dict[str, Any],
    *,
    limit: int,
    required_status: str,
    min_iteration: int,
) -> list[dict[str, Any]]:
    rows = snapshot.get("rows") or snapshot.get("ratings") or []
    candidates = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "")
        if required_status and status != required_status:
            continue
        checkpoint_ref = str(row.get("checkpoint_ref") or "").strip()
        if not checkpoint_ref:
            continue
        iteration = row.get("iteration")
        try:
            iteration_int = int(iteration)
        except (TypeError, ValueError):
            iteration_int = None
        if iteration_int is not None and iteration_int < min_iteration:
            continue
        rank = _row_rank(row, index)
        candidates.append((rank, index, row))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [row for _rank, _index, row in candidates[:limit]]


def _validate_checkpoint_ref(ref: str) -> None:
    if ref.startswith(("runs:", "control:")):
        raise ValueError(f"checkpoint ref must be volume-relative, got {ref!r}")
    if "latest" in ref or "ckpt_best" in ref:
        raise ValueError(f"checkpoint ref is mutable: {ref}")
    if not ITERATION_CKPT_RE.fullmatch(PurePosixPath(ref).name):
        raise ValueError(f"checkpoint ref must end in iteration_N.pth.tar: {ref}")


def _selection_entry(row: dict[str, Any], *, ordinal: int) -> dict[str, Any]:
    checkpoint_ref = str(row.get("checkpoint_ref") or "").strip()
    _validate_checkpoint_ref(checkpoint_ref)
    return {
        "ordinal": ordinal,
        "rank": _row_rank(row, ordinal),
        "status": row.get("status"),
        "checkpoint_ref": checkpoint_ref,
        "checkpoint_id": row.get("checkpoint_id"),
        "run_id": row.get("run_id"),
        "attempt_id": row.get("attempt_id"),
        "iteration": row.get("iteration"),
        "rating": row.get("rating"),
        "label": row.get("label"),
        "evidence": row.get("evidence"),
    }


def _quoted_command(parts: Sequence[str]) -> str:
    return shlex.join([str(part) for part in parts])


def _rematerialization_commands(
    selected: Sequence[dict[str, Any]],
    *,
    source_volume: str,
    target_volume: str,
    mirror_root: Path,
) -> list[str]:
    commands = []
    for entry in selected:
        ref = str(entry["checkpoint_ref"])
        local_path = mirror_root / ref
        commands.append(_quoted_command(["mkdir", "-p", str(local_path.parent)]))
        commands.append(
            _quoted_command(
                [
                    "uv",
                    "run",
                    "--extra",
                    "modal",
                    "modal",
                    "volume",
                    "get",
                    "--force",
                    source_volume,
                    ref,
                    str(local_path),
                ]
            )
        )
        commands.append(
            _quoted_command(
                [
                    "uv",
                    "run",
                    "--extra",
                    "modal",
                    "modal",
                    "volume",
                    "put",
                    "--force",
                    target_volume,
                    str(local_path),
                    ref,
                ]
            )
        )
    return commands


def _rating_command(
    *,
    refs_txt: Path,
    tournament_id: str,
    rating_run_id: str,
) -> list[str]:
    checkpoint_refs_expr = f"$(paste -sd, {shlex.quote(str(refs_txt))})"
    return [
        "REFS=" + checkpoint_refs_expr,
        _quoted_command(
            [
                "uv",
                "run",
                "--extra",
                "modal",
                "modal",
                "run",
                "--detach",
                "-m",
                "curvyzero.infra.modal.curvyzero_checkpoint_tournament",
                "--mode",
                "rating",
                "--tournament-id",
                tournament_id,
                "--rating-run-id",
                rating_run_id,
                "--checkpoint-refs",
                "$REFS",
                "--pair-selection",
                "adaptive_v0",
                "--pairs-per-round",
                "300",
                "--active-pool-limit",
                "100",
                "--games-per-pair",
                "21",
                "--games-per-shard",
                "1",
                "--max-steps",
                str(CURVYTRON_SOURCE_MAX_STEPS),
                "--decision-source-frames",
                "1",
                "--decision-ms",
                str(CURVYTRON_DECISION_MS),
                "--source-physics-step-ms",
                str(CURVYTRON_DECISION_MS),
                "--policy-trail-render-mode",
                CURVYTRON_POLICY_TRAIL_RENDER_MODE,
                "--policy-bonus-render-mode",
                CURVYTRON_POLICY_BONUS_RENDER_MODE,
                "--policy-mode",
                "eval",
                "--num-simulations",
                str(CURVYTRON_DEFAULT_NUM_SIMULATIONS),
                "--save-gif",
                "--gif-sample-games-per-pair",
                "5",
                "--gif-sample-strategy",
                "evenly_spaced",
            ]
        ).replace("'$REFS'", '"$REFS"'),
    ]


def _audit_refs_command(
    *,
    refs_txt: Path,
    volume_name: str,
    output_path: Path,
    allow_non_v2: bool = False,
) -> str:
    parts = [
        "uv",
        "run",
        "python",
        "scripts/audit_curvytron_launch_manifest_refs.py",
        "--refs-file",
        str(refs_txt),
        "--check-modal",
        "--runs-volume-name",
        volume_name,
        "--output",
        str(output_path),
    ]
    if allow_non_v2:
        parts.append("--allow-non-v2-runs-volume")
    return _quoted_command(parts)


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if not args.target_volume.endswith("-v2"):
        raise ValueError(f"target volume must be all-v2, got {args.target_volume!r}")
    snapshot = _load_snapshot(args.source_snapshot)
    rows = _select_rows(
        snapshot,
        limit=args.limit,
        required_status=args.required_status,
        min_iteration=int(args.min_iteration),
    )
    if len(rows) < args.limit and args.require_full_limit:
        raise ValueError(f"selected only {len(rows)} rows, expected {args.limit}")
    selected = [_selection_entry(row, ordinal=index) for index, row in enumerate(rows, start=1)]
    refs = [entry["checkpoint_ref"] for entry in selected]
    if len(refs) != len(set(refs)):
        raise ValueError("selected checkpoint refs are not unique")
    iterations = [
        int(entry["iteration"])
        for entry in selected
        if isinstance(entry.get("iteration"), int)
    ]

    output_dir = args.output_root / args.plan_id
    refs_txt = output_dir / "refs.txt"
    source_audit_path = output_dir / "source-refs-old-volume-audit.json"
    target_audit_path = output_dir / "source-refs-v2-target-after-copy-audit.json"
    mirror_root = args.local_mirror_root or Path("/private/tmp") / args.plan_id / "remat"
    remat_commands = _rematerialization_commands(
        selected,
        source_volume=args.source_volume,
        target_volume=args.target_volume,
        mirror_root=mirror_root,
    )
    rating_command = _rating_command(
        refs_txt=refs_txt,
        tournament_id=args.tournament_id,
        rating_run_id=args.rating_run_id,
    )
    return {
        "schema_id": SCHEMA_ID,
        "plan_id": args.plan_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_snapshot": str(args.source_snapshot),
        "source_leaderboard_id": snapshot.get("leaderboard_id"),
        "source_snapshot_id": snapshot.get("snapshot_id"),
        "source_context": snapshot.get("context"),
        "source_volume": args.source_volume,
        "target_volume": args.target_volume,
        "target_tournament_volume": args.tournament_volume,
        "selection": {
            "required_status": args.required_status,
            "limit": args.limit,
            "min_required_iteration": int(args.min_iteration),
            "selected_count": len(selected),
            "iteration_zero_count": sum(1 for value in iterations if value == 0),
            "min_iteration": min(iterations) if iterations else None,
            "max_iteration": max(iterations) if iterations else None,
        },
        "rerate": {
            "tournament_id": args.tournament_id,
            "rating_run_id": args.rating_run_id,
            "pair_selection": "adaptive_v0",
            "pairs_per_round": 300,
            "active_pool_limit": 100,
            "games_per_pair": 21,
            "games_per_shard": 1,
            "max_steps": CURVYTRON_SOURCE_MAX_STEPS,
            "decision_ms": CURVYTRON_DECISION_MS,
            "policy_trail_render_mode": CURVYTRON_POLICY_TRAIL_RENDER_MODE,
            "policy_bonus_render_mode": CURVYTRON_POLICY_BONUS_RENDER_MODE,
            "num_simulations": CURVYTRON_DEFAULT_NUM_SIMULATIONS,
            "save_gif": True,
            "gif_sample_games_per_pair": 5,
            "gif_sample_strategy": "evenly_spaced",
        },
        "output_refs": {
            "refs_txt": str(refs_txt),
            "selection_json": str(output_dir / "selection.json"),
            "commands_txt": str(output_dir / "commands.txt"),
        },
        "selected": selected,
        "commands": {
            "audit_source": [
                _audit_refs_command(
                    refs_txt=refs_txt,
                    volume_name=args.source_volume,
                    output_path=source_audit_path,
                    allow_non_v2=not args.source_volume.endswith("-v2"),
                )
            ],
            "rematerialize": remat_commands,
            "audit_target_after_copy": [
                _audit_refs_command(
                    refs_txt=refs_txt,
                    volume_name=args.target_volume,
                    output_path=target_audit_path,
                )
            ],
            "rating": rating_command,
        },
    }


def _write_plan(plan: dict[str, Any], *, output_root: Path) -> dict[str, str]:
    output_dir = output_root / str(plan["plan_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    refs_path = output_dir / "refs.txt"
    selection_path = output_dir / "selection.json"
    commands_path = output_dir / "commands.txt"
    refs_path.write_text(
        "\n".join(entry["checkpoint_ref"] for entry in plan["selected"]) + "\n",
        encoding="utf-8",
    )
    selection_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command_lines = [
        "# Review before running. First prove the selected old checkpoint refs exist.",
        *plan["commands"]["audit_source"],
        "",
        "# Copy selected old checkpoints into v2.",
        *plan["commands"]["rematerialize"],
        "",
        "# After the copy succeeds, prove every selected ref exists in v2.",
        *plan["commands"]["audit_target_after_copy"],
        "",
        "# After the copy succeeds, start the fresh all-v2 source rerate.",
        *plan["commands"]["rating"],
        "",
    ]
    commands_path.write_text("\n".join(command_lines), encoding="utf-8")
    return {
        "refs_txt": str(refs_path),
        "selection_json": str(selection_path),
        "commands_txt": str(commands_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_snapshot", type=Path)
    parser.add_argument("--plan-id", default=DEFAULT_PLAN_ID)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--required-status", default="active")
    parser.add_argument("--min-iteration", type=int, default=0)
    parser.add_argument("--require-full-limit", action="store_true", default=True)
    parser.add_argument("--allow-short-selection", dest="require_full_limit", action="store_false")
    parser.add_argument("--source-volume", default=DEFAULT_SOURCE_VOLUME)
    parser.add_argument("--target-volume", default=DEFAULT_CURVYTRON_RUNS_VOLUME_NAME)
    parser.add_argument("--tournament-volume", default=DEFAULT_CURVYTRON_TOURNAMENT_VOLUME_NAME)
    parser.add_argument("--tournament-id", default=DEFAULT_TOURNAMENT_ID)
    parser.add_argument("--rating-run-id", default=DEFAULT_RATING_RUN_ID)
    parser.add_argument("--local-mirror-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stdout-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plan = build_plan(args)
    outputs = {} if args.stdout_only else _write_plan(plan, output_root=args.output_root)
    print(json.dumps({"selected_count": len(plan["selected"]), **outputs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
