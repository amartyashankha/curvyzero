#!/usr/bin/env python
"""Compare 2P source/original state against CurvyZero raw visual observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvyzero.env.source_env import CurvyTronSourceEnv  # noqa: E402
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv  # noqa: E402
from curvyzero.env.vector_visual_observation import (  # noqa: E402
    SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY,
)
from curvyzero.env.vector_visual_observation import (  # noqa: E402
    SOURCE_STATE_GRAY64_RENDERER_IMPL_ID,
)
from curvyzero.env.vector_visual_observation import SOURCE_STATE_GRAY64_SCHEMA_ID  # noqa: E402
from curvyzero.env.vector_visual_observation import SOURCE_STATE_GRAY64_SHAPE  # noqa: E402
from curvyzero.env.vector_visual_observation import render_source_snapshot_gray64  # noqa: E402
from curvyzero.env.vector_visual_observation import render_source_state_gray64  # noqa: E402


DEFAULT_SCENARIO = (
    REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_lifecycle_long_1v1_no_bonus_wall_round_done.json"
)
SCHEMA_ID = "curvyzero_2p_raw_visual_observation_parity/v0"


def run_comparison(
    *,
    scenario_path: Path = DEFAULT_SCENARIO,
    max_steps: int | None = None,
    include_original_js_reset: bool = True,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run source-shaped 2P rollout and compare raw visual tensors.

    The compared raw frame is the learned observation raster:
    ``uint8[1,64,64]`` generated from source arena coordinates. It is not the
    original browser canvas.
    """

    scenario_path = Path(scenario_path)
    if not scenario_path.is_absolute():
        scenario_path = REPO_ROOT / scenario_path
    if max_steps is not None and int(max_steps) < 0:
        raise ValueError("max_steps must be non-negative")

    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    if int(scenario["player_count"]) != 2:
        raise ValueError("this comparison tool is intentionally 2P-only")

    source_setup = scenario["source_setup"]
    random_values = source_setup["random"]["math_random_sequence"]
    room = source_setup["room"]
    lifecycle = scenario["lifecycle"]
    rollout = scenario["rollout"]
    step_ms = float(rollout["step_ms"])
    advance_timers_ms = float(rollout["advance_timers_ms"])
    step_count = int(rollout["step_count"])
    if max_steps is not None:
        step_count = min(step_count, int(max_steps))
    source_moves = [int(move) for move in rollout["moves"]]
    action_ids = [_source_move_to_action_id(move) for move in source_moves]

    source_env = CurvyTronSourceEnv(
        random_values=random_values,
        max_score=float(room["max_score"]),
        include_deaths_snapshot=True,
        include_bonus_snapshot=True,
    )
    source_snapshot = source_env.reset(
        player_count=2,
        players=scenario["players"],
        warmup_ms=float(lifecycle["new_round_time_ms"]),
    )
    source_env.advance_timers(advance_timers_ms)
    source_snapshot = source_env.snapshot("after_reset_timer_advance")

    vector_env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        decision_ms=step_ms,
        max_score=int(room["max_score"]),
        random_tape_capacity=len(random_values),
        player_ids=tuple(str(player["id"]) for player in scenario["players"]),
    )
    vector_env.reset(
        seed=np.asarray([0], dtype=np.uint64),
        source_fixture_random_tape_values=np.asarray([random_values], dtype=np.float64),
        source_fixture_ref=str(Path(scenario_path).name),
        source_fixture_new_round_time_ms=float(lifecycle["new_round_time_ms"]),
        source_fixture_warmup_advance_ms=advance_timers_ms,
    )

    rows: list[dict[str, Any]] = []
    max_abs_diff = 0
    mismatch_pixels_total = 0
    first_mismatch: dict[str, Any] | None = None

    def compare(label: str, tick: int) -> tuple[np.ndarray, np.ndarray]:
        nonlocal first_mismatch
        nonlocal max_abs_diff
        nonlocal mismatch_pixels_total

        source_frame = render_source_snapshot_gray64(
            source_snapshot,
            world_bodies=source_env.world_bodies_snapshot(),
            bonus_bodies=source_env.bonus_bodies_snapshot(),
            avatar_body_metadata=source_env.avatar_body_metadata_snapshot(),
        )
        vector_frame = render_source_state_gray64(vector_env.state)
        diff = np.abs(
            source_frame.astype(np.int16, copy=False)
            - vector_frame.astype(np.int16, copy=False)
        )
        row_max = int(diff.max())
        row_mismatches = int(np.count_nonzero(diff))
        max_abs_diff = max(max_abs_diff, row_max)
        mismatch_pixels_total += row_mismatches
        row = {
            "label": label,
            "tick": int(tick),
            "source_nonzero_pixels": int(np.count_nonzero(source_frame)),
            "vector_nonzero_pixels": int(np.count_nonzero(vector_frame)),
            "max_abs_diff": row_max,
            "mismatch_pixels": row_mismatches,
        }
        if row_mismatches and first_mismatch is None:
            yx = np.argwhere(diff[0] != 0)[0]
            first_mismatch = {
                "label": label,
                "tick": int(tick),
                "channel": 0,
                "y": int(yx[0]),
                "x": int(yx[1]),
                "source": int(source_frame[0, yx[0], yx[1]]),
                "vector": int(vector_frame[0, yx[0], yx[1]]),
            }
        rows.append(row)
        return source_frame, vector_frame

    reset_source_frame, reset_vector_frame = compare("after_reset", 0)
    original_js_reset = (
        _original_js_reset_check(Path(scenario_path), reset_vector_frame)
        if include_original_js_reset
        else None
    )

    final_source_frame = reset_source_frame
    final_vector_frame = reset_vector_frame
    terminal_seen = False
    for tick in range(step_count):
        source_env.advance_timers(advance_timers_ms)
        source_snapshot = source_env.step(source_moves, elapsed_ms=step_ms)
        batch = vector_env.step(
            np.asarray([action_ids], dtype=np.int16),
            timer_advance_ms=advance_timers_ms,
        )
        final_source_frame, final_vector_frame = compare(f"after_step_{tick}", tick + 1)
        if bool(batch.done[0]):
            terminal_seen = True
            break

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_pgm(out_dir / "reset_source.pgm", reset_source_frame)
        _write_pgm(out_dir / "reset_vector.pgm", reset_vector_frame)
        _write_pgm(out_dir / "final_source.pgm", final_source_frame)
        _write_pgm(out_dir / "final_vector.pgm", final_vector_frame)
        _write_pgm(
            out_dir / "final_absdiff.pgm",
            np.abs(
                final_source_frame.astype(np.int16, copy=False)
                - final_vector_frame.astype(np.int16, copy=False)
            ).astype(np.uint8),
        )

    return {
        "schema_id": SCHEMA_ID,
        "scenario_id": scenario["scenario_id"],
        "scenario_path": _display_path(scenario_path),
        "source_arena_size": int(source_env.snapshot("report")["game"]["size"]),
        "raw_observation_schema_id": SOURCE_STATE_GRAY64_SCHEMA_ID,
        "raw_observation_shape": list(SOURCE_STATE_GRAY64_SHAPE),
        "raw_observation_dtype": "uint8",
        "renderer_impl_id": SOURCE_STATE_GRAY64_RENDERER_IMPL_ID,
        "browser_canvas_pixel_fidelity": SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY,
        "source_runner": "CurvyTronSourceEnv",
        "source_runner_claim": (
            "Python source-shaped env backed by JS-verified fixtures; not browser canvas"
        ),
        "original_js_reset_check": original_js_reset,
        "steps_requested": step_count,
        "frames_compared": len(rows),
        "terminal_seen": terminal_seen,
        "match": max_abs_diff == 0 and mismatch_pixels_total == 0,
        "max_abs_diff": max_abs_diff,
        "mismatch_pixels": mismatch_pixels_total,
        "first_mismatch": first_mismatch,
        "frames": rows,
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _original_js_reset_check(
    scenario_path: Path,
    vector_reset_frame: np.ndarray,
) -> dict[str, Any]:
    if shutil.which("node") is None:
        return {"available": False, "reason": "node executable is not available"}
    result = subprocess.run(
        [
            "node",
            str(REPO_ROOT / "tools" / "js_reuse_probe" / "curvytron_env_cli.js"),
            str(scenario_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    reset_state = payload["reset"]["state"]
    js_frame = render_source_snapshot_gray64(
        {
            "game": {"size": reset_state["game"]["size"]},
            "avatars": reset_state["players"],
        },
        world_bodies=(),
        bonus_bodies=(),
    )
    diff = np.abs(
        js_frame.astype(np.int16, copy=False)
        - vector_reset_frame.astype(np.int16, copy=False)
    )
    return {
        "available": True,
        "runner": "tools/js_reuse_probe/curvytron_env_cli.js",
        "loaded_original_source": bool(payload["capabilities"]["originalSourceLoaded"]),
        "source_arena_size": int(reset_state["game"]["size"]),
        "frames_compared": 1,
        "match": int(diff.max()) == 0 and int(np.count_nonzero(diff)) == 0,
        "max_abs_diff": int(diff.max()),
        "mismatch_pixels": int(np.count_nonzero(diff)),
        "scope": "reset-frame source-state raster only; not browser canvas pixels",
    }


def _source_move_to_action_id(move: int) -> int:
    if move == -1:
        return 0
    if move == 0:
        return 1
    if move == 1:
        return 2
    raise ValueError(f"unsupported source move {move!r}")


def _write_pgm(path: Path, frame: np.ndarray) -> None:
    image = np.asarray(frame)
    if image.shape != SOURCE_STATE_GRAY64_SHAPE or image.dtype != np.uint8:
        raise ValueError("PGM output expects uint8[1,64,64]")
    header = f"P5\n{image.shape[2]} {image.shape[1]}\n255\n".encode("ascii")
    path.write_bytes(header + image[0].tobytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--include-original-js-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_comparison(
        scenario_path=args.scenario,
        max_steps=args.max_steps,
        include_original_js_reset=bool(args.include_original_js_reset),
        out_dir=args.out_dir,
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    status = "PASS" if report["match"] else "FAIL"
    js = report["original_js_reset_check"]
    js_status = "skipped"
    if isinstance(js, dict) and js.get("available"):
        js_status = "pass" if js.get("match") else "fail"
    print(
        f"{status} {report['scenario_id']} "
        f"arena={report['source_arena_size']} raw={tuple(report['raw_observation_shape'])} "
        f"frames={report['frames_compared']} max_abs_diff={report['max_abs_diff']} "
        f"mismatch_pixels={report['mismatch_pixels']} js_reset={js_status} "
        f"browser_canvas_pixel_fidelity={report['browser_canvas_pixel_fidelity']}"
    )


if __name__ == "__main__":
    main()
