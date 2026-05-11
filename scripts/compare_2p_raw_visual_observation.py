#!/usr/bin/env python
"""Compare 2P source/original state against CurvyZero raw visual observations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from collections.abc import Sequence
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for root in (SCRIPT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import compare_vector_arrays_to_fidelity as vector_compare  # noqa: E402
import seed_vector_state_from_fixtures as seed_bridge  # noqa: E402
from curvyzero.env import vector_runtime  # noqa: E402
from curvyzero.env.source_env import SourceBodyState  # noqa: E402
from curvyzero.env.source_env import CurvyTronSourceEnv  # noqa: E402
from curvyzero.env.vector_multiplayer_env import (  # noqa: E402
    RANDOM_TAPE_SOURCE_SOURCE_FIXTURE,
)
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
NATURAL_BONUS_2P_FIXTURE_IDS = frozenset(
    {
        "source_bonus_spawn_type_position_rng_step",
        "source_bonus_spawn_game_world_retry_step",
        "source_bonus_spawn_bonus_world_retry_step",
        "source_bonus_spawn_cap_twenty_step",
    }
)
CORE_2P_SUITE = (
    DEFAULT_SCENARIO,
    REPO_ROOT / "scenarios" / "environment" / "forced_two_player_turn_step.json",
    REPO_ROOT / "scenarios" / "environment" / "source_kinematics_straight_step.json",
    REPO_ROOT / "scenarios" / "environment" / "source_kinematics_left_turn_step.json",
    REPO_ROOT / "scenarios" / "environment" / "source_kinematics_right_turn_step.json",
    REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_kinematics_straight_multistep.json",
    REPO_ROOT / "scenarios" / "environment" / "source_kinematics_turn_multistep.json",
    REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_kinematics_varied_elapsed_multistep.json",
    REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_live_movement_event_trace_2p_no_bonus_multistep.json",
    REPO_ROOT / "scenarios" / "environment" / "source_normal_wall_death_step.json",
    REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_normal_wall_same_frame_draw_step.json",
    REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_collision_death_point_kills_later_player_step.json",
    REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_collision_head_head_reverse_order_single_death_step.json",
    REPO_ROOT / "scenarios" / "environment" / "source_borderless_wrap_step.json",
    REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_borderless_exact_edge_corner_axis_step.json",
    REPO_ROOT / "scenarios" / "environment" / "source_bonus_self_small_catch_step.json",
    REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_bonus_self_small_tangent_no_catch_step.json",
    REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_bonus_self_small_wall_death_no_catch_step.json",
    REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_bonus_self_small_expiry_restore_step.json",
    REPO_ROOT / "scenarios" / "environment" / "source_bonus_game_clear_immediate_step.json",
    REPO_ROOT / "scenarios" / "environment" / "source_bonus_game_borderless_catch_step.json",
    REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_bonus_game_borderless_expiry_restore_step.json",
    REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_bonus_spawn_type_position_rng_step.json",
    REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_bonus_spawn_game_world_retry_step.json",
    REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_bonus_spawn_bonus_world_retry_step.json",
    REPO_ROOT
    / "scenarios"
    / "environment"
    / "source_bonus_spawn_cap_twenty_step.json",
)


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
    if _player_count(scenario) != 2:
        raise ValueError("this comparison tool is intentionally 2P-only")
    if "rollout" not in scenario:
        if _scenario_id(scenario) in NATURAL_BONUS_2P_FIXTURE_IDS:
            return _run_natural_bonus_spawn_comparison(
                scenario=scenario,
                scenario_path=scenario_path,
                max_steps=max_steps,
                out_dir=out_dir,
            )
        return _run_forced_state_comparison(
            scenario=scenario,
            scenario_path=scenario_path,
            max_steps=max_steps,
            out_dir=out_dir,
        )

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
        "scenario_id": _scenario_id(scenario),
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


def run_suite_comparison(
    paths: tuple[Path, ...] = CORE_2P_SUITE,
    *,
    max_steps: int | None = None,
) -> dict[str, Any]:
    reports = [
        run_comparison(
            scenario_path=path,
            max_steps=max_steps,
            include_original_js_reset=False,
        )
        for path in paths
    ]
    failed = [report for report in reports if not report["match"]]
    return {
        "schema_id": f"{SCHEMA_ID}_suite",
        "suite_id": "core_2p_source_state_gray64",
        "scenario_count": len(reports),
        "passed": len(reports) - len(failed),
        "failed": len(failed),
        "match": not failed,
        "max_abs_diff": max((int(report["max_abs_diff"]) for report in reports), default=0),
        "mismatch_pixels": sum(int(report["mismatch_pixels"]) for report in reports),
        "reports": reports,
    }


def _run_forced_state_comparison(
    *,
    scenario: dict[str, Any],
    scenario_path: Path,
    max_steps: int | None,
    out_dir: Path | None,
) -> dict[str, Any]:
    """Compare a forced-state fixture with one or more scripted steps."""

    steps = list(scenario.get("steps", []))
    if not steps:
        raise ValueError("forced-state comparison needs scenario.steps")
    if max_steps is not None:
        steps = steps[: int(max_steps)]

    source_env = _source_env_from_forced_scenario(scenario)
    source_snapshot = source_env.snapshot("after_forced_reset")

    fixture = seed_bridge.seed_fixture(scenario_path, body_capacity=16)
    state = vector_compare.array_state_from_seed(fixture)
    player_count = int(state["pos"].shape[1])
    if player_count != 2:
        raise ValueError("this comparison tool is intentionally 2P-only")
    first_step = vector_compare.prepare_fixture_array_step(fixture, step_index=0)
    decision_ms = float(first_step["step_ms"]) or 1.0
    vector_env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        decision_ms=decision_ms,
        body_capacity=int(state["body_active"].shape[1]),
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=max(8, int(state["random_tape_values"].shape[1])),
        event_mode="debug-event",
    )
    vector_env.reset_from_state_arrays(state, reset_seed=np.asarray([101], dtype=np.uint64))
    for bonus in _active_bonuses(scenario):
        vector_env.seed_active_bonus(
            row=0,
            bonus_type=str(bonus["type"]),
            x=float(bonus["x"]),
            y=float(bonus["y"]),
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
        rows.append(
            {
                "label": label,
                "tick": int(tick),
                "source_nonzero_pixels": int(np.count_nonzero(source_frame)),
                "vector_nonzero_pixels": int(np.count_nonzero(vector_frame)),
                "max_abs_diff": row_max,
                "mismatch_pixels": row_mismatches,
            }
        )
        return source_frame, vector_frame

    reset_source_frame, reset_vector_frame = compare("after_forced_reset", 0)
    final_source_frame = reset_source_frame
    final_vector_frame = reset_vector_frame
    terminal_seen = False
    for index, _step in enumerate(steps):
        prepared = vector_compare.prepare_fixture_array_step(fixture, step_index=index)
        step_ms = float(prepared["step_ms"])
        timer_advance_ms = float(prepared.get("timer_advance_ms", 0.0))
        source_moves = [int(move) for move in prepared["source_moves"]]
        source_env.advance_timers(timer_advance_ms)
        source_snapshot = source_env.step(source_moves, elapsed_ms=step_ms)
        if step_ms == 0.0:
            vector_env.decision_ms = 0.0
        else:
            vector_env.decision_ms = step_ms
        batch = vector_env.step(
            np.asarray([[move + 1 for move in source_moves]], dtype=np.int16),
            timer_advance_ms=timer_advance_ms,
        )
        final_source_frame, final_vector_frame = compare(f"after_step_{index}", index + 1)
        if bool(batch.done[0]):
            terminal_seen = True
            break

    if out_dir is not None:
        scenario_out_dir = out_dir / _scenario_id(scenario)
        scenario_out_dir.mkdir(parents=True, exist_ok=True)
        _write_pgm(scenario_out_dir / "reset_source.pgm", reset_source_frame)
        _write_pgm(scenario_out_dir / "reset_vector.pgm", reset_vector_frame)
        _write_pgm(scenario_out_dir / "final_source.pgm", final_source_frame)
        _write_pgm(scenario_out_dir / "final_vector.pgm", final_vector_frame)
        _write_pgm(
            scenario_out_dir / "final_absdiff.pgm",
            np.abs(
                final_source_frame.astype(np.int16, copy=False)
                - final_vector_frame.astype(np.int16, copy=False)
            ).astype(np.uint8),
        )

    return {
        "schema_id": SCHEMA_ID,
        "scenario_id": _scenario_id(scenario),
        "scenario_path": _display_path(scenario_path),
        "comparison_kind": "forced_state_fixture",
        "source_arena_size": int(source_env.snapshot("report")["game"]["size"]),
        "raw_observation_schema_id": SOURCE_STATE_GRAY64_SCHEMA_ID,
        "raw_observation_shape": list(SOURCE_STATE_GRAY64_SHAPE),
        "raw_observation_dtype": "uint8",
        "renderer_impl_id": SOURCE_STATE_GRAY64_RENDERER_IMPL_ID,
        "browser_canvas_pixel_fidelity": SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY,
        "source_runner": "CurvyTronSourceEnv",
        "source_runner_claim": (
            "Python source-shaped fixture runner for JS-oracle-pinned scenarios; "
            "not browser canvas"
        ),
        "original_js_reset_check": None,
        "steps_requested": len(steps),
        "frames_compared": len(rows),
        "terminal_seen": terminal_seen,
        "match": max_abs_diff == 0 and mismatch_pixels_total == 0,
        "max_abs_diff": max_abs_diff,
        "mismatch_pixels": mismatch_pixels_total,
        "first_mismatch": first_mismatch,
        "frames": rows,
    }


def _run_natural_bonus_spawn_comparison(
    *,
    scenario: dict[str, Any],
    scenario_path: Path,
    max_steps: int | None,
    out_dir: Path | None,
) -> dict[str, Any]:
    """Compare a natural bonus spawn/retry/cap fixture.

    These fixtures are not ordinary forced-state fixtures. Source reset uses a
    neutral random constant, then the fixture RNG tape starts at BonusManager.
    The vector side mirrors that setup and advances the natural bonus timer.
    """

    steps = list(scenario.get("steps", []))
    if not steps:
        raise ValueError("natural bonus comparison needs scenario.steps")
    if max_steps is not None:
        steps = steps[: int(max_steps)]

    source_setup = _mapping(scenario["source_setup"], "source_setup")
    room = _mapping(source_setup["room"], "source_setup.room")
    random_setup = _mapping(source_setup["random"], "source_setup.random")
    random_values = _random_values(random_setup["math_random_sequence"])
    players = _sequence(scenario["players"], "players")
    game_setup = _mapping(source_setup.get("game"), "source_setup.game", default={})

    source_env = CurvyTronSourceEnv(
        random_constant=0.5,
        max_score=float(room["max_score"]),
        include_deaths_snapshot=True,
        include_bonus_snapshot=True,
    )
    source_env.reset(
        player_count=_player_count(scenario),
        players=players,
        warmup_ms=0.0,
    )
    _apply_source_initial_players(source_env, players)
    if bool(game_setup.get("world_active", False)):
        if source_env.game is None or source_env.game.world is None:
            raise RuntimeError("source env world is not active")
        source_env.game.world.activate()
    for body in _initial_world_bodies(scenario):
        avatar_ids_by_player = _avatar_ids_by_player(players)
        owner_key = (
            body.get("player_id")
            or body.get("playerId")
            or body.get("avatar_id")
            or body.get("avatarId")
            or body.get("avatar")
        )
        if owner_key is None:
            raise ValueError("initial_state.world_bodies entries need a player id")
        _seed_source_body(
            source_env,
            owner_id=avatar_ids_by_player[str(owner_key)],
            x=float(body["x"]),
            y=float(body["y"]),
            radius=float(body["radius"]),
            num=int(body.get("num", 0)),
        )
    source_env.set_random_sequence(random_setup["math_random_sequence"])
    source_env.source_game_on_start(
        bonus_types=room["bonuses"],
        bonus_rate=float(room["bonus_rate"]),
    )
    for bonus in _active_bonuses(scenario):
        source_env.seed_active_bonus(
            str(bonus["type"]),
            x=float(bonus["x"]),
            y=float(bonus["y"]),
        )
    source_env.events.clear()
    source_snapshot = source_env.snapshot("after_natural_bonus_setup")

    vector_env = _vector_env_from_natural_bonus_scenario(
        scenario,
        scenario_path=scenario_path,
        random_values=random_values,
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
        rows.append(
            {
                "label": label,
                "tick": int(tick),
                "source_nonzero_pixels": int(np.count_nonzero(source_frame)),
                "vector_nonzero_pixels": int(np.count_nonzero(vector_frame)),
                "max_abs_diff": row_max,
                "mismatch_pixels": row_mismatches,
            }
        )
        return source_frame, vector_frame

    reset_source_frame, reset_vector_frame = compare("after_natural_bonus_setup", 0)
    final_source_frame = reset_source_frame
    final_vector_frame = reset_vector_frame
    terminal_seen = False
    for index, step in enumerate(steps):
        step_mapping = _mapping(step, f"steps[{index}]")
        step_ms = float(step_mapping["step_ms"])
        timer_advance_ms = float(step_mapping.get("advance_timers_ms", 0.0))
        source_moves = [
            int(move["move"])
            for move in _sequence(step_mapping["moves"], f"steps[{index}].moves")
        ]
        source_env.advance_timers(timer_advance_ms)
        source_snapshot = source_env.step(source_moves, elapsed_ms=step_ms)
        _set_vector_natural_bonus_due_in(vector_env, timer_advance_ms)
        vector_env.decision_ms = step_ms
        batch = vector_env.step(
            np.asarray([[move + 1 for move in source_moves]], dtype=np.int16),
            timer_advance_ms=timer_advance_ms,
        )
        final_source_frame, final_vector_frame = compare(f"after_step_{index}", index + 1)
        if bool(batch.done[0]):
            terminal_seen = True
            break

    if out_dir is not None:
        scenario_out_dir = out_dir / _scenario_id(scenario)
        scenario_out_dir.mkdir(parents=True, exist_ok=True)
        _write_pgm(scenario_out_dir / "reset_source.pgm", reset_source_frame)
        _write_pgm(scenario_out_dir / "reset_vector.pgm", reset_vector_frame)
        _write_pgm(scenario_out_dir / "final_source.pgm", final_source_frame)
        _write_pgm(scenario_out_dir / "final_vector.pgm", final_vector_frame)
        _write_pgm(
            scenario_out_dir / "final_absdiff.pgm",
            np.abs(
                final_source_frame.astype(np.int16, copy=False)
                - final_vector_frame.astype(np.int16, copy=False)
            ).astype(np.uint8),
        )

    return {
        "schema_id": SCHEMA_ID,
        "scenario_id": _scenario_id(scenario),
        "scenario_path": _display_path(scenario_path),
        "comparison_kind": "natural_bonus_spawn_fixture",
        "source_arena_size": int(source_env.snapshot("report")["game"]["size"]),
        "raw_observation_schema_id": SOURCE_STATE_GRAY64_SCHEMA_ID,
        "raw_observation_shape": list(SOURCE_STATE_GRAY64_SHAPE),
        "raw_observation_dtype": "uint8",
        "renderer_impl_id": SOURCE_STATE_GRAY64_RENDERER_IMPL_ID,
        "browser_canvas_pixel_fidelity": SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY,
        "source_runner": "CurvyTronSourceEnv",
        "source_runner_claim": (
            "Python source-shaped natural bonus fixture runner for JS-oracle-pinned "
            "scenarios; not browser canvas"
        ),
        "original_js_reset_check": None,
        "steps_requested": len(steps),
        "frames_compared": len(rows),
        "terminal_seen": terminal_seen,
        "match": max_abs_diff == 0 and mismatch_pixels_total == 0,
        "max_abs_diff": max_abs_diff,
        "mismatch_pixels": mismatch_pixels_total,
        "first_mismatch": first_mismatch,
        "source_final_bonus_count": len(source_env.active_bonuses),
        "vector_final_bonus_count": int(vector_env.state["bonus_count"][0]),
        "source_random_call_count": len(source_env.random_calls),
        "vector_random_tape_cursor": int(vector_env.state["random_tape_cursor"][0]),
        "vector_natural_bonus_pop_count": int(vector_env._natural_bonus_pop_count[0]),  # noqa: SLF001
        "frames": rows,
    }


def _vector_env_from_natural_bonus_scenario(
    scenario: Mapping[str, Any],
    *,
    scenario_path: Path,
    random_values: Sequence[float],
) -> VectorMultiplayerEnv:
    source_setup = _mapping(scenario["source_setup"], "source_setup")
    room = _mapping(source_setup["room"], "source_setup.room")
    players = _sequence(scenario["players"], "players")
    first_step = _mapping(_sequence(scenario["steps"], "steps")[0], "steps[0]")
    vector_env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        decision_ms=float(first_step["step_ms"]) or 1.0,
        max_score=int(room["max_score"]),
        body_capacity=16,
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=max(8, len(random_values)),
        natural_bonus_spawn=True,
        natural_bonus_type_codes=(vector_runtime.BONUS_TYPE_SELF_SMALL,),
        natural_bonus_capacity=vector_runtime.SOURCE_MAX_ACTIVE_BONUSES,
        natural_bonus_position_attempt_capacity=max(8, len(random_values)),
        event_mode="debug-event",
        player_ids=tuple(str(player["id"]) for player in players),
    )
    vector_env._ensure_seeded_bonus_arrays(  # noqa: SLF001
        bonus_capacity=vector_runtime.SOURCE_MAX_ACTIVE_BONUSES,
        stack_capacity=vector_runtime.SOURCE_MAX_ACTIVE_BONUSES,
    )
    for name, template in vector_env.reset_template.items():
        np.copyto(vector_env.state[name], template)
    vector_env._has_reset = True  # noqa: SLF001
    vector_env._needs_reset[:] = False  # noqa: SLF001
    vector_env._random_tape_source[0] = RANDOM_TAPE_SOURCE_SOURCE_FIXTURE  # noqa: SLF001
    vector_env._rng_impl_id[0] = "source_fixture_random_tape_values/v0"  # noqa: SLF001
    vector_env._source_fixture_ref[0] = str(Path(scenario_path).name)  # noqa: SLF001
    state = vector_env.state
    state["episode_id"][0] = 1
    state["round_id"][0] = 1
    state["env_active"][0] = True
    state["started"][0] = True
    state["in_round"][0] = True
    state["alive"][0, :] = True
    state["present"][0, :] = True
    state["random_tape_values"][0, :] = 0.0
    state["random_tape_values"][0, : len(random_values)] = np.asarray(
        random_values,
        dtype=np.float64,
    )
    state["random_tape_length"][0] = len(random_values)
    state["random_tape_cursor"][0] = 0
    state["random_tape_draw_count"][0] = 0
    state["random_tape_exhausted"][0] = False

    for index, player in enumerate(players):
        initial = _mapping(player.get("initial"), "player.initial")
        state["pos"][0, index] = (float(initial["x"]), float(initial["y"]))
        state["prev_pos"][0, index] = state["pos"][0, index]
        state["heading"][0, index] = float(initial["angle_rad"])
        state["printing"][0, index] = bool(initial.get("printing", False))

    game_setup = _mapping(source_setup.get("game"), "source_setup.game", default={})
    state["world_active"][0] = bool(game_setup.get("world_active", False))
    avatar_indices_by_player = _avatar_indices_by_player(players)
    for slot, body in enumerate(_initial_world_bodies(scenario)):
        owner_key = (
            body.get("player_id")
            or body.get("playerId")
            or body.get("avatar_id")
            or body.get("avatarId")
            or body.get("avatar")
        )
        if owner_key is None:
            raise ValueError("initial_state.world_bodies entries need a player id")
        state["body_active"][0, slot] = True
        state["body_pos"][0, slot] = (float(body["x"]), float(body["y"]))
        state["body_radius"][0, slot] = float(body["radius"])
        state["body_owner"][0, slot] = avatar_indices_by_player[str(owner_key)]
        state["body_num"][0, slot] = int(body.get("num", 0))
        state["body_insert_kind"][0, slot] = vector_runtime.BODY_KIND_NORMAL
        state["body_write_cursor"][0] = slot + 1
        owner_index = int(state["body_owner"][0, slot])
        state["body_count"][0, owner_index] = max(
            int(state["body_count"][0, owner_index]),
            int(body.get("num", 0)) + 1,
        )
    state["world_body_count"][0] = int(state["body_active"][0].sum())

    vector_env._natural_bonus_spawn_enabled[0] = True  # noqa: SLF001
    vector_env._clear_natural_bonus_spawn_rows(np.asarray([True], dtype=bool))  # noqa: SLF001
    vector_env._schedule_natural_bonus_pop_rows(  # noqa: SLF001
        np.asarray([True], dtype=bool),
        label="bonus.start_delay",
        delay_origin_ms=0.0,
    )
    for slot, bonus in enumerate(_active_bonuses(scenario)):
        _seed_vector_bonus_slot(vector_env, slot=slot, bonus=bonus)
    return vector_env


def _set_vector_natural_bonus_due_in(
    env: VectorMultiplayerEnv,
    advance_ms: float,
) -> None:
    env._natural_bonus_timer_active[0] = True  # noqa: SLF001
    env._natural_bonus_timer_remaining_ms[0] = float(advance_ms)  # noqa: SLF001
    env._natural_bonus_next_due_elapsed_ms[0] = (  # noqa: SLF001
        float(env.state["elapsed_ms"][0]) + float(advance_ms)
    )


def _source_env_from_forced_scenario(scenario: Mapping[str, Any]) -> CurvyTronSourceEnv:
    source_setup = _mapping(scenario["source_setup"], "source_setup")
    room = _mapping(source_setup["room"], "source_setup.room")
    random_setup = _mapping(source_setup["random"], "source_setup.random")
    if "math_random_sequence" in random_setup:
        env = CurvyTronSourceEnv(
            random_values=random_setup["math_random_sequence"],
            max_score=float(room["max_score"]),
            include_deaths_snapshot=True,
            include_bonus_snapshot=True,
        )
    else:
        env = CurvyTronSourceEnv(
            random_constant=float(random_setup.get("math_random", 0.5)),
            max_score=float(room["max_score"]),
            include_deaths_snapshot=True,
            include_bonus_snapshot=True,
        )
    game_setup = _mapping(source_setup.get("game"), "source_setup.game", default={})
    env.reset(
        player_count=_player_count(scenario),
        players=scenario["players"],
        warmup_ms=0.0,
        borderless=bool(game_setup.get("borderless", False)),
        bonus_types=room.get("bonuses", []),
        bonus_rate=float(room.get("bonus_rate", 0.0)),
    )
    env.advance_timers(0.0)
    if env.game is None:
        raise RuntimeError("source env reset did not create game")
    env.game.print_start_due_ms = None
    env.game.bonus_pop_due_ms = None
    env.game.started = bool(game_setup.get("started", env.game.started))
    env.game.in_round = bool(game_setup.get("in_round", env.game.in_round))
    env.game.borderless = bool(game_setup.get("borderless", env.game.borderless))
    env.game.world_active = bool(game_setup.get("world_active", env.game.world_active))
    if env.game.world is not None:
        env.game.world.active = env.game.world_active
    for player in _sequence(scenario["players"], "players"):
        initial = _mapping(player.get("initial"), "player.initial")
        avatar = env.set_avatar_state(
            int(player["avatar_id"]),
            x=float(initial["x"]),
            y=float(initial["y"]),
            angle=float(initial["angle_rad"]),
        )
        avatar.printing = bool(initial.get("printing", avatar.printing))
    initial_state = _mapping(scenario.get("initial_state"), "initial_state", default={})
    avatar_ids_by_player = _avatar_ids_by_player(scenario["players"])
    for body in _sequence(initial_state.get("world_bodies", []), "initial_state.world_bodies"):
        owner_key = (
            body.get("player_id")
            or body.get("playerId")
            or body.get("avatar_id")
            or body.get("avatarId")
            or body.get("avatar")
        )
        if owner_key is None:
            raise ValueError("initial_state.world_bodies entries need a player id")
        _seed_source_body(
            env,
            owner_id=avatar_ids_by_player[str(owner_key)],
            x=float(body["x"]),
            y=float(body["y"]),
            radius=float(body["radius"]),
            num=int(body.get("num", 0)),
        )
    for bonus in _active_bonuses(scenario):
        env.seed_active_bonus(str(bonus["type"]), x=float(bonus["x"]), y=float(bonus["y"]))
    env.events.clear()
    env.random.calls.clear()
    return env


def _apply_source_initial_players(
    env: CurvyTronSourceEnv,
    players: Sequence[Mapping[str, Any]],
) -> None:
    for player in _sequence(players, "players"):
        initial = _mapping(player.get("initial"), "player.initial")
        avatar = env.set_avatar_state(
            int(player["avatar_id"]),
            x=float(initial["x"]),
            y=float(initial["y"]),
            angle=float(initial["angle_rad"]),
        )
        avatar.printing = bool(initial.get("printing", avatar.printing))


def _scenario_id(scenario: Mapping[str, Any]) -> str:
    value = scenario.get("scenario_id", scenario.get("id"))
    if not isinstance(value, str) or not value:
        raise ValueError("scenario needs scenario_id or id")
    return value


def _player_count(scenario: Mapping[str, Any]) -> int:
    if "player_count" in scenario:
        return int(scenario["player_count"])
    source_setup = _mapping(scenario.get("source_setup"), "source_setup", default={})
    if "player_count" in source_setup:
        return int(source_setup["player_count"])
    players = scenario.get("players")
    if isinstance(players, Sequence) and not isinstance(players, (str, bytes, bytearray)):
        return len(players)
    raise ValueError("scenario needs player_count")


def _seed_source_body(
    env: CurvyTronSourceEnv,
    *,
    owner_id: int,
    x: float,
    y: float,
    radius: float,
    num: int,
) -> None:
    if env.game is None or env.game.world is None:
        raise RuntimeError("source env world is not active")
    owner = env.avatar_by_id(owner_id)
    body = SourceBodyState(
        x=x,
        y=y,
        radius=radius,
        avatar_id=owner.id,
        num=num,
        birth_ms=env.now_ms,
        trail_latency=owner.trail_latency,
    )
    env.game.world.add_body(body)
    env.game.world_body_count = env.game.world.body_count
    owner.body_count = max(owner.body_count, num + 1)


def _active_bonuses(scenario: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    initial_state = _mapping(scenario.get("initial_state"), "initial_state", default={})
    return _sequence(initial_state.get("active_bonuses", []), "initial_state.active_bonuses")


def _initial_world_bodies(scenario: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    initial_state = _mapping(scenario.get("initial_state"), "initial_state", default={})
    return _sequence(initial_state.get("world_bodies", []), "initial_state.world_bodies")


def _avatar_ids_by_player(players: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for player in _sequence(players, "players"):
        avatar_id = int(player["avatar_id"])
        for key in ("id", "player_id", "playerId", "avatar_id", "avatarId", "name"):
            if key in player:
                result[str(player[key])] = avatar_id
    return result


def _avatar_indices_by_player(players: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, player in enumerate(_sequence(players, "players")):
        for key in ("id", "player_id", "playerId", "avatar_id", "avatarId", "name"):
            if key in player:
                result[str(player[key])] = index
    return result


def _seed_vector_bonus_slot(
    env: VectorMultiplayerEnv,
    *,
    slot: int,
    bonus: Mapping[str, Any],
) -> None:
    state = env.state
    bonus_code = _bonus_type_code(str(bonus["type"]))
    bonus_id = slot + 1
    state["bonus_world_active"][0] = True
    state["bonus_active"][0, slot] = True
    state["bonus_type"][0, slot] = bonus_code
    state["bonus_id"][0, slot] = bonus_id
    state["bonus_pos"][0, slot] = (float(bonus["x"]), float(bonus["y"]))
    state["bonus_radius"][0, slot] = vector_runtime.SOURCE_BONUS_RADIUS
    state["bonus_count"][0] = int(state["bonus_active"][0].sum())
    state["bonus_world_body_count"][0] = int(state["bonus_count"][0])
    state["bonus_next_id"][0] = max(int(state["bonus_next_id"][0]), bonus_id + 1)


def _bonus_type_code(name: str) -> int:
    try:
        return int(vector_runtime.BONUS_TYPE_NAME_BY_CODE.index(name))
    except ValueError as exc:
        raise ValueError(f"unsupported source bonus type {name!r}") from exc


def _random_values(values: Sequence[Any]) -> list[float]:
    result: list[float] = []
    for index, value in enumerate(values):
        raw_value = value.get("value") if isinstance(value, Mapping) else value
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise ValueError(f"random value {index} must be numeric")
        number = float(raw_value)
        if not np.isfinite(number) or number < 0.0 or number >= 1.0:
            raise ValueError(f"random value {index} must be finite in [0, 1)")
        result.append(number)
    return result


def _mapping(value: Any, name: str, *, default: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    if value is None and default is not None:
        return default
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence")
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{name}[{index}] must be an object")
    return value


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
    parser.add_argument("--suite", choices=("core2p",), default=None)
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
    if args.suite == "core2p":
        report = run_suite_comparison(max_steps=args.max_steps)
        if args.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
            return
        status = "PASS" if report["match"] else "FAIL"
        print(
            f"{status} {report['suite_id']} scenarios={report['scenario_count']} "
            f"passed={report['passed']} failed={report['failed']} "
            f"max_abs_diff={report['max_abs_diff']} "
            f"mismatch_pixels={report['mismatch_pixels']}"
        )
        return
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
