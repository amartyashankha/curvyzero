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
from typing import Callable

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
    SOURCE_STATE_BONUS64_STACK4_BONUS_MASK_CHANNEL,
)
from curvyzero.env.vector_visual_observation import (  # noqa: E402
    SOURCE_STATE_BONUS64_STACK4_BONUS_TYPE_CHANNEL,
)
from curvyzero.env.vector_visual_observation import (  # noqa: E402
    SOURCE_STATE_BONUS64_STACK4_GAME_BORDERLESS_CHANNEL,
)
from curvyzero.env.vector_visual_observation import (  # noqa: E402
    SOURCE_STATE_BONUS64_STACK4_GAME_TTL_CHANNEL,
)
from curvyzero.env.vector_visual_observation import (  # noqa: E402
    SOURCE_STATE_BONUS64_STACK4_MAX_BONUS_TYPE_CODE,
)
from curvyzero.env.vector_visual_observation import (  # noqa: E402
    SOURCE_STATE_BONUS64_STACK4_MAX_TTL_MS,
)
from curvyzero.env.vector_visual_observation import (  # noqa: E402
    SOURCE_STATE_BONUS64_STACK4_OTHER_STATUS_CHANNELS,
)
from curvyzero.env.vector_visual_observation import (  # noqa: E402
    SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_SCHEMA_ID,
)
from curvyzero.env.vector_visual_observation import (  # noqa: E402
    SOURCE_STATE_BONUS64_STACK4_SELF_STATUS_CHANNELS,
)
from curvyzero.env.vector_visual_observation import (  # noqa: E402
    SOURCE_STATE_CANVAS_GRAY64_BROWSER_PIXEL_FIDELITY as SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY,
)
from curvyzero.env.vector_visual_observation import (  # noqa: E402
    SOURCE_STATE_CANVAS_GRAY64_RENDERER_IMPL_ID as SOURCE_STATE_GRAY64_RENDERER_IMPL_ID,
)
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID as SOURCE_STATE_GRAY64_SCHEMA_ID  # noqa: E402
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SHAPE as SOURCE_STATE_GRAY64_SHAPE  # noqa: E402
from curvyzero.env.vector_visual_observation import (  # noqa: E402
    render_source_state_bonus64_stack4_player_perspective_v1,
)
from curvyzero.env.vector_visual_observation import render_source_snapshot_canvas_gray64 as render_source_snapshot_gray64  # noqa: E402
from curvyzero.env.vector_visual_observation import render_source_state_canvas_gray64 as render_source_state_gray64  # noqa: E402


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
PROGRAMMATIC_2P_STRESS_SCENARIO_IDS = (
    "source_printing_trail_point_visual_stress",
    "source_body_opponent_tangent_then_overlap_visual_stress",
    "source_body_own_latency_delta3_then_delta4_visual_stress",
    "source_print_manager_trail_gap_boundary_visual_stress",
    "source_lifecycle_survivor_score_2p_warmdown_visual_stress",
    "source_bonus_self_master_body_block_then_wall_death_visual_stress",
    "source_bonus_game_borderless_expiry_then_wall_death_visual_stress",
    "source_bonus_game_borderless_same_frame_expiry_wall_death_visual_stress",
    "source_bonus_game_clear_clears_future_collision_body_visual_stress",
)
VISUAL_MISMATCH_CANARY_IDS = (
    "missing_visible_world_body",
    "missing_visible_map_bonus",
)
TYPED_BONUS_VISUAL_STATUS_GATE_ID = "source_default_bonus64_type_status_planes_2p"
FINAL_OBSERVATION_AUTORESET_GATE_ID = (
    "source_state_canvas_gray64_final_observation_before_reset_2p"
)
FULL_2P_VISUAL_GATE_ID = "full_2p_source_state_visual_gate"
TYPED_BONUS_VISUAL_STATUS_GATE_TYPES = tuple(
    vector_runtime.BONUS_TYPE_NAME_BY_CODE[int(code)]
    for code in vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES
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
    include_programmatic_stress: bool | None = None,
) -> dict[str, Any]:
    if include_programmatic_stress is None:
        include_programmatic_stress = tuple(paths) == CORE_2P_SUITE
    reports = [
        run_comparison(
            scenario_path=path,
            max_steps=max_steps,
            include_original_js_reset=False,
        )
        for path in paths
    ]
    if include_programmatic_stress:
        reports.extend(
            run_programmatic_stress_comparison(
                scenario_id,
                max_steps=max_steps,
            )
            for scenario_id in PROGRAMMATIC_2P_STRESS_SCENARIO_IDS
        )
    failed = [report for report in reports if not report["match"]]
    return {
        "schema_id": f"{SCHEMA_ID}_suite",
        "suite_id": "core_2p_source_state_canvas_gray64",
        "scenario_count": len(reports),
        "passed": len(reports) - len(failed),
        "failed": len(failed),
        "match": not failed,
        "max_abs_diff": max((int(report["max_abs_diff"]) for report in reports), default=0),
        "mismatch_pixels": sum(int(report["mismatch_pixels"]) for report in reports),
        "reports": reports,
    }


def run_programmatic_stress_comparison(
    scenario_id: str,
    *,
    max_steps: int | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run a source-env-backed visual stress case that has no JSON fixture.

    These cases use CurvyTronSourceEnv snapshots as truth and mirror the same
    state in VectorMultiplayerEnv. They intentionally do not assert source
    expectations beyond source-vs-vector gray64 parity for the constructed
    state sequence.
    """

    if max_steps is not None and int(max_steps) < 0:
        raise ValueError("max_steps must be non-negative")
    if scenario_id == "source_printing_trail_point_visual_stress":
        return _run_printing_trail_point_visual_stress(
            max_steps=max_steps,
            out_dir=out_dir,
        )
    if scenario_id == "source_body_opponent_tangent_then_overlap_visual_stress":
        return _run_body_opponent_tangent_then_overlap_visual_stress(
            max_steps=max_steps,
            out_dir=out_dir,
        )
    if scenario_id == "source_body_own_latency_delta3_then_delta4_visual_stress":
        return _run_body_own_latency_delta3_then_delta4_visual_stress(
            max_steps=max_steps,
            out_dir=out_dir,
        )
    if scenario_id == "source_print_manager_trail_gap_boundary_visual_stress":
        return _run_print_manager_trail_gap_boundary_visual_stress(
            max_steps=max_steps,
            out_dir=out_dir,
        )
    if scenario_id == "source_lifecycle_survivor_score_2p_warmdown_visual_stress":
        return _run_lifecycle_survivor_score_2p_next_round_visual_stress(
            max_steps=max_steps,
            out_dir=out_dir,
        )
    if scenario_id == "source_bonus_self_master_body_block_then_wall_death_visual_stress":
        return _run_bonus_self_master_body_block_then_wall_death_visual_stress(
            max_steps=max_steps,
            out_dir=out_dir,
        )
    if scenario_id == "source_bonus_game_borderless_expiry_then_wall_death_visual_stress":
        return _run_bonus_game_borderless_expiry_then_wall_death_visual_stress(
            max_steps=max_steps,
            out_dir=out_dir,
        )
    if scenario_id == "source_bonus_game_borderless_same_frame_expiry_wall_death_visual_stress":
        return _run_bonus_game_borderless_same_frame_expiry_wall_death_visual_stress(
            max_steps=max_steps,
            out_dir=out_dir,
        )
    if scenario_id == "source_bonus_game_clear_clears_future_collision_body_visual_stress":
        return _run_bonus_game_clear_clears_future_collision_body_visual_stress(
            max_steps=max_steps,
            out_dir=out_dir,
        )
    raise ValueError(f"unknown programmatic stress scenario {scenario_id!r}")


def run_visual_mismatch_canary(canary_id: str) -> dict[str, Any]:
    """Prove the visual comparison fails when a visible source fact is missing."""

    if canary_id == "missing_visible_world_body":
        source_env, vector_env = _manual_2p_stress_env(
            positions=((20.0, 20.0), (70.0, 70.0)),
            headings=(0.0, 0.0),
            world_bodies=(
                {"owner_id": 2, "x": 25.0, "y": 20.0, "radius": 1.2, "num": 0},
            ),
        )
        state = vector_env.state
        state["body_active"][0, 0] = False
        state["body_pos"][0, 0] = 0.0
        state["body_radius"][0, 0] = 0.0
        state["world_body_count"][0] = 0
        state["body_count"][0, 1] = 0
        expected_hole = "world body missing from vector state"
    elif canary_id == "missing_visible_map_bonus":
        source_env, vector_env = _manual_2p_stress_env(
            positions=((20.0, 20.0), (70.0, 70.0)),
            headings=(0.0, 0.0),
            active_bonus={"type": "BonusSelfSmall", "x": 25.0, "y": 20.0},
        )
        state = vector_env.state
        state["bonus_active"][0, :] = False
        state["bonus_type"][0, :] = vector_runtime.BONUS_TYPE_NONE
        state["bonus_pos"][0, :, :] = 0.0
        state["bonus_count"][0] = 0
        expected_hole = "map bonus missing from vector state"
    else:
        raise ValueError(f"unknown visual mismatch canary {canary_id!r}")

    return _single_frame_visual_report(
        report_id=canary_id,
        comparison_kind="intentional_visual_mismatch_canary",
        source_env=source_env,
        vector_env=vector_env,
        label="after_intentional_mismatch",
        extra={
            "expected_hole": expected_hole,
            "expected_match": False,
        },
    )


def run_typed_bonus_visual_status_gate() -> dict[str, Any]:
    """Compare source bonus identities/status against the v1 typed planes.

    The gray64 frame intentionally renders all map bonuses as one grayscale
    value and omits bonus status. This gate keeps the promoted v1 mask/type
    channels and post-catch status channels honest for every source-default
    bonus type in a 2P row.
    """

    reports = [
        _run_typed_bonus_visual_status_case(bonus_type)
        for bonus_type in TYPED_BONUS_VISUAL_STATUS_GATE_TYPES
    ]
    failed = [report for report in reports if not report["match"]]
    return {
        "schema_id": f"{SCHEMA_ID}_bonus64_typed_gate",
        "gate_id": TYPED_BONUS_VISUAL_STATUS_GATE_ID,
        "comparison_kind": "typed_bonus_visual_status_gate",
        "visual_observation_schema_id": SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_SCHEMA_ID,
        "source_bonus_types": list(TYPED_BONUS_VISUAL_STATUS_GATE_TYPES),
        "source_backed_status_scope": [
            "active map bonus mask",
            "active map bonus type code",
            "post-catch radius ratio",
            "post-catch speed ratio",
            "post-catch inverse flag",
            "post-catch turn override flag",
            "post-catch invincible flag",
            "post-catch printing flag",
            "post-catch player bonus ttl",
            "post-catch game borderless flag",
            "post-catch game bonus ttl",
        ],
        "missing_source_backed_proof": [
            "BonusAllColor post-catch color rotation is not encoded in bonus64 v1 status planes",
            "BonusGameClear has no typed status plane after catch; clear geometry remains covered by gray64/runtime gates",
        ],
        "case_count": len(reports),
        "passed": len(reports) - len(failed),
        "failed": len(failed),
        "match": not failed,
        "max_abs_diff": max(
            (float(report["max_abs_diff"]) for report in reports),
            default=0.0,
        ),
        "mismatch_pixels": sum(int(report["mismatch_pixels"]) for report in reports),
        "reports": reports,
    }


def run_final_observation_autoreset_visual_gate(
    *,
    terminal_seed: int = 21,
    reset_seed: int = 22,
) -> dict[str, Any]:
    """Prove the source-state visual final observation is copied before reset.

    The public vector env exposes debug metadata as its native observation; the
    model-facing source-state visual adapter owns the canvas-gray64 stack. This
    gate keeps the adapter's terminal stack tied to the terminal source-state
    render and verifies a later reset does not mutate the copied final payload.
    """

    from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (  # noqa: E501
        CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv,
    )
    from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
        STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
    )
    from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (
        STACKED_SOURCE_STATE_GRAY64_SHAPE,
    )

    env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(
        {
            "seed": int(terminal_seed),
            "source_max_steps": 1,
        }
    )
    reset_observation = env.reset(seed=int(terminal_seed))
    timestep = env.step(0)
    final_observation = timestep.info.get("final_observation")
    terminal_gray64 = env.render("source_state_grayscale64_visual_tensor")
    if terminal_gray64 is None:
        raise RuntimeError("source-state visual env did not expose terminal gray64 render")

    final_stack = (
        None
        if not isinstance(final_observation, Mapping)
        else np.asarray(final_observation.get("observation"), dtype=np.float32)
    )
    final_action_mask = (
        None
        if not isinstance(final_observation, Mapping)
        else np.asarray(final_observation.get("action_mask"))
    )
    returned_stack = np.asarray(timestep.obs["observation"], dtype=np.float32)
    expected_terminal_frame = (
        terminal_gray64[0].astype(np.float32) * np.float32(1.0 / 255.0)
    )
    if final_stack is None:
        terminal_frame_diff = np.ones((64, 64), dtype=np.float32)
        returned_stack_diff = np.ones(STACKED_SOURCE_STATE_GRAY64_SHAPE, dtype=np.float32)
        final_stack_before_reset = None
    else:
        terminal_frame_diff = np.abs(final_stack[-1] - expected_terminal_frame)
        returned_stack_diff = np.abs(final_stack - returned_stack)
        final_stack_before_reset = final_stack.copy()

    post_reset_gray64 = terminal_gray64.copy()
    post_reset_observation = reset_observation
    post_reset_seed = int(reset_seed)
    post_reset_frame_diff_pixels = 0
    for candidate_seed in range(int(reset_seed), int(reset_seed) + 8):
        post_reset_seed = candidate_seed
        post_reset_observation = env.reset(seed=candidate_seed)
        rendered = env.render("source_state_grayscale64_visual_tensor")
        if rendered is None:
            raise RuntimeError("source-state visual env did not expose reset gray64 render")
        post_reset_gray64 = rendered
        post_reset_frame_diff_pixels = int(np.count_nonzero(post_reset_gray64 != terminal_gray64))
        if post_reset_frame_diff_pixels:
            break

    final_stack_after_reset = (
        None
        if not isinstance(final_observation, Mapping)
        else np.asarray(final_observation.get("observation"), dtype=np.float32)
    )
    final_stack_survived_reset = (
        final_stack_before_reset is not None
        and final_stack_after_reset is not None
        and np.array_equal(final_stack_before_reset, final_stack_after_reset)
    )
    action_mask_terminal = (
        final_action_mask is not None
        and np.array_equal(final_action_mask, np.zeros(3, dtype=np.int8))
    )
    terminal_frame_max_abs_diff = float(terminal_frame_diff.max())
    terminal_frame_mismatch_pixels = int(np.count_nonzero(terminal_frame_diff > 1e-7))
    returned_stack_max_abs_diff = float(returned_stack_diff.max())
    returned_stack_mismatch_pixels = int(np.count_nonzero(returned_stack_diff > 1e-7))
    mismatch_pixels = terminal_frame_mismatch_pixels + returned_stack_mismatch_pixels
    match = (
        bool(timestep.done)
        and isinstance(final_observation, Mapping)
        and final_stack is not None
        and tuple(final_stack.shape) == STACKED_SOURCE_STATE_GRAY64_SHAPE
        and action_mask_terminal
        and terminal_frame_mismatch_pixels == 0
        and returned_stack_mismatch_pixels == 0
        and final_stack_survived_reset
        and post_reset_frame_diff_pixels > 0
    )
    return {
        "schema_id": f"{SCHEMA_ID}_final_observation_autoreset_gate",
        "gate_id": FINAL_OBSERVATION_AUTORESET_GATE_ID,
        "comparison_kind": "source_state_canvas_gray64_final_observation_autoreset_gate",
        "visual_observation_schema_id": STACKED_SOURCE_STATE_GRAY64_SCHEMA_ID,
        "single_frame_schema_id": SOURCE_STATE_GRAY64_SCHEMA_ID,
        "raw_observation_schema_id": SOURCE_STATE_GRAY64_SCHEMA_ID,
        "source_state_canvas_gray64_schema_id": SOURCE_STATE_GRAY64_SCHEMA_ID,
        "source_runner": "CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv",
        "source_runner_claim": (
            "model-facing source-state canvas-gray64 adapter over VectorMultiplayerEnv; "
            "not browser canvas"
        ),
        "terminal_seed": int(terminal_seed),
        "post_reset_seed": int(post_reset_seed),
        "reset_after_terminal_api": "reset(seed=...)",
        "autoreset_claim": (
            "final_observation is copied from the terminal visual stack before "
            "any reset/autoreset mutates current source-state visuals"
        ),
        "terminal_seen": bool(timestep.done),
        "terminal_reason": str(timestep.info.get("terminal_reason")),
        "final_observation_present": isinstance(final_observation, Mapping),
        "final_observation_shape": (
            None if final_stack is None else list(final_stack.shape)
        ),
        "expected_final_observation_shape": list(STACKED_SOURCE_STATE_GRAY64_SHAPE),
        "final_action_mask_terminal": action_mask_terminal,
        "terminal_frame_max_abs_diff": terminal_frame_max_abs_diff,
        "terminal_frame_mismatch_pixels": terminal_frame_mismatch_pixels,
        "returned_stack_max_abs_diff": returned_stack_max_abs_diff,
        "returned_stack_mismatch_pixels": returned_stack_mismatch_pixels,
        "final_stack_survived_reset": final_stack_survived_reset,
        "post_reset_frame_mismatch_pixels": post_reset_frame_diff_pixels,
        "post_reset_current_observation_shape": list(
            np.asarray(post_reset_observation["observation"]).shape
        ),
        "match": match,
        "max_abs_diff": max(terminal_frame_max_abs_diff, returned_stack_max_abs_diff),
        "mismatch_pixels": mismatch_pixels,
        "browser_canvas_pixel_fidelity": SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY,
        "visual_limits": (
            "covers source-state canvas-gray64 final-observation copy semantics; "
            "does not prove replay serialization or real browser canvas pixels"
        ),
    }


def run_full_2p_visual_gate(*, max_steps: int | None = None) -> dict[str, Any]:
    """Run the full current 2P model-facing visual gate.

    This combines the exact gray64 source-state comparison, the typed
    bonus/status companion gate, and the expected-failure visual canaries. It is
    intentionally still not a browser-canvas or full trainer/replay claim.
    """

    gray64 = run_suite_comparison(max_steps=max_steps)
    typed_bonus = run_typed_bonus_visual_status_gate()
    final_observation = run_final_observation_autoreset_visual_gate()
    canaries = [run_visual_mismatch_canary(canary_id) for canary_id in VISUAL_MISMATCH_CANARY_IDS]
    canary_passes = [
        (not report["match"])
        and int(report["max_abs_diff"]) > 0
        and int(report["mismatch_pixels"]) > 0
        for report in canaries
    ]
    failed_gates: list[str] = []
    if not gray64["match"]:
        failed_gates.append("gray64_source_state")
    if not typed_bonus["match"]:
        failed_gates.append("bonus64_typed_status")
    if not final_observation["match"]:
        failed_gates.append("final_observation_autoreset")
    if not all(canary_passes):
        failed_gates.append("visual_expected_failure_canaries")

    return {
        "schema_id": f"{SCHEMA_ID}_full_visual_gate",
        "gate_id": FULL_2P_VISUAL_GATE_ID,
        "comparison_kind": "full_2p_model_facing_visual_gate",
        "scope": "2P source-state model-facing visual observation; not browser canvas; not trainer/replay propagation",
        "not_a_training_ready_claim": True,
        "remaining_training_work": [
            "use one product path: source-state browser-like RGB64 raw frame -> grayscale64 -> stack",
            "keep bonus64 v1 as a diagnostic/proof tensor, not a training observation path",
            "prove the canvas-gray64 observation survives replay serialization paths",
            "add source/original fixtures for programmatic bonus stack/death stress probes",
        ],
        "match": not failed_gates,
        "failed_gates": failed_gates,
        "gray64": gray64,
        "typed_bonus": typed_bonus,
        "final_observation": final_observation,
        "visual_canaries": canaries,
        "visual_canary_count": len(canaries),
        "visual_canaries_passed": sum(1 for passed in canary_passes if passed),
        "mismatch_pixels": int(gray64["mismatch_pixels"])
        + int(typed_bonus["mismatch_pixels"])
        + int(final_observation["mismatch_pixels"]),
        "max_abs_diff": max(
            float(gray64["max_abs_diff"]),
            float(typed_bonus["max_abs_diff"]),
            float(final_observation["max_abs_diff"]),
        ),
        "expected_canary_mismatch_pixels": sum(
            int(report["mismatch_pixels"]) for report in canaries
        ),
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


def _run_bonus_self_master_body_block_then_wall_death_visual_stress(
    *,
    max_steps: int | None,
    out_dir: Path | None,
) -> dict[str, Any]:
    source_env, vector_env = _manual_2p_stress_env(
        positions=((20.0, 20.0), (70.0, 70.0)),
        headings=(0.0, 0.0),
        active_bonus={"type": "BonusSelfMaster", "x": 20.0, "y": 20.0},
    )

    def catch_bonus() -> tuple[Mapping[str, Any], Any]:
        return _step_stress_pair(source_env, vector_env, step_ms=0.0)

    def body_collision_is_blocked() -> tuple[Mapping[str, Any], Any]:
        _seed_source_body(
            source_env,
            owner_id=2,
            x=20.0,
            y=21.0,
            radius=1.0,
            num=0,
        )
        _seed_vector_world_body(
            vector_env,
            slot=0,
            owner_index=1,
            x=20.0,
            y=21.0,
            radius=1.0,
            num=0,
        )
        return _step_stress_pair(source_env, vector_env, step_ms=0.0)

    def later_wall_death() -> tuple[Mapping[str, Any], Any]:
        _set_stress_avatar_state(
            source_env,
            vector_env,
            player_index=0,
            x=0.3,
            y=20.0,
            angle=np.pi,
        )
        return _step_stress_pair(source_env, vector_env, step_ms=0.0)

    return _run_programmatic_stress_case(
        scenario_id="source_bonus_self_master_body_block_then_wall_death_visual_stress",
        source_env=source_env,
        vector_env=vector_env,
        steps=(
            ("after_bonus_self_master_catch", catch_bonus),
            ("after_invincible_body_collision_probe", body_collision_is_blocked),
            ("after_later_wall_death", later_wall_death),
        ),
        max_steps=max_steps,
        out_dir=out_dir,
        expected_terminal=True,
        visual_limits=(
            "gray64 does not encode the SelfMaster stack or invincible flag; it "
            "only exposes the surviving live head after the body hit and the "
            "later wall-death geometry"
        ),
    )


def _run_printing_trail_point_visual_stress(
    *,
    max_steps: int | None,
    out_dir: Path | None,
) -> dict[str, Any]:
    source_env, vector_env = _manual_2p_stress_env(
        positions=((20.0, 20.0), (70.0, 70.0)),
        headings=(0.0, 0.0),
    )
    _set_stress_avatar_state(
        source_env,
        vector_env,
        player_index=0,
        x=0.0,
        y=20.0,
        angle=0.0,
    )
    _set_stress_avatar_printing(
        source_env,
        vector_env,
        player_index=0,
        printing=True,
        last_x=0.0,
        last_y=20.0,
    )

    def emit_visible_trail_point() -> tuple[Mapping[str, Any], Any]:
        return _step_stress_pair(source_env, vector_env, step_ms=38.0)

    return _run_programmatic_stress_case(
        scenario_id="source_printing_trail_point_visual_stress",
        source_env=source_env,
        vector_env=vector_env,
        steps=(("after_printing_trail_point", emit_visible_trail_point),),
        max_steps=max_steps,
        out_dir=out_dir,
        expected_terminal=False,
        visual_limits=(
            "gray64 can see the emitted trail/body point, but it does not "
            "prove PrintManager random-call order or hidden print timer state"
        ),
    )


def _run_body_opponent_tangent_then_overlap_visual_stress(
    *,
    max_steps: int | None,
    out_dir: Path | None,
) -> dict[str, Any]:
    source_env, vector_env = _manual_2p_stress_env(
        positions=((20.0, 20.0), (70.0, 70.0)),
        headings=(0.0, 0.0),
        world_bodies=(
            {
                "owner_id": 2,
                "x": 21.200000000000003,
                "y": 20.0,
                "radius": 0.6,
                "num": 0,
            },
        ),
    )

    def tangent_is_safe() -> tuple[Mapping[str, Any], Any]:
        return _step_stress_pair(source_env, vector_env, step_ms=0.0)

    def overlap_kills() -> tuple[Mapping[str, Any], Any]:
        _seed_source_body(
            source_env,
            owner_id=2,
            x=21.19,
            y=20.0,
            radius=0.6,
            num=1,
        )
        _seed_vector_world_body(
            vector_env,
            slot=1,
            owner_index=1,
            x=21.19,
            y=20.0,
            radius=0.6,
            num=1,
        )
        return _step_stress_pair(source_env, vector_env, step_ms=0.0)

    return _run_programmatic_stress_case(
        scenario_id="source_body_opponent_tangent_then_overlap_visual_stress",
        source_env=source_env,
        vector_env=vector_env,
        steps=(
            ("after_opponent_tangent_safe_probe", tangent_is_safe),
            ("after_opponent_overlap_death_probe", overlap_kills),
        ),
        max_steps=max_steps,
        out_dir=out_dir,
        expected_terminal=True,
        visual_limits=(
            "gray64 sees the live heads, opponent body, and later death point; "
            "the strict tangent-vs-overlap rule is still asserted through the "
            "source/vector terminal transition"
        ),
    )


def _run_body_own_latency_delta3_then_delta4_visual_stress(
    *,
    max_steps: int | None,
    out_dir: Path | None,
) -> dict[str, Any]:
    source_env, vector_env = _manual_2p_stress_env(
        positions=((20.0, 20.0), (70.0, 70.0)),
        headings=(0.0, 0.0),
        world_bodies=(
            {"owner_id": 1, "x": 20.0, "y": 20.0, "radius": 0.6, "num": 0},
        ),
    )
    _set_stress_body_counters(
        source_env,
        vector_env,
        player_index=0,
        body_count=3,
    )

    def own_delta3_is_safe() -> tuple[Mapping[str, Any], Any]:
        return _step_stress_pair(source_env, vector_env, step_ms=0.0)

    def own_delta4_kills() -> tuple[Mapping[str, Any], Any]:
        _set_stress_body_counters(
            source_env,
            vector_env,
            player_index=0,
            body_count=4,
        )
        return _step_stress_pair(source_env, vector_env, step_ms=0.0)

    return _run_programmatic_stress_case(
        scenario_id="source_body_own_latency_delta3_then_delta4_visual_stress",
        source_env=source_env,
        vector_env=vector_env,
        steps=(
            ("after_own_body_delta3_safe_probe", own_delta3_is_safe),
            ("after_own_body_delta4_death_probe", own_delta4_kills),
        ),
        max_steps=max_steps,
        out_dir=out_dir,
        expected_terminal=True,
        visual_limits=(
            "gray64 sees the own stored body and live/dead head geometry, but "
            "not the hidden body counters that make delta 3 safe and delta 4 fatal"
        ),
    )


def _run_print_manager_trail_gap_boundary_visual_stress(
    *,
    max_steps: int | None,
    out_dir: Path | None,
) -> dict[str, Any]:
    source_env, vector_env = _manual_2p_stress_env(
        positions=((20.0, 20.0), (70.0, 70.0)),
        headings=(0.0, 0.0),
    )
    _set_stress_avatar_printing(
        source_env,
        vector_env,
        player_index=0,
        printing=True,
        last_x=20.0,
        last_y=20.0,
    )
    _set_stress_print_manager(
        source_env,
        vector_env,
        player_index=0,
        active=True,
        distance=0.5,
        last_x=20.0,
        last_y=20.0,
    )

    def print_to_hole_boundary() -> tuple[Mapping[str, Any], Any]:
        return _step_stress_pair(source_env, vector_env, step_ms=100.0)

    def hole_to_print_boundary() -> tuple[Mapping[str, Any], Any]:
        avatar = source_env.avatar_by_id(1)
        _set_stress_print_manager(
            source_env,
            vector_env,
            player_index=0,
            active=True,
            distance=0.5,
            last_x=float(avatar.x),
            last_y=float(avatar.y),
        )
        return _step_stress_pair(source_env, vector_env, step_ms=100.0)

    return _run_programmatic_stress_case(
        scenario_id="source_print_manager_trail_gap_boundary_visual_stress",
        source_env=source_env,
        vector_env=vector_env,
        steps=(
            ("after_print_to_hole_boundary", print_to_hole_boundary),
            ("after_hole_to_print_boundary", hole_to_print_boundary),
        ),
        max_steps=max_steps,
        out_dir=out_dir,
        expected_terminal=False,
        visual_limits=(
            "gray64 sees the boundary bodies emitted when the PrintManager "
            "switches printing off and on; random gap distances and hidden "
            "manager state are outside this visual tensor"
        ),
    )


def _run_lifecycle_survivor_score_2p_next_round_visual_stress(
    *,
    max_steps: int | None,
    out_dir: Path | None,
) -> dict[str, Any]:
    source_env, vector_env = _manual_2p_stress_env(
        positions=((20.591, 44.0), (87.0, 44.0)),
        headings=(np.pi, 0.0),
        episode_end_mode="match",
        max_score=10,
    )
    vector_env.decision_ms = 100.0

    def end_round() -> tuple[Mapping[str, Any], Any]:
        source_snapshot = source_env.step((0, 0), elapsed_ms=100.0)
        batch = vector_env.step(np.asarray([[1, 1]], dtype=np.int16))
        return source_snapshot, batch

    def move_survivor_during_warmdown() -> tuple[Mapping[str, Any], Any]:
        source_env.now_ms = 1150.0
        source_env._update_game(1150.0, require_in_round=False)  # noqa: SLF001
        source_snapshot = source_env.snapshot("after_explicit_warmdown_frame")
        batch = vector_env.advance_warmdown_frame(
            np.asarray([[1, -1]], dtype=np.int16),
            elapsed_ms=1150.0,
        )
        return source_snapshot, batch

    return _run_programmatic_stress_case(
        scenario_id="source_lifecycle_survivor_score_2p_warmdown_visual_stress",
        source_env=source_env,
        vector_env=vector_env,
        steps=(
            ("after_survivor_round_end", end_round),
            ("after_warmdown_survivor_wall_death", move_survivor_during_warmdown),
        ),
        max_steps=max_steps,
        out_dir=out_dir,
        expected_terminal=False,
        visual_limits=(
            "gray64 shows survivor movement/death during an explicit warmdown "
            "frame; next-round RNG/reset, event order, and score carry remain "
            "protected by source/public lifecycle tests"
        ),
    )


def _run_bonus_game_borderless_expiry_then_wall_death_visual_stress(
    *,
    max_steps: int | None,
    out_dir: Path | None,
) -> dict[str, Any]:
    source_env, vector_env = _manual_2p_stress_env(
        positions=((20.0, 20.0), (70.0, 70.0)),
        headings=(0.0, 0.0),
        active_bonus={"type": "BonusGameBorderless", "x": 20.0, "y": 20.0},
    )

    def catch_bonus() -> tuple[Mapping[str, Any], Any]:
        return _step_stress_pair(source_env, vector_env, step_ms=0.0)

    def expire_borderless() -> tuple[Mapping[str, Any], Any]:
        return _step_stress_pair(
            source_env,
            vector_env,
            step_ms=0.0,
            timer_advance_ms=10_000.0,
        )

    def later_wall_death() -> tuple[Mapping[str, Any], Any]:
        _set_stress_avatar_state(
            source_env,
            vector_env,
            player_index=0,
            x=0.3,
            y=20.0,
            angle=np.pi,
        )
        return _step_stress_pair(source_env, vector_env, step_ms=0.0)

    return _run_programmatic_stress_case(
        scenario_id="source_bonus_game_borderless_expiry_then_wall_death_visual_stress",
        source_env=source_env,
        vector_env=vector_env,
        steps=(
            ("after_bonus_game_borderless_catch", catch_bonus),
            ("after_bonus_game_borderless_expiry", expire_borderless),
            ("after_post_expiry_wall_death", later_wall_death),
        ),
        max_steps=max_steps,
        out_dir=out_dir,
        expected_terminal=True,
        visual_limits=(
            "gray64 does not encode the game borderless flag or game bonus "
            "stack; it only exposes the later normal-wall death once the "
            "source/vector states have applied expiry"
        ),
    )


def _run_bonus_game_borderless_same_frame_expiry_wall_death_visual_stress(
    *,
    max_steps: int | None,
    out_dir: Path | None,
) -> dict[str, Any]:
    source_env, vector_env = _manual_2p_stress_env(
        positions=((20.0, 20.0), (70.0, 70.0)),
        headings=(0.0, 0.0),
        active_bonus={"type": "BonusGameBorderless", "x": 20.0, "y": 20.0},
    )

    def catch_bonus() -> tuple[Mapping[str, Any], Any]:
        return _step_stress_pair(source_env, vector_env, step_ms=0.0)

    def expire_borderless_and_die_on_wall() -> tuple[Mapping[str, Any], Any]:
        _set_stress_avatar_state(
            source_env,
            vector_env,
            player_index=0,
            x=0.3,
            y=20.0,
            angle=np.pi,
        )
        return _step_stress_pair(
            source_env,
            vector_env,
            step_ms=0.0,
            timer_advance_ms=10_000.0,
        )

    return _run_programmatic_stress_case(
        scenario_id="source_bonus_game_borderless_same_frame_expiry_wall_death_visual_stress",
        source_env=source_env,
        vector_env=vector_env,
        steps=(
            ("after_bonus_game_borderless_catch", catch_bonus),
            (
                "after_same_frame_borderless_expiry_wall_death",
                expire_borderless_and_die_on_wall,
            ),
        ),
        max_steps=max_steps,
        out_dir=out_dir,
        expected_terminal=True,
        visual_limits=(
            "gray64 sees the terminal wall-death geometry after borderless "
            "expires in the same simulated frame; the expiry event/TTL is "
            "covered by source/vector state, not a separate gray64 channel"
        ),
    )


def _run_bonus_game_clear_clears_future_collision_body_visual_stress(
    *,
    max_steps: int | None,
    out_dir: Path | None,
) -> dict[str, Any]:
    source_env, vector_env = _manual_2p_stress_env(
        positions=((20.0, 20.0), (70.0, 70.0)),
        headings=(0.0, 0.0),
        active_bonus={"type": "BonusGameClear", "x": 20.0, "y": 20.0},
        world_bodies=(
            {"owner_id": 2, "x": 25.0, "y": 20.0, "radius": 1.2, "num": 0},
        ),
    )

    def catch_bonus_and_clear_body() -> tuple[Mapping[str, Any], Any]:
        return _step_stress_pair(source_env, vector_env, step_ms=0.0)

    def probe_cleared_collision_site() -> tuple[Mapping[str, Any], Any]:
        _set_stress_avatar_state(
            source_env,
            vector_env,
            player_index=0,
            x=25.0,
            y=20.0,
            angle=0.0,
        )
        return _step_stress_pair(source_env, vector_env, step_ms=0.0)

    return _run_programmatic_stress_case(
        scenario_id="source_bonus_game_clear_clears_future_collision_body_visual_stress",
        source_env=source_env,
        vector_env=vector_env,
        steps=(
            ("after_bonus_game_clear_catch", catch_bonus_and_clear_body),
            ("after_cleared_body_collision_probe", probe_cleared_collision_site),
        ),
        max_steps=max_steps,
        out_dir=out_dir,
        expected_terminal=False,
        visual_limits=(
            "gray64 shows the seeded body disappearing and the later live head, "
            "but it does not encode the BonusGameClear event or bonus identity "
            "after the map bonus is caught"
        ),
    )


def _run_programmatic_stress_case(
    *,
    scenario_id: str,
    source_env: CurvyTronSourceEnv,
    vector_env: VectorMultiplayerEnv,
    steps: Sequence[tuple[str, Callable[[], tuple[Mapping[str, Any], Any]]]],
    max_steps: int | None,
    out_dir: Path | None,
    expected_terminal: bool,
    visual_limits: str,
) -> dict[str, Any]:
    selected_steps = list(steps)
    if max_steps is not None:
        selected_steps = selected_steps[: int(max_steps)]

    rows: list[dict[str, Any]] = []
    max_abs_diff = 0
    mismatch_pixels_total = 0
    first_mismatch: dict[str, Any] | None = None

    def compare(
        label: str,
        tick: int,
        source_snapshot: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
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

    reset_source_frame, reset_vector_frame = compare(
        "after_programmatic_stress_setup",
        0,
        source_env.snapshot("after_programmatic_stress_setup"),
    )
    final_source_frame = reset_source_frame
    final_vector_frame = reset_vector_frame
    terminal_seen = False
    for index, (label, step_fn) in enumerate(selected_steps):
        source_snapshot, batch = step_fn()
        final_source_frame, final_vector_frame = compare(label, index + 1, source_snapshot)
        if bool(batch.done[0]):
            terminal_seen = True
            break

    if out_dir is not None:
        scenario_out_dir = out_dir / scenario_id
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
        "scenario_id": scenario_id,
        "scenario_path": None,
        "comparison_kind": "programmatic_source_snapshot_stress",
        "source_arena_size": int(source_env.snapshot("report")["game"]["size"]),
        "raw_observation_schema_id": SOURCE_STATE_GRAY64_SCHEMA_ID,
        "raw_observation_shape": list(SOURCE_STATE_GRAY64_SHAPE),
        "raw_observation_dtype": "uint8",
        "renderer_impl_id": SOURCE_STATE_GRAY64_RENDERER_IMPL_ID,
        "browser_canvas_pixel_fidelity": SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY,
        "source_runner": "CurvyTronSourceEnv",
        "source_runner_claim": (
            "Python source-shaped env snapshot stress case; not browser canvas"
        ),
        "original_js_reset_check": None,
        "steps_requested": len(selected_steps),
        "frames_compared": len(rows),
        "terminal_seen": terminal_seen,
        "expected_terminal": expected_terminal,
        "match": max_abs_diff == 0 and mismatch_pixels_total == 0,
        "max_abs_diff": max_abs_diff,
        "mismatch_pixels": mismatch_pixels_total,
        "first_mismatch": first_mismatch,
        "visual_limits": visual_limits,
        "frames": rows,
    }


def _manual_2p_stress_env(
    *,
    positions: Sequence[tuple[float, float]],
    headings: Sequence[float],
    active_bonus: Mapping[str, Any] | None = None,
    world_bodies: Sequence[Mapping[str, Any]] = (),
    borderless: bool = False,
    episode_end_mode: str = "round",
    max_score: int = 10,
) -> tuple[CurvyTronSourceEnv, VectorMultiplayerEnv]:
    players = [
        {
            "id": f"p{index}",
            "client_id": f"p{index}-client",
            "avatar_id": index + 1,
            "name": f"p{index}",
            "color": "#ff0000" if index == 0 else "#00ff00",
            "initial": {
                "x": float(position[0]),
                "y": float(position[1]),
                "angle_rad": float(headings[index]),
                "printing": False,
            },
        }
        for index, position in enumerate(positions)
    ]
    source_env = CurvyTronSourceEnv(
        random_constant=0.5,
        max_score=float(max_score),
        include_deaths_snapshot=True,
        include_bonus_snapshot=True,
    )
    source_env.reset(
        player_count=2,
        players=players,
        warmup_ms=0.0,
        borderless=borderless,
        bonus_types=[],
        bonus_rate=0.0,
    )
    source_env.advance_timers(0.0)
    if source_env.game is None or source_env.game.world is None:
        raise RuntimeError("manual stress source env did not create a world")
    source_env.game.print_start_due_ms = None
    source_env.game.bonus_pop_due_ms = None
    source_env.game.started = True
    source_env.game.in_round = True
    source_env.game.world_active = True
    source_env.game.world.activate()
    for index, position in enumerate(positions):
        avatar = source_env.set_avatar_state(
            index + 1,
            x=float(position[0]),
            y=float(position[1]),
            angle=float(headings[index]),
        )
        avatar.printing = False
        avatar.print_manager.clear()
    for body in world_bodies:
        _seed_source_body(
            source_env,
            owner_id=int(body["owner_id"]),
            x=float(body["x"]),
            y=float(body["y"]),
            radius=float(body["radius"]),
            num=int(body.get("num", 0)),
        )
    if active_bonus is not None:
        source_env.seed_active_bonus(
            str(active_bonus["type"]),
            x=float(active_bonus["x"]),
            y=float(active_bonus["y"]),
        )
    source_env.events.clear()
    source_env.random.calls.clear()

    vector_env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        decision_ms=1.0,
        body_capacity=8,
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=8,
        event_mode="debug-event",
        episode_end_mode=episode_end_mode,
        max_score=max_score,
    )
    vector_env.reset(
        seed=np.asarray([101], dtype=np.uint64),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    vector_env.decision_ms = 0.0
    state = vector_env.state
    state["timer_active"][0] = False
    state["env_active"][0] = True
    state["started"][0] = True
    state["in_round"][0] = True
    state["done"][0] = False
    state["terminated"][0] = False
    state["truncated"][0] = False
    state["world_active"][0] = True
    state["borderless"][0] = bool(borderless)
    state["alive"][0] = True
    state["present"][0] = True
    state["pos"][0] = np.asarray(positions, dtype=np.float64)
    state["prev_pos"][0] = state["pos"][0]
    state["heading"][0] = np.asarray(headings, dtype=np.float64)
    state["printing"][0] = False
    state["print_manager_active"][0] = False
    state["print_manager_distance"][0] = 0.0
    state["print_manager_last_pos"][0] = state["pos"][0]
    state["has_draw_cursor"][0] = False
    state["draw_cursor_pos"][0] = 0.0
    state["body_active"][0] = False
    state["body_pos"][0] = 0.0
    state["body_radius"][0] = 0.0
    state["body_owner"][0] = -1
    state["body_num"][0] = -1
    state["body_insert_tick"][0] = -1
    state["body_insert_kind"][0] = -1
    state["body_write_cursor"][0] = 0
    state["world_body_count"][0] = 0
    state["body_count"][0] = 0
    state["live_body_num"][0] = 0
    for slot, body in enumerate(world_bodies):
        _seed_vector_world_body(
            vector_env,
            slot=slot,
            owner_index=int(body["owner_id"]) - 1,
            x=float(body["x"]),
            y=float(body["y"]),
            radius=float(body["radius"]),
            num=int(body.get("num", 0)),
        )
    if active_bonus is not None:
        vector_env.seed_active_bonus(
            row=0,
            bonus_type=str(active_bonus["type"]),
            x=float(active_bonus["x"]),
            y=float(active_bonus["y"]),
        )
    return source_env, vector_env


def _step_stress_pair(
    source_env: CurvyTronSourceEnv,
    vector_env: VectorMultiplayerEnv,
    *,
    step_ms: float,
    timer_advance_ms: float = 0.0,
    source_moves: Sequence[int] = (0, 0),
) -> tuple[Mapping[str, Any], Any]:
    source_env.advance_timers(timer_advance_ms)
    source_snapshot = source_env.step(source_moves, elapsed_ms=step_ms)
    vector_env.decision_ms = float(step_ms)
    batch = vector_env.step(
        np.asarray([[move + 1 for move in source_moves]], dtype=np.int16),
        timer_advance_ms=float(timer_advance_ms),
    )
    return source_snapshot, batch


def _set_stress_avatar_state(
    source_env: CurvyTronSourceEnv,
    vector_env: VectorMultiplayerEnv,
    *,
    player_index: int,
    x: float,
    y: float,
    angle: float,
) -> None:
    source_env.set_avatar_state(player_index + 1, x=float(x), y=float(y), angle=float(angle))
    state = vector_env.state
    state["pos"][0, player_index] = np.asarray([float(x), float(y)], dtype=np.float64)
    state["prev_pos"][0, player_index] = state["pos"][0, player_index]
    state["heading"][0, player_index] = float(angle)
    state["print_manager_last_pos"][0, player_index] = state["pos"][0, player_index]


def _set_stress_avatar_printing(
    source_env: CurvyTronSourceEnv,
    vector_env: VectorMultiplayerEnv,
    *,
    player_index: int,
    printing: bool,
    last_x: float,
    last_y: float,
) -> None:
    avatar = source_env.avatar_by_id(player_index + 1)
    avatar.printing = bool(printing)
    avatar.print_manager.active = False
    avatar.print_manager.distance = 999_999.0
    avatar.print_manager.last_x = float(last_x)
    avatar.print_manager.last_y = float(last_y)
    avatar.trail_point_count = 1
    avatar.trail_last_x = float(last_x)
    avatar.trail_last_y = float(last_y)
    state = vector_env.state
    state["printing"][0, player_index] = bool(printing)
    state["print_manager_active"][0, player_index] = False
    state["print_manager_distance"][0, player_index] = 999_999.0
    state["print_manager_last_pos"][0, player_index] = (float(last_x), float(last_y))
    state["body_count"][0, player_index] = 1
    state["live_body_num"][0, player_index] = 1
    state["visible_trail_count"][0, player_index] = 1
    state["has_visible_trail_last"][0, player_index] = True
    state["visible_trail_last_pos"][0, player_index] = (float(last_x), float(last_y))


def _set_stress_print_manager(
    source_env: CurvyTronSourceEnv,
    vector_env: VectorMultiplayerEnv,
    *,
    player_index: int,
    active: bool,
    distance: float,
    last_x: float,
    last_y: float,
) -> None:
    avatar = source_env.avatar_by_id(player_index + 1)
    avatar.print_manager.active = bool(active)
    avatar.print_manager.distance = float(distance)
    avatar.print_manager.last_x = float(last_x)
    avatar.print_manager.last_y = float(last_y)
    state = vector_env.state
    state["print_manager_active"][0, player_index] = bool(active)
    state["print_manager_distance"][0, player_index] = float(distance)
    state["print_manager_last_pos"][0, player_index] = (float(last_x), float(last_y))


def _set_stress_body_counters(
    source_env: CurvyTronSourceEnv,
    vector_env: VectorMultiplayerEnv,
    *,
    player_index: int,
    body_count: int,
) -> None:
    avatar = source_env.avatar_by_id(player_index + 1)
    avatar.body_count = int(body_count)
    avatar.body_num = int(body_count)
    state = vector_env.state
    state["body_count"][0, player_index] = int(body_count)
    state["live_body_num"][0, player_index] = int(body_count)


def _seed_vector_world_body(
    env: VectorMultiplayerEnv,
    *,
    slot: int,
    owner_index: int,
    x: float,
    y: float,
    radius: float,
    num: int,
) -> None:
    state = env.state
    state["body_active"][0, slot] = True
    state["body_pos"][0, slot] = (float(x), float(y))
    state["body_radius"][0, slot] = float(radius)
    state["body_owner"][0, slot] = int(owner_index)
    state["body_num"][0, slot] = int(num)
    state["body_insert_tick"][0, slot] = int(state["tick"][0])
    state["body_insert_kind"][0, slot] = vector_runtime.BODY_KIND_NORMAL
    state["body_write_cursor"][0] = max(int(state["body_write_cursor"][0]), slot + 1)
    state["world_body_count"][0] = int(state["body_active"][0].sum())
    state["body_count"][0, owner_index] = max(
        int(state["body_count"][0, owner_index]),
        int(num) + 1,
    )


def _single_frame_visual_report(
    *,
    report_id: str,
    comparison_kind: str,
    source_env: CurvyTronSourceEnv,
    vector_env: VectorMultiplayerEnv,
    label: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_snapshot = source_env.snapshot(label)
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
    max_abs_diff = int(diff.max())
    mismatch_pixels = int(np.count_nonzero(diff))
    first_mismatch = None
    if mismatch_pixels:
        yx = np.argwhere(diff[0] != 0)[0]
        first_mismatch = {
            "label": label,
            "tick": 0,
            "channel": 0,
            "y": int(yx[0]),
            "x": int(yx[1]),
            "source": int(source_frame[0, yx[0], yx[1]]),
            "vector": int(vector_frame[0, yx[0], yx[1]]),
        }
    report = {
        "schema_id": SCHEMA_ID,
        "scenario_id": report_id,
        "scenario_path": None,
        "comparison_kind": comparison_kind,
        "source_arena_size": int(source_env.snapshot("report")["game"]["size"]),
        "raw_observation_schema_id": SOURCE_STATE_GRAY64_SCHEMA_ID,
        "raw_observation_shape": list(SOURCE_STATE_GRAY64_SHAPE),
        "raw_observation_dtype": "uint8",
        "renderer_impl_id": SOURCE_STATE_GRAY64_RENDERER_IMPL_ID,
        "browser_canvas_pixel_fidelity": SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY,
        "source_runner": "CurvyTronSourceEnv",
        "source_runner_claim": (
            "Python source-shaped env snapshot canary; not browser canvas"
        ),
        "steps_requested": 0,
        "frames_compared": 1,
        "terminal_seen": False,
        "match": max_abs_diff == 0 and mismatch_pixels == 0,
        "max_abs_diff": max_abs_diff,
        "mismatch_pixels": mismatch_pixels,
        "first_mismatch": first_mismatch,
        "frames": (
            {
                "label": label,
                "tick": 0,
                "source_nonzero_pixels": int(np.count_nonzero(source_frame)),
                "vector_nonzero_pixels": int(np.count_nonzero(vector_frame)),
                "max_abs_diff": max_abs_diff,
                "mismatch_pixels": mismatch_pixels,
            },
        ),
    }
    if extra is not None:
        report.update(dict(extra))
    return report


def _run_typed_bonus_visual_status_case(bonus_type: str) -> dict[str, Any]:
    active_source_env, active_vector_env = _manual_2p_stress_env(
        positions=((20.0, 20.0), (70.0, 70.0)),
        headings=(0.0, 0.0),
        active_bonus={"type": bonus_type, "x": 44.0, "y": 44.0},
    )
    source_mask, source_type = _source_bonus64_mask_type_planes(active_source_env)
    vector_tensor = render_source_state_bonus64_stack4_player_perspective_v1(
        active_vector_env.state,
        controlled_player=0,
    )
    vector_mask = vector_tensor[SOURCE_STATE_BONUS64_STACK4_BONUS_MASK_CHANNEL]
    vector_type = vector_tensor[SOURCE_STATE_BONUS64_STACK4_BONUS_TYPE_CHANNEL]
    first_mismatch: dict[str, Any] | None = None
    map_max_abs_diff = 0.0
    map_mismatch_pixels = 0
    for channel, source_plane, vector_plane in (
        (SOURCE_STATE_BONUS64_STACK4_BONUS_MASK_CHANNEL, source_mask, vector_mask),
        (SOURCE_STATE_BONUS64_STACK4_BONUS_TYPE_CHANNEL, source_type, vector_type),
    ):
        diff = np.abs(source_plane.astype(np.float32, copy=False) - vector_plane)
        mismatch = diff > np.float32(1e-6)
        map_max_abs_diff = max(map_max_abs_diff, float(diff.max()))
        map_mismatch_pixels += int(np.count_nonzero(mismatch))
        if bool(mismatch.any()) and first_mismatch is None:
            yx = np.argwhere(mismatch)[0]
            first_mismatch = {
                "label": "after_active_typed_bonus_setup",
                "tick": 0,
                "channel": int(channel),
                "y": int(yx[0]),
                "x": int(yx[1]),
                "source": float(source_plane[yx[0], yx[1]]),
                "vector": float(vector_plane[yx[0], yx[1]]),
            }

    status_source_env, status_vector_env = _manual_2p_stress_env(
        positions=((20.0, 20.0), (70.0, 70.0)),
        headings=(0.0, 0.0),
        active_bonus={"type": bonus_type, "x": 20.0, "y": 20.0},
    )
    _step_stress_pair(status_source_env, status_vector_env, step_ms=0.0)
    status_tensor = render_source_state_bonus64_stack4_player_perspective_v1(
        status_vector_env.state,
        controlled_player=0,
    )
    status_values = _source_bonus64_status_channel_values(
        status_source_env,
        controlled_player=0,
    )
    status_max_abs_diff = 0.0
    status_mismatch_pixels = 0
    for channel, source_value in status_values.items():
        vector_plane = status_tensor[channel]
        source_plane = np.full((64, 64), np.float32(source_value), dtype=np.float32)
        diff = np.abs(source_plane - vector_plane)
        mismatch = diff > np.float32(1e-6)
        status_max_abs_diff = max(status_max_abs_diff, float(diff.max()))
        status_mismatch_pixels += int(np.count_nonzero(mismatch))
        if bool(mismatch.any()) and first_mismatch is None:
            yx = np.argwhere(mismatch)[0]
            first_mismatch = {
                "label": "after_typed_bonus_catch",
                "tick": 1,
                "channel": int(channel),
                "y": int(yx[0]),
                "x": int(yx[1]),
                "source": float(source_plane[yx[0], yx[1]]),
                "vector": float(vector_plane[yx[0], yx[1]]),
            }

    max_abs_diff = max(map_max_abs_diff, status_max_abs_diff)
    mismatch_pixels = map_mismatch_pixels + status_mismatch_pixels
    bonus_records = active_source_env.bonus_bodies_snapshot()
    if len(bonus_records) != 1:
        raise RuntimeError("typed bonus gate expected exactly one source bonus")
    bonus_record = bonus_records[0]
    map_size = float(active_source_env.snapshot("report")["game"]["size"])
    px = int(
        np.clip(
            np.rint((float(bonus_record["x"]) / map_size) * 63.0),
            0,
            63,
        )
    )
    py = int(
        np.clip(
            np.rint((float(bonus_record["y"]) / map_size) * 63.0),
            0,
            63,
        )
    )
    bonus_type_code = _bonus_type_code(str(bonus_record["type"]))
    expected_type_value = float(
        np.float32(bonus_type_code)
        / np.float32(SOURCE_STATE_BONUS64_STACK4_MAX_BONUS_TYPE_CODE)
    )
    return {
        "schema_id": f"{SCHEMA_ID}_bonus64_typed_case",
        "scenario_id": f"{TYPED_BONUS_VISUAL_STATUS_GATE_ID}:{bonus_type}",
        "scenario_path": None,
        "comparison_kind": "typed_bonus_visual_status_case",
        "source_arena_size": int(map_size),
        "visual_observation_schema_id": SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_SCHEMA_ID,
        "bonus_type": bonus_type,
        "bonus_type_code": bonus_type_code,
        "center": {"y": py, "x": px},
        "source_mask_at_center": float(source_mask[py, px]),
        "vector_mask_at_center": float(vector_mask[py, px]),
        "source_type_at_center": float(source_type[py, px]),
        "vector_type_at_center": float(vector_type[py, px]),
        "expected_type_at_center": expected_type_value,
        "match": max_abs_diff <= 1e-6 and mismatch_pixels == 0,
        "max_abs_diff": max_abs_diff,
        "mismatch_pixels": mismatch_pixels,
        "map_type_max_abs_diff": map_max_abs_diff,
        "map_type_mismatch_pixels": map_mismatch_pixels,
        "status_max_abs_diff": status_max_abs_diff,
        "status_mismatch_pixels": status_mismatch_pixels,
        "first_mismatch": first_mismatch,
        "frames_compared": 2,
        "planes_compared": (
            SOURCE_STATE_BONUS64_STACK4_BONUS_MASK_CHANNEL,
            SOURCE_STATE_BONUS64_STACK4_BONUS_TYPE_CHANNEL,
            *status_values.keys(),
        ),
    }


def _source_bonus64_mask_type_planes(
    source_env: CurvyTronSourceEnv,
) -> tuple[np.ndarray, np.ndarray]:
    snapshot = source_env.snapshot("source_bonus64")
    map_size = float(snapshot["game"]["size"])
    mask_plane = np.zeros((64, 64), dtype=np.float32)
    type_plane = np.zeros((64, 64), dtype=np.float32)
    winners = np.full((64, 64), -1, dtype=np.int32)
    for fallback_slot, bonus in enumerate(source_env.bonus_bodies_snapshot()):
        mask_info = _source_bonus64_world_circle_patch_mask(
            float(bonus["x"]),
            float(bonus["y"]),
            float(bonus["radius"]),
            map_size,
        )
        if mask_info is None:
            continue
        y_slice, x_slice, circle_mask = mask_info
        slot_id = int(bonus.get("id", fallback_slot + 1))
        patch_winners = winners[y_slice, x_slice]
        replace = circle_mask & (slot_id >= patch_winners)
        if not bool(replace.any()):
            continue
        patch_winners[replace] = slot_id
        mask_plane[y_slice, x_slice][replace] = 1.0
        code = _bonus_type_code(str(bonus["type"]))
        type_plane[y_slice, x_slice][replace] = (
            np.float32(code)
            / np.float32(SOURCE_STATE_BONUS64_STACK4_MAX_BONUS_TYPE_CODE)
        )
    return mask_plane, type_plane


def _source_bonus64_status_channel_values(
    source_env: CurvyTronSourceEnv,
    *,
    controlled_player: int,
) -> dict[int, float]:
    player_count = len(source_env.avatars)
    controlled = int(controlled_player)
    if controlled < 0 or controlled >= player_count:
        raise ValueError("controlled_player is outside the source avatar range")
    other = 1 - controlled if player_count == 2 else _first_other_player(controlled, player_count)
    values: dict[int, float] = {}
    for channels, player in (
        (SOURCE_STATE_BONUS64_STACK4_SELF_STATUS_CHANNELS, controlled),
        (SOURCE_STATE_BONUS64_STACK4_OTHER_STATUS_CHANNELS, other),
    ):
        player_values = _source_bonus64_player_status_values(source_env, player)
        for channel, value in zip(channels, player_values, strict=True):
            values[int(channel)] = float(value)

    game = source_env.game
    values[SOURCE_STATE_BONUS64_STACK4_GAME_BORDERLESS_CHANNEL] = (
        1.0 if game is not None and bool(game.borderless) else 0.0
    )
    game_bonuses = () if game is None else tuple(game.active_bonuses)
    values[SOURCE_STATE_BONUS64_STACK4_GAME_TTL_CHANNEL] = _source_bonus64_ttl_value(
        bonus.duration for bonus in game_bonuses
    )
    return values


def _source_bonus64_player_status_values(
    source_env: CurvyTronSourceEnv,
    player: int,
) -> tuple[float, float, float, float, float, float, float]:
    avatar = source_env.avatars[int(player)]
    reference = source_env.reference
    radius = _source_bonus64_ratio_value(
        avatar.radius,
        reference.avatar_radius,
        scale=4.0,
    )
    speed = _source_bonus64_ratio_value(
        avatar.velocity,
        reference.avatar_velocity_units_per_s,
        scale=2.0,
    )
    turn_override = (
        (not bool(avatar.direction_in_loop))
        or not np.isclose(
            float(avatar.angular_velocity_base),
            float(reference.angular_velocity_radians_per_ms),
        )
    )
    ttl = _source_bonus64_ttl_value(bonus.duration for bonus in avatar.active_bonuses)
    return (
        radius,
        speed,
        1.0 if bool(avatar.inverse) else 0.0,
        1.0 if turn_override else 0.0,
        1.0 if bool(avatar.invincible) else 0.0,
        1.0 if bool(avatar.printing) else 0.0,
        ttl,
    )


def _source_bonus64_ratio_value(value: float, base: float, *, scale: float) -> float:
    value_float = float(value)
    base_float = float(base)
    if not np.isfinite(value_float) or not np.isfinite(base_float) or base_float <= 0.0:
        return 0.0
    return float(np.clip(value_float / base_float, 0.0, scale) / scale)


def _source_bonus64_ttl_value(durations: Any) -> float:
    values = np.asarray(tuple(durations), dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(
        np.clip(
            np.nanmax(values),
            0.0,
            SOURCE_STATE_BONUS64_STACK4_MAX_TTL_MS,
        )
        / SOURCE_STATE_BONUS64_STACK4_MAX_TTL_MS
    )


def _first_other_player(controlled: int, player_count: int) -> int:
    for player in range(player_count):
        if player != controlled:
            return player
    return controlled


def _source_bonus64_world_circle_patch_mask(
    x: float,
    y: float,
    radius: float,
    map_size: float,
) -> tuple[slice, slice, np.ndarray] | None:
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(radius):
        return None
    if x + radius <= 0.0 or x - radius >= map_size:
        return None
    if y + radius <= 0.0 or y - radius >= map_size:
        return None
    radius_px = int(max(0, np.ceil((radius / map_size) * 64.0)))
    px = int(np.clip(np.rint((x / map_size) * 63.0), 0, 63))
    py = int(np.clip(np.rint((y / map_size) * 63.0), 0, 63))
    if radius_px == 0:
        return slice(py, py + 1), slice(px, px + 1), np.ones((1, 1), dtype=bool)
    x0 = max(0, px - radius_px)
    x1 = min(63, px + radius_px)
    y0 = max(0, py - radius_px)
    y1 = min(63, py + radius_px)
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    circle_mask = (xx - px) ** 2 + (yy - py) ** 2 <= radius_px**2
    return slice(y0, y1 + 1), slice(x0, x1 + 1), circle_mask


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
    parser.add_argument("--suite", choices=("core2p", "full2p"), default=None)
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
    if args.suite == "full2p":
        report = run_full_2p_visual_gate(max_steps=args.max_steps)
        if args.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
            return
        status = "PASS" if report["match"] else "FAIL"
        gray64 = report["gray64"]
        typed = report["typed_bonus"]
        final_observation = report["final_observation"]
        print(
            f"{status} {report['gate_id']} "
            f"canvas_gray64={gray64['passed']}/{gray64['scenario_count']} "
            f"typed_bonus={typed['passed']}/{typed['case_count']} "
            f"final_obs={'pass' if final_observation['match'] else 'fail'} "
            f"canaries={report['visual_canaries_passed']}/{report['visual_canary_count']} "
            f"mismatch_pixels={report['mismatch_pixels']} "
            f"max_abs_diff={report['max_abs_diff']} "
            f"expected_canary_mismatch_pixels={report['expected_canary_mismatch_pixels']}"
        )
        return
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
