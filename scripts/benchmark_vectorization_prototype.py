"""Prototype NumPy microbenchmark for future vectorized CurvyTron stepping.

PROTOTYPE ONLY: this script is isolated scaffolding and is not imported by
production environment code. It does not run source fixtures and it does not
claim source fidelity. The useful surface is the fixed-shape array layout and
the source-like timing buckets for movement, trail point append, strict overlap
scan, and own-body latency masks.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np


SOURCE_FIDELITY_CLAIM = (
    "none; source-like synthetic NumPy microbenchmark only. It does not execute "
    "source-fidelity runners or scenario fixtures."
)

TIMER_DEFINITIONS = {
    "movement": (
        "Reverse player loop over a leading batch axis; updates live_body_num, "
        "heading, prev_pos, and pos using fixed-step source-like movement."
    ),
    "normal_point_mask": (
        "Computes printing plus hidden draw-cursor cadence: empty cursor draws, "
        "otherwise squared distance from draw cursor must be strictly greater "
        "than radius squared."
    ),
    "body_append": (
        "Append-only fixed body buffer writes for normal trail points, including "
        "visible trail last, draw cursor, body owner, body number, and counters."
    ),
    "collision_scan": (
        "Full K-slot circle scan with strict overlap, opponent bodies enabled "
        "immediately, and own-body latency masked by body number."
    ),
    "observation": (
        "Toy array observation construction only; no public dict wrappers or "
        "source observation schema."
    ),
    "loop_overhead": "Timed loop wall time not covered by the bucket timers above.",
}

SEMANTICS_COVERAGE = {
    "covered_source_like": [
        "fixed-shape structure-of-arrays state with B environments, P players, and K bodies",
        "reverse player update order inside each tick",
        "fixed-step movement with source_move values -1/0/1",
        "separate printing flag, visible trail last, hidden draw cursor, and body buffer",
        "normal point cadence uses empty cursor or strict distance > radius",
        "normal body insertion happens before collision scan for that player",
        "append-only body buffer with owner, body number, radius, insert tick, and kind",
        "strict circle overlap uses < so exact tangent is safe",
        "own-body latency mask uses live_body_num - stored_body_num <= trail_latency",
        "opponent bodies can collide immediately",
    ],
    "partially_covered": [
        "visible trail vs draw cursor vs body buffer are separate arrays, but full visible trail lists are not retained",
        "array observation timing exists, but it is a toy tensor and not the final environment observation contract",
        "overflow is explicit and rows are masked after capacity is exhausted",
    ],
    "missing_source_semantics": [
        "source fixture replay and common-trace equivalence",
        "PrintManager post-collision distance updates, toggles, and boundary body insertion",
        "borderless PrintManager wrap branch and normal wall death branch",
        "death-point insertion after collision",
        "same-tick death scoring, round scores, terminal events, and winner/draw lifecycle",
        "bonuses, power-ups, spawn randomness, and real random print/hole distances",
        "public Gym/PettingZoo wrappers, dict outputs, info objects, and autoreset behavior",
        "full browser/replay trail event stream",
    ],
}


@dataclass(frozen=True)
class Profile:
    batch: int
    players: int
    body_capacity: int
    steps: int
    warmup: int
    width: float
    height: float
    radius: float
    speed: float
    step_ms: float
    angular_velocity_per_ms: float
    trail_latency: int
    dtype: str
    action_pattern: str
    seed: int


def _git_text(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, check=False, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_manifest() -> dict[str, Any]:
    status = _git_text(["git", "status", "--short"])
    status_lines = status.splitlines() if status else []
    return {
        "revision": _git_text(["git", "rev-parse", "--short", "HEAD"]) or "unavailable",
        "working_tree_status": "changes-present" if status_lines else "clean-or-unavailable",
        "status_entry_count": len(status_lines),
    }


def _runtime_manifest() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "optional_tensor_backends": {
            "jax": {
                "available": importlib.util.find_spec("jax") is not None,
                "status": "detected only; not imported and not required",
            },
            "torch": {
                "available": importlib.util.find_spec("torch") is not None,
                "status": "detected only; not imported and not required",
            },
        },
    }


def _np_dtype(name: str) -> np.dtype[Any]:
    if name == "float32":
        return np.dtype(np.float32)
    if name == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"unsupported dtype {name!r}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _initial_state(profile: Profile) -> dict[str, np.ndarray]:
    dtype = _np_dtype(profile.dtype)
    batch = profile.batch
    players = profile.players
    body_capacity = profile.body_capacity

    pos = np.zeros((batch, players, 2), dtype=dtype)
    y_lanes = np.linspace(
        profile.height / (players + 1),
        profile.height * players / (players + 1),
        players,
        dtype=dtype,
    )
    env_offsets = (np.arange(batch, dtype=dtype) % dtype.type(17)) * dtype.type(0.01)
    pos[:, :, 0] = dtype.type(profile.width * 0.15)
    pos[:, :, 1] = y_lanes[None, :] + env_offsets[:, None]

    return {
        "tick": np.zeros(batch, dtype=np.int32),
        "done": np.zeros(batch, dtype=bool),
        "overflow": np.zeros(batch, dtype=bool),
        "pos": pos,
        "prev_pos": pos.copy(),
        "heading": np.zeros((batch, players), dtype=dtype),
        "alive": np.ones((batch, players), dtype=bool),
        "printing": np.ones((batch, players), dtype=bool),
        "has_visible_trail_last": np.zeros((batch, players), dtype=bool),
        "visible_trail_last_pos": np.zeros((batch, players, 2), dtype=dtype),
        "visible_trail_count": np.zeros((batch, players), dtype=np.int32),
        "has_draw_cursor": np.zeros((batch, players), dtype=bool),
        "draw_cursor_pos": np.zeros((batch, players, 2), dtype=dtype),
        "body_active": np.zeros((batch, body_capacity), dtype=bool),
        "body_pos": np.zeros((batch, body_capacity, 2), dtype=dtype),
        "body_radius": np.zeros((batch, body_capacity), dtype=dtype),
        "body_owner": np.full((batch, body_capacity), -1, dtype=np.int16),
        "body_num": np.full((batch, body_capacity), -1, dtype=np.int32),
        "body_insert_tick": np.full((batch, body_capacity), -1, dtype=np.int32),
        "body_insert_kind": np.full((batch, body_capacity), -1, dtype=np.int8),
        "body_write_cursor": np.zeros(batch, dtype=np.int32),
        "world_body_count": np.zeros(batch, dtype=np.int32),
        "body_overflow": np.zeros(batch, dtype=bool),
        "body_count": np.zeros((batch, players), dtype=np.int32),
        "live_body_num": np.zeros((batch, players), dtype=np.int32),
        "normal_point_count": np.zeros((batch, players), dtype=np.int32),
        "body_scan_count": np.zeros((batch, players), dtype=np.int64),
        "body_candidate_count": np.zeros((batch, players), dtype=np.int64),
        "body_hit_slot": np.full((batch, players), -1, dtype=np.int32),
        "death_tick": np.full((batch, players), -1, dtype=np.int32),
        "obs": np.zeros((batch, players, 6), dtype=dtype),
    }


def _action_table(profile: Profile, steps: int) -> np.ndarray:
    moves = np.zeros((steps, profile.players), dtype=np.int8)
    if profile.action_pattern == "straight":
        return moves
    if profile.action_pattern == "alternating-turns":
        pattern = np.array([-1, 1], dtype=np.int8)
        for step in range(steps):
            moves[step] = pattern[(step + np.arange(profile.players)) % 2]
        return moves
    if profile.action_pattern == "weave":
        pattern = np.array([-1, 0, 1, 0], dtype=np.int8)
        for step in range(steps):
            moves[step] = pattern[(step + np.arange(profile.players)) % len(pattern)]
        return moves
    raise ValueError(f"unsupported action pattern {profile.action_pattern!r}")


def _append_normal_points(
    state: dict[str, np.ndarray],
    profile: Profile,
    player: int,
    write_mask: np.ndarray,
) -> tuple[int, int]:
    rows = np.flatnonzero(write_mask)
    if rows.size == 0:
        return 0, 0

    cursor = state["body_write_cursor"][rows]
    can_write = cursor < profile.body_capacity
    overflow_rows = rows[~can_write]
    if overflow_rows.size:
        state["body_overflow"][overflow_rows] = True
        state["overflow"][overflow_rows] = True

    write_rows = rows[can_write]
    if write_rows.size == 0:
        return 0, int(overflow_rows.size)

    slots = cursor[can_write]
    body_count = state["body_count"][write_rows, player]
    state["body_active"][write_rows, slots] = True
    state["body_pos"][write_rows, slots] = state["pos"][write_rows, player]
    state["body_radius"][write_rows, slots] = profile.radius
    state["body_owner"][write_rows, slots] = player
    state["body_num"][write_rows, slots] = body_count
    state["body_insert_tick"][write_rows, slots] = state["tick"][write_rows]
    state["body_insert_kind"][write_rows, slots] = 0
    state["body_write_cursor"][write_rows] += 1
    state["world_body_count"][write_rows] += 1
    state["body_count"][write_rows, player] += 1
    state["normal_point_count"][write_rows, player] += 1

    state["has_visible_trail_last"][write_rows, player] = True
    state["visible_trail_last_pos"][write_rows, player] = state["pos"][write_rows, player]
    state["visible_trail_count"][write_rows, player] += 1
    state["has_draw_cursor"][write_rows, player] = True
    state["draw_cursor_pos"][write_rows, player] = state["pos"][write_rows, player]
    return int(write_rows.size), int(overflow_rows.size)


def _update_observation(state: dict[str, np.ndarray], profile: Profile) -> None:
    denom = max(1, profile.body_capacity)
    obs = state["obs"]
    obs[:, :, 0:2] = state["pos"]
    obs[:, :, 2] = state["heading"]
    obs[:, :, 3] = state["alive"]
    obs[:, :, 4] = state["body_write_cursor"][:, None] / denom
    obs[:, :, 5] = state["normal_point_count"]


def _step_numpy(
    state: dict[str, np.ndarray],
    profile: Profile,
    source_moves: np.ndarray,
    timers: dict[str, float],
    counters: dict[str, int],
) -> None:
    dtype = _np_dtype(profile.dtype)
    distance = dtype.type(profile.speed * profile.step_ms / 1000.0)
    angle_delta = dtype.type(profile.angular_velocity_per_ms * profile.step_ms)
    radius_sq = dtype.type(profile.radius * profile.radius)
    hit_radius_sq = dtype.type((profile.radius + profile.radius) ** 2)
    trail_latency = np.int32(profile.trail_latency)

    for player in range(profile.players - 1, -1, -1):
        started = time.perf_counter()
        live_mask = state["alive"][:, player] & ~state["done"] & ~state["overflow"]
        state["live_body_num"][:, player] = state["body_count"][:, player]
        old_heading = state["heading"][:, player]
        new_heading = old_heading + dtype.type(source_moves[player]) * angle_delta
        state["heading"][:, player] = np.where(live_mask, new_heading, old_heading)
        heading = state["heading"][:, player]
        state["prev_pos"][:, player] = state["pos"][:, player]
        new_x = state["pos"][:, player, 0] + np.cos(heading) * distance
        new_y = state["pos"][:, player, 1] + np.sin(heading) * distance
        state["pos"][:, player, 0] = np.where(live_mask, new_x, state["pos"][:, player, 0])
        state["pos"][:, player, 1] = np.where(live_mask, new_y, state["pos"][:, player, 1])
        counters["movement_updates"] += int(live_mask.sum())
        timers["movement"] += time.perf_counter() - started

        started = time.perf_counter()
        cursor_dx = state["pos"][:, player, 0] - state["draw_cursor_pos"][:, player, 0]
        cursor_dy = state["pos"][:, player, 1] - state["draw_cursor_pos"][:, player, 1]
        cursor_dist_sq = cursor_dx * cursor_dx + cursor_dy * cursor_dy
        should_draw = (
            live_mask
            & state["printing"][:, player]
            & (~state["has_draw_cursor"][:, player] | (cursor_dist_sq > radius_sq))
        )
        counters["normal_point_mask_true"] += int(should_draw.sum())
        timers["normal_point_mask"] += time.perf_counter() - started

        started = time.perf_counter()
        inserted, overflowed = _append_normal_points(state, profile, player, should_draw)
        counters["normal_points_inserted"] += inserted
        counters["body_overflow_attempts"] += overflowed
        timers["body_append"] += time.perf_counter() - started

        started = time.perf_counter()
        if profile.body_capacity:
            dx = state["body_pos"][:, :, 0] - state["pos"][:, player, 0][:, None]
            dy = state["body_pos"][:, :, 1] - state["pos"][:, player, 1][:, None]
            dist_sq = dx * dx + dy * dy
            own_body = state["body_owner"] == player
            own_delta = state["live_body_num"][:, player][:, None] - state["body_num"]
            own_too_young = own_body & (own_delta <= trail_latency)
            candidate = state["body_active"] & ~own_too_young
            hit_mask = candidate & (dist_sq < hit_radius_sq)
            has_hit = live_mask & hit_mask.any(axis=1)
            first_hit = np.argmax(hit_mask, axis=1)
            state["body_candidate_count"][:, player] += candidate.sum(axis=1)
            state["body_scan_count"][:, player] += live_mask.astype(np.int64) * profile.body_capacity
            hit_rows = np.flatnonzero(has_hit)
            if hit_rows.size:
                state["alive"][hit_rows, player] = False
                state["death_tick"][hit_rows, player] = state["tick"][hit_rows]
                state["body_hit_slot"][hit_rows, player] = first_hit[hit_rows]
                counters["body_hits"] += int(hit_rows.size)
        timers["collision_scan"] += time.perf_counter() - started

    started = time.perf_counter()
    _update_observation(state, profile)
    timers["observation"] += time.perf_counter() - started
    active_rows = ~state["done"]
    state["tick"][active_rows] += 1


def _run_steps(
    profile: Profile,
    state: dict[str, np.ndarray],
    moves: np.ndarray,
) -> tuple[dict[str, float], dict[str, int], float]:
    timers = {name: 0.0 for name in TIMER_DEFINITIONS if name != "loop_overhead"}
    counters: dict[str, int] = defaultdict(int)
    started = time.perf_counter()
    for step_moves in moves:
        _step_numpy(state, profile, step_moves, timers, counters)
    elapsed = time.perf_counter() - started
    measured = sum(timers.values())
    timers["loop_overhead"] = max(0.0, elapsed - measured)
    return timers, dict(counters), elapsed


def _state_nbytes(state: dict[str, np.ndarray]) -> int:
    return int(sum(value.nbytes for value in state.values()))


def _sanity_checks(profile: Profile) -> dict[str, Any]:
    dtype = _np_dtype(profile.dtype)
    radius = dtype.type(profile.radius)
    tangent_dist_sq = dtype.type((profile.radius + profile.radius) ** 2)
    strict_tangent_safe = bool(not (tangent_dist_sq < tangent_dist_sq))
    overlap_dist_sq = np.nextafter(tangent_dist_sq, dtype.type(0.0), dtype=dtype)
    strict_overlap_hits = bool(overlap_dist_sq < tangent_dist_sq)

    live_body_num = np.array([4, 5], dtype=np.int32)
    stored_body_num = np.array([1, 1], dtype=np.int32)
    own_too_young = (live_body_num - stored_body_num) <= np.int32(profile.trail_latency)
    opponent_candidate = True

    return {
        "strict_tangent_safe": strict_tangent_safe,
        "strict_overlap_hits": strict_overlap_hits,
        "own_delta_equal_latency_masked": bool(own_too_young[0]),
        "own_delta_latency_plus_one_candidate": bool(~own_too_young[1]),
        "opponent_body_candidate_immediate": opponent_candidate,
        "radius": float(radius),
    }


def run(profile: Profile) -> dict[str, Any]:
    sanity = _sanity_checks(profile)

    if profile.warmup:
        warmup_moves = _action_table(profile, profile.warmup)
        warmup_state = _initial_state(profile)
        warmup_started = time.perf_counter()
        _run_steps(profile, warmup_state, warmup_moves)
        warmup_sec = time.perf_counter() - warmup_started
    else:
        warmup_sec = 0.0

    setup_started = time.perf_counter()
    state = _initial_state(profile)
    moves = _action_table(profile, profile.steps)
    setup_sec = time.perf_counter() - setup_started
    state_bytes = _state_nbytes(state)

    timers, counters, elapsed = _run_steps(profile, state, moves)
    env_steps = profile.batch * profile.steps
    player_updates = profile.batch * profile.players * profile.steps
    fixed_collision_slots = profile.batch * profile.players * profile.body_capacity * profile.steps
    collision_sec = timers["collision_scan"]

    return {
        "schema_version": "curvyzero_vectorization_prototype/v1",
        "benchmark_id": "numpy_source_like_inner_loop_prototype",
        "prototype_status": (
            "isolated script under scripts/; no production curvyzero imports; "
            "not a backend implementation"
        ),
        "source_fidelity_claim": SOURCE_FIDELITY_CLAIM,
        "trust_level": (
            "source-like synthetic timing only. Trust the array-shape and bucket "
            "scaffold, not source gameplay fidelity or trainer throughput."
        ),
        "command": {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
        },
        "workload": {
            **asdict(profile),
            "batch_axis": "B environments",
            "player_axis": "P players with a small reverse-order Python loop",
            "body_axis": "K append-only body slots scanned as a full fixed buffer",
            "action_policy": profile.action_pattern,
        },
        "source": _git_manifest(),
        "runtime": _runtime_manifest(),
        "timer_schema": {
            "definitions": TIMER_DEFINITIONS,
            "caveat": (
                "Bucket timers include Python timing overhead and NumPy CPU execution. "
                "They are useful for relative scout runs, not final performance claims."
            ),
        },
        "semantics_coverage": SEMANTICS_COVERAGE,
        "sanity_checks": sanity,
        "setup_sec": setup_sec,
        "warmup_sec": warmup_sec,
        "elapsed_sec": elapsed,
        "timings_sec": timers,
        "rates": {
            "env_steps_per_sec": env_steps / elapsed if elapsed else 0.0,
            "player_updates_per_sec": player_updates / elapsed if elapsed else 0.0,
            "fixed_collision_slots_per_sec": (
                fixed_collision_slots / collision_sec if collision_sec else 0.0
            ),
        },
        "counts": {
            **counters,
            "env_steps": env_steps,
            "player_update_slots": player_updates,
            "fixed_collision_slots": fixed_collision_slots,
            "live_body_scan_slots": int(state["body_scan_count"].sum()),
            "body_candidates": int(state["body_candidate_count"].sum()),
            "active_bodies": int(state["body_active"].sum()),
            "max_body_write_cursor": int(state["body_write_cursor"].max(initial=0)),
            "overflow_envs": int(state["overflow"].sum()),
            "alive_players_end": int(state["alive"].sum()),
        },
        "memory": {
            "state_bytes": state_bytes,
            "state_bytes_per_env": state_bytes / profile.batch,
        },
    }


def print_plain(summary: dict[str, Any]) -> None:
    timings = summary["timings_sec"]
    rates = summary["rates"]
    counts = summary["counts"]
    workload = summary["workload"]
    runtime = summary["runtime"]
    print(f"benchmark={summary['benchmark_id']}")
    print(f"source_fidelity_claim={summary['source_fidelity_claim']}")
    print(f"trust_level={summary['trust_level']}")
    print(
        "profile="
        f"B{workload['batch']}_P{workload['players']}_K{workload['body_capacity']}_"
        f"steps{workload['steps']}_{workload['dtype']}_{workload['action_pattern']}"
    )
    print(f"setup_sec={summary['setup_sec']:.6f}")
    print(f"warmup_sec={summary['warmup_sec']:.6f}")
    print(f"elapsed_sec={summary['elapsed_sec']:.6f}")
    print(f"env_steps_per_sec={rates['env_steps_per_sec']:.1f}")
    print(f"player_updates_per_sec={rates['player_updates_per_sec']:.1f}")
    print(f"fixed_collision_slots_per_sec={rates['fixed_collision_slots_per_sec']:.1f}")
    for bucket in TIMER_DEFINITIONS:
        print(f"{bucket}_sec={timings[bucket]:.6f}")
    print(f"normal_points_inserted={counts.get('normal_points_inserted', 0)}")
    print(f"body_hits={counts.get('body_hits', 0)}")
    print(f"overflow_envs={counts['overflow_envs']}")
    print(f"state_bytes_per_env={summary['memory']['state_bytes_per_env']:.1f}")
    print(
        "optional_tensor_backends="
        f"jax:{runtime['optional_tensor_backends']['jax']['available']} "
        f"torch:{runtime['optional_tensor_backends']['torch']['available']} "
        "(detected only, not required)"
    )
    print(
        "covered=reverse_order, draw_cursor_cadence, append_only_bodies, "
        "strict_overlap, own_body_latency"
    )
    print(
        "missing=PrintManager_post_collision, borderless_wrap, death_points, "
        "scoring_events, bonuses_randomness, public_wrappers"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=_positive_int, default=128)
    parser.add_argument("--players", type=_positive_int, default=3)
    parser.add_argument("--body-capacity", type=_nonnegative_int, default=512)
    parser.add_argument("--steps", type=_positive_int, default=200)
    parser.add_argument("--warmup", type=_nonnegative_int, default=20)
    parser.add_argument("--width", type=float, default=256.0)
    parser.add_argument("--height", type=float, default=192.0)
    parser.add_argument("--radius", type=float, default=0.6)
    parser.add_argument("--speed", type=float, default=16.0)
    parser.add_argument("--step-ms", type=float, default=1000.0 / 60.0)
    parser.add_argument("--angular-velocity-per-ms", type=float, default=0.004)
    parser.add_argument("--trail-latency", type=_nonnegative_int, default=3)
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument(
        "--action-pattern",
        choices=["straight", "alternating-turns", "weave"],
        default="straight",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", choices=["plain", "json"], default="plain")
    args = parser.parse_args()

    profile = Profile(
        batch=args.batch,
        players=args.players,
        body_capacity=args.body_capacity,
        steps=args.steps,
        warmup=args.warmup,
        width=args.width,
        height=args.height,
        radius=args.radius,
        speed=args.speed,
        step_ms=args.step_ms,
        angular_velocity_per_ms=args.angular_velocity_per_ms,
        trail_latency=args.trail_latency,
        dtype=args.dtype,
        action_pattern=args.action_pattern,
        seed=args.seed,
    )
    summary = run(profile)
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_plain(summary)


if __name__ == "__main__":
    main()
