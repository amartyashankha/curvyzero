"""Profile CurvyTron source-state rendering over fixed trajectory lengths.

This is a local Optimizer tool. It uses the current source-state LightZero env
wrapper, forces no-death rollouts, and times the gray64 render entry points so
we can separate render time from the rest of env stepping.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvyzero.env.trainer_contract import ACTION_ID_TO_SOURCE_MOVE  # noqa: E402
from curvyzero.env import vector_runtime  # noqa: E402
from curvyzero.training import curvyzero_source_state_visual_survival_lightzero_env as env_mod  # noqa: E402,E501
from curvyzero.training.curvyzero_source_state_visual_survival_lightzero_env import (  # noqa: E402,E501
    CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv,
    OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
    OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
    OPPONENT_RUNTIME_MODE_NORMAL,
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
    SOURCE_STATE_TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
    SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES,
)


REPORT_SCHEMA_ID = "curvyzero_render_trajectory_profile/v0"
DEFAULT_LENGTHS = (100, 200, 500, 1000, 2000)
DEFAULT_RENDER_MODES = (
    SOURCE_STATE_TRAIL_RENDER_MODE_BROWSER_LINES,
    SOURCE_STATE_TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int, default=list(DEFAULT_LENGTHS))
    parser.add_argument(
        "--render-modes",
        nargs="+",
        default=list(DEFAULT_RENDER_MODES),
        choices=list(DEFAULT_RENDER_MODES),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--source-max-steps", type=int, default=65536)
    parser.add_argument(
        "--opponent-runtime-mode",
        choices=[OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP, OPPONENT_RUNTIME_MODE_NORMAL],
        default=OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
    )
    parser.add_argument(
        "--policy",
        choices=["wall_avoidant", "straight", "cycle"],
        default="wall_avoidant",
    )
    parser.add_argument("--natural-bonus-spawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--cell-output",
        type=Path,
        default=None,
        help="Optional JSONL path written after each completed cell.",
    )
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    cell_output = args.cell_output
    if cell_output is None and args.output is not None:
        cell_output = args.output.with_name(f"{args.output.stem}.cells.jsonl")
    report = run_profile_grid(
        lengths=args.lengths,
        render_modes=args.render_modes,
        repeats=args.repeats,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        source_max_steps=args.source_max_steps,
        opponent_runtime_mode=args.opponent_runtime_mode,
        policy=args.policy,
        natural_bonus_spawn=args.natural_bonus_spawn,
        cell_output=cell_output,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.markdown:
        print(markdown_report(report))
    else:
        print(text)
    return 0


def run_profile_grid(
    *,
    lengths: Iterable[int],
    render_modes: Iterable[str],
    repeats: int,
    warmup_steps: int,
    seed: int,
    source_max_steps: int,
    opponent_runtime_mode: str,
    policy: str,
    natural_bonus_spawn: bool,
    cell_output: Path | None = None,
) -> dict[str, Any]:
    checked_lengths = [int(length) for length in lengths]
    if not checked_lengths or any(length < 1 for length in checked_lengths):
        raise ValueError("lengths must be positive")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    cells: list[dict[str, Any]] = []
    if cell_output is not None:
        cell_output.parent.mkdir(parents=True, exist_ok=True)
        cell_output.write_text("", encoding="utf-8")
    for render_mode in render_modes:
        for length in checked_lengths:
            print(
                f"profile cell start render={render_mode} steps={length} repeats={repeats}",
                file=sys.stderr,
                flush=True,
            )
            runs = []
            for repeat in range(repeats):
                cell_seed = int(seed) + repeat * 100_000 + length
                if warmup_steps:
                    _run_one_rollout(
                        steps=warmup_steps,
                        render_mode=render_mode,
                        seed=cell_seed - 1,
                        source_max_steps=max(source_max_steps, warmup_steps + 8),
                        opponent_runtime_mode=opponent_runtime_mode,
                        policy=policy,
                        natural_bonus_spawn=natural_bonus_spawn,
                    )
                runs.append(
                    _run_one_rollout(
                        steps=length,
                        render_mode=render_mode,
                        seed=cell_seed,
                        source_max_steps=max(source_max_steps, length + 8),
                        opponent_runtime_mode=opponent_runtime_mode,
                        policy=policy,
                        natural_bonus_spawn=natural_bonus_spawn,
                    )
                )
            cell = _summarize_cell(render_mode=render_mode, length=length, runs=runs)
            cells.append(cell)
            if cell_output is not None:
                _append_jsonl(cell_output, cell)
            print(
                (
                    f"profile cell done render={render_mode} steps={length} "
                    f"wall={cell['wall_sec_median']:.3f}s "
                    f"render={cell['render_sec_median']:.3f}s "
                    f"render_frac={100.0 * cell['render_fraction_of_wall']:.1f}%"
                ),
                file=sys.stderr,
                flush=True,
            )
    return {
        "schema_id": REPORT_SCHEMA_ID,
        "created_at_unix": time.time(),
        "scope": "local source_state_fixed_opponent env-only no-death trajectory profile",
        "live_training_path_checked": (
            "current Coach batches use stock --mode train source_state_fixed_opponent; "
            "this profiler exercises that env wrapper locally, not live Modal training"
        ),
        "surface_caveat": (
            "fixed-opponent LightZero wrapper only; includes scalar full render and, "
            "when enabled by that wrapper, the dirty/player-perspective render cache. "
            "Does not measure the newer multiplayer trainer surface, MCTS/search, "
            "learner, replay, subprocess collection, checkpoints, eval, or GIFs"
        ),
        "includes_policy_search": False,
        "includes_learner": False,
        "render_timing": (
            "render_sec is scalar_render_sec plus perspective_render_sec. These wrap "
            "curvyzero_source_state_visual_survival_lightzero_env."
            "render_source_state_canvas_gray64 and "
            "render_source_state_canvas_gray64_player_perspectives"
        ),
        "env_timing": "profile_env_timing_sec from the current source-state env wrapper",
        "config": {
            "lengths": checked_lengths,
            "render_modes": list(render_modes),
            "repeats": int(repeats),
            "warmup_steps": int(warmup_steps),
            "seed": int(seed),
            "source_max_steps": int(source_max_steps),
            "death_mode": vector_runtime.DEATH_MODE_PROFILE_NO_DEATH,
            "opponent_runtime_mode": opponent_runtime_mode,
            "opponent_policy_kind": OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
            "policy": policy,
            "natural_bonus_spawn": bool(natural_bonus_spawn),
            "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        },
        "cells": cells,
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _run_one_rollout(
    *,
    steps: int,
    render_mode: str,
    seed: int,
    source_max_steps: int,
    opponent_runtime_mode: str,
    policy: str,
    natural_bonus_spawn: bool,
) -> dict[str, Any]:
    stats = _RenderStats()
    cfg = {
        "seed": int(seed),
        "source_max_steps": int(source_max_steps),
        "death_mode": vector_runtime.DEATH_MODE_PROFILE_NO_DEATH,
        "source_state_trail_render_mode": render_mode,
        "opponent_policy_kind": OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        "opponent_runtime_mode": opponent_runtime_mode,
        "reward_variant": REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        "natural_bonus_spawn": bool(natural_bonus_spawn),
        "profile_env_timing_enabled": True,
        "telemetry_stride": 1,
    }
    with _timed_gray64_render(stats):
        env = CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv(cfg)
        reset_started = time.perf_counter()
        env.reset(seed=int(seed))
        reset_wall_sec = time.perf_counter() - reset_started
        stats.reset()
        timing_sums: Counter[str] = Counter()
        terminal_count = 0
        started = time.perf_counter()
        for step_index in range(int(steps)):
            action = _select_action(env, policy=policy, step_index=step_index)
            timestep = env.step(action)
            info = getattr(timestep, "info", {})
            profile_timing = info.get("profile_env_timing_sec", {})
            if isinstance(profile_timing, dict):
                for key, value in profile_timing.items():
                    timing_sums[str(key)] += float(value)
            if bool(getattr(timestep, "done", False)):
                terminal_count += 1
                env.reset(seed=int(seed) + step_index + 1)
        wall_sec = time.perf_counter() - started
    return {
        "steps": int(steps),
        "seed": int(seed),
        "wall_sec": float(wall_sec),
        "reset_wall_sec_excluded": float(reset_wall_sec),
        "render_sec": float(stats.seconds),
        "render_calls": int(stats.calls),
        "scalar_render_sec": float(stats.scalar_seconds),
        "scalar_render_calls": int(stats.scalar_calls),
        "perspective_render_sec": float(stats.perspective_seconds),
        "perspective_render_calls": int(stats.perspective_calls),
        "scalar_dirty_render_cache_stats": _cache_stats(
            getattr(env, "_scalar_dirty_render_cache", None)
        ),
        "scalar_trail_layer_cache_stats": _cache_stats(
            getattr(env, "_scalar_trail_layer_cache", None)
        ),
        "timing_sums": dict(sorted(timing_sums.items())),
        "terminal_count": int(terminal_count),
    }


def _select_action(
    env: CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv,
    *,
    policy: str,
    step_index: int,
) -> int:
    if policy == "straight":
        return 1
    if policy == "cycle":
        return (0, 1, 2, 1)[step_index % 4]
    return _wall_avoidant_action(env)


def _wall_avoidant_action(env: CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv) -> int:
    state = env._env.state  # noqa: SLF001 - local profiler over the current wrapper.
    player = int(env.ego_player_index)
    pos = state["pos"][0, player]
    radius = float(state["radius"][0, player])
    map_size = float(state["map_size"][0])
    clearances = np.asarray(
        [
            float(pos[0] - radius),
            float(map_size - (pos[0] + radius)),
            float(pos[1] - radius),
            float(map_size - (pos[1] + radius)),
        ],
        dtype=np.float64,
    )
    safe_margin = 12.0
    if float(clearances.min()) > safe_margin:
        return 1
    danger_left = max(0.0, safe_margin - float(clearances[0]))
    danger_right = max(0.0, safe_margin - float(clearances[1]))
    danger_top = max(0.0, safe_margin - float(clearances[2]))
    danger_bottom = max(0.0, safe_margin - float(clearances[3]))
    away = np.asarray((danger_left - danger_right, danger_top - danger_bottom))
    norm = float(np.hypot(away[0], away[1]))
    if norm <= 1e-9:
        center = np.asarray((map_size * 0.5, map_size * 0.5), dtype=np.float64)
        away = center - np.asarray(pos, dtype=np.float64)
        norm = float(np.hypot(away[0], away[1]))
    if norm <= 1e-9:
        return 1
    away = away / norm
    heading = float(state["heading"][0, player])
    angular_velocity = float(state["angular_velocity_per_ms"][0, player])
    decision_ms = float(env._decision_ms)  # noqa: SLF001

    def score(action_id: int) -> float:
        source_move = float(ACTION_ID_TO_SOURCE_MOVE[int(action_id)])
        if "inverse" in state and bool(state["inverse"][0, player]):
            source_move = -source_move
        post = heading + source_move * angular_velocity * decision_ms
        return float(np.cos(post) * away[0] + np.sin(post) * away[1])

    left_score = score(0)
    straight_score = score(1)
    right_score = score(2)
    scores = (left_score, straight_score, right_score)
    return int(max(range(3), key=lambda action_id: scores[action_id]))


def _cache_stats(cache: Any) -> dict[str, Any] | None:
    stats = getattr(cache, "stats", None)
    as_dict = getattr(stats, "as_dict", None)
    if callable(as_dict):
        return _json_safe(as_dict())
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    return str(value)


class _RenderStats:
    def __init__(self) -> None:
        self.scalar_seconds = 0.0
        self.scalar_calls = 0
        self.perspective_seconds = 0.0
        self.perspective_calls = 0

    def reset(self) -> None:
        self.scalar_seconds = 0.0
        self.scalar_calls = 0
        self.perspective_seconds = 0.0
        self.perspective_calls = 0

    @property
    def seconds(self) -> float:
        return self.scalar_seconds + self.perspective_seconds

    @property
    def calls(self) -> int:
        return self.scalar_calls + self.perspective_calls


@contextmanager
def _timed_gray64_render(stats: _RenderStats):
    original_scalar = env_mod.render_source_state_canvas_gray64
    original_perspective = env_mod.render_source_state_canvas_gray64_player_perspectives

    def wrapped_scalar(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_scalar(*args, **kwargs)
        finally:
            stats.scalar_seconds += time.perf_counter() - started
            stats.scalar_calls += 1

    def wrapped_perspective(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_perspective(*args, **kwargs)
        finally:
            stats.perspective_seconds += time.perf_counter() - started
            stats.perspective_calls += 1

    env_mod.render_source_state_canvas_gray64 = wrapped_scalar
    env_mod.render_source_state_canvas_gray64_player_perspectives = wrapped_perspective
    try:
        yield
    finally:
        env_mod.render_source_state_canvas_gray64 = original_scalar
        env_mod.render_source_state_canvas_gray64_player_perspectives = original_perspective


def _summarize_cell(
    *,
    render_mode: str,
    length: int,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    walls = [float(run["wall_sec"]) for run in runs]
    renders = [float(run["render_sec"]) for run in runs]
    scalar_renders = [float(run.get("scalar_render_sec", 0.0)) for run in runs]
    perspective_renders = [
        float(run.get("perspective_render_sec", 0.0)) for run in runs
    ]
    observation = [
        float(run["timing_sums"].get("observation_sec", 0.0)) for run in runs
    ]
    vector = [float(run["timing_sums"].get("vector_step_sec", 0.0)) for run in runs]
    opponent = [
        float(run["timing_sums"].get("opponent_action_sec", 0.0)) for run in runs
    ]
    reward = [float(run["timing_sums"].get("reward_sec", 0.0)) for run in runs]
    physical = [
        float(run["timing_sums"].get("physical_loop_sec", 0.0)) for run in runs
    ]
    render_median = statistics.median(renders)
    scalar_render_median = statistics.median(scalar_renders)
    perspective_render_median = statistics.median(perspective_renders)
    wall_median = statistics.median(walls)
    observation_median = statistics.median(observation)
    vector_median = statistics.median(vector)
    opponent_median = statistics.median(opponent)
    reward_median = statistics.median(reward)
    physical_median = statistics.median(physical)
    return {
        "render_mode": render_mode,
        "steps": int(length),
        "repeats": len(runs),
        "wall_sec_median": wall_median,
        "steps_per_sec_median": float(length) / wall_median if wall_median > 0 else None,
        "render_sec_median": render_median,
        "scalar_render_sec_median": scalar_render_median,
        "perspective_render_sec_median": perspective_render_median,
        "render_fraction_of_wall": render_median / wall_median if wall_median > 0 else None,
        "other_sec_median": wall_median - render_median,
        "observation_sec_median": observation_median,
        "observation_fraction_of_wall": (
            observation_median / wall_median if wall_median > 0 else None
        ),
        "vector_step_sec_median": vector_median,
        "physical_loop_sec_median": physical_median,
        "opponent_action_sec_median": opponent_median,
        "reward_sec_median": reward_median,
        "render_calls_median": statistics.median(
            [int(run["render_calls"]) for run in runs]
        ),
        "scalar_render_calls_median": statistics.median(
            [int(run.get("scalar_render_calls", 0)) for run in runs]
        ),
        "perspective_render_calls_median": statistics.median(
            [int(run.get("perspective_render_calls", 0)) for run in runs]
        ),
        "scalar_dirty_render_cache_stats_last": runs[-1].get(
            "scalar_dirty_render_cache_stats"
        ),
        "scalar_trail_layer_cache_stats_last": runs[-1].get(
            "scalar_trail_layer_cache_stats"
        ),
        "terminal_count_total": int(sum(int(run["terminal_count"]) for run in runs)),
        "runs": runs,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# {REPORT_SCHEMA_ID}",
        "",
        "Scope: local env-only no-death profile of the current source-state fixed-opponent wrapper.",
        "",
    ]
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for cell in report["cells"]:
        by_mode.setdefault(str(cell["render_mode"]), []).append(cell)
    for mode, cells in by_mode.items():
        lines.extend(
            [
                f"## {mode}",
                "",
                "| steps | wall s | steps/s | render s | render % | scalar s | perspective s | observation s | vector step s | other s |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for cell in sorted(cells, key=lambda item: int(item["steps"])):
            lines.append(
                "| "
                f"{cell['steps']} | "
                f"{cell['wall_sec_median']:.3f} | "
                f"{cell['steps_per_sec_median']:.1f} | "
                f"{cell['render_sec_median']:.3f} | "
                f"{100.0 * cell['render_fraction_of_wall']:.1f}% | "
                f"{cell['scalar_render_sec_median']:.3f} | "
                f"{cell['perspective_render_sec_median']:.3f} | "
                f"{cell['observation_sec_median']:.3f} | "
                f"{cell['vector_step_sec_median']:.3f} | "
                f"{cell['other_sec_median']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
