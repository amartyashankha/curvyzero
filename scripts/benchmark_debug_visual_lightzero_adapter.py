"""Time the no-train debug visual LightZero-shaped CurvyZero env adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvyzero.training.curvyzero_debug_visual_lightzero_env import (  # noqa: E402
    CurvyZeroDebugVisualLightZeroEnv,
    LIGHTZERO_DEBUG_VISUAL_ENV_ID,
    LIGHTZERO_DEBUG_VISUAL_ENV_TYPE,
)
from curvyzero.training.curvyzero_debug_visual_lightzero_smoke import (  # noqa: E402
    ADAPTER_IMPL_ID,
)
from curvyzero.training.curvytron_visual_observation import (  # noqa: E402
    DebugOccupancyGray64FrameStack,
)


BENCHMARK_ID = "curvyzero_debug_visual_lightzero_adapter_timing/v0"
CAVEAT = (
    "No-train debug visual adapter timing only: direct reset/step/rendered "
    "float32[1,64,64] LightZero-shaped env payloads with optional optimizer-owned "
    "frame-stack plumbing; no policy search, replay, learner, ALE, or source visual "
    "fidelity claim."
)


def run_benchmark(
    *,
    steps: int,
    seed: int,
    action_policy: str,
    fixed_action: int,
    source_step_ms: float,
    source_max_steps: int,
    auto_reset_on_done: bool,
    stack: bool = False,
    stack_copy: bool = True,
) -> dict[str, Any]:
    env = CurvyZeroDebugVisualLightZeroEnv(
        {
            "seed": seed,
            "dynamic_seed": False,
            "source_step_ms": source_step_ms,
            "source_max_steps": source_max_steps,
        }
    )
    rng = np.random.default_rng(seed)
    reset_count = 0
    reset_sec = 0.0
    step_sec = 0.0
    frame_stack_update_copy_sec = 0.0
    done_count = 0
    rewards: list[float] = []
    last_timestep: Any | None = None
    frame_stack = DebugOccupancyGray64FrameStack() if stack else None
    last_stack_payload: np.ndarray | None = None

    reset_started = time.perf_counter()
    reset_obs = env.reset(seed=seed)
    reset_sec += time.perf_counter() - reset_started
    reset_count += 1

    loop_started = time.perf_counter()
    try:
        for step_index in range(steps):
            if action_policy == "random":
                action = int(rng.integers(3))
            elif action_policy == "env-random":
                action = int(env.random_action())
            else:
                action = int(fixed_action)

            step_started = time.perf_counter()
            last_timestep = env.step(np.int64(action))
            step_sec += time.perf_counter() - step_started

            if frame_stack is not None:
                frame_stack_started = time.perf_counter()
                last_stack_payload = frame_stack.update(
                    np.asarray(last_timestep.obs["observation"]),
                    copy=stack_copy,
                )
                frame_stack_update_copy_sec += time.perf_counter() - frame_stack_started

            rewards.append(float(last_timestep.reward))
            if bool(last_timestep.done):
                done_count += 1
                if step_index != steps - 1 and auto_reset_on_done:
                    reset_started = time.perf_counter()
                    env.reset(seed=seed + reset_count)
                    reset_sec += time.perf_counter() - reset_started
                    reset_count += 1
    finally:
        env.close()

    loop_elapsed_sec = time.perf_counter() - loop_started
    step_obs = last_timestep.obs if last_timestep is not None else reset_obs
    reset_payload = np.asarray(reset_obs["observation"])
    step_payload = np.asarray(step_obs["observation"])
    obs_payload = last_stack_payload if last_stack_payload is not None else step_payload
    info = getattr(last_timestep, "info", None) if last_timestep is not None else None

    return {
        "schema_id": BENCHMARK_ID,
        "label": "CurvyZero debug visual LightZero adapter timing",
        "caveat": CAVEAT,
        "call_policy": "does_not_train; does_not_call_lzero_entrypoints",
        "adapter": {
            "env_id": LIGHTZERO_DEBUG_VISUAL_ENV_ID,
            "lightzero_env_type": LIGHTZERO_DEBUG_VISUAL_ENV_TYPE,
            "adapter_impl_id": ADAPTER_IMPL_ID,
            "class": type(env).__module__ + "." + type(env).__name__,
            "imports_lightzero_required": False,
        },
        "workload": {
            "steps_requested": int(steps),
            "transitions": int(len(rewards)),
            "seed": int(seed),
            "action_policy": action_policy,
            "fixed_action": int(fixed_action) if action_policy == "fixed" else None,
            "source_step_ms": float(source_step_ms),
            "source_max_steps": int(source_max_steps),
            "auto_reset_on_done": bool(auto_reset_on_done),
            "frame_stack_enabled": bool(stack),
            "frame_stack_copy": bool(stack and stack_copy),
            "reset_count": int(reset_count),
            "done_count": int(done_count),
        },
        "timed_components": {
            "env_reset": True,
            "env_step_total": True,
            "render": True,
            "frame_stack_update_copy": bool(stack),
            "stack": bool(stack),
            "policy_search": False,
            "replay": False,
            "learner": False,
        },
        "timing_sec": {
            "env_reset": reset_sec,
            "env_step_total": step_sec,
            "frame_stack_update_copy": frame_stack_update_copy_sec,
            "loop_elapsed": loop_elapsed_sec,
            "loop_overhead": max(
                0.0,
                loop_elapsed_sec - step_sec - frame_stack_update_copy_sec,
            ),
        },
        "throughput": {
            "transitions_per_sec_env_step_total": len(rewards) / step_sec if step_sec else 0.0,
            "transitions_per_sec_frame_stack_update_copy": (
                len(rewards) / frame_stack_update_copy_sec
                if frame_stack_update_copy_sec
                else 0.0
            ),
            "transitions_per_sec_loop_elapsed": (
                len(rewards) / loop_elapsed_sec if loop_elapsed_sec else 0.0
            ),
        },
        "obs_payload": {
            "reset_shape": _shape(reset_payload),
            "reset_dtype": str(reset_payload.dtype),
            "step_shape": _shape(step_payload),
            "step_dtype": str(step_payload.dtype),
            "obs_payload_shape": _shape(obs_payload),
            "obs_payload_dtype": str(obs_payload.dtype),
            "frame_stack_shape": (
                _shape(last_stack_payload) if last_stack_payload is not None else None
            ),
            "frame_stack_dtype": (
                str(last_stack_payload.dtype) if last_stack_payload is not None else None
            ),
            "min": float(obs_payload.min()) if obs_payload.size else None,
            "max": float(obs_payload.max()) if obs_payload.size else None,
        },
        "action_schema": _action_schema(env.action_space, step_obs.get("action_mask")),
        "reward_schema": _reward_schema(env.reward_space, rewards),
        "step_result": {
            "timestep_type": (
                type(last_timestep).__module__ + "." + type(last_timestep).__name__
                if last_timestep is not None
                else None
            ),
            "done": bool(getattr(last_timestep, "done", False)),
            "last_reward": float(rewards[-1]) if rewards else None,
            "info_selected": _selected_info(info),
        },
    }


def _shape(value: Any) -> list[int]:
    return [int(item) for item in np.asarray(value).shape]


def _action_schema(action_space: Any, action_mask: Any) -> dict[str, Any]:
    mask = np.asarray(action_mask) if action_mask is not None else np.asarray([])
    return {
        "space_type": _space_type(action_space),
        "n": _space_n(action_space),
        "action_mask_shape": _shape(mask),
        "action_mask_dtype": str(mask.dtype),
        "legal_actions": [0, 1, 2],
    }


def _reward_schema(reward_space: Any, rewards: list[float]) -> dict[str, Any]:
    return {
        "space_type": _space_type(reward_space),
        "shape": _space_shape(reward_space),
        "dtype": _space_dtype(reward_space),
        "observed_min": min(rewards) if rewards else None,
        "observed_max": max(rewards) if rewards else None,
        "terminal_only_nonzero": True,
    }


def _selected_info(info: Any) -> dict[str, Any]:
    if not isinstance(info, dict):
        return {}
    keys = (
        "surface",
        "truth_level",
        "source_fidelity_level",
        "source_backed_observation_fidelity",
        "uses_ale",
        "ale_usage",
        "frame_stack_owner",
        "adapter_timestep",
        "done",
        "terminated",
        "truncated",
    )
    return {key: _plain(info.get(key)) for key in keys if key in info}


def _space_type(space: Any) -> str | None:
    if isinstance(space, dict):
        return str(space.get("type"))
    return type(space).__name__ if space is not None else None


def _space_n(space: Any) -> int | None:
    if isinstance(space, dict):
        value = space.get("n")
    else:
        value = getattr(space, "n", None)
    return int(value) if value is not None else None


def _space_shape(space: Any) -> list[int] | None:
    if isinstance(space, dict):
        value = space.get("shape")
    else:
        value = getattr(space, "shape", None)
    return [int(item) for item in value] if value is not None else None


def _space_dtype(space: Any) -> str | None:
    if isinstance(space, dict):
        value = space.get("dtype")
    else:
        value = getattr(space, "dtype", None)
    return str(value) if value is not None else None


def _plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=_positive_int, default=64)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument(
        "--action-policy",
        choices=("fixed", "random", "env-random"),
        default="fixed",
    )
    parser.add_argument("--fixed-action", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--source-step-ms", type=_nonnegative_float, default=1000.0 / 60.0)
    parser.add_argument("--source-max-steps", type=_nonnegative_int, default=2000)
    parser.add_argument("--auto-reset-on-done", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--stack",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also time optimizer-owned 4x64x64 frame-stack update/copy plumbing.",
    )
    parser.add_argument(
        "--stack-copy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy the updated frame stack payload when --stack is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_benchmark(
        steps=int(args.steps),
        seed=int(args.seed),
        action_policy=str(args.action_policy),
        fixed_action=int(args.fixed_action),
        source_step_ms=float(args.source_step_ms),
        source_max_steps=int(args.source_max_steps),
        auto_reset_on_done=bool(args.auto_reset_on_done),
        stack=bool(args.stack),
        stack_copy=bool(args.stack_copy),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
