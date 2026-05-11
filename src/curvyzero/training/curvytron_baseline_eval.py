"""Simple CurvyTron baseline survival evals.

This file intentionally avoids LightZero. It answers one question:
how long do simple hand-written policies survive in the public CurvyTron env?
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from curvyzero.env import vector_reset
from curvyzero.env.trainer_contract import RAY_CHANNEL_NAMES
from curvyzero.env.trainer_contract import RAY_COUNT
from curvyzero.env.trainer_contract import SCALAR_NAMES
from curvyzero.env.trainer_contract import ACTION_NAMES
from curvyzero.env.vector_trainer_env import VectorTrainerEnv1v1NoBonus
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.training.policy_row_mapping import NOOP_ACTION_ID


BASELINE_KINDS = (
    "straight",
    "left",
    "right",
    "split_turn",
    "weave",
    "random_legal",
    "mostly_straight",
    "wall_avoid",
    "ray_clearance",
)
ENV_SURFACES = ("trainer", "metadata")
_RAY_VALUE_COUNT = RAY_COUNT * len(RAY_CHANNEL_NAMES)
_TERMINAL_REASON_NAMES = {
    vector_reset.TERMINAL_REASON_NONE: "none",
    vector_reset.TERMINAL_REASON_SURVIVOR_WIN: "round_survivor_win",
    vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW: "round_all_dead_draw",
    vector_reset.TERMINAL_REASON_TIMEOUT_TRUNCATED: "timeout",
    vector_reset.TERMINAL_REASON_EVENT_OVERFLOW_TRUNCATED: "event_overflow_truncated",
    vector_reset.TERMINAL_REASON_BODY_OVERFLOW_TRUNCATED: "capacity_truncated",
}


@dataclass(frozen=True)
class _EpisodeResult:
    policy_kind: str
    episode_index: int
    row: int
    steps: int
    winner: int | None
    terminal_reason: str


def run_curvytron_baseline_eval(
    *,
    policy_kind: str,
    seed: int = 0,
    episodes: int = 64,
    batch_size: int = 64,
    max_steps: int = 4096,
    decision_ms: float = 300.0,
    max_ticks: int | None = None,
    mostly_straight_turn_probability: float = 0.05,
    weave_period: int = 24,
    env_surface: str = "trainer",
    observation_summary_dir: str | Path | None = None,
    observation_summary_limit: int = 0,
) -> dict[str, Any]:
    if policy_kind not in BASELINE_KINDS:
        raise ValueError(f"policy_kind must be one of: {', '.join(BASELINE_KINDS)}")
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if not 0.0 <= mostly_straight_turn_probability <= 1.0:
        raise ValueError("mostly_straight_turn_probability must be in [0, 1]")
    if weave_period < 1:
        raise ValueError("weave_period must be >= 1")
    if env_surface not in ENV_SURFACES:
        raise ValueError(f"env_surface must be one of: {', '.join(ENV_SURFACES)}")
    if policy_kind in {"wall_avoid", "ray_clearance"} and env_surface != "trainer":
        raise ValueError(f"{policy_kind!r} requires env_surface='trainer'")
    if observation_summary_limit < 0:
        raise ValueError("observation_summary_limit must be >= 0")

    started = time.perf_counter()
    resolved_max_ticks = max_steps if max_ticks is None else max_ticks
    if env_surface == "trainer":
        env = VectorTrainerEnv1v1NoBonus(
            batch_size=batch_size,
            autoreset=True,
            seed=seed,
            decision_ms=decision_ms,
            max_ticks=resolved_max_ticks,
        )
    else:
        env = VectorMultiplayerEnv(
            batch_size=batch_size,
            player_count=2,
            seed=seed,
            decision_ms=decision_ms,
            max_ticks=resolved_max_ticks,
        )
    rng = np.random.default_rng(seed)
    batch = env.reset(seed=None)
    row_episode_start_step = np.zeros(batch_size, dtype=np.int64)
    completed: list[_EpisodeResult] = []
    action_counts: Counter[int] = Counter()
    total_env_steps = 0
    observation_summaries: list[dict[str, Any]] = []

    while len(completed) < episodes and total_env_steps < max_steps * max(1, episodes):
        needs_reset = np.asarray(batch.info.get("needs_reset", np.zeros(batch_size)), dtype=bool)
        if env_surface == "metadata" and bool(needs_reset.any()):
            row_episode_start_step[needs_reset] = total_env_steps
            batch = env.autoreset_done_rows(row_mask=needs_reset)

        if len(observation_summaries) < observation_summary_limit:
            observation_summaries.extend(
                _observation_summaries(
                    batch,
                    step_index=total_env_steps,
                    limit=observation_summary_limit - len(observation_summaries),
                )
            )

        joint_action = _baseline_joint_action(
            policy_kind,
            action_mask=batch.action_mask,
            observation=batch.observation if env_surface == "trainer" else None,
            step_index=total_env_steps,
            rng=rng,
            mostly_straight_turn_probability=mostly_straight_turn_probability,
            weave_period=weave_period,
        )
        for action in joint_action.reshape(-1):
            action_counts[int(action)] += 1

        batch = env.step(joint_action)
        total_env_steps += 1

        done = np.asarray(batch.done, dtype=bool)
        if not bool(done.any()):
            continue

        step_index = np.asarray(
            batch.info.get("step_index", np.full(batch_size, total_env_steps)),
            dtype=np.int64,
        )
        winners = np.asarray(batch.info.get("winner", np.full(batch_size, -1)), dtype=np.int64)
        reasons = _terminal_reason_names(batch.info, batch_size=batch_size)
        for row in np.flatnonzero(done):
            if len(completed) >= episodes:
                break
            if env_surface == "metadata":
                # step_index is one-based after env.step because the env increments
                # its public episode step counter before returning terminal info.
                steps = int(step_index[int(row)])
            else:
                steps = int(total_env_steps - row_episode_start_step[int(row)])
            winner_raw = int(winners[int(row)])
            completed.append(
                _EpisodeResult(
                    policy_kind=policy_kind,
                    episode_index=len(completed),
                    row=int(row),
                    steps=steps,
                    winner=winner_raw if winner_raw >= 0 else None,
                    terminal_reason=str(reasons[int(row)]),
                )
            )
        if env_surface == "trainer":
            row_episode_start_step[done] = total_env_steps

    elapsed_sec = time.perf_counter() - started
    steps = [item.steps for item in completed]
    result = {
        "ok": len(completed) == episodes,
        "policy_kind": policy_kind,
        "env_surface": env_surface,
        "seed": int(seed),
        "episodes_requested": int(episodes),
        "episodes_completed": int(len(completed)),
        "batch_size": int(batch_size),
        "max_steps": int(max_steps),
        "max_ticks": int(resolved_max_ticks),
        "decision_ms": float(decision_ms),
        "action_names": list(ACTION_NAMES),
        "action_counts": {str(k): int(v) for k, v in sorted(action_counts.items())},
        "action_histogram": {
            ACTION_NAMES[int(k)]: int(v)
            for k, v in sorted(action_counts.items())
            if 0 <= int(k) < len(ACTION_NAMES)
        },
        "survival_steps": _step_summary(steps),
        "terminal_reasons": dict(Counter(item.terminal_reason for item in completed)),
        "winners": dict(Counter("draw" if item.winner is None else str(item.winner) for item in completed)),
        "observation_summary_count": int(len(observation_summaries)),
        "observation_summary_path": None,
        "observation_summaries": observation_summaries,
        "episodes": [item.__dict__ for item in completed[: min(len(completed), 128)]],
        "episodes_truncated": len(completed) > 128,
        "elapsed_sec": round(elapsed_sec, 6),
        "episodes_per_sec": round(len(completed) / elapsed_sec, 6) if elapsed_sec > 0 else None,
    }
    if observation_summary_dir is not None and observation_summaries:
        out_dir = Path(observation_summary_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{policy_kind}_observation_summaries.json"
        out_path.write_text(json.dumps(_plain(observation_summaries), indent=2, sort_keys=True) + "\n")
        result["observation_summary_path"] = str(out_path)
    return _plain(result)


def run_curvytron_baseline_suite(
    *,
    policy_kinds: list[str] | None = None,
    seed: int = 0,
    episodes: int = 64,
    batch_size: int = 64,
    max_steps: int = 4096,
    decision_ms: float = 300.0,
    max_ticks: int | None = None,
    env_surface: str = "trainer",
    observation_summary_dir: str | Path | None = None,
    observation_summary_limit: int = 0,
) -> dict[str, Any]:
    kinds = list(BASELINE_KINDS if policy_kinds is None else policy_kinds)
    results = [
        run_curvytron_baseline_eval(
            policy_kind=kind,
            seed=seed,
            episodes=episodes,
            batch_size=batch_size,
            max_steps=max_steps,
            decision_ms=decision_ms,
            max_ticks=max_ticks,
            env_surface=env_surface,
            observation_summary_dir=observation_summary_dir,
            observation_summary_limit=observation_summary_limit,
        )
        for kind in kinds
    ]
    return {
        "ok": all(bool(item["ok"]) for item in results),
        "seed": int(seed),
        "episodes_per_policy": int(episodes),
        "batch_size": int(batch_size),
        "max_steps": int(max_steps),
        "env_surface": env_surface,
        "results": results,
        "ranking_by_mean_steps": sorted(
            [
                {
                    "policy_kind": item["policy_kind"],
                    "mean_steps": item["survival_steps"]["mean"],
                    "median_steps": item["survival_steps"]["median"],
                    "max_steps": item["survival_steps"]["max"],
                }
                for item in results
            ],
            key=lambda row: (
                -float(row["mean_steps"] or 0.0),
                -float(row["median_steps"] or 0.0),
            ),
        ),
    }


def _baseline_joint_action(
    policy_kind: str,
    *,
    action_mask: np.ndarray,
    observation: np.ndarray | None,
    step_index: int,
    rng: np.random.Generator,
    mostly_straight_turn_probability: float,
    weave_period: int,
) -> np.ndarray:
    mask = np.asarray(action_mask, dtype=bool)
    if mask.ndim != 3 or mask.shape[1:] != (2, len(ACTION_NAMES)):
        raise ValueError("action_mask must have shape [B,2,3]")
    actions = np.full(mask.shape[:2], NOOP_ACTION_ID, dtype=np.int16)
    active_slots = np.argwhere(mask.any(axis=2))
    for row, player in active_slots:
        legal = np.flatnonzero(mask[int(row), int(player)])
        if legal.size == 0:
            continue
        preferred = _preferred_action(
            policy_kind,
            row=int(row),
            player=int(player),
            observation=observation,
            step_index=int(step_index),
            rng=rng,
            mostly_straight_turn_probability=mostly_straight_turn_probability,
            weave_period=weave_period,
        )
        if preferred in legal:
            actions[int(row), int(player)] = int(preferred)
        else:
            actions[int(row), int(player)] = int(legal[0])
    return actions


def _preferred_action(
    policy_kind: str,
    *,
    row: int,
    player: int,
    observation: np.ndarray | None,
    step_index: int,
    rng: np.random.Generator,
    mostly_straight_turn_probability: float,
    weave_period: int,
) -> int:
    if policy_kind == "straight":
        return 1
    if policy_kind == "left":
        return 0
    if policy_kind == "right":
        return 2
    if policy_kind == "split_turn":
        return 0 if player == 0 else 2
    if policy_kind == "weave":
        phase = (step_index // weave_period + player) % 2
        return 0 if phase == 0 else 2
    if policy_kind == "random_legal":
        return int(rng.integers(0, len(ACTION_NAMES)))
    if policy_kind == "mostly_straight":
        if float(rng.random()) < mostly_straight_turn_probability:
            return int(rng.choice([0, 2]))
        return 1
    if policy_kind == "wall_avoid":
        return _clearance_action(observation, row=row, player=player, channel_indices=(0,))
    if policy_kind == "ray_clearance":
        return _clearance_action(observation, row=row, player=player, channel_indices=(0, 1, 2, 3))
    raise ValueError(f"unknown policy_kind {policy_kind!r}")


def _clearance_action(
    observation: np.ndarray | None,
    *,
    row: int,
    player: int,
    channel_indices: tuple[int, ...],
) -> int:
    if observation is None:
        raise ValueError("observation is required for clearance policies")
    rays, _scalars = _split_observation(np.asarray(observation)[int(row), int(player)])
    hazard = rays[:, list(channel_indices)].min(axis=1)
    # Ray angles are ego-left/counter-clockwise from forward. Right turn maps to
    # the high-angle wraparound sector.
    scores = np.asarray(
        [
            float(np.mean(hazard[[1, 2, 3, 4]])),
            float(np.mean(hazard[[23, 0, 1]])) + 0.015,
            float(np.mean(hazard[[20, 21, 22, 23]])),
        ],
        dtype=np.float64,
    )
    return int(np.argmax(scores))


def _split_observation(flat_observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(flat_observation, dtype=np.float32)
    rays = flat[:_RAY_VALUE_COUNT].reshape(RAY_COUNT, len(RAY_CHANNEL_NAMES))
    scalars = flat[_RAY_VALUE_COUNT : _RAY_VALUE_COUNT + len(SCALAR_NAMES)]
    return rays, scalars


def _observation_summaries(
    batch: Any,
    *,
    step_index: int,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    observation = getattr(batch, "observation", None)
    if observation is None or np.asarray(observation).ndim != 3:
        return []
    mask = np.asarray(batch.action_mask, dtype=bool)
    summaries: list[dict[str, Any]] = []
    for row in range(int(observation.shape[0])):
        for player in range(int(observation.shape[1])):
            if len(summaries) >= limit:
                return summaries
            rays, scalars = _split_observation(observation[row, player])
            per_channel_min = rays.min(axis=0)
            summaries.append(
                {
                    "step_index": int(step_index),
                    "row": int(row),
                    "player": int(player),
                    "legal_actions": [
                        ACTION_NAMES[action]
                        for action in np.flatnonzero(mask[row, player])
                    ],
                    "ray_channel_min": {
                        name: float(per_channel_min[index])
                        for index, name in enumerate(RAY_CHANNEL_NAMES)
                    },
                    "forward_rays": {
                        name: float(rays[0, index])
                        for index, name in enumerate(RAY_CHANNEL_NAMES)
                    },
                    "scalar_values": {
                        name: float(scalars[index])
                        for index, name in enumerate(SCALAR_NAMES)
                    },
                }
            )
    return summaries


def _terminal_reason_names(info: dict[str, Any], *, batch_size: int) -> np.ndarray:
    if "terminal_reason_name" in info:
        return np.asarray(info["terminal_reason_name"], dtype=object)
    raw = np.asarray(info.get("terminal_reason", np.full(batch_size, -1)), dtype=np.int64)
    return np.asarray(
        [_TERMINAL_REASON_NAMES.get(int(value), f"unknown:{int(value)}") for value in raw],
        dtype=object,
    )


def _step_summary(steps: list[int]) -> dict[str, Any]:
    if not steps:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p10": None,
            "p90": None,
        }
    arr = np.asarray(steps, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-kinds", default=",".join(BASELINE_KINDS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--decision-ms", type=float, default=300.0)
    parser.add_argument("--env-surface", choices=ENV_SURFACES, default="trainer")
    parser.add_argument("--observation-summary-dir", default=None)
    parser.add_argument("--observation-summary-limit", type=int, default=0)
    args = parser.parse_args()
    policy_kinds = [item.strip() for item in args.policy_kinds.split(",") if item.strip()]
    print(
        json.dumps(
            run_curvytron_baseline_suite(
                policy_kinds=policy_kinds,
                seed=args.seed,
                episodes=args.episodes,
                batch_size=args.batch_size,
                max_steps=args.max_steps,
                max_ticks=args.max_ticks,
                decision_ms=args.decision_ms,
                env_surface=args.env_surface,
                observation_summary_dir=args.observation_summary_dir,
                observation_summary_limit=args.observation_summary_limit,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
