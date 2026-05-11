"""Trace and fingerprint helpers for the current Python toy environment.

These helpers describe ``curvyzero-v0`` Python behavior only. They are useful
for deterministic toy-env checks, not as proof of CurvyTron source fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping

from curvyzero.env.core import ActionMap, CurvyTronEnv

TRACE_SCOPE = "curvyzero-v0-python-toy"
TRACE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TraceFrame:
    """A small, stable snapshot of simulator state after reset or step."""

    tick: int
    positions: tuple[tuple[float, ...], ...]
    headings: tuple[float, ...]
    alive: tuple[bool, ...]
    scores: tuple[int, ...]
    round_scores: tuple[int, ...]
    rewards: dict[str, float] | None = None
    terminated: dict[str, bool] | None = None
    truncated: dict[str, bool] | None = None

    @classmethod
    def from_env(cls, env: CurvyTronEnv, step_result: Any | None = None) -> "TraceFrame":
        if env.state is None:
            raise RuntimeError("reset must be called before tracing env state")

        state = env.state
        return cls(
            tick=int(state.tick),
            positions=tuple(
                tuple(float(value) for value in position) for position in state.positions.tolist()
            ),
            headings=tuple(float(value) for value in state.headings.tolist()),
            alive=tuple(bool(value) for value in state.alive.tolist()),
            scores=tuple(0 for _ in range(env.config.players)),
            round_scores=tuple(0 for _ in range(env.config.players)),
            rewards=_maybe_float_mapping(getattr(step_result, "rewards", None)),
            terminated=_maybe_bool_mapping(getattr(step_result, "terminated", None)),
            truncated=_maybe_bool_mapping(getattr(step_result, "truncated", None)),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "tick": self.tick,
            "positions": self.positions,
            "headings": self.headings,
            "alive": self.alive,
            "scores": self.scores,
            "roundScores": self.round_scores,
            "rewards": self.rewards,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class EnvTrace:
    """Trace plus canonical metadata for a scripted action sequence."""

    scope: str
    schema_version: int
    ruleset: str
    rules_hash: str
    seed: int | None
    scripted_actions: tuple[dict[str, int], ...]
    frames: tuple[TraceFrame, ...]

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_payload(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "schema_version": self.schema_version,
            "ruleset": self.ruleset,
            "rules_hash": self.rules_hash,
            "seed": self.seed,
            "scripted_actions": self.scripted_actions,
            "frames": tuple(frame.to_payload() for frame in self.frames),
        }


def trace_scripted_actions(
    env: CurvyTronEnv,
    action_sequence: Iterable[ActionMap],
    *,
    seed: int | None = None,
) -> EnvTrace:
    """Run a scripted action sequence and return a deterministic toy-env trace."""

    scripted_actions = tuple(_normalize_actions(actions) for actions in action_sequence)

    env.reset(seed=seed)
    frames = [TraceFrame.from_env(env)]
    for actions in scripted_actions:
        step_result = env.step(actions)
        frames.append(TraceFrame.from_env(env, step_result))
        if any(step_result.terminated.values()) or any(step_result.truncated.values()):
            break

    return EnvTrace(
        scope=TRACE_SCOPE,
        schema_version=TRACE_SCHEMA_VERSION,
        ruleset=env.config.ruleset,
        rules_hash=env.config.rules_hash,
        seed=seed,
        scripted_actions=scripted_actions,
        frames=tuple(frames),
    )


def fingerprint_scripted_actions(
    env: CurvyTronEnv,
    action_sequence: Iterable[ActionMap],
    *,
    seed: int | None = None,
) -> str:
    """Return the stable trace fingerprint for a scripted toy-env rollout."""

    return trace_scripted_actions(env, action_sequence, seed=seed).fingerprint


def _normalize_actions(actions: Mapping[str, int]) -> dict[str, int]:
    return {str(agent): int(action) for agent, action in sorted(actions.items())}


def _maybe_float_mapping(values: Mapping[str, float] | None) -> dict[str, float] | None:
    if values is None:
        return None
    return {str(key): float(value) for key, value in sorted(values.items())}


def _maybe_bool_mapping(values: Mapping[str, bool] | None) -> dict[str, bool] | None:
    if values is None:
        return None
    return {str(key): bool(value) for key, value in sorted(values.items())}
