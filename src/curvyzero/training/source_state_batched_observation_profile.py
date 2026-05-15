"""Profile-only batched source-state observation facade.

This module is a local timing scaffold for the future batched renderer shape.
It owns a vector env batch, updates policy-shaped ``[4, 64, 64]`` stacks, and
keeps pack/render/readback timing slots visible while the only implementation is
the trusted CPU oracle.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from typing import Protocol
import time

import numpy as np

from curvyzero.env import vector_runtime
from curvyzero.env.observation_surface_contract import DEFAULT_POLICY_OBSERVATION_BACKEND
from curvyzero.env.observation_surface_contract import POLICY_BONUS_RENDER_MODE
from curvyzero.env.observation_surface_contract import POLICY_FRAME_STACK_DEPTH
from curvyzero.env.observation_surface_contract import POLICY_OBSERVATION_BACKEND_CPU
from curvyzero.env.observation_surface_contract import POLICY_OBSERVATION_BACKEND_GPU
from curvyzero.env.observation_surface_contract import POLICY_STACK_SHAPE
from curvyzero.env.observation_surface_contract import POLICY_TRAIL_RENDER_MODE
from curvyzero.env.vector_multiplayer_env import ACTION_COUNT
from curvyzero.env.vector_multiplayer_env import DEFAULT_BODY_CAPACITY
from curvyzero.env.vector_multiplayer_env import SOURCE_PHYSICS_STEP_MS
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SHAPE
from curvyzero.env.vector_visual_observation import SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
from curvyzero.env.vector_visual_observation import SourceStateGray64DownsampleScratch
from curvyzero.env.vector_visual_observation import render_source_state_canvas_gray64


SOURCE_STATE_BATCHED_OBSERVATION_PROFILE_IMPL_ID = (
    "curvyzero_source_state_batched_observation_profile/v0"
)
SOURCE_STATE_BATCHED_OBSERVATION_PROFILE_SCHEMA_ID = (
    "curvyzero_profile_only_batched_observation_stack4/v0"
)
SOURCE_STATE_BATCHED_OBSERVATION_PROFILE_ROLE = "profile_only_mock_collector_facade"
SOURCE_STATE_BATCHED_OBSERVATION_FUTURE_GPU_BACKEND = "future_batched_gpu_renderer"
SOURCE_STATE_BATCHED_OBSERVATION_STRAIGHT_ACTION_ID = 1
SOURCE_STATE_BATCHED_OBSERVATION_SELF_LUMA = 96
SOURCE_STATE_BATCHED_OBSERVATION_OTHER_LUMA = 128
SOURCE_STATE_BATCHED_OBSERVATION_TELEMETRY_FIELDS = (
    "pack_sec",
    "render_sec",
    "readback_sec",
    "stack_sec",
    "reset_sec",
    "final_obs_sec",
)
SOURCE_STATE_BATCHED_OBSERVATION_DRIFT_FIELDS = (
    "trail_render_mode",
    "bonus_render_mode",
    "controlled_player",
    "avatar_color",
    "visual_trail_write_cursor",
    "visual_trail_break_before",
    "terminal_final_observation",
    "reset_stack_policy",
)


def _empty_telemetry() -> dict[str, float]:
    return {field: 0.0 for field in SOURCE_STATE_BATCHED_OBSERVATION_TELEMETRY_FIELDS}


def _elapsed(started: float) -> float:
    return max(0.0, time.perf_counter() - started)


@dataclass(frozen=True, slots=True)
class SourceStateBatchedRenderRequest:
    """Boundary object a future GPU renderer should be able to consume."""

    state: Mapping[str, np.ndarray]
    row_indices: np.ndarray
    controlled_players: np.ndarray
    out: np.ndarray
    trail_render_mode: str = POLICY_TRAIL_RENDER_MODE
    bonus_render_mode: str = POLICY_BONUS_RENDER_MODE


@dataclass(frozen=True, slots=True)
class SourceStateBatchedRenderResult:
    frames: np.ndarray
    telemetry: dict[str, float]


class SourceStateBatchedObservationRenderer(Protocol):
    """Small renderer seam: pack source rows, render gray64, optionally read back."""

    backend_name: str

    def render(self, request: SourceStateBatchedRenderRequest) -> SourceStateBatchedRenderResult:
        ...


@dataclass(frozen=True, slots=True)
class SourceStateBatchedObservationStep:
    observation: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    info: dict[str, Any]
    final_observation: np.ndarray | None = None


class CpuOracleBatchedObservationRenderer:
    """CPU oracle implementation behind the future batched-renderer boundary."""

    backend_name = POLICY_OBSERVATION_BACKEND_CPU

    def __init__(self) -> None:
        self._rgb_work = np.empty(
            (
                SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
                SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
                3,
            ),
            dtype=np.uint8,
        )
        self._downsample_scratch = SourceStateGray64DownsampleScratch(
            SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
        )

    def render(self, request: SourceStateBatchedRenderRequest) -> SourceStateBatchedRenderResult:
        telemetry = _empty_telemetry()

        started = time.perf_counter()
        rows = np.asarray(request.row_indices, dtype=np.int64)
        controlled_players = np.asarray(request.controlled_players, dtype=np.int64)
        out = _validate_render_out(request.out, row_count=rows.shape[0])
        if controlled_players.shape != rows.shape:
            raise ValueError(
                "controlled_players must have the same shape as row_indices; "
                f"got {controlled_players.shape} and {rows.shape}"
            )
        telemetry["pack_sec"] = _elapsed(started)

        started = time.perf_counter()
        for output_row, source_row in enumerate(rows):
            controlled_player = int(controlled_players[output_row])
            render_source_state_canvas_gray64(
                request.state,
                row=int(source_row),
                out=out[output_row],
                rgb_out=self._rgb_work,
                downsample_scratch=self._downsample_scratch,
                player_rgb=source_state_controlled_player_palette(
                    request.state,
                    row=int(source_row),
                    controlled_player=controlled_player,
                ),
                trail_render_mode=request.trail_render_mode,
                bonus_render_mode=request.bonus_render_mode,
            )
        telemetry["render_sec"] = _elapsed(started)

        started = time.perf_counter()
        # CPU frames are already host-resident. The slot stays visible for GPU parity.
        telemetry["readback_sec"] = _elapsed(started)
        return SourceStateBatchedRenderResult(frames=out, telemetry=telemetry)


class SourceStateBatchedObservationProfileFacade:
    """Owns local env rows and emits stacked observations for profiling only."""

    def __init__(
        self,
        batch_size: int,
        *,
        player_count: int = 2,
        controlled_players: int | Sequence[int] = 0,
        seed: int = 0,
        max_ticks: int = 2_000,
        body_capacity: int = DEFAULT_BODY_CAPACITY,
        decision_source_frames: int = 1,
        source_physics_step_ms: float = SOURCE_PHYSICS_STEP_MS,
        death_mode: str = vector_runtime.DEATH_MODE_PROFILE_NO_DEATH,
        natural_bonus_spawn: bool = False,
        observation_backend: str = DEFAULT_POLICY_OBSERVATION_BACKEND,
        renderer: SourceStateBatchedObservationRenderer | None = None,
    ) -> None:
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.player_count = _positive_int(player_count, "player_count")
        if self.player_count < 2:
            raise ValueError("player_count must be at least 2")
        self.controlled_players = _controlled_players(
            controlled_players,
            batch_size=self.batch_size,
            player_count=self.player_count,
        )
        self.observation_backend = str(observation_backend)
        if self.observation_backend != POLICY_OBSERVATION_BACKEND_CPU:
            raise ValueError(
                "profile facade currently permits only cpu_oracle; "
                f"future boundary is {SOURCE_STATE_BATCHED_OBSERVATION_FUTURE_GPU_BACKEND!r}"
            )
        self.renderer = renderer if renderer is not None else CpuOracleBatchedObservationRenderer()
        if self.renderer.backend_name != POLICY_OBSERVATION_BACKEND_CPU:
            raise ValueError("custom renderers must keep backend_name='cpu_oracle' for this facade")

        self.env = VectorMultiplayerEnv(
            self.batch_size,
            player_count=self.player_count,
            seed=int(seed),
            decision_source_frames=int(decision_source_frames),
            source_physics_step_ms=float(source_physics_step_ms),
            max_ticks=int(max_ticks),
            body_capacity=int(body_capacity),
            natural_bonus_spawn=bool(natural_bonus_spawn),
            death_mode=str(death_mode),
        )
        self._seed = int(seed)
        self._has_reset = False
        self._row_indices = np.arange(self.batch_size, dtype=np.int64)
        self._raw_frames = np.zeros(
            (self.batch_size, *SOURCE_STATE_CANVAS_GRAY64_SHAPE),
            dtype=np.uint8,
        )
        self._stacks = np.zeros((self.batch_size, *POLICY_STACK_SHAPE), dtype=np.float32)
        self._last_telemetry = _empty_telemetry()

    @property
    def state(self) -> Mapping[str, np.ndarray]:
        return self.env.state

    @property
    def observation(self) -> np.ndarray:
        return self._stacks.copy()

    @property
    def last_telemetry(self) -> dict[str, float]:
        return dict(self._last_telemetry)

    def reset(self, seed: int | None = None) -> SourceStateBatchedObservationStep:
        telemetry = _empty_telemetry()
        started = time.perf_counter()
        reset_seed = self._seed if seed is None else int(seed)
        batch = self.env.reset(seed=reset_seed)
        self._seed = reset_seed
        self._has_reset = True
        self._stacks.fill(0.0)
        telemetry["reset_sec"] = _elapsed(started)

        render_result = self._render_all_rows()
        reset_sec = telemetry["reset_sec"]
        telemetry.update(render_result.telemetry)
        telemetry["reset_sec"] = reset_sec
        telemetry["stack_sec"] = self._push_frames(render_result.frames)
        self._last_telemetry = telemetry

        return SourceStateBatchedObservationStep(
            observation=self.observation,
            reward=batch.reward.copy(),
            done=batch.done.copy(),
            final_observation=None,
            info=self._info(
                event="reset",
                telemetry=telemetry,
                vector_info=batch.info,
            ),
        )

    def step(
        self,
        actions: int | Sequence[int] | np.ndarray,
        *,
        other_actions: int | Sequence[int] | np.ndarray | None = None,
    ) -> SourceStateBatchedObservationStep:
        if not self._has_reset:
            raise RuntimeError("reset must be called before step")
        telemetry = _empty_telemetry()
        joint_actions = self._joint_actions(actions, other_actions=other_actions)
        batch = self.env.step(joint_actions)

        render_result = self._render_all_rows()
        telemetry.update(render_result.telemetry)
        telemetry["stack_sec"] = self._push_frames(render_result.frames)

        final_observation = None
        if bool(batch.done.any()):
            started = time.perf_counter()
            final_observation = np.zeros_like(self._stacks)
            done_rows = np.flatnonzero(batch.done).astype(np.int64)
            final_observation[done_rows] = self._stacks[done_rows]
            telemetry["final_obs_sec"] = _elapsed(started)

        self._last_telemetry = telemetry
        return SourceStateBatchedObservationStep(
            observation=self.observation,
            reward=batch.reward.copy(),
            done=batch.done.copy(),
            final_observation=final_observation,
            info=self._info(
                event="step",
                telemetry=telemetry,
                vector_info=batch.info,
                joint_actions=joint_actions,
            ),
        )

    def contract(self) -> dict[str, Any]:
        return source_state_batched_observation_profile_contract(
            batch_size=self.batch_size,
            player_count=self.player_count,
            controlled_players=self.controlled_players,
            backend=self.observation_backend,
        )

    def render_boundary(self) -> dict[str, Any]:
        return self.contract()["future_gpu_render_boundary"]

    def close(self) -> None:
        return None

    def _render_all_rows(self) -> SourceStateBatchedRenderResult:
        request = SourceStateBatchedRenderRequest(
            state=self.env.state,
            row_indices=self._row_indices,
            controlled_players=self.controlled_players,
            out=self._raw_frames,
            trail_render_mode=POLICY_TRAIL_RENDER_MODE,
            bonus_render_mode=POLICY_BONUS_RENDER_MODE,
        )
        return self.renderer.render(request)

    def _push_frames(self, frames: np.ndarray) -> float:
        started = time.perf_counter()
        raw = _validate_render_out(frames, row_count=self.batch_size)
        self._stacks[:, :-1] = self._stacks[:, 1:]
        np.multiply(
            raw[:, 0],
            np.float32(1.0 / 255.0),
            out=self._stacks[:, -1],
            casting="unsafe",
        )
        return _elapsed(started)

    def _joint_actions(
        self,
        actions: int | Sequence[int] | np.ndarray,
        *,
        other_actions: int | Sequence[int] | np.ndarray | None,
    ) -> np.ndarray:
        controlled = _action_vector(actions, batch_size=self.batch_size, name="actions")
        other = (
            np.full(
                self.batch_size,
                SOURCE_STATE_BATCHED_OBSERVATION_STRAIGHT_ACTION_ID,
                dtype=np.int16,
            )
            if other_actions is None
            else _action_vector(other_actions, batch_size=self.batch_size, name="other_actions")
        )
        joint = np.full(
            (self.batch_size, self.player_count),
            SOURCE_STATE_BATCHED_OBSERVATION_STRAIGHT_ACTION_ID,
            dtype=np.int16,
        )
        for row, controlled_player in enumerate(self.controlled_players):
            joint[row, :] = other[row]
            joint[row, int(controlled_player)] = controlled[row]
        return joint

    def _info(
        self,
        *,
        event: str,
        telemetry: Mapping[str, float],
        vector_info: Mapping[str, Any],
        joint_actions: np.ndarray | None = None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "event": event,
            "profile_only": True,
            "live_training_run": False,
            "stock_lightzero_integrated": False,
            "trainer_defaults_changed": False,
            "observation_backend": self.observation_backend,
            "telemetry": dict(telemetry),
            "contract": self.contract(),
            "vector_info": dict(vector_info),
        }
        if joint_actions is not None:
            info["joint_actions"] = joint_actions.copy()
        return info


def source_state_controlled_player_palette(
    state: Mapping[str, np.ndarray],
    *,
    row: int,
    controlled_player: int,
) -> tuple[tuple[int, int, int], ...]:
    player_count = int(np.asarray(state["pos"]).shape[1])
    player = int(controlled_player)
    if player < 0 or player >= player_count:
        raise ValueError("controlled_player must be in [0, player_count)")
    color_indices = np.arange(player_count, dtype=np.int64)
    if "avatar_color" in state:
        avatar_color = np.asarray(state["avatar_color"])
        if avatar_color.ndim >= 2:
            color_indices = np.asarray(avatar_color[int(row), :player_count], dtype=np.int64)
    if bool((color_indices < 0).any()):
        raise ValueError("avatar_color indices must be non-negative")
    max_color_index = int(color_indices.max()) if color_indices.size else player_count - 1
    self_rgb = (
        SOURCE_STATE_BATCHED_OBSERVATION_SELF_LUMA,
        SOURCE_STATE_BATCHED_OBSERVATION_SELF_LUMA,
        SOURCE_STATE_BATCHED_OBSERVATION_SELF_LUMA,
    )
    other_rgb = (
        SOURCE_STATE_BATCHED_OBSERVATION_OTHER_LUMA,
        SOURCE_STATE_BATCHED_OBSERVATION_OTHER_LUMA,
        SOURCE_STATE_BATCHED_OBSERVATION_OTHER_LUMA,
    )
    palette = [other_rgb for _ in range(max(player_count, max_color_index + 1))]
    palette[int(color_indices[player])] = self_rgb
    return tuple(palette)


def source_state_batched_observation_profile_contract(
    *,
    batch_size: int,
    player_count: int,
    controlled_players: Sequence[int] | np.ndarray,
    backend: str,
) -> dict[str, Any]:
    controlled = np.asarray(controlled_players, dtype=np.int64)
    return {
        "schema_id": SOURCE_STATE_BATCHED_OBSERVATION_PROFILE_SCHEMA_ID,
        "impl_id": SOURCE_STATE_BATCHED_OBSERVATION_PROFILE_IMPL_ID,
        "role": SOURCE_STATE_BATCHED_OBSERVATION_PROFILE_ROLE,
        "profile_only": True,
        "calls_stock_trainer": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "stock_lightzero_integrated": False,
        "trainer_defaults_changed": False,
        "batch_size": int(batch_size),
        "player_count": int(player_count),
        "controlled_players": controlled.astype(int).tolist(),
        "stack_shape": list(POLICY_STACK_SHAPE),
        "stack_depth": int(POLICY_FRAME_STACK_DEPTH),
        "single_frame_shape": list(SOURCE_STATE_CANVAS_GRAY64_SHAPE),
        "stack_dtype": "float32",
        "raw_frame_dtype": "uint8",
        "trail_render_mode": POLICY_TRAIL_RENDER_MODE,
        "bonus_render_mode": POLICY_BONUS_RENDER_MODE,
        "current_backend": str(backend),
        "current_backend_expected": POLICY_OBSERVATION_BACKEND_CPU,
        "production_default_backend": DEFAULT_POLICY_OBSERVATION_BACKEND,
        "lab_scalar_gpu_backend": POLICY_OBSERVATION_BACKEND_GPU,
        "telemetry_fields": list(SOURCE_STATE_BATCHED_OBSERVATION_TELEMETRY_FIELDS),
        "drift_sensitive_fields": list(SOURCE_STATE_BATCHED_OBSERVATION_DRIFT_FIELDS),
        "future_gpu_render_boundary": {
            "backend_name": SOURCE_STATE_BATCHED_OBSERVATION_FUTURE_GPU_BACKEND,
            "input_state": "VectorMultiplayerEnv.state arrays plus row_indices",
            "input_control": "controlled_players per output row",
            "input_surface": "browser_lines + simple_symbols",
            "output": "uint8 [B, 1, 64, 64] host-visible gray frames",
            "readback_slot": "explicit telemetry slot even when current CPU path is no-op",
            "not_implemented_here": True,
        },
    }


def _positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


def _controlled_players(
    value: int | Sequence[int],
    *,
    batch_size: int,
    player_count: int,
) -> np.ndarray:
    if np.asarray(value).ndim == 0:
        players = np.full(batch_size, int(np.asarray(value).item()), dtype=np.int64)
    else:
        players = np.asarray(value, dtype=np.int64)
        if players.shape != (batch_size,):
            raise ValueError(
                f"controlled_players must be scalar or shape ({batch_size},), got {players.shape}"
            )
    if bool(((players < 0) | (players >= player_count)).any()):
        raise ValueError("controlled_players entries must be in [0, player_count)")
    return players


def _action_vector(
    value: int | Sequence[int] | np.ndarray,
    *,
    batch_size: int,
    name: str,
) -> np.ndarray:
    if np.asarray(value).ndim == 0:
        actions = np.full(batch_size, int(np.asarray(value).item()), dtype=np.int16)
    else:
        actions = np.asarray(value, dtype=np.int16)
        if actions.shape != (batch_size,):
            raise ValueError(f"{name} must be scalar or shape ({batch_size},), got {actions.shape}")
    if bool(((actions < 0) | (actions >= ACTION_COUNT)).any()):
        raise ValueError(f"{name} entries must be in [0, {ACTION_COUNT})")
    return actions


def _validate_render_out(out: np.ndarray, *, row_count: int) -> np.ndarray:
    array = np.asarray(out)
    expected = (int(row_count), *SOURCE_STATE_CANVAS_GRAY64_SHAPE)
    if array.shape != expected:
        raise ValueError(f"render out must have shape {expected}, got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"render out dtype must be uint8, got {array.dtype}")
    return array


__all__ = [
    "CpuOracleBatchedObservationRenderer",
    "SOURCE_STATE_BATCHED_OBSERVATION_DRIFT_FIELDS",
    "SOURCE_STATE_BATCHED_OBSERVATION_FUTURE_GPU_BACKEND",
    "SOURCE_STATE_BATCHED_OBSERVATION_PROFILE_IMPL_ID",
    "SOURCE_STATE_BATCHED_OBSERVATION_PROFILE_SCHEMA_ID",
    "SOURCE_STATE_BATCHED_OBSERVATION_STRAIGHT_ACTION_ID",
    "SOURCE_STATE_BATCHED_OBSERVATION_TELEMETRY_FIELDS",
    "SourceStateBatchedObservationProfileFacade",
    "SourceStateBatchedObservationRenderer",
    "SourceStateBatchedObservationStep",
    "SourceStateBatchedRenderRequest",
    "SourceStateBatchedRenderResult",
    "source_state_batched_observation_profile_contract",
    "source_state_controlled_player_palette",
]
