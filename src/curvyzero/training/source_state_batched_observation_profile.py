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
SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND = "jax_gpu_batched_profile"
SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_CONTROLLED_ROWS = "controlled_rows"
SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_BOTH_PLAYERS = "both_players"
SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_MODES = (
    SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_CONTROLLED_ROWS,
    SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_BOTH_PLAYERS,
)
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
class SourceStateBatchedRenderStateRowOverlay:
    rows: np.ndarray
    state: Mapping[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class SourceStateBatchedRenderRequest:
    """Boundary object a future GPU renderer should be able to consume."""

    state: Mapping[str, np.ndarray]
    row_indices: np.ndarray
    controlled_players: np.ndarray
    out: np.ndarray
    trail_render_mode: str = POLICY_TRAIL_RENDER_MODE
    bonus_render_mode: str = POLICY_BONUS_RENDER_MODE
    device_only: bool = False
    synchronize_device: bool = True
    state_row_overlays: Sequence[SourceStateBatchedRenderStateRowOverlay] = ()


@dataclass(frozen=True, slots=True)
class SourceStateBatchedRenderResult:
    frames: np.ndarray
    telemetry: dict[str, float]
    device_frames: Any | None = None


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
        if bool(request.device_only):
            raise ValueError("CPU oracle renderer cannot satisfy device_only=True")
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
        state = source_state_render_state_with_row_overlays(request.state, request.state_row_overlays)
        for output_row, source_row in enumerate(rows):
            controlled_player = int(controlled_players[output_row])
            render_source_state_canvas_gray64(
                state,
                row=int(source_row),
                out=out[output_row],
                rgb_out=self._rgb_work,
                downsample_scratch=self._downsample_scratch,
                player_rgb=source_state_controlled_player_palette(
                    state,
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
        observation_backend: str = POLICY_OBSERVATION_BACKEND_CPU,
        player_view_mode: str = SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_CONTROLLED_ROWS,
        renderer: SourceStateBatchedObservationRenderer | None = None,
    ) -> None:
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.player_count = _positive_int(player_count, "player_count")
        if self.player_count < 2:
            raise ValueError("player_count must be at least 2")
        self.player_view_mode = _player_view_mode(player_view_mode)
        self.controlled_players = _controlled_players(
            controlled_players,
            batch_size=self.batch_size,
            player_count=self.player_count,
        )
        self.observation_backend = str(observation_backend)
        self.renderer = self._resolve_renderer(renderer)

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
        self._row_indices = self._profile_render_row_indices()
        self._render_controlled_players = self._profile_render_controlled_players()
        self._raw_frames = np.zeros(
            (self._row_indices.shape[0], *SOURCE_STATE_CANVAS_GRAY64_SHAPE),
            dtype=np.uint8,
        )
        if self.player_view_mode == SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_BOTH_PLAYERS:
            stack_shape = (self.batch_size, self.player_count, *POLICY_STACK_SHAPE)
        else:
            stack_shape = (self.batch_size, *POLICY_STACK_SHAPE)
        self._stacks = np.zeros(stack_shape, dtype=np.float32)
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
            # TODO: model autoreset ordering when the profile grows an env-manager row reuse path.

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
                final_observation=final_observation,
                done=batch.done,
            ),
        )

    def contract(self) -> dict[str, Any]:
        return source_state_batched_observation_profile_contract(
            batch_size=self.batch_size,
            player_count=self.player_count,
            controlled_players=self.controlled_players,
            backend=self.observation_backend,
            player_view_mode=self.player_view_mode,
        )

    def render_boundary(self) -> dict[str, Any]:
        return self.contract()["future_gpu_render_boundary"]

    def close(self) -> None:
        return None

    def _resolve_renderer(
        self,
        renderer: SourceStateBatchedObservationRenderer | None,
    ) -> SourceStateBatchedObservationRenderer:
        if self.observation_backend == POLICY_OBSERVATION_BACKEND_CPU:
            resolved = renderer if renderer is not None else CpuOracleBatchedObservationRenderer()
            if resolved.backend_name != POLICY_OBSERVATION_BACKEND_CPU:
                raise ValueError(
                    "cpu_oracle observation_backend requires renderer.backend_name='cpu_oracle'"
                )
            return resolved
        if self.observation_backend == SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND:
            if renderer is None:
                raise ValueError(
                    "jax_gpu_batched_profile requires an explicit renderer; "
                    "no hidden CPU fallback is allowed"
                )
            if renderer.backend_name != SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND:
                raise ValueError(
                    "jax_gpu_batched_profile requires renderer.backend_name="
                    f"{SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND!r}"
                )
            return renderer
        raise ValueError(
            "profile facade currently supports only cpu_oracle or explicit "
            f"{SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND!r}; "
            f"got {self.observation_backend!r}. Scalar {POLICY_OBSERVATION_BACKEND_GPU!r} "
            "is not the batched speed path."
        )

    def _render_all_rows(self) -> SourceStateBatchedRenderResult:
        request = SourceStateBatchedRenderRequest(
            state=self.env.state,
            row_indices=self._row_indices,
            controlled_players=self._render_controlled_players,
            out=self._raw_frames,
            trail_render_mode=POLICY_TRAIL_RENDER_MODE,
            bonus_render_mode=POLICY_BONUS_RENDER_MODE,
        )
        return self.renderer.render(request)

    def _push_frames(self, frames: np.ndarray) -> float:
        started = time.perf_counter()
        raw = _validate_render_out(frames, row_count=self._row_indices.shape[0])
        if self.player_view_mode == SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_BOTH_PLAYERS:
            raw_view = raw.reshape(
                self.batch_size,
                self.player_count,
                *SOURCE_STATE_CANVAS_GRAY64_SHAPE,
            )
            self._stacks[:, :, :-1] = self._stacks[:, :, 1:]
            np.multiply(
                raw_view[:, :, 0],
                np.float32(1.0 / 255.0),
                out=self._stacks[:, :, -1],
                casting="unsafe",
            )
        else:
            self._stacks[:, :-1] = self._stacks[:, 1:]
            np.multiply(
                raw[:, 0],
                np.float32(1.0 / 255.0),
                out=self._stacks[:, -1],
                casting="unsafe",
            )
        return _elapsed(started)

    def _profile_render_row_indices(self) -> np.ndarray:
        rows = np.arange(self.batch_size, dtype=np.int64)
        if self.player_view_mode == SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_BOTH_PLAYERS:
            return np.repeat(rows, self.player_count)
        return rows

    def _profile_render_controlled_players(self) -> np.ndarray:
        if self.player_view_mode == SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_BOTH_PLAYERS:
            players = np.arange(self.player_count, dtype=np.int64)
            return np.tile(players, self.batch_size)
        return self.controlled_players.copy()

    def _profile_render_row_major_pairs(self) -> list[list[int]]:
        return [
            [int(row), int(player)]
            for row, player in zip(
                self._row_indices,
                self._render_controlled_players,
                strict=True,
            )
        ]

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
        final_observation: np.ndarray | None = None,
        done: np.ndarray | None = None,
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
        if final_observation is not None:
            if done is None:
                raise ValueError("done is required when final_observation is present")
            done_mask = np.asarray(done, dtype=bool).copy()
            final_rows = np.flatnonzero(done_mask).astype(np.int32)
            info["final_observation"] = final_observation.copy()
            info["final_observation_rows"] = final_rows
            info["final_observation_row_mask"] = done_mask
            info["final_observation_policy"] = {
                "schema_id": "profile_only_terminal_stack_before_reset/v0",
                "array": "final_observation",
                "row_mask": done_mask.copy(),
                "rows": final_rows.copy(),
                "source": "source_state_batched_observation_profile.stacked_observation",
                "observation_shape": list(final_observation.shape),
                "player_view_mode": self.player_view_mode,
                "autoreset": "not modeled",
            }
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
    player_view_mode: str = SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_CONTROLLED_ROWS,
) -> dict[str, Any]:
    controlled = np.asarray(controlled_players, dtype=np.int64)
    view_mode = _player_view_mode(player_view_mode)
    if view_mode == SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_BOTH_PLAYERS:
        observation_shape = [int(batch_size), int(player_count), *POLICY_STACK_SHAPE]
        render_row_count = int(batch_size) * int(player_count)
        render_output_shape = [
            int(batch_size) * int(player_count),
            *SOURCE_STATE_CANVAS_GRAY64_SHAPE,
        ]
    else:
        observation_shape = [int(batch_size), *POLICY_STACK_SHAPE]
        render_row_count = int(batch_size)
        render_output_shape = [int(batch_size), *SOURCE_STATE_CANVAS_GRAY64_SHAPE]
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
        "action_controlled_players": controlled.astype(int).tolist(),
        "controlled_players": controlled.astype(int).tolist(),
        "player_view_mode": view_mode,
        "player_view_modes": list(SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_MODES),
        "observation_shape": observation_shape,
        "stack_shape": list(POLICY_STACK_SHAPE),
        "stack_depth": int(POLICY_FRAME_STACK_DEPTH),
        "single_frame_shape": list(SOURCE_STATE_CANVAS_GRAY64_SHAPE),
        "stack_dtype": "float32",
        "raw_frame_dtype": "uint8",
        "trail_render_mode": POLICY_TRAIL_RENDER_MODE,
        "bonus_render_mode": POLICY_BONUS_RENDER_MODE,
        "current_backend": str(backend),
        "current_backend_expected": POLICY_OBSERVATION_BACKEND_CPU,
        "profile_gpu_candidate_backend": SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND,
        "production_default_backend": DEFAULT_POLICY_OBSERVATION_BACKEND,
        "lab_scalar_gpu_backend": POLICY_OBSERVATION_BACKEND_GPU,
        "telemetry_fields": list(SOURCE_STATE_BATCHED_OBSERVATION_TELEMETRY_FIELDS),
        "drift_sensitive_fields": list(SOURCE_STATE_BATCHED_OBSERVATION_DRIFT_FIELDS),
        "future_gpu_render_boundary": {
            "backend_name": SOURCE_STATE_BATCHED_OBSERVATION_FUTURE_GPU_BACKEND,
            "input_state": "VectorMultiplayerEnv.state arrays plus row_indices",
            "input_control": "render_controlled_players per output row",
            "player_view_mode": view_mode,
            "render_row_count": render_row_count,
            "render_row_indices": (
                "row-major repeated env rows for both_players; one row per env for controlled_rows"
            ),
            "render_controlled_players": (
                "row-major [0..P-1] per env row for both_players; action_controlled_players "
                "for controlled_rows"
            ),
            "render_order": "row-major [(row0,p0), (row0,p1), ...] for both_players",
            "autoreset": "TODO: not modeled; terminal final_observation is captured before any reset",
            "input_surface": "browser_lines + simple_symbols",
            "output_shape": render_output_shape,
            "output": "host-visible uint8 gray frames matching output_shape",
            "readback_slot": "explicit telemetry slot even when current CPU path is no-op",
            "not_implemented_here": True,
        },
    }


def _positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


def _player_view_mode(value: str) -> str:
    mode = str(value)
    if mode not in SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_MODES:
        valid = ", ".join(SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_MODES)
        raise ValueError(f"player_view_mode must be one of: {valid}; got {mode!r}")
    return mode


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


def source_state_render_state_with_row_overlays(
    state: Mapping[str, np.ndarray],
    overlays: Sequence[SourceStateBatchedRenderStateRowOverlay] = (),
    *,
    batch_size: int | None = None,
) -> dict[str, np.ndarray]:
    result = {str(key): np.asarray(value) for key, value in state.items()}
    overlay_rows_seen: set[int] = set()
    copied_keys: set[str] = set()
    for overlay in overlays:
        rows_raw = np.asarray(overlay.rows)
        if rows_raw.ndim != 1:
            raise ValueError(f"state row overlay rows must be rank 1, got {rows_raw.shape}")
        if not np.issubdtype(rows_raw.dtype, np.integer):
            raise ValueError("state row overlay rows must be integers")
        rows = rows_raw.astype(np.int64, copy=False)
        if bool((rows < 0).any()):
            raise ValueError("state row overlay rows must be nonnegative")
        if batch_size is not None and bool((rows >= int(batch_size)).any()):
            raise ValueError(f"state row overlay rows must be in [0, {int(batch_size)})")
        if np.unique(rows).shape[0] != rows.shape[0]:
            raise ValueError("state row overlay rows must be duplicate-free")
        duplicate_rows = {int(row) for row in rows if int(row) in overlay_rows_seen}
        if duplicate_rows:
            raise ValueError("state row overlay rows must be duplicate-free")
        overlay_rows_seen.update(int(row) for row in rows)
        overlay_state = {str(key): np.asarray(value) for key, value in overlay.state.items()}
        for key, overlay_value in overlay_state.items():
            if key not in result:
                raise ValueError(f"state row overlay key {key!r} is missing from base state")
            base_value = result[key]
            if base_value.ndim < 1:
                raise ValueError(f"state row overlay key {key!r} does not have a row dimension")
            if batch_size is None:
                key_batch_size = int(base_value.shape[0])
            else:
                key_batch_size = int(batch_size)
                if int(base_value.shape[0]) < key_batch_size:
                    raise ValueError(
                        f"state row overlay base key {key!r} has too few rows: "
                        f"{base_value.shape[0]} < {key_batch_size}"
                    )
            if bool((rows >= key_batch_size).any()):
                raise ValueError(
                    f"state row overlay rows for key {key!r} must be in [0, {key_batch_size})"
                )
            expected_shape = (int(rows.shape[0]), *base_value.shape[1:])
            if overlay_value.shape != expected_shape:
                raise ValueError(
                    f"state row overlay key {key!r} must have shape {expected_shape}, "
                    f"got {overlay_value.shape}"
                )
            if key not in copied_keys:
                result[key] = np.asarray(base_value).copy()
                copied_keys.add(key)
            result[key][rows] = overlay_value
    return result


__all__ = [
    "CpuOracleBatchedObservationRenderer",
    "SOURCE_STATE_BATCHED_OBSERVATION_DRIFT_FIELDS",
    "SOURCE_STATE_BATCHED_OBSERVATION_FUTURE_GPU_BACKEND",
    "SOURCE_STATE_BATCHED_OBSERVATION_GPU_CANDIDATE_BACKEND",
    "SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_BOTH_PLAYERS",
    "SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_CONTROLLED_ROWS",
    "SOURCE_STATE_BATCHED_OBSERVATION_PLAYER_VIEW_MODES",
    "SOURCE_STATE_BATCHED_OBSERVATION_PROFILE_IMPL_ID",
    "SOURCE_STATE_BATCHED_OBSERVATION_PROFILE_SCHEMA_ID",
    "SOURCE_STATE_BATCHED_OBSERVATION_STRAIGHT_ACTION_ID",
    "SOURCE_STATE_BATCHED_OBSERVATION_TELEMETRY_FIELDS",
    "SourceStateBatchedObservationProfileFacade",
    "SourceStateBatchedObservationRenderer",
    "SourceStateBatchedObservationStep",
    "SourceStateBatchedRenderRequest",
    "SourceStateBatchedRenderResult",
    "SourceStateBatchedRenderStateRowOverlay",
    "source_state_batched_observation_profile_contract",
    "source_state_controlled_player_palette",
    "source_state_render_state_with_row_overlays",
]
